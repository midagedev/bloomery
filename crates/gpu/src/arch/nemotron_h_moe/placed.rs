//! A Bolt load whose routed experts the card cannot all hold: the plan's card
//! segment of each layer's stacks resident, the rest served by the common
//! host tier ([`crate::host`]) in relu². A layer the plan holds whole runs the
//! whole-card layer ([`super::moe::moe_layer`]); a layer with a host share
//! runs this module's three parts around the tier's step port, the module doc
//! of [`super::moe`] in the host leg's order:
//!
//! - **front**: the norm and its q8_1, the router, the latent down into the
//!   boundary's activation (the host reads the latent, whose width is the
//!   boundary's), the handoff of the ids, weights and latent into the page
//!   with each slot's place in the card's stacks, and the go;
//! - **shadow**, under the host's work: the shared expert's up, then the
//!   card's slots (the latent's q8_1, up, squared ReLU, down over the places
//!   the handoff wrote; a [`HOST`](crate::hybrid::HOST) slot's warps return
//!   before their first load), then the shared expert's down;
//! - **back**: the wait, the card's slots' weighted sum (`ds41_ffn_card_acc_9`,
//!   the places below the layer's card count) plus the host's sum in the latent,
//!   the latent up, the shared expert, the residual.
//!
//! The prompt runs as passes of up to [`PLACED_ROWS`] rows through the host
//! tier's batch port ([`super::prefill`]): the same three parts over a pass,
//! the host serving each hosted layer's slots of all the pass's rows as one
//! union call under the card's own slots, which read their places from the
//! slot map ([`PlacedSide::enqueue_places`]) and, past eight rows, through
//! each layer's map to the card's stacks ([`PlacedSide::map_of`]).
//!
//! The card's share of a layer and the host's sum to the whole-card layer's
//! weighted sum up to rounding: the host quantizes the latent with its own
//! kernel and sums its slots in its own order. Bolt's own parts are what its
//! architecture forces: ungated experts (one up and one down stack, relu²),
//! the latent the experts run in, and the interleaved MoE layers (the slot
//! map's rows cover every layer, only the MoE layers' hold experts).

use super::body::Layer;
use super::moe::{MoeKernels, MoeLayer, MoeScratch, MoeShape, one_expert, stack};
use super::router::TOP_K;
use crate::arch::hybrid::{self, Acts, Cx, attn_layer, mamba_layer, residual};
use crate::card_slots::{CardAcc, CardSlotKernels, ColPlaces};
use crate::fault::FaultSink;
use crate::flash_gqa::GROUP_16;
use crate::host::StepLeg;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, SlotMap};
use crate::kquant::SelDown;
use crate::tensor::DeviceTensor;
use crate::{Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::{GgmlType, Split};
use model::arch::nemotron_h_moe::host::routed_layers;
use model::arch::nemotron_h_moe::hparams::Hparams;
use model::moe::UNION_MAX_COLS;
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const WHAT: &str = "nemotron_h_moe placed load";

/// The most rows a placed prompt pass takes: the host union's most columns,
/// so each host expert is read once a pass.
pub(super) const PLACED_ROWS: usize = UNION_MAX_COLS;

/// The slot map the plan makes over every layer of the chain, one row of
/// `stride` places a layer (the widest layer's expert count; a narrower
/// layer's row ends in places no id reaches). A plan of other than one card
/// holding every layer, or with an expert tier, is refused by name.
pub(super) fn plan_map(plan: &Plan<'_>, hp: &Hparams) -> Result<(SlotMap, usize), GpuError> {
    let n = hp.n_layer;
    let spec = plan
        .machine
        .cards
        .first()
        .ok_or(GpuError::shape(WHAT, "a plan of no card"))?;
    if plan.machine.cards.len() != 1 || !plan.machine.tiers.is_empty() || spec.layers != (0..n) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a plan of {} cards and {} expert tiers, card 0 running layers {:?}; the body \
                 runs every layer (0..{n}) on one card with no expert tier",
                plan.machine.cards.len(),
                plan.machine.tiers.len(),
                spec.layers
            ),
        ));
    }
    let stride = hp
        .experts
        .iter()
        .flatten()
        .map(|x| x.n_expert)
        .max()
        .ok_or(GpuError::shape(WHAT, "a file with no routed layer"))?;
    Ok((SlotMap::of_plan(plan, 0, None, 0..n, stride)?, stride))
}

/// The card's side of a placed load: which layers the host serves a share
/// of, the slot map's card copy, the handoff and sum launches and the
/// buffers of one layer's host leg.
pub(super) struct PlacedSide {
    hosted: Vec<bool>,
    /// Each hosted layer's expert to place in the card's stacks, a layer's own
    /// `n_expert` words: the map the prompt pass's GEMM route is built from
    /// (`None` where the layer is not hosted or holds no expert).
    maps: Vec<Option<DeviceBuffer<u32>>>,
    slots: DeviceTensor<u32>,
    /// The places of a row of the card copy: the widest layer's expert count.
    stride: usize,
    handoff: HandoffKernels,
    acc_k: CardSlotKernels,
    /// The routed slots' places in the card's stacks, or `HOST`.
    sel: DeviceBuffer<u32>,
    /// The card slots' weighted sum, then that and the host's sum, `latent`.
    acc: DeviceBuffer<f32>,
    sum: DeviceBuffer<f32>,
}

impl PlacedSide {
    /// Whether the host serves a share of layer `l`.
    pub(super) fn hosted(&self, l: usize) -> bool {
        self.hosted[l]
    }

    /// The card's slot sums ([`CardSlotKernels`]).
    pub(super) fn acc(&self) -> &CardSlotKernels {
        &self.acc_k
    }

    /// Layer `l`'s map of the experts to their places in the card's stacks, for
    /// a layer that holds an expert on the card; refused by name for any other.
    pub(super) fn map_of(&self, l: usize) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.maps
            .get(l)
            .and_then(Option::as_ref)
            .ok_or(GpuError::state(
                WHAT,
                "a map of the card's experts for a hosted layer that holds some",
            ))
    }

    /// Layer `l`'s places for the `m` routed rows `ids` (`TOP_K` slots a
    /// row, `n_expert` experts in the layer) into `sel`: each slot's place in
    /// the card's stacks, or [`HOST`](crate::hybrid::HOST) for the host's.
    pub(super) fn enqueue_places(
        &self,
        stream: &CudaStream,
        (l, n_expert): (usize, usize),
        (ids, sel, m): (&DeviceBuffer<u32>, &mut DeviceBuffer<u32>, usize),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.acc_k.enqueue_places_cols(
            stream,
            &ColPlaces {
                ids,
                ids_pitch: TOP_K,
                used: TOP_K,
                sel_pitch: TOP_K,
                m,
                map: self.slots.buf(),
                row_off: l * self.stride,
                n_expert,
            },
            fault,
            sel,
        )
    }
}

/// The placed side of a load: the host tier over the layers with a host
/// share, and what the walk reads beside it.
pub struct Placed {
    pub(super) hybrid: Hybrid<HostRun>,
    pub(super) side: PlacedSide,
}

impl Placed {
    /// The placed side over `layers` (the chain as loaded) of a plan whose
    /// slot map is `map` at `stride` places a row: the host run over the
    /// layers whose card count is under their experts, the boundary at the
    /// latent's width, the tier watching the fault word and holding the host
    /// set `set`, its batch port and union slabs for prompt passes of up to
    /// `rows` rows. Refused by name: a layer whose card count the map and the
    /// layer disagree on. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        hp: &Hparams,
        layers: &[Layer],
        (map, stride): (SlotMap, usize),
        (open, rows): (PlacedOpen<'_>, usize),
    ) -> Result<Placed, GpuError> {
        let n = layers.len();
        let mut hosted = vec![false; n];
        for (l, layer) in layers.iter().enumerate() {
            let Layer::Moe(m) = layer else { continue };
            let on_card = map.on_card(l)?;
            if on_card != m.n_card {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: the slot map puts {on_card} experts on the card, the layer \
                         loaded {}",
                        m.n_card
                    ),
                ));
            }
            let experts = hp.experts[l]
                .ok_or(GpuError::state(WHAT, "experts read for a MoE layer"))?
                .n_expert;
            hosted[l] = on_card < experts;
        }
        let set: Vec<usize> = (0..n).filter(|&l| hosted[l]).collect();
        let stream = gpu.stream();
        let stage = map.stage_view();
        let slots = DeviceTensor::upload(stream, &stage, n, stride)?;
        // A hosted layer that holds an expert on the card: its row of the
        // card copy, the layer's own `n_expert` places.
        let maps = layers
            .iter()
            .enumerate()
            .map(|(l, layer)| match layer {
                Layer::Moe(m) if hosted[l] && m.n_card > 0 => {
                    let n_expert = hp.experts[l].map_or(0, |x| x.n_expert);
                    let row = stage
                        .get(l * stride..l * stride + n_expert)
                        .ok_or(GpuError::state(WHAT, "a row of the slot map per layer"))?;
                    Ok(Some(DeviceBuffer::from_host(stream, row)?))
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let boundary = Boundary::new(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: hp.latent,
                n_used: TOP_K,
            },
        )?;
        let mut experts = crate::model::host_run(
            gpu.context(),
            open.plan,
            &open.file,
            set.iter().copied(),
            open.host.r8,
            open.set.arena(),
            HostWidths {
                embd: hp.latent,
                ff: hp.experts.iter().flatten().map(|x| x.ff).max().unwrap_or(0),
                n_used: TOP_K,
            },
            |src| routed_layers(src, hp, &set),
        )?;
        experts.prepare_union(rows)?;
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(open.set);
        hybrid.prepare_batch(gpu.context(), rows)?;
        let z = |len: usize| DeviceBuffer::zeroed(stream, len.max(1));
        Ok(Placed {
            hybrid,
            side: PlacedSide {
                hosted,
                maps,
                slots,
                stride,
                handoff: HandoffKernels::load(gpu.context())?,
                acc_k: CardSlotKernels::load(gpu.context())?,
                sel: DeviceBuffer::zeroed(stream, TOP_K)?,
                acc: z(hp.latent)?,
                sum: z(hp.latent)?,
            },
        })
    }

    /// Device bytes the placed side holds: the slot map's card copy and the
    /// layers' maps of it, the places and the sums, and the boundary.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let s = &self.side;
        s.slots.buf().num_bytes()
            + s.maps
                .iter()
                .flatten()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + s.sel.num_bytes()
            + s.acc.num_bytes()
            + s.sum.num_bytes()
            + self.hybrid.boundary().device_bytes()
    }

    /// The host tier ([`crate::host::HostTier::stats`] and the rest).
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    /// The layers the host serves a share of.
    #[must_use]
    pub fn hosted_layers(&self) -> Vec<usize> {
        (0..self.side.hosted.len())
            .filter(|&l| self.side.hosted[l])
            .collect()
    }

    /// The host tier's settling ([`crate::host::HostTier::settle`]).
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.settle(stream)
    }
}

/// What [`Placed::new`] builds a load's placed side from.
pub(super) struct PlacedOpen<'a> {
    /// The plan the load's weights were uploaded by.
    pub(super) plan: &'a Plan<'a>,
    /// The file, shared with the host tier.
    pub(super) file: Arc<Split>,
    pub(super) host: HostCfg,
    /// The plan's host set as the load left it.
    pub(super) set: HostResidency,
}

impl TierBody for Placed {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// The load runs no residency machine: the plan's slot map stays.
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        None
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
    }
}

/// Layer `l`'s front on a hosted layer: the module doc's first part.
fn front(
    cx: &Cx<'_>,
    (k, sh): (&MoeKernels, MoeShape),
    side: &mut PlacedSide,
    (l, m): (usize, &mut MoeLayer),
    (t, acts): (&mut hybrid::Scratch, &mut Acts),
    (src, leg): (&DeviceBuffer<f32>, &mut StepLeg<'_, HostRun>),
) -> Result<(), GpuError> {
    let (gpu, what) = (cx.gpu, cx.what);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    gpu.fused().enqueue_norm_quant(
        stream,
        src,
        cx.f32(&m.norm)?,
        cx.sh.eps,
        acts.q8_1_at(what, cx.sh.hidden, 1)?,
        &mut t.buf.xn,
        fault,
    )?;
    k.routers.enqueue(
        stream,
        cx.w.f32_tensor(what, &m.router)?,
        &t.buf.xn,
        cx.f32(&m.bias)?,
        sh.scale,
        &mut m.route,
        fault,
    )?;
    let hy = leg.hybrid();
    m.latent_down
        .project(cx, acts, l, &t.buf.xn, hy.boundary_mut().normed_mut())?;
    let h = Handoff {
        ids: m.route.ids(),
        weights: m.route.weights(),
        map: side.slots.buf(),
        row_off: l * side.stride,
        n_expert: m.route.n_expert(),
    };
    let target = hy.boundary_mut().handoff_target_of(0)?;
    side.handoff
        .enqueue_handoff(stream, &h, target, fault, &mut side.sel)?;
    hy.boundary().enqueue_go_of(stream, l, 0)
}

/// Layer `l`'s shadow on a hosted layer: the module doc's second part.
fn shadow(
    cx: &Cx<'_>,
    (k, sh): (&MoeKernels, MoeShape),
    side: &PlacedSide,
    (l, m): (usize, &MoeLayer),
    (s, acts): (&mut MoeScratch, &mut Acts),
    leg: &mut StepLeg<'_, HostRun>,
) -> Result<(), GpuError> {
    let (gpu, what) = (cx.gpu, cx.what);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    {
        // The shared expert's up reads the norm's (hidden, 1) q8_1 form
        // before the latent's quantize may overwrite it (`moe`'s module doc).
        let act = acts.q8_1_at(what, cx.sh.hidden, 1)?;
        one_expert(
            cx,
            k,
            l,
            &m.up_shexp,
            sh.shared_ff,
            act,
            &s.zero_id,
            &mut s.sh_up,
        )?;
    }
    if m.n_card > 0 {
        let act = acts.q8_1_at(what, sh.latent, 1)?;
        gpu.enqueue_quantize_q8_1_layer(leg.hybrid().boundary().normed(), act, l)?;
        let up = SelDown {
            w: stack(cx.w, what, m.up()?)?,
            act,
            sel: &side.sel,
            n_slots: TOP_K,
            rows_per_expert: m.ff,
        };
        k.iq.enqueue_gemv_sel_cols(stream, GgmlType::IQ2_XXS, &up, TOP_K, fault, &mut s.up)?;
        let act = acts.q8_1_at(what, m.ff, TOP_K)?;
        gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.up, act, l)?;
        let down = SelDown {
            w: stack(cx.w, what, m.down()?)?,
            act,
            sel: &side.sel,
            n_slots: TOP_K,
            rows_per_expert: sh.latent,
        };
        k.iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut s.down)?;
    }
    let act = acts.q8_1_at(what, sh.shared_ff, 1)?;
    gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.sh_up, act, l)?;
    one_expert(
        cx,
        k,
        l,
        &m.down_shexp,
        cx.sh.hidden,
        act,
        &s.zero_id,
        &mut s.sh,
    )
}

/// Layer `l`'s back on a hosted layer: the module doc's third part.
fn back(
    cx: &Cx<'_>,
    sh: MoeShape,
    side: &mut PlacedSide,
    (l, m): (usize, &MoeLayer),
    (t, s, acts): (&mut hybrid::Scratch, &mut MoeScratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    leg: &mut StepLeg<'_, HostRun>,
) -> Result<(), GpuError> {
    let gpu = cx.gpu;
    let stream = gpu.stream();
    let hy = leg.hybrid();
    hy.boundary().enqueue_back_of(stream, 0, l)?;
    {
        let hsum = hy.boundary().hsum_of(0)?;
        let merged: &DeviceBuffer<f32> = if m.n_card > 0 {
            side.acc_k.enqueue_card_acc(
                stream,
                &CardAcc {
                    down: &s.down,
                    w: m.route.weights(),
                    sel: &side.sel,
                    n: sh.latent,
                    m: 1,
                    n_card: m.n_card,
                    n_used: TOP_K,
                    w_pitch: TOP_K,
                    fault: gpu.layer_sink(l)?,
                },
                &mut side.acc,
            )?;
            gpu.elem()
                .enqueue_add(stream, &side.acc, hsum, sh.latent, &mut side.sum)?;
            &side.sum
        } else {
            hsum
        };
        m.latent_up.project(cx, acts, l, merged, &mut t.buf.delta)?;
    }
    gpu.elem()
        .enqueue_add(stream, &t.buf.delta, &s.sh, cx.sh.hidden, &mut s.sum)?;
    residual(cx, src, &s.sum, dst)?;
    hy.row_enqueued(l, 0)
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's
/// step port: each layer's front, shadow and back, the layers without a host
/// share their whole-card programs in the front.
pub(super) struct StepWalk<'a> {
    pub(super) cx: Cx<'a>,
    pub(super) mk: &'a MoeKernels,
    pub(super) moe: MoeShape,
    pub(super) layers: &'a mut [Layer],
    pub(super) t: &'a mut hybrid::Scratch,
    pub(super) ms: &'a mut MoeScratch,
    pub(super) acts: &'a mut Acts,
    pub(super) side: &'a mut PlacedSide,
    /// The head's input and the body's own residual buffer.
    pub(super) hin: &'a mut DeviceBuffer<f32>,
    pub(super) r: &'a mut DeviceBuffer<f32>,
}

impl<'a> StepWalk<'a> {
    /// Walk every layer through `leg`.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let o = Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        };
        let n = self.layers.len();
        sched::walk(o, n, leg, &mut self)
    }

    /// Whether layer `l` reads the head's input: the residual stream
    /// alternates between it and the body's buffer, and the last layer
    /// writes the head's whatever the layer count's parity.
    fn reads_head(&self, l: usize) -> bool {
        self.layers.len().is_multiple_of(2) != (l % 2 == 1)
    }
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted[at.layer]
    }

    /// A layer with no host share whole; a hosted one up to its go.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) = if self.reads_head(l) {
            (&*self.hin, &mut *self.r)
        } else {
            (&*self.r, &mut *self.hin)
        };
        let hosted = self.side.hosted[l];
        let (t, acts) = (&mut *self.t, &mut *self.acts);
        match &mut self.layers[l] {
            Layer::Mamba(m) => mamba_layer(&self.cx, (l, m), (t, acts), (src, dst)),
            Layer::Attn(a) => attn_layer::<GROUP_16>(&self.cx, (l, a), (t, acts), (src, dst)),
            Layer::Moe(m) if !hosted => super::moe::moe_layer(
                &self.cx,
                (self.mk, self.moe),
                (l, m),
                (t, &mut *self.ms, acts),
                (src, dst),
            ),
            Layer::Moe(m) => front(
                &self.cx,
                (self.mk, self.moe),
                self.side,
                (l, m),
                (t, acts),
                (src, leg),
            ),
        }
    }

    fn shadow(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] {
            return Ok(());
        }
        let Layer::Moe(m) = &self.layers[l] else {
            return Err(GpuError::state(WHAT, "a hosted layer is a MoE layer"));
        };
        shadow(
            &self.cx,
            (self.mk, self.moe),
            self.side,
            (l, m),
            (&mut *self.ms, &mut *self.acts),
            leg,
        )
    }

    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] {
            return Ok(());
        }
        let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) = if self.reads_head(l) {
            (&*self.hin, &mut *self.r)
        } else {
            (&*self.r, &mut *self.hin)
        };
        let Layer::Moe(m) = &self.layers[l] else {
            return Err(GpuError::state(WHAT, "a hosted layer is a MoE layer"));
        };
        back(
            &self.cx,
            self.moe,
            self.side,
            (l, m),
            (&mut *self.t, &mut *self.ms, &mut *self.acts),
            (src, dst),
            leg,
        )
    }
}
