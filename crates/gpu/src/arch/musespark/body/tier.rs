//! The muse-spark expert tier ([`crate::host::leg_tier`]): plan (b′)'s second
//! card holds each layer's next ids after the stage card's prefix and computes
//! the routed slots the slot map sends it, in the host leg's shadow, through
//! the common card leg (`iq2_xxs_gate_up_sel` with SiLU, `iq2_xxs_gemv_sel`) in
//! the latent: the tier's activation and its rows are `moe_latent_size` wide.
//!
//! What is Muse's here is the family's reading ([`NODE_FAMILY`]) and the
//! tier's share of the experts' sum ([`TierParts`]). A tier slot's row is the
//! expert's raw down output; the stage card scales it by `ffn_post_exp_norm`
//! and norms it at the post epsilon — the card slots' and the host slots' own
//! rule — before the router weights apply, then sums the tier's slots in slot
//! order (`ds41_ffn_card_acc_8` over the tier places). The layer's sum is
//! `(card + tier) + host`: the tier's slots join the card's as a second sum, so
//! a tiered layer agrees with the one-card layer over the same experts to the
//! sums' order, not bit for bit.
//!
//! The step's tier handoff writes the tier image and its places with the host
//! image ([`handoff_go_tier`]) and the back waits for both
//! ([`Hybrid::enqueue_tier_back`]). A prompt pass downloads the tier places with
//! the host's rows ([`Hybrid::enqueue_download_tiered_pitched`]) and its back
//! reads the tier's rows through the batch port's join
//! ([`crate::host::BatchLeg::join_tiered`]).

use std::error::Error;
use std::ops::Range;

use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::models::{LayerSpec, Moe};
use model::arch::musespark::place::{self, PlanInputs};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::accepts::Program;
use model::r8file::R8Source;

use crate::card_slots::{CardAcc, CardSlotKernels, ColPlaces};
use crate::fault::FaultSink;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::leg_tier::{self, LegFamily, LegWidths, TierSide};
use crate::host::node::NodeFamily;
use crate::host::run::HostRun;
use crate::hybrid::{Hybrid, SlotMap};
use crate::kquant::Act;
use crate::routed_block::handoff_go_tier;
use crate::{Gpu, GpuError};

/// What the tier's errors name.
const WHAT: &str = "musespark tier";

/// The tier page's rows: the step's one row in flight.
pub(super) const PAGE_ROWS: usize = 1;

/// Tokens one tier run covers in a prompt block.
const CHUNK: usize = 8;

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(place::TIER_CHUNK as usize == CHUNK);

/// Muse's architecture facts for the common tier node service: the file's
/// reading, the latent's widths and the SiLU-gated experts.
struct MuseSpark;

impl LegFamily for MuseSpark {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "musespark";
    const ROWS: usize = PAGE_ROWS;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

    fn describe(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn into_model(inputs: PlanInputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &PlanInputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    /// The experts run in the latent: the tier's activation and rows are
    /// `moe_latent_size` wide.
    fn widths(inputs: &PlanInputs) -> LegWidths {
        let hp = &inputs.hp;
        LegWidths {
            n_expert: hp.n_expert,
            n_used: hp.n_used,
            n_embd: hp.latent,
            ff: hp.expert_width(),
            program: Program::MuseSparkBody,
        }
    }

    /// Refused by name: a node tier's host run would serve its host experts
    /// without the post-expert norm each row takes before its weight
    /// ([`HostRun::set_post_norm`]), which the node's run does not set.
    fn routed_views(
        _: R8Source<'_>,
        _: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
        Err(format!(
            "{WHAT}: a node tier's host run over layers {run:?}: each host expert's row is \
             normed before its weight, which the node's host run does not set"
        )
        .into())
    }

    /// `silu(g) · u`: the file's experts carry no clamp.
    fn tier_act(_: &Moe) -> Result<Act, GpuError> {
        Ok(Act::SiluMul)
    }
}

/// Muse's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<MuseSpark>();

/// The stage card's side of a tiered load: the common tier side (the slot
/// map's tier view and the step's tier places) and a pass's tier places.
pub(super) struct TierParts {
    side: TierSide,
    /// A prompt pass's tier places, `n_used` a row.
    pass_tsel: DeviceBuffer<u32>,
}

/// What a tier layer's sum reads: the tier's raw rows `trows` of `m` rows of
/// `n_used` slots (slot-major, `n` values a slot), the layer's post-expert
/// gain and the post epsilon, the slots' weights and tier places, and the
/// layer's tier experts.
pub(super) struct TierSum<'a> {
    pub(super) trows: &'a DeviceBuffer<f32>,
    pub(super) gain: &'a DeviceBuffer<f32>,
    pub(super) post_eps: f32,
    pub(super) w: &'a DeviceBuffer<f32>,
    pub(super) tsel: &'a DeviceBuffer<u32>,
    pub(super) n_tier: usize,
    pub(super) n: usize,
    pub(super) m: usize,
    pub(super) n_used: usize,
    pub(super) fault: FaultSink,
}

impl TierParts {
    /// The parts for `map`'s tier over `layers` layers, a pass of up to
    /// `rows` rows of `n_used` slots. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        (rows, n_used): (usize, usize),
    ) -> Result<TierParts, GpuError> {
        Ok(TierParts {
            side: TierSide::new(gpu, map, layers, (PAGE_ROWS, n_used))?,
            pass_tsel: DeviceBuffer::zeroed(gpu.stream(), rows.max(1) * n_used)?,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(super) fn k(&self, l: usize) -> usize {
        self.side.k(l)
    }

    /// Device bytes the parts hold.
    pub(super) fn bytes(&self) -> usize {
        self.side.bytes() + self.pass_tsel.num_bytes()
    }

    /// The step's tier places, as the last handoff wrote them.
    pub(super) fn step_tsel(&self) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.side.tsel(0)
    }

    /// A pass's tier places, as [`TierParts::enqueue_pass_places`] wrote them.
    pub(super) fn pass_tsel(&self) -> &DeviceBuffer<u32> {
        &self.pass_tsel
    }

    /// Layer `l`'s step handoff of a tier layer and the go that releases the
    /// host and the tier ([`handoff_go_tier`]): the host image, the tier
    /// image (the boundary's activation and each slot's tier place) and each
    /// slot's stage place into `sel`.
    pub(super) fn enqueue_step_go(
        &mut self,
        gpu: &Gpu,
        k: &HandoffKernels,
        h: &Handoff<'_>,
        hybrid: &mut Hybrid<HostRun>,
        l: usize,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let (tmap, tsel) = self.side.handoff_parts(0)?;
        handoff_go_tier(gpu, k, h, tmap, hybrid, (l, 0), (sel, tsel))
    }

    /// Each routed slot's tier place of a pass's `m` rows of routed ids
    /// (`n_used` a row, at `n_used` a row in the places), from the tier view's
    /// row of layer `l` (`n_expert` places a row): a slot the tier does not
    /// hold is [`HOST`](crate::hybrid::HOST). One launch.
    pub(super) fn enqueue_pass_places(
        &mut self,
        stream: &CudaStream,
        k: &CardSlotKernels,
        (ids, l, m): (&DeviceBuffer<u32>, usize, usize),
        (n_used, n_expert): (usize, usize),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        k.enqueue_places_cols(
            stream,
            &ColPlaces {
                ids,
                ids_pitch: n_used,
                used: n_used,
                sel_pitch: n_used,
                m,
                map: self.side.places(),
                row_off: l * n_expert,
                n_expert,
            },
            fault,
            &mut self.pass_tsel,
        )
    }
}

/// The tier's share of a layer's experts' sum (module doc): each tier slot's
/// row of `s.trows` scaled by the post-expert gain and normed at the post
/// epsilon into `normed`, then the tier slots' weighted sum in slot order into
/// `acc`. Two launches.
pub(super) fn enqueue_tier_sum(
    gpu: &Gpu,
    k: &CardSlotKernels,
    s: &TierSum<'_>,
    (normed, acc): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    gpu.elem().enqueue_gain_rms_norm(
        stream,
        s.trows,
        s.gain,
        s.post_eps,
        s.n,
        s.m * s.n_used,
        normed,
    )?;
    k.enqueue_card_acc(
        stream,
        &CardAcc {
            down: &*normed,
            w: s.w,
            sel: s.tsel,
            n: s.n,
            m: s.m,
            n_card: s.n_tier,
            n_used: s.n_used,
            w_pitch: s.n_used,
            fault: s.fault,
        },
        acc,
    )
}
