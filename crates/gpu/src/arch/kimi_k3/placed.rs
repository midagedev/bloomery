//! A Kimi K3 load whose routed experts the stage card cannot all hold: the plan's card share of
//! each layer's stacks resident (the id prefix `[0, n_card)`), an expert tier card beside it
//! (the family's tier module) holding the next ids, and the rest served by the common host tier
//! ([`crate::host`]) and, past its room, by the NVMe tier. The token embedding is a host row
//! ([`EmbedTable`]): the plan leaves the table on the host, so the stage card holds the dense
//! trunk and the stores.
//!
//! A layer the plan holds whole runs the whole-card block; a layer with a tier or host share runs
//! this module's three parts of the block around the host tier's step port, the module doc of
//! [`super`] in the host leg's order:
//!
//! - **front**: the layer's mixer and the block's norm (`Walk::attn_half`), the router on the
//!   full-width row, the latent down into the boundary's activation (the host and the tier read
//!   the experts' own input, whose width is the latent's), the handoff of the ids, weights and that
//!   row with each slot's place in the card's stacks, and the go;
//! - **shadow**, under the host's work: the shared stack on the full-width row and the card's
//!   slots (the row's q8_1, the gate·up under SiTU and the down over the places the handoff wrote;
//!   a [`HOST`](crate::hybrid::HOST) slot's warps return before their first load), summed in slot
//!   order;
//! - **back**: the wait, the card's, the tier's and the host's sums added (the routed sum is
//!   linear in the experts' rows, so each tier sums the weighted rows it holds), the routed norm,
//!   the latent up, and the shared stack's output added.
//!
//! The sum's order is `(card + tier) + host`: up to rounding the whole-card layer's, which sums
//! the sixteen slots in one launch. The body walks a prompt one id a step, so there is no rows
//! pass here.

use super::super::tier::{
    NODE_FAMILY, PAGE_ROWS, TierParts, TierSum, enqueue_tier_sum, host_layer,
};
use super::{Ffn, Instance, Layer, MoeDims, Walk};
use crate::arch::hybrid::{Acts, GqaCx};
use crate::arch::routed_ffn::{FfnKernels, Glu, Routed};
use crate::card_leg::{LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::card_slots::CardSlotKernels;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::leg_tier::{Hang, LegOpen};
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::host::shell::TierShell;
use crate::host::swap_source::ResidencyGlue;
use crate::host::tier::TierOpen;
use crate::host::{StepLeg, refuse_tier_count};
use crate::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use crate::kquant::Act;
use crate::model::hosted_hybrid_at;
use crate::routed_block::{RouteIn, RouterEntry, handoff_go};
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::kimi_k3::names;
use model::arch::kimi_k3::place::PlanInputs;
use model::embed::EmbedTable;
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const PLACED: &str = "kimi_k3 placed load";

/// What the placed step's refusals name.
const STEP: &str = "kimi_k3 placed step";

/// The step's walk: one row through the host tier's step port.
pub(super) const STEP_WALK: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The rows in flight a load's host boundary and tier page hold: the step's one.
pub(super) const LOAD_ROWS: usize = PAGE_ROWS;

/// The token embedding as the host reads it: the table and one dequantized row.
struct EmbedHost {
    table: EmbedTable,
    row: Vec<f32>,
}

/// The stage card's side of a placed load: which layers the host or the tier serve a share of,
/// each routed layer's card count and slot-map row, the slot map's card copy, the handoff and sum
/// launches and the places of one layer's slots.
pub(super) struct Side {
    hosted: Vec<bool>,
    n_card: Vec<usize>,
    row_off: Vec<usize>,
    /// The slot map's card copy, one row of `n_expert` places a routed layer.
    slots: Arc<DeviceTensor<u32>>,
    handoff: HandoffKernels,
    acc_k: CardSlotKernels,
    /// The routed slots' places in the card's stacks, or `HOST`.
    sel: DeviceBuffer<u32>,
    /// The stage card's side of the expert tier, when the plan hangs one.
    tier: Option<TierParts>,
}

/// The placed side of a load: the host tier and its glue, what the walk reads beside them, and the
/// token embedding. The shell is the first field: the residency machine stops before any card or
/// pinned memory frees.
pub struct Placed {
    pub(super) shell: TierShell<HostRun>,
    pub(super) side: Side,
    embed: EmbedHost,
}

impl Drop for Placed {
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

/// What [`Placed::new`] is made from.
pub(super) struct Open<'a> {
    pub(super) file: &'a Arc<Split>,
    pub(super) plan: &'a Plan<'a>,
    pub(super) inputs: &'a PlanInputs,
    pub(super) host: HostCfg,
    pub(super) set: HostResidency,
    pub(super) glue: ResidencyGlue,
}

impl Placed {
    /// The placed side of the load `open` describes, over the routed layers `layers` of the chain
    /// and the slot map `map` of the routed run: the host tier over the layers whose card count is
    /// under their experts, the plan's expert tier card hung under it, the card's side and the
    /// embedding table. Refused by name: more tier cards than the host tier serves, a layer whose
    /// stacks are not its card count's, and a vocabulary that is not the file's. The host serves
    /// its experts' share under SiTU at the file's betas ([`host_layer`]). Load-time only.
    pub(super) fn new(
        (gpu, w): (&Gpu, &Weights),
        layers: &[Layer],
        map: SlotMap,
        open: Open<'_>,
    ) -> Result<Placed, GpuError> {
        let Open {
            file,
            plan,
            inputs,
            host,
            set,
            glue,
        } = open;
        let hp = &inputs.hp;
        let (n_expert, n_used) = (hp.experts.n_expert, hp.experts.n_used);
        let n = layers.len();
        let tiers = TierOpen::of_machine(plan.machine)?;
        refuse_tier_count(PLACED, tiers.len())?;
        let (mut hosted, mut n_card, mut row_off) = (vec![false; n], vec![0; n], vec![0; n]);
        let shape = LegShape {
            cap: 1,
            slots_per_col: n_used,
            n: hp.latent,
            ff: hp.experts.ff,
        };
        let run = map.layers();
        for l in run.clone() {
            let on_card = map.on_card(l)?;
            n_card[l] = on_card;
            row_off[l] = map.row_offset(l).ok_or_else(|| {
                GpuError::shape(PLACED, format!("layer {l}: a routed layer with no map row"))
            })?;
            hosted[l] = on_card < n_expert;
            if on_card > 0 {
                let Ffn::Routed(r) = &layers[l].ffn else {
                    return Err(GpuError::shape(
                        PLACED,
                        format!("layer {l}: experts on the card of a layer with no routed block"),
                    ));
                };
                LegStacks::of_resident(w, r.stacks.each_ref().map(String::as_str))?
                    .check(on_card, shape)?;
            }
        }
        let hosted_set: Vec<usize> = (0..n).filter(|&l| hosted[l]).collect();
        let stream = gpu.stream();
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            n_expert,
        )?);
        let hang = (!tiers.is_empty()).then(|| Hang {
            tiers: &tiers,
            family: &NODE_FAMILY,
            open: LegOpen {
                n_expert,
                layers: n,
                hidden: hp.latent,
                n_used,
                rows: LOAD_ROWS,
            },
            cols: LOAD_ROWS,
            card_dontneed: host.card_dontneed,
            bytes: crate::model::CardBytes::File,
        });
        let hybrid = hosted_hybrid_at(
            (gpu, file, plan),
            (&hosted_set, n, map),
            (host, set, &glue),
            (
                HostWidths {
                    embd: hp.latent,
                    ff: hp.experts.ff,
                    n_used,
                },
                LOAD_ROWS,
            ),
            hang.as_ref(),
            |src| hosted_set.iter().map(|&l| host_layer(src, hp, l)).collect(),
        )?;
        let tier = if tiers.is_empty() {
            None
        } else {
            Some(TierParts::new(gpu, hybrid.slots(), n, n_used)?)
        };
        let table = EmbedTable::new(Arc::clone(file), &names::token_embd(), hp.n_embd)
            .map_err(|e| GpuError::shape(PLACED, format!("the embedding table: {e}")))?;
        if table.n_vocab() != hp.n_vocab {
            return Err(GpuError::shape(
                PLACED,
                format!(
                    "the embedding has {} rows, the file's vocabulary {}",
                    table.n_vocab(),
                    hp.n_vocab
                ),
            ));
        }
        Ok(Placed {
            shell: TierShell::new(hybrid, glue, hp.n_embd, PLACED),
            side: Side {
                hosted,
                n_card,
                row_off,
                slots,
                handoff: HandoffKernels::load(gpu.context())?,
                acc_k: CardSlotKernels::load(gpu.context())?,
                sel: DeviceBuffer::zeroed(stream, n_used)?,
                tier,
            },
            embed: EmbedHost {
                table,
                row: vec![0.0; hp.n_embd],
            },
        })
    }

    /// Device bytes the placed side holds beside the plan's terms for the tier: the slot map's
    /// card copy, the places and the boundary.
    pub(super) fn device_bytes(&self) -> usize {
        let s = &self.side;
        s.slots.buf().num_bytes() + s.sel.num_bytes() + self.shell.hybrid.boundary().device_bytes()
    }

    /// Device bytes the stage card's side of the expert tier holds: what the plan's tier reserve
    /// counts.
    pub(super) fn tier_bytes(&self) -> usize {
        self.side.tier.as_ref().map_or(0, TierParts::bytes)
    }

    /// Layer `l` of the model: a layer past it panics by name, as no accessor below answers for a
    /// layer the model does not have.
    fn layer(&self, what: &str, l: usize) -> usize {
        assert!(
            l < self.side.hosted.len(),
            "{what}: layer {l} is past the model's {} layers",
            self.side.hosted.len()
        );
        l
    }

    /// The expert tier's experts of layer `l`: 0 off the tier and on a load that hangs none. A
    /// layer past the model panics by name.
    #[must_use]
    pub fn tier_k(&self, l: usize) -> usize {
        let l = self.layer("tier_k", l);
        self.side.tier.as_ref().map_or(0, |t| t.k(l))
    }

    /// Whether the host or the tier serve a share of layer `l`. A layer past the model panics by
    /// name.
    #[must_use]
    pub fn hosted(&self, l: usize) -> bool {
        self.side.hosted[self.layer("hosted", l)]
    }

    /// The card's experts of layer `l`: 0 on the dense layer. A layer past the model panics by
    /// name.
    #[must_use]
    pub fn n_card(&self, l: usize) -> usize {
        self.side.n_card[self.layer("n_card", l)]
    }

    /// The host tier, for a gate's instrument.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.shell.hybrid
    }

    /// The host tier's settling ([`crate::host::HostTier::settle`]).
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.shell.hybrid.settle(stream)
    }

    /// The step's embedding row of `token` into `y` (the first sum buffer), and the position
    /// words the card's embedding kernel writes (`pos`, the live key count `pos + 1`) from the
    /// host.
    pub(super) fn refresh(
        &mut self,
        stream: &CudaStream,
        token: u32,
        (y, pos, n_keys): (
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<u32>,
            &mut DeviceBuffer<u32>,
        ),
        at: u32,
    ) -> Result<(), GpuError> {
        let EmbedHost { table, row } = &mut self.embed;
        table
            .row_into(token, row)
            .map_err(|e| GpuError::shape(STEP, format!("token {token}: {e}")))?;
        y.copy_from_host(stream, row)?;
        pos.copy_from_host(stream, &[at])?;
        n_keys.copy_from_host(stream, &[at + 1])?;
        Ok(())
    }
}

impl TierBody for Placed {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.shell.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.shell.hybrid
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.shell.residency_parts(&self.side.slots)
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.shell.hybrid.serve_captured_of(chain)
    }
}

/// One router instance's buffers of a hosted layer's block, beside the whole-card block's
/// (`Router`): the router's results, the card leg's, the card's, the tier's and the cards' sums
/// and the block's stages. Every buffer is one row.
pub(super) struct Bufs<R: RouterEntry> {
    route: R::Out,
    leg: LegBufs,
    /// The card slots' weighted sum, the tier slots' and the two cards', a latent row each.
    acc: DeviceBuffer<f32>,
    tier: DeviceBuffer<f32>,
    cards: DeviceBuffer<f32>,
    /// The routed sum (the cards' plus the host's) and its norm, a latent row each.
    routed: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    /// The latent up's output and the shared stack's, a row each; the shared stack's
    /// intermediates.
    up: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
}

impl<R: RouterEntry> Bufs<R> {
    /// The buffers of a model of `n` values a row at `dims`. Load-time only.
    pub(super) fn new(stream: &CudaStream, n: usize, dims: &MoeDims) -> Result<Bufs<R>, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let leg = LegShape {
            cap: 1,
            slots_per_col: R::N_USED,
            n: dims.latent,
            ff: dims.expert_ff,
        };
        Ok(Bufs {
            route: R::new_out(stream)?,
            leg: LegBufs::for_down(stream, leg, dims.shape().down)?,
            acc: z(dims.latent)?,
            tier: z(dims.latent)?,
            cards: z(dims.latent)?,
            routed: z(dims.latent)?,
            normed: z(dims.latent)?,
            up: z(n)?,
            shared: z(n)?,
            gate: z(dims.shared_ff)?,
            value: z(dims.shared_ff)?,
            act: z(dims.shared_ff)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        R::out_bytes(&self.route)
            + self.leg.bytes()
            + [
                &self.acc,
                &self.tier,
                &self.cards,
                &self.routed,
                &self.normed,
                &self.up,
                &self.shared,
                &self.gate,
                &self.value,
                &self.act,
            ]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
    }
}

/// The shared stack over the full-width row `xn` into `y`: `down(situ(gate(xn), up(xn)))`.
fn shared_stack<R: RouterEntry>(
    cx: &GqaCx<'_>,
    (acts, rule): (&mut Acts, Act),
    (l, g): (usize, &Glu),
    xn: &DeviceBuffer<f32>,
    p: &mut Bufs<R>,
) -> Result<(), GpuError> {
    let Bufs {
        gate,
        value,
        act,
        shared,
        ..
    } = p;
    g.gate.project(cx, acts, l, xn, gate)?;
    g.up.project(cx, acts, l, xn, value)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (&*gate, &*value), g.ff, act)?;
    g.down.project(cx, acts, l, act, shared)
}

/// What one hosted layer's three parts read beside the router instance: the launch context, the
/// forms, the rule and the router's scale, the layer, and the normed row the layer's mixer left.
pub(super) struct Block<'a> {
    pub(super) cx: &'a GqaCx<'a>,
    /// The experts' width: the latent's.
    pub(super) latent: usize,
    pub(super) rule: Act,
    pub(super) scale: f32,
    pub(super) l: usize,
    pub(super) f: &'a Ffn,
    pub(super) xn: &'a DeviceBuffer<f32>,
}

impl Block<'_> {
    fn routed(&self) -> Result<&Routed, GpuError> {
        match self.f {
            Ffn::Routed(r) => Ok(r),
            Ffn::Dense(_) => Err(GpuError::shape(
                STEP,
                format!(
                    "layer {}: a dense block with experts on the host tier",
                    self.l
                ),
            )),
        }
    }
}

/// A hosted layer's front from the block's norm: the router, the latent down into the boundary's
/// activation, the handoff and the go.
pub(super) fn front<R: RouterEntry>(
    (k, p): (&FfnKernels<R>, &mut Bufs<R>),
    b: &Block<'_>,
    acts: &mut Acts,
    side: &mut Side,
    hy: &mut Hybrid<HostRun>,
) -> Result<(), GpuError> {
    let (cx, l) = (b.cx, b.l);
    let (gpu, w) = (cx.gpu, cx.w);
    let r = b.routed()?;
    let lat = r.latent.as_deref().ok_or_else(|| {
        GpuError::shape(
            STEP,
            format!("layer {l}: a routed block with no latent pair"),
        )
    })?;
    let rin = RouteIn {
        w: w.f32_tensor(STEP, &r.router)?,
        bias: Some(w.f32_buf(STEP, &r.bias)?),
        scale: b.scale,
        fault: gpu.layer_sink(l)?,
    };
    k.router.route(gpu.stream(), b.xn, &rin, &mut p.route)?;
    lat.down
        .project(cx, acts, l, b.xn, hy.boundary_mut().normed_mut())?;
    let Side {
        slots,
        row_off,
        handoff,
        sel,
        tier,
        ..
    } = side;
    let h = Handoff {
        ids: R::out_ids(&p.route),
        weights: R::out_weights(&p.route),
        map: slots.buf(),
        row_off: row_off[l],
        n_expert: R::N_EXPERT,
    };
    match tier.as_mut().filter(|t| t.k(l) > 0) {
        Some(t) => t.enqueue_step_go(gpu, handoff, &h, hy, l, sel),
        None => handoff_go(gpu, handoff, &h, hy, (l, 0), sel),
    }
}

/// A hosted layer's card work under the host's: the shared stack on the full-width row and the
/// card's slots over the latent row the handoff left in the boundary.
pub(super) fn shadow<R: RouterEntry>(
    (k, p): (&FfnKernels<R>, &mut Bufs<R>),
    b: &Block<'_>,
    acts: &mut Acts,
    side: &Side,
    hy: &Hybrid<HostRun>,
) -> Result<(), GpuError> {
    let (cx, l) = (b.cx, b.l);
    let r = b.routed()?;
    shared_stack(cx, (acts, b.rule), (l, &r.shared), b.xn, p)?;
    let n_card = side.n_card[l];
    if n_card == 0 {
        return Ok(());
    }
    let stacks = LegStacks::of_resident(cx.w, r.stacks.each_ref().map(String::as_str))?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: hy.boundary().normed(),
        m: 1,
        sel: &side.sel,
        n_card,
        rule: b.rule,
    };
    let Bufs {
        route, leg, acc, ..
    } = p;
    k.leg.enqueue(
        cx.gpu,
        leg,
        &run,
        Some(LegSum {
            w: R::out_weights(route),
            acc,
        }),
    )
}

/// A hosted layer's back from the join on: the wait, the cards' and the host's sums, the routed
/// norm, the latent up, and the block's update (the routed delta plus the shared stack) into
/// `update`.
pub(super) fn back<R: RouterEntry>(
    p: &mut Bufs<R>,
    b: &Block<'_>,
    acts: &mut Acts,
    side: &Side,
    hy: &mut Hybrid<HostRun>,
    update: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (cx, l) = (b.cx, b.l);
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let r = b.routed()?;
    let lat = r.latent.as_deref().ok_or_else(|| {
        GpuError::shape(
            STEP,
            format!("layer {l}: a routed block with no latent pair"),
        )
    })?;
    let gain = r.sum_norm.as_deref().ok_or_else(|| {
        GpuError::shape(STEP, format!("layer {l}: a routed block with no sum norm"))
    })?;
    let n_card = side.n_card[l];
    let n_tier = side.tier.as_ref().map_or(0, |t| t.k(l));
    if n_tier > 0 {
        hy.enqueue_tier_back(stream, l, 0)?;
    } else {
        hy.boundary().enqueue_back_of(stream, 0, l)?;
    }
    let hsum = hy.boundary().hsum_of(0)?;
    let weights = R::out_weights(&p.route);
    if n_tier > 0 {
        let t = side.tier.as_ref().ok_or(GpuError::state(
            STEP,
            "the stage card's tier side of a tier layer",
        ))?;
        let sum = TierSum {
            trows: hy.tier_rows(0)?,
            w: weights,
            tsel: t.step_tsel()?,
            n_tier,
            n: b.latent,
            n_used: R::N_USED,
            fault: gpu.layer_sink(l)?,
        };
        enqueue_tier_sum(gpu, &side.acc_k, &sum, &mut p.tier)?;
    }
    // The card's slots, then the tier's, then the host's.
    let cards: Option<&DeviceBuffer<f32>> = match (n_card > 0, n_tier > 0) {
        (true, true) => {
            gpu.elem()
                .enqueue_add(stream, &p.acc, &p.tier, b.latent, &mut p.cards)?;
            Some(&p.cards)
        }
        (true, false) => Some(&p.acc),
        (false, true) => Some(&p.tier),
        (false, false) => None,
    };
    let routed: &DeviceBuffer<f32> = match cards {
        Some(c) => {
            gpu.elem()
                .enqueue_add(stream, c, hsum, b.latent, &mut p.routed)?;
            &p.routed
        }
        None => hsum,
    };
    gpu.elem().enqueue_rms_norm(
        stream,
        routed,
        w.f32_buf(STEP, gain)?,
        cx.eps,
        b.latent,
        1,
        &mut p.normed,
    )?;
    lat.up.project(cx, acts, l, &p.normed, &mut p.up)?;
    gpu.elem()
        .enqueue_add(stream, &p.up, &p.shared, cx.hidden, update)?;
    hy.row_enqueued(l, 0)
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's step port: each layer's
/// mixer half, then its block whole or as the front, shadow and back around the host's go.
pub(super) struct StepWalk<'a> {
    pub(super) walk: Walk<'a>,
    pub(super) layers: &'a mut [Layer],
    pub(super) router: &'a mut Instance,
    pub(super) side: &'a mut Side,
    /// The blocks' rule and the router's scale.
    pub(super) rule: (Act, f32),
    /// The experts' width: the latent's.
    pub(super) latent: usize,
}

impl<'a> StepWalk<'a> {
    /// Walk every layer through `leg`.
    pub(super) fn run(&mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let n = self.layers.len();
        sched::walk(STEP_WALK, n, leg, self)
    }
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted[at.layer]
    }

    /// A layer with no tier or host share whole; a hosted one up to its go.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (rule, scale) = self.rule;
        let StepWalk {
            walk,
            layers,
            router,
            side,
            latent,
            ..
        } = self;
        let layer = &mut layers[l];
        walk.attn_half(l, layer)?;
        if !side.hosted[l] {
            router.run(
                &walk.cx,
                &mut *walk.acts,
                (rule, scale),
                (l, &layer.ffn),
                &walk.s.ffn_xn,
                &mut walk.s.update,
            )?;
            return walk.close(l);
        }
        let b = Block {
            cx: &walk.cx,
            latent: *latent,
            rule,
            scale,
            l,
            f: &layer.ffn,
            xn: &walk.s.ffn_xn,
        };
        router.front(&b, &mut *walk.acts, side, leg.hybrid())
    }

    /// The card's slots and the shared stack of a hosted layer, under the host's work.
    fn shadow(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] {
            return Ok(());
        }
        let (rule, scale) = self.rule;
        let StepWalk {
            walk,
            layers,
            router,
            side,
            latent,
            ..
        } = self;
        let b = Block {
            cx: &walk.cx,
            latent: *latent,
            rule,
            scale,
            l,
            f: &layers[l].ffn,
            xn: &walk.s.ffn_xn,
        };
        router.shadow(&b, &mut *walk.acts, side, &*leg.hybrid())
    }

    /// A hosted layer from the join on: the cards' and the host's sums and the block's tail, then
    /// the update added to the sum.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted[l] {
            return Ok(());
        }
        let (rule, scale) = self.rule;
        let StepWalk {
            walk,
            layers,
            router,
            side,
            latent,
            ..
        } = self;
        {
            let b = Block {
                cx: &walk.cx,
                latent: *latent,
                rule,
                scale,
                l,
                f: &layers[l].ffn,
                xn: &walk.s.ffn_xn,
            };
            router.back(
                &b,
                &mut *walk.acts,
                side,
                (leg.hybrid(), &mut walk.s.update),
            )?;
        }
        walk.close(l)
    }
}
