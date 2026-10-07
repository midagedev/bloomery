//! A placed load of the family's chain ([`Placed`]): `Body` (Qwen3-30B-A3B)
//! and `Body35` (Qwen3.6-35B-A3B) with the trunk on the card and the routed
//! experts split by a placement plan — each layer's experts the plan puts on
//! the card in its stacks, the rest on the host tier (`crate::host`), which
//! computes them with qdot's fused kernels while the card runs its own.
//!
//! The walks are the layer program's (`program`) over a host leg: the
//! decode step `(1, 1, Step)` through the tier's step port ([`StepWalk`],
//! captured), a prompt unit `(1, m, Batch)` of up to [`GEMV_COLS`] rows
//! through its batch port ([`BatchWalk`]). A layer's parts:
//! - the front: the mixer (`dispatch::mixer`, the chain's own), the FFN
//!   norm with its q8_1 form and the router — the norm into the step
//!   boundary's activation, or the arena's for a unit — then the host leg's
//!   send: the step's handoff into the page with each routed slot's place
//!   and the go; a unit's download of its rows and routing with its places
//!   (`hostleg`);
//! - the shadow: the card's routed experts of the layer when it has any —
//!   the K-quant family's Q4_K gate·up over the places, the q8_1 of the card
//!   slots' rows, the Q4_K or Q6_K down `_sel` — and on Qwen3.6 the shared
//!   expert on its own stacks, a stack of one expert on a fixed route of
//!   expert 0 ([`SharedFfn`]); a slot the host serves is [`HOST`] there and
//!   no launch reads or writes it;
//! - the back: the step's wait (a unit's serve and upload are the walk's),
//!   then the combine of the card's slots with the host's routed sum, the
//!   residual and the shared expert (`hostleg`), into the next layer's
//!   input — the head's after the last layer of the step.
//!
//! The slot map is the plan's for the load's life (no residency machine):
//! its card copy is what the handoff and the places read, and the host tier
//! skips the same slots by the same map, so which expert runs where has one
//! owner. The shared expert is never one of the router's slots here: a
//! placed load keeps Qwen3.6's routed stacks and its shared expert apart
//! (the whole-card load folds the shared expert into its stacks as one more
//! expert, which a card holding part of a layer's experts cannot).
//!
//! Numerics: the routing is the whole-card chain's bit for bit (the step's
//! norm and router are `norm_quant` and the router, which the whole-card
//! step's fused launch writes bit for bit); the host's experts are qdot's
//! (q8_K activations, the host's sum order), the card's the K-quant
//! family's — the same Walk A arithmetic as the whole-card launches — so a
//! placed load agrees with the whole-card one to the error of those sums.

use super::body::{Kernels, kq_site};
use super::dispatch::{self, Ctx, PassCtx};
use super::hostleg::{HostCombineArgs, HostLegKernels, PlacesArgs};
use super::plan::{FfnPlan, FfnRoute, LayerPlan, SiteTy};
use super::program::{Stores, Tail, enqueue_tail};
use super::scratch::{Arena, Dims, Io};
use super::wide::{GEMV_COLS, act_col_bytes};
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::run::{HostRun, HostWidths};
use crate::host::{BatchLeg, StepLeg};
use crate::hybrid::{Boundary, BoundaryShape, HOST, HostResidency, Hybrid, Refusal, SlotMap};
use crate::kquant::{Act, GateUpAct, KquantKernels};
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::model::{HostServed, MAX_PASS_ROWS};
use crate::q4k_sel::QuantSel;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const WHAT: &str = "qwen3moe placed load";

/// The shared expert of a placed Qwen3.6 layer: its three stacks of one
/// expert each, as the file holds them, and its down's type.
pub(super) struct SharedFfn {
    gate: String,
    up: String,
    down: String,
    down_ty: SiteTy,
}

/// A placed layer's routed experts on the card: how many the slot map puts
/// there, the file's three stacks (the card holds their rows of those
/// experts, in the map's slot order) and the down's type; and the shared
/// expert, on a chain that has one.
pub(super) struct PlacedFfn {
    n_card: usize,
    gate: String,
    up: String,
    down: String,
    down_ty: SiteTy,
    shared: Option<SharedFfn>,
}

/// What the host leg's launches take a placed chain to be: the experts a
/// layer, the routed slots a token, the router's slots a token (the routed
/// ones and a gated router's shared slot), the model width and an expert's.
#[derive(Clone, Copy, Debug)]
pub(super) struct RouteShape {
    pub(super) n_expert: usize,
    pub(super) used: usize,
    pub(super) pitch: usize,
    pub(super) hidden: usize,
    pub(super) ff: usize,
}

impl RouteShape {
    /// The shape of arena dims `d`, whose router routes; a chain of dense
    /// FFNs is refused by name.
    fn of(d: &Dims) -> Result<RouteShape, GpuError> {
        let r = d.routed(WHAT)?;
        Ok(RouteShape {
            n_expert: r.experts(),
            used: r.used(),
            pitch: d.slots(),
            hidden: d.hidden,
            ff: d.ff,
        })
    }

    /// Whether the router's slots hold the shared expert's beside the routed
    /// ones (Qwen3.6's gated router).
    fn shared(self) -> bool {
        self.pitch > self.used
    }
}

/// One arena's buffers of the host leg: each slot's place (`pitch` a row),
/// and for the shared expert its fixed route (every row's slot on expert 0),
/// its SwiGLU rows, their q8_1 form (one activation of `m` columns for each
/// gemv width `m`) and its output.
pub(super) struct PlacedRows {
    sel: DeviceBuffer<u32>,
    shared: Option<SharedRows>,
}

/// The shared expert's buffers of [`PlacedRows`].
struct SharedRows {
    zero: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    act: Vec<Q8Act>,
    y: DeviceBuffer<f32>,
}

impl PlacedRows {
    /// The buffers of an arena of `rows` rows (at most [`GEMV_COLS`]) for
    /// `shape`, every place [`HOST`] until a launch writes it. Load-time
    /// only.
    fn new(stream: &CudaStream, shape: RouteShape, rows: usize) -> Result<PlacedRows, GpuError> {
        if rows == 0 || rows > GEMV_COLS {
            return Err(GpuError::shape(
                WHAT,
                format!("a host leg's rows for an arena of {rows} rows (1..={GEMV_COLS})"),
            ));
        }
        let sel = DeviceBuffer::from_host(stream, &vec![HOST; rows * shape.pitch])?;
        let shared = shape
            .shared()
            .then(|| {
                Ok::<_, GpuError>(SharedRows {
                    zero: DeviceBuffer::zeroed(stream, rows)?,
                    h: DeviceBuffer::zeroed(stream, rows * shape.ff)?,
                    act: (1..=rows)
                        .map(|m| Q8Act::with_k(stream, m, shape.ff))
                        .collect::<Result<Vec<_>, _>>()?,
                    y: DeviceBuffer::zeroed(stream, rows * shape.hidden)?,
                })
            })
            .transpose()?;
        Ok(PlacedRows { sel, shared })
    }

    /// Device bytes.
    fn bytes(&self) -> usize {
        self.sel.num_bytes()
            + self.shared.as_ref().map_or(0, |s| {
                s.zero.num_bytes()
                    + s.h.num_bytes()
                    + s.act.iter().map(Q8Act::device_bytes).sum::<usize>()
                    + s.y.num_bytes()
            })
    }
}

/// The kernels of the host leg: the handoff into the page, the K-quant
/// family's gate·up, and this family's places and combine (`hostleg`).
pub(super) struct PlacedKernels {
    handoff: HandoffKernels,
    kquant: KquantKernels,
    leg: HostLegKernels,
}

/// What every placed walk reads besides the arena: each layer's card
/// experts, the slot map's card copy, the route's shape and the kernels.
pub(super) struct PlacedSide {
    ffn: Vec<PlacedFfn>,
    slots: DeviceTensor<u32>,
    shape: RouteShape,
    k: PlacedKernels,
}

/// The placed side of a load: the host tier over every layer's routed
/// experts, what the walks read, and the step's and the pass arena's host
/// leg buffers, with the host sums a unit's upload lands in.
pub struct Placed {
    pub(super) hybrid: Hybrid<HostRun>,
    pub(super) side: PlacedSide,
    pub(super) step: PlacedRows,
    pub(super) pass: PlacedRows,
    pub(super) pass_hsum: DeviceBuffer<f32>,
}

/// What [`Placed::new`] builds a load's placed side from.
pub(super) struct PlacedOpen<'a> {
    /// The plan the load's weights were uploaded by; its card 0 the load's.
    pub(super) plan: &'a Plan<'a>,
    /// The file, shared with the host tier.
    pub(super) file: Arc<Split>,
    pub(super) host: HostCfg,
    /// The plan's host set as the load left it.
    pub(super) set: HostResidency,
    /// The architecture the host run's refusals name.
    pub(super) arch: &'static str,
}

impl Placed {
    /// The placed side of a load of `plans` over arena dims `d`, from `o`:
    /// the plan's slot map ([`SlotMap::of_plan`]) and its card copy, each
    /// layer's card stacks checked against its card count and the types the
    /// launches take ([`kq_site`]: a Q4_K gate and up, a Q4_K or Q6_K down;
    /// a stack of other rows would leave slots that neither side sums), the
    /// shared expert's stacks on a gated router's chain, the boundary of one
    /// row, the host tier over every layer's routed stacks (qdot's, refused
    /// by name for a stack no fused kernel serves) watching the fault word
    /// and holding the host set, its batch port for passes of up to
    /// [`MAX_PASS_ROWS`] rows, and the buffers. A plan of other than one
    /// card, with an expert tier, or not of every layer is refused by name.
    /// Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        plans: &[LayerPlan],
        d: &Dims,
        o: PlacedOpen<'_>,
    ) -> Result<Placed, GpuError> {
        let n = plans.len();
        let spec = o
            .plan
            .machine
            .cards
            .first()
            .ok_or(GpuError::shape(WHAT, "a plan of no card"))?;
        if o.plan.machine.cards.len() != 1
            || !o.plan.machine.tiers.is_empty()
            || spec.layers != (0..n)
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a plan of {} cards and {} expert tiers, card 0 running layers {:?}; the \
                     program runs every layer (0..{n}) on one card with no expert tier",
                    o.plan.machine.cards.len(),
                    o.plan.machine.tiers.len(),
                    spec.layers
                ),
            ));
        }
        let shape = RouteShape::of(d)?;
        let stream = gpu.stream();
        let map = SlotMap::of_plan(o.plan, 0, None, 0..n, shape.n_expert)?;
        let ffn = plans
            .iter()
            .enumerate()
            .map(|(l, p)| placed_ffn(w, &map, shape, p, l))
            .collect::<Result<Vec<_>, _>>()?;
        let slots = DeviceTensor::upload(stream, &map.stage_view(), n, shape.n_expert)?;
        let boundary = Boundary::new(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: shape.hidden,
                n_used: shape.used,
            },
        )?;
        let widths = HostWidths {
            embd: shape.hidden,
            ff: shape.ff,
            n_used: shape.used,
        };
        let dims = model::arch::qwen35moe::host::RoutedDims {
            n_layer: n,
            n_expert: shape.n_expert,
            embd: shape.hidden,
            ff: shape.ff,
        };
        let mut experts = {
            // The NVMe expert tier of a paged plan, built by the common
            // chain and attached to the run before its first call: the
            // plan's host segments keep the mapping, every other id the
            // arena serves.
            let tier = crate::model::nvme_tier(o.plan, &o.file, o.host.r8)?;
            let mut experts = HostRun::build(o.file, 0, o.host.r8, widths, |src| {
                model::arch::qwen35moe::host::routed_layers(src, dims, 0..n, o.arch)
            })?;
            experts.attach_tier(tier)?;
            experts
        };
        experts.prepare_union(MAX_PASS_ROWS)?;
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(o.set);
        hybrid.prepare_batch(gpu.context(), MAX_PASS_ROWS)?;
        let ctx = gpu.context();
        let k = PlacedKernels {
            handoff: HandoffKernels::load(ctx)?,
            kquant: KquantKernels::load(ctx, gpu.fault_word())?,
            leg: HostLegKernels::load(ctx)?,
        };
        let placed = Placed {
            hybrid,
            side: PlacedSide {
                ffn,
                slots,
                shape,
                k,
            },
            step: PlacedRows::new(stream, shape, 1)?,
            pass: PlacedRows::new(stream, shape, MAX_PASS_ROWS)?,
            pass_hsum: DeviceBuffer::zeroed(stream, MAX_PASS_ROWS * shape.hidden)?,
        };
        let made = placed.device_bytes() - placed.hybrid.boundary().device_bytes();
        let counted = placed_bytes(d, n)?;
        if made as u64 != counted {
            return Err(GpuError::shape(
                WHAT,
                format!("the placed side holds {made} card bytes, its count is {counted}"),
            ));
        }
        Ok(placed)
    }

    /// Each layer's routed experts on the card, in layer order.
    #[must_use]
    pub fn card_counts(&self) -> Vec<usize> {
        self.side.ffn.iter().map(|f| f.n_card).collect()
    }

    /// Device bytes the placed side holds: the slot map's card copy, the
    /// buffers, the host sums and the boundary.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.side.slots.buf().num_bytes()
            + self.step.bytes()
            + self.pass.bytes()
            + self.pass_hsum.num_bytes()
            + self.hybrid.boundary().device_bytes()
    }

    /// The host tier ([`crate::host::HostTier::stats`] and the rest).
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    /// The host tier, for a gate's instrument or test seam
    /// ([`crate::host::HostTier::plant_refusal`]).
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// The host tier's settling ([`crate::host::HostTier::settle`]): the
    /// words checked back where a load leaves them; a refusal's poison the
    /// model lifts once every slot the refused call ran on has been reset
    /// ([`crate::host::HostTier::lift_refusal`]).
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.settle(stream)
    }
}

/// Device bytes [`Placed`] holds for a chain of `layers` layers over arena
/// dims `d`, beside the boundary (which the context and margin cover): what
/// a plan counts in the card's scratch for it. A chain of dense FFNs is
/// refused by name.
pub(super) fn placed_bytes(d: &Dims, layers: usize) -> Result<u64, GpuError> {
    let s = RouteShape::of(d)?;
    let rows = |r: usize| {
        let shared = if s.shared() {
            r * 4
                + r * s.ff * 4
                + (1..=r).map(|m| m * act_col_bytes(s.ff)).sum::<usize>()
                + r * s.hidden * 4
        } else {
            0
        };
        r * s.pitch * 4 + shared
    };
    let total =
        layers * s.n_expert * 4 + rows(1) + rows(MAX_PASS_ROWS) + MAX_PASS_ROWS * s.hidden * 4;
    u64::try_from(total).map_err(|_| GpuError::shape(WHAT, "the placed side's bytes pass u64"))
}

/// Layer `l`'s card experts under `map` (module doc, [`Placed::new`]).
fn placed_ffn(
    w: &Weights,
    map: &SlotMap,
    s: RouteShape,
    p: &LayerPlan,
    l: usize,
) -> Result<PlacedFfn, GpuError> {
    let n_card = map.on_card(l)?;
    let (gate, up, down) = (
        model::arch::qwen3moe::names::ffn_gate_exps(l),
        model::arch::qwen3moe::names::ffn_up_exps(l),
        model::arch::qwen3moe::names::ffn_down_exps(l),
    );
    let (h, ff) = (s.hidden, s.ff);
    let down_ty = if n_card > 0 {
        kq_site(w, &gate, n_card * ff, h, &[SiteTy::Q4K])?;
        kq_site(w, &up, n_card * ff, h, &[SiteTy::Q4K])?;
        kq_site(w, &down, n_card * h, ff, &[SiteTy::Q4K, SiteTy::Q6K])?
    } else {
        p.ffn.down_ty
    };
    let shared = match &p.ffn.route {
        FfnRoute::Router {
            shared: Some(_), ..
        } => {
            let (g, u, dn) = (
                format!("blk.{l}.ffn_gate_shexp.weight"),
                format!("blk.{l}.ffn_up_shexp.weight"),
                format!("blk.{l}.ffn_down_shexp.weight"),
            );
            kq_site(w, &g, ff, h, &[SiteTy::Q4K])?;
            kq_site(w, &u, ff, h, &[SiteTy::Q4K])?;
            let down_ty = kq_site(w, &dn, h, ff, &[SiteTy::Q4K, SiteTy::Q6K])?;
            Some(SharedFfn {
                gate: g,
                up: u,
                down: dn,
                down_ty,
            })
        }
        FfnRoute::Router { shared: None, .. } => None,
        FfnRoute::Dense => {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l} is a dense FFN; a placed load routes every layer"),
            ));
        }
    };
    if shared.is_some() != s.shared() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}'s shared expert and the router's {} slots a token of {} routed \
                 disagree",
                s.pitch, s.used
            ),
        ));
    }
    Ok(PlacedFfn {
        n_card,
        gate,
        up,
        down,
        down_ty,
        shared,
    })
}

impl HostServed for Placed {
    fn serve_captured(&mut self, chain: crate::hybrid::Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    /// The host tier's refusal poison lifted
    /// ([`crate::host::HostTier::lift_refusal`]) — the settling a reset
    /// runs is [`crate::host::HostTier::settle`]'s ([`Placed::reset`]).
    fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.hybrid.lift_refusal(stream)
    }

    /// The refusal the host tier is poisoned by now
    /// ([`crate::host::HostTier::refusal_poison`]).
    fn refusal_poison(&self) -> Option<Refusal> {
        self.hybrid.refusal_poison()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }
}

/// The FFN norm of `m` rows of `s.ffn_inp` into `normed` (the arena's own
/// when `None`) and their q8_1 form, then the router over the normed rows:
/// `norm_quant`, then the plain or the gated router — the bytes the
/// whole-card step's norm-fused router writes at one row.
fn route(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    normed: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT_R: &str = "qwen3moe placed route";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let i = s.col(m)?;
    let FfnRoute::Router { gate_inp, shared } = &n.route else {
        return Err(GpuError::shape(
            WHAT_R,
            format!("layer {}: a dense FFN has no router", c.layer),
        ));
    };
    let Arena {
        ffn_inp,
        act_ffn,
        normed: own,
        route,
        ..
    } = s;
    let normed = match normed {
        Some(b) => b,
        None => own,
    };
    gpu.fused().enqueue_norm_quant(
        stream,
        ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        c.eps,
        &mut act_ffn[i],
        normed,
        c.sink,
    )?;
    let router = f32_tensor(w, gate_inp)?;
    match shared {
        None => k
            .router
            .enqueue_fused(stream, router, normed, m, c.sink, route.plain(WHAT_R)?),
        Some(_) => k.q35(WHAT_R)?.router.enqueue_fused(
            stream,
            router,
            normed,
            m,
            c.sink,
            route.gated(WHAT_R)?,
        ),
    }
}

/// Layer `l`'s card work under the host leg over `m` rows of arena `s`
/// (module doc's shadow): the card's routed slots by the places in `rows`
/// when the layer has card experts, then the shared expert.
fn shadow(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &mut PlacedRows,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let f = side
        .ffn
        .get(c.layer)
        .ok_or(GpuError::state(WHAT, "a placed FFN for every layer"))?;
    let RouteShape {
        pitch, hidden, ff, ..
    } = side.shape;
    let i = s.col(m)?;
    if f.n_card > 0 {
        let n_slots = m * pitch;
        side.k.kquant.enqueue_gate_up_q4k(
            stream,
            &GateUpAct {
                wg: kq_weight(w, &f.gate)?,
                wu: kq_weight(w, &f.up)?,
                act: &s.act_ffn[i],
                sel: &rows.sel,
                n_slots,
                rows_per_expert: ff,
                slots_per_col: pitch,
                rule: Act::SiluMul,
            },
            c.sink,
            &mut s.h,
        )?;
        gpu.q4k_sel().enqueue_quantize_sel(
            stream,
            &QuantSel {
                x: &s.h,
                cols: 0..n_slots,
                sel: &rows.sel,
                n_card: f.n_card,
            },
            c.sink,
            &mut s.act_h[i],
        )?;
        down_sel(
            gpu,
            k,
            (f.down_ty, kq_weight(w, &f.down)?),
            (&s.act_h[i], &rows.sel),
            n_slots,
            hidden,
            &mut s.down,
        )?;
    }
    if let (Some(sh), Some(r)) = (&f.shared, rows.shared.as_mut()) {
        side.k.kquant.enqueue_gate_up_q4k(
            stream,
            &GateUpAct {
                wg: kq_weight(w, &sh.gate)?,
                wu: kq_weight(w, &sh.up)?,
                act: &s.act_ffn[i],
                sel: &r.zero,
                n_slots: m,
                rows_per_expert: ff,
                slots_per_col: 1,
                rule: Act::SiluMul,
            },
            c.sink,
            &mut r.h,
        )?;
        let act = r.act.get_mut(i).ok_or(GpuError::state(
            WHAT,
            "the shared expert's activations of m columns",
        ))?;
        gpu.enqueue_quantize_q8_1_layer(&r.h, act, c.layer)?;
        down_sel(
            gpu,
            k,
            (sh.down_ty, kq_weight(w, &sh.down)?),
            (act, &r.zero),
            m,
            hidden,
            &mut r.y,
        )?;
    }
    Ok(())
}

/// The down `_sel` of `n_slots` slots of the stack `wd` of type `ty`: slot
/// `s` the rows of expert `sel[s]` against column `s` of `act`, a [`HOST`]
/// slot left as it was.
fn down_sel(
    gpu: &Gpu,
    k: &Kernels,
    (ty, wd): (SiteTy, &DeviceTensor<u32>),
    (act, sel): (&Q8Act, &DeviceBuffer<u32>),
    n_slots: usize,
    hidden: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    match ty {
        SiteTy::Q4K => gpu
            .q4k_sel()
            .enqueue_gemv_q4k_sel(stream, wd, act, sel, n_slots, hidden, y),
        SiteTy::Q6K => k
            .q6_sel
            .enqueue_gemv_q6k_sel(stream, wd, act, sel, n_slots, hidden, y),
        other => Err(GpuError::shape(
            WHAT,
            format!("a {other:?} down: the placed load runs Q4_K and Q6_K downs"),
        )),
    }
}

/// The combine of layer `l`'s `m` rows of arena `s` with the host's sums
/// `hsum` into `out` (`None`: the arena's `x`; module doc's back).
fn combine(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &PlacedRows,
    s: &mut Arena,
    m: usize,
    hsum: &DeviceBuffer<f32>,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let f = side
        .ffn
        .get(c.layer)
        .ok_or(GpuError::state(WHAT, "a placed FFN for every layer"))?;
    let shape = side.shape;
    let Arena {
        x,
        down,
        route,
        ffn_inp,
        ..
    } = s;
    let shared = match (&f.shared, &rows.shared) {
        (Some(_), Some(r)) => Some((&r.y, shape.used)),
        (None, _) => None,
        (Some(_), None) => {
            return Err(GpuError::state(WHAT, "the shared expert's buffers"));
        }
    };
    side.k.leg.enqueue_combine(
        c.gpu.stream(),
        HostCombineArgs {
            down,
            w: route.weights(),
            sel: &rows.sel,
            hsum,
            resid: ffn_inp,
            shared,
            rows: shape.hidden,
            pitch: shape.pitch,
            n_card: f.n_card,
            m,
            fault: c.sink,
            y: out.unwrap_or(x),
        },
    )
}

/// What a placed walk reads and writes: the chain's context, its stores,
/// the arena and its input, the unit's rows, the placed side and the
/// arena's host leg buffers.
pub(super) struct WalkParts<'a, S: Stores + ?Sized> {
    pub(super) c: &'a PassCtx<'a>,
    pub(super) stores: &'a mut S,
    pub(super) s: &'a mut Arena,
    pub(super) io: &'a Io<'a>,
    pub(super) m: usize,
    pub(super) side: &'a PlacedSide,
    pub(super) rows: &'a mut PlacedRows,
}

impl<S: Stores + ?Sized> WalkParts<'_, S> {
    /// Layer `l`'s context and its mixer at the walk's rows (the embedding
    /// in front of layer 0): the front up to the FFN norm.
    fn mixer(&mut self, l: usize) -> Result<(), GpuError> {
        let c = self.c;
        let (p, st) = c
            .plans
            .get(l)
            .zip(self.stores.store(l))
            .ok_or(GpuError::state(WHAT, "a plan and a store for every layer"))?;
        let lc = Ctx::new(c.gpu, c.w, (p, l), c.k, c.mma, c.eps, c.table)?;
        dispatch::mixer(&lc, st, self.s, self.io, self.m, l == 0)
    }
}

/// Layer `l`'s context in the walk's `c`.
fn ctx_of<'a>(c: &'a PassCtx<'a>, l: usize) -> Result<Ctx<'a>, GpuError> {
    let p = c
        .plans
        .get(l)
        .ok_or(GpuError::state(WHAT, "a plan for every layer"))?;
    Ctx::new(c.gpu, c.w, (p, l), c.k, c.mma, c.eps, c.table)
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's
/// step port, then the [`Tail`] (a step's, with its taps).
pub(super) struct StepWalk<'a, S: Stores + ?Sized> {
    pub(super) p: WalkParts<'a, S>,
    pub(super) tail: Tail<'a>,
}

impl<'a, S: Stores + ?Sized> StepWalk<'a, S> {
    /// Walk every layer through `leg`, then the tail. A walk of other than
    /// one row is refused by name.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        if self.p.m != 1 || self.p.c.plans.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a step walk of {} rows over {} layers; the step is one row of a chain",
                    self.p.m,
                    self.p.c.plans.len()
                ),
            ));
        }
        let o = Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        };
        let layers = self.p.c.plans.len();
        sched::walk(o, layers, leg, &mut self)
    }
}

impl<'a, S: Stores + ?Sized> LayerProgram for StepWalk<'a, S> {
    type Port = StepLeg<'a, HostRun>;

    /// The mixer, the norm into the boundary's activation and the router,
    /// then the handoff into the page with the places, and the go.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.mixer(l)?;
        let p = &mut self.p;
        let lc = ctx_of(p.c, l)?;
        let hy = port.hybrid();
        route(&lc, &lc.p.ffn, p.s, 1, Some(hy.boundary_mut().normed_mut()))?;
        let shape = p.side.shape;
        let h = Handoff {
            ids: p.s.route.ids(),
            weights: p.s.route.weights(),
            map: p.side.slots.buf(),
            row_off: l * shape.n_expert,
            n_expert: shape.n_expert,
        };
        let stream = lc.gpu.stream();
        let target = hy.boundary_mut().handoff_target_of(0)?;
        p.side
            .k
            .handoff
            .enqueue_handoff(stream, &h, target, lc.sink, &mut p.rows.sel)?;
        hy.boundary().enqueue_go_of(stream, l, 0)
    }

    /// The card's routed experts and the shared expert.
    fn shadow(&mut self, _: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        shadow(&lc, p.side, p.rows, p.s, 1)
    }

    /// The wait, the combine — into the head's input after the last layer
    /// — the layer's tap row when taps are on, and the host tier told the
    /// layer is enqueued (an eager step is served there).
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let last = l + 1 == self.p.c.plans.len();
        let StepWalk { p, tail } = self;
        let lc = ctx_of(p.c, l)?;
        let stream = lc.gpu.stream();
        let hy = port.hybrid();
        hy.boundary().enqueue_back_of(stream, 0)?;
        let hsum = hy.boundary().hsum_of(0)?;
        match tail {
            Tail::Step { head, taps, .. } => {
                let out = last.then(|| head.input_mut());
                combine(&lc, p.side, p.rows, p.s, 1, hsum, out)?;
                if let Some(t) = taps {
                    let src: &DeviceBuffer<f32> = if last { head.input_mut() } else { &p.s.x };
                    t.rows[l].copy_from_device_async(src, stream)?;
                }
            }
            _ => combine(&lc, p.side, p.rows, p.s, 1, hsum, None)?,
        }
        port.hybrid().row_enqueued(l, 0)
    }

    /// The walk's [`Tail`].
    fn end(&mut self, _: usize) -> Result<(), GpuError> {
        let StepWalk { p, tail } = self;
        enqueue_tail(p.c, p.s, p.m, tail)
    }
}

/// A prompt unit's walk `(1, m, Batch)` of up to [`GEMV_COLS`] rows through
/// the host tier's batch port; its [`Tail`] follows the walk
/// ([`BatchWalk::walk`]).
pub(super) struct BatchWalk<'a, S: Stores + ?Sized> {
    pub(super) p: WalkParts<'a, S>,
}

impl<'a, S: Stores + ?Sized> BatchWalk<'a, S> {
    /// Walk every layer at the unit's rows through `leg`, then `tail`. A
    /// unit of no row or past [`GEMV_COLS`] rows is refused by name.
    pub(super) fn walk(
        mut self,
        leg: &mut BatchLeg<'a, HostRun>,
        mut tail: Tail<'a>,
    ) -> Result<(), GpuError> {
        let m = self.p.m;
        if m == 0 || m > GEMV_COLS || self.p.c.plans.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a unit of {m} rows over {} layers; a placed unit walks 1..={GEMV_COLS} rows",
                    self.p.c.plans.len()
                ),
            ));
        }
        let o = Overlap {
            units: 1,
            cols: m,
            port: PortKind::Batch,
        };
        let layers = self.p.c.plans.len();
        sched::walk(o, layers, leg, &mut self)?;
        let p = self.p;
        enqueue_tail(p.c, p.s, m, &mut tail)
    }
}

impl<'a, S: Stores + ?Sized> LayerProgram for BatchWalk<'a, S> {
    type Port = BatchLeg<'a, HostRun>;

    /// The mixer, the norm into the arena's `normed` and the router, the
    /// download of the unit's rows, weights and ids, and the places.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.mixer(l)?;
        let p = &mut self.p;
        let lc = ctx_of(p.c, l)?;
        let m = p.m;
        route(&lc, &lc.p.ffn, p.s, m, None)?;
        let shape = p.side.shape;
        let key = port.key(at);
        let stream = lc.gpu.stream();
        let s = &*p.s;
        port.hybrid().enqueue_download_pitched(
            stream,
            [&s.normed, s.route.weights()],
            s.route.ids(),
            shape.pitch,
            key,
        )?;
        p.side.k.leg.enqueue_places(
            stream,
            &PlacesArgs {
                ids: s.route.ids(),
                map: p.side.slots.buf(),
                row_off: l * shape.n_expert,
                n_expert: shape.n_expert,
                pitch: shape.pitch,
                used: shape.used,
                m,
            },
            lc.sink,
            &mut p.rows.sel,
        )
    }

    /// The card's routed experts and the shared expert.
    fn shadow(&mut self, _: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        shadow(&lc, p.side, p.rows, p.s, p.m)
    }

    /// The combine over the host sums the walk's serve uploaded.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        combine(&lc, p.side, p.rows, p.s, p.m, port.hsum(), None)
    }
}
