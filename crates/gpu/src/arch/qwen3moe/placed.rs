//! A placed load of the family's chain ([`Placed`]): `Body` (Qwen3-30B-A3B)
//! and `Body35` (Qwen3.6-35B-A3B) with the trunk on the card and the routed
//! experts split by a placement plan — each layer's experts the plan puts on
//! the stage card in its stacks, the ones it puts on an expert tier through
//! the common card leg (`tier_leg`, [`crate::host::leg_tier`]), the rest on
//! the host tier (`crate::host`), which computes them with qdot's fused
//! kernels while the cards run their own. A plan of no expert tier is the
//! stage card and the host alone; the load hangs at most one tier card.
//!
//! The walks are layer programs of their own over a host leg (the FFN half
//! splits around the leg, which the chain's walk does not lay out): the
//! decode step `(1, 1, Step)` through the tier's step port ([`StepWalk`],
//! captured), a pass of resident slots `(1, m, Step)` of up to
//! [`MAX_PASS_ROWS`] columns through the same port ([`ColsWalk`],
//! [`Chain::Cols`] when more than one column), a prompt unit
//! `(1, m, Batch)` of up to [`PlacedOpen::rows`] rows through its batch
//! port ([`BatchWalk`]): a unit of up to
//! [`GEMV_COLS`] rows runs the gemv arm's launches below, a longer one the
//! wide arm's (`wide::ffn_front`, `wide::ffn_card`, `wide::ffn_combine`) over
//! the layer's own map and table of the card's experts. A layer's parts:
//! - the front: the chain's walk ahead of the FFN half (`walk::front`: the
//!   embedding in front of layer 0, then the mixer, over the family's ops
//!   [`Program`]), the FFN norm with its q8_1 form and the router — the norm into the step
//!   boundary's activation, or the arena's for a unit — then the host leg's
//!   send: the step's handoff into the page with each routed slot's place
//!   and the go — a slot pass hands each column off, then one go; on a
//!   layer the tier holds experts of, the tier handoff writes the stage
//!   page and the tier image and the go waits for the tier (`tier_leg`);
//!   a unit's download of its rows and routing with its places
//!   (`hostleg`);
//! - the shadow: the card's routed experts of the layer when it has any —
//!   by the layer's leg ([`FfnLeg`], `accepts::qwen35_leg`): the dispatch's
//!   K-quant launches over the places ([`GateUpEntry`]: Q4_K or Q5_K, the
//!   table's entry for the stack's type; the down `_sel` the table names for
//!   the stack's type, [`DownArm`]: Q4_K, Q5_K or Q6_K), or the common card
//!   leg's ([`CardLeg`]: every type its arms run, the unit's wide arm its
//!   route GEMMs, [`RouteBlock`]) — the q8_1 of the card slots' rows, and on
//!   Qwen3.6 the shared expert by its types ([`SharedFfn`]): on its own
//!   stacks, a stack of one expert on a fixed route of expert 0, when the
//!   same table runs all three parts, else as its own dense FFN at its site
//!   types; a slot the host serves is [`HOST`] there and no launch reads or
//!   writes it;
//! - the back: the step's wait (a unit's serve and upload are the walk's),
//!   then the combine of the card's slots with the host's routed sum, the
//!   residual and the shared expert (`hostleg`), into the next layer's
//!   input — the head's after the last layer of the step. On a layer the
//!   tier holds experts of, the tier's slots in the same slot order
//!   (`tier_leg`) join into that sum in the one combine.
//!
//! The slot map is the plan's for the load's life while the load's
//! residency glue runs no machine ([`Placed`]'s `glue`): its card copy is
//! what the handoff and the places read, and the host tier skips the same
//! slots by the same map, so which expert runs where has one owner. Under a
//! machine the card copy moves between passes (which experts fill a layer's
//! slots, never how many: [`PlacedFfn`]'s `n_card` is the map's capacity);
//! the step and the gemv arm read it live, and each wide unit's maps are
//! first brought to it ([`PlacedSide::refresh_maps`]). The
//! shared expert is never one of the router's slots here: a placed load
//! keeps Qwen3.6's routed stacks and its shared expert apart (the
//! whole-card load folds the shared expert into its stacks as one more
//! expert, which a card holding part of a layer's experts cannot).
//!
//! A load whose arena holds more than [`GEMV_COLS`] rows (both bodies', at
//! the host union's [`PLACED_ROWS`]) reads each host expert once a unit. The
//! wide arm splits by the plan: the plain router (`Body`'s) runs no shared
//! expert, the gated one (`Body35`'s) sees it kept apart on every layer
//! ([`apart_plans`]).
//!
//! A pass of several columns ([`ColsWalk`]) and a prompt unit
//! ([`BatchWalk`]) are refused by name when any layer holds a tier expert.
//! A hung tier that holds no expert of any layer leaves the pass and the
//! unit on their existing path.
//!
//! Numerics: the routing is the whole-card chain's bit for bit (the step's
//! norm and router are `norm_quant` and the router, which the whole-card
//! step's fused launch writes bit for bit); the host's experts are qdot's
//! (q8_K activations, the host's sum order), the card's the K-quant
//! family's — the same Walk A arithmetic as the whole-card launches — so a
//! placed load agrees with the whole-card one to the error of those sums.
//! A pass of resident slots runs each slot's sequence-bound launches as the
//! whole-card pass does and each column's routed experts as the step does,
//! so each column is that slot's own step.

use super::body::{Kernels, stack_tys};
use super::dispatch::{self, Ctx, PassCtx, SharedDenseBufs};
use super::hostleg::{HostCombineArgs, HostLegKernels, PlacesArgs};
use super::plan::{
    FfnPlan, FfnRoute, GateUpEntry, LayerPlan, SharedPlan, SharedSites, SiteTy, down_entry,
    gate_up_entry, shared_entry,
};
use super::program::{Program, Stores, Tail, enqueue_tail};
use super::scratch::{Arena, Dims, Forms, Io};
use super::tier_leg::{self, TierCombineArgs, TierJoin};
use super::wide::{self, CardRoute, GEMV_COLS, HostSide, arena_bytes};
use crate::card_leg::{
    CardLeg, LegBufs, LegRun, LegShape, LegStacks, RouteBlock, RouteIo, RouteW, leg_bytes_for,
};
use crate::gemm::{Gemm32Kernels, GemmRoute, GemmTile};
use crate::host::costs::ExpertShape;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::leg_tier::{self, Hang, LegOpen, TierSide};
use crate::host::prompt_stream::PromptStream;
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::host::swap::PassReport;
use crate::host::swap_source::DEADLINE;
use crate::host::swap_source::{FileSwap, ResidencyGlue};
use crate::host::tier::TierOpen;
use crate::host::xstream::{self, LAND_BATCHES, LandBatch, RingLayer, XCfg};
use crate::host::{BatchLeg, PassKind, StepLeg, refuse_tier_count};
use crate::hybrid::{
    Boundary, BoundaryShape, Chain, HOST, HandoffLayout, HandoffTarget, HostResidency, Hybrid,
    SlotMap,
};
use crate::kquant::{Act, GateUpAct, SelDown};
use crate::model::fit::{act_col_bytes, route_bytes};
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::model::{MAX_PASS_ROWS, SlotRange};
use crate::q4k_sel::QuantSel;
use crate::quant_sel::QuantSelKernels;
use crate::site::{self, WideActs, WideForms};
use crate::tensor::{DeviceTensor, Q8Act, Window, WindowMut};
use crate::weights::{DevWeight, Weights};
use crate::{FaultSink, Gpu, GpuError};
use bloomery_levers::HostCfg;
use chain::ChainError;
use chain::arch::qwen3moe::ops::LayerCtx;
use chain::arch::qwen3moe::read::card_leg;
use chain::arch::qwen3moe::walk;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::qwen3moe::names::{ffn_down_exps, ffn_gate_exps, ffn_up_exps};
use model::moe::UNION_MAX_COLS;
use model::placement::Plan;
use model::placement::accepts::{self, LegSplit};
use model::placement::kernels::{self, DownAct, DownEntry};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const WHAT: &str = "qwen3moe placed load";

/// The rows of a placed prompt unit that reads each host expert once: the
/// host union's most columns. A placed load's prompt runs in units of up to
/// this many positions through the host tier's batch port, so its ubatch
/// arena holds that many rows ([`placed_scratch`]), whatever ubatch the
/// caller asked.
pub(super) const PLACED_ROWS: usize = UNION_MAX_COLS;

/// The rows of a placed decode step: the step arena's one row.
const STEP_ROWS: usize = 1;

/// A pass of several columns beside a tier expert. The page is one column,
/// so the rounds run a slot at a time.
const PASS_BESIDE_TIER: &str = "a pass of several slots beside an expert tier: the page is one \
                                column, so the rounds run a slot at a time";

/// A prompt unit beside a tier expert. The tier serves the step.
const PROMPT_BESIDE_TIER: &str =
    "a prompt unit beside an expert tier: the tier serves the step, one column";

/// The shared expert of a placed Qwen3.6 layer, run by its plan's types
/// ([`shared_ffn`]): the placed twin of the whole-card load's
/// `Joined`/`Apart` split.
pub(super) enum SharedFfn {
    /// All three parts are types the card kernel table's `_sel` family runs
    /// (a gate and up of one type with a [`gate_up_entry`], a down with a
    /// [`down_entry`]): its three stacks of one expert each, as the file holds
    /// them, on a fixed route of expert 0.
    Stack(SharedStack),
    /// Any other mix of the types [`shared_entry`] admits: the shared expert
    /// as its own dense FFN at its site types (`dispatch::shared_dense`).
    Sites,
}

/// A shared expert run as a stack of one expert ([`SharedFfn::Stack`]): its
/// tensors, its gate·up entry and its down's type.
pub(super) struct SharedStack {
    gate: String,
    up: String,
    down: String,
    gate_up: GateUpEntry,
    down_ty: SiteTy,
}

/// How a placed layer's card experts run: the dispatch's own launches for the
/// types the family's FFN dispatch holds, or the common card leg
/// (`crate::card_leg`) for the rest of the types its arms run
/// ([`accepts::qwen35_leg`]).
#[derive(Clone, Copy)]
enum FfnLeg {
    /// The K-quant family's gate·up at the table's entry and its down by the
    /// stack's site type ([`gate_up_sel`], [`down_sel`]).
    Dispatch {
        gate_up: GateUpEntry,
        down_ty: SiteTy,
    },
    /// The common leg's launches over the stacks' file types: the gate and up
    /// of `gate_up`, the down of `down`.
    Common { gate_up: GgmlType, down: GgmlType },
}

/// A placed layer's routed experts on the card: its slots there (the slot
/// map's capacity, fixed for the load's life: the residency machine moves
/// which experts fill them, never how many there are), the file's three
/// stacks (the card holds their rows of those slots), the leg that runs them
/// ([`FfnLeg`]); and the shared expert, on a chain that has one.
pub(super) struct PlacedFfn {
    /// The slots of the layer's card stacks: the bound a place must be below
    /// to name one, and the width of the layer's routed table.
    n_card: usize,
    gate: String,
    up: String,
    down: String,
    /// `None` on a layer with no card expert, which launches none.
    leg: Option<FfnLeg>,
    shared: Option<SharedFfn>,
}

impl PlacedFfn {
    /// The types of a layer the common leg runs, when it is one with card
    /// experts.
    fn common(&self) -> Option<[GgmlType; 2]> {
        match self.leg {
            Some(FfnLeg::Common { gate_up, down }) => Some([gate_up, down]),
            Some(FfnLeg::Dispatch { .. }) | None => None,
        }
    }
}

/// The forms a placed load's common-leg layers hold buffers for: which of the
/// leg's two down forms ([`DownAct`]) some layer's down reads, and which of the
/// wide arm's two GEMM forms its gate·up reads (the unit's normed rows) and its
/// down reads (the SwiGLU rows). A load with no common layer holds
/// [`CommonForms::NONE`] and allocates none of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CommonForms {
    act8: bool,
    blocks32: bool,
    x128: bool,
    x32: bool,
    h128: bool,
    h32: bool,
}

impl CommonForms {
    /// No common layer.
    pub(super) const NONE: CommonForms = CommonForms {
        act8: false,
        blocks32: false,
        x128: false,
        x32: false,
        h128: false,
        h32: false,
    };

    /// The forms of the layers whose `[gate·up, down]` file types are `tys`:
    /// the down's launch form ([`kernels::down`]) and each GEMM's form
    /// ([`kernels::gemm`], [`kernels::gemm_reads_blocks32`]).
    pub(super) fn of(tys: impl IntoIterator<Item = [GgmlType; 2]>) -> CommonForms {
        let mut f = CommonForms::NONE;
        for [gate_up, down] in tys {
            match kernels::down(down) {
                Some((_, DownAct::Q8Act)) => f.act8 = true,
                Some((_, DownAct::Q8Blocks32)) => f.blocks32 = true,
                None => {}
            }
            match kernels::gemm(gate_up).map(kernels::gemm_reads_blocks32) {
                Some(true) => f.x32 = true,
                Some(false) => f.x128 = true,
                None => {}
            }
            match kernels::gemm(down).map(kernels::gemm_reads_blocks32) {
                Some(true) => f.h32 = true,
                Some(false) => f.h128 = true,
                None => {}
            }
        }
        f
    }

    /// Whether any layer is a common one.
    fn any(self) -> bool {
        self != CommonForms::NONE
    }

    /// The leg's device bytes for an arena of `rows` rows of `shape`
    /// ([`LegBufs`] of each down form held), and past [`GEMV_COLS`] rows the
    /// wide arm's: the gate and up rows, and the unit's normed rows and the
    /// SwiGLU rows in the forms the GEMMs read.
    fn bytes(self, shape: RouteShape, rows: usize) -> usize {
        let leg = LegShape {
            cap: rows.min(GEMV_COLS),
            slots_per_col: shape.pitch,
            n: shape.hidden,
            ff: shape.ff,
        };
        let forms = [
            (self.act8, DownAct::Q8Act),
            (self.blocks32, DownAct::Q8Blocks32),
        ];
        let bufs: usize = forms
            .into_iter()
            .filter(|&(held, _)| held)
            .map(|(_, act)| leg_bytes_for(leg, act))
            .sum();
        if rows <= GEMV_COLS || !self.any() {
            return bufs;
        }
        let slots = rows * shape.pitch;
        bufs + 2 * slots * shape.ff * 4
            + WideForms::q8(self.x128, self.x32).bytes(rows, shape.hidden)
            + WideForms::q8(self.h128, self.h32).bytes(slots, shape.ff)
    }
}

/// The residency machine's tally shape for a placed load over arena dims
/// `d`: the ids a row routes a layer, and the rows a pass runs. The ids are
/// the step boundary's `n_used` (the router's routed slots, not the shared
/// expert's), which the step port notes slot by slot; the rows are a pass
/// of slots ([`MAX_PASS_ROWS`]). A one-row step keeps one row of that. A
/// prompt unit's rows go through the batch port, which notes no id (a
/// prompt call keeps 0 rows). A chain of dense FFNs is refused by name.
pub(super) fn tally_shape(d: &Dims) -> Result<(usize, usize), GpuError> {
    Ok((RouteShape::of(d)?.used, MAX_PASS_ROWS))
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
/// gemv width `m`) and its output; when the arena holds more than
/// [`GEMV_COLS`] rows, each layer's table over the card's experts (`None`
/// on a layer with none), and once the expert stream runs the ring pass's
/// buffers ([`PlacedRows::arm_ring`]).
pub(super) struct PlacedRows {
    sel: DeviceBuffer<u32>,
    shared: Option<SharedRows>,
    routes: Vec<Option<GemmRoute>>,
    ring: Option<RingRows>,
    /// The buffers of the load's common-leg layers; `None` on a load with none.
    common: Option<CommonRows>,
}

/// An arena's buffers for the common-leg layers ([`FfnLeg::Common`]): the
/// common card leg's own (one [`LegBufs`] per down form the load's layers
/// read, of the gemv arm's [`GEMV_COLS`] rows at most), and, past that many
/// rows, the wide arm's: the unit's normed rows
/// and the SwiGLU rows in the forms the route GEMMs read, and the gate and up
/// rows.
struct CommonRows {
    act8: Option<LegBufs>,
    blocks32: Option<LegBufs>,
    wide: Option<CommonWide>,
}

/// The wide arm's buffers of the common-leg layers ([`CommonRows`]).
struct CommonWide {
    x: WideActs,
    h: WideActs,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
}

impl CommonRows {
    /// The buffers of an arena of `rows` rows for `shape` over the forms
    /// `forms` name; `None` for a load with no common layer. Load-time only.
    fn new(
        stream: &CudaStream,
        shape: RouteShape,
        rows: usize,
        forms: CommonForms,
    ) -> Result<Option<CommonRows>, GpuError> {
        if !forms.any() {
            return Ok(None);
        }
        let leg = LegShape {
            cap: rows.min(GEMV_COLS),
            slots_per_col: shape.pitch,
            n: shape.hidden,
            ff: shape.ff,
        };
        let slots = rows * shape.pitch;
        let wide = (rows > GEMV_COLS)
            .then(|| {
                Ok::<_, GpuError>(CommonWide {
                    x: WideActs::new(
                        stream,
                        WideForms::q8(forms.x128, forms.x32),
                        rows,
                        shape.hidden,
                    )?,
                    h: WideActs::new(
                        stream,
                        WideForms::q8(forms.h128, forms.h32),
                        slots,
                        shape.ff,
                    )?,
                    g: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
                    u: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
                })
            })
            .transpose()?;
        Ok(Some(CommonRows {
            act8: forms.act8.then(|| LegBufs::new(stream, leg)).transpose()?,
            blocks32: forms
                .blocks32
                .then(|| LegBufs::for_down(stream, leg, DownAct::Q8Blocks32))
                .transpose()?,
            wide,
        }))
    }

    /// The leg's buffers a down of file type `down` reads its columns in
    /// ([`kernels::down`]'s form), refused by name when the load holds none.
    fn bufs(&mut self, down: GgmlType) -> Result<&mut LegBufs, GpuError> {
        let held = match kernels::down(down) {
            Some((_, DownAct::Q8Act)) => self.act8.as_mut(),
            Some((_, DownAct::Q8Blocks32)) => self.blocks32.as_mut(),
            None => None,
        };
        held.ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "a common-leg layer of a {down} down, whose form the load holds no buffers of"
                ),
            )
        })
    }

    /// Device bytes.
    fn bytes(&self) -> usize {
        self.act8.as_ref().map_or(0, LegBufs::bytes)
            + self.blocks32.as_ref().map_or(0, LegBufs::bytes)
            + self.wide.as_ref().map_or(0, |w| {
                w.x.bytes() + w.h.bytes() + w.g.num_bytes() + w.u.num_bytes()
            })
    }
}

/// The ring pass's buffers of a wide arena ([`ring_pass`]): the routed table
/// over a half of the expert stream's ring for the arena's slots, the union
/// places (`pitch` a row: the card's stacks' experts and the streamed ones,
/// [`HOST`] for the rest), and per landing batch of a layer's stream the
/// batch's map `n_expert + 1` wide — the stream's row, then [`HOST`] for the
/// shared expert's id, which the remap cannot do without ([`widen_map`]).
struct RingRows {
    route: GemmRoute,
    union: DeviceBuffer<u32>,
    maps: Vec<DeviceBuffer<u32>>,
    /// The union check's records while it is on ([`Placed::set_ring_check`]).
    checks: Option<Vec<RingCheck>>,
}

/// One ring pass's union places as the union check read them back
/// ([`Placed::set_ring_check`]): the layer, its routed slots on a streamed
/// expert, and the slots whose place is not the one the stack's row and the
/// ring's row give — a streamed expert's `n_card` plus its slot in the half,
/// any other expert's place in the card's stacks or [`HOST`], the shared
/// slot [`HOST`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingCheck {
    pub layer: usize,
    pub streamed: usize,
    pub wrong: usize,
}

/// The shared expert's buffers of [`PlacedRows`].
struct SharedRows {
    zero: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    act: Vec<Q8Act>,
    y: DeviceBuffer<f32>,
}

impl PlacedRows {
    /// The buffers of an arena of `rows` rows for `shape`, every place
    /// [`HOST`] until a launch writes it: the shared expert's rows for the
    /// gemv arm's widths (at most [`GEMV_COLS`]), and past that many rows the
    /// routed table of each layer over its `cards` experts. Load-time only.
    fn new(
        stream: &CudaStream,
        shape: RouteShape,
        rows: usize,
        (cards, common): (&[usize], CommonForms),
    ) -> Result<PlacedRows, GpuError> {
        if rows == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a host leg's rows for an arena of no row",
            ));
        }
        let narrow = rows.min(GEMV_COLS);
        let sel = DeviceBuffer::from_host(stream, &vec![HOST; rows * shape.pitch])?;
        let shared = shape
            .shared()
            .then(|| {
                Ok::<_, GpuError>(SharedRows {
                    zero: DeviceBuffer::zeroed(stream, narrow)?,
                    h: DeviceBuffer::zeroed(stream, narrow * shape.ff)?,
                    act: (1..=narrow)
                        .map(|m| Q8Act::with_k(stream, m, shape.ff))
                        .collect::<Result<Vec<_>, _>>()?,
                    y: DeviceBuffer::zeroed(stream, narrow * shape.hidden)?,
                })
            })
            .transpose()?;
        let routes = if rows > GEMV_COLS {
            cards
                .iter()
                .map(|&n| {
                    (n > 0)
                        .then(|| GemmRoute::new(stream, rows * shape.pitch, n))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        Ok(PlacedRows {
            sel,
            shared,
            routes,
            ring: None,
            common: CommonRows::new(stream, shape, rows, common)?,
        })
    }

    /// The ring pass's buffers for a half of `half_slots` slots of the expert
    /// stream's ring: its routed table over the arena's slots, the union
    /// places and [`LAND_BATCHES`] maps of `shape.n_expert + 1` words, every
    /// word [`HOST`] until a pass writes it. Load-time only; a second arming
    /// is refused by name.
    fn arm_ring(
        &mut self,
        stream: &CudaStream,
        shape: RouteShape,
        half_slots: usize,
    ) -> Result<(), GpuError> {
        if self.ring.is_some() {
            return Err(GpuError::state(
                WHAT,
                "pass rows the expert stream's ring is not armed on",
            ));
        }
        let slots = self.sel.len();
        let map = vec![HOST; shape.n_expert + 1];
        self.ring = Some(RingRows {
            route: GemmRoute::new(stream, slots, half_slots)?,
            union: DeviceBuffer::from_host(stream, &vec![HOST; slots])?,
            maps: (0..LAND_BATCHES)
                .map(|_| DeviceBuffer::from_host(stream, &map))
                .collect::<Result<_, _>>()?,
            checks: None,
        });
        Ok(())
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
            + self
                .routes
                .iter()
                .flatten()
                .map(GemmRoute::bytes)
                .sum::<usize>()
            + self.ring.as_ref().map_or(0, |r| {
                r.route.bytes()
                    + r.union.num_bytes()
                    + r.maps.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
            })
            + self.common.as_ref().map_or(0, CommonRows::bytes)
    }
}

/// The kernels of the host leg: the handoff into the page, the K-quant
/// family's gate·up, and this family's places and combine (`hostleg`); and,
/// on a side whose units run the wide arm (past [`GEMV_COLS`] rows), the
/// kernels that build a layer's routed table through its map, which a K-quant
/// chain's kernel set does not hold.
pub(super) struct PlacedKernels {
    handoff: HandoffKernels,
    kquant: QuantSelKernels,
    leg: HostLegKernels,
    g32: Option<Gemm32Kernels>,
    /// The common card leg's modules, on a load with a common-leg layer.
    card: Option<CardLeg>,
    /// The tier join, on a load that hangs an expert tier.
    join: Option<TierJoin>,
}

/// What every placed walk reads besides the arena: each layer's card
/// experts, the slot map's card copy, the route's shape and the kernels, the
/// most rows of a unit, and — past [`GEMV_COLS`] of them — each layer's map
/// of the experts to their places in the card's stacks (the router's expert
/// count and one entry for the shared expert's id, [`HOST`]). A map is the
/// card copy's row with that entry after it, which the wide arm's remap
/// cannot do without (an id at the router's expert count faults it): under a
/// running residency machine the card copy moves between passes and the maps
/// are brought to it at the head of each wide unit
/// ([`PlacedSide::refresh_maps`]); the step and the gemv arm read the card
/// copy itself.
pub(super) struct PlacedSide {
    ffn: Vec<PlacedFfn>,
    slots: Arc<DeviceTensor<u32>>,
    maps: Vec<DeviceBuffer<u32>>,
    /// Whether a residency machine runs, so the card copy moves.
    live: bool,
    shape: RouteShape,
    k: PlacedKernels,
    /// The stage card's side of an expert tier, when the plan hangs one.
    /// Its places are the handoff's and sit outside [`Placed::device_bytes`]:
    /// the plan's scratch is the stage side's.
    tier: Option<TierSide>,
    /// The rows of the most a unit walks.
    pub(super) cap: usize,
}

impl PlacedSide {
    /// Each layer's map brought to the card copy's row of that layer, in
    /// stream order ahead of the unit's first read: one device-to-device copy
    /// of the router's expert count words a layer, the map's last entry (the
    /// shared expert's id, [`HOST`]) left as it is. A unit of at most
    /// [`GEMV_COLS`] rows reads no map, and a load that runs no machine keeps
    /// the plan's map for its life: neither enqueues a copy. The machine
    /// writes the card copy on the engine stream at its boundaries
    /// (`SwapMachine::boundary`, `reset`), and a prompt call takes no
    /// boundary inside it, so a map read after this call is the row its
    /// unit's places and the host's skip set read.
    pub(super) fn refresh_maps(&mut self, stream: &CudaStream, m: usize) -> Result<(), GpuError> {
        for l in 0..self.maps.len() {
            self.refresh_map(stream, m, l)?;
        }
        Ok(())
    }

    /// Layer `l`'s map brought to the card copy's row of the layer, as
    /// [`PlacedSide::refresh_maps`] brings each: in a streaming prompt call
    /// the layer's pick moves the card copy's row in the walk, and the map
    /// follows it before the layer's card route reads it. Nothing at a unit
    /// the map is not read for or on a load that runs no machine. A layer
    /// past the maps is refused by name.
    fn refresh_map(&mut self, stream: &CudaStream, m: usize, l: usize) -> Result<(), GpuError> {
        if !self.live || m <= GEMV_COLS {
            return Ok(());
        }
        let n = self.shape.n_expert;
        let PlacedSide { slots, maps, .. } = self;
        let map = maps.get_mut(l).ok_or(GpuError::state(
            WHAT,
            "a map of the card's experts for every layer of a wide unit",
        ))?;
        widen_map(stream, (slots.buf(), l * n), n, map)
    }

    /// Whether layer `l` holds routed experts on the card: the layers a
    /// streaming prompt call picks for and reads.
    fn has_card(&self, l: usize) -> bool {
        self.ffn.get(l).is_some_and(|f| f.n_card > 0)
    }
}

/// The placed side of a load: the host tier over every layer's routed
/// experts, what the walks read, and the step's and the pass arena's host
/// leg buffers, with the host sums a unit's upload lands in (a row of the
/// model's width for each of the unit's most rows), and the load's residency
/// glue ([`ResidencyGlue`]), which carries the residency calls of the shared
/// [`HostServed`](crate::model::HostServed) impl ([`TierBody`]) when it holds
/// a machine.
pub struct Placed {
    pub(super) hybrid: Hybrid<HostRun>,
    pub(super) side: PlacedSide,
    pub(super) step: PlacedRows,
    pub(super) pass: PlacedRows,
    pub(super) pass_hsum: DeviceBuffer<f32>,
    glue: ResidencyGlue,
}

/// What [`Placed::wide_route`] read back: the unit's routed ids, the places
/// the last walked layer's front cut, and each layer's card-expert side, for
/// gates and diagnostics.
pub struct WideRoute {
    /// The unit's routed ids, token-major at the router's pitch (the shared
    /// expert's id last a token).
    pub ids: Vec<u32>,
    /// The last walked layer's places of the unit's slots, token-major at the
    /// router's pitch ([`HOST`] for the host's and the shared expert's).
    pub sel: Vec<u32>,
    /// Each layer's side, in layer order.
    pub layers: Vec<WideRouteLayer>,
}

/// One layer's side of a [`WideRoute`]: its card expert count, its map of the
/// experts to their card slots or [`HOST`], and its routed table as the last
/// unit's card route left it.
pub struct WideRouteLayer {
    /// The layer's card experts.
    pub n_card: usize,
    /// The layer's map, `n_expert + 1` words (the last the shared expert's
    /// [`HOST`], which the wide arm's remap skips).
    pub map: Vec<u32>,
    /// The table's slot list and tiles ([`GemmRoute::read_back`]).
    pub cols: Vec<u32>,
    pub tiles: Vec<GemmTile>,
    /// The slots the table refused (an expert id past the card set).
    pub refused: usize,
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
    /// The load's residency side: [`ResidencyGlue::off`] on a load that runs
    /// no machine.
    pub(super) glue: ResidencyGlue,
    /// The architecture the host run's refusals name.
    pub(super) arch: &'static str,
    /// The tier node service, when the plan hangs an expert tier.
    pub(super) family: &'static crate::host::node::NodeFamily,
    /// The most rows a prompt unit holds: the arena's, [`PLACED_ROWS`] on a
    /// load that runs the wide arm.
    pub(super) rows: usize,
}

impl Placed {
    /// The placed side of a load of `plans` over arena dims `d`, from `o`:
    /// the plan's slot map ([`SlotMap::of_plan`]) and its card copy, each
    /// layer's card stacks checked against its card count and the types the
    /// launches take ([`stack_tys`]: the table's gate·up and down entries the
    /// FFN dispatch launches; a stack of other rows would leave slots that
    /// neither side sums), the shared expert of a gated router's chain
    /// admitted by its plan's types ([`shared_ffn`]), the step boundary of one
    /// row of [`MAX_PASS_ROWS`] columns, the host tier over every layer's
    /// routed stacks (qdot's, refused by name for a stack no fused kernel
    /// serves) watching the fault word and holding the host set, its batch
    /// port for units of up to `o.rows` rows, and the buffers (past
    /// [`GEMV_COLS`] rows each layer's map and table of its card experts).
    /// The glue is given the host run's NVMe tier
    /// ([`ResidencyGlue::attach_tier`]), and the machine it may hold is
    /// started by the body once its pieces are sized
    /// ([`HostServed::start_residency`](crate::model::HostServed::start_residency)).
    /// A plan of other than one stage card, of more than one expert tier, or
    /// not of every layer is refused by name. One expert tier is hung under
    /// the host tier ([`leg_tier::hang_tiers`]). Load-time only.
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
        let n_tiers = o.plan.machine.expert_tiers().count();
        refuse_tier_count(WHAT, n_tiers)?;
        if o.plan.machine.cards.len() != 1 || spec.layers != (0..n) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a plan of {} cards and {} expert tiers, card 0 running layers {:?}; the \
                     program runs every layer (0..{n}) on one stage card and at most one expert \
                     tier",
                    o.plan.machine.cards.len(),
                    n_tiers,
                    spec.layers
                ),
            ));
        }
        let shape = RouteShape::of(d)?;
        let stream = gpu.stream();
        let tier_cards: Vec<usize> = (0..n_tiers)
            .map(|i| o.plan.machine.cards.len() + i)
            .collect();
        let map = SlotMap::of_plan_tiers(o.plan, 0, &tier_cards, 0..n, shape.n_expert)?;
        let ffn = plans
            .iter()
            .enumerate()
            .map(|(l, p)| placed_ffn(w, &map, shape, p, l))
            .collect::<Result<Vec<_>, _>>()?;
        let stage = map.stage_view();
        let slots = Arc::new(DeviceTensor::upload(stream, &stage, n, shape.n_expert)?);
        let cards: Vec<usize> = ffn.iter().map(|f| f.n_card).collect();
        let common = CommonForms::of(ffn.iter().filter_map(PlacedFfn::common));
        let live = o.glue.source().is_some();
        let maps = if o.rows > GEMV_COLS {
            layer_maps(stream, &stage, n, shape.n_expert)?
        } else {
            Vec::new()
        };
        let boundary = Boundary::with_cols(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: shape.hidden,
                n_used: shape.used,
            },
            1,
            MAX_PASS_ROWS,
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
        // The common chain's host-run build: a paged plan's NVMe expert tier
        // is attached before the run's first call, the plan's host segments
        // keep the mapping and every other id the arena serves.
        let mut experts = crate::model::host_run(
            gpu.context(),
            o.plan,
            &o.file,
            0..n,
            o.host.r8,
            o.set.arena(),
            widths,
            |src| model::arch::qwen35moe::host::routed_layers(src, dims, 0..n, o.arch),
        )?;
        o.glue.attach_tier(experts.tier())?;
        experts.prepare_union(o.rows)?;
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(o.set);
        // The tier page is one column. The batch port's width is the placed
        // load's unit rows, so the plan's tier reserve is checked at that
        // width. Empty of a tier, the hang is nothing and the batch port is
        // made once, as a load of no tier.
        if n_tiers == 1 {
            let tiers = TierOpen::of_machine(o.plan.machine)?;
            let hang = Hang {
                tiers: &tiers,
                family: o.family,
                open: LegOpen {
                    n_expert: shape.n_expert,
                    layers: n,
                    hidden: shape.hidden,
                    n_used: shape.used,
                    rows: tier_leg::PAGE_ROWS,
                },
                cols: o.rows,
                card_dontneed: o.host.card_dontneed,
                bytes: crate::model::CardBytes::File,
            };
            leg_tier::hang_tiers((gpu, &o.file, o.plan), &hang, &mut hybrid)?;
        }
        hybrid.prepare_batch(gpu.context(), o.rows)?;
        let tier = if n_tiers == 1 {
            Some(TierSide::new(
                gpu,
                hybrid.slots(),
                n,
                (tier_leg::PAGE_ROWS, shape.used),
            )?)
        } else {
            None
        };
        let ctx = gpu.context();
        let k = PlacedKernels {
            handoff: HandoffKernels::load(ctx)?,
            kquant: QuantSelKernels::load(ctx, gpu.fault_word())?,
            leg: HostLegKernels::load(ctx)?,
            g32: (o.rows > GEMV_COLS)
                .then(|| Gemm32Kernels::load(ctx))
                .transpose()?,
            card: common.any().then(|| CardLeg::load(gpu)).transpose()?,
            join: (n_tiers == 1).then(|| TierJoin::load(ctx)).transpose()?,
        };
        let placed = Placed {
            hybrid,
            side: PlacedSide {
                ffn,
                slots,
                maps,
                live,
                shape,
                k,
                tier,
                cap: o.rows,
            },
            step: PlacedRows::new(stream, shape, STEP_ROWS, (&[], common))?,
            pass: PlacedRows::new(stream, shape, o.rows, (&cards, common))?,
            pass_hsum: DeviceBuffer::zeroed(stream, o.rows * shape.hidden)?,
            glue: o.glue,
        };
        let made = placed.device_bytes() - placed.hybrid.boundary().device_bytes();
        let tables: usize = cards
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| route_bytes(o.rows * shape.pitch, c))
            .sum();
        let counted = placed_total(
            d,
            n,
            o.rows,
            (if o.rows > GEMV_COLS { tables } else { 0 }, common),
        )?;
        if made as u64 != counted {
            return Err(GpuError::shape(
                WHAT,
                format!("the placed side holds {made} card bytes, its count is {counted}"),
            ));
        }
        Ok(placed)
    }

    /// Start the expert stream for `split` ([`HostTier::xstream_start`]: the
    /// ring sized from the card's free bytes, the lane's probe, the costs
    /// this device and host price for the routed expert's shape over the
    /// router's experts and routed slots) and arm the pass
    /// rows for its ring ([`PlacedRows::arm_ring`]); a started stream and
    /// armed rows are left as they are. Refused by name: a pass arena of the
    /// gemv arm alone (the ring pass is the wide arm's), a load that runs no
    /// residency machine and the stream's refusals, the card's room among
    /// them. Load-time only.
    ///
    /// [`HostTier::xstream_start`]: crate::host::HostTier::xstream_start
    pub(super) fn start_xstream(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let shape = self.side.shape;
        if let Some(l) = self.side.ffn.iter().position(|f| f.common().is_some()) {
            return Err(GpuError::shape(
                "BLOOMERY_XSTREAM",
                format!(
                    "split on a placed load whose layer {l} runs its card experts through the \
                     common leg: the ring pass reads the dispatch's K-quant words only"
                ),
            ));
        }
        if self.side.cap <= GEMV_COLS {
            return Err(GpuError::shape(
                "BLOOMERY_XSTREAM",
                format!(
                    "split on a placed load whose units walk at most {} rows: the ring pass is \
                     the wide arm's, past {GEMV_COLS} rows",
                    self.side.cap
                ),
            ));
        }
        if self.hybrid.xstream().is_none() {
            // The plan reserves no ring: it takes what the card's margin leaves past the keep-free.
            let expert = ExpertShape {
                hidden: shape.hidden,
                ff: shape.ff,
                top_k: shape.used,
                units: 1,
            };
            let cfg = XCfg::new(expert, shape.n_expert, DEADLINE, None);
            self.hybrid
                .xstream_start(gpu.context(), gpu.stream(), cfg, xstream::PROBE)?;
        }
        if self.pass.ring.is_none() {
            let half = self
                .hybrid
                .xstream()
                .map_or(0, xstream::XStream::half_slots);
            self.pass.arm_ring(gpu.stream(), shape, half)?;
        }
        Ok(())
    }

    /// Whether each ring pass reads its union places back against the
    /// stack's row and the ring's ([`RingCheck`], taken by
    /// [`Placed::take_ring_checks`]): off at the arming. A debug view, for
    /// the residency gate: on, every streamed layer's ring pass synchronizes
    /// the engine stream. Refused by name before `split` armed the ring.
    pub fn set_ring_check(&mut self, on: bool) -> Result<(), GpuError> {
        let ring = self.pass.ring.as_mut().ok_or(GpuError::state(
            WHAT,
            "pass rows armed for the expert stream's ring (Placed::start_xstream)",
        ))?;
        ring.checks = on.then(Vec::new);
        Ok(())
    }

    /// The union check's records since the last take, in the order the ring
    /// passes ran ([`Placed::set_ring_check`]); empty while it is off.
    pub fn take_ring_checks(&mut self) -> Vec<RingCheck> {
        self.pass
            .ring
            .as_mut()
            .and_then(|r| r.checks.as_mut())
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Each layer's routed experts on the card, in layer order.
    #[must_use]
    pub fn card_counts(&self) -> Vec<usize> {
        self.side.ffn.iter().map(|f| f.n_card).collect()
    }

    /// The wide arm's card route as the last prompt unit left it, read back
    /// with `ids` (the ubatch arena's routed ids) for gates and diagnostics:
    /// every layer's map of the experts to their card slots or [`HOST`], its
    /// routed table ([`GemmRoute::read_back`], [`GemmRoute::refused_back`])
    /// and the places the last walked layer's front cut. A layer with no card
    /// expert holds the empty table (its oracle's). Blocking reads on
    /// `stream`. Refused by name on a load whose pass arena holds no wide
    /// tables (no unit past [`GEMV_COLS`] ran), a layer the load holds no map
    /// for, and a layer with card experts whose table is missing.
    pub fn wide_route(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
    ) -> Result<WideRoute, GpuError> {
        if self.pass.routes.is_empty() {
            return Err(GpuError::state(
                WHAT,
                "the pass rows hold no wide tables (no unit past the gemv arm's cut ran)",
            ));
        }
        let missing = |what: &'static str| GpuError::state(WHAT, what);
        let mut layers = Vec::with_capacity(self.side.ffn.len());
        for (l, f) in self.side.ffn.iter().enumerate() {
            let map = self
                .side
                .maps
                .get(l)
                .ok_or_else(|| missing("a map of the card's experts for every layer"))?
                .to_host_vec(stream)?;
            let route = self.pass.routes.get(l).and_then(Option::as_ref);
            let (cols, tiles, refused) = match route {
                Some(route) => {
                    let (cols, tiles) = route.read_back(stream)?;
                    (cols, tiles, route.refused_back(stream)?.len())
                }
                None if f.n_card == 0 => (Vec::new(), Vec::new(), 0),
                None => return Err(missing("a table of the card's experts for every layer")),
            };
            layers.push(WideRouteLayer {
                n_card: f.n_card,
                map,
                cols,
                tiles,
                refused,
            });
        }
        Ok(WideRoute {
            ids: ids.to_host_vec(stream)?,
            sel: self.pass.sel.to_host_vec(stream)?,
            layers,
        })
    }

    /// Some layer holds an expert on the hung expert tier ([`TierSide::any`]):
    /// the load runs no pass of several slots beside one — the tier's page is
    /// one column, so its rounds run a slot at a time — and a hung tier that
    /// holds no expert leaves the pass on its path.
    #[must_use]
    pub fn tier_any(&self) -> bool {
        self.side.tier.as_ref().is_some_and(TierSide::any)
    }

    /// Device bytes the placed side holds: the slot map's card copy, the
    /// layers' maps, the buffers, the host sums and the boundary. The tier
    /// side's places are the handoff's and sit outside this count: the plan's
    /// scratch is the stage side's.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.side.slots.buf().num_bytes()
            + self
                .side
                .maps
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.step.bytes()
            + self.pass.bytes()
            + self.pass_hsum.num_bytes()
            + self.hybrid.boundary().device_bytes()
    }

    /// Refuses by name a placed side that holds more card bytes (the boundary
    /// beside it, which the context and margin cover, left out) than `counted`,
    /// the bytes the plan counts for it ([`placed_bytes`]): a side past its
    /// count would take card bytes the plan's scratch does not hold.
    pub(super) fn within(&self, counted: u64) -> Result<(), GpuError> {
        let held = self.device_bytes() - self.hybrid.boundary().device_bytes();
        if u64::try_from(held)
            .map_err(|_| GpuError::shape(WHAT, "the placed side's bytes pass u64"))?
            > counted
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the placed side holds {held} card bytes, past the {counted} the plan counts for it"
                ),
            ));
        }
        Ok(())
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

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.glue.source()
    }

    /// Keep the residency boundaries' reports from here on
    /// ([`ResidencyGlue::log_passes`]): `passes` the most boundaries between
    /// two [`Placed::take_residency_passes`]. Nothing is kept until asked.
    pub fn log_residency(&mut self, passes: usize) {
        self.glue.log_passes(passes);
    }

    /// The residency boundaries' reports since the last take, in order, each
    /// with the kind of the pass it ended.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.glue.take_passes()
    }

    /// The layers whose wide map is not the card copy's row of that layer
    /// followed by [`HOST`], read back on `stream`: none on a map the
    /// refresh keeps ([`PlacedSide::refresh_maps`]) and on a load whose
    /// units are at most [`GEMV_COLS`] rows (it holds no map). A debug view,
    /// for the residency gate: after a wide unit under a machine every map
    /// is its row.
    pub fn stale_maps(&self, stream: &CudaStream) -> Result<Vec<usize>, GpuError> {
        let n = self.side.shape.n_expert;
        let copy = self.side.slots.buf().to_host_vec(stream)?;
        let mut stale = Vec::new();
        for (l, map) in self.side.maps.iter().enumerate() {
            let held = map.to_host_vec(stream)?;
            let row = copy.get(l * n..(l + 1) * n);
            let is_row = held.len() == n + 1
                && held.last() == Some(&HOST)
                && row.is_some_and(|r| held[..n] == *r);
            if !is_row {
                stale.push(l);
            }
        }
        Ok(stale)
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
/// dims `d` with units of up to `rows` rows, beside the boundary (which the
/// context and margin cover): what a plan counts in the card's scratch for
/// it. Past [`GEMV_COLS`] rows each layer's table over its card experts is
/// counted at the larger of the file's expert count and one expert, the
/// bound the plan's expert split (which this count is an input to) cannot
/// know; [`Placed::new`] checks the load against the count at its split. The
/// common-leg layers `common` names have their buffers counted beside the
/// rest, at the forms the file's types hold (a load whose plan gives none of
/// a form's layers card experts holds fewer, which [`Placed::new`] checks at
/// its split). A chain of dense FFNs is refused by name.
pub(super) fn placed_bytes(
    d: &Dims,
    layers: usize,
    rows: usize,
    common: CommonForms,
) -> Result<u64, GpuError> {
    let s = RouteShape::of(d)?;
    let slots = rows * s.pitch;
    let table = route_bytes(slots, s.n_expert).max(route_bytes(slots, 1));
    placed_total(
        d,
        layers,
        rows,
        (if rows > GEMV_COLS { layers * table } else { 0 }, common),
    )
}

/// [`placed_bytes`] with the layers' tables counted as `tables` bytes in all
/// (past [`GEMV_COLS`] rows), and the common-leg layers' buffers of `common`'s
/// forms in the step's and the pass's arena.
fn placed_total(
    d: &Dims,
    layers: usize,
    rows: usize,
    (tables, common): (usize, CommonForms),
) -> Result<u64, GpuError> {
    let s = RouteShape::of(d)?;
    let shared = |r: usize| {
        if s.shared() {
            r * 4
                + r * s.ff * 4
                + (1..=r).map(|m| m * act_col_bytes(s.ff)).sum::<usize>()
                + r * s.hidden * 4
        } else {
            0
        }
    };
    let leg = |sel: usize, narrow: usize| sel * s.pitch * 4 + shared(narrow);
    let maps = if rows > GEMV_COLS {
        layers * (s.n_expert + 1) * 4 + tables
    } else {
        0
    };
    let total = layers * s.n_expert * 4
        + leg(1, 1)
        + leg(rows, rows.min(GEMV_COLS))
        + rows * s.hidden * 4
        + maps
        + common.bytes(s, STEP_ROWS)
        + common.bytes(s, rows);
    u64::try_from(total).map_err(|_| GpuError::shape(WHAT, "the placed side's bytes pass u64"))
}

/// Rows of the ubatch arena for ubatches of `u` tokens on a `ctx`-row cache:
/// at least a pass's [`GEMV_COLS`] (the plan's passes and tail walk it too),
/// at most the cache.
pub(super) fn prompt_rows(u: usize, ctx: usize) -> usize {
    u.max(GEMV_COLS).min(ctx)
}

/// The card bytes a placed load of `layers` layers over arena dims `d`, forms
/// `forms` and common-leg forms `common` holds past the m = 1 scratch: its
/// ubatch arena at the placed units' rows ([`PLACED_ROWS`], [`arena_bytes`])
/// and its placed side ([`placed_bytes`]) — what the plan's machine counts in
/// the card's scratch (`model::placement::workstation::placed_with_arena`),
/// and what [`counted_arena`] holds the load to.
pub(super) fn placed_scratch(
    d: &Dims,
    layers: usize,
    (forms, common): (Forms, CommonForms),
) -> Result<u64, GpuError> {
    let arena = arena_bytes(d, prompt_rows(PLACED_ROWS, d.ctx), forms) as u64;
    Ok(arena + placed_bytes(d, layers, PLACED_ROWS, common)?)
}

/// The ubatch arena of `d` for ubatches of `u` tokens on a placed load,
/// after the check that it and the placed side's `side` bytes
/// ([`placed_bytes`]) fit the `counted` bytes the load's plan holds for them
/// in the card's scratch — not the card's free bytes, which a card budget
/// (`BLOOMERY_CARD_BUDGET`) does not bound — else a named refusal with both
/// terms, before anything is allocated. Load-time allocation.
pub(super) fn counted_arena(
    stream: &CudaStream,
    (d, forms): (Dims, Forms),
    u: usize,
    counted: u64,
    side: u64,
) -> Result<Arena, GpuError> {
    const WHAT_FIT: &str = "qwen3moe::placed::counted_arena";
    let rows = prompt_rows(u, d.ctx);
    let need = arena_bytes(&d, rows, forms) as u64;
    if need + side > counted {
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena for ubatches of {u} tokens ({rows} rows) needs {need} bytes and \
                 the placed side {side}; the plan counts {counted}"
            ),
        ));
    }
    let a = Arena::with(stream, d, rows, forms)?;
    if a.bytes() as u64 != need {
        return Err(GpuError::shape(
            WHAT_FIT,
            format!(
                "the ubatch arena holds {} bytes, its fit check counted {need}",
                a.bytes()
            ),
        ));
    }
    Ok(a)
}

/// `map`'s first `n` words brought to the `n` words of `src` from word `at`
/// — a layer's row of the slot map's card copy, or a landing batch's row of
/// the expert stream — in stream order on `stream`, its last entry (the
/// shared expert's id, [`HOST`]) left as it is: the `n + 1` map the wide
/// arm's remap reads, which faults on an id at the router's expert count.
/// One device-to-device copy. A map of other than `n + 1` words is refused
/// by name.
fn widen_map(
    stream: &CudaStream,
    (src, at): (&DeviceBuffer<u32>, usize),
    n: usize,
    map: &mut DeviceBuffer<u32>,
) -> Result<(), GpuError> {
    if map.len() != n + 1 {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a map of {} words for {n} experts and the shared expert's id",
                map.len()
            ),
        ));
    }
    let row = Window::<u32>::of(src, at * size_of::<u32>(), n)?;
    let mut head = WindowMut::<u32>::of_mut(map, 0, n)?;
    head.copy_from_device_async(&row, stream)?;
    Ok(())
}

/// Each layer's map of the experts to their places in the card's stacks:
/// `stage` the slot map's card copy (`n_expert` places a layer, [`HOST`]
/// for the host's), each row with one more entry, [`HOST`], for the shared
/// expert's id. The rows are the load's: a load that runs a residency
/// machine brings them to the card copy before every wide unit
/// ([`PlacedSide::refresh_maps`]). Load-time only.
fn layer_maps(
    stream: &CudaStream,
    stage: &[u32],
    layers: usize,
    n_expert: usize,
) -> Result<Vec<DeviceBuffer<u32>>, GpuError> {
    if stage.len() != layers * n_expert {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the slot map's card copy holds {} places, {layers} layers of {n_expert} \
                 experts hold {}",
                stage.len(),
                layers * n_expert
            ),
        ));
    }
    stage
        .chunks(n_expert)
        .map(|row| {
            let map: Vec<u32> = row.iter().copied().chain([HOST]).collect();
            DeviceBuffer::from_host(stream, &map)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// `plans` with every routed layer's shared expert kept apart from the
/// routed stacks: a layer the plan folds into its stacks
/// ([`SharedPlan::Joined`], whose three types the stacks and the shared
/// expert share) names the file's routed stacks and the shared expert's own
/// three, at those types, as a layer of other types does. A placed load joins
/// no stacks (the card holds a layer's routed experts in the file's own, the
/// shared expert on its own), so the wide arm reads the plan this makes: a
/// fact of the placed load, made once in `Pre35::read_placed` for both the
/// plan's count (`GpuModel::placed_arena_bytes`) and the open, so on a file
/// whose layers are all folded the arena bytes the plan counts are the bytes
/// the open allocates.
pub(super) fn apart_plans(plans: Vec<LayerPlan>) -> Vec<LayerPlan> {
    plans
        .into_iter()
        .enumerate()
        .map(|(l, plan)| {
            let LayerPlan {
                mixer,
                ffn,
                residual,
            } = plan;
            let ffn = match ffn.route {
                FfnRoute::Router {
                    gate_inp,
                    shared: Some(SharedPlan::Joined),
                } => FfnPlan {
                    route: FfnRoute::Router {
                        gate_inp,
                        shared: Some(SharedPlan::Apart(SharedSites {
                            gate: format!("blk.{l}.ffn_gate_shexp.weight"),
                            up: format!("blk.{l}.ffn_up_shexp.weight"),
                            down: format!("blk.{l}.ffn_down_shexp.weight"),
                            gate_ty: ffn.gate_ty,
                            up_ty: ffn.up_ty,
                            down_ty: ffn.down_ty,
                        })),
                    },
                    gate: ffn_gate_exps(l),
                    up: ffn_up_exps(l),
                    down: ffn_down_exps(l),
                    ..ffn
                },
                route => FfnPlan { route, ..ffn },
            };
            LayerPlan {
                mixer,
                ffn,
                residual,
            }
        })
        .collect()
}

/// Layer `l`'s routed stacks `[gate, up, down]` of `n` experts of `ff` rows of
/// `h` values, each a resident word plane of its shape ([`DevWeight::KQuant`]:
/// the file's blocks), as their file types: the types the placed card leg's
/// classifier reads ([`accepts::qwen35_leg`]). A stack that is not one, or of
/// another shape, is refused by name with the layer, `what` naming the entry
/// point that read the stacks.
pub(super) fn stack_files(
    w: &Weights,
    names: [&str; 3],
    (n, ff, h): (usize, usize, usize),
    l: usize,
    what: &'static str,
) -> Result<[GgmlType; 3], GpuError> {
    let stack = |name: &str, rows: usize, k: usize| -> Result<GgmlType, GpuError> {
        let Some(DevWeight::KQuant { ty, w: t, k: wk }) = w.get(name) else {
            return Err(GpuError::tensor(
                what,
                name,
                "a resident K-quant word plane",
            ));
        };
        if t.rows() != rows || *wk != k {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {l}: {name} is {ty} {} rows x {wk}; the chain takes {rows} rows x {k}",
                    t.rows()
                ),
            ));
        }
        Ok(*ty)
    };
    let [gate, up, down] = names;
    Ok([
        stack(gate, n * ff, h)?,
        stack(up, n * ff, h)?,
        stack(down, n * h, ff)?,
    ])
}

/// The forms the layers the common card leg runs ([`LegSplit::Common`]) hold
/// buffers of, from `file`'s header: `layers` are each routed layer with its
/// expert count, `ff` and `h` the width of an expert's stacks. A load whose
/// layers the dispatch's launches run alone holds [`CommonForms::NONE`].
pub(super) fn common_forms(
    file: &Split,
    what: &'static str,
    layers: impl IntoIterator<Item = (usize, usize)>,
    (ff, h): (usize, usize),
) -> Result<CommonForms, GpuError> {
    let mut tys = Vec::new();
    for (l, n) in layers {
        if let Some((LegSplit::Common, [gate_up, _, down])) = card_leg(file, what, l, (ff, h, n))? {
            tys.push([gate_up, down]);
        }
    }
    Ok(CommonForms::of(tys))
}

/// Layer `l`'s card stacks under `map` (module doc, [`Placed::new`]): their
/// slots are the map's capacity, which a residency machine's evictions and
/// admissions leave as it is (`SlotMap::evict`), where the live count drops
/// by the spare slots the machine empties at its start.
fn placed_ffn(
    w: &Weights,
    map: &SlotMap,
    s: RouteShape,
    p: &LayerPlan,
    l: usize,
) -> Result<PlacedFfn, GpuError> {
    let n_card = map.capacity(l)?;
    let (gate, up, down) = (ffn_gate_exps(l), ffn_up_exps(l), ffn_down_exps(l));
    let (h, ff) = (s.hidden, s.ff);
    let leg = if n_card > 0 {
        let tys = stack_files(w, [&gate, &up, &down], (n_card, ff, h), l, WHAT)?;
        Some(match accepts::qwen35_leg(tys) {
            Some(LegSplit::Dispatch) => {
                let [g, u, d] = stack_tys(w, [&gate, &up, &down], (n_card, ff, h), l)?;
                FfnLeg::Dispatch {
                    gate_up: gate_up_entry(g, u, WHAT)?,
                    down_ty: d,
                }
            }
            Some(LegSplit::Common) => FfnLeg::Common {
                gate_up: tys[0],
                down: tys[2],
            },
            None => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}'s card stacks are {tys:?} (gate, up, down): no card leg runs \
                         them (`accepts::qwen35_leg`)"
                    ),
                ));
            }
        })
    } else {
        None
    };
    let shared = match &p.ffn.route {
        FfnRoute::Router {
            shared: Some(SharedPlan::Apart(sh)),
            ..
        } => Some(shared_ffn(w, sh, (ff, h), l)?),
        FfnRoute::Router {
            shared: Some(SharedPlan::Joined),
            ..
        } => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}'s shared expert is folded into its stacks; a placed load keeps it \
                     apart (apart_plans)"
                ),
            ));
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
        leg,
        shared,
    })
}

/// How layer `l`'s shared expert `sh` (the plan's [`SharedSites`], the file
/// header's types) runs on the placed side, after its admission: each part's
/// type is one [`shared_entry`] names a dense site for, refused by name with
/// the layer, and each part is resident as that type at its shape
/// (`site::site`; a placed load's weights are checked here, not by
/// `check_resident`).
///
/// The rule is the placed twin of the whole-card `Joined`/`Apart` split: a
/// shared expert the `_sel` table runs — a gate and up of one type with a
/// [`gate_up_entry`] and a down with a [`down_entry`], which are the K-quant
/// word planes today's stack of one expert reads — keeps that path, so a
/// Q4_K_M placed load launches what it did; any other mix takes the dense
/// FFN at its site types ([`SharedFfn::Sites`]).
fn shared_ffn(
    w: &Weights,
    sh: &SharedSites,
    (ff, h): (usize, usize),
    l: usize,
) -> Result<SharedFfn, GpuError> {
    let at = |e: ChainError| GpuError::shape(WHAT, format!("layer {l}: {e}"));
    for (name, ty, shape) in [
        (&sh.gate, sh.gate_ty, (ff, h)),
        (&sh.up, sh.up_ty, (ff, h)),
        (&sh.down, sh.down_ty, (h, ff)),
    ] {
        shared_entry(ty, WHAT).map_err(at)?;
        site::site(w, WHAT, name, shape, ty)?;
    }
    let stack = gate_up_entry(sh.gate_ty, sh.up_ty, WHAT)
        .ok()
        .zip(down_entry(sh.down_ty, WHAT).ok());
    Ok(match stack {
        Some((gate_up, _)) => SharedFfn::Stack(SharedStack {
            gate: sh.gate.clone(),
            up: sh.up.clone(),
            down: sh.down.clone(),
            gate_up,
            down_ty: sh.down_ty,
        }),
        None => SharedFfn::Sites,
    })
}

impl TierBody for Placed {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.hybrid
    }

    /// The machine's parts when the glue holds one; a load that runs none
    /// keeps the plan's slot map and the residency calls do nothing.
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.glue.source()?;
        Some(ResidencyParts {
            glue: &mut self.glue,
            hybrid: &mut self.hybrid,
            slots: &self.side.slots,
        })
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
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
///
/// `normed` is the f32 normed rows a Q8_0 or F32 shared-expert part reads
/// ([`shared_sites`]): the step boundary's activation, where the step's
/// front wrote them, or the arena's own when `None`, where a unit's front
/// did.
fn shadow(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &mut PlacedRows,
    s: &mut Arena,
    normed: Option<&DeviceBuffer<f32>>,
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
        match f.leg {
            Some(FfnLeg::Dispatch { gate_up, down_ty }) => {
                let n_slots = m * pitch;
                gate_up_sel(
                    (stream, &side.k.kquant),
                    gate_up,
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
                    (gpu, k, &side.k.kquant),
                    (down_ty, kq_weight(w, &f.down)?),
                    (&s.act_h[i], &rows.sel),
                    n_slots,
                    hidden,
                    c.sink,
                    &mut s.down,
                )?;
            }
            Some(FfnLeg::Common { down, .. }) => {
                common_shadow(c, side, rows, s, normed, (f, down, m))?;
            }
            None => {
                return Err(GpuError::state(WHAT, "a leg for a layer with card experts"));
            }
        }
    }
    match (&f.shared, rows.shared.as_mut()) {
        (Some(SharedFfn::Stack(sh)), Some(r)) => shared_stack(c, side, (sh, r), s, m)?,
        (Some(SharedFfn::Sites), Some(r)) => shared_sites(c, r, s, normed, m)?,
        (None, _) => {}
        (Some(_), None) => {
            return Err(GpuError::state(WHAT, "the shared expert's buffers"));
        }
    }
    Ok(())
}

/// The plan's card leg arms the common-leg layers' stacks are admitted by
/// ([`accepts::leg_arms`], one pair for both bodies of the family), and what
/// a refusal of a stack of another type names.
fn common_stacks<'a>(w: &'a Weights, f: &'a PlacedFfn) -> Result<LegStacks<'a>, GpuError> {
    let arms = accepts::leg_arms(accepts::Program::Qwen35Body)
        .ok_or(GpuError::state(WHAT, "the placed card leg's arms"))?;
    LegStacks::of_arms(
        w,
        [&f.gate, &f.up, &f.down],
        arms,
        (
            "a resident routed gate or up stack of a type the placed card leg runs \
             (`accepts::prompt_leg_gate_up`)",
            "a resident routed down stack of a type the placed card leg runs \
             (`accepts::prompt_leg_down`)",
        ),
    )
}

/// Layer `c.layer`'s card experts through the common leg over `m` rows of the
/// gemv arm's width (at most [`GEMV_COLS`]): the four launches of
/// [`CardLeg::enqueue_rows`] over the f32 normed rows (`normed`, the step
/// boundary's, or the arena's own when `None`) and the places in `rows`,
/// writing the slots' down rows into the arena's `down` as the dispatch's
/// launches do. `f` the layer's placed FFN and `down` the file type of its
/// down.
fn common_shadow(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &mut PlacedRows,
    s: &mut Arena,
    normed: Option<&DeviceBuffer<f32>>,
    (f, down, m): (&PlacedFfn, GgmlType, usize),
) -> Result<(), GpuError> {
    let leg = side.k.card.as_ref().ok_or(GpuError::state(
        WHAT,
        "the common card leg's modules on a load with a common-leg layer",
    ))?;
    let PlacedRows { sel, common, .. } = rows;
    let bufs = common
        .as_mut()
        .ok_or(GpuError::state(
            WHAT,
            "the common-leg buffers of a load with a common-leg layer",
        ))?
        .bufs(down)?;
    let stacks = common_stacks(c.w, f)?;
    stacks.check(f.n_card, bufs.shape())?;
    let Arena {
        normed: own,
        down: out,
        ..
    } = s;
    let run = LegRun {
        layer: c.layer,
        stacks: &stacks,
        normed: normed.unwrap_or(&*own),
        m,
        sel,
        n_card: f.n_card,
        rule: Act::SiluMul,
    };
    leg.enqueue_rows(c.gpu, bufs, &run, out)
}

/// Layer `c.layer`'s card experts through the common leg over `m` rows past
/// the gemv arm's width: the layer's routed table through its map, the unit's
/// normed rows in the form the gate and up route GEMMs read, and the routed
/// block ([`RouteBlock`]: gate, up, the SwiGLU into the form the down reads,
/// down) over the card's stacks, writing the slots' down rows into the arena's
/// `down` as the dispatch's wide route does. The shared expert is the caller's.
fn common_wide(
    c: &Ctx<'_>,
    side: &PlacedSide,
    (table, common): (&mut GemmRoute, &mut CommonRows),
    s: &mut Arena,
    (f, map, m): (&PlacedFfn, &DeviceBuffer<u32>, usize),
) -> Result<(), GpuError> {
    let g32 = side.k.g32.as_ref().ok_or(GpuError::state(
        WHAT,
        "the 32-value kernels of a side whose units run the wide arm",
    ))?;
    let stack = common_stacks(c.w, f)?;
    let [gate, up, down] = stack.routes()?;
    let (gpu, stream) = (c.gpu, c.gpu.stream());
    let CommonWide { x, h, g, u } = common.wide.as_mut().ok_or(GpuError::state(
        WHAT,
        "the wide arm's buffers of a load with a common-leg layer",
    ))?;
    let d = s.dims;
    let slots = d.slots();
    let Arena {
        normed,
        route,
        down: out,
        ..
    } = s;
    g32.enqueue_route_remap(stream, route.ids(), map, m * slots, table, c.sink)?;
    match gate {
        RouteW::Grouped(..) => {
            let a = x.q128.as_mut().ok_or(GpuError::state(
                WHAT,
                "the unit's q8_1 rows of 128 a grouped gate and up read",
            ))?;
            gpu.enqueue_quantize_gemm(normed, m, a, c.sink)?;
        }
        RouteW::Blocks32(_) => {
            let a = x.q32.as_mut().ok_or(GpuError::state(
                WHAT,
                "the unit's q8 rows of 32 a 32-value gate and up read",
            ))?;
            g32.enqueue_quantize_gemm32(stream, normed, m, a, c.sink)?;
        }
    }
    let blk = RouteBlock {
        gate,
        up,
        down,
        ff: d.ff,
        hidden: d.hidden,
    };
    blk.enqueue(
        gpu,
        (&c.k.gemm, g32),
        (&*table, slots),
        m * slots,
        RouteIo {
            x128: x.q128.as_ref(),
            x32: x.q32.as_ref(),
            g,
            u,
            h128: h.q128.as_mut(),
            h32: h.q32.as_mut(),
            y: out,
        },
        c.sink,
    )
}

/// Layer `c.layer`'s shared expert as a stack of one expert on a fixed route
/// of expert 0 ([`SharedFfn::Stack`]) over `m` rows: the gate·up `_sel`
/// over the q8_1 of the normed rows into the SwiGLU rows, their q8_1, and
/// the down `_sel` into the output `combine` adds.
fn shared_stack(
    c: &Ctx<'_>,
    side: &PlacedSide,
    (sh, r): (&SharedStack, &mut SharedRows),
    s: &Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let RouteShape { hidden, ff, .. } = side.shape;
    let i = s.col(m)?;
    gate_up_sel(
        (gpu.stream(), &side.k.kquant),
        sh.gate_up,
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
        (gpu, k, &side.k.kquant),
        (sh.down_ty, kq_weight(w, &sh.down)?),
        (act, &r.zero),
        m,
        hidden,
        c.sink,
        &mut r.y,
    )
}

/// Layer `c.layer`'s shared expert as its own dense FFN at its site types
/// ([`SharedFfn::Sites`]) over `m` rows: `dispatch::shared_dense`, the run
/// the whole-card load's apart arm makes. The gate and up read the q8_1 of
/// the normed rows (`act_ffn[i]`, which the front's `norm_quant` wrote) or,
/// for a Q8_0 or F32 part, the f32 normed rows: `normed`, or the arena's own
/// when `None` (the same two buffers `route` wrote). The gate and up rows
/// are the arena's apart buffers (`Forms::apart` sizes them for every placed
/// load of a chain with a shared expert); the SwiGLU rows, their q8_1 and the
/// output are the host leg's `r`, whose output `combine` adds.
fn shared_sites(
    c: &Ctx<'_>,
    r: &mut SharedRows,
    s: &mut Arena,
    normed: Option<&DeviceBuffer<f32>>,
    m: usize,
) -> Result<(), GpuError> {
    let sh = c.p.ffn.apart().ok_or(GpuError::state(
        WHAT,
        "a shared expert kept apart on a layer whose placed side runs its sites",
    ))?;
    let i = s.col(m)?;
    let d = s.dims;
    let Arena {
        act_ffn,
        normed: own,
        cols,
        apart,
        ..
    } = s;
    let a = apart.as_mut().ok_or(GpuError::state(
        WHAT,
        "the arena's buffers of a shared expert kept apart",
    ))?;
    let act = r.act.get_mut(i).ok_or(GpuError::state(
        WHAT,
        "the shared expert's activations of m columns",
    ))?;
    dispatch::shared_dense(
        c,
        sh,
        (d.ff, d.hidden),
        (&act_ffn[i], normed.unwrap_or(&*own)),
        m,
        cols.as_mut(),
        SharedDenseBufs {
            g: &mut a.g,
            u: &mut a.u,
            h: &mut r.h,
            act,
            y: &mut r.y,
        },
    )
}

/// The gate·up `_sel` of the slots `a` selects over the stacks `a` names, at
/// the table's entry for their type: the K-quant family's Q4_K or Q5_K, each
/// skipping a [`HOST`] slot. The entries the family's FFN dispatch does not
/// launch (`plan::gate_up_entry` admits none of them) are named, not run.
fn gate_up_sel(
    (stream, kq): (&CudaStream, &QuantSelKernels),
    entry: GateUpEntry,
    a: &GateUpAct<'_>,
    sink: FaultSink,
    h: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    match entry {
        GateUpEntry::Q4k => kq.enqueue_gate_up_q4k(stream, a, sink, h),
        GateUpEntry::Q5k => kq.enqueue_gate_up_q5k(stream, a, sink, h),
        GateUpEntry::Q8_0
        | GateUpEntry::Q6k
        | GateUpEntry::Iq2xxs
        | GateUpEntry::Iq2xs
        | GateUpEntry::Iq2s
        | GateUpEntry::Iq1s
        | GateUpEntry::Iq1m
        | GateUpEntry::Iq3xxs
        | GateUpEntry::Iq3s
        | GateUpEntry::Iq4xs
        | GateUpEntry::Mxfp4
        | GateUpEntry::Nvfp4 => Err(GpuError::state(
            WHAT,
            "a gate·up entry the plan admits for no stack here",
        )),
    }
}

/// The down `_sel` launches a placed stack takes: the table's entries
/// (`kernels::down`) that read the `Q8Act` columns the placed side
/// quantizes its SwiGLU rows into (`enqueue_quantize_sel`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DownArm {
    /// `q4k_gemv_sel`.
    Q4k,
    /// `q5k_gemv_sel` (Walk A, the K-quant family).
    Q5k,
    /// `q6k_gemv_sel`.
    Q6k,
}

/// The arm a stack of file type `ty` takes, or why it takes none. The match
/// is on the table's entry with no wildcard, so an entry the table gains does
/// not compile here until it is placed or refused.
fn down_arm(ty: GgmlType) -> Result<DownArm, GpuError> {
    let refuse = |why: String| GpuError::shape(WHAT, format!("a {ty} down: {why}"));
    let Some((entry, act)) = kernels::down(ty) else {
        return Err(refuse("no common down `_sel` launches it".into()));
    };
    let arm = match entry {
        DownEntry::Q4k => DownArm::Q4k,
        DownEntry::Q5k => DownArm::Q5k,
        DownEntry::Q6k => DownArm::Q6k,
        // These read `Q8Blocks32` columns (the 32-value quantizer,
        // `enqueue_quantize_q8_sel`); the placed side's rows are `Q8Act`.
        DownEntry::Q5_0
        | DownEntry::Q5_1
        | DownEntry::Q8_0
        | DownEntry::Iq4nl
        | DownEntry::Nvfp4 => {
            return Err(refuse(format!(
                "its entry {entry:?} reads {act:?} columns; the placed side quantizes its \
                 SwiGLU rows as {:?}",
                DownAct::Q8Act
            )));
        }
        // These read `Q8Act` columns (the i-quant downs of `iq_sel.rs`, MXFP4)
        // and have no launch on the placed side.
        DownEntry::Iq2xxs
        | DownEntry::Iq2xs
        | DownEntry::Iq2s
        | DownEntry::Iq1s
        | DownEntry::Iq1m
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4xs
        | DownEntry::Mxfp4 => {
            return Err(refuse(format!(
                "its entry {entry:?} has no launch on the placed side"
            )));
        }
    };
    if act != DownAct::Q8Act {
        return Err(refuse(format!(
            "its entry {entry:?} reads {act:?} columns; the placed side quantizes its SwiGLU \
             rows as {:?}",
            DownAct::Q8Act
        )));
    }
    Ok(arm)
}

/// The down `_sel` of `n_slots` slots of the stack `wd` of type `ty`: slot
/// `s` the rows of expert `sel[s]` against column `s` of `act`, a [`HOST`]
/// slot left as it was.
#[allow(
    clippy::too_many_arguments,
    reason = "one launch's kernels, stack, columns, slots, width, fault sink and output (rust-quality R8)"
)]
fn down_sel(
    (gpu, k, kq): (&Gpu, &Kernels, &QuantSelKernels),
    (ty, wd): (SiteTy, &DeviceTensor<u32>),
    (act, sel): (&Q8Act, &DeviceBuffer<u32>),
    n_slots: usize,
    hidden: usize,
    sink: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    match down_arm(ty.ggml())? {
        DownArm::Q4k => gpu
            .q4k_sel()
            .enqueue_gemv_q4k_sel(stream, wd, act, sel, n_slots, hidden, y),
        DownArm::Q5k => kq.enqueue_gemv_q5k_sel(
            stream,
            &SelDown {
                w: wd,
                act,
                sel,
                n_slots,
                rows_per_expert: hidden,
            },
            sink,
            y,
        ),
        DownArm::Q6k => k
            .q6_sel
            .enqueue_gemv_q6k_sel(stream, wd, act, sel, n_slots, hidden, y),
    }
}

/// Layer `layer`'s tier handoff and go: the stage page and the tier image
/// in one launch, then the go that waits for the tier. A load with no tier
/// side is refused by name. One column.
fn tier_go(
    side: &mut PlacedSide,
    sel: &mut DeviceBuffer<u32>,
    hybrid: &mut Hybrid<HostRun>,
    route: (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
    layer: usize,
    sink: FaultSink,
    stream: &CudaStream,
) -> Result<(), GpuError> {
    let (ids, weights) = route;
    let shape = side.shape;
    let PlacedSide { tier, slots, k, .. } = side;
    let tier = tier
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the stage card's tier side"))?;
    let h = Handoff {
        ids,
        weights,
        map: slots.buf(),
        row_off: layer * shape.n_expert,
        n_expert: shape.n_expert,
    };
    {
        let (places, tsel) = tier.handoff_parts(0)?;
        let targets = hybrid.tier_handoff(0, leg_tier::STAGE_TIER)?;
        k.handoff
            .enqueue_handoff_tier(stream, &h, places, targets, sink, (sel, tsel))?;
    }
    hybrid.enqueue_tier_go(stream, layer, 0)
}

/// The combine of a layer the tier holds experts of: the stage card's slots
/// and the tier's, in slot order, with the host's sums (`tier_leg`). `n_tier`
/// is the layer's experts on the tier. `out` as [`combine`].
#[allow(
    clippy::too_many_arguments,
    reason = "one join's context, side, rows, arena, width, both sums, the tier's experts and the output"
)]
fn combine_tier(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &PlacedRows,
    s: &mut Arena,
    m: usize,
    sums: (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    n_tier: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    let (hsum, trows) = sums;
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
    let tsel = side
        .tier
        .as_ref()
        .ok_or(GpuError::state(WHAT, "the stage card's tier side"))?
        .tsel(0)?;
    let join = side.k.join.as_ref().ok_or(GpuError::state(
        WHAT,
        "the tier join of a load with an expert tier",
    ))?;
    join.enqueue(
        c.gpu.stream(),
        TierCombineArgs {
            down,
            trows,
            w: route.weights(),
            sel: &rows.sel,
            tsel,
            hsum,
            resid: ffn_inp,
            shared,
            rows: shape.hidden,
            pitch: shape.pitch,
            n_used: shape.used,
            n_card: f.n_card,
            n_tier,
            m,
            fault: c.sink,
            y: out.unwrap_or(x),
        },
    )
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

/// Layer `l`'s card work under the host leg over `m` rows (more than
/// [`GEMV_COLS`]) of arena `s`: the wide arm's routed table over the layer's
/// card experts through its map, the routed GEMMs, and the shared expert.
fn card_wide(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &mut PlacedRows,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let map = side.maps.get(c.layer).ok_or(GpuError::state(
        WHAT,
        "a map of the card's experts for every layer of a wide unit",
    ))?;
    let PlacedRows { routes, common, .. } = rows;
    let table = routes
        .get_mut(c.layer)
        .ok_or(GpuError::state(
            WHAT,
            "a table of the card's experts for every layer of a wide unit",
        ))?
        .as_mut();
    let remap = side.k.g32.as_ref().ok_or(GpuError::state(
        WHAT,
        "the remap kernels of a side whose units run the wide arm",
    ))?;
    let f = side
        .ffn
        .get(c.layer)
        .ok_or(GpuError::state(WHAT, "a placed FFN for every layer"))?;
    // A common-leg layer's routed experts run through the common leg; the
    // wide arm's own routed GEMMs (the table) are the dispatch layers'.
    let table = match (f.common(), table, common.as_mut()) {
        (Some(_), Some(table), Some(common)) => {
            common_wide(c, side, (table, common), s, (f, map, m))?;
            None
        }
        (Some(_), _, _) => {
            return Err(GpuError::state(
                WHAT,
                "a table and the common-leg buffers for a common-leg layer with card experts",
            ));
        }
        (None, table, _) => table,
    };
    wide::ffn_card(c, &c.p.ffn, s, m, Some(CardRoute { map, table, remap }))
}

/// Layer `c.layer`'s ring pass over `m` rows (more than [`GEMV_COLS`]) of
/// arena `s` after its card route ([`card_wide`]), under the layer's stream
/// `r`: the union places of the unit's routed slots into the ring rows, cut
/// from the stream's union row read in place (the places read no id past a
/// row's routed slots, so that row needs no shared entry), then one GEMM
/// batch a landing batch of the stream — the batch's map widened
/// ([`widen_map`]) and the half's stacks through it ([`wide::ffn_ring`]),
/// each behind its own landed event. Returns the union's card places, the
/// layer's card experts and the streamed ones (`n_card + r.n`), which the
/// combine reads ([`StreamWalk`]'s back). Refused by name: rows the ring is
/// not armed on, and a stream of more batches than the rows hold maps for or
/// of more experts than its half.
fn ring_pass(
    c: &Ctx<'_>,
    side: &PlacedSide,
    rows: &mut PlacedRows,
    s: &mut Arena,
    m: usize,
    r: &RingLayer,
) -> Result<usize, GpuError> {
    let shape = side.shape;
    let n_card = side
        .ffn
        .get(c.layer)
        .ok_or(GpuError::state(WHAT, "a placed FFN for every layer"))?
        .n_card;
    let ring = rows.ring.as_mut().ok_or(GpuError::state(
        WHAT,
        "pass rows armed for the expert stream's ring (Placed::start_xstream)",
    ))?;
    if r.batches.len() > ring.maps.len() || r.n > r.half_slots {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: a stream of {} experts in {} landing batches over a half of {} \
                 slots; the ring rows hold {} batch maps",
                c.layer,
                r.n,
                r.batches.len(),
                r.half_slots,
                ring.maps.len()
            ),
        ));
    }
    let n = &c.p.ffn;
    let (ctx, stream) = (c.gpu.context(), c.gpu.stream());
    let cols = |name: &str| kq_weight(c.w, name).map(DeviceTensor::cols);
    let (gate_cols, up_cols, down_cols) = (cols(&n.gate)?, cols(&n.up)?, cols(&n.down)?);
    let rows = [shape.ff, shape.ff, shape.hidden];
    let cols = [gate_cols, up_cols, down_cols];
    wide::with_ring_windows(r, ctx, shape.n_expert, rows, cols, |union_map, stacks| {
        side.k.leg.enqueue_places(
            stream,
            &PlacesArgs {
                ids: s.route.ids(),
                map: union_map,
                row_off: 0,
                n_expert: shape.n_expert,
                pitch: shape.pitch,
                used: shape.used,
                m,
            },
            c.sink,
            &mut ring.union,
        )?;
        for (b, map) in r.batches.iter().zip(ring.maps.iter_mut()) {
            b.with_map(ctx, shape.n_expert, |row| {
                widen_map(stream, (row, 0), shape.n_expert, map)
            })?;
            wide::ffn_ring(c, n, s, m, map, &mut ring.route, b.event.as_deref(), stacks)?;
        }
        if let Some(checks) = ring.checks.as_mut() {
            let maps = ring.maps.get(..r.batches.len()).unwrap_or_default();
            let ids = s.route.ids();
            checks.push(union_check(c, side, (ids, &ring.union), maps, m, n_card)?);
        }
        Ok(n_card + r.n)
    })
}

/// Layer `c.layer`'s union places of `m` rows (`union`, cut from the
/// unit's `ids`) read back against the stack's row (the slot map's card
/// copy) and the ring's slots from the layer's widened batch maps (`maps`,
/// each streamed expert in one of them): [`RingCheck`]. Synchronizes the
/// engine stream; the union check's alone ([`Placed::set_ring_check`]).
fn union_check(
    c: &Ctx<'_>,
    side: &PlacedSide,
    (ids, union): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
    maps: &[DeviceBuffer<u32>],
    m: usize,
    n_card: usize,
) -> Result<RingCheck, GpuError> {
    let (shape, l) = (side.shape, c.layer);
    let (n, pitch) = (shape.n_expert, shape.pitch);
    let stream = c.gpu.stream();
    let mut ring = vec![HOST; n];
    for map in maps {
        for (r, &i) in ring.iter_mut().zip(&map.to_host_vec(stream)?) {
            if i != HOST {
                *r = i;
            }
        }
    }
    let ids = ids.to_host_vec(stream)?;
    let got = union.to_host_vec(stream)?;
    let stack = side.slots.buf().to_host_vec(stream)?;
    let stack = stack.get(l * n..(l + 1) * n).ok_or(GpuError::state(
        WHAT,
        "a row of the slot map's card copy for every layer",
    ))?;
    let base = u32::try_from(n_card).map_err(|_| GpuError::shape(WHAT, "a card count past u32"))?;
    let (mut streamed, mut wrong) = (0, 0);
    for at in 0..m * pitch {
        let want = if at % pitch >= shape.used {
            HOST
        } else {
            let id = ids.get(at).map_or(usize::MAX, |&id| id as usize);
            match (ring.get(id), stack.get(id)) {
                (Some(&i), _) if i != HOST => {
                    streamed += 1;
                    base + i
                }
                (_, Some(&p)) => p,
                _ => HOST,
            }
        };
        if got.get(at) != Some(&want) {
            wrong += 1;
        }
    }
    Ok(RingCheck {
        layer: l,
        streamed,
        wrong,
    })
}

/// What a placed walk reads and writes: the chain's context, its stores,
/// the arena and its input, the unit's rows, the placed side (a streaming
/// prompt call's walk brings a layer's map to its pick,
/// [`PlacedSide::refresh_map`]) and the arena's host leg buffers.
pub(super) struct WalkParts<'a, S: Stores + ?Sized> {
    pub(super) c: &'a PassCtx<'a>,
    pub(super) stores: &'a mut S,
    pub(super) s: &'a mut Arena,
    pub(super) io: &'a Io<'a>,
    pub(super) m: usize,
    pub(super) side: &'a mut PlacedSide,
    pub(super) rows: &'a mut PlacedRows,
}

impl<S: Stores + ?Sized> WalkParts<'_, S> {
    /// Layer `l`'s front up to the FFN norm at the walk's rows: the chain's
    /// walk ahead of the FFN half ([`walk::front`]: the embedding in front of
    /// layer 0, then the mixer) over the family's ops.
    fn mixer(&mut self, l: usize) -> Result<(), GpuError> {
        let cx = LayerCtx::of(self.c.plans, l, self.m)?;
        let mut ops = Program {
            c: self.c,
            stores: &mut *self.stores,
            s: &mut *self.s,
            io: self.io,
            m: self.m,
            tail: Tail::Pass,
        };
        walk::front(&mut ops, cx)
    }

    /// Layer `lc`'s places of the unit's routed slots into the leg's `sel`,
    /// cut from the card copy's row of the layer as the stream stands: the
    /// combine's split of each row between the card and the host.
    fn places(&mut self, lc: &Ctx<'_>) -> Result<(), GpuError> {
        let shape = self.side.shape;
        self.side.k.leg.enqueue_places(
            lc.gpu.stream(),
            &PlacesArgs {
                ids: self.s.route.ids(),
                map: self.side.slots.buf(),
                row_off: lc.layer * shape.n_expert,
                n_expert: shape.n_expert,
                pitch: shape.pitch,
                used: shape.used,
                m: self.m,
            },
            lc.sink,
            &mut self.rows.sel,
        )
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
    /// then the handoff into the page with the places, and the go. On a
    /// layer the tier holds experts of, the tier handoff and go replace
    /// them.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.mixer(l)?;
        let p = &mut self.p;
        let lc = ctx_of(p.c, l)?;
        let tier_k = p.side.tier.as_ref().map_or(0, |t| t.k(l));
        {
            let normed = port.hybrid().boundary_mut().normed_mut();
            route(&lc, &lc.p.ffn, p.s, 1, Some(normed))?;
        }
        let stream = lc.gpu.stream();
        if tier_k > 0 {
            return tier_go(
                p.side,
                &mut p.rows.sel,
                port.hybrid(),
                (p.s.route.ids(), p.s.route.weights()),
                l,
                lc.sink,
                stream,
            );
        }
        let shape = p.side.shape;
        let h = Handoff {
            ids: p.s.route.ids(),
            weights: p.s.route.weights(),
            map: p.side.slots.buf(),
            row_off: l * shape.n_expert,
            n_expert: shape.n_expert,
        };
        let target = port.hybrid().boundary_mut().handoff_target_of(0)?;
        p.side
            .k
            .handoff
            .enqueue_handoff(stream, &h, target, lc.sink, &mut p.rows.sel)?;
        port.hybrid().boundary().enqueue_go_of(stream, l, 0)
    }

    /// The card's routed experts and the shared expert (the norm's f32 rows
    /// it may read are the boundary's).
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        let normed = port.hybrid().boundary().normed();
        shadow(&lc, p.side, p.rows, p.s, Some(normed), 1)
    }

    /// The wait, the combine — into the head's input after the last layer
    /// — the layer's tap row when taps are on, and the host tier told the
    /// layer is enqueued (an eager step is served there). On a layer the
    /// tier holds experts of, the wait covers the tier and the combine
    /// joins its slots.
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let last = l + 1 == self.p.c.plans.len();
        let StepWalk { p, tail } = self;
        let lc = ctx_of(p.c, l)?;
        let stream = lc.gpu.stream();
        let tier_k = p.side.tier.as_ref().map_or(0, |t| t.k(l));
        let hy = port.hybrid();
        if tier_k > 0 {
            hy.enqueue_tier_back(stream, l, 0)?;
            let sums = hy.tier_join_rows(0)?;
            match tail {
                Tail::Step { head, taps, .. } => {
                    let out = last.then(|| head.input_mut());
                    combine_tier(&lc, p.side, p.rows, p.s, 1, sums, tier_k, out)?;
                    if let Some(t) = taps {
                        let src: &DeviceBuffer<f32> = if last { head.input_mut() } else { &p.s.x };
                        t.rows[l].copy_from_device_async(src, stream)?;
                    }
                }
                _ => combine_tier(&lc, p.side, p.rows, p.s, 1, sums, tier_k, None)?,
            }
        } else {
            hy.boundary().enqueue_back_of(stream, 0, l)?;
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
        }
        port.hybrid().row_enqueued(l, 0)
    }

    /// The walk's [`Tail`].
    fn end(&mut self, _: usize) -> Result<(), GpuError> {
        let StepWalk { p, tail } = self;
        enqueue_tail(p.c, p.s, p.m, tail)
    }
}

/// The chain a pass of `rows` replays. One column is the step; more columns
/// are one [`Chain::Cols`]. The placed replay serves it. A whole-card load
/// has no host service of the pass.
pub(super) fn slot_chain(rows: &[SlotRange]) -> Chain {
    match rows.last().map(|r| r.rows.end).unwrap_or(0) {
        0 | 1 => Chain::Step,
        m => Chain::Cols(m),
    }
}

/// What a pass of columns mixes before the placed FFN: one layer's
/// sequence-bound launches, the body's own.
pub(super) trait ColsMix {
    /// Layer `layer` over the pass's rows, leaving each row's FFN input in
    /// the arena.
    fn mix_layer(&mut self, layer: usize, c: &PassCtx<'_>, s: &mut Arena) -> Result<(), GpuError>;
}

/// A pass of resident slots `(1, m, Step)` through the host tier's step
/// port: [`ColsMix`] a layer, then that layer's placed FFN over the `m`
/// columns, then the [`Tail`]. The tail is one head of every row.
pub(super) struct ColsWalk<'a, M: ColsMix> {
    pub(super) c: &'a PassCtx<'a>,
    pub(super) s: &'a mut Arena,
    pub(super) m: usize,
    pub(super) side: &'a mut PlacedSide,
    pub(super) rows: &'a mut PlacedRows,
    pub(super) mix: M,
    pub(super) tail: Tail<'a>,
}

impl<'a, M: ColsMix> ColsWalk<'a, M> {
    /// Walk every layer through `leg`, then the tail. No row, a row past
    /// the page or past [`MAX_PASS_ROWS`], a chain of no layer, or a tail
    /// other than one head of every row is refused by name.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let cols = leg.hybrid().boundary().layout().cols();
        if self.m == 0 || self.m > MAX_PASS_ROWS || self.m > cols || self.c.plans.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a slot pass of {} rows over {} layers on a page of {cols} columns; a pass \
                     is 1..={MAX_PASS_ROWS} rows of a chain",
                    self.m,
                    self.c.plans.len()
                ),
            ));
        }
        if !matches!(self.tail, Tail::One { .. }) {
            return Err(GpuError::shape(
                WHAT,
                "a slot pass ends in one head of every row",
            ));
        }
        if self.m > 1 && self.side.tier.as_ref().is_some_and(TierSide::any) {
            return Err(GpuError::shape(WHAT, PASS_BESIDE_TIER));
        }
        let o = Overlap {
            units: 1,
            cols: self.m,
            port: PortKind::Step,
        };
        sched::walk(o, self.c.plans.len(), leg, &mut self)
    }
}

impl<'a, M: ColsMix> LayerProgram for ColsWalk<'a, M> {
    type Port = StepLeg<'a, HostRun>;

    /// The mixer's layer, the norm into the boundary's activation and the
    /// router, then each column's handoff into the page and the one go. On
    /// a layer the tier holds experts of, one column takes the tier handoff
    /// and go; several columns are refused by name.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.mix.mix_layer(l, self.c, self.s)?;
        let lc = ctx_of(self.c, l)?;
        let tier_k = self.side.tier.as_ref().map_or(0, |t| t.k(l));
        if tier_k > 0 && self.m != 1 {
            return Err(GpuError::shape(WHAT, PASS_BESIDE_TIER));
        }
        {
            let normed = port.hybrid().boundary_mut().normed_mut();
            route(&lc, &lc.p.ffn, self.s, self.m, Some(normed))?;
        }
        if tier_k > 0 {
            return tier_go(
                self.side,
                &mut self.rows.sel,
                port.hybrid(),
                (self.s.route.ids(), self.s.route.weights()),
                l,
                lc.sink,
                lc.gpu.stream(),
            );
        }
        let shape = self.side.shape;
        handoff_columns(
            &self.side.k.handoff,
            self.side.slots.buf(),
            shape,
            self.s.route.ids(),
            self.s.route.weights(),
            port.hybrid().boundary_mut(),
            self.m,
            l,
            lc.sink,
            lc.gpu.stream(),
            &mut self.rows.sel,
        )?;
        port.hybrid()
            .boundary()
            .enqueue_go_of(lc.gpu.stream(), l, 0)
    }

    /// The card's routed experts and the shared expert over the pass's
    /// rows (the norm's f32 rows are the boundary's).
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let lc = ctx_of(self.c, at.layer)?;
        let normed = port.hybrid().boundary().normed();
        shadow(&lc, self.side, self.rows, self.s, Some(normed), self.m)
    }

    /// The wait, the combine into the arena's `x`, and the host tier told
    /// the layer is enqueued (an eager pass is served there). On a layer
    /// the tier holds experts of, the wait covers the tier and the combine
    /// joins its slots; several columns are refused by name.
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let lc = ctx_of(self.c, l)?;
        let stream = lc.gpu.stream();
        let tier_k = self.side.tier.as_ref().map_or(0, |t| t.k(l));
        let hy = port.hybrid();
        if tier_k > 0 {
            if self.m != 1 {
                return Err(GpuError::shape(WHAT, PASS_BESIDE_TIER));
            }
            hy.enqueue_tier_back(stream, l, 0)?;
            let sums = hy.tier_join_rows(0)?;
            combine_tier(
                &lc, self.side, self.rows, self.s, self.m, sums, tier_k, None,
            )?;
        } else {
            hy.boundary().enqueue_back_of(stream, 0, l)?;
            let hsum = hy.boundary().hsum_of(0)?;
            combine(&lc, self.side, self.rows, self.s, self.m, hsum, None)?;
        }
        port.hybrid().row_enqueued(l, 0)
    }

    /// The walk's [`Tail`].
    fn end(&mut self, _: usize) -> Result<(), GpuError> {
        enqueue_tail(self.c, self.s, self.m, &mut self.tail)
    }
}

/// Column `0..m` of layer `layer` into the step page: each column's routed
/// ids, weights and activation, at the router's pitch (the shared slot
/// stays [`HOST`]), through the handoff, into that column of the page.
#[allow(
    clippy::too_many_arguments,
    reason = "one column handoff's kernels, map, shape, routing, page, rows, layer, fault, stream and places (rust-quality R8)"
)]
fn handoff_columns(
    handoff: &HandoffKernels,
    map: &DeviceBuffer<u32>,
    shape: RouteShape,
    ids: &DeviceBuffer<u32>,
    weights: &DeviceBuffer<f32>,
    boundary: &mut Boundary,
    m: usize,
    layer: usize,
    sink: FaultSink,
    stream: &CudaStream,
    sel: &mut DeviceBuffer<u32>,
) -> Result<(), GpuError> {
    let RouteShape {
        used,
        pitch,
        hidden,
        n_expert,
        ..
    } = shape;
    let cols = boundary.layout().cols();
    let base = boundary.layout().handoff();
    if pitch < used || base.n_used != used || m == 0 || m > cols {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a handoff of {m} columns of {used} routed slots at pitch {pitch} on a page of \
                 {cols} columns of {} routed slots",
                base.n_used
            ),
        ));
    }
    let HandoffTarget { x, seq, image, .. } = boundary.handoff_target_of(0)?;
    let word = size_of::<u32>();
    for j in 0..m {
        let ids_j = Window::of(ids, j * pitch * word, used)?;
        let w_j = Window::of(weights, j * pitch * word, used)?;
        let x_j = Window::of(x, j * hidden * word, hidden)?;
        let mut sel_j = WindowMut::of_mut(sel, j * pitch * word, used)?;
        let shifted = HandoffLayout {
            seq: base.seq,
            ids: base.ids + j * used,
            weights: base.weights + j * used,
            x: base.x + j * hidden,
            n_used: base.n_used,
            hidden: base.hidden,
        };
        handoff.enqueue_handoff(
            stream,
            &Handoff {
                ids: &ids_j,
                weights: &w_j,
                map,
                row_off: layer * n_expert,
                n_expert,
            },
            HandoffTarget {
                x: &x_j,
                seq,
                image: &mut *image,
                layout: shifted,
            },
            sink,
            &mut sel_j,
        )?;
    }
    Ok(())
}

/// A prompt unit's walk `(1, m, Batch)` of up to [`PlacedSide::cap`] rows
/// through the host tier's batch port; its [`Tail`] follows the walk
/// ([`BatchWalk::walk`]). A unit of up to [`GEMV_COLS`] rows is the gemv
/// arm's launches (`route`, `shadow`, `combine`), a longer one the wide
/// arm's.
pub(super) struct BatchWalk<'a, S: Stores + ?Sized> {
    pub(super) p: WalkParts<'a, S>,
}

impl<'a, S: Stores + ?Sized> BatchWalk<'a, S> {
    /// Walk every layer at the unit's rows through `leg`, then `tail`. A
    /// unit of no row or past the placed side's most rows is refused by name.
    pub(super) fn walk(
        mut self,
        leg: &mut BatchLeg<'a, HostRun>,
        mut tail: Tail<'a>,
    ) -> Result<(), GpuError> {
        let o = self.overlap()?;
        let layers = self.p.c.plans.len();
        sched::walk(o, layers, leg, &mut self)?;
        let p = self.p;
        enqueue_tail(p.c, p.s, o.cols, &mut tail)
    }

    /// [`BatchWalk::walk`] in a streaming prompt call ([`StreamWalk`]):
    /// `stream` the call's, `walk_end` the unit's last position. The walk's
    /// last card layer's return, which no shadow follows, is the call end's
    /// ([`PromptStream::return_owed`]). Refused by name as `walk` refuses.
    pub(super) fn walk_streamed(
        self,
        leg: &mut BatchLeg<'a, HostRun>,
        mut tail: Tail<'a>,
        stream: &mut PromptStream,
        walk_end: usize,
    ) -> Result<(), GpuError> {
        let o = self.overlap()?;
        let layers = self.p.c.plans.len();
        let mut w = StreamWalk {
            walk: self,
            stream,
            walk_end,
            ring_at: None,
        };
        sched::walk(o, layers, leg, &mut w)?;
        let engine = leg.stream();
        w.stream.return_owed(leg.hybrid(), engine)?;
        let p = w.walk.p;
        enqueue_tail(p.c, p.s, o.cols, &mut tail)
    }

    /// The unit's point `(1, m, Batch)`. A unit of no row or past the
    /// placed side's most rows is refused by name.
    fn overlap(&self) -> Result<Overlap, GpuError> {
        let m = self.p.m;
        let cap = self.p.side.cap;
        if m == 0 || m > cap || self.p.c.plans.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a unit of {m} rows over {} layers; a placed unit walks 1..={cap} rows",
                    self.p.c.plans.len()
                ),
            ));
        }
        if self.p.side.tier.as_ref().is_some_and(TierSide::any) {
            return Err(GpuError::shape(WHAT, PROMPT_BESIDE_TIER));
        }
        Ok(Overlap {
            units: 1,
            cols: m,
            port: PortKind::Batch,
        })
    }

    /// After the pick of `l` moved the layer's map (`land`, its landing
    /// batches, not empty): the layer's map brought to the moved card copy
    /// ([`PlacedSide::refresh_map`]) and its places cut again from it — the
    /// front cut them from the row before the pick, which counts an expert
    /// the pick moved from the host to the card on neither side and one it
    /// sent the other way on both — then the engine stream waiting the last
    /// landing batch's event: the wide arm's card route reads the layer's
    /// slots through one remap, so it waits every copy before its first
    /// GEMM, not a GEMM batch a landing batch.
    fn moved(&mut self, l: usize, land: &[LandBatch]) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, l)?;
        let stream = lc.gpu.stream();
        p.side.refresh_map(stream, p.m, l)?;
        p.places(&lc)?;
        if let Some(e) = land.iter().rev().find_map(|b| b.event.as_deref()) {
            stream.wait(e)?;
        }
        Ok(())
    }

    /// [`BatchWalk`]'s wide back for a layer whose shadow ran the ring pass
    /// ([`ring_pass`]): the combine over the union places, below `n` the
    /// card's — its stacks' experts and the streamed ones — and the host
    /// sums the walk's serve uploaded, which leave the streamed experts out
    /// (the serve's exclusion set).
    fn back_union(
        &mut self,
        port: &mut BatchLeg<'_, HostRun>,
        at: At,
        n: usize,
    ) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        let ring = p.rows.ring.as_ref().ok_or(GpuError::state(
            WHAT,
            "pass rows armed for the expert stream's ring (Placed::start_xstream)",
        ))?;
        let host = HostSide {
            sel: &ring.union,
            hsum: port.hsum(),
            n_card: n,
            leg: &p.side.k.leg,
        };
        wide::ffn_combine(&lc, &lc.p.ffn, p.s, p.m, Some(host), None)
    }
}

impl<'a, S: Stores + ?Sized> LayerProgram for BatchWalk<'a, S> {
    type Port = BatchLeg<'a, HostRun>;

    /// The mixer, the norm into the arena's `normed` and the router (the
    /// wide arm's front past [`GEMV_COLS`] rows), the download of the unit's
    /// rows, weights and ids, and the places.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.p.side.tier.as_ref().is_some_and(TierSide::any) {
            return Err(GpuError::shape(WHAT, PROMPT_BESIDE_TIER));
        }
        let l = at.layer;
        self.p.mixer(l)?;
        let p = &mut self.p;
        let lc = ctx_of(p.c, l)?;
        let m = p.m;
        if m > GEMV_COLS {
            wide::ffn_front(&lc, &lc.p.ffn, p.s, m)?;
        } else {
            route(&lc, &lc.p.ffn, p.s, m, None)?;
        }
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
        p.places(&lc)
    }

    /// The card's routed experts and the shared expert.
    fn shadow(&mut self, _: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        if p.m > GEMV_COLS {
            return card_wide(&lc, p.side, p.rows, p.s, p.m);
        }
        shadow(&lc, p.side, p.rows, p.s, None, p.m)
    }

    /// The combine over the host sums the walk's serve uploaded.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let p = &mut self.p;
        let lc = ctx_of(p.c, at.layer)?;
        if p.m > GEMV_COLS {
            let f = p
                .side
                .ffn
                .get(at.layer)
                .ok_or(GpuError::state(WHAT, "a placed FFN for every layer"))?;
            let host = HostSide {
                sel: &p.rows.sel,
                hsum: port.hsum(),
                n_card: f.n_card,
                leg: &p.side.k.leg,
            };
            return wide::ffn_combine(&lc, &lc.p.ffn, p.s, p.m, Some(host), None);
        }
        combine(&lc, p.side, p.rows, p.s, p.m, port.hsum(), None)
    }
}

/// A prompt unit's walk in a streaming prompt call (the [`XMode`] lever's
/// `admit` and `split`, [`PromptStream`]): [`BatchWalk`]'s parts with the
/// stream's around them. At a layer's shadow the return of the layer whose
/// host service ran last, then — on a layer with card experts — the layer's
/// pick for the unit, the map and places it moved ([`BatchWalk::moved`]),
/// the card route under them, under `split` the ring pass over the layer's
/// stream ([`ring_pass`]), and the layer's last read of its slots; at its
/// back the combine — over the union places when the shadow ran the ring
/// pass ([`BatchWalk::back_union`]) — and the layer's return owed in the
/// returning call's last walk. The serve of a layer runs after its shadow
/// (the batch order's), so it reads the host map the pick moved and leaves
/// out what the layer streamed. A unit of at most [`GEMV_COLS`] rows picks
/// nothing (the pick's floor is past it); a stream on one is refused by
/// name, since the gemv arm has no ring pass.
///
/// [`XMode`]: crate::host::xstream::XMode
struct StreamWalk<'a, 'b, S: Stores + ?Sized> {
    walk: BatchWalk<'a, S>,
    stream: &'b mut PromptStream,
    /// The unit's last position: the call's last walk returns.
    walk_end: usize,
    /// The layer whose shadow ran the ring pass and its union's card places,
    /// for that layer's back (one unit: a layer's back runs before the next
    /// layer's front).
    ring_at: Option<(usize, usize)>,
}

impl<'a, S: Stores + ?Sized> LayerProgram for StreamWalk<'a, '_, S> {
    type Port = BatchLeg<'a, HostRun>;

    /// [`BatchWalk`]'s front: its places are cut from the row before the
    /// layer's pick.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        self.walk.front(port, at)
    }

    /// The return owed, the pick, the moved map and places when it moved
    /// any, the card's routed experts and the shared expert, the ring pass
    /// when the layer streams, the read.
    fn shadow(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let StreamWalk {
            walk,
            stream,
            walk_end,
            ring_at,
        } = self;
        let m = walk.p.m;
        stream.return_at_shadow(port, at, *walk_end)?;
        let side = &*walk.p.side;
        let (ring, land) = stream.pick(port, at, (m, 1), *walk_end, |l| side.has_card(l))?;
        if ring.is_some() && m <= GEMV_COLS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: the expert stream's ring on a unit of {m} rows; the ring pass \
                     is the wide arm's, past {GEMV_COLS} rows"
                ),
            ));
        }
        if !land.is_empty() {
            walk.moved(l, &land)?;
        }
        walk.shadow(port, at)?;
        *ring_at = match ring {
            Some(r) => {
                let p = &mut walk.p;
                let lc = ctx_of(p.c, l)?;
                Some((l, ring_pass(&lc, p.side, p.rows, p.s, m, &r)?))
            }
            None => None,
        };
        let side = &*walk.p.side;
        stream.read(port, l, true, |l| side.has_card(l))
    }

    /// [`BatchWalk`]'s back, over the union places when the layer's shadow
    /// ran the ring pass; then, in the returning call's last walk, the
    /// layer's return is due, asked at the next layer's shadow. A union kept
    /// for another layer is refused by name.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        match self.ring_at.take() {
            Some((rl, n)) if rl == at.layer => self.walk.back_union(port, at, n)?,
            Some((rl, _)) => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {}'s back after layer {rl}'s ring pass: a layer's union is \
                         read at its own back",
                        at.layer
                    ),
                ));
            }
            None => self.walk.back(port, at)?,
        }
        let l = at.layer;
        if self.stream.returns_here(self.walk_end) && self.walk.p.side.has_card(l) {
            self.stream.owe_return(l);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DownArm, down_arm};
    use gguf::quant::GgmlType;

    /// The placed side launches the table's K-quant `Q8Act` downs, each at its
    /// own entry, and refuses every other type by name: the `Q8Blocks32`
    /// entries with the form they read, the entries it has no launch for with
    /// that fact, a type with no entry with that fact.
    #[test]
    fn the_placed_down_follows_the_tables_entry() {
        for (ty, arm) in [
            (GgmlType::Q4_K, DownArm::Q4k),
            (GgmlType::Q5_K, DownArm::Q5k),
            (GgmlType::Q6_K, DownArm::Q6k),
        ] {
            assert_eq!(down_arm(ty).ok(), Some(arm), "{ty}");
        }
        for ty in [
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
        ] {
            let why = down_arm(ty).err().map(|e| e.to_string());
            assert!(
                why.as_deref().is_some_and(|w| w.contains("Q8Blocks32")),
                "{ty}: {why:?}"
            );
        }
        // PIN(2026-10-11): the i-quant downs have table entries now
        // (`iq_sel.rs`'s `Q8Act` downs), which the placed side has no launch
        // for; they were in the no-entry list below.
        // PIN(2026-10-25): and IQ2_S, IQ1_S and IQ1_M, whose card entries are the same kind.
        for ty in [
            GgmlType::IQ2_XS,
            GgmlType::IQ2_S,
            GgmlType::IQ1_S,
            GgmlType::IQ1_M,
            GgmlType::IQ3_XXS,
            GgmlType::IQ4_XS,
            GgmlType::MXFP4,
        ] {
            let why = down_arm(ty).err().map(|e| e.to_string());
            assert!(
                why.as_deref()
                    .is_some_and(|w| w.contains("no launch on the placed side")),
                "{ty}: {why:?}"
            );
        }
        for ty in [GgmlType::Q3_K, GgmlType::F16] {
            let why = down_arm(ty).err().map(|e| e.to_string());
            assert!(
                why.as_deref()
                    .is_some_and(|w| w.contains("no common down `_sel`")),
                "{ty}: {why:?}"
            );
        }
    }
}
