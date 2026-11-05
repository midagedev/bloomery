//! The MoE sub-layer of the V4.1 step: the norm, the router, the go to the
//! host tier, the card's routed and shared experts, the wait, the combine
//! and HC_POST with the next fold. See [`super`] for the piece contract.
//!
//! One layer's launches, in stream order:
//!
//! ```text
//! norm+Q8 → router → handoff → go → HC_PRE → [gate·up → h q8_1 → down] → shared gate·up → shared down
//!         → (the caller's shadow work) → wait → combine+HC_POST (+ fold)
//! ```
//!
//! The shared expert runs in its file format ([`crate::dense`]): q8_0 gate·up
//! on the f32 activation and a q8_0 down; or Q3_K gate·up on the norm's q8_1
//! form, then a Q4_K down on the q8_1 of its output (one launch more) or a
//! Q5_K down on the f32 output.
//!
//! - The norm writes the f32 activation into the boundary's handoff region
//!   and its q8_1 form into the piece's own scratch; the router and the
//!   shared expert read the f32 form there.
//! - `handoff` (`ds41_ffn_handoff`, the host tier's launch:
//!   [`bloomery_gpu::host::handoff`]) writes the handoff straight into its
//!   image in the host-mapped page — the region's sequence word, the router's
//!   ids and weights, the activation — so no copy node carries it, and writes
//!   each slot's place in the card's routed stacks: the layer's row of the
//!   slot map's card copy at the id, [`HOST`] for an expert the card does not
//!   hold. The card's routed launches read those places, so which experts
//!   the card computes comes from the map alone. An id past the expert count
//!   has no place: it raises `FaultSite::ExpertId` there.
//! - The card's routed launches are its slots' gate·up, the q8_1 of those
//!   slots' columns of `h` alone (`Q4kSelKernels::enqueue_quantize_sel`) and
//!   the down gemv by the down stack's type ([`enqueue_fused_experts`]): a
//!   q4_K down `q4k_sel`'s entry reads, a q5_K one the kquant `_sel` over the
//!   same q8_1 form. A slot the host serves leaves its column of `h` as it
//!   was — a reset does not clear it, so after a fault it can hold a NaN —
//!   and no card launch reads that column.
//! - The go signals the host tier; the wait takes its answer back
//!   ([`bloomery_gpu::hybrid`]); the wait sits after the shadow work.
//! - HC_PRE sits after the go: its result feeds only this sub-layer's HC_POST
//!   and the next sub-layer's fold (the lag), so it runs while the host
//!   computes. So do the card's routed experts and the shared expert.
//! - A caller may hand the layer more work for that shadow
//!   ([`ShadowWork`], [`FfnPiece::enqueue_shadowed`]): it is enqueued after
//!   the piece's own and before the wait, so the join's inputs never queue
//!   behind it.
//! - A layer whose slot-map row holds no card expert runs no routed launch.
//! - One launch joins (`ds41_ffn_post`, `ds41_ffn_post_streams`): the combine
//!   ([`combine_elem`]) of the card's slots, the host's partial sum read in
//!   place through the mapping and the shared expert's output, then HC_POST
//!   of it by the HC_POST launch's own rule — the new streams and, except
//!   into an engram layer and after the last layer, the next sub-layer's
//!   fold. The combine's output is written too, for a gate to read.
//! - The join is built for 6 slots a token (this piece's) and 8 (`_8`
//!   entries, for a program with eight), the handoff also for 10: their
//!   launchers ([`FfnKernels::enqueue_handoff`], [`FfnKernels::enqueue_post`])
//!   pick the entry from the slot count (the handoff's own table,
//!   [`Slots::of`] for the join) and refuse any other by name. A tier layer's
//!   card sum ([`FfnBatchKernels::enqueue_card_acc_tier`]) also has the
//!   four-slot entry a top-4 program runs; the launchers with no four-slot
//!   entry refuse that count by name.
//!
//! Launches per layer: ten kernels with card experts, seven without (one more
//! each for a shared down projection that reads q8_1), and
//! each layer's two stream memory-operation batches (go, wait); no copy.
//! A caller's shadow work is its own and is not counted here.
//!
//! A layer is enqueued in two halves — up to the shadow work
//! ([`FfnPiece::enqueue_go_half`], itself the front up to the go and the
//! shadow after it) and from the wait on ([`FfnPiece::enqueue_join_half`])
//! — on one row of the piece's buffers. A
//! pass whose two rows run one layer apart puts the other row's layer
//! between them, so every buffer a layer writes is the row's own; only
//! HC_PRE's reduction scratch, which its own launches consume, is shared.
//!
//! The piece takes the resident weights at enqueue, not at [`FfnPiece::new`]:
//! the engine keeps its weights and its chain body apart, and the body that
//! holds this piece cannot borrow them. `new` resolves every name once; an
//! enqueue looks each up and checks its format.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegShape, LegStacks, QuantRun};
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::gemm::GemmRoute;
pub use bloomery_gpu::host::handoff::Handoff;
use bloomery_gpu::host::handoff::HandoffKernels;
use bloomery_gpu::host::rank::Sums;
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::tier::TierTarget;
use bloomery_gpu::hybrid::{Boundary, HOST, HandoffTarget, HostExperts, Hybrid, SlotMap};
use bloomery_gpu::kquant::act::Act;
use bloomery_gpu::kquant::sel::SelDown;
use bloomery_gpu::q4k_sel::QuantSel;
use bloomery_gpu::quant_sel::QuantSelKernels;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, FaultSink, Gpu, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::{host, names};
use model::moe::UNION_MAX_COLS;
use model::placement::accepts::{self, LegArms, Program};
use model::placement::kernels::{self, DownForms};
use model::placement::{Role, tp};
use runtime::combine::Slots;

use crate::chain::join::TRUNK;
use crate::dense::{Dense, DenseKernels};
use crate::experts::{ExpertGateUp, ExpertKernels};
use crate::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcFn, HcFnArgs, HcKernels, HcPostArgs,
    HcPreScratch, hc_fold_elem, hc_post_elem,
};
use crate::router::{N_EXPERT, N_USED, RouterKernels, RouterOut};
use crate::transpose::TransposeKernels;

mod batch;
mod tier;
pub use batch::{
    BatchLayer, BlockIo, CardAccTier, CardGather, ChunkIo, FfnBatch, FfnBatchKernels, JoinIo,
    LEG_PIECE, LegBlock, TiledGateUp, enqueue_leg_block,
};
// The card half of the combine, and the places and sum launches the common
// card slots own (`bloomery_gpu::card_slots`).
pub use bloomery_gpu::card_slots::{
    CardAcc, CardJoin, DownRows, Places, SharedJoin, card_sum_elem,
};
pub use tier::{Ds41Tier, STAGE_TIER, TierPiece};

/// What the enqueue path's errors name.
const ENQUEUE: &str = "FfnPiece::enqueue";

/// Work a caller puts in a layer's host-leg shadow besides the piece's own:
/// enqueued on the step's stream after the layer's go and the piece's own
/// shadow work, and before the wait. It must read nothing the
/// host tier writes for the layer and write nothing the layer's join reads.
pub trait ShadowWork {
    /// Enqueue the work. Asynchronous, allocation-free, capturable.
    fn enqueue(&mut self, gpu: &Gpu) -> Result<(), GpuError>;
}

/// Threads per block of the combine-and-HC_POST.
const POST_THREADS: u32 = 256;

// The join's contracts spell the slot count as a literal.
const _: () = assert!(N_USED == 6);

/// Slots a token of the `_4` entries: the join's first instance
/// ([`Slots::Four`]), whose contracts spell it as a literal.
const SLOTS_4: usize = 4;
const _: () = assert!(SLOTS_4 == 4);

/// Slots a token of the `_8` entries: the join's second instance
/// ([`Slots::Eight`]), whose contracts spell it as a literal.
const SLOTS_8: usize = 8;
const _: () = assert!(SLOTS_8 == 8);
// The instance table's counts are the entries' own.
const _: () =
    assert!(Slots::Four.n() == SLOTS_4 && Slots::Six.n() == N_USED && Slots::Eight.n() == SLOTS_8);

/// The combine of one output value: the card's slots in slot order by
/// fused multiply-adds from zero ([`card_sum_elem`]), then `(acc + hsum) +
/// shexp` ([`join_elem`]). The one rule the device and a gate's host side
/// both run; a prompt batch runs its two halves in two launches.
#[inline(always)]
pub fn combine_elem(
    down: [f32; N_USED],
    w: [f32; N_USED],
    card: [bool; N_USED],
    hsum: f32,
    shexp: f32,
) -> f32 {
    join_elem(card_sum_elem(down, w, card), hsum, shexp)
}

/// The combine's join: the card sum, then the host's, then the shared
/// expert's, `(acc + hsum) + shexp`.
#[inline(always)]
pub fn join_elem(acc: f32, hsum: f32, shexp: f32) -> f32 {
    (acc + hsum) + shexp
}

#[cuda_module]
mod ffn_kernels {
    use super::*;

    /// The combine and HC_POST with the next fold, one thread per value `d`
    /// ([`combine_post_at`]): writes the combine's `y[d]`, the four new
    /// streams at `d` to `out` and their fold by the HC_PRE result's `pre` to
    /// `fold`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows,
            fold.len() >= rows
        )
    )]
    pub fn ds41_ffn_post(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows, and the launch contract gives every length the
        // helper's contract asks for.
        let (yv, o, pre) = unsafe { combine_post_at(&a, rows, n_card, d) };
        // SAFETY: d < rows <= y.len() and fold.len(), and d + 3·rows < 4·rows
        // <= out.len(), by the launch contract; thread d is the only writer of
        // y[d], fold[d] and the four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
            *fold.get_unchecked_mut(d) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post`] without the fold: into an engram layer and after the
    /// last layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows
        )
    )]
    pub fn ds41_ffn_post_streams(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows (checked above), and the launch contract gives
        // every length the helper's contract asks for.
        let (yv, o, _) = unsafe { combine_post_at(&a, rows, n_card, d) };
        // SAFETY: d < rows <= y.len() and d + 3·rows < 4·rows <= out.len(),
        // by the launch contract; thread d is the only writer of y[d] and the
        // four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
        }
    }

    /// [`ds41_ffn_post`] of eight slots a token ([`combine_post_at_n`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 8 * rows,
            w.len() >= 8,
            sel.len() >= 8,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows,
            fold.len() >= rows
        )
    )]
    pub fn ds41_ffn_post_8(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_8 slots.
        let (yv, o, pre) = unsafe { combine_post_at_n::<SLOTS_8>(&a, rows, n_card, d) };
        // SAFETY: d < rows <= y.len() and fold.len(), and d + 3·rows < 4·rows
        // <= out.len(), by the launch contract; thread d is the only writer of
        // y[d], fold[d] and the four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
            *fold.get_unchecked_mut(d) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post_8`] without the fold: into an engram layer and after
    /// the last layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 8 * rows,
            w.len() >= 8,
            sel.len() >= 8,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows
        )
    )]
    pub fn ds41_ffn_post_streams_8(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows (checked above), and the launch contract gives
        // every length the helper's contract asks for at SLOTS_8 slots.
        let (yv, o, _) = unsafe { combine_post_at_n::<SLOTS_8>(&a, rows, n_card, d) };
        // SAFETY: d < rows <= y.len() and d + 3·rows < 4·rows <= out.len(),
        // by the launch contract; thread d is the only writer of y[d] and the
        // four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
        }
    }
}

/// What the combine-and-HC_POST launches read: `down` the six slots' down
/// outputs slot-major (`rows` each), `w` the six routing weights, `sel` each
/// slot's place (a slot is the card's when its place is below `n_card`, and
/// every other slot's rows are never read), `hsum` the host's partial sum,
/// `shexp` the shared expert's output, `res` the four streams the sub-layer
/// read (`rows` each) and `hc` its HC_PRE result.
struct PostIn<'a> {
    down: &'a [f32],
    w: &'a [f32],
    sel: &'a [u32],
    hsum: &'a [f32],
    shexp: &'a [f32],
    res: &'a [f32],
    hc: &'a [f32],
}

/// Value `d`'s combine ([`combine_elem`]), then HC_POST of it (`hc_post_elem`,
/// the HC_POST launch's own rule): the combine's `y`, the four new stream
/// values and `pre`.
///
/// SAFETY: `d < rows`, `down.len() >= 6 * rows`, `w.len()` and `sel.len() >=
/// 6`, `hsum.len()` and `shexp.len() >= rows`, `res.len() >= 4 * rows` and
/// `hc.len() >= 24`.
#[inline(always)]
unsafe fn combine_post_at(
    a: &PostIn<'_>,
    rows: usize,
    n_card: u32,
    d: usize,
) -> (f32, [f32; 4], [f32; 4]) {
    let mut dv = [0.0f32; N_USED];
    let mut wv = [0.0f32; N_USED];
    let mut card = [false; N_USED];
    let mut j = 0usize;
    while j < N_USED {
        // SAFETY: j < 6 <= sel.len() and w.len() by this fn's contract.
        let (place, wj) = unsafe { (*a.sel.get_unchecked(j), *a.w.get_unchecked(j)) };
        if place < n_card {
            card[j] = true;
            wv[j] = wj;
            // SAFETY: j < 6 and d < rows, so j·rows + d < 6·rows <=
            // down.len() by this fn's contract.
            dv[j] = unsafe { *a.down.get_unchecked(j * rows + d) };
        }
        j += 1;
    }
    // SAFETY: d < rows <= hsum.len() and shexp.len(); d + 3·rows < 4·rows <=
    // res.len(); 23 < 24 <= hc.len() — all by this fn's contract.
    let (hs, sh, r, pre, post, comb) = unsafe {
        (
            *a.hsum.get_unchecked(d),
            *a.shexp.get_unchecked(d),
            [
                *a.res.get_unchecked(d),
                *a.res.get_unchecked(d + rows),
                *a.res.get_unchecked(d + 2 * rows),
                *a.res.get_unchecked(d + 3 * rows),
            ],
            [
                *a.hc.get_unchecked(0),
                *a.hc.get_unchecked(1),
                *a.hc.get_unchecked(2),
                *a.hc.get_unchecked(3),
            ],
            [
                *a.hc.get_unchecked(4),
                *a.hc.get_unchecked(5),
                *a.hc.get_unchecked(6),
                *a.hc.get_unchecked(7),
            ],
            [
                *a.hc.get_unchecked(8),
                *a.hc.get_unchecked(9),
                *a.hc.get_unchecked(10),
                *a.hc.get_unchecked(11),
                *a.hc.get_unchecked(12),
                *a.hc.get_unchecked(13),
                *a.hc.get_unchecked(14),
                *a.hc.get_unchecked(15),
                *a.hc.get_unchecked(16),
                *a.hc.get_unchecked(17),
                *a.hc.get_unchecked(18),
                *a.hc.get_unchecked(19),
                *a.hc.get_unchecked(20),
                *a.hc.get_unchecked(21),
                *a.hc.get_unchecked(22),
                *a.hc.get_unchecked(23),
            ],
        )
    };
    let y = combine_elem(dv, wv, card, hs, sh);
    (y, hc_post_elem(y, r, post, &comb), pre)
}

/// The four stream values of value `d` into `out`, stream-major (`rows`
/// each).
///
/// SAFETY: `d < rows`, `out.len() >= 4 * rows`, and the four slots are the
/// calling thread's alone.
#[inline(always)]
unsafe fn store4(out: &mut DisjointSlice<f32>, rows: usize, d: usize, o: [f32; 4]) {
    // SAFETY: d + 3·rows < 4·rows <= out.len() by this fn's contract.
    unsafe {
        *out.get_unchecked_mut(d) = o[0];
        *out.get_unchecked_mut(d + rows) = o[1];
        *out.get_unchecked_mut(d + 2 * rows) = o[2];
        *out.get_unchecked_mut(d + 3 * rows) = o[3];
    }
}

/// Value `d`'s combine of `N` slots, then HC_POST of it: [`combine_post_at`]
/// with `a.down`, `a.w` and `a.sel` holding `N` slots, the card sum run as it
/// reads them — `acc = fma(down_j, w_j, acc)` from zero for each slot `j`
/// below `N` whose place is below `n_card`, in ascending `j` (the order
/// `runtime::combine::combine` pins; this order is the gate), then
/// [`join_elem`]. The combine's `y`, the four new stream values and `pre`.
///
/// SAFETY: `d < rows`, `down.len() >= N * rows`, `w.len()` and `sel.len() >=
/// N`, `hsum.len()` and `shexp.len() >= rows`, `res.len() >= 4 * rows` and
/// `hc.len() >= 24`.
#[inline(always)]
unsafe fn combine_post_at_n<const N: usize>(
    a: &PostIn<'_>,
    rows: usize,
    n_card: u32,
    d: usize,
) -> (f32, [f32; 4], [f32; 4]) {
    let mut acc = 0.0f32;
    for j in 0..N {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: j < N <= sel.len() and w.len() by this fn's contract.
        let (place, wj) = unsafe { (*a.sel.get_unchecked(j), *a.w.get_unchecked(j)) };
        if place < n_card {
            // SAFETY: j < N and d < rows, so j·rows + d < N·rows <=
            // down.len() by this fn's contract.
            let dj = unsafe { *a.down.get_unchecked(j * rows + d) };
            acc = dj.mul_add(wj, acc);
        }
    }
    // SAFETY: d < rows <= hsum.len() and shexp.len(); d + 3·rows < 4·rows <=
    // res.len(); 23 < 24 <= hc.len() — all by this fn's contract.
    let (hs, sh, r, pre, post, comb) = unsafe {
        (
            *a.hsum.get_unchecked(d),
            *a.shexp.get_unchecked(d),
            [
                *a.res.get_unchecked(d),
                *a.res.get_unchecked(d + rows),
                *a.res.get_unchecked(d + 2 * rows),
                *a.res.get_unchecked(d + 3 * rows),
            ],
            [
                *a.hc.get_unchecked(0),
                *a.hc.get_unchecked(1),
                *a.hc.get_unchecked(2),
                *a.hc.get_unchecked(3),
            ],
            [
                *a.hc.get_unchecked(4),
                *a.hc.get_unchecked(5),
                *a.hc.get_unchecked(6),
                *a.hc.get_unchecked(7),
            ],
            [
                *a.hc.get_unchecked(8),
                *a.hc.get_unchecked(9),
                *a.hc.get_unchecked(10),
                *a.hc.get_unchecked(11),
                *a.hc.get_unchecked(12),
                *a.hc.get_unchecked(13),
                *a.hc.get_unchecked(14),
                *a.hc.get_unchecked(15),
                *a.hc.get_unchecked(16),
                *a.hc.get_unchecked(17),
                *a.hc.get_unchecked(18),
                *a.hc.get_unchecked(19),
                *a.hc.get_unchecked(20),
                *a.hc.get_unchecked(21),
                *a.hc.get_unchecked(22),
                *a.hc.get_unchecked(23),
            ],
        )
    };
    let y = join_elem(acc, hs, sh);
    (y, hc_post_elem(y, r, post, &comb), pre)
}

/// A layer's routed stacks on the card: the experts its slot-map row puts
/// there, slot `s` at rows `s · rows .. (s + 1) · rows` of each, with each
/// stack's file type beside it ([`CardStacks::gate_up_ty`],
/// [`CardStacks::down_ty`]) — what a launcher that reads the stacks reads by
/// instead of assuming the pair it was built for. Either the fused layer
/// ([`accepts::ds41_fused`]: gate and up q3_K, `ff` rows of `n_embd` per
/// expert, and a down of `n_embd` rows of `ff` that is q4_K or q5_K), which
/// the chain's own launches read, or a layer of types the common card leg
/// runs ([`accepts::ds41_pair`]), which [`CardStacks::legs`] hands to it.
#[derive(Clone, Copy)]
pub struct CardStacks<'a> {
    pub gate: &'a DeviceTensor<u32>,
    pub up: &'a DeviceTensor<u32>,
    pub down: &'a DeviceTensor<u32>,
    gate_up_ty: GgmlType,
    down_ty: GgmlType,
    /// The stacks as the common card leg reads them, for a layer on it;
    /// `None` for the fused layer.
    legs: Option<LegStacks<'a>>,
}

impl<'a> CardStacks<'a> {
    /// The stacks a placed load holds for layer `layer`: its routed gate, up
    /// and down file tensors in `w`, which a plan uploads with the experts of
    /// its card segment in slot order. `None` when `w` holds none of the
    /// three; an error when it holds some, holds one in another format or of a
    /// type the chain's card experts do not run, or holds a gate and up and a
    /// down that do not pair ([`accepts::ds41_pair`]).
    pub fn of(w: &'a Weights, layer: usize) -> Result<Option<CardStacks<'a>>, GpuError> {
        const WHAT: &str = "CardStacks::of";
        let names = [
            names::ffn_gate_exps(layer),
            names::ffn_up_exps(layer),
            names::ffn_down_exps(layer),
        ];
        let held = names.each_ref().map(|n| w.name_of(n));
        if held.iter().all(Option::is_none) {
            return Ok(None);
        }
        let [Some(gate), Some(up), Some(down)] = held else {
            let missing = held.iter().position(Option::is_none).unwrap_or(0);
            return Err(GpuError::Tensor {
                what: WHAT,
                name: names[missing].clone(),
                need: "a resident routed stack beside the layer's other two",
            });
        };
        let legs = LegStacks::of_arms(
            w,
            [gate, up, down],
            LegArms {
                gate_up: accepts::ds41_gate_up,
                down: accepts::ds41_down,
            },
            (
                "a resident routed gate or up stack of a type V4.1's card experts run \
                 (`accepts::ds41_gate_up`)",
                "a resident routed down stack of a type V4.1's card experts run \
                 (`accepts::ds41_down`)",
            ),
        )?;
        let types = [legs.gate_up_ty(), legs.gate_up_ty(), legs.down_ty()];
        if accepts::leg_runs(Program::Deepseek41Chain, types) != Some(true) {
            return Err(GpuError::Tensor {
                what: WHAT,
                name: names[2].clone(),
                need: "a down stack of a type that pairs with the layer's gate and up \
                       (`accepts::ds41_pair`)",
            });
        }
        let [gate, up, down] = legs.words().map_err(|_| GpuError::Tensor {
            what: WHAT,
            name: names[0].clone(),
            need: "routed stacks in the file's blocks",
        })?;
        let fused = accepts::ds41_fused(types);
        Ok(Some(CardStacks {
            gate,
            up,
            down,
            gate_up_ty: types[0],
            down_ty: types[2],
            legs: (!fused).then_some(legs),
        }))
    }

    /// The fused layer ([`accepts::ds41_fused`]) over stacks a caller holds
    /// and the down type it reads them in, which the chain's own launches
    /// read. The gate and up are the fused rule's q3_K; a down type outside
    /// [`accepts::DS41_FUSED_DOWNS`] is the down dispatch's named refusal.
    #[must_use]
    pub fn fused_over(
        gate: &'a DeviceTensor<u32>,
        up: &'a DeviceTensor<u32>,
        down: &'a DeviceTensor<u32>,
        down_ty: GgmlType,
    ) -> CardStacks<'a> {
        CardStacks {
            gate,
            up,
            down,
            gate_up_ty: GgmlType::Q3_K,
            down_ty,
            legs: None,
        }
    }

    /// The gate's and the up's type.
    #[must_use]
    pub fn gate_up_ty(&self) -> GgmlType {
        self.gate_up_ty
    }

    /// The down's type.
    #[must_use]
    pub fn down_ty(&self) -> GgmlType {
        self.down_ty
    }

    /// Whether the layer is the fused one the chain's own launches read
    /// ([`accepts::ds41_fused`]); else its stacks run on the common card leg.
    #[must_use]
    pub fn fused(&self) -> bool {
        self.legs.is_none()
    }

    /// The stacks as the common card leg reads them, for a layer that runs on
    /// it.
    #[must_use]
    pub fn legs(&self) -> Option<&LegStacks<'a>> {
        self.legs.as_ref()
    }
}

/// The buffers the piece shares with the rest of the step, for one layer.
pub struct FfnIo<'a> {
    /// The streams the sub-layer reads — HC_PRE's input and HC_POST's
    /// residual: [`HC_STREAMS`] · `n_embd` f32.
    pub streams: &'a DeviceBuffer<f32>,
    /// The folded input the norm reads: `n_embd` f32.
    pub fold_in: &'a DeviceBuffer<f32>,
    /// The new streams: [`HC_STREAMS`] · `n_embd` f32.
    pub streams_out: &'a mut DeviceBuffer<f32>,
    /// The next sub-layer's fold, `n_embd` f32: `Some` exactly where the
    /// layer folds ([`FfnPiece::folds`]).
    pub fold_out: Option<&'a mut DeviceBuffer<f32>>,
    /// The slot map's card copy: a row of `n_expert` places per layer of the
    /// host tier's map, in its order.
    pub slots: &'a DeviceTensor<u32>,
}

/// One layer's tensor names and constants, resolved at load.
struct LayerCfg {
    norm: String,
    router: String,
    bias: String,
    sh_gate: String,
    sh_up: String,
    sh_down: String,
    hc_fn: String,
    hc_scale: String,
    hc_base: String,
    /// Experts the slot map puts on the card: 0 runs no routed launch.
    n_card: usize,
    /// The layer's row in the map's card copy, as a word offset.
    row_off: usize,
    limit: f32,
    limit_shared: f32,
    /// The shared down projection reads its input in q8_1: one launch more.
    sh_down_q8_1: bool,
    /// The shared gate and up each run a gemv over the f32 row ([`sh_rows`]):
    /// two launches more.
    sh_rows: bool,
    /// HC_POST with the next fold (`false`: HC_POST alone).
    fold: bool,
}

/// A layer's MoE sub-layer on one row between its front and its shadow
/// ([`FfnPiece::enqueue_go_front`], [`FfnPiece::enqueue_go_shadow`]): the
/// layer's weights and card stacks as the front resolved and checked them.
pub struct GoFront<'w> {
    layer: usize,
    row: usize,
    i: usize,
    lw: LayerWeights<'w>,
    card: Option<CardStacks<'w>>,
}

impl GoFront<'_> {
    /// The layer and the row the front ran.
    #[must_use]
    pub fn at(&self) -> (usize, usize) {
        (self.layer, self.row)
    }
}

/// One layer's resident weights, looked up and format-checked at enqueue.
struct LayerWeights<'w> {
    gain: &'w DeviceBuffer<f32>,
    router: &'w DeviceTensor<f32>,
    bias: &'w DeviceBuffer<f32>,
    sh_gate: &'w DevWeight,
    sh_up: &'w DevWeight,
    /// A shared gate and up each read over the f32 row ([`sh_rows`]).
    sh_rows: Option<[Dense<'w>; 2]>,
    sh_down: Dense<'w>,
    hc: HcFn<'w>,
}

impl<'w> LayerWeights<'w> {
    /// Layer `c`'s tensors in `w`, each in the format its launch reads, with
    /// the HC_PRE constants; the shared expert's down projection takes `ff`
    /// values onto `n_embd` rows.
    fn resolve(
        c: &LayerCfg,
        w: &'w Weights,
        [n_embd, ff]: [usize; 2],
        hc_eps: f32,
        hc_iters: u32,
    ) -> Result<LayerWeights<'w>, GpuError> {
        let gain = w.f32_buf(ENQUEUE, &c.norm)?;
        let router = w.f32_tensor(ENQUEUE, &c.router)?;
        let bias = w.f32_buf(ENQUEUE, &c.bias)?;
        let (sh_gate, sh_up) = (
            w.resident(ENQUEUE, &c.sh_gate)?,
            w.resident(ENQUEUE, &c.sh_up)?,
        );
        let sh_rows = if c.sh_rows {
            Some([
                Dense::of(w, &c.sh_gate, n_embd, ff)?,
                Dense::of(w, &c.sh_up, n_embd, ff)?,
            ])
        } else {
            None
        };
        let sh_down = Dense::of(w, &c.sh_down, ff, n_embd)?;
        Ok(LayerWeights {
            gain,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_rows,
            sh_down,
            hc: HcFn::of(
                w,
                ENQUEUE,
                [&c.hc_fn, &c.hc_scale, &c.hc_base],
                (hc_eps, hc_iters),
            )?,
        })
    }
}

/// The piece's own kernels, the handoff and the join: loaded once by
/// [`FfnPiece::new`], or alone by a gate that drives the handoff with ids a
/// router cannot produce.
pub struct FfnKernels {
    module: ffn_kernels::LoadedModule,
    handoff: HandoffKernels,
    /// The wide F32 tile GEMM the router's scores run on.
    g32: bloomery_gpu::gemm::Gemm32Kernels,
}

impl FfnKernels {
    /// Load the piece's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FfnKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(ffn_kernels, ctx)? };
        Ok(FfnKernels {
            module,
            handoff: HandoffKernels::load(ctx)?,
            g32: bloomery_gpu::gemm::Gemm32Kernels::load(ctx)?,
        })
    }

    /// Enqueue the handoff of `target.layout.n_used` slots a token: the host
    /// tier's launch ([`HandoffKernels::enqueue_handoff`]), which picks the
    /// entry from the slot count and refuses any other by name.
    pub fn enqueue_handoff(
        &self,
        stream: &CudaStream,
        h: &Handoff<'_>,
        target: HandoffTarget<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.handoff.enqueue_handoff(stream, h, target, fault, sel)
    }

    /// Enqueue the handoff of a tier layer of eight slots a token, the tier
    /// places from `tmap` into `tsel` and the tier image
    /// ([`HandoffKernels::enqueue_handoff_tier`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's routing, maps, targets, sink and places (rust-quality R8)"
    )]
    pub fn enqueue_handoff_tier(
        &self,
        stream: &CudaStream,
        h: &Handoff<'_>,
        tmap: &DeviceBuffer<u32>,
        targets: (HandoffTarget<'_>, TierTarget<'_>),
        fault: FaultSink,
        sels: (&mut DeviceBuffer<u32>, &mut DeviceBuffer<u32>),
    ) -> Result<(), GpuError> {
        self.handoff
            .enqueue_handoff_tier(stream, h, tmap, targets, fault, sels)
    }

    /// Enqueue the combine and HC_POST of `p.n_used` slots a token
    /// (`ds41_ffn_post`/`ds41_ffn_post_streams` for 6, their `_8` entries for
    /// 8; any other count, four included, is refused by name), one thread per value `d <
    /// p.rows`: the combine's `y[d]`, the four new streams at `d` into `out`
    /// and, when `fold` is given, their fold by the HC_PRE result's `pre`
    /// into it. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_post(
        &self,
        stream: &CudaStream,
        p: &Post<'_>,
        y: &mut DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        fold: Option<&mut DeviceBuffer<f32>>,
    ) -> Result<(), GpuError> {
        let slots = slots_of("ds41_ffn_post", p.n_used)?;
        let what = match (slots, fold.is_some()) {
            (Slots::Four, _) => return Err(no_entry("ds41_ffn_post", slots)),
            (Slots::Six, true) => "ds41_ffn_post",
            (Slots::Six, false) => "ds41_ffn_post_streams",
            (Slots::Eight, true) => "ds41_ffn_post_8",
            (Slots::Eight, false) => "ds41_ffn_post_streams_8",
        };
        let grid = launch_u32(what, "grid", p.rows.div_ceil(POST_THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, POST_THREADS, 0);
        let rows = launch_u32(what, "rows", p.rows)?;
        let n_card = launch_u32(what, "n_card", p.n_card)?;
        let m = &self.module;
        match (slots, fold) {
            (Slots::Four, _) => return Err(no_entry(what, slots)),
            (Slots::Six, Some(fold)) => {
                let prep = m.prepare_ds41_ffn_post(cfg)?;
                m.ds41_ffn_post(
                    stream, &prep, p.down, p.w, p.sel, p.hsum, p.shexp, p.res, p.hc, rows, n_card,
                    y, out, fold,
                )?;
            }
            (Slots::Six, None) => {
                let prep = m.prepare_ds41_ffn_post_streams(cfg)?;
                m.ds41_ffn_post_streams(
                    stream, &prep, p.down, p.w, p.sel, p.hsum, p.shexp, p.res, p.hc, rows, n_card,
                    y, out,
                )?;
            }
            (Slots::Eight, Some(fold)) => {
                let prep = m.prepare_ds41_ffn_post_8(cfg)?;
                m.ds41_ffn_post_8(
                    stream, &prep, p.down, p.w, p.sel, p.hsum, p.shexp, p.res, p.hc, rows, n_card,
                    y, out, fold,
                )?;
            }
            (Slots::Eight, None) => {
                let prep = m.prepare_ds41_ffn_post_streams_8(cfg)?;
                m.ds41_ffn_post_streams_8(
                    stream, &prep, p.down, p.w, p.sel, p.hsum, p.shexp, p.res, p.hc, rows, n_card,
                    y, out,
                )?;
            }
        }
        Ok(())
    }
}

/// What [`FfnKernels::enqueue_post`] reads: `down` the `n_used` slots' down
/// outputs slot-major (`rows` each), `w` the `n_used` routing weights, `sel`
/// each slot's place (the card's below `n_card`; every other slot's rows are
/// never read), `hsum` the host's partial sum, `shexp` the shared expert's
/// output (`rows` each), `res` the four streams the sub-layer read (`rows`
/// each) and `hc` its HC_PRE result ([`HC_MIX`] values).
pub struct Post<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub hsum: &'a DeviceBuffer<f32>,
    pub shexp: &'a DeviceBuffer<f32>,
    pub res: &'a DeviceBuffer<f32>,
    pub hc: &'a DeviceBuffer<f32>,
    pub rows: usize,
    pub n_card: usize,
    pub n_used: usize,
}

/// The join's instance for `n_used` slots a token ([`Slots::of`]), or
/// `what`'s shape error naming the count.
fn slots_of(what: &'static str, n_used: usize) -> Result<Slots, GpuError> {
    Slots::of(n_used).map_err(|e| GpuError::Shape {
        what,
        detail: e.to_string(),
    })
}

/// `what`'s shape error for an instance of the join it has no entry for: of
/// the join's launchers only the tier card sum
/// ([`FfnBatchKernels::enqueue_card_acc_tier`]) has the four-slot one.
fn no_entry(what: &'static str, slots: Slots) -> GpuError {
    GpuError::Shape {
        what,
        detail: format!(
            "{} routed slots a token: this launch has no entry for them",
            slots.n()
        ),
    }
}

/// The piece's own buffers, as the last enqueued layer left them — what a
/// gate reads back.
pub struct FfnTaps<'a> {
    /// The router's per-expert logits and scores and its routing.
    pub router: &'a RouterOut,
    /// Each slot's place in the card's routed stacks, [`HOST`] for the host's.
    pub sel: &'a DeviceBuffer<u32>,
    /// The routed SwiGLU outputs, slot-major (`ff` each): the card's slots.
    pub h: &'a DeviceBuffer<f32>,
    /// The routed down outputs, slot-major (`n_embd` each): the card's slots.
    pub down: &'a DeviceBuffer<f32>,
    /// The shared expert's SwiGLU output and its down output.
    pub shexp_h: &'a DeviceBuffer<f32>,
    pub shexp: &'a DeviceBuffer<f32>,
    /// The combine's output.
    pub y: &'a DeviceBuffer<f32>,
    /// HC_PRE's result, [`HC_MIX`] values.
    pub hc: &'a DeviceBuffer<f32>,
}

/// One row's buffers: everything a layer's launches write, read back by its
/// join or by a gate. A pass whose rows run one layer apart enqueues the
/// other row's whole layer between a row's go and its join, so each row
/// keeps its own. A row holds an activation form only where a layer of the
/// piece reads it ([`Reads`]).
struct FfnRow {
    act_x: Q8Act,
    rout: RouterOut,
    sel: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    /// The card slots' columns of `h` in q8_1, for the fused layer's down.
    act_h: Option<Q8Act>,
    down: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    /// The shared gate's and up's f32 rows, for a pair of types read over the
    /// f32 row ([`sh_rows`]).
    sh_gu: Option<[DeviceBuffer<f32>; 2]>,
    /// `sh_h` in q8_1, for a K-quant shared down projection.
    act_sh: Option<Q8Act>,
    sh_y: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    hc_out: DeviceBuffer<f32>,
    /// The common card leg's buffers, for a layer of types it runs
    /// ([`FfnPiece::prepare_leg`]); its rows go to `down`.
    leg: Option<LegBufs>,
}

/// The activation forms of `h` a piece's layers read beside the f32 rows,
/// from their types at load: each row holds these and no other.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Reads {
    /// A fused layer ([`accepts::ds41_fused`]): its down reads the card slots'
    /// q8_1.
    fused: bool,
    /// A fused layer whose down is q5_K ([`accepts::DS41_FUSED_DOWNS`]): the
    /// kquant `_sel` down ([`FfnPiece`]'s `fused_q5k`, and the tile path's
    /// `KquantTileKernels`, which a q4_K-only file loads none of).
    fused_q5k_down: bool,
    /// A shared down of a K-quant reads `sh_h` in q8_1.
    sh_q8_1: bool,
    /// A shared gate and up of types read over the f32 row write rows of
    /// their own ([`sh_rows`]).
    sh_rows: bool,
    /// The forms the common card leg's downs read ([`kernels::down_act`] at
    /// `ff`), for its buffers ([`FfnPiece::prepare_leg`]).
    leg: DownForms,
}

impl Reads {
    /// These forms and those a layer of routed stacks `routed` (`[gate, up,
    /// down]`, none for a layer without them) and shared expert `[gate, up,
    /// down]` reads at `ff`.
    fn with_layer(
        mut self,
        routed: Option<[GgmlType; 3]>,
        [shared_gate, shared_up, shared_down]: [GgmlType; 3],
        ff: usize,
    ) -> Reads {
        if let Some(t) = routed {
            if accepts::ds41_fused(t) {
                self.fused = true;
                self.fused_q5k_down |= matches!(t[2], GgmlType::Q5_K);
            } else if accepts::leg_runs(Program::Deepseek41Chain, t) == Some(true) {
                self.leg = self.leg.with(kernels::down_act(t[2], ff as u64));
            }
        }
        self.sh_q8_1 |= sh_down_reads_q8_1(shared_down);
        self.sh_rows |= sh_rows(shared_gate, shared_up);
        self
    }
}

/// Whether a shared gate of type `gate` and up of type `up` each run a gemv
/// over the f32 row and then the clamped SwiGLU: MXFP8 or BF16, which no
/// fused pair reads.
fn sh_rows(gate: GgmlType, up: GgmlType) -> bool {
    let rows = |t| matches!(t, GgmlType::MXFP8 | GgmlType::BF16);
    rows(gate) && rows(up)
}

/// The shared expert's h over the f32 row `x` for a gate and up read over
/// the f32 row ([`sh_rows`]): a gemv each into `g` and `u`, then the clamped
/// SwiGLU at `limit` of the first `h.len()` values into `h`.
fn enqueue_sh_rows(
    gpu: &Gpu,
    dense: &DenseKernels,
    [gate, up]: [Dense<'_>; 2],
    (x, limit): (&DeviceBuffer<f32>, f32),
    (sink, [g, u], h): (
        FaultSink,
        &mut [DeviceBuffer<f32>; 2],
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    dense.enqueue(gpu, gate, x, None, g)?;
    dense.enqueue(gpu, up, x, None, u)?;
    let n = h.len();
    gpu.elem().enqueue_swiglu_rule(
        gpu.stream(),
        Act::SwigluClamp { limit },
        (&*g, &*u),
        n,
        (sink, h),
    )
}

/// Whether a shared down of type `ty` reads its input in q8_1: the K-quant
/// downs.
fn sh_down_reads_q8_1(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q3_K | GgmlType::Q4_K)
}

/// Whether a down of type `ty` in `role` reads rows of `ff` values: `ff` a
/// whole number of the K unit a card part of it keeps ([`tp::card_blocks`] at
/// its format block); else that unit, `None` for a type of no block.
fn down_k_fits(ty: GgmlType, role: Role, ff: usize) -> Result<(), Option<u64>> {
    let unit = ty
        .blck_size()
        .and_then(|b| tp::card_blocks(ty, [b, 1], role).ok())
        .map(|[k, _]| k);
    match unit {
        Some(k) if (ff as u64).is_multiple_of(k) => Ok(()),
        _ => Err(unit),
    }
}

impl FfnRow {
    fn new(
        stream: &CudaStream,
        n_embd: usize,
        ff: usize,
        reads: Reads,
    ) -> Result<FfnRow, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let q8 = |on: bool, m: usize| on.then(|| Q8Act::with_k(stream, m, ff)).transpose();
        Ok(FfnRow {
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            rout: RouterOut::new(stream)?,
            sel: DeviceBuffer::zeroed(stream, N_USED)?,
            h: z(N_USED * ff)?,
            act_h: q8(reads.fused, N_USED)?,
            down: z(N_USED * n_embd)?,
            sh_h: z(ff)?,
            sh_gu: if reads.sh_rows {
                Some([z(ff)?, z(ff)?])
            } else {
                None
            },
            act_sh: q8(reads.sh_q8_1, 1)?,
            sh_y: z(n_embd)?,
            y: z(n_embd)?,
            mixes: z(HC_MIX)?,
            hc_out: z(HC_MIX)?,
            leg: None,
        })
    }

    fn device_bytes(&self) -> usize {
        let f32s = [
            &self.h,
            &self.down,
            &self.sh_h,
            &self.sh_y,
            &self.y,
            &self.mixes,
            &self.hc_out,
            &self.rout.logits,
            &self.rout.probs,
            &self.rout.weights,
        ];
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.sel.num_bytes()
            + self.rout.ids.num_bytes()
            + ROUTER_TICKET_BYTES
            + q8act_bytes(&self.act_x)
            + self.act_h.as_ref().map_or(0, q8act_bytes)
            + self.act_sh.as_ref().map_or(0, q8act_bytes)
            + self
                .sh_gu
                .as_ref()
                .map_or(0, |g| g.iter().map(DeviceBuffer::num_bytes).sum())
            + self.leg.as_ref().map_or(0, LegBufs::bytes)
    }
}

/// The MoE sub-layer of every layer of one slot map, built once at load.
pub struct FfnPiece {
    fused: FusedKernels,
    router: RouterKernels,
    experts: ExpertKernels,
    dense: DenseKernels,
    hc: HcKernels,
    /// The row-major-to-token-major copy of a prompt batch's shared down
    /// projection ([`FfnBatch`]).
    transpose: TransposeKernels,
    kernels: FfnKernels,
    layers: Range<usize>,
    cfg: Vec<LayerCfg>,
    n_embd: usize,
    ff: usize,
    n_expert: usize,
    rms_eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    scale: f32,
    /// Per row, its buffers ([`FfnPiece::with_rows`]).
    rows: Vec<FfnRow>,
    /// The activation forms the layers read, which the rows hold.
    reads: Reads,
    /// The kquant `_sel` down of a fused layer with a q5_K down
    /// ([`Reads::fused_q5k_down`]), loaded once here; `None` where no fused
    /// layer's down is q5_K, so a q4_K-only file loads no module for it.
    fused_q5k: Option<QuantSelKernels>,
    /// The common card leg's launches, for the layers of types it runs
    /// ([`FfnPiece::prepare_leg`]).
    leg: Option<CardLeg>,
    /// The grouped GEMM's route tables, one per distinct count of card
    /// experts the map puts on a layer ([`FfnPiece::gemm_route_at`]), for the
    /// default prompt block's card experts ([`batch::enqueue_block_experts`]).
    /// Empty when no layer holds card experts: a block that needs a table is
    /// refused by name then.
    routes: Vec<GemmRoute>,
    /// The expert counts [`FfnPiece::routes`]' tables hold, in their order.
    counts: Vec<usize>,
    /// HC_PRE's reduction scratch, which its own launches consume.
    hc_scratch: HcPreScratch,
}

impl FfnPiece {
    /// The piece for every layer of `map`, whose rows say which experts each
    /// layer's card stacks hold: names, constants, the kernels and the
    /// piece's scratch, all resolved here. One row. Load-time only.
    pub fn new(gpu: &Gpu, hp: &Hparams, map: &SlotMap) -> Result<FfnPiece, GpuError> {
        FfnPiece::with_rows(gpu, hp, map, 1)
    }

    /// [`FfnPiece::new`] with `rows` rows of buffers, for a pass whose rows
    /// run one layer apart. Load-time only.
    pub fn with_rows(
        gpu: &Gpu,
        hp: &Hparams,
        map: &SlotMap,
        rows: usize,
    ) -> Result<FfnPiece, GpuError> {
        const WHAT: &str = "FfnPiece::new";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        if rows == 0 {
            return Err(refuse("a piece of no rows".to_string()));
        }
        let (n_embd, ff, ex) = (hp.n_embd, hp.experts.ff, &hp.experts);
        if ex.n_expert != N_EXPERT || ex.n_used != N_USED || map.n_expert() != ex.n_expert {
            return Err(refuse(format!(
                "{} experts, {} used, a map of {}: the kernels are built for {N_EXPERT} and {N_USED}",
                ex.n_expert,
                ex.n_used,
                map.n_expert()
            )));
        }
        if hp.hc.streams != HC_STREAMS {
            return Err(refuse(format!(
                "{} streams: want {HC_STREAMS}",
                hp.hc.streams
            )));
        }
        let layers = map.layers();
        if layers.end > hp.n_layer {
            return Err(refuse(format!(
                "the map's layers {layers:?} pass the model's {}",
                hp.n_layer
            )));
        }
        let mut cfg = Vec::with_capacity(layers.len());
        let mut reads = Reads::default();
        for (i, l) in layers.clone().enumerate() {
            let kind = &hp.layers[l];
            if !kind.routed {
                return Err(refuse(format!("layer {l} does not route")));
            }
            let downs = [
                kind.routed_types.map(|t| (t[2], Role::RoutedExperts)),
                Some((kind.shared_down, Role::SharedExpert)),
            ];
            for (ty, role) in downs.into_iter().flatten() {
                if let Err(unit) = down_k_fits(ty, role, ff) {
                    return Err(refuse(format!(
                        "layer {l}: ff {ff} is no whole number of the {ty} {role:?} down's K \
                         unit {unit:?} (`kernels::card_k_unit`)"
                    )));
                }
            }
            let [sh_gate, sh_up] = kind.shared_gate_up;
            reads = reads.with_layer(kind.routed_types, [sh_gate, sh_up, kind.shared_down], ff);
            cfg.push(LayerCfg {
                norm: names::ffn_norm(l),
                router: names::ffn_gate_inp(l),
                bias: names::exp_probs_b(l),
                sh_gate: names::ffn_gate_shexp(l),
                sh_up: names::ffn_up_shexp(l),
                sh_down: names::ffn_down_shexp(l),
                hc_fn: names::hc_ffn_fn(l),
                hc_scale: names::hc_ffn_scale(l),
                hc_base: names::hc_ffn_base(l),
                n_card: map.capacity(l)?,
                row_off: i * map.n_expert(),
                limit: kind.swiglu_limit,
                limit_shared: kind.swiglu_limit_shared,
                sh_down_q8_1: sh_down_reads_q8_1(kind.shared_down),
                sh_rows: sh_rows(sh_gate, sh_up),
                fold: hp.layers.get(l + 1).is_some_and(|k| k.engram.is_none()),
            });
        }
        let ctx = gpu.context();
        let stream = gpu.stream();
        let rows = (0..rows)
            .map(|_| FfnRow::new(stream, n_embd, ff, reads))
            .collect::<Result<Vec<_>, _>>()?;
        // The GEMM's route table holds the experts of one stack, so one table
        // a distinct card count; a block's fill passes its layer's count.
        let mut counts: Vec<usize> = cfg.iter().map(|c| c.n_card).filter(|&n| n > 0).collect();
        counts.sort_unstable();
        counts.dedup();
        let routes = counts
            .iter()
            .map(|&n| GemmRoute::new(stream, UNION_MAX_COLS * N_USED, n))
            .collect::<Result<_, _>>()?;
        Ok(FfnPiece {
            fused: FusedKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            dense: DenseKernels::load(ctx)?,
            hc: HcKernels::load(ctx)?,
            transpose: TransposeKernels::load(ctx)?,
            kernels: FfnKernels::load(ctx)?,
            layers,
            cfg,
            n_embd,
            ff,
            n_expert: ex.n_expert,
            rms_eps: hp.rms_eps,
            hc_eps: hp.hc.eps,
            hc_iters: u32::try_from(hp.hc.sinkhorn_iters)
                .map_err(|_| refuse(format!("{} Sinkhorn rounds", hp.hc.sinkhorn_iters)))?,
            scale: ex.routed_scale,
            rows,
            reads,
            fused_q5k: reads
                .fused_q5k_down
                .then(|| QuantSelKernels::load(ctx, gpu.fault_word()))
                .transpose()?,
            leg: None,
            routes,
            counts,
            hc_scratch: HcPreScratch::new(stream, HC_STREAMS * n_embd)?,
        })
    }

    /// The grouped GEMM's route table for a layer of `n_card` card experts,
    /// its index into [`FfnPiece::routes`], refused by name when the piece
    /// holds no table for the count.
    fn gemm_route_at(&self, layer: usize, n_card: usize) -> Result<usize, GpuError> {
        self.counts.iter().position(|&n| n == n_card).ok_or_else(|| GpuError::Shape {
            what: ENQUEUE,
            detail: format!(
                "layer {layer}: a GEMM route table of {n_card} card experts; the piece holds {:?}",
                self.counts
            ),
        })
    }

    /// The activation forms the piece's layers read beside the f32 rows,
    /// which a prompt batch over them holds too ([`FfnBatch::new`]).
    #[must_use]
    pub(crate) fn reads(&self) -> Reads {
        self.reads
    }

    /// The common card leg for the layers of types it runs
    /// ([`CardStacks::fused`] false): its launches and, per row, its buffers.
    /// A piece over a file whose card layers are all fused needs none, and an
    /// enqueue of a layer that does refuses by name until this has run.
    /// Load-time only.
    pub fn prepare_leg(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.leg.is_some() {
            return Ok(());
        }
        let shape = LegShape {
            cap: 1,
            slots_per_col: N_USED,
            n: self.n_embd,
            ff: self.ff,
        };
        let leg = CardLeg::load(gpu)?;
        for r in &mut self.rows {
            r.leg = Some(LegBufs::for_downs(gpu.stream(), shape, self.reads.leg)?);
        }
        self.leg = Some(leg);
        Ok(())
    }

    /// The layers the piece was built for.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Whether layer `layer` ends with HC_POST and the next fold (`true`) or
    /// HC_POST alone — into an engram layer and after the last layer.
    #[must_use]
    pub fn folds(&self, layer: usize) -> Option<bool> {
        self.cfg_of(layer).map(|c| c.fold)
    }

    /// Kernel launches layer `layer` enqueues, besides its two
    /// memory-operation batches; it enqueues no copy.
    #[must_use]
    pub fn launches(&self, layer: usize) -> Option<usize> {
        self.cfg_of(layer).map(|c| {
            let base = if c.n_card > 0 { 10 } else { 7 };
            base + usize::from(c.sh_down_q8_1) + 2 * usize::from(c.sh_rows)
        })
    }

    /// Device bytes of the piece's own scratch, every row's and the GEMM
    /// route tables'.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.rows.iter().map(FfnRow::device_bytes).sum::<usize>()
            + hc_pre_scratch_bytes(&self.hc_scratch)
            + self.routes.iter().map(GemmRoute::bytes).sum::<usize>()
    }

    /// Device bytes of one row's buffers.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.rows.first().map_or(0, FfnRow::device_bytes)
    }

    /// Rows of buffers the piece holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Row 0's buffers as the last layer enqueued on it left them.
    #[must_use]
    pub fn taps(&self) -> FfnTaps<'_> {
        self.taps_of(0).expect("every piece holds row 0")
    }

    /// Row `row`'s buffers as the last layer enqueued on it left them.
    pub fn taps_of(&self, row: usize) -> Result<FfnTaps<'_>, GpuError> {
        let r = self.row(row)?;
        Ok(FfnTaps {
            router: &r.rout,
            sel: &r.sel,
            h: &r.h,
            down: &r.down,
            shexp_h: &r.sh_h,
            shexp: &r.sh_y,
            y: &r.y,
            hc: &r.hc_out,
        })
    }

    fn row(&self, row: usize) -> Result<&FfnRow, GpuError> {
        self.rows.get(row).ok_or_else(|| GpuError::Shape {
            what: ENQUEUE,
            detail: format!("row {row} of a piece of {} rows", self.rows.len()),
        })
    }

    fn cfg_of(&self, layer: usize) -> Option<&LayerCfg> {
        self.index_of(layer).map(|i| &self.cfg[i])
    }

    /// Layer `layer`'s index in the piece, where the piece holds it.
    fn index_of(&self, layer: usize) -> Option<usize> {
        layer
            .checked_sub(self.layers.start)
            .filter(|&i| i < self.cfg.len())
    }

    /// [`FfnPiece::index_of`], or `what`'s shape error naming the piece's
    /// layers.
    fn layer_index(&self, layer: usize, what: &'static str) -> Result<usize, GpuError> {
        self.index_of(layer).ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {layer} is outside {:?}", self.layers),
        })
    }

    /// Enqueue layer `layer`'s MoE sub-layer (the module comment's order):
    /// `io` the step's shared buffers, `w` the resident weights, `card` the
    /// layer's routed stacks on the card (`None` exactly when its map row
    /// puts no expert there), `hybrid` the host tier, which is told the
    /// layer is enqueued last — an eager chain has it served then, a capture
    /// notes it for its replays. Asynchronous apart from that service,
    /// allocation-free, capturable. Row 0's buffers.
    pub fn enqueue<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        card: Option<CardStacks<'_>>,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
    ) -> Result<(), GpuError> {
        self.enqueue_shadowed(gpu, w, card, io, hybrid, layer, &mut [])
    }

    /// [`FfnPiece::enqueue`], with `extra` enqueued in order into the
    /// layer's host-leg shadow after the piece's own shadow work
    /// ([`ShadowWork`]): [`FfnPiece::enqueue_go_half`] then
    /// [`FfnPiece::enqueue_join_half`] on row 0.
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue's arguments and the shadow work; the pieces take the shared buffers flat (rust-quality R8)"
    )]
    pub fn enqueue_shadowed<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        card: Option<CardStacks<'_>>,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        extra: &mut [&mut dyn ShadowWork],
    ) -> Result<(), GpuError> {
        self.enqueue_go_half(gpu, w, card, &io, hybrid, layer, 0, extra)?;
        self.enqueue_join_half(gpu, io, hybrid, (layer, 0), None)
    }

    /// The layer's launches up to its join, on row `row`'s buffers:
    /// [`FfnPiece::enqueue_go_front`] then [`FfnPiece::enqueue_go_shadow`]. A
    /// pass whose rows run one layer apart enqueues the other row's work
    /// here, before this row's [`FfnPiece::enqueue_join_half`] of the same
    /// layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue's arguments, the row and the shadow work; the pieces take the shared buffers flat (rust-quality R8)"
    )]
    pub fn enqueue_go_half<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        card: Option<CardStacks<'_>>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        row: usize,
        extra: &mut [&mut dyn ShadowWork],
    ) -> Result<(), GpuError> {
        let go = self.enqueue_go_front(gpu, w, card, io, hybrid, layer, row)?;
        self.enqueue_go_shadow(gpu, go, io, hybrid.boundary(), extra)
    }

    /// The layer's front on row `row`'s buffers: the norm, the router, the
    /// handoff into the row's image and the go. What the shadow reads of it
    /// comes back as a [`GoFront`], for [`FfnPiece::enqueue_go_shadow`].
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue's arguments and the row; the pieces take the shared buffers flat (rust-quality R8)"
    )]
    pub fn enqueue_go_front<'w, H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &'w Weights,
        card: Option<CardStacks<'w>>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        row: usize,
    ) -> Result<GoFront<'w>, GpuError> {
        let i = self.check_io(layer, hybrid.slots(), io, row)?;
        let card = self.check_card(layer, i, card)?;
        let lw = LayerWeights::resolve(
            &self.cfg[i],
            w,
            [self.n_embd, self.ff],
            self.hc_eps,
            self.hc_iters,
        )?;
        self.enqueue_handoff(gpu, i, row, &lw, io, hybrid, layer)?;
        Ok(GoFront {
            layer,
            row,
            i,
            lw,
            card,
        })
    }

    /// The layer's shadow after its `go`: the piece's own shadow work, then
    /// `extra`, on the row the front ran on. `io` is the buffers the front
    /// took.
    pub fn enqueue_go_shadow(
        &mut self,
        gpu: &Gpu,
        go: GoFront<'_>,
        io: &FfnIo<'_>,
        boundary: &Boundary,
        extra: &mut [&mut dyn ShadowWork],
    ) -> Result<(), GpuError> {
        let GoFront {
            row, i, lw, card, ..
        } = go;
        self.enqueue_shadow(gpu, i, row, &lw, card, io, boundary)?;
        for work in extra.iter_mut() {
            work.enqueue(gpu)?;
        }
        Ok(())
    }

    /// The layer's rest on row `row`'s buffers, after its
    /// [`FfnPiece::enqueue_go_half`]: the wait, the join, and the host tier told the row's layer is enqueued.
    /// On a rank of a tensor-parallel load `sums` adds the ranks' combines before HC_POST reads
    /// them ([`FfnPiece::enqueue_join`]).
    pub fn enqueue_join_half<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        (layer, row): (usize, usize),
        sums: Option<&mut Sums>,
    ) -> Result<(), GpuError> {
        let i = self.check_io(layer, hybrid.slots(), &io, row)?;
        let boundary = hybrid.boundary();
        boundary.enqueue_back_of(gpu.stream(), row, layer)?;
        self.enqueue_join(gpu, (i, row), io, boundary, sums)?;
        hybrid.row_enqueued(layer, row)
    }

    /// Layer `layer`'s index in the piece, once the host tier's map, the
    /// shared buffers and the row are checked against it.
    fn check_io(
        &self,
        layer: usize,
        map: &SlotMap,
        io: &FfnIo<'_>,
        row: usize,
    ) -> Result<usize, GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: ENQUEUE,
            detail,
        };
        let n = self.n_embd;
        let i = self.layer_index(layer, ENQUEUE)?;
        let c = &self.cfg[i];
        if row >= self.rows.len() {
            return Err(refuse(format!(
                "row {row} of a piece of {} rows",
                self.rows.len()
            )));
        }
        let n_card = map.capacity(layer).ok();
        if map.layers() != self.layers || n_card != Some(c.n_card) {
            let held = n_card.map_or_else(
                || format!("no row for layer {layer}"),
                |n| format!("{n} experts of layer {layer} on the card"),
            );
            return Err(refuse(format!(
                "the host tier's map (layers {:?}, {held}) is not the piece's ({:?}, {})",
                map.layers(),
                self.layers,
                c.n_card
            )));
        }
        if io.slots.rows() != self.layers.len() || io.slots.cols() != self.n_expert {
            return Err(refuse(format!(
                "the map's card copy is {} x {}, want {} x {}",
                io.slots.rows(),
                io.slots.cols(),
                self.layers.len(),
                self.n_expert
            )));
        }
        if io.streams.len() < HC_STREAMS * n
            || io.streams_out.len() < HC_STREAMS * n
            || io.fold_in.len() < n
            || io.fold_out.as_ref().is_some_and(|f| f.len() < n)
            || io.fold_out.is_some() != c.fold
        {
            return Err(refuse(format!(
                "layer {layer}: streams {} and {} (want {}), fold in {} (want {n}), fold out {:?} \
                 (the layer folds: {})",
                io.streams.len(),
                io.streams_out.len(),
                HC_STREAMS * n,
                io.fold_in.len(),
                io.fold_out.as_ref().map(|f| f.len()),
                c.fold
            )));
        }
        Ok(i)
    }

    /// The card's stacks for layer `layer` (index `i`), checked against the
    /// experts its map row puts on the card; back.
    fn check_card<'s>(
        &self,
        layer: usize,
        i: usize,
        card: Option<CardStacks<'s>>,
    ) -> Result<Option<CardStacks<'s>>, GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: ENQUEUE,
            detail,
        };
        let (n, ff) = (self.n_embd, self.ff);
        match (card, self.cfg[i].n_card) {
            (None, 0) => Ok(None),
            (Some(s), k) if k > 0 && s.legs().is_some() => {
                let shape = LegShape {
                    cap: 1,
                    slots_per_col: N_USED,
                    n,
                    ff,
                };
                if let Some(legs) = s.legs() {
                    legs.check(k, shape)?;
                }
                Ok(Some(s))
            }
            (Some(s), k) if k > 0 => {
                // The fused stacks' word widths by their file types: the q3_K
                // gate and up, and the down of a type the fused launches read
                // (`accepts::DS41_FUSED_DOWNS`) by its own words a super-block.
                let d_sb = match s.down_ty() {
                    GgmlType::Q4_K => 36,
                    GgmlType::Q5_K => 44,
                    other => {
                        return Err(refuse(format!(
                            "layer {layer}: the fused layer's down is {other}, which its \
                             launches do not read (accepts::DS41_FUSED_DOWNS)"
                        )));
                    }
                };
                let (gu_words, d_words) = (110 * (n / 256) / 4, d_sb * (ff / 256));
                if s.gate.rows() != k * ff
                    || s.up.rows() != k * ff
                    || s.down.rows() != k * n
                    || s.gate.cols() != gu_words
                    || s.up.cols() != gu_words
                    || s.down.cols() != d_words
                {
                    return Err(refuse(format!(
                        "layer {layer}: {k} card experts need gate/up {} x {gu_words} and down {} x \
                         {d_words}; got {} x {}, {} x {}, {} x {}",
                        k * ff,
                        k * n,
                        s.gate.rows(),
                        s.gate.cols(),
                        s.up.rows(),
                        s.up.cols(),
                        s.down.rows(),
                        s.down.cols()
                    )));
                }
                Ok(Some(s))
            }
            (s, k) => Err(refuse(format!(
                "layer {layer}: the map puts {k} experts on the card and the stacks are {}",
                if s.is_some() { "given" } else { "absent" }
            ))),
        }
    }

    /// The norm, the router, the handoff into row `row`'s image and the go.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer, its row and weights, the shared buffers and the host tier (rust-quality R8)"
    )]
    fn enqueue_handoff<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        i: usize,
        row: usize,
        lw: &LayerWeights<'_>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let n = self.n_embd;
        let fault = gpu.layer_sink(layer)?;
        self.enqueue_route(gpu, row, lw, io, hybrid, layer)?;
        let c = &self.cfg[i];
        let r = &mut self.rows[row];
        let target = hybrid.boundary_mut().handoff_target_of(row)?;
        let lay = target.layout;
        if lay.n_used != N_USED || lay.hidden != n {
            return Err(GpuError::Shape {
                what: ENQUEUE,
                detail: format!(
                    "the boundary carries {} slots of {} values; the piece hands over {N_USED} of {n}",
                    lay.n_used, lay.hidden
                ),
            });
        }
        let h = Handoff {
            ids: &r.rout.ids,
            weights: &r.rout.weights,
            map: io.slots.buf(),
            row_off: c.row_off,
            n_expert: self.n_expert,
        };
        self.kernels
            .enqueue_handoff(stream, &h, target, fault, &mut r.sel)?;
        hybrid.boundary().enqueue_go_of(stream, layer, row)
    }

    /// The norm (its q8_1 form into row `row`'s scratch, the f32 activation
    /// into the boundary's handoff region) and the router, of layer `layer`.
    fn enqueue_route<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        row: usize,
        lw: &LayerWeights<'_>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let r = &mut self.rows[row];
        let fault = gpu.layer_sink(layer)?;
        self.fused.enqueue_norm_quant(
            stream,
            io.fold_in,
            lw.gain,
            self.rms_eps,
            &mut r.act_x,
            hybrid.boundary_mut().normed_mut(),
            fault,
        )?;
        self.router.enqueue_router(
            stream,
            lw.router,
            hybrid.boundary().normed(),
            lw.bias,
            self.scale,
            &mut r.rout,
            fault,
        )
    }

    /// The piece's own work in the shadow of the host's leg, on row `row`'s
    /// buffers: HC_PRE, the card's routed experts, the shared expert. The
    /// caller's shadow work and the wait follow it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer, its row, weights and stacks, and the shared buffers (rust-quality R8)"
    )]
    fn enqueue_shadow(
        &mut self,
        gpu: &Gpu,
        i: usize,
        row: usize,
        lw: &LayerWeights<'_>,
        card: Option<CardStacks<'_>>,
        io: &FfnIo<'_>,
        boundary: &Boundary,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let (n, ff) = (self.n_embd, self.ff);
        let layer = self.layers.start + i;
        let c = &self.cfg[i];
        let r = &mut self.rows[row];
        let pre = HcFnArgs {
            params: &lw.hc,
            x: io.streams,
            tokens: 1,
            rms_eps: self.rms_eps,
            fault: gpu.layer_sink(layer)?,
        };
        self.hc.enqueue_pre_fn(
            stream,
            &pre,
            &mut self.hc_scratch,
            &mut r.mixes,
            &mut r.hc_out,
        )?;
        match card {
            Some(s) if s.fused() => {
                let act_h = r.act_h.as_mut().ok_or(GpuError::State {
                    what: ENQUEUE,
                    missing: "the q8_1 columns of a fused layer's down, which a piece of no \
                              fused layer does not hold",
                })?;
                enqueue_fused_experts(
                    gpu,
                    &self.experts,
                    self.fused_q5k.as_ref(),
                    (layer, c.n_card, c.limit),
                    &s,
                    [n, ff],
                    &r.act_x,
                    &mut r.h,
                    act_h,
                    &mut r.sel,
                    &mut r.down,
                )?;
            }
            Some(s) => {
                let unprepared = || GpuError::State {
                    what: ENQUEUE,
                    missing: "the common card leg of a layer of types it runs \
                              (FfnPiece::prepare_leg)",
                };
                let leg = self.leg.as_ref().ok_or_else(unprepared)?;
                let bufs = r.leg.as_mut().ok_or_else(unprepared)?;
                let legs = s.legs().ok_or_else(unprepared)?;
                let run = QuantRun {
                    layer,
                    stacks: legs,
                    act: &r.act_x,
                    m: 1,
                    sel: &r.sel,
                    n_card: c.n_card,
                    rule: Act::SwigluClamp { limit: c.limit },
                };
                leg.enqueue_rows_quantized(gpu, bufs, &run, &mut r.down)?;
            }
            None => {}
        }
        match (lw.sh_rows, lw.sh_gate, lw.sh_up) {
            (Some([gate, up]), ..) => {
                let Some(gu) = r.sh_gu.as_mut() else {
                    return Err(GpuError::State {
                        what: ENQUEUE,
                        missing: "the shared gate's and up's f32 rows, which a piece of no such \
                                  layer does not hold",
                    });
                };
                enqueue_sh_rows(
                    gpu,
                    &self.dense,
                    [gate, up],
                    (boundary.normed(), c.limit_shared),
                    (gpu.layer_sink(layer)?, gu, &mut r.sh_h),
                )?;
            }
            (
                None,
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: g,
                    ..
                },
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: u,
                    ..
                },
            ) => self.dense.enqueue_shexp_gate_up_q3k(
                stream,
                g,
                u,
                &r.act_x,
                c.limit_shared,
                gpu.layer_sink(layer)?,
                &mut r.sh_h,
            )?,
            (None, gate, up) => self.experts.enqueue_shexp_gate_up(
                stream,
                gate,
                up,
                boundary.normed(),
                c.limit_shared,
                gpu.layer_sink(layer)?,
                &mut r.sh_h,
            )?,
        }
        if lw.sh_down.reads_q8_1() != c.sh_down_q8_1 {
            return Err(tensor_err(
                &c.sh_down,
                "of the format the file's header names for it",
            ));
        }
        let act = match (lw.sh_down.reads_q8_1(), r.act_sh.as_mut()) {
            (false, _) => None,
            (true, Some(act_sh)) => {
                gpu.enqueue_quantize_q8_1_layer(&r.sh_h, act_sh, layer)?;
                Some(&*act_sh)
            }
            (true, None) => {
                return Err(GpuError::State {
                    what: ENQUEUE,
                    missing: "the q8_1 form of a K-quant shared down's input, which a piece of \
                              no such layer does not hold",
                });
            }
        };
        self.dense
            .enqueue(gpu, lw.sh_down, &r.sh_h, act, &mut r.sh_y)?;
        Ok(())
    }

    /// The join on row `row`'s buffers and host sum: the combine and HC_POST
    /// in one launch, with the next fold where the layer folds. On a rank (`sums`) the combine is
    /// the rank's part of every expert's width: the launch's `y` is summed over the ranks
    /// ([`crate::chain::join::TRUNK`]) and HC_POST runs again on the sum. HC_POST writes a fold:
    /// where the layer does not fold, into the shared expert's output, which the combine has read.
    fn enqueue_join(
        &mut self,
        gpu: &Gpu,
        (i, row): (usize, usize),
        io: FfnIo<'_>,
        boundary: &Boundary,
        sums: Option<&mut Sums>,
    ) -> Result<(), GpuError> {
        let hsum = boundary.hsum_of(row)?;
        let r = &mut self.rows[row];
        let p = Post {
            down: &r.down,
            w: &r.rout.weights,
            sel: &r.sel,
            hsum,
            shexp: &r.sh_y,
            res: io.streams,
            hc: &r.hc_out,
            rows: self.n_embd,
            n_card: self.cfg[i].n_card,
            n_used: N_USED,
        };
        let Some(sums) = sums else {
            return self.kernels.enqueue_post(
                gpu.stream(),
                &p,
                &mut r.y,
                io.streams_out,
                io.fold_out,
            );
        };
        self.kernels
            .enqueue_post(gpu.stream(), &p, &mut r.y, io.streams_out, None)?;
        sums.join(gpu, TRUNK, &mut r.y)?;
        let fold = io.fold_out.unwrap_or(&mut r.sh_y);
        let post = HcPostArgs {
            x: &r.y,
            res: io.streams,
            hc: &r.hc_out,
            n_embd: self.n_embd,
            tokens: 1,
        };
        self.hc
            .enqueue_post(gpu.stream(), &post, io.streams_out, fold)
    }
}

/// The fused layer's routed launches on one row's buffers: the gate·up·SwiGLU
/// over the stacks' q3_K (`ds41_expert_gate_up`), the q8_1 of the card slots'
/// columns of `h` and the down gemv by the down stack's type
/// ([`accepts::DS41_FUSED_DOWNS`]) — a q4_K down the `q4k_sel` entry reads, a
/// q5_K one the kquant `_sel` ([`FfnPiece`]'s `fused_q5k`, held exactly where
/// a load read a q5_K down) over the same q8_1 form and selection. A down of
/// another type, or a q5_K one with no module held, is a named refusal that
/// launches nothing. Asynchronous, allocation-free, capturable.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer, its card count and clamp, the stacks and kernels, one row's buffers (rust-quality R8)"
)]
fn enqueue_fused_experts(
    gpu: &Gpu,
    experts: &ExpertKernels,
    fused_q5k: Option<&QuantSelKernels>,
    (layer, n_card, limit): (usize, usize, f32),
    stacks: &CardStacks<'_>,
    [n, ff]: [usize; 2],
    act_x: &Q8Act,
    h: &mut DeviceBuffer<f32>,
    act_h: &mut Q8Act,
    sel: &mut DeviceBuffer<u32>,
    down: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    const WHAT: &str = "enqueue_fused_experts";
    let stream = gpu.stream();
    // Resolved before the first launch: a down no fused launch reads refuses
    // the layer whole, and the q5_K module is a load's to hold.
    let q5k = match stacks.down_ty() {
        GgmlType::Q4_K => None,
        GgmlType::Q5_K => Some(fused_q5k.ok_or_else(|| GpuError::State {
            what: WHAT,
            missing: "the kquant _sel down of a fused layer with a q5_K down, which a piece \
                      of no such layer does not hold",
        })?),
        _ => {
            return Err(GpuError::Tensor {
                what: WHAT,
                name: names::ffn_down_exps(layer),
                need: "the fused layer's q4_K or q5_K down (accepts::DS41_FUSED_DOWNS)",
            });
        }
    };
    let args = ExpertGateUp {
        wg: stacks.gate,
        wu: stacks.up,
        act: act_x,
        sel: &*sel,
        n_slots: N_USED,
        rows_per_expert: ff,
        limit,
        fault: gpu.layer_sink(layer)?,
    };
    experts.enqueue_expert_gate_up(stream, &args, h)?;
    let q = QuantSel {
        x: h,
        cols: 0..N_USED,
        sel: &*sel,
        n_card,
    };
    gpu.q4k_sel()
        .enqueue_quantize_sel(stream, &q, gpu.layer_sink(layer)?, act_h)?;
    match q5k {
        None => {
            gpu.q4k_sel()
                .enqueue_gemv_q4k_sel(stream, stacks.down, act_h, sel, N_USED, n, down)
        }
        Some(q5k) => {
            let d = SelDown {
                w: stacks.down,
                act: act_h,
                sel: &*sel,
                n_slots: N_USED,
                rows_per_expert: n,
            };
            q5k.enqueue_gemv_q5k_sel(stream, &d, gpu.layer_sink(layer)?, down)
        }
    }
}

fn tensor_err(name: &str, need: &'static str) -> GpuError {
    GpuError::Tensor {
        what: ENQUEUE,
        name: name.to_string(),
        need,
    }
}

/// The router's block ticket count, which [`RouterOut`] keeps private: one
/// u32.
const ROUTER_TICKET_BYTES: usize = 4;

/// Device bytes an [`HcPreScratch`] of its K allocates (`HcPreScratch::new`'s
/// partial sums, squares and ticket count).
fn hc_pre_scratch_bytes(s: &HcPreScratch) -> usize {
    let pieces = s.k() / HC_PIECE;
    4 * HC_MIX * pieces * HC_MAX_TOKENS + 4 * pieces * HC_MAX_TOKENS + 4
}

/// Device bytes a [`Q8Act`] of its shape allocates (`Q8Act::with_k`'s five
/// buffers).
fn q8act_bytes(a: &Q8Act) -> usize {
    let (m, n_sb) = (a.m(), a.n_sb());
    8 * m * 64 * n_sb.div_ceil(2)
        + 4 * m * 256 * n_sb.div_ceil(4)
        + 4 * m * 128 * n_sb.div_ceil(2)
        + 4 * m * 8 * n_sb
        + 4 * m * 2 * n_sb
}

/// The first layer of the V4.1 host tier over the card's `layers`: the tier's
/// first routed layer (the dense lead's FFNs run on the card alone).
#[must_use]
pub fn host_first(hp: &Hparams, layers: &Range<usize>) -> usize {
    layers.start.max(hp.experts.dense_lead)
}

/// The routed experts' widths the host tier computes at ([`HostWidths`]).
#[must_use]
pub fn host_widths(hp: &Hparams) -> HostWidths {
    HostWidths {
        embd: hp.n_embd,
        ff: hp.experts.ff,
        n_used: hp.experts.n_used,
    }
}

/// The V4.1 host tier's computation over the layers `layers` of `file`, the
/// common [`HostRun`], with no NVMe tier: its run starts at the tier's first
/// routed layer ([`host_first`]) and its views are [`host::layers`]'. Load-time
/// only. A body passes its own mapping, so the pages its load populated are
/// the ones the step reads; the routed gates and ups come from `file`'s r8
/// sidecar when `r8` asks for it and there is one (the load's
/// `HostCfg::r8`). A load by a plan builds the run through the common
/// `bloomery_gpu::model::host_run` over [`host_first`], [`host_widths`] and
/// [`host::layers`], which attaches the plan's NVMe tier; this plan-less run
/// is the gates' over one file.
pub fn host_run(
    file: Arc<Split>,
    hp: &Hparams,
    layers: Range<usize>,
    r8: bool,
) -> Result<HostRun, GpuError> {
    let first = host_first(hp, &layers);
    HostRun::build(file, first, r8, host_widths(hp), |src| {
        host::layers(src, hp, first..layers.end)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shared gate and up read over the f32 row ([`enqueue_sh_rows`]),
    /// BF16 and MXFP8: `h` is the clamped SwiGLU of the gate's gemv and the
    /// up's, bit for bit, at a limit some values reach. Mutant: the gate's
    /// and the up's rows swapped into the SwiGLU.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-ds41-lib` runs it on the box"]
    fn hw_shared_rows_pair_is_two_gemvs_then_swiglu() {
        use crate::dense::testdata::f32s;
        use crate::experts::swiglu_clamp;
        use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight, e4m3_planes};
        const ROWS: usize = 24;
        const K: usize = 256;
        const LIMIT: f32 = 0.5;
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let dense = DenseKernels::load(gpu.context()).expect("dense");
        let x = DeviceBuffer::from_host(stream, &f32s(K, 7)).expect("x");
        // BF16 words of values in [-1/8, 1/8), two a word, low half first.
        let bf16 = |seed| {
            let words: Vec<u32> = f32s(ROWS * K, seed)
                .chunks(2)
                .map(|p| ((p[0] / 8.0).to_bits() >> 16) | ((p[1] / 8.0).to_bits() & 0xffff_0000))
                .collect();
            DeviceTensor::upload(stream, &words, ROWS, K / 2).expect("bf16")
        };
        // MXFP8 blocks at scale 2^-3, each value byte's top exponent bit clear (off E4M3's NaN).
        let mxfp8 = |seed| {
            let bytes: Vec<u8> = f32s(ROWS * K / 32 * 33, seed)
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if i % 33 == 0 {
                        124
                    } else {
                        ((v + 1.0) * 127.5) as u8 & 0xBF
                    }
                })
                .collect();
            let (qs, e) = e4m3_planes(E4m3Scales::Mxfp8, &bytes, ROWS, K).expect("planes");
            (
                DeviceTensor::upload(stream, &qs, ROWS, K / 4).expect("qs"),
                DeviceTensor::upload(stream, &e, ROWS, K.div_ceil(128)).expect("e"),
            )
        };
        let (bg, bu, mg, mu) = (bf16(1), bf16(2), mxfp8(3), mxfp8(4));
        fn mx((qs, e): &(DeviceTensor<u32>, DeviceTensor<u32>)) -> Dense<'_> {
            Dense::Mxfp8(E4m3Weight {
                qs,
                e,
                scales: E4m3Scales::Mxfp8,
            })
        }
        for (what, gate, up) in [
            ("bf16", Dense::Bf16(&bg), Dense::Bf16(&bu)),
            ("mxfp8", mx(&mg), mx(&mu)),
        ] {
            let gemv = |w| {
                let mut y = DeviceBuffer::zeroed(stream, ROWS).expect("y");
                dense.enqueue(&gpu, w, &x, None, &mut y).expect("gemv");
                y.to_host_vec(stream).expect("rows")
            };
            let want: Vec<u32> = gemv(gate)
                .iter()
                .zip(&gemv(up))
                .map(|(&g, &u)| swiglu_clamp(g, u, LIMIT).to_bits())
                .collect();
            let zeros = || DeviceBuffer::zeroed(stream, ROWS).expect("rows");
            let (mut gu, mut h) = ([zeros(), zeros()], zeros());
            let sink = gpu.unlabelled_sink();
            enqueue_sh_rows(
                &gpu,
                &dense,
                [gate, up],
                (&x, LIMIT),
                (sink, &mut gu, &mut h),
            )
            .expect(what);
            let got: Vec<u32> = h
                .to_host_vec(stream)
                .expect("h")
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(gpu.take_fault().expect("fault"), None, "{what}");
            assert_eq!(got, want, "{what}");
        }
    }

    /// A rank's part of the 2,304-wide experts (1,152 at two ranks, 576 at
    /// four) opens where its downs read it: an MXFP4 routed down over the
    /// 32-value form and an MXFP8 shared down; a down of 256-value
    /// super-blocks, routed or shared, is refused at those widths and runs at
    /// 2,304.
    #[test]
    fn a_ranks_ff_opens_where_its_downs_read_it() {
        for ff in [576, 1152, 2304] {
            assert_eq!(
                down_k_fits(GgmlType::MXFP4, Role::RoutedExperts, ff),
                Ok(())
            );
            assert_eq!(down_k_fits(GgmlType::MXFP8, Role::SharedExpert, ff), Ok(()));
        }
        for (ty, role) in [
            (GgmlType::Q4_K, Role::RoutedExperts),
            (GgmlType::Q3_K, Role::SharedExpert),
            (GgmlType::Q5_K, Role::SharedExpert),
        ] {
            assert_eq!(down_k_fits(ty, role, 576), Err(Some(256)), "{ty}");
            assert_eq!(down_k_fits(ty, role, 2304), Ok(()), "{ty}");
        }
    }

    /// Synthetic file bytes of a K-quant stack: `rows` rows of `k` values of
    /// `ty`, a wandering byte pattern whose every block scale is a small
    /// normal f16 (a Q3_K block's last two bytes: `d` 0.0625; another
    /// K-quant's first four: `d` 0.5, `dmin` 0.25), as the file holds them.
    /// Shared with the child modules' tests.
    pub(super) fn kquant_bytes(ty: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let (blck, size) = (
            ty.blck_size().expect("a block size") as usize,
            ty.type_size().expect("a type size") as usize,
        );
        let mut out = vec![0u8; rows * (k / blck) * size];
        let mut state = seed | 1;
        for (i, b) in out.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = (state >> 48) as u8;
            // The block's f16 scales: a Q3_K block's one `d` is its last two
            // bytes, every other K-quant's `d` and `dmin` its first four.
            let at = i % size;
            if ty == GgmlType::Q3_K && at >= size - 2 {
                *b = if at == size - 1 { 0x2c } else { 0x00 };
            } else if ty != GgmlType::Q3_K && at < 4 {
                *b = match at {
                    1 => 0x38,
                    3 => 0x34,
                    _ => 0x00,
                };
            }
        }
        out
    }

    /// A stack's bytes as its words, whole rows whole words (the K-quant
    /// families' rows are a whole number of words).
    pub(super) fn stack_words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// The fused layer's decode down by the down stack's type
    /// ([`enqueue_fused_experts`]): over the same gate·up's `h`, its q8_1 and
    /// selection, a q5_K down's rows are the kquant `_sel` entry's own
    /// launch's bit for bit and a q4_K down's the `q4k_sel` entry's, so the
    /// dispatch reaches each entry with the step's own arguments. A down of
    /// another type, or a q5_K one where the load holds no module, is a named
    /// refusal that launches nothing. Mutant: the q5_K arm dispatching to the
    /// q4_K decoder (`.mutants/fused-q5k-decode.patch`).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-ds41-lib` runs it on the box"]
    fn hw_the_fused_decode_down_dispatches_by_its_type() {
        let (n, ff, experts, layer) = (512usize, 512usize, 4usize, 3usize);
        let gpu = Gpu::new().expect("a card");
        let stream = gpu.stream();
        let stack = |ty: GgmlType, rows: usize, k: usize, seed: u64, row_k: usize| {
            let rb = ty.type_size().expect("a size") as usize * (row_k / 256);
            DeviceTensor::upload(
                stream,
                &stack_words(&kquant_bytes(ty, rows, k, seed)),
                rows,
                rb / 4,
            )
            .expect("the stack")
        };
        let gate = stack(GgmlType::Q3_K, experts * ff, n, 0x11, n);
        let up = stack(GgmlType::Q3_K, experts * ff, n, 0x22, n);
        let down_of = |ty| stack(ty, experts * n, ff, 0x33 + ty.as_u32() as u64, ff);
        let downs = [
            (GgmlType::Q4_K, down_of(GgmlType::Q4_K)),
            (GgmlType::Q5_K, down_of(GgmlType::Q5_K)),
            (GgmlType::Q6_K, down_of(GgmlType::Q6_K)),
        ];
        let expert_kernels = ExpertKernels::load(gpu.context()).expect("the expert module");
        let q5k = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant one");
        let sel: Vec<u32> = [0, 1, 2, 3, 0, 1].to_vec();
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let mut act_x = Q8Act::with_k(stream, 1, n).expect("the x form");
        let x_dev = DeviceBuffer::from_host(stream, &x).expect("x");
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act_x)
            .expect("the quantize");
        let read = |b: &DeviceBuffer<f32>| b.to_host_vec(stream).unwrap();
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let nan = vec![f32::NAN; N_USED * n];
        // The dispatch into poisoned buffers: `h` poisoned too, so a launch
        // that ran shows in it and a refusal that launched nothing leaves
        // both as they were.
        let run = |stacks: &CardStacks<'_>, module: Option<&QuantSelKernels>| {
            let mut h = DeviceBuffer::from_host(stream, &vec![f32::NAN; N_USED * ff]).unwrap();
            let mut act_h = Q8Act::with_k(stream, N_USED, ff).expect("the h form");
            let mut sel_dev = DeviceBuffer::from_host(stream, &sel).unwrap();
            let mut down = DeviceBuffer::from_host(stream, &nan).unwrap();
            let done = enqueue_fused_experts(
                &gpu,
                &expert_kernels,
                module,
                (layer, experts, 7.0),
                stacks,
                [n, ff],
                &act_x,
                &mut h,
                &mut act_h,
                &mut sel_dev,
                &mut down,
            );
            (done, h, down)
        };
        // The same `h` quantized again, the down launched by its entry alone.
        let direct = |h: &DeviceBuffer<f32>, ty: GgmlType, w: &DeviceTensor<u32>| {
            let mut act_h = Q8Act::with_k(stream, N_USED, ff).expect("the h form");
            let sel_dev = DeviceBuffer::from_host(stream, &sel).unwrap();
            let fault = gpu.layer_sink(layer).unwrap();
            let q = QuantSel {
                x: h,
                cols: 0..N_USED,
                sel: &sel_dev,
                n_card: experts,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, fault, &mut act_h)
                .unwrap();
            let mut out = DeviceBuffer::from_host(stream, &nan).unwrap();
            match ty {
                GgmlType::Q4_K => gpu
                    .q4k_sel()
                    .enqueue_gemv_q4k_sel(stream, w, &act_h, &sel_dev, N_USED, n, &mut out)
                    .unwrap(),
                _ => {
                    let d = SelDown {
                        w,
                        act: &act_h,
                        sel: &sel_dev,
                        n_slots: N_USED,
                        rows_per_expert: n,
                    };
                    q5k.enqueue_gemv_q5k_sel(stream, &d, fault, &mut out)
                        .unwrap();
                }
            }
            out
        };
        let (q4, q5, q6) = (&downs[0], &downs[1], &downs[2]);
        for (ty, w) in [(q4.0, &q4.1), (q5.0, &q5.1)] {
            let stacks = CardStacks::fused_over(&gate, &up, w, ty);
            let (done, h, down) = run(&stacks, Some(&q5k));
            done.unwrap_or_else(|e| panic!("the {ty} dispatch: {e}"));
            assert!(
                read(&h).iter().all(|v| v.is_finite()),
                "the gate·up wrote the {ty} run's h"
            );
            assert!(
                same(&read(&down), &read(&direct(&h, ty, w))),
                "the {ty} down's rows are its entry's own launch's"
            );
            assert!(
                gpu.take_fault().unwrap().is_none(),
                "the {ty} dispatch raises no fault"
            );
        }
        // A down of another type, and a q5_K one with no module held, are
        // named refusals that launch nothing: both buffers keep their poison.
        for (ty, w, module, names_it) in
            [(q6.0, &q6.1, Some(&q5k), true), (q5.0, &q5.1, None, false)]
        {
            let stacks = CardStacks::fused_over(&gate, &up, w, ty);
            let (done, h, down) = run(&stacks, module);
            let err = done.expect_err("the refusal");
            assert!(
                if names_it {
                    err.to_string().contains(&names::ffn_down_exps(layer))
                } else {
                    err.to_string().contains("q5_K")
                },
                "the refusal names its cause: {err}"
            );
            assert!(same(&read(&h), &vec![f32::NAN; N_USED * ff]), "h untouched");
            assert!(same(&read(&down), &nan), "no down row written");
        }
    }
}
