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
//!   the down. A slot the host serves leaves its column of `h` as it was —
//!   a reset does not clear it, so after a fault it can hold a NaN — and no
//!   card launch reads that column.
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
//!   [`Slots::of`] for the join) and refuse any other by name.
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

use bloomery_gpu::fused::FusedKernels;
pub use bloomery_gpu::host::handoff::Handoff;
use bloomery_gpu::host::handoff::HandoffKernels;
use bloomery_gpu::hybrid::{Boundary, HOST, HandoffTarget, HostExperts, Hybrid, SlotMap};
use bloomery_gpu::q4k_sel::QuantSel;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, FaultSink, Gpu, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::{host, names};
use model::moe::{HostLayer, HostScratch, UNION_MAX_COLS, UnionScratch};
use model::r8file::R8Pair;
use model::{Tensor2, Tensor2View};
use runtime::combine::Slots;

use crate::dense::{Dense, DenseKernels};
use crate::experts::{ExpertGateUp, ExpertKernels};
use crate::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcParams, HcPreArgs, HcPreScratch,
    hc_fold_elem, hc_post_elem,
};
use crate::router::{N_EXPERT, N_USED, RouterKernels, RouterOut};
use crate::transpose::TransposeKernels;

mod batch;
mod tier;
pub use batch::{
    BatchLayer, BlockIo, CardAcc, CardGather, ChunkIo, FfnBatch, FfnBatchKernels, JoinIo, Places,
    TiledGateUp,
};
pub use tier::{Ds41Tier, TierPiece};

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

/// Slots a token of the `_8` entries: the join's second instance
/// ([`Slots::Eight`]), whose contracts spell it as a literal.
const SLOTS_8: usize = 8;
const _: () = assert!(SLOTS_8 == 8);
// The instance table's counts are the entries' own.
const _: () = assert!(Slots::Six.n() == N_USED && Slots::Eight.n() == SLOTS_8);

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

/// The combine's card half: `acc = fma(down_j, w_j, acc)` from zero for each
/// slot `j` whose place is on the card, in slot order.
#[inline(always)]
pub fn card_sum_elem(down: [f32; N_USED], w: [f32; N_USED], card: [bool; N_USED]) -> f32 {
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < N_USED {
        if card[j] {
            acc = down[j].mul_add(w[j], acc);
        }
        j += 1;
    }
    acc
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
/// there, slot `s` at rows `s · rows .. (s + 1) · rows` of each — gate and up
/// q3_K (`ff` rows of `n_embd` per expert), down q4_K (`n_embd` rows of `ff`).
#[derive(Clone, Copy)]
pub struct CardStacks<'a> {
    pub gate: &'a DeviceTensor<u32>,
    pub up: &'a DeviceTensor<u32>,
    pub down: &'a DeviceTensor<u32>,
}

impl<'a> CardStacks<'a> {
    /// The stacks a placed load holds for layer `layer`: its routed gate, up
    /// and down file tensors in `w`, which a plan uploads with the experts of
    /// its card segment in slot order. `None` when `w` holds none of the
    /// three; an error when it holds some, or holds one in another format.
    pub fn of(w: &'a Weights, layer: usize) -> Result<Option<CardStacks<'a>>, GpuError> {
        const WHAT: &str = "CardStacks::of";
        let names = [
            names::ffn_gate_exps(layer),
            names::ffn_up_exps(layer),
            names::ffn_down_exps(layer),
        ];
        let found = names.each_ref().map(|n| w.get(n));
        if found.iter().all(Option::is_none) {
            return Ok(None);
        }
        let [gate, up, down] = [
            (0, GgmlType::Q3_K),
            (1, GgmlType::Q3_K),
            (2, GgmlType::Q4_K),
        ]
        .map(|(i, want)| match found[i] {
            Some(DevWeight::KQuant { ty, w: stack, .. }) if *ty == want => Ok(stack),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: names[i].clone(),
                need: "a resident K-quant routed stack beside the layer's other two",
            }),
        });
        Ok(Some(CardStacks {
            gate: gate?,
            up: up?,
            down: down?,
        }))
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
    sh_down: Dense<'w>,
    hc: HcParams<'w>,
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
        let gain = f32_weight(w, &c.norm)?;
        let router = f32_tensor(w, &c.router)?;
        let bias = f32_weight(w, &c.bias)?;
        let (sh_gate, sh_up) = (weight(w, &c.sh_gate)?, weight(w, &c.sh_up)?);
        let sh_down = Dense::of(w, &c.sh_down, ff, n_embd)?;
        let DevWeight::KQuant {
            ty: GgmlType::Q3_K,
            w: hc_w,
            ..
        } = weight(w, &c.hc_fn)?
        else {
            return Err(tensor_err(&c.hc_fn, "a q3_K file tensor"));
        };
        Ok(LayerWeights {
            gain,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
            hc: HcParams {
                w: hc_w,
                scale: f32_weight(w, &c.hc_scale)?,
                base: f32_weight(w, &c.hc_base)?,
                eps: hc_eps,
                iters: hc_iters,
            },
        })
    }
}

/// The piece's own kernels, the handoff and the join: loaded once by
/// [`FfnPiece::new`], or alone by a gate that drives the handoff with ids a
/// router cannot produce.
pub struct FfnKernels {
    module: ffn_kernels::LoadedModule,
    handoff: HandoffKernels,
}

impl FfnKernels {
    /// Load the piece's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FfnKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { ffn_kernels::load(ctx)? };
        Ok(FfnKernels {
            module,
            handoff: HandoffKernels::load(ctx)?,
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

    /// Enqueue the combine and HC_POST of `p.n_used` slots a token
    /// (`ds41_ffn_post`/`ds41_ffn_post_streams` for 6, their `_8` entries for
    /// 8; any other count is refused by name), one thread per value `d <
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
/// keeps its own.
struct FfnRow {
    act_x: Q8Act,
    rout: RouterOut,
    sel: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    down: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    /// `sh_h` in q8_1, for a K-quant shared down projection.
    act_sh: Q8Act,
    sh_y: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    hc_out: DeviceBuffer<f32>,
}

impl FfnRow {
    fn new(stream: &CudaStream, n_embd: usize, ff: usize) -> Result<FfnRow, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(FfnRow {
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            rout: RouterOut::new(stream)?,
            sel: DeviceBuffer::zeroed(stream, N_USED)?,
            h: z(N_USED * ff)?,
            act_h: Q8Act::with_k(stream, N_USED, ff)?,
            down: z(N_USED * n_embd)?,
            sh_h: z(ff)?,
            act_sh: Q8Act::with_k(stream, 1, ff)?,
            sh_y: z(n_embd)?,
            y: z(n_embd)?,
            mixes: z(HC_MIX)?,
            hc_out: z(HC_MIX)?,
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
            + q8act_bytes(&self.act_h)
            + q8act_bytes(&self.act_sh)
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
        if hp.hc.streams != HC_STREAMS || !ff.is_multiple_of(256) {
            return Err(refuse(format!(
                "{} streams and ff {ff}: want {HC_STREAMS} streams and whole super-blocks",
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
        for (i, l) in layers.clone().enumerate() {
            let kind = &hp.layers[l];
            if !kind.routed {
                return Err(refuse(format!("layer {l} does not route")));
            }
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
                sh_down_q8_1: matches!(kind.shared_down, GgmlType::Q3_K | GgmlType::Q4_K),
                fold: hp.layers.get(l + 1).is_some_and(|k| k.engram.is_none()),
            });
        }
        let ctx = gpu.context();
        let stream = gpu.stream();
        let rows = (0..rows)
            .map(|_| FfnRow::new(stream, n_embd, ff))
            .collect::<Result<Vec<_>, _>>()?;
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
            hc_scratch: HcPreScratch::new(stream, HC_STREAMS * n_embd)?,
        })
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
        self.cfg_of(layer)
            .map(|c| if c.n_card > 0 { 10 } else { 7 } + usize::from(c.sh_down_q8_1))
    }

    /// Device bytes of the piece's own scratch, every row's.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.rows.iter().map(FfnRow::device_bytes).sum::<usize>()
            + hc_pre_scratch_bytes(&self.hc_scratch)
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
        self.enqueue_join_half(gpu, io, hybrid, layer, 0)
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
    pub fn enqueue_join_half<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let i = self.check_io(layer, hybrid.slots(), &io, row)?;
        let boundary = hybrid.boundary();
        boundary.enqueue_back_of(gpu.stream(), row)?;
        self.enqueue_join(gpu, i, row, io, boundary)?;
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
            (Some(s), k) if k > 0 => {
                let (gu_words, d_words) = (110 * (n / 256) / 4, 36 * (ff / 256));
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
        let pre = HcPreArgs {
            params: &lw.hc,
            x: io.streams,
            tokens: 1,
            rms_eps: self.rms_eps,
            fault: gpu.layer_sink(layer)?,
        };
        self.hc.enqueue_pre(
            stream,
            &pre,
            &mut self.hc_scratch,
            &mut r.mixes,
            &mut r.hc_out,
        )?;
        if let Some(s) = card {
            let args = ExpertGateUp {
                wg: s.gate,
                wu: s.up,
                act: &r.act_x,
                sel: &r.sel,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit: c.limit,
            };
            self.experts
                .enqueue_expert_gate_up(stream, &args, &mut r.h)?;
            let q = QuantSel {
                x: &r.h,
                cols: 0..N_USED,
                sel: &r.sel,
                n_card: c.n_card,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, gpu.layer_sink(layer)?, &mut r.act_h)?;
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                s.down,
                &r.act_h,
                &r.sel,
                N_USED,
                n,
                &mut r.down,
            )?;
        }
        match (lw.sh_gate, lw.sh_up) {
            (
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
                &mut r.sh_h,
            )?,
            (gate, up) => self.experts.enqueue_shexp_gate_up(
                stream,
                gate,
                up,
                boundary.normed(),
                c.limit_shared,
                &mut r.sh_h,
            )?,
        }
        if lw.sh_down.reads_q8_1() != c.sh_down_q8_1 {
            return Err(tensor_err(
                &c.sh_down,
                "of the format the file's header names for it",
            ));
        }
        let act = if lw.sh_down.reads_q8_1() {
            gpu.enqueue_quantize_q8_1_layer(&r.sh_h, &mut r.act_sh, layer)?;
            Some(&r.act_sh)
        } else {
            None
        };
        self.dense
            .enqueue(gpu, lw.sh_down, &r.sh_h, act, &mut r.sh_y)?;
        Ok(())
    }

    /// The join on row `row`'s buffers and host sum: the combine and HC_POST
    /// in one launch, with the next fold where the layer folds.
    fn enqueue_join(
        &mut self,
        gpu: &Gpu,
        i: usize,
        row: usize,
        io: FfnIo<'_>,
        boundary: &Boundary,
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
        self.kernels
            .enqueue_post(gpu.stream(), &p, &mut r.y, io.streams_out, io.fold_out)
    }
}

/// The resident weight `name`.
fn weight<'w>(w: &'w Weights, name: &str) -> Result<&'w DevWeight, GpuError> {
    w.get(name)
        .ok_or_else(|| tensor_err(name, "a resident weight"))
}

/// The resident f32 tensor `name` (an f32 file tensor, or bf16 decoded at
/// load).
fn f32_tensor<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<f32>, GpuError> {
    match weight(w, name)? {
        DevWeight::F32 { w, .. } => Ok(w),
        _ => Err(tensor_err(name, "an f32 tensor")),
    }
}

/// The resident f32 vector `name`, as the buffer a kernel reads.
fn f32_weight<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    f32_tensor(w, name).map(DeviceTensor::buf)
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

/// The V4.1 host tier's computation ([`HostExperts`]): each layer's routed
/// stacks as [`HostLayer`] reads them from the file, and the scratch every
/// call writes, made at load — the union's for a prompt batch at its first
/// call.
pub struct Ds41Host {
    /// The file and the r8 reading its layers were built from, checked
    /// against each other at load; every call reads through this pair.
    file: R8Pair,
    /// Per layer of `layers`, its host view; `None` for a layer that does
    /// not route.
    layers: Vec<Option<HostLayer>>,
    first: usize,
    scratch: HostScratch,
    /// The union's blocks for [`UNION_MAX_COLS`] columns, made by the first
    /// batch service: a decode that never prefills a batch never holds them.
    union: Option<UnionScratch>,
    embd: usize,
    ff: usize,
    /// Routed slots a token: the union's list width.
    n_used: usize,
}

impl Ds41Host {
    /// The tier for layers `layers` of `file`, whose hyperparameters are
    /// `hp`. Load-time only: every stack is found and checked here. A body
    /// passes its own mapping, so the pages its load populated are the ones
    /// the step reads; the routed gates and ups come from `file`'s r8 sidecar
    /// when `r8` asks for it and there is one ([`R8Pair::at_load`] under the
    /// load's `HostCfg::r8`: the reading and the mapping the load's host set
    /// took).
    pub fn build(
        file: impl Into<Arc<Split>>,
        hp: &Hparams,
        layers: Range<usize>,
        r8: bool,
    ) -> Result<Ds41Host, GpuError> {
        let file = R8Pair::at_load(file.into(), r8)?;
        let first = layers.start;
        let views = layers
            .map(|l| host::layer(file.source(), hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Ds41Host {
            file,
            layers: views,
            first,
            scratch: HostScratch::new(hp.n_embd, hp.experts.ff, hp.experts.n_used)?,
            union: None,
            embd: hp.n_embd,
            ff: hp.experts.ff,
            n_used: hp.experts.n_used,
        })
    }
}

/// Layer `layer`'s view in `layers`, the tier's views from layer `first` on;
/// refused for a layer outside them or one that does not route.
fn host_view<'a>(
    layers: &'a [Option<HostLayer>],
    first: usize,
    layer: usize,
    what: &'static str,
) -> Result<&'a HostLayer, GpuError> {
    layer
        .checked_sub(first)
        .and_then(|i| layers.get(i))
        .and_then(Option::as_ref)
        .ok_or(GpuError::State {
            what,
            missing: "the layer's routed stacks: it is outside the tier or does not route",
        })
}

impl Ds41Host {
    /// The union's scratch for [`UNION_MAX_COLS`] columns, made now if no
    /// batch service has made it: a prompt timed after this allocates
    /// nothing.
    pub fn prepare_union(&mut self) -> Result<(), GpuError> {
        if self.union.is_none() {
            self.union = Some(union_scratch(self.embd, self.ff, self.n_used)?);
        }
        Ok(())
    }
}

/// The one shape of the tier's union scratch, for [`Ds41Host::prepare_union`]
/// and the lazy path alike: [`UNION_MAX_COLS`] columns of the model's `n_used`
/// routed slots; a longer list is refused by name.
fn union_scratch(embd: usize, ff: usize, n_used: usize) -> Result<UnionScratch, GpuError> {
    Ok(UnionScratch::new_routed(embd, ff, UNION_MAX_COLS, n_used)?)
}

impl HostExperts for Ds41Host {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let view = host_view(&self.layers, self.first, layer, "Ds41Host::experts_into")?;
        view.experts_into(self.file.source(), x, experts, out, &mut self.scratch)?;
        Ok(())
    }

    fn experts_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let Ds41Host {
            file,
            layers,
            first,
            union,
            embd,
            ff,
            n_used,
            ..
        } = self;
        let view = host_view(layers, *first, layer, "Ds41Host::experts_union_into")?;
        let scratch = match union {
            Some(s) => s,
            None => union.insert(union_scratch(*embd, *ff, *n_used)?),
        };
        view.experts_union_into(file.source(), x, lists, out, scratch)?;
        Ok(())
    }
}
