//! Qwen3.8's gated-residual hyper-connections on the card: the mix around a
//! sub-layer (the streams' grouped norm, the rank bottleneck's down and up
//! gemvs, the sigmoid gate and the mean over the streams), the combine that
//! adds a sub-layer's output back into every stream by its own weight, and
//! the head's mix. The rule, its layouts and the one table of shapes these
//! kernels are built for are `runtime::hc_gated`'s; its `card` module is the
//! f32 transcription of every kernel below, stage for stage.
//!
//! Three launches a mix, `m` columns (1..=8) in each:
//! - [`hc_kernels::hc_gated_norm_4`] — one block a (stream, column): first
//!   the combine of the previous site, in place ([`Before`]), then the
//!   stream's sum of squares, then `xn = (x·r)·γ` into the scratch;
//! - [`hc_kernels::hc_gated_down_4x320`] — the down's partial dots per
//!   stream, `[m][rank][streams]`: a block takes eight rows of one stream,
//!   each warp one row through the q8f32 Q8_0 core
//!   ([`crate::q8f32::q8_0_lane_partials_mcol`]) on the `[rank·streams ×
//!   hidden]` view of the weight, so the eight rows share the stream's
//!   activations in L1; at an inject site, sixteen more blocks take the F32
//!   inject rows, a block one (row, stream) in eight chunks through
//!   [`crate::q8f32::f32_lane_partial_1col`];
//! - [`hc_kernels::hc_gated_up_mix_4x320`] — every block first adds each
//!   down row's partials in stream order and takes `silu(·/4)` into shared
//!   memory (block 0 also writes them out, and at an inject site the
//!   combine's weights `2·σ(·/4)`), then a warp per hidden value runs the
//!   four up rows of that value against the bottleneck in the q8f32 lane
//!   order, and lane c writes column c's `(xn_0·σ(g_0), fma for the
//!   rest)·(1/4)`.
//!
//! A BF16 down or up (in its words, [`HcMatrix::Bf16`]) runs its own entry
//! of the same blocks, `hc_gated_down_bf16_4x320` and
//! `hc_gated_up_mix_bf16_4x320`: each row through the F32 core's lane walk
//! over the widened values (lane `L` values `32i + L` in increasing `i`, one
//! fused multiply-add each, then the butterfly), the down's partials and the
//! up's mix as above. Down and up pick their entry each by its own type.
//!
//! [`hc_kernels::hc_gated_combine_4`] is the combine alone, for a site whose
//! next reader is not a mix (the PLE site's layer).
//!
//! Numeric contract (reruns are bit-identical; column c of an `m`-column
//! launch is its `m = 1` launch bit for bit): a norm thread takes values `t,
//! t + 256, …` of its stream by fused multiply-adds, each warp's by the xor
//! butterfly, the eight warps' in order from warp 0; the Q8_0 rows are the
//! q8f32 gemv's lane walk and butterfly; an inject chunk is the F32 core's
//! lane walk and butterfly, a block's eight in order; the streams' partials in
//! stream order; `silu` and `sigmoid` are [`crate::linear`]'s. The combine is
//! one fused multiply-add a value.
//!
//! No silent failure: a stream whose sum of squares is not finite, a partial
//! dot, a bottleneck value, a combine weight or a mixed value that is not
//! finite raises [`SITE`]; every value is written as computed.
//!
//! The wide arm ([`HcWideKernels`], a device module of its own) runs a mix
//! over a whole ubatch of `m` columns, any count: the norm is the entry
//! above with the eight-column cap lifted (`hc_gated_norm_4w`, the same
//! body), over streams held as F32 values or, as a ubatch walk carries
//! them, as BF16 values ([`Streams`]; `hc_gated_norm_4w16`), whose combine
//! or init rounds each value it writes to nearest even before the norm
//! reads it; the combine alone (`hc_gated_combine_4w`) reads BF16 streams
//! and writes F32 ones, the walk's PLE input and last streams; the down
//! and the up are the 32-value GEMM (`crate::gemm`'s `gemm_q8_0p` over a
//! one-expert table) on `xn` and on the bottleneck quantized per 32 values;
//! the inject rows are the wide F32 tile, and a BF16 down or up the BF16
//! prompt product (`Q8F32Kernels::enqueue_bf16_prompt`) on `xn` or the
//! bottleneck, with BF16 operands on a compute capability 12 device and F32 activations otherwise.
//! On a compute capability 12 device a BF16 down's site keeps `xn` as BF16 values: the norm rounds
//! each to nearest even as it writes it (`hc_gated_norm_4w_bf16`,
//! `hc_gated_norm_4w16_bf16`), and the
//! down, its inject rows and the mix read those values. The down and an
//! inject site's inject rows are one product (`hc_gated_down_mma_4`,
//! [`HcDownRows`]): the inject rows below the down's, rounded to BF16 as
//! they are staged, so `xn` is read once;
//! two glue entries finish the rule —
//! `hc_gated_lo_4x320` (`silu(·/4)` of each down row, `2·σ(·/4)` of each
//! inject row) and `hc_gated_mix_4` (`(xn_0·σ(g_0), fma for the rest)·¼`,
//! the up's order; `hc_gated_mix_4_bf16` over BF16 `xn`). A column's values
//! are a function of that column alone.
//! The wide products use a different sum order from the narrow arm, with
//! Q8_0 activations for Q8_0 weights and BF16 operands for BF16 weights on a
//! compute capability 12 device. The gate checks their error bands, not bit equality between arms.
//! The same values raise [`SITE`].

use crate::cores::l2_prefetch;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::half_bits_to_f32;
use crate::gemm::{
    Bf16Cols, Bf16Out, Bf16Rows, Bf16WordCols, F32Cols, F32Rows, Gemm32Args, Gemm32Kernels,
    Gemm32Weight, GemmAct32, GemmInput, GemmRoute, HcUpMixArgs, MMA_COLS, MMA_ROWS, MMA_THREADS,
    MMA_WARP_COLS, MMA_WARP_ROWS, MMA_WARP_SCRATCH, MmaCols, MmaForm, MmaRows, MmaTile, OutForm,
    STAGE_FLOATS, STAGES, TileRows, bf16_mma_body, bf16_mma_launch, bf16_mma_tile, form_col,
    land_values, mma_launch_words, mma_tile_origin, store_refused,
};
use crate::linear::{sigmoid, silu};
use crate::q8f32::{
    Bf16Words, Q8F32Kernels, RowValues, f32_lane_partial_1col, f32_lane_partials_view,
    gemv_lane_sums, q8_0_lane_partials_mcol,
};
use crate::tensor::{DeviceTensor, Window, WindowMut, check_alignment};
use crate::view::StreamFma;
use crate::weights::{DevWeight, Weights};
use crate::{GpuError, launch_u32, store_cols};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::vector::{F32x4, U32x4};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use gguf::quant::GgmlType;
use runtime::hc_gated::card::{DOWN_WARPS, NORM_THREADS, NORM_WARPS};
use runtime::hc_gated::{Geometry, HcRefused, INSTANCES, MAX_COLS, Sizes};
use std::sync::Arc;

/// Streams of the instance these kernels are built for.
pub const STREAMS: usize = 4;
/// The bottleneck's width of that instance.
pub const RANK: usize = 320;
/// Values a norm thread loads at once: the stream is walked in chunks of
/// this many values a thread.
const NORM_CHUNK: usize = 16;
const _: () = assert!(
    INSTANCES.len() == 1
        && INSTANCES[0].streams as usize == STREAMS
        && INSTANCES[0].rank as usize == RANK
);
const _: () = assert!(MAX_COLS == 8 && NORM_THREADS == 256 && NORM_WARPS == 8);
// Lane c of an up warp mixes column c.
const _: () = assert!(MAX_COLS <= 32);
const _: () = assert!(DOWN_WARPS == 8);

/// The fault site every entry here raises.
pub const SITE: FaultSite = FaultSite::HcMix;

/// `1/streams`: the scale inside the silu and the sigmoid, and the mean's.
const INV: f32 = 1.0 / STREAMS as f32;

#[inline(always)]
fn gate_value(g: f32) -> f32 {
    if g.is_finite() { sigmoid(g) } else { f32::NAN }
}

#[inline(always)]
fn mix_step(acc: f32, x: f32, gate: f32, stream: usize) -> f32 {
    if stream == 0 {
        x * gate
    } else {
        x.mul_add(gate, acc)
    }
}

/// # Safety
/// i is this thread's unique live output index inside mixed.
#[inline(always)]
unsafe fn store_mix(acc: f32, i: usize, fault: FaultSink, mixed: &mut DisjointSlice<f32>) {
    let out = acc * INV;
    if !out.is_finite() {
        fault.raise(SITE);
    }
    // SAFETY: the caller owns this live output index.
    unsafe { *mixed.get_unchecked_mut(i) = out };
}
/// Threads of a down block: [`DOWN_WARPS`] warps.
const DOWN_THREADS: u32 = 256;
/// Down blocks of Q8_0 rows: `RANK / DOWN_WARPS` row tiles × the streams.
const DOWN_Q8_BLOCKS: usize = RANK / DOWN_WARPS * STREAMS;
/// Down blocks of F32 inject rows: a (row, stream) each.
const DOWN_INJ_BLOCKS: usize = STREAMS * STREAMS;
const _: () = assert!(RANK.is_multiple_of(DOWN_WARPS) && DOWN_THREADS as usize == DOWN_WARPS * 32);
const _: () = assert!(DOWN_Q8_BLOCKS == 160 && DOWN_INJ_BLOCKS == 16);

/// Threads of an up block: a warp per hidden value.
const UP_THREADS: u32 = 512;
const UP_WARPS: usize = UP_THREADS as usize / 32;
/// Code words of an up row, and the most one lane takes.
const UP_WORDS: usize = RANK / 4;
const UP_LANE_WORDS: usize = 3;
/// Scale blocks of an up row.
const UP_BLOCKS: usize = RANK / 32;
const _: () = assert!(UP_WORDS == 80 && UP_LANE_WORDS * 32 >= UP_WORDS && UP_WORDS > 64);
/// Values of a BF16 up row one lane takes.
const UP_LANE_VALUES: usize = RANK / 32;
const _: () = assert!(UP_LANE_VALUES * 32 == RANK);
/// Columns one walk of a BF16 up row takes.
const UP_GROUP: usize = 4;
const _: () = assert!((runtime::hc_gated::HIDDEN_ALIGN as usize).is_multiple_of(UP_WARPS));

/// Threads of a combine block, a value each.
const COMBINE_THREADS: u32 = 256;

// ------------------------------------------------------------------ cores

/// `f` plus one Q8_0 word's four terms in byte order, `fma(q_j·d, x_j, f)`:
/// the q8f32 word dot, so a row walked word by word here sums as the q8f32
/// gemv sums it.
#[inline(always)]
fn word_dot(f: f32, q: u32, d: f32, x: [f32; 4]) -> f32 {
    let f = f32::mul_add(q as u8 as i8 as f32 * d, x[0], f);
    let f = f32::mul_add((q >> 8) as u8 as i8 as f32 * d, x[1], f);
    let f = f32::mul_add((q >> 16) as u8 as i8 as f32 * d, x[2], f);
    f32::mul_add((q >> 24) as u8 as i8 as f32 * d, x[3], f)
}

/// The block-wide sum of `v` over a [`NORM_THREADS`] block: each warp's by
/// the butterfly, then the warps' in order from warp 0. Every thread calls
/// it and gets the same value (one barrier).
///
/// # Safety
/// `ws` points at [`NORM_WARPS`] f32 of this block's shared memory that
/// nothing else uses across the call, and nothing writes after it: a warp
/// may still read the slots when another returns.
#[inline(always)]
unsafe fn block_sum(v: f32, tid: usize, ws: *mut f32) -> f32 {
    let s = warp::reduce_sum_f32(v);
    if tid.is_multiple_of(32) {
        // SAFETY: tid / 32 < NORM_WARPS; lane 0 of each warp owns its slot.
        unsafe { *ws.add(tid / 32) = s };
    }
    thread::sync_threads();
    // SAFETY: every slot was written before the barrier above.
    let mut sum = unsafe { *ws };
    for w in 1..NORM_WARPS {
        thread::__unroll_config::<0>();
        // SAFETY: w < NORM_WARPS.
        sum += unsafe { *ws.add(w) };
    }
    sum
}

/// Rows `8·tile + warp` of stream `s` of the down's `[rank·streams ×
/// hidden]` view against `M` columns of `xn`, lane 0 storing column c's sum
/// at `part[(c·rank + row)·streams + s]`.
///
/// # Safety
/// `qs`, `d` hold the down's `RANK` rows of `STREAMS·hidden` values in the
/// q8f32 planes, `xn` `M` columns of `STREAMS·hidden`, `part` `M·RANK·STREAMS`
/// slots, `hidden` a positive multiple of 32 (of 4: every stream starts
/// 16-byte aligned), `lane < 32`, `row < RANK`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
unsafe fn down_row<const M: usize>(
    qs: &[u32],
    d: &[u16],
    xn: &[f32],
    hidden: usize,
    row: usize,
    s: usize,
    lane: usize,
    fault: FaultSink,
    part: &mut DisjointSlice<f32>,
) {
    // SAFETY: view row row·STREAMS + s of hidden values is the down row's
    // stream-s span; its words and scales are inside the planes, and column c
    // of xn at s·hidden + c·STREAMS·hidden spans hidden values inside xn.
    let f = unsafe {
        q8_0_lane_partials_mcol::<M>(
            qs,
            d,
            xn,
            hidden as u32,
            row * STREAMS + s,
            s * hidden,
            STREAMS * hidden,
            lane,
        )
    };
    // SAFETY: down_row_end's contract is this fn's: the warp's (row, s)
    // and its M·RANK·STREAMS slots.
    unsafe { down_row_end::<M>(f, (row, s, lane), fault, part) };
}

/// Lane 0's end of a down row: the warp's `M` column sums of the lane
/// partials `f`, a non-finite one raised as the site's fault, column c's sum
/// stored at `part[(c·rank + row)·streams + s]`.
///
/// # Safety
/// `part` holds `M·RANK·STREAMS` slots, `row < RANK`, `s < STREAMS`, and
/// this warp's (row, s) owns them.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn down_row_end<const M: usize>(
    f: [f32; 8],
    (row, s, lane): (usize, usize, usize),
    fault: FaultSink,
    part: &mut DisjointSlice<f32>,
) {
    let sums = gemv_lane_sums(f, M as u32);
    if lane == 0 {
        let mut finite = true;
        for c in 0..M {
            thread::__unroll_config::<0>();
            finite &= sums[c].is_finite();
        }
        if !finite {
            fault.raise(SITE);
        }
        // SAFETY: the slots (c·RANK + row)·STREAMS + s, c < M, lie inside
        // part's M·RANK·STREAMS, and this warp's (row, s) owns them.
        unsafe { store_cols(part, row * STREAMS + s, RANK * STREAMS, M, sums) };
    }
}

/// Hidden value `h`'s four up rows against the `M` columns of the
/// bottleneck in shared memory `lo_s` (`[M][RANK]`), then lane c's mixed
/// value of column c (module doc).
///
/// # Safety
/// `qs`, `d` hold the up's `STREAMS·hidden` rows of `RANK` values in the
/// q8f32 planes, `lo_s` `M·RANK` f32 of this block's shared memory written
/// before a barrier, `xn` `M` columns of `STREAMS·hidden`, `mixed`
/// `M·hidden` slots, `h < hidden`, `lane < 32`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn up_mix_value<const M: usize>(
    qs: &[u32],
    d: &[u16],
    lo_s: *const f32,
    xn: &[f32],
    hidden: usize,
    h: usize,
    lane: usize,
    fault: FaultSink,
    mixed: &mut DisjointSlice<f32>,
) {
    let mut q = [[0u32; UP_LANE_WORDS]; STREAMS];
    let mut sc = [[0.0f32; UP_LANE_WORDS]; STREAMS];
    for s in 0..STREAMS {
        thread::__unroll_config::<0>();
        let row = s * hidden + h;
        for t in 0..UP_LANE_WORDS {
            thread::__unroll_config::<0>();
            let wd = lane + 32 * t;
            if wd < UP_WORDS {
                // SAFETY: row < STREAMS·hidden and wd < UP_WORDS: the word
                // and its block's scale are inside the planes.
                let (w, b) = unsafe {
                    (
                        *qs.get_unchecked(row * UP_WORDS + wd),
                        *d.get_unchecked(row * UP_BLOCKS + wd / 8),
                    )
                };
                q[s][t] = w;
                sc[s][t] = half_bits_to_f32(b);
            }
        }
    }
    // Columns outer, streams inner: a column's bottleneck words and its four
    // sums live for that column only, and lane c keeps column c's.
    let mut mine = [0.0f32; STREAMS];
    for c in 0..M {
        thread::__unroll_config::<0>();
        let mut f = [0.0f32; STREAMS];
        for t in 0..UP_LANE_WORDS {
            thread::__unroll_config::<0>();
            let wd = lane + 32 * t;
            if wd < UP_WORDS {
                // SAFETY: c < M and 4·wd + 3 < RANK: inside lo_s.
                let x = unsafe {
                    let p = lo_s.add(c * RANK + 4 * wd);
                    [*p, *p.add(1), *p.add(2), *p.add(3)]
                };
                for s in 0..STREAMS {
                    thread::__unroll_config::<0>();
                    f[s] = word_dot(f[s], q[s][t], sc[s][t], x);
                }
            }
        }
        keep_sums(&f, lane == c, &mut mine);
    }
    if lane < M {
        // SAFETY: lane < M, so this fn's contract is mix_col's for column lane.
        unsafe { mix_col(&mine, xn, (hidden, h, lane), fault, mixed) };
    }
}

/// The butterfly of each of a column's four lane partials `f`, kept in
/// `mine` on the lane that owns the column (`own`). Every lane of the warp
/// calls it.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
fn keep_sums(f: &[f32; STREAMS], own: bool, mine: &mut [f32; STREAMS]) {
    for s in 0..STREAMS {
        thread::__unroll_config::<0>();
        let g = warp::reduce_sum_f32(f[s]);
        if own {
            mine[s] = g;
        }
    }
}

/// The mixed value of column `c` for hidden value `h` from its four up rows'
/// sums `g[s]` (module doc), on the lane that keeps them.
///
/// # Safety
/// `xn` holds `c + 1` columns of `STREAMS·hidden`, `mixed` `(c + 1)·hidden`
/// slots, `h < hidden`; the calling warp owns `h`.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn mix_col(
    g: &[f32; STREAMS],
    xn: &[f32],
    (hidden, h, c): (usize, usize, usize),
    fault: FaultSink,
    mixed: &mut DisjointSlice<f32>,
) {
    let mut acc = 0.0f32;
    for s in 0..STREAMS {
        thread::__unroll_config::<0>();
        // SAFETY: (c·STREAMS + s)·hidden + h < (c + 1)·STREAMS·hidden <= xn.len().
        let x = unsafe { *xn.get_unchecked((c * STREAMS + s) * hidden + h) };
        acc = mix_step(acc, x, gate_value(g[s]), s);
    }
    // SAFETY: c·hidden + h < (c + 1)·hidden <= mixed.len(); this warp owns h.
    unsafe { store_mix(acc, c * hidden + h, fault, mixed) };
}

/// Inject block `ib` (row `ib / 4`, stream `ib % 4`) of the down kernels:
/// warp `w` its chunk `w` of `hidden/8` values through the F32 core, the
/// block's eight in order into `ipart[(c·4 + o)·4 + s]` (module doc).
///
/// # Safety
/// The calling kernel's launch contract holds with `inject` 1 (`inj` 16 rows
/// of `4·hidden`, `xn` `m` columns, `ipart` `m·16` slots), `hidden` a
/// positive multiple of 256, `ib < DOWN_INJ_BLOCKS`, every thread of the
/// block calls it, and `ip` is [`DOWN_WARPS`]` · `[`MAX_COLS`] f32 of the
/// block's shared memory nothing else uses.
#[inline(always)]
unsafe fn down_inject(
    inj: &[f32],
    xn: &[f32],
    (hid, m, ib): (usize, u32, usize),
    fault: FaultSink,
    ipart: &mut DisjointSlice<f32>,
    ip: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let wp = tid / 32;
    let lane = warp::lane_id() as usize;
    let (o, s) = (ib / STREAMS, ib % STREAMS);
    let chunk = hid / DOWN_WARPS;
    let at = s * hid + wp * chunk;
    // The inject weight as rows of `chunk`: row o's chunk at `at`.
    let wrow = (o * STREAMS * hid + at) / chunk;
    let cols = m as usize;
    let mut c = 0usize;
    while c < cols {
        let x0 = c * STREAMS * hid + at;
        // SAFETY: x0 + chunk <= m·4·hidden <= xn.len(); wrow·chunk +
        // chunk <= 16·hidden <= inj.len() with inject 1; chunk a positive
        // multiple of 32 (hidden % 256 == 0).
        let v = unsafe {
            let xs = xn.get_unchecked(x0..x0 + chunk);
            warp::reduce_sum_f32(f32_lane_partial_1col(inj, xs, chunk as u32, wrow, lane))
        };
        if lane == 0 {
            // SAFETY: wp < DOWN_WARPS and c < MAX_COLS.
            unsafe { *ip.add(wp * MAX_COLS + c) = v };
        }
        c += 1;
    }
    thread::sync_threads();
    if tid < cols {
        // SAFETY: slots tid, tid + 8, … written before the barrier.
        let mut v = unsafe { *ip.add(tid) };
        for w in 1..DOWN_WARPS {
            thread::__unroll_config::<0>();
            // SAFETY: w < DOWN_WARPS and tid < MAX_COLS.
            v += unsafe { *ip.add(w * MAX_COLS + tid) };
        }
        if !v.is_finite() {
            fault.raise(SITE);
        }
        // SAFETY: (tid·4 + o)·4 + s < m·16 <= ipart.len(); this block
        // owns (o, s) and this thread column tid.
        unsafe { *ipart.get_unchecked_mut((tid * STREAMS + o) * STREAMS + s) = v };
    }
}

/// The up kernels' first stage: every block adds each down row's four
/// partials in stream order into `silu(·/4)` in shared memory `lo_s` (block 0
/// also into `lo`, and with `inject` 1 the weights `2·σ(·/4)` of the inject
/// partials into `wgt`), then the barrier.
///
/// # Safety
/// The calling kernel's launch contract holds (`dpart` `m·1280`, `ipart`
/// `m·16`, `lo` `m·320`, `wgt` `m·4`, `m <= 8`), every thread of its
/// [`UP_THREADS`] block calls it, and `lo_s` is `MAX_COLS·RANK` f32 of the
/// block's shared memory.
#[inline(always)]
unsafe fn up_stage(
    dpart: &[f32],
    ipart: &[f32],
    (m, inject): (u32, u32),
    fault: FaultSink,
    (lo, wgt): (&mut DisjointSlice<f32>, &mut DisjointSlice<f32>),
    lo_s: *mut f32,
) {
    let b = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    let cols = m as usize;
    let mut e = tid;
    while e < cols * RANK {
        // SAFETY: e·4 + 3 < m·1280 <= dpart.len().
        let v = unsafe {
            let p = dpart.as_ptr().add(e * STREAMS);
            ((*p + *p.add(1)) + *p.add(2)) + *p.add(3)
        };
        let l = if v.is_finite() {
            silu(v * INV)
        } else {
            f32::NAN
        };
        // SAFETY: e < m·RANK <= MAX_COLS·RANK, this thread's slot.
        unsafe { *lo_s.add(e) = l };
        if b == 0 {
            if !l.is_finite() {
                fault.raise(SITE);
            }
            // SAFETY: e < m·320 <= lo.len(); block 0 alone writes lo.
            unsafe { *lo.get_unchecked_mut(e) = l };
        }
        e += UP_THREADS as usize;
    }
    if b == 0 && inject == 1 && tid < cols * STREAMS {
        // SAFETY: tid·4 + 3 < m·16 <= ipart.len().
        let v = unsafe {
            let p = ipart.as_ptr().add(tid * STREAMS);
            ((*p + *p.add(1)) + *p.add(2)) + *p.add(3)
        };
        let w = if v.is_finite() {
            2.0 * sigmoid(v * INV)
        } else {
            f32::NAN
        };
        if !w.is_finite() {
            fault.raise(SITE);
        }
        // SAFETY: tid < m·4 <= wgt.len(); block 0 alone writes wgt.
        unsafe { *wgt.get_unchecked_mut(tid) = w };
    }
    thread::sync_threads();
}

/// [`down_row`] for a BF16 down in its words: the F32 core's lane walk over
/// the widened values of the view row (module doc).
///
/// # Safety
/// `w` holds the down's `RANK` rows of `STREAMS·hidden` values two a word,
/// otherwise [`down_row`]'s contract.
#[inline(always)]
unsafe fn down_row_bf16<const M: usize>(
    w: &[u32],
    xn: &[f32],
    (hidden, row, s, lane): (usize, usize, usize, usize),
    fault: FaultSink,
    part: &mut DisjointSlice<f32>,
) {
    // SAFETY: view row row·STREAMS + s of hidden values is the down row's
    // stream-s span, inside the words; column c of xn at s·hidden +
    // c·STREAMS·hidden spans hidden values inside xn.
    let f = unsafe {
        f32_lane_partials_view::<_, M>(
            Bf16Words(w),
            xn,
            (hidden, row * STREAMS + s),
            (s * hidden, STREAMS * hidden),
            lane,
        )
    };
    // SAFETY: down_row_end's contract is this fn's: the warp's (row, s)
    // and its M·RANK·STREAMS slots.
    unsafe { down_row_end::<M>(f, (row, s, lane), fault, part) };
}

/// [`up_mix_value`] for a BF16 up in its words: each of the four up rows
/// against each column of the bottleneck by the F32 core's lane walk over
/// the widened values (lane `L` values `32t + L`, `t` up to `RANK/32`), then
/// the butterfly and [`mix_col`]. The walk takes up to [`UP_GROUP`] columns
/// at once, the lane's four values of step `t` against each of them, so a
/// value is loaded once for the group and only the group's sums stay live;
/// a column's multiply-adds keep their order in `t`.
///
/// # Safety
/// `w` holds the up's `STREAMS·hidden` rows of `RANK` values two a word,
/// otherwise [`up_mix_value`]'s contract.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn up_mix_value_bf16<const M: usize>(
    w: &[u32],
    lo_s: *const f32,
    xn: &[f32],
    (hidden, h, lane): (usize, usize, usize),
    fault: FaultSink,
    mixed: &mut DisjointSlice<f32>,
) {
    let mut mine = [0.0f32; STREAMS];
    for g in 0..M.div_ceil(UP_GROUP) {
        thread::__unroll_config::<0>();
        let mut f = [[0.0f32; STREAMS]; UP_GROUP];
        for t in 0..UP_LANE_VALUES {
            thread::__unroll_config::<0>();
            let mut v = [0.0f32; STREAMS];
            for s in 0..STREAMS {
                thread::__unroll_config::<0>();
                // SAFETY: row s·hidden + h < STREAMS·hidden and 32t + lane <
                // RANK: inside the words.
                v[s] = unsafe { Bf16Words(w).at((s * hidden + h) * RANK + 32 * t + lane) };
            }
            for j in 0..UP_GROUP {
                thread::__unroll_config::<0>();
                let c = g * UP_GROUP + j;
                if c < M {
                    // SAFETY: c < M and 32t + lane < RANK: inside lo_s.
                    let x = unsafe { *lo_s.add(c * RANK + 32 * t + lane) };
                    for s in 0..STREAMS {
                        thread::__unroll_config::<0>();
                        f[j][s] = f32::mul_add(v[s], x, f[j][s]);
                    }
                }
            }
        }
        for j in 0..UP_GROUP {
            thread::__unroll_config::<0>();
            let c = g * UP_GROUP + j;
            if c < M {
                keep_sums(&f[j], lane == c, &mut mine);
            }
        }
    }
    if lane < M {
        // SAFETY: lane < M, so this fn's contract is mix_col's for column lane.
        unsafe { mix_col(&mine, xn, (hidden, h, lane), fault, mixed) };
    }
}

/// Stage the bottleneck and mix one warp's four BF16 rows.
///
/// # Safety
/// Every thread of a 512-thread block calls this with M in 1..=8. The
/// BF16 up entry's launch contract holds for M columns, and lo_s owns
/// M * RANK shared f32 slots for this block.
#[inline(always)]
unsafe fn up_bf16<const M: usize>(
    w: &[u32],
    parts: (&[f32], &[f32]),
    xn: &[f32],
    (hidden, inject): (u32, u32),
    fault: FaultSink,
    (lo, wgt, mixed): (
        &mut DisjointSlice<f32>,
        &mut DisjointSlice<f32>,
        &mut DisjointSlice<f32>,
    ),
    lo_s: *mut f32,
) {
    // SAFETY: the entry's buffers satisfy up_stage for M columns, and
    // every thread calls it before any row guard.
    unsafe { up_stage(parts.0, parts.1, (M as u32, inject), fault, (lo, wgt), lo_s) };
    let h = thread::blockIdx_x() as usize * UP_WARPS + thread::threadIdx_x() as usize / 32;
    if h >= hidden as usize {
        return;
    }
    // SAFETY: h < hidden; up_stage initialized M * RANK shared values
    // before its barrier, and the entry owns this warp's output.
    unsafe {
        up_mix_value_bf16::<M>(
            w,
            lo_s,
            xn,
            (hidden as usize, h, warp::lane_id() as usize),
            fault,
            mixed,
        )
    };
}

/// Values a kernel reads and writes in place ([`Streams`], `xn`): F32
/// values as computed, or BF16 halfwords, each value rounded to nearest even
/// as [`land_values`] rounds it and widened exactly as it is read.
pub(crate) trait Slots {
    /// Value `i`.
    ///
    /// # Safety
    /// Value `i` is live and this thread's alone.
    unsafe fn load(&mut self, i: usize) -> f32;

    /// Writes `v` as value `i` and returns the value it now holds.
    ///
    /// # Safety
    /// Value `i` is live and this thread's alone.
    unsafe fn store(&mut self, i: usize, v: f32) -> f32;
}

impl Slots for DisjointSlice<'_, f32> {
    #[inline(always)]
    unsafe fn load(&mut self, i: usize) -> f32 {
        // SAFETY: this thread's own live value (this fn's contract).
        unsafe { *self.get_unchecked_mut(i) }
    }

    #[inline(always)]
    unsafe fn store(&mut self, i: usize, v: f32) -> f32 {
        // SAFETY: this thread's own live value (this fn's contract).
        unsafe { *self.get_unchecked_mut(i) = v };
        v
    }
}

impl Slots for DisjointSlice<'_, u16> {
    #[inline(always)]
    unsafe fn load(&mut self, i: usize) -> f32 {
        // SAFETY: this thread's own live value (this fn's contract).
        Bf16Words::widen(u32::from(unsafe { *self.get_unchecked_mut(i) }), false)
    }

    #[inline(always)]
    unsafe fn store(&mut self, i: usize, v: f32) -> f32 {
        // Both halves hold v's rounding; the low one is its BF16 bits.
        let pair = cvt_bf16x2_f32(v, v);
        // SAFETY: this thread's own live value (this fn's contract).
        unsafe { *self.get_unchecked_mut(i) = pair as u16 };
        Bf16Words::widen(pair, false)
    }
}

/// One block of `hc_gated_norm_4` and the `hc_gated_norm_4w` family, which
/// say what it computes and how `res` and `xn` hold their values: stream `s`
/// of column `c`, this thread `tid` of its
/// [`NORM_THREADS`]. A thread walks its values `tid`, `tid + NORM_THREADS`
/// and on in chunks of [`NORM_CHUNK`]: a pass loads a chunk's values, then
/// uses them in that order. A value the combine or the init writes into
/// `res` enters the norm as `res` holds it.
///
/// # Safety
/// The entry's launch contract holds: `res` and `xn` hold `m * 4 * hidden`
/// values, `gamma` `4 * hidden`, `y` `(combine + init) * m * hidden`, `wgt`
/// `combine * m * 4`, and `combine + init <= 1`. `c < m`, `s < 4`, `tid <
/// NORM_THREADS`, every thread of the block calls it, and `ws` is as
/// [`block_sum`] asks. The block owns stream `s` of column `c` of `res` and
/// `xn`, and each thread its values `tid + k * NORM_THREADS` of them.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
// PIN(2026-10-27): retain indexed ranges because CUDA unrolling needs range loops.
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn norm_stream<R: Slots, X: Slots>(
    y: &[f32],
    wgt: &[f32],
    gamma: &[f32],
    (hidden, eps): (u32, f32),
    (combine, init): (u32, u32),
    fault: FaultSink,
    (res, xn): (&mut R, &mut X),
    (s, c, tid): (usize, usize, usize),
    ws: *mut f32,
) {
    let hid = hidden as usize;
    let base = (c * STREAMS + s) * hid;
    let w = if combine == 1 {
        // SAFETY: c * 4 + s < m * 4 <= wgt.len() when combine is 1.
        unsafe { *wgt.get_unchecked(c * STREAMS + s) }
    } else {
        0.0
    };
    let reads_y = combine == 1 || init == 1;
    // A stream of one chunk a thread keeps its values in `xv` across the
    // sum, and loads its gains into `yv` before it, instead of reading both
    // back after the sum.
    let one = hid <= NORM_CHUNK * NORM_THREADS;
    let (mut xv, mut yv) = ([0.0f32; NORM_CHUNK], [0.0f32; NORM_CHUNK]);
    let mut acc = 0.0f32;
    let mut i0 = tid;
    while i0 < hid {
        for j in 0..NORM_CHUNK {
            thread::__unroll_config::<0>();
            let i = i0 + j * NORM_THREADS;
            if i < hid {
                // SAFETY: base + i < m * 4 * hidden values of res, this
                // thread's own value (# Safety).
                xv[j] = unsafe { res.load(base + i) };
                if reads_y {
                    // SAFETY: c * hidden + i < m * hidden <= y.len() when
                    // combine or init is 1.
                    yv[j] = unsafe { *y.get_unchecked(c * hid + i) };
                }
            }
        }
        for j in 0..NORM_CHUNK {
            thread::__unroll_config::<0>();
            let i = i0 + j * NORM_THREADS;
            if i < hid {
                let x = if combine == 1 {
                    f32::mul_add(w, yv[j], xv[j])
                } else if init == 1 {
                    yv[j]
                } else {
                    xv[j]
                };
                let x = if reads_y {
                    // SAFETY: base + i < m * 4 * hidden values of res, this
                    // thread's own value (# Safety).
                    unsafe { res.store(base + i, x) }
                } else {
                    x
                };
                xv[j] = x;
                acc = f32::mul_add(x, x, acc);
            }
        }
        i0 += NORM_CHUNK * NORM_THREADS;
    }
    if one {
        for j in 0..NORM_CHUNK {
            thread::__unroll_config::<0>();
            let i = tid + j * NORM_THREADS;
            if i < hid {
                // SAFETY: s * hidden + i < 4 * hidden <= gamma.len().
                yv[j] = unsafe { *gamma.get_unchecked(s * hid + i) };
            }
        }
    }
    // SAFETY: every thread of the block is here, and `ws` is as
    // `block_sum` asks (# Safety).
    let sum = unsafe { block_sum(acc, tid, ws) };
    let mean = sum / hidden as f32 + eps;
    let r = if sum.is_finite() && mean.is_finite() {
        1.0 / mean.sqrt()
    } else {
        if tid == 0 {
            fault.raise(SITE);
        }
        f32::NAN
    };
    let mut i0 = tid;
    while i0 < hid {
        if !one {
            for j in 0..NORM_CHUNK {
                thread::__unroll_config::<0>();
                let i = i0 + j * NORM_THREADS;
                if i < hid {
                    // SAFETY: base + i < m * 4 * hidden values of res, this
                    // thread's own value, which it wrote or read in the
                    // first pass (# Safety).
                    xv[j] = unsafe { res.load(base + i) };
                    // SAFETY: s * hidden + i < 4 * hidden <= gamma.len().
                    yv[j] = unsafe { *gamma.get_unchecked(s * hid + i) };
                }
            }
        }
        for j in 0..NORM_CHUNK {
            thread::__unroll_config::<0>();
            let i = i0 + j * NORM_THREADS;
            if i < hid {
                // SAFETY: base + i < m * 4 * hidden values of xn, this
                // thread's own value (# Safety).
                unsafe { xn.store(base + i, (xv[j] * r) * yv[j]) };
            }
        }
        i0 += NORM_CHUNK * NORM_THREADS;
    }
}

/// Block `b` of a `hc_gated_norm_4w` family launch: stream `b % 4` of column
/// `b / 4` ([`norm_stream`]); a block past the `m` columns returns.
///
/// # Safety
/// [`norm_stream`]'s, but for `c`, `s` and `tid`, which this fn derives
/// from the block's 256 threads.
#[inline(always)]
unsafe fn norm_wide<R: Slots, X: Slots>(
    (y, wgt, gamma): (&[f32], &[f32], &[f32]),
    (hidden, eps): (u32, f32),
    (m, combine, init): (u32, u32, u32),
    fault: FaultSink,
    (res, xn): (&mut R, &mut X),
    ws: *mut f32,
) {
    let b = thread::blockIdx_x() as usize;
    let (s, c) = (b % STREAMS, b / STREAMS);
    if c >= m as usize {
        return; // block-uniform
    }
    let tid = thread::threadIdx_x() as usize;
    let at = (s, c, tid);
    // SAFETY: c < m, s < 4 and tid < 256, and the rest is this fn's
    // contract.
    unsafe {
        norm_stream(
            y,
            wgt,
            gamma,
            (hidden, eps),
            (combine, init),
            fault,
            (res, xn),
            at,
            ws,
        )
    };
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod hc_kernels {
    use super::*;

    /// The norm, and before it what the flags name: block `b` is stream `b
    /// % 4` of column `b / 4`. With `combine` each value first becomes
    /// `fma(wgt[c·4 + s], y[c][i], res)`, with `init` `y[c][i]`, either
    /// written back into `res`; with neither the streams are read as they
    /// are and `y`, `wgt` not at all. Then the stream's sum of squares (module
    /// doc), `r = 1/√(sum/hidden + eps)`, `xn = (x·r)·γ`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m <= 8,
            combine + init <= 1,
            y.len() >= (combine + init) * m * hidden,
            wgt.len() >= combine * m * 4,
            gamma.len() >= 4 * hidden,
            res.len() >= m * 4 * hidden,
            xn.len() >= m * 4 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_norm_4(
        y: &[f32],
        wgt: &[f32],
        gamma: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        combine: u32,
        init: u32,
        fault: FaultSink,
        mut res: DisjointSlice<f32>,
        mut xn: DisjointSlice<f32>,
    ) {
        static mut WS: SharedArray<f32, NORM_WARPS> = SharedArray::UNINIT;
        let b = thread::blockIdx_x() as usize;
        let (s, c) = (b % STREAMS, b / STREAMS);
        if c >= m as usize {
            return; // block-uniform
        }
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `norm_stream`'s `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: the launch contract above holds, c < m, s < 4 and tid <
        // 256 (the contract's block), every thread of the block is here,
        // and `ws` is this block's NORM_WARPS f32.
        unsafe {
            norm_stream(
                y,
                wgt,
                gamma,
                (hidden, eps),
                (combine, init),
                fault,
                (&mut res, &mut xn),
                (s, c, tid),
                ws,
            );
        }
    }

    /// The down's partial dots: blocks below 160 are Q8_0 row tiles, block
    /// `b` stream `b % 4`, rows `8·(b / 4) + warp`; with `inject` 1, blocks
    /// 160..176 are the F32 inject rows, block `160 + 4o + s` row `o` stream
    /// `s`, warp `w` its chunk `w` of `hidden/8` values, the block's eight in
    /// order into `ipart[(c·4 + o)·4 + s]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m <= 8,
            inject <= 1,
            4 * qs.len() >= 1280 * hidden,
            32 * d.len() >= 1280 * hidden,
            inj.len() >= inject * 16 * hidden,
            xn.len() >= m * 4 * hidden,
            dpart.len() >= m * 1280,
            ipart.len() >= m * 16
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_down_4x320(
        qs: &[u32],
        d: &[u16],
        inj: &[f32],
        xn: &[f32],
        hidden: u32,
        m: u32,
        inject: u32,
        fault: FaultSink,
        mut dpart: DisjointSlice<f32>,
        mut ipart: DisjointSlice<f32>,
    ) {
        static mut IP: SharedArray<f32, { DOWN_WARPS * MAX_COLS }> = SharedArray::UNINIT;
        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let wp = tid / 32;
        let lane = warp::lane_id() as usize;
        let hid = hidden as usize;
        if b < DOWN_Q8_BLOCKS {
            let (s, row) = (b % STREAMS, (b / STREAMS) * DOWN_WARPS + wp);
            // SAFETY: row < RANK (160 blocks of 8 rows over 4 streams), the
            // planes cover RANK rows of 4·hidden values and xn m columns by
            // the launch contract; hidden is a positive multiple of 256
            // (host-validated).
            unsafe {
                match m {
                    1 => down_row::<1>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    2 => down_row::<2>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    3 => down_row::<3>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    4 => down_row::<4>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    5 => down_row::<5>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    6 => down_row::<6>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    7 => down_row::<7>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    8 => down_row::<8>(qs, d, xn, hid, row, s, lane, fault, &mut dpart),
                    _ => {}
                }
            }
            return;
        }
        if inject == 1 && b - DOWN_Q8_BLOCKS < DOWN_INJ_BLOCKS {
            // SAFETY: this block's own shared allocation, reached without a
            // reference, which `down_inject` alone uses; the launch contract
            // above holds and the block is an inject block (block-uniform).
            unsafe {
                let ip = SharedArray::as_raw_mut_ptr(&raw mut IP);
                down_inject(inj, xn, (hid, m, b - DOWN_Q8_BLOCKS), fault, &mut ipart, ip);
            }
        }
    }

    /// `hc_gated_down_4x320` over a BF16 down in its words: the same blocks,
    /// each warp's row through the F32 core's lane walk over the widened
    /// values (`down_row_bf16`), the inject blocks unchanged.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m <= 8,
            inject <= 1,
            2 * w.len() >= 1280 * hidden,
            inj.len() >= inject * 16 * hidden,
            xn.len() >= m * 4 * hidden,
            dpart.len() >= m * 1280,
            ipart.len() >= m * 16
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_down_bf16_4x320(
        w: &[u32],
        inj: &[f32],
        xn: &[f32],
        hidden: u32,
        m: u32,
        inject: u32,
        fault: FaultSink,
        mut dpart: DisjointSlice<f32>,
        mut ipart: DisjointSlice<f32>,
    ) {
        static mut IP: SharedArray<f32, { DOWN_WARPS * MAX_COLS }> = SharedArray::UNINIT;
        let b = thread::blockIdx_x() as usize;
        let wp = thread::threadIdx_x() as usize / 32;
        let lane = warp::lane_id() as usize;
        let hid = hidden as usize;
        if b < DOWN_Q8_BLOCKS {
            let (s, row) = (b % STREAMS, (b / STREAMS) * DOWN_WARPS + wp);
            let at = (hid, row, s, lane);
            // SAFETY: row < RANK (160 blocks of 8 rows over 4 streams), the
            // words cover RANK rows of 4·hidden values and xn m columns by
            // the launch contract; hidden is a positive multiple of 256
            // (host-validated).
            unsafe {
                match m {
                    1 => down_row_bf16::<1>(w, xn, at, fault, &mut dpart),
                    2 => down_row_bf16::<2>(w, xn, at, fault, &mut dpart),
                    3 => down_row_bf16::<3>(w, xn, at, fault, &mut dpart),
                    4 => down_row_bf16::<4>(w, xn, at, fault, &mut dpart),
                    5 => down_row_bf16::<5>(w, xn, at, fault, &mut dpart),
                    6 => down_row_bf16::<6>(w, xn, at, fault, &mut dpart),
                    7 => down_row_bf16::<7>(w, xn, at, fault, &mut dpart),
                    8 => down_row_bf16::<8>(w, xn, at, fault, &mut dpart),
                    _ => {}
                }
            }
            return;
        }
        if inject == 1 && b - DOWN_Q8_BLOCKS < DOWN_INJ_BLOCKS {
            // SAFETY: IP is this block's private DOWN_WARPS * MAX_COLS allocation;
            // every thread calls down_inject, which writes it before its barrier.
            unsafe {
                let ip = SharedArray::as_raw_mut_ptr(&raw mut IP);
                down_inject(inj, xn, (hid, m, b - DOWN_Q8_BLOCKS), fault, &mut ipart, ip);
            }
        }
    }

    /// The bottleneck, the combine's weights and the mixed values: every
    /// block adds each down row's four partials in stream order into
    /// `silu(·/4)` in shared memory (block 0 also into `lo`, and with
    /// `inject` 1 the weights `2·σ(·/4)` of the inject partials into `wgt`);
    /// then warp `w` of block `b` is hidden value `16b + w` (module doc).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            m <= 8,
            inject <= 1,
            4 * qs.len() >= 1280 * hidden,
            32 * d.len() >= 1280 * hidden,
            dpart.len() >= m * 1280,
            ipart.len() >= m * 16,
            xn.len() >= m * 4 * hidden,
            lo.len() >= m * 320,
            wgt.len() >= m * 4,
            mixed.len() >= m * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mix_4x320(
        qs: &[u32],
        d: &[u16],
        dpart: &[f32],
        ipart: &[f32],
        xn: &[f32],
        hidden: u32,
        m: u32,
        inject: u32,
        fault: FaultSink,
        mut lo: DisjointSlice<f32>,
        mut wgt: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
    ) {
        static mut LO: SharedArray<f32, { MAX_COLS * RANK }> = SharedArray::UNINIT;
        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let hid = hidden as usize;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `up_stage` writes it, read after its barrier.
        let lo_s = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LO) };
        // SAFETY: the launch contract above holds; every thread of the
        // block calls it.
        unsafe { up_stage(dpart, ipart, (m, inject), fault, (&mut lo, &mut wgt), lo_s) };
        let h = b * UP_WARPS + tid / 32;
        if h >= hid {
            return; // warp-uniform; no barrier follows
        }
        // SAFETY: the planes hold 4·hidden rows of RANK values by the launch
        // contract, lo_s m·RANK values written before the barrier, xn m
        // columns and mixed m·hidden slots; h < hidden.
        unsafe {
            match m {
                1 => up_mix_value::<1>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                2 => up_mix_value::<2>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                3 => up_mix_value::<3>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                4 => up_mix_value::<4>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                5 => up_mix_value::<5>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                6 => up_mix_value::<6>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                7 => up_mix_value::<7>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                8 => up_mix_value::<8>(qs, d, lo_s, xn, hid, h, lane, fault, &mut mixed),
                _ => {}
            }
        }
    }

    /// `hc_gated_up_mix_4x320` over a BF16 up in its words: the same stage,
    /// then each warp's four up rows through `up_mix_value_bf16`.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            m <= 8,
            inject <= 1,
            2 * w.len() >= 1280 * hidden,
            dpart.len() >= m * 1280,
            ipart.len() >= m * 16,
            xn.len() >= m * 4 * hidden,
            lo.len() >= m * 320,
            wgt.len() >= m * 4,
            mixed.len() >= m * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mix_bf16_4x320(
        w: &[u32],
        dpart: &[f32],
        ipart: &[f32],
        xn: &[f32],
        hidden: u32,
        m: u32,
        inject: u32,
        fault: FaultSink,
        mut lo: DisjointSlice<f32>,
        mut wgt: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
    ) {
        static mut LO: SharedArray<f32, { MAX_COLS * RANK }> = SharedArray::UNINIT;
        // SAFETY: LO owns MAX_COLS * RANK values in this block.
        let lo_s = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LO) };
        let parts = (dpart, ipart);
        let geometry = (hidden, inject);
        let outputs = (&mut lo, &mut wgt, &mut mixed);
        // SAFETY: the launch contract holds for m columns; every thread
        // takes the same arm and enters the stage before any row guard.
        unsafe {
            match m {
                2 => up_bf16::<2>(w, parts, xn, geometry, fault, outputs, lo_s),
                3 => up_bf16::<3>(w, parts, xn, geometry, fault, outputs, lo_s),
                4 => up_bf16::<4>(w, parts, xn, geometry, fault, outputs, lo_s),
                5 => up_bf16::<5>(w, parts, xn, geometry, fault, outputs, lo_s),
                6 => up_bf16::<6>(w, parts, xn, geometry, fault, outputs, lo_s),
                7 => up_bf16::<7>(w, parts, xn, geometry, fault, outputs, lo_s),
                8 => up_bf16::<8>(w, parts, xn, geometry, fault, outputs, lo_s),
                _ => {}
            }
        }
    }

    /// One-column BF16 up and mix, with only its live bottleneck in shared memory.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            inject <= 1,
            2 * w.len() >= 1280 * hidden,
            dpart.len() >= 1280,
            ipart.len() >= 16,
            xn.len() >= 4 * hidden,
            lo.len() >= 320,
            wgt.len() >= 4,
            mixed.len() >= hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mix_bf16_1col(
        w: &[u32],
        dpart: &[f32],
        ipart: &[f32],
        xn: &[f32],
        hidden: u32,
        inject: u32,
        fault: FaultSink,
        mut lo: DisjointSlice<f32>,
        mut wgt: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
    ) {
        static mut LO: SharedArray<f32, RANK> = SharedArray::UNINIT;
        // SAFETY: LO owns this column's RANK shared values in this block.
        let lo_s = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LO) };
        // SAFETY: the launch contract holds for one column, and every
        // thread calls the stage before any row guard.
        unsafe {
            up_bf16::<1>(
                w,
                (dpart, ipart),
                xn,
                (hidden, inject),
                fault,
                (&mut lo, &mut wgt, &mut mixed),
                lo_s,
            )
        };
    }

    /// The combine alone: value `i` of `res` (`[m][4][hidden]`) becomes
    /// `fma(wgt[c·4 + s], y[c][i % hidden], res[i])`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            m <= 8,
            y.len() >= m * hidden,
            wgt.len() >= m * 4,
            res.len() >= m * 4 * hidden
        )
    )]
    pub fn hc_gated_combine_4(
        y: &[f32],
        wgt: &[f32],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut res: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let hid = hidden as usize;
        if i >= m as usize * STREAMS * hid {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `hid` is this
        // kernel's parameter and `i` this thread's `index_1d` past the
        // `i < m·4·hidden` guard, and the launch is `domain = 1` with the
        // contract's exact 1-D block.
        let cell = unsafe { StreamFma::<STREAMS>::new(y, wgt, &mut res, i, hid) };
        let out = cell.map(f32::mul_add);
        if !out.is_finite() {
            fault.raise(SITE);
        }
    }
}

// ---------------------------------------------------------------- the host

/// What the norm launch does to the streams before it norms them.
#[derive(Clone, Copy)]
pub enum Before<'a> {
    /// Nothing: the streams as they stand (the site after a standalone
    /// combine, or after the PLE site).
    Plain,
    /// The previous site's combine: `res += wgt · y` per stream, `wgt` the
    /// scratch's, written by the previous inject site's mix.
    Combine { y: &'a DeviceBuffer<f32> },
    /// The first layer's streams: every stream a copy of `y` (the token's
    /// embedding).
    Init { y: &'a DeviceBuffer<f32> },
}

/// One site's weights on the card: `γ` `[streams][hidden]` F32 with the
/// `+1` folded in, the down (`RANK` rows of `streams·hidden`) and up
/// (`streams·hidden` rows of `RANK`), each in the q8f32 Q8_0 planes or BF16
/// words ([`HcMatrix`]), and the F32 inject (`streams` rows of
/// `streams·hidden`), `None` for the head.
pub struct SiteWeights<'a> {
    pub gamma: &'a DeviceBuffer<f32>,
    pub down: HcMatrix<'a>,
    pub up: HcMatrix<'a>,
    pub inject: Option<&'a DeviceTensor<f32>>,
}

/// A resident matrix as a site's down or up and the embedding lookup read
/// it: Q8_0 planes or BF16 words.
#[derive(Clone, Copy)]
pub enum HcMatrix<'a> {
    Q8_0 {
        qs: &'a DeviceTensor<u32>,
        d: &'a DeviceTensor<u16>,
    },
    Bf16(&'a DeviceTensor<u32>),
}

impl<'a> HcMatrix<'a> {
    /// The resident matrix `name` of `w`; another format is refused by name
    /// as `what`'s.
    pub fn of(w: &'a Weights, what: &'static str, name: &str) -> Result<Self, GpuError> {
        let dw = w.resident(what, name)?;
        Self::of_weight(dw).ok_or_else(|| GpuError::tensor(what, name, "Q8_0 planes or BF16 words"))
    }

    /// The resident matrix `dw`; `None` for another format.
    #[must_use]
    pub fn of_weight(dw: &'a DevWeight) -> Option<Self> {
        match dw {
            DevWeight::Q8_0 { qs, d, .. } => Some(HcMatrix::Q8_0 { qs, d }),
            DevWeight::Bf16 { w, .. } => Some(HcMatrix::Bf16(w)),
            _ => None,
        }
    }

    /// Its rows and row width, or what is wrong with its planes.
    pub(crate) fn shape(&self) -> Result<(usize, usize), String> {
        match *self {
            HcMatrix::Q8_0 { qs, d } if qs.rows() == d.rows() && qs.cols() == 8 * d.cols() => {
                Ok((d.rows(), 32 * d.cols()))
            }
            HcMatrix::Q8_0 { qs, d } => Err(format!(
                "Q8_0 planes qs {}x{} and d {}x{}: equal rows, 8 words a scale",
                qs.rows(),
                qs.cols(),
                d.rows(),
                d.cols()
            )),
            HcMatrix::Bf16(w) => Ok((w.rows(), 2 * w.cols())),
        }
    }
}

/// The intermediates of a mix, allocated once for `MAX_COLS` columns and
/// shared by every site: a mix's up writes `wgt` after its norm has read the
/// previous site's, so one set serves the whole trunk in stream order.
pub struct HcScratch {
    /// `[m][streams][hidden]`: the normed streams.
    pub xn: DeviceBuffer<f32>,
    /// `[m][RANK][streams]`: the down's partial dots.
    pub dpart: DeviceBuffer<f32>,
    /// `[m][streams][streams]`: the inject's partial dots.
    pub ipart: DeviceBuffer<f32>,
    /// `[m][RANK]`: the bottleneck after its silu (block 0's copy).
    pub lo: DeviceBuffer<f32>,
    /// `[m][streams]`: the last inject site's combine weights.
    pub wgt: DeviceBuffer<f32>,
    geo: Geometry,
}

impl HcScratch {
    /// Allocate for `geo` at [`MAX_COLS`] columns. Load-time only.
    pub fn new(stream: &CudaStream, geo: Geometry) -> Result<HcScratch, GpuError> {
        let z = geo.sizes(MAX_COLS);
        Ok(HcScratch {
            xn: DeviceBuffer::zeroed(stream, z.xn)?,
            dpart: DeviceBuffer::zeroed(stream, z.down_part)?,
            ipart: DeviceBuffer::zeroed(stream, z.inject_part)?,
            lo: DeviceBuffer::zeroed(stream, z.lo)?,
            wgt: DeviceBuffer::zeroed(stream, z.wgt)?,
            geo,
        })
    }

    /// The geometry it was sized for.
    #[must_use]
    pub fn geometry(&self) -> Geometry {
        self.geo
    }
}

/// [`HcGatedKernels::enqueue_mix`]'s arguments: the streams (`[m][streams]
/// [hidden]`, updated in place by `before`), the site's weights, the norm's
/// epsilon (`attention.layer_norm_rms_epsilon`), the column count, the
/// scratch, and the sub-layer's input (`[m][hidden]`).
pub struct MixArgs<'a> {
    pub res: &'a mut DeviceBuffer<f32>,
    pub before: Before<'a>,
    pub w: SiteWeights<'a>,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub scratch: &'a mut HcScratch,
    pub mixed: &'a mut DeviceBuffer<f32>,
}

/// The file's ggml types for one site's tensors, as the reader classifies
/// them: `γ` and the inject F32, the down and the up Q8_0 or BF16. Any other
/// type is refused by name — the kernels read no other.
///
/// # Errors
/// [`GpuError::Shape`] naming the tensor and its type.
pub fn check_types(
    gamma: GgmlType,
    down: GgmlType,
    up: GgmlType,
    inject: Option<GgmlType>,
) -> Result<(), GpuError> {
    const F32: &[GgmlType] = &[GgmlType::F32];
    const MATRIX: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
    let want = [
        ("hc norm (γ)", gamma, F32),
        ("hc down", down, MATRIX),
        ("hc up", up, MATRIX),
    ];
    for (name, got, types) in want
        .into_iter()
        .chain(inject.map(|t| ("hc inject", t, F32)))
    {
        if !types.contains(&got) {
            return Err(GpuError::shape(
                "hc_gated::check_types",
                format!("{name} is {got:?}; the gated-residual kernels read {types:?} only"),
            ));
        }
    }
    Ok(())
}

/// A host rule's refusal as the launcher's error.
fn refused(what: &'static str, e: HcRefused) -> GpuError {
    GpuError::shape(what, e.to_string())
}

/// The checks both arms' mixes run before their first launch, named `what`:
/// the streams, the sub-layer's input `y` (when `before` reads one) and the
/// mixed output hold the `z` of their columns, and the site's weights are
/// `geo`'s — `γ` its `streams·hidden` values, the down `RANK` rows and the
/// up `streams·hidden` rows, each at the other's width, the inject
/// (when there is one) `streams` rows of `streams·hidden`.
fn check_mix(
    what: &'static str,
    (geo, z): (Geometry, Sizes),
    (res, y, mixed): (usize, Option<&DeviceBuffer<f32>>, &DeviceBuffer<f32>),
    w: &SiteWeights<'_>,
) -> Result<(), GpuError> {
    let (wide, r) = (geo.wide(), geo.r());
    let lens = [
        ("res", res, z.res),
        ("gamma", w.gamma.len(), wide),
        ("mixed", mixed.len(), z.mixed),
        (
            "y",
            y.map_or(0, DeviceBuffer::len),
            if y.is_some() { z.y } else { 0 },
        ),
    ];
    if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
        return Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        ));
    }
    for (name, matrix, rows, k) in [("down", w.down, r, wide), ("up", w.up, wide, r)] {
        match matrix.shape() {
            Ok(got) if got == (rows, k) => {}
            Ok((got_rows, got_k)) => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "{name} planes are {got_rows} rows of {got_k} values, want {rows} rows \
                         of {k}"
                    ),
                ));
            }
            Err(e) => return Err(GpuError::shape(what, format!("{name} planes: {e}"))),
        }
    }
    if let Some(inj) = w.inject
        && (inj.rows() != geo.s() || inj.cols() != wide)
    {
        return Err(GpuError::shape(
            what,
            format!(
                "inject is {}x{}, want {}x{wide}",
                inj.rows(),
                inj.cols(),
                geo.s()
            ),
        ));
    }
    Ok(())
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct HcGatedKernels {
    module: hc_kernels::LoadedModule,
}

impl HcGatedKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcGatedKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(hc_kernels, ctx)? };
        Ok(HcGatedKernels { module })
    }

    /// Enqueue one site's mix over `m` columns: the norm (after `before`),
    /// the down, the up and mean — three launches in order on `stream`. An
    /// inject site (weights with an inject) also leaves its combine weights
    /// in the scratch for the next site's [`Before::Combine`]; the head (no
    /// inject) leaves them as they were. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_mix(&self, stream: &CudaStream, a: MixArgs<'_>) -> Result<(), GpuError> {
        let what = "hc_gated::enqueue_mix";
        let geo = a.scratch.geo;
        let m = geo.cols(a.m).map_err(|e| refused(what, e))?;
        let (y, combine, init) = match a.before {
            Before::Plain => (None, 0u32, 0u32),
            Before::Combine { y } => (Some(y), 1, 0),
            Before::Init { y } => (Some(y), 0, 1),
        };
        check_mix(what, (geo, geo.sizes(m)), (a.res.len(), y, &*a.mixed), &a.w)?;
        let hidden = launch_u32(what, "hidden", geo.d())?;
        let mu = launch_u32(what, "m", m)?;
        let inject = u32::from(a.w.inject.is_some());
        let s = &mut *a.scratch;
        let prep = self.module.prepare_hc_gated_norm_4(LaunchConfig1D::new(
            launch_u32(what, "norm grid", STREAMS * m)?,
            NORM_THREADS as u32,
            0,
        ))?;
        // Without a combine or an init the norm reads neither `y` nor `wgt`,
        // and its contract asks nothing of their lengths: `γ` stands in.
        self.module.hc_gated_norm_4(
            stream,
            &prep,
            y.unwrap_or(a.w.gamma),
            &s.wgt,
            a.w.gamma,
            hidden,
            a.eps,
            mu,
            combine,
            init,
            a.fault,
            a.res,
            &mut s.xn,
        )?;
        let grid = DOWN_Q8_BLOCKS + if inject == 1 { DOWN_INJ_BLOCKS } else { 0 };
        let cfg = LaunchConfig1D::new(launch_u32(what, "down grid", grid)?, DOWN_THREADS, 0);
        let inj_buf = match a.w.inject {
            Some(t) => t.buf(),
            None => &s.wgt,
        };
        match a.w.down {
            HcMatrix::Q8_0 { qs, d } => {
                let prep = self.module.prepare_hc_gated_down_4x320(cfg)?;
                self.module.hc_gated_down_4x320(
                    stream,
                    &prep,
                    qs.buf(),
                    d.buf(),
                    inj_buf,
                    &s.xn,
                    hidden,
                    mu,
                    inject,
                    a.fault,
                    &mut s.dpart,
                    &mut s.ipart,
                )?;
            }
            HcMatrix::Bf16(w) => {
                let prep = self.module.prepare_hc_gated_down_bf16_4x320(cfg)?;
                self.module.hc_gated_down_bf16_4x320(
                    stream,
                    &prep,
                    w.buf(),
                    inj_buf,
                    &s.xn,
                    hidden,
                    mu,
                    inject,
                    a.fault,
                    &mut s.dpart,
                    &mut s.ipart,
                )?;
            }
        }
        let cfg = LaunchConfig1D::new(
            launch_u32(what, "up grid", geo.d() / UP_WARPS)?,
            UP_THREADS,
            0,
        );
        match a.w.up {
            HcMatrix::Q8_0 { qs, d } => {
                let prep = self.module.prepare_hc_gated_up_mix_4x320(cfg)?;
                self.module.hc_gated_up_mix_4x320(
                    stream,
                    &prep,
                    qs.buf(),
                    d.buf(),
                    &s.dpart,
                    &s.ipart,
                    &s.xn,
                    hidden,
                    mu,
                    inject,
                    a.fault,
                    &mut s.lo,
                    &mut s.wgt,
                    a.mixed,
                )?;
            }
            HcMatrix::Bf16(w) if mu == 1 => {
                let prep = self.module.prepare_hc_gated_up_mix_bf16_1col(cfg)?;
                self.module.hc_gated_up_mix_bf16_1col(
                    stream,
                    &prep,
                    w.buf(),
                    &s.dpart,
                    &s.ipart,
                    &s.xn,
                    hidden,
                    inject,
                    a.fault,
                    &mut s.lo,
                    &mut s.wgt,
                    a.mixed,
                )?;
            }
            HcMatrix::Bf16(w) => {
                let prep = self.module.prepare_hc_gated_up_mix_bf16_4x320(cfg)?;
                self.module.hc_gated_up_mix_bf16_4x320(
                    stream,
                    &prep,
                    w.buf(),
                    &s.dpart,
                    &s.ipart,
                    &s.xn,
                    hidden,
                    mu,
                    inject,
                    a.fault,
                    &mut s.lo,
                    &mut s.wgt,
                    a.mixed,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue the combine alone over `m` columns: `res += wgt · y` per
    /// stream with the scratch's weights (the last inject site's). One
    /// launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        res: &mut DeviceBuffer<f32>,
        y: &DeviceBuffer<f32>,
        m: usize,
        fault: FaultSink,
        scratch: &HcScratch,
    ) -> Result<(), GpuError> {
        let what = "hc_gated::enqueue_combine";
        let geo = scratch.geo;
        let m = geo.cols(m).map_err(|e| refused(what, e))?;
        let z = geo.sizes(m);
        for (name, got, need) in [("res", res.len(), z.res), ("y", y.len(), z.y)] {
            if got < need {
                return Err(GpuError::shape(
                    what,
                    format!("{name}.len() {got} < {need}"),
                ));
            }
        }
        let grid = launch_u32(what, "grid", z.res.div_ceil(COMBINE_THREADS as usize))?;
        let prep = self.module.prepare_hc_gated_combine_4(LaunchConfig1D::new(
            grid,
            COMBINE_THREADS,
            0,
        ))?;
        self.module.hc_gated_combine_4(
            stream,
            &prep,
            y,
            &scratch.wgt,
            launch_u32(what, "hidden", geo.d())?,
            launch_u32(what, "m", m)?,
            fault,
            res,
        )?;
        Ok(())
    }
}

// ------------------------------------------------------------ the wide arm

/// Threads of a wide glue block, a value each.
const GLUE_THREADS: u32 = 256;

#[inline(always)]
fn hc_stream_row(hidden: usize, row: usize) -> usize {
    (row % STREAMS) * hidden + row / STREAMS
}

#[derive(Clone, Copy)]
struct HcUpRows<'a> {
    words: &'a [u32],
    hidden: usize,
}

impl HcUpRows<'_> {
    /// The stored value of logical value `at`: logical rows interleave the streams.
    #[inline(always)]
    fn stored_at(self, at: usize) -> usize {
        hc_stream_row(self.hidden, at / RANK) * RANK + at % RANK
    }
}

impl TileRows for HcUpRows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: the logical row is live, and RANK preserves the aligned
        // four-value piece. The permutation stays inside the checked weight.
        unsafe { Bf16Rows(self.words).stage(dst, self.stored_at(at)) };
    }
}

impl MmaRows for HcUpRows<'_> {
    type Piece = ();
    const REPLAY_ROWS: usize = 2;

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: as in `stage`, RANK preserves the aligned eight-value piece.
        unsafe { Bf16Rows(self.words).fetch(dst, self.stored_at(at)) };
    }
}

/// An inject site's down and inject as one weight of `RANK + STREAMS` rows:
/// the down's BF16 words, then the inject's F32 rows, logical value `at`
/// past `split` (the down's values) the inject's value `at - split`. The
/// MMA tile stages an inject piece rounded to BF16 ([`land_values`]) as it
/// fetches it, so a thread holds no inject values across an MMA step; the
/// exact replay stages its F32 values.
#[derive(Clone, Copy)]
struct HcDownRows<'a> {
    down: &'a [u32],
    inject: &'a [f32],
    split: usize,
}

impl TileRows for HcDownRows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        if at < self.split {
            // SAFETY: the four values lie in the down's rows (`split` is a
            // multiple of 4), by this fn's contract.
            unsafe { Bf16Rows(self.down).stage(dst, at) };
        } else {
            // SAFETY: they are the inject's values `at - split ..`, inside its
            // rows, a multiple of 4 from its checked 16-byte aligned base.
            unsafe { F32Rows(self.inject).stage(dst, at - self.split) };
        }
    }
}

impl MmaRows for HcDownRows<'_> {
    /// Whether an inject piece's values pass `mma_fits`; true for a down
    /// piece, copied as its words.
    type Piece = bool;

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> bool {
        if at < self.split {
            // SAFETY: the eight values lie in the down's rows (`split` is a
            // multiple of 8), by this fn's contract.
            unsafe { Bf16Rows(self.down).fetch(dst, at) };
            return true;
        }
        // SAFETY: the inject's values `at - split ..` of eight lie inside its
        // rows, 16-byte aligned (a multiple of 8 from the checked base).
        let v = unsafe { F32Cols(self.inject).fetch(dst, at - self.split) };
        // SAFETY: dst is this thread's own aligned 16 bytes, which no warp
        // reads before the barrier after this fetch's wait.
        unsafe { land_values(v, dst) }
    }

    #[inline(always)]
    unsafe fn land(fits: bool, _dst: *mut u32, _at: usize) -> bool {
        fits
    }
}

/// Value `i` of `hc_gated_mix_4`'s mixed output over the normed streams
/// `xn`, as its doc says; past `m · hidden` nothing.
///
/// # Safety
/// `xn` and `g` hold `m · 4 · hidden` values and `mixed` `m · hidden`, and
/// the calling thread is value `i`'s only writer.
#[inline(always)]
unsafe fn mix_glue<X: RowValues>(
    i: usize,
    xn: X,
    g: &[f32],
    (hidden, m): (u32, u32),
    fault: FaultSink,
    mixed: &mut DisjointSlice<f32>,
) {
    let hid = hidden as usize;
    if i >= m as usize * hid {
        return;
    }
    let (c, h) = (i / hid, i % hid);
    let mut acc = 0.0f32;
    for s in 0..STREAMS {
        thread::__unroll_config::<0>();
        let at = (c * STREAMS + s) * hid + h;
        // SAFETY: at < m·4·hidden, inside xn (this fn's contract).
        let x = unsafe { xn.at(at) };
        // SAFETY: the checked g span has the same live index.
        let gv = unsafe { *g.get_unchecked(at) };
        acc = mix_step(acc, x, gate_value(gv), s);
    }
    // SAFETY: i < m·hidden <= mixed.len(); one thread a value.
    unsafe { store_mix(acc, i, fault, mixed) };
}

/// Columns of the warp's outputs one pass of [`mma_mix`] holds in the
/// scratch, all [`MMA_WARP_ROWS`] rows of each.
const MIX_COLS: usize = MMA_WARP_SCRATCH / MMA_WARP_ROWS;
/// Hidden indices of the warp's rows, and the warp's columns one lane-wide
/// step covers: lane `L` mixes hidden index `L % MIX_HIDDEN` of every
/// `MIX_STEP_COLS`-th column of the pass from `L / MIX_HIDDEN`, so a step's
/// 32 bytes of one stream of a column are one sector of `xn`.
const MIX_HIDDEN: usize = MMA_WARP_ROWS / STREAMS;
const MIX_STEP_COLS: usize = 32 / MIX_HIDDEN;
const _: () =
    assert!(MIX_COLS == 16 && MIX_COLS.is_multiple_of(8) && MMA_WARP_COLS == 2 * MIX_COLS);
const _: () = assert!(MIX_HIDDEN == 16 && MIX_HIDDEN * MIX_STEP_COLS == 32);
/// Spans of `xn` one tile's mix reads: each holds the tile's hidden indices
/// of one stream of one column, 64 bytes of BF16 values.
const XN_LINES: usize = MMA_COLS * STREAMS;
const _: () = assert!(XN_LINES == 2 * MMA_THREADS as usize && MMA_ROWS / STREAMS * 2 == 64);

/// Scratch slot of the gate in warp row `row` and pass column `col`: lines
/// of 32 floats, each two rows of [`MIX_COLS`], their 32 offsets permuted
/// by the line, so both a lane-wide step's 32 gates (sixteen hidden indices
/// of two columns, one stream) and a fragment cell's 32 stores (eight rows
/// of an aligned eight, four even or odd columns of an aligned eight) fall
/// in 32 banks.
#[inline(always)]
fn mix_slot(row: usize, col: usize) -> usize {
    let line = row / 2;
    let q = line / 2;
    let swap = (line & 1) | (q & 6) | ((q & 1) << 3) | ((q & 8) << 1);
    line * 32 + (((row % 2) * MIX_COLS + col) ^ swap)
}

/// Mixes the tile's gates: each pass stores the gates of [`MIX_COLS`] of the
/// warp's columns in its scratch ([`mix_slot`]), and each lane then mixes the
/// streams of one hidden index of the warp's rows for every
/// [`MIX_STEP_COLS`]-th column, so a lane-wide step reads and writes sixteen
/// consecutive values of two columns.
///
/// # Safety
/// All 256 threads call converged. The tile and spans come from the checked
/// HC launch, `xn` the normed streams' BF16 words, and `scratch` is the
/// warp's [`MmaTile::warp_scratch`] after `needs_replay`. If FROM_G, replay
/// completed before this call; otherwise the completed MMA values are used.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the fused epilogue's spans, sizes and scratch, as its kernel passes them"
)]
unsafe fn mma_mix<const FROM_G: bool>(
    tile: &MmaTile,
    xn: &[u32],
    g: &mut DisjointSlice<f32>,
    hid: usize,
    m: u32,
    fault: FaultSink,
    mixed: &mut DisjointSlice<f32>,
    scratch: *mut f32,
) {
    let wide = STREAMS * hid;
    let lane = warp::lane_id() as usize;
    let (r0, c0) = tile.warp_origin();
    let (q, step_col) = (lane % MIX_HIDDEN, lane / MIX_HIDDEN);
    let h = r0 / STREAMS + q;
    for pass in 0..MMA_WARP_COLS / MIX_COLS {
        thread::__unroll_config::<0>();
        let first = c0 + pass * MIX_COLS;
        for mi in 0..MMA_WARP_ROWS / 16 {
            thread::__unroll_config::<0>();
            for half in 0..MIX_COLS / 8 {
                thread::__unroll_config::<0>();
                for i in 0..4 {
                    thread::__unroll_config::<0>();
                    let cell = mi * 16 + (pass * MIX_COLS / 8 + half) * 4 + i;
                    let (r, c) = tile.coordinates(cell);
                    let value = if FROM_G {
                        if r < wide && c < m as usize {
                            // SAFETY: replay completed and this thread owns the live cell.
                            unsafe { *g.get_unchecked_mut(c * wide + r) }
                        } else {
                            0.0
                        }
                    } else {
                        tile.value(cell)
                    };
                    // SAFETY: the cell's row lies in the warp's rows and its
                    // column in this pass's, so the slot is inside the
                    // scratch, this lane's alone.
                    unsafe { *scratch.add(mix_slot(r - r0, c - first)) = gate_value(value) };
                }
            }
        }
        warp::sync_mask(u32::MAX);
        for t in 0..MIX_COLS / MIX_STEP_COLS {
            thread::__unroll_config::<0>();
            let pc = step_col + t * MIX_STEP_COLS;
            let col = first + pc;
            if h < hid && col < m as usize {
                let mut acc = 0.0f32;
                for s in 0..STREAMS {
                    thread::__unroll_config::<0>();
                    // SAFETY: row STREAMS·q + s, column `pc` of the scratch,
                    // published by the warp sync above.
                    let sg = unsafe { *scratch.add(mix_slot(STREAMS * q + s, pc)) };
                    let at = col * wide + hc_stream_row(hid, h * STREAMS + s);
                    // SAFETY: this live column/hidden pair addresses its stream in xn.
                    let x = unsafe { Bf16Words(xn).at(at) };
                    acc = mix_step(acc, x, sg, s);
                }
                // SAFETY: only this lane stores this hidden value and column.
                unsafe { store_mix(acc, col * hid + h, fault, mixed) };
            }
        }
        // The next pass overwrites the scratch this one read.
        warp::sync_mask(u32::MAX);
    }
}

/// The wide arm's up product of the BF16 up rows `w` over the bottleneck
/// (`lo` and its BF16 words and refusal words), its gates mixed with `xn`
/// (the normed streams' BF16 words) in the MMA epilogue ([`mma_mix`]) into
/// `mixed`, and `form` of `mixed`: once
/// the block's stores are published, warp `w` reads back the tile's 32
/// hidden values of its columns `w, w + 8, ...` and writes their form.
///
/// # Safety
/// All 256 threads of a block call converged on the checked launch of
/// `hc_gated_up_mma_mix_4` (its spans, `m > 8` and the MMA grid), `hidden` a
/// multiple of 32; `form` holds room for the `m` columns of `hidden` values,
/// and `sh` is the block's aligned allocation of `STAGES * STAGE_FLOATS`
/// floats.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel's operands, sizes, outputs and form, as its entries pass them"
)]
pub(crate) unsafe fn up_mma_mix_body<F: OutForm>(
    w: &[u32],
    (lo, lo_words, lo_refused, xn): (&[f32], &[u32], &[u32], &[u32]),
    (hidden, m): (u32, u32),
    fault: FaultSink,
    g: &mut DisjointSlice<f32>,
    mixed: &mut DisjointSlice<f32>,
    form: F,
    sh: *mut f32,
) {
    let hid = hidden as usize;
    let wide = STREAMS * hid;
    let rows = HcUpRows {
        words: w,
        hidden: hid,
    };
    let origin = mma_tile_origin::<HcUpRows<'_>>(wide, m as usize);
    // Ask L2 for the tile's XN_LINES spans of xn, thread `tid` spans `tid`
    // and `tid + MMA_THREADS`, so the mix after the MMA finds them there.
    // A hint: it reads nothing into registers and changes no value.
    if let Some((row0, col0)) = origin {
        let h0 = row0 / STREAMS;
        let bytes = 2 * (hid - h0).min(MMA_ROWS / STREAMS);
        let tid = thread::threadIdx_x() as usize;
        for k in 0..XN_LINES / MMA_THREADS as usize {
            thread::__unroll_config::<0>();
            let line = tid + k * MMA_THREADS as usize;
            let (col, s) = (col0 + line / STREAMS, line % STREAMS);
            if col < m as usize {
                // Hidden indices h0 .. h0 + bytes / 2 of stream s of a live
                // column: inside xn.
                l2_prefetch(xn, 2 * (col * wide + s * hid + h0), bytes, 0);
            }
        }
    }
    let cols = Bf16Cols {
        words: lo_words,
        refused: lo_refused,
        exact: lo,
    };
    // SAFETY: the checked launch supplies the shared allocation, 256 threads,
    // the MMA grid, aligned bottleneck columns and the complete BF16 weight.
    let Some(mut tile) = (unsafe { bf16_mma_tile(rows, cols, wide as u32, RANK as u32, m, sh) })
    else {
        return;
    };
    let scratch = tile.warp_scratch(sh, RANK);
    // SAFETY: every thread shares this checked tile's allocation and K.
    if unsafe { tile.needs_replay(sh, RANK) } {
        // SAFETY: g covers the logical output tile, with one writer per cell.
        unsafe { tile.store::<true>(g, wide, m as usize) };
        thread::sync_threads();
        // SAFETY: stores are published and the exact owner uses the same spans.
        unsafe { tile.replay(rows, F32Rows(lo), (wide, RANK, m as usize), g, sh) };
        // SAFETY: replay finished; g, xn and mixed cover this tile.
        unsafe { mma_mix::<true>(&tile, xn, g, hid, m, fault, mixed, scratch) };
    } else {
        // SAFETY: no replay is needed; the checked live cells remain in registers.
        unsafe { mma_mix::<false>(&tile, xn, g, hid, m, fault, mixed, scratch) };
    }
    if F::WRITES
        && let Some((row0, col0)) = origin
    {
        // Publish every warp's mixed stores before the warps read them back.
        thread::sync_threads();
        let h0 = row0 / STREAMS;
        let lane = warp::lane_id() as usize;
        let warps = MMA_THREADS as usize / 32;
        let mut c = col0 + thread::threadIdx_x() as usize / 32;
        while c < (col0 + MMA_COLS).min(m as usize) {
            // SAFETY: the barrier above published this tile's value (c, h0 +
            // lane), inside mixed: c < m and h0 + 32 <= hidden.
            let v = unsafe { *mixed.as_mut_ptr().add(c * hid + h0 + lane) };
            // SAFETY: c < m is warp-uniform, so the warp is converged on a live
            // column, and the 32 values lie in it (this fn's contract).
            unsafe { form.store([v], c, h0, hid) };
            c += warps;
        }
    }
}

#[cuda_module]
mod hc_wide_kernels {
    use super::*;

    /// Interleave four streams in MMA rows and mix g through shared memory,
    /// unstored, with the normed streams' BF16 words `xn`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            hidden > 0, m > 8,
            w.len() >= 4 * hidden * 160,
            lo.len() >= m * 320,
            lo_words.len() >= m * 160,
            lo_refused.len() >= m,
            2 * xn.len() >= m * 4 * hidden,
            g.len() >= m * 4 * hidden,
            mixed.len() >= m * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mma_mix_4(
        w: &[u32],
        lo: &[f32],
        lo_words: &[u32],
        lo_refused: &[u32],
        xn: &[u32],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut g: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
    ) {
        static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let x = (lo, lo_words, lo_refused, xn);
        // SAFETY: the checked launch gives 256 converged threads, the MMA grid
        // and the spans above; sh is this block's allocation.
        unsafe { up_mma_mix_body(w, x, (hidden, m), fault, &mut g, &mut mixed, (), sh) };
    }

    /// [`hc_gated_up_mma_mix_4`], also writing `mixed`'s BF16 form: its
    /// words and each column's refusal word, zero before the launch.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            hidden > 0, m > 8,
            w.len() >= 4 * hidden * 160,
            lo.len() >= m * 320,
            lo_words.len() >= m * 160,
            lo_refused.len() >= m,
            2 * xn.len() >= m * 4 * hidden,
            g.len() >= m * 4 * hidden,
            mixed.len() >= m * hidden,
            2 * words.len() >= m * hidden,
            refused.len() >= m
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mma_mix_4_bf16(
        w: &[u32],
        lo: &[f32],
        lo_words: &[u32],
        lo_refused: &[u32],
        xn: &[u32],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut g: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
        mut words: DisjointSlice<u32>,
        mut refused: DisjointSlice<u32>,
    ) {
        static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let x = (lo, lo_words, lo_refused, xn);
        let form = Bf16Out {
            words: words.as_mut_ptr(),
            refused: refused.as_mut_ptr(),
        };
        // SAFETY: the checked launch gives 256 converged threads, the MMA grid
        // and the spans above, the form's words and refusal words for the m
        // columns of `hidden` values; sh is this block's allocation.
        unsafe { up_mma_mix_body(w, x, (hidden, m), fault, &mut g, &mut mixed, form, sh) };
    }

    /// [`hc_kernels::hc_gated_norm_4`] over any number of columns: block `b`
    /// is stream `b % 4` of column `b / 4`, the same body ([`norm_stream`]).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            combine + init <= 1,
            y.len() >= (combine + init) * m * hidden,
            wgt.len() >= combine * m * 4,
            gamma.len() >= 4 * hidden,
            res.len() >= m * 4 * hidden,
            xn.len() >= m * 4 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_norm_4w(
        y: &[f32],
        wgt: &[f32],
        gamma: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        combine: u32,
        init: u32,
        fault: FaultSink,
        mut res: DisjointSlice<f32>,
        mut xn: DisjointSlice<f32>,
    ) {
        static mut WS: SharedArray<f32, NORM_WARPS> = SharedArray::UNINIT;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `norm_stream`'s `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let (yw, rx) = ((y, wgt, gamma), (&mut res, &mut xn));
        // SAFETY: the launch contract above holds, every thread of the block
        // is here, and `ws` is this block's NORM_WARPS f32.
        unsafe { norm_wide(yw, (hidden, eps), (m, combine, init), fault, rx, ws) };
    }

    /// [`hc_gated_norm_4w`] writing `xn` as BF16 values, each the F32 value
    /// rounded to nearest even: the form a dense MMA card's BF16 down and the
    /// mix read.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            combine + init <= 1,
            y.len() >= (combine + init) * m * hidden,
            wgt.len() >= combine * m * 4,
            gamma.len() >= 4 * hidden,
            res.len() >= m * 4 * hidden,
            xn.len() >= m * 4 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_norm_4w_bf16(
        y: &[f32],
        wgt: &[f32],
        gamma: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        combine: u32,
        init: u32,
        fault: FaultSink,
        mut res: DisjointSlice<f32>,
        mut xn: DisjointSlice<u16>,
    ) {
        static mut WS: SharedArray<f32, NORM_WARPS> = SharedArray::UNINIT;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `norm_stream`'s `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let (yw, rx) = ((y, wgt, gamma), (&mut res, &mut xn));
        // SAFETY: the launch contract above holds, every thread of the block
        // is here, and `ws` is this block's NORM_WARPS f32.
        unsafe { norm_wide(yw, (hidden, eps), (m, combine, init), fault, rx, ws) };
    }

    /// [`hc_gated_norm_4w`] over streams held as BF16 values ([`Streams`]):
    /// each value read widened, each the combine or the init writes rounded
    /// to nearest even, and the norm over the values the streams hold.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            combine + init <= 1,
            y.len() >= (combine + init) * m * hidden,
            wgt.len() >= combine * m * 4,
            gamma.len() >= 4 * hidden,
            res.len() >= m * 4 * hidden,
            xn.len() >= m * 4 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_norm_4w16(
        y: &[f32],
        wgt: &[f32],
        gamma: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        combine: u32,
        init: u32,
        fault: FaultSink,
        mut res: DisjointSlice<u16>,
        mut xn: DisjointSlice<f32>,
    ) {
        static mut WS: SharedArray<f32, NORM_WARPS> = SharedArray::UNINIT;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `norm_stream`'s `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let (yw, rx) = ((y, wgt, gamma), (&mut res, &mut xn));
        // SAFETY: the launch contract above holds, every thread of the block
        // is here, and `ws` is this block's NORM_WARPS f32.
        unsafe { norm_wide(yw, (hidden, eps), (m, combine, init), fault, rx, ws) };
    }

    /// [`hc_gated_norm_4w16`] writing `xn` as [`hc_gated_norm_4w_bf16`] does.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            combine + init <= 1,
            y.len() >= (combine + init) * m * hidden,
            wgt.len() >= combine * m * 4,
            gamma.len() >= 4 * hidden,
            res.len() >= m * 4 * hidden,
            xn.len() >= m * 4 * hidden
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_norm_4w16_bf16(
        y: &[f32],
        wgt: &[f32],
        gamma: &[f32],
        hidden: u32,
        eps: f32,
        m: u32,
        combine: u32,
        init: u32,
        fault: FaultSink,
        mut res: DisjointSlice<u16>,
        mut xn: DisjointSlice<u16>,
    ) {
        static mut WS: SharedArray<f32, NORM_WARPS> = SharedArray::UNINIT;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `norm_stream`'s `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let (yw, rx) = ((y, wgt, gamma), (&mut res, &mut xn));
        // SAFETY: the launch contract above holds, every thread of the block
        // is here, and `ws` is this block's NORM_WARPS f32.
        unsafe { norm_wide(yw, (hidden, eps), (m, combine, init), fault, rx, ws) };
    }

    /// [`hc_kernels::hc_gated_combine_4`] over any number of columns, from
    /// streams held as BF16 values into F32 ones: value `i` of `res` is
    /// `fma(wgt[c·4 + s], y[c][i % hidden], streams[i])`, `streams[i]`
    /// widened exactly.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            y.len() >= m * hidden,
            wgt.len() >= m * 4,
            streams.len() >= m * 4 * hidden,
            res.len() >= m * 4 * hidden
        )
    )]
    pub fn hc_gated_combine_4w(
        y: &[f32],
        wgt: &[f32],
        streams: &[u16],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut res: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let hid = hidden as usize;
        if i >= m as usize * STREAMS * hid {
            return;
        }
        let (cs, h) = (i / hid, i % hid);
        // SAFETY: i < m·4·hidden <= streams.len() (the `requires` above).
        let x = Bf16Words::widen(u32::from(unsafe { *streams.get_unchecked(i) }), false);
        // SAFETY: cs < m·4 <= wgt.len().
        let w = unsafe { *wgt.get_unchecked(cs) };
        // SAFETY: (cs / 4)·hidden + h < m·hidden <= y.len().
        let yv = unsafe { *y.get_unchecked(cs / STREAMS * hid + h) };
        let out = f32::mul_add(w, yv, x);
        // SAFETY: i < m·4·hidden <= res.len(); one thread a value.
        unsafe { *res.get_unchecked_mut(i) = out };
        if !out.is_finite() {
            fault.raise(SITE);
        }
    }

    /// One warp per column writes the F32 bottleneck, its BF16 words and
    /// tensor-core refusal, and the inject's combine weights. Dots are
    /// `[m][320]`, then `[m][4]` with `inject`; with `packed` they are
    /// `[m][324]`, a column's inject dots after its down dots.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            inject <= 1,
            packed <= inject,
            dots.len() >= m * (320 + 4 * inject),
            lo.len() >= m * 320,
            lo_words.len() >= m * 160,
            lo_refused.len() >= m,
            wgt.len() >= inject * m * 4
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub fn hc_gated_lo_4x320(
        dots: &[f32],
        m: u32,
        inject: u32,
        packed: u32,
        fault: FaultSink,
        mut lo: DisjointSlice<f32>,
        mut lo_words: DisjointSlice<u32>,
        mut lo_refused: DisjointSlice<u32>,
        mut wgt: DisjointSlice<f32>,
    ) {
        let c = form_col();
        let lane = warp::lane_id() as usize;
        let stride = RANK + packed as usize * STREAMS;
        let mut fits = true;
        if c < m as usize {
            let mut piece = lane;
            while piece < RANK / 8 {
                let at = c * RANK + 8 * piece;
                // SAFETY: this piece's eight dots lie in column c, aligned
                // to 16 bytes for both the 320- and 324-value strides.
                let mut v =
                    unsafe { F32Cols(dots).fetch(core::ptr::null_mut(), c * stride + 8 * piece) };
                for j in 0..8 {
                    thread::__unroll_config::<8>();
                    let d = v[j];
                    let l = if d.is_finite() {
                        silu(d * INV)
                    } else {
                        f32::NAN
                    };
                    if !l.is_finite() {
                        fault.raise(SITE);
                    }
                    v[j] = l;
                    // SAFETY: this lane alone owns these eight values of
                    // live column c, inside the checked bottleneck span.
                    unsafe { *lo.get_unchecked_mut(at + j) = l };
                }
                // SAFETY: this lane owns four aligned words of column c.
                fits &= unsafe { land_values(v, lo_words.as_mut_ptr().add(at / 2)) };
                piece += 32;
            }
        }
        // SAFETY: all lanes participate, and this warp owns column c.
        unsafe { store_refused(&mut lo_refused, c, m as usize, fits) };
        if c >= m as usize || inject == 0 || lane >= STREAMS {
            return;
        }
        let j = c * STREAMS + lane;
        let at = if packed == 1 {
            c * stride + RANK + lane
        } else {
            m as usize * RANK + j
        };
        // SAFETY: this live column's inject dot lies in the checked span.
        let v = unsafe { *dots.get_unchecked(at) };
        let w = if v.is_finite() {
            2.0 * sigmoid(v * INV)
        } else {
            f32::NAN
        };
        if !w.is_finite() {
            fault.raise(SITE);
        }
        // SAFETY: this lane alone writes its live column's inject weight.
        unsafe { *wgt.get_unchecked_mut(j) = w };
    }

    /// The mixed values of `m` columns: value `i = c·hidden + h` is `(xn_0·
    /// σ(g_0), then fma(xn_s, σ(g_s), ·) for s = 1, 2, 3)·¼`, `xn_s` and
    /// `g_s` value `s·hidden + h` of column `c` of the normed streams and of
    /// the up's rows (`[m][4·hidden]` both) ([`mix_glue`]).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            xn.len() >= m * 4 * hidden,
            g.len() >= m * 4 * hidden,
            mixed.len() >= m * hidden
        )
    )]
    pub fn hc_gated_mix_4(
        xn: &[f32],
        g: &[f32],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut mixed: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract above holds, one thread a value.
        unsafe { mix_glue(i, xn, g, (hidden, m), fault, &mut mixed) };
    }

    /// [`hc_gated_mix_4`] over the normed streams' BF16 words `xn`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            2 * xn.len() >= m * 4 * hidden,
            g.len() >= m * 4 * hidden,
            mixed.len() >= m * hidden
        )
    )]
    pub fn hc_gated_mix_4_bf16(
        xn: &[u32],
        g: &[f32],
        hidden: u32,
        m: u32,
        fault: FaultSink,
        mut mixed: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract above holds, one thread a value.
        unsafe { mix_glue(i, Bf16Words(xn), g, (hidden, m), fault, &mut mixed) };
    }

    /// The down over the normed streams' BF16 words as one BF16 tensor-core
    /// product ([`HcDownRows`], `bf16_mma_body`), with an inject site's
    /// (`inject` 1) inject rows below the down's: `y[c·(320 + 4·inject) +
    /// r]`, row `r < 320` the down's dot of column `c` and row `320 + s` the
    /// inject's, each cell of an out-of-range operand replayed on the exact
    /// F32 owner over the widened words.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            hidden > 0,
            m > 0,
            inject <= 1,
            down.len() >= 640 * hidden,
            inj.len() >= inject * 16 * hidden,
            2 * xn.len() >= m * 4 * hidden,
            y.len() >= m * (320 + 4 * inject)
        )
    )]
    pub fn hc_gated_down_mma_4(
        down: &[u32],
        inj: &[f32],
        xn: &[u32],
        hidden: u32,
        m: u32,
        inject: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let wide = STREAMS * hidden as usize;
        let rows = HcDownRows {
            down,
            inject: inj,
            split: RANK * wide,
        };
        let n_rows = RANK + STREAMS * inject as usize;
        // SAFETY: the checked launch supplies 256 converged threads, the MMA
        // grid, the aligned normed streams' words and weights and the output
        // spans of the contract: 320 BF16 rows, with an inject 4 F32 rows
        // below them, of 4·hidden values.
        unsafe {
            bf16_mma_body(
                rows,
                Bf16WordCols(xn),
                n_rows as u32,
                wide as u32,
                m,
                &mut y,
                sh,
            )
        };
    }
}

/// The intermediates of a wide mix, allocated once for up to `cols` columns
/// and shared by every site of a walk, as [`HcScratch`] is: the normed
/// streams and their 32-value activations, the down's and the inject's whole
/// dots, the bottleneck and its activations, the up's rows, and the last
/// inject site's combine weights.
pub struct HcWideScratch {
    /// `[m][streams][hidden]`: F32 values, or where a dense MMA card's BF16
    /// down reads them, BF16 values in its first `m·streams·hidden` halfwords
    /// (a scratch made for BF16 alone holds only those).
    pub xn: DeviceBuffer<f32>,
    pub xq: GemmAct32,
    /// The down rows' and the inject rows' dots: `[m][RANK]`, then
    /// `[m][streams]` at an inject site; `[m][RANK + streams]`, a column's
    /// inject dots after its down dots, where one product packs both
    /// (`hc_gated_down_mma_4`).
    pub down: DeviceBuffer<f32>,
    /// `[m][RANK]`: the bottleneck after its silu.
    pub lo: DeviceBuffer<f32>,
    /// BF16 bottleneck pairs and one tensor-core refusal word per column.
    pub lo_words: DeviceBuffer<u32>,
    pub lo_refused: DeviceBuffer<u32>,
    pub loq: GemmAct32,
    /// `[m][streams·hidden]`: the up's rows.
    pub g: DeviceBuffer<f32>,
    /// `[m][streams]`: the last inject site's combine weights.
    pub wgt: DeviceBuffer<f32>,
    cols: usize,
    geo: Geometry,
}

impl HcWideScratch {
    /// Allocate for `geo` at `cols` (1..=`GEMM_MAX_SLOTS`) columns, the
    /// normed streams at a halfword a value when `bf16_xn` (every mix the
    /// scratch serves has a BF16 down on a dense MMA card; a mix that norms
    /// into F32 values is then refused by name). Load-time only.
    pub fn new(
        stream: &CudaStream,
        geo: Geometry,
        cols: usize,
        bf16_xn: bool,
    ) -> Result<HcWideScratch, GpuError> {
        let z = geo.sizes(cols);
        Ok(HcWideScratch {
            xn: DeviceBuffer::zeroed(stream, if bf16_xn { z.xn.div_ceil(2) } else { z.xn })?,
            xq: GemmAct32::new(stream, cols, geo.wide())?,
            down: DeviceBuffer::zeroed(stream, z.lo + z.wgt)?,
            lo: DeviceBuffer::zeroed(stream, z.lo)?,
            lo_words: DeviceBuffer::zeroed(stream, z.lo / 2)?,
            lo_refused: DeviceBuffer::zeroed(stream, cols)?,
            loq: GemmAct32::new(stream, cols, geo.r())?,
            g: DeviceBuffer::zeroed(stream, z.xn)?,
            wgt: DeviceBuffer::zeroed(stream, z.wgt)?,
            cols,
            geo,
        })
    }

    /// Columns it holds.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Device bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        [&self.xn, &self.down, &self.lo, &self.g, &self.wgt]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.xq.bytes()
            + self.loq.bytes()
            + self.lo_words.num_bytes()
            + self.lo_refused.num_bytes()
    }
}

/// Residual streams `[m][streams][hidden]` a kernel reads and writes in
/// place: F32 values, or BF16 values (a ubatch walk's), each rounded to
/// nearest even as it is written and widened exactly as it is read
/// ([`Slots`]). The BF16 form is ours: ik's and mainline's Qwen3.8 graphs
/// keep the streams as F32 tensors between layers (ik
/// `src/graphs/build_qwen4exp.cpp:590` the init from the F32 embedding rows,
/// `:96` the combine's add; mainline `src/models/qwen4exp.cpp:389` and
/// `:346`, its fused combine taking only an F32 residual, `ggml.c:6572`).
pub enum Streams<'a> {
    F32(&'a mut DeviceBuffer<f32>),
    Bf16(&'a mut DeviceBuffer<u16>),
}

impl Streams<'_> {
    /// Values it holds.
    pub(crate) fn len(&self) -> usize {
        match self {
            Streams::F32(b) => b.len(),
            Streams::Bf16(b) => b.len(),
        }
    }
}

/// [`HcWideKernels::enqueue_mix`]'s arguments: [`MixArgs`]' over `m`
/// columns of the wide scratch, with the streams in either form, and the
/// GEMM family with the one-expert table its down and up read — filled for
/// exactly `m` slots (`GemmKernels::enqueue_route_dense`, refused by name
/// otherwise), which the caller owns.
pub struct WideMixArgs<'a> {
    pub res: Streams<'a>,
    pub before: Before<'a>,
    pub w: SiteWeights<'a>,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub scratch: &'a mut HcWideScratch,
    pub gemm: &'a Gemm32Kernels,
    /// The BF16 prompt kernels, for a site whose down or up is BF16.
    pub q8f32: &'a Q8F32Kernels,
    pub dense: &'a GemmRoute,
    pub mixed: &'a mut DeviceBuffer<f32>,
    /// A form of `mixed` for the scratch beside it: written by the up/mix
    /// launch when the up and the mix run as one MMA product, else marked as
    /// no rows' ([`GemmAct32::holds`] tells which).
    pub out: Option<(MmaForm, &'a mut GemmAct32)>,
}

/// The wide arm's loaded module. Owns no stream: each enqueue takes the
/// engine stream.
pub struct HcWideKernels {
    module: hc_wide_kernels::LoadedModule,
}

impl HcWideKernels {
    /// Load the wide module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcWideKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(hc_wide_kernels, ctx)? };
        Ok(HcWideKernels { module })
    }

    /// Enqueue one site's mix over `m` columns (module doc's wide arm): the
    /// norm after `before` over the streams in either form, the down and
    /// optional inject projections, the bottleneck and weights, and the
    /// up/mix. Q8 projections add activation
    /// quantizers. On a compute capability 12 device a BF16 down reads the normed streams as BF16
    /// words, which the norm writes in place of their F32 values, and the
    /// mix reads the same words: the down and an inject site's inject are one
    /// product, and a BF16 up with m > 8 consumes g in the MMA epilogue.
    /// Other paths use separate down and inject products and separate up and
    /// mix launches. An inject site leaves its combine weights in the scratch
    /// for the next site's [`Before::Combine`]. `m` outside
    /// `1..=scratch.cols()`, a short buffer, planes of another shape
    /// (the narrow arm's check, `check_mix`), a table filled for another
    /// count or, when the up and the mix run as one product, an `out`
    /// without room for its form of `mixed` are refused by name before any
    /// launch. Asynchronous, allocation-free.
    pub fn enqueue_mix(&self, stream: &CudaStream, a: WideMixArgs<'_>) -> Result<(), GpuError> {
        let what = "hc_gated::enqueue_mix_wide";
        let geo = a.scratch.geo;
        let m = a.m;
        if m == 0 || m > a.scratch.cols {
            return Err(GpuError::shape(
                what,
                format!("{m} columns; the wide scratch holds 1..={}", a.scratch.cols),
            ));
        }
        let z = geo.sizes(m);
        let (wide, r) = (geo.wide(), geo.r());
        let (y, combine, init) = match a.before {
            Before::Plain => (None, 0u32, 0u32),
            Before::Combine { y } => (Some(y), 1, 0),
            Before::Init { y } => (Some(y), 0, 1),
        };
        check_mix(what, (geo, z), (a.res.len(), y, &*a.mixed), &a.w)?;
        if a.dense.filled() != Some(m) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the one-expert table is filled for {:?} slots, the mix has {m} columns",
                    a.dense.filled()
                ),
            ));
        }
        let hidden = launch_u32(what, "hidden", geo.d())?;
        let mu = launch_u32(what, "m", m)?;
        let inject = u32::from(a.w.inject.is_some());
        // The BF16 down's product over the normed streams' BF16 words, the
        // inject rows (if any) below the down's.
        let words = match a.w.down {
            HcMatrix::Bf16(down) if a.q8f32.dense_mma => {
                let rows = r + geo.s() * inject as usize;
                let t = mma_launch_words(what, rows, wide, &a.scratch.xn, m, &a.scratch.down)?;
                check_alignment::<U32x4, _>(what, "down", down.buf())?;
                if let Some(inj) = a.w.inject {
                    check_alignment::<F32x4, _>(what, "inject", inj.buf())?;
                }
                Some((down, t))
            }
            _ => None,
        };
        if words.is_none() && a.scratch.xn.len() < z.xn {
            return Err(GpuError::shape(
                what,
                format!(
                    "the scratch's normed streams hold {} f32, made for BF16 words; an F32 norm \
                     (no BF16 down on a dense MMA card) needs {}",
                    a.scratch.xn.len(),
                    z.xn
                ),
            ));
        }
        let fused = if let HcMatrix::Bf16(w) = a.w.up
            && words.is_some()
            && m > MAX_COLS
        {
            Some((w, bf16_mma_launch(what, w, &a.scratch.lo, m, &a.scratch.g)?))
        } else {
            None
        };
        // The fused up/mix writes `mixed`'s form: its room is checked, and its
        // refusal words zeroed, before any launch.
        let mut out = a.out;
        let planes = match (&fused, &mut out) {
            (Some(_), Some((form, act))) => {
                Some((*form, act.out_planes(stream, what, *form, (geo.d(), m))?))
            }
            _ => None,
        };
        let s = &mut *a.scratch;
        let norm = LaunchConfig1D::new(
            launch_u32(what, "norm grid", STREAMS * m)?,
            NORM_THREADS as u32,
            0,
        );
        // Without a combine or an init the norm reads neither `y` nor `wgt`:
        // `γ` stands in.
        let y = y.unwrap_or(a.w.gamma);
        let (wgt, gamma, eps, fault) = (&s.wgt, a.w.gamma, a.eps, a.fault);
        match (a.res, words.is_some()) {
            (Streams::F32(res), false) => {
                let prep = self.module.prepare_hc_gated_norm_4w(norm)?;
                self.module.hc_gated_norm_4w(
                    stream, &prep, y, wgt, gamma, hidden, eps, mu, combine, init, fault, res,
                    &mut s.xn,
                )?;
            }
            (Streams::F32(res), true) => {
                let prep = self.module.prepare_hc_gated_norm_4w_bf16(norm)?;
                let mut xn = WindowMut::<u16>::of_mut(&mut s.xn, 0, z.xn)?;
                self.module.hc_gated_norm_4w_bf16(
                    stream, &prep, y, wgt, gamma, hidden, eps, mu, combine, init, fault, res,
                    &mut xn,
                )?;
            }
            (Streams::Bf16(res), false) => {
                let prep = self.module.prepare_hc_gated_norm_4w16(norm)?;
                self.module.hc_gated_norm_4w16(
                    stream, &prep, y, wgt, gamma, hidden, eps, mu, combine, init, fault, res,
                    &mut s.xn,
                )?;
            }
            (Streams::Bf16(res), true) => {
                let prep = self.module.prepare_hc_gated_norm_4w16_bf16(norm)?;
                let mut xn = WindowMut::<u16>::of_mut(&mut s.xn, 0, z.xn)?;
                self.module.hc_gated_norm_4w16_bf16(
                    stream, &prep, y, wgt, gamma, hidden, eps, mu, combine, init, fault, res,
                    &mut xn,
                )?;
            }
        }
        // The normed streams' BF16 words, two values a word.
        let xw = match words {
            Some(_) => Some(Window::<u32>::of(&s.xn, 0, z.xn / 2)?),
            None => None,
        };
        let packed = if let (Some((down, t)), Some(xw)) = (words, &xw) {
            let prep = self
                .module
                .prepare_hc_gated_down_mma_4(LaunchConfig1D::new(t.grid, MMA_THREADS, 0))?;
            // Without an inject the launch reads no inject rows: `γ` stands in.
            let inj = a.w.inject.map_or(a.w.gamma, |inj| inj.buf());
            self.module.hc_gated_down_mma_4(
                stream,
                &prep,
                down.buf(),
                inj,
                xw,
                hidden,
                mu,
                inject,
                &mut s.down,
            )?;
            inject
        } else {
            match a.w.down {
                HcMatrix::Q8_0 { qs, d } => {
                    a.gemm
                        .enqueue_quantize_gemm32(stream, &s.xn, m, &mut s.xq, a.fault)?;
                    a.gemm.enqueue_gemm32(
                        stream,
                        Gemm32Args {
                            w: Gemm32Weight::Q8_0Plane { qs, d },
                            rows_per_expert: r,
                            act: &s.xq,
                            route: a.dense,
                            input: GemmInput::PerSlot,
                            y: &mut s.down,
                        },
                    )?;
                }
                HcMatrix::Bf16(w) => {
                    a.q8f32
                        .enqueue_bf16_prompt(stream, w, &s.xn, m, &mut s.down)?
                }
            }
            if let Some(inj) = a.w.inject {
                let mut dots = WindowMut::of_mut(&mut s.down, size_of::<f32>() * z.lo, z.wgt)?;
                a.gemm.enqueue_f32_tile(stream, inj, &s.xn, m, &mut dots)?;
            }
            0
        };
        let prep = self.module.prepare_hc_gated_lo_4x320(LaunchConfig1D::new(
            launch_u32(what, "lo grid", m.div_ceil(GLUE_THREADS as usize / 32))?,
            GLUE_THREADS,
            0,
        ))?;
        // Without an inject the launch writes no `wgt`: the scratch's own
        // stands in, its length untested.
        self.module.hc_gated_lo_4x320(
            stream,
            &prep,
            &s.down,
            mu,
            inject,
            packed,
            a.fault,
            &mut s.lo,
            &mut s.lo_words,
            &mut s.lo_refused,
            &mut s.wgt,
        )?;
        if let (Some((w, t)), Some(xw)) = (fused, &xw) {
            let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
            let at = a.mixed.cu_deviceptr();
            match planes {
                None => {
                    let prep = self.module.prepare_hc_gated_up_mma_mix_4(cfg)?;
                    self.module.hc_gated_up_mma_mix_4(
                        stream,
                        &prep,
                        w.buf(),
                        &s.lo,
                        &s.lo_words,
                        &s.lo_refused,
                        xw,
                        hidden,
                        mu,
                        a.fault,
                        &mut s.g,
                        a.mixed,
                    )?;
                }
                Some((MmaForm::Bf16, [words, _, refused])) => {
                    let prep = self.module.prepare_hc_gated_up_mma_mix_4_bf16(cfg)?;
                    self.module.hc_gated_up_mma_mix_4_bf16(
                        stream,
                        &prep,
                        w.buf(),
                        &s.lo,
                        &s.lo_words,
                        &s.lo_refused,
                        xw,
                        hidden,
                        mu,
                        a.fault,
                        &mut s.g,
                        a.mixed,
                        words,
                        refused,
                    )?;
                }
                Some((MmaForm::Mxfp8, planes)) => {
                    a.gemm.enqueue_hc_up_mix_mxfp8(
                        stream,
                        HcUpMixArgs {
                            cfg,
                            w: w.buf(),
                            lo: &s.lo,
                            lo_words: &s.lo_words,
                            lo_refused: &s.lo_refused,
                            xn: xw,
                            hidden,
                            m: mu,
                            fault: a.fault,
                            g: &mut s.g,
                            mixed: a.mixed,
                            planes,
                        },
                    )?;
                }
            }
            if let Some((form, act)) = out {
                act.wrote(form, at, m);
            }
            return Ok(());
        }
        if let Some((form, act)) = out {
            act.forget(form);
        }
        match a.w.up {
            HcMatrix::Q8_0 { qs, d } => {
                a.gemm
                    .enqueue_quantize_gemm32(stream, &s.lo, m, &mut s.loq, a.fault)?;
                a.gemm.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w: Gemm32Weight::Q8_0Plane { qs, d },
                        rows_per_expert: wide,
                        act: &s.loq,
                        route: a.dense,
                        input: GemmInput::PerSlot,
                        y: &mut s.g,
                    },
                )?;
            }
            HcMatrix::Bf16(w) => a.q8f32.enqueue_bf16_prompt(stream, w, &s.lo, m, &mut s.g)?,
        }
        let mix = LaunchConfig1D::new(
            launch_u32(what, "mix grid", z.mixed.div_ceil(GLUE_THREADS as usize))?,
            GLUE_THREADS,
            0,
        );
        match &xw {
            Some(xw) => {
                let prep = self.module.prepare_hc_gated_mix_4_bf16(mix)?;
                self.module
                    .hc_gated_mix_4_bf16(stream, &prep, xw, &s.g, hidden, mu, a.fault, a.mixed)?;
            }
            None => {
                let prep = self.module.prepare_hc_gated_mix_4(mix)?;
                self.module
                    .hc_gated_mix_4(stream, &prep, &s.xn, &s.g, hidden, mu, a.fault, a.mixed)?;
            }
        }
        Ok(())
    }

    /// Enqueue the combine alone over `m` columns with the wide scratch's
    /// weights (the last inject site's), from BF16 streams into F32 ones:
    /// `res = streams + wgt · y` per stream, each value of `streams` widened
    /// exactly. One launch. Asynchronous, allocation-free.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        (streams, res): (&DeviceBuffer<u16>, &mut DeviceBuffer<f32>),
        y: &DeviceBuffer<f32>,
        m: usize,
        fault: FaultSink,
        scratch: &HcWideScratch,
    ) -> Result<(), GpuError> {
        let what = "hc_gated::enqueue_combine_wide";
        if m == 0 || m > scratch.cols {
            return Err(GpuError::shape(
                what,
                format!("{m} columns; the wide scratch holds 1..={}", scratch.cols),
            ));
        }
        let geo = scratch.geo;
        let z = geo.sizes(m);
        for (name, got, need) in [
            ("streams", streams.len(), z.res),
            ("res", res.len(), z.res),
            ("y", y.len(), z.y),
        ] {
            if got < need {
                return Err(GpuError::shape(
                    what,
                    format!("{name}.len() {got} < {need}"),
                ));
            }
        }
        let grid = launch_u32(what, "grid", z.res.div_ceil(COMBINE_THREADS as usize))?;
        let prep = self
            .module
            .prepare_hc_gated_combine_4w(LaunchConfig1D::new(grid, COMBINE_THREADS, 0))?;
        self.module.hc_gated_combine_4w(
            stream,
            &prep,
            y,
            &scratch.wgt,
            streams,
            launch_u32(what, "hidden", geo.d())?,
            launch_u32(what, "m", m)?,
            fault,
            res,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_mma_interleaved_rows_and_mix_order() {
        for hidden in [64, 2560] {
            for h in 0..hidden {
                for stream in 0..STREAMS {
                    assert_eq!(
                        hc_stream_row(hidden, h * STREAMS + stream),
                        stream * hidden + h
                    );
                }
            }
        }
        assert_eq!(mix_step(0.0, -0.0, 0.5, 0).to_bits(), (-0.0f32).to_bits());
        // The fused product retains -2^-46 instead of rounding to zero.
        let x = f32::from_bits(0x3f80_0001);
        let gate = f32::from_bits(0x3f7f_fffe);
        assert_eq!(mix_step(-1.0, x, gate, 1).to_bits(), 0xa880_0000);
    }

    #[test]
    fn hc_mix_slots_fill_the_scratch_and_every_bank() {
        let mut seen = std::collections::HashSet::new();
        for row in 0..MMA_WARP_ROWS {
            for col in 0..MIX_COLS {
                assert!(seen.insert(mix_slot(row, col)));
            }
        }
        assert_eq!(seen.len(), MMA_WARP_SCRATCH);
        assert!(seen.iter().all(|&slot| slot < MMA_WARP_SCRATCH));
        // Each lane-wide read of the mix: lane L at hidden index L % MIX_HIDDEN
        // of column L / MIX_HIDDEN + t·MIX_STEP_COLS, one stream.
        for t in 0..MIX_COLS / MIX_STEP_COLS {
            for s in 0..STREAMS {
                let banks: std::collections::HashSet<usize> = (0..32)
                    .map(|lane| {
                        let (q, step_col) = (lane % MIX_HIDDEN, lane / MIX_HIDDEN);
                        mix_slot(STREAMS * q + s, step_col + t * MIX_STEP_COLS) % 32
                    })
                    .collect();
                assert_eq!(banks.len(), 32);
            }
        }
        // Each lane-wide store of one fragment cell: lane L at row
        // L / 4 and column 2·(L % 4) + p of an aligned eight by eight.
        for r8 in (0..MMA_WARP_ROWS).step_by(8) {
            for c8 in (0..MIX_COLS).step_by(8) {
                for p in 0..2 {
                    let banks: std::collections::HashSet<usize> = (0..32)
                        .map(|lane| mix_slot(r8 + lane / 4, c8 + 2 * (lane % 4) + p) % 32)
                        .collect();
                    assert_eq!(banks.len(), 32);
                }
            }
        }
    }
}
