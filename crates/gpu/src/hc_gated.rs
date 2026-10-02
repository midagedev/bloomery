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
//!   order, and lane 0 writes `(xn_0·σ(g_0), fma for the rest)·(1/4)`.
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
//! over a whole ubatch of `m` columns, any count: the norm and the combine
//! are the entries above with the eight-column cap lifted
//! (`hc_gated_norm_4w`, `hc_gated_combine_4w`, the same bodies); the down
//! and the up are the 32-value GEMM (`crate::gemm`'s `gemm_q8_0p` over a
//! one-expert table) on `xn` and on the bottleneck quantized per 32 values;
//! the inject rows are the wide F32 tile; two glue entries finish the rule —
//! `hc_gated_lo_4x320` (`silu(·/4)` of each down row, `2·σ(·/4)` of each
//! inject row) and `hc_gated_mix_4` (`(xn_0·σ(g_0), fma for the rest)·¼`,
//! the up's order). A column's values are a function of that column alone;
//! against the narrow arm the down and the up read q8 activations and sum
//! each row whole, so the two arms agree to the error of that quantization
//! and are not bit-equal. The same values raise [`SITE`].

use crate::fault::{FaultSink, FaultSite};
use crate::flash::half_bits_to_f32;
use crate::gemm::{Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmRoute};
use crate::linear::{sigmoid, silu};
use crate::q8f32::{f32_lane_partial_1col, gemv_lane_sums, q8_0_lane_partials_mcol};
use crate::tensor::DeviceTensor;
use crate::view::{Elem, StreamFma};
use crate::{GpuError, launch_u32, store_cols};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
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
const _: () = assert!(
    INSTANCES.len() == 1
        && INSTANCES[0].streams as usize == STREAMS
        && INSTANCES[0].rank as usize == RANK
);
const _: () = assert!(MAX_COLS == 8 && NORM_THREADS == 256 && NORM_WARPS == 8);
const _: () = assert!(DOWN_WARPS == 8);

/// The fault site every entry here raises.
pub const SITE: FaultSite = FaultSite::HcMix;

/// `1/streams`: the scale inside the silu and the sigmoid, and the mean's.
const INV: f32 = 1.0 / STREAMS as f32;
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
/// it and gets the same value (two barriers).
///
/// # Safety
/// `ws` points at [`NORM_WARPS`] f32 of this block's shared memory that
/// nothing else uses across the call.
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
    thread::sync_threads();
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
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
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
/// bottleneck in shared memory `lo_s` (`[M][RANK]`), then lane 0's mixed
/// value of each column (module doc).
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
    // Columns outer, streams inner: a column's bottleneck words live for that column only.
    let mut g = [[0.0f32; MAX_COLS]; STREAMS];
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
        for s in 0..STREAMS {
            thread::__unroll_config::<0>();
            g[s][c] = warp::reduce_sum_f32(f[s]);
        }
    }
    if lane != 0 {
        return;
    }
    for c in 0..M {
        thread::__unroll_config::<0>();
        let mut acc = 0.0f32;
        for s in 0..STREAMS {
            thread::__unroll_config::<0>();
            // SAFETY: (c·STREAMS + s)·hidden + h < M·STREAMS·hidden <= xn.len().
            let x = unsafe { *xn.get_unchecked((c * STREAMS + s) * hidden + h) };
            let sg = sigmoid(g[s][c]);
            acc = if s == 0 {
                x * sg
            } else {
                f32::mul_add(x, sg, acc)
            };
        }
        let out = acc * INV;
        if !out.is_finite() {
            fault.raise(SITE);
        }
        // SAFETY: c·hidden + h < M·hidden <= mixed.len(); this warp owns h.
        unsafe { *mixed.get_unchecked_mut(c * hidden + h) = out };
    }
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
        let tid = thread::threadIdx_x() as usize;
        let hid = hidden as usize;
        let base = (c * STREAMS + s) * hid;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let w = if combine == 1 {
            // SAFETY: c·4 + s < m·4 <= wgt.len().
            unsafe { *wgt.get_unchecked(c * STREAMS + s) }
        } else {
            0.0
        };
        let mut acc = 0.0f32;
        let mut i = tid;
        while i < hid {
            // SAFETY: base + i < m·4·hidden <= res.len(), c·hidden + i <
            // m·hidden <= y.len(); this block owns stream s of column c and
            // this thread value i of it.
            let x = unsafe {
                let r = res.get_unchecked_mut(base + i);
                if combine == 1 {
                    *r = f32::mul_add(w, *y.get_unchecked(c * hid + i), *r);
                } else if init == 1 {
                    *r = *y.get_unchecked(c * hid + i);
                }
                *r
            };
            acc = f32::mul_add(x, x, acc);
            i += NORM_THREADS;
        }
        // SAFETY: `ws` is NORM_WARPS f32 of this block's shared memory.
        let sum = unsafe { block_sum(acc, tid, ws) };
        if tid == 0 && !sum.is_finite() {
            fault.raise(SITE);
        }
        let r = 1.0 / (sum / hidden as f32 + eps).sqrt();
        let mut i = tid;
        while i < hid {
            // SAFETY: as above; s·hidden + i < 4·hidden <= gamma.len(). The
            // value was written by this thread in the pass above.
            unsafe {
                let x = *res.get_unchecked_mut(base + i);
                *xn.get_unchecked_mut(base + i) = (x * r) * *gamma.get_unchecked(s * hid + i);
            }
            i += NORM_THREADS;
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
        let ib = b - DOWN_Q8_BLOCKS;
        if inject == 0 || ib >= DOWN_INJ_BLOCKS {
            return; // block-uniform
        }
        let (o, s) = (ib / STREAMS, ib % STREAMS);
        let chunk = hid / DOWN_WARPS;
        let at = s * hid + wp * chunk;
        // The inject weight as rows of `chunk`: row o's chunk at `at`.
        let wrow = (o * STREAMS * hid + at) / chunk;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; every slot is written before the barrier that precedes
        // its reads.
        let ip = unsafe { SharedArray::as_raw_mut_ptr(&raw mut IP) };
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
        let cols = m as usize;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; written below, read after the barrier.
        let lo_s = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LO) };
        let mut e = tid;
        while e < cols * RANK {
            // SAFETY: e·4 + 3 < m·1280 <= dpart.len().
            let v = unsafe {
                let p = dpart.as_ptr().add(e * STREAMS);
                ((*p + *p.add(1)) + *p.add(2)) + *p.add(3)
            };
            let l = silu(v * INV);
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
            let w = 2.0 * sigmoid(v * INV);
            if !w.is_finite() {
                fault.raise(SITE);
            }
            // SAFETY: tid < m·4 <= wgt.len(); block 0 alone writes wgt.
            unsafe { *wgt.get_unchecked_mut(tid) = w };
        }
        thread::sync_threads();
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
/// (`streams·hidden` rows of `RANK`) in the q8f32 Q8_0 planes
/// (`crate::weights::q8_0_planes`), and the F32 inject (`streams` rows of
/// `streams·hidden`), `None` for the head.
pub struct SiteWeights<'a> {
    pub gamma: &'a DeviceBuffer<f32>,
    pub down_qs: &'a DeviceTensor<u32>,
    pub down_d: &'a DeviceTensor<u16>,
    pub up_qs: &'a DeviceTensor<u32>,
    pub up_d: &'a DeviceTensor<u16>,
    pub inject: Option<&'a DeviceTensor<f32>>,
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
/// them: `γ` and the inject F32, the down and the up Q8_0. Any other type is
/// refused by name — the kernels read no other.
///
/// # Errors
/// [`GpuError::Shape`] naming the tensor and its type.
pub fn check_types(
    gamma: GgmlType,
    down: GgmlType,
    up: GgmlType,
    inject: Option<GgmlType>,
) -> Result<(), GpuError> {
    let want = [
        ("hc norm (γ)", gamma, GgmlType::F32),
        ("hc down", down, GgmlType::Q8_0),
        ("hc up", up, GgmlType::Q8_0),
    ];
    for (name, got, ty) in want
        .into_iter()
        .chain(inject.map(|t| ("hc inject", t, GgmlType::F32)))
    {
        if got != ty {
            return Err(GpuError::shape(
                "hc_gated::check_types",
                format!("{name} is {got:?}; the gated-residual kernels read {ty:?} only"),
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
/// up `streams·hidden` rows of Q8_0 planes at the other's width, the inject
/// (when there is one) `streams` rows of `streams·hidden`.
fn check_mix(
    what: &'static str,
    (geo, z): (Geometry, Sizes),
    (res, y, mixed): (
        &DeviceBuffer<f32>,
        Option<&DeviceBuffer<f32>>,
        &DeviceBuffer<f32>,
    ),
    w: &SiteWeights<'_>,
) -> Result<(), GpuError> {
    let (wide, r) = (geo.wide(), geo.r());
    let lens = [
        ("res", res.len(), z.res),
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
    let planes = [
        ("down", w.down_qs, w.down_d, r, wide),
        ("up", w.up_qs, w.up_d, wide, r),
    ];
    for (name, qs, d, rows, k) in planes {
        if qs.rows() != rows || qs.cols() != k / 4 || d.rows() != rows || d.cols() != k / 32 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{name} planes are qs {}x{} and d {}x{}, want {rows}x{} and {rows}x{} \
                     ({rows} rows of {k} Q8_0 values)",
                    qs.rows(),
                    qs.cols(),
                    d.rows(),
                    d.cols(),
                    k / 4,
                    k / 32
                ),
            ));
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
        check_mix(what, (geo, geo.sizes(m)), (&*a.res, y, &*a.mixed), &a.w)?;
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
        let prep = self
            .module
            .prepare_hc_gated_down_4x320(LaunchConfig1D::new(
                launch_u32(what, "down grid", grid)?,
                DOWN_THREADS,
                0,
            ))?;
        let inj_buf = match a.w.inject {
            Some(t) => t.buf(),
            None => &s.wgt,
        };
        self.module.hc_gated_down_4x320(
            stream,
            &prep,
            a.w.down_qs.buf(),
            a.w.down_d.buf(),
            inj_buf,
            &s.xn,
            hidden,
            mu,
            inject,
            a.fault,
            &mut s.dpart,
            &mut s.ipart,
        )?;
        let prep = self
            .module
            .prepare_hc_gated_up_mix_4x320(LaunchConfig1D::new(
                launch_u32(what, "up grid", geo.d() / UP_WARPS)?,
                UP_THREADS,
                0,
            ))?;
        self.module.hc_gated_up_mix_4x320(
            stream,
            &prep,
            a.w.up_qs.buf(),
            a.w.up_d.buf(),
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

#[cuda_module]
mod hc_wide_kernels {
    use super::*;

    /// [`hc_kernels::hc_gated_norm_4`] over any number of columns: block `b`
    /// is stream `b % 4` of column `b / 4`, the same body.
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
        let b = thread::blockIdx_x() as usize;
        let (s, c) = (b % STREAMS, b / STREAMS);
        if c >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let hid = hidden as usize;
        let base = (c * STREAMS + s) * hid;
        // SAFETY: this block's own shared allocation, reached without a
        // reference; `block_sum` is its only user.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WS) };
        let w = if combine == 1 {
            // SAFETY: c·4 + s < m·4 <= wgt.len().
            unsafe { *wgt.get_unchecked(c * STREAMS + s) }
        } else {
            0.0
        };
        let mut acc = 0.0f32;
        let mut i = tid;
        while i < hid {
            // SAFETY: base + i < m·4·hidden <= res.len(), c·hidden + i <
            // m·hidden <= y.len(); this block owns stream s of column c and
            // this thread value i of it.
            let x = unsafe {
                let r = res.get_unchecked_mut(base + i);
                if combine == 1 {
                    *r = f32::mul_add(w, *y.get_unchecked(c * hid + i), *r);
                } else if init == 1 {
                    *r = *y.get_unchecked(c * hid + i);
                }
                *r
            };
            acc = f32::mul_add(x, x, acc);
            i += NORM_THREADS;
        }
        // SAFETY: `ws` is NORM_WARPS f32 of this block's shared memory.
        let sum = unsafe { block_sum(acc, tid, ws) };
        if tid == 0 && !sum.is_finite() {
            fault.raise(SITE);
        }
        let r = 1.0 / (sum / hidden as f32 + eps).sqrt();
        let mut i = tid;
        while i < hid {
            // SAFETY: base + i < m·4·hidden <= res.len(), xn.len(), and
            // s·hidden + i < 4·hidden <= gamma.len(), by the launch contract;
            // this thread wrote res[base + i] in the pass above and is the
            // only writer of xn[base + i].
            unsafe {
                let x = *res.get_unchecked_mut(base + i);
                *xn.get_unchecked_mut(base + i) = (x * r) * *gamma.get_unchecked(s * hid + i);
            }
            i += NORM_THREADS;
        }
    }

    /// [`hc_kernels::hc_gated_combine_4`] over any number of columns: value
    /// `i` of `res` becomes `fma(wgt[c·4 + s], y[c][i % hidden], res[i])`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            y.len() >= m * hidden,
            wgt.len() >= m * 4,
            res.len() >= m * 4 * hidden
        )
    )]
    pub fn hc_gated_combine_4w(
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

    /// The bottleneck and the combine's weights of `m` columns: thread `i <
    /// m·320` writes `lo[i] = silu(down[i]·¼)`, `down` the down rows'
    /// whole dots `[m][320]`; with `inject` 1, thread `m·320 + j` writes
    /// `wgt[j] = 2·σ(inj[j]·¼)`, `inj` the inject rows' dots `[m][4]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            inject <= 1,
            down.len() >= m * 320,
            inj.len() >= inject * m * 4,
            lo.len() >= m * 320,
            wgt.len() >= inject * m * 4
        )
    )]
    pub fn hc_gated_lo_4x320(
        down: &[f32],
        inj: &[f32],
        m: u32,
        inject: u32,
        fault: FaultSink,
        mut lo: DisjointSlice<f32>,
        mut wgt: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let n = m as usize * RANK;
        if i < n {
            // SAFETY: the `requires` above were launcher-checked, `n = m·320`
            // comes from this kernel's parameter and `i` is this thread's
            // `index_1d` below it, and the launch is `domain = 1` with the
            // contract's exact 1-D block.
            let e = unsafe { Elem::new(down, &mut lo, i) };
            e.map(move |d| {
                let l = silu(d * INV);
                if !l.is_finite() {
                    fault.raise(SITE);
                }
                l
            });
            return;
        }
        let j = i - n;
        if inject == 0 || j >= m as usize * STREAMS {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked (with `inject`
        // 1 they read `inj.len()`, `wgt.len() >= m·4`), `j` is this thread's
        // `index_1d` less the grid-uniform `n`, below `m·4` past the guard,
        // and the launch is `domain = 1` with the contract's exact 1-D block.
        let e = unsafe { Elem::new(inj, &mut wgt, j) };
        e.map(move |v| {
            let w = 2.0 * sigmoid(v * INV);
            if !w.is_finite() {
                fault.raise(SITE);
            }
            w
        });
    }

    /// The mixed values of `m` columns: value `i = c·hidden + h` is `(xn_0·
    /// σ(g_0), then fma(xn_s, σ(g_s), ·) for s = 1, 2, 3)·¼`, `xn_s` and
    /// `g_s` value `s·hidden + h` of column `c` of the normed streams and of
    /// the up's rows (`[m][4·hidden]` both).
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
        let hid = hidden as usize;
        if i >= m as usize * hid {
            return;
        }
        let (c, h) = (i / hid, i % hid);
        let mut acc = 0.0f32;
        for s in 0..STREAMS {
            thread::__unroll_config::<0>();
            let at = (c * STREAMS + s) * hid + h;
            // SAFETY: at < m·4·hidden <= xn.len(), g.len().
            let (x, gv) = unsafe { (*xn.get_unchecked(at), *g.get_unchecked(at)) };
            let sg = sigmoid(gv);
            acc = if s == 0 {
                x * sg
            } else {
                f32::mul_add(x, sg, acc)
            };
        }
        let out = acc * INV;
        if !out.is_finite() {
            fault.raise(SITE);
        }
        // SAFETY: i < m·hidden <= mixed.len(); one thread a value.
        unsafe { *mixed.get_unchecked_mut(i) = out };
    }
}

/// The intermediates of a wide mix, allocated once for up to `cols` columns
/// and shared by every site of a walk, as [`HcScratch`] is: the normed
/// streams and their 32-value activations, the down's and the inject's whole
/// dots, the bottleneck and its activations, the up's rows, and the last
/// inject site's combine weights.
pub struct HcWideScratch {
    /// `[m][streams][hidden]`.
    pub xn: DeviceBuffer<f32>,
    pub xq: GemmAct32,
    /// `[m][RANK]`: the down rows' dots.
    pub down: DeviceBuffer<f32>,
    /// `[m][streams]`: the inject rows' dots.
    pub inj: DeviceBuffer<f32>,
    /// `[m][RANK]`: the bottleneck after its silu.
    pub lo: DeviceBuffer<f32>,
    pub loq: GemmAct32,
    /// `[m][streams·hidden]`: the up's rows.
    pub g: DeviceBuffer<f32>,
    /// `[m][streams]`: the last inject site's combine weights.
    pub wgt: DeviceBuffer<f32>,
    cols: usize,
    geo: Geometry,
}

impl HcWideScratch {
    /// Allocate for `geo` at `cols` (1..=`GEMM_MAX_SLOTS`) columns. Load-time
    /// only.
    pub fn new(stream: &CudaStream, geo: Geometry, cols: usize) -> Result<HcWideScratch, GpuError> {
        let z = geo.sizes(cols);
        Ok(HcWideScratch {
            xn: DeviceBuffer::zeroed(stream, z.xn)?,
            xq: GemmAct32::new(stream, cols, geo.wide())?,
            down: DeviceBuffer::zeroed(stream, z.lo)?,
            inj: DeviceBuffer::zeroed(stream, z.wgt)?,
            lo: DeviceBuffer::zeroed(stream, z.lo)?,
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
        [
            &self.xn, &self.down, &self.inj, &self.lo, &self.g, &self.wgt,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.xq.bytes()
            + self.loq.bytes()
    }
}

/// [`HcWideKernels::enqueue_mix`]'s arguments: [`MixArgs`]' over `m`
/// columns of the wide scratch, and the GEMM family with the one-expert
/// table its down and up read — filled for exactly `m` slots
/// (`GemmKernels::enqueue_route_dense`, refused by name otherwise), which
/// the caller owns.
pub struct WideMixArgs<'a> {
    pub res: &'a mut DeviceBuffer<f32>,
    pub before: Before<'a>,
    pub w: SiteWeights<'a>,
    pub eps: f32,
    pub m: usize,
    pub fault: FaultSink,
    pub scratch: &'a mut HcWideScratch,
    pub gemm: &'a Gemm32Kernels,
    pub dense: &'a GemmRoute,
    pub mixed: &'a mut DeviceBuffer<f32>,
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
    /// norm after `before`, `xn` quantized, the down GEMM, the inject tile at
    /// an inject site, the bottleneck and weights, the bottleneck quantized,
    /// the up GEMM, the mixed values — seven launches, eight at an inject
    /// site, in order on `stream`. An inject site leaves its combine weights
    /// in the scratch for the next site's [`Before::Combine`]. `m` outside
    /// `1..=scratch.cols()`, a short buffer, planes of another shape
    /// (the narrow arm's check, `check_mix`) or a table filled for another
    /// count are refused by name before any launch. Asynchronous,
    /// allocation-free.
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
        check_mix(what, (geo, z), (&*a.res, y, &*a.mixed), &a.w)?;
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
        let s = &mut *a.scratch;
        let prep = self.module.prepare_hc_gated_norm_4w(LaunchConfig1D::new(
            launch_u32(what, "norm grid", STREAMS * m)?,
            NORM_THREADS as u32,
            0,
        ))?;
        // Without a combine or an init the norm reads neither `y` nor `wgt`:
        // `γ` stands in.
        self.module.hc_gated_norm_4w(
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
        a.gemm
            .enqueue_quantize_gemm32(stream, &s.xn, m, &mut s.xq, a.fault)?;
        a.gemm.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: Gemm32Weight::Q8_0Plane {
                    qs: a.w.down_qs,
                    d: a.w.down_d,
                },
                rows_per_expert: r,
                act: &s.xq,
                route: a.dense,
                input: GemmInput::PerSlot,
                y: &mut s.down,
            },
        )?;
        if let Some(inj) = a.w.inject {
            a.gemm.enqueue_f32_tile(stream, inj, &s.xn, m, &mut s.inj)?;
        }
        let n_lo = m * r + if inject == 1 { m * STREAMS } else { 0 };
        let prep = self.module.prepare_hc_gated_lo_4x320(LaunchConfig1D::new(
            launch_u32(what, "lo grid", n_lo.div_ceil(GLUE_THREADS as usize))?,
            GLUE_THREADS,
            0,
        ))?;
        // Without an inject the launch reads no `inj` and writes no `wgt`:
        // the scratch's own buffers stand in, their lengths untested.
        self.module.hc_gated_lo_4x320(
            stream, &prep, &s.down, &s.inj, mu, inject, a.fault, &mut s.lo, &mut s.wgt,
        )?;
        a.gemm
            .enqueue_quantize_gemm32(stream, &s.lo, m, &mut s.loq, a.fault)?;
        a.gemm.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: Gemm32Weight::Q8_0Plane {
                    qs: a.w.up_qs,
                    d: a.w.up_d,
                },
                rows_per_expert: wide,
                act: &s.loq,
                route: a.dense,
                input: GemmInput::PerSlot,
                y: &mut s.g,
            },
        )?;
        let prep = self.module.prepare_hc_gated_mix_4(LaunchConfig1D::new(
            launch_u32(what, "mix grid", z.mixed.div_ceil(GLUE_THREADS as usize))?,
            GLUE_THREADS,
            0,
        ))?;
        self.module
            .hc_gated_mix_4(stream, &prep, &s.xn, &s.g, hidden, mu, a.fault, a.mixed)?;
        Ok(())
    }

    /// Enqueue the combine alone over `m` columns with the wide scratch's
    /// weights (the last inject site's): `res += wgt · y` per stream. One
    /// launch. Asynchronous, allocation-free.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        res: &mut DeviceBuffer<f32>,
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
        for (name, got, need) in [("res", res.len(), z.res), ("y", y.len(), z.y)] {
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
            launch_u32(what, "hidden", geo.d())?,
            launch_u32(what, "m", m)?,
            fault,
            res,
        )?;
        Ok(())
    }
}
