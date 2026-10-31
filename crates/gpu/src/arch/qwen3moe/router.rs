//! The gated softmax router, Qwen3.6's and Qwen3.8's: the router of an MoE
//! block whose shared expert runs as one more routed slot ([`gated`]). It is
//! the numeric contract of [`crate::route_softmax`] over `n` experts plus the
//! shared expert's sigmoid gate as the router weight's last row, and it takes
//! its buffers ([`RouterBufs`](crate::route_softmax::RouterBufs)), its
//! norm-fused row body ([`norm_row_body`](crate::route_softmax::norm_row_body))
//! and its launch bounds from there.

use crate::fault::{FaultSink, FaultSite};
use crate::gemm::{NARROW_ROWS, bf16_mma_narrow};
use crate::q8f32::{
    Bf16Words, RowValues, f32_lane_partial_1col_w32, f32_lane_partials, gemv_lane_sums,
};
use crate::route_core::{sigmoid, take};
use crate::route_softmax::{FUSED_WARPS, MAX_TOKENS};
use crate::store_cols;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, thread, threadfence, warp};

/// `(v, i)` becomes the running best `(bv, bi)` when it beats it under
/// [`take`]`::<false>`.
#[inline(always)]
fn keep(v: f32, i: u32, bv: &mut f32, bi: &mut u32) {
    if take::<false>(v, i, *bv, *bi) {
        *bv = v;
        *bi = i;
    }
}

/// A probability `p` as a rescan candidate: itself when its bit `free` is set,
/// else `−inf`, which [`take`]`::<false>` never takes over the seed `(−inf, 0)`
/// or a probability. A select, not a branch, so a walk's loads issue together.
#[inline(always)]
fn untaken(free: u32, p: f32) -> f32 {
    if free != 0 { p } else { f32::NEG_INFINITY }
}

/// One token's routing by one warp for a router whose last row is a shared
/// expert's gate ([`gated`]'s contract): `32 · PER_LANE` experts, lane `L`
/// owning experts `L + 32 j`. On entry `p` holds the token's expert logits,
/// entry `e` expert `e`'s, each written by the lane that owns it, and every
/// lane brings the gate's logit in `g`. The warp turns `p` into the
/// probabilities in place and writes `probs[t·n + e]` for every expert
/// `e < n`, then per slot `ids[t·(used + 1) + s]` and
/// `weights[t·(used + 1) + s]`: slots `s < used` in rank order as
/// [`crate::route_softmax`]'s routing warp writes them (one pick a lane),
/// slot `used` the shared expert, `(n, sigmoid(g))`. The logits live in
/// shared memory rather than in a register array: each lane walks its
/// `PER_LANE` entries in loops that a register array would put in a local
/// depot.
///
/// A token with a non-finite logit among the `n + 1` raises
/// [`FaultSite::Router`] on `fault`, gets NaN probabilities and weights in
/// every slot, and keeps its ids; the warp returns before the softmax.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same `g`, `p`,
/// `used` and `t`; `p` points to `32 · PER_LANE` entries of the block's
/// shared memory that no other warp touches, written as above; `1 <= used
/// <= 32`; `probs.len() >= (t + 1)·n`, and `ids.len()`,
/// `weights.len() >= (t + 1)·(used + 1)`, those slots written by this warp
/// alone; `PER_LANE` a multiple of 4, at most 32.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn route_warp_gated<const PER_LANE: usize>(
    g: f32,
    p: *mut f32,
    used: usize,
    t: usize,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let n = 32 * PER_LANE;
    let slots = used + 1;
    let lane = warp::lane_id() as usize;
    let pb = t * n + lane;
    let mut finite = g.is_finite();
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: lane + 32 j < n, this lane's own entry (this fn's
        // contract).
        let x = unsafe { *p.add(lane + 32 * j) };
        // `&`, not `&&`: every test runs.
        finite &= x.is_finite();
        j += 1;
    }
    // Warp-uniform: every lane leaves the ballot with the same mask.
    if warp::ballot(!finite) != 0 {
        let mut j = 0usize;
        while j < PER_LANE {
            // SAFETY: lane + 32 j < n — the token's probs slots (this fn's
            // contract); entries `lane + 32 j` are this lane's own.
            unsafe { *probs.get_unchecked_mut(pb + 32 * j) = f32::NAN };
            j += 1;
        }
        if lane == 0 {
            fault.raise(FaultSite::Router);
            let mut s = 0usize;
            while s < slots {
                // SAFETY: s < used + 1, inside the token's weights slots
                // (this fn's contract); lane 0 is the only writer.
                unsafe { *weights.get_unchecked_mut(t * slots + s) = f32::NAN };
                s += 1;
            }
        }
        return;
    }
    let mut mx = f32::NEG_INFINITY;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: as in the finiteness walk.
        mx = mx.max(unsafe { *p.add(lane + 32 * j) });
        j += 1;
    }
    let mut off = 16u32;
    while off > 0 {
        mx = mx.max(warp::shuffle_xor_f32(mx, off));
        off >>= 1;
    }
    // Each entry becomes its exp, summed in f64 in ascending `j`.
    let mut sum = 0.0f64;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: as in the finiteness walk; this lane is the entry's only
        // reader and writer.
        unsafe {
            let at = p.add(lane + 32 * j);
            let e = (*at - mx).exp();
            *at = e;
            sum += f64::from(e);
        }
        j += 1;
    }
    let mut off = 16u32;
    while off > 0 {
        // Both lanes of a pair add the same two values, so every lane leaves
        // with the same sum.
        sum += warp::shuffle_xor_f64(sum, off);
        off >>= 1;
    }
    let inv = sum as f32;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: lane + 32 j < n — inside the token's shared entries and
        // its probs slots (this fn's contract); entries `lane + 32 j` are
        // this lane's own.
        unsafe {
            let at = p.add(lane + 32 * j);
            let pj = *at / inv;
            *at = pj;
            *probs.get_unchecked_mut(pb + 32 * j) = pj;
        }
        j += 1;
    }
    // Every lane reads each round's winner, whichever lane wrote it.
    warp::sync_mask(u32::MAX);

    let mut taken = 0u32;
    // The chosen probabilities' f64 sum in slot order (the same on every
    // lane), and this lane's own slot's probability.
    let mut chosen = 0.0f64;
    let mut mine = 0.0f32;
    let mut s = 0usize;
    while s < used {
        // Four running bests, one per entry `j` mod 4, merged after the walk:
        // `take` is a total order, so the merge picks the entry one running
        // best picks, on chains a quarter as long.
        let (mut v0, mut i0) = (f32::NEG_INFINITY, 0u32);
        let (mut v1, mut i1) = (f32::NEG_INFINITY, 0u32);
        let (mut v2, mut i2) = (f32::NEG_INFINITY, 0u32);
        let (mut v3, mut i3) = (f32::NEG_INFINITY, 0u32);
        let mut j = 0usize;
        while j < PER_LANE {
            let e = lane + 32 * j;
            // SAFETY: e + 96 < 32·PER_LANE = n (j + 3 < PER_LANE, a multiple
            // of 4), this lane's own entries.
            let (a, b, c, d) =
                unsafe { (*p.add(e), *p.add(e + 32), *p.add(e + 64), *p.add(e + 96)) };
            let free = !taken >> j;
            keep(untaken(free & 1, a), e as u32, &mut v0, &mut i0);
            keep(untaken(free & 2, b), (e + 32) as u32, &mut v1, &mut i1);
            keep(untaken(free & 4, c), (e + 64) as u32, &mut v2, &mut i2);
            keep(untaken(free & 8, d), (e + 96) as u32, &mut v3, &mut i3);
            j += 4;
        }
        keep(v1, i1, &mut v0, &mut i0);
        keep(v3, i3, &mut v2, &mut i2);
        let (mut bv, mut bi) = (v0, i0);
        keep(v2, i2, &mut bv, &mut bi);
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
            if take::<false>(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
            off >>= 1;
        }
        // Every lane leaves the butterfly with the same winner; its owner
        // marks it taken.
        if bi as usize % 32 == lane {
            taken |= 1 << (bi as usize / 32);
        }
        // SAFETY: bi < n — inside the token's entries, published by the warp
        // sync above and not written since. The probability is read from the
        // winner's entry, as `route_warp` reads it.
        let pw = unsafe { *p.add(bi as usize) };
        chosen += f64::from(pw);
        if lane == s {
            mine = pw;
        }
        if lane == 0 {
            // SAFETY: s < used, inside the token's ids slots (this fn's
            // contract); lane 0 is the only writer.
            unsafe { *ids.get_unchecked_mut(t * slots + s) = bi };
        }
        s += 1;
    }
    let chosen = chosen as f32;
    if lane < used {
        // SAFETY: lane < used, inside the token's weights slots (this fn's
        // contract); lane `lane` is slot `lane`'s only writer.
        unsafe { *weights.get_unchecked_mut(t * slots + lane) = mine / chosen };
    }
    if lane == 0 {
        // SAFETY: slot `used` is the token's last ids and weights slot (this
        // fn's contract); lane 0 is its only writer.
        unsafe {
            *ids.get_unchecked_mut(t * slots + used) = n as u32;
            *weights.get_unchecked_mut(t * slots + used) = sigmoid(g);
        }
    }
}

/// The fused entries' tail of [`crate::route_softmax`] (`publish_route`) for a
/// gated router of `32 · PER_LANE` experts and `used` routed slots: token
/// `t`'s `n + 1` logits at `logits[t·(n + 1) ..]`, the last the gate's, and
/// [`route_warp_gated`] in place of the plain routing warp.
///
/// # Safety
///
/// `publish_route`'s, with `p` holding `m · n` entries, `logits` `(n + 1)
/// · m`, `probs` `n · m`, and `ids`, `weights` `(used + 1) · m`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn publish_route_gated<const PER_LANE: usize>(
    m: usize,
    used: usize,
    last: *mut u32,
    p: *mut f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
) {
    threadfence();
    thread::sync_threads();

    let n = 32 * PER_LANE;
    let tid = thread::threadIdx_x() as usize;
    let wi = tid / 32;
    let lane = warp::lane_id() as usize;
    let ticket = done.as_mut_ptr();
    if tid == 0 {
        // SAFETY: `ticket` is done[0] (this fn's contract); every access to
        // it is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
        let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
        if is_last {
            // Every block has drawn its ticket.
            count.store(0, AtomicOrdering::Relaxed);
        }
        // SAFETY: block-shared, one element, written before the barrier
        // that publishes it.
        unsafe { *last = u32::from(is_last) };
    }
    thread::sync_threads();
    // SAFETY: block-shared, one element, written by thread 0 before the
    // barrier above.
    if unsafe { *last } == 0 || wi >= m {
        return;
    }

    // The last block, warp `wi` routing token `wi`: every block's rows are
    // published. Volatile loads, as `publish_route` reads them, each lane's
    // logits into its own entries of token wi's `p`.
    let base = wi * (n + 1);
    let lp = logits.as_mut_ptr();
    let pt = p.wrapping_add(wi * n);
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: base + lane + 32 j < base + n < m·(n + 1) <= logits.len(),
        // and wi·n + lane + 32 j < m·n, inside `p` (this fn's contract); the
        // entry is this lane's own.
        unsafe { *pt.add(lane + 32 * j) = core::ptr::read_volatile(lp.add(base + lane + 32 * j)) };
        j += 1;
    }
    // SAFETY: base + n < m·(n + 1) <= logits.len(); every lane reads the
    // gate's logit.
    let g = unsafe { core::ptr::read_volatile(lp.add(base + n)) };
    // SAFETY: wi < m, so token wi's n entries are inside `p` and no other
    // warp's, its expert logits written above by their lanes; the whole warp
    // is here (`wi` is warp-uniform), converged past the barrier; 1 <= used
    // <= 32 and its slots are inside probs, ids and weights (this fn's
    // contract).
    unsafe {
        route_warp_gated::<PER_LANE>(g, pt, used, wi, probs, ids, weights, fault);
    }
}

/// The gated router's fused launch body, `qwen3moe_router_fused`'s at `n + 1
/// = 32 · PER_LANE + 1` rows: warp 0 of block `r` computes row `r`'s logit of
/// each of the `m_cols` tokens with `f32_gemv`'s row body and stores it at
/// `logits[c·(n + 1) + r]`, then [`publish_route_gated`]. `w` is an F32
/// weight's values or a BF16 weight's words
/// ([`crate::q8f32::Bf16Words`]), whose logits are the F32 walk's over the
/// widened values bit for bit.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the grid is `n + 1`
/// blocks of [`FUSED_THREADS`](crate::route_softmax::FUSED_THREADS); `last` and `p` are 1 and `MAX_TOKENS · n`
/// entries of the block's shared memory that nothing else touches; `1 <=
/// m_cols <= MAX_TOKENS`, `1 <= used <= 32`, `k` a positive multiple of 32,
/// `w` holds at least `(n + 1)·k` values, `x.len() >= m_cols·k`, the outputs
/// sized as [`publish_route_gated`] takes them for `m_cols` tokens, and
/// `done[0]` zero before the launch.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gated_fused_body<const PER_LANE: usize, W: RowValues>(
    w: W,
    x: &[f32],
    k: u32,
    m_cols: u32,
    used: u32,
    last: *mut u32,
    p: *mut f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
) {
    let rows = 32 * PER_LANE + 1;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wi = tid / 32;
    let m = m_cols as usize;
    // The row guard is warp-uniform, so the warp tree sees a full warp; the
    // other warps, and a block past the rows, still meet the barrier and the
    // ticket.
    let row = thread::blockIdx_x() as usize;
    if wi == 0 && row < rows {
        // SAFETY: row < rows, so w holds rows·k >= (row + 1)·k values;
        // x.len() >= m_cols·k, k a positive multiple of 32, m_cols in 1..=8
        // (this fn's contract); the whole warp is here, converged.
        let sums = unsafe { fused_row_sums(w, x, k, row, m_cols, lane) };
        if lane == 0 {
            // SAFETY: row < rows and 1 <= m <= 8, so the slots c·rows + row
            // (c < m) lie inside logits (this fn's contract); lane 0 of the
            // row's warp is their only writer.
            unsafe { store_cols(logits, row, rows, m, sums) };
        }
    }
    // SAFETY: this fn's contract is publish_route_gated's for m tokens.
    unsafe {
        publish_route_gated::<PER_LANE>(
            m,
            used as usize,
            last,
            p,
            logits,
            probs,
            ids,
            weights,
            done,
            fault,
        );
    }
}

/// Row `row`'s logit of each of the first `m_cols` tokens by one warp, the
/// fused entries' row body: `f32_gemv`'s lane sums (at one token through
/// [`f32_lane_partial_1col_w32`], which keeps 32 chunks' loads in flight)
/// and its warp tree. Every lane returns the sums; columns past `m_cols` are
/// 0.0.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same arguments but
/// `lane`; `w` holds at least `(row + 1)·k` values, `x.len() >= m_cols·k`,
/// `k` a positive multiple of 32, `m_cols` in 1..=8.
#[inline(always)]
unsafe fn fused_row_sums<W: RowValues>(
    w: W,
    x: &[f32],
    k: u32,
    row: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    let partials = if m_cols == 1 {
        // SAFETY: m_cols == 1, so this fn's contract gives w (row + 1)·k values, x.len() >= k, k a
        // positive multiple of 32, and lane < 32.
        let f0 = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
        [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    } else {
        // SAFETY: this fn's contract gives w (row + 1)·k values, x.len() >= m_cols·k, k a positive
        // multiple of 32, m_cols in 1..=8 and lane < 32.
        unsafe { f32_lane_partials(w, x, k, row, m_cols, lane) }
    };
    gemv_lane_sums(partials, m_cols)
}

/// [`gated_fused_body`] on the dense MMA tile's arithmetic, for a BF16
/// weight's words on a device whose ubatch logits run that tile: block `b`
/// computes rows `16·b ..` of every token as [`bf16_mma_narrow`] does, so
/// each logit is the ubatch launch's bits for its token. A cell the narrow
/// block leaves to the exact owner (its row or its token holds an operand
/// outside the tensor cores' range) takes the fused entry's row sum
/// ([`fused_row_sums`], `f32_gemv`'s bits over the widened weight), the
/// exact tile's bits that the ubatch launch's replay leaves: warp `w` sums
/// the block's rows `2w` and `2w + 1` when one of their cells replays. Then
/// [`publish_route_gated`], over the stages' memory.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the grid is `⌈(n + 1) /
/// 16⌉` blocks of [`FUSED_THREADS`](crate::route_softmax::FUSED_THREADS); `last` is one block-shared word and
/// `sh` the narrow block's `gemm::NARROW_WORDS` 16-byte aligned shared words,
/// which nothing else touches; `1 <= m_cols <= MAX_TOKENS`, `1 <= used <=
/// 32`, `k` a positive multiple of 32; `w` holds `(n + 1)·k / 2` words and
/// `x` `m_cols·k` values, both 16-byte aligned; the outputs sized as
/// [`publish_route_gated`] takes them for `m_cols` tokens, and `done[0]`
/// zero before the launch.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
#[inline(always)]
unsafe fn gated_mma_fused_body<const PER_LANE: usize>(
    w: &[u32],
    x: &[f32],
    k: u32,
    m_cols: u32,
    used: u32,
    last: *mut u32,
    sh: *mut u32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
) {
    let rows = 32 * PER_LANE + 1;
    let m = m_cols as usize;
    let row0 = thread::blockIdx_x() as usize * NARROW_ROWS;
    let wi = thread::threadIdx_x() as usize / 32;
    let lane = warp::lane_id() as usize;
    // SAFETY: all FUSED_THREADS = 256 threads are here, converged; row0 < rows (the grid's last
    // block starts below rows), n = m_cols <= MAX_TOKENS = NARROW_COLS, k a positive multiple of
    // 32, w holds rows·k/2 words and x m_cols·k values, both 16-byte aligned, and sh is the
    // block's NARROW_WORDS aligned shared words, by this fn's contract.
    let tile = unsafe { bf16_mma_narrow(w, x, (rows, k as usize, m), row0, sh) };
    if wi == 0 {
        // SAFETY: warp 0; logits holds rows·m_cols outputs and this block
        // alone writes its rows.
        unsafe { tile.store(logits, (rows, m), row0) };
    }
    let mut i = 0usize;
    while i < NARROW_ROWS / FUSED_WARPS {
        let r = (NARROW_ROWS / FUSED_WARPS) * wi + i;
        // Warp-uniform: the replay flags are the block's.
        if row0 + r < rows && tile.replays_row(r, m) {
            // SAFETY: row row0 + r < rows, so the words hold its k values;
            // x.len() >= m_cols·k, k a positive multiple of 32, m_cols in
            // 1..=8; the whole warp is here, converged.
            let sums = unsafe { fused_row_sums(Bf16Words(w), x, k, row0 + r, m_cols, lane) };
            if lane == 0 {
                // Unrolled, so each column reads `sums` at a constant index.
                for c in 0..MAX_TOKENS {
                    thread::__unroll_config::<0>();
                    if c < m && tile.replays(r, c) {
                        // SAFETY: column c < m_cols and row row0 + r < rows lie
                        // inside logits; lane 0 of the row's warp is the
                        // replayed cell's only writer.
                        unsafe { *logits.get_unchecked_mut(c * rows + row0 + r) = sums[c] };
                    }
                }
            }
        }
        i += 1;
    }
    // SAFETY: the stages' memory holds NARROW_WORDS >= MAX_TOKENS · n words
    // (const assert beside the entry), and nothing reads it after the narrow
    // block's last barrier; the rest is this fn's contract for m tokens.
    unsafe {
        publish_route_gated::<PER_LANE>(
            m,
            used as usize,
            last,
            sh.cast::<f32>(),
            logits,
            probs,
            ids,
            weights,
            done,
            fault,
        );
    }
}

/// The routing of `n_tok` tokens from their logits (`n_tok · (n + 1)`, as
/// [`f32_tile_body`](crate::gemm::f32_tile_body) writes them at `n + 1`
/// rows): warp `w` of block `b` routes token `FUSED_WARPS·b + w` with
/// [`route_warp_gated`].
///
/// # Safety
///
/// Every thread of the block calls it; the block is [`FUSED_THREADS`](crate::route_softmax::FUSED_THREADS) wide;
/// `p` is the block's own `FUSED_WARPS · n` shared entries; `1 <= used <=
/// 32`; `logits.len() >= (n + 1)·n_tok`, `probs.len() >= n·n_tok`,
/// `ids.len()`, `weights.len() >= (used + 1)·n_tok`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gated_route_body<const PER_LANE: usize>(
    logits: &[f32],
    n_tok: u32,
    used: u32,
    p: *mut f32,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let n = 32 * PER_LANE;
    let wi = thread::threadIdx_x() as usize / 32;
    let t = thread::blockIdx_x() as usize * FUSED_WARPS + wi;
    if t >= n_tok as usize {
        return; // warp-uniform
    }
    let lane = warp::lane_id() as usize;
    let base = t * (n + 1);
    let pt = p.wrapping_add(wi * n);
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: base + lane + 32 j < base + n < (t + 1)·(n + 1) <=
        // logits.len(), and wi·n + lane + 32 j < FUSED_WARPS·n, inside `p`
        // (this fn's contract); the entry is this lane's own.
        unsafe { *pt.add(lane + 32 * j) = *logits.get_unchecked(base + lane + 32 * j) };
        j += 1;
    }
    // SAFETY: base + n < (t + 1)·(n + 1) <= logits.len().
    let g = unsafe { *logits.get_unchecked(base + n) };
    // SAFETY: warp wi < FUSED_WARPS takes entries wi·n .. of the block's
    // shared scratch, which no other warp touches, its expert logits written
    // above by their lanes; the whole warp is here (the guard is
    // warp-uniform); 1 <= used <= 32, and token t's slots are inside probs,
    // ids and weights (this fn's contract), written by this warp alone.
    unsafe {
        route_warp_gated::<PER_LANE>(g, pt, used as usize, t, probs, ids, weights, fault);
    }
}

/// The router of an MoE block whose shared expert runs as one more routed
/// slot (Qwen3.6, Qwen3.8: `n` experts, the top `k`, and a shared expert
/// behind a sigmoid gate).
///
/// The router weight is the file's `ffn_gate_inp` with the shared expert's
/// gate `ffn_gate_inp_shexp` joined below it as row `n` (`n + 1` rows of the
/// hidden width), and the expert stacks carry the shared expert as expert
/// `n`. Each token gets `k + 1` slots: slots `0 .. k` are the numeric
/// contract of [`crate::route_softmax`] over the `n` experts' logits, op for
/// op (softmax, the top `k`, the weights renormalized over them); slot `k` is
/// `(n, sigmoid(logit_n))` — `route_core::sigmoid`, not renormalized. So the
/// combine's slot-ascending sum over a token's slots is `((Σ_{s < k}
/// w_s·d_s) + w_sh·d_sh) + resid`. `k` is a launch argument of one to 32,
/// read from the buffers' [`RouterDims`](crate::route_softmax::RouterDims);
/// `n` is the instance's.
///
/// Two instances of the const-generic bodies: [`ROW`](gated::ROW) (256
/// experts) with five entries — `qwen35moe_router_fused` (up to `MAX_TOKENS`
/// tokens, one block per row), `qwen35moe_router_norm` (one token, the FFN
/// norm folded in), `qwen35moe_router_norm_m` (up to `MAX_TOKENS` tokens, the
/// FFN norm folded in), `qwen35moe_router_logits` (a ubatch's logits, `⌈(n +
/// 1) / 32⌉` row blocks per 32 tokens, the last one's rows past `n + 1`
/// staged as the last row and not stored) and `qwen35moe_router_route` (the
/// routing alone) — and [`ROW_WIDE`](gated::ROW_WIDE) (512 experts) with the
/// three of them that are not norm-fused, `qwen35moe_router_*_512`, whose
/// router reads the hyper-connection mix with no norm in front of it. The
/// launchers pick the entry from the buffers' instance. A BF16 weight at the wide instance runs
/// `qwen35moe_router_{fused,logits}_512_bf16`, the F32 entries' bits over
/// its values widened; on a device whose dense prompt products run on BF16
/// tensor cores ([`crate::dense_mma`]) it runs the `_bf16_mma` pair instead,
/// both on the dense MMA tile's arithmetic (activations rounded once to BF16,
/// F32 accumulation), so there too the two ubatch launches leave each token
/// the fused launch's bits. A token with a non-finite value among its `n + 1`
/// logits raises [`FaultSite::Router`], gets NaN probabilities and NaN
/// weights in all `k + 1` slots, and keeps its ids.
pub mod gated {
    use super::{gated_fused_body, gated_mma_fused_body, gated_route_body, publish_route_gated};
    use crate::elem::RMS_WARPS;
    use crate::fault::FaultSink;
    use crate::gemm::{
        Bf16Rows, F32_TILE_LINE, F32Cols, F32Rows, F32T_THREADS_U32, MMA_THREADS, NARROW_ROWS,
        NARROW_WORDS, STAGE_FLOATS, STAGES, bf16_mma_body, f32_tile_body, mma_launch, tile_launch,
    };
    use crate::q8f32::Bf16Words;
    use crate::route_softmax::{
        FUSED_THREADS_U32, FUSED_WARPS, MAX_TOKENS, NORM_K, RouterBufs, norm_row_body,
    };
    use crate::tensor::{DeviceTensor, Q8Act, check_alignment};
    use crate::weights::{DevWeight, Weights};
    use crate::{GpuError, launch_u32};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
    use cuda_device::vector::U32x4;
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract};
    use cuda_host::cuda_module;
    use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
    use std::sync::Arc;

    /// Experts each lane of a routing warp owns (`lane + 32 j`) at the
    /// narrow and the wide instance.
    const PER_LANE: usize = 8;
    const PER_LANE_WIDE: usize = 16;
    /// Experts each instance softmaxes over, and the rows of its joined
    /// router weight (the logits a token): the experts, then the shared gate.
    const EXPERTS: usize = 32 * PER_LANE;
    const ROWS: usize = EXPERTS + 1;
    const EXPERTS_WIDE: usize = 32 * PER_LANE_WIDE;
    const ROWS_WIDE: usize = EXPERTS_WIDE + 1;
    // The entries' launch contracts spell these four out: a `requires` takes integer literals, not constants.
    const _: () = assert!(EXPERTS == 256);
    const _: () = assert!(ROWS == 257);
    const _: () = assert!(EXPERTS_WIDE == 512);
    const _: () = assert!(ROWS_WIDE == 513);

    /// The narrow instance's row: 256 experts, `k` one to 32.
    pub const ROW: RouterInst = shape::router_row(RouterBody::Qwen35moe, 32 * PER_LANE as u32);
    /// The wide instance's row: 512 experts, `k` one to 32.
    pub const ROW_WIDE: RouterInst =
        shape::router_row(RouterBody::Qwen35moe, 32 * PER_LANE_WIDE as u32);
    const _: () = assert!(
        shape::router_row_is(
            ROW,
            rules::SOFTMAX_NORM_GATED,
            false,
            (1, shape::LANE_PICKS)
        ) && shape::router_row_is(
            ROW_WIDE,
            rules::SOFTMAX_NORM_GATED,
            false,
            (1, shape::LANE_PICKS)
        )
    );
    // Each lane's taken experts are the bits of one u32, one per `j`, and its
    // rescan walks them four at a time.
    const _: () = assert!(
        PER_LANE <= 32
            && PER_LANE_WIDE <= 32
            && PER_LANE.is_multiple_of(4)
            && PER_LANE_WIDE.is_multiple_of(4)
    );
    // The fused and norm-fused grids are one block per row, and the norm
    // quantizes group `g` in block `g` — row `t`'s group `g` in block
    // `t · (k / 128) + g` at more than one row.
    const _: () = assert!(NORM_K / 128 <= ROWS && MAX_TOKENS * (NORM_K / 128) <= ROWS);
    // The m-row norm-fused entry's routing scratch is its shared row once the
    // last gemv has read it.
    const _: () = assert!(MAX_TOKENS * EXPERTS <= NORM_K);
    // A slot id is a u32.
    const _: () = assert!(ROWS_WIDE <= u32::MAX as usize);
    // The fused and routing entries' static shared memory fits the 48 KiB a
    // block may declare: `P` for eight tokens, and the ticket.
    const _: () = assert!((MAX_TOKENS * EXPERTS_WIDE + 1) * 4 <= 48 * 1024);
    const _: () = assert!(FUSED_WARPS * EXPERTS_WIDE * 4 <= 48 * 1024);
    // The narrow MMA entry routes over its stages' memory, and its replay
    // gives each warp the same number of the block's rows.
    const _: () = assert!(MAX_TOKENS * EXPERTS_WIDE <= NARROW_WORDS);
    const _: () =
        assert!(NARROW_ROWS.is_multiple_of(FUSED_WARPS) && FUSED_THREADS_U32 == MMA_THREADS);

    #[cuda_module]
    mod qwen35moe_router_kernels {
        use super::*;

        /// The router of `m_cols` tokens ([`crate::route_softmax`]'s numeric
        /// contract) at the narrow instance: `w` the joined router weight
        /// ([`ROWS`] rows of `k` f32), `x` the tokens' normed activations
        /// (`m_cols` columns of `k`), `used` the routed experts a token keeps.
        /// Writes
        /// `logits[t·ROWS + r]`, `probs[t·EXPERTS + e]`, `ids[t·(used + 1) +
        /// s]` and `weights[t·(used + 1) + s]`. `done[0]` is the block ticket
        /// count: zero before the launch, zero again after it. Grid
        /// [`ROWS`] blocks of 256; warp 0 of block `r` owns row `r` with
        /// `f32_gemv`'s row body, and the block that draws the last ticket
        /// routes every token, warp `t` token `t`.
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
                m_cols >= 1,
                m_cols <= 8,
                used >= 1,
                used <= 32,
                w.len() >= 257 * k,
                x.len() >= m_cols * k,
                logits.len() >= 257 * m_cols,
                probs.len() >= 256 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused(
            w: &[f32],
            x: &[f32],
            k: u32,
            m_cols: u32,
            used: u32,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { MAX_TOKENS * EXPERTS }> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: LAST and P are this block's own shared allocations (the
            // raw form reaches each `static mut` without a reference), touched
            // by nothing else; the grid is ROWS blocks of 256 (the
            // launcher's), every thread arrives converged; 1 <= m_cols <= 8,
            // 1 <= used <= 32, the input and output lengths are the launch
            // contract's, the launcher passes k a positive multiple of 32, and
            // `done[0]` is zero before the launch.
            unsafe {
                gated_fused_body::<PER_LANE, _>(
                    w,
                    x,
                    k,
                    m_cols,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// The FFN norm and the router of one token in one launch
        /// (`qwen3moe_router_norm`'s contract at [`ROWS`] rows): the q8_1 of
        /// the normed row into `q3 … d8`, `norm_quant`'s bytes, and the
        /// router's outputs as `qwen35moe_router_fused` at one token. `k` a
        /// multiple of 128 up to [`NORM_K`] (host-checked).
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
                k <= 2048,
                used >= 1,
                used <= 32,
                w.len() >= 257 * k,
                x.len() >= k,
                gain.len() >= k,
                q3.len() >= 64 * half_it,
                q4.len() >= 256 * quad_it,
                q6.len() >= 128 * half_it,
                s8.len() >= 8 * n_sb,
                d8.len() >= 2 * n_sb,
                logits.len() >= 257,
                probs.len() >= 256,
                ids.len() >= used + 1,
                weights.len() >= used + 1,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_norm(
            w: &[f32],
            x: &[f32],
            gain: &[f32],
            eps: f32,
            k: u32,
            n_sb: u32,
            half_it: u32,
            quad_it: u32,
            used: u32,
            mut q3: DisjointSlice<u64>,
            mut q4: DisjointSlice<u32>,
            mut q6: DisjointSlice<u32>,
            mut s8: DisjointSlice<i32>,
            mut d8: DisjointSlice<f32>,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
            static mut NORMED: SharedArray<f32, NORM_K> = SharedArray::UNINIT;
            static mut P: SharedArray<f32, EXPERTS> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: WSUM and NORMED are this block's own shared allocations
            // (the raw form reaches each `static mut` without a reference),
            // touched by nothing else; every thread arrives converged with
            // token 0, and the grid is ROWS >= k / 128 blocks of 256 (the
            // launcher's, and the const assert beside ROWS); the launcher
            // passes k a positive multiple of 128 up to NORM_K, n_sb = k/256
            // and its halves and quarters rounded up; the lengths are the
            // launch contract's.
            unsafe {
                norm_row_body::<false>(
                    w,
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    half_it,
                    quad_it,
                    0,
                    ROWS,
                    SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    &mut logits,
                    fault,
                );
            }
            // SAFETY: LAST and P are this block's own shared allocations of 1
            // and EXPERTS entries, untouched until here; every thread arrives
            // converged, 1 <= used <= 32, the output lengths are the launch
            // contract's at one token and `done[0]` is zero before the launch.
            unsafe {
                publish_route_gated::<PER_LANE>(
                    1,
                    used as usize,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// `qwen35moe_router_norm` with warp 0 of each block asking L2 for its router row
        /// first ([`l2_prefetch`]): the entry an integrated device launches.
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
                k <= 2048,
                used >= 1,
                used <= 32,
                w.len() >= 257 * k,
                x.len() >= k,
                gain.len() >= k,
                q3.len() >= 64 * half_it,
                q4.len() >= 256 * quad_it,
                q6.len() >= 128 * half_it,
                s8.len() >= 8 * n_sb,
                d8.len() >= 2 * n_sb,
                logits.len() >= 257,
                probs.len() >= 256,
                ids.len() >= used + 1,
                weights.len() >= used + 1,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_norm_prefetch(
            w: &[f32],
            x: &[f32],
            gain: &[f32],
            eps: f32,
            k: u32,
            n_sb: u32,
            half_it: u32,
            quad_it: u32,
            used: u32,
            mut q3: DisjointSlice<u64>,
            mut q4: DisjointSlice<u32>,
            mut q6: DisjointSlice<u32>,
            mut s8: DisjointSlice<i32>,
            mut d8: DisjointSlice<f32>,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
            static mut NORMED: SharedArray<f32, NORM_K> = SharedArray::UNINIT;
            static mut P: SharedArray<f32, EXPERTS> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: WSUM and NORMED are this block's own shared allocations
            // (the raw form reaches each `static mut` without a reference),
            // touched by nothing else; every thread arrives converged with
            // token 0, and the grid is ROWS >= k / 128 blocks of 256 (the
            // launcher's, and the const assert beside ROWS); the launcher
            // passes k a positive multiple of 128 up to NORM_K, n_sb = k/256
            // and its halves and quarters rounded up; the lengths are the
            // launch contract's.
            unsafe {
                norm_row_body::<true>(
                    w,
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    half_it,
                    quad_it,
                    0,
                    ROWS,
                    SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    &mut logits,
                    fault,
                );
            }
            // SAFETY: LAST and P are this block's own shared allocations of 1
            // and EXPERTS entries, untouched until here; every thread arrives
            // converged, 1 <= used <= 32, the output lengths are the launch
            // contract's at one token and `done[0]` is zero before the launch.
            unsafe {
                publish_route_gated::<PER_LANE>(
                    1,
                    used as usize,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// The FFN norm and the router of `m_cols` tokens in one launch
        /// (`qwen3moe_router_norm_m`'s contract at [`ROWS`] rows):
        /// `fused::norm_quant` then `qwen35moe_router_fused` at `m_cols`
        /// tokens, bit for bit — the q8_1 of the normed rows into `q3 … d8`
        /// (`m_cols` columns), the router's outputs as
        /// `qwen35moe_router_fused` at `m_cols` tokens. Grid [`ROWS`] blocks
        /// of 256: `qwen35moe_router_norm`'s body ([`norm_row_body`]) for
        /// each token in turn, then [`publish_route_gated`], whose scratch is
        /// the shared row, read by then. `k` a multiple of 128 up to
        /// [`NORM_K`] (host-checked).
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        // Four blocks an SM, 64 registers a thread: a grid of 257 blocks stays
        // one wave on 82 SMs.
        #[launch_bounds(256, 4)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                m_cols >= 1,
                m_cols <= 8,
                k <= 2048,
                used >= 1,
                used <= 32,
                w.len() >= 257 * k,
                x.len() >= m_cols * k,
                gain.len() >= k,
                q3.len() >= m_cols * 64 * half_it,
                q4.len() >= m_cols * 256 * quad_it,
                q6.len() >= m_cols * 128 * half_it,
                s8.len() >= m_cols * 8 * n_sb,
                d8.len() >= m_cols * 2 * n_sb,
                logits.len() >= 257 * m_cols,
                probs.len() >= 256 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_norm_m(
            w: &[f32],
            x: &[f32],
            gain: &[f32],
            eps: f32,
            k: u32,
            m_cols: u32,
            n_sb: u32,
            half_it: u32,
            quad_it: u32,
            used: u32,
            mut q3: DisjointSlice<u64>,
            mut q4: DisjointSlice<u32>,
            mut q6: DisjointSlice<u32>,
            mut s8: DisjointSlice<i32>,
            mut d8: DisjointSlice<f32>,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
            static mut NORMED: SharedArray<f32, NORM_K> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

            let mut t = 0usize;
            while t < m_cols as usize {
                // SAFETY: WSUM and NORMED are this block's own shared
                // allocations (the raw form reaches each `static mut` without
                // a reference), touched by nothing else; every thread arrives
                // converged with the same t < m_cols <= 8, and the grid is
                // ROWS >= m_cols · (k / 128) blocks of 256 (the launcher's,
                // and the const assert beside ROWS); the launcher passes k a
                // positive multiple of 128 up to NORM_K, n_sb = k/256 and its
                // halves and quarters rounded up; the lengths are the launch
                // contract's at m_cols tokens.
                unsafe {
                    norm_row_body::<false>(
                        w,
                        x,
                        gain,
                        eps,
                        k,
                        n_sb,
                        half_it,
                        quad_it,
                        t,
                        ROWS,
                        SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                        SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                        &mut q3,
                        &mut q4,
                        &mut q6,
                        &mut s8,
                        &mut d8,
                        &mut logits,
                        fault,
                    );
                }
                t += 1;
            }
            // SAFETY: LAST is this block's own and untouched until here; the
            // routing's scratch is NORMED's first m_cols · EXPERTS <= NORM_K
            // entries (const assert), which warp 0's last gemv has read
            // before the routing's first barrier and nothing touches after
            // it; every thread arrives converged, 1 <= m_cols <= 8, 1 <= used
            // <= 32, the output lengths are the launch contract's and
            // `done[0]` is zero before the launch.
            unsafe {
                publish_route_gated::<PER_LANE>(
                    m_cols as usize,
                    used as usize,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// `qwen35moe_router_norm_m` with warp 0 of each block asking L2 for its router row
        /// first ([`l2_prefetch`]): the entry an integrated device launches.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        // The plain entry's bounds: four blocks an SM, 64 registers a thread.
        #[launch_bounds(256, 4)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                m_cols >= 1,
                m_cols <= 8,
                k <= 2048,
                used >= 1,
                used <= 32,
                w.len() >= 257 * k,
                x.len() >= m_cols * k,
                gain.len() >= k,
                q3.len() >= m_cols * 64 * half_it,
                q4.len() >= m_cols * 256 * quad_it,
                q6.len() >= m_cols * 128 * half_it,
                s8.len() >= m_cols * 8 * n_sb,
                d8.len() >= m_cols * 2 * n_sb,
                logits.len() >= 257 * m_cols,
                probs.len() >= 256 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_norm_m_prefetch(
            w: &[f32],
            x: &[f32],
            gain: &[f32],
            eps: f32,
            k: u32,
            m_cols: u32,
            n_sb: u32,
            half_it: u32,
            quad_it: u32,
            used: u32,
            mut q3: DisjointSlice<u64>,
            mut q4: DisjointSlice<u32>,
            mut q6: DisjointSlice<u32>,
            mut s8: DisjointSlice<i32>,
            mut d8: DisjointSlice<f32>,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
            static mut NORMED: SharedArray<f32, NORM_K> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

            let mut t = 0usize;
            while t < m_cols as usize {
                // SAFETY: WSUM and NORMED are this block's own shared
                // allocations (the raw form reaches each `static mut` without
                // a reference), touched by nothing else; every thread arrives
                // converged with the same t < m_cols <= 8, and the grid is
                // ROWS >= m_cols · (k / 128) blocks of 256 (the launcher's,
                // and the const assert beside ROWS); the launcher passes k a
                // positive multiple of 128 up to NORM_K, n_sb = k/256 and its
                // halves and quarters rounded up; the lengths are the launch
                // contract's at m_cols tokens.
                unsafe {
                    norm_row_body::<true>(
                        w,
                        x,
                        gain,
                        eps,
                        k,
                        n_sb,
                        half_it,
                        quad_it,
                        t,
                        ROWS,
                        SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                        SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                        &mut q3,
                        &mut q4,
                        &mut q6,
                        &mut s8,
                        &mut d8,
                        &mut logits,
                        fault,
                    );
                }
                t += 1;
            }
            // SAFETY: LAST is this block's own and untouched until here; the
            // routing's scratch is NORMED's first m_cols · EXPERTS <= NORM_K
            // entries (const assert), which warp 0's last gemv has read
            // before the routing's first barrier and nothing touches after
            // it; every thread arrives converged, 1 <= m_cols <= 8, 1 <= used
            // <= 32, the output lengths are the launch contract's and
            // `done[0]` is zero before the launch.
            unsafe {
                publish_route_gated::<PER_LANE>(
                    m_cols as usize,
                    used as usize,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// The router logits of `n_tok` tokens at the narrow instance, a
        /// ubatch's first router launch: the wide F32 product's tiles
        /// ([`f32_tile_body`]) over [`ROWS`] rows, `⌈ROWS / 32⌉` row blocks
        /// per 32 tokens; each logit the fused launch's bits for its token.
        /// Writes `logits[t·ROWS + r]`. `k` a positive multiple of 64, `w`
        /// and `x` 16-byte aligned (host-checked).
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                n_tok >= 1,
                w.len() >= 257 * k,
                x.len() >= n_tok * k,
                logits.len() >= 257 * n_tok
            )
        )]
        pub fn qwen35moe_router_logits(
            w: &[f32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own 16-byte aligned shared
            // allocation of STAGES · STAGE_FLOATS values (the raw form
            // reaches the `static mut` without a reference); the block is 256
            // wide and the grid the launcher's ⌈n_tok / 32⌉ · ⌈ROWS / 32⌉;
            // the launcher checks k and both alignments, and the lengths are
            // the launch contract's.
            unsafe {
                f32_tile_body(
                    F32Rows(w),
                    x,
                    ROWS as u32,
                    k,
                    n_tok,
                    &mut logits,
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                );
            }
        }

        /// The routing of `n_tok` tokens at the narrow instance from their
        /// logits (`n_tok · ROWS`, as `qwen35moe_router_logits` writes
        /// them): warp `w` of block `b` routes token `8·b + w` at `used`
        /// routed experts and writes its probabilities, ids and weights where
        /// the fused launch writes a token's.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                used >= 1,
                used <= 32,
                logits.len() >= 257 * n_tok,
                probs.len() >= 256 * n_tok,
                ids.len() >= (used + 1) * n_tok,
                weights.len() >= (used + 1) * n_tok
            )
        )]
        pub fn qwen35moe_router_route(
            logits: &[f32],
            n_tok: u32,
            used: u32,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { FUSED_WARPS * EXPERTS }> = SharedArray::UNINIT;
            // SAFETY: P is this block's own shared allocation of FUSED_WARPS
            // tokens' entries (the raw form reaches the `static mut` without a
            // reference); the block is 256 wide; 1 <= used <= 32 and the
            // lengths are the launch contract's.
            unsafe {
                gated_route_body::<PER_LANE>(
                    logits,
                    n_tok,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    fault,
                );
            }
        }

        /// The router of `m_cols` tokens ([`crate::route_softmax`]'s numeric
        /// contract) at the wide instance: `w` the joined router weight
        /// ([`ROWS_WIDE`] rows of `k` f32), `x` the tokens' normed activations
        /// (`m_cols` columns of `k`), `used` the routed experts a token keeps.
        /// Writes
        /// `logits[t·ROWS_WIDE + r]`, `probs[t·EXPERTS_WIDE + e]`, `ids[t·(used + 1) +
        /// s]` and `weights[t·(used + 1) + s]`. `done[0]` is the block ticket
        /// count: zero before the launch, zero again after it. Grid
        /// [`ROWS_WIDE`] blocks of 256; warp 0 of block `r` owns row `r` with
        /// `f32_gemv`'s row body, and the block that draws the last ticket
        /// routes every token, warp `t` token `t`.
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
                m_cols >= 1,
                m_cols <= 8,
                used >= 1,
                used <= 32,
                w.len() >= 513 * k,
                x.len() >= m_cols * k,
                logits.len() >= 513 * m_cols,
                probs.len() >= 512 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused_512(
            w: &[f32],
            x: &[f32],
            k: u32,
            m_cols: u32,
            used: u32,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { MAX_TOKENS * EXPERTS_WIDE }> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: LAST and P are this block's own shared allocations (the
            // raw form reaches each `static mut` without a reference), touched
            // by nothing else; the grid is ROWS_WIDE blocks of 256 (the
            // launcher's), every thread arrives converged; 1 <= m_cols <= 8,
            // 1 <= used <= 32, the input and output lengths are the launch
            // contract's, the launcher passes k a positive multiple of 32, and
            // `done[0]` is zero before the launch.
            unsafe {
                gated_fused_body::<PER_LANE_WIDE, _>(
                    w,
                    x,
                    k,
                    m_cols,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// `qwen35moe_router_fused_512` over a BF16 joined router weight:
        /// `w` its words ([`ROWS_WIDE`] rows of `k / 2`, value `2j` of a row
        /// in the low half of word `j`), each value widened as it is read
        /// ([`Bf16Words`]), so every output is the F32 entry's over the
        /// widened weight bit for bit.
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
                m_cols >= 1,
                m_cols <= 8,
                used >= 1,
                used <= 32,
                2 * w.len() >= 513 * k,
                x.len() >= m_cols * k,
                logits.len() >= 513 * m_cols,
                probs.len() >= 512 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused_512_bf16(
            w: &[u32],
            x: &[f32],
            k: u32,
            m_cols: u32,
            used: u32,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { MAX_TOKENS * EXPERTS_WIDE }> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: LAST/P belong only to this block; all threads call converged on a ROWS_WIDE-block grid of 256 threads.
            // The launch contract bounds m, used and all slices; the launcher checks positive k divisible by 32; done[0] is zero.
            unsafe {
                gated_fused_body::<PER_LANE_WIDE, _>(
                    Bf16Words(w),
                    x,
                    k,
                    m_cols,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// `qwen35moe_router_fused_512_bf16` on the dense MMA tile's
        /// arithmetic, the entry a device launches whose ubatch logits run
        /// `qwen35moe_router_logits_512_bf16_mma`: grid `⌈ROWS_WIDE / 16⌉`
        /// blocks of 256, block `b` the rows `16·b ..` of every token
        /// ([`gated_mma_fused_body`]), so each logit is that entry's bits for
        /// its token. `k` a positive multiple of 32, `w` and `x` 16-byte
        /// aligned (host-checked).
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
                m_cols >= 1,
                m_cols <= 8,
                used >= 1,
                used <= 32,
                2 * w.len() >= 513 * k,
                x.len() >= m_cols * k,
                logits.len() >= 513 * m_cols,
                probs.len() >= 512 * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused_512_bf16_mma(
            w: &[u32],
            x: &[f32],
            k: u32,
            m_cols: u32,
            used: u32,
            mut logits: DisjointSlice<f32>,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            mut done: DisjointSlice<u32>,
            fault: FaultSink,
        ) {
            static mut STAGE: SharedArray<u32, NARROW_WORDS, 16> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: STAGE and LAST are this block's own shared allocations
            // (the raw form reaches each `static mut` without a reference),
            // STAGE 16-byte aligned; every thread arrives converged on the
            // launcher's grid of ⌈ROWS_WIDE / 16⌉ blocks of 256; the launcher
            // checks k a positive multiple of 32 and both alignments, the
            // launch contract bounds m_cols, used and every slice, and
            // `done[0]` is zero before the launch.
            unsafe {
                gated_mma_fused_body::<PER_LANE_WIDE>(
                    w,
                    x,
                    k,
                    m_cols,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// The router logits of `n_tok` tokens at the wide instance, a
        /// ubatch's first router launch: the wide F32 product's tiles
        /// ([`f32_tile_body`]) over [`ROWS_WIDE`] rows, `⌈ROWS_WIDE / 32⌉`
        /// row blocks per 32 tokens; each logit the fused launch's bits for
        /// its token. Writes `logits[t·ROWS_WIDE + r]`. `k` a positive
        /// multiple of 64, `w` and `x` 16-byte aligned (host-checked).
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                n_tok >= 1,
                w.len() >= 513 * k,
                x.len() >= n_tok * k,
                logits.len() >= 513 * n_tok
            )
        )]
        pub fn qwen35moe_router_logits_512(
            w: &[f32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own 16-byte aligned shared
            // allocation of STAGES · STAGE_FLOATS values (the raw form
            // reaches the `static mut` without a reference); the block is 256
            // wide and the grid the launcher's ⌈n_tok / 32⌉ · ⌈ROWS_WIDE / 32⌉;
            // the launcher checks k and both alignments, and the lengths are
            // the launch contract's.
            unsafe {
                f32_tile_body(
                    F32Rows(w),
                    x,
                    ROWS_WIDE as u32,
                    k,
                    n_tok,
                    &mut logits,
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                );
            }
        }

        /// `qwen35moe_router_logits_512` over a BF16 joined router weight:
        /// `w` its words ([`ROWS_WIDE`] rows of `k / 2`), each staged value
        /// widened to f32 as it is stored, so every logit is the F32 entry's
        /// over the widened weight bit for bit. `k` a positive multiple of
        /// 64, `w` and `x` 16-byte aligned (host-checked).
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                n_tok >= 1,
                2 * w.len() >= 513 * k,
                x.len() >= n_tok * k,
                logits.len() >= 513 * n_tok
            )
        )]
        pub fn qwen35moe_router_logits_512_bf16(
            w: &[u32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own 16-byte-aligned STAGES * STAGE_FLOATS shared values; all 256 threads converge.
            // The launcher proves the logits grid, positive k divisible by the tile line and both alignments; the contract bounds all slices.
            unsafe {
                f32_tile_body(
                    Bf16Rows(w),
                    x,
                    ROWS_WIDE as u32,
                    k,
                    n_tok,
                    &mut logits,
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                );
            }
        }

        /// `qwen35moe_router_logits_512_bf16` on the dense MMA tile
        /// ([`bf16_mma_body`]: the weight's words and the activations rounded
        /// to BF16, F32 accumulation, a cell whose row or token holds an
        /// operand outside the tensor cores' range replayed on the exact
        /// tile), the entry a dense-MMA device launches. Grid `⌈n_tok / 128⌉ ·
        /// ⌈ROWS_WIDE / 128⌉` blocks of 256; each logit is
        /// `qwen35moe_router_fused_512_bf16_mma`'s bits for its token. `k` a
        /// positive multiple of 64, `w` and `x` 16-byte aligned
        /// (host-checked).
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                n_tok >= 1,
                2 * w.len() >= 513 * k,
                x.len() >= n_tok * k,
                logits.len() >= 513 * n_tok
            )
        )]
        pub fn qwen35moe_router_logits_512_bf16_mma(
            w: &[u32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own shared allocation of STAGES ·
            // STAGE_FLOATS values, 16-byte aligned; the raw form reaches the
            // `static mut` without a reference.
            let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
            // SAFETY: every thread of the 256-wide block calls it converged;
            // the launcher sized the MMA grid and checked k and both
            // alignments; the launch contract bounds the words, x and logits.
            unsafe {
                bf16_mma_body(
                    Bf16Rows(w),
                    F32Cols(x),
                    ROWS_WIDE as u32,
                    k,
                    n_tok,
                    &mut logits,
                    sh,
                )
            };
        }

        /// The routing of `n_tok` tokens at the wide instance from their
        /// logits (`n_tok · ROWS_WIDE`, as `qwen35moe_router_logits_512` writes
        /// them): warp `w` of block `b` routes token `8·b + w` at `used`
        /// routed experts and writes its probabilities, ids and weights where
        /// the fused launch writes a token's.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                used >= 1,
                used <= 32,
                logits.len() >= 513 * n_tok,
                probs.len() >= 512 * n_tok,
                ids.len() >= (used + 1) * n_tok,
                weights.len() >= (used + 1) * n_tok
            )
        )]
        pub fn qwen35moe_router_route_512(
            logits: &[f32],
            n_tok: u32,
            used: u32,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { FUSED_WARPS * EXPERTS_WIDE }> = SharedArray::UNINIT;
            // SAFETY: P is this block's own shared allocation of FUSED_WARPS
            // tokens' entries (the raw form reaches the `static mut` without a
            // reference); the block is 256 wide; 1 <= used <= 32 and the
            // lengths are the launch contract's.
            unsafe {
                gated_route_body::<PER_LANE_WIDE>(
                    logits,
                    n_tok,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    fault,
                );
            }
        }
    }

    /// The gated router's buffers ([`RouterKernels`]).
    pub type RouterOut = RouterBufs;

    /// Which instance's entries buffers cut for `out`'s row take.
    #[derive(Clone, Copy)]
    enum Width {
        Narrow,
        Wide,
    }

    /// The instance of `out`'s buffers, and `used` as the launch argument;
    /// buffers cut for another router are refused as `what`.
    fn width(what: &'static str, out: &RouterBufs) -> Result<(Width, u32), GpuError> {
        let used = out.used_for(what, RouterBody::Qwen35moe, None)?;
        let inst = out.dims().inst();
        if inst == ROW {
            Ok((Width::Narrow, used))
        } else if inst == ROW_WIDE {
            Ok((Width::Wide, used))
        } else {
            Err(GpuError::shape(
                what,
                format!("buffers cut for {}; no entry here runs it", inst.at),
            ))
        }
    }

    /// A joined router weight as the card holds it: f32 values, or a BF16
    /// tensor's words (two values a word), which the wide instance's `_bf16`
    /// entries widen as they read them, so its outputs are the F32 entries'
    /// over the widened weight bit for bit.
    #[derive(Clone, Copy)]
    pub enum RouterWeight<'a> {
        F32(&'a DeviceTensor<f32>),
        Bf16 { w: &'a DeviceTensor<u32> },
    }

    impl<'a> RouterWeight<'a> {
        /// The resident joined router weight `name`, F32 or BF16; another
        /// form, or none, refused by name as `what`.
        pub fn of(
            w: &'a Weights,
            what: &'static str,
            name: &str,
        ) -> Result<RouterWeight<'a>, GpuError> {
            match w.resident(what, name)? {
                DevWeight::F32 { w, .. } => Ok(RouterWeight::F32(w)),
                DevWeight::Bf16 { w, .. } => Ok(RouterWeight::Bf16 { w }),
                _ => Err(GpuError::tensor(what, name, "an f32 or bf16 tensor")),
            }
        }

        /// The weight's rows and values a row.
        fn shape(self) -> (usize, usize) {
            match self {
                RouterWeight::F32(w) => (w.rows(), w.cols()),
                RouterWeight::Bf16 { w } => (w.rows(), 2 * w.cols()),
            }
        }

        /// The device address of the weight's first value.
        fn at(self) -> u64 {
            match self {
                RouterWeight::F32(w) => w.buf().cu_deviceptr(),
                RouterWeight::Bf16 { w, .. } => w.buf().cu_deviceptr(),
            }
        }
    }

    impl<'a> From<&'a DeviceTensor<f32>> for RouterWeight<'a> {
        fn from(w: &'a DeviceTensor<f32>) -> Self {
            RouterWeight::F32(w)
        }
    }

    /// Err unless a joined router weight of `(have, k)` rows and values has
    /// `rows` rows and a width that is a positive multiple of `unit` (and
    /// at most `most`): its width.
    fn router_width(
        what: &'static str,
        (have, k): (usize, usize),
        rows: usize,
        unit: usize,
        most: usize,
    ) -> Result<usize, GpuError> {
        if have != rows || k == 0 || !k.is_multiple_of(unit) || k > most {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {have} x {k}, want {rows} rows of a positive multiple of \
                     {unit} up to {most}"
                ),
            ));
        }
        Ok(k)
    }

    /// A fused launch's `(k, m, grid)` for `m` tokens of `x` through `w`
    /// into `out`; its refusal named `what`.
    fn fused_dims(
        what: &'static str,
        w: RouterWeight<'_>,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &RouterBufs,
    ) -> Result<(u32, u32, u32), GpuError> {
        let rows = out.dims().logits();
        let k = router_width(what, w.shape(), rows, 32, usize::MAX)?;
        if !(1..=MAX_TOKENS.min(out.tokens())).contains(&m) || x.len() < m * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} tokens: one launch routes 1..={MAX_TOKENS}, into buffers for {}; \
                     x.len() {} for {m} x {k}",
                    out.tokens(),
                    x.len()
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let grid = launch_u32(what, "grid", rows)?;
        Ok((k, m, grid))
    }

    /// A ubatch logits launch's `(k, n_tok, grid)` for `n` tokens of `x`
    /// through `w` into `out`, the grid the MMA tile's with `mma`; its
    /// refusal named `what`.
    fn ubatch_dims(
        what: &'static str,
        w: RouterWeight<'_>,
        x: &DeviceBuffer<f32>,
        n: usize,
        out: &mut RouterBufs,
        mma: bool,
    ) -> Result<(u32, u32, u32), GpuError> {
        let rows = out.dims().logits();
        let k = router_width(what, w.shape(), rows, F32_TILE_LINE, usize::MAX)?;
        if n > out.tokens() {
            return Err(GpuError::shape(
                what,
                format!("{n} tokens into buffers for {}", out.tokens()),
            ));
        }
        let w_at = w.at();
        if !w_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("router weight at {w_at:#x} must be 16-byte aligned"),
            ));
        }
        let t = if mma {
            mma_launch(what, rows, k, x, n, &out.logits)?
        } else {
            tile_launch(what, rows, k, x, n, &out.logits)?
        };
        Ok((t.k, t.n, t.grid))
    }

    /// A routing launch's `(n_tok, grid)` for the first `n` tokens of `out`;
    /// its refusal named `what`.
    fn route_dims(what: &'static str, n: usize, out: &RouterBufs) -> Result<(u32, u32), GpuError> {
        if !(1..=out.tokens()).contains(&n) {
            return Err(GpuError::shape(
                what,
                format!("{n} tokens into buffers for {}", out.tokens()),
            ));
        }
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(what, "grid", n.div_ceil(FUSED_WARPS))?;
        Ok((n_tok, grid))
    }

    /// The refusal of a BF16 router weight at the narrow instance, which
    /// has no BF16 entries.
    fn narrow_bf16(what: &'static str) -> GpuError {
        GpuError::shape(
            what,
            format!(
                "a bf16 router weight at {}; only the {} instance reads bf16 words",
                ROW.at, ROW_WIDE.at
            ),
        )
    }

    /// The loaded gated router module, both instances. Owns no stream: each
    /// enqueue takes the engine stream.
    pub struct RouterKernels {
        module: qwen35moe_router_kernels::LoadedModule,
        /// [`crate::row_prefetch`] of the context's device.
        row_prefetch: bool,
        /// [`crate::dense_mma`] of the context's device: a BF16 weight at the
        /// wide instance runs the `_bf16_mma` entries, every other weight
        /// and device the exact ones.
        mma: bool,
    }

    impl RouterKernels {
        /// Load this file's device bundle into `ctx`. Load-time only.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
            // SAFETY: this package owns the embedded device bundle produced
            // for the module above; each launcher checks its launch contract.
            let module = unsafe { crate::shared_module!(qwen35moe_router_kernels, ctx)? };
            Ok(RouterKernels {
                module,
                row_prefetch: crate::row_prefetch(ctx)?,
                mma: crate::dense_mma(ctx)?,
            })
        }

        /// Enqueue the router of `m` tokens, 1..=`MAX_TOKENS` and at most
        /// `out.tokens()`, at `out`'s instance and `used`: `w` the joined
        /// router weight (`n + 1` rows of `k`, `k` a positive multiple of
        /// 32; BF16 at the wide instance only, refused by name at the
        /// narrow one), `x` the tokens' `m` columns of `k` activations;
        /// results into `out`, a token with a non-finite logit raising
        /// [`FaultSite::Router`](crate::FaultSite::Router) on `fault`.
        /// Asynchronous, allocation-free, capturable.
        pub fn enqueue_fused<'w>(
            &self,
            stream: &CudaStream,
            w: impl Into<RouterWeight<'w>>,
            x: &DeviceBuffer<f32>,
            m: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_fused";
            let (width, used) = width(what, out)?;
            let w = w.into();
            let (k, m, grid) = fused_dims(what, w, x, m, out)?;
            let cfg = LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0);
            match (width, w) {
                (Width::Narrow, RouterWeight::Bf16 { .. }) => {
                    return Err(narrow_bf16(what));
                }
                (Width::Narrow, RouterWeight::F32(w)) => {
                    let prep = self.module.prepare_qwen35moe_router_fused(cfg)?;
                    self.module.qwen35moe_router_fused(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        m,
                        used,
                        &mut out.logits,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        &mut out.done,
                        fault,
                    )?;
                }
                (Width::Wide, RouterWeight::F32(w)) => {
                    let prep = self.module.prepare_qwen35moe_router_fused_512(cfg)?;
                    self.module.qwen35moe_router_fused_512(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        m,
                        used,
                        &mut out.logits,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        &mut out.done,
                        fault,
                    )?;
                }
                (Width::Wide, RouterWeight::Bf16 { w, .. }) if self.mma => {
                    check_alignment::<U32x4, _>(what, "router weight", w.buf())?;
                    check_alignment::<U32x4, _>(what, "x", x)?;
                    let grid = launch_u32(what, "grid", out.dims().logits().div_ceil(NARROW_ROWS))?;
                    let cfg = LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0);
                    let prep = self
                        .module
                        .prepare_qwen35moe_router_fused_512_bf16_mma(cfg)?;
                    self.module.qwen35moe_router_fused_512_bf16_mma(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        m,
                        used,
                        &mut out.logits,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        &mut out.done,
                        fault,
                    )?;
                }
                (Width::Wide, RouterWeight::Bf16 { w, .. }) => {
                    let prep = self.module.prepare_qwen35moe_router_fused_512_bf16(cfg)?;
                    self.module.qwen35moe_router_fused_512_bf16(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        m,
                        used,
                        &mut out.logits,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        &mut out.done,
                        fault,
                    )?;
                }
            }
            Ok(())
        }

        /// Enqueue the router of a ubatch of `n` (1..=`out.tokens()`)
        /// tokens in two launches at `out`'s instance,
        /// `qwen35moe_router_logits` then `qwen35moe_router_route`
        /// ([`RouterKernels::enqueue_route`]): `w` the joined router weight
        /// (`n + 1` rows of `k`, `k` a positive multiple of 64), `x` the
        /// tokens' `n` columns of `k` normed activations, both 16-byte
        /// aligned; each token's results the bits
        /// [`RouterKernels::enqueue_fused`] leaves for it, its refusal raised
        /// on `fault` as there. Asynchronous, allocation-free, capturable.
        pub fn enqueue_ubatch<'w>(
            &self,
            stream: &CudaStream,
            w: impl Into<RouterWeight<'w>>,
            x: &DeviceBuffer<f32>,
            n: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_ubatch";
            let (width, _) = width(what, out)?;
            let w = w.into();
            let mma = self.mma && matches!((width, w), (Width::Wide, RouterWeight::Bf16 { .. }));
            let (k, n_tok, grid) = ubatch_dims(what, w, x, n, out, mma)?;
            let cfg = LaunchConfig1D::new(grid, F32T_THREADS_U32, 0);
            match (width, w) {
                (Width::Narrow, RouterWeight::Bf16 { .. }) => {
                    return Err(narrow_bf16(what));
                }
                (Width::Narrow, RouterWeight::F32(w)) => {
                    let prep = self.module.prepare_qwen35moe_router_logits(cfg)?;
                    self.module.qwen35moe_router_logits(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
                (Width::Wide, RouterWeight::F32(w)) => {
                    let prep = self.module.prepare_qwen35moe_router_logits_512(cfg)?;
                    self.module.qwen35moe_router_logits_512(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
                (Width::Wide, RouterWeight::Bf16 { w, .. }) if mma => {
                    let cfg = LaunchConfig1D::new(grid, MMA_THREADS, 0);
                    let prep = self
                        .module
                        .prepare_qwen35moe_router_logits_512_bf16_mma(cfg)?;
                    self.module.qwen35moe_router_logits_512_bf16_mma(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
                (Width::Wide, RouterWeight::Bf16 { w, .. }) => {
                    let prep = self.module.prepare_qwen35moe_router_logits_512_bf16(cfg)?;
                    self.module.qwen35moe_router_logits_512_bf16(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
            }
            self.enqueue_route(stream, n, fault, out)
        }

        /// Enqueue the routing alone of the first `n` (1..=`out.tokens()`)
        /// tokens' logits in `out.logits` at `out`'s instance
        /// (`qwen35moe_router_route`): each token's probabilities, ids and
        /// weights into `out` where the fused launch writes a token's, a
        /// token with a non-finite logit raising
        /// [`FaultSite::Router`](crate::FaultSite::Router) on `fault`. The
        /// second launch of [`RouterKernels::enqueue_ubatch`].
        /// Asynchronous, allocation-free, capturable.
        pub fn enqueue_route(
            &self,
            stream: &CudaStream,
            n: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_route";
            let (width, used) = width(what, out)?;
            let (n_tok, grid) = route_dims(what, n, out)?;
            let cfg = LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0);
            match width {
                Width::Narrow => {
                    let prep = self.module.prepare_qwen35moe_router_route(cfg)?;
                    self.module.qwen35moe_router_route(
                        stream,
                        &prep,
                        &out.logits,
                        n_tok,
                        used,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        fault,
                    )?;
                }
                Width::Wide => {
                    let prep = self.module.prepare_qwen35moe_router_route_512(cfg)?;
                    self.module.qwen35moe_router_route_512(
                        stream,
                        &prep,
                        &out.logits,
                        n_tok,
                        used,
                        &mut out.probs,
                        &mut out.ids,
                        &mut out.weights,
                        fault,
                    )?;
                }
            }
            Ok(())
        }

        /// Enqueue the FFN norm and the router of one token in one launch
        /// (`qwen35moe_router_norm`), at the narrow instance only (the wide
        /// one has no norm-fused entry, and its buffers are refused by name):
        /// `x` the token's FFN input residual, `gain` the norm's gain
        /// (`w.cols()` values each), `eps` its epsilon; the q8_1 of the
        /// normed row into `act` (one column of `w.cols()`) — the bytes
        /// `fused::norm_quant` writes — and the routing into `out`'s first
        /// token, as [`RouterKernels::enqueue_fused`] at one token. `fault`
        /// takes the norm's raise and the router's. Asynchronous,
        /// allocation-free, capturable.
        #[allow(
            clippy::too_many_arguments,
            reason = "host launcher over the norm's and the router's buffers, the shape of the kernel's arguments"
        )]
        pub fn enqueue_norm_fused(
            &self,
            stream: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            gain: &DeviceBuffer<f32>,
            eps: f32,
            act: &mut Q8Act,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_norm_fused";
            let used = out.used_for(what, RouterBody::Qwen35moe, Some(ROW.per_lane()))?;
            let k = router_width(what, (w.rows(), w.cols()), ROWS, 128, NORM_K)?;
            if act.m() != 1 || act.k() != k || x.len() < k || gain.len() < k {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "one column of {k}: act {} x {}, x.len() {}, gain.len() {}",
                        act.m(),
                        act.k(),
                        x.len(),
                        gain.len()
                    ),
                ));
            }
            let n_sb = act.n_sb();
            let k = launch_u32(what, "k", k)?;
            let n_sb = launch_u32(what, "n_sb", n_sb)?;
            let grid = launch_u32(what, "grid", ROWS)?;
            if self.row_prefetch {
                let prep =
                    self.module
                        .prepare_qwen35moe_router_norm_prefetch(LaunchConfig1D::new(
                            grid,
                            FUSED_THREADS_U32,
                            0,
                        ))?;
                self.module.qwen35moe_router_norm_prefetch(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    n_sb.div_ceil(2),
                    n_sb.div_ceil(4),
                    used,
                    &mut act.q3,
                    &mut act.q4,
                    &mut act.q6,
                    &mut act.s8,
                    &mut act.d8,
                    &mut out.logits,
                    &mut out.probs,
                    &mut out.ids,
                    &mut out.weights,
                    &mut out.done,
                    fault,
                )?;
            } else {
                let prep = self
                    .module
                    .prepare_qwen35moe_router_norm(LaunchConfig1D::new(
                        grid,
                        FUSED_THREADS_U32,
                        0,
                    ))?;
                self.module.qwen35moe_router_norm(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    n_sb.div_ceil(2),
                    n_sb.div_ceil(4),
                    used,
                    &mut act.q3,
                    &mut act.q4,
                    &mut act.q6,
                    &mut act.s8,
                    &mut act.d8,
                    &mut out.logits,
                    &mut out.probs,
                    &mut out.ids,
                    &mut out.weights,
                    &mut out.done,
                    fault,
                )?;
            }
            Ok(())
        }

        /// Enqueue the FFN norm and the router of `act.m()` tokens
        /// (1..=`MAX_TOKENS`, at most `out.tokens()`) in one launch
        /// (`qwen35moe_router_norm_m`), at the narrow instance only (the
        /// wide one has no norm-fused entry, and its buffers are refused by
        /// name): `x` the tokens' FFN input residuals (`act.m()` rows of
        /// `w.cols()`), `gain` the norm's gain, `eps` its epsilon; the q8_1
        /// of the normed rows into `act` — the bytes `fused::norm_quant`
        /// writes — and the routing into `out`, as
        /// [`RouterKernels::enqueue_fused`] at `act.m()` tokens. `fault`
        /// takes the norm's raise and the router's. Asynchronous,
        /// allocation-free, capturable.
        #[allow(
            clippy::too_many_arguments,
            reason = "host launcher over the norm's and the router's buffers, the shape of the kernel's arguments"
        )]
        pub fn enqueue_norm_fused_m(
            &self,
            stream: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            gain: &DeviceBuffer<f32>,
            eps: f32,
            act: &mut Q8Act,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_norm_fused_m";
            let used = out.used_for(what, RouterBody::Qwen35moe, Some(ROW.per_lane()))?;
            let k = router_width(what, (w.rows(), w.cols()), ROWS, 128, NORM_K)?;
            let m = act.m();
            if !(1..=MAX_TOKENS.min(out.tokens())).contains(&m)
                || act.k() != k
                || x.len() < m * k
                || gain.len() < k
            {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "{m} rows of {k}: one launch routes 1..={MAX_TOKENS}, into buffers for \
                         {}; act {m} x {}, x.len() {}, gain.len() {}",
                        out.tokens(),
                        act.k(),
                        x.len(),
                        gain.len()
                    ),
                ));
            }
            let n_sb = act.n_sb();
            let k = launch_u32(what, "k", k)?;
            let m = launch_u32(what, "m", m)?;
            let n_sb = launch_u32(what, "n_sb", n_sb)?;
            let grid = launch_u32(what, "grid", ROWS)?;
            if self.row_prefetch {
                let prep =
                    self.module
                        .prepare_qwen35moe_router_norm_m_prefetch(LaunchConfig1D::new(
                            grid,
                            FUSED_THREADS_U32,
                            0,
                        ))?;
                self.module.qwen35moe_router_norm_m_prefetch(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    gain,
                    eps,
                    k,
                    m,
                    n_sb,
                    n_sb.div_ceil(2),
                    n_sb.div_ceil(4),
                    used,
                    &mut act.q3,
                    &mut act.q4,
                    &mut act.q6,
                    &mut act.s8,
                    &mut act.d8,
                    &mut out.logits,
                    &mut out.probs,
                    &mut out.ids,
                    &mut out.weights,
                    &mut out.done,
                    fault,
                )?;
            } else {
                let prep = self
                    .module
                    .prepare_qwen35moe_router_norm_m(LaunchConfig1D::new(
                        grid,
                        FUSED_THREADS_U32,
                        0,
                    ))?;
                self.module.qwen35moe_router_norm_m(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    gain,
                    eps,
                    k,
                    m,
                    n_sb,
                    n_sb.div_ceil(2),
                    n_sb.div_ceil(4),
                    used,
                    &mut act.q3,
                    &mut act.q4,
                    &mut act.q6,
                    &mut act.s8,
                    &mut act.d8,
                    &mut out.logits,
                    &mut out.probs,
                    &mut out.ids,
                    &mut out.weights,
                    &mut out.done,
                    fault,
                )?;
            }
            Ok(())
        }
    }
}
