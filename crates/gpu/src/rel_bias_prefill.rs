//! Rel-bias attention over a prompt pass's rows: the attention of `rel_bias`
//! (the score's distance-gathered bias, the per-(row, head) sink, the window,
//! the plane or the ring) with the key and value tiles staged once for every
//! row of a block and the products on the tensor cores — the prompt form of
//! that file's step entries, at [`GROUP_3`] (Inkling 2 Mini's 24 query heads
//! over 8) and [`GROUP_6`] query heads a key head (the Small's 48 over 8),
//! one group-generic body.
//!
//! Row `t` attends over the keys `[first, limit)` of key head `kh` with
//! `limit = n_keys[t]` and `first` the window cut of `limit`
//! ([`window_cut`]: `window` 0 attends every live key), the step entry's
//! rule; its position is `limit − 1`, so a pass's rows over its own appended
//! keys are causal by their counts. The score of key `j` is
//! `(q · kⱼ) · scale + bⱼ`, `bⱼ = Σ_{d < REL} r_d · rel_bias[d, limit − 1 −
//! j]`, 0 at a distance at or past `extent`; the row's sink `t` joins the
//! softmax last as the partial `(t, 1, 0)`; the output is the weighted value
//! sum over the denominator.
//!
//! Geometry: block `b` is key head `b % n_kv` and query tile `n_tiles − 1 −
//! b / n_kv` (the deepest first), [`POSITIONS`] consecutive rows of `t_rows`;
//! its `G` warps are the group's query heads, warp `w` head `kh·G + w` of
//! those sixteen rows — one `m16` fragment. The block walks the keys of the
//! union of its rows' ranges in [`KEY_TILE`]-key tiles at absolute positions
//! (tile `i` holds keys `[32·i, 32·i + 32)`), staging each tile's key and
//! value rows into shared memory once for all its `16·G` query rows
//! (`cp.async`, the value tile landing under the scores, the next key tile
//! under the value product), each staged row the cache row's 256 bytes with
//! its sixteen 16-byte chunks permuted within each 128-byte half by `chunk ^
//! (row & 7)` (`flash_gqa_prefill`'s layout). A key outside every row's
//! range is staged as zeros and never read from the cache.
//!
//! Per tile and warp:
//! 1. the bias tile, warp-private: lane `j` is key `32·i + j`, and for each
//!    of the warp's sixteen rows the bias of that key in the step entry's
//!    order — `REL` fused multiply-adds ascending `d` from 0, `r` read from
//!    shared, the table from global memory — or 0 for a key outside the row's
//!    range or past the extent: bit for bit the step's bias;
//! 2. `S = Q·Kᵀ` on `mma.m16n8k16` with f32 accumulation, the query row split
//!    into two f16 halves `q = q_hi + q_lo` (`q_hi` its f16 rounding, `q_lo`
//!    the f16 rounding of the rest): the key rows are f16 already, so both
//!    products are exact and the split dot carries the query to `2⁻²²` of each
//!    value — the f32 dot of the step, in the tensor core's order; the eight
//!    k16 steps ascend, the high half before the low one at each. Then `s =
//!    S · scale` rounded, the bias added (one rounded add), `−inf` outside
//!    the row's range;
//! 3. the online softmax in f32: the tile max over the row's four lanes, a
//!    rescale `f = dev_exp(m − m')` on a bump (0 for the first), the weights
//!    `p = dev_exp(s − m')` (0 outside the range), the lane's running sum `l =
//!    l·f + Σ p`, the accumulators times `f` unless every lane has `f = 1`;
//! 4. `O += P·V` on `mma.m16n8k16`, the weights split as the query is (`p =
//!    p_hi + p_lo`, both against the f16 values), the tile's two k16 steps
//!    ascending, the high half before the low one.
//!
//! After the last tile: the row sum over its four lanes `(l + l^1) + l^2`,
//! the sink folded as `online_fold` folds it (a NaN or `+inf` sink has no
//! defined softmax and writes NaN), and `o · (1/l)`. The order of the steps
//! above is the gate: a rerun is bit-identical, and a row's bits depend only
//! on its own query, `r` and sink, the keys and values of the tiles its range
//! touches and the fixed tile partition — not on `t_rows`, the rows sharing
//! its block or the grid. The step entry walks its own segments and tiles, so
//! a row's output is the step's within the rounding of two f32 sum orders,
//! not its bits; the owning gate bands that distance.
//!
//! Over a ring of `ring_rows` rows key `p` is row `p mod ring_rows`: the
//! caller appends a pass's rows before the attention, so the ring must hold
//! every key a row of the pass reads — the window and the pass's rows,
//! `t_rows + window − 1 <= ring_rows`, which the host refuses otherwise (a
//! windowless ring must hold the whole context).
//!
//! No silent failure: a live count of zero or past `ctx` on a row below
//! `t_rows` raises [`FaultSite::KeyCount`] and the row's output is NaN; a
//! query value outside the f16 range makes its high half infinite and its
//! low half NaN, so the row is NaN, never a clamped score.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::dev_exp;
use crate::flash_gqa::{HEAD, window_cut};
use crate::flash_gqa_h128_swa::{GROUP_3, GROUP_6, live_keys};
use crate::flash_gqa_prefill::{layout_holds, staged};
use crate::rel_bias::REL;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::convert::{cvt_f16x2_f32, cvt_f32x2_f16x2};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::vector::U32x4;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Consecutive rows a block takes: one `m16` fragment a warp.
pub const POSITIONS: usize = 16;
/// Keys one tile stages. Tile `i` holds keys `[32·i, 32·i + 32)`.
pub const KEY_TILE: usize = 32;

/// u32 words of a key or value row, and of a staged tile of them.
const ROW_WORDS: usize = HEAD / 2;
const TILE_WORDS: usize = KEY_TILE * ROW_WORDS;
/// 16-byte chunks of a row, and of a tile.
const ROW_CHUNKS: usize = HEAD / 8;
const TILE_CHUNKS: usize = KEY_TILE * ROW_CHUNKS;
const QK_STEPS: usize = HEAD / 16;
const KEY_NT: usize = KEY_TILE / 8;
const PV_STEPS: usize = KEY_TILE / 16;
const DIM_NT: usize = HEAD / 8;
const DIM_PAIRS: usize = HEAD / 16;
/// u32 words of half a staged row (the permutation never crosses it), and
/// the `ldmatrix` chunk pairs in it.
const HALF_ROW: usize = ROW_WORDS / 2;
const HALF_PAIRS: usize = HALF_ROW / 8;
/// The bias tile's row stride: 40 f32, so a fragment's eight rows of
/// two-value reads hit the 32 banks in two wavefronts.
const BIAS_STRIDE: usize = KEY_TILE + 8;
/// f32 of a warp's bias tile and of its rows' `r` values.
const BIAS_WARP: usize = POSITIONS * BIAS_STRIDE;
const R_WARP: usize = POSITIONS * REL;
/// The rows' live counts and first keys, as the walk uses them.
const COUNTS: usize = 2 * POSITIONS;

const _: () = assert!(HEAD == 128 && REL == 16 && KEY_TILE == 32 && POSITIONS == 16);
const _: () = assert!(KEY_TILE == 2 * 16 && HEAD.is_multiple_of(16) && ROW_CHUNKS == 16);
const _: () = assert!(layout_holds(KEY_TILE, ROW_WORDS) && (ROW_WORDS * 4).is_multiple_of(128));
const _: () = assert!(GROUP_3 == 3 && GROUP_6 == 6);

/// Copies a thread issues for one tile at `g` warps a block (the last pass
/// ragged: the tile's chunks are not a multiple of the block's threads).
const fn copy_passes(g: usize) -> usize {
    TILE_CHUNKS.div_ceil(32 * g)
}

/// The two f16 halves of a pair of f32: the f16 roundings `hi`, then the f16
/// roundings of what they leave, `lo` — packed as the `mma` A fragment reads
/// them, the first value in the low half.
#[inline(always)]
fn split_pair(a: f32, b: f32) -> (u32, u32) {
    let hi = cvt_f16x2_f32(a, b);
    let (ha, hb) = cvt_f32x2_f16x2(hi);
    (hi, cvt_f16x2_f32(a - ha, b - hb))
}

/// Stage tile `kb` of one side into `dst`: chunk `e = tid + 32·G·i` is key
/// `32·kb + e / 16`'s chunk `e % 16`, copied from its cache row (row `key mod
/// plane_rows` under `RING`) when the key lies in `[lo, hi)` and zeroed
/// otherwise, then committed as one group.
///
/// SAFETY: `dst` is a tile of [`TILE_WORDS`] u32 in this block's shared
/// memory, 128-byte aligned; `src` is the key head's plane (`plane_rows`
/// rows of [`ROW_WORDS`] words, 16-byte aligned), and every key in `[lo,
/// hi)` has its row inside it (`hi <= plane_rows` on a plane; any key on a
/// ring); no warp reads `dst` before the barrier after this stage's wait.
#[inline(always)]
unsafe fn stage_tile<const G: usize, const RING: bool>(
    dst: *mut u32,
    src: *const u32,
    kb: usize,
    (lo, hi): (usize, usize),
    plane_rows: usize,
    tid: usize,
) {
    // The ring's slot of the tile's first key, once: a ring holds at least a
    // tile's rows (the host's refusal), so a row's slot wraps at most once.
    let base = if RING {
        ((KEY_TILE * kb) as u32 % plane_rows as u32) as usize
    } else {
        KEY_TILE * kb
    };
    for i in 0..copy_passes(G) {
        cuda_device::thread::__unroll_config::<0>();
        let e = tid + 32 * G * i;
        if e < TILE_CHUNKS {
            let row = e / ROW_CHUNKS;
            let chunk = e - row * ROW_CHUNKS;
            let key = KEY_TILE * kb + row;
            // SAFETY: row < KEY_TILE and 4·chunk + 4 <= ROW_WORDS: one whole
            // chunk of the tile, which `staged` keeps whole and 16-byte
            // aligned; this thread alone names it.
            let d = unsafe { dst.add(staged(row, 4 * chunk, ROW_WORDS)) };
            if key >= lo && key < hi {
                let at = base + row;
                let at = if RING && at >= plane_rows {
                    at - plane_rows
                } else {
                    at
                };
                // SAFETY: the key's row is inside the plane (the caller's
                // contract), 16-byte aligned (whole 256-byte rows from an
                // aligned base).
                unsafe { cp_async_cg_16(d, src.add(at * ROW_WORDS + 4 * chunk)) };
            } else {
                // SAFETY: `d` is this thread's 16 aligned bytes of the tile.
                unsafe { *d.cast::<U32x4>() = U32x4::splat(0) };
            }
        }
    }
    // SAFETY: closes this thread's copies above as one group.
    unsafe { cp_async_commit_group() };
}

/// The attention of the module doc at `G` query heads a key head, on a plane
/// (`RING` false, `plane_rows = ctx`) or a ring of `plane_rows` rows.
///
/// SAFETY: the entry's launch contract at `n_kv · G` query heads (every
/// buffer bound it names; under `RING` every key a row reads resident in the
/// ring), `q` 8-byte and the planes 16-byte aligned, a block of `32·G`
/// threads, and the block's shared tiles as the entries declare them: `kt`
/// and `vt` of [`TILE_WORDS`] u32, 128-byte aligned; `bw` of `G ·`
/// [`BIAS_WARP`] f32; `rs` of `G ·` [`R_WARP`] f32; `cs` of [`COUNTS`] u32.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments and its shared tiles, handed on (rust-quality R8)"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn rel_prefill<const G: usize, const RING: bool>(
    q: &[f32],
    (kc, vc): (&[u16], &[u16]),
    n_keys_buf: &[u32],
    (r, bias, sinks): (&[f32], &[f32], &[f32]),
    (extent, scale): (u32, f32),
    (n_kv, ctx, plane_rows, t_rows, window): (u32, u32, u32, u32, u32),
    fault: FaultSink,
    y: &mut DisjointSlice<f32>,
    (kt, vt, bw, rs, cs): (*mut u32, *mut u32, *mut f32, *mut f32, *mut u32),
) {
    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let rows = t_rows as usize;
    let n_tiles = rows.div_ceil(POSITIONS);
    if b >= n_tiles * nkv {
        return; // block-uniform
    }
    let qt = n_tiles - 1 - b / nkv;
    let kh = b % nkv;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let w = tid / 32;
    let g = lane / 4;
    let t4 = lane % 4;
    let n_head = nkv * G;
    let h = kh * G + w;
    let t0 = qt * POSITIONS;
    let ctx = ctx as usize;
    let extent = extent as usize;
    let plane_rows = plane_rows as usize;

    // The rows' counts and first keys (0 for a row past t_rows or refused),
    // and each warp's rows of r.
    if tid < POSITIONS {
        let t = t0 + tid;
        let (limit, first) = if t < rows {
            // SAFETY: t < t_rows <= n_keys_buf.len() (launch contract).
            let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
            if limit == 0 {
                fault.raise(FaultSite::KeyCount);
            }
            (limit, window_cut(limit, window as usize).0)
        } else {
            (0, 0)
        };
        // SAFETY: tid < POSITIONS: inside cs, this thread's two slots.
        unsafe {
            *cs.add(tid) = limit as u32;
            *cs.add(POSITIONS + tid) = first as u32;
        }
    }
    let mut e = lane;
    while e < R_WARP {
        let i = e / REL;
        let t = t0 + i;
        let v = if t < rows {
            // SAFETY: t < t_rows and h < n_head: (t·n_head + h)·REL + e % REL
            // is inside r (launch contract).
            unsafe { *r.get_unchecked((t * n_head + h) * REL + e % REL) }
        } else {
            0.0
        };
        // SAFETY: w < G and e < R_WARP: inside rs, this lane's slot.
        unsafe { *rs.add(w * R_WARP + e) = v };
        e += 32;
    }
    thread::sync_threads();

    // The block's key range: the smallest first key and the largest count of
    // its live rows; the lane's two rows' own.
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for i in 0..POSITIONS {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: inside cs, written before the barrier above.
        let (lim, fst) = unsafe { (*cs.add(i) as usize, *cs.add(POSITIONS + i) as usize) };
        if lim > 0 {
            lo = lo.min(fst);
            hi = hi.max(lim);
        }
    }
    let mut cnt = [0usize; 2];
    let mut fst = [0usize; 2];
    let mut bad = [false; 2];
    for hh in 0..2 {
        cuda_device::thread::__unroll_config::<0>();
        let i = g + 8 * hh;
        // SAFETY: i < POSITIONS: inside cs, written before the barrier above.
        unsafe {
            cnt[hh] = *cs.add(i) as usize;
            fst[hh] = *cs.add(POSITIONS + i) as usize;
        }
        bad[hh] = t0 + i < rows && cnt[hh] == 0;
    }

    // The warp's query rows as split A fragments, one pair a k16 step: a0 is
    // row g's values 16·kk + 2·t4 .. +1, a1 row g + 8's, a2 and a3 the same
    // eight values on.
    let q64 = q.as_ptr().cast::<u64>();
    let mut qh = [[0u32; 4]; QK_STEPS];
    let mut ql = [[0u32; 4]; QK_STEPS];
    for kk in 0..QK_STEPS {
        cuda_device::thread::__unroll_config::<0>();
        for e in 0usize..4 {
            cuda_device::thread::__unroll_config::<0>();
            let t = t0 + g + 8 * (e % 2);
            let col = 16 * kk + 2 * t4 + 8 * (e / 2);
            let raw = if t < rows {
                // SAFETY: t < t_rows and h < n_head put the pair inside q's
                // row (launch contract); col is even and q 8-byte aligned.
                unsafe { *q64.add(((t * n_head + h) * HEAD + col) / 2) }
            } else {
                0
            };
            let (hi16, lo16) = split_pair(
                f32::from_bits(raw as u32),
                f32::from_bits((raw >> 32) as u32),
            );
            qh[kk][e] = hi16;
            ql[kk][e] = lo16;
        }
    }

    // This lane's `ldmatrix` addresses in the first sixteen rows and the
    // first half-row: the scores' key row and chunk `2·j + (lane/8) % 2`,
    // the values' row and chunk `2·j + lane/16` (`flash_gqa_prefill`'s).
    let k_row = lane % 8 + 8 * (lane / 16);
    let v_row = lane % 8 + 8 * ((lane / 8) % 2);
    let mut k_at = [0u32; HALF_PAIRS];
    let mut v_at = [0u32; HALF_PAIRS];
    for j in 0..HALF_PAIRS {
        cuda_device::thread::__unroll_config::<0>();
        // SAFETY: k_row, v_row < 16 <= KEY_TILE and words 8·j + 4 .. + 4 <=
        // HALF_ROW: inside the tiles; only the addresses are formed here.
        unsafe {
            k_at[j] = shared::cvta_generic_to_shared_u32(
                kt.add(staged(k_row, 8 * j + 4 * ((lane / 8) % 2), ROW_WORDS))
                    .cast_const()
                    .cast::<u8>(),
            );
            v_at[j] = shared::cvta_generic_to_shared_u32(
                vt.add(staged(v_row, 8 * j + 4 * (lane / 16), ROW_WORDS))
                    .cast_const()
                    .cast::<u8>(),
            );
        }
    }

    let plane = kh * plane_rows * ROW_WORDS;
    let k_src = kc.as_ptr().cast::<u32>().wrapping_add(plane);
    let v_src = vc.as_ptr().cast::<u32>().wrapping_add(plane);
    let (kb_lo, kb_hi) = if hi > 0 {
        (lo / KEY_TILE, hi.div_ceil(KEY_TILE))
    } else {
        (0, 0)
    };
    if kb_lo < kb_hi {
        // SAFETY: the tile and the plane as the caller's contract names
        // them; keys in [lo, hi) are rows the rows read.
        unsafe { stage_tile::<G, RING>(kt, k_src, kb_lo, (lo, hi), plane_rows, tid) };
    }

    let mut m = [f32::NEG_INFINITY; 2];
    let mut l = [0.0f32; 2];
    let mut o = [[0.0f32; 4]; DIM_NT];
    let mut kb = kb_lo;
    while kb < kb_hi {
        let key0 = KEY_TILE * kb;
        // ---- the warp's bias tile: lane j is key key0 + j (this order is
        // the gate: the step's REL fused multiply-adds ascending d).
        let key = key0 + lane;
        let mut i = 0usize;
        while i < POSITIONS {
            // SAFETY: inside cs and rs, written before the first barrier.
            let (lim, first) = unsafe { (*cs.add(i) as usize, *cs.add(POSITIONS + i) as usize) };
            let mut bv = 0.0f32;
            if key >= first && key < lim && lim - 1 - key < extent {
                let dist = lim - 1 - key;
                // SAFETY: w < G, i < POSITIONS: the row's REL values of rs.
                let rb = unsafe { rs.add(w * R_WARP + i * REL) };
                for d in 0..REL {
                    cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: d·extent + dist < REL·extent <= bias.len()
                    // (launch contract); rb + d inside the row's values.
                    bv = f32::mul_add(
                        unsafe { *rb.add(d) },
                        unsafe { *bias.get_unchecked(d * extent + dist) },
                        bv,
                    );
                }
            }
            // SAFETY: w < G, i < POSITIONS, lane < KEY_TILE: this lane's slot
            // of the warp's bias tile.
            unsafe { *bw.add(w * BIAS_WARP + i * BIAS_STRIDE + lane) = bv };
            i += 1;
        }

        // Key tile kb has landed, every warp is done with value tile kb − 1,
        // and the bias tile above is published.
        // SAFETY: waits for this thread's outstanding groups; the barrier then
        // publishes every thread's copies and stores.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        // SAFETY: as the first stage.
        unsafe { stage_tile::<G, RING>(vt, v_src, kb, (lo, hi), plane_rows, tid) };

        // ---- S = (Q_hi + Q_lo)·Kᵀ.
        let mut sc = [[0.0f32; 4]; KEY_NT];
        for kk in 0..QK_STEPS {
            cuda_device::thread::__unroll_config::<0>();
            for nj in 0..KEY_NT / 2 {
                cuda_device::thread::__unroll_config::<0>();
                // Key row 16·nj + k_row, chunk 2·kk + (lane/8) % 2.
                let at = k_at[kk % HALF_PAIRS]
                    + (4 * (16 * nj * ROW_WORDS + HALF_ROW * (kk / HALF_PAIRS))) as u32;
                // SAFETY: the row is below KEY_TILE and the chunk whole inside
                // KT (staged, layout_holds), published by the barrier above;
                // every lane issues the load, then the four `mma.sync`.
                unsafe {
                    let bf = wmma::ldmatrix_x4_shared_u32(at);
                    let b0 = [bf[0], bf[1]];
                    let b1 = [bf[2], bf[3]];
                    sc[2 * nj] = wmma::mma_m16n8k16_f32_f16(sc[2 * nj], qh[kk], b0);
                    sc[2 * nj] = wmma::mma_m16n8k16_f32_f16(sc[2 * nj], ql[kk], b0);
                    sc[2 * nj + 1] = wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], qh[kk], b1);
                    sc[2 * nj + 1] = wmma::mma_m16n8k16_f32_f16(sc[2 * nj + 1], ql[kk], b1);
                }
            }
        }

        // ---- the scores, the tile max and the online softmax.
        let mut tmax = [f32::NEG_INFINITY; 2];
        let mut live = [[false; 4]; KEY_NT];
        for nt in 0..KEY_NT {
            cuda_device::thread::__unroll_config::<0>();
            for j in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let hh = j / 2;
                let col = 8 * nt + 2 * t4 + j % 2;
                let key = key0 + col;
                let ok = key >= fst[hh] && key < cnt[hh];
                // SAFETY: row g + 8·hh < POSITIONS and col < KEY_TILE: inside
                // the warp's bias tile, published by the barrier above.
                let bv = unsafe { *bw.add(w * BIAS_WARP + (g + 8 * hh) * BIAS_STRIDE + col) };
                let s = if ok {
                    add_rn_f32(mul_rn_f32(sc[nt][j], scale), bv)
                } else {
                    f32::NEG_INFINITY
                };
                sc[nt][j] = s;
                live[nt][j] = ok;
                tmax[hh] = tmax[hh].max(s);
            }
        }
        let mut f = [1.0f32; 2];
        for hh in 0..2 {
            cuda_device::thread::__unroll_config::<0>();
            let mut x = tmax[hh];
            x = x.max(warp::shuffle_xor_f32(x, 1));
            x = x.max(warp::shuffle_xor_f32(x, 2));
            let m_new = m[hh].max(x);
            if m_new > m[hh] {
                f[hh] = if m[hh] > f32::NEG_INFINITY {
                    dev_exp(m[hh] - m_new)
                } else {
                    0.0
                };
                m[hh] = m_new;
            }
            l[hh] = mul_rn_f32(l[hh], f[hh]);
        }
        let mut pah = [[0u32; 4]; PV_STEPS];
        let mut pal = [[0u32; 4]; PV_STEPS];
        for kk2 in 0..PV_STEPS {
            cuda_device::thread::__unroll_config::<0>();
            for e in 0usize..4 {
                cuda_device::thread::__unroll_config::<0>();
                let nt = 2 * kk2 + e / 2;
                let j0 = 2 * (e % 2);
                let hh = e % 2;
                let p0 = if live[nt][j0] {
                    dev_exp(sc[nt][j0] - m[hh])
                } else {
                    0.0
                };
                let p1 = if live[nt][j0 + 1] {
                    dev_exp(sc[nt][j0 + 1] - m[hh])
                } else {
                    0.0
                };
                l[hh] = add_rn_f32(add_rn_f32(l[hh], p0), p1);
                let (ph, pl) = split_pair(p0, p1);
                pah[kk2][e] = ph;
                pal[kk2][e] = pl;
            }
        }
        // The accumulators times f, unless no lane of the warp has a factor
        // other than 1 (the vote keeps the branch uniform).
        if warp::any(f[0] != 1.0 || f[1] != 1.0) {
            for nd in 0..DIM_NT {
                cuda_device::thread::__unroll_config::<0>();
                for j in 0usize..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    o[nd][j] = mul_rn_f32(o[nd][j], f[j / 2]);
                }
            }
        }

        // The value tile has landed, and every warp is done with the key tile
        // and its bias tile.
        // SAFETY: as the wait above.
        unsafe { cp_async_wait_group(0) };
        thread::sync_threads();
        if kb + 1 < kb_hi {
            // SAFETY: as the first stage.
            unsafe { stage_tile::<G, RING>(kt, k_src, kb + 1, (lo, hi), plane_rows, tid) };
        }

        // ---- O += (P_hi + P_lo)·V.
        for kk2 in 0..PV_STEPS {
            cuda_device::thread::__unroll_config::<0>();
            for nd in 0..DIM_PAIRS {
                cuda_device::thread::__unroll_config::<0>();
                // Value row 16·kk2 + v_row, chunk 2·nd + lane/16.
                let at = v_at[nd % HALF_PAIRS]
                    + (4 * (16 * kk2 * ROW_WORDS + HALF_ROW * (nd / HALF_PAIRS))) as u32;
                // SAFETY: the row is below KEY_TILE and the chunk whole inside
                // VT, published by the barrier above; every lane issues the
                // load, then the four `mma.sync`.
                unsafe {
                    let bf = wmma::ldmatrix_x4_trans_shared_u32(at);
                    let b0 = [bf[0], bf[1]];
                    let b1 = [bf[2], bf[3]];
                    o[2 * nd] = wmma::mma_m16n8k16_f32_f16(o[2 * nd], pah[kk2], b0);
                    o[2 * nd] = wmma::mma_m16n8k16_f32_f16(o[2 * nd], pal[kk2], b0);
                    o[2 * nd + 1] = wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pah[kk2], b1);
                    o[2 * nd + 1] = wmma::mma_m16n8k16_f32_f16(o[2 * nd + 1], pal[kk2], b1);
                }
            }
        }
        kb += 1;
    }

    // ---- the row sums, the sink, then o · (1/l).
    for hh in 0..2 {
        cuda_device::thread::__unroll_config::<0>();
        let t = t0 + g + 8 * hh;
        let a = add_rn_f32(l[hh], warp::shuffle_xor_f32(l[hh], 1));
        let s = add_rn_f32(a, warp::shuffle_xor_f32(a, 2));
        if t < rows {
            // SAFETY: t < t_rows and h < n_head: inside sinks (launch
            // contract).
            let sink = unsafe { *sinks.get_unchecked(t * n_head + h) };
            // `online_fold` of the partial (sink, 1, 0): the side with the
            // smaller max rescaled.
            let (fo, den) = if sink > m[hh] {
                let fo = if m[hh] > f32::NEG_INFINITY {
                    dev_exp(m[hh] - sink)
                } else {
                    0.0
                };
                (fo, f32::mul_add(s, fo, 1.0))
            } else {
                (1.0, f32::mul_add(1.0, dev_exp(sink - m[hh]), s))
            };
            let nan = bad[hh] || sink.is_nan() || sink == f32::INFINITY;
            let inv = 1.0 / den;
            for nd in 0..DIM_NT {
                cuda_device::thread::__unroll_config::<0>();
                for jj in 0usize..2 {
                    cuda_device::thread::__unroll_config::<0>();
                    let d = 8 * nd + 2 * t4 + jj;
                    let v = if nan {
                        f32::NAN
                    } else {
                        mul_rn_f32(f32::mul_add(o[nd][2 * hh + jj], fo, 0.0), inv)
                    };
                    // SAFETY: t < t_rows, h < n_head, d < HEAD: inside y
                    // (launch contract); one owner lane per value.
                    unsafe { *y.get_unchecked_mut((t * n_head + h) * HEAD + d) = v };
                }
            }
        }
    }
}

/// Shared tiles of one block at `G` warps: the key and value tiles, then the
/// bias tiles, the `r` rows and the counts.
macro_rules! rel_prefill_entry {
    ($g:expr, $ring:literal, ($q:ident, $kc:ident, $vc:ident, $n_keys_buf:ident, $r:ident,
     $bias:ident, $sinks:ident, $extent:ident, $scale:ident, $n_kv:ident, $ctx:ident,
     $plane_rows:expr, $t_rows:ident, $window:ident, $fault:ident, $y:ident)) => {{
        // 128-byte aligned: the layout's sectors and halves are aligned only
        // if the tile base is.
        static mut KT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;
        static mut VT: SharedArray<u32, TILE_WORDS, 128> = SharedArray::UNINIT;
        static mut BW: SharedArray<f32, { $g * BIAS_WARP }> = SharedArray::UNINIT;
        static mut RS: SharedArray<f32, { $g * R_WARP }> = SharedArray::UNINIT;
        static mut CS: SharedArray<u32, COUNTS> = SharedArray::UNINIT;
        let mut y = $y;
        // SAFETY: each `static mut` above is this block's own shared
        // allocation at the sizes `rel_prefill` names for `G` warps; the raw
        // form reaches it without a reference. The launch contract is
        // `rel_prefill`'s and the block is 32·G threads.
        unsafe {
            rel_prefill::<{ $g }, $ring>(
                $q,
                ($kc, $vc),
                $n_keys_buf,
                ($r, $bias, $sinks),
                ($extent, $scale),
                ($n_kv, $ctx, $plane_rows, $t_rows, $window),
                $fault,
                &mut y,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut KT),
                    SharedArray::as_raw_mut_ptr(&raw mut VT),
                    SharedArray::as_raw_mut_ptr(&raw mut BW),
                    SharedArray::as_raw_mut_ptr(&raw mut RS),
                    SharedArray::as_raw_mut_ptr(&raw mut CS),
                ),
            )
        };
    }};
}

#[cuda_module]
mod rel_bias_prefill_kernels {
    use super::*;

    /// The module doc's attention on a plane at [`GROUP_3`] query heads a
    /// key head: a block of 96 threads, warp `w` query head `kh·3 + w` of
    /// [`POSITIONS`] rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(96)]
    #[launch_contract(
        domain = 1,
        block = (96, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            extent >= 1,
            q.len() >= t_rows * n_kv * 3 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            r.len() >= t_rows * n_kv * 3 * 16,
            bias.len() >= 16 * extent,
            sinks.len() >= t_rows * n_kv * 3,
            y.len() >= t_rows * n_kv * 3 * 128
        )
    )]
    pub fn rel_prefill_h128_g3(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        sinks: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        rel_prefill_entry!(
            GROUP_3,
            false,
            (
                q, kc, vc, n_keys_buf, r, bias, sinks, extent, scale, n_kv, ctx, ctx, t_rows,
                window, fault, y
            )
        );
    }

    /// [`rel_prefill_h128_g3`] on a ring of `ring_rows` rows a key head: key
    /// `p` is row `p mod ring_rows`, `ctx` stays the bound of the live count.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(96)]
    #[launch_contract(
        domain = 1,
        block = (96, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            extent >= 1,
            ring_rows >= 32,
            q.len() >= t_rows * n_kv * 3 * 128,
            kc.len() >= n_kv * ring_rows * 128,
            vc.len() >= n_kv * ring_rows * 128,
            r.len() >= t_rows * n_kv * 3 * 16,
            bias.len() >= 16 * extent,
            sinks.len() >= t_rows * n_kv * 3,
            y.len() >= t_rows * n_kv * 3 * 128
        )
    )]
    pub fn rel_prefill_h128_g3_ring(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        sinks: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        ring_rows: u32,
        t_rows: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        rel_prefill_entry!(
            GROUP_3,
            true,
            (
                q, kc, vc, n_keys_buf, r, bias, sinks, extent, scale, n_kv, ctx, ring_rows, t_rows,
                window, fault, y
            )
        );
    }

    /// The module doc's attention on a plane at [`GROUP_6`] query heads a
    /// key head: a block of 192 threads, warp `w` query head `kh·6 + w`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            extent >= 1,
            q.len() >= t_rows * n_kv * 6 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            r.len() >= t_rows * n_kv * 6 * 16,
            bias.len() >= 16 * extent,
            sinks.len() >= t_rows * n_kv * 6,
            y.len() >= t_rows * n_kv * 6 * 128
        )
    )]
    pub fn rel_prefill_h128_g6(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        sinks: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        t_rows: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        rel_prefill_entry!(
            GROUP_6,
            false,
            (
                q, kc, vc, n_keys_buf, r, bias, sinks, extent, scale, n_kv, ctx, ctx, t_rows,
                window, fault, y
            )
        );
    }

    /// [`rel_prefill_h128_g6`] on a ring of `ring_rows` rows a key head.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(192)]
    #[launch_contract(
        domain = 1,
        block = (192, 1, 1),
        requires = (
            n_keys_buf.len() >= t_rows,
            extent >= 1,
            ring_rows >= 32,
            q.len() >= t_rows * n_kv * 6 * 128,
            kc.len() >= n_kv * ring_rows * 128,
            vc.len() >= n_kv * ring_rows * 128,
            r.len() >= t_rows * n_kv * 6 * 16,
            bias.len() >= 16 * extent,
            sinks.len() >= t_rows * n_kv * 6,
            y.len() >= t_rows * n_kv * 6 * 128
        )
    )]
    pub fn rel_prefill_h128_g6_ring(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        sinks: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        ring_rows: u32,
        t_rows: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        rel_prefill_entry!(
            GROUP_6,
            true,
            (
                q, kc, vc, n_keys_buf, r, bias, sinks, extent, scale, n_kv, ctx, ring_rows, t_rows,
                window, fault, y
            )
        );
    }
}

/// Arguments of one prompt rel-bias launch: `rel_bias::RelBiasArgs` without
/// the partials (one launch, no merge). `window` 0 attends every live key;
/// `bias` is the `[REL][extent]` f32 table, row-major; `r` holds `REL` values
/// a (row, head), `sinks` one f32 a (row, head); `ring_rows` `None` runs the
/// plane entries, `Some(rows)` the ring entries over a ring of `rows` rows a
/// key head; `y` is `[m][n_head][HEAD]`.
pub struct RelPrefillArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub r: &'a DeviceBuffer<f32>,
    pub bias: &'a DeviceBuffer<f32>,
    pub extent: usize,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub ring_rows: Option<usize>,
    pub m: usize,
    pub window: usize,
    pub sinks: &'a DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// Blocks a pass of `m` rows over `n_kv` key heads launches.
#[must_use]
pub fn blocks_for(m: usize, n_kv: usize) -> usize {
    m.div_ceil(POSITIONS) * n_kv
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RelBiasPrefillKernels {
    module: rel_bias_prefill_kernels::LoadedModule,
}

impl RelBiasPrefillKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RelBiasPrefillKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(rel_bias_prefill_kernels, ctx)? };
        Ok(RelBiasPrefillKernels { module })
    }

    /// Enqueue `m` rows at [`GROUP_3`] or [`GROUP_6`] query heads a key head
    /// (`n_head / n_kv`; any other ratio is refused by name, as a non-finite
    /// scale, a zero extent or count, a short buffer and a ring that cannot
    /// hold the window and the pass's rows are). A live count of zero or past
    /// `ctx` raises [`FaultSite::KeyCount`] and the row is NaN. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        args: RelPrefillArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "rel_bias_prefill::enqueue";
        let RelPrefillArgs {
            q,
            kc,
            vc,
            n_keys,
            r,
            bias,
            extent,
            scale,
            n_kv,
            ctx,
            ring_rows,
            m,
            window,
            sinks,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 || extent == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_kv, ctx, m and extent >= 1, got n_kv={n_kv} ctx={ctx} m={m} \
                     extent={extent}"
                ),
            ));
        }
        let group = n_head / n_kv;
        if n_head != n_kv * group || (group != GROUP_3 && group != GROUP_6) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the prompt rel-bias kernels take {GROUP_3} or {GROUP_6} query heads a key \
                     head; got {n_head} heads over {n_kv}"
                ),
            ));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("the score scale must be finite, got {scale}"),
            ));
        }
        if let Some(rows) = ring_rows {
            let holds = rows >= KEY_TILE
                && rows <= u32::MAX as usize
                && if window == 0 {
                    ctx <= rows
                } else {
                    m + window - 1 <= rows
                };
            if !holds {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "a ring of {rows} rows under a pass of {m} rows at window {window} (ctx \
                         {ctx}): the ring must hold the window and the pass's rows, or the whole \
                         context without a window, and at least a key tile of {KEY_TILE}"
                    ),
                ));
            }
        }
        let plane_rows = ring_rows.unwrap_or(ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kc", kc.len(), n_kv * plane_rows * HEAD),
            ("vc", vc.len(), n_kv * plane_rows * HEAD),
            ("n_keys", n_keys.len(), m),
            ("r", r.len(), m * n_head * REL),
            ("bias", bias.len(), REL * extent),
            ("sinks", sinks.len(), m * n_head),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", blocks_for(m, n_kv))?;
        let n_kv_u = launch_u32(what, "n_kv", n_kv)?;
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let extent_u = launch_u32(what, "extent", extent)?;
        let ring_u = launch_u32(what, "ring_rows", plane_rows)?;
        let m_u = launch_u32(what, "m", m)?;
        let window_u = launch_u32(what, "window", window)?;
        let threads = launch_u32(what, "threads", 32 * group)?;
        let cfg = LaunchConfig1D::new(grid, threads, 0);
        match (group == GROUP_6, ring_rows.is_some()) {
            (false, false) => {
                let prep = self.module.prepare_rel_prefill_h128_g3(cfg)?;
                self.module.rel_prefill_h128_g3(
                    stream, &prep, q, kc, vc, n_keys, r, bias, sinks, extent_u, scale, n_kv_u,
                    ctx_u, m_u, window_u, fault, y,
                )?;
            }
            (false, true) => {
                let prep = self.module.prepare_rel_prefill_h128_g3_ring(cfg)?;
                self.module.rel_prefill_h128_g3_ring(
                    stream, &prep, q, kc, vc, n_keys, r, bias, sinks, extent_u, scale, n_kv_u,
                    ctx_u, ring_u, m_u, window_u, fault, y,
                )?;
            }
            (true, false) => {
                let prep = self.module.prepare_rel_prefill_h128_g6(cfg)?;
                self.module.rel_prefill_h128_g6(
                    stream, &prep, q, kc, vc, n_keys, r, bias, sinks, extent_u, scale, n_kv_u,
                    ctx_u, m_u, window_u, fault, y,
                )?;
            }
            (true, true) => {
                let prep = self.module.prepare_rel_prefill_h128_g6_ring(cfg)?;
                self.module.rel_prefill_h128_g6_ring(
                    stream, &prep, q, kc, vc, n_keys, r, bias, sinks, extent_u, scale, n_kv_u,
                    ctx_u, ring_u, m_u, window_u, fault, y,
                )?;
            }
        }
        Ok(())
    }
}
