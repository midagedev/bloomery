//! Grouped-query flash attention for `m` query rows — one decode position,
//! or the consecutive positions of a prompt chunk: [`GROUP`] query heads
//! share one key/value head of [`HEAD`] values, and the cache is a pair of
//! f16 planes, `[n_kv][ctx][HEAD]` each (the layout
//! `rope_neox::head_norm_neox_append` writes). Row `t` sees the keys below
//! its own live count `n_keys[t]` — the causal limit of its position. The
//! segment + merge shape of `flash.rs`: the key range is cut into
//! `seg_keys`-key segments, block `(segment, kv head, row)` walks its
//! segment for all the group's query heads of that row at once — one warp
//! per query head over one staging of the K and V tiles — and writes per
//! head the softmax partials `(max, Σ exp, Σ exp·V)`; the merge folds a
//! head's live segments in ascending order. A row's arithmetic does not
//! depend on `m` or on the other rows: one row is the one-position launch.
//! The segment count comes from the cache height, so a captured graph's
//! grid is fixed; a segment wholly past its row's live count writes the
//! neutral partial (`m = −inf`, `s = 0`) and reads no key row, and the merge
//! stops at the row's last live segment.
//!
//! A live count of zero or past the cache has no defined attention. The
//! segment passes and the merge read the count through one rule
//! ([`live_keys`]): such a row reads no key, and the merge raises
//! [`FaultSite::KeyCount`] and writes NaN into the row's output.
//!
//! Reduction structure, the fixed contract of this kernel (reruns and
//! replays are bit-identical; bit identity with the reference is not
//! claimed — its exponential is another function):
//! - the score of key `j` is lane `j`'s dot of the query row with the key
//!   row over [`ILP`] rotating f32 partials (partial `p` takes the value
//!   pairs `p, p + ILP, …`, both values of a pair by one fused multiply-add
//!   each), combined as `(a0 + a1) + (a2 + a3)`, then one multiply by
//!   `scale`;
//! - online softmax in [`KEY_TILE`]-key tiles, keys ascending: the tile max
//!   by the five-step butterfly, the running max bumped, each weight
//!   `dev_exp(s − m)`, the weight sum by the five-step butterfly, the
//!   running sum and the value partials rescaled by `dev_exp(m_old − m)` on
//!   a bump;
//! - lane `l` accumulates dims `4l .. 4l + 3` over a tile's keys ascending,
//!   one fused multiply-add per key;
//! - the merge: `flash::online_fold` over the live segments ascending, then
//!   `r · (1/s)`.
//!
//! Two segment passes share everything after the scores (`fold_tile`) and
//! the merge: [`flash_gqa_kernels::gqa_flash_seg`] computes each score as
//! the f32 dot above, [`flash_gqa_kernels::gqa_flash_seg_mma`] on the tensor
//! cores from the query rounded to f16 (`mma.m16n8k16`, the group's eight
//! heads in rows 0–7 of a 16-row tile). [`gqa_mma`] picks one per process.
//!
//! Keys at or past the live count get weight exactly zero and are never
//! loaded: the staging writes zero for them, so a padded row holding a NaN
//! pattern reaches no result.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{
    MERGE_BATCH, MMA_K, MMA_NTILE, MMA_ROWS, dev_exp, f32_to_f16_bits, half_bits_to_f32,
    mma_row_words, online_fold,
};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values per head.
pub const HEAD: usize = 128;
/// Query heads per key/value head, and so warps per segment block.
pub const GROUP: usize = 8;
/// Keys per online-softmax tile: one key per lane.
pub const KEY_TILE: usize = 32;
/// Keys per segment. A multiple of [`KEY_TILE`]; the choice keeps the grid
/// above the card's SM count from about a thousand keys on (four key heads
/// times one block per segment).
pub const SEG_KEYS: usize = 64;
/// Rotating partials of the score dot.
const ILP: usize = 4;

const THREADS: usize = GROUP * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS);
/// u32 words of one key row, and the padded stride the K tile is staged at:
/// lane `j` reads word `w` of key `j` at `j·65 + w`, a distinct bank per lane.
const ROW_WORDS: usize = HEAD / 2;
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of one value row: lane `l` reads word `l`, dims `4l .. 4l + 3`.
const ROW_QWORDS: usize = HEAD / 4;
const _: () = assert!(ROW_QWORDS == 32 && SEG_KEYS.is_multiple_of(KEY_TILE));
const _: () = assert!(ROW_WORDS.is_multiple_of(ILP));

/// u32 words per staged row of the tensor-core pass's query and key tiles.
const MMA_ROW_W: usize = mma_row_words(HEAD);
const _: () = assert!(MMA_ROW_W * 4 % 128 == 16 && MMA_ROWS == 2 * GROUP);
const _: () = assert!(KEY_TILE == 4 * MMA_NTILE && HEAD.is_multiple_of(2 * MMA_K));

/// Merge threads per head: one per dim.
const MERGE_THREADS: u32 = HEAD as u32;

/// Whether [`FlashGqaKernels::enqueue`] runs the tensor-core segment pass
/// ([`flash_gqa_kernels::gqa_flash_seg_mma`]) rather than the f32 scalar one
/// ([`flash_gqa_kernels::gqa_flash_seg`]): `BLOOMERY_GQA_MMA=1` (the default)
/// picks it, `0` the scalar pass. Read once, at first use, so a captured
/// graph and its replays run one pass; a value that is set but unusable
/// panics rather than falling back.
pub fn gqa_mma() -> bool {
    static MMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MMA.get_or_init(|| match std::env::var("BLOOMERY_GQA_MMA") {
        Err(_) => true,
        Ok(v) if v == "1" => true,
        Ok(v) if v == "0" => false,
        Ok(v) => panic!("BLOOMERY_GQA_MMA={v} is neither `1` nor `0`"),
    })
}

/// Segments a `ctx`-row cache is cut into.
#[must_use]
pub fn segments_for(ctx: usize) -> usize {
    ctx.max(1).div_ceil(SEG_KEYS)
}

/// The `Σ exp·V` partials' length for `m` rows of `n_head` query heads over
/// a `ctx`-row cache.
#[must_use]
pub fn partials_v_len(m: usize, n_head: usize, ctx: usize) -> usize {
    m * n_head * segments_for(ctx) * HEAD
}

/// The `(max, Σ exp)` partials' length.
#[must_use]
pub fn partials_ms_len(m: usize, n_head: usize, ctx: usize) -> usize {
    m * n_head * segments_for(ctx) * 2
}

/// A row's live key count as every kernel here reads it: `count` when it
/// lies in `1..=ctx`, else 0 — a refused row, whose segments are all
/// neutral and whose merge raises [`FaultSite::KeyCount`].
#[inline(always)]
fn live_keys(count: u32, ctx: usize) -> usize {
    let c = count as usize;
    if c > ctx { 0 } else { c }
}

/// One tile's online-softmax step for warp `w`'s head, lane `lane` holding
/// key `lane`'s score `sc` (`live` false past the segment's end): the tile
/// max by the five-step butterfly, the running `(mx, s_sum)` and the value
/// partials rescaled on a bump, the weight `dev_exp(sc − mx)` (0 when not
/// live), the weight sum by the five-step butterfly, and the value partials
/// of dims `4·lane .. +3` accumulated over the tile's keys ascending, one
/// fused multiply-add per key. Both segment passes run their tiles through
/// it, so their softmax and value orders are one.
///
/// SAFETY: `ws` is the block's `GROUP · KEY_TILE` f32 weight tile and `vs`
/// its staged `KEY_TILE · ROW_QWORDS` value tile (published by a block
/// barrier); `w < GROUP`; all 32 lanes of the warp call it together.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "the tile step's state is the caller's registers, passed by reference"
)]
unsafe fn fold_tile(
    sc: f32,
    live: bool,
    mx: &mut f32,
    s_sum: &mut f32,
    acc: &mut [f32; 4],
    ws: *mut f32,
    vs: *const u64,
    w: usize,
    lane: usize,
) {
    let mut tmax = sc;
    let mut off = 16u32;
    while off > 0 {
        tmax = tmax.max(warp::shuffle_xor_f32(tmax, off));
        off >>= 1;
    }
    let m_new = mx.max(tmax);
    let pw = if live { dev_exp(sc - m_new) } else { 0.0 };
    let mut psum = pw;
    let mut off = 16u32;
    while off > 0 {
        psum += warp::shuffle_xor_f32(psum, off);
        off >>= 1;
    }
    if m_new > *mx {
        let f = if *mx > f32::NEG_INFINITY {
            dev_exp(*mx - m_new)
        } else {
            0.0
        };
        *s_sum *= f;
        acc[0] *= f;
        acc[1] *= f;
        acc[2] *= f;
        acc[3] *= f;
        *mx = m_new;
    }
    *s_sum += psum;

    // SAFETY: w·32 + lane < GROUP·KEY_TILE; this warp's own slots.
    unsafe { *ws.add(w * KEY_TILE + lane) = pw };
    warp::sync_mask(u32::MAX);
    let mut j = 0usize;
    while j < KEY_TILE {
        // SAFETY: j < 32 and lane < 32: inside WS and VS, written before
        // the caller's barrier (VS) or the warp sync (WS) above.
        let (pj, vw) = unsafe { (*ws.add(w * KEY_TILE + j), *vs.add(j * ROW_QWORDS + lane)) };
        acc[0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[0]);
        acc[1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[1]);
        acc[2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[2]);
        acc[3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[3]);
        j += 1;
    }
    warp::sync_mask(u32::MAX);
}

#[cuda_module]
mod flash_gqa_kernels {
    use super::*;

    /// The segment pass. Block `b = (seg·n_kv + kh)·m + t` walks keys
    /// `[seg · seg_keys, min((seg + 1)·seg_keys, n_keys))` of key head `kh`
    /// for row `t`'s query heads `kh·GROUP ..`; warp `w` is query head `h =
    /// kh·GROUP + w`, whose partials land at index `(t·n_head + h)·segs +
    /// seg`. `n_keys` is [`live_keys`] of `n_keys_buf[t]`: a refused row's
    /// segments are all neutral. The rows of one segment are neighbouring
    /// blocks, so they read its key rows while they are in L2.
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
            n_keys_buf.len() >= m,
            segs * seg_keys >= ctx,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let n_seg = segs as usize;
        let rows = m as usize;
        if b >= rows * nkv * n_seg {
            return; // block-uniform
        }
        let t = b % rows;
        let sk = b / rows;
        let seg = sk / nkv;
        let kh = sk - seg * nkv;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let n_head = nkv * GROUP;
        let h = kh * GROUP + w;
        let idx = (t * n_head + h) * n_seg + seg;
        let ctx = ctx as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
        let lo = seg * seg_keys as usize;
        if lo >= limit {
            if lane == 0 {
                // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform: lo and limit are the block's
        }
        let hi = (lo + seg_keys as usize).min(limit);

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. Every
        // index below is inside its array, and every write is ordered before
        // its reads by a block barrier (or, for WS, written and read by one
        // warp between two barriers).
        let (qs, ks, vs, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };

        // Row t's group query rows: thread `tid` stages values 4·tid .. +3 of
        // the group's 1024.
        let qb = (t * n_head + kh * GROUP) * HEAD;
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: qb + 4·tid + i < (t·n_head + (kh + 1)·GROUP)·HEAD <=
            // m·n_kv·1024 <= q.len(); the shared index 4·tid + i < GROUP·HEAD.
            unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
            i += 1;
        }

        let plane = kh * ctx * HEAD;
        let k64 = kc.as_ptr() as *const u64;
        let v64 = vc.as_ptr() as *const u64;
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = [0.0f32; 4];
        let mut t0 = lo;
        while t0 < hi {
            thread::sync_threads();
            // Stage the tile: 32 keys × 32 u64 of K and of V, four of each
            // per thread; a key at or past `hi` stages zeros.
            let mut i = 0usize;
            while i < 4 {
                let e = tid + THREADS * i;
                let key = e / ROW_QWORDS;
                let word = e - key * ROW_QWORDS;
                let (kw, vw) = if t0 + key < hi {
                    // SAFETY: t0 + key < hi <= ctx, so the row is inside key
                    // head kh's plane; a row is 128 u16 = 32 u64 and the
                    // planes start 8-byte aligned (a device allocation), so
                    // the u64 read is aligned and inside both planes.
                    unsafe {
                        let r = (plane + (t0 + key) * HEAD) / 4 + word;
                        (*k64.add(r), *v64.add(r))
                    }
                } else {
                    (0u64, 0u64)
                };
                // SAFETY: key < 32 and word < 32: 2·word + 1 < K_STRIDE and
                // key·32 + word < KEY_TILE·ROW_QWORDS.
                unsafe {
                    *ks.add(key * K_STRIDE + 2 * word) = kw as u32;
                    *ks.add(key * K_STRIDE + 2 * word + 1) = (kw >> 32) as u32;
                    *vs.add(key * ROW_QWORDS + word) = vw;
                }
                i += 1;
            }
            thread::sync_threads();

            // The score of key t0 + lane for head h.
            let live = t0 + lane < hi;
            let mut a = [0.0f32; ILP];
            let mut wd = 0usize;
            while wd < ROW_WORDS {
                let mut p = 0usize;
                while p < ILP {
                    // SAFETY: lane < 32, wd + p < ROW_WORDS: inside KS; the
                    // query index w·HEAD + 2(wd + p) + 1 < GROUP·HEAD.
                    let (kw, q0, q1) = unsafe {
                        (
                            *ks.add(lane * K_STRIDE + wd + p),
                            *qs.add(w * HEAD + 2 * (wd + p)),
                            *qs.add(w * HEAD + 2 * (wd + p) + 1),
                        )
                    };
                    let k0 = half_bits_to_f32(kw as u16);
                    let k1 = half_bits_to_f32((kw >> 16) as u16);
                    a[p] = f32::mul_add(q0, k0, a[p]);
                    a[p] = f32::mul_add(q1, k1, a[p]);
                    p += 1;
                }
                wd += ILP;
            }
            let dot = (a[0] + a[1]) + (a[2] + a[3]);
            let sc = if live { dot * scale } else { f32::NEG_INFINITY };

            // SAFETY: WS and VS are this block's tiles, VS staged before the
            // barrier above; w < GROUP and lane < 32.
            unsafe { fold_tile(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane) };
            t0 += KEY_TILE;
        }

        // SAFETY: idx < m·n_kv·GROUP·segs; the four dims 4·lane .. +3 of the
        // partial row are this lane's alone, and lane 0 writes (m, s).
        unsafe {
            let o = idx * HEAD + 4 * lane;
            *part_v.get_unchecked_mut(o) = acc[0];
            *part_v.get_unchecked_mut(o + 1) = acc[1];
            *part_v.get_unchecked_mut(o + 2) = acc[2];
            *part_v.get_unchecked_mut(o + 3) = acc[3];
            if lane == 0 {
                *part_ms.get_unchecked_mut(2 * idx) = mx;
                *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
            }
        }
    }

    /// The segment pass with `S = Q·Kᵀ` on the tensor cores: the grid,
    /// the partials and every step after the scores are [`gqa_flash_seg`]'s
    /// (`fold_tile`), so [`gqa_flash_merge`] serves both. The group's eight
    /// query rows are staged as f16 (rounded to nearest even) into rows
    /// `0..8` of a 16-row tile whose rows `8..16` stay zero, and a tile's 32
    /// keys are staged as the cache's own f16 pairs; warps `0..4` each take
    /// eight keys and walk the 128 dims in four `ldmatrix.x4` steps of two
    /// `mma.m16n8k16` each (the fragment roles of `flash_latent_mma`), and
    /// write `scale · S` for rows `0..8` into the logit tile. Both staged
    /// strides are `mma_row_words(128)` words, so eight rows of one
    /// `ldmatrix` phase cover the 32 banks once. Keys at or past the
    /// segment's end are staged zero and their logits are `−inf`.
    ///
    /// The query rows and the products are f16, a different arithmetic
    /// class from the f32 scalar pass, with its own band.
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
            n_keys_buf.len() >= m,
            segs * seg_keys >= ctx,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_mma(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QT: SharedArray<u32, { MMA_ROWS * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut KT: SharedArray<u32, { KEY_TILE * MMA_ROW_W }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        let n_seg = segs as usize;
        let rows = m as usize;
        if b >= rows * nkv * n_seg {
            return; // block-uniform
        }
        let t = b % rows;
        let sk = b / rows;
        let seg = sk / nkv;
        let kh = sk - seg * nkv;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let n_head = nkv * GROUP;
        let h = kh * GROUP + w;
        let idx = (t * n_head + h) * n_seg + seg;
        let ctx = ctx as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
        let lo = seg * seg_keys as usize;
        if lo >= limit {
            if lane == 0 {
                // SAFETY: idx < m·n_kv·GROUP·segs, both slots inside part_ms.
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform: lo and limit are the block's
        }
        let hi = (lo + seg_keys as usize).min(limit);

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. Every
        // index below is inside its array, and every write is ordered before
        // its reads by a block barrier (or, for WS, by the warp sync inside
        // `fold_tile`).
        let (qt, kt, vs, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QT),
                SharedArray::as_raw_mut_ptr(&raw mut KT),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };

        // The query tile: warp `w` rounds row t's head `w`'s 64 value pairs,
        // two per lane, and writes the zero rows `8 + w`. Each pair is one
        // 8-byte load, and the loads come before the roundings.
        let qb = (t * n_head + kh * GROUP + w) * HEAD;
        let q64 = q.as_ptr() as *const u64;
        let mut raw = [0u64; 2];
        let mut i = 0usize;
        #[unroll]
        while i < 2 {
            // SAFETY: qb is a multiple of HEAD, so word (qb + 2·wd)/2 holds
            // values qb + 2·wd and + 1, with qb + 2·wd + 1 < (t·n_head +
            // kh·GROUP + w + 1)·HEAD <= m·n_kv·1024 <= q.len(); the buffer
            // starts 8-byte aligned (a device allocation).
            unsafe { raw[i] = *q64.add(qb / 2 + lane + 32 * i) };
            i += 1;
        }
        let mut i = 0usize;
        #[unroll]
        while i < 2 {
            let wd = lane + 32 * i;
            let lo16 = f32_to_f16_bits(f32::from_bits(raw[i] as u32)) as u32;
            let hi16 = f32_to_f16_bits(f32::from_bits((raw[i] >> 32) as u32)) as u32;
            // SAFETY: the tile words w·MMA_ROW_W + wd and (8 + w)·MMA_ROW_W +
            // wd are inside QT (wd < 64 < MMA_ROW_W, 8 + w < MMA_ROWS).
            unsafe {
                *qt.add(w * MMA_ROW_W + wd) = lo16 | (hi16 << 16);
                *qt.add((GROUP + w) * MMA_ROW_W + wd) = 0;
            }
            i += 1;
        }

        let plane = kh * ctx * HEAD;
        let k64 = kc.as_ptr() as *const u64;
        let v64 = vc.as_ptr() as *const u64;
        let key0 = (w % 4) * MMA_NTILE;
        let arow = lane % MMA_ROWS;
        let ahalf = lane / MMA_ROWS;
        let bkey = lane % MMA_NTILE;
        let boct = lane / MMA_NTILE;
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = [0.0f32; 4];
        let mut t0 = lo;
        while t0 < hi {
            thread::sync_threads();
            let mut i = 0usize;
            while i < 4 {
                let e = tid + THREADS * i;
                let key = e / ROW_QWORDS;
                let word = e - key * ROW_QWORDS;
                let (kw, vw) = if t0 + key < hi {
                    // SAFETY: as in gqa_flash_seg — the row is inside key
                    // head kh's plane and the u64 read is aligned.
                    unsafe {
                        let r = (plane + (t0 + key) * HEAD) / 4 + word;
                        (*k64.add(r), *v64.add(r))
                    }
                } else {
                    (0u64, 0u64)
                };
                // SAFETY: key < 32 and 2·word + 1 < 64 < MMA_ROW_W: inside
                // KT; key·32 + word inside VS.
                unsafe {
                    *kt.add(key * MMA_ROW_W + 2 * word) = kw as u32;
                    *kt.add(key * MMA_ROW_W + 2 * word + 1) = (kw >> 32) as u32;
                    *vs.add(key * ROW_QWORDS + word) = vw;
                }
                i += 1;
            }
            thread::sync_threads();

            if w < 4 {
                let mut c = [0.0f32; 4];
                let mut d = 0usize;
                while d < HEAD {
                    // SAFETY: key row key0 + bkey < KEY_TILE and words
                    // d/2 + 4·boct .. + 4 <= HEAD/2 are inside KT; every
                    // lane of the warp issues the load (the branch is on the
                    // warp index), after the barrier that staged it.
                    let bf = unsafe {
                        let bp = kt.add((key0 + bkey) * MMA_ROW_W + d / 2 + boct * 4);
                        wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            bp.cast_const().cast::<u8>(),
                        ))
                    };
                    // SAFETY: query row arow < MMA_ROWS and words d/2 +
                    // 4·ahalf .. + 4 + MMA_K/2 <= HEAD/2 are inside QT,
                    // published by the first barrier; the whole warp issues
                    // both loads and both `mma.sync` with its own fragments.
                    unsafe {
                        let ap0 = qt.add(arow * MMA_ROW_W + d / 2 + ahalf * (MMA_K / 4));
                        let af0 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap0.cast_const().cast::<u8>(),
                        ));
                        c = wmma::mma_m16n8k16_f32_f16(c, af0, [bf[0], bf[1]]);
                        let ap1 = ap0.add(MMA_K / 2);
                        let af1 = wmma::ldmatrix_x4_shared_u32(shared::cvta_generic_to_shared_u32(
                            ap1.cast_const().cast::<u8>(),
                        ));
                        c = wmma::mma_m16n8k16_f32_f16(c, af1, [bf[2], bf[3]]);
                    }
                    d += 2 * MMA_K;
                }
                // c[0], c[1]: head lane/4 at keys key0 + 2·(lane%4) + {0, 1};
                // c[2], c[3] are the zero rows.
                let g = lane / 4;
                let kk = key0 + 2 * (lane % 4);
                let s0 = if t0 + kk < hi {
                    scale * c[0]
                } else {
                    f32::NEG_INFINITY
                };
                let s1 = if t0 + kk + 1 < hi {
                    scale * c[1]
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: g < GROUP and kk + 1 < KEY_TILE: inside WS; each
                // (head, key) slot has one writer.
                unsafe {
                    *ws.add(g * KEY_TILE + kk) = s0;
                    *ws.add(g * KEY_TILE + kk + 1) = s1;
                }
            }
            thread::sync_threads();
            // SAFETY: w < GROUP and lane < KEY_TILE: inside WS, written
            // before the barrier above.
            let sc = unsafe { *ws.add(w * KEY_TILE + lane) };
            let live = t0 + lane < hi;
            warp::sync_mask(u32::MAX);
            // SAFETY: WS and VS are this block's tiles, VS staged before the
            // barriers above; w < GROUP and lane < 32.
            unsafe { fold_tile(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane) };
            t0 += KEY_TILE;
        }

        // SAFETY: as in gqa_flash_seg.
        unsafe {
            let o = idx * HEAD + 4 * lane;
            *part_v.get_unchecked_mut(o) = acc[0];
            *part_v.get_unchecked_mut(o + 1) = acc[1];
            *part_v.get_unchecked_mut(o + 2) = acc[2];
            *part_v.get_unchecked_mut(o + 3) = acc[3];
            if lane == 0 {
                *part_ms.get_unchecked_mut(2 * idx) = mx;
                *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
            }
        }
    }

    /// The merge: block `b = t·n_head + h` folds row `t`'s query head `h`'s
    /// partials of the segments that hold its keys — the first
    /// `ceil(n_keys / seg_keys)` of `segs`, with `n_keys` [`live_keys`] of
    /// `n_keys_buf[t]` as the segment pass reads it — in ascending order
    /// through `online_fold`, skipping one whose `Σ exp` is zero, and writes
    /// `y[(t·n_head + h)·HEAD + d] = r · (1/s)`, thread `d` owning dim `d`.
    /// A refused row raises [`FaultSite::KeyCount`] on `fault` and gets NaN
    /// in every dim.
    /// The partials are loaded [`MERGE_BATCH`] segments at a time ahead of
    /// their folds; the folds and their order are the one-at-a-time walk's.
    /// A segment past the row's last live one holds the neutral partial and
    /// is never read.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys_buf.len() >= m,
            segs * seg_keys >= ctx,
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn gqa_flash_merge(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let nh = n_head as usize;
        if b >= m as usize * nh {
            return; // block-uniform
        }
        let t = b / nh;
        let d = thread::threadIdx_x() as usize;
        let n_seg = segs as usize;
        // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
        let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx as usize);
        if limit == 0 {
            if d == 0 {
                fault.raise(FaultSite::KeyCount);
            }
            // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
            unsafe { *y.get_unchecked_mut(b * HEAD + d) = f32::NAN };
            return; // block-uniform: the row is the block's
        }
        let sk = seg_keys as usize;
        // limit <= ctx <= segs·seg_keys (launch contract), so live <= segs.
        let live = limit.div_ceil(sk);
        let mut mx = f32::NEG_INFINITY;
        let mut s = 0.0f32;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < live {
            // The batch's loads are issued ahead of its folds, so the walk
            // does not wait on memory twice per segment.
            let mut ms = [0.0f32; 2 * MERGE_BATCH];
            let mut vs = [0.0f32; MERGE_BATCH];
            let mut i = 0usize;
            #[unroll]
            while i < MERGE_BATCH {
                if j + i < live {
                    let idx = b * n_seg + j + i;
                    // SAFETY: idx < m·n_head·segs: both slots inside part_ms,
                    // and idx·HEAD + d < m·n_head·segs·HEAD <= part_v.len().
                    // A live segment wrote all three.
                    unsafe {
                        ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                        ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                        vs[i] = *part_v.get_unchecked(idx * HEAD + d);
                    }
                }
                i += 1;
            }
            let mut i = 0usize;
            #[unroll]
            while i < MERGE_BATCH {
                if j + i < live && ms[2 * i + 1] != 0.0 {
                    (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
                }
                i += 1;
            }
            j += MERGE_BATCH;
        }
        // SAFETY: b·HEAD + d < m·n_head·HEAD <= y.len(); thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = acc * (1.0 / s) };
    }
}

/// [`FlashGqaKernels::enqueue`]'s arguments: `m` rows of `n_kv · GROUP`
/// query heads (roped, unscaled, token-major), the layer's two planes of
/// `n_kv · ctx` rows, each row's live key count on the device (`m` of them),
/// the partials scratch ([`partials_v_len`], [`partials_ms_len`] of `m`
/// rows), the sink a refused count raises on, and the output rows,
/// token-major.
pub struct GqaArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct FlashGqaKernels {
    module: flash_gqa_kernels::LoadedModule,
}

impl FlashGqaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashGqaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { flash_gqa_kernels::load(ctx)? };
        Ok(FlashGqaKernels { module })
    }

    /// Enqueue `m` rows' attention through the pass [`gqa_mma`] names. Two
    /// launches. Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, args: GqaArgs<'_>) -> Result<(), GpuError> {
        self.enqueue_pass(stream, args, gqa_mma())
    }

    /// Enqueue `m` rows' attention: the segment pass (`m · n_kv ·
    /// segments_for(ctx)` blocks; the tensor-core one when `mma`) and the
    /// merge (`m · n_kv · GROUP` blocks). The segments cover the whole
    /// cache, so every count the kernels accept is walked in full; a count
    /// of zero or past `ctx` raises [`FaultSite::KeyCount`] on `args.fault`
    /// and its row is NaN. Two launches. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_pass(
        &self,
        stream: &CudaStream,
        args: GqaArgs<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa::enqueue";
        let GqaArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let n_head = n_kv * GROUP;
        let segs = segments_for(ctx);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kc", kc.len(), n_kv * ctx * HEAD),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), m),
            ("part_v", part_v.len(), partials_v_len(m, n_head, ctx)),
            ("part_ms", part_ms.len(), partials_ms_len(m, n_head, ctx)),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        let n_kv = launch_u32(what, "n_kv", n_kv)?;
        let ctx = launch_u32(what, "ctx", ctx)?;
        let segs = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m = launch_u32(what, "m", m)?;
        let cfg = LaunchConfig1D::new(grid, THREADS_U32, 0);
        if mma {
            let prep = self.module.prepare_gqa_flash_seg_mma(cfg)?;
            self.module.gqa_flash_seg_mma(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        } else {
            let prep = self.module.prepare_gqa_flash_seg(cfg)?;
            self.module.gqa_flash_seg(
                stream, &prep, q, kc, vc, n_keys, scale, n_kv, ctx, segs, seg_keys, m, part_v,
                part_ms,
            )?;
        }
        let prep = self.module.prepare_gqa_flash_merge(LaunchConfig1D::new(
            merge_grid,
            MERGE_THREADS,
            0,
        ))?;
        self.module.gqa_flash_merge(
            stream, &prep, part_v, part_ms, n_keys, heads, ctx, segs, seg_keys, m, fault, y,
        )?;
        Ok(())
    }
}
