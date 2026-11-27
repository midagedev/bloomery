//! Rel-bias attention over the head-128 segment walk: the entries an
//! architecture whose scores carry a distance-gathered bias row runs at a
//! group of [`GROUP_3`] (Inkling 2 Mini's 24 over 8) or [`GROUP_6`] (the Small's
//! 48 over 8) query heads a key head.
//!
//! The walk, the tiles and the softmax are `flash_gqa_h128_swa`'s — the same
//! [`seg_span`] window cut, the same [`fold_tile`] step, one warp a query
//! head, the partials `(max, Σ exp, Σ exp·V)` a segment — with two facts of
//! the bias laid in:
//!
//! - the score of key `j` gains `bⱼ = Σ_{d < REL} rel_bias[d, distⱼ] · r_d`,
//!   `distⱼ = live − 1 − key position`. Lane `l` of a tile owns key `l`, so it
//!   reads its own `REL` table entries straight from global memory (the
//!   positions of a tile are consecutive, so a warp's read of one table row is
//!   one coalesced line), one tile ahead of the score that uses them. A
//!   distance at or past `extent` reads 0 — the reference's pad column, which
//!   the caller does not upload. The `REL` values of `r` and the row's sink `t`
//!   are plain launch data, computed upstream;
//! - the sink joins the softmax as the partial `(t, 1, 0)` folded last by
//!   the merge — one more logit in the denominator, nothing in the value —
//!   read per (row, head), so the merge is this file's own: a sink that
//!   varies a row is the whole difference from
//!   `gqa_flash_merge_h128_swa`, whose sink is one value a head.
//!
//! `exp(s − t) / (1 + Σ exp(s_k − t))` and this file's `exp(s) / (exp(t) +
//! Σ exp(s_k))` are one distribution; the rounding points differ, and the
//! owning gate bands that distance. The segment body is this file's own
//! copy of the walk, not a `seg_h128` instance: the block's key and value
//! tiles are staged by 16-byte `cp.async` copies, a tile ahead of the one
//! being scored and two tiles in flight at the start of a segment, so a
//! segment of a few tiles costs one memory round trip, not one a tile. The
//! plane and ring forms and the two groups are one body under `RING` and
//! `G`, as there; the merge takes the head count as data and serves both
//! groups.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{MERGE_BATCH, half_bits_to_f32, online_fold};
use crate::flash_gqa::{HEAD, KEY_TILE, SEG_KEYS, seg_span, window_cut, window_segments};
use crate::flash_gqa_h128_swa::{
    GROUP_3, GROUP_6, MERGE_THREADS, ROW_QWORDS, fold_tile, live_keys,
};
use crate::launch_u32;
use crate::tensor::check_alignment;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::vector::{F32x4, U32x4};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Relative dimensions a head: the width of `r` and of the bias table's
/// first axis.
pub const REL: usize = 16;
const THREADS_G3_U32: u32 = (GROUP_3 * 32) as u32;
const THREADS_G6_U32: u32 = (GROUP_6 * 32) as u32;
// The entries' launch contracts spell the groups and their blocks out as 3 and
// 96 threads, 6 and 192.
const _: () =
    assert!(GROUP_3 == 3 && GROUP_6 == 6 && THREADS_G3_U32 == 96 && THREADS_G6_U32 == 192);
const _: () = assert!(KEY_TILE == 32 && REL <= 32);

/// u32 words of one key row, the span the score loop's rounds walk.
const ROW_WORDS: usize = HEAD / 2;
/// 16-byte chunks of one key or value row, each one `cp.async` copy.
const ROW_CHUNKS: usize = ROW_WORDS / 4;
/// u32 words a staged key row takes: the row and one chunk of padding, so the
/// eight lanes of a quarter-warp's 16-byte read land on eight distinct bank
/// groups (lane `j` reads at chunk `17·j + c`, and 17 is odd).
const K_ROW: usize = ROW_WORDS + 4;
/// u32 words of one staged key tile, and of one value tile (`ROW_QWORDS` u64 a
/// row).
const K_TILE_WORDS: usize = KEY_TILE * K_ROW;
const V_TILE_WORDS: usize = KEY_TILE * ROW_QWORDS * 2;
/// Tiles staged at once: the one being scored and the one after it.
const STAGES: usize = 2;
const _: () =
    assert!(ROW_WORDS.is_multiple_of(4) && ROW_CHUNKS == 16 && ROW_QWORDS * 2 == ROW_WORDS);
const _: () = assert!(K_ROW % 4 == 0 && (K_ROW / 4) % 2 == 1);

/// Rows of a tile one pass of a block's copy threads covers (`32·G` threads,
/// [`ROW_CHUNKS`] a row), and the passes a tile takes.
const fn rows_a_pass(g: usize) -> usize {
    2 * g
}
const fn copy_passes(g: usize) -> usize {
    KEY_TILE.div_ceil(rows_a_pass(g))
}

/// Stage the key and value tile whose first key is `t0` into `dk` (stride
/// [`K_ROW`]) and `dv` (a row is [`ROW_WORDS`] words), one commit group: row
/// `r` is key `t0 + r`, held by the 16-byte copies of the block's threads,
/// every chunk of a row by its own thread; a row at or past `hi` is zero. `sk`
/// and `sv` are the key head's plane base, and on a ring (`RING`) a key's row
/// is its position modulo `plane_rows`, found by stepping from the first
/// thread row's slot (`plane_rows >= KEY_TILE`, so one subtraction wraps it).
///
/// Contract (the caller keeps it; rust-quality R28): `dk` and `dv` are the
/// block's [`STAGES`] shared tiles (16-byte aligned, [`K_TILE_WORDS`] and
/// [`V_TILE_WORDS`] words each) and `slot < STAGES` the one filled, which no
/// warp reads until the wait and barrier that follow this commit; `sk` and
/// `sv` are 16-byte aligned planes in which every key of `[t0, hi)` has its
/// row (below `plane_rows` on a plane, any key on a ring); the block is
/// `32·G` threads and `tid` the caller's thread index.
#[inline(always)]
fn stage_tile<const RING: bool, const G: usize>(
    (dk, dv): (*mut u32, *mut u32),
    (sk, sv): (*const u32, *const u32),
    slot: usize,
    t0: usize,
    hi: usize,
    plane_rows: u32,
    tid: usize,
) {
    let chunk = tid % ROW_CHUNKS;
    let mut row = tid / ROW_CHUNKS;
    let first = (t0 + row) as u32;
    let mut at = if RING { first % plane_rows } else { first };
    let mut pass = 0usize;
    while pass < copy_passes(G) {
        cuda_device::thread::__unroll_config::<0>();
        if row < KEY_TILE {
            // SAFETY: row < KEY_TILE and chunk < ROW_CHUNKS: this thread's
            // 16 aligned bytes of each tile, which no other thread names.
            unsafe {
                let k_at = dk.add(slot * K_TILE_WORDS + row * K_ROW + 4 * chunk);
                let v_at = dv.add(slot * V_TILE_WORDS + row * ROW_WORDS + 4 * chunk);
                if t0 + row < hi {
                    // The key's row is inside the plane (the caller's
                    // contract).
                    let from = at as usize * ROW_WORDS + 4 * chunk;
                    cp_async_cg_16(k_at, sk.add(from));
                    cp_async_cg_16(v_at, sv.add(from));
                } else {
                    *k_at.cast::<U32x4>() = U32x4::splat(0);
                    *v_at.cast::<U32x4>() = U32x4::splat(0);
                }
            }
        }
        row += rows_a_pass(G);
        at += rows_a_pass(G) as u32;
        if RING && at >= plane_rows {
            at -= plane_rows;
        }
        pass += 1;
    }
    // SAFETY: closes this thread's copies above as one group.
    unsafe { cp_async_commit_group() };
}

/// Lane `lane`'s `REL` bias values of the tile starting at `t0`: the table
/// column of the lane's key distance, all 0 for a key the segment does not
/// hold and past the extent (the pad rule).
///
/// Contract (the caller keeps it; rust-quality R28): `bias` holds `REL · extent` values and `hi <= limit`, so a held
/// key's distance is below `limit` (the launch contract).
#[inline(always)]
fn bias_lane(
    bias: &[f32],
    (t0, hi, limit, extent): (usize, usize, usize, usize),
    lane: usize,
) -> [f32; REL] {
    let mut out = [0.0f32; REL];
    let key = t0 + lane;
    if key < hi {
        let dist = limit - 1 - key;
        if dist < extent {
            for d in 0..REL {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: d·extent + dist < REL·extent <= bias.len().
                out[d] = unsafe { *bias.get_unchecked(d * extent + dist) };
            }
        }
    }
    out
}

/// The segment pass of this file's entries: `seg_h128`'s walk at `G`
/// ([`GROUP_3`] or [`GROUP_6`]) query heads a key head with the bias laid
/// into the score, on a plane (`RING` false) or a ring (`RING` true). Block
/// `b = (seg·n_kv + kh)·m + t`, warp `w` is query head `kh·G + w`; the
/// partials, the window cut and the neutral-segment rule are `seg_h128`'s.
///
/// The score's order is the gate: the f32 dot of `seg_h128` (its rotating
/// ILP partials, combined `(a0 + a1) + (a2 + a3)`), one multiply by
/// `scale`, then the bias dot added — `REL` fused multiply-adds ascending
/// `d`, the r row one value a lane, read by shuffle. The tiles fold ascending
/// through [`fold_tile`].
///
/// SAFETY: the entry's launch contract at `G` query heads a key head
/// (every buffer bound it names, the planes of `n_kv · plane_rows`
/// rows 16-byte aligned, and under `RING` `plane_rows >= KEY_TILE` with every
/// key the row reads resident in the ring), a block of `G · 32` threads, and
/// the four tiles this block's shared memory as the entry declared them: the
/// group's query rows (`G · HEAD` f32, 16-byte aligned), [`STAGES`] key tiles
/// of [`K_TILE_WORDS`] words and value tiles of [`V_TILE_WORDS`] words (16-byte
/// aligned), and the weight tile (`G · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
pub(crate) unsafe fn seg_h128_rel<const RING: bool, const G: usize>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    r: &[f32],
    bias: &[f32],
    extent: u32,
    scale: f32,
    n_kv: u32,
    ctx: u32,
    plane_rows: u32,
    segs: u32,
    seg_keys: u32,
    m: u32,
    window: u32,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qs: *mut f32,
    ks: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(G == GROUP_3 || G == GROUP_6) };
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
    let n_head = nkv * G;
    let h = kh * G + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let (first, n_w) = window_cut(limit, window as usize);
    let span = seg_span(n_w, n_seg, seg_keys as usize);
    let lo = first + seg * span;
    if lo >= first + n_w {
        if lane == 0 {
            // SAFETY: idx < m·n_kv·G·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: the window cut is the block's
    }
    let hi = (lo + span).min(first + n_w);
    let n_tiles = (hi - lo).div_ceil(KEY_TILE);
    let extent = extent as usize;

    // The key head's plane in u32 words, and the first two tiles in flight
    // before anything else is read: tile `i` lives in buffer `i % STAGES`.
    let plane = kh * plane_rows as usize * ROW_WORDS;
    // SAFETY: the plane is inside kc and vc by the launch contract.
    let (sk_base, sv_base) = unsafe {
        (
            (kc.as_ptr() as *const u32).add(plane),
            (vc.as_ptr() as *const u32).add(plane),
        )
    };
    let vs32 = vs as *mut u32;
    // The first tile of the segment in buffer 0 and the second in buffer 1.
    let planes = (sk_base, sv_base);
    stage_tile::<RING, G>((ks, vs32), planes, 0, lo, hi, plane_rows, tid);
    if n_tiles > 1 {
        stage_tile::<RING, G>((ks, vs32), planes, 1, lo + KEY_TILE, hi, plane_rows, tid);
    }
    let mut bias_now = bias_lane(bias, (lo, hi, limit, extent), lane);

    let qb = (t * n_head + kh * G) * HEAD;
    let mut i = 0usize;
    while i < 4 {
        // SAFETY: qb + 4·tid + i is inside row t's group, and that row is
        // inside q by the launch contract. The shared index is inside
        // G·HEAD, which 4·(G·32) equals.
        unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
        i += 1;
    }
    // Lane l holds the head's r value l mod REL; the bias dot reads value d
    // from lane d by a shuffle, so the row stays in one register a lane.
    // SAFETY: (t·n_head + h)·REL + lane mod REL is inside r by the launch
    // contract.
    let r_lane = unsafe { *r.get_unchecked((t * n_head + h) * REL + lane % REL) };

    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut tile = 0usize;
    while tile < n_tiles {
        let t0 = lo + tile * KEY_TILE;
        // SAFETY: waits for this thread's copies but the newest tile's, and
        // the barrier publishes every thread's copies (and the query rows,
        // on the first tile); every warp is past the tile that last shared
        // the buffer the next tile's copies are about to fill.
        unsafe {
            if tile + 1 < n_tiles {
                cp_async_wait_group(1);
            } else {
                cp_async_wait_group(0);
            }
        }
        thread::sync_threads();
        // The next tile's bias values are in flight under this tile's score.
        let bias_next = if tile + 1 < n_tiles {
            bias_lane(bias, (t0 + KEY_TILE, hi, limit, extent), lane)
        } else {
            [0.0f32; REL]
        };

        let live = t0 + lane < hi;
        let mut a = [0.0f32; 4];
        // SAFETY: lane < 32 and the buffer is this tile's, published by the
        // barrier above: the key row at stride K_ROW (16-byte aligned, so is
        // each chunk read), the query row of warp w (G·HEAD f32, chunks of
        // four).
        unsafe {
            let krow = ks
                .add((tile % STAGES) * K_TILE_WORDS + lane * K_ROW)
                .cast::<U32x4>();
            let qrow = qs.add(w * HEAD).cast::<F32x4>();
            let mut c = 0usize;
            while c < ROW_CHUNKS {
                let kw = (*krow.add(c)).0;
                let (q_lo, q_hi) = ((*qrow.add(2 * c)).0, (*qrow.add(2 * c + 1)).0);
                let qv = [
                    q_lo[0], q_lo[1], q_lo[2], q_lo[3], q_hi[0], q_hi[1], q_hi[2], q_hi[3],
                ];
                let mut p = 0usize;
                while p < 4 {
                    let k0 = half_bits_to_f32(kw[p] as u16);
                    let k1 = half_bits_to_f32((kw[p] >> 16) as u16);
                    a[p] = f32::mul_add(qv[2 * p], k0, a[p]);
                    a[p] = f32::mul_add(qv[2 * p + 1], k1, a[p]);
                    p += 1;
                }
                c += 1;
            }
        }
        let dot = (a[0] + a[1]) + (a[2] + a[3]);
        // The bias dot's order is the gate: REL fused multiply-adds
        // ascending d against the lane's own table entries.
        let mut bsum = 0.0f32;
        let mut d = 0usize;
        while d < REL {
            let rd = warp::shuffle_f32(r_lane, d as u32);
            bsum = f32::mul_add(rd, bias_now[d], bsum);
            d += 1;
        }
        let sc = if live {
            dot * scale + bsum
        } else {
            f32::NEG_INFINITY
        };

        // SAFETY: the weight tile is this block's, the value tile of this
        // tile's buffer was published by the barrier above, and w < G.
        unsafe {
            fold_tile(
                sc,
                live,
                &mut mx,
                &mut s_sum,
                &mut acc,
                ws,
                vs.add((tile % STAGES) * (V_TILE_WORDS / 2)),
                w,
                lane,
            )
        };
        bias_now = bias_next;
        if tile + STAGES < n_tiles {
            // Every warp is done with this tile's buffers before the tile
            // two ahead is copied over them.
            thread::sync_threads();
            stage_tile::<RING, G>(
                (ks, vs32),
                planes,
                tile % STAGES,
                t0 + STAGES * KEY_TILE,
                hi,
                plane_rows,
                tid,
            );
        }
        tile += 1;
    }

    // SAFETY: idx is inside the partials. The four dims 4·lane .. +3 are
    // this lane's alone, and lane 0 writes (max, sum).
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

#[cuda_module]
mod rel_bias_kernels {
    use super::*;

    /// Segment pass on a plane (`seg_h128_rel`): a block of 96 threads,
    /// warp `w` is query head `kh·3 + w`.
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
            n_keys_buf.len() >= m,
            segs >= 1,
            extent >= 1,
            q.len() >= m * n_kv * 3 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            r.len() >= m * n_kv * 3 * 16,
            bias.len() >= 16 * extent,
            part_v.len() >= m * n_kv * 3 * segs * 128,
            part_ms.len() >= m * n_kv * 3 * segs * 2
        )
    )]
    pub fn rel_seg_h128(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP_3 * HEAD }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { STAGES * K_TILE_WORDS }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { STAGES * V_TILE_WORDS / 2 }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_3 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_3 query heads and the two staged tiles; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s on a plane, and the block is GROUP_3 · 32
        // threads.
        unsafe {
            seg_h128_rel::<false, GROUP_3>(
                q,
                kc,
                vc,
                n_keys_buf,
                r,
                bias,
                extent,
                scale,
                n_kv,
                ctx,
                ctx,
                segs,
                seg_keys,
                m,
                window,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`rel_seg_h128`] on a ring of `ring_rows` rows (`seg_h128_rel` under
    /// `RING`): position `p` is row `p mod ring_rows`, `ctx` stays the bound
    /// of the live count.
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
            n_keys_buf.len() >= m,
            segs >= 1,
            extent >= 1,
            ring_rows >= 32,
            q.len() >= m * n_kv * 3 * 128,
            kc.len() >= n_kv * ring_rows * 128,
            vc.len() >= n_kv * ring_rows * 128,
            r.len() >= m * n_kv * 3 * 16,
            bias.len() >= 16 * extent,
            part_v.len() >= m * n_kv * 3 * segs * 128,
            part_ms.len() >= m * n_kv * 3 * segs * 2
        )
    )]
    pub fn rel_seg_h128_ring(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        ring_rows: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP_3 * HEAD }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { STAGES * K_TILE_WORDS }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { STAGES * V_TILE_WORDS / 2 }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_3 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_3 query heads and the two staged tiles; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s under RING, and the block is GROUP_3 · 32
        // threads.
        unsafe {
            seg_h128_rel::<true, GROUP_3>(
                q,
                kc,
                vc,
                n_keys_buf,
                r,
                bias,
                extent,
                scale,
                n_kv,
                ctx,
                ring_rows,
                segs,
                seg_keys,
                m,
                window,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`rel_seg_h128`] at [`GROUP_6`] query heads a key head: a block of
    /// 192 threads, warp `w` is query head `kh·6 + w`.
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
            n_keys_buf.len() >= m,
            segs >= 1,
            extent >= 1,
            q.len() >= m * n_kv * 6 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            r.len() >= m * n_kv * 6 * 16,
            bias.len() >= 16 * extent,
            part_v.len() >= m * n_kv * 6 * segs * 128,
            part_ms.len() >= m * n_kv * 6 * segs * 2
        )
    )]
    pub fn rel_seg_h128_g6(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP_6 * HEAD }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { STAGES * K_TILE_WORDS }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { STAGES * V_TILE_WORDS / 2 }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_6 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_6 query heads and the two staged tiles; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s on a plane, and the block is GROUP_6 · 32
        // threads.
        unsafe {
            seg_h128_rel::<false, GROUP_6>(
                q,
                kc,
                vc,
                n_keys_buf,
                r,
                bias,
                extent,
                scale,
                n_kv,
                ctx,
                ctx,
                segs,
                seg_keys,
                m,
                window,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// [`rel_seg_h128_g6`] on a ring of `ring_rows` rows (`seg_h128_rel` under
    /// `RING`): position `p` is row `p mod ring_rows`, `ctx` stays the bound
    /// of the live count.
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
            n_keys_buf.len() >= m,
            segs >= 1,
            extent >= 1,
            ring_rows >= 32,
            q.len() >= m * n_kv * 6 * 128,
            kc.len() >= n_kv * ring_rows * 128,
            vc.len() >= n_kv * ring_rows * 128,
            r.len() >= m * n_kv * 6 * 16,
            bias.len() >= 16 * extent,
            part_v.len() >= m * n_kv * 6 * segs * 128,
            part_ms.len() >= m * n_kv * 6 * segs * 2
        )
    )]
    pub fn rel_seg_h128_ring_g6(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        r: &[f32],
        bias: &[f32],
        extent: u32,
        scale: f32,
        n_kv: u32,
        ctx: u32,
        ring_rows: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP_6 * HEAD }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { STAGES * K_TILE_WORDS }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { STAGES * V_TILE_WORDS / 2 }, 16> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_6 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_6 query heads and the two staged tiles; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s under RING, and the block is GROUP_6 · 32
        // threads.
        unsafe {
            seg_h128_rel::<true, GROUP_6>(
                q,
                kc,
                vc,
                n_keys_buf,
                r,
                bias,
                extent,
                scale,
                n_kv,
                ctx,
                ring_rows,
                segs,
                seg_keys,
                m,
                window,
                part_v,
                part_ms,
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
    }

    /// The merge: the segments folded ascending through `online_fold`, then
    /// row `t`, head `h`'s sink as the partial `(sink, 1, 0)` — one more
    /// logit in the denominator, nothing in the value — and `acc · (1/s)`.
    /// Segments ascending, then the sink: this order is the gate. A sink of
    /// `−inf` is neutral; a NaN or `+inf` sink has no defined softmax and
    /// writes NaN into the row. A refused row raises
    /// [`FaultSite::KeyCount`] and writes NaN in every dim.
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
            segs >= 1,
            sinks.len() >= m * n_head,
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn rel_merge_h128(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        sinks: &[f32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
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
            // SAFETY: b·HEAD + d is inside y; thread d owns it.
            unsafe { *y.get_unchecked_mut(b * HEAD + d) = f32::NAN };
            return; // block-uniform: the row is the block's
        }
        let (_, n_w) = window_cut(limit, window as usize);
        let live = n_w.div_ceil(seg_span(n_w, n_seg, seg_keys as usize));
        let mut mx = f32::NEG_INFINITY;
        let mut s = 0.0f32;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < live {
            let mut ms = [0.0f32; 2 * MERGE_BATCH];
            let mut vs = [0.0f32; MERGE_BATCH];
            for i in 0..MERGE_BATCH {
                cuda_device::thread::__unroll_config::<0>();
                if j + i < live {
                    let idx = b * n_seg + j + i;
                    // SAFETY: idx is inside the row's segments, so both
                    // partials and the value dim are inside the buffers.
                    unsafe {
                        ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                        ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                        vs[i] = *part_v.get_unchecked(idx * HEAD + d);
                    }
                }
            }
            for i in 0..MERGE_BATCH {
                cuda_device::thread::__unroll_config::<0>();
                if j + i < live && ms[2 * i + 1] != 0.0 {
                    (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
                }
            }
            j += MERGE_BATCH;
        }
        // The sink joins last, after every live segment. It is row t, head
        // h's: `b` is that pair's index in `sinks`.
        // SAFETY: b < m·n_head <= sinks.len() by the launch contract.
        let sink = unsafe { *sinks.get_unchecked(b) };
        let (_, s, acc) = online_fold(mx, s, acc, sink, 1.0, 0.0);
        let out = if sink.is_nan() || sink == f32::INFINITY {
            f32::NAN
        } else {
            acc * (1.0 / s)
        };
        // SAFETY: b·HEAD + d is inside y; thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = out };
    }
}

/// Arguments of one rel-bias attention launch. `window` 0 attends every
/// live key. `bias` is the `[REL][extent]` f32 table, row-major; a distance
/// at or past `extent` needs no column. `r` holds `REL` values a (row,
/// head). `sinks` holds one f32 a (row, head). `ring_rows` `None` runs the
/// plane entries; `Some(rows)` the ring entries over a ring of `rows` rows
/// a key head. The partials hold [`window_segments`]`(window)` segments a
/// head.
pub struct RelBiasArgs<'a> {
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
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct RelBiasKernels {
    module: rel_bias_kernels::LoadedModule,
}

impl RelBiasKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RelBiasKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(rel_bias_kernels, ctx)? };
        Ok(RelBiasKernels { module })
    }

    /// Enqueue `m` rows at [`GROUP_3`] or [`GROUP_6`] query heads a key head
    /// (`n_head` must equal `3 · n_kv` or `6 · n_kv`; any other ratio is
    /// refused by name, as a non-finite scale, a zero extent or count, a
    /// ring of fewer rows than a key tile, a key or value plane not 16-byte
    /// aligned or a short buffer is). A
    /// live count of zero or past `ctx` raises [`FaultSite::KeyCount`] and
    /// the row is NaN.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        args: RelBiasArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "rel_bias::enqueue";
        let RelBiasArgs {
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
            part_v,
            part_ms,
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
        let group = if n_head == n_kv * GROUP_3 {
            GROUP_3
        } else if n_head == n_kv * GROUP_6 {
            GROUP_6
        } else {
            return Err(GpuError::shape(
                what,
                format!(
                    "the rel-bias kernels take {GROUP_3} or {GROUP_6} query heads a key head; \
                     got {n_head} heads over {n_kv}"
                ),
            ));
        };
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("the score scale must be finite, got {scale}"),
            ));
        }
        if let Some(rows) = ring_rows.filter(|&rows| rows < KEY_TILE) {
            return Err(GpuError::shape(
                what,
                format!("a ring holds at least a key tile of {KEY_TILE} rows, got {rows}"),
            ));
        }
        // The tiles are staged by 16-byte copies.
        check_alignment::<U32x4, _>(what, "kc", kc)?;
        check_alignment::<U32x4, _>(what, "vc", vc)?;
        let plane_rows = ring_rows.unwrap_or(ctx);
        let segs = window_segments(window);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kc", kc.len(), n_kv * plane_rows * HEAD),
            ("vc", vc.len(), n_kv * plane_rows * HEAD),
            ("n_keys", n_keys.len(), m),
            ("r", r.len(), m * n_head * REL),
            ("bias", bias.len(), REL * extent),
            ("sinks", sinks.len(), m * n_head),
            ("part_v", part_v.len(), m * n_head * segs * HEAD),
            ("part_ms", part_ms.len(), m * n_head * segs * 2),
            ("y", y.len(), m * n_head * HEAD),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let n_kv_u = launch_u32(what, "n_kv", n_kv)?;
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let extent_u = launch_u32(what, "extent", extent)?;
        let ring_u = launch_u32(what, "ring_rows", plane_rows)?;
        let segs_u = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m_u = launch_u32(what, "m", m)?;
        let window_u = launch_u32(what, "window", window)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        match (group, ring_rows.is_some()) {
            (GROUP_3, true) => {
                let prep = self.module.prepare_rel_seg_h128_ring(LaunchConfig1D::new(
                    grid,
                    THREADS_G3_U32,
                    0,
                ))?;
                self.module.rel_seg_h128_ring(
                    stream,
                    &prep,
                    q,
                    kc,
                    vc,
                    n_keys,
                    r,
                    bias,
                    extent_u,
                    scale,
                    n_kv_u,
                    ctx_u,
                    ring_u,
                    segs_u,
                    seg_keys,
                    m_u,
                    window_u,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
            (GROUP_3, false) => {
                let prep = self.module.prepare_rel_seg_h128(LaunchConfig1D::new(
                    grid,
                    THREADS_G3_U32,
                    0,
                ))?;
                self.module.rel_seg_h128(
                    stream,
                    &prep,
                    q,
                    kc,
                    vc,
                    n_keys,
                    r,
                    bias,
                    extent_u,
                    scale,
                    n_kv_u,
                    ctx_u,
                    segs_u,
                    seg_keys,
                    m_u,
                    window_u,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
            (_, true) => {
                let prep = self
                    .module
                    .prepare_rel_seg_h128_ring_g6(LaunchConfig1D::new(grid, THREADS_G6_U32, 0))?;
                self.module.rel_seg_h128_ring_g6(
                    stream,
                    &prep,
                    q,
                    kc,
                    vc,
                    n_keys,
                    r,
                    bias,
                    extent_u,
                    scale,
                    n_kv_u,
                    ctx_u,
                    ring_u,
                    segs_u,
                    seg_keys,
                    m_u,
                    window_u,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
            (_, false) => {
                let prep = self.module.prepare_rel_seg_h128_g6(LaunchConfig1D::new(
                    grid,
                    THREADS_G6_U32,
                    0,
                ))?;
                self.module.rel_seg_h128_g6(
                    stream,
                    &prep,
                    q,
                    kc,
                    vc,
                    n_keys,
                    r,
                    bias,
                    extent_u,
                    scale,
                    n_kv_u,
                    ctx_u,
                    segs_u,
                    seg_keys,
                    m_u,
                    window_u,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
        }
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let merge = LaunchConfig1D::new(merge_grid, MERGE_THREADS, 0);
        let prep = self.module.prepare_rel_merge_h128(merge)?;
        self.module.rel_merge_h128(
            stream, &prep, part_v, part_ms, n_keys, sinks, heads, ctx_u, segs_u, seg_keys, m_u,
            window_u, fault, &mut *y,
        )?;
        Ok(())
    }
}
