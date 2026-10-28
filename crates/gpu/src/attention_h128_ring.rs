//! Ring-addressed head-128 primitives. Logical positions index RoPE; only KV addresses wrap.

use crate::GpuError;
use crate::elem::rms_scale_of_mean_checked;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::f32_to_f16_bits;
use crate::flash::{MERGE_BATCH, half_bits_to_f32, online_fold};
use crate::flash_gqa::{
    GROUP, HEAD, KEY_TILE, SEG_KEYS, SEGMENTS, merge_sink_body, seg_span, window_cut,
    window_segments,
};
use crate::flash_gqa_h128_swa::{GROUP_4, H128SwaArgs, fold_tile, served_group};
use crate::launch_u32;
use crate::rope_neox::NeoxArgs;
use crate::rope_neox::neox_pair;
use cuda_core::{CudaContext, CudaStream, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// u32 words of one key row, and the padded stride the key tile is staged at.
const ROW_WORDS: usize = HEAD / 2;
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of one value row.
const ROW_QWORDS: usize = HEAD / 4;
const ILP: usize = 4;
const THREADS: usize = GROUP * 32;
const THREADS_U32: u32 = THREADS as u32;
const THREADS_G4_U32: u32 = (GROUP_4 * 32) as u32;
const MERGE_THREADS: u32 = HEAD as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && MERGE_THREADS as usize == HEAD);
// The entries' launch contracts spell GROUP and GROUP_4 out as 8 and 4, and
// their blocks as 256 and 128 threads.
const _: () = assert!(GROUP == 8 && GROUP_4 == 4 && THREADS_U32 == 256 && THREADS_G4_U32 == 128);
const _: () = assert!(ROW_QWORDS == 32 && ROW_WORDS.is_multiple_of(ILP) && K_STRIDE % 2 == 1);
// The launch's window: 513 positions is nine 64-key segments, and a row of
// 600 live keys starts at key 87.
const _: () = {
    assert!(window_segments(0) == SEGMENTS && window_segments(513) == 9);
    let (first, n_w) = window_cut(600, 513);
    assert!(first == 87 && n_w == 513 && first + n_w == 600);
};

#[inline(always)]
fn live_keys(count: u32, ctx: usize) -> usize {
    let c = count as usize;
    if c > ctx { 0 } else { c }
}

#[cuda_module]
mod append_kernels {
    use super::*;
    const THREADS: usize = 64;
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            rows >= 1,
            rows <= ctx,
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * rows * 128,
            cache_v.len() >= n_kv * rows * 128
        )
    )]
    pub fn qk_rms_append_ring(
        gq: &[f32],
        gk: &[f32],
        pos: &[u32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let t = b / heads;
        let h = b - t * heads;
        let tid = thread::threadIdx_x() as usize;
        let is_q = h < n_head as usize;
        let kh = h.wrapping_sub(n_head as usize);
        let base = if is_q {
            (t * n_head as usize + h) * HEAD
        } else {
            (t * n_kv as usize + kh) * HEAD
        };
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;

        // SAFETY: base + tid + 64 is inside the head's buffer by the launch
        // contract. One thread per value pair.
        let (x0, x1) = unsafe {
            if is_q {
                (
                    *q.get_unchecked_mut(base + tid),
                    *q.get_unchecked_mut(base + tid + THREADS),
                )
            } else {
                (
                    *k.get_unchecked_mut(base + tid),
                    *k.get_unchecked_mut(base + tid + THREADS),
                )
            }
        };
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        // SAFETY: WSUM is this block's own shared allocation. Two slots, one
        // per warp, written before the barrier that publishes them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 2. One lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: both slots were written before the barrier above.
        let sum = unsafe { *ws.add(0) + *ws.add(1) };
        let mean = (sum / HEAD as f64) as f32;
        let scale = rms_scale_of_mean_checked(mean, eps, fault, tid == 0);

        if p >= ctx as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            // SAFETY: the positions read above. This thread owns them.
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(base + tid) = f32::NAN;
                    *q.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                } else {
                    *k.get_unchecked_mut(base + tid) = f32::NAN;
                    *k.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                }
            }
            return; // block-uniform: p is the token's
        }
        // SAFETY: tid + 64 < 128 <= the gain's length.
        let (g0, g1) = unsafe {
            let g = if is_q { gq } else { gk };
            (*g.get_unchecked(tid), *g.get_unchecked(tid + THREADS))
        };
        let n0 = (scale * g0) * x0;
        let n1 = (scale * g1) * x1;

        if is_q {
            // SAFETY: the positions read above. This thread owns them.
            unsafe {
                *q.get_unchecked_mut(base + tid) = n0;
                *q.get_unchecked_mut(base + tid + THREADS) = n1;
            }
            return;
        }
        // SAFETY: the positions read above. This thread owns them.
        unsafe {
            *k.get_unchecked_mut(base + tid) = n0;
            *k.get_unchecked_mut(base + tid + THREADS) = n1;
        }
        let row = (kh * rows as usize + p % rows as usize) * HEAD;
        // SAFETY: kh < n_kv and p % rows < rows, so the row is inside both rings.
        // base + tid + 64 is inside v. One thread per pair. The tokens of
        // one launch hold distinct positions, so no two blocks write one row.
        unsafe {
            *cache_k.get_unchecked_mut(row + tid) = f32_to_f16_bits(n0);
            *cache_k.get_unchecked_mut(row + tid + THREADS) = f32_to_f16_bits(n1);
            *cache_v.get_unchecked_mut(row + tid) = f32_to_f16_bits(*v.get_unchecked(base + tid));
            *cache_v.get_unchecked_mut(row + tid + THREADS) =
                f32_to_f16_bits(*v.get_unchecked(base + tid + THREADS));
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        requires = (
            rows >= 1,
            rows <= ctx,
            q.len() >= m * n_head * 128,
            k.len() >= m * n_kv * 128,
            v.len() >= m * n_kv * 128,
            gq.len() >= 128,
            gk.len() >= 128,
            table.len() >= ctx * 128,
            pos.len() >= m,
            cache_k.len() >= n_kv * rows * 128,
            cache_v.len() >= n_kv * rows * 128
        )
    )]
    pub fn head_norm_neox_append_ring(
        gq: &[f32],
        gk: &[f32],
        table: &[f32],
        pos: &[u32],
        v: &[f32],
        eps: f32,
        n_head: u32,
        n_kv: u32,
        ctx: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        mut cache_k: DisjointSlice<u16>,
        mut cache_v: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f64, 2> = SharedArray::UNINIT;

        let heads = (n_head + n_kv) as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * heads {
            return; // block-uniform
        }
        let t = b / heads;
        let h = b - t * heads;
        let tid = thread::threadIdx_x() as usize;
        let is_q = h < n_head as usize;
        let kh = h.wrapping_sub(n_head as usize);
        let base = if is_q {
            (t * n_head as usize + h) * HEAD
        } else {
            (t * n_kv as usize + kh) * HEAD
        };
        // SAFETY: t < m <= pos.len() by the launch contract.
        let p = unsafe { *pos.get_unchecked(t) } as usize;

        // SAFETY: base + tid + 64 < (t·n + h + 1)·128 <= m·n·128, inside the
        // head's buffer by the launch contract; one thread per value pair.
        let (x0, x1) = unsafe {
            if is_q {
                (
                    *q.get_unchecked_mut(base + tid),
                    *q.get_unchecked_mut(base + tid + THREADS),
                )
            } else {
                (
                    *k.get_unchecked_mut(base + tid),
                    *k.get_unchecked_mut(base + tid + THREADS),
                )
            }
        };
        let mut acc = f64::from(x0 * x0) + f64::from(x1 * x1);
        acc += warp::shuffle_xor_f64(acc, 16);
        acc += warp::shuffle_xor_f64(acc, 8);
        acc += warp::shuffle_xor_f64(acc, 4);
        acc += warp::shuffle_xor_f64(acc, 2);
        acc += warp::shuffle_xor_f64(acc, 1);
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Two slots, one per warp, written before the barrier that publishes
        // them.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < 2; one lane per warp writes its slot.
            unsafe { *ws.add(tid / 32) = acc };
        }
        thread::sync_threads();
        // SAFETY: both slots were written before the barrier above.
        let sum = unsafe { *ws.add(0) + *ws.add(1) };
        let mean = (sum / HEAD as f64) as f32;
        let scale = rms_scale_of_mean_checked(mean, eps, fault, tid == 0);

        if p >= ctx as usize {
            if tid == 0 {
                fault.raise(FaultSite::CachePos);
            }
            // SAFETY: the positions read above; this thread owns them.
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(base + tid) = f32::NAN;
                    *q.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                } else {
                    *k.get_unchecked_mut(base + tid) = f32::NAN;
                    *k.get_unchecked_mut(base + tid + THREADS) = f32::NAN;
                }
            }
            return; // block-uniform: p is the token's
        }
        // SAFETY: tid + 64 < 128 <= the gain's length; p < ctx and 2·tid + 1
        // < 128, so the table pair is inside row p of the table's ctx rows.
        let (g0, g1, c, s) = unsafe {
            let g = if is_q { gq } else { gk };
            (
                *g.get_unchecked(tid),
                *g.get_unchecked(tid + THREADS),
                *table.get_unchecked(p * HEAD + 2 * tid),
                *table.get_unchecked(p * HEAD + 2 * tid + 1),
            )
        };
        let n0 = (scale * g0) * x0;
        let n1 = (scale * g1) * x1;
        let (y0, y1) = neox_pair(n0, n1, c, s);

        if is_q {
            // SAFETY: the positions read above; this thread owns them.
            unsafe {
                *q.get_unchecked_mut(base + tid) = y0;
                *q.get_unchecked_mut(base + tid + THREADS) = y1;
            }
            return;
        }
        // SAFETY: the positions read above; this thread owns them.
        unsafe {
            *k.get_unchecked_mut(base + tid) = y0;
            *k.get_unchecked_mut(base + tid + THREADS) = y1;
        }
        let row = (kh * rows as usize + p % rows as usize) * HEAD;
        // SAFETY: kh < n_kv and p % rows < rows, so row + 127 < n_kv·rows·128 <= the
        // planes' lengths; base + tid + 64 < m·n_kv·128 <= v.len(). One
        // thread per pair of the row; the tokens of one launch hold distinct
        // positions (the launch contract; the engine's embedding launch
        // writes consecutive ones), so no two blocks write one row.
        unsafe {
            *cache_k.get_unchecked_mut(row + tid) = f32_to_f16_bits(y0);
            *cache_k.get_unchecked_mut(row + tid + THREADS) = f32_to_f16_bits(y1);
            *cache_v.get_unchecked_mut(row + tid) = f32_to_f16_bits(*v.get_unchecked(base + tid));
            *cache_v.get_unchecked_mut(row + tid + THREADS) =
                f32_to_f16_bits(*v.get_unchecked(base + tid + THREADS));
        }
    }
}

/// The segment pass of the ring entries at `G` query heads a key head. Block
/// `b = (seg·n_kv + kh)·m + t` walks the windowed keys `[first + seg·span,
/// first + min((seg + 1)·span, n_w))` of key head `kh`, where `(first, n_w)` is
/// [`window_cut`] of the row's live count and `span` is [`seg_span`] of `n_w`;
/// key position `p` is row `p mod rows` of the key head's ring. Warp `w` is
/// query head `kh·G + w`. A segment past the window writes a neutral partial.
///
/// SAFETY: the entry's launch contract at `G` query heads a key head (every
/// buffer bound it names, with every key the row reads resident in the ring), a
/// block of `G · 32` threads, and the four tiles this block's shared memory as
/// the entry declared them: the group's query rows (`G · HEAD` f32), the key
/// tile at stride `K_STRIDE` words, the value tile (`KEY_TILE · ROW_QWORDS`
/// u64) and the weight tile (`G · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
unsafe fn seg_h128_ring<const G: usize>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
    scale: f32,
    n_kv: u32,
    ctx: u32,
    segs: u32,
    seg_keys: u32,
    rows: u32,
    m: u32,
    window: u32,
    mut part_v: DisjointSlice<f32>,
    mut part_ms: DisjointSlice<f32>,
    qs: *mut f32,
    ks: *mut u32,
    vs: *mut u64,
    ws: *mut f32,
) {
    const { assert!(G == GROUP || G == GROUP_4) };
    // Every thread stages the same whole number of key and value words, and
    // HEAD / 32 query values: the block's threads cover the group's rows.
    const { assert!((KEY_TILE * ROW_QWORDS).is_multiple_of(G * 32) && G * HEAD == G * 32 * 4) };
    let threads = G * 32;
    let stage = KEY_TILE * ROW_QWORDS / threads;

    let b = thread::blockIdx_x() as usize;
    let nkv = n_kv as usize;
    let n_seg = segs as usize;
    let tokens = m as usize;
    if b >= tokens * nkv * n_seg {
        return; // block-uniform
    }
    let t = b % tokens;
    let sk = b / tokens;
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

    let qb = (t * n_head + kh * G) * HEAD;
    let mut i = 0usize;
    while i < 4 {
        // SAFETY: qb + 4·tid + i is inside row t's group, and that row is
        // inside q by the launch contract. The shared index is inside
        // G·HEAD.
        unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
        i += 1;
    }

    let plane = kh * rows as usize * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        let mut i = 0usize;
        while i < stage {
            let e = tid + threads * i;
            let key = e / ROW_QWORDS;
            let word = e - key * ROW_QWORDS;
            let (kw, vw) = if t0 + key < hi {
                // SAFETY: t0 + key < hi <= limit <= ctx, so the row is
                // mapped modulo rows inside key head kh's ring. A row is 32 u64 and the
                // planes start 8-byte aligned (a device allocation).
                unsafe {
                    let r = (plane + ((t0 + key) % rows as usize) * HEAD) / 4 + word;
                    (*k64.add(r), *v64.add(r))
                }
            } else {
                (0u64, 0u64)
            };
            // SAFETY: key < 32 and word < 32, inside the staged tiles.
            unsafe {
                *ks.add(key * K_STRIDE + 2 * word) = kw as u32;
                *ks.add(key * K_STRIDE + 2 * word + 1) = (kw >> 32) as u32;
                *vs.add(key * ROW_QWORDS + word) = vw;
            }
            i += 1;
        }
        thread::sync_threads();

        let live = t0 + lane < hi;
        let mut a = [0.0f32; ILP];
        let mut wd = 0usize;
        while wd < ROW_WORDS {
            let mut p = 0usize;
            while p < ILP {
                // SAFETY: lane < 32 and wd + p < ROW_WORDS, inside the
                // staged key tile. The query index is inside G·HEAD.
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

        // SAFETY: the weight and value tiles are this block's, the value
        // tile was published by the barrier above, and w < G.
        unsafe { fold_tile(sc, live, &mut mx, &mut s_sum, &mut acc, ws, vs, w, lane) };
        t0 += KEY_TILE;
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
mod flash_kernels {
    use super::*;
    /// Segment pass at [`GROUP`] query heads a key head (`seg_h128_ring`): a
    /// block of 256 threads, warp `w` is query head `kh·8 + w`.
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
            rows >= 1,
            rows <= ctx,
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * rows * 128,
            vc.len() >= n_kv * rows * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_h128_swa_ring(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        rows: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP query heads; the raw form reaches it
        // without a reference. The launch contract is `seg_h128_ring`'s at
        // GROUP, and the block is GROUP · 32 threads.
        unsafe {
            seg_h128_ring::<GROUP>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                rows,
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

    /// [`gqa_flash_seg_h128_swa_ring`] at [`GROUP_4`] query heads a key head
    /// (`seg_h128_ring`): a block of 128 threads, warp `w` is query head
    /// `kh·4 + w`, and a head's partials are the group of eight's for the same
    /// keys.
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
            rows >= 1,
            rows <= ctx,
            n_keys_buf.len() >= m,
            segs >= 1,
            q.len() >= m * n_kv * 4 * 128,
            kc.len() >= n_kv * rows * 128,
            vc.len() >= n_kv * rows * 128,
            part_v.len() >= m * n_kv * 4 * segs * 128,
            part_ms.len() >= m * n_kv * 4 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_h128_swa_ring_g4(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        n_kv: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        rows: u32,
        m: u32,
        window: u32,
        part_v: DisjointSlice<f32>,
        part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { GROUP_4 * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_4 * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_4 query heads; the raw form reaches it
        // without a reference. The launch contract is `seg_h128_ring`'s at
        // GROUP_4, and the block is GROUP_4 · 32 threads.
        unsafe {
            seg_h128_ring::<GROUP_4>(
                q,
                kc,
                vc,
                n_keys_buf,
                scale,
                n_kv,
                ctx,
                segs,
                seg_keys,
                rows,
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
            sinks.len() >= n_head,
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn gqa_flash_merge_h128_swa_ring(
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
        let h = b - t * nh;
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
        // SAFETY: h < n_head <= sinks.len() by the launch contract.
        let sink = unsafe { *sinks.get_unchecked(h) };
        // The sink joins last, after every live segment.
        let (_, s, acc) = online_fold(mx, s, acc, sink, 1.0, 0.0);
        let out = if sink.is_nan() || sink == f32::INFINITY {
            f32::NAN
        } else {
            acc * (1.0 / s)
        };
        // SAFETY: b·HEAD + d is inside y; thread d owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = out };
    }

    /// The merge of a layer with no sinks: [`gqa_flash_merge_h128_swa_ring`]'s
    /// walk over the segments (`merge_sink_body` without the sink),
    /// `acc · (1/s)`. Segments ascending: this order is the gate.
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
            part_v.len() >= m * n_head * segs * 128,
            part_ms.len() >= m * n_head * segs * 2,
            y.len() >= m * n_head * 128
        )
    )]
    pub fn gqa_flash_merge_h128_swa_ring_nosink(
        part_v: &[f32],
        part_ms: &[f32],
        n_keys_buf: &[u32],
        n_head: u32,
        ctx: u32,
        segs: u32,
        seg_keys: u32,
        m: u32,
        window: u32,
        fault: FaultSink,
        y: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is `merge_sink_body`'s at HEAD without
        // the sink, and the block is HEAD threads.
        unsafe {
            merge_sink_body::<HEAD, false>(
                part_v,
                part_ms,
                n_keys_buf,
                &[],
                n_head,
                ctx,
                segs,
                seg_keys,
                m,
                window as usize,
                fault,
                y,
            )
        };
    }
}

/// New ring entries; legacy plane entries stay in their original modules.
pub(crate) struct Kernels {
    append: append_kernels::LoadedModule,
    flash: flash_kernels::LoadedModule,
}

impl Kernels {
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        // SAFETY: both embedded modules belong to this crate; every launch is checked below.
        let append = unsafe { crate::shared_module!(append_kernels, ctx)? };
        // SAFETY: this crate owns this embedded module too.
        let flash = unsafe { crate::shared_module!(flash_kernels, ctx)? };
        Ok(Self { append, flash })
    }

    /// # Safety
    /// Valid positions are distinct modulo `rows`.
    pub(crate) unsafe fn append(
        &self,
        stream: &CudaStream,
        args: NeoxArgs<'_>,
        rows: usize,
        rotate: bool,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "attention_h128::ring_append";
        let NeoxArgs {
            q,
            k,
            v,
            gq,
            gk,
            table,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k,
            cache_v,
        } = args;
        if n_head == 0 || n_kv == 0 || ctx == 0 || rows == 0 || rows > ctx || m == 0 || m > rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "need heads/kv >=1, 1<=rows<=ctx and 1<=m<=rows: heads={n_head} kv={n_kv} ctx={ctx} rows={rows} m={m}"
                ),
            ));
        }
        let plane = n_kv
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(HEAD))
            .ok_or_else(|| GpuError::shape(WHAT, "cache shape overflow"))?;
        let q_len = m
            .checked_mul(n_head)
            .and_then(|n| n.checked_mul(HEAD))
            .ok_or_else(|| GpuError::shape(WHAT, "query shape overflow"))?;
        let kv_len = m
            .checked_mul(n_kv)
            .and_then(|n| n.checked_mul(HEAD))
            .ok_or_else(|| GpuError::shape(WHAT, "key shape overflow"))?;
        for (name, got, need) in [
            ("q", q.len(), q_len),
            ("k", k.len(), kv_len),
            ("v", v.len(), kv_len),
            ("gq", gq.len(), HEAD),
            ("gk", gk.len(), HEAD),
            ("pos", pos.len(), m),
            ("cache_k", cache_k.len(), plane),
            ("cache_v", cache_v.len(), plane),
            (
                "table",
                table.len(),
                if rotate {
                    ctx.checked_mul(HEAD)
                        .ok_or_else(|| GpuError::shape(WHAT, "table shape overflow"))?
                } else {
                    0
                },
            ),
        ] {
            if got < need {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}.len() {got} < {need}"),
                ));
            }
        }
        let grid = launch_u32(
            WHAT,
            "grid",
            m.checked_mul(
                n_head
                    .checked_add(n_kv)
                    .ok_or_else(|| GpuError::shape(WHAT, "head count overflow"))?,
            )
            .ok_or_else(|| GpuError::shape(WHAT, "grid overflow"))?,
        )?;
        let nh = launch_u32(WHAT, "heads", n_head)?;
        let nkv = launch_u32(WHAT, "kv", n_kv)?;
        let ctx = launch_u32(WHAT, "ctx", ctx)?;
        let rows = launch_u32(WHAT, "rows", rows)?;
        let m = launch_u32(WHAT, "m", m)?;
        if rotate {
            let prep = self
                .append
                .prepare_head_norm_neox_append_ring(LaunchConfig1D::new(grid, 64, 0))?;
            self.append.head_norm_neox_append_ring(
                stream, &prep, gq, gk, table, pos, v, eps, nh, nkv, ctx, rows, m, fault, q, k,
                cache_k, cache_v,
            )?;
        } else {
            let prep = self
                .append
                .prepare_qk_rms_append_ring(LaunchConfig1D::new(grid, 64, 0))?;
            self.append.qk_rms_append_ring(
                stream, &prep, gq, gk, pos, v, eps, nh, nkv, ctx, rows, m, fault, q, k, cache_k,
                cache_v,
            )?;
        }
        Ok(())
    }

    /// Enqueue `m` rows of a window layer over a ring of `rows` rows. `n_head`
    /// is `8 · n_kv` or `4 · n_kv`; any other group is refused by name.
    /// `sinks` `None` runs the merge with no sink term.
    pub(crate) fn attention(
        &self,
        stream: &CudaStream,
        args: H128SwaArgs<'_>,
        n_head: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "attention_h128::ring_attention";
        let H128SwaArgs {
            q,
            kc,
            vc,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            window,
            sinks,
            part_v,
            part_ms,
            fault,
            y,
        } = args;
        if n_kv == 0
            || ctx == 0
            || rows == 0
            || rows > ctx
            || m == 0
            || !scale.is_finite()
            || window == 0
            || window > rows
            || m > rows - window + 1
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "invalid ring geometry: heads={n_head} kv={n_kv} ctx={ctx} rows={rows} m={m} window={window} scale={scale}"
                ),
            ));
        }
        let Some(group) = served_group(n_head, n_kv) else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the ring kernels take {GROUP} or {GROUP_4} query heads a key head; got \
                     {n_head} heads over {n_kv}"
                ),
            ));
        };
        let segs = window_segments(window);
        let heads = m
            .checked_mul(n_head)
            .ok_or_else(|| GpuError::shape(WHAT, "heads overflow"))?;
        let partials = heads
            .checked_mul(segs)
            .ok_or_else(|| GpuError::shape(WHAT, "partials overflow"))?;
        for (name, got, need) in [
            ("q", q.len(), heads.checked_mul(HEAD)),
            (
                "kc",
                kc.len(),
                n_kv.checked_mul(rows).and_then(|n| n.checked_mul(HEAD)),
            ),
            (
                "vc",
                vc.len(),
                n_kv.checked_mul(rows).and_then(|n| n.checked_mul(HEAD)),
            ),
            ("n_keys", n_keys.len(), Some(m)),
            ("part_v", part_v.len(), partials.checked_mul(HEAD)),
            ("part_ms", part_ms.len(), partials.checked_mul(2)),
            ("y", y.len(), heads.checked_mul(HEAD)),
        ] {
            let need =
                need.ok_or_else(|| GpuError::shape(WHAT, format!("{name} shape overflow")))?;
            if got < need {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{name}.len() {got} < {need}"),
                ));
            }
        }
        if let Some(sinks) = sinks
            && sinks.len() < n_head
        {
            return Err(GpuError::shape(
                WHAT,
                format!("sinks.len() {} < {n_head}", sinks.len()),
            ));
        }
        let grid = launch_u32(
            WHAT,
            "grid",
            m.checked_mul(n_kv)
                .and_then(|n| n.checked_mul(segs))
                .ok_or_else(|| GpuError::shape(WHAT, "grid overflow"))?,
        )?;
        let nkv = launch_u32(WHAT, "kv", n_kv)?;
        let ctx = launch_u32(WHAT, "ctx", ctx)?;
        let rows = launch_u32(WHAT, "rows", rows)?;
        let ns = launch_u32(WHAT, "segments", segs)?;
        let sk = launch_u32(WHAT, "segment keys", SEG_KEYS)?;
        let mu = launch_u32(WHAT, "m", m)?;
        let window = launch_u32(WHAT, "window", window)?;
        let nh = launch_u32(WHAT, "heads", n_head)?;
        if group == GROUP_4 {
            let prep = self
                .flash
                .prepare_gqa_flash_seg_h128_swa_ring_g4(LaunchConfig1D::new(
                    grid,
                    THREADS_G4_U32,
                    0,
                ))?;
            self.flash.gqa_flash_seg_h128_swa_ring_g4(
                stream,
                &prep,
                q,
                kc,
                vc,
                n_keys,
                scale,
                nkv,
                ctx,
                ns,
                sk,
                rows,
                mu,
                window,
                &mut *part_v,
                &mut *part_ms,
            )?;
        } else {
            let prep = self
                .flash
                .prepare_gqa_flash_seg_h128_swa_ring(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
            self.flash.gqa_flash_seg_h128_swa_ring(
                stream,
                &prep,
                q,
                kc,
                vc,
                n_keys,
                scale,
                nkv,
                ctx,
                ns,
                sk,
                rows,
                mu,
                window,
                &mut *part_v,
                &mut *part_ms,
            )?;
        }
        let merge = LaunchConfig1D::new(launch_u32(WHAT, "merge grid", heads)?, MERGE_THREADS, 0);
        if let Some(sinks) = sinks {
            let prep = self.flash.prepare_gqa_flash_merge_h128_swa_ring(merge)?;
            self.flash.gqa_flash_merge_h128_swa_ring(
                stream, &prep, part_v, part_ms, n_keys, sinks, nh, ctx, ns, sk, mu, window, fault,
                y,
            )?;
        } else {
            let prep = self
                .flash
                .prepare_gqa_flash_merge_h128_swa_ring_nosink(merge)?;
            self.flash.gqa_flash_merge_h128_swa_ring_nosink(
                stream, &prep, part_v, part_ms, n_keys, nh, ctx, ns, sk, mu, window, fault, y,
            )?;
        }
        Ok(())
    }
}
