//! Head-128 grouped-query attention with a window and optional per-head sinks.
//!
//! One block holds the query heads of one key head, one warp each: a group of
//! [`GROUP`] (eight, the `_swa` entries) or of [`GROUP_4`] (four, the `_swa_g4`
//! entries, a block of 128 threads). Both groups, on a plane here and on a
//! ring in `attention_h128_ring.rs`, run the one segment body (`seg_h128`), so
//! a head's scores, tile order and partials are the same at either group and
//! either cache. Scores run over [`HEAD`] key values and the values are
//! [`HEAD`] wide, the same scalar tile as `gqa_flash_seg`. The keys a row
//! attends are [`window_cut`]: the last `window` live keys, or every live key
//! when `window` is 0, cut into [`window_segments`] from that first key. A
//! layer with sinks has its sink join the softmax last, as the partial
//! `(sink, 1, 0)`, after the segments in order; a layer with none has no sink
//! term, and its merge (`..._nosink`) is the same walk without it. Only the
//! group of eight carries sinks (`group_sinks_refusal`). A full layer
//! launches window 0.
//!
//! The window's lower bound is [`window_cut`]'s. This file does not own a
//! second inequality.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{MERGE_BATCH, dev_exp, half_bits_to_f32, online_fold};
use crate::flash_gqa::{
    GROUP, HEAD, KEY_TILE, SEG_KEYS, SEGMENTS, merge_sink_body, seg_span, window_cut,
    window_segments,
};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// u32 words of one key row, and the padded stride the key tile is staged at.
const ROW_WORDS: usize = HEAD / 2;
pub(crate) const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of one value row.
pub(crate) const ROW_QWORDS: usize = HEAD / 4;
const ILP: usize = 4;
/// Query heads a key head that the `_g4` entries serve: the segment body of
/// the group of eight with four warps.
pub const GROUP_4: usize = 4;
const THREADS: usize = GROUP * 32;
/// Threads of a block at the group of eight, and at the group of four.
pub(crate) const THREADS_U32: u32 = THREADS as u32;
pub(crate) const THREADS_G4_U32: u32 = (GROUP_4 * 32) as u32;
/// Threads of a merge block: one a value dim.
pub(crate) const MERGE_THREADS: u32 = HEAD as u32;
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

/// A row's live key count: `count` when it lies in `1..=ctx`, else 0. A
/// refused row's segments are neutral and its merge raises
/// [`FaultSite::KeyCount`].
#[inline(always)]
pub(crate) fn live_keys(count: u32, ctx: usize) -> usize {
    let c = count as usize;
    if c > ctx { 0 } else { c }
}

/// One tile's online-softmax step for warp `w`'s head. The same step
/// `gqa_flash_seg` runs, so a tile's softmax and its value order match that
/// pass at this width.
///
/// # Safety
///
/// `ws` is the block's `G · KEY_TILE` f32 weight tile and `vs` its staged
/// `KEY_TILE · ROW_QWORDS` value tile, published by a block barrier, at the
/// block's group `G`. `w < G`. All 32 lanes of the warp call it together.
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

    // SAFETY: w·32 + lane < G·KEY_TILE; this warp's own slots.
    unsafe { *ws.add(w * KEY_TILE + lane) = pw };
    warp::sync_mask(u32::MAX);
    let mut j = 0usize;
    while j < KEY_TILE {
        // SAFETY: j < 32 and lane < 32, inside the weight and value tiles.
        let (pj, vw) = unsafe { (*ws.add(w * KEY_TILE + j), *vs.add(j * ROW_QWORDS + lane)) };
        acc[0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[0]);
        acc[1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[1]);
        acc[2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[2]);
        acc[3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[3]);
        j += 1;
    }
    warp::sync_mask(u32::MAX);
}

/// The segment pass of the `_swa` entries at `G` query heads a key head, on a
/// plane (`RING` false) or on a ring (`RING` true, the entries of
/// `attention_h128_ring.rs`). Block `b = (seg·n_kv + kh)·m + t` walks the
/// windowed keys `[first + seg·span, first + min((seg + 1)·span, n_w))` of key
/// head `kh`, where `(first, n_w)` is [`window_cut`] of the row's live count
/// and `span` is [`seg_span`] of `n_w`. Warp `w` is query head `kh·G + w`. A
/// segment past the window writes a neutral partial.
///
/// The planes hold `plane_rows` rows a key head: `ctx` of them in a plane
/// entry (key position `p` is row `p`), and under `RING` a ring of
/// `plane_rows <= ctx` rows (position `p` is row `p mod plane_rows`). `ctx`
/// stays the bound of the live count.
///
/// SAFETY: the entry's launch contract at `G` query heads a key head (every
/// buffer bound it names, the planes of `n_kv · plane_rows` rows, and under
/// `RING` `plane_rows >= 1` with every key the row reads resident in the
/// ring), a block of `G · 32` threads, and the four tiles this block's shared
/// memory as the entry declared them: the group's query rows (`G · HEAD`
/// f32), the key tile at stride `K_STRIDE` words, the value tile (`KEY_TILE ·
/// ROW_QWORDS` u64) and the weight tile (`G · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
pub(crate) unsafe fn seg_h128<const G: usize, const RING: bool>(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    n_keys_buf: &[u32],
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
    const { assert!(G == GROUP || G == GROUP_4) };
    // Every thread stages the same whole number of key and value words, and
    // HEAD / 32 query values: the block's threads cover the group's rows.
    const { assert!((KEY_TILE * ROW_QWORDS).is_multiple_of(G * 32) && G * HEAD == G * 32 * 4) };
    let threads = G * 32;
    let stage = KEY_TILE * ROW_QWORDS / threads;

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

    let qb = (t * n_head + kh * G) * HEAD;
    let mut i = 0usize;
    while i < 4 {
        // SAFETY: qb + 4·tid + i is inside row t's group, and that row is
        // inside q by the launch contract. The shared index is inside
        // G·HEAD.
        unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
        i += 1;
    }

    let plane_rows = plane_rows as usize;
    let plane = kh * plane_rows * HEAD;
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
                // inside key head kh's plane (below plane_rows = ctx), or
                // mapped modulo plane_rows inside its ring. A row is 32 u64
                // and the planes start 8-byte aligned (a device allocation).
                unsafe {
                    let at = if RING {
                        (t0 + key) % plane_rows
                    } else {
                        t0 + key
                    };
                    let r = (plane + at * HEAD) / 4 + word;
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
mod h128_swa_kernels {
    use super::*;

    /// Segment pass at [`GROUP`] query heads a key head (`seg_h128`): a block
    /// of 256 threads, warp `w` is query head `kh·8 + w`.
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
            segs >= 1,
            q.len() >= m * n_kv * 8 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 8 * segs * 128,
            part_ms.len() >= m * n_kv * 8 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_h128_swa(
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
        // without a reference. The launch contract is `seg_h128`'s on a plane
        // at GROUP, and the block is GROUP · 32 threads.
        unsafe {
            seg_h128::<GROUP, false>(
                q,
                kc,
                vc,
                n_keys_buf,
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

    /// [`gqa_flash_seg_h128_swa`] at [`GROUP_4`] query heads a key head
    /// (`seg_h128`): a block of 128 threads, warp `w` is query head `kh·4 + w`,
    /// and a head's partials are the group of eight's for the same keys.
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
            q.len() >= m * n_kv * 4 * 128,
            kc.len() >= n_kv * ctx * 128,
            vc.len() >= n_kv * ctx * 128,
            part_v.len() >= m * n_kv * 4 * segs * 128,
            part_ms.len() >= m * n_kv * 4 * segs * 2
        )
    )]
    pub fn gqa_flash_seg_h128_swa_g4(
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
        // without a reference. The launch contract is `seg_h128`'s on a plane
        // at GROUP_4, and the block is GROUP_4 · 32 threads.
        unsafe {
            seg_h128::<GROUP_4, false>(
                q,
                kc,
                vc,
                n_keys_buf,
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

    /// Merge. Segments ascending, then the sink. This order is the gate.
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
    pub fn gqa_flash_merge_h128_swa(
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

    /// The merge of a layer with no sinks: [`gqa_flash_merge_h128_swa`]'s walk
    /// over the segments (`merge_sink_body` without the sink), `acc · (1/s)`.
    /// Segments ascending: this order is the gate.
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
    pub fn gqa_flash_merge_h128_swa_nosink(
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

/// Arguments of one head-128 windowed launch. `window` 0 attends every live
/// key. `sinks` holds one f32 a query head, or is `None` for a layer with no
/// sink term. The partials hold [`window_segments`]`(window)` segments a head.
pub struct H128SwaArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub kc: &'a DeviceBuffer<u16>,
    pub vc: &'a DeviceBuffer<u16>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scale: f32,
    pub n_kv: usize,
    pub ctx: usize,
    pub m: usize,
    pub window: usize,
    pub sinks: Option<&'a DeviceBuffer<f32>>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The query heads a key head of `n_head` query heads over `n_kv` key heads,
/// when an entry serves them: [`GROUP`] or [`GROUP_4`].
#[must_use]
pub(crate) fn served_group(n_head: usize, n_kv: usize) -> Option<usize> {
    if n_kv == 0 {
        return None;
    }
    [GROUP, GROUP_4]
        .into_iter()
        .find(|&g| n_kv.checked_mul(g) == Some(n_head))
}

/// Why a launch of `group` query heads a key head cannot carry per-head sinks,
/// or `None` when it can: only [`GROUP`] runs the merge with a sink term, and
/// the shape table's group-of-four row (`models::shape::GQA`) claims none. The
/// one owner of that rule: `attention_h128`'s admission and both launchers
/// (this file's and the ring's) ask it.
#[must_use]
pub(crate) fn group_sinks_refusal(group: usize, sinks: bool) -> Option<String> {
    (sinks && group != GROUP).then(|| {
        format!(
            "a group of {group} query heads a key head folds no per-head sinks; only a group of \
             {GROUP} does"
        )
    })
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct H128SwaKernels {
    module: h128_swa_kernels::LoadedModule,
}

impl H128SwaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<H128SwaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(h128_swa_kernels, ctx)? };
        Ok(H128SwaKernels { module })
    }

    /// Enqueue `m` rows. `n_head` is `8 · n_kv` or `4 · n_kv` (one block a key
    /// head); any other group is refused by name. `sinks` `None` runs the
    /// merge with no sink term; sinks at a group of four are refused by name
    /// (`group_sinks_refusal`). A non-finite scale, a zero count or a short
    /// buffer is refused by name before the launch. A live count of zero or
    /// past `ctx` raises [`FaultSite::KeyCount`] and the row is NaN.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        args: H128SwaArgs<'_>,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let what = "flash_gqa_h128_swa::enqueue";
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
        if n_kv == 0 || ctx == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need n_kv, ctx and m >= 1, got n_kv={n_kv} ctx={ctx} m={m}"),
            ));
        }
        let Some(group) = served_group(n_head, n_kv) else {
            return Err(GpuError::shape(
                what,
                format!(
                    "the kernels take {GROUP} or {GROUP_4} query heads a key head; got {n_head} \
                     heads over {n_kv}"
                ),
            ));
        };
        if let Some(why) = group_sinks_refusal(group, sinks.is_some()) {
            return Err(GpuError::shape(what, why));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("the score scale must be finite, got {scale}"),
            ));
        }
        let segs = window_segments(window);
        let lens = [
            ("q", q.len(), m * n_head * HEAD),
            ("kc", kc.len(), n_kv * ctx * HEAD),
            ("vc", vc.len(), n_kv * ctx * HEAD),
            ("n_keys", n_keys.len(), m),
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
        if let Some(sinks) = sinks
            && sinks.len() < n_head
        {
            return Err(GpuError::shape(
                what,
                format!("sinks.len() {} < {n_head}", sinks.len()),
            ));
        }
        let grid = launch_u32(what, "grid", m * n_kv * segs)?;
        let n_kv_u = launch_u32(what, "n_kv", n_kv)?;
        let ctx_u = launch_u32(what, "ctx", ctx)?;
        let segs_u = launch_u32(what, "segs", segs)?;
        let seg_keys = launch_u32(what, "seg_keys", SEG_KEYS)?;
        let m_u = launch_u32(what, "m", m)?;
        let window_u = launch_u32(what, "window", window)?;
        let heads = launch_u32(what, "n_head", n_head)?;
        if group == GROUP_4 {
            let prep = self
                .module
                .prepare_gqa_flash_seg_h128_swa_g4(LaunchConfig1D::new(grid, THREADS_G4_U32, 0))?;
            self.module.gqa_flash_seg_h128_swa_g4(
                stream,
                &prep,
                q,
                kc,
                vc,
                n_keys,
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
        } else {
            let prep = self
                .module
                .prepare_gqa_flash_seg_h128_swa(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
            self.module.gqa_flash_seg_h128_swa(
                stream,
                &prep,
                q,
                kc,
                vc,
                n_keys,
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
        let merge_grid = launch_u32(what, "merge grid", m * n_head)?;
        let merge = LaunchConfig1D::new(merge_grid, MERGE_THREADS, 0);
        if let Some(sinks) = sinks {
            let prep = self.module.prepare_gqa_flash_merge_h128_swa(merge)?;
            self.module.gqa_flash_merge_h128_swa(
                stream, &prep, part_v, part_ms, n_keys, sinks, heads, ctx_u, segs_u, seg_keys, m_u,
                window_u, fault, &mut *y,
            )?;
        } else {
            let prep = self.module.prepare_gqa_flash_merge_h128_swa_nosink(merge)?;
            self.module.gqa_flash_merge_h128_swa_nosink(
                stream, &prep, part_v, part_ms, n_keys, heads, ctx_u, segs_u, seg_keys, m_u,
                window_u, fault, &mut *y,
            )?;
        }
        Ok(())
    }
}
