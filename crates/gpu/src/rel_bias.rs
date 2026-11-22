//! Rel-bias attention over the head-128 segment walk: the entries an
//! architecture whose scores carry a distance-gathered bias row runs at a
//! group of [`GROUP_3`] query heads a key head (Inkling 2's 24 over 8).
//!
//! The walk, the tiles and the softmax are `flash_gqa_h128_swa`'s — the same
//! [`seg_span`] window cut, the same [`fold_tile`] step, one warp a query
//! head, the partials `(max, Σ exp, Σ exp·V)` a segment — with two facts of
//! the bias laid in:
//!
//! - the score of key `j` gains `bⱼ = Σ_{d < REL} rel_bias[d, distⱼ] · r_d`,
//!   `distⱼ = live − 1 − key position`, the key positions of a tile being
//!   consecutive so the tile's `[REL][KEY_TILE]` slice of the table is staged
//!   in shared beside the key and value tiles. A distance at or past
//!   `extent` reads 0 — the reference's pad column, which the caller does
//!   not upload. The `REL` values of `r` and the row's sink `t` are plain
//!   launch data, computed upstream;
//! - the sink joins the softmax as the partial `(t, 1, 0)` folded last by
//!   the merge — one more logit in the denominator, nothing in the value —
//!   read per (row, head), so the merge is this file's own: a sink that
//!   varies a row is the whole difference from
//!   `gqa_flash_merge_h128_swa`, whose sink is one value a head.
//!
//! `exp(s − t) / (1 + Σ exp(s_k − t))` and this file's `exp(s) / (exp(t) +
//! Σ exp(s_k))` are one distribution; the rounding points differ, and the
//! owning gate bands that distance. The segment body is this file's own
//! copy of the walk, not a [`GROUP_3`] instance of `seg_h128`: a group of
//! three leaves `KEY_TILE · ROW_QWORDS` key and value words uneven over the
//! block's threads, so the even-division staging `seg_h128` asserts does not
//! hold at 96 threads — the walk stays that body's, the staging is ragged
//! here (the same reason `flash_gqa` keeps a body of its own for the group
//! of five). The plane and ring forms are one body under `RING`, as there.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{MERGE_BATCH, half_bits_to_f32, online_fold};
use crate::flash_gqa::{HEAD, KEY_TILE, SEG_KEYS, seg_span, window_cut, window_segments};
use crate::flash_gqa_h128_swa::{
    GROUP_3, K_STRIDE, MERGE_THREADS, ROW_QWORDS, fold_tile, live_keys,
};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Relative dimensions a head: the width of `r` and of the bias table's
/// first axis.
pub const REL: usize = 16;
const THREADS: usize = GROUP_3 * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS == 96);
const _: () = assert!(REL * KEY_TILE == 512 && K_STRIDE % 2 == 1 && REL <= 32);

/// The segment pass of this file's entries: `seg_h128`'s walk at [`GROUP_3`]
/// query heads a key head with the bias laid into the score, on a plane
/// (`RING` false) or a ring (`RING` true). Block
/// `b = (seg·n_kv + kh)·m + t`, warp `w` is query head `kh·3 + w`; the
/// partials, the window cut and the neutral-segment rule are `seg_h128`'s.
///
/// The score's order is the gate: the f32 dot of `seg_h128` (its rotating
/// ILP partials, combined `(a0 + a1) + (a2 + a3)`), one multiply by
/// `scale`, then the bias dot added — `REL` fused multiply-adds ascending
/// `d`, the r row one value a lane, read by shuffle.
///
/// SAFETY: the entry's launch contract at [`GROUP_3`] query heads a key
/// head (every buffer bound it names, the planes of `n_kv · plane_rows`
/// rows, and under `RING` `plane_rows >= 1` with every key the row reads
/// resident in the ring), a block of `GROUP_3 · 32` threads, and the five
/// tiles this block's shared memory as the entry declared them: the group's
/// query rows (`GROUP_3 · HEAD` f32), the key tile at stride `K_STRIDE`
/// words, the value tile (`KEY_TILE · ROW_QWORDS` u64), the weight tile
/// (`GROUP_3 · KEY_TILE` f32) and the bias tile (`REL · KEY_TILE` f32).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's flat arguments, handed on (rust-quality R8)"
)]
pub(crate) unsafe fn seg_h128_rel<const RING: bool>(
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
    bs: *mut f32,
) {
    let threads = THREADS;
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
    let n_head = nkv * GROUP_3;
    let h = kh * GROUP_3 + w;
    let idx = (t * n_head + h) * n_seg + seg;
    let ctx = ctx as usize;
    // SAFETY: t < m <= n_keys_buf.len() by the launch contract.
    let limit = live_keys(unsafe { *n_keys_buf.get_unchecked(t) }, ctx);
    let (first, n_w) = window_cut(limit, window as usize);
    let span = seg_span(n_w, n_seg, seg_keys as usize);
    let lo = first + seg * span;
    if lo >= first + n_w {
        if lane == 0 {
            // SAFETY: idx < m·n_kv·GROUP_3·segs, both slots inside part_ms.
            unsafe {
                *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
            }
        }
        return; // block-uniform: the window cut is the block's
    }
    let hi = (lo + span).min(first + n_w);

    let qb = (t * n_head + kh * GROUP_3) * HEAD;
    let mut i = 0usize;
    while i < 4 {
        // SAFETY: qb + 4·tid + i is inside row t's group, and that row is
        // inside q by the launch contract. The shared index is inside
        // GROUP_3·HEAD, which 4·(GROUP_3·32) equals.
        unsafe { *qs.add(4 * tid + i) = *q.get_unchecked(qb + 4 * tid + i) };
        i += 1;
    }
    // Lane l holds the head's r value l mod REL; the bias dot reads value d
    // from lane d by a shuffle, so the row stays in one register a lane.
    // SAFETY: (t·n_head + h)·REL + lane mod REL is inside r by the launch
    // contract.
    let r_lane = unsafe { *r.get_unchecked((t * n_head + h) * REL + lane % REL) };

    let plane_rows = plane_rows as usize;
    let plane = kh * plane_rows * HEAD;
    let k64 = kc.as_ptr() as *const u64;
    let v64 = vc.as_ptr() as *const u64;
    let extent = extent as usize;
    let mut mx = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    let mut acc = [0.0f32; 4];
    let mut t0 = lo;
    while t0 < hi {
        thread::sync_threads();
        let mut e = tid;
        while e < KEY_TILE * ROW_QWORDS {
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
                    let rw = (plane + at * HEAD) / 4 + word;
                    (*k64.add(rw), *v64.add(rw))
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
            e += threads;
        }
        // The bias tile: column `key` holds the table column of the key's
        // distance, 0 past the extent (the pad rule) and 0 for a key the
        // segment does not hold.
        let mut e = tid;
        while e < REL * KEY_TILE {
            let d = e / KEY_TILE;
            let key = e - d * KEY_TILE;
            let val = if t0 + key < hi {
                let dist = limit - 1 - (t0 + key);
                if dist < extent {
                    // SAFETY: d·extent + dist < REL·extent <= bias.len() by
                    // the launch contract.
                    unsafe { *bias.get_unchecked(d * extent + dist) }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            // SAFETY: e < REL·KEY_TILE, inside the staged bias tile.
            unsafe { *bs.add(e) = val };
            e += threads;
        }
        thread::sync_threads();

        let live = t0 + lane < hi;
        let mut a = [0.0f32; 4];
        let mut wd = 0usize;
        while wd < ROW_WORDS {
            let mut p = 0usize;
            while p < 4 {
                // SAFETY: lane < 32 and wd + p < ROW_WORDS, inside the
                // staged key tile. The query index is inside GROUP_3·HEAD.
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
            wd += 4;
        }
        let dot = (a[0] + a[1]) + (a[2] + a[3]);
        // The bias dot's order is the gate: REL fused multiply-adds
        // ascending d against the staged tile.
        let mut b = 0.0f32;
        let mut d = 0usize;
        while d < REL {
            let rd = warp::shuffle_f32(r_lane, d as u32);
            // SAFETY: d·KEY_TILE + lane < REL·KEY_TILE, inside the staged
            // bias tile, published by the barrier above.
            b = f32::mul_add(rd, unsafe { *bs.add(d * KEY_TILE + lane) }, b);
            d += 1;
        }
        let sc = if live {
            dot * scale + b
        } else {
            f32::NEG_INFINITY
        };

        // SAFETY: the weight and value tiles are this block's, the value
        // tile was published by the barrier above, and w < GROUP_3.
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

/// u32 words of one key row, the span the score loop's rounds walk.
const ROW_WORDS: usize = HEAD / 2;
const _: () = assert!(ROW_WORDS.is_multiple_of(4));

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
        static mut QS: SharedArray<f32, { GROUP_3 * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_3 * KEY_TILE }> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, { REL * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_3 query heads and the bias tile; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s on a plane, and the block is GROUP_3 · 32
        // threads.
        unsafe {
            seg_h128_rel::<false>(
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
                SharedArray::as_raw_mut_ptr(&raw mut BS),
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
            ring_rows >= 1,
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
        static mut QS: SharedArray<f32, { GROUP_3 * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { GROUP_3 * KEY_TILE }> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, { REL * KEY_TILE }> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, sized for GROUP_3 query heads and the bias tile; the
        // raw form reaches it without a reference. The launch contract is
        // `seg_h128_rel`'s under RING, and the block is GROUP_3 · 32
        // threads.
        unsafe {
            seg_h128_rel::<true>(
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
                SharedArray::as_raw_mut_ptr(&raw mut BS),
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

    /// Enqueue `m` rows at [`GROUP_3`] query heads a key head (`n_head`
    /// must equal `3 · n_kv`; any other ratio is refused by name, as a
    /// non-finite scale, a zero extent or count or a short buffer is). A
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
        if n_head != n_kv * GROUP_3 {
            return Err(GpuError::shape(
                what,
                format!(
                    "the rel-bias kernels take {GROUP_3} query heads a key head; got {n_head} \
                     heads over {n_kv}"
                ),
            ));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                what,
                format!("the score scale must be finite, got {scale}"),
            ));
        }
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
        if ring_rows.is_some() {
            let prep =
                self.module
                    .prepare_rel_seg_h128_ring(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
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
        } else {
            let prep =
                self.module
                    .prepare_rel_seg_h128(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
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
