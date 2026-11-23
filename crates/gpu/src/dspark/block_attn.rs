//! A draft block's attention at a group of three query heads a key head (Inkling 2's draft: 24 query
//! heads over 8 key heads of [`HEAD`] values), over one layer's context ring and the block's own rows.
//!
//! # The planes
//!
//! A layer holds, per key head, a plane of `cap + ROWS` rows of [`HEAD`] f16 values, two to a `u32`
//! word: `cap` ring rows of context, then the block's [`ROWS`] rows. Context position `p` stands in
//! ring row `(p − base) mod cap`; the ring keeps the last `cap` positions, so a layer whose ring is
//! its window holds exactly the rows a block reads, and a layer without a window holds the whole
//! context in a ring of the context capacity. A block row reads the `vis` ring rows the stage word
//! names (`min(count, cap)`) and all `m` block rows, the block being non-causal; every row reads the
//! same keys. Softmax over a set of keys does not depend on their order, so key `j` of a row is ring
//! row `j` for `j < vis` and block row `j − vis` after them ([`key_slot`]).
//!
//! # The pass
//!
//! [`BlockAttn::enqueue_attn`] is two launches. The segment pass runs a block per (segment, key
//! head) of `ROWS · GROUP` warps, warp `w` row `w / GROUP` and query head `kh · GROUP + w mod GROUP`:
//! the key head's K and V tiles of 32 keys are staged once for all of the block's warps, lane `j`
//! scores key `j` against the warp's query row, and the tile's online-softmax step is
//! `flash_gqa`'s (the butterfly max and sum, `dev_exp`, the rescale, a fused multiply-add per key
//! for each value). A row's `vis + m` keys are cut into [`SEGS`] segments of `seg_span` keys. The
//! merge folds a row's segments in order with `online_fold`. A stage word past `cap` reads no key,
//! raises [`FaultSite::KeyCount`] and writes the rows NaN.
//!
//! [`BlockAttn::enqueue_put`] writes rows of `n_kv · HEAD` f32 (token-major, key heads in order)
//! into a plane's rows as f16, row `t` at plane row `offset + (start + t) mod modulus`, `start` a
//! stage word: a context append's rows at their ring rows (offset 0, modulus `cap`), the block's at
//! its rows behind the ring (offset `cap`, modulus [`ROWS`]).

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;

use crate::flash::{dev_exp, f32x2_to_f16x2_bits, half_bits_to_f32, online_fold};
use crate::flash_gqa::seg_span;
use crate::{FaultSink, FaultSite, GpuError, launch_u32};

/// Values per head.
pub const HEAD: usize = 128;
/// Query heads per key head.
pub const GROUP: usize = 3;
/// Rows of a block: the warps of a segment block are `ROWS · GROUP`.
pub const ROWS: usize = 8;
/// Segments a row's keys are cut into.
pub const SEGS: usize = 20;
/// Keys per online-softmax tile: one key per lane.
const KEY_TILE: usize = 32;
const WARPS: usize = ROWS * GROUP;
const THREADS: usize = WARPS * 32;
const THREADS_U32: u32 = THREADS as u32;
const _: () = assert!(THREADS_U32 as usize == THREADS && THREADS <= 1024);
/// u32 words of one key row, and the padded stride the K tile is staged at: lane `j` reads word `w`
/// of key `j` at `j·65 + w`, a distinct bank per lane.
const ROW_WORDS: usize = HEAD / 2;
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of one row: lane `l` holds value dims `4l .. 4l + 3`.
const ROW_QWORDS: usize = HEAD / 4;
const _: () = assert!(ROW_QWORDS == 32);
/// Tile u64 words each thread stages, rounded up.
const STAGE_PER: usize = (KEY_TILE * ROW_QWORDS).div_ceil(THREADS);
/// Rotating partials of the score dot.
const ILP: usize = 4;
const MERGE_THREADS: u32 = HEAD as u32;
const PUT_THREADS: u32 = ROW_WORDS as u32;

/// The plane row of key `j` of a row reading `vis` ring rows of a ring of `cap`: ring row `j`, then
/// the block's rows behind the ring.
#[inline(always)]
const fn key_slot(j: usize, vis: usize, cap: usize) -> usize {
    if j < vis { j } else { cap + (j - vis) }
}

#[cuda_module]
mod block_attn_kernels {
    use super::*;

    /// The segment pass (module doc): block `b = seg · n_kv + kh`, warp `w` row `t = w / GROUP` and
    /// query head `h = kh · GROUP + w mod GROUP`. Rows at or past `m` stage and score as the others
    /// (every warp meets every barrier) and write nothing. Partial `(t · n_q + h) · SEGS + seg`:
    /// [`HEAD`] value sums in `part_v`, `(max, Σ exp)` in `part_ms`; a segment past the row's keys
    /// writes the neutral `(−∞, 0)`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes"
    )]
    #[kernel]
    #[launch_bounds(768)]
    #[launch_contract(
        domain = 1,
        block = (768, 1, 1),
        requires = (
            words.len() > word,
            m >= 1,
            m <= 8,
            q.len() >= m * n_kv * 3 * 128,
            kc.len() >= n_kv * (cap + 8) * 64,
            vc.len() >= n_kv * (cap + 8) * 64,
            part_v.len() >= m * n_kv * 3 * 20 * 128,
            part_ms.len() >= m * n_kv * 3 * 20 * 2
        )
    )]
    pub fn dspark_g3_seg(
        q: &[f32],
        kc: &[u32],
        vc: &[u32],
        words: &[u32],
        word: u32,
        scale: f32,
        n_kv: u32,
        cap: u32,
        m: u32,
        fault: FaultSink,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { WARPS * HEAD }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { KEY_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut VS: SharedArray<u64, { KEY_TILE * ROW_QWORDS }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { WARPS * KEY_TILE }> = SharedArray::UNINIT;

        let nkv = n_kv as usize;
        let b = thread::blockIdx_x() as usize;
        if b >= SEGS * nkv {
            return; // block-uniform
        }
        let seg = b / nkv;
        let kh = b - seg * nkv;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let t = w / GROUP;
        let h = kh * GROUP + (w - t * GROUP);
        let n_q = nkv * GROUP;
        let rows = m as usize;
        let live_row = t < rows;
        let idx = (t * n_q + h) * SEGS + seg;
        let ring = cap as usize;
        let planes = ring + ROWS;
        // SAFETY: word < words.len() by the launch contract.
        let vis = unsafe { *words.get_unchecked(word as usize) } as usize;
        if vis > ring && b == 0 && tid == 0 {
            fault.raise(FaultSite::KeyCount);
        }
        let n = if vis > ring { 0 } else { vis + rows };
        let span = seg_span(n, SEGS, KEY_TILE);
        let lo = seg * span;
        if lo >= n {
            if live_row && lane == 0 {
                // SAFETY: t < m, so idx < m·n_q·SEGS: both slots inside part_ms.
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform: lo and n are the block's
        }
        let hi = if lo + span < n { lo + span } else { n };

        // SAFETY: each `static mut` above is this block's own shared allocation; the raw form
        // reaches it without a reference.
        let (qs, ks, vs, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };
        // The warp's query row, four values a lane; a dead row stages zeros.
        let qb = (t * n_q + h) * HEAD + 4 * lane;
        for i in 0..4 {
            cuda_device::thread::__unroll_config::<0>();
            let v = if live_row {
                // SAFETY: t < m, so qb + i < m·n_q·HEAD <= q.len().
                unsafe { *q.get_unchecked(qb + i) }
            } else {
                0.0
            };
            // SAFETY: w·HEAD + 4·lane + i < WARPS·HEAD.
            unsafe { *qs.add(w * HEAD + 4 * lane + i) = v };
        }

        let plane = kh * planes * ROW_WORDS;
        let k64 = kc.as_ptr() as *const u64;
        let v64 = vc.as_ptr() as *const u64;
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = [0.0f32; 4];
        let mut t0 = lo;
        while t0 < hi {
            thread::sync_threads();
            // Stage the tile: 32 keys × HEAD/4 u64 of K and of V; a key at or past `hi` stages zeros.
            let mut i = 0usize;
            while i < STAGE_PER {
                let e = tid + THREADS * i;
                if e < KEY_TILE * ROW_QWORDS {
                    let key = e / ROW_QWORDS;
                    let qw = e - key * ROW_QWORDS;
                    let (kw, vw) = if t0 + key < hi {
                        let slot = key_slot(t0 + key, vis, ring);
                        // SAFETY: t0 + key < n <= vis + m with vis <= cap and m <= ROWS, so slot <
                        // cap + ROWS = planes: a row of key head kh's plane, inside both planes by the
                        // launch contract. A row is HEAD/4 u64 and the planes start 8-byte aligned.
                        unsafe {
                            let r = (plane + slot * ROW_WORDS) / 2 + qw;
                            (*k64.add(r), *v64.add(r))
                        }
                    } else {
                        (0u64, 0u64)
                    };
                    // SAFETY: key < 32 and qw < 32: 2·qw + 1 < K_STRIDE and key·32 + qw < 1024.
                    unsafe {
                        *ks.add(key * K_STRIDE + 2 * qw) = kw as u32;
                        *ks.add(key * K_STRIDE + 2 * qw + 1) = (kw >> 32) as u32;
                        *vs.add(key * ROW_QWORDS + qw) = vw;
                    }
                }
                i += 1;
            }
            thread::sync_threads();

            // The score of key t0 + lane against the warp's query row.
            let live = t0 + lane < hi;
            let mut a = [0.0f32; ILP];
            let mut wd = 0usize;
            while wd < ROW_WORDS {
                let mut p = 0usize;
                while p < ILP {
                    // SAFETY: lane < 32, wd + p < ROW_WORDS: inside KS; the query index
                    // w·HEAD + 2(wd + p) + 1 < WARPS·HEAD.
                    let (kw, q0, q1) = unsafe {
                        (
                            *ks.add(lane * K_STRIDE + wd + p),
                            *qs.add(w * HEAD + 2 * (wd + p)),
                            *qs.add(w * HEAD + 2 * (wd + p) + 1),
                        )
                    };
                    a[p] = f32::mul_add(q0, half_bits_to_f32(kw as u16), a[p]);
                    a[p] = f32::mul_add(q1, half_bits_to_f32((kw >> 16) as u16), a[p]);
                    p += 1;
                }
                wd += ILP;
            }
            let dot = (a[0] + a[1]) + (a[2] + a[3]);
            let sc = if live { dot * scale } else { f32::NEG_INFINITY };

            // The tile's online-softmax step: flash_gqa's fold at one value slice.
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
            if m_new > mx {
                let f = if mx > f32::NEG_INFINITY {
                    dev_exp(mx - m_new)
                } else {
                    0.0
                };
                s_sum *= f;
                for c in 0..4 {
                    cuda_device::thread::__unroll_config::<0>();
                    acc[c] *= f;
                }
                mx = m_new;
            }
            s_sum += psum;
            // SAFETY: w·32 + lane < WARPS·KEY_TILE: this warp's own slots.
            unsafe { *ws.add(w * KEY_TILE + lane) = pw };
            warp::sync_mask(u32::MAX);
            let mut j = 0usize;
            while j < KEY_TILE {
                // SAFETY: j < 32 and lane < 32: inside WS (written before the warp sync above) and VS
                // (staged before the block barrier).
                let (pj, vw) =
                    unsafe { (*ws.add(w * KEY_TILE + j), *vs.add(j * ROW_QWORDS + lane)) };
                acc[0] = f32::mul_add(pj, half_bits_to_f32(vw as u16), acc[0]);
                acc[1] = f32::mul_add(pj, half_bits_to_f32((vw >> 16) as u16), acc[1]);
                acc[2] = f32::mul_add(pj, half_bits_to_f32((vw >> 32) as u16), acc[2]);
                acc[3] = f32::mul_add(pj, half_bits_to_f32((vw >> 48) as u16), acc[3]);
                j += 1;
            }
            warp::sync_mask(u32::MAX);
            t0 += KEY_TILE;
        }

        if live_row {
            // SAFETY: t < m, so idx < m·n_q·SEGS; dims 4·lane .. +3 of the partial row are this
            // lane's alone, and lane 0 writes (max, sum).
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
    }

    /// The merge (module doc): block `b = t · n_q + h`, thread `d` value `d`, the row's [`SEGS`]
    /// partials folded in segment order; a neutral partial is skipped. A stage word past `cap`
    /// writes NaN (the segment pass raised the fault). Output `y[b · HEAD + d]`, the rows
    /// token-major with their query heads in order.
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
            words.len() > word,
            part_v.len() >= m * n_q * 20 * 128,
            part_ms.len() >= m * n_q * 20 * 2,
            y.len() >= m * n_q * 128
        )
    )]
    pub fn dspark_g3_merge(
        part_v: &[f32],
        part_ms: &[f32],
        words: &[u32],
        word: u32,
        cap: u32,
        n_q: u32,
        m: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        if b >= m as usize * n_q as usize {
            return; // block-uniform
        }
        let d = thread::threadIdx_x() as usize;
        // SAFETY: word < words.len() by the launch contract.
        let vis = unsafe { *words.get_unchecked(word as usize) };
        let out = if vis > cap {
            f32::NAN
        } else {
            let mut mx = f32::NEG_INFINITY;
            let mut s = 0.0f32;
            let mut acc = 0.0f32;
            let mut seg = 0usize;
            while seg < SEGS {
                let i = b * SEGS + seg;
                // SAFETY: i < m·n_q·SEGS: both slots inside part_ms, and i·HEAD + d inside part_v.
                let (mj, sj) = unsafe {
                    (
                        *part_ms.get_unchecked(2 * i),
                        *part_ms.get_unchecked(2 * i + 1),
                    )
                };
                if sj != 0.0 {
                    // SAFETY: as above.
                    let vj = unsafe { *part_v.get_unchecked(i * HEAD + d) };
                    (mx, s, acc) = online_fold(mx, s, acc, mj, sj, vj);
                }
                seg += 1;
            }
            acc * (1.0 / s)
        };
        // SAFETY: b·HEAD + d < m·n_q·HEAD <= y.len(); thread d of block b owns it.
        unsafe { *y.get_unchecked_mut(b * HEAD + d) = out };
    }

    /// Rows into a plane (module doc): block `b = t · n_kv + h`, thread `i` the word of values `2i,
    /// 2i + 1` of key head `h` of row `t`, written at plane row `offset + (words[word] + t) mod
    /// modulus` of head `h`'s plane of `planes` rows.
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
            words.len() > word,
            modulus >= 1,
            offset + modulus <= planes,
            src.len() >= n * n_kv * 128,
            dst.len() >= n_kv * planes * 64
        )
    )]
    pub fn dspark_kv_put(
        src: &[f32],
        words: &[u32],
        word: u32,
        offset: u32,
        modulus: u32,
        n_kv: u32,
        planes: u32,
        n: u32,
        mut dst: DisjointSlice<u32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let nkv = n_kv as usize;
        if b >= n as usize * nkv {
            return; // block-uniform
        }
        let t = b / nkv;
        let h = b - t * nkv;
        let i = thread::threadIdx_x() as usize;
        // SAFETY: word < words.len() by the launch contract.
        let start = unsafe { *words.get_unchecked(word as usize) } as usize;
        let slot = offset as usize + (start + t) % modulus as usize;
        let s = (t * nkv + h) * HEAD + 2 * i;
        // SAFETY: t < n and h < n_kv, so s + 1 < n·n_kv·HEAD <= src.len(); slot < offset + modulus
        // <= planes, so the word is inside head h's plane; thread i of block b owns it.
        unsafe {
            let bits = f32x2_to_f16x2_bits(*src.get_unchecked(s), *src.get_unchecked(s + 1));
            *dst.get_unchecked_mut((h * planes as usize + slot) * ROW_WORDS + i) = bits;
        }
    }
}

/// [`BlockAttn::enqueue_attn`]'s arguments.
pub struct AttnArgs<'a> {
    /// The rows' query heads, `m · n_kv · GROUP · HEAD` f32, token-major.
    pub q: &'a DeviceBuffer<f32>,
    /// The layer's K and V planes, `n_kv · (cap + ROWS) · HEAD / 2` words each.
    pub kc: &'a DeviceBuffer<u32>,
    pub vc: &'a DeviceBuffer<u32>,
    /// The stage words; `words[word]` is the ring rows a row reads.
    pub words: &'a DeviceBuffer<u32>,
    pub word: usize,
    /// The ring's rows.
    pub cap: usize,
    pub n_kv: usize,
    /// The block's rows, `1..=ROWS`.
    pub m: usize,
    pub scale: f32,
    pub fault: FaultSink,
    /// [`partials_v_len`] and [`partials_ms_len`] of `m` rows.
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    /// The output, `m · n_kv · GROUP · HEAD` f32, token-major.
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`BlockAttn::enqueue_put`]'s arguments.
pub struct PutArgs<'a> {
    /// `n` rows of `n_kv · HEAD` f32, token-major.
    pub src: &'a DeviceBuffer<f32>,
    /// `words[word]` is the first row's start before the offset.
    pub words: &'a DeviceBuffer<u32>,
    pub word: usize,
    pub offset: usize,
    pub modulus: usize,
    pub n_kv: usize,
    /// Rows of each key head's plane.
    pub planes: usize,
    pub n: usize,
    /// The plane, `n_kv · planes · HEAD / 2` words.
    pub dst: &'a mut DeviceBuffer<u32>,
}

/// The value partials' length for `m` rows of `n_q` query heads.
#[must_use]
pub const fn partials_v_len(m: usize, n_q: usize) -> usize {
    m * n_q * SEGS * HEAD
}

/// The `(max, sum)` partials' length for `m` rows of `n_q` query heads.
#[must_use]
pub const fn partials_ms_len(m: usize, n_q: usize) -> usize {
    m * n_q * SEGS * 2
}

/// The loaded block-attention module. Owns no stream.
pub struct BlockAttn {
    module: block_attn_kernels::LoadedModule,
}

impl BlockAttn {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<BlockAttn, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; each
        // launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(block_attn_kernels, ctx)? };
        Ok(BlockAttn { module })
    }

    /// Enqueue the segment pass and the merge (module doc). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_attn(&self, stream: &CudaStream, a: AttnArgs<'_>) -> Result<(), GpuError> {
        let what = "BlockAttn::enqueue_attn";
        let AttnArgs {
            q,
            kc,
            vc,
            words,
            word,
            cap,
            n_kv,
            m,
            scale,
            fault,
            part_v,
            part_ms,
            y,
        } = a;
        let n_q = n_kv * GROUP;
        let plane = n_kv * (cap + ROWS) * ROW_WORDS;
        if n_kv == 0
            || cap == 0
            || !(1..=ROWS).contains(&m)
            || word >= words.len()
            || q.len() < m * n_q * HEAD
            || kc.len() < plane
            || vc.len() < plane
            || part_v.len() < partials_v_len(m, n_q)
            || part_ms.len() < partials_ms_len(m, n_q)
            || y.len() < m * n_q * HEAD
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_kv} key heads, a ring of {cap}, {m} rows (1..={ROWS}), word {word} of {}, \
                     q {}, planes {} and {} (need {plane}), partials {} and {}, y {}",
                    words.len(),
                    q.len(),
                    kc.len(),
                    vc.len(),
                    part_v.len(),
                    part_ms.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", SEGS * n_kv)?;
        let prep = self
            .module
            .prepare_dspark_g3_seg(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.dspark_g3_seg(
            stream,
            &prep,
            q,
            kc,
            vc,
            words,
            launch_u32(what, "word", word)?,
            scale,
            launch_u32(what, "n_kv", n_kv)?,
            launch_u32(what, "cap", cap)?,
            launch_u32(what, "m", m)?,
            fault,
            &mut *part_v,
            &mut *part_ms,
        )?;
        let grid = launch_u32(what, "merge grid", m * n_q)?;
        let prep =
            self.module
                .prepare_dspark_g3_merge(LaunchConfig1D::new(grid, MERGE_THREADS, 0))?;
        self.module.dspark_g3_merge(
            stream,
            &prep,
            &*part_v,
            &*part_ms,
            words,
            launch_u32(what, "word", word)?,
            launch_u32(what, "cap", cap)?,
            launch_u32(what, "n_q", n_q)?,
            launch_u32(what, "m", m)?,
            y,
        )?;
        Ok(())
    }

    /// Enqueue rows into a plane (module doc). Asynchronous, allocation-free, capturable.
    pub fn enqueue_put(&self, stream: &CudaStream, a: PutArgs<'_>) -> Result<(), GpuError> {
        let what = "BlockAttn::enqueue_put";
        let PutArgs {
            src,
            words,
            word,
            offset,
            modulus,
            n_kv,
            planes,
            n,
            dst,
        } = a;
        if n_kv == 0
            || n == 0
            || modulus == 0
            || offset + modulus > planes
            || word >= words.len()
            || src.len() < n * n_kv * HEAD
            || dst.len() < n_kv * planes * ROW_WORDS
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n} rows of {n_kv} key heads at offset {offset} modulo {modulus} of {planes} \
                     plane rows, word {word} of {}, src {}, dst {}",
                    words.len(),
                    src.len(),
                    dst.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", n * n_kv)?;
        let prep = self
            .module
            .prepare_dspark_kv_put(LaunchConfig1D::new(grid, PUT_THREADS, 0))?;
        self.module.dspark_kv_put(
            stream,
            &prep,
            src,
            words,
            launch_u32(what, "word", word)?,
            launch_u32(what, "offset", offset)?,
            launch_u32(what, "modulus", modulus)?,
            launch_u32(what, "n_kv", n_kv)?,
            launch_u32(what, "planes", planes)?,
            launch_u32(what, "n", n)?,
            dst,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;

    /// The host rule of [`BlockAttn::enqueue_attn`] in f64 over the planes' f16 values: each row's query
    /// head `h` attends to key head `h / GROUP`'s `vis` ring rows and `m` block rows. `planes` are the
    /// f16 bits of the keys and the values, `n_kv · (cap + ROWS) · HEAD` each.
    fn attn_host(
        q: &[f32],
        (kc, vc): (&[u16], &[u16]),
        vis: usize,
        cap: usize,
        n_kv: usize,
        m: usize,
        scale: f32,
    ) -> Vec<f32> {
        let n_q = n_kv * GROUP;
        let planes = cap + ROWS;
        let f16 = |b: u16| f64::from(gguf::quant::half_to_f32(b));
        let mut y = vec![0.0f32; m * n_q * HEAD];
        for t in 0..m {
            for h in 0..n_q {
                let kh = h / GROUP;
                let qr = &q[(t * n_q + h) * HEAD..][..HEAD];
                let keys: Vec<usize> = (0..vis + m).map(|j| key_slot(j, vis, cap)).collect();
                let sc: Vec<f64> = keys
                    .iter()
                    .map(|&slot| {
                        let k = &kc[(kh * planes + slot) * HEAD..][..HEAD];
                        let dot: f64 = qr.iter().zip(k).map(|(&a, &b)| f64::from(a) * f16(b)).sum();
                        dot * f64::from(scale)
                    })
                    .collect();
                let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                let sum: f64 = p.iter().sum();
                for d in 0..HEAD {
                    let v: f64 = keys
                        .iter()
                        .zip(&p)
                        .map(|(&slot, &w)| w * f16(vc[(kh * planes + slot) * HEAD + d]))
                        .sum();
                    y[(t * n_q + h) * HEAD + d] = (v / sum) as f32;
                }
            }
        }
        y
    }

    /// Seeded values in [−2, 2).
    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 4_194_304.0 - 2.0
            })
            .collect()
    }

    /// The card's block attention is the host rule within f32 rounding: Inkling 2's draft heads
    /// (8 key heads, 24 query heads of 128) over a ring of 64 rows read 37 deep (the block's rows
    /// not adjacent to the ring's) and full, at 8 rows and at 3; the block's K and V written by the
    /// put launch at its rows behind the ring and a context row wrapped to the ring's front; a stage
    /// word past the ring writes NaN.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_g3_attention_is_the_host_rule() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let k = BlockAttn::load(gpu.context()).expect("the block attention module");
        let (n_kv, cap) = (8, 64);
        let n_q = n_kv * GROUP;
        let planes = cap + ROWS;
        let scale = 1.0 / (HEAD as f32).sqrt();
        let mut worst = 0.0f32;
        for (i, &(vis, m)) in [(37usize, 8usize), (64, 8), (37, 3)].iter().enumerate() {
            let seed = 11 + 5 * i as u32;
            // The ring's rows as f16 bits; the block's rows come through the put launch.
            let ring_k: Vec<u16> = seeded(n_kv * planes * HEAD, seed)
                .iter()
                .map(|&v| gguf::quant::f32_to_f16_bits(v))
                .collect();
            let ring_v: Vec<u16> = seeded(n_kv * planes * HEAD, seed + 1)
                .iter()
                .map(|&v| gguf::quant::f32_to_f16_bits(v))
                .collect();
            let blk_k = seeded(m * n_kv * HEAD, seed + 2);
            let blk_v = seeded(m * n_kv * HEAD, seed + 3);
            let q = seeded(m * n_q * HEAD, seed + 4);
            // The host planes: the block's rows at cap + t, f16 of the f32 rows.
            let mut host_k = ring_k.clone();
            let mut host_v = ring_v.clone();
            for t in 0..m {
                for h in 0..n_kv {
                    for d in 0..HEAD {
                        let at = (h * planes + cap + t) * HEAD + d;
                        let from = (t * n_kv + h) * HEAD + d;
                        host_k[at] = gguf::quant::f32_to_f16_bits(blk_k[from]);
                        host_v[at] = gguf::quant::f32_to_f16_bits(blk_v[from]);
                    }
                }
            }
            let want = attn_host(&q, (&host_k, &host_v), vis, cap, n_kv, m, scale);
            let words_of = |w: &[u16]| -> Vec<u32> {
                w.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16))
                    .collect()
            };
            let mut kc = DeviceBuffer::from_host(stream, &words_of(&ring_k)).expect("kc");
            let mut vc = DeviceBuffer::from_host(stream, &words_of(&ring_v)).expect("vc");
            // Word 0 the ring rows read, word 1 the block's start (0).
            let words = DeviceBuffer::from_host(stream, &[vis as u32, 0]).expect("words");
            let bk = DeviceBuffer::from_host(stream, &blk_k).expect("blk k");
            let bv = DeviceBuffer::from_host(stream, &blk_v).expect("blk v");
            for (src, dst) in [(&bk, &mut kc), (&bv, &mut vc)] {
                k.enqueue_put(
                    stream,
                    PutArgs {
                        src,
                        words: &words,
                        word: 1,
                        offset: cap,
                        modulus: ROWS,
                        n_kv,
                        planes,
                        n: m,
                        dst,
                    },
                )
                .expect("put");
            }
            let qd = DeviceBuffer::from_host(stream, &q).expect("q");
            let mut pv = DeviceBuffer::zeroed(stream, partials_v_len(m, n_q)).expect("part v");
            let mut pm = DeviceBuffer::zeroed(stream, partials_ms_len(m, n_q)).expect("part ms");
            let mut y = DeviceBuffer::zeroed(stream, m * n_q * HEAD).expect("y");
            k.enqueue_attn(
                stream,
                AttnArgs {
                    q: &qd,
                    kc: &kc,
                    vc: &vc,
                    words: &words,
                    word: 0,
                    cap,
                    n_kv,
                    m,
                    scale,
                    fault: gpu.unlabelled_sink(),
                    part_v: &mut pv,
                    part_ms: &mut pm,
                    y: &mut y,
                },
            )
            .expect("attn");
            let got = y.to_host_vec(stream).expect("y back");
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("g3 attention: vis {vis} of {cap}, m {m}: max |card - host| {err:.3e}");
            worst = worst.max(err);
        }
        // A context row put at ring row (62 + 3) mod 64 = 1: the put wraps to the ring's front.
        let row = seeded(n_kv * HEAD, 99);
        let src = DeviceBuffer::from_host(
            stream,
            &[row.clone(), row.clone(), row.clone(), row.clone()].concat(),
        )
        .expect("rows");
        let words = DeviceBuffer::from_host(stream, &[62u32]).expect("start");
        let mut dst = DeviceBuffer::zeroed(stream, n_kv * planes * ROW_WORDS).expect("plane");
        k.enqueue_put(
            stream,
            PutArgs {
                src: &src,
                words: &words,
                word: 0,
                offset: 0,
                modulus: cap,
                n_kv,
                planes,
                n: 4,
                dst: &mut dst,
            },
        )
        .expect("wrapped put");
        let plane = dst.to_host_vec(stream).expect("plane back");
        let at = |h: usize, slot: usize, d: usize| {
            let w = plane[(h * planes + slot) * ROW_WORDS + d / 2];
            gguf::quant::half_to_f32(if d.is_multiple_of(2) {
                w as u16
            } else {
                (w >> 16) as u16
            })
        };
        for (slot, live) in [
            (61, false),
            (62, true),
            (63, true),
            (0, true),
            (1, true),
            (2, false),
        ] {
            let got = at(5, slot, 7);
            let want = if live {
                gguf::quant::half_to_f32(gguf::quant::f32_to_f16_bits(row[5 * HEAD + 7]))
            } else {
                0.0
            };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "ring row {slot} after a wrapped put"
            );
        }
        // A stage word past the ring: NaN rows.
        let words = DeviceBuffer::from_host(stream, &[cap as u32 + 1]).expect("bad word");
        let kc = DeviceBuffer::zeroed(stream, n_kv * planes * ROW_WORDS).expect("kc");
        let vc = DeviceBuffer::zeroed(stream, n_kv * planes * ROW_WORDS).expect("vc");
        let qd = DeviceBuffer::from_host(stream, &seeded(n_q * HEAD, 5)).expect("q");
        let mut pv = DeviceBuffer::zeroed(stream, partials_v_len(1, n_q)).expect("part v");
        let mut pm = DeviceBuffer::zeroed(stream, partials_ms_len(1, n_q)).expect("part ms");
        let mut y = DeviceBuffer::zeroed(stream, n_q * HEAD).expect("y");
        k.enqueue_attn(
            stream,
            AttnArgs {
                q: &qd,
                kc: &kc,
                vc: &vc,
                words: &words,
                word: 0,
                cap,
                n_kv,
                m: 1,
                scale,
                fault: gpu.unlabelled_sink(),
                part_v: &mut pv,
                part_ms: &mut pm,
                y: &mut y,
            },
        )
        .expect("refused attn");
        let got = y.to_host_vec(stream).expect("y back");
        assert!(
            got.iter().all(|v| v.is_nan()),
            "a stage word past the ring writes NaN rows"
        );
        let fault = gpu.take_fault().expect("the fault word");
        assert!(
            fault.is_some_and(|f| f.code == FaultSite::KeyCount as u32),
            "a stage word past the ring raises KeyCount: {fault:?}"
        );
        // f32 dots of 128 values and dev_exp against an f64 rule: values within 2^-13 of a value
        // of magnitude 2.
        assert!(
            worst <= 2.5e-4,
            "block attention against the host rule: max {worst:.3e}"
        );
    }
}
