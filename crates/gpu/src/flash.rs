//! P5: the KV append (f32 -> f16) and the latent (MLA) flash attention over
//! the absorbed cache (docs/gpu-design.md 작업 꾸러미 P5). One cache row per
//! token per layer, `[rope_dims | latent]` u16 with the rope part first —
//! the row order the CPU engine's `kvr` columns carry (`crates/model/src/
//! kv.rs`, oracle-verified there) — and one f32 query row per (token, head),
//! `[q_rope | q_nope2]` in the same order. The attention output is the
//! softmax(scale · q·k)-weighted sum of the rows' latent tails, projected
//! later by `attn_kv_b`'s V part (not this package).
//!
//! The f16 rounding reuses `model::attn::f32_to_f16_bits` itself — the one
//! owner of the CPU oracle's conversion — so cache bits are equal by
//! construction, not by transcription; the gate asserts it element for
//! element on real `kvr-L` rows and on the IEEE edges.
//!
//! Launch geometry: one CUDA block of [`LATENT`] threads per query row and
//! key segment, so a thread owns exactly one latent dim of that row's output
//! for the whole run and needs no accumulator array. Everything a thread must
//! share with the rest of the block (the staged query row, a tile's logits,
//! its weights, the rescale) goes through shared memory.
//!
//! A decode step's 16 heads are 16 query rows. Sixteen blocks leave most of
//! the card idle, so the key range is cut into [`seg_keys`]-key segments and
//! block `(row, segment)` attends only its own slice: the grid is
//! `q_rows * segments` and the launch is two kernels, [`flash_latent_seg`]
//! writing each segment's softmax partials (running max, `Σ exp` relative to
//! it, and the un-normalised `Σ exp·V`) and `flash_merge` folding the
//! segments of a row into the final latent row. The segment count comes from
//! the cache height, not from the live key count, so a captured graph's grid
//! is fixed and the step still follows `n_keys_buf` — a segment wholly past
//! the causal limit writes a neutral partial (`m = −inf`, `s = 0`) and reads
//! no key row at all, which is also what keeps padded rows holding NaN out of
//! every result. A cache short enough to hold one segment takes the
//! single-launch [`flash_latent`] instead, which needs no partials and no
//! merge; that choice is made from the cache height when the launch is
//! enqueued, never per key count.
//!
//! Merge order is fixed (ascending segment), so reruns and graph replays are
//! bit-identical; the segment split does move the floating-point order
//! against the single-block form, which is why both live inside the same
//! band rather than being bit-equal to each other.
//!
//! Reduction structure (the fixed, deterministic contract of this family;
//! reruns are bit-identical, CPU bit-identity is not claimed):
//! - QK dot per key: the block splits as [`KEY_TILE`] keys x [`DIM_SPLIT`]
//!   threads. Thread `d` of a key walks dims `d, d + DIM_SPLIT, …` into
//!   [`ILP`] rotating f32 partials (one fused multiply-add each per group of
//!   `ILP` steps, so that many loads are in flight at once), combines them by
//!   `(a0+a1)+(a2+a3)` with the trailing steps folded into `a0`, and the
//!   key's threads combine by the fixed four-step xor butterfly.
//! - Online softmax in `KEY_TILE`-key tiles, keys ascending — the fa4 tile
//!   size. Warp 0 owns the running `(m, s)`: tile max by the five-step
//!   butterfly max, weights `exp(s − m)` per lane, weight sum by the
//!   five-step butterfly sum, state rescale on a max bump (`s` immediately,
//!   the V partials before the tile's accumulation), and publishes the
//!   tile's 32 weights and the rescale to the block.
//! - V accumulation: every thread accumulates its own latent dim over the
//!   tile's keys into [`ILP`] rotating partials (keys `l … l + ILP − 1`), one
//!   fused multiply-add each, combined by the same fixed tree once at the
//!   end.
//! - Final row: `r · (1/s)`, one plain multiply.
//!
//! Keys at or past the causal limit (mask, cache padding) carry weight
//! exactly `0.0`. Only the last tile of a run can reach past the limit, and
//! it takes a guarded path whose `wl == 0.0` test skips those loads — so
//! padded rows holding NaN bit patterns are never read into any result. A
//! tile wholly inside the limit reads only real rows and needs no guard.
//!
//! Beside the shipped entries this module carries one probe entry per stage
//! of the segment walk, and one for the merge fold ([`TWICE_QK`] and its
//! siblings). Each runs its stage a second time and folds that result in
//! with a zero launch scalar, so it writes what the shipped entry writes and
//! costs what the shipped entry costs plus that stage — a same-binary A/B
//! over them prices the stages of an attention step against each other.
//! They are instruments; no step ships through one.

use crate::GpuError;
use crate::q8_1_quant_vals;
use crate::tensor::{DeviceTensor, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::cvt_f32_f16x2_lo;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The CPU oracle's f32 -> f16 conversion, reused as this module's device
/// and host rounding so the cache bits cannot drift from the engine's.
pub use model::attn::f32_to_f16_bits;

/// Latent tail this kernel family's geometry is built for, and so the block's
/// thread count: one thread per latent dim. `enqueue_flash_latent` rejects
/// any other latent width.
pub const LATENT: usize = 512;
/// Keys per online-softmax tile — one warp's worth, so a tile's max and
/// weight sum are single warp butterflies.
pub const KEY_TILE: usize = 32;
/// Threads sharing one key's QK dot; the block's `LATENT` threads cover
/// `KEY_TILE` keys at a time.
pub const DIM_SPLIT: usize = LATENT / KEY_TILE;
/// Widest `rope_dims + latent` row the shared staging buffer holds.
pub const MAX_WIDTH: usize = 640;
/// Rotating partials each hot loop carries, and so the loads a thread keeps
/// in flight. Four is measured, not derived: eight measured slower at depth
/// on this card, so the loops are not short of memory-level parallelism. The
/// partials are combined by a fixed tree, never by the loop order.
pub const ILP: usize = 4;
/// Keys one segment block of the split launch walks. A multiple of
/// [`KEY_TILE`], so only the last live segment of a row can meet a partial
/// tile and the guarded tail path is the one already there. The value is
/// measured, not derived: a segment twice this size still fills the card at
/// four thousand keys but leaves it half empty at one thousand, and this one
/// measured faster at both depths.
pub const SEG_KEYS: usize = 128;

/// Keys per segment, [`SEG_KEYS`] unless `BLOOMERY_FLASH_SEG` names another
/// multiple of [`KEY_TILE`]. Read once, at first use: the value fixes a
/// captured graph's grid, so it must not change between capture and replay.
/// A value that is set but unusable panics rather than falling back — a
/// sweep row that silently ran the default would be a wrong measurement.
pub fn seg_keys() -> usize {
    static SEG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SEG.get_or_init(|| match std::env::var("BLOOMERY_FLASH_SEG") {
        Err(_) => SEG_KEYS,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= KEY_TILE && n.is_multiple_of(KEY_TILE) => n,
            _ => {
                panic!("BLOOMERY_FLASH_SEG={v} is not a positive multiple of KEY_TILE ({KEY_TILE})")
            }
        },
    })
}

/// Stages of the segment pass one probe entry does a second time, as the
/// bits of [`FlashKernels::enqueue_flash_latent_seg_twice`]'s `twice` and of
/// `latent_range`'s const parameter. A probe entry is a TIMING INSTRUMENT:
/// it folds its second result in with [`PROBE_JIG`], so its partials are the
/// shipped entry's and the slowdown against them is that stage's price in
/// the step. The shipped entries pass [`TWICE_NONE`] and every probe branch
/// folds away at compile time.
pub const TWICE_NONE: u32 = 0;
/// The QK dot loop, over the thread's own key row or another (`shift`).
pub const TWICE_QK: u32 = 1;
/// The V accumulation loop, over the same tile's rows or another's.
pub const TWICE_V: u32 = 1 << 1;
/// The key butterfly, and the tile max and weight sum warp reductions.
pub const TWICE_COLL: u32 = 1 << 2;
/// The tile's two block barriers.
pub const TWICE_SYNC: u32 = 1 << 3;
/// Warp 0's per-lane softmax arithmetic, both exponentials included.
pub const TWICE_SM: u32 = 1 << 4;

/// The weight a probe entry folds its second result in with:
/// `fma(PROBE_JIG, second, first)`. Zero, so the fold returns `first` to the
/// bit — every value a probe folds is finite — and a launch scalar rather
/// than a literal, so no pass can see that it is zero and delete the second
/// pass it weighs.
pub const PROBE_JIG: f32 = 0.0;

/// Segments a `cache_rows`-tall cache is cut into. One means the cache fits
/// in a single segment and the single-launch [`flash_latent`] serves it.
pub fn segments_for(cache_rows: usize) -> usize {
    cache_rows.max(1).div_ceil(seg_keys())
}

/// Length the split launch's `Σ exp·V` partials buffer needs for `q_rows`
/// query rows over a `cache_rows`-tall cache.
pub fn partials_v_len(q_rows: usize, cache_rows: usize) -> usize {
    q_rows * segments_for(cache_rows) * LATENT
}

/// Length the split launch's `(max, Σ exp)` partials buffer needs — two f32
/// per (query row, segment).
pub fn partials_ms_len(q_rows: usize, cache_rows: usize) -> usize {
    q_rows * segments_for(cache_rows) * 2
}

// --------------------------------------------------------------- cores

/// `exp(x)` on device: the hardware's `ex2.approx.f32` on `x · log2(e)` — a
/// deterministic instruction whose ~1e-7 relative error sits far inside every
/// band this package gates against, with no dependence on host libm.
#[inline(always)]
pub fn dev_exp(x: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(x * std::f32::consts::LOG2_E)
}

/// One `f16` bit pattern widened to `f32` by the hardware's widening
/// convert. Widening `f16` to `f32` is exact, so this is
/// `cores::half_to_f32`'s value for every finite and infinite input — one
/// instruction instead of a decode whose subnormal branch is a loop.
#[inline(always)]
pub fn half_bits_to_f32(bits: u16) -> f32 {
    cvt_f32_f16x2_lo(bits as u32)
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod flash_kernels {
    use super::*;

    /// Convert `m` new KV rows (`m * width` f32, row-major, token-major) to
    /// f16 with the oracle's rounding and store them at rows
    /// `pos..pos + m` of the preallocated cache. `pos` is a launch scalar:
    /// a captured graph replays it frozen at its capture-time value — the
    /// captured step wants `kv_append_pos_buf`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= m * width, dst.len() >= (pos + m) * width)
    )]
    pub fn kv_append(m: u32, width: u32, pos: u32, src: &[f32], mut dst: DisjointSlice<u16>) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        let row = i / w;
        let col = i - row * w;
        // SAFETY: i < total <= src.len(); row < m and col < width, so the
        // store index is < (pos + m) * width <= dst.len() (launch contract).
        unsafe {
            *dst.get_unchecked_mut((pos as usize + row) * w + col) =
                f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// `kv_append` with `pos` read from `pos_buf[0]` at run time — the
    /// variant a captured decode step replays against a new position by
    /// rewriting the buffer between launches. Rows that would land at or
    /// past `dst_rows` (the cache's allocated height) are skipped: `pos`
    /// comes from device memory, so that bound cannot be a launch contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            pos_buf.len() >= 1,
            src.len() >= m * width,
            dst.len() >= dst_rows * width
        )
    )]
    pub fn kv_append_pos_buf(
        m: u32,
        width: u32,
        dst_rows: u32,
        pos_buf: &[u32],
        src: &[f32],
        mut dst: DisjointSlice<u16>,
    ) {
        let w = width as usize;
        let total = m as usize * w;
        let i = thread::index_1d().get();
        if i >= total {
            return;
        }
        // SAFETY: pos_buf.len() >= 1 by the launch contract.
        let pos = unsafe { *pos_buf.get_unchecked(0) } as usize;
        let row = i / w;
        if pos + row >= dst_rows as usize {
            return;
        }
        // SAFETY: i < total <= src.len(); pos + row < dst_rows and col <
        // width, so the store index is < dst_rows * width <= dst.len().
        unsafe {
            let col = i - row * w;
            *dst.get_unchecked_mut((pos + row) * w + col) = f32_to_f16_bits(*src.get_unchecked(i));
        }
    }

    /// Latent flash attention over the absorbed KV cache, `m` query tokens
    /// (1..=8) with every head in one launch. `q` holds `m * n_heads` rows
    /// of `width = rope_dims + latent` f32, row `t * n_heads + h` (the
    /// `kqv_compressed` column order); `kv` is the `[dst_rows x width]` u16
    /// cache; the live key count is `n_keys_buf[0]`. Query `t` attends to
    /// keys `0..n_keys − m + t` (its own rows are already appended, so
    /// `n_keys >= m`); `y` holds `m * n_heads` rows of `latent` f32 in the
    /// same row order. Deterministic by the module doc's reduction contract.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            y.len() >= q_rows * latent
        )
    )]
    pub fn flash_latent(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // The query row, staged once: every key's dot reads all of it.
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        // The tile's scaled logits, then its weights, then the max-bump
        // rescale at [0] and the final `1/s` at [1].
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let rope = rope_dims as usize;
        let lat = latent as usize;
        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw pointer form is the only way to reach it
        // without a reference to a `static mut`. Every access below is
        // bounded by the array's length and ordered by `sync_threads`.
        let (qs, klog, kw, st) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QROW),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut ST),
            )
        };
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        // SAFETY: the pointers are this block's shared scratch, `row` is
        // inside q's rows and `0..limit` inside kv's (launch contract).
        let (_, s_sum, r) = unsafe {
            latent_range::<TWICE_NONE>(
                q, kv, qs, klog, kw, st, row, tid, 0, limit, rope, lat, scale, 0, PROBE_JIG,
            )
        };

        if tid == 0 {
            // SAFETY: thread 0 alone writes ST[1]; no thread reads it before
            // the barrier below. `s_sum` is the block state, valid in warp 0.
            unsafe {
                *st.add(1) = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            }
        }
        thread::sync_threads();
        // SAFETY: ST[1] is published above.
        let s_inv = unsafe { *st.add(1) };
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * lat + tid) = s_inv * r;
        }
    }

    /// [`flash_latent`] with [`flash_merge_q8`]'s q8_1 side output. The
    /// single-segment path never reaches the merge, so without this twin a
    /// cache short enough to hold one segment would take the folded step
    /// down a path that does not quantize at all — the two paths have to
    /// agree or neither may fold.
    ///
    /// The staging buffer is `QROW`, the query row's own: the last read of
    /// it is inside `latent_range`, which ends at a `sync_threads` before
    /// this writes.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            y.len() >= q_rows * latent,
            q3a.len() >= m_lo * 64 * half_it,
            q4a.len() >= m_lo * 256 * quad_it,
            q6a.len() >= m_lo * 128 * half_it,
            s8a.len() >= m_lo * 8 * n_sb,
            d8a.len() >= m_lo * 2 * n_sb,
            q3b.len() >= (q_rows - m_lo) * 64 * half_it,
            q4b.len() >= (q_rows - m_lo) * 256 * quad_it,
            q6b.len() >= (q_rows - m_lo) * 128 * half_it,
            s8b.len() >= (q_rows - m_lo) * 8 * n_sb,
            d8b.len() >= (q_rows - m_lo) * 2 * n_sb
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_q8(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut y: DisjointSlice<f32>,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let rope = rope_dims as usize;
        let lat = latent as usize;
        // SAFETY: as in `flash_latent` — each `static mut` is this block's
        // own shared allocation, every access bounded and ordered by
        // `sync_threads`.
        let (qs, klog, kw, st) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QROW),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut ST),
            )
        };
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        // SAFETY: the pointers are this block's shared scratch, `row` is
        // inside q's rows and `0..limit` inside kv's (launch contract).
        let (_, s_sum, r) = unsafe {
            latent_range::<TWICE_NONE>(
                q, kv, qs, klog, kw, st, row, tid, 0, limit, rope, lat, scale, 0, PROBE_JIG,
            )
        };

        if tid == 0 {
            // SAFETY: thread 0 alone writes ST[1]; no thread reads it before
            // the barrier below. `s_sum` is the block state, valid in warp 0.
            unsafe {
                *st.add(1) = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            }
        }
        thread::sync_threads();
        // SAFETY: ST[1] is published above.
        let s_inv = unsafe { *st.add(1) };
        let v = s_inv * r;
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * lat + tid) = v;
        }
        // SAFETY: `qs` is this block's `MAX_WIDTH >= LATENT` shared array
        // and its last read is inside `latent_range`, before the barrier
        // above; the helper's own preconditions hold as in `flash_merge_q8`.
        unsafe {
            quant_row(
                v, qs, tid, lat, row, m_lo, n_sb, half_it, quad_it, &mut q3a, &mut q4a, &mut q6a,
                &mut s8a, &mut d8a, &mut q3b, &mut q4b, &mut q6b, &mut s8b, &mut d8b,
            );
        }
    }

    /// The segment pass of the split launch: block `(row, segment)` attends
    /// keys `[segment * seg_keys, min((segment + 1) * seg_keys, limit))` of
    /// query row `row` and writes that segment's partials — `part_ms` holds
    /// `(running max, Σ exp)` per (row, segment), `part_v` the row's
    /// un-normalised `Σ exp·V` relative to that max. Segment is the slow
    /// block index, so the blocks sharing a key slice (every head of the
    /// token) run together and hit the same cache rows in L2.
    ///
    /// A segment wholly past the row's causal limit writes the neutral
    /// partial `(−inf, 0)` and returns without reading a key row: `part_v`
    /// stays untouched there and the merge skips it, which is what keeps
    /// padded rows holding NaN out of every result.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: each `static mut` above is this block's own shared
        // allocation, which is `seg_pass`'s precondition on the scratch; the
        // rest of it is this kernel's launch contract.
        unsafe {
            seg_pass::<TWICE_NONE>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                0,
                PROBE_JIG,
            );
        }
    }

    /// The segment pass's body, shared by [`flash_latent_seg`] and the probe
    /// entries below so the thing timed is the thing shipped: the block's
    /// `(row, segment)` bookkeeping, the neutral partial of a segment past
    /// the limit, the walk, and the partial stores. `TWICE`, `shift` and
    /// `jig` go straight to `latent_range`.
    ///
    /// SAFETY: `scratch` is the calling block's own
    /// `(MAX_WIDTH, KEY_TILE, KEY_TILE, 2)` f32 shared arrays, and the
    /// arguments satisfy [`flash_latent_seg`]'s launch contract.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn seg_pass<const TWICE: u32>(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        part_v: &mut DisjointSlice<f32>,
        part_ms: &mut DisjointSlice<f32>,
        scratch: (*mut f32, *mut f32, *mut f32, *mut f32),
        shift: usize,
        jig: f32,
    ) {
        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let rows = q_rows as usize;
        let n_seg = segs as usize;
        if b >= rows * n_seg {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let seg = b / rows;
        let row = b - seg * rows;
        let idx = row * n_seg + seg;
        let rope = rope_dims as usize;
        let lat = latent as usize;
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        let lo = seg * seg_keys as usize;
        if lo >= limit {
            if tid == 0 {
                // SAFETY: idx < q_rows * segs, so both slots are inside
                // part_ms (launch contract).
                unsafe {
                    *part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform, and no key row of this segment is read
        }
        let hi = (lo + seg_keys as usize).min(limit);
        let (qs, klog, kw, st) = scratch;
        // SAFETY: the pointers are this block's shared scratch, `row` is
        // inside q's rows and `lo..hi <= limit <= dst_rows` inside kv's.
        let (mx, s_sum, r) = unsafe {
            latent_range::<TWICE>(
                q, kv, qs, klog, kw, st, row, tid, lo, hi, rope, lat, scale, shift, jig,
            )
        };
        // SAFETY: idx < q_rows * segs and tid < LATENT = latent
        // (host-validated), so both stores are inside their buffers.
        unsafe {
            *part_v.get_unchecked_mut(idx * lat + tid) = r;
            if tid == 0 {
                *part_ms.get_unchecked_mut(2 * idx) = mx;
                *part_ms.get_unchecked_mut(2 * idx + 1) = s_sum;
            }
        }
    }

    /// [`flash_latent_seg`] with the QK dot loop run a second time — a
    /// probe entry, not a path the step ships.
    ///
    /// Every probe entry in this family takes the segment pass's parameters
    /// plus `shift`, the second pass's row offset inside the segment (0
    /// re-reads the rows just read), and `jig`, the weight its second result
    /// folds in with. The host passes [`PROBE_JIG`], so every one of them
    /// writes [`flash_latent_seg`]'s partials to the bit and the only thing
    /// that separates them is time. See [`TWICE_QK`] and its siblings.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg_qk2(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: as in `flash_latent_seg` — the shared arrays are this
        // block's own and the arguments are that kernel's launch contract.
        unsafe {
            seg_pass::<TWICE_QK>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                shift as usize,
                jig,
            );
        }
    }

    /// [`flash_latent_seg`] with the V accumulation loop run a second time
    /// — a probe entry ([`TWICE_V`]). Of the family this is the one whose
    /// second pass needs registers of its own; it compiles to a wider
    /// budget than the shipped entry and a small local spill, at the same
    /// two blocks resident per multiprocessor. Keeping that residency is
    /// the point — an arm one block short would price the lost residency
    /// and not the loop.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg_v2(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: as in `flash_latent_seg_qk2`.
        unsafe {
            seg_pass::<TWICE_V>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                shift as usize,
                jig,
            );
        }
    }

    /// [`flash_latent_seg`] with the butterfly and the two warp reductions
    /// run a second time — a probe entry ([`TWICE_COLL`]).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg_coll2(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: as in `flash_latent_seg_qk2`.
        unsafe {
            seg_pass::<TWICE_COLL>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                shift as usize,
                jig,
            );
        }
    }

    /// [`flash_latent_seg`] with a second pair of block barriers per tile —
    /// a probe entry ([`TWICE_SYNC`]).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg_sync2(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: as in `flash_latent_seg_qk2`.
        unsafe {
            seg_pass::<TWICE_SYNC>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                shift as usize,
                jig,
            );
        }
    }

    /// [`flash_latent_seg`] with warp 0's softmax arithmetic run a second
    /// time — a probe entry ([`TWICE_SM`]).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            q.len() >= q_rows * (rope_dims + latent),
            kv.len() >= dst_rows * (rope_dims + latent),
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_latent_seg_sm2(
        q: &[f32],
        kv: &[u16],
        n_keys_buf: &[u32],
        scale: f32,
        m: u32,
        n_heads: u32,
        q_rows: u32,
        rope_dims: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut QROW: SharedArray<f32, MAX_WIDTH> = SharedArray::UNINIT;
        static mut KLOG: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, KEY_TILE> = SharedArray::UNINIT;
        static mut ST: SharedArray<f32, 2> = SharedArray::UNINIT;

        // SAFETY: as in `flash_latent_seg_qk2`.
        unsafe {
            seg_pass::<TWICE_SM>(
                q,
                kv,
                n_keys_buf,
                scale,
                m,
                n_heads,
                q_rows,
                rope_dims,
                latent,
                dst_rows,
                segs,
                seg_keys,
                &mut part_v,
                &mut part_ms,
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QROW),
                    SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                    SharedArray::as_raw_mut_ptr(&raw mut KW),
                    SharedArray::as_raw_mut_ptr(&raw mut ST),
                ),
                shift as usize,
                jig,
            );
        }
    }

    /// The merge pass of the split launch: one block per query row, one
    /// thread per latent dim, folding the row's partials by the standard
    /// online-softmax rescale in ascending segment order — a fixed order, so
    /// reruns and graph replays are bit-identical. The scan stops at the last
    /// segment the row's causal limit reaches, which is why this kernel reads
    /// `n_keys_buf` too: the grid is the cache's, the work is the live key
    /// count's. A neutral partial (`Σ exp == 0`) is skipped as well, so a
    /// segment past the limit is never folded in and its `part_v` slice is
    /// never read.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2,
            y.len() >= q_rows * latent
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_merge(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        q_rows: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        part_v: &[f32],
        part_ms: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let lat = latent as usize;
        // SAFETY: row < q_rows and tid < LATENT = latent (host-validated),
        // so the store is inside y's row segment (launch contract), and the
        // partial reads are inside theirs.
        unsafe {
            let v = merge_row::<false>(
                n_keys_buf, m, n_heads, dst_rows, segs, seg_keys, part_v, part_ms, row, tid, lat,
                0, PROBE_JIG,
            );
            *y.get_unchecked_mut(row * lat + tid) = v;
        }
    }

    /// [`flash_merge`] with the q8_1 side output: the same fold, the same
    /// `y`, and the merged row also leaves the block as the quantized
    /// activation the `wv_b` gemv reads — so the quantize launch that used
    /// to follow is deleted rather than merged.
    ///
    /// A block already holds a whole head's `latent` = 512 values, which is
    /// exactly four whole q8_1 blocks, so the scale reduction's partition is
    /// unchanged: warp `w` of the first four takes 128-value block `w` with
    /// lane ℓ holding values `4ℓ .. +3`, the same 32 lanes over the same 128
    /// values the standalone quantizer's warp had. The values go through
    /// shared memory because this kernel's threads hold one latent dim each
    /// and the quantizer's lanes want four consecutive — a transpose inside
    /// the block, not a different set of values.
    ///
    /// Rows below `m_lo` quantize into the `_a` outputs at column `row`, the
    /// rest into `_b` at column `row − m_lo`: the same split
    /// `q3k_quantize_q8_1_pair` made over the two halves of `kqvc`. The
    /// branch is on the block index, so no warp splits across it.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2,
            y.len() >= q_rows * latent,
            q3a.len() >= m_lo * 64 * half_it,
            q4a.len() >= m_lo * 256 * quad_it,
            q6a.len() >= m_lo * 128 * half_it,
            s8a.len() >= m_lo * 8 * n_sb,
            d8a.len() >= m_lo * 2 * n_sb,
            q3b.len() >= (q_rows - m_lo) * 64 * half_it,
            q4b.len() >= (q_rows - m_lo) * 256 * quad_it,
            q6b.len() >= (q_rows - m_lo) * 128 * half_it,
            s8b.len() >= (q_rows - m_lo) * 8 * n_sb,
            d8b.len() >= (q_rows - m_lo) * 2 * n_sb
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_merge_q8(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        q_rows: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        part_v: &[f32],
        part_ms: &[f32],
        mut y: DisjointSlice<f32>,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
    ) {
        static mut VROW: SharedArray<f32, LATENT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let lat = latent as usize;
        // SAFETY: as in `flash_merge`.
        let v = unsafe {
            let v = merge_row::<false>(
                n_keys_buf, m, n_heads, dst_rows, segs, seg_keys, part_v, part_ms, row, tid, lat,
                0, PROBE_JIG,
            );
            *y.get_unchecked_mut(row * lat + tid) = v;
            v
        };
        // SAFETY: VROW is this block's own shared allocation; tid < 512 =
        // LATENT and every read below is bounded by the same length and
        // ordered by the `sync_threads` inside the helper.
        let vs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VROW) };
        // SAFETY: the launch contract bounds both output sets; `row` picks a
        // column inside its own half, and the helper's preconditions on
        // lane, block and geometry are the ones this kernel's block shape
        // gives it.
        unsafe {
            quant_row(
                v, vs, tid, lat, row, m_lo, n_sb, half_it, quad_it, &mut q3a, &mut q4a, &mut q6a,
                &mut s8a, &mut d8a, &mut q3b, &mut q4b, &mut q6b, &mut s8b, &mut d8b,
            );
        }
    }

    /// [`flash_merge_q8`] with the partial fold run a second time — the
    /// merge pass's probe entry, not a path the step ships. `shift` enters
    /// the second fold that many segments along and `jig` is the weight its
    /// result folds in with; the host passes [`PROBE_JIG`], so this writes
    /// [`flash_merge_q8`]'s `y` and side output to the bit.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys_buf.len() >= 1,
            part_v.len() >= q_rows * segs * latent,
            part_ms.len() >= q_rows * segs * 2,
            y.len() >= q_rows * latent,
            q3a.len() >= m_lo * 64 * half_it,
            q4a.len() >= m_lo * 256 * quad_it,
            q6a.len() >= m_lo * 128 * half_it,
            s8a.len() >= m_lo * 8 * n_sb,
            d8a.len() >= m_lo * 2 * n_sb,
            q3b.len() >= (q_rows - m_lo) * 64 * half_it,
            q4b.len() >= (q_rows - m_lo) * 256 * quad_it,
            q6b.len() >= (q_rows - m_lo) * 128 * half_it,
            s8b.len() >= (q_rows - m_lo) * 8 * n_sb,
            d8b.len() >= (q_rows - m_lo) * 2 * n_sb
        )
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_merge2_q8(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        q_rows: u32,
        latent: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        part_v: &[f32],
        part_ms: &[f32],
        mut y: DisjointSlice<f32>,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
        shift: u32,
        jig: f32,
    ) {
        static mut VROW: SharedArray<f32, LATENT> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let lat = latent as usize;
        // SAFETY: as in `flash_merge_q8`.
        let v = unsafe {
            let v = merge_row::<true>(
                n_keys_buf,
                m,
                n_heads,
                dst_rows,
                segs,
                seg_keys,
                part_v,
                part_ms,
                row,
                tid,
                lat,
                shift as usize,
                jig,
            );
            *y.get_unchecked_mut(row * lat + tid) = v;
            v
        };
        // SAFETY: as in `flash_merge_q8`.
        let vs = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VROW) };
        // SAFETY: as in `flash_merge_q8`.
        unsafe {
            quant_row(
                v, vs, tid, lat, row, m_lo, n_sb, half_it, quad_it, &mut q3a, &mut q4a, &mut q6a,
                &mut s8a, &mut d8a, &mut q3b, &mut q4b, &mut q6b, &mut s8b, &mut d8b,
            );
        }
    }

    /// One query row's fold over its segments' partials, the standard
    /// online-softmax rescale in ascending segment order — the shared body
    /// of [`flash_merge`] and [`flash_merge_q8`], so the two cannot drift.
    /// Returns this thread's final latent value.
    ///
    /// `TWICE` is false in every shipped entry. The probe twin sets it and
    /// the fold runs a second time over the same partials, entered `shift`
    /// segments along and wrapped, its result folded in by `jig` — zero from
    /// the host, so the returned value is the shipped one.
    ///
    /// SAFETY: `row < q_rows`, `tid < latent`, and the partial buffers hold
    /// `q_rows * segs * latent` / `q_rows * segs * 2` elements.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn merge_row<const TWICE: bool>(
        n_keys_buf: &[u32],
        m: u32,
        n_heads: u32,
        dst_rows: u32,
        segs: u32,
        seg_keys: u32,
        part_v: &[f32],
        part_ms: &[f32],
        row: usize,
        tid: usize,
        lat: usize,
        shift: usize,
        jig: f32,
    ) -> f32 {
        let limit = causal_limit(n_keys_buf, dst_rows, row, n_heads, m);
        let n_seg = limit.div_ceil(seg_keys as usize).min(segs as usize);
        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut acc = 0.0f32;
        let mut seg = 0usize;
        while seg < n_seg {
            let idx = row * segs as usize + seg;
            // SAFETY: idx < q_rows * segs, so both slots are inside part_ms
            // (caller's contract).
            let (mj, sj) = unsafe {
                (
                    *part_ms.get_unchecked(2 * idx),
                    *part_ms.get_unchecked(2 * idx + 1),
                )
            };
            if sj != 0.0 {
                // SAFETY: sj != 0 => this segment wrote its partials, and
                // idx * lat + tid < q_rows * segs * latent.
                let vj = unsafe { *part_v.get_unchecked(idx * lat + tid) };
                if mj > mx {
                    // A first bump scales by 0.0 — the partials are still
                    // zero, so the reset and the scale are the same value.
                    let f = if mx > f32::NEG_INFINITY {
                        dev_exp(mx - mj)
                    } else {
                        0.0
                    };
                    s_sum = f32::mul_add(s_sum, f, sj);
                    acc = f32::mul_add(acc, f, vj);
                    mx = mj;
                } else {
                    let g = dev_exp(mj - mx);
                    s_sum = f32::mul_add(sj, g, s_sum);
                    acc = f32::mul_add(vj, g, acc);
                }
            }
            seg += 1;
        }
        if TWICE {
            // The same fold over the same partials, entered `shift`
            // segments along so no pass can rewrite its loads as the ones
            // above.
            let mut mx2 = f32::NEG_INFINITY;
            let mut s2 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut at = shift % n_seg.max(1);
            let mut left = n_seg;
            while left > 0 {
                let idx = row * segs as usize + at;
                // SAFETY: idx < q_rows * segs, so both slots are inside
                // part_ms (caller's contract).
                let (mj, sj) = unsafe {
                    (
                        *part_ms.get_unchecked(2 * idx),
                        *part_ms.get_unchecked(2 * idx + 1),
                    )
                };
                if sj != 0.0 {
                    // SAFETY: as in the fold above.
                    let vj = unsafe { *part_v.get_unchecked(idx * lat + tid) };
                    if mj > mx2 {
                        let f = if mx2 > f32::NEG_INFINITY {
                            dev_exp(mx2 - mj)
                        } else {
                            0.0
                        };
                        s2 = f32::mul_add(s2, f, sj);
                        acc2 = f32::mul_add(acc2, f, vj);
                        mx2 = mj;
                    } else {
                        let g = dev_exp(mj - mx2);
                        s2 = f32::mul_add(sj, g, s2);
                        acc2 = f32::mul_add(vj, g, acc2);
                    }
                }
                at += 1;
                if at == n_seg {
                    at = 0;
                }
                left -= 1;
            }
            acc = f32::mul_add(jig, acc2 + s2, acc);
        }
        let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
        s_inv * acc
    }

    /// The q8_1 side output of a block that holds one whole column: thread
    /// `tid` contributes its value `v`, the block transposes through shared
    /// memory, and the first `lat / 128` warps each quantize one 128-value
    /// block through `q8_1_quant_vals` — the one owner of that body.
    ///
    /// The transpose is what makes the fold bit-identical rather than merely
    /// close: the quantizer's warp sees the same 128 values in the same
    /// lanes as it would reading the stored column back, so its `amax` is
    /// the same reduction over the same set.
    ///
    /// SAFETY: `vs` is the calling block's own `LATENT`-element shared
    /// array, `tid < lat <= LATENT`, `lat` is a multiple of 128, `col` is a
    /// column of the half `row` selects, and the launch contract bounds both
    /// output sets.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn quant_row(
        v: f32,
        vs: *mut f32,
        tid: usize,
        lat: usize,
        row: usize,
        m_lo: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        q3a: &mut DisjointSlice<u64>,
        q4a: &mut DisjointSlice<u32>,
        q6a: &mut DisjointSlice<u32>,
        s8a: &mut DisjointSlice<i32>,
        d8a: &mut DisjointSlice<f32>,
        q3b: &mut DisjointSlice<u64>,
        q4b: &mut DisjointSlice<u32>,
        q6b: &mut DisjointSlice<u32>,
        s8b: &mut DisjointSlice<i32>,
        d8b: &mut DisjointSlice<f32>,
    ) {
        // SAFETY: tid < lat <= LATENT, the shared array's length.
        unsafe {
            *vs.add(tid) = v;
        }
        thread::sync_threads();
        let blk = tid / 32; // warp index, and the 128-value block it takes
        if blk >= lat / 128 {
            return; // warp-uniform: the collectives below stay full-warp
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: 128*blk + 4*lane + 3 < 128*(blk+1) <= lat <= LATENT.
        let vals = unsafe {
            let base = vs.add(128 * blk + 4 * lane);
            [*base, *base.add(1), *base.add(2), *base.add(3)]
        };
        let n_sb = n_sb as usize;
        // SAFETY (both arms): the values are block `blk`'s, held four per
        // lane in value order, which is `q8_1_quant_vals`'s precondition;
        // the column is inside the half's `m` and the launch contract bounds
        // that half's five outputs. The branch is on the block index.
        unsafe {
            if row < m_lo as usize {
                q8_1_quant_vals(
                    vals, row, blk, n_sb, half_it, quad_it, lane, q3a, q4a, q6a, s8a, d8a,
                );
            } else {
                q8_1_quant_vals(
                    vals,
                    row - m_lo as usize,
                    blk,
                    n_sb,
                    half_it,
                    quad_it,
                    lane,
                    q3b,
                    q4b,
                    q6b,
                    s8b,
                    d8b,
                );
            }
        }
    }

    /// The causal key limit of query row `row`: the live key count clamped to
    /// the cache's allocated rows (a lying `n_keys_buf` then reads at most
    /// what is allocated), plus the row's own offset inside the `m`-token
    /// batch. Query `t` of `m` attends keys `0..n_keys − m + t + 1`.
    #[inline(always)]
    fn causal_limit(n_keys_buf: &[u32], dst_rows: u32, row: usize, n_heads: u32, m: u32) -> usize {
        // SAFETY: n_keys_buf.len() >= 1 by every caller's launch contract.
        let n_keys = (unsafe { *n_keys_buf.get_unchecked(0) } as usize).min(dst_rows as usize);
        let t = row / n_heads as usize;
        (n_keys + t + 1).saturating_sub(m as usize)
    }

    /// One block's walk of keys `lo..hi` of query row `row`, the module doc's
    /// reduction contract: the query row is staged into `qs`, then the online
    /// softmax runs over [`KEY_TILE`]-key tiles. Returns this thread's
    /// `(running max, Σ exp, Σ exp·V)` — the first two are the block's shared
    /// state and are valid in warp 0 (thread 0 among them), the third is this
    /// thread's own latent dim, un-normalised and relative to that max.
    ///
    /// `lo` is a multiple of [`KEY_TILE`] and `hi <= limit <= dst_rows`, so
    /// only the final tile can reach past `hi` and it takes the guarded path
    /// whose `wl == 0.0` test skips those loads.
    ///
    /// `TWICE` is [`TWICE_NONE`] in every shipped entry and each probe
    /// branch below folds away with it. A probe entry sets one bit and that
    /// stage runs a second time, its result folded in by
    /// `fma(jig, second, first)` — the host passes `jig = 0.0`, so the
    /// returned triple is the shipped one. The two loops' second pass reads
    /// row `k + shift` for key `k`, wrapped inside `lo..hi`: `shift = 0`
    /// re-reads the rows this tile just read and a larger one reads rows it
    /// did not, never a row outside the walk.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn latent_range<const TWICE: u32>(
        q: &[f32],
        kv: &[u16],
        qs: *mut f32,
        klog: *mut f32,
        kw: *mut f32,
        st: *mut f32,
        row: usize,
        tid: usize,
        lo: usize,
        hi: usize,
        rope: usize,
        lat: usize,
        scale: f32,
        shift: usize,
        jig: f32,
    ) -> (f32, f32, f32) {
        let lane = warp::lane_id() as usize;
        let width = rope + lat;

        let mut s = tid;
        while s < width {
            // SAFETY: row is inside q's rows and s < width bound the load by
            // the caller's launch contract; s < width <= MAX_WIDTH
            // (host-validated) bounds the shared store.
            unsafe {
                *qs.add(s) = *q.get_unchecked(row * width + s);
            }
            s += LATENT;
        }

        // This thread's key inside a tile, and its slice of that key's dims.
        let key = tid / DIM_SPLIT;
        let dim0 = tid % DIM_SPLIT;
        // Probe only: the second passes' row offset, reduced into the walk
        // so a wrapped row stays inside `lo..hi`. A probe entry always walks
        // a live segment, so `hi > lo` wherever this is reached.
        let step = if TWICE & (TWICE_QK | TWICE_V) != 0 {
            shift % (hi - lo)
        } else {
            0
        };

        let mut mx = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        let mut r0 = 0.0f32;
        let mut r1 = 0.0f32;
        let mut r2 = 0.0f32;
        let mut r3 = 0.0f32;

        thread::sync_threads(); // the staged query row is visible

        let mut blk = lo;
        while blk < hi {
            // ---- QK dot: DIM_SPLIT threads per key, four rotating partials
            let mut acc = 0.0f32;
            if blk + key < hi {
                let kb = (blk + key) * width;
                let mut a0 = 0.0f32;
                let mut a1 = 0.0f32;
                let mut a2 = 0.0f32;
                let mut a3 = 0.0f32;
                let mut i = dim0;
                while i + (ILP - 1) * DIM_SPLIT < width {
                    // SAFETY: blk + key < hi <= dst_rows and every dim index
                    // below is < width by this loop's test, so each load is
                    // inside the key's row of kv (launch contract); the
                    // shared reads are below width <= MAX_WIDTH.
                    unsafe {
                        let k0 = half_bits_to_f32(*kv.get_unchecked(kb + i));
                        let k1 = half_bits_to_f32(*kv.get_unchecked(kb + i + DIM_SPLIT));
                        let k2 = half_bits_to_f32(*kv.get_unchecked(kb + i + 2 * DIM_SPLIT));
                        let k3 = half_bits_to_f32(*kv.get_unchecked(kb + i + 3 * DIM_SPLIT));
                        a0 = f32::mul_add(*qs.add(i), k0, a0);
                        a1 = f32::mul_add(*qs.add(i + DIM_SPLIT), k1, a1);
                        a2 = f32::mul_add(*qs.add(i + 2 * DIM_SPLIT), k2, a2);
                        a3 = f32::mul_add(*qs.add(i + 3 * DIM_SPLIT), k3, a3);
                    }
                    i += ILP * DIM_SPLIT;
                }
                while i < width {
                    // SAFETY: as above, for the trailing steps.
                    unsafe {
                        a0 = f32::mul_add(
                            *qs.add(i),
                            half_bits_to_f32(*kv.get_unchecked(kb + i)),
                            a0,
                        );
                    }
                    i += DIM_SPLIT;
                }
                acc = (a0 + a1) + (a2 + a3);
                if TWICE & TWICE_QK != 0 {
                    // The same walk over this key's partner row, `step`
                    // along and wrapped inside the walk.
                    let mut p = blk + key + step;
                    if p >= hi {
                        p -= hi - lo;
                    }
                    let pb = p * width;
                    let mut b0 = 0.0f32;
                    let mut b1 = 0.0f32;
                    let mut b2 = 0.0f32;
                    let mut b3 = 0.0f32;
                    let mut i = dim0;
                    while i + (ILP - 1) * DIM_SPLIT < width {
                        // SAFETY: lo <= p < hi <= dst_rows and every dim
                        // index is < width, so each load is inside that row
                        // of kv, exactly as in the pass above.
                        unsafe {
                            let k0 = half_bits_to_f32(*kv.get_unchecked(pb + i));
                            let k1 = half_bits_to_f32(*kv.get_unchecked(pb + i + DIM_SPLIT));
                            let k2 = half_bits_to_f32(*kv.get_unchecked(pb + i + 2 * DIM_SPLIT));
                            let k3 = half_bits_to_f32(*kv.get_unchecked(pb + i + 3 * DIM_SPLIT));
                            b0 = f32::mul_add(*qs.add(i), k0, b0);
                            b1 = f32::mul_add(*qs.add(i + DIM_SPLIT), k1, b1);
                            b2 = f32::mul_add(*qs.add(i + 2 * DIM_SPLIT), k2, b2);
                            b3 = f32::mul_add(*qs.add(i + 3 * DIM_SPLIT), k3, b3);
                        }
                        i += ILP * DIM_SPLIT;
                    }
                    while i < width {
                        // SAFETY: as above, for the trailing steps.
                        unsafe {
                            b0 = f32::mul_add(
                                *qs.add(i),
                                half_bits_to_f32(*kv.get_unchecked(pb + i)),
                                b0,
                            );
                        }
                        i += DIM_SPLIT;
                    }
                    acc = f32::mul_add(jig, (b0 + b1) + (b2 + b3), acc);
                }
            }
            // The key's DIM_SPLIT threads are one aligned lane group, so the
            // four-step butterfly stays inside the key. Every thread calls
            // it: the guard above shapes the value, never the control flow.
            acc += warp::shuffle_xor_f32(acc, 1);
            acc += warp::shuffle_xor_f32(acc, 2);
            acc += warp::shuffle_xor_f32(acc, 4);
            acc += warp::shuffle_xor_f32(acc, 8);
            if TWICE & TWICE_COLL != 0 {
                // The same butterfly again. Its input carries `jig` so no
                // pass can rewrite it as the one above.
                let mut c = acc + jig;
                c += warp::shuffle_xor_f32(c, 1);
                c += warp::shuffle_xor_f32(c, 2);
                c += warp::shuffle_xor_f32(c, 4);
                c += warp::shuffle_xor_f32(c, 8);
                acc = f32::mul_add(jig, c, acc);
            }
            if dim0 == 0 {
                let sv = if blk + key < hi {
                    scale * acc
                } else {
                    f32::NEG_INFINITY
                };
                // SAFETY: key < KEY_TILE, and one thread per key writes it.
                unsafe {
                    *klog.add(key) = sv;
                }
            }
            thread::sync_threads();
            if TWICE & TWICE_SYNC != 0 {
                thread::sync_threads();
            }

            // ---- online softmax: warp 0 owns (m, s) and publishes the tile
            if tid < KEY_TILE {
                // SAFETY: lane == tid < KEY_TILE here.
                let sv = unsafe { *klog.add(lane) };
                let mut smax = warp::reduce_max_f32(sv);
                if TWICE & TWICE_COLL != 0 {
                    smax = f32::mul_add(jig, warp::reduce_max_f32(sv + jig), smax);
                }
                // FlashMS update: s is rescaled here, the V partials just
                // before this tile's accumulation (the CPU oracle's order).
                // A first bump scales by 0.0 — the partials are still zero,
                // so the reset and the scale are the same value.
                let mut vms = 1.0f32;
                if smax > mx {
                    vms = if mx > f32::NEG_INFINITY {
                        dev_exp(mx - smax)
                    } else {
                        0.0
                    };
                    if TWICE & TWICE_SM != 0 {
                        let again = if mx > f32::NEG_INFINITY {
                            dev_exp((mx + jig) - smax)
                        } else {
                            0.0
                        };
                        vms = f32::mul_add(jig, again, vms);
                    }
                    s_sum *= vms;
                    mx = smax;
                }
                let mut w = if sv == f32::NEG_INFINITY {
                    0.0
                } else {
                    dev_exp(sv - mx)
                };
                if TWICE & TWICE_SM != 0 {
                    let sv2 = sv + jig;
                    let again = if sv2 == f32::NEG_INFINITY {
                        0.0
                    } else {
                        dev_exp(sv2 - mx)
                    };
                    w = f32::mul_add(jig, again, w);
                }
                s_sum += warp::reduce_sum_f32(w);
                if TWICE & TWICE_COLL != 0 {
                    s_sum = f32::mul_add(jig, warp::reduce_sum_f32(w + jig), s_sum);
                }
                // SAFETY: lane < KEY_TILE; one lane writes each slot, and
                // lane 0 alone writes the rescale.
                unsafe {
                    *kw.add(lane) = w;
                    if lane == 0 {
                        *st.add(0) = vms;
                    }
                }
            }
            thread::sync_threads();
            if TWICE & TWICE_SYNC != 0 {
                thread::sync_threads();
            }

            // ---- V accumulation: this thread's own latent dim, four keys
            // in flight. SAFETY: ST[0] and KW hold this tile's published
            // values (both barriers above).
            let vms = unsafe { *st.add(0) };
            r0 *= vms;
            r1 *= vms;
            r2 *= vms;
            r3 *= vms;
            let tail = rope + tid;
            if blk + KEY_TILE <= hi {
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: the whole tile is below hi <= dst_rows, so all
                    // ILP rows' tails are inside kv (launch contract); tail <
                    // width and l + ILP <= KEY_TILE.
                    unsafe {
                        let base = (blk + l) * width + tail;
                        let v0 = half_bits_to_f32(*kv.get_unchecked(base));
                        let v1 = half_bits_to_f32(*kv.get_unchecked(base + width));
                        let v2 = half_bits_to_f32(*kv.get_unchecked(base + 2 * width));
                        let v3 = half_bits_to_f32(*kv.get_unchecked(base + 3 * width));
                        r0 = f32::mul_add(*kw.add(l), v0, r0);
                        r1 = f32::mul_add(*kw.add(l + 1), v1, r1);
                        r2 = f32::mul_add(*kw.add(l + 2), v2, r2);
                        r3 = f32::mul_add(*kw.add(l + 3), v3, r3);
                    }
                    l += ILP;
                }
                if TWICE & TWICE_V != 0 {
                    // The same walk over a partner tile: the one `step`
                    // along while it is whole, else the walk's first tile,
                    // which is whole because this one is. Four partials of
                    // its own, as above, and one fold each at the end.
                    let nb = blk + step;
                    let pb = if nb + KEY_TILE <= hi { nb } else { lo };
                    let mut u0 = 0.0f32;
                    let mut u1 = 0.0f32;
                    let mut u2 = 0.0f32;
                    let mut u3 = 0.0f32;
                    let mut l = 0usize;
                    while l < KEY_TILE {
                        // SAFETY: lo <= pb and pb + KEY_TILE <= hi <=
                        // dst_rows, so all ILP rows' tails are inside kv;
                        // tail < width and l + ILP <= KEY_TILE.
                        unsafe {
                            let base = (pb + l) * width + tail;
                            let v0 = half_bits_to_f32(*kv.get_unchecked(base));
                            let v1 = half_bits_to_f32(*kv.get_unchecked(base + width));
                            let v2 = half_bits_to_f32(*kv.get_unchecked(base + 2 * width));
                            let v3 = half_bits_to_f32(*kv.get_unchecked(base + 3 * width));
                            u0 = f32::mul_add(*kw.add(l), v0, u0);
                            u1 = f32::mul_add(*kw.add(l + 1), v1, u1);
                            u2 = f32::mul_add(*kw.add(l + 2), v2, u2);
                            u3 = f32::mul_add(*kw.add(l + 3), v3, u3);
                        }
                        l += ILP;
                    }
                    r0 = f32::mul_add(jig, u0, r0);
                    r1 = f32::mul_add(jig, u1, r1);
                    r2 = f32::mul_add(jig, u2, r2);
                    r3 = f32::mul_add(jig, u3, r3);
                }
            } else {
                // The run's last tile is the only one that reaches past the
                // limit: a zero weight is exactly a row this must not load.
                let mut l = 0usize;
                while l < KEY_TILE {
                    // SAFETY: l < KEY_TILE.
                    let wl = unsafe { *kw.add(l) };
                    if wl != 0.0 {
                        // SAFETY: wl != 0 => key blk+l is live => blk+l < hi
                        // <= dst_rows, so its tail is inside kv.
                        unsafe {
                            r0 = f32::mul_add(
                                wl,
                                half_bits_to_f32(*kv.get_unchecked((blk + l) * width + tail)),
                                r0,
                            );
                        }
                    }
                    l += 1;
                }
                if TWICE & TWICE_V != 0 {
                    // The guarded walk again, each live key's partner row
                    // `step` along and wrapped inside the walk.
                    let mut u0 = 0.0f32;
                    let mut l = 0usize;
                    while l < KEY_TILE {
                        // SAFETY: l < KEY_TILE.
                        let wl = unsafe { *kw.add(l) };
                        if wl != 0.0 {
                            let mut p = blk + l + step;
                            if p >= hi {
                                p -= hi - lo;
                            }
                            // SAFETY: lo <= p < hi <= dst_rows, so that
                            // row's tail is inside kv.
                            unsafe {
                                u0 = f32::mul_add(
                                    wl,
                                    half_bits_to_f32(*kv.get_unchecked(p * width + tail)),
                                    u0,
                                );
                            }
                        }
                        l += 1;
                    }
                    r0 = f32::mul_add(jig, u0, r0);
                }
            }
            blk += KEY_TILE;
        }

        (mx, s_sum, (r0 + r1) + (r2 + r3))
    }
}

/// The loaded P5 device module: `kv_append`, `kv_append_pos_buf`,
/// `flash_latent`, `flash_latent_seg`, `flash_merge`, their q8 twins and the
/// stage-doubling probe entries. Owns no context and no
/// stream — every enqueue takes the
/// engine stream (`Gpu::stream()`), so launches order with the rest of the
/// step and are capturable.
pub struct FlashKernels {
    module: flash_kernels::LoadedModule,
}

impl FlashKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FlashKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { flash_kernels::load(ctx)? };
        Ok(FlashKernels { module })
    }

    /// Enqueue the f32 -> f16 append of `m` new rows of `cache.cols()` f32
    /// (`src.len() >= m * width`, token-major) at rows `pos..pos + m` of the
    /// `[cache.rows() x width]` u16 cache. `pos` is a launch scalar and is
    /// frozen inside a captured graph — for the captured decode step use
    /// [`FlashKernels::enqueue_kv_append_pos_buf`]. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_kv_append(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        check_append("enqueue_kv_append", src.len(), width, m)?;
        if pos as usize + m > rows {
            return Err(format!(
                "enqueue_kv_append: rows {}..{} land past the cache's {rows} rows",
                pos,
                pos as usize + m
            )
            .into());
        }
        let prep = self.module.prepare_kv_append(LaunchConfig1D::new(
            (m * width).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module.kv_append(
            stream,
            &prep,
            m as u32,
            width as u32,
            pos,
            src,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_kv_append`] with `pos` read from
    /// `pos_buf[0]` on the device at run time — the captured-graph form: the
    /// graph replays unchanged while the position advances through the
    /// buffer. Rows past the cache's height are skipped in-kernel. `pos_buf`
    /// must stay allocated and in place for as long as a captured graph
    /// replaying this launch lives. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_kv_append_pos_buf(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        pos_buf: &DeviceBuffer<u32>,
        cache: &mut DeviceTensor<u16>,
        m: usize,
    ) -> Result<(), GpuError> {
        let (rows, width) = (cache.rows(), cache.cols());
        if pos_buf.len() < 1 {
            return Err("enqueue_kv_append_pos_buf: pos_buf must hold 1 u32".into());
        }
        check_append("enqueue_kv_append_pos_buf", src.len(), width, m)?;
        let prep = self.module.prepare_kv_append_pos_buf(LaunchConfig1D::new(
            (m * width).div_ceil(256) as u32,
            256,
            0,
        ))?;
        self.module.kv_append_pos_buf(
            stream,
            &prep,
            m as u32,
            width as u32,
            rows as u32,
            pos_buf,
            src,
            cache.buf_mut(),
        )?;
        Ok(())
    }

    /// Enqueue the latent flash attention: `q` holds `m * n_heads` rows of
    /// `rope_dims + latent` f32 (row `t * n_heads + h`, content
    /// `[q_rope | q_nope2]`), `kv` is the `[rows x width]` u16 cache, and
    /// `n_keys_buf[0]` names the live key count (`>= m`: the batch's own
    /// rows are appended first) — query `t` attends to keys
    /// `0..n_keys − m + t`. `y` holds `m * n_heads` rows of `latent` f32,
    /// the `kqv_compressed` order. `scale` is `MlaParams::kq_scale` (the
    /// YaRN mscale is inside it — not `1/√d`). `latent` must be [`LATENT`],
    /// this family's block width; `rope_dims` is free as long as
    /// `rope_dims + latent` is a multiple of [`DIM_SPLIT`] and at most
    /// [`MAX_WIDTH`] (the shared staging row). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_flash_latent(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let q_rows = check_flash(
            "enqueue_flash_latent",
            q.len(),
            kv,
            n_keys_buf.len(),
            m,
            n_heads,
            rope_dims,
            latent,
            Some(y.len()),
        )?;
        let prep = self.module.prepare_flash_latent(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_latent(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            rope_dims as u32,
            latent as u32,
            kv.rows() as u32,
            y,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_latent`] with the key range spread over
    /// [`segments_for`]`(kv.rows())` blocks per query row: the segment pass
    /// writes partials into `part_v`/`part_ms` and the merge pass folds them
    /// into `y`. Returns the number of launches enqueued — one when the cache
    /// is short enough to hold a single segment (the single-block kernel
    /// serves it, no partials touched), two otherwise. That choice comes from
    /// the cache height alone, so it is fixed for the life of a captured
    /// graph, and the grid does not move with `n_keys_buf`.
    ///
    /// `part_v` must hold [`partials_v_len`] and `part_ms`
    /// [`partials_ms_len`] for the same `m * n_heads` rows and cache height.
    /// Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_latent_split(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        part_v: &mut DeviceBuffer<f32>,
        part_ms: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<usize, GpuError> {
        if segments_for(kv.rows()) == 1 {
            self.enqueue_flash_latent(
                stream, q, kv, n_keys_buf, scale, m, n_heads, rope_dims, latent, y,
            )?;
            return Ok(1);
        }
        self.enqueue_flash_latent_seg(
            stream, q, kv, n_keys_buf, scale, m, n_heads, rope_dims, latent, part_v, part_ms,
        )?;
        self.enqueue_flash_merge(
            stream,
            n_keys_buf,
            kv.rows(),
            m,
            n_heads,
            latent,
            part_v,
            part_ms,
            y,
        )?;
        Ok(2)
    }

    /// The segment pass alone — the first of the two launches
    /// [`FlashKernels::enqueue_flash_latent_split`] makes. A caller that
    /// wants the two launches timed or observed separately enqueues this and
    /// [`FlashKernels::enqueue_flash_merge`] itself, after asking
    /// [`segments_for`] whether the cache is tall enough to need them at all.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_latent_seg(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        part_v: &mut DeviceBuffer<f32>,
        part_ms: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let segs = segments_for(kv.rows());
        let q_rows = check_flash(
            "enqueue_flash_latent_seg",
            q.len(),
            kv,
            n_keys_buf.len(),
            m,
            n_heads,
            rope_dims,
            latent,
            None,
        )?;
        check_partials(
            "enqueue_flash_latent_seg",
            part_v.len(),
            part_ms.len(),
            q_rows,
            segs,
            latent,
        )?;
        let prep = self.module.prepare_flash_latent_seg(LaunchConfig1D::new(
            (q_rows * segs) as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_latent_seg(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            rope_dims as u32,
            latent as u32,
            kv.rows() as u32,
            segs as u32,
            seg_keys() as u32,
            part_v,
            part_ms,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_latent_seg`] through the probe entry
    /// that runs one stage of the walk a second time: `twice` is one of
    /// [`TWICE_QK`], [`TWICE_V`], [`TWICE_COLL`], [`TWICE_SYNC`] and
    /// [`TWICE_SM`], and `shift` is the second pass's row offset inside the
    /// segment — 0 re-reads the rows just read, [`KEY_TILE`] reads the next
    /// tile's — which only the two loop stages look at. A TIMING
    /// INSTRUMENT: the partials are the shipped entry's to the bit, so an
    /// arm's slowdown against it is that stage's price in the step.
    /// Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_latent_seg_twice(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        part_v: &mut DeviceBuffer<f32>,
        part_ms: &mut DeviceBuffer<f32>,
        twice: u32,
        shift: usize,
    ) -> Result<(), GpuError> {
        let segs = segments_for(kv.rows());
        let q_rows = check_flash(
            "enqueue_flash_latent_seg_twice",
            q.len(),
            kv,
            n_keys_buf.len(),
            m,
            n_heads,
            rope_dims,
            latent,
            None,
        )?;
        check_partials(
            "enqueue_flash_latent_seg_twice",
            part_v.len(),
            part_ms.len(),
            q_rows,
            segs,
            latent,
        )?;
        if segs == 1 {
            return Err(
                "enqueue_flash_latent_seg_twice: the probe entries are the split \
                        launch's, and a one-segment cache runs the single-block kernel"
                    .into(),
            );
        }
        let cfg = LaunchConfig1D::new((q_rows * segs) as u32, LATENT as u32, 0);
        macro_rules! probe {
            ($prepare:ident, $entry:ident) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
                    stream,
                    &prep,
                    q,
                    kv.buf(),
                    n_keys_buf,
                    scale,
                    m as u32,
                    n_heads as u32,
                    q_rows as u32,
                    rope_dims as u32,
                    latent as u32,
                    kv.rows() as u32,
                    segs as u32,
                    seg_keys() as u32,
                    part_v,
                    part_ms,
                    shift as u32,
                    PROBE_JIG,
                )?;
            }};
        }
        match twice {
            TWICE_QK => probe!(prepare_flash_latent_seg_qk2, flash_latent_seg_qk2),
            TWICE_V => probe!(prepare_flash_latent_seg_v2, flash_latent_seg_v2),
            TWICE_COLL => probe!(prepare_flash_latent_seg_coll2, flash_latent_seg_coll2),
            TWICE_SYNC => probe!(prepare_flash_latent_seg_sync2, flash_latent_seg_sync2),
            TWICE_SM => probe!(prepare_flash_latent_seg_sm2, flash_latent_seg_sm2),
            other => {
                return Err(format!(
                    "enqueue_flash_latent_seg_twice: {other} names no single probe stage"
                )
                .into());
            }
        }
        Ok(())
    }

    /// The merge pass alone — the second of the two launches. `cache_rows`
    /// is the height the partials were written for, so the segment count
    /// matches the segment pass's grid exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_merge(
        &self,
        stream: &CudaStream,
        n_keys_buf: &DeviceBuffer<u32>,
        cache_rows: usize,
        m: usize,
        n_heads: usize,
        latent: usize,
        part_v: &DeviceBuffer<f32>,
        part_ms: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let segs = segments_for(cache_rows);
        let q_rows = m * n_heads;
        if n_keys_buf.is_empty() {
            return Err("enqueue_flash_merge: n_keys_buf must hold 1 u32".into());
        }
        if latent != LATENT {
            return Err(format!(
                "enqueue_flash_merge: this family's latent tail is {LATENT}, got {latent}"
            )
            .into());
        }
        check_partials(
            "enqueue_flash_merge",
            part_v.len(),
            part_ms.len(),
            q_rows,
            segs,
            latent,
        )?;
        if y.len() < q_rows * latent {
            return Err(format!(
                "enqueue_flash_merge: y.len() {} < m*n_heads*latent = {}",
                y.len(),
                q_rows * latent
            )
            .into());
        }
        let prep = self.module.prepare_flash_merge(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_merge(
            stream,
            &prep,
            n_keys_buf,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            latent as u32,
            cache_rows as u32,
            segs as u32,
            seg_keys() as u32,
            part_v,
            part_ms,
            y,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_merge`] that also writes the q8_1 form
    /// of the merged rows into `lo` and `hi`: the rows below `lo.m()` fill
    /// `lo`'s columns in order, the rest `hi`'s. The bytes are the ones
    /// `Gpu::enqueue_quantize_q8_1_pair(y, 0, lo, lo.m() * latent, hi)`
    /// would have written, so this deletes that launch rather than merging
    /// it. Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_merge_q8(
        &self,
        stream: &CudaStream,
        n_keys_buf: &DeviceBuffer<u32>,
        cache_rows: usize,
        m: usize,
        n_heads: usize,
        latent: usize,
        part_v: &DeviceBuffer<f32>,
        part_ms: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        lo: &mut Q8Act,
        hi: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let segs = segments_for(cache_rows);
        let q_rows = m * n_heads;
        if n_keys_buf.is_empty() {
            return Err("enqueue_flash_merge_q8: n_keys_buf must hold 1 u32".into());
        }
        if latent != LATENT {
            return Err(format!(
                "enqueue_flash_merge_q8: this family's latent tail is {LATENT}, got {latent}"
            )
            .into());
        }
        check_partials(
            "enqueue_flash_merge_q8",
            part_v.len(),
            part_ms.len(),
            q_rows,
            segs,
            latent,
        )?;
        if y.len() < q_rows * latent {
            return Err(format!(
                "enqueue_flash_merge_q8: y.len() {} < m*n_heads*latent = {}",
                y.len(),
                q_rows * latent
            )
            .into());
        }
        let (m_lo, n_sb) = check_side_quant("enqueue_flash_merge_q8", lo, hi, q_rows, latent)?;
        let prep = self.module.prepare_flash_merge_q8(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_merge_q8(
            stream,
            &prep,
            n_keys_buf,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            latent as u32,
            cache_rows as u32,
            segs as u32,
            seg_keys() as u32,
            m_lo as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            part_v,
            part_ms,
            y,
            &mut lo.q3,
            &mut lo.q4,
            &mut lo.q6,
            &mut lo.s8,
            &mut lo.d8,
            &mut hi.q3,
            &mut hi.q4,
            &mut hi.q6,
            &mut hi.s8,
            &mut hi.d8,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_merge_q8`] through the probe entry that
    /// folds the partials a second time, `shift` segments along. A TIMING
    /// INSTRUMENT: `y` and the side output are the shipped entry's to the
    /// bit, so the arm's slowdown against it is the fold's price in the
    /// step. Asynchronous, allocation-free, capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_merge2_q8(
        &self,
        stream: &CudaStream,
        n_keys_buf: &DeviceBuffer<u32>,
        cache_rows: usize,
        m: usize,
        n_heads: usize,
        latent: usize,
        part_v: &DeviceBuffer<f32>,
        part_ms: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        lo: &mut Q8Act,
        hi: &mut Q8Act,
        shift: usize,
    ) -> Result<(), GpuError> {
        let segs = segments_for(cache_rows);
        let q_rows = m * n_heads;
        if n_keys_buf.is_empty() {
            return Err("enqueue_flash_merge2_q8: n_keys_buf must hold 1 u32".into());
        }
        if latent != LATENT {
            return Err(format!(
                "enqueue_flash_merge2_q8: this family's latent tail is {LATENT}, got {latent}"
            )
            .into());
        }
        check_partials(
            "enqueue_flash_merge2_q8",
            part_v.len(),
            part_ms.len(),
            q_rows,
            segs,
            latent,
        )?;
        if y.len() < q_rows * latent {
            return Err(format!(
                "enqueue_flash_merge2_q8: y.len() {} < m*n_heads*latent = {}",
                y.len(),
                q_rows * latent
            )
            .into());
        }
        let (m_lo, n_sb) = check_side_quant("enqueue_flash_merge2_q8", lo, hi, q_rows, latent)?;
        let prep = self.module.prepare_flash_merge2_q8(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_merge2_q8(
            stream,
            &prep,
            n_keys_buf,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            latent as u32,
            cache_rows as u32,
            segs as u32,
            seg_keys() as u32,
            m_lo as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            part_v,
            part_ms,
            y,
            &mut lo.q3,
            &mut lo.q4,
            &mut lo.q6,
            &mut lo.s8,
            &mut lo.d8,
            &mut hi.q3,
            &mut hi.q4,
            &mut hi.q6,
            &mut hi.s8,
            &mut hi.d8,
            shift as u32,
            PROBE_JIG,
        )?;
        Ok(())
    }

    /// [`FlashKernels::enqueue_flash_latent`] with the same q8_1 side output
    /// as [`FlashKernels::enqueue_flash_merge_q8`] — the single-segment path,
    /// so both paths of the folded step quantize.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_flash_latent_q8(
        &self,
        stream: &CudaStream,
        q: &DeviceBuffer<f32>,
        kv: &DeviceTensor<u16>,
        n_keys_buf: &DeviceBuffer<u32>,
        scale: f32,
        m: usize,
        n_heads: usize,
        rope_dims: usize,
        latent: usize,
        y: &mut DeviceBuffer<f32>,
        lo: &mut Q8Act,
        hi: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let q_rows = check_flash(
            "enqueue_flash_latent_q8",
            q.len(),
            kv,
            n_keys_buf.len(),
            m,
            n_heads,
            rope_dims,
            latent,
            Some(y.len()),
        )?;
        let (m_lo, n_sb) = check_side_quant("enqueue_flash_latent_q8", lo, hi, q_rows, latent)?;
        let prep = self.module.prepare_flash_latent_q8(LaunchConfig1D::new(
            q_rows as u32,
            LATENT as u32,
            0,
        ))?;
        self.module.flash_latent_q8(
            stream,
            &prep,
            q,
            kv.buf(),
            n_keys_buf,
            scale,
            m as u32,
            n_heads as u32,
            q_rows as u32,
            rope_dims as u32,
            latent as u32,
            kv.rows() as u32,
            m_lo as u32,
            n_sb as u32,
            n_sb.div_ceil(2) as u32,
            n_sb.div_ceil(4) as u32,
            y,
            &mut lo.q3,
            &mut lo.q4,
            &mut lo.q6,
            &mut lo.s8,
            &mut lo.d8,
            &mut hi.q3,
            &mut hi.q4,
            &mut hi.q6,
            &mut hi.s8,
            &mut hi.d8,
        )?;
        Ok(())
    }
}

/// The side output's shape check, shared by the two q8 twins: the two
/// scratches must together cover the launch's query rows one column each,
/// at the latent width, and `lat / 128` must be the block count per column
/// a 512-thread block can quantize in one pass. Returns `(lo.m(), n_sb)`.
fn check_side_quant(
    what: &str,
    lo: &Q8Act,
    hi: &Q8Act,
    q_rows: usize,
    latent: usize,
) -> Result<(usize, usize), GpuError> {
    if lo.k() != latent || hi.k() != latent {
        return Err(format!(
            "{what}: the side scratches' k must be the latent width {latent}, got {} and {}",
            lo.k(),
            hi.k()
        )
        .into());
    }
    if lo.m() + hi.m() != q_rows {
        return Err(format!(
            "{what}: the side scratches hold {} + {} columns, the launch has {q_rows} query rows",
            lo.m(),
            hi.m()
        )
        .into());
    }
    Ok((lo.m(), lo.n_sb()))
}

/// The partials both split launches share.
fn check_partials(
    what: &str,
    v_len: usize,
    ms_len: usize,
    q_rows: usize,
    segs: usize,
    latent: usize,
) -> Result<(), GpuError> {
    let (want_v, want_ms) = (q_rows * segs * latent, q_rows * segs * 2);
    if v_len < want_v || ms_len < want_ms {
        return Err(format!(
            "{what}: partials are {v_len}/{ms_len} f32, want {want_v}/{want_ms} for \
             {q_rows} rows x {segs} segments"
        )
        .into());
    }
    Ok(())
}

/// The geometry both flash entries share: the block width is the latent
/// tail, the query row is staged in shared memory, and the QK dot splits a
/// row across [`DIM_SPLIT`] threads; `y_len` is `None` for the segment pass,
/// which writes partials instead. Returns `m * n_heads`, the query rows.
#[allow(clippy::too_many_arguments)]
fn check_flash(
    what: &str,
    q_len: usize,
    kv: &DeviceTensor<u16>,
    n_keys_len: usize,
    m: usize,
    n_heads: usize,
    rope_dims: usize,
    latent: usize,
    y_len: Option<usize>,
) -> Result<usize, GpuError> {
    if !(1..=8).contains(&m) {
        return Err(format!("{what}: 1 <= m <= 8, got {m}").into());
    }
    if n_heads == 0 {
        return Err(format!("{what}: n_heads >= 1").into());
    }
    if latent != LATENT {
        return Err(format!(
            "{what}: this family's latent tail is {LATENT} (one block thread per dim), \
             got {latent}"
        )
        .into());
    }
    let width = rope_dims + latent;
    if !width.is_multiple_of(DIM_SPLIT) {
        return Err(format!(
            "{what}: the QK dot splits a row across {DIM_SPLIT} threads, need rope_dims + \
             latent = {width} a multiple of {DIM_SPLIT}"
        )
        .into());
    }
    if width > MAX_WIDTH {
        return Err(format!(
            "{what}: the query row is staged in {MAX_WIDTH} shared f32, got rope_dims + \
             latent = {width}"
        )
        .into());
    }
    if kv.cols() != width {
        return Err(format!(
            "{what}: kv is {}-wide, want rope_dims + latent = {width}",
            kv.cols()
        )
        .into());
    }
    if n_keys_len < 1 {
        return Err(format!("{what}: n_keys_buf must hold 1 u32").into());
    }
    let q_rows = m * n_heads;
    if q_len < q_rows * width {
        return Err(format!(
            "{what}: q.len() {q_len} < m*n_heads*width = {}",
            q_rows * width
        )
        .into());
    }
    // The segment pass has no `y` of its own — it writes partials.
    if let Some(y_len) = y_len
        && y_len < q_rows * latent
    {
        return Err(format!(
            "{what}: y.len() {y_len} < m*n_heads*latent = {}",
            q_rows * latent
        )
        .into());
    }
    Ok(q_rows)
}

/// Reject geometry the append kernels' launch contracts do not cover:
/// positive width, `m` rows, and `src` holding all of them. The landing-row
/// bound is checked only where `pos` is a host scalar.
fn check_append(what: &str, src_len: usize, width: usize, m: usize) -> Result<(), GpuError> {
    if width == 0 || m == 0 {
        return Err(format!("{what}: need width >= 1 and m >= 1, got {width}/{m}").into());
    }
    if src_len < m * width {
        return Err(format!("{what}: src.len() {src_len} < m*width = {}", m * width).into());
    }
    Ok(())
}
