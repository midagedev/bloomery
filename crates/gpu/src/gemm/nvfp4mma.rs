//! Native block-scaled tensor-core products (sm_120a, sm_121a): FP4 over
//! routed NVFP4 planes, and the dense MXFP8 W8A8 product over the
//! activations' MXFP8 form on the shared MMA tile (`f32tile.rs`).

use super::act32::ColsForm;
use super::f32tile::{
    Bf16Cols, Bf16Rows, ByteRows, HcUpMixArgs, MMA_THREADS, MmaCols, MmaRows, OutForm,
    STAGE_FLOATS, STAGES as MMA_STAGES, TileLaunch, TileRows, bf16_mma_body,
    bf16_mma_norm_gate_body, form_col, mma_fits, refuse_or, store_refused,
};
use super::{GemmAct32, GemmRoute};
use crate::GpuError;
use crate::linear::norm_gate::NormGateTail;
use crate::q8f32::{E4m3Weight, Mxfp8Bytes};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::F32x4;
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, ptx_asm, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

const TILE_ROWS: usize = 128;
const TILE_THREADS: u32 = 256;
/// Rows of the gate and of the up one `gate_up_nvfp4_mma` block computes:
/// half of [`TILE_ROWS`] each, two 32-value blocks of the SwiGLU's column.
const PAIR_ROWS: usize = TILE_ROWS / 2;
/// The SwiGLU tile's column pitch in shared memory: [`PAIR_ROWS`] values and
/// four of padding, so the epilogue's stores and the quantizer's aligned
/// four-value reads meet no bank twice.
const PAIR_PITCH: usize = PAIR_ROWS + 4;

pub(super) struct Act {
    q: DeviceBuffer<u32>,
    e: DeviceBuffer<u32>,
    scale: DeviceBuffer<f32>,
}

impl Act {
    pub(super) fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        Ok(Self {
            q: DeviceBuffer::zeroed(stream, cols * k / 8)?,
            e: DeviceBuffer::zeroed(stream, cols * k / 64)?,
            scale: DeviceBuffer::zeroed(stream, cols)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.q.num_bytes() + self.e.num_bytes() + self.scale.num_bytes()
    }
}

#[cuda_device::device]
// PIN(2026-10-24): indexed FP4 packing loops preserve cuda-oxide unrolling and avoid a local depot.
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
fn nvfp4_pack_q8(words: [u32; 4], d: f32, gs: f32) -> ([u32; 2], u32) {
    let mut values = [0.0f32; 16];
    #[unroll]
    for i in 0..16 {
        values[i] = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * d;
    }
    let mut largest = 0.0f32;
    #[unroll]
    for i in 0..16 {
        largest = largest.max(values[i].abs());
    }
    let mut scale = if gs.is_finite() {
        let x = largest / (6.0 * gs);
        u32::from(cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(x, x)) & 255
    } else {
        0
    };
    let decoded = crate::mxfp4::ue4m3_half(scale) * 2.0 * gs;
    let mut inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
    if !largest.is_finite()
        || !(6.0 * gs).is_finite()
        || decoded < f32::MIN_POSITIVE
        || !inv.is_finite()
    {
        let nonzero = words[0] | words[1] | words[2] | words[3];
        let ratio = if gs > 0.0 && nonzero != 0 {
            d / gs
        } else {
            0.0
        };
        largest = 0.0;
        #[unroll]
        for i in 0..16 {
            values[i] = ((words[i / 4] >> (8 * (i % 4))) as i8) as f32 * ratio;
            largest = largest.max(values[i].abs());
        }
        let x = largest / 6.0;
        scale = if gs.is_finite() {
            u32::from(cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32(x, x)) & 255
        } else {
            0
        };
        let decoded = crate::mxfp4::ue4m3_half(scale) * 2.0;
        inv = if decoded > 0.0 { 1.0 / decoded } else { 0.0 };
    }
    let mut packed = [0u32; 2];
    #[unroll]
    for i in 0..8 {
        let byte: u32;
        // SAFETY: this conversion only reads and writes registers.
        unsafe {
            ptx_asm!(
                "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; cvt.u32.u8 %0, b; }",
                out("=r") byte, in("f") values[2 * i] * inv, in("f") values[2 * i + 1] * inv,
            );
        }
        packed[i / 4] |= byte << (8 * (i & 3));
    }
    (packed, scale)
}

// Bound each block's cache to 32 KiB. Wider rows recompute the uncached quads.
const ACT_CACHE_VALUES: usize = 8192;
const _: () = assert!((ACT_CACHE_VALUES + 2) * 4 == 32776);

/// # Safety
/// Both inputs contain the aligned four-value span at `base`.
#[inline(always)]
unsafe fn act_quad(g: &[f32], u: &[f32], base: usize, code: u32, limit: f32) -> [f32; 4] {
    // SAFETY: the caller supplies an aligned, in-bounds gate quad.
    let g = unsafe { (*g.as_ptr().add(base).cast::<cuda_device::vector::F32x4>()).0 };
    // SAFETY: the caller supplies an aligned, in-bounds up quad.
    let u = unsafe { (*u.as_ptr().add(base).cast::<cuda_device::vector::F32x4>()).0 };
    if crate::fault::quad_finite(g) & crate::fault::quad_finite(u) {
        [
            crate::kquant::act::apply(code, limit, g[0], u[0]),
            crate::kquant::act::apply(code, limit, g[1], u[1]),
            crate::kquant::act::apply(code, limit, g[2], u[2]),
            crate::kquant::act::apply(code, limit, g[3], u[3]),
        ]
    } else {
        [f32::NAN; 4]
    }
}

// PIN(2026-10-24): native FP4 generated device wrappers require flat CUDA ABI arguments (R8 kernel exemption).
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed the SwiGLU kernel entries' flat arguments, and the device macro's same-name wrapper repeats them (rust-quality R8)"
)]
mod swiglu {
    use super::*;

    #[cuda_device::device]
    pub(super) fn fused_swiglu(
        g: &[f32],
        u: &[f32],
        code: u32,
        limit: f32,
        k: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut sums: DisjointSlice<i32>,
        mut out: DisjointSlice<u32>,
        mut scales: DisjointSlice<u32>,
        mut global: DisjointSlice<f32>,
        fault: crate::fault::FaultSink,
    ) {
        let col = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let k = k as usize;
        let cached = k.min(ACT_CACHE_VALUES);
        let cache = cuda_device::DynamicSharedArray::<f32>::get();
        let mut largest = 0.0f32;
        let mut at = 4 * tid;
        while at < k {
            // SAFETY: K divides 64; the loop's aligned quad is inside both checked input columns.
            let v = unsafe { act_quad(g, u, col * k + at, code, limit) };
            if at < cached {
                // SAFETY: this thread owns the aligned quad inside the allocated shared prefix.
                unsafe {
                    *cache.add(at).cast::<cuda_device::vector::F32x4>() =
                        cuda_device::vector::F32x4::new(v);
                }
            }
            largest = if crate::fault::quad_finite(v) {
                largest
                    .max(v[0].abs())
                    .max(v[1].abs())
                    .max(v[2].abs())
                    .max(v[3].abs())
            } else {
                f32::INFINITY
            };
            at += 256;
        }
        let mut delta = 16;
        while delta > 0 {
            largest = largest.max(warp::shuffle_xor_f32(largest, delta));
            delta /= 2;
        }
        if lane == 0 {
            // SAFETY: each warp owns one of the two shared reduction words after the cached prefix.
            unsafe {
                *cache.add(cached + tid / 32) = largest;
            }
        }
        thread::sync_threads();
        // SAFETY: the first warp published its maximum in the first reduction word.
        let first = unsafe { *cache.add(cached) };
        // SAFETY: the second warp published its maximum in the second reduction word.
        let second = unsafe { *cache.add(cached + 1) };
        let largest = first.max(second);
        let gs = if !largest.is_finite() {
            f32::NAN
        } else if largest > 0.0 {
            (largest / 127.0) * (127.0 / 6.0)
        } else {
            1.0
        };
        if tid == 0 {
            // SAFETY: this block owns the live column's global scale.
            unsafe {
                *global.get_unchecked_mut(col) = gs;
            }
        }
        let blocks = k / 32;
        let steps = k / 64;
        let mut group = tid / 32;
        while group < blocks.div_ceil(4) {
            let block = (4 * group + (lane >> 3)).min(blocks - 1);
            let at = 32 * block + 4 * (lane & 7);
            let v = if at < cached {
                // SAFETY: this aligned quad is inside the published shared prefix.
                unsafe { (*cache.add(at).cast::<cuda_device::vector::F32x4>()).0 }
            } else {
                // SAFETY: this aligned quad is inside both checked input columns.
                unsafe { act_quad(g, u, col * k + at, code, limit) }
            };
            // SAFETY: each warp owns one group of four 32-value blocks in this checked column.
            let refused = unsafe {
                crate::gemm::act32::quant32_group(
                    v, col, group, blocks, steps, lane, &mut q, &mut d, &mut sums,
                )
            };
            if refused && lane & 7 == 0 {
                fault.raise(crate::FaultSite::QuantColumn);
            }
            group += 2;
        }
        thread::sync_threads();
        let mut block = tid;
        while block - lane < k / 16 {
            let source = block.min(k / 16 - 1);
            let at = col * steps * 16 + source * 4;
            // SAFETY: this aligned four-word block is in the column's published Q8 plane.
            let words = unsafe { (*q.as_mut_ptr().add(at).cast::<cuda_device::vector::U32x4>()).0 };
            // SAFETY: the block barrier published this checked column's Q8 scales.
            let dx = unsafe { *d.as_mut_ptr().add(col * 2 * steps + source / 2) };
            let (packed, scale) = nvfp4_pack_q8(words, dx, gs);
            if block < k / 16 {
                // SAFETY: this lane owns the first word of the live 16-value sub-block.
                unsafe {
                    *out.get_unchecked_mut(col * k / 8 + 2 * block) = packed[0];
                }
                // SAFETY: this lane owns the second word of the live 16-value sub-block.
                unsafe {
                    *out.get_unchecked_mut(col * k / 8 + 2 * block + 1) = packed[1];
                }
            }
            let mut word = scale << (8 * (lane & 3));
            word |= warp::shuffle_xor(word, 1);
            word |= warp::shuffle_xor(word, 2);
            if block < k / 16 && lane & 3 == 0 {
                // SAFETY: one lane per quad owns these four packed scales.
                unsafe {
                    *scales.get_unchecked_mut(col * steps + block / 4) = word;
                }
            }
            block += 64;
        }
    }
}
use swiglu::fused_swiglu;

#[inline(always)]
fn packed_pair(a: u32, b: u32) -> [u32; 2] {
    let (lo, hi): (u32, u32);
    // SAFETY: shifts, nibble selection and byte permutation use registers only.
    unsafe {
        ptx_asm!(
            "{ .reg .b32 a4, a8, b4, b8, al, ah, bl, bh;\n\
             shr.u32 a4, %2, 4; shr.u32 a8, %2, 8;\n\
             shr.u32 b4, %3, 4; shr.u32 b8, %3, 8;\n\
             lop3.b32 al, %2, a4, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 ah, a4, a8, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 bl, %3, b4, 0x0f0f0f0f, 0xe4;\n\
             lop3.b32 bh, b4, b8, 0x0f0f0f0f, 0xe4;\n\
             prmt.b32 %0, al, bl, 0x6420;\n\
             prmt.b32 %1, ah, bh, 0x6420; }",
            out("=r") lo, out("=r") hi, in("r") a, in("r") b,
        );
    }
    [lo, hi]
}

#[inline(always)]
fn mma(c: [f32; 4], a: [u32; 4], b: [u32; 2], sa: u32, sb: u32) -> [f32; 4] {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    // SAFETY: the caller executes this register-only instruction in every lane of its warp.
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,0}, %15, {0,0};",
            out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1],
            in("f") c[0], in("f") c[1], in("f") c[2], in("f") c[3],
            in("r") sa, in("r") sb,
        );
    }
    [d0, d1, d2, d3]
}

/// 64-value halves of one K-walk step: each takes one MMA per fragment, so a
/// step's barrier and copies serve two.
const HALVES: usize = 2;
/// u32 words of one K-walk stage row: each half's eight code words.
const ROW_W: usize = HALVES * 8;
/// u32 words of one stage's [`TILE_ROWS`] weight rows.
const W_STAGE: usize = TILE_ROWS * ROW_W;
/// u32 words of one stage's 64 activation rows, the weight rows' pitch.
const B_STAGE: usize = 64 * ROW_W;
/// u32 words of one half's scale words: the weight rows', then the
/// activation rows' in [`col_scale`] order.
const S_HALF: usize = TILE_ROWS + 64;
const S_STAGE: usize = HALVES * S_HALF;
/// The K walk's shared stages: a step's planes land two steps ahead.
const STAGES: usize = 3;

/// Where activation column `c < 64` keeps its scale word among a half's
/// activation scale words: column `8n + g`, fragment `n`'s column of lane
/// group `g`, at `8g + n`, so a lane's scales of fragments `4q .. 4q + 4`
/// are one aligned quad ([`scale_quad`]).
#[inline(always)]
const fn col_scale(c: usize) -> usize {
    8 * (c % 8) + c / 8
}
const _: () = {
    let mut seen = 0u64;
    let mut c = 0;
    while c < 64 {
        seen |= 1 << col_scale(c);
        c += 1;
    }
    assert!(seen == u64::MAX && TILE_ROWS.is_multiple_of(4) && S_HALF.is_multiple_of(4));
};

/// Lane group `g`'s activation scales of fragments `4q .. 4q + 4`, from a
/// half's scale words `sp`.
///
/// # Safety
/// `sp` is a half's [`S_HALF`] scale words in shared memory, 16-byte
/// aligned, after the barrier that publishes them; `g < 8`, `q < 2`.
#[inline(always)]
unsafe fn scale_quad(sp: *mut u32, g: usize, q: usize) -> [u32; 4] {
    // SAFETY: TILE_ROWS + 8g + 4q + 4 <= S_HALF for g < 8, q < 2, a multiple
    // of four words past the aligned start (this fn's contract).
    unsafe {
        (*sp.add(TILE_ROWS + 8 * g + 4 * q)
            .cast::<cuda_device::vector::U32x4>())
        .0
    }
}

/// u32 words of a route tile's [`TileColumns`]: each of its 64 columns' slot,
/// then its input scale's bits.
const COLUMN_WORDS: usize = 2 * 64;
const _: () = assert!(
    (STAGES * (W_STAGE + B_STAGE + S_STAGE) + COLUMN_WORDS) * 4 <= 48 * 1024
        && 64 * PAIR_PITCH <= STAGES * W_STAGE
);

/// A route tile's column slots and input scales in the block's
/// [`COLUMN_WORDS`] shared words. The thread that copies column `sr < 64`
/// (half 0 of shared row `sr`) writes its two before the K walk, whose
/// first barrier publishes them, so the epilogue reads them without waiting
/// on a global load; a dead column holds its tile's last live column's.
#[derive(Clone, Copy)]
struct TileColumns(*mut u32);

impl TileColumns {
    /// Column `sr`'s slot `slot` and input scale `ag[col]`, by half 0 of
    /// shared row `sr < 64`.
    ///
    /// # Safety
    /// `self` is the block's [`COLUMN_WORDS`] shared words, which no thread
    /// reads before the walk's first barrier.
    #[inline(always)]
    unsafe fn put(self, (sr, half): (usize, usize), slot: usize, ag: &[f32], col: usize) {
        if sr < 64 && half == 0 {
            let scale = ag[col].to_bits();
            // SAFETY: sr < 64, this thread's own slot word.
            unsafe { *self.0.add(sr) = slot as u32 };
            // SAFETY: 64 + sr < COLUMN_WORDS, this thread's own scale word.
            unsafe { *self.0.add(64 + sr) = scale };
        }
    }

    /// Column `c`'s slot and input scale.
    ///
    /// # Safety
    /// `c < 64`, after the walk's first barrier.
    #[inline(always)]
    unsafe fn get(self, c: usize) -> (usize, f32) {
        // SAFETY: c < 64, a slot word the barrier published.
        let slot = unsafe { *self.0.add(c) } as usize;
        // SAFETY: 64 + c < COLUMN_WORDS, its scale word.
        let scale = f32::from_bits(unsafe { *self.0.add(64 + c) });
        (slot, scale)
    }
}

/// The u32 word where 16-byte chunk `q` of a stage row starts, half `q / 2`'s
/// chunk `q % 2`. The chunk number is XORed with bit 1 of the row index (onto
/// its bit 1) and bit 2 (onto its bit 0), so neither the eight rows one
/// `ldmatrix` phase reads at one chunk nor the four rows' two chunks of one
/// half that eight threads copy and pack at a time meet a bank twice.
#[inline(always)]
const fn chunk(row: usize, q: usize) -> usize {
    row * ROW_W + 4 * (q ^ ((row & 2) | ((row >> 2) & 1)))
}
// Every row's chunks are aligned, inside the row and distinct; the eight rows
// one `ldmatrix` phase reads at one chunk, and the four rows' two chunks of
// one half that eight threads (`sr = tid / 2`, `half = tid & 1`) copy and pack
// at a time, cover the 32 banks once.
const _: () = {
    use cuda_device::swizzle::{WARP_LANES, conflict_degree};
    let mut row = 0;
    while row < TILE_ROWS {
        let mut seen = 0usize;
        let mut q = 0;
        while q < 2 * HALVES {
            let at = chunk(row, q);
            assert!(at.is_multiple_of(4) && row * ROW_W <= at && at + 4 <= (row + 1) * ROW_W);
            seen |= 1 << ((at - row * ROW_W) / 4);
            let mut ldmatrix = [0; WARP_LANES];
            let mut copy = [0; WARP_LANES];
            let mut i = 0;
            while i < WARP_LANES {
                ldmatrix[i] = chunk(row / 8 * 8 + i / 4, q) + i % 4;
                copy[i] = chunk(row / 4 * 4 + i / 8, q / 2 * 2 + i / 4 % 2) + i % 4;
                i += 1;
            }
            assert!(conflict_degree(&ldmatrix, 4) == 1 && conflict_degree(&copy, 4) == 1);
            q += 1;
        }
        assert!(seen == (1 << (2 * HALVES)) - 1);
        row += 1;
    }
};

/// One thread's `cp.async` copies of step `$s` (64-value steps `HALVES * $s`
/// on, below `$steps`) into stage `$st` (`$s % STAGES`) of `nvfp4_walk!`'s
/// shared arrays `$wp`, `$bp`, `$sp` (its arguments): of each half, half
/// `$half` of weight row `$wrow`'s codes and, for `$sr < $bc`, of activation
/// column `$col`'s, into shared row `$sr`; and half `$half`'s scale words,
/// the activation's for `$sr < 64`. One group, empty for a step past the
/// last. `$full` (a literal) says every step
/// below `$steps` has both halves, so only the step itself is checked.
macro_rules! nvfp4_copy {
    (
        $full:literal,
        ($s:expr, $st:expr),
        ($wq:ident, $we:ident, $wrow:ident),
        ($aq:ident, $ae:ident, $col:ident),
        ($steps:ident, $sr:ident, $half:ident, $bc:expr),
        ($wp:ident, $bp:ident, $sp:ident)
    ) => {{
        let s: usize = $s;
        let st: usize = $st;
        if HALVES * s < $steps {
            for h in 0..HALVES {
                ::cuda_device::thread::__unroll_config::<0>();
                let k = HALVES * s + h;
                if $full || k < $steps {
                    let at = chunk($sr, 2 * h + $half);
                    // SAFETY: the weight row's aligned half of 64-value step k is in the checked plane
                    // (K divides 64); sr < TILE_ROWS, h < HALVES and half < 2 put at, this thread's
                    // own aligned four-word chunk, inside row sr of stage st of W, read by no other
                    // thread before the barrier after the wait that completes it.
                    unsafe {
                        cuda_device::async_copy::cp_async_cg_16(
                            $wp.add(st * W_STAGE + at),
                            $wq.as_ptr().add(($wrow * $steps + k) * 8 + 4 * $half),
                        )
                    };
                    if $sr < $bc {
                        // SAFETY: col is a live column whose aligned half of step k is in the plane;
                        // sr < bc <= 64, h < HALVES and half < 2 put at, this thread's own aligned four-word
                        // chunk, inside row sr of stage st of B, read by no other thread before the
                        // barrier after the wait that completes it.
                        unsafe {
                            cuda_device::async_copy::cp_async_cg_16(
                                $bp.add(st * B_STAGE + at),
                                $aq.as_ptr().add(($col * $steps + k) * 8 + 4 * $half),
                            )
                        };
                    }
                }
            }
            let k = HALVES * s + $half;
            if $full || k < $steps {
                // SAFETY: the weight row's scale word of step k is in the plane; this thread owns
                // that half's word of the row in stage st.
                unsafe {
                    cuda_device::async_copy::cp_async_ca_4(
                        $sp.add(st * S_STAGE + $half * S_HALF + $sr),
                        $we.as_ptr().add($wrow * $steps + k),
                    )
                };
                if $sr < 64 {
                    // SAFETY: the live column's scale word of step k is in the plane; this thread
                    // owns that half's word of the column in stage st.
                    unsafe {
                        cuda_device::async_copy::cp_async_ca_4(
                            $sp.add(st * S_STAGE + $half * S_HALF + TILE_ROWS + col_scale($sr)),
                            $ae.as_ptr().add($col * $steps + k),
                        )
                    };
                }
            }
        }
        // SAFETY: commits only this thread's copies of the step as one group.
        unsafe { cuda_device::async_copy::cp_async_commit_group() };
    }};
}

/// The K walk both native kernels share, over [`STAGES`] shared stages of
/// [`TILE_ROWS`] weight rows and 64 activation rows, [`HALVES`] 64-value
/// halves a step: each thread copies half `$half` of shared row `$sr` from
/// weight row `$wrow` (`$wq` codes, `$we` scales) and, for `$sr < 64`, from
/// activation column `$col` (`$aq`, `$ae`) two steps ahead
/// ([`nvfp4_copy`]), the activation codes of its first `$bc <= 64` rows
/// alone, and before a step's barrier turns its own weight halves of the
/// step's stage into the MMA's nibble order ([`packed_pair`]); then `$body`
/// reads each live half `$h` of the stage, in K order: its codes through
/// `$wp` and `$bp` at [`chunk`] `2 * $h` and `2 * $h + 1`, the activation's
/// in its first `$bc` rows, its scale words through `$sp`. Every thread of the block runs it, `$steps` 64-value
/// steps in all. `$full` (a literal) says `$steps` is even, so every step has
/// both halves and none of the walk's half checks remain.
macro_rules! nvfp4_walk {
    (
        $full:literal,
        ($wq:ident, $we:ident, $wrow:ident),
        ($aq:ident, $ae:ident, $col:ident),
        ($steps:ident, $sr:ident, $half:ident, $bc:expr),
        ($wp:ident, $bp:ident, $sp:ident, $h:ident),
        $body:block
    ) => {{
        nvfp4_copy!(
            $full,
            (0, 0),
            ($wq, $we, $wrow),
            ($aq, $ae, $col),
            ($steps, $sr, $half, $bc),
            ($wp, $bp, $sp)
        );
        nvfp4_copy!(
            $full,
            (1, 1),
            ($wq, $we, $wrow),
            ($aq, $ae, $col),
            ($steps, $sr, $half, $bc),
            ($wp, $bp, $sp)
        );
        // The stage this step reads, and the one its copies of step + 2 fill.
        let (mut st, mut fill) = (0, 2);
        let mut step = 0;
        while HALVES * step < $steps {
            let halves = if $full {
                HALVES
            } else {
                ($steps - HALVES * step).min(HALVES)
            };
            // SAFETY: each step commits one group, so the newest is step + 1's and waiting for all
            // but one completes this thread's copies of this step.
            unsafe { cuda_device::async_copy::cp_async_wait_group(1) };
            // SAFETY: st < STAGES, so stage st starts inside the STAGES·W_STAGE words of W.
            let ws = unsafe { $wp.add(st * W_STAGE) };
            // SAFETY: st < STAGES, so stage st starts inside the STAGES·B_STAGE words of B.
            let bs = unsafe { $bp.add(st * B_STAGE) };
            // SAFETY: st < STAGES, so stage st starts inside the STAGES·S_STAGE words of S.
            let ss = unsafe { $sp.add(st * S_STAGE) };
            for h in 0..HALVES {
                ::cuda_device::thread::__unroll_config::<0>();
                if h < halves {
                    let own = chunk($sr, 2 * h + $half);
                    // SAFETY: sr < TILE_ROWS and 2 * h + half < 2 * HALVES, so own is an aligned
                    // four-word chunk inside row sr of stage st of W; the wait above completed
                    // this thread's copy of it.
                    let w = unsafe { (*ws.add(own).cast::<cuda_device::vector::U32x4>()).0 };
                    let p0 = packed_pair(w[0], w[1]);
                    let p1 = packed_pair(w[2], w[3]);
                    // SAFETY: the same chunk; chunk maps each (row, q) to its own words, so this
                    // thread is its one writer, and no thread reads it before the barrier below.
                    unsafe {
                        *ws.add(own).cast::<cuda_device::vector::U32x4>() =
                            cuda_device::vector::U32x4::new([p0[0], p0[1], p1[0], p1[1]]);
                    }
                }
            }
            // Publishes the stage and tells that every thread is done with step - 1's, which the
            // copies of step + 2 refill.
            thread::sync_threads();
            nvfp4_copy!(
                $full,
                (step + 2, fill),
                ($wq, $we, $wrow),
                ($aq, $ae, $col),
                ($steps, $sr, $half, $bc),
                ($wp, $bp, $sp)
            );
            let ($wp, $bp) = (ws, bs);
            for $h in 0..HALVES {
                ::cuda_device::thread::__unroll_config::<0>();
                if $h < halves {
                    // SAFETY: h < halves <= HALVES, so half h's S_HALF scale words lie inside the
                    // S_STAGE words of stage st of S.
                    let $sp = unsafe { ss.add($h * S_HALF) };
                    $body
                }
            }
            st = if st + 1 == STAGES { 0 } else { st + 1 };
            fill = if fill + 1 == STAGES { 0 } else { fill + 1 };
            step += 1;
        }
    }};
}

/// The activation rows a walk over a tile's first `live` fragments copies:
/// the fragment pairs that hold them, which `b_pair` loads.
#[inline(always)]
const fn live_rows(live: usize) -> usize {
    16 * live.div_ceil(2)
}

/// A local `$walk!(true, live)` over a route tile of `$len` columns: its first
/// `live` 8-column fragments, the fewest of 1, 2, 4, 6 or 8 that hold its
/// columns, a literal in each arm, so the walk keeps no fragment check.
macro_rules! live_walk {
    ($len:expr, $walk:ident) => {
        match $len {
            0..=8 => $walk!(true, 1),
            9..=16 => $walk!(true, 2),
            17..=32 => $walk!(true, 4),
            33..=48 => $walk!(true, 6),
            _ => $walk!(true, 8),
        }
    };
}

/// An MXFP8 weight's planes, staged as stored for the native W8A8 tile. Its
/// units are E4M3 value pairs, so a 16-byte piece copies sixteen values as a
/// BF16 weight's words are copied; the exact owner decodes them as
/// [`ByteRows`] does.
#[derive(Clone, Copy)]
struct Mxfp8Rows<'a> {
    qs: &'a [u32],
    e: &'a [u32],
}

impl TileRows for Mxfp8Rows<'_> {
    #[inline(always)]
    unsafe fn stage(self, dst: *mut f32, at: usize) {
        // SAFETY: this fn's contract is ByteRows::stage's over the same planes.
        unsafe { ByteRows::<Mxfp8Bytes>::new(self.qs, self.e).stage(dst, at) };
    }
}

/// The scale word a piece checks when it does not start its row's 64-unit
/// span: every byte in range.
const UE8M0_FITS: u32 = 0x7f7f_7f7f;

/// Whether every UE8M0 byte of `word` keeps each nonzero E4M3 value it
/// scales normal and at most `2^47` ([`mma_fits`]): 10 to 165.
#[inline(always)]
fn ue8m0_fits(word: u32) -> bool {
    (word & 255).wrapping_sub(10) <= 155
        && ((word >> 8) & 255).wrapping_sub(10) <= 155
        && ((word >> 16) & 255).wrapping_sub(10) <= 155
        && (word >> 24).wrapping_sub(10) <= 155
}

impl MmaRows for Mxfp8Rows<'_> {
    /// The scale word of the row's 64-unit span a piece starts, which
    /// [`Self::land`] checks; [`UE8M0_FITS`] for the other pieces.
    type Piece = u32;
    const SCALED: bool = true;

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) -> u32 {
        // SAFETY: units at .. at + 8 are words at / 2 .. at / 2 + 4 of the
        // value plane, laid out as a BF16 weight's (this fn's contract).
        unsafe { Bf16Rows(self.qs).fetch(dst, at) };
        if at.is_multiple_of(64) {
            // SAFETY: the span starts inside the weight's rows.
            unsafe { self.scale_word(at) }
        } else {
            UE8M0_FITS
        }
    }

    #[inline(always)]
    unsafe fn land(word: u32, _: *mut u32, _: usize) -> bool {
        ue8m0_fits(word)
    }

    #[inline(always)]
    unsafe fn landed_fits(_: *const u32) -> bool {
        true
    }

    #[inline(always)]
    unsafe fn scale_word(self, at: usize) -> u32 {
        // SAFETY: a row of K values holds K / 128 scale words, the 64-unit
        // span at / 64's among them (this fn's contract).
        unsafe { *self.e.get_unchecked(at / 64) }
    }

    #[inline(always)]
    unsafe fn mma(
        acc: [f32; 4],
        a: [u32; 4],
        b: [u32; 2],
        sa: u32,
        sb: u32,
        ta: usize,
        tb: usize,
    ) -> [f32; 4] {
        let (d0, d1, d2, d3): (f32, f32, f32, f32);
        // The selectors are immediates: one instruction a pair of quad lanes,
        // `ta` and `tb` constants where the tile unrolls its calls.
        macro_rules! selected {
            ($($sel:pat => $ta:literal $tb:literal),*) => {
                match (ta & 1, tb & 3) {
                    // SAFETY: every lane of the converged warp executes this
                    // register-only instruction (this fn's contract).
                    $($sel => unsafe {
                        ptx_asm!(
                            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0,%16}, %15, {0,%17};",
                            out("=f") d0, out("=f") d1, out("=f") d2, out("=f") d3,
                            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
                            in("r") b[0], in("r") b[1],
                            in("f") acc[0], in("f") acc[1], in("f") acc[2], in("f") acc[3],
                            in("r") sa, in("r") sb, in("C") $ta, in("C") $tb,
                        );
                    },)*
                }
            };
        }
        selected!(
            (0, 0) => b"0\0" b"0\0", (0, 1) => b"0\0" b"1\0", (0, 2) => b"0\0" b"2\0", (0, 3) => b"0\0" b"3\0",
            (1, 0) => b"1\0" b"0\0", (1, 1) => b"1\0" b"1\0", (1, 2) => b"1\0" b"2\0", _ => b"1\0" b"3\0"
        );
        [d0, d1, d2, d3]
    }
}

/// The activations' MXFP8 form (`mxfp8_cols_body`): the E4M3 value pairs
/// staged as [`Bf16Cols`]' words are, with its refusal words and the f32
/// columns the replay reads, and the UE8M0 scale words, `K / 128` a column.
#[derive(Clone, Copy)]
struct Mxfp8Cols<'a> {
    pairs: Bf16Cols<'a>,
    e: &'a [u32],
}

impl MmaCols for Mxfp8Cols<'_> {
    type Piece = ();
    const SCALED: bool = true;

    #[inline(always)]
    fn exact(&self) -> &[f32] {
        self.pairs.exact
    }

    #[inline(always)]
    unsafe fn refused(self, c: usize) -> bool {
        // SAFETY: this fn's contract.
        unsafe { self.pairs.refused(c) }
    }

    #[inline(always)]
    unsafe fn fetch(self, dst: *mut u32, at: usize) {
        // SAFETY: this fn's contract, over the pairs' words.
        unsafe { self.pairs.fetch(dst, at) };
    }

    #[inline(always)]
    unsafe fn land(_: (), _: *mut u32) -> bool {
        true
    }

    #[inline(always)]
    unsafe fn scale_word(self, at: usize) -> u32 {
        // SAFETY: a column of K values holds K / 128 scale words, the 64-unit
        // span at / 64's among them (this fn's contract).
        unsafe { *self.e.get_unchecked(at / 64) }
    }
}

/// A 32-value block's UE8M0 scale byte for its largest magnitude `amax`,
/// `ceil(log2(amax / 448))` biased and clamped to 1..=254, and the inverse
/// scale its values are multiplied by before E4M3 rounding.
#[inline(always)]
fn ue8m0_block(amax: f32) -> (u32, f32) {
    let bits = (amax * (1.0 / 448.0)).to_bits();
    let exp = ((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)).clamp(1, 254);
    (exp, f32::from_bits((254 - exp) << 23))
}

/// [`Mxfp8Cols`]' planes of an epilogue's output columns ([`OutForm`]), with
/// `mxfp8_cols_body`'s bits: each 32-value block's E4M3 words and its scale
/// byte, and each column's refusal word, zero before the launch and ORed by
/// its pieces.
#[derive(Clone, Copy)]
struct Mxfp8Out {
    qs: *mut u32,
    e: *mut u32,
    refused: *mut u32,
}

impl Mxfp8Out {
    /// Writes the E4M3 words of the 32-value block from value `at` (a
    /// multiple of 32) that lane `l`'s `v` is value `at + l` of, lane 4q's
    /// word values 4q .. 4q + 3 as `mxfp8_cols_body`'s lane q writes them;
    /// the block's UE8M0 exponent, the same in every lane.
    ///
    /// # Safety
    ///
    /// The whole warp calls converged, and the block's 8 words lie inside the
    /// form.
    #[inline(always)]
    unsafe fn block(self, v: f32, at: usize) -> u32 {
        use cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32;
        let lane = warp::lane_id() as usize;
        let mut amax = v.abs();
        amax = amax.max(warp::shuffle_xor_f32(amax, 1));
        amax = amax.max(warp::shuffle_xor_f32(amax, 2));
        amax = amax.max(warp::shuffle_xor_f32(amax, 4));
        amax = amax.max(warp::shuffle_xor_f32(amax, 8));
        amax = amax.max(warp::shuffle_xor_f32(amax, 16));
        let (exp, inv) = ue8m0_block(amax);
        let scaled = v * inv;
        let pair = u32::from(cvt_rn_satfinite_e4m3x2_f32(
            scaled,
            warp::shuffle_xor_f32(scaled, 1),
        ));
        let word = pair | (warp::shuffle_xor(pair, 2) << 16);
        if lane.is_multiple_of(4) {
            // SAFETY: word (at + lane) / 4 is this lane's own (this fn's
            // contract).
            unsafe { *self.qs.add((at + lane) / 4) = word };
        }
        exp
    }
}

impl OutForm for Mxfp8Out {
    const WRITES: bool = true;

    #[inline(always)]
    #[allow(clippy::needless_range_loop, reason = "indexed CUDA-unrolled arrays")]
    unsafe fn store<const N: usize>(self, v: [f32; N], c: usize, row0: usize, k: usize) {
        const { assert!(N == 1 || N == 4) };
        let at = c * k + row0;
        let mut fits = true;
        let mut scales = 0u32;
        for b in 0..N {
            thread::__unroll_config::<0>();
            fits &= mma_fits(v[b]);
            // SAFETY: block b lies inside column c's values (this fn's
            // contract).
            scales |= unsafe { self.block(v[b], at + 32 * b) } << (8 * b);
        }
        if warp::lane_id() == 0 {
            if N == 1 {
                // SAFETY: byte at / 32 is the block's byte of its scale word,
                // inside the form's scale words; lane 0 of this warp is its
                // only writer.
                unsafe { *self.e.cast::<u8>().add(at / 32) = scales as u8 };
            } else {
                // SAFETY: word at / 128 is the head's scale word inside the
                // form's scale words; lane 0 of this warp is its only writer.
                unsafe { *self.e.add(at / 128) = scales };
            }
        }
        // SAFETY: c is a live column the form holds (this fn's contract).
        unsafe { refuse_or(self.refused, c, fits) };
    }
}

/// The first `n` columns of `k` f32 values (`k` a multiple of 128) as
/// [`Mxfp8Cols`]' planes: each 32-value block's scale exponent
/// `ceil(log2(amax / 448))` as a UE8M0 byte, the block's values over that
/// scale as E4M3 rounded to nearest even and saturated, and each column's
/// refusal word ([`mma_fits`]). A warp walks its column 128 values at a
/// time, eight lanes a block.
///
/// # Safety
///
/// x holds n aligned K-value columns, qs n·K/4 words, e n·K/128 and
/// refused n.
#[inline(always)]
unsafe fn mxfp8_cols_body(
    x: &[f32],
    k: u32,
    n: u32,
    qs: &mut DisjointSlice<u32>,
    e: &mut DisjointSlice<u32>,
    refused: &mut DisjointSlice<u32>,
) {
    use cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32;
    let (k, n) = (k as usize, n as usize);
    let c = form_col();
    let lane = warp::lane_id() as usize;
    let mut fits = true;
    if c < n {
        let mut span = 0;
        while span < k / 128 {
            let at = c * k + 128 * span + 4 * lane;
            // SAFETY: values at .. at + 4 lie in column c of x, 16-byte
            // aligned (this fn's contract).
            let v = unsafe { (*x.as_ptr().add(at).cast::<F32x4>()).0 };
            fits &= mma_fits(v[0]) & mma_fits(v[1]) & mma_fits(v[2]) & mma_fits(v[3]);
            let mut amax = v[0].abs().max(v[1].abs()).max(v[2].abs()).max(v[3].abs());
            amax = amax.max(warp::shuffle_xor_f32(amax, 1));
            amax = amax.max(warp::shuffle_xor_f32(amax, 2));
            amax = amax.max(warp::shuffle_xor_f32(amax, 4));
            let (exp, inv) = ue8m0_block(amax);
            let lo = u32::from(cvt_rn_satfinite_e4m3x2_f32(v[0] * inv, v[1] * inv));
            let hi = u32::from(cvt_rn_satfinite_e4m3x2_f32(v[2] * inv, v[3] * inv));
            // SAFETY: word at / 4 is this lane's own.
            unsafe { *qs.get_unchecked_mut(at / 4) = lo | (hi << 16) };
            // Lanes 0, 8, 16 and 24 hold the span's four block scales.
            let word = warp::shuffle(exp, 0)
                | (warp::shuffle(exp, 8) << 8)
                | (warp::shuffle(exp, 16) << 16)
                | (warp::shuffle(exp, 24) << 24);
            if lane == 0 {
                // SAFETY: lane 0 of column c's warp is the span's only writer.
                unsafe { *e.get_unchecked_mut(at / 128) = word };
            }
            span += 1;
        }
    }
    // SAFETY: c is this warp's column (this fn's contract).
    unsafe { store_refused(refused, c, n, fits) };
}

#[cuda_module]
mod native {
    use super::*;
    use cuda_device::shared::cvta_generic_to_shared_u32;
    use cuda_device::wmma::ldmatrix_x4_shared_u32;

    /// Fragments `n` and `n + 1` of half `h` of an activation stage `bp`, one
    /// `ldmatrix` of four eight-row sets: lanes `8i .. 8i + 8` address
    /// fragment `n + i / 2`'s chunk `2h + i % 2`, the sets the MMA's two
    /// registers take.
    ///
    /// # Safety
    /// Every lane of the warp calls it; `bp` is an activation stage after the
    /// barrier that publishes it, `n + 1 < 8` and `h < HALVES`.
    #[inline(always)]
    unsafe fn b_pair(bp: *mut u32, n: usize, h: usize, lane: usize) -> [[u32; 2]; 2] {
        let at = chunk(
            (n + (lane >> 4)) * 8 + (lane & 7),
            2 * h + ((lane >> 3) & 1),
        );
        // SAFETY: rows (n + lane / 16) * 8 + lane % 8 < 64 of the stage, at an aligned chunk of
        // half h, each set eight aligned rows (this fn's contract).
        let r = unsafe {
            ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                bp.add(at).cast_const().cast::<u8>(),
            ))
        };
        [[r[0], r[1]], [r[2], r[3]]]
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared_range = (264, 32776))]
    pub fn swiglu_nvfp4_32(
        g: &[f32],
        u: &[f32],
        code: u32,
        limit: f32,
        k: u32,
        q: DisjointSlice<u32>,
        d: DisjointSlice<f32>,
        sums: DisjointSlice<i32>,
        out: DisjointSlice<u32>,
        scales: DisjointSlice<u32>,
        global: DisjointSlice<f32>,
        fault: crate::fault::FaultSink,
    ) {
        fused_swiglu(g, u, code, limit, k, q, d, sums, out, scales, global, fault);
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared_range = (264, 32776))]
    pub fn swiglu_nvfp4_32_sel(
        g: &[f32],
        u: &[f32],
        sel: &[u32],
        n_card: u32,
        code: u32,
        limit: f32,
        k: u32,
        q: DisjointSlice<u32>,
        d: DisjointSlice<f32>,
        sums: DisjointSlice<i32>,
        out: DisjointSlice<u32>,
        scales: DisjointSlice<u32>,
        global: DisjointSlice<f32>,
        fault: crate::fault::FaultSink,
    ) {
        let place = sel[thread::blockIdx_x() as usize];
        if place == crate::hybrid::HOST {
            return;
        }
        if place >= n_card {
            if thread::threadIdx_x() == 0 {
                fault.raise(crate::FaultSite::ExpertId);
            }
            return;
        }
        fused_swiglu(g, u, code, limit, k, q, d, sums, out, scales, global, fault);
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn quantize_nvfp4_32(
        q: &[u32],
        d: &[f32],
        steps: u32,
        mut out: DisjointSlice<u32>,
        mut scales: DisjointSlice<u32>,
        mut global: DisjointSlice<f32>,
    ) {
        let col = thread::blockIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let steps = steps as usize;
        let mut max = 0.0f32;
        let mut block = lane;
        while block < 2 * steps {
            let at = col * 16 * steps + 8 * block;
            let nonzero = q[at]
                | q[at + 1]
                | q[at + 2]
                | q[at + 3]
                | q[at + 4]
                | q[at + 5]
                | q[at + 6]
                | q[at + 7];
            // A refused block's NaN scale makes the column's global scale NaN,
            // so every output of the column reads NaN.
            let dx = d[col * 2 * steps + block];
            if !dx.is_finite() {
                max = f32::INFINITY;
            } else if nonzero != 0 {
                max = max.max(dx);
            }
            block += 32;
        }
        let mut delta = 16;
        while delta > 0 {
            max = max.max(warp::shuffle_xor_f32(max, delta));
            delta /= 2;
        }
        let gs = if !max.is_finite() {
            f32::NAN
        } else if max > 0.0 {
            max * (127.0 / 6.0)
        } else {
            1.0
        };
        if lane == 0 {
            // SAFETY: one block owns this column, below the allocated column count.
            unsafe {
                *global.get_unchecked_mut(col) = gs;
            }
        }
        block = lane;
        while block - lane < 4 * steps {
            let source = block.min(4 * steps - 1);
            let at = col * steps * 16 + source * 4;
            let dx = d[col * 2 * steps + source / 2];
            let words = [q[at], q[at + 1], q[at + 2], q[at + 3]];
            let (packed, scale) = nvfp4_pack_q8(words, dx, gs);
            if block < 4 * steps {
                // SAFETY: this lane owns the live sub-block's first word.
                unsafe {
                    *out.get_unchecked_mut(col * steps * 8 + block * 2) = packed[0];
                }
                // SAFETY: this lane owns the live sub-block's second word.
                unsafe {
                    *out.get_unchecked_mut(col * steps * 8 + block * 2 + 1) = packed[1];
                }
            }
            let mut word = scale << (8 * (lane & 3));
            word |= warp::shuffle_xor(word, 1);
            word |= warp::shuffle_xor(word, 2);
            if block < 4 * steps && lane & 3 == 0 {
                // SAFETY: one lane in each live quad owns its four packed scales.
                unsafe {
                    *scales.get_unchecked_mut(col * steps + block / 4) = word;
                }
            }
            block += 32;
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn gemm_nvfp4_mma(
        wq: &[u32],
        we: &[u32],
        ws: &[f32],
        aq: &[u32],
        ae: &[u32],
        ag: &[f32],
        cols: &[u32],
        tiles: &[u32],
        ntiles: &[u32],
        rows: u32,
        steps: u32,
        slots: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block / row_tiles as usize;
        let rt = block % row_tiles as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let rows = rows as usize;
        let steps = steps as usize;
        if tile == 0 {
            let refused = ntiles[1] as usize;
            let first = slots as usize - refused;
            let mut i = tid;
            while i < refused * TILE_ROWS {
                let row = rt * TILE_ROWS + i % TILE_ROWS;
                if row < rows {
                    // SAFETY: the route's refused slots are disjoint from live tiles; one thread owns this row.
                    unsafe {
                        *y.get_unchecked_mut(cols[first + i / TILE_ROWS] as usize * rows + row) =
                            f32::NAN;
                    }
                }
                i += TILE_THREADS as usize;
            }
        }
        if tile >= ntiles[0] as usize {
            return;
        }
        let expert = tiles[2 * tile] as usize;
        let (start, len) = super::super::route::tile_parts(tiles[2 * tile + 1]);
        let (start, len) = (start as usize, len as usize);
        static mut W: cuda_device::SharedArray<u32, { STAGES * W_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut B: cuda_device::SharedArray<u32, { STAGES * B_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut S: cuda_device::SharedArray<u32, { STAGES * S_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut C: cuda_device::SharedArray<u32, COLUMN_WORDS, 16> =
            cuda_device::SharedArray::UNINIT;
        // SAFETY: W is this block's shared weight allocation.
        let wp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut W) };
        // SAFETY: B is this block's shared activation allocation.
        let bp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut B) };
        // SAFETY: S is this block's shared scale allocation.
        let sp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut S) };
        // SAFETY: C is this block's shared column allocation.
        let columns = TileColumns(unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut C) });
        let row = rt * TILE_ROWS + 16 * (tid / 32);
        let g = lane / 4;
        let t = lane & 3;
        let nt = len.div_ceil(8);
        let sr = tid / 2;
        let half = tid & 1;
        let wr = expert * rows + (rt * TILE_ROWS + sr).min(rows - 1);
        let source = cols[start + sr.min(len - 1)] as usize;
        let col = if slot_div == 1 {
            source
        } else {
            source / slot_div as usize
        };
        // SAFETY: the block's column words, before the walk.
        unsafe { columns.put((sr, half), source, ag, col) };
        let mut acc = [0.0f32; 32];
        // An even K walk over whole row tiles runs without the walk's and the
        // fragments' checks, over the tile's first `$live` fragments
        // ([`live_walk`]): a dead fragment among them runs over copies of its
        // last live column (`source`), which no store reads.
        macro_rules! walk {
            ($full:literal, $live:literal) => {
                nvfp4_walk!(
                    $full,
                    (wq, we, wr),
                    (aq, ae, col),
                    (steps, sr, half, live_rows($live)),
                    (wp, bp, sp, h),
                    {
                        if $full || row < rows {
                            let r = 16 * (tid / 32) + g;
                            let a_off = chunk(16 * (tid / 32) + (lane & 15), 2 * h + (lane >> 4));
                            // SAFETY: all lanes issue the load; its four sets of eight aligned rows
                            // lie in the live shared stage.
                            let a = unsafe {
                                ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                                    wp.add(a_off).cast_const().cast::<u8>(),
                                ))
                            };
                            // SAFETY: each pair of lanes reads the scale row selected by the MMA layout.
                            let sa = unsafe { *sp.add(r + 8 * (t & 1)) };
                            for q in 0..2 {
                                ::cuda_device::thread::__unroll_config::<0>();
                                if 4 * q >= $live || (!$full && 4 * q >= nt) {
                                    continue;
                                }
                                // SAFETY: q < 2, inside this half's scale words.
                                let sb = unsafe { scale_quad(sp, g, q) };
                                // The quad's fragments load before its MMAs, so their shared
                                // waits overlap.
                                let mut b = [[0u32; 2]; 4];
                                for p in 0..2 {
                                    ::cuda_device::thread::__unroll_config::<0>();
                                    let n = 4 * q + 2 * p;
                                    if n >= $live || (!$full && n >= nt) {
                                        continue;
                                    }
                                    // SAFETY: the whole warp loads; n + 1 < 8 and h < HALVES in this
                                    // published activation stage.
                                    [b[2 * p], b[2 * p + 1]] = unsafe { b_pair(bp, n, h, lane) };
                                }
                                for i in 0..4 {
                                    ::cuda_device::thread::__unroll_config::<0>();
                                    let n = 4 * q + i;
                                    if n >= $live || (!$full && n >= nt) {
                                        continue;
                                    }
                                    let c = mma(
                                        [acc[n * 4], acc[n * 4 + 1], acc[n * 4 + 2], acc[n * 4 + 3]],
                                        a,
                                        b[i],
                                        sa,
                                        sb[i],
                                    );
                                    acc[n * 4] = c[0];
                                    acc[n * 4 + 1] = c[1];
                                    acc[n * 4 + 2] = c[2];
                                    acc[n * 4 + 3] = c[3];
                                }
                            }
                        }
                    }
                )
            };
        }
        if steps.is_multiple_of(HALVES) && rows.is_multiple_of(TILE_ROWS) {
            live_walk!(len, walk);
        } else {
            walk!(false, 8);
        }
        if row >= rows {
            return;
        }
        let weight_scale = ws[expert];
        // The lane's sixteen columns, fragment n's column j at i = 2n + j: their
        // slots and input scales are all read before the first store.
        let column_of = |i: usize| 8 * (i / 2) + 2 * t + (i & 1);
        let mut slot = [0usize; 16];
        let mut input_scale = [0.0f32; 16];
        #[unroll]
        for i in 0..16 {
            if column_of(i) < len {
                // SAFETY: column_of(i) < len <= 64, after the walk's barriers.
                (slot[i], input_scale[i]) = unsafe { columns.get(column_of(i)) };
            }
        }
        #[unroll]
        for i in 0..16 {
            if column_of(i) < len {
                let (n, j) = (i / 2, i & 1);
                // SAFETY: this lane owns the first row of this live column; row counts divide 16.
                unsafe {
                    *y.get_unchecked_mut(slot[i] * rows + row + g) =
                        crate::gemm::grouped::scaled_sum(
                            [f64::from(acc[n * 4 + j]), 0.0],
                            [weight_scale, 0.0],
                            input_scale[i],
                            0.0,
                        ) as f32;
                }
                // SAFETY: this lane owns the row eight positions later in the same live column.
                unsafe {
                    *y.get_unchecked_mut(slot[i] * rows + row + g + 8) =
                        crate::gemm::grouped::scaled_sum(
                            [f64::from(acc[n * 4 + j + 2]), 0.0],
                            [weight_scale, 0.0],
                            input_scale[i],
                            0.0,
                        ) as f32;
                }
            }
        }
    }

    /// One route tile's [`PAIR_ROWS`] gate rows and the same up rows (block
    /// `tile · row_tiles + rt`, rows `rt · PAIR_ROWS ..`) over the tile's
    /// columns, each (slot, row) the value `gemm_nvfp4_mma` stores for the
    /// gate and for the up; then their SwiGLU (NaN where either is not
    /// finite) and the 32-value Q8 of each slot's two blocks into `q`, `d`,
    /// `s` (`quant32_group`'s planes of `row_tiles` steps a column), a
    /// refused block raised as [`crate::FaultSite::QuantColumn`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn gate_up_nvfp4_mma(
        gq: &[u32],
        ge: &[u32],
        gw: &[f32],
        uq: &[u32],
        ue: &[u32],
        uw: &[f32],
        aq: &[u32],
        ae: &[u32],
        ag: &[f32],
        cols: &[u32],
        tiles: &[u32],
        ntiles: &[u32],
        rows: u32,
        steps: u32,
        slot_div: u32,
        row_tiles: u32,
        mut q: DisjointSlice<u32>,
        mut d: DisjointSlice<f32>,
        mut s: DisjointSlice<i32>,
        fault: crate::fault::FaultSink,
    ) {
        let block = thread::blockIdx_x() as usize;
        let tile = block / row_tiles as usize;
        let rt = block % row_tiles as usize;
        if tile >= ntiles[0] as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let rows = rows as usize;
        let steps = steps as usize;
        let expert = tiles[2 * tile] as usize;
        let (start, len) = super::super::route::tile_parts(tiles[2 * tile + 1]);
        let (start, len) = (start as usize, len as usize);
        static mut W: cuda_device::SharedArray<u32, { STAGES * W_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut B: cuda_device::SharedArray<u32, { STAGES * B_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut S: cuda_device::SharedArray<u32, { STAGES * S_STAGE }, 16> =
            cuda_device::SharedArray::UNINIT;
        static mut C: cuda_device::SharedArray<u32, COLUMN_WORDS, 16> =
            cuda_device::SharedArray::UNINIT;
        // SAFETY: W is this block's shared weight allocation.
        let wp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut W) };
        // SAFETY: B is this block's shared activation allocation.
        let bp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut B) };
        // SAFETY: S is this block's shared scale allocation.
        let sp = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut S) };
        // SAFETY: C is this block's shared column allocation.
        let columns = TileColumns(unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut C) });
        // The SwiGLU tile reuses W once every warp has finished the walk.
        let vp = wp.cast::<f32>();
        let wi = tid / 32;
        let (wr, wc) = (wi & 3, wi >> 2);
        let g = lane / 4;
        let t = lane & 3;
        let nt = len.div_ceil(8);
        // Shared rows 0..PAIR_ROWS hold the gate's rows, the next PAIR_ROWS the up's.
        let sr = tid / 2;
        let half = tid & 1;
        let (wq, we) = if sr < PAIR_ROWS { (gq, ge) } else { (uq, ue) };
        let wrow = expert * rows + rt * PAIR_ROWS + sr % PAIR_ROWS;
        let source = cols[start + sr.min(len - 1)] as usize;
        let col = if slot_div == 1 {
            source
        } else {
            source / slot_div as usize
        };
        // SAFETY: the block's column words, before the walk.
        unsafe { columns.put((sr, half), source, ag, col) };
        let (gate_scale, up_scale) = (gw[expert], uw[expert]);
        let mut gacc = [0.0f32; 16];
        let mut uacc = [0.0f32; 16];
        // An even K walk runs without the walk's and the fragments' checks,
        // over the tile's first `$live` fragments ([`live_walk`]): a warp
        // whose four fragments are all past them waits at the barriers alone,
        // and a dead fragment among them runs over copies of its last live
        // column (`source`), which no store reads.
        macro_rules! walk {
            ($full:literal, $live:literal) => {
                nvfp4_walk!(
                    $full,
                    (wq, we, wrow),
                    (aq, ae, col),
                    (steps, sr, half, live_rows($live)),
                    (wp, bp, sp, h),
                    {
                        if $live <= 4 && wc != 0 {
                            continue;
                        }
                        let a_row = 16 * wr + (lane & 15);
                        let a_off = chunk(a_row, 2 * h + (lane >> 4));
                        // SAFETY: all lanes issue the load; its four sets of eight aligned gate
                        // rows lie in the live stage.
                        let a_gate = unsafe {
                            ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                                wp.add(a_off).cast_const().cast::<u8>(),
                            ))
                        };
                        // SAFETY: all lanes issue the load; its four sets of eight aligned rows are the
                        // gate's rows 16 * wr + (lane & 15) < PAIR_ROWS (wr < 4) moved PAIR_ROWS rows on,
                        // the up's rows, inside the TILE_ROWS rows of the live stage.
                        let a_up = unsafe {
                            ldmatrix_x4_shared_u32(cvta_generic_to_shared_u32(
                                wp.add(chunk(PAIR_ROWS + a_row, 2 * h + (lane >> 4)))
                                    .cast_const()
                                    .cast::<u8>(),
                            ))
                        };
                        let r = 16 * wr + g + 8 * (t & 1);
                        // SAFETY: each pair of lanes reads the gate scale row selected by the MMA layout.
                        let sa_gate = unsafe { *sp.add(r) };
                        // SAFETY: r < PAIR_ROWS, so PAIR_ROWS + r < TILE_ROWS is the up's scale row in this
                        // half's scale words.
                        let sa_up = unsafe { *sp.add(PAIR_ROWS + r) };
                        // SAFETY: wc < 2, inside this half's scale words.
                        let sb = unsafe { scale_quad(sp, g, wc) };
                        // The warp's fragments load before its MMAs, so their shared waits overlap.
                        let mut bs = [[0u32; 2]; 4];
                        for p in 0..2 {
                            ::cuda_device::thread::__unroll_config::<0>();
                            let nn = 4 * wc + 2 * p;
                            if nn >= $live || (!$full && nn >= nt) {
                                continue;
                            }
                            // SAFETY: the whole warp loads; nn + 1 < 8 and h < HALVES in this published
                            // activation stage.
                            [bs[2 * p], bs[2 * p + 1]] = unsafe { b_pair(bp, nn, h, lane) };
                        }
                        for n in 0..4 {
                            ::cuda_device::thread::__unroll_config::<0>();
                            if 4 * wc + n >= $live || (!$full && 4 * wc + n >= nt) {
                                continue;
                            }
                            let (b, sb) = (bs[n], sb[n]);
                            let c = mma(
                                [
                                    gacc[n * 4],
                                    gacc[n * 4 + 1],
                                    gacc[n * 4 + 2],
                                    gacc[n * 4 + 3],
                                ],
                                a_gate,
                                b,
                                sa_gate,
                                sb,
                            );
                            gacc[n * 4] = c[0];
                            gacc[n * 4 + 1] = c[1];
                            gacc[n * 4 + 2] = c[2];
                            gacc[n * 4 + 3] = c[3];
                            let c = mma(
                                [
                                    uacc[n * 4],
                                    uacc[n * 4 + 1],
                                    uacc[n * 4 + 2],
                                    uacc[n * 4 + 3],
                                ],
                                a_up,
                                b,
                                sa_up,
                                sb,
                            );
                            uacc[n * 4] = c[0];
                            uacc[n * 4 + 1] = c[1];
                            uacc[n * 4 + 2] = c[2];
                            uacc[n * 4 + 3] = c[3];
                        }
                    }
                )
            };
        }
        if steps.is_multiple_of(HALVES) {
            live_walk!(len, walk);
        } else {
            walk!(false, 8);
        }
        thread::sync_threads();
        // The lane's eight columns, fragment n's column j & 1 at i = 2n + (j & 1):
        // their input scales are all read before the first store.
        let column_of = |i: usize| (4 * wc + i / 2) * 8 + 2 * t + (i & 1);
        let mut input_scale = [0.0f32; 8];
        #[unroll]
        for i in 0..8 {
            if column_of(i) < len {
                // SAFETY: column_of(i) < len <= 64, after the walk's barriers.
                input_scale[i] = unsafe { columns.get(column_of(i)) }.1;
            }
        }
        #[unroll]
        for n in 0..4 {
            #[unroll]
            for j in 0..4 {
                let i = 2 * n + (j & 1);
                let c = column_of(i);
                if c < len {
                    let gv = crate::gemm::grouped::scaled_sum(
                        [f64::from(gacc[n * 4 + j]), 0.0],
                        [gate_scale, 0.0],
                        input_scale[i],
                        0.0,
                    ) as f32;
                    let uv = crate::gemm::grouped::scaled_sum(
                        [f64::from(uacc[n * 4 + j]), 0.0],
                        [up_scale, 0.0],
                        input_scale[i],
                        0.0,
                    ) as f32;
                    let v = if gv.is_finite() && uv.is_finite() {
                        crate::elem::silu_mul(gv, uv)
                    } else {
                        f32::NAN
                    };
                    // SAFETY: this lane owns (c, row) of the live tile; c < 64 and the row < PAIR_ROWS.
                    unsafe {
                        *vp.add(c * PAIR_PITCH + 16 * wr + g + 8 * (j >> 1)) = v;
                    }
                }
            }
        }
        thread::sync_threads();
        let osteps = row_tiles as usize;
        let i = lane & 7;
        // Each octet quantizes one (column, block) of the tile: 64 columns of two blocks.
        let mut item = 4 * wi + (lane >> 3);
        while item - (lane >> 3) < 2 * 64 {
            let (c, blk) = (item / 2, item & 1);
            // SAFETY: c < len <= 64, after the walk's barriers.
            let slot = (c < len).then(|| unsafe { columns.get(c) }.0);
            let v = if c < len {
                // SAFETY: the barrier published this live column's aligned quad.
                unsafe {
                    (*vp.add(c * PAIR_PITCH + 32 * blk + 4 * i)
                        .cast::<cuda_device::vector::F32x4>())
                    .0
                }
            } else {
                [0.0; 4]
            };
            let (word, sum, dv, refused) = super::super::act32::quant32_lane(v, lane);
            if let Some(slot) = slot {
                let b = 2 * rt + blk;
                // SAFETY: one lane owns word i of block b < 2·osteps of this live slot's column.
                unsafe {
                    *q.get_unchecked_mut(slot * 16 * osteps + 8 * b + i) = word;
                }
                if i == 0 {
                    // SAFETY: the octet's first lane owns the block's scale.
                    unsafe {
                        *d.get_unchecked_mut(slot * 2 * osteps + b) = dv;
                    }
                    // SAFETY: the octet's first lane owns the block's code sum.
                    unsafe {
                        *s.get_unchecked_mut(slot * 2 * osteps + b) = sum;
                    }
                    if refused {
                        fault.raise(crate::FaultSite::QuantColumn);
                    }
                }
            }
            item += 32;
        }
    }

    /// The MXFP8 form of `n` f32 columns of `k` values (`act32.rs`'s
    /// `enqueue_mxfp8_cols`): their E4M3 codes, UE8M0 scale words and
    /// refusal words.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n >= 1,
            x.len() >= n * k,
            4 * qs.len() >= n * k,
            128 * e.len() >= n * k,
            refused.len() >= n
        )
    )]
    pub fn mxfp8_cols(
        x: &[f32],
        k: u32,
        n: u32,
        mut qs: DisjointSlice<u32>,
        mut e: DisjointSlice<u32>,
        mut refused: DisjointSlice<u32>,
    ) {
        // SAFETY: the checked launch supplies aligned columns, K a multiple
        // of 128 (the form's room) and the spans in the contract.
        unsafe { mxfp8_cols_body(x, k, n, &mut qs, &mut e, &mut refused) };
    }

    /// Dense MXFP8 prompt product, W8A8 on block-scaled tensor cores: the
    /// weight's planes `qs` and `e` as stored against the activations' MXFP8
    /// form (`xq`, `xe` and `refused`, [`mxfp8_cols`]), with the f32 columns
    /// `x` the replay reads.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            rows >= 1,
            n >= 1,
            4 * qs.len() >= rows * k,
            128 * e.len() >= rows * k,
            x.len() >= n * k,
            4 * xq.len() >= n * k,
            128 * xe.len() >= n * k,
            refused.len() >= n,
            y.len() >= rows * n
        )
    )]
    pub fn mxfp8_w8a8_gemm(
        qs: &[u32],
        e: &[u32],
        x: &[f32],
        xq: &[u32],
        xe: &[u32],
        refused: &[u32],
        rows: u32,
        k: u32,
        n: u32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut STAGE: SharedArray<f32, { MMA_STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, accessed without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let cols = Mxfp8Cols {
            pairs: Bf16Cols {
                words: xq,
                refused,
                exact: x,
            },
            e: xe,
        };
        // SAFETY: the checked launch supplies 256 converged threads, the grid,
        // K a multiple of 128, 16-byte aligned planes and activation words and
        // the spans in the contract.
        unsafe { bf16_mma_body(Mxfp8Rows { qs, e }, cols, rows, k, n, &mut y, sh) };
    }

    /// [`mxfp8_w8a8_gemm`] as a gate projection with the shared sigmoid norm
    /// at its store (`f32tile.rs`'s `bf16_mma_norm_gate_body`), which also
    /// writes the output's MXFP8 form (`yq`, `ye`, `yrefused` zero before
    /// the launch) for the site that reads it.
    #[allow(clippy::too_many_arguments, reason = "flat kernel ABI")]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            rows >= 128,
            n >= 1,
            4 * qs.len() >= rows * k,
            128 * e.len() >= rows * k,
            x.len() >= n * k,
            4 * xq.len() >= n * k,
            128 * xe.len() >= n * k,
            refused.len() >= n,
            o.len() >= rows * n,
            gain.len() >= 128,
            z.len() >= rows * n,
            y.len() >= rows * n,
            4 * yq.len() >= rows * n,
            128 * ye.len() >= rows * n,
            yrefused.len() >= n
        )
    )]
    pub fn mxfp8_w8a8_norm_gate(
        qs: &[u32],
        e: &[u32],
        x: &[f32],
        xq: &[u32],
        xe: &[u32],
        refused: &[u32],
        rows: u32,
        k: u32,
        n: u32,
        o: &[f32],
        gain: &[f32],
        eps: f32,
        fault: crate::fault::FaultSink,
        mut z: DisjointSlice<f32>,
        mut y: DisjointSlice<f32>,
        mut yq: DisjointSlice<u32>,
        mut ye: DisjointSlice<u32>,
        mut yrefused: DisjointSlice<u32>,
    ) {
        static mut STAGE: SharedArray<f32, { MMA_STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, accessed without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let cols = Mxfp8Cols {
            pairs: Bf16Cols {
                words: xq,
                refused,
                exact: x,
            },
            e: xe,
        };
        let w = Mxfp8Rows { qs, e };
        let form = Mxfp8Out {
            qs: yq.as_mut_ptr(),
            e: ye.as_mut_ptr(),
            refused: yrefused.as_mut_ptr(),
        };
        // SAFETY: the checked launch gives 256 converged threads and the MMA
        // grid, rows a multiple of HEAD, k a multiple of 128, 16-byte-aligned
        // qs and xq, the contract's spans of qs, e, x, xq, xe, refused, o,
        // gain, z, y and the output form's yq, ye and yrefused, and sh this
        // block's aligned MMA_STAGES·STAGE_FLOATS floats.
        unsafe {
            bf16_mma_norm_gate_body(
                w, cols, rows, k, n, o, gain, eps, fault, &mut z, &mut y, form, sh,
            )
        };
    }

    /// `hc_gated_up_mma_mix_4`, also writing `mixed`'s MXFP8 form: its
    /// E4M3 words and scale words and each column's refusal word, zero
    /// before the launch.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            hidden > 0, m > 8,
            w.len() >= 4 * hidden * 160,
            lo.len() >= m * 320,
            lo_words.len() >= m * 160,
            lo_refused.len() >= m,
            xn.len() >= m * 4 * hidden,
            g.len() >= m * 4 * hidden,
            mixed.len() >= m * hidden,
            4 * qs.len() >= m * hidden,
            128 * e.len() >= m * hidden,
            refused.len() >= m
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn hc_gated_up_mma_mix_4_mxfp8(
        w: &[u32],
        lo: &[f32],
        lo_words: &[u32],
        lo_refused: &[u32],
        xn: &[f32],
        hidden: u32,
        m: u32,
        fault: crate::fault::FaultSink,
        mut g: DisjointSlice<f32>,
        mut mixed: DisjointSlice<f32>,
        mut qs: DisjointSlice<u32>,
        mut e: DisjointSlice<u32>,
        mut refused: DisjointSlice<u32>,
    ) {
        static mut STAGE: SharedArray<f32, { MMA_STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: this block's aligned shared allocation, accessed without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        let x = (lo, lo_words, lo_refused, xn);
        let form = Mxfp8Out {
            qs: qs.as_mut_ptr(),
            e: e.as_mut_ptr(),
            refused: refused.as_mut_ptr(),
        };
        // SAFETY: the checked launch gives 256 converged threads, the MMA grid
        // and the spans above, the form's planes for the m columns of `hidden`
        // values, `hidden` a multiple of 128; sh is this block's allocation.
        unsafe {
            crate::hc_gated::up_mma_mix_body(w, x, (hidden, m), fault, &mut g, &mut mixed, form, sh)
        };
    }
}

pub(super) struct Kernels {
    module: native::LoadedModule,
}

/// [`Kernels::gate_up`]'s arguments: the gate and up stacks `w` of `rows`
/// rows each over the activations `x` through `route`'s fill, launched over
/// `max_tiles` route tiles with `slot_div` slots per activation column, and
/// the down's activations `act` filled over their first `n_cols` columns.
pub(super) struct GateUpArgs<'a> {
    pub(super) w: [crate::mxfp4_sel::Nvfp4Stack<'a>; 2],
    pub(super) x: &'a GemmAct32,
    pub(super) route: &'a GemmRoute,
    pub(super) rows: u32,
    pub(super) max_tiles: u32,
    pub(super) slot_div: u32,
    pub(super) n_cols: usize,
    pub(super) act: &'a mut GemmAct32,
}

pub(super) struct SwigluArgs<'a> {
    pub(super) g: &'a DeviceBuffer<f32>,
    pub(super) u: &'a DeviceBuffer<f32>,
    pub(super) selected: Option<(&'a DeviceBuffer<u32>, u32)>,
    pub(super) rule: crate::kquant::Act,
    pub(super) cols: u32,
    pub(super) act: &'a mut GemmAct32,
}

/// [`Kernels::mxfp8_w8a8_norm_gate`]'s arguments: the weight `w` over the
/// MXFP8 `form` of `x`, the checked launch `t`, the exact replay's scratch
/// `z`, the norm and gate operands `tail`, and the output form's words,
/// scale words and zeroed refusal words `out` ([`GemmAct32::out_planes`]).
pub(super) struct NativeNormGateArgs<'a> {
    pub(super) w: E4m3Weight<'a>,
    pub(super) x: &'a DeviceBuffer<f32>,
    pub(super) form: &'a ColsForm,
    pub(super) t: &'a TileLaunch,
    pub(super) z: &'a mut DeviceBuffer<f32>,
    pub(super) tail: NormGateTail<'a>,
    pub(super) out: [&'a mut DeviceBuffer<u32>; 3],
}

pub(super) struct ProductArgs<'a> {
    pub(super) w: crate::mxfp4_sel::Nvfp4Stack<'a>,
    pub(super) act: &'a GemmAct32,
    pub(super) route: &'a GemmRoute,
    pub(super) rows: u32,
    pub(super) slots: u32,
    pub(super) max_tiles: u32,
    pub(super) slot_div: u32,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

impl Kernels {
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        // SAFETY: this package owns the embedded device module.
        let module = unsafe { crate::shared_module!(native, ctx)? };
        Ok(Self { module })
    }

    pub(super) fn quantize(
        &self,
        stream: &CudaStream,
        act: &mut GemmAct32,
    ) -> Result<(), GpuError> {
        let filled = act.filled() as u32;
        if let Some(out) = act.native.as_mut() {
            let prep = self
                .module
                .prepare_quantize_nvfp4_32(LaunchConfig1D::new(filled, 32, 0))?;
            self.module.quantize_nvfp4_32(
                stream,
                &prep,
                &act.q,
                &act.d,
                (act.k / 64) as u32,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
            )?;
        }
        Ok(())
    }

    pub(super) fn swiglu(
        &self,
        stream: &CudaStream,
        a: SwigluArgs<'_>,
        fault: crate::fault::FaultSink,
    ) -> Result<(), GpuError> {
        let SwigluArgs {
            g,
            u,
            selected,
            rule,
            cols,
            act,
        } = a;
        let (code, limit) = rule.code("NVFP4 activation")?;
        crate::tensor::check_alignment::<cuda_device::vector::F32x4, _>(
            "NVFP4 activation",
            "g",
            g,
        )?;
        crate::tensor::check_alignment::<cuda_device::vector::F32x4, _>(
            "NVFP4 activation",
            "u",
            u,
        )?;
        let k = act.k as u32;
        let shared = ((act.k.min(ACT_CACHE_VALUES) + 2) * 4) as u32;
        let out = act
            .native
            .as_mut()
            .ok_or_else(|| GpuError::shape("NVFP4 activation", "K must divide 64"))?;
        let config = LaunchConfig1D::new(cols, 64, shared);
        if let Some((sel, n_card)) = selected {
            let prep = self.module.prepare_swiglu_nvfp4_32_sel(config)?;
            self.module.swiglu_nvfp4_32_sel(
                stream,
                &prep,
                g,
                u,
                sel,
                n_card,
                code,
                limit,
                k,
                &mut act.q,
                &mut act.d,
                &mut act.s,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
                fault,
            )?;
        } else {
            let prep = self.module.prepare_swiglu_nvfp4_32(config)?;
            self.module.swiglu_nvfp4_32(
                stream,
                &prep,
                g,
                u,
                code,
                limit,
                k,
                &mut act.q,
                &mut act.d,
                &mut act.s,
                &mut out.q,
                &mut out.e,
                &mut out.scale,
                fault,
            )?;
        }
        Ok(())
    }

    /// Enqueue the gate and up of `a.route`'s card slots over `a.x` in one
    /// launch (`gate_up_nvfp4_mma`), the SwiGLU of each (slot, row) and its
    /// 32-value Q8 into `a.act` (the bytes `swiglu` stores for the slot),
    /// then `a.act`'s FP4 from that Q8 ([`Kernels::quantize`]) over its first
    /// `a.n_cols` columns. A slot the route does not list keeps its Q8 bytes.
    pub(super) fn gate_up(
        &self,
        stream: &CudaStream,
        a: GateUpArgs<'_>,
        fault: crate::fault::FaultSink,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "NVFP4 gate and up";
        let GateUpArgs {
            w: [gate, up],
            x,
            route,
            rows,
            max_tiles,
            slot_div,
            n_cols,
            act,
        } = a;
        let a = x
            .native
            .as_ref()
            .ok_or_else(|| GpuError::shape(WHAT, "activation K must divide 64"))?;
        if act.native.is_none() || act.k != rows as usize {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a SwiGLU of {rows} rows into native 32-value columns of {} values",
                    act.k
                ),
            ));
        }
        let rt = rows / PAIR_ROWS as u32;
        let grid = max_tiles
            .checked_mul(rt)
            .ok_or_else(|| GpuError::shape(WHAT, "grid overflow"))?;
        let prep =
            self.module
                .prepare_gate_up_nvfp4_mma(LaunchConfig1D::new(grid, TILE_THREADS, 0))?;
        self.module.gate_up_nvfp4_mma(
            stream,
            &prep,
            gate.qs.buf(),
            gate.e.buf(),
            gate.scales.buf(),
            up.qs.buf(),
            up.e.buf(),
            up.scales.buf(),
            &a.q,
            &a.e,
            &a.scale,
            &route.cols,
            &route.tiles,
            &route.n_tiles,
            rows,
            (x.k / 64) as u32,
            slot_div,
            rt,
            &mut act.q,
            &mut act.d,
            &mut act.s,
            fault,
        )?;
        act.fill(n_cols);
        self.quantize(stream, act)
    }

    pub(super) fn product(&self, stream: &CudaStream, a: ProductArgs<'_>) -> Result<(), GpuError> {
        let ProductArgs {
            w,
            act,
            route,
            rows,
            slots,
            max_tiles,
            slot_div,
            y,
        } = a;
        let a = act
            .native
            .as_ref()
            .ok_or_else(|| GpuError::shape("NVFP4 MMA", "activation K must divide 64"))?;
        let rt = rows.div_ceil(TILE_ROWS as u32);
        let grid = max_tiles
            .checked_mul(rt)
            .ok_or_else(|| GpuError::shape("NVFP4 MMA", "grid overflow"))?;
        let prep =
            self.module
                .prepare_gemm_nvfp4_mma(LaunchConfig1D::new(grid, TILE_THREADS, 0))?;
        self.module.gemm_nvfp4_mma(
            stream,
            &prep,
            w.qs.buf(),
            w.e.buf(),
            w.scales.buf(),
            &a.q,
            &a.e,
            &a.scale,
            &route.cols,
            &route.tiles,
            &route.n_tiles,
            rows,
            (act.k / 64) as u32,
            slots,
            slot_div,
            rt,
            y,
        )?;
        Ok(())
    }

    /// The MXFP8 form of `n` f32 columns of `k` values of `x` into `form`
    /// ([`native::mxfp8_cols`]) over the checked launch `cfg`.
    pub(super) fn mxfp8_cols(
        &self,
        stream: &CudaStream,
        cfg: LaunchConfig1D,
        x: &DeviceBuffer<f32>,
        (k, n): (u32, u32),
        form: &mut ColsForm,
    ) -> Result<(), GpuError> {
        let prep = self.module.prepare_mxfp8_cols(cfg)?;
        let (qs, e) = (&mut form.words, &mut form.scales);
        self.module
            .mxfp8_cols(stream, &prep, x, k, n, qs, e, &mut form.refused)?;
        Ok(())
    }

    /// The W8A8 product of the MXFP8 weight `w` and `form`, the MXFP8 form of
    /// `x`'s columns, over the checked tile launch `t`
    /// ([`native::mxfp8_w8a8_gemm`]).
    pub(super) fn mxfp8_w8a8(
        &self,
        stream: &CudaStream,
        w: E4m3Weight<'_>,
        x: &DeviceBuffer<f32>,
        form: &ColsForm,
        t: &TileLaunch,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
        let prep = self.module.prepare_mxfp8_w8a8_gemm(cfg)?;
        let (qs, e) = (w.qs.buf(), w.e.buf());
        let (xq, xe, refused) = (&form.words, &form.scales, &form.refused);
        self.module.mxfp8_w8a8_gemm(
            stream, &prep, qs, e, x, xq, xe, refused, t.rows, t.k, t.n, y,
        )?;
        Ok(())
    }

    /// [`Self::mxfp8_w8a8`] with the shared sigmoid norm `a.tail` at its
    /// store and the MXFP8 form of its output into `a.out`, which has room
    /// for it ([`native::mxfp8_w8a8_norm_gate`]); `a.z` is the exact
    /// replay's scratch.
    pub(super) fn mxfp8_w8a8_norm_gate(
        &self,
        stream: &CudaStream,
        a: NativeNormGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let NativeNormGateArgs {
            w,
            x,
            form,
            t,
            z,
            tail,
            out: [yq, ye, yrefused],
        } = a;
        let cfg = LaunchConfig1D::new(t.grid, MMA_THREADS, 0);
        let prep = self.module.prepare_mxfp8_w8a8_norm_gate(cfg)?;
        let (qs, e) = (w.qs.buf(), w.e.buf());
        let (xq, xe, refused) = (&form.words, &form.scales, &form.refused);
        self.module.mxfp8_w8a8_norm_gate(
            stream, &prep, qs, e, x, xq, xe, refused, t.rows, t.k, t.n, tail.o, tail.w, tail.eps,
            tail.fault, z, tail.y, yq, ye, yrefused,
        )?;
        Ok(())
    }

    /// `hc_gated_up_mma_mix_4` over the checked launch `a.cfg`, also writing
    /// `mixed`'s MXFP8 form into `a.planes`
    /// ([`native::hc_gated_up_mma_mix_4_mxfp8`]).
    pub(super) fn hc_up_mix_mxfp8(
        &self,
        stream: &CudaStream,
        a: HcUpMixArgs<'_>,
    ) -> Result<(), GpuError> {
        let HcUpMixArgs {
            cfg,
            w,
            lo,
            lo_words,
            lo_refused,
            xn,
            hidden,
            m,
            fault,
            g,
            mixed,
            planes: [qs, e, refused],
        } = a;
        let prep = self.module.prepare_hc_gated_up_mma_mix_4_mxfp8(cfg)?;
        self.module.hc_gated_up_mma_mix_4_mxfp8(
            stream, &prep, w, lo, lo_words, lo_refused, xn, hidden, m, fault, g, mixed, qs, e,
            refused,
        )?;
        Ok(())
    }
}
