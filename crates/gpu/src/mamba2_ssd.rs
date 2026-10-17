//! The Mamba-2 prompt scan as a chunked state-space dual (SSD): the scan of
//! [`crate::mamba2`]'s prompt pass over `U` tokens, rewritten so that every
//! product runs on the tensor cores. One launch per pass and layer, one block
//! per head; the block walks the pass in chunks of [`CHUNK`] tokens and keeps
//! the head's state in registers between them. It reads what the pass's conv
//! pre-pass wrote (the conv rows and each token's `σ`) and writes what the
//! pass's scan writes: `y` gated and un-normed, and the state after the last
//! token. The ring tail and the norm stay the pass's.
//!
//! The rule (head `h` of group `g = h / per`, channel `p` of its
//! [`HEAD_DIM`], state `n` of [`STATE`]; per token `x`, `B`, `C` the conv
//! outputs, `σ` the pre-pass's softplus, `a = σ·A`):
//!
//! ```text
//! s_t[p][n] = exp(a_t)·s_{t−1}[p][n] + B_t[n]·x_t[p]·σ_t        the scan
//! y_t[p]    = Σ_n C_t[n]·s_t[p][n]
//! out_t[p]  = SiLU(z_t[p])·(y_t[p] + x_t[p]·D)                  the gate
//! ```
//!
//! A chunk of tokens `0 .. L` from the state `S` it starts from, with `c_t =
//! Σ_{τ<=t} a_τ` the chunk's decay prefix:
//!
//! ```text
//! y_t      = exp(c_t)·Σ_n C_t[n]·S[p][n]                                  (C·Sᵀ, rows scaled)
//!          + Σ_{τ<=t} exp(c_t − c_τ)·(Σ_n C_t[n]·B_τ[n])·x_τ[p]·σ_τ       ((decay ∘ C·Bᵀ)·Xσ)
//! S'[p][n] = exp(c_L)·S[p][n] + Σ_τ exp(c_L − c_τ)·x_τ[p]·σ_τ·B_τ[n]       (Xwᵀ·B)
//! ```
//!
//! llama.cpp's CUDA backend (`ssm-scan.cu`, `ssm_scan_ssd_f32_cuda`) runs
//! the same algebra with chunks of 256 as six launches a chunk: the group's
//! `C·Bᵀ` in f32, a kernel that materializes `X·σ`, the decay-weighted `B`,
//! the decay-scaled `C` and the masked product `M` in f16, then `C·Sᵀ` in
//! f32, `X·Mᵀ` and the state update in f16 on cuBLAS, and a state scale.
//! Here one block holds a head's chunk in shared memory and fuses all of it:
//! nothing but the inputs, `y` and the state crosses the card's memory, and
//! the chunk is 32 so the chunk's operands fit two blocks an SM. Every product
//! takes f16 operands and accumulates in f32 (`mma.m16n8k16`), `C·Bᵀ` and
//! `C·Sᵀ` included, where llama.cpp keeps those two in f32.
//!
//! Geometry. A block's four warps split a chunk's 32 output rows into two
//! slabs of 16 and its 64 channels into two halves of 32: warp `w` takes slab
//! `w / 2`, half `w % 2`, and owns state rows `16w .. 16w + 16` (all 128
//! states, 64 f32 a lane). Per chunk:
//! 1. stage: the chunk's `C` and `B` (32 × 128) and `X·σ` and `X·σ·w` (32 ×
//!    64, `w_τ = exp(c_L − c_τ)`) as f16 in shared memory, rows padded by 16
//!    bytes so `ldmatrix` reads no bank twice; each warp builds `c` itself in
//!    log2 units with a shuffle scan of the chunk's `σ·A·log2(e)`;
//! 2. per warp, over the 128 states: `C·Sᵀ` for its 16 rows and 32 channels
//!    from the block's f16 copy of `S`, and `C·Bᵀ` for its rows over the keys
//!    `τ <= 16·slab + 15` only (the causal half);
//! 3. the `C·Sᵀ` rows scaled by `exp(c_t)`; `C·Bᵀ` masked to `τ <= t`, scaled
//!    by `exp(c_t − c_τ)` and rounded to f16 as an `mma` A operand in place;
//!    times `X·σ` into the same accumulators;
//! 4. `y + x·D`, `SiLU(z)·`, the finite check, `y` written;
//! 5. the state: `S·exp(c_L)`, plus `(X·σ·w)ᵀ·B` over the chunk; after the
//!    block's barrier each warp rewrites its rows of the f16 copy.
//!
//! Numeric contract. The pass's scan is f32 op for op; this one rounds each
//! product operand to f16 (`C`, `B`, `X·σ`, `X·σ·w`, `S`, the masked `M`) and
//! sums in f32, so a value differs from the exact rule by a few f16 roundings
//! of the sum of its terms' magnitudes (`gate_mamba2_ssd` derives the bound),
//! and no output is the pass's bit for bit. Decays are `ex2.approx` on f32
//! prefix sums; `SiLU` is `z·rcp(1 + ex2(−z·log2 e))`. An operand past the
//! f16 range becomes an infinity and so a non-finite `y`, never a finite
//! wrong one.
//!
//! No silent failure: a head width other than [`HEAD_DIM`], what
//! [`crate::mamba2::check`] refuses, `u` = 0, a `z` window past the row
//! stride, a buffer too short and a misaligned one are refused by name at
//! enqueue, launching nothing. A `y` that is not finite raises [`FaultSite::Mamba2Scan`] as the
//! pass's scan does. A non-finite input at token `t` reaches the rows of its
//! chunk that share a product with it — rows before `t` too, where a masked
//! zero multiplies it — and every later chunk through the state; rows of
//! earlier chunks keep their bits.

use crate::fault::{FaultSink, FaultSite};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::float::{ex2_approx_ftz_f32, rcp_approx_ftz_f32};
use cuda_device::vector::{F32x2, F32x4, U32x2};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, shared, thread, warp, wmma,
};
use cuda_host::cuda_module;
use model::mamba2::Checked;
pub use model::mamba2::STATE;
use std::f32::consts::LOG2_E;
use std::sync::Arc;

/// The one head width the SSD pass runs (Nemotron 3.5 Bolt's).
pub const HEAD_DIM: usize = 64;
/// Tokens of one chunk.
pub const CHUNK: usize = 32;
/// Threads of a block: four warps.
const BLOCK: usize = 128;
/// Shared words (f16 pairs) of one staged row of 128 states, padded by four
/// words so the eight rows an `ldmatrix` reads start on eight bank quads.
const ROW_W: usize = STATE / 2 + 4;
/// Shared words of one staged row of 64 channels, padded the same way.
const XROW_W: usize = HEAD_DIM / 2 + 4;
/// Word offsets of the block's shared tiles: the chunk's `C` and `B` (token
/// rows of 128 states), `X·σ` and `X·σ·w` (token rows of 64 channels), and
/// the state's f16 copy (channel rows of 128 states).
const C_AT: usize = 0;
const B_AT: usize = C_AT + CHUNK * ROW_W;
const X_AT: usize = B_AT + CHUNK * ROW_W;
const XW_AT: usize = X_AT + CHUNK * XROW_W;
const SH_AT: usize = XW_AT + CHUNK * XROW_W;
/// The block's shared words: 44 032 bytes, so two blocks an SM at sm_86's
/// 100 KB (each also reserves 1 KB).
const WORDS: usize = SH_AT + HEAD_DIM * ROW_W;

// The kernel's fragment walk is written for these widths: four warps of two
// 16-row slabs and two 32-channel halves, four 16-channel state slabs.
const _: () = assert!(STATE == 128 && HEAD_DIM == 64 && CHUNK == 32 && BLOCK == 128);
const _: () = assert!(2 * (4 * WORDS + 1024) <= 102_400 && 4 * WORDS <= 48 * 1024);
// Every tile and padded row starts 16-byte aligned, as `ldmatrix` rows must.
const _: () = assert!(
    ROW_W.is_multiple_of(4)
        && XROW_W.is_multiple_of(4)
        && X_AT.is_multiple_of(4)
        && SH_AT.is_multiple_of(4)
);

/// `SiLU(z) = z / (1 + exp(−z))` by the special-function unit: one `ex2`, one
/// reciprocal. `−∞` gives NaN (`−∞·0`), `+∞` gives `+∞`.
#[inline(always)]
fn silu_fast(z: f32) -> f32 {
    z * rcp_approx_ftz_f32(1.0 + ex2_approx_ftz_f32(-z * LOG2_E))
}

/// One entry of the chunk's `M`: `C·Bᵀ` decayed by `2^arg` (`arg` the key's
/// decay to the row, `<= 0`) where the key is at or before the row, 0 where
/// it is after (the decay there is not formed).
#[inline(always)]
fn masked(hit: bool, cb: f32, arg: f32) -> f32 {
    if hit {
        cb * ex2_approx_ftz_f32(arg)
    } else {
        0.0
    }
}

#[cuda_module]
mod mamba2_ssd_kernels {
    use super::*;

    /// The SSD scan of a pass of `u` tokens of one layer (module doc): block
    /// `h` is head `h`, its four warps the chunk's two row slabs times two
    /// channel halves. Reads the conv rows and `σ` of the pass's pre-pass
    /// (`conv`, `sd`), `z` from `rows` at column `z0`, the head's `A` and
    /// `D`, and the state; writes `y` row `t` (gated, un-normed) and the state
    /// after the last token.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    // Two blocks an SM: 255 registers a thread. A 128-head layer is one wave
    // on 82 or 84 SMs.
    #[launch_bounds(128, 2)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            per >= 1,
            n_group >= 1,
            u >= 1,
            n_head == per * n_group,
            ld >= z0 + n_head * HEAD_DIM,
            conv.len() >= u * (n_head * HEAD_DIM + 2 * n_group * STATE),
            sd.len() >= 2 * u * n_head,
            a.len() >= n_head,
            d.len() >= n_head,
            rows.len() >= u * ld,
            y.len() >= u * n_head * HEAD_DIM,
            state.len() >= n_head * HEAD_DIM * STATE
        )
    )]
    pub fn mamba2_ssd(
        rows: &[f32],
        a: &[f32],
        d: &[f32],
        conv: &[f32],
        sd: &[f32],
        u: u32,
        ld: u32,
        z0: u32,
        n_head: u32,
        per: u32,
        n_group: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut state: DisjointSlice<f32>,
    ) {
        static mut SM: SharedArray<u32, WORDS, 16> = SharedArray::UNINIT;

        let (n_head, per, n_group) = (n_head as usize, per as usize, n_group as usize);
        let h = thread::blockIdx_x() as usize;
        if h >= n_head {
            return; // block-uniform
        }
        let (u, ld, z0) = (u as usize, ld as usize, z0 as usize);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        // The warp's row slab and channel half; the lane's fragment row and
        // column pair.
        let (slab, half) = (w / 2, w % 2);
        let (gr, t4) = (lane / 4, lane % 4);
        let g = h / per;
        let inner = n_head * HEAD_DIM;
        let conv_dim = inner + 2 * n_group * STATE;
        let x_col = h * HEAD_DIM;
        let b_col = inner + g * STATE;
        let c_col = inner + n_group * STATE + g * STATE;
        // SAFETY: the block-shared static; the raw form reaches it without a
        // reference, and every access below is bounded and barrier-ordered.
        let sm = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SM) };
        // SAFETY: sm points at the start of the block's shared tiles.
        let base = unsafe { shared::cvta_generic_to_shared_u32(sm.cast_const().cast::<u8>()) };
        let (yp, sp) = (y.as_mut_ptr(), state.as_mut_ptr());
        let (convp, rowsp, sdp) = (conv.as_ptr(), rows.as_ptr(), sd.as_ptr());
        // SAFETY: h < n_head <= a.len(), d.len() by the contract.
        let (a2_h, d_h) = unsafe { (*a.get_unchecked(h) * LOG2_E, *d.get_unchecked(h)) };

        // The warp's state rows `16w + gr` and `+ 8`, states `8j + 2·t4` and
        // `+ 1`: the accumulator layout of the update's `mma`.
        let srow = (x_col + 16 * w + gr) * STATE + 2 * t4;
        let mut s = [[0.0f32; 4]; 16];
        for j in 0..16 {
            thread::__unroll_config::<0>();
            // SAFETY: (x_col + 16w + gr + 8)·128 + 8j + 2·t4 + 1 <
            // n_head·64·128 <= state.len() (contract); even offsets of an
            // allocation's f32s are 8-byte aligned.
            let (lo, hi) = unsafe {
                (
                    *sp.add(srow + 8 * j).cast::<F32x2>(),
                    *sp.add(srow + 8 * STATE + 8 * j).cast::<F32x2>(),
                )
            };
            s[j] = [lo.0[0], lo.0[1], hi.0[0], hi.0[1]];
        }
        // The warp's state rows into the block's f16 copy (the next chunk's
        // `C·Sᵀ` operand), one word a lane per fragment half: rows `16w + gr`
        // and `+ 8`, words `4j + t4` — 32 distinct banks a store.
        macro_rules! put_state {
            () => {
                for j in 0..16 {
                    thread::__unroll_config::<0>();
                    let at = SH_AT + (16 * w + gr) * ROW_W + 4 * j + t4;
                    // SAFETY: at + 8·ROW_W < SH_AT + 64·ROW_W = WORDS; this
                    // lane's own words, read only after the next barrier.
                    unsafe {
                        *sm.add(at) = cvt_f16x2_f32(s[j][0], s[j][1]);
                        *sm.add(at + 8 * ROW_W) = cvt_f16x2_f32(s[j][2], s[j][3]);
                    }
                }
            };
        }
        put_state!();

        // This lane's `ldmatrix` row addresses (bytes): `C` rows of its slab
        // (A operand), the f16 state's channel rows of its half and `B`'s
        // token rows (B operands, keys along the row), `X·σ` (B operand) and
        // `X·σ·w` (A operand) transposed, and `B` transposed (B operand).
        let quad = lane % 8;
        let (q16, q8) = (lane / 16, (lane / 8) % 2);
        let c_at = base + (4 * (C_AT + (16 * slab + lane % 16) * ROW_W + 4 * q16)) as u32;
        let sh_at = base + (4 * (SH_AT + (32 * half + quad + 8 * q16) * ROW_W + 4 * q8)) as u32;
        let b_at = base + (4 * (B_AT + (quad + 8 * q16) * ROW_W + 4 * q8)) as u32;
        let x_at = base + (4 * (X_AT + (quad + 8 * q8) * XROW_W + 16 * half + 4 * q16)) as u32;
        let xw_at = base + (4 * (XW_AT + (quad + 8 * q16) * XROW_W + 8 * w + 4 * q8)) as u32;
        let bt_at = base + (4 * (B_AT + (quad + 8 * q8) * ROW_W + 4 * q16)) as u32;
        // Byte strides: 16 rows of a 128-state tile, of a 64-channel tile.
        let rows16 = (4 * 16 * ROW_W) as u32;
        let xrows16 = (4 * 16 * XROW_W) as u32;

        // The output rows of this lane: chunk rows `16·slab + gr` and `+ 8`.
        let t_lo = 16 * slab + gr;
        let mut t0 = 0usize;
        loop {
            // ---- stage the chunk t0 .. t0 + 32.
            // Each warp scans the chunk's decay prefix itself, lane τ holding
            // token τ's: `cs = Σ_{τ'<=τ} σ·A·log2(e)`. A token past `u` holds
            // σ 0 and adds nothing.
            let tok = t0 + lane;
            let sigma = if tok < u {
                // SAFETY: 2·(tok·n_head + h) < 2·u·n_head <= sd.len()
                // (contract).
                unsafe { *sdp.add(2 * (tok * n_head + h)) }
            } else {
                0.0
            };
            let mut cs = if tok < u { sigma * a2_h } else { 0.0 };
            for k in 0..5u32 {
                thread::__unroll_config::<0>();
                let up = warp::shuffle_up_f32(cs, 1 << k);
                if (lane as u32) >= (1 << k) {
                    cs += up;
                }
            }
            let cs_last = warp::shuffle_f32(cs, 31);

            // `C` and `B`: warp w takes token rows w + 4k, lane `lane` four
            // states of each. `x`: rows 2w + lane/16 + 8k, four channels.
            let mut cb = [F32x4([0.0; 4]); 16];
            for k in 0..8 {
                thread::__unroll_config::<0>();
                let tk = t0 + w + 4 * k;
                if tk < u {
                    let row = tk * conv_dim + 4 * lane;
                    // SAFETY: tk < u, and c_col, b_col + 128 <= conv_dim, so
                    // both quads lie in conv's first u·conv_dim f32s
                    // (contract); conv_dim, c_col and b_col are multiples of
                    // 4 (64-wide heads, 128 states), so each quad is 16-byte
                    // aligned in the allocation.
                    unsafe {
                        cb[2 * k] = *convp.add(row + c_col).cast::<F32x4>();
                        cb[2 * k + 1] = *convp.add(row + b_col).cast::<F32x4>();
                    }
                }
            }
            let xr = 2 * w + lane / 16;
            let xq = 4 * (lane % 16);
            let mut xv = [F32x4([0.0; 4]); 4];
            for k in 0..4 {
                thread::__unroll_config::<0>();
                let tk = t0 + xr + 8 * k;
                if tk < u {
                    // SAFETY: tk < u and x_col + 64 <= inner < conv_dim:
                    // inside conv (contract); 16-byte aligned as above.
                    xv[k] = unsafe { *convp.add(tk * conv_dim + x_col + xq).cast::<F32x4>() };
                }
            }
            for k in 0..8 {
                thread::__unroll_config::<0>();
                let at = (w + 4 * k) * ROW_W + 2 * lane;
                let (c, b) = (cb[2 * k].0, cb[2 * k + 1].0);
                // SAFETY: at + 1 < CHUNK·ROW_W: inside the C and B tiles;
                // even word offsets of the 16-byte aligned tiles; this lane's
                // own words, read after the barrier below.
                unsafe {
                    *sm.add(C_AT + at).cast::<U32x2>() =
                        U32x2([cvt_f16x2_f32(c[0], c[1]), cvt_f16x2_f32(c[2], c[3])]);
                    *sm.add(B_AT + at).cast::<U32x2>() =
                        U32x2([cvt_f16x2_f32(b[0], b[1]), cvt_f16x2_f32(b[2], b[3])]);
                }
            }
            for k in 0..4 {
                thread::__unroll_config::<0>();
                let r = xr + 8 * k;
                // Token r's σ and its weight into the chunk-end state.
                let sig = warp::shuffle_f32(sigma, r as u32);
                let wgt = ex2_approx_ftz_f32(cs_last - warp::shuffle_f32(cs, r as u32));
                let x = xv[k].0;
                let xs = [x[0] * sig, x[1] * sig, x[2] * sig, x[3] * sig];
                let at = r * XROW_W + xq / 2;
                // SAFETY: at + 1 < CHUNK·XROW_W: inside the X and XW tiles;
                // even word offsets; this lane's own words, read after the
                // barrier below.
                unsafe {
                    *sm.add(X_AT + at).cast::<U32x2>() =
                        U32x2([cvt_f16x2_f32(xs[0], xs[1]), cvt_f16x2_f32(xs[2], xs[3])]);
                    *sm.add(XW_AT + at).cast::<U32x2>() = U32x2([
                        cvt_f16x2_f32(xs[0] * wgt, xs[1] * wgt),
                        cvt_f16x2_f32(xs[2] * wgt, xs[3] * wgt),
                    ]);
                }
            }
            // The output's `x` and `z`, read now and used after the products:
            // rows t_lo and t_lo + 8, channels 32·half + 8j + 2·t4 and + 1.
            let mut xe = [[0.0f32; 2]; 8];
            let mut ze = [[0.0f32; 2]; 8];
            for r in 0..2 {
                thread::__unroll_config::<0>();
                let tk = t0 + t_lo + 8 * r;
                if tk < u {
                    for j in 0..4 {
                        thread::__unroll_config::<0>();
                        let c = x_col + 32 * half + 8 * j + 2 * t4;
                        // SAFETY: tk < u; c + 1 < inner < conv_dim and z0 +
                        // c + 1 < z0 + inner <= ld: inside conv and rows
                        // (contract); c is even, so the pair is 8-byte
                        // aligned.
                        unsafe {
                            xe[4 * r + j] = (*convp.add(tk * conv_dim + c).cast::<F32x2>()).0;
                            let zr = rowsp.add(tk * ld + z0 + c);
                            ze[4 * r + j] = [*zr, *zr.add(1)];
                        }
                    }
                }
            }
            thread::sync_threads();

            // ---- C·Sᵀ (16 rows × 32 channels) and C·Bᵀ (16 rows × the
            // causal keys), over the 128 states in steps of 16.
            let mut ya = [[0.0f32; 4]; 4];
            let mut ga = [[0.0f32; 4]; 4];
            for kk in 0..8 {
                thread::__unroll_config::<0>();
                let ko = (32 * kk) as u32;
                // SAFETY: every address is a lane's row inside its staged
                // tile (the offsets above, plus 16-row and 16-state steps
                // that stay inside it), written before the barrier above;
                // the whole warp issues each ldmatrix and mma (slab is
                // warp-uniform).
                unsafe {
                    let af = wmma::ldmatrix_x4_shared_u32(c_at + ko);
                    for jj in 0..2 {
                        thread::__unroll_config::<0>();
                        let bf = wmma::ldmatrix_x4_shared_u32(sh_at + rows16 * jj as u32 + ko);
                        ya[2 * jj] = wmma::mma_m16n8k16_f32_f16(ya[2 * jj], af, [bf[0], bf[1]]);
                        ya[2 * jj + 1] =
                            wmma::mma_m16n8k16_f32_f16(ya[2 * jj + 1], af, [bf[2], bf[3]]);
                    }
                    let bf = wmma::ldmatrix_x4_shared_u32(b_at + ko);
                    ga[0] = wmma::mma_m16n8k16_f32_f16(ga[0], af, [bf[0], bf[1]]);
                    ga[1] = wmma::mma_m16n8k16_f32_f16(ga[1], af, [bf[2], bf[3]]);
                    if slab == 1 {
                        let bf = wmma::ldmatrix_x4_shared_u32(b_at + rows16 + ko);
                        ga[2] = wmma::mma_m16n8k16_f32_f16(ga[2], af, [bf[0], bf[1]]);
                        ga[3] = wmma::mma_m16n8k16_f32_f16(ga[3], af, [bf[2], bf[3]]);
                    }
                }
            }
            // The state term's rows decay from the chunk start.
            let cs_lo = warp::shuffle_f32(cs, t_lo as u32);
            let cs_hi = warp::shuffle_f32(cs, (t_lo + 8) as u32);
            let (e_lo, e_hi) = (ex2_approx_ftz_f32(cs_lo), ex2_approx_ftz_f32(cs_hi));
            for j in 0..4 {
                thread::__unroll_config::<0>();
                ya[j] = [
                    ya[j][0] * e_lo,
                    ya[j][1] * e_lo,
                    ya[j][2] * e_hi,
                    ya[j][3] * e_hi,
                ];
            }
            // M = C·Bᵀ masked to τ <= t and decayed by exp(c_t − c_τ), packed
            // as the A operand of the 16-key steps: tile jt's pairs are
            // fragment words 2·(jt % 2) and + 1 of step jt / 2.
            let mut pa = [[0u32; 4]; 2];
            for jt in 0..4 {
                thread::__unroll_config::<0>();
                if jt < 2 || slab == 1 {
                    let ta = 8 * jt + 2 * t4;
                    let cs_a = warp::shuffle_f32(cs, ta as u32);
                    let cs_b = warp::shuffle_f32(cs, (ta + 1) as u32);
                    let m0 = masked(ta <= t_lo, ga[jt][0], cs_lo - cs_a);
                    let m1 = masked(ta < t_lo, ga[jt][1], cs_lo - cs_b);
                    let m2 = masked(ta <= t_lo + 8, ga[jt][2], cs_hi - cs_a);
                    let m3 = masked(ta < t_lo + 8, ga[jt][3], cs_hi - cs_b);
                    pa[jt / 2][2 * (jt % 2)] = cvt_f16x2_f32(m0, m1);
                    pa[jt / 2][2 * (jt % 2) + 1] = cvt_f16x2_f32(m2, m3);
                }
            }
            // + M·(X·σ): the slab's key steps, the warp's 32 channels.
            for jj in 0..2 {
                thread::__unroll_config::<0>();
                if jj == 0 || slab == 1 {
                    for j2 in 0..2 {
                        thread::__unroll_config::<0>();
                        // SAFETY: as the loop above: X·σ rows 16jj + 0..16
                        // and channel octets inside the tile; warp-uniform.
                        unsafe {
                            let bf = wmma::ldmatrix_x4_trans_shared_u32(
                                x_at + xrows16 * jj as u32 + (32 * j2) as u32,
                            );
                            ya[2 * j2] =
                                wmma::mma_m16n8k16_f32_f16(ya[2 * j2], pa[jj], [bf[0], bf[1]]);
                            ya[2 * j2 + 1] =
                                wmma::mma_m16n8k16_f32_f16(ya[2 * j2 + 1], pa[jj], [bf[2], bf[3]]);
                        }
                    }
                }
            }

            // ---- the state: S·exp(c_L) + (X·σ·w)ᵀ·B over the chunk's keys.
            let decay = ex2_approx_ftz_f32(cs_last);
            for j in 0..16 {
                thread::__unroll_config::<0>();
                s[j] = [
                    s[j][0] * decay,
                    s[j][1] * decay,
                    s[j][2] * decay,
                    s[j][3] * decay,
                ];
            }
            for kk in 0..2 {
                thread::__unroll_config::<0>();
                // SAFETY: as above: X·σ·w rows 16kk + 0..16 at the warp's 16
                // channels, B rows 16kk + 0..16 at every state octet, inside
                // their tiles; warp-uniform.
                unsafe {
                    let af = wmma::ldmatrix_x4_trans_shared_u32(xw_at + xrows16 * kk as u32);
                    for jj in 0..8 {
                        thread::__unroll_config::<0>();
                        let bf = wmma::ldmatrix_x4_trans_shared_u32(
                            bt_at + rows16 * kk as u32 + (32 * jj) as u32,
                        );
                        s[2 * jj] = wmma::mma_m16n8k16_f32_f16(s[2 * jj], af, [bf[0], bf[1]]);
                        s[2 * jj + 1] =
                            wmma::mma_m16n8k16_f32_f16(s[2 * jj + 1], af, [bf[2], bf[3]]);
                    }
                }
            }

            // ---- y = SiLU(z)·(y + x·D), rows t_lo and t_lo + 8.
            for r in 0..2 {
                thread::__unroll_config::<0>();
                let tk = t0 + t_lo + 8 * r;
                if tk < u {
                    for j in 0..4 {
                        thread::__unroll_config::<0>();
                        let (x, z) = (xe[4 * r + j], ze[4 * r + j]);
                        let o0 = silu_fast(z[0]) * (ya[j][2 * r] + x[0] * d_h);
                        let o1 = silu_fast(z[1]) * (ya[j][2 * r + 1] + x[1] * d_h);
                        if !(o0.is_finite() && o1.is_finite()) {
                            fault.raise(FaultSite::Mamba2Scan);
                        }
                        let at = tk * inner + x_col + 32 * half + 8 * j + 2 * t4;
                        // SAFETY: tk < u, so at + 1 < u·inner <= y.len()
                        // (contract); at is even (8-byte aligned); this
                        // lane is the pair's one writer.
                        unsafe { *yp.add(at).cast::<F32x2>() = F32x2([o0, o1]) };
                    }
                }
            }

            t0 += CHUNK;
            if t0 >= u {
                break; // block-uniform
            }
            // Every warp is done reading the chunk's tiles and the state's
            // copy.
            thread::sync_threads();
            put_state!();
        }

        for j in 0..16 {
            thread::__unroll_config::<0>();
            // SAFETY: the loads' addresses above; each pair is this lane's
            // own.
            unsafe {
                *sp.add(srow + 8 * j).cast::<F32x2>() = F32x2([s[j][0], s[j][1]]);
                *sp.add(srow + 8 * STATE + 8 * j).cast::<F32x2>() = F32x2([s[j][2], s[j][3]]);
            }
        }
    }
}

/// [`Mamba2Ssd::enqueue`]'s arguments: the layer's widths, a pass of `u`
/// tokens, the pass's pre-pass outputs (`conv`: `u` conv rows of `conv_dim`;
/// `sd`: two f32 a (token, head), `σ` first — [`crate::mamba2::Mamba2PassScratch`]'s),
/// the pass's in-projection rows (`z` at column `z0`, row stride `ld`), the
/// layer's `A` and `D`, the sink a non-finite `y` raises on, and the outputs:
/// `y` (`u` rows of `d_inner`, gated and un-normed) and the scan state, read
/// and written in place.
pub struct SsdArgs<'a> {
    pub shape: Checked,
    pub u: usize,
    pub conv: &'a DeviceBuffer<f32>,
    pub sd: &'a DeviceBuffer<f32>,
    pub rows: &'a DeviceBuffer<f32>,
    pub ld: usize,
    pub z0: usize,
    pub a: &'a DeviceBuffer<f32>,
    pub d: &'a DeviceBuffer<f32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// What the SSD pass refuses beyond the pass a caller checked already: what
/// [`crate::mamba2::check`] refuses, a head width other than [`HEAD_DIM`] and
/// a `conv`, `y` or `state` buffer the kernel's vector accesses would
/// misalign, by name; the checked shape otherwise. [`Mamba2Ssd::enqueue`]
/// runs it, and a caller that launches before the scan runs it first, so a
/// refused pass launches nothing.
pub fn refuse(
    what: &'static str,
    shape: Checked,
    conv: &DeviceBuffer<f32>,
    y: &DeviceBuffer<f32>,
    state: &DeviceBuffer<f32>,
) -> Result<Checked, GpuError> {
    let shape = crate::mamba2::check(what, shape.geom())?;
    if shape.head_dim() != HEAD_DIM {
        return Err(GpuError::shape(
            what,
            format!(
                "head_dim {} is refused: the SSD pass runs head_dim {HEAD_DIM}",
                shape.head_dim()
            ),
        ));
    }
    // The kernel reads conv by f32 quads and y and the state by pairs: a
    // buffer at another offset is a misaligned access on the card.
    let starts = [
        ("conv", conv.cu_deviceptr(), 16u64),
        ("y", y.cu_deviceptr(), 8),
        ("state", state.cu_deviceptr(), 8),
    ];
    if let Some((name, at, align)) = starts
        .iter()
        .find(|(_, at, align)| !at.is_multiple_of(*align))
    {
        return Err(GpuError::shape(
            what,
            format!("{name} at {at:#x} is not {align}-byte aligned"),
        ));
    }
    Ok(shape)
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct Mamba2Ssd {
    module: mamba2_ssd_kernels::LoadedModule,
}

impl Mamba2Ssd {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Mamba2Ssd, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mamba2_ssd_kernels, ctx)? };
        Ok(Mamba2Ssd { module })
    }

    /// Enqueue the SSD scan of a pass of `u` tokens of one layer
    /// ([`SsdArgs`], module doc): `n_head` blocks of 128 threads. Refuses what
    /// [`refuse`] refuses, `u` = 0, a `z` window past the row stride and a
    /// buffer too short, by name, launching nothing. One launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, stream: &CudaStream, args: SsdArgs<'_>) -> Result<(), GpuError> {
        let what = "mamba2_ssd::enqueue";
        let SsdArgs {
            shape,
            u,
            conv,
            sd,
            rows,
            ld,
            z0,
            a,
            d,
            fault,
            y,
            state,
        } = args;
        let shape = refuse(what, shape, conv, y, state)?;
        if u == 0 {
            return Err(GpuError::shape(
                what,
                "u 0 is refused: a pass holds a token",
            ));
        }
        let g = shape.geom();
        let (inner, heads, groups) = (g.d_inner, g.n_heads, g.n_groups);
        let conv_dim = shape.conv_dim();
        if z0.saturating_add(inner) > ld {
            return Err(GpuError::shape(
                what,
                format!("the z window {z0} + {inner} is past the row stride ld {ld}"),
            ));
        }
        let lens = [
            ("conv", conv.len(), u.saturating_mul(conv_dim)),
            ("sd", sd.len(), u.saturating_mul(2 * heads)),
            ("rows", rows.len(), u.saturating_mul(ld)),
            ("a", a.len(), heads),
            ("d", d.len(), heads),
            ("y", y.len(), u.saturating_mul(inner)),
            ("state", state.len(), inner * STATE),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let prep = self.module.prepare_mamba2_ssd(LaunchConfig1D::new(
            launch_u32(what, "heads", heads)?,
            BLOCK as u32,
            0,
        ))?;
        self.module.mamba2_ssd(
            stream,
            &prep,
            rows,
            a,
            d,
            conv,
            sd,
            launch_u32(what, "u", u)?,
            launch_u32(what, "ld", ld)?,
            launch_u32(what, "z0", z0)?,
            launch_u32(what, "n_head", heads)?,
            launch_u32(what, "per", heads / groups)?,
            launch_u32(what, "n_group", groups)?,
            fault,
            y,
            state,
        )?;
        Ok(())
    }
}
