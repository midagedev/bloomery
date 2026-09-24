//! Non-causal multi-head attention over one image: every patch attends to every patch
//! (`F.scaled_dot_product_attention(q, k, v)`, no mask, scale `1/√64`), heads of 64 values.
//!
//! Why online softmax: a two-pass form writes the scores, `heads · N²` f32 — 1.9 GB at
//! `N = 5476` and 4.4 GB at the set's widest image (`N = 8296`). The online form keeps each
//! query row's running max and sum in registers and never stores a score.
//!
//! Geometry: one block of [`THREADS`] threads per (head, [`Q_ROWS`] query rows); each of its four
//! warps owns 16 query rows (one m16 fragment). The block walks the keys in [`KEY_TILE`]-key
//! tiles, staging the tile's key and value rows into shared memory as bf16 pairs:
//! 1. `S = Q·Kᵀ` on `mma.sync.m16n8k16` (bf16 inputs, exact products, f32 accumulation over the
//!    head's 64 values in four k16 steps), then `s = S · scale`; a key past `N` is `−inf`.
//! 2. Per query row: the tile max (the lane's values, then the xor butterfly over the four
//!    lanes of the row), `m' = max(m, tile max)`, `c = exp(m − m')`; `p = exp(s − m')`; the
//!    lane's partial sum `l = l·c + Σ p` and its output accumulators times `c`.
//! 3. `O += P·V` on `mma.sync` with `P` split into two bf16 operands, `hi = bf16(p)` and
//!    `lo = bf16(p − hi)`, both multiplied into the same f32 accumulators: the reference keeps
//!    `P` in f32 (its 3-D call runs torch's MATH backend in f32 — the oracle's MANIFEST `# sdpa`
//!    row), and one bf16 `P` would round every weight by up to 2⁻⁹ where `hi + lo` leaves 2⁻¹⁷.
//! 4. After the last tile: the row sum over the four lanes, `o / l`, rounded once to bf16.
//!
//! The host rule ([`attn_ref`]) is the exact attention (f64) of the same bf16 `q`, `k`, `v`; the
//! kernel differs from it by the f32 roundings of steps 1–4, which [`attn_bound`] bounds per row.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne, pack_bf16_pair};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one head.
pub const HEAD_DIM: usize = 64;
/// Query rows one block covers: four warps of 16.
pub const Q_ROWS: usize = 64;
/// Keys one tile stages.
pub const KEY_TILE: usize = 64;
/// Threads per block.
pub const THREADS: usize = 128;
const THREADS_U32: u32 = THREADS as u32;
/// u32 words per staged row: the head's 64 bf16 as 32 pairs, then four words of pad — an odd
/// multiple of 16 bytes, so an `ldmatrix` phase's eight rows hit eight distinct bank quads.
const ROW_WORDS: usize = HEAD_DIM / 2 + 4;
const TILE_WORDS: usize = KEY_TILE * ROW_WORDS;
/// Words each thread stages per tile, per operand.
const STAGE_PER_THREAD: usize = KEY_TILE * (HEAD_DIM / 2) / THREADS;

const _: () = assert!(Q_ROWS == KEY_TILE && Q_ROWS == 4 * 16 && THREADS == 128);
const _: () = assert!(HEAD_DIM == 64);
const _: () = assert!((ROW_WORDS * 4).is_multiple_of(16) && ((ROW_WORDS * 4) / 16) % 2 == 1);
const _: () = assert!(STAGE_PER_THREAD * THREADS == KEY_TILE * (HEAD_DIM / 2));

/// The rows of `[n, row_width]` bf16 `qkv` a head reads: query, key and value at columns
/// `q0 + h·64`, `k0 + h·64`, `v0 + h·64`.
#[derive(Clone, Copy, Debug)]
pub struct QkvLayout {
    pub row_width: usize,
    pub q0: usize,
    pub k0: usize,
    pub v0: usize,
    pub n_heads: usize,
}

/// One output of the exact attention: `o[d] = Σ_j p_j v_j[d]` in f64 for head `h`, query row `i`,
/// with `p = softmax(scale · q_i · k_j)`, and per dim `Σ_j p_j |v_j[d]|` and the row's largest
/// `scale · Σ_d |q_i[d] k_j[d]|` — the magnitudes [`attn_bound`] scales.
#[must_use]
pub fn attn_ref(
    qkv: &[u16],
    lay: QkvLayout,
    n: usize,
    scale: f64,
    h: usize,
    i: usize,
) -> ([f64; HEAD_DIM], [f64; HEAD_DIM], f64) {
    let col = |r: usize, c0: usize, d: usize| {
        f64::from(crate::bf16_f32(
            qkv[r * lay.row_width + c0 + h * HEAD_DIM + d],
        ))
    };
    let q: Vec<f64> = (0..HEAD_DIM).map(|d| col(i, lay.q0, d)).collect();
    let mut s = Vec::with_capacity(n);
    let mut logit_mag = 0.0f64;
    for j in 0..n {
        let (mut dot, mut mag) = (0.0f64, 0.0f64);
        for (d, qd) in q.iter().enumerate() {
            let p = qd * col(j, lay.k0, d);
            dot += p;
            mag += p.abs();
        }
        s.push(dot * scale);
        logit_mag = logit_mag.max(mag * scale);
    }
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let l: f64 = w.iter().sum();
    let (mut o, mut mag) = ([0.0f64; HEAD_DIM], [0.0f64; HEAD_DIM]);
    for (j, wj) in w.iter().enumerate() {
        let p = wj / l;
        for d in 0..HEAD_DIM {
            let v = col(j, lay.v0, d);
            o[d] += p * v;
            mag[d] += p * v.abs();
        }
    }
    (o, mag, logit_mag)
}

/// The widest distance the kernel's f32 arithmetic can put between its output (before the bf16
/// rounding) and the exact `o[d]`, from [`attn_ref`]'s magnitudes at `n` keys:
/// `Σ p|v| · (2⁻¹⁶ + 2⁻²¹ + 2·ε_s + (n/16 + 4)·2⁻²³) + |o|·(n + 4)·2⁻²³`. The terms, per weight:
/// the `hi + lo` split (2⁻¹⁷, doubled for the tile rescales that multiply it), `expf`'s error
/// (2 ulp) and the max-difference rounding; the logit error `ε_s = 65·2⁻²³·(the row's largest
/// scale·Σ|q k|)` moves a weight's exponent, twice (in `p` and in `l`); the f32 accumulation of
/// `o` over `n/16` tensor-core steps; and the f32 sum `l` over `n` weights, which scales the
/// whole output.
#[must_use]
pub fn attn_bound(o: f64, mag: f64, logit_mag: f64, n: usize) -> f64 {
    let u = f64::powi(2.0, -23);
    let eps_s = 65.0 * u * logit_mag;
    mag * (f64::powi(2.0, -16) + f64::powi(2.0, -21) + 2.0 * eps_s + (n as f64 / 16.0 + 4.0) * u)
        + o.abs() * (n as f64 + 4.0) * u
}

#[cuda_module]
mod attn_kernels {
    use super::*;

    /// The module doc's attention over `n` patches of `qkv` (rows of `row_width` bf16; query,
    /// key and value heads at columns `q0`, `k0`, `v0` plus `h·64`), `n_heads` heads, writing
    /// head `h` of row `i` at `out[i·out_width + h·64 ..]`. Block `b` is head `b % n_heads`,
    /// query rows `64·(b / n_heads) ..`; the row guard is block-uniform.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (qkv.len() >= n * row_width, out.len() >= n * out_width)
    )]
    pub fn vis_attn(
        qkv: &[u16],
        n: u32,
        row_width: u32,
        q0: u32,
        k0: u32,
        v0: u32,
        n_heads: u32,
        scale: f32,
        out_width: u32,
        mut out: DisjointSlice<u16>,
    ) {
        static mut QS: SharedArray<u32, TILE_WORDS> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, TILE_WORDS> = SharedArray::UNINIT;
        static mut VS: SharedArray<u32, TILE_WORDS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let blk = thread::blockIdx_x() as usize;
        let rows = n as usize;
        let heads = n_heads as usize;
        let h = blk % heads;
        let qb = blk / heads;
        if qb * Q_ROWS >= rows {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        // SAFETY: block-shared, TILE_WORDS words each, written only by the staging loops between
        // the barriers that publish them.
        let (qs, ks, vs) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut VS),
            )
        };
        let src = qkv.as_ptr().cast::<u32>();
        let rw = (row_width / 2) as usize; // words per qkv row
        let (qw, kw, vw) = (
            (q0 as usize + h * HEAD_DIM) / 2,
            (k0 as usize + h * HEAD_DIM) / 2,
            (v0 as usize + h * HEAD_DIM) / 2,
        );

        // Stage this block's query rows; a row past `n` is zero.
        let mut raw = [0u32; STAGE_PER_THREAD];
        let mut s = 0usize;
        while s < STAGE_PER_THREAD {
            cuda_device::thread::__unroll_config::<0>();
            let i = tid + s * THREADS;
            let r = i / (HEAD_DIM / 2);
            let w = i - r * (HEAD_DIM / 2);
            let row = qb * Q_ROWS + r;
            if row < rows {
                // SAFETY: row < n and qw + w < row_width / 2 (host-checked column layout) keep
                // the word inside `qkv` (contract); columns and row width are even, so aligned.
                raw[s] = unsafe { *src.add(row * rw + qw + w) };
            }
            s += 1;
        }
        let mut s = 0usize;
        while s < STAGE_PER_THREAD {
            cuda_device::thread::__unroll_config::<0>();
            let i = tid + s * THREADS;
            let r = i / (HEAD_DIM / 2);
            let w = i - r * (HEAD_DIM / 2);
            // SAFETY: r < Q_ROWS and w < HEAD_DIM / 2 < ROW_WORDS; one owner per word.
            unsafe {
                *qs.add(r * ROW_WORDS + w) = raw[s];
            }
            s += 1;
        }
        thread::sync_threads();

        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let group = lane / 4;
        let t4 = lane % 4;
        // This warp's 16 query rows as A fragments, one per k16 step of the head.
        let mut qa = [[0u32; 4]; 4];
        let mut kk = 0usize;
        while kk < 4 {
            cuda_device::thread::__unroll_config::<0>();
            let row = wid * 16 + (lane % 16);
            // SAFETY: row < Q_ROWS and word kk·8 + (lane / 16)·4 + 4 <= 32 keep the 16-byte row
            // inside the query tile, published by the barrier above.
            let p = unsafe { qs.add(row * ROW_WORDS + kk * 8 + (lane / 16) * 4) };
            // SAFETY: every lane reaches this load with the same qualifiers and an aligned
            // address inside the query tile.
            qa[kk] = unsafe {
                cuda_device::wmma::ldmatrix_x4_shared_u32(
                    cuda_device::shared::cvta_generic_to_shared_u32(p.cast_const().cast::<u8>()),
                )
            };
            kk += 1;
        }

        // Per row half (rows `group` and `group + 8` of the warp's 16): running max, the lane's
        // partial sum, and the output accumulators (8 n8 tiles of the head's 64 values).
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 8];

        let tiles = rows.div_ceil(KEY_TILE);
        let mut kb = 0usize;
        while kb < tiles {
            // The previous tile's fragment reads are done before this tile's staging writes.
            thread::sync_threads();
            let mut rk = [0u32; STAGE_PER_THREAD];
            let mut rv = [0u32; STAGE_PER_THREAD];
            let mut s = 0usize;
            while s < STAGE_PER_THREAD {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (HEAD_DIM / 2);
                let w = i - r * (HEAD_DIM / 2);
                let key = kb * KEY_TILE + r;
                if key < rows {
                    // SAFETY: key < n and the column layout (host-checked) keep both words
                    // inside `qkv` (contract), aligned as the query words.
                    unsafe {
                        rk[s] = *src.add(key * rw + kw + w);
                        rv[s] = *src.add(key * rw + vw + w);
                    }
                }
                s += 1;
            }
            let mut s = 0usize;
            while s < STAGE_PER_THREAD {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (HEAD_DIM / 2);
                let w = i - r * (HEAD_DIM / 2);
                // SAFETY: r < KEY_TILE and w < HEAD_DIM / 2 < ROW_WORDS; one owner per word.
                unsafe {
                    *ks.add(r * ROW_WORDS + w) = rk[s];
                    *vs.add(r * ROW_WORDS + w) = rv[s];
                }
                s += 1;
            }
            thread::sync_threads();

            // ---- S = Q·Kᵀ: 8 n8 tiles of keys, 4 k16 steps of the head.
            let mut sc = [[0.0f32; 4]; 8];
            let mut kk = 0usize;
            while kk < 4 {
                cuda_device::thread::__unroll_config::<0>();
                let mut nj = 0usize;
                while nj < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = nj * 16 + (lane % 8) + 8 * (lane / 16);
                    // SAFETY: key < KEY_TILE and word kk·8 + ((lane / 8) % 2)·4 + 4 <= 32 keep
                    // the row inside the key tile, published by the barrier above.
                    let p = unsafe { ks.add(key * ROW_WORDS + kk * 8 + ((lane / 8) % 2) * 4) };
                    // SAFETY: every lane reaches this load with the same qualifiers and an
                    // aligned address inside the key tile.
                    let bf = unsafe {
                        cuda_device::wmma::ldmatrix_x4_shared_u32(
                            cuda_device::shared::cvta_generic_to_shared_u32(
                                p.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    // SAFETY: the whole warp issues both `mma.sync` with fragments it loaded.
                    unsafe {
                        sc[2 * nj] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                            sc[2 * nj],
                            qa[kk],
                            [bf[0], bf[1]],
                        );
                        sc[2 * nj + 1] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                            sc[2 * nj + 1],
                            qa[kk],
                            [bf[2], bf[3]],
                        );
                    }
                    nj += 1;
                }
                kk += 1;
            }

            // ---- online softmax. Register j of tile t is row half j / 2, key
            // `kb·64 + 8t + 2·t4 + j % 2`.
            let mut tmax = [f32::NEG_INFINITY; 2];
            let mut t = 0usize;
            while t < 8 {
                cuda_device::thread::__unroll_config::<0>();
                let mut j = 0usize;
                while j < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = kb * KEY_TILE + 8 * t + 2 * t4 + (j % 2);
                    let v = if key < rows {
                        mul_rn_f32(sc[t][j], scale)
                    } else {
                        f32::NEG_INFINITY
                    };
                    sc[t][j] = v;
                    tmax[j / 2] = tmax[j / 2].max(v);
                    j += 1;
                }
                t += 1;
            }
            let mut corr = [0.0f32; 2];
            let mut hf = 0usize;
            while hf < 2 {
                cuda_device::thread::__unroll_config::<0>();
                let mut x = tmax[hf];
                x = x.max(warp::shuffle_xor_f32(x, 1));
                x = x.max(warp::shuffle_xor_f32(x, 2));
                let m_new = m[hf].max(x);
                corr[hf] = if m[hf] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m[hf] - m_new).exp()
                };
                m[hf] = m_new;
                l[hf] = mul_rn_f32(l[hf], corr[hf]);
                hf += 1;
            }
            let mut t = 0usize;
            while t < 8 {
                cuda_device::thread::__unroll_config::<0>();
                let mut j = 0usize;
                while j < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let p = if sc[t][j] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (sc[t][j] - m[j / 2]).exp()
                    };
                    sc[t][j] = p;
                    l[j / 2] = add_rn_f32(l[j / 2], p);
                    o[t][j] = mul_rn_f32(o[t][j], corr[j / 2]);
                    j += 1;
                }
                t += 1;
            }

            // ---- O += P·V with P = hi + lo. A fragment of key step kk2 is tiles 2·kk2 and
            // 2·kk2 + 1 of P: registers (0, 1), (2, 3) of each, packed low value first.
            let mut kk2 = 0usize;
            while kk2 < 4 {
                cuda_device::thread::__unroll_config::<0>();
                let mut hi = [0u32; 4];
                let mut lo = [0u32; 4];
                let mut e = 0usize;
                while e < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let tile = 2 * kk2 + e / 2;
                    let j0 = 2 * (e % 2);
                    let (p0, p1) = (sc[tile][j0], sc[tile][j0 + 1]);
                    let (h0, h1) = (f32_to_bf16_rne(p0), f32_to_bf16_rne(p1));
                    let l0 = f32_to_bf16_rne(p0 - bf16_to_f32(h0));
                    let l1 = f32_to_bf16_rne(p1 - bf16_to_f32(h1));
                    hi[e] = pack_bf16_pair(h0, h1);
                    lo[e] = pack_bf16_pair(l0, l1);
                    e += 1;
                }
                let mut nd = 0usize;
                while nd < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = kk2 * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                    // SAFETY: key < KEY_TILE and word nd·8 + (lane / 16)·4 + 4 <= 32 keep the row
                    // inside the value tile, published by the barrier above.
                    let p = unsafe { vs.add(key * ROW_WORDS + nd * 8 + (lane / 16) * 4) };
                    // SAFETY: every lane reaches this load with the same qualifiers and an
                    // aligned address inside the value tile.
                    let bf = unsafe {
                        cuda_device::wmma::ldmatrix_x4_trans_shared_u32(
                            cuda_device::shared::cvta_generic_to_shared_u32(
                                p.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    // SAFETY: the whole warp issues these `mma.sync` with fragments it holds.
                    unsafe {
                        o[2 * nd] =
                            cuda_device::wmma::mma_m16n8k16_f32_bf16(o[2 * nd], hi, [bf[0], bf[1]]);
                        o[2 * nd] =
                            cuda_device::wmma::mma_m16n8k16_f32_bf16(o[2 * nd], lo, [bf[0], bf[1]]);
                        o[2 * nd + 1] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                            o[2 * nd + 1],
                            hi,
                            [bf[2], bf[3]],
                        );
                        o[2 * nd + 1] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                            o[2 * nd + 1],
                            lo,
                            [bf[2], bf[3]],
                        );
                    }
                    nd += 1;
                }
                kk2 += 1;
            }
            kb += 1;
        }

        // ---- the row sums over the four lanes of a row, then o / l.
        let mut hf = 0usize;
        while hf < 2 {
            cuda_device::thread::__unroll_config::<0>();
            l[hf] = add_rn_f32(l[hf], warp::shuffle_xor_f32(l[hf], 1));
            l[hf] = add_rn_f32(l[hf], warp::shuffle_xor_f32(l[hf], 2));
            hf += 1;
        }
        let ow = out_width as usize;
        let mut t = 0usize;
        while t < 8 {
            cuda_device::thread::__unroll_config::<0>();
            let mut j = 0usize;
            while j < 4 {
                cuda_device::thread::__unroll_config::<0>();
                let row = qb * Q_ROWS + wid * 16 + group + 8 * (j / 2);
                if row < rows {
                    let d = 8 * t + 2 * t4 + (j % 2);
                    let y = f32_to_bf16_rne(div_rn_f32(o[t][j], l[j / 2]));
                    // SAFETY: row < n and h·64 + d < n_heads·64 <= out_width (host-checked)
                    // keep the index below n·out_width <= out.len(); one owner lane per value.
                    unsafe {
                        *out.get_unchecked_mut(row * ow + h * HEAD_DIM + d) = y;
                    }
                }
                j += 1;
            }
            t += 1;
        }
    }
}

/// [`AttnKernels::enqueue`]'s arguments: `n` patches of `qkv` in `layout`, the output rows of
/// `out_width` bf16 (the heads side by side from column 0).
pub struct AttnArgs<'a> {
    pub qkv: &'a DeviceBuffer<u16>,
    pub layout: QkvLayout,
    pub n: usize,
    pub scale: f32,
    pub out_width: usize,
    pub out: &'a mut DeviceBuffer<u16>,
}

/// The loaded attention module.
pub struct AttnKernels {
    module: attn_kernels::LoadedModule,
}

impl AttnKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AttnKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { attn_kernels::load(ctx)? };
        Ok(AttnKernels { module })
    }

    /// Enqueue the attention ([`AttnArgs`]). Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: AttnArgs<'_>) -> Result<(), GpuError> {
        let what = "AttnKernels::enqueue";
        let AttnArgs {
            qkv,
            layout: lay,
            n,
            scale,
            out_width,
            out,
        } = args;
        let span = lay.n_heads * HEAD_DIM;
        let fits = |c0: usize| c0.is_multiple_of(2) && c0 + span <= lay.row_width;
        if n == 0
            || lay.n_heads == 0
            || !lay.row_width.is_multiple_of(2)
            || !fits(lay.q0)
            || !fits(lay.k0)
            || !fits(lay.v0)
            || span > out_width
            || qkv.len() < n * lay.row_width
            || out.len() < n * out_width
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n {n}, {lay:?}, out_width {out_width}: qkv.len() {}, out.len() {}",
                    qkv.len(),
                    out.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(Q_ROWS) * lay.n_heads)?;
        let prep = self
            .module
            .prepare_vis_attn(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.vis_attn(
            stream,
            &prep,
            qkv,
            launch_u32(what, "n", n)?,
            launch_u32(what, "row_width", lay.row_width)?,
            launch_u32(what, "q0", lay.q0)?,
            launch_u32(what, "k0", lay.k0)?,
            launch_u32(what, "v0", lay.v0)?,
            launch_u32(what, "n_heads", lay.n_heads)?,
            scale,
            launch_u32(what, "out_width", out_width)?,
            out,
        )?;
        Ok(())
    }
}
