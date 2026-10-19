//! Non-causal multi-head attention for heads of 80 values (the RADIO tower of Nemotron 3.5 Bolt:
//! 16 heads of 1280 / 16): `vis_attn`'s geometry, online softmax and `hi + lo` split of `P` over
//! a head that is five whole k16 steps of the `Q·Kᵀ` product and ten n8 tiles of the value
//! product, so no operand is padded.
//!
//! A new entry, not [`crate::attn`]'s body with a wider constant: a kernel body moved into a
//! shared inline helper changes its PTX (docs/upstream/nvlabs-ledger.md row 39), and the entries
//! of heads of 64 and 72 hold their md5. The rule is [`crate::attn`]'s: one block of 128 threads
//! per (head, 64 query rows), each warp owning 16 rows; per query row the running max and sum in
//! registers, `s = (Q·Kᵀ) · scale` with a key past `n` at `−inf`, and `o / l` rounded once to
//! bf16 at the end. The host rule is the exact attention (f64) of the same bf16 `q`, `k`, `v`
//! ([`attn_80_ref`]); [`crate::attn::attn_bound`] bounds the kernel's f32 roundings against it.

use crate::attn::{AttnArgs, KEY_TILE, Q_ROWS, QkvLayout, THREADS};
use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne, pack_bf16_pair};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one head of the RADIO tower.
pub const HEAD_DIM_80: usize = 80;
const THREADS_U32: u32 = THREADS as u32;
/// u32 words per staged row of a head of 80: its 40 bf16 pairs, then four words of pad — an odd
/// multiple of 16 bytes (176), so an `ldmatrix` phase's eight rows hit eight distinct bank quads.
const ROW_WORDS_80: usize = HEAD_DIM_80 / 2 + 4;
const TILE_WORDS_80: usize = KEY_TILE * ROW_WORDS_80;
/// Words each thread stages per tile, per operand.
const STAGE_PER_THREAD_80: usize = KEY_TILE * (HEAD_DIM_80 / 2) / THREADS;

const _: () = assert!(Q_ROWS == KEY_TILE && Q_ROWS == 4 * 16 && THREADS == 128);
const _: () = assert!(HEAD_DIM_80.is_multiple_of(16));
const _: () = assert!(STAGE_PER_THREAD_80 * THREADS == KEY_TILE * (HEAD_DIM_80 / 2));
const _: () = assert!((ROW_WORDS_80 * 4).is_multiple_of(16) && ((ROW_WORDS_80 * 4) / 16) % 2 == 1);

/// One output of the exact attention for heads of `D` values: `o[d] = Σ_j p_j v_j[d]` in f64 for
/// head `h`, query row `i`, with `p = softmax(scale · q_i · k_j)`, per dim `Σ_j p_j |v_j[d]|`,
/// and the row's largest `scale · Σ_d |q_i[d] k_j[d]|` — the magnitudes
/// [`crate::attn::attn_bound`] scales.
fn exact<const D: usize>(
    qkv: &[u16],
    lay: QkvLayout,
    n: usize,
    scale: f64,
    h: usize,
    i: usize,
) -> ([f64; D], [f64; D], f64) {
    let col = |r: usize, c0: usize, d: usize| {
        f64::from(crate::bf16_f32(qkv[r * lay.row_width + c0 + h * D + d]))
    };
    let q: Vec<f64> = (0..D).map(|d| col(i, lay.q0, d)).collect();
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
    let (mut o, mut mag) = ([0.0f64; D], [0.0f64; D]);
    for (j, wj) in w.iter().enumerate() {
        let p = wj / l;
        for d in 0..D {
            let v = col(j, lay.v0, d);
            o[d] += p * v;
            mag[d] += p * v.abs();
        }
    }
    (o, mag, logit_mag)
}

/// [`crate::attn::attn_ref`] for heads of 80 values.
#[must_use]
pub fn attn_80_ref(
    qkv: &[u16],
    lay: QkvLayout,
    n: usize,
    scale: f64,
    h: usize,
    i: usize,
) -> ([f64; HEAD_DIM_80], [f64; HEAD_DIM_80], f64) {
    exact::<HEAD_DIM_80>(qkv, lay, n, scale, h, i)
}

#[cuda_module]
mod attn_80_kernels {
    use super::*;

    /// The module doc's attention over `n` rows of `qkv` (rows of `row_width` bf16; query, key
    /// and value heads at columns `q0`, `k0`, `v0` plus `h·80`), `n_heads` heads, writing head
    /// `h` of row `i` at `out[i·out_width + h·80 ..]`. Block `b` is head `b % n_heads`, query
    /// rows `64·(b / n_heads) ..`; the row guard is block-uniform.
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
    pub fn vis_attn_h80(
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
        static mut QS: SharedArray<u32, TILE_WORDS_80> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, TILE_WORDS_80> = SharedArray::UNINIT;
        static mut VS: SharedArray<u32, TILE_WORDS_80> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let blk = thread::blockIdx_x() as usize;
        let rows = n as usize;
        let heads = n_heads as usize;
        let h = blk % heads;
        let qb = blk / heads;
        if qb * Q_ROWS >= rows {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        // SAFETY: block-shared, TILE_WORDS_80 words each, written only by the staging loops
        // between the barriers that publish them.
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
            (q0 as usize + h * HEAD_DIM_80) / 2,
            (k0 as usize + h * HEAD_DIM_80) / 2,
            (v0 as usize + h * HEAD_DIM_80) / 2,
        );

        // Stage this block's query rows; a row past `n` is zero.
        let mut raw = [0u32; STAGE_PER_THREAD_80];
        let mut s = 0usize;
        while s < STAGE_PER_THREAD_80 {
            cuda_device::thread::__unroll_config::<0>();
            let i = tid + s * THREADS;
            let r = i / (HEAD_DIM_80 / 2);
            let w = i - r * (HEAD_DIM_80 / 2);
            let row = qb * Q_ROWS + r;
            if row < rows {
                // SAFETY: row < n and qw + w < row_width / 2 (host-checked column layout) keep
                // the word inside `qkv` (contract); columns and row width are even, so aligned.
                raw[s] = unsafe { *src.add(row * rw + qw + w) };
            }
            s += 1;
        }
        let mut s = 0usize;
        while s < STAGE_PER_THREAD_80 {
            cuda_device::thread::__unroll_config::<0>();
            let i = tid + s * THREADS;
            let r = i / (HEAD_DIM_80 / 2);
            let w = i - r * (HEAD_DIM_80 / 2);
            // SAFETY: r < Q_ROWS and w < HEAD_DIM_80 / 2 < ROW_WORDS_80; one owner per word.
            unsafe {
                *qs.add(r * ROW_WORDS_80 + w) = raw[s];
            }
            s += 1;
        }
        thread::sync_threads();

        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        let group = lane / 4;
        let t4 = lane % 4;
        // This warp's 16 query rows as A fragments, one per k16 step of the head.
        let mut qa = [[0u32; 4]; 5];
        let mut kk = 0usize;
        while kk < 5 {
            cuda_device::thread::__unroll_config::<0>();
            let row = wid * 16 + (lane % 16);
            // SAFETY: row < Q_ROWS and word kk·8 + (lane / 16)·4 + 4 <= 40 keep the 16-byte row
            // inside the query tile, published by the barrier above.
            let p = unsafe { qs.add(row * ROW_WORDS_80 + kk * 8 + (lane / 16) * 4) };
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
        // partial sum, and the output accumulators (10 n8 tiles of the head's 80 values).
        let mut m = [f32::NEG_INFINITY; 2];
        let mut l = [0.0f32; 2];
        let mut o = [[0.0f32; 4]; 10];

        let tiles = rows.div_ceil(KEY_TILE);
        let mut kb = 0usize;
        while kb < tiles {
            // The previous tile's fragment reads are done before this tile's staging writes.
            thread::sync_threads();
            let mut rk = [0u32; STAGE_PER_THREAD_80];
            let mut rv = [0u32; STAGE_PER_THREAD_80];
            let mut s = 0usize;
            while s < STAGE_PER_THREAD_80 {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (HEAD_DIM_80 / 2);
                let w = i - r * (HEAD_DIM_80 / 2);
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
            while s < STAGE_PER_THREAD_80 {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (HEAD_DIM_80 / 2);
                let w = i - r * (HEAD_DIM_80 / 2);
                // SAFETY: r < KEY_TILE and w < HEAD_DIM_80 / 2 < ROW_WORDS_80; one owner per
                // word.
                unsafe {
                    *ks.add(r * ROW_WORDS_80 + w) = rk[s];
                    *vs.add(r * ROW_WORDS_80 + w) = rv[s];
                }
                s += 1;
            }
            thread::sync_threads();

            // ---- S = Q·Kᵀ: 8 n8 tiles of keys, 5 k16 steps of the head.
            let mut sc = [[0.0f32; 4]; 8];
            let mut kk = 0usize;
            while kk < 5 {
                cuda_device::thread::__unroll_config::<0>();
                let mut nj = 0usize;
                while nj < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = nj * 16 + (lane % 8) + 8 * (lane / 16);
                    // SAFETY: key < KEY_TILE and word kk·8 + ((lane / 8) % 2)·4 + 4 <= 40 keep
                    // the row inside the key tile, published by the barrier above.
                    let p = unsafe { ks.add(key * ROW_WORDS_80 + kk * 8 + ((lane / 8) % 2) * 4) };
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
                    j += 1;
                }
                t += 1;
            }
            let mut t = 0usize;
            while t < 10 {
                cuda_device::thread::__unroll_config::<0>();
                let mut j = 0usize;
                while j < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    o[t][j] = mul_rn_f32(o[t][j], corr[j / 2]);
                    j += 1;
                }
                t += 1;
            }

            // ---- O += P·V with P = hi + lo. A fragment of key step kk2 is tiles 2·kk2 and
            // 2·kk2 + 1 of P: registers (0, 1), (2, 3) of each, packed low value first. Value
            // tiles 0..10 come in five ldmatrix pairs.
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
                while nd < 5 {
                    cuda_device::thread::__unroll_config::<0>();
                    let key = kk2 * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
                    // SAFETY: key < KEY_TILE and word nd·8 + (lane / 16)·4 + 4 <= 40 keep the row
                    // inside the value tile, published by the barrier above.
                    let p = unsafe { vs.add(key * ROW_WORDS_80 + nd * 8 + (lane / 16) * 4) };
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
                    }
                    // SAFETY: as above.
                    unsafe {
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
        while t < 10 {
            cuda_device::thread::__unroll_config::<0>();
            let mut j = 0usize;
            while j < 4 {
                cuda_device::thread::__unroll_config::<0>();
                let row = qb * Q_ROWS + wid * 16 + group + 8 * (j / 2);
                if row < rows {
                    let d = 8 * t + 2 * t4 + (j % 2);
                    let y = f32_to_bf16_rne(div_rn_f32(o[t][j], l[j / 2]));
                    // SAFETY: row < n and h·80 + d < n_heads·80 <= out_width (host-checked)
                    // keep the index below n·out_width <= out.len(); one owner lane per value.
                    unsafe {
                        *out.get_unchecked_mut(row * ow + h * HEAD_DIM_80 + d) = y;
                    }
                }
                j += 1;
            }
            t += 1;
        }
    }
}

/// The loaded attention module for heads of 80.
pub struct Attn80Kernels {
    module: attn_80_kernels::LoadedModule,
}

impl Attn80Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Attn80Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(attn_80_kernels, ctx)? };
        Ok(Attn80Kernels { module })
    }

    /// Enqueue the attention of heads of 80 values ([`AttnArgs`]). Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: AttnArgs<'_>) -> Result<(), GpuError> {
        let what = "Attn80Kernels::enqueue";
        let AttnArgs {
            qkv,
            layout: lay,
            n,
            scale,
            out_width,
            out,
        } = args;
        let span = lay.n_heads * HEAD_DIM_80;
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
            .prepare_vis_attn_h80(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.vis_attn_h80(
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
