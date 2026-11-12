//! Raon_mHC on the card: the four residual streams around every sub-layer,
//! mixed through the Helmert chart (llama.cpp `src/models/raon1.cpp`
//! `build_hc_pre` / `build_hc_post`, the HF `RaonMHC`).
//!
//! Layouts, every entry token-major: a token's streams are `4 · n` f32,
//! stream `i` at `i · n`, read flat the `K = 4n` values the selector takes.
//! A token's route is [`HC_OUT`] f32 — `pre[4]`, `post[4]`, then `comb[16]`
//! with `comb[j · 4 + i]` the weight of source stream `j` in stream `i`
//! (V4.1's HC_PRE layout, so the apply stages are the same rule).
//!
//! - [`raon_hc_kernels::raon_hc_mix`], one block a (token, piece of
//!   [`HC_PIECE`] values): the selector's [`HC_MIX`] dots of the widened f16
//!   rows `hc_*_fn` with the token's streams over the piece, then the
//!   piece's sum of squares, a warp a row: lane `l` sums its values
//!   `l + 32s` by `mul_add` from 0, `s` ascending, then
//!   `warp::reduce_sum_f32`. The pieces spread the selector over the card's
//!   SMs: one block walking every row is load-latency bound.
//! - [`raon_hc_kernels::raon_hc_route`], one block a token: each row's piece
//!   sums (lane `l` adds pieces `l + 32j` from 0, `j` ascending, then
//!   `warp::reduce_sum_f32`), then on thread 0
//!   `mix = dot / √(ss/K + eps)`, `pre = σ(α_pre·mix + b)`,
//!   `post = mag·σ(α_post·mix + b)`, the chart `e = H z Hᵀ` of the
//!   `(n−1)²` coordinates `z`, `ρ = 1 + max(0, max(−e/B))`, `φ = e/ρ + B`
//!   and `comb = (φ − I)·σ(η) + I`, with `B = (I + 11ᵀ/n)/2` and `H` the
//!   Helmert basis.
//! - [`raon_hc_kernels::raon_hc_route_folded`], the Nano file's folded
//!   parametrization of the same route, over the same piece sums: the α
//!   scales folded into the selector rows leave the affine plain, one
//!   sigmoid a pre and post pair, and the chart the conversion baked
//!   (`K = γ(H⊗H)` zero on the pre and post coordinates, `r = −1/(γB)`,
//!   `c0 = (1−γ)I + γB`) turns the coordinates straight into
//!   `e' = K·mixes = γe`, `ρ = 1 + max(e'·r)` (no clamp: every row of
//!   `H z Hᵀ` sums to zero, so the max is at least `−e/B` at `e = 0`) and
//!   `comb = e'/ρ + c0`.
//! - [`raon_hc_kernels::raon_hc_fold`]: the sub-layer's input `Σ_i pre_i ·
//!   s_i`, streams in order.
//! - [`raon_hc_kernels::raon_hc_post`]: stream `i` becomes `post_i · out +
//!   Σ_j comb[j][i] · s_j`, the branch first, then the sources in order (the
//!   reference's fused `dsv4_hc_post`).
//! - [`raon_hc_kernels::raon_hc_mean`]: the head's input, `((s_0 + s_1) +
//!   s_2) + s_3` then `· 1/4`.
//!
//! Every product and sum of the apply stages is rounded on its own, in the
//! reference's CPU ops' order. Raon-only today: V4.1's
//! `ds41_hc_fold` / `ds41_hc_post` / `ds41_hc_mean` are the same apply rule
//! in `bloomery-gpu-deepseek41`, which this crate cannot reach; lifting
//! them here makes the route the one per-model part.

use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, div_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Residual streams (`hyper_connection.count`).
pub const STREAMS: usize = 4;
/// Selector rows a site: `pre`, `post` and the chart's `(n−1)²` coordinates.
pub const HC_MIX: usize = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
/// Route values a token: `pre`, `post`, `comb`.
pub const HC_OUT: usize = 2 * STREAMS + STREAMS * STREAMS;
/// The folded chart a site of the Nano dialect carries: the `streams²`
/// Helmert mix rows over the whole mix vector, then the `streams²` rho
/// denominators and the `streams²` combination constants
/// (conversion/raon1_nano.py `_fold_mhc`).
pub const CHART_LEN: usize = HC_MIX * STREAMS * STREAMS + 2 * STREAMS * STREAMS;
const _: () = assert!(HC_MIX == 17 && HC_OUT == 24 && CHART_LEN == 304);
/// The most tokens one route launch takes (a block each).
pub const MAX_TOKENS: usize = 512;
/// Threads of a route block: eight warps.
const ROUTE_THREADS: u32 = 256;
const ROUTE_WARPS: usize = 8;
/// Values of `K` one `raon_hc_mix` block owns: eight a lane, 32 apart.
pub(crate) const HC_PIECE: usize = 256;
/// Partial sums a (token, piece): the [`HC_MIX`] selector dots, then the
/// squares.
const HC_PARTS: usize = HC_MIX + 1;
const _: () = assert!(HC_PIECE == 8 * 32 && HC_PARTS == 18 && HC_PARTS <= 3 * ROUTE_WARPS);
/// Threads of the element-wise entries.
const ELEM_THREADS: u32 = 256;

/// The route's partial sums a token of `n`-wide streams takes: [`HC_PARTS`]
/// a piece of [`HC_PIECE`] values. `None` when `4n` is not whole pieces.
pub fn parts_len(n: usize) -> Option<usize> {
    let k = STREAMS * n;
    (k > 0 && k.is_multiple_of(HC_PIECE)).then_some(k / HC_PIECE * HC_PARTS)
}

/// One lane's share of a piece's dot of `a` from `ia` with `b` from `ib`:
/// `a[ia + 32s] · b[ib + 32s]`, `s` = 0..8 ascending, by `mul_add` from 0,
/// every load issued before the first product.
///
/// SAFETY: `ia + 224 < a.len()` and `ib + 224 < b.len()`.
#[inline(always)]
unsafe fn lane_piece_dot(a: &[f32], ia: usize, b: &[f32], ib: usize) -> f32 {
    // SAFETY: every offset is at most `ia + 224` or `ib + 224`, in bounds by
    // this fn's contract.
    let (va, vb) = unsafe {
        (
            [
                *a.get_unchecked(ia),
                *a.get_unchecked(ia + 32),
                *a.get_unchecked(ia + 64),
                *a.get_unchecked(ia + 96),
                *a.get_unchecked(ia + 128),
                *a.get_unchecked(ia + 160),
                *a.get_unchecked(ia + 192),
                *a.get_unchecked(ia + 224),
            ],
            [
                *b.get_unchecked(ib),
                *b.get_unchecked(ib + 32),
                *b.get_unchecked(ib + 64),
                *b.get_unchecked(ib + 96),
                *b.get_unchecked(ib + 128),
                *b.get_unchecked(ib + 160),
                *b.get_unchecked(ib + 192),
                *b.get_unchecked(ib + 224),
            ],
        )
    };
    let mut acc = va[0].mul_add(vb[0], 0.0);
    acc = va[1].mul_add(vb[1], acc);
    acc = va[2].mul_add(vb[2], acc);
    acc = va[3].mul_add(vb[3], acc);
    acc = va[4].mul_add(vb[4], acc);
    acc = va[5].mul_add(vb[5], acc);
    acc = va[6].mul_add(vb[6], acc);
    va[7].mul_add(vb[7], acc)
}

/// Helmert basis entry `(i, j)` of `n` streams: `1` for `i ≤ j`, `−(j+1)`
/// for `i = j + 1`, else 0, over `√((j+1)(j+2))` — the reference's f32
/// values (`raon1.cpp`'s graph constructor).
#[inline(always)]
fn helmert(i: usize, j: usize) -> f32 {
    let num = if i <= j {
        1.0
    } else if i == j + 1 {
        -((j + 1) as f32)
    } else {
        0.0
    };
    div_rn_f32(num, sqrt_rn_f32(((j + 1) * (j + 2)) as f32))
}

/// `1 / (1 + e^(0 − x))`, ggml's sigmoid with ik's exponential.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    crate::linear::sigmoid(x)
}

#[allow(
    dead_code,
    reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
)]
#[cuda_module]
mod raon_hc_kernels {
    use super::*;

    /// The route's partial sums (module doc), block `t·pieces + p` token
    /// `t`'s piece `p` of [`HC_PIECE`] values: `w` the [`HC_MIX`] selector
    /// rows of `4n` f32, `x` the tokens' streams. Warp `w` owns rows `w`,
    /// `w + 8`, `w + 16` of the [`HC_PARTS`]: the selector rows' dots, then
    /// the squares; writes [`HC_PARTS`] values a (token, piece) into `part`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            pieces * 256 == 4 * n,
            w.len() >= 68 * n,
            x.len() >= m * 4 * n,
            part.len() >= m * pieces * 18
        )
    )]
    pub fn raon_hc_mix(
        w: &[f32],
        x: &[f32],
        n: u32,
        m: u32,
        pieces: u32,
        mut part: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let pieces = pieces as usize;
        let (t, p) = (b / pieces, b % pieces);
        if t >= m as usize {
            return;
        }
        let wi = thread::threadIdx_x() as usize / 32;
        let lane = warp::lane_id() as usize;
        let k = STREAMS * n as usize;
        let at = p * HC_PIECE + lane;
        let xt = t * k + at;
        let mut r = wi;
        while r < HC_PARTS {
            // SAFETY: at + 224 < (p + 1)·HC_PIECE <= k, so for r < HC_MIX
            // r·k + at + 224 < 17·k <= w.len(), and xt + 224 < (t + 1)·k <=
            // x.len(), by the contract.
            let acc = unsafe {
                if r < HC_MIX {
                    lane_piece_dot(w, r * k + at, x, xt)
                } else {
                    lane_piece_dot(x, xt, x, xt)
                }
            };
            let acc = warp::reduce_sum_f32(acc);
            if lane == 0 {
                // SAFETY: (t·pieces + p)·HC_PARTS + r < m·pieces·18 <=
                // part.len() by the contract; this warp alone owns row r.
                unsafe { *part.get_unchecked_mut((t * pieces + p) * HC_PARTS + r) = acc };
            }
            r += ROUTE_WARPS;
        }
    }

    /// One token's route (module doc), block `t` token `t`: `part` the
    /// tokens' [`raon_hc_mix`] sums over `pieces` pieces, `scale` the four
    /// scales (`α_pre`, `α_post`, `α_z`, `η`), `base` the [`HC_MIX`] biases;
    /// writes [`HC_OUT`] values a token into `hc`.
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
            pieces * 256 == 4 * n,
            part.len() >= m * pieces * 18,
            scale.len() >= 4,
            base.len() >= 17,
            hc.len() >= m * 24
        )
    )]
    pub fn raon_hc_route(
        part: &[f32],
        scale: &[f32],
        base: &[f32],
        n: u32,
        m: u32,
        pieces: u32,
        eps: f32,
        magnitude: f32,
        mut hc: DisjointSlice<f32>,
    ) {
        static mut MIX: SharedArray<f32, HC_PARTS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let k = STREAMS * n as usize;
        let pieces = pieces as usize;
        let row0 = t * pieces * HC_PARTS;
        // SAFETY: MIX is this block's own shared allocation; the raw form
        // reaches the `static mut` without a reference.
        let mix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MIX) };
        let mut r = wi;
        while r < HC_PARTS {
            let mut acc = 0.0f32;
            let mut j = lane;
            while j < pieces {
                // SAFETY: row0 + j·HC_PARTS + r < (t + 1)·pieces·18 <=
                // part.len() by the contract.
                acc = add_rn_f32(acc, unsafe { *part.get_unchecked(row0 + j * HC_PARTS + r) });
                j += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if lane == 0 {
                // SAFETY: r < HC_PARTS; this warp alone owns row r.
                unsafe { *mix.add(r) = acc };
            }
            r += ROUTE_WARPS;
        }
        thread::sync_threads();
        if tid != 0 {
            return;
        }
        // SAFETY: HC_MIX < HC_PARTS, written above, past the barrier.
        let sum = unsafe { *mix.add(HC_MIX) };
        let rms = sqrt_rn_f32(add_rn_f32(div_rn_f32(sum, k as f32), eps));
        // SAFETY: scale.len() >= 4 and base.len() >= HC_MIX by the contract.
        let (a_pre, a_post, a_z, eta) = unsafe {
            (
                *scale.get_unchecked(0),
                *scale.get_unchecked(1),
                *scale.get_unchecked(2),
                *scale.get_unchecked(3),
            )
        };
        let affine = |r: usize, a: f32| -> f32 {
            // SAFETY: r < HC_MIX, written above, past the barrier; base.len()
            // >= HC_MIX by the contract.
            let (v, b) = unsafe { (*mix.add(r), *base.get_unchecked(r)) };
            add_rn_f32(mul_rn_f32(div_rn_f32(v, rms), a), b)
        };
        let o = t * HC_OUT;
        let mut i = 0usize;
        while i < STREAMS {
            let pre = sigmoid(affine(i, a_pre));
            let post = mul_rn_f32(sigmoid(affine(STREAMS + i, a_post)), magnitude);
            // SAFETY: o + 8 <= m·HC_OUT <= hc.len() by the contract; thread 0
            // of block t alone writes token t's values.
            unsafe {
                *hc.get_unchecked_mut(o + i) = pre;
                *hc.get_unchecked_mut(o + STREAMS + i) = post;
            }
            i += 1;
        }
        const NM: usize = STREAMS - 1;
        let mut z = [[0.0f32; NM]; NM];
        let mut a = 0usize;
        while a < NM {
            let mut b = 0usize;
            while b < NM {
                z[a][b] = affine(2 * STREAMS + a * NM + b, a_z);
                b += 1;
            }
            a += 1;
        }
        // hz[i][a] = Σ_b H[i][b] · z[a][b], then e[i][j] = Σ_a H[j][a] ·
        // hz[i][a]: the reference's two products in their order.
        let mut hz = [[0.0f32; NM]; STREAMS];
        let mut i = 0usize;
        while i < STREAMS {
            let mut a = 0usize;
            while a < NM {
                let mut acc = 0.0f32;
                let mut b = 0usize;
                while b < NM {
                    acc = add_rn_f32(acc, mul_rn_f32(helmert(i, b), z[a][b]));
                    b += 1;
                }
                hz[i][a] = acc;
                a += 1;
            }
            i += 1;
        }
        let mut e = [[0.0f32; STREAMS]; STREAMS];
        let mut worst = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < STREAMS {
            let mut j = 0usize;
            while j < STREAMS {
                let mut acc = 0.0f32;
                let mut a = 0usize;
                while a < NM {
                    acc = add_rn_f32(acc, mul_rn_f32(helmert(j, a), hz[i][a]));
                    a += 1;
                }
                e[i][j] = acc;
                let chart = chart_base(i, j);
                let ratio = 0.0 - div_rn_f32(acc, chart);
                if ratio > worst {
                    worst = ratio;
                }
                j += 1;
            }
            i += 1;
        }
        let rho = add_rn_f32(if worst > 0.0 { worst } else { 0.0 }, 1.0);
        let gamma = sigmoid(eta);
        let mut i = 0usize;
        while i < STREAMS {
            let mut j = 0usize;
            while j < STREAMS {
                let id = if i == j { 1.0 } else { 0.0 };
                let phi = add_rn_f32(div_rn_f32(e[i][j], rho), chart_base(i, j));
                let comb = add_rn_f32(mul_rn_f32(add_rn_f32(phi, 0.0 - id), gamma), id);
                // The chart's (i, j) is source i's weight in stream j: the
                // reference's `comb` at (dst j, src i), stored at src·4 + dst.
                // SAFETY: o + 2·STREAMS + i·STREAMS + j < (t + 1)·HC_OUT <=
                // hc.len() by the contract; thread 0 of block t alone writes
                // token t's values.
                unsafe { *hc.get_unchecked_mut(o + 2 * STREAMS + i * STREAMS + j) = comb };
                j += 1;
            }
            i += 1;
        }
    }

    /// One token's route over the Nano file's folded chart (module doc),
    /// block `t` token `t`: `part` the tokens' [`raon_hc_mix`] sums over
    /// `pieces` pieces, `base` the [`HC_MIX`] biases (the α scales are folded
    /// into the selector rows), `chart` the [`CHART_LEN`] values the
    /// conversion baked — the `streams²·HC_MIX` Helmert mix `K` zero on the
    /// pre and post coordinates, the `streams²` rho denominators and the
    /// `streams²` combination constants (conversion/raon1_nano.py
    /// `_fold_mhc`); writes [`HC_OUT`] values a token into `hc`.
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
            pieces * 256 == 4 * n,
            part.len() >= m * pieces * 18,
            base.len() >= 17,
            chart.len() >= 304,
            hc.len() >= m * 24
        )
    )]
    pub fn raon_hc_route_folded(
        part: &[f32],
        base: &[f32],
        chart: &[f32],
        n: u32,
        m: u32,
        pieces: u32,
        eps: f32,
        magnitude: f32,
        mut hc: DisjointSlice<f32>,
    ) {
        static mut MIX: SharedArray<f32, HC_PARTS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let wi = tid / 32;
        let lane = warp::lane_id() as usize;
        let k = STREAMS * n as usize;
        let pieces = pieces as usize;
        let row0 = t * pieces * HC_PARTS;
        // SAFETY: MIX is this block's own shared allocation; the raw form
        // reaches the `static mut` without a reference.
        let mix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MIX) };
        let mut r = wi;
        while r < HC_PARTS {
            let mut acc = 0.0f32;
            let mut j = lane;
            while j < pieces {
                // SAFETY: row0 + j·HC_PARTS + r < (t + 1)·pieces·18 <=
                // part.len() by the contract.
                acc = add_rn_f32(acc, unsafe { *part.get_unchecked(row0 + j * HC_PARTS + r) });
                j += 32;
            }
            let acc = warp::reduce_sum_f32(acc);
            if lane == 0 {
                // SAFETY: r < HC_PARTS; this warp alone owns row r.
                unsafe { *mix.add(r) = acc };
            }
            r += ROUTE_WARPS;
        }
        thread::sync_threads();
        if tid != 0 {
            return;
        }
        // SAFETY: HC_MIX < HC_PARTS, written above, past the barrier.
        let sum = unsafe { *mix.add(HC_MIX) };
        let rms = sqrt_rn_f32(add_rn_f32(div_rn_f32(sum, k as f32), eps));
        // The α scales are folded into the selector rows the mix summed, so
        // the affine is the plain mix over the rms plus its bias.
        let affine = |r: usize| -> f32 {
            // SAFETY: r < HC_MIX, written above, past the barrier; base.len()
            // >= HC_MIX by the contract.
            let (v, b) = unsafe { (*mix.add(r), *base.get_unchecked(r)) };
            add_rn_f32(div_rn_f32(v, rms), b)
        };
        let o = t * HC_OUT;
        let mut i = 0usize;
        while i < STREAMS {
            let pre = sigmoid(affine(i));
            let post = mul_rn_f32(sigmoid(affine(STREAMS + i)), magnitude);
            // SAFETY: o + 8 <= m·HC_OUT <= hc.len() by the contract; thread 0
            // of block t alone writes token t's values.
            unsafe {
                *hc.get_unchecked_mut(o + i) = pre;
                *hc.get_unchecked_mut(o + STREAMS + i) = post;
            }
            i += 1;
        }
        // e'[j] = Σ_m K[j][m] · affine(m): K is zero on the pre and post
        // coordinates, so those terms are ±0.0 and the sum is the nine chart
        // coordinates' alone, each product and sum rounded in `m` order.
        let mut e = [0.0f32; STREAMS * STREAMS];
        let mut j = 0usize;
        while j < STREAMS * STREAMS {
            let mut acc = 0.0f32;
            let mut mid = 0usize;
            while mid < HC_MIX {
                // SAFETY: j·HC_MIX + mid < 272 and mid < 17: both inside
                // chart.len() by the contract.
                let km = unsafe { *chart.get_unchecked(j * HC_MIX + mid) };
                acc = add_rn_f32(acc, mul_rn_f32(km, affine(mid)));
                mid += 1;
            }
            e[j] = acc;
            j += 1;
        }
        // ρ = 1 + max_j e'[j]·r[j], no clamp: every row of H z Hᵀ sums to
        // zero, so the max is at least the anchor's own −e ratio
        // (raon1nano.cpp's comment on the folded layout).
        let k_at = STREAMS * STREAMS;
        let mut worst = f32::NEG_INFINITY;
        let mut j = 0usize;
        while j < k_at {
            // SAFETY: k_at + j < 288 <= chart.len() by the contract.
            let ratio = mul_rn_f32(e[j], unsafe { *chart.get_unchecked(HC_MIX * k_at + j) });
            if ratio > worst {
                worst = ratio;
            }
            j += 1;
        }
        let rho = add_rn_f32(worst, 1.0);
        let mut j = 0usize;
        while j < STREAMS * STREAMS {
            // SAFETY: (HC_MIX + 1)·k_at + j < 304 <= chart.len() and
            // o + 8 + j < (t + 1)·HC_OUT <= hc.len() by the contract; thread 0
            // of block t alone writes token t's values. The chart's entry `j`
            // is source `j / streams`'s weight in stream `j % streams`, the
            // layout the apply stages read (module doc).
            unsafe {
                let c0 = *chart.get_unchecked((HC_MIX + 1) * k_at + j);
                *hc.get_unchecked_mut(o + 2 * STREAMS + j) = add_rn_f32(div_rn_f32(e[j], rho), c0);
            }
            j += 1;
        }
    }

    /// `y[t·n + d] = Σ_i pre_i · s_i[d]` of token `t`, `i` ascending, each
    /// product and sum rounded: one thread a (token, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * 4 * n,
            hc.len() >= m * 24,
            y.len() >= m * n
        )
    )]
    pub fn raon_hc_fold(x: &[f32], hc: &[f32], n: u32, m: u32, mut y: DisjointSlice<f32>) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * n {
            return;
        }
        let (t, d) = (g / n, g % n);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < STREAMS {
            // SAFETY: t < m, i < 4 and d < n bound both reads by the contract.
            let (s, p) = unsafe {
                (
                    *x.get_unchecked(t * STREAMS * n + i * n + d),
                    *hc.get_unchecked(t * HC_OUT + i),
                )
            };
            let term = mul_rn_f32(s, p);
            acc = if i == 0 { term } else { add_rn_f32(acc, term) };
            i += 1;
        }
        // SAFETY: g < m·n <= y.len(); thread g alone writes it.
        unsafe { *y.get_unchecked_mut(g) = acc };
    }

    /// `next[t·4n + i·n + d] = post_i · out[t·n + d] + Σ_j comb[j][i] ·
    /// res_j[d]`, the branch then the sources ascending, each product and
    /// sum rounded: one thread a (token, stream, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            res.len() >= m * 4 * n,
            out.len() >= m * n,
            hc.len() >= m * 24,
            next.len() >= m * 4 * n
        )
    )]
    pub fn raon_hc_post(
        res: &[f32],
        out: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut next: DisjointSlice<f32>,
    ) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * STREAMS * n {
            return;
        }
        let (t, r) = (g / (STREAMS * n), g % (STREAMS * n));
        let (i, d) = (r / n, r % n);
        let base = t * STREAMS * n;
        let o = t * HC_OUT;
        // SAFETY: t < m, d < n and i < 4 bound both reads by the contract.
        let (b, p) = unsafe {
            (
                *out.get_unchecked(t * n + d),
                *hc.get_unchecked(o + STREAMS + i),
            )
        };
        let mut acc = mul_rn_f32(b, p);
        let mut j = 0usize;
        while j < STREAMS {
            // SAFETY: as above, and j < 4.
            let (s, c) = unsafe {
                (
                    *res.get_unchecked(base + j * n + d),
                    *hc.get_unchecked(o + 2 * STREAMS + j * STREAMS + i),
                )
            };
            acc = add_rn_f32(acc, mul_rn_f32(s, c));
            j += 1;
        }
        // SAFETY: g < m·4n <= next.len(); thread g alone writes it.
        unsafe { *next.get_unchecked_mut(g) = acc };
    }

    /// `y[t·n + d] = (((s_0 + s_1) + s_2) + s_3) · 1/4` of token `t`: one
    /// thread a (token, value).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            x.len() >= m * 4 * n,
            y.len() >= m * n
        )
    )]
    pub fn raon_hc_mean(x: &[f32], n: u32, m: u32, mut y: DisjointSlice<f32>) {
        let g = thread::index_1d().get();
        let n = n as usize;
        if g >= m as usize * n {
            return;
        }
        let (t, d) = (g / n, g % n);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < STREAMS {
            // SAFETY: t < m, i < 4 and d < n bound the read by the contract.
            let s = unsafe { *x.get_unchecked(t * STREAMS * n + i * n + d) };
            acc = if i == 0 { s } else { add_rn_f32(acc, s) };
            i += 1;
        }
        // SAFETY: g < m·n <= y.len(); thread g alone writes it.
        unsafe { *y.get_unchecked_mut(g) = mul_rn_f32(acc, 0.25) };
    }
}

/// `B = (I + 11ᵀ/n)/2` at `(i, j)`: the reference's `scale_bias(I, 0.5,
/// 0.5/n)`.
#[inline(always)]
fn chart_base(i: usize, j: usize) -> f32 {
    let id = if i == j { 1.0 } else { 0.0 };
    add_rn_f32(mul_rn_f32(id, 0.5), 0.5 / STREAMS as f32)
}

/// One site's resident pieces: the widened selector rows, the scales and
/// the biases.
pub struct SiteVecs<'a> {
    pub(crate) w: &'a DeviceBuffer<f32>,
    pub(crate) scale: &'a DeviceBuffer<f32>,
    pub(crate) base: &'a DeviceBuffer<f32>,
}

impl<'a> SiteVecs<'a> {
    /// The split dialect's site pieces.
    #[must_use]
    pub fn new(
        (w, scale, base): (
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<f32>,
        ),
    ) -> SiteVecs<'a> {
        SiteVecs { w, scale, base }
    }
}

/// One site's resident pieces of the Nano dialect's folded layout: the
/// selector rows (the α scales folded into them), the biases and the baked
/// chart ([`CHART_LEN`] values, all of them widened to f32 at load).
pub struct ChartVecs<'a> {
    pub(crate) w: &'a DeviceBuffer<f32>,
    pub(crate) base: &'a DeviceBuffer<f32>,
    pub(crate) chart: &'a DeviceBuffer<f32>,
}

impl<'a> ChartVecs<'a> {
    /// The folded dialect's site pieces.
    #[must_use]
    pub fn new(
        (w, base, chart): (
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<f32>,
            &'a DeviceBuffer<f32>,
        ),
    ) -> ChartVecs<'a> {
        ChartVecs { w, base, chart }
    }
}

/// The loaded module.
pub struct MhcKernels {
    module: raon_hc_kernels::LoadedModule,
}

impl MhcKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MhcKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(raon_hc_kernels, ctx)? };
        Ok(MhcKernels { module })
    }

    /// The route of `m` tokens' streams `x` (`4n` a token) at site `v` into
    /// `hc` ([`HC_OUT`] a token), through the partial sums `part`
    /// ([`parts_len`] a token): two launches. A `4n` that is not whole
    /// [`HC_PIECE`] pieces is refused by name. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the site's three vectors, the streams, the widths and the file's two constants"
    )]
    pub fn enqueue_route(
        &self,
        stream: &CudaStream,
        v: &SiteVecs<'_>,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        (eps, magnitude): (f32, f32),
        (part, hc): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_route";
        let k = STREAMS * n;
        let Some(per) = parts_len(n) else {
            return Err(GpuError::shape(
                what,
                format!("n={n}: the streams' {k} values are not whole {HC_PIECE}-value pieces"),
            ));
        };
        if m == 0
            || m > MAX_TOKENS
            || v.w.len() < HC_MIX * k
            || v.scale.len() < 4
            || v.base.len() < HC_MIX
            || x.len() < m * k
            || part.len() < m * per
            || hc.len() < m * HC_OUT
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m} (1..={MAX_TOKENS}), n={n}: w {} (want {}), scale {}, base {}, x {} \
                     (want {}), part {} (want {}), hc {} (want {})",
                    v.w.len(),
                    HC_MIX * k,
                    v.scale.len(),
                    v.base.len(),
                    x.len(),
                    m * k,
                    part.len(),
                    m * per,
                    hc.len(),
                    m * HC_OUT
                ),
            ));
        }
        let mu = launch_u32(what, "m", m)?;
        let pieces = launch_u32(what, "pieces", k / HC_PIECE)?;
        let nu = launch_u32(what, "n", n)?;
        let blocks = launch_u32(what, "m·pieces", m * (k / HC_PIECE))?;
        let prep =
            self.module
                .prepare_raon_hc_mix(LaunchConfig1D::new(blocks, ROUTE_THREADS, 0))?;
        self.module
            .raon_hc_mix(stream, &prep, v.w, x, nu, mu, pieces, &mut *part)?;
        let prep = self
            .module
            .prepare_raon_hc_route(LaunchConfig1D::new(mu, ROUTE_THREADS, 0))?;
        self.module.raon_hc_route(
            stream, &prep, &*part, v.scale, v.base, nu, mu, pieces, eps, magnitude, hc,
        )?;
        Ok(())
    }

    /// The route of `m` tokens' streams `x` at site `v` of the Nano
    /// dialect's folded layout ([`raon_hc_route_folded`], module doc): the
    /// same piece sums as [`MhcKernels::enqueue_route`] — one
    /// [`raon_hc_mix`] launch — then the folded chart's route over them.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the site's three vectors, the streams, the widths and the file's two constants"
    )]
    pub fn enqueue_route_folded(
        &self,
        stream: &CudaStream,
        v: &ChartVecs<'_>,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        (eps, magnitude): (f32, f32),
        (part, hc): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_route_folded";
        let k = STREAMS * n;
        let Some(per) = parts_len(n) else {
            return Err(GpuError::shape(
                what,
                format!("n={n}: the streams' {k} values are not whole {HC_PIECE}-value pieces"),
            ));
        };
        if m == 0
            || m > MAX_TOKENS
            || v.w.len() < HC_MIX * k
            || v.base.len() < HC_MIX
            || v.chart.len() < CHART_LEN
            || x.len() < m * k
            || part.len() < m * per
            || hc.len() < m * HC_OUT
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m} (1..={MAX_TOKENS}), n={n}: w {} (want {}), base {}, chart {} (want \
                     {}), x {} (want {}), part {} (want {}), hc {} (want {})",
                    v.w.len(),
                    HC_MIX * k,
                    v.base.len(),
                    v.chart.len(),
                    CHART_LEN,
                    x.len(),
                    m * k,
                    part.len(),
                    m * per,
                    hc.len(),
                    m * HC_OUT
                ),
            ));
        }
        let mu = launch_u32(what, "m", m)?;
        let pieces = launch_u32(what, "pieces", k / HC_PIECE)?;
        let nu = launch_u32(what, "n", n)?;
        let blocks = launch_u32(what, "m·pieces", m * (k / HC_PIECE))?;
        let prep =
            self.module
                .prepare_raon_hc_mix(LaunchConfig1D::new(blocks, ROUTE_THREADS, 0))?;
        self.module
            .raon_hc_mix(stream, &prep, v.w, x, nu, mu, pieces, &mut *part)?;
        let prep = self
            .module
            .prepare_raon_hc_route_folded(LaunchConfig1D::new(mu, ROUTE_THREADS, 0))?;
        self.module.raon_hc_route_folded(
            stream, &prep, &*part, v.base, v.chart, nu, mu, pieces, eps, magnitude, hc,
        )?;
        Ok(())
    }

    /// The fold of `m` tokens' streams `x` by their routes `hc` into `y`
    /// (`n` a token). Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_fold(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        hc: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_fold";
        if m == 0 || x.len() < m * STREAMS * n || hc.len() < m * HC_OUT || y.len() < m * n {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, n={n}: x {}, hc {}, y {}",
                    x.len(),
                    hc.len(),
                    y.len()
                ),
            ));
        }
        let total = launch_u32(what, "m·n", m * n)?;
        let prep = self.module.prepare_raon_hc_fold(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_fold(
            stream,
            &prep,
            x,
            hc,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            y,
        )?;
        Ok(())
    }

    /// The streams after a sub-layer: `res` mixed by the routes `hc` plus
    /// the branch `out`, into `next`. Asynchronous, allocation-free,
    /// capturable.
    pub(crate) fn enqueue_post(
        &self,
        stream: &CudaStream,
        (res, out, hc): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (n, m): (usize, usize),
        next: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_post";
        let k = STREAMS * n;
        if m == 0
            || res.len() < m * k
            || out.len() < m * n
            || hc.len() < m * HC_OUT
            || next.len() < m * k
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, n={n}: res {}, out {}, hc {}, next {}",
                    res.len(),
                    out.len(),
                    hc.len(),
                    next.len()
                ),
            ));
        }
        let total = launch_u32(what, "m·4n", m * k)?;
        let prep = self.module.prepare_raon_hc_post(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_post(
            stream,
            &prep,
            res,
            out,
            hc,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            next,
        )?;
        Ok(())
    }

    /// The mean of `m` tokens' streams `x` into `y` (`n` a token).
    /// Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_mean(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 mhc::enqueue_mean";
        if m == 0 || x.len() < m * STREAMS * n || y.len() < m * n {
            return Err(GpuError::shape(
                what,
                format!("m={m}, n={n}: x {}, y {}", x.len(), y.len()),
            ));
        }
        let total = launch_u32(what, "m·n", m * n)?;
        let prep = self.module.prepare_raon_hc_mean(LaunchConfig1D::new(
            total.div_ceil(ELEM_THREADS),
            ELEM_THREADS,
            0,
        ))?;
        self.module.raon_hc_mean(
            stream,
            &prep,
            x,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            y,
        )?;
        Ok(())
    }
}
