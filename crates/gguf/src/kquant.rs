//! Q4_K, Q5_K, Q6_K and Q8_0 quantization on the host: ggml's `quantize_row_q4_K_ref` / `_impl` and the same for
//! Q5_K and Q6_K, and `quantize_row_q8_0_ref` (`ggml/src/ggml-quants.c`), step for step, so the bytes are the
//! bytes `llama-quantize` writes.
//!
//! `ggml_quantize_chunk` takes the `_impl` rule (the importance-weighted search) when it is handed `quant_weights`
//! and the `_ref` rule when it is not, and Q8_0 has one rule and ignores them; [`quantize_row`] makes the same choice
//! from its `qw` argument. Every multiply, add, subtract, divide and square root is its own f32 operation rounded
//! once to nearest even in C's left-to-right order: the reference build is baseline x86-64 (no FMA), and Rust's
//! operators never contract, so no product is fused into a sum anywhere here. `MAX` and `MIN` are ggml's
//! (`a > b ? a : b`, `a < b ? a : b`), the searches keep ggml's candidate order and its strict tie-breaks, and an
//! integer stored into a byte wraps as C's conversion does.
//!
//! A row with a value that is not finite is refused by name rather than quantized: ggml's result for it is a defined
//! but meaningless byte pattern.

use crate::iq_quant::{Ieee, nearest_int};
use crate::quant::{GgmlType, f32_to_f16_bits, half_to_f32};

/// Values per super-block (ggml's `QK_K`).
const QK: usize = 256;
/// `GROUP_MAX_EPS`.
const GROUP_MAX_EPS: f32 = 1e-15;

/// The types this module writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KType {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl KType {
    /// The type a ggml type id names, when this module writes it.
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<KType> {
        match ty {
            GgmlType::Q4_K => Some(KType::Q4K),
            GgmlType::Q5_K => Some(KType::Q5K),
            GgmlType::Q6_K => Some(KType::Q6K),
            GgmlType::Q8_0 => Some(KType::Q8_0),
            _ => None,
        }
    }

    /// The ggml type it writes.
    #[must_use]
    pub fn ggml(self) -> GgmlType {
        match self {
            KType::Q4K => GgmlType::Q4_K,
            KType::Q5K => GgmlType::Q5_K,
            KType::Q6K => GgmlType::Q6_K,
            KType::Q8_0 => GgmlType::Q8_0,
        }
    }

    /// Values per block: 256 for the K types, 32 for Q8_0.
    #[must_use]
    pub fn block_values(self) -> usize {
        match self {
            KType::Q8_0 => 32,
            _ => QK,
        }
    }

    /// Bytes per block.
    #[must_use]
    pub fn block_bytes(self) -> usize {
        match self {
            KType::Q4K => 144,
            KType::Q5K => 176,
            KType::Q6K => 210,
            KType::Q8_0 => 34,
        }
    }
}

/// Why a row is not quantized.
#[derive(Debug, thiserror::Error)]
pub enum KQuantError {
    #[error("a row of {n} values is not a whole number of {block}-value blocks")]
    RowLength { n: usize, block: usize },
    #[error("{got} importance values for a row of {n}")]
    Weights { got: usize, n: usize },
    #[error("value {index} is {value}, not finite")]
    NotFinite { index: usize, value: f32 },
    #[error("importance value {index} is {value}, not finite")]
    WeightNotFinite { index: usize, value: f32 },
}

#[inline(always)]
fn nint(x: f32) -> i32 {
    nearest_int::<Ieee>(x)
}

#[inline(always)]
fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
fn imin(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `sum += x[l] * x[l]` over `x`, in order.
fn sum_sq(x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    s
}

/// ggml's `make_qx_quants` at `rmse_type == 1` (Q6_K's), with `qw` optional: the scale of `n` values at `nmax`
/// levels, the levels (offset by `nmax`) in `l`.
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8], qw: Option<&[f32]>) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        l.fill(0);
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = -nmaxf / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let li = imax(-nmax, imin(nmax - 1, nint(iscale * x[i])));
        l[i] = (li + nmax) as i8;
        let w = qw.map_or_else(|| x[i] * x[i], |q| q[i]);
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmaxf + 0.1f32 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = imax(-nmax, imin(nmax - 1, nint(iscale * x[i])));
            let w = qw.map_or_else(|| x[i] * x[i], |q| q[i]);
            sumlx += w * x[i] * li as f32;
            suml2 += w * li as f32 * li as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nint(iscale * x[i]);
                l[i] = (nmax + imax(-nmax, imin(nmax - 1, li))) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml's `make_qkx2_quants` (the `_ref` rules' 32-value search, `use_mad == false`): the scale, the minimum's
/// magnitude and the levels; the last argument is ggml's `(rmin, rdelta, nstep)`.
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32; 32],
    weights: &[f32; 32],
    l: &mut [u8],
    laux: &mut [u8; 32],
    (rmin, rdelta, nstep): (f32, f32, i32),
) -> (f32, f32) {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..32 {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / (max - min);
    let mut scale = 1.0f32 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..32 {
        let li = nint(iscale * (x[i] - min));
        l[i] = imax(0, imin(nmax, li)) as u8;
        let diff = scale * f32::from(l[i]) + min - x[i];
        let diff = diff * diff;
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmaxf) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..32 {
            let li = imax(0, imin(nmax, nint(iscale * (x[i] - min))));
            laux[i] = li as u8;
            let w = weights[i];
            let lf = li as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..32 {
                let diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                let diff = diff * diff;
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l[..32].copy_from_slice(laux);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml's `make_qkx3_quants` (the `_impl` rules' 32-value search, `use_mad == false`, weights always given).
fn make_qkx3_quants(
    nmax: i32,
    x: &[f32; 32],
    weights: &[f32; 32],
    l: &mut [u8],
    laux: &mut [u8; 32],
    (rmin, rdelta, nstep): (f32, f32, i32),
) -> (f32, f32) {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..32 {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        l.fill(0);
        return (0.0, -min);
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / (max - min);
    let mut scale = 1.0f32 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..32 {
        let li = nint(iscale * (x[i] - min));
        l[i] = imax(0, imin(nmax, li)) as u8;
        let diff = scale * f32::from(l[i]) + min - x[i];
        let diff = diff * diff;
        best_mad += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmaxf) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..32 {
            let v = iscale * (x[i] - min);
            // `v > 0 ? nearest_int(MIN(v, nmax)) : 0`, MIN as ggml's macro on a float and an int
            let li = if v > 0.0 {
                nint(if v < nmaxf { v } else { nmaxf })
            } else {
                0
            };
            laux[i] = li as u8;
            let w = weights[i];
            let lf = li as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..32 {
                let diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                let diff = diff * diff;
                mad += weights[i] * diff;
            }
            if mad < best_mad {
                l[..32].copy_from_slice(laux);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml's `make_qp_quants`: the scale of the eight non-negative `x` at `nmax` levels weighted by `qw`.
fn make_qp_quants(nmax: i32, x: &[f32; 8], l: &mut [u8; 8], qw: &[f32; 8]) -> f32 {
    let mut max = 0.0f32;
    for &v in x {
        max = fmax(max, v);
    }
    if max < GROUP_MAX_EPS {
        l.fill(0);
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut iscale = nmaxf / max;
    for i in 0..8 {
        l[i] = nint(iscale * x[i]) as u8;
    }
    let scale = 1.0f32 / iscale;
    let mut best_mse = 0.0f32;
    for i in 0..8 {
        let diff = x[i] - scale * f32::from(l[i]);
        best_mse += qw[i] * diff * diff;
    }
    for is in -4i32..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1f32 * is as f32 + nmaxf) / max;
        let scale_is = 1.0f32 / iscale_is;
        let mut mse = 0.0f32;
        for i in 0..8 {
            let li = imin(nmax, nint(iscale_is * x[i]));
            let diff = x[i] - scale_is * li as f32;
            mse += qw[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..8 {
        let li = imin(nmax, nint(iscale * x[i]));
        l[i] = li as u8;
        let w = qw[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..8 {
            let w = qw[i];
            let mut slx = sumlx - w * x[i] * f32::from(l[i]);
            let mut sl2 = suml2 - w * f32::from(l[i]) * f32::from(l[i]);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = imin(nmax, nint(x[i] * sl2 / slx));
                if new_l != i32::from(l[i]) {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

/// ggml's `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// The 6-bit scale and minimum codes of the eight 32-value sub-blocks packed as `block_q4_K.scales`.
fn pack_scales(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            s[j] = ls[j];
            s[j + 4] = lm[j];
        } else {
            s[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
            s[j - 4] |= (ls[j] >> 4) << 6;
            s[j] |= (lm[j] >> 4) << 6;
        }
    }
    s
}

/// The levels of a Q4_K / Q5_K block once its `d`, `dmin` and scale codes are fixed: ggml's final loop. `l` keeps
/// the search's levels for a sub-block whose `d` is zero.
fn requant_levels(x: &[f32], scales: &[u8; 12], d: f32, dmin: f32, top: i32, l: &mut [u8; QK]) {
    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, scales);
        let dd = d * f32::from(sc);
        if dd == 0.0 {
            continue;
        }
        let dm = dmin * f32::from(m);
        for ii in 0..32 {
            let li = nint((x[32 * j + ii] + dm) / dd);
            l[32 * j + ii] = imax(0, imin(top, li)) as u8;
        }
    }
}

/// One Q4_K or Q5_K block. `q5` selects the 5-bit rules; `qw` the `_impl` search, else the `_ref` one.
fn kquant45_block(x: &[f32], qw: Option<&[f32]>, q5: bool, out: &mut Vec<u8>) {
    let nmax = if q5 { 31 } else { 15 };
    let mut l = [0u8; QK];
    let mut laux = [0u8; 32];
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut weights = [0.0f32; 32];
    let (d_bits, dmin_bits, sc);
    if let Some(qw) = qw {
        let sum_x2 = sum_sq(x);
        let sigma2 = 2.0f32 * sum_x2 / QK as f32;
        let mut sw = [0.0f32; 8];
        for j in 0..8 {
            for k in 0..32 {
                let v = x[32 * j + k];
                weights[k] = qw[32 * j + k] * (sigma2 + v * v).sqrt();
            }
            let mut sumw = 0.0f32;
            for &w in &weights {
                sumw += w;
            }
            sw[j] = sumw;
            let xs: &[f32; 32] = x[32 * j..32 * j + 32].try_into().expect("32 values");
            let (s, m) = make_qkx3_quants(
                nmax,
                xs,
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                (-0.9, 0.05, 36),
            );
            scales[j] = s;
            mins[j] = m;
        }
        let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
        let d_block = make_qp_quants(63, &scales, &mut ls, &sw);
        let m_block = make_qp_quants(63, &mins, &mut lm, &sw);
        for j in 0..8 {
            ls[j] = ls[j].min(63);
            lm[j] = lm[j].min(63);
        }
        sc = pack_scales(&ls, &lm);
        d_bits = f32_to_f16_bits(d_block);
        dmin_bits = f32_to_f16_bits(m_block);
    } else {
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..8 {
            let xs: &[f32; 32] = x[32 * j..32 * j + 32].try_into().expect("32 values");
            let sum_x2 = sum_sq(xs);
            let av_x = (sum_x2 / 32.0f32).sqrt();
            for k in 0..32 {
                weights[k] = av_x + xs[k].abs();
            }
            let (rmin, rdelta, nstep) = if q5 { (-0.5, 0.1, 15) } else { (-1.0, 0.1, 20) };
            let (s, m) = make_qkx2_quants(
                nmax,
                xs,
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                (rmin, rdelta, nstep),
            );
            scales[j] = s;
            mins[j] = m;
            if s > max_scale {
                max_scale = s;
            }
            if m > max_min {
                max_min = m;
            }
        }
        let inv_scale = if max_scale > 0.0 {
            63.0f32 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 {
            63.0f32 / max_min
        } else {
            0.0
        };
        let (mut ls, mut lm) = ([0u8; 8], [0u8; 8]);
        for j in 0..8 {
            ls[j] = (nint(inv_scale * scales[j]) as u8).min(63);
            lm[j] = (nint(inv_min * mins[j]) as u8).min(63);
        }
        sc = pack_scales(&ls, &lm);
        d_bits = f32_to_f16_bits(max_scale / 63.0f32);
        dmin_bits = f32_to_f16_bits(max_min / 63.0f32);
    }
    requant_levels(
        x,
        &sc,
        half_to_f32(d_bits),
        half_to_f32(dmin_bits),
        if q5 { 31 } else { 15 },
        &mut l,
    );
    out.extend_from_slice(&d_bits.to_le_bytes());
    out.extend_from_slice(&dmin_bits.to_le_bytes());
    out.extend_from_slice(&sc);
    if q5 {
        let mut qh = [0u8; 32];
        let mut ql = [0u8; 128];
        let (mut m1, mut m2) = (1u8, 2u8);
        for (half, n) in (0..QK).step_by(64).enumerate() {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[32 * half + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&ql);
    } else {
        for j in (0..QK).step_by(64) {
            for k in 0..32 {
                out.push(l[j + k] | (l[j + k + 32] << 4));
            }
        }
    }
}

/// One Q6_K block (`qw` selects the importance-weighted scale search, else the squared-value weights).
fn q6k_block(x: &[f32], qw: Option<&[f32]>, out: &mut Vec<u8>) {
    let mut l = [0i8; QK];
    let mut scales = [0.0f32; 16];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;
    for ib in 0..16 {
        let scale = make_qx_quants(
            32,
            &x[16 * ib..16 * ib + 16],
            &mut l[16 * ib..16 * ib + 16],
            qw.map(|q| &q[16 * ib..16 * ib + 16]),
        );
        scales[ib] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }
    if max_abs_scale < GROUP_MAX_EPS {
        out.extend_from_slice(&[0u8; 210]);
        return;
    }
    let iscale = -128.0f32 / max_scale;
    let d_bits = f32_to_f16_bits(1.0f32 / iscale);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = imin(127, nint(iscale * scales[ib])) as i8;
    }
    let d = half_to_f32(d_bits);
    for j in 0..16 {
        let dd = d * f32::from(sc[j]);
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let li = nint(x[16 * j + ii] / dd);
            l[16 * j + ii] = (imax(-32, imin(31, li)) + 32) as i8;
        }
    }
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for (half, j) in (0..QK).step_by(128).enumerate() {
        for k in 0..32 {
            let q = |o: usize| l[j + k + o] as u8;
            ql[64 * half + k] = (q(0) & 0xF) | ((q(64) & 0xF) << 4);
            ql[64 * half + k + 32] = (q(32) & 0xF) | ((q(96) & 0xF) << 4);
            qh[32 * half + k] =
                (q(0) >> 4) | ((q(32) >> 4) << 2) | ((q(64) >> 4) << 4) | ((q(96) >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(sc.iter().map(|&s| s as u8));
    out.extend_from_slice(&d_bits.to_le_bytes());
}

/// One Q8_0 block of 32 values.
fn q8_0_block(x: &[f32], out: &mut Vec<u8>) {
    let mut amax = 0.0f32;
    for &v in x {
        amax = fmax(amax, v.abs());
    }
    let d = amax / 127.0f32;
    let id = if d != 0.0 { 1.0f32 / d } else { 0.0 };
    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
    for &v in x {
        out.push((v * id).round() as i8 as u8);
    }
}

/// One row of `x` quantized to `kt` into ggml's block bytes. `qw` is the importance, one value per column: with it
/// Q4_K, Q5_K and Q6_K take ggml's `_impl` rules, without it the `_ref` rules; Q8_0 ignores it.
pub fn quantize_row(kt: KType, x: &[f32], qw: Option<&[f32]>) -> Result<Vec<u8>, KQuantError> {
    let n = x.len();
    let bv = kt.block_values();
    if !n.is_multiple_of(bv) {
        return Err(KQuantError::RowLength { n, block: bv });
    }
    if let Some(q) = qw {
        if q.len() != n {
            return Err(KQuantError::Weights { got: q.len(), n });
        }
        if kt != KType::Q8_0
            && let Some((index, &value)) = q.iter().enumerate().find(|(_, v)| !v.is_finite())
        {
            return Err(KQuantError::WeightNotFinite { index, value });
        }
    }
    if let Some((index, &value)) = x.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(KQuantError::NotFinite { index, value });
    }
    let mut out = Vec::with_capacity(n / bv * kt.block_bytes());
    for (i, xb) in x.chunks_exact(bv).enumerate() {
        let qb = qw.map(|q| &q[i * bv..(i + 1) * bv]);
        match kt {
            KType::Q4K => kquant45_block(xb, qb, false, &mut out),
            KType::Q5K => kquant45_block(xb, qb, true, &mut out),
            KType::Q6K => q6k_block(xb, qb, &mut out),
            KType::Q8_0 => q8_0_block(xb, &mut out),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
