//! The host transcription of the 32-value GEMM family (`bloomery_gpu::gemm`'s
//! `Gemm32Kernels`): its quantizer and the per-output numeric contract that
//! `gemm32.rs`'s module doc states. One owner, so every gate that holds a
//! `gemm_q8_0p`, `gemm_q8_0f` or `gemm_q5_1` output bit for bit holds it to
//! the same transcription (`gate_gemm`'s `g32_*` cases, `gate_glm5next_gemm`).

/// A `GemmAct32`'s three planes on the host, in the device layout: codes,
/// scales, sums.
pub type Planes32 = (Vec<u32>, Vec<f32>, Vec<i32>);

/// The host quantizer of one column into the device layout: per 32-value
/// block `d = amax/127` (1 for an all-zero block), codes `round(x/d)`
/// clamped to ±127 and their sum; a block holding a non-finite value
/// refused (NaN `d`, zero codes and sum). Returns the column's `16·steps`
/// code words, `2·steps` scales and sums, padding zero.
#[must_use]
pub fn quant_col(x: &[f32], steps: usize) -> Planes32 {
    let (mut q, mut d, mut s) = (
        vec![0u32; 16 * steps],
        vec![0f32; 2 * steps],
        vec![0i32; 2 * steps],
    );
    for (b, blk) in x.as_chunks::<32>().0.iter().enumerate() {
        if blk.iter().any(|v| !v.is_finite()) {
            d[b] = f32::NAN;
            continue;
        }
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let db = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        d[b] = db;
        let mut sum = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let c = (v / db).round().clamp(-127.0, 127.0) as i8;
            sum += i32::from(c);
            q[8 * b + i / 4] |= u32::from(c as u8) << (8 * (i % 4));
        }
        s[b] = sum;
    }
    (q, d, s)
}

/// The activations as the contract reads them: per column its codes (value
/// order), block scales and sums.
pub struct HostAct {
    pub k: usize,
    pub codes: Vec<i8>,
    pub d: Vec<f32>,
    pub s: Vec<i32>,
}

/// Quantize `x` (`n` columns of `k`) on the host; also returns the three
/// planes in the device layout for a bitwise comparison.
#[must_use]
pub fn host_act(x: &[f32], k: usize, n: usize) -> (HostAct, Planes32) {
    let steps = k.div_ceil(64);
    let kb = k / 32;
    let mut planes = (Vec::new(), Vec::new(), Vec::new());
    let mut ha = HostAct {
        k,
        codes: Vec::with_capacity(n * k),
        d: Vec::with_capacity(n * kb),
        s: Vec::with_capacity(n * kb),
    };
    for c in 0..n {
        let (q, d, s) = quant_col(&x[c * k..(c + 1) * k], steps);
        for w in &q[..k / 4] {
            ha.codes.extend(w.to_le_bytes().iter().map(|&b| b as i8));
        }
        ha.d.extend_from_slice(&d[..kb]);
        ha.s.extend_from_slice(&s[..kb]);
        planes.0.extend(q);
        planes.1.extend(d);
        planes.2.extend(s);
    }
    (ha, planes)
}

/// One output of weight row `(wq, wd, wm)` — its `k` codes, its `k/32` block
/// scales and (Q5_1, `mins`) block mins, read only when `mins` — against
/// column `col` of `a`, three ways: the contract's transcription (per block
/// the exact i32 dot, then `acc = fma(d_a, d_w·f32(isum), acc)`, the min
/// term `fma(m_w, f32(s_a), ·)` inside for Q5_1, from 0, blocks in
/// increasing k), the f64 value and its band's magnitude
/// `Σ_b (|D_b| + |M_b|)`.
#[must_use]
pub fn dot32(
    wq: &[i8],
    wd: &[f32],
    wm: &[f32],
    a: &HostAct,
    col: usize,
    mins: bool,
) -> (f32, f64, f64) {
    let kb = a.k / 32;
    let (mut acc, mut y, mut mag) = (0.0f32, 0.0f64, 0.0f64);
    for b in 0..kb {
        let wb = &wq[32 * b..32 * b + 32];
        let ab = &a.codes[col * a.k + 32 * b..col * a.k + 32 * b + 32];
        let isum: i32 = wb
            .iter()
            .zip(ab)
            .map(|(&p, &q)| i32::from(p) * i32::from(q))
            .sum();
        let dw = wd[b];
        let (da, sa) = (a.d[col * kb + b], a.s[col * kb + b]);
        let t = dw * isum as f32;
        let t = if mins { wm[b].mul_add(sa as f32, t) } else { t };
        acc = da.mul_add(t, acc);
        let dd = f64::from(da) * f64::from(dw) * f64::from(isum);
        let mm = if mins {
            f64::from(da) * f64::from(wm[b]) * f64::from(sa)
        } else {
            0.0
        };
        y += dd + mm;
        mag += dd.abs() + mm.abs();
    }
    (acc, y, mag)
}
