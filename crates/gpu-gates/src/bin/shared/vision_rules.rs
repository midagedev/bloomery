//! The checks of the image towers' gates that bind a kernel to its host rule: a launch against the
//! exact product inside the accumulation-order bound, a bit-exact rule, a rule the device's libm
//! differs from by ulps, and the erf GELU inside its bound. Shared by the image towers' gates.

use bloomery_gpu_gates::verdict;
use bloomery_gpu_vision::gelu_tanh::gelu_erf_ref;
use bloomery_gpu_vision::gemm_bf16::{gelu_erf, gemm_ref, order_bound, within_order};
use bloomery_gpu_vision::layer_norm::layer_norm_ref;
use bloomery_gpu_vision::{bf16_f32, f32_bf16};

/// Threads the host rules run on.
const HOST_THREADS: usize = 8;

/// `f(row)` for every row on [`HOST_THREADS`] threads, summed.
pub fn par_rows<T: Send + Default + std::ops::AddAssign>(
    rows: usize,
    f: impl Fn(usize) -> T + Sync,
) -> T {
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..HOST_THREADS)
            .map(|t| {
                let f = &f;
                s.spawn(move || {
                    let mut acc = T::default();
                    let mut r = t;
                    while r < rows {
                        acc += f(r);
                        r += HOST_THREADS;
                    }
                    acc
                })
            })
            .collect();
        let mut total = T::default();
        for h in hs {
            total += h.join().expect("host rule thread");
        }
        total
    })
}

#[derive(Default, Clone, Copy)]
pub struct Count {
    pub outside: usize,
    pub differ: usize,
}

impl std::ops::AddAssign for Count {
    fn add_assign(&mut self, o: Count) {
        self.outside += o.outside;
        self.differ += o.differ;
    }
}

/// One GEMM of the chain against [`gemm_ref`]: every output inside the accumulation-order
/// bound, and the count that differ from the exact rounding.
#[allow(
    clippy::too_many_arguments,
    reason = "one GEMM's operands and shape, as the kernel takes them"
)]
pub fn gemm_check(
    name: &str,
    a: &[u16],
    b: &[u16],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
    got: &[u16],
) -> bool {
    let g = order_bound(k);
    let c = par_rows(m, |i| {
        let mut c = Count::default();
        for j in 0..n {
            let (exact, mag) = gemm_ref(a, b, bias, k, i, j);
            let y = got[i * n + j];
            c.outside += usize::from(!within_order(y, exact, g * mag));
            c.differ += usize::from(y != f32_bf16(exact as f32));
        }
        c
    });
    let pass = c.outside == 0;
    println!(
        "rule gemm {name:<12} m {m} n {n} k {k}: outside the order bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
        c.outside,
        m * n,
        c.differ,
        c.differ as f64 / (m * n) as f64,
        verdict(pass)
    );
    pass
}

/// A bit-exact rule: `got` equals `want` everywhere.
pub fn exact_check(name: &str, got: &[u16], want: &[u16]) -> bool {
    let differ =
        got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
    let pass = differ == 0;
    println!(
        "rule {name:<17} bit-exact: {differ} of {} differ  {}",
        want.len(),
        verdict(pass)
    );
    pass
}

/// A rule the device's libm differs from by ulps: at most `pin` values differ, none by more
/// than one bf16 step.
pub fn near_check(name: &str, got: &[u16], want: &[u16], pin: usize) -> bool {
    let (mut differ, mut far) = (0usize, 0usize);
    for (&a, &b) in got.iter().zip(want) {
        if a != b {
            differ += 1;
            far += usize::from(a.abs_diff(b) > 1 || (a ^ b) & 0x8000 != 0);
        }
    }
    let pass = got.len() == want.len() && differ <= pin && far == 0;
    println!(
        "rule {name:<17} {differ} of {} differ (pin {pin}), {far} by more than one step  {}",
        want.len(),
        verdict(pass)
    );
    pass
}

/// The erf GELU's allowed distance from the host rule's f32 value, in units of `|x|`. The
/// device runs the host's rule step for step (`z = x·√½`, `e = erf z`, `s = 1 + e`, `0.5·x·s`,
/// each op rounded to f32), so the two differ by the `erf` alone and by how the roundings fall:
/// the device's `erff` is within 2 ulp of the true value and the host's within 0.5, so
/// `|Δe| ≤ 2.5·2⁻²⁴` (the ulp of a magnitude below 1); the two roundings of `s` add at most
/// `2·2⁻²⁴` where `s ≥ 1` and `2·2⁻²⁵` below it; the two roundings of the product at most
/// `2·2⁻²⁴` of its magnitude. For x > 0 that is `0.5·|x|·4.5·2⁻²⁴ + 2·2⁻²⁴·|x| = 2.125·2⁻²³·|x|`;
/// for x < 0, where the sum `1 + e` cancels toward a small result and `s < 1`,
/// `0.5·|x|·3.5·2⁻²⁴ + 2⁻²⁴·|x| = 1.375·2⁻²³·|x|`. `2⁻²¹ = 4·2⁻²³` covers both. The bound follows
/// `|x|`, not the result: where `erf` is within an ulp of −1 the cancellation moves a result of
/// 1e-6 by several bf16 steps of itself, and a one-step band would call that a defect.
const ERF_BOUND: f64 = 1.0 / 2_097_152.0;

/// The erf GELU's results `got` for the bf16 inputs `xs` against the host rule inside
/// [`ERF_BOUND`]: how many are not a rounding of a value within `|x|·ERF_BOUND` of the rule's
/// f32 value, how many differ from the rule's own rounding, and the first outside input and
/// result.
pub fn erf_outside(xs: &[u16], got: &[u16]) -> (usize, usize, Option<(u16, u16)>) {
    let (mut outside, mut differ, mut first) = (0usize, 0usize, None);
    for (&x, &y) in xs.iter().zip(got) {
        let xf = bf16_f32(x);
        let centre = f64::from(gelu_erf(xf));
        if !within_order(y, centre, f64::from(xf).abs() * ERF_BOUND) {
            outside += 1;
            first.get_or_insert((x, y));
        }
        differ += usize::from(y != gelu_erf_ref(x));
    }
    (outside, differ, first)
}

/// [`erf_outside`] as a clause: no result outside the bound.
pub fn erf_check(name: &str, xs: &[u16], got: &[u16]) -> bool {
    let (outside, differ, first) = erf_outside(xs, got);
    let pass = xs.len() == got.len() && outside == 0;
    println!(
        "rule {name:<17} {outside} of {} outside the erf bound |x|·2⁻²¹, {differ} differ from the host rounding  {}",
        xs.len(),
        verdict(pass)
    );
    if let Some((x, y)) = first {
        println!(
            "rule {name:<17} first outside: x {:#06x} ({:e}) gives {:#06x} ({:e}), the host rule {:#06x}",
            x,
            bf16_f32(x),
            y,
            bf16_f32(y),
            gelu_erf_ref(x)
        );
    }
    pass
}

/// Layer norm of every row of `x` (rows of the gain's width) with the rule `layer_norm_ref`.
pub fn norm_rows(x: &[u16], g: &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<u16> {
    x.chunks_exact(g.0.len())
        .flat_map(|r| layer_norm_ref(r, &g.0, &g.1, eps))
        .collect()
}
