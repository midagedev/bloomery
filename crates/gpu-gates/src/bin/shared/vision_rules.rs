//! The checks of the image towers' gates that bind a kernel to its host rule: a launch against the
//! exact product inside the accumulation-order bound, a bit-exact rule, and a rule the device's
//! libm differs from by ulps. Shared by `gate_vision_encoder` and `gate_qwen3vl_tower`.

use bloomery_gpu_gates::verdict;
use bloomery_gpu_vision::f32_bf16;
use bloomery_gpu_vision::gemm_bf16::{gemm_ref, order_bound, within_order};

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
