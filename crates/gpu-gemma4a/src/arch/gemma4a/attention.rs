//! Gemma4a's blocked Transformer-XL shift and strictly bounded causal window.

use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

pub const WIDTH: usize = 1024;
pub const HEAD: usize = 128;
pub const HEADS: usize = WIDTH / HEAD;
pub const CONTEXT: usize = 12;
pub const RPE: usize = CONTEXT + 1;

/// The reference's blocked mask and its sinusoidal input, in ggml order.
#[must_use]
pub fn inputs(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(
        (1..=super::MAX_TOKENS).contains(&tokens),
        "gemma4a attention input token bound"
    );
    let mut mask = vec![-1e9; tokens.div_ceil(CONTEXT) * CONTEXT * 2 * CONTEXT];
    for b in 0..tokens.div_ceil(CONTEXT) {
        for q in 0..CONTEXT {
            let gq = b * CONTEXT + q;
            for k in 0..2 * CONTEXT {
                let gk = (b * CONTEXT + k).checked_sub(CONTEXT);
                if gq < tokens && gk.is_some_and(|gk| gk < tokens && gk <= gq && gq - gk < CONTEXT)
                {
                    mask[k + 2 * CONTEXT * (q + b * CONTEXT)] = 0.0;
                }
            }
        }
    }
    let increment = 10000.0_f32.ln() / (WIDTH / 2 - 1) as f32;
    let mut position = vec![0.0; WIDTH * RPE];
    for p in 0..RPE {
        for i in 0..WIDTH / 2 {
            let v = (CONTEXT - p) as f32 * (-(i as f32) * increment).exp();
            position[p * WIDTH + i] = v.sin();
            position[p * WIDTH + i + WIDTH / 2] = v.cos();
        }
    }
    (position, mask)
}

#[cuda_module]
pub(crate) mod kernels {
    use super::*;

    /// One valid query/head per thread, equivalent to blocking, shifting and trimming.
    #[allow(clippy::too_many_arguments, reason = "kernel entry: flat device ABI")]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (q.len() >= n * 1024, k.len() >= n * 1024, v.len() >= n * 1024, pos.len() >= 13 * 1024, qs.len() >= 128, ks.len() >= 128, y.len() >= n * 1024))]
    pub fn local_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        pos: &[f32],
        qs: &[f32],
        ks: &[f32],
        n: u32,
        qscale: f32,
        kscale: f32,
        fault: bloomery_gpu::FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize * HEADS {
            return;
        }
        let t = i / HEADS;
        let h = i % HEADS;
        let first = (t + 1).saturating_sub(CONTEXT);
        let count = t - first + 1;
        let mut scores = [0.0f32; CONTEXT];
        let mut max = f32::NEG_INFINITY;
        for j in 0..count {
            let key = first + j;
            let rel = CONTEXT - (t - key);
            let mut ac = 0.0f32;
            let mut bd = 0.0f32;
            for d in 0..HEAD {
                let a = (q[t * WIDTH + h * HEAD + d] * qscale) * qs[d];
                let b = (k[key * WIDTH + h * HEAD + d] * kscale) * ks[d];
                ac = a.mul_add(b, ac);
                bd = a.mul_add(pos[rel * WIDTH + h * HEAD + d], bd);
            }
            let raw = ac + bd;
            let s = if raw.is_finite() {
                (raw * (1.0 / 50.0)).tanh() * 50.0
            } else {
                fault.raise(bloomery_gpu::FaultSite::F32Product);
                f32::NAN
            };
            scores[j] = s;
            max = max.max(s);
        }
        let mut sum = 0.0f32;
        for j in 0..count {
            scores[j] = (scores[j] - max).exp();
            sum += scores[j];
        }
        for d in 0..HEAD {
            let mut out = 0.0f32;
            for j in 0..count {
                out = (scores[j] / sum).mul_add(v[(first + j) * WIDTH + h * HEAD + d], out);
            }
            // SAFETY: one thread owns each query/head, and d traverses that head only.
            unsafe {
                *y.get_unchecked_mut(t * WIDTH + h * HEAD + d) = out;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_mask_and_relative_shift() {
        assert!(std::panic::catch_unwind(|| inputs(0)).is_err());
        assert!(std::panic::catch_unwind(|| inputs(751)).is_err());
        assert_eq!(inputs(1).1.len(), 12 * 24);
        assert_eq!(inputs(750).1.len(), 63 * 12 * 24);
        let (pos, mask) = inputs(25);
        assert_eq!(pos.len(), 13 * 1024);
        assert_eq!(mask.len(), 3 * 12 * 24);
        for t in 0usize..25 {
            let (b, q) = (t / 12, t % 12);
            for k in 0..24 {
                let gk = (b * 12 + k).checked_sub(12);
                let live = gk.is_some_and(|gk| gk <= t && t - gk < 12);
                assert_eq!(mask[k + 24 * (q + 12 * b)] == 0.0, live);
                if live {
                    let flat = q * 24 + k;
                    assert_eq!(flat / 25, q);
                    assert_eq!(12 - flat % 25, t - gk.expect("live key"));
                }
            }
        }
        assert_eq!(
            mask[24 * (1 + 24)..].iter().filter(|&&x| x == 0.0).count(),
            0
        );
        assert!(pos[12 * 1024..12 * 1024 + 512].iter().all(|&x| x == 0.0));
        assert!(pos[12 * 1024 + 512..].iter().all(|&x| x == 1.0));
    }
}
