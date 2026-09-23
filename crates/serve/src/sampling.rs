//! The server's own sampler, used until the sampler crate is plugged in through a
//! [`SamplerFactory`]. llama-server's default chain restricted to the knobs this
//! server accepts: top_k, then top_p, then min_p, then temperature, then a draw.

use std::sync::Arc;

use crate::engine::{Sampler, SamplerFactory, SamplingParams};

/// The factory the server uses when none is given.
#[must_use]
pub fn reference_factory() -> SamplerFactory {
    Arc::new(|p: &SamplingParams| -> Sampler {
        let p = p.clone();
        let mut rng = SplitMix64(p.seed);
        let mut cand: Vec<(u32, f32)> = Vec::new();
        Box::new(move |logits: &[f32], _history: &[u32]| sample(&p, &mut rng, &mut cand, logits))
    })
}

/// Index of the largest logit (first of equals). NaN sorts above every number
/// under `total_cmp`, which surfaces a NaN logit instead of hiding it.
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if v.total_cmp(&logits[best]).is_gt() {
            best = i;
        }
    }
    u32::try_from(best).expect("a vocabulary fits u32")
}

fn sample(
    p: &SamplingParams,
    rng: &mut SplitMix64,
    cand: &mut Vec<(u32, f32)>,
    logits: &[f32],
) -> u32 {
    cand.clear();
    cand.extend(
        logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (u32::try_from(i).expect("a vocabulary fits u32"), l)),
    );
    let by_logit_desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if let Ok(k) = usize::try_from(p.top_k)
        && k > 0
        && k < cand.len()
    {
        cand.select_nth_unstable_by(k - 1, by_logit_desc);
        cand.truncate(k);
    }
    cand.sort_unstable_by(by_logit_desc);
    let Some(&(first, max)) = cand.first() else {
        return 0;
    };
    // top_p and min_p act on the untempered softmax, as in llama.cpp's chain.
    let probs: Vec<f64> = cand
        .iter()
        .map(|&(_, l)| f64::from(l - max).exp())
        .collect();
    let total: f64 = probs.iter().sum();
    let mut keep = cand.len();
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, pr) in probs.iter().enumerate() {
            acc += pr / total;
            if acc >= f64::from(p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        // probs[0] is the maximum, exp(0) = 1.
        let floor = f64::from(p.min_p);
        keep = keep.min(probs.iter().take_while(|&&pr| pr >= floor).count().max(1));
    }
    cand.truncate(keep);
    let t = f64::from(p.temperature);
    let weights: Vec<f64> = cand
        .iter()
        .map(|&(_, l)| (f64::from(l - max) / t).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    let mut r = rng.next_f64() * sum;
    for (&(id, _), w) in cand.iter().zip(&weights) {
        if r < *w {
            return id;
        }
        r -= w;
    }
    cand.last().map_or(first, |c| c.0)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        // 53 random mantissa bits in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_draws_and_greedy_limit() {
        let logits = [0.0f32, 1.0, 1.1, 0.9, -3.0];
        let p = SamplingParams {
            temperature: 1.5,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        };
        let f = reference_factory();
        let a: Vec<u32> = {
            let mut s = f(&p);
            (0..32).map(|_| s(&logits, &[])).collect()
        };
        let b: Vec<u32> = {
            let mut s = f(&p);
            (0..32).map(|_| s(&logits, &[])).collect()
        };
        assert_eq!(a, b);
        assert!(
            a.iter().any(|&x| x != a[0]),
            "temperature 1.5 should vary: {a:?}"
        );
        let mut k1 = f(&SamplingParams { top_k: 1, ..p });
        assert!((0..16).all(|_| k1(&logits, &[]) == 2));
        assert_eq!(argmax(&logits), 2);
    }
}
