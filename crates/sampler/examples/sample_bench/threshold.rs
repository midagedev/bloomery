//! Experimental relative-weight cutoff; the benchmark's chain owns the draw.
//!
//! Filters use untempered weights and top-p's denominator precedes min-p, as
//! Bloomery requires. A scalar cutoff deliberately exposes boundary ties: a
//! probability alone cannot select only some equal-probability token ids.

use sampler::{Candidate, SamplerParams};

pub(super) struct Threshold {
    params: SamplerParams,
    draw: super::Chain,
    rng: Xoshiro256,
    weights: Vec<f64>,
}

impl Threshold {
    pub(super) fn new(params: SamplerParams) -> Self {
        assert_eq!(params.repeat_penalty, 1.0, "penalty-free benchmark arm");
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
        Self {
            params,
            draw: super::Chain::default(),
            rng: Xoshiro256::seeded(params.seed),
            weights: Vec::new(),
        }
    }

    pub(super) fn candidates(&self) -> &[Candidate] {
        &self.draw.cand
    }

    pub(super) fn sample(&mut self, row: &[f32]) -> u32 {
        self.draw.cand.clear();
        let max = row
            .iter()
            .copied()
            .map(super::sanitize)
            .fold(f32::NEG_INFINITY, f32::max);
        // These rows have no softmax. Keep the engine's lowest-id argmax,
        // without advancing its random generator.
        if !max.is_finite() || self.params.temperature <= 0.0 || self.params.top_k == 1 {
            if let Some((id, logit)) = row
                .iter()
                .copied()
                .map(super::sanitize)
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            {
                self.draw.cand.push(Candidate {
                    id: u32::try_from(id).expect("benchmark vocabulary fits u32"),
                    logit,
                });
                return self.draw.cand[0].id;
            }
            return 0;
        }
        self.weights.clear();
        self.weights.extend(
            row.iter()
                .map(|&v| super::weight(super::sanitize(v), max, 1.0)),
        );
        let k = self.params.top_k as usize;
        let k_floor = if k > 0 && k < row.len() {
            cutoff(&self.weights, 0.0, Goal::Count(k))
        } else {
            0.0
        };
        // min-p must not shrink the top-p normalizer.
        let p_floor = if self.params.top_p < 1.0 {
            let norm: f64 = self.weights.iter().filter(|&&w| w >= k_floor).sum();
            cutoff(
                &self.weights,
                k_floor,
                Goal::Mass(f64::from(self.params.top_p) * norm),
            )
        } else {
            0.0
        };
        let floor = k_floor.max(p_floor).max(f64::from(self.params.min_p));
        for (id, (&v, &w)) in row.iter().zip(&self.weights).enumerate() {
            if w >= floor {
                self.draw.cand.push(Candidate {
                    id: u32::try_from(id).expect("benchmark vocabulary fits u32"),
                    logit: super::sanitize(v),
                });
            }
        }
        self.draw.cand.sort_unstable_by(super::best_first);
        let u = if self.draw.cand.len() > 1 {
            self.rng.next_f64()
        } else {
            0.0
        };
        self.draw.draw(self.params.temperature, u)
    }
}

/// The production RNG, checked against seeded sampling without a second filter pass.
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn seeded(seed: u64) -> Self {
        let mut generator = super::Gen(seed);
        Self {
            s: std::array::from_fn(|_| generator.u64()),
        }
    }

    fn next_f64(&mut self) -> f64 {
        let s = &mut self.s;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        (out >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[derive(Clone, Copy)]
enum Goal {
    Count(usize),
    Mass(f64),
}

/// Descend eight byte-radix histograms of nonnegative f64 weights. Only the
/// bucket containing the boundary participates in the next pass; no full sort.
fn cutoff(weights: &[f64], floor: f64, goal: Goal) -> f64 {
    let (mut prefix, mut mask) = (0u64, 0u64);
    let (mut count_hi, mut mass_hi) = (0usize, 0.0f64);
    for shift in (0..8).rev().map(|byte| byte * 8) {
        let mut counts = [0usize; 256];
        let mut masses = [0.0f64; 256];
        for &w in weights {
            let bits = w.to_bits();
            if w >= floor && bits & mask == prefix {
                let bucket = ((bits >> shift) & 255) as usize;
                counts[bucket] += 1;
                masses[bucket] += w;
            }
        }
        let mut chosen = None;
        for bucket in (0..256).rev() {
            let fired = counts[bucket] > 0
                && match goal {
                    Goal::Count(n) => count_hi + counts[bucket] >= n,
                    Goal::Mass(m) => mass_hi + masses[bucket] >= m,
                };
            if fired {
                chosen = Some(bucket);
                break;
            }
            count_hi += counts[bucket];
            mass_hi += masses[bucket];
        }
        let Some(bucket) = chosen else {
            return floor;
        };
        prefix |= (bucket as u64) << shift;
        mask |= 255u64 << shift;
    }
    f64::from_bits(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sampler::Sampler;

    #[test]
    fn rank_and_mass_boundaries() {
        let weights = [0.25, 1.0, 0.5, 0.5];
        assert_eq!(cutoff(&weights, 0.0, Goal::Count(2)), 0.5);
        assert_eq!(cutoff(&weights, 0.0, Goal::Mass(1.25)), 0.5);
        assert_eq!(cutoff(&weights, 0.5, Goal::Mass(3.0)), 0.5);
    }

    #[test]
    fn same_seed_after_singleton() {
        let params = SamplerParams {
            top_k: 0,
            ..SamplerParams::default()
        };
        let mut real = Sampler::new(params).unwrap();
        let mut threshold = Threshold::new(params);
        for row in [&[20.0, 0.0, -1.0][..], &[0.0, 0.2, -0.1][..]]
            .into_iter()
            .cycle()
            .take(64)
        {
            assert_eq!(threshold.sample(row), real.sample(row, &[]));
            assert_eq!(threshold.candidates(), real.candidates());
        }
    }

    #[test]
    fn a_scalar_cannot_cut_a_tie_by_id() {
        let params = SamplerParams {
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            ..SamplerParams::default()
        };
        let mut real = Sampler::new(params).unwrap();
        let mut threshold = Threshold::new(params);
        let _ = real.sample(&[0.0; 4], &[]);
        threshold.sample(&[0.0; 4]);
        assert_eq!(real.candidates().len(), 2);
        assert_eq!(threshold.candidates().len(), 4);
    }
}
