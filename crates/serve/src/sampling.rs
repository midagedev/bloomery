//! The server's own sampler, used until the sampler crate is plugged in through a
//! [`SamplerFactory`]. llama-server's default chain restricted to the knobs this
//! server accepts: the penalties over the last `repeat_last_n` ids of the history
//! (the prompt's tail then the generated ids, as llama-server feeds the whole
//! prompt into the sampler), then top_k, then top_p, then min_p, then temperature,
//! then a draw; temperature `<= 0` takes the best candidate after the penalties.
//! [`sampler_factory`] is that plug, every seat's: the sampler crate's chain.

use std::collections::BTreeMap;
use std::sync::Arc;

use sampler::{Sampler as Chain, SamplerParams};

use crate::engine::{Sampler, SamplerFactory, SamplerRefused, SamplingParams};

/// The factory the server uses when none is given. It takes every parameter
/// the API lets through, so it refuses none.
#[must_use]
pub fn reference_factory() -> SamplerFactory {
    Arc::new(|p: &SamplingParams| -> Result<Sampler, SamplerRefused> {
        let p = p.clone();
        let mut rng = SplitMix64(p.seed);
        let mut cand: Vec<(u32, f32)> = Vec::new();
        Ok(Box::new(move |logits: &[f32], history: &[u32]| {
            sample(&p, &mut rng, &mut cand, logits, history)
        }))
    })
}

/// The server's `SamplingParams` as the sampler crate's chain: the penalties
/// over the request's window, `top_k <= 0` keeps every candidate. The server
/// never builds a sampler for `temperature <= 0`: those requests take the
/// engine's argmax.
#[must_use]
pub fn sampler_params(p: &SamplingParams) -> SamplerParams {
    SamplerParams {
        temperature: p.temperature,
        top_k: u32::try_from(p.top_k).unwrap_or(0),
        top_p: p.top_p,
        // Past 1 keeps only the best candidate, as 1 does; the chain refuses it.
        min_p: p.min_p.min(1.0),
        repeat_penalty: p.repeat_penalty,
        frequency_penalty: p.frequency_penalty,
        presence_penalty: p.presence_penalty,
        repeat_last_n: p.repeat_last_n,
        seed: p.seed,
    }
}

/// A factory of sampler-crate samplers. A parameter the chain still refuses
/// (a non-finite value) refuses the request, in the chain's own words.
#[must_use]
pub fn sampler_factory() -> SamplerFactory {
    Arc::new(|p: &SamplingParams| -> Result<Sampler, SamplerRefused> {
        let mut s = Chain::new(sampler_params(p)).map_err(|e| SamplerRefused(e.to_string()))?;
        Ok(Box::new(move |logits: &[f32], recent: &[u32]| {
            s.sample(logits, recent)
        }))
    })
}

/// Index of the largest logit (first of equals). NaN sorts above every number
/// under `total_cmp`, which surfaces a NaN logit instead of hiding it.
///
/// # Panics
///
/// On an empty row, which has no largest logit.
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let n = u32::try_from(logits.len()).expect("a vocabulary fits u32");
    argmax_in(logits, 0..n).expect("a model's vocabulary is non-empty")
}

/// The largest logit among `ids`, ties to the lowest id and NaN above every
/// number, whatever order `ids` comes in: the rule [`argmax`] applies to a whole
/// row. `None` for an empty set or an id past the row.
#[must_use]
pub fn argmax_in(logits: &[f32], ids: impl IntoIterator<Item = u32>) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for id in ids {
        let v = *logits.get(usize::try_from(id).ok()?)?;
        best = match best {
            Some((b, bv)) => match v.total_cmp(&bv) {
                std::cmp::Ordering::Greater => Some((id, v)),
                std::cmp::Ordering::Equal if id < b => Some((id, v)),
                _ => Some((b, bv)),
            },
            None => Some((id, v)),
        };
    }
    best.map(|(id, _)| id)
}

fn sample(
    p: &SamplingParams,
    rng: &mut SplitMix64,
    cand: &mut Vec<(u32, f32)>,
    logits: &[f32],
    history: &[u32],
) -> u32 {
    cand.clear();
    cand.extend(
        logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (u32::try_from(i).expect("a vocabulary fits u32"), l)),
    );
    if p.penalizes() {
        penalize(
            p,
            cand,
            &history[history.len().saturating_sub(p.repeat_last_n)..],
        );
    }
    // The engine's order: the larger logit first, equal ones (`-0` and `+0`
    // too) by id, so a row pruned to its best on the card draws the same.
    let by_logit_desc = |a: &(u32, f32), b: &(u32, f32)| {
        if a.1 == b.1 {
            a.0.cmp(&b.0)
        } else {
            b.1.total_cmp(&a.1)
        }
    };
    if p.temperature <= 0.0 {
        return cand
            .iter()
            .min_by(|a, b| by_logit_desc(a, b))
            .map_or(0, |c| c.0);
    }
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

/// llama.cpp's penalties (`llama_sampler_penalties_apply`): a token counted
/// `k > 0` times in `window` has its logit divided by `repeat_penalty` when
/// positive and multiplied by it otherwise, then `k * frequency_penalty +
/// presence_penalty` subtracted. `cand` is indexed by id.
fn penalize(p: &SamplingParams, cand: &mut [(u32, f32)], window: &[u32]) {
    let mut count: BTreeMap<u32, u32> = BTreeMap::new();
    for &t in window {
        *count.entry(t).or_insert(0) += 1;
    }
    for (t, k) in count {
        if let Some((_, l)) = usize::try_from(t).ok().and_then(|i| cand.get_mut(i)) {
            let r = if *l <= 0.0 {
                *l * p.repeat_penalty
            } else {
                *l / p.repeat_penalty
            };
            *l = r - (k as f32 * p.frequency_penalty + p.presence_penalty);
        }
    }
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

    /// The reference sampler ties `-0` and `+0` by id, as the engine and the
    /// card's best read do: a row of `-0` before a `+0` draws the same id
    /// whole as pruned to its best 64.
    #[test]
    fn the_reference_sampler_ties_zeros_by_id() {
        let p = SamplingParams {
            top_k: 1,
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let mut row = vec![-1.0f32; 100];
        row[..64].fill(-0.0);
        row[64] = 0.0;
        let mut pruned = row.clone();
        pruned[64..].fill(f32::NEG_INFINITY);
        let mut draw = reference_factory()(&p).unwrap_or_else(|e| panic!("refused: {e}"));
        assert_eq!([draw(&row, &[]), draw(&pruned, &[])], [0, 0]);
    }

    /// The factory's sampler is the sampler crate's chain over the request's
    /// parameters, and a parameter the chain refuses (a non-finite
    /// temperature) is the request's refusal in the chain's own words: no
    /// sampler that draws something else stands in for it.
    #[test]
    fn the_sampler_factory_refuses_what_the_chain_refuses() {
        use crate::engine::SamplingParams;
        let factory = super::sampler_factory();
        let greedy = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut draw = factory(&greedy).unwrap_or_else(|e| panic!("refused: {e}"));
        assert_eq!(draw(&[0.0, 3.0, 1.0], &[]), 1, "the chain's argmax");
        let Err(refused) = factory(&SamplingParams {
            temperature: f32::NAN,
            ..greedy
        }) else {
            panic!("a NaN temperature built a sampler");
        };
        assert!(
            refused.0.contains("temperature") && refused.0.contains("a finite number"),
            "{refused}"
        );
    }

    #[test]
    fn same_seed_same_draws_and_greedy_limit() {
        let logits = [0.0f32, 1.0, 1.1, 0.9, -3.0];
        let p = SamplingParams {
            temperature: 1.5,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
            ..SamplingParams::default()
        };
        let factory = reference_factory();
        let f = |p: &SamplingParams| factory(p).expect("the reference refuses none");
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
        let mut greedy = f(&SamplingParams {
            temperature: 0.0,
            presence_penalty: 0.5,
            ..p
        });
        assert_eq!(greedy(&logits, &[]), 2);
        assert_eq!(greedy(&logits, &[2]), 1, "1.1 - 0.5 < 1.0");
    }

    /// Ties go to the lowest id, NaN is above every number, and `-0.0` is
    /// below `0.0`: the order of `total_cmp`, the same over a whole row and
    /// over a subset.
    #[test]
    fn argmax_follows_the_total_order() {
        let nan = f32::NAN;
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let rows: [(&[f32], u32); 10] = [
            (&[1.0, 3.0, 3.0], 1),
            (&[3.0, 3.0, 1.0], 0),
            (&[nan, 1.0], 0),
            (&[1.0, nan], 1),
            (&[nan, nan], 0),
            (&[0.0, -0.0], 0),
            (&[-0.0, 0.0], 1),
            (&[neg_nan, f32::NEG_INFINITY], 1),
            (&[f32::NEG_INFINITY; 3], 0),
            (&[5.0], 0),
        ];
        for (row, want) in rows {
            assert_eq!(argmax(row), want, "{row:?}");
            let n = u32::try_from(row.len()).expect("a short row");
            assert_eq!(argmax_in(row, 0..n), Some(want), "{row:?}");
            assert_eq!(argmax_in(row, (0..n).rev()), Some(want), "{row:?}");
        }
    }

    /// An empty row has no largest logit: a named panic, not an id.
    #[test]
    #[should_panic(expected = "a model's vocabulary is non-empty")]
    fn argmax_of_an_empty_row_is_a_named_panic() {
        let _ = argmax(&[]);
    }
}
