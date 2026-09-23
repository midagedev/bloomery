//! Token sampling over one row of logits, in the order of the reference's
//! default chain (ik_llama.cpp `common/sampling.cpp`): repetition penalty
//! (`llama_sampling_prepare`), then top-k, top-p, min-p and temperature
//! (`sampler_queue`, `samplers_sequence` in `common/sampling.h`), then a
//! seeded draw from the softmax of what survives
//! (`llama_sample_token_with_rng_impl`). Top-p and min-p therefore judge the
//! untempered logits, and top-p's probabilities are normalized over the
//! top-k survivors, not the vocabulary. Every stage keeps at least one
//! candidate (the reference's `min_keep = max(1, 0)`).
//!
//! Candidates are ordered by (logit descending, id ascending) — the engine's
//! greedy tie rule (`argmax_take` in `crates/gpu/src/elem.rs`) — so a draw is
//! a function of the logits, the recent tokens and the seed alone, and
//! `top_k = 1` answers exactly what greedy answers.

use std::cmp::Ordering;
use std::fmt;

/// The sampling chain's knobs. Each stage has a value that switches it off:
/// `temperature <= 0` is greedy (the argmax, after the repetition penalty),
/// `top_k == 0` keeps every candidate, `top_p >= 1` and `min_p <= 0` keep
/// every candidate, and `repeat_penalty == 1` or `repeat_last_n == 0`
/// penalizes nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerParams {
    /// Divides the surviving logits before the draw; `<= 0` samples greedily.
    pub temperature: f32,
    /// Keep the `top_k` best candidates; 0 keeps all.
    pub top_k: u32,
    /// Keep the smallest best-first prefix whose probability mass is `>= top_p`.
    pub top_p: f32,
    /// Drop candidates whose probability is below `min_p` times the best one's.
    pub min_p: f32,
    /// Divides a positive logit and multiplies a non-positive one, once per
    /// distinct token in the window.
    pub repeat_penalty: f32,
    /// How many of the most recent tokens the penalty window covers.
    pub repeat_last_n: usize,
    /// Seeds the draw; the same seed and inputs give the same tokens.
    pub seed: u64,
}

impl Default for SamplerParams {
    /// The reference's defaults (`X_COMMON_PARAMS_SAMPLING`): temperature
    /// 0.8, top-k 40, top-p 0.95, min-p 0.05, no repetition penalty over a
    /// 64-token window. The reference seeds from the clock by default; this
    /// one uses seed 0 so a default sampler is reproducible.
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerParams {
    /// Plain greedy decoding: the argmax of the logits as given.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 0,
        }
    }
}

/// A parameter [`Sampler::new`] refuses, with the rule it breaks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamError {
    /// The field of [`SamplerParams`].
    pub param: &'static str,
    /// The value given.
    pub value: f32,
    /// What the field must be.
    pub rule: &'static str,
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sampler parameter {} = {} is invalid: {}",
            self.param, self.value, self.rule
        )
    }
}

impl std::error::Error for ParamError {}

/// One candidate token: its id and its logit after the repetition penalty
/// (before temperature).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Token id, the logit's index in the row.
    pub id: u32,
    /// The logit after the penalty; a NaN input reads as `-inf` here.
    pub logit: f32,
}

/// A seeded sampler. Its scratch is sized by the first call and reused, so
/// later calls over rows of the same length do not allocate.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplerParams,
    rng: Xoshiro256,
    cand: Vec<Candidate>,
    cum: Vec<f64>,
    // Tokens already penalized in this call; cleared again before `sample` returns.
    seen: Vec<bool>,
}

impl Sampler {
    /// A sampler over `params`, or the first parameter that is not a finite
    /// number in its range.
    pub fn new(params: SamplerParams) -> Result<Self, ParamError> {
        let check = |param, value: f32, ok: bool, rule| {
            if ok {
                Ok(())
            } else {
                Err(ParamError { param, value, rule })
            }
        };
        let t = params.temperature;
        check("temperature", t, t.is_finite(), "a finite number")?;
        let p = params.top_p;
        check("top_p", p, p.is_finite(), "a finite number")?;
        let m = params.min_p;
        check(
            "min_p",
            m,
            m.is_finite() && m <= 1.0,
            "a finite number <= 1",
        )?;
        let r = params.repeat_penalty;
        check(
            "repeat_penalty",
            r,
            r.is_finite() && r > 0.0,
            "a finite number > 0",
        )?;
        Ok(Self {
            params,
            rng: Xoshiro256::seeded(params.seed),
            cand: Vec::new(),
            cum: Vec::new(),
            seen: Vec::new(),
        })
    }

    /// The parameters this sampler was built with.
    #[must_use]
    pub fn params(&self) -> SamplerParams {
        self.params
    }

    /// Draw the next token from one row of `logits` (id = index), penalizing
    /// the tail of `recent` (the tokens so far, oldest first). An empty row
    /// answers 0, as the engine's argmax does; ids past `u32::MAX` are not
    /// candidates. With `temperature <= 0` and no active penalty the answer is
    /// the engine's argmax bit for bit: the largest logit, ties to the lower
    /// id, a NaN never winning.
    #[must_use]
    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        let p = self.params;
        self.cand.clear();
        if logits.is_empty() {
            return 0;
        }
        let window = &recent[recent.len().saturating_sub(p.repeat_last_n)..];
        let penalize = p.repeat_penalty != 1.0 && !window.is_empty();
        if p.temperature <= 0.0 && !penalize {
            let (id, logit) = argmax(ids(logits));
            self.cand.push(Candidate { id, logit });
            return id;
        }
        self.cand.extend(ids(logits).map(|(id, v)| Candidate {
            id,
            logit: sanitize(v),
        }));
        if penalize {
            self.penalize(window);
        }
        if p.temperature <= 0.0 {
            let (id, logit) = argmax(self.cand.iter().map(|c| (c.id, c.logit)));
            self.cand.clear();
            self.cand.push(Candidate { id, logit });
            return id;
        }
        self.top_k();
        self.top_p_min_p();
        self.draw()
    }

    /// The candidates the last [`sample`](Self::sample) drew from, best first
    /// (logit descending, id ascending). After a greedy call this is the one
    /// token it chose.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.cand
    }

    /// The reference's repetition penalty (`llama_sample_repetition_penalties_impl`):
    /// once per distinct in-vocabulary token of `window`, a logit `<= 0` is
    /// multiplied by the penalty and a positive one divided by it.
    fn penalize(&mut self, window: &[u32]) {
        let r = self.params.repeat_penalty;
        let n = self.cand.len();
        if self.seen.len() < n {
            self.seen.resize(n, false);
        }
        for &t in window {
            let i = t as usize;
            if i < n && !self.seen[i] {
                self.seen[i] = true;
                let c = &mut self.cand[i];
                let l = c.logit;
                c.logit = sanitize(if l <= 0.0 { l * r } else { l / r });
            }
        }
        for &t in window {
            if let Some(s) = self.seen.get_mut(t as usize) {
                *s = false;
            }
        }
    }

    /// Keep the `top_k` best candidates, unordered (`llama_sample_top_k_impl`).
    fn top_k(&mut self) {
        let k = self.params.top_k as usize;
        if k > 0 && k < self.cand.len() {
            self.cand.select_nth_unstable_by(k - 1, best_first);
            self.cand.truncate(k);
        }
    }

    /// Top-p, then min-p, leaving the survivors sorted best first.
    ///
    /// Both keep a best-first prefix of the top-k survivors: top-p the
    /// shortest one whose mass (normalized over all top-k survivors,
    /// `llama_sample_top_p_impl`) reaches `top_p`, min-p the ones whose logit
    /// is `>= max + ln(min_p)` (`llama_sample_min_p_impl`, sorted path). The
    /// chain's answer is the shorter prefix, so min-p filters first and only
    /// its survivors are sorted; top-p's normalizer is summed before that
    /// over every top-k survivor, so its cut is the one the reference makes.
    fn top_p_min_p(&mut self) {
        let p = self.params;
        let max = self
            .cand
            .iter()
            .fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
        if !max.is_finite() {
            // Every candidate is -inf, or one is +inf: no softmax exists, and
            // the best candidate is the only one with a meaning.
            let best = self.cand.iter().copied().min_by(best_first);
            self.cand.clear();
            self.cand.extend(best);
            return;
        }
        let top_p = p.top_p < 1.0;
        let norm: f64 = if top_p {
            self.cand.iter().map(|c| weight(c.logit, max, 1.0)).sum()
        } else {
            0.0
        };
        if p.min_p > 0.0 {
            let floor = max + p.min_p.ln();
            self.cand.retain(|c| c.logit >= floor);
        }
        self.cand.sort_unstable_by(best_first);
        if top_p {
            let target = f64::from(p.top_p);
            let mut mass = 0.0;
            let mut keep = self.cand.len();
            for (i, c) in self.cand.iter().enumerate() {
                mass += weight(c.logit, max, 1.0) / norm;
                if mass >= target {
                    keep = i + 1;
                    break;
                }
            }
            self.cand.truncate(keep);
        }
    }

    /// Temperature, softmax and a seeded draw over the sorted survivors
    /// (`llama_sample_temp_impl`, `llama_sample_token_with_rng_impl`). A single
    /// survivor is returned without drawing.
    fn draw(&mut self) -> u32 {
        let [first, rest @ ..] = self.cand.as_slice() else {
            return 0;
        };
        if rest.is_empty() {
            return first.id;
        }
        let max = first.logit;
        let inv_t = 1.0 / f64::from(self.params.temperature);
        self.cum.clear();
        let mut total = 0.0;
        for c in &self.cand {
            total += weight(c.logit, max, inv_t);
            self.cum.push(total);
        }
        let u = self.rng.next_f64() * total;
        // `u < total` in exact arithmetic; the clamp covers a product that
        // rounds up to `total` (the reference pads its last bin for the same reason).
        let i = self
            .cum
            .partition_point(|&c| c <= u)
            .min(self.cand.len() - 1);
        self.cand[i].id
    }
}

/// `(id, logit)` pairs of a row; ids stop at `u32::MAX`.
fn ids(logits: &[f32]) -> impl Iterator<Item = (u32, f32)> + '_ {
    (0..=u32::MAX).zip(logits.iter().copied())
}

/// The engine's greedy rule (`argmax_take`): a strictly greater value wins, an
/// equal one wins at a lower id; the running best starts at `(0, -inf)`, so a
/// NaN never wins and an all-`-inf` or all-NaN row answers id 0.
fn argmax(xs: impl Iterator<Item = (u32, f32)>) -> (u32, f32) {
    let (mut best_i, mut best_v) = (0u32, f32::NEG_INFINITY);
    for (i, v) in xs {
        if v > best_v || (v == best_v && i < best_i) {
            best_i = i;
            best_v = v;
        }
    }
    (best_i, best_v)
}

/// A logit as the candidate order sees it: NaN as `-inf` (it can never win,
/// as in [`argmax`]) and `-0.0` as `+0.0` (they tie, as `==` says), so
/// `total_cmp` agrees with the greedy rule on every input.
fn sanitize(v: f32) -> f32 {
    if v.is_nan() {
        f32::NEG_INFINITY
    } else {
        v + 0.0
    }
}

/// Logit descending, then id ascending — a total order on sanitized candidates.
fn best_first(a: &Candidate, b: &Candidate) -> Ordering {
    b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id))
}

/// `exp((logit - max) · inv_t)`, the unnormalized softmax weight.
fn weight(logit: f32, max: f32, inv_t: f64) -> f64 {
    (f64::from(logit - max) * inv_t).exp()
}

/// xoshiro256** (Blackman and Vigna), seeded through splitmix64.
#[derive(Debug, Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn seeded(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
