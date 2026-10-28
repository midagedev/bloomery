//! ggml's rope table, as host data: the values every backend's rotation
//! reads, computed once per position the way ik's CPU build does. The table
//! is ggml's recipe ([`ggml_rope_cache`]); a backend's table ([`RopeTable`])
//! holds what does not depend on the position, so a position costs per pair
//! two multiplies, one fused multiply-add under YaRN, a `sin` and a `cos`.
//! A backend's own pieces — a card's rows for every cache position
//! (CUDA's `RopeRows`), the unfused rotation core of its NORM-mode kernels —
//! stay with it; which values form pair `i` and how they turn is the
//! kernel's.

use std::sync::Arc;

use crate::ChainError;

/// Which way a table turns: `Forward` is rope, `Back` its inverse — ik's
/// `ROPE_BACK`, the same table with the sines negated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Back,
}

impl Direction {
    /// ggml's `sin_sign`.
    fn sin_sign(self) -> f32 {
        match self {
            Direction::Forward => 1.0,
            Direction::Back => -1.0,
        }
    }
}

/// The constants one rope site hands ggml's rope (`ggml_rope_ext`'s
/// arguments after the mode, and its frequency factors). ik's V4.1 graph uses
/// two (`build_deepseek4.cpp`): [`RopeSpec::window`] and [`RopeSpec::yarn`];
/// a plain rope over a whole head is [`RopeSpec::window`] at the head width.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeSpec {
    /// Values rotated at the tail of each head (`rope.dimension_count`).
    pub n_dims: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// ggml's `n_ctx_orig`, an `int`.
    pub n_ctx_orig: i32,
    /// ggml's `freq_factors` (`rope_freqs.weight`): pair `i`'s angle is divided by
    /// factor `i` before the rest of the recipe, so a very large factor leaves its
    /// pair unturned. `n_dims / 2` positive finite values ([`RopeSpec::with_factors`]
    /// checks them, [`RopeTable::new`] again); `None`: no factors.
    pub factors: Option<Arc<[f32]>>,
}

impl RopeSpec {
    /// The rope of a layer that keeps no compressed stream
    /// (`attention.compress_ratios` 0): `rope.freq_base` and no scaling —
    /// ik passes `freq_scale` 1, `ext_factor` 0, betas 0 and `n_ctx_orig` 0,
    /// and `dsv4_rope_attn_factor` is 1 at `ext_factor` 0, so the table is
    /// plain `cos`/`sin` of `p·θ_i`.
    #[must_use]
    pub fn window(freq_base: f32, n_dims: usize) -> RopeSpec {
        RopeSpec {
            n_dims,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            n_ctx_orig: 0,
            factors: None,
        }
    }

    /// The rope of every other site — the compressed layers' heads, the
    /// pooled rows, the index keys and the indexer query:
    /// `attention.compress_rope_freq_base`, YaRN at `freq_scale =
    /// 1/rope.scaling.factor` with `ext_factor` 1 (llama.cpp's value for a
    /// `yarn` file) and the file's betas and original context, and
    /// `attn_factor = 1/(1 + 0.1·ln(1/freq_scale))` rounded op by op — ik's
    /// `dsv4_rope_attn_factor`, in libllama, which is built without FMA.
    #[must_use]
    pub fn yarn(
        freq_base: f32,
        scaling_factor: f32,
        n_ctx_orig: i32,
        beta_fast: f32,
        beta_slow: f32,
        n_dims: usize,
    ) -> RopeSpec {
        let freq_scale = 1.0 / scaling_factor;
        RopeSpec {
            n_dims,
            freq_base,
            freq_scale,
            ext_factor: 1.0,
            attn_factor: 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln()),
            beta_fast,
            beta_slow,
            n_ctx_orig,
            factors: None,
        }
    }

    /// `self` with the file's frequency factors, one per pair of the `n_dims` values the
    /// rope turns. A table of another length, or a factor that is zero, negative or not
    /// finite, is refused by name: ggml divides by it, and the quotient is undefined.
    pub fn with_factors(mut self, factors: &[f32]) -> Result<RopeSpec, ChainError> {
        check_factors("RopeSpec::with_factors", factors, self.n_dims)?;
        self.factors = Some(factors.into());
        Ok(self)
    }
}

/// Refuse by name a factor table of other than `n_dims / 2` values, or one with a value that is
/// not positive and finite.
fn check_factors(what: &'static str, factors: &[f32], n_dims: usize) -> Result<(), ChainError> {
    if factors.len() != n_dims / 2 {
        return Err(ChainError::shape(
            what,
            format!(
                "{} frequency factors for a rope of {n_dims} values; one for each of the {} pairs",
                factors.len(),
                n_dims / 2
            ),
        ));
    }
    if let Some((i, f)) = factors
        .iter()
        .enumerate()
        .find(|(_, f)| !(f.is_finite() && **f > 0.0))
    {
        return Err(ChainError::shape(
            what,
            format!(
                "frequency factor {i} is {f}; a factor divides the pair's angle, so it is \
                 positive and finite"
            ),
        ));
    }
    Ok(())
}

/// One site's cos/sin table at any position: [`ggml_rope_cache`]'s values,
/// with what does not depend on the position — `theta_scale`, the YaRN ramp
/// of each pair, the magnitude factor, the frequency factors — computed or
/// held once, so a position costs per pair two multiplies, one division with
/// factors, one fused multiply-add under YaRN, a `sin` and a `cos`.
#[derive(Clone, Debug)]
pub struct RopeTable {
    n_dims: usize,
    freq_scale: f32,
    theta_scale: f32,
    /// Per pair, `(ramp_mix, 1 − ramp_mix)`; empty without YaRN.
    ramp: Vec<(f32, f32)>,
    /// Per pair, ggml's `freq_factors`; empty without them.
    factors: Vec<f32>,
    /// `attn_factor`, times ggml's `1 + 0.1·ln(1/freq_scale)` under YaRN.
    mscale: f32,
}

impl RopeTable {
    /// The table of `spec`; `n_dims` must be even and at least 2, and the spec's frequency
    /// factors, when it has them, one positive finite value for each pair.
    pub fn new(spec: &RopeSpec) -> Result<RopeTable, ChainError> {
        let nd = spec.n_dims;
        if nd < 2 || !nd.is_multiple_of(2) {
            return Err(ChainError::shape(
                "RopeTable::new",
                format!("n_dims must be even and at least 2, got {nd}"),
            ));
        }
        let factors = match &spec.factors {
            Some(f) => {
                check_factors("RopeTable::new", f, nd)?;
                f.to_vec()
            }
            None => Vec::new(),
        };
        let (ramp, mscale) = if spec.ext_factor == 0.0 {
            (Vec::new(), spec.attn_factor)
        } else {
            let corr = ggml_rope_yarn_corr_dims(spec);
            let ramp = (0..nd / 2)
                .map(|i| {
                    let mix = rope_yarn_ramp(corr[0], corr[1], 2 * i) * spec.ext_factor;
                    (mix, 1.0 - mix)
                })
                .collect();
            (ramp, spec.attn_factor * yarn_mscale(spec.freq_scale))
        };
        Ok(RopeTable {
            n_dims: nd,
            freq_scale: spec.freq_scale,
            theta_scale: theta_scale(spec),
            ramp,
            factors,
            mscale,
        })
    }

    /// Values per position: `n_dims`, a cos and a sin per pair.
    #[must_use]
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Append position `pos`'s table to `out`: `n_dims` f32, `[cos_0, sin_0,
    /// cos_1, …]` with the sines negated for [`Direction::Back`] — the
    /// layout the kernels read at `t·n_dims` for token `t`. `pos` becomes
    /// f32 as ggml's `int64_t` does, exactly below 2^24.
    pub fn push(&self, pos: u32, dir: Direction, out: &mut Vec<f32>) {
        let sign = dir.sin_sign();
        let mut theta = pos as f32;
        for i in 0..self.n_dims / 2 {
            let extrap = match self.factors.get(i) {
                Some(&ff) => theta / ff,
                None => theta,
            };
            let interp = self.freq_scale * extrap;
            let th = match self.ramp.get(i) {
                Some(&(mix, keep)) => interp.mul_add(keep, extrap * mix),
                None => interp,
            };
            let (s, c) = th.sin_cos();
            out.push(c * self.mscale);
            out.push(s * self.mscale * sign);
            theta *= self.theta_scale;
        }
    }
}

/// ggml's `theta_scale = powf(freq_base, −2.0f/n_dims)`.
fn theta_scale(spec: &RopeSpec) -> f32 {
    spec.freq_base.powf(-2.0 / spec.n_dims as f32)
}

/// `1.0f + 0.1f·logf(1.0f/freq_scale)`, the factor `rope_yarn` multiplies
/// its magnitude by under YaRN — one fused multiply-add in ik's libggml.
fn yarn_mscale(freq_scale: f32) -> f32 {
    0.1f32.mul_add((1.0 / freq_scale).ln(), 1.0)
}

/// ggml's `MAX(a, b)`: `a > b ? a : b` — a NaN `b` comes back.
fn c_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// ggml's `MIN(a, b)`: `a < b ? a : b` — a NaN `b` comes back.
fn c_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `ggml_rope_yarn_corr_dim`: `n_dims·logf(n_ctx_orig/(n_rot·2·π))/(2·logf(base))`.
fn ggml_rope_yarn_corr_dim(n_dims: usize, n_ctx_orig: i32, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

/// `ggml_rope_yarn_corr_dims`: the pair range the YaRN ramp runs over.
fn ggml_rope_yarn_corr_dims(spec: &RopeSpec) -> [f32; 2] {
    let n = spec.n_dims;
    let start = ggml_rope_yarn_corr_dim(n, spec.n_ctx_orig, spec.beta_fast, spec.freq_base).floor();
    let end = ggml_rope_yarn_corr_dim(n, spec.n_ctx_orig, spec.beta_slow, spec.freq_base).ceil();
    [c_max(0.0, start), c_min(n as f32 - 1.0, end)]
}

/// `rope_yarn_ramp`: `1 − MIN(1, MAX(0, (i0/2 − low)/MAX(0.001, high − low)))`,
/// `i0/2` an integer division.
fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = ((i0 / 2) as f32 - low) / c_max(0.001, high - low);
    1.0 - c_min(1.0, c_max(0.0, y))
}

/// ggml's rope cache for one position, transcribed statement for statement
/// from ik's `ggml.c` (`ggml_rope_cache_init`, `rope_yarn`, `rope_yarn_ramp`,
/// `ggml_rope_yarn_corr_dims`) as its CPU build compiles them: GCC, at
/// `-march=native` with GNU C's default contraction, fuses
/// `theta_interp·(1 − ramp_mix) + theta_extrap·ramp_mix` and
/// `1 + 0.1·ln(1/freq_scale)` into fused multiply-adds and rounds every other
/// op on its own, and `MAX`/`MIN` are the C macros — the window rope's
/// correction range, from `n_ctx_orig` 0, is NaN and unused. `ne0` is the
/// head width: the cache holds `ne0` values and the tail rope reads the
/// first `n_dims`. The frequency factors are `ggml_rope_cache_init`'s
/// `theta/ff` (mainline's ggml-cpu `ops.cpp` and ik's `ggml.c` agree), the
/// division done on the angle before the rest of the recipe. The
/// verification side's reading of the recipe; the engine's tables come from
/// [`RopeTable`], which the unit test pins to it.
///
/// # Panics
///
/// On a spec whose factors do not cover the `ne0 / 2` pairs the cache fills,
/// by name: ggml would read past its table.
#[must_use]
pub fn ggml_rope_cache(spec: &RopeSpec, pos: u32, ne0: usize, dir: Direction) -> Vec<f32> {
    if let Some(f) = &spec.factors {
        assert!(
            f.len() >= ne0 / 2,
            "ggml_rope_cache: {} frequency factors for a cache of {} pairs",
            f.len(),
            ne0 / 2
        );
    }
    let theta_scale = theta_scale(spec);
    let corr_dims = ggml_rope_yarn_corr_dims(spec);
    let mut cache = Vec::with_capacity(ne0);
    let mut theta = pos as f32;
    for i in 0..ne0 / 2 {
        let ff = spec.factors.as_ref().map_or(1.0, |f| f[i]);
        let theta_extrap = theta / ff;
        let theta_interp = spec.freq_scale * theta_extrap;
        let mut th = theta_interp;
        let mut mscale = spec.attn_factor;
        if spec.ext_factor != 0.0 {
            let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], 2 * i) * spec.ext_factor;
            th = theta_interp.mul_add(1.0 - ramp_mix, theta_extrap * ramp_mix);
            mscale *= yarn_mscale(spec.freq_scale);
        }
        cache.push(th.cos() * mscale);
        cache.push(th.sin() * mscale * dir.sin_sign());
        theta *= theta_scale;
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use std::hint::black_box;

    /// The factor table of LFM3's file: 32 factors of 1, then 32 of 1e30 — the first 32 pairs of
    /// a 128-value head turn and the last 32 are the identity.
    fn turning_then_still_factors() -> Vec<f32> {
        let mut factors = vec![1.0; 32];
        factors.extend([1e30_f32; 32]);
        factors
    }

    /// The engine's table is ggml's recipe bit for bit at positions the
    /// oracle sets do not reach — up to the model's context length — for
    /// both of V4.1's ropes, qwen3moe's and MiMo-V2's, both ways. V4.1's constants are
    /// its file's (`rope.freq_base`, `attention.compress_rope_freq_base`,
    /// `rope.scaling.{factor, original_context_length, yarn_beta_fast,
    /// yarn_beta_slow}`, `rope.dimension_count`), and the recipe fills a
    /// 512-value head's cache of which the tail reads its first `n_dims`;
    /// qwen3moe's rope is plain at `rope.freq_base` over the whole 128-value
    /// head, so its recipe fills 128 values; MiMo-V2's is plain at its full
    /// layers' `rope.freq_base` over the first 64 values of a 192-value head,
    /// so its recipe fills 192 (its sliding layers' base is the window row's);
    /// LFM3's full layers' is plain over the whole 128-value head with the
    /// file's factors, 32 of 1 then 32 of 1e30. Every
    /// input goes through `black_box`, so neither side is folded at compile
    /// time by a libm other than the one the other side calls.
    #[test]
    fn table_is_the_ggml_recipe_at_large_positions() {
        let specs = [
            (
                "window",
                RopeSpec::window(black_box(10_000.0), black_box(64)),
                512,
            ),
            (
                "qwen3moe",
                RopeSpec::window(black_box(10_000_000.0), black_box(128)),
                128,
            ),
            (
                "mimo2",
                RopeSpec::window(black_box(10_000_000.0), black_box(64)),
                192,
            ),
            (
                "yarn",
                RopeSpec::yarn(
                    black_box(160_000.0),
                    black_box(16.0),
                    black_box(65_536),
                    black_box(32.0),
                    black_box(1.0),
                    black_box(64),
                ),
                512,
            ),
            (
                "factored",
                RopeSpec::window(black_box(1_000_000.0), black_box(128))
                    .with_factors(&turning_then_still_factors())
                    .expect("32 + 32 factors for 64 pairs"),
                128,
            ),
        ];
        for (name, spec, ne0) in specs {
            let nd = spec.n_dims;
            let table = RopeTable::new(&spec).expect("n_dims is even");
            for pos in [0u32, 1, 1025, 65_535, 65_536, 1_048_575] {
                for dir in [Direction::Forward, Direction::Back] {
                    let mut got = Vec::new();
                    table.push(black_box(pos), dir, &mut got);
                    let want = ggml_rope_cache(&spec, black_box(pos), ne0, dir);
                    let same = got.len() == nd
                        && got
                            .iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        same,
                        "{name} p={pos} {dir:?}: table {got:?} vs recipe {:?}",
                        &want[..nd]
                    );
                }
            }
        }
    }

    /// A pair with factor 1 turns as it does with no factors, bit for bit, and a pair with factor
    /// 1e30 turns nothing: its angle is at most `pos / 1e30`, so its cos is exactly 1 and its sin
    /// below `pos / 1e30` (sin x = x that small). Mutant: `push` ignores `self.factors` (the
    /// `extrap` match yields `theta`), or the table keeps the factors in the wrong order.
    #[test]
    fn a_huge_factor_leaves_its_pair_unturned_and_a_unit_factor_changes_nothing() {
        let plain = RopeTable::new(&RopeSpec::window(1_000_000.0, 128)).expect("even n_dims");
        let spec = RopeSpec::window(1_000_000.0, 128)
            .with_factors(&turning_then_still_factors())
            .expect("64 factors for 64 pairs");
        let factored = RopeTable::new(&spec).expect("a valid factored spec");
        for pos in [1u32, 7, 1000, 262_143] {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            plain.push(pos, Direction::Forward, &mut a);
            factored.push(pos, Direction::Forward, &mut b);
            let bound = pos as f32 / 1e30;
            for pair in 0..64 {
                let (want, got) = (&a[2 * pair..2 * pair + 2], &b[2 * pair..2 * pair + 2]);
                if pair < 32 {
                    assert!(
                        want.iter()
                            .zip(got)
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                        "p={pos} pair {pair}: {got:?} vs plain {want:?}"
                    );
                } else {
                    assert!(
                        got[0] == 1.0 && got[1].abs() <= bound,
                        "p={pos} pair {pair}: {got:?} turns"
                    );
                }
            }
        }
    }

    /// Factors of 2 halve every angle exactly (a power of two scales each product, the YaRN
    /// blend and the running θ without rounding), so the table at position `2q` is the table
    /// with no factors at `q`, bit for bit — plain, and under YaRN, where the blend reads the
    /// divided angle on both sides. Mutant: the YaRN branch of `push` blends `theta * mix`, the
    /// undivided angle, or the division happens after `freq_scale`.
    #[test]
    fn factors_of_two_halve_the_position() {
        let specs = [
            RopeSpec::window(black_box(10_000.0), black_box(64)),
            RopeSpec::yarn(
                black_box(160_000.0),
                black_box(16.0),
                black_box(65_536),
                black_box(32.0),
                black_box(1.0),
                black_box(64),
            ),
        ];
        for (k, spec) in specs.into_iter().enumerate() {
            let halved = spec
                .clone()
                .with_factors(&[2.0; 32])
                .expect("32 factors for 32 pairs");
            let (plain, factored) = (
                RopeTable::new(&spec).expect("even n_dims"),
                RopeTable::new(&halved).expect("a valid factored spec"),
            );
            for q in [0u32, 1, 5, 1000, 100_000] {
                let (mut a, mut b) = (Vec::new(), Vec::new());
                plain.push(q, Direction::Forward, &mut a);
                factored.push(2 * q, Direction::Forward, &mut b);
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "spec {k} q={q}: {b:?} vs {a:?}"
                );
            }
        }
    }

    /// A factor table ggml would divide wrongly is refused by name, where the spec is built and
    /// again where the table is: one value too few or too many, and a zero, negative, NaN or
    /// infinite factor. Mutant: drop the `is_finite() && > 0.0` test of `check_factors`, or the
    /// `check_factors` call of `RopeTable::new`.
    #[test]
    fn a_factor_table_that_ggml_would_divide_wrongly_is_refused_by_name() {
        let spec = RopeSpec::window(10_000.0, 8);
        let cases: [(&[f32], &str); 6] = [
            (&[1.0; 3], "3 frequency factors for a rope of 8 values"),
            (&[1.0; 5], "5 frequency factors for a rope of 8 values"),
            (&[1.0, 0.0, 1.0, 1.0], "frequency factor 1 is 0;"),
            (&[1.0, 1.0, -2.0, 1.0], "frequency factor 2 is -2;"),
            (&[f32::NAN, 1.0, 1.0, 1.0], "frequency factor 0 is NaN;"),
            (
                &[1.0, 1.0, 1.0, f32::INFINITY],
                "frequency factor 3 is inf;",
            ),
        ];
        for (factors, want) in cases {
            let err = spec
                .clone()
                .with_factors(factors)
                .expect_err("a refused table")
                .to_string();
            assert!(err.contains(want), "{factors:?}: {err}");
            let mut by_hand = spec.clone();
            by_hand.factors = Some(factors.into());
            let err = RopeTable::new(&by_hand)
                .expect_err("a refused table")
                .to_string();
            assert!(err.contains(want), "{factors:?} by hand: {err}");
        }
    }

    /// The reference cache fills `ne0 / 2` pairs, so it needs that many factors, and refuses by
    /// name when the spec has fewer. Mutant: drop the assertion (the index panics unnamed).
    #[test]
    #[should_panic(expected = "ggml_rope_cache: 3 frequency factors for a cache of 4 pairs")]
    fn the_reference_cache_needs_a_factor_for_every_pair_it_fills() {
        let mut spec = RopeSpec::window(10_000.0, 8);
        spec.factors = Some([1.0f32; 3].as_slice().into());
        let _ = ggml_rope_cache(&spec, 1, 8, Direction::Forward);
    }
}
