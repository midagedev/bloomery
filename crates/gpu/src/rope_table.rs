//! ggml's rope table on the host, and the unfused rotation core of the NORM
//! mode kernels. Shared by every architecture that rotates like ggml: the
//! table is ggml's recipe as ik's CPU build compiles it
//! ([`ggml_rope_cache`]), computed once per position ([`RopeTable`]); a
//! kernel reads `[cos_0, sin_0, cos_1, …]` per token and turns pair `i` with
//! table pair `i`. Which values form pair `i` and how they turn is the
//! kernel's: adjacent `(2i, 2i + 1)` through [`rope_pair_rn`] in NORM mode,
//! `(i, i + n_dims/2)` through `rope_neox::neox_pair` in NEOX mode — the
//! table is the same.

use crate::GpuError;
use cuda_device::float::{add_rn_f32, mul_rn_f32};

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
/// arguments after the mode). ik's V4.1 graph uses two
/// (`build_deepseek4.cpp`): [`RopeSpec::window`] and [`RopeSpec::yarn`];
/// a plain rope over a whole head is [`RopeSpec::window`] at the head width.
#[derive(Clone, Copy, Debug, PartialEq)]
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
        }
    }
}

/// One site's cos/sin table at any position: [`ggml_rope_cache`]'s values,
/// with what does not depend on the position — `theta_scale`, the YaRN ramp
/// of each pair, the magnitude factor — computed once, so a position costs
/// per pair two multiplies, one fused multiply-add under YaRN, a `sin` and a
/// `cos`.
#[derive(Clone, Debug)]
pub struct RopeTable {
    n_dims: usize,
    freq_scale: f32,
    theta_scale: f32,
    /// Per pair, `(ramp_mix, 1 − ramp_mix)`; empty without YaRN.
    ramp: Vec<(f32, f32)>,
    /// `attn_factor`, times ggml's `1 + 0.1·ln(1/freq_scale)` under YaRN.
    mscale: f32,
}

impl RopeTable {
    /// The table of `spec`; `n_dims` must be even and at least 2.
    pub fn new(spec: &RopeSpec) -> Result<RopeTable, GpuError> {
        let nd = spec.n_dims;
        if nd < 2 || !nd.is_multiple_of(2) {
            return Err(GpuError::Shape {
                what: "RopeTable::new",
                detail: format!("n_dims must be even and at least 2, got {nd}"),
            });
        }
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
            let interp = self.freq_scale * theta;
            let th = match self.ramp.get(i) {
                Some(&(mix, keep)) => interp.mul_add(keep, theta * mix),
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
/// first `n_dims`. The verification side's reading of the recipe; the
/// engine's tables come from [`RopeTable`], which the unit test pins to it.
#[must_use]
pub fn ggml_rope_cache(spec: &RopeSpec, pos: u32, ne0: usize, dir: Direction) -> Vec<f32> {
    let theta_scale = theta_scale(spec);
    let corr_dims = ggml_rope_yarn_corr_dims(spec);
    let mut cache = Vec::with_capacity(ne0);
    let mut theta = pos as f32;
    for i in 0..ne0 / 2 {
        let theta_extrap = theta;
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

// ------------------------------------------------------------------ cores

/// One pair turned by `(c, s)`:
/// `y0 = x0·c − x1·s`, `y1 = x0·s + x1·c`, each product and each sum rounded
/// on its own. ik's CPU rope rotates unfused; the compiler contracts a plain
/// `a*b − c*d` into an FMA, so every op here is an explicit round-to-nearest
/// intrinsic, which it never contracts. The inverse turn is this core on a
/// table with `s` negated.
#[inline(always)]
pub fn rope_pair_rn(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    let y0 = add_rn_f32(mul_rn_f32(x0, c), -mul_rn_f32(x1, s));
    let y1 = add_rn_f32(mul_rn_f32(x0, s), mul_rn_f32(x1, c));
    (y0, y1)
}

#[cfg(test)]
mod tests {
    use super::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use std::hint::black_box;

    /// The engine's table is ggml's recipe bit for bit at positions the
    /// oracle sets do not reach — up to the model's context length — for
    /// both of V4.1's ropes and qwen3moe's, both ways. V4.1's constants are
    /// its file's (`rope.freq_base`, `attention.compress_rope_freq_base`,
    /// `rope.scaling.{factor, original_context_length, yarn_beta_fast,
    /// yarn_beta_slow}`, `rope.dimension_count`), and the recipe fills a
    /// 512-value head's cache of which the tail reads its first `n_dims`;
    /// qwen3moe's rope is plain at `rope.freq_base` over the whole 128-value
    /// head, so its recipe fills 128 values. Every
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
}
