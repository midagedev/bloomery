//! GPU gate for ERNIE 5.5 Flash Preview's trunk at the released file's shapes: four
//! streams of 4,096 values (K = 16,384), a dense block of 14,336, 201,216 logits.
//! The kernels are `bloomery_gpu::hc_f32_pieces` (the F32 hyper-connection mix over
//! 512-value pieces with ERNIE's RMS factor, and the head's four-row mix under V4's),
//! the common post and fold beside them, and `bloomery_gpu::multimax`; the components
//! are `bloomery_gpu_ernie55`'s `Trunk` and `Head` over them. Seeded inputs, no model
//! file: what the components load is a synthetic one-layer GGUF the gate writes and
//! loads through the engine's own path (`Weights::load_where`). The rules are
//! `tools/ref/ernie55_ref.py`'s (llama.cpp PR #31925's graph), restated in f64 and,
//! where a clause is bit for bit, in f32 op by op.
//!
//! Three kinds of clause, as `gate_hy4_hc` has them:
//! - **bound**: the launch against the rule in f64, within a bound this file derives
//!   from the kernel's own rounding count (`u = 2^-24`; each bound's comment counts its
//!   roundings) — never a tuned band;
//! - **bit**: the launch against the same rule in f32, every op rounded as the
//!   kernel's contract names it;
//! - **fence**: a non-finite value is refused by the fault word, a launcher refuses a
//!   bad shape by name, a captured graph replays the eager run bit for bit.
//!
//! The pre (`hc_pre_f32_rmsadd`) at K = 16,384 for 1, 5 and 13 tokens (the first
//! group one token) and 20 (the first group four): the mixes within the dot's rounding
//! bound of the f64 rule and bit for bit the host transcription of the kernel's lane
//! walk; `pre` and `post` within their rounding bound of the f64 gates over the
//! device's own mixes and the whole 24-value row bit for bit `hc_host::hc_pre_f32`
//! over them (the Sinkhorn `comb` has no f64 clause: twenty normalizations have no
//! rigorous first-order bound short of a contraction argument, and its arithmetic is
//! V4.1's `hc_pre_lane`, lifted unchanged). The RMS factor is told from V4.1's: on
//! streams a thousandth the unit, where the epsilon weighs, the mixes sit inside
//! ERNIE's bound (`raw / (sqrt(mean) + eps)`) and over a hundred bound-widths from V4's
//! (`raw · rsqrt(mean + eps)`); the head's clause is the other way about. A NaN
//! stream, and streams whose squares overflow, make that token's 48 outputs NaN (never
//! the zeros a division by infinity gives), raise `F32Product`, and leave every other
//! token's bits. A captured launch replays; a short output and a scratch of too few
//! token groups are refused by name.
//!
//! The head (`hc_head_f32`, then the common `ds41_hc_fold`) at the same cases: the four
//! mixes under V4's factor, the four `pre` gates, the row's other twenty values zero;
//! the fold bit for bit the host's fused multiply-adds over the device's gates and
//! within bound of the f64 fold of them.
//!
//! The post (`ds41_hc_post`) at the hidden width 4096: within the rounding bound of the
//! f64 rule, bit for bit the host's fused multiply-adds, its fold output bit for bit.
//!
//! MultiMax (`multimax_rows`) over 201,216 logits: the whole rule within its rounding
//! bound and bit for bit op by op; each of the four terms alone (the other three's
//! edges put out of the logits' range, their weights still nonzero), the same two
//! checks; a NaN, a +inf and a -inf logit each raise `Logit` and stay non-finite while
//! every other logit keeps its bits; a captured launch replays; a shape refusal.
//!
//! The trunk (`ernie55::Trunk`) over the synthetic file for 5 tokens (the m-column arm)
//! and 13 (the wide arm): the embedding row repeated into the four streams bit for bit;
//! each site's `pre`/`post` gates, fold and norm and its post against the f64 rule; the
//! second site reading the first site's post and the other site's weights; the dense
//! SiTU block on FP8_B128 within the bound the projections' roundings and the rule's
//! slopes give. The head (`ernie55::Head`) over the same file: the normed rows and the
//! MultiMax logits within bound of the f64 chain over the prompt's last three rows, the
//! token the argmax of the MultiMax logits (and not of the raw ones), a NaN stream
//! refused under the head's layer. The file's `output` is stored as the MXFP8 blocks the
//! loader holds a BF16 head in, so the projection's rule reads the rows the card reads.
//!
//! This gate needs an NVIDIA device.

#[cfg(feature = "ernie55")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_trunk", gate::run())
}

#[cfg(feature = "ernie55")]
mod gate {
    use bloomery_gpu::hc::{
        HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcPostArgs, HcPreScratch,
    };
    use bloomery_gpu::hc_f32::{HcF32Args, HcF32Params};
    use bloomery_gpu::hc_f32_pieces::HcF32PiecesKernels;
    use bloomery_gpu::head::Head as OutputHead;
    use bloomery_gpu::multimax::{MULTIMAX, MultiMaxArgs, MultiMaxKernels};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{
        DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_HEAD, LAYER_NONE, col_group_count,
        dense_mma,
    };
    use bloomery_gpu_ernie55::head::Head;
    use bloomery_gpu_ernie55::trunk::{Facts, Trunk};
    use bloomery_gpu_gates::hc_host::{exp_ours, hc_pre_f32};
    use bloomery_gpu_gates::rounding::{U, bf16_dot_rel, butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, GateRng, bits_equal, checks_failed, par_map, verdict, write_gguf,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use model::arch::ernie55::names::{
        OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE, OUTPUT_MULTIMAX_RANGES, OUTPUT_MULTIMAX_TS,
    };
    use model::arch::names::{
        Sub, attn_norm, ffn_down, ffn_gate, ffn_norm, ffn_up, hc_base, hc_fn, hc_scale, output,
        output_norm, token_embd,
    };
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// Values of a row, at the released file's width.
    const HIDDEN: usize = 4096;
    /// Values of a token's four streams: the mixes' K.
    const K: usize = HC_STREAMS * HIDDEN;
    /// Pieces of K one token's mix walks.
    const NP: usize = K / HC_PIECE;
    /// The dense block's width.
    const FF: usize = 14_336;
    /// The vocabulary of the MultiMax clauses.
    const VOCAB: usize = 201_216;
    /// The synthetic file's vocabulary and embedding rows: the head's projection is the
    /// common head's own (its clauses are `gate_head_gpu`'s) and a row count is no shape
    /// a clause here reads.
    const FILE_ROWS: usize = 2048;
    /// The rows the head component projects a run.
    const HEAD_ROWS: usize = 3;
    /// The tokens of the head's prompt chunk.
    const HEAD_TOKENS: usize = 5;
    /// The most tokens a mix case holds, and a trunk walk.
    const MAX_TOKENS: usize = 20;
    const WALK_TOKENS: usize = 13;
    /// The header's `hyper_connection.epsilon`, `.sinkhorn_iterations` and
    /// `attention.layer_norm_rms_epsilon`.
    const HC_EPS: f32 = 1e-6;
    const ITERS: u32 = 20;
    const RMS_EPS: f32 = 1e-6;
    /// `activation.situ_beta` and the device rule's linear beta.
    const SITU_BETA: f32 = 4.0;
    const SITU_LINEAR_BETA: f32 = 25.0;
    /// Second-order terms of every first-order bound below are at most
    /// `(rounding count · u)^2` of it; a bound carries this factor.
    const SECOND_ORDER: f64 = 1.0 + 1e-3;
    /// Roundings one product of a mix's dot passes through: its lane's sixteen chained
    /// fused multiply-adds, the five butterfly levels, the `NP - 1` piece sums.
    const DOT_ROUNDINGS: usize = 16 + 5 + (NP - 1);
    /// SiTU's roundings on a relative scale: two `tanh` of at most 2 ulp (4u each) and
    /// the rounded arguments' 2u each, σ's `exp` (4u), add (u) and divide (u), the two
    /// scalings and two products (4u).
    const SITU_ROUNDINGS: f64 = 12.0 + 6.0 + 2.0 + 4.0;
    /// MultiMax's roundings on the magnitude of its terms: the two linear terms'
    /// subtract, product and add (3 each), the two squares' subtract, two products and
    /// add (4 each).
    const MULTIMAX_ROUNDINGS: f64 = 3.0 + 3.0 + 4.0 + 4.0;
    const _: () = assert!(K.is_multiple_of(HC_PIECE) && NP == 32 && HC_MAX_TOKENS == 8);

    /// Which RMS factor a mix's rule puts on its dots: ERNIE's, the epsilon added
    /// outside the root (`build_hc_pre_ernie`), or V4's `rsqrt(mean + eps)`
    /// (`build_hc_head`, every entry of `hc.rs`).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Rule {
        RmsAdd,
        Rsqrt,
    }

    impl Rule {
        /// The same rule in f32, op for op as the kernels round it: the mean and the root;
        /// then ERNIE's division by `root + eps`, or V4's reciprocal of `mean + eps` under
        /// the root and the product.
        fn mix_f32(self, raw: f32, squares: f32) -> f32 {
            match self {
                Rule::RmsAdd => raw / ((squares / K as f32).sqrt() + RMS_EPS),
                Rule::Rsqrt => raw * (1.0 / (squares / K as f32 + RMS_EPS).sqrt()),
            }
        }
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{name} {detail} {}", verdict(pass));
        pass
    }

    /// Every `|dev − want| <= bound`, all finite, and the largest `|dev − want| / bound`.
    /// A value, rule value or bound that is not a number is no pass and an infinite ratio:
    /// `f64::max` and a comparison both skip a NaN, so neither may carry the verdict or
    /// the number alone.
    fn within(dev: &[f32], want: &[f64], bound: &[f64]) -> (bool, f64) {
        if dev.len() != want.len() || want.len() != bound.len() {
            return (false, f64::INFINITY);
        }
        let mut ok = true;
        let mut worst = 0.0f64;
        for ((&d, &w), &b) in dev.iter().zip(want).zip(bound) {
            let err = (f64::from(d) - w).abs();
            let broken = !d.is_finite() || err.is_nan() || b.is_nan();
            if broken || err > b {
                ok = false;
            }
            worst = worst.max(if broken {
                f64::INFINITY
            } else if b > 0.0 {
                err / b
            } else if err > 0.0 {
                f64::INFINITY
            } else {
                0.0
            });
        }
        (ok, worst)
    }

    /// The largest `|dev − other| / bound`: how many bound-widths a wrong rule's values
    /// lie from the device's. A value that is not a number is infinitely far.
    fn widths(dev: &[f32], other: &[f64], bound: &[f64]) -> f64 {
        dev.iter()
            .zip(other)
            .zip(bound)
            .filter(|&(_, &b)| b > 0.0 || b.is_nan())
            .map(|((&d, &o), &b)| {
                let apart = (f64::from(d) - o).abs() / b;
                if apart.is_nan() { f64::INFINITY } else { apart }
            })
            .fold(0.0, f64::max)
    }

    /// A launcher's refusal that names `what`, as `Shape`.
    fn named(r: Result<(), GpuError>, what: &str) -> bool {
        matches!(r, Err(GpuError::Shape { what: w, .. }) if w == what)
    }

    fn fill(rng: &mut GateRng, n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|_| rng.uni(-amp, amp)).collect()
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn relu(z: f64) -> f64 {
        z.max(0.0)
    }

    fn argmax_of(v: &[f32]) -> usize {
        let mut hi = 0;
        for (i, x) in v.iter().enumerate() {
            if x.total_cmp(&v[hi]).is_gt() {
                hi = i;
            }
        }
        hi
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn layer_fault(layer: u32, site: FaultSite) -> Option<Fault> {
        Some(Fault::at(layer, site))
    }

    // ------------------------------------------------------------ the rules

    /// One token's dots against `rows` rows of `w` and its squares, in f64 (the f32
    /// products are exact there): per row the dot and the sum of the absolute terms.
    struct Dots {
        dot: Vec<f64>,
        abs: Vec<f64>,
        squares: f64,
    }

    fn dots64(w: &[f32], x: &[f32], rows: usize) -> Dots {
        let squares = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let (mut dot, mut abs) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
        for r in 0..rows {
            let (mut d, mut a) = (0.0f64, 0.0f64);
            for (&wv, &xv) in w[r * K..][..K].iter().zip(x) {
                let p = f64::from(wv) * f64::from(xv);
                d += p;
                a += p.abs();
            }
            dot.push(d);
            abs.push(a);
        }
        Dots { dot, abs, squares }
    }

    /// Row `row`'s scaled mix under `rule` in f64, and its bound. The dot is within
    /// `γ(DOT_ROUNDINGS)` of its absolute terms (any order of its terms and these
    /// roundings); the squares, all terms positive, within `γ(DOT_ROUNDINGS)` relative;
    /// K is a power of two so the mean is exact. ERNIE: the root halves that and rounds
    /// once, `+ eps` and the divide once each, so the divisor is within
    /// `γ/2 + 2u` and the quotient within `γ/2 + 3u` of itself plus the dot's error over
    /// the divisor. V4: `+ eps` (inside the root), the root, the reciprocal and the
    /// product, the scale within `γ/2 + 3u` and the product one more.
    fn mix_ref(rule: Rule, d: &Dots, row: usize) -> (f64, f64) {
        let g = gamma(DOT_ROUNDINGS);
        let e_raw = g * d.abs[row];
        let mean = d.squares / K as f64;
        let eps = f64::from(RMS_EPS);
        match rule {
            Rule::RmsAdd => {
                let denom = mean.sqrt() + eps;
                let mix = d.dot[row] / denom;
                (
                    mix,
                    (e_raw / denom + mix.abs() * (g / 2.0 + 3.0 * U)) * SECOND_ORDER,
                )
            }
            Rule::Rsqrt => {
                let scale = 1.0 / (mean + eps).sqrt();
                let mix = d.dot[row] * scale;
                (
                    mix,
                    (e_raw * scale + mix.abs() * (g / 2.0 + 4.0 * U)) * SECOND_ORDER,
                )
            }
        }
    }

    /// The error of `σ(v)` as the kernel computes it, the affine's input `v` taken as
    /// given: the `fma` rounds `v` (relative u), the f64 exponential rounds to f32
    /// (u), `1 + e` and the divide one each. `σ` has slope `g(1 − g)`, so
    /// `g(1 − g)·u(|v| + 1) + 2u·g`.
    fn sig_err(v: f64) -> f64 {
        let g = sigmoid(v);
        g * (1.0 - g) * U * (v.abs() + 1.0) + 2.0 * U * g
    }

    /// `pre = σ(v) + eps` for `v` known within `ev` before the kernel's roundings, and
    /// the bound: the slope times `ev`, the sigmoid's own error, the last add.
    fn pre_ref(v: f64, ev: f64, eps: f64) -> (f64, f64) {
        let g = sigmoid(v);
        let val = g + eps;
        (
            val,
            (g * (1.0 - g) * ev + sig_err(v) + U * val) * SECOND_ORDER,
        )
    }

    /// `post = 2σ(v)`: the doubling is exact.
    fn post_ref(v: f64, ev: f64) -> (f64, f64) {
        let g = sigmoid(v);
        (
            2.0 * g,
            2.0 * (g * (1.0 - g) * ev + sig_err(v)) * SECOND_ORDER,
        )
    }

    /// One gate per stream: its f64 value and its bound.
    type Gates64 = [(f64, f64); HC_STREAMS];

    /// A row's pre and post gates in f64 from the f32 mixes `mix` (exact inputs) and
    /// the three scales and the offsets, and their bounds.
    fn gates64(mix: &[f32], sc: [f32; 3], base: &[f32], eps: f32) -> (Gates64, Gates64) {
        let v = |i: usize, s: f32| f64::from(mix[i]) * f64::from(s) + f64::from(base[i]);
        (
            std::array::from_fn(|i| pre_ref(v(i, sc[0]), 0.0, f64::from(eps))),
            std::array::from_fn(|i| post_ref(v(HC_STREAMS + i, sc[1]), 0.0)),
        )
    }

    /// The kernels' mixes in f32, op for op (`hc_f32_pieces`' module doc, steps 1-4):
    /// lane `l`'s sixteen fused multiply-adds of a piece in (s, j) order from 0, the
    /// butterfly, the squares on warp 0 the same way, both summed over the pieces in
    /// ascending order from piece 0's value, then `rule`. `rows` rows of `w` (the rest
    /// of a 24-wide row zero).
    fn piece_mixes(w: &[f32], x: &[f32], rows: usize, t: usize, rule: Rule) -> Vec<f32> {
        let at = |p: usize, lane: usize, i: usize| HC_PIECE * p + 4 * lane + 128 * (i / 4) + i % 4;
        let mut mixes = vec![0.0f32; t * HC_MIX];
        for tt in 0..t {
            let xt = &x[tt * K..][..K];
            let mut squares = 0.0f32;
            let mut sums = vec![0.0f32; rows];
            for p in 0..NP {
                let v: [[f32; 16]; 32] =
                    std::array::from_fn(|l| std::array::from_fn(|i| xt[at(p, l, i)]));
                let sq = butterfly(std::array::from_fn(|l| {
                    v[l].iter().fold(0.0f32, |acc, &x| x.mul_add(x, acc))
                }));
                squares = if p == 0 { sq } else { squares + sq };
                for (r, sum) in sums.iter_mut().enumerate() {
                    let part = butterfly(std::array::from_fn(|l| {
                        let mut f = 0.0f32;
                        for (i, &x) in v[l].iter().enumerate() {
                            f = w[r * K + at(p, l, i)].mul_add(x, f);
                        }
                        f
                    }));
                    *sum = if p == 0 { part } else { *sum + part };
                }
            }
            for (r, &raw) in sums.iter().enumerate() {
                mixes[tt * HC_MIX + r] = rule.mix_f32(raw, squares);
            }
        }
        mixes
    }

    /// The post of one value in f32 as the card fuses it: `fma(x, post_i, comb_0i·r_0)`
    /// then `fma(comb_ji, r_j, ·)` for j = 1..3.
    fn post_host(x: f32, res: [f32; 4], post: [f32; 4], comb: &[f32]) -> [f32; 4] {
        std::array::from_fn(|i| {
            let mut s = x.mul_add(post[i], comb[i] * res[0]);
            for j in 1..4 {
                s = comb[4 * j + i].mul_add(res[j], s);
            }
            s
        })
    }

    /// The fold of four stream values by `pre`: `o_0·pre_0`, then one fused
    /// multiply-add per further stream.
    fn fold_host(o: [f32; 4], pre: [f32; 4]) -> f32 {
        let mut y = o[0] * pre[0];
        for s in 1..4 {
            y = o[s].mul_add(pre[s], y);
        }
        y
    }

    /// MultiMax in f32, each op rounded on its own (a Rust `a + b * c` is two roundings),
    /// the `relu` the reference's.
    fn multimax32(l: f32, r: &[f32; MULTIMAX], t: &[f32; MULTIMAX]) -> f32 {
        let relu = |z: f32| if z.is_nan() | (z > 0.0) { z } else { 0.0 };
        let (a, b) = (relu(r[0] - l), relu(l - r[1]));
        let (c, d) = (relu(r[2] - l), relu(l - r[3]));
        let y = l + t[0] * a;
        let y = y + t[1] * b;
        let y = y + t[2] * (c * c);
        y + t[3] * (d * d)
    }

    /// MultiMax in f64 for a logit `l` known within `bl` (before the kernel's own
    /// roundings), and the bound: the kernel's roundings on the magnitude of the logit
    /// and its terms, and the slope's reach over `bl` — each `relu` is monotone and an
    /// indicator is at most 1, so `1 + |t0| + |t1| + 2|t2|·c + 2|t3|·d` with `c` and `d`
    /// taken at the interval's edges bounds the slope.
    fn multimax64(l: f64, bl: f64, r: &[f32; MULTIMAX], t: &[f32; MULTIMAX]) -> (f64, f64) {
        let (r, t) = (r.map(f64::from), t.map(f64::from));
        let terms = [
            t[0] * relu(r[0] - l),
            t[1] * relu(l - r[1]),
            t[2] * relu(r[2] - l).powi(2),
            t[3] * relu(l - r[3]).powi(2),
        ];
        let y = l + terms.iter().sum::<f64>();
        let mag = l.abs() + terms.iter().map(|x| x.abs()).sum::<f64>();
        let slope = 1.0
            + t[0].abs()
            + t[1].abs()
            + 2.0 * t[2].abs() * relu(r[2] - (l - bl))
            + 2.0 * t[3].abs() * relu((l + bl) - r[3]);
        (
            y,
            (MULTIMAX_ROUNDINGS * U * mag + slope * bl) * SECOND_ORDER,
        )
    }

    /// SiTU in f64, `ernie55_ref.py`'s `situ`: `β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`.
    fn situ64(g: f64, u: f64) -> f64 {
        let (b, l) = (f64::from(SITU_BETA), f64::from(SITU_LINEAR_BETA));
        let a = (g / b).tanh() * b;
        let v = (u / l).tanh() * l;
        (a * sigmoid(g)) * v
    }

    /// `rms_norm(x, gain)` of one row `x` known within `ex` per value: the value
    /// `x_i·gain_i / sqrt(mean + eps)` and its bound. The sum of squares, positive terms
    /// in any order, is within `γ(n)`; the scale's root halves it; the mean, the add, the
    /// root, the reciprocal and the two products are six more roundings. The input
    /// error moves a value by `gain·(ex·s + |x|·ds)` with `ds <= s³·Σ|x||ex| / n`.
    fn norm_ref(x: &[f64], ex: &[f64], gain: &[f32], eps: f32) -> (Vec<f64>, Vec<f64>) {
        let n = x.len() as f64;
        let mean = x.iter().map(|v| v * v).sum::<f64>() / n;
        let s = 1.0 / (mean + f64::from(eps)).sqrt();
        let ds = s.powi(3) * x.iter().zip(ex).map(|(v, e)| v.abs() * e).sum::<f64>() / n;
        let rel = gamma(x.len()) / 2.0 + 6.0 * U;
        let want: Vec<f64> = x
            .iter()
            .zip(gain)
            .map(|(v, &g)| v * f64::from(g) * s)
            .collect();
        let bound = (0..x.len())
            .map(|i| {
                let g = f64::from(gain[i]).abs();
                (g * (ex[i] * s + x[i].abs() * ds) + rel * want[i].abs()) * SECOND_ORDER
            })
            .collect();
        (want, bound)
    }

    // ------------------------------------------------------------ the kernels

    struct Cx {
        gpu: Gpu,
        pieces: HcF32PiecesKernels,
        hc: HcKernels,
        multimax: MultiMaxKernels,
    }

    /// The token streams of a mix case: `tokens` tokens of K values uniform in
    /// `±amp`, each token's amplitude its own multiple so the norm differs by token.
    fn streams(rng: &mut GateRng, tokens: usize, amp: f32) -> Vec<f32> {
        const SCALE: [f32; 5] = [1.0, 8.0, 0.25, 3.0, 0.5];
        let mut x = fill(rng, tokens * K, amp);
        for (t, row) in x.as_chunks_mut::<K>().0.iter_mut().enumerate() {
            for v in row {
                *v *= SCALE[t % 5];
            }
        }
        x
    }

    /// The (tokens, first group) of the mix cases: a lone token, one past the first
    /// group, two full groups behind a one-token group, and a four-token first group.
    const CASES: [(usize, usize); 4] = [(1, 1), (5, 1), (13, 1), (20, 4)];

    fn read(st: &cuda_core::CudaStream, b: &DeviceBuffer<f32>) -> Result<Vec<f32>, GateError> {
        Ok(b.to_host_vec(st)?)
    }

    fn poison(st: &cuda_core::CudaStream, b: &mut DeviceBuffer<f32>) -> Result<(), GateError> {
        b.copy_from_host(st, &vec![f32::NAN; b.len()])?;
        Ok(())
    }

    fn nan_row(v: &[f32], t: usize) -> bool {
        v[t * HC_MIX..][..HC_MIX].iter().all(|x| x.is_nan())
    }

    fn same_row(a: &[f32], b: &[f32], t: usize) -> bool {
        bits_equal(&a[t * HC_MIX..][..HC_MIX], &b[t * HC_MIX..][..HC_MIX])
    }

    // ---------------------------------------------------------------- the pre

    fn pre_clauses(cx: &Cx) -> Result<bool, GateError> {
        const WHAT: &str = "HcF32PiecesKernels::enqueue_pre";
        let st = cx.gpu.stream();
        let mut rng = GateRng(0x5eed_e550_0001);
        let x = streams(&mut rng, MAX_TOKENS, 1.0);
        let w = fill(&mut rng, HC_MIX * K, 0.05);
        let scale = [0.45f32, 0.35, 0.3];
        let base = fill(&mut rng, HC_MIX, 1.0);
        let dw = DeviceTensor::upload(st, &w, HC_MIX, K)?;
        let dscale = DeviceBuffer::from_host(st, &scale)?;
        let dbase = DeviceBuffer::from_host(st, &base)?;
        let params = HcF32Params {
            w: &dw,
            scale: &dscale,
            base: &dbase,
            eps: HC_EPS,
            iters: ITERS,
        };
        let dx = DeviceBuffer::from_host(st, &x)?;
        let mut scratch = HcPreScratch::with_groups(st, K, col_group_count(1, MAX_TOKENS))?;
        let pre = |scratch: &mut HcPreScratch,
                   src: &DeviceBuffer<f32>,
                   (t, lead): (usize, usize),
                   mixes: &mut DeviceBuffer<f32>,
                   hc: &mut DeviceBuffer<f32>| {
            let a = HcF32Args {
                params: &params,
                x: src,
                tokens: t,
                rms_eps: RMS_EPS,
                fault: cx.gpu.unlabelled_sink(),
            };
            cx.pieces.enqueue_pre(st, &a, lead, scratch, mixes, hc)
        };
        let mut ok = true;
        let mut five = (vec![], vec![]);
        for (t, lead) in CASES {
            let tag = format!("pre t={t} lead={lead}");
            let mut mixes = DeviceBuffer::zeroed(st, t * HC_MIX)?;
            let mut hc = DeviceBuffer::zeroed(st, t * HC_MIX)?;
            cx.gpu.take_fault()?;
            pre(&mut scratch, &dx, (t, lead), &mut mixes, &mut hc)?;
            st.synchronize()?;
            let clean = cx.gpu.take_fault()?.is_none();
            let (m_dev, hc_dev) = (read(st, &mixes)?, read(st, &hc)?);
            let (mut want, mut bound) = (vec![], vec![]);
            for tt in 0..t {
                let d = dots64(&w, &x[tt * K..][..K], HC_MIX);
                for r in 0..HC_MIX {
                    let (m, e) = mix_ref(Rule::RmsAdd, &d, r);
                    want.push(m);
                    bound.push(e);
                }
            }
            let (in_bound, worst) = within(&m_dev, &want, &bound);
            ok &= clause(
                &format!("{tag} mixes bound"),
                in_bound && clean,
                &format!("worst={worst:.3} of the bound fault_free={clean}"),
            );
            let host = piece_mixes(&w, &x, HC_MIX, t, Rule::RmsAdd);
            ok &= clause(
                &format!("{tag} mixes bits"),
                bits_equal(&m_dev, &host),
                "the lane walk, op by op",
            );
            // The split over the device's own mixes: pre and post against the f64
            // gates, the whole row against the f32 rule.
            let (mut dev, mut gw, mut gb) = (vec![], vec![], vec![]);
            let mut rows = Vec::with_capacity(t * HC_MIX);
            for tt in 0..t {
                let row = &m_dev[tt * HC_MIX..][..HC_MIX];
                let (pre_g, post_g) = gates64(row, scale, &base, HC_EPS);
                let dh = &hc_dev[tt * HC_MIX..][..HC_MIX];
                for i in 0..HC_STREAMS {
                    dev.extend([dh[i], dh[HC_STREAMS + i]]);
                    gw.extend([pre_g[i].0, post_g[i].0]);
                    gb.extend([pre_g[i].1, post_g[i].1]);
                }
                rows.extend(hc_pre_f32(row, scale, &base, HC_EPS, ITERS, exp_ours));
            }
            let (gate_ok, gworst) = within(&dev, &gw, &gb);
            ok &= clause(
                &format!("{tag} split"),
                gate_ok && bits_equal(&hc_dev, &rows),
                &format!(
                    "pre/post worst={gworst:.3} of the bound, the rows bit for bit hc_pre_f32 over the device mixes"
                ),
            );
            if (t, lead) == (5, 1) {
                five = (m_dev, hc_dev);
            }
        }

        // The RMS factor is ERNIE's, told from V4.1's: streams a thousandth the unit,
        // where the epsilon weighs.
        let small = streams(&mut rng, 3, 1.732e-3);
        let dsmall = DeviceBuffer::from_host(st, &small)?;
        let (mut ms, mut hs) = (
            DeviceBuffer::zeroed(st, 3 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 3 * HC_MIX)?,
        );
        pre(&mut scratch, &dsmall, (3, 1), &mut ms, &mut hs)?;
        st.synchronize()?;
        let m_dev = read(st, &ms)?;
        let (mut ernie, mut v4, mut bound) = (vec![], vec![], vec![]);
        for tt in 0..3 {
            let d = dots64(&w, &small[tt * K..][..K], HC_MIX);
            for r in 0..HC_MIX {
                let (m, e) = mix_ref(Rule::RmsAdd, &d, r);
                ernie.push(m);
                bound.push(e);
                v4.push(mix_ref(Rule::Rsqrt, &d, r).0);
            }
        }
        let (in_bound, _) = within(&m_dev, &ernie, &bound);
        let apart = widths(&m_dev, &v4, &bound);
        ok &= clause(
            "pre rms factor",
            in_bound && apart > 100.0,
            &format!("inside ERNIE's bound, {apart:.0} bound-widths from V4's rule"),
        );

        // A NaN stream and streams whose squares overflow have no norm: that token's
        // outputs are NaN, the fault word names the site, the other tokens do not move.
        let mut bad = x;
        bad[K + 5] = f32::NAN;
        bad[3 * K..4 * K].fill(1e20);
        let dbad = DeviceBuffer::from_host(st, &bad)?;
        let (mut mb, mut hb) = (
            DeviceBuffer::zeroed(st, 5 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 5 * HC_MIX)?,
        );
        cx.gpu.take_fault()?;
        pre(&mut scratch, &dbad, (5, 1), &mut mb, &mut hb)?;
        st.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        let (mbh, hbh) = (read(st, &mb)?, read(st, &hb)?);
        let poisoned = [1, 3].iter().all(|&t| nan_row(&mbh, t) && nan_row(&hbh, t));
        let spared = [0, 2, 4]
            .iter()
            .all(|&t| same_row(&mbh, &five.0, t) && same_row(&hbh, &five.1, t));
        ok &= clause(
            "pre refuses a NaN stream and overflowing squares",
            fault == layer_fault(LAYER_NONE, FaultSite::F32Product) && poisoned && spared,
            &format!("fault={fault:?} poisoned={poisoned} others_unmoved={spared}"),
        );

        // One captured launch replays the eager run bit for bit.
        let (mut ea, mut ha) = (
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
        );
        pre(&mut scratch, &dx, (13, 1), &mut ea, &mut ha)?;
        st.synchronize()?;
        let (m_ref, h_ref) = (read(st, &ea)?, read(st, &ha)?);
        let (mut mg, mut hg) = (
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
        );
        let graph = cx
            .gpu
            .capture(|s| pre_on(cx, s, &params, &dx, (13, 1), &mut scratch, &mut mg, &mut hg))?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            poison(st, &mut mg)?;
            poison(st, &mut hg)?;
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&read(st, &mg)?, &m_ref) && bits_equal(&read(st, &hg)?, &h_ref);
        }
        replay &= cx.gpu.take_fault()?.is_none();
        ok &= clause(
            "pre capture",
            replay,
            "one node, two exact replays over poisoned outputs",
        );

        // A short output and a scratch of too few token groups are refused by name.
        let mut short = DeviceBuffer::zeroed(st, 13 * HC_MIX - 1)?;
        let mut full = DeviceBuffer::zeroed(st, 20 * HC_MIX)?;
        let mut few = HcPreScratch::with_groups(st, K, 1)?;
        let mut m20 = DeviceBuffer::zeroed(st, 20 * HC_MIX)?;
        let refused = named(pre(&mut scratch, &dx, (13, 1), &mut short, &mut full), WHAT)
            && named(pre(&mut few, &dx, (20, 1), &mut m20, &mut full), WHAT);
        ok &= clause(
            "pre host boundary",
            refused,
            "a short mix and a scratch of one group for twenty tokens are named Shape errors",
        );
        Ok(ok)
    }

    /// One pre launch on `s` (a capture's stream or the engine's).
    #[allow(clippy::too_many_arguments)]
    fn pre_on(
        cx: &Cx,
        s: &cuda_core::CudaStream,
        params: &HcF32Params<'_>,
        x: &DeviceBuffer<f32>,
        (t, lead): (usize, usize),
        scratch: &mut HcPreScratch,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let a = HcF32Args {
            params,
            x,
            tokens: t,
            rms_eps: RMS_EPS,
            fault: cx.gpu.unlabelled_sink(),
        };
        cx.pieces.enqueue_pre(s, &a, lead, scratch, mixes, hc)
    }

    // --------------------------------------------------------------- the head

    fn head_clauses(cx: &Cx) -> Result<bool, GateError> {
        const WHAT: &str = "HcF32PiecesKernels::enqueue_head";
        let st = cx.gpu.stream();
        let mut rng = GateRng(0x5eed_e550_0002);
        let x = streams(&mut rng, MAX_TOKENS, 1.0);
        let w = fill(&mut rng, HC_STREAMS * K, 0.05);
        let scale = [1.25f32];
        let base4 = fill(&mut rng, HC_STREAMS, 1.0);
        let dw = DeviceTensor::upload(st, &w, HC_STREAMS, K)?;
        let dscale = DeviceBuffer::from_host(st, &scale)?;
        let dbase = DeviceBuffer::from_host(st, &base4)?;
        let params = HcF32Params {
            w: &dw,
            scale: &dscale,
            base: &dbase,
            eps: HC_EPS,
            iters: ITERS,
        };
        let dx = DeviceBuffer::from_host(st, &x)?;
        let mut scratch = HcPreScratch::with_groups(st, K, col_group_count(1, MAX_TOKENS))?;
        let head = |scratch: &mut HcPreScratch,
                    src: &DeviceBuffer<f32>,
                    (t, lead): (usize, usize),
                    mixes: &mut DeviceBuffer<f32>,
                    hc: &mut DeviceBuffer<f32>| {
            head_on(cx, st, &params, src, (t, lead), scratch, mixes, hc)
        };
        let mut base24 = base4;
        base24.resize(HC_MIX, 0.0);
        let zero_tail = |v: &[f32], t: usize| {
            v[t * HC_MIX + HC_STREAMS..][..HC_MIX - HC_STREAMS]
                .iter()
                .all(|x| x.to_bits() == 0)
        };
        let mut ok = true;
        let mut thirteen = (vec![], vec![]);
        for (t, lead) in [(1, 1), (13, 1), (20, 4)] {
            let tag = format!("head t={t} lead={lead}");
            let mut mixes = DeviceBuffer::zeroed(st, t * HC_MIX)?;
            let mut hc = DeviceBuffer::zeroed(st, t * HC_MIX)?;
            cx.gpu.take_fault()?;
            head(&mut scratch, &dx, (t, lead), &mut mixes, &mut hc)?;
            st.synchronize()?;
            let clean = cx.gpu.take_fault()?.is_none();
            let (m_dev, hc_dev) = (read(st, &mixes)?, read(st, &hc)?);
            let (mut dev, mut want, mut bound) = (vec![], vec![], vec![]);
            for tt in 0..t {
                let d = dots64(&w, &x[tt * K..][..K], HC_STREAMS);
                for r in 0..HC_STREAMS {
                    let (m, e) = mix_ref(Rule::Rsqrt, &d, r);
                    dev.push(m_dev[tt * HC_MIX + r]);
                    want.push(m);
                    bound.push(e);
                }
            }
            let (in_bound, worst) = within(&dev, &want, &bound);
            let tails = (0..t).all(|tt| zero_tail(&m_dev, tt) && zero_tail(&hc_dev, tt));
            ok &= clause(
                &format!("{tag} mixes bound"),
                in_bound && tails && clean,
                &format!(
                    "worst={worst:.3} of the bound, the other twenty values zero={tails} fault_free={clean}"
                ),
            );
            ok &= clause(
                &format!("{tag} mixes bits"),
                bits_equal(&m_dev, &piece_mixes(&w, &x, HC_STREAMS, t, Rule::Rsqrt)),
                "the lane walk, op by op",
            );
            // The four pre gates over the device's own mixes.
            let (mut gdev, mut gw, mut gb) = (vec![], vec![], vec![]);
            let mut rows = Vec::with_capacity(HC_STREAMS * t);
            for tt in 0..t {
                let row = &m_dev[tt * HC_MIX..][..HC_MIX];
                let sc = [scale[0], 0.0, 0.0];
                let (pre_g, _) = gates64(row, [sc[0], 0.0, 0.0], &base24, HC_EPS);
                gdev.extend_from_slice(&hc_dev[tt * HC_MIX..][..HC_STREAMS]);
                gw.extend(pre_g.iter().map(|g| g.0));
                gb.extend(pre_g.iter().map(|g| g.1));
                rows.extend_from_slice(
                    &hc_pre_f32(row, sc, &base24, HC_EPS, ITERS, exp_ours)[..HC_STREAMS],
                );
            }
            let (gate_ok, gworst) = within(&gdev, &gw, &gb);
            ok &= clause(
                &format!("{tag} gates"),
                gate_ok && bits_equal(&gdev, &rows),
                &format!(
                    "pre worst={gworst:.3} of the bound, bit for bit hc_pre_f32 over the device mixes"
                ),
            );
            // The common fold by those gates.
            let mut dy = DeviceBuffer::zeroed(st, t * HIDDEN)?;
            cx.hc.enqueue_fold(st, &dx, &hc, HIDDEN, t, &mut dy)?;
            st.synchronize()?;
            let y = read(st, &dy)?;
            let (mut bits, mut fw, mut fb) = (vec![], vec![], vec![]);
            for tt in 0..t {
                let pre4: [f32; 4] = std::array::from_fn(|s| hc_dev[tt * HC_MIX + s]);
                for d in 0..HIDDEN {
                    let o: [f32; 4] = std::array::from_fn(|s| x[tt * K + s * HIDDEN + d]);
                    bits.push(fold_host(o, pre4));
                    let terms: Vec<f64> = (0..4)
                        .map(|s| f64::from(pre4[s]) * f64::from(o[s]))
                        .collect();
                    fw.push(terms.iter().sum());
                    fb.push(4.0 * U * terms.iter().map(|v| v.abs()).sum::<f64>() * SECOND_ORDER);
                }
            }
            let (fold_ok, fworst) = within(&y, &fw, &fb);
            ok &= clause(
                &format!("{tag} fold"),
                fold_ok && bits_equal(&y, &bits),
                &format!(
                    "worst={fworst:.3} of the bound, bit for bit the host's fused multiply-adds"
                ),
            );
            if (t, lead) == (13, 1) {
                thirteen = (m_dev, hc_dev);
            }
        }

        // V4's factor, told from ERNIE's.
        let small = streams(&mut rng, 3, 1.732e-3);
        let dsmall = DeviceBuffer::from_host(st, &small)?;
        let (mut ms, mut hs) = (
            DeviceBuffer::zeroed(st, 3 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 3 * HC_MIX)?,
        );
        head(&mut scratch, &dsmall, (3, 1), &mut ms, &mut hs)?;
        st.synchronize()?;
        let m_dev = read(st, &ms)?;
        let (mut dev, mut v4, mut ernie, mut bound) = (vec![], vec![], vec![], vec![]);
        for tt in 0..3 {
            let d = dots64(&w, &small[tt * K..][..K], HC_STREAMS);
            for r in 0..HC_STREAMS {
                let (m, e) = mix_ref(Rule::Rsqrt, &d, r);
                dev.push(m_dev[tt * HC_MIX + r]);
                v4.push(m);
                bound.push(e);
                ernie.push(mix_ref(Rule::RmsAdd, &d, r).0);
            }
        }
        let (in_bound, _) = within(&dev, &v4, &bound);
        let apart = widths(&dev, &ernie, &bound);
        ok &= clause(
            "head rms factor",
            in_bound && apart > 100.0,
            &format!("inside V4's bound, {apart:.0} bound-widths from ERNIE's rule"),
        );

        // A NaN stream and overflowing squares: the token's four mixes and gates are
        // NaN, the site is raised, the other tokens do not move.
        let mut bad = x;
        bad[K + 5] = f32::NAN;
        bad[3 * K..4 * K].fill(1e20);
        let dbad = DeviceBuffer::from_host(st, &bad)?;
        let (mut mb, mut hb) = (
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
        );
        cx.gpu.take_fault()?;
        head(&mut scratch, &dbad, (13, 1), &mut mb, &mut hb)?;
        st.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        let (mbh, hbh) = (read(st, &mb)?, read(st, &hb)?);
        let four_nan =
            |v: &[f32], t: usize| v[t * HC_MIX..][..HC_STREAMS].iter().all(|x| x.is_nan());
        let poisoned = [1, 3]
            .iter()
            .all(|&t| four_nan(&mbh, t) && four_nan(&hbh, t));
        let spared = (0..13)
            .filter(|t| ![1, 3].contains(t))
            .all(|t| same_row(&mbh, &thirteen.0, t) && same_row(&hbh, &thirteen.1, t));
        ok &= clause(
            "head refuses a NaN stream and overflowing squares",
            fault == layer_fault(LAYER_NONE, FaultSite::F32Product) && poisoned && spared,
            &format!("fault={fault:?} poisoned={poisoned} others_unmoved={spared}"),
        );

        // The mix and the fold captured as two nodes replay the eager run.
        let (mut ea, mut ha) = (
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
        );
        let mut ya = DeviceBuffer::zeroed(st, 13 * HIDDEN)?;
        head(&mut scratch, &dx, (13, 1), &mut ea, &mut ha)?;
        cx.hc.enqueue_fold(st, &dx, &ha, HIDDEN, 13, &mut ya)?;
        st.synchronize()?;
        let y_ref = read(st, &ya)?;
        let (mut mg, mut hg) = (
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
            DeviceBuffer::zeroed(st, 13 * HC_MIX)?,
        );
        let mut yg = DeviceBuffer::zeroed(st, 13 * HIDDEN)?;
        let graph = cx.gpu.capture(|s| {
            head_on(cx, s, &params, &dx, (13, 1), &mut scratch, &mut mg, &mut hg)?;
            cx.hc.enqueue_fold(s, &dx, &hg, HIDDEN, 13, &mut yg)
        })?;
        let mut replay = graph.node_count() == 2;
        for _ in 0..2 {
            poison(st, &mut yg)?;
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&read(st, &yg)?, &y_ref);
        }
        replay &= cx.gpu.take_fault()?.is_none();
        ok &= clause(
            "head capture",
            replay,
            "two nodes, two exact replays over a poisoned fold",
        );

        // A weight of the wrong row count and a short output are refused by name.
        let w3 = fill(&mut rng, 3 * K, 0.05);
        let dw3 = DeviceTensor::upload(st, &w3, 3, K)?;
        let bad_params = HcF32Params {
            w: &dw3,
            scale: &dscale,
            base: &dbase,
            eps: HC_EPS,
            iters: ITERS,
        };
        let mut full = DeviceBuffer::zeroed(st, 13 * HC_MIX)?;
        let mut short = DeviceBuffer::zeroed(st, 13 * HC_MIX - 1)?;
        let refused = named(
            head_on(
                cx,
                st,
                &bad_params,
                &dx,
                (13, 1),
                &mut scratch,
                &mut full,
                &mut hb,
            ),
            WHAT,
        ) && named(
            head_on(
                cx,
                st,
                &params,
                &dx,
                (13, 1),
                &mut scratch,
                &mut short,
                &mut hb,
            ),
            WHAT,
        );
        ok &= clause(
            "head host boundary",
            refused,
            "three weight rows and a short mix are named Shape errors",
        );
        Ok(ok)
    }

    /// One head launch on `s`.
    #[allow(clippy::too_many_arguments)]
    fn head_on(
        cx: &Cx,
        s: &cuda_core::CudaStream,
        params: &HcF32Params<'_>,
        x: &DeviceBuffer<f32>,
        (t, lead): (usize, usize),
        scratch: &mut HcPreScratch,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let a = HcF32Args {
            params,
            x,
            tokens: t,
            rms_eps: RMS_EPS,
            fault: cx.gpu.unlabelled_sink(),
        };
        cx.pieces.enqueue_head(s, &a, lead, scratch, mixes, hc)
    }

    // ---------------------------------------------------------------- the post

    fn post_clauses(cx: &Cx) -> Result<bool, GateError> {
        const WHAT: &str = "HcKernels::enqueue_post";
        const T: usize = 5;
        let st = cx.gpu.stream();
        let mut rng = GateRng(0x5eed_e550_0003);
        let res = fill(&mut rng, T * K, 1.0);
        let xx = fill(&mut rng, T * HIDDEN, 1.0);
        let mut rows = Vec::with_capacity(T * HC_MIX);
        for _ in 0..T {
            for i in 0..HC_MIX {
                rows.push(match i {
                    0..4 => rng.uni(0.05, 0.95),
                    4..8 => rng.uni(0.0, 2.0),
                    _ => rng.uni(0.0, 1.0),
                });
            }
        }
        let (dres, dx, dhc) = (
            DeviceBuffer::from_host(st, &res)?,
            DeviceBuffer::from_host(st, &xx)?,
            DeviceBuffer::from_host(st, &rows)?,
        );
        let args = HcPostArgs {
            x: &dx,
            res: &dres,
            hc: &dhc,
            n_embd: HIDDEN,
            tokens: T,
        };
        let mut out = DeviceBuffer::zeroed(st, T * K)?;
        let mut fold = DeviceBuffer::zeroed(st, T * HIDDEN)?;
        cx.gpu.take_fault()?;
        cx.hc.enqueue_post(st, &args, &mut out, &mut fold)?;
        st.synchronize()?;
        let clean = cx.gpu.take_fault()?.is_none();
        let (o, f) = (read(st, &out)?, read(st, &fold)?);
        let (mut want32, mut want, mut bound) = (
            vec![0.0f32; T * K],
            vec![0.0f64; T * K],
            vec![0.0f64; T * K],
        );
        let mut fold32 = vec![0.0f32; T * HIDDEN];
        for tt in 0..T {
            let hc = &rows[tt * HC_MIX..][..HC_MIX];
            let (pre, post, comb) = (&hc[..4], &hc[4..8], &hc[8..]);
            for d in 0..HIDDEN {
                let r4: [f32; 4] = std::array::from_fn(|j| res[tt * K + j * HIDDEN + d]);
                let x = xx[tt * HIDDEN + d];
                let new = post_host(x, r4, std::array::from_fn(|i| post[i]), comb);
                for i in 0..4 {
                    let at = tt * K + i * HIDDEN + d;
                    want32[at] = new[i];
                    let mut terms = vec![f64::from(x) * f64::from(post[i])];
                    terms.extend((0..4).map(|j| f64::from(comb[4 * j + i]) * f64::from(r4[j])));
                    want[at] = terms.iter().sum();
                    bound[at] = 5.0 * U * terms.iter().map(|v| v.abs()).sum::<f64>() * SECOND_ORDER;
                }
                fold32[tt * HIDDEN + d] = fold_host(new, std::array::from_fn(|i| pre[i]));
            }
        }
        let (in_bound, worst) = within(&o, &want, &bound);
        let mut ok = clause(
            "post at hidden 4096",
            in_bound && bits_equal(&o, &want32) && clean,
            &format!("worst={worst:.3} of the bound, bit for bit the host's fused multiply-adds"),
        );
        ok &= clause(
            "post fold",
            bits_equal(&f, &fold32),
            "the new streams folded by the same pre, bit for bit",
        );
        let mut again = DeviceBuffer::zeroed(st, T * K)?;
        let mut fagain = DeviceBuffer::zeroed(st, T * HIDDEN)?;
        let graph = cx
            .gpu
            .capture(|s| cx.hc.enqueue_post(s, &args, &mut again, &mut fagain))?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            poison(st, &mut again)?;
            poison(st, &mut fagain)?;
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&read(st, &again)?, &o) && bits_equal(&read(st, &fagain)?, &f);
        }
        ok &= clause(
            "post capture",
            replay,
            "one node, two exact replays over poisoned outputs",
        );
        let mut short = DeviceBuffer::zeroed(st, T * K - 1)?;
        ok &= clause(
            "post host boundary",
            named(cx.hc.enqueue_post(st, &args, &mut short, &mut fold), WHAT),
            "a short stream output is a named Shape error",
        );
        Ok(ok)
    }

    // ------------------------------------------------------------- MultiMax

    /// The logits `logits` through `multimax_rows` at the edges `r` and weights `t`:
    /// the values back and the fault word the launch left.
    fn mm_case(
        cx: &Cx,
        logits: &[f32],
        r: [f32; MULTIMAX],
        t: [f32; MULTIMAX],
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let st = cx.gpu.stream();
        let mut y = DeviceBuffer::from_host(st, logits)?;
        let (dr, dt) = (
            DeviceBuffer::from_host(st, &r)?,
            DeviceBuffer::from_host(st, &t)?,
        );
        cx.gpu.take_fault()?;
        cx.multimax.enqueue(
            st,
            MultiMaxArgs {
                y: &mut y,
                n: logits.len(),
                ranges: &dr,
                ts: &dt,
                fault: cx.gpu.unlabelled_sink(),
            },
        )?;
        st.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        Ok((read(st, &y)?, fault))
    }

    fn multimax_clauses(cx: &Cx) -> Result<bool, GateError> {
        const WHAT: &str = "MultiMaxKernels::enqueue";
        const EDGES: [f32; MULTIMAX] = [-2.5, 1.0, 0.5, 2.0];
        const TS: [f32; MULTIMAX] = [0.4, -0.7, 0.3, -0.2];
        /// A term's edge put far outside the logits: its relu is zero.
        const OFF: f32 = 1e9;
        let st = cx.gpu.stream();
        let mut rng = GateRng(0x5eed_e550_0004);
        let logits = fill(&mut rng, VOCAB, 6.0);
        let mut cases = vec![("all four terms".to_string(), EDGES)];
        for i in 0..MULTIMAX {
            let mut r = [-OFF, OFF, -OFF, OFF];
            r[i] = EDGES[i];
            cases.push((format!("term {i} alone"), r));
        }
        let mut ok = true;
        for (name, r) in &cases {
            let (y, fault) = mm_case(cx, &logits, *r, TS)?;
            let want32: Vec<f32> = logits.iter().map(|&l| multimax32(l, r, &TS)).collect();
            let (mut want, mut bound) = (Vec::with_capacity(VOCAB), Vec::with_capacity(VOCAB));
            for &l in &logits {
                let (v, b) = multimax64(f64::from(l), 0.0, r, &TS);
                want.push(v);
                bound.push(b);
            }
            let (in_bound, worst) = within(&y, &want, &bound);
            // Every term's active and silent branch is in the logits' range.
            let active = |i: usize| {
                logits
                    .iter()
                    .filter(|&&l| match i {
                        0 => l < r[0],
                        1 => l > r[1],
                        2 => l < r[2],
                        _ => l > r[3],
                    })
                    .count()
            };
            let alone = name.starts_with("term");
            let both = (0..MULTIMAX)
                .filter(|&i| !alone || name == &format!("term {i} alone"))
                .all(|i| (1..VOCAB).contains(&active(i)));
            ok &= clause(
                &format!("multimax {name}"),
                in_bound && bits_equal(&y, &want32) && fault.is_none() && both,
                &format!(
                    "worst={worst:.3} of the bound, bit for bit op by op, both branches in range={both}"
                ),
            );
        }

        // A non-finite logit has no MultiMax: the site is raised, the output stays
        // non-finite, every other logit keeps its bits.
        let hits = [17usize, 100_000, 150_001];
        let mut bad = logits.clone();
        bad[hits[0]] = f32::NAN;
        bad[hits[1]] = f32::INFINITY;
        bad[hits[2]] = f32::NEG_INFINITY;
        let (clean_y, _) = mm_case(cx, &logits, EDGES, TS)?;
        let (y, fault) = mm_case(cx, &bad, EDGES, TS)?;
        let kept = hits.iter().all(|&i| !y[i].is_finite());
        let spared = y
            .iter()
            .zip(&clean_y)
            .enumerate()
            .all(|(i, (a, b))| hits.contains(&i) || a.to_bits() == b.to_bits());
        ok &= clause(
            "multimax refuses a non-finite logit",
            fault == layer_fault(LAYER_NONE, FaultSite::Logit) && kept && spared,
            &format!("fault={fault:?} stays_non_finite={kept} others_unmoved={spared}"),
        );

        // One captured launch replays the eager run bit for bit.
        let (dr, dt) = (
            DeviceBuffer::from_host(st, &EDGES)?,
            DeviceBuffer::from_host(st, &TS)?,
        );
        let mut y = DeviceBuffer::from_host(st, &logits)?;
        let graph = cx.gpu.capture(|s| {
            cx.multimax.enqueue(
                s,
                MultiMaxArgs {
                    y: &mut y,
                    n: VOCAB,
                    ranges: &dr,
                    ts: &dt,
                    fault: cx.gpu.unlabelled_sink(),
                },
            )
        })?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            y.copy_from_host(st, &logits)?;
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&read(st, &y)?, &clean_y);
        }
        replay &= cx.gpu.take_fault()?.is_none();
        ok &= clause(
            "multimax capture",
            replay,
            "one node, two exact replays over reset logits",
        );

        // No logit, a short logit buffer and a short range are refused by name.
        let launch = |y: &mut DeviceBuffer<f32>, n: usize, ranges: &DeviceBuffer<f32>| {
            cx.multimax.enqueue(
                st,
                MultiMaxArgs {
                    y,
                    n,
                    ranges,
                    ts: &dt,
                    fault: cx.gpu.unlabelled_sink(),
                },
            )
        };
        let short = DeviceBuffer::from_host(st, &EDGES[..3])?;
        let refused = named(launch(&mut y, 0, &dr), WHAT)
            && named(launch(&mut y, VOCAB + 1, &dr), WHAT)
            && named(launch(&mut y, VOCAB, &short), WHAT);
        ok &= clause(
            "multimax host boundary",
            refused,
            "n = 0, n past the buffer and three range edges are named Shape errors",
        );
        Ok(ok)
    }

    // ------------------------------------------------------- the synthetic file

    /// `rows` FP8_B128 rows of `k` values drawn from `seed`: each block's f32 scale
    /// `scale · (1 + f)` for a uniform `f` in `[0, 1)`, its 128 value bytes uniform over
    /// the 254 E4M3 codes that hold a value (0x7f and 0xff are NaN).
    fn fp8_b128_bytes(rows: usize, k: usize, seed: u64, scale: f32) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let codes: Vec<u8> = (0..=u8::MAX).filter(|b| b & 0x7f != 0x7f).collect();
        let blocks = rows * k / 128;
        let mut out = Vec::with_capacity(blocks * 132);
        for _ in 0..blocks {
            out.extend((scale * (1.0 + rng.unit())).to_le_bytes());
            for _ in 0..128 {
                out.push(codes[rng.below(codes.len())]);
            }
        }
        out
    }

    /// `rows` MXFP8 rows of `k` values drawn from `seed`: each block's E8M0 scale byte
    /// uniform over 117..=119 (`2^-10` to `2^-8`), its 32 value bytes a random sign and
    /// mantissa under an exponent field uniform over 7..=11 (magnitudes 1 to 30), so a
    /// row is a few units' dot of a unit-scale input and every product is a normal f32.
    fn mxfp8_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let mut out = Vec::with_capacity(rows * k / 32 * 33);
        for _ in 0..rows * k / 32 {
            out.push(117 + rng.below(3) as u8);
            for _ in 0..32 {
                let (sign, field, mantissa) = (rng.below(2), 7 + rng.below(5), rng.below(8));
                out.push((sign << 7 | field << 3 | mantissa) as u8);
            }
        }
        out
    }

    /// The `rows` rows of `k` values of a type-`ty` tensor's bytes dequantized on the host.
    fn dequant_rows(ty: GgmlType, bytes: &[u8], (rows, k): (usize, usize)) -> Vec<Vec<f32>> {
        let row = bytes.len() / rows;
        par_map(rows, |r| {
            let mut v = vec![0.0f32; k];
            dequant_row(ty, &bytes[r * row..][..row], &mut v)
                .unwrap_or_else(|e| panic!("row {r}: {e}"));
            v
        })
    }

    /// One mix site's tensors on the host: `rows` rows of K, the scales, the offsets.
    struct SiteHost {
        w: Vec<f32>,
        scale: Vec<f32>,
        base: Vec<f32>,
    }

    /// What the synthetic file holds, dequantized where the file quantizes. The released
    /// file's head is BF16, which the loader holds in MXFP8 planes (`placement::
    /// file_card_format`, a speed choice that rounds each weight to E4M3 under its block's
    /// scale); the synthetic file stores the head as the MXFP8 blocks the card holds, so
    /// the head's rule reads the rows the card reads.
    struct Host {
        path: PathBuf,
        embd: Vec<f32>,
        sites: [SiteHost; 2],
        gains: [Vec<f32>; 2],
        gate: Vec<Vec<f32>>,
        up: Vec<Vec<f32>>,
        down: Vec<Vec<f32>>,
        out_norm: Vec<f32>,
        out_rows: Vec<Vec<f32>>,
        head: SiteHost,
        ranges: [f32; MULTIMAX],
        ts: [f32; MULTIMAX],
    }

    impl Host {
        fn make(dir: &Path) -> Result<Host, GateError> {
            let mut rng = GateRng(0x5eed_e550_00b0);
            let vector = |rng: &mut GateRng, n: usize, lo: f32, hi: f32| -> Vec<f32> {
                (0..n).map(|_| rng.uni(lo, hi)).collect()
            };
            let embd = fill(&mut rng, FILE_ROWS * HIDDEN, 1.0);
            let mut site = |scale: [f32; 3]| SiteHost {
                w: fill(&mut rng, HC_MIX * K, 0.05),
                scale: scale.to_vec(),
                base: fill(&mut rng, HC_MIX, 1.0),
            };
            let sites = [site([0.45, 0.35, 0.3]), site([0.6, 0.25, 0.4])];
            let gains = [
                vector(&mut rng, HIDDEN, 0.5, 1.5),
                vector(&mut rng, HIDDEN, 0.5, 1.5),
            ];
            // The gate and up rows read a unit-scale input to a dot of a few units; the
            // down rows read SiTU's output to a dot of about one.
            let gate_b = fp8_b128_bytes(FF, HIDDEN, 0x5eed_e550_00b1, 1.0 / 4096.0);
            let up_b = fp8_b128_bytes(FF, HIDDEN, 0x5eed_e550_00b2, 1.0 / 4096.0);
            let down_b = fp8_b128_bytes(HIDDEN, FF, 0x5eed_e550_00b3, 1.0 / 32768.0);
            let out_norm = vector(&mut rng, HIDDEN, 0.5, 1.5);
            let out_b = mxfp8_bytes(FILE_ROWS, HIDDEN, 0x5eed_e550_00b4);
            let head = SiteHost {
                w: fill(&mut rng, HC_STREAMS * K, 0.05),
                scale: vec![1.25],
                base: fill(&mut rng, HC_STREAMS, 1.0),
            };
            // Edges inside the logits' range and a squared term on the upper side whose
            // weight is negative: the argmax of the MultiMax logits is no raw argmax.
            let (ranges, ts) = ([-1.5, 1.0, 0.0, 2.0], [0.5, -0.3, 0.2, -1.0]);
            let vec1 = |name: String, n: usize, v: &[f32]| {
                (name, vec![n as u64], GgmlType::F32, f32_bytes(v))
            };
            let mut decls = vec![(
                token_embd(),
                vec![HIDDEN as u64, FILE_ROWS as u64],
                GgmlType::F32,
                f32_bytes(&embd),
            )];
            for (s, sub) in [Sub::Attn, Sub::Ffn].into_iter().enumerate() {
                decls.push((
                    hc_fn(0, sub),
                    vec![K as u64, HC_MIX as u64],
                    GgmlType::F32,
                    f32_bytes(&sites[s].w),
                ));
                decls.push(vec1(hc_scale(0, sub), 3, &sites[s].scale));
                decls.push(vec1(hc_base(0, sub), HC_MIX, &sites[s].base));
            }
            decls.push(vec1(attn_norm(0), HIDDEN, &gains[0]));
            decls.push(vec1(ffn_norm(0), HIDDEN, &gains[1]));
            let proj = |name: String, (cols, rows): (usize, usize), bytes: &Vec<u8>| {
                (
                    name,
                    vec![cols as u64, rows as u64],
                    GgmlType::FP8_B128,
                    bytes.clone(),
                )
            };
            decls.push(proj(ffn_gate(0), (HIDDEN, FF), &gate_b));
            decls.push(proj(ffn_up(0), (HIDDEN, FF), &up_b));
            decls.push(proj(ffn_down(0), (FF, HIDDEN), &down_b));
            decls.push(vec1(output_norm(), HIDDEN, &out_norm));
            decls.push((
                output(),
                vec![HIDDEN as u64, FILE_ROWS as u64],
                GgmlType::MXFP8,
                out_b.clone(),
            ));
            decls.push((
                OUTPUT_HC_FN.to_string(),
                vec![K as u64, HC_STREAMS as u64],
                GgmlType::F32,
                f32_bytes(&head.w),
            ));
            decls.push(vec1(OUTPUT_HC_SCALE.to_string(), 1, &head.scale));
            decls.push(vec1(OUTPUT_HC_BASE.to_string(), HC_STREAMS, &head.base));
            decls.push(vec1(OUTPUT_MULTIMAX_RANGES.to_string(), MULTIMAX, &ranges));
            decls.push(vec1(OUTPUT_MULTIMAX_TS.to_string(), MULTIMAX, &ts));
            let path = dir.join("ernie55-trunk.gguf");
            write_gguf(&path, &[], &decls)?;
            Ok(Host {
                path,
                embd,
                sites,
                gains,
                gate: dequant_rows(GgmlType::FP8_B128, &gate_b, (FF, HIDDEN)),
                up: dequant_rows(GgmlType::FP8_B128, &up_b, (FF, HIDDEN)),
                down: dequant_rows(GgmlType::FP8_B128, &down_b, (HIDDEN, FF)),
                out_norm,
                out_rows: dequant_rows(GgmlType::MXFP8, &out_b, (FILE_ROWS, HIDDEN)),
                head,
                ranges,
                ts,
            })
        }
    }

    // --------------------------------------------------------- the trunk's sites

    /// What a site's pre left: the HC_PRE rows, the fold, the normed fold.
    struct PreOut {
        hc: Vec<f32>,
        lin: Vec<f32>,
        normed: Vec<f32>,
    }

    impl PreOut {
        fn read(st: &cuda_core::CudaStream, trunk: &Trunk) -> Result<PreOut, GateError> {
            Ok(PreOut {
                hc: read(st, trunk.hc_rows())?,
                lin: read(st, trunk.lin())?,
                normed: read(st, trunk.cur())?,
            })
        }
    }

    /// A site's pre against the rule, over the streams `s` it read (`t` tokens): its
    /// `pre` and `post` gates within bound of the f64 rule from the f64 mixes (the mix
    /// error carried through the sigmoid's slope), the fold bit for bit the host's
    /// fused multiply-adds over the device's own `pre`, the norm within bound of the f64
    /// norm of the device's own fold.
    fn check_pre(
        tag: &str,
        s: &[f32],
        site: &SiteHost,
        gain: &[f32],
        out: &PreOut,
        t: usize,
    ) -> bool {
        let sc = [site.scale[0], site.scale[1], site.scale[2]].map(f64::from);
        let eps = f64::from(HC_EPS);
        let (mut dev, mut want, mut bound) = (vec![], vec![], vec![]);
        for tt in 0..t {
            let d = dots64(&site.w, &s[tt * K..][..K], HC_MIX);
            let row = &out.hc[tt * HC_MIX..][..HC_MIX];
            for i in 0..HC_STREAMS {
                let (m, e) = mix_ref(Rule::RmsAdd, &d, i);
                let v = m * sc[0] + f64::from(site.base[i]);
                let (p, pe) = pre_ref(v, e * sc[0].abs(), eps);
                let (m2, e2) = mix_ref(Rule::RmsAdd, &d, HC_STREAMS + i);
                let v2 = m2 * sc[1] + f64::from(site.base[HC_STREAMS + i]);
                let (q, qe) = post_ref(v2, e2 * sc[1].abs());
                dev.extend([row[i], row[HC_STREAMS + i]]);
                want.extend([p, q]);
                bound.extend([pe, qe]);
            }
        }
        let (gates_ok, gworst) = within(&dev, &want, &bound);
        let mut fold = Vec::with_capacity(t * HIDDEN);
        for tt in 0..t {
            let pre4: [f32; 4] = std::array::from_fn(|i| out.hc[tt * HC_MIX + i]);
            for d in 0..HIDDEN {
                fold.push(fold_host(
                    std::array::from_fn(|j| s[tt * K + j * HIDDEN + d]),
                    pre4,
                ));
            }
        }
        let fold_ok = bits_equal(&out.lin[..t * HIDDEN], &fold);
        let (mut ndev, mut nwant, mut nbound) = (vec![], vec![], vec![]);
        for tt in 0..t {
            let lin: Vec<f64> = out.lin[tt * HIDDEN..][..HIDDEN]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let (w, b) = norm_ref(&lin, &vec![0.0; HIDDEN], gain, RMS_EPS);
            ndev.extend_from_slice(&out.normed[tt * HIDDEN..][..HIDDEN]);
            nwant.extend(w);
            nbound.extend(b);
        }
        let (norm_ok, nworst) = within(&ndev, &nwant, &nbound);
        clause(
            tag,
            gates_ok && fold_ok && norm_ok,
            &format!(
                "gates worst={gworst:.3} of the bound, fold bits={fold_ok}, norm worst={nworst:.3} of the bound"
            ),
        )
    }

    /// A site's post against the rule: `res` the streams it read, `x` the sub-layer's
    /// output, `hc` the pre's rows, `out` the new streams — bit for bit the host's
    /// fused multiply-adds, within the f64 rule's bound.
    fn check_post(
        tag: &str,
        (res, x, hc): (&[f32], &[f32], &[f32]),
        out: &[f32],
        t: usize,
    ) -> bool {
        let (mut want32, mut want, mut bound) = (
            vec![0.0f32; t * K],
            vec![0.0f64; t * K],
            vec![0.0f64; t * K],
        );
        for tt in 0..t {
            let row = &hc[tt * HC_MIX..][..HC_MIX];
            let (post, comb) = (&row[4..8], &row[8..]);
            for d in 0..HIDDEN {
                let r4: [f32; 4] = std::array::from_fn(|j| res[tt * K + j * HIDDEN + d]);
                let xv = x[tt * HIDDEN + d];
                let new = post_host(xv, r4, std::array::from_fn(|i| post[i]), comb);
                for i in 0..4 {
                    let at = tt * K + i * HIDDEN + d;
                    want32[at] = new[i];
                    let mut terms = vec![f64::from(xv) * f64::from(post[i])];
                    terms.extend((0..4).map(|j| f64::from(comb[4 * j + i]) * f64::from(r4[j])));
                    want[at] = terms.iter().sum();
                    bound[at] = 5.0 * U * terms.iter().map(|v| v.abs()).sum::<f64>() * SECOND_ORDER;
                }
            }
        }
        let dev = &out[..t * K];
        let (in_bound, worst) = within(dev, &want, &bound);
        clause(
            tag,
            in_bound && bits_equal(dev, &want32),
            &format!("worst={worst:.3} of the bound, bit for bit the host's fused multiply-adds"),
        )
    }

    /// The dense block against the rule stage by stage over the block's own taps (the
    /// gate and up sums, the SiTU product): each projection within `rel(k) · Σ|w·x|` of
    /// the f64 dot of the f32 values it read — any order of the dot's terms plus a few
    /// roundings in the m-column gemv's arm, BF16 operands and F32 accumulation in the
    /// wide arm of a compute capability 12 card — and SiTU within its roundings of the
    /// f64 rule over the device's own sums.
    fn check_dense(
        tag: &str,
        gpu: &Gpu,
        host: &Host,
        taps: (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (x, y): (&[f32], &[f32]),
        t: usize,
    ) -> Result<bool, GateError> {
        let st = gpu.stream();
        let mma = t > 8 && dense_mma(gpu.context())?;
        let rel = |k: usize| if mma { bf16_dot_rel(k) } else { gamma(k + 8) };
        let (g, u, a) = (read(st, taps.0)?, read(st, taps.1)?, read(st, taps.2)?);
        let mut ok = true;
        for (name, sums, rows) in [("gate", &g, &host.gate), ("up", &u, &host.up)] {
            let (mut dev, mut want, mut bound) = (vec![], vec![], vec![]);
            for c in 0..t {
                let xc = &x[c * HIDDEN..][..HIDDEN];
                let dots = par_map(FF, |j| {
                    let (mut d, mut s) = (0.0f64, 0.0f64);
                    for (&wv, &xv) in rows[j].iter().zip(xc) {
                        let p = f64::from(wv) * f64::from(xv);
                        d += p;
                        s += p.abs();
                    }
                    (d, s)
                });
                for (j, (d, s)) in dots.into_iter().enumerate() {
                    dev.push(sums[c * FF + j]);
                    want.push(d);
                    bound.push(rel(HIDDEN) * s * SECOND_ORDER);
                }
            }
            let (pass, worst) = within(&dev, &want, &bound);
            ok &= clause(
                &format!("{tag} {name}"),
                pass,
                &format!("worst={worst:.3} of the bound (bf16 operands={mma})"),
            );
        }
        let want: Vec<f64> = (0..t * FF)
            .map(|i| situ64(f64::from(g[i]), f64::from(u[i])))
            .collect();
        let bound: Vec<f64> = want
            .iter()
            .map(|v| SITU_ROUNDINGS * U * v.abs() * SECOND_ORDER)
            .collect();
        let (pass, worst) = within(&a[..t * FF], &want, &bound);
        ok &= clause(
            &format!("{tag} situ"),
            pass,
            &format!("worst={worst:.3} of the bound over the device's own sums"),
        );
        let (mut want, mut bound) = (vec![], vec![]);
        for c in 0..t {
            let ac = &a[c * FF..][..FF];
            let dots = par_map(HIDDEN, |r| {
                let (mut d, mut s) = (0.0f64, 0.0f64);
                for (&wv, &av) in host.down[r].iter().zip(ac) {
                    let p = f64::from(wv) * f64::from(av);
                    d += p;
                    s += p.abs();
                }
                (d, s)
            });
            for (d, s) in dots {
                want.push(d);
                bound.push(rel(FF) * s * SECOND_ORDER);
            }
        }
        let (pass, worst) = within(&y[..t * HIDDEN], &want, &bound);
        ok &= clause(
            &format!("{tag} down"),
            pass,
            &format!("worst={worst:.3} of the bound over the device's own product"),
        );
        Ok(ok)
    }

    /// The sites' ids a walk embeds: spread over the file's rows, the ends included.
    const TOKENS: [u32; WALK_TOKENS] =
        [3, 2047, 1024, 0, 777, 5, 1999, 64, 300, 1500, 17, 2000, 999];

    /// One walk of layer 0 over `t` tokens: embed, the attention site's pre and post,
    /// the feed-forward site's pre, the dense block on its normed input, its post.
    fn walk(
        gpu: &Gpu,
        w: &Weights,
        host: &Host,
        trunk: &mut Trunk,
        t: usize,
    ) -> Result<bool, GateError> {
        let st = gpu.stream();
        let tag = format!("trunk t={t}");
        let mut rng = GateRng(0x5eed_e550_0010 + t as u64);
        gpu.take_fault()?;
        trunk.embed(gpu.stream(), &TOKENS[..t])?;
        let s0 = read(st, trunk.streams())?[..t * K].to_vec();
        let embedded = (0..t).all(|tt| {
            (0..HC_STREAMS).all(|s| {
                bits_equal(
                    &s0[tt * K + s * HIDDEN..][..HIDDEN],
                    &host.embd[TOKENS[tt] as usize * HIDDEN..][..HIDDEN],
                )
            })
        });
        let mut ok = clause(
            &format!("{tag} embed"),
            embedded,
            "each token's row repeated into the four streams, bit for bit",
        );
        trunk.enqueue_pre(gpu, w, 0, Sub::Attn)?;
        st.synchronize()?;
        let attn = PreOut::read(st, trunk)?;
        ok &= check_pre(
            &format!("{tag} attention pre"),
            &s0,
            &host.sites[0],
            &host.gains[0],
            &attn,
            t,
        );
        let x1 = fill(&mut rng, t * HIDDEN, 1.0);
        let dx1 = DeviceBuffer::from_host(st, &x1)?;
        trunk.enqueue_post(gpu, 0, Sub::Attn, &dx1)?;
        st.synchronize()?;
        let s1 = read(st, trunk.streams())?[..t * K].to_vec();
        ok &= check_post(
            &format!("{tag} attention post"),
            (&s0, &x1, &attn.hc),
            &s1,
            t,
        );
        trunk.enqueue_pre(gpu, w, 0, Sub::Ffn)?;
        st.synchronize()?;
        let ffn = PreOut::read(st, trunk)?;
        ok &= check_pre(
            &format!("{tag} feed-forward pre"),
            &s1,
            &host.sites[1],
            &host.gains[1],
            &ffn,
            t,
        );
        let dxn = DeviceBuffer::from_host(st, &ffn.normed[..t * HIDDEN])?;
        let mut dy = DeviceBuffer::zeroed(st, t * HIDDEN)?;
        trunk.enqueue_dense(gpu, w, 0, &dxn, &mut dy)?;
        st.synchronize()?;
        let y = read(st, &dy)?;
        let taps = trunk.dense_taps(0).ok_or("layer 0 has a dense block")?;
        ok &= check_dense(
            &format!("{tag} dense"),
            gpu,
            host,
            taps,
            (&ffn.normed[..t * HIDDEN], &y),
            t,
        )?;
        trunk.enqueue_post(gpu, 0, Sub::Ffn, &dy)?;
        st.synchronize()?;
        let s2 = read(st, trunk.streams())?[..t * K].to_vec();
        ok &= check_post(
            &format!("{tag} feed-forward post"),
            (&s1, &y, &ffn.hc),
            &s2,
            t,
        );
        let clean = gpu.take_fault()?.is_none();
        ok &= clause(
            &format!("{tag} fault word"),
            clean,
            "no launch of the walk raised",
        );
        Ok(ok)
    }

    /// A trunk refuses what it cannot walk, by name.
    fn trunk_refusals(gpu: &Gpu, w: &Weights, trunk: &mut Trunk) -> Result<bool, GateError> {
        let st = gpu.stream();
        let none = named(trunk.embed(gpu.stream(), &[]), "ernie55::Trunk::embed");
        let many = named(
            trunk.embed(gpu.stream(), &[0; WALK_TOKENS + 1]),
            "ernie55::Trunk::embed",
        );
        let past = trunk.embed(gpu.stream(), &[FILE_ROWS as u32]).is_err();
        trunk.embed(gpu.stream(), &TOKENS[..2])?;
        let dx = DeviceBuffer::zeroed(st, 2 * HIDDEN)?;
        let mut dy = DeviceBuffer::zeroed(st, 2 * HIDDEN)?;
        let no_dense = named(
            trunk.enqueue_dense(gpu, w, 1, &dx, &mut dy),
            "ernie55::Trunk::enqueue_dense",
        );
        let mut short = DeviceBuffer::zeroed(st, HIDDEN)?;
        let short_y = named(
            trunk.enqueue_dense(gpu, w, 0, &dx, &mut short),
            "ernie55::Trunk::enqueue_dense",
        );
        let no_site = trunk.enqueue_pre(gpu, w, 1, Sub::Attn).is_err();
        Ok(clause(
            "trunk host boundary",
            none && many && past && no_dense && short_y && no_site,
            &format!(
                "no token={none} too many={many} token past the file={past} layer without a dense block={no_dense} short output={short_y} layer past the trunk={no_site}"
            ),
        ))
    }

    // ----------------------------------------------------------- the head

    /// The head's fold of the last rows of `tokens` tokens' streams into the common head of as
    /// many rows, then that head with the MultiMax epilogue: the model's head, run whole.
    fn run_head(
        gpu: &Gpu,
        w: &Weights,
        (head, out): (&mut Head, &mut OutputHead),
        streams: &DeviceBuffer<f32>,
        tokens: usize,
    ) -> Result<(), GpuError> {
        head.enqueue_fold(gpu, w, streams, tokens, out.input_mut())?;
        let n = out.n_vocab() * out.m();
        out.enqueue_with_epilogue(gpu, w, |logits| head.enqueue_multimax(gpu, logits, n))
    }

    /// The head over the prompt's last three rows of five.
    fn head_component(
        gpu: &Gpu,
        w: &Weights,
        host: &Host,
        facts: &Facts,
    ) -> Result<bool, GateError> {
        let st = gpu.stream();
        let mut head = Head::load(gpu, w, facts, HEAD_ROWS)?;
        let mut out = OutputHead::with_m(gpu, w, facts.rms_eps, HEAD_ROWS)?;
        let mut rng = GateRng(0x5eed_e550_0020);
        let hs = streams(&mut rng, HEAD_TOKENS, 1.0);
        let dhs = DeviceBuffer::from_host(st, &hs)?;
        gpu.take_fault()?;
        run_head(gpu, w, (&mut head, &mut out), &dhs, HEAD_TOKENS)?;
        st.synchronize()?;
        let normed = out.normed_to_host(gpu)?;
        let logits = out.logits_to_host(gpu)?;
        let tokens = out.tokens(gpu)?;
        let clean = gpu.take_fault()?.is_none();
        let first = HEAD_TOKENS - HEAD_ROWS;
        let eps = f64::from(HC_EPS);
        let s = f64::from(host.head.scale[0]);
        let (mut ndev, mut nwant, mut nbound) = (vec![], vec![], vec![]);
        for c in 0..HEAD_ROWS {
            let xt = &hs[(first + c) * K..][..K];
            let d = dots64(&host.head.w, xt, HC_STREAMS);
            let pre: Vec<(f64, f64)> = (0..HC_STREAMS)
                .map(|r| {
                    let (m, e) = mix_ref(Rule::Rsqrt, &d, r);
                    pre_ref(m * s + f64::from(host.head.base[r]), e * s.abs(), eps)
                })
                .collect();
            let (mut fold, mut ex) = (vec![0.0f64; HIDDEN], vec![0.0f64; HIDDEN]);
            for j in 0..HIDDEN {
                let o: Vec<f64> = (0..HC_STREAMS)
                    .map(|r| f64::from(xt[r * HIDDEN + j]))
                    .collect();
                fold[j] = (0..HC_STREAMS).map(|r| pre[r].0 * o[r]).sum();
                let rounding: f64 = (0..HC_STREAMS).map(|r| (pre[r].0 * o[r]).abs()).sum();
                ex[j] = ((0..HC_STREAMS).map(|r| o[r].abs() * pre[r].1).sum::<f64>()
                    + 4.0 * U * rounding)
                    * SECOND_ORDER;
            }
            let (want, bound) = norm_ref(&fold, &ex, &host.out_norm, RMS_EPS);
            ndev.extend_from_slice(&normed[c * HIDDEN..][..HIDDEN]);
            nwant.extend(want);
            nbound.extend(bound);
        }
        let (pass, worst) = within(&ndev, &nwant, &nbound);
        let mut ok = clause(
            "head normed rows",
            pass && clean,
            &format!(
                "worst={worst:.3} of the bound over the four-row mix, its fold and the norm, fault_free={clean}"
            ),
        );
        let rows = par_map(FILE_ROWS * HEAD_ROWS, |i| {
            let (v, c) = (i / HEAD_ROWS, i % HEAD_ROWS);
            let (mut d, mut a) = (0.0f64, 0.0f64);
            for (&wv, &xv) in host.out_rows[v].iter().zip(&normed[c * HIDDEN..][..HIDDEN]) {
                let p = f64::from(wv) * f64::from(xv);
                d += p;
                a += p.abs();
            }
            (d, gamma(HIDDEN + 8) * a)
        });
        let (mut want, mut bound) = (vec![], vec![]);
        for &(raw, bl) in &rows {
            let (y, b) = multimax64(raw, bl, &host.ranges, &host.ts);
            want.push(y);
            bound.push(b);
        }
        let (pass, worst) = within(&logits, &want, &bound);
        ok &= clause(
            "head multimax logits",
            pass,
            &format!(
                "worst={worst:.3} of the bound: MultiMax of the f64 projection of the device's own normed rows"
            ),
        );
        let argmax_dev: Vec<u32> = (0..HEAD_ROWS)
            .map(|c| {
                let col: Vec<f32> = (0..FILE_ROWS).map(|v| logits[v * HEAD_ROWS + c]).collect();
                argmax_of(&col) as u32
            })
            .collect();
        let raw_argmax: Vec<u32> = (0..HEAD_ROWS)
            .map(|c| {
                let mut hi = 0;
                for v in 0..FILE_ROWS {
                    if rows[v * HEAD_ROWS + c].0 > rows[hi * HEAD_ROWS + c].0 {
                        hi = v;
                    }
                }
                hi as u32
            })
            .collect();
        let moved = tokens.iter().zip(&raw_argmax).all(|(t, r)| t != r);
        ok &= clause(
            "head token",
            tokens == argmax_dev && moved,
            &format!(
                "tokens={tokens:?} are the MultiMax logits' argmax, the raw logits' were {raw_argmax:?}"
            ),
        );

        // A NaN stream in the last row has no mix: the head's layer is raised, and the
        // token read refuses it.
        let mut bad = hs;
        bad[(HEAD_TOKENS - 1) * K + 7] = f32::NAN;
        let dbad = DeviceBuffer::from_host(st, &bad)?;
        run_head(gpu, w, (&mut head, &mut out), &dbad, HEAD_TOKENS)?;
        st.synchronize()?;
        let refused = matches!(out.tokens(gpu), Err(GpuError::Fault { .. }));
        let fault = gpu.take_fault()?;
        let site = fault.map(|f| (f.layer, f.site()));
        ok &= clause(
            "head refuses a NaN stream",
            refused && site == Some((LAYER_HEAD, Some(FaultSite::F32Product))),
            &format!("token read refused={refused} fault={fault:?}"),
        );

        // A chunk shorter than the rows projected, and streams short of the chunk.
        let two = named(
            head.enqueue_fold(gpu, w, &dhs, HEAD_ROWS - 1, out.input_mut()),
            "ernie55::Head",
        );
        let short = DeviceBuffer::zeroed(st, HEAD_TOKENS * K - 1)?;
        let held = named(
            head.enqueue_fold(gpu, w, &short, HEAD_TOKENS, out.input_mut()),
            "ernie55::Head",
        );
        ok &= clause(
            "head host boundary",
            two && held,
            &format!(
                "a chunk of {} tokens={two}, streams a value short={held}",
                HEAD_ROWS - 1
            ),
        );
        Ok(ok)
    }

    // ---------------------------------------------------------------- the run

    fn component_clauses(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
        let st = gpu.stream();
        let host = Host::make(dir)?;
        let split = Arc::new(
            Split::open(&host.path).map_err(|e| format!("open {}: {e}", host.path.display()))?,
        );
        let w = Weights::load_where(st, &split, |n| n != token_embd())?;
        let facts = Facts {
            hidden: HIDDEN,
            layers: 1,
            hc_eps: HC_EPS,
            sinkhorn_iters: ITERS,
            rms_eps: RMS_EPS,
            dense: vec![(0, FF)],
            situ_beta: SITU_BETA,
        };
        let mut trunk = Trunk::load(gpu, &w, &facts, Arc::clone(&split), WALK_TOKENS)?;
        let mut ok = true;
        for t in [5usize, WALK_TOKENS] {
            ok &= walk(gpu, &w, &host, &mut trunk, t)?;
        }
        ok &= trunk_refusals(gpu, &w, &mut trunk)?;
        ok &= head_component(gpu, &w, &host, &facts)?;
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_ernie55_trunk: device {}", gpu.device_name()?);
        let ctx = gpu.context();
        let cx = Cx {
            pieces: HcF32PiecesKernels::load(ctx)?,
            hc: HcKernels::load(ctx)?,
            multimax: MultiMaxKernels::load(ctx)?,
            gpu,
        };
        let mut ok = pre_clauses(&cx)?;
        ok &= head_clauses(&cx)?;
        ok &= post_clauses(&cx)?;
        ok &= multimax_clauses(&cx)?;
        let dir =
            std::env::temp_dir().join(format!("bloomery-ernie55-trunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome = component_clauses(&cx.gpu, &dir);
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        ok &= outcome?;
        cleanup?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
