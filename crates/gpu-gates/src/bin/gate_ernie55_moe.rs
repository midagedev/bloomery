//! GPU gate for ERNIE 5.5's latent mixture of experts on the card (the block of
//! `bloomery_gpu_ernie55::moe`: the split-feature sigmoid router at 512 experts and the top 10,
//! `bloomery_gpu::route_body::sigmoid_split_512_10`; the latent pair; the MXFP4 experts through the
//! common card leg under SiTU; the routed sum's norm; the shared expert) and the same picks split
//! between the card and the common host tier (`HostLayer::build_situ` over a synthetic file). Seeded
//! inputs at the released file's shapes (hidden 4096, latent 2048, 512 experts, the top 10, expert
//! width 2048) and at the fixture's expert width (512); nothing is read from a model file.
//!
//! The reference is `tools/ref/ernie55_ref.py` (`route`, `_moe_ffn`, `situ`, `rms_norm`) as f64
//! rules. A stage's rule reads the stage before it as the card left it (the latent row, the leg's
//! SiTU rows, the routed sum), so a band holds only that stage's own f32 sums and the device
//! `expf`/`tanh` beside the f64 ones; the card leg's q8_1 activations the rule quantizes as the card
//! does (`q8_1_dequant`, per 128 values, over the card's own f32 rows).
//!
//! Clauses, one a contract:
//! - `router_ids`: every token's kept ids EQUAL the f64 rule's: the top 10 of `σ(l0) + σ(l1) +
//!   bias`, an equal value to the larger id (the common selection's order, ggml's `argsort_top_k`;
//!   the Python reference's stable argsort keeps the lower id, a difference only an exact tie
//!   sees), and at least one token's biased top 10 differs from the unbiased one (the bias steers
//!   the pick). The rows are drawn with the cut's gap above [`MIN_GAP`], so the ids decide the
//!   rule and not a rounding; the detail names the rows skipped and the smallest gap kept.
//! - `router_logits` / `router_scores` / `router_weights`: the logits, the summed scores and the
//!   weights within band of the f64 rule; the weights from the unbiased scores, `p / max(Σp, 1e-20)
//!   · scale[id]`.
//! - `router_floor_*`: a row whose every logit is −60 (the kept ten sum under the 1e-20 floor): the
//!   ids the ties' order, the weights the f64 rule's, and the renormalized weights' sum far from 1
//!   (the divisor floored, not the sum renormalized to one). `router_zero_weights`: every logit
//!   −100 (the f32 sigmoid 0): every weight 0.
//! - `router_ties_*`: every score equal (a zero row through a zero router, a zero bias): the larger
//!   id's slot first, the weights `1/10 · scale[id]`.
//! - `router_rows_is_step` / `prompt_router_*`: a prompt unit's ids BIT-EQUAL the steps' at those
//!   tokens, its scores and weights the steps' rules within band (the wide tile GEMM's logits
//!   beside the gemv's).
//! - `step_*`: the step's stages against their f64 rules over the card's own rows, every token's
//!   worst: the latent row, the leg's SiTU rows and down rows, the routed sum, the norm, the latent
//!   up, the shared expert, the output.
//! - `prompt_is_step_*`: a unit of `TOKENS` rows' stages and output the steps' within band (the wide
//!   arm's tile GEMMs: on a compute capability 12 device the E4M3 MMA rounds each activation to
//!   BF16).
//! - `split_step_*` / `split_rows_*`: the same picks with the card holding all but the last
//!   `N_HOST` experts and the host tier the rest (`experts_into` a column, `experts_union_into` the
//!   unit): the card's slots ran, the host's share not empty, the routed sums (card + host) and the
//!   outputs the all-card block's within band (the host quantizes its activations in blocks of 32
//!   where the card's leg reads 128), the unit's split the steps' within band.
//! - `fault_*`: a NaN row raises the fault word (the layer's, the router's site among them) and the
//!   output row is NaN, then the word clears.
//! - `refused`: a width the block does not run, a router count it does not serve, a layer with no
//!   routed block, a unit past the buffers, a back with neither a card slot nor a host sum, a front
//!   into a short latent buffer, a whole-card launch on a card that holds a share of the experts, a
//!   rows launch with no logits buffer, each refused by name before anything is enqueued.

#[cfg(feature = "ernie55")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_moe", gate::run())
}

#[cfg(feature = "ernie55")]
mod gate {
    use bloomery_gpu::arch::hybrid::GqaCx;
    use bloomery_gpu::flash_gqa::FlashGqaKernels;
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmKernels};
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::parts;
    use bloomery_gpu::route_body::sigmoid_split_512_10 as split;
    use bloomery_gpu::routed_block::RoutedRows;
    use bloomery_gpu::site::KGemvKernels;
    use bloomery_gpu::weights::{TensorPart, Weights};
    use bloomery_gpu::{FaultSite, Gpu, GpuError};
    use bloomery_gpu_ernie55::moe::{
        Moe, MoeBufs, MoeDims, RouterRead, Tap, check_card_stacks, host_layer,
    };
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, par_map, q8_1_dequant, verdict};
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::{GgmlType, Split, TensorInfo};
    use model::arch::ernie55::names;
    use model::moe::{HostScratch, UnionScratch};
    use model::ops::Tensor2;
    use model::part::Take;
    use model::placement::{CardFormat, ExpertList};
    use model::r8file::R8Source;
    use std::path::{Path, PathBuf};

    /// The gate's name: the recipe's, the refusals' and the contexts'.
    const NAME: &str = "gate_ernie55_moe";
    /// `embedding_length`, the released file's and the fixture's.
    const HIDDEN: usize = 4096;
    /// `expert_latent_length` (`moe_latent_size`), both files'.
    const LATENT: usize = 2048;
    /// `expert_count`, the released file's and the fixture's.
    const N_EXPERT: usize = 512;
    /// `expert_used_count`, both files'.
    const N_USED: usize = 10;
    /// The one routed layer the synthetic file holds.
    const LAYER: usize = 0;
    /// `activation.situ_beta`.
    const BETA: f32 = 4.0;
    /// `activation.situ_linear_beta`.
    const LINEAR_BETA: f32 = 25.0;
    /// `rms_norm_eps`.
    const EPS: f32 = 1e-6;
    /// `expert_feed_forward_length` (the shared expert's too): the released file's, then the
    /// fixture's.
    const WIDTHS: [usize; 2] = [2048, 512];
    /// The steps the router and ladder clauses run over: a prompt unit's rows.
    const TOKENS: usize = 16;
    /// The experts the host tier serves in the split clauses: the last of the 512.
    const N_HOST: usize = 64;
    /// The router's logits, scores and weights against the f64 rule, `max|Δ| / max|ref|`.
    /// [derived] Each logit is an f32 dot of 4096 terms: within ε·√k·(Σ|t|/|Σt|) of the f64 sum,
    /// about 1e-5 at its share of cancellation; the scores are its sigmoid's (a contraction) and
    /// the weights the scores' own bits through a division and a scale. The band at ten times the
    /// logits' share.
    const ROUTER_BAND: f64 = 1e-4;
    /// The smallest relative gap between a token's 10th and 11th biased score a drawn row is kept
    /// at: a row under it is skipped and the next drawn, so the ids clauses decide the rule and
    /// never a rounding. [derived] A dot's f32 rounding moves a score by about 1e-6 of a biased
    /// score of about 2 (5e-7 relative); the gap at 1e-4 is 200 times that.
    const MIN_GAP: f64 = 1e-4;
    /// The rows drawn at most for the `TOKENS` the gap keeps: [derived] a drawn row's gap is about
    /// exponential with a mean of 1.5e-2, so one in 150 is under 1e-4.
    const MAX_DRAWN: usize = 16 * TOKENS;
    /// A stage of the block against its f64 rule over the card's own rows, `max|Δ| / max|ref|`.
    /// [derived] The f32 dot of 2048 or 4096 terms (about 1e-5 at its share of cancellation), the
    /// device `expf` and `tanh` beside the f64 ones (ulps of a value), the q8_1 form both sides
    /// share.
    const STAGE_BAND: f64 = 1e-4;
    /// A pure f32 sum of the card's own rows (the slot sum's fused multiply-adds, the output's
    /// add), `max|Δ| / max|ref|`.
    const SUM_BAND: f64 = 1e-5;
    /// The split block against the all-card one: the host tier's experts beside the card leg's.
    /// [derived] The host quantizes its activation column in blocks of 32 values, the card's leg
    /// in blocks of 128: the two quantizations differ by a step of their block's largest magnitude
    /// over 127, under 1 % of the row's RMS each, passing the dot as the root of the two squares
    /// (about 1 % of the dot's RMS) and carried through the down; the band at five times that.
    const SPLIT_BAND: f64 = 5e-2;
    /// A prompt unit's rows against the steps', `max|Δ| / max|ref|`.
    /// [derived] The unit's dense sites run the wide arm's tile GEMMs; on a compute capability 12
    /// device the E4M3 MMA rounds each activation to BF16 (2^-9 relative a product), a share of
    /// about 2e-3 of an output's magnitude over a 4096-term dot, carried through the block; the
    /// band at ten times that share.
    const PROMPT_BAND: f64 = 2e-2;
    /// A logit far below zero, where a sigmoid score is 9e-27 and the kept ten sum under the 1e-20
    /// floor.
    const FLOOR_LOGIT: f32 = -60.0;
    /// A logit whose f32 sigmoid is 0 (the exponential overflows).
    const ZERO_LOGIT: f32 = -100.0;
    /// A weight below this is the zero the f32 sigmoid underflows to.
    const ZERO_WEIGHT: f32 = 1e-30;
    /// The E2M1 code values of MXFP4's low nibble (the high nibble's the same negated).
    const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    /// The seed of the synthetic file's tensors and of the step rows.
    const SEED: u64 = 0x4552_4e49_4535_3555;
    /// The eight stages of the block, in [`Stages`]' order.
    const STAGE_NAMES: [&str; 8] = [
        "lat", "act", "down", "routed", "normed", "lat_up", "shared", "out",
    ];
    const _: () = assert!(N_HOST < N_EXPERT);

    // --- lines and numbers ---------------------------------------------------------------------------------------

    /// One clause's line and whether it passed.
    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{NAME} {name}: {detail} {}", verdict(pass));
        pass
    }

    /// `max|got − want| / max|want|` over f64 values, infinite on a length mismatch or a non-finite
    /// value.
    fn rel(got: &[f64], want: &[f64]) -> f64 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let num = got
            .iter()
            .zip(want)
            .map(|(&g, &w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        let den = want.iter().fold(0.0f64, |a, &w| a.max(w.abs()));
        if num.is_finite() && den > 0.0 {
            num / den
        } else {
            f64::INFINITY
        }
    }

    /// [`rel`] over an f32 output.
    fn rel32(got: &[f32], want: &[f64]) -> f64 {
        let got: Vec<f64> = got.iter().map(|&g| f64::from(g)).collect();
        rel(&got, want)
    }

    /// A clause of a relative error against its band.
    fn banded(name: &str, r: f64, band: f64) -> bool {
        clause(
            name,
            r <= band,
            &format!("max_rel {r:.3e} (band {band:.0e})"),
        )
    }

    /// A refusal clause: the call refused by its shape, state or tensor check.
    fn refused(name: &str, r: Result<(), GpuError>) -> bool {
        clause(
            &format!("refused {name}"),
            matches!(
                r,
                Err(GpuError::Shape { .. } | GpuError::State { .. } | GpuError::Tensor { .. })
            ),
            &format!("{r:?}"),
        )
    }

    /// A draw about normal of standard deviation `sd`: three uniforms summed.
    fn normal(rng: &mut GateRng, sd: f32) -> f32 {
        (rng.uni(-1.0, 1.0) + rng.uni(-1.0, 1.0) + rng.uni(-1.0, 1.0)) * sd
    }

    /// f32 values as f64.
    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    /// The q8_1 form of `m` columns of `k` f32 values as the card quantizes them, as f64.
    fn q8(v: &[f32], (k, m): (usize, usize)) -> Vec<f64> {
        f64s(&q8_1_dequant(v, k, m))
    }

    /// The picks' ids in the ties' order: the larger id first.
    fn tie_order(ids: &[u32]) -> bool {
        ids.len() == N_USED
            && ids
                .iter()
                .enumerate()
                .all(|(s, &e)| e as usize == N_EXPERT - 1 - s)
    }

    // --- the synthetic file --------------------------------------------------------------------------------------

    /// The f32 values of the 256 E4M3 bytes (`_fp8_e4m3_value`): NaN at 0x7F and 0xFF.
    fn e4m3() -> Vec<f64> {
        (0..256)
            .map(|b| {
                let s = if b & 0x80 == 0 { 1.0 } else { -1.0 };
                let (e, m) = ((b >> 3) & 0xF, b & 7);
                if e == 15 && m == 7 {
                    f64::NAN
                } else if e == 0 {
                    s * f64::from(m) / 8.0 / 64.0
                } else {
                    s * (1.0 + f64::from(m) / 8.0) * f64::from(2u32).powi(e - 7)
                }
            })
            .collect()
    }

    /// The FP8_B128 code nearest `v` (a finite value): the nearest of the 127 non-negative finite
    /// bytes, in ascending magnitude (`lut[0..127]`), the sign in bit 7.
    fn e4m3_code(lut: &[f64], v: f32) -> u8 {
        let a = f64::from(v.abs());
        let above = lut[..127].partition_point(|&m| m < a).min(126);
        let below = above.saturating_sub(1);
        let best = if (lut[above] - a).abs() < (a - lut[below]).abs() {
            above
        } else {
            below
        };
        let code = best as u8;
        if v.is_sign_negative() {
            code | 0x80
        } else {
            code
        }
    }

    /// `rows` rows of `k` values drawn about normal of `sd` as FP8_B128 (132 bytes a row block of
    /// 128: the block's f32 scale, a power of two, then its 128 codes; the rows of a block row
    /// share one scale, as the checkpoint's 128 by 128 blocks do), and the matrix the bytes
    /// dequantize to.
    fn fp8_rows((rows, k): (usize, usize), sd: f32, seed: u64) -> (Vec<u8>, Vec<f64>) {
        let lut = e4m3();
        let mut rng = GateRng(seed);
        let v: Vec<f32> = (0..rows * k).map(|_| normal(&mut rng, sd)).collect();
        let mut bytes = vec![0u8; rows * (k / 128) * 132];
        let mut deq = vec![0.0f64; rows * k];
        for (br, rows_chunk) in v.chunks(128 * k).enumerate() {
            let n = rows_chunk.len() / k;
            for bc in 0..k / 128 {
                let amax = (0..n)
                    .map(|r| {
                        rows_chunk[r * k + bc * 128..r * k + bc * 128 + 128]
                            .iter()
                            .fold(0.0f32, |a, &x| a.max(x.abs()))
                    })
                    .fold(0.0f32, f32::max);
                let scale = if amax == 0.0 {
                    1.0
                } else {
                    2.0f32.powi((amax / 448.0).log2().ceil() as i32)
                };
                for (r, row) in rows_chunk.chunks(k).enumerate() {
                    let at = ((br * 128 + r) * (k / 128) + bc) * 132;
                    bytes[at..at + 4].copy_from_slice(&scale.to_le_bytes());
                    for (c, &x) in row[bc * 128..bc * 128 + 128].iter().enumerate() {
                        let code = e4m3_code(&lut, x / scale);
                        bytes[at + 4 + c] = code;
                        deq[(br * 128 + r) * k + bc * 128 + c] =
                            lut[usize::from(code)] * f64::from(scale);
                    }
                }
            }
        }
        (bytes, deq)
    }

    /// The E2M1 code of `v / d`: the nearest of the eight magnitudes, the sign in bit 3.
    fn e2m1_code(v: f32, d: f32) -> u8 {
        let a = (v / d).abs();
        let idx = if a < 0.25 {
            0
        } else if a < 0.75 {
            1
        } else if a < 1.25 {
            2
        } else if a < 1.75 {
            3
        } else if a < 2.5 {
            4
        } else if a < 3.5 {
            5
        } else if a < 5.0 {
            6
        } else {
            7
        };
        idx | (u8::from(v.is_sign_negative()) * 8)
    }

    /// One MXFP4 row of `k` values (17 bytes a block of 32: the E8M0 scale, then 16 code bytes,
    /// value `j` the low nibble of byte `j` and `j + 16` its high nibble), quantized from draws
    /// about normal of `sd`.
    fn mxfp4_row(row: &mut [u8], k: usize, rng: &mut GateRng, sd: f32) {
        for b in 0..k / 32 {
            let mut v = [0.0f32; 32];
            let mut amax = 0.0f32;
            for x in &mut v {
                *x = normal(rng, sd);
                amax = amax.max(x.abs());
            }
            let e = if amax == 0.0 {
                127
            } else {
                ((amax / 6.0).log2().floor() as i32 + 127).clamp(2, 254)
            };
            let d = 2.0f32.powi(e - 127);
            let at = b * 17;
            row[at] = e as u8;
            for j in 0..16 {
                row[at + 1 + j] = e2m1_code(v[j], d) | (e2m1_code(v[j + 16], d) << 4);
            }
        }
    }

    /// A stack of `experts` experts, `rows` rows of `k` values each, as MXFP4 file bytes: expert
    /// `e` at `e · rows · row_bytes`, row-major.
    fn mxfp4_stack((experts, rows, k): (usize, usize, usize), sd: f32, seed: u64) -> Vec<u8> {
        let row_bytes = k / 32 * 17;
        let mut bytes = vec![0u8; experts * rows * row_bytes];
        let workers = 32;
        let chunk = (experts * rows).div_ceil(workers).max(1);
        std::thread::scope(|s| {
            for (i, part) in bytes.chunks_mut(chunk * row_bytes).enumerate() {
                let seed = seed.wrapping_add((i as u64) * 0x9E37_79B9_7F4A_7C15);
                s.spawn(move || {
                    let mut rng = GateRng(seed);
                    for row in part.chunks_mut(row_bytes) {
                        mxfp4_row(row, k, &mut rng, sd);
                    }
                });
            }
        });
        bytes
    }

    /// One expert's matrix of a stack, dequantized: `rows` rows of `k` values.
    fn expert(bytes: &[u8], (rows, k): (usize, usize), e: usize) -> Vec<f64> {
        let row_bytes = k / 32 * 17;
        let at = e * rows * row_bytes;
        let mut out = vec![0.0f64; rows * k];
        for (r, row) in bytes[at..at + rows * row_bytes]
            .chunks(row_bytes)
            .enumerate()
        {
            for b in 0..k / 32 {
                let blk = &row[b * 17..b * 17 + 17];
                let d = f64::from(2.0f32.powi(i32::from(blk[0]) - 127));
                for j in 0..16 {
                    let (lo, hi) = (usize::from(blk[1 + j] & 15), usize::from(blk[1 + j] >> 4));
                    out[r * k + b * 32 + j] =
                        E2M1[lo & 7] * d * if lo & 8 == 0 { 1.0 } else { -1.0 };
                    out[r * k + b * 32 + 16 + j] =
                        E2M1[hi & 7] * d * if hi & 8 == 0 { 1.0 } else { -1.0 };
                }
            }
        }
        out
    }

    // --- the f64 rules -------------------------------------------------------------------------------------------

    /// `W · x` for a `rows × k` matrix and `k` values, in f64.
    fn gemv(w: &[f64], x: &[f64], (rows, k): (usize, usize)) -> Vec<f64> {
        par_map(rows, |r| {
            w[r * k..r * k + k]
                .iter()
                .zip(x)
                .map(|(&a, &b)| a * b)
                .sum()
        })
    }

    /// SiTU in f64 (`situ`): `β·tanh(g/β)·σ(g) · λ·tanh(u/λ)`.
    fn situ64(g: &[f64], u: &[f64]) -> Vec<f64> {
        g.iter()
            .zip(u)
            .map(|(&g, &u)| {
                let a = (g / f64::from(BETA)).tanh() * f64::from(BETA);
                let s = 1.0 / (1.0 + (-g).exp());
                let v = (u / f64::from(LINEAR_BETA)).tanh() * f64::from(LINEAR_BETA);
                (a * s) * v
            })
            .collect()
    }

    /// RMS norm in f64: `x / sqrt(mean(x²) + eps) · gain`.
    fn rms64(x: &[f64], gain: &[f64]) -> Vec<f64> {
        let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
        x.iter().zip(gain).map(|(&v, &g)| v * inv * g).collect()
    }

    /// The layer's vectors and dense matrices as the f64 rules read them: the two routers and the
    /// three small vectors as f32, the dense projections dequantized.
    struct Vectors {
        w0: Vec<f32>,
        w1: Vec<f32>,
        bias: Vec<f32>,
        escale: Vec<f32>,
        gain: Vec<f64>,
        lat_down: Vec<f64>,
        lat_up: Vec<f64>,
        sh_gate: Vec<f64>,
        sh_up: Vec<f64>,
        sh_down: Vec<f64>,
    }

    /// The layer's three stacks in the file, expert `e`'s matrix of each dequantized on demand.
    struct Stacks<'a> {
        gate: &'a [u8],
        up: &'a [u8],
        down: &'a [u8],
        ff: usize,
    }

    impl Stacks<'_> {
        fn gate(&self, e: usize) -> Vec<f64> {
            expert(self.gate, (self.ff, LATENT), e)
        }

        fn up(&self, e: usize) -> Vec<f64> {
            expert(self.up, (self.ff, LATENT), e)
        }

        fn down(&self, e: usize) -> Vec<f64> {
            expert(self.down, (LATENT, self.ff), e)
        }
    }

    /// The f64 router rule of one row: the two logits, the scores, the picks and their weights.
    struct Route {
        l0: Vec<f64>,
        l1: Vec<f64>,
        p: Vec<f64>,
        ids: Vec<u32>,
        w: Vec<f64>,
        /// How many of the biased top 10 the unbiased one does not hold.
        moved: usize,
        /// The relative gap between the 10th and the 11th biased score.
        gap: f64,
    }

    /// The f64 rule of the router over `x`: the top 10 of `p + bias` (an equal score to the
    /// larger id, the common selection's order), the weights `p[id] / max(Σ p[id], 1e-20) ·
    /// scale[id]`.
    fn route64(v: &Vectors, x: &[f32]) -> Route {
        let x: Vec<f64> = f64s(x);
        let dots = |w: &[f32]| -> Vec<f64> {
            par_map(N_EXPERT, |e| {
                w[e * HIDDEN..e * HIDDEN + HIDDEN]
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| f64::from(a) * b)
                    .sum()
            })
        };
        let (l0, l1) = (dots(&v.w0), dots(&v.w1));
        let sig = |t: f64| 1.0 / (1.0 + (-t).exp());
        let p: Vec<f64> = l0.iter().zip(&l1).map(|(&a, &b)| sig(a) + sig(b)).collect();
        let biased: Vec<f64> = p
            .iter()
            .zip(&v.bias)
            .map(|(&a, &b)| a + f64::from(b))
            .collect();
        let mut order: Vec<usize> = (0..N_EXPERT).collect();
        order.sort_by(|&a, &b| biased[b].partial_cmp(&biased[a]).unwrap().then(b.cmp(&a)));
        let mut plain: Vec<usize> = (0..N_EXPERT).collect();
        plain.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(b.cmp(&a)));
        let ids: Vec<u32> = order[..N_USED].iter().map(|&e| e as u32).collect();
        let sum: f64 = ids.iter().map(|&e| p[e as usize]).sum();
        let den = sum.max(1e-20);
        let w: Vec<f64> = ids
            .iter()
            .map(|&e| p[e as usize] / den * f64::from(v.escale[e as usize]))
            .collect();
        let moved = ids
            .iter()
            .filter(|&&e| !plain[..N_USED].contains(&(e as usize)))
            .count();
        let (k10, k11) = (biased[order[N_USED - 1]], biased[order[N_USED]]);
        let gap = (k10 - k11).abs() / k10.abs().max(1e-300);
        Route {
            l0,
            l1,
            p,
            ids,
            w,
            moved,
            gap,
        }
    }

    /// The block's stages of a run as the card left them: its rows concatenated token-major (the
    /// leg's slot rows slot-major within a token).
    #[derive(Default)]
    struct Stages {
        lat: Vec<f32>,
        act: Vec<f32>,
        down: Vec<f32>,
        routed: Vec<f32>,
        normed: Vec<f32>,
        lat_up: Vec<f32>,
        shared: Vec<f32>,
        out: Vec<f32>,
    }

    /// The f64 rules of the block's stages over a run's rows ([`ladder64`]'s).
    struct Stages64 {
        lat: Vec<f64>,
        act: Vec<f64>,
        down: Vec<f64>,
        routed: Vec<f64>,
        normed: Vec<f64>,
        lat_up: Vec<f64>,
        shared: Vec<f64>,
        out: Vec<f64>,
    }

    /// The f64 rules of the block's stages over the card's own rows: the latent row from `x`, the
    /// slots' SiTU rows from the card's latent rows in q8_1, the down rows from the card's SiTU
    /// rows in q8_1, the routed sum in slot order, the norm, the latent up from the card's normed
    /// rows, the shared expert from `x`, the output from the card's two halves.
    fn ladder64(
        (v, st): (&Vectors, &Stacks<'_>),
        ff: usize,
        xs: &[Vec<f32>],
        (ids, ws): (&[Vec<u32>], &[Vec<f32>]),
        got: &Stages,
    ) -> Stages64 {
        let tokens = xs.len();
        let x64: Vec<Vec<f64>> = xs.iter().map(|x| f64s(x)).collect();
        let lat: Vec<f64> = x64
            .iter()
            .flat_map(|x| gemv(&v.lat_down, x, (LATENT, HIDDEN)))
            .collect();
        let latq: Vec<Vec<f64>> = (0..tokens)
            .map(|t| q8(&got.lat[t * LATENT..t * LATENT + LATENT], (LATENT, 1)))
            .collect();
        let act: Vec<f64> = par_map(tokens * N_USED, |i| {
            let (t, s) = (i / N_USED, i % N_USED);
            let e = ids[t][s] as usize;
            situ64(
                &gemv(&st.gate(e), &latq[t], (ff, LATENT)),
                &gemv(&st.up(e), &latq[t], (ff, LATENT)),
            )
        })
        .into_iter()
        .flatten()
        .collect();
        let down: Vec<f64> = par_map(tokens * N_USED, |i| {
            let e = ids[i / N_USED][i % N_USED] as usize;
            let hq = q8(&got.act[i * ff..i * ff + ff], (ff, 1));
            gemv(&st.down(e), &hq, (LATENT, ff))
        })
        .into_iter()
        .flatten()
        .collect();
        let routed: Vec<f64> = (0..tokens * LATENT)
            .map(|i| {
                let (t, j) = (i / LATENT, i % LATENT);
                (0..N_USED)
                    .map(|s| f64::from(ws[t][s]) * down[(t * N_USED + s) * LATENT + j])
                    .sum()
            })
            .collect();
        let normed: Vec<f64> = (0..tokens)
            .flat_map(|t| rms64(&f64s(&got.routed[t * LATENT..t * LATENT + LATENT]), &v.gain))
            .collect();
        let lat_up: Vec<f64> = (0..tokens)
            .flat_map(|t| {
                gemv(
                    &v.lat_up,
                    &f64s(&got.normed[t * LATENT..t * LATENT + LATENT]),
                    (HIDDEN, LATENT),
                )
            })
            .collect();
        let shared: Vec<f64> = (0..tokens)
            .flat_map(|t| {
                let h = situ64(
                    &gemv(&v.sh_gate, &x64[t], (ff, HIDDEN)),
                    &gemv(&v.sh_up, &x64[t], (ff, HIDDEN)),
                );
                gemv(&v.sh_down, &h, (HIDDEN, ff))
            })
            .collect();
        let out: Vec<f64> = f64s(&got.lat_up)
            .iter()
            .zip(&f64s(&got.shared))
            .map(|(&a, &b)| a + b)
            .collect();
        Stages64 {
            lat,
            act,
            down,
            routed,
            normed,
            lat_up,
            shared,
            out,
        }
    }

    /// The worst relative error of each stage against its rule so far.
    #[derive(Default)]
    struct Worst([f64; 8]);

    impl Worst {
        /// See one run's stages against their rules.
        fn see(&mut self, got: &Stages, want: &Stages64) {
            let pairs = [
                (&got.lat, &want.lat),
                (&got.act, &want.act),
                (&got.down, &want.down),
                (&got.routed, &want.routed),
                (&got.normed, &want.normed),
                (&got.lat_up, &want.lat_up),
                (&got.shared, &want.shared),
                (&got.out, &want.out),
            ];
            for (r, (g, w)) in self.0.iter_mut().zip(pairs) {
                *r = r.max(rel32(g, w));
            }
        }

        /// A clause a stage: the two sums under [`SUM_BAND`], the rest under `band`.
        fn clauses(&self, tag: &str, band: f64) -> Vec<bool> {
            STAGE_NAMES
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let band = if matches!(i, 3 | 7) { SUM_BAND } else { band };
                    banded(&format!("{tag}{s}"), self.0[i], band)
                })
                .collect()
        }
    }

    // --- the device side -----------------------------------------------------------------------------------------

    /// The clauses' shared device state.
    struct Dev {
        gpu: Gpu,
        kgemv: KGemvKernels,
        g32: Gemm32Kernels,
        gemm: GemmKernels,
        flash: FlashGqaKernels,
    }

    impl Dev {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn up<T: DeviceCopy>(&self, v: &[T]) -> Result<DeviceBuffer<T>, GateError> {
            Ok(DeviceBuffer::from_host(self.stream(), v)?)
        }

        fn zeroed(&self, n: usize) -> Result<DeviceBuffer<f32>, GateError> {
            Ok(DeviceBuffer::zeroed(self.stream(), n)?)
        }

        fn read<T: DeviceCopy>(&self, b: &DeviceBuffer<T>) -> Result<Vec<T>, GateError> {
            self.stream().synchronize()?;
            Ok(b.to_host_vec(self.stream())?)
        }

        fn cx<'a>(&'a self, w: &'a Weights) -> GqaCx<'a> {
            GqaCx {
                what: NAME,
                gpu: &self.gpu,
                w,
                kgemv: &self.kgemv,
                g32: &self.g32,
                flash: &self.flash,
                mma: false,
                out_gate: None,
                hidden: HIDDEN,
                eps: EPS,
                n_kv: 1,
                ctx: 1,
            }
        }
    }

    /// The stages of the last `m`-row launch read back through `bufs` (it synchronizes the
    /// stream): `out`'s first `m` rows included.
    fn read_stages(
        dev: &Dev,
        bufs: &MoeBufs,
        m: usize,
        out: &DeviceBuffer<f32>,
    ) -> Result<Stages, GateError> {
        let tap = |t: Tap| -> Result<Vec<f32>, GateError> { Ok(bufs.tap((dev.stream(), m), t)?) };
        Ok(Stages {
            lat: tap(Tap::Lat)?,
            act: tap(Tap::Act)?,
            down: tap(Tap::Down)?,
            routed: tap(Tap::Routed)?,
            normed: tap(Tap::Normed)?,
            lat_up: tap(Tap::LatUp)?,
            shared: tap(Tap::Shared)?,
            out: dev.read(out)?[..m * HIDDEN].to_vec(),
        })
    }

    /// One token's router read back: the two logits, the scores, the ids, the weights.
    type RouterGot = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>);

    /// What [`edge_pick`] returns: the ids, the weights, the scores.
    type EdgePick = (Vec<u32>, Vec<f32>, Vec<f32>);

    /// The router clauses over a run's tokens: `routes[t]` the f64 rule of token `t` and `got[t]`
    /// the card's (the two logits, the scores, the ids, the weights).
    fn router_clauses(
        routes: &[Route],
        got: &[RouterGot],
        (tag, skipped): (&str, usize),
    ) -> Vec<bool> {
        let mut ids_ok = true;
        let mut moved = 0;
        let mut min_gap = f64::INFINITY;
        let mut l = [f64::NEG_INFINITY; 3];
        let mut w = f64::NEG_INFINITY;
        for (t, (r, (l0, l1, p, ids, ws))) in routes.iter().zip(got).enumerate() {
            if ids.len() != N_USED || ids.iter().zip(&r.ids).any(|(&a, &b)| a != b) {
                ids_ok = false;
                println!(
                    "{NAME} {tag}router_ids: token {t}: got {ids:?}, the rule {:?}",
                    r.ids
                );
            }
            moved += r.moved;
            min_gap = min_gap.min(r.gap);
            l[0] = l[0].max(rel32(l0, &r.l0));
            l[1] = l[1].max(rel32(l1, &r.l1));
            l[2] = l[2].max(rel32(p, &r.p));
            w = w.max(rel32(ws, &r.w));
        }
        vec![
            clause(
                &format!("{tag}router_ids"),
                ids_ok && moved > 0,
                &format!(
                    "{} tokens ({skipped} rows skipped, smallest cut gap {min_gap:.3e}), {moved} biased picks the unbiased top 10 does not hold",
                    routes.len()
                ),
            ),
            banded(&format!("{tag}router_logits"), l[0].max(l[1]), ROUTER_BAND),
            banded(&format!("{tag}router_scores"), l[2], ROUTER_BAND),
            banded(&format!("{tag}router_weights"), w, ROUTER_BAND),
        ]
    }

    /// The router launched alone over the row `x = e0` through a pair of the edge routers (`a`,
    /// `b`: a matrix whose every row is one logit at column 0), the bias zero and the scale the
    /// file's: the ids, the weights and the scores.
    fn edge_pick(
        dev: &Dev,
        (router, edge): (&split::RouterKernels, &Weights),
        x: &DeviceBuffer<f32>,
        out: &mut split::RouterOut,
        (a, b): (&str, &str),
    ) -> Result<EdgePick, GateError> {
        router.enqueue_router(
            dev.stream(),
            (edge.f32_tensor(NAME, a)?, edge.f32_tensor(NAME, b)?),
            x,
            edge.f32_buf(NAME, "edge_bias")?,
            edge.f32_buf(NAME, "edge_scale")?,
            out,
            dev.gpu.layer_sink(LAYER)?,
        )?;
        Ok((
            out.ids.to_host_vec(dev.stream())?,
            out.weights.to_host_vec(dev.stream())?,
            out.probs.to_host_vec(dev.stream())?,
        ))
    }

    /// An f32 tensor of the edge routers' weights as a part: `bytes` its values.
    fn f32_part<'a>(name: &str, dims: Vec<u64>, bytes: &'a [u8]) -> TensorPart<'a> {
        TensorPart {
            info: TensorInfo {
                name: name.to_string(),
                dims,
                ty: GgmlType::F32,
                offset: 0,
                nbytes: bytes.len() as u64,
            },
            spans: vec![bytes],
            scale: None,
        }
    }

    /// The floor, zero and ties clauses.
    fn router_edges(dev: &Dev, escale: &[f32]) -> Result<Vec<bool>, GateError> {
        let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let router_bytes = |logit: f32| {
            let mut m = vec![0.0f32; N_EXPERT * HIDDEN];
            for e in 0..N_EXPERT {
                m[e * HIDDEN] = logit;
            }
            f32_bytes(&m)
        };
        let (floor, zero, tie) = (
            router_bytes(FLOOR_LOGIT),
            router_bytes(ZERO_LOGIT),
            router_bytes(0.0),
        );
        let (bias, scale) = (f32_bytes(&vec![0.0f32; N_EXPERT]), f32_bytes(escale));
        let (e, n) = (N_EXPERT as u64, HIDDEN as u64);
        let mut edge = Weights::default();
        edge.add_parts(
            dev.stream(),
            &[
                f32_part("edge_r0_floor", vec![n, e], &floor),
                f32_part("edge_r1_floor", vec![n, e], &floor),
                f32_part("edge_r0_zero", vec![n, e], &zero),
                f32_part("edge_r1_zero", vec![n, e], &zero),
                f32_part("edge_r0_tie", vec![n, e], &tie),
                f32_part("edge_r1_tie", vec![n, e], &tie),
                f32_part("edge_bias", vec![e], &bias),
                f32_part("edge_scale", vec![e], &scale),
            ],
        )?;
        let router = split::RouterKernels::load(dev.gpu.context())?;
        let mut x = vec![0.0f32; HIDDEN];
        x[0] = 1.0;
        let x = dev.up(&x)?;
        let mut out = split::RouterOut::new(dev.stream())?;
        let escale64 = f64s(escale);
        let mut ok = Vec::new();
        // The floor: every score 2·σ(−60) = 1.75e-26, the kept ten 1.75e-25 under the 1e-20 divisor.
        let (ids, w, _) = edge_pick(
            dev,
            (&router, &edge),
            &x,
            &mut out,
            ("edge_r0_floor", "edge_r1_floor"),
        )?;
        let p = 2.0 / (1.0 + f64::from(-FLOOR_LOGIT).exp());
        let want: Vec<f64> = ids
            .iter()
            .map(|&e| p / 1e-20 * escale64[e as usize])
            .collect();
        let renorm: f64 = w
            .iter()
            .zip(&ids)
            .map(|(&x, &e)| f64::from(x) / escale64[e as usize])
            .sum();
        ok.push(clause(
            "router_floor_ids",
            tie_order(&ids),
            &format!("{ids:?}"),
        ));
        ok.push(banded(
            "router_floor_weights",
            rel32(&w, &want),
            ROUTER_BAND,
        ));
        ok.push(clause(
            "router_floor_divisor",
            (1e-7..1e-4).contains(&renorm),
            &format!(
                "the renormalized weights sum to {renorm:.3e} (the divisor floored at 1e-20: 1.75e-5; \
                 renormalized to one: 1.0)"
            ),
        ));
        // The zero: every f32 sigmoid 0, every weight 0.
        let (ids, w, _) = edge_pick(
            dev,
            (&router, &edge),
            &x,
            &mut out,
            ("edge_r0_zero", "edge_r1_zero"),
        )?;
        let max_w = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        ok.push(clause(
            "router_zero_weights",
            max_w < ZERO_WEIGHT && tie_order(&ids),
            &format!("largest |w| {max_w:.3e}, ids {ids:?}"),
        ));
        // Ties: every score 1, the larger id first, the weights 1/10 · scale[id].
        let (ids, w, _) = edge_pick(
            dev,
            (&router, &edge),
            &x,
            &mut out,
            ("edge_r0_tie", "edge_r1_tie"),
        )?;
        let want: Vec<f64> = ids.iter().map(|&e| 0.1 * escale64[e as usize]).collect();
        ok.push(clause(
            "router_ties_ids",
            tie_order(&ids),
            &format!("{ids:?}"),
        ));
        ok.push(banded("router_ties_weights", rel32(&w, &want), ROUTER_BAND));
        drop((x, out, edge));
        Ok(ok)
    }

    /// The synthetic layer's file of expert width `ff` at `dir`: every tensor of the block (`blk.0.`'s
    /// two routers, the bias, the learned scale, the latent pair, the norm's gain, the shared
    /// expert's three, the three stacks of all 512 experts), written expert by expert; and its
    /// vectors and dense matrices as the f64 rules read them.
    fn write_file(dir: &Path, ff: usize) -> Result<(PathBuf, Vectors), GateError> {
        let mut rng = GateRng(SEED);
        let draw = |rng: &mut GateRng, n: usize, sd: f32| -> Vec<f32> {
            (0..n).map(|_| normal(rng, sd)).collect()
        };
        let uni = |rng: &mut GateRng, n: usize, lo: f32, hi: f32| -> Vec<f32> {
            (0..n).map(|_| rng.uni(lo, hi)).collect()
        };
        let (lat_down_b, lat_down) = fp8_rows((LATENT, HIDDEN), 0.02, SEED + 0x10);
        let (lat_up_b, lat_up) = fp8_rows((HIDDEN, LATENT), 0.02, SEED + 0x11);
        let (sh_gate_b, sh_gate) = fp8_rows((ff, HIDDEN), 0.02, SEED + 0x12);
        let (sh_up_b, sh_up) = fp8_rows((ff, HIDDEN), 0.02, SEED + 0x13);
        let (sh_down_b, sh_down) = fp8_rows((HIDDEN, ff), 0.02, SEED + 0x14);
        let v = Vectors {
            w0: draw(&mut rng, N_EXPERT * HIDDEN, 2.0 / (HIDDEN as f32).sqrt()),
            w1: draw(&mut rng, N_EXPERT * HIDDEN, 2.0 / (HIDDEN as f32).sqrt()),
            bias: uni(&mut rng, N_EXPERT, -0.5, 0.5),
            escale: uni(&mut rng, N_EXPERT, 1.0, 4.0),
            gain: f64s(&uni(&mut rng, LATENT, 0.5, 1.5)),
            lat_down,
            lat_up,
            sh_gate,
            sh_up,
            sh_down,
        };
        let gain32: Vec<f32> = v.gain.iter().map(|&g| g as f32).collect();
        let f32_decl = |name: String, dims: Vec<u64>, values: &[f32]| {
            (
                name,
                dims,
                GgmlType::F32,
                values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
        };
        let (e, n, lat, f) = (N_EXPERT as u64, HIDDEN as u64, LATENT as u64, ff as u64);
        let dense = vec![
            f32_decl(names::ffn_gate_inp(LAYER), vec![n, e], &v.w0),
            f32_decl(names::ffn_gate_inp_1(LAYER), vec![n, e], &v.w1),
            f32_decl(names::exp_probs_b(LAYER), vec![e], &v.bias),
            f32_decl(names::ffn_exp_scale(LAYER), vec![e], &v.escale),
            f32_decl(names::ffn_routed_norm(LAYER), vec![lat], &gain32),
            (
                names::ffn_routed_down(LAYER),
                vec![n, lat],
                GgmlType::FP8_B128,
                lat_down_b,
            ),
            (
                names::ffn_routed_up(LAYER),
                vec![lat, n],
                GgmlType::FP8_B128,
                lat_up_b,
            ),
            (
                names::ffn_gate_shexp(LAYER),
                vec![n, f],
                GgmlType::FP8_B128,
                sh_gate_b,
            ),
            (
                names::ffn_up_shexp(LAYER),
                vec![n, f],
                GgmlType::FP8_B128,
                sh_up_b,
            ),
            (
                names::ffn_down_shexp(LAYER),
                vec![f, n],
                GgmlType::FP8_B128,
                sh_down_b,
            ),
        ];
        let stacks = [
            (names::ffn_gate_exps(LAYER), (LATENT, ff)),
            (names::ffn_up_exps(LAYER), (LATENT, ff)),
            (names::ffn_down_exps(LAYER), (ff, LATENT)),
        ];
        let tensors = dense
            .iter()
            .map(|(name, dims, ty, bytes)| gguf::write::TensorDecl {
                name: name.clone(),
                dims: dims.clone(),
                type_id: ty.as_u32(),
                nbytes: bytes.len() as u64,
            })
            .chain(
                stacks
                    .iter()
                    .map(|(name, (k, rows))| gguf::write::TensorDecl {
                        name: name.clone(),
                        dims: vec![*k as u64, *rows as u64, e],
                        type_id: GgmlType::MXFP4.as_u32(),
                        nbytes: (N_EXPERT * rows * k / 32 * 17) as u64,
                    }),
            )
            .collect();
        let path = dir.join(format!("ernie55-moe-{ff}.gguf"));
        let layout = gguf::write::Layout::new(&[], tensors)?;
        let mut writer = gguf::write::Writer::new(
            std::io::BufWriter::new(std::fs::File::create(&path)?),
            layout,
        )?;
        for (name, _, _, bytes) in &dense {
            writer.tensor(name, bytes)?;
        }
        for (i, (name, (k, rows))) in stacks.iter().enumerate() {
            let bytes = mxfp4_stack((N_EXPERT, *rows, *k), 0.03, SEED + 0x20 + i as u64);
            writer.tensor(name, &bytes)?;
        }
        writer.finish()?;
        Ok((path, v))
    }

    // --- the run -------------------------------------------------------------------------------------------------

    /// Run every clause at both widths.
    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!("{NAME}: device {}", gpu.device_name()?);
        let dev = Dev {
            kgemv: KGemvKernels::load(&gpu)?,
            g32: Gemm32Kernels::load(gpu.context())?,
            gemm: GemmKernels::load(gpu.context())?,
            flash: FlashGqaKernels::load(gpu.context())?,
            gpu,
        };
        let dir = std::env::temp_dir().join(format!("bloomery-ernie55-moe-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome =
            WIDTHS
                .iter()
                .try_fold(Vec::new(), |mut ok, &ff| -> Result<Vec<bool>, GateError> {
                    ok.extend(width(&dev, &dir, ff)?);
                    Ok(ok)
                });
        let cleanup =
            std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup {}: {e}", dir.display()));
        let ok = outcome?;
        cleanup?;
        if ok.iter().all(|p| *p) {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// The synthetic widths of expert width `ff`.
    fn dims(ff: usize) -> MoeDims {
        MoeDims {
            hidden: HIDDEN,
            latent: LATENT,
            expert_ff: ff,
            shared_ff: ff,
            n_expert: N_EXPERT,
            n_used: N_USED,
            weights_norm: true,
            eps: EPS,
            situ_beta: BETA,
            situ_linear_beta: LINEAR_BETA,
        }
    }

    /// The layer's tensors on the card as a placed load makes them (`parts::load` over the file):
    /// every tensor whole but the three expert stacks, which a program's plan holds as the file's
    /// blocks (`CardFormat::KQuant`, the format MXFP4 has no default for) and of which the card
    /// takes the first `experts`.
    fn card_weights(dev: &Dev, split: &Split, experts: usize) -> Result<Weights, GateError> {
        let stack = |name: &str| name.contains("_exps");
        let list = ExpertList::range(0..experts as u64).map_err(|e| format!("{e}"))?;
        let rule = |name: &str, _: GgmlType, _: &[u64]| -> Result<Take, String> {
            Ok(if stack(name) && experts < N_EXPERT {
                Take::Experts(list.clone())
            } else {
                Take::Whole
            })
        };
        let card = |t: &TensorInfo| stack(&t.name).then_some(CardFormat::KQuant);
        Ok(parts::load(dev.stream(), split, rule, card, false)?)
    }

    /// The clauses at expert width `ff`.
    fn width(dev: &Dev, dir: &Path, ff: usize) -> Result<Vec<bool>, GateError> {
        let dims = dims(ff);
        dims.check()?;
        let (path, v) = write_file(dir, ff)?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let stack_names = [
            names::ffn_gate_exps(LAYER),
            names::ffn_up_exps(LAYER),
            names::ffn_down_exps(LAYER),
        ];
        let stacks = Stacks {
            gate: split.data(&stack_names[0])?,
            up: split.data(&stack_names[1])?,
            down: split.data(&stack_names[2])?,
            ff,
        };
        let mut rng = GateRng(SEED ^ 0x544f_4b45);
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(TOKENS);
        let mut routes: Vec<Route> = Vec::with_capacity(TOKENS);
        let mut drawn = 0usize;
        while xs.len() < TOKENS {
            if drawn == MAX_DRAWN {
                return Err(format!(
                    "{MAX_DRAWN} rows drawn, {} of {TOKENS} with a cut gap of {MIN_GAP:.0e}",
                    xs.len()
                )
                .into());
            }
            drawn += 1;
            let x: Vec<f32> = (0..HIDDEN).map(|_| normal(&mut rng, 1.0)).collect();
            let r = route64(&v, &x);
            if r.gap >= MIN_GAP {
                xs.push(x);
                routes.push(r);
            }
        }
        let skipped = drawn - TOKENS;
        let tag = format!("w{ff} ");
        let mut ok: Vec<bool> = Vec::new();
        let sink = dev.gpu.layer_sink(LAYER)?;

        // --- the whole-card block ---------------------------------------------------------------------------
        let w_all = card_weights(dev, &split, N_EXPERT)?;
        let (moe, mut bufs) = Moe::load_layers(&dev.gpu, &split, &w_all, &dims, &[LAYER], TOKENS)?;
        let cx = dev.cx(&w_all);
        let x_t: Vec<DeviceBuffer<f32>> = xs.iter().map(|x| dev.up(x)).collect::<Result<_, _>>()?;
        let mut got_routes = Vec::new();
        let mut steps = Stages::default();
        let mut ids_t: Vec<Vec<u32>> = Vec::new();
        let mut ws_t: Vec<Vec<f32>> = Vec::new();
        let mut worst = Worst::default();
        for (t, x) in x_t.iter().enumerate() {
            let mut out = dev.zeroed(HIDDEN)?;
            moe.enqueue_step((&dev.gpu, &w_all, &cx), &mut bufs, LAYER, x, &mut out)?;
            let (ids, weights) = bufs.picks(dev.stream())?;
            let RouterRead {
                logits_0: l0,
                logits_1: l1,
                probs: p,
            } = bufs.scores(dev.stream())?;
            got_routes.push((l0, l1, p, ids.clone(), weights.clone()));
            let got = read_stages(dev, &bufs, 1, &out)?;
            worst.see(
                &got,
                &ladder64(
                    (&v, &stacks),
                    ff,
                    &xs[t..t + 1],
                    (std::slice::from_ref(&ids), std::slice::from_ref(&weights)),
                    &got,
                ),
            );
            ids_t.push(ids);
            ws_t.push(weights);
            let Stages {
                lat,
                act,
                down,
                routed,
                normed,
                lat_up,
                shared,
                out,
            } = got;
            steps.lat.extend(lat);
            steps.act.extend(act);
            steps.down.extend(down);
            steps.routed.extend(routed);
            steps.normed.extend(normed);
            steps.lat_up.extend(lat_up);
            steps.shared.extend(shared);
            steps.out.extend(out);
        }
        ok.extend(router_clauses(&routes, &got_routes, (&tag, skipped)));
        ok.extend(worst.clauses(&format!("{tag}step_"), STAGE_BAND));
        // The unit.
        let x_unit = dev.up(&xs.concat())?;
        let mut out_u = dev.zeroed(TOKENS * HIDDEN)?;
        moe.enqueue_prompt(
            (&dev.gpu, &w_all, &cx, &dev.gemm),
            &mut bufs,
            LAYER,
            TOKENS,
            &x_unit,
            &mut out_u,
        )?;
        let unit = read_stages(dev, &bufs, TOKENS, &out_u)?;
        let (ids_u, w_u) = bufs.picks_rows((dev.stream(), TOKENS))?;
        let probs_u = bufs.tap((dev.stream(), TOKENS), Tap::Probs)?;
        ok.push(clause(
            &format!("{tag}prompt_ids_is_step"),
            ids_u == ids_t.concat(),
            &format!("{TOKENS} rows"),
        ));
        let mut scores = f64::NEG_INFINITY;
        let mut weights = f64::NEG_INFINITY;
        for (t, r) in routes.iter().enumerate() {
            scores = scores.max(rel32(&probs_u[t * N_EXPERT..t * N_EXPERT + N_EXPERT], &r.p));
            weights = weights.max(rel32(&w_u[t * N_USED..t * N_USED + N_USED], &r.w));
        }
        ok.push(banded(
            &format!("{tag}prompt_router_scores"),
            scores,
            ROUTER_BAND,
        ));
        ok.push(banded(
            &format!("{tag}prompt_router_weights"),
            weights,
            ROUTER_BAND,
        ));
        let pairs = [
            (&unit.lat, &steps.lat),
            (&unit.act, &steps.act),
            (&unit.down, &steps.down),
            (&unit.routed, &steps.routed),
            (&unit.normed, &steps.normed),
            (&unit.lat_up, &steps.lat_up),
            (&unit.shared, &steps.shared),
            (&unit.out, &steps.out),
        ];
        for (i, (g, s)) in pairs.iter().enumerate() {
            ok.push(banded(
                &format!("{tag}prompt_is_step_{i}"),
                rel32(g, &f64s(s)),
                PROMPT_BAND,
            ));
        }
        // The fault.
        let mut x_nan = xs[0].clone();
        x_nan[7] = f32::NAN;
        let x_nan = dev.up(&x_nan)?;
        let mut out_nan = dev.zeroed(HIDDEN)?;
        moe.enqueue_step(
            (&dev.gpu, &w_all, &cx),
            &mut bufs,
            LAYER,
            &x_nan,
            &mut out_nan,
        )?;
        let fault = dev.gpu.fault()?;
        let router_bit = 1u32 << (FaultSite::Router as u32);
        ok.push(clause(
            &format!("{tag}fault_word"),
            matches!(&fault, Some(f) if usize::try_from(f.layer).ok() == Some(LAYER) && f.sites & router_bit != 0),
            &format!("{fault:?}"),
        ));
        let out_nan = dev.read(&out_nan)?;
        let nans = out_nan.iter().filter(|v| v.is_nan()).count();
        ok.push(clause(
            &format!("{tag}fault_output_nan"),
            nans == HIDDEN,
            &format!("{nans} of {HIDDEN} values NaN"),
        ));
        dev.gpu.clear_fault()?;
        ok.push(clause(
            &format!("{tag}fault_cleared"),
            dev.gpu.fault()?.is_none(),
            "the word reads none after clear_fault",
        ));
        // The refusals.
        let mut out = dev.zeroed(HIDDEN)?;
        ok.push(refused(
            &format!("{tag}width"),
            MoeDims {
                expert_ff: 100,
                ..dims
            }
            .check(),
        ));
        ok.push(refused(
            &format!("{tag}router_count"),
            MoeDims { n_used: 8, ..dims }.check(),
        ));
        ok.push(refused(
            &format!("{tag}no_block"),
            moe.enqueue_step(
                (&dev.gpu, &w_all, &cx),
                &mut bufs,
                LAYER + 1,
                &x_t[0],
                &mut out,
            ),
        ));
        let x_over = dev.zeroed((TOKENS + 1) * HIDDEN)?;
        let mut out_over = dev.zeroed((TOKENS + 1) * HIDDEN)?;
        ok.push(refused(
            &format!("{tag}unit_rows"),
            moe.enqueue_prompt(
                (&dev.gpu, &w_all, &cx, &dev.gemm),
                &mut bufs,
                LAYER,
                TOKENS + 1,
                &x_over,
                &mut out_over,
            ),
        ));
        ok.push(refused(
            &format!("{tag}back_bare"),
            moe.enqueue_back(
                (&dev.gpu, &w_all, &cx),
                &mut bufs,
                LAYER,
                (false, None),
                &mut out,
            ),
        ));
        let mut lat_short = dev.zeroed(LATENT - 1)?;
        ok.push(refused(
            &format!("{tag}front_short_lat"),
            moe.enqueue_front(
                (&dev.gpu, &w_all, &cx),
                &mut bufs,
                LAYER,
                &x_t[0],
                (None, &mut lat_short),
            ),
        ));
        let router = split::RouterKernels::load(dev.gpu.context())?;
        let (w0, w1) = (
            w_all.f32_tensor(NAME, &names::ffn_gate_inp(LAYER))?,
            w_all.f32_tensor(NAME, &names::ffn_gate_inp_1(LAYER))?,
        );
        let (bias_b, scale_b) = (
            w_all.f32_buf(NAME, &names::exp_probs_b(LAYER))?,
            w_all.f32_buf(NAME, &names::ffn_exp_scale(LAYER))?,
        );
        let mut p_b = dev.zeroed(TOKENS * N_EXPERT)?;
        let mut i_b = DeviceBuffer::zeroed(dev.stream(), TOKENS * N_USED)?;
        let mut w_b = dev.zeroed(TOKENS * N_USED)?;
        ok.push(refused(
            &format!("{tag}rows_no_logits"),
            router.enqueue_router_rows(
                (dev.stream(), &dev.g32),
                (w0, w1),
                &x_unit,
                bias_b,
                scale_b,
                TOKENS,
                RoutedRows {
                    probs: &mut p_b,
                    logits: None,
                    ids: &mut i_b,
                    weights: &mut w_b,
                },
                sink,
            ),
        ));
        ok.extend(router_edges(dev, &v.escale)?);
        drop((w0, w1, bias_b, scale_b, router, cx));
        drop((
            bufs, moe, w_all, p_b, i_b, w_b, lat_short, out_nan, out, x_over, out_over,
        ));

        // --- the split: the card holds the first N_EXPERT − N_HOST experts, the host tier the rest ---------------
        let n_card = N_EXPERT - N_HOST;
        let w_card = card_weights(dev, &split, n_card)?;
        let (moe_c, mut bufs_c) =
            Moe::load_layers(&dev.gpu, &split, &w_card, &dims, &[LAYER], TOKENS)?;
        let cx_c = dev.cx(&w_card);
        let src = R8Source::rows(&split);
        let host = host_layer(src, &dims, LAYER)?;
        ok.push(clause(
            &format!("{tag}split_card_stacks"),
            check_card_stacks(&w_card, &dims, LAYER, n_card).is_ok(),
            &format!("the card's stacks hold the first {n_card} experts"),
        ));
        ok.push(refused(
            &format!("{tag}split_card_stacks_count"),
            check_card_stacks(&w_card, &dims, LAYER, n_card - 1),
        ));
        let mut hs = HostScratch::new(LATENT, ff, N_USED)?;
        let mut us = UnionScratch::new_routed(LATENT, ff, TOKENS, N_USED)?;
        // The slots' places: the card's slot of an expert it holds (the first `n_card`, in order),
        // the host mark of the rest; any other place the card's sum answers with NaN and a fault.
        let places = |ids: &[u32]| -> Vec<u32> {
            ids.iter()
                .map(|&e| if (e as usize) < n_card { e } else { HOST })
                .collect()
        };
        let host_list = |ids: &[u32], ws: &[f32]| -> Vec<(u32, f32)> {
            ids.iter()
                .zip(ws)
                .filter(|(e, _)| **e as usize >= n_card)
                .map(|(&e, &w)| (e, w))
                .collect()
        };
        let mut all_ran = true;
        let mut host_picks = 0usize;
        let mut split_routed = Vec::new();
        let mut split_out = Vec::new();
        let mut host_steps = Vec::new();
        for x in &x_t {
            let mut lat = dev.zeroed(LATENT)?;
            moe_c.enqueue_front(
                (&dev.gpu, &w_card, &cx_c),
                &mut bufs_c,
                LAYER,
                x,
                (None, &mut lat),
            )?;
            let lat_host = dev.read(&lat)?;
            let (ids, w) = bufs_c.picks(dev.stream())?;
            let list = host_list(&ids, &w);
            host_picks += list.len();
            let mut hsum = vec![0.0f32; LATENT];
            if !list.is_empty() {
                host.experts_into(
                    src,
                    &Tensor2::from_vec(LATENT, 1, lat_host),
                    &list,
                    &mut hsum,
                    &mut hs,
                )?;
            }
            let sel = dev.up(&places(&ids))?;
            let hsum_d = dev.up(&hsum)?;
            all_ran &= moe_c.enqueue_shadow(
                (&dev.gpu, &w_card, &cx_c),
                &mut bufs_c,
                LAYER,
                x,
                &lat,
                (&sel, n_card),
            )?;
            let mut out = dev.zeroed(HIDDEN)?;
            moe_c.enqueue_back(
                (&dev.gpu, &w_card, &cx_c),
                &mut bufs_c,
                LAYER,
                (true, Some(&hsum_d)),
                &mut out,
            )?;
            split_routed.extend(bufs_c.tap((dev.stream(), 1), Tap::Summed)?);
            split_out.extend(dev.read(&out)?);
            host_steps.extend(hsum);
        }
        ok.push(clause(
            &format!("{tag}split_step_ran"),
            all_ran,
            &format!("{TOKENS} steps, the card's slots ran"),
        ));
        ok.push(clause(
            &format!("{tag}split_step_host"),
            host_picks > 0,
            &format!("{host_picks} host picks over {TOKENS} tokens"),
        ));
        ok.push(banded(
            &format!("{tag}split_step_routed"),
            rel32(&split_routed, &f64s(&steps.routed)),
            SPLIT_BAND,
        ));
        ok.push(banded(
            &format!("{tag}split_step_out"),
            rel32(&split_out, &f64s(&steps.out)),
            SPLIT_BAND,
        ));
        // The unit, split: the latent rows are the buffers' own, where the batch port's download reads them.
        moe_c.enqueue_front_rows(
            (&dev.gpu, &w_card, &cx_c, &dev.gemm),
            &mut bufs_c,
            LAYER,
            TOKENS,
            &x_unit,
        )?;
        let lat_all = dev.read(bufs_c.lat())?;
        let (ids_u2, w_u2) = bufs_c.picks_rows((dev.stream(), TOKENS))?;
        let lists: Vec<Vec<(u32, f32)>> = (0..TOKENS)
            .map(|t| {
                host_list(
                    &ids_u2[t * N_USED..t * N_USED + N_USED],
                    &w_u2[t * N_USED..t * N_USED + N_USED],
                )
            })
            .collect();
        let host_picks_u: usize = lists.iter().map(Vec::len).sum();
        let list_refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut hsum_all = vec![0.0f32; TOKENS * LATENT];
        if host_picks_u > 0 {
            host.experts_union_into(
                src,
                &Tensor2::from_vec(LATENT, TOKENS, lat_all),
                &list_refs,
                &mut hsum_all,
                &mut us,
            )?;
        }
        let sel_u = dev.up(&places(&ids_u2))?;
        let hsum_u = dev.up(&hsum_all)?;
        let ran = moe_c.enqueue_shadow_rows(
            (&dev.gpu, &w_card, &cx_c, &dev.gemm),
            &mut bufs_c,
            LAYER,
            TOKENS,
            &x_unit,
            (&sel_u, n_card),
        )?;
        let mut out_su = dev.zeroed(TOKENS * HIDDEN)?;
        moe_c.enqueue_back_rows(
            (&dev.gpu, &w_card, &cx_c, &dev.gemm),
            &mut bufs_c,
            LAYER,
            TOKENS,
            (true, Some(&hsum_u)),
            &mut out_su,
        )?;
        let summed_u = bufs_c.tap((dev.stream(), TOKENS), Tap::Summed)?;
        let out_su = dev.read(&out_su)?;
        ok.push(clause(
            &format!("{tag}split_rows_ran"),
            ran,
            "the unit's card slots ran",
        ));
        ok.push(clause(
            &format!("{tag}split_rows_host"),
            host_picks_u > 0,
            &format!("{host_picks_u} host picks over {TOKENS} rows"),
        ));
        ok.push(banded(
            &format!("{tag}split_rows_routed"),
            rel32(&summed_u, &f64s(&unit.routed)),
            SPLIT_BAND,
        ));
        ok.push(banded(
            &format!("{tag}split_rows_out"),
            rel32(&out_su, &f64s(&unit.out)),
            SPLIT_BAND,
        ));
        ok.push(banded(
            &format!("{tag}split_rows_is_step_routed"),
            rel32(&summed_u, &f64s(&split_routed)),
            SPLIT_BAND,
        ));
        ok.push(banded(
            &format!("{tag}split_rows_is_step_host"),
            rel32(&hsum_all, &f64s(&host_steps)),
            PROMPT_BAND,
        ));
        // The whole-card launch on a card that holds a share.
        let mut out = dev.zeroed(HIDDEN)?;
        ok.push(refused(
            &format!("{tag}split_whole_card"),
            moe_c.enqueue_step(
                (&dev.gpu, &w_card, &cx_c),
                &mut bufs_c,
                LAYER,
                &x_t[0],
                &mut out,
            ),
        ));
        drop((bufs_c, moe_c, w_card, host, hs, us));
        std::fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?;
        Ok(ok)
    }
}
