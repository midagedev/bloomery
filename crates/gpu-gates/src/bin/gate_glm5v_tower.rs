//! GPU gate for the glm5v image tower on the card (`bloomery_gpu_vision::glm5v`, GLM-5.3-Flash's
//! mmproj: llama.cpp's `glm4v` graph with a clamped SwiGLU): the whole chain from an image's
//! patches to the rows the text model reads, tap by tap against llama.cpp mainline's mtmd graph
//! (the glm5next vision seat's tap set B, `just dump-ref-glmv fx_ref_glm5next_vis_taps`, with its
//! CPU twin), and every op of block 0 and of the merger against its host rule.
//!
//! The fixture tier is the only tier with sets: the seat's text model is the fixture's respelled
//! copy, and the tower itself runs on the real mmproj. The real tier has no set, so it ends by
//! name with 0 clauses run — never green on nothing.
//!
//! Per image of the set (nine), in chain order (the first failing line names the earliest op):
//! 1. The plan equals the set's: canvas, patch grid and merged tokens.
//! 2. The four taps the set holds (`vit_out`, `after_fc_proj`, `after_ffn_proj`, `embd`) against the
//!    reference's f32, as `rms(ours − ref) / rms(ref)`, within [`K_TAP`] times the larger of three
//!    controls of that tap: the reference's own CPU-twin distance from its CUDA run, the distance of
//!    the reference rounded to bf16 from itself (the floor of any bf16 tower), and the network's
//!    sensitivity — our own encode against our encode with the last bit of one patch value in
//!    sixteen flipped. All four are free-running taps (24 blocks and the merger before them).
//! 3. Teacher-forced: the merger and projector MLP fed the reference's `vit_out`, held to
//!    [`K_TAP`] times the larger of the rounding floor and the step's own sensitivity, at
//!    `after_fc_proj` and `after_ffn_proj`.
//! 4. `embd`, `after_ffn_proj` and the encode's rows are one buffer on the card (bit for bit), and
//!    the set's own `embd` is its `after_ffn_proj`.
//! 5. Launches per image, two plain encodes bit-identical to each other and to the tapped run.
//!
//! On the smallest square grid (`free-448x448`), block 0 and the merger op by op against the host
//! rules of `bloomery_gpu_vision` and the vision crate, each on our own tapped inputs: the norms,
//! the per-head norms (also through the row kernel) and the RoPE bit for bit (the RoPE table from
//! the transcription of ggml's cache loop in `vision_taps.rs`, in merge order); every GEMM inside
//! its accumulation-order bound, the merger's convolution against a host convolution built from the
//! file's weight in the file's K order; the attention inside its f32 bound; the residual epilogues
//! bit for bit; the clamped SwiGLU at pinned counts; the merger's LayerNorm and erf GELU inside the
//! erf's libm bound.
//!
//! Then once: the erf GELU and the clamped SwiGLU kernels on sweeps against their host rules (the
//! clamp over g and u on both sides of the limit, NaN lanes included), the card bytes, and an
//! encode does not depend on the larger image encoded before it in the reused buffers.

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5v_tower", gate::run())
}

#[cfg(feature = "vision")]
#[path = "shared/vision_rules.rs"]
mod vision_rules;

#[cfg(feature = "vision")]
#[path = "shared/vision_taps.rs"]
mod vision_taps;

#[cfg(feature = "vision")]
mod gate {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use super::vision_rules::{Count, exact_check, gemm_check, near_check, par_rows};
    use super::vision_taps::{
        Collect, HostFile, band_check, dist_f32, eager_twice, floor, ggml_table, nudge, nudged,
        read_image, reuse_after_largest, sens as sensitivity, to_bf16,
    };
    use bloomery_gpu_gates::tier::Tier;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{HEAD_DIM, QkvLayout, attn_bound, attn_ref};
    use bloomery_gpu_vision::gelu_tanh::{GeluKind, GeluTanhKernels, gelu_erf_ref, gelu_tanh_ref};
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, epilogue_ref, gelu_erf, within_order};
    use bloomery_gpu_vision::glm5v::Encoder;
    use bloomery_gpu_vision::layer_norm::layer_norm_ref;
    use bloomery_gpu_vision::mlp::MlpKernels;
    use bloomery_gpu_vision::norm::{NormKernels, head_rms_norm_ref, rms_norm_ref};
    use bloomery_gpu_vision::rope2d::{PAIRS, RopeTable, rope_ref};
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
    use gguf::Gguf;
    use refset::arch::glm5next::vis::{MMPROJ_BF16, PROFILE, TAPS};
    use refset::clefvis::{AFTER_FC_PROJ, AFTER_FFN_PROJ, ClefvisSet, EMBD, VIT_OUT};
    use vision::Patches;
    use vision::arch::glm5v::card::{self, ScratchLens};
    use vision::arch::glm5v::rules::{
        conv_k_in_file, reorder_merger_weight, swiglu_clamp as swiglu_clamp_rule,
    };
    use vision::arch::glm5v::{
        Hparams, MERGER_FF, MERGER_NORM_EPS, TEMPORAL_FRAMES, names as n, preprocess,
    };

    const NAME: &str = "gate_glm5v_tower";
    /// The set's image whose block 0 and merger the op rules run on: the smallest square grid.
    const FULL: &str = "free-448x448";
    /// The set's image with the largest grid the pixel budget allows (7904 of 8000 tokens).
    const LARGEST: &str = "cap-3000x2200";
    /// The tokens the buffers are sized for: the file's pixel budget over a merged token's pixels
    /// (6,272,000 / 28²).
    const MAX_TOKENS: usize = 8000;
    /// The card figures of the file's tower: the weights as uploaded (arithmetic from the tensor
    /// table, `card::weights`), and the scratch at [`MAX_TOKENS`] (252,096 B per token).
    const WEIGHT_BYTES: u64 = 1_128_026_112;
    const SCRATCH_BYTES: u64 = 2_016_768_000;
    /// Chain launches of one image at 24 blocks: the patch GEMM, eleven per block, the final norm
    /// and the merger's seven (`Encoder::launches`, a sum of per-stage counts).
    const LAUNCHES: usize = 273;
    /// A tap passes when `rms(ours − ref) / rms(ref)` is at most this multiple of the larger of its
    /// controls.
    /// PIN(2026-10-12): the Qwen3-VL tower's multiple, fixed before the first run of this tower
    /// against its set: the same chain of bf16 ops, 24 blocks instead of 27, so the design's
    /// prediction of a block output at about 1.6× the rounding floor holds, and 4 leaves room for a
    /// free-running network's amplification without admitting an order-one defect (the mutants'
    /// ratios are the proof; see the round's report). First measurement: the 36 free-running
    /// ratios lie in 0.92–3.78, the smallest image's `after_fc_proj` (a grid of 8×10 patches, whose
    /// controls are the smallest) the highest; the nine teacher-forced pairs in 1.45–1.72.
    const K_TAP: f64 = 4.0;
    /// The erf GELU's allowed distance from the host rule's f32 value, in units of `|x|`. The
    /// device runs the host's rule step for step (`z = x·√½`, `e = erf z`, `s = 1 + e`, `0.5·x·s`,
    /// each op rounded to f32), so the two differ by the `erf` alone and by how the roundings fall:
    /// the device's `erff` is within 2 ulp of the true value and the host's within 0.5, so
    /// `|Δe| ≤ 2.5·2⁻²⁴` (the ulp of a magnitude below 1); the two roundings of `s` add at most
    /// `2·2⁻²⁴` where `s ≥ 1` and `2·2⁻²⁵` below it; the two roundings of the product at most
    /// `2·2⁻²⁴` of its magnitude. For x > 0 that is `0.5·|x|·4.5·2⁻²⁴ + 2·2⁻²⁴·|x| = 2.125·2⁻²³·|x|`;
    /// for x < 0, where the sum `1 + e` cancels toward a small result and `s < 1`,
    /// `0.5·|x|·3.5·2⁻²⁴ + 2⁻²⁴·|x| = 1.375·2⁻²³·|x|`. `2⁻²¹ = 4·2⁻²³` covers both. The bound follows
    /// `|x|`, not the result: where `erf` is within an ulp of −1 the cancellation moves a result of
    /// 1e-6 by several bf16 steps of itself, and a one-step band would call that a defect.
    const ERF_BOUND: f64 = 1.0 / 2_097_152.0;
    /// Values of the clamped SwiGLU sweep that may differ from the f32 rule rounded once, each by
    /// one bf16 step at most: the device's `expf` (2 ulp) against the host's `exp` moves `silu` by
    /// at most 3.6e-7 relative, and a bf16 rounding boundary lies that close to a value with a
    /// probability of at most 3.6e-7 / 2⁻⁸ ≈ 9e-5 a lane, so at most about 43 of the sweep's 469,084
    /// lanes [derived].
    /// PIN(2026-10-12): the first measurement of this build, 26; see the sweep line.
    const CLAMP_SWEEP_PIN: usize = 26;
    /// The same for block 0's MLP rows of the full-tap image (4,194,304 values).
    /// PIN(2026-10-12): the first measurement of this build, 0; see the rule line.
    const ACT_BLK0_PIN: usize = 0;
    /// The same for the merger MLP's rows of the full-tap image (2,621,440 values): the sweep's
    /// per-lane bound allows 240, the chain's gate values sit mostly near zero where `expf` is
    /// exact more often.
    /// PIN(2026-10-12): the first measurement of this build, 1; see the rule line.
    const ACT_MERGER_PIN: usize = 1;

    // ------------------------------------------------------------ taps

    /// The taps of the set (the oracle's names are ours), all free-running.
    const ORACLE_TAPS: [&str; 3] = [VIT_OUT, AFTER_FC_PROJ, AFTER_FFN_PROJ];

    /// Everything our encode is asked to tap for an image: the oracle's taps and `embd`, and for the
    /// full-tap image the block-0 and merger ops the host rules read.
    fn wanted(full: bool, n_layer: usize) -> HashSet<String> {
        let mut w: HashSet<String> = ORACLE_TAPS
            .iter()
            .chain(&[EMBD])
            .map(|t| (*t).to_string())
            .collect();
        if full {
            for op in [
                "norm1", "qkv", "qnorm", "knorm", "qrot", "krot", "sdpa", "attn", "resid1",
                "norm2", "gu", "act", "mlp",
            ] {
                w.insert(format!("blk0.{op}"));
            }
            for t in [
                "embed",
                "blk0",
                "merger.conv",
                "merger.fc",
                "merger.gu",
                "merger.act",
            ] {
                w.insert(t.to_string());
            }
            w.insert(format!("blk{}", n_layer - 1));
        }
        w
    }

    // ------------------------------------------------------------ host weights

    /// The host copies of the weights block 0 and the merger read, as the file holds them (the
    /// merger's convolution in the file's K order).
    struct HostW {
        patch_w: Vec<u16>,
        patch_b: Vec<f32>,
        ln1: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        qkv_w: Vec<u16>,
        qkv_b: Vec<f32>,
        o_w: Vec<u16>,
        o_b: Vec<f32>,
        ln2: Vec<f32>,
        /// The gate's rows then the up's, and their biases likewise.
        gu_w: Vec<u16>,
        gu_b: Vec<f32>,
        down_w: Vec<u16>,
        down_b: Vec<f32>,
        post_ln: Vec<f32>,
        conv_w: Vec<u16>,
        conv_b: Vec<f32>,
        fc_w: Vec<u16>,
        post_norm: (Vec<f32>, Vec<f32>),
        mgu_w: Vec<u16>,
        mdown_w: Vec<u16>,
    }

    fn host_w(file: &Gguf, hp: &Hparams) -> Result<HostW, GateError> {
        let hf = HostFile(file);
        let (b16, f32s) = (|name: String| hf.bf16(name), |name: String| hf.f32s(name));
        let fused = |gate: String, up: String| -> Result<Vec<u16>, GateError> {
            let mut v = b16(gate)?;
            v.extend(b16(up)?);
            Ok(v)
        };
        let (first, second) = (
            b16(n::patch_embd_weight())?,
            b16(n::patch_embd_weight_second_frame())?,
        );
        let k_half = 3 * hp.patch * hp.patch;
        let patch_w: Vec<u16> = first
            .chunks_exact(k_half)
            .zip(second.chunks_exact(k_half))
            .flat_map(|(a, b)| a.iter().chain(b).copied())
            .collect();
        let mut gu_b = f32s(n::ffn_gate_bias(0))?;
        gu_b.extend(f32s(n::ffn_up_bias(0))?);
        Ok(HostW {
            patch_w,
            patch_b: f32s(n::patch_embd_bias())?,
            ln1: f32s(n::ln1_weight(0))?,
            q_norm: f32s(n::attn_q_norm_weight(0))?,
            k_norm: f32s(n::attn_k_norm_weight(0))?,
            qkv_w: b16(n::attn_qkv_weight(0))?,
            qkv_b: f32s(n::attn_qkv_bias(0))?,
            o_w: b16(n::attn_out_weight(0))?,
            o_b: f32s(n::attn_out_bias(0))?,
            ln2: f32s(n::ln2_weight(0))?,
            gu_w: fused(n::ffn_gate_weight(0), n::ffn_up_weight(0))?,
            gu_b,
            down_w: b16(n::ffn_down_weight(0))?,
            down_b: f32s(n::ffn_down_bias(0))?,
            post_ln: f32s(n::post_ln_weight())?,
            conv_w: b16(n::patch_merger_weight())?,
            conv_b: f32s(n::patch_merger_bias())?,
            fc_w: b16(n::fc_weight())?,
            post_norm: (f32s(n::post_norm_weight())?, f32s(n::post_norm_bias())?),
            mgu_w: fused(n::mm_gate_weight(), n::mm_up_weight())?,
            mdown_w: b16(n::mm_down_weight())?,
        })
    }

    fn norm_rows(x: &[u16], gain: &[f32], eps: f32) -> Vec<u16> {
        x.chunks_exact(gain.len())
            .flat_map(|r| rms_norm_ref(r, gain, eps))
            .collect()
    }

    /// The row kernel (`NormKernels::enqueue`) over `x`, rows of `gain.len()` values.
    fn norm_rows_on_card(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        x: &[u16],
        gain: &[f32],
        eps: f32,
    ) -> Result<Vec<u16>, GateError> {
        let k = NormKernels::load(ctx)?;
        let dx = DeviceBuffer::from_host(stream, x)?;
        let dg = DeviceBuffer::from_host(stream, gain)?;
        let mut y = DeviceBuffer::<u16>::zeroed(stream, x.len())?;
        k.enqueue(stream, &dx, &dg, eps, x.len() / gain.len(), &mut y)?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    // ------------------------------------------------------------ op rules

    /// llama.cpp's `ggml_swiglu_clamp` on one value pair as the tower rounds it: the f32 rule of
    /// the vision crate rounded once to bf16.
    fn swiglu_want(g: u16, u: u16, limit: f32) -> u16 {
        f32_bf16(swiglu_clamp_rule(bf16_f32(g), bf16_f32(u), limit))
    }

    /// The clamped SwiGLU of fused `[gate | up]` rows of `gu` (`2·ff` values each) against `got`.
    fn act_rule(name: &str, gu: &[u16], ff: usize, got: &[u16], limit: f32, pin: usize) -> bool {
        let want: Vec<u16> = gu
            .chunks_exact(2 * ff)
            .flat_map(|r| (0..ff).map(move |j| swiglu_want(r[j], r[ff + j], limit)))
            .collect();
        near_check(name, got, &want, pin)
    }

    /// Block 0 and the merger of the full-tap image, op by op, on our own tapped inputs.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "one rule per op of the chain, in chain order, on the taps and weights it reads"
    )]
    fn host_rules(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        hp: &Hparams,
        w: &HostW,
        t: &HashMap<String, Vec<u16>>,
        patches: &[u16],
        grid: (usize, usize),
        out: &[u16],
    ) -> Result<bool, GateError> {
        let (dim, ff, heads, n) = (hp.dim, hp.ff, hp.n_head, grid.0 * grid.1);
        let (merge, out_dim) = (hp.merge, hp.out_dim);
        let n_tok = n / (merge * merge);
        let merged = merge * merge * dim;
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (embed, norm1, qkv) = (get("embed")?, get("blk0.norm1")?, get("blk0.qkv")?);
        let (qnorm, knorm, qrot, krot) = (
            get("blk0.qnorm")?,
            get("blk0.knorm")?,
            get("blk0.qrot")?,
            get("blk0.krot")?,
        );
        let (sdpa, attn_side, resid1, norm2) = (
            get("blk0.sdpa")?,
            get("blk0.attn")?,
            get("blk0.resid1")?,
            get("blk0.norm2")?,
        );
        let (gu, act, mlp_side, blk0) = (
            get("blk0.gu")?,
            get("blk0.act")?,
            get("blk0.mlp")?,
            get("blk0")?,
        );
        let (blk_last, vit, conv, fc) = (
            get(&format!("blk{}", hp.n_layer - 1))?,
            get(VIT_OUT)?,
            get("merger.conv")?,
            get("merger.fc")?,
        );
        let (after_fc, mgu, mact, rows) = (
            get(AFTER_FC_PROJ)?,
            get("merger.gu")?,
            get("merger.act")?,
            get(AFTER_FFN_PROJ)?,
        );
        let k_patch = TEMPORAL_FRAMES * 3 * hp.patch * hp.patch;
        let mut ok = true;

        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            Some(&w.patch_b),
            n,
            dim,
            k_patch,
            embed,
        );
        ok &= exact_check("blk0.norm1", norm1, &norm_rows(embed, &w.ln1, hp.eps));
        ok &= gemm_check(
            "blk0.wqkv",
            norm1,
            &w.qkv_w,
            Some(&w.qkv_b),
            n,
            3 * dim,
            dim,
            qkv,
        );

        // The per-head norms of q and k, on the GEMM's own rows: the host rule, then the row
        // kernel over the gathered [n·heads, 64] rows with the one gain of 64.
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: heads,
        };
        let cols = |x: &[u16], c0: usize| -> Vec<u16> {
            x.chunks_exact(3 * dim)
                .flat_map(|r| r[c0..c0 + dim].iter().copied())
                .collect()
        };
        let mut normed = qkv.clone();
        head_rms_norm_ref(&mut normed, 3 * dim, lay.q0, heads, &w.q_norm, hp.eps);
        head_rms_norm_ref(&mut normed, 3 * dim, lay.k0, heads, &w.k_norm, hp.eps);
        ok &= exact_check("blk0.qnorm", qnorm, &cols(&normed, lay.q0));
        ok &= exact_check("blk0.knorm", knorm, &cols(&normed, lay.k0));
        for (name, c0, tap, gain) in [
            ("blk0.qnorm rows", lay.q0, qnorm, &w.q_norm),
            ("blk0.knorm rows", lay.k0, knorm, &w.k_norm),
        ] {
            let got = norm_rows_on_card(ctx, stream, &cols(qkv, c0), gain, hp.eps)?;
            ok &= exact_check(name, tap, &got);
        }

        // RoPE on our own normed q and k, the table from the independent transcription.
        let table = RopeTable {
            n_h: grid.0,
            n_w: grid.1,
            cs: ggml_table(grid.0, grid.1, PAIRS),
        };
        let mut rot = qkv.clone();
        for (i, row) in rot.chunks_exact_mut(3 * dim).enumerate() {
            row[lay.q0..lay.q0 + dim].copy_from_slice(&qnorm[i * dim..(i + 1) * dim]);
            row[lay.k0..lay.k0 + dim].copy_from_slice(&knorm[i * dim..(i + 1) * dim]);
        }
        rope_ref(&mut rot, 3 * dim, lay.q0, heads, &table);
        rope_ref(&mut rot, 3 * dim, lay.k0, heads, &table);
        ok &= exact_check("blk0.rope q", qrot, &cols(&rot, lay.q0));
        ok &= exact_check("blk0.rope k", krot, &cols(&rot, lay.k0));

        // The attention reads our own rotated q and k and the untouched v.
        let mut ours_rot = qkv.clone();
        for (i, row) in ours_rot.chunks_exact_mut(3 * dim).enumerate() {
            row[lay.q0..lay.q0 + dim].copy_from_slice(&qrot[i * dim..(i + 1) * dim]);
            row[lay.k0..lay.k0 + dim].copy_from_slice(&krot[i * dim..(i + 1) * dim]);
        }
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let c = par_rows(n, |i| {
            let mut c = Count::default();
            for h in 0..heads {
                let (o, mag, logit) = attn_ref(&ours_rot, lay, n, scale, h, i);
                for d in 0..HEAD_DIM {
                    let y = sdpa[i * dim + h * HEAD_DIM + d];
                    let b = attn_bound(o[d], mag[d], logit, n);
                    c.outside += usize::from(!within_order(y, o[d], b));
                    c.differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
            c
        });
        let pass = c.outside == 0;
        println!(
            "rule attn blk0.sdpa     n {n}: outside the f32 bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
            c.outside,
            n * dim,
            c.differ,
            c.differ as f64 / (n * dim) as f64,
            verdict(pass)
        );
        ok &= pass;

        ok &= gemm_check(
            "blk0.wo",
            sdpa,
            &w.o_w,
            Some(&w.o_b),
            n,
            dim,
            dim,
            attn_side,
        );
        let want: Vec<u16> = attn_side
            .iter()
            .zip(embed)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0.resid1 epi", resid1, &want);
        ok &= exact_check("blk0.norm2", norm2, &norm_rows(resid1, &w.ln2, hp.eps));
        ok &= gemm_check("blk0.gu", norm2, &w.gu_w, Some(&w.gu_b), n, 2 * ff, dim, gu);
        ok &= act_rule("blk0.swiglu", gu, ff, act, hp.swiglu_clamp, ACT_BLK0_PIN);
        ok &= gemm_check(
            "blk0.down",
            act,
            &w.down_w,
            Some(&w.down_b),
            n,
            dim,
            ff,
            mlp_side,
        );
        let want: Vec<u16> = mlp_side
            .iter()
            .zip(resid1)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0 resid2 epi", blk0, &want);
        ok &= exact_check("post_ln", vit, &norm_rows(blk_last, &w.post_ln, hp.eps));

        // The merger's convolution twice: the card's GEMM against the exact product with the
        // weight as `reorder_merger_weight` lays it out, and against a host convolution that
        // never reorders the weight — the merged row is permuted into the file's K order
        // (`conv_k_in_file`) and meets the file's weight as it is.
        let card_w = reorder_merger_weight(&w.conv_w, dim, merge, out_dim);
        ok &= gemm_check(
            "merger.conv c",
            vit,
            &card_w,
            Some(&w.conv_b),
            n_tok,
            out_dim,
            merged,
            conv,
        );
        let mut a_file = vec![0u16; n_tok * merged];
        for tok in 0..n_tok {
            for ky in 0..merge {
                for kx in 0..merge {
                    let patch = tok * merge * merge + ky * merge + kx;
                    for ch in 0..dim {
                        a_file[tok * merged + conv_k_in_file(merge, kx, ky, ch)] =
                            vit[patch * dim + ch];
                    }
                }
            }
        }
        ok &= gemm_check(
            "merger.conv f",
            &a_file,
            &w.conv_w,
            Some(&w.conv_b),
            n_tok,
            out_dim,
            merged,
            conv,
        );
        ok &= gemm_check(
            "merger.fc",
            conv,
            &w.fc_w,
            None,
            n_tok,
            out_dim,
            out_dim,
            fc,
        );
        // The LayerNorm's output has no tap: its rule is composed with the erf GELU's.
        let ln: Vec<u16> = fc
            .chunks_exact(out_dim)
            .flat_map(|r| layer_norm_ref(r, &w.post_norm.0, &w.post_norm.1, MERGER_NORM_EPS))
            .collect();
        ok &= erf_check("merger ln+gelu_erf", &ln, after_fc);
        ok &= gemm_check(
            "merger.gu",
            after_fc,
            &w.mgu_w,
            None,
            n_tok,
            2 * MERGER_FF,
            out_dim,
            mgu,
        );
        ok &= act_rule(
            "merger.swiglu",
            mgu,
            MERGER_FF,
            mact,
            hp.swiglu_clamp,
            ACT_MERGER_PIN,
        );
        ok &= gemm_check(
            "merger.down",
            mact,
            &w.mdown_w,
            None,
            n_tok,
            out_dim,
            MERGER_FF,
            rows,
        );
        ok &= exact_check("rows = after_ffn", out, rows);
        Ok(ok)
    }

    // ------------------------------------------------------------ once-per-run rules

    /// How often the rule a mutant would put in the kernel's place must differ from the kernel's
    /// rule for a sweep to tell the two apart: ten times its pinned count, at least ten.
    fn premise_floor(pin: usize) -> usize {
        10 * pin.max(1)
    }

    /// The erf GELU's results `got` for the bf16 inputs `xs` against the host rule inside
    /// [`ERF_BOUND`]: how many are not a rounding of a value within `|x|·ERF_BOUND` of the rule's
    /// f32 value, how many differ from the rule's own rounding, and the first outside input and
    /// result.
    fn erf_outside(xs: &[u16], got: &[u16]) -> (usize, usize, Option<(u16, u16)>) {
        let (mut outside, mut differ, mut first) = (0usize, 0usize, None);
        for (&x, &y) in xs.iter().zip(got) {
            let xf = bf16_f32(x);
            let centre = f64::from(gelu_erf(xf));
            if !within_order(y, centre, f64::from(xf).abs() * ERF_BOUND) {
                outside += 1;
                first.get_or_insert((x, y));
            }
            differ += usize::from(y != gelu_erf_ref(x));
        }
        (outside, differ, first)
    }

    /// [`erf_outside`] as a clause: no result outside the bound.
    fn erf_check(name: &str, xs: &[u16], got: &[u16]) -> bool {
        let (outside, differ, first) = erf_outside(xs, got);
        let pass = xs.len() == got.len() && outside == 0;
        println!(
            "rule {name:<17} {outside} of {} outside the erf bound |x|·2⁻²¹, {differ} differ from the host rounding  {}",
            xs.len(),
            verdict(pass)
        );
        if let Some((x, y)) = first {
            println!(
                "rule {name:<17} first outside: x {:#06x} ({:e}) gives {:#06x} ({:e}), the host rule {:#06x}",
                x,
                bf16_f32(x),
                y,
                bf16_f32(y),
                gelu_erf_ref(x)
            );
        }
        pass
    }

    /// The erf GELU on every normal bf16 value in [−8, 8] against the host rule inside the erf
    /// bound, and the premise that the bound tells it from the tanh GELU: the tanh rule's results
    /// on the same values fall outside it somewhere.
    fn erf_sweep(ctx: &Arc<CudaContext>, stream: &CudaStream) -> Result<bool, GateError> {
        let xs: Vec<u16> = (0x0080..=0x4100u16).chain(0x8080..=0xC100).collect();
        let k = GeluTanhKernels::load(ctx)?;
        let mut d = DeviceBuffer::from_host(stream, &xs)?;
        k.enqueue_kind(stream, GeluKind::Erf, xs.len(), &mut d)?;
        stream.synchronize()?;
        let got = d.to_host_vec(stream)?;
        let mut ok = erf_check("gelu_erf sweep", &xs, &got);
        let tanh: Vec<u16> = xs.iter().map(|&x| gelu_tanh_ref(x)).collect();
        let tells = erf_outside(&xs, &tanh).0;
        println!(
            "rule gelu_erf premise   the tanh rule falls outside the erf bound at {tells} of {} values: {}",
            xs.len(),
            verdict(tells > 0)
        );
        ok &= tells > 0;
        Ok(ok)
    }

    /// The clamped SwiGLU kernel over every normal bf16 gate value in [−30, 30] against up values
    /// on both sides of the limit (and the limit itself, zeros, NaN), and NaN gates: the f32 rule
    /// rounded once at a pinned count, a NaN lane a NaN on the card and nothing else one, and the
    /// premise that the sweep tells the clamp from the plain SwiGLU.
    fn clamp_sweep(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        limit: f32,
    ) -> Result<bool, GateError> {
        const NAN: u16 = 0x7FC0;
        let gates: Vec<u16> = (0x0080..=0x41F0u16)
            .chain(0x8080..=0xC1F0)
            .chain([NAN])
            .collect();
        // 30, 12, 10, 9.5, 1, 0.25 and zero, each of either sign.
        let ups: Vec<u16> = [0x41F0u16, 0x4140, 0x4120, 0x4118, 0x3F80, 0x3E80, 0x0000]
            .iter()
            .flat_map(|&m| [m, m | 0x8000])
            .chain([NAN])
            .collect();
        let (ff, rows) = (gates.len(), ups.len());
        let mut u = Vec::with_capacity(rows * 2 * ff);
        for &up in &ups {
            u.extend(&gates);
            u.extend(std::iter::repeat_n(up, ff));
        }
        let du = DeviceBuffer::from_host(stream, &u)?;
        let mut dy = DeviceBuffer::<u16>::zeroed(stream, rows * ff)?;
        MlpKernels::load(ctx)?.enqueue(stream, &du, ff, rows, Some(limit), &mut dy)?;
        stream.synchronize()?;
        let got = dy.to_host_vec(stream)?;
        let (mut g_ok, mut w_ok, mut nan_lanes, mut nan_ok, mut bites) =
            (Vec::new(), Vec::new(), 0usize, true, 0usize);
        for (r, &up) in ups.iter().enumerate() {
            for (j, &g) in gates.iter().enumerate() {
                let want = swiglu_want(g, up, limit);
                let y = got[r * ff + j];
                if bf16_f32(want).is_nan() {
                    nan_lanes += 1;
                    nan_ok &= bf16_f32(y).is_nan();
                } else {
                    nan_ok &= !bf16_f32(y).is_nan();
                    g_ok.push(y);
                    w_ok.push(want);
                    bites += usize::from(swiglu_want(g, up, f32::INFINITY) != want);
                }
            }
        }
        let mut ok = near_check("swiglu_clamp sweep", &g_ok, &w_ok, CLAMP_SWEEP_PIN);
        println!(
            "rule swiglu_clamp NaN   {nan_lanes} lanes with a NaN input are NaN on the card, no other lane is: {}",
            verdict(nan_ok)
        );
        let tells = bites >= premise_floor(CLAMP_SWEEP_PIN);
        println!(
            "rule swiglu_clamp prem. the unclamped rule (one rounding) differs from the clamped one at {bites} of {} lanes (at least {}): {}",
            g_ok.len(),
            premise_floor(CLAMP_SWEEP_PIN),
            verdict(tells)
        );
        ok &= nan_ok && tells;
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    struct Img {
        patches: Patches,
        grid: (usize, usize),
    }

    fn load_image(hp: &Hparams, name: &str) -> Result<Img, GateError> {
        let patches = preprocess(&read_image(name)?, &hp.size_rule_at(hp.default_limits()?)?)?;
        let grid = (patches.n_vit_h, patches.n_vit_w);
        Ok(Img { patches, grid })
    }

    #[allow(clippy::too_many_lines, reason = "one image's clauses in chain order")]
    pub fn run() -> Result<(), GateError> {
        let tier = Tier::from_env()?;
        if tier != Tier::Fixture {
            return Err(format!(
                "0 clauses on the {} tier: the glm5v tower's oracle sets are the fixture tier's ({}, the respelled fixture's seat); no set of another tier exists, so nothing was compared. Run it under BLOOMERY_TIER=fixture",
                tier.name(),
                TAPS.sets[0]
            )
            .into());
        }
        let set = ClefvisSet::open(&TAPS.path(TAPS.sets[0]), &TAPS)?;
        let twin = ClefvisSet::open(&TAPS.path(TAPS.sets[1]), &TAPS)?;
        let sets_ok = !set.cpu && twin.cpu;
        println!(
            "sets: {} on {} and its CPU twin {} on {}: {}",
            TAPS.sets[0],
            set.card,
            TAPS.sets[1],
            twin.card,
            verdict(sets_ok)
        );
        let file = Gguf::open(MMPROJ_BF16)?;
        let hp = Hparams::read(&file)?;
        let max_tokens = usize::try_from(hp.max_pixels)? / (hp.patch * hp.merge).pow(2);
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(0)?;
        let mut enc = Encoder::load(&ctx, &stream, &file, max_tokens)?;
        let want_launches = Encoder::launches(hp.n_layer);
        let mut ok = sets_ok;
        println!(
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; activations {} B at {} tokens; {} launches per image",
            hp.n_layer,
            hp.dim,
            hp.n_head,
            hp.ff,
            hp.out_dim,
            enc.weight_bytes(),
            enc.scratch_bytes(),
            max_tokens,
            want_launches
        );
        let bytes_ok = enc.weight_bytes() == WEIGHT_BYTES
            && enc.weight_bytes() == card::weights(&hp)
            && enc.scratch_bytes() == SCRATCH_BYTES
            && enc.scratch_bytes() == ScratchLens::of(&hp, MAX_TOKENS).bytes()
            && max_tokens == MAX_TOKENS
            && want_launches == LAUNCHES
            && hp.out_dim == PROFILE.n_embd;
        println!(
            "bytes: weights {} (want {WEIGHT_BYTES}), scratch {} (want {SCRATCH_BYTES}), launches {want_launches} (want {LAUNCHES}): {}",
            enc.weight_bytes(),
            enc.scratch_bytes(),
            verdict(bytes_ok)
        );
        ok &= bytes_ok;
        let hw = host_w(&file, &hp)?;
        ok &= erf_sweep(&ctx, &stream)?;
        ok &= clamp_sweep(&ctx, &stream, hp.swiglu_clamp)?;

        let mut names: Vec<(String, bool)> = set
            .images
            .iter()
            .map(|i| (i.name.clone(), i.name == FULL))
            .collect();
        names.sort_by_key(|(n, full)| (!*full, n.clone()));
        let mut full_state: Option<(Img, Vec<u16>)> = None;

        for (name, full) in &names {
            let img = load_image(&hp, name)?;
            let info = set.image(name)?;
            let (n_h, n_w) = img.grid;
            let shape = (info.ny * hp.merge, info.nx * hp.merge) == img.grid
                && (info.best_w, info.best_h) == (img.patches.plan.best_w, img.patches.plan.best_h)
                && info.n_tokens == n_h * n_w / (hp.merge * hp.merge);
            println!(
                "image {name}: {n_h}x{n_w} patches, {} tokens, canvas {}x{}; the plan equals the set's: {}",
                n_h * n_w / (hp.merge * hp.merge),
                img.patches.plan.best_w,
                img.patches.plan.best_h,
                verdict(shape)
            );
            ok &= shape;
            // The tap pass of a set image equals its clean pass; every image holds the four taps.
            let mut sink = Collect::new(wanted(*full, hp.n_layer));
            let (out, tapped_launches) = {
                let r = enc.encode_tapped(&img.patches, &mut sink)?;
                stream.synchronize()?;
                (r.rows.buf().to_host_vec(&stream)?, r.launches)
            };

            // The sensitivity run: the same encode with one patch value in sixteen nudged.
            let mut noisy = img.patches.clone();
            nudge(&mut noisy.bf16);
            let mut nsink = sink.again();
            let nout = {
                let r = enc.encode_tapped(&noisy, &mut nsink)?;
                stream.synchronize()?;
                r.rows.buf().to_host_vec(&stream)?
            };

            for node in ORACLE_TAPS.iter().chain(&[EMBD]) {
                let ours = if *node == EMBD {
                    &out
                } else {
                    sink.got
                        .get(*node)
                        .ok_or_else(|| format!("tap {node} was not taken"))?
                };
                let nours = if *node == EMBD {
                    &nout
                } else {
                    nsink
                        .got
                        .get(*node)
                        .ok_or_else(|| format!("tap {node} was not taken (nudged)"))?
                };
                let (_, refv) = set.tap(name, node)?;
                let (_, twin_v) = twin.tap(name, node)?;
                ok &= band_check(
                    K_TAP,
                    name,
                    node,
                    ours,
                    &refv,
                    dist_f32(&twin_v, &refv).rms_rel,
                    floor(&refv),
                    Some(sensitivity(nours, ours)),
                );
            }

            // The output rows, `after_ffn_proj` and `embd` are one buffer; the set's own two agree.
            let one =
                sink.got.get(AFTER_FFN_PROJ) == Some(&out) && sink.got.get(EMBD) == Some(&out);
            let (_, set_embd) = set.tap(name, EMBD)?;
            let (_, set_ffn) = set.tap(name, AFTER_FFN_PROJ)?;
            let set_one = set_embd.len() == set_ffn.len()
                && set_embd
                    .iter()
                    .zip(&set_ffn)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "image {name}: embd = after_ffn_proj = the encode's rows on the card: {}; in the set: {}",
                verdict(one),
                verdict(set_one)
            );
            ok &= one && set_one;

            // Teacher-forced: the merger and the projector MLP on the reference's own `vit_out`. A
            // step's control is the larger of the rounding floor of its output and the step's own
            // sensitivity: the same step on its input with one value in sixteen nudged by an ulp.
            let x = to_bf16(&set.tap(name, VIT_OUT)?.1);
            let (fc_rows, rows) = enc.forced_tail(img.grid, &x)?;
            let (nfc_rows, nrows) = enc.forced_tail(img.grid, &nudged(&x))?;
            for (tap, got, ngot) in [
                (AFTER_FC_PROJ, &fc_rows, &nfc_rows),
                (AFTER_FFN_PROJ, &rows, &nrows),
            ] {
                let want = set.tap(name, tap)?.1;
                ok &= band_check(
                    K_TAP,
                    name,
                    &format!("forced.{tap}"),
                    got,
                    &want,
                    0.0,
                    floor(&want),
                    Some(sensitivity(ngot, got)),
                );
            }

            if *full {
                ok &= host_rules(
                    &ctx,
                    &stream,
                    &hp,
                    &hw,
                    &sink.got,
                    &img.patches.bf16,
                    img.grid,
                    &out,
                )?;
            }

            let started = Instant::now();
            let va = {
                let a = enc.encode(&img.patches)?;
                (a.rows.buf().to_host_vec(&stream)?, a.launches)
            };
            let plain = started.elapsed();
            let vb = enc.encode(&img.patches)?.rows.buf().to_host_vec(&stream)?;
            ok &= eager_twice(name, (&out, tapped_launches), &va, &vb, want_launches);
            println!(
                "image {name}: a plain encode took {:.0} ms ({n_h}x{n_w} patches)",
                plain.as_secs_f64() * 1e3
            );
            if *full {
                full_state = Some((img, out));
            }
        }

        // An encode does not depend on the larger image encoded before it in the reused buffers.
        if let Some((small, first)) = full_state {
            let largest = load_image(&hp, LARGEST)?;
            let big = enc
                .encode(&largest.patches)?
                .rows
                .buf()
                .to_host_vec(&stream)?;
            let again = enc
                .encode(&small.patches)?
                .rows
                .buf()
                .to_host_vec(&stream)?;
            let tokens = largest.grid.0 * largest.grid.1 / (hp.merge * hp.merge);
            ok &= reuse_after_largest(
                (FULL, LARGEST),
                (tokens, max_tokens, hp.out_dim),
                (&first, &big, &again),
            );
        } else {
            println!("reuse: {FULL} is not an image of the set: FAIL");
            ok = false;
        }

        if ok {
            println!(
                "{NAME}: PASS (fixture tier, {} images of {})",
                names.len(),
                TAPS.sets[0]
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
