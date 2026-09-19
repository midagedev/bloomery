//! Block-N MLA attention: `attn_norm-N` in, `kqv_out-N` out.
//!
//! Gate: `crates/model/tests/attn.rs` against the oracle. Owned by the attn round.
//!
//! The reference is ik_llama.cpp's MLA graph (weight-absorbed form over the 512-wide
//! latent), reproduced numerically, not just mathematically. Four of its choices are
//! invisible in the math and each one alone would blow the 1e-4 gate:
//!
//!   * **YaRN mscale does not scale the rope.** ik's graph divides the mscale back out
//!     before rope (`attn_factor_scaled = 1/(1+0.1·ln(1/freq_scale))` cancels the
//!     `mscale *= 1+0.1·ln(1/freq_scale)` inside `rope_yarn`; build_deepseek2.cpp:1250-1254
//!     and ggml.c's `rope_yarn`), so rope is a pure rotation and the real mscale lands
//!     in `kq_scale = mscale²/√key_length` on the attention logits.
//!   * **`wk_b` is requantized to Q8_0.** The file's `attn_kv_b` is Q3_K; ik's
//!     `llm_prepare_mla` (llama.cpp:3216-3230) dequantizes the k-up half, transposes it,
//!     and casts to Q8_0 — 32-value blocks along the 128-wide q_nope axis, f16 scales.
//!     `wv_b` stays Q3_K.
//!   * **The KV cache is F16.** `kvr` is rounded f32→f16 before attention reads it
//!     back as both K (576 wide) and V (its latent tail).
//!   * **Small-M activations quantize to ik's `block_q8_2`, not Q8_0.** At M ≤ 7 rows
//!     `iqk_mul_mat_4d` quantizes the F32 activation itself, with a bf16-rounded scale
//!     and `id = 1/d` — see [`quantize_act`]. The two conventions differ by up to 2⁻⁹
//!     relative, an order of magnitude past this gate.
//!   * **The reference CPU flash-attention is ik's `iqk` templates, not ggml's generic
//!     loop.** `iqk_flash_attn_noalibi` claims the node through its general prefill
//!     path and runs `FlashAttn<576, 512, ·, 32>` with F16 K/V helpers
//!     (iqk_flash_attn.cpp:576-605 → iqk_mul_mat.cpp:1379 → iqk_fa_576_512.cpp). The
//!     KV cache is padded to 256 entries (`llama_kv_cache::get_padding` returns 256
//!     with `-fa` on; `kv_self.n = max(pad, GGML_PAD(max_cell, pad))`, llama.cpp:7190)
//!     and the F16 mask carries −inf on every padded cell — so pad rows take weight
//!     exactly 0 and drop out of the arithmetic. q stays F32 end to end; the QK dot is
//!     f32 FMAs in the gemm's two-partial lane order; softmax weights come from ik's
//!     vector `v_expf` polynomial (not libm `expf`); V accumulates in an f32 FMA chain
//!     and the row is normalized by a plain `1/S` multiply (see [`flash_attn_latent`]).
//!
//! Everything here mirrors those choices in the same operation order — there is no
//! place where we are deliberately more exact than the reference.
//!
//! Contract: `x` is `[embd, n_tokens]` and `slots.len() == x.ne1`; the batch's own
//! tokens are the KV entries (prefill semantics). Token `t` attends to every batch
//! entry `u` with `slots[u].seq == slots[t].seq && slots[u].pos <= slots[t].pos`.

use crate::ops::{f32_tensor, matmul_q, rms_norm};
use crate::{ModelError, Slot, Tensor2};
use gguf::quant::half_to_f32;
use gguf::{Gguf, TensorInfo, dequant_row};

/// All intermediates, for the gate. `Tensor2` holds `ne0` contiguous with trailing
/// dims folded into `ne1`, laid out exactly as the oracle dumps flatten them.
pub struct AttnTrace {
    /// `q-N` occurrence 0: `attn_q @ x`, `{3072, n_tokens}`.
    pub q: Tensor2,
    /// `kv_rope_compressed-N`: `attn_kv_a_mqa @ x`, `{576, n_tokens}` —
    /// `[latent(512) ; rope(64)]` per token.
    pub kv_rope_compressed: Tensor2,
    /// `q_rope-N` occurrence 1: rope output, `{64, n_head·n_tokens}` (columns `t·n_head+h`).
    pub q_rope: Tensor2,
    /// `kv_compressed-N` occurrence 1: rms-normed latent, `{512, n_tokens}`.
    pub kv_compressed: Tensor2,
    /// `kvr-N`: `[k_rope(64) ; kv_compressed(512)]` per token, `{576, n_tokens}`.
    /// Concat order verified against the oracle: rope first, latent second.
    pub kvr: Tensor2,
    /// `q_nope2-N`: `wk_b(Q8_0)ᵀ @ q_nope` per head, `{512, n_head·n_tokens}` (columns `h·n_tokens+t`).
    pub q_nope2: Tensor2,
    /// `kqv_compressed-N`: attention over the latent, `{512, n_head·n_tokens}` (columns `t·n_head+h`).
    pub kqv_compressed: Tensor2,
    /// `kqv_out-N`: `attn_output @ kqv`, `{2048, n_tokens}`.
    pub kqv_out: Tensor2,
}

/// One block's attention. `x` is the block's `attn_norm-N` (already normed); the result
/// is `kqv_out-N`, which feeds the block's residual add.
pub fn block_attn(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    slots: &[Slot],
) -> Result<Tensor2, ModelError> {
    Ok(block_attn_trace(gguf, block, x, slots)?.kqv_out)
}

/// The same computation, keeping every intermediate the gate asserts.
pub fn block_attn_trace(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    slots: &[Slot],
) -> Result<AttnTrace, ModelError> {
    let p = MlaParams::read(gguf, block)?;
    if slots.len() != x.ne1 {
        return Err(ModelError::Shape {
            what: "attn slots vs tokens",
            want_ne0: x.ne0,
            want_ne1: x.ne1,
            got_ne0: x.ne0,
            got_ne1: slots.len(),
        });
    }
    let kv_width = p.latent + p.rope_dims;

    // 1. The two projections out of the normed activations.
    let wq = find(gguf, &format!("blk.{block}.attn_q.weight"))?;
    let wa = find(gguf, &format!("blk.{block}.attn_kv_a_mqa.weight"))?;
    let q = matmul_q(gguf, wq, x)?;
    let kv_rope_compressed = matmul_q(gguf, wa, x)?;

    // 2. Latent norm. The gain is F32 in the file.
    let gain_t = find(gguf, &format!("blk.{block}.attn_kv_a_norm.weight"))?;
    let gain = f32_tensor(gguf, gain_t)?;
    let mut latent = Tensor2::zeros(p.latent, x.ne1);
    for t in 0..x.ne1 {
        let src = &kv_rope_compressed.col(t)[..p.latent];
        latent.col_mut(t).copy_from_slice(src);
    }
    let kv_compressed = rms_norm(&latent, &gain, p.eps);

    // 3. Rope: one cos/sin cache per position, shared by every head. k_rope comes from
    //    the kv_a tail, q_rope from each q head's rope slice.
    let caches: Vec<Vec<f32>> = slots.iter().map(|s| p.rope.cache(s.pos)).collect();
    let mut k_rope = Tensor2::zeros(p.rope_dims, x.ne1);
    let mut q_rope = Tensor2::zeros(p.rope_dims, p.n_head * x.ne1);
    for t in 0..x.ne1 {
        let cache = &caches[t];
        let ksrc = &kv_rope_compressed.data[t * kv_width + p.latent..(t + 1) * kv_width];
        rope_pair(ksrc, k_rope.col_mut(t), cache);
        for h in 0..p.n_head {
            let base = t * q.ne0 + h * p.kq_head + p.nope;
            let qsrc = &q.data[base..base + p.rope_dims];
            rope_pair(qsrc, q_rope.col_mut(t * p.n_head + h), cache);
        }
    }

    // 4. kvr = [k_rope ; kv_compressed] (order verified against the oracle dump).
    let mut kvr = Tensor2::zeros(kv_width, x.ne1);
    for t in 0..x.ne1 {
        let dst = kvr.col_mut(t);
        dst[..p.rope_dims].copy_from_slice(k_rope.col(t));
        dst[p.rope_dims..].copy_from_slice(kv_compressed.col(t));
    }

    // 5. The F16 cache the reference attends over: K = f16(kvr), V = its latent tail.
    let cache16: Vec<Vec<u16>> = (0..x.ne1)
        .map(|t| {
            kvr.col(t)
                .iter()
                .map(|&v| f32_to_f16_bits(v))
                .collect::<Vec<u16>>()
        })
        .collect();

    // 6. q_nope2 = wk_b(Q8_0)ᵀ q_nope per head — the weight absorption.
    let wkb = find(gguf, &format!("blk.{block}.attn_kv_b.weight"))?;
    let q_nope2 = q_nope2_absorbed(gguf, wkb, &q, &p)?;

    // 7. Attention over the latent.
    let kqv_compressed = flash_attn_latent(&q_rope, &q_nope2, &cache16, slots, &p);

    // 8. wv_b (still Q3_K) per head, then the output projection.
    let kqv_2d = wv_b_heads(gguf, wkb, &kqv_compressed, &p)?;
    let wo = find(gguf, &format!("blk.{block}.attn_output.weight"))?;
    let kqv_out = matmul_q(gguf, wo, &kqv_2d)?;

    Ok(AttnTrace {
        q,
        kv_rope_compressed,
        q_rope,
        kv_compressed,
        kvr,
        q_nope2,
        kqv_compressed,
        kqv_out,
    })
}

fn find<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a TensorInfo, ModelError> {
    gguf.find(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))
}

// --------------------------------------------------------------------- rope

/// Everything rope needs, with the YaRN constants the graph passes.
///
/// `ext_factor`, `attn_factor` and the `beta_*` pair are not in the file; they are
/// llama.cpp's YaRN call-site constants (llama-graph.cpp: `beta_fast = 32.0f`,
/// `beta_slow = 1.0f`; YaRN runs with `ext_factor = attn_factor = 1.0f`).
pub struct RopeParams {
    pub n_dims: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    /// What the graph passes as rope's mscale: `1/(1+0.1·ln(1/freq_scale))`, which
    /// `rope_yarn` multiplies back to ≈1 — rope stays a pure rotation.
    pub mscale_param: f32,
    pub corr_dims: [f32; 2],
    /// `freq_base^(-2/n_dims)` as one f32 `powf`, as ggml computes it.
    pub theta_scale: f32,
}

impl RopeParams {
    /// `ggml_rope_yarn_corr_dims` (ggml.c:20771), same f32 op order. For this file:
    /// `[10, 23]` in pair-index units.
    fn corr_dims(
        n_dims: usize,
        ctx_orig: u32,
        freq_base: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> [f32; 2] {
        let two_pi = std::f64::consts::PI as f32; // (float)M_PI
        let corr = |n_rot: f32| {
            n_dims as f32 * (ctx_orig as f32 / (n_rot * 2.0 * two_pi)).ln() / (2.0 * freq_base.ln())
        };
        let start = corr(beta_fast).floor();
        let end = corr(beta_slow).ceil();
        [start.max(0.0), end.min(n_dims as f32 - 1.0)]
    }

    /// cos/sin cache for one position: `[cos0, sin0, cos1, sin1, …]` per pair.
    ///
    /// `ggml_rope_cache_init` (ggml.c:20813): theta starts at the position and is
    /// multiplied by `theta_scale` once per pair — a running product, which is why
    /// this is a loop and not `powi`. `sin_sign = +1` in the forward graph; the
    /// multiply by it is skipped as the exact no-op it is.
    pub fn cache(&self, pos: u32) -> Vec<f32> {
        let npairs = self.n_dims / 2;
        let mut out = vec![0.0f32; self.n_dims];
        let mut theta = pos as f32;
        for i in 0..npairs {
            let (c, s) = self.yarn(theta, 2 * i);
            out[2 * i] = c;
            out[2 * i + 1] = s;
            theta *= self.theta_scale;
        }
        out
    }

    /// `rope_yarn` (ggml.c:20794). `i0` is the *dim* loop variable (steps of 2); the
    /// ramp compares `i0/2` — the pair index — against `corr_dims`.
    fn yarn(&self, theta_extrap: f32, i0: usize) -> (f32, f32) {
        let theta_interp = self.freq_scale * theta_extrap;
        let mut theta = theta_interp;
        let mut mscale = self.mscale_param;
        if self.ext_factor != 0.0 {
            let ramp_mix =
                rope_yarn_ramp(self.corr_dims[0], self.corr_dims[1], i0) * self.ext_factor;
            theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0 + 0.1 * (1.0 / self.freq_scale).ln();
        }
        (theta.cos() * mscale, theta.sin() * mscale)
    }
}

/// `rope_yarn_ramp` (ggml.c:20789).
fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Adjacent-pair NORM rotation of one row: `y[2i] = x0·cos − x1·sin`,
/// `y[2i+1] = x0·sin + x1·cos`, f32 ops in ggml's order.
fn rope_pair(src: &[f32], dst: &mut [f32], cache: &[f32]) {
    for i in (0..src.len()).step_by(2) {
        let (x0, x1) = (src[i], src[i + 1]);
        let (c, s) = (cache[i], cache[i + 1]);
        dst[i] = x0 * c - x1 * s;
        dst[i + 1] = x0 * s + x1 * c;
    }
}

// -------------------------------------------------------------- parameters

/// The block's geometry and scalars, all read from the file (never literals), with the
/// derived relations cross-checked so a differently-shaped MLA file fails loudly.
pub struct MlaParams {
    pub n_head: usize,
    /// qk head dim = nope + rope.
    pub kq_head: usize,
    /// `key_length − rope.dimension_count`.
    pub nope: usize,
    pub rope_dims: usize,
    pub v_head: usize,
    /// kv_lora_rank — read from `attn_kv_b`'s row length, the same quantity ggml gets
    /// from hparams.
    pub latent: usize,
    pub eps: f32,
    pub rope: RopeParams,
    /// `mscale²/√kq_head` — the attention-logit scale. mscale itself stays out of rope.
    pub kq_scale: f32,
}

impl MlaParams {
    pub fn read(gguf: &Gguf, block: usize) -> Result<MlaParams, ModelError> {
        let need_u64 = |suffix: &str| -> Result<u64, ModelError> {
            gguf.arch_get_u64(suffix)
                .ok_or_else(|| ModelError::MissingTensor(format!("metadata deepseek2.{suffix}")))
        };
        let need_f32 = |key: &str| -> Result<f32, ModelError> {
            gguf.value(key)
                .and_then(|v| v.as_f32())
                .ok_or_else(|| ModelError::MissingTensor(format!("metadata {key}")))
        };

        let n_head = need_u64("attention.head_count")? as usize;
        let kq_head = need_u64("attention.key_length")? as usize;
        let v_head = need_u64("attention.value_length")? as usize;
        let rope_dims = need_u64("rope.dimension_count")? as usize;
        let eps = need_f32("deepseek2.attention.layer_norm_rms_epsilon")?;

        let scaling = need_f32("deepseek2.rope.scaling.factor")?;
        let ctx_orig = need_u64("rope.scaling.original_context_length")? as u32;
        let log_mul = need_f32("deepseek2.rope.scaling.yarn_log_multiplier")?;
        match gguf
            .value("deepseek2.rope.scaling.type")
            .and_then(|v| v.as_str())
        {
            Some("yarn") => {}
            other => {
                return Err(ModelError::MissingTensor(format!(
                    "deepseek2.rope.scaling.type: this MLA path is yarn-only, got {other:?}"
                )));
            }
        }
        let freq_base = need_f32("deepseek2.rope.freq_base")?;

        let wkb = find(gguf, &format!("blk.{block}.attn_kv_b.weight"))?;
        let wa = find(gguf, &format!("blk.{block}.attn_kv_a_mqa.weight"))?;
        let latent = wkb.dims[0] as usize;

        // Derived geometry must close, or the file is not shaped like this MLA.
        let nope = kq_head - rope_dims;
        let check = |what: &'static str, want: usize, got: usize| -> Result<(), ModelError> {
            if want == got {
                Ok(())
            } else {
                Err(ModelError::Shape {
                    what,
                    want_ne0: want,
                    want_ne1: 0,
                    got_ne0: got,
                    got_ne1: 0,
                })
            }
        };
        check(
            "attn_kv_b per-head span",
            wkb.dims[1] as usize,
            n_head * (nope + v_head),
        )?;
        check(
            "attn_kv_a_mqa output width",
            wa.dims[1] as usize,
            latent + rope_dims,
        )?;

        // YaRN scalars in the graph's own f32 order (build_deepseek2.cpp:1250-1254).
        let freq_scale = 1.0 / scaling;
        let mscale = 1.0 + log_mul * (1.0 / freq_scale).ln();
        let kq_scale = mscale * mscale / (kq_head as f32).sqrt();
        let mscale_param = 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln());

        let rope = RopeParams {
            n_dims: rope_dims,
            freq_base,
            freq_scale,
            ext_factor: 1.0,
            mscale_param,
            corr_dims: RopeParams::corr_dims(rope_dims, ctx_orig, freq_base, 32.0, 1.0),
            theta_scale: freq_base.powf(-2.0 / rope_dims as f32),
        };

        Ok(MlaParams {
            n_head,
            kq_head,
            nope,
            rope_dims,
            v_head,
            latent,
            eps,
            rope,
            kq_scale,
        })
    }
}

// ------------------------------------------------------------------ f16

/// f32 → f16 bits, round-to-nearest-even — the conversion `vcvtps2ph $0x0` performs
/// on the reference build (AVX2 + F16C; `ggml_fp32_to_fp16_row`, verified in the box's
/// libggml disassembly 2026-09-19 — the build is *not* AVX-512).
/// Subnormals and ties follow IEEE; NaN/inf collapse to inf (no NaN reaches the gated
/// graph). One f16 round-trip helper for the whole crate: q rows, the KV cache and the
/// V accumulator all round through here.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return sign | 0x7c00;
    }
    let exp = ((a >> 23) as i32) - 127;
    let frac = a & 0x007f_ffff;
    if exp > 15 {
        // Past f16's finite range; unreachable in the gated graph.
        return sign | 0x7c00;
    }
    if exp >= -14 {
        // Normal f16: keep 11 significand bits; round the dropped 13 with the
        // ties-to-even carry `v + 0x0fff + ((v >> 13) & 1)`.
        let v = (((exp + 15) as u32) << 23) | frac;
        let t = v + 0x0fff + ((v >> 13) & 1);
        let h = t >> 13;
        if h & 0x7c00 == 0x7c00 {
            return sign | 0x7c00;
        }
        return sign | h as u16;
    }
    // Subnormal f16: value in units of 2^-24, round-to-nearest-even on the shift.
    if exp < -25 {
        return sign;
    }
    let shift = (-1 - exp) as u32;
    let v = 0x0080_0000 | frac;
    let half = 1u32 << (shift - 1);
    let rem = v & ((1 << shift) - 1);
    let mut h = v >> shift;
    if rem > half || (rem == half && (h & 1) == 1) {
        h += 1; // a carry into the normal range is correct IEEE behaviour
    }
    sign | h as u16
}

// --------------------------------------------------------------- q_nope2

/// One Q8_0 block over 32 values: f16-stored scale, int8 codes.
#[derive(Clone, Copy)]
struct Q8Block {
    d: u16,
    q: [i8; 32],
}

/// The weight requant the reference's `wk_b` cast runs: `quantize_row_q8_0`, x86
/// branch (ggml-quants.c:938+): `d = amax/127` stored f16, `id = 127/amax` — a
/// different f32 value than `1/d` — and `_mm256_round_ps(_MM_ROUND_NEAREST)` codes.
/// The ref variant (`id = 1/d`, `roundf`) differs on last-ulp and tie cases and is
/// NOT what runs here: switching to it flips codes on 10 of 16 heads and opens ~1e-2
/// on the gate (measured 2026-09-19). Verified byte-identical to the reference's
/// derived `attn_k_b.weight` for block 0 (0 of 69632 bytes differ).
fn quantize_q8_0(x: &[f32]) -> Q8Block {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    let mut q = [0i8; 32];
    for (j, &v) in x.iter().enumerate() {
        q[j] = (v * id).round_ties_even().clamp(-128.0, 127.0) as i8;
    }
    Q8Block {
        d: f32_to_f16_bits(d),
        q,
    }
}

/// One activation block in the format ik's small-M kernels consume (`block_q8_2`):
/// the scale as **bf16**, int8 codes.
#[derive(Clone, Copy)]
struct ActBlock {
    d: f32,
    q: [i8; 32],
}

/// `ggml_compute_fp32_to_bf16` (ggml-impl.h:106): round-to-nearest-even via the carry
/// trick. NaN cannot reach the gated graph.
fn bf16_round(x: f32) -> f32 {
    let mut b = x.to_bits();
    b += 0x7fff + ((b >> 16) & 1);
    f32::from_bits(b & 0xffff_0000)
}

/// The activation quantizer the reference actually runs for this mul_mat —
/// `quantize_row_q8_1_x4_T<block_q8_2, …>` (iqk_quantize.cpp:1074+), **not**
/// `quantize_row_q8_0`. At M ≤ 7 rows, `iqk_mul_mat_4d` takes the raw F32 activation
/// and quantizes it into its own format before the dot, and that format's conventions
/// differ from Q8_0 twice: the scale `amax/127` is rounded to bf16 (8-bit mantissa),
/// and the codes use `id = 1/d` on that rounded scale, RNE (the `_mm256_round_ps`
/// there). Each difference alone moves gated values by up to 2⁻⁹ relative — far
/// outside the 1e-4 gate. Found by bisection against the oracle (2026-09-19): with
/// Q8_0 codes the q_nope2 residual is 4.6e-2; with this it is 1e-6. Codes past ±127
/// saturate, as the reference's `packs_epi8` does.
fn quantize_act(x: &[f32]) -> ActBlock {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = bf16_round(amax / 127.0);
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    let mut q = [0i8; 32];
    for (j, &v) in x.iter().enumerate() {
        q[j] = (v * id).round_ties_even().clamp(-128.0, 127.0) as i8;
    }
    ActBlock { d, q }
}

/// `q_nope2 = wk_b(Q8_0)ᵀ · q_nope` per head: the weight-absorption step.
///
/// `wk_b` is not in the file — the reference derives it at load (`llm_prepare_mla`,
/// llama.cpp:3229): the k-up rows of `attn_kv_b` are dequantized, transposed, and
/// cast to Q8_0 whose 32-value blocks run along the 128-wide q_nope axis. At M ≤ 7
/// rows ik's `iqk_mul_mat_4d` claims the node and quantizes the F32 activation itself,
/// into its small-M `block_q8_2` format (see [`quantize_act`]) — the dot is then a sum
/// over blocks of `f32(f16(dw)) · dq · Σ qw·qq` with an exact i32 inner sum. The block
/// sum accumulates in f64: ggml's f32 SIMD lane order differs in its last ulp; ours is
/// the exact sum.
pub fn q_nope2_absorbed(
    gguf: &Gguf,
    wkb: &TensorInfo,
    q: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    let bytes = gguf.data(wkb)?;
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    let nblocks = p.nope / 32;
    let mut out = Tensor2::zeros(p.latent, p.n_head * q.ne1);
    let mut wk_row = vec![0.0f32; p.latent];

    for h in 0..p.n_head {
        // The head's k-up rows, dequantized once: rows[d][j] = kv_b row h·span+d, elem j.
        let mut rows = vec![0.0f32; p.nope * p.latent];
        for d in 0..p.nope {
            let off = (h * (p.nope + p.v_head) + d) * row_bytes;
            dequant_row(wkb.ty, &bytes[off..off + row_bytes], wk_row.as_mut_slice())?;
            rows[d * p.latent..(d + 1) * p.latent].copy_from_slice(&wk_row);
        }
        // Q8_0 blocks run along the q_nope axis: block (j, b) = 32 d-values of column j.
        let mut wblocks = Vec::with_capacity(p.latent * nblocks);
        let mut vals = [0.0f32; 32];
        for j in 0..p.latent {
            for b in 0..nblocks {
                for (l, v) in vals.iter_mut().enumerate() {
                    *v = rows[(32 * b + l) * p.latent + j];
                }
                wblocks.push(quantize_q8_0(&vals));
            }
        }

        for t in 0..q.ne1 {
            let qbase = t * q.ne0 + h * p.kq_head;
            let qrow = &q.data[qbase..qbase + p.nope];
            let qblocks: Vec<ActBlock> = qrow.chunks_exact(32).map(quantize_act).collect();
            let dst = out.col_mut(h * q.ne1 + t);
            for j in 0..p.latent {
                let mut acc = 0.0f64;
                for (b, qb) in qblocks.iter().enumerate() {
                    let wb = &wblocks[j * nblocks + b];
                    let mut isum = 0i32;
                    for l in 0..32 {
                        isum += wb.q[l] as i32 * qb.q[l] as i32;
                    }
                    acc += (half_to_f32(wb.d) * qb.d) as f64 * f64::from(isum);
                }
                dst[j] = acc as f32;
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------------- attention

/// ik's vector `expf` (iqk_utils.h:160-206), transcribed op for op. The fast path is a
/// minimax polynomial in `b = x − n·ln2` (`n = round-to-integer via the 2²³·1.5 trick`)
/// rescaled by `2ⁿ` rebuilt from `z`'s bit pattern; the `|n| > 126` slow path
/// underflows through the `s1·s2` split. Every `mul_add` is the reference's fused op —
/// its last-ulp rounding is part of the contract, not a stylistic choice. `x = −inf`
/// (a masked logit) lands in the slow path and flushes to exactly `+0.0` (`s1² = 2⁻²⁵⁰`
/// is below the subnormal floor), which is what makes causal-masked keys and the padded
/// cache rows drop out with no residue.
fn v_expf(x: f32) -> f32 {
    // The C hex float literals have no Rust spelling; these bit patterns are their f32
    // values, each verified to round-trip (float.fromhex → pack '<f' → back).
    const R: f32 = f32::from_bits(0x4B40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3FB8_AA3B); // 0x1.715476p+0
    const LN2_LO: f32 = f32::from_bits(0x35BF_BE8E); // 0x1.7f7d1cp-20
    const LN2_HI: f32 = f32::from_bits(0x3F31_7200); // 0x1.62e4p-1
    const P3: f32 = f32::from_bits(0x3C07_2010); // 0x1.0e4020p-7
    const P2: f32 = f32::from_bits(0x3D2B_9F17); // 0x1.573e2ep-5
    const P1: f32 = f32::from_bits(0x3E2A_AF33); // 0x1.555e66p-3
    const P0: f32 = f32::from_bits(0x3EFF_FEDB); // 0x1.fffdb6p-2
    const PM1: f32 = f32::from_bits(0x3F7F_FFF6); // 0x1.ffffecp-1

    let z = x.mul_add(LOG2E, R);
    let n = z - R;
    let b = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000)); // bits(1.0f)
    let c = n.abs() > 126.0;
    let u = b * b;
    let j = P3
        .mul_add(b, P2)
        .mul_add(u, P1.mul_add(b, P0))
        .mul_add(u, PM1 * b);
    if !c {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// The QK dot in the fa4 gemm's own lane order (`mul_mat_Qx_Qy_MxN_fa4`,
/// iqk_gemm_floats.cpp:256 — the `QFT<float,3> × QFT<half,8>` instantiation
/// `iqk_gemm_default_floats` dispatches to for this shape). Per 8-element chunk `i`
/// (ascending), four fused adds land in a low partial (elements `8i+0..3`) and four in
/// a high partial (`8i+4..7`); the dot is the single plain add of the two. `q` stays
/// F32 — the FA node's q is never rounded to f16 — and every K element is the exact
/// f32 of its f16 cache bits (`F16::load(const char*)`).
fn kq_dot_fa4(q: &[f32], k: &[u16]) -> f32 {
    let mut plo = 0.0f32;
    let mut phi = 0.0f32;
    for base in (0..q.len()).step_by(8) {
        plo = half_to_f32(k[base]).mul_add(q[base], plo);
        plo = half_to_f32(k[base + 1]).mul_add(q[base + 1], plo);
        plo = half_to_f32(k[base + 2]).mul_add(q[base + 2], plo);
        plo = half_to_f32(k[base + 3]).mul_add(q[base + 3], plo);
        phi = half_to_f32(k[base + 4]).mul_add(q[base + 4], phi);
        phi = half_to_f32(k[base + 5]).mul_add(q[base + 5], phi);
        phi = half_to_f32(k[base + 6]).mul_add(q[base + 6], phi);
        phi = half_to_f32(k[base + 7]).mul_add(q[base + 7], phi);
    }
    plo + phi
}

/// `F16::reduce_add<32>` (iqk_fa_templates.h:54-206): `((v0+v1)+v2)+v3` elementwise
/// over the four 8-lane registers, then `hsum_float_8` — `t[j] = c[j]+c[j+4]`,
/// `(t0+t2)+(t1+t3)`. Masked lanes hold exactly `0.0` here, so they ride through the
/// tree without changing any rounding.
fn lane_tree_sum(w: &[f32; 32]) -> f32 {
    let c = |l: usize| ((w[l] + w[8 + l]) + w[16 + l]) + w[24 + l];
    let t = [c(0) + c(4), c(1) + c(5), c(2) + c(6), c(3) + c(7)];
    (t[0] + t[2]) + (t[1] + t[3])
}

/// The flash-attention over the latent, replicating ik's FA templates as they run on
/// the box: `iqk_flash_attn_noalibi` → general prefill path → `iqk_flash_attn_impl`
/// (passes because n_kv is 256, a multiple of 32) → `FlashAttn<576, 512, ·, 32>` with
/// F16 K/V helpers (`compute_helper`, since sink/M/S pointers are null).
///
/// Per (token, head) row, in the kernel's own op order:
///
///   * KQ dot per key by [`kq_dot_fa4`] over `[q_rope(64) ; q_nope2(512)]` — the
///     graph's concat order (build_deepseek2.cpp) — against the f16 cache row;
///   * `s = kq_scale·(kq + mask)` — the mask addend is exactly `+0.0` for allowed keys
///     and `−inf` otherwise, so allowed keys get one plain multiply;
///   * online M/S in 32-key blocks (`FlashMS`): block max, rescale-or-zero on M bump,
///     weights = [`v_expf`]`(s − M)`, S += [`lane_tree_sum`];
///   * V accumulation in an f32 FMA chain (`accumulate_qkv`), keys ascending;
///   * final row = `R · (1/S)`, one plain multiply per element.
///
/// Blocks with no allowed key are skipped outright: their weights are all exactly `0.0`
/// (see [`v_expf`]), `S += 0` and `fma(V, 0, R) = R` are exact no-ops, so skipping is
/// bit-identical to executing them. The padded cache rows beyond the batch never enter
/// for the same reason. One divergence, outside the gated regime: the M-bump rescale
/// uses the scalar `expf` (glibc) in the reference and `f32::exp` here — reachable
/// only when a row sees more than 32 allowed keys, which a 6-token prefill cannot.
pub fn flash_attn_latent(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    cache16: &[Vec<u16>],
    slots: &[Slot],
    p: &MlaParams,
) -> Tensor2 {
    let n_tokens = slots.len();
    let d_head = p.rope_dims + p.latent;
    let mut out = Tensor2::zeros(p.latent, n_tokens * p.n_head);

    let mut qrow = vec![0.0f32; d_head];
    let mut r = vec![0.0f32; p.latent];
    let mut w = [0.0f32; 32];
    for t in 0..n_tokens {
        for h in 0..p.n_head {
            // The FA row: [q_rope ; q_nope2], F32, never rounded.
            qrow[..p.rope_dims].copy_from_slice(q_rope.col(t * p.n_head + h));
            qrow[p.rope_dims..].copy_from_slice(q_nope2.col(h * n_tokens + t));

            let mut m = f32::NEG_INFINITY;
            let mut s_sum = 0.0f32;
            r.fill(0.0);
            for blk in (0..n_tokens).step_by(32) {
                let mut s = [f32::NEG_INFINITY; 32];
                let mut smax = f32::NEG_INFINITY;
                for (l, sl) in s.iter_mut().enumerate() {
                    let u = blk + l;
                    let su = match slots.get(u) {
                        Some(su) => su,
                        None => break, // past the batch: cache padding, weight exactly 0
                    };
                    if su.seq != slots[t].seq || su.pos > slots[t].pos {
                        continue; // the -inf half of the causal mask
                    }
                    let kq = kq_dot_fa4(&qrow, &cache16[u]);
                    *sl = p.kq_scale * kq;
                    smax = smax.max(*sl);
                }
                if smax == f32::NEG_INFINITY {
                    continue; // fully masked block: exact no-op (see the doc above)
                }
                // FlashMS::update_M — S scaled here, R at accumulate time.
                let mut vms = 1.0f32;
                let mut need: u8 = 0;
                if smax > m {
                    if m > f32::NEG_INFINITY {
                        vms = (m - smax).exp();
                        need = 1;
                    } else {
                        need = 2;
                    }
                    m = smax;
                }
                match need {
                    1 => s_sum *= vms,
                    2 => s_sum = 0.0,
                    _ => {}
                }
                for l in 0..32 {
                    w[l] = if s[l] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        v_expf(s[l] - m)
                    };
                }
                s_sum += lane_tree_sum(&w);
                match need {
                    1 => r.iter_mut().for_each(|v| *v *= vms),
                    2 => r.fill(0.0),
                    _ => {}
                }
                for l in 0..32 {
                    if w[l] == 0.0 {
                        continue; // masked lane: fma(V, 0, R) == R exactly
                    }
                    let krow = &cache16[blk + l];
                    for d in 0..p.latent {
                        r[d] = half_to_f32(krow[p.rope_dims + d]).mul_add(w[l], r[d]);
                    }
                }
            }

            let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            let dst = out.col_mut(t * p.n_head + h);
            for (d, &v) in r.iter().enumerate() {
                dst[d] = s_inv * v;
            }
        }
    }
    out
}

// ----------------------------------------------------------------- wv_b

/// `wv_b` stays Q3_K in the reference (only `wk_b` is requantized), so this is a plain
/// `matmul_q` per head over a synthetic view of `attn_kv_b`'s v-up rows — the same view
/// `ggml_view_3d` takes, expressed as a file offset. Activation gathering is needed
/// because `kqv_compressed` interleaves heads (`t·n_head+h`) while each head's matmul
/// wants six consecutive 512-wide rows.
pub fn wv_b_heads(
    gguf: &Gguf,
    wkb: &TensorInfo,
    kqv_compressed: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    let n_tokens = kqv_compressed.ne1 / p.n_head;
    let mut kqv_2d = Tensor2::zeros(p.n_head * p.v_head, n_tokens);
    let mut x_h = Tensor2::zeros(p.latent, n_tokens);

    for h in 0..p.n_head {
        for t in 0..n_tokens {
            let src = kqv_compressed.col(t * p.n_head + h);
            x_h.col_mut(t).copy_from_slice(src);
        }
        let view = TensorInfo {
            name: format!("{}.v_up.head{h}", wkb.name),
            dims: vec![p.latent as u64, p.v_head as u64],
            ty: wkb.ty,
            offset: wkb.offset + ((h * (p.nope + p.v_head) + p.nope) * row_bytes) as u64,
            nbytes: wkb.ty.type_size().unwrap()
                * (p.latent / wkb.ty.blck_size().unwrap() as usize) as u64
                * p.v_head as u64,
        };
        let out_h = matmul_q(gguf, &view, &x_h)?;
        for t in 0..n_tokens {
            let dst = kqv_2d.col_mut(t);
            dst[h * p.v_head..(h + 1) * p.v_head].copy_from_slice(out_h.col(t));
        }
    }
    Ok(kqv_2d)
}
