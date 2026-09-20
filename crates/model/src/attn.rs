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

use crate::derived::Derived;
use crate::kv::KvCache;
use crate::ops::{f32_tensor, matmul_q, matmul_q_batch, rms_norm};
use crate::profile;
use crate::{ModelError, Slot, Tensor2};
use gguf::quant::half_to_f32;
use gguf::{Gguf, TensorInfo};
use std::sync::Mutex;
use std::time::Instant;

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
    derived: &Derived,
) -> Result<Tensor2, ModelError> {
    Ok(block_attn_trace(gguf, block, x, slots, derived)?.kqv_out)
}

/// The same computation, keeping every intermediate the gate asserts.
///
/// Prefill semantics: the batch is the whole history. Implemented by running the cached
/// path against a cache that holds exactly this batch and nothing else — the two are one
/// code path, so "caching changes nothing" is a property of the structure and not only
/// of the measurement in `tests/kv.rs`.
pub fn block_attn_trace(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    slots: &[Slot],
    derived: &Derived,
) -> Result<AttnTrace, ModelError> {
    let p = MlaParams::read(gguf, block)?;
    let mut scratch = KvCache::new(1, p.latent + p.rope_dims);
    let range = scratch.begin(slots);
    block_attn_cached(gguf, block, x, slots, &mut scratch, 0, &range, derived)
}

/// One block's attention against a KV cache: `q_slots` are the queries this call carries,
/// the cache holds every key. The new rows are appended to `cache_block` first, so a
/// decode step attends against its own token as well as the prefix.
///
/// `cache_block` is the cache's block index, which is the model's `block` in a real
/// forward pass and 0 in the scratch cache `block_attn_trace` builds. They are separate
/// parameters because they are separate things, and conflating them would make the
/// scratch path allocate 27 empty blocks to use one.
#[allow(clippy::too_many_arguments)]
pub fn block_attn_cached(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    q_slots: &[Slot],
    cache: &mut KvCache,
    cache_block: usize,
    range: &std::ops::Range<usize>,
    derived: &Derived,
) -> Result<AttnTrace, ModelError> {
    // Profiler hook (crate::profile), coverage round: the attention prep that is
    // not a matmul, as piece timers around the un-hooked regions only — the
    // matmuls keep their own rows and coverage never counts a nanosecond twice:
    //   * `attn_params` — `MlaParams::read` and the five tensor finds, every one
    //     a linear scan of the tensor table (four pieces, they interleave with
    //     the hooked calls).
    //   * `attn_latent` — the latent copy out of `kv_rope_compressed`.
    //   * `attn_rope`   — the cos/sin caches and the k/q rope applications.
    //   * `attn_kvr`    — the kvr concat, the f32→f16 rounding and the cache push.
    let lvl = profile::level();
    let mut params_ns = 0u64;
    let t_p1 = if lvl > 0 { Some(Instant::now()) } else { None };
    let p = MlaParams::read(gguf, block)?;
    let slots = q_slots;
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
    if let Some(t_p1) = t_p1 {
        params_ns += t_p1.elapsed().as_nanos() as u64;
    }
    let q = matmul_q(gguf, wq, x)?;
    let kv_rope_compressed = matmul_q(gguf, wa, x)?;

    // 2. Latent norm. The gain is F32 in the file.
    let t_p2 = if lvl > 0 { Some(Instant::now()) } else { None };
    let gain_t = find(gguf, &format!("blk.{block}.attn_kv_a_norm.weight"))?;
    if let Some(t_p2) = t_p2 {
        params_ns += t_p2.elapsed().as_nanos() as u64;
    }
    let gain = f32_tensor(gguf, gain_t)?;
    let t_lat = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut latent = Tensor2::zeros(p.latent, x.ne1);
    for t in 0..x.ne1 {
        let src = &kv_rope_compressed.col(t)[..p.latent];
        latent.col_mut(t).copy_from_slice(src);
    }
    if let Some(t_lat) = t_lat {
        profile::record_time("attn_latent", t_lat.elapsed().as_nanos() as u64);
    }
    let kv_compressed = rms_norm(&latent, &gain, p.eps);

    // 3. Rope: one cos/sin cache per position, shared by every head. k_rope comes from
    //    the kv_a tail, q_rope from each q head's rope slice.
    let t_rope = if lvl > 0 { Some(Instant::now()) } else { None };
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
    if let Some(t_rope) = t_rope {
        profile::record_time("attn_rope", t_rope.elapsed().as_nanos() as u64);
    }

    // 4. kvr = [k_rope ; kv_compressed] (order verified against the oracle dump).
    let t_kvr = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut kvr = Tensor2::zeros(kv_width, x.ne1);
    for t in 0..x.ne1 {
        let dst = kvr.col_mut(t);
        dst[..p.rope_dims].copy_from_slice(k_rope.col(t));
        dst[p.rope_dims..].copy_from_slice(kv_compressed.col(t));
    }

    // 5. The F16 rows the reference attends over: K = f16(kvr), V = its latent tail.
    //    They go into the cache before attention, not after, because this batch's own
    //    tokens are keys for this batch's own queries (a prefill attends to itself).
    let new_rows: Vec<Vec<u16>> = (0..x.ne1)
        .map(|t| {
            kvr.col(t)
                .iter()
                .map(|&v| f32_to_f16_bits(v))
                .collect::<Vec<u16>>()
        })
        .collect();
    cache.push(cache_block, range, new_rows);
    if let Some(t_kvr) = t_kvr {
        profile::record_time("attn_kvr", t_kvr.elapsed().as_nanos() as u64);
    }

    // 6. q_nope2 = wk_b(Q8_0)ᵀ q_nope per head — the weight absorption. The wk_b
    //    requant is weight work and lives in `Derived`; only the per-token half
    //    runs here. `block` (the model's) selects the derived weights — never
    //    `cache_block`, which is 0 on the scratch path.
    let t_p3 = if lvl > 0 { Some(Instant::now()) } else { None };
    let wkb = find(gguf, &format!("blk.{block}.attn_kv_b.weight"))?;
    if let Some(t_p3) = t_p3 {
        params_ns += t_p3.elapsed().as_nanos() as u64;
    }
    let q_nope2 = q_nope2_absorbed(derived.wk_b_all_heads(block)?, &q, &p)?;

    // 7. Attention over the latent.
    let kqv_compressed = flash_attn_latent(
        &q_rope,
        &q_nope2,
        cache.keys(cache_block),
        cache.slots(),
        slots,
        &p,
    );

    // 8. wv_b (still Q3_K) per head, then the output projection.
    let kqv_2d = wv_b_heads(gguf, wkb, &kqv_compressed, &p)?;
    let t_p4 = if lvl > 0 { Some(Instant::now()) } else { None };
    let wo = find(gguf, &format!("blk.{block}.attn_output.weight"))?;
    if let Some(t_p4) = t_p4 {
        params_ns += t_p4.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("attn_params", params_ns);
    }
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

// `Q8Block` moved to qdot with the cell kernel that consumes it (MUL-38);
// re-exported here so `Derived`'s storage, the gates' `assert_eq!` and the
// 34-byte size assertion (`tests/derived.rs`) keep compiling unchanged.
// Field-wise equality on the type is equality of every byte it holds.
pub use qdot::Q8Block;

/// The weight requant the reference's `wk_b` cast runs: `quantize_row_q8_0`, x86
/// branch (ggml-quants.c:938+): `d = amax/127` stored f16, `id = 127/amax` — a
/// different f32 value than `1/d` — and `_mm256_round_ps(_MM_ROUND_NEAREST)` codes.
/// The ref variant (`id = 1/d`, `roundf`) differs on last-ulp and tie cases and is
/// NOT what runs here: switching to it flips codes on 10 of 16 heads and opens ~1e-2
/// on the gate (measured 2026-09-19). Verified byte-identical to the reference's
/// derived `attn_k_b.weight` for block 0 (0 of 69632 bytes differ).
pub fn quantize_q8_0(x: &[f32]) -> Q8Block {
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

// `ActBlock` moved to qdot with the cell kernel (MUL-38); the quantizer
// below still owns its conventions — the scale as **bf16**, int8 codes.
use qdot::ActBlock;

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
/// cast to Q8_0 whose 32-value blocks run along the 128-wide q_nope axis. That
/// derivation is a pure function of the weights, so it now runs once in
/// [`Derived`](crate::derived::Derived) at load; this function receives the blocks
/// (`wblocks`, one block's heads concatenated head-major) and does the per-token
/// half. At M ≤ 7 rows ik's `iqk_mul_mat_4d` claims the node and quantizes the F32
/// activation itself, into its small-M `block_q8_2` format (see [`quantize_act`]) —
/// the dot is then a sum over blocks of `f32(f16(dw)) · dq · Σ qw·qq` with an exact
/// i32 inner sum. The block sum accumulates in f64: ggml's f32 SIMD lane order
/// differs in its last ulp; ours is the exact sum. Since MUL-38 the i32 inner
/// sum runs in qdot's AVX2 maddubs kernel — bit-identical to the scalar form
/// by integer associativity (the argument and its gate: crates/qdot, MUL-38).
///
/// The (column, row) cell space runs on the resident pool (MUL-33). Each output
/// cell depends only on its own weight blocks and its column's quantized
/// activation, and the only accumulation — the f64 block sum — stays inside the
/// cell, so splitting cells cannot reorder anything and the output is
/// bit-identical to the serial loop; `tests/mt.rs` holds that as a byte compare
/// across thread counts. The activation quantization stays on the caller thread,
/// once per (h, t), before the split.
pub fn q_nope2_absorbed(
    wblocks: &[Q8Block],
    q: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    // Profiler hook (crate::profile). This site was hypothesis 3 — the per-step
    // wk_b requant — and `Derived` closed it: what is left is the per-token half.
    // The stage split keeps its axes (the caller-side quant pre-pass as
    // `quant_act`, the pool cell loops as `dot`); `dequant_w` can no longer fire
    // from this site, so a nonzero `dequant_w` row here would mean someone
    // reintroduced a per-step requant.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    // A wrong-length slice is the failure this check exists for: the indexing
    // below would otherwise read another head's weights silently and keep
    // producing plausible numbers.
    if wblocks.len() != p.n_head * span {
        return Err(ModelError::Shape {
            what: "q_nope2_absorbed wblocks (n_head·latent·nope/32)",
            want_ne0: p.n_head * span,
            want_ne1: 0,
            got_ne0: wblocks.len(),
            got_ne1: 0,
        });
    }
    let mut out = Tensor2::zeros(p.latent, p.n_head * q.ne1);

    // Quantize each (h, t) activation slice once, on the caller thread, before
    // the cell split — the same "quantize once per pair" argument
    // `matmul_q_multi` rides at its own quantization site. One 128-wide slice
    // feeds all `latent` output rows of its column, and once the columns live
    // on the pool a column straddling a chunk boundary must not be quantized
    // twice. Serial on purpose, as there: the quant stage is 0.4 of the site's
    // 11.0 ms (level 2, 2026-09-20, timer tax included), and `quantize_act`
    // is deterministic — a shared read-only `qall`, laid out column-major so
    // chunk `col` reads a contiguous run, is byte-identical to the per-(h, t)
    // Vec the serial loop used to build inside the t loop.
    let t_qa = if lvl >= 2 { Some(Instant::now()) } else { None };
    let mut qall: Vec<ActBlock> = Vec::with_capacity(p.n_head * q.ne1 * nblocks);
    for h in 0..p.n_head {
        for t in 0..q.ne1 {
            let qbase = t * q.ne0 + h * p.kq_head;
            let qrow = &q.data[qbase..qbase + p.nope];
            qall.extend(qrow.chunks_exact(32).map(quantize_act));
        }
    }
    if let Some(t_qa) = t_qa {
        pacc.add_quant_act(t_qa.elapsed().as_nanos() as u64);
    }

    // MUL-33: the output cell space — column `col = h·ne1 + t`, row `j` — runs
    // on the resident pool. This was the last big serial site of the decode
    // step (11.0 ms of a 41.2 ms wall, level 2, N=96, 2026-09-20; 27 calls per
    // step, one per layer), all of it on the caller thread while the pool
    // idled through the attention section. The independence argument is the
    // cell form of the row split `matmul_q` rides (MUL-23): cell (col, j)
    // reads only `wblocks[h·span + j·nblocks + b]` and its own column's
    // quantized blocks, and its only accumulation — the b = 0..4 f64 sum — is
    // confined inside the cell, so which thread computes which cell cannot
    // change a bit. Splitting the b axis would reorder that f64 accumulation
    // and is forbidden. ik dispatches this same node over (heads × row
    // chunks) on its own pool (iqk_mul_mat.cpp:637-676); the cell split is
    // this engine's spelling of that axis. `t` is the column axis, so prefill
    // (ne1 > 1) is the same space with more columns — nothing below branches
    // on which case it is.
    //
    // SAFETY (construction site): every participant computes cells of the
    // (col, j) space inside its own contiguous chunk and writes only cell
    // `col * latent + j` of `out.data` — the chunks partition the cell
    // space, so no two participants alias. The pool's completion protocol
    // publishes the writes before this function reads `out`, the same
    // ordering that publishes the matmul job slot. The per-cell arithmetic
    // runs in qdot's cell kernel (MUL-38): its SIMD is integer-exact by
    // associativity and its f64 block-sum epilogue is this loop's own
    // expression and order, so every cell is bit-identical to the serial
    // loop the kernel replaced — `tests/mt.rs` still holds the whole claim
    // as a byte compare across thread counts.
    let out_ptr = crate::ops::SharedOut(out.data.as_mut_ptr());
    let latent = p.latent;
    let ne1 = q.ne1;
    let collected: Mutex<Vec<profile::CallAcc>> =
        Mutex::new(Vec::with_capacity(threads::pool().threads()));
    threads::pool().for_each_chunk(p.n_head * ne1 * latent, |cells| {
        let mut acc = profile::CallAcc::new();
        // Per-worker scratch (MUL-38): qdot's kernel writes a segment's
        // cells here and they then go to `out` through `SharedOut::write` —
        // the one-cell method is what keeps this closure capturing a
        // `&Sync` type (see `SharedOut`'s doc). One latent-wide f32 run per
        // column segment, reused across the chunk's segments.
        let mut cellbuf = vec![0.0f32; latent];
        // Walk the chunk column-segment-wise: a chunk may start mid-column
        // and end mid-column, so each iteration takes the intersection of the
        // remaining chunk and one column — `whead`/`qcol` are then fetched
        // once per segment, not once per cell.
        let mut c = cells.start;
        while c < cells.end {
            let col = c / latent;
            let j0 = c - col * latent;
            let j_end = j0 + (cells.end - c).min(latent - j0);
            let h = col / ne1;
            let whead = &wblocks[h * span..(h + 1) * span];
            let qcol = &qall[col * nblocks..(col + 1) * nblocks];
            let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
            // MUL-38: the segment's per-block i32 dots run in qdot's AVX2
            // maddubs kernel — the Q6_K qY sign fold (weight magnitude on
            // the u8 side, weight sign folded into the activation bytes; no
            // +128 prepare, no compensation sum, no reachable saturation).
            // Integer associativity makes the lane order free and the
            // kernel's f64 epilogue is the loop's own, so the cells are
            // bit-identical to the scalar form — the argument and its gate
            // live in crates/qdot (MUL-38 section).
            let seg = &mut cellbuf[..j_end - j0];
            qdot::q_nope2_cells(whead, qcol, j0, j_end, seg);
            for (jj, &v) in seg.iter().enumerate() {
                // SAFETY: cell `col * latent + j0 + jj` belongs to this
                // chunk's range of the cell split alone — see the
                // construction-site comment above.
                unsafe { out_ptr.write(col * latent + j0 + jj, v) };
            }
            if let Some(t_dot) = t_dot {
                acc.add_dot(t_dot.elapsed().as_nanos() as u64);
            }
            // j_end is a row index INSIDE the column (0..latent); the walk
            // variable is absolute, so the next segment starts at the
            // column's base plus j_end. Writing `c = j_end` here kept c at
            // latent forever once the first column ended — the first
            // segment only looked right because its base was zero.
            c = col * latent + j_end;
        }
        // The one lock of the chunk — never per cell: a mutex inside the cell
        // loop would profile the mutex. The accumulator mutex stays out of the
        // measured region the same way `record`'s does.
        collected
            .lock()
            .expect("q_nope2_absorbed chunk accumulator")
            .push(acc);
    });

    // Post-join bookkeeping: fold the chunk accumulators so `record` still
    // fires exactly once per call, after its `Instant` pair closed. Arrival
    // order is nondeterministic and nothing depends on it — the fold is
    // addition, and there is no error precedence to keep because the shape
    // check fired before the dispatch and the cell loop cannot fail. Since
    // MUL-23 this section is bookkeeping only; level 2 times it as the
    // `gather` stage to keep that honest, mirroring `matmul_q_multi`'s tail.
    let t_gather = if lvl >= 2 { Some(Instant::now()) } else { None };
    for acc in collected
        .into_inner()
        .expect("q_nope2_absorbed chunk accumulator")
    {
        pacc.add_acc(&acc);
    }
    if let Some(t_gather) = t_gather {
        pacc.add_gather(t_gather.elapsed().as_nanos() as u64);
    }
    if let Some(t_call) = t_call {
        // Shape statement for the derived weights this call consumed:
        // `n_head · latent` output rows contracted over `nope`. The bytes are the
        // Q8_0 blocks' — tag 8 is ggml's Q8_0; the gguf crate has no variant for
        // it because nothing in the file is Q8_0 (these blocks are derived), and
        // the row should name what the step reads, not what the file holds.
        let w_rows = (p.n_head * p.latent) as u64;
        profile::record(
            "q_nope2_absorbed",
            gguf::GgmlType::Unknown(8),
            w_rows,
            w_rows * p.nope as u64,
            std::mem::size_of_val(wblocks) as u64,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
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
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
) -> Tensor2 {
    // Profiler hook (crate::profile): level-1 timer over the whole kernel. The
    // shape statement: `rows` is the (token, head) query rows produced, `k` the
    // keys·d_head dot work each row walks, and `weight_bytes` the f16 KV rows
    // read at least once — V is the latent tail of the same rows, so the bytes
    // are counted once. Level 1 only: a kq-dot / softmax / V-accumulate stage
    // split is a later round's tool, worth building once this row says whether
    // the site is big enough to care.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    // Queries and keys are two different counts the moment a KV cache exists: one
    // decode step is a single query against a whole prefix. They were one array while
    // the batch WAS the cache, and every index below that reads `n_tokens` is a query
    // index -- `q_rope` and `q_nope2` are laid out by the batch, never by the cache.
    assert_eq!(
        keys16.len(),
        key_slots.len(),
        "every cached key row needs its slot"
    );
    let n_tokens = q_slots.len();
    let d_head = p.rope_dims + p.latent;
    let mut out = Tensor2::zeros(p.latent, n_tokens * p.n_head);

    // MUL-29: the (token, head) query rows are fully independent — each one
    // walks the KV cache and writes its own contiguous `latent`-wide slice of
    // `out`, so the row space splits on the pool with the same argument
    // matmul_q rides (MUL-23). At short ctx this site was 10 ms/step and at
    // average ctx 53 it was 73.8 ms — 42 % of the step (measured, level 1,
    // N=96) — all of it serial on the caller thread while the pool idled
    // through the attention section.
    //
    // SAFETY (construction site): every participant computes rows of the
    // (t, h) space inside its own chunk and writes only cells
    // `row * latent .. (row + 1) * latent` of `out.data` — one contiguous
    // disjoint slice per row, so no two participants alias. The pool's
    // completion protocol publishes the writes before this function reads
    // `out`, the same ordering that publishes the matmul job slot. The
    // per-row arithmetic is untouched: `tests/mt.rs` holds the whole claim as
    // a byte compare across thread counts.
    let out_ptr = crate::ops::SharedOut(out.data.as_mut_ptr());
    let latent = p.latent;
    threads::pool().for_each_chunk(n_tokens * p.n_head, |rows| {
        // Per-worker scratch: the caller's stack buffers became one set per
        // participant. Sized exactly as the serial version had them.
        let mut qrow = vec![0.0f32; d_head];
        let mut r = vec![0.0f32; latent];
        let mut w = [0.0f32; 32];
        for row in rows {
            let t = row / p.n_head;
            let h = row % p.n_head;
            // The FA row: [q_rope ; q_nope2], F32, never rounded.
            qrow[..p.rope_dims].copy_from_slice(q_rope.col(t * p.n_head + h));
            qrow[p.rope_dims..].copy_from_slice(q_nope2.col(h * n_tokens + t));

            let mut m = f32::NEG_INFINITY;
            let mut s_sum = 0.0f32;
            r.fill(0.0);
            for blk in (0..key_slots.len()).step_by(32) {
                let mut s = [f32::NEG_INFINITY; 32];
                let mut smax = f32::NEG_INFINITY;
                for (l, sl) in s.iter_mut().enumerate() {
                    let u = blk + l;
                    let su = match key_slots.get(u) {
                        Some(su) => su,
                        None => break, // past the cache: padding, weight exactly 0
                    };
                    if su.seq != q_slots[t].seq || su.pos > q_slots[t].pos {
                        continue; // the -inf half of the causal mask
                    }
                    let kq = kq_dot_fa4(&qrow, &keys16[u]);
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
                    let krow = &keys16[blk + l];
                    for d in 0..latent {
                        r[d] = half_to_f32(krow[p.rope_dims + d]).mul_add(w[l], r[d]);
                    }
                }
            }

            let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
            for (d, &v) in r.iter().enumerate() {
                // SAFETY: cell `row * latent + d` belongs to this row alone —
                // see the construction-site comment above.
                unsafe { out_ptr.write(row * latent + d, s_inv * v) };
            }
        }
    });
    if let Some(t_call) = t_call {
        let acc = profile::CallAcc::new();
        profile::record(
            "flash_attn_latent",
            gguf::GgmlType::F16,
            (n_tokens * p.n_head) as u64,
            (key_slots.len() * d_head) as u64,
            (key_slots.len() * d_head * 2) as u64,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    out
}

// ----------------------------------------------------------------- wv_b

/// `wv_b` stays Q3_K in the reference (only `wk_b` is requantized), so this is a
/// `matmul_q_batch` over synthetic views of `attn_kv_b`'s v-up rows — the same views
/// `ggml_view_3d` takes, expressed as file offsets. Activation gathering is needed
/// because `kqv_compressed` interleaves heads (`t·n_head+h`) while each head's matmul
/// wants six consecutive 512-wide rows.
///
/// MUL-25: this used to be one `matmul_q` per head — sixteen pool dispatches per
/// layer, 432 per decode step, and MUL-24 had just measured small dispatches as
/// where the orchestration residual lives (26 µs each here). All heads share the
/// view shape, so the whole layer is one `matmul_q_batch` call now. Values are
/// bit-identical by the batch primitive's argument: per-pair rows and activations
/// are exactly what the per-head calls saw.
pub fn wv_b_heads(
    gguf: &Gguf,
    wkb: &TensorInfo,
    kqv_compressed: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    // Profiler hook (crate::profile): SELF time — the whole call minus what its
    // one hooked `matmul_q_batch` child recorded (see `profile::site_ns_total`).
    // What is left is the per-head activation gather, the view builds and the
    // scatter into `kqv_2d`. Per-piece timers were rejected on the overhead
    // rule: a piece here is a 512-float copy, the same order as an Instant
    // pair, so pieces would measure the timer. The subtraction form measures
    // the shuffling once, honestly.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let mm_before = if lvl > 0 {
        profile::site_ns_total("matmul_q_batch")
    } else {
        0
    };
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    let n_tokens = kqv_compressed.ne1 / p.n_head;
    let mut kqv_2d = Tensor2::zeros(p.n_head * p.v_head, n_tokens);

    let mut xhs: Vec<Tensor2> = (0..p.n_head)
        .map(|_| Tensor2::zeros(p.latent, n_tokens))
        .collect();
    let mut views: Vec<TensorInfo> = Vec::with_capacity(p.n_head);
    for h in 0..p.n_head {
        for t in 0..n_tokens {
            let src = kqv_compressed.col(t * p.n_head + h);
            xhs[h].col_mut(t).copy_from_slice(src);
        }
        views.push(TensorInfo {
            name: format!("{}.v_up.head{h}", wkb.name),
            dims: vec![p.latent as u64, p.v_head as u64],
            ty: wkb.ty,
            offset: wkb.offset + ((h * (p.nope + p.v_head) + p.nope) * row_bytes) as u64,
            nbytes: wkb.ty.type_size().unwrap()
                * (p.latent / wkb.ty.blck_size().unwrap() as usize) as u64
                * p.v_head as u64,
        });
    }
    let ws: Vec<&TensorInfo> = views.iter().collect();
    let xs: Vec<&Tensor2> = xhs.iter().collect();
    let outs = matmul_q_batch(gguf, &ws, &xs)?;
    for (h, out_h) in outs.iter().enumerate() {
        for t in 0..n_tokens {
            let dst = kqv_2d.col_mut(t);
            dst[h * p.v_head..(h + 1) * p.v_head].copy_from_slice(out_h.col(t));
        }
    }
    if let Some(t_call) = t_call {
        let child = profile::site_ns_total("matmul_q_batch") - mm_before;
        let self_ns = (t_call.elapsed().as_nanos() as u64).saturating_sub(child);
        profile::record_time("wv_b_heads", self_ns);
    }
    Ok(kqv_2d)
}
