//! Block-N MLA attention: `attn_norm-N` in, `kqv_out-N` out; gate
//! `crates/model/tests/attn.rs` against the oracle. The reference is ik_llama.cpp's
//! MLA graph (weight-absorbed, 512-wide latent), mirrored in the same op order — six
//! of its choices are invisible in the math, and each alone would blow the 1e-4 gate:
//!
//!   * YaRN mscale does not scale the rope — ik divides it back out before rope
//!     (build_deepseek2.cpp:1250-1254); mscale lands in `kq_scale = mscale²/√key_length`.
//!   * `wk_b` is requantized to Q8_0 — 32-value blocks along the 128-wide
//!     q_nope axis, f16 scales (llama.cpp:3216-3230); `wv_b` stays Q3_K.
//!   * The KV cache is F16: `kvr` is rounded f32→f16 before attention reads
//!     it back as both K (576 wide) and V (its latent tail).
//!   * At M ≤ 7 rows activations quantize to ik's `block_q8_2` (bf16 scale, `id = 1/d`),
//!     not Q8_0 (see [`quantize_act`]) — a 2⁻⁹-relative difference from Q8_0.
//!   * The reference flash-attention is ik's `iqk` templates, not ggml's generic loop:
//!     `FlashAttn<576, 512, ·, 32>`, F16 K/V helpers (iqk_fa_576_512.cpp), −inf-masked
//!     256-padded cache, q F32, fa4-lane QK dot, ik's `v_expf`, f32-FMA V chain, `1/S`
//!     multiply — see [`flash_attn_latent`].
//!   * Those inner loops are AVX2+FMA+F16C on the reference build; the scalar twin
//!     ([`flash_attn_latent_scalar`]) is the no-AVX2 fallback and oracle.
//!
//! Contract: `x` is `[embd, n_tokens]`, `slots.len() == x.ne1`, the batch's own tokens are the
//! KV entries (prefill); `t` attends to `u` iff `slots[u].seq == slots[t].seq && slots[u].pos <= slots[t].pos`.

use crate::derived::Derived;
use crate::kv::KvCache;
use crate::ops::{matmul_q, matmul_q_batch, matmul_q_group, rms_norm};
use crate::profile;
use crate::{ModelError, Slot, Tensor2};
use gguf::quant::half_to_f32;
use gguf::{Gguf, TensorInfo};
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

/// All intermediates, for the gate. `Tensor2` holds `ne0` contiguous with trailing dims folded into `ne1`, laid out exactly as the oracle dumps flatten them.
pub struct AttnTrace {
    /// `q-N` occurrence 0: `attn_q @ x`, `{3072, n_tokens}`.
    pub q: Tensor2,
    /// `kv_rope_compressed-N`: `attn_kv_a_mqa @ x`, `{576, n_tokens}` — `[latent(512) ; rope(64)]` per token.
    pub kv_rope_compressed: Tensor2,
    /// `q_rope-N` occurrence 1: rope output, `{64, n_head·n_tokens}` (columns `t·n_head+h`).
    pub q_rope: Tensor2,
    /// `kv_compressed-N` occurrence 1: rms-normed latent, `{512, n_tokens}`.
    pub kv_compressed: Tensor2,
    /// `kvr-N`: `[k_rope(64) ; kv_compressed(512)]` per token, `{576, n_tokens}` (rope first — oracle-verified).
    pub kvr: Tensor2,
    /// `q_nope2-N`: `wk_b(Q8_0)ᵀ @ q_nope` per head, `{512, n_head·n_tokens}` (columns `h·n_tokens+t`).
    pub q_nope2: Tensor2,
    /// `kqv_compressed-N`: attention over the latent, `{512, n_head·n_tokens}` (columns `t·n_head+h`).
    pub kqv_compressed: Tensor2,
    /// `kqv_out-N`: `attn_output @ kqv`, `{2048, n_tokens}`.
    pub kqv_out: Tensor2,
}

/// One block's attention: `x` is the block's `attn_norm-N` (already normed); the result `kqv_out-N` feeds the residual add.
pub fn block_attn(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    slots: &[Slot],
    derived: &Derived,
) -> Result<Tensor2, ModelError> {
    Ok(block_attn_trace(gguf, block, x, slots, derived)?.kqv_out)
}

/// The same computation, keeping every intermediate the gate asserts. Prefill semantics: the cached path runs against a cache holding exactly this batch — one code path, so "caching changes nothing" is structural (`tests/kv.rs`).
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

/// One block's attention against a KV cache: `q_slots` are this call's queries, the cache holds every key; new rows are appended first, so a decode step attends to its own token as well as the prefix. `cache_block` is the cache's block index — the model's `block` in a real pass, 0 in the scratch cache.
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
    // Profiler hook: piece timers around the attention prep that is not a matmul (the
    // matmuls keep their own rows): `attn_params`, `attn_latent`, `attn_rope`, `attn_kvr`.
    let lvl = profile::level();
    let mut params_ns = 0u64;
    let t_p1 = if lvl > 0 { Some(Instant::now()) } else { None };
    // The plan holds everything this block's attention reads that the tokens
    // cannot change: geometry, weight views, the decoded kv_a norm gain.
    // `Derived::new` already ran `MlaParams::read`'s cross-checks for it.
    let ap = derived.attn_plan(block)?;
    let p = &ap.params;
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

    // 1. The two projections out of the normed activations — one group
    //    dispatch over the same `x`: the pair mixes types and row counts,
    //    which is the heterogeneous shape.
    let wq = &ap.wq;
    let wa = &ap.wa;
    if let Some(t_p1) = t_p1 {
        params_ns += t_p1.elapsed().as_nanos() as u64;
    }
    let mut proj = matmul_q_group(gguf, &[wq, wa], &[x, x])?;
    let kv_rope_compressed = proj.pop().expect("one output per pair");
    let q = proj.pop().expect("one output per pair");

    // 2. Latent norm. The gain is F32 in the file, decoded once at load.
    let t_p2 = if lvl > 0 { Some(Instant::now()) } else { None };
    let gain = &ap.kv_a_norm_gain;
    if let Some(t_p2) = t_p2 {
        params_ns += t_p2.elapsed().as_nanos() as u64;
    }
    let t_lat = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut latent = Tensor2::scratch(p.latent, x.ne1);
    for t in 0..x.ne1 {
        let src = &kv_rope_compressed.col(t)[..p.latent];
        latent.col_mut(t).copy_from_slice(src);
    }
    if let Some(t_lat) = t_lat {
        profile::record_time("attn_latent", t_lat.elapsed().as_nanos() as u64);
    }
    let kv_compressed = rms_norm(&latent, gain, p.eps);

    // 3. Rope: one cos/sin cache per position, shared by every head; k_rope from the
    //    kv_a tail, q_rope from each q head's rope slice. The cache buffers are
    //    per-thread and recycled: the values are position-dependent, the storage
    //    is not, and every element is rewritten before it is read.
    let t_rope = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut k_rope = Tensor2::scratch(p.rope_dims, x.ne1);
    let mut q_rope = Tensor2::scratch(p.rope_dims, p.n_head * x.ne1);
    ROPE_BUFS.with(|pool| {
        let mut bufs = pool.borrow_mut();
        if bufs.len() < x.ne1 {
            bufs.resize_with(x.ne1, || vec![0.0f32; p.rope.n_dims]);
        }
        for (t, s) in slots.iter().enumerate() {
            p.rope.cache_into(s.pos, &mut bufs[t]);
        }
        for t in 0..x.ne1 {
            let cache = &bufs[t];
            let ksrc = &kv_rope_compressed.data[t * kv_width + p.latent..(t + 1) * kv_width];
            rope_pair(ksrc, k_rope.col_mut(t), cache);
            for h in 0..p.n_head {
                let base = t * q.ne0 + h * p.kq_head + p.nope;
                let qsrc = &q.data[base..base + p.rope_dims];
                rope_pair(qsrc, q_rope.col_mut(t * p.n_head + h), cache);
            }
        }
    });
    if let Some(t_rope) = t_rope {
        profile::record_time("attn_rope", t_rope.elapsed().as_nanos() as u64);
    }

    // 4. kvr = [k_rope ; kv_compressed] (order verified against the oracle dump).
    let t_kvr = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut kvr = Tensor2::scratch(kv_width, x.ne1);
    for t in 0..x.ne1 {
        let dst = kvr.col_mut(t);
        dst[..p.rope_dims].copy_from_slice(k_rope.col(t));
        dst[p.rope_dims..].copy_from_slice(kv_compressed.col(t));
    }

    // 5. The F16 rows the reference attends over: K = f16(kvr), V = its latent tail.
    //    Pushed before attention — a prefill attends to itself.
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

    // 6-8. The per-head chain — q_nope2 absorption, attention over the latent,
    //      wv_b — as ONE pool dispatch over the (token, head) rows; three
    //      separate dispatches paid two caller-alone barriers per block for
    //      stages that are independent per row. `block` (the model's) selects
    //      the derived weights — never `cache_block`, which is 0 on the
    //      scratch path.
    let t_p3 = if lvl > 0 { Some(Instant::now()) } else { None };
    let v_up_views = &ap.v_up_views;
    if let Some(t_p3) = t_p3 {
        params_ns += t_p3.elapsed().as_nanos() as u64;
    }
    let (q_nope2, kqv_compressed, kqv_2d) = attn_heads_fused(
        gguf,
        derived.wk_b_all_heads(block)?,
        &q,
        &q_rope,
        cache.keys(cache_block),
        cache.slots(),
        slots,
        v_up_views,
        p,
    )?;
    let t_p4 = if lvl > 0 { Some(Instant::now()) } else { None };
    let wo = &ap.wo;
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

/// Everything rope needs, with the YaRN constants the graph passes: `beta_fast = 32.0f`, `beta_slow = 1.0f`, `ext_factor = attn_factor = 1.0f` (llama-graph.cpp call-site constants, not in the file).
#[derive(Clone)]
pub struct RopeParams {
    pub n_dims: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    /// The graph's rope mscale: `1/(1+0.1·ln(1/freq_scale))` — `rope_yarn` multiplies it back to ≈1, so rope stays a pure rotation.
    pub mscale_param: f32,
    pub corr_dims: [f32; 2],
    /// `freq_base^(-2/n_dims)` as one f32 `powf`, as ggml computes it.
    pub theta_scale: f32,
}

impl RopeParams {
    /// `ggml_rope_yarn_corr_dims` (ggml.c:20771), same f32 op order; `[10, 23]` here.
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

    /// cos/sin cache for one position: `[cos0, sin0, …]` per pair.
    /// `ggml_rope_cache_init` (ggml.c:20813): theta is a running product (× `theta_scale` per pair), hence a loop, not `powi`; `sin_sign = +1`, so the multiply by it is skipped as the no-op it is.
    pub fn cache(&self, pos: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_dims];
        self.cache_into(pos, &mut out);
        out
    }

    /// [`cache`](RopeParams::cache) into caller storage — the same op order, so
    /// a recycled buffer and a fresh one hold the same bytes.
    pub fn cache_into(&self, pos: u32, dst: &mut Vec<f32>) {
        let npairs = self.n_dims / 2;
        if dst.len() != self.n_dims {
            dst.clear();
            dst.resize(self.n_dims, 0.0);
        }
        let mut theta = pos as f32;
        for i in 0..npairs {
            let (c, s) = self.yarn(theta, 2 * i);
            dst[2 * i] = c;
            dst[2 * i + 1] = s;
            theta *= self.theta_scale;
        }
    }

    /// `rope_yarn` (ggml.c:20794). `i0` is the *dim* loop variable (steps of 2); the ramp compares `i0/2` — the pair index — against `corr_dims`.
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

/// Adjacent-pair NORM rotation of one row: `y[2i] = x0·cos − x1·sin`, `y[2i+1] = x0·sin + x1·cos`, f32 ops in ggml's order.
fn rope_pair(src: &[f32], dst: &mut [f32], cache: &[f32]) {
    for i in (0..src.len()).step_by(2) {
        let (x0, x1) = (src[i], src[i + 1]);
        let (c, s) = (cache[i], cache[i + 1]);
        dst[i] = x0 * c - x1 * s;
        dst[i + 1] = x0 * s + x1 * c;
    }
}

thread_local! {
    /// Per-thread rope cos/sin buffers, recycled across calls: the values are
    /// recomputed for the position every call (the fill covers the whole row —
    /// rope dims are even, `MlaParams::read` checks it), only the storage is
    /// reused. One outer buffer per token of the batch.
    static ROPE_BUFS: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

// -------------------------------------------------------------- parameters

/// The block's geometry and scalars, all read from the file (never literals), with the derived relations cross-checked so a differently-shaped MLA file fails loudly.
#[derive(Clone)]
pub struct MlaParams {
    pub n_head: usize,
    /// qk head dim = nope + rope.
    pub kq_head: usize,
    /// `key_length − rope.dimension_count`.
    pub nope: usize,
    pub rope_dims: usize,
    pub v_head: usize,
    /// kv_lora_rank — read from `attn_kv_b`'s row length, the same quantity ggml gets from hparams.
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

        // Derived geometry must close, or the file is not shaped like this MLA
        // (checked_sub: a key_length short of the rope dims = malformed file).
        let Some(nope) = kq_head.checked_sub(rope_dims) else {
            return Err(ModelError::Shape {
                what: "kq_head - rope_dims (key_length must cover the rope dims)",
                want_ne0: kq_head,
                want_ne1: 0,
                got_ne0: rope_dims,
                got_ne1: 0,
            });
        };
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
        // Geometry the kernels below would truncate SILENTLY or panic on — the same
        // fail-loudly policy as the span checks: nope % 32 (`q_nope2_absorbed` works in
        // 32-value blocks), (rope_dims + latent) % 8 (`kq_dot_fa4`'s `step_by(8)` drops
        // a tail), rope_dims % 2 (rope rotates adjacent pairs). The SIMD leg's stricter
        // % 32 / latent % 8 contracts are `flash_simd`'s own gate, not encoded here.
        check("nope axis (32-value blocks)", 0, nope % 32)?;
        check(
            "kq row width rope+latent (8-value chunks)",
            0,
            (rope_dims + latent) % 8,
        )?;
        check("rope dims (adjacent pairs)", 0, rope_dims % 2)?;

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

/// f32 → f16 bits, round-to-nearest-even — `vcvtps2ph $0x0` on the reference build (AVX2 + F16C, not AVX-512). Subnormals and ties follow IEEE; NaN/inf collapse to inf (no NaN reaches the gated graph). The crate's one f16 round-trip helper: q rows, KV cache and V accumulator all round through here.
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
        // Normal f16: keep 11 significand bits; round the dropped 13 with the ties-to-even carry `v + 0x0fff + ((v >> 13) & 1)`.
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

// `Q8Block` lives in qdot with the cell kernel; re-exported so `Derived`'s storage, the gates' `assert_eq!` and the size assertion (`tests/derived.rs`) keep compiling; field-wise equality is equality of every byte.
pub use qdot::Q8Block;

/// The weight requant the reference's `wk_b` cast runs: `quantize_row_q8_0`, x86 branch
/// (ggml-quants.c:938+) — `d = amax/127` stored f16, `id = 127/amax` (a different f32
/// than `1/d`), `_mm256_round_ps(_MM_ROUND_NEAREST)` codes; the ref variant (`id = 1/d`,
/// `roundf`) is NOT what runs here.
pub fn quantize_q8_0(x: &[f32]) -> Q8Block {
    // One 32-value block at a time: a shorter slice would leave trailing codes at 0 silently.
    assert_eq!(x.len(), 32, "quantize_q8_0: one 32-value block at a time");
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    let mut q = [0i8; 32];
    for (j, &v) in x.iter().enumerate() {
        q[j] = qdot::nearest_int(v * id).clamp(-128, 127) as i8;
    }
    Q8Block {
        d: f32_to_f16_bits(d),
        q,
    }
}

// `ActBlock` lives in qdot with the cell kernel; the quantizer below still owns its conventions — the scale as **bf16**, int8 codes.
use qdot::ActBlock;

/// `ggml_compute_fp32_to_bf16` (ggml-impl.h:106): round-to-nearest-even via the carry trick. NaN cannot reach the gated graph.
fn bf16_round(x: f32) -> f32 {
    let mut b = x.to_bits();
    b += 0x7fff + ((b >> 16) & 1);
    f32::from_bits(b & 0xffff_0000)
}

/// The activation quantizer the reference actually runs for this mul_mat —
/// `quantize_row_q8_1_x4_T<block_q8_2, …>` (iqk_quantize.cpp:1074+), NOT
/// `quantize_row_q8_0`: at M ≤ 7 rows `iqk_mul_mat_4d` quantizes the F32 activation
/// itself, with the scale `amax/127` rounded to bf16 and the codes `id = 1/d` on that
/// rounded scale, RNE. Each difference alone moves gated values by up to 2⁻⁹ relative —
/// far outside the 1e-4 gate.
///
/// Codes clamp to ±127, ENFORCED, not defensive: a -128 activation code under a
/// negative weight code is the one input pair outside the sign-fold kernel's
/// bit-identity contract (`qdot::q_nope2_cells_avx2_inner`; the producer bound
/// |code| < 127.25 before RNE means the clamp never engages on legal inputs, so
/// -128 → -127 is bit-identical by derivation). `quantize_q8_0`'s clamp stays
/// -128: a WEIGHT code of -128 is legal.
fn quantize_act(x: &[f32]) -> ActBlock {
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = bf16_round(amax / 127.0);
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    let mut q = [0i8; 32];
    for (j, &v) in x.iter().enumerate() {
        q[j] = qdot::nearest_int(v * id).clamp(-127, 127) as i8;
    }
    ActBlock { d, q }
}

thread_local! {
    /// Per-thread recycled scratch for the pool dispatches: the (h, t)
    /// activation blocks and the gather on the caller thread, one cell/row
    /// buffer set per worker. Each dispatch rewrites what it reads; the
    /// buffers only ever grow, never shrink.
    static QALL: RefCell<Vec<ActBlock>> = const { RefCell::new(Vec::new()) };
    static CELL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// The flash row kernel's query row and V accumulator, one pair per worker
    /// thread — sliced to the exact `d_head`/`latent` the row kernels index.
    static FLASH_QROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static FLASH_R: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// `q_nope2 = wk_b(Q8_0)ᵀ · q_nope` per head — the weight-absorption step.
///
/// `wk_b` is not in the file: the reference derives it at load (`llm_prepare_mla`,
/// llama.cpp:3229 — k-up rows of `attn_kv_b`, dequantized, transposed, Q8_0 blocks
/// along q_nope), a pure function of the weights, so it runs once in
/// [`Derived`](crate::derived::Derived); this is the per-token half.
///
/// The dot is a sum over blocks of `f32(f16(dw)) · dq · Σ qw·qq` with an exact i32
/// inner sum (qdot's AVX2 maddubs kernel — bit-identical to scalar by integer
/// associativity; argument and gate in crates/qdot); the block sum accumulates in
/// f64, because ggml's f32 SIMD lane order differs in its last ulp and ours is the
/// exact sum. The cell space runs on the resident pool, bit-identical to the serial
/// loop (construction site below; `tests/mt.rs`); the activation quantization stays
/// on the caller thread, once per (h, t), before the split.
pub fn q_nope2_absorbed(
    wblocks: &[Q8Block],
    q: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    // Profiler hook: the per-token half — the quant pre-pass (`quant_act`) and
    // the pool cell loops (`dot`). `dequant_w` cannot fire here (the requant
    // lives in `Derived`); a nonzero row means a per-step requant came back.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    // A wrong-length slice is what this check exists for: the indexing below would read another head's weights silently.
    if wblocks.len() != p.n_head * span {
        return Err(ModelError::Shape {
            what: "q_nope2_absorbed wblocks (n_head·latent·nope/32)",
            want_ne0: p.n_head * span,
            want_ne1: 0,
            got_ne0: wblocks.len(),
            got_ne1: 0,
        });
    }
    let mut out = Tensor2::scratch(p.latent, p.n_head * q.ne1);

    // Quantize each (h, t) slice once, on the caller thread, before the cell split —
    // the "quantize once per pair" argument `matmul_q_multi` rides; a column
    // straddling a chunk boundary must not be quantized twice. Deterministic, so the
    // shared column-major `qall` is byte-identical to a per-(h, t) quant.
    let t_qa = if lvl >= 2 { Some(Instant::now()) } else { None };
    // Recycled across calls: the quantized (h, t) blocks are per-step values in
    // stable storage — the same capacity serves every block of a run. Held for
    // the whole call on this thread; pool workers read it by shared reference.
    let mut qall = QALL.with(|c| std::mem::take(&mut *c.borrow_mut()));
    qall.clear();
    qall.reserve(p.n_head * q.ne1 * nblocks);
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

    // The output cell space — column `col = h·ne1 + t`, row `j` — runs on the pool.
    // Cell (col, j) reads only its column's quantized blocks and
    // `wblocks[h·span + j·nblocks + b]`, and its only accumulation (the b = 0..4 f64
    // sum) is confined inside the cell, so the split cannot change a bit; splitting
    // the b axis would reorder that sum and is forbidden. `t` is the column axis, so
    // prefill is the same space with more columns. ik dispatches over (heads × row
    // chunks) (iqk_mul_mat.cpp:637-676); the cell split is this engine's spelling.
    //
    // SAFETY (construction site): every participant computes cells inside its own
    // chunk and writes only cell `col * latent + j` of `out.data`; the chunks
    // partition the cell space (no aliasing), and the join publishes the writes.
    let out_ptr = crate::ops::SharedOut(out.data.as_mut_ptr());
    let latent = p.latent;
    let ne1 = q.ne1;
    // Only a profiled dispatch pays for the chunk collector; the cell loop
    // cannot fail, so an unprofiled one needs no error channel either.
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<profile::CallAcc>::new())
    } else {
        None
    };
    threads::pool().for_each_chunk(p.n_head * ne1 * latent, |cells| {
        let mut acc = profile::CallAcc::new();
        // Per-worker scratch: qdot's kernel writes a segment's cells here, then they
        // go to `out` via `SharedOut::write` (the `&Sync` capture); one latent-wide
        // run per column segment. Thread-local and recycled — every pool thread
        // survives many dispatches.
        let mut cellbuf = CELL_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if cellbuf.len() < latent {
            cellbuf.resize(latent, 0.0);
        }
        // Walk column-segment-wise: a chunk may start and end mid-column, so each
        // iteration takes the intersection of the remaining chunk and one column —
        // `whead`/`qcol` fetched once per segment.
        let mut c = cells.start;
        while c < cells.end {
            let col = c / latent;
            let j0 = c - col * latent;
            let j_end = j0 + (cells.end - c).min(latent - j0);
            let h = col / ne1;
            let whead = &wblocks[h * span..(h + 1) * span];
            let qcol = &qall[col * nblocks..(col + 1) * nblocks];
            let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
            // The segment's i32 dots run in qdot's AVX2 maddubs kernel — the Q6_K
            // qY sign fold (no +128 prepare, no compensation, no reachable
            // saturation); integer associativity makes the lane order free. The
            // argument and its gate live in crates/qdot.
            let seg = &mut cellbuf[..j_end - j0];
            qdot::q_nope2_cells(whead, qcol, j0, j_end, seg);
            for (jj, &v) in seg.iter().enumerate() {
                // SAFETY: cell `col * latent + j0 + jj` is inside this chunk's range — see the construction site above.
                unsafe { out_ptr.write(col * latent + j0 + jj, v) };
            }
            if let Some(t_dot) = t_dot {
                acc.add_dot(t_dot.elapsed().as_nanos() as u64);
            }
            // j_end is a row index INSIDE the column; the walk variable is absolute,
            // so the next segment starts at the column base + j_end.
            c = col * latent + j_end;
        }
        CELL_BUF.with(|c| *c.borrow_mut() = cellbuf);
        // Only a profiled chunk has anything to report.
        if let Some(collected) = &collected {
            collected.push(acc);
        }
    });

    // Post-join: fold the chunk accumulators so `record` fires once per call; the
    // fold is addition, no error precedence to keep (the shape check fired before
    // the dispatch; the cell loop cannot fail).
    let t_gather = if lvl >= 2 { Some(Instant::now()) } else { None };
    if let Some(collected) = collected {
        for acc in collected.into_vec() {
            pacc.add_acc(&acc);
        }
    }
    QALL.with(|c| *c.borrow_mut() = qall);
    if let Some(t_gather) = t_gather {
        pacc.add_gather(t_gather.elapsed().as_nanos() as u64);
    }
    if let Some(t_call) = t_call {
        // Shape statement for the derived weights consumed: `n_head · latent` rows
        // contracted over `nope`; tag 8 is ggml's Q8_0 — the row names what the
        // step reads (derived blocks), not what the file holds.
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

/// ik's vector `expf` (iqk_utils.h:160-206), transcribed op for op: a minimax polynomial in `b = x − n·ln2` rescaled by `2ⁿ` from `z`'s bits; the `|n| > 126` slow path underflows through the `s1·s2` split. Every `mul_add` is the reference's fused op — its last-ulp rounding is contract. `x = −inf` flushes to exactly `+0.0`, which drops masked keys and padded cache rows with no residue.
fn v_expf(x: f32) -> f32 {
    // The C hex float literals have no Rust spelling; these bit patterns are their f32 values, each verified to round-trip (float.fromhex → pack '<f' → back).
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

/// The QK dot in the fa4 gemm's own lane order (`mul_mat_Qx_Qy_MxN_fa4`, iqk_gemm_floats.cpp:256): per 8-element chunk, four fused adds land in a low partial (`8i+0..3`) and four in a high partial (`8i+4..7`); the dot is the single plain add of the two. `q` stays F32 (the FA node's q is never rounded to f16); every K element is the exact f32 of its f16 cache bits.
/// The no-AVX2 fallback and the gate's scalar leg; the AVX2 twin ([`kq_dot_simd`]) computes the same contraction in a different sum order.
pub fn kq_dot_fa4(q: &[f32], k: &[u16]) -> f32 {
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

/// The AVX2+FMA+F16C twin of [`kq_dot_fa4`]: same contraction and operands (`_mm256_cvtph_ps` converts f16→f32 without rounding, exactly [`half_to_f32`]; every product one fused multiply-add), summed in lane order — per 32-element panel, partials `P0..P3`, dot = `hsum((P0+P1)+(P2+P3))`; a different ordering of the same additions as the scalar twin: last-ulps, banded in `tests/attn.rs`.
/// `#[target_feature]` is not optional: without it the intrinsics lower to scalar emulation with no error.
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C, and `q.len() == k.len()` must be a multiple of 32 — [`kq_dot_simd`] checks both for direct calls, `flash_simd` at the flash site.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn kq_dot_fa4_avx2(q: &[f32], k: &[u16]) -> f32 {
    // SAFETY: the fn contract above — ISA from the caller's detection, lengths validated one step up; offsets stay in bounds by the `% 32 == 0` walk.
    unsafe {
        use std::arch::x86_64::*;
        let qp = q.as_ptr();
        let kp = k.as_ptr();
        let mut p0 = _mm256_setzero_ps();
        let mut p1 = _mm256_setzero_ps();
        let mut p2 = _mm256_setzero_ps();
        let mut p3 = _mm256_setzero_ps();
        for base in (0..q.len()).step_by(32) {
            // Four f16 octets convert in one op each — the exact conversion the scalar loop pays a soft function per element for.
            let k0 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(base) as *const __m128i));
            let k1 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(base + 8) as *const __m128i));
            let k2 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(base + 16) as *const __m128i));
            let k3 = _mm256_cvtph_ps(_mm_loadu_si128(kp.add(base + 24) as *const __m128i));
            p0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(base)), k0, p0);
            p1 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(base + 8)), k1, p1);
            p2 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(base + 16)), k2, p2);
            p3 = _mm256_fmadd_ps(_mm256_loadu_ps(qp.add(base + 24)), k3, p3);
        }
        let s = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
        // The documented lane tree: 128-bit halves first, then lane pairs.
        let c = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        let t = _mm_add_ps(c, _mm_movehdup_ps(c));
        _mm_cvtss_f32(_mm_add_ps(t, _mm_movehl_ps(t, t)))
    }
}

/// [`kq_dot_fa4_avx2`] behind the loud-failure wrapper the gate's path comparison calls (the `dot_row_avx2` pattern from `crates/qdot`): on a CPU without the ISA, panic — no quiet fall back.
pub fn kq_dot_simd(q: &[f32], k: &[u16]) -> f32 {
    assert!(
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"),
        "kq_dot_simd called on a CPU without AVX2+FMA+F16C"
    );
    assert_eq!(
        q.len(),
        k.len(),
        "kq_dot_simd: q and k must be the same row"
    );
    assert!(
        q.len().is_multiple_of(32),
        "kq_dot_simd: the 32-element panel kernel needs len % 32 == 0, got {}",
        q.len()
    );
    // SAFETY: the ISA was asserted just above and the lengths meet the panel contract.
    unsafe { kq_dot_fa4_avx2(q, k) }
}

/// `F16::reduce_add<32>` (iqk_fa_templates.h:54-206): `((v0+v1)+v2)+v3` elementwise over the four 8-lane registers, then `hsum_float_8`. Masked lanes hold exactly `0.0` and ride through the tree without changing any rounding.
fn lane_tree_sum(w: &[f32; 32]) -> f32 {
    let c = |l: usize| ((w[l] + w[8 + l]) + w[16 + l]) + w[24 + l];
    let t = [c(0) + c(4), c(1) + c(5), c(2) + c(6), c(3) + c(7)];
    (t[0] + t[2]) + (t[1] + t[3])
}

/// The flash-attention over the latent, replicating ik's FA templates as they run on
/// the box: `iqk_flash_attn_noalibi` → general prefill → `iqk_flash_attn_impl` (n_kv
/// 256, a multiple of 32) → `FlashAttn<576, 512, ·, 32>` with F16 K/V helpers
/// (`compute_helper`; sink/M/S null). Per (token, head) row, in the kernel's own order:
///
///   * KQ dot per key over `[q_rope(64) ; q_nope2(512)]` (the graph's concat order)
///     against the f16 cache row;
///   * `s = kq_scale·(kq + mask)` — the addend is exactly `+0.0` for allowed keys,
///     `−inf` otherwise;
///   * online M/S in 32-key blocks (`FlashMS`): block max, rescale-or-zero on M
///     bump, weights = [`v_expf`]`(s − M)`, S += [`lane_tree_sum`];
///   * V accumulation in an f32 FMA chain (`accumulate_qkv`), keys ascending;
///   * final row = `R · (1/S)`, one plain multiply per element.
///
/// The two inner loops take the AVX2+FMA+F16C twin [`flash_row_avx2`] when
/// [`flash_simd`] allows; its one numerical difference is the kq sum order, which
/// `tests/attn.rs` bands against [`flash_attn_latent_scalar`], the twin that stays
/// bit-exact on exact inputs.
///
/// Blocks with no allowed key are skipped outright: their weights are all exactly
/// `0.0` ([`v_expf`]) and `S += 0` / `fma(V, 0, R) = R` are exact no-ops — skipping
/// is bit-identical to executing them, and padded cache rows never enter for the same
/// reason. One divergence, outside the gated regime: the M-bump rescale uses glibc
/// `expf` in the reference and `f32::exp` here — reachable only above 32 allowed keys,
/// which a 6-token prefill cannot.
pub fn flash_attn_latent(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
) -> Tensor2 {
    flash_attn_latent_impl(
        q_rope,
        q_nope2,
        keys16,
        key_slots,
        q_slots,
        p,
        flash_simd(p),
    )
}

/// The forced-scalar twin of [`flash_attn_latent`], `pub` for the gates (the
/// `dot_row_scalar` pattern from `crates/qdot`): the no-AVX2 fallback AND the oracle
/// the AVX2 twin is banded against — ik's own (fa4) sum order, bit-identical on the
/// oracle's exact inputs (`tests/attn.rs`).
pub fn flash_attn_latent_scalar(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
) -> Tensor2 {
    flash_attn_latent_impl(q_rope, q_nope2, keys16, key_slots, q_slots, p, false)
}

/// Whether [`flash_attn_latent`]'s row kernel takes the AVX2+FMA+F16C twin: the CPU
/// must carry all three, and the shapes must meet the vector panels' contract —
/// `d_head` a multiple of 32, the latent tail a multiple of 8. A differently-shaped
/// MLA keeps the scalar transcription rather than quietly truncating a row (the
/// [`MlaParams::read`] fail-loudly policy).
///
/// `BLOOMERY_FLASH_SIMD=0` forces the scalar path for a whole process (read once) —
/// the gates' A/B lever: same binary, same oracle, only the kernel choice changes.
fn flash_simd(p: &MlaParams) -> bool {
    static FORCE_SCALAR: OnceLock<bool> = OnceLock::new();
    if *FORCE_SCALAR.get_or_init(|| std::env::var("BLOOMERY_FLASH_SIMD").is_ok_and(|v| v == "0")) {
        return false;
    }
    (p.rope_dims + p.latent).is_multiple_of(32)
        && p.latent.is_multiple_of(8)
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
}

/// The single owner of the row split both dispatch legs ride; `simd` is decided once
/// per call. The row twins share the whole skeleton and differ only in the two
/// vectorized inner loops (kq dot, V accumulation) — keep them in lockstep.
fn flash_attn_latent_impl(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
    simd: bool,
) -> Tensor2 {
    // Profiler hook: level-1 timer over the whole kernel. Shape statement: `rows` the
    // (token, head) query rows, `k` the keys·d_head dot work, `weight_bytes` the f16
    // KV rows read (V is the same rows' tail).
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    // Queries and keys are different counts the moment a KV cache exists: one decode
    // step is a single query against a whole prefix. Every index below that reads
    // `n_tokens` is a query index — `q_rope`/`q_nope2` are laid out by the batch.
    assert_eq!(
        keys16.len(),
        key_slots.len(),
        "every cached key row needs its slot"
    );
    let n_tokens = q_slots.len();
    let d_head = p.rope_dims + p.latent;
    let mut out = Tensor2::scratch(p.latent, n_tokens * p.n_head);
    // Every cached row must be `d_head` wide — the AVX2 twin's unchecked loads need
    // it up front; a short row is a caller bug to catch first.
    assert!(
        keys16.iter().all(|k| k.len() == d_head),
        "flash_attn_latent: KV rows must be rope+latent = {d_head} wide"
    );

    // The (token, head) query rows are fully independent — each walks the KV cache
    // and writes its own contiguous `latent`-wide slice of `out`, the same row-split
    // argument `matmul_q` rides.
    //
    // SAFETY (construction site): every participant computes rows inside its own
    // chunk and writes only cells `row * latent .. (row + 1) * latent` of `out.data`
    // — disjoint slices, no aliasing; the pool's join publishes the writes before
    // `out` is read. Either row twin writes the same cells (`tests/mt.rs`).
    let out_ptr = crate::ops::SharedOut(out.data.as_mut_ptr());
    let latent = p.latent;
    threads::pool().for_each_chunk(n_tokens * p.n_head, |rows| {
        // Per-worker scratch, one set per participant, sized exactly as the
        // serial version had them — recycled across dispatches. The exact-length
        // slices are load-bearing: the row kernels walk the whole `qrow` slice
        // against `d_head`-wide key rows.
        let mut qrow_buf = FLASH_QROW.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut r_buf = FLASH_R.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if qrow_buf.len() < d_head {
            qrow_buf.resize(d_head, 0.0);
        }
        if r_buf.len() < latent {
            r_buf.resize(latent, 0.0);
        }
        let qrow = &mut qrow_buf[..d_head];
        let r = &mut r_buf[..latent];
        let mut w = [0.0f32; 32];
        for row in rows {
            if simd {
                // SAFETY: `flash_simd` checked ISA and panel shapes; the row-width assert above and the scratch sizing complete the contract.
                unsafe {
                    flash_row_avx2(
                        q_rope, q_nope2, keys16, key_slots, q_slots, p, n_tokens, row, qrow, r,
                        &mut w, &out_ptr,
                    )
                };
            } else {
                flash_row_scalar(
                    q_rope, q_nope2, keys16, key_slots, q_slots, p, n_tokens, row, qrow, r, &mut w,
                    &out_ptr,
                );
            }
        }
        FLASH_QROW.with(|c| *c.borrow_mut() = qrow_buf);
        FLASH_R.with(|c| *c.borrow_mut() = r_buf);
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

/// The per-row body of the flash kernel, scalar transcription — the no-AVX2
/// fallback and the gates' oracle (see [`flash_attn_latent_scalar`]); the
/// AVX2 twin below differs ONLY in the two vectorized inner loops.
#[allow(clippy::too_many_arguments)]
fn flash_row_scalar(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
    n_tokens: usize,
    row: usize,
    qrow: &mut [f32],
    r: &mut [f32],
    w: &mut [f32; 32],
    out: &crate::ops::SharedOut,
) {
    let latent = p.latent;
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
            // TWIN: flash_row_avx2 — every edit outside the two marked
            // regions must be made in both. (This is the kq dot region; the
            // scalar form is the bit-exact oracle. The AVX2 twin also
            // prefetches the next KV row inside this region — a cache hint
            // touches no value, so it has no scalar counterpart.)
            let kq = kq_dot_fa4(qrow, &keys16[u]);
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
        s_sum += lane_tree_sum(w);
        match need {
            1 => r.iter_mut().for_each(|v| *v *= vms),
            2 => r.fill(0.0),
            _ => {}
        }
        for l in 0..32 {
            if w[l] == 0.0 {
                continue; // masked lane: fma(V, 0, R) == R exactly
            }
            // TWIN: flash_row_avx2 — every edit outside the two marked
            // regions must be made in both. (This is the V accumulation
            // region.)
            let krow = &keys16[blk + l];
            for d in 0..latent {
                r[d] = half_to_f32(krow[p.rope_dims + d]).mul_add(w[l], r[d]);
            }
        }
    }

    let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
    for (d, &v) in r.iter().enumerate() {
        // SAFETY: cell `row * latent + d` belongs to this row alone — see the construction site at the dispatch.
        unsafe { out.write(row * latent + d, s_inv * v) };
    }
}

/// The AVX2+FMA+F16C twin of [`flash_row_scalar`]: same skeleton, same softmax /
/// `v_expf` / online M/S scan / row split; the two inner loops vectorized:
///
///   * the kq dot — [`kq_dot_fa4_avx2`]: only the SUM ORDER changes (fa4
///     two-partial chain → lane groups); the twins' one numerical difference,
///     gated by the explicit reassociation band in `tests/attn.rs`;
///   * the V accumulation — eight `d` per FMA. The j (=d) axis reorder is
///     nothing: each output element's FMA chain still descends key order
///     exactly as the scalar wrote it, so given the same weights this stage is
///     bit-identical to the scalar twin.
///
/// `#[target_feature]` is not optional: without it the intrinsics lower to scalar
/// emulation with no error. The one split-out helper is the kq dot, at the scalar
/// path's own per-key granularity — nothing inside the contraction or the V loop
/// is split (finer splits round through memory).
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C, and the caller must hold the contract
/// `flash_simd` gated on: `d_head = rope_dims + latent` a multiple of 32,
/// `latent` a multiple of 8, every KV row `d_head` wide (asserted at the
/// dispatch), and the scratch slices sized `d_head` / `latent` / `32`.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn flash_row_avx2(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
    n_tokens: usize,
    row: usize,
    qrow: &mut [f32],
    r: &mut [f32],
    w: &mut [f32; 32],
    out: &crate::ops::SharedOut,
) {
    // SAFETY: the fn contract above — ISA, row width and scratch lengths all
    // come from the dispatch; offsets stay inside those bounds.
    unsafe {
        use std::arch::x86_64::*;
        let latent = p.latent;
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
                // TWIN: flash_row_scalar — every edit outside the two marked
                // regions must be made in both. (This is the kq dot region;
                // only the sum order may differ from the scalar oracle.)
                // Each key row is its own heap Vec, so the hardware
                // prefetcher restarts at every row boundary; pull the next
                // row's lines while this one is dotted. A prefetch is a
                // cache hint, never a value — no scalar twin of this.
                if let Some(next) = keys16.get(u + 1) {
                    let base = next.as_ptr().cast::<u8>();
                    let mut off = 0usize;
                    while off < next.len() * 2 {
                        // SAFETY: prefetch reads nothing; `off` stays inside
                        // the row's own `len() * 2` bytes.
                        _mm_prefetch::<_MM_HINT_T0>(base.add(off) as *const i8);
                        off += 64;
                    }
                }
                // SAFETY: plus the fn contract: `u` indexes keys16 in bounds
                // (len asserted at the dispatch) and its row is d_head wide.
                let kq = kq_dot_fa4_avx2(qrow, &keys16[u]);
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
            s_sum += lane_tree_sum(w);
            match need {
                1 => r.iter_mut().for_each(|v| *v *= vms),
                2 => r.fill(0.0),
                _ => {}
            }
            for l in 0..32 {
                if w[l] == 0.0 {
                    continue; // masked lane: fma(V, 0, R) == R exactly
                }
                // TWIN: flash_row_scalar — every edit outside the two marked
                // regions must be made in both. (This is the V accumulation
                // region; given the same weights it is bit-identical to the
                // scalar form — only the kq region may differ.)
                let krow = &keys16[blk + l];
                let wl = _mm256_set1_ps(w[l]);
                // The latent tail: eight f16 per cvtph, the j-axis FMA
                // `r[d..d+8] = fma(v, w, r[d..d+8])` — see the fn doc: reorders nothing.
                let vp = krow.as_ptr().add(p.rope_dims);
                let rp = r.as_mut_ptr();
                for d in (0..latent).step_by(8) {
                    // SAFETY: latent % 8 == 0 and r.len() == latent (the fn
                    // contract), so d+8 <= latent in both the key row and r.
                    let v = _mm256_cvtph_ps(_mm_loadu_si128(vp.add(d) as *const __m128i));
                    let acc = _mm256_loadu_ps(rp.add(d));
                    _mm256_storeu_ps(rp.add(d), _mm256_fmadd_ps(v, wl, acc));
                }
            }
        }

        let s_inv = if s_sum > 0.0 { 1.0 / s_sum } else { 0.0 };
        for (d, &v) in r.iter().enumerate() {
            // SAFETY: cell `row * latent + d` belongs to this row alone — see the construction site at the dispatch.
            out.write(row * latent + d, s_inv * v);
        }
    }
}

// ----------------------------------------------------------------- wv_b

/// `wv_b` stays Q3_K in the reference (only `wk_b` is requantized), so this is a
/// `matmul_q_batch` over synthetic views of `attn_kv_b`'s v-up rows — the views
/// `ggml_view_3d` takes, as file offsets. The gather is needed because
/// `kqv_compressed` interleaves heads (`t·n_head+h`) while each head's matmul
/// wants consecutive 512-wide rows; all heads share the view shape, so the layer
/// is one call — bit-identical by the batch primitive's argument (per-pair
/// inputs are exactly what per-head `matmul_q` would see).
pub fn wv_b_heads(
    gguf: &Gguf,
    wkb: &TensorInfo,
    kqv_compressed: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    let views = v_up_views(wkb, p);
    wv_b_heads_with(gguf, &views, kqv_compressed, p)
}

/// The per-head v_up views of `attn_kv_b` — the views `ggml_view_3d` takes, as
/// file offsets. One owner of the geometry: `Derived::new` builds these once per
/// block, and the direct-call path above builds them per call. Pure function of
/// the tensor info and the geometry — no error path.
pub(crate) fn v_up_views(wkb: &TensorInfo, p: &MlaParams) -> Vec<TensorInfo> {
    let row_bytes =
        wkb.ty.type_size().unwrap() as usize * (p.latent / wkb.ty.blck_size().unwrap() as usize);
    (0..p.n_head)
        .map(|h| TensorInfo {
            name: format!("{}.v_up.head{h}", wkb.name),
            dims: vec![p.latent as u64, p.v_head as u64],
            ty: wkb.ty,
            offset: wkb.offset + ((h * (p.nope + p.v_head) + p.nope) * row_bytes) as u64,
            nbytes: wkb.ty.type_size().unwrap()
                * (p.latent / wkb.ty.blck_size().unwrap() as usize) as u64
                * p.v_head as u64,
        })
        .collect()
}

/// The `wv_b` gather-matmul-scatter over prebuilt views — the step path, which
/// takes the views from [`Derived`](crate::derived::Derived).
pub fn wv_b_heads_with(
    gguf: &Gguf,
    views: &[TensorInfo],
    kqv_compressed: &Tensor2,
    p: &MlaParams,
) -> Result<Tensor2, ModelError> {
    // Profiler hook: SELF time — the whole call minus what its one hooked
    // `matmul_q_batch` child recorded (`profile::site_ns_total`): the gather, view
    // builds and scatter. Per-piece timers would measure the timer (a piece is a
    // 512-float copy).
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let mm_before = if lvl > 0 {
        profile::site_ns_total("matmul_q_batch")
    } else {
        0
    };
    // kqv_compressed interleaves heads as t·n_head+h; a column count not a multiple
    // of n_head would truncate n_tokens and read another head's tokens silently.
    assert!(
        kqv_compressed.ne1.is_multiple_of(p.n_head),
        "wv_b_heads: kqv_compressed has {} columns, not a multiple of n_head {}",
        kqv_compressed.ne1,
        p.n_head
    );
    let n_tokens = kqv_compressed.ne1 / p.n_head;
    let mut kqv_2d = Tensor2::scratch(p.n_head * p.v_head, n_tokens);

    let mut xhs: Vec<Tensor2> = (0..p.n_head)
        .map(|_| Tensor2::scratch(p.latent, n_tokens))
        .collect();
    for h in 0..p.n_head {
        for t in 0..n_tokens {
            let src = kqv_compressed.col(t * p.n_head + h);
            xhs[h].col_mut(t).copy_from_slice(src);
        }
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

// ------------------------------------------------------------ fused step

/// The step path: the whole per-head attention chain — `q_nope2` absorption,
/// flash over the latent, `wv_b` — as ONE pool dispatch over the (token,
/// head) rows. Between dispatches the calling thread works alone while every
/// worker idles, and the three stages are independent per row, so the two
/// barriers the chain paid bought nothing. The three stage functions above
/// stay `pub`: they are the oracle this path is gated against, bit for bit.
///
/// Bit identity with the chain `q_nope2_absorbed` → `flash_attn_latent` →
/// `wv_b_heads_with` (`hw_attn_heads_fused_bit_identical`): every
/// `quantize_act`, `q_nope2_cells`, flash-row and `quantize_col`/`dot_row`
/// call receives exactly the bytes it receives in the chain — `wv_b`'s
/// gathered input is a copy of the `kqv_compressed` column, and a copy
/// quantizes to the same bytes — so only WHICH thread makes each call
/// changes. The two index conventions are the chain's own: `q_nope2` columns
/// are head-major (`h·ne1 + t`), `kqv_compressed` rows and `kqv_2d` head
/// spans are token-major (`t·n_head + h`).
#[allow(clippy::too_many_arguments)]
pub fn attn_heads_fused(
    gguf: &Gguf,
    wblocks: &[Q8Block],
    q: &Tensor2,
    q_rope: &Tensor2,
    keys16: &[Vec<u16>],
    key_slots: &[Slot],
    q_slots: &[Slot],
    views: &[TensorInfo],
    p: &MlaParams,
) -> Result<(Tensor2, Tensor2, Tensor2), ModelError> {
    // Profiler hook: level-1 timer over the whole call, typeless
    // (`record_time`) — the call walks three differently-shaped weight reads
    // (derived Q8_0 blocks, F16 KV rows, Q3_K v_up views), and no single
    // rows/k/weight-bytes triple would be honest for all three. The wall is
    // the statement, the same convention `wv_b_heads`' self time used.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    // A wrong-length slice would read another head's weights silently.
    if wblocks.len() != p.n_head * span {
        return Err(ModelError::Shape {
            what: "attn_heads wblocks (n_head·latent·nope/32)",
            want_ne0: p.n_head * span,
            want_ne1: 0,
            got_ne0: wblocks.len(),
            got_ne1: 0,
        });
    }
    let ne1 = q_slots.len();
    assert_eq!(q.ne1, ne1, "one q column per query slot");
    assert_eq!(
        keys16.len(),
        key_slots.len(),
        "every cached key row needs its slot"
    );
    let d_head = p.rope_dims + p.latent;
    assert!(
        keys16.iter().all(|k| k.len() == d_head),
        "attn_heads_fused: KV rows must be rope+latent = {d_head} wide"
    );
    assert_eq!(
        views.len(),
        p.n_head,
        "attn_heads_fused: one v_up view per head"
    );

    let mut q_nope2 = Tensor2::scratch(p.latent, p.n_head * ne1);
    let mut kqv_compressed = Tensor2::scratch(p.latent, ne1 * p.n_head);
    let mut kqv_2d = Tensor2::scratch(p.n_head * p.v_head, ne1);
    // The row twin is decided once per call, exactly as `flash_attn_latent` does.
    let simd = flash_simd(p);

    // The (token, head) rows are fully independent: row (t, h) writes only
    // its own cells in all three outputs — `q_nope2` column `h·ne1 + t`,
    // `kqv_compressed` row `t·n_head + h`, `kqv_2d` column `t`'s span
    // `h·v_head..(h+1)·v_head` — and reads, inside the same iteration, only
    // cells that same row wrote (flash reads the `q_nope2` column stage (a)
    // produced, the `wv_b` matvec the `kqv_compressed` row flash produced) —
    // same thread, so the read is sequenced after the write.
    //
    // SAFETY (construction site): the pool chunks partition the row space,
    // each participant writes only its own rows' cells through these three
    // pointers (no aliasing between participants), and the join publishes
    // every write before any tensor is read. The stage-(b)/(c) reads go
    // through shared references into cells no other participant touches.
    let qn2_ptr = crate::ops::SharedOut(q_nope2.data.as_mut_ptr());
    let kc_ptr = crate::ops::SharedOut(kqv_compressed.data.as_mut_ptr());
    let kv2_ptr = crate::ops::SharedOut(kqv_2d.data.as_mut_ptr());
    // The matvec wants a real `&mut` span, so this one's base is dereferenced
    // in the closure (`kv2.0.add(..)`) — through this `&SharedOut` local, so
    // the closure captures the wrapper (Sync) and not the raw-pointer field
    // (not Sync); the method calls below get that for free from `&self`.
    let kv2 = &kv2_ptr;
    // The matvec is the one fallible stage; lowest-row-first precedence, the
    // `run_row_pool` error-channel pattern.
    let gate = crate::ops::ErrGate::new();
    let latent = p.latent;
    let v_head = p.v_head;
    let kv2_ne0 = kqv_2d.ne0;
    threads::pool().for_each_chunk(ne1 * p.n_head, |rows| {
        // Per-worker scratch, recycled across dispatches: the four
        // thread-locals the three stages use separately, taken once per
        // chunk and returned even when a row error breaks the loop early.
        // Every element is rewritten before it is read.
        let mut qcol = QALL.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut cellbuf = CELL_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut qrow_buf = FLASH_QROW.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut r_buf = FLASH_R.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if qcol.len() < nblocks {
            qcol.resize(
                nblocks,
                ActBlock {
                    d: 0.0,
                    q: [0i8; 32],
                },
            );
        }
        if cellbuf.len() < latent {
            cellbuf.resize(latent, 0.0);
        }
        if qrow_buf.len() < d_head {
            qrow_buf.resize(d_head, 0.0);
        }
        if r_buf.len() < latent {
            r_buf.resize(latent, 0.0);
        }
        let qrow = &mut qrow_buf[..d_head];
        let r = &mut r_buf[..latent];
        let seg = &mut cellbuf[..latent];
        let mut w = [0.0f32; 32];
        for row in rows {
            let t = row / p.n_head;
            let h = row % p.n_head;
            // (a) `q_nope2` column `h·ne1 + t`: the same `quantize_act`
            // blocks the chain quantizes on the caller, quantized by this
            // row's worker; `q_nope2_cells` computes each cell from its own
            // weight blocks and the column's blocks, so the whole-column
            // call and the chain's segmented pool call agree bit for bit.
            let qbase = t * q.ne0 + h * p.kq_head;
            let qnope = &q.data[qbase..qbase + p.nope];
            for (b, chk) in qnope.as_chunks::<32>().0.iter().enumerate() {
                qcol[b] = quantize_act(chk);
            }
            let whead = &wblocks[h * span..(h + 1) * span];
            qdot::q_nope2_cells(whead, &qcol[..nblocks], 0, latent, seg);
            for (j, &v) in seg.iter().enumerate() {
                // SAFETY: cell `(h·ne1 + t)·latent + j` is this row's own — see the construction site.
                unsafe { qn2_ptr.write((h * ne1 + t) * latent + j, v) };
            }
            // (b) the flash row, unchanged: it reads the column (a) just
            // wrote on this thread and writes `kqv_compressed` row `row`.
            if simd {
                // SAFETY: `flash_simd` checked ISA and panel shapes; the row-width assert above and the scratch sizing complete the contract.
                unsafe {
                    flash_row_avx2(
                        q_rope, &q_nope2, keys16, key_slots, q_slots, p, ne1, row, qrow, r, &mut w,
                        &kc_ptr,
                    )
                };
            } else {
                flash_row_scalar(
                    q_rope, &q_nope2, keys16, key_slots, q_slots, p, ne1, row, qrow, r, &mut w,
                    &kc_ptr,
                );
            }
            // (c) head h's `wv_b` leg: the same one-column `matmul_q` bytes
            // the chain's batch computes — its gathered input is a copy of
            // this exact column, and a copy quantizes to the same bytes.
            // SAFETY: `t·kv2_ne0 + h·v_head .. +v_head` is this row's own span of `kqv_2d` column `t` — see the construction site.
            let out_col = unsafe {
                std::slice::from_raw_parts_mut(kv2.0.add(t * kv2_ne0 + h * v_head), v_head)
            };
            if let Err(e) =
                crate::ops::matvec_q_local(gguf, &views[h], kqv_compressed.col(row), out_col)
            {
                gate.offer(row, e);
                break;
            }
        }
        QALL.with(|c| *c.borrow_mut() = qcol);
        CELL_BUF.with(|c| *c.borrow_mut() = cellbuf);
        FLASH_QROW.with(|c| *c.borrow_mut() = qrow_buf);
        FLASH_R.with(|c| *c.borrow_mut() = r_buf);
    });
    if let Some(e) = gate.take() {
        return Err(e);
    }
    if let Some(t_call) = t_call {
        profile::record_time("attn_heads", t_call.elapsed().as_nanos() as u64);
    }
    Ok((q_nope2, kqv_compressed, kqv_2d))
}
