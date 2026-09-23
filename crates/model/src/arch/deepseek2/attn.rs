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

use super::derived::Derived;
use super::names;
use crate::kv::{KvCache, KvRows};
use crate::ops::{matmul_q, matmul_q_batch, matmul_q_group, rms_norm};
use crate::profile;
use crate::{ModelError, Slot, Tensor2};
use gguf::QuantError;
use gguf::quant::{f32_to_f16_bits, half_to_f32};
use gguf::{Gguf, TensorInfo};
use std::cell::RefCell;
use std::ops::Range;
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
    //    Pushed before attention — a prefill attends to itself. One flat row-major
    //    buffer — the cache's own layout — filled column by column, since `kvr` is
    //    column-major (one column per token); the f16 roundings run in the same
    //    per-token order they always did.
    let mut new_rows: Vec<u16> = Vec::with_capacity(x.ne1 * kv_width);
    for t in 0..x.ne1 {
        new_rows.extend(kvr.col(t).iter().map(|&v| f32_to_f16_bits(v)));
    }
    cache.push(cache_block, range, &new_rows);
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
        // Every metadata key here is prefixed with the file's own
        // `general.architecture`; the prefix is never a literal.
        let key = |suffix: &str| gguf.arch_key(suffix);
        let need_u64 = |suffix: &str| -> Result<u64, ModelError> {
            gguf.arch_get_u64(suffix)
                .ok_or_else(|| ModelError::MissingTensor(format!("metadata {}", key(suffix))))
        };
        let need_f32 = |suffix: &str| -> Result<f32, ModelError> {
            gguf.arch_get_f32(suffix)
                .ok_or_else(|| ModelError::MissingTensor(format!("metadata {}", key(suffix))))
        };

        let n_head = need_u64("attention.head_count")? as usize;
        let kq_head = need_u64("attention.key_length")? as usize;
        let v_head = need_u64("attention.value_length")? as usize;
        let rope_dims = need_u64("rope.dimension_count")? as usize;
        let eps = need_f32("attention.layer_norm_rms_epsilon")?;

        let scaling = need_f32("rope.scaling.factor")?;
        let ctx_orig = need_u64("rope.scaling.original_context_length")? as u32;
        let log_mul = need_f32("rope.scaling.yarn_log_multiplier")?;
        match gguf.arch_get_str("rope.scaling.type") {
            Some("yarn") => {}
            other => {
                return Err(ModelError::MissingTensor(format!(
                    "{}: this MLA path is yarn-only, got {other:?}",
                    key("rope.scaling.type")
                )));
            }
        }
        let freq_base = need_f32("rope.freq_base")?;

        let wkb = find(gguf, &names::attn_kv_b(block))?;
        let wa = find(gguf, &names::attn_kv_a_mqa(block))?;
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

// --------------------------------------------------------------- q_nope2

// `Q8Block` lives in `gguf::quant` with the other GGUF block formats; re-exported so `Derived`'s storage, the gates' `assert_eq!` and the size assertion (`tests/derived.rs`) keep compiling; field-wise equality is equality of every byte.
pub use gguf::quant::Q8Block;

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
    /// The serial row path's per-segment partial accumulators, one latent
    /// range per segment, per worker.
    static FLASH_PARTS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// `q_nope2 = wk_b(Q8_0)ᵀ · q_nope` per head — the weight-absorption step.
///
/// `wk_b` is not in the file: the reference derives it at load (`llm_prepare_mla`,
/// llama.cpp:3229 — k-up rows of `attn_kv_b`, dequantized, transposed, Q8_0 blocks
/// along q_nope), a pure function of the weights, so it runs once in
/// [`Derived`](super::derived::Derived); this is the per-token half in staged form —
/// the step folds it into [`attn_heads_fused`], and `tests/attn.rs` holds the two
/// together.
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
            qall.extend(qrow.as_chunks::<32>().0.iter().map(|chk| quantize_act(chk)));
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
        // contracted over `nope`; the type names what the step reads (derived
        // Q8_0 blocks), not what the file holds.
        let w_rows = (p.n_head * p.latent) as u64;
        profile::record(
            "q_nope2_absorbed",
            gguf::GgmlType::Q8_0,
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
/// Split-K: a row's visible keys ([`visible_end`]) are cut into
/// [`FLASH_SEGMENTS`] fixed segments of whole 32-key blocks ([`flash_segment`]);
/// each runs the scan above on its own and leaves `(m, S, R)`, and
/// [`combine_segments`] merges them in a fixed order. A row of ≤ 32 keys has one
/// segment and keeps the single-pass bits; a longer one sums in a different order
/// than ik's (whose split depends on its thread count) — `tests/attn.rs` bands it
/// against the single-pass plan.
///
/// The two inner loops take the AVX2+FMA+F16C twin [`flash_seg_avx2`] when
/// [`flash_simd`] allows; its one numerical difference is the kq sum order, which
/// `tests/attn.rs` bands against [`flash_attn_latent_scalar`], the twin that stays
/// bit-exact on exact inputs.
///
/// Blocks with no allowed key are skipped outright: their weights are all exactly
/// `0.0` ([`v_expf`]) and `S += 0` / `fma(V, 0, R) = R` are exact no-ops — skipping
/// is bit-identical to executing them, and padded cache rows never enter for the same
/// reason. One divergence from the reference: the M-bump rescale inside a segment
/// uses `f32::exp` where ik uses glibc `expf` — reachable only in a segment holding
/// more than one block with allowed keys.
pub fn flash_attn_latent(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: KvRows<'_>,
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
        flash_segments(),
    )
}

/// The forced-scalar twin of [`flash_attn_latent`], `pub` for the gates (the
/// `dot_row_scalar` pattern from `crates/qdot`): the no-AVX2 fallback AND the oracle
/// the AVX2 twin is banded against — ik's own (fa4) sum order, bit-identical on the
/// oracle's exact inputs (`tests/attn.rs`).
pub fn flash_attn_latent_scalar(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: KvRows<'_>,
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
        false,
        flash_segments(),
    )
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

/// How many key rows ahead the AVX2 flash twin prefetches on a decode row;
/// 0 is off. `BLOOMERY_KV_PREFETCH_ROWS=d` sets the distance (default 16: 1
/// leaves the stream short of what one core keeps in flight, 8 reaches it
/// and 16 reads a little better still on the deep rows) and
/// `BLOOMERY_KV_PREFETCH=0` turns the hint off whatever the distance says —
/// same binary, same bytes: only the cache hint changes. Read once; a value
/// that does not parse panics rather than falling back.
fn kv_prefetch_rows_env() -> usize {
    static S: OnceLock<usize> = OnceLock::new();
    *S.get_or_init(|| {
        if std::env::var("BLOOMERY_KV_PREFETCH").is_ok_and(|v| v == "0") {
            return 0;
        }
        match std::env::var("BLOOMERY_KV_PREFETCH_ROWS") {
            Ok(v) => v.parse().unwrap_or_else(|_| {
                panic!("BLOOMERY_KV_PREFETCH_ROWS={v:?}: expected a row count (0 = off)")
            }),
            Err(_) => 16,
        }
    })
}

/// "No override": the distance follows the environment.
const KV_PREFETCH_FOLLOW_ENV: usize = usize::MAX;

/// The in-process override the prefetch-lever tests drive: the env vars are
/// read once per process, so this is how one binary exercises every arm.
static KV_PREFETCH_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(KV_PREFETCH_FOLLOW_ENV);

/// Test override for the KV prefetch distance: `Some(d)` forces `d` rows
/// ahead (0 = off), `None` follows `BLOOMERY_KV_PREFETCH_ROWS` /
/// `BLOOMERY_KV_PREFETCH`.
#[doc(hidden)]
pub fn set_kv_prefetch_rows(rows: Option<usize>) {
    KV_PREFETCH_OVERRIDE.store(
        rows.unwrap_or(KV_PREFETCH_FOLLOW_ENV),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// On/off form of [`set_kv_prefetch_rows`]: `Some(true)` is one row ahead,
/// `Some(false)` is off, `None` follows the environment.
#[doc(hidden)]
pub fn set_kv_prefetch(mode: Option<bool>) {
    set_kv_prefetch_rows(mode.map(usize::from));
}

/// What the last AVX2 flash dispatch decided, stored as distance + 1:
/// 0 = no dispatch yet, 1 = no prefetch, `d + 1` = `d` rows ahead. The lever
/// tests read it; a lever whose effect is only "the gate still passes"
/// proves nothing about the branch.
static LAST_KV_PREFETCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The observable behind [`set_kv_prefetch_rows`]: the distance the last
/// AVX2 flash dispatch (any `n_tokens`) resolved to — `Some(0)` if it did
/// not prefetch — or `None` before any ran.
#[doc(hidden)]
pub fn last_kv_prefetch_rows() -> Option<usize> {
    LAST_KV_PREFETCH
        .load(std::sync::atomic::Ordering::Relaxed)
        .checked_sub(1)
}

/// On/off form of [`last_kv_prefetch_rows`].
#[doc(hidden)]
pub fn last_kv_prefetch() -> Option<bool> {
    last_kv_prefetch_rows().map(|d| d > 0)
}

fn kv_prefetch_rows() -> usize {
    match KV_PREFETCH_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        KV_PREFETCH_FOLLOW_ENV => kv_prefetch_rows_env(),
        d => d,
    }
}

/// Keys per flash block: the online-softmax step of the row kernels (ik's
/// `k_step`). Segment boundaries fall on multiples of it.
const FLASH_BLOCK: usize = 32;

/// How many fixed segments one query row's keys are cut into (split-K), and
/// the most [`flash_segments`] admits. Never the thread count: the plan — and
/// with it every bit of the output — depends on the row's visible key count
/// alone.
const FLASH_SEGMENTS: usize = 32;

/// A decode row with at most this many 32-key blocks stays on the
/// one-dispatch shape ([`heads_per_row`]); above it the split-K shape
/// ([`heads_split_k`]) pays for its two extra dispatches.
const SPLITK_MIN_BLOCKS: usize = 16;

/// The segment count the flash plan runs at: [`FLASH_SEGMENTS`] unless
/// `BLOOMERY_FLASH_SEGMENTS=n` (1 to 32; read once; any other value panics) —
/// the same-binary lever for the split: 1 is the single-pass online softmax,
/// the plan the split is banded against (`tests/attn.rs`).
fn flash_segments() -> usize {
    static S: OnceLock<usize> = OnceLock::new();
    *S.get_or_init(|| match std::env::var("BLOOMERY_FLASH_SEGMENTS") {
        Err(_) => FLASH_SEGMENTS,
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) if (1..=FLASH_SEGMENTS).contains(&n) => n,
            _ => panic!("BLOOMERY_FLASH_SEGMENTS must be 1 to {FLASH_SEGMENTS}, got {v:?}"),
        },
    })
}

/// One past the last key the query in `q` may attend to, 0 if it sees none.
/// Every key at or past it is masked, so walking `0..visible_end` instead of
/// the whole cache changes no value; the plan is cut over this length, not
/// the cache length, so a prefill row and the decode step at the same
/// position cut the same segments (the KV cache changes nothing). A decode
/// query is the cache's last row, so the scan stops at once.
fn visible_end(key_slots: &[Slot], q: &Slot) -> usize {
    key_slots
        .iter()
        .rposition(|k| k.seq == q.seq && k.pos <= q.pos)
        .map_or(0, |u| u + 1)
}

/// The number of non-empty segments of an `n_vis`-key row: one per 32-key
/// block up to `n_seg`. The non-empty ones are always the first
/// (`threads::chunk_bounds` hands the extra blocks to the lowest indices).
fn flash_segments_used(n_vis: usize, n_seg: usize) -> usize {
    n_vis.div_ceil(FLASH_BLOCK).min(n_seg)
}

/// Segment `s` of `n_seg` over `0..n_vis`: whole 32-key blocks, split by
/// `threads::chunk_bounds` over the block count, the last one clipped to
/// `n_vis`. A pure function of `(n_vis, n_seg, s)`.
fn flash_segment(n_vis: usize, n_seg: usize, s: usize) -> Range<usize> {
    let (b0, b1) = threads::chunk_bounds(n_vis.div_ceil(FLASH_BLOCK), n_seg, s);
    b0 * FLASH_BLOCK..(b1 * FLASH_BLOCK).min(n_vis)
}

/// The fixed-order merge of a row's segment partials into `r`, returning `S`:
/// `M* = max m_s`; for `s` ascending, `α = v_expf(m_s − M*)`, `S = fma(α,
/// S_s, S)`, `r[d] = fma(α, R_s[d], r[d])`. A segment with `m_s = −inf` saw no
/// allowed key and is skipped (its partial is never read — it may hold
/// anything). Segment `s`'s `R` range is `parts[s·stride + off ..][..r.len()]`.
///
/// One segment merges to itself bit for bit: `α = v_expf(0) = 1`, `fma(1, x,
/// +0) = x`, and no partial is ever `−0` (every chain starts at `+0` and
/// round-to-nearest never sums to `−0`), so a row of ≤ 32 keys keeps the
/// single-pass kernel's exact bits. Each element's chain is its own, so a
/// latent range merges to the bits the whole row has there.
fn combine_segments(
    ms: &[f32],
    ss: &[f32],
    parts: &[f32],
    stride: usize,
    off: usize,
    r: &mut [f32],
) -> f32 {
    let m_star = ms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    r.fill(0.0);
    let mut s_tot = 0.0f32;
    if m_star == f32::NEG_INFINITY {
        return s_tot;
    }
    for (s, (&m, &s_s)) in ms.iter().zip(ss).enumerate() {
        if m == f32::NEG_INFINITY {
            continue;
        }
        let a = v_expf(m - m_star);
        s_tot = a.mul_add(s_s, s_tot);
        let part = &parts[s * stride + off..][..r.len()];
        for (rd, &x) in r.iter_mut().zip(part) {
            *rd = a.mul_add(x, *rd);
        }
    }
    s_tot
}

/// The single owner of the row split both dispatch legs ride; `simd` is decided once
/// per call. The row twins share the whole skeleton and differ only in the two
/// vectorized inner loops (kq dot, V accumulation) — keep them in lockstep.
#[allow(
    clippy::too_many_arguments,
    reason = "the chain's inputs plus the two plan levers"
)]
fn flash_attn_latent_impl(
    q_rope: &Tensor2,
    q_nope2: &Tensor2,
    keys16: KvRows<'_>,
    key_slots: &[Slot],
    q_slots: &[Slot],
    p: &MlaParams,
    simd: bool,
    n_seg: usize,
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
    // Every cached row is `width` wide by construction (one flat buffer, fixed
    // stride) — the AVX2 twin's unchecked loads need the stride to BE `d_head`,
    // which a mismatched cache width would silently violate.
    assert!(
        keys16.width() == d_head,
        "flash_attn_latent: KV rows must be rope+latent = {d_head} wide, cache width is {}",
        keys16.width()
    );
    let ahead = kv_prefetch_for(n_tokens, simd);

    // The (token, head) query rows are fully independent — each walks the KV cache
    // segment by segment, merges its partials and writes its own contiguous
    // `latent`-wide slice of `out`, the same row-split argument `matmul_q` rides.
    //
    // SAFETY (construction site): every participant computes rows inside its own
    // chunk and writes only cells `row * latent .. (row + 1) * latent` of `out.data`
    // — disjoint slices, no aliasing; the pool's join publishes the writes before
    // `out` is read. Either row twin writes the same cells (`tests/mt.rs`).
    let out_ptr = crate::ops::SharedOut(out.data.as_mut_ptr());
    let latent = p.latent;
    threads::pool().for_each_chunk(n_tokens * p.n_head, |rows| {
        let mut sc = RowScratch::take(d_head, latent, latent, n_seg);
        let mut vis = (usize::MAX, 0usize);
        for row in rows {
            let t = row / p.n_head;
            let h = row % p.n_head;
            if vis.0 != t {
                vis = (t, visible_end(key_slots, &q_slots[t]));
            }
            let qn2 = q_nope2.col(h * n_tokens + t);
            let rk = RowKeys {
                keys16,
                key_slots,
                q_slot: &q_slots[t],
                n_vis: vis.1,
            };
            flash_row(
                simd,
                ahead,
                q_rope.col(t * p.n_head + h),
                qn2,
                &rk,
                p,
                n_seg,
                0,
                &mut sc,
                row * latent,
                &out_ptr,
            );
        }
        sc.give_back();
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

/// The prefetch distance a flash dispatch of `n_tokens` queries runs at, and
/// its record for [`last_kv_prefetch_rows`]: decode rows only (one query) and
/// only the AVX2 twin prefetches. Decided once per dispatch, not per row — the
/// lever is a process-wide constant, and a per-row store bounced one shared
/// line between every worker.
fn kv_prefetch_for(n_tokens: usize, simd: bool) -> usize {
    if !simd {
        return 0;
    }
    let ahead = if n_tokens == 1 { kv_prefetch_rows() } else { 0 };
    LAST_KV_PREFETCH.store(ahead + 1, std::sync::atomic::Ordering::Relaxed);
    ahead
}

/// What one query row reads besides its own q: the cache, its slots, the
/// query's slot and the row's plan length ([`visible_end`]).
struct RowKeys<'a> {
    keys16: KvRows<'a>,
    key_slots: &'a [Slot],
    q_slot: &'a Slot,
    n_vis: usize,
}

/// A worker's recycled scratch for the serial row path: the FA query row, the
/// merged accumulator, and one partial accumulator per segment. Taken from
/// the thread-locals once per chunk and handed back after it; every element
/// is rewritten before it is read.
struct RowScratch {
    qrow: Vec<f32>,
    r: Vec<f32>,
    parts: Vec<f32>,
    d_head: usize,
    d_len: usize,
    w: [f32; 32],
}

impl RowScratch {
    fn take(d_head: usize, latent: usize, d_len: usize, n_seg: usize) -> Self {
        let mut qrow = FLASH_QROW.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut r = FLASH_R.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let mut parts = FLASH_PARTS.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if qrow.len() < d_head {
            qrow.resize(d_head, 0.0);
        }
        if r.len() < latent {
            r.resize(latent, 0.0);
        }
        if parts.len() < n_seg * d_len {
            parts.resize(n_seg * d_len, 0.0);
        }
        RowScratch {
            qrow,
            r,
            parts,
            d_head,
            d_len,
            w: [0.0; 32],
        }
    }

    fn give_back(self) {
        FLASH_QROW.with(|c| *c.borrow_mut() = self.qrow);
        FLASH_R.with(|c| *c.borrow_mut() = self.r);
        FLASH_PARTS.with(|c| *c.borrow_mut() = self.parts);
    }
}

/// The FA row `[q_rope ; q_nope2]`, F32, never rounded.
fn fill_qrow(qrow: &mut [f32], q_rope_col: &[f32], qn2: &[f32]) {
    let (rope, nope) = qrow.split_at_mut(q_rope_col.len());
    rope.copy_from_slice(q_rope_col);
    nope.copy_from_slice(qn2);
}

/// One query row of the flash kernel, serially: every non-empty segment of
/// the row's plan through the segment twin, then [`combine_segments`], then
/// `R · (1/S)` into `out[out_base + d_off ..][..d_len]`. `d_off`/`d_len`
/// (`sc.d_len`) pick a latent range; every step on `R` is per element, so a
/// range carries the bits the whole latent has there. The split-K dispatch
/// in [`attn_heads_split`] computes the same segments on different threads
/// and the same merge, so it writes the same bits.
#[allow(
    clippy::too_many_arguments,
    reason = "a row's inputs, its plan and its output cells"
)]
fn flash_row(
    simd: bool,
    ahead: usize,
    q_rope_col: &[f32],
    qn2: &[f32],
    rk: &RowKeys<'_>,
    p: &MlaParams,
    n_seg: usize,
    d_off: usize,
    sc: &mut RowScratch,
    out_base: usize,
    out: &crate::ops::SharedOut,
) {
    let qrow = &mut sc.qrow[..sc.d_head];
    fill_qrow(qrow, q_rope_col, qn2);
    let d_len = sc.d_len;
    let used = flash_segments_used(rk.n_vis, n_seg);
    let mut ms = [f32::NEG_INFINITY; FLASH_SEGMENTS];
    let mut ss = [0.0f32; FLASH_SEGMENTS];
    // `flash_segments` caps the plan at the arrays' length.
    debug_assert!(used <= FLASH_SEGMENTS);
    for s in 0..used {
        let keys = flash_segment(rk.n_vis, n_seg, s);
        let r_s = &mut sc.parts[s * d_len..(s + 1) * d_len];
        let (m, s_sum) = flash_segment_row(simd, ahead, qrow, rk, p, keys, d_off, r_s, &mut sc.w);
        ms[s] = m;
        ss[s] = s_sum;
    }
    let r = &mut sc.r[..d_len];
    let s_tot = combine_segments(&ms[..used], &ss[..used], &sc.parts, d_len, 0, r);
    let s_inv = if s_tot > 0.0 { 1.0 / s_tot } else { 0.0 };
    for (d, &v) in r.iter().enumerate() {
        // SAFETY: cell `out_base + d_off + d` belongs to this row's latent range alone — see the construction site at the dispatch.
        unsafe { out.write(out_base + d_off + d, s_inv * v) };
    }
}

/// One segment of one row through the row twin `simd` selects: `R` for the
/// latent range `d_off..d_off + r.len()` into `r`, returning `(m, S)`.
#[allow(
    clippy::too_many_arguments,
    reason = "the segment kernel's operands plus the twin choice"
)]
fn flash_segment_row(
    simd: bool,
    ahead: usize,
    qrow: &[f32],
    rk: &RowKeys<'_>,
    p: &MlaParams,
    keys: Range<usize>,
    d_off: usize,
    r: &mut [f32],
    w: &mut [f32; 32],
) -> (f32, f32) {
    if simd {
        // SAFETY: `simd` is `flash_simd`'s verdict (ISA and panel shapes); every
        // dispatch asserts `keys16.width() == d_head`, sizes `qrow` to `d_head`,
        // and keeps `d_off + r.len() <= latent` with `r.len() % 8 == 0`.
        unsafe { flash_seg_avx2(qrow, rk, p, keys, ahead, d_off, r, w) }
    } else {
        flash_seg_scalar(qrow, rk, p, keys, d_off, r, w)
    }
}

/// Scalar twin: `r[d] = fma(V[blk+l][v_off + d], w[l], r[d])` for lanes `l`
/// ascending, `d` over `r` — the V stage of one 32-key block, split out at
/// block granularity (the kq dot is the per-key split). `v_off` is where `r`'s
/// slice of the row starts: `rope_dims` for the whole latent, `rope_dims +
/// d_off` for one latent range of a split row. Every element's chain is its
/// own, so a range computes exactly the bits the whole latent does there. Lanes with
/// `w[l] == 0` are skipped: the row kernel writes exactly `0.0` there (causally
/// masked or past the cache end), and `fma(V, 0, R) == R` exactly. `pub` +
/// `#[doc(hidden)]` so the twin-pair gate in `tests/attn.rs` can call both
/// halves (the `dot_row_scalar` pattern).
// TWIN: flash_v_accum_avx2 — same FMAs in the same per-element order; every
// edit to the arithmetic or the lane semantics must be made in both.
#[doc(hidden)]
pub fn flash_v_accum_scalar(
    w: &[f32; 32],
    keys16: KvRows<'_>,
    blk: usize,
    v_off: usize,
    r: &mut [f32],
) {
    for (l, &wl) in w.iter().enumerate() {
        if wl == 0.0 {
            continue; // masked lane: fma(V, 0, R) == R exactly
        }
        let krow = keys16.row(blk + l);
        for (rd, &v) in r.iter_mut().zip(&krow[v_off..]) {
            *rd = half_to_f32(v).mul_add(wl, *rd);
        }
    }
}

/// The AVX2+FMA+F16C twin of [`flash_v_accum_scalar`], register-blocked:
/// d-outer in tiles of `T` ymm vectors, key-inner over the block's active
/// lanes — the `w[l] != 0` lanes, collected into a stack list once per call
/// (a masked lane contributes nothing, exactly the scalar twin's `continue`).
/// Per tile the `T` accumulators load from `r` once, every active lane FMAs
/// its `T` V vectors in (weight broadcast by `_mm256_set1_ps`), and the
/// accumulators store once — `r` crosses the lane walk once per tile instead
/// of once per key. The numerical contract: each element `d`'s FMA chain
/// applies the active lanes in ascending lane order, one FMA per lane, on the
/// running element, so the tile nesting reorders nothing and the output bits
/// equal the scalar twin's. A latent not a multiple of the tile takes a
/// single-vector tail with the same per-element chain.
///
/// `#[target_feature]` is not optional: without it the intrinsics lower to
/// scalar emulation with no error.
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C; `v_off + r.len() <= keys16.width()`;
/// `r.len()` a multiple of 8; `blk + l < keys16.len()` for every lane with
/// `w[l] != 0.0` — inactive lanes are never read.
// TWIN: flash_v_accum_scalar — same FMAs in the same per-element order; every
// edit to the arithmetic or the lane semantics must be made in both.
#[doc(hidden)]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn flash_v_accum_avx2(
    w: &[f32; 32],
    keys16: KvRows<'_>,
    blk: usize,
    v_off: usize,
    r: &mut [f32],
) {
    // SAFETY: the fn contract above — the caller proves the ISA and the row
    // width, and only lanes the list below admits are ever dereferenced.
    unsafe {
        use std::arch::x86_64::*;
        // 8 ymm per tile: 8 accumulators + one V vector + the broadcast
        // weight leaves the 16-register ymm file headroom for address math.
        const T: usize = 8;

        // The active lanes and their V-row tails, resolved once per call: the
        // tile walk below visits each lane once per tile, and a lane's row
        // address never moves inside the call. Membership is `w[l] == 0.0` —
        // the predicate the scalar twin's `continue` tests — so the lane set
        // is identical whatever `w` holds. Stack storage only: no allocation.
        let mut tails = [std::ptr::null::<u16>(); 32];
        let mut lanes = [0u8; 32];
        let mut n_lanes = 0usize;
        for (l, &wl) in w.iter().enumerate() {
            if wl == 0.0 {
                continue; // masked lane: fma(V, 0, R) == R exactly
            }
            // SAFETY: the fn contract keeps `blk + l` inside the cache, and
            // `v_off + r.len() <= width`, so the tail pointer stays inside
            // the row.
            tails[n_lanes] = keys16.row(blk + l).as_ptr().add(v_off);
            lanes[n_lanes] = l as u8;
            n_lanes += 1;
        }

        let latent = r.len();
        let rp = r.as_mut_ptr();
        let full = latent / (T * 8) * (T * 8);
        for d0 in (0..full).step_by(T * 8) {
            let mut acc = [_mm256_setzero_ps(); T];
            for (j, a) in acc.iter_mut().enumerate() {
                // SAFETY: `d0 + j*8 + 8 <= full <= latent = r.len()` — the
                // accumulator load stays inside `r`.
                *a = _mm256_loadu_ps(rp.add(d0 + j * 8));
            }
            for (&l, tp) in lanes[..n_lanes].iter().zip(tails[..n_lanes].iter()) {
                let wl = _mm256_set1_ps(w[l as usize]);
                let vp = tp.add(d0);
                for (j, a) in acc.iter_mut().enumerate() {
                    // SAFETY: row `blk + l` is at least `v_off + latent`
                    // wide, so the tile's eight f16 octets at `v_off + d0 +
                    // j*8` lie inside it.
                    let v = _mm256_cvtph_ps(_mm_loadu_si128(vp.add(j * 8) as *const __m128i));
                    *a = _mm256_fmadd_ps(v, wl, *a);
                }
            }
            for (j, &a) in acc.iter().enumerate() {
                // SAFETY: the same span the load above took — inside `r`.
                _mm256_storeu_ps(rp.add(d0 + j * 8), a);
            }
        }
        // The tail: latents not a multiple of the tile, one vector at a
        // time, lanes inner — each element's chain still descends the active
        // lanes in ascending order.
        for d in (full..latent).step_by(8) {
            // SAFETY: `d + 8 <= latent = r.len()` — inside `r`.
            let mut acc = _mm256_loadu_ps(rp.add(d));
            for (&l, tp) in lanes[..n_lanes].iter().zip(tails[..n_lanes].iter()) {
                let wl = _mm256_set1_ps(w[l as usize]);
                // SAFETY: the row is at least `v_off + latent` wide and
                // `d + 8 <= latent`, so the octet at `v_off + d` lies inside it.
                let v = _mm256_cvtph_ps(_mm_loadu_si128(tp.add(d) as *const __m128i));
                acc = _mm256_fmadd_ps(v, wl, acc);
            }
            // SAFETY: the same span the load above took — inside `r`.
            _mm256_storeu_ps(rp.add(d), acc);
        }
    }
}

/// One segment of one query row, scalar transcription — the no-AVX2 fallback
/// and the gates' oracle (see [`flash_attn_latent_scalar`]); the AVX2 twin
/// below differs ONLY in the two vectorized inner loops.
///
/// `qrow` is the row's `[q_rope ; q_nope2]`. The keys are `keys` (whole
/// 32-key blocks from a block boundary, the last one possibly short), walked
/// with the online M/S scan exactly as a whole row was: the scores and the
/// softmax always run over the whole `d_head` row; `r` holds the latent range
/// `d_off..d_off + r.len()` and only that range is accumulated. Returns the
/// segment's `(m, S)`; `r` is left unnormalized for [`combine_segments`].
#[allow(
    clippy::too_many_arguments,
    reason = "the segment kernel's operands, mirrored by its AVX2 twin"
)]
fn flash_seg_scalar(
    qrow: &[f32],
    rk: &RowKeys<'_>,
    p: &MlaParams,
    keys: Range<usize>,
    d_off: usize,
    r: &mut [f32],
    w: &mut [f32; 32],
) -> (f32, f32) {
    let (keys16, key_slots, q) = (rk.keys16, rk.key_slots, rk.q_slot);
    let mut m = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    r.fill(0.0);
    for blk in keys.clone().step_by(FLASH_BLOCK) {
        let mut s = [f32::NEG_INFINITY; 32];
        let mut smax = f32::NEG_INFINITY;
        for (l, sl) in s.iter_mut().enumerate() {
            let u = blk + l;
            if u >= keys.end {
                break; // past the segment: weight exactly 0
            }
            let su = &key_slots[u];
            if su.seq != q.seq || su.pos > q.pos {
                continue; // the -inf half of the causal mask
            }
            // TWIN: flash_seg_avx2 — every edit outside the two marked
            // regions must be made in both. (This is the kq dot region; the
            // scalar form is the bit-exact oracle. The AVX2 twin also
            // prefetches KV rows inside this region — a cache hint touches
            // no value, so it has no scalar counterpart.)
            let kq = kq_dot_fa4(qrow, keys16.row(u));
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
        // TWIN: flash_seg_avx2 — every edit outside the two marked regions
        // must be made in both. (This is the V accumulation region; the
        // arithmetic lives in the split-out twin pair above.)
        flash_v_accum_scalar(w, keys16, blk, p.rope_dims + d_off, r);
    }
    (m, s_sum)
}

/// The AVX2+FMA+F16C twin of [`flash_seg_scalar`]: same skeleton, same softmax /
/// `v_expf` / online M/S scan / latent range; the two inner loops vectorized:
///
///   * the kq dot — [`kq_dot_fa4_avx2`]: only the SUM ORDER changes (fa4
///     two-partial chain → lane groups); the twins' one numerical difference,
///     gated by the explicit reassociation band in `tests/attn.rs`;
///   * the V accumulation — [`flash_v_accum_avx2`], register-blocked: the
///     latent runs d-outer in tiles of 8 ymm whose accumulators stay in
///     registers while the block's active lanes walk inner, so `r` is read
///     and written once per tile instead of once per key. The axis reorder
///     is nothing: each output element's FMA chain still descends key order
///     exactly as the scalar wrote it, so given the same weights this stage
///     is bit-identical to the scalar twin.
///
/// `ahead` is the KV prefetch distance ([`kv_prefetch_for`]; 0 = none): the
/// first `ahead` rows of the segment are hinted before the walk and row
/// `u + ahead` while row `u` is dotted, never past the segment's end — the
/// rows after it are another segment's.
///
/// `#[target_feature]` is not optional: without it the intrinsics lower to scalar
/// emulation with no error. The split-out helpers are the kq dot, at the scalar
/// path's own per-key granularity, and the V accumulation, at block granularity
/// — the kq contraction stays whole (finer splits round through memory); the V
/// stage tiles the other axis to keep its accumulators in registers.
///
/// # Safety
/// The CPU must support AVX2+FMA+F16C, and the caller must hold the contract
/// `flash_simd` gated on: `d_head = rope_dims + latent` a multiple of 32,
/// `latent` a multiple of 8, `keys16.width() == d_head` (asserted at the
/// dispatch) — the unchecked row loads index through `keys16.row(u)`, whose
/// checked bounds and fixed stride are the whole guarantee — `qrow` sized
/// `d_head`, `keys.end <= keys16.len()`, and `r` a latent range: `d_off +
/// r.len() <= latent`, `r.len()` a multiple of 8.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
#[allow(
    clippy::too_many_arguments,
    reason = "the segment kernel's operands, mirrored by its scalar twin"
)]
unsafe fn flash_seg_avx2(
    qrow: &[f32],
    rk: &RowKeys<'_>,
    p: &MlaParams,
    keys: Range<usize>,
    ahead: usize,
    d_off: usize,
    r: &mut [f32],
    w: &mut [f32; 32],
) -> (f32, f32) {
    // SAFETY: the fn contract above — ISA, row width and scratch lengths all
    // come from the dispatch; offsets stay inside those bounds.
    unsafe {
        use std::arch::x86_64::*;
        let (keys16, key_slots, q) = (rk.keys16, rk.key_slots, rk.q_slot);
        let mut m = f32::NEG_INFINITY;
        let mut s_sum = 0.0f32;
        r.fill(0.0);
        // Hint one KV row's lines — a cache hint, never a value.
        let hint = |u: usize| {
            let next = keys16.row(u);
            let base = next.as_ptr().cast::<u8>();
            let mut off = 0usize;
            while off < next.len() * 2 {
                // SAFETY: prefetch reads nothing; `off` stays inside the
                // row's own `len() * 2` bytes.
                _mm_prefetch::<_MM_HINT_T0>(base.add(off) as *const i8);
                off += 64;
            }
        };
        if ahead != 0 {
            for u in keys.start..keys.end.min(keys.start + ahead) {
                hint(u);
            }
        }
        for blk in keys.clone().step_by(FLASH_BLOCK) {
            let mut s = [f32::NEG_INFINITY; 32];
            let mut smax = f32::NEG_INFINITY;
            for (l, sl) in s.iter_mut().enumerate() {
                let u = blk + l;
                if u >= keys.end {
                    break; // past the segment: weight exactly 0
                }
                let su = &key_slots[u];
                if su.seq != q.seq || su.pos > q.pos {
                    continue; // the -inf half of the causal mask
                }
                // TWIN: flash_seg_scalar — every edit outside the two marked
                // regions must be made in both. (This is the kq dot region;
                // only the sum order may differ from the scalar oracle.)
                // Pull the lines of the row `ahead` keys on while this one is
                // dotted: the kq walk streams each row exactly once per
                // segment pass. A prefetch is a cache hint, never a value —
                // no scalar twin of this.
                if ahead != 0 && u + ahead < keys.end {
                    hint(u + ahead);
                }
                // SAFETY: plus the fn contract: `keys16.row(u)` is
                // bounds-checked and its width is `d_head` (asserted at the
                // dispatch), so the panel walk stays inside one row.
                let kq = kq_dot_fa4_avx2(qrow, keys16.row(u));
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
            // TWIN: flash_seg_scalar — every edit outside the two marked
            // regions must be made in both. (This is the V accumulation
            // region; given the same weights it is bit-identical to the
            // scalar form — only the kq region may differ.)
            // SAFETY: the fn contract plus the mask above: every lane left
            // out of the active list is causally masked or past the segment
            // end, so `blk + l < keys.end <= keys16.len()` holds for each lane
            // read; the dispatch asserted `keys16.width() == rope_dims +
            // latent` and keeps `d_off + r.len() <= latent`, and `r.len() % 8
            // == 0` (`flash_simd` gated `latent % 8`, the split `latent % 16`).
            flash_v_accum_avx2(w, keys16, blk, p.rope_dims + d_off, r);
        }
        (m, s_sum)
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
    let views = v_up_views(wkb, p)?;
    wv_b_heads_with(gguf, &views, kqv_compressed, p)
}

/// Bytes one `latent`-wide row of `attn_kv_b` occupies in the file: the type's
/// block size in bytes times the number of blocks the row spans. One owner, so
/// the k-up and v_up sides of the same tensor cannot drift apart.
pub(crate) fn wkb_row_bytes(wkb: &TensorInfo, latent: usize) -> Result<usize, ModelError> {
    let ts = wkb.ty.type_size().ok_or(QuantError::Unsupported(wkb.ty))? as usize;
    let blck = wkb.ty.blck_size().ok_or(QuantError::Unsupported(wkb.ty))? as usize;
    Ok(ts * (latent / blck))
}

/// The per-head v_up views of `attn_kv_b` — the views `ggml_view_3d` takes, as
/// file offsets. One owner of the geometry: `Derived::new` builds these once per
/// block, and the direct-call path above builds them per call. The only way this
/// fails is a weight type with no block geometry, which [`wkb_row_bytes`] names.
pub(crate) fn v_up_views(wkb: &TensorInfo, p: &MlaParams) -> Result<Vec<TensorInfo>, ModelError> {
    let row_bytes = wkb_row_bytes(wkb, p.latent)?;
    Ok((0..p.n_head)
        .map(|h| TensorInfo {
            name: format!("{}.v_up.head{h}", wkb.name),
            dims: vec![p.latent as u64, p.v_head as u64],
            ty: wkb.ty,
            offset: wkb.offset + ((h * (p.nope + p.v_head) + p.nope) * row_bytes) as u64,
            nbytes: (row_bytes * p.v_head) as u64,
        })
        .collect())
}

/// The `wv_b` gather-matmul-scatter over prebuilt views, in staged form: the step
/// folds it into [`attn_heads_fused`], and `tests/attn.rs` holds the two together.
/// The views come from [`Derived`](super::derived::Derived).
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
    for (h, xh) in xhs.iter_mut().enumerate() {
        for t in 0..n_tokens {
            let src = kqv_compressed.col(t * p.n_head + h);
            xh.col_mut(t).copy_from_slice(src);
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
/// head) rows, each split into [`attn_halves`] latent ranges. Between
/// dispatches the calling thread works alone while every worker idles, and
/// the three stages are independent per row, so the two barriers the chain
/// paid bought nothing. The three stage functions above stay `pub`: they are
/// the oracle this path is gated against, bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn attn_heads_fused(
    gguf: &Gguf,
    wblocks: &[Q8Block],
    q: &Tensor2,
    q_rope: &Tensor2,
    keys16: KvRows<'_>,
    key_slots: &[Slot],
    q_slots: &[Slot],
    views: &[TensorInfo],
    p: &MlaParams,
) -> Result<(Tensor2, Tensor2, Tensor2), ModelError> {
    let halves = attn_halves(p);
    attn_heads_split(
        gguf, wblocks, q, q_rope, keys16, key_slots, q_slots, views, p, halves,
    )
}

/// How many latent ranges each (token, head) row of [`attn_heads_fused`]
/// splits into: 1 unless `BLOOMERY_ATTN_HALVES=2` (read once; 1 or 2, any
/// other value panics). One is the default because the split halves only the
/// V accumulation: both halves still stream every whole key row for the
/// scores, and a row's time follows that stream, so a decode step on twice
/// the threads reads twice the bytes for no shorter row. A latent that does
/// not halve into 8-value runs stays whole.
fn attn_halves(p: &MlaParams) -> usize {
    static LEVER: OnceLock<usize> = OnceLock::new();
    let lever = *LEVER.get_or_init(|| match std::env::var("BLOOMERY_ATTN_HALVES") {
        Err(_) => 1,
        Ok(v) => match v.trim() {
            "1" => 1,
            "2" => 2,
            other => panic!("BLOOMERY_ATTN_HALVES must be 1 or 2, got {other:?}"),
        },
    });
    if p.latent.is_multiple_of(16) {
        lever
    } else {
        1
    }
}

/// One row's arrival count for the `wv_b` handoff, on its own cache line: the
/// two participants of a row touch it once each.
#[repr(align(64))]
struct RowMark(std::sync::atomic::AtomicU32);

thread_local! {
    /// The dispatching thread's recycled arrival counters, one per (token,
    /// head) row; reset before every split dispatch, only ever grown.
    static ROW_MARKS: RefCell<Vec<RowMark>> = const { RefCell::new(Vec::new()) };
}

/// [`attn_heads_fused`] with the multi-query row split given. `pub` so the
/// gate can pin every shape against the chain at any depth in one process.
///
/// Two dispatch shapes, one set of bits:
///
///   * **one query** (a decode step) — split-K, three dispatches: (1)
///     `q_nope2` over (head, latent half) items; (2) the flash segments over
///     (segment, head) items, each leaving its partial `(m, S, R)`; (3) the
///     merge over (head, latent half) items, the row's last arrival running
///     `wv_b`. `halves` does not apply: stages 1 and 3 always split the latent
///     in two (when it halves into 8-value runs) and stage 2 splits the keys.
///   * **several queries** (prefill, a batch) — one dispatch over the (token,
///     head) rows, each split into `halves` latent ranges: 1 runs each row on
///     one participant, 2 on two, each accumulating one half of the latent.
///     Each row walks its segments serially ([`flash_row`]).
///
/// Bit identity with the chain `q_nope2_absorbed` → `flash_attn_latent` →
/// `wv_b_heads_with` (`hw_attn_heads_fused_bit_identical`,
/// `hw_attn_heads_split_bit_identical`): every `quantize_act`,
/// `q_nope2_cells`, flash-segment, merge and `quantize_col`/`dot_row` call
/// receives exactly the bytes it receives in the chain, so only WHICH thread
/// makes each call changes:
///
///   * (a) every `q_nope2` cell is computed from its own weight blocks and the
///     column's quantized blocks, whatever range the call covers;
///   * (b) a segment's scan is a function of the FA row, its key range and
///     the latent range it accumulates — the chain runs the same segments of
///     the same plan ([`flash_segment`] over [`visible_end`]); every step on
///     the accumulator (fill, rescale, the per-element FMA chain in key
///     order, the merge, `s_inv·v`) is per element;
///   * (c) the row's last participant to finish runs `wv_b` on the whole
///     `kqv_compressed` row — the gathered input the chain's batch copies.
///
/// The two index conventions are the chain's own: `q_nope2` columns are
/// head-major (`h·ne1 + t`), `kqv_compressed` rows and `kqv_2d` head spans
/// are token-major (`t·n_head + h`).
#[allow(
    clippy::too_many_arguments,
    reason = "the fused chain's inputs plus the multi-query split"
)]
pub fn attn_heads_split(
    gguf: &Gguf,
    wblocks: &[Q8Block],
    q: &Tensor2,
    q_rope: &Tensor2,
    keys16: KvRows<'_>,
    key_slots: &[Slot],
    q_slots: &[Slot],
    views: &[TensorInfo],
    p: &MlaParams,
    halves: usize,
) -> Result<(Tensor2, Tensor2, Tensor2), ModelError> {
    // Profiler hook: level-1 timer over the whole call, typeless — the call
    // walks three differently-shaped weight reads (derived Q8_0 blocks, F16
    // KV rows, Q3_K v_up views), and no single rows/k/weight-bytes triple
    // would be honest for all three. The wall is the statement; the span
    // columns carry the dispatch walls and their busiest chunks, summed over
    // the call's dispatches (chunk skew).
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
        keys16.width() == d_head,
        "attn_heads_fused: KV rows must be rope+latent = {d_head} wide, cache width is {}",
        keys16.width()
    );
    assert_eq!(
        views.len(),
        p.n_head,
        "attn_heads_fused: one v_up view per head"
    );
    // The AVX2 V twin walks 8-value runs, so each latent range must be one.
    assert!(
        (halves == 1 || halves == 2) && p.latent.is_multiple_of(8 * halves),
        "attn_heads_split: halves must be 1 or 2 and split the latent {} into 8-value runs, got {halves}",
        p.latent
    );

    let mut out = HeadsOut {
        q_nope2: Tensor2::scratch(p.latent, p.n_head * ne1),
        kqv_compressed: Tensor2::scratch(p.latent, ne1 * p.n_head),
        kqv_2d: Tensor2::scratch(p.n_head * p.v_head, ne1),
    };
    let io = HeadsIo {
        gguf,
        wblocks,
        q,
        q_rope,
        keys16,
        key_slots,
        q_slots,
        views,
        p,
        // The row twin is decided once per call, exactly as `flash_attn_latent` does.
        simd: flash_simd(p),
    };
    let mut acc = profile::CallAcc::new();
    // A decode row splits its keys across the pool only once it has more
    // than `SPLITK_MIN_BLOCKS` blocks: below that the two extra dispatches
    // cost more than the split buys, and the one-dispatch shape runs the
    // same plan and the same merge, so the bits are the same either way.
    let split =
        ne1 == 1 && visible_end(key_slots, &q_slots[0]).div_ceil(FLASH_BLOCK) > SPLITK_MIN_BLOCKS;
    let res = if split {
        heads_split_k(&io, &mut out, &mut acc)
    } else {
        heads_per_row(&io, &mut out, halves, &mut acc)
    };
    res?;
    if let Some(t_call) = t_call {
        profile::record(
            "attn_heads",
            gguf::GgmlType::Unknown(0),
            0,
            0,
            0,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    Ok((out.q_nope2, out.kqv_compressed, out.kqv_2d))
}

/// The inputs every attention-heads dispatch reads.
struct HeadsIo<'a> {
    gguf: &'a Gguf,
    wblocks: &'a [Q8Block],
    q: &'a Tensor2,
    q_rope: &'a Tensor2,
    keys16: KvRows<'a>,
    key_slots: &'a [Slot],
    q_slots: &'a [Slot],
    views: &'a [TensorInfo],
    p: &'a MlaParams,
    simd: bool,
}

/// The three tensors the chain produces.
struct HeadsOut {
    q_nope2: Tensor2,
    kqv_compressed: Tensor2,
    kqv_2d: Tensor2,
}

/// A profiled dispatch's wall and busiest chunk, folded into `acc`: `f` runs
/// on the pool over `0..n`; only a profiled call pays for the collector.
fn timed_dispatch<F>(acc: &mut profile::CallAcc, n: usize, f: F)
where
    F: Fn(std::ops::Range<usize>) + Sync,
{
    if profile::level() == 0 {
        threads::pool().for_each_chunk(n, f);
        return;
    }
    let busy = profile::ChunkSlots::<u64>::new();
    let t_span = Instant::now();
    threads::pool().for_each_chunk(n, |items| {
        let t_busy = Instant::now();
        f(items);
        busy.push(t_busy.elapsed().as_nanos() as u64);
    });
    let span_ns = t_span.elapsed().as_nanos() as u64;
    acc.add_span(span_ns, busy.into_vec().into_iter().max().unwrap_or(0));
}

/// Quantize head `h`'s `q_nope` slice of token `t` into `qcol` — the same
/// `quantize_act` blocks the chain quantizes on its caller thread.
fn quantize_q_nope(q: &Tensor2, p: &MlaParams, t: usize, h: usize, qcol: &mut [ActBlock]) {
    let qbase = t * q.ne0 + h * p.kq_head;
    let qnope = &q.data[qbase..qbase + p.nope];
    for (b, chk) in qnope.as_chunks::<32>().0.iter().enumerate() {
        qcol[b] = quantize_act(chk);
    }
}

/// A worker's `quantize_act` column and `q_nope2` cell buffer, taken from the
/// thread-locals for one chunk (every element rewritten before it is read).
fn take_qn2_scratch(nblocks: usize, latent: usize) -> (Vec<ActBlock>, Vec<f32>) {
    let mut qcol = QALL.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut cellbuf = CELL_BUF.with(|c| std::mem::take(&mut *c.borrow_mut()));
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
    (qcol, cellbuf)
}

fn give_qn2_scratch(qcol: Vec<ActBlock>, cellbuf: Vec<f32>) {
    QALL.with(|c| *c.borrow_mut() = qcol);
    CELL_BUF.with(|c| *c.borrow_mut() = cellbuf);
}

/// The arrival counters for the `wv_b` handoff, one per row, zeroed before
/// the dispatch publishes them — only when a row has more than one
/// participant; a whole row needs none.
fn take_row_marks(rows_n: usize, parts: usize) -> Vec<RowMark> {
    let mut marks = ROW_MARKS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if parts > 1 {
        if marks.len() < rows_n {
            marks.resize_with(rows_n, || RowMark(std::sync::atomic::AtomicU32::new(0)));
        }
        for m in &marks[..rows_n] {
            m.0.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    marks
}

/// Whether this arrival is row `row`'s last of `parts`: of two arrivals the
/// one that sees 1 is the last; the AcqRel count makes the partner's range
/// visible before the caller reads it.
fn last_arrival(marks: &[RowMark], row: usize, parts: usize) -> bool {
    parts == 1
        || marks[row]
            .0
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            == 1
}

/// Head `h`'s `wv_b` leg for token `t`: the same one-column `matmul_q` bytes
/// the chain's batch computes — its gathered input is a copy of this exact
/// `kqv_compressed` row, and a copy quantizes to the same bytes.
///
/// # Safety
/// `kv2` points at `kqv_2d`, the span `t·ne0 + h·v_head .. +v_head` is this
/// row's own and no other thread touches it during the call.
#[allow(
    clippy::too_many_arguments,
    reason = "the leg's inputs and its output cell span"
)]
unsafe fn wv_b_leg(
    io: &HeadsIo<'_>,
    kqv_row: &[f32],
    kv2: &crate::ops::SharedOut,
    kv2_ne0: usize,
    t: usize,
    h: usize,
) -> Result<(), ModelError> {
    let v_head = io.p.v_head;
    // SAFETY: the fn contract — this row's own span of `kqv_2d` column `t`.
    let out_col =
        unsafe { std::slice::from_raw_parts_mut(kv2.0.add(t * kv2_ne0 + h * v_head), v_head) };
    crate::ops::matvec_q_local(io.gguf, &io.views[h], kqv_row, out_col)
}

/// The multi-query shape: one dispatch over (latent range, row) items, each
/// running (a) its column's `q_nope2`, (b) the row's segments serially and
/// their merge over its latent range, (c) `wv_b` if it is the row's last
/// arrival.
fn heads_per_row(
    io: &HeadsIo<'_>,
    out: &mut HeadsOut,
    halves: usize,
    acc: &mut profile::CallAcc,
) -> Result<(), ModelError> {
    let p = io.p;
    let ne1 = io.q_slots.len();
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    let latent = p.latent;
    let d_head = p.rope_dims + latent;
    let rows_n = ne1 * p.n_head;
    let d_len = latent / halves;
    let ahead = kv_prefetch_for(ne1, io.simd);
    let n_seg = flash_segments();
    let marks = take_row_marks(rows_n, halves);
    let marks_ref = &marks[..];

    // Item `i` is row `i % rows_n`, latent range `i / rows_n` — the two
    // ranges of a row sit `rows_n` items apart.
    //
    // Item (half, row) writes only: `q_nope2` column `h·ne1 + t` (half 0
    // alone), `kqv_compressed` cells `row·latent + half·d_len ..+d_len`, and —
    // for the row's last arrival alone — `kqv_2d` column `t`'s span
    // `h·v_head..(h+1)·v_head`. It reads, besides the inputs, only its own
    // scratch and, in (c), the `kqv_compressed` row whose other range the
    // partner wrote before its arrival (the AcqRel count orders it).
    //
    // SAFETY (construction site): the pool chunks partition the item space,
    // the item cells above are disjoint across items (exactly one arrival is
    // the last), and the join publishes every write before any tensor is
    // read. The stage-(c) read goes through a shared reference into cells
    // both writers finished before the counter handed the row over.
    let qn2_ptr = crate::ops::SharedOut(out.q_nope2.data.as_mut_ptr());
    let kc_ptr = crate::ops::SharedOut(out.kqv_compressed.data.as_mut_ptr());
    let kv2_ptr = crate::ops::SharedOut(out.kqv_2d.data.as_mut_ptr());
    let kv2 = &kv2_ptr;
    let kv2_ne0 = out.kqv_2d.ne0;
    let kqv_compressed = &out.kqv_compressed;
    // The matvec is the one fallible stage; lowest-row-first precedence, the
    // `run_row_pool` error-channel pattern.
    let gate = crate::ops::ErrGate::new();
    timed_dispatch(acc, halves * rows_n, |items| {
        let (mut qcol, mut cellbuf) = take_qn2_scratch(nblocks, latent);
        let mut sc = RowScratch::take(d_head, latent, d_len, n_seg);
        let mut vis = (usize::MAX, 0usize);
        for item in items {
            let half = item / rows_n;
            let row = item % rows_n;
            let t = row / p.n_head;
            let h = row % p.n_head;
            // (a) `q_nope2` column `h·ne1 + t`, the whole column.
            quantize_q_nope(io.q, p, t, h, &mut qcol);
            let seg = &mut cellbuf[..latent];
            qdot::q_nope2_cells(
                &io.wblocks[h * span..(h + 1) * span],
                &qcol[..nblocks],
                0,
                latent,
                seg,
            );
            if half == 0 {
                for (j, &v) in seg.iter().enumerate() {
                    // SAFETY: cell `(h·ne1 + t)·latent + j` is this row's own, written by its half 0 alone — see the construction site.
                    unsafe { qn2_ptr.write((h * ne1 + t) * latent + j, v) };
                }
            }
            // (b) the row's segments and their merge over this item's latent
            // range, fed the column (a) left in `seg`.
            if vis.0 != t {
                vis = (t, visible_end(io.key_slots, &io.q_slots[t]));
            }
            let rk = RowKeys {
                keys16: io.keys16,
                key_slots: io.key_slots,
                q_slot: &io.q_slots[t],
                n_vis: vis.1,
            };
            flash_row(
                io.simd,
                ahead,
                io.q_rope.col(t * p.n_head + h),
                seg,
                &rk,
                p,
                n_seg,
                half * d_len,
                &mut sc,
                row * latent,
                &kc_ptr,
            );
            // (c) head h's `wv_b` leg, run by the row's last arrival.
            if !last_arrival(marks_ref, row, halves) {
                continue;
            }
            // SAFETY: the row's own span, written by its last arrival alone — see the construction site.
            if let Err(e) = unsafe { wv_b_leg(io, kqv_compressed.col(row), kv2, kv2_ne0, t, h) } {
                gate.offer(row, e);
                break;
            }
        }
        give_qn2_scratch(qcol, cellbuf);
        sc.give_back();
    });
    ROW_MARKS.with(|c| *c.borrow_mut() = marks);
    match gate.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

thread_local! {
    /// The dispatching thread's recycled split-K partials: per (row, segment)
    /// one latent-wide `R`, and its `m` and `S`. Only ever grown; every cell
    /// the merge reads was written by the segment dispatch of the same call.
    static SPLITK_R: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SPLITK_M: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SPLITK_S: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The one-query shape: split-K in three dispatches (see [`attn_heads_split`]).
///
/// Stage 2's items are (segment, head) pairs, segment-major, and a chunk
/// takes the items whose first block falls in it by cumulative block weight:
/// a segment of 5 blocks weighs 5, so the busiest thread holds about the mean
/// work plus one item whatever the segment sizes are. Which thread runs an
/// item moves no bit — each item is a whole segment of one row.
fn heads_split_k(
    io: &HeadsIo<'_>,
    out: &mut HeadsOut,
    acc: &mut profile::CallAcc,
) -> Result<(), ModelError> {
    let p = io.p;
    let nblocks = p.nope / 32;
    let span = p.latent * nblocks;
    let latent = p.latent;
    let d_head = p.rope_dims + latent;
    let rows_n = p.n_head;
    // Stages 1 and 3 split each head's latent in two when the halves are
    // 8-value runs (the AVX2 V twin's contract).
    let parts = if latent.is_multiple_of(16) { 2 } else { 1 };
    let d_len = latent / parts;
    let ahead = kv_prefetch_for(1, io.simd);
    let n_seg = flash_segments();
    let n_vis = visible_end(io.key_slots, &io.q_slots[0]);
    let used = flash_segments_used(n_vis, n_seg);
    let n_blocks = n_vis.div_ceil(FLASH_BLOCK);

    // Stage 1 — `q_nope2` over (latent part, head) items. Item `i` writes
    // only head `i % rows_n`'s column cells of part `i / rows_n`.
    //
    // SAFETY (construction site, all three stages): the pool chunks partition
    // each stage's item space and every item writes only its own cells —
    // stage 1 its `q_nope2` cells, stage 2 its (row, segment) partial `R`,
    // `m` and `S`, stage 3 its `kqv_compressed` range and (last arrival) its
    // `kqv_2d` span; each join publishes the writes before the next stage
    // reads them.
    let qn2_ptr = crate::ops::SharedOut(out.q_nope2.data.as_mut_ptr());
    timed_dispatch(acc, parts * rows_n, |items| {
        let (mut qcol, mut cellbuf) = take_qn2_scratch(nblocks, latent);
        for item in items {
            let part = item / rows_n;
            let h = item % rows_n;
            quantize_q_nope(io.q, p, 0, h, &mut qcol);
            let (j0, j1) = (part * d_len, (part + 1) * d_len);
            let seg = &mut cellbuf[..d_len];
            qdot::q_nope2_cells(
                &io.wblocks[h * span..(h + 1) * span],
                &qcol[..nblocks],
                j0,
                j1,
                seg,
            );
            for (jj, &v) in seg.iter().enumerate() {
                // SAFETY: cell `h·latent + j0 + jj` is this item's own — see the construction site.
                unsafe { qn2_ptr.write(h * latent + j0 + jj, v) };
            }
        }
        give_qn2_scratch(qcol, cellbuf);
    });

    // Stage 2 — the segments, over block-weighted (segment, head) items.
    let mut pr = SPLITK_R.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut pm = SPLITK_M.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut ps = SPLITK_S.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if pr.len() < rows_n * n_seg * latent {
        pr.resize(rows_n * n_seg * latent, 0.0);
    }
    if pm.len() < rows_n * n_seg {
        pm.resize(rows_n * n_seg, 0.0);
        ps.resize(rows_n * n_seg, 0.0);
    }
    {
        let pr_ptr = crate::ops::SharedOut(pr.as_mut_ptr());
        let pr_out = &pr_ptr;
        let pm_ptr = crate::ops::SharedOut(pm.as_mut_ptr());
        let ps_ptr = crate::ops::SharedOut(ps.as_mut_ptr());
        let q_nope2 = &out.q_nope2;
        let rk = RowKeys {
            keys16: io.keys16,
            key_slots: io.key_slots,
            q_slot: &io.q_slots[0],
            n_vis,
        };
        timed_dispatch(acc, n_blocks * rows_n, |units| {
            let mut qrow = FLASH_QROW.with(|c| std::mem::take(&mut *c.borrow_mut()));
            if qrow.len() < d_head {
                qrow.resize(d_head, 0.0);
            }
            let qrow = &mut qrow;
            let mut w = [0.0f32; 32];
            for s in 0..used {
                let keys = flash_segment(n_vis, n_seg, s);
                let b0 = keys.start / FLASH_BLOCK;
                let bs = keys.len().div_ceil(FLASH_BLOCK);
                // Item (s, h) starts at unit `b0·rows_n + h·bs`; this chunk
                // runs the items that start inside it.
                let base = b0 * rows_n;
                let first = |u: usize| u.saturating_sub(base).div_ceil(bs).min(rows_n);
                for h in first(units.start)..first(units.end) {
                    let qr = &mut qrow[..d_head];
                    fill_qrow(qr, io.q_rope.col(h), q_nope2.col(h));
                    let cell = h * n_seg + s;
                    // SAFETY: partial `R` of (h, s) is this item's own `latent`
                    // cells — see the construction site.
                    let r = unsafe {
                        std::slice::from_raw_parts_mut(pr_out.0.add(cell * latent), latent)
                    };
                    let (m, s_sum) =
                        flash_segment_row(io.simd, ahead, qr, &rk, p, keys.clone(), 0, r, &mut w);
                    // SAFETY: `m`/`S` of (h, s) are this item's own cells.
                    unsafe {
                        pm_ptr.write(cell, m);
                        ps_ptr.write(cell, s_sum);
                    }
                }
            }
            FLASH_QROW.with(|c| *c.borrow_mut() = std::mem::take(qrow));
        });
    }

    // Stage 3 — the merge over (latent part, head) items, then `wv_b` by the
    // head's last arrival.
    let marks = take_row_marks(rows_n, parts);
    let marks_ref = &marks[..];
    let kc_ptr = crate::ops::SharedOut(out.kqv_compressed.data.as_mut_ptr());
    let kv2_ptr = crate::ops::SharedOut(out.kqv_2d.data.as_mut_ptr());
    let kv2 = &kv2_ptr;
    let kv2_ne0 = out.kqv_2d.ne0;
    let kqv_compressed = &out.kqv_compressed;
    let (pr_ref, pm_ref, ps_ref) = (&pr[..], &pm[..], &ps[..]);
    let gate = crate::ops::ErrGate::new();
    timed_dispatch(acc, parts * rows_n, |items| {
        let mut r_buf = FLASH_R.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if r_buf.len() < d_len {
            r_buf.resize(d_len, 0.0);
        }
        for item in items {
            let part = item / rows_n;
            let h = item % rows_n;
            let cells = h * n_seg..h * n_seg + used;
            let r = &mut r_buf[..d_len];
            let s_tot = combine_segments(
                &pm_ref[cells.clone()],
                &ps_ref[cells],
                &pr_ref[h * n_seg * latent..],
                latent,
                part * d_len,
                r,
            );
            let s_inv = if s_tot > 0.0 { 1.0 / s_tot } else { 0.0 };
            for (d, &v) in r.iter().enumerate() {
                // SAFETY: cell `h·latent + part·d_len + d` is this item's own — see the construction site.
                unsafe { kc_ptr.write(h * latent + part * d_len + d, s_inv * v) };
            }
            if !last_arrival(marks_ref, h, parts) {
                continue;
            }
            // SAFETY: head h's own span, written by its last arrival alone — see the construction site.
            if let Err(e) = unsafe { wv_b_leg(io, kqv_compressed.col(h), kv2, kv2_ne0, 0, h) } {
                gate.offer(h, e);
                break;
            }
        }
        FLASH_R.with(|c| *c.borrow_mut() = r_buf);
    });
    ROW_MARKS.with(|c| *c.borrow_mut() = marks);
    SPLITK_R.with(|c| *c.borrow_mut() = pr);
    SPLITK_M.with(|c| *c.borrow_mut() = pm);
    SPLITK_S.with(|c| *c.borrow_mut() = ps);
    match gate.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
