//! The whole graph: token ids in, last position's logits out.
//!
//! This module wires `attn`, `ffn`, `moe` and `head`; it owns no math of its own
//! beyond the embedding lookup and the two residual adds. That is deliberate — each
//! sub-block was gated against the oracle while being **fed the oracle's own input**,
//! and the question this round asks is the one those gates cannot: does the chain
//! still land on ik's numbers when every input is our own output? See
//! `crates/model/tests/forward.rs` for the measured per-block answer.
//!
//! What this is not, so the tok/s next to it is read correctly:
//!
//!   * **CPU only.** Not one byte of this runs on either card.
//!   * **`ops::matmul_q` still dequantizes a weight row to f32 before a scalar dot.**
//!     It is parallel over output rows now (2026-09-19), but the arithmetic is the
//!     reference arithmetic: `crates/q3k-cpu` holds the AVX2 int8 path that fuses the
//!     dequant into the dot, and stage 1 does not call it yet. The profiler measured
//!     that fusion, not thread count, as the remaining per-core factor.
//!
//! Struck 2026-09-19: this header used to say "there is no KV cache" and "single-
//! threaded". Both were true when it was written and neither is now — `KvCache`
//! landed in `fcd59e8` and the row parallelization the same day.
//!
//! The last block is a special case in the reference graph and not here: ik inserts
//! `inp_out_ids` before block 26's FFN (`last_attn-26`/`last_ffn_inp-26` are
//! `GET_ROWS {2048, 1}` in the manifest) so that only the position it will sample
//! pays for the last FFN and the head. We run all columns through and slice at the
//! end. Every stage after attention is column-independent, so the two agree on the
//! column that survives; ik's version is cheaper, ours is simpler, and 1-5 is where
//! that trade starts to matter.
use crate::derived::Derived;
use crate::kv::KvCache;
use crate::ops::{Tensor2, f32_tensor, rms_norm};
use crate::profile;
use crate::{ModelError, Slot};
use gguf::{Gguf, TensorInfo, dequant_row};
use std::time::Instant;

/// Every block's residual output, kept for the gate. `l_out[b]` is the oracle's
/// `l_out-<b>`; `logits` is `result_output`, one column for the sampled position.
pub struct ForwardTrace {
    /// `inp_embd`: the embedding rows, `[embd, n_tokens]`.
    pub inp_embd: Tensor2,
    /// `l_out-<b>` for every block, all `n_tokens` columns wide — including the last
    /// block, where the reference keeps only the sampled column (see the module doc).
    pub l_out: Vec<Tensor2>,
    /// `result_output`: `[vocab, 1]` for the last position.
    pub logits: Tensor2,
}

/// `inp_embd`: one dequantized `token_embd.weight` row per id, in ggml's `GET_ROWS`
/// layout — token `t` occupies `data[t * embd .. (t + 1) * embd]`.
///
/// The row is dequantized, not converted: `token_embd.weight` is Q3_K in this file and
/// the 1-1 gate proved our Q3_K dequant is bit-identical to ggml's `to_float`, so this
/// lookup is expected to match the oracle **exactly**, not within a tolerance.
pub fn embed(gguf: &Gguf, tokens: &[u32]) -> Result<Tensor2, ModelError> {
    // Profiler hook (crate::profile): the whole lookup is weight dequant, so the
    // level-2 split has one stage and it is `dequant_w`.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let w = gguf
        .find("token_embd.weight")
        .ok_or_else(|| ModelError::MissingTensor("token_embd.weight".into()))?;
    let embd = w.dims[0] as usize;
    let vocab = w.dims[1] as usize;
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / vocab;

    let mut out = Tensor2::zeros(embd, tokens.len());
    for (t, &id) in tokens.iter().enumerate() {
        let id = id as usize;
        if id >= vocab {
            return Err(ModelError::Shape {
                what: "token id past the vocabulary",
                want_ne0: vocab,
                want_ne1: tokens.len(),
                got_ne0: id,
                got_ne1: t,
            });
        }
        let src = &bytes[id * row_bytes..(id + 1) * row_bytes];
        let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
        dequant_row(w.ty, src, out.col_mut(t))?;
        if let Some(t_d) = t_d {
            pacc.add_dequant_w(t_d.elapsed().as_nanos() as u64);
        }
    }
    if let Some(t_call) = t_call {
        profile::record(
            "embed",
            w.ty,
            tokens.len() as u64,
            tokens.len() as u64 * embd as u64,
            (tokens.len() * row_bytes) as u64,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
    }
    Ok(out)
}

/// Whether block `b` routes to experts. Decided by the file, the same way `ffn.rs`
/// decides between the dense and the shared-expert trio: presence, never a block
/// number. A V2-Lite file has one dense block and the V4.1 files ahead have three.
fn is_moe(gguf: &Gguf, b: usize) -> bool {
    // Profiler hook (crate::profile): the deciding `find` is a linear scan of
    // the tensor table (377 tensors in this file), paid per block per step.
    // Level-1 only, typeless: no weight is read, only looked for.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let routed = gguf.find(&format!("blk.{b}.ffn_gate_inp.weight")).is_some();
    if let Some(t_call) = t_call {
        profile::record_time("is_moe", t_call.elapsed().as_nanos() as u64);
    }
    routed
}

/// The architecture-wide rms epsilon, from the file. The same key `head.rs` reads; the
/// 1e-4 gates on every `attn_norm-N` are the proof that it is the right one.
fn rms_eps(gguf: &Gguf) -> f32 {
    gguf.architecture()
        .and_then(|a| gguf.value(&format!("{a}.attention.layer_norm_rms_epsilon")))
        .and_then(gguf::Value::as_f32)
        .expect("rms eps must come from the file, never from a literal")
}

fn gain(gguf: &Gguf, name: &str) -> Result<Vec<f32>, ModelError> {
    // Profiler hook (crate::profile): SELF time — the whole call minus what the
    // hooked `f32_tensor` child recorded (see `profile::site_ns_total`), which
    // leaves the `find`: a linear scan over the file's tensor table, paid twice
    // per block per step for the two norm gains.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let f32_before = if lvl > 0 {
        profile::site_ns_total("f32_tensor")
    } else {
        0
    };
    let t: &TensorInfo = gguf
        .find(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    let v = f32_tensor(gguf, t);
    if let Some(t_call) = t_call {
        let child = profile::site_ns_total("f32_tensor") - f32_before;
        let self_ns = (t_call.elapsed().as_nanos() as u64).saturating_sub(child);
        profile::record_time("gain", self_ns);
    }
    v
}

/// One transformer block against a KV cache: `l_out-(b-1)` in, `l_out-b` out.
///
/// Identical to [`block`] except that attention reads the cache instead of only the
/// batch. `block` is the shim now — it builds a cache holding exactly its batch.
#[allow(clippy::too_many_arguments)]
pub fn block_cached(
    gguf: &Gguf,
    b: usize,
    x: &Tensor2,
    q_slots: &[Slot],
    eps: f32,
    cache: &mut KvCache,
    range: &std::ops::Range<usize>,
    derived: &Derived,
) -> Result<Tensor2, ModelError> {
    let attn_gain = gain(gguf, &format!("blk.{b}.attn_norm.weight"))?;
    let normed = rms_norm(x, &attn_gain, eps);
    let kqv_out =
        crate::attn::block_attn_cached(gguf, b, &normed, q_slots, cache, b, range, derived)?
            .kqv_out;
    let ffn_inp = add(x, &kqv_out);

    let ffn_gain = gain(gguf, &format!("blk.{b}.ffn_norm.weight"))?;
    let ffn_normed = rms_norm(&ffn_inp, &ffn_gain, eps);
    let ffn_out = if is_moe(gguf, b) {
        crate::moe::moe_ffn(gguf, b, &ffn_normed)?
    } else {
        crate::ffn::dense_ffn(gguf, b, &ffn_normed)?
    };
    Ok(add(&ffn_inp, &ffn_out))
}

/// One transformer block: `l_out-(b-1)` in, `l_out-b` out.
///
/// `attn_norm → attention → residual → ffn_norm → (dense | moe) → residual`, which is
/// `build_deepseek2`'s block verbatim. The intermediate after the first add is the
/// oracle's `ffn_inp-N`.
pub fn block(
    gguf: &Gguf,
    b: usize,
    x: &Tensor2,
    slots: &[Slot],
    eps: f32,
    derived: &Derived,
) -> Result<Tensor2, ModelError> {
    let attn_gain = gain(gguf, &format!("blk.{b}.attn_norm.weight"))?;
    let normed = rms_norm(x, &attn_gain, eps);
    let kqv_out = crate::attn::block_attn(gguf, b, &normed, slots, derived)?;
    let ffn_inp = add(x, &kqv_out);

    let ffn_gain = gain(gguf, &format!("blk.{b}.ffn_norm.weight"))?;
    let ffn_normed = rms_norm(&ffn_inp, &ffn_gain, eps);
    let ffn_out = if is_moe(gguf, b) {
        crate::moe::moe_ffn(gguf, b, &ffn_normed)?
    } else {
        crate::ffn::dense_ffn(gguf, b, &ffn_normed)?
    };
    Ok(add(&ffn_inp, &ffn_out))
}

/// ggml's `ADD` over two same-shaped blocks. f32, elementwise, no accumulation order
/// to get wrong.
fn add(a: &Tensor2, b: &Tensor2) -> Tensor2 {
    // Profiler hook (crate::profile): the two residual adds per block are the
    // only math this module owns. Level-1 only, typeless — activation work, no
    // weight read.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    assert_eq!(
        (a.ne0, a.ne1),
        (b.ne0, b.ne1),
        "residual add needs both sides at the same shape"
    );
    let out = Tensor2::from_vec(
        a.ne0,
        a.ne1,
        a.data.iter().zip(&b.data).map(|(&x, &y)| x + y).collect(),
    );
    if let Some(t_call) = t_call {
        profile::record_time("residual_add", t_call.elapsed().as_nanos() as u64);
    }
    out
}

/// The logits for the **last** token of `tokens`, which is the one a sampler reads.
///
/// Positions are `0..tokens.len()` in sequence 0: this is a prefill, and it is the only
/// shape there is until the KV cache lands.
///
/// A convenience wrapper: it builds the [`Derived`] weights itself, every call.
/// Callers that run more than one pass — a decode loop, a gate over prompts —
/// should build once and drive [`step`], so the weight derivation is paid per
/// model and not per call. The signature stays token-in/logits-out because
/// `tests/forward.rs` and `tests/prompts.rs` call it as the one-shot oracle
/// path, and this round does not move those gates.
pub fn forward(gguf: &Gguf, tokens: &[u32]) -> Result<Tensor2, ModelError> {
    Ok(forward_trace(gguf, tokens)?.logits)
}

/// The same pass, keeping every block's residual output for the gate.
///
/// Builds its own [`Derived`] once per call (see [`forward`]'s note on when that
/// is the wrong trade).
pub fn forward_trace(gguf: &Gguf, tokens: &[u32]) -> Result<ForwardTrace, ModelError> {
    let n_block = gguf
        .block_count()
        .ok_or_else(|| ModelError::MissingTensor("metadata key block_count".into()))?
        as usize;
    let eps = rms_eps(gguf);
    let slots: Vec<Slot> = (0..tokens.len() as u32)
        .map(|pos| Slot { seq: 0, pos })
        .collect();
    let derived = Derived::new(gguf)?;

    let inp_embd = embed(gguf, tokens)?;
    let mut x = inp_embd.clone();
    let mut l_out = Vec::with_capacity(n_block);
    for b in 0..n_block {
        x = block(gguf, b, &x, &slots, eps, &derived)?;
        l_out.push(x.clone());
    }

    // The head runs on the sampled column only — the reference's `inp_out_ids`, applied
    // here instead of before the last block's FFN.
    let last = tokens.len() - 1;
    let tail = Tensor2::from_vec(x.ne0, 1, x.col(last).to_vec());
    let logits = crate::head::head(gguf, &tail)?;
    Ok(ForwardTrace {
        inp_embd,
        l_out,
        logits,
    })
}

/// The highest-scoring token id, ties to the lower id — `llama_sampler_init_greedy`'s
/// rule. Greedy decode is the only sampler stage 1 has, and it is what the 32-prompt
/// gate compares against ik.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The cache this model's blocks need, sized from the file.
///
/// The row width is `MlaParams`' `latent + rope_dims` for block 0 — 576 here — read
/// from the file rather than written down, because a differently shaped MLA file must
/// fail on the first push instead of caching truncated rows.
pub fn new_cache(gguf: &Gguf) -> Result<KvCache, ModelError> {
    let n_block = gguf
        .block_count()
        .ok_or_else(|| ModelError::MissingTensor("metadata key block_count".into()))?
        as usize;
    let p = crate::attn::MlaParams::read(gguf, 0)?;
    Ok(KvCache::new(n_block, p.latent + p.rope_dims))
}

/// Push `tokens` through the model against `cache`, and return the last position's logits.
///
/// **Only `tokens` flows.** Everything before them is in the cache and is not re-embedded,
/// re-normed or re-projected — that is the entire point, and passing the whole context
/// here on every step is exactly the bug this function exists to remove. The positions
/// continue from what the cache already holds, so a decode step is
/// `step(gguf, &[next], &mut cache)` and nothing else.
///
/// The first call on an empty cache is the prefill; there is no separate entry point,
/// because a prefill is a step whose batch happens to be longer than one.
///
/// `derived` is the weight side of the step — built once per model (by the caller,
/// right after open) and shared by every step against this `gguf`. The cache is the
/// sequence side; the two are passed separately because they are cleared for
/// different reasons (see `derived`'s module doc).
pub fn step(
    gguf: &Gguf,
    tokens: &[u32],
    cache: &mut KvCache,
    derived: &Derived,
) -> Result<Tensor2, ModelError> {
    let n_block = cache.n_block();
    let eps = rms_eps(gguf);
    let base = cache.next_pos(0);
    let slots: Vec<Slot> = (0..tokens.len() as u32)
        .map(|i| Slot {
            seq: 0,
            pos: base + i,
        })
        .collect();
    let range = cache.begin(&slots);

    let mut x = embed(gguf, tokens)?;
    for b in 0..n_block {
        x = block_cached(gguf, b, &x, &slots, eps, cache, &range, derived)?;
    }
    let last = tokens.len() - 1;
    let tail = Tensor2::from_vec(x.ne0, 1, x.col(last).to_vec());
    crate::head::head(gguf, &tail)
}
