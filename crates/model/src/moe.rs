//! MoE FFN: `ffn_norm-N` in, `ffn_out-N` out, for N >= 1.
//!
//! Dispatch is expert-bucketed, not per-token — `docs/research/quant-decode-efficiency.md`
//! §Q6 decision 2, and it cannot be retrofitted. Build the bucket table first, then one
//! matmul per bucket over the tokens routed to it, then scatter back through the inverse
//! permutation and weight. A per-token loop that happens to pass the gate at n_tokens = 6
//! is still the wrong structure and will be rejected in review. This mirrors what ik's
//! own CPU `mul_mat_id` does (`matrix_row_counts`/`matrix_rows` group rows by expert
//! before any dot runs — ggml.c:18258).
//!
//! The other structural line this module holds: only routed experts are read, never all
//! 64. `docs/research/mistralrs-prior-art.md` §4.4 measures the failure mode — mistral.rs
//! dequantizes every expert per token on x86_64 — and [`last_touched_experts`] exists so
//! the gate can keep that out of this engine.
//!
//! Gate: `crates/model/tests/moe.rs`. Routing ids are compared EXACTLY (the oracle holds
//! them as integer-valued f32); numerics come second at 1e-4, which the activation
//! quantization below makes honest rather than optimistic.
//!
//! Two activation paths, because the reference has two: a plain `MUL_MAT` quantizes f32
//! activations on its way into the dot and [`crate::ops::matmul_q`] models that path
//! exactly (gate-ops, 1.5e-5), while the MoE dot path (`MUL_MAT_ID` and the fused
//! up/gate ops) pre-quantizes activations through `type_traits[ty].vec_dot_type`, which
//! on the AVX2 reference build is the 32-block q8 for every quant this crate can read
//! except Q3_K. [`moe_dot`] owns that split.

use std::cell::{Cell, RefCell};

use gguf::quant::half_to_f32;
use gguf::{GgmlType, Gguf, TensorInfo, dequant_row};

use crate::ModelError;
use crate::ops::{Tensor2, matmul_q};

/// Tokens grouped by the expert they were routed to.
///
/// `offsets` has `n_experts + 1` entries; the tokens for expert `e` are
/// `order[offsets[e]..offsets[e + 1]]`, each an index into the input's `ne1`.
pub struct Buckets {
    pub offsets: Vec<u32>,
    pub order: Vec<u32>,
    /// The router weight for each entry of `order`, in the same order.
    pub weight: Vec<f32>,
}

impl Buckets {
    /// The token indices routed to expert `e` (`offsets` slices, so no bounds math at
    /// call sites — getting an off-by-one into `order` must not be possible).
    pub fn bucket(&self, e: usize) -> &[u32] {
        &self.order[self.offsets[e] as usize..self.offsets[e + 1] as usize]
    }

    /// The router weights aligned with [`Buckets::bucket`] — same slice geometry.
    pub fn bucket_weights(&self, e: usize) -> &[f32] {
        &self.weight[self.offsets[e] as usize..self.offsets[e + 1] as usize]
    }
}

/// What the last `route`/`moe_ffn` call computed, in the oracle's own layouts, so the
/// gate can compare intermediates by name instead of this module exposing them as API.
///
/// Flat layouts are ggml's: the last axis is contiguous, so e.g. `gate_par` entry
/// `(dim, slot, token)` lives at `token * n_used * ff + slot * ff + dim`.
#[derive(Debug, Clone)]
pub struct MoETrace {
    pub n_expert: usize,
    pub n_used: usize,
    pub n_tokens: usize,
    /// Router logits `{n_expert, n_tokens}`.
    pub logits: Vec<f32>,
    /// Softmax over all experts `{n_expert, n_tokens}`.
    pub probs: Vec<f32>,
    /// Chosen expert ids in rank order `{n_used, n_tokens}`.
    pub ids: Vec<i32>,
    /// Router weight of each chosen expert `{n_used, n_tokens}` — raw softmax probs,
    /// not renormalized (deepseek2 does not renormalize; verified against the oracle).
    pub weights: Vec<f32>,
    /// `silu(gate) * up` per routed expert `{expert_ff, n_used, n_tokens}`.
    pub gate_par: Vec<f32>,
    /// Down projection per routed expert `{embd, n_used, n_tokens}`.
    pub down: Vec<f32>,
    /// Weighted sum over experts `{embd, n_tokens}` — the `ffn_moe_out` tensor.
    pub routed_out: Vec<f32>,
    /// Shared-expert FFN output `{embd, n_tokens}` — the `ffn_shexp` tensor.
    pub shexp_out: Vec<f32>,
}

thread_local! {
    /// Intermediates of the last `route`/`moe_ffn` call on this thread (see [`MoETrace`]).
    static TRACE: RefCell<Option<MoETrace>> = const { RefCell::new(None) };

    /// Bit `e` set = expert `e`'s stacked weights were actually read by the last
    /// `moe_ffn` call on this thread. Set where the bytes are fetched, not where the
    /// bucket table says they should be, so the structural gate measures reality.
    static TOUCHED: Cell<u64> = const { Cell::new(0) };
}

/// The trace of the last `route`/`moe_ffn` call on this thread, if any.
pub fn last_trace() -> Option<MoETrace> {
    TRACE.with(|t| t.borrow().clone())
}

/// The expert mask actually dequantized by the last `moe_ffn` call on this thread.
/// Zero after `route` alone — routing reads the router, never the expert stacks.
pub fn last_touched_experts() -> u64 {
    TOUCHED.with(|c| c.get())
}

/// The hyperparameters this module reads from the file, never from literals.
struct Meta {
    n_expert: usize,
    n_used: usize,
    /// `expert_feed_forward_length` — the per-expert hidden width.
    ff: usize,
    /// `expert_weights_scale`. Applied to every router weight; 1.0 in this model, so
    /// the multiply is an exact no-op (IEEE: x * 1.0 == x).
    scale: f32,
}

impl Meta {
    fn read(gguf: &Gguf) -> Result<Meta, ModelError> {
        let get = |key: &str| -> Result<u64, ModelError> {
            gguf.arch_get_u64(key).ok_or_else(|| {
                // ModelError has no metadata variant; MissingTensor carries the key so
                // the message still names what was absent.
                ModelError::MissingTensor(format!("metadata key {key}"))
            })
        };
        let n_expert = get("expert_count")? as usize;
        let n_used = get("expert_used_count")? as usize;
        let ff = get("expert_feed_forward_length")? as usize;
        if n_expert == 0 || n_used == 0 || n_used > n_expert || ff == 0 || n_expert > 64 {
            return Err(ModelError::Shape {
                what: "moe metadata",
                want_ne0: 1,
                want_ne1: n_used.min(n_expert),
                got_ne0: n_expert,
                got_ne1: ff,
            });
        }
        let scale = gguf
            .architecture()
            .and_then(|a| gguf.value(&format!("{a}.expert_weights_scale")))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);
        Ok(Meta {
            n_expert,
            n_used,
            ff,
            scale,
        })
    }
}

fn tensor<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a TensorInfo, ModelError> {
    gguf.find(name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_string()))
}

/// Route every token to its `n_used` highest-probability experts: gate matmul, softmax
/// over all experts, top-k by (probability desc, id asc).
pub fn route(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Buckets, ModelError> {
    let meta = Meta::read(gguf)?;
    let (buckets, sel) = route_inner(gguf, block, x, &meta)?;
    TRACE.with(|t| {
        *t.borrow_mut() = Some(MoETrace {
            n_expert: meta.n_expert,
            n_used: meta.n_used,
            n_tokens: x.ne1,
            logits: sel.logits,
            probs: sel.probs,
            ids: sel.ids,
            weights: sel.weights,
            gate_par: Vec::new(),
            down: Vec::new(),
            routed_out: Vec::new(),
            shexp_out: Vec::new(),
        })
    });
    Ok(buckets)
}

/// What routing computes besides the bucket table. Kept separate so `moe_ffn` reuses
/// the exact same routing code the gate exercises through [`route`].
struct Selection {
    logits: Vec<f32>,
    probs: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

fn route_inner(
    gguf: &Gguf,
    block: usize,
    x: &Tensor2,
    meta: &Meta,
) -> Result<(Buckets, Selection), ModelError> {
    let n_expert = meta.n_expert;
    let n_used = meta.n_used;
    let n_tokens = x.ne1;

    let gate_inp = tensor(gguf, &format!("blk.{block}.ffn_gate_inp.weight"))?;
    let logits = matmul_q(gguf, gate_inp, x)?; // {n_expert, n_tokens}, F32 weights

    // Softmax over all experts, per token. ggml accumulates the sum in f64 and divides
    // in f32; an f32 sum differs after the 6th digit, which the 1e-4 gate cannot see.
    let mut probs = vec![0.0f32; n_expert * n_tokens];
    for t in 0..n_tokens {
        let src = logits.col(t);
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        let dst = &mut probs[t * n_expert..(t + 1) * n_expert];
        for (o, &v) in dst.iter_mut().zip(src) {
            let e = (v - max).exp();
            *o = e;
            sum += e as f64;
        }
        for o in dst.iter_mut() {
            *o /= sum as f32;
        }
    }

    // Top-k with ties broken toward the smaller expert id, matching ggml's argsort
    // ordering on the reference data (verified: distinct probabilities throughout).
    let mut ids = vec![0i32; n_used * n_tokens];
    let mut weights = vec![0.0f32; n_used * n_tokens];
    let mut ranked: Vec<u32> = (0..n_expert as u32).collect();
    for t in 0..n_tokens {
        let p = &probs[t * n_expert..(t + 1) * n_expert];
        ranked.sort_by(|&a, &b| {
            p[b as usize]
                .partial_cmp(&p[a as usize])
                .unwrap()
                .then(a.cmp(&b))
        });
        for (s, &e) in ranked.iter().take(n_used).enumerate() {
            ids[t * n_used + s] = e as i32;
            weights[t * n_used + s] = p[e as usize];
        }
    }

    // The bucket table: counts -> prefix offsets -> fill in token order, so each
    // expert's bucket lists its tokens ascending.
    let mut counts = vec![0u32; n_expert];
    for &e in &ids {
        counts[e as usize] += 1;
    }
    let mut offsets = Vec::with_capacity(n_expert + 1);
    offsets.push(0);
    for &c in &counts {
        let last = *offsets.last().unwrap();
        offsets.push(last + c);
    }
    let mut order = vec![0u32; n_used * n_tokens];
    let mut weight = vec![0.0f32; n_used * n_tokens];
    let mut cursor = offsets[..n_expert].to_vec();
    for t in 0..n_tokens {
        for s in 0..n_used {
            let e = ids[t * n_used + s] as usize;
            order[cursor[e] as usize] = t as u32;
            weight[cursor[e] as usize] = weights[t * n_used + s];
            cursor[e] += 1;
        }
    }

    Ok((
        Buckets {
            offsets,
            order,
            weight,
        },
        Selection {
            logits: logits.data,
            probs,
            ids,
            weights,
        },
    ))
}

/// `silu(x) = x / (1 + e^{-x})`, ggml's `ggml_vec_silu_f` form.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// IEEE-754 binary16 with round-to-nearest-even, matching `GGML_FP32_TO_FP16`.
/// Only scales go through it; the tail cases matter because a quiet 32-block can sit
/// anywhere in the half subnormal range.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x007f_ffff;

    if exp == 0xff {
        // Inf/NaN; keep NaN payload bits that fit, like GGML_FP32_TO_FP16.
        let nan = if frac & 0x1fff != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | nan | (frac >> 13) as u16;
    }
    // f32 exponent field (bias 127) rebased to f16 (bias 15).
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00; // overflow rounds to infinity
    }
    if e > 0 {
        // Normal half: 23 fraction bits -> 10, round to nearest even.
        let t = frac >> 13;
        let r = frac & 0x1fff;
        let mut h = (e as u32) << 10 | t;
        if r > 0x1000 || (r == 0x1000 && t & 1 == 1) {
            h += 1; // carries into the exponent field when it must
        }
        return sign | h as u16;
    }
    // Subnormal half: m = 1.frac scaled to units of 2^-24, shifted right by (1 - e).
    let s = (1 - e) as u32;
    if s >= 25 {
        return sign; // below half of the smallest subnormal step
    }
    let m24 = 0x0080_0000 | frac;
    let t = m24 >> s;
    let rem = m24 & ((1u32 << s) - 1);
    let half = 1u32 << (s - 1);
    let mut m = t;
    if rem > half || (rem == half && t & 1 == 1) {
        m += 1;
    }
    if m >= 0x400 {
        // Rounding overflowed the subnormal range into the smallest normal.
        return sign | 0x0400;
    }
    sign | m as u16
}

/// Which 16-bit format the reference's block scale `d` was rounded through before the
/// codes were scaled by it. The two MoE ops differ here, on the same build, because
/// they resolve different quantizers:
///
/// * `Half` — the fused up/gate op's activations: f16 scale (empirical, gate-confirmed
///   at 4.6e-5; the f16 shape is `quantize_row_q8_1_x4`'s, iqk_quantize.cpp:1170).
/// * `Bf16` — `MUL_MAT_ID`'s activations: `type_traits[Q8_2_X4].from_float` =
///   `quantize_row_q8_2_x4`, whose AVX2 branch rounds `d` through **bf16** and computes
///   `id = 1/bf16(d)` (iqk_quantize.cpp:1095-1103). An f16 scale here leaves the down
///   projection 0.2 % away from the oracle — measured 2.8e-2 on values up to 14.
#[derive(Clone, Copy, PartialEq)]
enum ActScale {
    Half,
    Bf16,
}

/// bf16 with round-to-nearest-even, as `GGML_FP32_TO_BF16`'s software path: add a
/// rounding bias to the lower 16 bits, then truncate. NaN stays a NaN (payload shifted).
fn f32_to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let bias = 0x7fff + ((b >> 16) & 1);
    let mut r = b.wrapping_add(bias) >> 16;
    if b & 0x7fff_ffff > 0x7f80_0000 {
        r |= 0x0040; // keep NaN-ness when truncation would quiet it away
    }
    r as u16
}

/// Round-trip f32 activations through the 32-block q8 quantization that ggml's MoE dot
/// path applies before an expert matmul on the reference build (AVX2:
/// `type_traits[Q4_K/Q5_K/Q5_0/Q5_1/Q6_K].vec_dot_type == Q8_2_X4`, ggml.c:756-771).
///
/// Per 32 values: `d = amax/127` in f32, rounded to `scale`'s 16-bit format for
/// storage; codes are `round_ties_even(x * id)` saturated at +127 (the pack
/// instructions saturate, and |x * id| cannot exceed 127.5); the effective value is
/// `d16 * q`. Where `id` comes from differs by [`ActScale`] — see its doc.
///
/// `x.len()` must be a multiple of 32; every row this multiplies is.
fn quantize_row_q8_x4_roundtrip(x: &mut [f32], scale: ActScale) {
    assert!(x.len().is_multiple_of(32), "q8 x4 blocks are 32 values");
    for blk in x.chunks_exact_mut(32) {
        let mut amax = 0.0f32;
        for v in blk.iter() {
            amax = amax.max(v.abs());
        }
        if amax == 0.0 {
            blk.fill(0.0);
            continue;
        }
        let d = amax / 127.0;
        let d16 = match scale {
            ActScale::Half => half_to_f32(f32_to_f16_bits(d)),
            ActScale::Bf16 => f32::from_bits((f32_to_bf16_bits(d) as u32) << 16),
        };
        // Half computes id from the unrounded f32 d; Bf16 overwrote d with bf16(d)
        // before computing id, in the reference's template.
        let id = match scale {
            ActScale::Half => 1.0 / d,
            ActScale::Bf16 => 1.0 / d16,
        };
        for v in blk.iter_mut() {
            let q = (*v * id).round_ties_even().min(127.0);
            *v = d16 * q;
        }
    }
}

/// A bucket matmul: weight rows dequantized through `dequant_row`, activations
/// pre-quantized by [`quantize_row_q8_x4_roundtrip`] with `scale`, dot in f32. Same
/// shape contract as `ops::matmul_q` (`W.dims == [k, n]`, `x` is `[k, n_tokens]`).
fn matmul_q8_x4(
    gguf: &Gguf,
    w: &TensorInfo,
    x: &Tensor2,
    scale: ActScale,
) -> Result<Tensor2, ModelError> {
    let k = w.dims[0] as usize;
    let n = w.dims[1] as usize;
    if x.ne0 != k {
        return Err(ModelError::Shape {
            what: "matmul_q8_x4 input",
            want_ne0: k,
            want_ne1: x.ne1,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / n;
    let mut act = x.data.clone();
    for t in 0..x.ne1 {
        quantize_row_q8_x4_roundtrip(&mut act[t * k..(t + 1) * k], scale);
    }
    let mut row = vec![0.0f32; k];
    let mut out = Tensor2::zeros(n, x.ne1);
    for r in 0..n {
        dequant_row(w.ty, &bytes[r * row_bytes..(r + 1) * row_bytes], &mut row)?;
        for t in 0..x.ne1 {
            let xc = &act[t * k..(t + 1) * k];
            // f64 accumulation: both operands are exactly representable (weight =
            // f16 scale x small int, activation = bf16/f16 scale x small int), so the
            // products are exact in f64 and the only error left is the reference's own
            // block-scaled f32 accumulation. An f32 loop over k=1408 rounds every add
            // and lands ~3e-4 away from the oracle's exact-integer block dots.
            let mut acc = 0.0f64;
            for (a, &v) in row.iter().zip(xc) {
                acc += *a as f64 * v as f64;
            }
            out.data[t * n + r] = acc as f32;
        }
    }
    Ok(out)
}

/// Matmul with the activation quantization the reference's MoE dot path used. The
/// `scale` argument is the op's, not the type's — fused up/gate and `MUL_MAT_ID`
/// quantized through different 16-bit scale formats (see [`ActScale`]). Q3_K pairs
/// with Q8_K on every build and F32/F16 never quantize — `ops::matmul_q` already
/// models both exactly, so those types go there instead of a second implementation.
fn moe_dot(
    gguf: &Gguf,
    w: &TensorInfo,
    x: &Tensor2,
    scale: ActScale,
) -> Result<Tensor2, ModelError> {
    match w.ty {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q6_K => {
            matmul_q8_x4(gguf, w, x, scale)
        }
        _ => matmul_q(gguf, w, x),
    }
}

/// The `e`-th matrix of a stacked expert tensor `{k, n, n_expert}` as a 2-D
/// `TensorInfo` into the same mapping. `gguf.data` bounds-checks the view again, so a
/// miscomputed slice errors instead of reading a neighbor expert.
fn expert_view(w: &TensorInfo, e: usize, n_expert: usize) -> Result<TensorInfo, ModelError> {
    if w.dims.len() != 3
        || w.dims[2] as usize != n_expert
        || !w.nbytes.is_multiple_of(n_expert as u64)
    {
        return Err(ModelError::Shape {
            what: "expert stack",
            want_ne0: 3,
            want_ne1: n_expert,
            got_ne0: w.dims.len(),
            got_ne1: w.dims.get(2).copied().unwrap_or(0) as usize,
        });
    }
    let per = w.nbytes / n_expert as u64;
    Ok(TensorInfo {
        name: w.name.clone(),
        dims: vec![w.dims[0], w.dims[1]],
        ty: w.ty,
        offset: w.offset + e as u64 * per,
        nbytes: per,
    })
}

/// The MoE FFN: `ffn_norm-N` in, `ffn_out-N` out — routed experts plus shared experts.
///
/// Routed half: one bucket at a time, three matmuls per routed expert, scatter through
/// the inverse permutation with the router weights. Shared half: the dense FFN with
/// `_shexp` names. This round computes it inline; it should later call
/// `ffn::dense_ffn` once that module lands — the lead wires that swap.
pub fn moe_ffn(gguf: &Gguf, block: usize, x: &Tensor2) -> Result<Tensor2, ModelError> {
    TOUCHED.with(|c| c.set(0));
    let meta = Meta::read(gguf)?;
    let n_expert = meta.n_expert;
    let n_used = meta.n_used;
    let ff = meta.ff;
    let n_tokens = x.ne1;
    let embd = x.ne0;

    let gate_exps = tensor(gguf, &format!("blk.{block}.ffn_gate_exps.weight"))?;
    let up_exps = tensor(gguf, &format!("blk.{block}.ffn_up_exps.weight"))?;
    let down_exps = tensor(gguf, &format!("blk.{block}.ffn_down_exps.weight"))?;
    let expect_stack = |w: &TensorInfo, k: usize, n: usize| -> Result<(), ModelError> {
        if w.dims != vec![k as u64, n as u64, n_expert as u64] {
            return Err(ModelError::Shape {
                what: "expert stack",
                want_ne0: k,
                want_ne1: n,
                got_ne0: w.dims.first().copied().unwrap_or(0) as usize,
                got_ne1: w.dims.get(1).copied().unwrap_or(0) as usize,
            });
        }
        Ok(())
    };
    expect_stack(gate_exps, embd, ff)?;
    expect_stack(up_exps, embd, ff)?;
    expect_stack(down_exps, ff, embd)?;

    let (buckets, sel) = route_inner(gguf, block, x, &meta)?;

    let mut routed = Tensor2::zeros(embd, n_tokens);
    let mut gate_par = vec![0.0f32; ff * n_used * n_tokens];
    let mut down_all = vec![0.0f32; embd * n_used * n_tokens];
    let mut touched = 0u64;

    for e in 0..n_expert {
        let bucket = buckets.bucket(e);
        if bucket.is_empty() {
            continue; // the whole point: an unrouted expert's bytes are never read
        }
        // Set where the weights are fetched, so this mask counts dequantized reality.
        touched |= 1 << e;

        let m = bucket.len();
        let mut xb = Tensor2::zeros(embd, m);
        for (i, &t) in bucket.iter().enumerate() {
            xb.col_mut(i).copy_from_slice(x.col(t as usize));
        }

        let g = moe_dot(
            gguf,
            &expert_view(gate_exps, e, n_expert)?,
            &xb,
            ActScale::Half,
        )?;
        let u = moe_dot(
            gguf,
            &expert_view(up_exps, e, n_expert)?,
            &xb,
            ActScale::Half,
        )?;
        let mut par = Tensor2::zeros(ff, m);
        for (p, (&gv, &uv)) in par.data.iter_mut().zip(g.data.iter().zip(&u.data)) {
            *p = silu(gv) * uv;
        }

        let d = moe_dot(
            gguf,
            &expert_view(down_exps, e, n_expert)?,
            &par,
            ActScale::Bf16,
        )?;

        let weights = buckets.bucket_weights(e);
        for i in 0..m {
            let t = bucket[i] as usize;
            let w = weights[i] * meta.scale;
            // Rank of `e` in token `t`'s selection — only the trace layout needs it.
            let s = (0..n_used)
                .find(|&s| sel.ids[t * n_used + s] as usize == e)
                .expect("every bucket entry comes from its token's selection");
            gate_par[t * n_used * ff + s * ff..t * n_used * ff + (s + 1) * ff]
                .copy_from_slice(par.col(i));
            down_all[t * n_used * embd + s * embd..t * n_used * embd + (s + 1) * embd]
                .copy_from_slice(d.col(i));
            let acc = routed.col_mut(t);
            for (a, &dv) in acc.iter_mut().zip(d.col(i)) {
                *a += w * dv;
            }
        }
    }
    TOUCHED.with(|c| c.set(touched));

    // Shared experts: same dense FFN shape as block 0 with `_shexp` names. The
    // reference computes gate/up through the fused up/gate op (32-block q8 activations,
    // hence `moe_dot`) and the down projection through a plain MUL_MAT (`matmul_q`).
    let shexp_gate = tensor(gguf, &format!("blk.{block}.ffn_gate_shexp.weight"))?;
    let shexp_up = tensor(gguf, &format!("blk.{block}.ffn_up_shexp.weight"))?;
    let shexp_down = tensor(gguf, &format!("blk.{block}.ffn_down_shexp.weight"))?;
    let g = moe_dot(gguf, shexp_gate, x, ActScale::Half)?;
    let u = moe_dot(gguf, shexp_up, x, ActScale::Half)?;
    let mut par = Tensor2::zeros(g.ne0, g.ne1);
    for (p, (&gv, &uv)) in par.data.iter_mut().zip(g.data.iter().zip(&u.data)) {
        *p = silu(gv) * uv;
    }
    let sh = matmul_q(gguf, shexp_down, &par)?;

    let mut out = routed.clone();
    for (o, &sv) in out.data.iter_mut().zip(&sh.data) {
        *o += sv;
    }

    TRACE.with(|t| {
        *t.borrow_mut() = Some(MoETrace {
            n_expert,
            n_used,
            n_tokens,
            logits: sel.logits,
            probs: sel.probs,
            ids: sel.ids,
            weights: sel.weights,
            gate_par,
            down: down_all,
            routed_out: routed.data.clone(),
            shexp_out: sh.data.clone(),
        })
    });
    Ok(out)
}
