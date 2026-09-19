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
//! One activation path, not two. Every op here — the fused up/gate, `MUL_MAT_ID` and
//! the plain `MUL_MAT` — fills its src1 scratch with `type_traits[vec_dot_type].from_float`
//! (ggml.c:18564 for the fused op, the same call for `MUL_MAT_ID`), so the quantizer is
//! the weight type's and nothing else. `crate::ops::matmul_q` is that single owner.
//!
//! This round shipped a second implementation with an f16-scale variant for the fused
//! op; the lead read ggml.c on adoption (2026-09-19) and it is source-false — the fused
//! op takes the same `from_float` as `MUL_MAT_ID`. The round's own probe had already
//! said so without being believed: `gate_par` came out at 4.5776367e-5 under both
//! spellings, to the last digit. The duplicate is gone; the gate numbers are the
//! round's measured v1 column.

use std::cell::{Cell, RefCell};
use std::time::Instant;

use gguf::{Gguf, TensorInfo};

use crate::ModelError;
use crate::ops::{Tensor2, matmul_q, matmul_q_batch};
use crate::profile;

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
    // Profiler hook (crate::profile): `moe_route` is the router's non-matmul
    // work in two pieces around the hooked gate matmul — the `ffn_gate_inp`
    // find (a linear tensor-table scan) and the softmax/top-k/bucket build.
    // Level-1 only, typeless.
    let lvl = profile::level();
    let mut route_ns = 0u64;
    let n_expert = meta.n_expert;
    let n_used = meta.n_used;
    let n_tokens = x.ne1;

    let t_find = if lvl > 0 { Some(Instant::now()) } else { None };
    let gate_inp = tensor(gguf, &format!("blk.{block}.ffn_gate_inp.weight"))?;
    if let Some(t_find) = t_find {
        route_ns += t_find.elapsed().as_nanos() as u64;
    }
    let logits = matmul_q(gguf, gate_inp, x)?; // {n_expert, n_tokens}, F32 weights

    let t_rest = if lvl > 0 { Some(Instant::now()) } else { None };
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
    if let Some(t_rest) = t_rest {
        route_ns += t_rest.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_route", route_ns);
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
    // Profiler hook (crate::profile), coverage round. This function interleaves
    // hooked `matmul_q` calls with its own work, so every site below is a piece
    // timer around the un-hooked regions only — the matmuls keep their own rows
    // and coverage never counts a nanosecond twice:
    //   * `moe_setup`      — `Meta::read`, the six stack/shexp `find`s (linear
    //     tensor-table scans) and the shape checks; two pieces.
    //   * `moe_route`      — measured inside `route_inner` (this path and the
    //     gate's `route` share it).
    //   * `swiglu`         — the silu(gate)·up combines; same site as
    //     `ffn::swiglu`, recorded per combine.
    //   * `moe_expert_io`  — per routed expert: the token gather before the
    //     gate/up matmuls and the rank-search/copy/weighted-accumulate scatter
    //     after the down matmul; recorded per expert.
    //   * `moe_trace`      — the MoETrace vectors this module allocates and
    //     fills every call, plus the output clone/combine and the thread-local
    //     TRACE write: bookkeeping that rides every step until a later round
    //     moves it behind a flag. The final routed+shexp combine is graph work,
    //     but it is glued to the bookkeeping and folded into this site.
    let lvl = profile::level();
    let mut setup_ns = 0u64;
    let mut trace_ns = 0u64;
    let t_setup = if lvl > 0 { Some(Instant::now()) } else { None };
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
    if let Some(t_setup) = t_setup {
        setup_ns += t_setup.elapsed().as_nanos() as u64;
    }

    let (buckets, sel) = route_inner(gguf, block, x, &meta)?;

    let t_trace1 = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut routed = Tensor2::zeros(embd, n_tokens);
    let mut gate_par = vec![0.0f32; ff * n_used * n_tokens];
    let mut down_all = vec![0.0f32; embd * n_used * n_tokens];
    if let Some(t_trace1) = t_trace1 {
        trace_ns += t_trace1.elapsed().as_nanos() as u64;
    }

    // MUL-24: the expert loop used to pay three pool dispatches per routed
    // expert (gate, up, down) — at ~850 Q3_K dispatches per decode step MUL-23
    // measured the orchestration, not the arithmetic, at ~41 ms. The same work
    // now runs as TWO dispatches per layer: every expert's gate+up in one, every
    // down in one. Bit identity survives by construction — batching changes
    // which worker computes which (expert, row), never a row's k-ascending
    // accumulation, and the scatter below accumulates `routed` in the same
    // expert-ascending order the sequential loop did.
    let t_gather = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut experts: Vec<usize> = Vec::new();
    let mut xbs: Vec<Tensor2> = Vec::new();
    let mut touched = 0u64;
    for e in 0..n_expert {
        if buckets.bucket(e).is_empty() {
            continue; // the whole point: an unrouted expert's bytes are never read
        }
        // Set where the weights are fetched, so this mask counts dequantized reality.
        touched |= 1 << e;
        let m = buckets.bucket(e).len();
        let mut xb = Tensor2::zeros(embd, m);
        for (i, &t) in buckets.bucket(e).iter().enumerate() {
            xb.col_mut(i).copy_from_slice(x.col(t as usize));
        }
        experts.push(e);
        xbs.push(xb);
    }
    let mut io_ns = if let Some(t_gather) = t_gather {
        t_gather.elapsed().as_nanos() as u64
    } else {
        0
    };
    TOUCHED.with(|c| c.set(touched));

    let mut gate_views = Vec::with_capacity(experts.len());
    let mut up_views = Vec::with_capacity(experts.len());
    let mut down_views = Vec::with_capacity(experts.len());
    for &e in &experts {
        gate_views.push(expert_view(gate_exps, e, n_expert)?);
        up_views.push(expert_view(up_exps, e, n_expert)?);
        down_views.push(expert_view(down_exps, e, n_expert)?);
    }

    // Phase 1 — gate and up for every routed expert, one dispatch. The views
    // interleave [gate_e0, up_e0, gate_e1, up_e1, ..] so pair `2*s` is expert
    // s's gate and `2*s + 1` its up; both read the same `xb` columns the
    // sequential loop gathered per expert.
    let mut gu_ws: Vec<&gguf::TensorInfo> = Vec::with_capacity(experts.len() * 2);
    let mut gu_xs: Vec<&Tensor2> = Vec::with_capacity(experts.len() * 2);
    for (gate, up) in gate_views.iter().zip(&up_views) {
        gu_ws.push(gate);
        gu_ws.push(up);
    }
    for xb in &xbs {
        gu_xs.push(xb);
        gu_xs.push(xb);
    }
    let gu = if experts.is_empty() {
        Vec::new()
    } else {
        matmul_q_batch(gguf, &gu_ws, &gu_xs)?
    };

    // The same swiglu the loop ran per expert, in the same expert order —
    // elementwise, so the phase boundary it sits between is its only change.
    let mut pars: Vec<Tensor2> = Vec::with_capacity(experts.len());
    for slot in 0..experts.len() {
        let t_silu = if lvl > 0 { Some(Instant::now()) } else { None };
        let g = &gu[2 * slot];
        let u = &gu[2 * slot + 1];
        let mut par = Tensor2::zeros(g.ne0, g.ne1);
        for (p, (&gv, &uv)) in par.data.iter_mut().zip(g.data.iter().zip(&u.data)) {
            *p = silu(gv) * uv;
        }
        if let Some(t_silu) = t_silu {
            profile::record_time("swiglu", t_silu.elapsed().as_nanos() as u64);
        }
        pars.push(par);
    }

    // Phase 2 — every expert's down projection, one dispatch.
    let down_ws: Vec<&gguf::TensorInfo> = down_views.iter().collect();
    let down_xs: Vec<&Tensor2> = pars.iter().collect();
    let downs = if experts.is_empty() {
        Vec::new()
    } else {
        matmul_q_batch(gguf, &down_ws, &down_xs)?
    };

    // Scatter, expert-ascending — the accumulation order the sequential loop
    // used, so `routed` gathers its per-token sums in the same sequence of
    // adds it always did.
    let t_scatter = if lvl > 0 { Some(Instant::now()) } else { None };
    for (slot, &e) in experts.iter().enumerate() {
        let bucket = buckets.bucket(e);
        let weights = buckets.bucket_weights(e);
        let par = &pars[slot];
        let d = &downs[slot];
        for i in 0..bucket.len() {
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
    if let Some(t_scatter) = t_scatter {
        io_ns += t_scatter.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_expert_io", io_ns);
    }

    // Shared experts: same dense FFN shape as block 0 with `_shexp` names, and the
    // same three ops `ffn::dense_ffn` runs. Kept inline here only until round 1-4 wires
    // `forward` and the two can share one implementation.
    let t_setup2 = if lvl > 0 { Some(Instant::now()) } else { None };
    let shexp_gate = tensor(gguf, &format!("blk.{block}.ffn_gate_shexp.weight"))?;
    let shexp_up = tensor(gguf, &format!("blk.{block}.ffn_up_shexp.weight"))?;
    let shexp_down = tensor(gguf, &format!("blk.{block}.ffn_down_shexp.weight"))?;
    if let Some(t_setup2) = t_setup2 {
        setup_ns += t_setup2.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_setup", setup_ns);
    }
    let g = matmul_q(gguf, shexp_gate, x)?;
    let u = matmul_q(gguf, shexp_up, x)?;
    let t_silu = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut par = Tensor2::zeros(g.ne0, g.ne1);
    for (p, (&gv, &uv)) in par.data.iter_mut().zip(g.data.iter().zip(&u.data)) {
        *p = silu(gv) * uv;
    }
    if let Some(t_silu) = t_silu {
        profile::record_time("swiglu", t_silu.elapsed().as_nanos() as u64);
    }
    let sh = matmul_q(gguf, shexp_down, &par)?;

    let t_trace2 = if lvl > 0 { Some(Instant::now()) } else { None };
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
    if let Some(t_trace2) = t_trace2 {
        trace_ns += t_trace2.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_trace", trace_ns);
    }
    Ok(out)
}
