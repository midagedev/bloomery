//! MoE FFN: `ffn_norm-N` in, `ffn_out-N` out, for N >= 1.
//!
//! Dispatch is expert-bucketed, not per-token — `docs/research/quant-decode-efficiency.md`
//! §Q6 decision 2, and it cannot be retrofitted. Build the bucket table first, then one
//! matmul per bucket over the tokens routed to it, then scatter back through the inverse
//! permutation and weight. This mirrors what ik's own CPU `mul_mat_id` does
//! (`matrix_row_counts`/`matrix_rows` group rows by expert before any dot runs —
//! ggml.c:18258).
//!
//! The other structural line this module holds: only routed experts are read, never the
//! whole stack. `docs/research/mistralrs-prior-art.md` §4.4 measures the failure mode — mistral.rs
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

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use gguf::{GgmlType, Gguf, Split, TensorInfo};

use crate::ModelError;
use crate::ops::{
    self, DEFER_MAX_COLS, ExpertForm, ExpertInput, ExpertStack, GroupClaims, GroupInput, Lanes,
    PostNorm, R8Stack, RowLayout, ShardTensor, StackTier, Tensor2, Tensor2View, UnionCall,
    UnionPlanView, UnionSlabs, UnionStack, WarmSpan, Weight, defers, matmul_q, matmul_q_group,
    matmul_q_group_into_on, matmul_q_group_swiglu, matmul_q_group_swiglu_into_on,
};
use crate::profile;
use crate::r8file::{R8Error, R8Source};

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

/// What the last `route_with`/`moe_ffn_with` call computed, in the oracle's own layouts, so the
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
    /// Intermediates of the last `route_with`/`moe_ffn_with` call on this thread
    /// (see [`MoETrace`]).
    static TRACE: RefCell<Option<MoETrace>> = const { RefCell::new(None) };

    /// Bit `e % 64` of word `e / 64` set = expert `e`'s stacked weights were actually
    /// read by the last `moe_ffn_with` call on this thread. Set where the bytes are
    /// fetched, not where the bucket table says they should be, so the structural gate
    /// measures reality. One word per 64 experts of the file.
    static TOUCHED: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Opt-in for the [`MoETrace`] capture: filling the trace vectors rides every
/// MoE block of every step, so the default is off and the gates that read
/// [`last_trace`] switch it on. One relaxed load per `moe_ffn_with` call.
static TRACE_ON: AtomicBool = AtomicBool::new(false);

/// Switch the per-call [`MoETrace`] capture on or off (off by default). The
/// trace is gate bookkeeping, not decode work.
pub fn set_trace_enabled(on: bool) {
    TRACE_ON.store(on, Relaxed);
}

fn trace_on() -> bool {
    TRACE_ON.load(Relaxed)
}

/// The trace of the last `route_with`/`moe_ffn_with` call on this thread, if any —
/// present only while [`set_trace_enabled`] is on.
pub fn last_trace() -> Option<MoETrace> {
    TRACE.with(|t| t.borrow().clone())
}

/// The experts actually dequantized by the last `moe_ffn_with` call on this
/// thread, ascending. Empty after `route_with` alone — routing reads the
/// router, never the expert stacks.
pub fn last_touched_experts() -> Vec<usize> {
    TOUCHED.with(|t| {
        t.borrow()
            .iter()
            .enumerate()
            .flat_map(|(w, &bits)| {
                (0..64)
                    .filter(move |b| (bits >> b) & 1 == 1)
                    .map(move |b| w * 64 + b)
            })
            .collect()
    })
}

/// The hyperparameters this module reads from the file, never from literals.
/// Part of [`MoeBlockPlan`], so its checks run at load.
pub struct Meta {
    pub n_expert: usize,
    pub n_used: usize,
    /// `expert_feed_forward_length` — the per-expert hidden width.
    pub ff: usize,
    /// `expert_weights_scale`. Applied to every router weight; 1.0 in this model, so
    /// the multiply is an exact no-op (IEEE: x * 1.0 == x).
    pub scale: f32,
}

impl Meta {
    pub fn read(gguf: &Gguf) -> Result<Meta, ModelError> {
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
        if n_expert == 0 || n_used == 0 || n_used > n_expert || ff == 0 {
            return Err(ModelError::Shape {
                what: "moe metadata",
                want_ne0: 1,
                want_ne1: n_used.min(n_expert),
                got_ne0: n_expert,
                got_ne1: ff,
            });
        }
        let scale = gguf.arch_get_f32("expert_weights_scale").unwrap_or(1.0);
        Ok(Meta {
            n_expert,
            n_used,
            ff,
            scale,
        })
    }
}

/// Route every token to its `n_used` highest-probability experts over an
/// already-resolved router weight: gate matmul, softmax over all experts, top-k
/// by (probability desc, id asc).
pub(crate) fn route_with(
    gguf: &Gguf,
    gate_inp: &TensorInfo,
    x: &Tensor2,
) -> Result<Buckets, ModelError> {
    let meta = Meta::read(gguf)?;
    let (buckets, sel) = route_inner(gguf, gate_inp, x, &meta)?;
    if trace_on() {
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
    }
    Ok(buckets)
}

/// What routing computes besides the bucket table. Kept separate so
/// `moe_ffn_with` reuses the exact same routing code the gate exercises
/// through `route_with`.
struct Selection {
    logits: Vec<f32>,
    probs: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
}

fn route_inner(
    gguf: &Gguf,
    gate_inp: &TensorInfo,
    x: &Tensor2,
    meta: &Meta,
) -> Result<(Buckets, Selection), ModelError> {
    // Profiler hook (crate::profile): `moe_route` is the router's non-matmul
    // work around the hooked gate matmul — the softmax/top-k/bucket build.
    // Level-1 only, typeless.
    let lvl = profile::level();
    let mut route_ns = 0u64;
    let n_expert = meta.n_expert;
    let n_used = meta.n_used;
    let n_tokens = x.ne1;

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
        // `total_cmp` rather than `partial_cmp().unwrap()`: the probabilities are
        // `exp(·)/sum`, so neither sign of zero nor any ordering differs from the
        // partial order, and a NaN out of a broken router sorts instead of panicking.
        ranked.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
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
    let mut end = 0u32;
    offsets.push(end);
    for &c in &counts {
        end += c;
        offsets.push(end);
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
            logits: logits.into_data(),
            probs,
            ids,
            weights,
        },
    ))
}

/// The `e`-th matrix of a stacked expert tensor `{k, n, n_expert}` as a 2-D
/// `TensorInfo` into the same mapping. `gguf.data` bounds-checks the view again, so a
/// miscomputed slice errors instead of reading a neighbor expert.
pub fn expert_view(w: &TensorInfo, e: usize, n_expert: usize) -> Result<TensorInfo, ModelError> {
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

/// One MoE block's load-time plan: the router, every per-expert view of the
/// three stacks (built once, indexed by routed expert), and the shared-expert
/// trio. Built once per file by `Derived::new`, and once per call by the
/// direct-call entry the architecture exposes. All shape checks (`Meta::read`,
/// the stack checks, `expert_view`) run in [`MoeBlockPlan::build`], at load.
pub struct MoeBlockPlan {
    /// The block the plan resolves — the expert log names it.
    pub block: usize,
    pub meta: Meta,
    pub gate_inp: TensorInfo,
    /// The three routed stacks `[gate, up, down]`, `{k, n, n_expert}` each:
    /// what a union call reads, every expert an index into its stack.
    pub exps: Box<[TensorInfo; 3]>,
    /// One view per expert id, in expert order — `n_expert` entries each.
    pub gate_views: Vec<TensorInfo>,
    pub up_views: Vec<TensorInfo>,
    pub down_views: Vec<TensorInfo>,
    pub shexp_gate: TensorInfo,
    pub shexp_up: TensorInfo,
    pub shexp_down: TensorInfo,
}

/// The routed-expert weight stacks of `block`, shape-checked against the
/// metadata and the model width: gate and up map `embd -> ff`, down maps
/// `ff -> embd`, all stacked `n_expert` deep. A mis-shaped stack must error
/// before any expert runs, not stride into a neighbor expert's bytes.
fn expect_stack(w: &TensorInfo, k: usize, n: usize, n_expert: usize) -> Result<(), ModelError> {
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
}

/// The seven weights one MoE block reads, already resolved by the caller: the
/// router, the three routed-expert stacks and the shared-expert trio. Which
/// tensor each one is belongs to the architecture's `plan` module; this module
/// checks their shapes and cuts the views.
pub(crate) struct MoeTensors<'a> {
    pub gate_inp: &'a TensorInfo,
    pub gate_exps: &'a TensorInfo,
    pub up_exps: &'a TensorInfo,
    pub down_exps: &'a TensorInfo,
    pub shexp_gate: &'a TensorInfo,
    pub shexp_up: &'a TensorInfo,
    pub shexp_down: &'a TensorInfo,
}

impl MoeBlockPlan {
    /// Check the resolved stacks against the metadata and `embd`, then cut one
    /// view per expert. Every shape error this block can raise is raised here,
    /// at load, before any expert runs.
    pub(crate) fn build(
        gguf: &Gguf,
        block: usize,
        embd: usize,
        t: MoeTensors<'_>,
    ) -> Result<MoeBlockPlan, ModelError> {
        let meta = Meta::read(gguf)?;
        let n_expert = meta.n_expert;
        let ff = meta.ff;

        expect_stack(t.gate_exps, embd, ff, n_expert)?;
        expect_stack(t.up_exps, embd, ff, n_expert)?;
        expect_stack(t.down_exps, ff, embd, n_expert)?;
        let views = |stack: &TensorInfo| -> Result<Vec<TensorInfo>, ModelError> {
            (0..n_expert)
                .map(|e| expert_view(stack, e, n_expert))
                .collect()
        };
        let gate_views = views(t.gate_exps)?;
        let up_views = views(t.up_exps)?;
        let down_views = views(t.down_exps)?;

        Ok(MoeBlockPlan {
            block,
            meta,
            gate_inp: t.gate_inp.clone(),
            exps: Box::new([t.gate_exps.clone(), t.up_exps.clone(), t.down_exps.clone()]),
            gate_views,
            up_views,
            down_views,
            shexp_gate: t.shexp_gate.clone(),
            shexp_up: t.shexp_up.clone(),
            shexp_down: t.shexp_down.clone(),
        })
    }
}

/// Per-call routed-expert intermediates the scatter fills: the weighted
/// accumulation and the two trace tensors.
struct RoutedSums {
    routed: Tensor2,
    gate_par: Vec<f32>,
    down_all: Vec<f32>,
}

/// Clear this thread's touched mask to one word per 64 of the file's
/// `n_expert` experts.
fn reset_touched(n_expert: usize) {
    TOUCHED.with(|t| {
        let mut t = t.borrow_mut();
        t.clear();
        t.resize(n_expert.div_ceil(64), 0);
    });
}

/// Gather each routed expert's input columns and mark `touched` (one bit per
/// expert, cleared by the caller). An empty bucket contributes nothing — an
/// unrouted expert's bytes are never read — and the mask is set where the
/// weights are fetched, so it counts dequantized reality, not what the
/// bucket table implies.
fn gather_expert_inputs(
    x: &Tensor2,
    buckets: &Buckets,
    n_expert: usize,
    lvl: u8,
    touched: &mut [u64],
) -> (Vec<usize>, Vec<Option<Tensor2>>, u64) {
    let t_gather = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut experts: Vec<usize> = Vec::new();
    let mut xbs: Vec<Option<Tensor2>> = Vec::new();
    for e in 0..n_expert {
        if buckets.bucket(e).is_empty() {
            continue;
        }
        touched[e / 64] |= 1 << (e % 64);
        let bucket = buckets.bucket(e);
        experts.push(e);
        // A bucket that is every token in order gathers a copy of `x` — always
        // the case in decode. `None` stands for `x` itself: the batch then sees
        // ONE distinct input across those experts and quantizes it once.
        if bucket.len() == x.ne1 && bucket.iter().enumerate().all(|(i, &t)| t as usize == i) {
            xbs.push(None);
            continue;
        }
        let mut xb = Tensor2::scratch(x.ne0, bucket.len());
        for (i, &t) in bucket.iter().enumerate() {
            xb.col_mut(i).copy_from_slice(x.col(t as usize));
        }
        xbs.push(Some(xb));
    }
    let gather_ns = if let Some(t_gather) = t_gather {
        t_gather.elapsed().as_nanos() as u64
    } else {
        0
    };
    (experts, xbs, gather_ns)
}

/// Phase 1 — gate and up for every routed expert AND the shared expert, one
/// dispatch. The views interleave [gate_e0, up_e0, gate_e1, up_e1, ..] so
/// pair `2*s` is expert s's gate and `2*s + 1` its up; the shared pair
/// [shexp_gate, shexp_up] sits last, on the same `x` the router read. The
/// doubled references are free: the group quantizes each DISTINCT (input,
/// encoding) once — every routed expert and the shared trio read `ffn_norm`
/// in decode, two encodings when their weight formats differ. Prefill's
/// gathered bucket copies ride the same code with their own slots.
fn gate_up_batch(
    gguf: &Gguf,
    plan: &MoeBlockPlan,
    experts: &[usize],
    x: &Tensor2,
    xbs: &[Option<Tensor2>],
) -> Result<Vec<Tensor2>, ModelError> {
    let mut gu_ws: Vec<&gguf::TensorInfo> = Vec::with_capacity(experts.len() * 2 + 2);
    let mut gu_xs: Vec<&Tensor2> = Vec::with_capacity(experts.len() * 2 + 2);
    for &e in experts {
        gu_ws.push(&plan.gate_views[e]);
        gu_ws.push(&plan.up_views[e]);
    }
    for xb in xbs {
        let xb = xb.as_ref().unwrap_or(x);
        gu_xs.push(xb);
        gu_xs.push(xb);
    }
    gu_ws.push(&plan.shexp_gate);
    gu_ws.push(&plan.shexp_up);
    gu_xs.push(x);
    gu_xs.push(x);
    matmul_q_group(gguf, &gu_ws, &gu_xs)
}

/// Scatter, expert-ascending — the accumulation order a sequential expert
/// loop produces, so `routed` gathers its per-token sums in a fixed sequence
/// of adds regardless of the dispatch shape. The rank search and the
/// `gate_par`/`down_all` copies exist only for the trace layout (keyed by
/// `(token, rank)`) and run only when the trace is switched on.
#[allow(clippy::too_many_arguments)]
fn scatter_experts(
    buckets: &Buckets,
    sel: &Selection,
    experts: &[usize],
    pars: &[Tensor2],
    downs: &[Tensor2],
    meta: &Meta,
    sums: &mut RoutedSums,
    trace: bool,
) {
    let embd = sums.routed.ne0;
    for (slot, &e) in experts.iter().enumerate() {
        let bucket = buckets.bucket(e);
        let weights = buckets.bucket_weights(e);
        let par = &pars[slot];
        let d = &downs[slot];
        for i in 0..bucket.len() {
            let t = bucket[i] as usize;
            let w = weights[i] * meta.scale;
            if trace {
                let s = (0..meta.n_used)
                    .find(|&s| sel.ids[t * meta.n_used + s] as usize == e)
                    .expect("every bucket entry comes from its token's selection");
                sums.gate_par[t * meta.n_used * meta.ff + s * meta.ff
                    ..t * meta.n_used * meta.ff + (s + 1) * meta.ff]
                    .copy_from_slice(par.col(i));
                sums.down_all
                    [t * meta.n_used * embd + s * embd..t * meta.n_used * embd + (s + 1) * embd]
                    .copy_from_slice(d.col(i));
            }
            let acc = sums.routed.col_mut(t);
            for (a, &dv) in acc.iter_mut().zip(d.col(i)) {
                *a += w * dv;
            }
        }
    }
}

/// Debug lever: with `BLOOMERY_EXPERT_LOG=<path>` every `moe_ffn_with` call appends
/// `block<TAB>n_tokens<TAB>id,id,...` (ids in `{n_used, n_tokens}` order). Off:
/// one `OnceLock` read. It answers "which experts would k tokens share".
fn log_experts(block: usize, n_tokens: usize, ids: &[i32]) {
    use std::io::Write;
    static LOG: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> =
        std::sync::OnceLock::new();
    let log = LOG.get_or_init(|| {
        let path = std::env::var_os("BLOOMERY_EXPERT_LOG")?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("BLOOMERY_EXPERT_LOG must be writable");
        Some(std::sync::Mutex::new(file))
    });
    if let Some(file) = log {
        let ids: Vec<String> = ids.iter().map(i32::to_string).collect();
        let mut file = file.lock().expect("expert log poisoned");
        writeln!(file, "{block}\t{n_tokens}\t{}", ids.join(",")).expect("expert log write");
    }
}

/// The MoE FFN over a load-time plan: `ffn_norm-N` in, `ffn_out-N` out —
/// routed experts plus shared experts.
///
/// Routed half: route, then one gate+up dispatch for every routed expert, one
/// down dispatch, and a scatter through the inverse permutation with the
/// router weights. Shared half: the dense FFN with the plan's shared trio.
/// Batching changes which worker computes which (expert, row), never a row's
/// k-ascending accumulation; the scatter accumulates `routed`
/// expert-ascending, the order a sequential expert loop produced.
pub fn moe_ffn_with(gguf: &Gguf, plan: &MoeBlockPlan, x: &Tensor2) -> Result<Tensor2, ModelError> {
    // Profiler hook (crate::profile): the matmuls keep their own rows, so every
    // piece timer below wraps only un-hooked regions — coverage never counts a
    // nanosecond twice. `moe_setup` is the plan resolution (metadata, stacks,
    // shexp trio, resolved before the call); `moe_route` is
    // measured inside route_inner; `swiglu` per combine — caller-side for
    // multi-column inputs, inside the down dispatch (recorded once by the
    // engine) for decode; `moe_expert_io` per expert (the token gather
    // before gate/up, the rank-search/scatter after down); `moe_trace` the
    // final combine and, when the trace is on, the trace vectors.
    let lvl = profile::level();
    let mut setup_ns = 0u64;
    let mut trace_ns = 0u64;
    let n_tokens = x.ne1;
    let embd = x.ne0;
    let trace = trace_on();

    let t_setup = if lvl > 0 { Some(Instant::now()) } else { None };
    let meta = &plan.meta;
    reset_touched(meta.n_expert);
    if let Some(t_setup) = t_setup {
        setup_ns += t_setup.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_setup", setup_ns);
    }

    let (buckets, sel) = route_inner(gguf, &plan.gate_inp, x, meta)?;
    log_experts(plan.block, n_tokens, &sel.ids);

    let t_trace1 = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut sums = RoutedSums {
        routed: Tensor2::zeros(embd, n_tokens),
        gate_par: if trace {
            vec![0.0f32; meta.ff * meta.n_used * n_tokens]
        } else {
            Vec::new()
        },
        down_all: if trace {
            vec![0.0f32; embd * meta.n_used * n_tokens]
        } else {
            Vec::new()
        },
    };
    if let Some(t_trace1) = t_trace1 {
        trace_ns += t_trace1.elapsed().as_nanos() as u64;
    }

    let (experts, xbs, gather_ns) = TOUCHED
        .with(|t| gather_expert_inputs(x, &buckets, meta.n_expert, lvl, &mut t.borrow_mut()));

    // The routed half and the shared half are independent math: the gate/up
    // projections of both ride one group dispatch, and so do the down
    // projections. Regrouping only — the same ops on the same inputs, the
    // scatter still accumulates expert-ascending and the combine still adds
    // the shared output last, so no value moves.
    let gu = gate_up_batch(gguf, plan, &experts, x, &xbs)?;
    let n_e = experts.len();
    // Phase 2 — every expert's down projection and the shared expert's, one
    // dispatch whose pairs' inputs are the SwiGLU combines themselves: in
    // decode (one column) the dispatch's claimant produces and quantizes
    // each par; wider inputs keep the caller-side combine. The same swiglu a
    // sequential loop ran per expert, in the same expert order — the
    // elementwise combine, moved inside the dispatch that consumes it.
    let mut down_ws: Vec<&gguf::TensorInfo> = Vec::with_capacity(n_e + 1);
    for &e in &experts {
        down_ws.push(&plan.down_views[e]);
    }
    down_ws.push(&plan.shexp_down);
    let mut srcs: Vec<GroupInput> = Vec::with_capacity(n_e + 1);
    for slot in 0..n_e {
        srcs.push(GroupInput::Swiglu(&gu[2 * slot], &gu[2 * slot + 1]));
    }
    srcs.push(GroupInput::Swiglu(&gu[2 * n_e], &gu[2 * n_e + 1]));
    let (downs_all, pars) = matmul_q_group_swiglu(gguf, &down_ws, &srcs)?;
    // `split_last` yields (last, rest): the shared down is the group's last
    // output, the routed downs keep expert order in the slice before it — and
    // the pars carry the same order.
    let (shexp_down, downs) = downs_all
        .split_last()
        .expect("the down group always carries the shared pair");
    let routed_pars = pars.split_last().expect("one par per SwiGLU input").1;

    let t_scatter = if lvl > 0 { Some(Instant::now()) } else { None };
    scatter_experts(
        &buckets,
        &sel,
        &experts,
        routed_pars,
        downs,
        meta,
        &mut sums,
        trace,
    );
    let mut io_ns = gather_ns;
    if let Some(t_scatter) = t_scatter {
        io_ns += t_scatter.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_expert_io", io_ns);
    }

    let t_trace2 = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut out = sums.routed.clone();
    for (o, &sv) in out.data.iter_mut().zip(&shexp_down.data) {
        *o += sv;
    }

    if trace {
        TRACE.with(|t| {
            *t.borrow_mut() = Some(MoETrace {
                n_expert: meta.n_expert,
                n_used: meta.n_used,
                n_tokens,
                logits: sel.logits,
                probs: sel.probs,
                ids: sel.ids,
                weights: sel.weights,
                gate_par: sums.gate_par,
                down: sums.down_all,
                routed_out: sums.routed.data.clone(),
                shexp_out: shexp_down.data.clone(),
            })
        });
    }
    if let Some(t_trace2) = t_trace2 {
        trace_ns += t_trace2.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("moe_trace", trace_ns);
    }
    Ok(out)
}

/// A given list of routed experts of ONE token, weighted and summed into
/// `out`: `out = Σ_i w_i · down_{e_i}(silu(gate_{e_i}·x) ⊙ (up_{e_i}·x))`,
/// accumulated in list order.
///
/// The host tier of a hybrid engine calls this for the experts its card does
/// not hold. It is the decode shape of [`moe_ffn_with`]'s routed half: one
/// group dispatch for every listed expert's gate and up (`x` quantized once,
/// by the weight type's own activation rule), one for their downs with each
/// SwiGLU combine produced inside it. The weights are applied as given — the
/// caller's router has already scaled them. `x` is one column of the model
/// width and `out` holds that width; an empty list writes zeros. The list
/// holds at most the experts `scratch` was made for ([`HostScratch::new`],
/// the model's routed width), a longer one is refused by name. The
/// dispatches write into the caller's `scratch`, made at load for the
/// block's widths, so a call allocates nothing.
pub fn experts_into(
    gguf: &Gguf,
    plan: &MoeBlockPlan,
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &mut [f32],
    scratch: &mut HostScratch,
) -> Result<(), ModelError> {
    check_host_call(x, experts, out, scratch.n_used)?;
    if !scratch.fits(x.ne0, plan.meta.ff) {
        return Err(ModelError::Shape {
            what: "host experts: the scratch must be made for the block's widths",
            want_ne0: x.ne0,
            want_ne1: plan.meta.ff,
            got_ne0: scratch.embd,
            got_ne1: scratch.ff,
        });
    }
    out.fill(0.0);
    if experts.is_empty() {
        return Ok(());
    }
    // The plan's views are headers of `gguf`, the one file this model has.
    let view = |views: &[TensorInfo], e: u32| -> Result<Weight<'_>, ModelError> {
        Weight::in_file(gguf, expert_of(views, e, plan.block)?)
    };
    let weights = |e: u32| -> Result<ExpertLegs<'_>, ModelError> {
        Ok(ExpertLegs::Gated([
            view(&plan.gate_views, e)?,
            view(&plan.up_views, e)?,
            view(&plan.down_views, e)?,
        ]))
    };
    serve(weights, LegRun::flat(), x, experts, out, scratch)
}

/// Expert `e`'s view in one of a block plan's per-expert view lists; an id
/// past the list is the caller's error, named with the block.
fn expert_of(views: &[TensorInfo], e: u32, block: usize) -> Result<&TensorInfo, ModelError> {
    views
        .get(e as usize)
        .ok_or_else(|| ModelError::MissingTensor(format!("expert {e} of block {block}")))
}

/// A host call's three shape conditions, each its own error: `x` is one
/// token, `out` holds its width, and the list holds at most the `n_used`
/// experts its scratch was made for.
fn check_host_call(
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &[f32],
    n_used: usize,
) -> Result<(), ModelError> {
    if x.ne1 != 1 {
        return Err(ModelError::Shape {
            what: "host experts: x must be one token",
            want_ne0: x.ne0,
            want_ne1: 1,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }
    if out.len() != x.ne0 {
        return Err(ModelError::Shape {
            what: "host experts: out must hold x's width",
            want_ne0: x.ne0,
            want_ne1: 1,
            got_ne0: out.len(),
            got_ne1: 1,
        });
    }
    if experts.len() > n_used {
        return Err(ModelError::Shape {
            what: "host experts: at most the experts the scratch was made for",
            want_ne0: n_used,
            want_ne1: 1,
            got_ne0: experts.len(),
            got_ne1: 1,
        });
    }
    Ok(())
}

/// `v` emptied and handed back as a list of `B` over the same allocation.
/// `A` and `B` differ only in a lifetime, so the collect reuses the buffer
/// and keeps its room: a call's view lists borrow for the call alone, and
/// between calls the scratch holds them empty with the room made at load.
fn relist<A, B>(mut v: Vec<A>) -> Vec<B> {
    const {
        assert!(size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>());
    }
    v.clear();
    v.into_iter()
        .map(|_| -> B { unreachable!("the list was cleared") })
        .collect()
}

/// The blocks one host call writes — the gate and up outputs, the SwiGLU
/// combines and the down outputs of up to `n_used` experts, the model's
/// routed width — and the room of the call's view lists, made once for a
/// layer shape and held across calls, so a call allocates nothing. After a
/// call, [`HostScratch::gate`] and its siblings read what it computed for
/// the list's `i`-th expert.
pub struct HostScratch {
    embd: usize,
    ff: usize,
    n_used: usize,
    /// `[gate_0, up_0, gate_1, up_1, ..]`, `{ff, 1}` each.
    gate_up: Vec<Tensor2>,
    /// The combines, `{ff, 1}` each.
    pars: Vec<Tensor2>,
    /// The down outputs, `{embd, 1}` each.
    downs: Vec<Tensor2>,
    /// Per-expert activation scales, restored before weighting the down outputs.
    scales: Vec<f32>,
    /// The call's gate and up matrices, interleaved; empty between calls.
    gu_w: Vec<Weight<'static>>,
    /// The call's down matrices; empty between calls.
    down_w: Vec<Weight<'static>>,
    /// The gate/up dispatch's inputs, `x` once a pair; empty between calls.
    xs: Vec<&'static Tensor2>,
    /// The down dispatch's inputs, the combines; empty between calls.
    srcs: Vec<GroupInput<'static>>,
    /// The call's expert ids as the tier's `ensure` takes them, reused
    /// across calls so a paged call allocates nothing.
    tier_ids: Vec<u32>,
}

impl HostScratch {
    /// Blocks for lists of up to `n_used` experts that map
    /// `embd -> ff -> embd`; a width of none is a named error.
    pub fn new(embd: usize, ff: usize, n_used: usize) -> Result<HostScratch, ModelError> {
        if n_used == 0 {
            return Err(ModelError::Shape {
                what: "host scratch: at least one routed expert a token",
                want_ne0: 1,
                want_ne1: 1,
                got_ne0: 0,
                got_ne1: 1,
            });
        }
        let blocks =
            |count: usize, rows: usize| (0..count).map(|_| Tensor2::zeros(rows, 1)).collect();
        Ok(HostScratch {
            embd,
            ff,
            n_used,
            gate_up: blocks(2 * n_used, ff),
            pars: blocks(n_used, ff),
            downs: blocks(n_used, embd),
            scales: Vec::new(),
            gu_w: Vec::with_capacity(2 * n_used),
            down_w: Vec::with_capacity(n_used),
            xs: Vec::with_capacity(2 * n_used),
            srcs: Vec::with_capacity(n_used),
            tier_ids: Vec::new(),
        })
    }

    /// Allocate protection scales at load; direct quantization needs none.
    pub fn set_expert_input(&mut self, policy: ExpertInput) {
        match policy {
            ExpertInput::Direct => self.scales = Vec::new(),
            ExpertInput::AmaxF16 => self.scales.resize(self.n_used, 1.0),
        }
    }

    /// The policy this scratch has load-allocated scale storage for.
    pub fn expert_input(&self) -> ExpertInput {
        if self.scales.is_empty() {
            ExpertInput::Direct
        } else {
            ExpertInput::AmaxF16
        }
    }

    fn fits(&self, embd: usize, ff: usize) -> bool {
        (self.embd, self.ff) == (embd, ff)
    }

    /// The feed-forward width the blocks were made for.
    pub fn ff(&self) -> usize {
        self.ff
    }

    /// The most experts one call through this scratch takes.
    pub fn n_used(&self) -> usize {
        self.n_used
    }

    /// The gate output of the last call's `i`-th expert (a relu² layer has
    /// none: its slot holds no output of the call).
    pub fn gate(&self, i: usize) -> &Tensor2 {
        &self.gate_up[2 * i]
    }

    /// The up output of the last call's `i`-th expert.
    pub fn up(&self, i: usize) -> &Tensor2 {
        &self.gate_up[2 * i + 1]
    }

    /// The SwiGLU combine the last call's `i`-th expert's down read.
    pub fn par(&self, i: usize) -> &Tensor2 {
        &self.pars[i]
    }

    /// The down output of the last call's `i`-th expert, before its weight.
    pub fn down(&self, i: usize) -> &Tensor2 {
        &self.downs[i]
    }
}

/// How one host leg of one token runs: the experts' form (which stacks the
/// leg reads and their combine), how the leg's rows are laid over the pool,
/// and the norm each expert's down takes before its weight when the layer has
/// one ([`HostLayer::set_post_norm`]).
#[derive(Clone, Copy)]
struct LegRun<'n> {
    form: ExpertForm,
    input: ExpertInput,
    lanes: Lanes,
    post: Option<&'n PostNorm>,
}

impl LegRun<'_> {
    /// The gated leg without a clamp, rows in the pool's flat lanes, the
    /// plain weighted sum.
    fn flat() -> LegRun<'static> {
        LegRun {
            form: ExpertForm::Gated { limit: 0.0 },
            input: ExpertInput::Direct,
            lanes: Lanes::Flat,
            post: None,
        }
    }
}

/// One expert's stacks as its layer's form reads them, resolved by the
/// caller from wherever the layer holds them.
enum ExpertLegs<'w> {
    /// `[gate, up, down]`.
    Gated([Weight<'w>; 3]),
    /// The up and the down.
    Relu2 { up: Weight<'w>, down: Weight<'w> },
}

/// The host leg of one token: each listed expert's stacks from `weights`,
/// resolved in list order into the scratch's view lists, then
/// [`serve_resolved`]. The lists go back to the scratch, emptied, whether the
/// call succeeds or not. The caller has checked the list against the
/// scratch's width, so no push grows a list.
fn serve<'w>(
    weights: impl Fn(u32) -> Result<ExpertLegs<'w>, ModelError>,
    run: LegRun<'_>,
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &mut [f32],
    s: &mut HostScratch,
) -> Result<(), ModelError> {
    let mut gu: Vec<Weight<'w>> = relist(std::mem::take(&mut s.gu_w));
    let mut down: Vec<Weight<'w>> = relist(std::mem::take(&mut s.down_w));
    let r = experts
        .iter()
        .try_for_each(|&(e, _)| {
            match weights(e)? {
                ExpertLegs::Gated([g, u, d]) => {
                    gu.extend([g, u]);
                    down.push(d);
                }
                ExpertLegs::Relu2 { up, down: d } => {
                    gu.push(up);
                    down.push(d);
                }
            }
            Ok(())
        })
        .and_then(|()| serve_resolved(&gu, &down, run, x, experts, out, s));
    s.gu_w = relist(gu);
    s.down_w = relist(down);
    r
}

/// The host leg of one token over resolved weights: gated, `gu` holds each
/// listed expert's gate and up interleaved; a relu² leg holds its ups alone.
/// One group dispatch for the first stacks (`x` quantized once), one for
/// every down with each combine — the form's — produced inside it, then the
/// weighted sum in list order into `out` (already zeroed). Profile level 1
/// records the leg in five parts that do not overlap: `host_x_quant` (x's
/// quantization), `host_gate_up` (or `host_up`, a relu² leg's single stack;
/// the rows, per weight type), `host_h_quant` (the combines and their
/// quantization), `host_down` (the down rows, per weight type) and
/// `host_sum`.
fn serve_resolved(
    gu: &[Weight<'_>],
    down: &[Weight<'_>],
    run: LegRun<'_>,
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &mut [f32],
    s: &mut HostScratch,
) -> Result<(), ModelError> {
    let n = experts.len();
    if run.input == ExpertInput::AmaxF16 && s.scales.len() < n {
        return Err(ModelError::Metadata {
            key: "host expert input".to_string(),
            detail: "AmaxF16 requires HostScratch::set_expert_input at load".to_string(),
        });
    }
    let lvl = profile::level();
    let HostScratch {
        gate_up,
        pars,
        downs,
        scales,
        xs,
        srcs,
        ..
    } = s;
    // A gated leg's first dispatch interleaves [gate, up] pairs — two blocks
    // and two reads of `x` an expert; a relu² leg's holds the ups alone, and
    // moves them to the up slots once the down has read them.
    let (first, up_site) = match run.form {
        ExpertForm::Gated { .. } | ExpertForm::GatedOai { .. } => (2 * n, "host_gate_up"),
        ExpertForm::Relu2 => (n, "host_up"),
    };
    let mut xl: Vec<&Tensor2> = relist(std::mem::take(xs));
    xl.extend(std::iter::repeat_n(x, first));
    let gt = matmul_q_group_into_on(run.lanes, up_site, gu, &xl, &mut gate_up[..first]);
    *xs = relist(xl);
    let gt = gt?;
    let mut sl: Vec<GroupInput<'_>> = relist(std::mem::take(srcs));
    let dt = if run.input == ExpertInput::AmaxF16 {
        for i in 0..n {
            let (gate, up) = match run.form {
                ExpertForm::Gated { .. } | ExpertForm::GatedOai { .. } => {
                    (gate_up[2 * i].col(0), gate_up[2 * i + 1].col(0))
                }
                ExpertForm::Relu2 => (&[][..], gate_up[i].col(0)),
            };
            run.form.combine_into(gate, up, pars[i].col_mut(0));
            scales[i] = run.input.protect(pars[i].col_mut(0));
        }
        sl.extend(pars[..n].iter().map(GroupInput::Ready));
        matmul_q_group_swiglu_into_on(run.lanes, "host_down", down, &sl, &mut downs[..n], &mut [])
    } else {
        match run.form {
            ExpertForm::Gated { limit } => {
                for [gate, up] in gate_up[..2 * n].as_chunks::<2>().0 {
                    sl.push(GroupInput::SwigluClamp(gate, up, limit));
                }
            }
            ExpertForm::GatedOai { alpha, limit } => {
                for [gate, up] in gate_up[..2 * n].as_chunks::<2>().0 {
                    sl.push(GroupInput::SwigluOai(gate, up, alpha, limit));
                }
            }
            ExpertForm::Relu2 => {
                for up in &gate_up[..n] {
                    sl.push(GroupInput::Relu2(up));
                }
            }
        }
        matmul_q_group_swiglu_into_on(
            run.lanes,
            "host_down",
            down,
            &sl,
            &mut downs[..n],
            &mut pars[..n],
        )
    };
    *srcs = relist(sl);
    let dt = dt?;
    if run.input == ExpertInput::AmaxF16 {
        for (d, &scale) in downs[..n].iter_mut().zip(&scales[..n]) {
            for v in d.col_mut(0) {
                *v *= scale;
            }
        }
    }
    if let ExpertForm::Relu2 = run.form {
        // Back to back from the top: expert `i`'s up goes to slot `2 * i + 1`,
        // where `HostScratch::up` reads it, and no block moves twice.
        for i in (0..n).rev() {
            gate_up.swap(i, 2 * i + 1);
        }
    }
    let t_sum = if lvl > 0 { Some(Instant::now()) } else { None };
    for (&(_, w), d) in experts.iter().zip(&downs[..n]) {
        match run.post {
            Some(post) => post.add_normed(out, w, d.col(0)),
            None => {
                for (o, &dv) in out.iter_mut().zip(d.col(0)) {
                    *o += w * dv;
                }
            }
        }
    }
    if let Some(t_sum) = t_sum {
        let sum_ns = t_sum.elapsed().as_nanos() as u64;
        profile::record_time("host_x_quant", gt.stage_ns);
        profile::record_time("host_h_quant", dt.stage_ns);
        profile::record_time("host_sum", sum_ns);
    }
    Ok(())
}

/// The most token columns a batch's union scratch can be made for
/// ([`UnionScratch::new_routed`]): one prefill ubatch. A call takes at most the
/// columns its scratch was made for ([`experts_union_into`],
/// [`HostLayer::experts_union_into`]); a group tail's scratch is the one
/// exception ([`UnionScratch::new_tail`]).
pub const UNION_MAX_COLS: usize = 512;

/// The most batches of [`UNION_MAX_COLS`] a group tail's scratch spans
/// ([`UnionScratch::new_tail`]): the batches that pass one layer together
/// before its tail experts run once over all their columns.
pub const UNION_TAIL_MAX_GROUPS: usize = 8;

/// The most columns a group tail's scratch can be made for.
pub const UNION_TAIL_MAX_COLS: usize = UNION_TAIL_MAX_GROUPS * UNION_MAX_COLS;

/// Experts of the NVMe tier's arena one batch of a tiered union call holds:
/// the unit a claim, a fill and a pin move in ([`HostLayer::experts_union_into`]).
/// A call whose arena ids are over the layer's slots less two batches runs in
/// batches, and the layer's slots must hold two of them and the lane's slots.
/// The placement's value, which its arena floor counts.
pub const UNION_BATCH_IDS: usize = crate::placement::UNION_BATCH_IDS as usize;

/// The listed slots — (expert, column) pairs, a column once per expert — of
/// one batch at its worst, every column listing `n_used` experts: the least
/// slot budget a group tail's scratch for that routed width takes, so any
/// batch's columns fit it.
#[must_use]
pub const fn union_batch_slots(n_used: usize) -> usize {
    UNION_MAX_COLS * n_used
}

/// A union call's plan, in storage its scratch made at load: the distinct
/// experts in ascending id, and per expert the columns that list it,
/// ascending, each once. Expert `d`'s columns are `cols[off[d]..off[d + 1]]`,
/// and the same range names its slots in the scratch's slabs.
struct UnionPlan {
    ids: Vec<u32>,
    off: Vec<usize>,
    cols: Vec<usize>,
    /// Every listed `(expert, column)`, sorted to build the rest.
    pairs: Vec<(u32, usize)>,
}

impl UnionPlan {
    /// Room for `cols` columns of `per_list` experts each.
    fn with_room(cols: usize, per_list: usize) -> UnionPlan {
        let slots = cols * per_list;
        UnionPlan {
            ids: Vec::with_capacity(slots),
            off: Vec::with_capacity(slots + 1),
            cols: Vec::with_capacity(slots),
            pairs: Vec::with_capacity(slots),
        }
    }

    /// Room for the plan of one batch of a call that holds at most `slots`
    /// slots: no `pairs`, a batch's plan is cut from the whole's
    /// ([`UnionPlan::cut`]).
    fn batch_room(slots: usize) -> UnionPlan {
        UnionPlan {
            ids: Vec::with_capacity(slots),
            off: Vec::with_capacity(slots + 1),
            cols: Vec::with_capacity(slots),
            pairs: Vec::new(),
        }
    }

    /// The plan of `whole`'s experts `ds` (indices into `whole`, any order)
    /// alone, in that order: compact, its slots from 0, each expert's columns
    /// as `whole` lists them.
    fn cut(&mut self, whole: &UnionPlan, ds: &[u32]) {
        self.ids.clear();
        self.off.clear();
        self.cols.clear();
        for &d in ds {
            let d = d as usize;
            self.ids.push(whole.ids[d]);
            self.off.push(self.cols.len());
            self.cols
                .extend_from_slice(&whole.cols[whole.off[d]..whole.off[d + 1]]);
        }
        self.off.push(self.cols.len());
    }

    /// The plan of `lists` in place, or the named error of a list past
    /// `per_list` experts, the routed width a scratch was made for. The caller has checked the column count against
    /// the room, so no push grows a buffer.
    fn build(&mut self, lists: &[&[(u32, f32)]], per_list: usize) -> Result<(), ModelError> {
        if let Some(list) = lists.iter().find(|l| l.len() > per_list) {
            return Err(ModelError::Shape {
                what: "host union: at most the scratch's routed width of experts per column",
                want_ne0: per_list,
                want_ne1: 1,
                got_ne0: list.len(),
                got_ne1: 1,
            });
        }
        self.pairs.clear();
        for (j, list) in lists.iter().enumerate() {
            self.pairs.extend(list.iter().map(|&(e, _)| (e, j)));
        }
        self.pairs.sort_unstable();
        self.ids.clear();
        self.off.clear();
        self.cols.clear();
        for (i, &(e, j)) in self.pairs.iter().enumerate() {
            let prev = i.checked_sub(1).map(|p| self.pairs[p]);
            if prev.is_none_or(|(pe, _)| pe != e) {
                self.ids.push(e);
                self.off.push(self.cols.len());
            }
            // A list that names an expert twice gives its column once.
            if prev != Some((e, j)) {
                self.cols.push(j);
            }
        }
        self.off.push(self.cols.len());
        Ok(())
    }

    /// The distinct experts.
    fn n(&self) -> usize {
        self.ids.len()
    }

    /// Whether the plan runs through slabs of `slots` slots.
    fn fits(&self, slots: usize) -> bool {
        self.cols.len() <= slots
    }

    /// The plan as the passes read it.
    fn view(&self) -> UnionPlanView<'_> {
        UnionPlanView {
            ids: &self.ids,
            off: &self.off,
            cols: &self.cols,
        }
    }
}

/// The order a tiered call serves its experts in and where each one's downs
/// lie in the store, in storage the scratch made at load: the batches of
/// [`HostLayer::experts_union_into`] over a layer with an arena.
struct UnionBatches {
    /// The plan's experts, as indices into the plan, in serving order: the
    /// plan's host segment's first (the file mapping's, one batch), then the
    /// arena's keep set, then the rest, each run ascending.
    order: Vec<u32>,
    /// The ids of `order`, for the claims.
    ids: Vec<u32>,
    /// Where each batch ends in `order`.
    ends: Vec<usize>,
    /// Plan expert `d`'s first slot in the store: the store holds the batches
    /// one after the other, each as its own compact plan.
    slot: Vec<usize>,
    /// The batch being served, as a plan of its own.
    plan: UnionPlan,
}

/// How [`UnionBatches::lay_out`] cut a call: the batches of the host segment
/// (none or one) and of the keep set, the rest being the other arena ids'.
struct BatchCut {
    host: usize,
    keep: usize,
}

impl UnionBatches {
    /// Room for a call of at most `slots` slots.
    fn with_room(slots: usize) -> UnionBatches {
        UnionBatches {
            order: Vec::with_capacity(slots),
            ids: Vec::with_capacity(slots),
            ends: Vec::with_capacity(slots / UNION_BATCH_IDS + 4),
            slot: Vec::with_capacity(slots),
            plan: UnionPlan::batch_room(slots),
        }
    }

    /// `plan`'s experts cut into batches of `batch`: first the ones `host`
    /// holds, all at once, then the arena's — the `keep` of them with the most
    /// columns (the ties to the lower id) in batches of their own, the rest
    /// after, each run ascending by id. The caller has checked that the
    /// arena's ids are more than `keep`, and `keep` is at least one.
    fn lay_out(
        &mut self,
        plan: &UnionPlan,
        host: &std::ops::Range<u32>,
        keep: usize,
        batch: usize,
    ) -> BatchCut {
        self.order.clear();
        self.ends.clear();
        let is_host = |d: usize| host.contains(&plan.ids[d]);
        self.order
            .extend((0..plan.n()).filter(|&d| is_host(d)).map(|d| d as u32));
        let host_end = self.order.len();
        if host_end > 0 {
            self.ends.push(host_end);
        }
        self.order
            .extend((0..plan.n()).filter(|&d| !is_host(d)).map(|d| d as u32));
        let arena = &mut self.order[host_end..];
        assert!(
            0 < keep && keep < arena.len(),
            "host union: a keep set of at least one, under the call's arena ids"
        );
        let columns = |d: u32| plan.off[d as usize + 1] - plan.off[d as usize];
        arena.select_nth_unstable_by_key(keep - 1, |&d| (std::cmp::Reverse(columns(d)), d));
        let (kept, rest) = arena.split_at_mut(keep);
        kept.sort_unstable();
        rest.sort_unstable();
        let keep_end = host_end + keep;
        for (from, to) in [(host_end, keep_end), (keep_end, self.order.len())] {
            let mut at = from;
            while at < to {
                at = (at + batch).min(to);
                self.ends.push(at);
            }
        }
        self.ids.clear();
        self.ids
            .extend(self.order.iter().map(|&d| plan.ids[d as usize]));
        BatchCut {
            host: usize::from(host_end > 0),
            keep: keep.div_ceil(batch),
        }
    }

    /// `plan`'s experts cut for a step served in two groups: the early group
    /// — the ones `host` holds, then the arena's `lent` ones — and the late
    /// one, the arena's others, each run ascending by id. `lent` is the tier's
    /// answer for an arena id (held by the claim, or in flight). It is asked
    /// once per id and the answer is the id's class: a fill published between
    /// two asks would leave the id in neither group and the late span short
    /// of the plan. The late ids wait in `ids` until `order` takes them.
    /// Returns the early group's host experts and its all.
    fn split_step(
        &mut self,
        plan: &UnionPlan,
        host: &std::ops::Range<u32>,
        lent: impl Fn(u32) -> bool,
    ) -> (usize, usize) {
        let (order, ids) = (&mut self.order, &mut self.ids);
        order.clear();
        ids.clear();
        let ds = || plan.ids.iter().enumerate().map(|(d, &e)| (d as u32, e));
        order.extend(ds().filter(|(_, e)| host.contains(e)).map(|(d, _)| d));
        let n_host = order.len();
        for (d, e) in ds().filter(|(_, e)| !host.contains(e)) {
            if lent(e) {
                order.push(d);
            } else {
                ids.push(d);
            }
        }
        let early = order.len();
        order.extend_from_slice(ids);
        ids.clear();
        ids.extend(order.iter().map(|&d| plan.ids[d as usize]));
        (n_host, early)
    }
}

/// The slabs a union call writes, made once for a layer shape and a column
/// count and held across calls, so a call allocates nothing: `x` quantized,
/// and per slot — a listed (expert, column) pair, expert `d`'s slots at the
/// plan's `off[d]..off[d + 1]` — its gate and up outputs, its combine
/// quantized in the down's encoding and its down output, kept until the
/// list-order sums at the end; the plan's buffers; and the claim states of a
/// narrow call's down pass, one per expert its columns can list.
///
/// For `cols` columns, `slots` slots and `embd` and `ff` wide experts, it
/// holds `4 · (2 · ff + embd) · slots` bytes of gate/up and down slabs,
/// `slots` combine columns and twice `cols` columns of `x` (the gate's
/// encoding and the up's) of `ops::col_bytes_max` bytes each (a little over
/// `ff` and `embd` bytes), and a few words per slot of plan.
///
/// A batch's scratch ([`UnionScratch::new_routed`]) takes up to
/// [`UNION_MAX_COLS`] columns and holds their every slot: each column's list
/// is at most the routed width it was made for. A group tail's ([`UnionScratch::new_tail`]) takes
/// up to [`UNION_TAIL_MAX_COLS`] and holds a slot budget instead. A call
/// whose plan holds no more slots than the scratch runs as one call; one
/// that holds more runs as consecutive calls of [`UNION_MAX_COLS`] columns,
/// each of which fits, and is counted ([`UnionScratch::cut_calls`]). Either
/// way a column's bits are the ones any call that lists it the same way
/// writes.
pub struct UnionScratch {
    embd: usize,
    ff: usize,
    max_cols: usize,
    /// The most slots one call's plan may hold.
    max_slots: usize,
    /// The most experts one column's list may name.
    per_list: usize,
    /// Calls that did not fit the slabs and ran cut into calls of
    /// [`UNION_MAX_COLS`] columns.
    cut_calls: u64,
    /// Passes the calls completed ([`UnionScratch::passes`]).
    passes: u64,
    /// `x` in the gate's encoding, `ops::col_bytes_max(embd)` bytes a column.
    xq: Vec<u8>,
    /// `x` in the up's encoding, the same room, for an up whose rows read
    /// other bytes than the gate's.
    xq_up: Vec<u8>,
    /// Slot columns of `ff`: expert `d`'s gate at slots `2·off[d]` on, its up
    /// at the `m` after them (`m` its columns).
    gu: Vec<f32>,
    /// Slot `q`'s combine in the down's encoding at `q · cb`, `cb` the down's
    /// column bytes — at most `ops::col_bytes_max(ff)`.
    qc: Vec<u8>,
    /// Slot `q`'s down output at `store[q·embd..(q + 1)·embd]`.
    store: Vec<f32>,
    /// Per-slot activation scales; claims publish them before the down reads.
    scales: Vec<AtomicU32>,
    plan: UnionPlan,
    /// A narrow call's claim state per distinct expert:
    /// `min(max_cols, DEFER_MAX_COLS) · per_list`, every expert such a call
    /// can list.
    claims: Vec<AtomicU8>,
    /// A step union's distinct ids the NVMe tier lends, ascending
    /// ([`HostLayer::experts_step_union_into`]): room for every slot.
    tier_ids: Vec<u32>,
    /// A tiered call's batches and the store's layout for them.
    batches: UnionBatches,
}

impl UnionScratch {
    /// Slabs for a batch's calls, up to `cols` columns over experts that map
    /// `embd -> ff -> embd`, each column listing at most `n_used` experts —
    /// the model's routed width — so it holds `n_used · cols` slots; a call
    /// with a longer list is refused by name. `cols` outside
    /// `1..=UNION_MAX_COLS` and a width of none are named errors.
    pub fn new_routed(
        embd: usize,
        ff: usize,
        cols: usize,
        n_used: usize,
    ) -> Result<UnionScratch, ModelError> {
        if cols == 0 || cols > UNION_MAX_COLS {
            return Err(ModelError::Shape {
                what: "host union scratch: 1..=UNION_MAX_COLS columns",
                want_ne0: embd,
                want_ne1: UNION_MAX_COLS,
                got_ne0: embd,
                got_ne1: cols,
            });
        }
        if n_used == 0 {
            return Err(ModelError::Shape {
                what: "host union scratch: at least one expert per column",
                want_ne0: 1,
                want_ne1: cols,
                got_ne0: n_used,
                got_ne1: cols,
            });
        }
        Ok(UnionScratch::with_caps(
            embd,
            ff,
            cols,
            cols * n_used,
            n_used,
        ))
    }

    /// Slabs for a group tail's call: the columns of `groups` batches of
    /// [`UNION_MAX_COLS`] in one call, each column listing at most `n_used`
    /// experts, so each expert it lists is read once for all of them, over
    /// `slots` slots — the budget, from [`union_batch_slots`] of `n_used` (so
    /// a batch's columns always fit) to every slot of the columns. `groups`
    /// outside `1..=UNION_TAIL_MAX_GROUPS`, a width of none or a budget
    /// outside that range is a named error. A call through it may take fewer
    /// columns (a short last batch), and batch calls fit it too.
    pub fn new_tail(
        embd: usize,
        ff: usize,
        groups: usize,
        slots: usize,
        n_used: usize,
    ) -> Result<UnionScratch, ModelError> {
        if groups == 0 || groups > UNION_TAIL_MAX_GROUPS {
            return Err(ModelError::Shape {
                what: "host union tail scratch: 1..=UNION_TAIL_MAX_GROUPS batches",
                want_ne0: embd,
                want_ne1: UNION_TAIL_MAX_GROUPS,
                got_ne0: embd,
                got_ne1: groups,
            });
        }
        if n_used == 0 {
            return Err(ModelError::Shape {
                what: "host union tail scratch: at least one expert per column",
                want_ne0: 1,
                want_ne1: groups,
                got_ne0: n_used,
                got_ne1: groups,
            });
        }
        let cols = groups * UNION_MAX_COLS;
        if !(union_batch_slots(n_used)..=cols * n_used).contains(&slots) {
            return Err(ModelError::Shape {
                what: "host union tail scratch: one batch's slots at worst ..= every column's slots",
                want_ne0: union_batch_slots(n_used),
                want_ne1: cols * n_used,
                got_ne0: slots,
                got_ne1: groups,
            });
        }
        Ok(UnionScratch::with_caps(embd, ff, cols, slots, n_used))
    }

    /// `cols` columns a call of lists of at most `per_list` experts over
    /// `slots` slots.
    fn with_caps(
        embd: usize,
        ff: usize,
        cols: usize,
        slots: usize,
        per_list: usize,
    ) -> UnionScratch {
        let (x_col, c_col) = (ops::col_bytes_max(embd), ops::col_bytes_max(ff));
        UnionScratch {
            embd,
            ff,
            max_cols: cols,
            max_slots: slots,
            per_list,
            cut_calls: 0,
            passes: 0,
            xq: vec![0; cols * x_col],
            xq_up: vec![0; cols * x_col],
            gu: vec![0.0; 2 * slots * ff],
            qc: vec![0; slots * c_col],
            store: vec![0.0; slots * embd],
            scales: Vec::new(),
            plan: UnionPlan::with_room(cols, per_list),
            claims: (0..cols.min(DEFER_MAX_COLS) * per_list)
                .map(|_| AtomicU8::new(0))
                .collect(),
            tier_ids: Vec::with_capacity(cols * per_list),
            batches: UnionBatches::with_room(slots),
        }
    }

    /// Allocate protection scales at load; direct quantization needs none.
    pub fn set_expert_input(&mut self, policy: ExpertInput) {
        self.scales = match policy {
            ExpertInput::Direct => Vec::new(),
            ExpertInput::AmaxF16 => (0..self.max_slots)
                .map(|_| AtomicU32::new(1.0f32.to_bits()))
                .collect(),
        };
    }

    /// The most columns a call through this scratch takes.
    pub fn max_cols(&self) -> usize {
        self.max_cols
    }

    /// The feed-forward width the slabs were made for: a layer of that
    /// width or less serves through them.
    pub fn ff(&self) -> usize {
        self.ff
    }

    /// The most slots one call runs as a whole: a batch's scratch holds its
    /// columns' every slot, a tail's its budget.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Calls through this scratch that did not fit it and ran as calls of
    /// [`UNION_MAX_COLS`] columns, since it was made. A batch's scratch never
    /// cuts one.
    pub fn cut_calls(&self) -> u64 {
        self.cut_calls
    }

    /// Passes the calls through this scratch completed, since it was made —
    /// each one pool dispatch, which a pool of one thread runs on the caller.
    /// A call wider than [`DEFER_MAX_COLS`] columns runs five: `x` quantized,
    /// every expert's gate and up, every combine quantized, every expert's
    /// down, the sums. A narrower one runs two — its gate/up and down row
    /// passes, which claim the quantizations — or, with the deferral lever
    /// off, four; it sums on the caller. A call that lists no expert runs
    /// none, and a cut call its calls'. A pass that fails ends its call and
    /// is not counted.
    pub fn passes(&self) -> u64 {
        self.passes
    }
}

/// The union call's shape conditions, each its own error: at most the
/// scratch's columns, one list per column of `x`, `out` holding every
/// column's width, the scratch made for the block's width (`embd`), and its
/// expert width a capacity — the layer's `ff` at or below it, so one scratch
/// serves the run's layers of differing widths, and its combine slab (sized
/// by [`ops::col_bytes_max`] of the capacity) holding the layer's down's
/// columns.
fn check_union_call(
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &[f32],
    ff: usize,
    scratch: &UnionScratch,
) -> Result<(), ModelError> {
    let (embd, cols) = (x.ne0(), x.ne1());
    if cols > scratch.max_cols {
        return Err(ModelError::Shape {
            what: "host union: at most the columns the scratch was made for",
            want_ne0: embd,
            want_ne1: scratch.max_cols,
            got_ne0: embd,
            got_ne1: cols,
        });
    }
    if cols != lists.len() {
        return Err(ModelError::Shape {
            what: "host union: x must have one column per list",
            want_ne0: embd,
            want_ne1: lists.len(),
            got_ne0: embd,
            got_ne1: cols,
        });
    }
    if out.len() != embd * cols {
        return Err(ModelError::Shape {
            what: "host union: out must hold every column's width",
            want_ne0: embd,
            want_ne1: cols,
            got_ne0: out.len(),
            got_ne1: 1,
        });
    }
    if scratch.embd != embd {
        return Err(ModelError::Shape {
            what: "host union: the scratch must be made for the block's width",
            want_ne0: embd,
            want_ne1: ff,
            got_ne0: scratch.embd,
            got_ne1: scratch.ff,
        });
    }
    if ff > scratch.ff || ops::col_bytes_max(ff) > ops::col_bytes_max(scratch.ff) {
        return Err(ModelError::Shape {
            what: "host union: the scratch must be made for the block's widths (its expert \
                   width is a capacity: the layer's ff at or below it, with room for its column)",
            want_ne0: embd,
            want_ne1: ff,
            got_ne0: scratch.embd,
            got_ne1: scratch.ff,
        });
    }
    Ok(())
}

/// The host leg of `k` tokens over the union of their experts: column `j`
/// of `out` (`out[j·embd..(j+1)·embd]`) is exactly what [`serve`] writes for
/// column `j` of `x` and `lists[j]`, bit for bit, in five pool passes
/// ([`UnionScratch::passes`], [`UnionCall::run`]): `x` quantized once, by
/// the gate's and the up's activation rule; every distinct expert's gate and
/// up over the columns that list it, one row dispatch in expert order — each
/// weight row read once per call, its runs of up to [`qdot::TILE_COLS`]
/// listed columns through the tile kernel (of up to [`qdot::WIDE_COLS`]
/// through the wide kernel, for a type with one), which writes each column's
/// `dot_row` value; every slot's combine, quantized in the down's encoding;
/// every expert's down over its combines, one row dispatch; and each
/// column's sum, in that column's list order from zero — the order its
/// one-column call adds in. A call of at most [`DEFER_MAX_COLS`] columns
/// claims its quantizations inside its two row dispatches instead (`x`
/// column by column, each expert's combines at once), unless the deferral
/// lever is off, and sums on the caller. Nothing here depends on the call's
/// width, on the order of the work or on who runs it, so a column's bits
/// are the same in a call of any width that lists it the same way (a group
/// tail's included). A plan past the scratch's slots — a tail call over its
/// budget — runs as consecutive calls of [`UNION_MAX_COLS`] columns, each
/// within them, and counts in [`UnionScratch::cut_calls`]. `stacks` are the
/// layer's `[gate, up, down]`, resolved once for the call; `lists` has been
/// checked against `x`, `out` and the scratch. With `tier`, each call reads
/// the ids outside the plan's host segment from the NVMe tier's arena
/// ([`serve_tiered`]); without, every id from `stacks`' own bytes.
fn serve_union<'w>(
    stacks: [ExpertStack<'w>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
    tier: Option<&'w LayerTier>,
) -> Result<(), ModelError> {
    s.plan.build(lists, s.per_list)?;
    if s.plan.fits(s.max_slots) {
        return serve_one(stacks, run, x, lists, out, s, tier);
    }
    s.cut_calls += 1;
    let embd = x.ne0();
    let mut c0 = 0;
    while c0 < lists.len() {
        let c1 = (c0 + UNION_MAX_COLS).min(lists.len());
        let part = Tensor2View::new(&x.data()[c0 * embd..c1 * embd], embd, c1 - c0)?;
        s.plan.build(&lists[c0..c1], s.per_list)?;
        assert!(
            s.plan.fits(s.max_slots),
            "host union: a batch's columns fit every scratch (union_batch_slots or more)"
        );
        serve_one(
            stacks,
            run,
            part,
            &lists[c0..c1],
            &mut out[c0 * embd..c1 * embd],
            s,
            tier,
        )?;
        c0 = c1;
    }
    Ok(())
}

/// [`serve_union`]'s one call: the plan in `s.plan` served from the tier when
/// there is one, an empty plan's included, so the tier is told of every
/// column the layer serves ([`TierSlots::note_union`]).
fn serve_one<'w>(
    stacks: [ExpertStack<'w>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
    tier: Option<&'w LayerTier>,
) -> Result<(), ModelError> {
    match tier {
        Some(tier) => serve_tiered(stacks, run, x, lists, out, s, tier),
        None => serve_planned(stacks, run, x, lists, out, s),
    }
}

/// A call served in batches: the layer's tier and the arena's ids kept (the
/// `S − 2B` of [`serve_tiered`]).
#[derive(Clone, Copy)]
struct Batching<'t> {
    tier: &'t LayerTier,
    keep: usize,
}

/// [`serve_planned`] for the plan in `s.plan` over a layer with an arena: every
/// id outside the plan's host segment is read from its slot, so each is
/// claimed, filled and pinned before the passes read it — and a pin lasts
/// only while the passes read it. The layer's slots `S` and the batch `B`
/// decide the shape. A call whose arena ids are at most `S − 2B` claims them
/// all as [`Claim::Keep`] at once and runs whole, as a decode step's union
/// does. A larger call runs its experts in batches of `B`, the arena's ids
/// as `lay_out` orders them: the `S − 2B` with the most columns claimed
/// [`Claim::Keep`], the rest [`Claim::Transient`] — the arena's cold end, the
/// first victims — each batch pinned while it computes and no longer, and the
/// next batch's claim issued before the compute, so the pinned slots are at
/// most `2B`. Its gate/up, combine and down passes run per batch, each
/// expert's downs stored by their slot, and the per-column sums run once
/// after the last batch, in list order from zero: every value is the
/// whole call's, bit for bit. A layer under `2B` and the lane's slots is
/// refused by name ([`TierError::BatchFloor`]) before any claim, and so is an
/// expert past the stacks'. The host segment's experts compute first, under
/// the first batch's fill. A call with no arena id (an empty plan, or the
/// host segment's ids alone) claims nothing and is told as one that fits.
fn serve_tiered<'w>(
    stacks: [ExpertStack<'w>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
    tier: &'w LayerTier,
) -> Result<(), ModelError> {
    let batch = tier.batch;
    assert!(batch > 0, "host union: a batch holds at least one expert");
    // Refused before any claim: an expert past the stacks' (the passes' own
    // check, made once here).
    UnionCall::new(s.plan.view(), stacks, x, run.form)?;
    let mut stacks = stacks;
    tier.lend(&mut stacks);
    s.tier_ids.clear();
    s.tier_ids.extend(
        s.plan
            .ids
            .iter()
            .copied()
            .filter(|e| !tier.host.contains(e)),
    );
    if s.tier_ids.is_empty() {
        tier.slots.note_union(
            tier.layer,
            &s.plan.ids,
            &s.plan.off,
            &s.plan.cols,
            lists.len(),
            false,
        );
        return serve_planned(stacks, run, x, lists, out, s);
    }
    let slots = tier.slots.slots(tier.layer);
    if slots >= 2 * batch && s.tier_ids.len() <= slots - 2 * batch {
        tier.slots.note_union(
            tier.layer,
            &s.plan.ids,
            &s.plan.off,
            &s.plan.cols,
            lists.len(),
            false,
        );
        let pin = tier
            .slots
            .ensure_pinned(tier.layer, &s.tier_ids, Claim::Keep)?;
        tier.check_lent(&stacks, &s.tier_ids)?;
        let served = serve_planned(stacks, run, x, lists, out, s);
        drop(pin);
        return served;
    }
    let need = 2 * batch + crate::placement::ARENA_LANE_SLOTS as usize;
    if slots < need {
        return Err(TierError::BatchFloor {
            layer: tier.layer,
            slots,
            need,
        }
        .into());
    }
    tier.slots.note_union(
        tier.layer,
        &s.plan.ids,
        &s.plan.off,
        &s.plan.cols,
        lists.len(),
        true,
    );
    let batching = Batching {
        tier,
        keep: slots - 2 * batch,
    };
    serve_batched(stacks, run, x, lists, out, s, batching)
}

/// [`serve_tiered`]'s batches: the arena's ids of the plan in `s.plan` cut as
/// `batching` says.
fn serve_batched<'w>(
    stacks: [ExpertStack<'w>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
    batching: Batching<'w>,
) -> Result<(), ModelError> {
    let Batching { tier, keep } = batching;
    let (batch, embd) = (tier.batch, x.ne0());
    let UnionScratch {
        plan,
        xq,
        xq_up,
        gu,
        qc,
        store,
        scales,
        passes,
        batches,
        ..
    } = s;
    let plan = &*plan;
    let cut = batches.lay_out(plan, &tier.host, keep, batch);
    let UnionBatches {
        order,
        ids,
        ends,
        slot,
        plan: sub,
    } = batches;
    let (order, ids, ends) = (&*order, &*ids, &*ends);
    let span = |g: usize| {
        let from = g.checked_sub(1).map_or(0, |p| ends[p]);
        from..ends[g]
    };
    let whole = UnionCall::new(plan.view(), stacks, x, run.form)?
        .spread_over(run.lanes)
        .post_normed(run.post)
        .expert_input(run.input, scales)?;
    let mut xqs = whole.quantize_batched(xq, xq_up, passes);
    slot.clear();
    slot.resize(plan.n(), 0);
    let claim_of = |g: usize| {
        if g < cut.host + cut.keep {
            Claim::Keep
        } else {
            Claim::Transient
        }
    };
    let mut pending = Some(tier.slots.prefetch_pinned(
        tier.layer,
        &ids[span(cut.host)],
        claim_of(cut.host),
        1,
        None,
    )?);
    let mut base = 0;
    for g in 0..ends.len() {
        let mut pin = None;
        if g >= cut.host {
            let ticket = pending.take().expect("a ticket for every arena batch");
            pin = Some(ticket.wait()?);
            if g + 1 < ends.len() {
                pending = Some(tier.slots.prefetch_pinned(
                    tier.layer,
                    &ids[span(g + 1)],
                    claim_of(g + 1),
                    1,
                    None,
                )?);
            }
            tier.check_lent(&stacks, &ids[span(g)])?;
        }
        sub.cut(plan, &order[span(g)]);
        for (i, &d) in order[span(g)].iter().enumerate() {
            slot[d as usize] = base + sub.off[i];
        }
        let batch_call = UnionCall::new(sub.view(), stacks, x, run.form)?
            .spread_over(run.lanes)
            .post_normed(run.post)
            .expert_input(run.input, scales)?;
        batch_call.run_batch(&mut xqs, gu, qc, &mut store[base * embd..], None, passes)?;
        base += sub.cols.len();
        drop(pin);
    }
    whole.sum_batched(slot, store, lists, out, passes);
    Ok(())
}

/// A decode step's union over a layer whose claim has misses in flight
/// ([`HostLayer::experts_step_union_into_on`]) as two groups of
/// [`serve_batched`]'s shape: the host segment's experts and the ones the
/// claim found filled compute while the misses fill on the tier's threads,
/// then the ticket is waited for and the misses compute. Each group runs the
/// two claiming passes of a narrow call (the three of the pre-pass shape when
/// the call is wide or the deferral is off), its downs stored by their slot,
/// and the per-column sums run once after the second, in list order from
/// zero: every value is the whole call's, bit for bit. A step with nothing
/// early or nothing late waits first and runs whole.
#[allow(
    clippy::too_many_arguments,
    reason = "the call's pieces: stacks, leg, input, lists, output, scratch, tier and ticket"
)]
fn serve_step_split<'w>(
    stacks: [ExpertStack<'w>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
    tier: &'w LayerTier,
    ticket: Ticket<'w>,
) -> Result<(), ModelError> {
    let (n_host, early) = {
        let UnionScratch { plan, batches, .. } = &mut *s;
        batches.split_step(plan, &tier.host, |e| {
            tier.slots.slot(tier.layer, e, 0).is_some()
        })
    };
    let n = s.plan.n();
    if early == 0 || early == n {
        let pin = ticket.wait()?;
        tier.check_lent(&stacks, &s.tier_ids)?;
        let served = serve_planned(stacks, run, x, lists, out, s);
        drop(pin);
        return served;
    }
    let embd = x.ne0();
    let UnionScratch {
        plan,
        xq,
        xq_up,
        gu,
        qc,
        store,
        scales,
        passes,
        batches,
        claims,
        ..
    } = s;
    let plan = &*plan;
    let UnionBatches {
        order,
        ids,
        slot,
        plan: sub,
        ..
    } = batches;
    let (order, ids) = (&*order, &*ids);
    let whole = UnionCall::new(plan.view(), stacks, x, run.form)?
        .spread_over(run.lanes)
        .post_normed(run.post)
        .expert_input(run.input, scales)?;
    let deferred = defers(x.ne1());
    let mut xqs = if deferred {
        whole.x_rooms(xq, xq_up)
    } else {
        whole.quantize_batched(xq, xq_up, passes)
    };
    slot.clear();
    slot.resize(n, 0);
    let mut ticket = Some(ticket);
    let mut pin = None;
    let mut base = 0;
    for (g, span) in [0..early, early..n].into_iter().enumerate() {
        if g == 0 {
            tier.check_lent(&stacks, &ids[n_host..early])?;
        } else {
            let t = (profile::level() > 0).then(Instant::now);
            pin = Some(
                ticket
                    .take()
                    .expect("one ticket for the late group")
                    .wait()?,
            );
            if let Some(t) = t {
                profile::record_time("host_fill_wait", t.elapsed().as_nanos() as u64);
            }
            tier.check_lent(&stacks, &ids[span.clone()])?;
        }
        sub.cut(plan, &order[span.clone()]);
        for (i, &d) in order[span].iter().enumerate() {
            slot[d as usize] = base + sub.off[i];
        }
        let call = UnionCall::new(sub.view(), stacks, x, run.form)?
            .spread_over(run.lanes)
            .post_normed(run.post)
            .expert_input(run.input, scales)?;
        let claim = deferred.then(|| GroupClaims {
            x: g == 0,
            states: &mut claims[..],
        });
        // The hit group runs while the miss ticket is still held. The late
        // group waits that ticket first, and that wait is not this span.
        let hit_t = (g == 0).then(Instant::now);
        call.run_batch(&mut xqs, gu, qc, &mut store[base * embd..], claim, passes)?;
        if let Some(hit_t) = hit_t {
            tier.slots
                .note_step_hits_under_read(hit_t.elapsed().as_nanos() as u64);
        }
        base += sub.cols.len();
    }
    drop(pin);
    whole.sum_batched(slot, store, lists, out, passes);
    Ok(())
}

/// [`serve_union`] for a plan already built into `s.plan` from `lists`,
/// which fits its slabs: the call ([`UnionCall::new`]) run through the
/// scratch.
fn serve_planned(
    stacks: [ExpertStack<'_>; 3],
    run: LegRun<'_>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
) -> Result<(), ModelError> {
    if s.plan.n() == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let call = UnionCall::new(s.plan.view(), stacks, x, run.form)?
        .spread_over(run.lanes)
        .post_normed(run.post)
        .expert_input(run.input, &s.scales)?;
    let slabs = UnionSlabs {
        xq: &mut s.xq,
        xq_up: &mut s.xq_up,
        gu: &mut s.gu,
        qc: &mut s.qc,
        store: &mut s.store,
        claims: &mut s.claims,
    };
    call.run(slabs, lists, out, &mut s.passes)
}

/// [`experts_into`] for many tokens at once — up to the columns `scratch` was
/// made for, at most [`UNION_MAX_COLS`] (a group tail's scratch,
/// [`UNION_TAIL_MAX_COLS`]): column `j` of `x` with its routed list
/// `lists[j]` into `out[j·embd..(j+1)·embd]`, equal to
/// `experts_into(x_j, lists[j])` bit for bit, while each distinct expert's
/// matrices are read once for every column that lists it (see
/// [`serve_union`]). `x` is a block ([`Tensor2`]) or any view of `k` columns
/// ([`Tensor2View`]), read in place. A list holds at most the routed width
/// `scratch` was made for; that bound, the column bound, a list count
/// other than `x`'s columns, an `out` of another length and an expert id past
/// the block's are named errors. The dispatches write into the caller's
/// `scratch`, made at load for the block's widths and a column count
/// ([`UnionScratch::new_routed`]), so a call allocates nothing.
pub fn experts_union_into<'x>(
    gguf: &Gguf,
    plan: &MoeBlockPlan,
    x: impl Into<Tensor2View<'x>>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    scratch: &mut UnionScratch,
) -> Result<(), ModelError> {
    let x = x.into();
    check_union_call(x, lists, out, plan.meta.ff, scratch)?;
    // The plan's stacks are headers of `gguf`, the one file this model has.
    let [gate, up, down] = &*plan.exps;
    let stacks = [
        ExpertStack::in_file(gguf, gate)?,
        ExpertStack::in_file(gguf, up)?,
        ExpertStack::in_file(gguf, down)?,
    ];
    serve_union(stacks, LegRun::flat(), x, lists, out, scratch, None)
}

/// What a host tier needs to serve one layer's routed experts, in the
/// file's own terms: the three stacks' tensor names, the expert count, the
/// widths and the routed experts' SwiGLU limit — the gated reading of a
/// layer ([`ExpertForm::Gated`], which [`HostLayer::build`] serves it by, or
/// [`ExpertForm::GatedOai`], which [`HostLayer::build_oai`] serves it by, the
/// spec's limit the OAI rule's); the relu² reading has its own spec
/// ([`Relu2Spec`], [`HostLayer::build_relu2`]).
/// The architecture fills it; this module reads no model's names or keys.
pub struct HostLayerSpec<'a> {
    pub gate: &'a str,
    pub up: &'a str,
    pub down: &'a str,
    pub n_expert: usize,
    pub embd: usize,
    pub ff: usize,
    /// `clamp(up, ±limit) · min(silu(gate), limit)`; at or below 1e-6 the
    /// plain combine.
    pub swiglu_limit: f32,
}

/// What a host tier needs to serve one relu² layer's routed experts
/// ([`ExpertForm::Relu2`]): the two stacks' tensor names, the expert count and
/// the widths — `embd` the width the experts run in, `ff` this layer's own.
/// The architecture fills it; there is no gate name to give.
pub struct Relu2Spec<'a> {
    pub up: &'a str,
    pub down: &'a str,
    pub n_expert: usize,
    pub embd: usize,
    pub ff: usize,
}

/// The host tier's matmul path for each (weight type, row width, row
/// layout) it serves, printed once per process when a layer first brings it
/// in: `load host_tier type=iq3_xxs k=4096 path=fused` — qdot's fused
/// kernel, the only path a host tier runs: [`HostLayer::build`] refuses a
/// stack without one before it prints — or `path=r8`, the row-lane tile over
/// the r8 sidecar's copy.
fn announce_paths(stacks: &[(GgmlType, usize, RowLayout)]) {
    static SEEN: Mutex<Vec<(GgmlType, usize, RowLayout)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    for &(ty, k, layout) in stacks {
        if seen.contains(&(ty, k, layout)) {
            continue;
        }
        seen.push((ty, k, layout));
        let path = match layout {
            RowLayout::Rows => "fused",
            RowLayout::R8 => "r8",
            RowLayout::Planes => "planes",
        };
        eprintln!("load host_tier type={ty} k={k} path={path}");
    }
}

/// A routed gate or up stack as a host tier reads it: the file's own rows,
/// or the r8 sidecar's row-lane copy of a Q3_K stack — the bytes pick the
/// kernel.
enum GateUp {
    File(ShardTensor),
    R8(R8Stack),
}

impl GateUp {
    /// The source's stack: the file's, or the one the sidecar's copy was
    /// checked against.
    fn source(&self) -> &ShardTensor {
        match self {
            GateUp::File(t) => t,
            GateUp::R8(s) => s.source(),
        }
    }

    fn layout(&self) -> RowLayout {
        match self {
            GateUp::File(t) => t.layout(),
            GateUp::R8(_) => RowLayout::R8,
        }
    }

    /// Matrix `e` as a dispatch weight.
    fn expert<'a>(&'a self, split: &'a Split, e: usize) -> Result<Weight<'a>, ModelError> {
        match self {
            GateUp::File(t) => t.expert(split, e),
            GateUp::R8(s) => s.expert(e),
        }
    }

    /// The stack as a union call reads it.
    fn stack<'a>(&'a self, split: &'a Split) -> Result<ExpertStack<'a>, ModelError> {
        match self {
            GateUp::File(t) => t.stack(split),
            GateUp::R8(s) => s.stack(),
        }
    }
}

/// Why the host tier's NVMe tier refused a call: the named refusals the
/// tier raises at the [`TierSlots`] seam, carried by [`ModelError::Tier`]
/// wherever the host leg can meet them.
#[derive(Debug, thiserror::Error)]
pub enum TierError {
    /// The tier holds the expert but its slot is unfilled: `ensure` did not
    /// fill it before the call read it — never a silent mapping fallback,
    /// which would hide a skipped `ensure` behind the right bytes.
    #[error(
        "the NVMe tier's slot for layer {layer} expert {id} is unfilled: ensure it before the call"
    )]
    Unfilled { layer: usize, id: u32 },
    /// A fill's read failed: the tier's own error, naming what it read.
    #[error("layer {layer} expert {id}: filling its slot: {source}")]
    Fill {
        layer: usize,
        id: u32,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A filled slot lends a part of another length than the stack's one
    /// matrix: the arena carved it for another stack.
    #[error(
        "the NVMe tier's slot for layer {layer} expert {id} part {part} lends {got} B; the \
         stack's matrix is {want} B"
    )]
    SlotLength {
        layer: usize,
        id: u32,
        part: usize,
        got: usize,
        want: usize,
    },
    /// One call names more distinct ids of the arena than the layer's slots:
    /// a union call reads every id it names at once, so they can never sit
    /// resident together, and the tier's books refuse the call before they
    /// claim any slot.
    #[error(
        "layer {layer}'s arena holds {slots} slots; one call names {ids} distinct ids of it, \
         which can never sit resident together"
    )]
    TooManyIds {
        layer: usize,
        ids: usize,
        slots: usize,
    },
    /// A batch pin was refused: the tier's own error, naming what it
    /// pinned.
    #[error("layer {layer}: pinning a batch of experts: {source}")]
    Pin {
        layer: usize,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A union call over more of the arena's ids than the layer's slots less
    /// two batches runs in batches, and the layer's slots are under the
    /// floor of that: the computing batch and the one being filled pinned at
    /// once, and the lane's slots beside them.
    #[error(
        "layer {layer}'s arena holds {slots} slots; a union call served in batches pins two batches and leaves the lane's slots, which takes {need}"
    )]
    BatchFloor {
        layer: usize,
        slots: usize,
        need: usize,
    },
}

/// How a claim places the slots it takes and the ids it finds filled in the
/// tier's LRU books ([`TierSlots::ensure_as`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    /// A decode step's claim: every id the call names takes a fresh tick, a
    /// miss as a hit.
    Step,
    /// A claim that holds what it takes: a fresh tick like [`Claim::Step`],
    /// a kind of its own so a caller can count what it kept.
    Keep,
    /// A claim that does not displace the hot set: a miss enters at the cold
    /// end and is the first victim, oldest first, before any hot slot; a hit
    /// leaves the slot as it is, hot or transient. A later [`Claim::Step`]
    /// or [`Claim::Keep`] hit turns a transient hot.
    Transient,
}

/// What a [`TierPin`] holds, handed back to its [`PinRelease`] when the pin
/// drops: the layer and the ids the pin counted. `ids` is a buffer the tier
/// lent the pin, moved in and moved back, so a pin makes no allocation of its
/// own.
pub struct PinToken {
    pub layer: usize,
    pub ids: Vec<u32>,
}

/// The tier's end of a [`TierPin`]: the drop of a pin hands its token here.
/// `Sync` because a pin crosses threads (a ticket's wait) holding a borrow of
/// the tier.
pub trait PinRelease: Sync {
    /// Unpin the ids of `token` and take its buffer back. Infallible, as a
    /// drop is: a count already at zero is a broken invariant the tier
    /// panics on by name (unless the thread is unwinding).
    fn release(&self, token: PinToken);
    /// The card's address of part `part` of `id`'s filled slot, on a tier
    /// whose slots the card maps; `None` everywhere else, and where
    /// [`TierSlots::slot`] lends nothing.
    fn device_slot(&self, _id: u32, _part: usize) -> Option<u64> {
        None
    }
}

/// A batch of experts pinned in the tier's slots ([`TierSlots::pin`]): the
/// ids stay unevictable until the guard drops. No box holds the guard's
/// state: a pin is made per paged layer per step, and the step does no
/// load-time work.
#[must_use = "the pin lasts until the guard drops"]
pub struct TierPin<'a> {
    release: &'a dyn PinRelease,
    token: Option<PinToken>,
}

impl<'a> TierPin<'a> {
    /// A guard over `token`'s ids, whose drop hands the token to `release`.
    pub fn new(release: &'a dyn PinRelease, token: PinToken) -> TierPin<'a> {
        TierPin {
            release,
            token: Some(token),
        }
    }

    /// The ids this pin holds.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        self.token.as_ref().map_or(&[], |t| &t.ids)
    }

    /// The card's address of part `part` (as [`TierSlots::slot`] counts
    /// parts) of `id`, one of this pin's ids, in its layer's slot: the bytes
    /// [`TierSlots::slot`] lends, which stay put while the pin lives. `None`
    /// for an id the pin does not hold and on a tier the card does not map.
    #[must_use]
    pub fn device_slot(&self, id: u32, part: usize) -> Option<u64> {
        if !self.ids().contains(&id) {
            return None;
        }
        self.release.device_slot(id, part)
    }
}

impl Drop for TierPin<'_> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.release.release(token);
        }
    }
}

/// What a [`Ticket`] waits on without boxing the wait. The tier that issued
/// the ticket is the value, and the ticket only borrows it: a steady step
/// issues one of these and allocates nothing for the handle. [`Ticket::wait`]
/// calls [`TicketWait::finish_ticket`]; a ticket dropped first calls
/// [`TicketWait::abandon_ticket`], which settles the claim the way a dropped
/// boxed wait does.
pub trait TicketWait<'a>: Sync {
    /// Join the fill and return the batch's pin.
    fn finish_ticket(&'a self) -> Result<TierPin<'a>, TierError>;
    /// The ticket was dropped. Settle the claim and drop the pin.
    fn abandon_ticket(&'a self);
}

/// A batch's claim in flight ([`TierSlots::prefetch_pinned`]): the batch's
/// slots are taken in the tier's books and being filled, and
/// [`Ticket::wait`] returns once every one is filled, with the batch pinned.
/// A boxed handle ([`Ticket::new`]) owns a completion the tier's threads can
/// run anywhere. A bound handle ([`Ticket::bound`]) borrows the tier instead,
/// so the steady one-column miss allocates no wait. A claim with nothing
/// to fill is a ready ticket, which holds its pin and allocates nothing. A
/// ticket dropped without a wait gives its claim back: the batch is unpinned,
/// once its fill has ended.
#[must_use = "the claim lasts until the ticket is waited for or dropped"]
pub struct Ticket<'a> {
    state: TicketState<'a>,
    /// A bound ticket still owns the tier's claim. Cleared before the wait
    /// runs, so dropping the ticket afterwards does not settle twice.
    open: bool,
}

enum TicketState<'a> {
    /// Filled and pinned already.
    Ready(TierPin<'a>),
    /// The fill ends as the handle returns.
    Pending(Box<dyn FnOnce() -> Result<TierPin<'a>, TierError> + Send + 'a>),
    /// The fill ends as the borrowed tier says.
    Bound(&'a dyn TicketWait<'a>),
    /// [`Ticket::wait`] has taken the state.
    Spent,
}

impl<'a> Ticket<'a> {
    /// A ticket that completes as `wait` returns: the batch's pin once every
    /// slot is filled, or the fill's refusal.
    pub fn new(wait: impl FnOnce() -> Result<TierPin<'a>, TierError> + Send + 'a) -> Ticket<'a> {
        Ticket {
            state: TicketState::Pending(Box::new(wait)),
            open: false,
        }
    }

    /// A ticket whose batch is filled and pinned already.
    pub fn ready(pin: TierPin<'a>) -> Ticket<'a> {
        Ticket {
            state: TicketState::Ready(pin),
            open: false,
        }
    }

    /// A ticket whose wait is `wait`, borrowed, not boxed.
    pub fn bound(wait: &'a dyn TicketWait<'a>) -> Ticket<'a> {
        Ticket {
            state: TicketState::Bound(wait),
            open: true,
        }
    }

    /// Whether the batch is filled and pinned already: a wait returns at once.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self.state, TicketState::Ready(_))
    }

    /// Block until the batch is filled; its pin, or the fill's refusal.
    pub fn wait(mut self) -> Result<TierPin<'a>, TierError> {
        self.open = false;
        match std::mem::replace(&mut self.state, TicketState::Spent) {
            TicketState::Ready(pin) => Ok(pin),
            TicketState::Pending(wait) => wait(),
            TicketState::Bound(wait) => wait.finish_ticket(),
            TicketState::Spent => unreachable!("a ticket is waited for once"),
        }
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if self.open {
            self.open = false;
            if let TicketState::Bound(wait) = &self.state {
                (*wait).abandon_ticket();
            }
        }
    }
}

/// The RAM arena a host layer's routed experts read from beside the file
/// mapping, as the gpu side holds it: `ensure` fills the slots of the ids a
/// call is about to read (a miss is read before the compute, so a call's
/// bytes are exact), `slot` lends one expert's part — `part` 0 the gate,
/// 1 the up, 2 the down — as the arena holds it, `None` when the slot is
/// unfilled. The trait is the model side's handle because the arena is the
/// gpu side's; the choice which expert reads where is the layer's
/// ([`HostLayer::experts_into`]), not the arena's.
pub trait TierSlots: Send + Sync {
    /// Fill the slots of `layer`'s `ids` the tier does not already hold, the
    /// call placed in the books as `claim` says.
    fn ensure_as(&self, layer: usize, ids: &[u32], claim: Claim) -> Result<(), TierError>;
    /// [`TierSlots::ensure_as`] as a decode step claims ([`Claim::Step`]).
    fn ensure(&self, layer: usize, ids: &[u32]) -> Result<(), TierError> {
        self.ensure_as(layer, ids, Claim::Step)
    }
    /// Pin `layer`'s `ids`, each of which the tier must hold filled: none is
    /// a victim of a later claim until the guard drops. Refused by name when
    /// an id is past the layer's experts or unfilled; nothing is pinned then.
    fn pin<'a>(&'a self, layer: usize, ids: &[u32]) -> Result<TierPin<'a>, TierError>;
    /// Layer `layer`'s slot count: what a union call sizes its batches
    /// against. 0 for a layer the tier does not cover.
    fn slots(&self, layer: usize) -> usize;
    /// [`TierSlots::ensure_as`] and [`TierSlots::pin`] as one claim, for a
    /// batch the caller computes at once: every id of `ids` the tier serves
    /// is filled and pinned when this returns, and no claim takes a slot of
    /// the batch between its claim and its pin — a hit is pinned as the
    /// claim finds it, a miss as its fill publishes, under one hold of the
    /// tier's books. Refused by name like the two it joins (an id past the
    /// layer's experts, every slot pinned); nothing is pinned then.
    fn ensure_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
    ) -> Result<TierPin<'a>, TierError>;
    /// [`TierSlots::ensure_pinned`] as a claim in flight: the batch's slots
    /// are claimed when this returns, the fill may run on other threads, and
    /// [`Ticket::wait`] returns the pin once it ends. A caller issues the
    /// next batch's ticket before it computes the one it holds, so the fill
    /// of the next runs under the compute of this. Once this returns, an id
    /// of `ids` whose [`TierSlots::slot`] lends is filled and held by the
    /// ticket's pin (the claim found it so), and an id whose slot lends
    /// nothing is in flight until the wait: a step computes the first early
    /// and the second after its wait. The default fills on the caller, the
    /// ticket ready. `columns` advances the tier's column clock; without
    /// `column_off`, zero keeps selections at the endpoint another leg counted.
    /// With `column_off`,
    /// `ids` is the full union plan, including host ids, and `ids[d]` routes
    /// `off[d + 1] - off[d]` columns at positions `positions[off[d]..off[d + 1]]`,
    /// where `column_off` is `(off, positions)`; without, each id routes once.
    fn prefetch_pinned<'a>(
        &'a self,
        layer: usize,
        ids: &[u32],
        claim: Claim,
        _columns: u64,
        _column_off: Option<(&[usize], &[usize])>,
    ) -> Result<Ticket<'a>, TierError> {
        self.ensure_pinned(layer, ids, claim).map(Ticket::ready)
    }
    /// Layer `layer`, expert `id`, part `part`'s bytes as the tier holds
    /// them; `None` when its slot is unfilled. The caller reads the bytes
    /// only while it holds `id` pinned ([`TierSlots::pin`],
    /// [`TierSlots::ensure_pinned`], a [`Ticket`]'s pin): no claim takes a
    /// pinned slot, and an unpinned one is the LRU's to refill.
    fn slot(&self, layer: usize, id: u32, part: usize) -> Option<&[u8]>;
    /// A prompt union call of `cols` columns on `layer`, told once per tiered
    /// union call, before its first batch's claim, as it takes its path:
    /// `batched` when its arena ids are more than the layer's slots less two
    /// batches. `ids` are the call's distinct ids, ascending, those of the
    /// plan's host segment included, and the columns that list `ids[d]`
    /// are `positions[off[d]..off[d + 1]]`. A call refused before its path is told
    /// is not told. It must not lock or allocate unless its tier keeps books
    /// of the call (a gate's record of it, or the routing counts its
    /// victims are ranked by); a tier that keeps none takes the default.
    fn note_union(
        &self,
        _layer: usize,
        _ids: &[u32],
        _off: &[usize],
        _positions: &[usize],
        _cols: usize,
        _batched: bool,
    ) {
    }
    /// Host compute of a step's hit group while its miss ticket was still
    /// in flight, in ns. A tier that does not count it takes the default.
    /// A ready ticket never reaches the hit group, so it adds nothing.
    fn note_step_hits_under_read(&self, _ns: u64) {}
}

/// One layer's tier handle: the layer's own index (the arena's books are
/// per layer), the plan's host segment for it (the ids the file mapping
/// serves; every other id — the NVMe segment's and the card's victims — is
/// the arena's), and the arena itself. Attached after the layer's build
/// ([`HostLayer::attach_tier`]); a layer with none reads the mapping as it
/// always did.
struct LayerTier {
    layer: usize,
    host: std::ops::Range<u32>,
    slots: std::sync::Arc<dyn TierSlots>,
    /// The experts a batch of the layer's tiered union holds.
    batch: usize,
}

impl LayerTier {
    /// `stacks` reading every id outside the plan's host segment from the
    /// arena's slots.
    fn lend<'w>(&'w self, stacks: &mut [ExpertStack<'w>; 3]) {
        for (part, stack) in stacks.iter_mut().enumerate() {
            *stack = stack.with_tier(StackTier::new(&*self.slots, self.layer, &self.host, part));
        }
    }

    /// Each of `ids`' three parts as the arena lends them, against `stacks`:
    /// refused by name when a slot is unfilled ([`TierError::Unfilled`]) or
    /// of another length than the stack's matrix ([`TierError::SlotLength`]).
    fn check_lent(&self, stacks: &[ExpertStack<'_>; 3], ids: &[u32]) -> Result<(), TierError> {
        for &id in ids {
            for (part, stack) in stacks.iter().enumerate() {
                match self.slots.slot(self.layer, id, part) {
                    None => {
                        return Err(TierError::Unfilled {
                            layer: self.layer,
                            id,
                        });
                    }
                    Some(bytes) if bytes.len() != stack.per() => {
                        return Err(TierError::SlotLength {
                            layer: self.layer,
                            id,
                            part,
                            got: bytes.len(),
                            want: stack.per(),
                        });
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(())
    }
}

/// The tensor names a layer build reads; a relu² layer names no gate.
struct StackNames<'a> {
    gate: Option<&'a str>,
    up: &'a str,
    down: &'a str,
}

/// One layer's routed experts as a host tier serves them: its form
/// ([`ExpertForm`]) and the stacks that form reads — gated, the gate and the
/// up from the file or from its r8 sidecar and the down from the file; a
/// relu² layer, the up and the down, no gate. Built once at load
/// ([`HostLayer::build`], [`HostLayer::build_relu2`]); a call cuts its
/// experts' matrices from the stacks, so a layer holds headers, not one view
/// per expert.
pub struct HostLayer {
    /// The gate stack — `None` on a relu² layer, which holds none.
    gate: Option<GateUp>,
    up: GateUp,
    down: ShardTensor,
    n_expert: usize,
    /// This layer's own feed-forward width: the scratch its calls write is
    /// made for it ([`HostScratch::new`]).
    ff: usize,
    form: ExpertForm,
    input: ExpertInput,
    tier: Option<LayerTier>,
    /// The norm each expert's down row takes before its weight
    /// ([`HostLayer::set_post_norm`]); `None`, the plain weighted sum.
    post: Option<PostNorm>,
}

impl HostLayer {
    /// Find the three stacks and check them against the spec: gate and up
    /// map `embd -> ff`, down maps `ff -> embd`, all `n_expert` deep and cut
    /// evenly, each with a fused kernel at its row width — the union call's
    /// own check ([`UnionStack::of_call`]), whose error names the stack. Every
    /// stack error a call could meet is raised here, at load.
    ///
    /// When `src` reads the source's r8 sidecar ([`R8Source`]), the gate and
    /// the up are the sidecar's copies of their source stacks (`R8Stack::of`
    /// — the same name, dims and bytes, so every expert keeps its byte
    /// range), and the down, which a sidecar never holds, is the source's.
    /// Refused by name: a sidecar that lacks the gate or the up, and one that
    /// holds the down. The sidecar is a second byte source the load resolves,
    /// not a property of the layer's tensors, so it rides beside the spec, in
    /// the pair.
    pub fn build(src: R8Source<'_>, spec: &HostLayerSpec<'_>) -> Result<HostLayer, ModelError> {
        Self::build_form(
            src,
            StackNames {
                gate: Some(spec.gate),
                up: spec.up,
                down: spec.down,
            },
            (spec.n_expert, spec.embd, spec.ff),
            ExpertForm::Gated {
                limit: spec.swiglu_limit,
            },
        )
    }

    /// [`HostLayer::build`] for a SwiGLU-OAI layer ([`ExpertForm::GatedOai`]):
    /// the same three stacks and the same checks, the spec's `swiglu_limit`
    /// the OAI rule's `limit` and `alpha` beside it, like the relu² form, an
    /// activation fact the load resolves. A non-finite or negative limit and
    /// a non-finite or non-positive alpha are refused by name.
    pub fn build_oai(
        src: R8Source<'_>,
        spec: &HostLayerSpec<'_>,
        alpha: f32,
    ) -> Result<HostLayer, ModelError> {
        if !(spec.swiglu_limit.is_finite() && spec.swiglu_limit >= 0.0) {
            return Err(ModelError::Metadata {
                key: "swiglu_oai limit".to_string(),
                detail: format!(
                    "the limit must be finite and not negative, got {}",
                    spec.swiglu_limit
                ),
            });
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(ModelError::Metadata {
                key: "swiglu_oai alpha".to_string(),
                detail: format!("the alpha must be finite and positive, got {alpha}"),
            });
        }
        Self::build_form(
            src,
            StackNames {
                gate: Some(spec.gate),
                up: spec.up,
                down: spec.down,
            },
            (spec.n_expert, spec.embd, spec.ff),
            ExpertForm::GatedOai {
                alpha,
                limit: spec.swiglu_limit,
            },
        )
    }

    /// [`HostLayer::build`] for a relu² layer ([`Relu2Spec`]): the up and the
    /// down checked as the gated build checks them, no gate stack looked for.
    /// The form rides beside the spec like the r8 sidecar does: an activation
    /// fact the load resolves, not a property of the layer's tensors.
    pub fn build_relu2(src: R8Source<'_>, spec: &Relu2Spec<'_>) -> Result<HostLayer, ModelError> {
        Self::build_form(
            src,
            StackNames {
                gate: None,
                up: spec.up,
                down: spec.down,
            },
            (spec.n_expert, spec.embd, spec.ff),
            ExpertForm::Relu2,
        )
    }

    /// [`HostLayer::build`], [`HostLayer::build_oai`] and
    /// [`HostLayer::build_relu2`]'s one body over
    /// the named stacks: a gate name (`Some`) is a stack the layer holds, found,
    /// checked and served; `None`, one it does not.
    fn build_form(
        src: R8Source<'_>,
        names: StackNames<'_>,
        (n_expert, embd, ff): (usize, usize, usize),
        form: ExpertForm,
    ) -> Result<HostLayer, ModelError> {
        let StackNames { gate, up, down } = names;
        let split = src.split();
        let found = gate.map(|g| ShardTensor::find(split, g)).transpose()?;
        let up_stack = ShardTensor::find(split, up)?;
        let down_stack = ShardTensor::find(split, down)?;
        if let Some(gate) = &found {
            expect_stack(gate.info(), embd, ff, n_expert)?;
            // The per-expert cut: `expert_view`'s check, once, here.
            expert_view(gate.info(), 0, n_expert)?;
        }
        for (t, k, n) in [(&up_stack, embd, ff), (&down_stack, ff, embd)] {
            expect_stack(t.info(), k, n, n_expert)?;
            // The per-expert cut: `expert_view`'s check, once, here.
            expert_view(t.info(), 0, n_expert)?;
        }
        let (gate, up) = match src.sidecar() {
            None => (found.map(GateUp::File), GateUp::File(up_stack)),
            Some(side) => {
                if side.find(down).is_some() {
                    return Err(R8Error::HoldsDown {
                        path: side.path().to_path_buf(),
                        tensor: down.to_string(),
                    }
                    .into());
                }
                let gate = gate
                    .map(|g| R8Stack::of(src, g).map(GateUp::R8))
                    .transpose()?;
                (gate, GateUp::R8(R8Stack::of(src, up)?))
            }
        };
        // A relu² layer's stacks as the union reads them: the up again in the
        // gate's slot, one stack where the gated form reads two.
        let gate_stack = || match &gate {
            Some(g) => g.stack(split),
            None => up.stack(split),
        };
        UnionStack::of_call(
            &[gate_stack()?, up.stack(split)?, down_stack.stack(split)?],
            embd,
            ff,
        )?;
        let mut paths = Vec::with_capacity(3);
        if let Some(gate) = &gate {
            paths.push((gate.source().info().ty, embd, gate.layout()));
        }
        paths.extend([
            (up.source().info().ty, embd, up.layout()),
            (down_stack.info().ty, ff, down_stack.layout()),
        ]);
        announce_paths(&paths);
        Ok(HostLayer {
            gate,
            up,
            down: down_stack,
            n_expert,
            ff,
            form,
            input: ExpertInput::Direct,
            tier: None,
            post: None,
        })
    }

    /// Select the activated expert input policy for every host serving path.
    /// This is a load-time setting; the default preserves direct quantization.
    pub fn set_expert_input(&mut self, policy: ExpertInput) {
        self.input = policy;
    }

    /// The loaded activated-expert input policy.
    pub fn expert_input(&self) -> ExpertInput {
        self.input
    }

    /// Give the layer the norm each routed expert's down row takes before
    /// its router weight ([`PostNorm`]): from then on every call that serves
    /// the layer — the one-column leg, the union and the step union, from the
    /// mapping or the tier — sums the normed, weighted rows in list order
    /// from zero. Load-time only. A norm of another width than the down's
    /// rows is refused by name.
    pub fn set_post_norm(&mut self, post: PostNorm) -> Result<(), ModelError> {
        let embd = self.down.info().dims[1] as usize;
        if post.width() != embd {
            return Err(ModelError::Shape {
                what: "host layer post-expert norm: a gain of the down's rows",
                want_ne0: embd,
                want_ne1: 1,
                got_ne0: post.width(),
                got_ne1: 1,
            });
        }
        self.post = Some(post);
        Ok(())
    }

    /// Give this layer — the `layer`-th of the run — the NVMe tier's arena
    /// and the plan's host segment for it: every id outside `host` the layer
    /// reads from the arena ([`TierSlots::slot`]), every id inside it from
    /// the file mapping as before. Load-time only, once; the arena outlives
    /// the layer (the run and the tiers share it).
    pub fn attach_tier(
        &mut self,
        layer: usize,
        host: std::ops::Range<u32>,
        slots: std::sync::Arc<dyn TierSlots>,
    ) {
        self.tier = Some(LayerTier {
            layer,
            host,
            slots,
            batch: UNION_BATCH_IDS,
        });
    }

    /// One expert's part as this layer serves it, the one owner of the
    /// choice: `cut` the file mapping's weight when the tier is absent or
    /// the plan's host segment holds the id, else the arena slot's bytes
    /// over the part's stack header — whose byte range the slot's bytes
    /// mirror — refused by name when the slot is unfilled, never a silent
    /// fallback to the mapping.
    fn tier_or_mapping<'a>(
        &'a self,
        e: u32,
        part: usize,
        stack: &ShardTensor,
        layout: RowLayout,
        cut: impl FnOnce() -> Result<Weight<'a>, ModelError>,
    ) -> Result<Weight<'a>, ModelError> {
        let Some(tier) = &self.tier else {
            return cut();
        };
        if tier.host.contains(&e) {
            return cut();
        }
        match tier.slots.slot(tier.layer, e, part) {
            Some(bytes) => {
                ops::expert_weight(stack.info(), bytes, 0, layout, stack.scale_of(e as usize)?)
            }
            None => Err(TierError::Unfilled {
                layer: tier.layer,
                id: e,
            }
            .into()),
        }
    }

    /// The three stacks, `[gate, up, down]`, as the source file holds them:
    /// for a gate or an up read from the r8 sidecar, the source stack its copy
    /// was checked against ([`HostLayer::layouts`] tells which). A relu²
    /// layer's gate slot holds its up again — the file holds no gate, and the
    /// form ([`HostLayer::form`]) is the one owner of which stacks a call
    /// reads.
    pub fn stacks(&self) -> [&ShardTensor; 3] {
        [
            self.gate.as_ref().map_or(self.up.source(), GateUp::source),
            self.up.source(),
            &self.down,
        ]
    }

    /// Where each stack's rows lie, `[gate, up, down]` — every expert of a
    /// stack has its stack's layout: [`RowLayout::R8`] for a stack the layer
    /// reads from the r8 sidecar, which no path reads as Q3_K rows, and
    /// [`RowLayout::Planes`] for a stack whose experts the split keeps where a
    /// checkpoint folder stores them. A relu² layer's gate slot carries its
    /// up's layout. What the load's reading made of the layer, for a caller
    /// that holds it to that reading.
    pub fn layouts(&self) -> [RowLayout; 3] {
        [
            self.gate.as_ref().map_or(self.up.layout(), GateUp::layout),
            self.up.layout(),
            self.down.layout(),
        ]
    }

    /// The layer's expert form.
    pub fn form(&self) -> ExpertForm {
        self.form
    }

    /// `src` is a pair this layer reads beside: one that reads the open of
    /// the sidecar its r8 stacks were built from — a pointer compare, the
    /// pair's own check having run when it was made. A layer that reads only
    /// the file takes any pair: it reads no sidecar.
    fn read_beside(&self, src: R8Source<'_>) -> Result<(), ModelError> {
        let ups = [self.gate.as_ref(), Some(&self.up)];
        for s in ups.into_iter().flatten() {
            if let GateUp::R8(s) = s {
                src.reads(s.sidecar(), "a host layer's call")?;
            }
        }
        Ok(())
    }

    /// The stacks as a union call reads them, `[gate, up, down]`: a relu²
    /// layer's up again in the gate's slot, one stack where the gated form
    /// reads two.
    fn stacks_of<'a>(&'a self, split: &'a Split) -> Result<[ExpertStack<'a>; 3], ModelError> {
        let gate = match &self.gate {
            Some(g) => g.stack(split)?,
            None => self.up.stack(split)?,
        };
        Ok([gate, self.up.stack(split)?, self.down.stack(split)?])
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// This layer's routed experts' feed-forward width.
    pub fn ff(&self) -> usize {
        self.ff
    }

    /// The routed experts' SwiGLU limit, a gated layer's; `0.0` for a relu²
    /// layer, which clamps nothing. A SwiGLU-OAI layer has no SwiGLU clamp (its
    /// limit is the OAI rule's, [`HostLayer::form`]), so it panics by name
    /// instead of handing that number to a clamp.
    pub fn swiglu_limit(&self) -> f32 {
        match self.form {
            ExpertForm::Gated { limit } => limit,
            ExpertForm::GatedOai { .. } => {
                panic!("swigluoai has no SwiGLU clamp: read the OAI rule's limit from the form")
            }
            ExpertForm::Relu2 => 0.0,
        }
    }

    /// [`experts_into`] for this layer: a given list of routed experts of
    /// ONE token — `out = Σ_i w_i · down_{e_i}(clamp(up_{e_i}·x, ±L) ⊙
    /// min(silu(gate_{e_i}·x), L))` gated, `out = Σ_i w_i ·
    /// down_{e_i}(relu(up_{e_i}·x)²)` for a relu² layer — in list order,
    /// each matrix read from the shard that holds it, or the sidecar. `src`
    /// is the pair the layer was built from. `scratch` is the caller's, made
    /// for this layer's widths and the model's routed width, and a list past
    /// that width is refused by name; after the call it holds what the call
    /// computed. The leg's two row dispatches are laid over the pool's CCDs
    /// ([`Lanes::host_tier`]), which changes no value.
    pub fn experts_into(
        &self,
        src: R8Source<'_>,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        scratch: &mut HostScratch,
    ) -> Result<(), ModelError> {
        self.experts_into_on(Lanes::host_tier(), src, x, experts, out, scratch, None, 1)
    }

    /// [`HostLayer::experts_into`] with its two row dispatches laid over the
    /// pool as `lanes` says: the same values, bit for bit, whichever lanes.
    /// `union` is the load's slabs. With them, a tiered call's claim is
    /// [`TierSlots::prefetch_pinned`]: a ready ticket, or a plan that does
    /// not fit, waits and then runs today's [`serve`]; a pending ticket whose
    /// plan fits runs the one-list split. Without slabs the ticket is
    /// waited before serving. `columns` is one for a new column, zero if
    /// another tier leg already counted it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the leg's pieces, plus the load's slabs when the call is resident"
    )]
    pub fn experts_into_on(
        &self,
        lanes: Lanes,
        src: R8Source<'_>,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        scratch: &mut HostScratch,
        union: Option<&mut UnionScratch>,
        columns: u64,
    ) -> Result<(), ModelError> {
        self.read_beside(src)?;
        let split = src.split();
        check_host_call(x, experts, out, scratch.n_used)?;
        if !scratch.fits(x.ne0, self.ff) {
            return Err(ModelError::Shape {
                what: "host experts: the scratch must be made for the layer's widths",
                want_ne0: x.ne0,
                want_ne1: self.ff,
                got_ne0: scratch.embd,
                got_ne1: scratch.ff,
            });
        }
        if self.tier.is_some()
            && let Some(union) = union
        {
            return self
                .serve_column_resident(lanes, split, x, experts, out, scratch, union, columns);
        }
        // The call's ids stay pinned until the leg's reads end: no claim of
        // a lane's victim takes one of their slots between the fill and the
        // read, whatever the slot count.
        let _pin = match &self.tier {
            Some(tier) => {
                scratch.tier_ids.clear();
                scratch.tier_ids.extend(experts.iter().map(|&(e, _)| e));
                Some(
                    tier.slots
                        .prefetch_pinned(tier.layer, &scratch.tier_ids, Claim::Step, columns, None)?
                        .wait()?,
                )
            }
            None => None,
        };
        self.serve_column(lanes, split, x, experts, out, scratch)
    }

    /// Today's one-column leg: `out` is the list-order sum, each expert read
    /// from the mapping or from a slot the caller already pinned.
    fn serve_column(
        &self,
        lanes: Lanes,
        split: &Split,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        scratch: &mut HostScratch,
    ) -> Result<(), ModelError> {
        out.fill(0.0);
        if experts.is_empty() {
            return Ok(());
        }
        let weights = |e: u32| -> Result<ExpertLegs<'_>, ModelError> {
            let down = self.tier_or_mapping(e, 2, &self.down, self.down.layout(), || {
                self.down.expert(split, e as usize)
            })?;
            match &self.gate {
                Some(gate) => Ok(ExpertLegs::Gated([
                    self.tier_or_mapping(e, 0, gate.source(), gate.layout(), || {
                        gate.expert(split, e as usize)
                    })?,
                    self.tier_or_mapping(e, 1, self.up.source(), self.up.layout(), || {
                        self.up.expert(split, e as usize)
                    })?,
                    down,
                ])),
                None => Ok(ExpertLegs::Relu2 {
                    up: self.tier_or_mapping(e, 1, self.up.source(), self.up.layout(), || {
                        self.up.expert(split, e as usize)
                    })?,
                    down,
                }),
            }
        };
        let run = LegRun {
            form: self.form,
            input: self.input,
            lanes,
            post: self.post.as_ref(),
        };
        serve(weights, run, x, experts, out, scratch)
    }

    /// A step call's distinct ids, the refusal of an id past the layer, and
    /// the ids outside the host segment. The one-column path and the union
    /// path share it.
    fn prepare_step_ids(
        &self,
        tier: &LayerTier,
        lists: &[&[(u32, f32)]],
        s: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        s.plan.build(lists, s.per_list)?;
        if let Some(&e) = s.plan.ids.last()
            && e as usize >= self.n_expert
        {
            return Err(ModelError::MissingTensor(format!(
                "expert {e} of {}",
                self.up.source().info().name
            )));
        }
        s.tier_ids.clear();
        s.tier_ids.extend(
            s.plan
                .ids
                .iter()
                .copied()
                .filter(|e| !tier.host.contains(e)),
        );
        Ok(())
    }

    /// The one-column resident path. A ready ticket, a plan that does not fit
    /// the slabs, or a call with no arena id waits and runs [`serve_column`].
    /// A pending ticket whose plan fits computes hits while the misses fill.
    #[allow(
        clippy::too_many_arguments,
        reason = "the one-column leg's pieces plus the slabs the split reuses"
    )]
    fn serve_column_resident(
        &self,
        lanes: Lanes,
        split: &Split,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        scratch: &mut HostScratch,
        union: &mut UnionScratch,
        columns: u64,
    ) -> Result<(), ModelError> {
        let lists: [&[(u32, f32)]; 1] = [experts];
        let tier = self.tier.as_ref().expect("a resident column has a tier");
        self.prepare_step_ids(tier, &lists, union)?;
        let ticket =
            tier.slots
                .prefetch_pinned(tier.layer, &union.tier_ids, Claim::Step, columns, None)?;
        if ticket.is_ready() || !union.plan.fits(union.max_slots) {
            let _pin = ticket.wait()?;
            return self.serve_column(lanes, split, x, experts, out, scratch);
        }
        let mut stacks = self.stacks_of(split)?;
        tier.lend(&mut stacks);
        let run = LegRun {
            form: self.form,
            input: self.input,
            lanes,
            post: self.post.as_ref(),
        };
        serve_step_split(stacks, run, x.view(), &lists, out, union, tier, ticket)
    }

    /// [`experts_union_into`] for this layer: up to the columns `scratch`
    /// was made for, column `j` of `out` equal to
    /// [`HostLayer::experts_into`] of column `j` of `x` and `lists[j]` bit
    /// for bit, each distinct expert's matrices read once per call from the
    /// shard that holds them. `x` is read in place, a block or a view; `src`
    /// is the pair the layer was built from. A layer with an NVMe tier reads
    /// the ids outside the plan's host segment from the arena's slots instead,
    /// claimed, filled and pinned in batches when the call names more than the
    /// layer's slots less two batches ([`UNION_BATCH_IDS`]), so its tier ids
    /// never need the file mapping; the bits are the same.
    pub fn experts_union_into<'x>(
        &self,
        src: R8Source<'_>,
        x: impl Into<Tensor2View<'x>>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        self.union_on(Lanes::Flat, src, x, lists, out, scratch)
    }

    /// [`HostLayer::experts_union_into`] with the two row passes laid over
    /// the pool as `lanes` says: the same values, bit for bit, whichever
    /// lanes. A layer with an arena reads the ids outside the plan's host
    /// segment from its slots, in batches when the call names more than the
    /// layer's slots less two batches ([`serve_tiered`]); a layer with none
    /// reads the file mapping alone.
    fn union_on<'x>(
        &self,
        lanes: Lanes,
        src: R8Source<'_>,
        x: impl Into<Tensor2View<'x>>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        self.read_beside(src)?;
        let split = src.split();
        let x = x.into();
        check_union_call(
            x,
            lists,
            out,
            self.up.source().info().dims[1] as usize,
            scratch,
        )?;
        let run = LegRun {
            form: self.form,
            input: self.input,
            lanes,
            post: self.post.as_ref(),
        };
        serve_union(
            self.stacks_of(split)?,
            run,
            x,
            lists,
            out,
            scratch,
            self.tier.as_ref(),
        )
    }

    /// [`HostLayer::experts_union_into`] with every id outside the plan's
    /// host segment read from the NVMe tier's arena, as
    /// [`HostLayer::experts_into`] reads it — a decode step's union, whose
    /// ids the next step reads again. The call's distinct tier ids, ascending,
    /// are filled first ([`TierSlots::ensure`]); each one's three parts must
    /// then lend one matrix of its stack, refused by name when a slot is
    /// unfilled ([`TierError::Unfilled`]) or of another length
    /// ([`TierError::SlotLength`]); and the union reads them from their slots,
    /// the same bytes as the mapping's, so column `j` is still
    /// [`HostLayer::experts_into`] of column `j` bit for bit. A layer with no
    /// tier reads the mapping alone, as [`HostLayer::experts_union_into`].
    /// Refused by name before any fill: an id past the layer's experts. Its
    /// two row passes are laid over the pool's CCDs ([`Lanes::host_tier`]), as
    /// the one-column [`HostLayer::experts_into`]'s are.
    pub fn experts_step_union_into<'x>(
        &self,
        src: R8Source<'_>,
        x: impl Into<Tensor2View<'x>>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        self.experts_step_union_into_on(Lanes::host_tier(), src, x, lists, out, scratch)
    }

    /// [`HostLayer::experts_step_union_into`] with its two row passes laid
    /// over the pool as `lanes` says: the same values, bit for bit, whichever
    /// lanes.
    pub fn experts_step_union_into_on<'x>(
        &self,
        lanes: Lanes,
        src: R8Source<'_>,
        x: impl Into<Tensor2View<'x>>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        let Some(tier) = &self.tier else {
            return self.union_on(lanes, src, x, lists, out, scratch);
        };
        self.read_beside(src)?;
        let split = src.split();
        let x = x.into();
        check_union_call(
            x,
            lists,
            out,
            self.up.source().info().dims[1] as usize,
            scratch,
        )?;
        // The plan's ids are the call's distinct ids, ascending; the union
        // builds the same plan again.
        self.prepare_step_ids(tier, lists, scratch)?;
        let mut stacks = self.stacks_of(split)?;
        let run = LegRun {
            form: self.form,
            input: self.input,
            lanes,
            post: self.post.as_ref(),
        };
        // The claim is made now and the misses fill on the tier's threads;
        // the pin lasts until the union's reads end, as `experts_into_on`'s
        // ids are.
        let ticket = tier.slots.prefetch_pinned(
            tier.layer,
            &scratch.plan.ids,
            Claim::Step,
            lists.len() as u64,
            Some((&scratch.plan.off, &scratch.plan.cols)),
        )?;
        tier.lend(&mut stacks);
        if ticket.is_ready() || !scratch.plan.fits(scratch.max_slots) {
            let pin = ticket.wait()?;
            tier.check_lent(&stacks, &scratch.tier_ids)?;
            let served = serve_union(stacks, run, x, lists, out, scratch, None);
            drop(pin);
            return served;
        }
        serve_step_split(stacks, run, x, lists, out, scratch, tier, ticket)
    }

    /// The matrices a warm routine fills for `ids`, pushed to `out` in `ids`'
    /// order, each id's stacks as the leg reads them ([`HostLayer::experts_into`],
    /// [`HostLayer::experts_step_union_into`]) — the gate, the up and the down
    /// gated; the up and the down for a relu² layer — from the file mapping or
    /// the r8 sidecar, or — an id outside the plan's host segment — from its
    /// arena slot. An id whose slot is unfilled is
    /// left out whole: a warm never fills a slot ([`TierSlots::ensure`]).
    /// Each span carries the units of the step legs' pass that reads it, so
    /// the CCD that computes a row is the CCD that warms it
    /// ([`ops::warm_share`]); a union call wider than [`ops::DEFER_MAX_COLS`]
    /// columns cuts other units ([`ops::WarmSpan`]). Refused by name: an id
    /// past the layer's experts, and a slot of another length
    /// ([`TierError::SlotLength`]); on a refusal `out` is as it was.
    pub fn warm_spans<'a>(
        &'a self,
        src: R8Source<'a>,
        ids: &[u32],
        out: &mut Vec<WarmSpan<'a>>,
    ) -> Result<(), ModelError> {
        let before = out.len();
        self.push_warm_spans(src, ids, out)
            .inspect_err(|_| out.truncate(before))
    }

    /// [`HostLayer::warm_spans`]'s walk of the ids, which leaves the spans of
    /// the ids before a refusal in `out`.
    fn push_warm_spans<'a>(
        &'a self,
        src: R8Source<'a>,
        ids: &[u32],
        out: &mut Vec<WarmSpan<'a>>,
    ) -> Result<(), ModelError> {
        self.read_beside(src)?;
        let mut stacks = self.stacks_of(src.split())?;
        if let Some(tier) = &self.tier {
            tier.lend(&mut stacks);
        }
        // A relu² layer warms its up and its down; the stacks' gate slot is
        // the up again.
        let warmed = match self.form {
            ExpertForm::Gated { .. } | ExpertForm::GatedOai { .. } => &stacks[..],
            ExpertForm::Relu2 => &stacks[1..],
        };
        'ids: for &id in ids {
            if id as usize >= self.n_expert {
                return Err(ModelError::MissingTensor(format!(
                    "expert {id} of {}",
                    self.up.source().info().name
                )));
            }
            // A matrix is one span, or two for planes: its codes and its
            // group scales.
            let mut all: [[Option<WarmSpan<'a>>; 2]; 3] = [[None; 2]; 3];
            let parts = &mut all[..warmed.len()];
            for (part, stack) in warmed.iter().enumerate() {
                let Some(spans) = stack.warm_spans(id) else {
                    continue 'ids;
                };
                let got: usize = spans.iter().flatten().map(|s| s.bytes.len()).sum();
                if got != stack.per() {
                    return Err(TierError::SlotLength {
                        layer: self.tier.as_ref().map_or(0, |t| t.layer),
                        id,
                        part,
                        got,
                        want: stack.per(),
                    }
                    .into());
                }
                parts[part] = spans;
            }
            out.extend(parts.iter().flatten().copied().flatten());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Buckets, HostLayer, HostLayerSpec, HostScratch, Relu2Spec, TOUCHED, UnionScratch,
        check_host_call, gather_expert_inputs, last_touched_experts, relist, reset_touched,
    };
    use crate::ops::{self, ExpertForm, RowLayout, Tensor2};
    use crate::r8file::{HostR8, R8Source, Sidecar};
    use gguf::GgmlType;
    use gguf::write::{Layout, TensorDecl, Writer};

    /// The touched mask is as wide as the expert count: ids past 63 land in
    /// words of their own and read back ascending with the first word's.
    #[test]
    fn touched_mask_spans_every_expert() {
        let n_expert = 384;
        let routed = [0usize, 63, 64, 200, 383];
        let mut offsets = vec![0u32; n_expert + 1];
        for e in 0..n_expert {
            offsets[e + 1] = offsets[e] + u32::from(routed.contains(&e));
        }
        let buckets = Buckets {
            offsets,
            order: vec![0; routed.len()],
            weight: vec![1.0; routed.len()],
        };
        let x = Tensor2::zeros(4, 1);
        reset_touched(n_expert);
        let (experts, _, _) =
            TOUCHED.with(|t| gather_expert_inputs(&x, &buckets, n_expert, 0, &mut t.borrow_mut()));
        assert_eq!(experts, routed, "every routed expert is gathered");
        assert_eq!(
            last_touched_experts(),
            routed,
            "the mask records every routed id, past 63 too"
        );
    }

    /// A host call's three shape conditions are three errors, each naming its
    /// condition and carrying what was passed — `x`'s token count among them.
    #[test]
    fn host_call_shape_errors_name_their_condition() {
        let refusal = |x: &Tensor2, n: usize, out: usize| {
            let experts = vec![(0u32, 1.0f32); n];
            match check_host_call(x, &experts, &vec![0.0; out], 10) {
                Ok(()) => panic!("a bad host call must be refused"),
                Err(e) => e.to_string(),
            }
        };
        let (x1, x2) = (Tensor2::zeros(8, 1), Tensor2::zeros(8, 2));
        let e = refusal(&x2, 1, 8);
        assert!(e.contains("one token") && e.contains("got [8, 2]"), "{e}");
        let e = refusal(&x1, 1, 7);
        assert!(e.contains("width") && e.contains("got [7, 1]"), "{e}");
        let e = refusal(&x1, 11, 8);
        assert!(
            e.contains("the scratch was made for") && e.contains("expected [10, 1], got [11, 1]"),
            "{e}"
        );
        assert!(check_host_call(&x1, &[(0, 1.0); 10], &[0.0; 8], 10).is_ok());
    }

    /// A scratch is made for the model's routed width, ten as readily as
    /// six, and a width of none is refused by name.
    #[test]
    fn host_scratch_takes_the_routed_width() {
        for n_used in [6, 8, 10] {
            let s = HostScratch::new(16, 32, n_used).expect("a routed width");
            assert_eq!(s.n_used(), n_used);
            assert_eq!((s.gate(n_used - 1).ne0, s.down(n_used - 1).ne0), (32, 16));
        }
        let e = HostScratch::new(16, 32, 0)
            .err()
            .expect("a width of none is refused")
            .to_string();
        assert!(e.contains("at least one routed expert"), "{e}");
    }

    /// A list relisted at another lifetime keeps its allocation and its
    /// room, so a call that refills it to the room it was made with
    /// allocates nothing.
    #[test]
    fn relist_keeps_the_room() {
        let mut v: Vec<&'static u32> = Vec::with_capacity(20);
        let at = v.as_ptr();
        v.push(&7);
        let local = 3u32;
        let mut w: Vec<&u32> = relist(v);
        assert!(w.is_empty());
        assert_eq!((w.capacity(), w.as_ptr()), (20, at));
        w.extend(std::iter::repeat_n(&local, 20));
        assert_eq!(w.as_ptr(), at, "a refill to the room reallocates nothing");
        let back: Vec<&'static u32> = relist(w);
        assert_eq!((back.capacity(), back.as_ptr()), (20, at));
    }

    /// A stack `{k, n, n_expert}` of `ty` whose every value is finite and
    /// small: seeded bytes, each block's f16 scales set to 2^-7 (and a
    /// Q4_K's mins to 2^-8), an f32 stack's values in [-0.5, 0.5).
    fn stack_bytes(ty: GgmlType, [k, n, n_expert]: [usize; 3], seed: u64) -> Vec<u8> {
        let (bs, ts) = (
            ty.blck_size().unwrap() as usize,
            ty.type_size().unwrap() as usize,
        );
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let blocks = k / bs * n * n_expert;
        let mut b = Vec::with_capacity(blocks * ts);
        for _ in 0..blocks {
            let at = b.len();
            b.extend((0..ts).map(|_| next() as u8));
            let blk = &mut b[at..];
            match ty {
                // block_q3_K: d f16 at 108.
                GgmlType::Q3_K => blk[108..110].copy_from_slice(&0x2000u16.to_le_bytes()),
                // block_q4_K and block_q5_K: d f16 at 0, dmin f16 at 2.
                GgmlType::Q4_K | GgmlType::Q5_K => {
                    blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                    blk[2..4].copy_from_slice(&0x1C00u16.to_le_bytes());
                }
                // block_q6_K: d f16 at 208.
                GgmlType::Q6_K => blk[208..210].copy_from_slice(&0x2000u16.to_le_bytes()),
                GgmlType::F32 => {
                    let v = (next() >> 40) as f32 / 16_777_216.0 - 0.5;
                    blk.copy_from_slice(&v.to_le_bytes());
                }
                _ => panic!("no synthetic stack of {ty}"),
            }
        }
        b
    }

    /// A one-file model in the temp dir holding one layer's routed stacks
    /// `{gate,up,down}_exps` of types `tys`, `n_expert`
    /// deep over widths `embd` and `ff`; returns its path.
    fn layer_file(
        tag: &str,
        tys: [GgmlType; 3],
        embd: usize,
        ff: usize,
        n_expert: usize,
    ) -> String {
        let names = ["gate", "up", "down"].map(|m| format!("{m}_exps"));
        let dims = [
            [embd, ff, n_expert],
            [embd, ff, n_expert],
            [ff, embd, n_expert],
        ];
        let tensors: Vec<(TensorDecl, Vec<u8>)> = (0..3)
            .map(|i| {
                let b = stack_bytes(tys[i], dims[i], 0x5eed + i as u64);
                let decl = TensorDecl {
                    name: names[i].clone(),
                    dims: dims[i].map(|d| d as u64).to_vec(),
                    type_id: tys[i].as_u32(),
                    nbytes: b.len() as u64,
                };
                (decl, b)
            })
            .collect();
        let layout = Layout::new(&[], tensors.iter().map(|(t, _)| t.clone()).collect()).unwrap();
        let path =
            std::env::temp_dir().join(format!("bloomery-moe-{}-{tag}.gguf", std::process::id()));
        let mut w = Writer::new(std::fs::File::create(&path).unwrap(), layout).unwrap();
        for (t, b) in &tensors {
            w.tensor(&t.name, b).unwrap();
        }
        w.finish().unwrap();
        path.to_string_lossy().into_owned()
    }

    /// The layer of [`layer_file`] as a host tier builds it.
    fn build_layer(
        split: &gguf::Split,
        embd: usize,
        ff: usize,
        n_expert: usize,
    ) -> Result<HostLayer, crate::ModelError> {
        HostLayer::build(
            R8Source::rows(split),
            &HostLayerSpec {
                gate: "gate_exps",
                up: "up_exps",
                down: "down_exps",
                n_expert,
                embd,
                ff,
                swiglu_limit: 0.0,
            },
        )
    }

    /// A layer with a stack the union cannot run — an f32 gate, no fused
    /// kernel — is refused when the host tier builds it, by the union's own
    /// error naming the matrix, the stack, its type and `k`; the same layer
    /// with a fused gate builds.
    #[test]
    fn host_layer_build_refuses_a_stack_without_a_fused_kernel() {
        let (embd, ff, n_expert) = (256, 256, 2);
        for (tag, gate, refused) in [("f32", GgmlType::F32, true), ("q4k", GgmlType::Q4_K, false)] {
            let path = layer_file(
                tag,
                [gate, GgmlType::Q4_K, GgmlType::Q4_K],
                embd,
                ff,
                n_expert,
            );
            let split = gguf::Split::open(&path).unwrap();
            let r = build_layer(&split, embd, ff, n_expert);
            std::fs::remove_file(&path).unwrap();
            match (r, refused) {
                (Ok(_), false) => println!("build {tag} gate: ok"),
                (Err(e), true) => {
                    let e = e.to_string();
                    for want in [
                        "host union: the gate must map embd -> ff",
                        "every expert of gate_exps",
                        &format!("({gate}, k = {embd})"),
                        "no fused qdot kernel",
                    ] {
                        assert!(e.contains(want), "the refusal names {want:?}, got {e:?}");
                    }
                    println!("build {tag} gate: refused: {e}");
                }
                (Ok(_), true) => panic!("a layer with an unfused stack must be refused at build"),
                (Err(e), false) => panic!("a fused layer must build: {e}"),
            }
        }
    }

    /// A post-expert norm as wide as the down's rows is the layer's; one of
    /// any other width — narrower, or the gate's rows — is refused by name
    /// and leaves the layer without a norm.
    #[test]
    fn host_layer_set_post_norm_refuses_another_width() {
        let (embd, ff, n_expert) = (256, 512, 2);
        let path = layer_file("postnorm", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let built = build_layer(&split, embd, ff, n_expert);
        std::fs::remove_file(&path).unwrap();
        let mut layer = built.unwrap();
        for width in [embd / 2, ff] {
            let post = ops::PostNorm::new(vec![1.0; width], 1e-6).unwrap();
            match layer.set_post_norm(post) {
                Err(crate::ModelError::Shape {
                    what,
                    want_ne0,
                    got_ne0,
                    ..
                }) => {
                    assert_eq!(
                        what, "host layer post-expert norm: a gain of the down's rows",
                        "the refusal names the norm"
                    );
                    assert_eq!(
                        (want_ne0, got_ne0),
                        (embd, width),
                        "the refusal names both widths"
                    );
                }
                Err(e) => panic!("a gain of {width} rows is refused by its own name, got {e}"),
                Ok(()) => panic!("a gain of {width} rows on a layer of {embd} must be refused"),
            }
            assert!(layer.post.is_none(), "a refused norm is not kept");
        }
        let post = ops::PostNorm::new(vec![1.0; embd], 1e-6).unwrap();
        layer.set_post_norm(post).unwrap();
        assert_eq!(
            layer.post.as_ref().map(ops::PostNorm::width),
            Some(embd),
            "a gain of the down's rows is the layer's norm"
        );
    }

    /// A relu² layer's build, by name: a spec whose stacks carry other dims
    /// than it says is refused with the stack's own error; a relu² layer that
    /// built reads its up again in the stacks' gate slot — the file's gate
    /// stack, which it holds here, is never read — and carries the relu²
    /// form; and a relu² layer with the NVMe tier attached reads the ids
    /// outside the host segment from the tier's slots, its up's part in the
    /// gate slot again, at every entry: the bits the file mapping gives.
    /// FAIL-first: each entry refused the tier by name ("gated experts
    /// only").
    #[test]
    fn host_layer_build_relu2_refuses_by_name() {
        let (embd, ff, n_expert) = (256, 512, 4);
        let check = |got: String, want: &str| {
            assert!(
                got.contains(want),
                "refusal must name {want:?}, got {got:?}"
            );
            println!("relu2 refusal {want:?}: {got}");
        };
        let spec = |embd, ff| Relu2Spec {
            up: "up_exps",
            down: "down_exps",
            n_expert,
            embd,
            ff,
        };
        let path = layer_file("relu2", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let src = R8Source::rows(&split);
        check(
            HostLayer::build_relu2(src, &spec(ff, embd))
                .err()
                .expect("a spec of other dims is refused")
                .to_string(),
            "expert stack: expected [512, 256], got [256, 512]",
        );
        let mut layer =
            HostLayer::build_relu2(src, &spec(embd, ff)).expect("two stacks of Q4_K build");
        assert_eq!(
            layer.form(),
            ExpertForm::Relu2,
            "the layer carries its form"
        );
        assert_eq!(
            (
                &layer.stacks()[0].info().name,
                &layer.stacks()[1].info().name
            ),
            (&"up_exps".to_string(), &"up_exps".to_string()),
            "the gate slot holds the up again"
        );

        let mut spans = Vec::new();
        layer.warm_spans(src, &[0], &mut spans).unwrap();
        assert_eq!(spans.len(), 2, "a relu² layer warms its up and its down");

        // The NVMe tier: every entry reads the ids outside the host segment
        // from the slots and gives the file mapping's bits.
        // PIN(2026-11-01): a relu² layer's NVMe tier is no longer refused ("gated experts
        // only"): the arena lends each expert its up and its down.
        let x = Tensor2::from_vec(
            embd,
            1,
            (0..embd).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect(),
        );
        let list = vec![(0u32, 0.75f32), (2, 0.25)];
        let mut host = HostScratch::new(embd, ff, n_expert).unwrap();
        let mut us = UnionScratch::new_routed(embd, ff, 4, n_expert).unwrap();
        let entries =
            |layer: &HostLayer, host: &mut HostScratch, us: &mut UnionScratch| -> [Vec<f32>; 3] {
                let mut out = [vec![0.0f32; embd], vec![0.0f32; embd], vec![0.0f32; embd]];
                layer
                    .experts_into(src, &x, &list, &mut out[0], host)
                    .expect("the one-column entry");
                layer
                    .experts_union_into(src, &x, &[&list[..]], &mut out[1], us)
                    .expect("the union entry");
                layer
                    .experts_step_union_into(src, &x, &[&list[..]], &mut out[2], us)
                    .expect("the step-union entry");
                out
            };
        let want = entries(&layer, &mut host, &mut us);
        let mut tier = MockTier::of(&split, &layer, true);
        tier.slots = 2 * super::UNION_BATCH_IDS + 8;
        let tier = std::sync::Arc::new(tier);
        layer.attach_tier(4, 1..2, tier.clone());
        let got = entries(&layer, &mut host, &mut us);
        for (entry, (g, w)) in ["one-column", "union", "step union"]
            .iter()
            .zip(got.iter().zip(&want))
        {
            assert_eq!(
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{entry}: a tiered relu² layer reads the mapping's bits"
            );
        }
        let asked = tier.asked.lock().unwrap();
        assert!(
            asked.iter().any(|&(l, id, _)| (l, id) == (4, 0))
                && asked.iter().any(|&(l, id, _)| (l, id) == (4, 2)),
            "the ids outside the host segment come from the tier: {asked:?}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A `BLOOMERY_QDOT_AVX512` value qdot refuses stops the host tier's load:
    /// in a fresh process with the lever at `2`, building a Q4_K layer panics
    /// naming the lever ([`qdot::q4k_avx512`], which [`ops::host_stack_runs`]
    /// asks first for every Q4_K stack), before any call. The lever is read
    /// before the CPU is asked, so a CPU without AVX-512, or off x86_64,
    /// refuses it too.
    #[test]
    fn host_layer_build_refuses_a_qdot_lever_value_by_name() {
        if !crate::test_process::in_process(
            concat!(
                module_path!(),
                "::host_layer_build_refuses_a_qdot_lever_value_by_name"
            ),
            ("BLOOMERY_QDOT_AVX512", "2"),
        ) {
            return;
        }
        let (embd, ff, n_expert) = (256, 256, 2);
        let path = layer_file("qdot-lever", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_layer(&split, embd, ff, n_expert).map(drop)
        }));
        std::fs::remove_file(&path).unwrap();
        let e = built.expect_err("a refused lever value must stop the build");
        let e = e.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            e.contains("BLOOMERY_QDOT_AVX512=\"2\""),
            "the refusal names the lever, got {e:?}"
        );
        println!("refusal: {e}");
    }

    /// The OAI form's layer over `split`: the spec of [`build_layer`] with
    /// `limit` and `alpha`.
    fn build_oai_layer(
        split: &gguf::Split,
        (embd, ff, n_expert): (usize, usize, usize),
        (alpha, limit): (f32, f32),
    ) -> Result<HostLayer, crate::ModelError> {
        HostLayer::build_oai(
            R8Source::rows(split),
            &HostLayerSpec {
                gate: "gate_exps",
                up: "up_exps",
                down: "down_exps",
                n_expert,
                embd,
                ff,
                swiglu_limit: limit,
            },
            alpha,
        )
    }

    /// A SwiGLU-OAI layer's list call is the OAI rule on the same weights:
    /// per expert the gate and up dots, [`qdot::swiglu_oai`] over them, the
    /// down over that, summed in list order — bit for bit the composition of
    /// the allocating group entry — and it differs from the clamped form's
    /// result over the same stacks.
    #[test]
    fn host_layer_oai_matches_the_rule_and_differs_from_the_clamp() {
        let (embd, ff, n_expert) = (256, 256, 3);
        let (alpha, limit) = (1.702f32, 7.0f32);
        let path = layer_file("oai", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let src = R8Source::rows(&split);
        let oai = build_oai_layer(&split, (embd, ff, n_expert), (alpha, limit)).unwrap();
        assert_eq!(
            oai.form(),
            ExpertForm::GatedOai { alpha, limit },
            "the layer carries its form"
        );
        let clamp = HostLayer::build(
            src,
            &HostLayerSpec {
                gate: "gate_exps",
                up: "up_exps",
                down: "down_exps",
                n_expert,
                embd,
                ff,
                swiglu_limit: limit,
            },
        )
        .unwrap();
        let x = Tensor2::from_vec(
            embd,
            1,
            (0..embd)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        let list = [(2u32, 0.5f32), (0, 0.25)];
        let run = |layer: &HostLayer| {
            let mut host = HostScratch::new(embd, ff, list.len()).unwrap();
            let mut out = vec![f32::NAN; embd];
            layer
                .experts_into(src, &x, &list, &mut out, &mut host)
                .unwrap();
            out
        };

        let g = gguf::Gguf::open(&path).unwrap();
        let view = |name: &str, e: u32| {
            super::expert_view(g.find(name).unwrap(), e as usize, n_expert).unwrap()
        };
        let mut want = vec![0.0f32; embd];
        for &(e, w) in &list {
            let [gate, up, down] = ["gate_exps", "up_exps", "down_exps"].map(|n| view(n, e));
            let gu = ops::matmul_q_group(&g, &[&gate, &up], &[&x, &x]).unwrap();
            let mut act = vec![0.0f32; ff];
            assert!(qdot::swiglu_oai(
                &gu[0].data,
                &gu[1].data,
                alpha,
                limit,
                &mut act
            ));
            let act = Tensor2::from_vec(ff, 1, act);
            let d = ops::matmul_q_group(&g, &[&down], &[&act]).unwrap();
            for (o, &dv) in want.iter_mut().zip(d[0].col(0)) {
                *o += w * dv;
            }
        }
        let got = run(&oai);
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("oai layer vs the rule: diff_cells={diff}");
        assert_eq!(diff, 0, "the layer's result must be the OAI composition");

        let clamped = run(&clamp);
        let apart = got
            .iter()
            .zip(&clamped)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            apart > 0,
            "the OAI form must not give the clamped form's result"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A SwiGLU-OAI layer has no SwiGLU clamp: `swiglu_limit` panics by name
    /// instead of returning the OAI rule's limit, while a clamped layer still
    /// returns its own.
    #[test]
    fn host_layer_oai_has_no_swiglu_clamp_limit() {
        let (embd, ff, n_expert) = (256, 256, 2);
        let path = layer_file("oailimit", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let oai = build_oai_layer(&split, (embd, ff, n_expert), (1.702, 7.0)).unwrap();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| oai.swiglu_limit()));
        std::fs::remove_file(&path).unwrap();
        let payload = caught.expect_err("an OAI layer has no SwiGLU clamp to return");
        let text = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            text.contains("swigluoai has no SwiGLU clamp"),
            "the panic must name the rule, got {text:?}"
        );
    }

    /// [`HostLayer::build_oai`] refuses by name a NaN alpha, a zero or
    /// negative alpha, a negative limit and a non-finite limit, before it looks
    /// for a stack.
    #[test]
    fn host_layer_build_oai_refuses_by_name() {
        let (embd, ff, n_expert) = (256, 256, 2);
        let path = layer_file("oairefuse", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        for (alpha, limit, want) in [
            (f32::NAN, 7.0, "swiglu_oai alpha"),
            (0.0, 7.0, "swiglu_oai alpha"),
            (-1.0, 7.0, "swiglu_oai alpha"),
            (1.702, -7.0, "swiglu_oai limit"),
            (1.702, f32::NAN, "swiglu_oai limit"),
            (1.702, f32::INFINITY, "swiglu_oai limit"),
        ] {
            let e = build_oai_layer(&split, (embd, ff, n_expert), (alpha, limit))
                .err()
                .unwrap_or_else(|| panic!("alpha {alpha}, limit {limit} must be refused"))
                .to_string();
            assert!(
                e.contains(want),
                "alpha {alpha}, limit {limit}: the refusal must name {want:?}, got {e:?}"
            );
        }
        build_oai_layer(&split, (embd, ff, n_expert), (1.702, 0.0))
            .expect("a zero limit is the rule's, not a refusal");
        std::fs::remove_file(&path).unwrap();
    }

    /// A union call over a SwiGLU-OAI layer writes, column for column and bit
    /// for bit, what the one-column call writes: narrow (3 columns) and wide
    /// (12).
    #[test]
    fn union_call_on_an_oai_layer_matches_its_columns() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let path = layer_file("oaiunion", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let layer = build_oai_layer(&split, (embd, ff, n_expert), (1.702, 7.0)).unwrap();
        let mut us = UnionScratch::new_routed(embd, ff, 16, 3).unwrap();
        let mut host = HostScratch::new(embd, ff, 3).unwrap();
        for cols in [3, 12] {
            let x = Tensor2::from_vec(
                embd,
                cols,
                (0..embd * cols)
                    .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                    .collect(),
            );
            let lists: Vec<Vec<(u32, f32)>> = (0..cols)
                .map(|j| {
                    [0, 1, 3]
                        .map(|s| (((j + s) % n_expert) as u32, 0.25 + 0.125 * s as f32))
                        .to_vec()
                })
                .collect();
            let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
            let mut want = vec![f32::NAN; embd * cols];
            for (j, list) in lists.iter().enumerate() {
                let xj = Tensor2::from_vec(embd, 1, x.col(j).to_vec());
                layer
                    .experts_into(
                        R8Source::rows(&split),
                        &xj,
                        list,
                        &mut want[j * embd..][..embd],
                        &mut host,
                    )
                    .unwrap();
            }
            let mut got = vec![f32::NAN; embd * cols];
            layer
                .experts_union_into(R8Source::rows(&split), &x, &slices, &mut got, &mut us)
                .unwrap();
            let diff = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{cols} columns: the union differs from its columns"
            );
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// A union call whose up reads other bytes than its gate — a Q3_K gate
    /// (q8_K columns) and a Q4_K up (q8_2_x4) — writes, column for column and
    /// bit for bit, what the one-column call writes, in both of its forms: a
    /// narrow call of 3 columns, whose gate/up pass claims x in both
    /// encodings, and a wide one of 12, which quantizes x twice in its first
    /// pass. Each runs the passes its form counts.
    #[test]
    fn union_call_with_an_up_on_other_bytes_matches_its_columns() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let tys = [GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("mixed", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut us = UnionScratch::new_routed(embd, ff, 16, 3).unwrap();
        let mut host = HostScratch::new(embd, ff, 3).unwrap();
        for (cols, want_passes) in [(3, if ops::defer_quant() { 2 } else { 4 }), (12, 5)] {
            let x = Tensor2::from_vec(
                embd,
                cols,
                (0..embd * cols)
                    .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                    .collect(),
            );
            let lists: Vec<Vec<(u32, f32)>> = (0..cols)
                .map(|j| {
                    [0, 1, 3]
                        .map(|s| (((j + s) % n_expert) as u32, 0.25 + 0.125 * s as f32))
                        .to_vec()
                })
                .collect();
            let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
            let mut want = vec![f32::NAN; embd * cols];
            for (j, list) in lists.iter().enumerate() {
                let xj = Tensor2::from_vec(embd, 1, x.col(j).to_vec());
                layer
                    .experts_into(
                        R8Source::rows(&split),
                        &xj,
                        list,
                        &mut want[j * embd..][..embd],
                        &mut host,
                    )
                    .unwrap();
            }
            let mut got = vec![f32::NAN; embd * cols];
            let p0 = us.passes();
            layer
                .experts_union_into(R8Source::rows(&split), &x, &slices, &mut got, &mut us)
                .unwrap();
            let passes = us.passes() - p0;
            let diff = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!("mixed gate/up k={cols}: diff_cells={diff} passes={passes}");
            assert_eq!(
                diff, 0,
                "{cols} columns: the union differs from its columns"
            );
            assert_eq!(
                passes, want_passes,
                "{cols} columns: the passes of its form"
            );
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// GLM-5.3-Flash's routed stacks — Q4_K gates and ups over a Q5_K or a
    /// Q6_K down, a clamped SwiGLU — through a union call write, column for
    /// column and bit for bit, what the one-column call writes, narrow (3
    /// columns) and wide (12).
    #[test]
    fn union_call_on_glm_stacks_matches_its_columns() {
        let (embd, ff, n_expert) = (256, 256, 6);
        for down in [GgmlType::Q5_K, GgmlType::Q6_K] {
            let tys = [GgmlType::Q4_K, GgmlType::Q4_K, down];
            let path = layer_file(&format!("glm-{down}"), tys, embd, ff, n_expert);
            let split = gguf::Split::open(&path).unwrap();
            let layer = HostLayer::build(
                R8Source::rows(&split),
                &HostLayerSpec {
                    gate: "gate_exps",
                    up: "up_exps",
                    down: "down_exps",
                    n_expert,
                    embd,
                    ff,
                    swiglu_limit: 0.01,
                },
            )
            .unwrap();
            let mut us = UnionScratch::new_routed(embd, ff, 16, 3).unwrap();
            let mut host = HostScratch::new(embd, ff, 3).unwrap();
            for cols in [3, 12] {
                let x = Tensor2::from_vec(
                    embd,
                    cols,
                    (0..embd * cols)
                        .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                        .collect(),
                );
                let lists: Vec<Vec<(u32, f32)>> = (0..cols)
                    .map(|j| {
                        [0, 2, 5]
                            .map(|s| (((j + s) % n_expert) as u32, 0.25 + 0.125 * s as f32))
                            .to_vec()
                    })
                    .collect();
                let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
                let mut want = vec![f32::NAN; embd * cols];
                for (j, list) in lists.iter().enumerate() {
                    let xj = Tensor2::from_vec(embd, 1, x.col(j).to_vec());
                    layer
                        .experts_into(
                            R8Source::rows(&split),
                            &xj,
                            list,
                            &mut want[j * embd..][..embd],
                            &mut host,
                        )
                        .unwrap();
                }
                let mut got = vec![f32::NAN; embd * cols];
                layer
                    .experts_union_into(R8Source::rows(&split), &x, &slices, &mut got, &mut us)
                    .unwrap();
                let diff = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!("glm stacks, {down} down, k={cols}: diff_cells={diff}");
                assert_eq!(
                    diff, 0,
                    "{down} down, {cols} columns: the union differs from its columns"
                );
            }
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// A scratch made for a routed width holds that many slots a column and
    /// refuses a longer list by name; a width of none is refused when it is
    /// made, and Qwen3.8's ten is made like any other.
    #[test]
    fn routed_scratch_refuses_a_list_past_its_width() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let tys = [GgmlType::Q4_K; 3];
        let e = UnionScratch::new_routed(embd, ff, 4, 0)
            .err()
            .expect("a routed width of none is refused")
            .to_string();
        assert!(e.contains("at least one expert per column"), "{e}");
        let ten = UnionScratch::new_routed(embd, ff, 4, 10).expect("a routed width of ten");
        assert_eq!(ten.max_slots(), 40, "ten slots a column");
        let mut us = UnionScratch::new_routed(embd, ff, 4, 3).unwrap();
        assert_eq!(us.max_slots(), 12, "three slots a column");
        let path = layer_file("routed", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let x = Tensor2::zeros(embd, 2);
        let mut out = vec![0.0f32; 2 * embd];
        let (three, four) = (
            [(0u32, 1.0f32), (1, 1.0), (2, 1.0)],
            [(0u32, 1.0f32), (1, 1.0), (2, 1.0), (3, 1.0)],
        );
        layer
            .experts_union_into(
                R8Source::rows(&split),
                &x,
                &[&three[..], &three[..]],
                &mut out,
                &mut us,
            )
            .unwrap();
        let e = layer
            .experts_union_into(
                R8Source::rows(&split),
                &x,
                &[&three[..], &four[..]],
                &mut out,
                &mut us,
            )
            .expect_err("a list past the routed width is refused")
            .to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(
            e.contains("the scratch's routed width") && e.contains("expected [3, 1], got [4, 1]"),
            "{e}"
        );
        println!("routed scratch: refusal {e}");
    }

    /// Qwen3.8's routed width, ten experts a column: through scratches made
    /// for ten, the one-column call takes a list of ten, and a narrow union
    /// call of eight columns whose lists share no expert — 80 distinct, each
    /// its own claim in the down pass — writes, column for column and bit for
    /// bit, what the one-column calls write, in the passes of its form.
    #[test]
    fn ten_a_column_through_the_narrow_claims() {
        let (embd, ff, n_used, cols) = (256, 256, 10, 8);
        let n_expert = n_used * cols;
        let tys = [GgmlType::Q4_K; 3];
        let path = layer_file("ten", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut us = UnionScratch::new_routed(embd, ff, cols, n_used).unwrap();
        let mut host = HostScratch::new(embd, ff, n_used).unwrap();
        let x = Tensor2::from_vec(
            embd,
            cols,
            (0..embd * cols)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        let lists: Vec<Vec<(u32, f32)>> = (0..cols)
            .map(|j| {
                (0..n_used)
                    .map(|s| ((j * n_used + s) as u32, 0.0625 * (1 + s) as f32))
                    .collect()
            })
            .collect();
        let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut want = vec![f32::NAN; embd * cols];
        for (j, list) in lists.iter().enumerate() {
            let xj = Tensor2::from_vec(embd, 1, x.col(j).to_vec());
            layer
                .experts_into(
                    R8Source::rows(&split),
                    &xj,
                    list,
                    &mut want[j * embd..][..embd],
                    &mut host,
                )
                .unwrap_or_else(|e| panic!("column {j}'s list of ten: {e}"));
        }
        let want_passes = if ops::defer_quant() { 2 } else { 4 };
        let mut got = vec![f32::NAN; embd * cols];
        let p0 = us.passes();
        layer
            .experts_union_into(R8Source::rows(&split), &x, &slices, &mut got, &mut us)
            .unwrap_or_else(|e| panic!("the union of eight lists of ten: {e}"));
        let passes = us.passes() - p0;
        std::fs::remove_file(&path).unwrap();
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("ten a column k={cols} distinct={n_expert}: diff_cells={diff} passes={passes}");
        assert_eq!(
            diff, 0,
            "the union of lists of ten differs from its columns"
        );
        assert_eq!(passes, want_passes, "a narrow call's passes");
    }

    // ------------------------------------------------- the NVMe tier's slot

    /// A [`super::TierSlots`] for the accessor tests: it owns every expert's
    /// three parts as its slots' bytes (read from the layer's own file, so a
    /// filled slot lends exactly the mapping's bytes), records the ids
    /// `ensure` saw and the `(layer, id, part)` every `slot` call asked for,
    /// and with `filled` off lends nothing — every slot unfilled.
    struct MockTier {
        parts: Vec<Vec<u8>>,
        filled: bool,
        /// What `slots` reports for every layer.
        slots: usize,
        ensured: std::sync::Mutex<Vec<(usize, Vec<u32>)>>,
        asked: std::sync::Mutex<Vec<(usize, u32, usize)>>,
        pins: std::sync::Mutex<PinLog>,
        /// The prompt note or step prefetch's column plan and batch path.
        told: std::sync::Mutex<Vec<Told>>,
        /// A step's claim: `None` holds every id of a ticket unwaited (the
        /// prompt batches' mode); `Some(late)` holds the claim's hits waited
        /// for at once and the ids in `late` in flight, lending nothing
        /// until their ticket's wait.
        late: Option<Vec<u32>>,
        flight: std::sync::Mutex<Vec<u32>>,
        /// An id the mock evicts the moment after its first lent probe: a
        /// tier that broke the pin, to show the refusal by name.
        evict: Option<u32>,
        gone: std::sync::Mutex<Vec<u32>>,
        /// Each lent `slot` and each ticket wait, in order.
        trail: std::sync::Mutex<Vec<Trail>>,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum Trail {
        Lent(u32, usize),
        Wait,
    }

    /// One union's column plan and batch path as the mock was told it.
    #[derive(Debug, PartialEq)]
    struct Told {
        layer: usize,
        ids: Vec<u32>,
        off: Vec<usize>,
        positions: Vec<usize>,
        cols: usize,
        batched: bool,
    }

    /// What the batch pins did, in order, and every read no pin covered.
    #[derive(Default)]
    struct PinLog {
        /// The ids pinned now with whether their ticket was waited for (a
        /// direct `ensure_pinned` is waited for as it returns).
        held: Vec<(u32, bool)>,
        /// The most ids pinned at once.
        peak: usize,
        events: Vec<PinEvent>,
        /// `(id, part)` of every `slot` call for an id that was not pinned
        /// by a batch whose ticket had been waited for.
        loose: Vec<(u32, usize)>,
    }

    #[derive(Debug, PartialEq)]
    enum PinEvent {
        Pinned(super::Claim, Vec<u32>),
        Unpinned(Vec<u32>),
    }

    /// A batch's pin is released here: the mock's log unpins the ids.
    impl super::PinRelease for MockTier {
        fn release(&self, token: super::PinToken) {
            let mut log = self.pins.lock().unwrap();
            for id in &token.ids {
                let at = log
                    .held
                    .iter()
                    .position(|h| h.0 == *id)
                    .expect("a pinned id");
                log.held.remove(at);
            }
            log.events.push(PinEvent::Unpinned(token.ids));
        }
    }

    impl MockTier {
        /// The tier over `layer`'s stacks of `split`, `filled` or not.
        fn of(split: &gguf::Split, layer: &HostLayer, filled: bool) -> MockTier {
            let n = layer.n_expert();
            let stacks = layer.stacks();
            let mut parts = Vec::with_capacity(3 * n);
            for e in 0..n {
                for s in stacks {
                    parts.push(
                        ops::ShardTensor::expert(s, split, e)
                            .unwrap()
                            .bytes()
                            .to_vec(),
                    );
                }
            }
            MockTier {
                parts,
                filled,
                slots: 0,
                ensured: std::sync::Mutex::new(Vec::new()),
                asked: std::sync::Mutex::new(Vec::new()),
                pins: std::sync::Mutex::new(PinLog::default()),
                told: std::sync::Mutex::new(Vec::new()),
                late: None,
                flight: std::sync::Mutex::new(Vec::new()),
                evict: None,
                gone: std::sync::Mutex::new(Vec::new()),
                trail: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// `ids` pinned and logged, `waited` when the caller has the pin.
        fn pin_batch(&self, ids: &[u32], claim: super::Claim, waited: bool) -> super::TierPin<'_> {
            let ids = ids.to_vec();
            let mut log = self.pins.lock().unwrap();
            log.held.extend(ids.iter().map(|&id| (id, waited)));
            log.peak = log.peak.max(log.held.len());
            log.events.push(PinEvent::Pinned(claim, ids.clone()));
            super::TierPin::new(self, super::PinToken { layer: 0, ids })
        }

        /// The batch pins' events so far.
        fn events(&self) -> Vec<PinEvent> {
            std::mem::take(&mut self.pins.lock().unwrap().events)
        }

        fn ensured(&self) -> Vec<(usize, Vec<u32>)> {
            self.ensured.lock().unwrap().clone()
        }

        fn asked(&self) -> Vec<(usize, u32, usize)> {
            self.asked.lock().unwrap().clone()
        }

        /// The path notes so far, taken.
        fn told(&self) -> Vec<Told> {
            std::mem::take(&mut self.told.lock().unwrap())
        }
    }

    impl super::TierSlots for MockTier {
        fn ensure_as(
            &self,
            layer: usize,
            ids: &[u32],
            _claim: super::Claim,
        ) -> Result<(), super::TierError> {
            self.ensured.lock().unwrap().push((layer, ids.to_vec()));
            Ok(())
        }

        fn pin<'a>(
            &'a self,
            _layer: usize,
            ids: &[u32],
        ) -> Result<super::TierPin<'a>, super::TierError> {
            Ok(self.pin_batch(ids, super::Claim::Step, true))
        }

        fn slots(&self, _layer: usize) -> usize {
            self.slots
        }

        fn ensure_pinned<'a>(
            &'a self,
            layer: usize,
            ids: &[u32],
            claim: super::Claim,
        ) -> Result<super::TierPin<'a>, super::TierError> {
            self.ensured.lock().unwrap().push((layer, ids.to_vec()));
            Ok(self.pin_batch(ids, claim, true))
        }

        /// The claim is made now, as a ready ticket's would be, and only the
        /// wait covers the batch's reads: a compute that reads it before its
        /// ticket was waited for reads a fill that may be running. A step's
        /// claim (`late` set) holds its hits at once; the ids in flight lend
        /// nothing until the wait.
        fn prefetch_pinned<'a>(
            &'a self,
            layer: usize,
            ids: &[u32],
            claim: super::Claim,
            columns: u64,
            column_off: Option<(&[usize], &[usize])>,
        ) -> Result<super::Ticket<'a>, super::TierError> {
            if let Some((off, positions)) = column_off {
                self.note_union(
                    layer,
                    ids,
                    off,
                    positions,
                    usize::try_from(columns).unwrap(),
                    false,
                );
            }
            self.ensured.lock().unwrap().push((layer, ids.to_vec()));
            let pin = self.pin_batch(ids, claim, false);
            if let Some(late) = &self.late {
                let mut log = self.pins.lock().unwrap();
                for h in log.held.iter_mut() {
                    h.1 = !late.contains(&h.0);
                }
                drop(log);
                self.flight
                    .lock()
                    .unwrap()
                    .extend(ids.iter().filter(|id| late.contains(id)));
            }
            let ids = ids.to_vec();
            Ok(super::Ticket::new(move || {
                self.trail.lock().unwrap().push(Trail::Wait);
                self.flight.lock().unwrap().clear();
                let mut log = self.pins.lock().unwrap();
                for id in &ids {
                    log.held
                        .iter_mut()
                        .find(|h| h.0 == *id)
                        .expect("a pinned id")
                        .1 = true;
                }
                drop(log);
                Ok(pin)
            }))
        }

        fn slot(&self, layer: usize, id: u32, part: usize) -> Option<&[u8]> {
            self.asked.lock().unwrap().push((layer, id, part));
            if self.flight.lock().unwrap().contains(&id) || self.gone.lock().unwrap().contains(&id)
            {
                return None;
            }
            if self.evict == Some(id) && self.filled {
                self.gone.lock().unwrap().push(id);
            }
            let mut log = self.pins.lock().unwrap();
            if !log.held.contains(&(id, true)) {
                log.loose.push((id, part));
            }
            drop(log);
            if self.filled {
                self.trail.lock().unwrap().push(Trail::Lent(id, part));
            }
            self.filled.then(|| &self.parts[3 * id as usize + part][..])
        }

        fn note_union(
            &self,
            layer: usize,
            ids: &[u32],
            off: &[usize],
            positions: &[usize],
            cols: usize,
            batched: bool,
        ) {
            self.told.lock().unwrap().push(Told {
                layer,
                ids: ids.to_vec(),
                off: off.to_vec(),
                positions: positions.to_vec(),
                cols,
                batched,
            });
        }
    }

    /// A warm lists each id's gate, up and down as the leg reads them — the
    /// mapping's bytes for an id of the host segment or a layer with no tier,
    /// the arena slot's for any other — in `ids`' order, each with its pass's
    /// units; a tier id whose slot is unfilled is left out whole; the tier is
    /// never asked to fill a slot; an id past the layer's experts is refused
    /// by name, and so is a slot of another length.
    #[test]
    fn a_warm_lists_what_the_leg_reads_and_never_fills() {
        let (embd, ff, n_expert) = (256, 512, 6);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("warmspans", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let src = R8Source::rows(&split);
        let units = [ff, ff, embd];
        let check = |layer: &HostLayer, out: &[super::WarmSpan<'_>], ids: &[usize]| {
            assert_eq!(out.len(), 3 * ids.len(), "three spans an id");
            for (i, &id) in ids.iter().enumerate() {
                for (part, stack) in layer.stacks().iter().enumerate() {
                    let want = ops::ShardTensor::expert(stack, &split, id).unwrap();
                    let span = out[3 * i + part];
                    assert_eq!(span.bytes, want.bytes(), "id {id} part {part}");
                    assert_eq!(span.units, units[part], "id {id} part {part}'s units");
                }
            }
        };

        let mut out = Vec::new();
        layer.warm_spans(src, &[4, 1], &mut out).unwrap();
        check(&layer, &out, &[4, 1]);

        // Host id 1; tier ids 3 and 2.
        let empty = std::sync::Arc::new(MockTier::of(&split, &layer, false));
        layer.attach_tier(4, 1..2, empty.clone());
        let mut out = Vec::new();
        layer.warm_spans(src, &[3, 1, 2], &mut out).unwrap();
        check(&layer, &out, &[1]);
        assert!(empty.ensured().is_empty(), "a warm fills nothing");

        let full = std::sync::Arc::new(MockTier::of(&split, &layer, true));
        layer.attach_tier(4, 1..2, full.clone());
        let mut out = Vec::new();
        layer.warm_spans(src, &[3, 1, 2], &mut out).unwrap();
        check(&layer, &out, &[3, 1, 2]);
        assert!(full.ensured().is_empty(), "a warm fills nothing");
        let asked: Vec<u32> = full.asked().iter().map(|&(_, id, _)| id).collect();
        assert!(
            asked.iter().all(|&id| id != 1),
            "the host segment's id was not the tier's: {asked:?}"
        );

        let err = layer
            .warm_spans(src, &[6], &mut Vec::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("expert 6"), "{err}");
        // A refusal after some ids' spans were pushed leaves `out` as it was.
        let mut kept = Vec::new();
        layer.warm_spans(src, &[1], &mut kept).unwrap();
        layer.warm_spans(src, &[4, 6], &mut kept).unwrap_err();
        assert_eq!(kept.len(), 3, "a refused call leaves what `out` held");
        let mut short = MockTier::of(&split, &layer, true);
        short.parts[3 * 3 + 1].pop();
        layer.attach_tier(4, 1..2, std::sync::Arc::new(short));
        match layer.warm_spans(src, &[3], &mut Vec::new()) {
            Err(crate::ModelError::Tier(super::TierError::SlotLength {
                layer: 4,
                id: 3,
                part: 1,
                ..
            })) => {}
            other => panic!("not the slot-length refusal by name: {other:?}"),
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// A layer read from the r8 sidecar warms the gate's and the up's copies
    /// of an expert — the sidecar's bytes, not the source's rows — as units of
    /// eight rows, the grain the pass cuts them by, and the down's source
    /// bytes as rows.
    #[test]
    fn a_warm_of_a_sidecar_layer_carries_the_groups_of_the_pass() {
        let (embd, ff, n_expert) = (256, 512, 4);
        let tys = [GgmlType::Q3_K, GgmlType::Q3_K, GgmlType::Q4_K];
        // The sidecar lands in a directory beside the source's: give the
        // source a directory of its own.
        let dir = std::env::temp_dir().join(format!("bloomery-moe-{}-warmr8", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let path = dir.join("src").join("layer.gguf");
        std::fs::rename(layer_file("warmr8", tys, embd, ff, n_expert), &path).unwrap();
        let split = gguf::Split::open(&path).unwrap();
        let sidecar = crate::r8file::sidecar_path(&path).unwrap();
        crate::r8file::convert(
            &split,
            &["gate_exps".to_string(), "up_exps".to_string()],
            &sidecar,
            &mut |_| {},
        )
        .unwrap();
        let side = HostR8::On(std::sync::Arc::new(
            Sidecar::open(&sidecar, &split, gguf::Weights::Mapped { populate: false }).unwrap(),
        ));
        let src = R8Source::of(&split, &side).unwrap();
        let layer = HostLayer::build(
            src,
            &HostLayerSpec {
                gate: "gate_exps",
                up: "up_exps",
                down: "down_exps",
                n_expert,
                embd,
                ff,
                swiglu_limit: 0.0,
            },
        )
        .unwrap();
        assert_eq!(
            layer.layouts(),
            [RowLayout::R8, RowLayout::R8, RowLayout::Rows]
        );
        let mut out = Vec::new();
        layer.warm_spans(src, &[2, 1], &mut out).unwrap();
        assert_eq!(out.len(), 6, "three spans an id");
        let held = side.sidecar().expect("a sidecar layer");
        for (i, id) in [2usize, 1].into_iter().enumerate() {
            for (part, name) in ["gate_exps", "up_exps"].into_iter().enumerate() {
                let span = out[3 * i + part];
                assert_eq!(span.units, ff / 8, "id {id} {name}: groups of eight rows");
                let stack = held.data(name).unwrap();
                let per = stack.len() / n_expert;
                assert_eq!(
                    span.bytes,
                    &stack[id * per..(id + 1) * per],
                    "id {id} {name}"
                );
                let source = ops::ShardTensor::find(&split, name).unwrap();
                assert_ne!(
                    span.bytes,
                    source.expert(&split, id).unwrap().bytes(),
                    "id {id} {name}: the sidecar's copy, not the source's rows"
                );
            }
            let down = out[3 * i + 2];
            let want = ops::ShardTensor::expert(layer.stacks()[2], &split, id).unwrap();
            assert_eq!(down.bytes, want.bytes(), "id {id} down");
            assert_eq!(down.units, embd, "id {id} down: rows");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A union over a layer with an arena runs in pinned batches: the keep
    /// set (the arena ids with the most columns, the ties to the lower id)
    /// first, each batch claimed [`super::Claim::Keep`], then the rest as
    /// [`super::Claim::Transient`], each batch ascending; every read of a slot
    /// lands after its batch's ticket was waited for and while it is pinned, at
    /// most two batches are pinned at once, and every column is bit for bit
    /// the file mapping's union — a call of ten columns (the five passes) and
    /// one of eight (the narrow form's claimed quantizations and inline sum
    /// against the batches' quantize-first passes). A call whose arena ids fit
    /// the slots less two batches takes one claim of all of them. A layer under
    /// two batches and the lane's slots, and an expert past the stacks', are
    /// refused by name before any claim, and an unfilled slot by name with
    /// every pin released.
    ///
    /// PIN(2026-10-12): the floor is `2 × batch + ARENA_LANE_SLOTS` = 11
    /// slots, where it was 7 (the lane's one claim): the calls run on the
    /// floor's own slots, a keep set of five, not on ten slots and a keep set
    /// of four.
    #[test]
    fn a_tiered_union_runs_in_pinned_batches_keep_first() {
        use super::{Claim, TierError};
        let (embd, ff, n_expert, batch) = (256, 256, 14, 3);
        let lane = crate::placement::ARENA_LANE_SLOTS as usize;
        let tys = [GgmlType::Q4_K; 3];
        let path = layer_file("tierbatch", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let src = R8Source::rows(&split);
        let plain = build_layer(&split, embd, ff, n_expert).unwrap();
        // Ids 0 and 1 are the host segment's; ids 2..14 the arena's. `counts`
        // are the columns that list each id (an id is listed by the first
        // `counts[id]` columns, each list rotated by its column).
        let lists_of = |counts: &[usize; 14], cols: usize| -> Vec<Vec<(u32, f32)>> {
            (0..cols)
                .map(|j| {
                    let mut ids: Vec<u32> = (0..n_expert as u32)
                        .filter(|&e| counts[e as usize] > j)
                        .collect();
                    let turn = j % ids.len();
                    ids.rotate_left(turn);
                    ids.iter()
                        .enumerate()
                        .map(|(i, &e)| (e, 0.125 + 0.0625 * ((i + j) % 5) as f32))
                        .collect()
                })
                .collect()
        };
        let x_of = |cols: usize| {
            Tensor2::from_vec(
                embd,
                cols,
                (0..embd * cols)
                    .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                    .collect(),
            )
        };
        let mut us = UnionScratch::new_routed(embd, ff, 10, n_expert).unwrap();
        // The mapping's union of `lists` over `x`.
        let reference = |us: &mut UnionScratch, x: &Tensor2, lists: &[Vec<(u32, f32)>]| {
            let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
            let mut want = vec![f32::NAN; embd * lists.len()];
            plain
                .experts_union_into(src, x, &slices, &mut want, us)
                .unwrap();
            want
        };
        let tiered = |filled: bool, slots: usize| {
            let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
            let mut mock = MockTier::of(&split, &layer, filled);
            mock.slots = slots;
            let mock = std::sync::Arc::new(mock);
            layer.attach_tier(4, 0..2, mock.clone());
            layer.tier.as_mut().expect("the tier just attached").batch = batch;
            (layer, mock)
        };
        let diff = |got: &[f32], want: &[f32]| {
            got.iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
        };

        // Ten columns; the arena holds the floor's eleven slots (two batches of
        // three and the lane's five): keep 11 - 2 * 3 = 5 of the twelve ids.
        // The top five by columns, ties to the lower id: 13, 3, 5, 7, 8 — not
        // the ascending head 2, 3, 4, 5, 6.
        let counts = [3usize, 1, 1, 9, 1, 4, 1, 2, 2, 1, 2, 1, 1, 10];
        let (lists, x) = (lists_of(&counts, 10), x_of(10));
        let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let want = reference(&mut us, &x, &lists);
        let (layer, mock) = tiered(true, 2 * batch + lane);
        let mut got = vec![f32::NAN; embd * 10];
        layer
            .experts_union_into(src, &x, &slices, &mut got, &mut us)
            .unwrap();
        assert_eq!(
            diff(&got, &want),
            0,
            "the batched union differs from the mapping's"
        );
        let events = mock.events();
        let claims: Vec<(Claim, Vec<u32>)> = events
            .iter()
            .filter_map(|e| match e {
                PinEvent::Pinned(claim, ids) => Some((*claim, ids.clone())),
                PinEvent::Unpinned(_) => None,
            })
            .collect();
        let mut by_count: Vec<u32> = (2..n_expert as u32).collect();
        by_count.sort_by_key(|&e| (std::cmp::Reverse(counts[e as usize]), e));
        let mut kept: Vec<u32> = by_count[..5].to_vec();
        kept.sort_unstable();
        assert_eq!(kept, [3, 5, 7, 8, 13], "the keep set the test expects");
        let keep_claimed: Vec<u32> = claims[..2]
            .iter()
            .flat_map(|(_, ids)| ids.clone())
            .collect();
        assert_eq!(
            keep_claimed, kept,
            "the first batches are the keep set, ascending"
        );
        assert_eq!(
            claims,
            vec![
                (Claim::Keep, vec![3, 5, 7]),
                (Claim::Keep, vec![8, 13]),
                (Claim::Transient, vec![2, 4, 6]),
                (Claim::Transient, vec![9, 10, 11]),
                (Claim::Transient, vec![12]),
            ],
            "batches of three, one kind each, the keep set's first, ascending"
        );
        {
            let log = mock.pins.lock().unwrap();
            assert!(
                log.loose.is_empty(),
                "a slot was read outside a waited, pinned batch: {:?}",
                log.loose
            );
            assert_eq!(log.peak, 2 * batch, "two batches pinned at the most");
            assert!(log.held.is_empty(), "every pin released after the call");
        }
        let pinned = events
            .iter()
            .filter(|e| matches!(e, PinEvent::Pinned(..)))
            .count();
        assert_eq!(pinned, events.len() - pinned, "one unpin for every pin");
        let read: std::collections::BTreeSet<u32> =
            mock.asked().iter().map(|&(_, id, _)| id).collect();
        assert_eq!(
            read.iter().copied().collect::<Vec<_>>(),
            (2..n_expert as u32).collect::<Vec<_>>(),
            "every arena id read from its slot, no host id"
        );

        // A call whose arena ids (three) fit the slots less two batches: one
        // claim of all of them, as keep, and the same bits.
        let small: Vec<Vec<(u32, f32)>> = vec![
            vec![(4, 0.5), (0, 0.25), (9, 0.125)],
            vec![(11, 0.75), (1, 0.5)],
        ];
        let small_refs: Vec<&[(u32, f32)]> = small.iter().map(Vec::as_slice).collect();
        let xs = Tensor2::from_vec(embd, 2, [x.col(0), x.col(1)].concat());
        let want_small = reference(&mut us, &xs, &small);
        let mut got_small = vec![f32::NAN; 2 * embd];
        layer
            .experts_union_into(src, &xs, &small_refs, &mut got_small, &mut us)
            .unwrap();
        assert_eq!(
            diff(&got_small, &want_small),
            0,
            "the one-claim union differs from the mapping's"
        );
        assert_eq!(
            mock.events(),
            vec![
                PinEvent::Pinned(Claim::Keep, vec![4, 9, 11]),
                PinEvent::Unpinned(vec![4, 9, 11]),
            ],
            "a small call claims all its arena ids at once, as keep"
        );

        // The floor's own slots (two batches and the lane's five) serve the
        // call with a keep set of five, and the narrow form (eight columns:
        // claimed quantizations and an inline sum whole) matches the batches'
        // passes; the ties at the cut go to the lower id (3 and 13 both list
        // in eight columns).
        let counts8 = [3usize, 1, 1, 8, 1, 4, 1, 2, 2, 1, 2, 1, 1, 8];
        let (lists8, x8) = (lists_of(&counts8, 8), x_of(8));
        let slices8: Vec<&[(u32, f32)]> = lists8.iter().map(Vec::as_slice).collect();
        let want8 = reference(&mut us, &x8, &lists8);
        let (layer, mock) = tiered(true, 2 * batch + lane);
        let mut got8 = vec![f32::NAN; embd * 8];
        layer
            .experts_union_into(src, &x8, &slices8, &mut got8, &mut us)
            .unwrap();
        assert_eq!(
            diff(&got8, &want8),
            0,
            "the narrow call's batches differ from its whole"
        );
        assert_eq!(
            mock.events().first(),
            Some(&PinEvent::Pinned(Claim::Keep, vec![3, 5, 7])),
            "a keep set of five, the first batch its three lowest ids, the tie to the lower id"
        );
        assert!(mock.pins.lock().unwrap().loose.is_empty());
        // One slot under the floor is refused by name before any claim.
        let (layer, mock) = tiered(true, 2 * batch + lane - 1);
        match layer.experts_union_into(src, &x, &slices, &mut got, &mut us) {
            Err(crate::ModelError::Tier(TierError::BatchFloor {
                layer: 4,
                slots: 10,
                need: 11,
            })) => {}
            other => panic!("not the floor's refusal by name: {other:?}"),
        }
        assert!(
            mock.events().is_empty(),
            "the refusal came before any claim"
        );

        // An expert past the stacks' is refused before any claim.
        let (layer, mock) = tiered(true, 2 * batch + lane);
        let past = [(14u32, 1.0f32)];
        let past_refs: Vec<&[(u32, f32)]> = vec![&past[..]];
        let x1 = Tensor2::zeros(embd, 1);
        let e = layer
            .experts_union_into(src, &x1, &past_refs, &mut got[..embd], &mut us)
            .expect_err("an expert past the stacks'")
            .to_string();
        assert!(e.contains("expert 14"), "{e}");
        assert!(
            mock.events().is_empty(),
            "the refusal came before any claim"
        );

        // An unfilled slot is refused by name once its batch is pinned, and
        // the pins are released with the refusal.
        let (layer, mock) = tiered(false, 2 * batch + lane);
        match layer.experts_union_into(src, &x, &slices, &mut got, &mut us) {
            Err(crate::ModelError::Tier(TierError::Unfilled { layer: 4, id: 3 })) => {}
            other => panic!("not the unfilled refusal by name: {other:?}"),
        }
        assert!(
            mock.pins.lock().unwrap().held.is_empty(),
            "no pin outlives the refusal"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A tiered union call tells its tier its path once, before its first
    /// batch's claim: the call's distinct ids ascending (the host segment's
    /// included) with their column offsets, its columns, `batched` when its arena ids are
    /// more than the slots less two batches and not when they fit. A call
    /// refused before its path (a layer under the floor, an expert past the
    /// stacks') tells nothing.
    #[test]
    fn a_tiered_union_tells_its_tier_its_path_once() {
        let (embd, ff, n_expert, batch) = (256, 256, 14, 3);
        let lane = crate::placement::ARENA_LANE_SLOTS as usize;
        let path = layer_file("tiernote", [GgmlType::Q4_K; 3], embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let src = R8Source::rows(&split);
        let tiered = |slots: usize| {
            let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
            let mut mock = MockTier::of(&split, &layer, true);
            mock.slots = slots;
            let mock = std::sync::Arc::new(mock);
            layer.attach_tier(4, 0..2, mock.clone());
            layer.tier.as_mut().expect("the tier just attached").batch = batch;
            (layer, mock)
        };
        let x = Tensor2::from_vec(
            embd,
            2,
            (0..2 * embd)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        let mut us = UnionScratch::new_routed(embd, ff, 10, n_expert).unwrap();
        let mut out = vec![f32::NAN; 2 * embd];
        let (layer, mock) = tiered(2 * batch + lane);
        // Six arena ids (2..=7) over eleven slots less two batches (five):
        // batched. Id 4 is listed by both columns.
        let big = [
            vec![(2u32, 0.5f32), (3, 0.25), (4, 0.125), (7, 0.0625)],
            vec![(4, 0.5), (5, 0.25), (6, 0.125)],
        ];
        let refs: Vec<&[(u32, f32)]> = big.iter().map(Vec::as_slice).collect();
        layer
            .experts_union_into(src, &x, &refs, &mut out, &mut us)
            .unwrap();
        assert_eq!(
            mock.told(),
            vec![Told {
                layer: 4,
                ids: vec![2, 3, 4, 5, 6, 7],
                off: vec![0, 1, 2, 4, 5, 6, 7],
                positions: vec![0, 0, 0, 1, 1, 1, 0],
                cols: 2,
                batched: true
            }],
            "one note, batched, with the call's ids and offsets"
        );
        // Two arena ids (2 and 7) beside the host's id 0: one claim, whole.
        let small = [vec![(0u32, 0.5f32), (2, 0.25)], vec![(2, 0.5), (7, 0.25)]];
        let refs: Vec<&[(u32, f32)]> = small.iter().map(Vec::as_slice).collect();
        layer
            .experts_union_into(src, &x, &refs, &mut out, &mut us)
            .unwrap();
        assert_eq!(
            mock.told(),
            vec![Told {
                layer: 4,
                ids: vec![0, 2, 7],
                off: vec![0, 1, 3, 4],
                positions: vec![0, 0, 1, 1],
                cols: 2,
                batched: false
            }],
            "one note, not batched, the host segment's id in the ids"
        );
        // Refused before the path: a layer one slot under the floor, and an
        // expert past the stacks'.
        let (floor, floor_mock) = tiered(2 * batch + lane - 1);
        let refs: Vec<&[(u32, f32)]> = big.iter().map(Vec::as_slice).collect();
        floor
            .experts_union_into(src, &x, &refs, &mut out, &mut us)
            .expect_err("a layer under the floor");
        assert!(
            floor_mock.told().is_empty(),
            "the floor's refusal told nothing"
        );
        let past = [vec![(14u32, 1.0f32)]];
        let refs: Vec<&[(u32, f32)]> = past.iter().map(Vec::as_slice).collect();
        layer
            .experts_union_into(
                src,
                &Tensor2::zeros(embd, 1),
                &refs,
                &mut out[..embd],
                &mut us,
            )
            .expect_err("an expert past the stacks'");
        assert!(mock.told().is_empty(), "the stacks' refusal told nothing");
        std::fs::remove_file(&path).unwrap();
    }

    /// The host segment's ids read the mapping and every other id the tier's
    /// slot: an unfilled slot is refused by name, naming its layer and
    /// expert, never a silent fallback to the mapping — and a filled slot
    /// lends bytes the call turns into the mapping's own values, bit for
    /// bit, while the `ensure` hook saw the call's ids.
    #[test]
    fn a_tier_slot_is_lent_or_refused_by_name() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("tier", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let x = Tensor2::from_vec(
            embd,
            1,
            (0..embd)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        let list = [(0u32, 0.4f32), (2, 0.6)];
        let mut scratch = HostScratch::new(embd, ff, 3).unwrap();

        // The unfilled slot: the plan's host segment holds id 1 alone, so
        // the call's other ids are the tier's, whose slots lend nothing.
        let empty = std::sync::Arc::new(MockTier::of(&split, &layer, false));
        layer.attach_tier(4, 1..2, empty.clone());
        match layer.experts_into(
            R8Source::rows(&split),
            &x,
            &list,
            &mut vec![0.0; embd],
            &mut scratch,
        ) {
            Err(crate::ModelError::Tier(e @ super::TierError::Unfilled { layer: 4, id: 0 })) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("unfilled") && msg.contains('4') && msg.contains('0'),
                    "{msg}"
                );
            }
            other => panic!("not the unfilled refusal by name: {other:?}"),
        }
        assert_eq!(
            empty.ensured(),
            vec![(4, vec![0, 2])],
            "the ensure hook saw the call's ids at the tier's layer"
        );
        assert_eq!(
            empty.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, vec![0, 2]),
                PinEvent::Unpinned(vec![0, 2]),
            ],
            "a step's ids are pinned as the step's, and let go when the call refuses"
        );

        // The filled slot: bytes the call turns into the mapping's own
        // values — the no-tier path's, bit for bit.
        let mut full = MockTier::of(&split, &layer, true);
        full.slots = 2 * super::UNION_BATCH_IDS + 8;
        let full = std::sync::Arc::new(full);
        layer.attach_tier(4, 1..2, full.clone());
        let mut want = vec![f32::NAN; embd];
        layer
            .experts_into(R8Source::rows(&split), &x, &list, &mut want, &mut scratch)
            .unwrap();
        let plain = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut got = vec![f32::NAN; embd];
        plain
            .experts_into(R8Source::rows(&split), &x, &list, &mut got, &mut scratch)
            .unwrap();
        assert_eq!(
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a filled slot's call writes the mapping path's values"
        );
        assert_eq!(
            full.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, vec![0, 2]),
                PinEvent::Unpinned(vec![0, 2]),
            ],
            "the step's ids stay pinned for the call and are let go after it"
        );
        assert!(
            full.pins.lock().unwrap().loose.is_empty(),
            "every slot the step read was pinned"
        );

        // The union path's claim: its columns' tier ids pinned once, as keep,
        // for the call, and the call's bits the plain path's.
        let cols = 3;
        let xw = Tensor2::from_vec(
            embd,
            cols,
            (0..embd * cols)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        let lists: Vec<Vec<(u32, f32)>> = (0..cols)
            .map(|j| [(j as u32 % 3, 0.5), (1, 0.5)].to_vec())
            .collect();
        let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut us_tier = UnionScratch::new_routed(embd, ff, 8, 3).unwrap();
        let mut us_plain = UnionScratch::new_routed(embd, ff, 8, 3).unwrap();
        let mut a = vec![f32::NAN; embd * cols];
        layer
            .experts_union_into(R8Source::rows(&split), &xw, &slices, &mut a, &mut us_tier)
            .unwrap();
        let mut b = vec![f32::NAN; embd * cols];
        plain
            .experts_union_into(R8Source::rows(&split), &xw, &slices, &mut b, &mut us_plain)
            .unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a tier-bearing union call writes the plain path's values"
        );
        assert_eq!(
            full.events(),
            vec![
                PinEvent::Pinned(super::Claim::Keep, vec![0, 2]),
                PinEvent::Unpinned(vec![0, 2]),
            ],
            "the union claimed the call's tier ids at once, as keep"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A decode step's union reads the tier's ids from their slots: the
    /// call's distinct ids outside the host segment, ascending, go to
    /// `ensure` once; an unfilled slot is refused by name with its layer and
    /// the first such id; a filled one gives the bits of the mapping's union
    /// — a layer with no tier and the same layer's mapping reader alike —
    /// and the tier is asked for every tier id's three parts and never for
    /// an id of the host segment; the same layer's prompt union reads the
    /// arena too and writes the same bits.
    #[test]
    fn a_step_union_reads_the_tier_from_its_slots() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("stepunion", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let cols = 3;
        let xw = Tensor2::from_vec(
            embd,
            cols,
            (0..embd * cols)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        // Host id 1 in two columns; tier ids 5, 2, 0 and 3, id 5 in two.
        let lists: [&[(u32, f32)]; 3] = [
            &[(5, 0.3), (1, 0.7)],
            &[(2, 0.5), (1, 0.25), (0, 0.25)],
            &[(5, 0.6), (3, 0.4)],
        ];
        let tier_ids = [0u32, 2, 3, 5];
        let claim_ids = [0u32, 1, 2, 3, 5];
        let src = R8Source::rows(&split);
        let mut us = UnionScratch::new_routed(embd, ff, 8, 3).unwrap();

        let empty = std::sync::Arc::new(MockTier::of(&split, &layer, false));
        layer.attach_tier(4, 1..2, empty.clone());
        match layer.experts_step_union_into(src, &xw, &lists, &mut vec![0.0; embd * cols], &mut us)
        {
            Err(crate::ModelError::Tier(e @ super::TierError::Unfilled { layer: 4, id: 0 })) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("unfilled") && msg.contains("layer 4") && msg.contains("expert 0"),
                    "{msg}"
                );
            }
            other => panic!("not the unfilled refusal by name: {other:?}"),
        }
        assert_eq!(
            empty.ensured(),
            vec![(4, claim_ids.to_vec())],
            "ensure saw the full plan's distinct ids once, ascending, at the tier's layer"
        );
        assert_eq!(
            empty.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, claim_ids.to_vec()),
                PinEvent::Unpinned(claim_ids.to_vec()),
            ],
            "the mock holds the full step plan and lets go when it refuses"
        );

        let mut full = MockTier::of(&split, &layer, true);
        full.slots = 2 * super::UNION_BATCH_IDS + 8;
        full.late = Some(Vec::new());
        let full = std::sync::Arc::new(full);
        layer.attach_tier(4, 1..2, full.clone());
        let mut got = vec![f32::NAN; embd * cols];
        layer
            .experts_step_union_into(src, &xw, &lists, &mut got, &mut us)
            .unwrap();
        assert_eq!(
            full.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, claim_ids.to_vec()),
                PinEvent::Unpinned(claim_ids.to_vec()),
            ],
            "the mock holds the full step plan for the call and lets go after it"
        );
        assert!(
            full.pins.lock().unwrap().loose.is_empty(),
            "every slot the step union read was pinned"
        );
        assert_eq!(
            full.told(),
            vec![Told {
                layer: 4,
                ids: claim_ids.to_vec(),
                off: vec![0, 1, 3, 4, 5, 7],
                positions: vec![1, 0, 1, 1, 2, 0, 2],
                cols,
                batched: false,
            }],
            "the step claim carries each id's routed columns, including host ids"
        );
        let asked = full.asked();
        assert!(
            asked.iter().all(|&(l, id, _)| l == 4 && id != 1),
            "the tier was asked for a host-segment id or another layer: {asked:?}"
        );
        let asked: std::collections::BTreeSet<(u32, usize)> =
            asked.iter().map(|&(_, id, part)| (id, part)).collect();
        let want_asked: std::collections::BTreeSet<(u32, usize)> = tier_ids
            .iter()
            .flat_map(|&id| (0..3).map(move |part| (id, part)))
            .collect();
        assert_eq!(
            asked, want_asked,
            "every tier id's three parts, from the tier"
        );

        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let mut prompt = vec![f32::NAN; embd * cols];
        layer
            .experts_union_into(src, &xw, &lists, &mut prompt, &mut us)
            .unwrap();
        let plain = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut want = vec![f32::NAN; embd * cols];
        plain
            .experts_union_into(src, &xw, &lists, &mut want, &mut us)
            .unwrap();
        assert_eq!(
            bits(&got),
            bits(&want),
            "the step union writes a tier-less union's bits"
        );
        assert_eq!(
            bits(&prompt),
            bits(&want),
            "the same layer's prompt union, from the arena, writes a tier-less union's bits"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A decode step whose claim has misses in flight computes the host
    /// segment's experts and the claim's hits first, waits for the ticket,
    /// and computes the misses: the misses lend nothing before the wait, a
    /// hit's down part is read by its pass before the wait, and each column
    /// is the one-column leg's bit for bit — lists of four experts of mixed
    /// weights, so a sum in another order moves bits. Two groups of two
    /// claiming passes run.
    #[test]
    fn a_step_union_computes_its_hits_under_the_misses_fill() {
        let (embd, ff, n_expert) = (256, 256, 8);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("stepsplit", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let cols = 3;
        let xw = Tensor2::from_vec(
            embd,
            cols,
            (0..embd * cols)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        // Host id 1; tier ids 0, 2, 3, 5, 6, 7 of which 0 and 6 are in flight.
        let lists: [&[(u32, f32)]; 3] = [
            &[(5, 0.9), (1, 1e-3), (2, 0.25), (0, 40.0)],
            &[(2, 0.5), (6, 3e-4), (1, 0.25), (5, 7.0)],
            &[(7, 0.6), (3, 2.0), (5, 1e-2), (0, 0.125)],
        ];
        let claim_ids = [0u32, 1, 2, 3, 5, 6, 7];
        let (late, hits) = ([0u32, 6], [2u32, 3, 5, 7]);
        let src = R8Source::rows(&split);
        let mut us = UnionScratch::new_routed(embd, ff, 8, 4).unwrap();
        let plain = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut hs = HostScratch::new(embd, ff, 4).unwrap();
        let mut want = vec![f32::NAN; embd * cols];
        for (j, list) in lists.iter().enumerate() {
            let xj = Tensor2::from_vec(embd, 1, xw.data[j * embd..(j + 1) * embd].to_vec());
            plain
                .experts_into(src, &xj, list, &mut want[j * embd..(j + 1) * embd], &mut hs)
                .unwrap();
        }

        let mut tier = MockTier::of(&split, &layer, true);
        tier.slots = 2 * super::UNION_BATCH_IDS + 8;
        tier.late = Some(late.to_vec());
        let tier = std::sync::Arc::new(tier);
        layer.attach_tier(4, 1..2, tier.clone());
        let mut got = vec![f32::NAN; embd * cols];
        let passes = us.passes();
        layer
            .experts_step_union_into(src, &xw, &lists, &mut got, &mut us)
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got),
            bits(&want),
            "the step split in two groups writes the one-column leg's bits"
        );
        assert_eq!(us.passes() - passes, 4, "two groups of two claiming passes");

        let trail = tier.trail.lock().unwrap().clone();
        let wait = trail
            .iter()
            .position(|t| *t == Trail::Wait)
            .expect("the ticket was waited for");
        let before = |id: u32, part: usize| {
            trail[..wait]
                .iter()
                .filter(|t| **t == Trail::Lent(id, part))
                .count()
        };
        for id in late {
            assert_eq!(
                (0..3).map(|part| before(id, part)).sum::<usize>(),
                0,
                "expert {id} lent a part before its fill ended"
            );
            assert!(
                trail[wait..]
                    .iter()
                    .any(|t| matches!(t, Trail::Lent(i, 2) if *i == id)),
                "expert {id} was computed after the wait"
            );
        }
        for id in hits {
            assert!(
                before(id, 2) >= 2,
                "expert {id}'s down part was read once by the lent check and again by its pass \
                 before the wait, got {} reads",
                before(id, 2)
            );
        }
        assert!(
            tier.pins.lock().unwrap().loose.is_empty(),
            "every slot the step read was pinned"
        );
        assert_eq!(
            tier.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, claim_ids.to_vec()),
                PinEvent::Unpinned(claim_ids.to_vec()),
            ],
            "one claim for the full plan, let go after it"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// The probe classifies a hit once, and the claim's pin holds it until the
    /// call ends. A tier that broke the pin — the mock evicts a hit the moment
    /// after its probe — is refused by name, before the ticket is waited for:
    /// the hit is not recomputed as a miss, and nothing is computed from an
    /// unfilled slot.
    #[test]
    fn a_hit_evicted_after_its_probe_is_refused_by_name() {
        let (embd, ff, n_expert) = (256, 256, 8);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("stepevict", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let xw = Tensor2::from_vec(
            embd,
            1,
            (0..embd).map(|i| (i % 13) as f32 / 7.0 - 0.9).collect(),
        );
        // Host id 1; tier ids 2 and 5 held by the claim, 0 in flight.
        let lists: [&[(u32, f32)]; 1] = [&[(5, 0.9), (1, 0.1), (2, 0.25), (0, 0.5)]];
        let src = R8Source::rows(&split);
        let mut us = UnionScratch::new_routed(embd, ff, 8, 4).unwrap();
        let mut tier = MockTier::of(&split, &layer, true);
        tier.slots = 2 * super::UNION_BATCH_IDS + 8;
        tier.late = Some(vec![0]);
        tier.evict = Some(5);
        let tier = std::sync::Arc::new(tier);
        layer.attach_tier(4, 1..2, tier.clone());
        let mut got = vec![f32::NAN; embd];
        match layer.experts_step_union_into(src, &xw, &lists, &mut got, &mut us) {
            Err(crate::ModelError::Tier(super::TierError::Unfilled { layer: 4, id: 5 })) => {}
            other => panic!("not the unfilled refusal of the evicted hit: {other:?}"),
        }
        assert!(
            !tier.trail.lock().unwrap().contains(&Trail::Wait),
            "the refusal came before the wait: the hit was not treated as a miss"
        );
        assert_eq!(
            tier.events(),
            vec![
                PinEvent::Pinned(super::Claim::Step, vec![0, 1, 2, 5]),
                PinEvent::Unpinned(vec![0, 1, 2, 5]),
            ],
            "the claim is let go whole"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// One column of a decode step, the resident path: the hits compute while
    /// the misses fill, and the list-order sum is the plain leg's, bit for
    /// bit. A ready ticket is not this case; the mock's ticket stays pending
    /// so the split runs.
    #[test]
    fn a_one_column_step_computes_its_hits_under_the_misses_fill() {
        let (embd, ff, n_expert) = (256, 256, 8);
        let tys = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K];
        let path = layer_file("stepcol", tys, embd, ff, n_expert);
        let split = gguf::Split::open(&path).unwrap();
        let mut layer = build_layer(&split, embd, ff, n_expert).unwrap();
        let x = Tensor2::from_vec(
            embd,
            1,
            (0..embd)
                .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5)
                .collect(),
        );
        // Host id 1. Tier ids 0, 2, 5; 0 is in flight, 2 and 5 are hits.
        // The weights differ enough that a sum in another order moves bits.
        let list = [(5u32, 0.9), (1, 1e-3), (2, 0.25), (0, 40.0)];
        let (late, hits) = ([0u32], [2u32, 5]);
        let src = R8Source::rows(&split);
        let mut hs = HostScratch::new(embd, ff, 4).unwrap();
        let mut us = UnionScratch::new_routed(embd, ff, 1, 4).unwrap();
        let plain = build_layer(&split, embd, ff, n_expert).unwrap();
        let mut want = vec![f32::NAN; embd];
        plain
            .experts_into(src, &x, &list, &mut want, &mut hs)
            .unwrap();

        let mut tier = MockTier::of(&split, &layer, true);
        tier.slots = 2 * super::UNION_BATCH_IDS + 8;
        tier.late = Some(late.to_vec());
        let tier = std::sync::Arc::new(tier);
        layer.attach_tier(4, 1..2, tier.clone());
        let mut got = vec![f32::NAN; embd];
        let passes = us.passes();
        layer
            .experts_into_on(
                crate::ops::Lanes::Flat,
                src,
                &x,
                &list,
                &mut got,
                &mut hs,
                Some(&mut us),
                1,
            )
            .unwrap();
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got),
            bits(&want),
            "the one-column split writes the plain leg's bits"
        );
        assert_eq!(us.passes() - passes, 4, "two groups of two claiming passes");
        let trail = tier.trail.lock().unwrap().clone();
        let wait = trail
            .iter()
            .position(|t| *t == Trail::Wait)
            .expect("the ticket was waited for");
        let before = |id: u32, part: usize| {
            trail[..wait]
                .iter()
                .filter(|t| **t == Trail::Lent(id, part))
                .count()
        };
        for id in late {
            assert_eq!(
                (0..3).map(|part| before(id, part)).sum::<usize>(),
                0,
                "expert {id} lent a part before its fill ended"
            );
        }
        for id in hits {
            assert!(
                before(id, 2) >= 2,
                "expert {id}'s down was read before the wait"
            );
        }
        assert!(
            tier.pins.lock().unwrap().loose.is_empty(),
            "every slot the step read was pinned"
        );
        std::fs::remove_file(&path).unwrap();
    }
}

/// The union call over located `NVFP4_ROWS` stacks in the planes layout: the
/// host tier's read of a checkpoint folder's experts in place.
#[cfg(test)]
mod planes_union_tests {
    use super::{Claim, LegRun, TierError, TierPin, TierSlots, UnionScratch, serve_union};
    use crate::ops::planes_tests::{
        ORDERS, Order, Shards, contiguous, place, planes_matrix, stack_info,
    };
    use crate::ops::{ExpertForm, ExpertInput, ExpertStack, Lanes, StackRuns, StackTier, Tensor2};
    use gguf::GgmlType;
    use gguf::quant::Nvfp4Rows;

    const TY: GgmlType = GgmlType::NVFP4_ROWS;
    const EMBD: usize = 160;
    const FF: usize = 96;
    const N_EXPERT: usize = 6;

    /// A checkpoint folder's layer: expert `e`'s gate, up and down — the
    /// checkpoint's w1, w3 and w2 — each stack's matrices in a scrambled order
    /// with junk between the runs, the three stacks in one file. `mats[e]` is
    /// `[gate, up, down]` as made.
    struct Folder {
        span: Vec<u8>,
        runs: [StackRuns; 3],
        mats: Vec<[Vec<u8>; 3]>,
    }

    /// Expert `e`'s tensor scale in stack `stack`: distinct and not powers of
    /// two, so a scale left out, swapped between stacks or between experts
    /// moves the answer far past any band.
    fn scale(stack: usize, e: usize) -> f32 {
        0.2 + 0.125 * e as f32 + 0.5 * stack as f32
    }

    fn scales(stack: usize) -> Vec<f32> {
        (0..N_EXPERT).map(|e| scale(stack, e)).collect()
    }

    fn folder(order: Order) -> Folder {
        let mats: Vec<[Vec<u8>; 3]> = (0..N_EXPERT)
            .map(|e| {
                let s = 1000 * e as u64;
                [
                    planes_matrix(EMBD, FF, s + 1),
                    planes_matrix(EMBD, FF, s + 2),
                    planes_matrix(FF, EMBD, s + 3),
                ]
            })
            .collect();
        let stack = |s: usize| -> Vec<Vec<u8>> { mats.iter().map(|m| m[s].clone()).collect() };
        let mut span = vec![0xEE; 24];
        let shapes = [(EMBD, FF), (EMBD, FF), (FF, EMBD)];
        let runs = [0, 2, 1].map(|s| place(&mut span, &stack(s), shapes[s], order));
        // `place` ran in the order gate, down, up: put them back in stack order.
        let [gate, down, up] = runs;
        Folder {
            span,
            runs: [gate, up, down],
            mats,
        }
    }

    /// The same matrices as three contiguous runs, expert after expert, each
    /// expert's codes then its scales.
    fn plain(f: &Folder) -> ([Vec<u8>; 3], [StackRuns; 3]) {
        let spans: [Vec<u8>; 3] =
            std::array::from_fn(|s| f.mats.iter().flat_map(|m| m[s].iter().copied()).collect());
        let shapes = [(EMBD, FF), (EMBD, FF), (FF, EMBD)];
        let runs = shapes.map(|(k, n)| contiguous(k, n, N_EXPERT));
        (spans, runs)
    }

    fn lists(cols: usize) -> Vec<Vec<(u32, f32)>> {
        (0..cols)
            .map(|j| {
                let w = 0.25 + 0.125 * j as f32;
                vec![
                    ((j % N_EXPERT) as u32, w),
                    (((5 * j + 1) % N_EXPERT) as u32, -w),
                ]
            })
            .collect()
    }

    /// `serve_union` over the stacks `spans` and `runs` make, `x`'s columns
    /// and `lists`.
    fn run(
        spans: [&[u8]; 3],
        runs: &[StackRuns; 3],
        x: &Tensor2,
        lists: &[Vec<(u32, f32)>],
    ) -> Vec<f32> {
        run_on(spans, runs, x, lists, None)
    }

    /// [`run`] with the ids of `tier` (outside its host segment) read from its
    /// slots instead of the stacks' own bytes.
    fn run_on(
        spans: [&[u8]; 3],
        runs: &[StackRuns; 3],
        x: &Tensor2,
        lists: &[Vec<(u32, f32)>],
        tier: Option<(&Slots, std::ops::Range<u32>)>,
    ) -> Vec<f32> {
        let infos = [
            stack_info("gate_exps", [EMBD, FF, N_EXPERT]),
            stack_info("up_exps", [EMBD, FF, N_EXPERT]),
            stack_info("down_exps", [FF, EMBD, N_EXPERT]),
        ];
        let tensor_scales = [scales(0), scales(1), scales(2)];
        let files = spans.map(|span| Shards(vec![span]));
        let stacks: [ExpertStack<'_>; 3] = std::array::from_fn(|s| {
            let stack =
                ExpertStack::located(&infos[s], &files[s], &runs[s], Some(&tensor_scales[s]))
                    .expect("a stack that fits its runs");
            match &tier {
                Some((slots, host)) => stack.with_tier(StackTier::new(*slots, 0, host, s)),
                None => stack,
            }
        });
        let cols = x.ne1;
        let mut us = UnionScratch::new_routed(EMBD, FF, cols, 2).unwrap();
        let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut out = vec![0.0f32; cols * EMBD];
        serve_union(
            stacks,
            LegRun::flat(),
            x.view(),
            &refs,
            &mut out,
            &mut us,
            None,
        )
        .expect("the union call runs");
        out
    }

    fn quantized(v: &[f32]) -> Vec<u8> {
        let mut q = vec![0u8; qdot::col_bytes(TY, v.len())];
        qdot::quantize_col(TY, v, &mut q);
        q
    }

    /// `m`'s `n` rows of `k` values (codes then scales, as made) dotted with
    /// the quantized column `acol`, each at `scale`.
    fn dots(m: &[u8], k: usize, n: usize, acol: &[u8], scale: f32) -> Vec<f32> {
        let (codes, scales) = m.split_at(n * k / 2);
        (0..n)
            .map(|r| {
                qdot::dot_nvfp4_planes_row(
                    &codes[r * k / 2..][..k / 2],
                    &scales[r * k / 16..][..k / 16],
                    acol,
                    k,
                ) * scale
            })
            .collect()
    }

    /// One column's leg written out on the source matrices, never through a
    /// stack: x quantized, each listed expert's gate and up dotted at its
    /// scale, the combine quantized, the down dotted at its scale, the
    /// weighted sum in list order from zero.
    fn transcription(f: &Folder, x: &[f32], list: &[(u32, f32)]) -> Vec<f32> {
        let xq = quantized(x);
        let mut out = vec![0.0f32; EMBD];
        for &(e, w) in list {
            let (m, e) = (&f.mats[e as usize], e as usize);
            let gate = dots(&m[0], EMBD, FF, &xq, scale(0, e));
            let up = dots(&m[1], EMBD, FF, &xq, scale(1, e));
            let mut par = vec![0.0f32; FF];
            assert!(qdot::swiglu_clamp(&gate, &up, 0.0, &mut par));
            let down = dots(&m[2], FF, EMBD, &quantized(&par), scale(2, e));
            for (o, d) in out.iter_mut().zip(&down) {
                *o += w * d;
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|v| v.to_bits()).collect()
    }

    fn column(c: usize) -> Vec<f32> {
        (0..EMBD)
            .map(|i| (((i * 31 + c * 17) % 97) as f32 - 48.0) / 40.0)
            .collect()
    }

    /// The union call over a located stack — the experts where the folder
    /// stores them, in no order, an expert's scales run after its codes run or
    /// a region away — gives the bits of the same experts in one contiguous
    /// planes run each, and the bits of the leg written out on the source
    /// matrices, at 1 and 3 columns (the claiming passes) and 9 (the pool's
    /// five).
    #[test]
    fn a_located_planes_stack_gives_the_union_call_the_contiguous_run_bits() {
        for order in ORDERS {
            let f = folder(order);
            let (spans, plain) = plain(&f);
            for cols in [1usize, 3, 9] {
                let lists = lists(cols);
                let xs: Vec<Vec<f32>> = (0..cols).map(column).collect();
                let x = Tensor2::from_vec(EMBD, cols, xs.concat());
                let got = run([&f.span, &f.span, &f.span], &f.runs, &x, &lists);
                let want = run([&spans[0], &spans[1], &spans[2]], &plain, &x, &lists);
                assert!(
                    got.iter().all(|v| v.is_finite()) && got.iter().any(|v| v.abs() > 1e-3),
                    "{order:?} {cols} columns: the call's values are finite and not all zero"
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{order:?} {cols} columns: located against contiguous"
                );
                for (j, (xc, list)) in xs.iter().zip(&lists).enumerate() {
                    assert_eq!(
                        bits(&got[j * EMBD..(j + 1) * EMBD]),
                        bits(&transcription(&f, xc, list)),
                        "{order:?} {cols} columns: column {j} against the leg on the source matrices"
                    );
                }
            }
        }
    }

    /// [`run`] for a relu² layer: the folder's up stack twice, in the gate's slot again as
    /// `HostLayer::stacks` holds a relu² layer's, then its down; `spans` and `runs` as [`run`]'s,
    /// the up's the second.
    fn run_relu2(
        spans: [&[u8]; 3],
        runs: &[StackRuns; 3],
        x: &Tensor2,
        lists: &[Vec<(u32, f32)>],
    ) -> Vec<f32> {
        let infos = [
            stack_info("up_exps", [EMBD, FF, N_EXPERT]),
            stack_info("down_exps", [FF, EMBD, N_EXPERT]),
        ];
        let tensor_scales = [scales(1), scales(2)];
        let files = spans.map(|span| Shards(vec![span]));
        let stack = |at: usize, part: usize| {
            ExpertStack::located(
                &infos[at],
                &files[part],
                &runs[part],
                Some(&tensor_scales[at]),
            )
            .expect("a stack that fits its runs")
        };
        let stacks = [stack(0, 1), stack(0, 1), stack(1, 2)];
        let cols = x.ne1;
        let mut us = UnionScratch::new_routed(EMBD, FF, cols, 2).unwrap();
        let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut out = vec![0.0f32; cols * EMBD];
        let relu2 = LegRun {
            form: ExpertForm::Relu2,
            input: ExpertInput::Direct,
            lanes: Lanes::Flat,
            post: None,
        };
        serve_union(stacks, relu2, x.view(), &refs, &mut out, &mut us, None)
            .expect("the union call runs");
        out
    }

    /// One column's relu² leg written out on the source matrices, never through a stack: x
    /// quantized, each listed expert's up dotted at its scale, `max(up, 0)²`, quantized, the down
    /// dotted at its scale, the weighted sum in list order from zero.
    fn transcription_relu2(f: &Folder, x: &[f32], list: &[(u32, f32)]) -> Vec<f32> {
        let xq = quantized(x);
        let mut out = vec![0.0f32; EMBD];
        for &(e, w) in list {
            let (m, e) = (&f.mats[e as usize], e as usize);
            let up = dots(&m[1], EMBD, FF, &xq, scale(1, e));
            let par: Vec<f32> = up
                .iter()
                .map(|&v| {
                    let r = v.max(0.0);
                    r * r
                })
                .collect();
            let down = dots(&m[2], FF, EMBD, &quantized(&par), scale(2, e));
            for (o, d) in out.iter_mut().zip(&down) {
                *o += w * d;
            }
        }
        out
    }

    /// The host tier serves a checkpoint folder's NVFP4 experts in the relu² form through the
    /// common planes row (`qdot::dot_nvfp4_planes_row`): the union call over a located relu²
    /// layer — the up in the gate's slot again, the down — gives the bits of the same experts in
    /// one contiguous planes run each and the bits of the leg written out on the source matrices,
    /// at 1, 3 and 9 columns in each order the folder may store the matrices. Mutant: the combine
    /// writing `max(up, 0)` (`relu2_into`'s `r * r` as `r`), which the transcription's square
    /// differs from.
    #[test]
    fn a_located_planes_relu2_layer_gives_the_union_call_the_transcription_bits() {
        for order in ORDERS {
            let f = folder(order);
            let (spans, plain) = plain(&f);
            for cols in [1usize, 3, 9] {
                let lists = lists(cols);
                let xs: Vec<Vec<f32>> = (0..cols).map(column).collect();
                let x = Tensor2::from_vec(EMBD, cols, xs.concat());
                let got = run_relu2([&f.span, &f.span, &f.span], &f.runs, &x, &lists);
                let want = run_relu2([&spans[0], &spans[1], &spans[2]], &plain, &x, &lists);
                assert!(
                    got.iter().all(|v| v.is_finite()) && got.iter().any(|v| v.abs() > 1e-3),
                    "{order:?} {cols} columns: the call's values are finite and not all zero"
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{order:?} {cols} columns: located against contiguous"
                );
                for (j, (xc, list)) in xs.iter().zip(&lists).enumerate() {
                    assert_eq!(
                        bits(&got[j * EMBD..(j + 1) * EMBD]),
                        bits(&transcription_relu2(&f, xc, list)),
                        "{order:?} {cols} columns: column {j} against the relu² leg on the source matrices"
                    );
                }
            }
        }
    }

    /// A tier whose slots lend every `(id, part)` as one run of the matrix as
    /// made — a planes expert's codes then its scales, what the arena holds —
    /// or, `filled` off, nothing. The `slot` of the trait and no more: a
    /// union of stacks over it reads nothing else.
    struct Slots {
        mats: Vec<[Vec<u8>; 3]>,
        filled: bool,
    }

    impl TierSlots for Slots {
        fn ensure_as(&self, _: usize, _: &[u32], _: Claim) -> Result<(), TierError> {
            unreachable!("a stack over these slots reads `slot` alone")
        }
        fn pin<'a>(&'a self, _: usize, _: &[u32]) -> Result<TierPin<'a>, TierError> {
            unreachable!("a stack over these slots reads `slot` alone")
        }
        fn slots(&self, _: usize) -> usize {
            unreachable!("a stack over these slots reads `slot` alone")
        }
        fn ensure_pinned<'a>(
            &'a self,
            _: usize,
            _: &[u32],
            _: Claim,
        ) -> Result<TierPin<'a>, TierError> {
            unreachable!("a stack over these slots reads `slot` alone")
        }
        fn slot(&self, _: usize, id: u32, part: usize) -> Option<&[u8]> {
            self.filled.then(|| &self.mats[id as usize][part][..])
        }
    }

    /// A planes stack lent by the NVMe tier reads its slot — one run of the
    /// matrix, codes then scales — as the same two runs the file holds: the
    /// union call over a tier whose host segment is two of the six experts
    /// gives the bits of the file-only call, the warm spans of a lent id are
    /// the codes' and the scales' of the matrix as made, and an unfilled slot
    /// warms nothing.
    #[test]
    fn a_planes_slot_is_read_as_the_two_runs_of_its_matrix() {
        let f = folder(Order::Separated);
        let slots = Slots {
            mats: f.mats.clone(),
            filled: true,
        };
        let host = 2..4u32;
        let cols = 9;
        let lists = lists(cols);
        let xs: Vec<Vec<f32>> = (0..cols).map(column).collect();
        let x = Tensor2::from_vec(EMBD, cols, xs.concat());
        let spans = [&f.span[..], &f.span[..], &f.span[..]];
        let want = run(spans, &f.runs, &x, &lists);
        let got = run_on(spans, &f.runs, &x, &lists, Some((&slots, host.clone())));
        assert_eq!(
            bits(&got),
            bits(&want),
            "a lent slot against the file's runs"
        );
        let empty = Slots {
            mats: Vec::new(),
            filled: false,
        };
        let info = stack_info("gate_exps", [EMBD, FF, N_EXPERT]);
        let tensor_scales = scales(0);
        let files = Shards(vec![&f.span]);
        let located =
            || ExpertStack::located(&info, &files, &f.runs[0], Some(&tensor_scales)).unwrap();
        let lent = located().with_tier(StackTier::new(&slots, 0, &host, 0));
        let unfilled = located().with_tier(StackTier::new(&empty, 0, &host, 0));
        let cp = EMBD * FF / 2;
        for e in 0..N_EXPERT as u32 {
            let (codes, plane) = f.mats[e as usize][0].split_at(cp);
            let spans = lent
                .warm_spans(e)
                .expect("a filled slot or the file's runs");
            let [Some(c), Some(p)] = spans else {
                panic!("expert {e}: a planes matrix warms two spans, got {spans:?}")
            };
            assert_eq!((c.bytes, p.bytes), (codes, plane), "expert {e}'s spans");
            assert_eq!((c.units, p.units), (FF, FF), "expert {e}'s units");
            assert_eq!(
                unfilled.warm_spans(e).is_some(),
                host.contains(&e),
                "expert {e}: only the host segment's ids warm with no slot filled"
            );
        }
    }

    /// `(scale · Σ w·a, band)` of one planes cell: the exact f64 dot of the
    /// row's unscaled dequantized values `w` with the activation `a` the call
    /// quantizes, times the stack's tensor scale, and the band of the leg's
    /// value, derived before the run. Per 32-value block `b` of `a`, `W_b =
    /// Σ|w|` and `M_b = max|a|` over it:
    ///
    /// 1. `a`'s q8_2 quantization: `d_a = bf16(M_b/127)` is within 2^-8 of
    ///    `M_b/127`, a code `nearest_int(fl(a·fl(1/d_a)))` is at most 127 in
    ///    magnitude, and `|d_a·q − a| ≤ (M_b/254)(1 + 2^-7)`. Summed:
    ///    `Σ_b W_b·(M_b/254)(1 + 2^-7)`.
    /// 2. Float order. A block is two exact integer sums, each times its
    ///    group's UE4M3 half (exact), added (one rounding), times `d_a` (one),
    ///    and added to the running sum in block order (one a block): `nb` adds
    ///    and two roundings, then the tensor scale's one multiply: `n = nb + 3`,
    ///    `γ_n·Σ|leaves|` with `|w·d_a·q| ≤ |w|·M_b·(128/127)(1 + 2^-7)`.
    /// 3. The f64 reference's own sum: `k·2^-53·Σ|w·a|`.
    ///
    /// All three are multiplied by the tensor scale.
    fn cell(w: &[f32], a: &[f32], scale: f32) -> (f64, f64) {
        let n = (w.len() / 32 + 3) as f64;
        let u = 2f64.powi(-24);
        let (mut quant, mut leaves, mut dot, mut wa) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (bw, ba) in w.chunks(32).zip(a.chunks(32)) {
            let m = ba.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
            let wb: f64 = bw.iter().map(|&v| f64::from(v).abs()).sum();
            quant += wb * m / 254.0 * (1.0 + 2f64.powi(-7));
            leaves += wb * m * (128.0 / 127.0) * (1.0 + 2f64.powi(-7));
            for (&x, &y) in bw.iter().zip(ba) {
                dot += f64::from(x) * f64::from(y);
                wa += (f64::from(x) * f64::from(y)).abs();
            }
        }
        let band = quant + n * u / (1.0 - n * u) * leaves + w.len() as f64 * 2f64.powi(-53) * wa;
        (f64::from(scale) * dot, f64::from(scale) * band)
    }

    /// The dequantized values of row `r` of `m` (`n` rows of `k` values, codes
    /// then scales, as made), tensor scale left out.
    fn dequantized(m: &[u8], k: usize, n: usize, r: usize) -> Vec<f32> {
        let (codes, scales) = m.split_at(n * k / 2);
        let mut w = vec![0.0f32; k];
        Nvfp4Rows::new(1.0)
            .decode_planes(
                &codes[r * k / 2..][..k / 2],
                &scales[r * k / 16..][..k / 16],
                &mut w,
            )
            .unwrap();
        w
    }

    /// The planes twin of the NVFP4 blocks' dequant gate
    /// (`tests/union.rs`): a located layer served for six slots, one expert a
    /// column at weight 1.0 so each output column is its slot's down, with
    ///
    /// - stage 1, each gate and up row's value (the planes row dot of the
    ///   call's quantized column, times the expert's tensor scale) within
    ///   [`cell`]'s band of the f64 dot of the row's dequantized values with the
    ///   activation, times the scale;
    /// - stage 2, each output cell within the band of the f64 dot of the down
    ///   row's dequantized values with the combine from those stage-1 values,
    ///   times the down's scale.
    #[test]
    fn a_located_planes_layer_is_within_the_derived_band_of_the_dequant_dot() {
        for order in ORDERS {
            let f = folder(order);
            let slots: Vec<u32> = (0..N_EXPERT as u32).rev().collect();
            let lists: Vec<Vec<(u32, f32)>> = slots.iter().map(|&e| vec![(e, 1.0)]).collect();
            let xs: Vec<Vec<f32>> = (0..slots.len()).map(column).collect();
            let x = Tensor2::from_vec(EMBD, slots.len(), xs.concat());
            let got = run([&f.span, &f.span, &f.span], &f.runs, &x, &lists);
            let (mut outside, mut worst) = (0usize, 0.0f64);
            let mut judge = |what: String, got: f64, (exact, band): (f64, f64)| {
                let err = (got - exact).abs();
                if err.is_nan() || err > band {
                    outside += 1;
                    eprintln!("{what}: {got:e} is {err:e} from {exact:e}, band {band:e}");
                } else if band > 0.0 {
                    worst = worst.max(err / band);
                }
            };
            for (j, &e) in slots.iter().enumerate() {
                let e = e as usize;
                let xq = quantized(&xs[j]);
                let mut stage1 = [vec![0.0f32; FF], vec![0.0f32; FF]];
                for (stack, out) in stage1.iter_mut().enumerate() {
                    *out = dots(&f.mats[e][stack], EMBD, FF, &xq, scale(stack, e));
                    for (r, &v) in out.iter().enumerate() {
                        let w = dequantized(&f.mats[e][stack], EMBD, FF, r);
                        let cell = cell(&w, &xs[j], scale(stack, e));
                        judge(
                            format!("{order:?} stage 1 stack {stack} slot {j} row {r}"),
                            f64::from(v),
                            cell,
                        );
                    }
                }
                let mut h = vec![0.0f32; FF];
                assert!(qdot::swiglu_clamp(&stage1[0], &stage1[1], 0.0, &mut h));
                for o in 0..EMBD {
                    let w = dequantized(&f.mats[e][2], FF, EMBD, o);
                    let cell = cell(&w, &h, scale(2, e));
                    judge(
                        format!("{order:?} stage 2 slot {j} out {o}"),
                        f64::from(got[j * EMBD + o]),
                        cell,
                    );
                }
            }
            println!("planes {order:?}: {outside} cells outside the band, max err/band {worst:.3}");
            assert_eq!(outside, 0, "{order:?}: every cell within the derived band");
        }
    }
}
