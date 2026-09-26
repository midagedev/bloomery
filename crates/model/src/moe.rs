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
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use gguf::{GgmlType, Gguf, Split, TensorInfo};

use crate::ModelError;
use crate::ops::{
    self, DEFER_MAX_COLS, ExpertStack, GroupInput, ShardTensor, Tensor2, Tensor2View, UnionCall,
    UnionPlanView, UnionSlabs, UnionStack, Weight, matmul_q, matmul_q_group, matmul_q_group_into,
    matmul_q_group_swiglu, matmul_q_group_swiglu_into,
};
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

/// The most experts one host call takes ([`experts_into`],
/// [`HostLayer::experts_into`]). Its dispatch lists live in stack arrays of
/// this size and [`HostScratch`] holds this many experts' blocks, so a call
/// allocates nothing.
pub const EXPERTS_INTO_MAX: usize = 8;

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
/// width and `out` holds that width; an empty list writes zeros. The
/// dispatches write into the caller's `scratch`, made at load for the
/// block's widths ([`HostScratch::new`]), so a call allocates nothing.
pub fn experts_into(
    gguf: &Gguf,
    plan: &MoeBlockPlan,
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &mut [f32],
    scratch: &mut HostScratch,
) -> Result<(), ModelError> {
    check_host_call(x, experts, out)?;
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
    let n = experts.len();
    if n == 0 {
        return Ok(());
    }
    // The plan's views are headers of `gguf`, the one file this model has.
    let view = |views: &[TensorInfo], e: u32| -> Result<Weight<'_>, ModelError> {
        Weight::in_file(gguf, expert_of(views, e, plan.block)?)
    };
    // The unused tails keep the first expert's gate; only the first n (2n)
    // are passed.
    let first = view(&plan.gate_views, experts[0].0)?;
    let mut gu = [first; 2 * EXPERTS_INTO_MAX];
    let mut down = [first; EXPERTS_INTO_MAX];
    for (i, &(e, _)) in experts.iter().enumerate() {
        gu[2 * i] = view(&plan.gate_views, e)?;
        gu[2 * i + 1] = view(&plan.up_views, e)?;
        down[i] = view(&plan.down_views, e)?;
    }
    serve(&gu[..2 * n], &down[..n], None, x, experts, out, scratch)
}

/// Expert `e`'s view in one of a block plan's per-expert view lists; an id
/// past the list is the caller's error, named with the block.
fn expert_of(views: &[TensorInfo], e: u32, block: usize) -> Result<&TensorInfo, ModelError> {
    views
        .get(e as usize)
        .ok_or_else(|| ModelError::MissingTensor(format!("expert {e} of block {block}")))
}

/// A host call's three shape conditions, each its own error: `x` is one
/// token, `out` holds its width, and the list fits [`EXPERTS_INTO_MAX`].
fn check_host_call(x: &Tensor2, experts: &[(u32, f32)], out: &[f32]) -> Result<(), ModelError> {
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
    if experts.len() > EXPERTS_INTO_MAX {
        return Err(ModelError::Shape {
            what: "host experts: at most EXPERTS_INTO_MAX experts",
            want_ne0: EXPERTS_INTO_MAX,
            want_ne1: 1,
            got_ne0: experts.len(),
            got_ne1: 1,
        });
    }
    Ok(())
}

/// The blocks one host call writes — the gate and up outputs, the SwiGLU
/// combines and the down outputs of up to [`EXPERTS_INTO_MAX`] experts —
/// made once for a layer shape and held across calls, so a call allocates
/// nothing. After a call, [`HostScratch::gate`] and its siblings read what
/// it computed for the list's `i`-th expert.
pub struct HostScratch {
    embd: usize,
    ff: usize,
    /// `[gate_0, up_0, gate_1, up_1, ..]`, `{ff, 1}` each.
    gate_up: [Tensor2; 2 * EXPERTS_INTO_MAX],
    /// The combines, `{ff, 1}` each.
    pars: [Tensor2; EXPERTS_INTO_MAX],
    /// The down outputs, `{embd, 1}` each.
    downs: [Tensor2; EXPERTS_INTO_MAX],
}

impl HostScratch {
    /// Blocks for experts that map `embd -> ff -> embd`.
    pub fn new(embd: usize, ff: usize) -> HostScratch {
        HostScratch {
            embd,
            ff,
            gate_up: std::array::from_fn(|_| Tensor2::zeros(ff, 1)),
            pars: std::array::from_fn(|_| Tensor2::zeros(ff, 1)),
            downs: std::array::from_fn(|_| Tensor2::zeros(embd, 1)),
        }
    }

    fn fits(&self, embd: usize, ff: usize) -> bool {
        (self.embd, self.ff) == (embd, ff)
    }

    /// The gate output of the last call's `i`-th expert.
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

/// The host leg of one token over resolved weights: `gu` holds each listed
/// expert's gate and up interleaved, `down` its down. One group dispatch for
/// every gate and up (`x` quantized once), one for every down with each
/// combine — clamped when `limit` is given — produced inside it, then the
/// weighted sum in list order into `out` (already zeroed). Profile level 1
/// records the leg in five parts that do not overlap: `host_x_quant` (x's
/// quantization), `host_gate_up` (the gate/up rows, per weight type),
/// `host_h_quant` (the combines and their quantization), `host_down` (the
/// down rows, per weight type) and `host_sum`.
fn serve(
    gu: &[Weight<'_>],
    down: &[Weight<'_>],
    limit: Option<f32>,
    x: &Tensor2,
    experts: &[(u32, f32)],
    out: &mut [f32],
    s: &mut HostScratch,
) -> Result<(), ModelError> {
    let n = experts.len();
    let lvl = profile::level();
    let xs: [&Tensor2; 2 * EXPERTS_INTO_MAX] = [x; 2 * EXPERTS_INTO_MAX];
    let gt = matmul_q_group_into("host_gate_up", gu, &xs[..2 * n], &mut s.gate_up[..2 * n])?;
    let srcs: [GroupInput<'_>; EXPERTS_INTO_MAX] = std::array::from_fn(|i| {
        if i >= n {
            return GroupInput::Ready(x);
        }
        let (gate, up) = (&s.gate_up[2 * i], &s.gate_up[2 * i + 1]);
        match limit {
            Some(limit) => GroupInput::SwigluClamp(gate, up, limit),
            None => GroupInput::Swiglu(gate, up),
        }
    });
    let dt = matmul_q_group_swiglu_into(
        "host_down",
        down,
        &srcs[..n],
        &mut s.downs[..n],
        &mut s.pars[..n],
    )?;
    let t_sum = if lvl > 0 { Some(Instant::now()) } else { None };
    for (&(_, w), d) in experts.iter().zip(&s.downs[..n]) {
        for (o, &dv) in out.iter_mut().zip(d.col(0)) {
            *o += w * dv;
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
/// ([`UnionScratch::new`]): one prefill ubatch. A call takes at most the
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

/// The listed slots — (expert, column) pairs, a column once per expert — of
/// one batch at its worst, every column listing [`EXPERTS_INTO_MAX`] experts:
/// the least slot budget a group tail's scratch takes, so any batch's columns
/// fit it.
pub const UNION_BATCH_SLOTS: usize = UNION_MAX_COLS * EXPERTS_INTO_MAX;

// A narrow call's claim states hold every expert its columns can list.
const _: () = assert!(DEFER_MAX_COLS * EXPERTS_INTO_MAX <= ops::UNION_CLAIM_EXPERTS);

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

    /// The plan of `lists` in place, or the named error of a list past
    /// `per_list` experts — [`EXPERTS_INTO_MAX`], or the routed width a
    /// scratch was made for. The caller has checked the column count against
    /// the room, so no push grows a buffer.
    fn build(&mut self, lists: &[&[(u32, f32)]], per_list: usize) -> Result<(), ModelError> {
        if let Some(list) = lists.iter().find(|l| l.len() > per_list) {
            return Err(ModelError::Shape {
                what: "host union: at most EXPERTS_INTO_MAX experts per column, or the \
                       scratch's routed width",
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

/// The slabs a union call writes, made once for a layer shape and a column
/// count and held across calls, so a call allocates nothing: `x` quantized,
/// and per slot — a listed (expert, column) pair, expert `d`'s slots at the
/// plan's `off[d]..off[d + 1]` — its gate and up outputs, its combine
/// quantized in the down's encoding and its down output, kept until the
/// list-order sums at the end; and the plan's buffers.
///
/// For `cols` columns, `slots` slots and `embd` and `ff` wide experts, it
/// holds `4 · (2 · ff + embd) · slots` bytes of gate/up and down slabs,
/// `slots` combine columns and twice `cols` columns of `x` (the gate's
/// encoding and the up's) of `ops::col_bytes_max` bytes each (a little over
/// `ff` and `embd` bytes), and a few words per slot of plan.
///
/// A batch's scratch ([`UnionScratch::new`], [`UnionScratch::new_routed`])
/// takes up to [`UNION_MAX_COLS`] columns and holds their every slot: each
/// column's list is at most [`EXPERTS_INTO_MAX`] experts, or the routed
/// width it was made for. A group tail's ([`UnionScratch::new_tail`]) takes
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
    plan: UnionPlan,
}

impl UnionScratch {
    /// Slabs for a batch's calls, up to `cols` columns over experts that map
    /// `embd -> ff -> embd`, each column listing at most
    /// [`EXPERTS_INTO_MAX`] experts; `cols` outside `1..=UNION_MAX_COLS` is a
    /// named error.
    pub fn new(embd: usize, ff: usize, cols: usize) -> Result<UnionScratch, ModelError> {
        UnionScratch::new_routed(embd, ff, cols, EXPERTS_INTO_MAX)
    }

    /// [`UnionScratch::new`] for lists of at most `n_used` experts — the
    /// model's routed width — so it holds `n_used · cols` slots, not
    /// `EXPERTS_INTO_MAX · cols`; a call with a longer list is refused by
    /// name. `n_used` outside `1..=EXPERTS_INTO_MAX` is a named error.
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
        if n_used == 0 || n_used > EXPERTS_INTO_MAX {
            return Err(ModelError::Shape {
                what: "host union scratch: 1..=EXPERTS_INTO_MAX experts per column",
                want_ne0: EXPERTS_INTO_MAX,
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
    /// [`UNION_MAX_COLS`] in one call, so each expert it lists is read once
    /// for all of them, over `slots` slots — the budget, from
    /// [`UNION_BATCH_SLOTS`] (so a batch's columns always fit) to every slot
    /// of the columns. `groups` outside `1..=UNION_TAIL_MAX_GROUPS` or a
    /// budget outside that range is a named error. A call through it may take
    /// fewer columns (a short last batch), and batch calls fit it too.
    pub fn new_tail(
        embd: usize,
        ff: usize,
        groups: usize,
        slots: usize,
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
        let cols = groups * UNION_MAX_COLS;
        if !(UNION_BATCH_SLOTS..=cols * EXPERTS_INTO_MAX).contains(&slots) {
            return Err(ModelError::Shape {
                what: "host union tail scratch: UNION_BATCH_SLOTS..=EXPERTS_INTO_MAX · columns slots",
                want_ne0: UNION_BATCH_SLOTS,
                want_ne1: cols * EXPERTS_INTO_MAX,
                got_ne0: slots,
                got_ne1: groups,
            });
        }
        Ok(UnionScratch::with_caps(
            embd,
            ff,
            cols,
            slots,
            EXPERTS_INTO_MAX,
        ))
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
            plan: UnionPlan::with_room(cols, per_list),
        }
    }

    /// The most columns a call through this scratch takes.
    pub fn max_cols(&self) -> usize {
        self.max_cols
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
/// column's width, and the scratch made for the block's widths.
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
    if (scratch.embd, scratch.ff) != (embd, ff) {
        return Err(ModelError::Shape {
            what: "host union: the scratch must be made for the block's widths",
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
/// listed columns through the tile kernel, which writes each column's
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
/// checked against `x`, `out` and the scratch.
fn serve_union(
    stacks: [ExpertStack<'_>; 3],
    limit: Option<f32>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
) -> Result<(), ModelError> {
    s.plan.build(lists, s.per_list)?;
    if s.plan.fits(s.max_slots) {
        return serve_planned(stacks, limit, x, lists, out, s);
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
            "host union: a batch's columns fit every scratch (UNION_BATCH_SLOTS slots or more)"
        );
        serve_planned(
            stacks,
            limit,
            part,
            &lists[c0..c1],
            &mut out[c0 * embd..c1 * embd],
            s,
        )?;
        c0 = c1;
    }
    Ok(())
}

/// [`serve_union`] for a plan already built into `s.plan` from `lists`,
/// which fits its slabs: the call ([`UnionCall::new`]) run through the
/// scratch.
fn serve_planned(
    stacks: [ExpertStack<'_>; 3],
    limit: Option<f32>,
    x: Tensor2View<'_>,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
) -> Result<(), ModelError> {
    if s.plan.n() == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let call = UnionCall::new(s.plan.view(), stacks, x, limit, s.ff)?;
    let slabs = UnionSlabs {
        xq: &mut s.xq,
        xq_up: &mut s.xq_up,
        gu: &mut s.gu,
        qc: &mut s.qc,
        store: &mut s.store,
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
/// ([`Tensor2View`]), read in place. A list holds at most
/// [`EXPERTS_INTO_MAX`] experts; that bound, the column bound, a list count
/// other than `x`'s columns, an `out` of another length and an expert id past
/// the block's are named errors. The dispatches write into the caller's
/// `scratch`, made at load for the block's widths and a column count
/// ([`UnionScratch::new`]), so a call allocates nothing.
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
    serve_union(stacks, None, x, lists, out, scratch)
}

/// What a host tier needs to serve one layer's routed experts, in the
/// file's own terms: the three stacks' tensor names, the expert count, the
/// widths and the routed experts' SwiGLU limit. The architecture fills it;
/// this module reads no model's names or keys.
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

/// The host tier's matmul path for each (weight type, row width) it serves,
/// printed once per process when a layer first brings it in:
/// `load host_tier type=iq3_xxs k=4096 path=fused` — qdot's fused kernel, the
/// only path a host tier runs: [`HostLayer::build`] refuses a stack without
/// one before it prints.
fn announce_paths(stacks: &[(GgmlType, usize)]) {
    static SEEN: Mutex<Vec<(GgmlType, usize)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    for &(ty, k) in stacks {
        if seen.contains(&(ty, k)) {
            continue;
        }
        seen.push((ty, k));
        eprintln!("load host_tier type={ty} k={k} path=fused");
    }
}

/// One layer's routed experts as a host tier serves them: the three stacks,
/// each with the shard that holds it, and the SwiGLU limit. Built once at
/// load ([`HostLayer::build`]); a call cuts its experts' matrices from the
/// stacks, so a layer holds three headers, not one view per expert.
pub struct HostLayer {
    gate: ShardTensor,
    up: ShardTensor,
    down: ShardTensor,
    n_expert: usize,
    limit: f32,
}

impl HostLayer {
    /// Find the three stacks and check them against the spec: gate and up
    /// map `embd -> ff`, down maps `ff -> embd`, all `n_expert` deep and cut
    /// evenly, each with a fused kernel at its row width — the union call's
    /// own check ([`UnionStack::of_call`]), whose error names the stack. Every
    /// stack error a call could meet is raised here, at load.
    pub fn build(split: &Split, spec: &HostLayerSpec<'_>) -> Result<HostLayer, ModelError> {
        let gate = ShardTensor::find(split, spec.gate)?;
        let up = ShardTensor::find(split, spec.up)?;
        let down = ShardTensor::find(split, spec.down)?;
        for (t, k, n) in [
            (&gate, spec.embd, spec.ff),
            (&up, spec.embd, spec.ff),
            (&down, spec.ff, spec.embd),
        ] {
            expect_stack(t.info(), k, n, spec.n_expert)?;
            // The per-expert cut: `expert_view`'s check, once, here.
            expert_view(t.info(), 0, spec.n_expert)?;
        }
        UnionStack::of_call(
            &[gate.stack(split)?, up.stack(split)?, down.stack(split)?],
            spec.embd,
            spec.ff,
        )?;
        announce_paths(&[
            (gate.info().ty, spec.embd),
            (up.info().ty, spec.embd),
            (down.info().ty, spec.ff),
        ]);
        Ok(HostLayer {
            gate,
            up,
            down,
            n_expert: spec.n_expert,
            limit: spec.swiglu_limit,
        })
    }

    /// The three stacks, `[gate, up, down]`.
    pub fn stacks(&self) -> [&ShardTensor; 3] {
        [&self.gate, &self.up, &self.down]
    }

    /// The three stacks as a union call reads them, resolved once.
    fn stacks_of<'a>(&'a self, split: &'a Split) -> Result<[ExpertStack<'a>; 3], ModelError> {
        Ok([
            self.gate.stack(split)?,
            self.up.stack(split)?,
            self.down.stack(split)?,
        ])
    }

    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    pub fn swiglu_limit(&self) -> f32 {
        self.limit
    }

    /// [`experts_into`] for this layer: a given list of routed experts of
    /// ONE token, `out = Σ_i w_i · down_{e_i}(clamp(up_{e_i}·x, ±L) ⊙
    /// min(silu(gate_{e_i}·x), L))` in list order, each matrix read from
    /// the shard that holds it. `scratch` is the caller's, made for this
    /// layer's widths; after the call it holds what the call computed.
    pub fn experts_into(
        &self,
        split: &Split,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        scratch: &mut HostScratch,
    ) -> Result<(), ModelError> {
        check_host_call(x, experts, out)?;
        out.fill(0.0);
        let n = experts.len();
        if n == 0 {
            return Ok(());
        }
        // The unused tails keep the first expert's gate; only the first n
        // (2n) are passed.
        let first = self.gate.expert(split, experts[0].0 as usize)?;
        let mut gu = [first; 2 * EXPERTS_INTO_MAX];
        let mut down = [first; EXPERTS_INTO_MAX];
        for (i, &(e, _)) in experts.iter().enumerate() {
            let e = e as usize;
            gu[2 * i] = self.gate.expert(split, e)?;
            gu[2 * i + 1] = self.up.expert(split, e)?;
            down[i] = self.down.expert(split, e)?;
        }
        serve(
            &gu[..2 * n],
            &down[..n],
            Some(self.limit),
            x,
            experts,
            out,
            scratch,
        )
    }

    /// [`experts_union_into`] for this layer: up to the columns `scratch`
    /// was made for, column `j` of `out` equal to
    /// [`HostLayer::experts_into`] of column `j` of `x` and `lists[j]` bit
    /// for bit, each distinct expert's matrices read once per call from the
    /// shard that holds them. `x` is read in place, a block or a view.
    pub fn experts_union_into<'x>(
        &self,
        split: &Split,
        x: impl Into<Tensor2View<'x>>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        let x = x.into();
        check_union_call(x, lists, out, self.gate.info().dims[1] as usize, scratch)?;
        serve_union(
            self.stacks_of(split)?,
            Some(self.limit),
            x,
            lists,
            out,
            scratch,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Buckets, EXPERTS_INTO_MAX, HostLayer, HostLayerSpec, HostScratch, TOUCHED, UnionScratch,
        check_host_call, gather_expert_inputs, last_touched_experts, reset_touched,
    };
    use crate::ops::{self, Tensor2};
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
            match check_host_call(x, &experts, &vec![0.0; out]) {
                Ok(()) => panic!("a bad host call must be refused"),
                Err(e) => e.to_string(),
            }
        };
        let (x1, x2) = (Tensor2::zeros(8, 1), Tensor2::zeros(8, 2));
        let e = refusal(&x2, 1, 8);
        assert!(e.contains("one token") && e.contains("got [8, 2]"), "{e}");
        let e = refusal(&x1, 1, 7);
        assert!(e.contains("width") && e.contains("got [7, 1]"), "{e}");
        let e = refusal(&x1, EXPERTS_INTO_MAX + 1, 8);
        assert!(
            e.contains("EXPERTS_INTO_MAX")
                && e.contains(&format!("got [{}, 1]", EXPERTS_INTO_MAX + 1)),
            "{e}"
        );
        assert!(check_host_call(&x1, &[(0, 1.0); EXPERTS_INTO_MAX], &[0.0; 8]).is_ok());
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
                // block_q4_K: d f16 at 0, dmin f16 at 2.
                GgmlType::Q4_K => {
                    blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                    blk[2..4].copy_from_slice(&0x1C00u16.to_le_bytes());
                }
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
            split,
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
        let mut us = UnionScratch::new(embd, ff, 16).unwrap();
        let mut host = HostScratch::new(embd, ff);
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
                    .experts_into(&split, &xj, list, &mut want[j * embd..][..embd], &mut host)
                    .unwrap();
            }
            let mut got = vec![f32::NAN; embd * cols];
            let p0 = us.passes();
            layer
                .experts_union_into(&split, &x, &slices, &mut got, &mut us)
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

    /// A scratch made for a routed width holds that many slots a column and
    /// refuses a longer list by name; a width outside `1..=EXPERTS_INTO_MAX`
    /// is refused when it is made.
    #[test]
    fn routed_scratch_refuses_a_list_past_its_width() {
        let (embd, ff, n_expert) = (256, 256, 6);
        let tys = [GgmlType::Q4_K; 3];
        for n_used in [0, EXPERTS_INTO_MAX + 1] {
            let e = UnionScratch::new_routed(embd, ff, 4, n_used)
                .err()
                .expect("a routed width outside 1..=EXPERTS_INTO_MAX is refused")
                .to_string();
            assert!(e.contains("1..=EXPERTS_INTO_MAX experts per column"), "{e}");
        }
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
            .experts_union_into(&split, &x, &[&three[..], &three[..]], &mut out, &mut us)
            .unwrap();
        let e = layer
            .experts_union_into(&split, &x, &[&three[..], &four[..]], &mut out, &mut us)
            .expect_err("a list past the routed width is refused")
            .to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(
            e.contains("or the scratch's routed width")
                && e.contains("expected [3, 1], got [4, 1]"),
            "{e}"
        );
        println!("routed scratch: refusal {e}");
    }
}
