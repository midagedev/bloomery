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
    DEFER_MAX_COLS, GroupInput, QuantizedCols, ShardTensor, Tensor2, Weight, matmul_q,
    matmul_q_group, matmul_q_group_cols_into, matmul_q_group_into, matmul_q_group_swiglu,
    matmul_q_group_swiglu_into,
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

/// The most token columns a union scratch can be made for
/// ([`UnionScratch::new`]): one prefill ubatch. A call takes at most the
/// columns its scratch was made for ([`experts_union_into`],
/// [`HostLayer::experts_union_into`]).
pub const UNION_MAX_COLS: usize = 512;

/// Distinct experts per dispatch pair of a union call: their gate and up
/// fill the group bookkeeping's inline block and their downs fit the claim
/// table, so a chunk allocates nothing. A call over `u` distinct experts
/// issues `2 · ceil(u / UNION_CHUNK)` dispatches.
pub const UNION_CHUNK: usize = EXPERTS_INTO_MAX;

/// Columns up to which a union call sums on the caller and keeps its downs
/// there: the decode and verify shapes, whose sums are shorter than a pool
/// dispatch pays back. A wider call splits both by column across the pool.
const UNION_INLINE_COLS: usize = 8;

/// A union call's plan, in storage its scratch made at load: the distinct
/// experts in ascending id, and per expert the columns that list it,
/// ascending, each once. Expert `d`'s columns are `cols[off[d]..off[d + 1]]`,
/// and the same range names its down columns in the scratch's `store`.
struct UnionPlan {
    ids: Vec<u32>,
    off: Vec<usize>,
    cols: Vec<usize>,
    /// Every listed `(expert, column)`, sorted to build the rest.
    pairs: Vec<(u32, usize)>,
}

impl UnionPlan {
    /// Room for `cols` columns of [`EXPERTS_INTO_MAX`] experts each.
    fn with_room(cols: usize) -> UnionPlan {
        let slots = cols * EXPERTS_INTO_MAX;
        UnionPlan {
            ids: Vec::with_capacity(slots),
            off: Vec::with_capacity(slots + 1),
            cols: Vec::with_capacity(slots),
            pairs: Vec::with_capacity(slots),
        }
    }

    /// The plan of `lists` in place, or the named error of a list past
    /// [`EXPERTS_INTO_MAX`]. The caller has checked the column count
    /// against the room, so no push grows a buffer.
    fn build(&mut self, lists: &[&[(u32, f32)]]) -> Result<(), ModelError> {
        if let Some(list) = lists.iter().find(|l| l.len() > EXPERTS_INTO_MAX) {
            return Err(ModelError::Shape {
                what: "host union: at most EXPERTS_INTO_MAX experts per column",
                want_ne0: EXPERTS_INTO_MAX,
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

    /// The distinct index of an expert the lists name.
    fn slot(&self, e: u32) -> usize {
        self.ids
            .binary_search(&e)
            .expect("every listed expert is in the union")
    }

    /// The columns that list distinct expert `d`.
    fn cols(&self, d: usize) -> &[usize] {
        &self.cols[self.off[d]..self.off[d + 1]]
    }
}

/// The blocks a union call writes, made once for a layer shape and a column
/// count and held across calls, so a call allocates nothing: the gate, up,
/// combine and down blocks of one chunk of [`UNION_CHUNK`] experts, reused
/// chunk after chunk, each narrowed in place to the columns that list its
/// expert; the down columns of every chunk but the last, kept in `store`
/// until the list-order sums at the end; `x` quantized once for every chunk
/// of a wide call; and the plan's buffers.
///
/// For `cols` columns, `embd` and `ff` wide experts, it holds
/// `4 · (3 · ff + embd) · UNION_CHUNK · cols` bytes of chunk blocks,
/// `4 · embd · EXPERTS_INTO_MAX · cols` of store, a few words per slot and,
/// from the first call wider than [`DEFER_MAX_COLS`], `cols` quantized
/// columns of `embd` values (`qdot::col_bytes` each, about `embd` bytes).
pub struct UnionScratch {
    embd: usize,
    ff: usize,
    max_cols: usize,
    /// `[gate_0, up_0, gate_1, up_1, ..]` of the chunk, `{ff, m}` each.
    gate_up: [Tensor2; 2 * UNION_CHUNK],
    /// The chunk's combines, `{ff, m}` each.
    pars: [Tensor2; UNION_CHUNK],
    /// The chunk's down outputs, `{embd, m}` each.
    downs: [Tensor2; UNION_CHUNK],
    /// Down column `q` of the plan (`q` in `off[d]..off[d + 1]` for expert
    /// `d`) at `store[q·embd..(q + 1)·embd]`, for experts outside the last
    /// chunk.
    store: Vec<f32>,
    /// A wide call's `x`, quantized once in the gate's encoding.
    xq: QuantizedCols,
    plan: UnionPlan,
}

impl UnionScratch {
    /// Blocks for calls of up to `cols` columns over experts that map
    /// `embd -> ff -> embd`; `cols` outside `1..=UNION_MAX_COLS` is a named
    /// error.
    pub fn new(embd: usize, ff: usize, cols: usize) -> Result<UnionScratch, ModelError> {
        if cols == 0 || cols > UNION_MAX_COLS {
            return Err(ModelError::Shape {
                what: "host union scratch: 1..=UNION_MAX_COLS columns",
                want_ne0: embd,
                want_ne1: UNION_MAX_COLS,
                got_ne0: embd,
                got_ne1: cols,
            });
        }
        Ok(UnionScratch {
            embd,
            ff,
            max_cols: cols,
            gate_up: std::array::from_fn(|_| Tensor2::zeros(ff, cols)),
            pars: std::array::from_fn(|_| Tensor2::zeros(ff, cols)),
            downs: std::array::from_fn(|_| Tensor2::zeros(embd, cols)),
            store: vec![0.0; embd * cols * EXPERTS_INTO_MAX],
            xq: QuantizedCols::new(),
            plan: UnionPlan::with_room(cols),
        })
    }

    /// The most columns a call through this scratch takes.
    pub fn max_cols(&self) -> usize {
        self.max_cols
    }
}

/// The union call's shape conditions, each its own error: at most the
/// scratch's columns, one list per column of `x`, `out` holding every
/// column's width, and the scratch made for the block's widths.
fn check_union_call(
    x: &Tensor2,
    lists: &[&[(u32, f32)]],
    out: &[f32],
    ff: usize,
    scratch: &UnionScratch,
) -> Result<(), ModelError> {
    if x.ne1 > scratch.max_cols {
        return Err(ModelError::Shape {
            what: "host union: at most the columns the scratch was made for",
            want_ne0: x.ne0,
            want_ne1: scratch.max_cols,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }
    if x.ne1 != lists.len() {
        return Err(ModelError::Shape {
            what: "host union: x must have one column per list",
            want_ne0: x.ne0,
            want_ne1: lists.len(),
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }
    if out.len() != x.ne0 * x.ne1 {
        return Err(ModelError::Shape {
            what: "host union: out must hold every column's width",
            want_ne0: x.ne0,
            want_ne1: x.ne1,
            got_ne0: out.len(),
            got_ne1: 1,
        });
    }
    if (scratch.embd, scratch.ff) != (x.ne0, ff) {
        return Err(ModelError::Shape {
            what: "host union: the scratch must be made for the block's widths",
            want_ne0: x.ne0,
            want_ne1: ff,
            got_ne0: scratch.embd,
            got_ne1: scratch.ff,
        });
    }
    Ok(())
}

// Per-thread column buffer of a split union sum: a column is summed here in
// its list order, then written out.
thread_local! {
    static UNION_SUM_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The host leg of `k` tokens over the union of their experts: column `j`
/// of `out` (`out[j·embd..(j+1)·embd]`) is exactly what [`serve`] writes for
/// column `j` of `x` and `lists[j]`, bit for bit. The distinct experts, in
/// ascending id, run in chunks of [`UNION_CHUNK`]: one group dispatch for the
/// chunk's gates and ups, each pair over the columns that list its expert
/// ([`GroupInput::Cols`] of the one `x`: every column quantized alone, by the
/// weight type's rule, as the one-column call quantizes it), then one for
/// their downs over those columns' combines. Each weight row is read once
/// per call; its runs of up to [`qdot::TILE_COLS`] listed columns go to the
/// tile kernel, which writes each column's `dot_row` value. A call of at
/// most [`DEFER_MAX_COLS`] columns has its quantization claimed inside each
/// row dispatch; a wider one quantizes `x` once, over the pool, before the
/// first chunk, and every chunk's gate and up read those bytes
/// ([`GroupInput::Quantized`]; a pair of another encoding keeps
/// [`GroupInput::Cols`]) — the same bytes either way. A chunk's downs are copied to the
/// store before the next chunk reuses their blocks, and each column's sum
/// runs last, in that column's list order from zero — the order its
/// one-column call adds in. `resolve` gives an expert's `[gate, up, down]`;
/// `lists` has been checked against `x`, `out` and the scratch.
fn serve_union<'w>(
    resolve: impl Fn(u32) -> Result<[Weight<'w>; 3], ModelError>,
    limit: Option<f32>,
    x: &Tensor2,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    s: &mut UnionScratch,
) -> Result<(), ModelError> {
    let UnionScratch {
        max_cols,
        gate_up,
        pars,
        downs,
        store,
        xq,
        plan,
        ..
    } = s;
    plan.build(lists)?;
    let lvl = profile::level();
    let (mut x_ns, mut h_ns) = (0u64, 0u64);
    let embd = x.ne0;
    // Only a call wider than any claim takes: the decode and verify shapes
    // keep their claimed, per-dispatch quantization untouched.
    let mut filled = false;
    if x.ne1 > DEFER_MAX_COLS && plan.n() > 0 {
        let t_x = if lvl > 0 { Some(Instant::now()) } else { None };
        let gate = resolve(plan.ids[0])?[0];
        filled = xq.fill(gate.ty(), x, *max_cols);
        if let Some(t_x) = t_x {
            x_ns += t_x.elapsed().as_nanos() as u64;
        }
    }
    let xq: Option<&QuantizedCols> = filled.then_some(&*xq);
    let n_chunks = plan.n().div_ceil(UNION_CHUNK);
    let per = if n_chunks == 0 {
        0
    } else {
        plan.n().div_ceil(n_chunks)
    };
    for c in 0..n_chunks {
        let (a, b) = (c * per, ((c + 1) * per).min(plan.n()));
        let n = b - a;
        // The unused tails keep the chunk's first expert's gate; only the
        // first n (2n) are passed.
        let first = resolve(plan.ids[a])?[0];
        let mut gu = [first; 2 * UNION_CHUNK];
        let mut down = [first; UNION_CHUNK];
        let mut srcs: [GroupInput<'_>; 2 * UNION_CHUNK] =
            [GroupInput::Cols(x, &[]); 2 * UNION_CHUNK];
        for i in 0..n {
            let d = a + i;
            let [g, u, dw] = resolve(plan.ids[d])?;
            gu[2 * i] = g;
            gu[2 * i + 1] = u;
            down[i] = dw;
            let cols = plan.cols(d);
            let src = |w: &Weight<'_>| match xq {
                Some(q) if q.serves(w.ty(), w.k()) => GroupInput::Quantized(x, q, cols),
                _ => GroupInput::Cols(x, cols),
            };
            srcs[2 * i] = src(&g);
            srcs[2 * i + 1] = src(&u);
            gate_up[2 * i].set_cols(cols.len());
            gate_up[2 * i + 1].set_cols(cols.len());
            pars[i].set_cols(cols.len());
            downs[i].set_cols(cols.len());
        }
        let gt = matmul_q_group_cols_into(
            "host_gate_up",
            &gu[..2 * n],
            &srcs[..2 * n],
            &mut gate_up[..2 * n],
            &mut [],
        )?;
        let combines: [GroupInput<'_>; UNION_CHUNK] = std::array::from_fn(|i| {
            if i >= n {
                return GroupInput::Ready(x);
            }
            let (gate, up) = (&gate_up[2 * i], &gate_up[2 * i + 1]);
            match limit {
                Some(limit) => GroupInput::SwigluClamp(gate, up, limit),
                None => GroupInput::Swiglu(gate, up),
            }
        });
        let dt = matmul_q_group_cols_into(
            "host_down",
            &down[..n],
            &combines[..n],
            &mut downs[..n],
            &mut pars[..n],
        )?;
        x_ns += gt.stage_ns;
        h_ns += dt.stage_ns;
        if c + 1 < n_chunks {
            stash_downs(&downs[..n], &plan.off[a..=b], embd, store);
        }
    }
    let t_sum = if lvl > 0 { Some(Instant::now()) } else { None };
    let found = UnionDowns {
        blocks: &downs[..],
        store: &store[..],
        plan: &*plan,
        last: n_chunks.saturating_sub(1) * per,
        embd,
    };
    let k = lists.len();
    if k <= UNION_INLINE_COLS {
        for (j, o) in out.chunks_exact_mut(embd).enumerate() {
            found.sum_col(lists[j], j, o);
        }
    } else {
        // SAFETY (construction site): each participant writes only the cells
        // of the columns in its own chunk of `0..k`; the chunks partition the
        // columns, and the join publishes the writes.
        let out_ptr = crate::ops::SharedOut(out.as_mut_ptr());
        threads::pool().for_each_chunk(k, |cols| {
            UNION_SUM_BUF.with(|b| {
                let mut buf = b.borrow_mut();
                if buf.len() < embd {
                    buf.resize(embd, 0.0);
                }
                let o = &mut buf[..embd];
                for j in cols {
                    found.sum_col(lists[j], j, o);
                    for (i, &v) in o.iter().enumerate() {
                        // SAFETY: column `j` is in this participant's chunk — see the construction site.
                        unsafe { out_ptr.write(j * embd + i, v) };
                    }
                }
            });
        });
    }
    if let Some(t_sum) = t_sum {
        profile::record_time("host_x_quant", x_ns);
        profile::record_time("host_h_quant", h_ns);
        profile::record_time("host_sum", t_sum.elapsed().as_nanos() as u64);
    }
    Ok(())
}

/// Where a finished union call's down columns are: distinct experts from
/// `last` on (the last chunk) still in their blocks, the rest in the store.
struct UnionDowns<'a> {
    blocks: &'a [Tensor2],
    store: &'a [f32],
    plan: &'a UnionPlan,
    last: usize,
    embd: usize,
}

impl<'a> UnionDowns<'a> {
    /// Distinct expert `d`'s down column `t`.
    fn col(&self, d: usize, t: usize) -> &'a [f32] {
        if d >= self.last {
            self.blocks[d - self.last].col(t)
        } else {
            let q = self.plan.off[d] + t;
            &self.store[q * self.embd..(q + 1) * self.embd]
        }
    }

    /// Column `j`'s weighted sum into `o`, from zero in the order of its
    /// list `list` — the one-column call's order.
    fn sum_col(&self, list: &[(u32, f32)], j: usize, o: &mut [f32]) {
        o.fill(0.0);
        for &(e, w) in list {
            let d = self.plan.slot(e);
            let t = self
                .plan
                .cols(d)
                .binary_search(&j)
                .expect("a listing column is in its expert's columns");
            for (o, &dv) in o.iter_mut().zip(self.col(d, t)) {
                *o += w * dv;
            }
        }
    }
}

/// A chunk's down columns into the store before the next chunk reuses their
/// blocks: block `i`'s columns land at store columns `off[i]..off[i + 1]`
/// (`off` is the plan's offsets of the chunk's experts, one past its last).
/// Columns up to [`UNION_INLINE_COLS`] copy on the caller, more split across
/// the pool.
fn stash_downs(blocks: &[Tensor2], off: &[usize], embd: usize, store: &mut [f32]) {
    let (q0, q1) = (off[0], off[off.len() - 1]);
    if q1 - q0 <= UNION_INLINE_COLS {
        for (blk, w) in blocks.iter().zip(off.windows(2)) {
            store[w[0] * embd..w[1] * embd].copy_from_slice(&blk.data[..(w[1] - w[0]) * embd]);
        }
        return;
    }
    // SAFETY (construction site): each participant writes only store columns
    // in its own chunk of `q0..q1`; the chunks partition them, and the join
    // publishes the writes.
    let store_ptr = crate::ops::SharedOut(store.as_mut_ptr());
    threads::pool().for_each_chunk(q1 - q0, |qs| {
        for q in q0 + qs.start..q0 + qs.end {
            // The block whose range holds `q`: `off` ascends.
            let i = off.partition_point(|&o| o <= q) - 1;
            let col = blocks[i].col(q - off[i]);
            for (x, &v) in col.iter().enumerate() {
                // SAFETY: store column `q` is in this participant's chunk — see the construction site.
                unsafe { store_ptr.write(q * embd + x, v) };
            }
        }
    });
}

/// [`experts_into`] for many tokens at once — up to the columns `scratch` was
/// made for, at most [`UNION_MAX_COLS`]: column `j` of `x` with its routed
/// list `lists[j]` into `out[j·embd..(j+1)·embd]`, equal to
/// `experts_into(x_j, lists[j])` bit for bit, while each distinct expert's
/// matrices are read once for every column that lists it (see
/// [`serve_union`]). A list holds at most [`EXPERTS_INTO_MAX`] experts; that
/// bound, the column bound, a list count other than `x`'s columns, an `out`
/// of another length and an expert id past the block's are named errors. The
/// dispatches write into the caller's `scratch`, made at load for the block's
/// widths and a column count ([`UnionScratch::new`]), so a call allocates
/// nothing.
pub fn experts_union_into(
    gguf: &Gguf,
    plan: &MoeBlockPlan,
    x: &Tensor2,
    lists: &[&[(u32, f32)]],
    out: &mut [f32],
    scratch: &mut UnionScratch,
) -> Result<(), ModelError> {
    check_union_call(x, lists, out, plan.meta.ff, scratch)?;
    let view = |views: &[TensorInfo], e: u32| -> Result<Weight<'_>, ModelError> {
        Weight::in_file(gguf, expert_of(views, e, plan.block)?)
    };
    serve_union(
        |e| {
            Ok([
                view(&plan.gate_views, e)?,
                view(&plan.up_views, e)?,
                view(&plan.down_views, e)?,
            ])
        },
        None,
        x,
        lists,
        out,
        scratch,
    )
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
/// `load host_tier type=iq3_xxs k=4096 path=fused` — qdot's fused kernel — or
/// `path=dequant_row`, the slow path that decodes the row to f32 before the dot.
/// The decision is [`qdot::fuses`], the one the matmul dispatch makes, so no run
/// measures the slow path without saying so at load.
fn announce_paths(stacks: &[(GgmlType, usize)]) {
    static SEEN: Mutex<Vec<(GgmlType, usize)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    for &(ty, k) in stacks {
        if seen.contains(&(ty, k)) {
            continue;
        }
        seen.push((ty, k));
        let path = if qdot::fuses(ty, k) {
            "fused"
        } else {
            "dequant_row"
        };
        eprintln!("load host_tier type={ty} k={k} path={path}");
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
    /// evenly. Every shape error a call could meet is raised here, at load.
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
    /// shard that holds them.
    pub fn experts_union_into(
        &self,
        split: &Split,
        x: &Tensor2,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        scratch: &mut UnionScratch,
    ) -> Result<(), ModelError> {
        check_union_call(x, lists, out, self.gate.info().dims[1] as usize, scratch)?;
        serve_union(
            |e| {
                let e = e as usize;
                Ok([
                    self.gate.expert(split, e)?,
                    self.up.expert(split, e)?,
                    self.down.expert(split, e)?,
                ])
            },
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
        Buckets, EXPERTS_INTO_MAX, TOUCHED, check_host_call, gather_expert_inputs,
        last_touched_experts, reset_touched,
    };
    use crate::ops::Tensor2;

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
}
