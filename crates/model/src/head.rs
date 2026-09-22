//! Output head: `l_out-26` in, `result_norm` then `result_output` out.
//!
//! Gate: `crates/model/tests/head.rs` against the oracle. Owned by the head round.
//!
//! Both stages are position-wise, so a batch in is a batch out: `head` returns one
//! logits column per input column and never special-cases the last position — picking
//! the position is the caller's job. That is the batch-first decision in `lib.rs`
//! applied to the last op pair in the graph.
use crate::ops::{f32_tensor, matmul_q, rms_norm};
use crate::profile;
use crate::{ModelError, Tensor2};
use gguf::{Gguf, TensorInfo};
use std::time::Instant;

/// The head's load-time state: the architecture-wide eps, the decoded
/// `output_norm.weight` gain and the `output.weight` view — everything a step
/// re-read from the file before this existed. Built once per file by the
/// architecture's load-time plan and per call by [`head`].
pub struct HeadPlan {
    pub(crate) eps: f32,
    pub(crate) gain: Vec<f32>,
    pub(crate) out_w: TensorInfo,
}

impl HeadPlan {
    /// `eps` is the caller's: which metadata key the final norm runs with is the
    /// architecture's to say.
    pub fn new(gguf: &Gguf, eps: f32) -> Result<HeadPlan, ModelError> {
        let norm_t = gguf
            .find("output_norm.weight")
            .ok_or_else(|| ModelError::MissingTensor("output_norm.weight".into()))?;
        let gain = f32_tensor(gguf, norm_t)?;
        let out_w = gguf
            .find("output.weight")
            .ok_or_else(|| ModelError::MissingTensor("output.weight".into()))?
            .clone();
        Ok(HeadPlan { eps, gain, out_w })
    }
}

/// Returns the logits for every token position; the oracle only holds the last one.
///
/// `x` is the final block's output `[embd, n_tokens]`; the result is
/// `[vocab, n_tokens]`, where the vocab axis is `output.weight`'s own width — the
/// gate checks it against `deepseek2.vocab_size` from the file, never a literal.
///
/// Resolves the head plan per call — the direct-call path; a decode step hands
/// the plan its architecture built at load to [`head_with`]. `eps` is the final
/// norm's, as [`HeadPlan::new`] takes it.
pub fn head(gguf: &Gguf, eps: f32, x: &Tensor2) -> Result<Tensor2, ModelError> {
    let plan = HeadPlan::new(gguf, eps)?;
    head_with(gguf, &plan, x)
}

/// The head over a load-time plan — the step path.
pub fn head_with(gguf: &Gguf, plan: &HeadPlan, x: &Tensor2) -> Result<Tensor2, ModelError> {
    // Profiler hook (crate::profile), coverage round: `head_params` covers the
    // plan resolution that the eps lookup and the two tensor finds used to be.
    // The norm itself records under `rms_norm`, the projection under `matmul_q`.
    let lvl = profile::level();
    let mut params_ns = 0u64;
    let t_p1 = if lvl > 0 { Some(Instant::now()) } else { None };
    let eps = plan.eps;
    let gain = &plan.gain;
    if let Some(t_p1) = t_p1 {
        params_ns += t_p1.elapsed().as_nanos() as u64;
    }

    if x.ne0 != gain.len() {
        return Err(ModelError::Shape {
            what: "head input",
            want_ne0: gain.len(),
            want_ne1: x.ne1,
            got_ne0: x.ne0,
            got_ne1: x.ne1,
        });
    }

    let normed = rms_norm(x, gain, eps);

    let t_p2 = if lvl > 0 { Some(Instant::now()) } else { None };
    let out_t = &plan.out_w;
    if let Some(t_p2) = t_p2 {
        params_ns += t_p2.elapsed().as_nanos() as u64;
    }
    if lvl > 0 {
        profile::record_time("head_params", params_ns);
    }
    matmul_q(gguf, out_t, &normed)
}
