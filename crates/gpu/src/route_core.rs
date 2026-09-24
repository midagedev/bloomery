//! The pieces the MoE router kernels share: the expert score functions and
//! the tie rules of the top-k selection.
//!
//! A router kernel keeps its own `#[kernel]` body — where the scores are
//! computed, the shared arrays they are parked in, which lanes hold which
//! candidates, the selection loop and the weight stage — and takes its score
//! and its tie rule from here. Both are scalar in, scalar out and
//! `#[inline(always)]`, and at that grain a kernel's instructions are the
//! same as with the code written in place. The selection loop, its butterfly
//! and the weight stage stay in the kernels: on this toolchain a function
//! boundary around any of them, even a scalar one, changes the kernel's
//! instruction stream (predicated selects become branches).
//!
//! [`Score`] names the element-wise score functions. Softmax is not one — it
//! needs all of a token's logits — and stays in its wrapper's own pass.
//! [`Sigmoid`] is compiled but no kernel routes with it yet.
//!
//! A weight stage that renormalizes its selected scores divides by
//! [`renorm_divisor`] of their sum: the one guard every such stage and its
//! host simulation share.

use crate::elem::argmax_take;

/// An element-wise expert score: `score(logit)` for one expert, independent
/// of every other expert's logit.
pub trait Score {
    /// The expert's score from its router logit.
    fn score(logit: f32) -> f32;
}

/// `sqrt(x > 20 ? x : ln(1 + e^x))`: ik's `SQRT_SOFTPLUS` gating in f32. One
/// function for both sides: on the device `exp`/`ln` are CUDA's libdevice,
/// on the host the system libm — ik's own, which is what a gate's host side
/// simulates.
#[inline(always)]
pub fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    sp.sqrt()
}

/// The divisor a weight stage renormalizes its selected scores by: their f32
/// sum plus `1e-20`, the guard exllamav3 (`routing.cu`), mistral.rs
/// (`deepseek2.rs`, `deepseek3.rs`) and ik's MoE graph for some other
/// architectures put on this division; ik divides V4's by the bare sum. A
/// [`sqrt_softplus`] score is exactly 0 below a logit of about −16.6, where
/// `1 + e^x` rounds to 1, so a token whose selected scores all sit there
/// divided 0 by 0. The addend is under half an ulp of every sum from 2^-42
/// up, and a nonzero [`sqrt_softplus`] score is at least
/// `sqrt(ln(1 + 2^-23))` ≈ 3.45e-4: every nonzero sum keeps its bits, and an
/// all-zero selection weighs 0.
#[inline(always)]
pub fn renorm_divisor(sum: f32) -> f32 {
    sum + 1e-20
}

/// `1 / (1 + e^-x)` in f32, the scalar form of ggml's `ggml_vec_sigmoid_f32`.
/// No kernel routes with it yet: the wrapper that does brings the gate that
/// pins it against its reference.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The [`sqrt_softplus`] score.
pub struct SqrtSoftplus;

impl Score for SqrtSoftplus {
    #[inline(always)]
    fn score(logit: f32) -> f32 {
        sqrt_softplus(logit)
    }
}

/// The [`sigmoid`] score.
pub struct Sigmoid;

impl Score for Sigmoid {
    #[inline(always)]
    fn score(logit: f32) -> f32 {
        sigmoid(logit)
    }
}

/// Whether candidate `(v, i)` beats the running best `(best_v, best_i)` in a
/// router's selection order: a greater value, or an equal value at the
/// larger index with `LARGER_ID`, at the smaller one without it (the greedy
/// sampler's rule, [`argmax_take`], which a serial scan with a strict `>`
/// over ascending ids realizes). Either is a total order on finite values,
/// so any fixed reduction over it picks the same winner; a NaN candidate
/// never wins (every comparison with it is false).
#[inline(always)]
pub fn take<const LARGER_ID: bool>(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
    if LARGER_ID {
        v > best_v || (v == best_v && i > best_i)
    } else {
        argmax_take(v, i, best_v, best_i)
    }
}
