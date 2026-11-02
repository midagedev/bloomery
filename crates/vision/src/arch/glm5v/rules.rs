//! The two facts of the `glm5v` tower that no shared kernel carries, as host rules the card's
//! kernels and their gate are held to: the K order of the merger's 2×2 convolution and the clamped
//! SwiGLU. Both are llama.cpp's graph: `glm4v.cpp`'s `ggml_conv_2d` over a merge group, and
//! `ggml_swiglu_clamp` (`ggml-cuda/unary.cuh`, `ggml_cuda_op_swiglu_clamp_single`).
//!
//! The convolution is a GEMM over `im2col` rows, the K order every 2×2 merger of that form shares
//! ([`crate::arch::merger`], re-exported here).

pub use crate::arch::merger::{conv_k_in_file, conv_k_on_card, reorder_merger_weight};

/// `ggml_swiglu_clamp(gate, up, limit)` on one f32: the gate is clamped from above only and the up
/// value from both sides, then `silu(gate) · up`. A NaN stays a NaN (llama.cpp's `fminf` and
/// `fmaxf` return the number and so hide it).
#[must_use]
pub fn swiglu_clamp(gate: f32, up: f32, limit: f32) -> f32 {
    let g = if gate > limit { limit } else { gate };
    let u = if up > limit {
        limit
    } else if up < -limit {
        -limit
    } else {
        up
    };
    g / (1.0 + (-g).exp()) * u
}

#[cfg(test)]
mod tests {
    use super::swiglu_clamp;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// The gate is clamped from above only, the up value from both sides, at the same limit.
    #[test]
    fn the_gate_clamps_one_side_and_the_up_value_both() {
        let l = 10.0;
        assert_eq!(swiglu_clamp(25.0, 3.0, l), silu(10.0) * 3.0);
        // below -limit the gate is not clamped: silu(-20), not silu(-10)
        assert_eq!(swiglu_clamp(-20.0, 3.0, l), silu(-20.0) * 3.0);
        assert_ne!(silu(-20.0), silu(-10.0));
        assert_eq!(swiglu_clamp(1.5, 25.0, l), silu(1.5) * 10.0);
        assert_eq!(swiglu_clamp(1.5, -25.0, l), silu(1.5) * -10.0);
        assert_eq!(swiglu_clamp(1.5, 2.5, l), silu(1.5) * 2.5);
    }

    /// A NaN input is a NaN output, not a clamped number.
    #[test]
    fn a_nan_stays_a_nan() {
        assert!(swiglu_clamp(f32::NAN, 1.0, 10.0).is_nan());
        assert!(swiglu_clamp(1.0, f32::NAN, 10.0).is_nan());
    }
}
