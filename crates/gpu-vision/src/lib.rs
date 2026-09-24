//! bloomery-gpu-vision — the image encoder on the card: DeepSeek-V4.1-Flash's ViT (32 blocks, 2D
//! RoPE, full bidirectional attention) and its aligner, from the bf16 patch tensor of
//! [`vision::preprocess`] to the bf16 rows the text model reads in place of an image's tokens.
//!
//! The device code lives in its own crate for the reason `bloomery-gpu-deepseek41` does: a
//! compiler defect in one of these kernels breaks the builds that depend on this crate and no
//! other. Every kernel entry carries the prefix `vis_` — cuda-oxide derives a kernel's host
//! symbol from its entry name alone, so a name another crate also declares fails to link.
//!
//! Numeric contract: every op takes bf16 inputs, computes in f32 and rounds its output to bf16
//! (round to nearest even) at the op boundaries of the reference (`inference/vision.py`, run by
//! torch in bf16), listed in [`encoder`]. Each module states its own rule and carries the host
//! transcription the gate compares the kernel with.
//!
//! * [`gemm_bf16`] — the tensor-core GEMM `C = A·Bᵀ (+ bias)` with its epilogues (GELU,
//!   residual add), every linear layer of the encoder.
//! * [`attn`] — non-causal multi-head attention over one image, online softmax.
//! * [`rope2d`] — the 2D rotary embedding of the query and key heads, and its angle table.
//! * [`norm`] — RMSNorm over bf16 rows with f32 gains.
//! * [`mlp`] — the SiLU gate between the MLP's two linears.
//! * [`aligner`] — the 3×3 unfold of the ViT grid into the aligner's input rows.
//! * [`encoder`] — the weights on the card and the whole chain.
//!
//! Like `bloomery-gpu`, this crate is built with `cargo oxide` only.

pub mod aligner;
pub mod attn;
pub mod encoder;
pub mod gemm_bf16;
pub mod mlp;
pub mod norm;
pub mod rope2d;

/// bf16 bits to the f32 they denote (exact).
#[inline(always)]
#[must_use]
pub fn bf16_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 to bf16 bits, round to nearest, ties to even; a NaN stays a NaN. The device's rounding
/// (`cuda_device::convert::f32_to_bf16_rne`) — one function, so the host rules round as the
/// kernels do.
#[inline(always)]
#[must_use]
pub fn f32_bf16(x: f32) -> u16 {
    cuda_device::convert::f32_to_bf16_rne(x)
}
