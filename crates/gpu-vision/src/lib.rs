//! bloomery-gpu-vision — the image encoders on the card: from the bf16 patch tensor of
//! [`vision::arch`] to the bf16 rows the text model reads in place of an image's tokens. Two
//! towers, one per projector type of the encoder file (an mmproj GGUF):
//!
//! * `deepseek41v` — DeepSeek-V4.1-Flash's ViT (32 blocks, 2D RoPE, full bidirectional
//!   attention) and its aligner ([`encoder`]);
//! * `qwen3vl_merger` — the Qwen3-VL ViT (27 blocks of heads of 72, LayerNorm with bias, GELU-tanh
//!   MLP, learned 48×48 positions resized to the grid, 2D RoPE) and its 2×2 merger, shared by
//!   Clef-Flash, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next, whose files differ only in the merger's
//!   output width ([`qwen3vl`]).
//!
//! [`open`] reads the file's projector type and loads the tower; [`Tower`] is what a seat holds.
//!
//! The device code lives in its own crate for the reason `bloomery-gpu-deepseek41` does: a
//! compiler defect in one of these kernels breaks the builds that depend on this crate and no
//! other. Every kernel entry carries the prefix `vis_` — cuda-oxide derives a kernel's host
//! symbol from its entry name alone, so a name another crate also declares fails to link.
//!
//! Numeric contract: every op takes bf16 inputs, computes in f32 and rounds its output to bf16
//! (round to nearest even) at the op boundaries of the reference (for V4.1 `inference/vision.py`,
//! run by torch in bf16; for Qwen3-VL the nodes of llama.cpp's graph), listed in each tower's
//! module. Each kernel module states its own rule and carries the host transcription the gate
//! compares the kernel with.
//!
//! * [`gemm_bf16`] — the tensor-core GEMM `C = A·Bᵀ (+ bias)` with its epilogues (GELU,
//!   residual add), every linear layer of both towers.
//! * [`attn`] — non-causal multi-head attention over one image, online softmax: heads of 64
//!   (`vis_attn`) and heads of 72 (`vis_attn_h72`).
//! * [`rope2d`] — the 2D rotary embedding of the query and key heads and its angle tables:
//!   heads of 64 (`vis_rope2d`) and heads of 72 (`vis_rope2d_h72`).
//! * [`norm`] — RMSNorm over bf16 rows with f32 gains.
//! * [`layer_norm`] — LayerNorm with gain and bias.
//! * [`mlp`] — the SiLU gate between V4.1's MLP linears.
//! * [`gelu_tanh`] — the tanh GELU of the Qwen3-VL tower, in place.
//! * [`pos_bilinear`] — the Qwen3-VL tower's learned position rows resized to the image's grid.
//! * [`aligner`] — the 3×3 unfold of V4.1's ViT grid into the aligner's input rows.
//! * [`chain`] — what both chains share: weight reading, taps, the rows an encode returns.
//! * [`encoder`], [`qwen3vl`] — each tower's weights on the card and whole chain.
//!
//! Like `bloomery-gpu`, this crate is built with `cargo oxide` only.

pub mod aligner;
pub mod attn;
pub mod chain;
pub mod encoder;
pub mod gelu_tanh;
pub mod gemm_bf16;
pub mod layer_norm;
pub mod mlp;
pub mod norm;
pub mod pos_bilinear;
pub mod qwen3vl;
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

use bloomery_gpu::GpuError;
use cuda_core::{CudaContext, CudaStream};
use gguf::Gguf;
use std::sync::Arc;
use vision::Patches;
use vision::arch::qwen3vl::TokenLimits;
use vision::arch::{deepseek41v, qwen3vl as qwen3vl_host};

/// One image tower on the card, whichever projector type its file declares. A call takes
/// `&mut self`: one image at a time owns the tower's activations, which are allocated at load, and
/// the rows it returns borrow them until the next call.
pub trait Tower {
    /// Encode one image's patches into the rows of its tokens, without waiting for the card;
    /// read them on the stream the tower was opened with.
    fn encode(&mut self, patches: &Patches) -> Result<chain::Encoded<'_>, GpuError>;
    /// Width of a row: the text model's embedding width.
    fn out_dim(&self) -> usize;
    /// Bytes of weights on the card.
    fn weight_bytes(&self) -> u64;
    /// Bytes of activations on the card.
    fn scratch_bytes(&self) -> u64;
}

impl Tower for encoder::Encoder {
    fn encode(&mut self, patches: &Patches) -> Result<chain::Encoded<'_>, GpuError> {
        encoder::Encoder::encode(self, patches)
    }
    fn out_dim(&self) -> usize {
        self.hparams().out_dim
    }
    fn weight_bytes(&self) -> u64 {
        encoder::Encoder::weight_bytes(self)
    }
    fn scratch_bytes(&self) -> u64 {
        encoder::Encoder::scratch_bytes(self)
    }
}

/// Open the tower the file's `clip.projector_type` names and load it onto the card: V4.1's sizes
/// its activations from its own grid plan and ignores `limits`; the Qwen3-VL tower sizes them for
/// images of at most `limits`' token count. A projector type with no tower here, or none at all,
/// is refused by name. Load-time only.
pub fn open(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    file: &Gguf,
    limits: TokenLimits,
) -> Result<Box<dyn Tower>, GpuError> {
    let projector = file
        .value("clip.projector_type")
        .and_then(gguf::Value::as_str);
    match projector {
        Some(deepseek41v::PROJECTOR_TYPE) => {
            Ok(Box::new(encoder::Encoder::load(ctx, stream, file)?))
        }
        Some(qwen3vl_host::PROJECTOR_TYPE) => {
            Ok(Box::new(qwen3vl::Encoder::load(ctx, stream, file, limits)?))
        }
        other => Err(GpuError::Shape {
            what: "gpu_vision::open",
            detail: format!(
                "clip.projector_type is {other:?}; this build has a tower for \"{}\" and \"{}\"",
                deepseek41v::PROJECTOR_TYPE,
                qwen3vl_host::PROJECTOR_TYPE
            ),
        }),
    }
}
