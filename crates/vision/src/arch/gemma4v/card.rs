//! The resident vision weights and the bounded activation buffers.

use super::{Hparams, MERGE, tensors};
use crate::{
    VisionError,
    arch::{TokenLimits, card::CardBytes},
};
use gguf::Gguf;

/// Element counts of the tower's buffers at the largest supported canvas.
#[derive(Clone, Copy)]
pub struct ScratchLens {
    pub patches: usize,
    pub raw: usize,
    pub hidden: usize,
    pub qkv: usize,
    pub ff: usize,
    pub rows: usize,
    pub pooled: usize,
}

impl ScratchLens {
    /// Buffer extents for `tokens` pooled rows; refuse overflow before allocation.
    pub fn new(hp: &Hparams, tokens: usize) -> Result<Self, VisionError> {
        if tokens == 0 || hp.patch == 0 || hp.dim == 0 || hp.ff == 0 || hp.out_dim == 0 {
            return Err(VisionError::Limits {
                detail: "gemma4v scratch extents must be positive".into(),
            });
        }
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|&v| v <= u32::MAX as usize)
                .ok_or_else(|| VisionError::Limits {
                    detail: format!("gemma4v scratch {a} times {b} does not fit u32"),
                })
        };
        let patches = mul(tokens, MERGE * MERGE)?;
        let patch_values = mul(mul(hp.patch, hp.patch)?, 3)?;
        let qkv_width = mul(hp.dim, 3)?;
        Ok(Self {
            patches,
            raw: mul(patches, patch_values)?,
            hidden: mul(patches, hp.dim)?,
            qkv: mul(patches, qkv_width)?,
            ff: mul(patches, hp.ff)?,
            rows: mul(tokens, hp.out_dim)?,
            pooled: mul(tokens, hp.dim)?,
        })
    }

    /// Bytes of the buffers allocated by the tower.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        // F32 patches, patch output, residuals and post norms; BF16 linears, attention,
        // pooling gather, gate/up/activation, output rows; f32 position angles.
        4 * (self.raw as u64 + 4 * self.hidden as u64)
            + 2 * (4 * self.hidden as u64
                + self.qkv as u64
                + 5 * self.ff as u64
                + 3 * self.pooled as u64
                + self.rows as u64)
            + 4 * 64 * self.patches as u64
    }
}

/// The vision weights and scratch held on the side seat's card.
pub fn of(file: &Gguf, hp: &Hparams, limits: TokenLimits) -> Result<CardBytes, VisionError> {
    hp.size_rule(limits)?;
    let weights: u64 = tensors::check(file, hp)?
        .iter()
        .filter(|r| {
            ![".input_min", ".input_max", ".output_min", ".output_max"]
                .iter()
                .any(|s| r.name.ends_with(s))
        })
        .map(|r| r.bytes())
        .sum();
    let weights = weights
        + 4 * (hp.dim + 64) as u64
        + if file.find("v.std_bias").is_none() {
            8 * hp.dim as u64
        } else {
            0
        };
    // The shared BF16 GEMM holds one f32 bias and one bf16 residual stand-in.
    let scratch = ScratchLens::new(hp, limits.max())?.bytes() + 6;
    Ok(CardBytes { weights, scratch })
}
