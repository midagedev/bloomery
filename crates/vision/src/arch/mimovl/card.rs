//! The tower's bytes on its card, from the hyperparameters and the token limits alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's vision tensors as the card holds them ([`weights`]). Every matrix is
//!   bf16 and the two f32 patch kernels are narrowed to bf16 on upload (`arch::narrow`), so the
//!   two kernels are one `[dim × 2·3·patch²]` matrix of the same bytes. Every output width of a
//!   linear layer is a multiple of the GEMM tile ([`super::hparams::TILE_N`]), so nothing is
//!   padded. Vectors and the sinks stay f32. The audio tower's tensors in the same file are not
//!   the vision tower's.
//! * the activations: the buffers of the largest image the limits allow ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use super::tensors::{OnCard, expected, heads_width, qkv_rows};
use super::{HEAD_DIM, Hparams, TEMPORAL_FRAMES};
pub use crate::arch::card::CardBytes;
use crate::arch::card::activation_bytes;
use crate::size::TokenLimits;

/// The tower of a file of `hp` at `limits` image tokens (module doc).
#[must_use]
pub fn of(hp: &Hparams, limits: TokenLimits) -> CardBytes {
    CardBytes {
        weights: weights(hp),
        scratch: ScratchLens::of(hp, limits).bytes(),
    }
}

/// Bytes of weights the tower uploads (module doc): the table's bytes, the narrowed tensors at
/// two bytes a value.
#[must_use]
pub fn weights(hp: &Hparams) -> u64 {
    expected(hp)
        .iter()
        .map(|e| match e.home {
            OnCard::File => e.bytes(),
            OnCard::Bf16 => e.dims.iter().product::<u64>() * 2,
        })
        .sum()
}

/// The values each activation buffer of the tower holds, for the largest image the limits allow:
/// `max_tokens` merged tokens and `merge²` times as many patches. `cs` holds f32 values, every
/// other buffer bf16. An image of fewer patches uses the first rows of each buffer.
///
/// A column-window block reads the patches in another order than a row-window block, so the
/// rotation table holds both orders (`cs`). The MLP's gate and up are one GEMM of `2·ff` outputs
/// (`gu`) and the SwiGLU's output a buffer beside it (`act`); the attention's output is as wide as
/// the heads together (`att`), not as the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The ViT rows the buffers are sized for.
    pub max_patches: usize,
    /// The merged tokens they are sized for.
    pub max_tokens: usize,
    /// The image's patches, `2·3·patch²` values each (the temporal frames repeat the image).
    pub patches: usize,
    /// The 2D RoPE tables of the two patch orders: a cosine and a sine per pair of a head
    /// (`head / 2` pairs), per patch.
    pub cs: usize,
    /// The block input and output, and the norm's output.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's fused gate and up rows, and the SwiGLU's output.
    pub gu: usize,
    pub act: usize,
    /// The merger's hidden rows (`merge²` patch rows wide) and its output rows, per token.
    pub hh: usize,
    pub rows: usize,
}

impl ScratchLens {
    /// The buffers of the largest plan `limits` allow.
    #[must_use]
    pub fn of(hp: &Hparams, limits: TokenLimits) -> ScratchLens {
        let tokens = limits.max();
        let n = tokens * hp.merge * hp.merge;
        let merged = hp.dim * hp.merge * hp.merge;
        ScratchLens {
            max_patches: n,
            max_tokens: tokens,
            patches: n * TEMPORAL_FRAMES * 3 * hp.patch * hp.patch,
            cs: 2 * n * HEAD_DIM,
            xa: n * hp.dim,
            xb: n * hp.dim,
            h: n * hp.dim,
            qkv: n * qkv_rows(hp) as usize,
            att: n * heads_width(hp) as usize,
            gu: n * 2 * hp.ff,
            act: n * hp.ff,
            hh: tokens * merged,
            rows: tokens * hp.out_dim,
        }
    }

    /// Bytes on the card: four per `cs` value, two per value of every other buffer.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        activation_bytes(
            self.cs,
            [
                self.patches,
                self.xa,
                self.xb,
                self.h,
                self.qkv,
                self.att,
                self.gu,
                self.act,
                self.hh,
                self.rows,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CardBytes, ScratchLens, of, weights};
    use crate::arch::mimovl::hparams::tests::hp;
    use crate::size::TokenLimits;

    fn at(tokens: usize) -> TokenLimits {
        TokenLimits::new(8, tokens).expect("limits")
    }

    /// Weights, derived. A block holds the qkv matrix (1280·3072·2 = 7,864,320) with its bias
    /// (12,288), the output matrix (2048·1280·2 = 5,242,880) with its bias (5,120), gate, up and
    /// down at 4608 (1280·4608·2 = 11,796,480 each) with biases (18,432, 18,432, 5,120) and two
    /// f32 norm gains (5,120 each): 48,566,272 B, and a windowed block's 32 f32 sinks add 128.
    /// 28 blocks and 24 sinks, the two patch kernels as one `[1280 × 1536]` bf16 matrix
    /// (3,932,160), the final gain (5,120), `mm.0` (5120²·2 = 52,428,800) and `mm.2`
    /// (5120·4096·2 = 41,943,040): 1,458,167,808 B.
    #[test]
    fn the_weights_of_the_file() {
        let block = 7_864_320
            + 12_288
            + 5_242_880
            + 5_120
            + 3 * 11_796_480
            + 2 * 18_432
            + 5_120
            + 2 * 5_120;
        assert_eq!(block, 48_566_272);
        let sum = 3_932_160 + 28 * block + 24 * 128 + 5_120 + 52_428_800 + 41_943_040;
        assert_eq!(sum, 1_458_167_808);
        assert_eq!(weights(&hp()), sum);
    }

    /// Activations, derived. A patch holds its 1536 input values (3,072 B), a cosine and sine per
    /// pair of its 64-wide head in each of the two orders (512 B), three rows of 1280 (7,680 B),
    /// the qkv row (3072 values, 6,144 B), the attention row (2048, 4,096 B), the fused gate and up
    /// row (9216, 18,432 B) and the SwiGLU row (4608, 9,216 B): 49,152 B, and a token's 4 patches
    /// 196,608 B. A token adds the merger's 5120-wide hidden row (10,240 B) and its output row
    /// (8,192 B). Per token of the limit 215,040 B.
    #[test]
    fn the_scratch_at_two_limits() {
        let per_patch: u64 = 3072 + 512 + 3 * 2560 + 6144 + 4096 + 18_432 + 9216;
        assert_eq!(per_patch, 49_152);
        let per_token = 4 * per_patch + 10_240 + 8192;
        assert_eq!(per_token, 215_040);
        for tokens in [1024, 8192] {
            assert_eq!(
                ScratchLens::of(&hp(), at(tokens)).bytes(),
                per_token * tokens as u64,
                "{tokens} tokens"
            );
        }
        assert_eq!(ScratchLens::of(&hp(), at(8192)).max_patches, 32_768);
    }

    /// The figure a plan reserves at the file's own budget (8,192 tokens): weights plus scratch.
    #[test]
    fn the_card_figure_at_the_files_budget() {
        let limits = hp().default_limits().expect("limits");
        assert_eq!(limits.max(), 8192);
        assert_eq!(
            of(&hp(), limits),
            CardBytes {
                weights: 1_458_167_808,
                scratch: 215_040 * 8192
            }
        );
        assert_eq!(of(&hp(), at(1024)).total(), 1_458_167_808 + 215_040 * 1024);
    }
}
