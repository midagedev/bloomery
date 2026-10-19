//! The tower's bytes on its card, from the hyperparameters and the token limits alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's tensors as the card holds them ([`weights`]). Every matrix is bf16;
//!   the query, key and value matrices of a block are one `[3·dim, dim]` matrix and their biases
//!   one vector; the patch kernel is bf16 and the video kernel is not uploaded; the prefix rows
//!   stay on the host (they are copied into each image's rows); the position table, the norm
//!   gains and the biases stay f32.
//! * the activations: the buffers of the largest image the limits allow ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use super::tensors::expected;
use super::{Hparams, PREFIX_ROWS, PROJECTOR_HIDDEN, names};
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

/// Bytes of weights the tower uploads (module doc).
#[must_use]
pub fn weights(hp: &Hparams) -> u64 {
    let (dim, p) = (hp.dim as u64, hp.patch as u64);
    let (patch, video, class) = (
        names::patch_embd_weight(),
        names::patch_embd_weight_video(),
        names::class_embd(),
    );
    expected(hp)
        .iter()
        .map(|e| {
            let name = e.name.as_str();
            if name == patch {
                dim * 3 * p * p * 2
            } else if name == video || name == class {
                0
            } else {
                e.bytes()
            }
        })
        .sum()
}

/// The values each activation buffer of the tower holds, for the largest image `limits` allow:
/// `max_tokens` merged tokens, `merge²` times as many patches and the prefix rows besides. Every
/// buffer is bf16; an image of fewer patches uses the first rows of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The patches the buffers are sized for.
    pub max_patches: usize,
    /// The merged tokens they are sized for.
    pub max_tokens: usize,
    /// The image's patches, `3·patch²` values each.
    pub patches: usize,
    /// The learned position rows added to the patch embedding.
    pub pos: usize,
    /// The block input and output, the attention-residual stream and the norm's output: the
    /// patches' rows followed by the prefix rows.
    pub stream: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's hidden rows.
    pub up: usize,
    /// The projector's hidden rows.
    pub hidden: usize,
    /// The rows the text model reads.
    pub rows: usize,
}

impl ScratchLens {
    /// The lengths for images of at most `limits.max()` tokens.
    #[must_use]
    pub fn of(hp: &Hparams, limits: TokenLimits) -> ScratchLens {
        let max_tokens = limits.max();
        let max_patches = max_tokens * hp.merge * hp.merge;
        let rows = max_patches + PREFIX_ROWS;
        ScratchLens {
            max_patches,
            max_tokens,
            patches: max_patches * 3 * hp.patch * hp.patch,
            pos: max_patches * hp.dim,
            stream: rows * hp.dim,
            qkv: rows * 3 * hp.dim,
            att: rows * hp.dim,
            up: rows * hp.ff,
            hidden: max_tokens * PROJECTOR_HIDDEN,
            rows: max_tokens * hp.out_dim,
        }
    }

    /// Bytes of every buffer: the patches, the position rows, the three streams, qkv, attention,
    /// up, the projector's hidden rows and the output rows, all bf16.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        activation_bytes(
            0,
            [
                self.patches,
                self.pos,
                3 * self.stream,
                self.qkv,
                self.att,
                self.up,
                self.hidden,
                self.rows,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, of, weights};
    use crate::arch::radio::Hparams;
    use crate::preprocess::ChannelStats;
    use crate::size::TokenLimits;

    fn hp() -> Hparams {
        Hparams {
            n_layer: 32,
            dim: 1280,
            n_head: 16,
            ff: 5120,
            patch: 16,
            merge: 2,
            out_dim: 4096,
            eps: 1e-6,
            stats: ChannelStats::HALF,
        }
    }

    /// The weights from the file's own tensors, by hand: per block four `dim²` matrices and two
    /// `dim·ff` ones in bf16 plus `9·dim + ff` f32 values; the bf16 patch kernel; the f32
    /// position table; the projector's f32 gain and two bf16 matrices.
    #[test]
    fn the_weights_are_the_files_tensors_as_the_card_holds_them() {
        let (dim, ff, hidden, out) = (1280u64, 5120u64, 20_480u64, 4096u64);
        let block = 2 * (4 * dim * dim + 2 * dim * ff) + 4 * (9 * dim + ff);
        let projector = 4 * 4 * dim + 2 * (4 * dim * hidden + hidden * out);
        let total = 32 * block + 2 * dim * 768 + 4 * 16_384 * dim + projector;
        assert_eq!(weights(&hp()), total);
        // The file's 1,733,662,720 B of tensor data less the video kernel, the prefix rows, and
        // the patch kernel's f32 narrowed to bf16.
        assert_eq!(
            total,
            1_733_662_720 - 4 * 6 * 256 * dim - 4 * 10 * dim - 2 * 768 * dim
        );
    }

    /// 3,328 tokens: 13,312 patches and the ten prefix rows.
    #[test]
    fn the_scratch_is_the_largest_images_buffers() {
        let limits = TokenLimits::new(256, 3328).expect("limits");
        let lens = ScratchLens::of(&hp(), limits);
        assert_eq!(lens.max_patches, 13_312);
        assert_eq!(lens.stream, 13_322 * 1280);
        assert_eq!(lens.hidden, 3328 * 20_480);
        assert_eq!(of(&hp(), limits).scratch, lens.bytes());
        assert_eq!(of(&hp(), limits).weights, weights(&hp()));
    }
}
