//! The tower's bytes on its card, from the hyperparameters and the token limits alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's tensors as the card holds them ([`weights`]). Every matrix is bf16,
//!   the f32 patch kernel and patch merger narrowed to it; a block's query, key and value
//!   matrices are one `[3·dim, dim]` matrix and its gate and up one `[2·ff, dim]`; the norm gains
//!   stay f32; the `[IMG_BREAK]` row is not a weight on the card: the tower writes it into its
//!   break rows ([`ScratchLens::breaks`]).
//! * the activations: the buffers of the largest image the limits allow ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use super::tensors::expected;
use super::{Hparams, names};
pub use crate::arch::card::CardBytes;
use crate::arch::card::activation_bytes;
use crate::size::TokenLimits;

/// `(cos, sin)` values a patch's RoPE table row holds: one pair per two values of a head of 128.
const CS_PER_PATCH: usize = 128;

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
    let gains = [names::pre_ln_weight(), names::input_norm_weight()];
    let narrowed = [names::patch_embd_weight(), names::patch_merger_weight()];
    let brk = names::img_break();
    expected(hp)
        .iter()
        .map(|e| {
            let name = e.name.as_str();
            let n: u64 = e.dims.iter().product();
            if name == brk {
                0
            } else if narrowed.iter().any(|t| t == name) {
                2 * n
            } else if gains.iter().any(|t| t == name)
                || name.ends_with(".ln1.weight")
                || name.ends_with(".ln2.weight")
            {
                4 * n
            } else {
                e.bytes()
            }
        })
        .sum()
}

/// The values each activation buffer of the tower holds, for the largest image `limits` allow:
/// `max_tokens` merged tokens, `merge²` times as many patches. Every buffer is bf16 but the RoPE
/// table; an image of fewer patches uses the first rows of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The patches the buffers are sized for.
    pub max_patches: usize,
    /// The merged tokens they are sized for.
    pub max_tokens: usize,
    /// The image's patches, `3·patch²` values each.
    pub patches: usize,
    /// The RoPE table, f32: a `(cos, sin)` pair per two values of a head, per patch.
    pub cs: usize,
    /// The block input and output, the attention-residual stream and the norm's output, each.
    pub stream: usize,
    /// The q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's gate and up rows.
    pub gu: usize,
    /// The SiLU-gated rows.
    pub act: usize,
    /// The projector's first linear's rows, GELU in place.
    pub mid: usize,
    /// The projector's output rows, one a merged token.
    pub proj: usize,
    /// The rows the text model reads: each grid row of `nx` merged tokens and the slot of the
    /// `[IMG_BREAK]` row after it, `ny·(nx + 1)` rows the row copies reach, the image's own rows
    /// the first `nx·ny + ny − 1` of them; at most `2·max_tokens` (a one-token-wide image).
    pub rows: usize,
    /// The `[IMG_BREAK]` row repeated `max_tokens` times — one more than the tallest grid's
    /// breaks, so the buffer is never empty: written once at load, copied after each grid row of
    /// an image.
    pub breaks: usize,
}

impl ScratchLens {
    /// The lengths for images of at most `limits.max()` tokens.
    #[must_use]
    pub fn of(hp: &Hparams, limits: TokenLimits) -> ScratchLens {
        let max_tokens = limits.max();
        let max_patches = max_tokens * hp.merge * hp.merge;
        ScratchLens {
            max_patches,
            max_tokens,
            patches: max_patches * 3 * hp.patch * hp.patch,
            cs: max_patches * CS_PER_PATCH,
            stream: max_patches * hp.dim,
            qkv: max_patches * 3 * hp.dim,
            att: max_patches * hp.dim,
            gu: max_patches * 2 * hp.ff,
            act: max_patches * hp.ff,
            mid: max_tokens * hp.out_dim,
            proj: max_tokens * hp.out_dim,
            rows: 2 * max_tokens * hp.out_dim,
            breaks: max_tokens * hp.out_dim,
        }
    }

    /// Bytes of every buffer: the f32 RoPE table, then the patches, the three streams, qkv,
    /// attention, gate and up, the gated rows, the projector's two and the rows with their breaks,
    /// all bf16.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        activation_bytes(
            self.cs,
            [
                self.patches,
                3 * self.stream,
                self.qkv,
                self.att,
                self.gu,
                self.act,
                self.mid,
                self.proj,
                self.rows,
                self.breaks,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, of, weights};
    use crate::arch::pixtral::tensors::tests::hp;
    use crate::size::TokenLimits;

    /// The weights from the file's own tensors, by hand: per block four `dim²` matrices and three
    /// `dim·ff` ones in bf16 plus two f32 gains; the patch kernel and the patch merger narrowed to
    /// bf16; the two f32 gains outside the blocks; the projector's two bf16 matrices; no
    /// `[IMG_BREAK]` row.
    #[test]
    fn the_weights_are_the_files_tensors_as_the_card_holds_them() {
        let (dim, ff, out) = (2048u64, 6144u64, 6144u64);
        let block = 2 * (4 * dim * dim + 3 * dim * ff) + 4 * 2 * dim;
        let total = 30 * block
            + 2 * dim * 588
            + 2 * 4 * dim * dim
            + 4 * 2 * dim
            + 2 * (dim * out + out * out);
        assert_eq!(weights(&hp()), total);
        // The file's 3,444,678,656 B of tensor data less the f32 patch kernel and merger narrowed
        // to bf16 and the break row.
        assert_eq!(
            total,
            3_444_678_656 - 2 * dim * 588 - 2 * 4 * dim * dim - 4 * out
        );
    }

    /// 1,024 tokens: 4,096 patches; a one-token-wide image of 1,024 tokens is 2,047 rows, and the
    /// row copies reach the break slot after its last row.
    #[test]
    fn the_scratch_is_the_largest_images_buffers() {
        let limits = TokenLimits::new(8, 1024).expect("limits");
        let lens = ScratchLens::of(&hp(), limits);
        assert_eq!(lens.max_patches, 4096);
        assert_eq!(lens.stream, 4096 * 2048);
        assert_eq!(crate::arch::pixtral::rows_of(1, 1024), 2047);
        assert_eq!(lens.rows, 2048 * 6144);
        assert_eq!(lens.breaks, 1024 * 6144);
        assert_eq!(of(&hp(), limits).scratch, lens.bytes());
        assert_eq!(of(&hp(), limits).weights, weights(&hp()));
    }
}
