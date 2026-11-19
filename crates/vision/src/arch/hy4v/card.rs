//! The tower's bytes on its card, from the hyperparameters and the token limits alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's tensors as the card holds them ([`weights`]). Every matrix is bf16,
//!   the f32 patch kernel, position table and both projector convolutions narrowed to it (each
//!   only when every value round-trips through bf16); a block's query, key and value matrices are
//!   one `[3·dim, dim]` matrix with one `3·dim` bias; `ffn_up`'s rows and bias and each `ffn_down`
//!   row are zero-padded to [`ff_padded`] so the GEMM's column tiles fit; every norm gain, norm
//!   bias and linear bias stays f32. The newline row is not a weight on the card: the tower writes
//!   it into its newline rows ([`ScratchLens::newlines`]).
//! * the activations: the buffers of the largest image the limits allow ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use super::hparams::TILE_N;
use super::{Hparams, POS_SIDE};
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

/// The columns `ffn_up` has after its zero padding: a multiple of the GEMM's column tile.
#[must_use]
pub fn ff_padded(hp: &Hparams) -> usize {
    hp.ff.next_multiple_of(TILE_N as usize)
}

/// Bytes of weights the tower uploads (module doc).
#[must_use]
pub fn weights(hp: &Hparams) -> u64 {
    let d = |n: usize| n as u64;
    let (dim, ff, out, merged) = (d(hp.dim), d(ff_padded(hp)), d(hp.out_dim), d(hp.merged()));
    let k_patch = d(3 * hp.patch * hp.patch);
    let pos = d(POS_SIDE * POS_SIDE) * dim;
    let block_mats = 3 * dim * dim + dim * dim + 2 * ff * dim;
    let block_vecs = 2 * dim + 3 * dim + dim + 2 * dim + ff + dim;
    2 * (d(hp.n_layer) * block_mats + dim * k_patch + pos + merged * merged + out * merged)
        + 4 * (d(hp.n_layer) * block_vecs + dim + dim + merged + out)
}

/// The values each activation buffer of the tower holds, for the largest image `limits` allow:
/// `max_tokens` merged tokens, `merge²` times as many patches. Every buffer is bf16; an image of
/// fewer patches uses the first rows of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The patches the buffers are sized for.
    pub max_patches: usize,
    /// The merged tokens they are sized for.
    pub max_tokens: usize,
    /// The image's patches, `3·patch²` values each.
    pub patches: usize,
    /// The position rows of the image's patches.
    pub pos: usize,
    /// The block input and output, the attention-residual stream and the norm's output, each.
    pub stream: usize,
    /// The q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's up rows, the GELU's in place, [`ff_padded`] wide.
    pub up: usize,
    /// The projector's hidden rows: the strided convolution's output, its GELU in place.
    pub hidden: usize,
    /// The projector's output rows, one a merged token.
    pub proj: usize,
    /// The rows the text model reads: each grid row of `nx` merged tokens and its newline row,
    /// `ny·(nx + 1)` rows; at most `2·max_tokens` (a one-token-wide image).
    pub rows: usize,
    /// The newline row repeated `max_tokens` times — one for each grid row of the tallest image:
    /// written once at load, copied after each grid row of an image.
    pub newlines: usize,
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
            pos: max_patches * hp.dim,
            stream: max_patches * hp.dim,
            qkv: max_patches * 3 * hp.dim,
            att: max_patches * hp.dim,
            up: max_patches * ff_padded(hp),
            hidden: max_tokens * hp.merged(),
            proj: max_tokens * hp.out_dim,
            rows: 2 * max_tokens * hp.out_dim,
            newlines: max_tokens * hp.out_dim,
        }
    }

    /// Bytes of every buffer, all bf16: the patches, the position rows, the three streams, qkv,
    /// attention, up, the projector's two and the rows with their newlines.
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
                self.proj,
                self.rows,
                self.newlines,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, ff_padded, of, weights};
    use crate::arch::hy4v::tensors::expected;
    use crate::arch::hy4v::tensors::tests::hp as tests_hp;
    use crate::size::TokenLimits;

    /// The weights are the file's bytes less what the card does not hold or holds narrower: the
    /// newline row, the f32 tensors the load narrows (patch kernel, position table, the two
    /// convolutions) at half their bytes, and the bf16 padding of the MLP.
    #[test]
    fn the_weights_are_the_files_tensors_as_the_card_holds_them() {
        let hp = tests_hp();
        assert_eq!(ff_padded(&hp), 4352);
        let file: u64 = expected(&hp).iter().map(|e| e.bytes()).sum();
        let narrowed: u64 = (1152 * 768 + 128 * 128 * 1152 + 4608 * 4608 + 3072 * 4608) * 2;
        let pad = 27 * ((4352 - 4304) * 1152 * 2 * 2 + (4352 - 4304) * 4);
        assert_eq!(weights(&hp), file - narrowed - 3072 * 4 + pad);
        assert_eq!(weights(&hp), 940_039_680);
    }

    /// At the file's own budget (4,096 tokens, 16,384 patches) the activations are the derived
    /// sum, and the figure grows with the limit.
    #[test]
    fn the_scratch_is_the_largest_images_buffers() {
        let hp = tests_hp();
        let at = |max| ScratchLens::of(&hp, TokenLimits::new(4, max).expect("limits"));
        let l = at(4096);
        assert_eq!((l.max_patches, l.max_tokens), (16_384, 4096));
        let n = 16_384u64;
        let values = n * 768
            + n * 1152
            + 3 * n * 1152
            + n * 3456
            + n * 1152
            + n * 4352
            + 4096 * 4608
            + 4096 * 3072
            + 2 * 4096 * 3072
            + 4096 * 3072;
        assert_eq!(l.bytes(), 2 * values);
        assert_eq!(l.bytes(), 608_174_080);
        assert!(at(1024).bytes() < l.bytes());
        let card = of(&hp, TokenLimits::new(4, 4096).expect("limits"));
        assert_eq!(card.weights + card.scratch, 1_548_213_760);
    }
}
