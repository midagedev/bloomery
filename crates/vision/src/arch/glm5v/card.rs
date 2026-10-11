//! The tower's bytes on its card, from the hyperparameters and the token limit alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's tensors as the card holds them ([`weights`]). Every matrix is bf16,
//!   and the three f32 convolution weights are narrowed to bf16 on upload (`arch::narrow`), so the
//!   two patch kernels are one `[dim × 2·3·patch²]` matrix of the same bytes. Every output width of
//!   a linear layer is a multiple of the GEMM tile ([`super::hparams::TILE_N`]), so nothing is padded. Vectors stay
//!   f32.
//! * the activations: the buffers of the largest image the limit allows ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use super::tensors::{OnCard, expected};
use super::{Hparams, MERGER_FF, TEMPORAL_FRAMES};
pub use crate::arch::card::CardBytes;
use crate::arch::card::activation_bytes;

/// The tower of a file of `hp` for images of at most `max_tokens` merged tokens (module doc).
#[must_use]
pub fn of(hp: &Hparams, max_tokens: usize) -> CardBytes {
    CardBytes {
        weights: weights(hp),
        scratch: ScratchLens::of(hp, max_tokens).bytes(),
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

/// The values each activation buffer of the tower holds, for the largest image the limit allows:
/// `max_tokens` merged tokens and `merge²` times as many patches. `cs` holds f32 values, every
/// other buffer bf16. An image of fewer patches uses the first rows of each buffer.
///
/// The tower's gate and up are one GEMM of `2·ff` outputs (`gu`) and the SwiGLU's output a buffer
/// beside it (`act`); the merger's likewise (`mgu`, `mact`). A merge group's four patch rows are
/// contiguous in `h` (patch major, channel fastest); the convolution's weight is stored with the
/// column fastest and the channel slowest, so its K axis is permuted to that order on upload, the
/// bytes unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The ViT rows the buffers are sized for.
    pub max_patches: usize,
    /// The merged tokens they are sized for.
    pub max_tokens: usize,
    /// The image's patches, `2·3·patch²` values each (the temporal frames repeat the image).
    pub patches: usize,
    /// Its RoPE table: a cosine and a sine per pair of a head (`head / 2` pairs), per patch.
    pub cs: usize,
    /// The block input and output, and the norm's output.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's fused gate and up rows, and the clamped SwiGLU's output.
    pub gu: usize,
    pub act: usize,
    /// The merger's convolution output and its linear's output (the LayerNorm and the GELU run in
    /// place), per token.
    pub conv: usize,
    pub fc: usize,
    /// The merger MLP's fused gate and up rows, the clamped SwiGLU's output and the MLP's output
    /// (the rows the text model reads), per token.
    pub mgu: usize,
    pub mact: usize,
    pub rows: usize,
}

impl ScratchLens {
    /// The buffers of the largest plan `max_tokens` allow.
    #[must_use]
    pub fn of(hp: &Hparams, max_tokens: usize) -> ScratchLens {
        let n = max_tokens * hp.merge * hp.merge;
        ScratchLens {
            max_patches: n,
            max_tokens,
            patches: n * TEMPORAL_FRAMES * 3 * hp.patch * hp.patch,
            cs: n * (hp.dim / hp.n_head),
            xa: n * hp.dim,
            xb: n * hp.dim,
            h: n * hp.dim,
            qkv: n * 3 * hp.dim,
            att: n * hp.dim,
            gu: n * 2 * hp.ff,
            act: n * hp.ff,
            conv: max_tokens * hp.out_dim,
            fc: max_tokens * hp.out_dim,
            mgu: max_tokens * 2 * MERGER_FF,
            mact: max_tokens * MERGER_FF,
            rows: max_tokens * hp.out_dim,
        }
    }

    /// Bytes of the per-patch buffers: four per `cs` value, two per value of every other.
    #[must_use]
    pub fn patch_bytes(&self) -> u64 {
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
            ],
        )
    }

    /// Bytes of the per-token buffers, all bf16.
    #[must_use]
    pub fn token_bytes(&self) -> u64 {
        activation_bytes(0, [self.conv, self.fc, self.mgu, self.mact, self.rows])
    }

    /// Bytes on the card.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.patch_bytes() + self.token_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, of, weights};
    use crate::arch::glm5v::MERGER_FF;
    use crate::arch::glm5v::hparams::TILE_N;
    use crate::arch::glm5v::hparams::tests::hp;
    use crate::arch::glm5v::tensors::expected;

    /// Every output width of a GEMM is a multiple of the tile, so no weight is padded: the second
    /// axis of every matrix (the convolution's last), and the fused gate and up of the tower and
    /// of the merger.
    #[test]
    fn every_gemm_width_is_a_tile() {
        let hp = hp();
        let widths = expected(&hp)
            .into_iter()
            .filter_map(|e| match e.dims.len() {
                2 => Some(e.dims[1]),
                4 => Some(e.dims[3]),
                _ => None,
            });
        for n in widths.chain([2 * hp.ff as u64, 2 * MERGER_FF as u64]) {
            assert!(n.is_multiple_of(TILE_N), "{n}");
        }
    }

    /// The weights are the shapes' bytes, summed by hand: bf16 values at two bytes — a block's
    /// qkv (1024·3072), out (1024²) and gate, up and down (3 · 1024·4096), 24 blocks; the
    /// merger's convolution (4096·4096), fc (4096²) and three MLP matrices (3 · 4096·10240); the
    /// two patch kernels (2 · 14·14·3·1024) — and f32 vectors at four: a block's norms (2 · 64 and
    /// 2 · 1024) and biases (3072 + 1024 + 2 · 4096 + 1024), 24 blocks; the patch bias, the final
    /// norm, the convolution's bias and the LayerNorm's gain and bias (1024, 1024, 4096, 2 · 4096).
    /// The file's own tensor data is 1,163,988,992 bytes: the kernels (2 · 2,408,448) and the
    /// convolution (67,108,864 f32) shrink by the narrowing to 2,408,448 and 33,554,432.
    #[test]
    fn the_weights_are_the_shapes_bytes() {
        let blk_mats: u64 = 1024 * 3072 + 1024 * 1024 + 3 * 1024 * 4096;
        let mats = 24 * blk_mats
            + (4096 * 4096 + 4096 * 4096 + 3 * 4096 * 10_240)
            + 2 * 14 * 14 * 3 * 1024;
        let blk_vecs: u64 = 2 * 64 + 2 * 1024 + 3072 + 1024 + 2 * 4096 + 1024;
        let vecs = 24 * blk_vecs + (1024 + 1024 + 4096 + 2 * 4096);
        assert_eq!(mats * 2 + vecs * 4, 1_128_026_112);
        assert_eq!(weights(&hp()), 1_128_026_112);
        assert_eq!(
            1_163_988_992 - 2_408_448 - (67_108_864 - 33_554_432),
            1_128_026_112
        );
    }

    /// Scratch is 41,520 bytes a patch and 86,016 a token: a patch holds its 1176 input values, 3
    /// block buffers and the qkv and attention rows (3 · 1024 + 3072 + 1024), the fused gate and
    /// up (8192) and the SwiGLU output (4096) in bf16, and 64 f32 RoPE values; a token holds the
    /// convolution and linear outputs (2 · 4096), the merger MLP's fused rows (20,480), its SwiGLU
    /// output (10,240) and its output rows (4096). A token takes four patches.
    #[test]
    fn scratch_is_41520_a_patch_and_86016_a_token() {
        let l = ScratchLens::of(&hp(), 8000);
        assert_eq!(l.max_patches, 32_000);
        assert_eq!(
            2 * (1176 + 3 * 1024 + 3072 + 1024 + 8192 + 4096) + 4 * 64,
            41_520
        );
        assert_eq!(2 * (2 * 4096 + 20_480 + 10_240 + 4096), 86_016);
        assert_eq!(l.patch_bytes(), 32_000 * 41_520);
        assert_eq!(l.token_bytes(), 8000 * 86_016);
        assert_eq!(l.bytes(), 2_016_768_000);
        assert_eq!(ScratchLens::of(&hp(), 4096).bytes(), 1_032_585_216);
    }

    /// The card figure at the file's own cap of 8000 tokens.
    #[test]
    fn the_card_figure() {
        let card = of(&hp(), 8000);
        assert_eq!((card.weights, card.scratch), (1_128_026_112, 2_016_768_000));
        assert_eq!(card.total(), 3_144_794_112);
    }
}
