//! The encoder's bytes on its card, from the hyperparameters alone ([`of`]) — the one figure the
//! loaded encoder (`gpu_vision::Encoder`) and a plan that reserves room for it both read:
//!
//! * the weights: every [`Home::Card`] row of the tensor table at its own type. The span's three
//!   delimiter rows ([`Home::Span`]) are the text model's input, read on the host, and are not
//!   among them. The encoder refuses a load whose uploads differ from this figure.
//! * the activations: the buffers of the largest image a plan makes ([`ScratchLens`]), which the
//!   encoder allocates from these lengths.

use super::Hparams;
use super::tensors::{Home, expected};

pub use crate::arch::card::CardBytes;

/// The encoder's bytes on its card for a file of `hp` (module doc).
#[must_use]
pub fn of(hp: &Hparams) -> CardBytes {
    CardBytes {
        weights: expected(hp)
            .iter()
            .filter(|e| e.home == Home::Card)
            .map(super::tensors::Expected::bytes)
            .sum(),
        scratch: ScratchLens::of(hp).bytes(),
    }
}

/// The values each activation buffer of the encoder holds, for the largest image `hp`'s plan
/// makes: `max_patches` ViT rows and `max_cells` aligner rows (`GridParams`). `cs` holds f32 values,
/// every other buffer bf16. An image of fewer patches uses the first rows of each buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The ViT rows the buffers are sized for.
    pub max_patches: usize,
    /// The aligner rows they are sized for.
    pub max_cells: usize,
    /// The image's patches, `3·patch²` values each.
    pub patches: usize,
    /// Its RoPE table: a cosine and a sine per pair of a head (`head / 2` pairs), per patch.
    pub cs: usize,
    /// The block input and output, and the aligner's input rows when `h` holds the final norm.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's gate and up rows, then their product.
    pub u: usize,
    pub act: usize,
    /// The aligner's unfolded rows, its hidden rows and its output rows.
    pub un: usize,
    pub hh: usize,
    pub rows: usize,
}

impl ScratchLens {
    /// The buffers of `hp`'s largest plan.
    #[must_use]
    pub fn of(hp: &Hparams) -> ScratchLens {
        let grid = hp.grid();
        let (n, n_llm) = (grid.max_patches(), grid.max_cells());
        let (dim, ff) = (hp.dim, hp.ff);
        ScratchLens {
            max_patches: n,
            max_cells: n_llm,
            patches: n * 3 * hp.patch * hp.patch,
            cs: n * (dim / hp.n_head),
            xa: n * dim,
            xb: n * dim,
            h: n * dim,
            qkv: n * 3 * dim,
            att: n * dim,
            u: n * 2 * ff,
            act: n * ff,
            un: n_llm * dim * hp.downsample * hp.downsample,
            hh: n_llm * hp.out_dim,
            rows: n_llm * hp.out_dim,
        }
    }

    /// Bytes on the card: four per `cs` value, two per value of every other buffer.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let bf16 = [
            self.patches,
            self.xa,
            self.xb,
            self.h,
            self.qkv,
            self.att,
            self.u,
            self.act,
            self.un,
            self.hh,
            self.rows,
        ];
        4 * self.cs as u64 + 2 * bf16.iter().map(|&v| v as u64).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::{CardBytes, of};
    use crate::arch::deepseek41v::tensors::tests::hp;

    /// The V4.1 encoder's card bytes. Weights: 970,088,448 B of bf16 matrices (the patch
    /// embedding, 32 blocks of q·k·v, output, gate, up and down, the aligner's two) and 835,584 B
    /// of f32 gains and biases — the span's three f32 rows of 5120 (61,440 B) are not uploaded.
    /// Activations: 9189 patches and 1021 aligner rows.
    #[test]
    fn the_v41_encoder_card_bytes_leave_the_span_rows_out() {
        assert_eq!(
            of(&hp()),
            CardBytes {
                weights: 970_924_032,
                scratch: 339_878_648,
            }
        );
    }
}
