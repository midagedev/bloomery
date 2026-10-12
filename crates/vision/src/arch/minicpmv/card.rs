//! The tower's bytes on its card, from the hyperparameters and the largest tile alone.
//!
//! * the weights: the file's tensors as the card holds them. Every matrix is bf16 (an f16 export
//!   would be narrowed on upload, `arch::narrow`); vectors, the patch kernel, the position table
//!   and the separator rows stay f32. The GEMM's tile padding of a width that is not a multiple
//!   of it (the MLP's 4304) is the tower's to add.
//! * the activations: the buffers one tile of `max_patches` patches needs ([`ScratchLens`]),
//!   which the tower allocates from these lengths. A tile is planned to about `slice_size²`
//!   pixels but a very wide image gives a long thin one, so the caller states its bound.

use super::Hparams;
use super::tensors::{Export, expected};
pub use crate::arch::card::CardBytes;
use crate::arch::card::activation_bytes;

/// The tower of a file of `hp` for tiles of at most `max_patches` patches.
#[must_use]
pub fn of(hp: &Hparams, max_patches: usize) -> CardBytes {
    CardBytes {
        weights: weights(hp),
        scratch: ScratchLens::of(hp, max_patches).bytes(),
    }
}

/// Bytes of weights the tower uploads: the table's tensors at their types.
#[must_use]
pub fn weights(hp: &Hparams) -> u64 {
    expected(hp, Export::Bf16).iter().map(|e| e.bytes()).sum()
}

/// The values each activation buffer holds, for a tile of `max_patches` patches. `cs` holds f32
/// values, every other buffer bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The ViT rows the buffers are sized for.
    pub max_patches: usize,
    /// The patches' values, `3·patch²` each.
    pub patches: usize,
    /// The learned position rows added to the patch embedding, resampled per tile (f32).
    pub cs: usize,
    /// The block input and output, and the norm's output.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The q, k and v rows.
    pub qkv: usize,
    /// The attention output.
    pub attn: usize,
    /// The MLP's hidden rows.
    pub mlp: usize,
    /// The rows after the ViT merger: a quarter of the patches, `dim · 4` wide, and the merger's
    /// MLP hidden rows.
    pub merged: usize,
    pub merged_mlp: usize,
    /// The final merger's rows: a sixteenth of the patches, `dim · 4` wide in and out of the
    /// norm and the up layer, and the output rows.
    pub final_in: usize,
    pub out: usize,
}

impl ScratchLens {
    /// The buffers of a tile of `max_patches` patches (a multiple of 16: whole merged tokens).
    #[must_use]
    pub fn of(hp: &Hparams, max_patches: usize) -> ScratchLens {
        let tokens = max_patches / 16;
        ScratchLens {
            max_patches,
            patches: max_patches * 3 * hp.patch * hp.patch,
            cs: max_patches * hp.dim,
            xa: max_patches * hp.dim,
            xb: max_patches * hp.dim,
            h: max_patches * hp.dim,
            qkv: 3 * max_patches * hp.dim,
            attn: max_patches * hp.dim,
            mlp: max_patches * hp.ff,
            merged: max_patches / 4 * hp.merged_dim(),
            merged_mlp: max_patches / 4 * 4 * hp.ff,
            final_in: 2 * tokens * hp.merged_dim(),
            out: tokens * hp.out_dim,
        }
    }

    /// The bytes of all the buffers.
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
                self.attn,
                self.mlp,
                self.merged,
                self.merged_mlp,
                self.final_in,
                self.out,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, of, weights};
    use crate::arch::minicpmv::hparams::tests::hp;
    use crate::arch::minicpmv::tensors::{Export, expected};

    /// The weights are the table's bytes, and a 448×448 tile (1024 patches, 64 tokens) allocates
    /// buffers whose element counts follow from the widths.
    #[test]
    fn the_card_follows_the_table() {
        let hp = hp();
        let table: u64 = expected(&hp, Export::Bf16).iter().map(|e| e.bytes()).sum();
        assert_eq!(weights(&hp), table);
        let s = ScratchLens::of(&hp, 1024);
        assert_eq!(s.patches, 1024 * 588);
        assert_eq!(s.mlp, 1024 * 4304);
        assert_eq!(s.merged, 256 * 4608);
        assert_eq!(s.out, 64 * 2048);
        assert_eq!(of(&hp, 1024).scratch, s.bytes());
        assert_eq!(of(&hp, 1024).total(), weights(&hp) + s.bytes());
    }
}
