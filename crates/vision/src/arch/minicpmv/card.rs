//! The tower's bytes on its card, from the hyperparameters and the largest tile alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights ([`weights`]): the file's tensors as the card holds them. Every matrix is bf16
//!   (an f16 export would be narrowed on upload, `arch::narrow`); the patch kernel is f32 on disk
//!   and bf16 on the card, so it is narrowed. The MLP's hidden width is padded up to the GEMM's
//!   tile ([`padded_ff`]): `ffn_up`'s weight and bias gain zero rows, `ffn_down`'s weight zero
//!   columns. Vectors stay f32. The 70×70 position table and the four separator rows stay on the
//!   host: a tile gathers its position rows from the table, and the separators are fed to the
//!   text model as the file holds them.
//! * the activations ([`ScratchLens`]): the buffers one tile of `max_patches` patches needs,
//!   which the tower allocates from these lengths. A tile is planned to about `slice_size²`
//!   pixels but a very wide image gives a long thin one, so the caller states its bound.

use super::hparams::TILE_N;
use super::tensors::{Export, expected};
use super::{Hparams, names};
pub use crate::arch::card::CardBytes;
use crate::arch::card::activation_bytes;
use std::collections::HashSet;

/// The tower of a file of `hp` for tiles of at most `max_patches` patches.
#[must_use]
pub fn of(hp: &Hparams, max_patches: usize) -> CardBytes {
    CardBytes {
        weights: weights(hp),
        scratch: ScratchLens::of(hp, max_patches).bytes(),
    }
}

/// The MLP's hidden width on the card: `ff` up to a multiple of the GEMM tile. The padded rows of
/// `ffn_up` are zero with a zero bias, so their activations are `gelu(0) = 0` and add nothing to
/// `ffn_down`.
#[must_use]
pub fn padded_ff(hp: &Hparams) -> usize {
    hp.ff.next_multiple_of(TILE_N as usize)
}

/// Bytes of weights the tower uploads (module doc).
#[must_use]
pub fn weights(hp: &Hparams) -> u64 {
    let (dim, ff) = (hp.dim as u64, padded_ff(hp) as u64);
    let k_patch = (3 * hp.patch * hp.patch) as u64;
    let named = |f: fn(usize) -> String| -> HashSet<String> { (0..hp.n_layer).map(f).collect() };
    let (up, down, up_bias) = (
        named(names::ffn_up_weight),
        named(names::ffn_down_weight),
        named(names::ffn_up_bias),
    );
    expected(hp, Export::Bf16)
        .iter()
        .map(|e| {
            let name = e.name.as_str();
            if name == names::patch_embd_weight() {
                dim * k_patch * 2
            } else if name == names::position_embd() || name == names::tok_embd_sep() {
                0
            } else if up.contains(name) || down.contains(name) {
                dim * ff * 2
            } else if up_bias.contains(name) {
                ff * 4
            } else {
                e.bytes()
            }
        })
        .sum()
}

/// The values each activation buffer holds, all bf16, for a tile of `max_patches` patches. A tile
/// of fewer patches uses the first rows of each buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLens {
    /// The ViT rows the buffers are sized for.
    pub max_patches: usize,
    /// The patches' values, `3·patch²` each.
    pub patches: usize,
    /// The learned position rows added to the patch embedding, gathered per grid and rounded to
    /// bf16, the patch GEMM's residual operand.
    pub pos: usize,
    /// The block input and output, the attention block's output, and the norm's output.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub attn: usize,
    /// The MLP's hidden rows at the padded width. The vit_merger's hidden rows (a quarter of the
    /// patches, `4·ff` wide) and the final merger's up rows (a sixteenth, `4·dim` wide) live in
    /// the same buffer, and its normed rows in `h`.
    pub up: usize,
    /// The down-sample's mean rows: a quarter of the patches, `dim` wide.
    pub mean: usize,
    /// The output rows: a sixteenth of the patches, `out_dim` wide.
    pub out: usize,
}

impl ScratchLens {
    /// The buffers of a tile of `max_patches` patches (a multiple of 16: whole merged tokens).
    #[must_use]
    pub fn of(hp: &Hparams, max_patches: usize) -> ScratchLens {
        let n = max_patches;
        ScratchLens {
            max_patches: n,
            patches: n * 3 * hp.patch * hp.patch,
            pos: n * hp.dim,
            xa: n * hp.dim,
            xb: n * hp.dim,
            h: n * hp.dim,
            qkv: 3 * n * hp.dim,
            attn: n * hp.dim,
            up: n * padded_ff(hp),
            mean: n / 4 * hp.dim,
            out: n / 16 * hp.out_dim,
        }
    }

    /// The bytes of all the buffers: two per value.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        activation_bytes(
            0,
            [
                self.patches,
                self.pos,
                self.xa,
                self.xb,
                self.h,
                self.qkv,
                self.attn,
                self.up,
                self.mean,
                self.out,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ScratchLens, of, padded_ff, weights};
    use crate::arch::minicpmv::hparams::tests::hp;

    /// The weights follow the table with the card's three differences (the patch kernel in bf16,
    /// the host-side position table and separators gone, the MLP padded), and a 448×448 tile
    /// (1024 patches, 64 tokens) allocates buffers whose element counts follow from the widths.
    #[test]
    fn the_card_follows_the_table() {
        use crate::arch::minicpmv::tensors::{Export, expected};
        let hp = hp();
        let table: u64 = expected(&hp, Export::Bf16).iter().map(|e| e.bytes()).sum();
        let pad = (padded_ff(&hp) - hp.ff) as u64;
        let per_block = 2 * 2 * pad * 1152 + 4 * pad;
        let narrowing = 2 * 14 * 14 * 3 * 1152;
        let (positions, separators) = (4 * 4900 * 1152, 4 * 4 * 2048);
        assert_eq!(
            weights(&hp),
            table + 27 * per_block - narrowing - positions - separators
        );
        let s = ScratchLens::of(&hp, 1024);
        assert_eq!(s.patches, 1024 * 588);
        assert_eq!(s.up, 1024 * 4352);
        assert!(256 * 4 * hp.ff <= s.up && 64 * hp.merged_dim() <= s.up);
        assert_eq!(s.mean, 256 * 1152);
        assert_eq!(s.out, 64 * 2048);
        assert_eq!(of(&hp, 1024).scratch, s.bytes());
        assert_eq!(of(&hp, 1024).total(), weights(&hp) + s.bytes());
    }

    /// The figures of the real 35B-A3B file's hyperparameters: 1,101,559,552 B of weights, and
    /// 29,144 B of activations per patch of a tile (29,843,456 B for the 448×448 slice).
    #[test]
    fn the_figures_of_the_checkpoint() {
        let hp = hp();
        assert_eq!(weights(&hp), 1_101_559_552);
        assert_eq!(of(&hp, 1024).scratch, 29_843_456);
        assert_eq!(of(&hp, 1120).scratch, 1120 * 29_144);
    }
}
