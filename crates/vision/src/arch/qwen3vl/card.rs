//! The tower's bytes on its card, from the hyperparameters and the token limits alone — the one
//! figure the loaded tower and a plan that reserves room for it both read:
//!
//! * the weights: the file's tensors as the card holds them ([`weights`]). Every matrix is bf16
//!   (an f16 export is narrowed on upload, `arch::narrow`), so the figure is the same for both
//!   exports. The two patch kernels are one `[dim × 2·3·patch²]` matrix, and the MLP's hidden
//!   width is padded up to the GEMM's tile ([`padded_ff`]): `ffn_up`'s weight and bias gain zero
//!   rows, `ffn_down`'s weight zero columns. Vectors and the position table stay f32.
//! * the activations: the buffers of the largest image the limits allow ([`ScratchLens`]), which
//!   the tower allocates from these lengths.

use std::collections::HashSet;

use super::hparams::TILE_N;
use super::size::TokenLimits;
use super::tensors::{Export, expected};
use super::{Hparams, TEMPORAL_FRAMES, names};
pub use crate::arch::card::CardBytes;

/// The tower of a file of `hp` at `limits` image tokens (module doc).
#[must_use]
pub fn of(hp: &Hparams, limits: TokenLimits) -> CardBytes {
    CardBytes {
        weights: weights(hp),
        scratch: ScratchLens::of(hp, limits).bytes(),
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
    let patch_values = (TEMPORAL_FRAMES * 3 * hp.patch * hp.patch) as u64;
    let named = |f: fn(usize) -> String| -> HashSet<String> { (0..hp.n_layer).map(f).collect() };
    let (up, down, up_bias) = (
        named(names::ffn_up_weight),
        named(names::ffn_down_weight),
        named(names::ffn_up_bias),
    );
    // A bf16 export's table: the types it gives are the ones the card holds, but for the kernels.
    expected(hp, Export::Bf16)
        .iter()
        .map(|e| {
            let name = e.name.as_str();
            if name == names::patch_embd_weight() {
                dim * patch_values * 2
            } else if name == names::patch_embd_weight_second_frame() {
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

/// The values each activation buffer of the tower holds, for the largest image `limits` allow:
/// `max_tokens` merged tokens and `merge²` times as many patches. `cs` holds f32 values, every
/// other buffer bf16. An image of fewer patches uses the first rows of each buffer.
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
    /// The learned position rows added to the patch embedding.
    pub pos: usize,
    /// The block input and output, and the norm's output.
    pub xa: usize,
    pub xb: usize,
    pub h: usize,
    /// The fused q, k, v rows.
    pub qkv: usize,
    /// The attention output.
    pub att: usize,
    /// The MLP's hidden rows at the padded width, the GELU applied in place.
    pub u: usize,
    /// The merger's hidden rows (`merge²` patch rows wide) and its output rows.
    pub hh: usize,
    pub rows: usize,
}

impl ScratchLens {
    /// The buffers of the largest plan `limits` allow.
    #[must_use]
    pub fn of(hp: &Hparams, limits: TokenLimits) -> ScratchLens {
        let tokens = limits.max_tokens();
        let n = tokens * hp.merge * hp.merge;
        let merged = hp.dim * hp.merge * hp.merge;
        ScratchLens {
            max_patches: n,
            max_tokens: tokens,
            patches: n * TEMPORAL_FRAMES * 3 * hp.patch * hp.patch,
            cs: n * (hp.dim / hp.n_head),
            pos: n * hp.dim,
            xa: n * hp.dim,
            xb: n * hp.dim,
            h: n * hp.dim,
            qkv: n * 3 * hp.dim,
            att: n * hp.dim,
            u: n * padded_ff(hp),
            hh: tokens * merged,
            rows: tokens * hp.out_dim,
        }
    }

    /// Bytes on the card: four per `cs` value, two per value of every other buffer.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let bf16 = [
            self.patches,
            self.pos,
            self.xa,
            self.xb,
            self.h,
            self.qkv,
            self.att,
            self.u,
            self.hh,
            self.rows,
        ];
        4 * self.cs as u64 + 2 * bf16.iter().map(|&v| v as u64).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::{CardBytes, ScratchLens, of, padded_ff, weights};
    use crate::arch::qwen3vl::{Hparams, TokenLimits};

    /// The hyperparameters of a file whose merger writes `out_dim` values.
    fn hp(out_dim: usize) -> Hparams {
        Hparams {
            n_layer: 27,
            dim: 1152,
            n_head: 16,
            ff: 4304,
            patch: 16,
            merge: 2,
            out_dim,
            eps: 1e-6,
        }
    }

    fn at(tokens: usize) -> TokenLimits {
        TokenLimits::new(8, tokens).expect("limits")
    }

    /// The MLP's hidden width is the next multiple of the GEMM's 64-wide tile: 4304 → 4352.
    #[test]
    fn the_hidden_width_pads_to_the_tile() {
        assert_eq!(padded_ff(&hp(4096)), 4352);
    }

    /// Weights, derived. A block holds four f32 norm vectors (4·1152·4 = 18,432 B), the qkv
    /// matrix (1152·3456·2 = 7,962,624) with its bias (13,824), the output matrix (2,654,208)
    /// with its bias (4,608), `ffn_up` and `ffn_down` at the padded 4352 (1152·4352·2 =
    /// 10,027,008 each) with `ffn_up`'s bias at 4352·4 = 17,408 and `ffn_down`'s at 4,608: 30,729,728
    /// B, 829,702,656 for 27. The head is the merged patch kernel (1152·1536·2 = 3,538,944), its
    /// bias (4,608) and the f32 position table (2304·1152·4 = 10,616,832); the tail the final
    /// norm (9,216), `mm.0` (4608²·2 = 42,467,328 and its bias 18,432) and `mm.2`
    /// (4608·out·2 and out·4). Together 886,358,016 + 9,220·out.
    #[test]
    fn the_weights_of_the_three_files() {
        for (file, out, want) in [
            ("Clef Flash", 4096, 924_123_136),
            ("Qwen3.6-35B-A3B", 2048, 905_240_576),
            ("Qwen3.8-Flash-Next", 2560, 909_961_216),
        ] {
            assert_eq!(weights(&hp(out)), want, "{file}");
            assert_eq!(want, 886_358_016 + 9_220 * out as u64, "{file}");
        }
    }

    /// Activations, derived. A patch holds its 1536 input values (3,072 B), a cosine and sine per
    /// pair of its 72-wide head (288 B), five rows of 1152 (positions, block input and output,
    /// norm output, attention output: 11,520 B), the qkv row (3456 values, 6,912 B) and the MLP
    /// row at 4352 (8,704 B): 30,496 B, and a token's 4 patches 121,984 B. A token adds the
    /// merger's 4608-wide hidden row (9,216 B) and its output row (2·out). Per token of the limit
    /// 131,200 + 2·out B.
    #[test]
    fn the_scratch_of_the_three_files_at_two_limits() {
        for (file, out, per_token) in [
            ("Clef Flash", 4096, 139_392),
            ("Qwen3.6-35B-A3B", 2048, 135_296),
            ("Qwen3.8-Flash-Next", 2560, 136_320),
        ] {
            assert_eq!(per_token, 131_200 + 2 * out as u64, "{file}");
            for tokens in [1024, 4096] {
                assert_eq!(
                    ScratchLens::of(&hp(out), at(tokens)).bytes(),
                    per_token * tokens as u64,
                    "{file} at {tokens} tokens"
                );
            }
        }
        assert_eq!(ScratchLens::of(&hp(4096), at(4096)).max_patches, 16_384);
    }

    /// The six figures a plan reserves: weights plus scratch.
    #[test]
    fn the_card_figures_at_the_two_limits() {
        let card = |out, tokens| of(&hp(out), at(tokens));
        assert_eq!(
            card(4096, 1024),
            CardBytes {
                weights: 924_123_136,
                scratch: 142_737_408
            }
        );
        assert_eq!(card(4096, 4096).scratch, 570_949_632);
        assert_eq!(card(2048, 1024).scratch, 138_543_104);
        assert_eq!(card(2048, 4096).scratch, 554_172_416);
        assert_eq!(card(2560, 1024).scratch, 139_591_680);
        assert_eq!(card(2560, 4096).scratch, 558_366_720);
        assert_eq!(card(2048, 4096).total(), 905_240_576 + 554_172_416);
    }
}
