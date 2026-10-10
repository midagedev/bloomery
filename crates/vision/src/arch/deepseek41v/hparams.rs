//! Every hyperparameter a `deepseek41v` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — the reference config (`inference/config.json`) has a single
//! value for each `vision_*` key — so each key must hold exactly that value, and the error names
//! the key, the value the file holds and the value the reference runs. Values the reference reads
//! but the file does not carry keep the reference's value: the RoPE base (`vision_rope_theta`) is
//! not in the file.

use gguf::Gguf;

use super::PROJECTOR_TYPE;
use super::grid::GridParams;
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, Meta, check_architecture, check_eps, check_mean_std, check_projector,
    check_true, counts, need, refusal,
};

const KEY_SILU: &str = "clip.use_silu";
const KEY_WH_RATIO: &str = "clip.vision.image_max_wh_ratio";

/// `(key, the reference's value, the config.json name it comes from)` for the counts.
const COUNTS: [(&str, u64, &str); 9] = [
    ("clip.vision.block_count", 32, "config.json vision_n_layers"),
    (
        "clip.vision.embedding_length",
        1024,
        "config.json vision_dim",
    ),
    (
        "clip.vision.attention.head_count",
        16,
        "config.json vision_n_heads",
    ),
    (
        "clip.vision.feed_forward_length",
        2816,
        "config.json vision_inter_dim",
    ),
    (
        "clip.vision.patch_size",
        14,
        "config.json vision_patch_size",
    ),
    (
        "clip.vision.projector.scale_factor",
        3,
        "config.json vision_downsample_ratio",
    ),
    (
        "clip.vision.image_max_tokens",
        1024,
        "config.json vision_max_n_token",
    ),
    (
        "clip.vision.image_min_pixels",
        295_936,
        "config.json vision_min_pixels",
    ),
    ("clip.vision.projection_dim", 5120, "config.json dim"),
];

/// vision.py `RMSNorm(dim, eps=1e-6)`; torch adds it to an f32 tensor, so it is this f32.
const EPS: f32 = 1e-6;
/// config.json `vision_rope_theta`.
const ROPE_THETA: f32 = 10_000.0;

/// The hyperparameters of one `deepseek41v` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads; a head is `dim / n_head` wide and RoPE turns all of it.
    pub n_head: usize,
    /// The MLP's hidden width (each of gate and up).
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per aligner cell side.
    pub downsample: usize,
    /// Most tokens per image, delimiters included.
    pub max_tokens: usize,
    /// Smaller images are scaled up to this many pixels.
    pub min_pixels: usize,
    /// The aligner's output width, the text model's embedding width.
    pub out_dim: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// 2D RoPE base; not in the file.
    pub rope_theta: f32,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this crate does not run.
    pub fn read(gguf: &Gguf) -> Result<Hparams, VisionError> {
        check_architecture(gguf)?;
        Hparams::from_meta(gguf)
    }

    /// The resize plan's parameters.
    #[must_use]
    pub fn grid(&self) -> GridParams {
        GridParams {
            patch: self.patch,
            downsample: self.downsample,
            max_tokens: self.max_tokens,
            min_pixels: self.min_pixels,
        }
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        let why = "the ViT and its SiLU-gated MLP";
        for key in [KEY_HAS_VISION, KEY_SILU] {
            check_true(m, key, why)?;
        }
        let [
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            downsample,
            max_tokens,
            min_pixels,
            out_dim,
        ] = counts(m, PROJECTOR_TYPE, COUNTS)?;
        let eps = check_eps(m, EPS, "vision.py's RMSNorm")?;
        check_mean_std(m, "load_image normalizes (x - 0.5) / 0.5 per channel")?;
        // The reference's `vision_max_wh_ratio` is null; the file writes that as 0.
        let v = need(m, KEY_WH_RATIO)?;
        if v.as_unsigned() != Some(0) {
            return Err(refusal(
                KEY_WH_RATIO,
                format!(
                    "is {v:?}; the reference sets no aspect limit (vision_max_wh_ratio null, written 0)"
                ),
            ));
        }
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            downsample,
            max_tokens,
            min_pixels,
            out_dim,
            eps,
            rope_theta: ROPE_THETA,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Hparams;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    /// The keys this module reads, with the values of smalinin's mmproj header.
    pub(crate) fn v41() -> Table {
        let half = || Value::Array(vec![Value::F32(0.5); 3]);
        Table(vec![
            ("clip.projector_type", Value::String("deepseek41v".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_silu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(32)),
            ("clip.vision.embedding_length", Value::U32(1024)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(2816)),
            ("clip.vision.patch_size", Value::U32(14)),
            ("clip.vision.projector.scale_factor", Value::U32(3)),
            ("clip.vision.image_max_tokens", Value::U32(1024)),
            ("clip.vision.image_min_pixels", Value::U32(295_936)),
            ("clip.vision.projection_dim", Value::U32(5120)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.image_mean", half()),
            ("clip.vision.image_std", half()),
            ("clip.vision.image_max_wh_ratio", Value::U32(0)),
        ])
    }

    #[test]
    fn v41_header_reads() {
        let hp = Hparams::from_meta(&v41()).expect("the V4.1 header reads");
        assert_eq!(
            (
                hp.n_layer,
                hp.dim,
                hp.n_head,
                hp.ff,
                hp.patch,
                hp.downsample,
                hp.max_tokens,
                hp.min_pixels,
                hp.out_dim
            ),
            (32, 1024, 16, 2816, 14, 3, 1024, 295_936, 5120)
        );
        assert_eq!((hp.eps, hp.rope_theta), (1e-6, 10_000.0));
    }

    /// Every value this crate does not run is refused, and the error names the key and the value
    /// the file holds.
    #[test]
    fn other_values_are_refused_by_key_and_value() {
        let cases = vec![
            (
                v41().with("clip.projector_type", Value::String("deepseek4v".into())),
                "metadata clip.projector_type: is \"deepseek4v\";",
            ),
            (
                v41().with("clip.vision.block_count", Value::U32(24)),
                "metadata clip.vision.block_count: is 24; deepseek41v runs 32",
            ),
            (
                v41().with("clip.vision.embedding_length", Value::U32(1152)),
                "metadata clip.vision.embedding_length: is 1152;",
            ),
            (
                v41().with("clip.vision.attention.head_count", Value::U32(12)),
                "metadata clip.vision.attention.head_count: is 12;",
            ),
            (
                v41().with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                v41().with("clip.vision.patch_size", Value::U32(16)),
                "metadata clip.vision.patch_size: is 16;",
            ),
            (
                v41().with("clip.vision.projector.scale_factor", Value::U32(2)),
                "metadata clip.vision.projector.scale_factor: is 2;",
            ),
            (
                v41().with("clip.vision.image_max_tokens", Value::U32(384)),
                "metadata clip.vision.image_max_tokens: is 384;",
            ),
            (
                v41().with("clip.vision.image_min_pixels", Value::U32(200_704)),
                "metadata clip.vision.image_min_pixels: is 200704;",
            ),
            (
                v41().with("clip.vision.projection_dim", Value::U32(4096)),
                "metadata clip.vision.projection_dim: is 4096;",
            ),
            (
                v41().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-5;",
            ),
            (
                v41().with(
                    "clip.vision.image_mean",
                    Value::Array(vec![Value::F32(0.485); 3]),
                ),
                "metadata clip.vision.image_mean: is Array",
            ),
            (
                v41().with("clip.vision.image_max_wh_ratio", Value::U32(2)),
                "metadata clip.vision.image_max_wh_ratio: is U32(2);",
            ),
            (
                v41().with("clip.use_silu", Value::Bool(false)),
                "metadata clip.use_silu: is Some(false);",
            ),
            (
                v41().without("clip.vision.patch_size"),
                "metadata clip.vision.patch_size: is absent",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }
}
