//! Every hyperparameter a `hy_v4_flash` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — Hy4-flash's `vision_config` has a single value for each key —
//! so each count must hold exactly that value, and the error names the key, the value the file
//! holds and the config entry the module runs. The image budget is two pixel counts, 4,096 and
//! 4,194,304 pixels: 4 and 4,096 merged tokens of 32² pixels, the limits a seat starts from
//! ([`Hparams::default_limits`]). The projection width is the text model's embedding width; the
//! other values the file does not carry are the tensors' shapes ([`super::tensors`]).

use gguf::Gguf;

use super::PROJECTOR_TYPE;
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, KEY_MEAN, KEY_STD, Meta, channels, check_architecture, check_eps,
    check_projector, check_true, count_multiple, counts,
};
use crate::preprocess::ChannelStats;
use crate::size::{SizeRule, TokenLimits};

const KEY_GELU: &str = "clip.use_gelu";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";

/// The tile width of the encoder's GEMM (`gpu_vision`'s `gemm_bf16`): every output width of a
/// linear layer is a multiple of it, the projector's included.
pub const TILE_N: u64 = 64;

/// `(key, the value the module runs, where that value comes from)` for the counts. The projection
/// width is not among them: it is the text model's width.
const COUNTS: [(&str, u64, &str); 9] = [
    (
        "clip.vision.block_count",
        27,
        "config.json vision_config.num_hidden_layers",
    ),
    (
        "clip.vision.embedding_length",
        1152,
        "config.json vision_config.hidden_size",
    ),
    (
        "clip.vision.attention.head_count",
        16,
        "config.json vision_config.num_attention_heads",
    ),
    (
        "clip.vision.feed_forward_length",
        4304,
        "config.json vision_config.intermediate_size",
    ),
    (
        "clip.vision.patch_size",
        16,
        "config.json vision_config.patch_size",
    ),
    (
        "clip.vision.spatial_merge_size",
        2,
        "config.json vision_config.spatial_merge_size",
    ),
    (
        "clip.vision.image_size",
        2048,
        "config.json vision_config.max_image_size",
    ),
    (
        "clip.vision.image_min_pixels",
        4096,
        "preprocessor_config.json min_pixels",
    ),
    (
        "clip.vision.image_max_pixels",
        4_194_304,
        "preprocessor_config.json max_pixels",
    ),
];

/// The vision tower's `rms_norm_eps`, which is also the LayerNorm epsilon of its blocks and the
/// projector's RMS norm epsilon, as the f32 the file stores.
const EPS: f32 = 1e-5;

/// The hyperparameters of one `hy_v4_flash` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads; a head is `dim / n_head` wide.
    pub n_head: usize,
    /// The MLP's hidden width.
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the projector folds `merge`² patches into one token.
    pub merge: usize,
    /// The projector's output width: the embedding width of the text model the file pairs with.
    pub out_dim: usize,
    /// LayerNorm and RMS norm epsilon.
    pub eps: f32,
    /// The per-channel normalization of the pixels.
    pub stats: ChannelStats,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this crate does not run.
    pub fn read(gguf: &Gguf) -> Result<Hparams, VisionError> {
        check_architecture(gguf)?;
        Hparams::from_meta(gguf)
    }

    /// The image-token limits of the file's own pixel budget: 4 and 4,096 merged tokens of 32²
    /// pixels each. A seat that is given no limit starts from these.
    pub fn default_limits(&self) -> Result<TokenLimits, VisionError> {
        let per_token = (self.patch * self.merge).pow(2);
        TokenLimits::new(
            COUNTS[7].1 as usize / per_token,
            COUNTS[8].1 as usize / per_token,
        )
    }

    /// The size rule an image is planned with, at `limits` image tokens.
    pub fn size_rule(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        SizeRule::new(self.patch, self.merge, limits)
    }

    /// The projector's hidden width: the 2×2 convolution maps a merge group's `merge²·dim` values
    /// to as many channels (`mm.0.weight`'s output count).
    #[must_use]
    pub fn merged(&self) -> usize {
        self.dim * self.merge * self.merge
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        let why = "the SigLIP ViT and its GELU MLP";
        for key in [KEY_HAS_VISION, KEY_GELU] {
            check_true(m, key, why)?;
        }
        let [
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            _image_size,
            _min_pixels,
            _max_pixels,
        ] = counts(m, PROJECTOR_TYPE, COUNTS)?;
        let out_dim = count_multiple(
            m,
            KEY_PROJECTION_DIM,
            TILE_N,
            "the projector's output width is a GEMM tile count: it must be",
        )?;
        let eps = check_eps(
            m,
            EPS,
            "the vision tower's LayerNorm and the projector's RMS norm",
        )?;
        let stats = ChannelStats {
            mean: channels(m, KEY_MEAN, ChannelStats::HALF.mean)?,
            std: channels(m, KEY_STD, ChannelStats::HALF.std)?,
        };
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            out_dim,
            eps,
            stats,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Hparams;
    use crate::arch::header::testing::{Table, expect_refusals};
    use crate::preprocess::ChannelStats;
    use gguf::Value;

    fn triple(v: [f32; 3]) -> Value {
        Value::Array(v.iter().map(|&x| Value::F32(x)).collect())
    }

    /// The keys this module reads, with the values of Hy4-flash's mmproj header.
    pub(crate) fn hy4() -> Table {
        Table(vec![
            ("clip.projector_type", Value::String("hy_v4_flash".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_gelu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(27)),
            ("clip.vision.embedding_length", Value::U32(1152)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(4304)),
            ("clip.vision.patch_size", Value::U32(16)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(3072)),
            ("clip.vision.image_size", Value::U32(2048)),
            ("clip.vision.image_min_pixels", Value::U32(4096)),
            ("clip.vision.image_max_pixels", Value::U32(4_194_304)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
            ("clip.vision.image_mean", triple(ChannelStats::HALF.mean)),
            ("clip.vision.image_std", triple(ChannelStats::HALF.std)),
        ])
    }

    #[test]
    fn hy4_header_reads() {
        let hp = Hparams::from_meta(&hy4()).expect("Hy4-flash's header reads");
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (27, 1152, 16, 4304, 16, 2, 3072)
        );
        assert_eq!(hp.eps, 1e-5);
        assert_eq!(hp.dim / hp.n_head, 72);
        assert_eq!(hp.merged(), 4608);
        assert_eq!(hp.stats, ChannelStats::HALF);
        let limits = hp.default_limits().expect("limits");
        assert_eq!((limits.min(), limits.max()), (4, 4096));
    }

    /// The header prints the epsilon as 9.999999747378752e-06: that f64 is the f32 1e-5.
    #[test]
    fn the_files_epsilon_is_the_f32_one() {
        assert_eq!(9.999_999_747_378_752e-6_f64 as f32, super::EPS);
    }

    /// Every value this crate does not run is refused, and the error names the key and the value
    /// the file holds.
    #[test]
    fn other_values_are_refused_by_key_and_value() {
        let cases = vec![
            (
                hy4().with("clip.projector_type", Value::String("hy_v4_vl".into())),
                "metadata clip.projector_type: is \"hy_v4_vl\";",
            ),
            (
                hy4().with("clip.vision.block_count", Value::U32(26)),
                "metadata clip.vision.block_count: is 26; hy_v4_flash runs 27",
            ),
            (
                hy4().with("clip.vision.embedding_length", Value::U32(1024)),
                "metadata clip.vision.embedding_length: is 1024;",
            ),
            (
                hy4().with("clip.vision.attention.head_count", Value::U32(12)),
                "metadata clip.vision.attention.head_count: is 12;",
            ),
            (
                hy4().with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                hy4().with("clip.vision.patch_size", Value::U32(14)),
                "metadata clip.vision.patch_size: is 14;",
            ),
            (
                hy4().with("clip.vision.spatial_merge_size", Value::U32(3)),
                "metadata clip.vision.spatial_merge_size: is 3;",
            ),
            (
                hy4().with("clip.vision.image_max_pixels", Value::U32(1_048_576)),
                "metadata clip.vision.image_max_pixels: is 1048576;",
            ),
            (
                hy4().with("clip.vision.projection_dim", Value::U32(3000)),
                "metadata clip.vision.projection_dim: is 3000; the projector's output width is a GEMM tile count: it must be a positive multiple of 64",
            ),
            (
                hy4().with("clip.use_gelu", Value::Bool(false)),
                "metadata clip.use_gelu:",
            ),
            (
                hy4().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-6;",
            ),
            (
                hy4().with("clip.vision.image_mean", triple([0.48, 0.46, 0.41])),
                "metadata clip.vision.image_mean:",
            ),
            (
                hy4().with("clip.vision.image_std", Value::U32(1)),
                "metadata clip.vision.image_std:",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }
}
