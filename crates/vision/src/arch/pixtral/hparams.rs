//! Every hyperparameter a `pixtral` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — Mistral Large 4's mmproj — so each count must hold exactly
//! the header's value, and the error names the key, the value the file holds and the value the
//! module runs. The image budget is mtmd's for this projector, 8 to 1,024 merged tokens
//! ([`Hparams::default_limits`]): the file carries no pixel bounds. The projection width is the
//! text model's embedding width; the other values the file does not carry are the tensors' shapes
//! ([`super::tensors`]).

use gguf::Gguf;

use super::{PROJECTOR_TYPE, ROPE_THETA};
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, KEY_MEAN, KEY_STD, Meta, channels, check_architecture, check_eps,
    check_projector, check_true, count_multiple, counts, need_f32, refusal,
};
use crate::preprocess::{CLIP, ChannelStats};
use crate::size::{SizeRule, TokenLimits};

const KEY_SILU: &str = "clip.use_silu";
const KEY_GELU: &str = "clip.use_gelu";
const KEY_ROPE_BASE: &str = "clip.rope.freq_base";
const KEY_PREPROC_SIZE: &str = "clip.vision.preproc_image_size";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";

/// The tile width of the encoder's GEMM (`gpu_vision`'s `gemm_bf16`): every output width of a
/// linear layer is a multiple of it, the projector's included.
pub const TILE_N: u64 = 64;

/// mtmd's image-token bounds for a `pixtral` file with no `preproc_image_size`
/// (`set_limit_image_tokens(8, 1024)`).
const MIN_TOKENS: usize = 8;
const MAX_TOKENS: usize = 1024;

/// `(key, the value the module runs, where that value comes from)` for the counts. The projection
/// width is not among them: it is the text model's width.
const COUNTS: [(&str, u64, &str); 7] = [
    ("clip.vision.block_count", 30, "the mmproj header"),
    ("clip.vision.embedding_length", 2048, "the mmproj header"),
    ("clip.vision.attention.head_count", 16, "the mmproj header"),
    ("clip.vision.feed_forward_length", 6144, "the mmproj header"),
    ("clip.vision.patch_size", 14, "the mmproj header"),
    ("clip.vision.spatial_merge_size", 2, "the mmproj header"),
    ("clip.vision.image_size", 1540, "the mmproj header"),
];

/// The vision tower's RMS norm epsilon, as the f32 the file stores.
const EPS: f32 = 1e-5;

/// The hyperparameters of one `pixtral` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads; a head is `dim / n_head` wide.
    pub n_head: usize,
    /// The MLP's hidden width (its gate and its up each).
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the merger folds `merge`² patches into one token.
    pub merge: usize,
    /// The projector's output width: the embedding width of the text model the file pairs with.
    pub out_dim: usize,
    /// RMS norm epsilon, the ViT's and the projector's.
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

    /// The image-token limits mtmd plans this projector with: 8 and 1,024 merged tokens of 28²
    /// pixels each. A seat that is given no limit starts from these.
    pub fn default_limits(&self) -> Result<TokenLimits, VisionError> {
        TokenLimits::new(MIN_TOKENS, MAX_TOKENS)
    }

    /// The size rule an image is planned with, at `limits` image tokens.
    pub fn size_rule(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        SizeRule::new(self.patch, self.merge, limits)
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        check_true(m, KEY_HAS_VISION, "the Pixtral ViT")?;
        check_true(m, KEY_SILU, "the ViT's SiLU-gated MLP")?;
        if let Some(v) = m.value(KEY_GELU)
            && v.as_bool() != Some(false)
        {
            return Err(refusal(
                KEY_GELU,
                format!("is {v:?}; the ViT's MLP is SiLU-gated (clip.use_silu), not GELU"),
            ));
        }
        if let Some(v) = m.value(KEY_PREPROC_SIZE) {
            return Err(refusal(
                KEY_PREPROC_SIZE,
                format!(
                    "is {v:?}; mtmd plans such a file by its longest edge, and this crate runs \
                     the dyn_size plan of a file without the key"
                ),
            ));
        }
        if m.value(KEY_ROPE_BASE).is_some() {
            let base = need_f32(m, KEY_ROPE_BASE)?;
            if base != ROPE_THETA {
                return Err(refusal(
                    KEY_ROPE_BASE,
                    format!("is {base:e}; mtmd turns this projector's heads at {ROPE_THETA:e}"),
                ));
            }
        }
        let [n_layer, dim, n_head, ff, patch, merge, _image_size] =
            counts(m, PROJECTOR_TYPE, COUNTS)?;
        let out_dim = count_multiple(
            m,
            KEY_PROJECTION_DIM,
            TILE_N,
            "the projector's output width is a GEMM tile count: it must be",
        )?;
        let eps = check_eps(m, EPS, "the vision tower's RMS norm")?;
        let stats = ChannelStats {
            mean: channels(m, KEY_MEAN, CLIP.mean)?,
            std: channels(m, KEY_STD, CLIP.std)?,
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
    use crate::preprocess::CLIP;
    use gguf::Value;

    fn triple(v: [f32; 3]) -> Value {
        Value::Array(v.iter().map(|&x| Value::F32(x)).collect())
    }

    /// The keys this module reads, with the values of Mistral Large 4's mmproj header.
    pub(crate) fn ml4() -> Table {
        Table(vec![
            ("clip.projector_type", Value::String("pixtral".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_silu", Value::Bool(true)),
            ("clip.rope.freq_base", Value::F32(10_000.0)),
            ("clip.vision.block_count", Value::U32(30)),
            ("clip.vision.embedding_length", Value::U32(2048)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(6144)),
            ("clip.vision.patch_size", Value::U32(14)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(6144)),
            ("clip.vision.image_size", Value::U32(1540)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
            ("clip.vision.image_mean", triple(CLIP.mean)),
            ("clip.vision.image_std", triple(CLIP.std)),
        ])
    }

    #[test]
    fn ml4_header_reads() {
        let hp = Hparams::from_meta(&ml4()).expect("the header reads");
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (30, 2048, 16, 6144, 14, 2, 6144)
        );
        assert_eq!(hp.eps, 1e-5);
        assert_eq!(hp.dim / hp.n_head, 128);
        assert_eq!(hp.stats, CLIP);
        let limits = hp.default_limits().expect("limits");
        assert_eq!((limits.min(), limits.max()), (8, 1024));
        // mtmd's pixel bounds: the token limits times 28² pixels.
        assert_eq!(hp.size_rule(limits).expect("rule").align(), 28);
        // the rotary base is optional: mtmd fixes it
        assert!(Hparams::from_meta(&ml4().without("clip.rope.freq_base")).is_ok());
        assert!(Hparams::from_meta(&ml4().with("clip.use_gelu", Value::Bool(false))).is_ok());
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
                ml4().with("clip.projector_type", Value::String("lightonocr".into())),
                "metadata clip.projector_type: is \"lightonocr\";",
            ),
            (
                ml4().with("clip.vision.block_count", Value::U32(24)),
                "metadata clip.vision.block_count: is 24; pixtral runs 30",
            ),
            (
                ml4().with("clip.vision.embedding_length", Value::U32(1024)),
                "metadata clip.vision.embedding_length: is 1024;",
            ),
            (
                ml4().with("clip.vision.attention.head_count", Value::U32(32)),
                "metadata clip.vision.attention.head_count: is 32;",
            ),
            (
                ml4().with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                ml4().with("clip.vision.patch_size", Value::U32(16)),
                "metadata clip.vision.patch_size: is 16;",
            ),
            (
                ml4().without("clip.vision.spatial_merge_size"),
                "metadata clip.vision.spatial_merge_size: is absent",
            ),
            (
                ml4().with("clip.vision.projection_dim", Value::U32(6000)),
                "metadata clip.vision.projection_dim: is 6000; the projector's output width is a GEMM tile count: it must be a positive multiple of 64",
            ),
            (
                ml4().with("clip.use_silu", Value::Bool(false)),
                "metadata clip.use_silu:",
            ),
            (
                ml4().with("clip.use_gelu", Value::Bool(true)),
                "metadata clip.use_gelu: is Bool(true);",
            ),
            (
                ml4().with("clip.vision.preproc_image_size", Value::U32(1540)),
                "metadata clip.vision.preproc_image_size: is U32(1540); mtmd plans such a file by its longest edge",
            ),
            (
                ml4().with("clip.rope.freq_base", Value::F32(1e6)),
                "metadata clip.rope.freq_base: is 1e6; mtmd turns this projector's heads at 1e4",
            ),
            (
                ml4().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-6;",
            ),
            (
                ml4().with("clip.vision.image_mean", triple([0.5; 3])),
                "metadata clip.vision.image_mean:",
            ),
            (
                ml4().with("clip.vision.image_std", Value::U32(1)),
                "metadata clip.vision.image_std:",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }
}
