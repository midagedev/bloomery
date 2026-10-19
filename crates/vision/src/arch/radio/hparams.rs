//! Every hyperparameter a `nemotron_omni` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — Bolt's `vision_config` has a single value for each key — so
//! each count must hold exactly that value, and the error names the key, the value the file holds
//! and the config entry the module runs. The image budget is two pixel counts, 1,024 and 13,312
//! patches of 16² pixels: the limits a seat starts from ([`Hparams::default_limits`]). The
//! projection width is the text model's ([`Hparams::check_text_width`]); the other values the file
//! does not carry are the tensors' shapes ([`super::tensors`]).

use gguf::Gguf;

use super::PROJECTOR_TYPE;
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, KEY_MEAN, KEY_STD, Meta, check_architecture, check_eps, check_projector,
    check_true, count_multiple, counts, need, refusal,
};
use crate::preprocess::ChannelStats;
use crate::size::{SizeRule, TokenLimits};

const KEY_GELU: &str = "clip.use_gelu";
const KEY_NEMOTRON_OMNI: &str = "clip.vision.nemotron_omni";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";

/// The tile width of the encoder's GEMM (`gpu_vision`'s `gemm_bf16`): every output width of a
/// linear layer is a multiple of it, the projector's included.
pub const TILE_N: u64 = 64;

/// `(key, the value the module runs, where that value comes from)` for the counts. The projection
/// width is not among them: it is the text model's width ([`Hparams::check_text_width`]).
const COUNTS: [(&str, u64, &str); 9] = [
    (
        "clip.vision.block_count",
        32,
        "config.json vision_config.num_hidden_layers",
    ),
    (
        "clip.vision.embedding_length",
        1280,
        "config.json vision_config.hidden_size",
    ),
    (
        "clip.vision.attention.head_count",
        16,
        "config.json vision_config.num_attention_heads",
    ),
    (
        "clip.vision.feed_forward_length",
        5120,
        "config.json vision_config.hidden_size times mlp_ratio 4.0",
    ),
    (
        "clip.vision.patch_size",
        16,
        "config.json vision_config.patch_size",
    ),
    (
        "clip.vision.projector.scale_factor",
        2,
        "config.json downsample_ratio 0.5",
    ),
    (
        "clip.vision.image_size",
        512,
        "config.json force_image_size",
    ),
    (
        "clip.vision.image_min_pixels",
        262_144,
        "preprocessor_config.json min_num_patches 1024 of 16x16 pixels",
    ),
    (
        "clip.vision.image_max_pixels",
        3_407_872,
        "preprocessor_config.json max_num_patches 13312 of 16x16 pixels",
    ),
];

/// The vision tower's `nn.LayerNorm(eps=1e-6)`, as the f32 the file stores.
const EPS: f32 = 1e-6;

/// config.json `norm_mean` and `norm_std`, as the f32 values the file stores.
const MEAN: [f32; 3] = [0.481_454_67, 0.457_827_5, 0.408_210_72];
const STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

/// The hyperparameters of one `nemotron_omni` file.
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
    /// LayerNorm epsilon.
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

    /// The file pairs with a text model of embedding width `n_embd`: the projector's output rows
    /// are that model's input rows, so any other width is refused by name.
    pub fn check_text_width(&self, n_embd: usize) -> Result<(), VisionError> {
        if self.out_dim == n_embd {
            return Ok(());
        }
        Err(refusal(
            KEY_PROJECTION_DIM,
            format!(
                "is {}; the text model's embedding width is {n_embd}, so this encoder file is \
                 not its own",
                self.out_dim
            ),
        ))
    }

    /// The image-token limits of the file's own pixel budget: 256 and 3,328 merged tokens of
    /// 32² pixels each. A seat that is given no limit starts from these.
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

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        let why = "the RADIO ViT and its GELU MLP";
        for key in [KEY_HAS_VISION, KEY_GELU, KEY_NEMOTRON_OMNI] {
            check_true(m, key, why)?;
        }
        let [
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            _warm_up,
            _min_pixels,
            _max_pixels,
        ] = counts(m, PROJECTOR_TYPE, COUNTS)?;
        let out_dim = count_multiple(
            m,
            KEY_PROJECTION_DIM,
            TILE_N,
            "the projector's output width is a GEMM tile count: it must be",
        )?;
        let eps = check_eps(m, EPS, "the vision tower's LayerNorm")?;
        let stats = ChannelStats {
            mean: channels(m, KEY_MEAN, MEAN)?,
            std: channels(m, KEY_STD, STD)?,
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

/// A per-channel key: three f32 values, each equal to the config's.
fn channels(m: &impl Meta, key: &str, want: [f32; 3]) -> Result<[f32; 3], VisionError> {
    let v = need(m, key)?;
    let got = match v {
        gguf::Value::Array(a) => a
            .iter()
            .map(gguf::Value::as_f32)
            .collect::<Option<Vec<f32>>>(),
        _ => None,
    };
    if got.as_deref() != Some(&want[..]) {
        return Err(refusal(
            key,
            format!("is {v:?}; the preprocessing normalizes with config.json's {want:?}"),
        ));
    }
    Ok(want)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Hparams;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    fn triple(v: [f32; 3]) -> Value {
        Value::Array(v.iter().map(|&x| Value::F32(x)).collect())
    }

    /// The keys this module reads, with the values of Bolt's mmproj header.
    pub(crate) fn bolt() -> Table {
        Table(vec![
            ("clip.projector_type", Value::String("nemotron_omni".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_gelu", Value::Bool(true)),
            ("clip.vision.nemotron_omni", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(32)),
            ("clip.vision.embedding_length", Value::U32(1280)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(5120)),
            ("clip.vision.patch_size", Value::U32(16)),
            ("clip.vision.projector.scale_factor", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(4096)),
            ("clip.vision.image_size", Value::U32(512)),
            ("clip.vision.image_min_pixels", Value::U32(262_144)),
            ("clip.vision.image_max_pixels", Value::U32(3_407_872)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.image_mean", triple(super::MEAN)),
            ("clip.vision.image_std", triple(super::STD)),
        ])
    }

    #[test]
    fn bolt_header_reads() {
        let hp = Hparams::from_meta(&bolt()).expect("Bolt's header reads");
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (32, 1280, 16, 5120, 16, 2, 4096)
        );
        assert_eq!(hp.eps, 1e-6);
        assert_eq!(hp.dim / hp.n_head, 80);
        let limits = hp.default_limits().expect("limits");
        assert_eq!((limits.min(), limits.max()), (256, 3328));
    }

    /// The header prints the epsilon as 9.999999974752427e-07: that f64 is the f32 1e-6.
    #[test]
    fn the_files_epsilon_is_the_f32_one() {
        assert_eq!(9.999_999_974_752_427e-7_f64 as f32, super::EPS);
    }

    /// The statistics are the f32 values the header prints as doubles.
    #[test]
    fn the_files_statistics_are_the_f32_ones() {
        assert_eq!(0.481_454_670_429_229_74_f64 as f32, super::MEAN[0]);
        assert_eq!(0.275_777_101_516_723_63_f64 as f32, super::STD[2]);
        assert_eq!(super::MEAN[0], 0.481_454_66);
        assert_eq!(super::STD[0], 0.268_629_54);
    }

    /// The projector's rows are the text model's input rows: a file whose width is not the
    /// model's is refused by the key and both widths.
    #[test]
    fn the_text_models_width_is_the_check() {
        let hp = Hparams::from_meta(&bolt()).expect("header reads");
        assert_eq!(
            hp.check_text_width(2048).unwrap_err().to_string(),
            "metadata clip.vision.projection_dim: is 4096; the text model's embedding width is \
             2048, so this encoder file is not its own"
        );
        hp.check_text_width(4096).expect("same width");
    }

    /// Every value this crate does not run is refused, and the error names the key and the value
    /// the file holds.
    #[test]
    fn other_values_are_refused_by_key_and_value() {
        let cases = vec![
            (
                bolt().with(
                    "clip.projector_type",
                    Value::String("nemotron_v2_vl".into()),
                ),
                "metadata clip.projector_type: is \"nemotron_v2_vl\";",
            ),
            (
                bolt().with("clip.vision.block_count", Value::U32(24)),
                "metadata clip.vision.block_count: is 24; nemotron_omni runs 32",
            ),
            (
                bolt().with("clip.vision.embedding_length", Value::U32(1024)),
                "metadata clip.vision.embedding_length: is 1024;",
            ),
            (
                bolt().with("clip.vision.attention.head_count", Value::U32(20)),
                "metadata clip.vision.attention.head_count: is 20;",
            ),
            (
                bolt().with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                bolt().with("clip.vision.patch_size", Value::U32(14)),
                "metadata clip.vision.patch_size: is 14;",
            ),
            (
                bolt().with("clip.vision.projector.scale_factor", Value::U32(3)),
                "metadata clip.vision.projector.scale_factor: is 3;",
            ),
            (
                bolt().with("clip.vision.image_max_pixels", Value::U32(1_048_576)),
                "metadata clip.vision.image_max_pixels: is 1048576;",
            ),
            (
                bolt().with("clip.vision.projection_dim", Value::U32(4000)),
                "metadata clip.vision.projection_dim: is 4000; the projector's output width is a GEMM tile count: it must be a positive multiple of 64",
            ),
            (
                bolt().with("clip.use_gelu", Value::Bool(false)),
                "metadata clip.use_gelu:",
            ),
            (
                bolt().without("clip.vision.nemotron_omni"),
                "metadata clip.vision.nemotron_omni: is absent",
            ),
            (
                bolt().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-5;",
            ),
            (
                bolt().with("clip.vision.image_mean", triple([0.5; 3])),
                "metadata clip.vision.image_mean:",
            ),
            (
                bolt().with("clip.vision.image_std", Value::U32(1)),
                "metadata clip.vision.image_std:",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }
}
