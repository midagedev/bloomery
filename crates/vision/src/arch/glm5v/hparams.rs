//! Every hyperparameter a `glm5v` encoder file declares, read once from its header, and a refusal
//! for every value this crate does not run.
//!
//! The encoder is one fixed network — GLM-5.3-Flash's mmproj has a single value for each key — so
//! each key must hold exactly that value, and the error names the key, the value the file holds and
//! where the module's value comes from. The image budget is two pixel counts (16 and 8000 merged
//! tokens of 28² pixels each): the limits a seat starts from ([`Hparams::default_limits`]), which
//! its token flags replace ([`Hparams::size_rule_at`]); the other keys the converter writes, such
//! as `image_size`, are the warm-up size mtmd never reads for planning. Values the file does not
//! carry are the tensors' shapes ([`super::tensors`]).

use gguf::{Gguf, Value};

use super::size::SizeRule;
use super::{CLIP, PROJECTOR_TYPE};
use crate::VisionError;
use crate::arch::TokenLimits;
use crate::arch::header::{
    KEY_HAS_VISION, KEY_MEAN, KEY_STD, Meta, check_architecture, check_eps, check_projector,
    check_true, count_multiple, counts, need, need_f32, refusal,
};

const KEY_SILU: &str = "clip.use_silu";
const KEY_GELU: &str = "clip.use_gelu";
const KEY_SWIGLU_CLAMP: &str = "clip.vision.swiglu_clamp";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";

/// The tile width of the encoder's GEMM (`gpu_vision`'s `gemm_bf16`): every output width of a
/// linear layer is a multiple of it, the merger's included.
pub const TILE_N: u64 = 64;

/// `(key, the value the module runs, where that value comes from)` for the counts. The
/// projection width is not among them: it is the text model's width ([`Hparams::check_text_width`]).
const COUNTS: [(&str, u64, &str); 9] = [
    (
        "clip.vision.block_count",
        24,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.embedding_length",
        1024,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.attention.head_count",
        16,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.feed_forward_length",
        4096,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.patch_size",
        14,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.spatial_merge_size",
        2,
        "GLM-5.3-Flash's mmproj header",
    ),
    (
        "clip.vision.image_size",
        448,
        "the converter's warm-up size",
    ),
    (
        "clip.vision.image_min_pixels",
        12_544,
        "16 merged tokens of 28x28 pixels",
    ),
    (
        "clip.vision.image_max_pixels",
        6_272_000,
        "8000 merged tokens of 28x28 pixels",
    ),
];

/// The vision tower's RMS norm epsilon, as the f32 the file stores.
const EPS: f32 = 1e-5;

/// The clamp of the SwiGLU in the tower's and the merger's MLPs: `clip.vision.swiglu_clamp`.
const SWIGLU_CLAMP: f32 = 10.0;

/// The hyperparameters of one `glm5v` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads; a head is `dim / n_head` wide and the vision RoPE turns all of it.
    pub n_head: usize,
    /// The tower's MLP hidden width.
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the merger folds `merge`² patches into one token.
    pub merge: usize,
    /// The merger's width and output width: the embedding width of the text model the file pairs
    /// with.
    pub out_dim: usize,
    /// RMS norm epsilon.
    pub eps: f32,
    /// The SwiGLU's clamp on gate and up.
    pub swiglu_clamp: f32,
    /// The least pixels a canvas holds in the file's own budget (16 merged tokens).
    pub min_pixels: u64,
    /// The most pixels a canvas holds in the file's own budget (8000 merged tokens).
    pub max_pixels: u64,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this crate does not run.
    pub fn read(gguf: &Gguf) -> Result<Hparams, VisionError> {
        check_architecture(gguf)?;
        Hparams::from_meta(gguf)
    }

    /// The file pairs with a text model of embedding width `n_embd`: the merger's output rows
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

    /// The image-token limits of the file's own pixel budget: its pixel counts over the pixels of
    /// one merged token (16 and 8000). A seat that is given no limit starts from these.
    pub fn default_limits(&self) -> Result<TokenLimits, VisionError> {
        let file = SizeRule::new(self.patch, self.merge, self.min_pixels, self.max_pixels)?;
        TokenLimits::new(file.min_tokens(), file.max_tokens())
    }

    /// The size rule an image is planned with when it may take `limits` merged tokens: mtmd's
    /// canvas inside a budget of the limits' pixels, each side replacing the file's own. A budget
    /// that does not fit llama.cpp's 32-bit pixel counts is refused by name.
    pub fn size_rule_at(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        let pixels = |tokens: usize| {
            let side = (self.patch as u64).checked_mul(self.merge as u64)?;
            let per_token = side.checked_mul(side)?;
            (tokens as u64)
                .checked_mul(per_token)
                .filter(|&p| p <= i32::MAX as u64)
        };
        let (Some(min_pixels), Some(max_pixels)) = (pixels(limits.min()), pixels(limits.max()))
        else {
            return Err(VisionError::Limits {
                detail: format!(
                    "{}..={} tokens of patch {} and merge {} do not fit a 32-bit pixel count",
                    limits.min(),
                    limits.max(),
                    self.patch,
                    self.merge
                ),
            });
        };
        SizeRule::new(self.patch, self.merge, min_pixels, max_pixels)
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        check_true(m, KEY_HAS_VISION, "the ViT")?;
        check_true(m, KEY_SILU, "the SwiGLU MLPs")?;
        check_not_gelu(m)?;
        let [
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            _warm_up,
            min_pixels,
            max_pixels,
        ] = counts(m, PROJECTOR_TYPE, COUNTS)?;
        let out_dim = count_multiple(
            m,
            KEY_PROJECTION_DIM,
            TILE_N,
            "the merger's output width is a GEMM tile count: it must be",
        )?;
        let eps = check_eps(m, EPS, "the vision tower's RMS norm")?;
        let swiglu_clamp = need_f32(m, KEY_SWIGLU_CLAMP)?;
        if swiglu_clamp != SWIGLU_CLAMP {
            return Err(refusal(
                KEY_SWIGLU_CLAMP,
                format!(
                    "is {swiglu_clamp:e}; the SwiGLU of {PROJECTOR_TYPE} clamps at {SWIGLU_CLAMP:e}"
                ),
            ));
        }
        check_clip_stats(m)?;
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            out_dim,
            eps,
            swiglu_clamp,
            min_pixels: min_pixels as u64,
            max_pixels: max_pixels as u64,
        })
    }
}

/// `clip.use_gelu` is absent or false: mtmd refuses a file that sets both activations.
fn check_not_gelu(m: &impl Meta) -> Result<(), VisionError> {
    match m.value(KEY_GELU) {
        None => Ok(()),
        Some(v) if v.as_bool() == Some(false) => Ok(()),
        Some(v) => Err(refusal(
            KEY_GELU,
            format!("is {v:?}; the MLPs are SwiGLU, so {KEY_SILU} alone is set"),
        )),
    }
}

/// Both channel keys hold CLIP's three values ([`CLIP`]), the f32 pairs the preprocessing
/// normalizes with.
fn check_clip_stats(m: &impl Meta) -> Result<(), VisionError> {
    for (key, want) in [(KEY_MEAN, CLIP.mean), (KEY_STD, CLIP.std)] {
        let v = need(m, key)?;
        let got = match v {
            Value::Array(a) => a.iter().map(Value::as_f32).collect::<Option<Vec<f32>>>(),
            _ => None,
        };
        if got.as_deref() != Some(&want[..]) {
            return Err(refusal(
                key,
                format!("is {v:?}; the preprocessing normalizes with CLIP's {want:?} per channel"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Hparams;
    use crate::arch::TokenLimits;
    use crate::arch::glm5v::SizeRule;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    /// CLIP's statistics as the f32 bits the file stores them in.
    fn clip(bits: [u32; 3]) -> Value {
        Value::Array(bits.map(|b| Value::F32(f32::from_bits(b))).to_vec())
    }

    /// The keys this module reads, with the values of GLM-5.3-Flash's mmproj header.
    pub(crate) fn glm() -> Table {
        Table(vec![
            ("clip.projector_type", Value::String("glm5v".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_silu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(24)),
            ("clip.vision.embedding_length", Value::U32(1024)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(4096)),
            ("clip.vision.patch_size", Value::U32(14)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(4096)),
            ("clip.vision.image_size", Value::U32(448)),
            ("clip.vision.image_min_pixels", Value::U32(12_544)),
            ("clip.vision.image_max_pixels", Value::U32(6_272_000)),
            ("clip.vision.swiglu_clamp", Value::F32(10.0)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
            (
                "clip.vision.image_mean",
                clip([0x3ef6_813a, 0x3eea_685e, 0x3ed1_00ff]),
            ),
            (
                "clip.vision.image_std",
                clip([0x3e89_89d0, 0x3e85_c974, 0x3e8d_32a8]),
            ),
        ])
    }

    /// GLM-5.3-Flash's hyperparameters, read from [`glm`].
    pub(crate) fn hp() -> Hparams {
        Hparams::from_meta(&glm()).expect("GLM-5.3-Flash's header reads")
    }

    /// The header reads to the network of the file.
    #[test]
    fn glm_header_reads() {
        let hp = Hparams::from_meta(&glm()).expect("GLM-5.3-Flash's header reads");
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (24, 1024, 16, 4096, 14, 2, 4096)
        );
        assert_eq!(hp.dim / hp.n_head, 64);
        assert_eq!((hp.eps, hp.swiglu_clamp), (1e-5, 10.0));
    }

    /// With no limit given a seat starts from the file's own budget, 16 and 8000 tokens of 28²
    /// pixels; a rule at those limits is the file's own rule, and each limit moves its own side of
    /// the budget.
    #[test]
    fn each_limit_replaces_its_side_of_the_files_budget() {
        let hp = hp();
        let default = hp.default_limits().expect("limits");
        assert_eq!((default.min(), default.max()), (16, 8000));
        let at = |min, max| {
            hp.size_rule_at(TokenLimits::new(min, max).expect("limits"))
                .expect("rule")
        };
        assert_eq!(
            hp.size_rule_at(default).expect("rule"),
            SizeRule::new(14, 2, 12_544, 6_272_000).expect("the file's own rule")
        );
        let rule = at(16, 1024);
        assert_eq!((rule.min_tokens(), rule.max_tokens()), (16, 1024));
        let rule = at(100, 8000);
        assert_eq!((rule.min_tokens(), rule.max_tokens()), (100, 8000));
        let rule = at(1, 1);
        assert_eq!((rule.min_tokens(), rule.max_tokens()), (1, 1));
    }

    /// A budget past llama.cpp's 32-bit pixel count is refused by name: 2,000,000 tokens of 784
    /// pixels are 1,568,000,000, under 2^31 - 1, and 3,000,000 are 2,352,000,000, over it.
    #[test]
    fn a_budget_past_32_bits_is_refused_by_name() {
        let hp = hp();
        let rule = hp
            .size_rule_at(TokenLimits::new(16, 2_000_000).expect("limits"))
            .expect("fits");
        assert_eq!(rule.max_tokens(), 2_000_000);
        for max in [3_000_000, usize::MAX] {
            let limits = TokenLimits::new(16, max).expect("limits");
            assert_eq!(
                hp.size_rule_at(limits).unwrap_err().to_string(),
                format!(
                    "image size limits: 16..={max} tokens of patch 14 and merge 2 do not fit a \
                     32-bit pixel count"
                )
            );
        }
    }

    /// The header prints the epsilon as 9.999999747378752e-06: that f64 is the f32 1e-5.
    #[test]
    fn the_files_epsilon_is_the_f32_one() {
        assert_eq!(9.999_999_747_378_752e-6_f64 as f32, super::EPS);
    }

    /// The constants are the f32 bits of the header's CLIP values (printed there as f64).
    #[test]
    fn clip_constants_are_the_files_f32_values() {
        let bits = |a: [f32; 3]| a.map(f32::to_bits);
        assert_eq!(
            bits(crate::arch::glm5v::CLIP.mean),
            [0x3ef6_813a, 0x3eea_685e, 0x3ed1_00ff]
        );
        assert_eq!(
            bits(crate::arch::glm5v::CLIP.std),
            [0x3e89_89d0, 0x3e85_c974, 0x3e8d_32a8]
        );
    }

    /// The merger's rows are the text model's input rows: another width is refused by the key and
    /// both widths, and the model's own width passes.
    #[test]
    fn the_text_models_width_is_the_check() {
        let hp = Hparams::from_meta(&glm()).expect("header reads");
        assert_eq!(
            hp.check_text_width(2048).unwrap_err().to_string(),
            "metadata clip.vision.projection_dim: is 4096; the text model's embedding width is \
             2048, so this encoder file is not its own"
        );
        hp.check_text_width(4096).expect("same width");
    }

    /// Every value this crate does not run, and every key the file lacks, is refused by name.
    #[test]
    fn other_values_and_missing_keys_are_refused_by_key() {
        let cases = vec![
            (
                glm().with("clip.projector_type", Value::String("glm4v".into())),
                "metadata clip.projector_type: is \"glm4v\"; this module reads \"glm5v\"",
            ),
            (
                glm().with("clip.vision.block_count", Value::U32(27)),
                "metadata clip.vision.block_count: is 27; glm5v runs 24",
            ),
            (
                glm().with("clip.vision.embedding_length", Value::U32(1152)),
                "metadata clip.vision.embedding_length: is 1152;",
            ),
            (
                glm().with("clip.vision.patch_size", Value::U32(16)),
                "metadata clip.vision.patch_size: is 16;",
            ),
            (
                glm().with("clip.vision.image_min_pixels", Value::U32(8 * 784)),
                "metadata clip.vision.image_min_pixels: is 6272; glm5v runs 12544",
            ),
            (
                glm().with("clip.vision.image_max_pixels", Value::U32(4096 * 784)),
                "metadata clip.vision.image_max_pixels: is 3211264;",
            ),
            (
                glm().without("clip.vision.image_max_pixels"),
                "metadata clip.vision.image_max_pixels: is absent",
            ),
            (
                glm().without("clip.vision.image_min_pixels"),
                "metadata clip.vision.image_min_pixels: is absent",
            ),
            (
                glm().without("clip.vision.swiglu_clamp"),
                "metadata clip.vision.swiglu_clamp: is absent",
            ),
            (
                glm().with("clip.vision.swiglu_clamp", Value::F32(7.0)),
                "metadata clip.vision.swiglu_clamp: is 7e0; the SwiGLU of glm5v clamps at 1e1",
            ),
            (
                glm().with("clip.vision.swiglu_clamp", Value::U32(10)),
                "metadata clip.vision.swiglu_clamp: is U32(10), not an f32",
            ),
            (
                glm().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-6;",
            ),
            (
                glm().with("clip.vision.projection_dim", Value::U32(4000)),
                "metadata clip.vision.projection_dim: is 4000; the merger's output width is a GEMM tile count: it must be a positive multiple of 64",
            ),
            (
                glm().without("clip.vision.projection_dim"),
                "metadata clip.vision.projection_dim: is absent",
            ),
            (
                glm().with("clip.use_silu", Value::Bool(false)),
                "metadata clip.use_silu: is Some(false);",
            ),
            (
                glm().without("clip.use_silu"),
                "metadata clip.use_silu: is absent",
            ),
            (
                glm().with("clip.use_gelu", Value::Bool(true)),
                "metadata clip.use_gelu: is Bool(true);",
            ),
            (
                glm().with(
                    "clip.vision.image_mean",
                    Value::Array(vec![Value::F32(0.5); 3]),
                ),
                "metadata clip.vision.image_mean: is Array",
            ),
            (
                glm().with("clip.vision.image_std", Value::F32(0.26)),
                "metadata clip.vision.image_std: is F32(0.26)",
            ),
            (
                glm().without("clip.vision.image_std"),
                "metadata clip.vision.image_std: is absent",
            ),
            (
                glm().without("clip.vision.patch_size"),
                "metadata clip.vision.patch_size: is absent",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }

    /// A `clip.use_gelu` of false beside `use_silu` is what mtmd reads as SwiGLU, so it passes.
    #[test]
    fn a_false_gelu_key_is_no_gelu() {
        Hparams::from_meta(&glm().with("clip.use_gelu", Value::Bool(false))).expect("reads");
    }
}
