//! Every hyperparameter a `mimovl` encoder file declares, read once from its header, and a refusal
//! for every value this crate does not run.
//!
//! The encoder is one fixed network — MiMo-V2.6-Flash's mmproj has a single value for each key — so
//! each key must hold exactly that value, and the error names the key, the value the file holds and
//! where the module's value comes from. The attention pattern is the file's per-block table
//! (`clip.vision.wa_pattern_mode`: −1 full, 0 a row-order band with sinks, 1 a column-order band
//! with sinks), held to the one network's. The image budget is the file's own two pixel counts
//! (8 and 8,192 merged tokens of 32² pixels each): the limits a seat starts from
//! ([`Hparams::default_limits`]), which its token flags replace ([`Hparams::size_rule`]). Values the
//! file does not carry are the tensors' shapes ([`super::tensors`]).

use gguf::{Gguf, Value};

use super::{IMAGENET, PROJECTOR_TYPE};
use crate::VisionError;
use crate::arch::TokenLimits;
use crate::arch::header::{
    KEY_HAS_VISION, Meta, check_architecture, check_channel_stats, check_eps, check_projector,
    check_true, count_multiple, counts, need, refusal,
};
use crate::size::SizeRule;

const KEY_SILU: &str = "clip.use_silu";
const KEY_GELU: &str = "clip.use_gelu";
const KEY_PROJECTION_DIM: &str = "clip.vision.projection_dim";
const KEY_PATTERN: &str = "clip.vision.wa_pattern_mode";

/// The tile width of the encoder's GEMM (`gpu_vision`'s `gemm_bf16`): every output width of a
/// linear layer is a multiple of it, the merger's included.
pub const TILE_N: u64 = 64;

/// `(key, the value the module runs, where that value comes from)` for the counts. The projection
/// width is not among them: it is the text model's width ([`Hparams::check_text_width`]).
const COUNTS: [(&str, u64, &str); 11] = [
    (
        "clip.vision.block_count",
        28,
        "config.json vision_config.depth",
    ),
    (
        "clip.vision.embedding_length",
        1280,
        "config.json vision_config.hidden_size",
    ),
    (
        "clip.vision.attention.head_count",
        32,
        "config.json vision_config.num_heads",
    ),
    (
        "clip.vision.attention.head_count_kv",
        8,
        "config.json vision_config.num_key_value_heads",
    ),
    (
        "clip.vision.feed_forward_length",
        4608,
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
        560,
        "the converter's warm-up size",
    ),
    (
        "clip.vision.window_size",
        64,
        "config.json vision_config.visual_token_window_size",
    ),
    (
        "clip.vision.image_min_pixels",
        8_192,
        "preprocessor_config.json min_pixels",
    ),
    (
        "clip.vision.image_max_pixels",
        8_388_608,
        "preprocessor_config.json max_pixels",
    ),
];

/// The RMS norm epsilon of the tower, as the f32 the file stores (`rms_norm_eps` of the modeling
/// file's block).
const EPS: f32 = 1e-6;

/// How one block attends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Every patch attends to every patch, no sink.
    Full,
    /// A band of ±`window` patches in the merge order, with a sink a head.
    Row,
    /// The same band over the sequence with its merge groups in column-major order.
    Col,
}

/// `vit_window_attn_types` of the one network: full blocks 0, 9, 18 and 27 (`fullatt_block_indexes`),
/// the four blocks after each of the first three in row order, then four in column order.
const fn pattern() -> [Mode; 28] {
    let mut modes = [Mode::Full; 28];
    let mut i = 0;
    while i < 27 {
        let phase = i % 9;
        modes[i] = match phase {
            0 => Mode::Full,
            1..=4 => Mode::Row,
            _ => Mode::Col,
        };
        i += 1;
    }
    modes
}

/// The hyperparameters of one `mimovl` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Query heads; a head is [`super::HEAD_DIM`] wide, so the heads' width is not `dim`.
    pub n_head: usize,
    /// Key/value heads: each serves `n_head / n_kv_head` query heads.
    pub n_kv_head: usize,
    /// The MLP's hidden width.
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the merger folds `merge`² patches into one token.
    pub merge: usize,
    /// The band's radius in patches: a windowed block attends to the patches within it.
    pub window: usize,
    /// The merger's output width: the embedding width of the text model the file pairs with.
    pub out_dim: usize,
    /// RMS norm epsilon.
    pub eps: f32,
    /// How each block attends.
    pub modes: Vec<Mode>,
    /// The least pixels an image holds in the file's own budget (8 merged tokens).
    pub min_pixels: u64,
    /// The most pixels an image holds in the file's own budget (8,192 merged tokens).
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
    /// one merged token (8 and 8,192). A seat that is given no limit starts from these.
    pub fn default_limits(&self) -> Result<TokenLimits, VisionError> {
        let token = (self.patch * self.merge).pow(2) as u64;
        if !self.min_pixels.is_multiple_of(token) || !self.max_pixels.is_multiple_of(token) {
            return Err(refusal(
                "clip.vision.image_min_pixels",
                format!(
                    "{} and {} pixels are not whole tokens of {token}",
                    self.min_pixels, self.max_pixels
                ),
            ));
        }
        TokenLimits::new(
            (self.min_pixels / token) as usize,
            (self.max_pixels / token) as usize,
        )
    }

    /// The size rule an image is planned with, at `limits` image tokens.
    pub fn size_rule(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        SizeRule::new(self.patch, self.merge, limits)
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
            n_kv_head,
            ff,
            patch,
            merge,
            _warm_up,
            window,
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
        check_channel_stats(m, &IMAGENET, "ImageNet's")?;
        let modes = read_pattern(m, n_layer)?;
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            n_kv_head,
            ff,
            patch,
            merge,
            window,
            out_dim,
            eps,
            modes,
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

/// The per-block attention table: one entry a block, each −1, 0 or 1, and the one network's.
fn read_pattern(m: &impl Meta, n_layer: usize) -> Result<Vec<Mode>, VisionError> {
    let v = need(m, KEY_PATTERN)?;
    let Value::Array(entries) = v else {
        return Err(refusal(KEY_PATTERN, format!("is {v:?}, not an array")));
    };
    if entries.len() != n_layer {
        return Err(refusal(
            KEY_PATTERN,
            format!("has {} entries; one a block is {n_layer}", entries.len()),
        ));
    }
    let want = pattern();
    let mut modes = Vec::with_capacity(n_layer);
    for (i, entry) in entries.iter().enumerate() {
        let mode = match signed(entry) {
            Some(-1) => Mode::Full,
            Some(0) => Mode::Row,
            Some(1) => Mode::Col,
            _ => {
                return Err(refusal(
                    KEY_PATTERN,
                    format!("entry {i} is {entry:?}, not -1, 0 or 1"),
                ));
            }
        };
        if mode != want[i] {
            return Err(refusal(
                KEY_PATTERN,
                format!(
                    "block {i} is {mode:?}; {PROJECTOR_TYPE} runs {:?} there (config.json \
                     vision_config.vit_window_attn_types)",
                    want[i]
                ),
            ));
        }
        modes.push(mode);
    }
    Ok(modes)
}

/// An integer entry of any stored width.
fn signed(v: &Value) -> Option<i64> {
    match *v {
        Value::I8(x) => Some(i64::from(x)),
        Value::I16(x) => Some(i64::from(x)),
        Value::I32(x) => Some(i64::from(x)),
        Value::I64(x) => Some(x),
        _ => v.as_u64().and_then(|u| i64::try_from(u).ok()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Hparams, Mode};
    use crate::arch::TokenLimits;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    /// ImageNet's statistics as the f32 values the file stores them in.
    fn stats(v: [f32; 3]) -> Value {
        Value::Array(v.map(Value::F32).to_vec())
    }

    /// The `wa_pattern_mode` of the file: −1 on blocks 0, 9, 18 and 27, then four 0 and four 1.
    fn pattern() -> Value {
        let mut p = Vec::new();
        for _ in 0..3 {
            p.extend([-1, 0, 0, 0, 0, 1, 1, 1, 1]);
        }
        p.push(-1);
        Value::Array(p.into_iter().map(Value::I32).collect())
    }

    /// The keys this module reads, with the values of MiMo-V2.6-Flash's mmproj header.
    pub(crate) fn mimo() -> Table {
        Table(vec![
            ("clip.vision.projector_type", Value::String("mimovl".into())),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_silu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(28)),
            ("clip.vision.embedding_length", Value::U32(1280)),
            ("clip.vision.attention.head_count", Value::U32(32)),
            ("clip.vision.attention.head_count_kv", Value::U32(8)),
            ("clip.vision.feed_forward_length", Value::U32(4608)),
            ("clip.vision.patch_size", Value::U32(16)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(4096)),
            ("clip.vision.image_size", Value::U32(560)),
            ("clip.vision.window_size", Value::U32(64)),
            ("clip.vision.image_min_pixels", Value::U32(8_192)),
            ("clip.vision.image_max_pixels", Value::U32(8_388_608)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.wa_pattern_mode", pattern()),
            ("clip.vision.image_mean", stats([0.485, 0.456, 0.406])),
            ("clip.vision.image_std", stats([0.229, 0.224, 0.225])),
        ])
    }

    /// MiMo-V2.6-Flash's hyperparameters, read from [`mimo`].
    pub(crate) fn hp() -> Hparams {
        Hparams::from_meta(&mimo()).expect("MiMo-V2.6-Flash's header reads")
    }

    /// The header reads to the network of the file: the heads' width is not the model width, and
    /// the blocks attend full, in rows or in columns as the pattern says.
    #[test]
    fn mimo_header_reads() {
        let hp = hp();
        assert_eq!(
            (
                hp.n_layer,
                hp.dim,
                hp.n_head,
                hp.n_kv_head,
                hp.ff,
                hp.patch,
                hp.merge,
                hp.window,
                hp.out_dim
            ),
            (28, 1280, 32, 8, 4608, 16, 2, 64, 4096)
        );
        assert_eq!(hp.eps, 1e-6);
        let count = |m| hp.modes.iter().filter(|&&x| x == m).count();
        assert_eq!(
            (count(Mode::Full), count(Mode::Row), count(Mode::Col)),
            (4, 12, 12)
        );
        assert_eq!(hp.modes[0], Mode::Full);
        assert_eq!(hp.modes[1], Mode::Row);
        assert_eq!(hp.modes[5], Mode::Col);
        assert_eq!(hp.modes[9], Mode::Full);
        assert_eq!(hp.modes[27], Mode::Full);
    }

    /// With no limit given a seat starts from the file's own budget, 8 and 8,192 tokens of 32²
    /// pixels, and a rule at those limits plans with them.
    #[test]
    fn the_files_budget_is_8_to_8192_tokens() {
        let hp = hp();
        let default = hp.default_limits().expect("limits");
        assert_eq!((default.min(), default.max()), (8, 8192));
        let rule = hp.size_rule(default).expect("rule");
        assert_eq!(rule.align(), 32);
        // 2600x1800 is 4,680,000 pixels, inside the budget: each side rounds to 32.
        assert_eq!(rule.aligned(2600, 1800).expect("size"), (2592, 1792));
    }

    /// The pairing check names the width the file projects to.
    #[test]
    fn the_text_width_is_checked() {
        let hp = hp();
        hp.check_text_width(4096).expect("MiMo's width");
        assert!(
            hp.check_text_width(2048)
                .unwrap_err()
                .to_string()
                .starts_with("metadata clip.vision.projection_dim: is 4096;")
        );
    }

    /// Every key refused by name: another network's count, a missing key, a wrong pattern entry or
    /// length, another activation, another statistics pair, an epsilon the norm does not use.
    #[test]
    fn every_other_header_is_refused_by_name() {
        let mut shifted = vec![-1, 0, 0, 0, 0, 1, 1, 1, 1];
        shifted.swap(1, 5);
        let entries = |v: Vec<i32>| Value::Array(v.into_iter().map(Value::I32).collect());
        let cases = vec![
            (
                mimo().with("clip.vision.block_count", Value::U32(27)),
                "metadata clip.vision.block_count: is 27;",
            ),
            (
                mimo().with("clip.vision.attention.head_count_kv", Value::U32(16)),
                "metadata clip.vision.attention.head_count_kv: is 16;",
            ),
            (
                mimo().with("clip.vision.window_size", Value::U32(128)),
                "metadata clip.vision.window_size: is 128;",
            ),
            (
                mimo().with("clip.vision.projection_dim", Value::U32(4100)),
                "metadata clip.vision.projection_dim: is 4100;",
            ),
            (
                mimo().without("clip.vision.attention.head_count_kv"),
                "metadata clip.vision.attention.head_count_kv: is absent",
            ),
            (
                mimo().with("clip.vision.image_max_pixels", Value::U32(4_194_304)),
                "metadata clip.vision.image_max_pixels: is 4194304;",
            ),
            (
                mimo().with("clip.vision.wa_pattern_mode", entries(shifted)),
                "metadata clip.vision.wa_pattern_mode: has 9 entries; one a block is 28",
            ),
            (
                mimo().with(
                    "clip.vision.wa_pattern_mode",
                    entries({
                        let mut p = vec![-1, 0, 0, 0, 0, 1, 1, 1, 1];
                        p = [p.clone(), p.clone(), p].concat();
                        p.push(-1);
                        p[1] = 1;
                        p
                    }),
                ),
                "metadata clip.vision.wa_pattern_mode: block 1 is Col; mimovl runs Row there",
            ),
            (
                mimo().with(
                    "clip.vision.wa_pattern_mode",
                    Value::Array(vec![Value::Bool(true); 28]),
                ),
                "metadata clip.vision.wa_pattern_mode: entry 0 is Bool(true), not -1, 0 or 1",
            ),
            (
                mimo().with("clip.vision.wa_pattern_mode", Value::U32(0)),
                "metadata clip.vision.wa_pattern_mode: is U32(0), not an array",
            ),
            (
                mimo().with("clip.use_gelu", Value::Bool(true)),
                "metadata clip.use_gelu: is Bool(true);",
            ),
            (
                mimo().with("clip.use_silu", Value::Bool(false)),
                "metadata clip.use_silu: is Some(false);",
            ),
            (
                mimo().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-5;",
            ),
            (
                mimo().with("clip.vision.image_mean", stats([0.5; 3])),
                "metadata clip.vision.image_mean: is Array",
            ),
            (
                mimo().with("clip.vision.image_std", stats([0.5; 3])),
                "metadata clip.vision.image_std: is Array",
            ),
            (
                mimo().with("clip.vision.projector_type", Value::String("glm5v".into())),
                "metadata clip.projector_type: is \"glm5v\"; this module reads \"mimovl\"",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
        let hp = hp();
        assert!(
            hp.size_rule(TokenLimits::new(1, 4).expect("limits"))
                .is_ok()
        );
    }
}
