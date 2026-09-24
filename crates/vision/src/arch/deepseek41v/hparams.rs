//! Every hyperparameter a `deepseek41v` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — the reference config (`inference/config.json`) has a single
//! value for each `vision_*` key — so each key must hold exactly that value, and the error names
//! the key, the value the file holds and the value the reference runs. Values the reference reads
//! but the file does not carry keep the reference's value: the RoPE base (`vision_rope_theta`) is
//! not in the file.

use gguf::{Gguf, Value};

use super::PROJECTOR_TYPE;
use crate::VisionError;
use crate::grid::GridParams;

const ARCHITECTURE: &str = "clip";
const KEY_PROJECTOR: &str = "clip.projector_type";
const KEY_HAS_VISION: &str = "clip.has_vision_encoder";
const KEY_SILU: &str = "clip.use_silu";
const KEY_EPS: &str = "clip.vision.attention.layer_norm_epsilon";
const KEY_MEAN: &str = "clip.vision.image_mean";
const KEY_STD: &str = "clip.vision.image_std";
const KEY_WH_RATIO: &str = "clip.vision.image_max_wh_ratio";

/// `(key, the reference's value, the reference name it comes from)` for the counts.
const COUNTS: [(&str, u64, &str); 9] = [
    ("clip.vision.block_count", 32, "vision_n_layers"),
    ("clip.vision.embedding_length", 1024, "vision_dim"),
    ("clip.vision.attention.head_count", 16, "vision_n_heads"),
    ("clip.vision.feed_forward_length", 2816, "vision_inter_dim"),
    ("clip.vision.patch_size", 14, "vision_patch_size"),
    (
        "clip.vision.projector.scale_factor",
        3,
        "vision_downsample_ratio",
    ),
    ("clip.vision.image_max_tokens", 1024, "vision_max_n_token"),
    ("clip.vision.image_min_pixels", 295_936, "vision_min_pixels"),
    ("clip.vision.projection_dim", 5120, "dim"),
];

/// vision.py `RMSNorm(dim, eps=1e-6)`; torch adds it to an f32 tensor, so it is this f32.
const EPS: f32 = 1e-6;
/// The `(x - 0.5) / 0.5` of `load_image`, per channel.
const MEAN_STD: f32 = 0.5;
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
        match gguf.architecture() {
            Some(ARCHITECTURE) => {}
            other => {
                return Err(VisionError::Metadata {
                    key: "architecture".into(),
                    detail: format!("is {other:?}; an encoder file is \"{ARCHITECTURE}\""),
                });
            }
        }
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
        match need(m, KEY_PROJECTOR)?.as_str() {
            Some(PROJECTOR_TYPE) => {}
            Some(other) => {
                return Err(refusal(
                    KEY_PROJECTOR,
                    format!("is \"{other}\"; this module reads \"{PROJECTOR_TYPE}\""),
                ));
            }
            None => return Err(refusal(KEY_PROJECTOR, "is not a string")),
        }
        for key in [KEY_HAS_VISION, KEY_SILU] {
            match need(m, key)?.as_bool() {
                Some(true) => {}
                other => {
                    return Err(refusal(
                        key,
                        format!("is {other:?}; the ViT and its SiLU-gated MLP need true"),
                    ));
                }
            }
        }
        let mut counts = [0usize; COUNTS.len()];
        for (slot, (key, want, from)) in counts.iter_mut().zip(COUNTS) {
            let v = need(m, key)?;
            let got = v
                .as_unsigned()
                .ok_or_else(|| refusal(key, format!("is {v:?}, not a count")))?;
            if got != want {
                return Err(refusal(
                    key,
                    format!("is {got}; {PROJECTOR_TYPE} runs {want} (config.json {from})"),
                ));
            }
            *slot = usize::try_from(got)
                .map_err(|_| refusal(key, format!("{got} does not fit usize")))?;
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
        ] = counts;
        let eps = need_f32(m, KEY_EPS)?;
        if eps != EPS {
            return Err(refusal(
                KEY_EPS,
                format!("is {eps:e}; vision.py's RMSNorm uses {EPS:e}"),
            ));
        }
        for key in [KEY_MEAN, KEY_STD] {
            let v = need(m, key)?;
            let per_channel = match v {
                Value::Array(a) => a.iter().map(Value::as_f32).collect::<Option<Vec<f32>>>(),
                _ => None,
            };
            if per_channel.as_deref() != Some(&[MEAN_STD; 3]) {
                return Err(refusal(
                    key,
                    format!("is {v:?}; load_image normalizes (x - 0.5) / 0.5 per channel"),
                ));
            }
        }
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

/// Where the keys come from: the file, or a table in the tests.
trait Meta {
    fn value(&self, key: &str) -> Option<&Value>;
}

impl Meta for Gguf {
    fn value(&self, key: &str) -> Option<&Value> {
        Gguf::value(self, key)
    }
}

fn refusal(key: &str, detail: impl Into<String>) -> VisionError {
    VisionError::Metadata {
        key: key.to_string(),
        detail: detail.into(),
    }
}

fn need<'a>(m: &'a impl Meta, key: &str) -> Result<&'a Value, VisionError> {
    m.value(key).ok_or_else(|| refusal(key, "is absent"))
}

fn need_f32(m: &impl Meta, key: &str) -> Result<f32, VisionError> {
    let v = need(m, key)?;
    match v {
        Value::F32(x) => Ok(*x),
        _ => Err(refusal(key, format!("is {v:?}, not an f32"))),
    }
}

#[cfg(test)]
mod tests {
    use super::{Hparams, Meta};
    use gguf::Value;

    /// A deepseek41v header as a key table.
    struct Table(Vec<(&'static str, Value)>);

    impl Meta for Table {
        fn value(&self, key: &str) -> Option<&Value> {
            self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
        }
    }

    /// The keys this module reads, with the values of smalinin's mmproj header.
    fn v41() -> Table {
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

    fn with(key: &'static str, v: Value) -> Table {
        let mut t = v41();
        match t.0.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = v,
            None => t.0.push((key, v)),
        }
        t
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
        let cases = [
            (
                with("clip.projector_type", Value::String("deepseek4v".into())),
                "metadata clip.projector_type: is \"deepseek4v\";",
            ),
            (
                with("clip.vision.block_count", Value::U32(24)),
                "metadata clip.vision.block_count: is 24; deepseek41v runs 32",
            ),
            (
                with("clip.vision.embedding_length", Value::U32(1152)),
                "metadata clip.vision.embedding_length: is 1152;",
            ),
            (
                with("clip.vision.attention.head_count", Value::U32(12)),
                "metadata clip.vision.attention.head_count: is 12;",
            ),
            (
                with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                with("clip.vision.patch_size", Value::U32(16)),
                "metadata clip.vision.patch_size: is 16;",
            ),
            (
                with("clip.vision.projector.scale_factor", Value::U32(2)),
                "metadata clip.vision.projector.scale_factor: is 2;",
            ),
            (
                with("clip.vision.image_max_tokens", Value::U32(384)),
                "metadata clip.vision.image_max_tokens: is 384;",
            ),
            (
                with("clip.vision.image_min_pixels", Value::U32(200_704)),
                "metadata clip.vision.image_min_pixels: is 200704;",
            ),
            (
                with("clip.vision.projection_dim", Value::U32(4096)),
                "metadata clip.vision.projection_dim: is 4096;",
            ),
            (
                with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-5;",
            ),
            (
                with(
                    "clip.vision.image_mean",
                    Value::Array(vec![Value::F32(0.485); 3]),
                ),
                "metadata clip.vision.image_mean: is Array",
            ),
            (
                with("clip.vision.image_max_wh_ratio", Value::U32(2)),
                "metadata clip.vision.image_max_wh_ratio: is U32(2);",
            ),
            (
                with("clip.use_silu", Value::Bool(false)),
                "metadata clip.use_silu: is Some(false);",
            ),
            (
                Table(
                    v41()
                        .0
                        .into_iter()
                        .filter(|(k, _)| *k != "clip.vision.patch_size")
                        .collect(),
                ),
                "metadata clip.vision.patch_size: is absent",
            ),
        ];
        let wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(|(i, (t, want))| match Hparams::from_meta(&t) {
                Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                Err(e) if e.to_string().starts_with(want) => None,
                Err(e) => Some(format!("case {i}: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
