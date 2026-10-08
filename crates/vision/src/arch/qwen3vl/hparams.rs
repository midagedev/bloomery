//! Every hyperparameter a `qwen3vl_merger` encoder file declares, read once from its header, and a
//! refusal for every value this crate does not run.
//!
//! The encoder is one fixed network — Clef Flash's `vision_config` has a single value for each
//! key — so each key must hold exactly that value, and the error names the key, the value the file
//! holds and the config.json entry the module runs. Values the file does not carry are the
//! tensors' shapes ([`super::tensors`]).

use gguf::{Gguf, Value};

use super::PROJECTOR_TYPE;
use super::size::{SizeRule, TokenLimits};
use crate::VisionError;
use crate::arch::header::{
    KEY_HAS_VISION, Meta, check_architecture, check_eps, check_mean_std, check_projector,
    check_true, counts, need, refusal,
};

const KEY_GELU: &str = "clip.use_gelu";
const KEY_DEEPSTACK: &str = "clip.vision.is_deepstack_layers";

/// `(key, the value the module runs, where that value comes from)` for the counts.
const COUNTS: [(&str, u64, &str); 8] = [
    (
        "clip.vision.block_count",
        27,
        "config.json vision_config.depth",
    ),
    (
        "clip.vision.embedding_length",
        1152,
        "config.json vision_config.hidden_size",
    ),
    (
        "clip.vision.attention.head_count",
        16,
        "config.json vision_config.num_heads",
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
        "clip.vision.projection_dim",
        4096,
        "config.json vision_config.out_hidden_size",
    ),
    (
        "clip.vision.image_size",
        768,
        "the converter's warm-up size",
    ),
];

/// The vision tower's `nn.LayerNorm(eps=1e-6)`, as the f32 the file stores.
const EPS: f32 = 1e-6;

/// The hyperparameters of one `qwen3vl_merger` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// ViT blocks.
    pub n_layer: usize,
    /// ViT width.
    pub dim: usize,
    /// Attention heads; a head is `dim / n_head` wide and RoPE turns all of it.
    pub n_head: usize,
    /// The MLP's hidden width.
    pub ff: usize,
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the merger folds `merge`² patches into one token.
    pub merge: usize,
    /// The merger's output width, the text model's embedding width.
    pub out_dim: usize,
    /// LayerNorm epsilon.
    pub eps: f32,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this crate does not run.
    pub fn read(gguf: &Gguf) -> Result<Hparams, VisionError> {
        check_architecture(gguf)?;
        Hparams::from_meta(gguf)
    }

    /// The size rule an image is planned with, at `limits` image tokens.
    pub fn size_rule(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        SizeRule::new(self.patch, self.merge, limits)
    }

    fn from_meta(m: &impl Meta) -> Result<Hparams, VisionError> {
        check_projector(m, PROJECTOR_TYPE)?;
        let why = "the ViT and its GELU MLP";
        for key in [KEY_HAS_VISION, KEY_GELU] {
            check_true(m, key, why)?;
        }
        let [n_layer, dim, n_head, ff, patch, merge, out_dim, _warm_up] =
            counts(m, PROJECTOR_TYPE, COUNTS)?;
        let eps = check_eps(m, EPS, "the vision tower's LayerNorm")?;
        check_mean_std(
            m,
            "the preprocessing normalizes (x - 0.5) / 0.5 per channel",
        )?;
        check_no_deepstack(m, n_layer)?;
        Ok(Hparams {
            n_layer,
            dim,
            n_head,
            ff,
            patch,
            merge,
            out_dim,
            eps,
        })
    }
}

/// `clip.vision.is_deepstack_layers` holds one bool a block, and none is true: a DeepStack tap
/// feeds an intermediate block's rows to the text model, which this crate does not run.
fn check_no_deepstack(m: &impl Meta, n_layer: usize) -> Result<(), VisionError> {
    let v = need(m, KEY_DEEPSTACK)?;
    let Value::Array(layers) = v else {
        return Err(refusal(
            KEY_DEEPSTACK,
            format!("is {v:?}, not an array of bools"),
        ));
    };
    if layers.len() != n_layer {
        return Err(refusal(
            KEY_DEEPSTACK,
            format!("has {} entries; one a block is {n_layer}", layers.len()),
        ));
    }
    for (i, layer) in layers.iter().enumerate() {
        match layer.as_bool() {
            Some(false) => {}
            Some(true) => {
                return Err(refusal(
                    KEY_DEEPSTACK,
                    format!("block {i} is true; this module runs no DeepStack tap"),
                ));
            }
            None => {
                return Err(refusal(
                    KEY_DEEPSTACK,
                    format!("entry {i} is {layer:?}, not a bool"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Hparams;
    use crate::arch::header::testing::{Table, expect_refusals};
    use gguf::Value;

    /// The keys this module reads, with the values of Clef Flash's mmproj header.
    fn clef() -> Table {
        let half = || Value::Array(vec![Value::F32(0.5); 3]);
        Table(vec![
            (
                "clip.projector_type",
                Value::String("qwen3vl_merger".into()),
            ),
            ("clip.has_vision_encoder", Value::Bool(true)),
            ("clip.use_gelu", Value::Bool(true)),
            ("clip.vision.block_count", Value::U32(27)),
            ("clip.vision.embedding_length", Value::U32(1152)),
            ("clip.vision.attention.head_count", Value::U32(16)),
            ("clip.vision.feed_forward_length", Value::U32(4304)),
            ("clip.vision.patch_size", Value::U32(16)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.projection_dim", Value::U32(4096)),
            ("clip.vision.image_size", Value::U32(768)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.image_mean", half()),
            ("clip.vision.image_std", half()),
            (
                "clip.vision.is_deepstack_layers",
                Value::Array(vec![Value::Bool(false); 27]),
            ),
        ])
    }

    /// The deepstack array with block `i` set.
    fn deepstack_at(i: usize) -> Value {
        let mut a = vec![Value::Bool(false); 27];
        a[i] = Value::Bool(true);
        Value::Array(a)
    }

    #[test]
    fn clef_header_reads() {
        let hp = Hparams::from_meta(&clef()).expect("Clef Flash's header reads");
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (27, 1152, 16, 4304, 16, 2, 4096)
        );
        assert_eq!(hp.eps, 1e-6);
        assert_eq!(hp.dim / hp.n_head, 72);
    }

    /// The header prints the epsilon as 9.999999974752427e-07: that f64 is the f32 1e-6.
    #[test]
    fn the_files_epsilon_is_the_f32_one() {
        assert_eq!(9.999999974752427e-07_f64 as f32, super::EPS);
    }

    /// Every value this crate does not run is refused, and the error names the key and the value
    /// the file holds.
    #[test]
    fn other_values_are_refused_by_key_and_value() {
        let cases = vec![
            (
                clef().with(
                    "clip.projector_type",
                    Value::String("qwen2vl_merger".into()),
                ),
                "metadata clip.projector_type: is \"qwen2vl_merger\";",
            ),
            (
                clef().with("clip.projector_type", Value::U32(7)),
                "metadata clip.projector_type: is not a string",
            ),
            (
                clef().with("clip.vision.block_count", Value::U32(24)),
                "metadata clip.vision.block_count: is 24; qwen3vl_merger runs 27",
            ),
            (
                clef().with("clip.vision.embedding_length", Value::U32(1024)),
                "metadata clip.vision.embedding_length: is 1024;",
            ),
            (
                clef().with("clip.vision.attention.head_count", Value::U32(12)),
                "metadata clip.vision.attention.head_count: is 12;",
            ),
            (
                clef().with("clip.vision.feed_forward_length", Value::U32(4096)),
                "metadata clip.vision.feed_forward_length: is 4096;",
            ),
            (
                clef().with("clip.vision.patch_size", Value::U32(14)),
                "metadata clip.vision.patch_size: is 14;",
            ),
            (
                clef().with("clip.vision.spatial_merge_size", Value::U32(3)),
                "metadata clip.vision.spatial_merge_size: is 3;",
            ),
            (
                clef().with("clip.vision.projection_dim", Value::U32(5120)),
                "metadata clip.vision.projection_dim: is 5120;",
            ),
            (
                clef().with("clip.vision.image_size", Value::U32(448)),
                "metadata clip.vision.image_size: is 448;",
            ),
            (
                clef().with("clip.vision.block_count", Value::String("27".into())),
                "metadata clip.vision.block_count: is String(\"27\"), not a count",
            ),
            (
                clef().with("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
                "metadata clip.vision.attention.layer_norm_epsilon: is 1e-5;",
            ),
            (
                clef().with(
                    "clip.vision.image_mean",
                    Value::Array(vec![Value::F32(0.485); 3]),
                ),
                "metadata clip.vision.image_mean: is Array",
            ),
            (
                clef().with(
                    "clip.vision.image_std",
                    Value::Array(vec![Value::F32(0.229); 3]),
                ),
                "metadata clip.vision.image_std: is Array",
            ),
            (
                clef().with("clip.use_gelu", Value::Bool(false)),
                "metadata clip.use_gelu: is Some(false);",
            ),
            (
                clef().with("clip.has_vision_encoder", Value::Bool(false)),
                "metadata clip.has_vision_encoder: is Some(false);",
            ),
            (
                clef().with("clip.vision.is_deepstack_layers", deepstack_at(5)),
                "metadata clip.vision.is_deepstack_layers: block 5 is true;",
            ),
            (
                clef().with("clip.vision.is_deepstack_layers", deepstack_at(26)),
                "metadata clip.vision.is_deepstack_layers: block 26 is true;",
            ),
            (
                clef().with(
                    "clip.vision.is_deepstack_layers",
                    Value::Array(vec![Value::Bool(false); 24]),
                ),
                "metadata clip.vision.is_deepstack_layers: has 24 entries;",
            ),
            (
                clef().with(
                    "clip.vision.is_deepstack_layers",
                    Value::Array(vec![Value::U32(0); 27]),
                ),
                "metadata clip.vision.is_deepstack_layers: entry 0 is U32(0), not a bool",
            ),
            (
                clef().with("clip.vision.is_deepstack_layers", Value::Bool(false)),
                "metadata clip.vision.is_deepstack_layers: is Bool(false), not an array",
            ),
            (
                clef().without("clip.vision.is_deepstack_layers"),
                "metadata clip.vision.is_deepstack_layers: is absent",
            ),
            (
                clef().without("clip.vision.patch_size"),
                "metadata clip.vision.patch_size: is absent",
            ),
            (
                clef().without("clip.vision.attention.layer_norm_epsilon"),
                "metadata clip.vision.attention.layer_norm_epsilon: is absent",
            ),
        ];
        expect_refusals(cases, Hparams::from_meta);
    }
}
