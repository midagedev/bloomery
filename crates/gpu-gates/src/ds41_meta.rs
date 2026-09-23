//! The V4.1 model file's rope and norm constants, as the gates of the ops
//! that turn and normalize rows read them from its metadata.

use crate::GateError;
use gguf::{Split, Value};
use model::arch::Arch;

/// The file's rope and norm constants, from its metadata, with its two
/// ropes as `S` (the device crate's `RopeSpec`).
pub struct RopeMeta<S> {
    /// Values a rope turns at the tail of a row: `rope.dimension_count`.
    pub n_dims: usize,
    /// The rope of the heads of a layer with no compressed stream.
    pub window: S,
    /// YaRN: the heads of a layer with a stream, the pooled rows, the index
    /// keys and the indexer query.
    pub yarn: S,
    /// `attention.compress_ratios`, one per layer.
    pub ratios: Vec<u64>,
    /// The norms' epsilon: `attention.layer_norm_rms_epsilon`.
    pub eps: f32,
    /// Rows of a layer's raw window ring: `attention.sliding_window`.
    pub ring: usize,
}

impl<S> RopeMeta<S> {
    /// The constants of `split`, which must be a V4.1 file; `recipe` is the
    /// `just` recipe the error for any other file names. The ropes are built
    /// by `window` and `yarn` — `RopeSpec::window` and `RopeSpec::yarn`, whose
    /// argument orders these are. They come in as arguments because this
    /// crate does not name the V4.1 device crate: named here, it would be
    /// linked, device bundle and all, into every gate built with the
    /// feature, the ones that launch none of its kernels too.
    pub fn read(
        split: &Split,
        recipe: &str,
        window: fn(f32, usize) -> S,
        yarn: fn(f32, f32, i32, f32, f32, usize) -> S,
    ) -> Result<RopeMeta<S>, GateError> {
        let want = Arch::Deepseek41.name();
        if split.architecture() != Some(want) {
            return Err(format!(
                "the model file is {:?}, want {want} — run through `just {recipe}`, \
                 which picks the deepseek41 profile",
                split.architecture()
            )
            .into());
        }
        let f = |s: &str| -> Result<f32, GateError> {
            split
                .arch_get_f32(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let u = |s: &str| -> Result<u64, GateError> {
            split
                .arch_get_u64(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let scaling = split.arch_get_str("rope.scaling.type");
        if scaling != Some("yarn") {
            return Err(format!("rope.scaling.type is {scaling:?}, want \"yarn\"").into());
        }
        let n_dims = usize::try_from(u("rope.dimension_count")?)?;
        let n_ctx_orig = i32::try_from(u("rope.scaling.original_context_length")?)?;
        let key = split.arch_key("attention.compress_ratios");
        let Some(Value::Array(list)) = split.value(&key) else {
            return Err(format!("metadata {key} is absent or not an array").into());
        };
        let ratios = list
            .iter()
            .map(|v| v.as_unsigned().ok_or_else(|| format!("{key} holds {v:?}")))
            .collect::<Result<Vec<u64>, String>>()?;
        Ok(RopeMeta {
            n_dims,
            window: window(f("rope.freq_base")?, n_dims),
            yarn: yarn(
                f("attention.compress_rope_freq_base")?,
                f("rope.scaling.factor")?,
                n_ctx_orig,
                f("rope.scaling.yarn_beta_fast")?,
                f("rope.scaling.yarn_beta_slow")?,
                n_dims,
            ),
            ratios,
            eps: f("attention.layer_norm_rms_epsilon")?,
            ring: usize::try_from(u("attention.sliding_window")?)?,
        })
    }

    /// The rope of layer `l`'s heads: ik's `use_compress_rope` is
    /// `compress_ratios[l] != 0`.
    pub fn head_spec(&self, l: usize) -> Result<(&S, &'static str), GateError> {
        match self.ratios.get(l) {
            Some(0) => Ok((&self.window, "window")),
            Some(_) => Ok((&self.yarn, "yarn")),
            None => Err(format!("layer {l} has no compress ratio").into()),
        }
    }
}
