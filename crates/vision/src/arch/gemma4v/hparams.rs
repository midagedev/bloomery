//! The vision half of the Gemma4 projector header.

use super::{MERGE, PROJECTOR_TYPE};
use crate::VisionError;
use crate::arch::{
    TokenLimits,
    header::{
        KEY_HAS_VISION, KEY_MEAN, KEY_STD, check_architecture, check_eps, check_projector,
        check_true, counts, need, refusal,
    },
};
use crate::size::SizeRule;
use gguf::Gguf;

/// Dimensions of the EmbeddingGemma 2 vision tower.
#[derive(Clone, Debug)]
pub struct Hparams {
    pub n_layer: usize,
    pub dim: usize,
    pub ff: usize,
    pub n_head: usize,
    pub patch: usize,
    pub out_dim: usize,
    pub eps: f32,
    pub pos_rows: usize,
    /// The header chooses tanh GELU; without that key mainline uses Quick GELU.
    pub gelu_tanh: bool,
}

impl Hparams {
    /// Read the vision header and refuse a geometry this tower does not run.
    pub fn read(file: &Gguf) -> Result<Self, VisionError> {
        check_architecture(file)?;
        check_projector(file, PROJECTOR_TYPE)?;
        check_true(file, KEY_HAS_VISION, "the vision tower")?;
        if let Some(v) = file.value("clip.vision.projector.scale_factor")
            && v.as_u64() != Some(MERGE as u64)
        {
            return Err(refusal(
                "clip.vision.projector.scale_factor",
                format!("is {v:?}; gemma4v pools {MERGE} by {MERGE} patches"),
            ));
        }
        let [n_layer, dim, ff, n_head, patch, out_dim] = counts(
            file,
            PROJECTOR_TYPE,
            [
                (
                    "clip.vision.block_count",
                    16,
                    "the EmbeddingGemma 2 projector",
                ),
                (
                    "clip.vision.embedding_length",
                    768,
                    "the EmbeddingGemma 2 projector",
                ),
                (
                    "clip.vision.feed_forward_length",
                    3072,
                    "the EmbeddingGemma 2 projector",
                ),
                (
                    "clip.vision.attention.head_count",
                    12,
                    "the EmbeddingGemma 2 projector",
                ),
                (
                    "clip.vision.patch_size",
                    16,
                    "the EmbeddingGemma 2 projector",
                ),
                (
                    "clip.vision.projection_dim",
                    512,
                    "the EmbeddingGemma 2 projector",
                ),
            ],
        )?;
        for (key, want) in [(KEY_MEAN, 0.0f32), (KEY_STD, 1.0)] {
            let v = need(file, key)?;
            if !matches!(v, gguf::Value::Array(a) if a.len() == 3 && a.iter().all(|x| x.as_f32() == Some(want)))
            {
                return Err(refusal(key, format!("is {v:?}; gemma4v uses [{want}; 3]")));
            }
        }
        let eps = check_eps(file, 1e-6, "the Gemma4 vision RMS norm")?;
        let gelu_tanh = activation(file)?;
        Ok(Self {
            n_layer,
            dim,
            ff,
            n_head,
            patch,
            out_dim,
            eps,
            pos_rows: 10240,
            gelu_tanh,
        })
    }

    /// The mainline image budget when neither flag supplies a bound.
    pub fn default_limits(&self) -> Result<TokenLimits, VisionError> {
        TokenLimits::new(70, 1120)
    }

    /// The shared mtmd dynamic resize rule at `limits`.
    pub fn size_rule(&self, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        SizeRule::new(self.patch, MERGE, limits)
    }
}

fn activation(m: &impl crate::arch::header::Meta) -> Result<bool, VisionError> {
    let flag = |key| match m.value(key) {
        None => Ok(false),
        Some(gguf::Value::Bool(v)) => Ok(*v),
        Some(v) => Err(refusal(key, format!("is {v:?}, not a boolean"))),
    };
    if flag("clip.use_silu")? {
        return Err(refusal("clip.use_silu", "gemma4v does not run SiLU"));
    }
    flag("clip.use_gelu")
}

#[cfg(test)]
mod tests {
    use super::activation;
    use crate::arch::header::testing::Table;
    use gguf::Value;
    #[test]
    fn absent_activation_flags_select_mainline_quick_gelu() {
        let table = Table(vec![]);
        assert!(!activation(&table).unwrap());
        assert!(activation(&Table(vec![]).with("clip.use_gelu", Value::Bool(true))).unwrap());
        assert!(activation(&Table(vec![]).with("clip.use_silu", Value::Bool(true))).is_err());
        assert!(activation(&table.with("clip.use_gelu", Value::U32(1))).is_err());
    }
}
