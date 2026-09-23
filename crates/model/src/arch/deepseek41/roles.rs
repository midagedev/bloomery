//! The role of every tensor in a DeepSeek-V4.1 file, by name: the model-level
//! names first, then the per-layer rules in order, the first match winning —
//! the rule list of `docs/research/v41-placement/derive.py` `role()`. A name
//! no rule matches fails the file with every such name listed; there is no
//! catch-all role.

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// How a rule matches the part of a name after `blk.N.`.
#[derive(Clone, Copy)]
enum Pattern {
    Prefix(&'static str),
    Suffix(&'static str),
    Contains(&'static str),
}

impl Pattern {
    fn matches(self, s: &str) -> bool {
        match self {
            Pattern::Prefix(p) => s.starts_with(p),
            Pattern::Suffix(p) => s.ends_with(p),
            Pattern::Contains(p) => s.contains(p),
        }
    }
}

/// The per-layer rules; order matters (`engram_embd` before `engram_`,
/// `exp_probs_b.` before `exp_probs_b_vl`, the shared expert before `_exps`).
const LAYER_RULES: &[(Pattern, Role)] = &[
    (Pattern::Prefix("engram_embd"), Role::EngramTable),
    (Pattern::Prefix("engram_"), Role::EngramDense),
    (Pattern::Prefix("hc_"), Role::HyperConnection),
    (Pattern::Prefix("ffn_gate_inp"), Role::Router),
    (Pattern::Prefix("exp_probs_b."), Role::Router),
    (Pattern::Prefix("exp_probs_b_vl"), Role::Unread),
    (Pattern::Suffix("_shexp.weight"), Role::SharedExpert),
    (Pattern::Contains("_exps"), Role::RoutedExperts),
    (Pattern::Prefix("ffn_norm"), Role::FfnNorm),
    (Pattern::Prefix("attn"), Role::Attention),
    (Pattern::Prefix("indexer"), Role::Attention),
];

/// A tensor's role and layer by its name; `None` when no rule matches. The
/// per-layer rules apply to `blk.N.` names only.
fn role(name: &str) -> Option<(Role, Option<usize>)> {
    if name.starts_with("token_embd") {
        return Some((Role::TokenEmbedding, None));
    }
    if name.starts_with("output") {
        return Some((Role::Head, None));
    }
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer = layer.parse().ok()?;
    LAYER_RULES
        .iter()
        .find(|(p, _)| p.matches(rest))
        .map(|&(_, r)| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts, and the model's
/// layer and expert counts from [`Hparams`].
pub fn classify(split: &Split) -> Result<ModelTensors, PlacementError> {
    let hp = Hparams::read(split)?;
    let mut tensors = Vec::with_capacity(split.tensor_count());
    let mut unclassified = Vec::new();
    for (shard, t) in split.iter_tensors() {
        let Some((role, layer)) = role(&t.name) else {
            unclassified.push(t.name.clone());
            continue;
        };
        tensors.push(ModelTensor {
            name: t.name.clone(),
            shard,
            layer,
            role,
            ty: t.ty,
            dims: t.dims.clone(),
            file_bytes: t.nbytes,
            gathered_rows: None,
        });
    }
    if !unclassified.is_empty() {
        return Err(PlacementError::Unclassified {
            names: unclassified,
        });
    }
    for t in &mut tensors {
        t.gathered_rows = gathered_rows(split, t)?;
    }
    Ok(ModelTensors {
        tensors,
        layers: hp.n_layer,
        experts: hp.experts.n_expert as u64,
        experts_used: hp.experts.n_used as u64,
    })
}

/// Rows one token reads from a row-gathered table: one row of the token
/// embedding; of an engram layer's table, as many rows as its `engram_wkv`
/// takes in — that tensor's input width over the table's row width.
fn gathered_rows(split: &Split, t: &ModelTensor) -> Result<Option<u64>, PlacementError> {
    let refuse = |detail: String| PlacementError::Tensor {
        name: t.name.clone(),
        detail,
    };
    match (t.role, t.layer) {
        (Role::TokenEmbedding, _) => Ok(Some(1)),
        (Role::EngramTable, Some(layer)) => {
            let wkv = names::engram_wkv(layer);
            let Some((_, w)) = split.find(&wkv) else {
                return Err(refuse(format!("its layer has no {wkv}")));
            };
            let input = w.dims.first().copied().unwrap_or(0);
            let width = t.dims.first().copied().unwrap_or(0);
            if width == 0 || !input.is_multiple_of(width) {
                return Err(refuse(format!(
                    "rows of {width} values do not tile {wkv}'s input of {input}"
                )));
            }
            Ok(Some(input / width))
        }
        (Role::EngramTable, None) => Err(refuse("an engram table outside a layer".to_string())),
        _ => Ok(None),
    }
}
