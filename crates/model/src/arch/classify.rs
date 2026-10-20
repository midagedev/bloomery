//! The tensor walk every family's classifier shares: the file's tensors, the
//! family's role table, and the refusal that names every tensor no role
//! claims. A family owns its role table — the GGUF names are its
//! architecture's strings — and the counts it reads from its own `Hparams`;
//! the walk is common ([`classify_with`]).

use gguf::Split;

use crate::arch::stems::Stem;
use crate::placement::{ModelTensor, ModelTensors, PlacementError, Role};

/// The model-level counts a family reports, as [`ModelTensors`] takes them:
/// the layer count it reports (a family with next-token layers reports its
/// trunk's), the routed experts and the experts one token uses.
#[derive(Clone, Copy)]
pub(crate) struct Counts {
    pub(crate) layers: usize,
    pub(crate) experts: u64,
    pub(crate) experts_used: u64,
}

/// The role of a model-level tensor every family names alike: the token embedding, and the output
/// norm and matrix (the head); `None` for any other name.
pub(crate) fn model_level_role(name: &str) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => Some((Role::TokenEmbedding, None)),
        "output_norm.weight" | "output.weight" => Some((Role::Head, None)),
        _ => None,
    }
}

/// A `blk.{layer}.{stem}` name as its layer and stem; `None` for any other name and for a layer
/// not below `n_layer`.
pub(crate) fn block_stem(name: &str, n_layer: usize) -> Option<(usize, &str)> {
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < n_layer)?;
    Some((layer, stem))
}

/// The role and layer of `name` in a family whose layers' stems sit in `groups`: the model-level
/// three, else a `blk.{layer}.{stem}` below `n_layer` whose stem a group holds; `None` for any other
/// name.
pub(crate) fn role_in(
    groups: &[&[Stem]],
    name: &str,
    n_layer: usize,
) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    let (layer, stem) = block_stem(name, n_layer)?;
    groups
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| (s.role, Some(layer)))
}

/// Every tensor of `split` with the role `role_of` gives its name, refusing
/// the file with every name no role claims listed ([`PlacementError::Unclassified`]
/// — no catch-all role exists). The token embedding reads one gathered row; a
/// family whose own tables gather rows (an engram or per-layer-embedding
/// table) sets them beside this call, where its architecture's fact lives.
pub(crate) fn classify_with(
    split: &Split,
    role_of: impl Fn(&str) -> Option<(Role, Option<usize>)>,
    counts: Counts,
) -> Result<ModelTensors, PlacementError> {
    let mut tensors = Vec::with_capacity(split.tensor_count());
    let mut unclassified = Vec::new();
    for (shard, t) in split.iter_tensors() {
        let Some((role, layer)) = role_of(&t.name) else {
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
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        });
    }
    if !unclassified.is_empty() {
        return Err(PlacementError::Unclassified {
            names: unclassified,
        });
    }
    Ok(ModelTensors {
        tensors,
        layers: counts.layers,
        experts: counts.experts,
        experts_used: counts.experts_used,
    })
}

#[cfg(test)]
mod tests {
    use super::{Counts, classify_with};
    use crate::arch::synthetic;
    use crate::placement::{PlacementError, Role};

    /// A family's table in miniature: the token embedding, a layer's mixer.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        match name {
            "token_embd.weight" => Some((Role::TokenEmbedding, None)),
            "blk.0.attn_q.weight" => Some((Role::Attention, Some(0))),
            _ => None,
        }
    }

    /// The model-level names are the three the GGUF convention gives every family, and a block name
    /// splits into its layer and stem only below the layer count. Mutant: the head's role on the
    /// token embedding, or the bound `<=`.
    #[test]
    fn the_shared_name_walk_reads_the_gguf_convention() {
        use super::{block_stem, model_level_role};
        assert_eq!(
            model_level_role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        for head in ["output_norm.weight", "output.weight"] {
            assert_eq!(model_level_role(head), Some((Role::Head, None)), "{head}");
        }
        assert_eq!(model_level_role("blk.0.attn_q.weight"), None);
        assert_eq!(model_level_role("rope_freqs.weight"), None);
        assert_eq!(
            block_stem("blk.47.attn_q.weight", 48),
            Some((47, "attn_q.weight"))
        );
        assert_eq!(block_stem("blk.48.attn_q.weight", 48), None);
        assert_eq!(block_stem("blk.x.attn_q.weight", 48), None);
        assert_eq!(block_stem("blk.3", 48), None);
        assert_eq!(block_stem("token_embd.weight", 48), None);
    }

    /// The grouped walk gives a model-level name its role, a stem of any group its role on its
    /// layer, and every other name — a stem no group holds, a layer past the model — none. Mutant:
    /// the layer bound `<=`, or the first group only searched.
    #[test]
    fn the_grouped_walk_finds_a_stem_in_any_group() {
        use super::role_in;
        use crate::arch::stems::req;
        const A: &[crate::arch::stems::Stem] = &[req("attn_q.weight", Role::Attention)];
        const B: &[crate::arch::stems::Stem] = &[req("ffn_gate_inp.weight", Role::Router)];
        let groups = [A, B];
        assert_eq!(
            role_in(&groups, "output.weight", 4),
            Some((Role::Head, None))
        );
        assert_eq!(
            role_in(&groups, "blk.3.attn_q.weight", 4),
            Some((Role::Attention, Some(3)))
        );
        assert_eq!(
            role_in(&groups, "blk.0.ffn_gate_inp.weight", 4),
            Some((Role::Router, Some(0)))
        );
        for name in [
            "blk.4.attn_q.weight",
            "blk.0.attn_k.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role_in(&groups, name, 4), None, "{name}");
        }
    }

    /// A name no role claims fails the file with every such name listed, in
    /// file order: there is no catch-all role, so the refusal is the only
    /// thing that stands between a renamed tensor and a silent miss.
    #[test]
    fn unclaimed_names_fail_the_file_by_name() {
        let path = synthetic::header(
            "classify-unclaimed",
            "qwen3moe",
            &[],
            &[
                "token_embd.weight".to_string(),
                "blk.0.attn_q.weight".to_string(),
                "blk.0.ffn_up.weight".to_string(),
                "rope_freqs.weight".to_string(),
            ],
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = classify_with(
            &split,
            role,
            Counts {
                layers: 1,
                experts: 8,
                experts_used: 2,
            },
        )
        .expect_err("two of the four names have no role");
        let PlacementError::Unclassified { names } = err else {
            panic!("the refusal is Unclassified, got {err}");
        };
        assert_eq!(
            names,
            &[
                "blk.0.ffn_up.weight".to_string(),
                "rope_freqs.weight".to_string()
            ],
            "every unclaimed name, in file order"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The claimed tensors keep their header facts — name, shard, layer, dims,
    /// bytes — the token embedding reads one gathered row, every other role
    /// none, and the counts pass through as given.
    #[test]
    fn the_claimed_tensors_carry_their_header_facts() {
        let path = synthetic::header(
            "classify-claimed",
            "qwen3moe",
            &[],
            &[
                "token_embd.weight".to_string(),
                "blk.0.attn_q.weight".to_string(),
            ],
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let model = classify_with(
            &split,
            role,
            Counts {
                layers: 48,
                experts: 256,
                experts_used: 8,
            },
        )
        .expect("both names have roles");
        let [embd, mixer] = &model.tensors[..] else {
            panic!("two tensors, got {}", model.tensors.len());
        };
        assert_eq!(embd.name, "token_embd.weight");
        assert_eq!(
            (embd.shard, embd.layer, embd.gathered_rows),
            (0, None, Some(1))
        );
        assert_eq!(embd.dims, &[1]);
        assert_eq!(embd.file_bytes, 4);
        assert_eq!(mixer.name, "blk.0.attn_q.weight");
        assert_eq!(
            (mixer.shard, mixer.layer, mixer.gathered_rows),
            (0, Some(0), None)
        );
        assert_eq!(
            (model.layers, model.experts, model.experts_used),
            (48, 256, 8)
        );
        let _ = std::fs::remove_file(&path);
    }
}
