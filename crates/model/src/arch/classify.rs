//! The tensor walk every family's classifier shares: the file's tensors, the
//! family's role table, and the refusal that names every tensor no role
//! claims. A family owns its role table — the GGUF names are its
//! architecture's strings — and the counts it reads from its own `Hparams`;
//! the walk is common ([`classify_with`]).

use gguf::Split;

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
