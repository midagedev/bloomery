//! The tensor walk every family's classifier shares: the file's tensors, the
//! family's role table, and the refusal that names every tensor no role
//! claims. A family owns its role table — the GGUF names are its
//! architecture's strings — and the counts it reads from its own `Hparams`;
//! the walk is common ([`classify_with`]).

use gguf::{GgmlType, Split};

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

impl Counts {
    /// The counts of a family that reports `layers` layers, `experts` routed experts and `used` of
    /// them per token.
    pub(crate) const fn new(layers: usize, experts: usize, used: usize) -> Counts {
        Counts {
            layers,
            experts: experts as u64,
            experts_used: used as u64,
        }
    }
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

/// The rope's per-pair frequency factors (llama.cpp's `rope_freqs`), a model-level tensor of
/// `rope.dimension_count / 2` values that only a family whose rope divides by them reads.
pub(crate) const ROPE_FREQS: &str = "rope_freqs.weight";

/// Whether the file carries the rope's factors: `rope_freqs.weight`, one f32 for each of the
/// `rope_dims / 2` pairs, or none. A table of another length or type is refused by name.
pub(crate) fn rope_factors(split: &Split, rope_dims: usize) -> Result<bool, PlacementError> {
    let Some((_, t)) = split.find(ROPE_FREQS) else {
        return Ok(false);
    };
    let pairs = [(rope_dims / 2) as u64];
    if t.dims != pairs {
        return Err(PlacementError::Tensor {
            name: ROPE_FREQS.to_string(),
            detail: format!(
                "has dims {:?}; the rope turns {rope_dims} values of a head, {} pairs, one factor \
                 each ({pairs:?})",
                t.dims,
                rope_dims / 2
            ),
        });
    }
    if t.ty != GgmlType::F32 {
        return Err(PlacementError::Tensor {
            name: ROPE_FREQS.to_string(),
            detail: format!("is {}; the rope's factors are f32", t.ty),
        });
    }
    Ok(true)
}

/// The role of the rope's factors for a family that reads them: [`Role::Unread`], on no layer. The
/// load reads its few values once, by name, into the rope table the layers turn by; no step reads
/// the tensor, and the placement holds no bytes for it. `None` for any other name.
pub(crate) fn rope_freqs_role(name: &str) -> Option<(Role, Option<usize>)> {
    (name == ROPE_FREQS).then_some((Role::Unread, None))
}

/// A `blk.{layer}.{stem}` name as its layer and stem; `None` for any other name and for a layer
/// not below `n_layer`.
pub(crate) fn block_stem(name: &str, n_layer: usize) -> Option<(usize, &str)> {
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < n_layer)?;
    Some((layer, stem))
}

/// The role and layer of `name` in a family whose layer `l` carries the stem groups of
/// `groups_of(l)`: the model-level three, else a `blk.{layer}.{stem}` below `n_layer` whose stem a
/// group of its layer holds; `None` for any other name.
pub(crate) fn role_by_layer<'a, G: IntoIterator<Item = &'a [Stem]>>(
    name: &str,
    n_layer: usize,
    groups_of: impl FnOnce(usize) -> G,
) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    let (layer, stem) = block_stem(name, n_layer)?;
    groups_of(layer)
        .into_iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| (s.role, Some(layer)))
}

/// The role and layer of `name` in a family whose every layer's stems sit in `groups`
/// ([`role_by_layer`]).
pub(crate) fn role_in(
    groups: &[&[Stem]],
    name: &str,
    n_layer: usize,
) -> Option<(Role, Option<usize>)> {
    role_by_layer(name, n_layer, |_| groups.iter().copied())
}

/// The role and layer of a `blk.{layer}.{stem}` name in a family whose layers from `n_trunk` to
/// `n_layer` are next-token layers: a trunk layer's stem takes the role `trunk_role` gives it, and
/// a next-token layer's stem, whether the trunk's or one of its own (`is_nextn`), is
/// [`Role::Unused`] (carried, never loaded). `None` for any other name.
pub(crate) fn block_role_with_nextn(
    name: &str,
    n_layer: usize,
    n_trunk: usize,
    trunk_role: impl Fn(&str) -> Option<Role>,
    is_nextn: impl Fn(&str) -> bool,
) -> Option<(Role, Option<usize>)> {
    let (layer, stem) = block_stem(name, n_layer)?;
    let role = trunk_role(stem);
    if layer < n_trunk {
        return role.map(|r| (r, Some(layer)));
    }
    (role.is_some() || is_nextn(stem)).then_some((Role::Unused, Some(layer)))
}

/// The role and layer of `name` in a family whose trunk stems sit in the `trunk` groups and whose
/// next-token layers carry the stems `nextn` besides: the model-level three, else
/// [`block_role_with_nextn`] over the tables.
pub(crate) fn role_with_nextn(
    name: &str,
    n_layer: usize,
    n_trunk: usize,
    trunk: &[&[Stem]],
    nextn: &[Stem],
) -> Option<(Role, Option<usize>)> {
    model_level_role(name).or_else(|| {
        block_role_with_nextn(
            name,
            n_layer,
            n_trunk,
            |stem| {
                trunk
                    .iter()
                    .flat_map(|g| g.iter())
                    .find(|s| s.name == stem)
                    .map(|s| s.role)
            },
            |stem| nextn.iter().any(|s| s.name == stem),
        )
    })
}

/// The role of a format's tensor scale, `<stem>.scale` beside a `<stem>.weight`
/// of `split` whose type reads one ([`gguf::GgmlType::reads_scale`]: NVFP4, one
/// a tensor or an expert, and bloomery's F8 and NVFP4 rows), which the load
/// reads by name with its tensor: unread, on the tensor's layer. `None` for
/// any other name, and for a scale beside a tensor of another type or one
/// `role_of` does not claim.
fn tensor_scale_role(
    split: &Split,
    name: &str,
    role_of: &impl Fn(&str) -> Option<(Role, Option<usize>)>,
) -> Option<(Role, Option<usize>)> {
    let weight = gguf::scaled_weight(split, name)?;
    role_of(&weight).map(|(_, layer)| (Role::Unread, layer))
}

/// The role of an NVFP4 tensor's activation scale, `<stem>.input_scale` beside an NVFP4
/// `<stem>.weight` (mainline's converter writes ModelOpt's): the load quantizes activations from
/// their own blocks and never reads it, so it is unread, on the tensor's layer. `None` for any other
/// name, and for one beside a tensor of another type or one `role_of` does not claim.
fn input_scale_role(
    split: &Split,
    name: &str,
    role_of: &impl Fn(&str) -> Option<(Role, Option<usize>)>,
) -> Option<(Role, Option<usize>)> {
    let weight = format!("{}.weight", name.strip_suffix(".input_scale")?);
    let (_, t) = split.find(&weight)?;
    if t.ty != gguf::GgmlType::NVFP4 {
        return None;
    }
    role_of(&weight).map(|(_, layer)| (Role::Unread, layer))
}

/// Every tensor of `split` with the role `role_of` gives its name, or a
/// format's tensor scale's or an NVFP4 activation scale's beside its tensor
/// ([`tensor_scale_role`], [`input_scale_role`]), refusing
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
        let Some((role, layer)) = role_of(&t.name)
            .or_else(|| tensor_scale_role(split, &t.name, &role_of))
            .or_else(|| input_scale_role(split, &t.name, &role_of))
        else {
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
        bf16_words: |_| false,
    })
}

/// [`classify_with`] for a family whose layer `l` carries the stem groups of `groups_of(l)`, below
/// the `counts.layers` the family reports ([`role_by_layer`]).
pub(crate) fn classify_layers<'a, G: IntoIterator<Item = &'a [Stem]>>(
    split: &Split,
    counts: Counts,
    groups_of: impl Fn(usize) -> G,
) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role_by_layer(name, counts.layers, &groups_of),
        counts,
    )
}

/// [`classify_with`] for a family whose every layer's stems sit in `groups` ([`role_in`]).
pub(crate) fn classify_groups(
    split: &Split,
    groups: &[&[Stem]],
    counts: Counts,
) -> Result<ModelTensors, PlacementError> {
    classify_layers(split, counts, |_| groups.iter().copied())
}

/// What a family's flat stem groups hold, for its test: each stem has its own role on a layer, no
/// group lists a stem twice, and a stem two groups share names one role in both.
#[cfg(test)]
pub(crate) fn assert_one_role_per_stem(groups: &[&[Stem]]) {
    for (g, group) in groups.iter().enumerate() {
        for (i, s) in group.iter().enumerate() {
            assert_eq!(
                role_in(groups, &format!("blk.0.{}", s.name), 1),
                Some((s.role, Some(0))),
                "{}",
                s.name
            );
            assert!(
                !group[..i].iter().any(|t| t.name == s.name),
                "{} twice",
                s.name
            );
            for other in &groups[g + 1..] {
                for t in *other {
                    assert!(
                        t.name != s.name || t.role == s.role,
                        "{} in two roles",
                        s.name
                    );
                }
            }
        }
    }
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
            super::rope_freqs_role("rope_freqs.weight"),
            Some((Role::Unread, None))
        );
        for name in ["rope_freqs", "blk.0.rope_freqs.weight", "token_embd.weight"] {
            assert_eq!(super::rope_freqs_role(name), None, "{name}");
        }
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

    /// A layer's own groups decide its stems: a stem another layer's group holds has no role here,
    /// and a layer past the bound has none at all. Mutant: the groups of layer 0 for every layer,
    /// or the bound `<=`.
    #[test]
    fn a_layers_own_groups_decide_its_stems() {
        use super::role_by_layer;
        use crate::arch::stems::{Stem, req};
        const A: &[Stem] = &[req("attn_q.weight", Role::Attention)];
        const B: &[Stem] = &[req("ffn_gate_inp.weight", Role::Router)];
        let of = |l: usize| [if l.is_multiple_of(2) { A } else { B }];
        let role = |name: &str| role_by_layer(name, 4, of);
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(
            role("blk.2.attn_q.weight"),
            Some((Role::Attention, Some(2)))
        );
        assert_eq!(
            role("blk.3.ffn_gate_inp.weight"),
            Some((Role::Router, Some(3)))
        );
        for name in [
            "blk.1.attn_q.weight",
            "blk.2.ffn_gate_inp.weight",
            "blk.4.attn_q.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// A next-token layer carries every stem of the trunk and of its own as unused, a trunk layer
    /// only the trunk's, and a stem in neither is claimed by no layer. Mutant: the trunk bound
    /// `<=`, or a next-token stem given a role on a trunk layer.
    #[test]
    fn a_next_token_layer_carries_every_stem_unused() {
        use super::role_with_nextn;
        use crate::arch::stems::{Stem, req};
        const TRUNK: &[Stem] = &[req("attn_q.weight", Role::Attention)];
        const FFN: &[Stem] = &[req("ffn_gate_inp.weight", Role::Router)];
        const NEXTN: &[Stem] = &[req("nextn.eh_proj.weight", Role::Unused)];
        let role = |name: &str| role_with_nextn(name, 6, 4, &[TRUNK, FFN], NEXTN);
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.3.attn_q.weight"),
            Some((Role::Attention, Some(3)))
        );
        assert_eq!(
            role("blk.3.ffn_gate_inp.weight"),
            Some((Role::Router, Some(3)))
        );
        for (name, layer) in [
            ("blk.4.attn_q.weight", 4),
            ("blk.5.ffn_gate_inp.weight", 5),
            ("blk.4.nextn.eh_proj.weight", 4),
        ] {
            assert_eq!(role(name), Some((Role::Unused, Some(layer))), "{name}");
        }
        for name in [
            "blk.3.nextn.eh_proj.weight",
            "blk.4.attn_k.weight",
            "blk.6.attn_q.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// The group walk takes each layer's own groups below the family's layer count: a stem the
    /// file puts on a layer that does not carry it, and a layer past the count, fail the file by
    /// name. Mutant: the layer count as the bound `<=`, or every layer given every group.
    #[test]
    fn the_layer_walk_refuses_a_stem_on_a_layer_that_does_not_carry_it() {
        use super::{classify_groups, classify_layers};
        use crate::arch::stems::{Stem, req};
        const A: &[Stem] = &[req("attn_q.weight", Role::Attention)];
        const B: &[Stem] = &[req("ffn_gate_inp.weight", Role::Router)];
        let path = synthetic::header(
            "classify-layers",
            "qwen3moe",
            &[],
            &[
                "token_embd.weight".to_string(),
                "blk.0.attn_q.weight".to_string(),
                "blk.1.ffn_gate_inp.weight".to_string(),
                "blk.1.attn_q.weight".to_string(),
                "blk.2.attn_q.weight".to_string(),
            ],
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let counts = Counts::new(2, 8, 2);
        let err = classify_layers(&split, counts, |l| [if l == 0 { A } else { B }])
            .expect_err("two names no layer's groups claim");
        let PlacementError::Unclassified { names } = err else {
            panic!("the refusal is Unclassified, got {err}");
        };
        assert_eq!(names, ["blk.1.attn_q.weight", "blk.2.attn_q.weight"]);
        let err = classify_groups(&split, &[A], counts).expect_err("the gate is no group's");
        let PlacementError::Unclassified { names } = err else {
            panic!("the refusal is Unclassified, got {err}");
        };
        assert_eq!(names, ["blk.1.ffn_gate_inp.weight", "blk.2.attn_q.weight"]);
        let _ = std::fs::remove_file(&path);
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

    /// A format's tensor scale needs no row in a family's table: `<stem>.scale`
    /// beside an NVFP4 `<stem>.weight` the table claims is unread on that
    /// tensor's layer, and so is its `<stem>.input_scale`. One beside a tensor of a
    /// type that reads none (an F32 token embedding) and one with no tensor
    /// beside it fail the file by name. Mutants: no scale rule, no input-scale
    /// rule, or one that skips the type.
    #[test]
    fn a_tensor_scale_takes_its_tensors_layer() {
        use crate::arch::synthetic::{Ty, header_typed};

        let typed = |extra: &[&str]| {
            let mut t = vec![
                ("token_embd.weight".to_string(), vec![1], Ty::F32),
                ("blk.0.attn_q.weight".to_string(), vec![64, 1], Ty::Nvfp4),
                ("blk.0.attn_q.scale".to_string(), vec![1], Ty::F32),
                ("blk.0.attn_q.input_scale".to_string(), vec![1], Ty::F32),
            ];
            t.extend(extra.iter().map(|n| ((*n).to_string(), vec![1], Ty::F32)));
            t
        };
        let counts = Counts {
            layers: 1,
            experts: 8,
            experts_used: 2,
        };
        let path = header_typed("classify-scale", "qwen3moe", &[], &[], &typed(&[]));
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let model = classify_with(&split, role, counts).expect("the scale takes its tensor's role");
        let scale = model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_q.scale")
            .map(|t| (t.role, t.layer));
        assert_eq!(scale, Some((Role::Unread, Some(0))));
        let input = model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_q.input_scale")
            .map(|t| (t.role, t.layer));
        assert_eq!(input, Some((Role::Unread, Some(0))));
        let _ = std::fs::remove_file(&path);

        let path = header_typed(
            "classify-scale-refused",
            "qwen3moe",
            &[],
            &[],
            &typed(&["token_embd.scale", "rope.scale", "token_embd.input_scale"]),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = classify_with(&split, role, counts).expect_err("two scales read by no tensor");
        let PlacementError::Unclassified { names } = err else {
            panic!("the refusal is Unclassified, got {err}");
        };
        assert_eq!(
            names,
            ["token_embd.scale", "rope.scale", "token_embd.input_scale"]
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
        let model =
            classify_with(&split, role, Counts::new(48, 256, 8)).expect("both names have roles");
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
