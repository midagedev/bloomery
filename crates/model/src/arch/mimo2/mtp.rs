//! A MiMo-V2.6-Flash MTP sidecar read against its target: the first of the
//! three next-token layers it carries (`blk.{block_count − 3}`, 48), as a
//! [`DraftSpec::Mtp`], or a refusal naming the key or the tensor that stops
//! it.
//!
//! The file is ggml-org's own GGUF (`mtp-<checkpoint>-<quant>.gguf`):
//! `block_count` is the target's layers plus three, `nextn_predict_layers` is
//! 3, and it carries no trunk layer, so the target reader (`hparams`) refuses
//! it by its first trunk tensor and this reader holds the keys itself. Two
//! checkpoints of the model (MOPD, RL) share every key and tensor shape but
//! `general.name`, and a head reads its own checkpoint's hidden state, so the
//! name is compared first ([`models::same_checkpoint`]). The lines cited are
//! llama.cpp's `src/models/mimo2.cpp`. The graph norms the target's row
//! *before* its output norm (:233-235), joins it with the token's embedding,
//! runs the layer as a window layer with sinks and a dense FFN (:357-372), and
//! norms the layer's own row before its head by `layer_output_norm`
//! (:378-384). It would read `nextn.embed_tokens`, `nextn.shared_head_head`
//! or `nextn.shared_head_norm` in place of `token_embd`, `output` and
//! `layer_output_norm` where the file carries them, so a file that carries one
//! is refused: the description names the matrices the file's other tensors
//! stand for.

use gguf::{GgmlType, Split, Value};
use models::headpick::HeadSet;
use models::{
    Arch, Borrows, CheckpointMismatch, DraftSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, MtpDraft,
    MtpHeadNorm, MtpInput, MtpSource, Residual, same_checkpoint,
};

use super::place::PlanInputs;
use crate::arch::draft::{EMBEDDING, HEAD, borrows, same, same_token_ids};
use crate::arch::{architecture_is, meta_arr, meta_f32, meta_usize, metadata, n_vocab, spec_u32};
use crate::placement::PlacementError;

/// The next-token layers the sidecar carries: three chained heads, of which
/// the first is built.
const NEXTN_LAYERS: usize = 3;

/// Tensors the MTP graph reads in place of the ones this reader builds the
/// draft over: the tensor, and what it replaces.
const SUBSTITUTES: &[(&str, &str)] = &[
    ("nextn.embed_tokens.weight", "token_embd"),
    ("nextn.shared_head_head.weight", "output"),
    ("nextn.shared_head_norm.weight", "layer_output_norm"),
];

/// The description of the first MTP layer `draft` carries, read against the
/// target `spec` was read from (`target`, for its token ids and its
/// embedding and output matrices). The checkpoint, then the keys are
/// compared with the target's before any tensor is looked at.
pub fn mtp_of(
    draft: &Split,
    target: &Split,
    spec: &ModelSpec,
) -> Result<DraftSpec, PlacementError> {
    let t = Target::of(spec)?;
    if draft.architecture() != Some("mimo2") {
        return Err(architecture_is(&format!(
            "{}; an MTP sidecar of a mimo2 target is mimo2",
            draft.architecture().unwrap_or("<missing>")
        )));
    }
    same_checkpoint(name_of(target), name_of(draft)).map_err(checkpoint_error)?;
    let nextn = meta_usize(draft, "nextn_predict_layers")?;
    if nextn != NEXTN_LAYERS {
        return Err(metadata(
            draft,
            "nextn_predict_layers",
            format!("is {nextn}; the sidecar carries {NEXTN_LAYERS} next-token layers"),
        ));
    }
    let n_main = spec.layers.len();
    let n_layer = meta_usize(draft, "block_count")?;
    if n_layer != n_main + NEXTN_LAYERS {
        return Err(metadata(
            draft,
            "block_count",
            format!(
                "is {n_layer}, not the target's {n_main} layers and the {NEXTN_LAYERS} next-token ones"
            ),
        ));
    }
    let index = n_main;
    let rms_eps = keys_agree(draft, target, spec, &t)?;
    let kv = layer_item(draft, "attention.head_count_kv", n_layer, index)?;
    same(draft, "attention.head_count_kv", kv, t.gqa.kv_heads)?;
    let kind = layer_item(draft, "attention.sliding_window_pattern", n_layer, index)?;
    if kind != 1 {
        return Err(metadata(
            draft,
            "attention.sliding_window_pattern",
            format!("is {kind} at layer {index}; the MTP layer is built as a window layer (1)"),
        ));
    }
    let sinks = layer_tensors(draft, index, spec, &t)?;
    let borrows = borrows(draft)?;
    matrices_agree(draft, target, spec, borrows)?;
    let layer = LayerSpec {
        mixer: Mixer::Gqa(Gqa {
            sinks,
            ..t.gqa.clone()
        }),
        ffn: t.ffn.clone(),
        residual: Residual::Plain,
        extras: Vec::new(),
    };
    Ok(DraftSpec::Mtp(Box::new(MtpDraft {
        source: MtpSource::File {
            first_shard: draft
                .shard_path(0)
                .ok_or_else(|| PlacementError::Metadata {
                    key: "split".to_string(),
                    detail: "has no first shard".to_string(),
                })?
                .to_path_buf(),
            // Every tensor of the file, the two chained heads after the
            // first included.
            bytes: draft.iter_tensors().map(|(_, t)| t.nbytes).sum(),
            borrows,
        },
        layer,
        index: spec_u32("block_count", index)?,
        hidden: spec.hidden,
        vocab: spec.vocab,
        rms_eps,
        hc: None,
        input: MtpInput::PreNorm,
        head_norm: MtpHeadNorm::Rms,
        head_rows: HeadSet::Full,
    })))
}

/// The sidecar's keys equal the target's: the widths, the window, the rope,
/// the value scale, the vocabulary, the special ids and the norm's epsilon,
/// which it returns. Floats are held bit for bit.
fn keys_agree(
    draft: &Split,
    target: &Split,
    spec: &ModelSpec,
    t: &Target<'_>,
) -> Result<f32, PlacementError> {
    for (suffix, target) in [
        ("embedding_length", spec.hidden),
        ("attention.head_count", t.gqa.heads),
        ("attention.key_length", t.gqa.head_dim),
        ("attention.value_length", t.gqa.value_dim),
        ("attention.sliding_window", t.window),
        ("rope.dimension_count", t.gqa.rope.dims),
        ("feed_forward_length", t.ff),
    ] {
        same(draft, suffix, meta_usize(draft, suffix)?, target)?;
    }
    same(draft, "vocab_size", n_vocab(draft)?, spec.vocab)?;
    let rms_eps = meta_f32(draft, "attention.layer_norm_rms_epsilon")?;
    same_bits(
        draft,
        "attention.layer_norm_rms_epsilon",
        rms_eps,
        spec.rms_eps,
    )?;
    same_bits(
        draft,
        "attention.value_scale",
        meta_f32(draft, "attention.value_scale")?,
        t.value_scale,
    )?;
    // The window layers' base: the key where the file has it, else the
    // full layers' (the target reader's own fallback).
    let swa_base = match draft.value(&draft.arch_key("rope.freq_base_swa")) {
        Some(_) => meta_f32(draft, "rope.freq_base_swa")?,
        None => meta_f32(draft, "rope.freq_base")?,
    };
    same_bits(draft, "rope.freq_base_swa", swa_base, t.gqa.rope.base)?;
    same_token_ids(draft, target)?;
    Ok(rms_eps)
}

/// The token embedding and the output matrix are Q8_0 `[hidden, vocab]`, in
/// the file each is read from (`borrows`): the type as well as the shape, as
/// the draft's head and embedding run as Q8_0 planes.
fn matrices_agree(
    draft: &Split,
    target: &Split,
    spec: &ModelSpec,
    borrows: Borrows,
) -> Result<(), PlacementError> {
    let (hidden, vocab) = (u64::from(spec.hidden), u64::from(spec.vocab));
    for (name, borrowed) in [(EMBEDDING, borrows.embedding), (HEAD, borrows.head)] {
        let (file, whose) = if borrowed {
            (target, "the target's")
        } else {
            (draft, "the draft's")
        };
        let found = file.find(name).map(|(_, t)| t);
        let ok = found.is_some_and(|t| t.dims == [hidden, vocab] && t.ty == GgmlType::Q8_0);
        if !ok {
            return Err(PlacementError::Tensor {
                name: name.to_string(),
                detail: format!(
                    "is {} in {whose} file; the MTP layer reads a Q8_0 [{hidden}, {vocab}] matrix",
                    found.map_or("absent".to_string(), |t| format!("{} {:?}", t.ty, t.dims))
                ),
            });
        }
    }
    Ok(())
}

/// Whether the mimo2 set whose first shard is `target` can run a sidecar made
/// from `checkpoint` ([`crate::arch::draft::draft_usable`]'s mimo2 rule): the
/// set describes (`PlanInputs::describe`), and the checkpoint is its own. The
/// sidecar is not fetched yet when this runs; [`mtp_of`] holds the rest of it
/// once it is.
pub fn target_usable(target: &Split, checkpoint: Option<&str>) -> Result<(), PlacementError> {
    PlanInputs::describe(target)?;
    same_checkpoint(name_of(target), checkpoint).map_err(checkpoint_error)
}

/// What the draft is compared with: the target's attention geometry on a
/// window layer, and the dense block of its first layer.
struct Target<'a> {
    gqa: &'a Gqa,
    window: u32,
    value_scale: f32,
    ffn: &'a Ffn,
    ff: u32,
}

impl<'a> Target<'a> {
    fn of(spec: &'a ModelSpec) -> Result<Target<'a>, PlacementError> {
        if spec.arch != Arch::MiMo2 {
            return Err(PlacementError::Metadata {
                key: "target".to_string(),
                detail: format!(
                    "is {}; an MTP sidecar pairs with a mimo2 target",
                    spec.arch.name()
                ),
            });
        }
        let (gqa, window) = spec
            .layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::Gqa(g) => g.window.map(|w| (g, w)),
                _ => None,
            })
            .ok_or_else(|| target_lacks("a window-attention layer"))?;
        let value_scale = gqa
            .value_scale
            .ok_or_else(|| target_lacks("a value scale"))?;
        let (ffn, ff) = spec
            .layers
            .iter()
            .find_map(|l| match &l.ffn {
                Ffn::Dense { ff, .. } => Some((&l.ffn, *ff)),
                _ => None,
            })
            .ok_or_else(|| target_lacks("a dense feed-forward layer"))?;
        Ok(Target {
            gqa,
            window,
            value_scale,
            ffn,
            ff,
        })
    }
}

/// The error naming what the target lacks for an MTP sidecar to be read
/// against it.
fn target_lacks(what: &str) -> PlacementError {
    PlacementError::Metadata {
        key: "target".to_string(),
        detail: format!("has no {what}; an MTP sidecar pairs with a mimo2 target"),
    }
}

/// `split`'s `general.name`; `None` when the key is absent or not a string.
fn name_of(split: &Split) -> Option<&str> {
    split.value("general.name").and_then(Value::as_str)
}

/// A checkpoint mismatch as the metadata error that names `general.name`.
fn checkpoint_error(e: CheckpointMismatch) -> PlacementError {
    PlacementError::Metadata {
        key: "general.name".to_string(),
        detail: e.to_string(),
    }
}

/// Err naming `<architecture>.<suffix>` when the draft's float is not the
/// target's, bit for bit.
fn same_bits(draft: &Split, suffix: &str, value: f32, target: f32) -> Result<(), PlacementError> {
    if value.to_bits() == target.to_bits() {
        return Ok(());
    }
    Err(metadata(
        draft,
        suffix,
        format!("is {value}; the target's is {target}"),
    ))
}

/// Layer `at` of the per-layer count array `<architecture>.<suffix>`, which
/// has one value for each of the `n_layer` layers.
fn layer_item(
    draft: &Split,
    suffix: &str,
    n_layer: usize,
    at: usize,
) -> Result<usize, PlacementError> {
    let items = meta_arr(draft, suffix)?;
    if items.len() != n_layer {
        return Err(metadata(
            draft,
            suffix,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items[at]
        .as_unsigned()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| metadata(draft, suffix, format!("has no count at layer {at}")))
}

/// The draft's tensor `name` at exactly `dims`.
fn tensor_dims(draft: &Split, name: &str, dims: &[u64]) -> Result<(), PlacementError> {
    let (_, t) = draft.find(name).ok_or_else(|| PlacementError::Tensor {
        name: name.to_string(),
        detail: "is not in the draft file".to_string(),
    })?;
    if t.dims == dims {
        return Ok(());
    }
    Err(PlacementError::Tensor {
        name: name.to_string(),
        detail: format!("is {:?}; the MTP layer reads {dims:?}", t.dims),
    })
}

/// The layer `blk.{index}` carries the tensors the MTP graph reads at their
/// dims — the fused qkv, the output projection, the dense FFN, the input's
/// projection and norms and the head norm — and none of the [`SUBSTITUTES`].
/// Whether it carries the sinks: the one optional tensor.
fn layer_tensors(
    draft: &Split,
    index: usize,
    spec: &ModelSpec,
    t: &Target<'_>,
) -> Result<bool, PlacementError> {
    let blk = |stem: &str| format!("blk.{index}.{stem}");
    let e = u64::from(spec.hidden);
    let (heads, kv) = (u64::from(t.gqa.heads), u64::from(t.gqa.kv_heads));
    let (key, value) = (u64::from(t.gqa.head_dim), u64::from(t.gqa.value_dim));
    let ff = u64::from(t.ff);
    for (stem, dims) in [
        ("attn_norm.weight", vec![e]),
        ("attn_qkv.weight", vec![e, heads * key + kv * (key + value)]),
        ("attn_output.weight", vec![heads * value, e]),
        ("ffn_norm.weight", vec![e]),
        ("ffn_gate.weight", vec![e, ff]),
        ("ffn_up.weight", vec![e, ff]),
        ("ffn_down.weight", vec![ff, e]),
        ("nextn.eh_proj.weight", vec![2 * e, e]),
        ("nextn.enorm.weight", vec![e]),
        ("nextn.hnorm.weight", vec![e]),
        ("layer_output_norm.weight", vec![e]),
    ] {
        tensor_dims(draft, &blk(stem), &dims)?;
    }
    for (stem, instead) in SUBSTITUTES {
        let name = blk(stem);
        if draft.find(&name).is_some() {
            return Err(PlacementError::Tensor {
                name,
                detail: format!(
                    "is in the draft file, and the MTP graph reads it in place of {instead}; the \
                     description is built over {instead}"
                ),
            });
        }
    }
    let sinks = blk("attn_sinks.weight");
    let has_sinks = draft.find(&sinks).is_some();
    if has_sinks {
        tensor_dims(draft, &sinks, &[heads])?;
    }
    Ok(has_sinks)
}

#[cfg(test)]
mod tests {
    use models::{DraftSpec, Ffn, Mixer, ModelSpec, MtpDraft, MtpHeadNorm, MtpInput, MtpSource};

    use super::super::hparams::tests::{keys, tensors};
    use super::{EMBEDDING, HEAD, mtp_of, target_usable};
    use crate::arch::draft::draft_usable;
    use crate::arch::synthetic::{Ty, V, header_typed};

    const MOPD: &str = "Fixture-MOPD";
    const RL: &str = "Fixture-RL";

    /// The keys the target and the sidecar both carry: the tokenizer's and
    /// the checkpoint's name.
    fn global(name: &'static str) -> Vec<(&'static str, V)> {
        vec![
            ("general.name", V::Str(name)),
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.ggml.eos_token_id", V::U32(1)),
            ("tokenizer.ggml.padding_token_id", V::U32(0)),
        ]
    }

    /// `kv` with `key` set to `v`.
    fn with(kv: Vec<(&'static str, V)>, key: &'static str, v: V) -> Vec<(&'static str, V)> {
        let mut v = Some(v);
        let mut out: Vec<(&'static str, V)> = kv
            .into_iter()
            .filter_map(|(k, old)| {
                if k == key {
                    v.take().map(|v| (k, v))
                } else {
                    Some((k, old))
                }
            })
            .collect();
        if let Some(v) = v {
            out.push((key, v));
        }
        out
    }

    /// The small file of the `hparams` tests as a target of four layers
    /// (the next-token layer dropped), the matrices Q8_0.
    fn target_kv() -> Vec<(&'static str, V)> {
        let kv = with(keys(), "block_count", V::U32(4));
        let kv = kv
            .into_iter()
            .filter(|(k, _)| *k != "nextn_predict_layers")
            .collect();
        let kv = with(kv, "attention.head_count_kv", V::I32s(vec![2, 1, 1, 2]));
        with(
            kv,
            "attention.sliding_window_pattern",
            V::I32s(vec![0, 1, 1, 0]),
        )
    }

    fn typed(ts: Vec<(String, Vec<u64>)>) -> Vec<(String, Vec<u64>, Ty)> {
        ts.into_iter()
            .map(|(n, d)| {
                let ty = if n == EMBEDDING || n == HEAD {
                    Ty::Q8_0
                } else {
                    Ty::F32
                };
                (n, d, ty)
            })
            .collect()
    }

    fn target_tensors() -> Vec<(String, Vec<u64>, Ty)> {
        typed(
            tensors()
                .into_iter()
                .filter(|(n, _)| !n.starts_with("blk.4."))
                .collect(),
        )
    }

    /// The sidecar of that target: seven layers, the last three next-token,
    /// each carrying `blk.4`'s tensors, and the two matrices of its own.
    fn draft_kv() -> Vec<(&'static str, V)> {
        let kv = with(keys(), "block_count", V::U32(7));
        let kv = with(kv, "nextn_predict_layers", V::U32(3));
        let kv = with(
            kv,
            "attention.head_count_kv",
            V::I32s(vec![2, 1, 1, 2, 1, 1, 1]),
        );
        with(
            kv,
            "attention.sliding_window_pattern",
            V::I32s(vec![0, 1, 1, 0, 1, 1, 1]),
        )
    }

    fn draft_tensors() -> Vec<(String, Vec<u64>, Ty)> {
        let layer: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| n.starts_with("blk.4."))
            .collect();
        let mut out = typed(
            tensors()
                .into_iter()
                .filter(|(n, _)| !n.starts_with("blk."))
                .collect(),
        );
        for l in 4..7 {
            out.extend(typed(
                layer
                    .iter()
                    .map(|(n, d)| (n.replacen("blk.4.", &format!("blk.{l}."), 1), d.clone()))
                    .collect(),
            ));
        }
        out
    }

    /// A target and a sidecar header, read.
    struct Case {
        target_name: &'static str,
        draft_name: &'static str,
        kv: Vec<(&'static str, V)>,
        tensors: Vec<(String, Vec<u64>, Ty)>,
    }

    impl Case {
        fn of_the_same_checkpoint() -> Case {
            Case {
                target_name: MOPD,
                draft_name: MOPD,
                kv: draft_kv(),
                tensors: draft_tensors(),
            }
        }

        fn read(&self, tag: &str) -> Result<MtpDraft, String> {
            let tpath = header_typed(
                &format!("{tag}-t"),
                "mimo2",
                &target_kv(),
                &global(self.target_name),
                &target_tensors(),
            );
            let dpath = header_typed(
                &format!("{tag}-d"),
                "mimo2",
                &self.kv,
                &global(self.draft_name),
                &self.tensors,
            );
            let target = gguf::Split::open(&tpath).expect("the synthetic target opens");
            let draft = gguf::Split::open(&dpath).expect("the synthetic draft opens");
            let spec: ModelSpec = super::super::spec::read(&target)
                .expect("the target reads")
                .spec;
            let out = mtp_of(&draft, &target, &spec).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&tpath);
            let _ = std::fs::remove_file(&dpath);
            out.map(|d| match d {
                DraftSpec::Mtp(m) => *m,
                DraftSpec::Block(_) => panic!("an MTP file read as a block draft"),
            })
        }

        fn refused(&self, tag: &str, want: &str) {
            let err = self.read(tag).expect_err("the sidecar is refused");
            assert!(err.contains(want), "want {want:?} in: {err}");
        }
    }

    /// The first next-token layer, as a window layer with sinks over the
    /// target's dense block, fed the rows before the norms.
    #[test]
    fn a_sidecar_of_the_targets_checkpoint_reads() {
        let m = Case::of_the_same_checkpoint()
            .read("mimo2-mtp-ok")
            .expect("the sidecar reads");
        assert_eq!(m.index, 4);
        assert_eq!((m.hidden, m.vocab), (64, 2));
        assert_eq!(m.input, MtpInput::PreNorm);
        assert_eq!(m.head_norm, MtpHeadNorm::Rms);
        assert!(m.hc.is_none());
        let Mixer::Gqa(g) = &m.layer.mixer else {
            panic!("a GQA layer, got {:?}", m.layer.mixer)
        };
        assert_eq!(
            (g.heads, g.kv_heads, g.head_dim, g.value_dim),
            (4, 1, 16, 8)
        );
        assert_eq!((g.window, g.sinks), (Some(8), true));
        assert_eq!(g.value_scale, Some(0.707));
        assert!(
            matches!(m.layer.ffn, Ffn::Dense { ff: 48, .. }),
            "{:?}",
            m.layer.ffn
        );
        let MtpSource::File { borrows, .. } = m.source else {
            panic!("a file source")
        };
        assert!(!borrows.embedding && !borrows.head, "{borrows:?}");
    }

    /// The RL sidecar beside the MOPD target is refused naming both
    /// checkpoints, whatever else it holds.
    #[test]
    fn a_sidecar_of_another_checkpoint_is_refused_naming_both() {
        let mut c = Case::of_the_same_checkpoint();
        c.draft_name = RL;
        c.refused("mimo2-mtp-rl", "general.name");
        c.refused("mimo2-mtp-rl2", MOPD);
        c.refused("mimo2-mtp-rl3", RL);
        // The name is compared before any tensor: a sidecar of the other
        // checkpoint that also lacks its layer is refused on the name.
        c.tensors.retain(|(n, _, _)| !n.starts_with("blk."));
        c.refused("mimo2-mtp-rl4", "general.name");
    }

    #[test]
    fn the_matching_rule_reaches_the_fetch() {
        let tpath = header_typed(
            "mimo2-mtp-usable",
            "mimo2",
            &target_kv(),
            &global(MOPD),
            &target_tensors(),
        );
        let target = gguf::Split::open(&tpath).expect("the synthetic target opens");
        assert!(target_usable(&target, Some(MOPD)).is_ok());
        assert!(draft_usable(&target, Some(MOPD)).is_ok());
        for other in [Some(RL), None] {
            assert!(target_usable(&target, other).is_err(), "{other:?}");
            assert!(draft_usable(&target, other).is_err(), "{other:?}");
        }
        let _ = std::fs::remove_file(&tpath);
    }

    #[test]
    fn one_next_token_layer_or_a_block_count_off_the_target_is_refused() {
        let mut c = Case::of_the_same_checkpoint();
        c.kv = with(c.kv, "nextn_predict_layers", V::U32(1));
        c.refused("mimo2-mtp-n1", "mimo2.nextn_predict_layers: is 1");
        let mut c = Case::of_the_same_checkpoint();
        c.kv = with(c.kv, "block_count", V::U32(6));
        c.refused("mimo2-mtp-bc", "mimo2.block_count: is 6");
    }

    #[test]
    fn a_key_unlike_the_targets_is_refused_by_name() {
        for (key, v) in [
            ("attention.sliding_window", V::U32(16)),
            ("embedding_length", V::U32(32)),
            ("attention.value_scale", V::F32(0.5)),
            ("feed_forward_length", V::U32(96)),
        ] {
            let mut c = Case::of_the_same_checkpoint();
            c.kv = with(c.kv, key, v);
            c.refused("mimo2-mtp-key", &format!("mimo2.{key}"));
        }
    }

    #[test]
    fn a_full_attention_layer_or_other_kv_heads_are_refused() {
        let mut c = Case::of_the_same_checkpoint();
        c.kv = with(
            c.kv,
            "attention.sliding_window_pattern",
            V::I32s(vec![0, 1, 1, 0, 0, 1, 1]),
        );
        c.refused(
            "mimo2-mtp-full",
            "mimo2.attention.sliding_window_pattern: is 0 at layer 4",
        );
        let mut c = Case::of_the_same_checkpoint();
        c.kv = with(
            c.kv,
            "attention.head_count_kv",
            V::I32s(vec![2, 1, 1, 2, 2, 1, 1]),
        );
        c.refused(
            "mimo2-mtp-kv",
            "mimo2.attention.head_count_kv: is 2; the target's is 1",
        );
    }

    #[test]
    fn a_missing_or_misshapen_layer_tensor_is_refused_by_name() {
        let mut c = Case::of_the_same_checkpoint();
        c.tensors
            .retain(|(n, _, _)| n != "blk.4.layer_output_norm.weight");
        c.refused("mimo2-mtp-miss", "blk.4.layer_output_norm.weight");
        let mut c = Case::of_the_same_checkpoint();
        for t in &mut c.tensors {
            if t.0 == "blk.4.nextn.eh_proj.weight" {
                t.1 = vec![64, 64];
            }
        }
        c.refused("mimo2-mtp-shape", "blk.4.nextn.eh_proj.weight");
    }

    #[test]
    fn a_tensor_the_graph_substitutes_is_refused() {
        for stem in [
            "nextn.embed_tokens.weight",
            "nextn.shared_head_head.weight",
            "nextn.shared_head_norm.weight",
        ] {
            let mut c = Case::of_the_same_checkpoint();
            c.tensors.push((format!("blk.4.{stem}"), vec![64], Ty::F32));
            c.refused("mimo2-mtp-subst", &format!("blk.4.{stem}"));
        }
    }

    #[test]
    fn a_matrix_not_q8_0_of_hidden_by_vocab_is_refused() {
        let mut c = Case::of_the_same_checkpoint();
        for t in &mut c.tensors {
            if t.0 == HEAD {
                t.2 = Ty::F32;
            }
        }
        c.refused(
            "mimo2-mtp-head",
            "tensor output.weight: is f32 [64, 2] in the draft's file",
        );
    }

    /// A sidecar that carries neither matrix borrows the target's (the
    /// format's rule: a matrix absent from the file is the target's).
    #[test]
    fn a_sidecar_without_the_matrices_borrows_the_targets() {
        let mut c = Case::of_the_same_checkpoint();
        c.tensors.retain(|(n, _, _)| n != EMBEDDING && n != HEAD);
        let m = c.read("mimo2-mtp-shared").expect("the sidecar reads");
        let MtpSource::File { borrows, .. } = m.source else {
            panic!("a file source")
        };
        assert!(borrows.embedding && borrows.head, "{borrows:?}");
    }
}
