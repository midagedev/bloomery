//! A qwen4exp MTP draft file read against its target: the one next-token
//! layer it carries, as a [`DraftSpec::Mtp`], or a refusal naming the key or
//! the tensor that stops it.
//!
//! The file is its own GGUF (unsloth's `mtp-…-Q8_0.gguf`): `block_count` is
//! the target's layers plus one, `nextn_predict_layers` is 1, and its only
//! layer is `blk.{block_count − 1}`. The lines cited are ik_llama.cpp's. It
//! loads that layer as full attention whatever `full_attention_interval`
//! says (`src/llama-hparams.cpp:653-656`), and reads a ratio array covering
//! either `block_count` layers or only the main ones, the layer then
//! inheriting the last main layer's (:659-673). A layer with a ratio above 0
//! would select its positions; this reader builds only the dense one. The
//! file uses the target's `token_embd` and `output` where it carries none
//! (`src/llama-spec-features.cpp:242-291`); `nextn_shared_target_tensors`
//! declares that it carries neither.

use gguf::{GgmlType, Split, TensorInfo};
use models::{
    Arch, Borrows, DraftSpec, Ffn, Gqa, HcKind, HeadRows, LayerSpec, Mixer, ModelSpec, MtpDraft,
    MtpHeadNorm, MtpInput, MtpSource, Residual, Rope, RopeMode,
};

use super::hparams::{
    ATTN_STEMS, EXP_LAYER_STEMS, GDN_STEMS, NORM_STEMS, PLE_STEMS, Variant, expert_ff,
    refuse_unless, sections,
};
use super::spec::out_gate;
use crate::arch::{
    architecture_is, meta_arr, meta_f32, meta_usize, metadata, n_vocab, qwen35moe_variant, spec_u32,
};
use crate::placement::PlacementError;

/// The key that declares a file carries neither `token_embd` nor `output`.
const SHARED: &str = "nextn_shared_target_tensors";

/// The token embedding and the output matrix, by name.
const EMBEDDING: &str = "token_embd.weight";
const HEAD: &str = "output.weight";

/// The layer's own stems: the input's norms and projection and the
/// gated-residual head.
const NEXTN_STEMS: &[&str] = &[
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.hc_head_norm.weight",
    "nextn.hc_head_down.weight",
    "nextn.hc_head_up.weight",
];

/// The attention stems a selecting layer reads and a dense one does not: a
/// file may carry them either way.
const INDEXER_STEMS: &[&str] = &[
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
];

/// The description of the MTP layer `draft` carries, read against the
/// target `spec` was read from (`target`, for its token ids and its
/// embedding and output matrices). The keys are compared with the target's
/// before any tensor is looked at.
pub fn mtp_of(
    draft: &Split,
    target: &Split,
    spec: &ModelSpec,
) -> Result<DraftSpec, PlacementError> {
    let t = Target::of(spec)?;
    if qwen35moe_variant(draft)? != Variant::Qwen4Exp {
        return Err(architecture_is(
            "qwen35moe; an MTP draft file of this family is qwen4exp",
        ));
    }
    let nextn = meta_usize(draft, "nextn_predict_layers")?;
    if nextn != 1 {
        return Err(metadata(
            draft,
            "nextn_predict_layers",
            format!("is {nextn}; one next-token layer is built"),
        ));
    }
    let n_main = spec.layers.len();
    let n_layer = meta_usize(draft, "block_count")?;
    if n_layer != n_main + 1 {
        return Err(metadata(
            draft,
            "block_count",
            format!("is {n_layer}, not the target's {n_main} layers and the next-token one"),
        ));
    }
    let index = n_main;
    let hidden = meta_usize(draft, "embedding_length")?;
    same(draft, "embedding_length", hidden, spec.hidden)?;
    let heads = meta_usize(draft, "attention.head_count")?;
    same(draft, "attention.head_count", heads, t.gqa.heads)?;
    let kv_heads = meta_usize(draft, "attention.head_count_kv")?;
    same(draft, "attention.head_count_kv", kv_heads, t.gqa.kv_heads)?;
    let head_dim = meta_usize(draft, "attention.key_length")?;
    same(draft, "attention.key_length", head_dim, t.gqa.head_dim)?;
    if draft
        .value(&draft.arch_key("attention.value_length"))
        .is_some()
    {
        let v = meta_usize(draft, "attention.value_length")?;
        same(draft, "attention.value_length", v, t.gqa.head_dim)?;
    }
    let moe = spec.layers[0]
        .moe()
        .ok_or_else(|| target_lacks("a routed first layer"))?;
    same(
        draft,
        "expert_count",
        meta_usize(draft, "expert_count")?,
        moe.experts,
    )?;
    same(
        draft,
        "expert_used_count",
        meta_usize(draft, "expert_used_count")?,
        moe.top_k,
    )?;
    same(
        draft,
        "expert_feed_forward_length",
        expert_ff(draft)?,
        moe.expert_ff,
    )?;
    let shared = moe.shared.ok_or_else(|| target_lacks("a shared expert"))?;
    same(
        draft,
        "expert_shared_feed_forward_length",
        meta_usize(draft, "expert_shared_feed_forward_length")?,
        shared.ff,
    )?;
    refuse_unless(draft, "expert_gating_func", |v| v.as_u64() == Some(1))?;
    refuse_unless(draft, "expert_weights_norm", |v| v.as_bool() == Some(true))?;
    refuse_unless(draft, "expert_weights_scale", |v| v.as_f32() == Some(1.0))?;
    let hc = spec.hc.ok_or_else(|| target_lacks("hyper-connections"))?;
    let streams = meta_usize(draft, "hyper_connection.count")?;
    same(draft, "hyper_connection.count", streams, hc.streams)?;
    let rank = hc_rank(spec)?;
    same(
        draft,
        "hyper_connection.low_rank",
        meta_usize(draft, "hyper_connection.low_rank")?,
        rank,
    )?;
    same(draft, "vocab_size", n_vocab(draft)?, spec.vocab)?;
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        let (d, t) = (token_id(draft, key)?, token_id(target, key)?);
        if d != t {
            return Err(PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("is {d:?} in the draft file, {t:?} in the target's"),
            });
        }
    }
    let ratio = ratio(draft, n_layer, n_main)?;
    if ratio > 0 {
        return Err(metadata(
            draft,
            "attention.compress_ratios",
            format!(
                "gives the MTP layer {index} a pool of {ratio}: a selecting MTP layer is not built"
            ),
        ));
    }
    layer_tensors(draft, index)?;
    let borrows = borrows(draft)?;
    let (hidden_u, vocab_u) = (u64::from(spec.hidden), u64::from(spec.vocab));
    // The types as well as the shape: the embedding runs as Q8_0 planes, the
    // head as Q8_0 planes or Q6_K rows (`gpu::q6k_ids`), where ik checks the
    // shape alone and clones any type it has kernels for.
    for (name, borrowed, kinds) in [
        (EMBEDDING, borrows.embedding, "Q8_0"),
        (HEAD, borrows.head, "Q8_0 or Q6_K"),
    ] {
        let (file, whose) = if borrowed {
            (target, "the target's")
        } else {
            (draft, "the draft's")
        };
        let found = file.find(name).map(|(_, t)| t);
        let ok = |t: &TensorInfo| {
            t.dims == [hidden_u, vocab_u]
                && (t.ty == GgmlType::Q8_0 || (name == HEAD && t.ty == GgmlType::Q6_K))
        };
        if !found.is_some_and(ok) {
            return Err(PlacementError::Tensor {
                name: name.to_string(),
                detail: format!(
                    "is {} in {whose} file; the MTP layer reads a {kinds} [{hidden_u}, {vocab_u}] \
                     matrix",
                    found.map_or("absent".to_string(), |t| format!("{} {:?}", t.ty, t.dims))
                ),
            });
        }
    }
    let blk = |stem: &str| format!("blk.{index}.{stem}");
    let wide = u64::from(streams_u32(streams)?) * hidden_u;
    for (stem, dims) in [
        ("nextn.eh_proj.weight", vec![2 * hidden_u, hidden_u]),
        ("nextn.enorm.weight", vec![hidden_u]),
        ("nextn.hnorm.weight", vec![wide]),
    ] {
        let name = blk(stem);
        let t = tensor(draft, &name)?;
        if t.dims != dims {
            return Err(PlacementError::Tensor {
                name,
                detail: format!(
                    "is {:?}; the input joins the {hidden_u}-value embedding with each of the {wide}-value streams' rows, {dims:?}",
                    t.dims
                ),
            });
        }
    }
    let q = blk("attn_q.weight");
    let rows = tensor(draft, &q)?.dims.get(1).copied();
    let layer = LayerSpec {
        mixer: Mixer::Gqa(Gqa {
            heads: t.gqa.heads,
            kv_heads: t.gqa.kv_heads,
            head_dim: t.gqa.head_dim,
            value_dim: t.gqa.value_dim,
            rope: Rope {
                mode: RopeMode::Imrope {
                    sections: sections(draft)?,
                },
                dims: spec_u32(
                    "rope.dimension_count",
                    meta_usize(draft, "rope.dimension_count")?,
                )?,
                base: meta_f32(draft, "rope.freq_base")?,
                yarn: None,
            },
            qk_norm: true,
            out_gate: out_gate(q, rows, t.gqa.heads, t.gqa.head_dim)?,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }),
        // The keys matched the target's, and the architecture's router and
        // shared-expert gate are its constants: the target's block.
        ffn: Ffn::Moe(moe.clone()),
        residual: Residual::Hc,
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
            bytes: draft.iter_tensors().map(|(_, t)| t.nbytes).sum(),
            borrows,
        },
        layer,
        index: spec_u32("block_count", index)?,
        hidden: spec.hidden,
        vocab: spec.vocab,
        rms_eps: meta_f32(draft, "attention.layer_norm_rms_epsilon")?,
        hc: Some(hc),
        input: MtpInput::Streams,
        head_norm: MtpHeadNorm::HcHead,
        head_rows: HeadRows::Full,
    })))
}

/// What the draft is compared with: the target's attention geometry.
struct Target<'a> {
    gqa: &'a Gqa,
}

impl<'a> Target<'a> {
    fn of(spec: &'a ModelSpec) -> Result<Target<'a>, PlacementError> {
        if spec.arch != Arch::Qwen4Exp {
            return Err(PlacementError::Metadata {
                key: "target".to_string(),
                detail: format!(
                    "is {}; an MTP draft pairs with a qwen4exp target",
                    spec.arch.name()
                ),
            });
        }
        let gqa = spec
            .layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::Gqa(g) => Some(g),
                _ => None,
            })
            .ok_or_else(|| target_lacks("an attention layer"))?;
        Ok(Target { gqa })
    }
}

/// The error naming what the target lacks for an MTP draft to be read against it.
fn target_lacks(what: &str) -> PlacementError {
    PlacementError::Metadata {
        key: "target".to_string(),
        detail: format!("has no {what}; an MTP draft pairs with a qwen4exp target"),
    }
}

/// The target's hyper-connection rank.
fn hc_rank(spec: &ModelSpec) -> Result<u32, PlacementError> {
    match spec.hc.map(|h| h.kind) {
        Some(HcKind::Gated { rank }) => Ok(rank),
        _ => Err(target_lacks("gated-residual hyper-connections")),
    }
}

fn streams_u32(streams: usize) -> Result<u32, PlacementError> {
    spec_u32("hyper_connection.count", streams)
}

/// Err naming `<architecture>.<suffix>` when the draft's `value` is not the target's.
fn same(draft: &Split, suffix: &str, value: usize, target: u32) -> Result<(), PlacementError> {
    if u64::try_from(value).ok() == Some(u64::from(target)) {
        return Ok(());
    }
    Err(metadata(
        draft,
        suffix,
        format!("is {value}; the target's is {target}"),
    ))
}

/// `key`, an un-prefixed token id; `None` when absent.
fn token_id(split: &Split, key: &str) -> Result<Option<u64>, PlacementError> {
    match split.value(key) {
        None => Ok(None),
        Some(v) => v
            .as_unsigned()
            .map(Some)
            .ok_or_else(|| PlacementError::Metadata {
                key: key.to_string(),
                detail: format!("is {v:?}, not a token id"),
            }),
    }
}

/// The MTP layer's pool from `attention.compress_ratios`, as ik reads it: an
/// array of `n_layer` values gives the layer its own; one of the `n_main`
/// main layers' makes it inherit the last main layer's. Any other length,
/// which ik truncates or reads as all zeros, is refused.
fn ratio(draft: &Split, n_layer: usize, n_main: usize) -> Result<usize, PlacementError> {
    let key = "attention.compress_ratios";
    let items = meta_arr(draft, key)?;
    let at = if items.len() == n_layer {
        n_layer - 1
    } else if items.len() == n_main {
        n_main - 1
    } else {
        return Err(metadata(
            draft,
            key,
            format!(
                "has {} values; the file covers its {n_layer} layers or the {n_main} main ones",
                items.len()
            ),
        ));
    };
    items[at]
        .as_unsigned()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| metadata(draft, key, format!("has no count at {at}")))
}

/// The draft's tensor `name`.
fn tensor<'a>(draft: &'a Split, name: &str) -> Result<&'a TensorInfo, PlacementError> {
    draft
        .find(name)
        .map(|(_, t)| t)
        .ok_or_else(|| PlacementError::Tensor {
            name: name.to_string(),
            detail: "is not in the draft file".to_string(),
        })
}

/// The draft carries the layer's tensors on `blk.{index}` and nothing else
/// but `token_embd` and `output`: no PLE tensor, no GDN or block-norm stem,
/// no tensor of another layer.
fn layer_tensors(draft: &Split, index: usize) -> Result<(), PlacementError> {
    let own = |stem: &&str| format!("blk.{index}.{stem}");
    let required: Vec<String> = ATTN_STEMS
        .iter()
        .filter(|s| !INDEXER_STEMS.contains(s))
        .chain(EXP_LAYER_STEMS)
        .chain(NEXTN_STEMS)
        .map(own)
        .collect();
    let allowed: Vec<String> = INDEXER_STEMS.iter().map(own).collect();
    let ple: Vec<String> = PLE_STEMS
        .iter()
        .map(own)
        .chain(["per_layer_token_embd.weight".to_string()])
        .collect();
    let names: Vec<&str> = draft.iter_tensors().map(|(_, t)| t.name.as_str()).collect();
    if let Some(name) = names.iter().find(|n| ple.iter().any(|p| p == *n)) {
        return Err(PlacementError::Tensor {
            name: (*name).to_string(),
            detail: "is a PLE tensor; the MTP layer carries no PLE site".to_string(),
        });
    }
    let missing: Vec<&String> = required
        .iter()
        .filter(|r| !names.contains(&r.as_str()))
        .collect();
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: (*name).clone(),
            detail: format!(
                "is not in the draft file ({} tensor(s) the MTP layer needs are missing: {})",
                missing.len(),
                missing
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    let other: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| {
            !required.iter().any(|r| r == n)
                && !allowed.iter().any(|a| a == n)
                && *n != EMBEDDING
                && *n != HEAD
        })
        .collect();
    if let Some(name) = other.first() {
        let kind = if GDN_STEMS.iter().chain(NORM_STEMS).any(|s| own(s) == *name) {
            "a GDN or block-norm stem, and the MTP layer is attention under hyper-connections"
        } else {
            "not the MTP layer's"
        };
        return Err(PlacementError::Tensor {
            name: (*name).to_string(),
            detail: format!(
                "is in the draft file and is {kind} ({} such: {})",
                other.len(),
                other.join(", ")
            ),
        });
    }
    Ok(())
}

/// Which of the target's matrices the draft uses: those it carries none of,
/// or both when `nextn_shared_target_tensors` is true. A declaration the
/// tensors present contradict is refused.
fn borrows(draft: &Split) -> Result<Borrows, PlacementError> {
    let has_embedding = draft.find(EMBEDDING).is_some();
    let has_head = draft.find(HEAD).is_some();
    let declared = match draft.value(&draft.arch_key(SHARED)) {
        None => None,
        Some(v) => Some(
            v.as_bool()
                .ok_or_else(|| metadata(draft, SHARED, format!("is {v:?}, not a bool")))?,
        ),
    };
    let carried: Vec<&str> = [(EMBEDDING, has_embedding), (HEAD, has_head)]
        .iter()
        .filter(|(_, has)| *has)
        .map(|(n, _)| *n)
        .collect();
    match declared {
        Some(true) if !carried.is_empty() => Err(metadata(
            draft,
            SHARED,
            format!(
                "is true, and the file carries {}: a shared file uses the target's token_embd and output",
                carried.join(" and ")
            ),
        )),
        Some(false) if carried.len() != 2 => Err(metadata(
            draft,
            SHARED,
            format!(
                "is false, and the file carries only [{}] of token_embd and output",
                carried.join(", ")
            ),
        )),
        _ => Ok(Borrows {
            embedding: !has_embedding,
            head: !has_head,
        }),
    }
}

/// The borrowed `output.weight`'s form, as [`mtp_of`] admitted it: Q8_0
/// planes, or the Q6_K rows the card head's quantizer feeds
/// (`bloomery_gpu::q6k_ids`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorrowedHead {
    Q8_0,
    Q6K,
}

/// The head [`mtp_of`] admitted for the `target`'s `output.weight`: the form
/// its card program launches. [`mtp_of`] refused every other type and shape,
/// so a tensor of one here is refused by name — the admission and this
/// reading stay one contract.
pub fn head_kind(target: &Split, hidden: u32, vocab: u32) -> Result<BorrowedHead, PlacementError> {
    let found = target
        .find(HEAD)
        .map(|(_, t)| t)
        .ok_or_else(|| PlacementError::Tensor {
            name: HEAD.to_string(),
            detail: format!(
                "is absent in the target file; the MTP layer reads a Q8_0 or Q6_K [{hidden}, \
                 {vocab}] matrix"
            ),
        })?;
    match output_form(found.ty) {
        Some(kind) if found.dims == [u64::from(hidden), u64::from(vocab)] => Ok(kind),
        _ => Err(PlacementError::Tensor {
            name: HEAD.to_string(),
            detail: format!(
                "is {} {:?} in the target file; the MTP layer reads a Q8_0 or Q6_K [{hidden}, \
                 {vocab}] matrix",
                found.ty, found.dims
            ),
        }),
    }
}

/// The form the draft's borrowed head reads an `output` matrix of, `None`
/// for a type no gemv of its program runs — the one owner of that list,
/// matched by [`head_kind`] and asked by
/// [`PlanInputs::mtp_borrows`](super::place::PlanInputs::mtp_borrows), so
/// the borrow check and the reader admit the same types.
#[must_use]
pub fn output_form(ty: GgmlType) -> Option<BorrowedHead> {
    match ty {
        GgmlType::Q8_0 => Some(BorrowedHead::Q8_0),
        GgmlType::Q6_K => Some(BorrowedHead::Q6K),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use models::{Borrows, DraftSpec, Mixer, ModelSpec, MtpDraft, MtpSource};

    use super::super::hparams::tests::{keys, tensors};
    use super::super::hparams::{ATTN_STEMS, EXP_LAYER_STEMS};
    use super::{BorrowedHead, EMBEDDING, HEAD, NEXTN_STEMS, head_kind, mtp_of};
    use crate::arch::synthetic::{Ty, V, header_typed};

    /// The small target of the `hparams` tests: four layers, the last one
    /// attention (its `attn_q` writing the gate), 64 values, a two-token
    /// vocabulary, the embedding and head Q8_0.
    fn target_tensors() -> Vec<(String, Vec<u64>, Ty)> {
        tensors()
            .into_iter()
            .map(|(n, d)| match n.as_str() {
                "blk.3.attn_q.weight" => (n, vec![64, 128], Ty::F32),
                EMBEDDING | HEAD => (n, vec![64, 2], Ty::Q8_0),
                _ => (n, d, Ty::F32),
            })
            .collect()
    }

    /// The un-prefixed keys both files carry.
    fn tokens() -> Vec<(&'static str, V)> {
        vec![
            ("tokenizer.ggml.pre", V::Str("qwen35")),
            ("tokenizer.ggml.eos_token_id", V::U32(1)),
            ("tokenizer.ggml.padding_token_id", V::U32(0)),
        ]
    }

    /// A shared draft of the target: `block_count` 5, its layer `blk.4`
    /// dense (`[4] = 0`, the last main layer's pool 4).
    fn draft_keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(5)),
            ("nextn_predict_layers", V::U32(1)),
            ("nextn_shared_target_tensors", V::Bool(true)),
            ("embedding_length", V::U32(64)),
            ("attention.head_count", V::U32(4)),
            ("attention.head_count_kv", V::U32(2)),
            ("attention.key_length", V::U32(16)),
            ("attention.value_length", V::U32(16)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("rope.dimension_count", V::U32(8)),
            ("rope.dimension_sections", V::I32s(vec![2, 2, 0, 0])),
            ("rope.freq_base", V::F32(1e7)),
            ("expert_count", V::U32(8)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(16)),
            ("expert_shared_feed_forward_length", V::U32(16)),
            ("hyper_connection.count", V::U32(4)),
            ("hyper_connection.low_rank", V::U32(8)),
            ("attention.compress_ratios", V::I32s(vec![0, 0, 0, 4, 0])),
        ]
    }

    fn draft_tensors() -> Vec<(String, Vec<u64>, Ty)> {
        ATTN_STEMS
            .iter()
            .chain(EXP_LAYER_STEMS)
            .chain(NEXTN_STEMS)
            .map(|s| {
                let dims = match *s {
                    "attn_q.weight" => vec![64, 128],
                    "nextn.eh_proj.weight" => vec![128, 64],
                    "nextn.enorm.weight" => vec![64],
                    "nextn.hnorm.weight" => vec![256],
                    _ => vec![1],
                };
                (format!("blk.4.{s}"), dims, Ty::F32)
            })
            .collect()
    }

    /// The two matrices of an unshared draft.
    fn own_matrices() -> Vec<(String, Vec<u64>, Ty)> {
        [EMBEDDING, HEAD]
            .iter()
            .map(|n| ((*n).to_string(), vec![64, 2], Ty::Q8_0))
            .collect()
    }

    /// `kv` with `key` set to `v`, or dropped for `None`.
    fn with(kv: Vec<(&'static str, V)>, key: &'static str, v: Option<V>) -> Vec<(&'static str, V)> {
        let mut v = v;
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

    /// A draft header, read against the small target.
    struct Case {
        arch: &'static str,
        kv: Vec<(&'static str, V)>,
        global: Vec<(&'static str, V)>,
        tensors: Vec<(String, Vec<u64>, Ty)>,
        /// The target's keys and tensors.
        target_kv: Vec<(&'static str, V)>,
        target: Vec<(String, Vec<u64>, Ty)>,
    }

    impl Case {
        fn shared() -> Case {
            Case {
                arch: "qwen4exp",
                kv: draft_keys(),
                global: tokens(),
                tensors: draft_tensors(),
                target_kv: keys(),
                target: target_tensors(),
            }
        }

        /// [`Case::shared`] at `embedding_length` 256, the smallest hidden a
        /// Q6_K or Q4_K block divides: the reader refuses a K-quant row
        /// whose width is not whole blocks, so a target head of one needs a
        /// hidden the block divides.
        fn wide() -> Case {
            let kv = |mut kv: Vec<(&'static str, V)>| {
                kv.iter_mut().for_each(|(k, v)| {
                    if *k == "embedding_length" {
                        *v = V::U32(256);
                    } else if *k == "embedding_length_per_layer_input" {
                        // Four PLE rows a ngram: heads·row stays the
                        // embedding's width.
                        *v = V::U32(64);
                    }
                });
                kv
            };
            let tensors = |ts: Vec<(String, Vec<u64>, Ty)>| {
                ts.into_iter()
                    .map(|(n, d, ty)| {
                        let d = match n.rsplit('.').next().unwrap_or("") {
                            "weight" if d == [64] => vec![256],
                            "weight" if d == [128, 64] => vec![512, 256],
                            "weight" if d == [256] => vec![1024],
                            "weight" if d == [64, 128] => vec![256, 128],
                            "weight" if d == [64, 2] => vec![256, 2],
                            "weight" if d == [16, 20] => vec![64, 20],
                            _ => d,
                        };
                        (n, d, ty)
                    })
                    .collect()
            };
            Case {
                arch: "qwen4exp",
                kv: kv(draft_keys()),
                global: tokens(),
                tensors: tensors(draft_tensors()),
                target_kv: kv(keys()),
                target: tensors(target_tensors()),
            }
        }

        fn unshared() -> Case {
            let mut c = Case::shared();
            c.kv = with(c.kv, "nextn_shared_target_tensors", None);
            c.tensors.extend(own_matrices());
            c
        }

        fn read_with(
            &self,
            tag: &str,
            edit: impl FnOnce(&mut ModelSpec),
        ) -> Result<MtpDraft, String> {
            let tpath = header_typed(
                &format!("{tag}-t"),
                "qwen4exp",
                &self.target_kv,
                &tokens(),
                &self.target,
            );
            let dpath = header_typed(
                &format!("{tag}-d"),
                self.arch,
                &self.kv,
                &self.global,
                &self.tensors,
            );
            let target = gguf::Split::open(&tpath).expect("the synthetic target opens");
            let draft = gguf::Split::open(&dpath).expect("the synthetic draft opens");
            let mut spec = super::super::spec::read(&target)
                .expect("the target reads")
                .spec;
            edit(&mut spec);
            let out = mtp_of(&draft, &target, &spec).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&tpath);
            let _ = std::fs::remove_file(&dpath);
            out.map(|d| match d {
                DraftSpec::Mtp(m) => *m,
                DraftSpec::Block(_) => panic!("an MTP file read as a block draft"),
            })
        }

        fn read(&self, tag: &str) -> Result<MtpDraft, String> {
            self.read_with(tag, |_| {})
        }

        fn refused(&self, tag: &str, want: &str) {
            let err = self.read(tag).expect_err("the draft is refused");
            assert!(err.contains(want), "want {want:?} in: {err}");
        }
    }

    fn borrows_of(m: &MtpDraft) -> Borrows {
        match &m.source {
            MtpSource::File { borrows, .. } => *borrows,
            MtpSource::InFile { .. } => panic!("a draft file read as in the target's"),
        }
    }

    fn dense(m: &MtpDraft) -> bool {
        matches!(&m.layer.mixer, Mixer::Gqa(g) if g.select.is_none() && g.out_gate && g.qk_norm)
    }

    #[test]
    fn a_shared_file_borrows_both_matrices() {
        let m = Case::shared()
            .read("mtp-shared")
            .expect("the shared draft reads");
        assert_eq!(m.index, 4);
        assert!(dense(&m), "{:?}", m.layer.mixer);
        assert_eq!(
            borrows_of(&m),
            Borrows {
                embedding: true,
                head: true
            }
        );
        assert_eq!((m.hidden, m.vocab), (64, 2));
    }

    #[test]
    fn an_unshared_file_borrows_neither() {
        let m = Case::unshared()
            .read("mtp-own")
            .expect("the unshared draft reads");
        assert_eq!(
            borrows_of(&m),
            Borrows {
                embedding: false,
                head: false
            }
        );
        let MtpSource::File { bytes, .. } = m.source else {
            panic!("a file source")
        };
        let own = Case::shared()
            .read("mtp-own-b")
            .expect("the shared draft reads");
        let MtpSource::File { bytes: shared, .. } = own.source else {
            panic!("a file source")
        };
        assert_eq!(bytes - shared, 2 * (64 * 2 / 32 * 34));
    }

    /// With no declaration, each matrix the file lacks is the target's.
    #[test]
    fn absence_borrows_each_matrix() {
        let mut c = Case::unshared();
        c.tensors.retain(|(n, _, _)| n != EMBEDDING);
        let m = c.read("mtp-half").expect("the draft reads");
        assert_eq!(
            borrows_of(&m),
            Borrows {
                embedding: true,
                head: false
            }
        );
    }

    #[test]
    fn a_shared_declaration_beside_the_matrices_is_refused() {
        let mut c = Case::shared();
        c.tensors.extend(own_matrices().into_iter().take(1));
        c.refused(
            "mtp-contra-t",
            "qwen4exp.nextn_shared_target_tensors: is true, and the file carries token_embd.weight",
        );
    }

    #[test]
    fn an_unshared_declaration_without_the_matrices_is_refused() {
        let mut c = Case::shared();
        c.kv = with(c.kv, "nextn_shared_target_tensors", Some(V::Bool(false)));
        c.refused(
            "mtp-contra-f",
            "qwen4exp.nextn_shared_target_tensors: is false, and the file carries only []",
        );
    }

    /// An array of `block_count` values gives the layer its own ratio: 0
    /// here, where the last main layer's is 4.
    #[test]
    fn a_ratio_array_of_block_count_takes_the_layers_own() {
        let m = Case::shared().read("mtp-r5").expect("the draft reads");
        assert!(dense(&m));
        let mut c = Case::shared();
        c.kv = with(
            c.kv,
            "attention.compress_ratios",
            Some(V::I32s(vec![0, 0, 0, 0, 4])),
        );
        c.refused(
            "mtp-r5-sel",
            "gives the MTP layer 4 a pool of 4: a selecting MTP layer is not built",
        );
    }

    /// An array of the main layers' values makes the layer inherit the last one's.
    #[test]
    fn a_ratio_array_of_the_main_layers_inherits_the_last() {
        let mut c = Case::shared();
        c.kv = with(
            c.kv,
            "attention.compress_ratios",
            Some(V::I32s(vec![0, 0, 0, 4])),
        );
        c.refused(
            "mtp-r4-sel",
            "gives the MTP layer 4 a pool of 4: a selecting MTP layer is not built",
        );
        let mut c = Case::shared();
        c.kv = with(
            c.kv,
            "attention.compress_ratios",
            Some(V::I32s(vec![0, 0, 4, 0])),
        );
        assert!(dense(&c.read("mtp-r4-dense").expect("the draft reads")));
    }

    #[test]
    fn a_ratio_array_of_another_length_or_none_is_refused() {
        for (tag, v) in [
            ("mtp-r3", Some(vec![0, 0, 0])),
            ("mtp-r6", Some(vec![0; 6])),
            ("mtp-r0", None),
        ] {
            let mut c = Case::shared();
            c.kv = with(c.kv, "attention.compress_ratios", v.map(V::I32s));
            c.refused(tag, "qwen4exp.attention.compress_ratios");
        }
    }

    /// Every key the layer shares with the target, set off by one, is refused by its name.
    #[test]
    fn a_key_unlike_the_targets_is_refused_by_name() {
        for key in [
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_feed_forward_length",
            "hyper_connection.count",
            "hyper_connection.low_rank",
        ] {
            let mut c = Case::shared();
            let v = match c.kv.iter().find(|(k, _)| *k == key) {
                Some((_, V::U32(x))) => *x,
                _ => panic!("{key} is a u32 of the draft keys"),
            };
            c.kv = with(c.kv, key, Some(V::U32(v + 1)));
            c.refused(
                &format!("mtp-key-{key}"),
                &format!("qwen4exp.{key}: is {}; the target's is {v}", v + 1),
            );
        }
    }

    #[test]
    fn a_vocabulary_unlike_the_targets_is_refused() {
        let mut c = Case::shared();
        c.kv.push(("vocab_size", V::U32(3)));
        c.refused("mtp-vocab", "qwen4exp.vocab_size: is 3; the target's is 2");
    }

    #[test]
    fn an_eos_or_pad_unlike_the_targets_is_refused() {
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
        ] {
            let mut c = Case::shared();
            c.global = with(c.global, key, Some(V::U32(7)));
            c.refused(
                &format!("mtp-{key}"),
                &format!("metadata {key}: is Some(7) in the draft file"),
            );
        }
    }

    #[test]
    fn a_draft_of_another_architecture_is_refused() {
        let mut c = Case::shared();
        c.arch = "qwen35moe";
        c.refused(
            "mtp-arch",
            "is qwen35moe; an MTP draft file of this family is qwen4exp",
        );
    }

    #[test]
    fn a_target_of_another_architecture_is_refused() {
        let err = Case::shared()
            .read_with("mtp-tarch", |s| s.arch = models::Arch::Qwen35Moe)
            .expect_err("the target is refused");
        assert!(
            err.contains(
                "metadata target: is qwen35moe; an MTP draft pairs with a qwen4exp target"
            ),
            "{err}"
        );
    }

    #[test]
    fn more_than_one_next_token_layer_is_refused() {
        let mut c = Case::shared();
        c.kv = with(c.kv, "nextn_predict_layers", Some(V::U32(2)));
        c.refused("mtp-nextn", "qwen4exp.nextn_predict_layers: is 2");
    }

    #[test]
    fn a_block_count_off_the_target_is_refused() {
        let mut c = Case::shared();
        c.kv = with(c.kv, "block_count", Some(V::U32(6)));
        c.refused("mtp-blocks", "qwen4exp.block_count: is 6");
    }

    #[test]
    fn a_ple_tensor_is_refused() {
        let mut c = Case::shared();
        c.tensors
            .push(("blk.4.ple_key.weight".to_string(), vec![1], Ty::F32));
        c.refused("mtp-ple", "tensor blk.4.ple_key.weight: is a PLE tensor");
    }

    #[test]
    fn a_missing_nextn_tensor_is_refused() {
        for stem in NEXTN_STEMS {
            let mut c = Case::shared();
            let name = format!("blk.4.{stem}");
            c.tensors.retain(|(n, _, _)| *n != name);
            c.refused(
                &format!("mtp-miss-{stem}"),
                &format!("tensor {name}: is not in the draft file"),
            );
        }
    }

    #[test]
    fn a_gdn_stem_or_block_norm_is_refused() {
        for stem in ["attn_qkv.weight", "attn_norm.weight"] {
            let mut c = Case::shared();
            c.tensors.push((format!("blk.4.{stem}"), vec![1], Ty::F32));
            c.refused(&format!("mtp-extra-{stem}"), "a GDN or block-norm stem");
        }
    }

    #[test]
    fn a_matrix_not_q8_0_of_hidden_by_vocab_is_refused() {
        let mut c = Case::unshared();
        c.tensors.retain(|(n, _, _)| n != HEAD);
        c.tensors.push((HEAD.to_string(), vec![64, 2], Ty::F32));
        c.refused(
            "mtp-head-ty",
            "tensor output.weight: is f32 [64, 2] in the draft's file",
        );
    }

    /// A borrowed matrix is the target's, held to the forms it reads: an
    /// F32 or Q4_K target head is refused by name.
    #[test]
    fn a_target_head_not_q8_0_or_q6_k_is_refused() {
        for (tag, ty, shown) in [
            ("mtp-thead-f32", Ty::F32, "f32"),
            ("mtp-thead-q4k", Ty::Q4K, "q4_K"),
        ] {
            let mut c = Case::wide();
            for t in &mut c.target {
                if t.0 == HEAD {
                    t.2 = ty;
                }
            }
            c.refused(
                tag,
                &format!("tensor output.weight: is {shown} [256, 2] in the target's file"),
            );
        }
    }

    /// A Q6_K target head is admitted: the card program reads its rows
    /// through the row-map gemv, and `head_kind` records the form.
    #[test]
    fn a_q6_k_target_head_is_admitted() {
        let mut c = Case::wide();
        for t in &mut c.target {
            if t.0 == HEAD {
                t.2 = Ty::Q6K;
            }
        }
        let m = c.read("mtp-thead-q6k").expect("the draft reads");
        assert_eq!(
            borrows_of(&m),
            Borrows {
                embedding: true,
                head: true
            }
        );
        let mut tk = keys();
        tk.iter_mut().for_each(|(k, v)| {
            if *k == "embedding_length" {
                *v = V::U32(256);
            }
        });
        let path = header_typed("mtp-hk-q6k", "qwen4exp", &tk, &tokens(), &c.target);
        let target = gguf::Split::open(&path).expect("the synthetic target opens");
        let got = head_kind(&target, 256, 2).expect("the target's head reads");
        assert_eq!(got, BorrowedHead::Q6K);
        let _ = std::fs::remove_file(&path);
    }

    /// The embedding stays Q8_0 whatever the head: a Q6_K `token_embd` is
    /// refused by name.
    #[test]
    fn a_q6_k_token_embd_is_still_refused() {
        let mut c = Case::wide();
        for t in &mut c.target {
            if t.0 == EMBEDDING {
                t.2 = Ty::Q6K;
            }
        }
        c.refused(
            "mtp-temb-q6k",
            "tensor token_embd.weight: is q6_K [256, 2] in the target's file",
        );
    }

    /// `head_kind` names the target's `output` form, refusing what `mtp_of`
    /// would have refused.
    #[test]
    fn head_kind_reads_the_targets_head_form() {
        let path = header_typed("mtp-hk", "qwen4exp", &keys(), &tokens(), &target_tensors());
        let target = gguf::Split::open(&path).expect("the synthetic target opens");
        let got = head_kind(&target, 64, 2).expect("the target's head reads");
        assert_eq!(got, BorrowedHead::Q8_0);
        let _ = std::fs::remove_file(&path);
        let mut f32s = target_tensors();
        for t in &mut f32s {
            if t.0 == HEAD {
                t.2 = Ty::F32;
            }
        }
        let path = header_typed("mtp-hk-f32", "qwen4exp", &keys(), &tokens(), &f32s);
        let target = gguf::Split::open(&path).expect("the synthetic target opens");
        let err = head_kind(&target, 64, 2)
            .err()
            .map_or("read".to_string(), |e| e.to_string());
        assert!(
            err.contains("tensor output.weight: is f32 [64, 2] in the target file"),
            "{err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_matrix_of_another_shape_is_refused() {
        let mut c = Case::unshared();
        for t in &mut c.tensors {
            if t.0 == HEAD {
                t.1 = vec![64, 32];
            }
        }
        c.refused(
            "mtp-head-dims",
            "tensor output.weight: is q8_0 [64, 32] in the draft's file",
        );
    }

    #[test]
    fn an_input_projection_of_another_shape_is_refused() {
        let mut c = Case::shared();
        for t in &mut c.tensors {
            if t.0 == "blk.4.nextn.hnorm.weight" {
                t.1 = vec![64];
            }
        }
        c.refused("mtp-hnorm", "tensor blk.4.nextn.hnorm.weight: is [64]");
    }
}
