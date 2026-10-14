//! A qwen4exp MTP draft file read against its target (`mtp_of`), and a
//! qwen35moe file's own next-token layer (`in_file`). The two readers do not
//! share a branch: qwen4exp's layer is hyper-connected streams in its own
//! file, and qwen35moe's is one plain full-attention layer in the target.
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
//! declares that it carries neither, and the format's rule is
//! [`crate::arch::draft::borrows`].

use gguf::{GgmlType, Split, TensorInfo};
use models::headpick::HeadSet;
use models::{
    Arch, DraftSpec, Ffn, Gqa, HcKind, LayerSpec, Mixer, ModelSpec, MtpDraft, MtpHeadNorm,
    MtpInput, MtpSource, Residual, Rope, RopeMode,
};

use super::hparams::{
    ATTN_STEMS, EXP_LAYER_STEMS, GDN_STEMS, Hparams, NORM_STEMS, PLE_STEMS, Variant, expert_ff,
    refuse_unless, sections,
};
use super::place::{PlaceError, PlanInputs};
use super::spec::{out_gate, routed_at};
use crate::arch::draft::{EMBEDDING, HEAD, borrows, same, same_token_ids};
use crate::arch::{
    architecture_is, meta_arr, meta_f32, meta_usize, metadata, n_vocab, qwen35moe_variant, spec_u32,
};
use crate::placement::{ModelTensor, ModelTensors, PlacementError};
use runtime::stores::{self, dense_kv_bytes};

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
    same_token_ids(draft, target)?;
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
        head_rows: HeadSet::Full,
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

/// Whether the qwen4exp set whose first shard is `target` can run its draft
/// file ([`crate::arch::draft::draft_usable`]'s qwen4exp rule): the plan's
/// own rule for the matrices the draft borrows
/// ([`PlanInputs::mtp_borrows`]). The draft is one shared file, so a set that
/// names the checkpoint it is made from is a table error, refused.
pub fn target_usable(target: &Split, checkpoint: Option<&str>) -> Result<(), PlaceError> {
    if let Some(c) = checkpoint {
        return Err(PlacementError::Metadata {
            key: "general.name".to_string(),
            detail: format!("is named {c:?} for a qwen4exp draft, which is one shared file"),
        }
        .into());
    }
    PlanInputs::describe(target)?.mtp_borrows()
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

/// The join `nextn.eh_proj` reads: the embedding's half, then the head's row
/// (`2 × hidden` values). [`MtpInput::HeadRow`] states that order and has no
/// field for it.
pub const EH_JOIN: [&str; 2] = ["e", "h"];

/// The stems a shaped qwen35moe next-token layer carries. No hyper-connection
/// head: the layer is one plain stream.
const DRAFT_STEMS: &[&str] = &[
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "post_attention_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
];

/// Matrices the layer borrows from the target. A file that carries its own
/// is refused; the role table does not name them either.
const OWN_MATRICES: &[&str] = &["nextn.embed_tokens.weight", "nextn.shared_head_head.weight"];

/// What a qwen35moe file's own next-token layer is ([`in_file`]).
#[derive(Debug)]
pub enum InFile {
    /// One shaped draft at `n_trunk`. Boxed: the layer description is many
    /// times the other variants.
    Draft(Box<MtpDraft>),
    /// The key is absent, or the blocks past the trunk are trunk layers.
    Absent,
    /// The key is set and the tail block holds no tensor. `reason` is the
    /// line [`crate::arch::nextn_layers`] records.
    Stripped {
        /// The defaults line, naming the block.
        reason: String,
    },
    /// The key is 1 and every tail tensor is a one-element vector: the
    /// role table's stub, not a draft's shapes.
    Unshaped,
}

/// The qwen35moe file's own next-token layer, or why it is not a draft.
/// A file that carries `nextn.embed_tokens` or `nextn.shared_head_head` is
/// refused here, by those names.
pub fn in_file(split: &Split) -> Result<InFile, PlacementError> {
    refuse_own(split)?;
    let hp = Hparams::read(split)?;
    if hp.variant != Variant::Qwen35Moe {
        return Err(architecture_is(&format!(
            "{:?}; an in-file next-token draft of this family is qwen35moe",
            hp.variant
        )));
    }
    let tensors = super::roles::classify(split, &hp)?;
    Ok(match describe(&hp, &tensors)? {
        Some(draft) => InFile::Draft(Box::new(draft)),
        None => no_draft(&hp, &tensors)?,
    })
}

/// The draft `hp` and `tensors` describe, when they are a shaped qwen35moe
/// next-token layer. `None` when the file has no such layer: another
/// variant, no key, a stripped tail, or a one-element stub. A shaped tail
/// that is not this layer is an error naming the key or the tensor.
pub(crate) fn describe(
    hp: &Hparams,
    tensors: &ModelTensors,
) -> Result<Option<MtpDraft>, PlacementError> {
    if hp.variant != Variant::Qwen35Moe {
        return Ok(None);
    }
    let past = hp
        .n_layer
        .checked_sub(hp.n_trunk)
        .ok_or_else(|| PlacementError::Metadata {
            key: "block_count".to_string(),
            detail: format!("is {}, under the trunk {}", hp.n_layer, hp.n_trunk),
        })?;
    if past == 0 {
        return Ok(None);
    }
    if past != 1 {
        return Err(PlacementError::Metadata {
            key: format!("{}.nextn_predict_layers", Arch::Qwen35Moe.name()),
            detail: format!("is {past}; one next-token layer is built"),
        });
    }
    let index = hp.n_trunk;
    let tail: Vec<&ModelTensor> = tensors
        .tensors
        .iter()
        .filter(|t| t.layer == Some(index))
        .collect();
    if let Some(name) = tail.iter().map(|t| t.name.as_str()).find(|n| gdn_name(n)) {
        return Err(PlacementError::Tensor {
            name: name.to_string(),
            detail: "is a gated-delta stem on the next-token layer, which is full attention"
                .to_string(),
        });
    }
    if let Some(name) = tail.iter().map(|t| t.name.as_str()).find(|n| own_name(n)) {
        return Err(own_error(name, &[name]));
    }
    if tail.is_empty() || tail.iter().all(|t| t.dims == [1]) {
        return Ok(None);
    }
    Ok(Some(shaped(hp, tensors, index)?))
}

/// Card bytes of this in-file draft at `ctx` positions: the next-token
/// layer's own tensors in `tensors`, one f32 residual stream of
/// `draft.hidden` and one dense attention KV ([`dense_kv_bytes`]) of the
/// layer's GQA. The `token_embd` and `output` it borrows are the target's
/// bytes, counted with the target.
#[must_use = "the count is this draft's card bytes"]
pub fn in_file_card_bytes(
    draft: &MtpDraft,
    tensors: &ModelTensors,
    ctx: usize,
) -> Result<u64, PlacementError> {
    let Mixer::Gqa(g) = &draft.layer.mixer else {
        return Err(PlacementError::Metadata {
            key: format!("{}.nextn", Arch::Qwen35Moe.name()),
            detail: "the in-file draft's mixer is not full attention".to_string(),
        });
    };
    if draft.hc.is_some() || draft.layer.residual != Residual::Plain {
        return Err(PlacementError::Metadata {
            key: format!("{}.nextn", Arch::Qwen35Moe.name()),
            detail: "the in-file draft is one plain stream".to_string(),
        });
    }
    let own = tensors
        .tensors
        .iter()
        .filter(|t| t.layer == Some(draft.index as usize))
        .try_fold(0u64, |n, t| n.checked_add(t.file_bytes));
    let stream = u64::from(draft.hidden) * stores::F32_BYTES;
    let kv = dense_kv_bytes(g.kv_heads as usize, g.head_dim as usize, ctx);
    own.and_then(|n| n.checked_add(stream))
        .and_then(|n| n.checked_add(kv))
        .ok_or_else(|| PlacementError::Metadata {
            key: format!("{}.nextn", Arch::Qwen35Moe.name()),
            detail: "the draft's card bytes overflow u64".to_string(),
        })
}

/// `split` carries a matrix the next-token layer borrows from the target.
fn refuse_own(split: &Split) -> Result<(), PlacementError> {
    let found: Vec<&str> = split
        .iter_tensors()
        .map(|(_, t)| t.name.as_str())
        .filter(|n| own_name(n))
        .collect();
    match found.first() {
        Some(name) => Err(own_error(name, &found)),
        None => Ok(()),
    }
}

fn own_name(name: &str) -> bool {
    OWN_MATRICES.iter().any(|s| name.ends_with(s))
}

fn own_error(name: &str, found: &[&str]) -> PlacementError {
    PlacementError::Tensor {
        name: name.to_string(),
        detail: format!(
            "is in the file; the next-token layer borrows token_embd.weight and output.weight \
             ({}: {})",
            found.len(),
            found.join(", ")
        ),
    }
}

fn gdn_name(name: &str) -> bool {
    let Some(stem) = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map(|(_, s)| s)
    else {
        return false;
    };
    stem.starts_with("ssm_") || GDN_STEMS.contains(&stem)
}

fn no_draft(hp: &Hparams, tensors: &ModelTensors) -> Result<InFile, PlacementError> {
    if hp.n_layer == hp.n_trunk {
        return Ok(InFile::Absent);
    }
    let index = hp.n_trunk;
    let any = tensors.tensors.iter().any(|t| t.layer == Some(index));
    if !any {
        let reason = hp
            .defaults
            .iter()
            .find(|d| d.contains("stripped next-token layer"))
            .cloned()
            .ok_or_else(|| PlacementError::Metadata {
                key: format!("{}.nextn_predict_layers", Arch::Qwen35Moe.name()),
                detail: format!(
                    "is set and blk.{index} carries no tensor, and the read recorded no \
                     stripped-layer line"
                ),
            })?;
        return Ok(InFile::Stripped { reason });
    }
    Ok(InFile::Unshaped)
}

fn shaped(hp: &Hparams, tensors: &ModelTensors, index: usize) -> Result<MtpDraft, PlacementError> {
    let missing: Vec<String> = DRAFT_STEMS
        .iter()
        .map(|s| blk(index, s))
        .filter(|n| named(tensors, n).is_err())
        .collect();
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is not in the file ({} tensor(s) the next-token layer needs are missing: {})",
                missing.len(),
                missing.join(", ")
            ),
        });
    }
    require_widths(hp, tensors, index)?;
    let layer = layer_of(hp, tensors, index)?;
    Ok(MtpDraft {
        source: MtpSource::InFile {
            layer: spec_u32("block_count", index)?,
        },
        layer,
        index: spec_u32("block_count", index)?,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        rms_eps: hp.rms_eps,
        hc: None,
        input: MtpInput::HeadRow,
        head_norm: MtpHeadNorm::Rms,
        head_rows: HeadSet::Full,
    })
}

fn require_widths(
    hp: &Hparams,
    tensors: &ModelTensors,
    index: usize,
) -> Result<(), PlacementError> {
    let n = hp.n_embd as u64;
    let join = [2 * n, n];
    let eh = named(tensors, &blk(index, "nextn.eh_proj.weight"))?;
    if eh.dims.as_slice() != join {
        return Err(PlacementError::Tensor {
            name: eh.name.clone(),
            detail: format!(
                "is {:?}; the join is {} then {} ({} values to {})",
                eh.dims,
                EH_JOIN[0],
                EH_JOIN[1],
                2 * hp.n_embd,
                hp.n_embd
            ),
        });
    }
    for stem in [
        "nextn.hnorm.weight",
        "nextn.enorm.weight",
        "nextn.shared_head_norm.weight",
        "attn_norm.weight",
        "post_attention_norm.weight",
    ] {
        let t = named(tensors, &blk(index, stem))?;
        if t.dims.as_slice() != [n] {
            return Err(PlacementError::Tensor {
                name: t.name.clone(),
                detail: format!(
                    "is {:?}; one stream norms {} values, not {}",
                    t.dims,
                    hp.n_embd,
                    4 * hp.n_embd
                ),
            });
        }
    }
    let head = hp.head_dim as u64;
    for stem in ["attn_q_norm.weight", "attn_k_norm.weight"] {
        let t = named(tensors, &blk(index, stem))?;
        if t.dims.as_slice() != [head] {
            return Err(PlacementError::Tensor {
                name: t.name.clone(),
                detail: format!("is {:?}; q and k are normed over the head ({head})", t.dims),
            });
        }
    }
    let borrowed = [n, hp.n_vocab as u64];
    for name in [EMBEDDING, HEAD] {
        let t = named(tensors, name)?;
        if t.dims.as_slice() != borrowed {
            return Err(PlacementError::Tensor {
                name: name.to_string(),
                detail: format!(
                    "is {:?}; the next-token layer borrows a [{}, {}] matrix",
                    t.dims, hp.n_embd, hp.n_vocab
                ),
            });
        }
    }
    Ok(())
}

fn layer_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    index: usize,
) -> Result<LayerSpec, PlacementError> {
    let heads = spec_u32("attention.head_count", hp.n_head)?;
    let head_dim = spec_u32("attention.key_length", hp.head_dim)?;
    let kv_heads = spec_u32("attention.head_count_kv", hp.n_head_kv)?;
    let q = blk(index, "attn_q.weight");
    let rows = named(tensors, &q)?.dims.get(1).copied();
    let gate = out_gate(q.clone(), rows, heads, head_dim)?;
    if !gate {
        return Err(PlacementError::Tensor {
            name: q,
            detail: "writes the query without the per-head gate".to_string(),
        });
    }
    let has = |name: String| tensors.tensors.iter().any(|t| t.name == name);
    let ffn = routed_at(hp, Arch::Qwen35Moe, index, true, &has)?;
    {
        let Ffn::Moe(moe) = &ffn else {
            return Err(PlacementError::Metadata {
                key: format!("{}.expert_count", Arch::Qwen35Moe.name()),
                detail: format!("layer {index} is not routed"),
            });
        };
        let shared = moe.shared.as_ref().ok_or_else(|| PlacementError::Tensor {
            name: blk(index, "ffn_up_shexp.weight"),
            detail: "is not in the file; the next-token layer runs a shared expert".to_string(),
        })?;
        if !shared.sigmoid_gate {
            return Err(PlacementError::Tensor {
                name: blk(index, "ffn_gate_inp_shexp.weight"),
                detail: "is not in the file; the shared expert's gate is a sigmoid".to_string(),
            });
        }
    }
    Ok(LayerSpec {
        mixer: Mixer::Gqa(Gqa {
            heads,
            kv_heads,
            head_dim,
            value_dim: head_dim,
            rope: Rope {
                mode: RopeMode::Imrope {
                    sections: hp.rope_sections,
                },
                dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
                base: hp.rope_base,
                yarn: None,
            },
            qk_norm: true,
            out_gate: true,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }),
        ffn,
        residual: Residual::Plain,
        extras: Vec::new(),
    })
}

fn blk(index: usize, stem: &str) -> String {
    format!("blk.{index}.{stem}")
}

fn named<'a>(tensors: &'a ModelTensors, name: &str) -> Result<&'a ModelTensor, PlacementError> {
    tensors
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| PlacementError::Tensor {
            name: name.to_string(),
            detail: "is not in the file".to_string(),
        })
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

#[cfg(test)]
mod in_file_tests {
    use models::{Ffn, Mixer, MtpHeadNorm, MtpInput, MtpSource, Residual};

    use super::{EH_JOIN, InFile, in_file, in_file_card_bytes};
    use crate::arch::draft::draft_usable;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::arch::{Read, qwen35moe};
    use crate::placement::Role;

    fn keys(blocks: u32, nextn: Option<u32>) -> Vec<(&'static str, V)> {
        let mut kv = vec![
            ("block_count", V::U32(blocks)),
            ("embedding_length", V::U32(64)),
            ("context_length", V::U32(128)),
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
            ("ssm.conv_kernel", V::U32(4)),
            ("ssm.state_size", V::U32(8)),
            ("ssm.group_count", V::U32(2)),
            ("ssm.time_step_rank", V::U32(4)),
            ("ssm.inner_size", V::U32(32)),
            ("full_attention_interval", V::U32(4)),
        ];
        if let Some(n) = nextn {
            kv.push(("nextn_predict_layers", V::U32(n)));
        }
        kv
    }

    /// The trunk: three delta layers, then attention. `embed` is the borrowed
    /// matrices' shape.
    fn trunk(embed: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = vec![
            ("token_embd.weight".to_string(), embed.clone()),
            ("output_norm.weight".to_string(), vec![64]),
            ("output.weight".to_string(), embed),
        ];
        for l in 0..4 {
            if l == 3 {
                t.push((format!("blk.{l}.attn_q.weight"), vec![64, 128]));
            } else {
                t.push((format!("blk.{l}.attn_qkv.weight"), vec![1]));
            }
            t.push((format!("blk.{l}.ffn_gate_inp.weight"), vec![1]));
        }
        t
    }

    /// Layer `l` as a shaped next-token layer. `hnorm`, `eh` and `q` are the
    /// widths the contract distinguishes.
    fn tail(l: usize, hnorm: Vec<u64>, eh: Vec<u64>, q: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        vec![
            (format!("blk.{l}.attn_norm.weight"), vec![64]),
            (format!("blk.{l}.attn_q.weight"), q),
            (format!("blk.{l}.attn_k.weight"), vec![64, 32]),
            (format!("blk.{l}.attn_v.weight"), vec![64, 32]),
            (format!("blk.{l}.attn_output.weight"), vec![64, 64]),
            (format!("blk.{l}.attn_q_norm.weight"), vec![16]),
            (format!("blk.{l}.attn_k_norm.weight"), vec![16]),
            (format!("blk.{l}.post_attention_norm.weight"), vec![64]),
            (format!("blk.{l}.ffn_gate_inp.weight"), vec![64, 8]),
            (format!("blk.{l}.ffn_gate_exps.weight"), vec![1]),
            (format!("blk.{l}.ffn_up_exps.weight"), vec![1]),
            (format!("blk.{l}.ffn_down_exps.weight"), vec![1]),
            (format!("blk.{l}.ffn_gate_inp_shexp.weight"), vec![64]),
            (format!("blk.{l}.ffn_gate_shexp.weight"), vec![1]),
            (format!("blk.{l}.ffn_up_shexp.weight"), vec![1]),
            (format!("blk.{l}.ffn_down_shexp.weight"), vec![1]),
            (format!("blk.{l}.nextn.eh_proj.weight"), eh),
            (format!("blk.{l}.nextn.enorm.weight"), vec![64]),
            (format!("blk.{l}.nextn.hnorm.weight"), hnorm),
            (format!("blk.{l}.nextn.shared_head_norm.weight"), vec![64]),
        ]
    }

    fn shaped() -> Vec<(String, Vec<u64>)> {
        let mut t = trunk(vec![64, 2]);
        t.extend(tail(4, vec![64], vec![128, 64], vec![64, 128]));
        t
    }

    /// Every tail tensor a one-element vector, the role table's stub.
    /// `tail` writes the shaped widths; this file is the names alone.
    fn stub() -> Vec<(String, Vec<u64>)> {
        let mut t = trunk(vec![1]);
        t.extend(
            tail(4, vec![1], vec![1], vec![1])
                .into_iter()
                .map(|(n, _)| (n, vec![1])),
        );
        t
    }

    fn open(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> (Read, Result<InFile, String>) {
        let path = header_shaped(
            tag,
            "qwen35moe",
            kv,
            &[("tokenizer.ggml.pre", V::Str("qwen35"))],
            tensors,
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split);
        let draft = in_file(&split).map_err(|e| e.to_string());
        drop(split);
        let _ = std::fs::remove_file(&path);
        (read.expect("the description reads"), draft)
    }

    fn refused(tag: &str, kv: &[(&str, V)], tensors: &[(String, Vec<u64>)]) -> String {
        let path = header_shaped(tag, "qwen35moe", kv, &[], tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = in_file(&split).map_or_else(|e| e.to_string(), |_| "accepted".to_string());
        drop(split);
        let _ = std::fs::remove_file(&path);
        err
    }

    #[test]
    fn d1_a_shaped_file_is_an_in_file_head_row_draft() {
        let (read, draft) = open("q35m-d1", &keys(5, Some(1)), &shaped());
        let InFile::Draft(d) = draft.expect("the layer is a draft") else {
            panic!("not a draft");
        };
        assert!(matches!(d.source, MtpSource::InFile { layer: 4 }));
        assert_eq!(d.index, 4);
        assert_eq!(d.input, MtpInput::HeadRow);
        assert_eq!(d.head_norm, MtpHeadNorm::Rms);
        assert!(d.hc.is_none());
        assert_eq!(d.layer.residual, Residual::Plain);
        assert_eq!(d.hidden, 64);
        assert_eq!(d.vocab, 2);
        let Mixer::Gqa(g) = &d.layer.mixer else {
            panic!("the mixer is not full attention: {:?}", d.layer.mixer);
        };
        assert!(g.out_gate && g.qk_norm && g.select.is_none());
        let Ffn::Moe(moe) = &d.layer.ffn else {
            panic!("the feed-forward block is not routed");
        };
        assert!(moe.shared.is_some_and(|s| s.sigmoid_gate));
        assert_eq!(read.spec.mtp.as_slice(), std::slice::from_ref(&d.layer));
        assert_eq!(read.spec.layers.len(), 4);
    }

    #[test]
    fn m_a_the_join_is_embedding_then_hidden() {
        assert_eq!(EH_JOIN, ["e", "h"]);
        let mut t = shaped();
        for row in &mut t {
            if row.0 == "blk.4.nextn.eh_proj.weight" {
                row.1 = vec![64, 64];
            }
        }
        let err = refused("q35m-ma", &keys(5, Some(1)), &t);
        assert!(
            err.contains("e then h") && err.contains("blk.4.nextn.eh_proj.weight"),
            "{err}"
        );
    }

    #[test]
    fn m_b_the_tail_is_full_attention_where_the_interval_says_recurrent() {
        let (_read, draft) = open("q35m-mb", &keys(5, Some(1)), &shaped());
        let InFile::Draft(d) = draft.expect("the layer is a draft") else {
            panic!("not a draft");
        };
        // Index 4 with interval 4: (4 + 1) % 4 != 0, so the trunk's interval
        // rule would call this layer recurrent. The draft is full attention.
        assert_ne!((d.index as usize + 1) % 4, 0);
        assert!(
            matches!(d.layer.mixer, Mixer::Gqa(_)),
            "classified {:?}, a gated-delta layer",
            d.layer.mixer
        );
    }

    #[test]
    fn m_c_a_shaped_draft_does_not_require_a_hyper_connection_head() {
        let (_read, draft) = open("q35m-mc", &keys(5, Some(1)), &shaped());
        let InFile::Draft(_) = draft.expect("no hc_head tensor is required") else {
            panic!("not a draft");
        };
    }

    #[test]
    fn m_d_a_key_with_no_tensors_is_not_a_draft() {
        let (read, draft) = open("q35m-md", &keys(5, Some(1)), &trunk(vec![1]));
        assert!(
            read.spec.mtp.is_empty(),
            "a stripped layer is a draft layer"
        );
        let InFile::Stripped { reason } = draft.expect("the file still reads") else {
            panic!("a stripped layer was accepted as a draft");
        };
        assert!(reason.contains("stripped next-token layer"), "{reason}");
        assert!(
            read.defaults.iter().any(|d| d == &reason),
            "{:?}",
            read.defaults
        );
    }

    #[test]
    fn m_e_hnorm_is_one_stream_of_n_embd() {
        let (_read, draft) = open("q35m-me", &keys(5, Some(1)), &shaped());
        assert!(
            matches!(draft, Ok(InFile::Draft(_))),
            "hnorm of n_embd was refused: {draft:?}"
        );
        let mut wide = shaped();
        for row in &mut wide {
            if row.0 == "blk.4.nextn.hnorm.weight" {
                row.1 = vec![256];
            }
        }
        let err = refused("q35m-me-wide", &keys(5, Some(1)), &wide);
        assert!(err.contains("one stream norms 64 values, not 256"), "{err}");
    }

    #[test]
    fn m_f_the_trunk_keeps_the_tail_unused() {
        let path = header_shaped("q35m-mf", "qwen35moe", &keys(5, Some(1)), &[], &stub());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = qwen35moe::hparams::Hparams::read(&split).expect("the header reads");
        let model = qwen35moe::roles::classify(&split, &hp).expect("every tensor has a role");
        drop(split);
        let _ = std::fs::remove_file(&path);
        let tail: Vec<_> = model
            .tensors
            .iter()
            .filter(|t| t.layer == Some(4))
            .collect();
        assert_eq!(tail.len(), 20);
        assert!(
            tail.iter().all(|t| t.role == Role::Unused),
            "blk.4 ran in the trunk: {tail:?}"
        );
    }

    #[test]
    fn d3_spec_mtp_is_one_layer_only_for_a_shaped_tail() {
        let (shaped_read, shaped_draft) = open("q35m-d3", &keys(5, Some(1)), &shaped());
        assert_eq!(shaped_read.spec.mtp.len(), 1);
        let InFile::Draft(d) = shaped_draft.expect("draft") else {
            panic!("not a draft");
        };
        assert_eq!(shaped_read.spec.mtp[0], d.layer);

        let (absent, absent_draft) = open("q35m-d3-abs", &keys(4, None), &trunk(vec![1]));
        assert!(absent.spec.mtp.is_empty());
        assert!(matches!(absent_draft, Ok(InFile::Absent)));

        let (plain, _) = open("q35m-d3-base", &keys(4, None), &trunk(vec![1]));
        let (with_stub, stub_draft) = open("q35m-d3-stub", &keys(5, Some(1)), &stub());
        assert!(matches!(stub_draft, Ok(InFile::Unshaped)));
        assert_eq!(with_stub.spec, plain.spec, "a stub is no layer of the spec");
        assert_eq!(with_stub.defaults, plain.defaults[1..]);
    }

    #[test]
    fn d3_a_count_other_than_one_is_refused_by_name() {
        let err = refused("q35m-n2", &keys(6, Some(2)), &trunk(vec![1]));
        assert!(
            err.contains("qwen35moe.nextn_predict_layers: is 2; one next-token layer is built"),
            "{err}"
        );
    }

    #[test]
    fn d3_a_gated_delta_stem_on_the_tail_is_refused_by_name() {
        let mut t = stub();
        t.push(("blk.4.ssm_a".to_string(), vec![1]));
        let err = refused("q35m-gdn", &keys(5, Some(1)), &t);
        assert!(
            err.contains("blk.4.ssm_a") && err.contains("full attention"),
            "{err}"
        );
    }

    #[test]
    fn d1_a_file_that_carries_its_own_embedding_or_head_is_refused_by_name() {
        let mut t = shaped();
        t.push(("blk.4.nextn.embed_tokens.weight".to_string(), vec![64, 2]));
        t.push((
            "blk.4.nextn.shared_head_head.weight".to_string(),
            vec![64, 2],
        ));
        let err = refused("q35m-own", &keys(5, Some(1)), &t);
        assert!(
            err.contains("borrows token_embd.weight")
                && err.contains("nextn.embed_tokens.weight")
                && err.contains("nextn.shared_head_head.weight"),
            "{err}"
        );
    }

    /// A tail the draft refuses leaves the plain read as it was: no draft
    /// layer, and the same layers and defaults as the file without the key.
    #[test]
    fn d6_a_refused_tail_leaves_the_plain_read_alone() {
        let mut gdn = stub();
        gdn.push(("blk.4.ssm_a".to_string(), vec![1]));
        for (tag, kv, tensors) in [
            ("q35m-d6-n2", keys(6, Some(2)), trunk(vec![1])),
            ("q35m-d6-gdn", keys(5, Some(1)), gdn),
        ] {
            let (read, draft) = open(tag, &kv, &tensors);
            assert!(read.spec.mtp.is_empty(), "{tag}: {:?}", read.spec.mtp);
            assert!(draft.is_err(), "{tag}: the draft is refused by name");
        }
    }

    #[test]
    fn d4_qwen35moe_has_an_in_file_rule() {
        let path = header_shaped("q35m-d4", "qwen35moe", &keys(5, Some(1)), &[], &shaped());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        draft_usable(&split, None).expect("a shaped in-file draft can run");
        let named = draft_usable(&split, Some("other-checkpoint")).expect_err("not a sidecar");
        drop(split);
        let _ = std::fs::remove_file(&path);
        assert!(named.to_string().contains("other-checkpoint"), "{named}");

        let path = header_shaped(
            "q35m-d4-abs",
            "qwen35moe",
            &keys(4, None),
            &[],
            &trunk(vec![1]),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = draft_usable(&split, None)
            .expect_err("no layer")
            .to_string();
        drop(split);
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("nextn_predict_layers") && !err.contains("no rule"),
            "{err}"
        );

        let path = header_shaped(
            "q35m-d4-strip",
            "qwen35moe",
            &keys(5, Some(1)),
            &[],
            &trunk(vec![1]),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = draft_usable(&split, None)
            .expect_err("stripped")
            .to_string();
        drop(split);
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("stripped next-token layer"), "{err}");
    }

    #[test]
    fn d5_card_bytes_are_the_layer_one_stream_and_one_kv() {
        let (read, draft) = open("q35m-d5", &keys(5, Some(1)), &shaped());
        let InFile::Draft(d) = draft.expect("draft") else {
            panic!("not a draft");
        };
        let got = in_file_card_bytes(&d, &read.tensors, 1).expect("the draft sizes");
        let Mixer::Gqa(g) = &d.layer.mixer else {
            panic!("not full attention");
        };
        let stream = u64::from(d.hidden) * runtime::stores::F32_BYTES;
        let kv = runtime::stores::dense_kv_bytes(g.kv_heads as usize, g.head_dim as usize, 1);
        let own: u64 = read
            .tensors
            .tensors
            .iter()
            .filter(|t| t.layer == Some(d.index as usize))
            .map(|t| t.file_bytes)
            .sum();
        let head = read
            .tensors
            .tensors
            .iter()
            .find(|t| t.name == "output.weight")
            .expect("the borrowed head")
            .file_bytes;
        assert!(own > 0, "the layer's own tensors");
        assert_eq!(got, own + stream + kv);
        assert_ne!(
            got,
            own + stream + kv + head,
            "the borrowed head is the target's"
        );
        assert_ne!(got, own + 4 * stream + kv, "a four-stream reserve");
    }
}
