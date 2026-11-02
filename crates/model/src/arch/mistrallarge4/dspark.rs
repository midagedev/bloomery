//! `mistral-large4`'s DSpark draft: the `dflash` file whose blocks are dense
//! MLA stages. It is not V4.1's draft (hyper-connections, a routed MoE,
//! [`crate::arch::dspark`]) and not DFlash2's (Qwen3-shaped GQA, a selector,
//! [`crate::arch::dflash2`]); the three share the architecture string, and
//! the keys and tensors tell them apart.
//!
//! The draft: `fc` projects the target's rows at `target_layers` (each
//! `embedding_length` wide) and `enc.output_norm` normalizes the projection;
//! every block turns that row into its own latent (`attn_kv_a_mqa`, the RMS
//! gain `attn_kv_a_norm` on the first `kv_lora_rank` values, the rope on the
//! last `rope.dimension_count`) at the row's position. A block of
//! `block_size` rows then runs through `block_count` dense stages: row 0 is
//! the anchor token's embedding and row `k` the file's `mask_embd` row
//! `k - 1`, so the block's masks are the file's own and no token's
//! embedding. A stage is MLA with no query latent (`attn_q` projects the
//! normed row to every head's `key_length_mla` values; `attn_k_b` and
//! `attn_v_b` are the latent's per-head up-projections, the query absorbed
//! against the first), attending the context rows inside the sliding window
//! and every row of the block (the block is not causal), then a SiLU-gated
//! dense FFN. `output_norm` precedes the target's output projection, whose
//! logits take the target's soft cap. The head is a Markov recurrence:
//! `markov_w1` embeds the previous row's id (row 0's is the anchor token);
//! `joint_proj` over `[state | markov_w1[prev] | hidden]` gives three
//! `markov_rank`-wide chunks — a gate, a candidate and the vector the
//! correction reads — the state becomes `gate * state + (1 - gate) *
//! tanh(candidate)`, and `markov_w2` projects `tanh(third chunk)` to the
//! vocabulary as a bias on that row's logits; the row's id is the argmax of
//! the biased logits and the next row's `prev`. With `sample_from_anchor` the
//! anchor row drafts the first id. The token embedding and the output
//! projection are the target's and are not in this file. The reference is
//! llama.cpp's `src/models/dflash.cpp` (`graph_mistral_large4`,
//! `build_dspark_markov_head`).
//!
//! This module reads the file and does not run it: [`Hparams::read`] takes
//! every key it needs, [`names`] owns the model-level tensor names and
//! [`tensors`] states the shape and type of every tensor; [`check`] holds the
//! file to that statement and [`fit`] holds it to a target. [`card_bytes`]
//! states what the draft takes on its card before any load, in the common
//! draft's terms ([`DraftCardBytes`]). The taps are the file's `target_layers`
//! as they stand.

use std::num::NonZeroU64;

use gguf::{GgmlType, Split, Value};
use runtime::dspark::{DraftVocab, DsparkShape, HeadKind};

use super::hparams::{Hparams as Target, YarnKeys};
use super::names as stem;
use crate::arch::dspark::{DraftCardBytes, SCRATCH_ALLOWANCE, format_buffers};
use crate::arch::names::{blk, output};
use crate::arch::{
    DFLASH, is_dflash, meta_arr, meta_bool, meta_f32, meta_str, meta_usize, metadata, n_vocab,
    positive, unread_keys_of,
};
use crate::placement::{CardFormat, PlacementError, allocator_bytes};

/// Rows one block pass and one append group take, the Q8_0 row kernel's
/// column bound: each stage's ring keeps this many rows past the window.
pub const PASS_ROWS: usize = 8;

/// The key whose presence makes a `dflash` file an MLA draft: V4.1's draft
/// has a query latent and no kv latent rank, DFlash2's neither.
const KV_LORA: &str = "attention.kv_lora_rank";

/// Every `dflash.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "vocab_size",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.kv_lora_rank",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "rope.scaling.yarn_log_multiplier",
    "final_logit_softcapping",
    "block_size",
    "target_layers",
    "sample_from_anchor",
    "has_confidence_head",
];

/// Whether `split` is an MLA DSpark draft: a `dflash` file with a kv latent
/// rank and a Markov head.
pub fn is_draft(split: &Split) -> bool {
    is_dflash(split)
        && split.value(&split.arch_key(KV_LORA)).is_some()
        && split.find(&names::markov_w1()).is_some()
}

/// The hyperparameters of a Mistral Large 4 DSpark draft file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: the draft's stages.
    pub n_layer: usize,
    /// `embedding_length`: the draft's width, the target's too.
    pub n_embd: usize,
    /// `feed_forward_length`: the dense FFN's width.
    pub n_ff: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.kv_lora_rank`: the latent's values.
    pub kv_lora: usize,
    /// `attention.key_length_mla`: values per head, unrotated and rotated
    /// together.
    pub head_k: usize,
    /// `attention.value_length_mla`: value values per head.
    pub head_v: usize,
    /// `rope.dimension_count`: the rotated tail of the latent row and of each
    /// head's key.
    pub rope_dims: usize,
    /// `rope.freq_base`.
    pub rope_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// The rope's YaRN keys. The draft rotates with no LeRoPE factors.
    pub yarn: YarnKeys,
    /// `attention.sliding_window`: the past positions a block row attends.
    pub window: usize,
    /// `final_logit_softcapping`: the bound on the head's logits.
    pub softcap: f32,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `block_size`: rows of one block, the anchor and `block_size - 1`
    /// masks.
    pub block_size: usize,
    /// `target_layers`: the target layers whose rows `fc` reads, in the order
    /// their rows are concatenated.
    pub target_layers: Vec<usize>,
    /// `sample_from_anchor`: whether the anchor row drafts the first id.
    pub sample_from_anchor: bool,
    /// `vocab_size`, else the token list's length: the target's ids.
    pub n_vocab: usize,
    /// `markov_w1`'s row length: the Markov recurrence's rank, which no key
    /// states.
    pub markov_rank: usize,
}

impl Hparams {
    /// Every hyperparameter from `split`'s header. A missing or malformed
    /// key, a file of another draft, or a value this reader does not serve (a
    /// confidence head, another rope kind, a key length the latent and the
    /// rope do not sum to, a block of one row, a layer list that is empty or
    /// not increasing) is an error naming the key or tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        if !is_draft(split) {
            return Err(PlacementError::Metadata {
                key: "architecture".to_string(),
                detail: format!(
                    "is {:?} without {} and {}, not a {DFLASH:?} MLA DSpark draft",
                    split.architecture(),
                    split.arch_key(KV_LORA),
                    names::markov_w1()
                ),
            });
        }
        let kv_lora = positive(split, KV_LORA)?;
        let rope_dims = positive(split, "rope.dimension_count")?;
        let key_length = positive(split, "attention.key_length")?;
        if key_length != kv_lora + rope_dims {
            let detail = format!(
                "is {key_length}, the latent and the rope sum to {}",
                kv_lora + rope_dims
            );
            return Err(metadata(split, "attention.key_length", detail));
        }
        let value_length = meta_usize(split, "attention.value_length")?;
        if value_length != kv_lora {
            let detail = format!("is {value_length}, the latent is {kv_lora} values");
            return Err(metadata(split, "attention.value_length", detail));
        }
        let kv_heads = meta_usize(split, "attention.head_count_kv")?;
        if kv_heads != 1 {
            let detail = format!("is {kv_heads}; the latent is one head's");
            return Err(metadata(split, "attention.head_count_kv", detail));
        }
        let head_k = positive(split, "attention.key_length_mla")?;
        if head_k <= rope_dims {
            let detail = format!("is {head_k}, not past the rope's {rope_dims}");
            return Err(metadata(split, "attention.key_length_mla", detail));
        }
        let scaling = meta_str(split, "rope.scaling.type")?;
        if scaling != "yarn" {
            let detail = format!("is {scaling:?}; the draft rotates with YaRN");
            return Err(metadata(split, "rope.scaling.type", detail));
        }
        if meta_bool(split, "has_confidence_head")? {
            let detail = "is true; the draft reads no confidence projection";
            return Err(metadata(split, "has_confidence_head", detail));
        }
        let block_size = positive(split, "block_size")?;
        if block_size < 2 {
            return Err(metadata(split, "block_size", "is 1: no row to draft"));
        }
        let yarn = YarnKeys {
            factor: meta_f32(split, "rope.scaling.factor")?,
            orig_ctx: positive(split, "rope.scaling.original_context_length")?,
            beta_fast: meta_f32(split, "rope.scaling.yarn_beta_fast")?,
            beta_slow: meta_f32(split, "rope.scaling.yarn_beta_slow")?,
            log_mul: meta_f32(split, "rope.scaling.yarn_log_multiplier")?,
        };
        let magnitude = yarn.magnitude();
        if magnitude != 1.0 {
            let detail = format!(
                "is {}, a net magnitude of {magnitude}; the rope applies none",
                yarn.log_mul
            );
            return Err(metadata(split, "rope.scaling.yarn_log_multiplier", detail));
        }
        Ok(Hparams {
            n_layer: positive(split, "block_count")?,
            n_embd: positive(split, "embedding_length")?,
            n_ff: positive(split, "feed_forward_length")?,
            n_head: positive(split, "attention.head_count")?,
            kv_lora,
            head_k,
            head_v: positive(split, "attention.value_length_mla")?,
            rope_dims,
            rope_base: meta_f32(split, "rope.freq_base")?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            yarn,
            window: positive(split, "attention.sliding_window")?,
            softcap: meta_f32(split, "final_logit_softcapping")?,
            n_ctx_train: positive(split, "context_length")?,
            block_size,
            target_layers: target_layers(split)?,
            sample_from_anchor: meta_bool(split, "sample_from_anchor")?,
            n_vocab: n_vocab(split)?,
            markov_rank: markov_rank(split)?,
        })
    }

    /// The unrotated values of a head's key: `key_length_mla` less the rope.
    #[must_use]
    pub fn nope(&self) -> usize {
        self.head_k - self.rope_dims
    }

    /// Values of one context row before `fc`: every tap's row, in
    /// `target_layers` order.
    #[must_use]
    pub fn fc_in(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }

    /// The width of `joint_proj`'s input: the state, the previous id's Markov
    /// embedding and the row's hidden state.
    #[must_use]
    pub fn joint_in(&self) -> usize {
        self.n_embd + 2 * self.markov_rank
    }
}

/// The geometry the common draft owner runs this file at: a block of
/// `block_size` rows whose row 0 drafts (`sample_from_anchor`, which
/// [`Hparams::read`] takes as the file states it and the shape's check holds
/// the recurrent head to), the taps as the file lists them, the target's
/// vocabulary, the sliding window and the recurrent Markov head of
/// `markov_rank`. The file names no mask token: its block rows `1..` are
/// `mask_embd`'s rows, which the layers embed themselves (the owner's
/// `EmbedSource::Target`, the only source the MLA layers take), so the
/// owner never reads `mask_id`, which is 0. Pass the result to
/// `DsparkShape::check` before use.
#[must_use]
pub fn shape(hp: &Hparams) -> DsparkShape {
    DsparkShape {
        block: hp.block_size,
        row0_drafts: hp.sample_from_anchor,
        taps: hp.target_layers.clone(),
        hidden: hp.n_embd,
        vocab: DraftVocab::Full { n: hp.n_vocab },
        mask_id: 0,
        window: Some(hp.window),
        head: HeadKind::MarkovRnn {
            rank: hp.markov_rank,
        },
    }
}

/// The `dflash.` keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, READ)
}

/// `target_layers`: non-empty and strictly increasing, one layer index each.
fn target_layers(split: &Split) -> Result<Vec<usize>, PlacementError> {
    let key = "target_layers";
    let items: &[Value] = meta_arr(split, key)?;
    let mut layers: Vec<usize> = Vec::new();
    for (i, v) in items.iter().enumerate() {
        let l = v
            .as_unsigned()
            .and_then(|l| usize::try_from(l).ok())
            .ok_or_else(|| metadata(split, key, format!("entry {i} is not a layer")))?;
        if layers.last().is_some_and(|&p| p >= l) {
            let detail = format!("entry {i} ({l}) does not follow the one before");
            return Err(metadata(split, key, detail));
        }
        layers.push(l);
    }
    if layers.is_empty() {
        return Err(metadata(split, key, "is empty"));
    }
    Ok(layers)
}

/// `markov_w1`'s row length: the recurrence's rank.
fn markov_rank(split: &Split) -> Result<usize, PlacementError> {
    let name = names::markov_w1();
    let (_, t) = split.find(&name).ok_or_else(|| PlacementError::Tensor {
        name: name.clone(),
        detail: "is not in the file, and it alone states the Markov rank".to_string(),
    })?;
    usize::try_from(t.dims[0]).map_err(|_| PlacementError::Tensor {
        name,
        detail: format!("row length {} does not fit usize", t.dims[0]),
    })
}

/// The model-level tensor names of the draft. A stage's are the target
/// family's stems under `blk.{l}.` ([`tensors`]).
pub mod names {
    /// `fc.weight`: the projection of the concatenated target rows to the
    /// draft's width.
    #[must_use]
    pub fn fc() -> String {
        "fc.weight".to_string()
    }

    /// `enc.output_norm.weight`: the RMS gain on `fc`'s output.
    #[must_use]
    pub fn enc_output_norm() -> String {
        "enc.output_norm.weight".to_string()
    }

    /// `output_norm.weight`: the draft's own final norm, before the target's
    /// output projection.
    #[must_use]
    pub fn output_norm() -> String {
        "output_norm.weight".to_string()
    }

    /// `mask_embd.weight`: the embeddings of the block's mask rows, one row
    /// per mask.
    #[must_use]
    pub fn mask_embd() -> String {
        "mask_embd.weight".to_string()
    }

    /// `markov_w1.weight`: one rank-sized row per token, the previous id's
    /// Markov embedding.
    #[must_use]
    pub fn markov_w1() -> String {
        "markov_w1.weight".to_string()
    }

    /// `markov_w2.weight`: the projection of the correction vector to the
    /// vocabulary, added to the row's logits.
    #[must_use]
    pub fn markov_w2() -> String {
        "markov_w2.weight".to_string()
    }

    /// `joint_proj.weight`: the recurrence's gate, candidate and correction
    /// vector from `[state | markov_w1[prev] | hidden]`.
    #[must_use]
    pub fn joint_proj() -> String {
        "joint_proj.weight".to_string()
    }

    /// `joint_proj.bias`.
    #[must_use]
    pub fn joint_proj_bias() -> String {
        "joint_proj.bias".to_string()
    }
}

/// A tensor the draft reads: its name, shape in ggml's `ne` order, and the
/// type this reader takes it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: Ty,
}

impl Tensor {
    /// The bytes the file holds for the tensor: f32 values, or Q8_0 blocks of
    /// 32 values in 34 bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let n: u64 = self.dims.iter().product();
        match self.ty {
            Ty::F32 => 4 * n,
            Ty::Q8_0 => n / 32 * 34,
        }
    }
}

/// The types a tensor may have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A gain or a bias: f32.
    F32,
    /// A matrix or a table: Q8_0, whose first dim is a multiple of 32.
    Q8_0,
}

impl Ty {
    fn takes(self, ty: GgmlType) -> bool {
        match self {
            Ty::F32 => ty == GgmlType::F32,
            Ty::Q8_0 => ty == GgmlType::Q8_0,
        }
    }
}

/// Every tensor of the draft, stage by stage and then the model-level ones,
/// with the shapes `hp` implies.
#[must_use]
pub fn tensors(hp: &Hparams) -> Vec<Tensor> {
    let u = |v: usize| v as u64;
    let embd = u(hp.n_embd);
    let heads = u(hp.n_head);
    let (latent, nope) = (u(hp.kv_lora), u(hp.nope()));
    let t = |name: String, dims: Vec<u64>, ty: Ty| Tensor { name, dims, ty };
    let mut out = Vec::new();
    for l in 0..hp.n_layer {
        let b = |s: &str| blk(l, s);
        out.extend([
            t(b(stem::ATTN_NORM), vec![embd], Ty::F32),
            t(b(stem::ATTN_Q), vec![embd, heads * u(hp.head_k)], Ty::Q8_0),
            t(
                b(stem::ATTN_KV_A_MQA),
                vec![embd, latent + u(hp.rope_dims)],
                Ty::Q8_0,
            ),
            t(b(stem::ATTN_KV_A_NORM), vec![latent], Ty::F32),
            t(b(stem::ATTN_K_B), vec![nope, latent, heads], Ty::Q8_0),
            t(
                b(stem::ATTN_V_B),
                vec![latent, u(hp.head_v), heads],
                Ty::Q8_0,
            ),
            t(
                b(stem::ATTN_OUTPUT),
                vec![heads * u(hp.head_v), embd],
                Ty::Q8_0,
            ),
            t(b(stem::FFN_NORM), vec![embd], Ty::F32),
            t(b(stem::FFN_GATE), vec![embd, u(hp.n_ff)], Ty::Q8_0),
            t(b(stem::FFN_UP), vec![embd, u(hp.n_ff)], Ty::Q8_0),
            t(b(stem::FFN_DOWN), vec![u(hp.n_ff), embd], Ty::Q8_0),
        ]);
    }
    let (rank, vocab) = (u(hp.markov_rank), u(hp.n_vocab));
    out.extend([
        t(names::fc(), vec![u(hp.fc_in()), embd], Ty::Q8_0),
        t(names::enc_output_norm(), vec![embd], Ty::F32),
        t(names::output_norm(), vec![embd], Ty::F32),
        t(
            names::mask_embd(),
            vec![embd, u(hp.block_size - 1)],
            Ty::Q8_0,
        ),
        t(names::markov_w1(), vec![rank, vocab], Ty::Q8_0),
        t(names::markov_w2(), vec![rank, vocab], Ty::Q8_0),
        t(
            names::joint_proj(),
            vec![u(hp.joint_in()), 3 * rank],
            Ty::Q8_0,
        ),
        t(names::joint_proj_bias(), vec![3 * rank], Ty::F32),
    ]);
    out
}

/// Hold `split` to [`tensors`]: every stated tensor present with its shape
/// and type, and no tensor the statement does not name. Each mismatch is an
/// error naming the tensor.
pub fn check(split: &Split, hp: &Hparams) -> Result<(), PlacementError> {
    let stated = tensors(hp);
    for s in &stated {
        let (_, t) = split.find(&s.name).ok_or_else(|| PlacementError::Tensor {
            name: s.name.clone(),
            detail: "is not in the file".to_string(),
        })?;
        if t.dims != s.dims {
            return Err(PlacementError::Tensor {
                name: s.name.clone(),
                detail: format!(
                    "has dims {:?}; the hyperparameters imply {:?}",
                    t.dims, s.dims
                ),
            });
        }
        if !s.ty.takes(t.ty) {
            return Err(PlacementError::Tensor {
                name: s.name.clone(),
                detail: format!("is {:?}; this reader takes {:?}", t.ty, s.ty),
            });
        }
    }
    if let Some((_, t)) = split
        .iter_tensors()
        .find(|(_, t)| !stated.iter().any(|s| s.name == t.name))
    {
        return Err(PlacementError::Tensor {
            name: t.name.clone(),
            detail: "is in the file but not in the MLA DSpark statement".to_string(),
        });
    }
    Ok(())
}

/// Hold the draft `hp` to the target `target` it would draft for: the same
/// stream width, the same ids and the same logit soft cap, and every tapped
/// layer inside the target (the entry `n_layer` is the row the last layer
/// leaves). The first disagreement is refused by name.
pub fn fit(hp: &Hparams, target: &Target) -> Result<(), PlacementError> {
    let off = |suffix: &str, detail: String| PlacementError::Metadata {
        key: format!("{DFLASH}.{suffix}"),
        detail,
    };
    if hp.n_embd != target.n_embd {
        return Err(off(
            "embedding_length",
            format!("is {}, the target's stream is {}", hp.n_embd, target.n_embd),
        ));
    }
    if hp.n_vocab != target.n_vocab {
        return Err(off(
            "vocab_size",
            format!("is {}, the target has {} ids", hp.n_vocab, target.n_vocab),
        ));
    }
    if hp.softcap != target.softcap {
        return Err(off(
            "final_logit_softcapping",
            format!("is {}, the target's is {}", hp.softcap, target.softcap),
        ));
    }
    if let Some(&l) = hp.target_layers.iter().find(|&&l| l > target.n_layer) {
        return Err(off(
            "target_layers",
            format!("names layer {l}, the target has {} layers", target.n_layer),
        ));
    }
    Ok(())
}

/// The draft's resident bytes on its card from the two files' headers,
/// before any load: `draft` held to its statement ([`check`]) and to the
/// target ([`fit`]), the target's `output` read from `target`, then
/// [`card_bytes_of`].
pub fn card_bytes(
    draft: &Split,
    target: &Split,
    granule: NonZeroU64,
) -> Result<DraftCardBytes, PlacementError> {
    let hp = Hparams::read(draft)?;
    check(draft, &hp)?;
    fit(&hp, &Target::read(target)?)?;
    let name = output();
    let (_, head) = target.find(&name).ok_or_else(|| PlacementError::Tensor {
        name: name.clone(),
        detail: "is not in the target file".to_string(),
    })?;
    card_bytes_of(&hp, &head.dims, head.ty, granule)
}

/// The draft's resident bytes on a card of allocation granule `granule`,
/// from its hyperparameters `hp` and the target's `output` of dims `head` and
/// type `head_ty` (which the head's row kernel reads as Q6_K): the load's
/// buffers in its allocation order (the stages' weights, `fc`, the head's
/// norm, the target's `output` copied to the card, the Markov tables, then
/// the rings: [`tensors`] in order, less `mask_embd`, whose rows sit in the
/// block's input buffer) through the plan's allocator model, and the scratch:
/// the working buffers ([`working_bytes`]) beside [`SCRATCH_ALLOWANCE`]. A ring
/// does not grow with the serving context: it holds the window and
/// [`PASS_ROWS`] rows of `kv_lora + rope` f16 values. A type the load has no
/// card buffers for is refused by name.
pub fn card_bytes_of(
    hp: &Hparams,
    head: &[u64],
    head_ty: GgmlType,
    granule: NonZeroU64,
) -> Result<DraftCardBytes, PlacementError> {
    let rows_of = |dims: &[u64]| dims.iter().skip(1).product::<u64>();
    let mut weights: Vec<u64> = Vec::new();
    for t in tensors(hp) {
        if t.name == names::mask_embd() {
            continue;
        }
        let (format, ty, rows) = match t.ty {
            Ty::F32 => (CardFormat::F32, GgmlType::F32, 1),
            Ty::Q8_0 => (CardFormat::Q8_0Planes, GgmlType::Q8_0, rows_of(&t.dims)),
        };
        weights.extend(format_buffers(&t.name, Some(format), ty, &t.dims, rows)?);
        if t.name == names::output_norm() {
            let name = output();
            if head_ty != GgmlType::Q6_K {
                return Err(PlacementError::Tensor {
                    name,
                    detail: format!("is {head_ty}; the draft head's row kernel reads Q6_K"),
                });
            }
            weights.extend(format_buffers(
                &name,
                Some(CardFormat::KQuant),
                head_ty,
                head,
                rows_of(head),
            )?);
        }
    }
    let ring = (hp.window + PASS_ROWS) as u64 * (hp.kv_lora + hp.rope_dims) as u64 * 2;
    let rings = vec![ring; hp.n_layer];
    let (w, kv): (u64, u64) = (weights.iter().sum(), rings.iter().sum());
    let taken = allocator_bytes(granule, weights.into_iter().chain(rings));
    Ok(DraftCardBytes {
        weights: w,
        kv,
        rounding: taken - w - kv,
        scratch: SCRATCH_ALLOWANCE + working_bytes(hp),
    })
}

/// The working buffers the load allocates beside the weights and rings, from `hp`, each of
/// 4-byte elements: the common owner's (`x` of the window's rows, `fc_out`, the tapped rows'
/// inbox of the window's rows of `taps · n_embd` values and the block's one-word inbox), the
/// stages' scratch and images of [`PASS_ROWS`] rows, and the head's logits, Markov correction
/// and normed rows. The head's quantized rows and words are left to [`SCRATCH_ALLOWANCE`]: the
/// load holds its working buffers to the sum of both ([`DraftCardBytes::check_held`]). The
/// window's rows are the inbox's, not the ring's: it holds `window` rows of every tapped layer,
/// which is the draft's largest working buffer.
#[must_use]
pub fn working_bytes(hp: &Hparams) -> u64 {
    let u = |v: usize| v as u64;
    let (rows, e, heads) = (u(PASS_ROWS), u(hp.n_embd), u(hp.n_head));
    let (rope, lat, vocab) = (u(hp.rope_dims), u(hp.kv_lora), u(hp.n_vocab));
    let width = rope + lat;
    let capacity = u(hp.window).div_ceil(rows) * rows;
    let owner = capacity.max(rows) * e + rows * e + capacity * u(hp.target_layers.len()) * e + 1;
    let scratch = rows
        * (3 * e
            + 2 * heads * u(hp.head_k)
            + heads * width
            + 3 * width
            + heads * lat
            + heads * u(hp.head_v)
            + 3 * u(hp.n_ff))
        + 2 * rows * heads * rope;
    let images = rows * rope + rows + 1 + rows * e + capacity * rope + capacity;
    let head = rows * vocab + vocab + rows * e;
    4 * (owner + scratch + images + head)
}

#[cfg(test)]
mod tests {
    use gguf::{GgmlType, Split};

    use super::{
        Hparams, PASS_ROWS, SCRATCH_ALLOWANCE, Tensor, Ty, card_bytes_of, check, names, tensors,
        working_bytes,
    };
    use crate::arch::mistrallarge4::hparams::YarnKeys;
    use crate::arch::synthetic::{self, Ty as Written, V};

    /// The draft of `unsloth/Mistral-Large-4-1T-A52B-GGUF`, as its Q8_0 file's
    /// header states it.
    fn real() -> Hparams {
        Hparams {
            n_layer: 3,
            n_embd: 6144,
            n_ff: 24_576,
            n_head: 64,
            kv_lora: 512,
            head_k: 128,
            head_v: 128,
            rope_dims: 64,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            yarn: YarnKeys {
                factor: 2048.0,
                orig_ctx: 512,
                beta_fast: 32.0,
                beta_slow: 1.0,
                log_mul: 1.0,
            },
            window: 2048,
            softcap: 30.0,
            n_ctx_train: 1_048_576,
            block_size: 3,
            target_layers: vec![2, 19, 37, 55, 72],
            sample_from_anchor: true,
            n_vocab: 204_800,
            markov_rank: 512,
        }
    }

    /// A draft small enough to write as a file: every Q8_0 first dim a
    /// multiple of 32.
    fn tiny() -> Hparams {
        Hparams {
            n_layer: 2,
            n_embd: 64,
            n_ff: 64,
            n_head: 2,
            kv_lora: 32,
            head_k: 40,
            head_v: 32,
            rope_dims: 8,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            yarn: real().yarn,
            window: 16,
            softcap: 30.0,
            n_ctx_train: 4096,
            block_size: 3,
            target_layers: vec![1, 2],
            sample_from_anchor: true,
            n_vocab: 96,
            markov_rank: 32,
        }
    }

    /// The statement reproduces the 41 tensors of the real file with their
    /// dims, 12 of them f32, and the bytes its offsets add up to.
    #[test]
    fn statement_matches_the_real_header() {
        let hp = real();
        let t = tensors(&hp);
        assert_eq!(t.len(), 41);
        let dims = |name: &str| {
            t.iter()
                .find(|x| x.name == name)
                .map(|x| x.dims.clone())
                .unwrap_or_else(|| panic!("{name} not stated"))
        };
        assert_eq!(dims("fc.weight"), vec![30_720, 6144]);
        assert_eq!(dims("mask_embd.weight"), vec![6144, 2]);
        assert_eq!(dims("markov_w1.weight"), vec![512, 204_800]);
        assert_eq!(dims("markov_w2.weight"), vec![512, 204_800]);
        assert_eq!(dims("joint_proj.weight"), vec![7168, 1536]);
        assert_eq!(dims("joint_proj.bias"), vec![1536]);
        assert_eq!(dims("blk.0.attn_q.weight"), vec![6144, 8192]);
        assert_eq!(dims("blk.0.attn_kv_a_mqa.weight"), vec![6144, 576]);
        assert_eq!(dims("blk.1.attn_k_b.weight"), vec![64, 512, 64]);
        assert_eq!(dims("blk.2.attn_v_b.weight"), vec![512, 128, 64]);
        assert_eq!(dims("blk.2.attn_output.weight"), vec![8192, 6144]);
        assert_eq!(dims("blk.2.ffn_down.weight"), vec![24_576, 6144]);
        assert_eq!(t.iter().filter(|x| x.ty == Ty::F32).count(), 12);
        let bytes: u64 = t.iter().map(super::Tensor::bytes).sum();
        assert_eq!(bytes, 2_231_370_496);
        assert_eq!(hp.nope(), 64);
        assert_eq!(hp.fc_in(), 30_720);
        assert_eq!(hp.joint_in(), 7168);
    }

    /// The real draft's card bytes from its statement alone: the 41 tensors less `mask_embd` (whose
    /// rows sit in the block's input buffer), the target's Q6_K `output` (210 bytes a 256-value
    /// block, 204,800 rows of 24 blocks) between the head's norm and the Markov tables, three
    /// rings of `window + PASS_ROWS` f16 rows of `kv_lora + rope`, and the allocator model over
    /// the 69 weight buffers and the 3 rings in load order (Q8_0 as its two planes), run by hand
    /// [derived: 32,359,424 B of 2 MiB granules past the 3,263,549,440 + 7,105,536 B the buffers
    /// hold], and the scratch: the working buffers the owner, the stages and the head make for
    /// 8 rows (the tapped rows' inbox of 2,048 rows of 5 · 6,144 f32 the largest, 251,658,240 B;
    /// 316,768,296 B in all [derived]; the load's own count is 317,505,440 B [measured at the
    /// draft's load]) beside the 64 MiB allowance. No term reads the serving
    /// context.
    #[test]
    fn the_real_draft_takes_its_card_bytes() {
        let hp = real();
        let granule = crate::placement::workstation::GRANULE;
        let b = card_bytes_of(&hp, &[6144, 204_800], GgmlType::Q6_K, granule)
            .unwrap_or_else(|e| panic!("the real draft: {e}"));
        let file: u64 = tensors(&hp)
            .iter()
            .filter(|t| t.name != names::mask_embd())
            .map(Tensor::bytes)
            .sum();
        let head = 210 * (6144 / 256) * 204_800;
        assert_eq!(b.weights, file + head);
        assert_eq!(b.weights, 3_263_549_440);
        assert_eq!(b.kv, 3 * (2048 + PASS_ROWS as u64) * (512 + 64) * 2);
        assert_eq!(b.kv, 7_105_536);
        assert_eq!(b.rounding, 32_359_424);
        assert_eq!(b.buffers() % granule.get(), 0);
        assert_eq!(working_bytes(&hp), 316_768_296);
        assert_eq!(b.scratch, SCRATCH_ALLOWANCE + 316_768_296);
        assert_eq!(b.total(), 3_686_891_560);
    }

    /// The tiny draft: every buffer shares one granule, so the allocator takes one and the
    /// rounding is what the buffers leave of it; the head's `output` is 96 rows of two Q6_K blocks
    /// (420 bytes, 105 words, a row).
    #[test]
    fn a_small_draft_shares_one_granule() {
        let hp = tiny();
        let granule = crate::placement::workstation::GRANULE;
        let b = card_bytes_of(&hp, &[512, 96], GgmlType::Q6_K, granule)
            .unwrap_or_else(|e| panic!("the tiny draft: {e}"));
        let file: u64 = tensors(&hp)
            .iter()
            .filter(|t| t.name != names::mask_embd())
            .map(Tensor::bytes)
            .sum();
        assert_eq!(b.weights, file + 420 * 96);
        assert_eq!(b.kv, 2 * (16 + PASS_ROWS as u64) * (32 + 8) * 2);
        assert_eq!(b.buffers(), granule.get());
        assert_eq!(b.rounding, granule.get() - b.weights - b.kv);
    }

    /// A target `output` the head's row kernel does not read, or one off the Q6_K block, is
    /// refused naming the tensor.
    #[test]
    fn a_head_the_row_kernel_cannot_read_is_named() {
        let hp = real();
        let granule = crate::placement::workstation::GRANULE;
        let name = crate::arch::names::output();
        let q8 = card_bytes_of(&hp, &[6144, 204_800], GgmlType::Q8_0, granule)
            .expect_err("a Q8_0 head")
            .to_string();
        assert!(q8.contains(&name) && q8.contains("Q6_K"), "{q8}");
        let off = card_bytes_of(&hp, &[6000, 204_800], GgmlType::Q6_K, granule)
            .expect_err("a head off the block")
            .to_string();
        assert!(off.contains(&name), "{off}");
    }

    /// The real file's shape: a block of 3 rows drafting from row 0, the
    /// file's taps and window, the recurrent head of rank 512; it checks.
    #[test]
    fn the_real_shape_drafts_three_ids_from_row_zero() {
        let s = super::shape(&real());
        assert_eq!(s.check(), Ok(()));
        assert_eq!((s.ids_per_block(), s.rows(3), s.first_row()), (3, 3, 0));
        assert_eq!(s.taps, vec![2, 19, 37, 55, 72]);
        assert_eq!(s.window, Some(2048));
        assert_eq!(s.head, runtime::dspark::HeadKind::MarkovRnn { rank: 512 });
    }

    /// The tiny draft's statement written out as a file and read back: the
    /// keys give the hyperparameters, the tensors pass the check.
    fn tiny_file(
        tag: &str,
        tensors_of: &Hparams,
        edit: impl FnOnce(&mut Vec<(&'static str, V)>),
    ) -> std::path::PathBuf {
        let hp = tiny();
        let mut kv: Vec<(&'static str, V)> = vec![
            ("block_count", V::U32(hp.n_layer as u32)),
            ("context_length", V::U32(hp.n_ctx_train as u32)),
            ("embedding_length", V::U32(hp.n_embd as u32)),
            ("feed_forward_length", V::U32(hp.n_ff as u32)),
            ("vocab_size", V::U32(hp.n_vocab as u32)),
            ("attention.head_count", V::U32(hp.n_head as u32)),
            ("attention.head_count_kv", V::U32(1)),
            (
                "attention.key_length",
                V::U32((hp.kv_lora + hp.rope_dims) as u32),
            ),
            ("attention.value_length", V::U32(hp.kv_lora as u32)),
            ("attention.key_length_mla", V::U32(hp.head_k as u32)),
            ("attention.value_length_mla", V::U32(hp.head_v as u32)),
            ("attention.kv_lora_rank", V::U32(hp.kv_lora as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(hp.rms_eps)),
            ("attention.sliding_window", V::U32(hp.window as u32)),
            ("rope.dimension_count", V::U32(hp.rope_dims as u32)),
            ("rope.freq_base", V::F32(hp.rope_base)),
            ("rope.scaling.type", V::Str("yarn")),
            ("rope.scaling.factor", V::F32(hp.yarn.factor)),
            (
                "rope.scaling.original_context_length",
                V::U32(hp.yarn.orig_ctx as u32),
            ),
            ("rope.scaling.yarn_beta_fast", V::F32(hp.yarn.beta_fast)),
            ("rope.scaling.yarn_beta_slow", V::F32(hp.yarn.beta_slow)),
            ("rope.scaling.yarn_log_multiplier", V::F32(hp.yarn.log_mul)),
            ("final_logit_softcapping", V::F32(hp.softcap)),
            ("block_size", V::U32(hp.block_size as u32)),
            ("target_layers", V::I32s(vec![1, 2])),
            ("sample_from_anchor", V::Bool(true)),
            ("has_confidence_head", V::Bool(false)),
        ];
        edit(&mut kv);
        let typed: Vec<(String, Vec<u64>, Written)> = tensors(tensors_of)
            .into_iter()
            .map(|t| {
                let ty = match t.ty {
                    Ty::F32 => Written::F32,
                    Ty::Q8_0 => Written::Q8_0,
                };
                (t.name, t.dims, ty)
            })
            .collect();
        synthetic::header_typed(tag, "dflash", &kv, &[], &typed)
    }

    /// A file written from the statement reads back to the hyperparameters it
    /// was written from and passes the check; the rank comes from
    /// `markov_w1`, no key.
    #[test]
    fn a_file_of_the_statement_reads_and_checks() {
        let path = tiny_file("ml4dspark-ok", &tiny(), |_| {});
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let hp = Hparams::read(&split).expect("read the keys");
        assert_eq!(hp, tiny());
        check(&split, &hp).expect("the statement's own file checks");
    }

    /// A key the reader needs, absent, is refused naming the key.
    #[test]
    fn a_missing_key_is_named() {
        let path = tiny_file("ml4dspark-nokey", &tiny(), |kv| {
            kv.retain(|(k, _)| *k != "block_size");
        });
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let err = Hparams::read(&split)
            .expect_err("no block_size")
            .to_string();
        assert!(err.contains("block_size"), "{err}");
    }

    /// A file whose tensor has other dims than the keys imply is refused
    /// naming it: here the Markov tables are one row short of the vocabulary.
    #[test]
    fn a_tensor_off_the_statement_is_named() {
        let mut other = tiny();
        other.n_vocab = 64;
        let path = tiny_file("ml4dspark-dims", &other, |_| {});
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let mut hp = tiny();
        hp.markov_rank = 32;
        let err = check(&split, &hp).expect_err("other dims").to_string();
        assert!(err.contains(&names::markov_w1()), "{err}");
    }

    /// A confidence head is refused naming its key: the draft reads no
    /// confidence projection.
    #[test]
    fn a_confidence_head_is_refused() {
        let path = tiny_file("ml4dspark-conf", &tiny(), |kv| {
            kv.retain(|(k, _)| *k != "has_confidence_head");
            kv.push(("has_confidence_head", V::Bool(true)));
        });
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let err = Hparams::read(&split)
            .expect_err("a confidence head")
            .to_string();
        assert!(err.contains("has_confidence_head"), "{err}");
    }

    /// A latent that is not one head's, or a key length the latent and the
    /// rope do not sum to, is refused naming the key.
    #[test]
    fn a_latent_of_other_geometry_is_refused() {
        let path = tiny_file("ml4dspark-kvheads", &tiny(), |kv| {
            kv.retain(|(k, _)| *k != "attention.head_count_kv");
            kv.push(("attention.head_count_kv", V::U32(2)));
        });
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let err = Hparams::read(&split).expect_err("two kv heads").to_string();
        assert!(err.contains("attention.head_count_kv"), "{err}");
        let path = tiny_file("ml4dspark-keylen", &tiny(), |kv| {
            kv.retain(|(k, _)| *k != "attention.key_length");
            kv.push(("attention.key_length", V::U32(41)));
        });
        let split = Split::open(&path).expect("open the synthetic draft");
        let _ = std::fs::remove_file(&path);
        let err = Hparams::read(&split)
            .expect_err("a key length off")
            .to_string();
        assert!(err.contains("attention.key_length"), "{err}");
    }
}
