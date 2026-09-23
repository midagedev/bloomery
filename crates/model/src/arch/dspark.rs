//! `dspark` — the DSpark draft of DeepSeek-V4.1-Flash, the file whose
//! architecture string is `dflash`: three V4.1-shaped blocks (window-only
//! attention, hyper-connections, a routed MoE with MXFP4 experts and a shared
//! expert) behind `fc`, which projects the target's hidden states at
//! `target_layers`, and a head of its own norm plus the Markov correction
//! (`markov_w1`/`markov_w2`) and a confidence projection (`conf_proj`). The
//! token embedding and the output projection are the target's and are not in
//! this file.
//!
//! This module reads the file and does not run it: [`DraftHparams::read`]
//! takes every key with no defaults, [`names`] owns the draft's tensor names,
//! [`tensors`] states the shape and type of every tensor the draft reads, and
//! [`inventory`] checks the file against that statement and counts its bytes
//! by group, next to the two tensors borrowed from the target.
//!
//! Two keys the file carries are not read. The rope keys of YaRN
//! (`rope.scaling.*`) and `attention.compress_rope_freq_base`: every block's
//! `compress_ratios` entry is 0, and a window-only layer rotates with plain
//! rope at `rope.freq_base` (the reference `model.py` and ik both ignore the
//! scaling on such a layer). And the indexer keys (`attention.indexer.*`): no
//! block carries an indexer.

use gguf::{GgmlType, Split, Value};

use super::deepseek41::hparams::HyperConnections;
use crate::placement::{CardFormat, PlacementError};

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` (llama-hparams.h:18).
const IK_SQRT_SOFTPLUS: u64 = 4;

/// The token list whose length is the vocabulary when the file has no `vocab_size`.
const TOKENS: &str = "tokenizer.ggml.tokens";

/// The id of the mask token the draft fills a block's unknown positions with.
const MASK_TOKEN: &str = "tokenizer.ggml.mask_token_id";

/// The hyperparameters of the draft file.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftHparams {
    /// `block_count` — the draft's layers.
    pub n_layer: usize,
    /// `embedding_length` — one hyper-connection stream's width, the target's too.
    pub n_embd: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals.
    pub head_dim: usize,
    /// `attention.q_lora_rank`.
    pub q_lora_rank: usize,
    /// `attention.output_group_count`.
    pub o_groups: usize,
    /// `attention.output_lora_rank`.
    pub o_lora_rank: usize,
    /// `rope.dimension_count` — the rotated tail of each head.
    pub rope_dims: usize,
    /// `rope.freq_base` — plain rope, no scaling (see the module comment).
    pub rope_base: f32,
    /// `attention.sliding_window` — the past positions a block row attends.
    pub window: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `vocab_size`, else the length of `tokenizer.ggml.tokens`.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// The hyper-connections, read as V4.1 reads them.
    pub hc: HyperConnections,
    /// The router and the experts.
    pub experts: DraftExperts,
    /// `swiglu_clamp_exp`, one per layer.
    pub swiglu_limit: Vec<f32>,
    /// `swiglu_clamp_shexp`, one per layer.
    pub swiglu_limit_shared: Vec<f32>,
    /// `block_size` — the tokens one draft pass proposes a block of.
    pub block_size: usize,
    /// `target_layers` — the target layers whose hidden states `fc` reads, in
    /// the order their rows are concatenated.
    pub target_layers: Vec<usize>,
    /// `tokenizer.ggml.mask_token_id`.
    pub mask_token: u32,
    /// The Markov correction's rank: `markov_w1`'s row length. No key carries it.
    pub markov_rank: usize,
}

/// The draft's mixture of experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftExperts {
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_shared_count`.
    pub n_shared: usize,
    /// `expert_feed_forward_length`.
    pub ff: usize,
    /// `expert_weights_scale`.
    pub routed_scale: f32,
    /// `expert_weights_norm`.
    pub weights_norm: bool,
}

impl DraftHparams {
    /// Every hyperparameter from `split`'s header. A missing or malformed key,
    /// a file of another architecture, or a value the draft's graph does not
    /// have (a compressed layer, a hash-routed layer, a gating other than
    /// sqrt-softplus) is an error naming the key.
    pub fn read(split: &Split) -> Result<DraftHparams, PlacementError> {
        if !super::is_dflash(split) {
            return Err(PlacementError::Metadata {
                key: "architecture".to_string(),
                detail: format!("is {:?}, not {:?}", split.architecture(), super::DFLASH),
            });
        }
        let n_layer = usize_of(split, "block_count")?;
        let head_dim = usize_of(split, "attention.key_length")?;
        let value_length = usize_of(split, "attention.value_length")?;
        if value_length != head_dim {
            let detail = format!("is {value_length}, the key length is {head_dim}");
            return Err(metadata(split, "attention.value_length", detail));
        }
        let rope_dims = usize_of(split, "rope.dimension_count")?;
        if rope_dims > head_dim {
            let detail = format!("is {rope_dims}, more than a head's {head_dim}");
            return Err(metadata(split, "rope.dimension_count", detail));
        }
        window_only(split, n_layer)?;
        let hash = usize_of(split, "hash_layer_count")?;
        if hash != 0 {
            let detail = format!("is {hash}; the draft routes every layer by its scores");
            return Err(metadata(split, "hash_layer_count", detail));
        }
        let block_size = usize_of(split, "block_size")?;
        if block_size == 0 {
            return Err(metadata(split, "block_size", "is 0"));
        }
        Ok(DraftHparams {
            n_layer,
            n_embd: usize_of(split, "embedding_length")?,
            n_head: usize_of(split, "attention.head_count")?,
            n_head_kv: usize_of(split, "attention.head_count_kv")?,
            head_dim,
            q_lora_rank: usize_of(split, "attention.q_lora_rank")?,
            o_groups: usize_of(split, "attention.output_group_count")?,
            o_lora_rank: usize_of(split, "attention.output_lora_rank")?,
            rope_dims,
            rope_base: f32_of(split, "rope.freq_base")?,
            window: usize_of(split, "attention.sliding_window")?,
            rms_eps: f32_of(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: usize_of(split, "context_length")?,
            hc: HyperConnections {
                streams: usize_of(split, "hyper_connection.count")?,
                sinkhorn_iters: usize_of(split, "hyper_connection.sinkhorn_iterations")?,
                eps: f32_of(split, "hyper_connection.epsilon")?,
            },
            experts: DraftExperts::read(split)?,
            swiglu_limit: per_layer_f32(split, "swiglu_clamp_exp", n_layer)?,
            swiglu_limit_shared: per_layer_f32(split, "swiglu_clamp_shexp", n_layer)?,
            block_size,
            target_layers: target_layers(split)?,
            mask_token: mask_token(split)?,
            markov_rank: markov_rank(split)?,
        })
    }

    /// The width of one hyper-connection mix vector: `streams` pre weights,
    /// `streams` post weights and the `streams²` combine matrix.
    pub fn hc_mix(&self) -> usize {
        (2 + self.hc.streams) * self.hc.streams
    }
}

impl DraftExperts {
    fn read(split: &Split) -> Result<DraftExperts, PlacementError> {
        let gating = u64_of(split, "expert_gating_func")?;
        if gating != IK_SQRT_SOFTPLUS {
            let detail = format!("is {gating}; the draft's router scores with sqrt-softplus");
            return Err(metadata(split, "expert_gating_func", detail));
        }
        let n_expert = usize_of(split, "expert_count")?;
        let n_used = usize_of(split, "expert_used_count")?;
        if n_used == 0 || n_used > n_expert {
            let detail = format!("is {n_used}, of {n_expert} experts");
            return Err(metadata(split, "expert_used_count", detail));
        }
        let key = "expert_weights_norm";
        let weights_norm = split
            .value(&split.arch_key(key))
            .and_then(Value::as_bool)
            .ok_or_else(|| metadata(split, key, "is absent or not a bool"))?;
        Ok(DraftExperts {
            n_expert,
            n_used,
            n_shared: usize_of(split, "expert_shared_count")?,
            ff: usize_of(split, "expert_feed_forward_length")?,
            routed_scale: f32_of(split, "expert_weights_scale")?,
            weights_norm,
        })
    }
}

/// The draft's tensor names. The per-block names are V4.1's, the same
/// strings; the model-level ones below are the draft's own.
pub mod names {
    pub use crate::arch::deepseek41::names::{
        attn_kv, attn_kv_a_norm, attn_norm, attn_output_a, attn_output_b, attn_q_a, attn_q_a_norm,
        attn_q_b, attn_sinks, exp_probs_b, ffn_down_exps, ffn_down_shexp, ffn_gate_exps,
        ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up_exps, ffn_up_shexp, hc_attn_base,
        hc_attn_fn, hc_attn_scale, hc_ffn_base, hc_ffn_fn, hc_ffn_scale, output_norm,
    };

    /// `fc.weight` — the projection of the concatenated target hidden states
    /// to one stream's width.
    pub fn fc() -> String {
        "fc.weight".to_string()
    }

    /// `enc.output_norm.weight` — the RMS gain on `fc`'s output.
    pub fn enc_output_norm() -> String {
        "enc.output_norm.weight".to_string()
    }

    /// `markov_w1.weight` — one rank-sized row per token: the previous token's
    /// Markov embedding.
    pub fn markov_w1() -> String {
        "markov_w1.weight".to_string()
    }

    /// `markov_w2.weight` — the projection of that embedding to the vocabulary,
    /// added to the logits.
    pub fn markov_w2() -> String {
        "markov_w2.weight".to_string()
    }

    /// `conf_proj.weight` — the confidence score from the folded hidden state
    /// and the Markov embedding.
    pub fn conf_proj() -> String {
        "conf_proj.weight".to_string()
    }
}

/// What a draft tensor is for; the groups of the byte table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    /// A block's attention: norms, projections, sinks.
    Attention,
    /// A block's hyper-connection mixes.
    HyperConnection,
    /// A block's router: the gate and the selection bias.
    Router,
    /// A block's FFN norm and shared expert.
    SharedExpert,
    /// A block's routed experts.
    RoutedExperts,
    /// `fc` and its norm.
    Fc,
    /// The Markov correction.
    Markov,
    /// The confidence projection.
    Confidence,
    /// The draft's own final norm.
    OutputNorm,
}

/// A tensor the draft reads: its name, block, group, shape in ggml's `ne`
/// order, and the type this port reads it in.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftTensor {
    pub name: String,
    pub block: Option<usize>,
    pub group: Group,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
}

/// Every tensor of the draft, block by block and then the model-level ones,
/// with the shapes `hp` implies.
pub fn tensors(hp: &DraftHparams) -> Vec<DraftTensor> {
    let u = |v: usize| v as u64;
    let (embd, ff) = (u(hp.n_embd), u(hp.experts.ff));
    let q_lora = u(hp.q_lora_rank);
    let latent = u(hp.head_dim);
    let heads_latent = u(hp.n_head) * latent;
    let groups_out = u(hp.o_groups) * u(hp.o_lora_rank);
    let hc_in = u(hp.hc.streams) * embd;
    let mix = u(hp.hc_mix());
    let n_expert = u(hp.experts.n_expert);
    let shared_ff = ff * u(hp.experts.n_shared);
    let mut out = Vec::new();
    for b in 0..hp.n_layer {
        let mut t = |name: String, group: Group, dims: &[u64], ty: GgmlType| {
            out.push(DraftTensor {
                name,
                block: Some(b),
                group,
                dims: dims.to_vec(),
                ty,
            });
        };
        use GgmlType::{BF16, F32, MXFP4, Q8_0};
        use Group::{Attention, HyperConnection, RoutedExperts, Router, SharedExpert};
        t(names::attn_norm(b), Attention, &[embd], F32);
        t(names::attn_q_a(b), Attention, &[embd, q_lora], Q8_0);
        t(names::attn_q_a_norm(b), Attention, &[q_lora], F32);
        t(names::attn_q_b(b), Attention, &[q_lora, heads_latent], Q8_0);
        t(
            names::attn_kv(b),
            Attention,
            &[embd, u(hp.n_head_kv) * latent],
            Q8_0,
        );
        t(names::attn_kv_a_norm(b), Attention, &[latent], F32);
        t(names::attn_sinks(b), Attention, &[u(hp.n_head)], F32);
        let group_in = heads_latent / u(hp.o_groups);
        t(
            names::attn_output_a(b),
            Attention,
            &[group_in, groups_out],
            Q8_0,
        );
        t(
            names::attn_output_b(b),
            Attention,
            &[groups_out, embd],
            Q8_0,
        );
        t(names::hc_attn_fn(b), HyperConnection, &[hc_in, mix], F32);
        t(names::hc_attn_base(b), HyperConnection, &[mix], F32);
        t(names::hc_attn_scale(b), HyperConnection, &[3], F32);
        t(names::hc_ffn_fn(b), HyperConnection, &[hc_in, mix], F32);
        t(names::hc_ffn_base(b), HyperConnection, &[mix], F32);
        t(names::hc_ffn_scale(b), HyperConnection, &[3], F32);
        t(names::ffn_gate_inp(b), Router, &[embd, n_expert], BF16);
        t(names::exp_probs_b(b), Router, &[n_expert], F32);
        t(names::ffn_norm(b), SharedExpert, &[embd], F32);
        t(
            names::ffn_gate_shexp(b),
            SharedExpert,
            &[embd, shared_ff],
            Q8_0,
        );
        t(
            names::ffn_up_shexp(b),
            SharedExpert,
            &[embd, shared_ff],
            Q8_0,
        );
        t(
            names::ffn_down_shexp(b),
            SharedExpert,
            &[shared_ff, embd],
            Q8_0,
        );
        t(
            names::ffn_gate_exps(b),
            RoutedExperts,
            &[embd, ff, n_expert],
            MXFP4,
        );
        t(
            names::ffn_up_exps(b),
            RoutedExperts,
            &[embd, ff, n_expert],
            MXFP4,
        );
        t(
            names::ffn_down_exps(b),
            RoutedExperts,
            &[ff, embd, n_expert],
            MXFP4,
        );
    }
    let mut t = |name: String, group: Group, dims: &[u64], ty: GgmlType| {
        out.push(DraftTensor {
            name,
            block: None,
            group,
            dims: dims.to_vec(),
            ty,
        });
    };
    let features = u(hp.target_layers.len()) * embd;
    let rank = u(hp.markov_rank);
    let vocab = u(hp.n_vocab);
    t(names::fc(), Group::Fc, &[features, embd], GgmlType::Q8_0);
    t(names::enc_output_norm(), Group::Fc, &[embd], GgmlType::F32);
    t(
        names::markov_w1(),
        Group::Markov,
        &[rank, vocab],
        GgmlType::BF16,
    );
    t(
        names::markov_w2(),
        Group::Markov,
        &[rank, vocab],
        GgmlType::BF16,
    );
    t(
        names::conf_proj(),
        Group::Confidence,
        &[embd + rank, 1],
        GgmlType::BF16,
    );
    t(
        names::output_norm(),
        Group::OutputNorm,
        &[embd],
        GgmlType::F32,
    );
    out
}

/// How the draft uses a tensor of the target's file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Borrow {
    /// Rows are read per step (the block's token ids); nothing is resident.
    RowSource,
    /// Copied whole to the draft's card at load.
    Copied,
}

/// A target tensor the draft borrows.
#[derive(Clone, Debug, PartialEq)]
pub struct Borrowed {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub bytes: u64,
    pub borrow: Borrow,
}

/// One tensor the draft reads, against what the file holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub want: DraftTensor,
    /// The file's dims, type and bytes; `None` when the file lacks the name.
    pub found: Option<(Vec<u64>, GgmlType, u64)>,
}

impl Row {
    /// The file holds the tensor with the stated shape and type.
    pub fn holds(&self) -> bool {
        self.found
            .as_ref()
            .is_some_and(|(d, ty, _)| *d == self.want.dims && *ty == self.want.ty)
    }

    /// Its bytes on the draft's card. Every type but MXFP4 goes by the GPU
    /// loader's own rule ([`CardFormat`]: bf16 widened to f32, the rest at the
    /// file's size); MXFP4 has no card format there yet and counts at its file
    /// bytes, the native layout its kernels would read. `None` when the file
    /// does not hold the tensor as stated.
    pub fn card_bytes(&self) -> Option<u64> {
        if !self.holds() {
            return None;
        }
        let (dims, ty, bytes) = self.found.as_ref()?;
        match CardFormat::of(*ty) {
            None if *ty == GgmlType::MXFP4 => Some(*bytes),
            None => None,
            Some(f) => f.resident_bytes(*ty, dims[0], dims[1..].iter().product()),
        }
    }
}

/// The draft file against [`tensors`], and the target tensors it borrows.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftInventory {
    /// One row per tensor the draft reads, in [`tensors`]' order.
    pub rows: Vec<Row>,
    /// Tensors in the file that no row names.
    pub unread: Vec<String>,
    /// `token_embd` (row source) and `output` (copied) from the target.
    pub borrowed: Vec<Borrowed>,
}

impl DraftInventory {
    /// The first disagreement between the file and [`tensors`]: an absent or
    /// misshapen tensor, or one the draft does not read.
    pub fn check(&self) -> Result<(), PlacementError> {
        if !self.unread.is_empty() {
            return Err(PlacementError::Unclassified {
                names: self.unread.clone(),
            });
        }
        let Some(bad) = self.rows.iter().find(|r| !r.holds()) else {
            return Ok(());
        };
        let detail = match &bad.found {
            None => "is not in the file".to_string(),
            Some((d, ty, _)) => format!(
                "is {ty} {d:?}, the draft reads {} {:?}",
                bad.want.ty, bad.want.dims
            ),
        };
        Err(PlacementError::Tensor {
            name: bad.want.name.clone(),
            detail,
        })
    }

    /// The file bytes of `group`'s tensors (every block's).
    pub fn file_bytes(&self, group: Group) -> u64 {
        self.rows
            .iter()
            .filter(|r| r.want.group == group)
            .filter_map(|r| r.found.as_ref().map(|f| f.2))
            .sum()
    }

    /// The card bytes of `group`'s tensors ([`Row::card_bytes`]); `None` if
    /// any of them has none.
    pub fn card_bytes(&self, group: Group) -> Option<u64> {
        self.rows
            .iter()
            .filter(|r| r.want.group == group)
            .map(Row::card_bytes)
            .try_fold(0u64, |acc, b| acc.checked_add(b?))
    }
}

/// `draft` against [`tensors`] of `hp`, and the target's `token_embd` and
/// `output` read from `target`'s header. The target's tensors must be as wide
/// as the draft's stream and its vocabulary.
pub fn inventory(
    draft: &Split,
    hp: &DraftHparams,
    target: &Split,
) -> Result<DraftInventory, PlacementError> {
    let want = tensors(hp);
    let unread = draft
        .iter_tensors()
        .map(|(_, t)| &t.name)
        .filter(|n| !want.iter().any(|w| &w.name == *n))
        .cloned()
        .collect();
    let rows = want
        .into_iter()
        .map(|w| {
            let found = draft
                .find(&w.name)
                .map(|(_, t)| (t.dims.clone(), t.ty, t.nbytes));
            Row { want: w, found }
        })
        .collect();
    let target_names = [
        (super::deepseek41::names::token_embd(), Borrow::RowSource),
        (super::deepseek41::names::output(), Borrow::Copied),
    ];
    let mut borrowed = Vec::with_capacity(target_names.len());
    for (name, borrow) in target_names {
        let Some((_, t)) = target.find(&name) else {
            return Err(PlacementError::Tensor {
                name,
                detail: "is not in the target file".to_string(),
            });
        };
        let wide = [hp.n_embd as u64, hp.n_vocab as u64];
        if t.dims != wide {
            let detail = format!("is {:?} in the target, the draft needs {wide:?}", t.dims);
            return Err(PlacementError::Tensor { name, detail });
        }
        borrowed.push(Borrowed {
            name,
            dims: t.dims.clone(),
            ty: t.ty,
            bytes: t.nbytes,
            borrow,
        });
    }
    Ok(DraftInventory {
        rows,
        unread,
        borrowed,
    })
}

/// `attention.compress_ratios`, the first `n_layer` entries, must all be 0:
/// the draft's blocks attend their window and nothing compressed.
fn window_only(split: &Split, n_layer: usize) -> Result<(), PlacementError> {
    let key = "attention.compress_ratios";
    let items = arr_of(split, key)?;
    if items.len() < n_layer {
        let detail = format!("has {} entries for {n_layer} layers", items.len());
        return Err(metadata(split, key, detail));
    }
    if let Some((l, v)) = items[..n_layer]
        .iter()
        .enumerate()
        .find(|(_, v)| v.as_unsigned() != Some(0))
    {
        let detail = format!("layer {l} is {v:?}; every draft layer is window-only");
        return Err(metadata(split, key, detail));
    }
    Ok(())
}

/// The first `n_layer` entries of the per-layer float table `suffix`.
fn per_layer_f32(split: &Split, suffix: &str, n_layer: usize) -> Result<Vec<f32>, PlacementError> {
    let items = arr_of(split, suffix)?;
    if items.len() < n_layer {
        let detail = format!("has {} entries for {n_layer} layers", items.len());
        return Err(metadata(split, suffix, detail));
    }
    items[..n_layer]
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_f32()
                .ok_or_else(|| metadata(split, suffix, format!("entry {l} is not a float")))
        })
        .collect()
}

/// `target_layers`: non-empty and strictly increasing.
fn target_layers(split: &Split) -> Result<Vec<usize>, PlacementError> {
    let key = "target_layers";
    let mut layers: Vec<usize> = Vec::new();
    for (i, v) in arr_of(split, key)?.iter().enumerate() {
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

/// `tokenizer.ggml.mask_token_id`.
fn mask_token(split: &Split) -> Result<u32, PlacementError> {
    split
        .value(MASK_TOKEN)
        .and_then(Value::as_unsigned)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| PlacementError::Metadata {
            key: MASK_TOKEN.to_string(),
            detail: "is absent or not a token id".to_string(),
        })
}

/// `markov_w1`'s row length: the Markov correction's rank, which no key states.
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

/// The vocabulary size where ik takes it (llama-hparams.cpp:155):
/// `vocab_size` when the file carries it, else the token list's length.
fn n_vocab(split: &Split) -> Result<usize, PlacementError> {
    let key = "vocab_size";
    if split.value(&split.arch_key(key)).is_some() {
        return usize_of(split, key);
    }
    match split.value(TOKENS) {
        Some(Value::Array(tokens)) => Ok(tokens.len()),
        _ => Err(PlacementError::Metadata {
            key: TOKENS.to_string(),
            detail: format!(
                "is absent or not an array, and so is {}",
                split.arch_key(key)
            ),
        }),
    }
}

fn metadata(split: &Split, suffix: &str, detail: impl Into<String>) -> PlacementError {
    PlacementError::Metadata {
        key: split.arch_key(suffix),
        detail: detail.into(),
    }
}

fn u64_of(split: &Split, suffix: &str) -> Result<u64, PlacementError> {
    split
        .arch_get_u64(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not an unsigned integer"))
}

fn usize_of(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    let v = u64_of(split, suffix)?;
    usize::try_from(v).map_err(|_| metadata(split, suffix, format!("{v} does not fit usize")))
}

fn f32_of(split: &Split, suffix: &str) -> Result<f32, PlacementError> {
    split
        .arch_get_f32(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not a float"))
}

fn arr_of<'a>(split: &'a Split, suffix: &str) -> Result<&'a [Value], PlacementError> {
    split
        .arch_get_arr(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not an array"))
}
