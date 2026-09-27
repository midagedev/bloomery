//! Every `<architecture>.*` key the description reads, and each layer's kind
//! from the tensors it carries. Required keys are llama.cpp's required set; a
//! key it reads as optional keeps its reading of the absence, and every
//! default taken is recorded ([`Hparams::defaults`]) with the line that sets
//! it. The qwen4exp keys ([`Exp`]) are read for that variant only, and its
//! tensors are checked against its layers' kinds ([`Hparams::read`]).

use gguf::{Split, Value};

use crate::arch::{meta_arr, meta_f32, meta_u64, meta_usize, metadata, n_vocab, qwen35moe_variant};
use crate::placement::PlacementError;

/// llama.cpp's `LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX`.
const SOFTMAX: u64 = 1;

/// llama.cpp's `LLAMA_MAX_PLE_NGRAM` and `LLAMA_MAX_PLE_HEADS`
/// (`src/llama-hparams.h`).
const MAX_PLE_NGRAM: usize = 8;
const MAX_PLE_HEADS: usize = 64;

/// The family member a file is, by its `general.architecture`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// `qwen35moe`: Qwen3.5/3.6.
    Qwen35Moe,
    /// `qwen4exp`: Qwen3.8-Flash-Next — gated-residual hyper-connections and no
    /// block norm, a sigmoid GDN output gate, a mean-pool selector on the
    /// attention layers, a PLE site.
    Qwen4Exp,
}

impl Variant {
    /// The lines of the variant's llama.cpp builder that pass the router's
    /// softmax, renormalization and scale as constants.
    fn router_lines(self) -> &'static str {
        match self {
            Variant::Qwen35Moe => "qwen35moe.cpp:499-508",
            Variant::Qwen4Exp => "qwen4exp.cpp:992-1001",
        }
    }

    /// The lines that default the attention interval.
    fn interval_lines(self) -> &'static str {
        match self {
            Variant::Qwen35Moe => "qwen35moe.cpp:21-25",
            Variant::Qwen4Exp => "qwen4exp.cpp:128-134",
        }
    }
}

/// What a trunk layer mixes with, from the tensors it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `attn_qkv`: a gated delta-rule layer.
    DeltaRule,
    /// `attn_q`: a gated GQA layer.
    Attention,
}

/// The keys only a qwen4exp file carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Exp {
    /// `hyper_connection.count`: above 1.
    pub hc_streams: usize,
    /// `hyper_connection.low_rank`: the gates' bottleneck.
    pub hc_rank: usize,
    /// `attention.indexer.head_count`.
    pub idx_heads: usize,
    /// `attention.indexer.key_length`.
    pub idx_dim: usize,
    /// `attention.indexer.top_k`: tokens kept, whole blocks of every ratio.
    pub idx_top_k: usize,
    /// `attention.compress_ratios`, per layer: an attention layer's tokens per
    /// pooled key, 0 on a GDN layer.
    pub ratios: Vec<usize>,
    /// `None`: the file lists no PLE site.
    pub ple: Option<Ple>,
}

/// The PLE site's keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ple {
    /// `ple.layers`: its one layer, a GDN one.
    pub layer: usize,
    /// `ple.ngram_size`: n-grams of 2 to this many tokens.
    pub ngram: usize,
    /// `ple.heads_per_ngram`.
    pub heads_per_ngram: usize,
    /// `ple.conv_kernel`.
    pub conv: usize,
    /// `ple.eos_token_id`: the token that resets the window, not the tokenizer's end of text.
    pub eos: u32,
    /// `ple.image_token_id`; `None` when absent.
    pub image: Option<u32>,
    /// `embedding_length_per_layer_input`: values per table row.
    pub row: usize,
}

impl Ple {
    /// Table rows one token gathers: `heads_per_ngram` per n-gram size.
    #[must_use]
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
}

/// The hyperparameters of one file of the family.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    pub variant: Variant,
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, which the value length equals.
    pub head_dim: usize,
    /// `rope.dimension_count`: the rotated values of a head.
    pub rope_dims: usize,
    /// `rope.dimension_sections`: pairs per position axis.
    pub rope_sections: [u32; 4],
    /// `rope.freq_base`.
    pub rope_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`.
    pub expert_ff: usize,
    /// `expert_shared_feed_forward_length`; `None` when the file has no shared expert.
    pub shared_ff: Option<usize>,
    /// `ssm.conv_kernel`.
    pub conv: usize,
    /// `ssm.state_size`: values per key and value head.
    pub state: usize,
    /// `ssm.time_step_rank`: value heads.
    pub v_heads: usize,
    /// `ssm.group_count`: key heads.
    pub k_heads: usize,
    /// `full_attention_interval`.
    pub interval: usize,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// `Some` for a qwen4exp file, `None` for a qwen35moe one.
    pub exp: Option<Exp>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value the
    /// description cannot hold, or a layer whose tensors disagree with the
    /// interval is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let variant = qwen35moe_variant(split)?;
        let mut defaults = Vec::new();
        let n_layer = meta_usize(split, "block_count")?;
        if let Some(n) = optional_usize(split, "nextn_predict_layers")?
            && n > 0
        {
            return Err(metadata(
                split,
                "nextn_predict_layers",
                format!("is {n}; this reader reads no next-token layer"),
            ));
        }
        let n_embd = meta_usize(split, "embedding_length")?;
        let n_head = meta_usize(split, "attention.head_count")?;
        let n_head_kv = match optional_usize(split, "attention.head_count_kv")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "attention.head_count_kv = {n_head} (the query heads, llama-hparams.cpp)"
                ));
                n_head
            }
        };
        if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!("is {n_head_kv} for {n_head} query heads"),
            ));
        }
        let head_dim = match optional_usize(split, "attention.key_length")? {
            Some(v) => v,
            None => {
                let v = n_embd / n_head;
                defaults.push(format!(
                    "attention.key_length = {v} (embedding_length / head_count, llama-hparams.cpp)"
                ));
                v
            }
        };
        if let Some(v) = optional_usize(split, "attention.value_length")?
            && v != head_dim
        {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {v}, the key length {head_dim}; one width serves q, k and v"),
            ));
        }
        let rope_dims = match optional_usize(split, "rope.dimension_count")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "rope.dimension_count = {head_dim} (the head, llama-hparams.cpp)"
                ));
                head_dim
            }
        };
        if rope_dims > head_dim || rope_dims % 2 != 0 {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a head of {head_dim}"),
            ));
        }
        let rope_sections = sections(split)?;
        let conv = positive(split, "ssm.conv_kernel")?;
        let state = meta_usize(split, "ssm.state_size")?;
        let v_heads = meta_usize(split, "ssm.time_step_rank")?;
        let k_heads = meta_usize(split, "ssm.group_count")?;
        let inner = meta_usize(split, "ssm.inner_size")?;
        if k_heads == 0 || v_heads == 0 || !v_heads.is_multiple_of(k_heads) {
            return Err(metadata(
                split,
                "ssm.group_count",
                format!("is {k_heads} for {v_heads} value heads"),
            ));
        }
        if inner != v_heads * state {
            return Err(metadata(
                split,
                "ssm.inner_size",
                format!("is {inner}, not ssm.time_step_rank {v_heads} x ssm.state_size {state}"),
            ));
        }
        let interval = match optional_usize(split, "full_attention_interval")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "full_attention_interval = 4 ({})",
                    variant.interval_lines()
                ));
                4
            }
        };
        if interval == 0 {
            return Err(metadata(split, "full_attention_interval", "is 0"));
        }
        let kinds = kinds(split, n_layer, interval)?;
        let n_expert = meta_usize(split, "expert_count")?;
        let n_used = meta_usize(split, "expert_used_count")?;
        if n_expert == 0 || n_used == 0 || n_used > n_expert {
            return Err(metadata(
                split,
                "expert_used_count",
                format!("is {n_used} of expert_count {n_expert}"),
            ));
        }
        let expert_ff = expert_ff(split)?;
        refuse_unless(split, "expert_gating_func", |v| v.as_u64() == Some(SOFTMAX))?;
        refuse_unless(split, "expert_weights_norm", |v| v.as_bool() == Some(true))?;
        refuse_unless(split, "expert_weights_scale", |v| v.as_f32() == Some(1.0))?;
        for (key, value) in [
            ("expert_gating_func", "softmax"),
            ("expert_weights_norm", "true"),
            ("expert_weights_scale", "1"),
        ] {
            if split.value(&split.arch_key(key)).is_none() {
                defaults.push(format!("{key} = {value} ({})", variant.router_lines()));
            }
        }
        let n_vocab = n_vocab(split)?;
        let exp = match variant {
            Variant::Qwen35Moe => None,
            Variant::Qwen4Exp => Some(exp(split, n_embd, n_vocab, &kinds, &mut defaults)?),
        };
        let hp = Hparams {
            variant,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            rope_dims,
            rope_sections,
            rope_base: meta_f32(split, "rope.freq_base")?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab,
            n_ctx_train: meta_usize(split, "context_length")?,
            n_expert,
            n_used,
            expert_ff,
            shared_ff: optional_usize(split, "expert_shared_feed_forward_length")?,
            conv,
            state,
            v_heads,
            k_heads,
            interval,
            kinds,
            exp,
            defaults,
        };
        if let Some(exp) = &hp.exp {
            exp_tensors_agree(split, &hp.kinds, exp)?;
        }
        Ok(hp)
    }
}

/// `expert_feed_forward_length`, one width for every layer: the per-layer
/// array llama.cpp's qwen4exp loader also takes (`get_key_or_arr`,
/// qwen4exp.cpp:27) is refused by name.
fn expert_ff(split: &Split) -> Result<usize, PlacementError> {
    let key = "expert_feed_forward_length";
    if let Some(Value::Array(items)) = split.value(&split.arch_key(key)) {
        return Err(metadata(
            split,
            key,
            format!(
                "is a per-layer array of {} values; this reader reads one expert width",
                items.len()
            ),
        ));
    }
    match meta_usize(split, key)? {
        0 => Err(metadata(split, key, "is 0")),
        v => Ok(v),
    }
}

/// The qwen4exp keys (qwen4exp.cpp:26-147), each required where that loader
/// requires it, cross-checked with the layers' `kinds`.
fn exp(
    split: &Split,
    n_embd: usize,
    n_vocab: usize,
    kinds: &[Kind],
    defaults: &mut Vec<String>,
) -> Result<Exp, PlacementError> {
    let hc_streams = meta_usize(split, "hyper_connection.count")?;
    if hc_streams <= 1 {
        return Err(metadata(
            split,
            "hyper_connection.count",
            format!("is {hc_streams}; one stream has nothing to mix (qwen4exp.cpp:45-51)"),
        ));
    }
    let hc_rank = positive(split, "hyper_connection.low_rank")?;
    for key in [
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
    ] {
        if split.value(&split.arch_key(key)).is_some() {
            return Err(metadata(
                split,
                key,
                "is in the file beside hyper_connection.low_rank; gated-residual hyper-connections run no Sinkhorn",
            ));
        }
    }
    let idx_heads = positive(split, "attention.indexer.head_count")?;
    let idx_dim = positive(split, "attention.indexer.key_length")?;
    let idx_top_k = positive(split, "attention.indexer.top_k")?;
    let ratios = ratios(split, kinds, idx_top_k)?;
    let ple = ple(split, n_embd, n_vocab, kinds, defaults)?;
    Ok(Exp {
        hc_streams,
        hc_rank,
        idx_heads,
        idx_dim,
        idx_top_k,
        ratios,
        ple,
    })
}

/// `attention.compress_ratios`, required as a per-layer array: above 0 on an
/// attention layer (its pool), 0 on a GDN layer, and dividing `top_k`.
fn ratios(split: &Split, kinds: &[Kind], top_k: usize) -> Result<Vec<usize>, PlacementError> {
    let key = "attention.compress_ratios";
    let items = meta_arr(split, key)?;
    if items.len() != kinds.len() {
        return Err(metadata(
            split,
            key,
            format!("has {} values for {} layers", items.len(), kinds.len()),
        ));
    }
    items
        .iter()
        .zip(kinds)
        .enumerate()
        .map(|(l, (v, kind))| {
            let r = v
                .as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| metadata(split, key, format!("has no count at layer {l}")))?;
            let fits = match kind {
                Kind::DeltaRule => r == 0,
                Kind::Attention => r > 0 && top_k.is_multiple_of(r),
            };
            if !fits {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "is {r} at layer {l}, a {kind:?} layer: a GDN layer takes 0, an attention layer a pool dividing attention.indexer.top_k {top_k}"
                    ),
                ));
            }
            Ok(r)
        })
        .collect()
}

/// The PLE site (qwen4exp.cpp:66-121): none when `ple.layers` is absent;
/// else exactly one GDN layer, its geometry within llama.cpp's bounds and
/// its rows tiling the embedding, its token ids in the vocabulary, its hash
/// arrays of the declared lengths with every head's row range inside an i32,
/// and the table tall enough for every range.
fn ple(
    split: &Split,
    n_embd: usize,
    n_vocab: usize,
    kinds: &[Kind],
    defaults: &mut Vec<String>,
) -> Result<Option<Ple>, PlacementError> {
    let Some(items) = split.arch_get_arr("ple.layers") else {
        if split.value(&split.arch_key("ple.layers")).is_some() {
            return Err(metadata(split, "ple.layers", "is not an array"));
        }
        defaults.push("ple.layers = [] (no PLE site, qwen4exp.cpp:66-69)".to_string());
        return Ok(None);
    };
    let layer = match items {
        [v] => v.as_unsigned().and_then(|x| usize::try_from(x).ok()),
        _ => {
            return Err(metadata(
                split,
                "ple.layers",
                format!("lists {} layers; one PLE site is supported", items.len()),
            ));
        }
    };
    let layer = layer
        .filter(|&l| kinds.get(l) == Some(&Kind::DeltaRule))
        .ok_or_else(|| {
            metadata(
                split,
                "ple.layers",
                format!("is {items:?}: the PLE site must be a GDN layer of the file's"),
            )
        })?;
    let ngram = meta_usize(split, "ple.ngram_size")?;
    if !(2..=MAX_PLE_NGRAM).contains(&ngram) {
        return Err(metadata(
            split,
            "ple.ngram_size",
            format!("is {ngram}, outside 2..={MAX_PLE_NGRAM}"),
        ));
    }
    let heads_per_ngram = meta_usize(split, "ple.heads_per_ngram")?;
    let heads = (ngram - 1) * heads_per_ngram;
    if !(1..=MAX_PLE_HEADS).contains(&heads) {
        return Err(metadata(
            split,
            "ple.heads_per_ngram",
            format!("makes {heads} heads, outside 1..={MAX_PLE_HEADS}"),
        ));
    }
    let row = positive(split, "embedding_length_per_layer_input")?;
    if heads * row != n_embd {
        return Err(metadata(
            split,
            "embedding_length_per_layer_input",
            format!("is {row}: {heads} rows of it are not the embedding's {n_embd} values"),
        ));
    }
    let token = |suffix: &str| -> Result<u32, PlacementError> {
        let v = meta_u64(split, suffix)?;
        u32::try_from(v)
            .ok()
            .filter(|&t| (t as usize) < n_vocab)
            .ok_or_else(|| {
                metadata(
                    split,
                    suffix,
                    format!("is {v}, not a token of the {n_vocab}-token vocabulary"),
                )
            })
    };
    let eos = token("ple.eos_token_id")?;
    let image = match split.value(&split.arch_key("ple.image_token_id")) {
        Some(_) => Some(token("ple.image_token_id")?),
        None => {
            defaults.push(
                "ple.image_token_id = none (an image position hashes as the EOS, qwen4exp.cpp:1069-1075)"
                    .to_string(),
            );
            None
        }
    };
    u64s(split, "ple.layer_multipliers", ngram)?;
    let offsets = u64s(split, "ple.head_offsets", heads)?;
    let vocab = u64s(split, "ple.head_vocab_sizes", heads)?;
    let mut rows = 0u64;
    for (h, (&off, &n)) in offsets.iter().zip(&vocab).enumerate() {
        let end = off
            .checked_add(n)
            .filter(|&e| n > 0 && e <= i32::MAX as u64);
        let Some(end) = end else {
            return Err(metadata(
                split,
                "ple.head_vocab_sizes",
                format!(
                    "is {n} at head {h}, offset {off}: the range is empty or past an i32 row index"
                ),
            ));
        };
        rows = rows.max(end);
    }
    let table = "per_layer_token_embd.weight";
    let dims = split.find(table).map(|(_, t)| t.dims.as_slice());
    if !matches!(dims, Some(&[w, n]) if w == row as u64 && n >= rows) {
        return Err(PlacementError::Tensor {
            name: table.to_string(),
            detail: format!("is {dims:?}; the PLE heads read rows of {row} values, {rows} of them"),
        });
    }
    Ok(Some(Ple {
        layer,
        ngram,
        heads_per_ngram,
        conv: positive(split, "ple.conv_kernel")?,
        eos,
        image,
        row,
    }))
}

/// `<architecture>.<suffix>` as an array of exactly `n` unsigned integers.
fn u64s(split: &Split, suffix: &str, n: usize) -> Result<Vec<u64>, PlacementError> {
    let items = meta_arr(split, suffix)?;
    if items.len() != n {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values, not {n}", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_unsigned()
                .ok_or_else(|| metadata(split, suffix, format!("has no unsigned value at {i}")))
        })
        .collect()
}

/// A qwen4exp layer's tensors by kind (qwen4exp.cpp `load_arch_tensors`).
const GDN_STEMS: &[&str] = &[
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
];
const ATTN_STEMS: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
];
/// Every qwen4exp layer's: the two hyper-connection modules and the MoE with
/// its gated shared expert.
const EXP_LAYER_STEMS: &[&str] = &[
    "hc_attn_norm.weight",
    "hc_attn_down.weight",
    "hc_attn_up.weight",
    "hc_attn_inject.weight",
    "hc_ffn_norm.weight",
    "hc_ffn_down.weight",
    "hc_ffn_up.weight",
    "hc_ffn_inject.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];
/// The PLE site's layer tensors.
const PLE_STEMS: &[&str] = &[
    "ple_key.weight",
    "ple_value.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
    "ple_conv1d.weight",
];
/// The block norms qwen35moe carries and qwen4exp does not: its
/// hyper-connection modules' grouped norms and head replace them.
const NORM_STEMS: &[&str] = &["attn_norm.weight", "post_attention_norm.weight"];
/// The model-level tensors a qwen4exp file carries, the PLE table aside.
const EXP_MODEL_TENSORS: &[&str] = &[
    "token_embd.weight",
    "output.weight",
    "output_hc_norm.weight",
    "output_hc_down.weight",
    "output_hc_up.weight",
];

/// Every qwen4exp layer carries its kind's tensors, every layer's and, on the
/// PLE site, the site's; none carries a block norm, the other kind's tensors
/// or a site's off it; the model carries its head's and no `output_norm`. The
/// first missing or misplaced name, with how many more there are.
fn exp_tensors_agree(split: &Split, kinds: &[Kind], exp: &Exp) -> Result<(), PlacementError> {
    let has = |name: &str| split.find(name).is_some();
    let site = exp.ple.map(|p| p.layer);
    let mut missing: Vec<String> = EXP_MODEL_TENSORS
        .iter()
        .filter(|n| !has(n))
        .map(|n| (*n).to_string())
        .collect();
    let mut misplaced: Vec<String> = ["output_norm.weight"]
        .into_iter()
        .chain(site.is_none().then_some("per_layer_token_embd.weight"))
        .filter(|n| has(n))
        .map(str::to_string)
        .collect();
    for (l, kind) in kinds.iter().enumerate() {
        let (own, other) = match kind {
            Kind::DeltaRule => (GDN_STEMS, ATTN_STEMS),
            Kind::Attention => (ATTN_STEMS, GDN_STEMS),
        };
        let (need_ple, ban_ple): (&[&str], &[&str]) = if site == Some(l) {
            (PLE_STEMS, &[])
        } else {
            (&[], PLE_STEMS)
        };
        let name = |s: &&str| format!("blk.{l}.{s}");
        missing.extend(
            own.iter()
                .chain(EXP_LAYER_STEMS)
                .chain(need_ple)
                .map(name)
                .filter(|n| !has(n)),
        );
        misplaced.extend(
            other
                .iter()
                .chain(ban_ple)
                .chain(NORM_STEMS)
                .map(name)
                .filter(|n| has(n)),
        );
    }
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is not in the file ({} tensor(s) the layers' kinds need are missing: {})",
                missing.len(),
                missing.join(", ")
            ),
        });
    }
    if let Some(name) = misplaced.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is in the file, and qwen4exp carries none there ({} such: {})",
                misplaced.len(),
                misplaced.join(", ")
            ),
        });
    }
    Ok(())
}

/// `<architecture>.<suffix>` as a count above 0.
fn positive(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    match meta_usize(split, suffix)? {
        0 => Err(metadata(split, suffix, "is 0")),
        v => Ok(v),
    }
}

/// `rope.dimension_sections`, required (qwen35moe.cpp:8): four counts.
fn sections(split: &Split) -> Result<[u32; 4], PlacementError> {
    let key = "rope.dimension_sections";
    let items = meta_arr(split, key)?;
    if items.len() != 4 {
        return Err(metadata(
            split,
            key,
            format!("has {} values, not 4", items.len()),
        ));
    }
    let mut out = [0u32; 4];
    for (i, v) in items.iter().enumerate() {
        out[i] = v
            .as_unsigned()
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(|| metadata(split, key, format!("has no count at {i}")))?;
    }
    Ok(out)
}

/// Every layer's kind from its tensors, cross-checked with the key that also
/// says it: `attention.recurrent_layers` when the file carries it, else every
/// `interval`-th layer attends (qwen35moe.cpp:18-26).
fn kinds(split: &Split, n_layer: usize, interval: usize) -> Result<Vec<Kind>, PlacementError> {
    let recurrent = match split.arch_get_arr("attention.recurrent_layers") {
        None => None,
        Some(items) => {
            if items.len() != n_layer {
                return Err(metadata(
                    split,
                    "attention.recurrent_layers",
                    format!("has {} values for {n_layer} layers", items.len()),
                ));
            }
            Some(
                items
                    .iter()
                    .enumerate()
                    .map(|(l, v)| {
                        v.as_bool()
                            .or_else(|| v.as_unsigned().map(|x| x != 0))
                            .ok_or_else(|| {
                                metadata(
                                    split,
                                    "attention.recurrent_layers",
                                    format!("has no flag for layer {l}"),
                                )
                            })
                    })
                    .collect::<Result<Vec<bool>, _>>()?,
            )
        }
    };
    (0..n_layer)
        .map(|l| {
            let qkv = format!("blk.{l}.attn_qkv.weight");
            let q = format!("blk.{l}.attn_q.weight");
            let kind = match (split.find(&qkv).is_some(), split.find(&q).is_some()) {
                (true, false) => Kind::DeltaRule,
                (false, true) => Kind::Attention,
                (both, _) => {
                    return Err(PlacementError::Tensor {
                        name: q,
                        detail: format!(
                            "is {} the file, and so is {qkv}: layer {l} needs exactly one",
                            if both { "in" } else { "not in" }
                        ),
                    });
                }
            };
            let (said, key) = match &recurrent {
                Some(r) => (r[l], "attention.recurrent_layers"),
                None => ((l + 1) % interval != 0, "full_attention_interval"),
            };
            if said != (kind == Kind::DeltaRule) {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "makes layer {l} {}, and its tensors make it {kind:?}",
                        if said { "recurrent" } else { "attention" }
                    ),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// `<architecture>.<suffix>` as a count, `None` when absent.
fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

/// Err naming `suffix` when the file carries it with a value `ok` refuses:
/// llama.cpp's builder passes the router's softmax, renormalization and
/// scale 1 as constants (qwen35moe.cpp:499-508).
fn refuse_unless(
    split: &Split,
    suffix: &str,
    ok: impl Fn(&gguf::Value) -> bool,
) -> Result<(), PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        Some(v) if !ok(v) => Err(metadata(
            split,
            suffix,
            format!("is {v:?}; the architecture's router takes softmax, renormalized, scale 1"),
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ATTN_STEMS, EXP_LAYER_STEMS, EXP_MODEL_TENSORS, GDN_STEMS, Hparams, Kind, PLE_STEMS, Ple,
        Variant,
    };
    use crate::arch::synthetic::{V, header_shaped};

    /// Four layers: three GDN (1 the PLE site), then attention.
    const KINDS: [Kind; 4] = [
        Kind::DeltaRule,
        Kind::DeltaRule,
        Kind::DeltaRule,
        Kind::Attention,
    ];

    fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(4)),
            ("embedding_length", V::U32(64)),
            ("context_length", V::U32(1024)),
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
            ("hyper_connection.count", V::U32(4)),
            ("hyper_connection.low_rank", V::U32(8)),
            ("attention.indexer.head_count", V::U32(2)),
            ("attention.indexer.key_length", V::U32(8)),
            ("attention.indexer.top_k", V::U32(16)),
            ("attention.compress_ratios", V::I32s(vec![0, 0, 0, 4])),
            ("ple.layers", V::I32s(vec![1])),
            ("ple.ngram_size", V::U32(3)),
            ("ple.heads_per_ngram", V::U32(2)),
            ("ple.conv_kernel", V::U32(4)),
            ("ple.eos_token_id", V::U32(1)),
            ("ple.image_token_id", V::U32(0)),
            ("embedding_length_per_layer_input", V::U32(16)),
            ("ple.layer_multipliers", V::U64s(vec![3, 5, 7])),
            ("ple.head_offsets", V::U64s(vec![0, 5, 10, 15])),
            ("ple.head_vocab_sizes", V::U64s(vec![5, 5, 5, 5])),
        ]
    }

    fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut out: Vec<(String, Vec<u64>)> = EXP_MODEL_TENSORS
            .iter()
            .map(|n| ((*n).to_string(), vec![1]))
            .collect();
        out.push(("per_layer_token_embd.weight".to_string(), vec![16, 20]));
        for (l, kind) in KINDS.iter().enumerate() {
            let mut stems: Vec<&str> = match kind {
                Kind::DeltaRule => GDN_STEMS.to_vec(),
                Kind::Attention => ATTN_STEMS.to_vec(),
            };
            stems.extend(EXP_LAYER_STEMS);
            if l == 1 {
                stems.extend(PLE_STEMS);
            }
            out.extend(stems.iter().map(|s| (format!("blk.{l}.{s}"), vec![1])));
        }
        out
    }

    /// `kv` with `key` set to `v`, or dropped for `None`.
    fn with(key: &str, v: Option<V>) -> Vec<(&'static str, V)> {
        let mut v = v;
        keys()
            .into_iter()
            .filter_map(|(k, old)| {
                if k == key {
                    v.take().map(|v| (k, v))
                } else {
                    Some((k, old))
                }
            })
            .collect()
    }

    /// The read of a header of `kv` and `tensors`, or its error's text.
    fn read(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let path = header_shaped(tag, arch, kv, &[], tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        hp
    }

    fn refused(tag: &str, kv: &[(&str, V)], tensors: &[(String, Vec<u64>)], want: &str) {
        let err = read(tag, "qwen4exp", kv, tensors).expect_err("the header is refused");
        assert!(err.contains(want), "want {want:?} in: {err}");
    }

    #[test]
    fn a_small_qwen4exp_header_reads() {
        let hp = read("q4x-ok", "qwen4exp", &keys(), &tensors()).expect("the header reads");
        assert_eq!(hp.variant, Variant::Qwen4Exp);
        assert_eq!(hp.kinds, KINDS);
        let exp = hp.exp.expect("the qwen4exp keys");
        assert_eq!((exp.hc_streams, exp.hc_rank), (4, 8));
        assert_eq!(exp.ratios, [0, 0, 0, 4]);
        assert_eq!(
            exp.ple,
            Some(Ple {
                layer: 1,
                ngram: 3,
                heads_per_ngram: 2,
                conv: 4,
                eos: 1,
                image: Some(0),
                row: 16,
            })
        );
        assert_eq!(hp.defaults.len(), 3, "{:?}", hp.defaults);
    }

    #[test]
    fn a_per_layer_expert_width_is_refused_by_name() {
        let kv = with("expert_feed_forward_length", Some(V::I32s(vec![16; 4])));
        refused(
            "q4x-ff",
            &kv,
            &tensors(),
            "qwen4exp.expert_feed_forward_length: is a per-layer array of 4 values",
        );
    }

    /// The window's reset token is its own key: the tokenizer's end of text
    /// is another token, and no fallback stands in for it.
    #[test]
    fn the_ple_eos_is_read_by_its_key() {
        let kv = with("ple.eos_token_id", None);
        refused("q4x-eos", &kv, &tensors(), "qwen4exp.ple.eos_token_id");
    }

    #[test]
    fn a_missing_image_token_is_a_recorded_default() {
        let hp = read(
            "q4x-img",
            "qwen4exp",
            &with("ple.image_token_id", None),
            &tensors(),
        )
        .expect("the header reads");
        assert_eq!(hp.exp.and_then(|e| e.ple).and_then(|p| p.image), None);
        assert!(
            hp.defaults
                .iter()
                .any(|d| d.starts_with("ple.image_token_id")),
            "{:?}",
            hp.defaults
        );
    }

    #[test]
    fn sinkhorn_beside_low_rank_is_refused() {
        let mut kv = keys();
        kv.push(("hyper_connection.sinkhorn_iterations", V::U32(20)));
        refused(
            "q4x-sk",
            &kv,
            &tensors(),
            "qwen4exp.hyper_connection.sinkhorn_iterations",
        );
    }

    #[test]
    fn a_block_norm_is_refused() {
        let mut t = tensors();
        t.push(("blk.0.attn_norm.weight".to_string(), vec![1]));
        refused(
            "q4x-norm",
            &keys(),
            &t,
            "tensor blk.0.attn_norm.weight: is in",
        );
    }

    /// A delta conv of no tap is refused at the header, as the PLE conv is.
    #[test]
    fn a_delta_conv_of_no_tap_is_refused() {
        let kv = with("ssm.conv_kernel", Some(V::U32(0)));
        refused(
            "q4x-conv",
            &kv,
            &tensors(),
            "qwen4exp.ssm.conv_kernel: is 0",
        );
    }

    #[test]
    fn a_ple_site_on_an_attention_layer_is_refused() {
        let kv = with("ple.layers", Some(V::I32s(vec![3])));
        refused("q4x-site", &kv, &tensors(), "must be a GDN layer");
    }

    #[test]
    fn a_pool_on_a_gdn_layer_is_refused() {
        let kv = with("attention.compress_ratios", Some(V::I32s(vec![4, 0, 0, 4])));
        refused(
            "q4x-ratio",
            &kv,
            &tensors(),
            "is 4 at layer 0, a DeltaRule layer",
        );
    }

    #[test]
    fn a_short_ple_table_is_refused() {
        let t: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .map(|(n, d)| match n.as_str() {
                "per_layer_token_embd.weight" => (n, vec![16, 19]),
                _ => (n, d),
            })
            .collect();
        refused(
            "q4x-table",
            &keys(),
            &t,
            "tensor per_layer_token_embd.weight: is Some([16, 19])",
        );
    }

    /// The qwen4exp stems are not qwen35moe's: a Qwen3.6 file carrying a
    /// hyper-connection tensor fails as unclassified, named.
    #[test]
    fn a_qwen35moe_file_with_a_qwen4exp_stem_is_unclassified() {
        let kv: Vec<(&str, V)> = keys()
            .into_iter()
            .filter(|(k, _)| {
                !k.starts_with("hyper_connection.")
                    && !k.starts_with("attention.indexer.")
                    && !k.starts_with("ple.")
                    && !matches!(
                        *k,
                        "attention.compress_ratios" | "embedding_length_per_layer_input"
                    )
            })
            .collect();
        let mut t: Vec<(String, Vec<u64>)> =
            ["token_embd.weight", "output_norm.weight", "output.weight"]
                .iter()
                .map(|n| ((*n).to_string(), vec![1]))
                .collect();
        for (l, kind) in KINDS.iter().enumerate() {
            let marker = match kind {
                Kind::DeltaRule => "attn_qkv.weight",
                Kind::Attention => "attn_q.weight",
            };
            t.push((format!("blk.{l}.{marker}"), vec![1]));
        }
        t.push(("blk.0.hc_attn_norm.weight".to_string(), vec![1]));
        let path = header_shaped("q35-hc", "qwen35moe", &kv, &[], &t);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = crate::arch::spec(&split)
            .expect_err("the stem is not qwen35moe's")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("1 tensors have no role: blk.0.hc_attn_norm.weight"),
            "{err}"
        );
    }
}
