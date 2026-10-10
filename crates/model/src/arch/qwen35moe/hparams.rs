//! Every `<architecture>.*` key the description reads, and each layer's mixer
//! and feed-forward kind from the tensors it carries. Required keys are
//! llama.cpp's required set; a key it reads as optional keeps its reading of
//! the absence, and every default taken is recorded ([`Hparams::defaults`])
//! with the line that sets it. The expert keys are read when a layer routes,
//! the dense width when a layer is dense. The qwen4exp keys ([`Exp`]) are read
//! for that variant only, and its tensors are checked against its layers'
//! kinds ([`Hparams::read`]). A file that carries next-token layers after its
//! trunk (`nextn_predict_layers`) reads the trunk alone: every per-layer value
//! below spans the trunk, and the next-token layers' tensors are never
//! loaded (`roles`).

use gguf::{Split, Value};

use crate::arch::{
    meta_arr, meta_f32, meta_u64, meta_usize, metadata, n_vocab, nextn_layers, qwen35moe_variant,
};
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
    /// `qwen35`: the dense Qwen3.5 trunk — qwen35moe's layers with a SwiGLU
    /// FFN of `feed_forward_length` in place of the routed experts
    /// (qwen35.cpp:470-480).
    Qwen35,
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
            Variant::Qwen35 => "qwen35.cpp, which routes no layer",
            Variant::Qwen4Exp => "qwen4exp.cpp:992-1001",
        }
    }

    /// The lines that default the attention interval.
    fn interval_lines(self) -> &'static str {
        match self {
            Variant::Qwen35Moe => "qwen35moe.cpp:21-25",
            Variant::Qwen35 => "qwen35.cpp:18-24",
            Variant::Qwen4Exp => "qwen4exp.cpp:128-134",
        }
    }

    /// The feed-forward kind every layer of the variant has: llama.cpp's
    /// builder asserts a router on every qwen35moe and qwen4exp layer and none
    /// on a qwen35 one (qwen35moe.cpp:496, qwen35.cpp:472).
    fn ffn(self) -> FfnKind {
        match self {
            Variant::Qwen35Moe | Variant::Qwen4Exp => FfnKind::Routed,
            Variant::Qwen35 => FfnKind::Dense,
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

/// What a trunk layer's feed-forward block is, from the tensors it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// `ffn_gate_inp`: routed experts.
    Routed,
    /// `ffn_gate`: one SwiGLU block of `feed_forward_length`.
    Dense,
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
    /// `attention.compress_ratios`, per trunk layer: an attention layer's
    /// tokens per pooled key, 0 on a GDN layer.
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
    /// `block_count`: the trunk's layers and the next-token layers after them.
    pub n_layer: usize,
    /// `block_count` less `nextn_predict_layers` (absent: 0): the layers the
    /// description runs. [`Hparams::kinds`], [`Hparams::ffns`] and
    /// [`Exp::ratios`] hold one value per trunk layer.
    pub n_trunk: usize,
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
    /// `expert_count`; 0 on a file no layer of which routes (no expert key is
    /// read there).
    pub n_expert: usize,
    /// `expert_used_count`; 0 where `n_expert` is.
    pub n_used: usize,
    /// `expert_feed_forward_length`; 0 where `n_expert` is.
    pub expert_ff: usize,
    /// `expert_shared_feed_forward_length`; `None` when the file has no shared expert.
    pub shared_ff: Option<usize>,
    /// `feed_forward_length`, the dense layers' width; `None` on a file no
    /// layer of which is dense (the key is not read there).
    pub ff: Option<usize>,
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
    /// Every trunk layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// Every trunk layer's feed-forward kind, in order.
    pub ffns: Vec<FfnKind>,
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
        let n_layer = meta_usize(split, "block_count")?;
        let mut defaults = Vec::new();
        let n_trunk = n_layer
            - nextn_layers(
                split,
                n_layer,
                "llama-hparams.cpp:725,818",
                "the probe of llama-hparams.cpp:1662-1666",
                &mut defaults,
            )?;
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
        let kinds = kinds(split, n_trunk, n_layer, interval)?;
        let ffns = ffn_kinds(split, n_trunk, variant)?;
        let (n_expert, n_used, expert_ff) = if ffns.contains(&FfnKind::Routed) {
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
            (n_expert, n_used, expert_ff)
        } else {
            (0, 0, 0)
        };
        let ff = if ffns.contains(&FfnKind::Dense) {
            Some(dense_ff(split)?)
        } else {
            None
        };
        let n_vocab = n_vocab(split)?;
        let exp = match variant {
            Variant::Qwen35Moe | Variant::Qwen35 => None,
            Variant::Qwen4Exp => Some(exp(split, n_embd, n_vocab, &kinds, n_layer, &mut defaults)?),
        };
        let hp = Hparams {
            variant,
            n_layer,
            n_trunk,
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
            ff,
            conv,
            state,
            v_heads,
            k_heads,
            interval,
            kinds,
            ffns,
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
pub(super) fn expert_ff(split: &Split) -> Result<usize, PlacementError> {
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

/// `feed_forward_length`, one width for every dense layer: a per-layer array
/// (llama.cpp's `get_key_or_arr`) is refused by name, as 0 is.
fn dense_ff(split: &Split) -> Result<usize, PlacementError> {
    let key = "feed_forward_length";
    if let Some(Value::Array(items)) = split.value(&split.arch_key(key)) {
        return Err(metadata(
            split,
            key,
            format!(
                "is a per-layer array of {} values; this reader reads one dense width",
                items.len()
            ),
        ));
    }
    match meta_usize(split, key)? {
        0 => Err(metadata(split, key, "is 0")),
        v => Ok(v),
    }
}

/// Every trunk layer's feed-forward kind from its tensors — `ffn_gate_inp`
/// routes, `ffn_gate` is dense, exactly one of the two — held to the kind
/// `variant`'s builder asserts on every layer ([`Variant::ffn`]); the first
/// layer that disagrees is refused by name.
fn ffn_kinds(
    split: &Split,
    n_trunk: usize,
    variant: Variant,
) -> Result<Vec<FfnKind>, PlacementError> {
    (0..n_trunk)
        .map(|l| {
            let router = format!("blk.{l}.ffn_gate_inp.weight");
            let gate = format!("blk.{l}.ffn_gate.weight");
            let kind = match (split.find(&router).is_some(), split.find(&gate).is_some()) {
                (true, false) => FfnKind::Routed,
                (false, true) => FfnKind::Dense,
                (both, _) => {
                    return Err(PlacementError::Tensor {
                        name: gate,
                        detail: format!(
                            "is {} the file, and so is {router}: layer {l} needs exactly one",
                            if both { "in" } else { "not in" }
                        ),
                    });
                }
            };
            if kind != variant.ffn() {
                let found = match kind {
                    FfnKind::Routed => router,
                    FfnKind::Dense => gate,
                };
                return Err(PlacementError::Tensor {
                    name: found,
                    detail: format!(
                        "makes layer {l} {kind:?}, and a {variant:?} file runs every layer {:?}",
                        variant.ffn()
                    ),
                });
            }
            Ok(kind)
        })
        .collect()
}

/// The qwen4exp keys (qwen4exp.cpp:26-147), each required where that loader
/// requires it, cross-checked with the trunk layers' `kinds` (`n_layer`: the
/// file's, for the per-layer arrays that may cover its next-token layers).
fn exp(
    split: &Split,
    n_embd: usize,
    n_vocab: usize,
    kinds: &[Kind],
    n_layer: usize,
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
    let ratios = ratios(split, kinds, n_layer, idx_top_k)?;
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

/// `attention.compress_ratios`, required as a per-layer array of the trunk's
/// or the file's layers ([`covers_trunk`]): above 0 on an attention layer (its
/// pool), 0 on a GDN layer, and dividing `top_k`.
fn ratios(
    split: &Split,
    kinds: &[Kind],
    n_layer: usize,
    top_k: usize,
) -> Result<Vec<usize>, PlacementError> {
    let key = "attention.compress_ratios";
    let items = meta_arr(split, key)?;
    if !covers_trunk(items.len(), kinds.len(), n_layer) {
        return Err(metadata(
            split,
            key,
            format!(
                "has {} values for {}",
                items.len(),
                layers_text(kinds.len(), n_layer)
            ),
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
pub(super) const GDN_STEMS: &[&str] = &[
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
pub(super) const ATTN_STEMS: &[&str] = &[
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
pub(super) const EXP_LAYER_STEMS: &[&str] = &[
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
pub(super) const PLE_STEMS: &[&str] = &[
    "ple_key.weight",
    "ple_value.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
    "ple_conv1d.weight",
];
/// The block norms qwen35moe carries and qwen4exp does not: its
/// hyper-connection modules' grouped norms and head replace them.
pub(super) const NORM_STEMS: &[&str] = &["attn_norm.weight", "post_attention_norm.weight"];
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
pub(super) fn sections(split: &Split) -> Result<[u32; 4], PlacementError> {
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

/// Every trunk layer's kind from its tensors, cross-checked with the key that
/// also says it: `attention.recurrent_layers` when the file carries it (an
/// array of the trunk's or the file's layers, [`covers_trunk`]), else every
/// `interval`-th layer attends (qwen35moe.cpp:18-26). A next-token layer is an
/// attention layer whatever the interval says (`src/llama-hparams.cpp`
/// :736-739 for qwen4exp, :837-843 for qwen35moe) and is no trunk layer.
fn kinds(
    split: &Split,
    n_trunk: usize,
    n_layer: usize,
    interval: usize,
) -> Result<Vec<Kind>, PlacementError> {
    let recurrent = match split.arch_get_arr("attention.recurrent_layers") {
        None => None,
        Some(items) => {
            if !covers_trunk(items.len(), n_trunk, n_layer) {
                return Err(metadata(
                    split,
                    "attention.recurrent_layers",
                    format!(
                        "has {} values for {}",
                        items.len(),
                        layers_text(n_trunk, n_layer)
                    ),
                ));
            }
            Some(
                items
                    .iter()
                    .take(n_trunk)
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
    (0..n_trunk)
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

/// Whether a per-layer array of `len` values covers the trunk: one value per
/// layer of the file, next-token layers last (the trunk's are the first
/// `n_trunk`), or one per trunk layer.
fn covers_trunk(len: usize, n_trunk: usize, n_layer: usize) -> bool {
    len == n_trunk || len == n_layer
}

/// What a per-layer array that [`covers_trunk`] has values for, as a refusal
/// names it.
fn layers_text(n_trunk: usize, n_layer: usize) -> String {
    if n_layer == n_trunk {
        format!("{n_trunk} layers")
    } else {
        format!("{n_trunk} trunk layers or the file's {n_layer}")
    }
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
pub(super) fn refuse_unless(
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
pub(super) mod tests {
    use super::{
        ATTN_STEMS, EXP_LAYER_STEMS, EXP_MODEL_TENSORS, FfnKind, GDN_STEMS, Hparams, Kind,
        NORM_STEMS, PLE_STEMS, Ple, Variant,
    };
    use crate::arch::synthetic::{V, header_shaped};
    use crate::arch::{coverage, qwen35moe::roles};
    use crate::placement::Role;

    /// Four layers: three GDN (1 the PLE site), then attention.
    const KINDS: [Kind; 4] = [
        Kind::DeltaRule,
        Kind::DeltaRule,
        Kind::DeltaRule,
        Kind::Attention,
    ];

    pub(crate) fn keys() -> Vec<(&'static str, V)> {
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

    pub(crate) fn tensors() -> Vec<(String, Vec<u64>)> {
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
        // The router's three constants and the absent `nextn_predict_layers`.
        assert_eq!(hp.defaults.len(), 4, "{:?}", hp.defaults);
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
        let kv = qwen35moe_keys();
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
            t.push((format!("blk.{l}.ffn_gate_inp.weight"), vec![1]));
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

    /// The qwen35moe keys: [`keys`] without qwen4exp's hyper-connection,
    /// indexer, PLE and compress-ratio keys.
    fn qwen35moe_keys() -> Vec<(&'static str, V)> {
        keys()
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
            .collect()
    }

    /// The qwen35 keys: qwen35moe's trunk keys, no expert key, and the dense
    /// width.
    fn qwen35_keys() -> Vec<(&'static str, V)> {
        qwen35moe_keys()
            .into_iter()
            .filter(|(k, _)| !k.starts_with("expert_"))
            .chain([("feed_forward_length", V::U32(48))])
            .collect()
    }

    /// A qwen35 layer's tensors: its mixer's, the two block norms and the
    /// dense FFN's three matrices.
    fn qwen35_tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> =
            ["token_embd.weight", "output_norm.weight", "output.weight"]
                .iter()
                .map(|n| ((*n).to_string(), vec![1]))
                .collect();
        for (l, kind) in KINDS.iter().enumerate() {
            let mixer: &[&str] = match kind {
                Kind::DeltaRule => GDN_STEMS,
                Kind::Attention => &ATTN_STEMS[..6],
            };
            let stems = mixer.iter().chain(NORM_STEMS).chain(&[
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]);
            // The query projection writes each of the 4 heads' 16 values and
            // its gate beside them.
            let dims = |s: &str| {
                if s == "attn_q.weight" {
                    vec![64, 128]
                } else {
                    vec![1]
                }
            };
            t.extend(stems.map(|s| (format!("blk.{l}.{s}"), dims(s))));
        }
        t
    }

    /// A dense qwen35 header reads with every layer's FFN dense at
    /// `feed_forward_length`, no expert key read, and its description gives
    /// each layer `Ffn::Dense` and each FFN matrix the dense role.
    #[test]
    fn a_small_qwen35_header_reads_dense() {
        let hp =
            read("q35d-ok", "qwen35", &qwen35_keys(), &qwen35_tensors()).expect("the header reads");
        assert_eq!(hp.variant, Variant::Qwen35);
        assert_eq!(hp.kinds, KINDS);
        assert_eq!(hp.ffns, [FfnKind::Dense; 4]);
        assert_eq!(hp.ff, Some(48));
        assert_eq!((hp.n_expert, hp.n_used, hp.expert_ff), (0, 0, 0));
        assert!(hp.exp.is_none());
        let path = header_shaped(
            "q35d-spec",
            "qwen35",
            &qwen35_keys(),
            &[("tokenizer.ggml.pre", V::Str("qwen35"))],
            &qwen35_tensors(),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split);
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the description reads");
        assert_eq!(read.spec.arch, models::Arch::Qwen35);
        for l in &read.spec.layers {
            assert_eq!(
                l.ffn,
                models::Ffn::Dense {
                    ff: 48,
                    act: models::Act::SwiGlu { limit: None }
                }
            );
        }
        let dense = read
            .tensors
            .tensors
            .iter()
            .filter(|t| t.role == crate::placement::Role::DenseFfn)
            .count();
        assert_eq!(dense, 3 * KINDS.len());
    }

    /// A qwen35 layer that routes is refused by name: the architecture's
    /// builder asserts no router on any layer.
    #[test]
    fn a_routed_layer_in_a_qwen35_file_is_refused() {
        let t: Vec<(String, Vec<u64>)> = qwen35_tensors()
            .into_iter()
            .map(|(n, d)| match n.as_str() {
                "blk.2.ffn_gate.weight" => ("blk.2.ffn_gate_inp.weight".to_string(), d),
                _ => (n, d),
            })
            .collect();
        let err = read("q35d-route", "qwen35", &qwen35_keys(), &t).expect_err("refused");
        assert!(
            err.contains("blk.2.ffn_gate_inp.weight") && err.contains("Qwen35"),
            "{err}"
        );
    }

    /// A qwen35 layer with no FFN, and a per-layer dense width, are refused by
    /// name.
    #[test]
    fn a_dense_layer_without_its_gate_or_width_is_refused() {
        let t: Vec<(String, Vec<u64>)> = qwen35_tensors()
            .into_iter()
            .filter(|(n, _)| n != "blk.1.ffn_gate.weight")
            .collect();
        let err = read("q35d-gate", "qwen35", &qwen35_keys(), &t).expect_err("refused");
        assert!(err.contains("blk.1.ffn_gate.weight"), "{err}");
        let kv: Vec<(&str, V)> = qwen35_keys()
            .into_iter()
            .map(|(k, v)| match k {
                "feed_forward_length" => (k, V::I32s(vec![48; 4])),
                _ => (k, v),
            })
            .collect();
        let err = read("q35d-ff", "qwen35", &kv, &qwen35_tensors()).expect_err("refused");
        assert!(
            err.contains("qwen35.feed_forward_length: is a per-layer array of 4 values"),
            "{err}"
        );
    }

    /// llama.cpp's Clef layout (`clef`) is the Qwen3.5 dense body: the same
    /// description read under the file's own key prefix, the decision head's
    /// tensors beside the trunk's carried and never loaded.
    #[test]
    fn a_clef_header_reads_as_the_qwen35_body_and_its_head_is_unused() {
        let mut t = qwen35_tensors();
        let head = [
            "dec.blk.0.cross_attn_q.weight",
            "dec.blk.2.attn_norm.bias",
            "decision.proj_memory.weight",
            "decision.scales",
            "token_types.weight",
        ];
        t.extend(head.iter().map(|n| ((*n).to_string(), vec![1])));
        let hp = read("clef-ok", "clef", &qwen35_keys(), &t).expect("the header reads");
        assert_eq!(hp.variant, Variant::Qwen35);
        assert_eq!(hp.ff, Some(48));
        let path = header_shaped(
            "clef-spec",
            "clef",
            &qwen35_keys(),
            &[("tokenizer.ggml.pre", V::Str("qwen35"))],
            &t,
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split);
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the description reads");
        assert_eq!(read.spec.arch, models::Arch::Qwen35);
        for n in head {
            let row = read.tensors.tensors.iter().find(|r| r.name == n).expect(n);
            assert_eq!(
                (row.role, row.layer),
                (crate::placement::Role::Unused, None),
                "{n}"
            );
        }
        let unused = read
            .tensors
            .tensors
            .iter()
            .filter(|r| r.role == crate::placement::Role::Unused)
            .count();
        assert_eq!(unused, head.len());
    }

    /// A head tensor the table does not name is still a refusal by name, and a
    /// file of another architecture string is not the body.
    #[test]
    fn a_stray_head_tensor_and_another_architecture_are_refused() {
        let mut t = qwen35_tensors();
        t.push(("decoder.blk.0.attn_q.weight".to_string(), vec![1]));
        let path = header_shaped("clef-stray", "clef", &qwen35_keys(), &[], &t);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = crate::arch::spec(&split)
            .expect_err("a tensor no table holds")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("1 tensors have no role: decoder.blk.0.attn_q.weight"),
            "{err}"
        );
        let err =
            read("clef-other", "clef2", &qwen35_keys(), &qwen35_tensors()).expect_err("refused");
        assert!(err.contains("clef2") && err.contains("\"clef\""), "{err}");
    }

    /// `kv` with `key` set to `v` (added when `kv` has none).
    fn set(mut kv: Vec<(&'static str, V)>, key: &'static str, v: V) -> Vec<(&'static str, V)> {
        kv.retain(|(k, _)| *k != key);
        kv.push((key, v));
        kv
    }

    /// `kv` of a file of `KINDS.len() + 1` layers whose last is a next-token
    /// layer.
    fn with_next_token_layer(kv: Vec<(&'static str, V)>) -> Vec<(&'static str, V)> {
        set(
            set(kv, "block_count", V::U32(KINDS.len() as u32 + 1)),
            "nextn_predict_layers",
            V::U32(1),
        )
    }

    /// A qwen35moe file's trunk: the model's tensors and, per layer of
    /// [`KINDS`], its mixer's marker and its router (`attn_q` writes the 4
    /// heads' 16 values and the gate beside them).
    fn qwen35moe_tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> =
            ["token_embd.weight", "output_norm.weight", "output.weight"]
                .iter()
                .map(|n| ((*n).to_string(), vec![1]))
                .collect();
        for (l, kind) in KINDS.iter().enumerate() {
            t.push(match kind {
                Kind::DeltaRule => (format!("blk.{l}.attn_qkv.weight"), vec![1]),
                Kind::Attention => (format!("blk.{l}.attn_q.weight"), vec![64, 128]),
            });
            t.push((format!("blk.{l}.ffn_gate_inp.weight"), vec![1]));
        }
        t
    }

    /// The tensors of a qwen35moe next-token layer `l`: what unsloth's and
    /// bartowski's Qwen3.6 files carry on their last block — an attention
    /// layer whatever the interval says, with the layer's own four stems.
    fn qwen35moe_next_token_layer(l: usize) -> Vec<(String, Vec<u64>)> {
        const STEMS: [&str; 20] = [
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
        STEMS
            .iter()
            .map(|s| (format!("blk.{l}.{s}"), vec![1]))
            .collect()
    }

    /// The tensors of a qwen4exp next-token layer `l`: an attention layer's
    /// stems, the layer stems every qwen4exp layer has, and the seven
    /// next-token stems of ik's table (antirez's Qwen3.8 file carries six of
    /// them, the gated-residual head's three among them).
    fn qwen4exp_next_token_layer(l: usize) -> Vec<(String, Vec<u64>)> {
        const NEXTN: [&str; 7] = [
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
            "nextn.hc_head_norm.weight",
            "nextn.hc_head_down.weight",
            "nextn.hc_head_up.weight",
        ];
        ATTN_STEMS
            .iter()
            .chain(EXP_LAYER_STEMS)
            .chain(&NEXTN)
            .map(|s| (format!("blk.{l}.{s}"), vec![1]))
            .collect()
    }

    /// The description of a header of `kv` and `tensors`, or its error's text.
    fn described(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<crate::arch::Read, String> {
        let path = header_shaped(
            tag,
            arch,
            kv,
            &[("tokenizer.ggml.pre", V::Str("qwen35"))],
            tensors,
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    /// A qwen35moe file whose last block is a next-token layer reads its
    /// trunk: `n_trunk` is the block count less the layer, the layer's
    /// tensors are `Unused`, and the description and the coverage list are the
    /// file's without the layer. Layer 4 carries `attn_q` where the interval
    /// 4 says a delta layer, so a read of the layer as a trunk layer is
    /// refused by its kind.
    #[test]
    fn a_qwen35moe_next_token_layer_reads_as_unused() {
        let without = described(
            "q35n-base",
            "qwen35moe",
            &qwen35moe_keys(),
            &qwen35moe_tensors(),
        )
        .expect("the file without the layer reads");
        let kv = with_next_token_layer(qwen35moe_keys());
        let mut t = qwen35moe_tensors();
        t.extend(qwen35moe_next_token_layer(4));
        let hp = read("q35n-hp", "qwen35moe", &kv, &t).expect("the header reads");
        assert_eq!((hp.n_layer, hp.n_trunk), (5, 4));
        assert_eq!(hp.kinds, KINDS);
        assert_eq!(hp.ffns, [FfnKind::Routed; 4]);
        let with = described("q35n-spec", "qwen35moe", &kv, &t).expect("the description reads");
        assert_eq!(with.spec, without.spec, "the layer is no layer of the spec");
        assert_eq!(with.spec.layers.len(), 4);
        assert_eq!(with.tensors.layers, 4);
        assert!(
            without.defaults[0].starts_with("nextn_predict_layers = 0 ("),
            "{:?}",
            without.defaults
        );
        assert_eq!(
            with.defaults[..],
            without.defaults[1..],
            "the declared count is no default"
        );
        let rows = &with.tensors.tensors;
        let last: Vec<_> = rows.iter().filter(|r| r.layer == Some(4)).collect();
        assert_eq!(last.len(), 20, "every tensor of the layer is a row");
        assert!(last.iter().all(|r| r.role == Role::Unused), "{last:?}");
        assert!(
            rows.iter()
                .filter(|r| r.layer.is_some_and(|l| l < 4))
                .all(|r| r.role != Role::Unused),
            "a trunk row is never unused"
        );
        assert_eq!(
            coverage::check(&with.spec, &with.tensors),
            coverage::check(&without.spec, &without.tensors),
            "the layer adds no item to the coverage list"
        );
    }

    /// The same for qwen4exp, whose ratio array covers the file's layers (the
    /// layer's own entry is not read) and whose layer carries the
    /// gated-residual head's stems beside the others.
    #[test]
    fn a_qwen4exp_next_token_layer_reads_as_unused() {
        let kv = set(
            with_next_token_layer(keys()),
            "attention.compress_ratios",
            V::I32s(vec![0, 0, 0, 4, 7]),
        );
        let mut t = tensors();
        t.extend(qwen4exp_next_token_layer(4));
        let path = header_shaped("q4x-nextn", "qwen4exp", &kv, &[], &t);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).expect("the header reads");
        let model = roles::classify(&split, &hp).expect("every tensor has a role");
        let _ = std::fs::remove_file(&path);
        assert_eq!((hp.n_layer, hp.n_trunk), (5, 4));
        assert_eq!(hp.kinds, KINDS);
        let exp = hp.exp.expect("the qwen4exp keys");
        assert_eq!(exp.ratios, [0, 0, 0, 4]);
        assert!(exp.ple.is_some_and(|p| p.layer == 1));
        assert_eq!(hp.defaults.len(), 3, "{:?}", hp.defaults);
        assert_eq!(model.layers, 4);
        let last: Vec<_> = model
            .tensors
            .iter()
            .filter(|r| r.layer == Some(4))
            .collect();
        assert_eq!(last.len(), ATTN_STEMS.len() + EXP_LAYER_STEMS.len() + 7);
        assert!(last.iter().all(|r| r.role == Role::Unused), "{last:?}");
        let table = model
            .tensors
            .iter()
            .find(|r| r.name == "per_layer_token_embd.weight")
            .expect("the PLE table");
        assert_eq!((table.role, table.layer), (Role::EngramTable, Some(1)));
    }

    /// A per-layer array covers the trunk or the file: the next-token layer's
    /// own value is not read, and any other length is refused with both
    /// counts (the plain count when the file has no such layer).
    #[test]
    fn a_per_layer_array_covers_the_trunk_or_the_file() {
        let mut t = qwen35moe_tensors();
        t.extend(qwen35moe_next_token_layer(4));
        let recurrent = |v: Vec<i32>| {
            set(
                with_next_token_layer(qwen35moe_keys()),
                "attention.recurrent_layers",
                V::I32s(v),
            )
        };
        for (tag, v) in [
            ("q35n-r5", vec![1, 1, 1, 0, -1]),
            ("q35n-r4", vec![1, 1, 1, 0]),
        ] {
            let hp = read(tag, "qwen35moe", &recurrent(v), &t).expect("the header reads");
            assert_eq!(hp.kinds, KINDS, "{tag}");
        }
        let err = read("q35n-r3", "qwen35moe", &recurrent(vec![1, 1, 1]), &t)
            .expect_err("three values cover neither");
        assert!(
            err.contains(
                "qwen35moe.attention.recurrent_layers: has 3 values for 4 trunk layers or the file's 5"
            ),
            "{err}"
        );
        let kv = set(
            qwen35moe_keys(),
            "attention.recurrent_layers",
            V::I32s(vec![1, 1, 1]),
        );
        let err = read("q35n-r3p", "qwen35moe", &kv, &qwen35moe_tensors())
            .expect_err("three values for four layers");
        assert!(
            err.contains("qwen35moe.attention.recurrent_layers: has 3 values for 4 layers"),
            "{err}"
        );
        let mut t = tensors();
        t.extend(qwen4exp_next_token_layer(4));
        let ratios = |v: Vec<i32>| {
            set(
                with_next_token_layer(keys()),
                "attention.compress_ratios",
                V::I32s(v),
            )
        };
        for (tag, v) in [
            ("q4x-c5", vec![0, 0, 0, 4, 0]),
            ("q4x-c4", vec![0, 0, 0, 4]),
        ] {
            let hp = read(tag, "qwen4exp", &ratios(v), &t).expect("the header reads");
            assert_eq!(hp.exp.map(|e| e.ratios), Some(vec![0, 0, 0, 4]), "{tag}");
        }
        let err = read("q4x-c3", "qwen4exp", &ratios(vec![0, 0, 0]), &t)
            .expect_err("three values cover neither");
        assert!(
            err.contains(
                "qwen4exp.attention.compress_ratios: has 3 values for 4 trunk layers or the file's 5"
            ),
            "{err}"
        );
    }

    /// A next-token count the trunk cannot lose is refused by name: the whole
    /// file, or more.
    #[test]
    fn next_token_layers_covering_the_file_are_refused() {
        for n in [4u32, 5] {
            let kv = set(qwen35moe_keys(), "nextn_predict_layers", V::U32(n));
            let err = read("q35n-all", "qwen35moe", &kv, &qwen35moe_tensors())
                .expect_err("no trunk layer is left");
            assert!(
                err.contains(&format!(
                    "qwen35moe.nextn_predict_layers: is {n}, not below block_count 4"
                )),
                "{err}"
            );
            let kv = set(keys(), "nextn_predict_layers", V::U32(n));
            let err =
                read("q4x-all", "qwen4exp", &kv, &tensors()).expect_err("no trunk layer is left");
            assert!(
                err.contains(&format!(
                    "qwen4exp.nextn_predict_layers: is {n}, not below block_count 4"
                )),
                "{err}"
            );
        }
    }

    /// The owner of `nextn_predict_layers` (`arch::nextn_layers`): an absent
    /// key and a declared count whose first next-token layer has no
    /// `nextn.eh_proj` are each a recorded 0, and a count the tensors back is
    /// believed.
    #[test]
    fn a_declared_count_is_believed_only_where_its_probe_tensor_is() {
        let read_n = |tag: &str, kv: &[(&str, V)], t: &[(String, Vec<u64>)], probe_note: &str| {
            let path = header_shaped(tag, "qwen35moe", kv, &[], t);
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let mut defaults = Vec::new();
            let n = crate::arch::nextn_layers(&split, 5, "absent", probe_note, &mut defaults);
            let _ = std::fs::remove_file(&path);
            (n.expect("a count below block_count"), defaults)
        };
        let kv = with_next_token_layer(qwen35moe_keys());
        let (n, defaults) = read_n("q35n-pr-ok", &kv, &qwen35moe_next_token_layer(4), "");
        assert_eq!((n, defaults.len()), (1, 0));
        let (n, defaults) = read_n("q35n-pr-no", &kv, &qwen35moe_tensors(), "probe");
        assert_eq!(n, 0);
        assert_eq!(
            defaults,
            ["nextn_predict_layers = 0 (blk.4.nextn.eh_proj.weight is not in the file, probe)"]
        );
        let (_, defaults) = read_n("q35n-pr-nn", &kv, &qwen35moe_tensors(), "");
        assert_eq!(
            defaults,
            ["nextn_predict_layers = 0 (blk.4.nextn.eh_proj.weight is not in the file)"]
        );
        let kv = set(qwen35moe_keys(), "block_count", V::U32(5));
        let (n, defaults) = read_n("q35n-pr-abs", &kv, &qwen35moe_tensors(), "");
        assert_eq!(n, 0);
        assert_eq!(defaults, ["nextn_predict_layers = 0 (absent)"]);
    }

    /// A next-token stem is the next-token layer's alone: on a trunk layer, or
    /// in a file with no such layer, it is unclassified by name, as is a
    /// stem of the other variant and one ik's tables for these architectures
    /// do not hold (`embed_tokens`, `shared_head_head`).
    #[test]
    fn a_next_token_stem_off_the_next_token_layer_is_refused() {
        let refused = |tag: &str,
                       kv: &[(&str, V)],
                       mut t: Vec<(String, Vec<u64>)>,
                       extra: &[&str],
                       want: &str| {
            t.extend(extra.iter().map(|n| ((*n).to_string(), vec![1])));
            let err = described(tag, "qwen35moe", kv, &t).expect_err("the stem has no role");
            assert!(err.contains(want), "want {want:?} in: {err}");
        };
        let with_layer = || {
            let mut t = qwen35moe_tensors();
            t.extend(qwen35moe_next_token_layer(4));
            t
        };
        refused(
            "q35n-trunk",
            &with_next_token_layer(qwen35moe_keys()),
            with_layer(),
            &["blk.1.nextn.eh_proj.weight"],
            "1 tensors have no role: blk.1.nextn.eh_proj.weight",
        );
        refused(
            "q35n-other",
            &with_next_token_layer(qwen35moe_keys()),
            with_layer(),
            &[
                "blk.4.nextn.hc_head_norm.weight",
                "blk.4.hc_attn_norm.weight",
            ],
            "2 tensors have no role: blk.4.nextn.hc_head_norm.weight, blk.4.hc_attn_norm.weight",
        );
        refused(
            "q35n-head",
            &with_next_token_layer(qwen35moe_keys()),
            with_layer(),
            &[
                "blk.4.nextn.embed_tokens.weight",
                "blk.4.nextn.shared_head_head.weight",
            ],
            "2 tensors have no role: blk.4.nextn.embed_tokens.weight, blk.4.nextn.shared_head_head.weight",
        );
        // No next-token count: the stem sits on the last of four trunk layers.
        refused(
            "q35n-none",
            &qwen35moe_keys(),
            qwen35moe_tensors(),
            &["blk.3.nextn.enorm.weight"],
            "1 tensors have no role: blk.3.nextn.enorm.weight",
        );
    }
}
