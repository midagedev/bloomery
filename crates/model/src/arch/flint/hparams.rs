//! The scalar widths and per-layer masks of the architectures this reader
//! serves (`flint`, `exaone-moe`), checked before constructing a spec.

use super::roles;
use crate::arch::hybrid::layers_agree;
use crate::arch::rules::{ArchRules, ClampOrder, RopeFallback};
use crate::arch::{
    check_kv, dims_agree, expert_counts, finite_positive, meta_bool, meta_usize, metadata, mul,
    n_vocab, nextn_layers, one_expert_group, optional_usize, per_layer_bool, per_layer_count,
    per_layer_f32, plain_rope, positive, sigmoid_gating,
};
use crate::placement::PlacementError;
use gguf::{Split, Value};

/// A SwiGLU limit of at most this clamps nothing (llama-graph.cpp:2064's `eps`).
const EPS: f32 = 1e-6;

/// The line `nextn_predict_layers` is read at (llama.cpp:1490, optional).
const NEXTN_AT: &str = "llama.cpp:1490";

pub(super) struct Hparams {
    /// The description row of the file's own architecture string.
    pub(super) rules: &'static ArchRules,
    pub(super) arch: models::Arch,
    /// `block_count`: the trunk's blocks and the next-token ones.
    pub(super) blocks: usize,
    /// The trunk: the blocks no next-token layer takes.
    pub(super) trunk: usize,
    /// The next-token blocks the file carries tensors for: described in
    /// `ModelSpec.mtp`, run by no program.
    pub(super) mtp: usize,
    pub(super) hidden: usize,
    pub(super) vocab: usize,
    pub(super) context: usize,
    pub(super) eps: f32,
    pub(super) heads: usize,
    pub(super) kv: Vec<usize>,
    pub(super) head: usize,
    pub(super) ff: Vec<usize>,
    pub(super) expert_ff: Vec<usize>,
    pub(super) dense_lead: usize,
    pub(super) experts: usize,
    pub(super) used: usize,
    pub(super) shared_ff: usize,
    pub(super) norm: bool,
    pub(super) scale: f32,
    /// Each block's sliding window; `None`: the block attends every position.
    pub(super) window: Vec<Option<u32>>,
    /// Whether each block's head turns.
    pub(super) rope: Vec<bool>,
    /// Whether every block carries `attn_sinks.weight` (the check refuses a
    /// file where the layers disagree).
    pub(super) sinks: bool,
    pub(super) base: f32,
    /// Each block's SwiGLU limit: the routed experts', the shared expert's and
    /// the dense block's.
    pub(super) limit_exp: Vec<f32>,
    pub(super) limit_shexp: Vec<f32>,
    pub(super) limit_dense: Vec<f32>,
    pub(super) defaults: Vec<String>,
}

impl Hparams {
    /// Whether block `l`'s feed-forward side is dense: one of the trunk's dense
    /// lead, or a next-token layer (exaone-moe.cpp:68: the blocks past the
    /// trunk take the dense FFN).
    pub(super) fn dense(&self, l: usize) -> bool {
        l < self.dense_lead || l >= self.trunk
    }

    /// The blocks the file describes: the trunk's, and the next-token ones it
    /// carries tensors for.
    pub(super) fn described(&self) -> Vec<usize> {
        if self.mtp == 0 {
            return (0..self.trunk).collect();
        }
        (0..self.blocks).collect()
    }

    pub(super) fn read(split: &Split) -> Result<Self, PlacementError> {
        let rules = crate::arch::rules_of(split)?;
        let name = rules.arch.name();
        let mut defaults = Vec::new();
        // This common read checks the layer bound before expanding any scalar.
        let blocks = positive(split, "block_count")?;
        let nextn = nextn_layers(split, blocks, NEXTN_AT, NEXTN_AT, &mut defaults)?;
        let trunk = blocks - nextn.past_trunk;
        let kv = per_layer_count(split, "attention.head_count_kv", blocks)?;
        let hidden = positive(split, "embedding_length")?;
        let context = positive(split, "context_length")?;
        let vocab = n_vocab(split)?;
        if vocab == 0 {
            return Err(metadata(split, "vocab_size", "is 0"));
        }
        let eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        let heads = positive(split, "attention.head_count")?;
        let head = positive(split, "attention.key_length")?;
        mul(split, "attention.key_length", heads, head)?;
        let value = positive(split, "attention.value_length")?;
        if value != head {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, not key_length {head}"),
            ));
        }
        if !head.is_multiple_of(2) {
            return Err(metadata(
                split,
                "attention.key_length",
                "must be even for NeoX rotation",
            ));
        }
        if let Some(dims) = optional_usize(split, "rope.dimension_count")?
            && dims != head
        {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {dims}, not the full head {head}"),
            ));
        }
        plain_rope(split)?;
        let base = finite_positive(split, "rope.freq_base")?;
        let window = windows(split, blocks)?;
        let rope = rope_pattern(split, rules, blocks, &window, &mut defaults)?;
        let sinks = (0..blocks).any(|l| split.find(&roles::sink_name(l)).is_some());
        let dense_lead = meta_usize(split, "leading_dense_block_count")?;
        if dense_lead > trunk {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!("is {dense_lead}, above the trunk's {trunk} layers"),
            ));
        }
        let ff = per_layer_count(split, "feed_forward_length", blocks)?;
        let expert_ff = per_layer_count(split, "expert_feed_forward_length", blocks)?;
        let (experts, used) = expert_counts(split)?;
        one_expert_group(split, "the sigmoid router")?;
        let shared_count = positive(split, "expert_shared_count")?;
        let shared_ff = positive(split, "expert_shared_feed_forward_length")?;
        let norm = meta_bool(split, "expert_weights_norm")?;
        sigmoid_gating(split, None, &mut defaults)?;
        let scale = match split.value(&split.arch_key("expert_weights_scale")) {
            None => {
                defaults.push("expert_weights_scale = 1 (llama.cpp's default)".into());
                1.0
            }
            Some(_) => finite_positive(split, "expert_weights_scale")?,
        };
        if norm {
            defaults.push(rules.router_norm_line());
        }
        let (limit_exp, limit_shexp, limit_dense) = clamps(split, blocks, &mut defaults)?;
        let limits = [
            ("swiglu_clamp_exp", &limit_exp),
            ("swiglu_clamp_shexp", &limit_shexp),
            ("swiglu_clamp", &limit_dense),
        ];
        let first_limit = limits.iter().find_map(|(key, values)| {
            values
                .iter()
                .enumerate()
                .find(|&(_, &v)| v > EPS)
                .map(|(l, &v)| (*key, l, v))
        });
        if let Some((key, l, v)) = first_limit {
            if rules.clamp == ClampOrder::None {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "is {v} at layer {l}; the {name} row's clamp order is none, so no limit \
                         of it clamps anything"
                    ),
                ));
            }
            defaults.push(rules.clamp_line());
        }
        let mut hp = Self {
            rules,
            arch: rules.arch,
            blocks,
            trunk,
            mtp: nextn.carried,
            hidden,
            vocab,
            context,
            eps,
            heads,
            kv,
            head,
            ff,
            expert_ff,
            dense_lead,
            experts,
            used,
            shared_ff,
            norm,
            scale,
            window,
            rope,
            sinks,
            base,
            limit_exp,
            limit_shexp,
            limit_dense,
            defaults,
        };
        if hp.rope.iter().any(|&r| r) {
            hp.defaults.push(rules.rope_pairs_line());
        }
        if let Some(line) = rules.chat_line() {
            hp.defaults.push(line);
        }
        for l in hp.described() {
            check_kv(split, hp.heads, l, hp.kv[l])?;
            mul(split, "attention.key_length", hp.kv[l], hp.head)?;
            if hp.dense(l) {
                if hp.ff[l] == 0 {
                    return Err(metadata(
                        split,
                        "feed_forward_length",
                        format!("is 0 at dense layer {l}"),
                    ));
                }
            } else {
                if hp.expert_ff[l] == 0 {
                    return Err(metadata(
                        split,
                        "expert_feed_forward_length",
                        format!("is 0 at routed layer {l}"),
                    ));
                }
                let combined = mul(split, "expert_shared_count", hp.expert_ff[l], shared_count)?;
                if combined != hp.shared_ff {
                    return Err(metadata(
                        split,
                        "expert_shared_feed_forward_length",
                        format!(
                            "is {}, not expert_feed_forward_length * expert_shared_count \
                             {combined} at layer {l}",
                            hp.shared_ff
                        ),
                    ));
                }
            }
        }
        let described = hp.described();
        let kinds: Vec<roles::Kind> = described
            .iter()
            .map(|&l| {
                if l >= hp.trunk {
                    roles::Kind::Nextn
                } else if l < hp.dense_lead {
                    roles::Kind::Dense
                } else {
                    roles::Kind::Routed
                }
            })
            .collect();
        layers_agree(
            split,
            &kinds,
            |kind| roles::groups(kind, hp.sinks),
            &[roles::Kind::Dense, roles::Kind::Routed, roles::Kind::Nextn],
        )?;
        dims_agree(split, hidden, vocab, described.iter().map(|&l| hp.dims(l)))?;
        types_agree(split, blocks)?;
        Ok(hp)
    }

    pub(super) fn dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, d, q, kv) = (
            self.hidden as u64,
            self.head as u64,
            self.heads as u64 * self.head as u64,
            self.kv[l] as u64 * self.head as u64,
        );
        let mut out = vec![
            ("attn_norm.weight", vec![e]),
            ("attn_q.weight", vec![e, q]),
            ("attn_k.weight", vec![e, kv]),
            ("attn_v.weight", vec![e, kv]),
            ("attn_output.weight", vec![q, e]),
            ("attn_q_norm.weight", vec![d]),
            ("attn_k_norm.weight", vec![d]),
            ("ffn_norm.weight", vec![e]),
        ];
        if self.sinks {
            out.push(("attn_sinks.weight", vec![self.heads as u64]));
        }
        if self.dense(l) {
            let ff = self.ff[l] as u64;
            out.extend([
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
            if l >= self.trunk {
                // The next-token layer's input join and its two norms, beside a
                // head norm it may share the trunk's head through.
                out.extend([
                    ("nextn.eh_proj.weight", vec![2 * e, e]),
                    ("nextn.enorm.weight", vec![e]),
                    ("nextn.hnorm.weight", vec![e]),
                    ("nextn.shared_head_norm.weight", vec![e]),
                ]);
            }
        } else {
            let (ff, experts, shared) = (
                self.expert_ff[l] as u64,
                self.experts as u64,
                self.shared_ff as u64,
            );
            out.extend([
                ("ffn_gate_inp.weight", vec![e, experts]),
                (roles::SELECTION_BIAS, vec![experts]),
                ("ffn_gate_exps.weight", vec![e, ff, experts]),
                ("ffn_up_exps.weight", vec![e, ff, experts]),
                ("ffn_down_exps.weight", vec![ff, e, experts]),
                ("ffn_gate_shexp.weight", vec![e, shared]),
                ("ffn_up_shexp.weight", vec![e, shared]),
                ("ffn_down_shexp.weight", vec![shared, e]),
            ]);
        }
        out
    }
}

/// Each block's sliding window: `attention.sliding_window` as an array of one
/// count per block (0: the block attends every position), or the scalar with
/// `attention.sliding_window_pattern` naming the windowed blocks. Where the
/// file carries the array and the pattern both, each layer's entries must agree
/// (`window > 0` exactly where the pattern says), and a scalar without the
/// pattern names no layer's window.
fn windows(split: &Split, blocks: usize) -> Result<Vec<Option<u32>>, PlacementError> {
    let key = "attention.sliding_window";
    let one = |v: usize| {
        u32::try_from(v).map_err(|_| {
            metadata(
                split,
                key,
                format!("is {v}, which does not fit the spec's u32"),
            )
        })
    };
    let scalar = !matches!(split.value(&split.arch_key(key)), Some(Value::Array(_)));
    let pattern = match split.value(&split.arch_key("attention.sliding_window_pattern")) {
        Some(_) => Some(per_layer_bool(
            split,
            "attention.sliding_window_pattern",
            blocks,
        )?),
        None => None,
    };
    let window: Vec<Option<u32>> = if scalar {
        let Some(pattern) = &pattern else {
            return Err(metadata(
                split,
                "attention.sliding_window_pattern",
                "is absent; a scalar sliding_window names no layer's window without it",
            ));
        };
        let shared = one(positive(split, key)?)?;
        pattern.iter().map(|&swa| swa.then_some(shared)).collect()
    } else {
        per_layer_count(split, key, blocks)?
            .into_iter()
            .map(|v| Ok::<_, PlacementError>((v > 0).then_some(one(v)?)))
            .collect::<Result<Vec<_>, _>>()?
    };
    if let Some(pattern) = &pattern
        && !scalar
        && let Some(l) = window
            .iter()
            .zip(pattern)
            .position(|(w, &swa)| swa != w.is_some())
    {
        return Err(metadata(
            split,
            "attention.sliding_window_pattern",
            format!(
                "says {} at layer {l}, where the sliding_window array says a window of {:?} \
                 (window > 0 exactly where the pattern says)",
                pattern[l], window[l]
            ),
        ));
    }
    Ok(window)
}

/// Whether each block's head turns: `attention.rope_pattern` when the file
/// carries it, else the description row's fallback, whose absence is refused
/// naming the key and the row.
fn rope_pattern(
    split: &Split,
    rules: &ArchRules,
    blocks: usize,
    window: &[Option<u32>],
    defaults: &mut Vec<String>,
) -> Result<Vec<bool>, PlacementError> {
    if split
        .value(&split.arch_key("attention.rope_pattern"))
        .is_some()
    {
        return per_layer_bool(split, "attention.rope_pattern", blocks);
    }
    match rules.rope_fallback {
        Some(RopeFallback::WhereWindowed) => {
            defaults.push(rules.rope_fallback_line());
            Ok(window.iter().map(|w| w.is_some()).collect())
        }
        None => Err(metadata(
            split,
            "attention.rope_pattern",
            format!(
                "is not in the file and no description row of {} states it",
                rules.arch.name()
            ),
        )),
    }
}

/// Each block's three SwiGLU limits: the routed experts', the shared expert's
/// and the dense block's.
type Limits = (Vec<f32>, Vec<f32>, Vec<f32>);

/// The SwiGLU limits: `swiglu_clamp_exp`, `swiglu_clamp_shexp` and the dense
/// `swiglu_clamp`, each a shared f32 or one float per block (llama.cpp's
/// `get_key_or_arr`), 0 where the file carries no key (its fill, recorded) and
/// the shared limit falling back to the routed one where only that is carried.
/// A negative or non-finite entry is refused by name.
fn clamps(
    split: &Split,
    blocks: usize,
    defaults: &mut Vec<String>,
) -> Result<Limits, PlacementError> {
    let exp = clamp_of(split, "swiglu_clamp_exp", blocks, defaults)?;
    let shexp = if split.value(&split.arch_key("swiglu_clamp_shexp")).is_some() {
        clamp_of(split, "swiglu_clamp_shexp", blocks, defaults)?
    } else {
        defaults.push(
            "swiglu_clamp_shexp = swiglu_clamp_exp (llama.cpp deepseek4.cpp:29-31: the shared \
             limit falls back to the routed one)"
                .to_string(),
        );
        exp.clone()
    };
    let dense = clamp_of(split, "swiglu_clamp", blocks, defaults)?;
    Ok((exp, shexp, dense))
}

/// One clamp key's per-block limits (llama.cpp's fill where the file carries
/// none), each finite and at least 0.
fn clamp_of(
    split: &Split,
    key: &str,
    blocks: usize,
    defaults: &mut Vec<String>,
) -> Result<Vec<f32>, PlacementError> {
    let Some(values) = per_layer_f32(split, key, blocks)? else {
        defaults.push(format!(
            "{key} = 0 (llama.cpp's fill, llama-model.cpp:1555-1557)"
        ));
        return Ok(vec![0.0; blocks]);
    };
    for (l, v) in values.iter().enumerate() {
        if !v.is_finite() || *v < 0.0 {
            return Err(metadata(split, key, format!("is {v} at layer {l}")));
        }
    }
    Ok(values)
}

/// Every tensor of `split` is held in a type the file's own architecture reads
/// its stem in (`gguf::arch::allowed_types`, the stem tables of the
/// architectures this reader serves that carry one): the first tensor of
/// another type, by name, with the types the stem takes and how many more
/// there are. A stem the architecture's table does not hold is not this
/// check's: the layer groups and the classifier list it.
fn types_agree(split: &Split, blocks: usize) -> Result<(), PlacementError> {
    let Some(arch) = split.architecture() else {
        return Ok(());
    };
    let mut off: Vec<(String, String)> = Vec::new();
    for (_, t) in split.iter_tensors() {
        let stem = crate::arch::classify::block_stem(&t.name, blocks)
            .map_or(t.name.as_str(), |(_, stem)| stem);
        let Some(allowed) = gguf::arch::allowed_types(arch, stem) else {
            continue;
        };
        if !allowed.contains(&t.ty) {
            let list: Vec<String> = allowed.iter().map(ToString::to_string).collect();
            off.push((
                t.name.clone(),
                format!(
                    "{} is {} (the {arch} reader runs {stem} as {})",
                    t.name,
                    t.ty,
                    list.join(" or ")
                ),
            ));
        }
    }
    match off.first() {
        None => Ok(()),
        Some((name, _)) => Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is of a type the {arch} reader does not run the tensor in ({} such: {})",
                off.len(),
                off.iter()
                    .map(|(_, d)| d.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}
