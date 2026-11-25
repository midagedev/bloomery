//! Inkling 2's DSpark draft: its target's own `config.json` states it under `mtp_config`, beside the
//! target's `text_config`, and `mtp.safetensors` in the target's folder holds it, every tensor BF16.
//! [`read`] takes the two configs into the common shape ([`DsparkShape`]), the layers' numbers
//! ([`Backbone`]) and the target's facts the draft is held to ([`Target`]); [`tensors`] states every
//! tensor the draft reads, [`check`] holds a file's header to that statement and [`plan`] states what
//! a load builds beside the target's head. The reference is llama.cpp PR #32218
//! (`src/models/dspark.cpp`: `load_tensors_inkling2`, `graph_inkling2`, `build_inkling2_head`).
//!
//! The file in the common shape's terms:
//! - **Taps.** `target_layer_ids` count the target's layer applications in run order, a looped
//!   core's layers once a pass; the last id is the application count, the row past the output mix
//!   and norm ([`TapRow::Final`]). They stand as the file writes them: the reference taps visit
//!   `il`'s input row as `t_layer_inp[il]` and the normed final row as `t_layer_inp[n_layer]`.
//! - **The block.** Row 0 is the anchor: the target's embedding row of the token at the target's
//!   position, RMS-normed by the target's embedding norm. Rows 1.. are the file's `mask_embed` row,
//!   which the layers embed; no mask token is read, so `mask_id` is 0, as for Mistral Large 4's
//!   draft. A block row attends to every row of the block.
//! - **The context.** No `fc`: `fuse_weight` holds a row of one weight a tap for each draft layer
//!   and one more for the head ([`Context::Fuse`]). A mix is its row's softmax over the tapped rows,
//!   RMS-normed by its layer's `dflare_norm` (the head's: `markov_head.dflare_norm`). Each layer
//!   projects its mix to keys and values of its own (`k_proj_ctx`, `v_proj_ctx`, `k_norm_ctx`;
//!   [`Backbone::ctx_kv`]), roped at the position.
//! - **The layers.** Qwen3-shaped GQA with q and k norms and the whole head rotated at `rope_theta`;
//!   the layers `local_layer_ids` names read `sliding_window_size` positions back (the reference's
//!   mask reads the anchor's own position on top: its `n_swa` is the window plus one), the others
//!   the whole context; a SiLU-gated MLP.
//! - **The head.** `norm`, the target's output matrix over the block's rows, then the recurrent
//!   Markov head of `markov_rank` in the anchor form ([`RnnRow`]): `rnn_proj` over `[state |
//!   markov_w1[prev] | normed row | the head's mix]` to three `markov_rank` chunks plus its bias;
//!   the correction `hidden_out · tanh(e) + b` through the target's output matrix, which the load
//!   folds ([`runtime::dspark::Fold`]); the target's `final_logit_softcapping` on the sum. The ids
//!   are the target's unpadded vocabulary (`unpadded_vocab_size`); `markov_w1` holds a row for each
//!   padded id (`vocab_size`).

use gguf::{GgmlType, Split, Value as GgufValue};
use models::{LoopSpec, Visit, visits};
use runtime::dspark::{
    BaseHead, Context, DraftVocab, DsparkShape, Fold, HeadKind, HeadMatrix, HeadPlace, RnnCap,
    RnnOut, RnnRow, RnnStart, TapRow,
};
use safetensors::{Dtype, Safetensors};
use serde_json::Value;

use super::{Backbone, Error, field, flag, ids, kind, size, uint};

/// The head `markov_head_type` names, the one this reader takes.
const HEAD_TYPE: &str = "rnn_hidden";

/// The recurrent row of the file's head (the module doc's head).
const ROW: RnnRow = RnnRow {
    start: RnnStart::Anchor,
    ctx: true,
    cap: RnnCap::Sum,
    out: RnnOut::Hidden,
};

/// The target's facts the draft is held to: its `text_config`, or a GGUF target's header.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    /// The hidden width: a tapped row's, and that of the draft's rows the target's head reads.
    pub hidden: usize,
    /// `vocab_size`: the output matrix's rows, and `markov_w1`'s.
    pub padded_vocab: usize,
    /// `unpadded_vocab_size`: the tokenizer's ids, the ones a drafted row scores.
    pub unpadded_vocab: usize,
    /// `final_logit_softcapping`: the cap on a drafted row's sum.
    pub softcap: f32,
    /// The trunk's layer applications in run order ([`models::visits`]).
    pub applications: Vec<Visit>,
}

impl Target {
    /// A target `hidden` wide with `padded_vocab` output rows over `unpadded_vocab` ids, capped at
    /// `softcap`, whose `n_layer` stored layers run as `loop_` states (`None`: once each). A cap
    /// that is not finite and positive, ids outside `1..=padded_vocab` and a loop outside the
    /// layers are refused by name.
    pub fn new(
        hidden: usize,
        (padded_vocab, unpadded_vocab): (usize, usize),
        softcap: f32,
        n_layer: usize,
        loop_: Option<LoopSpec>,
    ) -> Result<Target, Error> {
        if let Some(l) = loop_
            && (l.core_start >= l.core_end || l.core_end as usize > n_layer || l.passes == 0)
        {
            return Err(Error::Unsupported(format!(
                "a loop over layers {}..{} of {n_layer}, {} passes",
                l.core_start, l.core_end, l.passes
            )));
        }
        if !softcap.is_finite() || softcap <= 0.0 {
            return Err(Error::Unsupported(format!(
                "`final_logit_softcapping` is {softcap}; the head caps its sum by a finite positive cap"
            )));
        }
        if unpadded_vocab == 0 || unpadded_vocab > padded_vocab {
            return Err(Error::Unsupported(format!(
                "`unpadded_vocab_size` is {unpadded_vocab} of a padded {padded_vocab}"
            )));
        }
        Ok(Target {
            hidden,
            padded_vocab,
            unpadded_vocab,
            softcap,
            applications: visits(n_layer, loop_),
        })
    }
}

/// The draft as its target's config states it.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub shape: DsparkShape,
    /// The recurrent head's rank, `markov_rank`: the shape's head's.
    pub rank: usize,
    pub backbone: Backbone,
    pub target: Target,
    /// Each tap against the target's applications ([`DsparkShape::tap_rows`]).
    pub taps: Vec<TapRow>,
}

/// What a load builds of the draft beside the target's resident output matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The target's output matrix, read in place: the block's base projection and the fold's.
    pub base: BaseHead,
    /// `W2 = output · hidden_out` and `c = output · b`, over the unpadded ids.
    pub fold: Fold,
    /// `fuse_weight`'s rows and row length: a mix a draft layer and the head's, a weight a tap.
    pub fuse: [usize; 2],
    /// `rnn_proj`'s rows and row length.
    pub joint: [usize; 2],
    /// `markov_w1`'s rows: one a padded id.
    pub w1_rows: usize,
}

/// The draft's numbers as a file states them, before the checks both forms share ([`draft`]).
struct Stated {
    hidden: usize,
    layers: usize,
    /// Per layer, whether it reads the window back.
    windowed: Vec<bool>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f32,
    rope_theta: f64,
    window: usize,
    block: usize,
    taps: Vec<usize>,
    rank: usize,
    /// The ids a drafted row scores.
    unpadded: usize,
}

/// The draft `s` states over `target`: its shape checked and its taps planned against the target's
/// applications. A width other than the target's, ids other than the target's, a window flag per
/// layer that does not count the layers and heads that do not group are refused by name.
fn draft(s: Stated, target: Target) -> Result<Draft, Error> {
    if s.hidden != target.hidden {
        return Err(Error::Unsupported(format!(
            "a draft {} wide over a target {} wide: the target's head reads the draft's rows",
            s.hidden, target.hidden
        )));
    }
    if s.unpadded != target.unpadded_vocab {
        return Err(Error::Unsupported(format!(
            "a draft over {} ids for a target of {}",
            s.unpadded, target.unpadded_vocab
        )));
    }
    if s.windowed.len() != s.layers {
        return Err(Error::Unsupported(format!(
            "{} window flags for {} layers",
            s.windowed.len(),
            s.layers
        )));
    }
    if s.kv_heads == 0 || !s.heads.is_multiple_of(s.kv_heads) {
        return Err(Error::Unsupported(format!(
            "{} query heads over {} key/value heads: a whole group a key/value head",
            s.heads, s.kv_heads
        )));
    }
    let shape = DsparkShape {
        block: s.block,
        row0_drafts: false,
        taps: s.taps,
        hidden: s.hidden,
        context: Context::Fuse {
            mixes: s.layers + 1,
        },
        vocab: DraftVocab::Full { n: s.unpadded },
        mask_id: 0,
        window: s.windowed.iter().all(|&w| w).then_some(s.window),
        head: HeadKind::MarkovRnn {
            rank: s.rank,
            row: ROW,
        },
    };
    shape.check()?;
    let taps = shape.tap_rows(&target.applications)?;
    let backbone = Backbone {
        layers: s.layers,
        hidden: s.hidden,
        heads: s.heads,
        kv_heads: s.kv_heads,
        head_dim: s.head_dim,
        intermediate: s.intermediate,
        eps: s.eps,
        rope_theta: s.rope_theta,
        window: Some(s.window),
        windowed: s.windowed,
        block_causal: false,
        ctx_kv: true,
    };
    Ok(Draft {
        shape,
        rank: s.rank,
        backbone,
        target,
        taps,
    })
}

fn section<'a>(root: &'a Value, key: &'static str) -> Result<&'a Value, Error> {
    let v = field(root, key)?;
    if v.is_object() {
        Ok(v)
    } else {
        Err(Error::Type {
            key,
            got: kind(v),
            want: "an object",
        })
    }
}

fn number(obj: &Value, key: &'static str) -> Result<f64, Error> {
    let v = field(obj, key)?;
    v.as_f64().ok_or_else(|| Error::Type {
        key,
        got: kind(v),
        want: "a number",
    })
}

fn word(obj: &Value, key: &'static str) -> Result<u32, Error> {
    let n = uint(obj, key)?;
    u32::try_from(n).map_err(|_| Error::Type {
        key,
        got: n.to_string(),
        want: "a 32-bit count",
    })
}

/// The target's facts: its width, its two vocabularies, its cap and its applications, the loop of
/// `loop_config` expanded when the config states one.
fn target(text: &Value) -> Result<Target, Error> {
    let n_layer = size(text, "num_hidden_layers")?;
    let loop_ = match text.get("loop_config").filter(|v| !v.is_null()) {
        None => None,
        Some(lp) => {
            if flag(lp, "is_varloop", false)? {
                return Err(Error::Unsupported(
                    "`loop_config.is_varloop` is true; the taps count a fixed loop's applications"
                        .into(),
                ));
            }
            Some(LoopSpec {
                core_start: word(lp, "core_start")?,
                core_end: word(lp, "core_end")?,
                passes: word(lp, "num_passes")?,
                gate_rank: word(lp, "gate_rank")?,
            })
        }
    };
    Target::new(
        size(text, "hidden_size")?,
        (
            size(text, "vocab_size")?,
            size(text, "unpadded_vocab_size")?,
        ),
        number(text, "final_logit_softcapping")? as f32,
        n_layer,
        loop_,
    )
}

/// The draft `config_json` (the target's whole config) states, its shape checked and its taps
/// planned against the target's applications. A head type other than `rnn_hidden`, a projection
/// bias, a width other than the target's and a layer list past the layers are refused by name.
pub fn read(config_json: &str) -> Result<Draft, Error> {
    let root: Value = serde_json::from_str(config_json).map_err(|e| Error::Json(e.to_string()))?;
    let mtp = section(&root, "mtp_config")?;
    let text = section(&root, "text_config")?;
    let target = target(text)?;
    let head = field(mtp, "markov_head_type")?;
    if head.as_str() != Some(HEAD_TYPE) {
        return Err(Error::Unsupported(format!(
            "`markov_head_type` is {}; the reader takes {HEAD_TYPE:?}",
            kind(head)
        )));
    }
    for key in ["q_bias", "o_bias"] {
        if flag(mtp, key, false)? {
            return Err(Error::Unsupported(format!(
                "`{key}` is true; the draft's projections carry no bias"
            )));
        }
    }
    let layers = size(mtp, "num_hidden_layers")?;
    let local = ids(mtp, "local_layer_ids")?;
    if let Some(l) = local.iter().find(|&&l| l >= layers) {
        return Err(Error::Unsupported(format!(
            "`local_layer_ids` names layer {l} of {layers}"
        )));
    }
    let stated = Stated {
        hidden: size(mtp, "hidden_size")?,
        layers,
        windowed: (0..layers).map(|l| local.contains(&l)).collect(),
        heads: size(mtp, "num_attention_heads")?,
        kv_heads: size(mtp, "num_key_value_heads")?,
        head_dim: size(mtp, "head_dim")?,
        intermediate: size(mtp, "intermediate_size")?,
        eps: number(mtp, "rms_norm_eps")? as f32,
        rope_theta: number(mtp, "rope_theta")?,
        window: size(mtp, "sliding_window_size")?,
        block: size(mtp, "block_size")?,
        taps: ids(mtp, "target_layer_ids")?,
        rank: size(mtp, "markov_rank")?,
        unpadded: target.unpadded_vocab,
    };
    draft(stated, target)
}

/// The key `dflash.dspark.markov_rank`'s suffix: with [`GGUF_FUSE`], what makes a `dflash` file this
/// draft's GGUF.
const GGUF_RANK: &str = "dspark.markov_rank";
/// The fused context's tensor in the GGUF ([`gguf_name`] of `fuse_weight`).
const GGUF_FUSE: &str = "dspark_fuse.weight";
/// The architecture string llama.cpp's converter writes for a DSpark draft.
const GGUF_ARCH: &str = "dflash";

/// Whether `split` is this draft's GGUF (llama.cpp's DSpark convention, `dspark-<model>-<quant>.gguf`):
/// a `dflash` file that states the Markov head's rank and holds the fused context. V4.1's, Mistral
/// Large 4's and DFlash2's drafts share the architecture string and hold no fused context.
#[must_use]
pub fn is_gguf(split: &Split) -> bool {
    split.architecture() == Some(GGUF_ARCH)
        && split.value(&split.arch_key(GGUF_RANK)).is_some()
        && split.find(GGUF_FUSE).is_some()
}

/// llama.cpp's converter's name of the draft's tensor `name` (a name of [`tensors`]), or `None` for
/// a name the draft does not read. Each pair was held value for value against the folder's
/// `mtp.safetensors`: the two final gains are told apart by value alone, `norm` the converter's
/// `output_norm` and the head's `dflare_norm` its `dspark_norm`.
#[must_use]
pub fn gguf_name(name: &str) -> Option<String> {
    let model = match name {
        "fuse_weight" => Some(GGUF_FUSE),
        "mask_embed" => Some("mask_embd.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "markov_head.dflare_norm.weight" => Some("dspark_norm.weight"),
        "markov_head.markov_w1.weight" => Some("markov_w1.weight"),
        "markov_head.rnn_proj.weight" => Some("joint_proj.weight"),
        "markov_head.rnn_proj.bias" => Some("joint_proj.bias"),
        "markov_head.hidden_out.weight" => Some("dspark_hidden_out.weight"),
        "markov_head.hidden_out.bias" => Some("dspark_hidden_out.bias"),
        _ => None,
    };
    if let Some(m) = model {
        return Some(m.to_string());
    }
    let (layer, tail) = name.strip_prefix("layers.")?.split_once('.')?;
    let l: usize = layer.parse().ok()?;
    let stem = match tail {
        "dflare_norm.weight" => "dflare_norm.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "self_attn.k_proj_ctx.weight" => "attn_k_ctx.weight",
        "self_attn.v_proj_ctx.weight" => "attn_v_ctx.weight",
        "self_attn.k_norm_ctx.weight" => "attn_k_ctx_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{l}.{stem}"))
}

/// The value of `dflash.<suffix>`, refused by name when missing.
fn key<'a>(split: &'a Split, suffix: &str) -> Result<&'a GgufValue, Error> {
    let k = split.arch_key(suffix);
    split.value(&k).ok_or_else(|| Error::Key {
        key: k,
        detail: "is missing".into(),
    })
}

fn key_err(split: &Split, suffix: &str, v: &GgufValue, want: &str) -> Error {
    Error::Key {
        key: split.arch_key(suffix),
        detail: format!("is {v:?}; want {want}"),
    }
}

fn key_size(split: &Split, suffix: &str) -> Result<usize, Error> {
    let v = key(split, suffix)?;
    v.as_unsigned()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| key_err(split, suffix, v, "a size"))
}

fn key_f32(split: &Split, suffix: &str) -> Result<f32, Error> {
    let v = key(split, suffix)?;
    v.as_f32()
        .ok_or_else(|| key_err(split, suffix, v, "a float"))
}

fn key_bool(split: &Split, suffix: &str) -> Result<bool, Error> {
    let v = key(split, suffix)?;
    v.as_bool()
        .ok_or_else(|| key_err(split, suffix, v, "a boolean"))
}

/// `dflash.<suffix>` as an array, each element read by `f`.
fn key_list<T>(
    split: &Split,
    suffix: &str,
    want: &str,
    f: impl Fn(&GgufValue) -> Option<T>,
) -> Result<Vec<T>, Error> {
    let v = key(split, suffix)?;
    let GgufValue::Array(items) = v else {
        return Err(key_err(split, suffix, v, want));
    };
    items
        .iter()
        .map(|i| f(i).ok_or_else(|| key_err(split, suffix, v, want)))
        .collect()
}

/// The draft the GGUF file `split` states, over `target` (the target's own header or config): the
/// same [`Draft`] the folder's config gives for the same draft. A file that is not this draft's
/// ([`is_gguf`]), a confidence head, a drafting anchor row, a causal block, a rotation of part of a
/// head, keys and values of different widths, and a width, vocabulary or cap other than the
/// target's are refused by name.
pub fn read_gguf(split: &Split, target: Target) -> Result<Draft, Error> {
    if !is_gguf(split) {
        return Err(Error::Unsupported(format!(
            "{:?} with{} `{}` and{} `{GGUF_FUSE}`: not an Inkling 2 DSpark draft",
            split.architecture(),
            if split.value(&split.arch_key(GGUF_RANK)).is_some() {
                ""
            } else {
                "out"
            },
            split.arch_key(GGUF_RANK),
            if split.find(GGUF_FUSE).is_some() {
                ""
            } else {
                "out"
            },
        )));
    }
    for (suffix, want) in [
        ("has_confidence_head", false),
        ("sample_from_anchor", false),
        ("attention.causal", false),
    ] {
        if key_bool(split, suffix)? != want {
            return Err(Error::Key {
                key: split.arch_key(suffix),
                detail: format!("is {}; the draft's anchor form takes {want}", !want),
            });
        }
    }
    let head_dim = key_size(split, "attention.key_length")?;
    for suffix in ["attention.value_length", "rope.dimension_count"] {
        let n = key_size(split, suffix)?;
        if n != head_dim {
            return Err(Error::Key {
                key: split.arch_key(suffix),
                detail: format!("is {n}; the draft's heads are {head_dim} wide, rotated whole"),
            });
        }
    }
    let padded = key_size(split, "vocab_size")?;
    if padded != target.padded_vocab {
        return Err(Error::Key {
            key: split.arch_key("vocab_size"),
            detail: format!(
                "is {padded}; the target's output matrix holds {} rows",
                target.padded_vocab
            ),
        });
    }
    let softcap = key_f32(split, "final_logit_softcapping")?;
    if softcap != target.softcap {
        return Err(Error::Key {
            key: split.arch_key("final_logit_softcapping"),
            detail: format!("is {softcap}; the target caps at {}", target.softcap),
        });
    }
    let stated = Stated {
        hidden: key_size(split, "embedding_length")?,
        layers: key_size(split, "block_count")?,
        windowed: key_list(
            split,
            "attention.sliding_window_pattern",
            "booleans",
            GgufValue::as_bool,
        )?,
        heads: key_size(split, "attention.head_count")?,
        kv_heads: key_size(split, "attention.head_count_kv")?,
        head_dim,
        intermediate: key_size(split, "feed_forward_length")?,
        eps: key_f32(split, "attention.layer_norm_rms_epsilon")?,
        rope_theta: f64::from(key_f32(split, "rope.freq_base")?),
        window: key_size(split, "attention.sliding_window")?,
        block: key_size(split, "block_size")?,
        taps: key_list(split, "target_layers", "sizes", |v| {
            v.as_unsigned().and_then(|n| usize::try_from(n).ok())
        })?,
        rank: key_size(split, GGUF_RANK)?,
        unpadded: key_size(split, "vocab_size_unpadded")?,
    };
    draft(stated, target)
}

/// Every tensor the draft reads, by the file's names, each with its shape rows first.
#[must_use]
pub fn tensors(d: &Draft) -> Vec<(String, Vec<usize>)> {
    let b = &d.backbone;
    let (h, hd, ff, r) = (b.hidden, b.head_dim, b.intermediate, d.rank);
    let (q, kv) = (b.heads * hd, b.kv_heads * hd);
    let model = |name: &str, shape: &[usize]| (name.to_string(), shape.to_vec());
    let mut out = vec![
        (
            "fuse_weight".to_string(),
            d.shape.fuse_dims().map_or_else(Vec::new, Vec::from),
        ),
        model("mask_embed", &[h]),
        model("norm.weight", &[h]),
        model("markov_head.dflare_norm.weight", &[h]),
        model("markov_head.markov_w1.weight", &[d.target.padded_vocab, r]),
        model("markov_head.rnn_proj.weight", &[3 * r, ROW.joint_in(r, h)]),
        model("markov_head.rnn_proj.bias", &[3 * r]),
        model("markov_head.hidden_out.weight", &[h, r]),
        model("markov_head.hidden_out.bias", &[h]),
    ];
    for l in 0..b.layers {
        let at = |name: &str, shape: &[usize]| (format!("layers.{l}.{name}"), shape.to_vec());
        out.extend([
            at("dflare_norm.weight", &[h]),
            at("input_layernorm.weight", &[h]),
            at("post_attention_layernorm.weight", &[h]),
            at("self_attn.q_proj.weight", &[q, h]),
            at("self_attn.k_proj.weight", &[kv, h]),
            at("self_attn.v_proj.weight", &[kv, h]),
            at("self_attn.o_proj.weight", &[h, q]),
            at("self_attn.q_norm.weight", &[hd]),
            at("self_attn.k_norm.weight", &[hd]),
            at("self_attn.k_proj_ctx.weight", &[kv, h]),
            at("self_attn.v_proj_ctx.weight", &[kv, h]),
            at("self_attn.k_norm_ctx.weight", &[hd]),
            at("mlp.gate_proj.weight", &[ff, h]),
            at("mlp.up_proj.weight", &[ff, h]),
            at("mlp.down_proj.weight", &[h, ff]),
        ]);
    }
    out
}

/// `file`'s header against [`tensors`]: each tensor BF16 at its shape, none missing and none the
/// draft does not read, the first that differs refused by name.
pub fn check(d: &Draft, file: &Safetensors) -> Result<(), Error> {
    let want = tensors(d);
    for (name, shape) in &want {
        let e = file.entry(name).ok_or_else(|| Error::Tensor {
            name: name.clone(),
            detail: "is missing".into(),
        })?;
        if e.dtype != Dtype::Bf16 || e.shape != *shape {
            return Err(Error::Tensor {
                name: name.clone(),
                detail: format!(
                    "is {} {:?}; the draft reads BF16 {shape:?}",
                    e.dtype.name(),
                    e.shape
                ),
            });
        }
    }
    match file.names().find(|n| !want.iter().any(|(w, _)| w == n)) {
        Some(extra) => Err(Error::Tensor {
            name: extra.to_string(),
            detail: "is not one the draft reads".into(),
        }),
        None => Ok(()),
    }
}

/// The GGUF `file`'s header against [`tensors`] under [`gguf_name`]: each tensor at its shape (ggml's
/// `ne` reversed) in a type the load reads (Q8_0, BF16 or F32: `Bf16Source`'s GGUF reader), none
/// missing and none the draft does not read, the first that differs refused by name.
pub fn check_gguf(d: &Draft, file: &Split) -> Result<(), Error> {
    let want = tensors(d);
    let mut names = Vec::with_capacity(want.len());
    for (name, shape) in &want {
        let refuse = |detail: String| Error::Tensor {
            name: name.clone(),
            detail,
        };
        let g = gguf_name(name).ok_or_else(|| refuse("has no name in the GGUF draft".into()))?;
        let (_, t) = file
            .find(&g)
            .ok_or_else(|| refuse(format!("is missing (the file's {g})")))?;
        let got: Vec<usize> = t.dims.iter().rev().map(|&n| n as usize).collect();
        if got != *shape || ![GgmlType::Q8_0, GgmlType::BF16, GgmlType::F32].contains(&t.ty) {
            return Err(refuse(format!(
                "is {g}, {} {got:?}; the draft reads Q8_0, BF16 or F32 {shape:?}",
                t.ty
            )));
        }
        names.push(g);
    }
    match file.iter_tensors().find(|(_, t)| !names.contains(&t.name)) {
        Some((_, extra)) => Err(Error::Tensor {
            name: extra.name.clone(),
            detail: "is not one the draft reads".into(),
        }),
        None => Ok(()),
    }
}

/// What a load builds beside the target's resident output matrix of type `matrix`, read in place.
pub fn plan(d: &Draft, matrix: HeadMatrix) -> Result<Plan, Error> {
    let base = BaseHead::plan(matrix, HeadPlace::InPlace)?;
    let fold = d
        .shape
        .fold()
        .ok_or_else(|| Error::Unsupported("a head with no fold".into()))?;
    let fuse = d
        .shape
        .fuse_dims()
        .ok_or_else(|| Error::Unsupported("a context with no fuse".into()))?;
    Ok(Plan {
        base,
        fold,
        fuse,
        joint: [3 * d.rank, ROW.joint_in(d.rank, d.backbone.hidden)],
        w1_rows: d.target.padded_vocab,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small looped target (4 layers, the core 1..3 run twice: 6 applications) and a draft of 2
    /// layers over 4 taps, the last the final row; the fuse is then 3 mixes of 4 weights.
    const CONFIG: &str = r#"{
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 96,
            "unpadded_vocab_size": 90, "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 1, "core_end": 3, "num_passes": 2, "gate_rank": 8,
                            "is_varloop": false}
        },
        "mtp_config": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0],
            "sliding_window_size": 32, "block_size": 4, "markov_rank": 32,
            "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 4, 5, 6],
            "q_bias": false, "o_bias": false
        }
    }"#;

    /// A file of `tensors` in BF16, zero-filled.
    fn file(tensors: &[(String, Vec<usize>)]) -> Safetensors {
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, s)| vec![0; 2 * s.iter().product::<usize>()])
            .collect();
        let rows: Vec<(&str, &str, &[usize], &[u8])> = tensors
            .iter()
            .zip(&data)
            .map(|((n, s), d)| (n.as_str(), "BF16", s.as_slice(), d.as_slice()))
            .collect();
        Safetensors::parse(safetensors::write(&rows)).expect("a written file parses")
    }

    /// The config reads as the anchor form over the target's loop: taps by application, the last
    /// the final row, the fused context, the per-layer windows and the context's own projections.
    #[test]
    fn the_draft_reads_as_the_anchor_form_over_the_loop() {
        let d = read(CONFIG).expect("the config reads");
        let at = |application, layer, pass| TapRow::Entering {
            application,
            visit: Visit { layer, pass },
        };
        assert_eq!(
            d.taps,
            vec![at(1, 1, 0), at(4, 2, 1), at(5, 3, 0), TapRow::Final]
        );
        let s = &d.shape;
        assert_eq!(
            (s.block, s.row0_drafts, s.width(8), s.rows(3)),
            (4, false, 3, 4)
        );
        assert_eq!(s.context, Context::Fuse { mixes: 3 });
        assert_eq!(s.vocab, DraftVocab::Full { n: 90 });
        assert_eq!(s.head, HeadKind::MarkovRnn { rank: 32, row: ROW });
        assert_eq!(s.window, None);
        let b = &d.backbone;
        assert_eq!(
            (b.window, b.windowed.as_slice(), b.ctx_kv, b.block_causal),
            (Some(32), [true, false].as_slice(), true, false)
        );
        assert_eq!(
            (b.rope_theta, d.target.padded_vocab, d.target.softcap),
            (1e6, 96, 30.0)
        );
        let past = CONFIG.replace("[1, 4, 5, 6]", "[1, 4, 5, 7]");
        assert!(matches!(
            read(&past),
            Err(Error::Shape(runtime::dspark::DsparkError::TapPast {
                tap: 7,
                applications: 6
            }))
        ));
    }

    /// The header is held to the statement: `fuse_weight` a row of one weight a tap for each mix,
    /// so its transpose is refused by name, as are a missing tensor and one the draft does not read.
    #[test]
    fn the_header_is_held_to_the_fuse_of_a_row_a_mix() {
        let d = read(CONFIG).expect("the config reads");
        let want = tensors(&d);
        assert_eq!(want.len(), 9 + 15 * 2);
        assert_eq!(want[0], ("fuse_weight".to_string(), vec![3, 4]));
        assert!(check(&d, &file(&want)).is_ok());
        let mut turned = want.clone();
        turned[0].1 = vec![4, 3];
        assert!(matches!(
            check(&d, &file(&turned)),
            Err(Error::Tensor { name, .. }) if name == "fuse_weight"
        ));
        let missing = &want[1..];
        assert!(matches!(
            check(&d, &file(missing)),
            Err(Error::Tensor { name, .. }) if name == "fuse_weight"
        ));
        let mut extra = want.clone();
        extra.push(("fc.weight".to_string(), vec![64, 256]));
        assert!(matches!(
            check(&d, &file(&extra)),
            Err(Error::Tensor { name, .. }) if name == "fc.weight"
        ));
    }

    /// The plan reads the target's Q8_0 or BF16 head in place and folds the correction over the
    /// unpadded ids; `markov_w1` keeps a row for each padded id.
    #[test]
    fn the_plan_reads_the_targets_head_in_place() {
        let d = read(CONFIG).expect("the config reads");
        for matrix in [HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            let p = plan(&d, matrix).expect("the plan");
            assert_eq!(
                p,
                Plan {
                    base: BaseHead {
                        matrix,
                        place: HeadPlace::InPlace
                    },
                    fold: Fold { rows: 90, rank: 32 },
                    fuse: [3, 4],
                    joint: [96, 192],
                    w1_rows: 96,
                }
            );
        }
    }

    /// [`CONFIG`]'s draft as llama.cpp's converter writes it: its `dflash.` keys, then each tensor
    /// of `tensors` under its GGUF name in the type the file's publisher writes it (the layers'
    /// matrices Q8_0, the head's BF16, every vector and the fuse F32), less `drop` and with `edit`
    /// applied to the keys last.
    fn gguf(
        drop: &str,
        edit: impl Fn(&mut Vec<(String, GgufValue)>),
        tensors: &[(String, Vec<usize>)],
    ) -> Split {
        use gguf::write::{Layout, TensorDecl};
        let u = |n: u32| GgufValue::U32(n);
        let k = |s: &str| format!("dflash.{s}");
        let mut kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                GgufValue::String("dflash".into()),
            ),
            (k("block_count"), u(2)),
            (k("embedding_length"), u(64)),
            (k("feed_forward_length"), u(128)),
            (k("attention.head_count"), u(4)),
            (k("attention.head_count_kv"), u(2)),
            (k("attention.key_length"), u(16)),
            (k("attention.value_length"), u(16)),
            (k("rope.dimension_count"), u(16)),
            (k("rope.freq_base"), GgufValue::F32(1e6)),
            (k("attention.layer_norm_rms_epsilon"), GgufValue::F32(1e-6)),
            (k("vocab_size"), u(96)),
            (k("vocab_size_unpadded"), u(90)),
            (k("block_size"), u(4)),
            (
                k("target_layers"),
                GgufValue::Array([1, 4, 5, 6].map(u).to_vec()),
            ),
            (k("dspark.markov_rank"), u(32)),
            (k("sample_from_anchor"), GgufValue::Bool(false)),
            (k("has_confidence_head"), GgufValue::Bool(false)),
            (k("attention.causal"), GgufValue::Bool(false)),
            (k("attention.sliding_window"), u(32)),
            (
                k("attention.sliding_window_pattern"),
                GgufValue::Array(vec![GgufValue::Bool(true), GgufValue::Bool(false)]),
            ),
            (k("final_logit_softcapping"), GgufValue::F32(30.0)),
        ];
        edit(&mut kvs);
        let decls = tensors
            .iter()
            .filter(|(n, _)| n != drop)
            .map(|(n, s)| {
                // A name of no draft tensor stands as the file's own.
                let name = gguf_name(n).unwrap_or_else(|| n.clone());
                let count: usize = s.iter().product();
                let (type_id, nbytes) = match s.len() {
                    1 => (0, 4 * count),
                    _ if n.starts_with("layers.") => (8, count / 32 * 34),
                    _ if n == "fuse_weight" => (0, 4 * count),
                    _ => (30, 2 * count),
                };
                TensorDecl {
                    name,
                    dims: s.iter().rev().map(|&d| d as u64).collect(),
                    type_id,
                    nbytes: nbytes as u64,
                }
            })
            .collect();
        let layout = Layout::new(&kvs, decls).expect("the header lays out");
        let path =
            std::env::temp_dir().join(format!("bloomery-ink2-dspark-{}.gguf", std::process::id()));
        Split::header_only(path, &layout).expect("the header-only split opens")
    }

    /// The GGUF states the folder's draft: over the folder's target, its keys give the same
    /// [`Draft`] the config does, and its header holds every tensor under its GGUF name.
    #[test]
    fn the_gguf_states_the_folders_draft() {
        let d = read(CONFIG).expect("the config reads");
        let want = tensors(&d);
        let file = gguf("", |_| {}, &want);
        assert!(is_gguf(&file));
        assert_eq!(
            read_gguf(&file, d.target.clone()).expect("the GGUF reads"),
            d
        );
        check_gguf(&d, &file).expect("the header holds the draft");
    }

    /// A GGUF for another draft or another target is refused by name: a drafting anchor row, a
    /// confidence head, a causal block, a rotation of part of a head, another output matrix, cap or
    /// width; a file without the fused context is not this draft's; a tensor turned, missing or one
    /// the draft does not read fails the header's check.
    #[test]
    fn a_gguf_for_another_draft_or_target_is_refused_by_name() {
        let d = read(CONFIG).expect("the config reads");
        let want = tensors(&d);
        let set = |key: &'static str, v: GgufValue| {
            move |kvs: &mut Vec<(String, GgufValue)>| {
                let at = kvs
                    .iter_mut()
                    .find(|(k, _)| k == &format!("dflash.{key}"))
                    .expect("the key is written");
                at.1 = v.clone();
            }
        };
        let refused = |key: &'static str, v: GgufValue| {
            let file = gguf("", set(key, v), &want);
            match read_gguf(&file, d.target.clone()) {
                Err(Error::Key { key: k, .. }) => assert_eq!(k, format!("dflash.{key}")),
                other => panic!("{key}: {other:?}"),
            }
        };
        refused("sample_from_anchor", GgufValue::Bool(true));
        refused("has_confidence_head", GgufValue::Bool(true));
        refused("attention.causal", GgufValue::Bool(true));
        refused("rope.dimension_count", GgufValue::U32(8));
        refused("vocab_size", GgufValue::U32(128));
        refused("final_logit_softcapping", GgufValue::F32(50.0));
        let wide = gguf("", set("embedding_length", GgufValue::U32(128)), &want);
        assert!(matches!(
            read_gguf(&wide, d.target.clone()),
            Err(Error::Unsupported(m)) if m.contains("128 wide over a target 64 wide")
        ));
        let other_ids = gguf("", set("vocab_size_unpadded", GgufValue::U32(89)), &want);
        assert!(matches!(
            read_gguf(&other_ids, d.target.clone()),
            Err(Error::Unsupported(m)) if m.contains("89 ids for a target of 90")
        ));
        let unfused = gguf("fuse_weight", |_| {}, &want);
        assert!(!is_gguf(&unfused));
        assert!(matches!(
            read_gguf(&unfused, d.target.clone()),
            Err(Error::Unsupported(_))
        ));
        let mut turned = want.clone();
        turned[0].1 = vec![4, 3];
        assert!(matches!(
            check_gguf(&d, &gguf("", |_| {}, &turned)),
            Err(Error::Tensor { name, .. }) if name == "fuse_weight"
        ));
        assert!(matches!(
            check_gguf(&d, &gguf("mask_embed", |_| {}, &want)),
            Err(Error::Tensor { name, .. }) if name == "mask_embed"
        ));
        let mut extra = want.clone();
        extra.push(("blk.0.attn_q_ctx.weight".into(), vec![64, 64]));
        assert!(matches!(
            check_gguf(&d, &gguf("", |_| {}, &extra)),
            Err(Error::Tensor { name, .. }) if name == "blk.0.attn_q_ctx.weight"
        ));
    }

    /// The Inkling 2 Mini's folder (`$BLOOMERY_INKLING2_HF`): its config reads, `mtp.safetensors`'s
    /// header holds every tensor the draft reads and nothing else, and the plan is the design's.
    #[test]
    #[ignore = "needs the Inkling 2 Mini folder's config.json and mtp.safetensors at $BLOOMERY_INKLING2_HF"]
    fn hw_the_inkling2_draft_header_plans() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("BLOOMERY_INKLING2_HF")
                .expect("$BLOOMERY_INKLING2_HF names the folder"),
        );
        let text = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let d = read(&text).expect("the config reads");
        let file = Safetensors::open(&dir.join("mtp.safetensors")).expect("mtp.safetensors");
        check(&d, &file).expect("the header holds the draft");
        let at = |application, layer, pass| TapRow::Entering {
            application,
            visit: Visit { layer, pass },
        };
        assert_eq!(d.target.applications.len(), 48);
        assert_eq!(
            d.taps,
            vec![at(5, 5, 0), at(23, 11, 1), at(41, 17, 2), TapRow::Final]
        );
        assert_eq!((d.shape.width(8), d.shape.rows(7)), (7, 8));
        assert_eq!(d.backbone.windowed, vec![true, false, true]);
        assert_eq!((d.backbone.rope_theta, d.target.softcap), (1e6, 30.0));
        for matrix in [HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            let p = plan(&d, matrix).expect("the plan");
            assert_eq!(
                (p.fuse, p.joint, p.w1_rows, p.base.own_bytes(201_024, 2304)),
                ([4, 4], [768, 5120], 201_024, 0)
            );
            assert_eq!(
                p.fold,
                Fold {
                    rows: 200_058,
                    rank: 256
                }
            );
            assert_eq!((p.fold.w2_bytes(), p.fold.c_bytes()), (54_415_776, 800_232));
        }
        let want = tensors(&d);
        assert_eq!(want.len(), 9 + 15 * 3);
        let bytes: usize = want
            .iter()
            .map(|(_, s)| 2 * s.iter().product::<usize>())
            .sum();
        println!(
            "inkling2 draft: {} tensors, {bytes} BF16 bytes, taps {:?}",
            want.len(),
            d.shape.taps
        );
    }
}
