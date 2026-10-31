//! GLM-5.3-Flash's Hugging Face checkpoint in the names and layouts the glm5next program reads,
//! as llama.cpp's converter (`conversion/glm.py`, `Glm5NextModel`) writes them into a GGUF.
//!
//! A checkpoint tensor maps to one glm5next tensor ([`names`]) and a [`Layout`]: as stored; the
//! routed experts' tensors of one projection stacked by expert id; `kv_b_proj` split per head
//! into the key half, transposed, and the value half; the KDA conv taps as `[kernel, 1,
//! channels]`; and `A_log` as `-exp`. `dt_bias` becomes the `ssm_dt` bias, and both
//! `model.language_model.` and `model.` prefixes are read. The next-token layers (index
//! `num_hidden_layers` and up) and the vision tower are left out; any other name is refused.
//!
//! Shapes are logical element counts in ggml's order, the fastest axis first; a packed
//! weight's caller gives its unpacked shape.
//!
//! Tensor parallel splits each mapped tensor by the family's rule ([`tp::cut_of`]); the bytes and
//! the blocks a cut keeps are its format's ([`Format::held`]), along K at least its card layout's
//! unit, and the plan is [`plan_tp`](model::placement::tp::plan_tp)'s. [`open`] gives the folder
//! as a load reads it ([`Checkpoint`]): each glm5next tensor made of its checkpoint
//! tensors, its layout the move after they are normalized ([`Layout::apply`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use gguf::GgmlType;
use hf::modelopt::{Normal, Plain, Use};
use hf::quant_config::{self, Config, Format};
use model::arch::glm5next::names::{self, Sub};
use model::arch::glm5next::tp;
use model::placement::tp::TpTensor;
use safetensors::{Dtype, Folder};
use serde_json::Value;

use super::module_class;
use crate::part::{self, Checkpoint, Made};

/// Why the checkpoint was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("glm5next hf: config.json: {0}")]
    Config(String),
    #[error("glm5next hf: {name}: {what}")]
    Tensor { name: String, what: String },
    #[error("glm5next hf: {0}")]
    Folder(#[from] safetensors::Error),
    #[error("glm5next hf: {0}")]
    Quant(#[from] quant_config::Error),
    #[error("glm5next hf: {0}")]
    Part(#[from] part::Error),
}

fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// The config values the map reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Trunk layers; the next-token layers follow.
    pub layers: usize,
    pub heads: usize,
    pub nope: usize,
    pub v: usize,
    pub experts: usize,
}

impl Geometry {
    /// `config.json`'s values, from `text_config` when the checkpoint nests them there.
    pub fn of(config: &Value) -> Result<Geometry, Error> {
        let c = config.get("text_config").unwrap_or(config);
        let n = |key: &str| {
            c.get(key)
                .and_then(Value::as_u64)
                .filter(|&n| n > 0)
                .map(|n| n as usize)
                .ok_or_else(|| Error::Config(format!("{key} is not a count")))
        };
        Ok(Geometry {
            layers: n("num_hidden_layers")?,
            heads: n("num_attention_heads")?,
            nope: n("qk_nope_head_dim")?,
            v: n("v_head_dim")?,
            experts: n("n_routed_experts")?,
        })
    }
}

/// How a glm5next tensor is made from the checkpoint's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The tensor's elements in stored order.
    Same,
    /// The expert tensors named, in expert id order, one after another.
    Stack(Vec<String>),
    /// `kv_b_proj` `[heads · (nope + v), rank]`: each head's first `nope` rows, transposed to
    /// `[rank, nope]`.
    KeyHalf { heads: usize, nope: usize, v: usize },
    /// Each head's last `v` rows, `[v, rank]`.
    ValueHalf { heads: usize, nope: usize, v: usize },
    /// `-exp(x)` of the first `heads` values, as f32, computed in f64 and rounded once.
    NegExp { heads: usize },
}

impl Layout {
    /// Move `n`, the normalized source of the glm5next tensor `name` (a stack's experts already
    /// one after another), into the layout: its bytes and dims become the tensor's. A layout
    /// that moves single values of a tensor of blocks, `-exp` of other than F32, and a source of
    /// other values than the layout reads are refused by name.
    pub fn apply(&self, name: &str, n: &mut Normal) -> Result<(), Error> {
        let ty = GgmlType::from_u32(n.type_id);
        let size = match ty {
            GgmlType::BF16 => 2,
            GgmlType::F32 => 4,
            _ if matches!(self, Layout::Same | Layout::Stack(_)) => return Ok(()),
            _ => {
                return Err(refuse(
                    name,
                    format!("{ty} holds blocks; {self:?} moves single values"),
                ));
            }
        };
        let held = |want: Option<usize>| {
            want.filter(|&w| w == n.bytes.len()).ok_or_else(|| {
                refuse(
                    name,
                    format!("{self:?} does not read {} bytes of {ty}", n.bytes.len()),
                )
            })
        };
        let (bytes, dims) = match *self {
            Layout::Same | Layout::Stack(_) => return Ok(()),
            Layout::KeyHalf { heads, nope, v } | Layout::ValueHalf { heads, nope, v } => {
                let rows = nope.checked_add(v).and_then(|w| w.checked_mul(heads));
                let rank = n.dims.first().map(|&r| r as usize).unwrap_or(0);
                held(rows.and_then(|r| r.checked_mul(rank)?.checked_mul(size)))?;
                let row = rank * size;
                let head = |h: usize| &n.bytes[h * (nope + v) * row..][..(nope + v) * row];
                if matches!(self, Layout::KeyHalf { .. }) {
                    let mut out = Vec::with_capacity(heads * nope * row);
                    for h in 0..heads {
                        for c in 0..rank {
                            for r in 0..nope {
                                out.extend_from_slice(&head(h)[(r * rank + c) * size..][..size]);
                            }
                        }
                    }
                    (out, vec![nope, rank, heads])
                } else {
                    let out = (0..heads)
                        .flat_map(|h| &head(h)[nope * row..])
                        .copied()
                        .collect();
                    (out, vec![rank, v, heads])
                }
            }
            Layout::NegExp { heads } => {
                if ty != GgmlType::F32 || n.bytes.len() < heads.saturating_mul(4) {
                    return Err(refuse(
                        name,
                        format!(
                            "-exp reads {heads} F32 values, not {ty} of {} bytes",
                            n.bytes.len()
                        ),
                    ));
                }
                let mut out = Vec::with_capacity(heads * 4);
                for (i, &b) in n.bytes.as_chunks::<4>().0[..heads].iter().enumerate() {
                    let y = super::neg_exp(f32::from_le_bytes(b))
                        .map_err(|why| refuse(name, format!("-exp at {i}: {why}")))?;
                    out.extend_from_slice(&y.to_le_bytes());
                }
                (out, vec![heads])
            }
        };
        n.bytes = bytes;
        n.dims = dims.into_iter().map(|d| d as u64).collect();
        Ok(())
    }
}

/// One glm5next tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The checkpoint tensor it is made from (a stack's first expert).
    pub source: String,
    pub layout: Layout,
    /// Logical element counts in ggml's order.
    pub dims: Vec<usize>,
}

/// The glm5next tensors, by name, of the checkpoint tensors `tensors` (name and logical shape,
/// row-major) under `g`. A layer's routed experts not all there or not matrices, and a layer
/// holding part of a set of tensors that come together (`parts`), are refused by name.
pub fn map<'a>(
    g: Geometry,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
) -> Result<BTreeMap<String, Mapped>, Error> {
    let mut out = BTreeMap::new();
    // A layer's experts: (layer, projection) to each expert's tensor name and shape.
    type ById<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;
    let mut experts: BTreeMap<(usize, &str), ById> = BTreeMap::new();
    let mut put = |name: String, m: Mapped| match out.insert(name, m) {
        Some(prev) => Err(refuse(
            &prev.source,
            "maps to a glm5next name another tensor has",
        )),
        None => Ok(()),
    };
    for (name, shape) in tensors {
        let short = text_name(name);
        let same = |dims: Vec<usize>| Mapped {
            source: name.to_string(),
            layout: Layout::Same,
            dims,
        };
        let reversed = || same(shape.iter().rev().copied().collect());
        match short {
            "embed_tokens.weight" => put(names::token_embd(), reversed())?,
            "norm.weight" => put(names::output_norm(), reversed())?,
            "lm_head.weight" => put(names::output(), reversed())?,
            _ if short.starts_with("visual.") || name.starts_with("model.visual.") => {}
            _ => {
                let (l, rest) =
                    text_layer(name).ok_or_else(|| refuse(name, "not a GLM-5.3-Flash tensor"))?;
                if l >= g.layers {
                    continue;
                }
                if let Some(e) = rest.strip_prefix("mlp.experts.") {
                    let (id, proj) = e
                        .split_once('.')
                        .and_then(|(id, p)| {
                            Some((id.parse::<usize>().ok()?, p.strip_suffix(".weight")?))
                        })
                        .filter(|(_, p)| ["gate_proj", "up_proj", "down_proj"].contains(p))
                        .ok_or_else(|| refuse(name, "not a routed expert projection"))?;
                    if let Some((other, _)) = experts
                        .entry((l, proj))
                        .or_default()
                        .insert(id, (name, shape))
                    {
                        return Err(refuse(
                            name,
                            format!("layer {l} {proj} expert {id} is also supplied by {other}"),
                        ));
                    }
                    continue;
                }
                for (glm, m) in layer(g, l, rest, name, shape)? {
                    put(glm, m)?;
                }
            }
        }
    }
    for ((l, proj), by_id) in experts {
        let (sources, dims) = safetensors::expert_stack(&by_id, g.experts)?;
        let glm = match proj {
            "gate_proj" => names::ffn_gate_exps(l),
            "up_proj" => names::ffn_up_exps(l),
            _ => names::ffn_down_exps(l),
        };
        put(
            glm,
            Mapped {
                source: sources[0].clone(),
                layout: Layout::Stack(sources),
                dims,
            },
        )?;
    }
    let layers: BTreeSet<usize> = out.keys().filter_map(|n| layer_of(n)).collect();
    for l in layers {
        for (trigger, required) in parts(l) {
            if let Some(have) = trigger.iter().find(|n| out.contains_key(*n))
                && let Some(missing) = required.iter().find(|n| !out.contains_key(*n))
            {
                return Err(refuse(
                    &out[have].source,
                    format!("layer {l} has {have} but not {missing}"),
                ));
            }
        }
    }
    Ok(out)
}

/// A group's exclusive stems detect it in layer `l`; every required role in it must then be present.
fn parts(l: usize) -> [(Vec<String>, Vec<String>); 6] {
    use model::arch::glm5next::roles;
    let group = |stems: &[roles::Stem], trigger: fn(&str) -> bool| {
        let names = |keep: fn(&str) -> bool| -> Vec<String> {
            roles::required(stems)
                .filter(|s| keep(s))
                .map(|s| format!("blk.{l}.{s}"))
                .collect()
        };
        (names(trigger), names(|_| true))
    };
    let mixer =
        |s: &str| s != "attn_norm.weight" && s != "attn_output.weight" && !s.starts_with("indexer");
    [
        group(roles::KDA, mixer),
        group(roles::LATENT, mixer),
        group(roles::DENSE, |s| s != "ffn_norm.weight"),
        group(roles::SHARED, |_| true),
        group(roles::MOE, |s| s != "ffn_norm.weight"),
        group(roles::HC, |_| true),
    ]
}

/// The glm5next tensors of `rest`, a tensor of layer `l` other than a routed expert.
fn layer(
    g: Geometry,
    l: usize,
    rest: &str,
    name: &str,
    shape: &[usize],
) -> Result<Vec<(String, Mapped)>, Error> {
    let mapped = |layout: Layout, dims: Vec<usize>| Mapped {
        source: name.to_string(),
        layout,
        dims,
    };
    let reversed = || mapped(Layout::Same, shape.iter().rev().copied().collect());
    let one = |glm: String| Ok(vec![(glm, reversed())]);
    match rest {
        "eh_proj.weight" => one(names::nextn_eh_proj(l)),
        "enorm.weight" => one(names::nextn_enorm(l)),
        "hnorm.weight" => one(names::nextn_hnorm(l)),
        "shared_head.norm.weight" => one(names::nextn_shared_head_norm(l)),
        "input_layernorm.weight" => one(names::attn_norm(l)),
        "post_attention_layernorm.weight" => one(names::ffn_norm(l)),
        "self_attn.o_proj.weight" => one(names::attn_output(l)),
        "self_attn.q_proj.weight" => one(names::attn_q(l)),
        "self_attn.k_proj.weight" => one(names::attn_k(l)),
        "self_attn.v_proj.weight" => one(names::attn_v(l)),
        "self_attn.f_a_proj.weight" => one(names::ssm_f_a(l)),
        "self_attn.f_b_proj.weight" => one(names::ssm_f_b(l)),
        "self_attn.g_a_proj.weight" => one(names::ssm_g_a(l)),
        "self_attn.g_b_proj.weight" => one(names::ssm_g_b(l)),
        "self_attn.b_proj.weight" => one(names::ssm_beta(l)),
        "self_attn.o_norm.weight" => one(names::ssm_norm(l)),
        "self_attn.dt_bias" => one(names::ssm_dt_bias(l)),
        "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight" => {
            let (&[channels, 1, kernel] | &[channels, kernel]) = shape else {
                return Err(refuse(
                    name,
                    format!("conv taps {shape:?} are not [channels, 1, kernel]"),
                ));
            };
            let part = rest.as_bytes()["self_attn.".len()] as char;
            Ok(vec![(
                names::ssm_conv1d(l, part),
                mapped(Layout::Same, vec![kernel, 1, channels]),
            )])
        }
        "self_attn.A_log" => {
            let count = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            if count.is_none_or(|n| n < g.heads) {
                return Err(refuse(
                    name,
                    format!("{shape:?} holds fewer than {} heads", g.heads),
                ));
            }
            Ok(vec![(
                names::ssm_a(l),
                mapped(Layout::NegExp { heads: g.heads }, vec![g.heads]),
            )])
        }
        "self_attn.q_a_proj.weight" => one(names::attn_q_a(l)),
        "self_attn.q_a_layernorm.weight" => one(names::attn_q_a_norm(l)),
        "self_attn.q_b_proj.weight" => one(names::attn_q_b(l)),
        "self_attn.kv_a_proj_with_mqa.weight" => one(names::attn_kv_a_mqa(l)),
        "self_attn.kv_a_layernorm.weight" => one(names::attn_kv_a_norm(l)),
        "self_attn.kv_b_proj.weight" => {
            let (heads, nope, v) = (g.heads, g.nope, g.v);
            let &[rows, rank] = shape else {
                return Err(refuse(name, format!("{shape:?} is not 2-D")));
            };
            if nope.checked_add(v).and_then(|w| w.checked_mul(heads)) != Some(rows) {
                return Err(refuse(
                    name,
                    format!("{rows} rows, not {heads} heads of {nope} + {v}"),
                ));
            }
            Ok(vec![
                (
                    names::attn_k_b(l),
                    mapped(Layout::KeyHalf { heads, nope, v }, vec![nope, rank, heads]),
                ),
                (
                    names::attn_v_b(l),
                    mapped(Layout::ValueHalf { heads, nope, v }, vec![rank, v, heads]),
                ),
            ])
        }
        "self_attn.indexer.wq_b.weight" => one(names::indexer_attn_q_b(l)),
        "self_attn.indexer.wk.weight" => one(names::indexer_attn_k(l)),
        "self_attn.indexer.k_norm.weight" => one(names::indexer_k_norm(l)),
        "self_attn.indexer.k_norm.bias" => one(names::indexer_k_norm_bias(l)),
        "self_attn.indexer.weights_proj.weight" => one(names::indexer_proj(l)),
        "self_attn.indexer.index_kpool_compress_ape" => one(names::indexer_compressor_ape(l)),
        "self_attn.indexer.index_kpool_compress_gate" => one(names::indexer_compressor_gate(l)),
        "mlp.gate.weight" => one(names::ffn_gate_inp(l)),
        "mlp.gate.e_score_correction_bias" => one(names::exp_probs_b(l)),
        "mlp.gate_proj.weight" => one(names::ffn_gate(l)),
        "mlp.up_proj.weight" => one(names::ffn_up(l)),
        "mlp.down_proj.weight" => one(names::ffn_down(l)),
        "mlp.shared_experts.gate_proj.weight" => one(names::ffn_gate_shexp(l)),
        "mlp.shared_experts.up_proj.weight" => one(names::ffn_up_shexp(l)),
        "mlp.shared_experts.down_proj.weight" => one(names::ffn_down_shexp(l)),
        "hc_attn_fn" => one(names::hc_fn(l, Sub::Attn)),
        "hc_attn_base" => one(names::hc_base(l, Sub::Attn)),
        "hc_attn_scale" => one(names::hc_scale(l, Sub::Attn)),
        "hc_ffn_fn" => one(names::hc_fn(l, Sub::Ffn)),
        "hc_ffn_base" => one(names::hc_base(l, Sub::Ffn)),
        "hc_ffn_scale" => one(names::hc_scale(l, Sub::Ffn)),
        _ => Err(refuse(name, "not a GLM-5.3-Flash tensor")),
    }
}

/// Each glm5next tensor, by name, with the format it is held in.
type HeldMap = BTreeMap<String, (Mapped, Format)>;

fn text_name(name: &str) -> &str {
    name.strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name)
}

fn text_layer(name: &str) -> Option<(usize, &str)> {
    let (l, rest) = text_name(name).strip_prefix("layers.")?.split_once('.')?;
    Some((l.parse().ok()?, rest))
}

/// The checkpoint whose tensors are `table` (name, dtype, stored shape) placed by `config` and
/// mapped under geometry `g`: the placement plan, and each glm5next tensor with the format it is
/// held in. A tensor the config refuses, a stack of mixed formats and a table with no text
/// tensors are refused by name.
fn placed_map(
    g: Geometry,
    config: &Config,
    table: &[(String, Dtype, Vec<usize>)],
) -> Result<(quant_config::Plan, HeldMap), Error> {
    let placed = quant_config::plan(
        config,
        table
            .iter()
            .filter(|(n, _, _)| !text_name(n).starts_with("visual."))
            .filter(|(n, _, _)| tp::in_trunk(text_layer(n).map(|(l, _)| l), g.layers))
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        module_class,
    );
    if let Some((name, why)) = placed.refused.iter().next() {
        return Err(refuse(name, why.clone()));
    }
    let shapes: BTreeMap<&str, &[usize]> = table
        .iter()
        .map(|(n, _, s)| (n.as_str(), s.as_slice()))
        .collect();
    let logical: Vec<(&str, Vec<usize>)> = placed
        .placed
        .iter()
        .map(|(n, p)| (n.as_str(), p.format.logical(shapes[n.as_str()])))
        .collect();
    let mapped = map(g, logical.iter().map(|(n, s)| (*n, s.as_slice())))?;
    if mapped.is_empty() {
        return Err(refuse("checkpoint", "no GLM-5.3-Flash text tensors"));
    }
    let format = |name: &str| placed.placed[name].format;
    let held = mapped
        .into_iter()
        .map(|(name, m)| {
            let f = match &m.layout {
                // `-exp(A_log)` is computed into f32.
                Layout::NegExp { .. } => Format::F32,
                Layout::Stack(sources) => {
                    let f = format(&m.source);
                    if let Some(other) = sources.iter().find(|s| format(s) != f) {
                        return Err(refuse(
                            &name,
                            format!("{other} is not {f} as {} is", m.source),
                        ));
                    }
                    f
                }
                _ => format(&m.source),
            };
            Ok((name, (m, f)))
        })
        .collect::<Result<_, Error>>()?;
    Ok((placed, held))
}

/// The tensor-parallel tensors of the checkpoint whose tensors are `table` (name, dtype, stored
/// shape) under `config` and geometry `g`, as the plan takes them: each placed by its format,
/// mapped to its glm5next name and dims, held and cut ([`mapped_tp`]). A refusal of
/// [`placed_map`] or of [`mapped_tp`] is refused by name.
#[cfg(test)]
fn tp_tensors(
    g: Geometry,
    config: &Config,
    table: &[(String, Dtype, Vec<usize>)],
) -> Result<Vec<TpTensor>, Error> {
    let (_, mapped) = placed_map(g, config, table)?;
    mapped_tp(g, &mapped)
}

/// The layer of the glm5next name `name` the map writes, `None` for a tensor of no layer.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let l = rest.split('.').next().unwrap_or_default();
    Some(l.parse().expect("mapped layer index"))
}

/// The tensors of `mapped` a TP rank of trunk geometry `g` holds on the card ([`tp::card_holds`]),
/// each in its format's held form, cut by the family's rule. A format with no held or resident
/// card form and a cut the rule refuses are refused by name.
fn mapped_tp(g: Geometry, mapped: &HeldMap) -> Result<Vec<TpTensor>, Error> {
    mapped
        .iter()
        .filter(|(name, _)| tp::card_holds(name, layer_of(name), g.layers))
        .map(|(name, (m, f))| {
            let dims: Vec<u64> = m.dims.iter().map(|&d| d as u64).collect();
            let (h, ty) = f
                .held(&dims)
                .zip(f.card_type())
                .ok_or_else(|| refuse(name, format!("no held form for {f} {dims:?}")))?;
            let split = tp::cut_of(name, ty, &dims, g.heads as u64)
                .map_err(|why| refuse(name, why.to_string()))?;
            let bytes = dims
                .split_first()
                .and_then(|(&k, rest)| {
                    let rows = rest.iter().try_fold(1u64, |a, &b| a.checked_mul(b))?;
                    model::placement::file_card_format(name, ty, k)?.resident_bytes(ty, k, rows)
                })
                .ok_or_else(|| refuse(name, format!("no resident card form for {ty} {dims:?}")))?;
            Ok(TpTensor {
                cut: split.map(|(c, _)| c),
                block: split.map_or(h.block, |(_, b)| b),
                name: name.clone(),
                dims,
                bytes,
                whole_bytes: h.whole_bytes,
            })
        })
        .collect()
}

/// How a load reads the glm5next tensor `m` held in format `f` ([`part::Made`]): its sources, the
/// role's read (BF16 as stored, an F32 or `-exp` source as F32, a quantized one in its card
/// type) and, for a layout that moves values, that move. A layout that moves single values of a
/// quantized source is refused by name.
fn made(name: &str, m: Mapped, f: Format) -> Result<Made, Error> {
    let moves = matches!(
        m.layout,
        Layout::KeyHalf { .. } | Layout::ValueHalf { .. } | Layout::NegExp { .. }
    );
    if moves && !matches!(f, Format::Bf16 | Format::F32) {
        return Err(refuse(
            name,
            format!("{:?} moves single values; {f} holds blocks", m.layout),
        ));
    }
    let role = Use::Matrix(if f == Format::Bf16 {
        Plain::Bf16
    } else {
        Plain::F32
    });
    let sources = match &m.layout {
        Layout::Stack(sources) => sources.clone(),
        _ => vec![m.source.clone()],
    };
    let step = moves.then(|| {
        let (layout, name) = (m.layout.clone(), name.to_string());
        Box::new(move |n: &mut Normal| layout.apply(&name, n).map_err(|e| e.to_string()))
            as part::Step
    });
    Ok(Made {
        sources,
        role,
        dims: m.dims.iter().map(|&d| d as u64).collect(),
        step,
    })
}

/// The required mapped names of a whole checkpoint, including its configured next-token layers.
fn required(config: &Value, g: Geometry, tensors: usize) -> Result<BTreeSet<String>, Error> {
    use model::arch::glm5next::roles;
    let c = config.get("text_config").unwrap_or(config);
    let count = |key: &str| {
        c.get(key).map_or(Ok(0), |v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::Config(format!("{key} is not a count")))
        })
    };
    let nextn = count("num_nextn_predict_layers")?;
    let shared = count("n_shared_experts")? > 0;
    let layers = g
        .layers
        .checked_add(nextn)
        .filter(|&n| n <= tensors)
        .ok_or_else(|| {
            Error::Config(
                "not enough tensors for the configured trunk and next-token layers".into(),
            )
        })?;
    let kinds = |key: &str| {
        c.get(key)
            .and_then(Value::as_array)
            .filter(|a| a.len() == g.layers)
            .ok_or_else(|| {
                Error::Config(format!("{key} must name each of {} trunk layers", g.layers))
            })
    };
    let attention = kinds("layer_types")?;
    let ffn = kinds("mlp_layer_types")?;
    let mut required = BTreeSet::from([names::token_embd(), names::output(), names::output_norm()]);
    for l in 0..layers {
        let trunk = l < g.layers;
        let mixer = if !trunk {
            roles::LATENT
        } else {
            match attention[l].as_str() {
                Some("linear_attention") => roles::KDA,
                Some("deepseek_sparse_attention") => roles::LATENT,
                _ => {
                    return Err(Error::Config(format!(
                        "layer_types[{l}]: unsupported {}",
                        attention[l]
                    )));
                }
            }
        };
        let routed = if !trunk {
            true
        } else {
            match ffn[l].as_str() {
                Some("dense") => false,
                Some("sparse") => true,
                _ => {
                    return Err(Error::Config(format!(
                        "mlp_layer_types[{l}]: unsupported {}",
                        ffn[l]
                    )));
                }
            }
        };
        for stems in [
            mixer,
            if routed { roles::MOE } else { roles::DENSE },
            if trunk { roles::HC } else { &[] },
            if shared && routed { roles::SHARED } else { &[] },
            if trunk { &[] } else { roles::NEXTN },
        ] {
            required.extend(roles::required(stems).map(|s| format!("blk.{l}.{s}")));
        }
    }
    Ok(required)
}

/// The model folder `dir` read whole: its config's trunk geometry, its quantization config
/// (`config.json`, with `hf_quant_config.json` when present), the folder, every shard's header
/// table, and the required tensors of every configured text layer, NextN's included, under the
/// geometry of the trunk layers.
struct Read {
    trunk: Geometry,
    required: BTreeSet<String>,
    config: Config,
    folder: Folder,
    table: Vec<(String, Dtype, Vec<usize>)>,
}

fn read(dir: &Path) -> Result<Read, Error> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
    let trunk = Geometry::of(&json)?;
    let config = Config::read(dir)?;
    let folder = Folder::open(dir)?;
    let table = folder
        .names()
        .map(|n| {
            let raw = folder.raw(n)?;
            Ok((n.to_string(), raw.dtype, raw.shape.to_vec()))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let required = required(&json, trunk, table.len())?;
    Ok(Read {
        trunk,
        required,
        config,
        folder,
        table,
    })
}

impl Read {
    /// The first required tensor of `g` that `present` lacks, refused by name.
    fn check_required<'a>(
        &self,
        g: Geometry,
        present: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), Error> {
        let present: BTreeSet<&str> = present.into_iter().collect();
        match self
            .required
            .iter()
            .find(|n| tp::in_trunk(layer_of(n), g.layers) && !present.contains(n.as_str()))
        {
            Some(name) => Err(refuse(name, "checkpoint is missing a required text tensor")),
            None => Ok(()),
        }
    }
}

/// The model folder `dir` as a load reads it ([`Checkpoint`]), with its trunk geometry,
/// header-only placement tensors and checkpoint parts ([`made`]). Every configured text layer
/// and required tensor of the target must be present. The checkpoint parts cover the trunk,
/// embedding, head and final norm ([`tp::in_trunk`]); NextN is neither planned nor made. The card
/// plan excludes the host-read embedding ([`mapped_tp`]).
pub fn open(dir: &Path) -> Result<(Geometry, Vec<TpTensor>, Checkpoint), Error> {
    let r = read(dir)?;
    let (plan, mapped) = placed_map(r.trunk, &r.config, &r.table)?;
    r.check_required(r.trunk, mapped.keys().map(String::as_str))?;
    let tensors = mapped_tp(r.trunk, &mapped)?;
    let made = mapped
        .into_iter()
        .map(|(name, (m, f))| Ok((name.clone(), made(&name, m, f)?)))
        .collect::<Result<_, Error>>()?;
    Ok((r.trunk, tensors, Checkpoint::new(r.folder, plan, made)?))
}

#[cfg(test)]
#[path = "../../tests/common/glm_hf.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::test_support::temp_dir;
    use model::part::{Source, Take};
    use model::placement::tp::{RankPool, TpError, plan_tp};

    /// Layers 0 (KDA, dense) and 3 (latent attention, 288 experts) and the unlayered tensors of
    /// nvidia/GLM-5.3-Flash-NVFP4 map to exactly the names and ggml dims of unsloth's
    /// GLM-5.3-Flash UD-Q4_K_XL for the same layers, and a routed stack's expert `e` is HF's.
    #[test]
    fn layers_0_and_3_map_to_the_unsloth_gguf_names_and_dims() {
        let hf = concat!(env!("CARGO_MANIFEST_DIR"), "/../hf/tests/fixtures");
        let config = Config::parse(
            &gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__config.json.gz"
            )),
            Some(&gunzip(&format!(
                "{hf}/quant-configs/nvidia__GLM-5.3-Flash-NVFP4__hf_quant_config.json.gz"
            ))),
        )
        .unwrap();
        let table = table(&gunzip(&format!(
            "{hf}/checkpoint-headers/glm-5.3-flash-nvfp4-layers-0-3.tensors.json.gz"
        )));
        let tensors = logical(&config, &table);
        let got = map(GLM, tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice()))).unwrap();
        let Layout::Stack(stack) = &got["blk.3.ffn_down_exps.weight"].layout else {
            panic!("blk.3.ffn_down_exps.weight is no stack");
        };
        assert_eq!(stack.len(), GLM.experts);
        for (e, source) in stack.iter().enumerate() {
            assert!(
                source.ends_with(&format!("layers.3.mlp.experts.{e}.down_proj.weight")),
                "expert {e}: {source}"
            );
        }
        let got: BTreeMap<String, Vec<usize>> = got.into_iter().map(|(n, m)| (n, m.dims)).collect();
        let want = gguf_table(
            include_str!("../../../hf/tests/fixtures/glm-5.3-flash-ud-q4_k_xl-layers-0-3.tsv"),
            |_| true,
        );
        assert_eq!(got, want);
    }

    /// kv_b_proj splits per head as llama.cpp's converter does (`view(heads, nope + v, rank)`,
    /// split, key half transposed); `A_log` becomes `-exp` of its first `heads` values; a move of
    /// blocks or of other values than it reads, a missing expert, an unknown name and a zero
    /// count in the config are refused by name.
    #[test]
    fn layouts_move_elements_as_the_converter_does() {
        // Two heads of nope 2 and v 1 over rank 3: element (row, col) holds 10 * row + col.
        let (heads, nope, v, rank) = (2, 2, 1, 3);
        let src: Vec<u8> = (0..heads * (nope + v))
            .flat_map(|r| (0..rank).map(move |c| (10 * r + c) as u8))
            .collect();
        let shape = [heads * (nope + v), rank];
        let kv = layer(
            Geometry {
                heads,
                nope,
                v,
                ..GLM
            },
            0,
            "self_attn.kv_b_proj.weight",
            "kv_b_proj",
            &shape,
        )
        .unwrap();
        let moved = |layout: &Layout, ty: GgmlType, bytes: Vec<u8>, dims: Vec<u64>| {
            let mut n = Normal {
                type_id: ty.as_u32(),
                dims,
                bytes,
                scale: vec![],
            };
            layout.apply("t", &mut n).unwrap();
            n
        };
        let wide: Vec<u8> = src
            .iter()
            .flat_map(|&b| u16::from(b).to_le_bytes())
            .collect();
        let dims = vec![rank as u64, (heads * (nope + v)) as u64];
        let k = moved(&kv[0].1.layout, GgmlType::BF16, wide.clone(), dims.clone());
        let narrow = |n: &Normal| n.bytes.iter().step_by(2).copied().collect::<Vec<u8>>();
        assert_eq!(narrow(&k), [0, 10, 1, 11, 2, 12, 30, 40, 31, 41, 32, 42]);
        assert_eq!(
            (k.dims, kv[0].1.dims.clone()),
            (vec![2, 3, 2], vec![2, 3, 2])
        );
        let v_half = moved(&kv[1].1.layout, GgmlType::BF16, wide, dims);
        assert_eq!(narrow(&v_half), [20, 21, 22, 50, 51, 52]);
        assert_eq!(v_half.dims, [3, 1, 2]);
        let a: Vec<u8> = [0.0f32, 1.0, 9.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let neg = moved(
            &Layout::NegExp { heads: 2 },
            GgmlType::F32,
            a.clone(),
            vec![3],
        );
        let neg: Vec<f32> = neg
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        assert_eq!(neg, [-1.0, -1.0f32.exp()]);
        for x in [
            100.0f32,
            -1000.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let mut n = Normal {
                type_id: GgmlType::F32.as_u32(),
                dims: vec![1],
                bytes: x.to_le_bytes().to_vec(),
                scale: vec![],
            };
            let e = Layout::NegExp { heads: 1 }
                .apply("blk.0.ssm_a", &mut n)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("blk.0.ssm_a") && e.contains("-exp at 0"),
                "{x}: {e}"
            );
            assert_eq!(n.bytes, x.to_le_bytes(), "a refusal leaves {x} unchanged");
        }
        for (layout, ty, bytes) in [
            (kv[0].1.layout.clone(), GgmlType::Q8_0, vec![0; 34]),
            (kv[0].1.layout.clone(), GgmlType::BF16, vec![0; 6]),
            (Layout::NegExp { heads: 4 }, GgmlType::F32, a.clone()),
            (Layout::NegExp { heads: 2 }, GgmlType::BF16, a),
        ] {
            let mut n = Normal {
                type_id: ty.as_u32(),
                dims: vec![32, 1],
                bytes,
                scale: vec![],
            };
            assert!(layout.apply("t", &mut n).is_err(), "{layout:?} {ty}");
        }

        let g = Geometry {
            layers: 2,
            heads: 2,
            nope: 2,
            v: 1,
            experts: 2,
        };
        let lone = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            &[4usize, 8][..],
        )];
        let e = map(g, lone).unwrap_err().to_string();
        assert!(
            e.contains("experts.0.up_proj.weight: the stack's 1 expert ids are not 0..2"),
            "{e}"
        );
        let twice = [
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                &[4usize, 8][..],
            ),
            ("model.layers.0.mlp.experts.1.up_proj.weight", &[4, 8]),
            (
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
                &[4, 8],
            ),
        ];
        let e = map(g, twice).unwrap_err().to_string();
        assert!(
            e.contains("layer 0 up_proj expert 0 is also supplied by model.layers.0.mlp"),
            "{e}"
        );
        let e = map(
            g,
            [("model.layers.0.self_attn.rotary.inv_freq", &[4usize][..])],
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("rotary.inv_freq: not a GLM-5.3-Flash tensor"),
            "{e}"
        );
        let vectors = [
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                &[4usize][..],
            ),
            ("model.layers.0.mlp.experts.1.gate_proj.weight", &[4]),
        ];
        let e = map(g, vectors).unwrap_err().to_string();
        assert!(e.contains("expert projection [4] is not 2-D"), "{e}");
        let e = map(
            g,
            [
                ("model.layers.0.mlp.gate_proj.weight", &[8usize, 4][..]),
                ("model.layers.0.post_attention_layernorm.weight", &[4]),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("layer 0 has blk.0.ffn_gate.weight but not blk.0.ffn_up.weight"),
            "{e}"
        );
        let wide = Geometry {
            heads: usize::MAX,
            ..g
        };
        let e = map(
            wide,
            [(
                "model.layers.0.self_attn.kv_b_proj.weight",
                &[6usize, 3][..],
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("6 rows, not"), "{e}");
        let skipped = [
            ("model.layers.2.eh_proj.weight", &[4usize, 8][..]),
            ("model.visual.blocks.0.attn.qkv.weight", &[4, 8]),
        ];
        assert!(map(g, skipped).unwrap().is_empty());
        let zero = serde_json::json!({ "num_hidden_layers": 2, "num_attention_heads": 0,
            "qk_nope_head_dim": 2, "v_head_dim": 1, "n_routed_experts": 2 });
        let e = Geometry::of(&zero).unwrap_err().to_string();
        assert!(e.contains("num_attention_heads is not a count"), "{e}");
    }

    type Table = Vec<(String, Dtype, Vec<usize>)>;

    /// GLM-5.3-Flash-FP4-Local's layers 0 (KDA, dense), 3 (latent, routed) and 4 (KDA, routed)
    /// with every unlayered tensor, its config and its geometry.
    fn fp4_local() -> (Geometry, Config, Value, Table) {
        let hf = concat!(env!("CARGO_MANIFEST_DIR"), "/../hf/tests/fixtures");
        let tag = format!("{hf}/quant-configs/nvidia__GLM-5.3-Flash-FP4-Local");
        let config = gunzip(&format!("{tag}__config.json.gz"));
        let quant = Config::parse(
            &config,
            Some(&gunzip(&format!("{tag}__hf_quant_config.json.gz"))),
        )
        .unwrap();
        let table = table(&gunzip(&format!(
            "{hf}/checkpoint-headers/glm-5.3-flash-fp4-local-layers-0-3-4.tensors.json.gz"
        )));
        (Geometry::of(&config).unwrap(), quant, config, table)
    }

    fn pools(n: usize) -> Vec<RankPool> {
        (0..n)
            .map(|i| RankPool {
                node: format!("n{i}"),
                pool_bytes: u64::MAX,
                device: None,
            })
            .collect()
    }

    /// Each rank's resident card bytes at N = 1, 2 and 4 on the three layers, and on the whole
    /// trunk by the layers of each kind the config lists: the host-read embedding excluded,
    /// BF16 expanded to F32, block scales decoded and FP8_B128 held with an f32 scale a 128 values.
    #[test]
    fn fp4_local_ranks_hold_the_header_tables_split_bytes() {
        let (g, quant, config, table) = fp4_local();
        let ts = tp_tensors(g, &quant, &table).unwrap();
        assert!(ts.iter().all(|t| t.name != "token_embd.weight"));
        let bf16 = ts
            .iter()
            .find(|t| t.name == "blk.0.attn_norm.weight")
            .unwrap();
        assert_eq!(bf16.bytes, bf16.dims.iter().product::<u64>() * 4);
        let q_b = ts
            .iter()
            .find(|t| t.name == "blk.3.attn_q_b.weight")
            .unwrap();
        assert_eq!(q_b.bytes, 16_384 * 1536 / 128 * 132);
        // The MXFP8 projections cut along K keep the planes' 128 values a rank, past the 32 of
        // the format's block.
        for name in ["blk.0.ffn_down.weight", "blk.0.attn_output.weight"] {
            let block = ts.iter().find(|t| t.name == name).map(|t| t.block);
            assert_eq!(block, Some([128, 1]), "{name}");
        }
        let mut no_bias = table.clone();
        no_bias.retain(|(n, _, _)| !n.ends_with(".mlp.gate.e_score_correction_bias"));
        assert_eq!(table.len() - no_bias.len(), 2);
        let without_bias = tp_tensors(g, &quant, &no_bias).unwrap();
        assert_eq!(ts.len() - without_bias.len(), 2);
        let bias_bytes: u64 = ts
            .iter()
            .filter(|t| {
                t.name
                    .ends_with(model::arch::glm5next::roles::SELECTION_BIAS)
            })
            .map(|t| t.bytes)
            .sum();
        assert_ne!(bias_bytes, 0);
        let text = &config["text_config"];
        let kind = |l: usize| {
            (
                text["layer_types"][l].clone(),
                text["mlp_layer_types"][l].clone(),
            )
        };
        let mut copies: BTreeMap<Option<usize>, u64> = BTreeMap::from([(None, 1)]);
        for l in [0, 3, 4] {
            let n = (0..g.layers).filter(|&m| kind(m) == kind(l)).count() as u64;
            copies.insert(Some(l), n);
        }
        assert_eq!(copies.values().sum::<u64>(), 1 + g.layers as u64);
        // PIN(2026-10-21): card budgets exclude the host-read embedding's 2,537,553,920 F32 bytes.
        for (n, fixture, trunk) in [
            (1, 9_540_311_176, 181_173_986_296),
            (2, 4_803_123_080, 91_041_026_808),
            (4, 2_434_529_032, 45_974_547_064),
        ] {
            let plan = plan_tp(&pools(n), &ts).unwrap();
            assert!(
                plan.iter().all(|r| r.bytes() == fixture),
                "N = {n}: {plan:?}"
            );
            let no_bias_plan = plan_tp(&pools(n), &without_bias).unwrap();
            assert!(
                no_bias_plan
                    .iter()
                    .all(|r| r.bytes() == fixture - bias_bytes),
                "N = {n}: {no_bias_plan:?}"
            );
            let mut sum = 0;
            for (l, k) in &copies {
                let part: Vec<_> = ts
                    .iter()
                    .filter(|t| layer_of(&t.name) == *l)
                    .cloned()
                    .collect();
                sum += k * plan_tp(&pools(n), &part).unwrap()[0].bytes();
            }
            assert_eq!(sum, trunk, "N = {n}");
        }
    }

    #[test]
    fn nvfp4_dense_cuts_keep_the_sites_activation_unit() {
        let (g, _, _, _) = fp4_local();
        let quant = Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({"quantization": {
                "quant_algo": "NVFP4", "group_size": 16
            }})),
        )
        .unwrap();
        let mut table = vec![(
            "model.layers.0.post_attention_layernorm.weight".into(),
            Dtype::Bf16,
            vec![256],
        )];
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("model.layers.0.mlp.{proj}");
            for (suffix, dtype, dims) in [
                ("weight", Dtype::U8, vec![256, 512]),
                ("weight_scale", Dtype::F8E4M3, vec![256, 64]),
                ("weight_scale_2", Dtype::F32, vec![]),
            ] {
                table.push((format!("{name}.{suffix}"), dtype, dims));
            }
        }
        let ts = tp_tensors(g, &quant, &table).unwrap();
        for n in [1, 2, 4] {
            assert!(plan_tp(&pools(n), &ts).is_ok(), "N = {n}");
        }
        assert_eq!(
            plan_tp(&pools(8), &ts).unwrap_err(),
            TpError::Block {
                name: "blk.0.ffn_down.weight".into(),
                dim: 0,
                extent: 1024,
                ranks: 8,
                block: 256,
            }
        );
    }

    /// A BF16 head 256 values wide is planned in the MXFP8 codes and scales its load makes, cut
    /// by rows at N = 1, 2 and 4.
    #[test]
    fn a_bf16_head_is_planned_in_its_mxfp8_planes() {
        let (g, _, _, _) = fp4_local();
        let quant = Config::parse(&serde_json::json!({}), None).unwrap();
        let table = [("lm_head.weight".to_string(), Dtype::Bf16, vec![64, 256])];
        let ts = tp_tensors(g, &quant, &table).unwrap();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].name, "output.weight");
        // 64 rows of 256 E4M3 codes and 256 / 32 E8M0 scales.
        assert_eq!(ts[0].bytes, 64 * (256 + 8));
        for n in [1, 2, 4] {
            let plan = plan_tp(&pools(n), &ts).unwrap();
            assert!(
                plan.iter().all(|r| r.bytes() == 16_896 / n as u64),
                "N = {n}: {plan:?}"
            );
        }
    }

    /// A three-layer GLM checkpoint of F32 tensors two wide (KDA and dense, latent and routed,
    /// KDA and routed) with one NextN layer, from the FP4-Local table: its config and its tensors
    /// (name, shape).
    fn small_glm() -> (Value, Vec<(String, Vec<usize>)>) {
        let (_, quant, _, raw) = fp4_local();
        let config = serde_json::json!({"text_config": {
            "num_hidden_layers": 3, "num_attention_heads": 2,
            "qk_nope_head_dim": 2, "v_head_dim": 2, "n_routed_experts": 2,
            "num_nextn_predict_layers": 1, "n_shared_experts": 1,
            "layer_types": ["linear_attention", "deepseek_sparse_attention", "linear_attention"],
            "mlp_layer_types": ["dense", "sparse", "sparse"]
        }});
        let mut table = Vec::new();
        for (name, shape) in logical(&quant, &raw) {
            if let Some(e) = name.split(".mlp.experts.").nth(1) {
                let id: usize = e.split('.').next().unwrap().parse().unwrap();
                if id >= 2 {
                    continue;
                }
            }
            let mut shape = vec![2; shape.len()];
            if name.contains("_conv1d.weight") {
                shape[1] = 1;
            }
            if name.ends_with(".kv_b_proj.weight") {
                shape[0] = 8;
            }
            let name = name
                .replace("layers.3.", "layers.1.")
                .replace("layers.4.", "layers.2.");
            if name.contains("layers.1.") && !name.contains(".hc_") {
                table.push((name.replace("layers.1.", "layers.3."), shape.clone()));
            }
            table.push((name, shape));
        }
        for (stem, shape) in [
            ("eh_proj", vec![2, 4]),
            ("enorm", vec![2]),
            ("hnorm", vec![2]),
            ("shared_head.norm", vec![2]),
        ] {
            table.push((
                format!("model.language_model.layers.3.{stem}.weight"),
                shape,
            ));
        }
        (config, table)
    }

    /// Write `config` and the tensors of `table` that no part of `omit` names into `dir` as a
    /// one-shard folder, every value distinct and finite.
    fn write_small(dir: &Path, config: &Value, table: &[(String, Vec<usize>)], omit: &[&str]) {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let kept: Vec<_> = table
            .iter()
            .filter(|(n, _)| !omit.iter().any(|p| n.contains(p)))
            .collect();
        let mut at = 0u32;
        let data: Vec<Vec<u8>> = kept
            .iter()
            .map(|(_, s)| {
                (0..s.iter().product::<usize>())
                    .flat_map(|_| {
                        at += 1;
                        (at as f32 / 64.0).to_le_bytes()
                    })
                    .collect()
            })
            .collect();
        let entries: Vec<_> = kept
            .iter()
            .zip(&data)
            .map(|((n, s), d)| (n.as_str(), "F32", s.as_slice(), d.as_slice()))
            .collect();
        std::fs::write(dir.join("model.safetensors"), safetensors::write(&entries)).unwrap();
        let weights: BTreeMap<_, _> = kept
            .iter()
            .map(|(n, _)| (n.as_str(), "model.safetensors"))
            .collect();
        std::fs::write(
            dir.join(Folder::INDEX),
            serde_json::json!({"weight_map": weights}).to_string(),
        )
        .unwrap();
    }

    /// The small GLM folder, opened as a load reads it, holds the plan's tensors at the plan's
    /// dims and gives each rank the part the GGUF file of the same model gives
    /// under the family's rule (`tp::rank_span`): every trunk tensor, the key and value halves
    /// and `-exp(A_log)` moved as the converter moves them before their cut, NextN left out.
    /// Too many ranks are refused by name before a read; a move of single values over a
    /// quantized source is refused when the folder opens.
    #[test]
    fn a_glm_folder_gives_each_rank_its_gguf_s_part() {
        let (mut config, mut table) = small_glm();
        config["text_config"]["num_attention_heads"] = 4.into();
        for (name, shape) in &mut table {
            for dim in shape.iter_mut().filter(|d| **d == 2) {
                *dim = 4;
            }
            if name.ends_with(".kv_b_proj.weight") {
                shape[0] = 16;
            }
        }
        let dir = temp_dir("tp-rank");
        write_small(&dir, &config, &table, &[]);
        let (g, opened_plan, checkpoint) = open(&dir).unwrap();
        let plan: BTreeMap<String, Vec<u64>> = opened_plan
            .iter()
            .map(|t| (t.name.clone(), t.dims.clone()))
            .collect();
        let mut held: BTreeMap<String, Vec<u64>> = checkpoint
            .made()
            .iter()
            .map(|(n, m)| (n.clone(), m.dims.clone()))
            .collect();
        assert_eq!(held.remove("token_embd.weight"), Some(vec![4, 4]));
        assert!(!plan.contains_key("token_embd.weight"));
        let nextn = |n: &String| layer_of(n) == Some(g.layers);
        assert!(
            !plan.keys().any(nextn),
            "the card plan holds no NextN tensor"
        );
        assert!(
            !held.keys().any(nextn),
            "the checkpoint makes no NextN tensor"
        );
        assert_eq!(held, plan);
        let folder = &checkpoint;
        let file = part::tests::gguf_of(folder, &dir.join("model.gguf"));
        let gguf = &file;
        let rule = |rank, ranks| {
            move |name: &str, ty, dims: &[u64]| {
                if name == "token_embd.weight" {
                    return Ok(Take::Skip);
                }
                let span = tp::rank_span(name, ty, dims, g.heads as u64, rank, ranks)?;
                Ok::<_, model::placement::PlacementError>(
                    span.map_or(Take::Whole, |(d, at)| Take::Part(d, at)),
                )
            }
        };
        fn parts<E: std::fmt::Display>(
            src: &impl Source,
            rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
        ) -> BTreeMap<String, (Vec<u64>, Vec<u8>)> {
            src.plan(rule)
                .unwrap()
                .iter()
                .map(|p| {
                    let part = src.read(p).unwrap();
                    (
                        p.info.name.clone(),
                        (part.info.dims.clone(), part.spans().concat()),
                    )
                })
                .collect()
        }
        let mut cut = 0;
        for ranks in [1, 2, 4] {
            let budget = plan_tp(&pools(ranks), &opened_plan).unwrap();
            for (rank, planned) in budget.iter().enumerate() {
                for loaded in [
                    gguf.plan(rule(rank, ranks)).unwrap(),
                    folder.plan(rule(rank, ranks)).unwrap(),
                ] {
                    let bytes: u64 = loaded
                        .iter()
                        .map(|p| {
                            let t = &p.info;
                            model::placement::file_card_format(&t.name, t.ty, t.dims[0])
                                .unwrap()
                                .resident_bytes(t.ty, t.dims[0], t.dims[1..].iter().product())
                                .unwrap()
                        })
                        .sum();
                    assert_eq!(planned.bytes(), bytes, "rank {rank} of {ranks}");
                }
                let (want, got) = (
                    parts(gguf, rule(rank, ranks)),
                    parts(folder, rule(rank, ranks)),
                );
                assert_eq!(got.len(), plan.len());
                assert_eq!(got, want, "rank {rank} of {ranks}");
                cut += got.iter().filter(|(n, (d, _))| plan[*n] != *d).count();
            }
        }
        assert!(cut > 20, "{cut} parts cut");
        let r = read(&dir).unwrap();
        let (placed, mapped) = placed_map(r.trunk, &r.config, &r.table).unwrap();
        let whole = parts(folder, rule(0, 1));
        let mut moved = 0;
        for (name, m) in checkpoint.made().iter().filter(|(_, m)| m.step.is_some()) {
            let source = [(m.sources[0].as_str(), &placed.placed[&m.sources[0]])];
            let mut want = hf::modelopt::normalize(|n| r.folder.raw(n), &source, m.role).unwrap();
            mapped[name].0.layout.apply(name, &mut want).unwrap();
            assert_eq!(whole[name].1, want.bytes, "{name}");
            moved += 1;
        }
        assert!(moved >= 3, "{moved} moved");
        // PIN(2026-10-21): four-head success replaces the two-head N=4 refusal; N=8 still refuses.
        let e = [
            gguf.plan(rule(0, 8)).unwrap_err().to_string(),
            folder.plan(rule(0, 8)).unwrap_err().to_string(),
        ];
        for e in e {
            assert!(e.contains("into 8 ranks"), "{e}");
        }
        let kv = Mapped {
            source: "kv_b_proj".into(),
            layout: Layout::KeyHalf {
                heads: 2,
                nope: 2,
                v: 2,
            },
            dims: vec![2, 2, 2],
        };
        let e = made("blk.1.attn_k_b.weight", kv, Format::Mxfp8)
            .err()
            .unwrap();
        assert!(e.to_string().contains("MXFP8 holds blocks"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_target_opens_when_only_unused_quantization_is_refused() {
        let (config, table) = small_glm();
        let dir = temp_dir("nextn-refusal");
        for prefix in ["model.language_model.", "model.", ""] {
            let table: Vec<_> = table
                .iter()
                .map(|(n, s)| (n.replace("model.language_model.", prefix), s.clone()))
                .collect();
            for (suffix, dtype, trunk) in [
                ("layers.3.eh_proj.weight", Dtype::U32, false),
                ("layers.3.mlp.experts.0.down_proj.weight", Dtype::U32, false),
                ("layers.1.self_attn.q_b_proj.weight", Dtype::U32, true),
                ("layers.1.mlp.experts.0.down_proj.weight", Dtype::U32, true),
                ("visual.blocks.0.attn.qkv.weight", Dtype::U16, false),
                ("visual.blocks.0.attn.qkv.weight", Dtype::U32, false),
            ] {
                let refused = format!("{prefix}{suffix}");
                let mut table = table.clone();
                if suffix.starts_with("visual.") {
                    table.push((refused.clone(), vec![2, 2]));
                }
                write_small(&dir, &config, &table, &[]);
                let folder = Folder::open(&dir).unwrap();
                let entries: Vec<_> = folder
                    .names()
                    .map(|n| {
                        let r = folder.raw(n).unwrap();
                        let dtype = if n == refused { dtype } else { Dtype::F32 };
                        (
                            n,
                            dtype.name(),
                            r.shape,
                            &r.bytes[..r.shape.iter().product::<usize>() * dtype.size()],
                        )
                    })
                    .collect();
                let bytes = safetensors::write(&entries);
                drop(folder);
                std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
                let r = read(&dir).unwrap();
                let raw = quant_config::plan(
                    &r.config,
                    r.table
                        .iter()
                        .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
                    module_class,
                );
                let e = &raw.refused[&refused];
                assert!(e.contains(dtype.name()), "{e}");
                if trunk {
                    let e = open(&dir).err().unwrap().to_string();
                    assert!(e.contains(&refused) && e.contains(dtype.name()), "{e}");
                    println!("TARGET_REFUSED {e}");
                    continue;
                }
                let (_, tensors, checkpoint) =
                    open(&dir).unwrap_or_else(|e| panic!("{refused}: {e}"));
                assert!(tensors.iter().all(|t| !t.name.starts_with("blk.3.")));
                assert!(checkpoint.made().keys().all(|n| !n.starts_with("blk.3.")));
                let parts = checkpoint
                    .plan(|_, _, _| Ok::<_, String>(Take::Whole))
                    .unwrap();
                assert!(!parts.is_empty());
                for part in &parts {
                    checkpoint.read(part).unwrap();
                }
                println!("TARGET_OK {refused}; UNUSED_REFUSED {e}");
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    // PIN(2026-10-23): NextN completeness is checked under its full geometry, separately from
    // the target-only folder load; the target still requires every trunk tensor.
    #[test]
    fn a_folder_requires_the_whole_configured_text_model() {
        let (config, table) = small_glm();
        let dir = temp_dir("tp-folder");
        let write = |config: &Value, omit: &[&str]| write_small(&dir, config, &table, omit);
        write(&config, &[]);
        let (_, full, _) = open(&dir).unwrap();
        let full_bytes = plan_tp(&pools(2), &full).unwrap()[0].bytes();
        assert!(full.iter().all(|t| !t.name.starts_with("blk.3.")));
        let mut accepted = Vec::new();
        for (what, omit, missing) in [
            ("trunk layer", &["layers.2."][..], "blk.2."),
            (
                "KDA attention",
                &["layers.0.self_attn.", "layers.0.input_layernorm."],
                "blk.0.",
            ),
            (
                "latent attention",
                &["layers.1.self_attn.", "layers.1.input_layernorm."],
                "blk.1.",
            ),
            (
                "dense feed-forward",
                &["layers.0.mlp.", "layers.0.post_attention_layernorm."],
                "blk.0.",
            ),
            (
                "routed feed-forward",
                &["layers.1.mlp.", "layers.1.post_attention_layernorm."],
                "blk.1.",
            ),
            ("hyper-connection", &["layers.0.hc_"], "blk.0.hc_"),
            (
                "trunk shared expert",
                &["layers.1.mlp.shared_experts."],
                "blk.1.ffn_",
            ),
            ("embedding", &["embed_tokens.weight"], "token_embd.weight"),
            ("head", &["lm_head.weight"], "output.weight"),
            (
                "final norm",
                &["language_model.norm.weight"],
                "output_norm.weight",
            ),
            ("next-token layer", &["layers.3."], "blk.3."),
            (
                "next-token projection",
                &["layers.3.eh_proj.weight"],
                "blk.3.nextn.eh_proj.weight",
            ),
            (
                "next-token embedding norm",
                &["layers.3.enorm.weight"],
                "blk.3.nextn.enorm.weight",
            ),
            (
                "next-token hidden norm",
                &["layers.3.hnorm.weight"],
                "blk.3.nextn.hnorm.weight",
            ),
            (
                "next-token attention",
                &["layers.3.self_attn.", "layers.3.input_layernorm."],
                "blk.3.",
            ),
            (
                "next-token feed-forward",
                &["layers.3.mlp.", "layers.3.post_attention_layernorm."],
                "blk.3.",
            ),
            (
                "next-token shared expert",
                &["layers.3.mlp.shared_experts."],
                "blk.3.ffn_",
            ),
        ] {
            write(&config, omit);
            let checked = if missing.starts_with("blk.3.") {
                let (_, tensors, _) = open(&dir).unwrap();
                assert_eq!(plan_tp(&pools(2), &tensors).unwrap()[0].bytes(), full_bytes);
                let r = read(&dir).unwrap();
                let all = Geometry {
                    layers: 4,
                    ..r.trunk
                };
                placed_map(all, &r.config, &r.table).and_then(|(_, mapped)| {
                    r.check_required(all, mapped.keys().map(String::as_str))
                })
            } else {
                open(&dir).map(|_| ())
            };
            match checked {
                Ok(()) => {
                    println!("ACCEPTED {what}");
                    accepted.push(what);
                }
                Err(e) => {
                    let e = e.to_string();
                    assert!(e.contains(missing), "{what}: {e}");
                    println!("REFUSED {what}: {e}");
                }
            }
        }
        for (key, value) in [
            ("layer_types", serde_json::json!([])),
            ("mlp_layer_types", serde_json::json!(["dense"])),
            (
                "layer_types",
                serde_json::json!(["unknown", "deepseek_sparse_attention", "linear_attention"]),
            ),
            (
                "mlp_layer_types",
                serde_json::json!(["unknown", "sparse", "sparse"]),
            ),
            ("num_nextn_predict_layers", serde_json::json!(-1)),
            ("n_shared_experts", serde_json::json!(false)),
            ("num_nextn_predict_layers", serde_json::json!(u64::MAX)),
        ] {
            let mut bad = config.clone();
            bad["text_config"][key] = value;
            write(&bad, &[]);
            match open(&dir) {
                Ok((_, tensors, _)) => {
                    println!(
                        "ACCEPTED {key}: PLAN_OK bytes={}",
                        plan_tp(&pools(2), &tensors).unwrap()[0].bytes()
                    );
                    accepted.push(key);
                }
                Err(e) => {
                    assert!(matches!(e, Error::Config(_)), "{e}");
                    println!("REFUSED {key}: {e}");
                }
            }
        }
        // Bias and the NextN head norm are optional; a zero NextN count needs no NextN tensors.
        write(
            &config,
            &["e_score_correction_bias", "shared_head.norm.weight"],
        );
        open(&dir).unwrap();
        let mut trunk_only = config;
        trunk_only["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
        write(&trunk_only, &["layers.3."]);
        assert_eq!(
            plan_tp(&pools(2), &open(&dir).unwrap().1).unwrap()[0].bytes(),
            full_bytes
        );
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            accepted.is_empty(),
            "incomplete folders accepted: {accepted:?}"
        );
    }

    // PIN(2026-10-20): was cuts_off_a_head_or_a_block_and_unknown_stems_are_refused; its per-tensor
    // refusals (a stem with no rule, a split tensor of other dims) moved with the rule to
    // model::arch::glm5next::tp::cut, whose test holds them.
    /// A table with no text tensors, 64 heads into 3 ranks, the MXFP8 down's 12,288 inputs into 64
    /// ranks (192 a rank, off the planes' 128) and a routed down's 2,048 (32 a rank, off the
    /// 64-value NVFP4 block) are refused by name.
    #[test]
    fn a_table_without_text_and_cuts_off_a_head_or_a_block_are_refused() {
        let (g, quant, _, table) = fp4_local();
        let visual = vec![(
            "model.visual.blocks.0.attn.qkv.weight".to_string(),
            Dtype::Bf16,
            vec![8, 4],
        )];
        for empty in [vec![], visual] {
            assert!(matches!(
                tp_tensors(g, &quant, &empty),
                Err(Error::Tensor { what, .. }) if what == "no GLM-5.3-Flash text tensors"
            ));
        }
        for (omit, missing) in [
            ("0.self_attn.o_proj.", "blk.0.attn_output.weight"),
            ("3.self_attn.o_proj.", "blk.3.attn_output.weight"),
            ("0.input_layernorm.", "blk.0.attn_norm.weight"),
            ("3.input_layernorm.", "blk.3.attn_norm.weight"),
            ("0.post_attention_layernorm.", "blk.0.ffn_norm.weight"),
            ("3.post_attention_layernorm.", "blk.3.ffn_norm.weight"),
            ("3.self_attn.indexer.", "blk.3.indexer.attn_k.weight"),
            ("0.hc_ffn_", "blk.0.hc_ffn_fn.weight"),
        ] {
            let prefix = format!("model.language_model.layers.{omit}");
            let incomplete: Vec<_> = table
                .iter()
                .filter(|(n, _, _)| !n.starts_with(&prefix))
                .cloned()
                .collect();
            assert!(incomplete.len() < table.len(), "{prefix}");
            let e = tp_tensors(g, &quant, &incomplete).unwrap_err().to_string();
            assert!(e.contains(&format!("but not {missing}")), "{prefix}: {e}");
        }
        let ts = tp_tensors(g, &quant, &table).unwrap();
        let e = plan_tp(&pools(3), &ts).unwrap_err();
        assert!(matches!(&e, TpError::Unit { ranks: 3, .. }), "{e}");
        // At 64 ranks the MXFP8 down's 192 values a rank are off its planes' 128, and the NVFP4
        // experts' 32 off their 64-value block.
        let e = plan_tp(&pools(64), &ts).unwrap_err();
        assert_eq!(
            e,
            TpError::Block {
                name: "blk.0.ffn_down.weight".into(),
                dim: 0,
                extent: 12288,
                ranks: 64,
                block: 128
            }
        );
        let exps: Vec<_> = ts
            .iter()
            .filter(|t| t.name == "blk.3.ffn_down_exps.weight")
            .cloned()
            .collect();
        assert_eq!(
            plan_tp(&pools(64), &exps).unwrap_err(),
            TpError::Block {
                name: "blk.3.ffn_down_exps.weight".into(),
                dim: 0,
                extent: 2048,
                ranks: 64,
                block: 64
            }
        );
        let q = ts
            .iter()
            .find(|t| t.name == "blk.3.attn_q_b.weight")
            .unwrap();
        assert_eq!(
            (q.dims.as_slice(), q.cut.unwrap().unit, q.block),
            (&[1536, 16384][..], 256, [128, 1])
        );
    }
}
