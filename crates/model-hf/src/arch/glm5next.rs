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
//! Tensor parallel splits each mapped tensor ([`tp_cut`]) along the dim and by the unit a rank's
//! part holds whole there, or keeps it whole on every rank; the bytes and the blocks a cut keeps
//! are its format's ([`Format::held`]), and the plan is
//! [`plan_tp`](model::placement::tp::plan_tp)'s. KDA and latent attention
//! split by head: their q, k, v, gate and beta rows, the conv channels, the decay and the latent
//! heads' `q_b`, `k_b` and `v_b`, and the output projection along its input. Dense blocks, the
//! shared expert and every routed expert split by inner width: gate and up by row, down along
//! its input. The head splits by vocabulary row. The norms, the low-rank inputs every head shares
//! (`ssm_f_a`, `ssm_g_a`, `attn_q_a`, `attn_kv_a_mqa`), the indexer, the router and its bias, the
//! hyper-connection mixes and the embedding stay whole on every rank.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hf::quant_config::{self, Config, Format};
use hf::safetensors::{Dtype, Folder};
use model::arch::glm5next::names::{self, Sub};
use model::placement::tp::{Cut, TpTensor};
use serde_json::Value;

/// Why the checkpoint was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("glm5next hf: config.json: {0}")]
    Config(String),
    #[error("glm5next hf: {name}: {what}")]
    Tensor { name: String, what: String },
    #[error("glm5next hf: {0}")]
    Folder(#[from] hf::safetensors::Error),
    #[error("glm5next hf: {0}")]
    Quant(#[from] quant_config::Error),
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
        let short = name
            .strip_prefix("model.language_model.")
            .or_else(|| name.strip_prefix("model."))
            .unwrap_or(name);
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
                let (l, rest) = short
                    .strip_prefix("layers.")
                    .and_then(|r| r.split_once('.'))
                    .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                    .ok_or_else(|| refuse(name, "not a GLM-5.3-Flash tensor"))?;
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
        let (sources, dims) = ::hf::safetensors::expert_stack(&by_id, g.experts)?;
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
    let layers: BTreeSet<usize> = out
        .keys()
        .filter_map(|n| n.strip_prefix("blk.")?.split_once('.')?.0.parse().ok())
        .collect();
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
            let (channels, kernel) = match *shape {
                [c, 1, k] | [c, k] => (c, k),
                _ => {
                    return Err(refuse(
                        name,
                        format!("conv taps {shape:?} are not [channels, 1, kernel]"),
                    ));
                }
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

/// Where the glm5next tensor `name` of logical `dims` splits among ranks, `None` for whole on
/// every rank. A head's part along a dim is that dim over `heads`; a stem with no rule, and a
/// split tensor of other dims than its stem has, are refused by name.
pub fn tp_cut(name: &str, dims: &[u64], heads: u64) -> Result<Option<Cut>, Error> {
    let stem = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, stem)| stem);
    // The dims the tensor has, the dim it splits along, and whether by head or by one value.
    let (rank, dim, by_head) = match stem {
        "token_embd.weight"
        | "output_norm.weight"
        | "attn_norm.weight"
        | "ffn_norm.weight"
        | "ssm_f_a.weight"
        | "ssm_g_a.weight"
        | "ssm_norm.weight"
        | "attn_q_a.weight"
        | "attn_q_a_norm.weight"
        | "attn_kv_a_mqa.weight"
        | "attn_kv_a_norm.weight"
        | "ffn_gate_inp.weight"
        | "exp_probs_b.bias" => return Ok(None),
        _ if stem.starts_with("indexer")
            || stem.starts_with("hc_")
            || stem.starts_with("nextn.") =>
        {
            return Ok(None);
        }
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "ssm_f_b.weight"
        | "ssm_g_b.weight" | "ssm_beta.weight" | "attn_q_b.weight" => (2, 1, true),
        "attn_output.weight" => (2, 0, true),
        "ssm_a" | "ssm_dt.bias" => (1, 0, true),
        "ssm_conv1d_q.weight"
        | "ssm_conv1d_k.weight"
        | "ssm_conv1d_v.weight"
        | "attn_k_b.weight"
        | "attn_v_b.weight" => (3, 2, true),
        "ffn_gate.weight"
        | "ffn_up.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "output.weight" => (2, 1, false),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => (3, 1, false),
        "ffn_down.weight" | "ffn_down_shexp.weight" => (2, 0, false),
        "ffn_down_exps.weight" => (3, 0, false),
        _ => return Err(refuse(name, "no tensor-parallel split rule")),
    };
    if dims.len() != rank {
        return Err(refuse(name, format!("{dims:?} is not {rank}-D")));
    }
    let extent = dims[dim];
    if !by_head {
        return Ok(Some(Cut { dim, unit: 1 }));
    }
    if heads == 0 || !extent.is_multiple_of(heads) {
        return Err(refuse(
            name,
            format!("dim {dim} of {extent} is not {heads} heads"),
        ));
    }
    Ok(Some(Cut {
        dim,
        unit: extent / heads,
    }))
}

/// The tensor-parallel tensors of the checkpoint whose tensors are `table` (name, dtype, stored shape)
/// under `config` and geometry `g`, as the plan takes them: each placed by its format, mapped to
/// its glm5next name and dims, held and cut. A tensor the config refuses, a format with no held
/// form, a stack of mixed formats and a table with no text tensors are refused by name.
pub fn tp_tensors(
    g: Geometry,
    config: &Config,
    table: &[(String, Dtype, Vec<usize>)],
) -> Result<Vec<TpTensor>, Error> {
    let placed = quant_config::plan(
        config,
        table.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
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
    mapped
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
            let dims: Vec<u64> = m.dims.iter().map(|&d| d as u64).collect();
            let h = f
                .held(&dims)
                .ok_or_else(|| refuse(&name, format!("no held form for {f} {dims:?}")))?;
            Ok(TpTensor {
                cut: tp_cut(&name, &dims, g.heads as u64)?,
                name,
                dims,
                bytes: h.bytes,
                whole_bytes: h.whole_bytes,
                block: h.block,
            })
        })
        .collect()
}

/// The required mapped names of a whole checkpoint, including its configured next-token layers.
fn required(
    config: &Value,
    g: Geometry,
    tensors: usize,
) -> Result<(Geometry, BTreeSet<String>), Error> {
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
    Ok((Geometry { layers, ..g }, required))
}

/// [`tp_tensors`] of the model folder `dir`: its `config.json` (the geometry and the quantization
/// config, with `hf_quant_config.json` when present) and every shard's header. Every configured
/// text layer and required tensor must be present, including NextN; the returned plan covers
/// the trunk and its embedding, head and final norm. NextN is not loaded by this plan.
pub fn read_tp(dir: &Path) -> Result<Vec<TpTensor>, Error> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| Error::Config(e.to_string()))?;
    let g = Geometry::of(&json)?;
    let config = Config::read(dir)?;
    let folder = Folder::open(dir)?;
    let table = folder
        .names()
        .map(|n| {
            let raw = folder.raw(n)?;
            Ok((n.to_string(), raw.dtype, raw.shape.to_vec()))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let (all, required) = required(&json, g, table.len())?;
    let tensors = tp_tensors(all, &config, &table)?;
    let present: BTreeSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    if let Some(name) = required.iter().find(|n| !present.contains(n.as_str())) {
        return Err(refuse(name, "checkpoint is missing a required text tensor"));
    }
    Ok(tensors
        .into_iter()
        .filter(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|n| n.split_once('.'))
                .is_none_or(|(l, _)| l.parse::<usize>().expect("mapped layer index") < g.layers)
        })
        .collect())
}

#[cfg(test)]
#[path = "../../tests/common/glm_hf.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
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
    /// split, key half transposed); `A_log` becomes `-exp` of its first `heads` values; a missing
    /// expert, an unknown name and a zero count in the config are refused by name.
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
        let k = apply(&kv[0].1.layout, &src, &shape, 1);
        assert_eq!(k, [0, 10, 1, 11, 2, 12, 30, 40, 31, 41, 32, 42]);
        let v_half = apply(&kv[1].1.layout, &src, &shape, 1);
        assert_eq!(v_half, [20, 21, 22, 50, 51, 52]);
        let a: Vec<u8> = [0.0f32, 1.0, 9.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let neg = apply(&Layout::NegExp { heads: 2 }, &a, &[3], 4);
        let neg: Vec<f32> = neg
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        assert_eq!(neg, [-1.0, -1.0f32.exp()]);

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

    /// Each rank's bytes at N = 1, 2 and 4 on the three layers, and on the whole trunk by the
    /// layers of each kind the config lists: the sums an independent count of the checkpoint's
    /// 42 headers gives (91,600,737,016 B a rank at N = 2, 46,684,372,088 at N = 4, the
    /// checkpoint's text weights with block scales decoded and BF16 as shipped).
    #[test]
    fn fp4_local_ranks_hold_the_header_tables_split_bytes() {
        let (g, quant, config, table) = fp4_local();
        let ts = tp_tensors(g, &quant, &table).unwrap();
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
        let layer = |name: &str| {
            name.strip_prefix("blk.")
                .and_then(|r| r.split('.').next()?.parse::<usize>().ok())
        };
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
        for (n, fixture, trunk) in [
            (1, 10_732_417_160, 181_433_466_872),
            (2, 6_019_132_296, 91_600_737_016),
            (4, 3_662_489_864, 46_684_372_088),
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
                    .filter(|t| layer(&t.name) == *l)
                    .cloned()
                    .collect();
                sum += k * plan_tp(&pools(n), &part).unwrap()[0].bytes();
            }
            assert_eq!(sum, trunk, "N = {n}");
        }
    }

    #[test]
    fn a_folder_requires_the_whole_configured_text_model() {
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
        let dir = std::env::temp_dir().join(format!(
            "glm-tp-folder-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let write = |config: &Value, omit: &[&str]| {
            std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
            let kept: Vec<_> = table
                .iter()
                .filter(|(n, _)| !omit.iter().any(|p| n.contains(p)))
                .collect();
            let data: Vec<_> = kept
                .iter()
                .map(|(_, s)| vec![0u8; 4 * s.iter().product::<usize>()])
                .collect();
            let entries: Vec<_> = kept
                .iter()
                .zip(&data)
                .map(|((n, s), d)| (n.as_str(), "F32", s.as_slice(), d.as_slice()))
                .collect();
            std::fs::write(
                dir.join("model.safetensors"),
                hf::safetensors::write(&entries),
            )
            .unwrap();
            let weights: BTreeMap<_, _> = kept
                .iter()
                .map(|(n, _)| (n.as_str(), "model.safetensors"))
                .collect();
            std::fs::write(
                dir.join(Folder::INDEX),
                serde_json::json!({"weight_map": weights}).to_string(),
            )
            .unwrap();
        };
        write(&config, &[]);
        let full = read_tp(&dir).unwrap();
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
            match read_tp(&dir) {
                Ok(tensors) => {
                    let bytes = plan_tp(&pools(2), &tensors).unwrap()[0].bytes();
                    println!("ACCEPTED {what}: PLAN_OK bytes={bytes}, full={full_bytes}");
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
            match read_tp(&dir) {
                Ok(tensors) => {
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
        read_tp(&dir).unwrap();
        let mut trunk_only = config.clone();
        trunk_only["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
        write(&trunk_only, &["layers.3."]);
        assert_eq!(
            plan_tp(&pools(2), &read_tp(&dir).unwrap()).unwrap()[0].bytes(),
            full_bytes
        );
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            accepted.is_empty(),
            "incomplete folders accepted: {accepted:?}"
        );
    }

    /// A table with no text tensors, 64 heads into 3 ranks, a routed down's 2,048 inputs into 64
    /// ranks (32 a rank, off the 64-value NVFP4 block) and a stem with no rule are refused by name.
    #[test]
    fn cuts_off_a_head_or_a_block_and_unknown_stems_are_refused() {
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
        let e = plan_tp(&pools(64), &ts).unwrap_err();
        assert_eq!(
            e,
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
            (&[1536, 16384][..], 256, [128, 128])
        );
        assert!(matches!(
            tp_cut("blk.3.attn_sinks.weight", &[64], 64),
            Err(Error::Tensor { what, .. }) if what == "no tensor-parallel split rule"
        ));
        for (name, dims) in [
            ("blk.3.ffn_up_exps.weight", &[4096u64, 288][..]),
            ("blk.0.ffn_down.weight", &[4096, 2048, 2]),
            ("blk.0.ssm_a", &[64, 1]),
        ] {
            assert!(matches!(
                tp_cut(name, dims, 64),
                Err(Error::Tensor { what, .. }) if what.ends_with("-D")
            ));
        }
    }
}
