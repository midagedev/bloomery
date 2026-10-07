//! The joint schema head inside a model file: llama.cpp's Clef layout (`general.architecture =
//! clef`), the Qwen3.5 trunk's tensors beside the head's, as `conversion/clef.py` writes them.
//!
//! The shape is the file's: `<arch>.decision.{type,routing_block_count,block_count,head_count}`,
//! the head's LayerNorm epsilon, the hidden width from `decision.proj_memory`, the head width
//! from it and the feed-forward width from `dec.blk.0.ffn_up`. The tensors are the release's
//! `state_dict` entries renamed by `clef.py`: the routing blocks first (`evidence_layers.i` is
//! `dec.blk.i`), then the decoder layers (`layers.j` is `dec.blk.{routing + j}`), each
//! `nn.MultiheadAttention`'s `in_proj_*` split into `q`, `k`, `v`, the three scalars stored as
//! the values the logits use (`decision.scales`: [`crate::head::scales_of`]), and ggml's
//! `ne[]` order, which reverses the torch shape. Each tensor is whatever type the file's quant
//! gave it; this reader converts every type `gguf::quant` dequantizes. A tensor in the head's
//! namespace (`dec.*`, `decision.*`, `token_types.weight`) that the head does not read is
//! refused by name.

use std::collections::HashSet;

use gguf::{Split, TensorInfo, quant};

use crate::Error;
use crate::head::{HeadConfig, LAYER_NORM_EPS, SCALARS, Scales, Weights};

/// What `<arch>.decision.type` names the head this module reads.
pub const DECISION_TYPE: &str = "clef";

/// The scalars' tensor: `[prior, joint, gate]`.
const SCALES: &str = "decision.scales";

/// Whether `name` is in the namespace llama.cpp gives the head's tensors.
fn in_head_namespace(name: &str) -> bool {
    name.starts_with("dec.") || name.starts_with("decision.") || name == "token_types.weight"
}

/// The file's tensors of one `state_dict` name, in the order they are joined: one, or `q`, `k`,
/// `v` for an `in_proj_*`; `None` for a name the layout does not carry. `routing` is the number
/// of routing blocks, which the decoder layers' blocks follow.
fn sources(name: &str, routing: usize) -> Option<Vec<String>> {
    let parts: Vec<&str> = name.split('.').collect();
    let one = |stem: String, p: &str| Some(vec![format!("{stem}.{p}")]);
    let qkv = |stem: &str, p: &str| {
        Some(
            ["q", "k", "v"]
                .iter()
                .map(|x| format!("{stem}_{x}.{p}"))
                .collect(),
        )
    };
    match parts.as_slice() {
        ["hidden_norm", p] => one("decision.hidden_norm".into(), p),
        [proj, "weight"] if proj.ends_with("_projection") => {
            let short = proj.strip_suffix("_projection")?;
            one(format!("decision.proj_{short}"), "weight")
        }
        ["type_embedding", "weight"] => Some(vec!["token_types.weight".to_string()]),
        ["option_summary_norm" | "field_norm" | "option_norm", p] => {
            one(format!("decision.{}", parts[0]), p)
        }
        ["residual_scorer", "0", p] => one("decision.scorer".into(), p),
        ["residual_scorer", "3", p] => one("decision.scorer_out".into(), p),
        ["evidence_layers", i, rest @ ..] => {
            let b = format!("dec.blk.{}", i.parse::<usize>().ok()?);
            match rest {
                ["query_norm", p] => one(format!("{b}.cross_attn_norm"), p),
                ["memory_norm", p] => one(format!("{b}.cross_attn_norm_kv"), p),
                ["attention", "in_proj_weight"] => qkv(&format!("{b}.cross_attn"), "weight"),
                ["attention", "in_proj_bias"] => qkv(&format!("{b}.cross_attn"), "bias"),
                ["attention", "out_proj", p] => one(format!("{b}.cross_attn_o"), p),
                ["feedforward_norm", p] => one(format!("{b}.ffn_norm"), p),
                ["feedforward", "0", p] => one(format!("{b}.ffn_up"), p),
                ["feedforward", "3", p] => one(format!("{b}.ffn_down"), p),
                _ => None,
            }
        }
        ["layers", j, rest @ ..] => {
            let b = format!("dec.blk.{}", routing + j.parse::<usize>().ok()?);
            match rest {
                ["self_attn", "in_proj_weight"] => qkv(&format!("{b}.attn"), "weight"),
                ["self_attn", "in_proj_bias"] => qkv(&format!("{b}.attn"), "bias"),
                ["self_attn", "out_proj", p] => one(format!("{b}.attn_o"), p),
                ["multihead_attn", "in_proj_weight"] => qkv(&format!("{b}.cross_attn"), "weight"),
                ["multihead_attn", "in_proj_bias"] => qkv(&format!("{b}.cross_attn"), "bias"),
                ["multihead_attn", "out_proj", p] => one(format!("{b}.cross_attn_o"), p),
                ["linear1", p] => one(format!("{b}.ffn_up"), p),
                ["linear2", p] => one(format!("{b}.ffn_down"), p),
                ["norm1", p] => one(format!("{b}.attn_norm"), p),
                ["norm2", p] => one(format!("{b}.cross_attn_norm"), p),
                ["norm3", p] => one(format!("{b}.ffn_norm"), p),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The head of a model file in llama.cpp's Clef layout, as a [`Weights`] source.
pub struct GgufHead<'a> {
    split: &'a Split,
    cfg: HeadConfig,
}

impl<'a> GgufHead<'a> {
    /// The head `split` carries: its shape from the keys and tensors named in the module header,
    /// a file with no head, another decision type, another LayerNorm epsilon or widths that
    /// disagree refused by name.
    pub fn open(split: &'a Split) -> Result<GgufHead<'a>, Error> {
        let bad = |what: String| Error::InFileHead(what);
        let key = |suffix: &str| split.arch_key(suffix);
        match split.arch_get_str("decision.type") {
            Some(DECISION_TYPE) => {}
            Some(other) => {
                return Err(bad(format!(
                    "{} is {other:?}, and this reads {DECISION_TYPE:?}",
                    key("decision.type")
                )));
            }
            None => {
                return Err(bad(format!(
                    "{} is absent: the file carries no decision head",
                    key("decision.type")
                )));
            }
        }
        let count = |suffix: &str| -> Result<usize, Error> {
            split
                .arch_get_u64(suffix)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| bad(format!("{} is absent or not a whole number", key(suffix))))
        };
        let eps = split
            .arch_get_f32("attention.layer_norm_epsilon")
            .ok_or_else(|| {
                bad(format!(
                    "{} is absent: the head's LayerNorm epsilon",
                    key("attention.layer_norm_epsilon")
                ))
            })?;
        if eps != LAYER_NORM_EPS {
            return Err(bad(format!(
                "{} is {eps}, and the head's LayerNorm runs at {LAYER_NORM_EPS}",
                key("attention.layer_norm_epsilon")
            )));
        }
        let dims = |name: &str| -> Result<Vec<u64>, Error> {
            split
                .find(name)
                .map(|(_, t)| t.dims.clone())
                .ok_or_else(|| bad(format!("the file holds no tensor {name}")))
        };
        let embedding = count("embedding_length")?;
        let proj = dims("decision.proj_memory.weight")?;
        let up = dims("dec.blk.0.ffn_up.weight")?;
        let (&[hidden, width], &[width_up, feedforward]) = (proj.as_slice(), up.as_slice()) else {
            return Err(bad(format!(
                "decision.proj_memory.weight has dims {proj:?} and dec.blk.0.ffn_up.weight {up:?}, \
                 not two each"
            )));
        };
        let whole = |v: u64| usize::try_from(v).map_err(|_| bad(format!("{v} does not fit usize")));
        let (hidden, width, width_up, feedforward) = (
            whole(hidden)?,
            whole(width)?,
            whole(width_up)?,
            whole(feedforward)?,
        );
        if hidden != embedding {
            return Err(bad(format!(
                "decision.proj_memory.weight reads {hidden} values and {} is {embedding}",
                key("embedding_length")
            )));
        }
        if width_up != width {
            return Err(bad(format!(
                "dec.blk.0.ffn_up.weight reads {width_up} values, and decision.proj_memory.weight \
                 gives the head width {width}"
            )));
        }
        let cfg = HeadConfig {
            hidden_size: hidden,
            width,
            routing_layers: count("decision.routing_block_count")?,
            layers: count("decision.block_count")?,
            heads: count("decision.head_count")?,
            feedforward,
        }
        .checked()?;
        Ok(GgufHead { split, cfg })
    }

    /// The head's shape.
    #[must_use]
    pub fn config(&self) -> HeadConfig {
        self.cfg
    }

    /// The tensor types the file stores `name` in (three for an `in_proj_*`), or `None` for a name
    /// the file does not carry.
    #[must_use]
    pub fn stored_types(&self, name: &str) -> Option<Vec<gguf::GgmlType>> {
        sources(name, self.cfg.routing_layers)?
            .iter()
            .map(|g| self.info(g).map(|t| t.ty))
            .collect()
    }

    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.split.find(name).map(|(_, t)| t)
    }

    /// `name`'s values as f32, whatever type the file stores them in.
    fn values(&self, name: &str) -> Result<Vec<f32>, Error> {
        let (shard, t) = self
            .split
            .find(name)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))?;
        let bad = |what: String| Error::InFileHead(format!("{name}: {what}"));
        let blck =
            t.ty.blck_size()
                .ok_or_else(|| bad(format!("{} has no block table", t.ty)))?;
        if t.dims.first().is_none_or(|d| d % blck != 0) {
            return Err(bad(format!(
                "dims {:?} are not whole {} blocks of {blck}",
                t.dims, t.ty
            )));
        }
        let n = usize::try_from(t.dims.iter().product::<u64>())
            .map_err(|_| bad("the element count does not fit usize".into()))?;
        let data = self
            .split
            .shard(shard)
            .ok_or_else(|| bad("its shard is not open".into()))?
            .data(t)
            .map_err(|e| bad(e.to_string()))?;
        let mut out = vec![0f32; n];
        quant::dequant_row(t.ty, data, &mut out).map_err(|e| bad(e.to_string()))?;
        Ok(out)
    }
}

impl Weights for GgufHead<'_> {
    fn names(&self) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::new();
        for (name, _) in self.cfg.weights() {
            if SCALARS.contains(&name.as_str()) {
                if self.info(SCALES).is_some() {
                    out.push(name);
                }
                continue;
            }
            let Some(src) = sources(&name, self.cfg.routing_layers) else {
                continue;
            };
            if src.iter().all(|g| self.info(g).is_some()) {
                seen.extend(src);
                out.push(name);
            }
        }
        out.extend(
            self.split
                .iter_tensors()
                .map(|(_, t)| &t.name)
                .filter(|n| in_head_namespace(n) && n.as_str() != SCALES && !seen.contains(*n))
                .cloned(),
        );
        out
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        if SCALARS.contains(&name) {
            return self.info(SCALES).map(|_| Vec::new());
        }
        let src = sources(name, self.cfg.routing_layers)?;
        let torch = |t: &TensorInfo| -> Vec<usize> {
            t.dims
                .iter()
                .rev()
                .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
                .collect()
        };
        let infos: Vec<&TensorInfo> = src
            .iter()
            .map(|g| self.info(g))
            .collect::<Option<Vec<_>>>()?;
        let mut shape = torch(infos[0]);
        // `q`, `k` and `v` stack into `in_proj_*`; three that disagree keep `q`'s shape, which is
        // not the stacked one, so the loader refuses them by name.
        if infos.len() == 3 && infos.iter().all(|t| t.dims == infos[0].dims) {
            shape[0] *= 3;
        }
        // llama.cpp declares the scorer's output weight `{w, 1}` and the file's writer drops the
        // trailing 1; torch holds it `[1, w]`.
        if name == "residual_scorer.3.weight" && shape.len() == 1 {
            shape.insert(0, 1);
        }
        Some(shape)
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, Error> {
        let src = sources(name, self.cfg.routing_layers)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))?;
        let mut out = Vec::new();
        for g in &src {
            out.extend(self.values(g)?);
        }
        Ok(out)
    }

    fn scales(&self) -> Result<Scales, Error> {
        let v = self.values(SCALES)?;
        let &[prior, joint, gate] = v.as_slice() else {
            return Err(Error::InFileHead(format!(
                "{SCALES} holds {} values, not the three scalars",
                v.len()
            )));
        };
        Ok((prior, joint, gate))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{Split, Value};

    use super::*;
    use crate::Error;
    use crate::head::tests::{CFG, file_of};
    use crate::head::{ClefHead, scales_of};
    use crate::safetensors::Safetensors;

    /// llama.cpp's name map of the head, as `gguf-py/gguf/tensor_mapping.py` lists it for `clef`
    /// (`{bid}` is the block id); the test's own copy, which [`sources`] must invert.
    const MAP: &[(&str, &str)] = &[
        ("joint_head.hidden_norm", "decision.hidden_norm"),
        ("joint_head.memory_projection", "decision.proj_memory"),
        ("joint_head.question_projection", "decision.proj_question"),
        (
            "joint_head.option_question_projection",
            "decision.proj_option_question",
        ),
        ("joint_head.global_projection", "decision.proj_global"),
        (
            "joint_head.option_context_projection",
            "decision.proj_option_context",
        ),
        (
            "joint_head.option_lexical_projection",
            "decision.proj_option_lexical",
        ),
        ("joint_head.type_embedding", "token_types"),
        (
            "joint_head.option_summary_norm",
            "decision.option_summary_norm",
        ),
        ("joint_head.field_norm", "decision.field_norm"),
        ("joint_head.option_norm", "decision.option_norm"),
        ("joint_head.residual_scorer.0", "decision.scorer"),
        ("joint_head.residual_scorer.3", "decision.scorer_out"),
        ("joint_head.layers.{bid}.norm1", "dec.blk.{bid}.attn_norm"),
        (
            "joint_head.layers.{bid}.norm2",
            "dec.blk.{bid}.cross_attn_norm",
        ),
        ("joint_head.layers.{bid}.norm3", "dec.blk.{bid}.ffn_norm"),
        (
            "joint_head.layers.{bid}.self_attn.q",
            "dec.blk.{bid}.attn_q",
        ),
        (
            "joint_head.layers.{bid}.self_attn.k",
            "dec.blk.{bid}.attn_k",
        ),
        (
            "joint_head.layers.{bid}.self_attn.v",
            "dec.blk.{bid}.attn_v",
        ),
        (
            "joint_head.layers.{bid}.self_attn.out_proj",
            "dec.blk.{bid}.attn_o",
        ),
        (
            "joint_head.layers.{bid}.multihead_attn.q",
            "dec.blk.{bid}.cross_attn_q",
        ),
        (
            "joint_head.layers.{bid}.multihead_attn.k",
            "dec.blk.{bid}.cross_attn_k",
        ),
        (
            "joint_head.layers.{bid}.multihead_attn.v",
            "dec.blk.{bid}.cross_attn_v",
        ),
        (
            "joint_head.layers.{bid}.multihead_attn.out_proj",
            "dec.blk.{bid}.cross_attn_o",
        ),
        ("joint_head.layers.{bid}.linear1", "dec.blk.{bid}.ffn_up"),
        ("joint_head.layers.{bid}.linear2", "dec.blk.{bid}.ffn_down"),
        (
            "joint_head.evidence_layers.{bid}.query_norm",
            "dec.blk.{bid}.cross_attn_norm",
        ),
        (
            "joint_head.evidence_layers.{bid}.memory_norm",
            "dec.blk.{bid}.cross_attn_norm_kv",
        ),
        (
            "joint_head.evidence_layers.{bid}.attention.q",
            "dec.blk.{bid}.cross_attn_q",
        ),
        (
            "joint_head.evidence_layers.{bid}.attention.k",
            "dec.blk.{bid}.cross_attn_k",
        ),
        (
            "joint_head.evidence_layers.{bid}.attention.v",
            "dec.blk.{bid}.cross_attn_v",
        ),
        (
            "joint_head.evidence_layers.{bid}.attention.out_proj",
            "dec.blk.{bid}.cross_attn_o",
        ),
        (
            "joint_head.evidence_layers.{bid}.feedforward_norm",
            "dec.blk.{bid}.ffn_norm",
        ),
        (
            "joint_head.evidence_layers.{bid}.feedforward.0",
            "dec.blk.{bid}.ffn_up",
        ),
        (
            "joint_head.evidence_layers.{bid}.feedforward.3",
            "dec.blk.{bid}.ffn_down",
        ),
    ];

    /// `map_tensor_name`: the GGUF name of `name` (`joint_head.<module>.weight|bias`).
    fn map_tensor_name(name: &str) -> String {
        let (module, suffix) = name.rsplit_once('.').expect("a suffix");
        for (pat, out) in MAP {
            let Some((pre, post)) = pat.split_once("{bid}") else {
                if *pat == module {
                    return format!("{out}.{suffix}");
                }
                continue;
            };
            let bid = module
                .strip_prefix(pre)
                .and_then(|m| m.strip_suffix(post))
                .filter(|m| m.parse::<usize>().is_ok());
            if let Some(bid) = bid {
                return format!("{}.{suffix}", out.replace("{bid}", bid));
            }
        }
        panic!("no mapping for {name}");
    }

    /// What `ClefModel.modify_tensors` (`conversion/clef.py`) makes of the head tensor `name` of
    /// `shape` and `data`: its GGUF tensors by name, each in torch shape. The decoder layers
    /// follow the routing blocks; an `in_proj_*` is chunked in three along dim 0.
    fn convert(
        name: &str,
        shape: &[usize],
        data: &[f32],
        routing: usize,
    ) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut parts: Vec<String> = format!("joint_head.{name}")
            .split('.')
            .map(str::to_string)
            .collect();
        if parts[1] == "layers" {
            parts[2] = (parts[2].parse::<usize>().unwrap() + routing).to_string();
        }
        let name = parts.join(".");
        for suffix in ["weight", "bias"] {
            if let Some(prefix) = name.strip_suffix(&format!("in_proj_{suffix}")) {
                let rows = shape[0] / 3;
                let mut one = shape.to_vec();
                one[0] = rows;
                return ["q", "k", "v"]
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let n = data.len() / 3;
                        (
                            map_tensor_name(&format!("{prefix}{x}.{suffix}")),
                            one.clone(),
                            data[i * n..(i + 1) * n].to_vec(),
                        )
                    })
                    .collect();
            }
        }
        vec![(map_tensor_name(&name), shape.to_vec(), data.to_vec())]
    }

    /// One GGUF tensor as the test writes it.
    struct T {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        bytes: Vec<u8>,
    }

    const F32: u32 = 0;
    const Q8_0: u32 = 8;
    const BF16: u32 = 30;

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// ggml's `quantize_row_q8_0_ref`: per 32 values `d = amax / 127` in f16, `q = round(x / d)`.
    fn q8_0_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for blk in v.chunks(32) {
            let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
            let d = amax / 127.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            out.extend(gguf::quant::f32_to_f16_bits(d).to_le_bytes());
            out.extend(blk.iter().map(|x| (x * id).round() as i8 as u8));
        }
        out
    }

    /// How the test stores one head tensor.
    #[derive(Clone, Copy, PartialEq)]
    enum Store {
        /// Every tensor f32.
        F32,
        /// The file's own mix (`Cloudflare_clef-flash-Q5_K_M.gguf`): matrices Q8_0, the scorer's
        /// output weight bf16, the rest f32.
        Mixed,
    }

    /// An edit of a test file's keys and tensors, made before it is written.
    type Edit<'a> = &'a dyn Fn(&mut Vec<(String, Value)>, &mut Vec<T>);

    /// A file in llama.cpp's Clef layout holding the head of `st` (shape `cfg`): the keys, the head's
    /// tensors by `convert`, the scalars as `scales`, and a stand-in backbone tensor.
    fn clef_file(
        tag: &str,
        cfg: HeadConfig,
        st: &Safetensors,
        store: Store,
        scales: Scales,
        edit: impl FnOnce(&mut Vec<(String, Value)>, &mut Vec<T>),
    ) -> PathBuf {
        let mut kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("clef".into()),
            ),
            (
                "clef.embedding_length".to_string(),
                Value::U32(cfg.hidden_size as u32),
            ),
            (
                "clef.decision.type".to_string(),
                Value::String("clef".into()),
            ),
            (
                "clef.decision.routing_block_count".to_string(),
                Value::U32(cfg.routing_layers as u32),
            ),
            (
                "clef.decision.block_count".to_string(),
                Value::U32(cfg.layers as u32),
            ),
            (
                "clef.decision.head_count".to_string(),
                Value::U32(cfg.heads as u32),
            ),
            (
                "clef.attention.layer_norm_epsilon".to_string(),
                Value::F32(LAYER_NORM_EPS),
            ),
        ];
        let mut ts = vec![T {
            name: "output_norm.weight".into(),
            dims: vec![cfg.hidden_size as u64],
            ty: F32,
            bytes: f32_bytes(&vec![1.0; cfg.hidden_size]),
        }];
        for (name, shape) in cfg.weights() {
            if SCALARS.contains(&name.as_str()) {
                continue;
            }
            let data = st.tensor(&name).unwrap().data;
            for (g, shape, data) in convert(&name, &shape, &data, cfg.routing_layers) {
                // ggml's `ne[]` is the torch shape reversed, and the writer drops a trailing 1.
                let mut dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
                if dims.len() == 2 && dims[1] == 1 {
                    dims.pop();
                }
                let matrix = shape.len() == 2 && shape[1].is_multiple_of(32);
                let (ty, bytes) = match store {
                    Store::Mixed if g == "decision.scorer_out.weight" => (BF16, bf16_bytes(&data)),
                    Store::Mixed if matrix => (Q8_0, q8_0_bytes(&data)),
                    _ => (F32, f32_bytes(&data)),
                };
                ts.push(T {
                    name: g,
                    dims,
                    ty,
                    bytes,
                });
            }
        }
        ts.push(T {
            name: SCALES.into(),
            dims: vec![3],
            ty: F32,
            bytes: f32_bytes(&[scales.0, scales.1, scales.2]),
        });
        edit(&mut kvs, &mut ts);
        let decls = ts
            .iter()
            .map(|t| TensorDecl {
                name: t.name.clone(),
                dims: t.dims.clone(),
                type_id: t.ty,
                nbytes: t.bytes.len() as u64,
            })
            .collect();
        let layout = Layout::new(&kvs, decls).expect("the layout");
        let path = std::env::temp_dir().join(format!(
            "bloomery-decision-{}-{tag}.gguf",
            std::process::id()
        ));
        let mut w = Writer::new(std::fs::File::create(&path).unwrap(), layout).unwrap();
        for t in &ts {
            w.tensor(&t.name, &t.bytes).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// The raw scalars of `st` as the release's file holds them, and as the logits use them.
    fn release_scales(st: &Safetensors) -> Scales {
        let raw = |n: &str| st.tensor(n).unwrap().data[0];
        scales_of(raw(SCALARS[0]), raw(SCALARS[1]), raw(SCALARS[2]))
    }

    fn open(path: &PathBuf) -> Result<ClefHead, Error> {
        let split = Split::open(path).expect("the file opens");
        let head = ClefHead::from_gguf(&split);
        let _ = std::fs::remove_file(path);
        head
    }

    fn logits_of(head: &ClefHead) -> Vec<Vec<f32>> {
        use crate::encode::encode_with;
        use crate::json;
        use crate::request::Request;
        let req = Request::from_json(
            &json::parse(
                r#"{"model":"m","state":"st","questions":{"a":{"type":"choice","criteria":{"x":1,"y":2,"z":3}},"b":{"type":"noul"}}}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let enc = encode_with(
            &mut |t: &str| t.bytes().map(u32::from).collect(),
            &req,
            16384,
        )
        .unwrap();
        let hs = head.config().hidden_size;
        let hidden: Vec<f32> = (0..enc.ids.len() * hs)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut rows = |ids: &[u32]| -> Result<Vec<f32>, String> {
            Ok(ids
                .iter()
                .flat_map(|&id| (0..hs).map(move |j| (id as f32 * 0.01 + j as f32 * 0.1).sin()))
                .collect())
        };
        head.forward(&hidden, &enc, &mut rows).unwrap()
    }

    /// The layout's head is the release's head: the same weights under llama.cpp's names, every
    /// block and chunk in its place, give the same logits bit for bit (f32 tensors, scalars
    /// transformed by the release's own f32 formula).
    #[test]
    fn a_clef_file_scores_as_the_release_head() {
        let st = file_of(CFG, None, false);
        let release = ClefHead::from_safetensors(&st, CFG).unwrap();
        let path = clef_file(
            "equal",
            CFG,
            &st,
            Store::F32,
            release_scales(&st),
            |_, _| {},
        );
        let head = open(&path).expect("the head reads");
        assert_eq!(head.config(), CFG);
        assert_eq!(head.scales(), release.scales());
        assert_eq!(logits_of(&head), logits_of(&release));
    }

    /// The names invert: every `state_dict` name of the head has its GGUF tensors, and a tensor
    /// from another block or chunk is another value.
    #[test]
    fn every_head_name_maps_to_the_tensors_clef_py_writes() {
        let st = file_of(CFG, None, false);
        let path = clef_file(
            "names",
            CFG,
            &st,
            Store::F32,
            release_scales(&st),
            |_, _| {},
        );
        let split = Split::open(&path).expect("the file opens");
        let src = GgufHead::open(&split).expect("the head opens");
        for (name, shape) in CFG.weights() {
            if SCALARS.contains(&name.as_str()) {
                continue;
            }
            assert_eq!(src.shape(&name).as_ref(), Some(&shape), "{name}");
            assert_eq!(
                src.tensor(&name).unwrap(),
                st.tensor(&name).unwrap().data,
                "{name}"
            );
        }
        let mut names = src.names();
        names.sort();
        let mut want: Vec<String> = CFG.weights().into_iter().map(|(n, _)| n).collect();
        want.sort();
        assert_eq!(names, want);
        let _ = std::fs::remove_file(&path);
    }

    /// The scalars `clef.py` computes in f64 are the f32 formula's within two ulps, and the head
    /// reads them as stored.
    #[test]
    fn the_stored_scales_are_the_f32_formula_within_two_ulps() {
        let st = file_of(CFG, None, false);
        let raw = |n: &str| f64::from(st.tensor(n).unwrap().data[0]);
        let (p, j, g) = (raw(SCALARS[0]), raw(SCALARS[1]), raw(SCALARS[2]));
        let cap = 100f64.ln();
        let py = (
            p.min(cap).exp() as f32,
            j.min(cap).exp() as f32,
            (1.0 / (1.0 + (-g).exp())) as f32,
        );
        let ours = release_scales(&st);
        for (a, b) in [(py.0, ours.0), (py.1, ours.1), (py.2, ours.2)] {
            assert!((a - b).abs() <= 2.0 * a.abs() * f32::EPSILON, "{a} vs {b}");
        }
        let path = clef_file("scales", CFG, &st, Store::F32, py, |_, _| {});
        assert_eq!(open(&path).unwrap().scales(), py);
    }

    /// A bigger head, stored as the file stores it (matrices Q8_0, the scorer's output weight bf16):
    /// every tensor within its block's bound of the release's, and the head scores.
    ///
    /// The bound: a Q8_0 block of 32 values has `d = amax / 127` stored as f16, `q = round(x / d)`.
    /// The error is at most `d / 2` from the rounding plus `127 · |d − f16(d)| ≤ 127 · d · 2⁻¹¹`
    /// from the scale: `0.5625 d`.
    #[test]
    fn a_q8_0_head_is_within_its_block_bound_of_the_release() {
        let cfg = HeadConfig {
            hidden_size: 64,
            width: 32,
            routing_layers: 2,
            layers: 1,
            heads: 2,
            feedforward: 64,
        };
        let st = file_of(cfg, None, false);
        let path = clef_file("q8", cfg, &st, Store::Mixed, release_scales(&st), |_, _| {});
        let split = Split::open(&path).expect("the file opens");
        let src = GgufHead::open(&split).expect("the head opens");
        let mut quantized = 0;
        for (name, shape) in cfg.weights() {
            if SCALARS.contains(&name.as_str()) {
                continue;
            }
            let want = st.tensor(&name).unwrap().data;
            let got = src.tensor(&name).unwrap();
            assert_eq!(src.shape(&name).as_ref(), Some(&shape), "{name}");
            if shape.len() == 2 && shape[1].is_multiple_of(32) && name != "residual_scorer.3.weight"
            {
                quantized += 1;
                for (w, g) in want.chunks(32).zip(got.chunks(32)) {
                    let amax = w.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let bound = 0.5625 * amax / 127.0 * (1.0 + 1e-6);
                    for (a, b) in w.iter().zip(g) {
                        assert!((a - b).abs() <= bound, "{name}: {a} vs {b}, bound {bound}");
                    }
                }
            } else {
                assert_eq!(got, want, "{name} is stored exactly");
            }
        }
        assert!(quantized > 10, "{quantized} matrices were quantized");
        let head = open(&path).expect("the head reads");
        assert!(logits_of(&head).iter().flatten().all(|l| l.is_finite()));
    }

    /// A file that is not a Clef head, or whose head is not whole, is refused by name.
    #[test]
    fn a_file_without_its_whole_head_is_refused_by_name() {
        let st = file_of(CFG, None, false);
        let scales = release_scales(&st);
        let refused = |tag: &str, edit: Edit<'_>, want: &str| {
            let path = clef_file(tag, CFG, &st, Store::F32, scales, |k, t| edit(k, t));
            let e = open(&path)
                .err()
                .unwrap_or_else(|| panic!("{tag} was read"));
            assert!(e.to_string().contains(want), "{tag}: {e}");
        };
        refused(
            "notype",
            &|k, _| k.retain(|(n, _)| n != "clef.decision.type"),
            "clef.decision.type is absent",
        );
        refused(
            "othertype",
            &|k, _| {
                for (n, v) in k.iter_mut() {
                    if n == "clef.decision.type" {
                        *v = Value::String("laya".into());
                    }
                }
            },
            "is \"laya\", and this reads \"clef\"",
        );
        refused(
            "eps",
            &|k, _| {
                for (n, v) in k.iter_mut() {
                    if n == "clef.attention.layer_norm_epsilon" {
                        *v = Value::F32(1e-6);
                    }
                }
            },
            "the head's LayerNorm runs at 0.00001",
        );
        refused(
            "width",
            &|k, _| {
                for (n, v) in k.iter_mut() {
                    if n == "clef.embedding_length" {
                        *v = Value::U32(32);
                    }
                }
            },
            "decision.proj_memory.weight reads 16 values and clef.embedding_length is 32",
        );
        refused(
            "heads",
            &|k, _| {
                for (n, v) in k.iter_mut() {
                    if n == "clef.decision.head_count" {
                        *v = Value::U32(3);
                    }
                }
            },
            "width 8 does not split into 3 heads",
        );
        // A tensor missing, one of the head's namespace the head does not read, a block the keys
        // do not count: each by name, with the others' names beside it in the one error.
        let path = clef_file("missing", CFG, &st, Store::F32, scales, |_, t| {
            t.retain(|x| x.name != "dec.blk.1.ffn_down.weight");
            t.push(T {
                name: "dec.blk.9.attn_q.weight".into(),
                dims: vec![1],
                ty: F32,
                bytes: f32_bytes(&[0.0]),
            });
        });
        match open(&path) {
            Err(Error::Weights {
                missing,
                extra,
                shapes,
            }) => {
                assert_eq!(missing, ["evidence_layers.1.feedforward.3.weight"]);
                assert_eq!(extra, ["dec.blk.9.attn_q.weight"]);
                assert!(shapes.is_empty(), "{shapes:?}");
            }
            other => panic!("{:?}", other.err()),
        }
        // q, k and v of other shapes keep q's, which is not the stacked shape.
        let path = clef_file("qkv", CFG, &st, Store::F32, scales, |_, t| {
            for x in t.iter_mut() {
                if x.name == "dec.blk.2.attn_k.weight" {
                    x.dims = vec![8, 4];
                    x.bytes = f32_bytes(&[0.0; 32]);
                }
            }
        });
        match open(&path) {
            Err(Error::Weights { shapes, .. }) => {
                assert_eq!(shapes.len(), 1, "{shapes:?}");
                assert!(
                    shapes[0].starts_with("layers.0.self_attn.in_proj_weight"),
                    "{shapes:?}"
                );
            }
            other => panic!("{:?}", other.err()),
        }
    }

    /// An architecture with no decision keys is no head: the refusal names the key.
    #[test]
    fn a_trunk_alone_has_no_head() {
        let st = file_of(CFG, None, false);
        let path = clef_file(
            "trunk",
            CFG,
            &st,
            Store::F32,
            release_scales(&st),
            |k, t| {
                k.retain(|(n, _)| !n.contains("decision"));
                t.retain(|x| !in_head_namespace(&x.name));
            },
        );
        let e = open(&path).err().expect("no head").to_string();
        assert!(
            e.contains("clef.decision.type is absent") && e.contains("no decision head"),
            "{e}"
        );
    }
}
