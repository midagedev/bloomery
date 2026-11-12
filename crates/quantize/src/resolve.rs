//! Which type each tensor gets: llama-quantize's `llama_tensor_get_type` (`src/llama-quant.cpp`) for the
//! file types `k31-quant.sh` runs and the architectures [`ARCHS`] names, step for step, with its running
//! counters.
//!
//! Only the branches those reach are ported; a file type, an architecture or a result type outside them is refused
//! by name, never typed some other way. The counters (`i_attention_wv`, `i_ffn_down`, …) move only when the
//! standard logic runs for a tensor, so a tensor a rule or `--output-tensor-type` types leaves them where they were,
//! as in ggml's driver; the order the caller passes the tensors in is therefore part of the result.

use crate::rules::Rules;
use crate::source::Source;
use crate::{Error, Result};
use gguf::quant::GgmlType;

/// The architectures the type logic is ported for, each under the lines of `llama-quant.cpp`
/// that show the ported branches are the whole logic for a set of it: none of the branches keyed
/// to an architecture (`tensor_allows_quantization`'s rows at `:337-410`,
/// `llama_tensor_get_type_impl`'s at `:534-573`, FALCON's at `:574,705-740`) applies to a set of
/// the row beyond what the port already carries, and a tensor that reaches a branch the port
/// does not (an `n_expert` below four's head-count ratio under an IQ type, the first eighth of
/// `ffn_down` under one) is refused by name.
pub const ARCHS: &[(&str, &str)] = &[
    (
        "kimi-k3",
        "the KIMI_K31 exclusions (`llama-quant.cpp:375-380`) name the indexer tensors the \
         pipeline drops at prepare (`tools/ref/k31-quant.sh`), and a prepared set holds no tensor \
         the GLM5_NEXT rows (`:382-400`, `:534-551`) name",
    ),
    (
        "exaone-moe",
        "no branch of `llama-quant.cpp` names it, and none needs one: `exp_probs_b.bias` fails \
         the `weight` ending, the `nextn` and `*_norm` tensors the norm rule, the next-token \
         block (`blk.<layers>.`) is a block like any other, and its `nextn.eh_proj` takes the \
         standard logic — the Q8_0 floor of it is glm5-next's alone (`:534-551`)",
    ),
];

/// The file types the logic is ported for (`llama_ftype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ftype {
    Q8_0,
    Q4KM,
    Q5KM,
    Q6K,
    Iq2Xxs,
    Iq1S,
}

impl Ftype {
    /// The type `llama-quantize` names `s` (case-insensitive), when it is one this port covers.
    pub fn parse(s: &str) -> Result<Ftype> {
        Ok(match s.to_ascii_uppercase().as_str() {
            "Q8_0" => Ftype::Q8_0,
            "Q4_K_M" => Ftype::Q4KM,
            "Q5_K_M" => Ftype::Q5KM,
            "Q6_K" => Ftype::Q6K,
            "IQ2_XXS" => Ftype::Iq2Xxs,
            "IQ1_S" => Ftype::Iq1S,
            o => {
                return Err(Error::Usage(format!(
                    "file type {o}: the quantizer covers Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_XXS and IQ1_S; bloomery-quantize --help prints the usage"
                )));
            }
        })
    }

    /// `llama_ftype`'s number: the value `general.file_type` holds.
    #[must_use]
    pub fn id(self) -> u32 {
        match self {
            Ftype::Q8_0 => 7,
            Ftype::Q4KM => 15,
            Ftype::Q5KM => 17,
            Ftype::Q6K => 18,
            Ftype::Iq2Xxs => 19,
            Ftype::Iq1S => 24,
        }
    }

    /// `llama_ftype_get_default_type`.
    #[must_use]
    pub fn default_type(self) -> GgmlType {
        match self {
            Ftype::Q8_0 => GgmlType::Q8_0,
            Ftype::Q4KM => GgmlType::Q4_K,
            Ftype::Q5KM => GgmlType::Q5_K,
            Ftype::Q6K => GgmlType::Q6_K,
            Ftype::Iq2Xxs => GgmlType::IQ2_XXS,
            Ftype::Iq1S => GgmlType::IQ1_S,
        }
    }

    /// The IQ file types, whose `n_gqa`/`n_expert` attention rules differ.
    fn is_iq(self) -> bool {
        matches!(self, Ftype::Iq2Xxs | Ftype::Iq1S)
    }
}

/// The type a flag or a rule names, among those the port writes or copies.
pub fn parse_type(s: &str) -> Result<GgmlType> {
    Ok(match s.to_ascii_uppercase().as_str() {
        "F32" => GgmlType::F32,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        "Q8_0" => GgmlType::Q8_0,
        "IQ1_S" => GgmlType::IQ1_S,
        "IQ2_XXS" => GgmlType::IQ2_XXS,
        o => {
            return Err(Error::refused(format!(
                "type {o}: the quantizer writes F32, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S and IQ2_XXS"
            )));
        }
    })
}

/// What the type logic reads of the model: `<arch>.block_count`, `<arch>.expert_count`.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    pub n_layer_all: u32,
    pub n_expert: u32,
}

impl Model {
    /// From the first shard's metadata; an architecture no row of [`ARCHS`] names is refused
    /// naming the rows.
    pub fn read(set: &dyn Source) -> Result<Model> {
        let arch = set.architecture().unwrap_or("");
        if !ARCHS.iter().any(|&(name, _)| name == arch) {
            let names: Vec<&str> = ARCHS.iter().map(|&(name, _)| name).collect();
            return Err(Error::refused(format!(
                "architecture {arch:?}: the type logic is ported for {} only",
                names.join(", ")
            )));
        }
        let get = |suffix: &str| {
            set.arch_get_u64(suffix)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| Error::refused(format!("no {}", set.arch_key(suffix))))
        };
        Ok(Model {
            n_layer_all: get("block_count")?,
            n_expert: get("expert_count")?,
        })
    }
}

/// The flags that move the logic.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub ftype: Ftype,
    pub output_tensor_type: Option<GgmlType>,
    pub token_embedding_type: Option<GgmlType>,
}

/// One tensor as the logic reads it.
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub ty: GgmlType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Output,
    TokenEmbd,
    AttentionQkv,
    AttentionKvB,
    AttentionV,
    AttentionK,
    AttentionQ,
    AttentionOutput,
    FfnUp,
    FfnGate,
    FfnLatent,
    AttentionGate,
    FfnDown,
    Other,
}

/// `tensor_get_category`.
fn category(name: &str) -> Category {
    let has = |s: &str| name.contains(s);
    if name == "output.weight" {
        Category::Output
    } else if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
        Category::TokenEmbd
    } else if has("attn_qkv.weight") || has("attn_qkv_gate.weight") {
        Category::AttentionQkv
    } else if has("attn_kv_b.weight") {
        Category::AttentionKvB
    } else if has("attn_v.weight") {
        Category::AttentionV
    } else if has("attn_k.weight") {
        Category::AttentionK
    } else if has("attn_q.weight") {
        Category::AttentionQ
    } else if has("attn_output.weight") {
        Category::AttentionOutput
    } else if has("ffn_up") {
        Category::FfnUp
    } else if has("ffn_gate") {
        Category::FfnGate
    } else if has("ffn_latent") {
        Category::FfnLatent
    } else if has("attn_gate.weight") {
        Category::AttentionGate
    } else if has("ffn_down") {
        Category::FfnDown
    } else {
        Category::Other
    }
}

fn is_attn_v(c: Category) -> bool {
    matches!(
        c,
        Category::AttentionV | Category::AttentionQkv | Category::AttentionKvB
    )
}

/// `tensor_allows_quantization`, as it reads with `quantize_output_tensor` on and no `only_copy`: the
/// arch-free checks every [`ARCHS`] set needs, no row of another architecture's own.
fn allows_quantization(name: &str, dims: &[u64]) -> bool {
    let n_dims = dims
        .iter()
        .rposition(|&d| d != 1)
        .map_or(1, |p| p + 1)
        .max(1);
    if n_dims < 2 || !name.ends_with("weight") {
        return false;
    }
    const EXCLUDED: &[&str] = &[
        "_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_tid2eid.weight",
        "engram_gate.weight",
        "altup",
        "laurel",
        "per_layer_model_proj",
        "ssm_conv1d",
        "shortconv.conv.weight",
        "conv2d_spatial.weight",
        "conv_dw",
        "indexer.k_proj.weight",
        "indexer.q_proj.weight",
        "time_mix_first.weight",
        "time_mix_w0.weight",
        "time_mix_w1.weight",
        "time_mix_w2.weight",
        "time_mix_v0.weight",
        "time_mix_v1.weight",
        "time_mix_v2.weight",
        "time_mix_a0.weight",
        "time_mix_a1.weight",
        "time_mix_a2.weight",
        "time_mix_g1.weight",
        "time_mix_g2.weight",
        "time_mix_decay_w1.weight",
        "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight",
        "attn_rel_b.weight",
        ".position_embd",
        "sam.pos_embd",
        "sam.neck.",
        "sam.net_",
        ".rel_pos",
        ".patch_embd",
        ".patch_merger",
        "a.rvq.codebook",
        "mm.a.code_embd",
    ];
    !(EXCLUDED.iter().any(|e| name.contains(e))
        || name.starts_with("a.conv1d.")
        || name.starts_with("a.gen.wav.")
        // the positional-embedding and token-type tensors LLM_TN names for BERT-like architectures
        || name == "position_embd.weight"
        || name == "token_types.weight")
}

/// `use_more_bits`.
fn use_more_bits(i_layer: i64, n_layers: i64) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// The layer `blk.<n>.` at the start of a name (`sscanf(name, "blk.%d.", …)`).
#[must_use]
pub fn layer_of(name: &str) -> Option<i64> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    (end > 0 && rest[end..].starts_with('.'))
        .then(|| rest[..end].parse().ok())
        .flatten()
}

struct State {
    n_attention_wv: i64,
    n_ffn_down: i64,
    i_attention_wv: i64,
    i_ffn_down: i64,
    has_tied_embeddings: bool,
}

/// `tensor_type_fallback`: a row that is not a whole number of the target's blocks.
fn fallback(name: &str, ne0: u64, target: GgmlType) -> Result<GgmlType> {
    let blck = target.blck_size().unwrap_or(1);
    if ne0.is_multiple_of(blck) {
        return Ok(target);
    }
    Err(Error::refused(format!(
        "{name}: {ne0} columns are not a whole number of {target:?} blocks of {blck}: llama-quantize falls back to another type there, which this port does not write"
    )))
}

/// `llama_tensor_get_type_impl` for the covered file types.
fn type_impl(
    st: &mut State,
    m: &Model,
    o: &Options,
    t: &TensorRef<'_>,
    mut new_type: GgmlType,
    cat: Category,
) -> Result<GgmlType> {
    let ftype = o.ftype;
    let n_expert = m.n_expert;
    let unported = |what: &str| {
        Err(Error::refused(format!(
            "{}: llama-quantize's type for it under {ftype:?} is {what}, which the quantizer does not write",
            t.name
        )))
    };
    let layer_info = |i_layer: i64, n_layer: i64| -> Result<(i64, i64)> {
        if n_expert > 1 {
            let l = layer_of(t.name).ok_or_else(|| {
                Error::refused(format!("Failed to determine layer for tensor {}", t.name))
            })?;
            if l < 0 || l >= n_layer {
                return Err(Error::refused(format!(
                    "Bad layer {l} for tensor {}. Must be in [0, {n_layer})",
                    t.name
                )));
            }
            Ok((l, n_layer))
        } else {
            Ok((i_layer, n_layer))
        }
    };
    if cat == Category::Output || (st.has_tied_embeddings && cat == Category::TokenEmbd) {
        if let Some(ot) = o.output_tensor_type {
            new_type = ot;
        } else {
            let qk_k = new_type.blck_size().unwrap_or(1);
            if !t.dims[0].is_multiple_of(qk_k) {
                new_type = GgmlType::Q8_0;
            } else if ftype.is_iq() {
                new_type = GgmlType::Q5_K;
            } else if new_type != GgmlType::Q8_0 {
                new_type = GgmlType::Q6_K;
            }
        }
    } else if cat == Category::TokenEmbd {
        if ftype.is_iq() {
            return unported("Q2_K");
        }
    } else if ftype.is_iq() {
        if is_attn_v(cat) {
            if n_expert >= 4 {
                new_type = GgmlType::Q4_K;
            } else {
                return unported("a type chosen by the head counts' ratio");
            }
            st.i_attention_wv += 1;
        } else if n_expert == 8 && cat == Category::AttentionK {
            new_type = GgmlType::Q4_K;
        } else if cat == Category::FfnDown {
            if st.i_ffn_down < st.n_ffn_down / 8 {
                return unported("Q2_K");
            }
            st.i_ffn_down += 1;
        } else if cat == Category::AttentionOutput {
            if n_expert == 8 {
                new_type = GgmlType::Q5_K;
            } else if ftype == Ftype::Iq1S {
                new_type = GgmlType::IQ2_XXS;
            }
        }
    } else if is_attn_v(cat) {
        if matches!(ftype, Ftype::Q4KM | Ftype::Q5KM)
            && use_more_bits(st.i_attention_wv, st.n_attention_wv)
        {
            new_type = GgmlType::Q6_K;
        }
        if n_expert == 8 {
            new_type = GgmlType::Q8_0;
        }
        st.i_attention_wv += 1;
    } else if cat == Category::AttentionK {
        if n_expert == 8 {
            new_type = GgmlType::Q8_0;
        }
    } else if cat == Category::FfnDown {
        let (i_layer, n_layer) = layer_info(st.i_ffn_down, st.n_ffn_down)?;
        if matches!(ftype, Ftype::Q4KM | Ftype::Q5KM) && use_more_bits(i_layer, n_layer) {
            new_type = GgmlType::Q6_K;
        }
        st.i_ffn_down += 1;
    } else if cat == Category::AttentionOutput {
        if n_expert == 8 && ftype == Ftype::Q4KM {
            new_type = GgmlType::Q5_K;
        }
    } else if cat == Category::FfnGate || cat == Category::FfnUp {
        layer_info(0, st.n_ffn_down)?;
    }
    Ok(new_type)
}

/// Every tensor's type, in the order given (the order ggml's driver walks them).
pub fn resolve(
    tensors: &[TensorRef<'_>],
    rules: &Rules,
    m: &Model,
    o: &Options,
) -> Result<Vec<GgmlType>> {
    let mut st = State {
        n_attention_wv: 0,
        n_ffn_down: i64::from(m.n_layer_all),
        i_attention_wv: 0,
        i_ffn_down: 0,
        has_tied_embeddings: true,
    };
    for t in tensors {
        let c = category(t.name);
        if is_attn_v(c) {
            st.n_attention_wv += 1;
        }
        if c == Category::Output {
            st.has_tied_embeddings = false;
        }
    }
    let mut out = Vec::with_capacity(tensors.len());
    for t in tensors {
        if !allows_quantization(t.name, t.dims) {
            out.push(t.ty);
            continue;
        }
        let cat = category(t.name);
        if let (Some(tt), Category::TokenEmbd) = (o.token_embedding_type, cat)
            && t.name != "per_layer_token_embd.weight"
        {
            out.push(tt);
            continue;
        }
        if let (Some(tt), Category::Output) = (o.output_tensor_type, cat) {
            out.push(tt);
            continue;
        }
        let mut new_type = o.ftype.default_type();
        let manual = match rules.type_of(t.name) {
            Some(r) => {
                new_type = parse_type(r)?;
                true
            }
            None => false,
        };
        if !manual {
            new_type = type_impl(&mut st, m, o, t, new_type, cat)?;
        }
        out.push(fallback(t.name, t.dims[0], new_type)?);
    }
    Ok(out)
}
