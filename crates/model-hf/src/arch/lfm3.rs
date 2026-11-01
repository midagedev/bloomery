//! LFM3's Hugging Face checkpoint (`lfm3`: LFM3-1.7B and its siblings) as MLX's convert stores
//! it, in the lfm3 names, layouts and keys llama.cpp's converter (`conversion/lfm3.py`,
//! `LFM3Model`, at `19cce69`) writes into a GGUF. A layer is a short convolution or attention
//! by `layer_types`, and every layer has the sandwich norms and a SwiGLU block.
//!
//! A checkpoint tensor maps to one tensor of the file the way its role reads it ([`Use`]). The
//! convolution's taps arrive `[channels, kernel, 1]` and are the file's `[kernel, channels]`
//! rows once the unit dim is dropped ([`Squeeze`]), tap order unchanged. The rope's frequency
//! factors, which the converter writes as the tensor `rope_freqs.weight`, are made of
//! `partial_rotary_factor` ([`FolderMap::constants`]). The model ties its head to the
//! embedding, so no `output.weight` is made; a folder whose `tie_word_embeddings` and head
//! disagree is refused ([`trunk::check_head`]). The header carries the keys the file does.

use std::collections::BTreeMap;

use gguf::Value;
use hf::modelopt::{Normal, Plain, Use};
use hf::schema::{Fields, Rule::*, Type};
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{
    Constant, Error, FolderMap, Mapped, Mapping, Origin, Squeeze, VocabSpec, refuse,
};

const ARCHITECTURE: &str = "lfm3";

/// The checkpoint's prefix of the text model's tensors.
const MODEL: &str = "language_model.model.";

/// The output head a folder of an untied model holds.
const HEAD: &str = "language_model.lm_head.weight";

/// The tensor names of the file: gguf-py's `constants.py` `TENSOR_NAMES` of the LFM3 tensors.
mod names {
    pub(super) use model::arch::names::{
        attn_k, attn_k_norm, attn_norm, attn_output, attn_q, attn_q_norm, attn_v, ffn_down,
        ffn_gate, ffn_norm, ffn_up, output, output_norm, rope_freqs, token_embd,
    };

    use model::arch::names::blk;

    /// `blk.{l}.post_attention_norm.weight`, the norm of the mixer's output.
    pub(super) fn post_attention_norm(l: usize) -> String {
        blk(l, "post_attention_norm.weight")
    }

    /// `blk.{l}.post_ffw_norm.weight`, the norm of the feed-forward block's output.
    pub(super) fn post_ffw_norm(l: usize) -> String {
        blk(l, "post_ffw_norm.weight")
    }

    /// `blk.{l}.attn_gate.weight`, the per-head sigmoid output gate's projection.
    pub(super) fn attn_gate(l: usize) -> String {
        blk(l, "attn_gate.weight")
    }

    /// `blk.{l}.shortconv.in_proj.weight`, the projection into `B | C | x`.
    pub(super) fn shortconv_in_proj(l: usize) -> String {
        blk(l, "shortconv.in_proj.weight")
    }

    /// `blk.{l}.shortconv.conv.weight`, the depthwise taps, oldest first.
    pub(super) fn shortconv_conv(l: usize) -> String {
        blk(l, "shortconv.conv.weight")
    }

    /// `blk.{l}.shortconv.out_proj.weight`.
    pub(super) fn shortconv_out_proj(l: usize) -> String {
        blk(l, "shortconv.out_proj.weight")
    }
}

const SAMPLING: &str =
    "the checkpoint's sampling defaults; a request's own settings decide how the program samples";

/// `config.json`, the text model's own object. `hf::quant_config` reads `quantization` and its
/// copy `quantization_config`; `hf::vocab::Config` reads the vocabulary's size and special ids.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["Lfm3ForCausalLM"]]"#)),
    ("model_type", Need(r#"["lfm3"]"#)),
    ("quantization", Read),
    ("quantization_config", Read),
    ("num_hidden_layers", Read),
    ("max_position_embeddings", Read),
    ("hidden_size", Read),
    ("intermediate_size", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("head_dim", Read),
    ("norm_eps", Read),
    // hf::vocab::Config's.
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("unk_token_id", Read),
    ("sep_token_id", Read),
    // The short convolution's kernel width, one tap each channel with no bias.
    ("conv_L_cache", Read),
    ("conv_bias", Only("[false]")),
    // Each layer's kind, the rope of each attention kind, and the window of a sliding layer.
    ("layer_types", Read),
    ("rope_parameters", Read),
    ("sliding_window", Read),
    // A norm after each mixer and each feed-forward block, its output added to the residual.
    ("use_sandwich_norm", Need("[true]")),
    trunk::HEAD_EITHER,
    // An empty tower config is no tower; a tower's tensors are not this map's.
    ("vision_config", Only("[{}]")),
    (
        "auto_map",
        Ignore(Type::Object, "names HF's classes for its own loader"),
    ),
    (
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
    ("do_sample", Ignore(Type::Bool, SAMPLING)),
    ("temperature", Ignore(Type::Number, SAMPLING)),
    ("top_k", Ignore(Type::Number, SAMPLING)),
    ("generation_config", Ignore(Type::Object, SAMPLING)),
];

/// The counts of [`ROOT`] the map reads: each a positive u32.
const COUNTS: &[&str] = &[
    "num_hidden_layers",
    "max_position_embeddings",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "conv_L_cache",
    "sliding_window",
];

/// `rope_parameters`: one object per attention kind.
const ROPE: &Fields = &[("full_attention", Read), ("sliding_attention", Read)];

/// `rope_parameters.full_attention`: the pairs the rope turns are the first
/// `partial_rotary_factor` of them, as the converter's frequency factors say.
const ROPE_FULL: &Fields = &[
    ("rope_type", Only(r#"["default"]"#)),
    ("rope_theta", Read),
    ("partial_rotary_factor", Read),
];

/// `rope_parameters.sliding_attention`: every pair turns, as the file has it.
const ROPE_SLIDING: &Fields = &[
    ("rope_type", Only(r#"["default"]"#)),
    ("rope_theta", Read),
    ("partial_rotary_factor", Only("[1.0, 1]")),
];

/// What a layer does, by `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Conv,
    Full,
    Sliding,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Conv => "conv",
            Kind::Full => "full_attention",
            Kind::Sliding => "sliding_attention",
        }
    }
}

/// The program's tensor of a checkpoint tensor: its name, the role it is read in, and whether
/// its unit dim is dropped.
struct Target {
    to: String,
    plain: Plain,
    squeeze: bool,
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    kinds: Vec<Kind>,
    hidden: u64,
    ffn: u64,
    heads: u64,
    query_rows: u64,
    key_rows: u64,
    head_dim: u64,
    taps: u64,
    tied: bool,
    /// The rope's frequency factor of each pair of a full-attention layer.
    rope_factors: Vec<f32>,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

impl Checkpoint {
    /// Each layer's kind: `layer_types` lists one a layer, and each is one the map reads.
    fn kinds(root: &Section<'_, Error>, layers: usize) -> Result<Vec<Kind>, Error> {
        let types = root.array("layer_types")?;
        if types.len() != layers {
            return Err(root.bad(
                "layer_types",
                &format!(
                    "lists {} layers, num_hidden_layers is {layers}",
                    types.len()
                ),
            ));
        }
        types
            .iter()
            .enumerate()
            .map(|(l, t)| match t.as_str() {
                Some("conv") => Ok(Kind::Conv),
                Some("full_attention") => Ok(Kind::Full),
                Some("sliding_attention") => Ok(Kind::Sliding),
                _ => Err(root.bad(
                    "layer_types",
                    &format!(
                        "holds {t} at layer {l}, not conv, full_attention or sliding_attention"
                    ),
                )),
            })
            .collect()
    }
}

impl FolderMap for Checkpoint {
    /// The convolution's unit dim is the one move.
    type Step = Squeeze;

    const MODEL_TYPES: &'static [&'static str] = &["lfm3"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT])?;
        let mut counts = BTreeMap::new();
        for &field in COUNTS {
            counts.insert(field, root.count(field)?);
        }
        let n = |key: &str| counts[key];
        let wide = |key: &str| u64::from(counts[key]);
        if !n("num_attention_heads").is_multiple_of(n("num_key_value_heads")) {
            return Err(root.bad("num_attention_heads", "must group by num_key_value_heads"));
        }
        if !n("head_dim").is_multiple_of(2) {
            return Err(root.bad("head_dim", "must be even: the rope turns pairs"));
        }
        let kinds = Self::kinds(&root, n("num_hidden_layers") as usize)?;

        let rope = Section::<Error>::new(
            root.field("rope_parameters")?,
            "config.json.rope_parameters",
        )?;
        rope.check(&[ROPE])?;
        let full = Section::<Error>::new(
            rope.field("full_attention")?,
            "config.json.rope_parameters.full_attention",
        )?;
        full.check(&[ROPE_FULL])?;
        let sliding_theta = match rope.value.get("sliding_attention") {
            Some(value) => {
                let sliding =
                    Section::<Error>::new(value, "config.json.rope_parameters.sliding_attention")?;
                sliding.check(&[ROPE_SLIDING])?;
                Some(sliding.number("rope_theta")?)
            }
            None if kinds.contains(&Kind::Sliding) => {
                return Err(rope.bad("sliding_attention", "is missing, and a layer slides"));
            }
            None => None,
        };
        // The converter's `int(partial * head_dim) // 2` angles turn; the others' factor is 1e30.
        let partial = match full.value.get("partial_rotary_factor") {
            None => 1.0,
            Some(v) => v
                .as_f64()
                .filter(|p| *p > 0.0 && *p <= 1.0)
                .ok_or_else(|| full.bad("partial_rotary_factor", "must be a number in (0, 1]"))?,
        };
        let pairs = n("head_dim") as usize / 2;
        let angles = (partial * f64::from(n("head_dim"))).floor() as usize / 2;
        if angles == 0 {
            return Err(full.bad("partial_rotary_factor", "turns no pair"));
        }
        let rope_factors = (0..pairs)
            .map(|i| if i < angles { 1.0 } else { 1e30 })
            .collect();

        let kv = i32::try_from(n("num_key_value_heads"))
            .map_err(|_| root.bad("num_key_value_heads", "does not fit i32"))?;
        let u32_key = |key: &str, from: &str| (key.to_string(), Value::U32(n(from)));
        let mut keys = vec![
            u32_key("block_count", "num_hidden_layers"),
            u32_key("context_length", "max_position_embeddings"),
            u32_key("embedding_length", "hidden_size"),
            u32_key("feed_forward_length", "intermediate_size"),
            u32_key("attention.head_count", "num_attention_heads"),
            (
                "attention.head_count_kv".to_string(),
                Value::Array(
                    kinds
                        .iter()
                        .map(|&k| Value::I32(if k == Kind::Conv { 0 } else { kv }))
                        .collect(),
                ),
            ),
            (
                "rope.freq_base".to_string(),
                Value::F32(full.number("rope_theta")?),
            ),
        ];
        if let Some(theta) = sliding_theta {
            keys.push(("rope.freq_base_swa".to_string(), Value::F32(theta)));
        }
        keys.extend([
            (
                "attention.layer_norm_rms_epsilon".to_string(),
                Value::F32(root.number("norm_eps")?),
            ),
            u32_key("attention.key_length", "head_dim"),
            u32_key("attention.value_length", "head_dim"),
            u32_key("rope.dimension_count", "head_dim"),
            (
                "attention.sliding_window_pattern".to_string(),
                Value::Array(
                    kinds
                        .iter()
                        .map(|&k| Value::Bool(k == Kind::Sliding))
                        .collect(),
                ),
            ),
            u32_key("attention.sliding_window", "sliding_window"),
            u32_key("shortconv.l_cache", "conv_L_cache"),
        ]);

        Ok(Self {
            hidden: wide("hidden_size"),
            ffn: wide("intermediate_size"),
            heads: wide("num_attention_heads"),
            query_rows: wide("num_attention_heads") * wide("head_dim"),
            key_rows: wide("num_key_value_heads") * wide("head_dim"),
            head_dim: wide("head_dim"),
            taps: wide("conv_L_cache"),
            tied: trunk::tied(&root)?,
            kinds,
            rope_factors,
            vocab: hf::vocab::Config::parse(config)?.with_pre("lfm2")?,
            keys,
        })
    }

    fn module_class(&self, module: &str) -> Option<hf::quant_config::ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Squeeze>, Error> {
        let head = tensors.iter().map(|&(name, _)| name).find(|&n| n == HEAD);
        trunk::check_head(self.tied, head).map_err(Error::Refused)?;
        let mut out: BTreeMap<String, Mapped<Squeeze>> = BTreeMap::new();
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let Target { to, plain, squeeze } = self.tensor(name, shape)?;
            let (steps, dims) = if squeeze {
                (vec![Squeeze], vec![self.taps, self.hidden])
            } else {
                (vec![], shape.iter().rev().map(|&d| d as u64).collect())
            };
            let mapped = Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(plain),
                steps,
                dims,
                origin: Origin::Made,
            };
            if let Some(prev) = out.insert(to, mapped) {
                return Err(refuse(
                    &prev.sources[0],
                    "maps to an lfm3 name another tensor has",
                ));
            }
        }
        self.complete(&out)?;
        Ok(Mapping {
            tensors: out,
            skipped: BTreeMap::new(),
        })
    }

    fn mover(&self, name: &str, steps: &[Squeeze]) -> Option<crate::part::Step> {
        (!steps.is_empty()).then(|| {
            let name = name.to_string();
            Box::new(move |n: &mut Normal| Squeeze.apply(n).map_err(|why| format!("{name}: {why}")))
                as crate::part::Step
        })
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn header(
        &self,
        _planned: &[(&str, &[u64])],
        _folder: &safetensors::Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        Ok(trunk::header(ARCHITECTURE, &self.keys))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }

    fn constants(&self) -> Result<BTreeMap<String, Constant>, Error> {
        Ok(BTreeMap::from([(
            names::rope_freqs(),
            Constant {
                dims: vec![self.rope_factors.len() as u64],
                values: self.rope_factors.clone(),
            },
        )]))
    }
}

impl Checkpoint {
    /// The program's tensor of the checkpoint tensor `name` (row-major logical `shape`) and the
    /// role it is read in; a tensor no LFM3 folder of MLX's has, or one a layer of another kind
    /// holds, is refused with why.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Target, Error> {
        let norm = |to: String| Target {
            to,
            plain: Plain::F32,
            squeeze: false,
        };
        let linear = |to: String| Target {
            to,
            plain: Plain::Q8_0,
            squeeze: false,
        };
        if name == HEAD {
            return Ok(linear(names::output()));
        }
        let short = name
            .strip_prefix(MODEL)
            .ok_or_else(|| refuse(name, "not an LFM3 tensor"))?;
        match short {
            "embed_tokens.weight" => return Ok(linear(names::token_embd())),
            "embedding_norm.weight" => return Ok(norm(names::output_norm())),
            _ => {}
        }
        let (l, rest) = short
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            .ok_or_else(|| refuse(name, "not an LFM3 tensor"))?;
        let Some(&kind) = self.kinds.get(l) else {
            return Err(refuse(name, format!("layer {l} of {}", self.kinds.len())));
        };
        // A tensor of the mixer a layer does not have.
        let holds = |conv: bool| {
            if (kind == Kind::Conv) == conv {
                return Ok(());
            }
            Err(refuse(
                name,
                format!(
                    "layer {l} is {} by layer_types, which holds no {} tensor",
                    kind.label(),
                    if conv { "conv" } else { "attention" }
                ),
            ))
        };
        Ok(match rest {
            "operator_norm.weight" => norm(names::attn_norm(l)),
            "post_operator_norm.weight" => norm(names::post_attention_norm(l)),
            "ffn_norm.weight" => norm(names::ffn_norm(l)),
            "post_ffn_norm.weight" => norm(names::post_ffw_norm(l)),
            "feed_forward.w1.weight" => linear(names::ffn_gate(l)),
            "feed_forward.w3.weight" => linear(names::ffn_up(l)),
            "feed_forward.w2.weight" => linear(names::ffn_down(l)),
            "conv.in_proj.weight" => {
                holds(true)?;
                linear(names::shortconv_in_proj(l))
            }
            "conv.out_proj.weight" => {
                holds(true)?;
                linear(names::shortconv_out_proj(l))
            }
            "conv.conv.weight" => {
                holds(true)?;
                if shape != [self.hidden as usize, self.taps as usize, 1] {
                    return Err(refuse(
                        name,
                        format!(
                            "{shape:?} is not MLX's [channels, kernel, 1] of {} channels and \
                             conv_L_cache {} taps",
                            self.hidden, self.taps
                        ),
                    ));
                }
                Target {
                    to: names::shortconv_conv(l),
                    plain: Plain::F32,
                    squeeze: true,
                }
            }
            "self_attn.gate_proj.weight" => {
                holds(false)?;
                linear(names::attn_gate(l))
            }
            other => {
                let (to, plain) =
                    trunk::attention(l, other).ok_or_else(|| refuse(name, "not an LFM3 tensor"))?;
                holds(false)?;
                Target {
                    to,
                    plain,
                    squeeze: false,
                }
            }
        })
    }

    /// The tensors layer `l` holds, with their ggml dims the config gives.
    fn layer_tensors(&self, l: usize) -> Vec<(String, Vec<u64>)> {
        let (h, f) = (self.hidden, self.ffn);
        let mut out = vec![
            (names::attn_norm(l), vec![h]),
            (names::post_attention_norm(l), vec![h]),
            (names::ffn_norm(l), vec![h]),
            (names::post_ffw_norm(l), vec![h]),
            (names::ffn_gate(l), vec![h, f]),
            (names::ffn_up(l), vec![h, f]),
            (names::ffn_down(l), vec![f, h]),
        ];
        match self.kinds[l] {
            Kind::Conv => out.extend([
                (names::shortconv_in_proj(l), vec![h, 3 * h]),
                (names::shortconv_conv(l), vec![self.taps, h]),
                (names::shortconv_out_proj(l), vec![h, h]),
            ]),
            Kind::Full | Kind::Sliding => out.extend([
                (names::attn_q(l), vec![h, self.query_rows]),
                (names::attn_k(l), vec![h, self.key_rows]),
                (names::attn_v(l), vec![h, self.key_rows]),
                (names::attn_output(l), vec![self.query_rows, h]),
                (names::attn_gate(l), vec![h, self.heads]),
                (names::attn_q_norm(l), vec![self.head_dim]),
                (names::attn_k_norm(l), vec![self.head_dim]),
            ]),
        }
        out
    }

    /// Every tensor the file holds is mapped, at the dims the config gives: the embedding, the
    /// output norm, the head of an untied model, and each layer's tensors.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Squeeze>>) -> Result<(), Error> {
        let vocab = u64::from(self.vocab.size());
        let mut want = vec![
            (names::token_embd(), vec![self.hidden, vocab]),
            (names::output_norm(), vec![self.hidden]),
        ];
        if !self.tied {
            want.push((names::output(), vec![self.hidden, vocab]));
        }
        want.extend((0..self.kinds.len()).flat_map(|l| self.layer_tensors(l)));
        for (name, dims) in want {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing from the folder"))?;
            if m.dims != dims {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {dims:?}", m.dims),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use gguf::GgmlType;
    use hf::quant_config::Config;
    use safetensors::Dtype;
    use serde_json::json;

    use super::*;
    use crate::folder;
    use crate::test_support::{Entries, entries, gguf_tensors, json, key_json};

    const CONFIG: &str = "quant-configs/LiquidAI__LFM3-1.7B-MLX-4bit__config.json.gz";
    const HEADER: &str = "checkpoint-headers/lfm3-1.7b-mlx-4bit-model.header.json.gz";
    const GGUF: &str = "checkpoint-headers/lfm3-1.7b-q4km-gguf.header.json.gz";
    const LAYERS: &str = "language_model.model.layers";

    /// The 594 tensors of the folder's one file.
    fn real_entries() -> Entries {
        let mut header = json(HEADER);
        header.as_object_mut().unwrap().remove("__metadata__");
        entries(&header)
    }

    fn planning(
        checkpoint: &Checkpoint,
        config: &Json,
        entries: &Entries,
    ) -> Result<folder::Planning<Squeeze>, folder::Error> {
        folder::plan_tensors(
            checkpoint,
            &Config::parse(config, None).unwrap(),
            entries,
            false,
            folder::Keep::InPlace,
        )
    }

    /// Why the folder is refused: the config's, or its tensors'.
    fn refusal(config: &Json, entries: &Entries) -> String {
        let checkpoint = match Checkpoint::parse(config) {
            Ok(c) => c,
            Err(e) => return e.to_string(),
        };
        match planning(&checkpoint, config, entries) {
            Ok(_) => panic!("the folder must be refused"),
            Err(e) => e.to_string(),
        }
    }

    /// The MLX 4-bit linear `stem`: its codes, scales and biases, of `rows` rows of `k` values.
    fn linear(stem: &str, rows: usize, k: usize) -> Entries {
        vec![
            (format!("{stem}.weight"), Dtype::U32, vec![rows, k / 8]),
            (format!("{stem}.scales"), Dtype::Bf16, vec![rows, k / 64]),
            (format!("{stem}.biases"), Dtype::Bf16, vec![rows, k / 64]),
        ]
    }

    /// Layer `l`'s kind by the real config's `layer_types`.
    fn kind_of(config: &Json, l: usize) -> Kind {
        match config["layer_types"][l].as_str().unwrap() {
            "conv" => Kind::Conv,
            "full_attention" => Kind::Full,
            _ => Kind::Sliding,
        }
    }

    /// The real 1.7B folder's header (594 tensors) plans as the GGUF file of the same model
    /// (`LFM3-1.7B-Q4_K_M.gguf`, 287 tensors): the same names at the same ggml dims, each linear
    /// and the embedding MLX's 4-bit blocks where the file has K-quants, every other tensor F32,
    /// each tensor made of the checkpoint tensor llama.cpp's converter reads it from (`w1` is
    /// `ffn_gate`, `w3` `ffn_up`, `post_operator_norm` `post_attention_norm`, the conv's unit
    /// dim dropped), `rope_freqs.weight` the one tensor no checkpoint tensor holds, and no head
    /// beside the embedding. Every stored byte is a byte a planned tensor holds. The header's keys
    /// are the file's `lfm3.*` keys. Mutants (each red at its line): `w1` and `w3` swapped (the
    /// `ffn_gate` sources), the conv's dims left as stored (`blk.0.shortconv.conv.weight` dims),
    /// `post_operator_norm` read as `ffn_norm` (`maps to an lfm3 name another tensor has`),
    /// `head_count_kv` of every layer (the keys).
    #[test]
    fn the_real_folder_plans_the_gguf_files_tensors_and_keys() {
        let config = json(CONFIG);
        let gguf = json(GGUF);
        let entries = real_entries();
        assert_eq!(entries.len(), 594);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let planning = planning(&checkpoint, &config, &entries).unwrap();
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        assert!(planning.skipped.is_empty());
        let want = gguf_tensors(&gguf);
        assert_eq!(want.len(), 287);
        let got: BTreeMap<&str, &folder::Planned<Squeeze>> = planning
            .planned
            .iter()
            .map(|p| (p.decl.name.as_str(), p))
            .collect();
        assert_eq!(
            got.keys().copied().collect::<Vec<_>>(),
            want.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (name, (dims, ty)) in &want {
            let mlx = GgmlType::from_u32(got[name.as_str()].decl.type_id);
            assert_eq!(got[name.as_str()].decl.dims, *dims, "{name}");
            // The file's K-quants are MLX's 4-bit blocks here; its F32 tensors are the folder's
            // BF16 vectors and taps, and the rope's factors.
            assert_eq!(*ty == GgmlType::F32, mlx == GgmlType::F32, "{name}");
            assert!(
                matches!(mlx, GgmlType::F32 | GgmlType::MLX_A4),
                "{name}: {mlx}"
            );
        }
        let made = |name: &str| match &got[name].make {
            folder::Make::Tensor(m) => m,
            _ => panic!("{name}: no tensor of the folder"),
        };
        let one = |name: &str, source: String, plain: Plain, steps: Vec<Squeeze>| {
            let m = made(name);
            assert_eq!(m.sources, [source], "{name}");
            assert_eq!(m.role, Use::Matrix(plain), "{name}");
            assert_eq!(m.steps, steps, "{name}");
        };
        one(
            "token_embd.weight",
            "language_model.model.embed_tokens.weight".into(),
            Plain::Q8_0,
            vec![],
        );
        one(
            "output_norm.weight",
            "language_model.model.embedding_norm.weight".into(),
            Plain::F32,
            vec![],
        );
        assert!(!got.contains_key("output.weight"));
        let mut conv = 0;
        for l in 0..24 {
            let kind = kind_of(&config, l);
            let all = [
                ("attn_norm", "operator_norm", Plain::F32),
                ("post_attention_norm", "post_operator_norm", Plain::F32),
                ("ffn_norm", "ffn_norm", Plain::F32),
                ("post_ffw_norm", "post_ffn_norm", Plain::F32),
                ("ffn_gate", "feed_forward.w1", Plain::Q8_0),
                ("ffn_up", "feed_forward.w3", Plain::Q8_0),
                ("ffn_down", "feed_forward.w2", Plain::Q8_0),
            ];
            let mixer: &[(&str, &str, Plain)] = if kind == Kind::Conv {
                conv += 1;
                &[
                    ("shortconv.in_proj", "conv.in_proj", Plain::Q8_0),
                    ("shortconv.conv", "conv.conv", Plain::F32),
                    ("shortconv.out_proj", "conv.out_proj", Plain::Q8_0),
                ]
            } else {
                &[
                    ("attn_q", "self_attn.q_proj", Plain::Q8_0),
                    ("attn_k", "self_attn.k_proj", Plain::Q8_0),
                    ("attn_v", "self_attn.v_proj", Plain::Q8_0),
                    ("attn_output", "self_attn.o_proj", Plain::Q8_0),
                    ("attn_gate", "self_attn.gate_proj", Plain::Q8_0),
                    ("attn_q_norm", "self_attn.q_norm", Plain::F32),
                    ("attn_k_norm", "self_attn.k_norm", Plain::F32),
                ]
            };
            for &(stem, tail, plain) in all.iter().chain(mixer) {
                let steps = if stem == "shortconv.conv" {
                    vec![Squeeze]
                } else {
                    vec![]
                };
                one(
                    &format!("blk.{l}.{stem}.weight"),
                    format!("{LAYERS}.{l}.{tail}.weight"),
                    plain,
                    steps,
                );
            }
        }
        assert_eq!(conv, 13);
        let folder::Make::Constant(rope) = &got["rope_freqs.weight"].make else {
            panic!("rope_freqs.weight is a tensor of the folder");
        };
        assert_eq!(rope.dims, [64]);
        let stored: usize = entries
            .iter()
            .map(|(_, d, s)| s.iter().product::<usize>() * d.size())
            .sum();
        let held: usize = planning
            .planned
            .iter()
            .filter(|p| !matches!(p.make, folder::Make::Constant(_)))
            .map(|p| match GgmlType::from_u32(p.decl.type_id) {
                GgmlType::F32 => p.decl.nbytes as usize / 2,
                _ => p.decl.nbytes as usize,
            })
            .sum();
        assert_eq!(held, stored);
        let got_keys: serde_json::Map<String, Json> = checkpoint
            .keys
            .iter()
            .map(|(k, v)| (format!("{ARCHITECTURE}.{k}"), key_json(v)))
            .collect();
        let want_keys: serde_json::Map<String, Json> = gguf["keys"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.starts_with("lfm3."))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(want_keys.len(), 15);
        assert_eq!(Json::Object(got_keys), Json::Object(want_keys));
    }

    /// `rope_freqs.weight` is the converter's `[1.0] * angles + [1e30] * (pairs - angles)` with
    /// `angles = int(partial * head_dim) // 2`: 32 ones then 32 of 1e30 for the file's 0.5 of a
    /// 128-wide head (the bytes of the file's tensor, which F32 1e30 spells `0x7149f2ca`), a
    /// factor of 1 (or none) turning all 64 pairs, 0.3 turning 19; a factor that turns no pair,
    /// is above 1, or is no number is refused by name. Mutants (each red at its line): `angles`
    /// of `partial * head_dim` without the halving (the 32), the tail 0 in place of 1e30 (the
    /// bytes), `floor` read as `round` (the 19).
    #[test]
    fn the_rope_factors_are_the_converters_values() {
        let config = json(CONFIG);
        let factors = |partial: Option<Json>| {
            let mut c = config.clone();
            match partial {
                Some(p) => c["rope_parameters"]["full_attention"]["partial_rotary_factor"] = p,
                None => {
                    c["rope_parameters"]["full_attention"]
                        .as_object_mut()
                        .unwrap()
                        .remove("partial_rotary_factor");
                }
            }
            Checkpoint::parse(&c).map(|c| c.rope_factors)
        };
        let ones = |n: usize| {
            let mut v = vec![1.0f32; n];
            v.resize(64, 1e30);
            v
        };
        assert_eq!(factors(Some(json!(0.5))).unwrap(), ones(32));
        assert_eq!(factors(Some(json!(1.0))).unwrap(), ones(64));
        assert_eq!(factors(None).unwrap(), ones(64));
        assert_eq!(factors(Some(json!(0.3))).unwrap(), ones(19));
        for bad in [
            json!(0.0),
            json!(0.01),
            json!(1.5),
            json!(-0.5),
            json!("half"),
        ] {
            let e = factors(Some(bad.clone())).unwrap_err().to_string();
            assert!(e.contains("partial_rotary_factor"), "{bad}: {e}");
        }
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let constants = checkpoint.constants().unwrap();
        assert_eq!(
            constants.keys().map(String::as_str).collect::<Vec<_>>(),
            ["rope_freqs.weight"]
        );
        let rope = &constants["rope_freqs.weight"];
        assert_eq!(rope.dims, [64]);
        let bytes = rope.bytes();
        assert_eq!(bytes.len(), 256);
        assert_eq!(bytes[..4], 1.0f32.to_le_bytes());
        assert_eq!(bytes[124..128], 1.0f32.to_le_bytes());
        assert_eq!(bytes[128..132], 0x7149_f2ca_u32.to_le_bytes());
        assert_eq!(bytes[252..], 0x7149_f2ca_u32.to_le_bytes());
    }

    /// The conv's taps, `[2048, 7, 1]` as MLX stores them, are the file's `[7, 2048]` rows with
    /// the tap order as stored (tap `k` of channel `c` is value `c * 7 + k`, the file's order and
    /// mlx-vlm's, which transposes only a `[channels, 1, kernel]` tensor): the planned dims, the
    /// move the part source runs, and a tensor stored in another layout or of another kernel
    /// width refused by name. Mutants (each red at its line): the move reversing each channel's
    /// taps (the bytes), the check dropped for `[2048, 1, 7]` (the refusal).
    #[test]
    fn the_conv_taps_keep_their_order_and_a_layout_that_is_not_mlx_s_is_refused() {
        let config = json(CONFIG);
        let entries = real_entries();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let planning = planning(&checkpoint, &config, &entries).unwrap();
        let planned = planning
            .planned
            .iter()
            .find(|p| p.decl.name == "blk.0.shortconv.conv.weight")
            .unwrap();
        assert_eq!(planned.decl.dims, [7, 2048]);
        let folder::Make::Tensor(m) = &planned.make else {
            panic!("a tensor of the folder");
        };
        let mover = checkpoint
            .mover(&planned.decl.name, &m.steps)
            .expect("the taps have a move");
        let values: Vec<f32> = (0..2048 * 7).map(|i| i as f32).collect();
        let mut taps = Normal {
            type_id: GgmlType::F32.as_u32(),
            dims: vec![1, 7, 2048],
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            scale: vec![],
        };
        let stored = taps.bytes.clone();
        mover(&mut taps).unwrap();
        assert_eq!(
            (taps.dims.clone(), taps.bytes.clone()),
            (vec![7, 2048], stored)
        );
        let mut flat = Normal {
            dims: vec![7, 2048],
            ..taps
        };
        let e = mover(&mut flat).unwrap_err();
        assert!(
            e.contains("blk.0.shortconv.conv.weight") && e.contains("one unit dim"),
            "{e}"
        );
        assert!(checkpoint.mover("blk.0.ffn_gate.weight", &[]).is_none());
        for shape in [[2048, 1, 7], [2048, 3, 1], [7, 2048, 1]] {
            let mut bad = entries.clone();
            let at = bad
                .iter()
                .position(|e| e.0 == format!("{LAYERS}.0.conv.conv.weight"))
                .unwrap();
            bad[at].2 = shape.to_vec();
            let e = refusal(&config, &bad);
            assert!(
                e.contains("0.conv.conv.weight") && e.contains("[channels, kernel, 1]"),
                "{shape:?}: {e}"
            );
        }
    }

    /// A folder is refused by name for: a tensor of the other mixer in a layer (attention in a
    /// conv layer, conv in an attention layer, by `layer_types`), a tensor the map does not know
    /// (a stray one, another model's prefix), a layer past the config's, a tensor whose dims the
    /// config does not give, and a tensor of the layer missing. Mutants (each red at its line):
    /// `holds` always `Ok`, the layer bound dropped, the dims check dropped, the presence check
    /// dropped.
    #[test]
    fn what_the_folder_does_not_hold_as_the_config_says_is_refused_by_name() {
        let config = json(CONFIG);
        let base = real_entries();
        let with = |extra: Entries| [base.clone(), extra].concat();
        let vector = |name: &str| (name.to_string(), Dtype::Bf16, vec![2048]);
        let e = refusal(
            &config,
            &with(linear(&format!("{LAYERS}.0.self_attn.q_proj"), 2048, 2048)),
        );
        assert!(
            e.contains("layers.0.self_attn.q_proj.weight")
                && e.contains("layer 0 is conv by layer_types")
                && e.contains("no attention tensor"),
            "{e}"
        );
        let e = refusal(
            &config,
            &with(linear(&format!("{LAYERS}.2.conv.in_proj"), 6144, 2048)),
        );
        assert!(
            e.contains("layer 2 is full_attention by layer_types") && e.contains("no conv tensor"),
            "{e}"
        );
        for (name, why) in [
            (format!("{LAYERS}.0.stray.weight"), "not an LFM3 tensor"),
            (
                "vision_tower.patch.weight".to_string(),
                "not an LFM3 tensor",
            ),
            (
                format!("{LAYERS}.24.operator_norm.weight"),
                "layer 24 of 24",
            ),
        ] {
            let e = refusal(&config, &with(vec![vector(&name)]));
            assert!(e.contains(&name) && e.contains(why), "{name}: {e}");
        }
        let mut narrow = base.clone();
        for e in &mut narrow {
            if e.0.starts_with(&format!("{LAYERS}.5.feed_forward.w1.")) {
                e.2[0] = 7000;
            }
        }
        let e = refusal(&config, &narrow);
        assert!(
            e.contains("5.feed_forward.w1.weight") && e.contains("holds dims"),
            "{e}"
        );
        let missing: Entries = base
            .iter()
            .filter(|e| !e.0.starts_with(&format!("{LAYERS}.2.self_attn.q_proj.")))
            .cloned()
            .collect();
        let e = refusal(&config, &missing);
        assert!(
            e.contains("blk.2.attn_q.weight") && e.contains("is missing from the folder"),
            "{e}"
        );
    }

    /// The head follows `tie_word_embeddings` both ways. The file's tied model (no head in the
    /// folder, the flag true) makes no `output.weight`; a folder that holds `lm_head` under a
    /// true flag and one that holds none under a false one are refused by name; an untied folder
    /// with its head makes `output.weight` at `[hidden, vocab]` as the embedding. Mutants (each
    /// red at its line): `check_head` returning `Ok` for either mismatch, the head mapped to
    /// `token_embd.weight`.
    #[test]
    fn the_head_follows_the_tie_flag_both_ways() {
        let config = json(CONFIG);
        let base = real_entries();
        let head = linear("language_model.lm_head", 128_000, 2048);
        let e = refusal(&config, &[base.clone(), head.clone()].concat());
        assert!(
            e.contains("language_model.lm_head.weight")
                && e.contains("tie_word_embeddings is true"),
            "{e}"
        );
        let mut untied = config.clone();
        untied["tie_word_embeddings"] = json!(false);
        let e = refusal(&untied, &base);
        assert!(
            e.contains("tie_word_embeddings is false") && e.contains("no head"),
            "{e}"
        );
        let both = [base, head].concat();
        let checkpoint = Checkpoint::parse(&untied).unwrap();
        let planning = planning(&checkpoint, &untied, &both).unwrap();
        let output = planning
            .planned
            .iter()
            .find(|p| p.decl.name == "output.weight")
            .expect("an untied model's head");
        assert_eq!(output.decl.dims, [2048, 128_000]);
        assert_eq!(output.decl.type_id, GgmlType::MLX_A4.as_u32());
        let folder::Make::Tensor(m) = &output.make else {
            panic!("a tensor of the folder");
        };
        assert_eq!(m.sources, ["language_model.lm_head.weight"]);
        assert_eq!(planning.planned.len(), 288);
        let tied = Checkpoint::parse(&config).unwrap();
        assert!(tied.tied && !checkpoint.keys.is_empty());
    }

    /// One row of the config's tables: the object, the key, a value accepted ("$" keeps the
    /// checkpoint's, `null` removes the key), a value refused, and what the refusal names.
    type Row = (String, String, Json, Json, String);

    fn config_rows() -> Vec<Row> {
        serde_json::from_str(
            r#"[
            ["", "architectures", ["Lfm3ForCausalLM"], ["Lfm3VlForConditionalGeneration"], "architectures"],
            ["", "model_type", "lfm3", "lfm2", "model_type"],
            ["", "quantization", "$", {"bits": 4}, "quantization"],
            ["", "quantization_config", "$", {"bits": 8}, "quantization"],
            ["", "num_hidden_layers", "$", 23, "layer_types"],
            ["", "max_position_embeddings", 131072, 0, "max_position_embeddings"],
            ["", "hidden_size", "$", 0, "hidden_size"],
            ["", "intermediate_size", "$", 0, "intermediate_size"],
            ["", "num_attention_heads", "$", 15, "num_attention_heads"],
            ["", "num_key_value_heads", "$", 5, "num_attention_heads"],
            ["", "head_dim", "$", 127, "head_dim"],
            ["", "norm_eps", 1e-6, 0, "norm_eps"],
            ["", "vocab_size", "$", 0, "vocab_size"],
            ["", "bos_token_id", 124894, "x", "bos_token_id"],
            ["", "eos_token_id", [124900, 124894], [], "eos_token_id"],
            ["", "pad_token_id", 124893, "x", "pad_token_id"],
            ["", "unk_token_id", 124893, "x", "unk_token_id"],
            ["", "sep_token_id", 124893, "x", "sep_token_id"],
            ["", "conv_L_cache", "$", 0, "conv_L_cache"],
            ["", "conv_bias", false, true, "conv_bias"],
            ["", "layer_types", "$", ["conv"], "layer_types"],
            ["", "rope_parameters", "$", 7, "rope_parameters"],
            ["", "sliding_window", "$", 0, "sliding_window"],
            ["", "use_sandwich_norm", true, false, "use_sandwich_norm"],
            ["", "tie_word_embeddings", true, "yes", "tie_word_embeddings"],
            ["", "vision_config", {}, {"hidden_size": 1024}, "vision_config"],
            ["", "auto_map", {}, 7, "auto_map"],
            ["", "dtype", "float32", 7, "dtype"],
            ["", "transformers_version", "5.0.0", 7, "transformers_version"],
            ["", "use_cache", false, "x", "use_cache"],
            ["", "do_sample", false, "x", "do_sample"],
            ["", "temperature", 0.6, "x", "temperature"],
            ["", "top_k", 40, "x", "top_k"],
            ["", "generation_config", {}, 7, "generation_config"],
            ["/rope_parameters", "full_attention", "$", 7, "full_attention"],
            ["/rope_parameters", "sliding_attention", "$", 7, "sliding_attention"],
            ["/rope_parameters/full_attention", "rope_type", "default", "yarn", "rope_type"],
            ["/rope_parameters/full_attention", "rope_theta", 500000.0, 0, "rope_theta"],
            ["/rope_parameters/full_attention", "partial_rotary_factor", 0.25, 1.5, "partial_rotary_factor"],
            ["/rope_parameters/sliding_attention", "rope_type", "default", "yarn", "rope_type"],
            ["/rope_parameters/sliding_attention", "rope_theta", 20000, 0, "rope_theta"],
            ["/rope_parameters/sliding_attention", "partial_rotary_factor", 1.0, 0.5, "partial_rotary_factor"]
        ]"#,
        )
        .unwrap()
    }

    /// The config's rows: each key accepts its value and refuses another by name, and a key no
    /// table names is refused in each object. A layer kind no table reads, a `layer_types` of
    /// another length, and a `sliding_attention` rope that a sliding layer needs and the config
    /// lacks, are refused by name. Mutant (red at the first line of each): a table's rule
    /// loosened to `Read`, the key's `Only` or `Need` dropped.
    #[test]
    fn the_config_accepts_its_values_and_refuses_others_by_name() {
        let base = json(CONFIG);
        let run = |c: &Json| -> Result<(), String> {
            Checkpoint::parse(c).map_err(|e| e.to_string())?;
            Config::parse(c, None).map(drop).map_err(|e| e.to_string())
        };
        let with = |at: &str, key: &str, value: &Json| {
            let mut c = base.clone();
            let object = c.pointer_mut(at).unwrap().as_object_mut().unwrap();
            match value {
                Json::String(s) if s == "$" => {}
                Json::Null => {
                    object.remove(key);
                }
                v => {
                    object.insert(key.to_string(), v.clone());
                }
            }
            c
        };
        assert_eq!(run(&base), Ok(()));
        let mut missed = vec![];
        for at in [
            "",
            "/rope_parameters",
            "/rope_parameters/full_attention",
            "/rope_parameters/sliding_attention",
        ] {
            match run(&with(at, "zzz", &json!(1))) {
                Err(e) if e.contains("zzz") => {}
                other => missed.push(format!("{at} zzz: {other:?}")),
            }
        }
        for (at, key, allowed, refused, what) in &config_rows() {
            if let Err(e) = run(&with(at, key, allowed)) {
                missed.push(format!("{at} {key}={allowed}: refused: {e}"));
            }
            match run(&with(at, key, refused)) {
                Err(e) if e.contains(what.as_str()) => {}
                other => missed.push(format!("{at} {key}={refused}: {other:?}")),
            }
        }
        let mut c = base.clone();
        c["layer_types"][0] = json!("conv2d");
        match run(&c) {
            Err(e) if e.contains("layer_types") && e.contains("layer 0") => {}
            other => missed.push(format!("a layer kind no table reads: {other:?}")),
        }
        let mut c = base.clone();
        c["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("sliding_attention");
        match run(&c) {
            Err(e) if e.contains("sliding_attention") => {}
            other => missed.push(format!("sliding layers with no rope: {other:?}")),
        }
        let mut c = base.clone();
        c["layer_types"] = json!(
            base["layer_types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| if t == "sliding_attention" {
                    json!("full_attention")
                } else {
                    t.clone()
                })
                .collect::<Vec<_>>()
        );
        c["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("sliding_attention");
        let c = Checkpoint::parse(&c).unwrap();
        assert!(
            !c.keys.iter().any(|(k, _)| k == "rope.freq_base_swa"),
            "a model with no sliding layer and no sliding rope writes no base for it"
        );
        let mut c = base.clone();
        c.as_object_mut().unwrap().remove("rope_parameters");
        match run(&c) {
            Err(e) if e.contains("rope_parameters") => {}
            other => missed.push(format!("no rope_parameters: {other:?}")),
        }
        assert!(
            missed.is_empty(),
            "accepted or unnamed:\n{}",
            missed.join("\n")
        );
    }

    /// Every key of the map's tables has a row in [`config_rows`], so a key added to a table is
    /// accepted and refused by a gate before it is read.
    #[test]
    fn every_key_of_the_tables_has_a_row() {
        let tables: [(&str, &Fields); 4] = [
            ("", ROOT),
            ("/rope_parameters", ROPE),
            ("/rope_parameters/full_attention", ROPE_FULL),
            ("/rope_parameters/sliding_attention", ROPE_SLIDING),
        ];
        let rows = config_rows();
        let missing: Vec<String> = tables
            .iter()
            .flat_map(|(at, table)| table.iter().map(move |(key, _)| (*at, *key)))
            .filter(|&(at, key)| {
                !rows
                    .iter()
                    .any(|r| (r.0.as_str(), r.1.as_str()) == (at, key))
            })
            .map(|(at, key)| format!("{at} {key}"))
            .collect();
        assert!(missing.is_empty(), "no row: {missing:?}");
    }
}
