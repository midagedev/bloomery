//! Llama's Hugging Face checkpoint (`llama`: Llama-3.1-8B-Instruct and its siblings) as MLX's
//! convert stores it, in the llama names, layouts and keys llama.cpp's converter (`LlamaModel`)
//! writes into a GGUF. Every layer is grouped-query attention with no q/k norms over a dense
//! SwiGLU block; the trunk's names, keys and checks are [`super::trunk`]'s.
//!
//! The converter moves two layouts, and so does the map. A query or key projection stores each
//! head's rows as HF's rotary does, the first half of a head's values paired with the second
//! (`rotate_half`); the file's rope pairs neighbours, so the map interleaves the rows of
//! `q_proj` and `k_proj` head by head ([`Interleave`]), as the converter's `permute` does. The
//! rope's frequency factors, which the converter writes as the tensor `rope_freqs.weight` for a
//! `rope_scaling` of type `llama3`, are made of that block of the config ([`llama3_factors`],
//! [`FolderMap::constants`]); a `rope_scaling` of another type is refused. The head is the
//! folder's `lm_head` when it holds one and the embedding when `tie_word_embeddings` says so, and
//! a folder whose flag and head disagree is refused ([`trunk::check_head`]).

use std::collections::BTreeMap;

use gguf::Value;
use hf::modelopt::{Normal, Plain, Use};
use hf::schema::{Fields, Rule::*, Type};
use model::arch::names as common;
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{Constant, Error, FolderMap, Mapped, Mapping, Origin, VocabSpec, refuse};

const ARCHITECTURE: &str = "llama";

/// Every layer's block is one dense SwiGLU of `intermediate_size`.
const FFN: trunk::Ffn = trunk::Ffn::Dense;

/// The output head a folder of an untied model holds.
const HEAD: &str = "lm_head.weight";

/// `config.json`'s keys beside [`trunk::KEYS`] and the dense block's: the checkpoint's class and
/// type, MLX's `quantization` block (which `hf::quant_config` reads, with its verbatim copy), the
/// rope's base and scaling, and the biases and tensor-parallel slicing that this map's tensors
/// have none of. A table's first rule for a key decides, so the head rule is this one's, not
/// [`trunk::KEYS`]'s.
const ROOT: &Fields = &[
    trunk::HEAD_EITHER,
    ("architectures", Need(r#"[["LlamaForCausalLM"]]"#)),
    ("model_type", Need(r#"["llama"]"#)),
    ("quantization", Read),
    ("quantization_config", Read),
    ("rope_theta", Read),
    ("rope_scaling", Read),
    ("mlp_bias", Only("[false]")),
    // HF's tensor-parallel slicing of a linear, which gives the same values.
    ("pretraining_tp", Only("[1]")),
    (
        "torch_dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
];

/// `rope_scaling`: the one scaling the map reads, with every number it uses spelled out.
const ROPE_SCALING: &Fields = &[
    ("rope_type", Need(r#"["llama3"]"#)),
    ("factor", Read),
    ("low_freq_factor", Read),
    ("high_freq_factor", Read),
    ("original_max_position_embeddings", Read),
];

/// What `rope_scaling` of type `llama3` says: the frequencies of wavelength past
/// `original_context / low_freq_factor` are divided by `factor`, those of wavelength under
/// `original_context / high_freq_factor` are kept, and the ones between slide from one to the
/// other.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Llama3 {
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_context: f64,
}

/// The rope's frequency factor of each of `head_dim / 2` pairs (what llama.cpp's converter writes
/// as `rope_freqs.weight`, and mlx-lm's `Llama3RoPE` divides by), for the base `base` and the
/// scaling `s`. Pair `i` turns at `base^(-2i / head_dim)` and has the wavelength `2π · base^(2i /
/// head_dim)`: a wavelength under `original_context / high_freq_factor` has the factor 1, one past
/// `original_context / low_freq_factor` has `factor`, and one between has
/// `1 / ((1 - t) / factor + t)` for `t = (original_context / wavelength - low_freq_factor) /
/// (high_freq_factor - low_freq_factor)`. Computed in f64 and rounded once to f32; the converter
/// computes in f32, so a factor between the two ends may differ from its file's in the last bits.
fn llama3_factors(head_dim: u32, base: f64, s: Llama3) -> Vec<f32> {
    let low_wavelength = s.original_context / s.low_freq_factor;
    let high_wavelength = s.original_context / s.high_freq_factor;
    (0..head_dim / 2)
        .map(|i| {
            let wavelength =
                std::f64::consts::TAU * base.powf(f64::from(2 * i) / f64::from(head_dim));
            let factor = if wavelength < high_wavelength {
                1.0
            } else if wavelength > low_wavelength {
                s.factor
            } else {
                let t = (s.original_context / wavelength - s.low_freq_factor)
                    / (s.high_freq_factor - s.low_freq_factor);
                1.0 / ((1.0 - t) / s.factor + t)
            };
            factor as f32
        })
        .collect()
}

/// The rows of a query or key projection moved from HF's layout to the file's: the head's rows
/// stored as the first half then the second half of its values (`rotate_half` pairs `i` and
/// `i + head_dim / 2`) become pairs of neighbours, row `i` then row `i + head_dim / 2` of each
/// head at rows `2i` and `2i + 1`. `heads` is the heads the projection holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interleave {
    heads: usize,
}

impl Interleave {
    /// The rows of the tensor `t` (ggml dims `[k, heads · head_dim]`, each row whole in the
    /// type's blocks) moved, head by head. A tensor of another rank or row count, or of rows that
    /// are not whole, is refused with why (the caller names the tensor), and `t` is then as it
    /// was.
    pub fn apply(&self, t: &mut Normal, head_dim: usize) -> Result<(), String> {
        let [_, rows] = t.dims[..] else {
            return Err(format!("{:?} is not a matrix", t.dims));
        };
        let rows = usize::try_from(rows).map_err(|_| format!("{rows} rows do not fit usize"))?;
        if head_dim == 0 || !head_dim.is_multiple_of(2) || rows != self.heads * head_dim {
            return Err(format!(
                "{rows} rows are not {} heads of {head_dim} values, an even width",
                self.heads
            ));
        }
        if !t.bytes.len().is_multiple_of(rows) {
            return Err(format!(
                "{} bytes are not whole rows of {rows}",
                t.bytes.len()
            ));
        }
        let row = t.bytes.len() / rows;
        let half = head_dim / 2;
        let mut moved = vec![0u8; t.bytes.len()];
        for head in 0..self.heads {
            for i in 0..half {
                for (to, from) in [(2 * i, i), (2 * i + 1, half + i)] {
                    let (to, from) = ((head * head_dim + to) * row, (head * head_dim + from) * row);
                    moved[to..to + row].copy_from_slice(&t.bytes[from..from + row]);
                }
            }
        }
        t.bytes = moved;
        Ok(())
    }
}

/// The program's tensor of a checkpoint tensor: its name, the role it is read in, and the move
/// its rows take.
struct Target {
    to: String,
    plain: Plain,
    step: Option<Interleave>,
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    layers: usize,
    hidden: u64,
    ffn: u64,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tied: bool,
    /// The rope's frequency factor of each pair, when `rope_scaling` has any.
    rope_factors: Option<Vec<f32>>,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

impl Checkpoint {
    /// The rope's factors of `rope_scaling`, `None` where the config has none (absent or null).
    fn factors(
        root: &Section<'_, Error>,
        head_dim: u32,
        base: f32,
    ) -> Result<Option<Vec<f32>>, Error> {
        let Some(scaling) = root.value.get("rope_scaling").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        let c = Section::<Error>::new(scaling, "config.json.rope_scaling")?;
        c.check(&[ROPE_SCALING])?;
        let s = Llama3 {
            factor: f64::from(c.number("factor")?),
            low_freq_factor: f64::from(c.number("low_freq_factor")?),
            high_freq_factor: f64::from(c.number("high_freq_factor")?),
            original_context: f64::from(c.count("original_max_position_embeddings")?),
        };
        if s.high_freq_factor <= s.low_freq_factor {
            return Err(c.bad("high_freq_factor", "must exceed low_freq_factor"));
        }
        Ok(Some(llama3_factors(head_dim, f64::from(base), s)))
    }
}

impl FolderMap for Checkpoint {
    /// The query and key projections' rows are the one move.
    type Step = Interleave;

    const MODEL_TYPES: &'static [&'static str] = &["llama"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT, trunk::KEYS, FFN.keys()])?;
        let mut counts = BTreeMap::new();
        for &field in trunk::COUNTS.iter().chain(FFN.counts()) {
            // A head's width is `hidden_size / num_attention_heads` where the config does not
            // spell it.
            if field != "head_dim" || root.value.get(field).is_some() {
                counts.insert(field, root.count(field)?);
            }
        }
        if !counts.contains_key("head_dim") {
            let (hidden, heads) = (counts["hidden_size"], counts["num_attention_heads"]);
            if !hidden.is_multiple_of(heads) {
                return Err(root.bad(
                    "hidden_size",
                    "must be a multiple of num_attention_heads where head_dim is absent",
                ));
            }
            counts.insert("head_dim", hidden / heads);
        }
        let n = |key: &str| counts[key];
        trunk::check_counts(&root, FFN, n)?;
        if !n("head_dim").is_multiple_of(2) {
            return Err(root.bad("head_dim", "must be even: the rope turns pairs"));
        }
        let wide = |key: &str| u64::from(counts[key]);
        let rope_base = root.number("rope_theta")?;
        let mut keys = trunk::metadata(n, FFN, rope_base, root.number("rms_norm_eps")?);
        keys.push((
            "rope.dimension_count".to_string(),
            Value::U32(n("head_dim")),
        ));
        Ok(Self {
            layers: n("num_hidden_layers") as usize,
            hidden: wide("hidden_size"),
            ffn: wide("intermediate_size"),
            heads: n("num_attention_heads") as usize,
            kv_heads: n("num_key_value_heads") as usize,
            head_dim: n("head_dim") as usize,
            tied: trunk::tied(&root)?,
            rope_factors: Self::factors(&root, n("head_dim"), rope_base)?,
            // Llama 3's regex split, the row llama.cpp names llama-bpe.
            vocab: hf::vocab::Config::parse(config)?.with_pre("llama-bpe")?,
            keys,
        })
    }

    fn module_class(&self, module: &str) -> Option<hf::quant_config::ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Interleave>, Error> {
        let head = tensors.iter().map(|&(name, _)| name).find(|&n| n == HEAD);
        trunk::check_head(self.tied, head).map_err(Error::Refused)?;
        let mut out: BTreeMap<String, Mapped<Interleave>> = BTreeMap::new();
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let Target { to, plain, step } = self.tensor(name)?;
            let mapped = Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(plain),
                steps: step.into_iter().collect(),
                dims: shape.iter().rev().map(|&d| d as u64).collect(),
                origin: Origin::Made,
            };
            if let Some(prev) = out.insert(to, mapped) {
                return Err(refuse(
                    &prev.sources[0],
                    "maps to a llama name another tensor has",
                ));
            }
        }
        self.complete(&out)?;
        Ok(Mapping {
            tensors: out,
            skipped: BTreeMap::new(),
        })
    }

    fn mover(&self, name: &str, steps: &[Interleave]) -> Option<crate::part::Step> {
        steps.first().map(|step| {
            let (step, name, head_dim) = (step.clone(), name.to_string(), self.head_dim);
            Box::new(move |n: &mut Normal| {
                step.apply(n, head_dim)
                    .map_err(|why| format!("{name}: {why}"))
            }) as crate::part::Step
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
        let mut keys = vec![(
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(ARCHITECTURE.into()),
        )];
        keys.extend(
            self.keys
                .iter()
                .map(|(k, v)| (format!("{ARCHITECTURE}.{k}"), v.clone())),
        );
        Ok(keys)
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }

    fn constants(&self) -> Result<BTreeMap<String, Constant>, Error> {
        Ok(self
            .rope_factors
            .iter()
            .map(|values| {
                (
                    common::rope_freqs(),
                    Constant {
                        dims: vec![values.len() as u64],
                        values: values.clone(),
                    },
                )
            })
            .collect())
    }
}

impl Checkpoint {
    /// The program's tensor of the checkpoint tensor `name` and the role it is read in; a tensor
    /// no Llama folder of MLX's has is refused with why.
    fn tensor(&self, name: &str) -> Result<Target, Error> {
        let bare = |(to, plain)| Target {
            to,
            plain,
            step: None,
        };
        if name == HEAD {
            return Ok(bare((common::output(), Plain::Q8_0)));
        }
        let short = name
            .strip_prefix("model.")
            .ok_or_else(|| refuse(name, "not a Llama tensor"))?;
        if let Some(global) = trunk::global(short) {
            return Ok(bare(global));
        }
        let (l, rest) = short
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            .ok_or_else(|| refuse(name, "not a Llama tensor"))?;
        if l >= self.layers {
            return Err(refuse(name, format!("layer {l} of {}", self.layers)));
        }
        if let Some((to, plain)) = trunk::projections(l, rest) {
            let heads = match rest {
                "self_attn.q_proj.weight" => Some(self.heads),
                "self_attn.k_proj.weight" => Some(self.kv_heads),
                _ => None,
            };
            return Ok(Target {
                to,
                plain,
                step: heads.map(|heads| Interleave { heads }),
            });
        }
        trunk::block_norms(l, rest, common::ffn_norm)
            .or_else(|| trunk::dense_ffn(l, rest))
            .map(bare)
            .ok_or_else(|| refuse(name, "not a Llama tensor"))
    }

    /// The ggml dims of the file's tensor `stem` (the name past `blk.N.`, or a model-level
    /// name) that the config's geometry gives.
    fn dims(&self, stem: &str) -> Vec<u64> {
        let (hidden, ffn) = (self.hidden, self.ffn);
        let (query, key) = (
            self.heads as u64 * self.head_dim as u64,
            self.kv_heads as u64 * self.head_dim as u64,
        );
        match stem {
            "token_embd.weight" | "output.weight" => vec![hidden, u64::from(self.vocab.size())],
            "output_norm.weight" | "attn_norm.weight" | "ffn_norm.weight" => vec![hidden],
            "attn_q.weight" => vec![hidden, query],
            "attn_k.weight" | "attn_v.weight" => vec![hidden, key],
            "attn_output.weight" => vec![query, hidden],
            "ffn_gate.weight" | "ffn_up.weight" => vec![hidden, ffn],
            "ffn_down.weight" => vec![ffn, hidden],
            other => unreachable!("{other} is a tensor of no Llama file"),
        }
    }

    /// Every tensor the file holds is mapped, at the dims the config gives: the embedding, the
    /// output norm, the head of an untied model, and each layer's nine.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Interleave>>) -> Result<(), Error> {
        const LAYER: [&str; 9] = [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        let mut globals = vec!["token_embd.weight", "output_norm.weight"];
        if !self.tied {
            globals.push("output.weight");
        }
        let globals = globals.into_iter().map(|stem| (stem.to_string(), stem));
        let layers = (0..self.layers).flat_map(|l| {
            LAYER
                .into_iter()
                .map(move |stem| (common::blk(l, stem), stem))
        });
        for (name, stem) in globals.chain(layers) {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing from the folder"))?;
            let want = self.dims(stem);
            if m.dims != want {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {want:?}", m.dims),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use serde_json::json;

    use super::*;
    use crate::folder;
    use crate::test_support::{
        Stored, bf16_tensor, entries, json, mlx_linear, temp_dir, tokenizer, write_folder,
    };

    const CONFIG: &str = "quant-configs/mlx-community__Llama-3.1-8B-Instruct-4bit__config.json.gz";
    const HEADER: &str = "checkpoint-headers/llama-3.1-8b-instruct-mlx-4bit-model.header.json.gz";

    /// Llama-3.1-8B's `rope_scaling`: the frequencies of a wavelength past 8192 divided by 8, those
    /// under 2048 kept.
    const LLAMA31: Llama3 = Llama3 {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_context: 8192.0,
    };

    /// Llama-3.1-8B's rope factors (head 128, base 500000) by hand: pair `i` has the wavelength
    /// `2π · 500000^(i / 64)`, which is under 2048 for `i` up to 28 (the factor 1) and past 8192
    /// from `i` 35 (the factor 8), and the six pairs between slide from one to the other. The
    /// slid values are the rule's `1 / ((1 - t) / 8 + t)` at `t = (8192 / wavelength - 1) / 3`,
    /// evaluated outside the code (Python's f64, rounded once to f32), and pair 32, whose
    /// wavelength is `2π · √500000`, is derived below from that closed form. mlx-lm's
    /// `Llama3RoPE` and llama.cpp's converter agree with them to 2e-16 in f64. Mutants (each red
    /// here): the slid factor's `t` halved (the six slid values), the two ends swapped (the first
    /// assert).
    #[test]
    fn the_llama3_factors_of_the_8b_are_the_rule_by_hand() {
        let factors = llama3_factors(128, 500_000.0, LLAMA31);
        assert_eq!(factors.len(), 64);
        assert!(
            factors[..29].iter().all(|&f| f == 1.0),
            "{:?}",
            &factors[..29]
        );
        assert!(
            factors[35..].iter().all(|&f| f == 8.0),
            "{:?}",
            &factors[35..]
        );
        for (i, want) in [
            (29, 1.207_483_9_f32),
            (30, 1.553_414_6),
            (31, 2.026_313_3),
            (32, 2.694_529_8),
            (33, 3.684_252_5),
            (34, 5.257_326_6),
        ] {
            let got = factors[i];
            assert!(
                (got - want).abs() <= want * 1e-6,
                "pair {i}: {got} for {want}"
            );
        }
        // Pair 32: the wavelength 2π√500000 = 4443.1; t = (8192 / 4443.1 - 1) / 3 = 0.28124.
        let wavelength = std::f64::consts::TAU * 500_000f64.sqrt();
        let t = (8192.0 / wavelength - 1.0) / 3.0;
        let closed = 1.0 / ((1.0 - t) / 8.0 + t);
        assert!((f64::from(factors[32]) - closed).abs() < 1e-6, "{closed}");
        // A narrow head: the same rule over 8 pairs.
        let small = llama3_factors(16, 500_000.0, LLAMA31);
        assert_eq!(small[..4], [1.0; 4]);
        assert!((small[4] - 2.694_529_8).abs() < 1e-6, "{}", small[4]);
        assert_eq!(small[5..], [8.0; 3]);
    }

    /// The query and key rows of a head move from HF's halves to the file's neighbours, head by
    /// head and whole rows at a time: with two heads of four values the rows 0 1 2 3 | 4 5 6 7
    /// become 0 2 1 3 | 4 6 5 7. A tensor of another rank, another row count than the heads give,
    /// an odd head width or rows that are not whole is refused by name and left as it was.
    /// Mutant: the second value of each pair taken from the first half too.
    #[test]
    fn the_rows_of_a_head_move_from_halves_to_neighbours() {
        let rows = |n: usize| -> Normal {
            Normal {
                type_id: GgmlType::MLX_A4.as_u32(),
                dims: vec![3, n as u64],
                bytes: (0..n).flat_map(|r| [r as u8; 5]).collect(),
                scale: vec![],
            }
        };
        let mut t = rows(8);
        Interleave { heads: 2 }.apply(&mut t, 4).unwrap();
        let order: Vec<u8> = t.bytes.chunks(5).map(|r| r[0]).collect();
        assert_eq!(order, [0, 2, 1, 3, 4, 6, 5, 7]);
        assert!(t.bytes.chunks(5).all(|r| r.iter().all(|&b| b == r[0])));
        assert_eq!(t.dims, [3, 8]);
        for (what, mut t, heads, head_dim, want) in [
            (
                "three heads of four",
                rows(8),
                3,
                4,
                "8 rows are not 3 heads of 4",
            ),
            ("an odd head", rows(9), 3, 3, "an even width"),
            (
                "a tensor of a rank",
                Normal {
                    dims: vec![3],
                    ..rows(8)
                },
                2,
                4,
                "is not a matrix",
            ),
            (
                "rows that are not whole",
                Normal {
                    bytes: vec![0; 41],
                    ..rows(8)
                },
                2,
                4,
                "bytes are not whole rows",
            ),
        ] {
            let before = (t.bytes.clone(), t.dims.clone());
            let e = Interleave { heads }.apply(&mut t, head_dim).unwrap_err();
            assert!(e.contains(want), "{what}: {e}");
            assert_eq!((t.bytes, t.dims), before, "{what}: left as it was");
        }
    }

    /// The real 8B folder's tensors, as the map is handed them: each quantized weight at its
    /// unpacked shape (the packed words hold eight 4-bit values), each norm as stored; the scales
    /// and biases are the weights' companions.
    fn real_tensors() -> Vec<(String, Vec<usize>)> {
        let mut header = json(HEADER);
        header.as_object_mut().unwrap().remove("__metadata__");
        entries(&header)
            .into_iter()
            .filter(|(name, _, _)| name.ends_with(".weight"))
            .map(|(name, dtype, mut shape)| {
                if dtype == safetensors::Dtype::U32 {
                    *shape.last_mut().unwrap() *= 8;
                }
                (name, shape)
            })
            .collect()
    }

    fn map_of(
        checkpoint: &Checkpoint,
        tensors: &[(String, Vec<usize>)],
    ) -> Result<Mapping<Interleave>, String> {
        let borrowed: Vec<(&str, &[usize])> = tensors
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        checkpoint.map(&borrowed).map_err(|e| e.to_string())
    }

    /// The real 8B folder's config and tensor names and shapes map to the file's: 291 tensors, the
    /// embedding, output norm and head and nine a layer, at the ggml dims of the config
    /// (`head_dim` absent from the config, so 4096 / 32); only the query and key projections move
    /// (32 and 8 heads), the linears are read as Q8_0 roles (MLX's blocks when stored so) and the
    /// norms as F32; the header's keys are the file's `llama.*` keys, `rope.dimension_count` the
    /// whole head, and the one constant is `rope_freqs.weight`, 64 factors. Mutants (each red
    /// here): `k_proj` moved by the query's heads, `v_proj` moved.
    #[test]
    fn the_real_8b_folder_maps_to_the_files_tensors_and_keys() {
        let config = json(CONFIG);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let mapping = map_of(&checkpoint, &real_tensors()).unwrap();
        assert_eq!(mapping.tensors.len(), 291);
        assert!(mapping.skipped.is_empty());
        let get = |name: &str| &mapping.tensors[name];
        assert_eq!(get("token_embd.weight").dims, [4096, 128_256]);
        assert_eq!(get("output.weight").dims, [4096, 128_256]);
        assert_eq!(get("output.weight").sources, ["lm_head.weight"]);
        assert_eq!(get("output_norm.weight").dims, [4096]);
        assert_eq!(get("output_norm.weight").role, Use::Matrix(Plain::F32));
        for (name, dims, role) in [
            ("attn_norm.weight", vec![4096], Plain::F32),
            ("attn_q.weight", vec![4096, 4096], Plain::Q8_0),
            ("attn_k.weight", vec![4096, 1024], Plain::Q8_0),
            ("attn_v.weight", vec![4096, 1024], Plain::Q8_0),
            ("attn_output.weight", vec![4096, 4096], Plain::Q8_0),
            ("ffn_norm.weight", vec![4096], Plain::F32),
            ("ffn_gate.weight", vec![4096, 14_336], Plain::Q8_0),
            ("ffn_up.weight", vec![4096, 14_336], Plain::Q8_0),
            ("ffn_down.weight", vec![14_336, 4096], Plain::Q8_0),
        ] {
            for l in [0, 31] {
                let m = get(&format!("blk.{l}.{name}"));
                assert_eq!((&m.dims, m.role), (&dims, Use::Matrix(role)), "{l} {name}");
                assert_eq!(m.origin, Origin::Made);
            }
        }
        for l in 0..32 {
            let steps = |name: &str| get(&format!("blk.{l}.{name}")).steps.clone();
            assert_eq!(steps("attn_q.weight"), [Interleave { heads: 32 }], "{l}");
            assert_eq!(steps("attn_k.weight"), [Interleave { heads: 8 }], "{l}");
            for name in [
                "attn_v.weight",
                "attn_output.weight",
                "attn_norm.weight",
                "ffn_down.weight",
            ] {
                assert!(steps(name).is_empty(), "{l} {name}");
            }
        }
        assert!(get("token_embd.weight").steps.is_empty() && get("output.weight").steps.is_empty());
        let keys: BTreeMap<&str, &Value> = checkpoint
            .keys
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let u32s = |key: &str| keys[key].clone();
        assert_eq!(keys.len(), 11);
        for (key, want) in [
            ("block_count", 32),
            ("context_length", 131_072),
            ("embedding_length", 4096),
            ("attention.head_count", 32),
            ("attention.head_count_kv", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("feed_forward_length", 14_336),
            ("rope.dimension_count", 128),
        ] {
            assert_eq!(u32s(key), Value::U32(want), "{key}");
        }
        assert_eq!(u32s("rope.freq_base"), Value::F32(500_000.0));
        assert_eq!(u32s("attention.layer_norm_rms_epsilon"), Value::F32(1e-5));
        let constants = checkpoint.constants().unwrap();
        assert_eq!(constants.keys().collect::<Vec<_>>(), ["rope_freqs.weight"]);
        let table = &constants["rope_freqs.weight"];
        assert_eq!(table.dims, [64]);
        assert_eq!(table.values, llama3_factors(128, 500_000.0, LLAMA31));
        assert_eq!(checkpoint.architecture(), "llama");
    }

    /// The config's every key is read or refused by name: a scaling of another type, a head of an
    /// odd width, a `head_dim` that does not divide the hidden width where it is absent, a bias,
    /// a sliced linear, another class, and a key no table names; a `rope_scaling` that is null or
    /// absent is the plain rope and makes no constant, and one that omits a number or has no slope
    /// between its two ends is refused.
    #[test]
    fn a_config_the_map_does_not_read_is_refused_by_name() {
        let real = json(CONFIG);
        let parse = |edit: &dyn Fn(&mut serde_json::Map<String, Json>)| {
            let mut config = real.clone();
            edit(config.as_object_mut().unwrap());
            Checkpoint::parse(&config).map_err(|e| e.to_string())
        };
        let scaling = |edit: &dyn Fn(&mut serde_json::Map<String, Json>)| {
            parse(&|c| edit(c["rope_scaling"].as_object_mut().unwrap()))
        };
        for (key, value, want) in [
            ("attention_bias", json!(true), "supported: [false]"),
            ("mlp_bias", json!(true), "supported: [false]"),
            ("pretraining_tp", json!(2), "supported: [1]"),
            ("hidden_act", json!("gelu"), "supported: [\"silu\"]"),
            ("model_type", json!("mistral"), "supported: [\"llama\"]"),
            (
                "architectures",
                json!(["MistralForCausalLM"]),
                "is unsupported",
            ),
            (
                "rope_parameters",
                json!({}),
                "is not a key this converter reads",
            ),
            (
                "num_key_value_heads",
                json!(5),
                "must group by num_key_value_heads",
            ),
            ("head_dim", json!(127), "must be even"),
            ("intermediate_size", json!(0), "must be a positive u32"),
        ] {
            let e = parse(&|c| {
                c.insert(key.to_string(), value.clone());
            })
            .unwrap_err();
            assert!(e.contains(key) && e.contains(want), "{key}: {e}");
        }
        let e = parse(&|c| {
            c.insert("hidden_size".into(), json!(4100));
        })
        .unwrap_err();
        assert!(e.contains("multiple of num_attention_heads"), "{e}");
        let e = parse(&|c| {
            c.remove("rope_theta");
        })
        .unwrap_err();
        assert!(e.contains("config.json.rope_theta is missing"), "{e}");
        // A scaling of another type, and one with a number left out or no slope.
        for rope_type in ["yarn", "linear", "dynamic", "longrope"] {
            let e = scaling(&|s| {
                s.insert("rope_type".into(), json!(rope_type));
            })
            .unwrap_err();
            assert!(
                e.contains("rope_type") && e.contains("supported: [\"llama3\"]"),
                "{e}"
            );
        }
        for key in [
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ] {
            let e = scaling(&|s| {
                s.remove(key);
            })
            .unwrap_err();
            assert!(
                e.contains(&format!("rope_scaling.{key} is missing")),
                "{key}: {e}"
            );
        }
        let e = scaling(&|s| {
            s.insert("high_freq_factor".into(), json!(1.0));
        })
        .unwrap_err();
        assert!(
            e.contains("high_freq_factor must exceed low_freq_factor"),
            "{e}"
        );
        let e = scaling(&|s| {
            s.insert("type".into(), json!("llama3"));
        })
        .unwrap_err();
        assert!(e.contains("rope_scaling.type is not a key"), "{e}");
        // No scaling is the plain rope: no constant, and the header's other keys are the same.
        for edit in [
            &(|c: &mut serde_json::Map<String, Json>| {
                c.insert("rope_scaling".into(), Json::Null);
            }) as &dyn Fn(&mut serde_json::Map<String, Json>),
            &|c| {
                c.remove("rope_scaling");
            },
        ] {
            let plain = parse(edit).unwrap();
            assert!(plain.constants().unwrap().is_empty());
        }
        // A config that spells `head_dim` is read as it says.
        let wide = parse(&|c| {
            c.insert("head_dim".into(), json!(256));
        })
        .unwrap();
        assert_eq!(wide.head_dim, 256);
    }

    /// A folder's tensor names and dims.
    type Tensors = Vec<(String, Vec<usize>)>;

    /// The map refuses, by name, a tensor no Llama folder of MLX's has (a head norm, a bias, an
    /// expert), a layer past the config's, a tensor the file needs that the folder lacks, and a
    /// tensor of other dims than the config gives.
    #[test]
    fn the_map_refuses_what_it_does_not_read_by_name() {
        let checkpoint = Checkpoint::parse(&json(CONFIG)).unwrap();
        let base = real_tensors();
        let refuse = |edit: &dyn Fn(&mut Tensors)| {
            let mut t = base.clone();
            edit(&mut t);
            map_of(&checkpoint, &t).err().expect("refused")
        };
        let l = "model.layers";
        for (extra, shape, want) in [
            ("self_attn.q_norm.weight", vec![128], "not a Llama tensor"),
            ("self_attn.q_proj.bias", vec![4096], "not a Llama tensor"),
            (
                "self_attn.rotary_emb.inv_freq",
                vec![64],
                "not a Llama tensor",
            ),
            ("mlp.gate.weight", vec![8, 4096], "not a Llama tensor"),
            (
                "mlp.experts.0.gate_proj.weight",
                vec![14_336, 4096],
                "not a Llama tensor",
            ),
        ] {
            let e = refuse(&|t| t.push((format!("{l}.3.{extra}"), shape.clone())));
            assert!(
                e.contains(&format!("{l}.3.{extra}")) && e.contains(want),
                "{e}"
            );
        }
        let e = refuse(&|t| t.push((format!("{l}.32.input_layernorm.weight"), vec![4096])));
        assert!(e.contains("layer 32 of 32"), "{e}");
        let e = refuse(&|t| t.push(("visual.weight".to_string(), vec![8])));
        assert!(
            e.contains("visual.weight") && e.contains("not a Llama tensor"),
            "{e}"
        );
        let e = refuse(&|t| t.retain(|(n, _)| n != &format!("{l}.7.mlp.up_proj.weight")));
        assert!(
            e.contains("blk.7.ffn_up.weight") && e.contains("is missing from the folder"),
            "{e}"
        );
        let e = refuse(&|t| {
            t.iter_mut()
                .find(|(n, _)| n == &format!("{l}.2.self_attn.k_proj.weight"))
                .unwrap()
                .1[0] /= 2;
        });
        assert!(
            e.contains(&format!("{l}.2.self_attn.k_proj.weight"))
                && e.contains("holds dims [4096, 512] where the config gives [4096, 1024]"),
            "{e}"
        );
        let e = refuse(&|t| t.retain(|(n, _)| n != "lm_head.weight"));
        assert!(
            e.contains("tie_word_embeddings is false") && e.contains("no head"),
            "{e}"
        );
    }

    /// A tied head is the embedding: a folder with `tie_word_embeddings` true and no `lm_head`
    /// maps without `output.weight`, and the flag and the head must agree either way.
    #[test]
    fn a_tied_folder_has_no_head_beside_the_embedding() {
        let mut config = json(CONFIG);
        config["tie_word_embeddings"] = json!(true);
        let tied = Checkpoint::parse(&config).unwrap();
        let mut t = real_tensors();
        let e = map_of(&tied, &t).err().expect("refused");
        assert!(e.contains("lm_head.weight") && e.contains("is true"), "{e}");
        t.retain(|(n, _)| n != "lm_head.weight");
        let mapping = map_of(&tied, &t).unwrap();
        assert_eq!(mapping.tensors.len(), 290);
        assert!(!mapping.tensors.contains_key("output.weight"));
        config
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        let e = Checkpoint::parse(&config).unwrap_err().to_string();
        assert!(e.contains("tie_word_embeddings is missing"), "{e}");
    }

    /// A folder of two layers of a Llama as small as the groups allow (4 heads over 2 of 16
    /// values, a hidden width of 64, `rope_scaling` of type `llama3`), written in MLX's names with
    /// bf16 scales and gains, opens as a llama file: the engine's own reader describes it (the
    /// pairwise rope with the factors, no head norm, a dense SwiGLU), the rope's factors are a
    /// tensor of the header whose bytes are the rule's, and each tensor reads whole from the
    /// source: the query and key rows moved, head by head, to neighbours, every other matrix as
    /// stored. Mutants (each red here): the key projection moved by the query's heads, the second
    /// value of each pair taken from the first half, the value projection moved.
    #[test]
    fn a_small_llama_folder_opens_as_a_llama_file() {
        use model::arch::models::{Mixer, RopeMode};
        use model::part::{Source, Take};

        let quantization = json!({"group_size": 64, "bits": 4});
        let config = json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "attention_bias": false, "mlp_bias": false, "pretraining_tp": 1,
            "tie_word_embeddings": false, "hidden_act": "silu",
            "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"},
            "bos_token_id": 2, "eos_token_id": [2, 3], "vocab_size": 8,
            "num_hidden_layers": 2, "max_position_embeddings": 4096, "hidden_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128,
            "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
            "quantization": quantization, "quantization_config": quantization,
        });
        let dir = temp_dir("llama-open");
        let mut t: Vec<Stored> = vec![];
        for stem in ["model.embed_tokens", "lm_head"] {
            t.extend(mlx_linear(stem, 4, &[8], 64));
        }
        t.push(bf16_tensor("model.norm.weight", &[64], |_| 0x3fc0));
        for l in 0..2 {
            let m = format!("model.layers.{l}");
            for (norm, bf16) in [
                ("input_layernorm", 0x3fc0),
                ("post_attention_layernorm", 0x3f80),
            ] {
                t.push(bf16_tensor(&format!("{m}.{norm}.weight"), &[64], |_| bf16));
            }
            for (proj, rows, k) in [
                ("gate_proj", 128, 64),
                ("up_proj", 128, 64),
                ("down_proj", 64, 128),
            ] {
                t.extend(mlx_linear(&format!("{m}.mlp.{proj}"), 4, &[rows], k));
            }
            for (stem, rows) in [
                ("q_proj", 64),
                ("k_proj", 32),
                ("v_proj", 32),
                ("o_proj", 64),
            ] {
                t.extend(mlx_linear(&format!("{m}.self_attn.{stem}"), 4, &[rows], 64));
            }
        }
        write_folder(&dir, &t, &config, &tokenizer("llama-bpe"));
        let opened = folder::open(&dir).unwrap();
        let named = folder::architecture(config.to_string().as_bytes()).unwrap();
        assert_eq!(Some(named), opened.header.architecture());
        let read = model::arch::spec(&opened.header).unwrap();
        let spec = &read.spec;
        assert_eq!((spec.layers.len(), spec.hidden, spec.vocab), (2, 64, 8));
        for layer in &spec.layers {
            let Mixer::Gqa(g) = &layer.mixer else {
                panic!("a layer mixes with {:?}", layer.mixer);
            };
            assert_eq!((g.heads, g.kv_heads, g.head_dim), (4, 2, 16));
            assert_eq!(
                (g.rope.mode, g.rope.dims, g.rope.base),
                (RopeMode::NormTail, 16, 500_000.0)
            );
            assert!(g.rope.factors && !g.qk_norm);
        }
        let ty = |name: &str| opened.header.find(name).unwrap().1.ty;
        assert_eq!(ty("token_embd.weight"), GgmlType::MLX_A4);
        assert_eq!(ty("output.weight"), GgmlType::MLX_A4);
        assert_eq!(ty("blk.1.attn_k.weight"), GgmlType::MLX_A4);
        assert_eq!(ty("blk.0.attn_norm.weight"), GgmlType::F32);
        assert_eq!(ty("rope_freqs.weight"), GgmlType::F32);
        assert_eq!(opened.header.iter_tensors().count(), 4 + 2 * 9);
        let planned = opened
            .source
            .plan(|_, _, _| Ok::<_, String>(Take::Whole))
            .unwrap();
        let read = |name: &str| -> Vec<u8> {
            let p = planned.iter().find(|p| p.info.name == name).unwrap();
            let part = opened.source.read(p).unwrap();
            assert_eq!(part.spans().concat().len() as u64, p.info.nbytes, "{name}");
            part.spans().concat()
        };
        let f32s = |name: &str| -> Vec<f32> {
            read(name)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        assert_eq!(
            f32s("rope_freqs.weight"),
            llama3_factors(16, 500_000.0, LLAMA31)
        );
        assert_eq!(f32s("blk.1.attn_norm.weight"), [1.5; 64]);
        assert_eq!(f32s("blk.0.ffn_norm.weight"), [1.0; 64]);
        // A group is one row here: its 8 code words, then its scale and its bias (0.5, -0.25).
        let words = |row: usize| -> Vec<u8> {
            let at = (row * 8) as u32;
            (at..at + 8)
                .flat_map(|i| i.wrapping_mul(0x9e37_79b9).to_le_bytes())
                .collect()
        };
        let stored = |name: &str, heads: usize, row: usize| -> Vec<u8> {
            // The file's row `row` of a projection of `heads` heads of 16 values, as stored:
            // head `h`, value `v` is the stored row `h * 16 + v`; the file's `2i` is the stored
            // `i` and `2i + 1` the stored `8 + i` of the head.
            let (head, at) = (row / 16, row % 16);
            let source = head * 16 + if at % 2 == 0 { at / 2 } else { 8 + at / 2 };
            let _ = (name, heads);
            words(source)
        };
        let q = read("blk.0.attn_q.weight");
        assert_eq!(q.len(), 64 * 36);
        for row in [0, 1, 2, 15, 16, 17, 33, 63] {
            assert_eq!(
                q[row * 36..][..32],
                stored("q", 4, row)[..],
                "attn_q row {row}"
            );
            assert_eq!(
                q[row * 36 + 32..][..4],
                [0x00, 0x3f, 0x80, 0xbe],
                "attn_q row {row}"
            );
        }
        let k = read("blk.0.attn_k.weight");
        assert_eq!(k.len(), 32 * 36);
        for row in [0, 1, 2, 3, 15, 16, 17, 31] {
            assert_eq!(
                k[row * 36..][..32],
                stored("k", 2, row)[..],
                "attn_k row {row}"
            );
        }
        // The value projection and the output are as stored.
        for (name, rows) in [
            ("blk.0.attn_v.weight", 32),
            ("blk.0.attn_output.weight", 64),
        ] {
            let bytes = read(name);
            assert_eq!(bytes.len(), rows * 36);
            for row in [0, 1, rows - 1] {
                assert_eq!(bytes[row * 36..][..32], words(row)[..], "{name} row {row}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
