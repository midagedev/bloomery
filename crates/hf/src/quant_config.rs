//! A checkpoint's quantization config read into one format a tensor.
//!
//! ModelOpt writes `hf_quant_config.json`: `quant_algo`, the per-module `quantized_layers` of a
//! MIXED_PRECISION export and `exclude_modules` globs. transformers' `fp8` (`config.json`
//! `quantization_config`) gives `weight_block_size` and `modules_to_not_convert`;
//! compressed-tensors gives `config_groups` and `ignore`; the `mxfp8` method gives MXFP8 to every
//! module outside `ignored_layers` and, with `moe_quant_algo`, NVFP4 pack-quantized
//! ([`Format::Nvfp4Ct`]) to the routed experts. MLX's own block (`config.json` `quantization`,
//! what mlx-lm's convert writes, with a verbatim copy at `quantization_config`) gives
//! `group_size`, `bits`, `mode` and per-module `{group_size, bits, mode}` overrides keyed as the
//! tensor names minus `.weight`, and `false` for a module it left unquantized. [`Config`] reads
//! them into the formats each module is declared, and [`plan`] gives every tensor of a checkpoint
//! one [`Format`]: each weight's stored tensors (dtype, shape and companion scales) are matched
//! against the formats its module is declared.
//!
//! A BF16, F16 or F32 tensor is placed as stored: configs quantize linears, and a tensor table cannot
//! tell a linear from a norm. A quantized weight whose module the config leaves unquantized, or
//! whose tensors fit none of its module's formats, is refused by name, as is a companion (a scale,
//! an input scale, a packed shape) no placed weight claims, and any other dtype. An unknown
//! `quant_method`, algorithm or scheme is refused when the config is read. `input_scale` (an
//! activation scale) is reported, not used.

use std::cell::LazyCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use gguf::GgmlType;
use regex::Regex;
use serde_json::Value;

use crate::schema::{
    self, Fields,
    Rule::{Only, Read},
};
use safetensors::Dtype;

/// Why a config or a tensor was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("{0}: not JSON: {1}")]
    Json(String, #[source] serde_json::Error),
    #[error("quant config: {0}")]
    Config(String),
}

fn bad(what: impl Into<String>) -> Error {
    Error::Config(what.into())
}

/// How a weight is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// U8 E2M1 pairs, F8_E4M3 `weight_scale` a 16 values, F32 `weight_scale_2` a tensor.
    Nvfp4,
    /// [`Format::Nvfp4`] with its `weight_scale` stored compressed as `csf`: a `nvfp4_csf_fixed`
    /// U8 slab a 16 rows and `nvfp4_csf_exceptions` U32.
    Nvfp4Csf,
    /// [`Format::Nvfp4`] with its `weight_scale` stored compressed as `csf_32x2_v1`: the
    /// `csf32_bases`, `_codes`, `_page_bases`, `_offsets` and `_exceptions` parts.
    Nvfp4Csf32,
    /// compressed-tensors `nvfp4-pack-quantized`: U8 E2M1 pairs in `weight_packed`, F8_E4M3
    /// `weight_scale` a 16 values, F32 `weight_global_scale` a matrix, a divisor of the block
    /// scales where ModelOpt's `weight_scale_2` is a multiplier of them, and an F32
    /// `input_global_scale`, the scale of a 4-bit activation.
    Nvfp4Ct,
    /// OCP MXFP8: F8_E4M3 values, one E8M0 scale (U8 or F8_E8M0) a 32 values.
    Mxfp8,
    /// F8_E4M3 with one scale a `rows` by `cols` block.
    Fp8Block {
        rows: usize,
        cols: usize,
    },
    /// F8_E4M3 with one scale an output row.
    Fp8Channel,
    /// F8_E4M3 with one scale a tensor.
    Fp8Tensor,
    /// OCP MXFP4: U8 or I8 E2M1 pairs, one E8M0 scale a 32 values.
    Mxfp4,
    /// compressed-tensors pack-quantized: eight symmetric int4 an I32, one scale a `group` values.
    Int4Pack {
        group: usize,
    },
    /// MLX's affine quantization (`config.json`'s `quantization`, mlx-lm's
    /// convert): `bits` 4-bit codes an U32 word, one BF16 `.scales` value and
    /// one BF16 `.biases` value a `group` values along the row. What a config
    /// declares, and what [`plan`] places a weight whose companions are BF16 as.
    MlxAffine {
        bits: usize,
        group: usize,
    },
    /// [`Format::MlxAffine`] whose `.scales` and `.biases` are F16: a config never
    /// declares it, [`plan`] places a weight whose companions are F16 as it.
    MlxAffineF16 {
        bits: usize,
        group: usize,
    },
    Bf16,
    F32,
    /// F16 as stored, read widened to f32 (exact) where a role asks for F32.
    F16,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::Nvfp4 => f.write_str("NVFP4-ModelOpt"),
            Format::Nvfp4Csf => f.write_str("NVFP4-csf"),
            Format::Nvfp4Csf32 => f.write_str("NVFP4-csf32"),
            Format::Nvfp4Ct => f.write_str("NVFP4-compressed-tensors"),
            Format::Mxfp8 => f.write_str("MXFP8"),
            Format::Fp8Block { rows, cols } => write!(f, "FP8-block {rows}x{cols}"),
            Format::Fp8Channel => f.write_str("FP8-channel"),
            Format::Fp8Tensor => f.write_str("FP8-tensor"),
            Format::Mxfp4 => f.write_str("MXFP4-OCP"),
            Format::Int4Pack { group } => write!(f, "INT4-pack group {group}"),
            Format::MlxAffine { bits, group } => {
                write!(f, "MLX-affine {bits}-bit group {group}")
            }
            Format::MlxAffineF16 { bits, group } => {
                write!(f, "MLX-affine F16 {bits}-bit group {group}")
            }
            Format::Bf16 => f.write_str("BF16"),
            Format::F32 => f.write_str("F32"),
            Format::F16 => f.write_str("F16"),
        }
    }
}

/// A tensor of one format as a card holds it, its block scales decoded to plain ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    /// One tensor scale a matrix, which every rank holds whole.
    pub whole_bytes: u64,
    /// The values a cut keeps whole along dims 0 and 1 (ggml's order).
    pub block: [u64; 2],
}

impl Format {
    /// The card type the format is held in, whose row ([`GgmlType::row`]) gives its bytes and
    /// cut units: NVFP4 in GGUF's NVFP4 block whichever encoding its scales were stored in. An
    /// F8_E4M3 weight keeps its values, each row block of the card type holding the scale of the
    /// checkpoint's block it lies in ([`Format::fp8_scale_block`], `hf::modelopt`'s reader):
    /// bloomery's FP8_B128 (an f32 a 128 values) when a scale covers whole 128-value blocks of a
    /// row, as one a tensor, a row or a block 128 values wide does; MXFP8 (an E8M0 byte a 32
    /// values) when it covers whole 32-value ones. Another FP8 block has no card type.
    #[must_use]
    pub fn card_type(self) -> Option<GgmlType> {
        match self {
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 | Format::Nvfp4Ct => {
                Some(GgmlType::NVFP4)
            }
            Format::Fp8Channel | Format::Fp8Tensor => Some(GgmlType::FP8_B128),
            Format::Mxfp8 => Some(GgmlType::MXFP8),
            Format::Fp8Block { rows, cols } if rows > 0 && cols > 0 => {
                if cols % 128 == 0 {
                    Some(GgmlType::FP8_B128)
                } else if cols % 32 == 0 {
                    Some(GgmlType::MXFP8)
                } else {
                    None
                }
            }
            Format::Mxfp4 => Some(GgmlType::MXFP4),
            // The block type holds the group's scale and bias beside its codes, so only the
            // block sizes' own group (64) has one.
            Format::MlxAffine { bits: 4, group: 64 } => Some(GgmlType::MLX_A4),
            Format::MlxAffine { bits: 8, group: 64 } => Some(GgmlType::MLX_A8),
            Format::MlxAffineF16 { bits: 4, group: 64 } => Some(GgmlType::MLX_A4H),
            Format::MlxAffineF16 { bits: 8, group: 64 } => Some(GgmlType::MLX_A8H),
            Format::Bf16 => Some(GgmlType::BF16),
            Format::F32 => Some(GgmlType::F32),
            Format::F16 => Some(GgmlType::F16),
            _ => None,
        }
    }

    /// The format a weight reads as once its block scales are decoded ([`crate::csf`]): plain
    /// NVFP4 for NVFP4 whose block scales are stored compressed, the format itself for any other.
    #[must_use]
    pub fn decoded(self) -> Format {
        match crate::csf::Encoding::of(self) {
            Some(_) => Format::Nvfp4,
            None => self,
        }
    }

    /// The bit width, the group and the dtype of the `.scales` and `.biases` of an MLX affine
    /// format; `None` for another.
    #[must_use]
    pub fn mlx_affine(self) -> Option<(usize, usize, Dtype)> {
        match self {
            Format::MlxAffine { bits, group } => Some((bits, group, Dtype::Bf16)),
            Format::MlxAffineF16 { bits, group } => Some((bits, group, Dtype::F16)),
            _ => None,
        }
    }

    /// The rows and values one scale of an F8_E4M3 weight of `rows` by `k` values covers in the
    /// format, when `shape` stores the grid of its scales, `[r, c]` rounded up: as `[r, c]`, with
    /// ModelOpt's unit axes `[r, 1, c, 1]`, a grid of one column as `[r]`, one scale as `[]`. One
    /// scale covers MXFP8's one row by 32 values, a block format's block, a row for a channel
    /// scale and the weight for a tensor scale; `None` for another format, an empty block or
    /// another shape.
    pub(crate) fn fp8_scale_block(
        self,
        rows: usize,
        k: usize,
        shape: &[usize],
    ) -> Option<[usize; 2]> {
        let [a, b] = match self {
            Format::Mxfp8 => [1, 32],
            Format::Fp8Block { rows, cols } => [rows, cols],
            Format::Fp8Channel => [1, k],
            Format::Fp8Tensor => [rows, k],
            _ => return None,
        };
        if a == 0 || b == 0 {
            return None;
        }
        let (r, c) = (rows.div_ceil(a), k.div_ceil(b));
        let fits = shape == [r, c]
            || shape == [r, 1, c, 1]
            || (c == 1 && shape == [r])
            || (r == 1 && c == 1 && shape.is_empty());
        fits.then_some([a, b])
    }

    /// The held form of a tensor of the format and logical `dims` (ggml's order: dim 0 the values
    /// a row holds, dim 1 the rows, the dims past them whole matrices) in its card type;
    /// `None` for a format no card holds, a zero dimension, a byte count past
    /// `u64`, or a row that is not whole blocks of its card type. NVFP4 adds its one F32 tensor
    /// scale a matrix.
    #[must_use]
    pub fn held(self, dims: &[u64]) -> Option<Held> {
        let k = *dims.first().filter(|_| !dims.contains(&0))?;
        let matrices = dims
            .iter()
            .skip(2)
            .try_fold(1u64, |a, &d| a.checked_mul(d))?;
        let row = self.card_type()?.row()?;
        let block = [row.blck, 1];
        if k % block[0] != 0 {
            return None;
        }
        gguf::tensor_bytes(row.blck, row.bytes, dims)?;
        let whole_bytes = match self {
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 | Format::Nvfp4Ct => {
                matrices.checked_mul(4)?
            }
            _ => 0,
        };
        Some(Held { whole_bytes, block })
    }

    /// The logical shape of a tensor of the format with `stored` shape (row-major): two E2M1
    /// values a stored byte for NVFP4 and MXFP4, eight int4 an I32 for the packed format, and
    /// `32 / bits` codes an U32 for the MLX one. `stored`
    /// is a weight [`plan`] placed in the format, whose unpacked count it checked fits.
    #[must_use]
    pub fn logical(self, stored: &[usize]) -> Vec<usize> {
        let mut s = stored.to_vec();
        let per = match self {
            Format::Nvfp4
            | Format::Nvfp4Csf
            | Format::Nvfp4Csf32
            | Format::Nvfp4Ct
            | Format::Mxfp4 => 2,
            Format::Int4Pack { .. } => 8,
            Format::MlxAffine { bits, .. } | Format::MlxAffineF16 { bits, .. } => 32 / bits,
            _ => 1,
        };
        if let Some(last) = s.last_mut() {
            *last *= per;
        }
        s
    }
}

/// A checkpoint's quantization config: the formats each module is declared.
#[derive(Debug, Default)]
pub struct Config {
    /// MLX's own `quantization` block, which takes a module before any other.
    mlx: Option<Mlx>,
    /// ModelOpt's config, which takes a module first.
    modelopt: Option<Modelopt>,
    /// `config.json`'s own config: the modules ModelOpt leaves, or every module.
    other: Option<Other>,
}

#[derive(Debug)]
struct Modelopt {
    /// The top-level algorithm's format; `None` for MIXED_PRECISION.
    all: Option<Format>,
    /// `quantized_layers`: a module and its format.
    layers: BTreeMap<String, Format>,
    exclude: Vec<String>,
}

/// MLX's `quantization` block: the affine format every module takes, and the
/// per-module `{group_size, bits, mode}` entries that override it, keyed
/// exactly as the tensor names minus `.weight` (mlx-lm's convert writes them
/// from the module paths it quantized). A `false` entry is a module the convert
/// left unquantized: `None`, whose tensors the plan holds to stored floats.
#[derive(Debug)]
struct Mlx {
    all: Format,
    layers: BTreeMap<String, Option<Format>>,
}

#[derive(Debug)]
enum Other {
    Fp8 {
        block: Format,
        mxfp4: bool,
        not_convert: Vec<String>,
        convert: Vec<String>,
    },
    CompressedTensors {
        groups: Vec<(Vec<Target>, Format)>,
        ignore: Vec<Target>,
    },
    /// `quant_method` `mxfp8`: MXFP8 on every module outside `ignored`, and with `nvfp4` the
    /// routed experts' pack-quantized NVFP4 beside it.
    Mxfp8 { nvfp4: bool, ignored: Vec<String> },
}

/// The module classes a model's checkpoint map can identify for class-name targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleClass {
    Linear,
    Embedding,
}

/// A compressed-tensors target: a module class, a module by name, or `re:` a regex from the start.
#[derive(Debug)]
enum Target {
    Class(ModuleClass),
    Name(String),
    Re(Regex),
}

impl Target {
    fn parse(t: &str) -> Result<Target, Error> {
        Ok(match t.strip_prefix("re:") {
            Some(re) => Target::Re(
                Regex::new(&format!("^(?:{re})"))
                    .map_err(|e| bad(format!("compressed-tensors pattern {t}: {e}")))?,
            ),
            None if t == "Linear" => Target::Class(ModuleClass::Linear),
            None if t == "Embedding" => Target::Class(ModuleClass::Embedding),
            None if !t.contains('.')
                && t.as_bytes().first().is_some_and(u8::is_ascii_uppercase) =>
            {
                return Err(bad(format!(
                    "compressed-tensors class {t} is not one this reader knows"
                )));
            }
            None => Target::Name(t.to_string()),
        })
    }

    fn matches(&self, module: &str, class: impl FnOnce() -> Option<ModuleClass>) -> bool {
        match self {
            Target::Class(want) => class() == Some(*want),
            Target::Name(n) => n == module,
            Target::Re(re) => re.is_match(module),
        }
    }
}

impl Config {
    /// The config of the model folder `dir`: `config.json`, and `hf_quant_config.json` when the
    /// folder has one.
    pub fn read(dir: &Path) -> Result<Config, Error> {
        let json = |name: &str| crate::read_json(dir, name, Error::Io, Error::Json);
        let config = json("config.json")?.ok_or_else(|| bad("the folder has no config.json"))?;
        Config::parse(&config, json("hf_quant_config.json")?.as_ref())
    }

    /// The config from `config.json` and ModelOpt's `hf_quant_config.json`.
    pub fn parse(config: &Value, hf_quant: Option<&Value>) -> Result<Config, Error> {
        let mlx = config.get("quantization").map(Mlx::parse).transpose()?;
        let qc = config
            .get("quantization_config")
            .or_else(|| config.get("text_config")?.get("quantization_config"));
        // MLX's convert writes its `quantization` block beside a verbatim copy
        // of it at `quantization_config`, with no `quant_method`; the copy is
        // not a second producer, and anything else beside the block is.
        let qc = match (qc, &mlx) {
            (Some(q), Some(_)) if !q.get("quant_method").is_some_and(Value::is_string) => {
                if q != config
                    .get("quantization")
                    .expect("the MLX block is present")
                {
                    return Err(bad(
                        "quantization_config beside quantization is not its copy",
                    ));
                }
                None
            }
            (q, _) => q,
        };
        if mlx.is_some() && hf_quant.is_some() {
            return Err(bad(
                "a quantization block beside ModelOpt's hf_quant_config.json",
            ));
        }
        let method = qc.map(|q| str_of(q, "quant_method")).transpose()?;
        if let (Some(_), Some(m)) = (&mlx, method) {
            return Err(bad(format!(
                "a quantization block beside quantization_config's quant_method {m}"
            )));
        }
        if method == Some("quark") {
            return Err(bad("Quark checkpoints are not read yet"));
        }
        let mut out = Config {
            mlx,
            ..Config::default()
        };
        // ModelOpt's own file wins over the copy it writes into config.json.
        let modelopt = match (hf_quant, method) {
            (Some(h), _) => Some(h.get("quantization").unwrap_or(h)),
            (None, Some("modelopt")) => qc,
            _ => None,
        };
        if let Some(m) = modelopt {
            out.modelopt = Some(Modelopt::parse(m)?);
        }
        out.other = match (qc, method) {
            (_, None | Some("modelopt")) => None,
            (Some(q), Some("fp8")) => Some(Other::fp8(q)?),
            (Some(q), Some("compressed-tensors")) => Some(Other::compressed_tensors(q)?),
            (Some(q), Some("mxfp8")) => Some(Other::mxfp8(q)?),
            (_, Some(m)) => {
                return Err(bad(format!(
                    "quant_method {m} is not one this reader knows"
                )));
            }
        };
        Ok(out)
    }

    /// The formats `module` is declared, and the module whose scale a tensor-scale format shares;
    /// empty when the config leaves it unquantized.
    fn declared(
        &self,
        module: &str,
        class_of: impl FnOnce() -> Option<ModuleClass>,
    ) -> (Vec<Format>, Option<&str>) {
        if let Some(m) = &self.mlx {
            // The block has no exclude list: a module takes its own entry or
            // the block's format, and is left unquantized only by a `false`
            // entry, which no quantized tensor of it fits.
            let formats = match m.layers.get(module) {
                Some(Some(own)) => vec![*own],
                Some(None) => vec![],
                None => vec![m.all],
            };
            return (formats, None);
        }
        if let Some(m) = &self.modelopt {
            let mut ancestor = module;
            let layer = loop {
                if let Some(entry) = m.layers.get_key_value(ancestor) {
                    break Some(entry);
                }
                let Some((parent, _)) = ancestor.rsplit_once('.') else {
                    break None;
                };
                ancestor = parent;
            };
            if let Some((k, f)) = layer {
                return (vec![*f], Some(k));
            }
            if let Some(f) = m.all
                && !m.exclude.iter().any(|p| glob(p, module))
            {
                return (vec![f], None);
            }
        }
        let formats = match &self.other {
            None => vec![],
            Some(Other::Fp8 {
                block,
                mxfp4,
                not_convert,
                convert,
            }) => {
                if skipped(module, not_convert) && !skipped(module, convert) {
                    vec![]
                } else if *mxfp4 {
                    vec![*block, Format::Mxfp4]
                } else {
                    vec![*block]
                }
            }
            Some(Other::Mxfp8 { nvfp4, ignored }) => {
                if skipped(module, ignored) {
                    vec![]
                } else if *nvfp4 {
                    vec![Format::Mxfp8, Format::Nvfp4Ct]
                } else {
                    vec![Format::Mxfp8]
                }
            }
            Some(Other::CompressedTensors { groups, ignore }) => {
                let class = LazyCell::new(class_of);
                let matches = |t: &Target| t.matches(module, || *class);
                if ignore.iter().any(matches) {
                    vec![]
                } else {
                    groups
                        .iter()
                        .filter(|(ts, _)| ts.iter().any(matches))
                        .map(|(_, f)| *f)
                        .collect()
                }
            }
        };
        (formats, None)
    }
}

/// A `quantized_layers` entry's keys, each read by [`modelopt_format`].
const LAYER: &Fields = &[
    ("quant_algo", Read),
    ("group_size", Read),
    ("weight_block_size", Read),
    ("weight_scale_encoding", Read),
];

/// A per-module `quantization` entry's keys, each read by [`mlx_format`]; `mode`
/// is the block's rule: mlx-lm writes it into every entry of a fine-grained
/// config.
const MLX_LAYER: &Fields = &[
    ("group_size", Read),
    ("bits", Read),
    ("mode", Only("[\"affine\"]")),
];

impl Mlx {
    fn parse(q: &Value) -> Result<Mlx, Error> {
        let object = q
            .as_object()
            .ok_or_else(|| bad("quantization is no object"))?;
        let mut block = serde_json::Map::new();
        let mut layers = BTreeMap::new();
        for (key, entry) in object {
            if matches!(key.as_str(), "group_size" | "bits" | "mode") {
                block.insert(key.clone(), entry.clone());
                continue;
            }
            if entry.as_bool() == Some(false) {
                layers.insert(key.clone(), None);
                continue;
            }
            let entry = entry.as_object().ok_or_else(|| {
                bad(format!(
                    "quantization {key} {entry} is neither a per-module object, false nor one of \
                     the block's keys"
                ))
            })?;
            let value = Value::Object(entry.clone());
            schema::check(&value, &format!("quantization {key}"), &[MLX_LAYER]).map_err(bad)?;
            layers.insert(key.clone(), Some(mlx_format(key, &value)?));
        }
        let block = Value::Object(block);
        schema::check(&block, "quantization", &[MLX_BLOCK]).map_err(bad)?;
        Ok(Mlx {
            all: mlx_format("the block", &block)?,
            layers,
        })
    }
}

/// The `quantization` block's own keys, each read by [`mlx_format`]; `mode`
/// must name the affine form this reader places. An absent `mode` is the
/// affine form: mlx-community's Qwen3-30B-A3B config names none.
const MLX_BLOCK: &Fields = &[
    ("group_size", Read),
    ("bits", Read),
    ("mode", Only("[\"affine\"]")),
];

/// The affine format of one `quantization` entry (`what` names it in a
/// refusal): a `bits` width and a `group_size`. Only the block sizes this
/// reader places are formats — 4 and 8 bits at a group of 64, gguf's MLX_A4
/// and MLX_A8, or MLX_A4H and MLX_A8H when the companions are F16 — and another value is refused by name, as is a width or group
/// the blocks do not have.
fn mlx_format(what: &str, entry: &Value) -> Result<Format, Error> {
    let refuse = |key: &str| {
        bad(format!(
            "quantization {key} {} for {what} is not one this reader places",
            entry.get(key).unwrap_or(&Value::Null)
        ))
    };
    let bits = entry
        .get("bits")
        .and_then(Value::as_u64)
        .filter(|&b| b == 4 || b == 8);
    let group = entry
        .get("group_size")
        .and_then(Value::as_u64)
        .filter(|&g| g == 64);
    if bits.is_none() {
        return Err(refuse("bits"));
    }
    if group.is_none() {
        return Err(refuse("group_size"));
    }
    Ok(Format::MlxAffine {
        bits: bits.unwrap() as usize,
        group: group.unwrap() as usize,
    })
}

/// An `mxfp8` `quantization_config`'s keys, each read by [`Other::mxfp8`]. The activations are
/// quantized at run time (`dynamic`), as this engine does whatever the scheme.
const MXFP8: &Fields = &[
    ("quant_method", Read),
    ("activation_scheme", Only(r#"["dynamic"]"#)),
    ("weight_block_size", Read),
    ("ignored_layers", Read),
    ("group_size", Read),
    ("moe_quant_algo", Read),
    ("moe_quant_format", Read),
];

impl Modelopt {
    fn parse(q: &Value) -> Result<Modelopt, Error> {
        let algo = str_of(q, "quant_algo")?;
        let all = match algo {
            "MIXED_PRECISION" => None,
            _ => Some(modelopt_format("the checkpoint", algo, q)?),
        };
        let mut layers = BTreeMap::new();
        if let Some(l) = q.get("quantized_layers") {
            let l = l
                .as_object()
                .ok_or_else(|| bad("modelopt quantized_layers is no object"))?;
            for (module, entry) in l {
                let entry = entry.as_object().ok_or_else(|| {
                    bad(format!("modelopt quantized_layers {module} is no object"))
                })?;
                let value = Value::Object(entry.clone());
                let format = modelopt_format(module, str_of(&value, "quant_algo")?, &value)?;
                schema::check(
                    &value,
                    &format!("modelopt quantized_layers {module}"),
                    &[LAYER],
                )
                .map_err(bad)?;
                layers.insert(module.clone(), format);
            }
        }
        let exclude = q
            .get("exclude_modules")
            .or_else(|| q.get("ignore"))
            .map_or(Ok(vec![]), |v| strings(v, "modelopt exclude_modules"))?;
        if let Some(p) = exclude.iter().find(|p| p.contains('[')) {
            return Err(bad(format!(
                "modelopt exclude pattern {p}: a [ class is not matched by this reader"
            )));
        }
        Ok(Modelopt {
            all,
            layers,
            exclude,
        })
    }
}

/// The format of ModelOpt's `algo` for `module`, with `entry`'s group or block size and NVFP4's
/// `weight_scale_encoding`. W4A16_NVFP4 stores its weights as NVFP4 does; only its activations
/// stay 16-bit, which no stored tensor shows.
fn modelopt_format(module: &str, algo: &str, entry: &Value) -> Result<Format, Error> {
    let group = match entry.get("group_size") {
        None | Some(Value::Null) => None,
        Some(g) => Some(g.as_u64().filter(|&g| g > 0).ok_or_else(|| {
            bad(format!(
                "modelopt group_size {g} for {module} is not a count"
            ))
        })?),
    };
    let refuse = || {
        bad(format!(
            "modelopt quant_algo {algo} for {module} is not a format this reader places"
        ))
    };
    let encoding = entry.get("weight_scale_encoding");
    if encoding.is_some() && algo != "NVFP4" {
        return Err(bad(format!(
            "modelopt quant_algo {algo} for {module}: weight_scale_encoding is NVFP4's alone"
        )));
    }
    match algo {
        "NVFP4" if group.is_none_or(|g| g == 16) => match encoding {
            None => Ok(Format::Nvfp4),
            Some(e) => match e.as_str() {
                Some("csf") => Ok(Format::Nvfp4Csf),
                Some("csf_32x2_v1") => Ok(Format::Nvfp4Csf32),
                _ => Err(bad(format!(
                    "modelopt weight_scale_encoding {e} for {module} is not one this reader knows"
                ))),
            },
        },
        "W4A16_NVFP4" if group.is_none_or(|g| g == 16) => Ok(Format::Nvfp4),
        "MXFP8"
            if group == Some(32)
                || block_of(entry.get("weight_block_size"))
                    == Some(Format::Fp8Block { rows: 1, cols: 32 }) =>
        {
            Ok(Format::Mxfp8)
        }
        "FP8" => Ok(Format::Fp8Tensor),
        "FP8_BLOCK_SCALES" if group.is_none_or(|g| g == 128) => Ok(Format::Fp8Block {
            rows: 128,
            cols: 128,
        }),
        "FP8_PB_WO" => block_of(entry.get("weight_block_size")).ok_or_else(refuse),
        _ => Err(refuse()),
    }
}

impl Other {
    fn fp8(q: &Value) -> Result<Other, Error> {
        let refuse = |what: &str| bad(format!("fp8 {what} is not one this reader places"));
        for (key, kind) in [
            ("fmt", Value::is_string as fn(&Value) -> bool),
            ("weight_per_tensor", Value::is_boolean),
            ("expert_dtype", Value::is_string),
            ("store_dtype", Value::is_string),
            ("mxfp4_block_size", Value::is_u64),
        ] {
            if q.get(key).is_some_and(|v| !v.is_null() && !kind(v)) {
                return Err(refuse(key));
            }
        }
        if let Some(f) = q
            .get("fmt")
            .and_then(Value::as_str)
            .filter(|f| *f != "e4m3")
        {
            return Err(refuse(&format!("fmt {f}")));
        }
        if q.get("weight_per_tensor").and_then(Value::as_bool) == Some(true) {
            return Err(refuse("weight_per_tensor"));
        }
        let block =
            block_of(q.get("weight_block_size")).ok_or_else(|| refuse("weight_block_size"))?;
        let mxfp4 = match (
            q.get("expert_dtype").and_then(Value::as_str),
            q.get("store_dtype").and_then(Value::as_str),
        ) {
            (None, None) => false,
            (Some("fp4"), None) | (None, Some("mxfp4")) => true,
            (e, s) => return Err(refuse(&format!("expert_dtype {e:?} store_dtype {s:?}"))),
        };
        if q.get("mxfp4_block_size")
            .and_then(Value::as_u64)
            .is_some_and(|b| b != 32)
        {
            return Err(refuse("mxfp4_block_size"));
        }
        let list = |key: &str| q.get(key).map_or(Ok(vec![]), |v| strings(v, key));
        let mut not_convert = list("modules_to_not_convert")?;
        not_convert.extend(list("ignored_layers")?);
        let convert = list("modules_to_convert")?;
        literal_patterns("fp8", not_convert.iter().chain(&convert))?;
        Ok(Other::Fp8 {
            block,
            mxfp4,
            not_convert,
            convert,
        })
    }

    fn mxfp8(q: &Value) -> Result<Other, Error> {
        schema::check(q, "mxfp8 quantization_config", &[MXFP8]).map_err(bad)?;
        // MXFP8 is one E8M0 a 32 values; a config that names its block names that one.
        if let Some(b) = q.get("weight_block_size")
            && block_of(Some(b)) != Some(Format::Fp8Block { rows: 1, cols: 32 })
        {
            return Err(bad(format!(
                "mxfp8 weight_block_size {b} is not [1, 32], the one block MXFP8 is stored in"
            )));
        }
        let moe = |key: &str| {
            q.get(key)
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| bad(format!("mxfp8 {key} {v} is not a string")))
                })
                .transpose()
        };
        let nvfp4 = match (
            moe("moe_quant_algo")?,
            moe("moe_quant_format")?,
            q.get("group_size"),
        ) {
            (None, None, None) => false,
            (Some("NVFP4"), Some("nvfp4-pack-quantized"), group)
                if group.is_none_or(|g| g.as_u64() == Some(16)) =>
            {
                true
            }
            (algo, format, group) => {
                return Err(bad(format!(
                    "mxfp8 moe_quant_algo {algo:?}, moe_quant_format {format:?} and group_size \
                     {group:?} are not NVFP4 pack-quantized with group 16, the one expert format \
                     this reader places"
                )));
            }
        };
        let ignored = q
            .get("ignored_layers")
            .map_or(Ok(vec![]), |v| strings(v, "mxfp8 ignored_layers"))?;
        literal_patterns("mxfp8", &ignored)?;
        Ok(Other::Mxfp8 { nvfp4, ignored })
    }

    fn compressed_tensors(q: &Value) -> Result<Other, Error> {
        let groups = q
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("compressed-tensors has no config_groups object"))?;
        let mut out = Vec::new();
        for (name, g) in groups {
            let format = g
                .get("format")
                .or_else(|| q.get("format"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let w = g
                .get("weights")
                .ok_or_else(|| bad(format!("compressed-tensors {name} has no weights")))?;
            let field = |k: &str| w.get(k).cloned().unwrap_or(Value::Null);
            let f = match (
                format,
                field("num_bits").as_u64(),
                field("type").as_str(),
                field("strategy").as_str(),
            ) {
                ("float-quantized", Some(8), Some("float"), Some("channel")) => {
                    Some(Format::Fp8Channel)
                }
                ("float-quantized", Some(8), Some("float"), Some("tensor")) => {
                    Some(Format::Fp8Tensor)
                }
                ("float-quantized", Some(8), Some("float"), Some("block")) => {
                    block_of(w.get("block_structure"))
                }
                ("pack-quantized", Some(4), Some("int"), Some("group"))
                    if field("symmetric") == Value::Bool(true) =>
                {
                    field("group_size")
                        .as_u64()
                        .filter(|&g| g > 0)
                        .map(|g| Format::Int4Pack { group: g as usize })
                }
                _ => None,
            };
            let f = f.ok_or_else(|| bad(format!("compressed-tensors {name}: {format} weights {w} are not a format this reader places")))?;
            let targets = strings(
                g.get("targets").unwrap_or(&Value::Null),
                "compressed-tensors targets",
            )?;
            out.push((
                targets
                    .iter()
                    .map(|t| Target::parse(t))
                    .collect::<Result<_, _>>()?,
                f,
            ));
        }
        let ignore = q
            .get("ignore")
            .map_or(Ok(vec![]), |v| strings(v, "compressed-tensors ignore"))?;
        Ok(Other::CompressedTensors {
            groups: out,
            ignore: ignore
                .iter()
                .map(|t| Target::parse(t))
                .collect::<Result<_, _>>()?,
        })
    }
}

/// A weight's place in the checkpoint: its format and the companion tensors it reads.
#[derive(Debug, PartialEq)]
pub struct Placed {
    pub format: Format,
    /// Scales and the packed shape, by name; ModelOpt's `input_scale` is listed in
    /// [`Plan::input_scales`], compressed-tensors NVFP4's `input_global_scale` stays here, last,
    /// as it is read with the weight.
    pub companions: Vec<String>,
}

/// Every tensor of a checkpoint placed or refused.
#[derive(Debug, Default)]
pub struct Plan {
    /// Each weight and each other placed tensor by name; companions are under their weight.
    pub placed: BTreeMap<String, Placed>,
    /// Each refused tensor by name, and why.
    pub refused: BTreeMap<String, String>,
    /// The `input_scale` tensors of placed weights, not read.
    pub input_scales: Vec<String>,
}

/// The companion suffixes a weight's format reads.
const COMPANIONS: [&str; 11] = [
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "weight_global_scale",
    "scale",
    "scales",
    "biases",
    "input_scale",
    "input_global_scale",
    "weight_shape",
    "weight_zero_point",
];

/// Every tensor of `tensors` (name, dtype, shape) placed by `config`. `class_of` supplies a module's
/// class from the model's map, or `None` for another or unknown role; only class targets ask for it.
pub fn plan<'a>(
    config: &Config,
    tensors: impl IntoIterator<Item = (&'a str, Dtype, &'a [usize])>,
    class_of: impl Fn(&str) -> Option<ModuleClass>,
) -> Plan {
    let table: BTreeMap<&str, (Dtype, &[usize])> =
        tensors.into_iter().map(|(n, d, s)| (n, (d, s))).collect();
    let mut out = Plan::default();
    let mut claimed = BTreeSet::new();
    for (&name, &(dtype, shape)) in &table {
        let Some(module) = name
            .strip_suffix(".weight")
            .or_else(|| name.strip_suffix(".weight_packed"))
        else {
            continue;
        };
        if plain(dtype).is_some() {
            continue;
        }
        let (formats, owner) = config.declared(module, || class_of(module));
        let found = formats
            .iter()
            .find_map(|&f| layout(f, name, module, owner, dtype, shape, &table));
        match found {
            Some((format, companions)) => {
                claimed.extend(companions.iter().cloned());
                let (inputs, companions): (Vec<_>, Vec<_>) = companions
                    .into_iter()
                    .partition(|c| c.ends_with(".input_scale"));
                out.input_scales.extend(inputs);
                out.placed
                    .insert(name.to_string(), Placed { format, companions });
            }
            None if formats.is_empty() => {
                out.refused.insert(
                    name.to_string(),
                    format!(
                        "{} {shape:?}, but the config leaves {module} unquantized",
                        dtype.name()
                    ),
                );
            }
            None => {
                let have: Vec<String> = COMPANIONS
                    .iter()
                    .filter_map(|s| table.get_key_value(format!("{module}.{s}").as_str()))
                    .map(|(n, (d, s))| format!("{n} {} {s:?}", d.name()))
                    .collect();
                let want: Vec<String> = formats.iter().map(Format::to_string).collect();
                let note = formats
                    .iter()
                    .any(|f| f.mlx_affine().is_some())
                    .then(|| mlx_dtype_note(module, &table))
                    .flatten()
                    .unwrap_or_default();
                out.refused.insert(
                    name.to_string(),
                    format!("{} {shape:?} with {have:?} fits none of {want:?}, the config's formats for {module}{note}", dtype.name()),
                );
            }
        }
    }
    for (&name, &(dtype, _)) in &table {
        if out.placed.contains_key(name) || out.refused.contains_key(name) || claimed.contains(name)
        {
            continue;
        }
        let companion = name
            .rsplit_once('.')
            .is_some_and(|(_, last)| COMPANIONS.contains(&last))
            || name.contains(".weight_scale.");
        let why = if companion {
            "a companion no placed weight reads".to_string()
        } else if let Some(f) = plain(dtype) {
            out.placed.insert(
                name.to_string(),
                Placed {
                    format: f,
                    companions: vec![],
                },
            );
            continue;
        } else {
            format!("{} outside a quantized weight", dtype.name())
        };
        out.refused.insert(name.to_string(), why);
    }
    out.input_scales.sort();
    out.input_scales.dedup();
    out
}

fn plain(dtype: Dtype) -> Option<Format> {
    match dtype {
        Dtype::Bf16 => Some(Format::Bf16),
        Dtype::F16 => Some(Format::F16),
        Dtype::F32 => Some(Format::F32),
        _ => None,
    }
}

/// The format a weight is placed in and the companions it reads when its tensors fit `format`
/// (the placed format is `format` but for an MLX affine weight, whose companions' dtype picks
/// [`Format::MlxAffine`] or [`Format::MlxAffineF16`]), or `None` when they do not. `owner` is the
/// module a tensor scale may be shared from.
fn layout(
    format: Format,
    name: &str,
    module: &str,
    owner: Option<&str>,
    dtype: Dtype,
    shape: &[usize],
    table: &BTreeMap<&str, (Dtype, &[usize])>,
) -> Option<(Format, Vec<String>)> {
    // MLX's expert stacks are one 3-D U32 tensor (`switch_mlp`), so the affine
    // format matches any rank whose last dim is the packed row; every other
    // format's weights here are 2-D.
    if let Format::MlxAffine { bits, group } = format {
        return mlx_layout(bits, group, module, dtype, shape, table);
    }
    let &[rows, cols] = shape else { return None };
    // The first of `suffixes` beside the weight (or, for a shared scale, beside `owner`) that is
    // one of `dtypes` and of `want`'s shape.
    let find = |at: &str, suffixes: &[&str], dtypes: &[Dtype], want: &dyn Fn(&[usize]) -> bool| {
        suffixes.iter().find_map(|s| {
            let n = format!("{at}.{s}");
            let &(d, sh) = table.get(n.as_str())?;
            (dtypes.contains(&d) && want(sh)).then_some(n)
        })
    };
    let scalar = |s: &[usize]| s.is_empty() || s == [1];
    // An F8_E4M3 format's scale grid ([`Format::fp8_scale_block`]).
    let fits = |s: &[usize]| format.fp8_scale_block(rows, cols, s).is_some();
    let input = |out: &mut Vec<String>| {
        out.extend(find(
            module,
            &["input_scale"],
            &[Dtype::F32, Dtype::Bf16],
            &scalar,
        ))
    };
    let mut out = Vec::new();
    match format {
        Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 => {
            let k = cols.checked_mul(2)?;
            if dtype != Dtype::U8 || k % 16 != 0 {
                return None;
            }
            let c = k / 16;
            let line = |s: &[usize]| s.len() == 1;
            match format {
                Format::Nvfp4 => {
                    out.push(find(module, &["weight_scale"], &[Dtype::F8E4M3], &|s| {
                        s == [rows, c]
                    })?)
                }
                Format::Nvfp4Csf => {
                    if rows % 16 != 0 || c % 2 != 0 {
                        return None;
                    }
                    let fixed = [rows / 16, 16 * (1 + c / 2)];
                    out.push(find(
                        module,
                        &["weight_scale.nvfp4_csf_fixed"],
                        &[Dtype::U8],
                        &|s| s == fixed,
                    )?);
                    out.push(find(
                        module,
                        &["weight_scale.nvfp4_csf_exceptions"],
                        &[Dtype::U32],
                        &line,
                    )?);
                }
                _ => {
                    let codes = rows.checked_mul(c)?.div_ceil(2);
                    for (part, dtype, want) in [
                        ("bases", Dtype::U8, Some(rows)),
                        ("codes", Dtype::U8, Some(codes)),
                        ("page_bases", Dtype::U32, None),
                        ("offsets", Dtype::U16, None),
                        ("exceptions", Dtype::U16, None),
                    ] {
                        let suffix = format!("weight_scale.csf32_{part}");
                        out.push(find(module, &[suffix.as_str()], &[dtype], &|s| {
                            line(s) && want.is_none_or(|n| s == [n])
                        })?);
                    }
                }
            }
            let own = find(module, &["weight_scale_2"], &[Dtype::F32], &scalar);
            out.push(own.or_else(|| find(owner?, &["weight_scale_2"], &[Dtype::F32], &scalar))?);
            input(&mut out);
        }
        Format::Mxfp8 => {
            if dtype != Dtype::F8E4M3 || cols % 32 != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale", "weight_scale_inv"],
                &[Dtype::U8, Dtype::F8E8M0],
                &fits,
            )?);
            input(&mut out);
        }
        Format::Nvfp4Ct => {
            let k = cols.checked_mul(2)?;
            if !name.ends_with(".weight_packed") || dtype != Dtype::U8 || k % 16 != 0 {
                return None;
            }
            out.push(find(module, &["weight_scale"], &[Dtype::F8E4M3], &|s| {
                s == [rows, k / 16]
            })?);
            out.push(find(
                module,
                &["weight_global_scale"],
                &[Dtype::F32],
                &scalar,
            )?);
            // The input scale is read and checked with the weight ([`crate::modelopt`]), so
            // it is a companion, not an `input_scales` entry.
            out.extend(find(
                module,
                &["input_global_scale"],
                &[Dtype::F32],
                &scalar,
            ));
        }
        Format::Mxfp4 => {
            let k = cols.checked_mul(2)?;
            if !matches!(dtype, Dtype::U8 | Dtype::I8) || k % 32 != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale", "scale"],
                &[Dtype::U8, Dtype::F8E8M0],
                &|s| s == [rows, k / 32],
            )?);
        }
        Format::Fp8Block { .. } if dtype == Dtype::F8E4M3 => {
            let dtypes = [Dtype::F32, Dtype::Bf16, Dtype::F8E8M0];
            out.push(find(
                module,
                &["weight_scale_inv", "weight_scale", "scale"],
                &dtypes,
                &fits,
            )?);
            input(&mut out);
        }
        Format::Fp8Channel if dtype == Dtype::F8E4M3 => {
            let dtypes = [Dtype::F32, Dtype::Bf16];
            out.push(find(module, &["weight_scale"], &dtypes, &fits)?);
            input(&mut out);
        }
        Format::Fp8Tensor if dtype == Dtype::F8E4M3 => {
            let dtypes = [Dtype::F32, Dtype::Bf16];
            let own = find(module, &["weight_scale"], &dtypes, &fits);
            out.push(own.or_else(|| find(owner?, &["weight_scale"], &dtypes, &fits))?);
            input(&mut out);
        }
        Format::Int4Pack { group } => {
            let k = cols.checked_mul(8)?;
            if dtype != Dtype::I32 || !name.ends_with(".weight_packed") || k % group != 0 {
                return None;
            }
            out.push(find(
                module,
                &["weight_scale"],
                &[Dtype::Bf16, Dtype::F16, Dtype::F32],
                &|s| s == [rows, k / group],
            )?);
            out.push(find(
                module,
                &["weight_shape"],
                &[Dtype::I64, Dtype::I32],
                &|s| s == [2],
            )?);
            if table.contains_key(format!("{module}.weight_zero_point").as_str()) {
                return None;
            }
        }
        _ => return None,
    }
    Some((format, out))
}

/// The format and companions an MLX affine weight reads: `.weight` U32 with its row's K
/// packed `bits` to a word along the last dim, beside `.scales` and `.biases` of the same
/// leading dims with K/`group` columns, both BF16 ([`Format::MlxAffine`]) or both F16
/// ([`Format::MlxAffineF16`]) — one placed tensor whatever the rank (a `switch_mlp` stack's
/// `[E, R, packed]` as much as a linear's `[R, packed]`). A pair of two dtypes fits neither.
fn mlx_layout(
    bits: usize,
    group: usize,
    module: &str,
    dtype: Dtype,
    shape: &[usize],
    table: &BTreeMap<&str, (Dtype, &[usize])>,
) -> Option<(Format, Vec<String>)> {
    if dtype != Dtype::U32 || shape.len() < 2 {
        return None;
    }
    let mut grid = shape.to_vec();
    let k = grid.pop()?.checked_mul(32 / bits)?;
    if k % group != 0 {
        return None;
    }
    grid.push(k / group);
    let find = |suffix: &str| {
        let n = format!("{module}.{suffix}");
        table
            .get(n.as_str())
            .filter(|&&(d, s)| matches!(d, Dtype::Bf16 | Dtype::F16) && s == grid)
            .map(|&(d, _)| (n, d))
    };
    let (scales, scale_dtype) = find("scales")?;
    let (biases, bias_dtype) = find("biases")?;
    let format = match (scale_dtype, bias_dtype) {
        (Dtype::Bf16, Dtype::Bf16) => Format::MlxAffine { bits, group },
        (Dtype::F16, Dtype::F16) => Format::MlxAffineF16 { bits, group },
        _ => return None,
    };
    Some((format, vec![scales, biases]))
}

/// What a refusal of an MLX affine weight adds when its `.scales` and `.biases` are of two
/// dtypes, or of one the format does not read.
fn mlx_dtype_note(module: &str, table: &BTreeMap<&str, (Dtype, &[usize])>) -> Option<String> {
    let dtype = |suffix: &str| {
        table
            .get(format!("{module}.{suffix}").as_str())
            .map(|&(d, _)| d)
    };
    match (dtype("scales")?, dtype("biases")?) {
        (a, b) if a != b => Some(format!(
            "; its scales are {} and its biases {}, and one weight's pair is of one dtype",
            a.name(),
            b.name()
        )),
        (a, _) if !matches!(a, Dtype::Bf16 | Dtype::F16) => Some(format!(
            "; its scales and biases are {}, and only BF16 or F16 are read",
            a.name()
        )),
        _ => None,
    }
}

/// Whether a pattern of `patterns` starts or ends `module`, as transformers matches a module
/// name against `modules_to_not_convert` and `ignored_layers`.
fn skipped(module: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|p| module.starts_with(p.as_str()) || module.ends_with(p.as_str()))
}

/// Refuses a module pattern of `patterns` with a regex character: [`skipped`] takes them
/// literally, as the checkpoint's own config lists whole names.
fn literal_patterns<'a>(
    what: &str,
    patterns: impl IntoIterator<Item = &'a String>,
) -> Result<(), Error> {
    match patterns
        .into_iter()
        .find(|p| p.contains(|c: char| "*?[]()|+^$\\{}".contains(c)))
    {
        Some(p) => Err(bad(format!(
            "{what} module pattern {p} is not one this reader places"
        ))),
        None => Ok(()),
    }
}

/// fnmatch's `*` (any run) and `?` (one character) over the whole of `s`.
fn glob(p: &str, s: &str) -> bool {
    let (p, s): (Vec<char>, Vec<char>) = (p.chars().collect(), s.chars().collect());
    let (mut i, mut j, mut star, mut mark) = (0, 0, None, 0);
    while j < s.len() {
        if i < p.len() && (p[i] == '?' || p[i] == s[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(at) = star {
            i = at + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    p[i..].iter().all(|&c| c == '*')
}

fn str_of<'v>(v: &'v Value, key: &str) -> Result<&'v str, Error> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(format!("{key} is not a string")))
}

fn strings(v: &Value, what: &str) -> Result<Vec<String>, Error> {
    v.as_array()
        .and_then(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect())
        .ok_or_else(|| bad(format!("{what} is not a list of strings")))
}

/// A `[rows, cols]` block size.
fn block_of(v: Option<&Value>) -> Option<Format> {
    let b = v?.as_array()?;
    match b.as_slice() {
        [r, c] => {
            let (rows, cols) = (
                r.as_u64().filter(|&r| r > 0)?,
                c.as_u64().filter(|&c| c > 0)?,
            );
            Some(Format::Fp8Block {
                rows: rows as usize,
                cols: cols as usize,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    /// A config fixture, gunzipped.
    fn fixture(name: &str) -> Option<Value> {
        let path = format!(
            "{}/tests/fixtures/quant-configs/{name}.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let gz = match std::fs::read(&path) {
            Ok(gz) => gz,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => panic!("{path}: {e}"),
        };
        let mut text = Vec::new();
        flate2::read::GzDecoder::new(gz.as_slice())
            .read_to_end(&mut text)
            .unwrap();
        Some(serde_json::from_slice(&text).unwrap())
    }

    /// The config of the checkpoint whose fixtures start with `tag`.
    fn config(tag: &str) -> Result<Config, Error> {
        let config = fixture(&format!("{tag}__config.json")).unwrap();
        Config::parse(
            &config,
            fixture(&format!("{tag}__hf_quant_config.json")).as_ref(),
        )
    }

    const B128: Format = Format::Fp8Block {
        rows: 128,
        cols: 128,
    };

    /// Each checkpoint config declares its modules' formats or is refused by producer or algorithm.
    #[test]
    fn checkpoint_configs_declare_formats_or_refuse_by_name() {
        let lm = "model.language_model.layers";
        type Modules<'a> = &'a [(&'a str, &'a [Format])];
        let cases: &[(&str, Modules)] = &[
            (
                "nvidia__GLM-5.3-Flash-NVFP4",
                &[
                    (&format!("{lm}.3.mlp.experts.0.gate_proj"), &[Format::Nvfp4]),
                    (&format!("{lm}.0.mlp.down_proj"), &[Format::Nvfp4]),
                    (&format!("{lm}.3.self_attn.q_a_proj"), &[]),
                    (&format!("{lm}.3.mlp.shared_experts.up_proj"), &[]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "nvidia__Qwen3.8-Flash-Next-NVFP4",
                &[
                    (&format!("{lm}.0.mlp.experts.7.up_proj"), &[Format::Nvfp4]),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding.shard_0"),
                        &[Format::Fp8Tensor],
                    ),
                    ("mtp.layers.0.mlp.experts.0.gate_proj", &[B128]),
                    (&format!("{lm}.0.linear_attn.in_proj_qkvz"), &[]),
                ],
            ),
            (
                "nvidia__GLM-5.3-Flash-FP4-Local",
                &[
                    (
                        &format!("{lm}.3.mlp.experts.0.gate_proj"),
                        &[Format::Nvfp4Csf32],
                    ),
                    (
                        &format!("{lm}.45.mlp.experts.0.gate_proj"),
                        &[Format::Nvfp4],
                    ),
                    (&format!("{lm}.0.self_attn.q_proj"), &[Format::Mxfp8]),
                    (&format!("{lm}.3.self_attn.q_a_proj"), &[B128]),
                    (&format!("{lm}.3.mlp.shared_experts.up_proj"), &[B128]),
                    (&format!("{lm}.3.self_attn.kv_b_proj"), &[]),
                    ("model.visual.blocks.0.attn.qkv", &[Format::Mxfp8]),
                    ("model.visual.blocks.0.mlp.down_proj", &[Format::Nvfp4]),
                    ("lm_head", &[Format::Mxfp8]),
                ],
            ),
            (
                "nvidia__Qwen3.8-Flash-Next-FP4-Local",
                &[
                    (
                        &format!("{lm}.0.mlp.experts.7.up_proj"),
                        &[Format::Nvfp4Csf],
                    ),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding.shard_0"),
                        &[Format::Nvfp4],
                    ),
                    (&format!("{lm}.0.linear_attn.in_proj_qkv"), &[Format::Mxfp8]),
                    ("mtp.layers.0.mlp.experts.0.gate_proj", &[Format::Nvfp4]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "nvidia__NVIDIA-Nemotron-3-Nano-4B-FP8",
                &[
                    ("backbone.layers.0.mixer.in_proj", &[Format::Fp8Tensor]),
                    ("backbone.layers.11.mixer.in_proj", &[]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "zai-org__GLM-5.3-Flash",
                &[
                    (&format!("{lm}.29.mlp.experts.78.down_proj"), &[B128]),
                    ("lm_head", &[]),
                ],
            ),
            (
                "Qwen__Qwen3.8-Flash-Next-FP8",
                &[
                    (&format!("{lm}.3.mlp.experts.0.down_proj"), &[B128]),
                    (
                        &format!("{lm}.1.ple.ple_embedding.ngram_embedding"),
                        &[B128],
                    ),
                    ("lm_head", &[]),
                ],
            ),
            (
                "deepseek-ai__DeepSeek-V4.1-Flash",
                &[(
                    "layers.21.ffn.experts.0.w1",
                    &[Format::Fp8Block { rows: 32, cols: 32 }, Format::Mxfp4],
                )],
            ),
            (
                "openbmb__MiniCPM-V-4.7-35B-A3B-FP8",
                &[
                    (
                        &format!("{lm}.20.mlp.experts.148.down_proj"),
                        &[Format::Fp8Channel],
                    ),
                    ("model.vision_tower.encoder.layers.0.self_attn.k_proj", &[]),
                ],
            ),
            (
                "openbmb__MiniCPM-V-4.7-35B-A3B-int4",
                &[
                    (
                        &format!("{lm}.36.mlp.experts.212.down_proj"),
                        &[Format::Int4Pack { group: 128 }],
                    ),
                    ("lm_head", &[]),
                ],
            ),
        ];
        for (tag, probes) in cases {
            let c = config(tag).unwrap_or_else(|e| panic!("{tag}: {e}"));
            for (module, want) in *probes {
                assert_eq!(
                    c.declared(module, || Some(ModuleClass::Linear)).0,
                    *want,
                    "{tag}: {module}"
                );
            }
        }
        let mut q =
            fixture("nvidia__NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB__hf_quant_config.json").unwrap();
        let layers = std::mem::take(&mut q["quantization"]["quantized_layers"]);
        for (module, algo) in [
            ("language_model.lm_head", "Q8_0"),
            (
                "language_model.backbone.layers.1.mixer.experts.0.up_proj",
                "IQ2_XXS",
            ),
        ] {
            q["quantization"]["quantized_layers"] = serde_json::json!({ (module): layers[module] });
            let e = Config::parse(&serde_json::json!({}), Some(&q)).unwrap_err();
            assert!(
                e.to_string()
                    .contains(&format!("quant_algo {algo} for {module} is not a format")),
                "{e}"
            );
        }
        for (entry, want) in [
            (
                serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf_64" }),
                "weight_scale_encoding \"csf_64\" for m is not one this reader knows",
            ),
            (
                serde_json::json!({ "quant_algo": "MXFP8", "group_size": 32, "weight_scale_encoding": "csf" }),
                "quant_algo MXFP8 for m: weight_scale_encoding is NVFP4's alone",
            ),
            (
                serde_json::json!({ "quant_algo": "MXFP8", "group_size": 16 }),
                "quant_algo MXFP8 for m is not a format",
            ),
            (
                serde_json::json!({ "quant_algo": "NVFP4", "group_size": 16, "zzz": 1 }),
                "modelopt quantized_layers m.zzz is not a key this converter reads",
            ),
        ] {
            let q = serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION", "quantized_layers": { "m": entry } } });
            let e = Config::parse(&serde_json::json!({}), Some(&q)).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        }
        let int4 = |group: Value| {
            serde_json::json!({ "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": { "g": { "format": "pack-quantized", "targets": ["Linear"],
                    "weights": { "num_bits": 4, "type": "int", "strategy": "group",
                        "group_size": group, "symmetric": true } } } } })
        };
        for (field, class) in [("targets", "LayerNorm"), ("ignore", "Conv2d")] {
            let mut config = int4(128.into());
            if field == "targets" {
                config["quantization_config"]["config_groups"]["g"]["targets"] =
                    serde_json::json!([class]);
            } else {
                config["quantization_config"]["ignore"] = serde_json::json!([class]);
            }
            let e = Config::parse(&config, None).unwrap_err();
            assert!(e.to_string().contains(class), "{e}");
        }
        let fp8 = |key: &str, v: Value| {
            let mut q =
                serde_json::json!({ "quant_method": "fp8", "weight_block_size": [128, 128] });
            q[key] = v;
            serde_json::json!({ "quantization_config": q })
        };
        let modelopt = |group: Value| {
            serde_json::json!({ "quantization_config": { "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION", "quantized_layers": {
                    "m": { "quant_algo": "NVFP4", "group_size": group } } } })
        };
        for (config, want) in [
            (int4(0.into()), "are not a format this reader places"),
            (
                modelopt("16".into()),
                "group_size \"16\" for m is not a count",
            ),
            (
                fp8("fmt", 4.into()),
                "fp8 fmt is not one this reader places",
            ),
            (
                fp8("weight_per_tensor", "no".into()),
                "fp8 weight_per_tensor is not",
            ),
            (fp8("expert_dtype", 4.into()), "fp8 expert_dtype is not"),
            (
                fp8("mxfp4_block_size", "32".into()),
                "fp8 mxfp4_block_size is not",
            ),
            (
                serde_json::json!({ "quantization_config": { "quant_method": "modelopt",
                    "quant_algo": "FP8", "exclude_modules": ["lm_hea[d]"] } }),
                "modelopt exclude pattern lm_hea[d]: a [ class",
            ),
            (
                serde_json::json!({ "quantization_config": { "quant_method": "quark" } }),
                "quant config: Quark checkpoints are not read yet",
            ),
        ] {
            let e = Config::parse(&config, None).unwrap_err();
            assert!(e.to_string().contains(want), "{config}: {e}");
        }
        let unknown =
            serde_json::json!({ "quantization_config": { "quant_method": "gptq", "bits": 4 } });
        let e = Config::parse(&unknown, None).unwrap_err();
        assert!(
            e.to_string()
                .contains("quant_method gptq is not one this reader knows"),
            "{e}"
        );
    }

    /// MiniMax-M3.1-Flash's own `quantization_config`: a module outside its `ignored_layers`
    /// is declared MXFP8 and NVFP4 compressed-tensors, the tensors it stores choosing between
    /// them; the router gates, the head, the embedding and the vision modules are left
    /// unquantized, the head and the embedding by the config's names without the checkpoint's
    /// `language_model.` prefix.
    #[test]
    fn the_minimax_m31_config_declares_mxfp8_and_nvfp4_compressed_tensors() {
        let c = config("MiniMaxAI__MiniMax-M3.1-Flash").unwrap_or_else(|e| panic!("{e}"));
        let lm = "language_model.model.layers";
        let declared = |module: &str| c.declared(module, || None).0;
        let both = [Format::Mxfp8, Format::Nvfp4Ct];
        for module in [
            format!("{lm}.0.self_attn.q_proj"),
            format!("{lm}.0.self_attn.index_k_proj"),
            format!("{lm}.0.mlp.gate_proj"),
            format!("{lm}.3.block_sparse_moe.shared_experts.up_proj"),
            format!("{lm}.3.block_sparse_moe.experts.127.w2"),
            format!("{lm}.59.block_sparse_moe.experts.0.w1"),
        ] {
            assert_eq!(declared(&module), both, "{module}");
        }
        for module in [
            format!("{lm}.3.block_sparse_moe.gate"),
            format!("{lm}.59.block_sparse_moe.gate"),
            "language_model.lm_head".to_string(),
            "language_model.model.embed_tokens".to_string(),
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj".to_string(),
            "multi_modal_projector.linear_1".to_string(),
            "patch_merge_mlp.linear_2".to_string(),
        ] {
            assert_eq!(declared(&module), [], "{module}");
        }
        // Layer 3's router entry is no prefix of layer 30's modules.
        assert_eq!(
            declared(&format!("{lm}.30.block_sparse_moe.shared_experts.up_proj")),
            both
        );
    }

    /// An `mxfp8` config this reader does not place is refused by name when read: a block other
    /// than [1, 32], a scheme other than dynamic, an expert format other than NVFP4
    /// pack-quantized at group 16 (or one half of it), a pattern with a regex character and a key
    /// nothing reads; a config without experts declares MXFP8 alone.
    #[test]
    fn an_mxfp8_config_this_reader_does_not_place_is_refused_by_name() {
        let base = || {
            serde_json::json!({
                "quant_method": "mxfp8", "activation_scheme": "dynamic",
                "weight_block_size": [1, 32], "ignored_layers": ["lm_head"],
                "group_size": 16, "moe_quant_algo": "NVFP4",
                "moe_quant_format": "nvfp4-pack-quantized",
            })
        };
        let parse =
            |q: Value| Config::parse(&serde_json::json!({ "quantization_config": q }), None);
        let c = parse(base()).unwrap();
        assert_eq!(c.declared("m", || None).0, [Format::Mxfp8, Format::Nvfp4Ct]);
        assert!(c.declared("lm_head", || None).0.is_empty());
        for (key, value, want) in [
            (
                "weight_block_size",
                serde_json::json!([1, 64]),
                "mxfp8 weight_block_size [1,64] is not [1, 32]",
            ),
            (
                "weight_block_size",
                serde_json::json!([1, 16]),
                "mxfp8 weight_block_size [1,16] is not [1, 32]",
            ),
            (
                "weight_block_size",
                serde_json::json!([32, 32]),
                "mxfp8 weight_block_size [32,32] is not [1, 32]",
            ),
            (
                "weight_block_size",
                serde_json::json!([128, 128]),
                "mxfp8 weight_block_size [128,128] is not [1, 32]",
            ),
            (
                "weight_block_size",
                serde_json::json!([32]),
                "mxfp8 weight_block_size [32] is not [1, 32]",
            ),
            (
                "weight_block_size",
                serde_json::json!("32"),
                "mxfp8 weight_block_size \"32\" is not [1, 32]",
            ),
            (
                "activation_scheme",
                serde_json::json!("static"),
                "activation_scheme \"static\" is unsupported",
            ),
            (
                "moe_quant_algo",
                serde_json::json!("MXFP4"),
                "moe_quant_algo Some(\"MXFP4\")",
            ),
            (
                "moe_quant_algo",
                serde_json::json!(4),
                "mxfp8 moe_quant_algo 4 is not a string",
            ),
            (
                "moe_quant_format",
                serde_json::json!("pack-quantized"),
                "moe_quant_format Some(\"pack-quantized\")",
            ),
            (
                "group_size",
                serde_json::json!(32),
                "group_size Some(Number(32))",
            ),
            (
                "ignored_layers",
                serde_json::json!(["a.*"]),
                "mxfp8 module pattern a.* is not one this reader places",
            ),
            (
                "ignored_layers",
                serde_json::json!("lm_head"),
                "mxfp8 ignored_layers is not a list of strings",
            ),
            (
                "kv_cache_scheme",
                serde_json::json!({}),
                "mxfp8 quantization_config.kv_cache_scheme is not a key this converter reads",
            ),
        ] {
            let mut q = base();
            q[key] = value;
            let e = parse(q).unwrap_err().to_string();
            assert!(e.contains(want), "{key}: {e}");
        }
        for (gone, want) in [
            ("moe_quant_format", "moe_quant_format None"),
            ("moe_quant_algo", "moe_quant_algo None"),
        ] {
            let mut q = base();
            q.as_object_mut().unwrap().remove(gone);
            let e = parse(q).unwrap_err().to_string();
            assert!(e.contains(want), "{gone}: {e}");
        }
        // A config without experts: the group size has no tensor to describe.
        let mut q = base();
        for key in ["moe_quant_algo", "moe_quant_format"] {
            q.as_object_mut().unwrap().remove(key);
        }
        let e = parse(q.clone()).unwrap_err().to_string();
        assert!(e.contains("group_size Some(Number(16))"), "{e}");
        q.as_object_mut().unwrap().remove("group_size");
        q.as_object_mut().unwrap().remove("weight_block_size");
        let c = parse(q).unwrap();
        assert_eq!(c.declared("m", || None).0, [Format::Mxfp8]);
    }

    /// MiniMax-M3.1-Flash's tensors under its config: an MXFP8 projection is F8_E4M3 with its
    /// `weight_scale_inv` (an E8M0 byte a 32 values), an expert is `weight_packed` with its block
    /// scale, `weight_global_scale` and `input_global_scale` (all three its companions, the input
    /// scale being read with the weight), and the router, bias and head stay as stored. A
    /// quantized weight in an ignored module, an expert stored as `.weight`, a global scale of
    /// another dtype and an orphaned scale are refused by name.
    #[test]
    fn minimax_m31_tensors_place_as_their_config_declares() {
        use Dtype::*;
        let c = config("MiniMaxAI__MiniMax-M3.1-Flash").unwrap();
        let attn = "language_model.model.layers.3.self_attn";
        let moe = "language_model.model.layers.3.block_sparse_moe";
        let expert = |e: u32, w: &str, part: &str| format!("{moe}.experts.{e}.{w}.{part}");
        let mut t: Vec<(String, Dtype, Vec<usize>)> = vec![
            (format!("{attn}.q_proj.weight"), F8E4M3, vec![64, 128]),
            (format!("{attn}.q_proj.weight_scale_inv"), U8, vec![64, 4]),
            (
                format!("{moe}.shared_experts.up_proj.weight"),
                F8E4M3,
                vec![32, 64],
            ),
            (
                format!("{moe}.shared_experts.up_proj.weight_scale_inv"),
                U8,
                vec![32, 2],
            ),
            (format!("{moe}.gate.weight"), F32, vec![128, 64]),
            (format!("{moe}.e_score_correction_bias"), F32, vec![128]),
            ("language_model.lm_head.weight".into(), Bf16, vec![16, 64]),
            (
                "vision_tower.vision_model.fc.weight".into(),
                F8E4M3,
                vec![32, 64],
            ),
            (
                "vision_tower.vision_model.fc.weight_scale_inv".into(),
                U8,
                vec![32, 2],
            ),
        ];
        for e in 0..2 {
            t.extend([
                (expert(e, "w1", "weight_packed"), U8, vec![16, 32]),
                (expert(e, "w1", "weight_scale"), F8E4M3, vec![16, 4]),
                (expert(e, "w1", "weight_global_scale"), F32, vec![]),
                (expert(e, "w1", "input_global_scale"), F32, vec![]),
            ]);
        }
        t.extend([
            // Stored as `.weight`, not `weight_packed`.
            (expert(2, "w1", "weight"), U8, vec![16, 32]),
            (expert(2, "w1", "weight_scale"), F8E4M3, vec![16, 4]),
            (expert(2, "w1", "weight_global_scale"), F32, vec![]),
            // A global scale that is no F32.
            (expert(3, "w2", "weight_packed"), U8, vec![16, 32]),
            (expert(3, "w2", "weight_scale"), F8E4M3, vec![16, 4]),
            (expert(3, "w2", "weight_global_scale"), Bf16, vec![]),
            // No weight.
            (expert(4, "w3", "weight_global_scale"), F32, vec![]),
        ]);
        let p = plan(
            &c,
            t.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            |_| None,
        );
        let got = |n: &str| p.placed.get(n).map(|pl| (pl.format, pl.companions.clone()));
        assert_eq!(
            got(&format!("{attn}.q_proj.weight")).unwrap(),
            (
                Format::Mxfp8,
                vec![format!("{attn}.q_proj.weight_scale_inv")]
            )
        );
        assert_eq!(
            got(&format!("{moe}.shared_experts.up_proj.weight"))
                .unwrap()
                .0,
            Format::Mxfp8
        );
        for e in 0..2 {
            assert_eq!(
                got(&expert(e, "w1", "weight_packed")).unwrap(),
                (
                    Format::Nvfp4Ct,
                    ["weight_scale", "weight_global_scale", "input_global_scale"]
                        .map(|part| expert(e, "w1", part))
                        .to_vec()
                )
            );
        }
        for (name, format) in [
            (format!("{moe}.gate.weight"), Format::F32),
            (format!("{moe}.e_score_correction_bias"), Format::F32),
            ("language_model.lm_head.weight".to_string(), Format::Bf16),
        ] {
            assert_eq!(got(&name).unwrap(), (format, vec![]), "{name}");
        }
        assert!(p.input_scales.is_empty(), "{:?}", p.input_scales);
        let refused = |n: &str, why: &str| assert!(p.refused[n].contains(why), "{n}: {p:?}");
        refused(
            "vision_tower.vision_model.fc.weight",
            "the config leaves vision_tower.vision_model.fc unquantized",
        );
        refused(
            "vision_tower.vision_model.fc.weight_scale_inv",
            "a companion no placed weight reads",
        );
        let both = "fits none of [\"MXFP8\", \"NVFP4-compressed-tensors\"]";
        refused(&expert(2, "w1", "weight"), both);
        refused(&expert(3, "w2", "weight_packed"), both);
        refused(
            &expert(4, "w3", "weight_global_scale"),
            "a companion no placed weight reads",
        );
        for part in ["weight_scale", "weight_global_scale"] {
            refused(&expert(2, "w1", part), "a companion no placed weight reads");
        }
        // Seven weights and other tensors placed, eight scales they claim, nine refused.
        assert_eq!((p.placed.len(), p.refused.len()), (7, 9), "{p:?}");
        assert_eq!(t.len(), 7 + 8 + 9);
    }

    /// The config and the tensors of MiniMax-M3.1-Flash's names through [`plan`] and
    /// [`normalize`]: a dense projection is MXFP8 blocks (the E8M0 byte of `weight_scale_inv`,
    /// then 32 values as stored), and a stack of two experts is NVFP4 whose `.scale` is each
    /// expert's `1 / weight_global_scale`.
    #[test]
    fn minimax_m31_tensors_normalize_through_their_plan() {
        use crate::modelopt::{Plain, Use, normalize};
        use safetensors::{Safetensors, write};
        let c = config("MiniMaxAI__MiniMax-M3.1-Flash").unwrap();
        let attn = "language_model.model.layers.3.self_attn.q_proj";
        let moe = "language_model.model.layers.3.block_sparse_moe.experts";
        let values: Vec<u8> = (0..4 * 64).map(|i| (i * 7 % 120) as u8).collect();
        let exponents = [120u8, 121, 122, 123, 124, 125, 126, 127];
        let codes: Vec<u8> = (0..16 * 32).map(|i| (i * 37 % 251) as u8).collect();
        let scales: Vec<u8> = (0..16 * 4).map(|i| (i % 126) as u8).collect();
        let globals = [8548.7f32, 4248.1];
        let mut list: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
            (
                format!("{attn}.weight"),
                "F8_E4M3",
                vec![4, 64],
                values.clone(),
            ),
            (
                format!("{attn}.weight_scale_inv"),
                "U8",
                vec![4, 2],
                exponents[..8].to_vec(),
            ),
        ];
        for (e, g) in globals.iter().enumerate() {
            let part = |p: &str| format!("{moe}.{e}.w1.{p}");
            list.extend([
                (part("weight_packed"), "U8", vec![16, 32], codes.clone()),
                (part("weight_scale"), "F8_E4M3", vec![16, 4], scales.clone()),
                (
                    part("weight_global_scale"),
                    "F32",
                    vec![],
                    g.to_le_bytes().to_vec(),
                ),
                (
                    part("input_global_scale"),
                    "F32",
                    vec![],
                    1.0f32.to_le_bytes().to_vec(),
                ),
            ]);
        }
        let refs: Vec<(&str, &str, &[usize], &[u8])> = list
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        let st = Safetensors::parse(write(&refs)).unwrap();
        let entries: Vec<(&str, Dtype, Vec<usize>)> = st
            .names()
            .map(|n| {
                let r = st.raw(n).unwrap();
                (n, r.dtype, r.shape.to_vec())
            })
            .collect();
        let p = plan(
            &c,
            entries.iter().map(|(n, d, s)| (*n, *d, s.as_slice())),
            |_| None,
        );
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        let m = Use::Matrix(Plain::Q8_0);

        let q = format!("{attn}.weight");
        let dense = normalize(|n| st.raw(n), &[(&q, &p.placed[&q])], m).unwrap();
        assert_eq!(
            (dense.type_id, &dense.dims),
            (GgmlType::MXFP8.as_u32(), &vec![64, 4])
        );
        let want: Vec<u8> = (0..4 * 2)
            .flat_map(|b| {
                let mut block = vec![exponents[b]];
                block.extend_from_slice(&values[b * 32..][..32]);
                block
            })
            .collect();
        assert_eq!(dense.bytes, want);
        assert!(dense.scale.is_empty());

        let names: Vec<String> = (0..2)
            .map(|e| format!("{moe}.{e}.w1.weight_packed"))
            .collect();
        let sources: Vec<(&str, &Placed)> =
            names.iter().map(|n| (n.as_str(), &p.placed[n])).collect();
        let stack = normalize(|n| st.raw(n), &sources, m).unwrap();
        assert_eq!(
            (stack.type_id, &stack.dims),
            (GgmlType::NVFP4.as_u32(), &vec![64, 16, 2])
        );
        let want: Vec<u32> = globals.iter().map(|g| (1.0f32 / g).to_bits()).collect();
        assert_eq!(
            stack.scale.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            want
        );
    }

    /// Declarations match complete module ancestors, with the nearest ancestor winning.
    #[test]
    fn modelopt_declarations_use_the_nearest_module_ancestor() {
        let c = Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({
                "quantization": { "quant_algo": "MIXED_PRECISION", "quantized_layers": {
                    "model": { "quant_algo": "FP8" },
                    "model.layers.3": { "quant_algo": "NVFP4", "group_size": 16 },
                    "model.layers.3.mlp": { "quant_algo": "FP8" }
                } }
            })),
        )
        .unwrap();
        for (module, format, declaration) in [
            ("model.layers.3", Format::Nvfp4, "model.layers.3"),
            ("model.layers.3.attn.q", Format::Nvfp4, "model.layers.3"),
            (
                "model.layers.3.mlp.experts.511.gate_proj",
                Format::Fp8Tensor,
                "model.layers.3.mlp",
            ),
            ("model.layers.30.mlp", Format::Fp8Tensor, "model"),
        ] {
            assert_eq!(
                c.declared(module, || None),
                (vec![format], Some(declaration))
            );
        }
        assert_eq!(c.declared("model2.layers.3", || None), (vec![], None));
    }

    /// Every tensor of a table gets one format or a refusal naming it: each format's layout is
    /// matched with its companions, a tensor scale is shared from the module the config names,
    /// input scales are reported, and a scale with no weight, a quantized weight the config leaves
    /// unquantized and a dtype outside every format are refused.
    #[test]
    fn every_tensor_gets_one_format_or_a_refusal_by_name() {
        use Dtype::*;
        let c = Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": null,
                "quantized_layers": {
                    "e": { "quant_algo": "NVFP4", "group_size": 16 },
                    "ple": { "quant_algo": "FP8" },
                    "mtp": { "quant_algo": "FP8_BLOCK_SCALES", "group_size": 128 },
                },
                "exclude_modules": ["attn*"],
            }})),
        )
        .unwrap();
        let t: &[(&str, Dtype, &[usize])] = &[
            ("e.0.weight", U8, &[256, 64]),
            ("e.0.weight_scale", F8E4M3, &[256, 8]),
            ("e.0.weight_scale_2", F32, &[]),
            ("e.0.input_scale", F32, &[]),
            ("ple.shard_0.weight", F8E4M3, &[10, 4]),
            ("ple.shard_1.weight", F8E4M3, &[10, 4]),
            ("ple.weight_scale", Bf16, &[1]),
            ("mtp.0.weight", F8E4M3, &[640, 2560]),
            ("mtp.0.weight_scale_inv", Bf16, &[5, 20]),
            ("attn.q.weight", F8E4M3, &[8, 8]),
            ("attn.q.weight_scale", F32, &[]),
            ("norm.weight", Bf16, &[8]),
            ("A_log", F32, &[8]),
            ("lone.weight_scale_2", F32, &[]),
            ("e.1.weight", U8, &[256, 64]),
            ("e.1.weight_scale", F8E4M3, &[256, 4]),
            ("ple.ids", I64, &[3]),
            ("e.2.weight", U8, &[0, 9_223_372_036_854_775_840]),
        ];
        let p = plan(&c, t.iter().copied(), |_| None);
        let got = |n: &str| p.placed.get(n).map(|pl| (pl.format, pl.companions.clone()));
        assert_eq!(
            got("e.0.weight").unwrap(),
            (
                Format::Nvfp4,
                vec!["e.0.weight_scale".into(), "e.0.weight_scale_2".into()]
            )
        );
        assert_eq!(
            got("ple.shard_1.weight").unwrap(),
            (Format::Fp8Tensor, vec!["ple.weight_scale".into()])
        );
        assert_eq!(
            got("mtp.0.weight").unwrap(),
            (B128, vec!["mtp.0.weight_scale_inv".into()])
        );
        assert_eq!(got("norm.weight").unwrap().0, Format::Bf16);
        assert_eq!(got("A_log").unwrap().0, Format::F32);
        assert_eq!(p.input_scales, ["e.0.input_scale"]);
        let refused = |n: &str, why: &str| assert!(p.refused[n].contains(why), "{n}: {p:?}");
        refused("attn.q.weight", "the config leaves attn.q unquantized");
        refused("attn.q.weight_scale", "a companion no placed weight reads");
        refused("lone.weight_scale_2", "a companion no placed weight reads");
        refused("e.1.weight", "fits none of [\"NVFP4-ModelOpt\"]");
        refused("ple.ids", "I64 outside a quantized weight");
        refused("e.2.weight", "fits none of [\"NVFP4-ModelOpt\"]");
        assert_eq!(p.placed.len() + p.refused.len() + 5, t.len());

        let ct = Config::parse(
            &serde_json::json!({ "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": { "g": { "format": "pack-quantized", "targets": ["Linear"],
                    "weights": { "num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "symmetric": true } } },
                "ignore": ["lm_head", "re:.*\\.mlp\\.gate$"],
            }}),
            None,
        )
        .unwrap();
        let t: &[(&str, Dtype, &[usize])] = &[
            ("x.weight_packed", I32, &[2048, 64]),
            ("x.weight_scale", Bf16, &[2048, 4]),
            ("x.weight_shape", I64, &[2]),
            ("l.mlp.gate.weight_packed", I32, &[8, 64]),
            ("y.weight_packed", I32, &[0, usize::MAX / 4]),
        ];
        let p = plan(&ct, t.iter().copied(), |_| Some(ModuleClass::Linear));
        assert_eq!(
            p.placed["x.weight_packed"].format,
            Format::Int4Pack { group: 128 }
        );
        assert!(p.refused["l.mlp.gate.weight_packed"].contains("leaves l.mlp.gate unquantized"));
        assert!(p.refused["y.weight_packed"].contains("fits none of"));
        let wide: &[usize] = &[0, usize::MAX / 2 + 1];
        for (format, shape) in [
            (Format::Mxfp4, wide),
            (Format::Nvfp4Csf32, &[usize::MAX / 2 + 1, 16]),
        ] {
            assert_eq!(
                layout(format, "m.weight", "m", None, U8, shape, &BTreeMap::new()),
                None
            );
        }
    }

    #[test]
    fn compressed_tensor_classes_do_not_quantize_each_others_weights() {
        use Dtype::*;
        let tensors: &[(&str, Dtype, &[usize])] = &[
            ("model.embed_tokens.weight_packed", I32, &[8, 16]),
            ("model.embed_tokens.weight_scale", Bf16, &[8, 1]),
            ("model.embed_tokens.weight_shape", I64, &[2]),
            (
                "model.layers.0.self_attn.q_proj.weight_packed",
                I32,
                &[8, 16],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight_scale",
                Bf16,
                &[8, 1],
            ),
            ("model.layers.0.self_attn.q_proj.weight_shape", I64, &[2]),
        ];
        for (target, kept, refused) in [
            (
                "Embedding",
                "model.embed_tokens",
                "model.layers.0.self_attn.q_proj",
            ),
            (
                "Linear",
                "model.layers.0.self_attn.q_proj",
                "model.embed_tokens",
            ),
        ] {
            let config = Config::parse(
                &serde_json::json!({"quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"g": {
                        "format": "pack-quantized", "targets": [target],
                        "weights": {"num_bits": 4, "type": "int", "strategy": "group",
                            "group_size": 128, "symmetric": true}
                    }}
                }}),
                None,
            )
            .unwrap();
            let placed = plan(&config, tensors.iter().copied(), |module| match module {
                "model.embed_tokens" => Some(ModuleClass::Embedding),
                "model.layers.0.self_attn.q_proj" => Some(ModuleClass::Linear),
                _ => None,
            });
            assert_eq!(
                placed
                    .placed
                    .get(&format!("{kept}.weight_packed"))
                    .map(|p| p.format),
                Some(Format::Int4Pack { group: 128 }),
                "{target}: {placed:?}"
            );
            assert!(
                placed.refused[&format!("{refused}.weight_packed")].contains("unquantized"),
                "{target}: {placed:?}"
            );
        }
    }

    /// Real headers: every tensor of nvidia/GLM-5.3-Flash-NVFP4's layers 0 (dense) and 3 (288
    /// experts) and its unlayered tensors under its config, and a deepseek-ai/DeepSeek-V4.1-Flash
    /// shard under its own, each tensor placed and none refused.
    #[test]
    fn real_checkpoint_headers_place_under_their_configs() {
        let header = |name: &str| -> serde_json::Map<String, Value> {
            let gz = std::fs::read(format!(
                "{}/tests/fixtures/checkpoint-headers/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let mut text = Vec::new();
            flate2::read::GzDecoder::new(gz.as_slice())
                .read_to_end(&mut text)
                .unwrap();
            let mut h: serde_json::Map<String, Value> = serde_json::from_slice(&text).unwrap();
            h.remove("__metadata__");
            h
        };
        let place = |tag: &str, files: &[&str]| {
            let heads: Vec<_> = files.iter().map(|f| header(f)).collect();
            let entries: Vec<(String, Dtype, Vec<usize>)> = heads
                .iter()
                .flatten()
                .map(|(n, e)| {
                    let shape = e["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize)
                        .collect();
                    (
                        n.clone(),
                        Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                        shape,
                    )
                })
                .collect();
            let p = plan(
                &config(tag).unwrap(),
                entries
                    .iter()
                    .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
                |_| None,
            );
            let mut count = BTreeMap::new();
            for pl in p.placed.values() {
                *count.entry(pl.format.to_string()).or_insert(0) += 1;
            }
            (count.into_iter().collect::<Vec<_>>(), p)
        };
        let (count, p) = place(
            "nvidia__GLM-5.3-Flash-NVFP4",
            &["glm-5.3-flash-nvfp4-layers-0-3.tensors.json.gz"],
        );
        let want = [("BF16", 47), ("F32", 6), ("NVFP4-ModelOpt", 867)];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(p.input_scales.len(), 867);
        // Layers 0 (linear attention, dense MLP), 3 (latent attention, experts) and 4 (linear
        // attention, experts): each expert's five csf32 parts and its two tensor scales are its
        // companions.
        let (count, p) = place(
            "nvidia__GLM-5.3-Flash-FP4-Local",
            &["glm-5.3-flash-fp4-local-layers-0-3-4.tensors.json.gz"],
        );
        let want = [
            ("BF16", 40),
            ("F32", 22),
            ("FP8-block 128x128", 10),
            ("MXFP8", 12),
            ("NVFP4-csf32", 1728),
        ];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(p.input_scales.len(), 1728);
        let expert = &p.placed["model.language_model.layers.3.mlp.experts.0.down_proj.weight"];
        assert_eq!(expert.companions.len(), 6, "{:?}", expert.companions);
        // One of Qwen3.8's 128 PLE parts: its tensor scale is the one the declared table holds.
        let (count, p) = place(
            "nvidia__Qwen3.8-Flash-Next-FP4-Local",
            &["qwen3.8-flash-next-fp4-local-ple-part-0.tensors.json.gz"],
        );
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(count, [("NVFP4-ModelOpt".to_string(), 1)]);
        let ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        assert_eq!(
            p.placed[&format!("{ple}.shard_0.weight")].companions,
            [
                format!("{ple}.shard_0.weight_scale"),
                format!("{ple}.weight_scale_2")
            ]
        );
        let (count, p) = place(
            "deepseek-ai__DeepSeek-V4.1-Flash",
            &["deepseek-v4.1-flash-model-00024-of-00048.header.json.gz"],
        );
        let want = [
            ("BF16", 5),
            ("F32", 9),
            ("FP8-block 32x32", 8),
            ("MXFP4-OCP", 1152),
        ];
        assert_eq!(count, want.map(|(f, n)| (f.to_string(), n)));
        assert!(p.refused.is_empty(), "{:?}", p.refused);
    }

    /// Each format's held blocks and tensor scales, including shape and byte-count refusals.
    #[test]
    fn each_format_holds_its_values_and_plain_scales() {
        let expert = [2048, 4096, 288];
        for f in [
            Format::Nvfp4,
            Format::Nvfp4Csf,
            Format::Nvfp4Csf32,
            Format::Nvfp4Ct,
        ] {
            assert_eq!(
                f.held(&expert),
                Some(Held {
                    whole_bytes: 4 * 288,
                    block: [64, 1],
                })
            );
        }
        assert_eq!(Format::Nvfp4.held(&[48, 16]), None);
        assert_eq!(
            Format::Mxfp8.held(&[4096, 8192]),
            Some(Held {
                whole_bytes: 0,
                block: [32, 1],
            })
        );
        assert_eq!(
            B128.held(&[16384, 4096]),
            Some(Held {
                whole_bytes: 0,
                block: [128, 1],
            })
        );
        for dims in [
            &[][..],
            &[0, 1],
            &[128, 0],
            &[128, 1, 0],
            &[96, 4096],
            &[128, u64::MAX],
            &[128, 1, u64::MAX, 2],
        ] {
            assert_eq!(B128.held(dims), None, "{dims:?}");
        }
        for ranks in [1, 2, 4] {
            assert!(B128.held(&[128, 1, ranks]).is_some());
        }
        for (rows, cols) in [(0, 128), (128, 0), (0, 0)] {
            assert_eq!(Format::Fp8Block { rows, cols }.held(&[128, 1]), None);
        }
        // PIN(2026-10-21): one E8M0 a 32 by 32 block is held as MXFP8, the square's scale on each
        // of its rows, where it counted one F32 a square and cut by squares.
        let fp8 = Format::Fp8Block { rows: 32, cols: 32 };
        assert_eq!(
            fp8.held(&[16384, 4096]),
            Some(Held {
                whole_bytes: 0,
                block: [32, 1],
            })
        );
        for f in [Format::Fp8Channel, Format::Fp8Tensor] {
            assert_eq!(
                f.held(&[16384, 4096]),
                Some(Held {
                    whole_bytes: 0,
                    block: [128, 1],
                })
            );
        }
        assert_eq!(Format::Fp8Block { rows: 32, cols: 48 }.held(&[96, 4]), None);
        // PIN(2026-10-21): resident byte counts belong to CardFormat; Held keeps cut units and scales.
        for (f, dims) in [(Format::Bf16, &[512, 256, 64][..]), (Format::F32, &[64])] {
            assert_eq!(
                f.held(dims),
                Some(Held {
                    whole_bytes: 0,
                    block: [1, 1],
                })
            );
        }
        assert_eq!(Format::Fp8Tensor.held(&[64, 64]), None);
        assert_eq!(Format::Bf16.held(&[0, 8192]), None);
        assert_eq!(Format::Bf16.held(&[64, 0]), None);
        assert_eq!(Format::Bf16.held(&[u64::MAX / 2, 4]), None);
        assert_eq!(Format::Mxfp8.held(&[1 << 40, 1 << 40]), None);
        assert_eq!(Format::Nvfp4Csf32.logical(&[2048, 1024]), [2048, 2048]);
        assert_eq!(Format::Nvfp4Ct.logical(&[2048, 1024]), [2048, 2048]);
        assert_eq!(Format::Mxfp8.logical(&[2048, 1024]), [2048, 1024]);
    }

    /// The oracle on real checkpoint rows (`tests/fixtures/fp8_scales.py`): each F8_E4M3 weight
    /// of the fixture, placed by its own checkpoint's config (one scale a row for MiniCPM-V 4.7
    /// FP8's routed experts, one a tensor for Nemotron 3 Nano 4B FP8's out_proj, one E8M0 a 32 by
    /// 32 block for DeepSeek-V4.1-Flash's shared expert), normalizes to its card type, the two
    /// experts as one stack, and each row `dequant_row` gives has the f32 bits the producer's
    /// rule gives in `fp8_scales.py ref`. Nano's in_proj, 3136 values a row, is not whole
    /// FP8_B128 blocks and is refused by name.
    #[test]
    fn real_fp8_rows_decode_to_their_producers_rule() {
        use gguf::quant::dequant_row;
        use safetensors::Safetensors;
        use sha2::{Digest, Sha256};

        use crate::modelopt::{Plain, Use, normalize};

        let st =
            Safetensors::parse(include_bytes!("../tests/fixtures/fp8-scales.safetensors").to_vec())
                .unwrap();
        let reference: Value =
            serde_json::from_str(include_str!("../tests/fixtures/fp8-scales-ref.json")).unwrap();
        let entries: Vec<(&str, Dtype, Vec<usize>)> = st
            .names()
            .map(|n| {
                let r = st.raw(n).unwrap();
                (n, r.dtype, r.shape.to_vec())
            })
            .collect();
        let placed = |tag: &str| {
            plan(
                &config(tag).unwrap(),
                entries.iter().map(|(n, d, s)| (*n, *d, s.as_slice())),
                |module| {
                    matches!(
                        module,
                        "model.language_model.layers.20.mlp.experts.148.down_proj"
                            | "model.language_model.layers.20.mlp.experts.149.down_proj"
                    )
                    .then_some(ModuleClass::Linear)
                },
            )
        };
        let m = Use::Matrix(Plain::Q8_0);
        let expert =
            |e: u32| format!("model.language_model.layers.20.mlp.experts.{e}.down_proj.weight");
        let nano = "nvidia__NVIDIA-Nemotron-3-Nano-4B-FP8";
        for (tag, weights, format, ty) in [
            (
                "openbmb__MiniCPM-V-4.7-35B-A3B-FP8",
                vec![expert(148), expert(149)],
                Format::Fp8Channel,
                GgmlType::FP8_B128,
            ),
            (
                nano,
                vec!["backbone.layers.0.mixer.out_proj.weight".to_string()],
                Format::Fp8Tensor,
                GgmlType::FP8_B128,
            ),
            (
                "deepseek-ai__DeepSeek-V4.1-Flash",
                vec!["layers.21.ffn.shared_experts.w2.weight".to_string()],
                Format::Fp8Block { rows: 32, cols: 32 },
                GgmlType::MXFP8,
            ),
        ] {
            let p = placed(tag);
            let sources: Vec<(&str, &Placed)> =
                weights.iter().map(|w| (w.as_str(), &p.placed[w])).collect();
            assert!(
                sources.iter().all(|(_, pl)| pl.format == format),
                "{tag}: {sources:?}"
            );
            let n = normalize(|n| st.raw(n), &sources, m).unwrap();
            assert_eq!(n.type_id, ty.as_u32(), "{tag}");
            let k = n.dims[0] as usize;
            let rows = n.dims[1..].iter().product::<u64>() as usize;
            let got: Vec<String> = n
                .bytes
                .chunks_exact(n.bytes.len() / rows)
                .map(|row| {
                    let mut v = vec![0f32; k];
                    dequant_row(ty, row, &mut v).unwrap();
                    let bits: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    format!("{:x}", Sha256::digest(&bits))
                })
                .collect();
            let want: Vec<&str> = weights
                .iter()
                .flat_map(|w| reference["weights"][w].as_array().unwrap())
                .map(|d| d.as_str().unwrap())
                .collect();
            assert_eq!(got, want, "{tag}");
        }
        let w = "backbone.layers.0.mixer.in_proj.weight";
        let p = placed(nano);
        assert_eq!(p.placed[w].format, Format::Fp8Tensor);
        let e = normalize(|n| st.raw(n), &[(w, &p.placed[w])], m).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!(
                "{w}: FP8-tensor F8_E4M3 [1, 3136] (3136 bytes), not F8_E4M3 rows of whole 128-value blocks"
            )
        );
    }

    /// The tensors of a committed header file of `tests/fixtures/checkpoint-headers`:
    /// name, dtype and shape.
    fn header_entries(file: &str) -> Vec<(String, Dtype, Vec<usize>)> {
        let gz = std::fs::read(format!(
            "{}/tests/fixtures/checkpoint-headers/{file}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let mut text = Vec::new();
        flate2::read::GzDecoder::new(gz.as_slice())
            .read_to_end(&mut text)
            .unwrap();
        let header: serde_json::Map<String, Value> = serde_json::from_slice(&text).unwrap();
        header
            .iter()
            .map(|(n, e)| {
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    e["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize)
                        .collect(),
                )
            })
            .collect()
    }

    fn tensors(
        entries: &[(String, Dtype, Vec<usize>)],
    ) -> impl Iterator<Item = (&str, Dtype, &[usize])> {
        entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
    }

    /// MLX's own `quantization` block (mlx-community's Qwen3.6 4-bit, its
    /// config.json and the dtype and shape of every tensor of layers 0 and 3,
    /// the unlayered text tensors, `lm_head` and one `vision_tower` pair):
    /// every text-model weight is placed — the 4-bit default, the 8-bit router
    /// and shared-expert gate by their per-module entries, the pre-stacked
    /// `switch_mlp` as one 3-D tensor — the BF16 tensors (the vision tower's
    /// among them) land plain and out of the quantization, and nothing is
    /// refused. A `mode` that is not affine, a width or group the blocks do
    /// not have, a second producer beside the block, a divergent
    /// `quantization_config` copy, a `.weight` without its
    /// `.scales`/`.biases` and a shape that does not match the group are all
    /// refused by name.
    #[test]
    fn mlx_affine_places_the_real_headers_or_refuses_by_name() {
        let tag = "mlx-community__Qwen3.6-35B-A3B-4bit";
        let real = fixture(&format!("{tag}__config.json")).unwrap();
        let alone = |real: &Value| {
            let mut c = real.clone();
            c.as_object_mut().unwrap().remove("quantization_config");
            c
        };
        let c = Config::parse(&real, None).unwrap();
        let entries = header_entries("qwen3.6-mlx-4bit-layers-0-3.tensors.json.gz");
        let p = plan(&c, tensors(&entries), |_| None);
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        let mut count = BTreeMap::new();
        for pl in p.placed.values() {
            *count.entry(pl.format.to_string()).or_insert(0) += 1;
        }
        let want = [
            ("BF16", 13),
            ("MLX-affine 4-bit group 64", 23),
            ("MLX-affine 8-bit group 64", 4),
        ];
        assert_eq!(
            count.into_iter().collect::<Vec<_>>(),
            want.map(|(f, n)| (f.to_string(), n))
        );
        let l0 = "language_model.model.layers.0";
        assert_eq!(
            p.placed[&format!("{l0}.mlp.gate.weight")].format,
            Format::MlxAffine { bits: 8, group: 64 }
        );
        assert_eq!(
            c.declared(&format!("{l0}.mlp.gate"), || None).0,
            [Format::MlxAffine { bits: 8, group: 64 }]
        );
        assert_eq!(
            c.declared(&format!("{l0}.mlp.shared_expert_gate"), || None)
                .0,
            [Format::MlxAffine { bits: 8, group: 64 }]
        );
        assert_eq!(
            c.declared(&format!("{l0}.linear_attn.out_proj"), || None).0,
            [Format::MlxAffine { bits: 4, group: 64 }]
        );
        // The pre-stacked expert tensor: [E, R, packed] with [E, R, K/group]
        // scales and biases, one placed tensor.
        let down = &p.placed[&format!("{l0}.mlp.switch_mlp.down_proj.weight")];
        assert_eq!(down.format, Format::MlxAffine { bits: 4, group: 64 });
        assert_eq!(
            down.companions,
            [
                format!("{l0}.mlp.switch_mlp.down_proj.scales"),
                format!("{l0}.mlp.switch_mlp.down_proj.biases")
            ]
        );
        assert_eq!(
            Format::MlxAffine { bits: 4, group: 64 }.logical(&[256, 2048, 64]),
            [256, 2048, 512]
        );
        assert_eq!(
            Format::MlxAffine { bits: 8, group: 64 }.card_type(),
            Some(GgmlType::MLX_A8)
        );
        assert_eq!(Format::MlxAffine { bits: 3, group: 64 }.card_type(), None);
        // The vision tower's BF16 tensors land plain, out of the quantization.
        assert_eq!(
            p.placed["vision_tower.patch_embed.proj.weight"].format,
            Format::Bf16
        );
        // A `.weight` without its companions, and one whose scales do not
        // match the group, are refused by name under the block's config.
        let mut dropped = entries.clone();
        dropped.retain(|(n, _, _)| n != &format!("{l0}.mlp.gate.scales"));
        let p = plan(&c, tensors(&dropped), |_| None);
        assert!(
            p.refused[&format!("{l0}.mlp.gate.weight")]
                .contains("fits none of [\"MLX-affine 8-bit group 64\"]"),
            "{:?}",
            p.refused
        );
        assert!(
            p.refused[&format!("{l0}.mlp.gate.biases")]
                .contains("a companion no placed weight reads"),
            "{:?}",
            p.refused
        );
        let mut narrow = entries;
        let at = narrow
            .iter()
            .position(|(n, _, _)| n == &format!("{l0}.mlp.gate.scales"))
            .unwrap();
        narrow[at].2 = vec![256, 31];
        let p = plan(&c, tensors(&narrow), |_| None);
        assert!(
            p.refused[&format!("{l0}.mlp.gate.weight")]
                .contains("fits none of [\"MLX-affine 8-bit group 64\"]"),
            "{:?}",
            p.refused
        );
        // The block's own refusals, each from the real config with one value
        // changed (the copy beside it removed, so the block stands alone).
        let refuse = |key: &str, v: Value, want: &str| {
            let mut config = alone(&real);
            config["quantization"][key] = v;
            let e = Config::parse(&config, None).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        };
        refuse(
            "mode",
            "affine2".into(),
            "quantization.mode \"affine2\" is unsupported",
        );
        refuse(
            "bits",
            3.into(),
            "quantization bits 3 for the block is not one this reader places",
        );
        refuse(
            "group_size",
            32.into(),
            "quantization group_size 32 for the block is not one this reader places",
        );
        let mut module = alone(&real);
        module["quantization"][&format!("{l0}.mlp.gate")]["bits"] = 2.into();
        let e = Config::parse(&module, None).unwrap_err();
        assert!(
            e.to_string()
                .contains(&format!("quantization bits 2 for {l0}.mlp.gate")),
            "{e}"
        );
        let mut unknown = alone(&real);
        unknown["quantization"][&format!("{l0}.mlp.gate")]["zzz"] = 1.into();
        let e = Config::parse(&unknown, None).unwrap_err();
        assert!(
            e.to_string().contains(&format!(
                "quantization {l0}.mlp.gate.zzz is not a key this converter reads"
            )),
            "{e}"
        );
        // A divergent copy beside the block, and a second producer, are
        // refused; the verbatim copy the real config carries parsed above.
        let mut divergent = real.clone();
        divergent["quantization_config"]["bits"] = 8.into();
        let e = Config::parse(&divergent, None).unwrap_err();
        assert!(
            e.to_string()
                .contains("quantization_config beside quantization is not its copy"),
            "{e}"
        );
        let e = Config::parse(
            &alone(&real),
            Some(&serde_json::json!({ "quantization": { "quant_algo": "FP8" } })),
        )
        .unwrap_err();
        assert!(
            e.to_string()
                .contains("a quantization block beside ModelOpt's hf_quant_config.json"),
            "{e}"
        );
        let mut fp8 = alone(&real);
        fp8["quantization_config"] = serde_json::json!({
            "quant_method": "fp8", "weight_block_size": [128, 128]
        });
        let e = Config::parse(&fp8, None).unwrap_err();
        assert!(
            e.to_string()
                .contains("a quantization block beside quantization_config's quant_method fp8"),
            "{e}"
        );
    }

    /// mlx-community's Qwen3-30B-A3B-Instruct-2507-4bit carries a `quantization` block that names
    /// no `mode`: it is the affine form. Its config and the dtype
    /// and shape of all 1,351 tensors place every one — the 4-bit default for the embedding, the
    /// head, the attention and the stacked experts, the 8-bit router by its per-module entries,
    /// the norm gammas plain — and a `mode` that is present still has to be `affine`.
    #[test]
    fn an_absent_mode_is_the_affine_form() {
        let real = fixture("mlx-community__Qwen3-30B-A3B-Instruct-2507-4bit__config.json").unwrap();
        assert!(real["quantization"].get("mode").is_none());
        let c = Config::parse(&real, None).unwrap();
        let entries = header_entries("qwen3-30b-a3b-instruct-2507-mlx-4bit.tensors.json.gz");
        assert_eq!(entries.len(), 1351);
        let p = plan(&c, tensors(&entries), |_| None);
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        let mut count = BTreeMap::new();
        for pl in p.placed.values() {
            *count.entry(pl.format.to_string()).or_insert(0) += 1;
        }
        assert_eq!(
            count.into_iter().collect::<Vec<_>>(),
            [
                ("BF16", 193),
                ("MLX-affine 4-bit group 64", 338),
                ("MLX-affine 8-bit group 64", 48),
            ]
            .map(|(f, n)| (f.to_string(), n))
        );
        assert_eq!(
            p.placed["model.layers.47.mlp.gate.weight"].format,
            Format::MlxAffine { bits: 8, group: 64 }
        );
        assert_eq!(
            p.placed["model.layers.47.mlp.switch_mlp.down_proj.weight"].format,
            Format::MlxAffine { bits: 4, group: 64 }
        );
        let mut other = real;
        other["quantization"]["mode"] = "affine2".into();
        other.as_object_mut().unwrap().remove("quantization_config");
        let e = Config::parse(&other, None).unwrap_err();
        assert!(
            e.to_string()
                .contains("quantization.mode \"affine2\" is unsupported"),
            "{e}"
        );
    }

    /// mlx-lm writes `mode` into every per-module entry of a fine-grained config. An `affine` one
    /// is the block's rule and parses to the layers an entry without it does; another mode is
    /// refused naming the module.
    #[test]
    fn a_per_module_mode_is_the_blocks_affine_form() {
        let gate = "model.layers.0.mlp.gate";
        let config = |mode: Option<&str>| {
            let mut entry = serde_json::json!({ "group_size": 64, "bits": 8 });
            if let Some(mode) = mode {
                entry["mode"] = mode.into();
            }
            let mut q = serde_json::json!({ "group_size": 64, "bits": 4, "mode": "affine" });
            q[gate] = entry;
            serde_json::json!({ "quantization": q })
        };
        let with = Config::parse(&config(Some("affine")), None).unwrap();
        let without = Config::parse(&config(None), None).unwrap();
        for module in [gate, "model.layers.0.self_attn.q_proj"] {
            assert_eq!(
                with.declared(module, || None).0,
                without.declared(module, || None).0,
                "{module}"
            );
        }
        assert_eq!(
            with.declared(gate, || None).0,
            [Format::MlxAffine { bits: 8, group: 64 }]
        );
        let e = Config::parse(&config(Some("mxfp4")), None).unwrap_err();
        assert!(
            e.to_string().contains(&format!(
                "quantization {gate}.mode \"mxfp4\" is unsupported"
            )),
            "{e}"
        );
    }

    /// mlx-lm writes `"<module>": false` for a module it left unquantized, a norm's among them.
    /// The module is declared no format: its stored floats are placed plain, and a quantized
    /// tensor or a scale beside them is refused by name. A value other than an object or `false`
    /// in an entry's place is refused too.
    #[test]
    fn a_false_entry_leaves_its_module_unquantized() {
        let norm = "model.layers.0.self_attn.q_norm";
        let proj = "model.layers.0.self_attn.q_proj";
        let config = |entries: &[(&str, Value)]| {
            let mut q = serde_json::json!({ "group_size": 64, "bits": 4 });
            for (module, entry) in entries {
                q[*module] = entry.clone();
            }
            serde_json::json!({ "quantization": q })
        };
        let entry = |name: String, dtype: Dtype, shape: &[usize]| (name, dtype, shape.to_vec());
        let stored = vec![
            entry(format!("{norm}.weight"), Dtype::Bf16, &[128]),
            entry(format!("{proj}.weight"), Dtype::U32, &[256, 32]),
            entry(format!("{proj}.scales"), Dtype::Bf16, &[256, 4]),
            entry(format!("{proj}.biases"), Dtype::Bf16, &[256, 4]),
        ];
        let off = Value::Bool(false);

        // The `false` module reads as unquantized, its BF16 weight is placed plain, and the
        // quantized module beside it keeps the block's format.
        let c = Config::parse(&config(&[(norm, off.clone())]), None).unwrap();
        assert!(
            c.declared(norm, || None).0.is_empty(),
            "a false entry declares its module no format"
        );
        let block = Format::MlxAffine { bits: 4, group: 64 };
        assert_eq!(c.declared(proj, || None).0, [block]);
        let p = plan(&c, tensors(&stored), |_| None);
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        assert_eq!(p.placed[&format!("{norm}.weight")].format, Format::Bf16);
        assert_eq!(p.placed[&format!("{proj}.weight")].format, block);

        // A quantized weight under a `false` entry is refused with its scales.
        let c = Config::parse(&config(&[(norm, off.clone()), (proj, off.clone())]), None).unwrap();
        let p = plan(&c, tensors(&stored), |_| None);
        assert!(
            p.refused[&format!("{proj}.weight")]
                .contains(&format!("the config leaves {proj} unquantized")),
            "{:?}",
            p.refused
        );
        assert!(
            p.refused[&format!("{proj}.scales")].contains("a companion no placed weight reads"),
            "{:?}",
            p.refused
        );

        // A scale beside a float weight of a `false` module is refused, the weight stays plain.
        let c = Config::parse(&config(&[(norm, off)]), None).unwrap();
        let strays = vec![
            entry(format!("{norm}.weight"), Dtype::Bf16, &[128]),
            entry(format!("{norm}.scales"), Dtype::Bf16, &[128, 2]),
        ];
        let p = plan(&c, tensors(&strays), |_| None);
        assert_eq!(p.placed[&format!("{norm}.weight")].format, Format::Bf16);
        assert!(
            p.refused[&format!("{norm}.scales")].contains("a companion no placed weight reads"),
            "{:?}",
            p.refused
        );

        // Any other value in an entry's place is refused by the module's name.
        for other in [Value::Bool(true), 3.into(), "no".into()] {
            let e = Config::parse(&config(&[(norm, other.clone())]), None).unwrap_err();
            assert!(
                e.to_string().contains(&format!(
                    "quantization {norm} {other} is neither a per-module object, false nor one \
                     of the block's keys"
                )),
                "{e}"
            );
        }
    }

    /// An MLX affine weight whose `.scales` and `.biases` are F16 is placed as
    /// [`Format::MlxAffineF16`] (card type MLX_A4H or MLX_A8H) beside BF16 ones of the same
    /// checkpoint placed as [`Format::MlxAffine`]; a weight whose pair is of two dtypes, or of one
    /// the format does not read, is refused by name with the dtypes; an F16 tensor outside a
    /// quantized weight is placed as [`Format::F16`].
    #[test]
    fn mlx_affine_places_f16_companions_as_their_own_format() {
        let c = Config::parse(
            &serde_json::json!({ "quantization": { "group_size": 64, "bits": 4 } }),
            None,
        )
        .unwrap();
        let entry =
            |name: &str, dtype: Dtype, shape: &[usize]| (name.to_string(), dtype, shape.to_vec());
        let linear = |layer: usize, scales: Dtype, biases: Dtype| {
            let m = format!("m.layers.{layer}.q");
            [
                entry(&format!("{m}.weight"), Dtype::U32, &[256, 32]),
                entry(&format!("{m}.scales"), scales, &[256, 4]),
                entry(&format!("{m}.biases"), biases, &[256, 4]),
            ]
        };
        let mut stored = vec![entry("m.layers.0.norm.weight", Dtype::F16, &[128])];
        stored.extend(linear(0, Dtype::F16, Dtype::F16));
        stored.extend(linear(1, Dtype::Bf16, Dtype::Bf16));
        stored.extend(linear(2, Dtype::F16, Dtype::Bf16));
        stored.extend(linear(3, Dtype::F32, Dtype::F32));
        let p = plan(&c, tensors(&stored), |_| None);
        let half = Format::MlxAffineF16 { bits: 4, group: 64 };
        let placed = &p.placed["m.layers.0.q.weight"];
        assert_eq!(placed.format, half);
        assert_eq!(
            placed.companions,
            ["m.layers.0.q.scales", "m.layers.0.q.biases"]
        );
        assert_eq!(
            p.placed["m.layers.1.q.weight"].format,
            Format::MlxAffine { bits: 4, group: 64 }
        );
        assert_eq!(p.placed["m.layers.0.norm.weight"].format, Format::F16);
        let two = &p.refused["m.layers.2.q.weight"];
        assert!(
            two.contains("fits none of")
                && two.contains(
                    "its scales are F16 and its biases BF16, and one weight's pair is of one dtype"
                ),
            "{two}"
        );
        let four = &p.refused["m.layers.3.q.weight"];
        assert!(
            four.contains("its scales and biases are F32, and only BF16 or F16 are read"),
            "{four}"
        );
        assert_eq!(half.to_string(), "MLX-affine F16 4-bit group 64");
        assert_eq!(Format::F16.to_string(), "F16");
        assert_eq!(half.card_type(), Some(GgmlType::MLX_A4H));
        assert_eq!(
            Format::MlxAffineF16 { bits: 8, group: 64 }.card_type(),
            Some(GgmlType::MLX_A8H)
        );
        assert_eq!(
            Format::MlxAffineF16 { bits: 3, group: 64 }.card_type(),
            None
        );
        assert_eq!(Format::F16.card_type(), Some(GgmlType::F16));
        assert_eq!(half.logical(&[256, 32]), [256, 256]);
        assert_eq!(half.mlx_affine(), Some((4, 64, Dtype::F16)));
        assert_eq!(
            Format::MlxAffine { bits: 8, group: 64 }.mlx_affine(),
            Some((8, 64, Dtype::Bf16))
        );
        assert_eq!(Format::F16.mlx_affine(), None);
    }
}
