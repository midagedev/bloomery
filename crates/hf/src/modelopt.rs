//! A ModelOpt checkpoint's tensors in GGUF's types: one table ([`normalize`]) from the format a
//! tensor is placed in ([`quant_config::Format`]) to the GGUF type it is written as and the
//! repack that takes it there. A model's map supplies only names, roles and shapes; the type
//! follows from the format and from how the role reads the tensor ([`Use`]).
//!
//! ModelOpt stores an NVFP4 linear `<stem>` as a triple: `<stem>.weight`, U8 `[rows, k/2]`, two
//! E2M1 codes a byte (value `2i` in the low nibble, `2i + 1` in the high); `<stem>.weight_scale`,
//! F8_E4M3 `[rows, k/16]`, one scale a 16-value group; and `<stem>.weight_scale_2`, one F32, the
//! tensor's scale. A value is `e2m1 · weight_scale · weight_scale_2`. The plan names the two
//! scales a weight reads ([`Placed::companions`]); a module that stores no tensor scale reads the
//! one its parent module shares.
//!
//! GGUF's NVFP4 block (type 40, [`gguf::quant::dequant_row`]) holds 64 values in 36 bytes: the four
//! groups' scale bytes, then 32 code bytes, group `s`'s value `j` in the low nibble of byte `8s + j`
//! and value `j + 8` in its high nibble. The reader moves the nibbles into that order and copies
//! the scale bytes as they are; the tensor scale becomes the f32 the GGUF file keeps beside the
//! weight as `<stem>.scale`. A routed stack ([`Nvfp4Stack`]) has one tensor scale per expert, so its
//! `.scale` holds one value per expert, in expert order.
//!
//! A scale byte UE4M3 does not define as a scale (bit 7 set, or `0x7F`) is refused by name, as the
//! block's decode refuses it, rather than stripped of bit 7. `<stem>.input_scale`, the activation
//! scale ModelOpt calibrates for 4-bit activations, has no place in the block; the reader leaves it.
//!
//! compressed-tensors' `nvfp4-pack-quantized` ([`Format::Nvfp4Ct`]) stores the same codes and block
//! scales as `<stem>.weight_packed` and `<stem>.weight_scale` and a `<stem>.weight_global_scale` `g`
//! that is a divisor: the tensor scale is `1 / g` in f32, where ModelOpt's `weight_scale_2` is the
//! multiplier itself. Past that one value it is read as ModelOpt's NVFP4 is, so every reader of the
//! normalized tensor is the same. Its `<stem>.input_global_scale`, the scale of a 4-bit activation,
//! is read and refused unless one finite positive F32, and then unused: activations are quantized
//! to 8 bits here.
//!
//! MLX's affine quantization stores a weight `<stem>.weight` as U32 words holding `bits`-bit codes
//! along the last dim, with `<stem>.scales` and `<stem>.biases`, BF16 or F16 `[.., k/64]` (one
//! dtype for both of a weight), one pair a group of 64 values; a value is `scale · code + bias`. A
//! routed stack is one tensor whose first dim is the experts. GGUF's MLX_A4 and MLX_A8 blocks (BF16
//! pairs) and MLX_A4H and MLX_A8H blocks (F16 pairs) ([`gguf::quant::dequant_row`]) hold one
//! group: its code words, then its scale, then its bias. The reader (`mlx_affine`) moves those
//! bytes group by group and computes none of them.

use std::collections::HashMap;
use std::ops::Range;

use gguf::quant::{
    E4m3Rows, GgmlType, Nvfp4Rows, Q8Block, QuantError, check_fp8_b128, check_mlx_affine,
    check_mxfp8, dequant_row, e8m0_to_f32_half, fp4_pair, nvfp4_group_to_block, ue4m3_to_f32_half,
};

use crate::csf;
use crate::quant_config::{Format, Placed};
use safetensors::{Dtype, Raw};

/// Why a ModelOpt tensor was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Safetensors(#[from] safetensors::Error),
    #[error("modelopt nvfp4: {name}: {what}")]
    Shape { name: String, what: String },
    #[error(transparent)]
    Csf(#[from] csf::Error),
    #[error("{name}: {format} {what}")]
    Format {
        name: String,
        format: Format,
        what: String,
    },
    #[error("modelopt nvfp4: {name}: row {row} group {group}: {source}")]
    Scale {
        name: String,
        row: usize,
        group: usize,
        source: QuantError,
    },
}

/// One NVFP4 tensor, or a routed stack of them, in GGUF's layout.
#[derive(Debug, Default)]
struct Nvfp4 {
    /// The NVFP4 blocks, row-major, a stack expert by expert: `rows · k / 64` blocks of 36 bytes
    /// an expert.
    blocks: Vec<u8>,
    /// ggml's dims: `[k, rows]`, or `[k, rows, n_expert]` for a stack.
    dims: Vec<u64>,
    /// The `<stem>.scale` values: the tensor scale, or one per expert for a stack.
    scale: Vec<f32>,
}

/// What a role makes of a weight its checkpoint stores unquantized (BF16 or F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plain {
    /// ggml's reference Q8_0 ([`Q8Block::quantize_ref`]), as llama.cpp's converters write it.
    Q8_0,
    F32,
    /// BF16 as stored; an F32 tensor is refused.
    Bf16,
}

/// How a role reads its tensor, which with the tensor's format picks its GGUF type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// A matrix or vector a kernel reads: a quantized format keeps its blocks; an unquantized
    /// one is written as the role asks.
    Matrix(Plain),
    /// A matrix the card reads as Q8_0 whatever its checkpoint stores. BF16 and F32 are written
    /// as [`Plain::Q8_0`] writes them; an FP8 weight of one scale a block, a row or the tensor
    /// is a quantized source quantized again: each value is exact (its E4M3 byte times its
    /// block's f32 scale, one rounding), and ggml's reference Q8_0 rounds it once more
    /// ([`Q8Block::quantize_ref`]). No lever keeps the stored format, and a format with no such
    /// source (NVFP4, MXFP8, MLX) has no type here.
    Q8_0,
    /// A table the host reads a row at a time ([`GgmlType::F8E4M3_ROWS`]).
    Rows,
}

/// One tensor in its GGUF type.
#[derive(Debug, Default, PartialEq)]
pub struct Normal {
    pub type_id: u32,
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
    /// Row-major in the type's blocks.
    pub bytes: Vec<u8>,
    /// The values of the `<stem>.scale` the type reads beside it; empty for a type that reads
    /// none.
    pub scale: Vec<f32>,
}

/// The GGUF type id `format` is written as when `role` reads it: the table [`normalize`] runs.
///
/// | format | [`Use::Matrix`] | [`Use::Rows`] |
/// |---|---|---|
/// | NVFP4 (ModelOpt) | NVFP4 (40), `.scale` a tensor or an expert ([`Nvfp4Stack`]) | [`GgmlType::NVFP4_ROWS`], `.scale` the table's |
/// | NVFP4 with `csf` or `csf_32x2_v1` scales | as NVFP4, the scales decoded first ([`csf`]) | none |
/// | NVFP4 compressed-tensors | as NVFP4, `.scale` the reciprocal of `weight_global_scale` | as NVFP4 |
/// | F8_E4M3 with one scale a block (MXFP8's E8M0 a 32 values, an FP8 block format's), a row or the tensor | the card type the format is held in ([`Format::card_type`]): [`GgmlType::FP8_B128`] or [`GgmlType::MXFP8`], values as stored, each row block the scale of the checkpoint's block it lies in (`fp8`) | none |
/// | MLX affine, 4 or 8 bits, groups of 64 | the card type the format is held in: [`GgmlType::MLX_A4`] or [`GgmlType::MLX_A8`] for BF16 scales and biases, [`GgmlType::MLX_A4H`] or [`GgmlType::MLX_A8H`] for F16 ones, each group's words, scale and bias as stored, a stack one tensor `[k, rows, n_expert]` (`mlx_affine`) | none |
/// | BF16, F32 | Q8_0, F32 or BF16 as [`Plain`] says | none |
/// | F16 | F32 only, each value widened exactly; Q8_0 and BF16 have no type for it | none |
/// | BF16, F32, or F8_E4M3 held as FP8_B128 | [`Use::Q8_0`]: Q8_0, an FP8 weight decoded exactly and quantized again | none |
/// | FP8 with one tensor scale (its companion) | as above | [`GgmlType::F8E4M3_ROWS`], bytes as stored |
///
/// Every other pair has no GGUF type here yet.
#[must_use]
pub fn type_of(format: Format, role: Use) -> Option<u32> {
    match (format, role) {
        (
            Format::Nvfp4 | Format::Nvfp4Csf | Format::Nvfp4Csf32 | Format::Nvfp4Ct,
            Use::Matrix(_),
        ) => Some(GgmlType::NVFP4.as_u32()),
        (Format::Nvfp4 | Format::Nvfp4Ct, Use::Rows) => Some(GgmlType::NVFP4_ROWS.as_u32()),
        (
            Format::Mxfp8
            | Format::Fp8Block { .. }
            | Format::Fp8Channel
            | Format::Fp8Tensor
            | Format::MlxAffine { .. }
            | Format::MlxAffineF16 { .. },
            Use::Matrix(_),
        ) => format.card_type().map(GgmlType::as_u32),
        (Format::Bf16 | Format::F32, Use::Matrix(Plain::Q8_0) | Use::Q8_0) => {
            Some(GgmlType::Q8_0.as_u32())
        }
        (Format::Fp8Block { .. } | Format::Fp8Channel | Format::Fp8Tensor, Use::Q8_0)
            if format.card_type() == Some(GgmlType::FP8_B128) =>
        {
            Some(GgmlType::Q8_0.as_u32())
        }
        (Format::Bf16 | Format::F32 | Format::F16, Use::Matrix(Plain::F32)) => {
            Some(GgmlType::F32.as_u32())
        }
        (Format::Bf16, Use::Matrix(Plain::Bf16)) => Some(GgmlType::BF16.as_u32()),
        (Format::Fp8Tensor, Use::Rows) => Some(GgmlType::F8E4M3_ROWS.as_u32()),
        _ => None,
    }
}

fn finite(name: &str, raw: &Raw<'_>) -> Result<(), Error> {
    let at = match raw.dtype {
        Dtype::F32 => raw
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .position(|b| !f32::from_le_bytes(*b).is_finite()),
        Dtype::Bf16 => raw
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .position(|b| u16::from_le_bytes(*b) & 0x7f80 == 0x7f80),
        Dtype::F16 => raw
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .position(|b| u16::from_le_bytes(*b) & 0x7c00 == 0x7c00),
        _ => {
            return Err(safetensors::Error::Dtype {
                name: name.into(),
                dtype: raw.dtype.name().into(),
            }
            .into());
        }
    };
    if let Some(at) = at {
        return Err(Error::Shape {
            name: name.into(),
            what: format!("non-finite value at {at}"),
        });
    }
    Ok(())
}

/// The one format of `sources` (one weight, or a routed stack's experts in expert order), each
/// with its place in the plan.
///
/// # Errors
///
/// No source, and a source placed in another format than the first, by name.
pub fn format_of(sources: &[(&str, &Placed)]) -> Result<Format, Error> {
    let &[(first, placed), ..] = sources else {
        return Err(Error::Shape {
            name: "a tensor".into(),
            what: "has no source".into(),
        });
    };
    if let Some((name, p)) = sources.iter().find(|(_, p)| p.format != placed.format) {
        return Err(Error::Format {
            name: name.to_string(),
            format: p.format,
            what: format!("in a stack whose first source {first} is {}", placed.format),
        });
    }
    Ok(placed.format)
}

/// The checkpoint tensors `sources` (one weight, or a routed stack's experts in expert order),
/// each read through the companions its place in the plan names, in the GGUF type [`type_of`]
/// gives for `role`.
///
/// # Errors
///
/// A refusal of [`format_of`]; a pair [`type_of`] has no type for; a tensor `get` refuses or
/// lacks; a refusal of NVFP4 normalization, `fp8`, `fp8_q8_0` or `mlx_affine`; a stack of a type other than
/// NVFP4 and FP8_B128 or of unquantized experts read other than as F32 or BF16, a stack of MLX
/// tensors (its experts are one tensor), and one of experts of differing shapes; a shape or
/// dtype the format does not hold; a non-finite value or scaled value; and a row a
/// [`Plain::Q8_0`] block does not divide, each by name.
pub fn normalize<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
) -> Result<Normal, Error> {
    let format = format_of(sources)?;
    let (name, placed) = sources[0];
    let refuse = |what: String| Error::Format {
        name: name.to_string(),
        format,
        what,
    };
    let type_id = type_of(format, role)
        .ok_or_else(|| refuse(format!("read as {role:?} has no GGUF type here")))?;
    match format {
        Format::Nvfp4 | Format::Nvfp4Ct | Format::Nvfp4Csf | Format::Nvfp4Csf32 => {}
        Format::MlxAffine { .. } | Format::MlxAffineF16 { .. } => {
            let [source] = sources else {
                return Err(refuse(format!(
                    "in a stack of {}: an MLX stack is one tensor",
                    sources.len()
                )));
            };
            return mlx_affine(get, *source, type_id);
        }
        Format::Bf16 | Format::F32
            if sources.len() > 1 && matches!(role, Use::Matrix(Plain::Bf16 | Plain::F32)) =>
        {
            return plain_stack(get, sources, role, type_id);
        }
        _ if sources.len() != 1 && type_id != GgmlType::FP8_B128.as_u32() => {
            return Err(refuse(format!(
                "in a stack of {}: only NVFP4 and FP8_B128 experts stack here",
                sources.len()
            )));
        }
        Format::Fp8Block { .. } | Format::Fp8Channel | Format::Fp8Tensor if role == Use::Q8_0 => {
            return fp8_q8_0(get, sources, type_id);
        }
        Format::Mxfp8 | Format::Fp8Block { .. } | Format::Fp8Channel | Format::Fp8Tensor
            if role != Use::Rows =>
        {
            return fp8(get, sources, type_id);
        }
        _ => return plain(get, name, placed, role, type_id),
    };
    if role == Use::Rows {
        let [source] = sources else {
            return Err(refuse("row tables need exactly one source".into()));
        };
        return nvfp4_rows(get, *source, type_id);
    }
    nvfp4(get, sources)
}

/// [`normalize`] of the part of `sources` that holds the values `at` along dim `dim` of `dims`, the
/// tensor's ggml dims as its map gives them (a weight's `[k, rows]`, a routed stack's
/// `[k, rows, n_expert]`, a plain tensor's own): along a stack's experts the experts `at`; along
/// a matrix dim each weight and the block scale its place names cut to `at` first
/// ([`check_part`]), a compressed scale decoded before it is cut, and the tensor scale kept whole.
///
/// # Errors
///
/// [`check_part`]'s, and every refusal of [`normalize`], each by name.
pub fn normalize_part<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
    dims: &[u64],
    at: (usize, Range<u64>),
) -> Result<Normal, Error> {
    let cuts = match plan_part(&get, sources, role, dims, at)? {
        PartPlan::Experts(r) => return normalize(get, &sources[r], role),
        PartPlan::Cuts(cuts) => cuts,
    };
    let encoding = match sources[0].1.format {
        Format::Nvfp4Csf => Some(csf::Encoding::Csf),
        Format::Nvfp4Csf32 => Some(csf::Encoding::Csf32),
        _ => None,
    };
    let (decoded, places) = match encoding {
        Some(encoding) => decode_scales(&get, sources, encoding)?,
        None => (HashMap::new(), vec![]),
    };
    let read = |n: &str| match decoded.get(n) {
        Some(t) => Ok(t.raw()),
        None => get(n),
    };
    let mut cut = HashMap::new();
    for c in cuts {
        let raw = read(&c.name)?;
        let mut bytes = Vec::new();
        for r in &c.runs {
            let run = usize::try_from(r.start)
                .ok()
                .zip(usize::try_from(r.end).ok())
                .and_then(|(a, b)| raw.bytes.get(a..b))
                .ok_or_else(|| Error::Format {
                    name: c.name.clone(),
                    format: sources[0].1.format,
                    what: format!("{} bytes do not hold the part's run {r:?}", raw.bytes.len()),
                })?;
            bytes.extend_from_slice(run);
        }
        let shape = c.dims.iter().rev().map(|&d| d as usize).collect();
        cut.insert(
            c.name,
            Stored {
                dtype: raw.dtype,
                shape,
                bytes,
            },
        );
    }
    let read = |n: &str| match cut.get(n) {
        Some(t) => Ok(t.raw()),
        None => read(n),
    };
    if encoding.is_some() {
        let sources: Vec<_> = places.iter().map(|(w, p)| (*w, p)).collect();
        return normalize(read, &sources, role);
    }
    normalize(read, sources, role)
}

/// Check the part of `sources` [`normalize_part`] makes at `at` against what they store, reading
/// headers alone: the cut keeps whole stored values of each weight and whole blocks of its scale
/// grid (a compressed scale's decoded one), except a grid of one block along the dim, which every
/// part reads whole. An MLX affine weight is read whole: its part is the whole of a dim.
///
/// # Errors
///
/// A format with no type for `role`, a row table's part; a dim `dims` lacks, values `dims` does
/// not hold or a stack of another count; a weight whose bytes are not `dims`; values off its
/// stored values or across a scale block, a scale grid of another shape than its format's, and a
/// cut of an MLX affine weight, each by name.
pub fn check_part<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
    dims: &[u64],
    at: (usize, Range<u64>),
) -> Result<(), Error> {
    plan_part(&get, sources, role, dims, at).map(|_| ())
}

/// What a part of a tensor reads ([`check_part`]): a stack's experts, or each weight and block
/// scale cut.
enum PartPlan {
    Experts(Range<usize>),
    Cuts(Vec<Cut>),
}

/// One stored tensor a part reads cut: its name, the part's ggml dims, and the byte runs of the
/// stored bytes that hold it.
struct Cut {
    name: String,
    dims: Vec<u64>,
    runs: Vec<Range<u64>>,
}

/// The plan of the part of `sources` at `at` ([`check_part`]), from headers alone.
fn plan_part<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
    dims: &[u64],
    (dim, at): (usize, Range<u64>),
) -> Result<PartPlan, Error> {
    let format = format_of(sources)?;
    let refuse = |name: &str, what: String| Error::Format {
        name: name.to_string(),
        format,
        what,
    };
    let first = sources[0].0;
    if role == Use::Rows {
        return Err(refuse(first, "is a row table, which is read whole".into()));
    }
    type_of(format, role)
        .ok_or_else(|| refuse(first, format!("read as {role:?} has no GGUF type here")))?;
    let n = sources.len();
    let matrix = match dims {
        [k, rows, experts] if *experts == n as u64 && get(first)?.shape.len() == 2 => {
            vec![*k, *rows]
        }
        _ if n == 1 => dims.to_vec(),
        _ => {
            return Err(refuse(
                first,
                format!("{dims:?} are not the dims of a stack of {n}"),
            ));
        }
    };
    if dims.len() == 3 && matrix.len() == 2 && dim == 2 {
        let picked = usize::try_from(at.start)
            .ok()
            .zip(usize::try_from(at.end).ok())
            .filter(|&(a, b)| a < b && b <= n)
            .ok_or_else(|| refuse(first, format!("experts {at:?} of a stack of {n}")))?;
        return Ok(PartPlan::Experts(picked.0..picked.1));
    }
    let values_per_element = format.logical(&[1])[0] as u64;
    let mut cuts = Vec::new();
    for &(weight, placed) in sources {
        let w = get(weight)?;
        let size = w.dtype.size() as u64;
        if gguf::tensor_bytes(values_per_element, size, &matrix) != Some(w.bytes.len() as u64) {
            return Err(refuse(
                weight,
                format!(
                    "{} {:?} ({} bytes) does not hold {matrix:?}",
                    w.dtype.name(),
                    w.shape,
                    w.bytes.len()
                ),
            ));
        }
        if format.mlx_affine().is_some() {
            mlx_planes(get, weight, placed)?;
            if at.start != 0 || matrix.get(dim) != Some(&at.end) {
                return Err(refuse(
                    weight,
                    format!(
                        "values {at:?} along dim {dim} of {matrix:?}: an MLX affine weight is read whole, its scales and biases are not cut"
                    ),
                ));
            }
            continue;
        }
        let runs = gguf::tensor_runs(values_per_element, size, &matrix, dim, at.clone())
            .ok_or_else(|| {
                refuse(
                    weight,
                    format!(
                        "values {at:?} along dim {dim} of {matrix:?} are not whole stored values"
                    ),
                )
            })?;
        let mut part = matrix.clone();
        part[dim] = at.end - at.start;
        part[0] /= values_per_element;
        cuts.push(Cut {
            name: weight.to_string(),
            dims: part,
            runs,
        });
        if let Some(scale) = scale_grid(get, weight, format, placed, &matrix)? {
            let cut = scale.cut(dim, &at, matrix[dim]);
            cuts.push(cut.map_err(|what| refuse(&scale.name, what))?);
        }
    }
    Ok(PartPlan::Cuts(cuts))
}

/// A weight's block scales as a grid: its name, ggml dims `[columns, rows]` of scales, bytes a
/// scale, and the values and rows one covers.
struct Grid {
    name: String,
    dims: [u64; 2],
    size: u64,
    block: [u64; 2],
}

impl Grid {
    /// The grid's part for values `at` along dim `dim` of its weight of `extent` values there.
    fn cut(&self, dim: usize, at: &Range<u64>, extent: u64) -> Result<Cut, String> {
        let unit = self.block[dim];
        let span = if self.dims[dim] == 1 {
            0..1
        } else if at.start.is_multiple_of(unit) && (at.end.is_multiple_of(unit) || at.end == extent)
        {
            at.start / unit..at.end.div_ceil(unit)
        } else {
            return Err(format!(
                "values {at:?} along dim {dim} split its {unit}-value scale blocks"
            ));
        };
        let runs = gguf::tensor_runs(1, self.size, &self.dims, dim, span.clone())
            .ok_or_else(|| format!("scales {span:?} along dim {dim} pass its grid"))?;
        let mut dims = self.dims.to_vec();
        dims[dim] = span.end - span.start;
        Ok(Cut {
            name: self.name.clone(),
            dims,
            runs,
        })
    }
}

/// The block scale grid of the weight placed as `placed` of ggml dims `matrix` (`[k, rows]`):
/// NVFP4's one F8_E4M3 a 16 values (a compressed scale's decoded grid), an F8_E4M3 format's by
/// [`Format::fp8_scale_block`]; `None` for an unquantized format.
fn scale_grid<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    weight: &str,
    format: Format,
    placed: &Placed,
    matrix: &[u64],
) -> Result<Option<Grid>, Error> {
    let two_d = || {
        let &[k, rows] = matrix else {
            return Err(Error::Format {
                name: weight.to_string(),
                format,
                what: format!("{matrix:?} is not a 2-D quantized weight"),
            });
        };
        Ok(placed.companions.first().map(|first| (first, [k, rows])))
    };
    let refuse = |name: &str, what: String| Error::Format {
        name: name.to_string(),
        format,
        what,
    };
    let grid = |name: String, [k, rows]: [u64; 2], size: usize, [a, b]: [usize; 2]| Grid {
        name,
        dims: [k.div_ceil(b as u64), rows.div_ceil(a as u64)],
        size: size as u64,
        block: [b as u64, a as u64],
    };
    let g = match format {
        Format::Nvfp4Csf | Format::Nvfp4Csf32 => {
            let Some((first, dims)) = two_d()? else {
                return Ok(None);
            };
            let (stem, _) = first
                .rsplit_once('.')
                .ok_or_else(|| refuse(first, "is not a part of a compressed block scale".into()))?;
            return Ok(Some(grid(stem.to_string(), dims, 1, [1, 16])));
        }
        Format::Nvfp4 | Format::Nvfp4Ct => {
            let Some((first, dims)) = two_d()? else {
                return Ok(None);
            };
            grid(first.clone(), dims, 1, [1, 16])
        }
        Format::Mxfp8 | Format::Fp8Block { .. } | Format::Fp8Channel | Format::Fp8Tensor => {
            let Some((first, [k, rows])) = two_d()? else {
                return Ok(None);
            };
            let s = get(first)?;
            let block = format
                .fp8_scale_block(rows as usize, k as usize, s.shape)
                .ok_or_else(|| {
                    refuse(
                        first,
                        format!("{:?} is not its weight's scale grid", s.shape),
                    )
                })?;
            grid(first.clone(), [k, rows], s.dtype.size(), block)
        }
        _ => return Ok(None),
    };
    let s = get(&g.name)?;
    if gguf::tensor_bytes(1, g.size, &g.dims) != Some(s.bytes.len() as u64) {
        return Err(refuse(
            &g.name,
            format!("{} bytes do not hold the grid {:?}", s.bytes.len(), g.dims),
        ));
    }
    Ok(Some(g))
}

/// A tensor this reader made: decoded from a compressed form or cut from a stored one.
struct Stored {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl Stored {
    fn raw(&self) -> Raw<'_> {
        Raw {
            dtype: self.dtype,
            shape: &self.shape,
            bytes: &self.bytes,
        }
    }
}

/// Decoded block scales by name, and each weight placed as plain NVFP4 reading them.
type Decoded<'s> = (HashMap<String, Stored>, Vec<(&'s str, Placed)>);

/// Each compressed block scale of the NVFP4 `sources` decoded by `encoding` to the bytes a plain
/// checkpoint stores, by the scale's name, and each weight placed as plain NVFP4 reading those
/// bytes and the tensor scale its own place names.
fn decode_scales<'s, 'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&'s str, &Placed)],
    encoding: csf::Encoding,
) -> Result<Decoded<'s>, Error> {
    let mut decoded = HashMap::new();
    let mut places = Vec::with_capacity(sources.len());
    for &(weight, placed) in sources {
        let refuse = |what: String| Error::Format {
            name: weight.to_string(),
            format: placed.format,
            what,
        };
        let (Some((scale, _)), [_, .., tensor_scale]) = (
            placed.companions.first().and_then(|c| c.rsplit_once('.')),
            placed.companions.as_slice(),
        ) else {
            return Err(refuse(format!(
                "its place names {:?}, not a compressed block scale and a tensor scale",
                placed.companions
            )));
        };
        let w = get(weight)?;
        let &[rows, half] = w.shape else {
            return Err(refuse(format!("{:?} is not [rows, k/2]", w.shape)));
        };
        if !half.is_multiple_of(8) {
            return Err(refuse(format!("k/2 = {half} is not whole 16-value groups")));
        }
        let bytes = csf::weight_scale(get, scale, encoding, rows, half / 8)?;
        decoded.insert(
            scale.to_string(),
            Stored {
                dtype: Dtype::F8E4M3,
                shape: vec![rows, half / 8],
                bytes,
            },
        );
        places.push((
            weight,
            Placed {
                format: Format::Nvfp4,
                companions: vec![scale.to_string(), tensor_scale.clone()],
            },
        ));
    }
    Ok((decoded, places))
}

/// The NVFP4 `sources` (one linear, or a routed stack's experts) in GGUF's NVFP4 block.
fn nvfp4<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
) -> Result<Normal, Error> {
    let mut stack = Nvfp4Stack::new(sources.len());
    for &source in sources {
        stack.push(&get, source)?;
    }
    stack.finish()
}

/// A stack normalized into one allocation, with only the current expert's decoded scales held.
pub struct Nvfp4Stack {
    out: Nvfp4,
    experts: usize,
}

impl Nvfp4Stack {
    /// Reserve the shape of a stack of `experts` sources; its bytes are allocated on first push.
    pub fn new(experts: usize) -> Self {
        Self {
            out: Nvfp4::default(),
            experts,
        }
    }

    /// Append one expert, decoding its compressed scales only for this call.
    pub fn push<'a>(
        &mut self,
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
        source: (&str, &Placed),
    ) -> Result<(), Error> {
        let left = self
            .experts
            .checked_sub(self.out.scale.len())
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Shape {
                name: source.0.into(),
                what: "more experts than the stack holds".into(),
            })?;
        let [rows, k] = append(&get, source, &mut self.out, left)?;
        let dims = [k as u64, rows as u64];
        if self.out.dims.is_empty() {
            self.out.dims = dims.to_vec();
        } else if self.out.dims != dims {
            return Err(Error::Shape {
                name: source.0.into(),
                what: format!(
                    "dims {dims:?}; the stack's first expert has {:?}",
                    self.out.dims
                ),
            });
        }
        Ok(())
    }

    /// Finish the stack, refusing a missing expert; a single source remains a matrix.
    pub fn finish(mut self) -> Result<Normal, Error> {
        if self.experts == 0 {
            return Err(Error::Shape {
                name: "routed stack".into(),
                what: "no expert".into(),
            });
        }
        if self.out.scale.len() != self.experts {
            return Err(Error::Shape {
                name: "routed stack".into(),
                what: format!(
                    "{} experts, expected {}",
                    self.out.scale.len(),
                    self.experts
                ),
            });
        }
        if self.experts > 1 {
            self.out.dims.push(self.experts as u64);
        }
        Ok(Normal {
            type_id: GgmlType::NVFP4.as_u32(),
            dims: self.out.dims,
            bytes: self.out.blocks,
            scale: self.out.scale,
        })
    }
}

/// The names of the scales a plain NVFP4 weight's place lists, in the order its format reads them.
struct Scales<'p> {
    block: &'p str,
    tensor: &'p str,
    /// compressed-tensors' `input_global_scale`, when the place lists one.
    input: Option<&'p str>,
}

/// The scales `placed` lists for `weight`: a block scale and a tensor scale, and for
/// [`Format::Nvfp4Ct`] an input scale after them.
fn scales_of<'p>(weight: &str, placed: &'p Placed) -> Result<Scales<'p>, Error> {
    match placed.companions.as_slice() {
        [block, tensor] => Ok(Scales {
            block,
            tensor,
            input: None,
        }),
        [block, tensor, input] if placed.format == Format::Nvfp4Ct => Ok(Scales {
            block,
            tensor,
            input: Some(input),
        }),
        names => Err(Error::Shape {
            name: weight.to_string(),
            what: format!("its place names {names:?}, not a block scale and a tensor scale"),
        }),
    }
}

/// The multiplier GGUF keeps beside an NVFP4 weight, from the tensor scale `stored` under `name`:
/// ModelOpt's `weight_scale_2` as it is, compressed-tensors' `weight_global_scale`, a divisor, as
/// its reciprocal, refused unless positive with a finite reciprocal.
fn multiplier(format: Format, weight: &str, name: &str, stored: f32) -> Result<f32, Error> {
    if format != Format::Nvfp4Ct {
        return Ok(stored);
    }
    let scale = 1.0 / stored;
    if stored > 0.0 && scale.is_finite() {
        return Ok(scale);
    }
    Err(Error::Format {
        name: weight.to_string(),
        format,
        what: format!(
            "weight_global_scale {name} is {stored}, not a positive divisor with a finite reciprocal"
        ),
    })
}

/// The `input_global_scale` `name` of the compressed-tensors NVFP4 `weight`: one F32, finite and
/// positive, or refused by name. The value is not used.
fn input_scale<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    weight: &str,
    format: Format,
    name: &str,
) -> Result<f32, Error> {
    let t = get(name)?;
    let refuse = |what: String| Error::Format {
        name: weight.to_string(),
        format,
        what,
    };
    let (Dtype::F32, &[a, b, c, d]) = (t.dtype, t.bytes) else {
        return Err(refuse(format!(
            "input scale {name} is {} {:?}, not one F32",
            t.dtype.name(),
            t.shape
        )));
    };
    let scale = f32::from_le_bytes([a, b, c, d]);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(refuse(format!(
            "input scale {name} is {scale}, not a finite positive F32"
        )));
    }
    Ok(scale)
}

/// The checked input scale of the compressed-tensors NVFP4 `weight` placed as `placed`, or `None`
/// when its place lists none (ModelOpt's NVFP4 keeps its `input_scale` out of the place).
///
/// # Errors
///
/// A place that does not list the weight's scales, a tensor `get` refuses or lacks, and an input
/// scale that is not one finite positive F32, each by name.
pub fn input_global_scale<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    weight: &str,
    placed: &Placed,
) -> Result<Option<f32>, Error> {
    scales_of(weight, placed)?
        .input
        .map(|name| input_scale(get, weight, placed.format, name))
        .transpose()
}

/// The tensor scale (the multiplier of [`multiplier`]) and block-scale name of plain NVFP4 rows.
/// Checks the code and scale planes' shapes and byte counts without copying either, and a
/// compressed-tensors weight's input scale.
pub fn nvfp4_rows_scale<'a, 'p>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    weight: &str,
    placed: &'p Placed,
) -> Result<(f32, &'p str), Error> {
    if !matches!(placed.format, Format::Nvfp4 | Format::Nvfp4Ct) {
        return Err(Error::Format {
            name: weight.to_string(),
            format: placed.format,
            what: "not plain NVFP4 code and scale rows".into(),
        });
    }
    let shape = |name: String, what: String| Error::Shape { name, what };
    let wname = weight.to_string();
    let Scales {
        block: sname,
        tensor: tname,
        input,
    } = scales_of(weight, placed)?;
    let (w, s) = (get(&wname)?, get(sname)?);
    let &[rows, half] = w.shape else {
        return Err(shape(
            wname,
            format!("shape {:?}, not [rows, k/2]", w.shape),
        ));
    };
    if w.dtype != Dtype::U8 || rows == 0 || half == 0 || !half.is_multiple_of(8) {
        return Err(shape(
            wname,
            format!(
                "{} {:?}, not U8 rows of whole 16-value groups",
                w.dtype.name(),
                w.shape
            ),
        ));
    }
    if (half as u64).checked_mul(2).is_none() {
        return Err(shape(wname.clone(), "row size overflow".into()));
    }
    let size = rows
        .checked_mul(half)
        .ok_or_else(|| shape(wname.clone(), "weight size overflow".into()))?;
    if (size / 8).checked_mul(9).is_none() {
        return Err(shape(wname.clone(), "row bytes size overflow".into()));
    }
    if w.bytes.len() != size {
        return Err(shape(wname, format!("{} bytes, not {size}", w.bytes.len())));
    }
    if s.dtype != Dtype::F8E4M3 || s.shape != [rows, half / 8] || s.bytes.len() != size / 8 {
        return Err(shape(
            sname.to_string(),
            format!(
                "{} {:?}; the groups are F8_E4M3 [{rows}, {}]",
                s.dtype.name(),
                s.shape,
                half / 8
            ),
        ));
    }
    let t = get(tname)?;
    let scale = match (t.dtype, t.bytes) {
        (Dtype::F32, &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]),
        _ => {
            return Err(shape(
                tname.to_string(),
                format!("{} {:?}, not one F32", t.dtype.name(), t.shape),
            ));
        }
    };
    if !scale.is_finite() {
        return Err(shape(
            tname.to_string(),
            format!("tensor scale {scale} is not finite"),
        ));
    }
    let scale = multiplier(placed.format, weight, tname, scale)?;
    if let Some(input) = input {
        input_scale(get, weight, placed.format, input)?;
    }
    Ok((scale, sname))
}

/// The NVFP4 weight `source` as [`GgmlType::NVFP4_ROWS`]: each 16-value group
/// its scale byte then its 8 code bytes, at the tensor's scale.
fn nvfp4_rows<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    type_id: u32,
) -> Result<Normal, Error> {
    let (scale, sname) = nvfp4_rows_scale(&get, weight, placed)?;
    let (w, s) = (get(weight)?, get(sname)?);
    let (rows, half) = (w.shape[0], w.shape[1]);
    let k = half as u64 * 2;
    let capacity = w.bytes.len() / 8 * 9;
    let decoder = Nvfp4Rows::new(scale);
    let mut values = [0.0; 16];
    let mut bytes = Vec::with_capacity(capacity);
    for (row, (codes, scales)) in w
        .bytes
        .chunks_exact(half)
        .zip(s.bytes.chunks_exact(half / 8))
        .enumerate()
    {
        for (group, (codes, &byte)) in codes.as_chunks::<8>().0.iter().zip(scales).enumerate() {
            bytes.push(byte);
            bytes.extend_from_slice(codes);
            decoder
                .decode_planes(codes, &[byte], &mut values)
                .map_err(|source| Error::Scale {
                    name: sname.to_string(),
                    row,
                    group,
                    source,
                })?;
        }
    }
    Ok(Normal {
        type_id,
        dims: vec![k, rows as u64],
        bytes,
        scale: vec![scale],
    })
}

/// The F8_E4M3 weight `sources` (one weight, or a routed stack's experts in expert order) in
/// `type_id`, the card type its format is held in ([`Format::card_type`]: MXFP8, else FP8_B128),
/// its values unchanged: each row's blocks of the type in order, each the scale of the
/// checkpoint's block it lies in ([`Format::fp8_scale_block`]: a block, a row or the tensor),
/// then its value bytes as stored; a stack's experts one after another, as `[k, rows, n_expert]`.
/// FP8_B128 writes the scale as an f32, widened from F32, BF16 or E8M0 (U8 or F8_E8M0); MXFP8 as
/// its E8M0 byte. A block the type's check refuses ([`check_fp8_b128`], [`check_mxfp8`]: a scale
/// that is not finite, an E4M3 NaN, a product past f32) is refused naming its row; an expert
/// whose shape differs from the first's, by name.
fn fp8<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    type_id: u32,
) -> Result<Normal, Error> {
    let ty = GgmlType::from_u32(type_id);
    let mut bytes = Vec::new();
    let mut shape = None;
    for &source in sources {
        let have = fp8_append(&get, source, ty, &mut bytes)?;
        let first = *shape.get_or_insert(have);
        if first != have {
            return Err(Error::Format {
                name: source.0.to_string(),
                format: source.1.format,
                what: format!(
                    "rows and k {have:?}; the stack's first expert {} has {first:?}",
                    sources[0].0
                ),
            });
        }
    }
    let Some([rows, k]) = shape else {
        return Err(Error::Shape {
            name: "an FP8 tensor".into(),
            what: "has no source".into(),
        });
    };
    let mut dims = vec![k as u64, rows as u64];
    if sources.len() > 1 {
        dims.push(sources.len() as u64);
    }
    Ok(Normal {
        type_id,
        dims,
        bytes,
        scale: vec![],
    })
}

/// Appends the `ty` blocks of the weight `source` ([`fp8`]) to `out`; returns its `[rows, k]`.
fn fp8_append<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    ty: GgmlType,
    out: &mut Vec<u8>,
) -> Result<[usize; 2], Error> {
    let refuse = |what: String| Error::Format {
        name: weight.to_string(),
        format: placed.format,
        what,
    };
    // The type's values a block, its scale's bytes and its check. Its block divides the width
    // of the checkpoint's: `card_type` picks it so, and a row or tensor scale spans `k`.
    type Check = fn(&[u8]) -> Result<(), QuantError>;
    let (blck, unit, check): (usize, usize, Check) = match ty {
        GgmlType::MXFP8 => (32, 1, check_mxfp8),
        _ => (128, 4, check_fp8_b128),
    };
    // A weight-only place names the scale alone; an input scale after it is left, since the
    // activations stay f32.
    let Some(sname) = placed.companions.first() else {
        return Err(refuse("its place names no scale".into()));
    };
    let (w, s) = (get(weight)?, get(sname)?);
    let &[rows, k] = w.shape else {
        return Err(refuse(format!("shape {:?}, not [rows, k]", w.shape)));
    };
    if w.dtype != Dtype::F8E4M3
        || rows == 0
        || k == 0
        || !k.is_multiple_of(blck)
        || Some(w.bytes.len()) != rows.checked_mul(k)
    {
        return Err(refuse(format!(
            "{} {:?} ({} bytes), not F8_E4M3 rows of whole {blck}-value blocks",
            w.dtype.name(),
            w.shape,
            w.bytes.len()
        )));
    }
    let scales: Vec<u8> = match (unit, s.dtype) {
        (4, Dtype::F32 | Dtype::Bf16) => safetensors::f32_values(sname, &s)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        (4, Dtype::U8 | Dtype::F8E8M0) => s
            .bytes
            .iter()
            .flat_map(|&e| (2.0 * e8m0_to_f32_half(e)).to_le_bytes())
            .collect(),
        (1, Dtype::U8 | Dtype::F8E8M0) => s.bytes.to_vec(),
        _ => vec![],
    };
    // The block one scale covers, when the grid's shape, dtype and bytes are the format's.
    let block = placed
        .format
        .fp8_scale_block(rows, k, s.shape)
        .filter(|&[a, b]| {
            let n = rows.div_ceil(a) * k.div_ceil(b);
            unit.checked_mul(n) == Some(scales.len())
                && s.dtype.size().checked_mul(n) == Some(s.bytes.len())
        });
    let Some([a, b]) = block else {
        let what = if unit == 4 {
            "F32, BF16 or E8M0"
        } else {
            "E8M0"
        };
        return Err(refuse(format!(
            "{sname} {} {:?} ({} bytes): not the {} grid of one {what} scale a block, as {ty:?} holds it",
            s.dtype.name(),
            s.shape,
            s.bytes.len(),
            placed.format
        )));
    };
    let c = k.div_ceil(b);
    let more = (rows * (k / blck))
        .checked_mul(blck + unit)
        .ok_or_else(|| refuse("block bytes size overflow".into()))?;
    out.try_reserve(more)
        .map_err(|e| refuse(format!("{more} block bytes: {e}")))?;
    os::advise_huge_within(out.spare_capacity_mut());
    for (row, values) in w.bytes.chunks_exact(k).enumerate() {
        let at = out.len();
        for (j, block) in values.chunks_exact(blck).enumerate() {
            let i = (row / a * c + j * blck / b) * unit;
            out.extend_from_slice(&scales[i..i + unit]);
            out.extend_from_slice(block);
        }
        check(&out[at..]).map_err(|e| refuse(format!("row {row}: {e}")))?;
    }
    Ok([rows, k])
}

/// The F8_E4M3 weight `sources` (exactly one: a stack of Q8_0 has no type here) in Q8_0, which
/// [`Use::Q8_0`] asks of an FP8 weight of one scale a block, a row or the tensor. [`fp8_append`]
/// reads it as FP8_B128's blocks, with every refusal it makes, so each value is its E4M3 byte
/// times the f32 scale of the checkpoint's block it lies in ([`dequant_row`]'s one rounding); a
/// row's values then go through ggml's reference Q8_0 rule ([`Q8Block::quantize_ref`]), 32 values
/// a block, one rounding more. A block that rule refuses (a scale past f16's range) is refused
/// naming its row and block.
fn fp8_q8_0<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    type_id: u32,
) -> Result<Normal, Error> {
    let &[(weight, placed)] = sources else {
        return Err(Error::Shape {
            name: "an FP8 tensor".into(),
            what: format!("read as Q8_0 takes one source, got {}", sources.len()),
        });
    };
    let refuse = |what: String| Error::Format {
        name: weight.to_string(),
        format: placed.format,
        what,
    };
    let mut blocks = Vec::new();
    let [rows, k] = fp8_append(&get, (weight, placed), GgmlType::FP8_B128, &mut blocks)?;
    // `fp8_append` holds whole 128-value blocks of 132 bytes, so a row is `k / 128` of them and
    // `k` a multiple of 32.
    let row_bytes = k / 128 * 132;
    let capacity = (rows * k / 32)
        .checked_mul(34)
        .ok_or_else(|| refuse("Q8_0 byte count overflows usize".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve(capacity)
        .map_err(|e| refuse(format!("{capacity} Q8_0 bytes: {e}")))?;
    let mut values = vec![0.0f32; k];
    for (row, src) in blocks.chunks_exact(row_bytes).enumerate() {
        dequant_row(GgmlType::FP8_B128, src, &mut values)
            .map_err(|e| refuse(format!("row {row}: {e}")))?;
        for (i, b) in values.as_chunks::<32>().0.iter().enumerate() {
            let q = Q8Block::quantize_ref(b).ok_or_else(|| {
                refuse(format!(
                    "row {row} Q8_0 block {i} has non-finite values or an unrepresentable scale"
                ))
            })?;
            bytes.extend(q.to_bytes());
        }
    }
    Ok(Normal {
        type_id,
        dims: vec![k as u64, rows as u64],
        bytes,
        scale: vec![],
    })
}

/// The tensor scale of the FP8 row table `name` placed as `placed` ([`Format::Fp8Tensor`] read as
/// [`Use::Rows`], whose bytes [`normalize`] keeps as stored), and its companion's name, from the
/// headers and that one value: a table that is not 2-D F8_E4M3, or a companion that is not one
/// finite F32 or BF16, is refused by name. A reader that maps the table's rows where they are
/// stored checks its parts here without copying them.
pub fn fp8_rows_scale<'a, 'p>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    name: &str,
    placed: &'p Placed,
) -> Result<(f32, &'p str), Error> {
    let refuse = |what: String| Error::Format {
        name: name.to_string(),
        format: placed.format,
        what,
    };
    let raw = get(name)?;
    if placed.format != Format::Fp8Tensor || raw.dtype != Dtype::F8E4M3 || raw.shape.len() != 2 {
        return Err(refuse(format!(
            "is {} {:?}, not F8_E4M3 rows",
            raw.dtype.name(),
            raw.shape
        )));
    }
    let [scale_name] = placed.companions.as_slice() else {
        return Err(refuse(format!(
            "reads {:?}, not one tensor scale",
            placed.companions
        )));
    };
    let s = get(scale_name)?;
    finite(scale_name, &s)?;
    let scale = match (s.dtype, s.bytes) {
        (Dtype::F32, &[a, b, c, d]) => f32::from_le_bytes([a, b, c, d]),
        (Dtype::Bf16, &[a, b]) => f32::from_bits(u32::from(u16::from_le_bytes([a, b])) << 16),
        _ => {
            return Err(refuse(format!(
                "scale {scale_name} is {} {:?}, not one F32 or BF16",
                s.dtype.name(),
                s.shape
            )));
        }
    };
    Ok((scale, scale_name))
}

/// A tensor of a format that is neither NVFP4 nor stacked: FP8 rows, or BF16 and F32.
fn plain<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    name: &str,
    placed: &Placed,
    role: Use,
    type_id: u32,
) -> Result<Normal, Error> {
    let refuse = |what: String| Error::Format {
        name: name.to_string(),
        format: placed.format,
        what,
    };
    let raw = get(name)?;
    let dims: Vec<u64> = raw.shape.iter().rev().map(|&d| d as u64).collect();
    if placed.format == Format::Fp8Tensor {
        let (scale, _) = fp8_rows_scale(&get, name, placed)?;
        let decoder = E4m3Rows::new(scale);
        decoder
            .check(raw.bytes)
            .map_err(|e| refuse(e.to_string()))?;
        return Ok(Normal {
            type_id,
            dims,
            bytes: raw.bytes.to_vec(),
            scale: vec![scale],
        });
    }
    finite(name, &raw)?;
    let bytes = match role {
        Use::Matrix(Plain::Bf16) => raw.bytes.to_vec(),
        Use::Matrix(Plain::F32) => safetensors::f32_values(name, &raw)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => {
            if !raw.shape.last().is_some_and(|k| k.is_multiple_of(32)) {
                return Err(refuse(format!(
                    "shape {:?}: a row is not whole 32-value Q8_0 blocks",
                    raw.shape
                )));
            }
            let values = safetensors::f32_values(name, &raw)?;
            let capacity = (values.len() / 32)
                .checked_mul(34)
                .ok_or_else(|| refuse("Q8_0 byte count overflows usize".into()))?;
            let mut bytes = Vec::with_capacity(capacity);
            for (i, b) in values.as_chunks::<32>().0.iter().enumerate() {
                let q = Q8Block::quantize_ref(b).ok_or_else(|| {
                    refuse(format!(
                        "Q8_0 block {i} has non-finite values or an unrepresentable scale"
                    ))
                })?;
                bytes.extend(q.to_bytes());
            }
            bytes
        }
    };
    Ok(Normal {
        type_id,
        dims,
        bytes,
        scale: vec![],
    })
}

/// Unquantized experts `sources`, each written as [`plain`] writes it, one after another as
/// `[k, rows, n_expert]`; an expert that is not a matrix of the first's shape is refused by name.
fn plain_stack<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    sources: &[(&str, &Placed)],
    role: Use,
    type_id: u32,
) -> Result<Normal, Error> {
    let mut out = Normal {
        type_id,
        ..Normal::default()
    };
    for &(name, placed) in sources {
        let n = plain(&get, name, placed, role, type_id)?;
        if n.dims.len() != 2 || (!out.dims.is_empty() && out.dims != n.dims) {
            return Err(Error::Format {
                name: name.to_string(),
                format: placed.format,
                what: format!(
                    "dims {:?}; the stack's first expert {} has {:?}",
                    n.dims, sources[0].0, out.dims
                ),
            });
        }
        out.dims = n.dims;
        out.bytes.extend(n.bytes);
    }
    out.dims.push(sources.len() as u64);
    Ok(out)
}

/// Reads the weight of `source` and the two scales its place names, and appends its blocks and
/// tensor scale to `out`, which holds room for the blocks of `left` experts of its shape (it and
/// the ones after it) once it returns: a stack's blocks are allocated once, at their length.
/// Returns `[rows, k]`.
fn append<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    out: &mut Nvfp4,
    left: usize,
) -> Result<[usize; 2], Error> {
    let encoding = match placed.format {
        Format::Nvfp4Csf => Some(csf::Encoding::Csf),
        Format::Nvfp4Csf32 => Some(csf::Encoding::Csf32),
        _ => None,
    };
    if let Some(encoding) = encoding {
        let (decoded, places) = decode_scales(get, &[(weight, placed)], encoding)?;
        let read = |n: &str| match decoded.get(n) {
            Some(t) => Ok(t.raw()),
            None => get(n),
        };
        return append_plain(&read, (places[0].0, &places[0].1), out, left);
    }
    append_plain(get, (weight, placed), out, left)
}

/// Append one expert whose block scales are already decoded.
fn append_plain<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    out: &mut Nvfp4,
    left: usize,
) -> Result<[usize; 2], Error> {
    let scales = scales_of(weight, placed)?;
    let typed = |name: &str, want: Dtype| -> Result<(String, Raw<'a>), Error> {
        let unfilled = |name: &str, what| Error::Shape {
            name: name.to_string(),
            what,
        };
        Ok((name.to_string(), stored(get, name, want, unfilled)?))
    };
    let shape = |name: &str, what: String| Error::Shape {
        name: name.to_string(),
        what,
    };
    let (wname, w) = typed(weight, Dtype::U8)?;
    let &[rows, half] = w.shape else {
        return Err(shape(
            &wname,
            format!("shape {:?}, not [rows, k/2]", w.shape),
        ));
    };
    let k = half
        .checked_mul(2)
        .ok_or_else(|| shape(&wname, format!("k/2 = {half} overflows k")))?;
    if k == 0 || !k.is_multiple_of(64) {
        return Err(shape(
            &wname,
            format!("k = {k}, not a positive multiple of the 64 values of a block"),
        ));
    }
    let (sname, s) = typed(scales.block, Dtype::F8E4M3)?;
    if s.shape != [rows, k / 16] {
        return Err(shape(
            &sname,
            format!(
                "shape {:?}; its weight's 16-value groups are [{rows}, {}]",
                s.shape,
                k / 16
            ),
        ));
    }
    let (tname, t) = typed(scales.tensor, Dtype::F32)?;
    let scale = <[u8; 4]>::try_from(t.bytes)
        .map(f32::from_le_bytes)
        .map_err(|_| shape(&tname, format!("shape {:?}, not one value", t.shape)))?;
    finite(&tname, &t)?;
    let scale = multiplier(placed.format, weight, &tname, scale)?;
    if let Some(input) = scales.input {
        input_scale(get, weight, placed.format, input)?;
    }

    let bytes = rows
        .checked_mul(k / 64)
        .and_then(|n| n.checked_mul(36))
        .and_then(|n| n.checked_mul(left))
        .ok_or_else(|| shape(&wname, "NVFP4 byte count overflows usize".into()))?;
    out.blocks
        .try_reserve_exact(bytes)
        .map_err(|e| shape(&wname, format!("cannot reserve {bytes} NVFP4 bytes: {e}")))?;
    os::advise_huge_within(out.blocks.spare_capacity_mut());
    // Each scale byte's half value, worked out once; a byte that is no scale is refused below.
    let halves: [Option<f32>; 256] = std::array::from_fn(|b| ue4m3_to_f32_half(b as u8).ok());
    for (row, (w, s)) in w
        .bytes
        .chunks_exact(half)
        .zip(s.bytes.chunks_exact(k / 16))
        .enumerate()
    {
        for (block, (w, d)) in w
            .as_chunks::<32>()
            .0
            .iter()
            .zip(s.as_chunks::<4>().0)
            .enumerate()
        {
            for (sub, &byte) in d.iter().enumerate() {
                let half_scale = halves[usize::from(byte)]
                    .map_or_else(|| ue4m3_to_f32_half(byte), Ok)
                    .map_err(|source| Error::Scale {
                        name: sname.clone(),
                        row,
                        group: 4 * block + sub,
                        source,
                    })?;
                if !(half_scale * 12.0 * scale).is_finite()
                    && w[sub * 8..][..8]
                        .iter()
                        .flat_map(|&b| fp4_pair(b))
                        .any(|v| !(half_scale * f32::from(v) * scale).is_finite())
                {
                    return Err(shape(
                        &wname,
                        format!(
                            "row {row} group {} times scale {tname} is not finite",
                            4 * block + sub
                        ),
                    ));
                }
            }
            let mut blocks = [0u8; 36];
            blocks[..4].copy_from_slice(d);
            for (g, o) in w
                .as_chunks::<8>()
                .0
                .iter()
                .zip(blocks[4..].as_chunks_mut::<8>().0)
            {
                *o = nvfp4_group_to_block(g);
            }
            out.blocks.extend_from_slice(&blocks);
        }
    }
    out.scale.push(scale);
    Ok([rows, k])
}

/// The stored tensor `name` of dtype `want` whose bytes fill its shape; bytes that do not are
/// `unfilled`'s refusal of the tensor, with what is wrong.
fn stored<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    name: &str,
    want: Dtype,
    unfilled: impl FnOnce(&str, String) -> Error,
) -> Result<Raw<'a>, Error> {
    let raw = get(name)?;
    if raw.dtype != want {
        return Err(safetensors::Error::Kind {
            name: name.to_string(),
            have: raw.dtype.name(),
            want: want.name(),
        }
        .into());
    }
    let bytes = raw
        .shape
        .iter()
        .try_fold(want.size(), |n, &d| n.checked_mul(d));
    if bytes != Some(raw.bytes.len()) {
        return Err(unfilled(
            name,
            format!(
                "shape {:?} does not match {} stored bytes",
                raw.shape,
                raw.bytes.len()
            ),
        ));
    }
    Ok(raw)
}

/// An MLX affine weight's stored planes, checked against one another.
struct MlxPlanes<'a> {
    ty: GgmlType,
    /// The U32 words of codes, `[.., k * bits / 32]`.
    codes: Raw<'a>,
    /// The scale of each group (BF16, or F16 as the format says), `[.., k / group]`.
    scales: Raw<'a>,
    /// The bias of each group, of the scale's dtype, `[.., k / group]`.
    biases: Raw<'a>,
    /// The values a row holds.
    k: usize,
    /// The rows of the tensor, its leading dims folded.
    rows: usize,
    groups: usize,
    /// The bytes of code words a group holds.
    group_bytes: usize,
}

/// The planes of the MLX affine weight `weight` and the scales and biases its place names, from
/// their headers alone.
///
/// # Errors
///
/// A format with no block type, a place that does not name a scales and a biases tensor, a
/// weight that is not U32 with a leading dim and a packed row, a row that is not whole groups, and
/// a scales or biases tensor that is not of the format's dtype (BF16, or F16 for
/// [`Format::MlxAffineF16`]) `[.., groups]` over the weight's leading dims, each by name.
fn mlx_planes<'a>(
    get: &impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    weight: &str,
    placed: &Placed,
) -> Result<MlxPlanes<'a>, Error> {
    let format = placed.format;
    let refuse = |name: &str, what: String| Error::Format {
        name: name.to_string(),
        format,
        what,
    };
    let (Some((bits, group, scale_dtype)), Some(ty)) = (format.mlx_affine(), format.card_type())
    else {
        return Err(refuse(
            weight,
            "is not an MLX affine tensor with a block type here".into(),
        ));
    };
    let [scales, biases] = placed.companions.as_slice() else {
        return Err(refuse(
            weight,
            format!(
                "its place names {:?}, not a scales and a biases tensor",
                placed.companions
            ),
        ));
    };
    let codes = stored(get, weight, Dtype::U32, refuse)?;
    let Some((&packed, leading)) = codes.shape.split_last().filter(|(_, l)| !l.is_empty()) else {
        return Err(refuse(
            weight,
            format!("shape {:?} is not rows of packed words", codes.shape),
        ));
    };
    if codes.shape.contains(&0) {
        return Err(refuse(
            weight,
            format!("shape {:?} has a zero dimension", codes.shape),
        ));
    }
    let k = packed
        .checked_mul(32 / bits)
        .filter(|k| k.is_multiple_of(group))
        .ok_or_else(|| {
            refuse(
                weight,
                format!("{packed} words a row are not whole groups of {group} values"),
            )
        })?;
    let rows = leading
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| refuse(weight, "row count overflows usize".into()))?;
    let mut grid = leading.to_vec();
    grid.push(k / group);
    let plane = |name: &String| {
        let raw = stored(get, name, scale_dtype, refuse)?;
        if raw.shape != grid {
            return Err(refuse(
                name,
                format!("shape {:?}; its weight's groups are {grid:?}", raw.shape),
            ));
        }
        Ok(raw)
    };
    Ok(MlxPlanes {
        ty,
        scales: plane(scales)?,
        biases: plane(biases)?,
        k,
        rows,
        groups: k / group,
        group_bytes: group * bits / 8,
        codes,
    })
}

/// The MLX affine weight `source` in GGUF's MLX_A4, MLX_A8, MLX_A4H or MLX_A8H block (the one its
/// format's card type names): each group's code words, then its scale, then its bias, as stored. A weight of more than two dims (a routed stack) keeps its
/// leading dims as ggml's trailing ones, `[k, rows, n_expert]`. Every block written passes
/// [`check_mlx_affine`], and a row whose block does not is refused by name.
///
/// # Errors
///
/// A refusal of `mlx_planes`, a byte count `usize` or the allocator cannot hold, and a group
/// whose scale or bias is no finite BF16 or F16 (as its type stores them) or whose top code leaves
/// f32's range, each by name.
fn mlx_affine<'a>(
    get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
    (weight, placed): (&str, &Placed),
    type_id: u32,
) -> Result<Normal, Error> {
    let p = mlx_planes(&get, weight, placed)?;
    let refuse = |what: String| Error::Format {
        name: weight.to_string(),
        format: placed.format,
        what,
    };
    let block = p.group_bytes + 4;
    let total = p
        .rows
        .checked_mul(p.groups)
        .and_then(|n| n.checked_mul(block))
        .ok_or_else(|| refuse("block byte count overflows usize".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve(total)
        .map_err(|e| refuse(format!("cannot reserve {total} block bytes: {e}")))?;
    let rows = p
        .codes
        .bytes
        .chunks_exact(p.groups * p.group_bytes)
        .zip(p.scales.bytes.chunks_exact(2 * p.groups))
        .zip(p.biases.bytes.chunks_exact(2 * p.groups));
    for (row, ((codes, scales), biases)) in rows.enumerate() {
        let at = bytes.len();
        for (codes, (scale, bias)) in codes.chunks_exact(p.group_bytes).zip(
            scales
                .as_chunks::<2>()
                .0
                .iter()
                .zip(biases.as_chunks::<2>().0),
        ) {
            bytes.extend_from_slice(codes);
            bytes.extend_from_slice(scale);
            bytes.extend_from_slice(bias);
        }
        check_mlx_affine(p.ty, &bytes[at..])
            .map_err(|source| refuse(format!("row {row}: {source}")))?;
    }
    let mut dims = vec![p.k as u64];
    dims.extend(p.codes.shape.iter().rev().skip(1).map(|&d| d as u64));
    Ok(Normal {
        type_id,
        dims,
        bytes,
        scale: vec![],
    })
}

#[cfg(test)]
mod tests {
    use gguf::quant::{GgmlType, bf16_bits_to_f32, dequant_row, fp32_to_bf16_bits, half_to_f32};
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::*;
    use safetensors::{Safetensors, write};

    const STEMS: [&str; 2] = [
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.1.gate_proj",
    ];

    fn fixture() -> Safetensors {
        Safetensors::parse(include_bytes!("../tests/fixtures/modelopt-nvfp4.safetensors").to_vec())
            .unwrap()
    }

    /// The place the plan gives the triple `stem`: its weight with its own block and tensor
    /// scales.
    fn own(stem: &str) -> (String, Placed) {
        let companions = ["weight_scale", "weight_scale_2"].map(|s| format!("{stem}.{s}"));
        (
            format!("{stem}.weight"),
            Placed {
                format: Format::Nvfp4,
                companions: companions.to_vec(),
            },
        )
    }

    /// The pairs [`normalize`] reads from owned places.
    fn sources(owned: &[(String, Placed)]) -> Vec<(&str, &Placed)> {
        owned.iter().map(|(w, p)| (w.as_str(), p)).collect()
    }

    fn lin<'a>(
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
        stem: &str,
    ) -> Result<Nvfp4, Error> {
        let (w, p) = own(stem);
        nvfp4(get, &[(&w, &p)]).map(packed)
    }

    fn stack<'a>(
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error>,
        stems: &[&str],
    ) -> Result<Nvfp4, Error> {
        let owned: Vec<_> = stems.iter().map(|s| own(s)).collect();
        nvfp4(get, &sources(&owned)).map(packed)
    }

    fn packed(n: Normal) -> Nvfp4 {
        Nvfp4 {
            dims: n.dims,
            blocks: n.bytes,
            scale: n.scale,
        }
    }

    /// The reference's row digests and tensor scale of `stem`.
    fn reference(stem: &str) -> (Vec<String>, f64) {
        let v: Value =
            serde_json::from_str(include_str!("../tests/fixtures/modelopt-nvfp4-ref.json"))
                .unwrap();
        let t = &v["tensors"][stem];
        let rows = t["rows"].as_array().unwrap();
        (
            rows.iter()
                .map(|r| r.as_str().unwrap().to_string())
                .collect(),
            t["scale"].as_f64().unwrap(),
        )
    }

    /// The sha256 of each row of `blocks` decoded by GGUF's NVFP4 decode and multiplied by
    /// `scale`, its f32 bits little-endian.
    fn decoded(blocks: &[u8], k: usize, scale: f32) -> Vec<String> {
        blocks
            .chunks_exact(k / 64 * 36)
            .map(|row| {
                let mut v = vec![0.0f32; k];
                dequant_row(GgmlType::NVFP4, row, &mut v).unwrap();
                let bits: Vec<u8> = v.iter().flat_map(|x| (x * scale).to_le_bytes()).collect();
                format!("{:x}", Sha256::digest(&bits))
            })
            .collect()
    }

    /// The oracle: every row of the synthetic tensor (every defined scale byte, every code at
    /// every position of a group) and of the real `gate_proj` rows, read here and decoded by
    /// GGUF's NVFP4 decode times the tensor scale, has the f32 bits the ModelOpt rule gives in
    /// `modelopt_nvfp4.py ref`. A nibble order other than the block's, a scale byte at another
    /// group, or a tensor scale from another expert changes a row's digest.
    #[test]
    fn linears_decode_to_the_modelopt_rule() {
        let st = fixture();
        for stem in ["synthetic", STEMS[0], STEMS[1]] {
            let t = lin(|n| st.raw(n), stem).unwrap();
            let (rows, scale) = reference(stem);
            let k = t.dims[0] as usize;
            assert_eq!(t.dims, [k as u64, rows.len() as u64], "{stem}");
            assert_eq!(t.scale, [scale as f32], "{stem}");
            assert_eq!(t.blocks.len(), rows.len() * k / 64 * 36, "{stem}");
            assert_eq!(decoded(&t.blocks, k, t.scale[0]), rows, "{stem}");
        }
    }

    /// A routed stack of two real experts with different tensor scales: the blocks are each
    /// expert's in expert order, `.scale` holds each expert's own scale, and every row decodes
    /// to the reference under its expert's scale. A stack's blocks are allocated at their length.
    ///
    /// FAIL-first: reserving each expert's blocks as it is appended grows three experts' to four.
    #[test]
    fn a_routed_stack_keeps_one_scale_an_expert() {
        let st = fixture();
        let three = stack(|n| st.raw(n), &[STEMS[0], STEMS[1], STEMS[0]]).unwrap();
        assert_eq!(three.blocks.capacity(), three.blocks.len());
        let t = stack(|n| st.raw(n), &STEMS).unwrap();
        assert_eq!(t.dims, [2560, 4, 2]);
        let per = t.blocks.len() / 2;
        for (e, stem) in STEMS.iter().enumerate() {
            let (rows, scale) = reference(stem);
            assert_eq!(t.scale[e], scale as f32, "{stem}");
            assert_eq!(
                decoded(&t.blocks[e * per..(e + 1) * per], 2560, t.scale[e]),
                rows,
                "{stem}"
            );
        }
        assert_ne!(t.scale[0], t.scale[1]);
    }

    /// Each expert reads the tensor scale its place names: one whose module stores none reads
    /// the one its parent module shares, beside one with its own, each in its own slot of the
    /// stack's `.scale`.
    ///
    /// FAIL-first: reading `<stem>.weight_scale_2` by name refuses expert 0 as missing.
    #[test]
    fn a_stack_reads_each_expert_s_planned_tensor_scale() {
        let codes = [0x11u8; 32];
        let scales = [0x38u8; 4];
        let (shared, local) = (2.0f32.to_le_bytes(), 3.0f32.to_le_bytes());
        let st = Safetensors::parse(write(&[
            ("e.0.weight", "U8", &[1, 32][..], &codes[..]),
            ("e.0.weight_scale", "F8_E4M3", &[1, 4], &scales),
            ("e.weight_scale_2", "F32", &[], &shared),
            ("e.1.weight", "U8", &[1, 32], &codes),
            ("e.1.weight_scale", "F8_E4M3", &[1, 4], &scales),
            ("e.1.weight_scale_2", "F32", &[], &local),
        ]))
        .unwrap();
        let config = crate::quant_config::Config::parse(
            &serde_json::json!({}),
            Some(&serde_json::json!({ "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": null,
                "quantized_layers": { "e": { "quant_algo": "NVFP4", "group_size": 16 } },
            }})),
        )
        .unwrap();
        let raws: Vec<_> = st.names().map(|n| (n, st.raw(n).unwrap())).collect();
        let plan = crate::quant_config::plan(
            &config,
            raws.iter().map(|(n, r)| (*n, r.dtype, r.shape)),
            |_| None,
        );
        let experts = ["e.0.weight", "e.1.weight"].map(|n| (n, &plan.placed[n]));
        let got = normalize(|n| st.raw(n), &experts, Use::Matrix(Plain::Q8_0)).unwrap();
        assert_eq!(got.scale, [2.0, 3.0]);
        assert_eq!(got.dims, [64, 1, 2]);
        let alone = normalize(|n| st.raw(n), &experts[..1], Use::Matrix(Plain::Q8_0)).unwrap();
        assert_eq!((alone.dims, alone.scale), (vec![64, 1], vec![2.0]));
    }

    /// The table: NVFP4 matrices and stacks keep their blocks and tensor scales;
    /// BF16 becomes ggml's reference Q8_0 of its values, their f32 widening, or itself; an F8
    /// row table keeps its bytes and reads its one BF16 scale; every pair without a line, a
    /// stack of anything but NVFP4 or FP8 block, an F32 asked for BF16 and a row Q8_0 does not
    /// divide are refused by name.
    #[test]
    fn the_table_writes_each_format_as_its_type() {
        let st = fixture();
        let get = |n: &str| st.raw(n);
        let placed = |format, companions: &[&str]| Placed {
            format,
            companions: companions.iter().map(|c| c.to_string()).collect(),
        };
        let owned = STEMS.map(own);
        let both = sources(&owned);
        let m = Use::Matrix(Plain::Q8_0);
        let one = normalize(get, &both[..1], m).unwrap();
        let first = lin(get, STEMS[0]).unwrap();
        assert_eq!(
            (one.type_id, &one.dims, &one.bytes, &one.scale),
            (40, &first.dims, &first.blocks, &first.scale)
        );
        let two = normalize(get, &both, m).unwrap();
        let exp = stack(get, &STEMS).unwrap();
        assert_eq!(
            (&two.dims, &two.bytes, &two.scale),
            (&exp.dims, &exp.blocks, &exp.scale)
        );
        let bf16_first = placed(Format::Bf16, &[]);
        let e = normalize(get, &[("b.weight", &bf16_first), both[0]], m).unwrap_err();
        assert!(
            e.to_string()
                .contains("NVFP4-ModelOpt in a stack whose first source b.weight is BF16"),
            "{e}"
        );
        let e = normalize(get, &[], m).unwrap_err();
        assert!(e.to_string().contains("has no source"), "{e}");
        let bare = placed(Format::Nvfp4, &[]);
        let e = normalize(get, &[(both[0].0, &bare)], m).unwrap_err();
        assert!(
            e.to_string()
                .contains("its place names [], not a block scale and a tensor scale"),
            "{e}"
        );

        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 20.5) / 7.0).collect();
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let wide: Vec<f32> = bf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect();
        let f8 = [0x38u8, 0x40, 0x30, 0x00, 0xB8, 0x7E];
        let f32s: Vec<u8> = wide.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale = 0x3a00u16.to_le_bytes();
        let mut nan = bf16[..128].to_vec();
        nan[2 * 37..][..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
        let file = write(&[
            ("b.weight", "BF16", &[2, 32][..], &bf16[..]),
            ("f.weight", "F32", &[2, 32], &f32s),
            ("odd.weight", "BF16", &[4, 16], &bf16),
            ("t.weight", "F8_E4M3", &[3, 2], &f8),
            ("t.scale", "BF16", &[1], &scale),
            ("nan.weight", "BF16", &[1, 64], &nan),
        ]);
        let st = Safetensors::parse(file).unwrap();
        let get = |n: &str| st.raw(n);
        let b = placed(Format::Bf16, &[]);
        let q = normalize(get, &[("b.weight", &b)], m).unwrap();
        let want: Vec<u8> = wide
            .as_chunks::<32>()
            .0
            .iter()
            .flat_map(|blk| Q8Block::quantize_ref(blk).unwrap().to_bytes())
            .collect();
        assert_eq!((q.type_id, &q.dims, &q.bytes), (8, &vec![32, 2], &want));
        let f = normalize(get, &[("b.weight", &b)], Use::Matrix(Plain::F32)).unwrap();
        assert_eq!((f.type_id, f.bytes.len()), (0, 256));
        assert!(
            f.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(&wide)
                .all(|(x, w)| f32::from_le_bytes(*x) == *w)
        );
        let kept = normalize(get, &[("b.weight", &b)], Use::Matrix(Plain::Bf16)).unwrap();
        assert_eq!((kept.type_id, &kept.bytes), (30, &bf16));
        let f32 = placed(Format::F32, &[]);
        let kept = normalize(get, &[("f.weight", &f32)], Use::Matrix(Plain::F32)).unwrap();
        assert_eq!((kept.type_id, &kept.bytes), (0, &f32s));
        let q = normalize(get, &[("f.weight", &f32)], m).unwrap();
        assert_eq!((q.type_id, &q.bytes), (8, &want));
        let zero = vec![0u8; bf16.len()];
        let get_pair = |name: &str| {
            if name == "zero.weight" {
                Ok(Raw {
                    dtype: Dtype::Bf16,
                    shape: &[2, 32],
                    bytes: &zero,
                })
            } else {
                get(name)
            }
        };
        let pair = [("b.weight", &b), ("zero.weight", &b)];
        let stack = normalize(get_pair, &pair, Use::Matrix(Plain::Bf16)).unwrap();
        assert_eq!(
            (&stack.dims, &stack.bytes),
            (&vec![32, 2, 2], &[&bf16[..], &zero].concat())
        );
        let ragged = [("b.weight", &b), ("odd.weight", &b)];
        let e = normalize(get, &ragged, Use::Matrix(Plain::F32)).unwrap_err();
        assert!(e.to_string().contains("odd.weight"), "{e}");
        let rows = normalize(
            get,
            &[("t.weight", &placed(Format::Fp8Tensor, &["t.scale"]))],
            Use::Rows,
        )
        .unwrap();
        assert_eq!(
            (rows.type_id, &rows.dims, &rows.bytes[..], &rows.scale[..]),
            (
                2000,
                &vec![2, 3],
                &f8[..],
                &[f32::from_bits(0x3a00u32 << 16)][..]
            )
        );

        // PIN(2026-10-20): positive MXFP8 matrix and NVFP4 row tests replace their unsupported-pair cases.
        let refused = |names: &[&str], p: Placed, role: Use, want: &str| {
            let names: Vec<_> = names.iter().map(|n| (*n, &p)).collect();
            let e = normalize(get, &names, role).unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        };
        refused(
            &["b.weight"],
            placed(Format::Mxfp8, &[]),
            Use::Rows,
            "b.weight: MXFP8 read as Rows has no GGUF type here",
        );
        // PIN(2026-10-20): FP8 block 128x128 read as a matrix is FP8_B128 now
        // (fp8_block_weights_take_their_square_s_scale_a_row_block); read as rows it has no type.
        refused(
            &["b.weight"],
            placed(
                Format::Fp8Block {
                    rows: 128,
                    cols: 128,
                },
                &[],
            ),
            Use::Rows,
            "FP8-block 128x128 read as Rows",
        );
        refused(
            &["t.weight"],
            placed(Format::Nvfp4Csf, &[]),
            Use::Rows,
            "NVFP4-csf read as Rows",
        );
        refused(
            &["b.weight", "b.weight"],
            b,
            m,
            "in a stack of 2: only NVFP4 and FP8_B128 experts stack here",
        );
        refused(
            &["f.weight"],
            placed(Format::F32, &[]),
            Use::Matrix(Plain::Bf16),
            "has no GGUF type here",
        );
        refused(
            &["odd.weight"],
            placed(Format::Bf16, &[]),
            m,
            "a row is not whole 32-value Q8_0 blocks",
        );
        refused(&["nan.weight"], placed(Format::Bf16, &[]), m, "nan.weight");
        let mut missed = vec![];
        for (dtype, format) in [(Dtype::F32, Format::F32), (Dtype::Bf16, Format::Bf16)] {
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let bits = x.to_bits();
                let bytes = if dtype == Dtype::F32 {
                    bits.to_le_bytes().to_vec()
                } else {
                    ((bits >> 16) as u16).to_le_bytes().to_vec()
                }
                .repeat(32);
                let st =
                    Safetensors::parse(write(&[("bad.weight", dtype.name(), &[1, 32], &bytes)]))
                        .unwrap();
                for plain in [Plain::F32, Plain::Bf16, Plain::Q8_0] {
                    if type_of(format, Use::Matrix(plain)).is_none() {
                        continue;
                    }
                    let got = normalize(
                        |n| st.raw(n),
                        &[("bad.weight", &placed(format, &[]))],
                        Use::Matrix(plain),
                    );
                    if !got
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("bad.weight"))
                    {
                        missed.push(format!("{dtype:?} {plain:?} {x}: {got:?}"));
                    }
                }
            }
        }
        for (byte, scale, dtype, want) in [
            (0x38, f32::INFINITY, Dtype::F32, "t.scale"),
            (0x38, f32::NEG_INFINITY, Dtype::F32, "t.scale"),
            (0x38, f32::NAN, Dtype::F32, "t.scale"),
            (0x38, f32::INFINITY, Dtype::Bf16, "t.scale"),
            (0x38, f32::NEG_INFINITY, Dtype::Bf16, "t.scale"),
            (0x38, f32::NAN, Dtype::Bf16, "t.scale"),
            (0x7f, 1.0, Dtype::F32, "t.weight"),
            (0xff, 1.0, Dtype::F32, "t.weight"),
            (0x7e, f32::MAX, Dtype::F32, "t.weight"),
        ] {
            let bytes = if dtype == Dtype::F32 {
                scale.to_le_bytes().to_vec()
            } else {
                ((scale.to_bits() >> 16) as u16).to_le_bytes().to_vec()
            };
            let st = Safetensors::parse(write(&[
                ("t.weight", "F8_E4M3", &[1, 1], &[byte]),
                ("t.scale", dtype.name(), &[1], &bytes),
            ]))
            .unwrap();
            let got = normalize(
                |n| st.raw(n),
                &[("t.weight", &placed(Format::Fp8Tensor, &["t.scale"]))],
                Use::Rows,
            );
            if !got.as_ref().is_err_and(|e| e.to_string().contains(want)) {
                missed.push(format!("FP8 {byte:#x} scale {scale}: {got:?}"));
            }
        }
        for (code, scale, ok) in [
            (0x77, f32::MAX, false),
            (0x00, f32::MAX, true),
            (0x77, 0.0, true),
        ] {
            let st = Safetensors::parse(write(&[
                ("t.weight", "U8", &[1, 32], &[code; 32]),
                ("t.weight_scale", "F8_E4M3", &[1, 4], &[0x7e; 4]),
                ("t.weight_scale_2", "F32", &[], &scale.to_le_bytes()),
            ]))
            .unwrap();
            let t = own("t");
            let got = normalize(|n| st.raw(n), &sources(std::slice::from_ref(&t)), m);
            if got.is_ok() != ok
                || (!ok
                    && !got
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("t.weight_scale_2")))
            {
                missed.push(format!("NVFP4 {code:#x} scale {scale}: {got:?}"));
            }
        }
        assert!(missed.is_empty(), "accepted: {missed:?}");
    }

    /// An MXFP8 weight moves into [`GgmlType::MXFP8`] bit for bit, whatever its role's plain
    /// type: block `(r, b)` is scale byte `[r, b]` then value bytes `[r, 32 b ..][..32]`, so
    /// splitting the blocks gives back the checkpoint's bytes, and `dequant_row` gives each
    /// value as its E4M3 byte times `2^(e - 127)` (decoded here from the bits). A stack, a block
    /// `check_mxfp8` refuses, and a weight or scale of the wrong shape are refused by name.
    #[test]
    fn mxfp8_weights_move_into_the_mxfp8_block_unchanged() {
        let (rows, k) = (3, 64);
        let values: Vec<u8> = (0..rows * k)
            .map(|i| (i * 37 % 256) as u8)
            .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
            .collect();
        let scales = [127u8, 120, 134, 1, 140, 0];
        let file = |values: &[u8], scales: &[u8], scale_shape: &[usize]| {
            Safetensors::parse(write(&[
                ("m.weight", "F8_E4M3", &[rows, k][..], values),
                ("m.weight_scale", "U8", scale_shape, scales),
            ]))
            .unwrap()
        };
        let p = Placed {
            format: Format::Mxfp8,
            companions: vec!["m.weight_scale".into()],
        };
        let st = file(&values, &scales, &[rows, 2]);
        let get = |n: &str| st.raw(n);
        for plain in [Plain::Q8_0, Plain::F32, Plain::Bf16] {
            let n = normalize(get, &[("m.weight", &p)], Use::Matrix(plain)).unwrap();
            assert_eq!(
                (n.type_id, &n.dims, n.bytes.len(), n.scale.len()),
                (GgmlType::MXFP8.as_u32(), &vec![64, 3], 6 * 33, 0)
            );
            let (mut back_scales, mut back_values) = (vec![], vec![]);
            for block in n.bytes.as_chunks::<33>().0 {
                back_scales.push(block[0]);
                back_values.extend_from_slice(&block[1..]);
            }
            assert_eq!((&back_scales[..], &back_values), (&scales[..], &values));
            let mut got = vec![0f32; rows * k];
            dequant_row(GgmlType::MXFP8, &n.bytes, &mut got).unwrap();
            for (i, (&g, &v)) in got.iter().zip(&values).enumerate() {
                let (exp, man) = (i32::from((v >> 3) & 0xF), f64::from(v & 7));
                let mag = if exp == 0 {
                    man / 8.0 * 2f64.powi(-6)
                } else {
                    (1.0 + man / 8.0) * 2f64.powi(exp - 7)
                };
                let sign = if v & 0x80 == 0 { 1.0 } else { -1.0 };
                let want = sign * mag * 2f64.powi(i32::from(scales[i / 32]) - 127);
                assert_eq!(
                    g.to_bits(),
                    (want as f32).to_bits(),
                    "value {i} byte {v:#04x}"
                );
            }
        }

        let refused = |st: &Safetensors, sources: &[(&str, &Placed)], want: &str| {
            let e = normalize(|n| st.raw(n), sources, Use::Matrix(Plain::Q8_0))
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e}");
        };
        let two = [("m.weight", &p), ("m.weight", &p)];
        refused(
            &st,
            &two,
            "in a stack of 2: only NVFP4 and FP8_B128 experts stack here",
        );
        let mut nan = scales;
        nan[3] = 0xFF;
        refused(
            &file(&values, &nan, &[rows, 2]),
            &two[..1],
            "m.weight: MXFP8 row 1:",
        );
        let mut bad = values.clone();
        bad[2 * k + 5] = 0x7F;
        refused(&file(&bad, &scales, &[rows, 2]), &two[..1], "MXFP8 row 2:");
        refused(
            &file(&values, &scales, &[rows * 2]),
            &two[..1],
            "m.weight_scale U8 [6] (6 bytes): not the MXFP8 grid of one E8M0 scale a block, as MXFP8 holds it",
        );
        let bare = Placed {
            format: Format::Mxfp8,
            companions: vec![],
        };
        refused(&st, &[("m.weight", &bare)], "its place names no scale");
        let odd = Safetensors::parse(write(&[
            ("m.weight", "F8_E4M3", &[4, 48][..], &values),
            ("m.weight_scale", "U8", &[4, 1], &scales[..4]),
        ]))
        .unwrap();
        refused(
            &odd,
            &two[..1],
            "F8_E4M3 [4, 48] (192 bytes), not F8_E4M3 rows of whole 32-value blocks",
        );
    }

    /// An F8_E4M3 weight moves into its format's card type with its values unchanged, whatever
    /// its role's plain type, each row block the scale of the checkpoint's block it lies in, over
    /// 130 rows of 256 values: one scale a 128 by 128 block (an F32 `[r, c]`, BF16 ModelOpt
    /// `[r, 1, c, 1]` or E8M0 grid, so the last two rows read the second row of squares), a row
    /// (F32 `[rows]`, BF16 `[rows, 1]`) or the tensor (F32 `[]`, BF16 `[1]`) in FP8_B128, and one
    /// E8M0 a 32 by 32 block in MXFP8 (five rows of squares). Block `(row, b)` is that scale (an
    /// f32, or the E8M0 byte) then the row's value bytes `[w b ..][..w]`, and `dequant_row` gives
    /// each value as its E4M3 byte times that scale, both decoded here from the bits. A stack of
    /// two is the two experts' blocks one after another, `[k, rows, 2]`. An expert of another
    /// shape than the first's, a block the type's check refuses, a row that is not whole blocks
    /// of the type, a grid of another shape or dtype, an MXFP8 stack, a place with no scale and
    /// a block no card type holds are refused by name.
    #[test]
    fn fp8_weights_take_their_block_s_scale_a_row_block() {
        let (rows, k) = (130, 256);
        let values: Vec<u8> = (0..rows * k)
            .map(|i| (i * 37 % 256) as u8)
            .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
            .collect();
        // Bytes of `v` as F32 or BF16 (whose values here are BF16's).
        let enc = |dtype: &str, v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| match dtype {
                    "F32" => x.to_le_bytes().to_vec(),
                    _ => ((x.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                })
                .collect()
        };
        let e8m0 =
            |e: &[u8]| -> Vec<f32> { e.iter().map(|&e| 2f32.powi(i32::from(e) - 127)).collect() };
        let grid = [0.5f32, f32::from_bits(0x3951 << 16), -3.0, 2f32.powi(-20)];
        let (f32s, bf16s) = (enc("F32", &grid), enc("BF16", &grid));
        let e8m0s = [126u8, 127, 128, 107];
        let row_scales: Vec<f32> = (0..rows as u32)
            .map(|r| f32::from_bits((0x3800 + 0x93 * r % 0x1000) << 16))
            .collect();
        let tensor_scale = [f32::from_bits(0x3e45 << 16)];
        let e32: Vec<u8> = (0..40).map(|i| 112 + (i * 7 % 29) as u8).collect();
        let file = |values: &[u8], dtype: &str, scales: &[u8], scale_shape: &[usize]| {
            Safetensors::parse(write(&[
                ("f.weight", "F8_E4M3", &[rows, k][..], values),
                ("f.weight_scale_inv", dtype, scale_shape, scales),
            ]))
            .unwrap()
        };
        let placed = |format| Placed {
            format,
            companions: vec!["f.weight_scale_inv".into()],
        };
        let b128 = Format::Fp8Block {
            rows: 128,
            cols: 128,
        };
        let b32 = Format::Fp8Block { rows: 32, cols: 32 };
        let p = placed(b128);
        let (channel, tensor) = (Format::Fp8Channel, Format::Fp8Tensor);
        type Case<'a> = (Format, &'a str, Vec<u8>, &'a [usize], Vec<f32>, [usize; 2]);
        let cases: [Case; 8] = [
            (
                b128,
                "F32",
                f32s.clone(),
                &[2, 2],
                grid.to_vec(),
                [128, 128],
            ),
            (
                b128,
                "BF16",
                bf16s.clone(),
                &[2, 1, 2, 1],
                grid.to_vec(),
                [128, 128],
            ),
            (
                b128,
                "F8_E8M0",
                e8m0s.to_vec(),
                &[2, 2],
                e8m0(&e8m0s),
                [128, 128],
            ),
            (
                channel,
                "F32",
                enc("F32", &row_scales),
                &[rows],
                row_scales.clone(),
                [1, k],
            ),
            (
                channel,
                "BF16",
                enc("BF16", &row_scales),
                &[rows, 1],
                row_scales.clone(),
                [1, k],
            ),
            (
                tensor,
                "F32",
                enc("F32", &tensor_scale),
                &[],
                tensor_scale.to_vec(),
                [rows, k],
            ),
            (
                tensor,
                "BF16",
                enc("BF16", &tensor_scale),
                &[1],
                tensor_scale.to_vec(),
                [rows, k],
            ),
            (b32, "F8_E8M0", e32.clone(), &[5, 8], e8m0(&e32), [32, 32]),
        ];
        for (format, dtype, scales, shape, want_grid, [a, b]) in &cases {
            let st = file(&values, dtype, scales, shape);
            let (ty, w, unit) = match format {
                Format::Fp8Block { cols: 32, .. } => (GgmlType::MXFP8, 32, 1),
                _ => (GgmlType::FP8_B128, 128, 4),
            };
            // The scale of value `i`: its block's in the grid, `k / b` (rounded up) a grid row.
            let scale_of = |i: usize| want_grid[i / k / a * k.div_ceil(*b) + i % k / b];
            for plain in [Plain::Q8_0, Plain::F32, Plain::Bf16] {
                let p = placed(*format);
                let n = normalize(|n| st.raw(n), &[("f.weight", &p)], Use::Matrix(plain)).unwrap();
                assert_eq!(
                    (n.type_id, &n.dims, n.bytes.len(), n.scale.len()),
                    (ty.as_u32(), &vec![256, 130], rows * k / w * (w + unit), 0),
                    "{format} {dtype}"
                );
                let mut back_values = vec![];
                for (blk, block) in n.bytes.chunks_exact(w + unit).enumerate() {
                    let d = match unit {
                        4 => f32::from_le_bytes([block[0], block[1], block[2], block[3]]),
                        _ => 2f32.powi(i32::from(block[0]) - 127),
                    };
                    assert_eq!(
                        d.to_bits(),
                        scale_of(blk * w).to_bits(),
                        "{format} {dtype} block {blk}"
                    );
                    back_values.extend_from_slice(&block[unit..]);
                }
                assert_eq!(back_values, values, "{format} {dtype}");
                let mut got = vec![0f32; rows * k];
                dequant_row(ty, &n.bytes, &mut got).unwrap();
                for (i, (&g, &v)) in got.iter().zip(&values).enumerate() {
                    let (exp, man) = (i32::from((v >> 3) & 0xF), f64::from(v & 7));
                    let mag = if exp == 0 {
                        man / 8.0 * 2f64.powi(-6)
                    } else {
                        (1.0 + man / 8.0) * 2f64.powi(exp - 7)
                    };
                    let sign = if v & 0x80 == 0 { 1.0 } else { -1.0 };
                    let want = (sign * mag * f64::from(scale_of(i))) as f32;
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{format} {dtype} value {i} byte {v:#04x}"
                    );
                }
            }
        }

        let st = file(&values, "F32", &f32s, &[2, 2]);
        let refused = |st: &Safetensors, sources: &[(&str, &Placed)], want: &str| {
            let e = normalize(|n| st.raw(n), sources, Use::Matrix(Plain::Q8_0))
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e}");
        };
        for shape in [[0, 128], [1, 0], [1, 96], [usize::MAX, 128]] {
            let e = normalize(
                |name| {
                    if name == "f.weight" {
                        Ok(Raw {
                            dtype: Dtype::F8E4M3,
                            shape: &shape,
                            bytes: &[],
                        })
                    } else {
                        st.raw(name)
                    }
                },
                &[("f.weight", &p)],
                Use::Matrix(Plain::Q8_0),
            )
            .unwrap_err()
            .to_string();
            assert!(
                e.contains("f.weight: FP8-block 128x128") && e.contains("whole 128-value blocks"),
                "{e}"
            );
        }
        for (dtype, bytes) in [
            ("F32", &f32s[..]),
            ("BF16", &bf16s[..]),
            ("F8_E8M0", &e8m0s[..]),
        ] {
            for shape in [&[0, 2][..], &[1, 2], &[2, 0], &[2, 1]] {
                let len = shape.iter().product::<usize>() * (bytes.len() / 4);
                refused(
                    &file(&values, dtype, &bytes[..len], shape),
                    &[("f.weight", &p)],
                    "f.weight_scale_inv",
                );
            }
            let mut nan = bytes.to_vec();
            match dtype {
                "F32" => nan[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
                "BF16" => nan[..2].copy_from_slice(&0x7fc0u16.to_le_bytes()),
                _ => nan[0] = 255,
            }
            refused(
                &file(&values, dtype, &nan, &[2, 2]),
                &[("f.weight", &p)],
                "not finite",
            );
        }
        let two = [("f.weight", &p), ("f.weight", &p)];
        let stack = normalize(|n| st.raw(n), &two, Use::Matrix(Plain::Q8_0)).unwrap();
        let one = normalize(|n| st.raw(n), &two[..1], Use::Matrix(Plain::Q8_0)).unwrap();
        assert_eq!(
            (
                &stack.dims,
                &stack.bytes[..one.bytes.len()],
                &stack.bytes[one.bytes.len()..]
            ),
            (&vec![256, 130, 2], &one.bytes[..], &one.bytes[..])
        );
        let short = Safetensors::parse(write(&[
            ("f.weight", "F8_E4M3", &[rows, k][..], &values),
            ("f.weight_scale_inv", "F32", &[2, 2], &f32s),
            ("g.weight", "F8_E4M3", &[2, 256][..], &values[..512]),
            ("g.weight_scale_inv", "F32", &[1, 2], &f32s[..8]),
        ]))
        .unwrap();
        let g = Placed {
            format: p.format,
            companions: vec!["g.weight_scale_inv".into()],
        };
        refused(
            &short,
            &[("f.weight", &p), ("g.weight", &g)],
            "g.weight: FP8-block 128x128 rows and k [2, 256]; the stack's first expert f.weight has [130, 256]",
        );
        let mut nan = grid;
        nan[3] = f32::NAN;
        let nan: Vec<u8> = nan.iter().flat_map(|v| v.to_le_bytes()).collect();
        refused(
            &file(&values, "F32", &nan, &[2, 2]),
            &two[..1],
            "f.weight: FP8-block 128x128 row 128:",
        );
        let mut bad = values.clone();
        bad[2 * k + 5] = 0x7F;
        refused(
            &file(&bad, "F32", &f32s, &[2, 2]),
            &two[..1],
            "FP8-block 128x128 row 2:",
        );
        refused(
            &file(&values, "F32", &f32s, &[4]),
            &two[..1],
            "f.weight_scale_inv F32 [4] (16 bytes): not the FP8-block 128x128 grid of one F32, BF16 or E8M0 scale a block, as FP8_B128 holds it",
        );
        refused(
            &file(&values, "F16", &bf16s, &[2, 2]),
            &two[..1],
            "f.weight_scale_inv F16 [2, 2]",
        );
        let bare = Placed {
            format: p.format,
            companions: vec![],
        };
        refused(&st, &[("f.weight", &bare)], "its place names no scale");
        let odd = Safetensors::parse(write(&[
            ("f.weight", "F8_E4M3", &[2, 192][..], &values[..384]),
            ("f.weight_scale_inv", "F32", &[1, 2], &f32s[..8]),
        ]))
        .unwrap();
        refused(
            &odd,
            &two[..1],
            "F8_E4M3 [2, 192] (384 bytes), not F8_E4M3 rows of whole 128-value blocks",
        );
        for format in [channel, tensor] {
            refused(
                &odd,
                &[("f.weight", &placed(format))],
                &format!(
                    "f.weight: {format} F8_E4M3 [2, 192] (384 bytes), not F8_E4M3 rows of whole 128-value blocks"
                ),
            );
        }
        refused(
            &file(&values, "F32", &enc("F32", &[1.0; 131]), &[131]),
            &[("f.weight", &placed(channel))],
            "f.weight_scale_inv F32 [131] (524 bytes): not the FP8-channel grid",
        );
        refused(
            &file(&values, "F32", &enc("F32", &row_scales), &[1, rows]),
            &[("f.weight", &placed(channel))],
            "f.weight_scale_inv F32 [1, 130] (520 bytes): not the FP8-channel grid",
        );
        refused(
            &file(&values, "F32", &f32s[..8], &[2]),
            &[("f.weight", &placed(tensor))],
            "f.weight_scale_inv F32 [2] (8 bytes): not the FP8-tensor grid",
        );
        let st32 = file(&values, "F8_E8M0", &e32, &[5, 8]);
        let p32 = placed(b32);
        refused(
            &file(&values, "F32", &enc("F32", &e8m0(&e32)), &[5, 8]),
            &[("f.weight", &p32)],
            "not the FP8-block 32x32 grid of one E8M0 scale a block, as MXFP8 holds it",
        );
        refused(
            &st32,
            &[("f.weight", &p32), ("f.weight", &p32)],
            "in a stack of 2: only NVFP4 and FP8_B128 experts stack here",
        );
        let pc = placed(channel);
        let stc = file(&values, "F32", &enc("F32", &row_scales), &[rows]);
        let stack = normalize(
            |n| stc.raw(n),
            &[("f.weight", &pc); 2],
            Use::Matrix(Plain::Q8_0),
        )
        .unwrap();
        let one = normalize(
            |n| stc.raw(n),
            &[("f.weight", &pc)],
            Use::Matrix(Plain::Q8_0),
        )
        .unwrap();
        assert_eq!(
            (&stack.dims, stack.bytes.len()),
            (&vec![256, 130, 2], 2 * one.bytes.len())
        );
        assert!(
            stack
                .bytes
                .chunks_exact(one.bytes.len())
                .all(|e| e == one.bytes)
        );
        // PIN(2026-10-21): FP8 block 32x32 read as a matrix is MXFP8 now (the case above); a
        // block of 48 values a row lies in no card type's blocks.
        refused(
            &st,
            &[("f.weight", &placed(Format::Fp8Block { rows: 32, cols: 48 }))],
            "FP8-block 32x48 read as Matrix(Q8_0) has no GGUF type here",
        );
    }

    /// An F8_E4M3 weight read as [`Use::Q8_0`] is its exact values rounded once more: each value
    /// its E4M3 byte (decoded here from its bits) times the f32 scale of the checkpoint's block it
    /// lies in, each 32 of a row a Q8_0 block by ggml's reference rule ([`Q8Block::quantize_ref`]),
    /// over 130 rows of 256 values with one scale a 128 by 128 block (the second row of squares
    /// starts at row 128), a row and the tensor. BF16 is written as `Plain::Q8_0` writes it. An
    /// MXFP8 weight, a block no FP8_B128 holds, a stack, a width of 96 and a byte E4M3 has no
    /// value for are refused by name.
    ///
    /// FAIL-first: in `fp8_q8_0`, setting each block's four scale bytes to `1.0f32` after
    /// `fp8_append` (`for blk in blocks.chunks_exact_mut(132) { blk[..4].copy_from_slice(&1.0f32.to_le_bytes()); }`)
    /// writes another `d` in every block, and block 0 (scale 0.5) differs.
    #[test]
    fn an_fp8_weight_read_as_q8_0_is_its_exact_values_quantized_once() {
        let (rows, k) = (130, 256);
        let values: Vec<u8> = (0..rows * k)
            .map(|i| (i * 37 % 256) as u8)
            .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
            .collect();
        let e4m3 = |b: u8| -> f32 {
            let (exp, man) = (i32::from((b >> 3) & 0xF), f64::from(b & 7));
            let mag = if exp == 0 {
                man / 8.0 * 2f64.powi(-6)
            } else {
                (1.0 + man / 8.0) * 2f64.powi(exp - 7)
            };
            (if b & 0x80 == 0 { mag } else { -mag }) as f32
        };
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let grid = [0.5f32, f32::from_bits(0x3951 << 16), -3.0, 2f32.powi(-20)];
        // Row scales from 2^-15 to just under 1: with E4M3's largest value 448 every Q8_0 block's
        // scale stays inside F16, which a larger checkpoint scale would leave (refused below).
        let row_scales: Vec<f32> = (0..rows as u32)
            .map(|r| f32::from_bits((0x3800 + 0x93 * r % 0x400) << 16))
            .collect();
        let tensor_scale = [f32::from_bits(0x3e45 << 16)];
        let file = |scales: &[u8], shape: &[usize]| {
            Safetensors::parse(write(&[
                ("f.weight", "F8_E4M3", &[rows, k][..], &values[..]),
                ("f.weight_scale_inv", "F32", shape, scales),
            ]))
            .unwrap()
        };
        let placed = |format| Placed {
            format,
            companions: vec!["f.weight_scale_inv".into()],
        };
        let b128 = Format::Fp8Block {
            rows: 128,
            cols: 128,
        };
        type Scale<'a> = &'a dyn Fn(usize) -> f32;
        let by_square: Scale = &|i| grid[i / k / 128 * 2 + i % k / 128];
        let by_row: Scale = &|i| row_scales[i / k];
        let by_tensor: Scale = &|_| tensor_scale[0];
        let cases = [
            (b128, f32_bytes(&grid), vec![2, 2], by_square),
            (
                Format::Fp8Channel,
                f32_bytes(&row_scales),
                vec![rows],
                by_row,
            ),
            (
                Format::Fp8Tensor,
                f32_bytes(&tensor_scale),
                vec![],
                by_tensor,
            ),
        ];
        for (format, scales, shape, scale_of) in &cases {
            let st = file(scales, shape);
            let n = normalize(|n| st.raw(n), &[("f.weight", &placed(*format))], Use::Q8_0).unwrap();
            assert_eq!(
                (n.type_id, &n.dims, n.scale.len(), n.bytes.len()),
                (
                    GgmlType::Q8_0.as_u32(),
                    &vec![256, 130],
                    0,
                    rows * k / 32 * 34
                ),
                "{format}"
            );
            for (blk, (got, chunk)) in n
                .bytes
                .as_chunks::<34>()
                .0
                .iter()
                .zip(values.as_chunks::<32>().0)
                .enumerate()
            {
                let x: [f32; 32] = std::array::from_fn(|j| e4m3(chunk[j]) * scale_of(blk * 32));
                let want = Q8Block::quantize_ref(&x).unwrap().to_bytes();
                assert_eq!(got.as_slice(), &want[..], "{format} Q8_0 block {blk}");
            }
        }

        // A scale whose Q8_0 block scale F16 cannot hold is refused by name, never written as inf.
        let huge = f32_bytes(&vec![30464.0f32; rows]);
        let st = file(&huge, &[rows]);
        let err = normalize(
            |n| st.raw(n),
            &[("f.weight", &placed(Format::Fp8Channel))],
            Use::Q8_0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unrepresentable scale"), "{err}");

        let bf16: Vec<u8> = (0..128)
            .map(|i| (((i as f32 - 60.0) / 7.0).to_bits() >> 16) as u16)
            .flat_map(u16::to_le_bytes)
            .collect();
        let bf =
            Safetensors::parse(write(&[("b.weight", "BF16", &[2, 64][..], &bf16[..])])).unwrap();
        let pb = Placed {
            format: Format::Bf16,
            companions: vec![],
        };
        let source = [("b.weight", &pb)];
        assert_eq!(
            normalize(|n| bf.raw(n), &source, Use::Q8_0).unwrap(),
            normalize(|n| bf.raw(n), &source, Use::Matrix(Plain::Q8_0)).unwrap()
        );

        let st = file(&f32_bytes(&grid), &[2, 2]);
        let refused = |st: &Safetensors, sources: &[(&str, &Placed)], want: &str| {
            let e = normalize(|n| st.raw(n), sources, Use::Q8_0)
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e}");
        };
        for format in [
            Format::Mxfp8,
            Format::Fp8Block { rows: 32, cols: 32 },
            Format::Fp8Block { rows: 32, cols: 48 },
        ] {
            refused(
                &st,
                &[("f.weight", &placed(format))],
                "read as Q8_0 has no GGUF type here",
            );
        }
        let p = placed(b128);
        refused(
            &st,
            &[("f.weight", &p), ("f.weight", &p)],
            "in a stack of 2: only NVFP4 and FP8_B128 experts stack here",
        );
        let mut bad = values.clone();
        bad[2 * k + 5] = 0x7F;
        refused(
            &Safetensors::parse(write(&[
                ("f.weight", "F8_E4M3", &[rows, k][..], &bad[..]),
                (
                    "f.weight_scale_inv",
                    "F32",
                    &[2, 2][..],
                    &f32_bytes(&grid)[..],
                ),
            ]))
            .unwrap(),
            &[("f.weight", &p)],
            "FP8-block 128x128 row 2:",
        );
        let odd = Safetensors::parse(write(&[
            ("f.weight", "F8_E4M3", &[2, 96][..], &values[..192]),
            (
                "f.weight_scale_inv",
                "F32",
                &[1, 1][..],
                &f32_bytes(&grid)[..4],
            ),
        ]))
        .unwrap();
        refused(&odd, &[("f.weight", &p)], "whole 128-value blocks");
    }

    /// The compressed-scale lines and the row line. A `csf` and a `csf_32x2_v1` weight, a
    /// linear and a two-expert stack, normalize to the bytes and scales of the same weight
    /// with its plain `weight_scale` (the fixture's encoded scales beside the bytes they encode).
    /// An NVFP4 row table keeps each group's scale byte then its 8 code bytes, its tensor scale
    /// the companion's; a scale byte that is no UE4M3 scale is refused naming its row and group.
    #[test]
    fn compressed_scales_and_row_tables_take_their_lines() {
        let csf =
            Safetensors::parse(include_bytes!("../tests/fixtures/csf-scales.safetensors").to_vec())
                .unwrap();
        let m = Use::Matrix(Plain::Q8_0);
        for (stem, format, rows, groups) in [
            ("syn_csf", Format::Nvfp4Csf, 48, 20),
            ("syn_csf32k", Format::Nvfp4Csf32, 64, 8),
        ] {
            let half = groups * 8;
            let codes: Vec<u8> = (0..rows * half).map(|i| (i * 37 % 251) as u8).collect();
            let two = 2.0f32.to_le_bytes();
            let plain_scale = csf
                .raw(&format!("{stem}.weight_scale"))
                .unwrap()
                .bytes
                .to_vec();
            let mut list: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
                ("c.weight".into(), "U8", vec![rows, half], codes.clone()),
                ("c.weight_scale_2".into(), "F32", vec![], two.to_vec()),
                ("p.weight".into(), "U8", vec![rows, half], codes),
                (
                    "p.weight_scale".into(),
                    "F8_E4M3",
                    vec![rows, groups],
                    plain_scale,
                ),
                ("p.weight_scale_2".into(), "F32", vec![], two.to_vec()),
            ];
            for n in csf
                .names()
                .filter(|n| n.starts_with(&format!("{stem}.weight_scale.")))
            {
                let raw = csf.raw(n).unwrap();
                let part = n.strip_prefix(stem).unwrap();
                list.push((
                    format!("c{part}"),
                    raw.dtype.name(),
                    raw.shape.to_vec(),
                    raw.bytes.to_vec(),
                ));
            }
            let refs: Vec<(&str, &str, &[usize], &[u8])> = list
                .iter()
                .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
                .collect();
            let st = Safetensors::parse(write(&refs)).unwrap();
            let get = |n: &str| st.raw(n);
            let c = Placed {
                format,
                companions: list
                    .iter()
                    .map(|(n, ..)| n)
                    .filter(|n| n.starts_with("c.weight_scale."))
                    .chain(["c.weight_scale_2".to_string()].iter())
                    .cloned()
                    .collect(),
            };
            let (pw, p) = own("p");
            let packed = |sources: &[(&str, &Placed)]| normalize(get, sources, m).unwrap();
            let ours = packed(&[("c.weight", &c)]);
            assert_eq!(ours, packed(&[(&pw, &p)]), "{stem}");
            assert_eq!(ours.dims, [2 * half as u64, rows as u64]);
            let stack = packed(&[("c.weight", &c), ("c.weight", &c)]);
            assert_eq!(stack, packed(&[(&pw, &p), (&pw, &p)]), "{stem}");
            let alone = Placed {
                format,
                companions: vec!["c.weight_scale_2".into()],
            };
            let e = normalize(get, &[("c.weight", &alone)], m).unwrap_err();
            assert!(
                e.to_string()
                    .contains("not a compressed block scale and a tensor scale"),
                "{e}"
            );
        }

        let codes: Vec<u8> = (0..2 * 16).map(|i| i as u8).collect();
        let mut scales = vec![0x38u8, 0x30, 0x40, 0x31];
        let file = |scales: &[u8]| {
            write(&[
                ("t.weight", "U8", &[2, 16][..], &codes[..]),
                ("t.weight_scale", "F8_E4M3", &[2, 2], scales),
                ("s.weight_scale_2", "F32", &[1], &0.5f32.to_le_bytes()),
            ])
        };
        let st = Safetensors::parse(file(&scales)).unwrap();
        let rows_of = |st: &Safetensors| {
            let p = Placed {
                format: Format::Nvfp4,
                companions: vec!["t.weight_scale".into(), "s.weight_scale_2".into()],
            };
            normalize(|n| st.raw(n), &[("t.weight", &p)], Use::Rows)
        };
        let n = rows_of(&st).unwrap();
        let mut want = Vec::new();
        for (row, group) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            want.push(scales[2 * row + group]);
            want.extend_from_slice(&codes[16 * row + 8 * group..][..8]);
        }
        assert_eq!(
            (n.type_id, &n.dims, &n.bytes, &n.scale),
            (
                GgmlType::NVFP4_ROWS.as_u32(),
                &vec![32, 2],
                &want,
                &vec![0.5]
            )
        );
        scales[3] = 0x80;
        let st = Safetensors::parse(file(&scales)).unwrap();
        let e = rows_of(&st).unwrap_err();
        assert!(
            matches!(&e, Error::Scale { name, row: 1, group: 1, .. } if name == "t.weight_scale"),
            "{e:?}"
        );
        let inf = Safetensors::parse(write(&[
            ("t.weight", "U8", &[2, 16][..], &codes[..]),
            ("t.weight_scale", "F8_E4M3", &[2, 2], &scales),
            (
                "s.weight_scale_2",
                "F32",
                &[1],
                &f32::INFINITY.to_le_bytes(),
            ),
        ]))
        .unwrap();
        let e = rows_of(&inf).unwrap_err().to_string();
        assert!(e.contains("tensor scale inf is not finite"), "{e}");
        for (code, group_scale, divisor) in [(0x77, 0x38, 8.0), (0x22, 0x40, 2.0)] {
            for tensor_scale in [f32::MAX, -f32::MAX] {
                let mut codes = [0x11; 160];
                let mut scales = [0x38; 20];
                codes[152..].fill(code);
                scales[19] = group_scale;
                let file = |scale: f32| {
                    Safetensors::parse(write(&[
                        ("t.weight", "U8", &[2, 80][..], &codes[..]),
                        ("t.weight_scale", "F8_E4M3", &[2, 10], &scales),
                        ("s.weight_scale_2", "F32", &[1], &scale.to_le_bytes()),
                    ]))
                    .unwrap()
                };
                rows_of(&file(tensor_scale / divisor)).unwrap();
                let e = rows_of(&file(tensor_scale)).unwrap_err().to_string();
                assert!(
                    e.contains("t.weight_scale: row 1 group 9") && e.contains("leaves f32's range"),
                    "{e}"
                );
            }
        }
        let empty = Safetensors::parse(write(&[
            ("t.weight", "U8", &[2, 0][..], &[][..]),
            ("t.weight_scale", "F8_E4M3", &[2, 0], &[]),
            ("s.weight_scale_2", "F32", &[1], &0.5f32.to_le_bytes()),
        ]))
        .unwrap();
        let e = rows_of(&empty).unwrap_err().to_string();
        assert!(
            e.contains("U8 [2, 0], not U8 rows of whole 16-value groups"),
            "{e}"
        );
        for (dims, weight_bytes, scale_bytes, want) in [
            ([1, usize::MAX - 7], &[][..], &[][..], "row size overflow"),
            (
                [4, usize::MAX / 2 / 8 * 8],
                &[],
                &[],
                "weight size overflow",
            ),
            (
                [16, usize::MAX / 16 / 8 * 8],
                &[],
                &[],
                "row bytes size overflow",
            ),
            ([0, 16], &[], &[], "not U8 rows"),
            ([2, 16], &codes[..31], &scales[..], "31 bytes, not 32"),
            ([2, 16], &codes[..], &scales[..3], "the groups are F8_E4M3"),
        ] {
            let p = Placed {
                format: Format::Nvfp4,
                companions: vec!["t.weight_scale".into(), "s.weight_scale_2".into()],
            };
            let get = |n: &str| {
                let mut raw = st.raw(n)?;
                if n == "t.weight" {
                    raw.shape = &dims;
                    raw.bytes = weight_bytes;
                } else if n == "t.weight_scale" {
                    raw.bytes = scale_bytes;
                }
                Ok(raw)
            };
            let e = normalize(get, &[("t.weight", &p)], Use::Rows)
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{dims:?}: {e}");
        }
        let one = Placed {
            format: Format::Nvfp4,
            companions: vec!["t.weight_scale".into()],
        };
        let e = normalize(|n| st.raw(n), &[("t.weight", &one)], Use::Rows).unwrap_err();
        assert!(
            e.to_string().contains(
                r#"its place names ["t.weight_scale"], not a block scale and a tensor scale"#
            ),
            "{e}"
        );
    }

    #[test]
    fn a_quantized_single_tensor_requires_a_matrix_before_reading() {
        for format in [
            Format::Nvfp4,
            Format::Nvfp4Csf,
            Format::Nvfp4Csf32,
            Format::Nvfp4Ct,
            Format::Mxfp8,
            Format::Fp8Block {
                rows: 128,
                cols: 128,
            },
            Format::Fp8Channel,
            Format::Fp8Tensor,
        ] {
            let p = Placed {
                format,
                companions: vec!["w.scale".into()],
            };
            let values_per_element = format.logical(&[1])[0];
            let bytes = vec![0; 64 * 2 * 2 / values_per_element];
            let shape = [2, 2, 64 / values_per_element];
            let e = check_part(
                |_| {
                    Ok(Raw {
                        dtype: Dtype::U8,
                        shape: &shape,
                        bytes: &bytes,
                    })
                },
                &[("w.weight", &p)],
                Use::Matrix(Plain::F32),
                &[64, 2, 2],
                (1, 0..1),
            )
            .unwrap_err()
            .to_string();
            assert!(
                e.contains("w.weight") && e.contains("not a 2-D quantized weight"),
                "{format}: {e}"
            );
        }
    }

    /// A row table is one source; a second source, found or missing, is refused, not dropped.
    #[test]
    fn a_row_table_of_two_sources_is_refused() {
        let st = fixture();
        let owned = STEMS.map(own);
        let (first, second) = (own(STEMS[0]), own("missing.stem"));
        for both in [
            sources(&owned),
            vec![(first.0.as_str(), &first.1), (second.0.as_str(), &second.1)],
        ] {
            let e = normalize(|n| st.raw(n), &both, Use::Rows).unwrap_err();
            assert!(e.to_string().contains("exactly one source"), "{e}");
        }
    }

    /// The existing GGUF path: a routed stack of the fixture's two real experts, normalized by the
    /// table and written as a one-shard split with its `.scale`, opens through `Split::open` and
    /// decodes by `dequant_row` times each expert's scale to the reference rows of
    /// `modelopt_nvfp4.py ref`, the ModelOpt rule.
    #[test]
    fn a_normalized_stack_reads_back_through_a_split_as_the_modelopt_rule() {
        use gguf::write::{Layout, TensorDecl, Writer};
        let st = fixture();
        let owned = STEMS.map(own);
        let n = normalize(|x| st.raw(x), &sources(&owned), Use::Matrix(Plain::Q8_0)).unwrap();
        let scale: Vec<u8> = n.scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let decls = vec![
            TensorDecl {
                name: "stack.weight".into(),
                dims: n.dims.clone(),
                type_id: n.type_id,
                nbytes: n.bytes.len() as u64,
            },
            TensorDecl {
                name: "stack.scale".into(),
                dims: vec![2],
                type_id: 0,
                nbytes: 8,
            },
        ];
        let kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                gguf::Value::String("qwen4exp".into()),
            ),
            ("split.no".to_string(), gguf::Value::U16(0)),
            ("split.count".to_string(), gguf::Value::U16(1)),
            ("split.tensors.count".to_string(), gguf::Value::I32(2)),
        ];
        let dir = std::env::temp_dir().join(format!("hf-modelopt-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stack-00001-of-00001.gguf");
        let mut w = Writer::new(
            std::fs::File::create(&path).unwrap(),
            Layout::new(&kvs, decls).unwrap(),
        )
        .unwrap();
        w.tensor("stack.weight", &n.bytes).unwrap();
        w.tensor("stack.scale", &scale).unwrap();
        w.finish().unwrap();
        let split = gguf::Split::open(&path).unwrap();
        let read = |name: &str| {
            let (i, t) = split.find(name).unwrap();
            split.shard(i).unwrap().data(t).unwrap().to_vec()
        };
        let blocks = read("stack.weight");
        let scales: Vec<f32> = read("stack.scale")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let per = blocks.len() / 2;
        for (e, stem) in STEMS.iter().enumerate() {
            let (rows, scale) = reference(stem);
            assert_eq!(scales[e], scale as f32, "{stem}");
            assert_eq!(
                decoded(&blocks[e * per..][..per], 2560, scales[e]),
                rows,
                "{stem}"
            );
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    /// A one-row triple of `k` values: the scale bytes, codes 1 (0.5) throughout, a tensor scale
    /// of 2 and, when `input` is given, an `input_scale` of that value.
    fn triple(k: usize, scales: &[u8], input: Option<f32>) -> Vec<u8> {
        let codes = vec![0x11u8; k / 2];
        let two = 2.0f32.to_le_bytes();
        let input = input.map(f32::to_le_bytes);
        let (wshape, sshape) = ([1, k / 2], [1, k / 16]);
        let mut list: Vec<(&str, &str, &[usize], &[u8])> = vec![
            ("t.weight", "U8", &wshape, &codes),
            ("t.weight_scale", "F8_E4M3", &sshape, scales),
            ("t.weight_scale_2", "F32", &[][..], &two),
        ];
        if let Some(b) = &input {
            list.push(("t.input_scale", "F32", &[][..], b));
        }
        write(&list)
    }

    /// `input_scale` is never read: two values, or none, give the same blocks and scale.
    #[test]
    fn input_scale_is_not_read() {
        let get = |file: Vec<u8>| {
            let st = Safetensors::parse(file).unwrap();
            lin(|n| st.raw(n), "t").unwrap()
        };
        let a = get(triple(64, &[0x38; 4], Some(0.25)));
        let b = get(triple(64, &[0x38; 4], Some(1e30)));
        let c = get(triple(64, &[0x38; 4], None));
        assert_eq!((&a.blocks, &a.scale), (&b.blocks, &b.scale));
        assert_eq!((&a.blocks, &a.scale), (&c.blocks, &c.scale));
    }

    /// A scale byte with bit 7 set, or `0x7F`, is refused naming the tensor, the row and the
    /// group, with the reason the block's decode gives; mainline's conversion strips bit 7.
    #[test]
    fn a_scale_byte_that_is_no_ue4m3_scale_is_refused_by_name() {
        for byte in [0x7Fu8, 0x80, 0xB8, 0xFF] {
            let mut scales = [0x38u8; 8];
            scales[6] = byte;
            let st = Safetensors::parse(triple(128, &scales, None)).unwrap();
            let e = lin(|n| st.raw(n), "t").unwrap_err();
            let why = ue4m3_to_f32_half(byte).unwrap_err();
            assert!(
                matches!(&e, Error::Scale { name, row: 0, group: 6, source }
                    if name == "t.weight_scale" && *source == why),
                "{e:?}"
            );
            assert!(e.to_string().contains("row 0 group 6"), "{e}");
        }
    }

    /// Each malformed triple and stack is refused by name.
    #[test]
    fn malformed_triples_and_stacks_are_refused_by_name() {
        let one = 1.0f32.to_le_bytes();
        let nan = f32::NAN.to_le_bytes();
        let inf = f32::INFINITY.to_le_bytes();
        let neg_inf = f32::NEG_INFINITY.to_le_bytes();
        let two = [0u8; 8];
        let codes = [0u8; 48];
        let s3 = [0x38u8; 6];
        let s4 = [0x38u8; 4];
        let shape = |list: &[(&str, &str, &[usize], &[u8])], name: &str| {
            let st = Safetensors::parse(write(list)).unwrap();
            let e = lin(|n| st.raw(n), "t").unwrap_err();
            assert!(
                matches!(&e, Error::Shape { name: n, .. } if n == name),
                "{e:?}"
            );
        };
        // k = 96 is not a multiple of 64.
        shape(
            &[
                ("t.weight", "U8", &[1, 48], &codes),
                ("t.weight_scale", "F8_E4M3", &[1, 6], &s3),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight",
        );
        // The scale's groups do not match the weight's.
        shape(
            &[
                ("t.weight", "U8", &[1, 32], &codes[..32]),
                ("t.weight_scale", "F8_E4M3", &[2, 2], &s4),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight_scale",
        );
        // Two tensor scales, and one that is NaN.
        for (shape2, bytes) in [
            (&[2][..], &two[..]),
            (&[][..], &nan[..]),
            (&[], &inf),
            (&[], &neg_inf),
        ] {
            shape(
                &[
                    ("t.weight", "U8", &[1, 32], &codes[..32]),
                    ("t.weight_scale", "F8_E4M3", &[1, 4], &s4),
                    ("t.weight_scale_2", "F32", shape2, bytes),
                ],
                "t.weight_scale_2",
            );
        }
        // A width whose k = 2 · (k/2) overflows, zero rows so the file holds no bytes.
        shape(
            &[
                ("t.weight", "U8", &[0, usize::MAX / 2 + 33], &[]),
                ("t.weight_scale", "F8_E4M3", &[0, 4], &[]),
                ("t.weight_scale_2", "F32", &[], &one),
            ],
            "t.weight",
        );
        // A lookup's tensors are checked as the file's are: bytes short of or beyond the
        // shape, and the overflowing width.
        let lookup = |list: &[(&str, Dtype, &[usize], &[u8])], name: &str| {
            let e = lin(
                |n| {
                    list.iter()
                        .find(|t| t.0 == n)
                        .map(|&(_, dtype, shape, bytes)| Raw {
                            dtype,
                            shape,
                            bytes,
                        })
                        .ok_or_else(|| safetensors::Error::Missing(n.into()))
                },
                "t",
            )
            .unwrap_err();
            assert!(
                matches!(&e, Error::Shape { name: n, .. } if n == name),
                "{e:?}"
            );
        };
        let s8 = [0x38u8; 8];
        for (wshape, w, sshape, s, name) in [
            (&[2, 32][..], &codes[..32], &[2, 4][..], &s8[..], "t.weight"),
            (&[1, 32], &codes[..], &[1, 4], &s4, "t.weight"),
            (&[1, 32], &codes[..32], &[1, 4], &s4[..3], "t.weight_scale"),
            (&[1, 32], &codes[..32], &[1, 4], &s8, "t.weight_scale"),
            (&[0, usize::MAX / 2 + 33], &[], &[0, 4], &[], "t.weight"),
        ] {
            lookup(
                &[
                    ("t.weight", Dtype::U8, wshape, w),
                    ("t.weight_scale", Dtype::F8E4M3, sshape, s),
                    ("t.weight_scale_2", Dtype::F32, &[], &one),
                ],
                name,
            );
        }
        // A weight of another dtype, and a missing scale.
        let st =
            Safetensors::parse(write(&[("t.weight", "BF16", &[1, 16], &codes[..32])])).unwrap();
        let e = lin(|n| st.raw(n), "t").unwrap_err();
        assert!(
            matches!(
                &e,
                Error::Safetensors(safetensors::Error::Kind {
                    have: "BF16",
                    want: "U8",
                    ..
                })
            ),
            "{e:?}"
        );
        let st = Safetensors::parse(write(&[("t.weight", "U8", &[1, 32], &codes[..32])])).unwrap();
        let e = lin(|n| st.raw(n), "t").unwrap_err();
        assert!(
            matches!(&e, Error::Safetensors(safetensors::Error::Missing(n)) if n == "t.weight_scale"),
            "{e:?}"
        );
        // A stack of no expert, one given more experts than it holds, and one whose second
        // expert has other rows.
        let st = fixture();
        let none: [&str; 0] = [];
        assert!(matches!(
            stack(|n| st.raw(n), &none),
            Err(Error::Shape { what, .. }) if what == "no expert"
        ));
        let owned: Vec<_> = STEMS.iter().map(|s| own(s)).collect();
        let two = sources(&owned);
        let mut full = Nvfp4Stack::new(1);
        full.push(|n| st.raw(n), two[0]).unwrap();
        let e = full.push(|n| st.raw(n), two[1]).unwrap_err();
        assert!(
            matches!(&e, Error::Shape { name, what }
                if name == two[1].0 && what == "more experts than the stack holds"),
            "{e:?}"
        );
        let e = stack(|n| st.raw(n), &[STEMS[0], "synthetic"]).unwrap_err();
        assert!(
            matches!(&e, Error::Shape { name, .. } if name == "synthetic.weight"),
            "{e:?}"
        );
    }

    // MLX's affine tensors pinned from real mx.quantize output: pip mlx 0.32.1,
    // mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=64, bits=4|8, mode="affine") over
    // an LCG weight (x = (1103515245·x + 12345) mod 2^31, value ((x >> 7) mod 2000 − 1000) / 347)
    // of shape [2, 2, 128] at 4 bits (seed 1, the group [1, 1, 64..128] set to 0.5) and of shape
    // [2, 128] at 8 bits (seed 7). The words are the returned `.weight` (LE U32), the scales and
    // biases its `.scales` and `.biases` BF16 bits; each digest is the sha256 of one row of
    // mx.dequantize's BF16 bits, little-endian.
    const A4_WORDS: [u32; 64] = [
        0x946c0ec6, 0xc56763c9, 0x4289ecff, 0x9d92efff, 0x45ccc1b1, 0x9bf463b2, 0x9a65574f,
        0x2a441f4c, 0xa491e959, 0x996d236f, 0xa7bc5dc0, 0x9bed53c4, 0x53fd6ddb, 0xf89732d0,
        0xdf0b3742, 0xc67148af, 0x38073bc4, 0x0adb4b8c, 0x6844a6af, 0x40e5dbc8, 0xfd3e3e64,
        0x8f218d16, 0xf595c059, 0x804744e1, 0x0a8df6cf, 0x19f362c0, 0xa11e74e2, 0xf07a38f2,
        0xb73b34f3, 0xd1ffd209, 0x32e8accd, 0xefb0fdf3, 0xcb0e57d5, 0x6f88f6e5, 0x774e686d,
        0xa6f89634, 0x79277ba2, 0x54034690, 0x2625154a, 0x219bc354, 0x5b8afd59, 0xd065e714,
        0x3d6a6410, 0x8f97f31a, 0x8afc2c87, 0x6b70099f, 0xa7db1037, 0x7b74ba92, 0xfe9bb188,
        0x58f205ec, 0x832685f6, 0x11027aef, 0xfdf92c66, 0x798e387c, 0xf5d69283, 0xee82339c,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ];
    const A4_SCALES: [u16; 8] = [
        0xbeb7, 0x3eb8, 0x3eb7, 0xbeb8, 0x3eb7, 0xbeb7, 0x3eb7, 0xb3d7,
    ];
    const A4_BIASES: [u16; 8] = [
        0x4037, 0xc038, 0xc037, 0x4038, 0xc037, 0x4037, 0xc037, 0x3f00,
    ];
    const A4_MX_DIGEST: [&str; 4] = [
        "0417c7b34c970c2ba060f7011aa1a7e250bd0ede7f2f5960beabd00f8302b414",
        "6249d447b1f4296685ff5575ac796a2a9f840bb117b6f3d339df8baacf260840",
        "d5e06430d99f199d7d20f8bf1096d64b6f50c40686c80b8b963f1dbec393d717",
        "2b09087daaac4e01a3391020f72abc73b9b2af3eb4c88b61f97e797590d2739e",
    ];
    const A8_WORDS: [u32; 64] = [
        0x4cb99788, 0x871c6df1, 0x076e3fd5, 0x50142382, 0x2e745e87, 0x8ab36463, 0xf129232b,
        0x01e1a280, 0xb2c45d77, 0x55ba5e7f, 0xff01bdb1, 0x4ee02145, 0xb4c8e1b1, 0x07eea016,
        0xfc603841, 0x0400214f, 0x9dbd8db9, 0x114ef704, 0x5f4766ce, 0x468f81a8, 0x316bffb2,
        0x00c5041a, 0x5f4f2dfc, 0xaf23be1d, 0xd7d88d44, 0x0599ece2, 0x42417f72, 0x9acc1e63,
        0x3dd3f16e, 0x8406f518, 0x2afdba6f, 0x84e6b9dd, 0x05e2cd3f, 0x2352be55, 0x60b9a94c,
        0xeed9618a, 0xff8eb398, 0x98c73036, 0xb78ac6a8, 0x3258f0d2, 0x40d80114, 0xb800ed2f,
        0x6682fd22, 0xd0dcb7c2, 0x9d01ff2f, 0x25c41676, 0x7b27e9e7, 0x8d699a39, 0x361b9008,
        0xd218ffb3, 0x94b5b8db, 0x74ef2331, 0x559a2532, 0x7480000a, 0xa1976227, 0x18b03a6c,
        0xe6577cb2, 0xbdd034fd, 0x3e6d0852, 0xe583c776, 0xa6a98392, 0xbb57d6ed, 0xb375cf10,
        0x1dcdd559,
    ];
    const A8_SCALES: [u16; 4] = [0xbcb0, 0x3cb3, 0x3cb7, 0xbcb8];
    const A8_BIASES: [u16; 4] = [0x4030, 0xc034, 0xc038, 0x4038];
    const A8_MX_DIGEST: [&str; 2] = [
        "7258079ebf93ca8ca99fb88b030d5036502e6f69d0bf04fecbb0413b6124b4c4",
        "dc7d770fbd8223c174e07e764ae391204e6faa7901980d3efa7ff2459c0f8461",
    ];

    /// The tensor `m` of an MLX checkpoint: its weight of `shape` (U32 words), its scales and its
    /// biases over the weight's rows and groups, and its place in the plan.
    fn mlx(
        bits: usize,
        shape: &[usize],
        planes: (&[u32], &[u16], &[u16]),
    ) -> (Safetensors, Placed) {
        mlx_stored("BF16", bits, shape, planes)
    }

    /// [`mlx`] with the scales and biases stored as `dtype` ("BF16" or "F16") and placed in the
    /// format the plan gives that dtype.
    fn mlx_stored(
        dtype: &str,
        bits: usize,
        shape: &[usize],
        (words, scales, biases): (&[u32], &[u16], &[u16]),
    ) -> (Safetensors, Placed) {
        let (&packed, rows) = shape.split_last().unwrap();
        let mut grid = rows.to_vec();
        grid.push(packed * 32 / bits / 64);
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let words: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let st = Safetensors::parse(write(&[
            ("m.weight", "U32", shape, &words),
            ("m.scales", dtype, &grid, &le(scales)),
            ("m.biases", dtype, &grid, &le(biases)),
        ]))
        .unwrap();
        let place = Placed {
            format: match dtype {
                "F16" => Format::MlxAffineF16 { bits, group: 64 },
                _ => Format::MlxAffine { bits, group: 64 },
            },
            companions: vec!["m.scales".into(), "m.biases".into()],
        };
        (st, place)
    }

    /// The values of the pinned groups by MLX's rule, read from the pinned words and pair and not
    /// from any block: code `i` of a group is the `bits` bits at `bits · (i mod per)` of word
    /// `i / per`, `per` codes a word, and the value is `s · code + b` in f32.
    fn mlx_rule(bits: usize, words: &[u32], scales: &[u16], biases: &[u16]) -> Vec<f32> {
        mlx_rule_with(bf16_bits_to_f32, bits, words, scales, biases)
    }

    /// [`mlx_rule`] with the scale and bias widened by `widen`.
    fn mlx_rule_with(
        widen: fn(u16) -> f32,
        bits: usize,
        words: &[u32],
        scales: &[u16],
        biases: &[u16],
    ) -> Vec<f32> {
        let per = 32 / bits;
        words
            .chunks(64 / per)
            .zip(scales.iter().zip(biases))
            .flat_map(|(group, (&s, &b))| {
                let (s, b) = (widen(s), widen(b));
                (0..64).map(move |i| {
                    let code = (group[i / per] >> (bits * (i % per))) & ((1 << bits) - 1);
                    s * code as f32 + b
                })
            })
            .collect()
    }

    /// An MLX tensor repacked into GGUF's block decodes, row by row, to the values of MLX's own
    /// rule bit for bit and, rounded once to BF16, to what `mx.dequantize` returned; a stack is
    /// one tensor `[k, rows, n_expert]` whose first expert is the matrix of that expert alone.
    /// A group's scale and bias swapped, or a word moved, changes a value.
    ///
    /// FAIL-first: writing each group's bias before its scale in `mlx_affine` is red here.
    #[test]
    fn mlx_tensors_repack_to_the_mx_quantize_rule() {
        let role = Use::Matrix(Plain::Q8_0);
        let a4 = (&A4_WORDS[..], &A4_SCALES[..], &A4_BIASES[..]);
        let a8 = (&A8_WORDS[..], &A8_SCALES[..], &A8_BIASES[..]);
        let cases: [(usize, &[usize], GgmlType, _, &[&str]); 2] = [
            (4, &[2, 2, 16], GgmlType::MLX_A4, a4, &A4_MX_DIGEST),
            (8, &[2, 32], GgmlType::MLX_A8, a8, &A8_MX_DIGEST),
        ];
        for (bits, shape, ty, planes, digests) in cases {
            let (st, place) = mlx(bits, shape, planes);
            let n = normalize(|n| st.raw(n), &[("m.weight", &place)], role).unwrap();
            let block = ty.type_size().unwrap() as usize;
            let mut dims = vec![128u64];
            dims.extend(shape[..shape.len() - 1].iter().rev().map(|&d| d as u64));
            assert_eq!(n.type_id, ty.as_u32(), "{ty}");
            assert_eq!(n.dims, dims, "{ty}");
            assert!(n.scale.is_empty(), "{ty}");
            assert_eq!(n.bytes.len(), digests.len() * 2 * block, "{ty}");
            let want = mlx_rule(bits, planes.0, planes.1, planes.2);
            for (row, digest) in digests.iter().enumerate() {
                let mut v = vec![0.0f32; 128];
                dequant_row(ty, &n.bytes[row * 2 * block..][..2 * block], &mut v).unwrap();
                for (i, (got, want)) in v.iter().zip(&want[row * 128..]).enumerate() {
                    assert_eq!(got.to_bits(), want.to_bits(), "{ty} row {row} value {i}");
                }
                let bf16: Vec<u8> = v
                    .iter()
                    .flat_map(|&x| fp32_to_bf16_bits(x).to_le_bytes())
                    .collect();
                assert_eq!(
                    format!("{:x}", Sha256::digest(&bf16)),
                    *digest,
                    "{ty} row {row}: mx.dequantize's BF16"
                );
            }
        }
        let (st, place) = mlx(4, &[2, 2, 16], a4);
        let stack = normalize(|n| st.raw(n), &[("m.weight", &place)], role).unwrap();
        let first = (&A4_WORDS[..32], &A4_SCALES[..4], &A4_BIASES[..4]);
        let (st, place) = mlx(4, &[2, 16], first);
        let alone = normalize(|n| st.raw(n), &[("m.weight", &place)], role).unwrap();
        assert_eq!(alone.dims, [128, 2]);
        assert_eq!(alone.bytes, stack.bytes[..alone.bytes.len()]);
    }

    /// An MLX tensor whose scales and biases are F16 repacks into the F16-pair block types
    /// (MLX_A4H, MLX_A8H): each group's words as stored, then its F16 scale bits and its F16 bias
    /// bits as stored, none rounded; the blocks decode to `s·q + b` with `s` and `b` widened from
    /// F16 bits (the pinned words under hand-picked pairs: 0.25, 1, 2, 0.5 over -0.5, 0, -1, 0.25).
    /// F16 companions of a weight placed as BF16 ones, and the other way round, are refused by
    /// name, as is a group whose F16 scale is NaN.
    ///
    /// FAIL-first: placing an F16 weight as MLX_A4 and MLX_A8 (`Format::card_type`), or reading
    /// the pair as BF16 bits in this rule, is red here.
    #[test]
    fn mlx_f16_tensors_repack_to_the_f16_block_types() {
        let role = Use::Matrix(Plain::Q8_0);
        let scales = [
            0x3400u16, 0x3c00, 0x4000, 0x3800, 0x3400, 0x3c00, 0x4000, 0x3800,
        ];
        let biases = [
            0xb800u16, 0x0000, 0xbc00, 0x3400, 0xb800, 0x0000, 0xbc00, 0x3400,
        ];
        let a4 = (&A4_WORDS[..], &scales[..], &biases[..]);
        let a8 = (&A8_WORDS[..], &scales[..4], &biases[..4]);
        let cases: [(usize, &[usize], GgmlType, _); 2] = [
            (4, &[2, 2, 16], GgmlType::MLX_A4H, a4),
            (8, &[2, 32], GgmlType::MLX_A8H, a8),
        ];
        for (bits, shape, ty, planes) in cases {
            let (st, place) = mlx_stored("F16", bits, shape, planes);
            let n = normalize(|n| st.raw(n), &[("m.weight", &place)], role).unwrap();
            let block = ty.type_size().unwrap() as usize;
            assert_eq!(n.type_id, ty.as_u32(), "{ty}");
            assert_eq!(n.bytes.len(), planes.1.len() * block, "{ty}");
            for (g, blk) in n.bytes.chunks_exact(block).enumerate() {
                assert_eq!(
                    blk[block - 4..block - 2],
                    planes.1[g].to_le_bytes(),
                    "{ty} {g}"
                );
                assert_eq!(blk[block - 2..], planes.2[g].to_le_bytes(), "{ty} {g}");
            }
            let want = mlx_rule_with(half_to_f32, bits, planes.0, planes.1, planes.2);
            let mut got = vec![0.0f32; want.len()];
            dequant_row(ty, &n.bytes, &mut got).unwrap();
            for (i, (got, want)) in got.iter().zip(&want).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "{ty} value {i}");
            }
            // Stored as BF16 under an F16 place, and the other way round.
            let (st, _) = mlx_stored("BF16", bits, shape, planes);
            let e = normalize(|n| st.raw(n), &[("m.weight", &place)], role)
                .unwrap_err()
                .to_string();
            assert!(e.contains("m.scales") && e.contains("BF16"), "{e}");
            let (st, place) = mlx_stored("F16", bits, shape, planes);
            let bf16_place = Placed {
                format: Format::MlxAffine { bits, group: 64 },
                ..place
            };
            let e = normalize(|n| st.raw(n), &[("m.weight", &bf16_place)], role)
                .unwrap_err()
                .to_string();
            assert!(e.contains("m.scales") && e.contains("F16"), "{e}");
        }
        // A NaN F16 scale in row 0's first group, refused naming the row and the block.
        let mut nan = scales;
        nan[0] = 0x7e00;
        let (st, place) = mlx_stored("F16", 4, &[2, 2, 16], (&A4_WORDS, &nan, &biases));
        let e = normalize(|n| st.raw(n), &[("m.weight", &place)], role)
            .unwrap_err()
            .to_string();
        for want in [
            "m.weight: MLX-affine F16 4-bit group 64",
            "row 0",
            "mlx_a4h: block 0 holds scale bits 0x7e00, not finite",
        ] {
            assert!(e.contains(want), "{e}: missing {want}");
        }
    }

    /// An F16 tensor (a norm gain) is read widened to f32, every value exact (1, -2, the smallest
    /// subnormal 2^-24, F16's largest 65504, zeros of both signs), and no other read has a type
    /// for it: Q8_0, BF16 and a row table are refused by name. A non-finite value is refused by
    /// name too.
    #[test]
    fn f16_tensors_are_read_widened_to_f32_or_refused() {
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let bits = [0x3c00u16, 0xc000, 0x0001, 0x7bff, 0x0000, 0x8000];
        let st = Safetensors::parse(write(&[("g.weight", "F16", &[2, 3], &le(&bits))])).unwrap();
        let place = Placed {
            format: Format::F16,
            companions: vec![],
        };
        let n = normalize(
            |n| st.raw(n),
            &[("g.weight", &place)],
            Use::Matrix(Plain::F32),
        )
        .unwrap();
        assert_eq!(n.type_id, GgmlType::F32.as_u32());
        assert_eq!(n.dims, [3, 2]);
        let got: Vec<f32> = n
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let want = [1.0f32, -2.0, 2.0f32.powi(-24), 65504.0, 0.0, -0.0];
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "value {i}");
        }
        for role in [
            Use::Matrix(Plain::Q8_0),
            Use::Matrix(Plain::Bf16),
            Use::Q8_0,
            Use::Rows,
        ] {
            assert_eq!(type_of(Format::F16, role), None, "{role:?}");
            let e = normalize(|n| st.raw(n), &[("g.weight", &place)], role)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("g.weight: F16") && e.contains("has no GGUF type here"),
                "{e}"
            );
        }
        let st = Safetensors::parse(write(&[("g.weight", "F16", &[2], &le(&[0x3c00, 0x7c00]))]))
            .unwrap();
        let e = normalize(
            |n| st.raw(n),
            &[("g.weight", &place)],
            Use::Matrix(Plain::F32),
        )
        .unwrap_err();
        assert!(
            matches!(&e, Error::Shape { name, what }
                if name == "g.weight" && what.contains("non-finite value at 1")),
            "{e:?}"
        );
    }

    /// The table writes an MLX affine tensor as its block type for a matrix role and gives a row
    /// table, a width and a group the blocks do not have no type; the F16-pair format has its own
    /// block types.
    #[test]
    fn mlx_affine_is_written_as_its_block_type() {
        let mlx = |bits, group| Format::MlxAffine { bits, group };
        let half = |bits, group| Format::MlxAffineF16 { bits, group };
        for role in [Plain::Q8_0, Plain::F32, Plain::Bf16] {
            let role = Use::Matrix(role);
            assert_eq!(type_of(mlx(4, 64), role), Some(GgmlType::MLX_A4.as_u32()));
            assert_eq!(type_of(mlx(8, 64), role), Some(GgmlType::MLX_A8.as_u32()));
            assert_eq!(type_of(mlx(3, 64), role), None);
            assert_eq!(type_of(mlx(4, 32), role), None);
            assert_eq!(type_of(half(4, 64), role), Some(GgmlType::MLX_A4H.as_u32()));
            assert_eq!(type_of(half(8, 64), role), Some(GgmlType::MLX_A8H.as_u32()));
            assert_eq!(type_of(half(3, 64), role), None);
            assert_eq!(type_of(half(4, 32), role), None);
        }
        assert_eq!(type_of(mlx(4, 64), Use::Rows), None);
        assert_eq!(type_of(half(4, 64), Use::Rows), None);
    }

    /// An MLX tensor the format does not hold is refused by name: a row that is not whole groups, a
    /// scales or biases tensor of another group count, row count or dtype, a weight of another
    /// dtype or of no row, a missing companion, a group whose scale or bias is not finite or
    /// whose top code leaves f32's range (naming its row and block), a place without its two
    /// companions, a stack of two sources, and a role or a width that has no type.
    #[test]
    fn mlx_tensors_the_format_does_not_hold_are_refused_by_name() {
        let role = Use::Matrix(Plain::Q8_0);
        let place = |bits, companions: &[&str]| Placed {
            format: Format::MlxAffine { bits, group: 64 },
            companions: companions.iter().map(|c| c.to_string()).collect(),
        };
        let both = ["m.scales", "m.biases"];
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        // Two rows of two 4-bit groups: 16 words a row, a pair a group.
        let words = [0x1234_5678u32; 32].map(u32::to_le_bytes).concat();
        let one = le(&[0x3f80; 4]);
        let zero = le(&[0; 4]);
        let run = |tensors: &[(&str, &str, &[usize], &[u8])], place: &Placed, role| {
            let st = Safetensors::parse(write(tensors)).unwrap();
            normalize(|n| st.raw(n), &[("m.weight", place)], role)
                .unwrap_err()
                .to_string()
        };
        let good = place(4, &both);
        let refuse = |tensors: &[(&str, &str, &[usize], &[u8])], want: &[&str]| {
            let e = run(tensors, &good, role);
            for w in want {
                assert!(e.contains(w), "{e}: missing {w}");
            }
        };
        let w = ("m.weight", "U32", &[2, 16][..], &words[..]);
        // 12 words a row are 96 values.
        refuse(
            &[
                ("m.weight", "U32", &[2, 12], &words[..96]),
                ("m.scales", "BF16", &[2, 1], &one[..4]),
                ("m.biases", "BF16", &[2, 1], &zero[..4]),
            ],
            &[
                "m.weight: MLX-affine 4-bit group 64",
                "12 words a row are not whole groups of 64 values",
            ],
        );
        refuse(
            &[
                w,
                ("m.scales", "BF16", &[2, 3], &[0; 12]),
                ("m.biases", "BF16", &[2, 2], &zero),
            ],
            &["m.scales: MLX-affine", "shape [2, 3]", "groups are [2, 2]"],
        );
        refuse(
            &[
                w,
                ("m.scales", "BF16", &[2, 2], &one),
                ("m.biases", "BF16", &[3, 2], &[0; 12]),
            ],
            &["m.biases: MLX-affine", "shape [3, 2]", "groups are [2, 2]"],
        );
        refuse(
            &[
                w,
                ("m.scales", "F32", &[2, 2], &[0; 16]),
                ("m.biases", "BF16", &[2, 2], &zero),
            ],
            &["m.scales", "F32"],
        );
        refuse(
            &[
                ("m.weight", "U8", &[2, 64], &words[..128]),
                ("m.scales", "BF16", &[2, 2], &one),
                ("m.biases", "BF16", &[2, 2], &zero),
            ],
            &["m.weight", "U8"],
        );
        refuse(
            &[
                ("m.weight", "U32", &[16], &words[..64]),
                ("m.scales", "BF16", &[2], &one[..4]),
                ("m.biases", "BF16", &[2], &zero[..4]),
            ],
            &["m.weight", "is not rows of packed words"],
        );
        refuse(
            &[
                ("m.weight", "U32", &[0, 16], &[]),
                ("m.scales", "BF16", &[0, 2], &[]),
                ("m.biases", "BF16", &[0, 2], &[]),
            ],
            &["m.weight", "has a zero dimension"],
        );
        refuse(&[w, ("m.scales", "BF16", &[2, 2], &one)], &["m.biases"]);
        // Row 1's first group has a NaN scale; row 0's second an infinite bias; row 0's first a
        // scale whose top code's product overflows.
        let nan_scale = le(&[0x3f80, 0x3f80, 0x7fc0, 0x3f80]);
        refuse(
            &[
                w,
                ("m.scales", "BF16", &[2, 2], &nan_scale),
                ("m.biases", "BF16", &[2, 2], &zero),
            ],
            &[
                "m.weight: MLX-affine",
                "row 1",
                "block 0 holds scale bits 0x7fc0, not finite",
            ],
        );
        refuse(
            &[
                w,
                ("m.scales", "BF16", &[2, 2], &one),
                ("m.biases", "BF16", &[2, 2], &le(&[0, 0x7f80, 0, 0])),
            ],
            &["row 0", "block 1 holds bias bits 0x7f80, not finite"],
        );
        refuse(
            &[
                w,
                (
                    "m.scales",
                    "BF16",
                    &[2, 2],
                    &le(&[0x7f7f, 0x3f80, 0x3f80, 0x3f80]),
                ),
                ("m.biases", "BF16", &[2, 2], &zero),
            ],
            &["row 0", "a value times it leaves f32's range"],
        );
        let ok = [
            w,
            ("m.scales", "BF16", &[2, 2], &one[..]),
            ("m.biases", "BF16", &[2, 2], &zero[..]),
        ];
        let e = run(&ok, &place(4, &["m.scales"]), role);
        assert!(e.contains("not a scales and a biases tensor"), "{e}");
        let e = run(&ok, &good, Use::Rows);
        assert!(
            e.contains("m.weight") && e.contains("has no GGUF type here"),
            "{e}"
        );
        let e = run(&ok, &place(3, &both), role);
        assert!(
            e.contains("MLX-affine 3-bit group 64") && e.contains("has no GGUF type here"),
            "{e}"
        );
        let st = Safetensors::parse(write(&ok)).unwrap();
        let (p0, p1) = (good, place(4, &both));
        let e = normalize(|n| st.raw(n), &[("m.weight", &p0), ("m.weight", &p1)], role)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("in a stack of 2: an MLX stack is one tensor"),
            "{e}"
        );
    }

    /// An MLX weight is read whole through a part: the whole of any dim is the tensor, and a cut
    /// of a dim, which would leave its scales and biases uncut, and a scales tensor that does
    /// not fit the weight are refused by name from the headers.
    #[test]
    fn an_mlx_weight_is_read_whole_through_a_part() {
        let role = Use::Matrix(Plain::Q8_0);
        let (st, place) = mlx(4, &[2, 2, 16], (&A4_WORDS, &A4_SCALES, &A4_BIASES));
        let sources = [("m.weight", &place)];
        let dims = [128u64, 2, 2];
        let whole = normalize(|n| st.raw(n), &sources, role).unwrap();
        for (dim, &extent) in dims.iter().enumerate() {
            let at = (dim, 0..extent);
            check_part(|n| st.raw(n), &sources, role, &dims, at.clone()).unwrap();
            let part = normalize_part(|n| st.raw(n), &sources, role, &dims, at).unwrap();
            assert_eq!(part, whole, "dim {dim}");
        }
        for at in [(0, 0..64), (1, 1..2), (2, 0..1)] {
            let e = check_part(|n| st.raw(n), &sources, role, &dims, at.clone())
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("m.weight: MLX-affine 4-bit group 64")
                    && e.contains("an MLX affine weight is read whole"),
                "{at:?}: {e}"
            );
            assert!(normalize_part(|n| st.raw(n), &sources, role, &dims, at).is_err());
        }
        // A one-row gate the map reads as a vector: dims `[k]` over the stored `[1, words]`.
        let (gate, gate_place) = mlx(
            8,
            &[1, 32],
            (&A8_WORDS[..32], &A8_SCALES[..2], &A8_BIASES[..2]),
        );
        let alone = [("m.weight", &gate_place)];
        let at = (0, 0..128);
        check_part(|n| gate.raw(n), &alone, role, &[128], at.clone()).unwrap();
        assert_eq!(
            normalize_part(|n| gate.raw(n), &alone, role, &[128], at).unwrap(),
            normalize(|n| gate.raw(n), &alone, role).unwrap()
        );
        let st = Safetensors::parse(write(&[
            ("m.weight", "U32", &[2, 2, 16], &[0u8; 256]),
            ("m.scales", "BF16", &[2, 2, 3], &[0u8; 24]),
            ("m.biases", "BF16", &[2, 2, 2], &[0u8; 16]),
        ]))
        .unwrap();
        let e = check_part(|n| st.raw(n), &sources, role, &dims, (0, 0..128))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("m.scales") && e.contains("groups are [2, 2, 2]"),
            "{e}"
        );
    }

    /// One `weight_global_scale` an expert, in the range the real file holds.
    const GLOBALS: [f32; 4] = [8548.7, 16887.3, 4248.1, 18229.9];

    /// The codes and valid UE4M3 scale bytes of expert `e`, `rows` by `k` values.
    fn expert_bytes(e: usize, rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let codes = (0..rows * k / 2)
            .map(|i| (i * 131 % 251 + e * 71) as u8)
            .collect();
        let scales = (0..rows * k / 16)
            .map(|i| ((i * 29 + e * 13) % 126) as u8)
            .collect();
        (codes, scales)
    }

    /// A safetensors file of the tensors `list`.
    fn file(list: Vec<(String, &str, Vec<usize>, Vec<u8>)>) -> Safetensors {
        let refs: Vec<(&str, &str, &[usize], &[u8])> = list
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        Safetensors::parse(write(&refs)).unwrap()
    }

    /// The compressed-tensors tensors of the experts `c.<e>` of [`GLOBALS`] and the ModelOpt
    /// tensors `m.<e>` that store the same codes and block scales with `weight_scale_2` the f32
    /// reciprocal of the expert's global scale.
    fn twins(rows: usize, k: usize) -> (Safetensors, Safetensors) {
        let (mut ct, mut mo) = (vec![], vec![]);
        for (e, g) in GLOBALS.iter().enumerate() {
            let (codes, scales) = expert_bytes(e, rows, k);
            let (w, s) = (vec![rows, k / 2], vec![rows, k / 16]);
            ct.extend([
                (
                    format!("c.{e}.weight_packed"),
                    "U8",
                    w.clone(),
                    codes.clone(),
                ),
                (
                    format!("c.{e}.weight_scale"),
                    "F8_E4M3",
                    s.clone(),
                    scales.clone(),
                ),
                (
                    format!("c.{e}.weight_global_scale"),
                    "F32",
                    vec![],
                    g.to_le_bytes().to_vec(),
                ),
                (
                    format!("c.{e}.input_global_scale"),
                    "F32",
                    vec![],
                    1.0f32.to_le_bytes().to_vec(),
                ),
            ]);
            mo.extend([
                (format!("m.{e}.weight"), "U8", w, codes),
                (format!("m.{e}.weight_scale"), "F8_E4M3", s, scales),
                (
                    format!("m.{e}.weight_scale_2"),
                    "F32",
                    vec![],
                    (1.0f32 / g).to_le_bytes().to_vec(),
                ),
            ]);
        }
        (file(ct), file(mo))
    }

    /// The place of the compressed-tensors weight `stem`: its block scale, global scale and
    /// input scale.
    fn ct_own(stem: &str) -> (String, Placed) {
        let companions = ["weight_scale", "weight_global_scale", "input_global_scale"]
            .map(|s| format!("{stem}.{s}"));
        (
            format!("{stem}.weight_packed"),
            Placed {
                format: Format::Nvfp4Ct,
                companions: companions.to_vec(),
            },
        )
    }

    /// Compressed-tensors NVFP4 experts normalize to what ModelOpt's NVFP4 of the same codes and
    /// block scales gives with `weight_scale_2 = 1 / g`, bit for bit: a linear and a stack as
    /// matrices, a linear as a row table, and a cut of each by rows and by experts; each
    /// expert's `.scale` is the f32 reciprocal of its own global scale, and every decoded value
    /// is the stored code and block scale divided by the global scale.
    ///
    /// FAIL-first: a `.scale` of `g` instead of `1 / g`, or of the first expert's `1 / g` for
    /// every expert, differs in the `.scale` and the decoded values.
    #[test]
    fn compressed_tensors_nvfp4_normalizes_to_the_modelopt_form_with_the_reciprocal_scale() {
        let (rows, k) = (3, 128);
        let (ct, mo) = twins(rows, k);
        let n = GLOBALS.len();
        let ct_owned: Vec<_> = (0..n).map(|e| ct_own(&format!("c.{e}"))).collect();
        let mo_owned: Vec<_> = (0..n).map(|e| own(&format!("m.{e}"))).collect();
        let (cs, ms) = (sources(&ct_owned), sources(&mo_owned));
        let m = Use::Matrix(Plain::Q8_0);
        let same = |role, c: &[(&str, &Placed)], m: &[(&str, &Placed)]| {
            let a = normalize(|x| ct.raw(x), c, role).unwrap();
            let b = normalize(|x| mo.raw(x), m, role).unwrap();
            assert_eq!(a, b, "{role:?} {:?}", c[0].0);
            a
        };
        let stack = same(m, &cs, &ms);
        assert_eq!(stack.type_id, GgmlType::NVFP4.as_u32());
        assert_eq!(stack.dims, [k as u64, rows as u64, n as u64]);
        for e in 0..n {
            let one = same(m, &cs[e..=e], &ms[e..=e]);
            assert_eq!(one.dims, [k as u64, rows as u64]);
            let rows_of = same(Use::Rows, &cs[e..=e], &ms[e..=e]);
            assert_eq!(rows_of.type_id, GgmlType::NVFP4_ROWS.as_u32());
            assert_eq!(rows_of.scale, one.scale);
            let (scale, name) = nvfp4_rows_scale(&|x| ct.raw(x), cs[e].0, cs[e].1).unwrap();
            assert_eq!(
                (scale, name),
                (one.scale[0], ct_owned[e].1.companions[0].as_str())
            );
        }
        let per = stack.bytes.len() / n;
        for (e, g) in GLOBALS.iter().enumerate() {
            assert_eq!(
                stack.scale[e].to_bits(),
                (1.0f32 / g).to_bits(),
                "expert {e}"
            );
            let (codes, scales) = expert_bytes(e, rows, k);
            let blocks = &stack.bytes[e * per..][..per];
            for (r, row) in blocks.chunks_exact(k / 64 * 36).enumerate() {
                let mut v = vec![0.0f32; k];
                dequant_row(GgmlType::NVFP4, row, &mut v).unwrap();
                for (i, got) in v.iter().enumerate() {
                    // Value `i` is the low nibble of its pair's byte when `i` is even.
                    let byte = codes[r * k / 2 + i / 2];
                    let code = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let doubled = fp4_pair(code)[0];
                    let half = ue4m3_to_f32_half(scales[r * k / 16 + i / 16]).unwrap();
                    let want = f64::from(doubled) * f64::from(half) / f64::from(*g);
                    let got = f64::from(*got * stack.scale[e]);
                    assert!(
                        (got - want).abs() <= want.abs() * 3e-7,
                        "expert {e} row {r} value {i}: {got} against {want}"
                    );
                }
            }
        }
        let dims = [k as u64, rows as u64];
        for (role, part) in [(m, (1, 1..3)), (m, (0, 0..64))] {
            let c = normalize_part(|x| ct.raw(x), &cs[..1], role, &dims, part.clone()).unwrap();
            let o = normalize_part(|x| mo.raw(x), &ms[..1], role, &dims, part.clone()).unwrap();
            assert_eq!(c, o, "{part:?}");
        }
        let dims = [k as u64, rows as u64, n as u64];
        let c = normalize_part(|x| ct.raw(x), &cs, m, &dims, (2, 1..3)).unwrap();
        let o = normalize_part(|x| mo.raw(x), &ms, m, &dims, (2, 1..3)).unwrap();
        assert_eq!(c, o);
        assert_eq!(c.scale.len(), 2);
    }

    /// A one-row compressed-tensors weight of `k` values beside the given global and input scales,
    /// each as `(dtype, shape, bytes)`, its place naming both.
    fn ct_triple(
        k: usize,
        global: (&str, &[usize], &[u8]),
        input: (&str, &[usize], &[u8]),
    ) -> (Safetensors, (String, Placed)) {
        let (codes, scales) = expert_bytes(0, 1, k);
        let st = file(vec![
            ("t.weight_packed".into(), "U8", vec![1, k / 2], codes),
            ("t.weight_scale".into(), "F8_E4M3", vec![1, k / 16], scales),
            (
                "t.weight_global_scale".into(),
                global.0,
                global.1.to_vec(),
                global.2.to_vec(),
            ),
            (
                "t.input_global_scale".into(),
                input.0,
                input.1.to_vec(),
                input.2.to_vec(),
            ),
        ]);
        (st, ct_own("t"))
    }

    /// A global scale that is no positive divisor, and an input scale that is no finite positive
    /// F32, are refused by name by every reader of the weight: the normalize of a matrix, a stack
    /// and a row table, the rows' scale and [`input_global_scale`]; any other positive value of
    /// the input scale is accepted and changes nothing.
    ///
    /// FAIL-first: reading the global scale as a multiplier accepts 0 and negative ones;
    /// dropping the input scale's check accepts NaN, 0 and negative ones.
    #[test]
    fn compressed_tensors_scalars_that_are_no_scale_are_refused_by_name() {
        let one = 1.0f32.to_le_bytes();
        let f32_scalar = |v: f32| ("F32", &[][..], v.to_le_bytes().to_vec());
        let build = |global: f32, input: f32| {
            let (g, i) = (f32_scalar(global), f32_scalar(input));
            ct_triple(64, (g.0, g.1, &g.2), (i.0, i.1, &i.2))
        };
        let readers = |st: &Safetensors, w: &(String, Placed)| {
            let get = |n: &str| st.raw(n);
            let src = [(w.0.as_str(), &w.1)];
            let m = Use::Matrix(Plain::Q8_0);
            vec![
                normalize(get, &src, m).map(|_| ()),
                normalize(get, &[src[0], src[0]], m).map(|_| ()),
                normalize(get, &src, Use::Rows).map(|_| ()),
                nvfp4_rows_scale(&get, src[0].0, src[0].1).map(|_| ()),
                input_global_scale(&get, src[0].0, src[0].1).map(|_| ()),
            ]
        };
        let (st, w) = build(2.0, 1.0);
        for r in readers(&st, &w) {
            r.unwrap();
        }
        let get = |n: &str| st.raw(n);
        assert_eq!(input_global_scale(&get, &w.0, &w.1).unwrap(), Some(1.0));
        let (mo, mown) = (fixture(), own(STEMS[0]));
        assert_eq!(
            input_global_scale(&|n| mo.raw(n), &mown.0, &mown.1).unwrap(),
            None
        );

        let base = normalize(get, &[(&w.0, &w.1)], Use::Matrix(Plain::Q8_0)).unwrap();
        for input in [0.25f32, 1e-30, 1e30] {
            let (st, w) = build(2.0, input);
            let n = normalize(|n| st.raw(n), &[(&w.0, &w.1)], Use::Matrix(Plain::Q8_0)).unwrap();
            assert_eq!(n, base, "input scale {input}");
        }
        for (input, shown) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "inf"),
            (f32::NEG_INFINITY, "-inf"),
            (0.0, "0"),
            (-0.0, "-0"),
            (-1.0, "-1"),
        ] {
            let (st, w) = build(2.0, input);
            let want =
                format!("input scale t.input_global_scale is {shown}, not a finite positive F32");
            for e in readers(&st, &w) {
                let e = e.unwrap_err().to_string();
                assert!(
                    e.starts_with("t.weight_packed: NVFP4-compressed-tensors input scale"),
                    "{e}"
                );
                assert!(e.contains(&want), "{e}");
            }
        }
        let bf16 = [0x80u8, 0x3f];
        let two_values = [one, one].concat();
        for (dtype, shape, bytes, want) in [
            ("BF16", &[][..], &bf16[..], "is BF16 [], not one F32"),
            ("F32", &[2][..], &two_values[..], "is F32 [2], not one F32"),
        ] {
            let (g, i) = (f32_scalar(2.0), (dtype, shape, bytes));
            let (st, w) = ct_triple(64, (g.0, g.1, &g.2), i);
            for e in readers(&st, &w) {
                let e = e.unwrap_err().to_string();
                assert!(
                    e.contains("t.input_global_scale") && e.contains(want),
                    "{e}"
                );
            }
        }
        for (global, shown, shape_error) in [
            (f32::NAN, "NaN", true),
            (f32::INFINITY, "inf", true),
            (0.0, "0", false),
            (-0.0, "-0", false),
            (-2.0, "-2", false),
            (
                1e-45,
                "0.000000000000000000000000000000000000000000001",
                false,
            ),
        ] {
            let (st, w) = build(global, 1.0);
            for e in readers(&st, &w).into_iter().take(4) {
                let e = e.unwrap_err().to_string();
                assert!(e.contains("t.weight_global_scale"), "{e}");
                if !shape_error {
                    assert!(
                        e.contains(&format!("weight_global_scale t.weight_global_scale is {shown}, not a positive divisor with a finite reciprocal")),
                        "{e}"
                    );
                }
            }
        }
    }
}
