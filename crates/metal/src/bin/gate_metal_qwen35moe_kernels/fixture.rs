//! The kernels' gate's fixture file: the row's family's trunk (qwen35moe's or qwen3moe's) whose real tensors the clauses read. The A8, A4H and A8H clauses read the same split's tensors requantized as that affine type ([`Fixture::rows_as`]).

use std::path::{Path, PathBuf};

use bloomery_metal::kernels::Quant;
use gguf::Split;
use gguf::quant::{
    GgmlType, bf16_bits_to_f32, dequant_row, f32_to_f16_bits, fp32_to_bf16_bits, half_to_f32,
};

/// The rows of the fixture's `output.weight`: Qwen3.6's vocabulary. The
/// head's launch takes its row count from the file; this is the fixture's, for
/// the clauses that read the tensor's tail.
pub const OUTPUT_ROWS: usize = 248_320;

/// The fixture file: the trunk of the row's family the gate reads its tensors
/// from, opened as a split — the one reader that accepts bloomery's own value
/// types, which an MLX-format fixture's tensors are.
pub struct Fixture {
    split: Split,
    pub path: PathBuf,
}

impl Fixture {
    pub fn open(path: &Path) -> Result<Self, String> {
        let split = Split::open(path).map_err(|e| {
            format!(
                "the fixture {} did not open: {e} (copy the row's fixture's trunk file there, or name it with --fixture)",
                path.display()
            )
        })?;
        Ok(Self {
            split,
            path: path.to_path_buf(),
        })
    }

    /// The vocabulary the fixture's head has: the rows of its `output.weight`.
    pub fn output_rows(&self) -> Result<usize, String> {
        let (_, t) = self.tensor("output.weight")?;
        match t.dims[..] {
            [_, rows] => usize::try_from(rows)
                .map_err(|_| format!("output.weight has {rows} rows, past usize")),
            _ => Err(format!("output.weight is {:?}, not a matrix", t.dims)),
        }
    }

    /// The tensor `name`: its shard and its info.
    fn tensor(&self, name: &str) -> Result<(usize, &gguf::TensorInfo), String> {
        self.split
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))
    }

    fn data(&self, shard: usize, info: &gguf::TensorInfo, name: &str) -> Result<&[u8], String> {
        let file = self
            .split
            .shard(shard)
            .ok_or_else(|| format!("{name}: shard {shard} is not open"))?;
        file.data(info).map_err(|e| format!("{name}: {e}"))
    }

    /// The bytes of the 2-D tensor `name`, `rows` rows of `cols` columns of
    /// type `ty`, or the first rows of it.
    pub fn rows(
        &self,
        name: &str,
        ty: GgmlType,
        cols: usize,
        rows: usize,
    ) -> Result<&[u8], String> {
        let (shard, t) = self.tensor(name)?;
        if t.ty != ty || t.dims.len() < 2 || t.dims[0] != cols as u64 || t.dims[1] < rows as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {ty:?} with {cols} columns and at least {rows} rows",
                t.ty, t.dims
            ));
        }
        let all = self.data(shard, t, name)?;
        let q = quant(ty)?;
        let row_bytes = cols / q.values_per_block() * q.block_bytes();
        Ok(&all[..rows * row_bytes])
    }

    /// The bytes of rows `at..at + rows` of the tensor `name` (see [`Fixture::rows`]).
    pub fn rows_at(
        &self,
        name: &str,
        ty: GgmlType,
        cols: usize,
        at: usize,
        rows: usize,
    ) -> Result<&[u8], String> {
        let all = self.rows(name, ty, cols, at + rows)?;
        let q = quant(ty)?;
        let row_bytes = cols / q.values_per_block() * q.block_bytes();
        Ok(&all[at * row_bytes..])
    }

    /// One expert of a 3-D tensor: expert `expert` of `dims[2]`, `rows` by `cols`.
    pub fn expert(
        &self,
        name: &str,
        ty: GgmlType,
        cols: usize,
        rows: usize,
        expert: usize,
    ) -> Result<&[u8], String> {
        let (shard, t) = self.tensor(name)?;
        let q = quant(ty)?;
        if t.ty != ty
            || t.dims.len() != 3
            || t.dims[0] != cols as u64
            || t.dims[1] != rows as u64
            || expert as u64 >= t.dims[2]
            || !cols.is_multiple_of(q.values_per_block())
        {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {ty:?} expert {expert} of {cols} columns by {rows} rows",
                t.ty, t.dims
            ));
        }
        let all = self.data(shard, t, name)?;
        let row_bytes = cols / q.values_per_block() * q.block_bytes();
        let one = rows * row_bytes;
        let start = expert * one;
        let end = start + one;
        if end > all.len() {
            return Err(format!(
                "{name} expert {expert} needs {end} bytes and the tensor holds {}",
                all.len()
            ));
        }
        Ok(&all[start..end])
    }

    /// The f32 tensor `name` of `len` values.
    pub fn f32s(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let (shard, t) = self.tensor(name)?;
        if t.ty != GgmlType::F32 || t.dims != [len as u64] {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {len} f32s",
                t.ty, t.dims
            ));
        }
        let bytes = self.data(shard, t, name)?;
        let mut out = vec![0f32; len];
        dequant_row(GgmlType::F32, bytes, &mut out).map_err(|e| format!("{name}: {e}"))?;
        Ok(out)
    }

    /// Whether the clauses read `quant`'s weights requantized
    /// ([`Fixture::rows_at_as`]): the split holds its typed sites at MLX_A4,
    /// so the other affine types are the same values in their own groups.
    pub fn requantizes(quant: Quant) -> bool {
        matches!(quant, Quant::A8 | Quant::A4H | Quant::A8H)
    }

    /// The bytes of rows `at..at + rows` of the 2-D tensor `name`, requantized
    /// as the affine type `quant` (A4, A8, A4H or A8H): the split's own rows of
    /// it, whatever type they are, dequantized to f32 and encoded as MLX's
    /// affine groups of that type — the clauses' weights of the types the
    /// split does not hold, the same values the split holds.
    pub fn rows_at_as(
        &self,
        quant: Quant,
        name: &str,
        cols: usize,
        at: usize,
        rows: usize,
    ) -> Result<Vec<u8>, String> {
        let (shard, t) = self.tensor(name)?;
        if t.dims.len() != 2 || t.dims[0] != cols as u64 || t.dims[1] < (at + rows) as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {cols} columns and rows {at}..{}",
                t.ty,
                t.dims,
                at + rows
            ));
        }
        let own = matvec_of(name, t.ty)?;
        let own_row = own.block_bytes() * (cols / own.values_per_block());
        let all = self.data(shard, t, name)?;
        let at = at * own_row;
        if at + rows * own_row > all.len() {
            return Err(format!(
                "{name} needs {} bytes and the tensor holds {}",
                at + rows * own_row,
                all.len()
            ));
        }
        let values = dequant_rows(name, t.ty, own_row, &all[at..], cols, rows)?;
        Ok(quantize_affine(quant, &values, cols))
    }

    /// The bytes of `rows` rows of the 2-D tensor `name`, requantized as
    /// `quant` (see [`Fixture::rows_at_as`]).
    pub fn rows_as(
        &self,
        quant: Quant,
        name: &str,
        cols: usize,
        rows: usize,
    ) -> Result<Vec<u8>, String> {
        self.rows_at_as(quant, name, cols, 0, rows)
    }

    /// One expert of the 3-D tensor `name`, requantized as `quant`: expert
    /// `expert` of `dims[2]`, `rows` by `cols` (see [`Fixture::rows_at_as`]).
    pub fn expert_as(
        &self,
        quant: Quant,
        name: &str,
        cols: usize,
        rows: usize,
        expert: usize,
    ) -> Result<Vec<u8>, String> {
        let (shard, t) = self.tensor(name)?;
        if t.dims.len() != 3
            || t.dims[0] != cols as u64
            || t.dims[1] != rows as u64
            || expert as u64 >= t.dims[2]
        {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads expert {expert} of {cols} columns by {rows} rows",
                t.ty, t.dims
            ));
        }
        let own = matvec_of(name, t.ty)?;
        let own_row = own.block_bytes() * (cols / own.values_per_block());
        let all = self.data(shard, t, name)?;
        let at = expert * rows * own_row;
        if at + rows * own_row > all.len() {
            return Err(format!(
                "{name} expert {expert} needs {} bytes and the tensor holds {}",
                at + rows * own_row,
                all.len()
            ));
        }
        let values = dequant_rows(name, t.ty, own_row, &all[at..], cols, rows)?;
        Ok(quantize_affine(quant, &values, cols))
    }

    /// The model's norm epsilon.
    pub fn rms_eps(&self) -> Result<f32, String> {
        self.split
            .shard(0)
            .ok_or_else(|| "the fixture's first shard is not open".to_string())?
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| "the fixture names no attention.layer_norm_rms_epsilon".to_string())
    }

    /// The first layer whose tensor `blk.L.{stem}` is of type `ty`.
    pub fn first_layer_of(&self, stem: &str, ty: GgmlType) -> Result<usize, String> {
        (0usize..)
            .map_while(|l| {
                self.split
                    .find(&format!("blk.{l}.{stem}"))
                    .map(|(_, t)| (l, t.ty))
            })
            .find_map(|(l, t)| (t == ty).then_some(l))
            .ok_or_else(|| format!("the fixture has no layer whose {stem} is {ty:?}"))
    }

    /// The bytes of an embedding table: `rows` rows of `cols`, its type's.
    pub fn table(&self, name: &str, cols: usize) -> Result<(&[u8], usize, GgmlType), String> {
        let (shard, t) = self.tensor(name)?;
        let ty = t.ty;
        if !matches!(ty, GgmlType::Q4_K | GgmlType::MLX_A4)
            || t.dims.len() != 2
            || t.dims[0] != cols as u64
        {
            return Err(format!(
                "{name} is {ty:?} {:?}, the gate reads a Q4_K or MLX_A4 table of {cols} columns",
                t.dims
            ));
        }
        let bytes = self.data(shard, t, name)?;
        Ok((bytes, t.dims[1] as usize, ty))
    }
}

/// The mat-vec type that reads `ty`: its block's values and bytes are the
/// kernels' own.
fn quant(ty: GgmlType) -> Result<Quant, String> {
    Quant::of_ggml(ty).ok_or_else(|| format!("{ty:?} is not a mat-vec type"))
}

/// The mat-vec type that reads `ty`, refusing by the tensor's name.
fn matvec_of(name: &str, ty: GgmlType) -> Result<Quant, String> {
    quant(ty).map_err(|e| format!("{name}: {e}"))
}

/// The f32 values of `rows` rows of `cols` columns whose bytes sit at `bytes`
/// in the tensor's own type, a row's span `own_row`: the reader's decode of
/// the split's rows.
fn dequant_rows(
    name: &str,
    ty: GgmlType,
    own_row: usize,
    bytes: &[u8],
    cols: usize,
    rows: usize,
) -> Result<Vec<f32>, String> {
    let mut values = vec![0f32; rows * cols];
    for (r, out) in values.chunks_mut(cols).enumerate() {
        dequant_row(ty, &bytes[r * own_row..], out).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(values)
}

/// `values` (`rows` rows of `cols` columns, `cols` a multiple of 64)
/// encoded as the affine type `quant`'s groups: a group of 64 the range
/// (max − min) over the type's top code (15 or 255) spans, its scale and bias
/// the range's ends in the type's pair storage (BF16 bits for A4 and A8, F16
/// bits for A4H and A8H), its codes the values on that grid. The fit is the
/// fixture's own — the reader's decode of these bytes is what the oracle and
/// the kernel read.
fn quantize_affine(quant: Quant, values: &[f32], cols: usize) -> Vec<u8> {
    debug_assert!(values.len().is_multiple_of(cols) && cols.is_multiple_of(64));
    let (bits, half): (usize, bool) = match quant {
        Quant::A4 => (4, false),
        Quant::A8 => (8, false),
        Quant::A4H => (4, true),
        Quant::A8H => (8, true),
        other => panic!("{} is no affine type", other.name()),
    };
    let top = (1u32 << bits) - 1;
    let per = 32 / bits;
    let block = quant.block_bytes();
    let groups = quant.values_per_block();
    let encode: fn(f32) -> u16 = if half {
        f32_to_f16_bits
    } else {
        fp32_to_bf16_bits
    };
    let decode: fn(u16) -> f32 = if half { half_to_f32 } else { bf16_bits_to_f32 };
    let mut out = vec![0u8; values.len() / groups * block];
    for (g, chunk) in values.chunks_exact(groups).enumerate() {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in chunk {
            min = min.min(v);
            max = max.max(v);
        }
        let s_bits = encode((max - min) / top as f32);
        let b_bits = encode(min);
        let (s, b) = (decode(s_bits), decode(b_bits));
        let mut words = vec![0u32; groups / per];
        for (j, &v) in chunk.iter().enumerate() {
            let code = if s > 0.0 {
                (((v - b) / s).round() as i64).clamp(0, i64::from(top)) as u32
            } else {
                0
            };
            words[j / per] |= code << (bits * (j % per));
        }
        let bytes = &mut out[g * block..];
        for (w, word) in words.iter().enumerate() {
            bytes[4 * w..4 * w + 4].copy_from_slice(&word.to_le_bytes());
        }
        let pair = block - 4;
        bytes[pair..pair + 2].copy_from_slice(&s_bits.to_le_bytes());
        bytes[pair + 2..pair + 4].copy_from_slice(&b_bits.to_le_bytes());
    }
    out
}
