//! The kernels' gate's fixture file: the row's family's trunk (qwen35moe's or qwen3moe's) whose real tensors the clauses read.

use std::path::{Path, PathBuf};

use bloomery_metal::kernels::Quant;
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row};

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
