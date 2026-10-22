//! The kernels' gate's fixture file: the qwen35moe trunk whose real tensors the clauses read.

use std::path::{Path, PathBuf};

use bloomery_metal::kernels::{BLOCK_VALUES, Quant};
use gguf::Gguf;
use gguf::quant::{GgmlType, dequant_row};

/// The fixture file: the qwen35moe trunk the gate reads its tensors from.
pub struct Fixture {
    file: Gguf,
    pub path: PathBuf,
}

impl Fixture {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = Gguf::open(path).map_err(|e| {
            format!(
                "the fixture {} did not open: {e} (copy the qwen35moe fixture's trunk file there, or name it with --fixture)",
                path.display()
            )
        })?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
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
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != ty || t.dims.len() < 2 || t.dims[0] != cols as u64 || t.dims[1] < rows as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {ty:?} with {cols} columns and at least {rows} rows",
                t.ty, t.dims
            ));
        }
        let all = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        let row_bytes = cols / BLOCK_VALUES * quant_block_bytes(ty)?;
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
        let row_bytes = cols / BLOCK_VALUES * quant_block_bytes(ty)?;
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
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != ty
            || t.dims.len() != 3
            || t.dims[0] != cols as u64
            || t.dims[1] != rows as u64
            || expert as u64 >= t.dims[2]
            || !cols.is_multiple_of(BLOCK_VALUES)
        {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {ty:?} expert {expert} of {cols} columns by {rows} rows",
                t.ty, t.dims
            ));
        }
        let all = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        let row_bytes = cols / BLOCK_VALUES * quant_block_bytes(ty)?;
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
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != GgmlType::F32 || t.dims != [len as u64] {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {len} f32s",
                t.ty, t.dims
            ));
        }
        let bytes = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        let mut out = vec![0f32; len];
        dequant_row(GgmlType::F32, bytes, &mut out).map_err(|e| format!("{name}: {e}"))?;
        Ok(out)
    }

    /// The model's norm epsilon.
    pub fn rms_eps(&self) -> Result<f32, String> {
        self.file
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| "the fixture names no attention.layer_norm_rms_epsilon".to_string())
    }

    /// The bytes of the Q4_K embedding table: `rows` rows of `cols`.
    pub fn table(&self, name: &str, cols: usize) -> Result<(&[u8], usize), String> {
        let t = self
            .file
            .find(name)
            .ok_or_else(|| format!("the fixture has no tensor {name}"))?;
        if t.ty != GgmlType::Q4_K || t.dims.len() != 2 || t.dims[0] != cols as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads a Q4_K table of {cols} columns",
                t.ty, t.dims
            ));
        }
        let bytes = self.file.data(t).map_err(|e| format!("{name}: {e}"))?;
        Ok((bytes, t.dims[1] as usize))
    }
}

fn quant_block_bytes(ty: GgmlType) -> Result<usize, String> {
    match ty {
        GgmlType::Q4_K => Ok(Quant::Q4K.block_bytes()),
        GgmlType::Q6_K => Ok(Quant::Q6K.block_bytes()),
        other => Err(format!("{other:?} is not a mat-vec type")),
    }
}
