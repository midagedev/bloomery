//! The kernels' gate's fixture file: the Nano fixture whose real tensors the
//! Q5_0 clauses read.

use std::path::{Path, PathBuf};

use bloomery_metal::kernels::Quant;
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row};

use super::flag;

/// Why a clause that needs the fixture cannot run: the machine has none (the
/// clause is skipped, with the reason), or the file is there and does not
/// read (the clause fails).
pub enum Unavailable {
    Absent(String),
    Broken(String),
}

/// The fixture file: the Nano trunk the gate reads its tensors from.
pub struct Fixture {
    split: Split,
    pub path: PathBuf,
}

/// The fixture the run reads: `--fixture PATH`, else the Nemotron fixture
/// under the fixture root (`BLOOMERY_FIXTURE_ROOT`), as the reference sets
/// resolve it. A path named with `--fixture` that does not open is broken; a
/// root that holds no fixture is absent.
pub fn open(args: &[String]) -> Result<Fixture, Unavailable> {
    let path = match flag(args, "--fixture") {
        Some(path) => PathBuf::from(path),
        None => match refset::fixture::first_shard("nemotron_h") {
            Ok(path) => PathBuf::from(path),
            Err(e) => {
                return Err(Unavailable::Absent(format!(
                    "no Nemotron fixture: {e} (copy the nemotron fixture there, or name its file with --fixture)"
                )));
            }
        },
    };
    Fixture::open(&path).map_err(Unavailable::Broken)
}

impl Fixture {
    fn open(path: &Path) -> Result<Self, String> {
        let split = Split::open(path).map_err(|e| {
            format!(
                "the fixture {} did not open: {e} (name the Nemotron fixture's file with --fixture)",
                path.display()
            )
        })?;
        Ok(Self {
            split,
            path: path.to_path_buf(),
        })
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

    /// The bytes of the 2-D tensor `name`, the first `rows` rows of `cols`
    /// columns of type `quant`.
    pub fn rows(
        &self,
        name: &str,
        quant: Quant,
        cols: usize,
        rows: usize,
    ) -> Result<&[u8], String> {
        let (shard, t) = self.tensor(name)?;
        if t.ty != quant.ggml()
            || t.dims.len() != 2
            || t.dims[0] != cols as u64
            || t.dims[1] < rows as u64
        {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads {:?} with {cols} columns and at least {rows} rows",
                t.ty,
                t.dims,
                quant.ggml()
            ));
        }
        let all = self.data(shard, t, name)?;
        let row_bytes = cols / quant.values_per_block() * quant.block_bytes();
        Ok(&all[..rows * row_bytes])
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

    /// The embedding table `name` of `cols` columns: its bytes and its rows.
    pub fn table(&self, name: &str, quant: Quant, cols: usize) -> Result<(&[u8], usize), String> {
        let (shard, t) = self.tensor(name)?;
        if t.ty != quant.ggml() || t.dims.len() != 2 || t.dims[0] != cols as u64 {
            return Err(format!(
                "{name} is {:?} {:?}, the gate reads a {:?} table of {cols} columns",
                t.ty,
                t.dims,
                quant.ggml()
            ));
        }
        Ok((self.data(shard, t, name)?, t.dims[1] as usize))
    }
}
