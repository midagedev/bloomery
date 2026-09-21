//! Host-side reference for the GPU kernel gates (docs/gpu-design.md decision
//! 3, kernel layer). One owner for "what is the right answer" so the kernel
//! tracks do not each grow a generator: a weight row is dequantized with
//! `gguf::quant::dequant_row` — the scalar transcription gate-1-1 pins
//! against ggml's `to_float` — and dotted with the activation in f64. No
//! device code here; the gate binaries under `src/bin/` bring the device.
//!
//! The kernel gate is `max|y - y_ref| / max|y_ref| <= 1e-2` per shape, the
//! stage-0 contract for q8_1-activation kernels (measured floor 3–5e-3).

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, TensorInfo};
use std::path::PathBuf;

/// The model every gate reads unless `BLOOMERY_REF_MODEL` says otherwise.
pub const DEFAULT_MODEL: &str = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

/// The stage-0 kernel gate band.
pub const KERNEL_BAND: f32 = 1e-2;

pub type GateError = Box<dyn std::error::Error>;

pub fn open_model() -> Result<Gguf, GateError> {
    let path = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    Ok(Gguf::open(path)?)
}

/// `m` activation columns of `k` f32 each, concatenated. A fixed LCG mapped
/// to [-1, 1) with every 61st value scaled by 8 so a block's amax is not
/// always near 1 — the quantizer's scale path sees spread. Values never
/// depend on time or on the host.
pub fn activations(k: usize, m: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    (0..k * m)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let u = ((s >> 8) & 0xffff) as f32 / 32_768.0 - 1.0;
            if i % 61 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

/// Bytes of one row of `k` values of type `ty`.
pub fn row_bytes(ty: GgmlType, k: usize) -> Result<usize, GateError> {
    let blck = ty.blck_size().ok_or("row_bytes: unsupported type")?.max(1) as usize;
    let tsz = ty.type_size().ok_or("row_bytes: unsupported type")? as usize;
    if !k.is_multiple_of(blck) {
        return Err(format!("row_bytes: k = {k} is not a multiple of block {blck}").into());
    }
    Ok(k / blck * tsz)
}

/// Reference `y = W[row0 .. row0 + n_rows] · x` for raw rows of type `ty`
/// with `k` values each: `n_rows * m` f32, row-major with `m` outputs per
/// row (the kernels' output layout). `w` starts at row 0 of the span.
pub fn ref_gemv(
    ty: GgmlType,
    w: &[u8],
    k: usize,
    n_rows: usize,
    x: &[f32],
    m: usize,
) -> Result<Vec<f32>, GateError> {
    let rb = row_bytes(ty, k)?;
    if w.len() < rb * n_rows || x.len() < k * m {
        return Err(format!(
            "ref_gemv: w.len() {} < {} or x.len() {} < {}",
            w.len(),
            rb * n_rows,
            x.len(),
            k * m
        )
        .into());
    }
    let mut row = vec![0.0f32; k];
    let mut y = vec![0.0f32; n_rows * m];
    for r in 0..n_rows {
        dequant_row(ty, &w[r * rb..(r + 1) * rb], &mut row)?;
        for c in 0..m {
            let xc = &x[c * k..(c + 1) * k];
            let dot: f64 = row
                .iter()
                .zip(xc)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            y[r * m + c] = dot as f32;
        }
    }
    Ok(y)
}

/// Raw bytes of tensor `name`, with its info (dims[0] is K, the row width).
pub fn tensor_bytes<'a>(
    gguf: &'a Gguf,
    name: &str,
) -> Result<(&'a TensorInfo, &'a [u8]), GateError> {
    let t = gguf
        .find(name)
        .ok_or_else(|| format!("tensor {name} not in the model"))?;
    Ok((t, gguf.data(t)?))
}

/// `max|y - y_ref| / max|y_ref|`; an all-zero reference is an error, not 0.
pub fn max_rel_err(y: &[f32], y_ref: &[f32]) -> Result<f32, GateError> {
    if y.len() != y_ref.len() {
        return Err(format!("max_rel_err: len {} vs {}", y.len(), y_ref.len()).into());
    }
    let denom = y_ref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if denom == 0.0 {
        return Err("max_rel_err: reference is all zero".into());
    }
    let num = y
        .iter()
        .zip(y_ref)
        .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
    if !num.is_finite() {
        return Err("max_rel_err: non-finite kernel output".into());
    }
    Ok(num / denom)
}

/// Raw little-endian bytes as `u32` words (the K-quant kernels' load unit).
/// A length that is not a multiple of 4 is zero-padded in the last word.
pub fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

// ------------------------------------------------------- ik CUDA dump

/// One `tensor` row of the CUDA oracle dump's MANIFEST.tsv. `ne` is
/// ne0..ne3 as written: ne0 is the contiguous dimension (values of an
/// activation, rows of a MUL_MAT output); `sum` is the dumper's element
/// sum, usable to prove a VIEW row carries the same bytes as its base.
#[derive(Debug, Clone)]
pub struct RefRow {
    pub name: String,
    pub occurrence: u32,
    pub ty: String,
    pub ne: [u64; 4],
    pub bytes: u64,
    pub sum: f64,
    pub op: String,
}

impl RefRow {
    /// Product of the four ne counts (elements, = bytes/4 for an f32 row).
    pub fn count(&self) -> u64 {
        self.ne.iter().product()
    }

    /// The dump file's name: `<name>.<occurrence>.f32`.
    pub fn file_name(&self) -> String {
        format!("{}.{}.f32", self.name, self.occurrence)
    }
}

/// Directory of the ik CUDA oracle dump (docs/gpu-design.md decision 3):
/// `$BLOOMERY_REF_CUDA` if set, else `$BLOOMERY_DATA/ref_cuda`, else the
/// workstation default. Read-only for every caller.
pub fn ref_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BLOOMERY_REF_CUDA") {
        return PathBuf::from(p);
    }
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    PathBuf::from(data).join("ref_cuda")
}

/// Parse the dump's MANIFEST.tsv. Header lines start with `#`; data rows
/// are tab-separated
/// `tensor name occurrence type ne0 ne1 ne2 ne3 bytes sum op`.
/// Tensor names may contain spaces, so fields are split on tabs only.
pub fn ref_manifest() -> Result<Vec<RefRow>, GateError> {
    let path = ref_dir().join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("ref_manifest: cannot read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || !line.starts_with("tensor\t") {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 11 {
            return Err(format!(
                "ref_manifest: {}: row has {} fields, want 11: {line}",
                path.display(),
                f.len()
            )
            .into());
        }
        let uparse = |s: &str| -> Result<u64, GateError> {
            s.parse::<u64>()
                .map_err(|e| format!("ref_manifest: {}: {s:?}: {e}", path.display()).into())
        };
        let fparse = |s: &str| -> Result<f64, GateError> {
            s.parse::<f64>()
                .map_err(|e| format!("ref_manifest: {}: {s:?}: {e}", path.display()).into())
        };
        rows.push(RefRow {
            name: f[1].to_string(),
            occurrence: uparse(f[2])? as u32,
            ty: f[3].to_string(),
            ne: [uparse(f[4])?, uparse(f[5])?, uparse(f[6])?, uparse(f[7])?],
            bytes: uparse(f[8])?,
            sum: fparse(f[9])?,
            op: f[10].to_string(),
        });
    }
    if rows.is_empty() {
        return Err(format!("ref_manifest: no tensor rows in {}", path.display()).into());
    }
    Ok(rows)
}

/// The manifest row for `(name, occurrence)`, if the dump holds it.
pub fn find_ref_row<'a>(
    man: &'a [RefRow],
    name: &str,
    occurrence: u32,
) -> Result<&'a RefRow, GateError> {
    man.iter()
        .find(|r| r.name == name && r.occurrence == occurrence)
        .ok_or_else(|| {
            format!(
                "find_ref_row: {}/{} not in {}MANIFEST.tsv",
                name,
                occurrence,
                ref_dir().display()
            )
            .into()
        })
}

/// Load one manifest row's f32 file, checking it against its own row: the
/// type must be f32, the row's byte count must equal 4·ne0·ne1·ne2·ne3 and
/// the file's length, and every value must be finite. Every error names the
/// offending path.
pub fn ref_tensor_of(row: &RefRow) -> Result<Vec<f32>, GateError> {
    let path = ref_dir().join(row.file_name());
    if row.ty != "f32" {
        return Err(format!(
            "ref_tensor_of: {} has type {}, want f32",
            path.display(),
            row.ty
        )
        .into());
    }
    let expect = 4_u64
        .checked_mul(row.count())
        .ok_or("ref_tensor_of: element count overflows")?;
    if row.bytes != expect {
        return Err(format!(
            "ref_tensor_of: {} manifest bytes {} != 4*count {}",
            path.display(),
            row.bytes,
            expect
        )
        .into());
    }
    let raw = std::fs::read(&path)
        .map_err(|e| format!("ref_tensor_of: cannot read {}: {e}", path.display()))?;
    if raw.len() as u64 != row.bytes {
        return Err(format!(
            "ref_tensor_of: {} is {} bytes, manifest says {}",
            path.display(),
            raw.len(),
            row.bytes
        )
        .into());
    }
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if let Some(i) = vals.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "ref_tensor_of: non-finite value at index {i} of {}",
            path.display()
        )
        .into());
    }
    Ok(vals)
}

/// `ref_tensor_of` over `find_ref_row`: load `(name, occurrence)`'s f32
/// file with its manifest row (dims, op, sum) for chain checking.
pub fn ref_tensor(name: &str, occurrence: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    let man = ref_manifest()?;
    let row = find_ref_row(&man, name, occurrence)?;
    Ok((row.clone(), ref_tensor_of(row)?))
}
