//! Host-side reference for the GPU kernel gates (docs/gpu-design.md decision
//! 3, kernel layer). One owner for "what is the right answer" so the kernel
//! tracks do not each grow a generator: a weight row is dequantized with
//! `gguf::quant::dequant_row` — the scalar transcription gate-1-1 pins
//! against ggml's `to_float` — and dotted with the activation in f64. No
//! device code here; the gate binaries under `src/bin/` bring the device.
//!
//! The kernel gate is `max|y - y_ref| / max|y_ref| <= 1e-2` per shape, with
//! `y_ref` computed on the SAME quantized activations the kernel consumes
//! (a correct kernel sits near 1e-7). Against the raw activations a correct
//! q8_1 kernel measures up to 2.4e-2 on real inputs and ik's own CUDA output
//! up to 1.7e-2: that distance is the design's quantization noise and is
//! judged at the block layer, not here (docs/gpu-design.md decision 3).

pub mod block;
pub mod prompts;
pub mod ptx;

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, TensorInfo};
use std::path::PathBuf;

/// The model every gate reads unless `BLOOMERY_REF_MODEL` says otherwise.
pub const DEFAULT_MODEL: &str = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

/// Kernel gate band against the quantized-input reference.
/// PIN(2026-09-21): tightened 1e-2 -> 1e-5. Measured worst on the landed
/// kernels: 1.149e-7 (gate_p1, 12 shapes), 1.375e-7 (gate_p2); the f32
/// gemvs already gate at 1e-5 (worst 1.9e-7). 1e-2 was sized for the
/// raw-activation comparison this band no longer makes, and would pass a
/// sub-block scale defect landing near 1e-3 on a shape with no hash pin.
pub const KERNEL_BAND: f32 = 1e-5;

pub type GateError = Box<dyn std::error::Error>;

/// The exit of every gate binary's `main`: `Ok` is success; an `Err` prints
/// `<name>: <error>` (the Display, not the Debug a `Result` main prints) and
/// each `source()` beneath it as `  caused by: ...` to stderr, then fails.
/// A message that already opens with `<name>: ` is not prefixed twice.
pub fn exit_with(name: &str, r: Result<(), GateError>) -> std::process::ExitCode {
    let Err(e) = r else {
        return std::process::ExitCode::SUCCESS;
    };
    let msg = e.to_string();
    if msg.starts_with(&format!("{name}: ")) {
        eprintln!("{msg}");
    } else {
        eprintln!("{name}: {msg}");
    }
    let mut cause = e.source();
    while let Some(c) = cause {
        eprintln!("  caused by: {c}");
        cause = c.source();
    }
    std::process::ExitCode::FAILURE
}

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

/// `max|y - y_ref| / max|y_ref|`; an all-zero reference or any non-finite
/// value on either side is an error, not a score.
pub fn max_rel_err(y: &[f32], y_ref: &[f32]) -> Result<f32, GateError> {
    if y.len() != y_ref.len() {
        return Err(format!("max_rel_err: len {} vs {}", y.len(), y_ref.len()).into());
    }
    // `f32::max` returns the other operand for NaN, so the folds below
    // cannot see one: an all-NaN output would score 0. Scan first.
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("max_rel_err: non-finite kernel output at {i}").into());
    }
    if let Some(i) = y_ref.iter().position(|v| !v.is_finite()) {
        return Err(format!("max_rel_err: non-finite reference at {i}").into());
    }
    let denom = y_ref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    if denom == 0.0 {
        return Err("max_rel_err: reference is all zero".into());
    }
    let num = y
        .iter()
        .zip(y_ref)
        .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
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
/// The four trailing fields are the v2 columns (`contig`, `logical`,
/// `src0`, `src1`); `None` on a set written before they existed.
#[derive(Debug, Clone)]
pub struct RefRow {
    pub name: String,
    pub occurrence: u32,
    pub ty: String,
    pub ne: [u64; 4],
    pub bytes: u64,
    pub sum: f64,
    pub op: String,
    pub contig: Option<u8>,
    pub logical: Option<u8>,
    pub src0: Option<String>,
    pub src1: Option<String>,
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

    /// Prove this row's type, dims and op — the chain check every consumer
    /// runs before trusting a tensor (`op` = what produced it). `op = "in"`
    /// only checks type and dims (an input's op varies). `what` names the
    /// call site in the error.
    pub fn expect(&self, what: &str, ty: &str, ne: [u64; 4], op: &str) -> Result<(), GateError> {
        if self.ty != ty || self.ne != ne || (op != "in" && self.op != op) {
            return Err(format!(
                "expect: {what}: {} is {} {:?} op {}, want {ty} {ne:?} op {op}",
                self.name, self.ty, self.ne, self.op
            )
            .into());
        }
        Ok(())
    }
}

/// Directory of the ik CUDA oracle dump (docs/gpu-design.md decision 3):
/// `$BLOOMERY_REF_CUDA` if set (an absolute path), else the set named by
/// `$BLOOMERY_REF_SET`, else `$BLOOMERY_DATA/ref_cuda_v2`, else the
/// workstation default. Read-only for every caller. `BLOOMERY_REF_SET` is
/// how a gate points itself at the pre-v2 `ref_cuda` (plain files only,
/// no logical twins) or the CPU `ref` without a code change.
pub fn ref_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BLOOMERY_REF_CUDA") {
        return PathBuf::from(p);
    }
    if let Ok(s) = std::env::var("BLOOMERY_REF_SET") {
        return ref_dir_named(&s);
    }
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    PathBuf::from(data).join("ref_cuda_v2")
}

/// A named dump set's directory: an absolute `set` is the directory
/// itself, anything else is `<$BLOOMERY_DATA>/<set>` (`ref_cuda_v2`, the
/// CPU `ref`, ...). Unlike `ref_dir` this consults no environment.
pub fn ref_dir_named(set: &str) -> PathBuf {
    let p = PathBuf::from(set);
    if p.is_absolute() {
        return p;
    }
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    PathBuf::from(data).join(set)
}

/// Parse the MANIFEST.tsv of the set at `dir`. Header lines start with
/// `#`; data rows are tab-separated
/// `tensor name occurrence type ne0 ne1 ne2 ne3 bytes sum op`
/// with four optional trailing fields `contig logical src0 src1` —
/// sets written before the v2 columns carry the 11-field width and parse
/// to `None`s. Tensor names may contain spaces, so fields are split on
/// tabs only.
pub fn ref_manifest_in(dir: &std::path::Path) -> Result<Vec<RefRow>, GateError> {
    let path = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("ref_manifest: cannot read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || !line.starts_with("tensor\t") {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 11 && f.len() != 15 {
            return Err(format!(
                "ref_manifest: {}: row has {} fields, want 11 or 15: {line}",
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
            contig: f.get(11).map(|s| uparse(s)).transpose()?.map(|v| v as u8),
            logical: f.get(12).map(|s| uparse(s)).transpose()?.map(|v| v as u8),
            src0: f.get(13).map(|s| s.to_string()),
            src1: f.get(14).map(|s| s.to_string()),
        });
    }
    if rows.is_empty() {
        return Err(format!("ref_manifest: no tensor rows in {}", path.display()).into());
    }
    Ok(rows)
}

/// `ref_manifest_in` of `ref_dir()`.
pub fn ref_manifest() -> Result<Vec<RefRow>, GateError> {
    ref_manifest_in(&ref_dir())
}

/// The manifest row for `(name, occurrence)`, if the dump holds it.
pub fn find_ref_row<'a>(
    man: &'a [RefRow],
    name: &str,
    occurrence: u32,
) -> Result<&'a RefRow, GateError> {
    find_ref_row_in(&ref_dir(), man, name, occurrence)
}

/// `find_ref_row` naming `dir` in the error (the manifest slice is
/// directory-independent; only the message was not).
pub fn find_ref_row_in<'a>(
    dir: &std::path::Path,
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
                dir.display()
            )
            .into()
        })
}

/// Load one manifest row's f32 file, checking it against its own row: the
/// type must be f32, the row's byte count must equal 4·ne0·ne1·ne2·ne3 and
/// the file's length, and every value must be finite. Every error names the
/// offending path.
pub fn ref_tensor_of(row: &RefRow) -> Result<Vec<f32>, GateError> {
    ref_tensor_of_in(&ref_dir(), row)
}

/// `ref_tensor_of` of the set at `dir`.
pub fn ref_tensor_of_in(dir: &std::path::Path, row: &RefRow) -> Result<Vec<f32>, GateError> {
    let path = dir.join(row.file_name());
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
    load_ref(&ref_manifest()?, name, occurrence)
}

/// `ref_tensor` over a manifest the caller already parsed: load
/// `(name, occ)`'s f32 file with its manifest row. A gate that reads many
/// tensors parses the manifest once and calls this.
pub fn load_ref(man: &[RefRow], name: &str, occ: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    let row = find_ref_row(man, name, occ)?;
    Ok((row.clone(), ref_tensor_of(row)?))
}

/// The tensor's LOGICAL elements in ggml index order from `ref_dir()`: the
/// `.logical.f32` twin when the dump wrote one, else the plain file when
/// the row is provably not a flat VIEW read (a contiguous tensor's plain
/// file IS its logical order; a pre-v2 manifest without the `contig`
/// column is accepted for non-VIEW rows only). A VIEW row without a
/// logical twin is an error, never a silent flat read — that flat read is
/// a different tensor than the one being asked for.
pub fn ref_tensor_logical(name: &str, occurrence: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    let man = ref_manifest()?;
    let row = find_ref_row(&man, name, occurrence)?;
    Ok((row.clone(), ref_tensor_logical_in(&ref_dir(), row)?))
}

/// `ref_tensor_logical` of the set at `dir`, over a row already found.
pub fn ref_tensor_logical_in(dir: &std::path::Path, row: &RefRow) -> Result<Vec<f32>, GateError> {
    if row.logical == Some(1) {
        if row.ty != "f32" {
            return Err(format!(
                "ref_tensor_logical: {} has type {}, want f32",
                row.name, row.ty
            )
            .into());
        }
        let path = dir.join(format!("{}.{}.logical.f32", row.name, row.occurrence));
        let expect = 4_u64
            .checked_mul(row.count())
            .ok_or("ref_tensor_logical: element count overflows")?;
        let raw = std::fs::read(&path)
            .map_err(|e| format!("ref_tensor_logical: cannot read {}: {e}", path.display()))?;
        if raw.len() as u64 != expect {
            return Err(format!(
                "ref_tensor_logical: {} is {} bytes, want {} (4*count)",
                path.display(),
                raw.len(),
                expect
            )
            .into());
        }
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Some(i) = vals.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "ref_tensor_logical: non-finite value at index {i} of {}",
                path.display()
            )
            .into());
        }
        return Ok(vals);
    }
    let plain_ok = match (row.contig, row.op.as_str()) {
        // v2 manifest: the column decides. Pre-v2 manifest: only the op
        // can rule out a flat VIEW read.
        (Some(c), _) => c == 1,
        (None, "VIEW") => false,
        (None, _) => true,
    };
    if !plain_ok {
        return Err(format!(
            "ref_tensor_logical: {} is a view/non-contiguous row with no \
             logical twin in {} — the plain file is a flat read, not the tensor",
            row.name,
            dir.display()
        )
        .into());
    }
    ref_tensor_of_in(dir, row)
}

// -------------------------------------------------- promoted gate helpers
// Single owners of the reference-side helpers the gate bins first wrote
// locally; the bins import them from here. `gate_p5` still carries its own
// `verdict` and `bits_equal` — its track owns that file.

/// The word a gate prints for one check's outcome: `PASS` or `FAIL`. The
/// gates' tables are compared line by line across rounds, so the spelling
/// has one owner.
pub fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// Same length and the same bits at every index — the bit-identity contract
/// the fusion, rerun and replay checks assert. Not `==`: that calls `-0.0`
/// equal to `0.0` and two NaNs unequal, and both are differences a fusion
/// defect can produce.
pub fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Mean microseconds per replay of graph `g`: one warm replay and a
/// synchronize, then `n` replays timed to a single synchronize at the end.
/// Lead-only timing under the machine lease — no correctness path calls it.
#[cfg(feature = "gpu")]
pub fn us_per_replay(
    g: &bloomery_gpu::Graph,
    stream: &cuda_core::CudaStream,
    n: u32,
) -> Result<f64, GateError> {
    g.launch(stream)?;
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        g.launch(stream)?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(n))
}

/// Prove the dump's VIEW convention for one view read as a PLAIN file:
/// `view` must equal `base` from `off` — flat memory from the view's base
/// pointer, not the materialized gather a consumer of the logical tensor
/// wants (gate_p4's local `view_flat`).
pub fn view_flat(view: &[f32], base: &[f32], off: usize, what: &str) -> Result<(), GateError> {
    if off + view.len() > base.len()
        || !view
            .iter()
            .zip(&base[off..off + view.len()])
            .all(|(x, y)| x.to_bits() == y.to_bits())
    {
        return Err(format!(
            "view_flat: {what}: dump is not the flat base memory at +{off} — \
             view convention changed"
        )
        .into());
    }
    Ok(())
}

/// Host scalar router reference, transcribed from the CPU engine's routing
/// (`model::moe::route_inner`): softmax with a serial f32 max fold, f32
/// `exp`, f64 sum accumulated ascending, f32 divide by `sum as f32`; top-k
/// by `(probability desc, id asc)`; weights `probs[id] * scale`. Layout
/// ggml t-major: logits/probs `[64, m]`, ids/weights `[6, m]`. Non-finite
/// logits are an error (gate_p6's local `route_ref` panics there).
pub fn route_ref(
    logits: &[f32],
    m: usize,
    scale: f32,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>), GateError> {
    let (n, k) = (64usize, 6usize);
    if logits.len() != n * m {
        return Err(format!("route_ref: logits.len() {} != {n}*{m}", logits.len()).into());
    }
    if let Some(i) = logits.iter().position(|v| !v.is_finite()) {
        return Err(format!("route_ref: non-finite logit at {i}").into());
    }
    let mut probs = vec![0.0f32; n * m];
    for t in 0..m {
        let src = &logits[t * n..(t + 1) * n];
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        let dst = &mut probs[t * n..(t + 1) * n];
        for (o, &v) in dst.iter_mut().zip(src) {
            let e = (v - max).exp();
            *o = e;
            sum += f64::from(e);
        }
        for o in dst.iter_mut() {
            *o /= sum as f32;
        }
    }
    let mut ids = vec![0i32; k * m];
    let mut weights = vec![0.0f32; k * m];
    let mut ranked: Vec<u32> = (0..n as u32).collect();
    for t in 0..m {
        let p = &probs[t * n..(t + 1) * n];
        ranked.sort_by(|&a, &b| {
            p[b as usize]
                .partial_cmp(&p[a as usize])
                .expect("route_ref: finite probs sorted")
                .then(a.cmp(&b))
        });
        for (s, &e) in ranked.iter().take(k).enumerate() {
            ids[t * k + s] = e as i32;
            weights[t * k + s] = p[e as usize] * scale;
        }
    }
    Ok((probs, ids, weights))
}

/// An F32 tensor from the model file as f32 (norm gains), length-checked
/// (gate_p4's local `f32_tensor`).
pub fn f32_tensor(gguf: &Gguf, name: &str, want: usize) -> Result<Vec<f32>, GateError> {
    let (t, b) = tensor_bytes(gguf, name)?;
    if t.ty != GgmlType::F32 || b.len() != want * 4 {
        return Err(format!(
            "f32_tensor: {name} is {:?} with {} bytes, want F32 x {want}",
            t.ty,
            b.len()
        )
        .into());
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// The f16 cache view's first `rows` rows as f16 bits: the dump widened
/// the halves to f32 exactly, so rounding back recovers ik's own bits
/// (gate_p5's local `widened_f16_bits`). The rounding is
/// `model::attn::f32_to_f16_bits` itself — the CPU oracle's own — so a
/// second transcription here cannot drift.
pub fn widened_f16_bits(row: &RefRow, rows: usize) -> Result<Vec<u16>, GateError> {
    use model::attn::f32_to_f16_bits;

    let path = ref_dir().join(row.file_name());
    let raw = std::fs::read(&path)
        .map_err(|e| format!("widened_f16_bits: cannot read {}: {e}", path.display()))?;
    if raw.len() as u64 != row.bytes {
        return Err(format!(
            "widened_f16_bits: {} is {} bytes, manifest says {}",
            path.display(),
            raw.len(),
            row.bytes
        )
        .into());
    }
    let width = row.ne[0] as usize;
    let count = row.count() as usize;
    if row.bytes != 4 * count as u64 {
        return Err(format!(
            "widened_f16_bits: {} is {} bytes for {count} widened values",
            path.display(),
            raw.len()
        )
        .into());
    }
    if rows * width > count {
        return Err(format!(
            "widened_f16_bits: {} holds {count} values, asked for {rows} rows of {width}",
            path.display()
        )
        .into());
    }
    Ok(raw
        .chunks_exact(4)
        .take(rows * width)
        .map(|c| f32_to_f16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect())
}

/// The topk row's ids from its LOGICAL twin: `ffn_moe_topk-L` is an i32
/// VIEW of the argsort output whose plain file is the flat parent read
/// (token 0's ranking only), so the ids of every token live in the
/// `.logical.f32` twin, each cast to f32 by the dumper. Accepted only when
/// the row carries a twin and every value is integral in `0..64`, which
/// rules out any stride or cast mix-up; the manifest's element-sum column
/// describes the plain file and is not checked here.
pub fn topk_ids_logical(row: &RefRow) -> Result<Vec<i32>, GateError> {
    if row.logical != Some(1) {
        return Err(format!(
            "topk_ids_logical: {} has no logical twin — the ids need a v2 dump set",
            row.name
        )
        .into());
    }
    let path = ref_dir().join(format!("{}.{}.logical.f32", row.name, row.occurrence));
    let raw = std::fs::read(&path)
        .map_err(|e| format!("topk_ids_logical: cannot read {}: {e}", path.display()))?;
    let expect = 4_u64
        .checked_mul(row.count())
        .ok_or("topk_ids_logical: topk element count overflows")?;
    if raw.len() as u64 != expect {
        return Err(format!(
            "topk_ids_logical: {} is {} bytes, want {expect} (4*count)",
            path.display(),
            raw.len()
        )
        .into());
    }
    raw.chunks_exact(4)
        .map(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if v.fract() == 0.0 && (0.0..64.0).contains(&v) {
                Ok(v as i32)
            } else {
                Err(format!(
                    "topk_ids_logical: {} holds non-integral id {v} — not ids cast to f32",
                    path.display()
                )
                .into())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::max_rel_err;

    /// `f32::max` drops NaN, so a fold alone scores an all-NaN output as 0.
    #[test]
    fn max_rel_err_rejects_nan_output() {
        let y_ref = [1.0f32, -2.0, 3.0];
        assert!(max_rel_err(&[f32::NAN; 3], &y_ref).is_err());
        assert!(max_rel_err(&[1.0, f32::NAN, 3.0], &y_ref).is_err());
        assert!(max_rel_err(&[1.0, f32::INFINITY, 3.0], &y_ref).is_err());
        assert!(max_rel_err(&[1.0, -2.0, 3.0], &[1.0, f32::NAN, 3.0]).is_err());
        assert_eq!(max_rel_err(&[1.0, -2.0, 3.0], &y_ref).unwrap(), 0.0);
    }
}
