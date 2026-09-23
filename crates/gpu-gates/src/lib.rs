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
pub mod oracle;
pub mod prompts;
pub mod ptx;

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, TensorInfo};
use std::path::{Path, PathBuf};

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

/// The error a gate returns when its `ok` accumulator came out false. Every
/// failing check has printed its own line by then; this ends the run through
/// [`exit_with`]. One spelling for every gate, as [`verdict`] is for a check.
pub fn checks_failed() -> GateError {
    "FAILED: one or more checks above did not pass".into()
}

/// The model file every gate opens: `$BLOOMERY_REF_MODEL`, with no default
/// here. The file is a property of the model profile
/// (`tools/ref/models/<architecture>.sh`); `tools/box.sh` exports it into
/// every box command, the same value the runners and the C++ harnesses
/// read. An empty value counts as unset, as in the profile. `open_model` and
/// any gate that prints the path read it here, so the printed name is the
/// file that was opened.
pub fn ref_model_path() -> Result<PathBuf, GateError> {
    match std::env::var_os("BLOOMERY_REF_MODEL") {
        Some(p) if !p.is_empty() => Ok(PathBuf::from(p)),
        _ => Err(
            "BLOOMERY_REF_MODEL unset — run through tools/box.sh or the just recipes, \
                  which export it from the model profile"
                .into(),
        ),
    }
}

/// The data directory on the box (reference dumps, oracle binaries):
/// `$BLOOMERY_DATA`, else the workstation default `tools/box.sh` also sets.
pub fn data_dir() -> PathBuf {
    std::env::var("BLOOMERY_DATA")
        .map_or_else(|_| PathBuf::from("/root/bloomery-data"), PathBuf::from)
}

pub fn open_model() -> Result<Gguf, GateError> {
    Ok(Gguf::open(ref_model_path()?)?)
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

/// Host transcription of the device q8_1 activation quantizer, for `m`
/// columns of `k` values: per 128-value block, d = amax/127 and q =
/// round(x/d) clamped to ±127, returned as the reconstructed q·d values. A
/// K-quant gemv's reference dot runs on these, so the activation
/// quantization noise it shares with the kernel cancels and [`KERNEL_BAND`]
/// measures the gemv arithmetic alone.
pub fn q8_1_dequant(x: &[f32], k: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * m];
    for c in 0..m {
        for b in 0..k / 128 {
            let blk = &x[c * k + b * 128..c * k + b * 128 + 128];
            let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (i, &v) in blk.iter().enumerate() {
                let q = (v / d).round().clamp(-127.0, 127.0);
                out[c * k + b * 128 + i] = q * d;
            }
        }
    }
    out
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

/// [`tensor_bytes`] of a tensor proven to be the one the gate was written
/// for: type `ty` and, when `dims` is given, exactly those dims (ggml order,
/// dims[0] = K). The error names the tensor, what the file holds and what
/// was wanted.
pub fn tensor_bytes_as<'a>(
    gguf: &'a Gguf,
    name: &str,
    ty: GgmlType,
    dims: Option<&[u64]>,
) -> Result<(&'a TensorInfo, &'a [u8]), GateError> {
    let (t, b) = tensor_bytes(gguf, name)?;
    if t.ty != ty || dims.is_some_and(|d| t.dims.as_slice() != d) {
        let want_dims = dims.map_or_else(String::new, |d| format!(" {d:?}"));
        return Err(format!(
            "tensor_bytes_as: {name} is {:?} {:?}, want {ty:?}{want_dims}",
            t.ty, t.dims
        )
        .into());
    }
    Ok((t, b))
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

/// Which kind of manifest row a file belongs to: a graph node (`tensor`) or
/// a graph leaf the host fills before the graph runs (`input` — token ids,
/// positions, masks, engram row ids). The two are separate occurrence
/// namespaces, and an input's files carry `.input` in their names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowKind {
    Tensor,
    Input,
}

impl RowKind {
    /// The manifest's spelling: `tensor` or `input`.
    pub fn as_str(self) -> &'static str {
        match self {
            RowKind::Tensor => "tensor",
            RowKind::Input => "input",
        }
    }
}

/// The element order of a set file: `Flat` is the tensor's memory read
/// contiguously from its data pointer (the plain file), `Logical` its
/// elements in ggml index order through its own strides (the twin every
/// view or non-contiguous row also gets).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    Flat,
    Logical,
}

impl Layout {
    /// The manifest's spelling: `flat` or `logical`.
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Flat => "flat",
            Layout::Logical => "logical",
        }
    }
}

/// What a set file holds per element: the f32 conversion every row has, or
/// an integer twin's lossless width (i8, i16 and i32 widen to `I32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileElem {
    F32,
    I32,
    I64,
}

impl FileElem {
    /// The file-name extension, also the manifest's spelling of a twin.
    pub fn ext(self) -> &'static str {
        match self {
            FileElem::F32 => "f32",
            FileElem::I32 => "i32",
            FileElem::I64 => "i64",
        }
    }

    /// Bytes per element.
    pub fn width(self) -> u64 {
        match self {
            FileElem::F32 | FileElem::I32 => 4,
            FileElem::I64 => 8,
        }
    }
}

/// A manifest name as the stem of its files: every `/`, `\` and space
/// becomes `_`, which keeps the name one path element. This is `safe_name`
/// in `tools/ref/dump_ref.cpp`. The manifest keeps the raw name, so a file
/// name formed from a row goes through [`dump_file_name`], which calls this.
pub fn dump_stem(name: &str) -> String {
    name.replace(['/', '\\', ' '], "_")
}

/// The name the dumper gives a set file:
/// `<stem>.<occurrence>[.input][.logical].<ext>` — `.input` on a graph
/// input's files, `.logical` on a logical twin, `ext` from `elem` (`f32`, or
/// an integer twin's `i32`/`i64`), and the stem from [`dump_stem`]. Every set
/// file name this crate reads is formed here.
pub fn dump_file_name(
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
    elem: FileElem,
) -> String {
    let input = match of {
        RowKind::Tensor => "",
        RowKind::Input => ".input",
    };
    let logical = match layout {
        Layout::Flat => "",
        Layout::Logical => ".logical",
    };
    format!(
        "{}.{occurrence}{input}{logical}.{}",
        dump_stem(name),
        elem.ext()
    )
}

/// One `tensor` or `input` row of a dump set's MANIFEST.tsv. `ne` is
/// ne0..ne3 as written: ne0 is the contiguous dimension (values of an
/// activation, rows of a MUL_MAT output); `sum` is the dumper's element
/// sum, usable to prove a VIEW row carries the same bytes as its base.
/// The four trailing fields are the v2 columns (`contig`, `logical`,
/// `src0`, `src1`); `None` on a set written before they existed.
#[derive(Debug, Clone)]
pub struct RefRow {
    pub kind: RowKind,
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

    /// The plain file's name ([`dump_file_name`]): the flat f32 read.
    pub fn file_name(&self) -> String {
        dump_file_name(
            &self.name,
            self.occurrence,
            self.kind,
            Layout::Flat,
            FileElem::F32,
        )
    }

    /// The logical twin's file name, a file that exists when `logical` is
    /// `Some(1)`.
    pub fn logical_file_name(&self) -> String {
        dump_file_name(
            &self.name,
            self.occurrence,
            self.kind,
            Layout::Logical,
            FileElem::F32,
        )
    }

    /// Prove this row's type, dims and op — the chain check every consumer
    /// runs before trusting a tensor (`op` = what produced it). `op = "in"`
    /// only checks type and dims (an input's op varies). `what` names the
    /// call site in the error.
    pub fn expect(&self, what: &str, ty: &str, ne: [u64; 4], op: &str) -> Result<(), GateError> {
        if self.ty != ty || self.ne != ne || (op != "in" && self.op != op) {
            return Err(format!(
                "expect: {what}: {}/{} is {} {:?} op {}, want {ty} {ne:?} op {op}",
                self.name, self.occurrence, self.ty, self.ne, self.op
            )
            .into());
        }
        Ok(())
    }
}

/// One `int` row: the lossless twin of an integer tensor's f32 file of the
/// same layout, raw little-endian in the same element order. `sum` is the
/// exact element sum (wrapping 64-bit, printed signed), `absmax` the largest
/// magnitude, `file` the name the dumper wrote.
#[derive(Debug, Clone)]
pub struct IntRow {
    pub name: String,
    pub occurrence: u32,
    /// The kind of the row this file twins.
    pub of: RowKind,
    /// The twinned tensor's ggml type (`i8`, `i16`, `i32` or `i64`).
    pub ty: String,
    /// `I32` (i8, i16 and i32 widened) or `I64`; the parser admits no `F32`.
    pub twin: FileElem,
    pub layout: Layout,
    pub count: u64,
    pub bytes: u64,
    pub sum: i64,
    pub absmax: u64,
    pub file: String,
}

/// `int <of> <name>/<occurrence> <layout> (<file>)` — how errors name a row.
impl std::fmt::Display for IntRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "int {} {}/{} {} ({})",
            self.of.as_str(),
            self.name,
            self.occurrence,
            self.layout.as_str(),
            self.file
        )
    }
}

/// A dump set's whole MANIFEST.tsv: the header lines the harness checks and
/// every row the dumper writes, by kind. A gate reads it once per set and
/// takes rows out of it.
#[derive(Debug, Clone)]
pub struct RefManifest {
    /// The set's directory; every file a row names resolves under it.
    pub dir: PathBuf,
    /// `# arch`, the model's `general.architecture`. `None` on a set dumped
    /// before the dumper wrote the line.
    pub arch: Option<String>,
    /// `# build`, the ik build that wrote the set.
    pub build: Option<String>,
    /// `# complete <written> <skipped>`: the node rows written and skipped.
    /// `None` when the trailer is missing — the dump that wrote the set died.
    pub complete: Option<(u64, u64)>,
    /// `tensor` rows (graph nodes), in file order.
    pub tensors: Vec<RefRow>,
    /// `input` rows (graph leaves), in file order.
    pub inputs: Vec<RefRow>,
    /// `int` rows, one per integer twin file, in file order.
    pub ints: Vec<IntRow>,
    /// `skip` rows: nodes written to no file (quantized, unhandled type,
    /// unallocated).
    pub skipped_nodes: u64,
    /// `skip-input` rows: the same for graph inputs.
    pub skipped_inputs: u64,
}

impl RefManifest {
    /// Parse `dir/MANIFEST.tsv`. Header lines start with `#`; data rows are
    /// tab-separated, the kind first:
    /// `tensor`/`input name occurrence type ne0 ne1 ne2 ne3 bytes sum op`
    /// with four optional trailing fields `contig logical src0 src1` (sets
    /// written before the v2 columns carry the 11-field width and parse to
    /// `None`s), `int name occurrence of type twin layout count bytes sum
    /// absmax file`, and `skip`/`skip-input name occurrence type reason`.
    /// Tensor names may contain spaces, so fields are split on tabs only. A
    /// row of any other kind or width is an error naming its line, and so is
    /// a manifest with no `tensor` row.
    pub fn read(dir: &Path) -> Result<RefManifest, GateError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("ref_manifest: cannot read {}: {e}", path.display()))?;
        let mut man = RefManifest {
            dir: dir.to_path_buf(),
            arch: None,
            build: None,
            complete: None,
            tensors: Vec::new(),
            inputs: Vec::new(),
            ints: Vec::new(),
            skipped_nodes: 0,
            skipped_inputs: 0,
        };
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            if let Some(v) = line.strip_prefix("# arch\t") {
                man.arch = Some(v.to_string());
            } else if let Some(v) = line.strip_prefix("# build\t") {
                man.build = Some(v.to_string());
            } else if let Some(v) = line.strip_prefix("# complete\t") {
                let (w, s) = v
                    .split_once('\t')
                    .ok_or_else(|| format!("ref_manifest: {at}: trailer {line:?}"))?;
                man.complete = Some((parse_u64(w, at)?, parse_u64(s, at)?));
            } else if line.starts_with('#') || line.is_empty() {
                continue;
            } else {
                let f: Vec<&str> = line.split('\t').collect();
                match f[0] {
                    "tensor" => man.tensors.push(parse_ref_row(RowKind::Tensor, &f, at)?),
                    "input" => man.inputs.push(parse_ref_row(RowKind::Input, &f, at)?),
                    "int" => man.ints.push(parse_int_row(&f, at)?),
                    "skip" | "skip-input" if f.len() != 5 => {
                        return Err(format!(
                            "ref_manifest: {at}: {} row has {} fields, want 5",
                            f[0],
                            f.len()
                        )
                        .into());
                    }
                    "skip" => man.skipped_nodes += 1,
                    "skip-input" => man.skipped_inputs += 1,
                    k => return Err(format!("ref_manifest: {at}: unknown row kind {k:?}").into()),
                }
            }
        }
        if man.tensors.is_empty() {
            return Err(format!("ref_manifest: no tensor rows in {}", path.display()).into());
        }
        Ok(man)
    }
}

/// A manifest line's position, `<path>:<line>`, formatted only into an error.
#[derive(Clone, Copy)]
struct At<'a>(&'a Path, usize);

impl std::fmt::Display for At<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.0.display(), self.1)
    }
}

fn parse_u64(s: &str, at: At) -> Result<u64, GateError> {
    s.parse::<u64>()
        .map_err(|e| format!("ref_manifest: {at}: {s:?}: {e}").into())
}

fn parse_narrow<T: TryFrom<u64>>(s: &str, at: At) -> Result<T, GateError> {
    T::try_from(parse_u64(s, at)?)
        .map_err(|_| format!("ref_manifest: {at}: {s:?} is out of range").into())
}

/// A `tensor` or `input` row, 11 or 15 fields (the kind is `f[0]`).
fn parse_ref_row(kind: RowKind, f: &[&str], at: At) -> Result<RefRow, GateError> {
    if f.len() != 11 && f.len() != 15 {
        return Err(format!(
            "ref_manifest: {at}: {} row has {} fields, want 11 or 15",
            kind.as_str(),
            f.len()
        )
        .into());
    }
    let ne = |i: usize| parse_u64(f[i], at);
    Ok(RefRow {
        kind,
        name: f[1].to_string(),
        occurrence: parse_narrow(f[2], at)?,
        ty: f[3].to_string(),
        ne: [ne(4)?, ne(5)?, ne(6)?, ne(7)?],
        bytes: parse_u64(f[8], at)?,
        sum: f[9]
            .parse::<f64>()
            .map_err(|e| format!("ref_manifest: {at}: {:?}: {e}", f[9]))?,
        op: f[10].to_string(),
        contig: f.get(11).map(|s| parse_narrow(s, at)).transpose()?,
        logical: f.get(12).map(|s| parse_narrow(s, at)).transpose()?,
        src0: f.get(13).map(|s| s.to_string()),
        src1: f.get(14).map(|s| s.to_string()),
    })
}

/// An `int` row, 12 fields.
fn parse_int_row(f: &[&str], at: At) -> Result<IntRow, GateError> {
    if f.len() != 12 {
        return Err(format!(
            "ref_manifest: {at}: int row has {} fields, want 12",
            f.len()
        )
        .into());
    }
    let bad = |col: &str, v: &str| -> GateError {
        format!("ref_manifest: {at}: int row {col} {v:?}").into()
    };
    let of = match f[3] {
        "tensor" => RowKind::Tensor,
        "input" => RowKind::Input,
        v => return Err(bad("of", v)),
    };
    let twin = match f[5] {
        "i32" => FileElem::I32,
        "i64" => FileElem::I64,
        v => return Err(bad("twin", v)),
    };
    let layout = match f[6] {
        "flat" => Layout::Flat,
        "logical" => Layout::Logical,
        v => return Err(bad("layout", v)),
    };
    Ok(IntRow {
        name: f[1].to_string(),
        occurrence: parse_narrow(f[2], at)?,
        of,
        ty: f[4].to_string(),
        twin,
        layout,
        count: parse_u64(f[7], at)?,
        bytes: parse_u64(f[8], at)?,
        sum: f[9]
            .parse::<i64>()
            .map_err(|e| format!("ref_manifest: {at}: {:?}: {e}", f[9]))?,
        absmax: parse_u64(f[10], at)?,
        file: f[11].to_string(),
    })
}

/// Directory of the ik CUDA oracle dump (docs/gpu-design.md decision 3):
/// `$BLOOMERY_REF_CUDA` if set (an absolute path), else the set named by
/// `$BLOOMERY_REF_SET`, else the deepseek2 table's CUDA set
/// ([`oracle::deepseek2::CUDA_SET`]) under [`data_dir`]. Read-only for every
/// caller. `BLOOMERY_REF_SET` is how a gate points itself at the pre-v2
/// `ref_cuda` (plain files only, no logical twins) or the CPU `ref` without
/// a code change.
pub fn ref_dir() -> PathBuf {
    if let Ok(p) = std::env::var("BLOOMERY_REF_CUDA") {
        return PathBuf::from(p);
    }
    if let Ok(s) = std::env::var("BLOOMERY_REF_SET") {
        return ref_dir_named(&s);
    }
    data_dir().join(oracle::deepseek2::CUDA_SET)
}

/// A named dump set's directory: an absolute `set` is the directory
/// itself, anything else is `<$BLOOMERY_DATA>/<set>` (`ref_cuda_v2`, the
/// CPU `ref`, ...). Unlike `ref_dir` this consults no `BLOOMERY_REF_*`
/// variable — only the data directory.
pub fn ref_dir_named(set: &str) -> PathBuf {
    let p = PathBuf::from(set);
    if p.is_absolute() {
        return p;
    }
    data_dir().join(set)
}

/// The `tensor` rows of the set at `dir` ([`RefManifest::read`]) — the
/// rows the node readers below look names up in.
pub fn ref_manifest_in(dir: &std::path::Path) -> Result<Vec<RefRow>, GateError> {
    Ok(RefManifest::read(dir)?.tensors)
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
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
        let path = dir.join(row.logical_file_name());
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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

/// The `int` row twinning `(name, occurrence)` of kind `of` in `layout`.
pub fn find_int_row<'a>(
    man: &'a RefManifest,
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
) -> Result<&'a IntRow, GateError> {
    man.ints
        .iter()
        .find(|r| r.name == name && r.occurrence == occurrence && r.of == of && r.layout == layout)
        .ok_or_else(|| {
            format!(
                "find_int_row: no {} integer twin of {} {name}/{occurrence} in {}",
                layout.as_str(),
                of.as_str(),
                man.dir.join("MANIFEST.tsv").display()
            )
            .into()
        })
}

/// The exact integers of int row `row` of the set at `dir`, widened to
/// i64, read from its `file` after proving that file is the one the row
/// describes: `file` is the name [`dump_file_name`] gives the row (the
/// dumper and this crate share one naming rule), `bytes` is `count`
/// elements of the twin's width and the file's length, and the `sum`
/// (wrapping) and `absmax` recomputed from the values equal the row's. A
/// wrong file, offset or width fails here, naming the row.
pub fn ref_ints_of_in(dir: &Path, row: &IntRow) -> Result<Vec<i64>, GateError> {
    let want = dump_file_name(&row.name, row.occurrence, row.of, row.layout, row.twin);
    if row.file != want {
        return Err(format!("ref_ints: {row}: the dumper names this row's file {want}").into());
    }
    let expect = row
        .count
        .checked_mul(row.twin.width())
        .ok_or_else(|| format!("ref_ints: {row}: element count overflows"))?;
    if row.bytes != expect {
        return Err(format!(
            "ref_ints: {row}: bytes {} != count {} x {}",
            row.bytes,
            row.count,
            row.twin.width()
        )
        .into());
    }
    let path = dir.join(&row.file);
    let raw = std::fs::read(&path)
        .map_err(|e| format!("ref_ints: {row}: cannot read {}: {e}", path.display()))?;
    if raw.len() as u64 != row.bytes {
        return Err(format!(
            "ref_ints: {row}: {} is {} bytes, the row says {}",
            path.display(),
            raw.len(),
            row.bytes
        )
        .into());
    }
    let vals: Vec<i64> = match row.twin {
        FileElem::I32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i64::from(i32::from_le_bytes(*c)))
            .collect(),
        FileElem::I64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect(),
        FileElem::F32 => return Err(format!("ref_ints: {row}: not an integer twin").into()),
    };
    let sum = vals
        .iter()
        .fold(0u64, |a, &v| a.wrapping_add(v.cast_unsigned()))
        .cast_signed();
    let absmax = vals.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    if sum != row.sum || absmax != row.absmax {
        return Err(format!(
            "ref_ints: {row}: the file sums to {sum} with absmax {absmax}, the row says {} and {}",
            row.sum, row.absmax
        )
        .into());
    }
    Ok(vals)
}

/// The exact integers of `(name, occurrence)`'s twin of kind `of` in
/// `layout`: [`find_int_row`] in `man`, then [`ref_ints_of_in`].
pub fn ref_ints(
    man: &RefManifest,
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
) -> Result<Vec<i64>, GateError> {
    ref_ints_of_in(&man.dir, find_int_row(man, name, occurrence, of, layout)?)
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
        ranked.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
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
    let (_, b) = tensor_bytes_as(gguf, name, GgmlType::F32, None)?;
    if b.len() != want * 4 {
        return Err(format!(
            "f32_tensor: {name} is F32 with {} bytes, want F32 x {want}",
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
/// `gguf::quant::f32_to_f16_bits` itself — the CPU oracle's own — so a
/// second transcription here cannot drift.
pub fn widened_f16_bits(row: &RefRow, rows: usize) -> Result<Vec<u16>, GateError> {
    use gguf::quant::f32_to_f16_bits;

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
        .as_chunks::<4>()
        .0
        .iter()
        .take(rows * width)
        .map(|c| f32_to_f16_bits(f32::from_le_bytes(*c)))
        .collect())
}

/// The topk row's ids from its LOGICAL twin: `ffn_moe_topk-L` is an i32
/// VIEW of the argsort output whose plain file is the flat parent read
/// (token 0's ranking only), so the ids of every token live in the logical
/// twin, read exact from its `.logical.i32` integer twin ([`ref_ints`]).
/// For a set whose manifest has no `int` rows at all (dumped before integer
/// twins existed) it reads the `.logical.f32` twin, each id cast to f32 by
/// the dumper. Accepted only when the row carries a logical twin and every
/// id is an integer in `0..64`, which rules out any stride or cast mix-up;
/// the manifest's element-sum column describes the plain file and is not
/// checked here. `row` is a row of `ref_dir()`'s manifest, read here.
pub fn topk_ids_logical(row: &RefRow) -> Result<Vec<i32>, GateError> {
    topk_ids_logical_in(&RefManifest::read(&ref_dir())?, row)
}

/// [`topk_ids_logical`] of a row of `man`, a manifest already read.
pub fn topk_ids_logical_in(man: &RefManifest, row: &RefRow) -> Result<Vec<i32>, GateError> {
    if row.logical != Some(1) {
        return Err(format!(
            "topk_ids_logical: {} has no logical twin — the ids need a v2 dump set",
            row.name
        )
        .into());
    }
    if !man.ints.is_empty() {
        let ids = ref_ints(man, &row.name, row.occurrence, row.kind, Layout::Logical)?;
        if ids.len() as u64 != row.count() {
            return Err(format!(
                "topk_ids_logical: {}/{} has {} elements, its integer twin {}",
                row.name,
                row.occurrence,
                row.count(),
                ids.len()
            )
            .into());
        }
        return ids
            .iter()
            .map(|&v| match i32::try_from(v) {
                Ok(id) if (0..64).contains(&id) => Ok(id),
                _ => Err(format!(
                    "topk_ids_logical: {}/{} holds id {v}, outside 0..64",
                    row.name, row.occurrence
                )
                .into()),
            })
            .collect();
    }
    let path = man.dir.join(row.logical_file_name());
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
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = f32::from_le_bytes(*c);
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
    use super::{GateError, RefManifest, max_rel_err, topk_ids_logical_in};

    /// The top-k reader takes the exact `.logical.i32` twin when the
    /// manifest has `int` rows and the `.logical.f32` twin when it has none
    /// — the f32 file below disagrees in one id, so the result shows which
    /// file was read — and a manifest with twins but none for the row is an
    /// error. The name has a space, so the files resolve only through the
    /// dumper's `safe_name` rule.
    #[test]
    fn topk_ids_take_the_integer_twin_when_the_set_has_twins() -> Result<(), GateError> {
        let dir = std::env::temp_dir().join(format!("bloomery-topk-twin-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let ids: [i32; 4] = [3, 63, 0, 17];
        let cast: [f32; 4] = [3.0, 63.0, 0.0, 16.0];
        let stem = "ffn_moe_topk-1_(view).0.logical";
        std::fs::write(
            dir.join(format!("{stem}.i32")),
            ids.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        std::fs::write(
            dir.join(format!("{stem}.f32")),
            cast.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let tensor =
            "tensor\tffn_moe_topk-1 (view)\t0\ti32\t2\t2\t1\t1\t16\t0.000000\tVIEW\t0\t1\t-\t-";
        let int = format!(
            "int\tffn_moe_topk-1 (view)\t0\ttensor\ti32\ti32\tlogical\t4\t16\t83\t63\t{stem}.i32"
        );
        let other = int.replacen("\t0\t", "\t1\t", 1);
        let read = |rows: &[&str]| -> Result<Vec<i32>, GateError> {
            std::fs::write(dir.join("MANIFEST.tsv"), rows.join("\n") + "\n")?;
            let man = RefManifest::read(&dir)?;
            topk_ids_logical_in(&man, &man.tensors[0])
        };
        assert_eq!(read(&[tensor, &int])?, ids);
        assert_eq!(read(&[tensor])?, [3, 63, 0, 16]);
        assert!(read(&[tensor, &other]).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

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
