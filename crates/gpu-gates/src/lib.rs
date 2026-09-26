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

pub mod act_rule;
#[cfg(feature = "deepseek41")]
pub mod bind;
pub mod block;
pub mod draft;
pub mod ds41_meta;
pub mod engine;
#[cfg(feature = "deepseek41")]
pub mod generate;
pub mod hc_host;
pub mod ik_norm;
pub mod ik_q8_2;
pub mod kld;
#[cfg(feature = "gpu")]
pub mod nodes;
pub mod oracle;
pub mod prompts;
pub mod ptx;
pub mod qwen3moe;
pub mod rounding;

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, Split, TensorInfo};
use model::arch::Arch;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Kernel gate band against the quantized-input reference.
/// PIN(2026-09-21): tightened 1e-2 -> 1e-5. Measured worst on the landed
/// kernels: 1.149e-7 (gate_p1, 12 shapes), 1.375e-7 (gate_p2); the f32
/// gemvs already gate at 1e-5 (worst 1.9e-7). 1e-2 was sized for the
/// raw-activation comparison this band no longer makes, and would pass a
/// sub-block scale defect landing near 1e-3 on a shape with no hash pin.
pub const KERNEL_BAND: f32 = 1e-5;

/// The V4.1 greedy rule, shared by the step gate's `--greedy` and the long
/// gate's free arm: at the first generated id that differs from ik's, our
/// top-1 margin must be below this — a near tie [derived, plan.md: ≈ 3 σ_rel].
pub const GREEDY_MARGIN: f32 = 1.5;

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

/// The model file ([`ref_model_path`]) as a split, proven to be of
/// architecture `arch` ([`expect_arch`]); `recipe` is the `just` recipe the
/// error for any other file names.
pub fn open_split(arch: Arch, recipe: &str) -> Result<Split, GateError> {
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    expect_arch(&split, arch, recipe)?;
    Ok(split)
}

/// The file's end-of-sequence token, `tokenizer.ggml.eos_token_id` (stored
/// as a u32 or a non-negative i32); a file that names none is an error.
pub fn eos_token_id(split: &Split) -> Result<u32, GateError> {
    split
        .value("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            gguf::Value::U32(e) => Some(*e),
            gguf::Value::I32(e) => u32::try_from(*e).ok(),
            _ => None,
        })
        .ok_or_else(|| "the file names no EOS token".into())
}

/// Err unless `split` is a model file of architecture `arch`. The error names
/// `recipe`, the `just` recipe that picks `arch`'s model profile.
pub fn expect_arch(split: &Split, arch: Arch, recipe: &str) -> Result<(), GateError> {
    let want = arch.name();
    if split.architecture() != Some(want) {
        return Err(format!(
            "the model file is {:?}, want {want} — run through `just {recipe}`, \
             which picks the {want} profile",
            split.architecture()
        )
        .into());
    }
    Ok(())
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

/// A quiet f16 NaN, as the gates write it into f16 rows, rings and scales a
/// kernel must not read (or must read and raise the fault word on).
pub const NAN_F16: u16 = 0x7e00;

/// Byte offset of the f16 super-block scale `d` inside one super-block of
/// the K-quant `ty`, as ggml lays the block out (`block_q3_K` ends with it,
/// `block_q4_K` opens with it, `block_q6_K` ends with it). Any other type is
/// refused by name: a gate that poisons a scale must not poison a guess.
pub fn kquant_d_at(ty: GgmlType) -> Result<usize, GateError> {
    match ty {
        GgmlType::Q3_K => Ok(108),
        GgmlType::Q4_K => Ok(0),
        GgmlType::Q6_K => Ok(208),
        other => Err(format!(
            "kquant_d_at: {other:?} is not a K-quant this helper knows the scale of \
             (Q3_K, Q4_K, Q6_K)"
        )
        .into()),
    }
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

/// A dump set's whole MANIFEST.tsv: its header lines and every row the
/// dumper writes, by kind, with an index over the rows. A gate reads it once
/// per set and takes rows out of it.
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
    /// The other header lines.
    pub header: RefHeader,
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
    index: RowIndex,
}

/// The header lines of a set that [`RefManifest`] does not hold as its own
/// fields (`# arch`, `# build`, the `# complete` trailer). A line the set
/// does not carry leaves its field `None`.
#[derive(Debug, Clone, Default)]
pub struct RefHeader {
    /// `# tokens`: the ids of the whole sequence the set was dumped for.
    pub tokens: Option<Vec<u32>>,
    /// `# tokens_file`: the file the dumper read the tokens from.
    pub tokens_file: Option<String>,
    /// `# tokens_file_sha256`: that file's digest.
    pub tokens_file_sha256: Option<String>,
    /// `# tokens_count`: how many of the file's tokens the dumper took.
    pub tokens_count: Option<u64>,
    /// `# model_file`: the name of the model file the set was dumped from.
    pub model_file: Option<String>,
    /// `# flags`: the dumper's command line.
    pub flags: Option<String>,
    /// `-c` in `# flags`: the context the dumper ran at.
    pub ctx: Option<u64>,
    /// `-t` in `# flags`: the threads the dumper ran with.
    pub threads: Option<u64>,
    /// `# prefill`: in a decode-step set, the tokens run before the step.
    pub prefill: Option<u32>,
    /// `# decode_pos`: in a decode-step set, the step's position.
    pub decode_pos: Option<u32>,
    /// `# state_inputs`: in a decode-step set, the dumper's note that the
    /// persistent leaves the step reads (caches, compressor states) are
    /// input rows, written at their first reader.
    pub state_inputs: Option<String>,
    /// `# fused_idx_topk`: whether the indexer's scores and top-k ran as one
    /// op (`1`) or as nodes of their own (`0`).
    pub fused_idx_topk: Option<bool>,
    /// Every other `#` line, verbatim and in file order: the dumper's title,
    /// `# model`, the column lines, and any line this parser does not know.
    pub other: Vec<String>,
}

impl RefHeader {
    /// `# model`: the path of the model file the set was dumped from, one of
    /// the lines kept verbatim in [`RefHeader::other`].
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.other.iter().find_map(|l| l.strip_prefix("# model\t"))
    }
}

/// Where each row sits in its vector, keyed the way readers look rows up:
/// by name, then `(kind, occurrence)`, and for an `int` row also its layout.
/// Built once by [`RefManifest::read`] over the rows it parsed; of a
/// repeated key it keeps the first row, the one a scan in file order finds.
/// A lookup checks that the row it lands on carries the key, so an answer
/// always comes from the rows themselves.
#[derive(Debug, Clone, Default)]
struct RowIndex {
    rows: HashMap<String, Vec<(RowKind, u32, usize)>>,
    ints: HashMap<String, Vec<(RowKind, u32, Layout, usize)>>,
    /// Tensor and input rows whose key an earlier row already carries.
    duplicates: usize,
    /// Per input row, the position in `tensors` of the first tensor row the
    /// file holds after it; `None` when no tensor row follows.
    touchers: Vec<Option<usize>>,
}

impl RowIndex {
    fn over(
        tensors: &[RefRow],
        inputs: &[RefRow],
        ints: &[IntRow],
        touchers: Vec<Option<usize>>,
    ) -> RowIndex {
        let mut ix = RowIndex {
            touchers,
            ..RowIndex::default()
        };
        for (at, r) in tensors.iter().enumerate().chain(inputs.iter().enumerate()) {
            let keys = ix.rows.entry(r.name.clone()).or_default();
            if keys
                .iter()
                .any(|&(k, o, _)| k == r.kind && o == r.occurrence)
            {
                ix.duplicates += 1;
            } else {
                keys.push((r.kind, r.occurrence, at));
            }
        }
        for (at, r) in ints.iter().enumerate() {
            let keys = ix.ints.entry(r.name.clone()).or_default();
            if !keys
                .iter()
                .any(|&(of, o, l, _)| of == r.of && o == r.occurrence && l == r.layout)
            {
                keys.push((r.of, r.occurrence, r.layout, at));
            }
        }
        ix
    }
}

impl RefManifest {
    /// Parse `dir/MANIFEST.tsv`. Header lines start with `#` ([`RefHeader`]
    /// and this struct's own `arch`, `build`, `complete`); data rows are
    /// tab-separated, the kind first:
    /// `tensor`/`input name occurrence type ne0 ne1 ne2 ne3 bytes sum op`
    /// with four optional trailing fields `contig logical src0 src1` (sets
    /// written before the v2 columns carry the 11-field width and parse to
    /// `None`s), `int name occurrence of type twin layout count bytes sum
    /// absmax file`, and `skip`/`skip-input name occurrence type reason`.
    /// Tensor names may contain spaces, so fields are split on tabs only. A
    /// row of any other kind or width is an error naming its line, and so is
    /// a manifest with no `tensor` row. The rows are indexed here, once,
    /// with each input row's first toucher ([`first_touched_by`](Self::first_touched_by)).
    pub fn read(dir: &Path) -> Result<RefManifest, GateError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("ref_manifest: cannot read {}: {e}", path.display()))?;
        let mut man = RefManifest {
            dir: dir.to_path_buf(),
            arch: None,
            build: None,
            complete: None,
            header: RefHeader::default(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            ints: Vec::new(),
            skipped_nodes: 0,
            skipped_inputs: 0,
            index: RowIndex::default(),
        };
        // Per input row, the tensor row that follows it in the file; the
        // inputs no tensor row has followed yet.
        let (mut touchers, mut pending) = (Vec::new(), Vec::new());
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            if line.starts_with('#') {
                man.header_line(line, at)?;
            } else if !line.is_empty() {
                let f: Vec<&str> = line.split('\t').collect();
                match f[0] {
                    "tensor" => {
                        let row = parse_ref_row(RowKind::Tensor, &f, at)?;
                        for p in pending.drain(..) {
                            touchers[p] = Some(man.tensors.len());
                        }
                        man.tensors.push(row);
                    }
                    "input" => {
                        man.inputs.push(parse_ref_row(RowKind::Input, &f, at)?);
                        pending.push(touchers.len());
                        touchers.push(None);
                    }
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
        man.index = RowIndex::over(&man.tensors, &man.inputs, &man.ints, touchers);
        Ok(man)
    }

    /// One `#` line. A `# <key>\t<value>` line of a key this parser knows
    /// fills its field — a key given twice, or a value that does not parse,
    /// is an error naming the line — and every other line is kept verbatim
    /// in `header.other`.
    fn header_line(&mut self, line: &str, at: At) -> Result<(), GateError> {
        let Some((key, v)) = line.strip_prefix("# ").and_then(|h| h.split_once('\t')) else {
            self.header.other.push(line.to_string());
            return Ok(());
        };
        let bad = |e: &dyn std::fmt::Display| -> GateError {
            format!("ref_manifest: {at}: {line:?}: {e}").into()
        };
        let text = || v.to_string();
        let h = &mut self.header;
        match key {
            "arch" => once(&mut self.arch, text(), key, at),
            "build" => once(&mut self.build, text(), key, at),
            "complete" => {
                let (w, s) = v.split_once('\t').ok_or_else(|| bad(&"not two counts"))?;
                once(
                    &mut self.complete,
                    (parse_u64(w, at)?, parse_u64(s, at)?),
                    key,
                    at,
                )
            }
            "tokens" => {
                let ids = v
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<u32>, _>>()
                    .map_err(|e| bad(&e))?;
                once(&mut h.tokens, ids, key, at)
            }
            "tokens_file" => once(&mut h.tokens_file, text(), key, at),
            "tokens_file_sha256" => once(&mut h.tokens_file_sha256, text(), key, at),
            "tokens_count" => once(&mut h.tokens_count, parse_u64(v, at)?, key, at),
            "model_file" => once(&mut h.model_file, text(), key, at),
            "flags" => {
                h.ctx = flag_value(v, "-c").map_err(|e| bad(&e))?;
                h.threads = flag_value(v, "-t").map_err(|e| bad(&e))?;
                once(&mut h.flags, text(), key, at)
            }
            "prefill" => once(&mut h.prefill, parse_narrow(v, at)?, key, at),
            "decode_pos" => once(&mut h.decode_pos, parse_narrow(v, at)?, key, at),
            "state_inputs" => once(&mut h.state_inputs, text(), key, at),
            "fused_idx_topk" => {
                let fused = match v {
                    "0" => false,
                    "1" => true,
                    _ => return Err(bad(&"want 0 or 1")),
                };
                once(&mut h.fused_idx_topk, fused, key, at)
            }
            _ => {
                h.other.push(line.to_string());
                Ok(())
            }
        }
    }

    /// Where the row of kind `kind` keyed `name`/`occurrence` sits among the
    /// rows of its kind (`tensors` or `inputs`, file order), through the
    /// index — the row a scan of the file finds first — or `None` if the set
    /// has none.
    pub fn position(&self, kind: RowKind, name: &str, occurrence: u32) -> Option<usize> {
        let &(_, _, at) = self
            .index
            .rows
            .get(name)?
            .iter()
            .find(|&&(k, o, _)| k == kind && o == occurrence)?;
        self.rows_of(kind)
            .get(at)
            .filter(|r| r.name == name && r.occurrence == occurrence)
            .map(|_| at)
    }

    /// The row of kind `kind` keyed `name`/`occurrence`, through the index —
    /// the row a scan of the file finds first — or `None` if the set has
    /// none.
    pub fn find(&self, kind: RowKind, name: &str, occurrence: u32) -> Option<&RefRow> {
        let at = self.position(kind, name, occurrence)?;
        self.rows_of(kind).get(at)
    }

    /// The first row of kind `kind` named `name`, whatever its occurrence —
    /// the row a scan of the file finds first. The dumper counts a name's
    /// skipped nodes in its occurrences, so that row need not be occurrence 0.
    pub fn first_named(&self, kind: RowKind, name: &str) -> Option<&RefRow> {
        let at = self
            .index
            .rows
            .get(name)?
            .iter()
            .filter(|&&(k, ..)| k == kind)
            .map(|&(.., at)| at)
            .min()?;
        self.rows_of(kind).get(at).filter(|r| r.name == name)
    }

    /// The one row of kind `kind` whose name opens with `prefix`, whatever
    /// its occurrence: the row of a tensor the graph names after the last of
    /// several callers (`dsv4_raw_mask_padded-<layer>`), which a reader cannot
    /// name in advance. No such row, or more than one, is an error naming the
    /// rows found and the set.
    pub fn only_with_prefix(&self, kind: RowKind, prefix: &str) -> Result<&RefRow, GateError> {
        let found: Vec<&RefRow> = self
            .rows_of(kind)
            .iter()
            .filter(|r| r.name.starts_with(prefix))
            .collect();
        match found[..] {
            [r] => Ok(r),
            _ => Err(format!(
                "ref_manifest: want one {} row named {prefix}…, {} has {:?}",
                kind.as_str(),
                self.dir.join("MANIFEST.tsv").display(),
                found
                    .iter()
                    .map(|r| format!("{}/{}", r.name, r.occurrence))
                    .collect::<Vec<_>>()
            )
            .into()),
        }
    }

    /// The `tensor` row `name`/`occurrence` ([`find`](Self::find)); a set
    /// without it is an error naming the set.
    pub fn tensor(&self, name: &str, occurrence: u32) -> Result<&RefRow, GateError> {
        self.row(RowKind::Tensor, name, occurrence)
    }

    /// The `tensor` row `name`/`occurrence` with its position in `tensors`,
    /// where a walk back through the rows before it starts
    /// ([`last_before`](Self::last_before)); a set without it is an error
    /// naming the set.
    pub fn tensor_at(&self, name: &str, occurrence: u32) -> Result<(usize, &RefRow), GateError> {
        let at = self
            .position(RowKind::Tensor, name, occurrence)
            .ok_or_else(|| self.missing(RowKind::Tensor, name, occurrence))?;
        Ok((at, &self.tensors[at]))
    }

    /// The `input` row `name`/`occurrence` ([`find`](Self::find)); a set
    /// without it is an error naming the set.
    pub fn input(&self, name: &str, occurrence: u32) -> Result<&RefRow, GateError> {
        self.row(RowKind::Input, name, occurrence)
    }

    /// The node a reader at `tensors[at]` reads as `name`, one of its source
    /// columns: the last tensor row of that name before it — manifest order
    /// is execution order — with its position. An empty column is an error.
    pub fn last_before(
        &self,
        at: usize,
        name: Option<&str>,
    ) -> Result<(usize, &RefRow), GateError> {
        let name = name.ok_or("a reader row has no src column")?;
        self.tensors[..at]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.name == name)
            .ok_or_else(|| format!("no node {name:?} before manifest row {at}").into())
    }

    /// The input rows tensor row `at` touches first, in file order. The
    /// dumper writes a graph input right before the node that first touches
    /// it — reads it, or writes into it as a `SET_ROWS` destination, which no
    /// source column names — so an input's first toucher is the tensor row
    /// that follows it in the file.
    pub fn first_touched_by(&self, at: usize) -> Vec<&RefRow> {
        self.inputs
            .iter()
            .zip(&self.index.touchers)
            .filter(|&(_, &t)| t == Some(at))
            .map(|(r, _)| r)
            .collect()
    }

    fn rows_of(&self, kind: RowKind) -> &[RefRow] {
        match kind {
            RowKind::Tensor => &self.tensors,
            RowKind::Input => &self.inputs,
        }
    }

    fn row(&self, kind: RowKind, name: &str, occurrence: u32) -> Result<&RefRow, GateError> {
        self.find(kind, name, occurrence)
            .ok_or_else(|| self.missing(kind, name, occurrence))
    }

    /// The error a lookup of a row the set does not hold returns.
    fn missing(&self, kind: RowKind, name: &str, occurrence: u32) -> GateError {
        format!(
            "ref_manifest: {} {name}/{occurrence} not in {}",
            kind.as_str(),
            self.dir.join("MANIFEST.tsv").display()
        )
        .into()
    }

    /// Tensor and input rows whose `(kind, name, occurrence)` an earlier row
    /// already carries; a lookup never lands on one of them.
    pub fn duplicate_keys(&self) -> usize {
        self.index.duplicates
    }

    /// The step the set holds, as its first position, its tokens and the
    /// tokens before it: in a batch set every token of `# tokens` from
    /// position 0; in a decode-step set the last token, at `# decode_pos`,
    /// after the rest. A decode step runs right after its prefill, so
    /// `# prefill`, where the set has it, is the step's position. A set with
    /// no `# tokens`, a `# decode_pos` that is not its last token, or a
    /// `# prefill` that differs from it is an error naming the set.
    pub fn step(&self) -> Result<(u32, &[u32], &[u32]), GateError> {
        let path = self.dir.join("MANIFEST.tsv");
        let h = &self.header;
        let tokens = h
            .tokens
            .as_deref()
            .ok_or_else(|| format!("{} has no # tokens line", path.display()))?;
        if let (Some(p), Some(n)) = (h.decode_pos, h.prefill)
            && p != n
        {
            return Err(format!(
                "{}: # decode_pos {p} does not follow # prefill {n}",
                path.display()
            )
            .into());
        }
        match h.decode_pos {
            None => Ok((0, tokens, &[])),
            Some(p) if p as usize + 1 == tokens.len() => {
                let at = p as usize;
                Ok((p, &tokens[at..], &tokens[..at]))
            }
            Some(p) => Err(format!(
                "{}: # decode_pos {p} is not the last of the {} tokens",
                path.display(),
                tokens.len()
            )
            .into()),
        }
    }
}

/// Fill a header field once: a second line of the same key is an error.
fn once<T>(slot: &mut Option<T>, v: T, key: &str, at: At) -> Result<(), GateError> {
    if slot.replace(v).is_some() {
        return Err(format!("ref_manifest: {at}: a second # {key} line").into());
    }
    Ok(())
}

/// A manifest line's position, `<path>:<line>`, formatted only into an error.
#[derive(Clone, Copy)]
struct At<'a>(&'a Path, usize);

impl std::fmt::Display for At<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.0.display(), self.1)
    }
}

/// The number after the first `flag` in a `# flags` command line; `None` when
/// the line has no such flag.
fn flag_value(line: &str, flag: &str) -> Result<Option<u64>, String> {
    let mut args = line.split_whitespace();
    if !args.by_ref().any(|a| a == flag) {
        return Ok(None);
    }
    let v = args.next().unwrap_or("");
    v.parse()
        .map(Some)
        .map_err(|e| format!("{flag} {v:?}: {e}"))
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
/// CPU `ref`, ...). A name of the V4.1 family (`ref_deepseek41…`) is the
/// set of that name for the V4.1 file the tree runs ([`gguf::v41::set`]),
/// so every reader of a V4.1 set by name follows the file choice. Unlike
/// `ref_dir` this consults no `BLOOMERY_REF_*` variable — only the data
/// directory and the V4.1 file.
pub fn ref_dir_named(set: &str) -> PathBuf {
    let p = PathBuf::from(set);
    if p.is_absolute() {
        return p;
    }
    if set.starts_with(V41_SET_FAMILY) {
        return data_dir().join(gguf::v41::set(set));
    }
    data_dir().join(set)
}

/// The prefix every V4.1 oracle set name starts with (the profile's
/// `REF_SET_CPU` and its decode-step sets).
pub const V41_SET_FAMILY: &str = "ref_deepseek41";

/// The `tensor` rows of the set at `dir` ([`RefManifest::read`]) — the
/// rows the node readers below look names up in.
pub fn ref_manifest_in(dir: &std::path::Path) -> Result<Vec<RefRow>, GateError> {
    Ok(RefManifest::read(dir)?.tensors)
}

/// `ref_manifest_in` of `ref_dir()`.
pub fn ref_manifest() -> Result<Vec<RefRow>, GateError> {
    ref_manifest_in(&ref_dir())
}

/// The manifest row for `(name, occurrence)`, if the dump holds it. A scan
/// of the slice: a gate holding a [`RefManifest`] looks rows up through its
/// index ([`RefManifest::tensor`]).
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
                "find_ref_row: {}/{} not in {}",
                name,
                occurrence,
                dir.join("MANIFEST.tsv").display()
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

/// `ref_dir()`'s `(name, occurrence)` f32 file with its manifest row (dims,
/// op, sum) for chain checking. Reads the manifest on every call: a gate
/// that loads more than one row holds a [`RefManifest`] and calls
/// [`load_ref_in`].
pub fn ref_tensor(name: &str, occurrence: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    load_ref_in(&RefManifest::read(&ref_dir())?, name, occurrence)
}

/// `ref_tensor` over a manifest the caller already parsed: load
/// `(name, occ)`'s f32 file with its manifest row. A gate that reads many
/// tensors parses the manifest once and calls this, or [`load_ref_in`] with
/// a held [`RefManifest`].
pub fn load_ref(man: &[RefRow], name: &str, occ: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    let row = find_ref_row(man, name, occ)?;
    Ok((row.clone(), ref_tensor_of(row)?))
}

/// Tensor row `(name, occ)` of the held manifest `man`, through its index,
/// and the row's f32 file ([`ref_tensor_of_in`]).
pub fn load_ref_in(
    man: &RefManifest,
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), GateError> {
    let row = man.tensor(name, occ)?;
    Ok((row.clone(), ref_tensor_of_in(&man.dir, row)?))
}

/// Tensor row `(name, occ)` of the held manifest `man`, through its index,
/// and the row's LOGICAL elements ([`ref_tensor_logical_in`]).
pub fn load_ref_logical_in(
    man: &RefManifest,
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), GateError> {
    let row = man.tensor(name, occ)?;
    Ok((row.clone(), ref_tensor_logical_in(&man.dir, row)?))
}

/// The tensor's LOGICAL elements in ggml index order from `ref_dir()`: the
/// `.logical.f32` twin when the dump wrote one, else the plain file when
/// the row is provably not a flat VIEW read (a contiguous tensor's plain
/// file IS its logical order; a pre-v2 manifest without the `contig`
/// column is accepted for non-VIEW rows only). A VIEW row without a
/// logical twin is an error, never a silent flat read — that flat read is
/// a different tensor than the one being asked for. Reads the manifest on
/// every call: a gate that loads more than one row holds a [`RefManifest`]
/// and calls [`load_ref_logical_in`].
pub fn ref_tensor_logical(name: &str, occurrence: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    load_ref_logical_in(&RefManifest::read(&ref_dir())?, name, occurrence)
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
    if !plain_is_logical(row) {
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

/// Whether a row's plain file is its logical order, so a whole-tensor read
/// of it is the tensor: a v2 manifest's `contig` column decides; on a pre-v2
/// manifest only the op can rule out a flat VIEW read.
fn plain_is_logical(row: &RefRow) -> bool {
    match (row.contig, row.op.as_str()) {
        (Some(c), _) => c == 1,
        (None, "VIEW") => false,
        (None, _) => true,
    }
}

/// The `int` row twinning `(name, occurrence)` of kind `of` in `layout`,
/// through the manifest's index.
pub fn find_int_row<'a>(
    man: &'a RefManifest,
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
) -> Result<&'a IntRow, GateError> {
    man.index
        .ints
        .get(name)
        .and_then(|keys| {
            keys.iter()
                .find(|&&(k, o, l, _)| k == of && o == occurrence && l == layout)
        })
        .and_then(|&(.., at)| man.ints.get(at))
        .filter(|r| {
            r.name == name && r.occurrence == occurrence && r.of == of && r.layout == layout
        })
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
// locally; the bins import them from here.

/// The word a gate prints for one check's outcome: `PASS` or `FAIL`. The
/// gates' tables are compared line by line across rounds, so the spelling
/// has one owner.
pub fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// The values of `v` with a comma between each two — how a gate's line
/// prints a list of positions, rows or slots.
pub fn comma_list(v: &[impl ToString]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// FNV-1a 64 over bytes fed in order: the digest a gate prints over a
/// kernel's output bits, and `gate_p1` pins, so a change that must move no
/// bit is read against one line. [`Default`] is the digest of no bytes.
#[derive(Clone, Copy, Debug)]
pub struct Fnv1a64(u64);

impl Default for Fnv1a64 {
    fn default() -> Fnv1a64 {
        Fnv1a64(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv1a64 {
    /// The digest continued over `bytes`.
    #[must_use]
    pub fn bytes(mut self, bytes: &[u8]) -> Fnv1a64 {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self
    }

    /// The digest continued over the bits of each value, little-endian.
    #[must_use]
    pub fn f32s(self, v: &[f32]) -> Fnv1a64 {
        v.iter()
            .fold(self, |h, x| h.bytes(&x.to_bits().to_le_bytes()))
    }

    /// The digest continued over each value, little-endian.
    #[must_use]
    pub fn u32s(self, v: &[u32]) -> Fnv1a64 {
        v.iter().fold(self, |h, x| h.bytes(&x.to_le_bytes()))
    }

    /// The digest's value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Same length and the same bits at every index — the bit-identity contract
/// the fusion, rerun and replay checks assert. Not `==`: that calls `-0.0`
/// equal to `0.0` and two NaNs unequal, and both are differences a fusion
/// defect can produce.
pub fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Values of `a` and `b` with the same bits, index by index over the shorter
/// of the two.
pub fn same_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count()
}

/// Ulps between two finite f32, through integers ordered like the floats
/// they encode (`-0.0` and `+0.0` the same point).
fn ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let b = v.to_bits().cast_signed();
        if b < 0 { i32::MIN - b } else { b }
    };
    key(a).abs_diff(key(b))
}

/// The largest distance in ulps between `a` and `b`, index by index; 0 when
/// either is empty.
pub fn max_ulps(a: &[f32], b: &[f32]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ulps(x, y))
        .max()
        .unwrap_or(0)
}

/// The shape checks a gate runs on the device code it carries, read off the
/// PTX in this executable ([`ptx::current_exe_bundles`]): each entry of
/// `entries` goes to `judge` with its [`ptx::Counts`] and its [`ptx::body`],
/// and `judge` returns whether the entry passes and the line that says why,
/// printed here with the verdict after it. A spilled accumulator array or a
/// block width the host does not launch changes no output and no band, only
/// the time, so nothing else a gate checks can see it. Returns whether every
/// entry passed; an entry the executable does not carry is an error, and so
/// is an error `judge` returns.
pub fn ptx_shapes(
    entries: &[&str],
    mut judge: impl FnMut(&ptx::Counts, &[u8]) -> Result<(bool, String), GateError>,
) -> Result<bool, GateError> {
    let bundles = ptx::current_exe_bundles()?;
    let modules = ptx::modules(&bundles)?;
    let mut ok = true;
    for &name in entries {
        let body = ptx::body(&modules, name)
            .ok_or_else(|| format!("no PTX entry {name} in this executable"))?;
        let (pass, line) = judge(&ptx::Counts::of(name, body), body)?;
        println!("{line} {}", verdict(pass));
        ok &= pass;
    }
    Ok(ok)
}

/// [`ptx_shapes`] with one judge: every entry of `entries` compiles with no
/// local depot and no local loads or stores. Prints one `shape` line per
/// entry.
pub fn no_local_depot(entries: &[&str]) -> Result<bool, GateError> {
    ptx_shapes(entries, |c, _| {
        Ok((
            !c.depot && c.ld_local == 0 && c.st_local == 0,
            format!(
                "shape kernel={} local_depot={} ld_local={} st_local={}",
                c.name, c.depot, c.ld_local, c.st_local
            ),
        ))
    })
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

/// Replace the `N` bytes at byte `at_bytes` of `buf` with `new` and return
/// the bytes that were there: a read, then a write, each finished on
/// `stream` before the next. Refused by name, before any copy, when the span
/// passes the allocation (`buf.num_bytes()`). The span is device memory, not
/// Rust memory; the caller keeps `stream` the only work touching `buf` while
/// it runs (a gate's model between calls).
#[cfg(feature = "gpu")]
pub fn patch_bytes<T, const N: usize>(
    stream: &cuda_core::CudaStream,
    buf: &cuda_core::DeviceBuffer<T>,
    at_bytes: usize,
    new: [u8; N],
) -> Result<[u8; N], GateError> {
    use cuda_core::{IntoResult, sys};
    let total = buf.num_bytes();
    if at_bytes.checked_add(N).is_none_or(|end| end > total) {
        return Err(format!(
            "patch_bytes: bytes {at_bytes}..{at_bytes}+{N} pass the allocation's {total} bytes"
        )
        .into());
    }
    let at = buf.cu_deviceptr() + u64::try_from(at_bytes)?;
    let mut old = [0u8; N];
    stream.synchronize()?;
    // SAFETY: `at .. at + N` lies inside `buf`'s allocation (checked above);
    // `old` outlives the copy, which completes at the synchronize.
    let rc =
        unsafe { sys::cuMemcpyDtoHAsync_v2(old.as_mut_ptr().cast(), at, N, stream.cu_stream()) };
    rc.result()?;
    stream.synchronize()?;
    // SAFETY: the same span; `new` outlives the copy, which completes at the
    // synchronize.
    let rc = unsafe { sys::cuMemcpyHtoDAsync_v2(at, new.as_ptr().cast(), N, stream.cu_stream()) };
    rc.result()?;
    stream.synchronize()?;
    Ok(old)
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

/// A router reference's outputs: the probabilities `[n_expert, m]`, then the
/// chosen ids and their weights `[n_used, m]`.
pub type Routing = (Vec<f32>, Vec<i32>, Vec<f32>);

/// The V2-Lite model's router: 64 experts, 6 used per token. [`route_ref`]
/// routes at this shape, and the top-k readers without a bound check ids
/// against its expert count — that of the model the V2-Lite sets were dumped
/// from.
const V2_LITE_ROUTER: (u32, usize) = (64, 6);

/// [`route_ref_within`] at the V2-Lite router's shape.
pub fn route_ref(logits: &[f32], m: usize, scale: f32) -> Result<Routing, GateError> {
    let (n_expert, n_used) = V2_LITE_ROUTER;
    route_ref_within(logits, m, usize::try_from(n_expert)?, n_used, scale)
}

/// Host scalar router reference over `n_expert` experts, `n_used` chosen a
/// token, transcribed from the CPU engine's routing
/// (`model::moe::route_inner`): softmax with a serial f32 max fold, f32
/// `exp`, f64 sum accumulated ascending, f32 divide by `sum as f32`; top-k
/// by `(probability desc, id asc)`; weights `probs[id] * scale`. Layout
/// ggml t-major: logits/probs `[n_expert, m]`, ids/weights `[n_used, m]`.
/// Non-finite logits are an error.
pub fn route_ref_within(
    logits: &[f32],
    m: usize,
    n_expert: usize,
    n_used: usize,
    scale: f32,
) -> Result<Routing, GateError> {
    let (n, k) = (n_expert, n_used);
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

/// An F32 tensor of a split model file as f32 (norm gains): its type F32
/// and `want` values, checked — [`f32_tensor`] across shards.
pub fn split_f32(split: &Split, name: &str, want: usize) -> Result<Vec<f32>, GateError> {
    let (s, t) = split
        .find(name)
        .ok_or_else(|| format!("{name} is not in the model file"))?;
    if t.ty != GgmlType::F32 || t.dims.iter().product::<u64>() != want as u64 {
        return Err(format!("{name} is {:?} {:?}, want F32 x {want}", t.ty, t.dims).into());
    }
    let g = split.shard(s).ok_or("shard index out of range")?;
    Ok(g.data(t)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// [`widened_f16_bits_in`] of `ref_dir()`'s set, rows `0..rows` — the f16
/// cache view's first `rows` rows.
pub fn widened_f16_bits(row: &RefRow, rows: usize) -> Result<Vec<u16>, GateError> {
    let idx: Vec<u32> = (0..u32::try_from(rows)?).collect();
    widened_f16_bits_in(&ref_dir(), row, &idx)
}

/// Rows `idx` of f16 row `row` of the set at `dir` as f16 bits, row after
/// row, `ne[0]` values a row. The dump widened each half to f32 exactly, so
/// rounding back with `gguf::quant::f32_to_f16_bits` — the CPU oracle's own
/// rounding, not a second transcription — recovers ik's bits; a value that
/// does not widen back to itself is an error naming the row. Only the rows
/// asked for are read from the file. The rows are read from the plain file,
/// so it must be the tensor: a row that is a view or not contiguous
/// ([`plain_is_logical`]) is an error.
pub fn widened_f16_bits_in(dir: &Path, row: &RefRow, idx: &[u32]) -> Result<Vec<u16>, GateError> {
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use std::io::{Read, Seek, SeekFrom};

    let path = dir.join(row.file_name());
    let what = || {
        format!(
            "{} {}/{} ({})",
            row.kind.as_str(),
            row.name,
            row.occurrence,
            path.display()
        )
    };
    let count = row.count();
    if row.ty != "f16" || 4_u64.checked_mul(count) != Some(row.bytes) {
        return Err(format!(
            "widened_f16_bits: {} is {} with {} bytes for {count} values, want f16 widened to f32",
            what(),
            row.ty,
            row.bytes
        )
        .into());
    }
    if !plain_is_logical(row) {
        return Err(format!(
            "widened_f16_bits: {} is a view/non-contiguous row — its plain file is a flat \
             read, not the tensor",
            what()
        )
        .into());
    }
    let io =
        |e: std::io::Error| -> GateError { format!("widened_f16_bits: {}: {e}", what()).into() };
    let mut file = std::fs::File::open(&path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    if len != row.bytes {
        return Err(format!(
            "widened_f16_bits: {} is {len} bytes, the row says {}",
            what(),
            row.bytes
        )
        .into());
    }
    let width = row.ne[0];
    let mut line = vec![0u8; 4 * usize::try_from(width)?];
    let mut out = Vec::with_capacity(idx.len() * line.len() / 4);
    for &r in idx {
        let first = u64::from(r) * width;
        if first + width > count {
            return Err(format!(
                "widened_f16_bits: {}: row {r} is past its {count} values",
                what()
            )
            .into());
        }
        file.seek(SeekFrom::Start(4 * first)).map_err(io)?;
        file.read_exact(&mut line).map_err(io)?;
        for c in line.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(*c);
            let h = f32_to_f16_bits(v);
            if half_to_f32(h).to_bits() != v.to_bits() {
                return Err(format!(
                    "widened_f16_bits: {} holds {v} in row {r}, not a widened f16",
                    what()
                )
                .into());
            }
            out.push(h);
        }
    }
    Ok(out)
}

/// Every row of f16 row `row` of the set at `dir`, as f16 bits:
/// [`widened_f16_bits_in`] over rows `0 .. count / ne[0]`, which refuses a
/// row whose plain file is not the tensor.
pub fn widened_f16_rows_in(dir: &Path, row: &RefRow) -> Result<Vec<u16>, GateError> {
    let rows = row.count().checked_div(row.ne[0]).unwrap_or(0);
    let idx: Vec<u32> = (0..u32::try_from(rows)?).collect();
    widened_f16_bits_in(dir, row, &idx)
}

/// The f16 bits of mask row `row` of the set at `dir`: an `f16` row, its
/// plain file its logical order ([`plain_is_logical`]), whose f32 file holds
/// every value exactly `0.0` (a cell the query sees) or `-inf` (one it does
/// not), each mapped to that value's f16 bits (`0x0000`, `0xfc00`). Anything
/// else — another type, a view or non-contiguous row, a byte count that is
/// not four a value, any other value, `-0.0` and NaN included — is an error
/// naming the row, and for a value, the first offending one and its index.
pub fn mask_bits_in(dir: &Path, row: &RefRow) -> Result<Vec<u16>, GateError> {
    use gguf::quant::f32_to_f16_bits;

    const ZERO: u32 = 0x0000_0000;
    const MINUS_INF: u32 = 0xff80_0000;
    let path = dir.join(row.file_name());
    let what = || {
        format!(
            "{} {}/{} ({})",
            row.kind.as_str(),
            row.name,
            row.occurrence,
            path.display()
        )
    };
    if row.ty != "f16" {
        return Err(format!("mask_bits: {} is {}, want an f16 mask", what(), row.ty).into());
    }
    if !plain_is_logical(row) {
        return Err(format!(
            "mask_bits: {} is a view/non-contiguous row — its plain file is a flat read, not \
             the mask",
            what()
        )
        .into());
    }
    let raw = std::fs::read(&path).map_err(|e| format!("mask_bits: {}: {e}", what()))?;
    let want = 4_u64.checked_mul(row.count());
    if want != Some(row.bytes) || want != Some(raw.len() as u64) {
        return Err(format!(
            "mask_bits: {} is {} bytes and its row says {}, want 4 x {} values",
            what(),
            raw.len(),
            row.bytes,
            row.count()
        )
        .into());
    }
    let visible = f32_to_f16_bits(f32::from_bits(ZERO));
    let hidden = f32_to_f16_bits(f32::from_bits(MINUS_INF));
    raw.as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(i, b)| match u32::from_le_bytes(*b) {
            ZERO => Ok(visible),
            MINUS_INF => Ok(hidden),
            v => Err(format!(
                "mask_bits: {} holds {} at {i}, neither 0 nor -inf",
                what(),
                f32::from_bits(v)
            )
            .into()),
        })
        .collect()
}

/// [`topk_ids_logical_in`] of a row of `ref_dir()`'s manifest, which it
/// reads on every call: a gate reading more than one row holds a
/// [`RefManifest`] and calls [`topk_ids_logical_in`] or
/// [`topk_ids_logical_within`].
pub fn topk_ids_logical(row: &RefRow) -> Result<Vec<i32>, GateError> {
    topk_ids_logical_in(&RefManifest::read(&ref_dir())?, row)
}

/// [`topk_ids_logical_within`] of a row of `man`, a manifest already read,
/// with every id below the V2-Lite model's expert count.
pub fn topk_ids_logical_in(man: &RefManifest, row: &RefRow) -> Result<Vec<i32>, GateError> {
    topk_ids_logical_within(man, row, V2_LITE_ROUTER.0)
}

/// The topk row's ids from its LOGICAL twin: `ffn_moe_topk-L` is an i32
/// VIEW of the argsort output whose plain file is the flat parent read
/// (token 0's ranking only), so the ids of every token live in the logical
/// twin, read exact from its `.logical.i32` integer twin ([`ref_ints`]).
/// For a set whose manifest has no `int` rows at all (dumped before integer
/// twins existed) it reads the `.logical.f32` twin, each id cast to f32 by
/// the dumper. Accepted only when the row carries a logical twin and every
/// id is an integer in `0..n_expert`, which rules out any stride or cast
/// mix-up; the manifest's element-sum column describes the plain file and
/// is not checked here. `row` is a row of `man`.
pub fn topk_ids_logical_within(
    man: &RefManifest,
    row: &RefRow,
    n_expert: u32,
) -> Result<Vec<i32>, GateError> {
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
                Ok(id) if u32::try_from(id).is_ok_and(|e| e < n_expert) => Ok(id),
                _ => Err(format!(
                    "topk_ids_logical: {}/{} holds id {v}, outside 0..{n_expert}",
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
            if v.fract() == 0.0 && (0.0..f64::from(n_expert)).contains(&f64::from(v)) {
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
    use super::{
        GateError, GgmlType, Layout, NAN_F16, RefManifest, RowKind, dequant_row, find_int_row,
        find_ref_row_in, kquant_d_at, mask_bits_in, max_rel_err, topk_ids_logical_in,
        topk_ids_logical_within, widened_f16_bits_in, widened_f16_rows_in,
    };
    use std::path::{Path, PathBuf};

    /// A fresh directory for one test's set; tests of one process run in
    /// parallel, so each passes its own `what`.
    fn set_dir(what: &str) -> Result<PathBuf, GateError> {
        let dir = std::env::temp_dir().join(format!("bloomery-{what}-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// `lines` as the set's MANIFEST.tsv, read back.
    fn manifest(dir: &Path, lines: &[&str]) -> Result<RefManifest, GateError> {
        std::fs::write(dir.join("MANIFEST.tsv"), lines.join("\n") + "\n")?;
        RefManifest::read(dir)
    }

    fn f32_file(path: &Path, vals: &[f32]) -> Result<(), GateError> {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// The top-k reader takes the exact `.logical.i32` twin when the
    /// manifest has `int` rows and the `.logical.f32` twin when it has none
    /// — the f32 file below disagrees in one id, so the result shows which
    /// file was read — and a manifest with twins but none for the row is an
    /// error. Ids are bounded by the caller's expert count, V2-Lite's 64
    /// where none is given. The name has a space, so the files resolve only
    /// through the dumper's `safe_name` rule.
    #[test]
    fn topk_ids_take_the_integer_twin_when_the_set_has_twins() -> Result<(), GateError> {
        let dir = set_dir("topk-twin")?;
        let ids: [i32; 4] = [3, 63, 0, 17];
        let stem = "ffn_moe_topk-1_(view).0.logical";
        std::fs::write(
            dir.join(format!("{stem}.i32")),
            ids.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        f32_file(&dir.join(format!("{stem}.f32")), &[3.0, 63.0, 0.0, 16.0])?;
        let tensor =
            "tensor\tffn_moe_topk-1 (view)\t0\ti32\t2\t2\t1\t1\t16\t0.000000\tVIEW\t0\t1\t-\t-";
        let int = format!(
            "int\tffn_moe_topk-1 (view)\t0\ttensor\ti32\ti32\tlogical\t4\t16\t83\t63\t{stem}.i32"
        );
        let other = int.replacen("\t0\t", "\t1\t", 1);
        let twins = manifest(&dir, &[tensor, &int])?;
        assert_eq!(topk_ids_logical_in(&twins, &twins.tensors[0])?, ids);
        assert!(topk_ids_logical_within(&twins, &twins.tensors[0], 63).is_err());
        let cast = manifest(&dir, &[tensor])?;
        assert_eq!(
            topk_ids_logical_in(&cast, &cast.tensors[0])?,
            [3, 63, 0, 16]
        );
        assert!(topk_ids_logical_within(&cast, &cast.tensors[0], 63).is_err());
        let unmatched = manifest(&dir, &[tensor, &other])?;
        assert!(topk_ids_logical_in(&unmatched, &unmatched.tensors[0]).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The index finds a row by `(kind, name, occurrence)` — a tensor and an
    /// input of one name are two rows — and of a key two rows carry, the
    /// first, the row a scan in file order finds; the second is counted as a
    /// duplicate. An `int` row is found by its twin's kind and layout too.
    /// The positional reads agree with a scan: a row's position, the first
    /// row of a name whose first written occurrence is not 0, the last row
    /// of a name before a reader, and an input's first toucher, the tensor
    /// row after it. A name prefix finds its one row, and more than one or
    /// none is an error.
    #[test]
    fn the_index_finds_the_row_a_scan_finds_first() -> Result<(), GateError> {
        let dir = set_dir("index")?;
        let man = manifest(
            &dir,
            &[
                "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t1.0\tADD",
                "tensor\tx\t1\tf32\t1\t1\t1\t1\t4\t2.0\tADD",
                "input\tx\t0\tf32\t1\t1\t1\t1\t4\t3.0\tNONE",
                "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t4.0\tADD",
                "int\tp\t0\tinput\ti32\ti32\tflat\t1\t4\t7\t7\tp.0.input.i32",
                "int\tp\t0\tinput\ti32\ti32\tlogical\t1\t4\t7\t7\tp.0.input.logical.i32",
                "skip\ty\t0\tq8_0\tquantized",
                "tensor\ty\t1\tf32\t1\t1\t1\t1\t4\t5.0\tADD",
            ],
        )?;
        let sum = |r: Option<&super::RefRow>| r.map(|r| r.sum);
        assert_eq!(sum(man.find(RowKind::Tensor, "x", 0)), Some(1.0));
        assert_eq!(
            sum(man.find(RowKind::Tensor, "x", 0)),
            Some(find_ref_row_in(&dir, &man.tensors, "x", 0)?.sum)
        );
        assert_eq!(man.tensor("x", 1)?.sum, 2.0);
        assert_eq!(man.input("x", 0)?.sum, 3.0);
        assert!(man.find(RowKind::Input, "x", 1).is_none());
        assert!(man.tensor("y", 0).is_err());
        assert_eq!(man.duplicate_keys(), 1);
        let twin = find_int_row(&man, "p", 0, RowKind::Input, Layout::Logical)?;
        assert_eq!(twin.file, "p.0.input.logical.i32");
        assert!(find_int_row(&man, "p", 0, RowKind::Tensor, Layout::Flat).is_err());

        assert_eq!(man.position(RowKind::Tensor, "x", 1), Some(1));
        assert_eq!(man.tensor_at("y", 1)?.0, 3);
        assert_eq!(sum(man.first_named(RowKind::Tensor, "x")), Some(1.0));
        assert_eq!(sum(man.first_named(RowKind::Tensor, "y")), Some(5.0));
        assert!(man.first_named(RowKind::Input, "y").is_none());
        assert_eq!(man.last_before(2, Some("x"))?.0, 1);
        assert!(man.last_before(0, Some("x")).is_err());
        assert!(man.last_before(3, None).is_err());
        let touched = |at: usize| {
            man.first_touched_by(at)
                .iter()
                .map(|r| r.sum)
                .collect::<Vec<_>>()
        };
        assert_eq!(touched(2), [3.0]);
        assert!(touched(1).is_empty() && touched(3).is_empty());

        assert_eq!(man.only_with_prefix(RowKind::Input, "x")?.sum, 3.0);
        assert_eq!(man.only_with_prefix(RowKind::Tensor, "y")?.sum, 5.0);
        assert!(man.only_with_prefix(RowKind::Tensor, "x").is_err());
        assert!(man.only_with_prefix(RowKind::Input, "y").is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Every header line the V4.1 and V2-Lite sets carry, spelled as they
    /// spell it (long values shortened): a known key fills its field — `-c`
    /// and `-t` out of `# flags`, the step out of `# tokens` and `# decode_pos` —
    /// every other `#` line is kept verbatim in file order, and a known key
    /// twice or a value that does not parse is an error.
    #[test]
    fn header_lines_fill_their_fields_or_are_kept() -> Result<(), GateError> {
        let dir = set_dir("header")?;
        let title = "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian";
        let model = "# model\t/models/M/M-00001-of-00009.gguf";
        let schedule =
            "# prefill_schedule\tevery-node — each node was asked for and computed alone";
        let columns = [
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1",
            "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile",
            "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1",
        ];
        let row = "tensor\tinp_embd\t0\tf32\t2\t1\t1\t1\t8\t0.5\tGET_ROWS";
        let step = [
            title,
            model,
            "# build\t49ef19d0",
            "# arch\tdeepseek41",
            "# model_file\tM-00001-of-00009.gguf",
            "# tokens\t5,19415,271",
            "# flags\t-m /models/M/M-00001-of-00009.gguf --expect-arch deepseek41 --tokens-file /d/c.ids \
             --tokens-count 3 -ngl 0 -c 2048 -t 32 --defer-experts --decode-step --no-fused-idx-topk",
            "# tokens_file\t/d/c.ids",
            "# tokens_file_sha256\tf7785d0f",
            "# tokens_count\t3",
            "# prefill\t2",
            "# decode_pos\t2",
            schedule,
            "# state_inputs\tpersistent leaves the step reads are input rows",
            "# fused_idx_topk\t0",
            columns[0],
            columns[1],
            columns[2],
            row,
            "# complete\t1\t0",
        ];
        let man = manifest(&dir, &step)?;
        let h = &man.header;
        assert_eq!(man.arch.as_deref(), Some("deepseek41"));
        assert_eq!(man.build.as_deref(), Some("49ef19d0"));
        assert_eq!(man.complete, Some((1, 0)));
        assert_eq!(h.model_file.as_deref(), Some("M-00001-of-00009.gguf"));
        assert_eq!(h.tokens.as_deref(), Some(&[5, 19415, 271][..]));
        assert_eq!(h.tokens_file.as_deref(), Some("/d/c.ids"));
        assert_eq!(h.tokens_file_sha256.as_deref(), Some("f7785d0f"));
        assert_eq!(h.tokens_count, Some(3));
        assert!(
            h.flags
                .as_deref()
                .is_some_and(|f| f.ends_with("--no-fused-idx-topk"))
        );
        assert_eq!((h.ctx, h.threads), (Some(2048), Some(32)));
        assert_eq!((h.prefill, h.decode_pos), (Some(2), Some(2)));
        assert!(h.state_inputs.is_some());
        assert_eq!(h.fused_idx_topk, Some(false));
        assert_eq!(
            h.other,
            [title, model, schedule, columns[0], columns[1], columns[2]]
        );
        assert_eq!(man.step()?, (2, &[271][..], &[5, 19415][..]));

        // A V2-Lite set: no V4.1 line, the 11-column rows; its step is the
        // whole sequence from position 0.
        let v2 = manifest(
            &dir,
            &[
                title,
                "# model\t/models/small/L.gguf",
                "# build\tc10fbbcc",
                "# tokens\t100000,549",
                "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top",
                row,
                "# complete\t1\t0",
            ],
        )?;
        let h = &v2.header;
        assert_eq!(
            (v2.arch.as_deref(), v2.build.as_deref()),
            (None, Some("c10fbbcc"))
        );
        assert_eq!(
            (
                h.model_file.as_deref(),
                h.flags.as_deref(),
                h.ctx,
                h.threads
            ),
            (None, None, None, None)
        );
        assert_eq!(
            (h.prefill, h.decode_pos, h.fused_idx_topk),
            (None, None, None)
        );
        assert_eq!(h.other.len(), 3);
        assert_eq!(v2.step()?, (0, &[100000, 549][..], &[][..]));

        for bad in [
            "# tokens\t5,x",
            "# flags\t-m M.gguf -c",
            "# flags\t-m M.gguf -t x -c 512",
            "# fused_idx_topk\tyes",
            "# decode_pos\t-1",
            "# build\tc10fbbcc",
        ] {
            let lines = ["# build\tc10fbbcc", bad, row];
            assert!(manifest(&dir, &lines).is_err(), "{bad:?} parsed");
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The mask reader maps `0.0` and `-inf` to their f16 bits and refuses
    /// every other value — `-0.0` and NaN too — naming the row and the first
    /// offending value, a row that is not f16, and one whose plain file is
    /// not its logical order.
    #[test]
    fn mask_bits_take_zero_and_minus_infinity_only() -> Result<(), GateError> {
        let dir = set_dir("mask")?;
        let node = "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t0\tNONE";
        let row = "input\tkq_mask\t0\tf16\t2\t2\t1\t1\t16\t-inf\tNONE\t1\t0\t-\t-";
        let man = manifest(&dir, &[node, row])?;
        let file = dir.join(man.inputs[0].file_name());
        let ninf = f32::NEG_INFINITY;
        f32_file(&file, &[0.0, ninf, ninf, 0.0])?;
        assert_eq!(
            mask_bits_in(&dir, &man.inputs[0])?,
            [0x0000, 0xfc00, 0xfc00, 0x0000]
        );
        for (bad, shown) in [(-0.0f32, "-0"), (1.0, "1"), (f32::NAN, "NaN")] {
            f32_file(&file, &[0.0, ninf, bad, bad])?;
            let e = mask_bits_in(&dir, &man.inputs[0]).unwrap_err().to_string();
            assert!(
                e.contains("kq_mask/0") && e.contains(&format!("holds {shown} at 2,")),
                "{e}"
            );
        }
        let not_f16 = manifest(&dir, &[node, &row.replace("\tf16\t", "\tf32\t")])?;
        assert!(mask_bits_in(&dir, &not_f16.inputs[0]).is_err());
        f32_file(&file, &[0.0, ninf, ninf, 0.0])?;
        let flat = manifest(&dir, &[node, &row.replace("NONE\t1\t0", "NONE\t0\t0")])?;
        assert!(mask_bits_in(&dir, &flat.inputs[0]).is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// The widened-f16 reader returns the rows asked for, in the order asked,
    /// as f16 bits, and reads no other row: a row holding a value no half
    /// widens to is refused when asked for and passes unread otherwise; a
    /// row past the tensor is an error. The whole-tensor reader asks for
    /// every row. Both refuse a row whose plain file is not its logical order.
    #[test]
    fn widened_f16_bits_read_only_the_rows_asked_for() -> Result<(), GateError> {
        let dir = set_dir("widened")?;
        let row = "tensor\tcache\t0\tf16\t2\t3\t1\t1\t24\t0\tVIEW\t1\t0\t-\t-";
        let man = manifest(&dir, &[row])?;
        let cache = &man.tensors[0];
        let min_normal = 2.0f32.powi(-14);
        f32_file(
            &dir.join(cache.file_name()),
            &[1.0, -2.0, 0.1, 0.5, 65504.0, min_normal],
        )?;
        assert_eq!(
            widened_f16_bits_in(&dir, cache, &[2, 0])?,
            [0x7bff, 0x0400, 0x3c00, 0xc000]
        );
        assert!(widened_f16_bits_in(&dir, cache, &[1]).is_err());
        assert!(widened_f16_bits_in(&dir, cache, &[3]).is_err());
        let not_f16 = manifest(&dir, &[&row.replace("\tf16\t", "\tf32\t")])?;
        assert!(widened_f16_bits_in(&dir, &not_f16.tensors[0], &[0]).is_err());

        assert!(widened_f16_rows_in(&dir, cache).is_err());
        f32_file(
            &dir.join(cache.file_name()),
            &[1.0, -2.0, 0.5, 0.25, 65504.0, min_normal],
        )?;
        assert_eq!(
            widened_f16_rows_in(&dir, cache)?,
            [0x3c00, 0xc000, 0x3800, 0x3400, 0x7bff, 0x0400]
        );
        let view = manifest(&dir, &[&row.replace("VIEW\t1\t0", "VIEW\t0\t0")])?;
        assert!(widened_f16_rows_in(&dir, &view.tensors[0]).is_err());
        assert!(widened_f16_bits_in(&dir, &view.tensors[0], &[0]).is_err());
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

    /// The bytes `kquant_d_at` names are the super-block scale `d` the
    /// dequant reads: a finite block dequantizes finite, and [`NAN_F16`]
    /// there makes every one of its 256 values NaN — a byte of the quants or
    /// the sub-block scales would leave most of them finite. A type the
    /// helper does not know is refused by name.
    #[test]
    fn kquant_d_at_is_the_scale_every_value_reads() -> Result<(), GateError> {
        for ty in [GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q6_K] {
            let size = usize::try_from(ty.type_size().ok_or("no type size")?)?;
            let at = kquant_d_at(ty)?;
            let mut block: Vec<u8> = (0..size).map(|i| (i * 37 % 251) as u8).collect();
            // d = 1.0 and, for Q4_K, dmin = 0.0 right after it.
            block[at..at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            if ty == GgmlType::Q4_K {
                block[2..4].fill(0);
            }
            let mut y = [0f32; 256];
            dequant_row(ty, &block, &mut y)?;
            assert!(y.iter().all(|v| v.is_finite()), "{ty:?}: the clean block");
            block[at..at + 2].copy_from_slice(&NAN_F16.to_le_bytes());
            dequant_row(ty, &block, &mut y)?;
            assert!(y.iter().all(|v| v.is_nan()), "{ty:?}: NaN at byte {at}");
        }
        let refused = kquant_d_at(GgmlType::Q8_0).map_err(|e| e.to_string());
        assert!(refused.is_err_and(|e| e.starts_with("kquant_d_at: Q8_0")));
        Ok(())
    }
}
