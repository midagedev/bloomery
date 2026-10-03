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
pub mod ds41_meta;
pub mod engine;
pub mod flip;
pub mod gemm32;
#[cfg(feature = "gpu")]
pub mod generate;
pub mod hc_host;
#[cfg(feature = "gpu")]
pub mod host_stats;
pub mod ik_norm;
pub mod ik_q8_2;
pub mod kld;
pub mod model_file;
#[cfg(feature = "gpu")]
pub mod nodes;
pub mod oracle;
pub mod prompts;
pub mod ptx;
pub mod qwen3moe;
pub mod record;
#[cfg(feature = "gpu")]
pub mod residency38;
pub mod rounding;
#[cfg(feature = "deepseek41")]
pub mod serve_client;

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, Split, TensorInfo};
use model::arch::Arch;
use std::path::PathBuf;

/// Kernel gate band against the quantized-input reference.
/// PIN(2026-09-21): tightened 1e-2 -> 1e-5. Measured worst on the landed
/// kernels: 1.149e-7 (gate_p1, 12 shapes), 1.375e-7 (gate_p2); the f32
/// gemvs already gate at 1e-5 (worst 1.9e-7). 1e-2 was sized for the
/// raw-activation comparison this band no longer makes, and would pass a
/// sub-block scale defect landing near 1e-3 on a shape with no hash pin.
pub const KERNEL_BAND: f32 = 1e-5;

/// The V4.1 greedy rule of the long gate's free arm: at the first generated
/// id that differs from ik's, our top-1 margin must be below this — a near
/// tie [derived, plan.md: ≈ 3 σ_rel].
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

/// The model file every gate and server opens: the one a server named on
/// its command line (`-m`, `--hf`: [`model_file`]), else
/// `$BLOOMERY_REF_MODEL`, with no default here. The variable's file is a
/// property of the model profile (`tools/ref/models/<architecture>.sh`);
/// `tools/box.sh` exports it into every box command, the same value the
/// runners and the C++ harnesses read. An empty value counts as unset, as in
/// the profile. `open_model` and any gate that prints the path read it here,
/// so the printed name is the file that was opened.
pub fn ref_model_path() -> Result<PathBuf, GateError> {
    if let Some(p) = model_file::model_file() {
        return Ok(p.clone());
    }
    match std::env::var_os("BLOOMERY_REF_MODEL") {
        Some(p) if !p.is_empty() => Ok(PathBuf::from(p)),
        _ => Err(hf::source::NONE.into()),
    }
}

pub use refset::data_dir;

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
// A set's MANIFEST.tsv, its rows and the files they name are read by
// `refset::ik`, the one reader the model crate's tests share; its names are
// re-exported here so the gate binaries import them from this crate. The
// functions below add the directory this crate's environment names.

pub use refset::RefError;
pub use refset::ik::{
    FileElem, IntRow, Layout, RefHeader, RefManifest, RefRow, RowKind, dump_file_name, dump_stem,
    find_int_row, find_ref_row_in, load_ref_in, load_ref_logical_in, mask_bits_in, ref_ints,
    ref_ints_of_in, ref_manifest_in, ref_tensor_logical_in, ref_tensor_of_in,
    topk_ids_logical_within, widened_f16_bits_in, widened_f16_rows_in,
};

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

/// `ref_manifest_in` of `ref_dir()`.
pub fn ref_manifest() -> Result<Vec<RefRow>, GateError> {
    Ok(ref_manifest_in(&ref_dir())?)
}

/// The manifest row for `(name, occurrence)`, if the dump holds it. A scan
/// of the slice: a gate holding a [`RefManifest`] looks rows up through its
/// index ([`RefManifest::tensor`]).
pub fn find_ref_row<'a>(
    man: &'a [RefRow],
    name: &str,
    occurrence: u32,
) -> Result<&'a RefRow, GateError> {
    Ok(find_ref_row_in(&ref_dir(), man, name, occurrence)?)
}

/// Load one manifest row's f32 file, checking it against its own row: the
/// type must be f32, the row's byte count must equal 4·ne0·ne1·ne2·ne3 and
/// the file's length, and every value must be finite. Every error names the
/// offending path.
pub fn ref_tensor_of(row: &RefRow) -> Result<Vec<f32>, GateError> {
    Ok(ref_tensor_of_in(&ref_dir(), row)?)
}

/// `ref_dir()`'s `(name, occ)` f32 file with its manifest row (dims, op, sum)
/// for chain checking, over a manifest the caller already parsed. A gate that
/// reads many tensors parses the manifest once and calls this, or
/// [`load_ref_in`] with a held [`RefManifest`].
pub fn load_ref(man: &[RefRow], name: &str, occ: u32) -> Result<(RefRow, Vec<f32>), GateError> {
    let row = find_ref_row(man, name, occ)?;
    Ok((row.clone(), ref_tensor_of(row)?))
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
    Ok(widened_f16_bits_in(&ref_dir(), row, &idx)?)
}

/// [`topk_ids_logical_within`] of a row of `man`, a manifest already read,
/// with every id below the V2-Lite model's expert count.
pub fn topk_ids_logical_in(man: &RefManifest, row: &RefRow) -> Result<Vec<i32>, GateError> {
    Ok(topk_ids_logical_within(man, row, V2_LITE_ROUTER.0)?)
}

#[cfg(test)]
mod tests {
    use super::{GateError, GgmlType, NAN_F16, dequant_row, kquant_d_at, max_rel_err};

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
