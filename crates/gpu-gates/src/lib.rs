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
pub mod beside;
#[cfg(feature = "deepseek41")]
pub mod bind;
pub mod block;
pub mod ctx_ladder;
#[cfg(feature = "deepseek41")]
pub mod ds41_media;
pub mod ds41_meta;
pub mod engine;
pub mod flash_band;
pub mod flip;
pub mod gemm32;
#[cfg(feature = "gpu")]
pub mod generate;
pub mod gpu_census;
pub mod hc_host;
#[cfg(feature = "gpu")]
pub mod host_stats;
pub mod ik_norm;
pub mod ik_q8_2;
pub mod image_plan;
#[cfg(feature = "gpu")]
pub mod iq_rows;
pub mod kld;
#[cfg(feature = "gpu")]
pub mod kv_planes;
pub mod llama_launch;
pub mod llama_place;
pub mod logit_rows;
pub mod logits_band;
pub mod model_file;
pub mod need;
#[cfg(feature = "gpu")]
pub mod nodes;
pub mod oracle;
#[cfg(feature = "gpu")]
pub mod prefill_gate;
pub mod prompts;
pub mod ptx;
pub mod q36_fixture;
pub mod q38_fixture;
pub mod qwen3moe;
pub mod qwen_seat;
pub mod record;
#[cfg(feature = "gpu")]
pub mod residency;
#[cfg(feature = "gpu")]
pub mod residency38;
#[cfg(feature = "gpu")]
pub mod rope_gate;
pub mod rounding;
#[cfg(all(feature = "deepseek41", feature = "hf"))]
pub mod seat_bin;
#[cfg(feature = "deepseek41")]
pub mod serve_client;
// A server start's settings file (`--config`), read as the flags it names.
#[cfg(feature = "deepseek41")]
pub mod serve_config;
#[cfg(feature = "deepseek41")]
pub mod side_seat;
#[cfg(feature = "gpu")]
pub mod slots_gate;
#[cfg(feature = "gpu")]
pub mod start;
#[cfg(feature = "gpu")]
pub mod swiglu_host;
pub mod tap_dump;
pub mod tier;
pub mod ubatch_pick;
// The image encoder's pieces the serve seats share; the tower itself is `bloomery-gpu-vision`'s, behind `vision`.
#[cfg(any(feature = "vision", feature = "deepseek41"))]
pub mod vision_seat;

use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, Split, TensorInfo};
use model::arch::Arch;
use model::placement::need_hook;
use std::path::{Path, PathBuf};

/// Kernel gate band against the quantized-input reference.
/// PIN(2026-09-21): tightened 1e-2 -> 1e-5. Measured worst on the landed
/// kernels: 1.149e-7 (gate_p1, 12 shapes), 1.375e-7 (gate_p2); the f32
/// gemvs already gate at 1e-5 (worst 1.9e-7). 1e-2 was sized for the
/// raw-activation comparison this band no longer makes, and would pass a
/// sub-block scale defect landing near 1e-3 on a shape with no hash pin.
pub const KERNEL_BAND: f32 = 1e-5;

/// The V4.1 greedy rule of the long gate's free arm: at the first generated
/// id that differs from ik's, our top-1 margin must be below this — a near
/// tie [derived, <https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md>: ≈ 3 σ_rel].
pub const GREEDY_MARGIN: f32 = 1.5;

pub type GateError = Box<dyn std::error::Error>;

/// Kill the child when its spawning process dies, including while stopped.
#[cfg(feature = "gpu")]
pub fn die_with_parent(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;

    let parent = std::process::id();
    let setup = move || {
        // SAFETY: prctl changes only this child's parent-death signal.
        let rc = unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: getppid reads this child's parent PID.
        let ppid = unsafe { libc::getppid() };
        if u32::try_from(ppid) != Ok(parent) {
            return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
        }
        Ok(())
    };
    // SAFETY: setup uses only async-signal-safe calls and no allocation.
    unsafe { cmd.pre_exec(setup) };
}

/// The exit of every gate binary's `main`: `Ok` is success; an `Err` prints
/// `<name>: <error>` (the Display, not the Debug a `Result` main prints) and
/// each `source()` beneath it as `  caused by: ...` to stderr, then fails.
/// A message that already opens with `<name>: ` is not prefixed twice. A
/// process whose load the need hook refused ([`need_hook::refusal`]) ends 75
/// by it, whichever error carried the refusal up, with the hook's line named
/// if the error lost it.
pub fn exit_with(name: &str, r: Result<(), GateError>) -> std::process::ExitCode {
    exit_status(name, r, need_hook::refusal())
}

/// [`exit_with`] for a process whose need hook refusal (if any) is `refusal`.
fn exit_status(
    name: &str,
    r: Result<(), GateError>,
    refusal: Option<String>,
) -> std::process::ExitCode {
    let Err(e) = r else {
        return std::process::ExitCode::SUCCESS;
    };
    print_refusal(name, &e, refusal.as_deref());
    failure_status(refusal.is_some())
}

/// The stderr lines of a failed run's error `e`: `<name>: <error>`, each `source()` beneath it
/// as `  caused by: ...` (one that says what the line above it says is not repeated), and the
/// need hook's `refusal` line when the error lost it. The text
/// every exit of a gate or a server prints before its status ([`exit_with`], `start::exit_serve`).
pub(crate) fn print_refusal(name: &str, e: &GateError, refusal: Option<&str>) {
    let msg = e.to_string();
    if msg.starts_with(&format!("{name}: ")) {
        eprintln!("{msg}");
    } else {
        eprintln!("{name}: {msg}");
    }
    let mut above = msg.clone();
    let mut cause = e.source();
    while let Some(c) = cause {
        let text = c.to_string();
        if text != above {
            eprintln!("  caused by: {text}");
        }
        above = text;
        cause = c.source();
    }
    if let Some(why) = refusal.filter(|why| !msg.contains(why)) {
        eprintln!("{name}: {why}");
    }
}

/// The status of a failed run: [`need_hook::REFUSED`] when the need hook
/// refused its load (`EX_TEMPFAIL`: the box could not take it now, ask
/// again), else 1.
fn failure_status(refused: bool) -> std::process::ExitCode {
    if refused {
        std::process::ExitCode::from(need_hook::REFUSED)
    } else {
        std::process::ExitCode::FAILURE
    }
}

/// The error a gate returns when its `ok` accumulator came out false. Every
/// failing check has printed its own line by then; this ends the run through
/// [`exit_with`]. One spelling for every gate, as [`verdict`] is for a check.
pub fn checks_failed() -> GateError {
    "FAILED: one or more checks above did not pass".into()
}

/// The optional `--bench-arm` prefix, refused unless it selects an arm.
/// Call before opening the card; `ops` names every arm the bench can run.
pub fn bench_arm(
    args: impl Iterator<Item = String>,
    mut ops: impl Iterator<Item = String>,
) -> Result<Option<String>, GateError> {
    let mut args = args.skip_while(|a| a != "--bench-arm");
    if args.next().is_none() {
        return Ok(None);
    }
    let arm = args
        .next()
        .ok_or("--bench-arm wants an arm name (a `bench op=` prefix)")?;
    if !ops.any(|op| op.starts_with(&arm)) {
        return Err(format!("--bench-arm {arm:?} names no arm").into());
    }
    Ok(Some(arm))
}

/// The model file every gate and server opens: the one a server named on
/// its command line (`-m`, `--hf`: [`model_file`]), else
/// `$BLOOMERY_REF_MODEL`, with no default here. The variable's file is a
/// property of the model profile (`tools/ref/models/<architecture>.sh`);
/// `tools/box.sh` exports it into every box command, the same value the
/// runners and the C++ harnesses read. An empty value counts as unset, as in
/// the profile. `open_model` and any gate that prints the path read it here,
/// so the printed name is the file that was opened. Under `BLOOMERY_TIER=fixture`
/// ([`tier`]) the file must be a whole fixture ([`model_arch`]: a GGUF split's
/// keys, a checkpoint folder's header and fixture cut): a path that is not one is a
/// named error here, never a gate that ran on a real file under the fixture's
/// name. The real tier, and no tier, return the path unread. A command that
/// runs the V4.1 profile reads the file's other owner too (`gguf::v41`, which
/// the engine's placement, the oracle sets and the tokenizer read): the two
/// naming different files is that owner's named error, raised here before any
/// gate opens either.
pub fn ref_model_path() -> Result<PathBuf, GateError> {
    let path = if let Some(p) = model_file::model_file() {
        p.clone()
    } else {
        match std::env::var_os("BLOOMERY_REF_MODEL") {
            Some(p) if !p.is_empty() => {
                if gguf::v41::PathEnv::from_env().runs_v41() {
                    gguf::v41::try_model()?;
                }
                PathBuf::from(p)
            }
            _ => return Err(model_file::NONE.into()),
        }
    };
    let tier = tier::Tier::from_env()?;
    if tier == tier::Tier::Fixture {
        model_arch(tier, &path)?;
    }
    Ok(path)
}

/// The architecture the model at `path` names: a GGUF split's, from its first shard's header, or
/// a checkpoint folder's ([`folder_arch`]). Under `tier` the fixture tier holds the model to a
/// whole fixture: a file's keys by [`tier::Tier::check_file`], a folder by [`folder_arch`].
pub fn model_arch(tier: tier::Tier, path: &Path) -> Result<Option<String>, GateError> {
    if path.is_dir() {
        return folder_arch(tier, path);
    }
    let g = Gguf::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    tier.check_file(path, g.iter_kv().map(|(k, _)| k))?;
    Ok(g.architecture().map(str::to_owned))
}

/// The architecture the checkpoint folder `dir`'s header names (`model_hf::folder::open`). Under
/// the fixture tier the folder is that architecture's fixture cut, `refset::fixture::first_shard`'s
/// directory, and its first shard's `bloomery.fixture.*` keys are a whole fixture's
/// (`refset::fixture::keys` refuses any other by name); a header that names no architecture is
/// refused there by name.
#[cfg(feature = "checkpoint")]
pub fn folder_arch(tier: tier::Tier, dir: &Path) -> Result<Option<String>, GateError> {
    let f = model_hf::folder::open(dir)
        .map_err(|e| format!("checkpoint folder {}: {e}", dir.display()))?;
    let arch = f.header.architecture().map(str::to_owned);
    if tier == tier::Tier::Fixture {
        let named = arch.as_deref().ok_or_else(|| {
            format!(
                "checkpoint folder {}: its header names no architecture, so no fixture cut is its",
                dir.display()
            )
        })?;
        let first = PathBuf::from(refset::fixture::first_shard(named)?);
        if first.parent() != Some(dir) {
            return Err(format!(
                "checkpoint folder {}: the fixture tier runs the {named} fixture, whose first shard is {}",
                dir.display(),
                first.display()
            )
            .into());
        }
        refset::fixture::keys(&first)?;
    }
    Ok(arch)
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn folder_arch(_tier: tier::Tier, dir: &Path) -> Result<Option<String>, GateError> {
    Err(format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        dir.display()
    )
    .into())
}

pub use refset::data_dir;

/// The binary `name` beside the running one.
pub fn beside(name: &str) -> Result<PathBuf, GateError> {
    Ok(std::env::current_exe()?.with_file_name(name))
}

/// The first `n` ids of `$BLOOMERY_DATA/<dir>/corpus-prose.ids`: text ids of the real tokenizer,
/// which a fixture's embedding table holds the rows of.
///
/// # Errors
/// A file that is missing, holds fewer ids, or holds a word that is not an id.
pub fn prose_ids(dir: &str, n: usize) -> Result<Vec<u32>, GateError> {
    let path = data_dir().join(dir).join("corpus-prose.ids");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    first_ids(&path, &text, n)
}

/// The first `n` whitespace-separated ids of `text`, read from `path`.
fn first_ids(path: &std::path::Path, text: &str, n: usize) -> Result<Vec<u32>, GateError> {
    let ids = text
        .split_whitespace()
        .take(n)
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() < n {
        return Err(format!("{}: {} ids, the gate reads {n}", path.display(), ids.len()).into());
    }
    Ok(ids)
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

/// Write `decls` (each tensor's name, dims, type and bytes) as a GGUF at
/// `path`, with the metadata `kvs`.
pub fn write_gguf(
    path: &Path,
    kvs: &[(String, gguf::Value)],
    decls: &[(String, Vec<u64>, GgmlType, Vec<u8>)],
) -> Result<(), GateError> {
    let tensors = decls
        .iter()
        .map(|(name, dims, ty, bytes)| gguf::write::TensorDecl {
            name: name.clone(),
            dims: dims.clone(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        })
        .collect();
    let layout = gguf::write::Layout::new(kvs, tensors)?;
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut writer = gguf::write::Writer::new(file, layout)?;
    for (name, _, _, bytes) in decls {
        writer.tensor(name, bytes)?;
    }
    writer.finish()?;
    Ok(())
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

/// A seeded xorshift64 stream for a gate's synthetic inputs: the same seed
/// draws the same values on every run.
pub struct GateRng(pub u64);

impl GateRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A draw in `[0, 1)`: the high 24 bits over `2^24`.
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A draw below `n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A draw in `[lo, hi)`.
    pub fn uni(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// `f(i)` for every `i < n`, in order, the units shared among worker
/// threads; a unit that panics panics the caller.
#[must_use]
pub fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let workers = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
    let chunk = n.div_ceil(workers).max(1);
    let f = &f;
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|i0| sc.spawn(move || (i0..n.min(i0 + chunk)).map(f).collect::<Vec<T>>()))
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// The values of `v` with a comma between each two — how a gate's line
/// prints a list of positions, rows or slots.
pub fn comma_list(v: &[impl ToString]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The value at quantile `q` of `v` (the `⌊q · (n − 1)⌉`-th smallest), a NaN-free slice; the median is
/// `quantile(v, 0.5)`. No values, or a `q` outside `[0, 1]`, is a named panic.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    assert!(
        !v.is_empty() && (0.0..=1.0).contains(&q),
        "quantile {q} of {} values",
        v.len()
    );
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[(q * (s.len() - 1) as f64).round() as usize]
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

/// The first index at which `a` and `b` differ in bits, or `None` when
/// equal (a length that differs is index 0).
pub fn first_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(0);
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

/// One clause line for `a` against `b`: how many values and whether they are
/// equal bit for bit ([`first_diff`]); whether they are.
pub fn equal(what: &str, a: &[f32], b: &[f32]) -> bool {
    let d = first_diff(a, b);
    println!(
        "    {what}: {} values {}",
        a.len(),
        d.map_or_else(
            || "equal bit for bit".to_string(),
            |i| format!("differ from value {i}")
        )
    );
    d.is_none()
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

/// The rotation row of the position triple `(t, h, w)` from the plain table
/// `plain` (`ROT` values a position, a cos and a sin per pair), by ggml's
/// interleaved M-RoPE written out: pair `s` of sector `s mod 32` takes the h
/// position at sectors 1 mod 3 below 33, the w position at 2 mod 3 below 30
/// and the t position at 0 mod 3 below 33.
pub fn oracle_row<const ROT: usize>(plain: &[f32], (t, h, w): (usize, usize, usize)) -> [f32; ROT] {
    let mut row = [0.0; ROT];
    for s in 0..ROT / 2 {
        let sector = s % 32;
        let q = if sector % 3 == 1 && sector < 33 {
            h
        } else if sector % 3 == 2 && sector < 30 {
            w
        } else if sector % 3 == 0 && sector < 33 {
            t
        } else {
            0
        };
        row[2 * s..2 * s + 2].copy_from_slice(&plain[q * ROT + 2 * s..q * ROT + 2 * s + 2]);
    }
    row
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
    use super::{
        GateError, GgmlType, NAN_F16, bench_arm, bits_equal, dequant_row, exit_status,
        failure_status, first_ids, kquant_d_at, max_rel_err, quantile,
    };

    /// A failed run ends 1, and 75 when the need hook refused its load,
    /// whatever error carried the refusal up (a message that lost the hook's
    /// line too); a run that did not fail ends 0.
    #[test]
    fn a_refused_load_ends_75() {
        use std::process::ExitCode;
        let boom = || -> Result<(), GateError> { Err("the load failed".into()) };
        assert_eq!(exit_status("t", Ok(()), None), ExitCode::SUCCESS);
        assert_eq!(exit_status("t", boom(), None), ExitCode::FAILURE);
        let refusal =
            Some("the need hook /h refused the load (exit 75); it said: busy".to_string());
        assert_eq!(exit_status("t", boom(), refusal), ExitCode::from(75));
        assert_eq!(failure_status(false), ExitCode::FAILURE);
        assert_eq!(failure_status(true), ExitCode::from(75));
    }

    #[test]
    fn a_bench_arm_must_select_an_existing_op() {
        let parse = |args: &[&str]| {
            bench_arm(
                args.iter().map(|s| (*s).to_owned()),
                ["mxfp8_4096", "q8_0_4096"].into_iter().map(str::to_owned),
            )
        };
        assert_eq!(parse(&[]).expect("all arms"), None);
        for prefix in ["mxfp8", "q8_0_4096", ""] {
            assert_eq!(
                parse(&["gate", "--bench-kernels", "--bench-arm", prefix]).expect("existing arm"),
                Some(prefix.to_owned())
            );
        }
        for (args, detail) in [
            (vec!["--bench-arm"], "wants an arm name"),
            (vec!["--bench-arm", "absent"], "names no arm"),
        ] {
            let err = parse(&args).expect_err("invalid arm").to_string();
            assert!(err.contains("--bench-arm") && err.contains(detail), "{err}");
        }
    }

    /// The corpus reader takes the first `n` ids and refuses a short file and a word that is no id
    /// by name. Mutant: a short file read as the ids it holds, or a bad word skipped.
    #[test]
    fn the_corpus_reader_takes_n_ids_and_refuses_the_rest() {
        let path = std::path::Path::new("corpus-prose.ids");
        assert_eq!(first_ids(path, "7 8\n9 10", 3).expect("ids"), [7, 8, 9]);
        let short = first_ids(path, "7 8", 3).expect_err("short").to_string();
        assert!(short.contains("2 ids, the gate reads 3"), "{short}");
        assert!(first_ids(path, "7 x 9", 3).is_err());
    }

    /// The quantile is the `⌊q · (n − 1)⌉`-th smallest of the sorted values, a half rounding up: ten values read the
    /// median at the 5th index and the 90th and 99th percentiles at the 8th and the last; an odd count reads the middle.
    /// No values and a `q` that is no fraction are refused by name, not read as an index.
    #[test]
    fn the_quantile_is_the_rounded_rank_and_refuses_no_values() {
        let ten = [5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 7.0, 8.0, 6.0, 0.0];
        let at = |q| quantile(&ten, q);
        assert_eq!(
            (at(0.0), at(0.5), at(0.9), at(0.99), at(1.0)),
            (0.0, 5.0, 8.0, 9.0, 9.0)
        );
        assert_eq!(quantile(&[3.0, 1.0, 2.0, 5.0, 4.0], 0.5), 3.0);
        assert_eq!(quantile(&[7.0], 0.9), 7.0);
        let named = |r: std::thread::Result<f64>| {
            r.is_err_and(|e| {
                e.downcast_ref::<String>()
                    .is_some_and(|m| m.starts_with("quantile "))
            })
        };
        assert!(named(std::panic::catch_unwind(|| quantile(&[], 0.5))));
        assert!(named(std::panic::catch_unwind(|| quantile(
            &[1.0],
            f64::NAN
        ))));
        assert!(named(std::panic::catch_unwind(|| quantile(
            &[1.0, 2.0],
            1.5
        ))));
    }

    /// The store and replay checks lean on [`bits_equal`]: a shorter side is
    /// a difference, `-0.0` is not `0.0`, and a NaN matches only its own bits.
    #[test]
    fn bits_equal_takes_the_length_and_every_bit() {
        let nan = f32::from_bits(0x7fc0_0001);
        assert!(bits_equal(&[1.0, -0.0, nan], &[1.0, -0.0, nan]));
        assert!(!bits_equal(&[1.0, -0.0], &[1.0]));
        assert!(!bits_equal(&[1.0], &[1.0, -0.0]));
        assert!(!bits_equal(&[-0.0], &[0.0]));
        assert!(!bits_equal(&[nan], &[f32::NAN]));
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
