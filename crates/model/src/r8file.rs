//! The r8 sidecar: one GGUF beside a model file that holds only the routed
//! experts' gate and up stacks, in the row-lane layout the host tier's tile
//! reads ([`qdot::repack_q3k_r8`]), under a private type id no other reader
//! sizes. This module is the one owner of the format: where the file lives
//! ([`sidecar_path`]), how it is written ([`convert`]), the identity check a
//! load runs ([`Sidecar::open`]), whether a load's host tier reads it
//! ([`HostR8::at_load`]) and the full comparison with its source
//! ([`verify`]).
//!
//! The file is GGUF v3 with architecture [`R8_ARCH`], `bloomery.r8.layout`
//! (u32, [`qdot::Q3K_R8_LAYOUT`]) and the source's identity:
//! `bloomery.r8.source.shards` (the shards' file names in split order),
//! `.shard_bytes` (each shard's length), `.header_sha256` (lowercase hex
//! sha256 of each shard's bytes before its data base), and per sidecar tensor,
//! in the sidecar's tensor order, `.head_sha256` and `.tail_sha256` (of the
//! source tensor's first and last 4 KiB). Each tensor keeps its source's
//! name, dims and byte count, is tagged [`Q3K_R8_TYPE`], and holds
//! `repack_q3k_r8` of the whole stack: `dims[1] × dims[2]` rows of `dims[0]`
//! values. An 8-row group occupies the same byte range in both layouts, so
//! every expert does too. There is no whole-tensor digest: the header goes out
//! before the data, and [`verify`] compares every byte with the source, which
//! the card upload reads anyway.

use std::collections::HashSet;
use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, WriteError, Writer};
use gguf::{
    GENERAL_ARCHITECTURE, GgmlType, Gguf, LoadError, PrivateType, Split, TensorInfo, Value, Weights,
};
use qdot::{Q3K_R8_LAYOUT, Q3K_R8_ROWS};

use crate::ModelError;
use crate::fileio::{self, sha256_hex};
use crate::placement::host_lock::{FileMapping, HostFile, drop_pages};

/// The sidecar's architecture string: a file with any other is not a sidecar.
pub const R8_ARCH: &str = "bloomery-r8";

/// The sidecar's tensor type id: bloomery's private range (1000 + the ggml
/// id of the layout's source type, Q3_K's 11), outside every ggml table, so
/// the strict readers and ggml's own refuse the file instead of reading its
/// bytes as another type's. Never 211: ik reads that as its `Q3_K_R4`.
pub const Q3K_R8_TYPE: u32 = 1011;

/// The table [`Sidecar::open`] opens with: an r8 group super-block of a row
/// is Q3_K's 256 values in 110 bytes.
const PRIVATE: [PrivateType; 1] = [PrivateType {
    id: Q3K_R8_TYPE,
    blck: 256,
    size: 110,
}];

const KEY_LAYOUT: &str = "bloomery.r8.layout";
const KEY_SHARDS: &str = "bloomery.r8.source.shards";
const KEY_SHARD_BYTES: &str = "bloomery.r8.source.shard_bytes";
const KEY_HEADERS: &str = "bloomery.r8.source.header_sha256";
const KEY_HEADS: &str = "bloomery.r8.source.head_sha256";
const KEY_TAILS: &str = "bloomery.r8.source.tail_sha256";

/// Bytes of a source tensor's head and of its tail that the identity hashes.
const SPAN: usize = 4096;

/// Why a sidecar is not written, not opened or not equal to its source. The
/// identity refusals carry the sidecar's path, what it recorded and what the
/// source is now.
#[derive(Debug, thiserror::Error)]
pub enum R8Error {
    #[error("no tensors to convert")]
    NoTensors,
    #[error("{}: no tensor {tensor} in the source", .path.display())]
    SourceMissing { path: PathBuf, tensor: String },
    #[error("{}: tensor {tensor} is {got:?}; a sidecar stack is Q3_K", .path.display())]
    SourceType {
        path: PathBuf,
        tensor: String,
        got: GgmlType,
    },
    #[error(
        "{}: tensor {tensor} has dims {dims:?}; a sidecar stack is [k, rows, experts] with k a multiple of 256 and rows of {}",
        .path.display(),
        Q3K_R8_ROWS
    )]
    Grid {
        path: PathBuf,
        tensor: String,
        dims: Vec<u64>,
    },
    #[error("{} exists; a sidecar is never overwritten", .path.display())]
    Exists { path: PathBuf },
    #[error("{} is locked by a conversion still running", .path.display())]
    Busy { path: PathBuf },
    #[error("{}: the sidecar needs {need} bytes, {free} are free", .path.display())]
    Space { path: PathBuf, need: u64, free: u64 },
    #[error("{}: {op}: {source}", .path.display())]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{}: {source}", .path.display())]
    Write { path: PathBuf, source: WriteError },
    #[error("{}: {source}", .path.display())]
    Open { path: PathBuf, source: LoadError },
    #[error("{}: architecture {got:?}, a sidecar's is {:?}", .path.display(), R8_ARCH)]
    Architecture { path: PathBuf, got: String },
    #[error("{}: row-lane layout {got}, this build reads {}", .path.display(), Q3K_R8_LAYOUT)]
    Layout { path: PathBuf, got: u32 },
    #[error("{}: {key} {detail}", .path.display())]
    Key {
        path: PathBuf,
        key: &'static str,
        detail: String,
    },
    #[error("{}: made from {recorded} shards, the source has {found}", .path.display())]
    ShardCount {
        path: PathBuf,
        recorded: usize,
        found: usize,
    },
    #[error("{}: shard {shard} was {recorded:?}, the source's is {found:?}", .path.display())]
    ShardName {
        path: PathBuf,
        shard: usize,
        recorded: String,
        found: String,
    },
    #[error("{}: shard {shard} was {recorded} bytes, it is {found}", .path.display())]
    ShardBytes {
        path: PathBuf,
        shard: String,
        recorded: u64,
        found: u64,
    },
    #[error("{}: shard {shard}'s header sha256 was {recorded}, it is {found}", .path.display())]
    HeaderDigest {
        path: PathBuf,
        shard: String,
        recorded: String,
        found: String,
    },
    #[error("{}: tensor {tensor} appears twice", .path.display())]
    DuplicateTensor { path: PathBuf, tensor: String },
    #[error("{}: tensor {tensor} has type id {got}, a sidecar's are {}", .path.display(), Q3K_R8_TYPE)]
    TensorType {
        path: PathBuf,
        tensor: String,
        got: u32,
    },
    #[error(
        "{}: tensor {tensor} is {recorded_dims:?} ({recorded_bytes} bytes), the source's is {found_dims:?} ({found_bytes} bytes)",
        .path.display()
    )]
    Shape {
        path: PathBuf,
        tensor: String,
        recorded_dims: Vec<u64>,
        recorded_bytes: u64,
        found_dims: Vec<u64>,
        found_bytes: u64,
    },
    #[error("{}: tensor {tensor}'s first 4 KiB hashed {recorded}, the source's hash {found}", .path.display())]
    Head {
        path: PathBuf,
        tensor: String,
        recorded: String,
        found: String,
    },
    #[error("{}: tensor {tensor}'s last 4 KiB hashed {recorded}, the source's hash {found}", .path.display())]
    Tail {
        path: PathBuf,
        tensor: String,
        recorded: String,
        found: String,
    },
    #[error("{}: no tensor {tensor} in the sidecar", .path.display())]
    NotInSidecar { path: PathBuf, tensor: String },
    #[error(
        "{}: the sidecar holds {tensor}, a layer's down stack; a host tier reads only a gate and an up from it",
        .path.display()
    )]
    HoldsDown { path: PathBuf, tensor: String },
    #[error(
        "{}: tensor {tensor} unpacks to other bytes than the source's, first at expert {expert} byte {byte}",
        .path.display()
    )]
    Mismatch {
        path: PathBuf,
        tensor: String,
        expert: u64,
        byte: u64,
    },
    #[error("the source split has no first shard to name the sidecar after")]
    NoFirstShard,
    #[error(
        "no r8 sidecar at {}: a gate that covers the r8 path reads one (just r8-sidecar); \
         BLOOMERY_R8=off runs it on the source",
        .path.display()
    )]
    NoSidecar { path: PathBuf },
    /// A value built beside one open sidecar, called with a pair that reads
    /// another open — of the same path or not — or none ([`R8Source`]).
    #[error(
        "{what}: built beside the sidecar {} open, called with a pair that reads {}",
        .held.display(),
        .read.as_ref().map_or_else(|| "none".to_string(), |p| format!("another open, {}", p.display()))
    )]
    OtherSidecar {
        what: &'static str,
        held: PathBuf,
        read: Option<PathBuf>,
    },
    /// A call that acts on the pair's sidecar, with a pair that reads none.
    #[error("{what}: the pair reads no sidecar")]
    PairReadsNone { what: &'static str },
    /// A page release of a resident copy, whose bytes are anonymous pages
    /// `MADV_DONTNEED` would zero.
    #[error(
        "{}: a resident copy, not its file's mapping: it has no file pages to release",
        .path.display()
    )]
    ResidentCopy { path: PathBuf },
}

fn io_err(path: &Path, op: &'static str, source: io::Error) -> ModelError {
    R8Error::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
    .into()
}

/// Where the sidecar of the model whose first shard is `first_shard` lives:
/// `<source dir>-r8/<stem>-r8.gguf`, the stem being the file name without
/// `.gguf` and without a split's `-00001-of-000NN`. A directory of its own,
/// so nothing that globs the source directory or finds shards by name ever
/// meets it. The rule is lexical while the source directory ends in a name;
/// one that does not — a bare file name's, `.`, `..` — is resolved on the
/// filesystem first (`canonicalize`), so every spelling of one directory
/// gives one sidecar and none gives a hidden `.-r8`. A path with no file
/// name is refused by name.
pub fn sidecar_path(first_shard: &Path) -> Result<PathBuf, R8Error> {
    let name = first_shard
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| R8Error::Io {
            path: first_shard.to_path_buf(),
            op: "name the sidecar",
            source: io::Error::new(io::ErrorKind::InvalidInput, "the path names no file"),
        })?;
    let dir = match first_shard.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let dir = if dir.file_name().is_some() {
        dir.to_path_buf()
    } else {
        std::fs::canonicalize(dir).map_err(|source| R8Error::Io {
            path: dir.to_path_buf(),
            op: "resolve the source directory",
            source,
        })?
    };
    let stem = name.strip_suffix(".gguf").unwrap_or(&name);
    let mut side = dir.into_os_string();
    side.push("-r8");
    Ok(PathBuf::from(side).join(format!("{}-r8.gguf", without_split(stem))))
}

/// `stem` without a trailing `-NNNNN-of-NNNNN`, llama.cpp's shard suffix.
fn without_split(stem: &str) -> &str {
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    let Some((head, count)) = stem.rsplit_once("-of-") else {
        return stem;
    };
    match head.rsplit_once('-') {
        Some((base, no)) if digits(no) && digits(count) => base,
        _ => stem,
    }
}

/// A source stack the sidecar takes: the shard that holds it, its header
/// entry, and its geometry — `rows` of `k` values per expert, `experts` of
/// them.
struct Stack<'s> {
    shard: &'s Gguf,
    info: &'s TensorInfo,
    k: usize,
    rows: usize,
    experts: usize,
}

impl<'s> Stack<'s> {
    fn name(&self) -> &str {
        &self.info.name
    }

    /// The stack's bytes in its shard's mapping.
    fn data(&self) -> Result<&'s [u8], ModelError> {
        Ok(self.shard.data(self.info)?)
    }
}

/// Source tensor `name` as a sidecar stack: present, Q3_K and on the grids
/// ([`grids`]). `path` names the file the check is for in the refusal.
fn stack<'s>(source: &'s Split, name: &str, path: &Path) -> Result<Stack<'s>, R8Error> {
    let missing = || R8Error::SourceMissing {
        path: path.to_path_buf(),
        tensor: name.to_string(),
    };
    let (i, info) = source.find(name).ok_or_else(missing)?;
    let shard = source.shard(i).ok_or_else(missing)?;
    let [k, rows, experts] = grids(path, name, info.ty, &info.dims)?;
    Ok(Stack {
        shard,
        info,
        k,
        rows,
        experts,
    })
}

/// A source stack of type `ty` and dims `dims` as the sidecar takes it: Q3_K
/// `[k, rows, experts]` with `k` on the 256-value grid and `rows` on the
/// 8-row one, so no group straddles two experts; `name` and `path` name it
/// in the refusal.
fn grids(path: &Path, name: &str, ty: GgmlType, dims: &[u64]) -> Result<[usize; 3], R8Error> {
    if ty != GgmlType::Q3_K {
        return Err(R8Error::SourceType {
            path: path.to_path_buf(),
            tensor: name.to_string(),
            got: ty,
        });
    }
    let sizes: Option<Vec<usize>> = dims.iter().map(|&d| usize::try_from(d).ok()).collect();
    match sizes.as_deref() {
        Some(&[k, rows, experts]) if k % 256 == 0 && rows % Q3K_R8_ROWS == 0 => {
            Ok([k, rows, experts])
        }
        _ => Err(R8Error::Grid {
            path: path.to_path_buf(),
            tensor: name.to_string(),
            dims: dims.to_vec(),
        }),
    }
}

/// A stack's head and tail, the bytes its identity hashes: its first and
/// last [`SPAN`] bytes, the whole tensor when it is shorter.
fn ends(data: &[u8]) -> [&[u8]; 2] {
    let n = data.len().min(SPAN);
    [&data[..n], &data[data.len() - n..]]
}

/// The path a split's refusals name: its first shard.
fn first_shard(source: &Split) -> Result<PathBuf, R8Error> {
    source
        .shard_path(0)
        .map(Path::to_path_buf)
        .ok_or(R8Error::NoFirstShard)
}

/// One tensor done: its name, its bytes and the seconds it took.
#[derive(Clone, Debug)]
pub struct TensorStat {
    pub name: String,
    pub bytes: u64,
    pub secs: f64,
}

/// What [`convert`] and [`verify`] report as they go, for a caller that
/// prints progress.
#[derive(Debug)]
pub enum Progress<'a> {
    /// A `<out>.part` that a run which did not finish left, now removed.
    RemovedPart(&'a Path),
    /// One tensor written, or checked.
    Tensor(&'a TensorStat),
}

/// A finished [`convert`]: the sidecar, its tensors, its length and the
/// seconds the whole call took.
#[derive(Clone, Debug)]
pub struct ConvertStats {
    pub out: PathBuf,
    pub tensors: Vec<TensorStat>,
    pub file_bytes: u64,
    pub secs: f64,
}

/// A finished [`verify`]: its tensors, their bytes and the seconds it took.
#[derive(Clone, Debug)]
pub struct VerifyStats {
    pub tensors: Vec<TensorStat>,
    pub bytes: u64,
    pub secs: f64,
}

/// Write the sidecar of `names` — stacks of `source`, in that order — to
/// `out`. A name the source lacks, a stack that is not Q3_K or not on the
/// grids, an `out` that exists (a sidecar is never overwritten) and a
/// filesystem without room for the whole file are named errors before
/// anything is written. The bytes go to `<out>.part`, locked for the run,
/// which becomes `out` only once complete and synced, so a killed run never
/// leaves a file at `out`; a `.part` a finished-or-killed run left is removed
/// and reported, one a live run holds is [`R8Error::Busy`]. Each stack is
/// repacked on every core into one buffer, written, synced and dropped from
/// the page cache, so the conversion does not push the source's pages out of
/// it. After a failed write the `.part` is removed.
pub fn convert(
    source: &Split,
    names: &[String],
    out: &Path,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<ConvertStats, ModelError> {
    let t0 = Instant::now();
    if names.is_empty() {
        return Err(R8Error::NoTensors.into());
    }
    let first = first_shard(source)?;
    let stacks = names
        .iter()
        .map(|n| stack(source, n, &first))
        .collect::<Result<Vec<_>, _>>()?;
    if out.try_exists().map_err(|e| io_err(out, "stat", e))? {
        return Err(R8Error::Exists {
            path: out.to_path_buf(),
        }
        .into());
    }
    let layout = sidecar_layout(source, &stacks, out)?;
    let dir = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(dir).map_err(|e| io_err(dir, "create the directory", e))?;
    let mut part = out.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    remove_leftover(&part, progress)?;
    let free = fileio::free_bytes(dir).map_err(|e| io_err(dir, "statvfs", e))?;
    if free < layout.file_len() {
        return Err(R8Error::Space {
            path: dir.to_path_buf(),
            need: layout.file_len(),
            free,
        }
        .into());
    }
    let file_bytes = layout.file_len();
    let file = create_part(&part)?;
    let tensors = match write_stacks(&stacks, layout, file.file(), &part, progress) {
        Ok(t) => t,
        Err(e) => {
            // The error is what the caller acts on; a .part this removal
            // misses is the next run's leftover, removed and reported there.
            let _ = std::fs::remove_file(&part);
            return Err(e);
        }
    };
    publish(file, &part, out, dir)?;
    Ok(ConvertStats {
        out: out.to_path_buf(),
        tensors,
        file_bytes,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// The sidecar's header: the identity pairs — the digests of what
/// [`Inputs::of`] collects of `source` for `stacks`, the collector [`check`]
/// compares against — and one declaration per stack.
fn sidecar_layout(source: &Split, stacks: &[Stack<'_>], out: &Path) -> Result<Layout, ModelError> {
    let now = Inputs::of(source, stacks.iter().map(Stack::name))?;
    let mut heads = Vec::with_capacity(stacks.len());
    let mut tails = Vec::with_capacity(stacks.len());
    for s in &now.stacks {
        let s = s
            .as_ref()
            .expect("every stack convert takes is the source's");
        heads.push(Value::String(sha256_hex(s.head)));
        tails.push(Value::String(sha256_hex(s.tail)));
    }
    let hex = |b: &[u8]| Value::String(sha256_hex(b));
    let kvs = [
        (GENERAL_ARCHITECTURE, Value::String(R8_ARCH.to_string())),
        (KEY_LAYOUT, Value::U32(Q3K_R8_LAYOUT)),
        (
            KEY_SHARDS,
            Value::Array(
                now.shards
                    .iter()
                    .map(|s| Value::String(s.name.clone()))
                    .collect(),
            ),
        ),
        (
            KEY_SHARD_BYTES,
            Value::Array(now.shards.iter().map(|s| Value::U64(s.len)).collect()),
        ),
        (
            KEY_HEADERS,
            Value::Array(now.shards.iter().map(|s| hex(s.header)).collect()),
        ),
        (KEY_HEADS, Value::Array(heads)),
        (KEY_TAILS, Value::Array(tails)),
    ]
    .map(|(k, v)| (k.to_string(), v));
    let decls = stacks
        .iter()
        .map(|s| TensorDecl {
            name: s.name().to_string(),
            dims: s.info.dims.clone(),
            type_id: Q3K_R8_TYPE,
            nbytes: s.info.nbytes,
        })
        .collect();
    Layout::new(&kvs, decls).map_err(|source| {
        R8Error::Write {
            path: out.to_path_buf(),
            source,
        }
        .into()
    })
}

/// A leftover `.part` is removed and reported, unless a live run holds its
/// lock.
fn remove_leftover(part: &Path, progress: &mut dyn FnMut(Progress<'_>)) -> Result<(), ModelError> {
    let held = match File::open(part) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_err(part, "open", e)),
    };
    let _held = PartLock::take(held, part)?;
    std::fs::remove_file(part).map_err(|e| io_err(part, "remove", e))?;
    progress(Progress::RemovedPart(part));
    Ok(())
}

/// A `.part` file under this run's lock (`flock`, which belongs to the open
/// file description). Its drop unlocks before the descriptor closes: a child
/// another thread forks holds a copy of every descriptor until its exec, so a
/// lock left to the close would outlive this value by that window and refuse
/// the next run as [`R8Error::Busy`]; an unlock releases the description's
/// lock whatever copies of it exist.
struct PartLock(File);

impl PartLock {
    /// `file`, the `.part` at `part`, locked; a live run's lock is
    /// [`R8Error::Busy`].
    fn take(file: File, part: &Path) -> Result<PartLock, ModelError> {
        match file.try_lock() {
            Ok(()) => Ok(PartLock(file)),
            Err(TryLockError::WouldBlock) => Err(R8Error::Busy {
                path: part.to_path_buf(),
            }
            .into()),
            Err(TryLockError::Error(e)) => Err(io_err(part, "lock", e)),
        }
    }

    fn file(&self) -> &File {
        &self.0
    }
}

impl Drop for PartLock {
    fn drop(&mut self) {
        // A failed unlock leaves the release to the close, as before the
        // unlock existed; a drop has no one to report it to.
        let _ = self.0.unlock();
    }
}

/// `<out>.part`, created fresh and locked for this run.
fn create_part(part: &Path) -> Result<PartLock, ModelError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(part)
        .map_err(|e| io_err(part, "create", e))?;
    PartLock::take(file, part)
}

/// Every stack repacked, written, synced and dropped from the page cache,
/// in order; one line of progress each.
fn write_stacks(
    stacks: &[Stack<'_>],
    layout: Layout,
    file: &File,
    part: &Path,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<Vec<TensorStat>, ModelError> {
    let write_err = |source| -> ModelError {
        R8Error::Write {
            path: part.to_path_buf(),
            source,
        }
        .into()
    };
    let mut largest = 0;
    for s in stacks {
        largest = largest.max(s.data()?.len());
    }
    let mut buf = vec![0u8; largest];
    let mut w = Writer::new(file, layout).map_err(write_err)?;
    let threads = workers();
    let mut stats = Vec::with_capacity(stacks.len());
    for s in stacks {
        let t = Instant::now();
        let src = s.data()?;
        let dst = &mut buf[..src.len()];
        repack(s, src, dst, threads)?;
        let range = w.tensor(s.name(), dst).map_err(write_err)?;
        sync_and_drop(file, part, range)?;
        let stat = TensorStat {
            name: s.name().to_string(),
            bytes: s.info.nbytes,
            secs: t.elapsed().as_secs_f64(),
        };
        progress(Progress::Tensor(&stat));
        stats.push(stat);
    }
    w.finish().map_err(write_err)?;
    Ok(stats)
}

/// Threads the parallel passes run on: every hardware thread.
fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// `repack_q3k_r8` of stack `s` from `src` into `dst`, experts split evenly
/// over `threads` scoped threads: a group never straddles two experts, so
/// each thread's range repacks on its own.
fn repack(s: &Stack<'_>, src: &[u8], dst: &mut [u8], threads: usize) -> Result<(), ModelError> {
    let expert = src.len() / s.experts;
    let row = expert / s.rows;
    let per = s.experts.div_ceil(threads).max(1) * expert;
    std::thread::scope(|scope| {
        let parts: Vec<_> = src
            .chunks(per)
            .zip(dst.chunks_mut(per))
            .map(|(a, b)| scope.spawn(move || qdot::repack_q3k_r8(a, a.len() / row, s.k, b)))
            .collect();
        parts
            .into_iter()
            .try_for_each(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
    })?;
    Ok(())
}

/// A file offset or length as the page-cache calls take it.
fn off(path: &Path, v: u64) -> Result<libc::off_t, ModelError> {
    libc::off_t::try_from(v).map_err(|_| {
        io_err(
            path,
            "offset",
            io::Error::new(io::ErrorKind::InvalidInput, format!("{v} passes off_t")),
        )
    })
}

/// Written bytes `range` of `file` out to the drive, then out of the page
/// cache.
fn sync_and_drop(file: &File, path: &Path, range: Range<u64>) -> Result<(), ModelError> {
    let (at, len) = (off(path, range.start)?, off(path, range.end - range.start)?);
    let flags = libc::SYNC_FILE_RANGE_WAIT_BEFORE
        | libc::SYNC_FILE_RANGE_WRITE
        | libc::SYNC_FILE_RANGE_WAIT_AFTER;
    // SAFETY: the descriptor is `file`'s own and lives across the call; the
    // call reads no memory of ours.
    if unsafe { libc::sync_file_range(file.as_raw_fd(), at, len, flags) } != 0 {
        return Err(io_err(path, "sync_file_range", io::Error::last_os_error()));
    }
    drop_cached(file, path, range)
}

/// `posix_fadvise(DONTNEED)` over `range`: the kernel drops the clean pages
/// that lie wholly inside it and no process maps.
fn drop_cached(file: &File, path: &Path, range: Range<u64>) -> Result<(), ModelError> {
    let (at, len) = (off(path, range.start)?, off(path, range.end - range.start)?);
    // SAFETY: as in `sync_and_drop`: `file`'s own descriptor, no memory read.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), at, len, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        return Err(io_err(
            path,
            "posix_fadvise(DONTNEED)",
            io::Error::from_raw_os_error(rc),
        ));
    }
    Ok(())
}

/// The finished `.part` becomes `out`: synced, still the file this run wrote
/// (a run that removed it in the moment before this one locked it would
/// otherwise be linked in), linked without replacing anything, unlinked, and
/// the directory synced. The lock goes with `lock`, after the link.
fn publish(lock: PartLock, part: &Path, out: &Path, dir: &Path) -> Result<(), ModelError> {
    let file = lock.file();
    file.sync_all().map_err(|e| io_err(part, "fsync", e))?;
    let ours = file.metadata().map_err(|e| io_err(part, "stat", e))?;
    let named = std::fs::metadata(part).map_err(|e| io_err(part, "stat", e))?;
    if (ours.dev(), ours.ino()) != (named.dev(), named.ino()) {
        return Err(R8Error::Busy {
            path: part.to_path_buf(),
        }
        .into());
    }
    std::fs::hard_link(part, out).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => R8Error::Exists {
            path: out.to_path_buf(),
        }
        .into(),
        _ => io_err(out, "link", e),
    })?;
    std::fs::remove_file(part).map_err(|e| io_err(part, "remove", e))?;
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io_err(dir, "fsync", e))?;
    drop(lock);
    Ok(())
}

/// An open sidecar that matched its source at open: its tensors' names and
/// bytes, and what its check read of that source.
pub struct Sidecar {
    path: PathBuf,
    gguf: Gguf,
    /// The bytes live in the file's own mapping, whose pages [`verify`] drops
    /// after it; a resident copy has no file pages mapped.
    mapped: bool,
    /// What the check this sidecar passed read of its source ([`Inputs`]).
    seen: Seen,
    /// The file at `path` when it was opened: device and inode.
    id: (u64, u64),
}

impl Sidecar {
    /// Open the sidecar at `path` and check it against `source`, reading
    /// headers and 8 KiB of each tensor's source, never the stacks: the
    /// architecture, the layout version, the shard count, names, lengths and
    /// header digests, and for every tensor its type id, its source's
    /// presence as a Q3_K stack on the grids with the same dims and bytes, and
    /// the digests of that stack's head and tail. Each difference is its own
    /// [`R8Error`]. The check runs on a lazy mapping; when `weights` asks for
    /// another backing, the file is opened again that way and checked again,
    /// so a refused file never costs its data's pages. The sidecar keeps what
    /// the last check read of `source`, which [`Sidecar::pairs`] compares a
    /// split with.
    pub fn open(
        path: impl AsRef<Path>,
        source: &Split,
        weights: Weights,
    ) -> Result<Sidecar, ModelError> {
        let path = path.as_ref();
        let id = file_id(path)?;
        let lazy = Weights::Mapped { populate: false };
        let mut gguf = open_sidecar(path, lazy)?;
        let mut now = Inputs::of(source, tensor_names(&gguf))?;
        check(path, &gguf, &now)?;
        if weights != lazy {
            gguf = open_sidecar(path, weights)?;
            now = Inputs::of(source, tensor_names(&gguf))?;
            check(path, &gguf, &now)?;
        }
        Ok(Sidecar {
            path: path.to_path_buf(),
            seen: now.to_seen(),
            gguf,
            mapped: matches!(weights, Weights::Mapped { .. }),
            id,
        })
    }

    /// Whether this is `source`'s sidecar, as its open found it: what
    /// `source` gives the identity check ([`Inputs`]) is compared with what
    /// the check this sidecar passed read; on any difference the check runs
    /// again against `source`, and its refusal names the difference. A
    /// reader takes a sidecar beside a split only as an [`R8Source`], which
    /// this check makes, so a sidecar opened for one model never serves
    /// another's split.
    pub(crate) fn pairs(&self, source: &Split) -> Result<(), ModelError> {
        let now = Inputs::of(source, tensor_names(&self.gguf))?;
        if !self.seen.gives(&now) {
            check(&self.path, &self.gguf, &now)?;
        }
        Ok(())
    }

    /// The sidecar's tensor names, in file order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        tensor_names(&self.gguf)
    }

    /// The file the sidecar was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The sidecar's bytes as its file's own read-only mapping; `None` for a
    /// resident copy, whose pages are anonymous — `MADV_DONTNEED` would zero
    /// them.
    pub(crate) fn file_pages(&self) -> Option<FileMapping<'_>> {
        // SAFETY: `mapped` is set only for a `Weights::Mapped` open, whose
        // `Gguf` keeps the file's own read-only shared mapping for as long as
        // this sidecar lives.
        self.mapped
            .then(|| unsafe { FileMapping::new(self.gguf.mapping()) })
    }

    /// Tensor `name`'s header entry, `None` for a name the sidecar lacks.
    pub(crate) fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.gguf.find(name)
    }

    /// The file offsets of tensor `name`'s bytes, from the header alone;
    /// `None` for a name the sidecar lacks.
    pub fn file_bytes(&self, name: &str) -> Option<Range<u64>> {
        self.find(name).map(|t| {
            let start = self.gguf.data_base() + t.offset;
            start..start + t.nbytes
        })
    }

    /// Tensor `name`'s header entry; a name the sidecar lacks is refused by
    /// name.
    pub(crate) fn tensor(&self, name: &str) -> Result<&TensorInfo, ModelError> {
        Ok(self.find(name).ok_or_else(|| R8Error::NotInSidecar {
            path: self.path.clone(),
            tensor: name.to_string(),
        })?)
    }

    /// Tensor `name`'s bytes, in the row-lane layout.
    pub fn data(&self, name: &str) -> Result<&[u8], ModelError> {
        self.data_of(self.tensor(name)?)
    }

    /// The bytes of `t`, one of this sidecar's own header entries
    /// ([`Sidecar::tensor`]), in the row-lane layout.
    pub(crate) fn data_of(&self, t: &TensorInfo) -> Result<&[u8], ModelError> {
        Ok(self.gguf.data(t)?)
    }

    /// The open file: its mapping and data base, for a walk over the pages
    /// a host set names.
    pub(crate) fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// Drop tensor `t`'s whole pages from this process's mapping (a file
    /// mapping only) and then from the page cache, which skips a mapped page.
    fn release(&self, file: &File, t: &TensorInfo) -> Result<(), ModelError> {
        let start = self.gguf.data_base() + t.offset;
        let name = HostFile::Sidecar(self.path.clone());
        drop_pages(&name, file, self.file_pages(), start..start + t.nbytes)?;
        Ok(())
    }
}

impl fmt::Debug for Sidecar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sidecar")
            .field("path", &self.path)
            .field("mapped", &self.mapped)
            .finish_non_exhaustive()
    }
}

/// Where a host tier reads its routed gates and ups: the source's sidecar,
/// or the source itself and why ([`HostR8::at_load`]).
#[derive(Clone, Debug)]
pub enum HostR8 {
    /// The sidecar, checked against the source.
    On(Arc<Sidecar>),
    /// `BLOOMERY_R8=off`.
    Lever,
    /// No file at the sidecar's path.
    Missing(PathBuf),
}

impl HostR8 {
    /// The reading for `source` at load. `r8` is the load's `BLOOMERY_R8`:
    /// `false` reads the source (the same-binary A/B arm); `true` reads the
    /// file at [`sidecar_path`] of the source's first shard. No file is not
    /// an error: the tier reads the source. A file that does not match the
    /// source is its named [`R8Error`] ([`Sidecar::open`]), never a
    /// fall-back to the source. While any holder keeps a sidecar open, a
    /// second reading of the same file — the same device and inode at the
    /// path — hands back the same one, paired with this source
    /// ([`Sidecar::pairs`]), so the pages a load populates and locks are the
    /// pages its host tier reads; a file replaced at the path is opened anew.
    /// Each distinct reading prints one `load host_tier r8=…` line per
    /// process.
    pub fn at_load(source: &Split, r8: bool) -> Result<HostR8, ModelError> {
        let r8 = if r8 {
            let path = sidecar_path(&first_shard(source)?)?;
            if path.try_exists().map_err(|e| io_err(&path, "stat", e))? {
                HostR8::On(shared(&path, source)?)
            } else {
                HostR8::Missing(path)
            }
        } else {
            HostR8::Lever
        };
        announce(&r8);
        Ok(r8)
    }

    /// [`HostR8::at_load`] for a gate that covers the r8 path: no file at the
    /// sidecar's path is refused by name ([`R8Error::NoSidecar`]) instead of
    /// read as the source, so such a gate never passes source against
    /// source unasked. `r8 = false` (`BLOOMERY_R8=off`) reads the source, as
    /// the load does.
    pub fn at_gate(source: &Split, r8: bool) -> Result<HostR8, ModelError> {
        match HostR8::at_load(source, r8)? {
            HostR8::Missing(path) => Err(R8Error::NoSidecar { path }.into()),
            read => Ok(read),
        }
    }

    /// The sidecar the tier reads, when it reads one.
    pub fn sidecar(&self) -> Option<&Arc<Sidecar>> {
        match self {
            HostR8::On(s) => Some(s),
            HostR8::Lever | HostR8::Missing(_) => None,
        }
    }
}

/// The load line's words: `r8=on (<path>)`, `r8=off (BLOOMERY_R8=off)` or
/// `r8=off (no sidecar at <path>: just r8-sidecar)`.
impl fmt::Display for HostR8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostR8::On(s) => write!(f, "r8=on ({})", s.path.display()),
            HostR8::Lever => write!(f, "r8=off (BLOOMERY_R8=off)"),
            HostR8::Missing(p) => {
                write!(f, "r8=off (no sidecar at {}: just r8-sidecar)", p.display())
            }
        }
    }
}

/// A split and the r8 sidecar a host tier reads beside it, as one value:
/// made only by the pairing check ([`R8Source::of`], [`R8Pair::at_load`]) or
/// with no sidecar ([`R8Source::rows`]), so every sidecar that reaches a
/// host reader — a layer's call, a host set's walk and lock, a page release
/// — is the checked partner of the split beside it. The readers take this,
/// never a `&Split`:
///
/// ```compile_fail,E0308
/// # fn call(
/// #     layer: &model::moe::HostLayer,
/// #     split: &gguf::Split,
/// #     x: &model::ops::Tensor2,
/// #     out: &mut [f32],
/// #     scratch: &mut model::moe::HostScratch,
/// # ) {
/// // A split with no pairing check behind it.
/// let _ = layer.experts_into(split, x, &[], out, scratch);
/// # }
/// ```
///
/// ```compile_fail,E0308
/// # fn walk(set: &model::placement::host_lock::HostSet, split: &gguf::Split) {
/// let _ = set.populate(split);
/// # }
/// ```
///
/// A value built beside one open sidecar (a layer's r8 stacks, a host set's
/// sidecar pages) is refused by name, one pointer compare a call, when the
/// pair reads another open or none ([`R8Error::OtherSidecar`]).
#[derive(Clone, Copy)]
pub struct R8Source<'a> {
    split: &'a Split,
    sidecar: Option<&'a Arc<Sidecar>>,
}

impl<'a> R8Source<'a> {
    /// `split` alone: a host tier that reads every stack from the source.
    #[must_use]
    pub fn rows(split: &'a Split) -> R8Source<'a> {
        R8Source {
            split,
            sidecar: None,
        }
    }

    /// `split` beside the sidecar `r8` reads, when it reads one, checked
    /// once here ([`Sidecar::pairs`]): a sidecar that is not `split`'s is
    /// its named [`R8Error`].
    pub fn of(split: &'a Split, r8: &'a HostR8) -> Result<R8Source<'a>, ModelError> {
        let sidecar = r8.sidecar();
        if let Some(side) = sidecar {
            side.pairs(split)?;
        }
        Ok(R8Source { split, sidecar })
    }

    /// The source split.
    #[must_use]
    pub fn split(&self) -> &'a Split {
        self.split
    }

    /// The sidecar the pair reads, when it reads one.
    #[must_use]
    pub fn sidecar(&self) -> Option<&'a Arc<Sidecar>> {
        self.sidecar
    }

    /// `held`, the sidecar a value was built beside, is this pair's own open
    /// of it; `what` names the value in the refusal.
    pub(crate) fn reads(&self, held: &Arc<Sidecar>, what: &'static str) -> Result<(), R8Error> {
        match self.sidecar {
            Some(side) if Arc::ptr_eq(side, held) => Ok(()),
            read => Err(R8Error::OtherSidecar {
                what,
                held: held.path.clone(),
                read: read.map(|s| s.path.clone()),
            }),
        }
    }
}

/// The split by its first shard, and the sidecar by its path.
impl fmt::Debug for R8Source<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("R8Source")
            .field("split", &self.split.shard_path(0))
            .field("sidecar", &self.sidecar.map(|s| s.path()))
            .finish()
    }
}

/// An [`R8Source`] a holder keeps beside the split it owns: the split and
/// its host reading, checked against each other once, when the reading is
/// made ([`R8Pair::at_load`]); [`R8Pair::source`] lends the pair with no
/// check, so a step reads through it at no cost.
#[derive(Clone)]
pub struct R8Pair {
    split: Arc<Split>,
    r8: HostR8,
}

impl R8Pair {
    /// `split` and its host reading under `r8` ([`HostR8::at_load`], which
    /// checks the sidecar it hands back against this split).
    pub fn at_load(split: Arc<Split>, r8: bool) -> Result<R8Pair, ModelError> {
        let r8 = HostR8::at_load(&split, r8)?;
        Ok(R8Pair { split, r8 })
    }

    /// The pair, checked when it was made.
    #[must_use]
    pub fn source(&self) -> R8Source<'_> {
        R8Source {
            split: &self.split,
            sidecar: self.r8.sidecar(),
        }
    }

    /// The split.
    #[must_use]
    pub fn split(&self) -> &Arc<Split> {
        &self.split
    }

    /// The host reading.
    #[must_use]
    pub fn r8(&self) -> &HostR8 {
        &self.r8
    }
}

impl fmt::Debug for R8Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("R8Pair")
            .field("split", &self.split.shard_path(0))
            .field("r8", &self.r8)
            .finish()
    }
}

/// What the identity reads of a source for a sidecar: each shard's file
/// name, length and header bytes, and per sidecar tensor, in the sidecar's
/// order, the source's stack of that name — its type, dims and byte count
/// and its [`ends`] — or `None` where the source lacks it. [`convert`]
/// records the digests of these and [`check`] compares with them, reading
/// nothing of the source besides; so a source that gives the same inputs
/// gets the same verdict, and [`Sidecar::pairs`] compares them instead of
/// hashing again. `B` holds the bytes: borrowed from the source's mappings
/// (`&[u8]`) or kept by a sidecar ([`Seen`]).
struct Inputs<B> {
    shards: Vec<ShardIn<B>>,
    stacks: Vec<Option<StackIn<B>>>,
}

/// A shard as [`Inputs`] reads it.
struct ShardIn<B> {
    name: String,
    len: u64,
    header: B,
}

/// A source stack as [`Inputs`] reads it.
struct StackIn<B> {
    ty: GgmlType,
    dims: Vec<u64>,
    nbytes: u64,
    head: B,
    tail: B,
}

/// The inputs a sidecar's check read, kept by the sidecar.
type Seen = Inputs<Box<[u8]>>;

impl<'s> Inputs<&'s [u8]> {
    /// `source`'s inputs to the identity of the stacks `names`, borrowed
    /// from its mappings.
    fn of<'n>(
        source: &'s Split,
        names: impl IntoIterator<Item = &'n str>,
    ) -> Result<Self, ModelError> {
        let mut shards = Vec::with_capacity(source.shard_count());
        for s in 0..source.shard_count() {
            let shard = source
                .shard(s)
                .expect("a split has a reader for every shard index");
            let file = source
                .shard_path(s)
                .expect("a split has a path for every shard index");
            shards.push(ShardIn {
                name: file
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                len: shard.mapping().len() as u64,
                header: shard.header_bytes(),
            });
        }
        let mut stacks = Vec::new();
        for name in names {
            let Some((s, info)) = source.find(name) else {
                stacks.push(None);
                continue;
            };
            let data = source
                .shard(s)
                .expect("a split has a reader for every shard index")
                .data(info)?;
            let [head, tail] = ends(data);
            stacks.push(Some(StackIn {
                ty: info.ty,
                dims: info.dims.clone(),
                nbytes: info.nbytes,
                head,
                tail,
            }));
        }
        Ok(Inputs { shards, stacks })
    }

    /// A copy a sidecar keeps.
    fn to_seen(&self) -> Seen {
        let bytes = |b: &[u8]| -> Box<[u8]> { b.into() };
        Inputs {
            shards: self
                .shards
                .iter()
                .map(|s| ShardIn {
                    name: s.name.clone(),
                    len: s.len,
                    header: bytes(s.header),
                })
                .collect(),
            stacks: self
                .stacks
                .iter()
                .map(|s| {
                    s.as_ref().map(|s| StackIn {
                        ty: s.ty,
                        dims: s.dims.clone(),
                        nbytes: s.nbytes,
                        head: bytes(s.head),
                        tail: bytes(s.tail),
                    })
                })
                .collect(),
        }
    }
}

impl Seen {
    /// Whether `now` is, field for field and byte for byte, what this check
    /// read. Every field is named here, so a field added to the inputs is
    /// compared or does not compile.
    fn gives(&self, now: &Inputs<&[u8]>) -> bool {
        let shard = |a: &ShardIn<Box<[u8]>>, b: &ShardIn<&[u8]>| {
            let ShardIn { name, len, header } = a;
            *name == b.name && *len == b.len && **header == *b.header
        };
        let stack = |a: &StackIn<Box<[u8]>>, b: &StackIn<&[u8]>| {
            let StackIn {
                ty,
                dims,
                nbytes,
                head,
                tail,
            } = a;
            *ty == b.ty
                && *dims == b.dims
                && *nbytes == b.nbytes
                && **head == *b.head
                && **tail == *b.tail
        };
        let Inputs { shards, stacks } = self;
        shards.len() == now.shards.len()
            && stacks.len() == now.stacks.len()
            && shards.iter().zip(&now.shards).all(|(a, b)| shard(a, b))
            && stacks.iter().zip(&now.stacks).all(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => stack(a, b),
                (None, None) => true,
                _ => false,
            })
    }
}

/// A sidecar's tensor names, in file order.
fn tensor_names(g: &Gguf) -> impl Iterator<Item = &str> {
    g.iter_tensors().map(|t| t.name.as_str())
}

/// The file at `path`: its device and inode.
fn file_id(path: &Path) -> Result<(u64, u64), ModelError> {
    let m = std::fs::metadata(path).map_err(|e| io_err(path, "stat", e))?;
    Ok((m.dev(), m.ino()))
}

/// A sidecar a holder keeps open, and the path it was opened from.
struct Held {
    path: PathBuf,
    side: Weak<Sidecar>,
}

/// Every sidecar a holder keeps open.
static OPEN: Mutex<Vec<Held>> = Mutex::new(Vec::new());

/// The sidecar at `path`, paired with `source`: the one a holder keeps open
/// of the file now at `path` (the same device and inode — it was stat'ed
/// before its open, so a file replaced in between only costs a second open),
/// or a fresh lazy mapping, the source's own backing.
fn shared(path: &Path, source: &Split) -> Result<Arc<Sidecar>, ModelError> {
    let id = file_id(path)?;
    let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
    open.retain(|h| h.side.strong_count() > 0);
    let live = open
        .iter()
        .filter(|h| h.path == path)
        .find_map(|h| h.side.upgrade().filter(|s| s.id == id));
    if let Some(s) = live {
        s.pairs(source)?;
        return Ok(s);
    }
    let s = Arc::new(Sidecar::open(
        path,
        source,
        Weights::Mapped { populate: false },
    )?);
    open.push(Held {
        path: path.to_path_buf(),
        side: Arc::downgrade(&s),
    });
    Ok(s)
}

/// The reading's load line, once per process for each distinct line.
fn announce(r8: &HostR8) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let line = r8.to_string();
    let mut seen = SEEN.lock().unwrap_or_else(PoisonError::into_inner);
    if !seen.contains(&line) {
        eprintln!("load host_tier {line}");
        seen.push(line);
    }
}

fn open_sidecar(path: &Path, weights: Weights) -> Result<Gguf, ModelError> {
    Gguf::open_private(path, weights, &PRIVATE).map_err(|source| {
        R8Error::Open {
            path: path.to_path_buf(),
            source,
        }
        .into()
    })
}

fn key_err(path: &Path, key: &'static str, detail: String) -> R8Error {
    R8Error::Key {
        path: path.to_path_buf(),
        key,
        detail,
    }
}

fn value<'g>(path: &Path, g: &'g Gguf, key: &'static str) -> Result<&'g Value, R8Error> {
    g.value(key)
        .ok_or_else(|| key_err(path, key, "is absent".to_string()))
}

/// Array pair `key` of exactly `n` entries, each read by `item`.
fn entries<'g, T>(
    path: &Path,
    g: &'g Gguf,
    key: &'static str,
    n: usize,
    item: impl Fn(&'g Value) -> Option<T>,
) -> Result<Vec<T>, R8Error> {
    let Value::Array(items) = value(path, g, key)? else {
        return Err(key_err(path, key, "is not an array".to_string()));
    };
    if items.len() != n {
        return Err(key_err(
            path,
            key,
            format!("has {} entries for {n}", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, v)| item(v).ok_or_else(|| key_err(path, key, format!("entry {i} is {v:?}"))))
        .collect()
}

/// The identity check of `g`, the sidecar at `path`, against `now`, what a
/// source gives it ([`Inputs::of`] over `g`'s tensors). It reads nothing of
/// the source but `now`.
fn check(path: &Path, g: &Gguf, now: &Inputs<&[u8]>) -> Result<(), R8Error> {
    let arch = g.architecture();
    if arch != Some(R8_ARCH) {
        return Err(R8Error::Architecture {
            path: path.to_path_buf(),
            got: arch.unwrap_or("<missing>").to_string(),
        });
    }
    let layout = match value(path, g, KEY_LAYOUT)? {
        Value::U32(v) => *v,
        other => {
            return Err(key_err(
                path,
                KEY_LAYOUT,
                format!("is {other:?}, not a u32"),
            ));
        }
    };
    if layout != Q3K_R8_LAYOUT {
        return Err(R8Error::Layout {
            path: path.to_path_buf(),
            got: layout,
        });
    }
    let recorded = match value(path, g, KEY_SHARDS)? {
        Value::Array(items) => items.len(),
        _ => return Err(key_err(path, KEY_SHARDS, "is not an array".to_string())),
    };
    let names = entries(path, g, KEY_SHARDS, recorded, Value::as_str)?;
    let bytes = entries(path, g, KEY_SHARD_BYTES, recorded, |v| match v {
        Value::U64(b) => Some(*b),
        _ => None,
    })?;
    let headers = entries(path, g, KEY_HEADERS, recorded, Value::as_str)?;
    if recorded != now.shards.len() {
        return Err(R8Error::ShardCount {
            path: path.to_path_buf(),
            recorded,
            found: now.shards.len(),
        });
    }
    for (i, s) in now.shards.iter().enumerate() {
        if names[i] != s.name {
            return Err(R8Error::ShardName {
                path: path.to_path_buf(),
                shard: i,
                recorded: names[i].to_string(),
                found: s.name.clone(),
            });
        }
        if bytes[i] != s.len {
            return Err(R8Error::ShardBytes {
                path: path.to_path_buf(),
                shard: s.name.clone(),
                recorded: bytes[i],
                found: s.len,
            });
        }
        let digest = sha256_hex(s.header);
        if headers[i] != digest {
            return Err(R8Error::HeaderDigest {
                path: path.to_path_buf(),
                shard: s.name.clone(),
                recorded: headers[i].to_string(),
                found: digest,
            });
        }
    }
    let n = g.tensor_count();
    assert_eq!(
        now.stacks.len(),
        n,
        "the inputs of a check are collected over its sidecar's tensors"
    );
    let heads = entries(path, g, KEY_HEADS, n, Value::as_str)?;
    let tails = entries(path, g, KEY_TAILS, n, Value::as_str)?;
    let mut seen = HashSet::with_capacity(n);
    for ((i, t), s) in g.iter_tensors().enumerate().zip(&now.stacks) {
        let tensor = || t.name.clone();
        if !seen.insert(t.name.as_str()) {
            return Err(R8Error::DuplicateTensor {
                path: path.to_path_buf(),
                tensor: tensor(),
            });
        }
        if t.ty != GgmlType::Unknown(Q3K_R8_TYPE) {
            return Err(R8Error::TensorType {
                path: path.to_path_buf(),
                tensor: tensor(),
                got: t.ty.as_u32(),
            });
        }
        let s = s.as_ref().ok_or_else(|| R8Error::SourceMissing {
            path: path.to_path_buf(),
            tensor: tensor(),
        })?;
        grids(path, &t.name, s.ty, &s.dims)?;
        same_shape(path, t, &s.dims, s.nbytes)?;
        let (head, tail) = (sha256_hex(s.head), sha256_hex(s.tail));
        if head != heads[i] {
            return Err(R8Error::Head {
                path: path.to_path_buf(),
                tensor: tensor(),
                recorded: heads[i].to_string(),
                found: head,
            });
        }
        if tail != tails[i] {
            return Err(R8Error::Tail {
                path: path.to_path_buf(),
                tensor: tensor(),
                recorded: tails[i].to_string(),
                found: tail,
            });
        }
    }
    Ok(())
}

/// Sidecar tensor `t` has its source stack's dims and bytes.
fn same_shape(path: &Path, t: &TensorInfo, dims: &[u64], nbytes: u64) -> Result<(), R8Error> {
    if t.dims == dims && t.nbytes == nbytes {
        return Ok(());
    }
    Err(R8Error::Shape {
        path: path.to_path_buf(),
        tensor: t.name.clone(),
        recorded_dims: t.dims.clone(),
        recorded_bytes: t.nbytes,
        found_dims: dims.to_vec(),
        found_bytes: nbytes,
    })
}

/// Compare every sidecar tensor, unpacked, with its source stack, byte for
/// byte, experts split over every core; the first differing byte is
/// [`R8Error::Mismatch`], naming the tensor, the expert and the byte within
/// that expert's Q3_K bytes. Each tensor's sidecar pages are dropped after
/// it, as [`convert`] drops what it writes.
pub fn verify(
    source: &Split,
    sidecar: &Sidecar,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<VerifyStats, ModelError> {
    let t0 = Instant::now();
    let file = File::open(&sidecar.path).map_err(|e| io_err(&sidecar.path, "open", e))?;
    let threads = workers();
    let mut tensors = Vec::with_capacity(sidecar.gguf.tensor_count());
    for t in sidecar.gguf.iter_tensors() {
        let t1 = Instant::now();
        let s = stack(source, &t.name, &sidecar.path)?;
        same_shape(&sidecar.path, t, &s.info.dims, s.info.nbytes)?;
        let src = s.data()?;
        let r8 = sidecar.gguf.data(t)?;
        if let Some(at) = first_mismatch(&s, r8, src, threads)? {
            let expert = src.len() / s.experts;
            return Err(R8Error::Mismatch {
                path: sidecar.path.clone(),
                tensor: t.name.clone(),
                expert: (at / expert) as u64,
                byte: (at % expert) as u64,
            }
            .into());
        }
        sidecar.release(&file, t)?;
        let stat = TensorStat {
            name: t.name.clone(),
            bytes: t.nbytes,
            secs: t1.elapsed().as_secs_f64(),
        };
        progress(Progress::Tensor(&stat));
        tensors.push(stat);
    }
    let bytes = tensors.iter().map(|t| t.bytes).sum();
    Ok(VerifyStats {
        tensors,
        bytes,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// The first byte at which `r8`, unpacked an 8-row group at a time, differs
/// from `src`, as an offset into `src`. Experts split evenly over `threads`
/// scoped threads, each stopping at its own first difference; the ranges are
/// in order, so the smallest is the first.
fn first_mismatch(
    s: &Stack<'_>,
    r8: &[u8],
    src: &[u8],
    threads: usize,
) -> Result<Option<usize>, ModelError> {
    let expert = src.len() / s.experts;
    let group = expert / s.rows * Q3K_R8_ROWS;
    let per = s.experts.div_ceil(threads).max(1) * expert;
    let found = std::thread::scope(|scope| {
        let parts: Vec<_> = r8
            .chunks(per)
            .zip(src.chunks(per))
            .enumerate()
            .map(|(p, (a, b))| {
                scope.spawn(move || -> Result<Option<usize>, qdot::QdotError> {
                    let mut rows = vec![0u8; group];
                    for (g, (x, y)) in a.chunks_exact(group).zip(b.chunks_exact(group)).enumerate()
                    {
                        qdot::unpack_q3k_r8(x, Q3K_R8_ROWS, s.k, &mut rows)?;
                        if let Some(i) = rows.iter().zip(y).position(|(u, v)| u != v) {
                            return Ok(Some(p * per + g * group + i));
                        }
                    }
                    Ok(None)
                })
            })
            .collect();
        parts.into_iter().try_fold(None, |first: Option<usize>, h| {
            let at = h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))?;
            Ok::<_, qdot::QdotError>(first.or(at))
        })
    })?;
    Ok(found)
}
