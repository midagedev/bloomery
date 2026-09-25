//! The r8 sidecar: one GGUF beside a model file that holds only the routed
//! experts' gate and up stacks, in the row-lane layout the host tier's tile
//! reads ([`qdot::repack_q3k_r8`]), under a private type id no other reader
//! sizes. This module is the one owner of the format: where the file lives
//! ([`sidecar_path`]), how it is written ([`convert`]), the identity check a
//! load runs ([`Sidecar::open`]) and the full comparison with its source
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
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, WriteError, Writer};
use gguf::{
    GENERAL_ARCHITECTURE, GgmlType, Gguf, LoadError, PrivateType, Split, TensorInfo, Value, Weights,
};
use qdot::{Q3K_R8_LAYOUT, Q3K_R8_ROWS};
use sha2::{Digest, Sha256};

use crate::ModelError;
use crate::placement::host_lock::page_bytes;

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
        "{}: tensor {tensor} unpacks to other bytes than the source's, first at expert {expert} byte {byte}",
        .path.display()
    )]
    Mismatch {
        path: PathBuf,
        tensor: String,
        expert: u64,
        byte: u64,
    },
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
/// meets it. The rule is lexical: a bare file name's directory is `.`.
pub fn sidecar_path(first_shard: &Path) -> PathBuf {
    let dir = match first_shard.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let name = first_shard
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name.strip_suffix(".gguf").unwrap_or(&name);
    let mut side = dir.as_os_str().to_owned();
    side.push("-r8");
    PathBuf::from(side).join(format!("{}-r8.gguf", without_split(stem)))
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

/// Source tensor `name` as a sidecar stack: present, Q3_K and on the grids —
/// `[k, rows, experts]` with `k` on the 256-value grid and `rows` on the
/// 8-row one, so no group straddles two experts. `path` names the file the
/// check is for in the refusal.
fn stack<'s>(source: &'s Split, name: &str, path: &Path) -> Result<Stack<'s>, R8Error> {
    let missing = || R8Error::SourceMissing {
        path: path.to_path_buf(),
        tensor: name.to_string(),
    };
    let (i, info) = source.find(name).ok_or_else(missing)?;
    let shard = source.shard(i).ok_or_else(missing)?;
    if info.ty != GgmlType::Q3_K {
        return Err(R8Error::SourceType {
            path: path.to_path_buf(),
            tensor: name.to_string(),
            got: info.ty,
        });
    }
    let dims: Option<Vec<usize>> = info.dims.iter().map(|&d| usize::try_from(d).ok()).collect();
    match dims.as_deref() {
        Some(&[k, rows, experts]) if k % 256 == 0 && rows % Q3K_R8_ROWS == 0 => Ok(Stack {
            shard,
            info,
            k,
            rows,
            experts,
        }),
        _ => Err(R8Error::Grid {
            path: path.to_path_buf(),
            tensor: name.to_string(),
            dims: info.dims.clone(),
        }),
    }
}

/// Lowercase hex sha256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The source's shard identity, in split order: file names, lengths, and the
/// sha256 of each shard's bytes before its data base.
struct Shards {
    names: Vec<String>,
    bytes: Vec<u64>,
    headers: Vec<String>,
}

fn shards_of(source: &Split) -> Shards {
    let mut s = Shards {
        names: Vec::new(),
        bytes: Vec::new(),
        headers: Vec::new(),
    };
    for i in 0..source.shard_count() {
        let g = source
            .shard(i)
            .expect("a split has a reader for every shard index");
        let path = source
            .shard_path(i)
            .expect("a split has a path for every shard index");
        let map = g.mapping();
        // A shard that ends inside its data-base padding (it holds no tensors)
        // hashes to its end.
        let end = usize::try_from(g.data_base()).map_or(map.len(), |b| b.min(map.len()));
        s.names.push(
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        );
        s.bytes.push(map.len() as u64);
        s.headers.push(sha256_hex(&map[..end]));
    }
    s
}

/// The source stack's head and tail digests: sha256 of its first and last
/// [`SPAN`] bytes, the whole tensor when it is shorter.
fn head_tail(s: &Stack<'_>) -> Result<(String, String), ModelError> {
    let data = s.data()?;
    let n = data.len().min(SPAN);
    Ok((sha256_hex(&data[..n]), sha256_hex(&data[data.len() - n..])))
}

/// The path a split's refusals name: its first shard.
fn first_shard(source: &Split) -> PathBuf {
    source
        .shard_path(0)
        .map(Path::to_path_buf)
        .unwrap_or_default()
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
    let first = first_shard(source);
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
    let free = free_bytes(dir)?;
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
    let tensors = match write_stacks(&stacks, layout, &file, &part, progress) {
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

/// The sidecar's header: the identity pairs and one declaration per stack.
fn sidecar_layout(source: &Split, stacks: &[Stack<'_>], out: &Path) -> Result<Layout, ModelError> {
    let shards = shards_of(source);
    let mut heads = Vec::with_capacity(stacks.len());
    let mut tails = Vec::with_capacity(stacks.len());
    for s in stacks {
        let (h, t) = head_tail(s)?;
        heads.push(Value::String(h));
        tails.push(Value::String(t));
    }
    let strings = |v: Vec<String>| Value::Array(v.into_iter().map(Value::String).collect());
    let kvs = [
        (GENERAL_ARCHITECTURE, Value::String(R8_ARCH.to_string())),
        (KEY_LAYOUT, Value::U32(Q3K_R8_LAYOUT)),
        (KEY_SHARDS, strings(shards.names)),
        (
            KEY_SHARD_BYTES,
            Value::Array(shards.bytes.into_iter().map(Value::U64).collect()),
        ),
        (KEY_HEADERS, strings(shards.headers)),
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
    match held.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            return Err(R8Error::Busy {
                path: part.to_path_buf(),
            }
            .into());
        }
        Err(TryLockError::Error(e)) => return Err(io_err(part, "lock", e)),
    }
    std::fs::remove_file(part).map_err(|e| io_err(part, "remove", e))?;
    progress(Progress::RemovedPart(part));
    Ok(())
}

/// `<out>.part`, created fresh and locked for this run.
fn create_part(part: &Path) -> Result<File, ModelError> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(part)
        .map_err(|e| io_err(part, "create", e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(R8Error::Busy {
            path: part.to_path_buf(),
        }
        .into()),
        Err(TryLockError::Error(e)) => Err(io_err(part, "lock", e)),
    }
}

/// Bytes free to this process on the filesystem that holds `dir`.
fn free_bytes(dir: &Path) -> Result<u64, ModelError> {
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).map_err(|e| {
        io_err(
            dir,
            "statvfs",
            io::Error::new(io::ErrorKind::InvalidInput, e),
        )
    })?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c` is a NUL-terminated path that lives across the call, and
    // `st` is storage for one `statvfs`, which the call fills on success.
    if unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) } != 0 {
        return Err(io_err(dir, "statvfs", io::Error::last_os_error()));
    }
    // SAFETY: the call returned 0, so it wrote every field of `st`.
    let st = unsafe { st.assume_init() };
    Ok(st.f_bavail.saturating_mul(st.f_frsize))
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
/// the directory synced. The lock goes with `file`, after the link.
fn publish(file: File, part: &Path, out: &Path, dir: &Path) -> Result<(), ModelError> {
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
    drop(file);
    Ok(())
}

/// An open sidecar that matched its source at open: its tensors' names and
/// bytes, nothing else.
pub struct Sidecar {
    path: PathBuf,
    gguf: Gguf,
    /// The bytes live in the file's own mapping, whose pages [`verify`] drops
    /// after it; a resident copy has no file pages mapped.
    mapped: bool,
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
    /// so a refused file never costs its data's pages.
    pub fn open(
        path: impl AsRef<Path>,
        source: &Split,
        weights: Weights,
    ) -> Result<Sidecar, ModelError> {
        let path = path.as_ref();
        let lazy = Weights::Mapped { populate: false };
        let mut gguf = open_sidecar(path, lazy)?;
        check(path, &gguf, source)?;
        if weights != lazy {
            gguf = open_sidecar(path, weights)?;
            check(path, &gguf, source)?;
        }
        Ok(Sidecar {
            path: path.to_path_buf(),
            gguf,
            mapped: matches!(weights, Weights::Mapped { .. }),
        })
    }

    /// The sidecar's tensor names, in file order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.gguf.iter_tensors().map(|t| t.name.as_str())
    }

    /// Tensor `name`'s bytes, in the row-lane layout.
    pub fn data(&self, name: &str) -> Result<&[u8], ModelError> {
        let t = self.gguf.find(name).ok_or_else(|| R8Error::NotInSidecar {
            path: self.path.clone(),
            tensor: name.to_string(),
        })?;
        Ok(self.gguf.data(t)?)
    }

    /// Drop tensor `t`'s whole pages from this process's mapping (a file
    /// mapping only) and then from the page cache, which skips a mapped page.
    fn release(&self, file: &File, t: &TensorInfo) -> Result<(), ModelError> {
        let page = page_bytes();
        let start = self.gguf.data_base() + t.offset;
        let (first, end) = (
            start.div_ceil(page) * page,
            (start + t.nbytes) / page * page,
        );
        if end <= first {
            return Ok(());
        }
        if self.mapped {
            let span = usize::try_from(first)
                .ok()
                .zip(usize::try_from(end).ok())
                .and_then(|(a, b)| self.gguf.mapping().get(a..b))
                .ok_or_else(|| R8Error::NotInSidecar {
                    path: self.path.clone(),
                    tensor: t.name.clone(),
                })?;
            // SAFETY: `span` is a live, page-aligned sub-slice of a read-only
            // MAP_SHARED file mapping. MADV_DONTNEED drops this process's
            // page-table entries only; the next touch re-faults the same file
            // bytes, so no borrow of the mapping ever sees other contents.
            let rc = unsafe {
                libc::madvise(
                    span.as_ptr().cast_mut().cast(),
                    span.len(),
                    libc::MADV_DONTNEED,
                )
            };
            if rc != 0 {
                return Err(io_err(
                    &self.path,
                    "madvise(MADV_DONTNEED)",
                    io::Error::last_os_error(),
                ));
            }
        }
        drop_cached(file, &self.path, first..end)
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

/// [`Sidecar::open`]'s check of `g`, the sidecar at `path`, against `source`.
fn check(path: &Path, g: &Gguf, source: &Split) -> Result<(), ModelError> {
    let arch = g.architecture();
    if arch != Some(R8_ARCH) {
        return Err(R8Error::Architecture {
            path: path.to_path_buf(),
            got: arch.unwrap_or("<missing>").to_string(),
        }
        .into());
    }
    let layout = match value(path, g, KEY_LAYOUT)? {
        Value::U32(v) => *v,
        other => return Err(key_err(path, KEY_LAYOUT, format!("is {other:?}, not a u32")).into()),
    };
    if layout != Q3K_R8_LAYOUT {
        return Err(R8Error::Layout {
            path: path.to_path_buf(),
            got: layout,
        }
        .into());
    }
    let recorded = match value(path, g, KEY_SHARDS)? {
        Value::Array(items) => items.len(),
        _ => return Err(key_err(path, KEY_SHARDS, "is not an array".to_string()).into()),
    };
    let names = entries(path, g, KEY_SHARDS, recorded, Value::as_str)?;
    let bytes = entries(path, g, KEY_SHARD_BYTES, recorded, |v| match v {
        Value::U64(b) => Some(*b),
        _ => None,
    })?;
    let headers = entries(path, g, KEY_HEADERS, recorded, Value::as_str)?;
    if recorded != source.shard_count() {
        return Err(R8Error::ShardCount {
            path: path.to_path_buf(),
            recorded,
            found: source.shard_count(),
        }
        .into());
    }
    let now = shards_of(source);
    for (i, name) in names.iter().enumerate() {
        if *name != now.names[i] {
            return Err(R8Error::ShardName {
                path: path.to_path_buf(),
                shard: i,
                recorded: (*name).to_string(),
                found: now.names[i].clone(),
            }
            .into());
        }
        if bytes[i] != now.bytes[i] {
            return Err(R8Error::ShardBytes {
                path: path.to_path_buf(),
                shard: now.names[i].clone(),
                recorded: bytes[i],
                found: now.bytes[i],
            }
            .into());
        }
        if headers[i] != now.headers[i] {
            return Err(R8Error::HeaderDigest {
                path: path.to_path_buf(),
                shard: now.names[i].clone(),
                recorded: headers[i].to_string(),
                found: now.headers[i].clone(),
            }
            .into());
        }
    }
    let n = g.tensor_count();
    let heads = entries(path, g, KEY_HEADS, n, Value::as_str)?;
    let tails = entries(path, g, KEY_TAILS, n, Value::as_str)?;
    let mut seen = HashSet::with_capacity(n);
    for (i, t) in g.iter_tensors().enumerate() {
        let tensor = || t.name.clone();
        if !seen.insert(t.name.as_str()) {
            return Err(R8Error::DuplicateTensor {
                path: path.to_path_buf(),
                tensor: tensor(),
            }
            .into());
        }
        if t.ty != GgmlType::Unknown(Q3K_R8_TYPE) {
            return Err(R8Error::TensorType {
                path: path.to_path_buf(),
                tensor: tensor(),
                got: t.ty.as_u32(),
            }
            .into());
        }
        let s = stack(source, &t.name, path)?;
        same_shape(path, t, &s)?;
        let (head, tail) = head_tail(&s)?;
        if head != heads[i] {
            return Err(R8Error::Head {
                path: path.to_path_buf(),
                tensor: tensor(),
                recorded: heads[i].to_string(),
                found: head,
            }
            .into());
        }
        if tail != tails[i] {
            return Err(R8Error::Tail {
                path: path.to_path_buf(),
                tensor: tensor(),
                recorded: tails[i].to_string(),
                found: tail,
            }
            .into());
        }
    }
    Ok(())
}

/// Sidecar tensor `t` has its source stack's dims and bytes.
fn same_shape(path: &Path, t: &TensorInfo, s: &Stack<'_>) -> Result<(), R8Error> {
    if t.dims == s.info.dims && t.nbytes == s.info.nbytes {
        return Ok(());
    }
    Err(R8Error::Shape {
        path: path.to_path_buf(),
        tensor: t.name.clone(),
        recorded_dims: t.dims.clone(),
        recorded_bytes: t.nbytes,
        found_dims: s.info.dims.clone(),
        found_bytes: s.info.nbytes,
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
        same_shape(&sidecar.path, t, &s)?;
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
