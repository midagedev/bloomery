//! The r8 sidecar's refusal type and its two identity constants: the parts of
//! the format every reader of it names, here so the placement planner
//! ([`crate::placement::PlacementError::R8`]) carries a sidecar's refusal by
//! name. The format's one owner — where the file lives, how it is written,
//! the identity check a load runs and the full comparison — is
//! `bloomery-model`'s `r8file`, which re-exports this module: writing and
//! checking a sidecar releases pages (`posix_fadvise`) and repacks through
//! `qdot`, neither of which a pure crate may call.

use std::io;
use std::path::PathBuf;

use gguf::write::WriteError;
use gguf::{GgmlType, LoadError};

/// The sidecar's architecture string: a file with any other is not a sidecar.
pub const R8_ARCH: &str = "bloomery-r8";

/// The sidecar's tensor type id: bloomery's private range (1000 + the ggml
/// id of the layout's source type, Q3_K's 11), outside every ggml table, so
/// the strict readers and ggml's own refuse the file instead of reading its
/// bytes as another type's. Never 211: ik reads that as its `Q3_K_R4`.
pub const Q3K_R8_TYPE: u32 = 1011;

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
        "{}: tensor {tensor} has dims {dims:?}; a sidecar stack is [k, rows, experts] with k a multiple of 256 and rows of {rows}",
        .path.display()
    )]
    Grid {
        path: PathBuf,
        tensor: String,
        dims: Vec<u64>,
        /// The rows a sidecar stack's row groups take: `qdot::Q3K_R8_ROWS`,
        /// passed by the constructor (`r8file`), the value's one owner.
        rows: usize,
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
    #[error("{}: row-lane layout {got}, this build reads {reads}", .path.display())]
    Layout {
        path: PathBuf,
        got: u32,
        /// The row-lane layout this build reads: `qdot::Q3K_R8_LAYOUT`,
        /// passed by the constructor (`r8file`), the value's one owner.
        reads: u32,
    },
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
