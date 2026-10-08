//! The gate fixture generator, common to every family: a model file much
//! smaller than the real one that keeps every per-layer shape, so the
//! engine's compile-time constants and the layer kinds its launch table is
//! built from all hold, and whose whole weight set fits in host memory.
//!
//! What a family supplies is a [`FixtureSpec`]: its layer map (fixture layer
//! `f` takes source layer `spec.layers[f]`), the per-layer ratios that map
//! must read, its key rules, its special tables, an ff override and its draft
//! file. The fixture is the map's layers and the source's globals. Every
//! tensor keeps its source's dims and type id and is renamed `blk.L.` →
//! `blk.f.`; the exceptions are a tensor the family's [`Tables`] resize and a
//! routed stack the ff override narrows. The metadata is the source header's:
//! the `split.`, `general.`, `tokenizer.` and `quantize.` keys copied (the
//! last two are the format's own namespaces), each architecture key by the
//! family's [`KeyRule`] (a key it does not name is an error), and five keys of
//! its own: [`KEY_VERSION`], [`KEY_SEED`], [`KEY_SOURCE_LAYERS`],
//! [`KEY_SOURCE_SHA256`] and [`KEY_CARD_BUDGET`]. A
//! file holding only some of the planned tensors also carries [`KEY_SUBSET`];
//! the engine must not run such a file.
//!
//! Weights are random codes and scales written directly, no quantizer: every
//! matrix element has standard deviation `1/√K` (`K = dims[0]`), so a
//! projection of a unit-RMS input has unit RMS. [`rule_for`] derives each
//! type's scale from its block layout (the dequant ports in `gguf::quant`)
//! and picks the widest band of scale codes whose `d` (and `dmin`) is a
//! normal f16 in the spec's [`Window`]. A 1-D F32 tensor holds the constant
//! the family names for it.
//!
//! The bytes are a function of the seed and the source header alone: each
//! tensor's stream is cut into chunks of about [`CHUNK_TARGET`] bytes, and a
//! chunk's generator is keyed by (seed, tensor name, chunk index), so any
//! thread count writes the same file.
//!
//! A family with a draft file ([`DraftSpec`]) also gets a draft fixture: the
//! real draft's tensors at their shapes (its ff narrowed by the same override)
//! with random weights by the same rules, its metadata as the family's
//! [`DraftRules`] rewrite it, and the same five keys.
//!
//! A family whose host tier reads an r8 sidecar ([`SidecarSpec`]) gets one
//! beside its whole fixture, written once the fixture's files are in place
//! ([`generate`]) and checked byte for byte by [`verify`].
//!
//! A 1-D F32 tensor holds the family's constant, or, for a tensor whose
//! elements must differ (a router's selection bias), a uniform spread
//! ([`Family::spread_value`]); each tensor's stream is keyed by its name, so
//! two spread tensors of one shape differ.

use std::io;
use std::path::{Path, PathBuf};

use gguf::write::WriteError;
use gguf::{GgmlType, LoadError};

use crate::placement::PlacementError;

mod budget;
mod fill;
mod plan;
mod sidecar;
mod spec;
mod verify;
mod write;

pub use budget::check as check_budget;
pub use fill::{Band, CHUNK_TARGET, FloatTy, Rule, Window, rule_for};
pub use plan::{FilePlan, Kvs, Plan, PlannedTensor, header_sha256, plan};
pub(crate) use plan::{int_like, items, unsigned};
pub use sidecar::{SidecarStat, path_of as sidecar_path};
pub use spec::{
    Budget, CardBudget, CardExperts, DraftRules, DraftSpec, Family, FixtureSpec, KeyRule, Options,
    SidecarSpec, Tables,
};
pub use verify::{
    Sample, VerifyStats, check_tensor, check_units, rms_within, sample_chunks, verify,
};
pub use write::{GenerateStats, TensorStat, generate};

/// The format of the fixture's own keys.
pub const FIXTURE_VERSION: u32 = 1;
/// u32 [`FIXTURE_VERSION`]; its presence marks a fixture.
pub const KEY_VERSION: &str = "bloomery.fixture.version";
/// u64: the seed every weight stream is keyed by.
pub const KEY_SEED: &str = "bloomery.fixture.seed";
/// u32 array: the spec's layer map, the source layer of each fixture layer.
pub const KEY_SOURCE_LAYERS: &str = "bloomery.fixture.source_layers";
/// String: lowercase hex sha256 of the source's header bytes, every shard's
/// bytes before its data base, in split order.
pub const KEY_SOURCE_SHA256: &str = "bloomery.fixture.source_header_sha256";
/// u64: the card byte budget (`BLOOMERY_CARD_BUDGET`) the fixture's gate
/// placement plans under.
pub const KEY_CARD_BUDGET: &str = "bloomery.fixture.card_budget";
/// String array: present only on a file that holds some of the planned
/// tensors — their names, in file order.
pub const KEY_SUBSET: &str = "bloomery.fixture.subset";

/// The seed `generate` takes when none is given.
pub const DEFAULT_SEED: u64 = 1;
/// Tensor data bytes a shard holds at most, unless one tensor is larger.
pub const DEFAULT_SHARD_BYTES: u64 = 16 << 30;

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Placement(#[from] PlacementError),
    #[error("{path}: {op}: {source}")]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{path}: writing: {source}")]
    Write { path: PathBuf, source: WriteError },
    #[error("the {set}'s metadata: {source}")]
    Alignment {
        set: &'static str,
        source: WriteError,
    },
    #[error("the card budget: {0}")]
    Budget(String),
    #[error("the r8 sidecar {path}: {detail}")]
    Sidecar { path: PathBuf, detail: String },
    #[error("the source is {got:?}, not a {want} file")]
    Architecture {
        got: Option<String>,
        want: &'static str,
    },
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    #[error("metadata {key} is not covered by the fixture map")]
    UncoveredKey { key: String },
    #[error("source layer {layer} is past the source's {n_layer} layers")]
    NoSourceLayer { layer: usize, n_layer: usize },
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("tensor {name}: type {ty} has no fixture generator")]
    UnsupportedType { name: String, ty: GgmlType },
    #[error("tensor {name}: no {what} of type {ty} at K = {k} puts d in {window}")]
    NoScale {
        name: String,
        ty: GgmlType,
        k: u64,
        what: &'static str,
        window: Window,
    },
    #[error("the d window [{lo:e}, {hi:e}] leaves the normal f16 values [2^-14, 65504]")]
    BadWindow { lo: f32, hi: f32 },
    #[error("tensor {name}: a 1-D {ty} tensor with no fill rule")]
    NoFillRule { name: String, ty: GgmlType },
    #[error("a draft source was given, and the {arch} fixture has no draft file")]
    NoDraft { arch: &'static str },
    #[error("--tensors names {name}, which the plan does not hold")]
    UnknownTensor { name: String },
    #[error("{} shards, and no fixture layer spans a shard boundary", shards)]
    NoSpanningLayer { shards: usize },
    #[error("{path} exists; a fixture is never written over")]
    Exists { path: PathBuf },
    #[error("{path}: {need} bytes to write, {free} free")]
    Space { path: PathBuf, need: u64, free: u64 },
    #[error("{error}; and removing {tmp} failed: {cleanup}")]
    Abandoned {
        error: Box<FixtureError>,
        tmp: PathBuf,
        cleanup: io::Error,
    },
    #[error("the fixture's {key} is {got}, the source's is {want}")]
    SourceMismatch {
        key: &'static str,
        got: String,
        want: String,
    },
    #[error("the fixture's {what}: {detail}")]
    Mismatch { what: String, detail: String },
    #[error("tensor {name}: block {block}: {detail}")]
    Block {
        name: String,
        block: usize,
        detail: String,
    },
}

fn io_err(path: &Path, op: &'static str, source: io::Error) -> FixtureError {
    FixtureError::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
}

/// A [`FixtureError::Metadata`] on `key`.
pub(crate) fn meta(key: &str, detail: impl Into<String>) -> FixtureError {
    FixtureError::Metadata {
        key: key.to_string(),
        detail: detail.into(),
    }
}
