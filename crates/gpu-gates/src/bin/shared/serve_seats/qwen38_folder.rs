//! A Qwen3.8 checkpoint folder as a server or a gate opens it: a model path that is a directory is
//! one, read as stored (`model_hf::arch::qwen35moe::folder::open`, no prepare step). A build
//! without the `checkpoint` feature refuses it by name.

use std::path::Path;
#[cfg(feature = "checkpoint")]
use std::path::PathBuf;
#[cfg(feature = "checkpoint")]
use std::sync::{Mutex, PoisonError};

use bloomery_gpu::arch::qwen3moe::Folder38;
use bloomery_gpu_gates::GateError;
use gguf::Split;
#[cfg(feature = "checkpoint")]
use model_hf::arch::qwen35moe::folder;
use tokenizer::Tokenizer;

/// A checkpoint folder opened for a load: its header, which plans as a split's first shard does,
/// its tokenizer, its MTP layer's header (`None` for a folder without one, or why the folder's
/// preparation refuses it), and the load of its parts, which takes the header back once it has
/// planned.
pub struct Opened {
    pub header: Split,
    pub tokenizer: Tokenizer,
    pub draft: Result<Option<Split>, String>,
    pub parts: Box<dyn FnOnce(Split) -> Folder38>,
}

/// The folder [`with_header`] opened, which the process's [`open`] of the same path takes: a
/// server reads the header to pick its seat, then the seat loads the folder, which is opened once.
#[cfg(feature = "checkpoint")]
static KEPT: Mutex<Option<(PathBuf, folder::Opened)>> = Mutex::new(None);

#[cfg(feature = "checkpoint")]
fn read(path: &Path) -> Result<folder::Opened, GateError> {
    folder::open(path).map_err(|e| format!("checkpoint folder {}: {e}", path.display()).into())
}

/// `f` of the header of the checkpoint folder `path` ([`open`]'s), the folder kept for the
/// process's [`open`] of it.
#[cfg(feature = "checkpoint")]
#[allow(
    dead_code,
    reason = "bloomery-serve reads the header to pick the seat; bloomery-serve-qwen38 is the seat"
)]
pub fn with_header<R>(path: &Path, f: impl FnOnce(&Split) -> R) -> Result<R, GateError> {
    let mut kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner);
    let opened = match kept.take() {
        Some((p, opened)) if p == path => opened,
        _ => read(path)?,
    };
    let r = f(&opened.header);
    *kept = Some((path.to_path_buf(), opened));
    Ok(r)
}

/// The checkpoint folder `path` opened for a load: a Qwen3.8 checkpoint; any other folder is
/// refused by name.
#[cfg(feature = "checkpoint")]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    let kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner).take();
    let f = match kept {
        Some((p, opened)) if p == path => opened,
        _ => read(path)?,
    };
    let (source, ple) = (f.source, f.ple);
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        draft: f.draft.map_err(|e| e.to_string()),
        parts: Box::new(move |header| Folder38::new(header, source, ple)),
    })
}

/// The checkpoint folder's MTP header, borrowing its target's embedding and head; a folder
/// without an MTP layer, and one whose layer the preparation refuses, are refused by name.
#[cfg(feature = "checkpoint")]
pub fn draft(path: &Path) -> Result<Split, GateError> {
    open(path)?
        .draft
        .map_err(|e| format!("checkpoint folder {}: its MTP layer: {e}", path.display()))?
        .ok_or_else(|| format!("checkpoint folder {} holds no MTP layer", path.display()).into())
}

/// A checkpoint folder's draft in a build without checkpoint support, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn draft(path: &Path) -> Result<Split, GateError> {
    open(path).map(|f| f.header)
}

/// `f` of a checkpoint folder's header in a build without the `checkpoint` feature, which
/// refuses the folder by name ([`open`]).
#[cfg(not(feature = "checkpoint"))]
#[allow(
    dead_code,
    reason = "bloomery-serve reads the header to pick the seat; bloomery-serve-qwen38 is the seat"
)]
pub fn with_header<R>(path: &Path, f: impl FnOnce(&Split) -> R) -> Result<R, GateError> {
    open(path).map(|o| f(&o.header))
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    Err(format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        path.display()
    )
    .into())
}
