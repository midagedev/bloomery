//! A Qwen3.8 checkpoint folder as a server or a gate opens it: a model path that is a directory is
//! one, read as stored (`model_hf::folder::open`, no prepare step). A build without the
//! `checkpoint` feature refuses it by name.

use std::path::Path;

use bloomery_gpu::arch::qwen3moe::Folder38;
use bloomery_gpu_gates::GateError;
use gguf::Split;
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

/// The checkpoint folder `path` opened for a load: a Qwen3.8 checkpoint; a folder of any other
/// architecture is refused by name.
#[cfg(feature = "checkpoint")]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    let ple_name = model::arch::qwen35moe::names::per_layer_token_embd();
    let mut f = super::folder::checked(path, "qwen4exp", std::slice::from_ref(&ple_name))?;
    let ple = f.rows.remove(&ple_name);
    let source = f.source;
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        draft: f.draft.map_err(|e| e.to_string()),
        parts: Box::new(move |header| Folder38::with(header, source, ple)),
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

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn open(path: &Path) -> Result<Opened, GateError> {
    Err(super::folder::without_checkpoint(path))
}
