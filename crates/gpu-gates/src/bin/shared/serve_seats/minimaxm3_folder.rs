//! A MiniMax-M3 model at a path as the body loads it: a GGUF split, or a checkpoint folder read as
//! stored (`model_hf::folder::open`, no prepare step), whose card tensors the body reads through the
//! folder's part source (`bloomery_gpu_minimaxm3::body::Folder`). This module is the one owner of that open: the serve seat opens a
//! folder once on its main thread ([`open`]) and loads it on the engine thread ([`model`]), and a gate
//! takes the model at a path in one call ([`model_at`]). A build without the `checkpoint` feature
//! refuses a folder by name.

use std::path::Path;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_minimaxm3::body::Model;
use gguf::Split;

pub use super::folder::{Opened, model};

/// The checkpoint folder `dir` read as stored: a MiniMax-M3 folder, whose card tensors the body reads
/// through its part source ([`super::folder::open`]).
pub fn open(dir: &Path) -> Result<Opened, GateError> {
    super::folder::open(dir, refset::arch::minimaxm3::ARCH)
}

/// The model at `path` as the body loads it: its GGUF split, or the checkpoint folder a directory is.
pub fn model_at(path: &Path) -> Result<Model, GateError> {
    if path.is_dir() {
        let f = open(path)?;
        return model(f.header, f.source);
    }
    Ok(Split::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .into())
}
