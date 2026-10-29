//! A dense Qwen3.5-family model at a path as the whole-card Body35 loads it: a GGUF split, or a
//! checkpoint folder read as stored (`model_hf::folder::open`, no prepare step), whose card tensors
//! the body reads through the folder's part source (`bloomery_gpu::folder::Model`). The folder's
//! `mtp.*` tensors are left out of the load. This module is the one owner of that open: the qwen3
//! seat opens a folder once on its main thread ([`open`]) and loads it on the engine thread
//! (`folder::model`), and a gate takes the model at a path in one call ([`model_at`]). A build
//! without the `checkpoint` feature refuses a folder by name.

use std::path::Path;

use bloomery_gpu::folder::Model;
use bloomery_gpu_gates::GateError;
use gguf::Split;

use super::folder;

/// The checkpoint folder `dir` read as stored: a dense qwen35 folder, whose card tensors the body
/// reads through its part source ([`folder::open`]); a folder of another architecture is refused
/// by name.
pub fn open(dir: &Path) -> Result<folder::Opened, GateError> {
    folder::open(dir, refset::arch::qwen35::ARCH)
}

/// The model at `path` as the qwen35moe body's whole-card open takes it: its GGUF split, or the
/// checkpoint folder a directory is.
#[allow(
    dead_code,
    reason = "the dense gates' open of a path; the server opens a folder once on its main thread (open)"
)]
pub fn model_at(path: &Path) -> Result<Model, GateError> {
    if path.is_dir() {
        let f = open(path)?;
        return folder::model(f.header, f.source);
    }
    Ok(Split::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?
        .into())
}
