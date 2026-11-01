//! A checkpoint folder as a serve seat or a gate opens it: the one owner of what every seat's folder
//! open shares. The folder is read as stored (`model_file::open_folder`, no prepare step) by the map
//! its config names, its architecture is held to the seat's, and its row tables to the ones the seat
//! reads; a build without the `checkpoint` feature refuses a folder by name. A seat's own open
//! (`minimaxm3_folder`, `qwen38_folder`, the nemotron seat's) adds only what its architecture forces.

use std::path::Path;

use bloomery_gpu::folder::Model;
use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::placement::Plan;
use tokenizer::Tokenizer;

/// What a folder's load reads card tensors through (`model_file::Folder::source`). A build
/// without the `checkpoint` feature opens no folder, so it has none.
#[cfg(feature = "checkpoint")]
pub type Source = bloomery_gpu_gates::model_file::Checkpoint;
#[cfg(not(feature = "checkpoint"))]
pub type Source = ();

/// A checkpoint folder that carries nothing beside its parts, opened for a load: its located header,
/// which plans as a split's first shard does, its tokenizer and its part source.
pub struct Opened {
    pub header: Split,
    pub tokenizer: Tokenizer,
    pub source: Source,
}

/// The checkpoint folder `dir` read as stored, held to a model of architecture `arch` and to the
/// row tables `rows` names: a folder of another architecture, and one with any other row table
/// (which this seat does not read), is refused by name.
#[cfg(feature = "checkpoint")]
pub fn checked(
    dir: &Path,
    arch: &str,
    rows: &[String],
) -> Result<bloomery_gpu_gates::model_file::Folder, GateError> {
    let f = bloomery_gpu_gates::model_file::open_folder(dir)?;
    if f.header.architecture() != Some(arch) {
        return Err(format!(
            "checkpoint folder {}: a {} model, where this seat serves {arch}",
            dir.display(),
            f.header.architecture().unwrap_or("<no architecture>"),
        )
        .into());
    }
    if let Some(name) = f.rows.keys().find(|name| !rows.contains(name)) {
        return Err(format!(
            "checkpoint folder {}: the row table {name}, which this seat does not read",
            dir.display()
        )
        .into());
    }
    Ok(f)
}

/// The checkpoint folder `dir` of a model of architecture `arch` that carries nothing beside its
/// parts ([`checked`] with no row table).
#[cfg(feature = "checkpoint")]
pub fn open(dir: &Path, arch: &str) -> Result<Opened, GateError> {
    let f = checked(dir, arch, &[])?;
    Ok(Opened {
        header: f.header,
        tokenizer: f.tokenizer,
        source: f.source,
    })
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn open(dir: &Path, _arch: &str) -> Result<Opened, GateError> {
    Err(without_checkpoint(dir))
}

/// The model of a folder whose located header is `header` and whose part source is `source`: a load
/// reads its card tensors through the source.
#[cfg(feature = "checkpoint")]
pub fn model(header: Split, source: Source) -> Result<Model, GateError> {
    Ok(bloomery_gpu::folder::Folder::new(header, source).into())
}

/// The least host bytes a load of card `card` of `plan` reads from a folder's part source
/// `source` whose located header is `header` (`bloomery_gpu::parts::least_reads`): what its plan
/// reserves on the host before the NVMe expert tier's split (`workstation::reserve_part_reads`).
#[cfg(feature = "checkpoint")]
pub fn least_reads(
    header: &Split,
    source: &Source,
    plan: &Plan<'_>,
    card: usize,
) -> Result<u64, GateError> {
    Ok(bloomery_gpu::parts::least_reads(
        source, header, plan, card,
    )?)
}

/// A build without the `checkpoint` feature opens no folder to read.
#[cfg(not(feature = "checkpoint"))]
pub fn least_reads(
    _header: &Split,
    _source: &Source,
    _plan: &Plan<'_>,
    _card: usize,
) -> Result<u64, GateError> {
    Err("a checkpoint folder is opened by a build with the checkpoint feature".into())
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
pub fn model(_header: Split, _source: Source) -> Result<Model, GateError> {
    Err("a checkpoint folder is opened by a build with the checkpoint feature".into())
}

/// The refusal of a checkpoint folder `dir` in a build without the `checkpoint` feature.
#[cfg(not(feature = "checkpoint"))]
pub fn without_checkpoint(dir: &Path) -> GateError {
    format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        dir.display()
    )
    .into()
}
