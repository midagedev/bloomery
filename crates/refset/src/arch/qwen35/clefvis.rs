//! The Clef-Flash image-input families: llama.cpp mainline's mtmd
//! `qwen3vl_merger` tower and its qwen35 text model on [`MODEL_FLASH_Q8`],
//! dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-clefvis`) in
//! the node dumps' set format, at the mainline tree [`LCPP_BUILD`], with the
//! bf16 projector [`MMPROJ_BF16`]. One family a kind of set
//! ([`crate::clefvis::Kind`], the family's `Identity::Clefvis`), so a set of one
//! is refused by name where another is read, by [`Family::check_set`] too; the tower's sets state the projector as their model file
//! (`# arch clip`), the prompts' sets the text model. No gate reads them yet
//! (R1 onward), so `consumers` is empty.

use super::{ARCH, LCPP_BUILD, MODEL_FLASH_Q8};
use crate::RefError;
use crate::clefvis::{Kind, MMPROJ_BF16};
use crate::family::{Build, Family, Identity};

/// The architecture the projector's manifest names in its `# arch` line.
pub const TOWER_ARCH: &str = "clip";

/// Set A: the 8 test images after mtmd's preprocess.
pub const PREPROC_SET: &str = "ref_clefvis_preproc";
/// Set B, and its twin on the CPU.
pub const TAPS_SET: &str = "ref_clefvis_taps";
pub const TAPS_CPU_SET: &str = "ref_clefvis_taps.cpu";

/// Set C of the three prompts (`tools/ref/clefvis/cases.tsv`), then their CPU twins.
pub const HIDDEN_SETS: &[&str] = &[
    "ref_clefvis_hidden_c1",
    "ref_clefvis_hidden_c2",
    "ref_clefvis_hidden_c3",
    "ref_clefvis_hidden_c1.cpu",
    "ref_clefvis_hidden_c2.cpu",
    "ref_clefvis_hidden_c3.cpu",
];

/// Set C′ (the image rows replaced by prose-id text rows) of the three prompts, then their CPU twins.
pub const PROSE_SETS: &[&str] = &[
    "ref_clefvis_prose_c1",
    "ref_clefvis_prose_c2",
    "ref_clefvis_prose_c3",
    "ref_clefvis_prose_c1.cpu",
    "ref_clefvis_prose_c2.cpu",
    "ref_clefvis_prose_c3.cpu",
];

/// Set C″ (mainline's own tower rows rounded to bf16) of the three prompts; the card only.
pub const BF16ROWS_SETS: &[&str] = &[
    "ref_clefvis_bf16rows_c1",
    "ref_clefvis_bf16rows_c2",
    "ref_clefvis_bf16rows_c3",
];

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// [`MODEL_FLASH_Q8`], as a prompt family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL_FLASH_Q8.to_string())
}

/// Set A: mtmd's preprocessed images, the tower's graph input.
pub static PREPROC: Family = Family {
    name: "clefvis-preproc",
    sets: &[PREPROC_SET],
    resolve: None,
    recipe: "just dump-ref-clefvis ref_clefvis_preproc",
    identity: Identity::Clefvis(Kind::Preproc),
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// Set B: the tower's named graph nodes of blocks 0, 1, 13 and 26 and its final embeddings.
pub static TAPS: Family = Family {
    name: "clefvis-taps",
    sets: &[TAPS_SET, TAPS_CPU_SET],
    resolve: None,
    recipe: "just dump-ref-clefvis [--cpu-twin] ref_clefvis_taps",
    identity: Identity::Clefvis(Kind::Taps),
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// Set C: `result_norm` of every position of a Clef prompt with mainline's tower rows, and the positions fed.
pub static HIDDEN: Family = Family {
    name: "clefvis-hidden",
    sets: HIDDEN_SETS,
    resolve: None,
    recipe: "just dump-ref-clefvis [--cpu-twin] ref_clefvis_hidden_c<N>",
    identity: Identity::Clefvis(Kind::Hidden),
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// Set C′: the same prompts with each image's rows replaced by prose-id text rows.
pub static PROSE: Family = Family {
    name: "clefvis-prose",
    sets: PROSE_SETS,
    resolve: None,
    recipe: "just dump-ref-clefvis [--cpu-twin] ref_clefvis_prose_c<N>",
    identity: Identity::Clefvis(Kind::Prose),
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// Set C″: the same prompts with mainline's tower rows rounded to bf16.
pub static BF16ROWS: Family = Family {
    name: "clefvis-bf16rows",
    sets: BF16ROWS_SETS,
    resolve: None,
    recipe: "just dump-ref-clefvis ref_clefvis_bf16rows_c<N>",
    identity: Identity::Clefvis(Kind::Bf16Rows),
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

#[cfg(test)]
mod tests;
