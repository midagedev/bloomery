//! The MiniMax-M3.1-Flash family. The node dumps are written without llama.cpp: `tools/ref/mm31/dump_tf.py` runs transformers'
//! own `MiniMaxM3VL` decoder layers in f32 on the CPU over the vendor's safetensors folder, [`MODEL`]'s directory, one layer
//! resident at a time (the MXFP8 and NVFP4 matrices dequantized on the fly, each rule checked bit for bit against the library
//! that owns it), and writes the sets the MiniMax-M3 gates already read: dump_ref-format sets (`MANIFEST.tsv`, one f32 file a node
//! and occurrence, an integer node's i32 twin and `int` row) that also hold `topk.tsv` and `greedy.tsv`
//! ([`prompt_tables`](super::prompt_tables)). The prompts are the MiniMax-M3 ones (`tools/ref/data/minimaxm3`): the two models
//! share one tokenizer file.
//!
//! The sets keep [`minimaxm3`](super::minimaxm3)'s tap names and layouts. They differ in the writer and in one rule: the vendor's
//! indexer takes the maximum over the index heads and one top-16 shared by every attention head, where llama.cpp and the card
//! kernel select per index head. The two choose the same blocks while a query sees at most 16 of them (2,048 positions), so
//! sets a to d agree under either; [`E`] states the vendor's rule and [`E_PERHEAD`] the per-head one over the same ids, each
//! naming its rule in its `# msa_rule` line (`pooled`, `per-head`). The gates run the card kernel's rule, so [`TF`] holds the
//! per-head set and [`POOLED`], which no gate reads, the vendor's: a gate that opened every set of one family would compare the
//! card's selection with the other rule's at the one context where they differ.
//!
//! The fixture tier's mirror, the dump of the cut folder (`fixture cut`), is in [`fixture`].

pub mod fixture;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every minimaxm31 manifest names in its `# arch` line: MiniMax-M3's, since the graph is the same. The
/// registry's `minimax_rows_are_both_modules_families` holds it equal to [`minimaxm3::ARCH`](super::minimaxm3::ARCH).
pub const ARCH: &str = "minimax-m3";

/// The first shard of the checkpoint folder every minimaxm31 set is dumped from: the identity each set's `# model` line states.
/// A set of the folder moved elsewhere, or of a cut of it, is stale by this line, not by its name.
pub const MODEL: &str = "/models/MiniMax-M3.1-Flash/model-00001-of-00048.safetensors";

/// The build every minimaxm31 set names: transformers' and torch's release numbers and the first 12 hex digits of the md5 of
/// `tools/ref/mm31/dump_tf.py`, the writer. `the_writer_pin_is_the_source_in_the_tree` holds it to the tree.
// PIN(2026-11-02): the writer that streams one layer at a time through transformers 5.12.1's modeling code at torch 2.14.1
// (the box's `/home/user/hfref-venv`) with the dequantization checked against compressed-tensors, and resumes only a run of
// the same cut, named by refset's whole fixture line; an edit of the writer moves the pin, and the sets are dumped again.
pub const BUILD: &str = "transformers-5.12.1+torch-2.14.1+dump_tf-afef8a515dfa";

/// Prompt a: an English paragraph, 41 ids, every per-layer tap and 64 greedy tokens.
pub const A: &str = "ref_minimaxm31_a";
/// Prompt b: a Korean paragraph, 55 ids, each layer's output and router ids, and 64 greedy tokens.
pub const B: &str = "ref_minimaxm31_b";
/// Prompt c: a Rust function, 116 ids, each layer's output and router ids, and 64 greedy tokens.
pub const C: &str = "ref_minimaxm31_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_minimaxm31_d";
/// Prompt e: the first 2,560 ids of wikitext-2, the sparse attention's block scores and selection under the vendor's rule (one
/// top-16 over the heads' maximum). The only context longer than the 2,048 positions that 16 blocks of 128 cover.
pub const E: &str = "ref_minimaxm31_e";
/// Prompt e's ids under llama.cpp's rule (one top-16 per index head): the selection the card kernel computes.
pub const E_PERHEAD: &str = "ref_minimaxm31_e_perhead";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E, E_PERHEAD];

/// The sets [`TF`] holds: the prompts under the per-head selection rule, the card kernel's and llama.cpp's.
pub const PERHEAD_SETS: &[&str] = &[A, B, C, D, E_PERHEAD];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The checkpoint folder's node dumps by transformers under the per-head selection rule, which the MiniMax-M3.1 gates read.
pub static TF: Family = Family {
    name: "tf-minimaxm31",
    sets: PERHEAD_SETS,
    resolve: None,
    recipe: "just dump-ref-minimaxm31",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-minimaxm3-e2e"],
};

/// Set e under the vendor's selection rule, the one dump of the same writer whose indexer pools the heads; no gate reads it.
pub static POOLED: Family = Family {
    name: "tf-minimaxm31-pooled",
    sets: &[E],
    resolve: None,
    recipe: "just dump-ref-minimaxm31",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The module's families, in the order `refset-check` lists them. [`fixture::TF`] joins when the cut exists and its sets are
/// dumped (`BLOOMERY_TIER=fixture just dump-ref-minimaxm31 all`): a family in the table is checked in place by `gate-refset`, and
/// one whose sets are absent is red there.
pub static FAMILIES: &[&Family] = &[&TF, &POOLED];
