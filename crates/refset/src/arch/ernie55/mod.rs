//! ERNIE 5.5 Flash Preview's reference sets, written without llama.cpp and without transformers: `tools/ref/ernie55_ref.py`, a numpy
//! f32 forward of the checkpoint folder (one layer resident at a time, the FP8 and MXFP4 weights dequantized per row chunk), run by
//! `tools/ref/ernie55-dump.sh` over the two prompts of `tools/ref/data/ernie55`: `short` (34 ids) and `long` (2,328 ids, the one prompt
//! past the DSA indexer's top-k threshold). A set is a dump_ref-format set: a `MANIFEST.tsv` with one f32 file a tensor row, an integer
//! row's i32 twin and `int` row, and `greedy.tsv`, the 8 greedy steps. The rows are the writer's (its docstring lists them): each layer's
//! `l_out-<layer>` (the four streams after the layer, the last four prompt positions), each routed layer's `ffn_moe_topk-<layer>` and
//! `ffn_moe_weights-<layer>`, each DSA layer's `indexer_top_k-<layer>` (`long` alone), `result_norm`, `result_output_raw` and
//! `result_output`.
//!
//! The fixture tier's family is [`fixture`]'s: the cut `fixture cut` writes (`tools/fixture-cuts/ernie55.json`: the real file's layers 0,
//! 1 and 7). The real file's sets (`ref_ernie55_{short,long}`, the same writer and prompts) have no family here yet: a family in the table
//! is checked in place, and one whose sets are absent is red there. Every set's identity is its folder's first shard (`# model`), the
//! writer (`# build`: `ernie55_ref-` and the first 12 hex digits of its md5, [`BUILD`]) and `# arch` ([`ARCH`]); a fixture set's
//! generation is its `# fixture` line too ([`crate::family::Identity::FixtureManifest`]).

pub mod fixture;

#[cfg(test)]
mod tests;

/// The architecture every ernie55 manifest names in its `# arch` line: the profile and the fixture table's name of the model
/// (`tools/ref/models/ernie5_5.sh`, `tools/fixture-dirs.tsv`), `gguf::arch::ernie55::ARCH`.
pub const ARCH: &str = "ernie5_5";

/// The build every ernie55 set names: the writer `tools/ref/ernie55_ref.py` by the first 12 hex digits of its md5.
/// `tests::the_writer_pin_is_the_source_in_the_tree` holds it to the tree.
pub const BUILD: &str = "ernie55_ref-051ae51bcc88";

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&crate::family::Family] = &[&fixture::NP];
