//! The Solar-Open2-250B families. The node dumps are dumped from the Q4_K_M split set, [`MODEL`], by prometheusAIR's
//! solar_open2 patch on llama.cpp ([`PAI_BUILD`]), the only faithful implementation of the architecture (it applies
//! beta = 2 sigmoid; mainline has no such architecture and no pull request, ik has none). `tools/ref/solar2-dump.sh` is
//! the writer: one set a prompt of `tools/ref/data/solar2`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a node and
//! occurrence) that also holds `topk.tsv` (the last position's 40 largest logits) and `greedy.tsv` (64 argmax tokens and their
//! margins), read by [`read`]. The fixture tier's mirror is in [`fixture`].

pub mod fixture;
pub mod read;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every solar2 manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "solar-open2";

/// The first shard of the Q4_K_M split set every solar2 set is dumped from and the tree runs: the identity each set's `# model`
/// line states. A set of the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf";

/// The build every solar2 set names: llama.cpp `6ea215d17` with prometheusAIR's patch (`solar_open2.patch`, sha256
/// `d2906be2c009...`), the tree the profile's `PAI` names.
// PIN(2026-10-20): the third-party baseline the user picked for a model with no pull request
// (tools/ref/models/solar-open2.sh); `tools/ref/solar2-dump-build.sh` records the library's md5 beside the binary.
pub const PAI_BUILD: &str = "6ea215d17+pai-d2906be2c009";

/// Prompt a: an English paragraph, 41 ids, every per-layer tap and 64 greedy tokens.
pub const A: &str = "ref_solar2_a";
/// Prompt b: a Korean paragraph, 40 ids, each layer's output and 64 greedy tokens.
pub const B: &str = "ref_solar2_b";
/// Prompt c: a Rust function, 114 ids, each layer's output and 64 greedy tokens.
pub const C: &str = "ref_solar2_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_solar2_d";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The PAI tree's node dumps, which the Solar end-to-end gate reads.
pub static PAI: Family = Family {
    name: "pai-solar2",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-solar2",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(PAI_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-solar2-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them. [`fixture::PAI`] joins when the fixture file exists and its
/// sets are dumped (`BLOOMERY_TIER=fixture just dump-ref-solar2`): a family in the table is checked in place by `gate-refset`, and
/// one whose sets are absent is red there.
pub static FAMILIES: &[&Family] = &[&PAI];
