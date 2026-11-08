//! The Muse Spark Open family (`muse-spark`). The node dumps are dumped from the UD-IQ2_XXS
//! split set, [`MODEL`], by mainline llama.cpp at [`COMMIT`] (a CPU build with no patch, the
//! merge of PR 30616, [`LCPP_BUILD`]): `src/models/muse-spark.cpp` is mainline's own graph, so
//! the oracle and the public baseline are one engine. `tools/ref/musespark-dump.sh` is the
//! writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a prompt of
//! `tools/ref/data/musespark`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a node
//! and occurrence, the router's expert ids with an i32 twin and `int` row) that also holds
//! `topk.tsv` (the last position's 40 largest logits) and `greedy.tsv` (argmax tokens and
//! their margins), read by [`prompt_tables`](super::prompt_tables). The fixture tier's mirror
//! is in [`fixture`].
//!
//! The nodes the body gate compares, the graph's own `cb()` names: each layer's attention
//! block at `attn_gate_proj` (the gate's projection), `attn_gate_sig`, `attn_gated` and
//! `attn_o_proj` (the output after the gate), `attn_post_norm` (the weightless 1e-8 norm) and
//! `ffn_inp` (the α/β residual), the latent pair `ffn_latent_down` and `ffn_latent_up` around
//! the experts, `ffn_moe_logits` (the router's pre-scaled logits — a body that multiplies by
//! the output multiplier again diverges here by a constant per logit), `ffn_moe_topk` (the
//! selection the bias steers) and `ffn_moe_out`, the FFN residual at `l_out`, and the
//! softcapped logits at `result_output`.

pub mod fixture;

/// The tables beside the node dumps are every mainline-API family's
/// ([`super::prompt_tables`]); the path stays the family's for its gate.
pub use super::prompt_tables as read;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every musespark manifest names in its `# arch` line: the GGUF
/// `general.architecture`.
pub const ARCH: &str = "muse-spark";

/// The llama.cpp commit the oracle tree is built at, PR 30616's merge
/// (`ba3c05f71da85cd0439e84ba42b0478b24d5db16`, the tree `/home/user/llama.cpp-muse`): its
/// first 9 digits are the tokenizer family's pin too ([`super::tokenizer`]).
pub const COMMIT: &str = "ba3c05f71";

/// The first shard of the UD-IQ2_XXS split set every musespark set is dumped from and the tree
/// runs: the identity each set's `# model` line states. A set of the file moved elsewhere is
/// stale by this line, not by its name.
pub const MODEL: &str = "/data/moved/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf";

/// The build every musespark set names: llama.cpp [`COMMIT`] and the first 12 hex digits of
/// the md5 of the `libllama.so` built from it (`/home/user/llama.cpp-muse`, the tree the
/// profile's `LCPP` names; `tools/ref/musespark-dump-build.sh` records the whole md5 beside
/// the binary and the run refuses a library that moved).
// PIN(2026-10-23): the merge of PR 30616 and the libllama md5 its first build on the box
// recorded. A rebuild of the tree moves the build and the sets are dumped again.
pub const LCPP_BUILD: &str = "ba3c05f71+libllama-d77207daa41e";

/// Prompt a: an English paragraph, every per-layer tap and 64 greedy tokens.
pub const A: &str = "ref_musespark_a";
/// Prompt b: a Korean paragraph, each layer's output and the router's expert ids, and 64
/// greedy tokens.
pub const B: &str = "ref_musespark_b";
/// Prompt c: a Rust function, the same as b.
pub const C: &str = "ref_musespark_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_musespark_d";
/// Prompt e: the first 2,560 ids of wikitext-2 (d's ids and 1,024 more): the only context
/// past the 2,048-position sliding window, so the sliding layers read a window of real cells
/// and the full layers the whole context.
pub const E: &str = "ref_musespark_e";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's node dumps, read by the body's end-to-end gate in the real tier
/// (`gate-gpu-musespark-e2e`: sets a, b, c and e).
pub static LCPP: Family = Family {
    name: "lcpp-musespark",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-musespark",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-musespark-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them. [`fixture::LCPP`]
/// joins when the fixture file exists and its sets are dumped
/// (`BLOOMERY_TIER=fixture just dump-ref-musespark`): a family in the table is checked in
/// place by `gate-refset`, and one whose sets are absent is red there.
pub static FAMILIES: &[&Family] = &[&LCPP];
