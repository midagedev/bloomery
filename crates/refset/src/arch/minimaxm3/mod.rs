//! The MiniMax-M3 family. The node dumps are dumped from the UD-IQ4_XS split set, [`MODEL`], by mainline llama.cpp at `cd0a5b6a1`
//! (a CPU build with no patch, [`LCPP_BUILD`]): `src/models/minimax-m3.cpp` is mainline's own graph, so the oracle and the public
//! baseline are one engine. `tools/ref/minimaxm3-dump.sh` is the writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a
//! prompt of `tools/ref/data/minimaxm3`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a node and occurrence, an integer
//! node's i32 twin and `int` row) that also holds `topk.tsv` (the last position's 40 largest logits) and `greedy.tsv` (argmax tokens
//! and their margins), read by [`prompt_tables`](super::prompt_tables). No fixture file exists yet, so the family has no fixture tier.

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every minimaxm3 manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "minimax-m3";

/// The first shard of the UD-IQ4_XS split set every minimaxm3 set is dumped from and the tree runs: the identity each set's
/// `# model` line states. A set of the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str = "/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf";

/// The build every minimaxm3 set names: llama.cpp `cd0a5b6a1` and the first 12 hex digits of the md5 of the `libllama.so` built from
/// it (`/home/user/llama.cpp-mainline-bolt`, the tree the profile's `LCPP` names; `tools/ref/minimaxm3-dump-build.sh` records the
/// whole md5 beside the binary and the run refuses a library that moved).
// PIN(2026-10-21): the mainline tree the nemotron_h_moe family also pins (`nemotron_h_moe::BUILD` spells the same library by its
// full md5); a rebuild of that tree moves this build, and the sets are dumped again.
pub const LCPP_BUILD: &str = "cd0a5b6a1+libllama-d846809bb9cd";

/// Prompt a: an English paragraph, 41 ids, every per-layer tap and 64 greedy tokens.
pub const A: &str = "ref_minimaxm3_a";
/// Prompt b: a Korean paragraph, 55 ids, each layer's output and router ids, and 64 greedy tokens.
pub const B: &str = "ref_minimaxm3_b";
/// Prompt c: a Rust function, 116 ids, each layer's output and router ids, and 64 greedy tokens.
pub const C: &str = "ref_minimaxm3_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_minimaxm3_d";
/// Prompt e: the first 2,560 ids of wikitext-2, the sparse attention's block scores and selection. The only context longer than the
/// 2,048 positions that 16 blocks of 128 cover, so the only one in which the top-16 drops a block.
pub const E: &str = "ref_minimaxm3_e";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's node dumps, which the MiniMax-M3 gates read once they exist.
pub static LCPP: Family = Family {
    name: "lcpp-minimaxm3",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-minimaxm3",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-minimaxm3-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LCPP];
