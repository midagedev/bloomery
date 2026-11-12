//! The Raon-1-nano-preview family. The node dumps are dumped from the Q8_0 file, [`MODEL`], by KRAFTON's llama.cpp
//! branch `raon-1-nano-preview` at `4d72ff0d4` (a CPU build with no patch, [`LCPP_BUILD`]): `src/models/raon1nano.cpp`
//! is the vendor's own graph, so the oracle and the public baseline are one engine. `tools/ref/raon1nano-dump.sh` is
//! the writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a prompt of `tools/ref/data/raon1` (the two
//! dialects share the vocabulary, so the prompts are mini's), each a dump_ref-format set (`MANIFEST.tsv`, one f32
//! file a node and occurrence) that also holds `topk.tsv` (the last position's 40 largest logits) and `greedy.tsv`
//! (argmax tokens and their margins), read by [`prompt_tables`](super::prompt_tables). The Q8_0 file is the family's
//! own fixture, so the family has no cut: its `tools/fixture-dirs.tsv` row is `self`.

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every raon1nano manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "raon1nano";

/// The Q8_0 file every raon1nano set is dumped from and the engine runs: the identity each set's `# model` line
/// states. A set of the file moved elsewhere is stale by this line, not by its name.
pub const MODEL: &str = "/data/models/Raon-1-nano-gguf/raon-1-nano-Q8_0.gguf";

/// The build every raon1nano set names: llama.cpp `4d72ff0d4` (krafton-ai/llama.cpp, branch
/// `raon-1-nano-preview`) and the first 12 hex digits of the md5 of the `libllama.so` built from it
/// (`/home/user/llama.cpp-4d72ff0d`, the tree the profile's `LCPP` names; `tools/ref/raon1nano-dump-build.sh`
/// records the whole md5 beside the binary and the run refuses a library that moved).
// PIN(2026-11-13): the vendor branch's tip when the family was added; a rebuild of that tree moves this build, and
// the sets are dumped again.
pub const LCPP_BUILD: &str = "4d72ff0d4+libllama-a0cc307f7a80";

/// Prompt a: an English paragraph, every per-layer tap and 32 greedy tokens.
pub const A: &str = "ref_raon1nano_a";
/// Prompt b: a Korean paragraph, each layer's output and 32 greedy tokens.
pub const B: &str = "ref_raon1nano_b";
/// Prompt c: a Rust function, each layer's output and 32 greedy tokens.
pub const C: &str = "ref_raon1nano_c";
/// Prompt d: the first 1,536 ids of wikitext-2, the last position's logits.
pub const D: &str = "ref_raon1nano_d";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The vendor branch's node dumps, which the Raon-1-nano gates read.
pub static LCPP: Family = Family {
    name: "lcpp-raon1nano",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-raon1nano",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LCPP];
