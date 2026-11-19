//! The Hy4-flash family (`hy_v4_flash`). The node dumps are dumped from Tencent's Q4_K_M split set, [`MODEL`] (168 GB in two shards), by
//! llama.cpp at pull request #32517's head, [`COMMIT`] (a CPU build with no patch, [`LCPP_BUILD`]): `src/models/hy-v4-flash.cpp` is
//! the PR's own graph, so the oracle and the format's reference are one engine. `tools/ref/hy4-dump.sh` is the writer (the shared
//! dumper, `tools/ref/cb_dump.cpp`): one set a prompt of `tools/ref/data/hy4`, each a dump_ref-format set (`MANIFEST.tsv`, one f32
//! file a node and occurrence, the router's expert ids with an i32 twin and `int` row) that also holds `topk.tsv` (the last
//! position's 40 largest logits) and `greedy.tsv` (argmax tokens and their margins), read by
//! [`prompt_tables`](super::prompt_tables). The fixture tier's mirror is in [`fixture`]; the real file's sets are dumped when a gate
//! reads them (no gate opens the 168 GB file yet).
//!
//! The nodes the body gates compare, the graph's own `cb()` names (a layer's name is `<node>-<il>`; a name a layer's kind does not
//! have is absent from that layer): `hc_init` (the embedding repeated into the four streams, unnumbered); a site layer's
//! `engram_rows` (the gathered rows), `engram_delta` and `engram_out` (the streams after the site); every layer's `attn_norm` (the
//! attention pre-mix's output after the norm), `attn_out` (the output projection after its sandwich norm), `hc_attn_out`,
//! `ffn_norm`, `ffn_out` (the feed-forward block's output after its sandwich norm) and `l_out` (the streams after the layer); the
//! hyper-connection's `hc_mixes`, `hc_pre` and `hc_post_gate`; the latent attention's `q_resid`, `kv_cmpr`, `attn_kqv` and
//! `attn_gated`; the sparse layers' indexer (`indexer_q`, `indexer_k`, `indexer_gate`, `indexer_pool_k`, `indexer_weights`,
//! `indexer_score`, `indexer_sel`, the cells the picked pools leave visible); the router's `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_probs_biased`, `ffn_moe_topk` (the
//! chosen ids), `ffn_moe_weights_norm` and `ffn_moe_weights_scaled`; the routed stack's `ffn_moe_out` (in the experts' latent) and
//! the shared stack's `ffn_shexp`; and the head's unnumbered `hc_head_mixes`, `hc_head_pre`, `hc_head`, `result_norm` and, always,
//! `result_output` (the last position's logits row).

pub mod fixture;
pub mod vis;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every hy4 manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "hy_v4_flash";

/// The llama.cpp commit the oracle tree is built at: the head of pull request #32517, "Model : add hy_v4_flash support"
/// (`0a054d0b02454c998e068381d15f3f6615bdd987`, the tree `/home/user/llama.cpp-hy4`), the commit the tokenizer pins
/// (`crates/tokenizer/tools/hy4-oracle.sh`) and the prompts' ids are taken from, so one tree answers the vocabulary and the graph.
pub const COMMIT: &str = "0a054d0b0";

/// The first shard of the Q4_K_M split set every hy4 set is dumped from and the tree runs: the identity each set's `# model` line
/// states. Shard 1 holds the metadata and the trunk; the tree opens shard 2 from it. A set of the file named by another path is
/// stale by this line, not by its name.
pub const MODEL: &str = "/models/Hy4-flash-GGUF/Hy4-flash-Q4_K_M-00001-of-00002.gguf";

/// The build every hy4 set names: llama.cpp [`COMMIT`] and the first 12 hex digits of the md5 of the `libllama.so` built from it
/// (`/home/user/llama.cpp-hy4`, the tree the profile's `LCPP` names; `tools/ref/hy4-dump.sh --build` records the whole md5 beside the
/// binary and the run refuses a library that moved).
// PIN(2026-11-18): the head of PR 32517 and the md5 of the libllama its first box build produced, `5d0c8c15ad913b07500366e964dfedd6`.
// A rebuild of the tree moves the build and the sets are dumped again.
pub const LCPP_BUILD: &str = "0a054d0b0+libllama-5d0c8c15ad91";

/// Prompt a: an English paragraph of 32 ids, every tap and 32 greedy tokens.
pub const A: &str = "ref_hy4_a";
/// Prompt b: a Korean paragraph of 31 ids, the layer taps and 32 greedy tokens.
pub const B: &str = "ref_hy4_b";
/// Prompt c: a Rust function head of 30 ids, the layer taps and 32 greedy tokens.
pub const C: &str = "ref_hy4_c";
/// Prompt d: a five-id question, the layer taps and 32 greedy tokens.
pub const D: &str = "ref_hy4_d";

/// Prompt e: the ids of a, b and c in that order, every tap and 32 greedy tokens. Its 93 ids are past the fixture's top_k and window
/// (both 64) and end on a partial pooling group (93 = 23 x 4 + 1), so it alone reaches three clauses: the indexer's selection (23
/// pools against the 15 plus the open group it picks, so `indexer_score`'s pick and `indexer_sel` drop pools), the window layers'
/// truncation (the 29 oldest positions of the last row fall outside the 64), and the pooled tail (the one-token open group the
/// selection keeps whole). a, b, c and d (at most 32 ids, 8 pools) leave all three whole.
pub const E: &str = "ref_hy4_e";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The PR tree's node dumps, which the Hy4-flash body gates read once they exist.
pub static LCPP: Family = Family {
    name: "lcpp-hy4",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-hy4",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them: the real file's, the fixture's, then the image input's
/// ([`vis`]).
pub static FAMILIES: &[&Family] = &[&LCPP, &fixture::LCPP, &vis::PREPROC, &vis::TAPS];
