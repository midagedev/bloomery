//! The Mistral Large 4 family (`mistral-large4`). The node dumps are dumped from the UD-IQ1_S
//! split set, [`MODEL`], by mainline llama.cpp at [`COMMIT`] (a CPU build with no patch, the
//! merge of PR 31376, [`LCPP_BUILD`]): `src/models/mistral-large4.cpp` is mainline's own graph,
//! so the oracle and the public baseline are one engine. `tools/ref/mistrallarge4-dump.sh` is
//! the writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a prompt of
//! `tools/ref/data/mistrallarge4`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a
//! node and occurrence, the router's expert ids and the indexer's positions with an i32 twin and
//! `int` row) that also holds `topk.tsv` (the last position's 40 largest logits) and
//! `greedy.tsv` (argmax tokens and their margins), read by
//! [`prompt_tables`](super::prompt_tables).
//!
//! The nodes the body gate compares, the graph's own `cb()` names: each layer's `attn_norm`;
//! on the layers with a full indexer `indexer_q_ssmax` (the queries after the SSMax scale),
//! `indexer_k` (the key at each stage: projection, norm, rope, Hadamard rotation) and
//! `indexer_weights`; `top_k`, the positions every layer attends to (a shared-indexer layer
//! reuses the previous full layer's, so the one tensor is dumped once under the last name it
//! was given: layer `il` reads the first `top_k-<j>` with `j >= il`); `Qcur_ssmax`, the absorbed query after its own SSMax scale; on the layers that
//! own a cache `kv_cmpr` (the latent before and after its norm) and `Kcur` (a NoPE layer's rope
//! part is zero-filled); `ffn_inp` (the attention residual), the router's `ffn_moe_logits` and
//! `ffn_moe_probs`, `ffn_moe_topk` (the experts chosen), `ffn_moe_weights` (their probabilities)
//! and `ffn_moe_weights_norm` (the weights the experts are summed with), `ffn_moe_out`,
//! `ffn_shexp` (the shared expert), `ffn_out` and `l_out`; and the head's `result_norm` and the
//! softcapped logits at `result_output`. The fixture tier's mirror is in [`fixture`].

pub mod fixture;

/// The tables beside the node dumps are every mainline-API family's
/// ([`super::prompt_tables`]); the path stays the family's for its gate.
pub use super::prompt_tables as read;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every mistrallarge4 manifest names in its `# arch` line: the GGUF
/// `general.architecture`.
pub const ARCH: &str = "mistral-large4";

/// The llama.cpp commit the oracle tree is built at, PR 31376's merge
/// (`78feec062d50ec9e8940244cac918ac4480b60cb`, the tree `/home/user/llama.cpp-ml4`): its first
/// 9 digits are the tokenizer family's pin too ([`super::tokenizer`]).
pub const COMMIT: &str = "78feec062";

/// The first shard of the UD-IQ1_S split set every mistrallarge4 set is dumped from and the
/// tree runs: the identity each set's `# model` line states. The files live on the root drive and
/// `/models/Mistral-Large-4-1T-A52B-UD-IQ1_S` is a symlink to their directory; this spelling is
/// the identity. A set of the file named by another path is stale by this line, not by its name.
pub const MODEL: &str =
    "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf";

/// The build every mistrallarge4 set names: llama.cpp [`COMMIT`] and the first 12 hex digits of
/// the md5 of the `libllama.so` built from it (`/home/user/llama.cpp-ml4`, the tree the
/// profile's `LCPP` names; `tools/ref/mistrallarge4-dump-build.sh` records the whole md5 beside
/// the binary and the run refuses a library that moved).
// PIN(2026-10-31): the merge of PR 31376 and the libllama md5 its first build on the box
// recorded. A rebuild of the tree moves the build and the sets are dumped again.
pub const LCPP_BUILD: &str = "78feec062+libllama-22407f1ce644";

/// Prompt a: an English sentence of 18 ids, every tap and 32 greedy tokens.
pub const A: &str = "ref_mistrallarge4_a";
/// Prompt b: a Rust function with a number, 42 ids, the same taps and 32 greedy tokens.
pub const B: &str = "ref_mistrallarge4_b";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's node dumps. No gate reads them yet: the body's end-to-end gate joins as a
/// consumer when it exists.
pub static LCPP: Family = Family {
    name: "lcpp-mistrallarge4",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-mistrallarge4",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them: the real file's, then the
/// fixture's ([`fixture::LCPP`], dumped by `BLOOMERY_TIER=fixture just dump-ref-mistrallarge4`).
pub static FAMILIES: &[&Family] = &[&LCPP, &fixture::LCPP];
