//! The Kimi K3 family (`kimi-k3`). The node dumps are dumped from the UD-IQ1_S split set, [`MODEL`] (594 GB in fourteen shards), by
//! mainline llama.cpp at [`COMMIT`] (a CPU build with no patch, [`LCPP_BUILD`]): `src/models/kimi-k3.cpp` is mainline's own graph, so
//! the oracle and the public baseline are one engine. `tools/ref/kimi-k3-dump.sh` is the writer (the shared dumper,
//! `tools/ref/cb_dump.cpp`): one set a prompt of `tools/ref/data/kimi-k3`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a
//! node and occurrence, the router's expert ids with an i32 twin and `int` row) that also holds `topk.tsv` (the last position's 40
//! largest logits) and `greedy.tsv` (argmax tokens and their margins), read by [`prompt_tables`](super::prompt_tables). The fixture
//! tier's mirror is in [`fixture`].
//!
//! The nodes the body gates compare, the graph's own `cb()` names (a layer's name is `<node>-<il>`; a name a layer's kind does not
//! have is absent from that layer):
//! `attn_norm` and `ffn_norm`, each site's input after the cross-layer residual's mix and the norm; `res_probs`, the mix's softmax
//! over the bank and the current stream (twice a layer, the attention site then the ffn site, except layer 0, whose attention site
//! has an empty bank, and once more, unnumbered, for the head's final mix); the delta layers' `kda_q_conv`, `kda_k_conv`, `kda_v_conv`, `kda_g1`,
//! `kda_beta`, `kda_scan_out`, `kda_g2`, `kda_normed` and `kda_out`; the latent layers' `mla_gated` (the gated attention output before
//! the output projection) and `mla_out`; `prefix_sum_attn` (the stream after the attention site; on a bank layer, every 12th, it is the attention output itself,
//! which has no `kda_out` or `mla_out` node there); the router's `ffn_moe_logits`,
//! `ffn_moe_probs`, `ffn_moe_probs_biased`, `ffn_moe_topk` (the chosen ids), `ffn_moe_weights` and `ffn_moe_weights_norm`; the routed
//! stack's `ffn_moe_situ` and `ffn_moe_out` (in the experts' latent), `ffn_shexp` (the shared stack), `ffn_out`, `l_out`; and the
//! head's `result_norm` and, always, `result_output` (the last position's logits row).

pub mod fixture;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every kimi-k3 manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "kimi-k3";

/// The llama.cpp commit the oracle tree is built at, the merge of PR 31376 (`78feec062d50ec9e8940244cac918ac4480b60cb`, the tree
/// `/home/user/llama.cpp-kimik3`): the commit Mistral Large 4's oracle and the kimik2 tokenizer pins are taken from, so one source
/// tree answers every vocabulary and graph question of both.
pub const COMMIT: &str = "78feec062";

/// The first shard of the UD-IQ1_S split set every kimi-k3 set is dumped from and the tree runs: the identity each set's `# model`
/// line states. Shard 1 holds the metadata and no tensors; the tensors are in shards 2 to 14, which the tree opens from shard 1. A
/// set of the file named by another path is stale by this line, not by its name.
pub const MODEL: &str = "/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";

/// The build every kimi-k3 set names: llama.cpp [`COMMIT`] and the first 12 hex digits of the md5 of the `libllama.so` built from it
/// (`/home/user/llama.cpp-kimik3`, the tree the profile's `LCPP` names; `tools/ref/kimi-k3-dump.sh --build` records the whole md5
/// beside the binary and the run refuses a library that moved).
// PIN(2026-11-05): the merge of PR 31376 and the libllama md5 the tree's first build on the box recorded (`just build-k3-lcpp`). A
// rebuild of the tree moves the build and the sets are dumped again.
pub const LCPP_BUILD: &str = "78feec062+libllama-ae99cf8db62c";

/// Prompt a: an English paragraph of 35 ids, every tap and 32 greedy tokens.
pub const A: &str = "ref_kimik3_a";
/// Prompt b: a Korean paragraph of 46 ids, the layer taps and 32 greedy tokens.
pub const B: &str = "ref_kimik3_b";
/// Prompt c: a Rust function head of 48 ids, the layer taps and 32 greedy tokens.
pub const C: &str = "ref_kimik3_c";
/// Prompt d: a five-id question, the layer taps and 32 greedy tokens.
pub const D: &str = "ref_kimik3_d";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's node dumps, which the Kimi K3 gates read once they exist.
pub static LCPP: Family = Family {
    name: "lcpp-kimi-k3",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-kimi-k3",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-kimik3-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them: the real file's, then the fixture's.
pub static FAMILIES: &[&Family] = &[&LCPP, &fixture::LCPP];
