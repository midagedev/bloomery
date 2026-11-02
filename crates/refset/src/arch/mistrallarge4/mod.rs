//! The Mistral Large 4 family (`mistral-large4`). The node dumps are dumped from the UD-IQ1_S
//! split set, [`MODEL`], and again from the Mix-256G split set, [`MODEL_MIX256G`] (a family of
//! its own: a set names its file's full path), by mainline llama.cpp at [`COMMIT`] (a CPU build
//! with no patch, the merge of PR 31376, [`LCPP_BUILD`]): `src/models/mistral-large4.cpp` is
//! mainline's own graph, so the oracle and the public baseline are one engine.
//! `tools/ref/mistrallarge4-dump.sh` is
//! the writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a prompt of
//! `tools/ref/data/mistrallarge4`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a
//! node and occurrence, the router's expert ids and the indexer's positions with an i32 twin and
//! `int` row) that also holds `topk.tsv` (the last position's 40 largest logits) and
//! `greedy.tsv` (argmax tokens and their margins), read by
//! [`prompt_tables`](super::prompt_tables). Beside the two prompts each real file has a deep set,
//! [`D3K`] (`just dump-ref-mistrallarge4 d3k`): a decode step at position [`D3K_POSITION`], past the
//! lightning indexer's top-k, after a quiet prefill of the prose that precedes it, so the
//! selection it dumps is a cut of the positions and not all of them ([`check_deep`]).
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

pub mod dspark;
pub mod fixture;
pub mod vis;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::ik::RefManifest;

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

/// The first shard of the Mix-256G split set (233 GB in six shards, the second real file of the
/// architecture) every `ref_mistrallarge4_mix256g_*` set is dumped from: the identity its
/// `# model` line states, the spelling of `just dump-ref-mistrallarge4-mix256g`.
pub const MODEL_MIX256G: &str =
    "/models/Mistral-Large-4-1T-A52B-Mix-256G/Mistral-Large-4-1T-A52B-Mix-256G-00001-of-00006.gguf";

/// The build every mistrallarge4 set names: llama.cpp [`COMMIT`] and the first 12 hex digits of
/// the md5 of the `libllama.so` built from it (`/home/user/llama.cpp-ml4`, the tree the
/// profile's `LCPP` names; `tools/ref/mistrallarge4-dump.sh --build` records the whole md5 beside
/// the binary and the run refuses a library that moved).
// PIN(2026-10-31): the merge of PR 31376 and the libllama md5 its first build on the box
// recorded. A rebuild of the tree moves the build and the sets are dumped again.
pub const LCPP_BUILD: &str = "78feec062+libllama-22407f1ce644";

/// Prompt a: an English sentence of 18 ids, every tap and 32 greedy tokens.
pub const A: &str = "ref_mistrallarge4_a";
/// Prompt b: a Rust function with a number, 42 ids, the same taps and 32 greedy tokens.
pub const B: &str = "ref_mistrallarge4_b";

/// The deep set: 3,001 ids of the tokenizer oracle's prose, a decode step at position
/// [`D3K_POSITION`] after a quiet prefill of the 3,000 before it, every tap one row, no greedy
/// continuation. The step's selection is 2,048 of its 3,001 positions on every full indexer layer.
pub const D3K: &str = "ref_mistrallarge4_d3k";

/// The deep sets' step position: the prefill's length, one past the ids it follows. It exceeds the
/// indexer's top-k (2,048, the header's `attention.indexer.top_k`), which a gate reading the set
/// states against the file in view; `tools/ref/mistrallarge4-dump.sh`'s `D3K_IDS` is this plus one.
pub const D3K_POSITION: u32 = 3000;

/// The prompt sets, in the order the dumps write them: the two prompts (`just dump-ref-mistrallarge4`),
/// then the deep set (`just dump-ref-mistrallarge4 d3k`).
pub const SETS: &[&str] = &[A, B, D3K];

/// Prompt a of the Mix-256G file: [`A`]'s ids, the Mix file's taps.
pub const A_MIX256G: &str = "ref_mistrallarge4_mix256g_a";
/// Prompt b of the Mix-256G file: [`B`]'s ids, the Mix file's taps.
pub const B_MIX256G: &str = "ref_mistrallarge4_mix256g_b";

/// The deep set of the Mix-256G file: [`D3K`]'s ids, the Mix file's taps.
pub const D3K_MIX256G: &str = "ref_mistrallarge4_mix256g_d3k";

/// The Mix-256G prompt sets, in the order the dumps write them.
pub const SETS_MIX256G: &[&str] = &[A_MIX256G, B_MIX256G, D3K_MIX256G];

/// The gate recipes that read the node dumps of either real file and of the fixture: the body's
/// end-to-end gate, whose real tier opens the family of the file it runs.
const CONSUMERS: &[&str] = &["gate-gpu-mistrallarge4-e2e"];

/// The deep set's input, refused by naming `name` unless it is what the dump writes: a decode
/// step ([`RefManifest::step`]: its position is the number of ids before it, the set holding one
/// id after them) at [`D3K_POSITION`]. A batch set, whose step is position 0, or a step at another
/// depth is refused, so a set the indexer's top-k does not cut (its selection every position)
/// never stands in for the deep one.
pub fn check_deep(man: &RefManifest, name: &str) -> Result<(), RefError> {
    let (pos, _, _) = man
        .step()
        .map_err(|e| RefError::malformed(name, e.to_string()))?;
    if pos != D3K_POSITION {
        return Err(RefError::malformed(
            name,
            format!(
                "decode position {pos}; the deep set is a decode step at position {D3K_POSITION}"
            ),
        ));
    }
    Ok(())
}

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// [`MODEL_MIX256G`], as a family's `runs`.
fn model_mix256g() -> Result<String, RefError> {
    Ok(MODEL_MIX256G.to_string())
}

/// Mainline's node dumps, read by the body's end-to-end gate (`gate-gpu-mistrallarge4-e2e`):
/// the real tier's sets a and b and the deep set when the file in view is the UD-IQ1_S one (its
/// fixture twin [`fixture::LCPP`] answers the fixture tier, which has no deep set).
pub static LCPP: Family = Family {
    name: "lcpp-mistrallarge4",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-mistrallarge4 [d3k]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: CONSUMERS,
};

/// Mainline's node dumps on the Mix-256G file, [`LCPP`]'s mirror: the same tree, build,
/// architecture and prompts, the file and the set names its own. The node-dump family the gates
/// read stays [`LCPP`] (`refset::arch::node_dumps` takes the first); a gate that reads these sets
/// names this family, as the end-to-end gate does when the Mix-256G file is the one in view.
pub static LCPP_MIX256G: Family = Family {
    name: "lcpp-mistrallarge4-mix256g",
    sets: SETS_MIX256G,
    resolve: None,
    recipe: "just dump-ref-mistrallarge4-mix256g [d3k]",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model_mix256g),
    draft_runs: None,
    consumers: CONSUMERS,
};

/// The architecture's families, in the order `refset-check` lists them: the UD-IQ1_S file's, the
/// Mix-256G file's, the fixture's ([`fixture::LCPP`], dumped by `BLOOMERY_TIER=fixture just
/// dump-ref-mistrallarge4`), the DSpark draft's calls ([`dspark::CALLS`]), then the image input's
/// ([`vis`]).
pub static FAMILIES: &[&Family] = &[
    &LCPP,
    &LCPP_MIX256G,
    &fixture::LCPP,
    &dspark::CALLS,
    &vis::PREPROC,
    &vis::TAPS,
];
