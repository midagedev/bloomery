//! The Inkling 2 family (`inkling2`, the Mini and the Small one architecture). The node dumps are dumped from the Mini's
//! UD-Q4_K_XL file, [`MODEL`], by llama.cpp at pull request #32695's head [`COMMIT`] carrying [`BUILD`]'s patch:
//! `src/models/inkling2.cpp` is the PR's own graph under the naming the Mini's file uses (`inkling2` + `num_loops` +
//! `blk.N.loop_*`), and the patch is upstream master's own fix for the fused NVFP4 stacks' per-expert scales
//! (`tools/ref/inkling2-scale.patch`, master's "model : load ffn_gate_up_exps scales" #32657) which the PR branch lacks —
//! without it the loader refuses both the Mini's and the fixture's files at the tensor count, their
//! `ffn_gate_up_exps.scale` rows never created. `tools/ref/inkling2-dump.sh` is the writer (the shared dumper,
//! `tools/ref/cb_dump.cpp`): one set a prompt of `tools/ref/data/inkling2`, each a dump_ref-format set (`MANIFEST.tsv`, one
//! f32 file a node and occurrence, the router's expert ids with an i32 twin and `int` row) that also holds `topk.tsv` (the
//! last position's 40 largest logits) and `greedy.tsv` (argmax tokens and their margins), read by
//! [`prompt_tables`](super::prompt_tables). The fixture tier's mirror is in [`fixture`].
//!
//! A looped visit's node names carry its **cache slot**, not a visit index: the graph loops over the slots (`n_layer_all`,
//! 48 on the Mini) with `loop_layer` mapping a slot to the stored layer whose weights it shares. The slot layout on this
//! tree: the stem 0-5, the core's passes in order (pass 0 the slots 6-17, pass 1 the 18-29, pass 2 the 30-41) and the
//! coda shifted past them at 42-47 — master's `inkling2-loop.cpp` appends the passes after the physical layers instead,
//! so the two trees' slot numbers differ while the visits are the same; one unique number a visit either way, the files
//! unambiguous with no pass suffix. Two renames the graph itself makes: a block-boundary visit's attention output is
//! re-cb'd `ffn_inp` (so `attn_out` is absent at the boundary slots 0, 6, 12, 18, 24, 30, 36, 42), and on the NVFP4
//! visits the patch's per-expert scale renames the fused expert rows `ffn_moe_gate_up_scaled` and the down projection
//! `ffn_moe_down_scaled` (the Q8_0 coda's keep the plain names).
//!
//! The nodes the body gates compare, the graph's own `cb()` names (a name a visit's kind does not have is absent from it):
//! the bare `norm` (occurrence 0 is the embedding norm, the model's input) and each visit's `norm-<slot>` occurrences (a
//! local visit's attention input after the block-residual mix and the message controller, then its ffn norm; a global
//! visit's three input norms then the ffn norm); `Qcur`, `Kcur` and `Vcur` after the short convolutions and their epilogues
//! (the L2 clips, v's gainless norm), `Rcur` the clipped relative projection; `kq_b` (the relative-bias rows the layer
//! gathers by key distance, the per-query sink subtracted — what `build_attn` adds to kq); `kqv_out` (the attention output
//! before its gate) and `attn_out` (after the gate, `wo` and the post-conv); `ffn_inp` (the block-residual add's output);
//! the router's `ffn_moe_topk` (the chosen ids, an i32 node) and `ffn_moe_weights_scaled` (their normalized scores times
//! the route scale 8.0); `ffn_out` after its post-conv and `l_out`; the loop controllers' `loop_q_inp` and `loop_ffn_inp`
//! (the message merges, on the core visits of the later passes only); and the head's `result_norm` and, always,
//! `result_output` (the last position's logits row, the unpadded vocabulary). Prompt a's set adds the attention's own
//! stages (`kq`, `kq_plus_kq_b`, `kq_soft_max`, `kqv`), the router's whole ladder and the expert stack's
//! (`ffn_moe_gate_up` … `ffn_moe_out`).

pub mod fixture;

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every inkling2 manifest names in its `# arch` line: the GGUF `general.architecture` (the Small's
/// loopless files declare it too).
pub const ARCH: &str = "inkling2";

/// The llama.cpp commit the oracle tree is built at: the head of pull request #32695, "convert: register the looped
/// Inkling 2 architecture separately", the naming the Mini's file uses; the prompts' ids come from the same tree
/// (`crates/tokenizer/tools/inkling2-oracle.sh`).
pub const COMMIT: &str = "6196d9802";

/// The Mini's UD-Q4_K_XL file every inkling2 set is dumped from and the tree runs: the identity each set's `# model` line
/// states. A set of the file named by another path — the Small's shards, a requantization, the fixture — is stale by this
/// line, not by its name.
pub const MODEL: &str = "/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf";

/// The build every inkling2 set names: llama.cpp [`COMMIT`] and the inkling2 scale patch,
/// `<commit>+<tag>-<first 12 hex of the patch's sha256>` (`tools/ref/inkling2-dump.sh --build` records the whole sha256
/// beside the binary and the run refuses a patch that moved). The patch is upstream master's own commit, applied unedited;
/// the day the PR branch merges it the patch goes, the build line becomes the plain commit's, and the sets are dumped
/// again.
// PIN(2026-11-22): PR #32695's head plus master's a1f422260 (#32657) unedited, the loader fix without which the tree
// refuses the family's files; the sets' first dump (docs/cards/inkling2-dump.card).
pub const BUILD: &str = "6196d9802+inkling2-scale-ab1511fbc024";

/// Prompt a: an English paragraph of 35 ids, every tap and 32 greedy tokens.
pub const A: &str = "ref_inkling2_a";
/// Prompt b: a Korean paragraph of 35 ids, the layer taps and 32 greedy tokens.
pub const B: &str = "ref_inkling2_b";
/// Prompt c: a Rust function head of 39 ids, the layer taps and 32 greedy tokens.
pub const C: &str = "ref_inkling2_c";
/// Prompt d: a five-id question, the layer taps and 32 greedy tokens.
pub const D: &str = "ref_inkling2_d";

/// Prompt e: the harbor narrative of 1,038 ids, the layer taps and 32 greedy tokens. Its 14 last positions attend exactly
/// 1,024 keys with the oldest dropped (the only prompt past the window the Mini and the fixture share), its global layers
/// see the whole prefix, and its prefill splits across ubatches (512 + 512 + 14), so it alone reaches the windowed layers'
/// truncation, the short convolutions' state carry across ubatch boundaries and the core's later passes reading windows
/// their own pass filled. a through d (at most 39 ids) never leave the first ubatch or reach the window.
pub const E: &str = "ref_inkling2_e";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D, E];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The patched PR tree's node dumps, which the Inkling 2 body gates read once they exist.
pub static LCPP: Family = Family {
    name: "lcpp-inkling2",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-inkling2",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them: the real file's, then the fixture's.
pub static FAMILIES: &[&Family] = &[&LCPP, &fixture::LCPP];
