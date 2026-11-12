//! The K-EXAONE-2.0 family (`exaone-moe`). The node dumps are dumped from the fixture GGUF, [`MODEL`], by llama.cpp at
//! [`COMMIT`] with the exaone-moe oracle patch ([`tools/ref/exaone-moe-dump.sh`] names it): `src/models/exaone-moe.cpp` is
//! mainline's own graph, read through the patch's arrays — `attention.sliding_window` a u32 one, the two `swiglu_clamp_*`
//! arrays, the arch on both `ggml_swiglu_clamp` lists — so the oracle and the engine see one description of the file. The
//! patch is the lead's, never upstream: a build line without it is a foreign build, not this family's.
//!
//! The fixture GGUF is `bloomery-quantize --hf`'s bootstrap mix of the cut folder (`fixture cut`, its rules
//! `tools/fixture-cuts/exaoneh.json`, `/models/fixtures/exaoneh`): the routed stacks Q4_K, the trunk and the head Q8_0, F32
//! norms, router and correction bias (`crates/gguf/src/arch/exaone_moe.rs` states the mix) — the one file both engines read,
//! the engine opening no FP8 folder (an FP8 attention site has no card column). It is not a `bloomery.fixture.*` file: the
//! quantizer writes llama-quantize's metadata only, so the family's identity is the `# model` line alone, the GGUF's full
//! path — a set of the cut folder, of the vocab-only file [`VOCABULARY`] names, or of a later real-file quantization is stale
//! by that line, not by its name.
//!
//! `tools/ref/exaone-moe-dump.sh` is the writer (the shared dumper, `tools/ref/cb_dump.cpp`): one set a prompt of
//! `tools/ref/data/exaone-moe`, each a dump_ref-format set (`MANIFEST.tsv`, one f32 file a node and occurrence, the router's
//! expert ids with an i32 twin and `int` row) that also holds `topk.tsv` and `greedy.tsv`
//! ([`prompt_tables`](super::prompt_tables)).
//!
//! The nodes the body gates compare, the graph's own `cb()` names (a layer's name is `<node>-<il>`; a name a layer's kind does
//! not have is absent from that layer — the dense blocks 0 and 1 and the next-token block 6 have no router nodes, the routed
//! ones no `ffn_gate`): `attn_norm` and `ffn_norm`, each site's input after the residual and the norm; `Qcur_normed` and
//! `Kcur_normed`, the query and key after their per-head RMS norms (before rope); `Qcur`, `Kcur` and `Vcur`, the roped query
//! and key and the value; `attn_out`, the attention output after its projection; `ffn_inp`, the residual sum; the router's
//! `ffn_moe_logits`, `ffn_moe_probs`, `ffn_moe_probs_biased` (the selection scores with the correction bias),
//! `ffn_moe_topk` (the chosen ids, an i32 node), `ffn_moe_weights`, `ffn_moe_weights_sum`, `ffn_moe_weights_sum_clamped`,
//! `ffn_moe_weights_norm` and `ffn_moe_weights_scaled` (the scale 2.5 applied — the weights the experts take);
//! `ffn_moe_out` and `ffn_shexp` (the routed stack's sum and the shared expert's), `ffn_out` and `l_out`; and the head's
//! `result_norm` and, always, `result_output` (the last position's logits row).

#[cfg(test)]
mod tests;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every exaone-moe manifest names in its `# arch` line: the GGUF `general.architecture`.
pub const ARCH: &str = "exaone-moe";

/// The vocab-only GGUF the converter writes from the hub folder: the file the `exaone`
/// tokenizer gate (`crates/tokenizer/tests/exaone.rs`) and its oracle
/// (`crates/tokenizer/tools/exaone-oracle.sh`) open; no node set is dumped from it (it holds no
/// tensors), so it never was this family's file.
pub const VOCABULARY: &str = "/data/models/K-EXAONE-2.0-vocab/exaone-moe-vocab.gguf";

/// The llama.cpp commit the oracle tree is built at, mainline's `src/models/exaone-moe.cpp`
/// carrying the exaone-moe oracle patch (`tools/ref/exaone-moe-dump.sh`'s `CB_PATCH`).
pub const COMMIT: &str = "7bcbc04fb";

/// The fixture GGUF every exaone-moe set is dumped from and the engine's gates load: the identity
/// each set's `# model` line states. A set of the file named by another path — the cut folder it
/// was quantized from, the vocab-only file, a requantization — is stale by this line, not by its
/// name.
pub const MODEL: &str = "/models/fixtures/exaoneh-gguf/exaoneh-gguf-fixture-00001-of-00001.gguf";

/// The build every exaone-moe set names: llama.cpp [`COMMIT`] and the exaone-moe oracle patch,
/// `<commit>+<tag>-<first 12 hex of the patch's sha256>` (`tools/ref/exaone-moe-dump.sh --build`
/// records the whole sha256 beside the binary and the run refuses a patch that moved).
// PIN(2026-11-12): mainline at the commit plus the oracle patch the fixture GGUF was checked
// against (llama-tokenize, llama-completion); a rebuild of either moves the pin, and the sets
// are dumped again.
pub const BUILD: &str = "7bcbc04fb+exaone-moe-c7a1bf15a85f";

/// Prompt a: an English paragraph, every tap and 32 greedy tokens.
pub const A: &str = "ref_exaone_a";
/// Prompt b: a Korean paragraph, the layer taps and 32 greedy tokens.
pub const B: &str = "ref_exaone_b";
/// Prompt c: a Rust function head, the layer taps and 32 greedy tokens.
pub const C: &str = "ref_exaone_c";
/// Prompt d: a five-id question, the layer taps and 32 greedy tokens.
pub const D: &str = "ref_exaone_d";

/// The prompt sets, in the order the dump writes them.
pub const SETS: &[&str] = &[A, B, C, D];

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// The patched mainline's node dumps on the fixture GGUF, which the exaone-moe gates read.
pub static LCPP: Family = Family {
    name: "lcpp-exaone-moe",
    sets: SETS,
    resolve: None,
    recipe: "just dump-ref-exaone-moe",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-gpu-exaone-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them: the fixture GGUF's today;
/// a real-file family, when one is dumped, goes before it.
pub static FAMILIES: &[&Family] = &[&LCPP];
