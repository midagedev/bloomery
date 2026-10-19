//! The Nemotron 3.5 Bolt image-input families: llama.cpp mainline's mtmd `nemotron_omni` tower (the RADIO encoder and its
//! pixel-shuffle projector) on [`MMPROJ`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-boltvis`) in the node
//! dumps' set format, at the mainline tree [`LCPP_BUILD`]. One family a kind of set, as the other seats' are
//! ([`crate::arch::qwen35moe::vis`]; the kinds and what they hold are [`crate::clefvis::Kind`]'s), and only the two the tower
//! reads: the preprocess (set A) and the tower's nodes (set B). The oracle tree is a CPU build, so a set is the CPU run and the
//! taps family holds no `.cpu` twin; the text model is read for its vocabulary only, so the prompt kinds are the media round's.
//! What this seat's sets state that a Qwen seat's do not is read from their own lines ([`crate::clefvis`]): the
//! `# projector nemotron_omni` layout, `# positions normal` and the `<image>` token; a block node of set B holds the 10 class
//! and register rows before the patches ([`crate::clefvis::RADIO_PREFIX_ROWS`]).

use crate::RefError;
use crate::clefvis::{Profile, ProjectorKind, vis_families};

/// The projector every Bolt image-input set is dumped with: NVIDIA's bf16 mmproj from the repository the text model comes from,
/// next to it under the box's `/models/nemotron-3.5-bolt`.
pub const MMPROJ: &str =
    "/models/nemotron-3.5-bolt/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";

/// The sha256 of [`MMPROJ`] (1,733,692,736 bytes).
// PIN(2026-10-18): nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF's mmproj, equal to the `SHA256SUMS` the download wrote
// beside the file (/models/nemotron-3.5-bolt) and to `clefvis.sh`'s own digest of it at every dump.
pub const MMPROJ_SHA256: &str = "0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0";

/// The mainline commit the sets name in their `# build` line: `/home/user/llama.cpp-mainline-bolt`, the tree of the logits
/// set ([`super::LOGITS`]), a CPU build.
pub const LCPP_BUILD: &str = "cd0a5b6a1";

/// The image token `<image>` (`img_context_token_id` of the model's config) in Bolt's vocabulary.
pub const IMAGE_TOKEN_ID: u32 = 18;

/// The seat: the projector, the text width 4,096 and the id of `<image>`.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ,
    mmproj_sha256: MMPROJ_SHA256,
    n_embd: 4096,
    image_pad_id: IMAGE_TOKEN_ID,
    projector: ProjectorKind::NemotronOmni,
    respell: None,
};

/// [`MMPROJ`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ.to_string())
}

vis_families! {
    recipe: "just dump-ref-boltvis",
    prefix: "ref_nemotron_h_moe_vis",
    name: "nemotron_h_moe-vis",
    profile: PROFILE,
    arch: super::ARCH,
    build: LCPP_BUILD,
    mmproj: mmproj,
    // The two tower kinds name the projector as their file; the prompt kinds' text model is the media round's.
    text: mmproj,
    kinds: [preproc, taps_cpu_only],
}
