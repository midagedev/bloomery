//! The MiMo-V2.6-Flash image-input families: llama.cpp mainline's mtmd `mimovl` tower and its `mimo2` text model on
//! [`super::MODEL`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-mimovis`) in the node dumps' set format, at the
//! mainline tree [`LCPP_BUILD`], with the bf16 projector [`MMPROJ_BF16`]. One family a kind of set, as the other seats' are
//! ([`crate::arch::qwen35moe::vis`]; the kinds and what they hold are [`crate::clefvis::Kind`]'s): the preprocess (set A), the
//! tower's nodes (set B), the hidden states of three chat prompts (set C) and the ids llama-server's chat path gives their
//! requests (set E). The oracle tree is a CPU build, so a set is the CPU run and no family holds a `.cpu` twin; the seat has
//! no prose, bf16-rows or decode arm. What this seat's sets state that a Qwen seat's do not is read from their own lines
//! ([`crate::clefvis`]): the `# projector mimovl` layout, `# positions normal` and the `<|image_pad|>` token of MiMo's
//! vocabulary; its tower nodes are the `mimovl` graph's ([`crate::clefvis::mimo_tap_names`]).
//!
//! The families are written out here, not through `vis_families!`: the macro names the Qwen seats' cases `c1`..`c3` and
//! their twins, and this seat's three prompts are `m1`..`m3`.

use super::{ARCH, MODEL};
use crate::RefError;
use crate::clefvis::{Kind, Profile, ProjectorKind, TOWER_ARCH};
use crate::family::{Build, Family, Identity};

/// The projector every MiMo image-input set is dumped with: ggml-org's bf16 file of the model's repository (the repository the
/// text file of [`MODEL`] comes from), under the box's `/models/mmproj`.
pub const MMPROJ_BF16: &str =
    "/models/mmproj/ggml-org--MiMo-V2.6-Flash-MOPD-GGUF/mmproj-MiMo-V2.6-Flash-MOPD-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`] (2,748,509,792 bytes).
// PIN(2026-10-18): ggml-org/MiMo-V2.6-Flash-MOPD-GGUF at rev e6d4bced739310a721eefad4c247ec7e0247c89d, equal to `sha256sum`
// of the file `tools/ref/clefvis/mimovis-dl.sh` downloaded and to the LFS oid of that revision's tree listing it checks against.
pub const MMPROJ_BF16_SHA256: &str =
    "9b7d98d8afe2a6ae25635ecc490fb37883f912bf1d5648d1334fdc703e4a565e";

/// The mainline commit the sets name in their `# build` line: `/home/user/llama.cpp-mainline-bolt`, a CPU build, the tree
/// that carries both the `mimovl` projector and the `mimo2` architecture.
// PIN(2026-10-18): llama.cpp cd0a5b6a1, the tree of the nemotron_h_moe profile; `tools/ref/clefvis/build-clefvis.sh` links
// the dump binary against it and records the commit.
pub const LCPP_BUILD: &str = "cd0a5b6a1";

/// The image token `<|image_pad|>` (`image_token_id` of the model's config) in MiMo's vocabulary.
pub const IMAGE_PAD_ID: u32 = 151_655;

/// The seat: the projector, the text width 4,096 and the id of `<|image_pad|>`.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: 4096,
    image_pad_id: IMAGE_PAD_ID,
    projector: ProjectorKind::Mimovl,
    respell: None,
};

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// [`MODEL`], as a prompt family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Set A: the test images after mtmd's preprocess, the tower's graph input.
pub static PREPROC: Family = Family {
    name: "mimo2-vis-preproc",
    sets: &["ref_mimo2_vis_preproc"],
    resolve: None,
    recipe: "just dump-ref-mimovis ref_mimo2_vis_preproc",
    identity: Identity::Clefvis(Kind::Preproc, &PROFILE),
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// Set B: the tower's nodes of blocks 0, 1, 5, 9 and 27, the patch embedding, the final norm, the merger's output and the final
/// embeddings of each test image.
pub static TAPS: Family = Family {
    name: "mimo2-vis-taps",
    sets: &["ref_mimo2_vis_taps"],
    resolve: None,
    recipe: "just dump-ref-mimovis ref_mimo2_vis_taps",
    identity: Identity::Clefvis(Kind::Taps, &PROFILE),
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// Set C: `result_norm` of every position of a chat prompt with mainline's tower rows, and the positions fed.
pub static HIDDEN: Family = Family {
    name: "mimo2-vis-hidden",
    sets: &[
        "ref_mimo2_vis_hidden_m1",
        "ref_mimo2_vis_hidden_m2",
        "ref_mimo2_vis_hidden_m3",
    ],
    resolve: None,
    recipe: "just dump-ref-mimovis ref_mimo2_vis_hidden_m<N>",
    identity: Identity::Clefvis(Kind::Hidden, &PROFILE),
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// Set E: the ids llama-server's chat path gives each request, with its rendered prompt.
pub static CHATIDS: Family = Family {
    name: "mimo2-vis-chatids",
    sets: &[
        "ref_mimo2_vis_chatids_m1",
        "ref_mimo2_vis_chatids_m2",
        "ref_mimo2_vis_chatids_m3",
    ],
    resolve: None,
    recipe: "just dump-ref-mimovis --ids",
    identity: Identity::Clefvis(Kind::ChatIds, &PROFILE),
    arch: Some(ARCH),
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};
