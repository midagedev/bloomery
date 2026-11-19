//! The Hy4-flash image-input families: llama.cpp pull request #32517's mtmd `hy_v4_flash` tower (the SigLIP ViT, the RMS-norm
//! convolution adapter and the newline rows) on [`MMPROJ`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-hy4vis`)
//! in the node dumps' set format, at the PR tree [`LCPP_BUILD`]. One family a kind of set, as the other seats' are
//! ([`crate::arch::mistrallarge4::vis`]; the kinds and what they hold are [`crate::clefvis::Kind`]'s), and only the two the tower
//! reads: the preprocess (set A) and the tower's output end (set B: the last block's `layer_out` and `embd`). The oracle tree is a
//! CPU build, so a set is the CPU run and the taps family holds no `.cpu` twin; the text model is read for its vocabulary only, so
//! the prompt kinds wait for the seat's placed load. What this seat's sets state that another seat's do not is read from their own
//! lines ([`crate::clefvis`]): the `# projector hy_v4_flash` layout, `# positions normal` and the pad token; an image's rows are its
//! tokens and the newline row after each of its rows of tokens ([`crate::clefvis::ProjectorKind::rows`]).

use crate::RefError;
use crate::clefvis::{Profile, ProjectorKind, vis_families};

/// The projector every Hy4-flash image-input set is dumped with: AngelSlim's bf16 mmproj under the box's `/models/mmproj`.
pub const MMPROJ: &str = "/models/mmproj/AngelSlim--Hy4-flash-GGUF/mmproj-Hy4-flash-bf16.gguf";

/// The sha256 of [`MMPROJ`] (1,044,397,440 bytes).
// PIN(2026-11-19): the LFS oid of AngelSlim/Hy4-flash-GGUF at rev ecd803c8 (the hub's tree listing), whose size is the file's on
// the box; `clefvis.sh` holds the file to it at every dump.
pub const MMPROJ_SHA256: &str = "8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda";

/// The PR commit the sets name in their `# build` line: the tree of the node dumps ([`super::COMMIT`],
/// `/home/user/llama.cpp-hy4`), a CPU build. `clefvis.sh` writes the tree's commit alone, not the libllama digest the node dumps'
/// build line carries.
pub const LCPP_BUILD: &str = super::COMMIT;

/// The image pad token `<｜hy_image▁pad｜>` in Hy4-flash's vocabulary.
pub const IMAGE_PAD_ID: u32 = 120_120;

/// The seat: the projector, the text width 3,072 and the id of the pad token.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ,
    mmproj_sha256: MMPROJ_SHA256,
    n_embd: 3072,
    image_pad_id: IMAGE_PAD_ID,
    projector: ProjectorKind::HyV4Flash,
    respell: None,
};

/// [`MMPROJ`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ.to_string())
}

vis_families! {
    recipe: "just dump-ref-hy4vis",
    prefix: "ref_hy4_vis",
    name: "hy4-vis",
    profile: PROFILE,
    arch: super::ARCH,
    build: LCPP_BUILD,
    mmproj: mmproj,
    // The two tower kinds name the projector as their file; the prompt kinds' text model waits for the seat's placed load.
    text: mmproj,
    kinds: [preproc, taps_cpu_only],
}
