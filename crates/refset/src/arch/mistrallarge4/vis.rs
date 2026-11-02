//! The Mistral Large 4 image-input families: llama.cpp mainline's mtmd `pixtral` tower (the Pixtral ViT, its patch merger and
//! GELU projector, the `[IMG_BREAK]` rows) on [`MMPROJ`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-ml4vis`) in
//! the node dumps' set format, at the mainline tree [`LCPP_BUILD`]. One family a kind of set, as the other seats' are
//! ([`crate::arch::nemotron_h_moe::vis`]; the kinds and what they hold are [`crate::clefvis::Kind`]'s), and only the two the tower
//! reads: the preprocess (set A) and the tower's output end (set B: the last block's `layer_out` and `embd`). The oracle tree is a
//! CPU build, so a set is the CPU run and the taps family holds no `.cpu` twin; the text model is read for its vocabulary only, so
//! the prompt kinds wait for the seat's placed load. What this seat's sets state that a Qwen seat's do not is read from their own
//! lines ([`crate::clefvis`]): the `# projector pixtral` layout, `# positions normal` and the `[IMG]` token; an image's rows are its
//! tokens and the break rows between its rows of tokens ([`crate::clefvis::ProjectorKind::rows`]).

use crate::RefError;
use crate::clefvis::{Profile, ProjectorKind, vis_families};

/// The projector every Mistral Large 4 image-input set is dumped with: the bf16 mmproj beside the text model's shards under the
/// box's `/models/Mistral-Large-4-1T-A52B-UD-IQ1_S`.
pub const MMPROJ: &str = "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/mmproj-BF16.gguf";

/// The sha256 of [`MMPROJ`] (3,444,696,448 bytes).
// PIN(2026-11-02): `sha256sum` of the file on the box (/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/mmproj-BF16.gguf), read once;
// `clefvis.sh` holds the file to it at every dump.
pub const MMPROJ_SHA256: &str = "1a21339f215e565850a7565a794e4ed43dde1d1097eddcfe8b9183cca7e4a307";

/// The mainline commit the sets name in their `# build` line: the tree of the node dumps ([`super::COMMIT`],
/// `/home/user/llama.cpp-ml4`), a CPU build. `clefvis.sh` writes the tree's commit alone, not the libllama digest the node dumps'
/// build line carries.
pub const LCPP_BUILD: &str = super::COMMIT;

/// The image token `[IMG]` in Mistral Large 4's vocabulary.
pub const IMAGE_TOKEN_ID: u32 = 10;

/// The seat: the projector, the text width 6,144 and the id of `[IMG]`.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ,
    mmproj_sha256: MMPROJ_SHA256,
    n_embd: 6144,
    image_pad_id: IMAGE_TOKEN_ID,
    projector: ProjectorKind::Pixtral,
    respell: None,
};

/// [`MMPROJ`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ.to_string())
}

vis_families! {
    recipe: "just dump-ref-ml4vis",
    prefix: "ref_mistrallarge4_vis",
    name: "mistrallarge4-vis",
    profile: PROFILE,
    arch: super::ARCH,
    build: LCPP_BUILD,
    mmproj: mmproj,
    // The two tower kinds name the projector as their file; the prompt kinds' text model waits for the seat's placed load.
    text: mmproj,
    kinds: [preproc, taps_cpu_only],
}
