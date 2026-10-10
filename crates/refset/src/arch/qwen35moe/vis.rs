//! The Qwen3.6-35B-A3B image-input families: llama.cpp mainline's mtmd `qwen3vl_merger` tower and its qwen35moe text
//! model on [`MODEL`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-qvis qwen35moe`) in the node dumps'
//! set format, at the mainline tree [`QVIS_LCPP_BUILD`], with the bf16 projector [`MMPROJ_BF16`]. One family a kind of
//! set, as the Clef-Flash families are ([`crate::arch::qwen35::clefvis`]); the kinds and what they hold are
//! [`crate::clefvis::Kind`]'s. A set of the Clef-Flash seat, or of Qwen3.8's, fails these families' check by its model
//! file or its projector.

use super::{ARCH, MODEL};
use crate::RefError;
use crate::clefvis::{Profile, QVIS_LCPP_BUILD, vis_families};

/// The projector every Qwen3.6 image-input set is dumped with: lmstudio-community's bf16 file, the repository
/// `docs/models.md` names for this model, under the box's `/models/mmproj`.
pub const MMPROJ_BF16: &str =
    "/models/mmproj/lmstudio-community--Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`], the LFS oid of that file (902,822,016 bytes).
// PIN(2026-10-10): lmstudio-community/Qwen3.6-35B-A3B-GGUF at rev 68a34855558a, read from its tree listing
// (huggingface.co/api/models/<repo>/tree/main) and equal to `sha256sum` of the downloaded file.
pub const MMPROJ_BF16_SHA256: &str =
    "e5c205cec2fd28f66c3895e4040021ab994b860323c3db8531640305ff49b322";

/// The seat: the projector, the text width 2,048 and the image-pad id of the shared Qwen vocabulary.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: 2048,
    image_pad_id: 248_056,
};

/// [`MMPROJ_BF16`], as a tower family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_BF16.to_string())
}

/// [`MODEL`], as a prompt family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

vis_families! {
    prefix: "ref_qwen35moe_vis",
    name: "qwen35moe-vis",
    model: "qwen35moe",
    profile: PROFILE,
    arch: ARCH,
    build: QVIS_LCPP_BUILD,
    mmproj: mmproj,
    text: model,
}
