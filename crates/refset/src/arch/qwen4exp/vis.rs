//! The Qwen3.8-Flash-Next image-input families: llama.cpp mainline's mtmd `qwen3vl_merger` tower and its qwen4exp text
//! model on [`MODEL`], dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-qvis qwen4exp`) in the node dumps'
//! set format, at the mainline tree [`QVIS_LCPP_BUILD`], with the bf16 projector [`MMPROJ_BF16`]. One family a kind of
//! set, as the Clef-Flash families are ([`crate::arch::qwen35::clefvis`]); the kinds and what they hold are
//! [`crate::clefvis::Kind`]'s. A set of the Clef-Flash seat, or of Qwen3.6's, fails these families' check by its model
//! file or its projector.

use super::{ARCH, MODEL};
use crate::RefError;
use crate::clefvis::{Profile, QVIS_LCPP_BUILD, vis_families};

/// The projector every Qwen3.8 image-input set is dumped with: unsloth's bf16 file from the repository of the seat's
/// default `--hf` (`crates/gpu-gates/src/bin/bloomery_serve.rs`), under the box's `/models/mmproj`.
pub const MMPROJ_BF16: &str = "/models/mmproj/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf";

/// The sha256 of [`MMPROJ_BF16`], the LFS oid of that file (907,542,944 bytes).
// PIN(2026-10-10): unsloth/Qwen3.8-Flash-Next-GGUF at rev 766911a6b736, read from its tree listing
// (huggingface.co/api/models/<repo>/tree/main) and equal to `sha256sum` of the downloaded file.
pub const MMPROJ_BF16_SHA256: &str =
    "2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355";

/// The seat: the projector, the text width 2,560 and the image-pad id of the shared Qwen vocabulary.
pub static PROFILE: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: 2560,
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
    prefix: "ref_qwen4exp_vis",
    name: "qwen4exp-vis",
    model: "qwen4exp",
    profile: PROFILE,
    arch: ARCH,
    build: QVIS_LCPP_BUILD,
    mmproj: mmproj,
    text: model,
}
