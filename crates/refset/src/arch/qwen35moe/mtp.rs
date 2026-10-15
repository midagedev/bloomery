//! The Qwen3.6 MTP step oracle: llama.cpp mainline's next-token graph
//! (`llama_model_qwen35moe::graph_mtp`) tap by tap, on the first 64 prose ids fed teacher-forced
//! (`tools/ref/q35mtp_ref.cpp`, `just dump-ref-q35mtp`), read by [`crate::mtptaps::TapSet`], at the mainline tree
//! [`QVIS_LCPP_BUILD`] (`MTP_LCPP` of tools/ref/models/qwen35moe.sh). The file is
//! unsloth's UD-Q4_K_M, the one with the next-token layer (`blk.40`); the node dumps' file has none.

use crate::RefError;
use crate::clefvis::QVIS_LCPP_BUILD;
use crate::family::{Build, Family, Identity};

/// The file the set is dumped from and the tree runs: the Q4_K_M mix that carries the next-token layer.
pub const MODEL: &str = "/models/qwen36-hf/unsloth-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

/// The set: 63 steps over the first 64 prose ids, taps at four of them.
pub const SET: &str = "ref-mtp/qwen35moe_mainline_prose64";

/// [`MODEL`], as the family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's MTP step taps (`just dump-ref-q35mtp`).
pub static MAINLINE: Family = Family {
    name: "mtp-qwen35moe-mainline",
    sets: &[SET],
    resolve: None,
    recipe: "just dump-ref-q35mtp",
    identity: Identity::MtpTaps,
    arch: Some(super::ARCH),
    build: Some(Build::Is(QVIS_LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

#[cfg(test)]
mod tests;
