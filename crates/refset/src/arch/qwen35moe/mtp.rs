//! The Qwen3.6 MTP step oracle: llama.cpp mainline's next-token graph
//! (`llama_model_qwen35moe::graph_mtp`) tap by tap, on the first 64 prose ids fed teacher-forced
//! (`tools/ref/q35mtp_ref.cpp`, `just dump-ref-q35mtp`), read by [`crate::mtptaps::TapSet`]. The file is
//! unsloth's UD-Q4_K_M, the one with the next-token layer (`blk.40`); the node dumps' file has none.

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The file the set is dumped from and the tree runs: the Q4_K_M mix that carries the next-token layer.
pub const MODEL: &str = "/models/qwen36-hf/unsloth-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

/// The mainline commit the set names in its `# build` line (`git rev-parse --short=9`).
// PIN(2026-10-15): /home/user/llama.cpp-36a73916 at its HEAD, the tree q35mtp_ref links against
// (`MTP_LCPP` of tools/ref/models/qwen35moe.sh).
pub const MAINLINE_BUILD: &str = "36a73916e";

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
    build: Some(Build::Is(MAINLINE_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

#[cfg(test)]
mod tests;
