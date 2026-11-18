//! The V4.1 attention-side gates, one process: the hyper-connections (`hc`),
//! the compressor and index key (`comp`), the indexer (`index`), the engram
//! host half (`engram`), the attention (`attn`) and the attention piece
//! (`chain_attn`), each on the clauses that need no ik dump: seeded or
//! synthetic inputs against this binary's transcription of our rule, depth
//! cases, graph replays and faults on the file the tree runs. A gate is the
//! module of its name (`shared/ds41_<name>_gate.rs`, whose head says what it
//! pins) and runs on a card handle and a reading of the model file of its own;
//! it keeps its name `gate_deepseek41_<name>` in the lines it prints and in
//! its errors.
//!
//! PIN(2026-11-18): the real-file ik set retired, and with it the rope, woa and chain_glue arms, every clause of which compared with the dump; ik cannot read bq3km's engram schema; no fixture set carries these clauses
//!
//! `gate_deepseek41_kernels [GATE...]` runs the named gates, or all six in
//! the order below; a gate that fails does not stop the next one, and the
//! process ends red when any failed (`shared/gate_set.rs`).

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_attn_gate.rs"]
mod attn;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_chain_attn_gate.rs"]
mod chain_attn;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_comp_gate.rs"]
mod comp;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_cand.rs"]
mod ds41_cand;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_engram_gate.rs"]
mod engram;
#[cfg(feature = "deepseek41")]
#[path = "shared/gate_set.rs"]
mod gate_set;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_hc_gate.rs"]
mod hc;
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_index_gate.rs"]
mod index;

/// The gates in run order: the op blocks first, then the pieces that chain them.
#[cfg(feature = "deepseek41")]
const GATES: [&str; 6] = ["hc", "comp", "index", "engram", "attn", "chain_attn"];

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    // A retired lever name that is set is refused by name before anything loads.
    if let Err(e) = bloomery_levers::at_main(&[]) {
        return bloomery_gpu_gates::exit_with("gate_deepseek41_kernels", Err(e.into()));
    }
    gate_set::run_set(
        "gate_deepseek41_kernels",
        "gate_deepseek41_",
        &GATES,
        std::env::args().skip(1),
        |name| match name {
            "hc" => hc::run(),
            "comp" => comp::run(),
            "index" => index::run(),
            "engram" => engram::run(),
            "attn" => attn::run(),
            "chain_attn" => chain_attn::run(),
            other => unreachable!("run_set passes only the names of GATES, got {other:?}"),
        },
    )
}
