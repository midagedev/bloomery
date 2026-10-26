//! The dense Nanbeige4.3-33B generation gate: the `llama` file (52 layers of grouped-query attention, 56/8 heads of 128
//! turned by NORM pairs over the whole head with no q/k norm and no output gate, each with a dense SwiGLU FFN of 22,016
//! and an untied head) whole on one card, the Body35 chain a request runs: the prompt call, the decode step, the pass,
//! the checkpoints and the resident slots, against llama.cpp mainline's forced-logits rows (`refset::arch::llama::dense`,
//! `just dump-logits-nanbeige43`). The trunk has no delta layer, so the load holds no store to copy, a cut keeps every
//! position, and the group of 7 query heads a KV head runs the tensor-core flash pass only (the load opens with `mma`).
//! The file fits either card, so the gate holds the same clauses in both tiers.
//!
//! The clauses (a) to (f) are the shared dense-gate module's (`shared/dense_e2e.rs`, the same code for every dense
//! Body35 file); this bin is the Nanbeige4.3-33B's row of it: the file, its sets, its header's geometry and the bands
//! below.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_llama_dense", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen35_open.rs"]
#[allow(
    dead_code,
    reason = "the loader serves the hidden, media and dense gates; this one reads the load and the whole-fit verdict"
)]
mod qwen35_open;

#[cfg(feature = "gpu")]
#[path = "shared/q35_clauses.rs"]
mod q35_clauses;

#[cfg(feature = "gpu")]
#[path = "shared/dense_e2e.rs"]
mod dense_e2e;

#[cfg(feature = "gpu")]
mod gate {
    use super::dense_e2e::{self, Bands, Row};
    use super::q35_clauses::Geometry35;
    use bloomery_gpu_gates::GateError;
    use refset::arch::llama::dense::{MODEL, Nanbeige43, P64, P4096};

    /// The real file's literals, beside the geometry its header gives (the move proof, `Geometry35::of`): 52 attention
    /// layers of 8 key heads of 128, and no delta layer.
    const REAL: Geometry35 = Geometry35 {
        n_attn: 52,
        n_delta: 0,
        n_kv: 8,
        head: 128,
        n_v: 0,
        head_v: 0,
        conv_c: 0,
    };

    /// PIN(2026-10-26): the bound on one row's relative L2 distance from mainline's CUDA row. Derivation [derived]:
    /// this engine and mainline read the same Q4_K/Q5_K/Q6_K/Q8_0 codes and differ in the 8-bit activations and the
    /// f32 sum order, as mainline's CPU twin does; the twin's rows against its CUDA rows are the oracle's own floor
    /// (`dense-logits.py floor`, per set): per-row relative L2 at most 0.0684 (P64) and 0.0808 (P4096). 0.11 is 1.36
    /// times the larger, as the Qwen3.8-27B row's 1.34. A wiring fault (a layer, a position or a forced id off by one)
    /// moves a row by order one.
    const REL_BAND: f64 = 0.11;
    /// PIN(2026-10-26): the bound on a set's median row [derived]. The floor's per-set medians are 0.0191 (P64) and
    /// 0.0198 (P4096); 0.025 is 1.26 times the larger, the factor the Qwen3.8-27B row's median band takes (1.25).
    const MEDIAN_BAND: f64 = 0.025;
    /// PIN(2026-10-26): the bound on a set's 90th percentile [derived]. The floor's per-set p90s are 0.0404 (P64) and
    /// 0.0451 (P4096); 0.059 is 1.31 times the larger, the factor the Qwen3.8-27B row's p90 band takes (1.3).
    const P90_BAND: f64 = 0.059;
    /// PIN(2026-10-26): the factor on a row's stability bound. Two logits swap only if their margin is within the
    /// difference of their two moves, at most `min(2 · max_abs, √2 · l2)` of the move's vector
    /// (`dense-logits.py stable_margin`); the move taken is the oracle's own spread on that row (twin against CUDA,
    /// the same positions amplify for any two realizations) times 1.3, the factor the p90 band takes.
    const STABLE_K: f64 = 1.3;
    /// PIN(2026-10-26): the rows at or under their stability bound (the only rows that could flip), per set: 28 of 32
    /// and 15 of 32 at [`STABLE_K`] (26 and 12 at a factor of 1, the floor's report). A count that moves means the
    /// sets or the twin moved, not the engine.
    const OTHER_ROWS: [usize; 2] = [28, 15];

    /// The Nanbeige4.3-33B's row: the tensor-core flash, the only pass a group of 7 has.
    const ROW: Row = Row {
        gate: "gate_llama_dense",
        model: MODEL,
        sets: [P64, P4096],
        real: REAL,
        unused_block: None,
        mma: true,
        bands: Bands {
            rel: REL_BAND,
            median: MEDIAN_BAND,
            p90: P90_BAND,
            stable_k: STABLE_K,
            other_rows: OTHER_ROWS,
        },
    };

    pub fn run() -> Result<(), GateError> {
        dense_e2e::run::<Nanbeige43>(&ROW)
    }
}
