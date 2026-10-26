//! The dense Qwen3.8-27B generation gate: the `qwen35` file (64 layers, 48 gated-delta and 16 gated GQA at 24/4 heads of
//! 256, each with a dense SwiGLU FFN of 17,408; the 15 `blk.64.*` next-token tensors unused) whole on one card, the
//! Body35 chain a request runs: the prompt call, the decode step, the pass, the checkpoints and the resident slots,
//! against llama.cpp mainline's forced-logits rows (`refset::arch::qwen35::dense`, `just dump-logits-qwen38d`). The
//! file fits either card, so the gate holds the same clauses in both tiers. `gate_clef_hidden` holds the trunk's
//! hidden-state band on the same geometry; this gate does not repeat it.
//!
//! The clauses (a) to (f) are the shared dense-gate module's (`shared/dense_e2e.rs`, the same code for every dense
//! Body35 file); this bin is the Qwen3.8-27B's row of it: the file, its sets, its header's geometry and the bands below.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35_dense", gate::run())
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
    use refset::arch::qwen35::dense::{MODEL, P64, P4096, Qwen38d};

    /// The real file's literals, beside the geometry its header gives (the move proof, `Geometry35::of`).
    const REAL: Geometry35 = Geometry35 {
        n_attn: 16,
        n_delta: 48,
        n_kv: 4,
        head: 256,
        n_v: 48,
        head_v: 128,
        conv_c: 10240,
    };
    /// The `blk.<n>.` prefix of the next-token block the Body35 reader leaves unused (`qwen35.nextn_predict_layers`
    /// is 1, after the 64 trunk layers).
    const UNUSED_BLOCK: &str = "blk.64.";

    /// PIN(2026-10-19): the bound on one row's relative L2 distance from mainline's CUDA row. Derivation: this engine
    /// and mainline read the same Q4_K/Q6_K codes and differ in the 8-bit activations (ours q8_1 per 128 values,
    /// mainline's MMQ q8_1 per 32) and the f32 sum order, as mainline's CPU twin does (q8_K per 256); the twin's rows
    /// against its CUDA rows are the oracle's own floor (`dense-logits.py floor`): per-row relative L2 at most
    /// 0.1492 (64-id set) and 0.1008 (4,096-id set). 0.20 is 1.34 times the largest. A wiring fault (a layer, a
    /// position or a forced id off by one) moves a row by order one.
    const REL_BAND: f64 = 0.20;
    /// PIN(2026-10-19): the bound on a set's median row. The floor's medians are 0.0445 and 0.0525; 0.066 is 1.25
    /// times the largest, the factor `gate_clef_hidden`'s median band takes over its twin.
    const MEDIAN_BAND: f64 = 0.066;
    /// PIN(2026-10-19): the bound on a set's 90th percentile. The floor's are 0.0643 and 0.0685; 0.09 is 1.3 times the
    /// largest, as `gate_clef_hidden`'s p90 band.
    const P90_BAND: f64 = 0.09;
    /// PIN(2026-10-19): the factor on a row's stability bound. Two logits swap only if their margin is within the
    /// difference of their two moves, at most `min(2 · max_abs, √2 · l2)` of the move's vector
    /// (`dense-logits.py stable_margin`); the move taken is the oracle's own spread on that row (twin against CUDA,
    /// the same positions amplify for any two realizations) times 1.3, the factor the p90 band takes.
    const STABLE_K: f64 = 1.3;
    /// PIN(2026-10-19): the rows at or under their stability bound (the only rows that could flip), per set: 20 of 32
    /// and 17 of 32 at [`STABLE_K`] (28 of 64 at a factor of 1, the floor's report). A count that moves means the
    /// sets or the twin moved, not the engine.
    const OTHER_ROWS: [usize; 2] = [20, 17];

    /// The Qwen3.8-27B's row: the scalar flash (the file's group of 6 runs the pairs' pass, which has no tensor-core
    /// form).
    const ROW: Row = Row {
        gate: "gate_qwen35_dense",
        model: MODEL,
        sets: [P64, P4096],
        real: REAL,
        unused_block: Some(UNUSED_BLOCK),
        mma: false,
        bands: Bands {
            rel: REL_BAND,
            median: MEDIAN_BAND,
            p90: P90_BAND,
            stable_k: STABLE_K,
            other_rows: OTHER_ROWS,
        },
    };

    pub fn run() -> Result<(), GateError> {
        dense_e2e::run::<Qwen38d>(&ROW)
    }
}
