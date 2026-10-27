//! The gate's rows: one fixture file each, with the family its reference sets
//! belong to and the clauses its architecture has no contract for.
//!
//! A clause reads the file's own numbers from the session (`Session::geom`,
//! `Session::token_layers`); a row names only what the file cannot say: which
//! fixture directory and which set family the run belongs to, whether the file
//! has a recurrent store, and the clauses that hold a contract the file's
//! architecture does not have, each with its reason, printed as that clause's
//! SKIP line.

use std::sync::OnceLock;

use refset::family::Family;

/// One model file the gate runs over.
pub struct Row {
    /// The name `--row` takes.
    pub name: &'static str,
    /// The architecture the fixture directory is named by.
    pub arch: &'static str,
    /// The family of ik's sets dumped from the fixture file.
    pub ik: &'static Family,
    /// The set of the prompt's tokens from position 0.
    pub batch: &'static str,
    /// The set of the prompt, then one decode step with every node dumped.
    pub step4: &'static str,
    /// The set of the step at the depth of its own prompt.
    pub d1k: &'static str,
    /// Whether the file has delta layers: a recurrent state with checkpoints,
    /// a conv ring and a pass's copies of them.
    pub recurrent: bool,
    /// The clauses the row does not run, each with the contract it lacks.
    pub skips: &'static [(&'static str, &'static str)],
}

impl Row {
    /// The reason clause `name` is not run on this row.
    pub fn skip_of(&self, name: &str) -> Option<&'static str> {
        self.skips
            .iter()
            .find(|(clause, _)| *clause == name)
            .map(|(_, why)| *why)
    }
}

/// Qwen3.6-35B-A3B's fixture: the gated delta net beside the attention layers.
pub static QWEN35MOE: Row = Row {
    name: "qwen35moe",
    arch: refset::arch::qwen35moe::ARCH,
    ik: &refset::arch::qwen35moe::fixture::IK,
    batch: refset::arch::qwen35moe::fixture::BATCH,
    step4: refset::arch::qwen35moe::fixture::STEP4,
    d1k: refset::arch::qwen35moe::fixture::D1K,
    recurrent: true,
    skips: &[],
};

/// Qwen3-30B-A3B's fixture: attention layers only, no shared expert.
pub static QWEN3MOE: Row = Row {
    name: "qwen3moe",
    arch: refset::arch::qwen3moe::ARCH,
    ik: &refset::arch::qwen3moe::fixture::IK,
    batch: refset::arch::qwen3moe::fixture::BATCH,
    step4: refset::arch::qwen3moe::fixture::STEP4,
    d1k: refset::arch::qwen3moe::fixture::D1K,
    recurrent: false,
    skips: &[
        (
            "k_keep",
            "checkpoints hold a recurrent state and the file has none: a cut keeps any position up \
             to it, which t_target and s_serve hold",
        ),
        ("m_mlx", "the MLX-format split is the qwen35moe fixture's"),
        (
            "q_state",
            "the clause compares a delta layer's state and conv ring and the file has no delta \
             layer; q_chunk and q_ref hold the pass's key and value rows",
        ),
        (
            "q_mark",
            "a pass's marks copy its delta stores and the file has none",
        ),
        (
            "q_mark_cut",
            "a cut back to a mark restores a delta store and the file has none",
        ),
    ],
};

/// Every row, the default first.
pub static ROWS: [&Row; 2] = [&QWEN35MOE, &QWEN3MOE];

/// The row `--row` names.
pub fn find(name: &str) -> Option<&'static Row> {
    ROWS.iter().copied().find(|r| r.name == name)
}

static CURRENT: OnceLock<&'static Row> = OnceLock::new();

/// Fix the row this process runs. Once, before any clause.
pub fn select(row: &'static Row) {
    if CURRENT.set(row).is_err() {
        panic!("the gate selects its row once, in main");
    }
}

/// The row this process runs.
pub fn current() -> &'static Row {
    CURRENT
        .get()
        .expect("main selects the row before a clause reads it")
}
