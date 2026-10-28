//! The gate's rows: one MLX split each, and the clauses its architecture has
//! no contract for.
//!
//! A clause reads the file's own numbers from the session (`Session::geom`,
//! `Session::token_layers`); a row names only what the file cannot say: which
//! fixture directory the split lives in, the d1k set whose prompt ids the
//! self-bit clauses feed, whether the file has a recurrent store, which
//! kernel mutants edit its program, and the clauses that hold a contract the
//! file's architecture does not have, each with its reason, printed as that
//! clause's SKIP line.

use std::sync::OnceLock;

use refset::family::Family;

/// The kernel mutants a row runs `--mutant` and `--mutants-all` over: the
/// tables of `mutants.rs` that edit the program the row's instance runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutants {
    /// The routed FFN's: every table but the dense row's.
    Routed,
    /// The dense FFN's: the pair and fold launches at one slot.
    Dense,
    /// None: the row runs the kernels of another row's tables.
    None,
}

/// One model file the gate runs over.
pub struct Row {
    /// The name `--row` takes.
    pub name: &'static str,
    /// The architecture the fixture directory is named by.
    pub arch: &'static str,
    /// The family of the d1k set the prompt ids are read from.
    pub ik: &'static Family,
    /// The set whose prompt ids the self-bit clauses feed.
    pub d1k: &'static str,
    /// Whether the file has delta layers: a recurrent state with checkpoints,
    /// a conv ring and a pass's copies of them.
    pub recurrent: bool,
    /// The kernel mutants that edit this row's program.
    pub mutants: Mutants,
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
    d1k: refset::arch::qwen35moe::fixture::D1K,
    recurrent: true,
    mutants: Mutants::Routed,
    skips: &[],
};

/// Qwen3-30B-A3B's fixture: attention layers only, no shared expert.
pub static QWEN3MOE: Row = Row {
    name: "qwen3moe",
    arch: refset::arch::qwen3moe::ARCH,
    ik: &refset::arch::qwen3moe::fixture::IK,
    d1k: refset::arch::qwen3moe::fixture::D1K,
    recurrent: false,
    mutants: Mutants::None,
    skips: &[
        (
            "q_state",
            "the clause compares a delta layer's state and conv ring and the file has no delta \
             layer; q_chunk holds the pass's key and value rows",
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

/// Qwen3.8-27B's fixture: the gated delta net beside the attention layers and
/// a dense FFN in every layer, no router and no experts. The prompt ids are
/// Qwen3.6's d1k set's: the two families share a vocabulary and a tokenizer.
pub static QWEN38D: Row = Row {
    name: "qwen38d",
    arch: "qwen38d",
    ik: &refset::arch::qwen35moe::fixture::IK,
    d1k: refset::arch::qwen35moe::fixture::D1K,
    recurrent: true,
    mutants: Mutants::Dense,
    skips: &[(
        "p_profile",
        "not on a dense FFN: the clause walks each layer's router picks and the experts' stack \
         slices of the file's own tensors",
    )],
};

/// Every row, the default first.
pub static ROWS: [&Row; 3] = [&QWEN35MOE, &QWEN3MOE, &QWEN38D];

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
