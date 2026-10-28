//! The gate's rows: one MLX model each, and the clauses its architecture has
//! no contract for.
//!
//! A clause reads the file's own numbers from the session (`Session::geom`,
//! `Session::token_layers`); a row names only what the file cannot say: which
//! fixture directory its model lives in and whether that is a split or the
//! checkpoint folder, where the self-bit clauses' prompt ids come from, the
//! id a generation starts at, whether the file has a recurrent store (a delta
//! state or a conv ring), the band its logits are held to against the pin,
//! the count of groups a token encodes where it is derived on paper, which
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

/// The file a row's clauses open, under the fixture root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// The first shard of `<arch>-mlxsplit`: the GGUF split `prepare` made of
    /// the folder `<arch>-mlx`.
    Split,
    /// `<arch>-mlx`, the checkpoint folder itself: a family `prepare` has no
    /// map for has no split, and the session opens the folder.
    Folder,
}

/// Where the ids of a row's prompts come from.
pub enum Prompt {
    /// The first ids of an ik d1k reference set's prompt, in the vocabulary
    /// of the family that set belongs to.
    D1k {
        /// The family of the set the prompt ids are read from.
        ik: &'static Family,
        /// The set whose prompt ids the self-bit clauses feed.
        set: &'static str,
    },
    /// Ids a fixed generator draws, for a vocabulary no d1k set speaks: `first`,
    /// then ids in `lo..hi`. The self-bit clauses compare the session with its
    /// own stepped path and need ids the vocabulary holds, not a text.
    Generated { first: u32, lo: u32, hi: u32 },
}

/// The groups and dispatches one token encodes, derived on paper from the
/// file's layer list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    pub groups: u32,
    pub dispatches: u32,
}

/// One model file the gate runs over.
pub struct Row {
    /// The name `--row` takes.
    pub name: &'static str,
    /// The architecture the fixture directory is named by.
    pub arch: &'static str,
    /// The file the clauses open.
    pub model: Model,
    /// Where the self-bit clauses' prompt ids come from.
    pub prompt: Prompt,
    /// The id every generation and prompt starts at: one the family's model
    /// runs from, which is not always id 0.
    pub seed: u32,
    /// Whether the file has a recurrent store: a delta layer's state and conv
    /// ring, or a short conv layer's ring, each with checkpoints.
    pub recurrent: bool,
    /// The largest |session − pin| on a logit `m_logits` accepts.
    pub logit_abs: f64,
    /// The groups and dispatches a token encodes where the row derives them
    /// from the layer list on paper, held beside the count model's own.
    pub count: Option<Count>,
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

/// PIN(2026-10-28): max |session − mlx-lm| on a logit, at each pinned step.
/// The pin is mlx-lm's modules in f32 over the exact dequant of the same
/// affine-4 weights (`tools/mlx-logits.py dump`), so the gap is the
/// session's own rounding. A bound: were every projection's input rounded
/// once into half (a relative error uniform within ±2^-11, rms
/// 2^-11/√3 = 2.8e-4), a dot of 2,048 products would carry Δ = 2.8e-4 of
/// its rms. The fixture is four layers plus the head: five projections. The
/// pins' largest logit is 5.03, so a logit moves at most 5Δ · 5.03 = 7.0e-3;
/// the band rounds that up. Measured: the worst healthy step is 1.92e-3,
/// and the nearest mutant (`rope_key_row_zero`) is 0.739 away.
const QWEN_LOGIT_ABS: f64 = 0.01;

/// PIN(2026-10-29): max |session − pin| on a logit of LFM3-1.7B at each
/// pinned step; the pin is mlx-vlm's `lfm3` in f32 over the exact dequant of
/// the folder's 4-bit weights. The Qwen band's error model, Δ = 2.8e-4 of a
/// dot's rms a projection whose input is rounded once into half, over this
/// model's 97 projections (four a layer, then the head) at its largest pinned
/// logit, 20.87: summed outright that is 0.57, past [`LFM3_LOGIT_CAP`]; the
/// projections' errors are independent roundings and add in quadrature,
/// √97 · 2.8e-4 · 20.87 = 0.058, rounded up. Measured: the worst step is
/// 3.07e-3 (step 3); a band of 1e-3 turns `m_logits` red.
const LFM3_LOGIT_ABS: f64 = 0.06;

/// Half the smallest top-2 gap of the pin's four rows, 0.8085 at step 1.
const LFM3_LOGIT_CAP: f64 = 0.40;

const _: () = assert!(LFM3_LOGIT_ABS <= LFM3_LOGIT_CAP);

/// Qwen3.6-35B-A3B's fixture: the gated delta net beside the attention layers.
pub static QWEN35MOE: Row = Row {
    name: "qwen35moe",
    arch: refset::arch::qwen35moe::ARCH,
    model: Model::Split,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen35moe::fixture::IK,
        set: refset::arch::qwen35moe::fixture::D1K,
    },
    seed: 0,
    recurrent: true,
    logit_abs: QWEN_LOGIT_ABS,
    count: None,
    mutants: Mutants::Routed,
    skips: &[],
};

/// Qwen3-30B-A3B's fixture: attention layers only, no shared expert.
pub static QWEN3MOE: Row = Row {
    name: "qwen3moe",
    arch: refset::arch::qwen3moe::ARCH,
    model: Model::Split,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen3moe::fixture::IK,
        set: refset::arch::qwen3moe::fixture::D1K,
    },
    seed: 0,
    recurrent: false,
    logit_abs: QWEN_LOGIT_ABS,
    count: None,
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
    model: Model::Split,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen35moe::fixture::IK,
        set: refset::arch::qwen35moe::fixture::D1K,
    },
    seed: 0,
    recurrent: true,
    logit_abs: QWEN_LOGIT_ABS,
    count: None,
    mutants: Mutants::Dense,
    skips: &[(
        "p_profile",
        "not on a dense FFN: the clause walks each layer's router picks and the experts' stack \
         slices of the file's own tensors",
    )],
};

/// LFM3-1.7B's checkpoint folder: a short conv beside the attention layers,
/// 13 conv, 6 full and 5 sliding-window, a sandwich residual in every layer, a
/// dense FFN and a tied head. No `prepare` map reads the family, so the
/// folder is the model. Its vocabulary is not Qwen's and no d1k set exists, so
/// the prompts are generated; the generation starts at the BOS id, since id 0
/// degenerates the model in the pin's own reference.
///
/// The count is 24 layers of seven groups, P1, mixer, out, join, FFN pair,
/// down, join, each one dispatch, and the embed and the head: 24 · 7 + 2.
pub static LFM3: Row = Row {
    name: "lfm3",
    arch: refset::arch::lfm3::ARCH,
    model: Model::Folder,
    prompt: Prompt::Generated {
        first: 124_894,
        lo: 256,
        hi: 100_000,
    },
    seed: 124_894,
    recurrent: true,
    logit_abs: LFM3_LOGIT_ABS,
    count: Some(Count {
        groups: 170,
        dispatches: 170,
    }),
    mutants: Mutants::None,
    skips: &[(
        "m_mlx_folder",
        "the folder is every clause's model and there is no prepared split to tell it against; \
             `m_logits` holds the folder's own logits to the pin",
    )],
};

/// Every row, the default first.
pub static ROWS: [&Row; 4] = [&QWEN35MOE, &QWEN3MOE, &QWEN38D, &LFM3];

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
