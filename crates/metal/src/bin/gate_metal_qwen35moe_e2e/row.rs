//! The gate's rows: one MLX model each, and the clauses its architecture has
//! no contract for.
//!
//! A clause reads the file's own numbers from the session (`Session::geom`,
//! `Session::token_layers`); a row names only what the file cannot say: which
//! fixture directory its model lives in and whether that is a split or the
//! checkpoint folder, where the self-bit clauses' prompt ids come from, the
//! id a generation starts at, whether the file has a recurrent store (a delta
//! state or a conv ring), the band its logits are held to against the pin and
//! the band of its deep pin where it has one, the count of groups a token
//! encodes where it is derived on paper, which kernel mutants edit its program,
//! and the clauses that hold a contract the file's architecture does not have,
//! each with its reason, printed as that clause's SKIP line.

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
    /// LFM3's: the windowed flash and the short conv's step ring, which only the
    /// row's deep pin reads (`m_logits_deep`).
    Lfm3,
    /// Llama's: the prologue of the no-head-norm, NORM-pairs attention, which only the dense
    /// Llama instance compiles.
    Llama,
    /// The dense Qwen3's: the head norm and the NEOX pairs of its group-of-four attention, which
    /// the first position reads none of (one key, a rope angle of 0).
    Qwen3,
    /// The dense Qwen2's: the bias add after P1, which every other row's layers have none of.
    Qwen2,
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
    /// The largest |session − pin| on a logit `m_logits_deep` accepts at the
    /// deep pin's positions; None where the row holds no deep pin.
    pub deep_logit_abs: Option<f64>,
    /// The groups and dispatches a token encodes where the row derives them
    /// from the layer list on paper, held beside the count model's own.
    pub count: Option<Count>,
    /// The groups and dispatches a verify of two rows encodes where the row
    /// derives them on paper from the layer list, held beside the count
    /// model's own (`count::verify_count`).
    pub verify: Option<Count>,
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

/// PIN(2026-10-29): max |session − pin| on a logit of LFM3-1.7B at each step of its
/// deep pin (`pins/deep/lfm3.txt`: a 600-id walk, rows kept at steps 255, 256, 257,
/// 300, 302, 357 and 599). [`LFM3_LOGIT_ABS`] sums the projections' rounding outright
/// and would see a window one key off, the defect this pin is for, at two of the
/// pinned rows only (mlx-vlm in f32: a window of 255 keys moves them by 0.023 to
/// 0.113, of 257 keys by 0 to 0.153, 0 where the window has not begun to cut). The
/// band is the same rounding rule simulated, not summed: mlx-vlm in f32 with the input of every
/// projection and the head, and the K and V it stores, rounded once into half, against
/// the pin's own rows. At the pinned steps its worst is 1.43e-2 (step 357), and at the
/// BOS pin's four steps 1.48e-2 where the session's worst is 3.07e-3 (`LFM3_LOGIT_ABS`),
/// so the simulation sits above the session there; the band rounds its worst up.
/// Ignoring the rope's factors moves the pinned rows by 0.34 to 0.87.
const LFM3_DEEP_LOGIT_ABS: f64 = 0.02;

/// Half the smallest top-2 gap of the deep pin's seven rows, 0.0203 at step 256, rounded
/// down.
const LFM3_DEEP_LOGIT_CAP: f64 = 0.02;

const _: () = assert!(LFM3_DEEP_LOGIT_ABS <= LFM3_DEEP_LOGIT_CAP);

/// PIN(2026-11-01): max |session − pin| on a logit of the 4-layer cut of Llama-3.1-8B-Instruct's
/// folder at each pinned step (`pins/llama.txt`); the pin is mlx-lm's `llama` in f32 over the
/// exact dequant of the folder's 4-bit weights. The Qwen band's error model, Δ = 2.8e-4 of a dot's
/// rms a projection whose input is rounded once into half, over the rounded inputs of the cut's
/// path (the shared input of q, k and v, the output projection's, the shared input of gate and up,
/// the down's: four a layer, then the head's, 17 for four layers), added in quadrature: √17 ·
/// 2.8e-4 = 1.15e-3 of a logit's scale. The pin's largest logit is 11.44 (step 1), so the band is
/// 1.15e-3 · 11.44 = 0.0132, rounded up. Measured: the worst step is 3.89e-3 (step 2), and the
/// mutant `rope_pair_reversed` is 8.13 away at step 2.
const LLAMA_LOGIT_ABS: f64 = 0.014;

/// Half the smallest top-2 gap of the pin's four rows, 0.1917 at step 3, rounded down.
const LLAMA_LOGIT_CAP: f64 = 0.19;

const _: () = assert!(LLAMA_LOGIT_ABS <= LLAMA_LOGIT_CAP);

/// PIN(2026-11-01): max |session − pin| on a logit of the 4-layer cut of Qwen3-8B's folder at each
/// pinned step; the pin is mlx-lm's `qwen3` in f32 over the exact dequant of the folder's 4-bit
/// weights. Derived from the pin's largest logit, 17.57 (step 0), and the folder's head norm gains,
/// never fitted to a run: the Qwen band's error model, Δ = 2.8e-4 of a dot's rms a value rounded once
/// into half, added in quadrature over the roundings on the cut's path. A layer rounds the shared input
/// of q, k and v, the output projection's input, the shared input of gate and up, the down's, and the
/// key and the value the cache stores in half; the head rounds its input once. Qwen3 norms each query
/// and key head by its gain before the rope, so a score is the dot of two heads that each carry
/// one: a rounding of q's or k's input or of the stored key moves a score by Δ of the heads'
/// product scale (the rms of a layer's q_norm gain times the rms of its k_norm gain: 6.67, 3.23, 2.76
/// and 3.24 in layers 0 to 3), and the softmax carries a score's error into the attention's output at
/// that relative size. That is three roundings a layer at the product scale (the squares of those
/// four scales sum to 73.1) and five at unit scale (the value's input share, the output projection's
/// input, the gate-up's, the down's and the stored value), the head's one more, all taken as
/// independent: √(3 · 73.1 + 4 · 5 + 1) = 15.5, so 15.5 · 2.8e-4 · 17.57 = 0.076 of a logit,
/// rounded up. The norm's own f32 arithmetic and the sums' order are 1e-7 and left out. The Llama
/// row's count of four roundings a layer carries no head norm and no cache rounding.
const QWEN3_LOGIT_ABS: f64 = 0.08;

/// Half the smallest top-2 gap of the pin's four rows, 0.4657 at step 1, rounded down.
const QWEN3_LOGIT_CAP: f64 = 0.23;

const _: () = assert!(QWEN3_LOGIT_ABS <= QWEN3_LOGIT_CAP);

/// PIN(2026-11-02): max |session − pin| on a logit of the 4-layer cut of Qwen3-4B-Instruct-2507's
/// folder at each pinned step (`pins/qwen3t.txt`); the pin is mlx-lm's `qwen3` in f32 over the
/// exact dequant of the folder's 4-bit weights, its head the embedding table. Derived like
/// `QWEN3_LOGIT_ABS`, from the pin's largest logit, 17.56 (step 2), and the folder's head norm
/// gains: the rms of a layer's q_norm gain times the rms of its k_norm gain is 8.60, 3.46, 2.84
/// and 3.25 in layers 0 to 3, the squares of those four scales sum to 104.5, so
/// √(3 · 104.5 + 4 · 5 + 1) = 18.3, and 18.3 · 2.8e-4 · 17.56 = 0.090 of a logit, rounded up. The
/// tied head reads the embedding's 4-bit rows as its matrix: the head's one input rounding is the
/// same as an untied head's, and the table's bits are the pin's own.
const QWEN3T_LOGIT_ABS: f64 = 0.09;

/// Half the smallest top-2 gap of the pin's four rows, 0.3450 at step 2, rounded down.
const QWEN3T_LOGIT_CAP: f64 = 0.17;

const _: () = assert!(QWEN3T_LOGIT_ABS <= QWEN3T_LOGIT_CAP);

/// PIN(2026-11-02): max |session − pin| on a logit of the 4-layer cut of Qwen2.5-Coder-7B-Instruct's
/// folder at each pinned step (`pins/qwen2.txt`; the pin is mlx-lm's `qwen2` in f32 over the exact
/// dequant of the folder's 4-bit weights, its q, k and v biases added in f32). The summed rule of
/// [`QWEN3_LOGIT_ABS`] takes each score's rounding at the product of the q and k rms, and Qwen2's k
/// bias puts that product at 103 in layer 0: the sum, √(3 · 10766 + 4 · 5 + 1) · 2.8e-4 · 20.75 =
/// 1.1, passes the cap. The band is the same rounding rule simulated, not summed, as
/// [`LFM3_DEEP_LOGIT_ABS`]'s: `tools/mlx-logits.py simulate` runs the pin's model in f32 with the
/// input of every projection and of the head, and the K and V the cache stores, rounded once into
/// half. Its worst against the pin is 1.96e-2 at step 1 (0 with no rounding, 1.63e-2 with the
/// stored keys alone), rounded up.
const QWEN2_LOGIT_ABS: f64 = 0.02;

/// Half the smallest top-2 gap of the pin's four rows, 0.3054 at step 3, rounded down.
const QWEN2_LOGIT_CAP: f64 = 0.15;

const _: () = assert!(QWEN2_LOGIT_ABS <= QWEN2_LOGIT_CAP);

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
    deep_logit_abs: None,
    count: None,
    verify: None,
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
    deep_logit_abs: None,
    count: None,
    verify: None,
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
///
/// The verify's count is derived on paper from the four layers (three delta
/// layers and an attention layer, each over a plain residual that hands on):
/// a delta layer is P1 (1 group, 1 dispatch), the core (3 groups, 4
/// dispatches: the prep, the ring and the scan, the post), the out (1 group of
/// 1 dispatch), two hand-ons, the pair and the down, 9 groups and 10
/// dispatches; an attention layer is P1, the flash (2 groups, 2 dispatches:
/// the append, the flash), the out, two hand-ons, the pair and the down, 8 and
/// 8; the embed 1 and 1, and the head 2 groups (the logits, the picks) of 1 +
/// 2 dispatches: 1 + 3 · 9 + 8 + 2 = 38 groups, 1 + 3 · 10 + 8 + 3 = 42
/// dispatches, 37 barriers.
// PIN(2026-11-04): the delta out is one M-row dispatch at A4 (the out stays
// one group), so a delta layer is 10 dispatches where it was a dispatch a
// row's 11.
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
    deep_logit_abs: None,
    count: None,
    verify: Some(Count {
        groups: 38,
        dispatches: 42,
    }),
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
/// down, join, each one dispatch, and the embed's one group and the head's two
/// (the logits, then the pick): 24 · 7 + 3.
/// PIN(2026-10-29): 171, was 170: the head is two dispatches, a barrier between.
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
    deep_logit_abs: Some(LFM3_DEEP_LOGIT_ABS),
    count: Some(Count {
        groups: 171,
        dispatches: 171,
    }),
    verify: None,
    mutants: Mutants::Lfm3,
    skips: &[(
        "m_mlx_folder",
        "the folder is every clause's model and there is no prepared split to tell it against; \
             `m_logits` holds the folder's own logits to the pin",
    )],
};

/// Llama-3.1-8B-Instruct's checkpoint folder cut to its first four layers (the folder's own
/// bits, `tools/mlx-fixture-cuts`): grouped-query attention of 32 heads over 8 key heads of 128
/// values with no head norm, the rope's `llama3` frequency factors, a plain residual and a dense
/// SwiGLU in every layer, and a head of its own. No `prepare` map reads the family, so the folder
/// is the model. Its vocabulary is not Qwen's and no d1k set exists, so the prompts are
/// generated; the generation starts at the BOS id, which is the model's own start.
///
/// The count is four layers of seven groups, P1 (q, k and v are one type, so one dispatch),
/// the flash, the out projection (adding into the stream), a hand-on, the FFN's gate-up (a dense
/// layer has no router), its down (adding into the stream) and a hand-on, and the embed's one
/// group and the head's two: 4 · 7 + 3 = 31, each one dispatch. The plain residual hands on
/// because every reader of a layer's two halves (the next layer's P1 of one weight type, the
/// gate-up, the head) is a launch of a type other than F32 (`Arena::hands_on`). A cut of another
/// layer count encodes 7 · layers + 3.
pub static LLAMA: Row = Row {
    name: "llama",
    arch: refset::arch::llama::ARCH,
    model: Model::Folder,
    prompt: Prompt::Generated {
        first: 128_000,
        lo: 256,
        hi: 100_000,
    },
    seed: 128_000,
    recurrent: false,
    logit_abs: LLAMA_LOGIT_ABS,
    deep_logit_abs: None,
    count: Some(Count {
        groups: 31,
        dispatches: 31,
    }),
    verify: None,
    mutants: Mutants::Llama,
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
        (
            "p_profile",
            "not on a dense FFN: the clause walks each layer's router picks and the experts' \
             stack slices of the file's own tensors",
        ),
        (
            "m_mlx_folder",
            "the folder is every clause's model and there is no prepared split to tell it \
             against; `m_logits` holds the folder's own logits to the pin",
        ),
    ],
};

/// The clauses a dense Qwen3 folder's row skips, with each reason.
const QWEN3_SKIPS: &[(&str, &str)] = &[
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
    (
        "p_profile",
        "not on a dense FFN: the clause walks each layer's router picks and the experts' \
             stack slices of the file's own tensors",
    ),
    (
        "m_mlx_folder",
        "the folder is every clause's model and there is no prepared split to tell it \
             against; `m_logits` holds the folder's own logits to the pin",
    ),
];

/// Qwen3-8B's checkpoint folder cut to its first four layers (the folder's own bits,
/// `tools/mlx-fixture-cuts`): grouped-query attention of 32 heads over 8 key heads of 128 values
/// with a q and a k head norm and the NEOX rope, a plain residual and a dense SwiGLU in every
/// layer, and a head of its own. No `prepare` map reads the family, so the folder is the model.
/// The prompt ids are Qwen3-30B-A3B's d1k set's: the two families share a vocabulary and a
/// tokenizer; a generation starts at id 0, as that row's.
///
/// The count is four layers of seven groups, P1 (q, k and v are one type, so one dispatch), the
/// flash (it norms and ropes the query and appends the normed, roped key itself, so the head norms
/// add no group), the out projection (adding into the stream), a hand-on, the FFN's gate-up (a
/// dense layer has no router), its down (adding into the stream) and a hand-on, and the embed's one
/// group and the head's two: 4 · 7 + 3 = 31, each one dispatch. The plain residual hands on because
/// every reader of a layer's two halves (the next layer's P1 of one weight type, the gate-up, the
/// head) is a launch of a type other than F32 (`Arena::hands_on`): every projection of the folder
/// is 4-bit with groups of 64. A cut of another layer count encodes 7 · layers + 3.
///
/// The verify's count is four attention layers of 8 groups and 8 dispatches (P1, the flash's
/// append and its flash, the out, two hand-ons, the pair, the down), the embed's 1 and 1 and the
/// head's 2 groups of 1 + 2 dispatches: 1 + 4 · 8 + 2 = 35 groups, 1 + 4 · 8 + 3 = 36 dispatches,
/// 34 barriers.
pub static QWEN3: Row = Row {
    name: "qwen3",
    arch: "qwen3",
    model: Model::Folder,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen3moe::fixture::IK,
        set: refset::arch::qwen3moe::fixture::D1K,
    },
    seed: 0,
    recurrent: false,
    logit_abs: QWEN3_LOGIT_ABS,
    deep_logit_abs: None,
    count: Some(Count {
        groups: 31,
        dispatches: 31,
    }),
    verify: Some(Count {
        groups: 35,
        dispatches: 36,
    }),
    mutants: Mutants::Qwen3,
    skips: QWEN3_SKIPS,
};

/// Qwen3-4B-Instruct-2507's checkpoint folder cut to its first four layers (`tools/mlx-fixture-cuts`),
/// the dense Qwen3 of the `qwen3` row at hidden width 2560 and with its head tied to the embedding:
/// the cut keeps no `lm_head` tensor and the config's `tie_word_embeddings` true, so the session's
/// head reads the embedding's table (`check::head_tensor`). The prompt ids and the generation's
/// first id are the `qwen3` row's.
///
/// The count is the `qwen3` row's: four layers of seven groups and the ends' three, 4 · 7 + 3 = 31,
/// each one dispatch. The tied head adds none: it is the head's logits and pick, over the table.
pub static QWEN3T: Row = Row {
    name: "qwen3t",
    arch: "qwen3t",
    model: Model::Folder,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen3moe::fixture::IK,
        set: refset::arch::qwen3moe::fixture::D1K,
    },
    seed: 0,
    recurrent: false,
    logit_abs: QWEN3T_LOGIT_ABS,
    deep_logit_abs: None,
    count: Some(Count {
        groups: 31,
        dispatches: 31,
    }),
    verify: None,
    mutants: Mutants::Qwen3,
    skips: QWEN3_SKIPS,
};

/// Qwen2.5-Coder-7B-Instruct's checkpoint folder cut to its first four layers
/// (`tools/mlx-fixture-cuts/qwen2-mlx.json`): grouped-query attention of 28 heads over 4 key
/// heads of 128 values (a group of seven) with a bias on the q, k and v projections, the NEOX rope
/// and no head norm, a plain residual and a dense SwiGLU in every layer, and a head of its own,
/// at hidden width 3584. The prompt ids and the generation's first id are the `qwen3` row's.
///
/// The count is the `qwen3` row's with one group a layer for the biases: four layers of eight
/// groups (P1, the bias add, the flash, the out projection, a hand-on after the mixer, the
/// gate-up, the down, a hand-on after the FFN) and the ends' three, 4 · 8 + 3 = 35, each one
/// dispatch (the biases are added to the rows in one launch).
pub static QWEN2: Row = Row {
    name: "qwen2",
    arch: "qwen2",
    model: Model::Folder,
    prompt: Prompt::D1k {
        ik: &refset::arch::qwen3moe::fixture::IK,
        set: refset::arch::qwen3moe::fixture::D1K,
    },
    seed: 0,
    recurrent: false,
    logit_abs: QWEN2_LOGIT_ABS,
    deep_logit_abs: None,
    count: Some(Count {
        groups: 35,
        dispatches: 35,
    }),
    verify: None,
    mutants: Mutants::Qwen2,
    skips: QWEN3_SKIPS,
};

/// Every row, the default first.
pub static ROWS: [&Row; 8] = [
    &QWEN35MOE, &QWEN3MOE, &QWEN38D, &LFM3, &LLAMA, &QWEN3, &QWEN3T, &QWEN2,
];

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
