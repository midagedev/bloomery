//! The levers `bloomery-serve-ds41` acts on, for its own `main` and for
//! `gate_ds41_serve`'s: the gate starts the server with its own environment,
//! so a lever the server would refuse is refused by the gate first.

use bloomery_levers::{
    CARD_BUDGET, CARD_DONTNEED, CED, DRAFT, ENGRAM_HELPER, EXPERT_RANK, HOST_LANES, HOST_LOCK,
    HOST_POPULATE, HOSTSTREAM, KV_FLOOR, KV_HOST, KvHost, MTP_WIDTH, PIN_MAIN, PREFILL,
    PREFILL_CUT, PREFILL_GROUP, R8, RESIDENCY, ROUTE_TRACE, SCHEMAJUMP, SCREEN, STEP_STATS,
};

/// Besides the pool's two: how a prompt is fed, and the batched feed's CED
/// triangle and group; the step rows' helper; the placement's card budget and
/// router rank; the main thread's pin and the lanes of the host tier's legs on its CCD
/// map; the host tier's load settings; the draft; the route
/// trace; adaptive expert residency and its prompt streaming; the step
/// statistics, whose rounds of several slots the seat counts, and which on
/// this engine also keeps the step rows' engram fill statistics; the width
/// a drafted window verifies; the start screen; how a structured-output
/// request's fixed runs are fed; the DSpark draft's card, read in place
/// where the draft opens (its file, `BLOOMERY_DSPARK_MODEL`, is no lever).
pub const ACTS_ON: &[&str] = &[
    CED,
    PREFILL,
    PREFILL_GROUP,
    PREFILL_CUT,
    ENGRAM_HELPER,
    CARD_BUDGET,
    EXPERT_RANK,
    KV_FLOOR,
    KV_HOST,
    PIN_MAIN,
    HOST_LANES,
    HOST_POPULATE,
    HOST_LOCK,
    CARD_DONTNEED,
    R8,
    DRAFT,
    MTP_WIDTH,
    ROUTE_TRACE,
    RESIDENCY,
    HOSTSTREAM,
    STEP_STATS,
    SCREEN,
    SCHEMAJUMP,
    bloomery_gpu_gates::llama_place::DRAFT_CARD,
];

/// What the server takes for the K/V levers it leaves unset: the routed experts spread at 8,192
/// positions (a context no more has no floor) and the selected rows' pages spill to the host
/// once the card's margin is reached. The server's one owner of these defaults; `off` stays a
/// value of both levers.
#[allow(
    dead_code,
    reason = "the serve seat reads the default; the other bins that include this file only read ACTS_ON"
)]
pub const KV_DEFAULT: model::placement::KvDefault = model::placement::KvDefault {
    floor: 8192,
    host: KvHost::Spill,
};
