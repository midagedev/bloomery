//! The levers `bloomery-serve-ds41` acts on, for its own `main` and for
//! `gate_ds41_serve`'s: the gate starts the server with its own environment,
//! so a lever the server would refuse is refused by the gate first.

use bloomery_levers::{
    CARD_BUDGET, CARD_DONTNEED, CED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, HOT_LIST, PIN_MAIN,
    PREFILL, PREFILL_GROUP, R8,
};

/// Besides the pool's two: how a prompt is fed, and the batched feed's CED
/// triangle and group; the step rows' helper; the placement's hot list and
/// card budget; the main thread's pin; the host tier's load settings.
pub const ACTS_ON: &[&str] = &[
    CED,
    PREFILL,
    PREFILL_GROUP,
    ENGRAM_HELPER,
    HOT_LIST,
    CARD_BUDGET,
    PIN_MAIN,
    HOST_POPULATE,
    HOST_LOCK,
    CARD_DONTNEED,
    R8,
];
