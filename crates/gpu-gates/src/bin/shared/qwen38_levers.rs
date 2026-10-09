//! The levers `bloomery-serve-qwen38` acts on ([`ACTS_ON`]), the one list its
//! `main` parses (`bloomery_levers::at_main`) and `gate_qwen38_serve` parses
//! too: the gate starts the server with its own environment, so a lever the
//! server would refuse is refused by the gate first. Each binary includes this
//! file; the bin root's unit test holds the list to the levers the seat reads.

/// The levers the Qwen3.8 server acts on.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::QWEN38_EXPERTS,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_HEAD_ROWS,
    bloomery_levers::MTP_DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_levers::RESIDENCY,
    bloomery_levers::XSTREAM,
    bloomery_levers::PREFILL,
    bloomery_levers::ROUTE_TRACE,
    bloomery_levers::STEP_STATS,
];
