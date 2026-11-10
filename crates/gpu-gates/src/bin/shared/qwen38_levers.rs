//! The levers `bloomery-serve-qwen38` acts on ([`ACTS_ON`]), the one list its
//! `main` parses (`bloomery_levers::at_main`) and `gate_qwen38_serve` parses
//! too: the gate starts the server with its own environment, so a lever the
//! server would refuse is refused by the gate first. Each binary includes this
//! file; the bin root's unit test holds the list to the levers the seat reads.

/// The levers the Qwen3.8 server acts on.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::QWEN38_EXPERTS,
    bloomery_levers::QWEN3_KV,
    bloomery_levers::KV_FLOOR,
    bloomery_levers::KV_HOST,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::CARD_EXPERTS_CAP,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
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
    bloomery_levers::ELASTIC,
    bloomery_levers::PREFILL,
    bloomery_levers::ROUTE_TRACE,
    bloomery_levers::STEP_STATS,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::AUTOTUNE,
    bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
];

/// What the server takes for the K/V levers it leaves unset: the routed
/// experts spread at 8,192 positions (a context no more has no floor) and the
/// planes' pages spill to the host once the card's margin is reached. The
/// server's one owner of these defaults; `off` stays a value of both levers.
pub const KV_DEFAULT: model::placement::KvDefault = model::placement::KvDefault {
    floor: 8192,
    host: bloomery_levers::KvHost::Spill,
};
