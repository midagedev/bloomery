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

/// What the server takes for the K/V levers it leaves unset: the common server default
/// ([`model::placement::KvDefault::SERVER`]).
pub const KV_DEFAULT: model::placement::KvDefault = model::placement::KvDefault::SERVER;

/// The K/V levers `levers` leaves unset resolved for a plan on the stage card `stage`
/// ([`KV_DEFAULT`] where the levers say nothing), and the page a host-capable room takes read
/// once from that card into `inputs` (`PlanInputs::kv_host_page`) wherever the backing is not
/// `off`: every plan under the resolved levers counts the planes' pages at it, and a plan under
/// `spill` or `all` with none is refused by name. Where the device cannot stage a host page the
/// unset default is `off` by name on stderr, and a set `spill` or `all` is the refusal itself.
/// The seat and the gate that restates its plans call this one function, so both plan under the
/// same page and the same default.
///
/// # Errors
/// A set `BLOOMERY_KV_HOST` other than `off` on a card that cannot back a page on the host.
pub fn kv_default_at(
    inputs: &mut model::arch::qwen35moe::place::PlanInputs,
    stage: model::placement::workstation::CardSpec,
    levers: &bloomery_levers::Levers,
) -> Result<model::placement::KvDefault, bloomery_gpu_gates::GateError> {
    use bloomery_levers::KvHost;

    let mut default = KV_DEFAULT;
    let set = levers.kv_host_set().and_then(KvHost::parse);
    if set.unwrap_or(default.host) != KvHost::Off {
        match bloomery_gpu_gates::generate::stage_host_page(stage) {
            Ok(page) => inputs.kv_host_page = Some(page),
            Err(e) if set.is_none() => {
                eprintln!(
                    "bloomery-serve-qwen38: BLOOMERY_KV_HOST is unset and {} cannot back a K/V \
                     page on the host ({e}); the planes stay on the card (kv_host=off)",
                    stage.name
                );
                default.host = KvHost::Off;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(default)
}
