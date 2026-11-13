//! The head128 program's slot-capable body behind the common hybrid HTTP seat: Flint's
//! files and K-EXAONE's `exaone-moe` ones, the file's router row picking the compiled
//! instance once, at `run`. Chat, tools, reasoning and end-of-generation tokens use the
//! canonical model contract.
//!
//! The body hangs one expert tier card ([`TIER_RULE`]): `--place bp` puts the next-largest card
//! of this process beside the stage, and `BLOOMERY_TIER_NODES` puts that tier on a node, where a
//! `tier_server` built with the `flint` feature serves it ([`bloomery_gpu_flint::tier`]).

use app::mtp::open_draft;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::flint::place;
use model::arch::flint::place::PlanInputs;
use model::placement::PlacementError;
use model::placement::workstation::TierBatchBytes;
use serve::ServeError;

use super::drafted::SlotDrafts;
use super::hybrid::Family;
use super::hybrid_draft::SeatDraft;

const NAME: &str = "bloomery-serve-flint";
const USAGE: &str = "usage: bloomery-serve --model flint [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel N|auto] [--slot-ctx S] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The common placement, host load, residency, prompt-group, expert tier node and caller settings.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::TIER_NODES,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
];

/// The family's input to the common unset rule (`Place::choose`): the one expert tier card the body
/// hangs, a card of this process or a node's (`BLOOMERY_TIER_NODES`), with no break-even — no
/// sitting has shown the tier not slower, so two cards in view run `a` until `--place bp` asks.
pub const TIER_RULE: TierRule = TierRule {
    tiers: bloomery_gpu::host::SERVED_TIERS,
    break_even: None,
    basis: "",
};

/// Architecture-specific facts; the common owner plans and serves every slot.
pub struct Flint;

/// The same facts over K-EXAONE's body: the biased top-eight router instance
/// the exaone-moe description selects.
pub struct Exaone;

macro_rules! flint_family {
    ($ty:ident, $body:ty, $accepts:expr, $family:expr, $serves:expr) => {
        impl Family for $ty {
            type Body = $body;
            type H = place::MoeWidths;

            const NAME: &'static str = self::NAME;
            const USAGE: &'static str = self::USAGE;
            const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
            const ALIAS: &'static str = "flint";
            const FAMILY: &'static str = $family;
            const SERVES: &'static str = $serves;
            const TIER_RULE: TierRule = TIER_RULE;
            const MTP: bool = true;

            /// The file's MTP layer ([`bloomery_gpu_flint::draft`]) as a window of two rows over
            /// the seat's one sequence, its walks eager.
            fn mtp(
                s: &mut app::Session<$body>,
                width: runtime::width::Mode,
            ) -> Result<Box<dyn SeatDraft<$body>>, GateError> {
                let mut log = crate::quiet::Quiet;
                Ok(Box::new(SlotDrafts::<$body, MTP_ROWS>::open(
                    s,
                    1,
                    width,
                    &mut log,
                    |m| open_draft(m, (), StepMode::Eager),
                )?))
            }

            fn draft_record(
                m: &bloomery_gpu::GpuModel<$body>,
                took: std::time::Duration,
            ) -> Result<record::Record, GateError> {
                Ok(record::Record::new(&record::LOAD_DRAFT_HYBRID)
                    .u("layer", m.layers().len())
                    .u(
                        "resident",
                        bloomery_gpu_flint::draft::draft_bytes(m)? as u64,
                    )
                    .u("width", <$body as app::mtp::MtpBody>::WIDTH as u64)
                    .w("head", "full")
                    .f("load_s", took.as_secs_f64()))
            }

            fn accepts(arch: Arch) -> bool {
                $accepts(arch)
            }

            fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
                place::PlanInputs::read(split)
            }

            fn configure(inputs: &mut PlanInputs, group: usize) -> Result<(), GateError> {
                Ok(place::configure_group(inputs, group)?)
            }

            fn register(args: &[String]) {
                record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_HYBRID, |flag| {
                    bloomery_gpu_gates::llama_launch::serve_option(args, flag)
                });
            }

            fn listening() -> record::Record {
                record::Record::new(&record::LISTENING_HYBRID)
            }

            fn load_record() -> record::Record {
                record::Record::new(&record::LOAD_HYBRID)
            }

            fn tier_batch(
                place: Place,
                inputs: &PlanInputs,
            ) -> Result<Option<TierBatchBytes>, GateError> {
                Ok((!place.tier_cards().is_empty())
                    .then(|| place::tier_batch(inputs))
                    .transpose()?)
            }
        }
    };
}

/// A window of the in-file draft: one proposed id and the token before it.
const MTP_ROWS: usize = 2;
const _: () = assert!(MTP_ROWS == <bloomery_gpu_flint::Body as app::mtp::MtpBody>::VERIFY_ROWS);
const _: () =
    assert!(MTP_ROWS == <bloomery_gpu_flint::ExaoneBody as app::mtp::MtpBody>::VERIFY_ROWS);

fn serves_flint(arch: Arch) -> bool {
    matches!(arch, Arch::Flint)
}

fn serves_exaone(arch: Arch) -> bool {
    matches!(arch, Arch::ExaoneMoe)
}

flint_family!(
    Flint,
    bloomery_gpu_flint::Body,
    serves_flint,
    "flint",
    "flint"
);
flint_family!(
    Exaone,
    bloomery_gpu_flint::ExaoneBody,
    serves_exaone,
    "exaone-moe",
    "exaone-moe"
);

/// Whether the model file the seat opens (`ref_model_path`, the path `-m` set at the start) is one
/// whose description selects K-EXAONE's router instance (`place::router_row_of`, the program's own
/// admission). A start with no model file yet (`--hf`, fetched later) takes Flint's, whose run
/// raises the same path error by name or whose load refuses a description of the other instance.
fn exaone_file() -> Result<bool, GateError> {
    let Ok(path) = bloomery_gpu_gates::ref_model_path() else {
        return Ok(false);
    };
    if path.is_dir() {
        return Ok(false);
    }
    let shown = path.display();
    let split = Split::open(&path).map_err(|e| format!("open {shown}: {e}"))?;
    let read = model::arch::spec(&split)?;
    let row = place::router_row_of(&read.spec)?;
    Ok(bloomery_gpu_flint::is_exaone_router(row))
}

/// Opens the real body the file's router picks and serves resident sequences by select and step.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    if exaone_file()? {
        super::hybrid::run::<Exaone>(args, side)
    } else {
        super::hybrid::run::<Flint>(args, side)
    }
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    super::hybrid::check::<Flint>(args, read_levers)
}
