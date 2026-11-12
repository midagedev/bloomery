//! The head128 program's slot-capable body behind the common hybrid HTTP seat: Flint's
//! files and K-EXAONE's `exaone-moe` ones, the file's router row picking the compiled
//! instance once, at `run`. Chat, tools, reasoning and end-of-generation tokens use the
//! canonical model contract.

use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::flint::place;
use model::arch::flint::place::PlanInputs;
use model::placement::PlacementError;
use model::placement::workstation::TierBatchBytes;
use serve::ServeError;

use super::hybrid::Family;

const NAME: &str = "bloomery-serve-flint";
const USAGE: &str = "usage: bloomery-serve --model flint [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel N|auto] [--slot-ctx S] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The common placement, host load, residency, prompt-group and caller settings.
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
];

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

/// Whether the model `args` name is a local file whose description selects
/// K-EXAONE's router instance (`place::router_row_of`, the program's own
/// admission): a start that names no local file (`--hf`) takes Flint's, whose
/// load refuses a description of the other instance by name.
fn exaone_file(args: &[String]) -> Result<bool, GateError> {
    for (flag, value) in bloomery_gpu_gates::llama_launch::serve_args(args) {
        if flag != "-m" && flag != "--model-file" {
            continue;
        }
        let Some(path) = value else {
            return Err(format!("{flag} needs a path").into());
        };
        let split = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let read = model::arch::spec(&split)?;
        let row = place::router_row_of(&read.spec)?;
        return Ok(bloomery_gpu_flint::is_exaone_router(row));
    }
    Ok(false)
}

/// Opens the real body the file's router picks and serves resident sequences by select and step.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    if exaone_file(args)? {
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
