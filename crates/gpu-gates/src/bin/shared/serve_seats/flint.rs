//! Flint's actual slot-capable body behind the common hybrid HTTP seat.
//! Chat, tools, reasoning and end-of-generation tokens use the canonical model contract.

use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::flint::place;
use model::arch::place_hybrid::PlanInputs;
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

impl Family for Flint {
    type Body = bloomery_gpu_flint::Body;
    type H = ();

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "flint";
    const FAMILY: &'static str = "flint";
    const SERVES: &'static str = "flint";

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::Flint)
    }

    fn read(split: &Split) -> Result<PlanInputs<()>, PlacementError> {
        place::PlanInputs::read(split)
    }

    fn configure(inputs: &mut PlanInputs<()>, group: usize) -> Result<(), GateError> {
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
        inputs: &PlanInputs<()>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| place::tier_batch(inputs))
            .transpose()?)
    }
}

/// Opens the real Flint session and serves resident sequences by select and step.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    super::hybrid::run::<Flint>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    super::hybrid::check::<Flint>(args, read_levers)
}
