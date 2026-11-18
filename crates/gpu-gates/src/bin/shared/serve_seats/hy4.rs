//! `bloomery-serve --model hy4` — the llama-server-compatible HTTP API on the Hy4-flash engine: a
//! hy-v4-flash file, every layer's dense part and the stores on one card, each routed layer's
//! card share of its experts beside them and the rest on the host tier (the common hybrid
//! planner through `PlanInputs::plan_at`), opened by the body's placed constructor
//! (`bloomery_gpu_hy4::Body::open_placed`).
//!
//!     bloomery-serve --model hy4 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--parallel N] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! What this file holds is what Hy4-flash's architecture forces. A window layer's ring and the
//! engram history hold the position they stand at alone, so a request shares a prefix with its
//! slot only where the sequence stands (`app::arch::hy4`). The body keeps its load's slot map for
//! its life, so the family says no residency but `off` ([`Family::SLOT_MAP_FIXED`]), and it
//! hangs no expert tier card ([`TierRule::UNTIERED`], the default). Each resident sequence owns
//! its stores and its engram history, so `--parallel N` serves N slots by a select and a step a
//! slot. The prompt is stepped one id at a time: the body has no rows pass.
//!
//! The chat surface is the model's own template: its tool-call markup is the key/value one with
//! the template's suffix in every tag (`serve::glmxml::KvSpelling::Hy4`, which the server picks
//! from the template), and its description names the think span (`hy4::spec::REASONING`),
//! so the server splits the span into `reasoning_content`.
//!
//! The seat itself is the common owner's ([`super::hybrid`]); its records are the common hybrid
//! seat's (`record::BLOOMERY_SERVE_HYBRID`).

use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::place::PlanInputs;
use model::placement::PlacementError;
use serve::ServeError;

use super::hybrid::Family;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-hy4";
const USAGE: &str = "usage: bloomery-serve --model hy4 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel N] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and margin, the residency machine's word
/// (only `off` loads), the host set's load settings, the engine thread's pin, the host legs' lane
/// rule, the start screen and the schema jump. There is no prompt group: the prompt is stepped.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::CARD_MARGIN,
    bloomery_levers::RESIDENCY,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// Architecture-specific facts; the common owner plans and serves the slots.
pub struct Hy4;

impl Family for Hy4 {
    type Body = bloomery_gpu_hy4::Body;
    type H = Hparams;

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "hy4-flash";
    const FAMILY: &'static str = "hy4";
    const SERVES: &'static str = "hy-v4-flash";
    const SLOT_MAP_FIXED: bool = true;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::Hy4)
    }

    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read(split)
    }

    /// A context the file trains past, or none, is refused by name ([`PlanInputs::check_ctx`]).
    fn check_ctx(inputs: &PlanInputs, ctx: usize) -> Result<(), GateError> {
        Ok(inputs.check_ctx(u64::try_from(ctx)?)?)
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
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    super::hybrid::run::<Hy4>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    super::hybrid::check::<Hy4>(args, read_levers)
}
