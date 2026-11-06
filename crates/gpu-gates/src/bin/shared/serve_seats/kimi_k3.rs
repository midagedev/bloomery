//! Kimi K3's actual body behind the common hybrid HTTP seat: the placed load (the dense trunk and
//! each routed layer's card experts on the stage card, an optional expert tier card, the host and
//! NVMe tiers behind them) serving one sequence, its prompt stepped one id at a time.
//!
//! What this file holds is what K3's architecture forces. The delta layers' state takes no
//! position back, so a request shares a prefix with the slot only where the sequence stands
//! (`app::arch::kimi_k3`). The body has one tier card and runs no residency but `off`, so the
//! family says one tier card ([`TierRule`]) and a step-fed prompt ([`Family::PREFILL_STEPS`]).
//! A K3.1 file's context is held to its indexer's top-k, where the body attends every position
//! the selection would pick among ([`PlanInputs::check_ctx`]): the bound is asked of `--ctx` as
//! given and of the context the plan chose, so an unset `--ctx` that fits past it is refused too.
//!
//! The chat surface is the model's own template, whose tool-call markup no parser reads, so a
//! request that asks for tool calls is the server's refusal, and whose description names no
//! reasoning tags, so the seat splits none and serves the reasoning inline.

use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::kimi_k3::hparams::Hparams;
use model::arch::kimi_k3::place::{self, PlanInputs};
use model::arch::models::ReasoningFormat;
use model::placement::PlacementError;
use model::placement::workstation::TierBatchBytes;
use serve::ServeError;
use serve::reasoning::ThinkTags;

use super::hybrid::Family;

const NAME: &str = "bloomery-serve-kimik3";
const USAGE: &str = "usage: bloomery-serve --model kimik3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel 1] [--slot-ctx S] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The placement, host load, residency, caller and expert tier node settings the seat acts on.
/// There is no prompt group: the ids are stepped.
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
    bloomery_levers::TIER_NODES,
];

/// Architecture-specific facts; the common owner plans and serves the slot.
pub struct KimiK3;

impl Family for KimiK3 {
    type Body = bloomery_gpu::arch::kimi_k3::Body;
    type H = Hparams;

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "kimi-k3";
    const FAMILY: &'static str = "kimik3";
    const SERVES: &'static str = "kimi-k3";

    /// One expert tier card, the stage card's neighbour. No sitting has shown the tier not
    /// slower, so an unset `--place` is `a` and the rule never picks.
    const TIER_RULE: TierRule = TierRule {
        tiers: 1,
        break_even: None,
        basis: "",
    };

    /// The body has no rows pass on the placed load: each prompt id is a decode step.
    const PREFILL_STEPS: bool = true;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::KimiK3)
    }

    /// The file as the plan reads it for a driver that renders no chat
    /// ([`PlanInputs::read_without_chat`]): every part the body does not run is refused by name,
    /// the template's tool-call parser excepted.
    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read_without_chat(split)
    }

    /// [`PlanInputs::check_ctx`]: a K3.1 file's `--ctx` past its indexer's top-k is refused by that
    /// function's own words.
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

    fn tier_batch(place: Place, inputs: &PlanInputs) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| place::tier_batch(&inputs.hp))
            .transpose()?)
    }

    /// The description's tags when it names a think span; it names none today, and the seat then
    /// serves the reasoning inline, as plain text.
    fn think_tags(inputs: &PlanInputs) -> Result<Option<ThinkTags>, GateError> {
        Ok(inputs
            .spec
            .chat
            .reasoning
            .map(|ReasoningFormat::ThinkSpan { open, close }| ThinkTags { open, close }))
    }
}

/// Opens the real K3 session and serves its one sequence by select and step.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    super::hybrid::run::<KimiK3>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    super::hybrid::check::<KimiK3>(args, read_levers)
}
