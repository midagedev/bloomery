//! Kimi K3's actual body behind the common hybrid HTTP seat: the placed load (the dense trunk and
//! each routed layer's card experts on the stage card, an optional expert tier card, the host and
//! NVMe tiers behind them) serving one sequence, its prompt run as placed passes.
//!
//! What this file holds is what K3's architecture forces. The delta layers' state takes no
//! position back, so a request shares a prefix with the slot only where the sequence stands
//! (`app::arch::kimi_k3`). The body has one tier card and keeps its load's slot map for its life,
//! so the family says one tier card ([`TierRule`]) and no residency but `off`
//! ([`Family::SLOT_MAP_FIXED`]). A K3.1 file runs every context ([`PlanInputs::check_ctx`]): past
//! its indexer's top-k the latent layers attend the rows the indexer picks.
//!
//! The body has a placed prompt pass ([`Family::pass`]): the widest pass shape the plan holds at
//! the seat's context ([`kimi_k3_pass::pick`], the rule `generate_kimi_k3 --walk pass` picks by),
//! planned with the shape's bytes as the prompt reserve, the expert tier's batch at its rows and
//! the NVMe arena the dial gives, at least its floor, and loaded by [`GpuModel::open_placed_pass`]; each request's prompt
//! then runs through `Batch::prompt`, which takes the pass. A plan that holds no shape loads the
//! stepped body, and `--prefill steps` asks it; the `listening` record names the walk.
//!
//! The chat surface is the model's own template, whose tool-call markup no parser reads, so a
//! request that asks for tool calls is the server's refusal, and whose description names the
//! reply's sections (`kimi_k3::spec::REASONING`), so the server splits the think span into
//! `reasoning_content` and serves the response section's text as `content`.

use bloomery_gpu::GpuModel;
use bloomery_gpu::arch::kimi_k3::Body;
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::kimi_k3_pass::{self, Picked};
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::kimi_k3::hparams::Hparams;
use model::arch::kimi_k3::place::{self, PlanInputs, tier_batch_at};
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, PlacementError};
use serve::ServeError;
use serve::reasoning::ThinkTags;

use super::hybrid::{Family, Passed};
use super::tiered::{BuildMachine, PassAttempt, PassPlan, think_tags_of};

const NAME: &str = "bloomery-serve-kimik3";
const USAGE: &str = "usage: bloomery-serve --model kimik3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel 1] [--slot-ctx S] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--prefill pass|steps] \
                     [--queue-depth Q] [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The placement, host load, residency, caller, expert tier node and K/V (floor, host backing)
/// settings the seat acts on.
/// There is no prompt group: a prompt runs as passes of the plan's shape, or stepped.
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
    bloomery_levers::KV_FLOOR,
    bloomery_levers::KV_HOST,
];

/// Architecture-specific facts; the common owner plans and serves the slot.
pub struct KimiK3;

impl Family for KimiK3 {
    type Body = Body;
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

    /// The body keeps its load's slot map for its life ([`Residency::Off`] in its placed open).
    ///
    /// [`Residency::Off`]: bloomery_gpu::host::swap::Residency::Off
    const SLOT_MAP_FIXED: bool = true;

    const PASS: bool = true;

    /// The widest pass shape the seat's plan holds at `ctx` ([`kimi_k3_pass::pick`]), each
    /// shape's machine with the expert tier's batch at its rows where `place` hangs a tier card,
    /// and its open ([`GpuModel::open_placed_pass`] on stage card 0); or why no shape fits.
    fn pass(
        inputs: &PlanInputs,
        place: Place,
        ctx: usize,
        build: &BuildMachine<'_>,
        attempt: &PassAttempt<'_, Hparams>,
    ) -> Result<Passed<Self>, GateError> {
        let tiered = !place.tier_cards().is_empty();
        let at_rows = |rows: u64| -> Result<Machine, GateError> {
            build(
                tiered
                    .then(|| tier_batch_at(&inputs.hp, rows))
                    .transpose()?,
            )
        };
        let Picked { shape, wider } = match kimi_k3_pass::pick(inputs, ctx, &at_rows, attempt)? {
            Ok(picked) => picked,
            Err(no) => return Ok(Err(no.to_string())),
        };
        let plan = PassPlan {
            inputs: kimi_k3_pass::inputs_for(inputs, ctx, shape)?,
            machine: at_rows(shape.rows as u64)?,
            rows: (shape.rows, shape.mla_rows),
            wider: wider.map(|(w, short)| ((w.rows, w.mla_rows), short)),
        };
        Ok(Ok((
            plan,
            Box::new(move |file, plan, inputs, host| {
                GpuModel::<Body>::open_placed_pass(file, plan, inputs, 0, host, shape)
            }),
        )))
    }

    /// The rows of the pass the placed load holds ([`Body::pass_shape`]), which `Batch::prompt`
    /// runs a prompt by.
    fn pass_rows(m: &GpuModel<Body>) -> Result<Option<usize>, GateError> {
        Ok(m.body(NAME)?.pass_shape().map(|s| s.rows))
    }

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::KimiK3)
    }

    /// The file as the plan reads it for a driver that renders no chat
    /// ([`PlanInputs::read_without_chat`]): every part the body does not run is refused by name,
    /// the template's tool-call parser excepted.
    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read_without_chat(split)
    }

    /// [`PlanInputs::check_ctx`], which takes every context, a K3.1 file's past its indexer's
    /// top-k included.
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

    /// The tags of the reply's sections the description names (the think span and the response
    /// section): the server splits the reasoning out and takes the section markup out of the
    /// content.
    fn think_tags(inputs: &PlanInputs) -> Result<Option<ThinkTags>, GateError> {
        Ok(inputs.spec.chat.reasoning.map(think_tags_of))
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
