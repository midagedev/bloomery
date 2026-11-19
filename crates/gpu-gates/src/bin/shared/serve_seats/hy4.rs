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
//!                    [--mmproj FILE [--image-min-tokens N] [--image-max-tokens N]]
//!
//! What this file holds is what Hy4-flash's architecture forces. A window layer's ring and the
//! engram history hold the position they stand at alone, so a request shares a prefix with its
//! slot only where the sequence stands (`app::arch::hy4`). The body runs the common residency
//! machine under the seat's word (unset, the serving word on a serving placement, which the plan
//! keeps, moves or turns `off`), and it hangs no expert tier card ([`TierRule::UNTIERED`], the
//! default). Each resident sequence owns
//! its stores and its engram history, so `--parallel N` serves N slots by a select and a step a
//! slot. The body has a placed prompt pass ([`Family::pass`]): the widest pass shape the plan
//! holds at the seat's context ([`hy4_pass::pick`]), planned with the shape's bytes as the prompt
//! reserve and loaded by `Body::open_placed_with`; each request's prompt then runs through
//! `Batch::prompt`, which takes the pass. A plan that holds no shape loads the stepped body (one id
//! a decode step), and `--prefill steps` asks it; the `listening` record names the walk.
//!
//! The prompt pass streams host experts under the common stream rule (`shared/xstream_rule.rs`,
//! `BLOOMERY_XSTREAM`): the plan keeps the stream's ring out of the expert budget where the rule
//! would start it ([`Family::ring`]), the pass's inputs keep it beside the pass's bytes
//! ([`hy4_pass::inputs_for`]) and the rule resolves the stream once the load is open
//! ([`Family::xstream`]); a load that runs its prompt stepped streams nothing.
//!
//! The sparse layers' cache rows are rooms, so the seat takes the common K/V default
//! ([`PlanInputs::plan_levers`]): an unset `BLOOMERY_KV_FLOOR` plans the experts' spread at 8,192
//! positions and an unset `BLOOMERY_KV_HOST` spills the rows' pages to the host once the card's
//! margin is reached. The pooled keys are no room and stay counted at the context.
//!
//! The chat surface is the model's own template: its tool-call markup is the key/value one with
//! the template's suffix in every tag (`serve::glmxml::KvSpelling::Hy4`, which the server picks
//! from the template), and its description names the think span (`hy4::spec::REASONING`),
//! so the server splits the span into `reasoning_content`.
//!
//! `--mmproj FILE` gives the seat image input (behind the `vision` feature): the `hy_v4_flash`
//! tower (`vision::arch::hy4v`), held against the text model's width and vocabulary and seated
//! where the placement leaves room by the common owner (`vision_seat::open_seat`). An image's
//! rows, a newline row after each grid row the tower's own, stand in place of the embeddings of
//! its pad positions between the start and end marks, at the sequence's ordinary positions; the
//! prompt runs through the placed pass ([`super::hybrid::media_prompt`], `Body::prefill_media`),
//! and a load without the pass refuses an image by name.
//!
//! The seat itself is the common owner's ([`super::hybrid`]); its records are the common hybrid
//! seat's (`record::BLOOMERY_SERVE_HYBRID`).

use bloomery_gpu::GpuModel;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::xstream::{XMode, XStreamBody};
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::hy4_pass::{self, Picked};
use bloomery_gpu_gates::{GateError, record, side_seat};
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::place::PlanInputs;
use model::placement::PlacementError;
use serve::ServeError;

use super::hybrid::{Family, Passed};
use super::tiered::{BuildMachine, PassAttempt, PassPlan};
use super::xstream_rule::{Stage, plans_ring, xstream_rule};

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-hy4";
const USAGE: &str = "usage: bloomery-serve --model hy4 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--parallel N] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--prefill pass|steps] \
                     [--queue-depth Q] [--slot-save-path DIR] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and margin, the residency machine's word,
/// the host set's load settings, the engine thread's pin, the host legs' lane
/// rule, the start screen, the schema jump and the K/V's floor and host backing. There is no prompt
/// group: a prompt runs as passes of the plan's shape, or stepped. `BLOOMERY_XSTREAM` is the
/// expert stream's word (the common rule).
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
    bloomery_levers::KV_FLOOR,
    bloomery_levers::KV_HOST,
    bloomery_levers::XSTREAM,
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
    const PASS: bool = true;

    /// The widest pass shape the seat's plan holds at `ctx` ([`hy4_pass::pick`]) on the seat's
    /// machine (no expert tier card hangs, so one machine serves every shape), and its open
    /// ([`Body::open_placed_with`] on stage card 0 with the shape, under the seat's residency); or
    /// why no shape fits.
    ///
    /// [`Body::open_placed_with`]: bloomery_gpu_hy4::Body::open_placed_with
    fn pass(
        inputs: &PlanInputs,
        _place: Place,
        ctx: usize,
        build: &BuildMachine<'_>,
        attempt: &PassAttempt<'_, Hparams>,
    ) -> Result<Passed<Self>, GateError> {
        let machine = build(None)?;
        let Picked { shape, wider } = match hy4_pass::pick(inputs, ctx, &machine, attempt)? {
            Ok(picked) => picked,
            Err(no) => return Ok(Err(no.to_string())),
        };
        let plan = PassPlan {
            inputs: hy4_pass::inputs_for(inputs, ctx, shape)?,
            machine,
            rows: (shape.walk_rows(), shape.q_rows),
            wider: wider.map(|(w, short)| ((w.walk_rows(), w.q_rows), short)),
        };
        Ok(Ok((
            plan,
            Box::new(move |file, plan, inputs, host, residency| {
                bloomery_gpu_hy4::Body::open_placed_with(
                    file,
                    (plan, inputs),
                    0,
                    host,
                    (Some(shape), residency),
                )
            }),
        )))
    }

    /// The ring is kept out of the expert budget where the stream starts at `split` ([`plans_ring`]
    /// over [`unset_word`]'s word).
    fn ring(levers: &Levers, place: Place) -> bool {
        plans_ring(
            Some(unset_word(
                levers.xstream(),
                levers.residency() == Some("off"),
            )),
            Stage::of(place),
            levers.residency() == Some("off"),
        )
    }

    /// The inputs of a plan that keeps the stream's ring beside the pass's bytes
    /// ([`hy4_pass::with_ring`]).
    fn with_ring(inputs: PlanInputs) -> Result<PlanInputs, GateError> {
        hy4_pass::with_ring(inputs)
    }

    /// The stream by the common rule once the load is open. A load that runs its prompt stepped
    /// has no pass to stream: unset, it streams nothing and says so; a word `BLOOMERY_XSTREAM`
    /// sets goes to the rule, which the body refuses by name where it needs the pass.
    fn xstream(
        m: &mut GpuModel<bloomery_gpu_hy4::Body>,
        set: Option<&str>,
        place: Place,
        residency: Residency,
    ) -> Result<Option<String>, GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        if set.is_none() && body.pass_rows().is_none() {
            body.set_xstream(gpu, XMode::Off)?;
            let word = body.xstream_word();
            return Ok(Some(format!(
                "xstream={word} (unset: the load runs its prompt stepped)"
            )));
        }
        let line = xstream_rule(
            gpu,
            body,
            Some(unset_word(set, residency == Residency::Off)),
            Stage::of(place),
            residency,
        )?;
        Ok(Some(if set.is_none() {
            line.replace("(set)", "(unset: admit, Hy4's pass streams by admission)")
        } else {
            line
        }))
    }

    /// The rows of the pass the placed load holds ([`bloomery_gpu_hy4::Body::pass_rows`]), which
    /// `Batch::prompt` runs a prompt by.
    fn pass_rows(m: &GpuModel<bloomery_gpu_hy4::Body>) -> Result<Option<usize>, GateError> {
        Ok(m.body(NAME)?.pass_rows())
    }

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

    #[cfg(feature = "vision")]
    const PROJECTORS: &'static [&'static str] = &[vision::arch::hy4v::PROJECTOR_TYPE];

    /// The image rows at their pad positions through the placed pass
    /// ([`super::hybrid::media_prompt`]).
    #[cfg(feature = "vision")]
    fn prompt_media(
        s: &mut app::Session<bloomery_gpu_hy4::Body>,
        v: &mut bloomery_gpu_gates::vision_seat::Resident,
        ids: &[u32],
        feeds: &[serve::media::MediaFeed],
    ) -> Result<u32, GateError> {
        super::hybrid::media_prompt(s, v, ids, feeds)
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

/// The stream word a Hy4 load runs: `BLOOMERY_XSTREAM` as set; unset, `admit` with a residency
/// and `off` without one (the stream needs the residency machine). Hy4 decodes faster at `admit`
/// than at `split` and runs its short prompts as fast (docs/cards/hy4-pp4096-diag.card), so its
/// seat keeps this one deviation from the common rule's unset `split`.
fn unset_word(set: Option<&str>, residency_off: bool) -> &str {
    set.unwrap_or(if residency_off { "off" } else { "admit" })
}
