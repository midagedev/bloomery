//! ERNIE 5.5 Flash's placed body behind the common hybrid HTTP seat: the dense part of every layer
//! and the stores on the stage card, each routed layer's card experts beside them, the host and
//! NVMe tiers behind them, serving one sequence.
//!
//! What this file holds is what ERNIE's architecture forces. The model ships as a checkpoint
//! folder (FP8 dense weights, MXFP4 experts), so the seat opens a folder once on the main thread
//! ([`Family::read_model`], `serve_seats::folder::open`) and the common owner loads it from its
//! located header and part source. An HCA layer's window ring takes no position back, so a request
//! shares a prefix with the slot only where the sequence stands (`app::arch::ernie55`). The body
//! loads its prompt pass (`app::arch::ernie55`'s `Batch::prompt`): a prompt runs as the pass's
//! units, each hosted layer's host experts read once for a unit's rows through the host tier's
//! batch port (`bloomery_gpu_ernie55::pass`), the call one residency pass; the `load` record names
//! the pass's rows. The stores are whole buffers sized at the load, no rooms, so the K/V levers are
//! not acted on. The body keeps no expert stream (a prompt unit's host picks are served through the
//! host tier's batch port and no expert moves to the card for it), so `BLOOMERY_XSTREAM` is `off`
//! ([`Family::xstream`]), the plan keeps no stream ring ([`Family::ring`] and [`Family::with_ring`]
//! are the defaults) and the pass reserve is the plan's own
//! (`model::arch::ernie55::place::prompt_reserve_bytes`). The body serves no expert tier card
//! ([`TierRule::UNTIERED`]).
//!
//! A folder carries no `tokenizer.chat_template` (the vendor's renderer is Python), so the seat
//! serves the server's port of it ([`Family::TEMPLATE`], `serve::ernie::TEMPLATE`) unless
//! `--chat-template-file` names another, and the server splits the reply's think span
//! (`serve::ernie::THINK`) into `reasoning_content`.

use std::path::Path;

use bloomery_gpu::GpuModel;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::xstream::{XMode, XStreamBody};
use bloomery_gpu_ernie55::body::Body;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::{GateError, record, side_seat};
use gguf::Split;
use model::arch::Arch;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::place::PlanInputs;
use model::placement::PlacementError;
use serve::ServeError;
use serve::reasoning::ThinkTags;

use super::hybrid::Family;
use super::tiered::{self, Read};
use super::xstream_rule::{Stage, xstream_rule};

const NAME: &str = "bloomery-serve-ernie55";
const USAGE: &str = "usage: bloomery-serve --model ernie55 [-m PATH | --hf <repo>] [--host H] \
                     [--port P] [--ctx C] [--parallel 1] [--slot-ctx S] [--alias NAME] \
                     [--chat-template-file PATH] [--place W] [--plan] [--queue-depth Q] \
                     [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]";

/// The placement, host load, residency, caller and start settings the seat acts on. There is no
/// prompt group (a prompt runs as the pass's units or stepped) and no K/V lever (the stores
/// declare no rooms); the expert stream is acted on as `off` alone.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::CARD_MARGIN,
    bloomery_levers::RESIDENCY,
    bloomery_levers::XSTREAM,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
];

/// Architecture-specific facts; the common owner plans and serves the slot.
pub struct Ernie55;

impl Family for Ernie55 {
    type Body = Body;
    type H = Hparams;

    const NAME: &'static str = self::NAME;
    const USAGE: &'static str = self::USAGE;
    const ACTS_ON: &'static [&'static str] = self::ACTS_ON;
    const ALIAS: &'static str = "ernie-5.5-flash-preview";
    const FAMILY: &'static str = "ernie55";
    const SERVES: &'static str = "ernie5_5";

    /// The server's port of the vendor's renderer: a folder carries no template.
    const TEMPLATE: Option<&'static str> = Some(serve::ernie::TEMPLATE);

    /// A checkpoint folder of an `ernie5_5` model ([`super::folder::open`]): its located header,
    /// tokenizer and part source; a path that is not a directory is a GGUF split.
    fn read_model(path: &Path) -> Result<Read, GateError> {
        if !path.is_dir() {
            return tiered::read_file(path);
        }
        let f = super::folder::open(path, gguf::arch::ernie55::ARCH)?;
        Ok(Read {
            split: f.header,
            tokenizer: Some(f.tokenizer),
            source: Some(f.source),
        })
    }

    /// The stream word the load runs: `off` unset (the body streams no expert, and says so on the
    /// line), a set `BLOOMERY_XSTREAM` through the common rule, whose `set_xstream` refuses
    /// `admit` and `split` by name.
    fn xstream(
        m: &mut GpuModel<Body>,
        set: Option<&str>,
        place: Place,
        residency: Residency,
    ) -> Result<Option<String>, GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        if set.is_none() {
            body.set_xstream(gpu, XMode::Off)?;
            let word = body.xstream_word();
            return Ok(Some(format!(
                "xstream={word} (unset: the body streams no host expert; a prompt unit's host picks \
                 are served through the host tier's batch port)"
            )));
        }
        Ok(Some(xstream_rule(
            gpu,
            body,
            set,
            Stage::of(place),
            residency,
        )?))
    }

    /// The rows of the prompt pass the load holds (`Body::pass_rows`).
    fn pass_rows(m: &GpuModel<Body>) -> Result<Option<usize>, GateError> {
        Ok(m.body(NAME)?.pass_rows())
    }

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::Ernie55)
    }

    /// The model as the plan reads it ([`PlanInputs::read`]): every part the program does not run
    /// is refused by name.
    fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::read(split)
    }

    /// [`PlanInputs::check_ctx`]: a context of no position or past the trained one is refused by
    /// name.
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

    /// The renderer's think span (`serve::ernie::THINK`): the description reads no template from
    /// a folder, so the seat names the span the server's port writes.
    fn think_tags(_inputs: &PlanInputs) -> Result<Option<ThinkTags>, GateError> {
        Ok(Some(serve::ernie::THINK))
    }
}

/// Opens the ERNIE 5.5 session and serves its one sequence by select and step.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    super::hybrid::run::<Ernie55>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`super::hybrid::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    super::hybrid::check::<Ernie55>(args, read_levers)
}
