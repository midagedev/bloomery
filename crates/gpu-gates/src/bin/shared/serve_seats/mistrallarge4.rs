//! `bloomery-serve --model mistrallarge4` — the llama-server-compatible HTTP API on the Mistral
//! Large 4 engine: a mistral-large4 file, its layers on one card, the routed experts the plan puts
//! on the card — and on the expert tier card beside it when two cards are in view — and the rest
//! on the host tier (`PlanInputs::plan_with_slots`), opened in graph mode with the step captured.
//!
//!     bloomery-serve --model mistrallarge4 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--parallel N] [--slot-ctx S] [--cache-ram MIB] [--queue-depth Q]
//!                    [--slot-save-path DIR] [--api-key KEY] [--api-key-file FNAME]
//!                    [--mmproj FILE] [--image-min-tokens N] [--image-max-tokens N]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module doc); `--ctx-size` and
//! `-c` are `--ctx` under llama-server's spellings. The model's first shard gives the vocabulary,
//! `tokenizer.chat_template` the chat template (`--chat-template-file` replaces it) and
//! `general.name` the alias (`--alias` replaces it; `mistral-large4` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty, `temperature <= 0` the
//! engine's argmax. Any other flag is refused as an unknown argument.
//!
//! Five facts of the architecture set this seat's shape:
//!
//! - Every layer's store is a plane of positions: a latent row a position on the layers that own a
//!   cache, an index key a position on the layers whose indexer is their own (the list and the
//!   indexer's scores are the step's scratch). So a cut keeps every held position
//!   (`app::Keep for Body`) and the plan counts planes (`KvLayout::of`).
//! - `--parallel N` serves N resident sequences ([`bloomery_gpu::model::Slots`]): each is those
//!   planes, exchanged by a select, and the plan counts N sequences' stores. The step walks one
//!   token, so a round of several slots is a select and a step each (`pass=turns`).
//! - A saved state is the planes' rows alone (`bloomery_gpu::checkpoint::planes`): no recurrent
//!   store and no checkpoint, so a state of `n` positions restores at any position up to `n`.
//!   `--cache-ram MIB` is the prompt cache's budget (`bind::CacheRam`: unset, the default; `0`
//!   turns it off), a `cache` line on stderr printing it with each term; the copy runs on a
//!   stream of its own into pinned staging the budget pins whole at the open. The selected slot's
//!   state also goes to and from a file (`--slot-save-path DIR`, `POST /slots/{id}?action=save`
//!   and `restore`: the state in the byte form of `bloomery_gpu::checkpoint::bytes`), refused by
//!   name for a state of another file, card or context.
//! - The prompt is fed in batches of rows (`app::arch::mistrallarge4::PROMPT_FEED`, the `load`
//!   record's `prompt` word `batch`): the common walk over units of the plan's prompt rows, a call
//!   one residency pass (`app::Prompt`). `BLOOMERY_PREFILL_GROUP` is the units a prompt group runs
//!   layer by layer: the plan prices the group's units, records and lists into its prompt reserve
//!   (`place::configure_group`, at the run and again at the open) and the open makes them once the
//!   step is captured (`set_prefill_group`), refused by name past that reserve; the `load` record
//!   prints the group the body holds.
//!   `BLOOMERY_RESIDENCY` is the common machine's lever over the stage card's routed stacks
//!   (`Body::open_placed_with`): set, its word as given (refused by name where the plan's card
//!   slots leave it no room, or when it asks the churn pool unhosted, which the body hosts);
//!   unset, the serving word on every placement but `gate`, which the plan then keeps, moves or
//!   turns `off` with why (`bloomery_levers::residency_unset`, `residency::at_plan`). An expert
//!   tier card's experts stay where the plan put them. `BLOOMERY_XSTREAM` resolves by the rule the
//!   Qwen3.8, MiniMax and Muse Spark seats run (`shared/xstream_rule.rs`): set as given; unset
//!   `split` under `--place a` with a residency machine (`admit` when the card has no room for the
//!   stream's ring), `admit` under a placement with an expert tier card, `off` everywhere else. A
//!   load that streams plans with the stream's room (`PlanInputs::with_stream`, asked before the
//!   plan by `plans_ring`); the open resolves the lever last, after the slots, and prints its
//!   `xstream=` line (the word, then why) before the `load` record. Set `split` under `--place bp`
//!   joins the ring with the expert tier's slots (`PromptStream::unit_join`).
//! - The chat surface is the file's template: reasoning is the `[THINK]` span (the template
//!   spells both tags; `Tiered::think`) and a tool call is `[TOOL_CALLS]NAME[ARGS]{json}`, which
//!   the server parses from the template's markup (`serve::mistral`).
//!
//! `BLOOMERY_DRAFT` serves a draft behind the width chooser `BLOOMERY_MTP_WIDTH` names
//! ([`ServedDraft`], the serve seats' one owner of it; the tiered run parses the two levers,
//! [`Tiered::DRAFTS`]), each verified by the body's pass of two rows (`app::Verify`): `lookup` is
//! the n-gram draft, one context a resident slot (the body keeps no token history, so a context
//! that falls behind a cut or a restored state starts over empty), and `dspark` the DSpark draft
//! of `$BLOOMERY_DSPARK_MODEL` on the card `$BLOOMERY_DSPARK_CARD` names (the common placement
//! rule otherwise), one slot. The target's feature tap for the draft file's `target_layers` is
//! attached before the step is captured (`bloomery_gpu_mistrallarge4::features`), so the step
//! and the verify pass lay the rows the draft reads. Under `dspark` the draft's card bytes (its
//! weights, the target's `output` it copies, its rings and the scratch allowance,
//! [`draft_file::card_bytes`]) are a reserve of every plan of the run, on the card the draft sits
//! on and ahead of the context's fit, so the unset context is the largest that leaves the draft
//! its room, a set `--ctx` that leaves none is refused by name with the largest that fits, and
//! the load is held to the bytes the plan set aside. An image under `dspark` is refused by name.
//! Unset, the seat serves the plain path.
//!
//! `--mmproj FILE` gives the seat image input (behind the `vision` feature): the Pixtral tower
//! (`pixtral`, Mistral Large 4's mmproj) held against the text model — its row width, and its
//! `[IMG]` and `[IMG_END]` ids in the vocabulary — before anything is planned
//! ([`vision_seat::open_seat`]), seated where the placement leaves room
//! ([`vision_seat::encoder_seat`]: a card the plan leaves off, else a reserve on the stage card;
//! `tiered::run` puts it into the plan's machine). An image's rows are one position a row of its
//! span, the break rows between grid rows the tower's own, and the prompt call takes them in place
//! of the embeddings of their positions ([`Seat::prefill_media`], `Body::prefill_media`): the rows
//! are staged by the batched walk as the text ids' embeddings are, one residency pass a call. The
//! `vision_seat` record follows the `plan` record and the `vision` record the `load` record.
//! `--no-mmproj` is the `--hf` repo's projector turned off, which this seat's flag parser takes no
//! valueless flag for: a repo's projector is taken only where the plan holds it
//! ([`vision_seat::kept`]).
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the largest visible card),
//! `bp` (plan (b′): the A6000 the stage card, the 3090 an expert tier holding each layer's next
//! ids), `gate`, or one card of the census by name or ordinal. Unset, the common rule decides
//! (`Place::choose` by [`TIER_RULE`]): two cards in view run `a` until a sitting has shown the tier
//! not slower (the rule carries no break-even), and a `place unset` record names the word and why.
//!
//! `--ctx` is the total context the slots share (`--ctx 0` is refused); `--parallel N` gives each
//! slot an Nth of it. The context comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's trained context stops by
//! name, and one the machine cannot hold stops with the largest it would hold. The plan reads the
//! host room (`BLOOMERY_HOST_ROOM` replaces the reading) and pages the routed experts the room
//! cannot hold on the NVMe expert tier, which the body's host run serves.
//!
//! The records on stderr, in order: `place unset` (word and why), the `residency lever` record
//! (the word and why), one `ctx` line, the `plan` record, the `vision_seat` record under
//! `--mmproj`, under `mid` the `residency host` record of the churn pool, the `need` record, the
//! `parallel` line (`rule=slots slots=N slot_ctx=C total=N*C from=<word> pass=one` for one slot
//! and `pass=turns` for several); then the `xstream=` line (the word, then why), the `load` record
//! (architecture, resident bytes, the body's scratch, a slot's context, the slots, layers, the
//! routed experts the plan keeps on the host and on the card, the step graph's nodes, how a prompt
//! is fed, the batches a prompt group runs and the load's wall), the `nvtier` record (on a load
//! whose plan paged routed experts), the `vision` record under `--mmproj` and, once the port is
//! bound, the `listening` record (`record::BLOOMERY_SERVE_MISTRALLARGE4`). `--plan` ends the
//! process after the `parallel` line, nothing loaded. An engine error ends the process with the
//! crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside them: the plan's card
//! budget, the host set's load settings, `BLOOMERY_PIN_MAIN` with `BLOOMERY_HOST_LANES` (the
//! engine thread pinned to the dispatcher's cpu slot, which gives the host legs the pool's CCD
//! map), the residency word, the batches a prompt group runs (unset, the row's default), the
//! expert stream's word, the start screen and the schema jump.
//!
//! The seat sits behind the `deepseek41` feature beside `mistrallarge4`: the server surface it
//! binds through (`bind`, the `serve` and `sampler` crates) is scoped to `deepseek41`; it runs no
//! V4.1 code.
//!
//! The run is the tiered seats' ([`super::tiered`]); the open is this module's.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use app::arch::mistrallarge4::{PROMPT_FEED, dspark};
use app::{Prompt, Session};
use bloomery_gpu::GpuError;
use bloomery_gpu::checkpoint::planes::{seq_bytes, seq_lane};
use bloomery_gpu::checkpoint::saved::SaveLane;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::{Place, TierRule};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::vision_seat::{self, PleImage, TextFacts};
use bloomery_gpu_mistrallarge4::body::Body;
use bloomery_gpu_mistrallarge4::body::rows::PASS_ROWS;
use bloomery_gpu_mistrallarge4::features::attach_features;
use bloomery_gpu_mistrallarge4::prefill::set_prefill_group;
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::arch::dspark::DraftCardBytes;
use model::arch::mistrallarge4::dspark as draft_file;
use model::arch::mistrallarge4::hparams::Hparams;
use model::arch::mistrallarge4::place::{self, PlanInputs as Ml4Inputs};
use model::arch::place_hybrid::PlanInputs;
use model::placement::workstation::{GRANULE, TierBatchBytes};
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use runtime::width::Mode as WidthMode;
#[cfg(feature = "vision")]
use serve::media::{MediaFeed, SharedMediaModel};
use serve::reasoning::ThinkTags;
use serve::{CacheNote, Drafted, EngineError, EngineProps, Pending, Saved, ServeError, StateError};
use vision::arch::pixtral;

use super::planes;
use super::served::ServedDraft;
use super::tiered::{self, CacheFlag, Drafting, Open, Tiered};
use super::xstream_rule::{Stage, plans_ring, xstream_rule};
use crate::draft::Draft;
use crate::quiet::Quiet;

/// The binary's name, its records' owner and the `note` line's.
const NAME: &str = "bloomery-serve-mistrallarge4";

const USAGE: &str = "usage: bloomery-serve --model mistrallarge4 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--parallel N] [--slot-ctx S] [--cache-ram MIB] \
                     [--queue-depth Q] [--slot-save-path DIR] [--api-key KEY] \
                     [--api-key-file FNAME] [--mmproj FILE] [--image-min-tokens N] \
                     [--image-max-tokens N]";

/// The levers this seat acts on: the draft (`BLOOMERY_DRAFT`, its width chooser and the DSpark
/// draft's card), the plan's card budget and the host set's load settings, which
/// the open reads (`PlanLevers::from_levers`, `Levers::host`), the residency machine's word, the
/// batches a prompt group runs, the engine thread's pin, the host legs' lane rule, the start
/// screen, the schema jump and the expert stream's word, which the plan and the open resolve by
/// the common rule ([`xstream_rule`]).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_WIDTH,
    bloomery_gpu_gates::llama_place::DRAFT_CARD,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::RESIDENCY,
    bloomery_levers::PREFILL_GROUP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::XSTREAM,
];

/// The family's input to the common unset rule (`Place::choose`): the one expert tier card the
/// body hangs, with no break-even — no sitting has shown the tier not slower, so two cards in view
/// run `a`.
pub const TIER_RULE: TierRule = TierRule {
    tiers: bloomery_gpu::host::SERVED_TIERS,
    break_even: None,
    basis: "",
};

/// What a refusal of another seat's state names ([`planes::resume`]).
const WHOSE: &str = "a mistrallarge4 body's";

/// The tags of the file's reasoning span, as its chat template spells them.
const THINK: ThinkTags = ThinkTags {
    open: "[THINK]",
    close: "[/THINK]",
};

/// The seat's own flags: `--cache-ram` and the image input's.
#[derive(Default)]
pub struct Flags {
    cache: CacheFlag,
    vision: vision_seat::Flags,
}

/// The seat: the session and the positions one slot's stores were sized for.
struct MistralLarge4 {
    s: Session<Body>,
    ctx: usize,
    /// The draft the seat serves ([`ServedDraft`]), verified by the body's pass of two rows
    /// behind the width chooser `BLOOMERY_MTP_WIDTH` names.
    draft: ServedDraft<Body>,
    /// The copy stream and pinned staging saved states are made on
    /// ([`Seat::snapshots_async`]); none with the cache off.
    lane: Option<SaveLane>,
    /// The `tier census` records' span start ([`record::host_tier::CensusSpan`]): each
    /// prompt prints the steps since the last one's, then its own.
    census: record::host_tier::CensusSpan,
    /// The resident image encoder under `--mmproj`.
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
}

/// The DSpark draft's file (`$BLOOMERY_DSPARK_MODEL`) opened, and the target layers its header
/// names: the layers the target's feature tap reads.
struct SparkFile {
    path: PathBuf,
    split: Split,
    taps: Vec<usize>,
}

impl SparkFile {
    fn read() -> Result<SparkFile, GateError> {
        let path = crate::dspark::draft_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let taps = dspark::taps(&split)?;
        Ok(SparkFile { path, split, taps })
    }
}

/// The DSpark draft's card bytes from the two files' headers ([`draft_file::card_bytes`]): what
/// the plan reserves for it before the context is fitted, and what its load is held to.
fn spark_bytes(spark: &Split, target: &Split) -> Result<DraftCardBytes, GateError> {
    draft_file::card_bytes(spark, target, GRANULE)
        .map_err(|e| format!("the DSpark draft's card bytes: {e}").into())
}

/// The DSpark draft of `spark` on the card the placement rule names
/// ([`crate::dspark::draft_card`]) for the target `s` loaded from `target` under `place`, behind
/// the width chooser of `width`, its `load draft=dspark` record printed. The plan set the draft's
/// bytes aside on that card (`machine` holds the reserve, else the open is refused by name before
/// any allocation), and the load is held to them, term by term and in all
/// ([`crate::draft::hold_load`]).
fn open_dspark(
    s: &mut Session<Body>,
    (target, place, machine): (&Path, Place, &Machine),
    spark: SparkFile,
    width: WidthMode,
) -> Result<ServedDraft<Body>, GateError> {
    let t = Instant::now();
    let on = crate::dspark::draft_card(place)?;
    let target_file =
        Arc::new(Split::open(target).map_err(|e| format!("open {}: {e}", target.display()))?);
    let planned = spark_bytes(&spark.split, &target_file)?;
    let seat = on.seat(planned.total());
    place.check_draft_card(seat, &on)?;
    seat.check_planned(machine)?;
    let context = Arc::clone(s.model().gpu().context());
    let mut d = dspark::open_draft::<Body>(
        context,
        &spark.split,
        target_file,
        (on.spec.name, on.spec.device),
    )?;
    let resident = d.resident_bytes() as u64;
    crate::draft::hold_load(&planned, d.held(), resident, seat, (place, &on.spec))?;
    let (free, total) = d.mem_info()?;
    let tapped = s.model().body(NAME)?;
    Record::new(&record::LOAD_DRAFT)
        .w("draft", "dspark")
        .w("card", d.card())
        .u(
            "width",
            <dspark::Ml4Dspark as runtime::Draft<Session<Body>>>::WIDTH,
        )
        .list("target_layers", tapped.feature_layers().unwrap_or_default())
        .u("feature_width", tapped.feature_width())
        .u("resident", resident)
        .u("working", d.held().working)
        .u("reserve", seat.reserve.bytes)
        .u("taken", d.taken_bytes())
        .u("draft_card_free", free)
        .u("draft_card_total", total)
        .f("load_s", t.elapsed().as_secs_f64())
        .eprint();
    ServedDraft::dspark::<_, PASS_ROWS>(s, d, width, &mut Quiet, Some(spark.path))
        .map_err(Into::into)
}

impl MistralLarge4 {
    /// The session of the file at `a.path` on `a.place`, the step captured, `a.slots` resident
    /// sequences parked after it (each capturing on its first use), the `load` record on stderr,
    /// on the calling thread, pinned to the dispatcher's cpu slot when asked.
    ///
    /// The plan is [`Ml4Inputs::plan_with_slots`] at `a.ctx` and `a.slots` over the machine the
    /// run planned on (the encoder's reserve in it), the same call the run's probe made, over
    /// inputs priced for the prompt group `a.group` ([`place::configure_group`], as the run priced
    /// them) and, where the run kept the expert stream's ring (`a.ring`), for the ring
    /// ([`Tiered::with_ring`], after the group: the ring's bytes join the group's reserve). The
    /// body loads under `a.residency`, is set to that group once the step is captured, and the
    /// expert stream's word is resolved last.
    fn open(a: Open<Flags>) -> Result<MistralLarge4, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        // The DSpark draft's file is read before the target's load, which takes a minute: its
        // header gives the layers the target's feature tap reads.
        let spark = match a.draft {
            Some(Drafting {
                word: Draft::Dspark,
                ..
            }) => Some(SparkFile::read()?),
            _ => None,
        };
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut inputs = Ml4Inputs::describe(&file)?;
        place::configure_group(&mut inputs, a.group)?;
        let inputs = if a.ring {
            MistralLarge4::with_ring(inputs)?
        } else {
            inputs
        };
        let machine = a.machine;
        let ctx = u64::try_from(a.ctx)
            .map_err(|_| format!("a context of {} positions passes u64", a.ctx))?;
        let plan = inputs
            .plan_with_slots(&machine, ctx, &a.plan, a.slots)
            .map_err(|e| GpuError::plan("mistrallarge4 seat", e))?;
        if plan.ctx_max != ctx {
            return Err(format!(
                "the plan serves {} positions; the slot asked for {ctx}",
                plan.ctx_max
            )
            .into());
        }
        let host_experts = plan.host.experts;
        let card_experts = plan
            .cards
            .first()
            .map(|c| c.experts)
            .ok_or("the plan has no card")?;
        let opened = t.elapsed();
        let mut model = Body::open_placed_with(file, &plan, &inputs, 0, a.host, a.residency)?;
        // The feature tap before the first capture: the captured step and verify pass hold its
        // copies.
        if let Some(spark) = &spark {
            attach_features(&mut model, &spark.taps)?;
        }
        model.set_mode(StepMode::Graph);
        let nodes = model.capture_step()?;
        set_prefill_group(&mut model, a.group)?;
        let ctx_u32 = u32::try_from(plan.ctx_max)
            .map_err(|_| format!("the plan's ctx_max {} passes u32", plan.ctx_max))?;
        let mut s = Session::from_model(model, ctx_u32);
        // The draft after the step's capture, its verify pass captured on the live sequence, and
        // before the other resident slots are made (each captures on its first use).
        let draft = match (a.draft, spark) {
            (None, _) => ServedDraft::off(),
            (
                Some(Drafting {
                    word: Draft::Lookup,
                    width,
                }),
                _,
            ) => ServedDraft::lookup(&mut s, a.slots.get(), width, &mut Quiet, None)?,
            (
                Some(Drafting {
                    word: Draft::Dspark,
                    width,
                }),
                Some(spark),
            ) => open_dspark(&mut s, (&a.path, a.place, &machine), spark, width)?,
            (Some(_), _) => {
                return Err("the draft the run parsed has no file to open".into());
            }
        };
        if a.slots.get() > 1 {
            s.add_slots(a.slots.get())?;
        }
        let lane = match a.cache_ram {
            0 => None,
            ram => Some(seq_lane(s.model_mut(), ram)?),
        };
        // The expert stream last: a started ring takes the card's free bytes past its keep, so
        // every card buffer the open makes comes before it.
        let (gpu, _, body) = s.model_mut().body_parts(NAME)?;
        let line = xstream_rule(gpu, body, a.xstream, Stage::of(a.place), a.residency)?;
        eprintln!("{line}");
        if let Some((card, rate)) = &a.lane {
            record::lane(card, rate).eprint();
        }
        let total = t.elapsed();
        Record::new(&record::LOAD_MISTRALLARGE4)
            .w("arch", <Body as ChainBody>::arch().name())
            .u("resident_bytes", s.model().resident_bytes())
            .u("scratch_bytes", s.model().body(NAME)?.scratch_bytes())
            .u("ctx", a.ctx)
            .u("slots", a.slots.get())
            .u("layers", s.model().layers().len())
            .u("host_experts", host_experts)
            .u("card_experts", card_experts)
            .u("graph_nodes", nodes)
            .w("prompt", PROMPT_FEED)
            .u("group", s.model().body(NAME)?.prefill_group())
            .w("pin_main", if a.pin_main { "on" } else { "off" })
            .w("pinned", pinned)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(s.model(), total, opened);
        let hybrid = s.model().body(NAME)?.hybrid();
        record::host_tier::eprint_nvtier(hybrid.nvme_tier());
        let mut census = record::host_tier::CensusSpan::default();
        census.mark("load", record::census_of(hybrid));
        // The encoder after the model's captures: its tower loads on the card the plan seated.
        #[cfg(feature = "vision")]
        let vision = vision_seat::open_resident(s.model().gpu().context(), a.vision)?;
        Ok(MistralLarge4 {
            s,
            ctx: a.ctx,
            draft,
            lane,
            census,
            #[cfg(feature = "vision")]
            vision,
        })
    }
}

impl Seat for MistralLarge4 {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The resident sequences the seat made at its open (`app::Session::slots`).
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot (`app::Session::select_slot`): the model exchanges its live sequence
    /// with the slot's parked state, so the DSpark draft no longer follows it
    /// ([`ServedDraft::moved`]); the lookup keeps a context a slot.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if slot == self.s.selected() {
            return Ok(());
        }
        self.s.select_slot(slot)?;
        self.draft.moved();
        Ok(())
    }

    /// The prompt in batches of rows (`app::Prompt for Body`), the argmax after the last, through
    /// the draft ([`ServedDraft::prefill`]: the DSpark draft's own call hands it the call's
    /// feature rows, the lookup takes the call's ids); the `tier census` records of the steps
    /// since the last prompt and of this one, then the `nvtier` record.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.census.steps(Some(self.s.model().body(NAME)?.hybrid()));
        let next = self.draft.prefill(&mut self.s, ids, |s| {
            Ok(<Body as Prompt>::prompt(s.model_mut(), ids)?)
        })?;
        self.census
            .prompt(Some(self.s.model().body(NAME)?.hybrid()));
        Ok(next)
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// Each feed's image encoded on the resident encoder and read back as bf16, one row a
    /// position of its span with the break rows between grid rows the tower's own
    /// ([`vision_seat::Resident::encode_whole`]), then one prompt call with the rows in place of
    /// their positions' embeddings ([`vision_seat::Encoded::prompt`]). The encode binds the
    /// model's card back before the prompt call.
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        let MistralLarge4 {
            s,
            vision,
            draft,
            census,
            ..
        } = self;
        let Some(v) = vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        census.steps(Some(s.model().body(NAME)?.hybrid()));
        let next = draft.prefill_media(s, ids, |s| {
            let encoded = v.encode_whole(
                s.model().gpu().context(),
                feeds
                    .iter()
                    .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
            )?;
            encoded.prompt::<Body, _>(s, ids, |rows| rows.media_spans())
        })?;
        census.prompt(Some(s.model().body(NAME)?.hybrid()));
        Ok(next)
    }

    /// One plain step, which the draft takes in ([`ServedDraft::stepped`]: the lookup's context
    /// gains the position, the DSpark draft appends the step's feature row).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.model_mut().step(&[last])?;
        self.draft.stepped(&mut self.s, last, next)?;
        Ok(next)
    }

    /// One pass from `last` through the draft ([`ServedDraft::pass`]); the plain step without one.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        match self.draft.pass(&mut self.s, last, out)? {
            Some(counts) => Ok(counts),
            None => {
                out.push(self.step(last)?);
                Ok(Drafted::default())
            }
        }
    }

    /// The most rows a pass verifies ([`ServedDraft::rows`]): the draft's window.
    fn pass_rows(&self) -> usize {
        self.draft.rows()
    }

    /// The lookup's state is a context a slot ([`ServedDraft::slot_drafts`]); the DSpark draft
    /// serves one slot.
    fn slot_drafts(&self) -> bool {
        self.draft.slot_drafts()
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The session's reset ([`bloomery_gpu::GpuModel::reset`] of the selected sequence only, and
    /// no verify's rows waiting for a commit), and the draft started over with it
    /// ([`ServedDraft::reset`]).
    fn reset(&mut self) -> Result<(), GateError> {
        self.s.reset()?;
        self.draft.reset(self.s.selected())
    }

    /// The session's cut (`app::Keep`), which takes only a position [`Seat::keep`] granted; any
    /// other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        self.draft.moved();
        Ok(self.s.cut(pos)?)
    }

    /// The model's rule (`app::Keep for Body`) and the sentence that kept less: every held
    /// position, the plane stores' rule; under the DSpark draft, a cut anywhere but where the
    /// draft stands also leaves the request's prompt call a window ([`ServedDraft::keep`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.draft.keep(n, self.s.pos(), |m| self.s.keep_query(m))
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    fn snapshot_size(&mut self) -> Result<Option<u64>, GateError> {
        planes::snapshot_size(self.s.model_mut())
    }

    /// The selected slot's sequence state ([`planes::snapshot`]), as the prompt cache holds it.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        planes::snapshot(self.s.model_mut())
    }

    /// Whether the saves copy on the lane: the seat holds one.
    fn snapshots_async(&self) -> bool {
        self.lane.is_some()
    }

    /// The selected slot's state copy begun on the lane ([`planes::snapshot_begin`]); a pool
    /// with no room is the server's `NoRoom`. Without a lane, the snapshot on the call.
    fn snapshot_begin(&mut self) -> Result<Pending, GateError> {
        match &self.lane {
            Some(lane) => planes::snapshot_begin(self.s.model_mut(), lane),
            None => Ok(Pending::landed(self.snapshot()?)),
        }
    }

    /// The state put back ([`planes::resume`]) on the selected slot after its reset.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        self.draft.replaced(self.s.selected())?;
        planes::resume(self.s.model_mut(), state, WHOSE)
    }

    /// The first `n` positions the state can be cut to put back on the selected slot after its
    /// reset ([`planes::resume_prefix`]): the sequence's next call runs as it would have after
    /// the whole state's resume and a rollback to the returned position.
    fn resume_prefix(&mut self, state: &dyn Saved, n: usize) -> Result<usize, GateError> {
        self.draft.replaced(self.s.selected())?;
        planes::resume_prefix(self.s.model_mut(), state, n, WHOSE)
    }

    /// The selected slot's state in the byte form of a slot file ([`planes::save_state`]).
    fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
        planes::save_state(self.s.model_mut(), out, WHOSE)
    }

    /// The state a slot file holds, put into the selected slot as a cached one is
    /// ([`planes::restore_state`]); the positions restored.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
        self.draft
            .replaced(self.s.selected())
            .map_err(|e| StateError::Engine(EngineError(e.to_string())))?;
        planes::restore_state(self.s.model_mut(), input, WHOSE)
    }

    /// `engine.draft`, and the draft's resident bytes in its card's placement row
    /// ([`ServedDraft::props`]).
    fn props(&self, p: EngineProps) -> EngineProps {
        self.draft.props(p)
    }

    /// The draft's width record of each slot's last request printed ([`ServedDraft::close`]).
    fn close(&mut self) -> Result<(), GateError> {
        self.draft.close();
        Ok(())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

impl Tiered for MistralLarge4 {
    type H = Hparams;
    type Flags = Flags;

    const NAME: &'static str = NAME;
    const USAGE: &'static str = USAGE;
    const ACTS_ON: &'static [&'static str] = ACTS_ON;
    const ALIAS: &'static str = "mistral-large4";
    const FAMILY: &'static str = "mistrallarge4";
    const SERVES: &'static str = "mistral-large4";
    const RESIDENT: bool = true;
    /// The body serves slots in turn ([`bloomery_gpu::model::Slots`], not one pass).
    const ONE_PASS: bool = false;
    /// The seat serves the lookup and the DSpark draft ([`ServedDraft`]).
    const DRAFTS: bool = true;
    const TIER_RULE: TierRule = TIER_RULE;
    const PROJECTORS: &'static [&'static str] = &[pixtral::PROJECTOR_TYPE];
    /// The pool support the body declares: its churn pool is the host set's.
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = <Body as ChainBody>::UNHOSTED_POOL;

    fn accepts(arch: Arch) -> bool {
        matches!(arch, Arch::MistralLarge4)
    }

    /// The ring is kept out of the expert budget where the common rule starts the stream at
    /// `split` ([`plans_ring`]).
    fn ring(levers: &Levers, place: Place) -> bool {
        plans_ring(
            levers.xstream(),
            Stage::of(place),
            levers.residency() == Some("off"),
        )
    }

    /// The inputs of a plan that keeps the stream's ring and its ring pass's buffers out of the
    /// expert budget (`PlanInputs::with_stream`).
    fn with_ring(inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        Ok(inputs.with_stream()?)
    }

    /// The family's own reading ([`Ml4Inputs::describe`]), as its end-to-end gates read the
    /// file: the coverage list still names the template's parsers, which the server runs.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError> {
        Ml4Inputs::describe(split)
    }

    /// The prompt group's units, records and lists priced into the plan's prompt reserve
    /// ([`place::configure_group`]), before the run's context probes; the open prices the same.
    fn configure(inputs: &mut PlanInputs<Self::H>, group: usize) -> Result<(), GateError> {
        Ok(place::configure_group(inputs, group)?)
    }

    fn flags(flag: &str, value: &str, out: &mut Flags) -> Result<bool, GateError> {
        if out.cache.take(flag, value)? {
            return Ok(true);
        }
        // The image flags are llama-server's, each with a value: the shared parser hands a flag
        // its next argument, so `--no-mmproj` (valueless) is an unknown argument here.
        out.vision.take(flag, &mut std::iter::once(value))
    }

    fn vision(flags: &Flags) -> Option<&vision_seat::Flags> {
        Some(&flags.vision)
    }

    /// The projector is held against the text model's width; the Pixtral tower's rows are
    /// placed by the sequence's ordinary positions, so the text rope asks nothing of it.
    fn text_facts(inputs: &PlanInputs<Self::H>) -> Result<TextFacts, GateError> {
        Ok(TextFacts {
            n_embd: inputs.spec.hidden as usize,
            imrope: None,
            ple: PleImage::Absent,
        })
    }

    /// [`CacheFlag::budget`] over the plan's store bytes ([`seq_bytes`]): a default under one
    /// saved state (one position) is off. A slot's context is the state's most.
    fn cache_budget(
        flags: &Flags,
        plan: &Plan<'_>,
        pool: u64,
        slot_ctx: usize,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<u64, GateError> {
        flags.cache.budget(plan, pool, slot_ctx, |n| {
            seq_bytes(&inputs.kv, inputs.model.layers, n)
        })
    }

    /// The DSpark draft's card bytes beside the target `split` ([`spark_bytes`] over
    /// `$BLOOMERY_DSPARK_MODEL`'s header), which every plan of the run reserves on the draft's
    /// card.
    fn dspark_bytes(split: &Split) -> Result<DraftCardBytes, GateError> {
        spark_bytes(&SparkFile::read()?.split, split)
    }

    /// The tier's prompt-batch bytes of the file's widths and down forms ([`place::tier_batch`]),
    /// reserved on the tier card and the host; `None` on a placement with no tier card.
    fn tier_batch(
        place: Place,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty())
            .then(|| place::tier_batch(inputs))
            .transpose()?)
    }

    /// The tags of the reasoning span the file's chat template spells; a template that spells
    /// none splits none beside the server's default.
    fn think(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        let spelled = inputs
            .spec
            .chat
            .template
            .as_deref()
            .is_some_and(|t| t.contains(THINK.open) && t.contains(THINK.close));
        Ok(spelled.then_some(THINK))
    }

    fn register(args: &[String]) {
        record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_MISTRALLARGE4, |flag| {
            bloomery_gpu_gates::llama_launch::serve_option(args, flag)
        });
    }

    fn listening() -> Record {
        Record::new(&record::LISTENING_MISTRALLARGE4)
    }

    fn open(a: Open<Self::Flags>) -> Result<Self, GateError> {
        MistralLarge4::open(a)
    }
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<MistralLarge4>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`tiered::check`]).
pub fn check(args: &[String], read_levers: bool) -> Result<(), GateError> {
    tiered::check::<MistralLarge4>(args, read_levers).map(drop)
}
