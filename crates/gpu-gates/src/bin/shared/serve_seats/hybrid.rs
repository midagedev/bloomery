//! A common hybrid HTTP seat over one chain on one card: opening, planning,
//! resident sequence selection, load records and prompt calls. Its body's
//! sequence capability preserves named refusals for single-sequence families.
//!
//! A seat gives its family the facts the run needs — the architectures it
//! accepts, its plan inputs, the words its refusals name and the records it
//! prints — and this module holds what two families already do
//! the same way: the load on the engine thread, the `load` record, the
//! prompt call, the step, the keep rule and the tags the model's own
//! description names.
//!
//! A family whose body has a prompt pass ([`Family::PASS`]) names the pass its plan holds
//! ([`Family::pass`]); the seat plans the load at it and opens it, or steps the prompt where the
//! plan holds none or `--prefill steps` asks it, and names the walk in its records.
//!
//! A family whose prompt calls stream host experts names the ring its plan keeps out of the
//! expert budget ([`Family::ring`], [`Family::with_ring`]) and resolves the stream once the load
//! is open ([`Family::xstream`]); the seat plans the pass beside the ring, the ring's bytes after
//! the pass's, or without it where the ring leaves the pass no room.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use app::arch::{Batch, HybridCfg, SequenceSlots};
use app::{Loaded, OpenArgs, OpenLog, Prompt, Session, SessionError};
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::TierBody;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel, HostYieldMove};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{Seat, note_line};
use bloomery_gpu_gates::generate::{Lane, Place, TierRule};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::vision_seat::{self, PleImage, TextFacts};
use bloomery_levers::{DraftWord, HostCfg, Levers, ResidencyPick};
use gguf::Split;
use model::arch::Arch;
use model::arch::place_hybrid::PlanInputs;
use model::placement::ctx_ask::DEFAULT_SLOTS;
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, PlacementError, Plan};
use runtime::Target as _;
use runtime::prompt::CallEnd;
use runtime::width::Mode as WidthMode;
use serve::reasoning::ThinkTags;
use serve::{CacheNote, DraftProps, Drafted, EngineProps, Saved, ServeError};
#[cfg(feature = "vision")]
use {
    bloomery_gpu::media::{MediaBody, MediaSpan},
    serve::media::{MediaFeed, SharedMediaModel},
};

use super::hybrid_draft::SeatDraft;
use super::tiered::{self, BuildMachine, Open, PassAttempt, PassPlan, Tiered, Walk};

/// The open of a family's prompt pass ([`Family::pass`]): the body by the plan its pass inputs
/// made, with the pass's buffers beside it, under the seat's residency word.
pub type PassOpen<F> = Box<
    dyn FnOnce(
            Split,
            &Plan<'_>,
            &PlanInputs<<F as Family>::H>,
            HostCfg,
            Residency,
        ) -> Result<GpuModel<<F as Family>::Body>, GpuError>
        + Send,
>;

/// The pass a family's plan holds and its open, or why the plan holds none.
pub type Passed<F> = Result<(PassPlan<<F as Family>::H>, PassOpen<F>), String>;

/// What a hybrid seat's family gives the common owner: its body, its plan
/// inputs, and the words its refusals and its records name.
pub trait Family: Sized + 'static {
    /// The body the seat serves: one chain on one card, fed in prompt
    /// batches.
    type Body: Batch<Inputs = PlanInputs<Self::H>> + TierBody<Experts = HostRun>;

    /// The family's hyperparameters: what its plan inputs are read for.
    type H: Clone + Send + 'static;

    /// The binary's name, its records' owner and the `note` line's.
    const NAME: &'static str;
    /// The usage line the seat's flag refusals print.
    const USAGE: &'static str;
    /// The levers this seat acts on, parsed once at `main`.
    const ACTS_ON: &'static [&'static str];
    /// The alias a file that names none serves under.
    const ALIAS: &'static str;
    /// The seat word a foreign file's refusal names.
    const FAMILY: &'static str;
    /// The files the seat serves.
    const SERVES: &'static str;

    /// The family's input to the common unset rule and the tier cards its body serves
    /// ([`Tiered::TIER_RULE`]). The default serves none, and a placement that names a tier card is
    /// refused by name.
    const TIER_RULE: TierRule = TierRule::UNTIERED;

    /// Whether the body has a prompt pass ([`Family::pass`]): the seat then takes
    /// `--prefill pass|steps`, which a family with none refuses by name.
    const PASS: bool = false;

    /// The pass the plan at `ctx` holds on `place` and its open ([`Tiered::pass_plan`]'s
    /// `build` and `attempt`), or why it holds none; asked only of a family with
    /// [`Family::PASS`]. A family that says it has a pass and names none is refused by name.
    /// `inputs` carry the stream's ring ([`Family::with_ring`]) where the load keeps it, and the
    /// pass's inputs keep it beside the pass's bytes ([`PlanInputs::with_pass_reserve`]).
    fn pass(
        _inputs: &PlanInputs<Self::H>,
        _place: Place,
        _ctx: usize,
        _build: &BuildMachine<'_>,
        _attempt: &PassAttempt<'_, Self::H>,
    ) -> Result<Passed<Self>, GateError> {
        Err(format!(
            "the {} family says it has a prompt pass and names none (Family::pass)",
            Self::FAMILY
        )
        .into())
    }

    /// The host experts' yields to the host K/V's growth since the last take, oldest first
    /// (`KvRoomBody::host_grow`): the seat prints each as a `host yield` record after the call
    /// that asked for it. None for a body whose loads yield nothing (the default).
    fn host_yields(_m: &mut GpuModel<Self::Body>) -> Result<Vec<HostYieldMove>, GateError> {
        Ok(Vec::new())
    }

    /// Whether the plan of a load on `place` keeps the expert stream's ring out of the expert
    /// budget ([`Family::with_ring`]): the body's prompt calls stream host experts there
    /// (`BLOOMERY_XSTREAM` by the common rule, `shared/xstream_rule.rs`; [`Tiered::ring`]). False
    /// by default: a body that streams none.
    fn ring(_levers: &Levers, _place: Place) -> bool {
        false
    }

    /// `inputs` for a plan that keeps the ring out of the expert budget; asked where
    /// [`Family::ring`] is true. A family that answers true without this is refused by name.
    fn with_ring(_inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        Err(format!(
            "the {} seat answered a stream ring and reserves none (Family::with_ring)",
            Self::FAMILY
        )
        .into())
    }

    /// The expert stream of the opened load `m` under `BLOOMERY_XSTREAM` as `set` (`None` unset) on
    /// `place` and the residency the plan chose, by the common rule (`shared/xstream_rule.rs`), the
    /// `xstream=` line it prints; asked once every buffer the open makes is made, for a started
    /// ring takes the card's free bytes past its keep. `None` (the default) is a body whose prompt
    /// calls stream nothing.
    fn xstream(
        _m: &mut GpuModel<Self::Body>,
        _set: Option<&str>,
        _place: Place,
        _residency: Residency,
    ) -> Result<Option<String>, GateError> {
        Ok(None)
    }

    /// The rows of the prompt pass the loaded body holds, which its prompt runs by; `None` for a
    /// body loaded with none (the default).
    fn pass_rows(_m: &GpuModel<Self::Body>) -> Result<Option<usize>, GateError> {
        Ok(None)
    }

    /// Whether the family's body carries an in-file MTP layer the seat drafts with under
    /// `BLOOMERY_DRAFT=mtp` ([`Family::mtp`]); the lever is refused by name where this is false.
    const MTP: bool = false;

    /// The `load draft=mtp` record of the loaded model `m`, `took` the draft window's open. Asked
    /// only of a family with [`Family::MTP`].
    fn draft_record(_m: &GpuModel<Self::Body>, _took: Duration) -> Result<Record, GateError> {
        Err(format!("the {} seat has no in-file MTP draft", Self::FAMILY).into())
    }

    /// The in-file MTP draft of the session `s` (the body loaded with its layer: the seat sets
    /// [`app::arch::HybridCfg::draft`]), its window opened over one sequence behind the width
    /// chooser `width`. Asked only of a family with [`Family::MTP`].
    fn mtp(
        _s: &mut Session<Self::Body>,
        _width: WidthMode,
    ) -> Result<Box<dyn SeatDraft<Self::Body>>, GateError> {
        Err(format!("the {} seat has no in-file MTP draft", Self::FAMILY).into())
    }

    /// Whether `arch` is a file this seat serves.
    fn accepts(arch: Arch) -> bool;

    /// The file's plan inputs, its own reading.
    fn read(split: &Split) -> Result<PlanInputs<Self::H>, PlacementError>;

    /// `ctx` positions refused by name when the body cannot serve that many; asked of the context
    /// as given, before any plan, and of the context the plan chose ([`Tiered::check_ctx`]).
    fn check_ctx(_inputs: &PlanInputs<Self::H>, _ctx: usize) -> Result<(), GateError> {
        Ok(())
    }

    /// Price the chosen prompt group before the common context probes.
    fn configure(_inputs: &mut PlanInputs<Self::H>, _group: usize) -> Result<(), GateError> {
        Ok(())
    }

    /// The records the seat prints, registered once at `main`.
    fn register(args: &[String]);

    /// The seat's listening record, the one the run fills once the port is
    /// bound.
    fn listening() -> Record;

    /// The seat's load record: the family names its kind, the common owner
    /// fills the fields every seat's load prints.
    fn load_record() -> Record;

    /// The expert tier's prompt-batch bytes the plan reserves on `place`'s
    /// tier card ([`Tiered::tier_batch`]); a body that serves no tier card
    /// reserves none (the default), and its plan refuses the card by name.
    fn tier_batch(
        place: Place,
        _inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        Ok((!place.tier_cards().is_empty()).then_some(TierBatchBytes {
            staging: 0,
            scratch: 0,
            host: 0,
            leg: None,
        }))
    }

    /// The tags the model's description names for its think span, which the
    /// server splits the reasoning on ([`serve::Server::with_think_tags`]); the
    /// description of every file of the family names them. A family whose
    /// description names none and serves plain text says `None`.
    fn think_tags(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        match inputs.spec.chat.reasoning {
            Some(format) => Ok(Some(tiered::think_tags_of(format))),
            None => Err(format!("the {} description names no reasoning tags", Self::FAMILY).into()),
        }
    }

    /// The projector types the family's text model runs (`clip.projector_type`): a family that
    /// names one takes `--mmproj` ([`Tiered::vision`]); none (the default) takes no image.
    const PROJECTORS: &'static [&'static str] = &[];

    /// What the projector is held against ([`Tiered::text_facts`]): the text model's width, and,
    /// by default, ordinary positions and no per-layer image input. A family whose rope or input
    /// the projector must agree with says so.
    fn text_facts(inputs: &PlanInputs<Self::H>) -> Result<TextFacts, GateError> {
        Ok(TextFacts {
            n_embd: inputs.spec.hidden as usize,
            imrope: None,
            ple: PleImage::Absent,
        })
    }

    /// One media prompt call of `ids` with `feeds`' images encoded on the resident encoder `v`;
    /// a family that names [`Family::PROJECTORS`] answers [`media_prompt`] over its body. The
    /// default refuses by name.
    #[cfg(feature = "vision")]
    fn prompt_media(
        _s: &mut Session<Self::Body>,
        _v: &mut vision_seat::Resident,
        _ids: &[u32],
        _feeds: &[MediaFeed],
    ) -> Result<u32, GateError> {
        Err(format!("the {} seat takes no image", Self::FAMILY).into())
    }
}

/// A hybrid family's media prompt call ([`Family::prompt_media`]): each feed's image encoded on the
/// resident encoder and read back as bf16, one row a position of its span, the rows the tower writes
/// between its grid rows included ([`vision_seat::Resident::encode_whole`]), then one prompt call
/// with the rows in place of their positions' embeddings ([`vision_seat::Encoded::prompt`]). The
/// encode binds the model's card back before the prompt call.
#[cfg(feature = "vision")]
pub fn media_prompt<B>(
    s: &mut Session<B>,
    v: &mut vision_seat::Resident,
    ids: &[u32],
    feeds: &[MediaFeed],
) -> Result<u32, GateError>
where
    B: Batch + for<'a> MediaBody<Span<'a> = MediaSpan<'a>>,
{
    let encoded = v.encode_whole(
        s.model().gpu().context(),
        feeds
            .iter()
            .map(|f| vision_seat::Image::of(f.at, &f.prepared)),
    )?;
    encoded.prompt::<B, _>(s, ids, |rows| rows.media_spans())
}

/// `--prefill`: how a family with a prompt pass runs a prompt.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prefill {
    /// As passes; the plan must hold one.
    Pass,
    /// One id a decode step, the stepped load.
    Steps,
}

/// The hybrid seat's own flags and what its plan hands the open: `--prefill` as given, the pass
/// the plan holds with its open ([`Family::pass`]), and the image input's flags for a family that
/// takes images ([`Family::PROJECTORS`]).
pub struct Flags<F: Family> {
    prefill: Option<Prefill>,
    pass: Option<(PlanInputs<F::H>, PassOpen<F>)>,
    vision: vision_seat::Flags,
}

impl<F: Family> Default for Flags<F> {
    fn default() -> Self {
        Flags {
            prefill: None,
            pass: None,
            vision: vision_seat::Flags::default(),
        }
    }
}

/// The seat of a hybrid family: the session and the positions its stores were
/// sized for.
pub struct Hybrid<F: Family> {
    s: Session<F::Body>,
    ctx: usize,
    /// The in-file MTP draft under `BLOOMERY_DRAFT=mtp` ([`mtp_asked`]); `None` the plain path.
    drafted: Option<Box<dyn SeatDraft<F::Body>>>,
    /// The model file, which `/props`' `engine.draft` names under a draft.
    path: std::path::PathBuf,
    /// The `tier census` records' span start ([`record::host_tier::CensusSpan`]): each
    /// prompt prints the steps since the last one's, then its own.
    census: record::host_tier::CensusSpan,
    /// The resident image encoder under `--mmproj`.
    #[cfg(feature = "vision")]
    vision: Option<vision_seat::Resident>,
}

/// The open's records: the plan's expert counts are kept until the step is
/// captured, where the model's size and the step graph's nodes complete the
/// `load` record.
struct Log<F> {
    t: Instant,
    /// The wall from `t` to the load, once the plan is made: the phases
    /// record's `open_s`.
    opened: Duration,
    ctx: usize,
    group: usize,
    slots: usize,
    host_experts: u64,
    card_experts: u64,
    graph_nodes: usize,
    /// `BLOOMERY_PIN_MAIN`'s ask and whether the pin took.
    pin_main: bool,
    pinned: bool,
    /// The lane's rate probed before the plan, and the card it ran on: the
    /// `lane` line beside the `load` record.
    lane: Option<Lane>,
    /// The family the log records for.
    _family: PhantomData<F>,
}

impl<F: Family> OpenLog<F::Body> for Log<F> {
    fn plan(
        &mut self,
        _place: &'static str,
        _inputs: &PlanInputs<F::H>,
        _machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        self.host_experts = plan.host.experts;
        self.card_experts = plan.cards[0].experts;
        self.opened = self.t.elapsed();
        Ok(true)
    }

    fn load(&mut self, _m: &GpuModel<F::Body>) -> Result<(), SessionError> {
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        self.graph_nodes = nodes;
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &GpuModel<F::Body>) -> Result<(), SessionError> {
        Ok(())
    }

    /// The card-byte invariant's readings, printed and refused as this process's
    /// card hold says ([`bloomery_gpu_gates::card_bytes::watch`]).
    fn card_watch(&mut self) -> Result<Box<dyn app::CardWatch>, SessionError> {
        bloomery_gpu_gates::card_bytes::watch()
    }
}

impl<F: Family> Log<F> {
    /// The `load` record of `m`, with every planned slot ready.
    fn record(&self, m: &GpuModel<F::Body>) -> Result<(), GateError> {
        if let Some((card, rate)) = &self.lane {
            record::lane(card, rate).eprint();
        }
        let total = self.t.elapsed();
        let r = F::load_record()
            .w("arch", <F::Body as ChainBody>::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("slots", self.slots)
            .u("layers", m.layers().len())
            .u("host_experts", self.host_experts)
            .u("card_experts", self.card_experts)
            .u("graph_nodes", self.graph_nodes)
            .u("group", self.group)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned);
        let r = match F::pass_rows(m)? {
            Some(rows) => r.u("pass_rows", rows),
            None => r,
        };
        r.f("load_s", total.as_secs_f64()).eprint();
        record::eprint_load_phases(m, total, self.opened);
        record::host_tier::eprint_nvtier(m.body(F::NAME).ok().and_then(|b| b.hybrid().nvme_tier()));
        Ok(())
    }
}

/// `BLOOMERY_DRAFT` as a hybrid seat reads it: `mtp` (with the width mode `BLOOMERY_MTP_WIDTH`
/// names) drafts with the file's MTP layer, `off` and unset run the plain path; any other word, and
/// `mtp` on a family with no MTP layer, are refused by name.
fn mtp_asked<F: Family>() -> Result<Option<WidthMode>, GateError> {
    let levers = Levers::from_env().map_err(|e| e.to_string())?;
    match levers.draft_word(&[DraftWord::Mtp, DraftWord::Off]) {
        Ok(Some(DraftWord::Mtp)) if F::MTP => Ok(Some(
            WidthMode::of(levers.mtp_width()).map_err(|e| e.to_string())?,
        )),
        Ok(Some(DraftWord::Mtp)) => Err(format!(
            "BLOOMERY_DRAFT=mtp: the {} seat has no in-file MTP layer to draft with",
            F::FAMILY
        )
        .into()),
        Ok(_) => Ok(None),
        Err(e) => Err(format!(
            "BLOOMERY_DRAFT is mtp or off on the {} seat, not {:?}",
            F::FAMILY,
            e.word.as_str()
        )
        .into()),
    }
}

/// The session of the file at `a.path` on `a.place`, the step captured, the
/// `load` record on stderr, on the calling thread, pinned to the dispatcher's
/// cpu slot when asked.
fn open_session<F: Family>(mut a: Open<Flags<F>>) -> Result<Hybrid<F>, GateError> {
    record::require(record::groups::hybrid::GROUP, "hybrid seat");
    let pinned = a.pin_main && threads::pool().pin_caller();
    let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
    let mut log = Log::<F> {
        t: Instant::now(),
        opened: Duration::ZERO,
        ctx: a.ctx,
        group: a.group,
        slots: a.slots.get(),
        pin_main: a.pin_main,
        pinned,
        host_experts: 0,
        card_experts: 0,
        graph_nodes: 0,
        lane: a.lane,
        _family: PhantomData,
    };
    let machine = a.machine;
    let mtp = mtp_asked::<F>()?;
    if mtp.is_some() && a.slots.get() != 1 {
        return Err(format!(
            "BLOOMERY_DRAFT=mtp on the {} seat drafts one slot: --parallel {} is refused \
             (--parallel 1)",
            F::FAMILY,
            a.slots
        )
        .into());
    }
    let cfg = HybridCfg {
        place: a.plan,
        host: a.host,
        residency: a.residency,
        group: a.group,
        slots: a.slots,
        pages: a.room_pages,
        draft: mtp.is_some(),
    };
    let loaded = match a.flags.pass.take() {
        Some((inputs, open)) => open_pass(file, (&inputs, open), &machine, (a.ctx, cfg), &mut log)?,
        None if a.ring => open_stepped_ring(file, &machine, (a.ctx, cfg), &mut log)?,
        None => {
            let args = OpenArgs {
                place: a.place.name(),
                machine: move |_| machine.clone(),
                ctx: a.ctx,
                mode: StepMode::Graph,
                cfg,
            };
            Loaded::<F::Body>::open(file, args, &mut log)?.ok_or("the open stopped at its plan")?
        }
    };
    let mut s = loaded.ready(&mut log)?;
    <F::Body as Batch>::Sequences::add(&mut s, a.slots)?;
    // The expert stream after every buffer the open makes: a started ring takes the card's free
    // bytes past its keep.
    if let Some(line) = F::xstream(s.model_mut(), a.xstream, a.place, a.residency)? {
        eprintln!("{line}");
    }
    log.record(s.model())?;
    let drafted = match mtp {
        Some(width) => {
            let t = Instant::now();
            let d = F::mtp(&mut s, width)?;
            F::draft_record(s.model(), t.elapsed())?.eprint();
            Some(d)
        }
        None => None,
    };
    let mut census = record::host_tier::CensusSpan::default();
    census.mark("load", record::census_of(s.model().body(F::NAME)?.hybrid()));
    // The encoder after the model's captures: its tower loads on the card the plan seated.
    #[cfg(feature = "vision")]
    let vision = vision_seat::open_resident(s.model().gpu().context(), a.vision.take())?;
    #[cfg(not(feature = "vision"))]
    if a.vision.is_some() {
        return Err(format!(
            "--mmproj: the {} seat is built without image input",
            F::FAMILY
        )
        .into());
    }
    Ok(Hybrid {
        s,
        ctx: a.ctx,
        drafted,
        path: a.path,
        census,
        #[cfg(feature = "vision")]
        vision,
    })
}

/// The load of a family's prompt pass: the plan of the pass's `inputs` on the run's `machine` at
/// `ctx` (`app::Open::plan`, as [`Loaded::open`] plans), handed to `log`, then the body by `open`
/// with the pass's buffers under the configuration's residency, in graph mode.
fn open_pass<F: Family>(
    file: Split,
    (inputs, open): (&PlanInputs<F::H>, PassOpen<F>),
    machine: &Machine,
    (ctx, cfg): (usize, HybridCfg),
    log: &mut Log<F>,
) -> Result<Loaded<F::Body>, GateError> {
    let plan = <F::Body as app::Open>::plan(inputs, machine, ctx, &cfg)?;
    log.plan("", inputs, machine, &plan)?;
    let held = u32::try_from(plan.ctx_max)
        .map_err(|_| format!("the plan's ctx_max {} passes u32", plan.ctx_max))?;
    let mut m = open(file, &plan, inputs, cfg.host, cfg.residency)?;
    m.set_mode(StepMode::Graph);
    Ok(Loaded::from_model(m, cfg, held).watched(log.card_watch()?)?)
}

/// The stepped load of a plan that keeps the stream's ring ([`Open::ring`]): the file's inputs
/// with the ring ([`Family::with_ring`]) planned on the run's `machine` at `ctx`, as the run's plan
/// of the same inputs was, and loaded by the body's own open ([`app::Open::open`]).
fn open_stepped_ring<F: Family>(
    file: Split,
    machine: &Machine,
    (ctx, cfg): (usize, HybridCfg),
    log: &mut Log<F>,
) -> Result<Loaded<F::Body>, GateError> {
    let mut inputs = <F::Body as app::Open>::inputs(&file)?;
    <F::Body as app::Open>::configure_inputs(&mut inputs, &cfg)?;
    let inputs = F::with_ring(inputs)?;
    let by = cfg.clone();
    let open: PassOpen<F> = Box::new(move |file, plan, inputs, _, _| {
        <F::Body as app::Open>::open(file, inputs, plan, &by)
    });
    open_pass(file, (&inputs, open), machine, (ctx, cfg), log)
}

impl<F: Family> Hybrid<F> {
    /// The `host yield` records of the host K/V's growth the call asked for, on stderr; nothing
    /// (and no allocation) on a call that yielded none.
    fn print_host_yields(&mut self) -> Result<(), GateError> {
        for m in F::host_yields(self.s.model_mut())? {
            record::host_yield(&m).eprint();
        }
        Ok(())
    }
}

impl<F: Family> Seat for Hybrid<F> {
    fn slots(&self) -> usize {
        <F::Body as Batch>::Sequences::count(&self.s)
    }

    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        Ok(<F::Body as Batch>::Sequences::select(&mut self.s, slot)?)
    }

    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt in batches (`app::Prompt for Body`), the argmax after the
    /// last;
    /// the `tier census` records of the steps since the last prompt and of
    /// this one, then the `nvtier` record.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.census
            .steps(Some(self.s.model().body(F::NAME)?.hybrid()));
        let next = match self.drafted.as_mut() {
            Some(d) => d.prompt_call(&mut self.s, ids, CallEnd::Argmax)?,
            None => <F::Body as Prompt>::prompt(self.s.model_mut(), ids)?,
        };
        self.s
            .card_check(model::placement::card_made::CardPhase::FirstPrompt)?;
        self.census
            .prompt(Some(self.s.model().body(F::NAME)?.hybrid()));
        self.print_host_yields()?;
        Ok(next)
    }

    /// The resident encoder's media description ([`vision_seat::Resident::media`]); `None`
    /// without `--mmproj`, a seat that takes no image.
    #[cfg(feature = "vision")]
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.vision.as_ref().map(vision_seat::Resident::media)
    }

    /// The family's media prompt call ([`Family::prompt_media`]) with the records and the card
    /// check of [`Seat::prefill`]. Refused by name: no `--mmproj`, and the in-file draft, whose
    /// window has no media call.
    #[cfg(feature = "vision")]
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        if feeds.is_empty() {
            return self.prefill(ids);
        }
        let Some(v) = self.vision.as_mut() else {
            return Err("no --mmproj: this seat serves no image".into());
        };
        if self.drafted.is_some() {
            return Err(format!(
                "an image prompt under BLOOMERY_DRAFT=mtp: the {} seat's draft has no media call",
                F::FAMILY
            )
            .into());
        }
        self.census
            .steps(Some(self.s.model().body(F::NAME)?.hybrid()));
        let next = F::prompt_media(&mut self.s, v, ids, feeds)?;
        self.s
            .card_check(model::placement::card_made::CardPhase::FirstPrompt)?;
        self.census
            .prompt(Some(self.s.model().body(F::NAME)?.hybrid()));
        self.print_host_yields()?;
        Ok(next)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = match self.drafted.as_mut() {
            Some(d) => d.step(&mut self.s, last)?,
            None => self.s.model_mut().step(&[last])?,
        };
        self.s
            .card_check(model::placement::card_made::CardPhase::FirstStep)?;
        self.print_host_yields()?;
        Ok(next)
    }

    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        match self.drafted.as_mut() {
            Some(d) => d.step_row(&mut self.s, last, row),
            None => {
                let arg = self.step(last)?;
                self.logits_into(row)
                    .map_err(|e| format!("logits of the step: {e}"))?;
                Ok(arg)
            }
        }
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    fn slot_drafts(&self) -> bool {
        self.drafted.as_ref().is_some_and(|d| d.drafts())
    }

    fn drafts_sampled(&self) -> bool {
        self.drafted.as_ref().is_some_and(|d| d.drafts())
    }

    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        match self.drafted.as_mut() {
            Some(d) => d.pass(&mut self.s, last, out),
            None => {
                out.push(self.step(last)?);
                Ok(Drafted::default())
            }
        }
    }

    fn pass_rows(&self) -> usize {
        self.drafted.as_ref().map_or(1, |d| d.pass_rows(&self.s))
    }

    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        match self.drafted.as_mut() {
            Some(d) => d.pass_sampled(&mut self.s, last, history, sampler, best, out),
            None => Err("this seat does not draft a sampled request".into()),
        }
    }

    /// The model's reset ([`GpuModel::reset`]), the draft started over with it under a draft.
    fn reset(&mut self) -> Result<(), GateError> {
        match self.drafted.as_mut() {
            Some(d) => d.reset(&mut self.s),
            None => Ok(self.s.model_mut().reset()?),
        }
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name. Under a draft the draft's store
    /// stays through the cut's own position ([`SeatDraft::cut`]).
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        match self.drafted.as_mut() {
            Some(d) => d.cut(&mut self.s, pos),
            None => Ok(self.s.cut(pos)?),
        }
    }

    /// `/props`' `engine.draft` under a draft: the target file, whose next-token layer it is.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if self.drafted.is_some() {
            p.draft = Some(DraftProps {
                model: self.path.file_name().map_or_else(
                    || self.path.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                ),
                n_max: self.drafted.as_ref().map(|d| d.width()),
                kind: Some("mtp".to_owned()),
                path: Some(self.path.display().to_string()),
                device: None,
            });
        }
        p
    }

    /// The model's rule (`app::Keep for Body`) and the sentence that kept
    /// less: every held position, the empty model, or a checkpoint; under a draft, the draft's
    /// own cut rule over it ([`SeatDraft::keep`]).
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let body = |m| self.s.keep_query(m);
        match self.drafted.as_ref() {
            Some(d) => d.keep(&self.s, n, &body),
            None => body(n),
        }
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps; the
    /// body takes no checkpoint at a mark yet.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {}", no_cache::<F>()).into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {}", no_cache::<F>()).into())
    }

    fn close(&mut self) -> Result<(), GateError> {
        if let Some(d) = self.drafted.as_mut() {
            d.close();
        }
        Ok(self.s.model_mut().close_nodes()?)
    }

    fn note(note: &CacheNote) {
        note_line(F::NAME, note);
    }
}

/// Why the seat takes no sequence state: it runs no prompt cache.
fn no_cache<F: Family>() -> String {
    format!("the {} seat runs no prompt cache", F::FAMILY)
}

impl<F: Family> Tiered for Hybrid<F> {
    type H = F::H;
    type Flags = Flags<F>;

    const NAME: &'static str = F::NAME;
    const USAGE: &'static str = F::USAGE;
    const ACTS_ON: &'static [&'static str] = F::ACTS_ON;
    const ALIAS: &'static str = F::ALIAS;
    const FAMILY: &'static str = F::FAMILY;
    const SERVES: &'static str = F::SERVES;
    const TIER_RULE: TierRule = F::TIER_RULE;
    const RESIDENT: bool = <<F::Body as Batch>::Sequences as SequenceSlots<F::Body>>::RESIDENT;
    const ONE_PASS: bool = false;

    fn accepts(arch: Arch) -> bool {
        F::accepts(arch)
    }

    fn read(split: &Split) -> Result<PlanInputs<F::H>, PlacementError> {
        F::read(split)
    }

    fn configure(inputs: &mut PlanInputs<Self::H>, group: usize) -> Result<(), GateError> {
        F::configure(inputs, group)?;
        if mtp_asked::<F>()?.is_some() {
            <F::Body as Batch>::configure_draft(inputs)?;
        }
        Ok(())
    }

    fn check_ctx(inputs: &PlanInputs<Self::H>, ctx: usize) -> Result<(), GateError> {
        F::check_ctx(inputs, ctx)
    }

    /// One slot under `BLOOMERY_DRAFT=mtp`: the draft holds one sequence's store.
    fn default_slots() -> Result<usize, GateError> {
        Ok(if mtp_asked::<F>()?.is_some() {
            1
        } else {
            DEFAULT_SLOTS
        })
    }

    fn prefill_steps(flags: &Flags<F>) -> bool {
        flags.prefill == Some(Prefill::Steps)
    }

    /// The common machine's pick ([`tiered::residency_pick`]) with the seat's prompt feed: a
    /// step-fed body is `off` unset and refuses any other word by name.
    fn residency(
        levers: &Levers,
        flags: &Flags<F>,
        place: Place,
        plan: &Plan<'_>,
    ) -> Result<Option<ResidencyPick>, GateError> {
        let steps = Self::prefill_steps(flags);
        tiered::residency_pick(levers, place, plan, steps, Self::UNHOSTED_POOL).map(Some)
    }

    /// `--prefill pass|steps`, for a family whose body has a prompt pass; any other word, and the
    /// flag on a family with none, refused by name.
    fn flags(flag: &str, value: &str, out: &mut Flags<F>) -> Result<bool, GateError> {
        if flag != "--prefill" {
            // The image flags are llama-server's, each with a value: the shared parser hands a
            // flag its next argument, so `--no-mmproj` (valueless) is an unknown argument here.
            if F::PROJECTORS.is_empty() {
                return Ok(false);
            }
            return out.vision.take(flag, &mut std::iter::once(value));
        }
        if !F::PASS {
            return Err(format!("--prefill: the {} body has no prompt pass", F::FAMILY).into());
        }
        out.prefill = Some(match value {
            "pass" => Prefill::Pass,
            "steps" => Prefill::Steps,
            other => return Err(format!("--prefill is pass or steps, not {other}").into()),
        });
        Ok(true)
    }

    /// The family's pass at `ctx` ([`Family::pass`]), its open kept in `flags` for the engine
    /// thread: unless `--prefill steps` asks the stepped load, which plans nothing more. A plan
    /// that holds no pass steps the prompt and says why, refused by name under `--prefill pass`.
    fn pass_plan(
        flags: &mut Flags<F>,
        inputs: &PlanInputs<F::H>,
        ring: Option<&PlanInputs<F::H>>,
        place: Place,
        ctx: usize,
        build: &BuildMachine<'_>,
        attempt: &PassAttempt<'_, F::H>,
    ) -> Result<Option<Walk<F::H>>, GateError> {
        if !F::PASS {
            return Ok(None);
        }
        if flags.prefill == Some(Prefill::Steps) {
            return Ok(Some(Walk::Steps("--prefill steps".to_owned())));
        }
        match beside_ring((ring, inputs), |i| F::pass(i, place, ctx, build, attempt))? {
            Ok((plan, open)) => {
                flags.pass = Some((plan.inputs.clone(), open));
                Ok(Some(Walk::Pass(Box::new(plan))))
            }
            Err(why) if flags.prefill == Some(Prefill::Pass) => {
                Err(format!("--place {} --prefill pass: {why}", place.name()).into())
            }
            Err(why) => Ok(Some(Walk::Steps(why))),
        }
    }

    fn think(inputs: &PlanInputs<Self::H>) -> Result<Option<ThinkTags>, GateError> {
        F::think_tags(inputs)
    }

    const PROJECTORS: &'static [&'static str] = F::PROJECTORS;

    /// The image flags of a family that takes images ([`Family::PROJECTORS`]); `None` otherwise.
    fn vision(flags: &Flags<F>) -> Option<&vision_seat::Flags> {
        (!F::PROJECTORS.is_empty()).then_some(&flags.vision)
    }

    fn text_facts(inputs: &PlanInputs<Self::H>) -> Result<TextFacts, GateError> {
        F::text_facts(inputs)
    }

    fn ring(levers: &Levers, place: Place) -> bool {
        F::ring(levers, place)
    }

    fn with_ring(inputs: PlanInputs<Self::H>) -> Result<PlanInputs<Self::H>, GateError> {
        F::with_ring(inputs)
    }

    fn tier_batch(
        place: Place,
        inputs: &PlanInputs<Self::H>,
    ) -> Result<Option<TierBatchBytes>, GateError> {
        F::tier_batch(place, inputs)
    }

    fn register(args: &[String]) {
        F::register(args);
    }

    fn listening() -> Record {
        F::listening()
    }

    fn open(a: Open<Self::Flags>) -> Result<Self, GateError> {
        open_session(a)
    }
}

/// What `ask` answers of the inputs that keep the stream's ring (`ring`, where the load keeps it),
/// else of the `plain` inputs: a pass the ring leaves no room is asked again without it, the ring's
/// refusal printed, and the plain answer is the seat's.
fn beside_ring<I, P>(
    (ring, plain): (Option<&I>, &I),
    mut ask: impl FnMut(&I) -> Result<Result<P, String>, GateError>,
) -> Result<Result<P, String>, GateError> {
    if let Some(ring) = ring {
        match ask(ring)? {
            Ok(pass) => return Ok(Ok(pass)),
            Err(why) => eprintln!(
                "xstream ring: the pass has no room beside it, so it plans without: {why}"
            ),
        }
    }
    ask(plain)
}

/// Loads the model, and the embedding model `side` names beside it, and serves
/// until the listener or the engine fails; `Ok` carries why the server ended.
pub fn run<F: Family>(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    tiered::run::<Hybrid<F>>(args, side)
}

/// The seat's levers (unless `read_levers` is false) and flags, checked before the model is fetched
/// ([`tiered::check`]).
pub fn check<F: Family>(args: &[String], read_levers: bool) -> Result<(), GateError> {
    tiered::check::<Hybrid<F>>(args, read_levers).map(drop)
}

#[cfg(test)]
mod tests {
    use bloomery_gpu_gates::GateError;

    use super::beside_ring;

    /// The pass of a load that keeps the stream's ring is asked of the ring's inputs first (the
    /// pass's bytes, the ring's after them), and where the ring leaves it no room of the plain
    /// ones; a load that keeps no ring asks the plain inputs alone, and no room anywhere is the
    /// plain inputs' reason. An error is the seat's, not a rung to step down from. Mutants: the
    /// plain inputs asked first (the ring never wins), the ring's refusal returned in the plain's
    /// place.
    #[test]
    fn a_pass_beside_the_ring_steps_down_to_the_plain_pass() {
        let fits = |held: &'static [&'static str]| {
            move |i: &&'static str| -> Result<Result<&'static str, String>, GateError> {
                Ok(if held.contains(i) {
                    Ok(*i)
                } else {
                    Err(format!("{i} has no room"))
                })
            }
        };
        let (ring, plain) = ("ring", "plain");
        let ask = |kept: Option<&&'static str>, held| beside_ring((kept, &plain), fits(held));
        assert_eq!(ask(Some(&ring), &["ring", "plain"]).unwrap(), Ok("ring"));
        assert_eq!(ask(Some(&ring), &["plain"]).unwrap(), Ok("plain"));
        assert_eq!(ask(None, &["ring", "plain"]).unwrap(), Ok("plain"));
        assert_eq!(
            ask(Some(&ring), &[]).unwrap(),
            Err("plain has no room".to_owned()),
            "the plain inputs' reason is the seat's"
        );
        let mut asked = Vec::new();
        let _ = beside_ring((None, &plain), |i: &&str| {
            asked.push(*i);
            Ok(Ok::<&str, String>(*i))
        });
        assert_eq!(asked, ["plain"], "no ring kept, none asked");
        let e = beside_ring((Some(&ring), &plain), |_: &&str| {
            Err::<Result<&str, String>, GateError>("a refusal".into())
        })
        .expect_err("an error ends the ask");
        assert!(e.to_string().contains("a refusal"), "{e}");
    }
}
