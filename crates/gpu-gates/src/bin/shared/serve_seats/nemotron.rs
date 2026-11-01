//! `bloomery-serve --model nemotron` — the llama-server-compatible HTTP API on
//! the nemotron engine, serving `nemotron_h` and `nemotron_h_moe` files with
//! one sequence: a Nemotron 3 Nano file (Mamba-2, attention and squared-ReLU
//! MLP layers, dense) or a Nemotron 3.5 Bolt file (Mamba-2, attention and
//! latent MoE layers), the whole model on one card, each opened as its gate
//! opens it (`NemotronHModel::open` as `gate_nemotron_e2e` opens the Nano,
//! `NemotronHMoeModel::open` as `gate_nemotron_h_moe_e2e` opens the Bolt; the
//! step captured in graph mode). The file's architecture picks the body once,
//! at `run`; everything else is one path over both.
//!
//!     bloomery-serve --model nemotron [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--queue-depth Q] [--slot-save-path DIR]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `nemotron` for a file that names none). Sampling is
//! the sampler crate's chain with no repetition penalty, `temperature <= 0`
//! the engine's argmax. Any other flag is refused as an unknown argument.
//!
//! Three facts of the architecture set this seat's shape, and nothing else
//! does:
//!
//! - The body holds one sequence: its attention layers' K and V planes of one
//!   context and its Mamba-2 layers' scan state and conv ring (a Bolt file's
//!   latent MoE layers hold nothing). It has no slot rows, so `--parallel`
//!   above 1 is refused by name before any plan.
//! - The prompt path is the body's batched one (`GpuModel::prefill`, passes
//!   of many rows); one id is a step.
//! - A Mamba-2 state cannot step back. A request keeps the prefix it shares
//!   with what the slot holds only when that prefix is the whole held
//!   sequence (the position the model stands at); any other shared prefix is
//!   kept 0, the slot is reset and the prompt fed again. A rollback to a
//!   position the model does not stand at, a snapshot and a resume are
//!   refused by name, and the seat runs no prompt cache.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule every serving seat takes decides
//! (`Place::choose_untiered`: the body serves no tier card), on one census
//! reading. A placement of more than one card or with a tier card is refused
//! before any load by the plan's own refusal (`PlaceError`).
//!
//! A Bolt file the card cannot hold whole (the whole-card plan refused at the
//! least context a seat asks, [`SEAT_STEPS`]) loads placed: the plan keeps the
//! routed experts the card's room holds resident and puts the rest on the
//! host tier (`Experts::Placed` over `Place::machine_placed`; the host set is
//! read in and locked as the host levers say), the prompt runs in passes of
//! up to the host union's columns through the tier's batch port, and the
//! card's room is the plan's, `BLOOMERY_CARD_BUDGET` included.
//! A file the card holds whole loads whole, as before.
//!
//! A model path that is a directory is a checkpoint folder of a Bolt-family
//! model (Nemotron 3.5 Lightning's ModelOpt NVFP4/FP8 checkpoint), served as
//! stored (`folder::open`, no prepare step): its located header takes the
//! first shard's place in every rule and plan, its tokenizer is the
//! vocabulary and its `chat_template.jinja` the template. A folder has no
//! whole-card open: it always loads placed, the engine reading its card
//! tensors through the folder's parts ([`Model`]) and its host experts where
//! the folder stores them, and a card that holds every expert is the folder's
//! whole-card load. A folder of any other architecture is refused by name.
//!
//! `--ctx` is the one slot's context (`--ctx 0` is refused). The context
//! comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's trained
//! context stops by name, and one the card cannot hold stops with the largest
//! it would hold. Unset, the slot asks the trained context and, where the card
//! cannot hold the stores and the resident bytes there, the largest context
//! under it that it holds.
//!
//! `--embd-model` is taken before this seat parses. The side model is seated
//! before the plan, and the plan's stage card holds its bytes
//! (`beside::reserve_on`, the reserve every seat counts). It loads after the
//! chat model, and the server answers its embeddings routes from it.
//!
//! The records on stderr, in order: `place unset` (word and why), the
//! `embedding_seat` record when one is named, one `ctx` line
//! (`placement::ctx_ask::Chosen::line`, plus the stage card's reserve note
//! when it holds the embedding model), the `plan` record (its `ctx_max` the
//! slot's), the `parallel` line (`rule=slots slots=1 slot_ctx=C total=C
//! from=<word> pass=one`); then the `load` record (architecture, resident
//! bytes, the slot's context, slots, layers, the step graph's nodes and the
//! load's wall), on a placed load whose plan paged routed experts the
//! `nvtier` record (the arena's budget, the paged bytes and its counters),
//! at load and again after each prompt call, and, once the port is bound,
//! the `listening` record (`record::BLOOMERY_SERVE_NEMOTRON`). `--plan` ends
//! the process after the `parallel` line, nothing loaded. An engine error ends
//! the process with the crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget and the start screen. The seat sits behind
//! the `deepseek41` feature: the server surface it binds through (`bind`, the
//! `serve` and `sampler` crates) is scoped to it; it runs no V4.1 code.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::folder::Model;
use bloomery_gpu::host::served::{TierBody, prompt_pass};
use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuError, GpuModel, NemotronHModel, NemotronHMoeModel};
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, slots_of,
};
use bloomery_gpu_gates::generate::{Place, SEAT_STEPS, choose_ctx};
use bloomery_gpu_gates::need;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::{GateError, gpu_census, model_file, ref_model_path};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::Arch;
use model::arch::nemotron_h::place as nemotron_place;
use model::arch::nemotron_h_moe::place::{self as moe_place, Experts, PlanInputs};
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::DeviceId;
use model::placement::{Machine, PlanLevers};
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

use super::folder;

const NAME: &str = "bloomery-serve-nemotron";

const USAGE: &str = "usage: bloomery-serve --model nemotron [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--slot-save-path DIR] \
                     [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget, the start screen, how a
/// structured-output request's fixed runs are fed, and the host set's load a placed
/// Bolt load reads (a whole-card load takes none).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::SCREEN,
    bloomery_levers::SCHEMAJUMP,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the nemotron seat runs no prompt cache";

/// Why a prefix shorter than the held sequence is not kept.
const NO_STEP_BACK: &str =
    "the Mamba-2 state cannot step back: a prefix is kept only where the sequence stands";

/// Where the seat's model is read from, decided once by the path: a directory is a checkpoint
/// folder, anything else a GGUF file. Each is opened on the main thread for its header, which the
/// plan is made from.
enum Weights {
    /// A checkpoint folder: its located header, tokenizer and part source (boxed: the tokenizer
    /// outweighs a header).
    Folder(Box<folder::Opened>),
    /// A GGUF file's header; the engine thread opens the file again.
    File(Split),
}

/// The model's open on the engine thread ([`Weights::hand_over`]).
type OpenModel = Box<dyn FnOnce() -> Result<Model, GateError> + Send>;

impl Weights {
    /// The weights at `path`: the checkpoint folder a directory is (held to this family's
    /// architecture), or the GGUF file.
    fn open(path: &Path) -> Result<Weights, GateError> {
        if path.is_dir() {
            let f = folder::open(path, refset::arch::nemotron_h_moe::ARCH)?;
            return Ok(Weights::Folder(Box::new(f)));
        }
        Ok(Weights::File(model_file::header(path)?))
    }

    /// The header the plan is made from: a split's first shard's, a folder's located header.
    fn header(&self) -> &Split {
        match self {
            Weights::Folder(f) => &f.header,
            Weights::File(header) => header,
        }
    }

    /// Whether the weights load only through a plan: a checkpoint folder has no whole-card open.
    fn is_folder(&self) -> bool {
        matches!(self, Weights::Folder(_))
    }

    /// The tokenizer the vocabulary is built from at `path`, and the model's open on the engine
    /// thread: a folder's header and part source go with it, a file is opened again there.
    fn hand_over(self, path: &Path) -> Result<(Tokenizer, OpenModel), GateError> {
        match self {
            Weights::Folder(f) => {
                let folder::Opened {
                    header,
                    tokenizer,
                    source,
                } = *f;
                let open: OpenModel = Box::new(move || folder::model(header, source));
                Ok((tokenizer, open))
            }
            Weights::File(header) => {
                drop(header);
                let tokenizer = Tokenizer::from_gguf(path)?;
                let file = path.to_path_buf();
                let open: OpenModel = Box::new(move || -> Result<Model, GateError> {
                    Ok(model_file::header(&file)?.into())
                });
                Ok((tokenizer, open))
            }
        }
    }
}

/// What the engine thread opens the seat with.
struct OpenSeat {
    /// The planned card, by name and device: the open takes the device the
    /// plan resolved.
    card: String,
    device: Option<DeviceId>,
    /// The model's open ([`Weights::hand_over`]).
    model: OpenModel,
    /// The slot's positions the K/V planes are sized for.
    ctx: usize,
}

/// What a placed Bolt load is opened from on the engine thread: the plan's
/// inputs, machine and levers, which make the plan again at the slot's
/// context (the plan borrows them, so the seat's own is dropped before the
/// spawn), and the host set's load.
struct PlacedSeat {
    inputs: PlanInputs,
    machine: Machine,
    levers: PlanLevers,
    host: HostCfg,
}

impl PlacedSeat {
    /// The placed load of `file`, a split or a checkpoint folder, at `ctx` positions (the plan
    /// the seat printed, made again here).
    fn open(self, file: Model, ctx: usize) -> Result<NemotronHMoeModel, GpuError> {
        let plan = self
            .inputs
            .plan_with(
                &self.machine,
                u64::try_from(ctx).unwrap_or(u64::MAX),
                &self.levers,
                Experts::Placed,
            )
            .map_err(|e| GpuError::plan("the nemotron seat's placed load", e))?;
        NemotronHMoeModel::open_placed(file, &plan, 0, self.host)
    }
}

/// The model on the engine thread: either family's body, the seat's one path
/// over both, with the body's prompt path.
struct Nemotron<B: ChainBody> {
    m: GpuModel<B>,
    prefill: fn(&mut GpuModel<B>, &[u32]) -> Result<u32, GpuError>,
}

/// The Nano's prompt path: several ids run the batched
/// [`GpuModel::prefill`] (passes of many rows); one id is a step.
fn prefill_nano(m: &mut NemotronHModel, ids: &[u32]) -> Result<u32, GpuError> {
    if ids.len() > 1 {
        m.prefill(ids)
    } else {
        m.step(ids)
    }
}

/// The Bolt's prompt path, the Nano's split: several ids the batched
/// [`GpuModel::prefill`], one id a step, as one prompt call of the host tier
/// ([`prompt_pass`]: a placed load's passes run through the tier's batch
/// port; a whole-card load has no tier and the call adds nothing).
fn prefill_bolt(m: &mut NemotronHMoeModel, ids: &[u32]) -> Result<u32, GpuError> {
    let next = prompt_pass(m, |m| {
        if ids.len() > 1 {
            m.prefill(ids)
        } else {
            m.step(ids)
        }
    })?;
    print_nvtier(m)?;
    Ok(next)
}

/// The load's cumulative `nvtier` record on stderr, on a placed load whose
/// plan paged routed experts; nothing on any other load.
fn print_nvtier(m: &NemotronHMoeModel) -> Result<(), GpuError> {
    record::host_tier::eprint_nvtier(m.body(NAME)?.placed().and_then(|p| p.hybrid().nvme_tier()));
    Ok(())
}

/// The split a whole-card load opens: a checkpoint folder has no whole-card open, its card
/// tensors come through a plan's load, which the seat always makes for one.
fn whole_card_file(model: Model) -> Result<Split, GpuError> {
    match model {
        Model::Split(file) => Ok(file),
        Model::Folder(_) => Err(GpuError::shape(
            "the nemotron seat's whole-card load",
            "a checkpoint folder loads through a plan",
        )),
    }
}

impl<B: ChainBody> Nemotron<B> {
    /// The model `a.model` opens (the file, or the folder the seat opened), built by
    /// `open` (the family's `GpuModel::open` on the planned card, or a placed load, which opens
    /// the card its plan names) and carrying its prompt path `prefill`,
    /// in graph mode with its step captured, the `load` record on stderr
    /// naming the body's architecture.
    fn open(
        a: OpenSeat,
        open: impl FnOnce(Model) -> Result<GpuModel<B>, GpuError>,
        prefill: fn(&mut GpuModel<B>, &[u32]) -> Result<u32, GpuError>,
    ) -> Result<Nemotron<B>, GateError> {
        let t = Instant::now();
        let model = (a.model)()?;
        let opened = t.elapsed();
        let mut m = open(model)?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let ctx = m.ctx_max();
        let total = t.elapsed();
        Record::new(&record::LOAD_NEMOTRON)
            .w("arch", B::arch().name())
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", ctx)
            .u("slots", 1)
            .u("layers", m.layers().len())
            .u("graph_nodes", nodes)
            .f("load_s", total.as_secs_f64())
            .eprint();
        record::eprint_load_phases(&m, total, opened);
        Ok(Nemotron { m, prefill })
    }
}

impl<B: ChainBody + 'static> Seat for Nemotron<B> {
    fn pos(&self) -> usize {
        self.m.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.m.ctx_max()
    }

    /// The body's prompt path, the `prefill` its `open` chose.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok((self.prefill)(&mut self.m, ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.m.step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.m.logits_into(row)?)
    }

    /// The model's reset ([`bloomery_gpu::GpuModel::reset`]): the scan state
    /// and the conv rings back to zero, the position to 0.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    /// Never asked for a position other than the one the model stands at
    /// ([`Seat::keep`] grants no other, and the engine answers a cut to the
    /// standing position itself); any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Err(format!(
            "a rollback to position {pos} from {}: {NO_STEP_BACK}",
            self.m.pos()
        )
        .into())
    }

    /// `n` when it is the whole held sequence, else nothing: the caller then
    /// resets the slot and feeds the prompt again.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        if n == self.pos() {
            (n, None)
        } else {
            (0, Some(NO_STEP_BACK.to_owned()))
        }
    }

    /// Nowhere: no prompt call is cut for a prefix a later request keeps.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bloomery_gpu_gates::bind::check_seat`]).
pub fn check(
    args: &[String],
    read_levers: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<SeatFlags>, GateError> {
    bloomery_gpu_gates::bind::check_seat(read_levers.then_some(ACTS_ON), args, |args, _| {
        SeatFlags::parse_one_sequence(args, USAGE, "nemotron")
    })
}

/// Loads the model, and the embedding model `side` names beside it, and serves until the listener
/// or the engine fails; `Ok` carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_NEMOTRON, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let (a, levers) = check(args, true)?.run();

    arm_screen(levers.screen());
    let path = ref_model_path()?;
    // A directory is a checkpoint folder, opened once here: its located header plans as a split's
    // first shard does, and the header and part source go to the engine thread.
    let weights = Weights::open(&path)?;
    let split = weights.header();
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if !matches!(arch, Arch::NemotronH | Arch::NemotronHMoe) {
        return Err(format!(
            "{} is a {} file; the nemotron seat serves nemotron_h and nemotron_h_moe files",
            path.display(),
            arch.name()
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "nemotron",
    )?;
    // The family's `place::read`, the first of the two branches the
    // architecture takes (the other the open, at the spawn below).
    let inputs = match arch {
        Arch::NemotronH => nemotron_place::read(split)?,
        Arch::NemotronHMoe => moe_place::read(split)?,
        _ => unreachable!("the arch check above refused every other architecture"),
    };
    let plan_levers = PlanLevers::from_levers(&levers)?;
    // The one slot, and what named it (`from` on the `parallel` line); the
    // start's note that `--ctx` is one request's context is left out, since
    // `--parallel` has no larger count to offer.
    let (slots, from) = slots_of(a.parallel, a.ctx.is_some(), DEFAULT_SLOTS)?;
    // The common unset rule, once, on one census reading.
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, "nemotron")?;
    chosen.record().eprint();
    let place = chosen.place;
    // The side model's seat before the plan, which holds its bytes when it sits on the stage card.
    let side = side.read()?;
    let side_seat = side.as_ref().map(|s| s.seat(place)).transpose()?;
    let held = Beside {
        image: None,
        embedding: side_seat.as_ref().and_then(|s| s.reserve),
    };
    let layers = inputs.model.layers;
    // A Bolt file the card cannot hold whole at the least context a seat asks
    // loads placed: the plan keeps the experts the card's room holds and the
    // host tier serves the rest. A checkpoint folder always loads placed (it has no
    // whole-card open). Everything else loads whole, as before. The
    // probe and the placed machine both carry the side seat's reserve.
    let mut whole = place.machine_untiered()?(layers);
    beside::reserve_on(&mut whole, held);
    let placed = weights.is_folder()
        || (arch == Arch::NemotronHMoe
            && inputs.plan(&whole, SEAT_STEPS.floor, &plan_levers).is_err());
    let (machine, experts) = if placed {
        let mut machine = place.machine_placed()?(layers);
        beside::reserve_on(&mut machine, held);
        (machine, Experts::Placed)
    } else {
        (whole, Experts::Card)
    };
    // The context, then the plan around it (`choose_ctx`, every seat's): a
    // tier card, a second card, a context the card cannot hold and a budget
    // that binds are refused here by name, before any load.
    let asked = choose_ctx(split, (a.ctx, slots), (None, None), |c| {
        probe_of(inputs.plan_with(&machine, c, &plan_levers, experts))
            .map_err(|e| beside::refusal(held, format!("--place {}: {e}", place.name())))
    })?;
    eprintln!("{}{}", asked.line(), beside::note(&machine.cards[0]));
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), &machine, &plan).eprint();
    need::print(&plan, 0, 0, true)?;
    parallel_line(slots, slot_ctx, from, Pass::One);
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        bloomery_gpu_gates::start::plan_exit();
    }
    let props = EngineProps {
        model: Some(model_props(split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = &machine.cards[0];
    let device = card.name.clone();
    let (card_name, card_device) = (card.name.clone(), card.device);
    let host = levers.host();
    drop(plan);
    // A folder's header goes to the engine thread with its parts; a file's is reopened there.
    let (tokenizer, model) = weights.hand_over(&path)?;
    let open = OpenSeat {
        card: card_name,
        device: card_device,
        model,
        ctx: slot_ctx,
    };
    let vocab = Arc::new(Vocab::new(tokenizer)?);
    // The family's `GpuModel::open` and prompt path, the engine thread's body.
    let engine = match arch {
        Arch::NemotronH => SeatEngine::spawn(
            move || {
                let (card, device, ctx) = (open.card.clone(), open.device, open.ctx);
                Nemotron::open(
                    open,
                    |file| {
                        NemotronHModel::open(
                            Gpu::open_card_after_hook(&card, device)?,
                            whole_card_file(file)?,
                            ctx,
                        )
                    },
                    prefill_nano,
                )
            },
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        Arch::NemotronHMoe if placed => {
            let seat = PlacedSeat {
                inputs,
                machine,
                levers: plan_levers,
                host,
            };
            SeatEngine::spawn(
                move || {
                    let ctx = open.ctx;
                    let n = Nemotron::open(open, |file| seat.open(file, ctx), prefill_bolt)?;
                    print_nvtier(&n.m)?;
                    Ok(n)
                },
                slot_ctx,
                vocab,
                device,
                props,
                0,
            )?
        }
        Arch::NemotronHMoe => SeatEngine::spawn(
            move || {
                let (card, device, ctx) = (open.card.clone(), open.device, open.ctx);
                Nemotron::open(
                    open,
                    |file| {
                        NemotronHMoeModel::open(
                            Gpu::open_card_after_hook(&card, device)?,
                            whole_card_file(file)?,
                            ctx,
                        )
                    },
                    prefill_bolt,
                )
            },
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        _ => unreachable!("the arch check above refused every other architecture"),
    };
    let side = side_seat::Side::start(side.zip(side_seat))?;
    let server = a.bind(engine, chat, &path, slots, &levers, side)?;
    Record::new(&record::LISTENING_NEMOTRON)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
