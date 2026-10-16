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
//!                    [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]
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
//! - The prompt path is the body's: the Nano's is batched
//!   (`GpuModel::prefill`, passes of many rows); Bolt's has no prompt batch,
//!   one step an id (`GpuModel::step` over the ids), so its prompt rate is its
//!   decode rate.
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
//! `--ctx` is the one slot's context (`--ctx 0` is refused). The context
//! comes first and the plan fills the machine around it
//! (`generate::choose_ctx`, every seat's): a set value past the file's trained
//! context stops by name, and one the card cannot hold stops with the largest
//! it would hold. Unset, the slot asks 262,144 positions or the trained
//! context where that is less, halved while the card cannot hold the stores
//! and the resident bytes.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (`placement::ctx_ask::Chosen::line`), the `plan` record (its
//! `ctx_max` the slot's), the `parallel` line (`rule=slots slots=1
//! slot_ctx=C total=C from=<word> pass=one`); then the `load` record
//! (architecture, resident bytes, the slot's context, slots, layers, the step
//! graph's nodes and the load's wall) and, once the port is bound, the
//! `listening` record (`record::BLOOMERY_SERVE_NEMOTRON`). `--plan` ends the
//! process after the `parallel` line, nothing loaded. An engine error ends the
//! process with the crash block and exit code 70, as every seat's.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget and the start screen. The seat sits behind
//! the `deepseek41` feature: the server surface it binds through (`bind`, the
//! `serve` and `sampler` crates) is scoped to it; it runs no V4.1 code.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{Gpu, GpuError, GpuModel, NemotronHModel, NemotronHMoeModel};
use bloomery_gpu_gates::bind::{
    ChatSurface, Pass, Seat, SeatEngine, SeatFlags, Vocab, arm_screen, model_props, note_line,
    parallel_line, placement_of, slots_of,
};
use bloomery_gpu_gates::generate::{Place, choose_ctx};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use gguf::Split;
use model::arch::Arch;
use model::arch::nemotron_h::place as nemotron_place;
use model::arch::nemotron_h_moe::place as moe_place;
use model::placement::PlanLevers;
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::DeviceId;
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-nemotron";

const USAGE: &str = "usage: bloomery-serve --model nemotron [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--queue-depth Q] [--api-key KEY] \
                     [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the start screen.
pub const ACTS_ON: &[&str] = &[bloomery_levers::CARD_BUDGET, bloomery_levers::SCREEN];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the nemotron seat runs no prompt cache";

/// Why a prefix shorter than the held sequence is not kept.
const NO_STEP_BACK: &str =
    "the Mamba-2 state cannot step back: a prefix is kept only where the sequence stands";

/// What the engine thread opens the seat with.
struct OpenSeat {
    /// The planned card, by name and device: the open takes the device the
    /// plan resolved.
    card: String,
    device: Option<DeviceId>,
    path: PathBuf,
    /// The slot's positions the K/V planes are sized for.
    ctx: usize,
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

/// A body with no prompt batch (Bolt's): one step an id, the argmax after the
/// last, so the prompt rate is the decode rate.
fn prefill_steps<B: ChainBody>(m: &mut GpuModel<B>, ids: &[u32]) -> Result<u32, GpuError> {
    m.step(ids)
}

impl<B: ChainBody> Nemotron<B> {
    /// The model of the file at `a.path` on the planned card, built by `open`
    /// (the family's `GpuModel::open`) and carrying its prompt path `prefill`,
    /// in graph mode with its step captured, the `load` record on stderr
    /// naming the body's architecture.
    fn open(
        a: OpenSeat,
        open: fn(Gpu, Split, usize) -> Result<GpuModel<B>, GpuError>,
        prefill: fn(&mut GpuModel<B>, &[u32]) -> Result<u32, GpuError>,
    ) -> Result<Nemotron<B>, GateError> {
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let opened = t.elapsed();
        let gpu = Gpu::open_card(&a.card, a.device)?;
        let mut m = open(gpu, file, a.ctx)?;
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

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(NAME, record::BLOOMERY_SERVE_NEMOTRON);
    arm_screen(levers.screen());
    let a = SeatFlags::parse(args, USAGE, |_, _| Ok(false))?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
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
    if let Some(n) = a.parallel.filter(|&n| n > 1) {
        return Err(format!(
            "--parallel {n}: the {} body holds one sequence and no resident slots; the \
             seat serves --parallel 1",
            arch.name()
        )
        .into());
    }
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.as_deref(),
        "nemotron",
    )?;
    // The family's `place::read`, the first of the two branches the
    // architecture takes (the other the open, at the spawn below).
    let inputs = match arch {
        Arch::NemotronH => nemotron_place::read(&split)?,
        Arch::NemotronHMoe => moe_place::read(&split)?,
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
    let machine = place.machine_untiered()?(inputs.model.layers);
    // The context, then the plan around it (`choose_ctx`, every seat's): a
    // tier card, a second card, a context the card cannot hold and a budget
    // that binds are refused here by name, before any load.
    let asked = choose_ctx(&split, (a.ctx, slots), (None, None), |c| {
        probe_of(inputs.plan(&machine, c, &plan_levers))
            .map_err(|e| format!("--place {}: {e}", place.name()))
    })?;
    eprintln!("{}", asked.line());
    let slot_ctx = usize::try_from(asked.slot_ctx)?;
    let plan = asked.plan;
    record::plan(place.name(), &machine, &plan).eprint();
    parallel_line(slots, slot_ctx, from, Pass::One);
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        std::process::exit(0);
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = &machine.cards[0];
    let open = OpenSeat {
        card: card.name.clone(),
        device: card.device,
        path: path.clone(),
        ctx: slot_ctx,
    };
    let device = card.name.clone();
    drop(plan);
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    // The family's `GpuModel::open` and prompt path, the engine thread's body.
    let engine = match arch {
        Arch::NemotronH => SeatEngine::spawn(
            move || Nemotron::open(open, NemotronHModel::open, prefill_nano),
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        Arch::NemotronHMoe => SeatEngine::spawn(
            move || Nemotron::open(open, NemotronHMoeModel::open, prefill_steps),
            slot_ctx,
            vocab,
            device,
            props,
            0,
        )?,
        _ => unreachable!("the arch check above refused every other architecture"),
    };
    let server = a.bind(engine, chat, &path, slots)?;
    Record::new(&record::LISTENING_NEMOTRON)
        .w("place", place.name())
        .u("ctx", slot_ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
