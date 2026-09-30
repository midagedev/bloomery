//! The GLM-5.3-Flash seat — the llama-server-compatible HTTP API on the
//! glm5next engine, opened and stepped as `generate_glm5next` opens and steps
//! the model. `bloomery-serve --model glm` is one call of [`run`], which
//! takes the process's arguments (`--model` already taken out by the
//! one-binary server).
//!
//!     [--host 127.0.0.1] [--port 8080] [--place a|gate] [--ctx C]
//!     [--alias NAME] [--chat-template-file PATH] [--prefill batch|steps]
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! `tokenizer.chat_template` the chat template (`--chat-template-file`
//! replaces it, as the qwen38 seat takes it) and `general.name` the default
//! alias. Tool calls are the template's: the server reads their markup out of
//! the generation (`serve::glmxml`, which this file's template teaches), so
//! the seat adds nothing there. The `plan` record, then the `load` and
//! `capture` lines of the open, go to stderr as `generate_glm5next` prints
//! them, then the `listening on http://<addr>` line once the model is loaded
//! and the port is bound (`--port 0` binds a free one). Sampling is the
//! sampler crate's chain with no repetition penalty; `temperature <= 0` is
//! the engine's argmax, the
//! ids `generate_glm5next --tokens <the prompt's ids>` prints.
//!
//! `--place` is `a` (the serving plan, `workstation::plan_a`, on the A6000 —
//! the default, as both other serving seats) or `gate` (the gate card's). The
//! positions the server serves are the stores the load sized (`--ctx`,
//! default 2048 as the CLI's, refused past `place::ORACLE_POSITIONS`):
//! `/props`' `n_ctx` is that number, a prompt that long is a 400 before it
//! reaches the engine, and generation stops there with `truncated`.
//!
//! The keep rule: each KDA layer holds one recurrent state, and its history
//! only in the checkpoints a prompt call takes (its start, every 512th
//! position of the whole context, its end — `CHECKPOINT_EVERY`), so a cut
//! keeps every fed position, the empty model, or the checkpoint at or below
//! it — never a position between two checkpoints. A request that shares less
//! than that with the slot is a `cache reuse` record with the rule. The seat
//! marks no user-start token, so the server asks nowhere to cut a prompt
//! call: the checkpoint spacing owns where a cut can land.
//!
//! The server's host prompt cache is off for this seat (its budget 0): a
//! checkpoint lives in the model's own host slots, not a value a cache could
//! hold, and the latent layers' cache rows have no copy either, so
//! `snapshot` and `resume` refuse by name and slot save/restore answers the
//! server's own 501.
//!
//! `pass` and `pass_rows` are the engine's defaults (no draft): one step a
//! pass; a shared MTP window would land there.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70. The levers this seat acts on ([`ACTS_ON`]) are
//! parsed once, at `main` (`bloomery_levers::at_main`), which refuses by name
//! a lever set outside them; `--levers` prints them with this process's
//! values and exits. The stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::arch::glm5next::GlmCfg;
use app::{Loaded, OpenLog, Session};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
};
use bloomery_gpu_gates::generate::mode_name;
use bloomery_gpu_gates::record::{self, Kind, Record};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode};
use gguf::Split;
use model::arch::glm5next::place::PlanInputs;
use model::placement::workstation;
use model::placement::{Machine, PlanLevers};
use runtime::seqstate::Why;
use runtime::{Target, Want};
use serve::{CacheNote, EngineProps, FATAL_LINGER, Saved, ServeError, Server, ServerConfig};
use tokenizer::Tokenizer;

/// The seat's name, as its records and errors print it.
const WHAT: &str = "bloomery-serve-glm";

const USAGE: &str = "usage: bloomery-serve [--model glm] [--host H] [--port P] [--place a|gate] \
                     [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--prefill batch|steps]";

/// The positions the stores are sized for when `--ctx` names none:
/// `generate_glm5next`'s default.
const CTX: usize = 2048;

/// Why no state is saved or put back: the recurrent state lives only in the
/// model's own checkpoint slots and the latent cache rows have no copy.
const SNAPSHOT_WHY: &str = "the KDA recurrent state lives only in the model's own checkpoint \
                            slots and the latent layers' cache rows have no copy: no sequence \
                            state is a value the cache could hold";

/// What this seat prints, all on stderr (its `--records-schema`): the `plan`,
/// `load` and `capture` lines `generate_glm5next` prints, the host set's
/// records of a placed load, the listening line, and the reuse records the
/// checkpoint rule answers.
static KINDS: &[&Kind] = &[
    &record::PLAN,
    &record::LOAD_GENERATOR,
    &record::HOST_POPULATE,
    &record::HOST_POPULATE_OFF,
    &record::HOST_LOCK,
    &record::CAPTURE,
    &record::LISTENING_GLM,
    &record::CACHE_REUSE,
];

/// The levers this seat acts on: those `generate_glm5next` reads for its load
/// (`BLOOMERY_ROUTE_TRACE` left out — its trace is one run's instrument,
/// which a server that serves many prompts does not wire, and a lever set
/// outside this list is refused by name at `main`), with
/// `BLOOMERY_PIN_MAIN` for the engine thread's cpu slot, as every serving
/// seat reads it.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
];

/// Where `--place` puts the plan's one card, as `generate_glm5next` takes it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GlmPlace {
    /// The serving plan (`workstation::plan_a`), on the A6000. The default:
    /// a server belongs on the serving card.
    A,
    /// The gate card's plan (`workstation::plan_gate`), on the 3090.
    Gate,
}

impl GlmPlace {
    fn parse(v: &str) -> Result<GlmPlace, GateError> {
        match v {
            "a" => Ok(GlmPlace::A),
            "gate" => Ok(GlmPlace::Gate),
            other => Err(format!("--place is a or gate, not {other}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            GlmPlace::A => "a",
            GlmPlace::Gate => "gate",
        }
    }

    fn machine(self) -> fn(usize) -> Machine {
        match self {
            GlmPlace::A => workstation::plan_a,
            GlmPlace::Gate => workstation::plan_gate,
        }
    }
}

struct Args {
    host: String,
    port: u16,
    place: GlmPlace,
    ctx: usize,
    alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
    prefill: PrefillMode,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: GlmPlace::A,
        ctx: CTX,
        alias: None,
        template_file: None,
        prefill: PrefillMode::Batch,
    };
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = v.parse()?,
            "--place" => a.place = GlmPlace::parse(v)?,
            "--ctx" => a.ctx = v.parse()?,
            "--alias" => a.alias = Some(v.to_owned()),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            "--prefill" => {
                a.prefill = PrefillMode::from_name(v)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {v}"))?;
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == 0 {
        return Err("--ctx 0: the stores hold no position".into());
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails; `Ok`
/// carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(WHAT, KINDS);
    let a = parse_args(args)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let path = ref_model_path()?;
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let template = match &a.template_file {
        Some(file) => std::fs::read_to_string(file)
            .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))?,
        None => inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned(),
    };
    let name = inv
        .value("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("glm-5.3-flash")
        .to_owned();
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::read(&split)?;
    let model = model_props(&split, &inputs.model);
    let machine = (a.place.machine())(inputs.model.layers);
    let plan = inputs.plan(&machine, u64::try_from(a.ctx)?, &plan_levers)?;
    record::plan(a.place.name(), &machine, &plan).eprint();
    let gpu = nvidia_smi_index(&machine.cards[0].name)
        .map(|i| format!("GPU{i}"))
        .and_then(|g| placement_props(&plan, &[g]));
    if let Err(e) = &gpu {
        eprintln!("{WHAT}: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        ctx: a.ctx,
        cfg: GlmCfg {
            place: plan_levers,
            host: levers.host(),
            prefill: a.prefill,
        },
        pin_main: levers.pin_main(),
        path: path.clone(),
    };
    // The prompt cache is off (its budget 0): the seat's snapshot refuses, and
    // slot save/restore is the server's own 501.
    let engine = SeatEngine::spawn(
        move || Glm::open(open),
        a.ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        0,
    )?;

    let config = ServerConfig {
        model_alias: a.alias.unwrap_or(name),
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
    };
    let server = Server::bind((a.host.as_str(), a.port), Box::new(engine), config)?;
    Record::new(&record::LISTENING_GLM)
        .w("place", a.place.name())
        .u("ctx", a.ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: GlmPlace,
    ctx: usize,
    cfg: GlmCfg,
    pin_main: bool,
    path: PathBuf,
}

/// The GLM session on the engine thread, and the positions its stores were
/// sized for.
struct Glm {
    s: Session<Body>,
    ctx: usize,
}

impl Glm {
    /// The session by `a.place` (the `plan` was printed before the engine
    /// thread started; the `load` and `capture` lines go to stderr as
    /// `generate_glm5next` prints them), on the calling thread, pinned to the
    /// dispatcher's cpu slot when asked.
    fn open(a: SeatArgs) -> Result<Glm, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut log = Log {
            top_k: 0,
            place: a.place.name(),
            prefill: a.cfg.prefill,
            ctx: a.ctx,
            pin_main: a.pin_main,
            pinned,
            t,
        };
        let args = app::OpenArgs {
            place: a.place.name(),
            machine: a.place.machine(),
            ctx: a.ctx,
            mode: StepMode::Graph,
            cfg: a.cfg,
        };
        let loaded = Loaded::<Body>::open(file, args, &mut log)?
            .ok_or_else(|| format!("{WHAT}: the open planned nothing"))?;
        let s = loaded.ready(&mut log)?;
        Ok(Glm { s, ctx: a.ctx })
    }
}

/// What the open prints: the plan's top_k, the load, the capture.
struct Log {
    /// The file's indexer top-k, which the plan read.
    top_k: usize,
    place: &'static str,
    prefill: PrefillMode,
    ctx: usize,
    pin_main: bool,
    pinned: bool,
    t: Instant,
}

impl OpenLog<Body> for Log {
    /// The plan was printed before the engine thread started.
    fn plan(
        &mut self,
        _place: &'static str,
        inputs: &PlanInputs,
        _machine: &Machine,
        _plan: &model::placement::Plan<'_>,
    ) -> Result<bool, app::SessionError> {
        self.top_k = inputs.hp.indexer.top_k;
        Ok(true)
    }

    fn load(&mut self, m: &Glm5nextModel) -> Result<(), app::SessionError> {
        let b = m.body(WHAT)?;
        Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("layers", b.kinds().len())
            .u("top_k", self.top_k)
            .w("shadow", "none")
            .u("shadow_bytes", 0)
            .u("unified_addressing", 0)
            .w("prefill", self.prefill.name())
            .w("mode", mode_name(StepMode::Graph))
            .w("place", self.place)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", self.t.elapsed().as_secs_f64())
            .eprint();
        if let Some(h) = b.hybrid().residency() {
            for r in record::host_residency(h) {
                r.eprint();
            }
        }
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), app::SessionError> {
        Record::new(&record::CAPTURE)
            .u("graph_nodes", nodes)
            .eprint();
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), app::SessionError> {
        Ok(())
    }
}

impl Seat for Glm {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The body's feed ([`bloomery_gpu_glm5next::feed`]): the batched prompt
    /// call or one step a position, as `--prefill` says, each call taking the
    /// checkpoints its marks name.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(self.s.prompt(ids, Want::Argmax)?.argmax())
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.step(last, Want::Argmax)?.argmax())
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.reset()?)
    }

    /// The body's rule ([`Body::kept`]): every fed position, the empty model,
    /// or the checkpoint at or below the cut, with the rule that kept less.
    /// A query past u32 positions is the query for every position.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let asked = u32::try_from(n).unwrap_or(u32::MAX);
        let k = self.s.kept(asked);
        let why = match k.why {
            Why::Current | Why::Empty => None,
            _ => Some(k.to_string()),
        };
        (k.at as usize, why)
    }

    /// The checkpoint the cut keeps, taken back by the body's own rollback;
    /// any other position is refused by name, the body's own message.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// Nowhere: this seat marks no user start, and a cut keeps the
    /// checkpoints the body's own rule takes — a call cut at a mark would add
    /// one the checkpoint spacing did not choose.
    fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// A GLM sequence state is not a value a cache could hold
    /// ([`SNAPSHOT_WHY`]). The prompt cache is off, so this is never asked;
    /// a caller that asks is refused by name.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot at position {}: {SNAPSHOT_WHY}", self.s.pos()).into())
    }

    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        Err(format!(
            "a resume of a state of {} positions: {SNAPSHOT_WHY}",
            state.n_tokens()
        )
        .into())
    }

    /// Only `Reuse` arrives (the cache is off and no call is cut); the other
    /// notes belong to a caching engine and print as they come.
    fn note(note: &CacheNote) {
        if let CacheNote::Reuse {
            common,
            ask,
            kept,
            held,
            reason,
        } = note
        {
            Record::new(&record::CACHE_REUSE)
                .u("common", common)
                .u("ask", ask)
                .u("kept", kept)
                .u("held", held)
                .w("reason", reason.as_deref().unwrap_or("unstated"))
                .eprint();
            return;
        }
        eprintln!("{WHAT}: {note}");
    }
}
