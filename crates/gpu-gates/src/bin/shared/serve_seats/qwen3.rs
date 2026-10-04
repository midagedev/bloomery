//! `bloomery-serve --model qwen3` — the llama-server-compatible HTTP API on
//! the single-card Qwen engines: a qwen3moe file (Qwen3-30B-A3B) or a
//! qwen35moe file (Qwen3.6-35B-A3B), the whole model on device 0, under
//! `--place` placed by its plan on the card the word names with the routed
//! experts the card's budget does not hold on the host tier.
//!
//!     bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--place W]
//!                    [--parallel N] [--queue-depth Q]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` is `--ctx` under llama-server's spelling, C defaults
//! to the file's trained context capped to what the card's free bytes fit (a
//! whole-card load; never under 4096, a multiple of 1024, one stderr line
//! when the cap binds), or — for a placed load, under `--place` or the plan
//! a run without it falls to — to the largest context whose placed plan
//! keeps the card experts the 4096-floor's plan keeps, one stderr line
//! naming it. 4096 for a file that states none. The model is opened as
//! `generate_qwen3moe` opens it — qwen3moe through `Qwen3moeModel::open` at
//! its levers' options, qwen35moe through `Qwen35moeModel::open` with the
//! tensor-core decode flash and the levers' ubatch — in graph mode with the
//! step and every pass captured, so a request's ids are the CLI's: the
//! server feeds the prompt less its last id as one prompt call — both bodies
//! by the seat's schedule (the ubatch walk from nine rows on — qwen3moe's
//! GEMM, qwen35moe's wide, no unit under the gemv arm's cut — passes below)
//! — then steps the last, which `generate_qwen3moe --last-step` does too.
//! Its first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template and `general.name` the alias. Two slots by default
//! (`--parallel`), one only at `--parallel 1`; sampling is the sampler
//! crate's chain with no repetition penalty, `temperature <= 0` the
//! engine's argmax.
//!
//! `--place W` opens the model as `generate_qwen3moe --place W` does
//! (`qwen3moe_place`): `W` a placement word of `generate::Place` (`a`,
//! `gate`, or a card list of one card; a tier card is refused by name), the
//! plan's `plan` record on stderr before the `load` record, the card budget
//! `BLOOMERY_CARD_BUDGET`'s and the host set's load the host levers' (each
//! of them refused by name without `--place`). A
//! plan with no host expert loads the whole model on the plan's card; any
//! other runs the step graph through the host tier and every prompt as
//! eager passes of up to eight ids, no pass captured.
//!
//! With `--place` unset the model opens as `generate_qwen3moe` opens it:
//! the whole file on device 0 while that fits the card's free bytes, else
//! the placed plan on `a`'s card, its `plan` record naming why
//! (`whole_does_not_fit`).
//!
//! A request keeps the longest prefix it shares with what the slot holds:
//! the session over the model answers the server's keep queries and cuts
//! (`app::Keep` — a qwen3moe file's [`Body`] grants every held position,
//! the caches being per-position; a qwen35moe file's [`Body35`] grants the
//! nearest checkpoint a marked prompt call took at or below the ask, every
//! 512 positions and each call's end, and its prompt calls take those
//! checkpoints), so the resend of a conversation keeps everything up to
//! where it diverges and a request that extends the held sequence keeps all
//! of it. The prompt call runs the ubatch walk from
//! `app::arch::qwen3moe::GEMM_FROM` rows on and passes below, so the rows a
//! kept prefix leaves behind are a whole fresh run's. The seat runs no
//! prompt cache: a save and a resume are refused by name. The
//! `load` record (architecture, resident bytes, context, layers, ubatch,
//! the step graph's nodes, the load's wall) then the `listening` record go
//! to stderr (`record::BLOOMERY_SERVE_QWEN3`). An engine error ends the
//! process with the crash block and exit code 70, as every seat's.
//!
//! `--parallel N` (`-np N`, default 2) serves N slots that take the one
//! model in turns (`serve::SwapEngine`); the default's second slot costs a
//! lone request nothing, the turns acting only on a second arrival, and
//! `--parallel 1` keeps the plain engine: a request that arrives while
//! another decodes preempts it at the next step, the live requests then
//! take turns of `serve::QUANTUM` tokens, and a preempted request comes
//! back by the re-prefill fallback — the engine reset to position 0 and its
//! held ids fed again before it steps (`serve::Park::Ids`; the seat holds
//! no snapshot to park, and a kept prefix does not survive another slot's
//! rows over the cache), the ids it feeds again counted in `/metrics`.
//! `--queue-depth Q` bounds the requests that wait for a slot; `--park-ram`
//! is refused by name: the park holds ids, no state, so it reads no budget.
//!
//! The seat sits behind the `deepseek41` feature, the server surface's
//! scope (`bind`, the `serve` and `sampler` crates); it runs no V4.1 code;
//! a placed load acts on the plan's and the host tier's levers
//! (`BLOOMERY_QWEN3_UBATCH` is read where the load sizes its arena).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[path = "../qwen3moe_place.rs"]
mod q3place;
use app::arch::qwen3moe::GEMM_FROM;
use q3place::PlaceQ3;

use bloomery_gpu::Qwen3moeModel;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::arch::qwen3moe::{Body, Body35, Open35, Qwen35moeModel};
use bloomery_gpu::model::{ChainBody, GpuModel, StepMode};
use bloomery_gpu_gates::bind::{Seat, SeatEngine, Vocab, sampler_factory};
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_levers::Levers;
use gguf::Split;
use model::arch::Arch;
use model::placement::PlanLevers;
use runtime::Target as _;
use serve::{
    CacheNote, Engine, EngineProps, FATAL_LINGER, Park, Saved, ServeError, Server, ServerConfig,
    SlotConfig, SwapEngine,
};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-qwen3";

const USAGE: &str = "usage: bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--place W] [--parallel N] \
                     [--queue-depth Q]";

/// The context unless `--ctx` says: `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// The `--ctx` the seat takes when the flag is unset. A whole-card load
/// takes the file's trained context capped to what the card's free bytes
/// fit — never under [`CTX`], a multiple of 1024, one line on stderr when
/// the cap binds; a file that states no trained context keeps [`CTX`]. A
/// placed load — under `--place`, or the plan a run without it falls to when
/// the whole file does not fit the card's free bytes — searches the same
/// shape over the plan's own expert split: the largest context whose plan
/// keeps the card experts the [`CTX`]-floor's plan keeps, so the solver's
/// context-for-experts trade never sits below the floor's split. One line
/// names the answer, at the floor or past it.
fn default_ctx(
    split: &Split,
    arch: Arch,
    place: Option<Place>,
    levers: &Levers,
) -> Result<usize, GateError> {
    let Some(trained) = q3place::trained_ctx(split) else {
        return Ok(CTX);
    };
    let plan_levers = PlanLevers::from_levers(levers)?;
    let placed_default = |place: Place| -> Result<usize, GateError> {
        let searched = match arch {
            Arch::Qwen3moe => q3place::placed_ctx_qwen3(split, place, CTX, &plan_levers)?,
            Arch::Qwen35moe => {
                let o = Open35 {
                    ctx: CTX,
                    mma: true,
                    ubatch: ubatch_size()?,
                };
                q3place::placed_ctx_qwen35(split, &o, place, CTX, &plan_levers)?
            }
            _ => None,
        };
        match searched {
            // The plan holds the trained context with the floor's experts:
            // the trade bound nothing, as the whole load's rule.
            Some(ctx) if ctx >= trained => Ok(ctx),
            Some(ctx) if ctx > CTX => {
                eprintln!(
                    "{NAME}: --ctx defaults to {ctx} of the file's {trained} trained positions \
                     (the placed plan keeps its card experts to there; pass --ctx to choose)"
                );
                Ok(ctx)
            }
            // The floor is the most the plan keeps its experts at, and a
            // floor that would not build keeps it too — the load's own plan
            // call names what refused it.
            _ => {
                eprintln!(
                    "{NAME}: --ctx defaults to {CTX} (the placed plan's card experts hold the \
                     context at the floor; pass --ctx to choose)"
                );
                Ok(CTX)
            }
        }
    };
    if let Some(p) = place {
        return placed_default(p);
    }
    let searched = match arch {
        Arch::Qwen3moe => q3place::whole_ctx_qwen3(split, CTX)?,
        Arch::Qwen35moe => {
            let o = Open35 {
                ctx: CTX,
                mma: true,
                ubatch: ubatch_size()?,
            };
            q3place::whole_ctx_qwen35(split, &o, CTX)?
        }
        _ => None,
    };
    match searched {
        Some(ctx) => {
            if trained > ctx {
                eprintln!(
                    "{NAME}: --ctx defaults to {ctx} of the file's {trained} trained positions \
                     (the whole load fits the card's free bytes; pass --ctx to choose)"
                );
            }
            Ok(ctx)
        }
        // The whole file does not fit the card's free bytes even at the
        // floor: the load falls to the placed plan on `a`'s card
        // (`open_unplaced_qwen3`), and its default is searched there.
        None => placed_default(Place::A),
    }
}

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the qwen3 seat runs no prompt cache";

struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
    place: Option<Place>,
    /// `--parallel`: slots that take the model in turns past 1.
    parallel: usize,
    queue_depth: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
        place: None,
        // The fixed default, not an elastic one: this seat's park holds ids
        // only (`Park::Ids`), no byte budget to size the slots from — a
        // state to park is a seat that saves one. A lone request pays
        // nothing for the second (the turns act only on a second arrival);
        // `--parallel 1` keeps the plain engine.
        parallel: 2,
        queue_depth: None,
    };
    let mut it = args.iter().map(String::as_str);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = v.parse().map_err(|e| format!("--port {v:?}: {e}"))?,
            "--ctx" | "--ctx-size" => match v.parse::<usize>() {
                Ok(n) if n > 0 => a.ctx = Some(n),
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            "--place" => a.place = Some(Place::parse(v)?),
            "--parallel" | "-np" => match v.parse::<usize>() {
                Ok(n) if n > 0 => a.parallel = n,
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            "--queue-depth" => match v.parse::<usize>() {
                Ok(q) => a.queue_depth = Some(q),
                _ => return Err(format!("{flag} takes a whole number, not {v:?}").into()),
            },
            "--park-ram" => {
                return Err(
                    "--park-ram names a budget of parked states; the qwen3 seat's park holds \
                     each slot's ids and re-prefills them on its return, so it parks no state \
                     and reads no budget"
                        .into(),
                );
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    Ok(a)
}

/// One of the two bodies the seat serves, opened as `generate_qwen3moe`
/// opens it, its prompt schedule and its keep rule.
trait Body3: ChainBody + Sized + 'static {
    /// The file's `general.architecture`, as the `load` record names it.
    const ARCH: &'static str;
    /// The model of `file` with a `ctx`-row cache: on device 0 — today's
    /// whole-card load while that fits the card's free bytes, else the
    /// placed plan on `a`'s card with its why
    /// (`q3place::open_unplaced_qwen3`/`_qwen35`) — or under `place` by
    /// its plan, the `plan` record on stderr first.
    fn open(
        file: Split,
        ctx: usize,
        place: Option<(Place, &Levers)>,
        levers: &Levers,
    ) -> Result<GpuModel<Self>, GateError>;
    /// The load runs its routed experts on the host tier too.
    fn placed(m: &GpuModel<Self>) -> Result<bool, GateError>;
    /// Every pass the prompt call replays, captured after the step.
    fn capture_passes(m: &mut GpuModel<Self>) -> Result<(), GateError>;
    /// The prompt call's ubatch size.
    fn ubatch(m: &GpuModel<Self>) -> Result<usize, GateError>;
    /// `ids` from where the model stands by the body's schedule; the argmax
    /// after the last.
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32]) -> Result<u32, GateError>;
    /// The body's rule as the server's keep query meets it: the longest
    /// prefix of at most `n` of the session's held positions a rollback
    /// keeps, and the rule's sentence when it keeps less.
    fn keep(s: &app::Session<Self>, n: usize) -> (usize, Option<String>);
    /// Take back the session's positions from `pos` on, one the rule
    /// granted; anything else is refused by name.
    fn rollback(s: &mut app::Session<Self>, pos: u32) -> Result<(), GateError>;
    /// Where to cut a prompt call of `first .. end` at `marks` so every run
    /// the cut makes still holds what the body's prompt schedule treats as
    /// one whole call; empty where the body cuts nowhere.
    fn splits(first: usize, end: usize, marks: &[usize]) -> Vec<usize>;
}

impl Body3 for Body {
    const ARCH: &'static str = "qwen3moe";

    fn open(
        file: Split,
        ctx: usize,
        place: Option<(Place, &Levers)>,
        levers: &Levers,
    ) -> Result<Qwen3moeModel, GateError> {
        let opts = Qwen3moeModel::lever_opts(ctx)?;
        let Some((p, levers)) = place else {
            return q3place::open_unplaced_qwen3(file, ctx, opts, levers, Record::eprint);
        };
        let q = PlaceQ3::qwen3(&file, p, ctx)?;
        let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
        q.record(&plan, None).eprint();
        q3place::open_qwen3(file, &plan, opts, levers.host())
    }

    fn placed(m: &Qwen3moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some())
    }

    fn capture_passes(m: &mut Qwen3moeModel) -> Result<(), GateError> {
        m.capture_prefill()?;
        Ok(())
    }

    fn ubatch(m: &Qwen3moeModel) -> Result<usize, GateError> {
        Ok(m.ubatch()?)
    }

    /// The session's own schedule (`app::Prompt for Body`): the GEMM walk
    /// from [`GEMM_FROM`] rows on, passes below.
    fn prompt(m: &mut Qwen3moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as app::Prompt>::prompt(m, ids)?)
    }

    /// Every held position (`app::Keep for Body`), the session's answer.
    fn keep(s: &app::Session<Body>, n: usize) -> (usize, Option<String>) {
        let pos = s.pos() as usize;
        let k = s.kept(u32::try_from(n).unwrap_or(u32::MAX));
        let at = k.at as usize;
        (at, (at < n.min(pos)).then(|| k.to_string()))
    }

    /// The session's cut, which takes only a position the rule granted.
    fn rollback(s: &mut app::Session<Body>, pos: u32) -> Result<(), GateError> {
        Ok(s.cut(pos)?)
    }

    /// The marks inside the call where both runs hold at least
    /// [`GEMM_FROM`] ids, so each run is the GEMM walk whose bits are the
    /// uncut call's.
    fn splits(first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let mut at = Vec::new();
        let mut last = first;
        for &u in marks {
            if u >= last + GEMM_FROM && u + GEMM_FROM <= end {
                at.push(u);
                last = u;
            }
        }
        at
    }
}

impl Body3 for Body35 {
    const ARCH: &'static str = "qwen35moe";

    fn open(
        file: Split,
        ctx: usize,
        place: Option<(Place, &Levers)>,
        levers: &Levers,
    ) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx,
            mma: true,
            ubatch: ubatch_size()?,
        };
        let mut m = match place {
            None => q3place::open_unplaced_qwen35(file, o, levers, Record::eprint)?,
            Some((p, levers)) => {
                let q = PlaceQ3::qwen35(&file, p, o)?;
                let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, None).eprint();
                q3place::open_qwen35(file, &plan, o, levers.host())?
            }
        };
        // The seat keeps prefixes, so its prompt calls take the checkpoints
        // of their marks.
        let (_, _, body) = m.body_parts(NAME)?;
        body.set_checkpoints(true);
        Ok(m)
    }

    fn placed(m: &Qwen35moeModel) -> Result<bool, GateError> {
        Ok(m.body(NAME)?.placed().is_some())
    }

    fn capture_passes(m: &mut Qwen35moeModel) -> Result<(), GateError> {
        m.capture_rows::<2>()?;
        m.capture_rows::<3>()?;
        m.capture_rows::<4>()?;
        m.capture_rows::<5>()?;
        m.capture_rows::<6>()?;
        m.capture_rows::<7>()?;
        m.capture_rows::<8>()?;
        Ok(())
    }

    fn ubatch(m: &Qwen35moeModel) -> Result<usize, GateError> {
        Ok(m.ubatch()?)
    }

    /// The session's own schedule (`app::Prompt for Body35`): the wide
    /// ubatch walk from [`GEMM_FROM`] rows on, passes below.
    fn prompt(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body35 as app::Prompt>::prompt(m, ids)?)
    }

    /// The checkpoints' rule (`app::Keep for Body35`), the session's answer:
    /// every held position, the nearest checkpoint at or below, or nothing
    /// with the rule's sentence.
    fn keep(s: &app::Session<Body35>, n: usize) -> (usize, Option<String>) {
        let pos = s.pos() as usize;
        let k = s.kept(u32::try_from(n).unwrap_or(u32::MAX));
        let at = k.at as usize;
        (at, (at < n.min(pos)).then(|| k.to_string()))
    }

    /// The session's cut, which takes only a position the rule granted.
    fn rollback(s: &mut app::Session<Body35>, pos: u32) -> Result<(), GateError> {
        Ok(s.cut(pos)?)
    }

    /// The marks inside the call where both runs hold at least
    /// [`GEMM_FROM`] ids, so each run is the wide walk whose bits are the
    /// uncut call's.
    fn splits(first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let mut at = Vec::new();
        let mut last = first;
        for &u in marks {
            if u >= last + GEMM_FROM && u + GEMM_FROM <= end {
                at.push(u);
                last = u;
            }
        }
        at
    }
}

/// The session on the engine thread and the positions its cache was sized
/// for.
struct Q3<B: Body3> {
    s: app::Session<B>,
    ctx: usize,
}

impl<B: Body3> Q3<B> {
    /// The model of the file at `path` in graph mode — under `place` by
    /// its plan, else the default load — its step captured and, unless
    /// placed, its passes, the session over it; the `load` record on
    /// stderr.
    fn open(
        path: &Path,
        ctx: usize,
        place: Option<(Place, &Levers)>,
        levers: &Levers,
    ) -> Result<Q3<B>, GateError> {
        let t = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut m = B::open(file, ctx, place, levers)?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        if !B::placed(&m)? {
            B::capture_passes(&mut m)?;
        }
        Record::new(&record::LOAD_QWEN3)
            .w("arch", B::ARCH)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", ctx)
            .u("layers", m.layers().len())
            .u("ubatch", B::ubatch(&m)?)
            .u("graph_nodes", nodes)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
        Ok(Q3 {
            s: app::Session::from_model(m, at),
            ctx,
        })
    }
}

impl<B: Body3> Seat for Q3<B> {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        B::prompt(self.s.model_mut(), ids)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset ([`GpuModel::reset`]): these bodies never hold a
    /// verify's rows, so the session's own reset adds nothing to it.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.model_mut().reset()?)
    }

    /// One [`Body3::keep`] granted ([`Seat::keep`]); a cut the rule does
    /// not grant is refused by the session, by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        B::rollback(&mut self.s, pos)
    }

    fn keep(&self, n: usize) -> (usize, Option<String>) {
        B::keep(&self.s, n)
    }

    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        B::splits(first, end, marks)
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    /// A prefix kept less of than shared is a `cache reuse` record; every
    /// other note prints as its line.
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
        eprintln!("{NAME}: {note}");
    }
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    // A placed load (`--place`) acts on the placement's levers; set beside
    // an unplaced one, they are refused by name.
    let placed = args.iter().any(|a| a == "--place");
    let levers = bloomery_levers::at_main(if placed { &q3place::PLACED_LEVERS } else { &[] })?;
    record::at_main(NAME, record::BLOOMERY_SERVE_QWEN3);
    let a = parse_args(args)?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    let template = split
        .value("tokenizer.chat_template")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
        .to_owned();
    let alias = split
        .value("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("qwen3")
        .to_owned();
    let ctx = match a.ctx {
        Some(c) => c,
        None => default_ctx(&split, arch, a.place, &levers)?,
    };
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let open = path.clone();
    let place = a.place;
    let device = place.map_or("device 0", Place::name).to_owned();
    let engine = match arch {
        Arch::Qwen3moe => SeatEngine::spawn(
            move || Q3::<Body>::open(&open, ctx, place.map(|p| (p, &levers)), &levers),
            ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
        Arch::Qwen35moe => SeatEngine::spawn(
            move || Q3::<Body35>::open(&open, ctx, place.map(|p| (p, &levers)), &levers),
            ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
        other => {
            return Err(format!(
                "{} is a {} file; the qwen3 seat serves qwen3moe and qwen35moe files",
                path.display(),
                other.name()
            )
            .into());
        }
    };
    let config = ServerConfig {
        model_alias: alias,
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
    };
    // One slot stays the plain engine; several take it in turns, a preempted
    // request's held ids re-prefilled from a reset on its return (the park
    // holds ids: the seat has no snapshot to park, and a kept prefix does
    // not survive another slot's rows over the cache).
    let engine: Box<dyn Engine> = match a.parallel {
        0 | 1 => Box::new(engine),
        n => Box::new(SwapEngine::new(Box::new(engine), n, Park::Ids)?),
    };
    let slots = SlotConfig {
        parallel: a.parallel,
        queue_depth: a.queue_depth,
        ..SlotConfig::default()
    };
    let server = Server::bind_with((a.host.as_str(), a.port), engine, config, slots)?;
    Record::new(&record::LISTENING_QWEN3)
        .w("arch", arch.name())
        .u("ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
