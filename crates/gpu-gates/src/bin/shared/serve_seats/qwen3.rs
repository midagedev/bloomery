//! `bloomery-serve --model qwen3` — the llama-server-compatible HTTP API on
//! the single-card Qwen engines: a qwen3moe file (Qwen3-30B-A3B) or a
//! qwen35moe file (Qwen3.6-35B-A3B), the whole model on device 0, under
//! `--place` placed by its plan on the card the word names with the routed
//! experts the card's budget does not hold on the host tier.
//!
//!     bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--place W]
//!                    [--cache-type-k f16|q8_0] [--parallel N] [--queue-depth Q]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` is `--ctx` under llama-server's spelling, C defaults
//! to the file's trained context capped to the largest whose whole load fits
//! what the card had, its arena and reserve counted (a whole-card load; never
//! under 4096, a multiple of 1024, one stderr line when the cap binds), or —
//! for a placed load, under `--place` or the plan
//! a run without it falls to — to the largest context whose placed plan
//! keeps the card experts the 4096-floor's plan keeps (a plan that keeps
//! every expert opens whole, so the whole fit must take it too), one stderr
//! line naming it. 4096 for a file that states none. The model is opened as
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
//! eager passes of up to eight ids (`app::Prompt for Body` takes passes at
//! every length on a placed load), no pass captured.
//!
//! With `--place` unset the model opens as `generate_qwen3moe` opens it:
//! the whole file on device 0 while the whole-fit verdict takes it (one
//! stderr line with its terms), else the placed plan on `a`'s card, its
//! `plan` record naming why (`whole_does_not_fit`). Each `--ctx` search
//! prints one line more: its probes and the census readings it took.
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
//! `load` record (architecture, resident bytes — every slot's sequences —,
//! context, slots, each slot's context, layers, ubatch,
//! the step graph's nodes, the load's wall) then the `listening` record go
//! to stderr (`record::BLOOMERY_SERVE_QWEN3`). An engine error ends the
//! process with the crash block and exit code 70, as every seat's.
//!
//! `--parallel N` (`-np N`, default 2) serves N slots. A qwen3moe file's
//! are N resident sequences inside the one model (`GpuModel::add_slots` on
//! the session), the context split across them as llama-server splits it
//! with `-np N` and no `-kvu`: the `--ctx` the flags named (or the auto
//! choice when unset) is the total, each slot `total / N` rows rounded down
//! — a cache row is the granularity, so the floor is exact, and N slots
//! never hold more rows than the one-sequence load — and `--parallel N` is
//! the slot count itself, the split's bytes the bound. A whole-card load
//! and a placed one — under `--place`, or the plan a run without it falls to
//! when the whole file does not fit the card's free bytes — split alike:
//! what the file loads as is decided once, before the open, at the total
//! (the whole-fit verdict there, or the placed plan made there, whose card
//! bytes count every slot's rows), and the open makes no fit call of its
//! own, so a total whose whole load does not fit never opens whole because
//! one slot's share would. A placed open holds the slots' cache to the
//! plan's KV term by name and prints both (`q3place::open_qwen3_slots`).
//! One token a slot a round, the sequences switched by pointer exchange, no
//! park and no re-prefill; the engine is the seat's own (`SeatEngine`), and
//! the server steps every running slot in one call. `--parallel 1` is
//! exactly the one-sequence server. A qwen35moe file's slots take its one
//! sequence in turns (`serve::SwapEngine`) over the whole context, until its
//! body holds resident sequences. Under
//! the turns the default's second slot costs a lone request nothing — the
//! turns act only on a second arrival, and an idle slot the engine left
//! holds nothing, so a lone request stays on the engine's slot — and
//! `--parallel 1` keeps the plain engine: a request that
//! arrives while another decodes preempts it at the next step, the live
//! requests then take turns of `serve::QUANTUM` tokens, and a preempted
//! request comes back by the re-prefill fallback — the engine reset to
//! position 0 and its held ids fed again before it steps
//! (`serve::Park::Ids`; the seat holds no snapshot to park, and a kept
//! prefix does not survive another slot's rows over the cache), the ids it
//! feeds again counted in `/metrics`. `--queue-depth Q` bounds the requests
//! that wait for a slot; `--park-ram` is refused by name: neither the turns'
//! park nor the resident slots hold a state the flag could budget.
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

use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::arch::qwen3moe::{Body, Body35, KvQ8, Open35, Qwen35moeModel};
use bloomery_gpu::model::{ChainBody, GpuModel, MAX_PASS_ROWS, StepMode};
use bloomery_gpu::{Gpu, Qwen3moeModel};
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
                     [--host H] [--port P] [--ctx C] [--place W] [--cache-type-k f16|q8_0] \
                     [--parallel N] [--queue-depth Q]";

/// The context unless `--ctx` says: `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// The `--ctx` the seat takes when the flag is unset. A whole-card load
/// takes the file's trained context capped to the largest whose whole load
/// the whole-fit verdict takes — weights, cache, the program's arena at the
/// load's ubatch and the reserve its load keeps free past it, against what
/// the card had (`q3place::whole_ctx_qwen3`) — never under [`CTX`], a
/// multiple of 1024, one line on stderr when the cap binds; a file that
/// states no trained context keeps [`CTX`]. A placed load — under `--place`,
/// or the plan a run without it falls to when the whole load does not fit —
/// searches the same shape over the plan's own expert split: the largest
/// context whose plan keeps the card experts the [`CTX`]-floor's plan keeps,
/// so the solver's context-for-experts trade never sits below the floor's
/// split, and whose whole load the verdict takes when that plan keeps every
/// expert on the card (the load then opens whole). One line names the answer, at the floor or past it, and every
/// search one line more: its probes and the census readings it took
/// ([`searched_line`]).
fn default_ctx(
    split: &Split,
    arch: Arch,
    place: Option<Place>,
    levers: &Levers,
    kv: KvQ8,
) -> Result<usize, GateError> {
    let Some(trained) = q3place::trained_ctx(split) else {
        return Ok(CTX);
    };
    let plan_levers = PlanLevers::from_levers(levers)?;
    let placed_default = |place: Place| -> Result<usize, GateError> {
        let before = q3place::reads();
        let searched = match arch {
            Arch::Qwen3moe => q3place::placed_ctx_qwen3(split, place, CTX, &plan_levers, kv)?,
            Arch::Qwen35moe => {
                let o = Open35 {
                    ctx: CTX,
                    mma: true,
                    ubatch: ubatch_size()?,
                    kv,
                };
                q3place::placed_ctx_qwen35(split, &o, place, CTX, &plan_levers)?
            }
            _ => None,
        };
        searched_line("placed", before);
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
    let before = q3place::reads();
    let searched = match arch {
        Arch::Qwen3moe => q3place::whole_ctx_qwen3(split, CTX, kv)?,
        Arch::Qwen35moe => {
            let o = Open35 {
                ctx: CTX,
                mma: true,
                ubatch: ubatch_size()?,
                kv,
            };
            q3place::whole_ctx_qwen35(split, &o, CTX)?
        }
        _ => None,
    };
    searched_line("whole", before);
    match searched {
        Some(ctx) => {
            if trained > ctx {
                eprintln!(
                    "{NAME}: --ctx defaults to {ctx} of the file's {trained} trained positions \
                     (the whole load fits what the card had, its arena and reserve counted; pass \
                     --ctx to choose)"
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

/// The line a `--ctx` search (`kind`, whole or placed) prints once it
/// answers: the probes it ran and the census readings it took since
/// `before` — one reading serves every probe, so a search that read the
/// census per probe shows here.
fn searched_line(kind: &str, before: q3place::Reads) {
    let r = q3place::reads().since(before);
    eprintln!(
        "{NAME}: the {kind} --ctx search ran {} probes on {} census reading(s)",
        r.probes, r.censuses
    );
}

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the qwen3 seat runs no prompt cache";

struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
    place: Option<Place>,
    /// `--cache-type-k`: the K/V planes' format (llama-server's spelling),
    /// over the `BLOOMERY_QWEN3_KV` lever's word when given.
    cache_type_k: Option<String>,
    /// `--parallel`: the slots the server serves — resident sequences on a
    /// qwen3moe file's load, turns over one sequence on a qwen35moe file's.
    parallel: usize,
    queue_depth: Option<usize>,
}

/// The K/V planes' format this run loads: `--cache-type-k`'s word when given
/// (refused by name on any other), else the `BLOOMERY_QWEN3_KV` lever's,
/// else f16.
fn cache_k(a: &Args, levers: &Levers) -> Result<KvQ8, GateError> {
    let word = a
        .cache_type_k
        .as_deref()
        .or(levers.qwen3_kv_set())
        .unwrap_or("f16");
    KvQ8::parse(word)
        .ok_or_else(|| format!("--cache-type-k takes f16 or q8_0, not {word:?}").into())
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
        place: None,
        cache_type_k: None,
        // The fixed default, not an elastic one: a qwen3moe file's second
        // slot is a resident sequence the context split bounds (no byte
        // budget to size it from), a qwen35moe file's a turn over the one
        // sequence (the park holds ids only, `Park::Ids` — a state to park
        // is a seat that saves one). A lone request pays nothing for the
        // second either way: the resident slots sit parked empty, and the
        // turns act only on a second arrival; `--parallel 1` keeps the
        // plain engine.
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
            "--cache-type-k" => a.cache_type_k = Some(v.to_owned()),
            "--cache-type-v" => {
                return Err(
                    "--cache-type-v does not exist: the qwen3 cache quantizes its K and V \
                     planes together, so --cache-type-k names the one format both take"
                        .into(),
                );
            }
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
                    "--park-ram names a budget of parked states; the qwen3 seat parks none — \
                     its resident slots keep their sequences on the card, and its turns park \
                     each slot's ids and re-prefill them on its return — so it reads no budget"
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
    /// What the open loads by beside the file and its rows: what the seat
    /// decided before the open ([`Q3Load`], a qwen3moe file's), or the
    /// placement word the open decides by itself (a qwen35moe file's).
    type By: Send + 'static;
    /// The model of `file` with a `ctx`-row cache by `by`, for `slots`
    /// sequences of that shape, the `plan` record on stderr first when a
    /// plan loads it.
    fn open(
        file: Split,
        ctx: usize,
        slots: usize,
        by: Self::By,
        levers: &Levers,
        kv: KvQ8,
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
    /// The session's model serving `n` resident sequences
    /// (`app::Session::add_slots`): N sequences over one set of weights,
    /// switched by pointer exchange. `Body` (a qwen3moe file) serves them;
    /// `Body35` (a qwen35moe file) refuses by name — its body parks no
    /// sequence yet, so its slots keep taking the engine in turns.
    fn add_slots(s: &mut app::Session<Self>, n: usize) -> Result<(), GateError>;
    /// The session's slot every later call acts on
    /// (`app::Session::select_slot`). `Body35` refuses by name, as
    /// [`Body3::add_slots`] does.
    fn select_slot(s: &mut app::Session<Self>, slot: usize) -> Result<(), GateError>;
}

impl Body3 for Body {
    const ARCH: &'static str = "qwen3moe";
    type By = Q3Load;

    /// As `load` decided before the open, making no fit call of its own: the
    /// whole model on device 0, or the placed plan made at the total, its
    /// load holding a sequence of `ctx` rows for each of `slots`
    /// (`q3place::open_qwen3_slots`).
    fn open(
        file: Split,
        ctx: usize,
        slots: usize,
        load: Q3Load,
        levers: &Levers,
        kv: KvQ8,
    ) -> Result<Qwen3moeModel, GateError> {
        let opts = Qwen3moeModel::lever_opts(ctx, kv)?;
        match load {
            Q3Load::Whole => Ok(Qwen3moeModel::open(Gpu::new()?, file, opts)?),
            Q3Load::Placed { q, total, why } => {
                let plan = q.plan(total, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, why).eprint();
                q3place::open_qwen3_slots(file, &plan, slots, opts, levers.host())
            }
        }
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
    /// from [`GEMM_FROM`] rows on, passes below; on a placed load passes
    /// at every length.
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

    /// Nowhere: the caches are per-position, so every position the body
    /// holds is keepable as it stands and a cut never needs the calls on
    /// either side of it to be whole prompt calls of their own — a token's
    /// values do not depend on its ubatch.
    fn splits(_first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// The session's resident sequences, the body's own
    /// (`impl Slots for Body`).
    fn add_slots(s: &mut app::Session<Body>, n: usize) -> Result<(), GateError> {
        Ok(s.add_slots(n)?)
    }

    /// The session's slot ([`app::Session::select_slot`]).
    fn select_slot(s: &mut app::Session<Body>, slot: usize) -> Result<(), GateError> {
        Ok(s.select_slot(slot)?)
    }
}

impl Body3 for Body35 {
    const ARCH: &'static str = "qwen35moe";
    type By = Option<Place>;

    /// On device 0 — the whole-card load while that fits the card's free
    /// bytes at `ctx`, else the placed plan on `a`'s card with its why
    /// (`q3place::open_unplaced_qwen35`) — or under `place` by its plan. One
    /// sequence: the body parks none, so its slots take turns and `slots`
    /// is 1.
    fn open(
        file: Split,
        ctx: usize,
        _slots: usize,
        place: Option<Place>,
        levers: &Levers,
        kv: KvQ8,
    ) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx,
            mma: true,
            ubatch: ubatch_size()?,
            kv,
        };
        let mut m = match place {
            None => q3place::open_unplaced_qwen35(file, o, levers, Record::eprint)?,
            Some(p) => {
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

    /// Refused by name: the qwen35moe body parks no sequence, so its slots
    /// take the engine in turns (the module doc's qwen35moe half).
    fn add_slots(_s: &mut app::Session<Body35>, _n: usize) -> Result<(), GateError> {
        Err(
            "--parallel: the qwen35moe body holds one sequence; its slots take the engine in \
             turns (resident slots are the qwen3moe body's)"
                .into(),
        )
    }

    /// Refused by name, as [`Body3::add_slots`] is.
    fn select_slot(_s: &mut app::Session<Body35>, _slot: usize) -> Result<(), GateError> {
        Err("slot: the qwen35moe seat serves slot 0 alone".into())
    }
}

/// The session on the engine thread, the positions its cache was sized
/// for, and the resident sequences the seat serves.
struct Q3<B: Body3> {
    s: app::Session<B>,
    ctx: usize,
    /// [`Seat::slots`]: 1, or the `--parallel` the seat made resident
    /// sequences of.
    slots: usize,
}

impl<B: Body3> Q3<B> {
    /// The model of the file at `path` in graph mode by `by`
    /// ([`Body3::open`]), its step captured and, unless placed, its passes,
    /// `slots` resident sequences parked after them (each capturing on its
    /// first use), the session over it; the `load` record on stderr.
    fn open(
        path: &Path,
        ctx: usize,
        slots: usize,
        by: B::By,
        levers: &Levers,
        kv: KvQ8,
    ) -> Result<Q3<B>, GateError> {
        let t = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut m = B::open(file, ctx, slots, by, levers, kv)?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        if !B::placed(&m)? {
            B::capture_passes(&mut m)?;
        }
        let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
        let mut s = app::Session::from_model(m, at);
        if slots > 1 {
            B::add_slots(&mut s, slots)?;
        }
        Record::new(&record::LOAD_QWEN3)
            .w("arch", B::ARCH)
            .u("resident_bytes", s.model().resident_bytes())
            .w("cache", kv.name())
            .u("ctx", ctx)
            .u("slots", slots)
            .u("slot_ctx", ctx)
            .u("layers", s.model().layers().len())
            .u("ubatch", B::ubatch(s.model())?)
            .u("graph_nodes", nodes)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        Ok(Q3 { s, ctx, slots })
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

    /// The resident sequences the seat made at its open: `--parallel` on a
    /// qwen3moe file's load, one on a qwen35moe file's.
    fn slots(&self) -> usize {
        self.slots
    }

    /// The session's slot ([`Body3::select_slot`]): the model exchanges its
    /// live sequence with the slot's parked state — pointer moves.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        B::select_slot(&mut self.s, slot)
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

/// What a qwen3moe file loads as, decided once before the open at the
/// total context — the context its whole-fit verdict and its plan count —
/// and carried into it ([`Body3::By`]), so the open makes no fit call of
/// its own ([`decide_qwen3`]).
enum Q3Load {
    /// The whole model on device 0: `--place` unset, and the whole-fit
    /// verdict at the total takes the whole load.
    Whole,
    /// The placed plan made at `total` positions on `q`'s card: `--place`'s
    /// (no `why`), or `a`'s card's when the whole load does not fit at the
    /// total (`why` names that, `q3place::WHY_NOT_WHOLE`).
    Placed {
        q: Box<PlaceQ3>,
        total: usize,
        why: Option<&'static str>,
    },
}

/// What a qwen3moe file loads as at `total` positions ([`Q3Load`]):
/// `--place`'s plan on its card, else the whole-fit verdict at the total
/// (`q3place::unplaced_qwen3`, its line on stderr) — the whole model on
/// device 0, or the placed plan on `a`'s card. Made at the total, never at a
/// slot's share: a card whose whole load fits a slot's rows and not the
/// total's would open whole, and the slots added after the open would hold
/// a cache the verdict never counted.
fn decide_qwen3(
    split: &Split,
    total: usize,
    place: Option<Place>,
    kv: KvQ8,
) -> Result<Q3Load, GateError> {
    let (q, why) = match place {
        Some(p) => (Box::new(PlaceQ3::qwen3(split, p, total, kv)?), None),
        None => match q3place::unplaced_qwen3(split, total, kv)? {
            q3place::Unplaced::Whole => return Ok(Q3Load::Whole),
            q3place::Unplaced::Placed(q) => (q, Some(q3place::WHY_NOT_WHOLE)),
        },
    };
    Ok(Q3Load::Placed { q, total, why })
}

/// The body the file opens as: a qwen3moe file with its load decided
/// before the open, or a qwen35moe file, whose open decides by `--place`
/// itself ([`Body3::By`]).
enum Seated {
    Qwen3(Q3Load),
    Qwen35,
}

/// How the seat serves `--parallel` for the file (the module doc's
/// `--parallel` half): resident slots with the context split across them,
/// or one sequence the slots take in turns.
enum Serving {
    /// `n` resident sequences, the model loaded with `slot_ctx` cache rows.
    Slots { n: usize, slot_ctx: usize },
    /// One sequence of `ctx` rows, `n` slots taking it in turns (one slot:
    /// the plain engine).
    Turns { n: usize, ctx: usize },
}

/// The least context a resident slot loads at, one rule for both loads
/// ([`slot_floor`] names why for each): [`MAX_TOKENS`] rows.
const SLOT_CTX_MIN: usize = MAX_TOKENS;
const _: () = assert!(SLOT_CTX_MIN == MAX_PASS_ROWS);

/// Why a slot of `load` takes at least [`SLOT_CTX_MIN`] rows. A whole-card
/// load captures its widest pass (`capture_prefill`, [`MAX_TOKENS`] rows),
/// whose capture writes that many rows into the cache — a cache of fewer
/// rows is refused by the launcher, deep in the load. A placed load captures
/// no pass; its prompt runs as eager passes of up to [`MAX_PASS_ROWS`] ids,
/// and a slot of fewer rows cannot hold one whole pass.
fn slot_floor(load: &Q3Load) -> String {
    match load {
        Q3Load::Whole => format!("the {SLOT_CTX_MIN} rows the load's widest captured pass writes"),
        Q3Load::Placed { .. } => format!(
            "the {SLOT_CTX_MIN} rows one whole prompt pass of a placed load writes (its prompt \
             runs as eager passes of up to {MAX_PASS_ROWS} ids)"
        ),
    }
}

/// How `parallel` slots serve the file `seated` at a total context of
/// `ctx`. A qwen3moe file past one slot holds them resident, on a
/// whole-card load and a placed one alike ([`Q3Load`], decided at the
/// total): the context split across them, each `ctx / parallel` rows
/// rounded down (a cache row the granularity, the floor exact; N slots
/// never more rows than the one-sequence load, so never more cache than the
/// verdict or the plan at the total counts), a split that leaves a slot
/// under [`SLOT_CTX_MIN`] rows refused by name before anything loads
/// ([`slot_floor`]). A qwen35moe file, whose body parks no sequence, keeps
/// one sequence the slots take in turns over the whole context.
fn serving(seated: &Seated, ctx: usize, parallel: usize) -> Result<Serving, GateError> {
    let load = match seated {
        Seated::Qwen3(load) if parallel > 1 => load,
        _ => return Ok(Serving::Turns { n: parallel, ctx }),
    };
    let slot_ctx = ctx / parallel;
    if slot_ctx < SLOT_CTX_MIN {
        return Err(format!(
            "--parallel {parallel} of a --ctx of {ctx}: a slot's context of {slot_ctx} rows is \
             below {}; give a larger --ctx or a smaller --parallel",
            slot_floor(load)
        )
        .into());
    }
    Ok(Serving::Slots {
        n: parallel,
        slot_ctx,
    })
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    // A placed load (`--place`) acts on the placement's levers; set beside
    // an unplaced one, they are refused by name.
    let placed = args.iter().any(|a| a == "--place");
    let mut acts_on = vec![bloomery_levers::QWEN3_KV];
    if placed {
        acts_on.extend(q3place::PLACED_LEVERS);
    }
    let levers = bloomery_levers::at_main(&acts_on)?;
    record::at_main(NAME, record::BLOOMERY_SERVE_QWEN3);
    let a = parse_args(args)?;
    let kv = cache_k(&a, &levers)?;
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
        None => default_ctx(&split, arch, a.place, &levers, kv)?,
    };
    // What the file loads as is decided here, once, at the total: the open
    // makes no fit call of its own (`decide_qwen3`).
    let seated = match arch {
        Arch::Qwen3moe => Seated::Qwen3(decide_qwen3(&split, ctx, a.place, kv)?),
        Arch::Qwen35moe => Seated::Qwen35,
        other => {
            return Err(format!(
                "{} is a {} file; the qwen3 seat serves qwen3moe and qwen35moe files",
                path.display(),
                other.name()
            )
            .into());
        }
    };
    let serving = serving(&seated, ctx, a.parallel)?;
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let open = path.clone();
    let place = a.place;
    let device = place.map_or("device 0", Place::name).to_owned();
    // Resident slots load the model with the slot ctx and park the rest of
    // the slots after its captures; slots that take turns load one sequence
    // of the whole context.
    let (load_ctx, resident) = match serving {
        Serving::Slots { n, slot_ctx } => (slot_ctx, n),
        Serving::Turns { ctx, .. } => (ctx, 1),
    };
    let engine = match seated {
        Seated::Qwen3(load) => SeatEngine::spawn(
            move || Q3::<Body>::open(&open, load_ctx, resident, load, &levers, kv),
            load_ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
        Seated::Qwen35 => SeatEngine::spawn(
            move || Q3::<Body35>::open(&open, load_ctx, resident, place, &levers, kv),
            load_ctx,
            vocab,
            device,
            EngineProps::default(),
            0,
        )?,
    };
    let config = ServerConfig {
        model_alias: alias,
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
    };
    // The seat's own slots are the engine's (the seat made them at its
    // open, and the server steps them together); one sequence several slots
    // take in turns stays the swap engine's, a preempted request's held ids
    // re-prefilled from a reset on its return (the park holds ids: the seat
    // has no snapshot to park, and a kept prefix does not survive another
    // slot's rows over the cache).
    let engine: Box<dyn Engine> = match serving {
        Serving::Slots { .. } | Serving::Turns { n: 0 | 1, .. } => Box::new(engine),
        Serving::Turns { n, .. } => Box::new(SwapEngine::new(Box::new(engine), n, Park::Ids)?),
    };
    let slots = SlotConfig {
        parallel: a.parallel,
        queue_depth: a.queue_depth,
        ..SlotConfig::default()
    };
    let server = Server::bind_with((a.host.as_str(), a.port), engine, config, slots)?;
    Record::new(&record::LISTENING_QWEN3)
        .w("arch", arch.name())
        .u("ctx", load_ctx)
        .u("slots", a.parallel)
        .u("slot_ctx", load_ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
