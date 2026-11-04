//! `bloomery-serve --model host` — the llama-server-compatible HTTP API on the
//! CPU engine with no card: the file's host forward (`deepseek2::HostTarget`)
//! served through `serve::TargetEngine`, the adapter the Metal server serves
//! its session through. One seat for every architecture the host engine runs
//! whole; today that is deepseek2 (DeepSeek-V2-Lite), the one architecture
//! whose every site the CPU engine owns.
//!
//!     bloomery-serve --model host [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--queue-depth Q] [--plan]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module doc);
//! `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. Unset,
//! the context is the file's trained one: the cache grows as positions are
//! fed, so no context is fitted, and a set context past the trained one stops
//! by name. The model is opened as `bloomery-decode` opens it — the file's
//! mapping prefaulted once, the load-time derived weights built once, a
//! `(seq, pos)` KV cache — and a request's ids are the CLI's: the server
//! feeds the prompt less its last id as one prompt call, then steps the last,
//! which is the same bits as the CLI's one call over the whole prompt (a
//! batch of any split gives the same bits as one id a step). Its first shard
//! gives the vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the alias (`host`
//! for a file that names none). Sampling is the sampler crate's chain with no
//! repetition penalty, `temperature <= 0` the engine's argmax.
//!
//! The architecture is told first ([`Arch::detect`]): deepseek2 has no model
//! reader, so it is served by its name; every other architecture is read by
//! its reader and checked against the host engine's rows
//! (`coverage::check_host`) — a file with an item on the list ends `unsupported`
//! (65) naming every one, before any weight is read. `--place` is refused by
//! name (the host stage's plan names no card), as are `--parallel` above 1
//! (the target holds one sequence), `--slot-save-path` (no prompt cache) and
//! the embedding model flags (their seat plans on a card).
//!
//! One sequence: a request keeps the longest prefix it shares with what the
//! slot holds (the target's `keepable` is every position it was fed, and a
//! cut takes the cache back to one). The records on stderr, in order: the
//! `ctx` line, the `plan` record (`plan stage host layers 0..<block_count>
//! head host`), the `need` record (the file's bytes as reclaimable cache,
//! no card), the `parallel` line, then after the load the `load` record and,
//! once the port is bound, the `listening` record
//! (`record::BLOOMERY_SERVE_HOST`). `--plan` ends the process after the
//! `parallel` line, nothing loaded. An engine error ends the process with the
//! crash block and exit code 70, as every seat's.
//!
//! The seat takes no card and no card lock: a start with no visible card, or
//! one whose llama placement flags ask for the host (`-dev none`, `-ngl 0`),
//! seats the file here (`drive::seat_of_file`). The levers it acts on are the
//! start screen and a structured-output request's fixed runs; a host-expert
//! or card lever is refused by name as one this seat does not read.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu_gates::bind::{
    self, ChatSurface, Listen, Pass, arm_screen, bind_server, check_seat, parallel_line,
    slots_given,
};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::start::{self, Refusal};
use bloomery_gpu_gates::{GateError, generate, model_file, ref_model_path};
use gguf::Split;
use model::ModelError;
use model::arch::deepseek2::target::HostTarget;
use model::arch::{Arch, coverage};
use model::placement::PlacementError;
use model::placement::ctx_ask::{self, Chosen, Limit};
use model::placement::workstation::NeedClasses;
use serve::exit::Code;
use serve::{EngineProps, ServeError, TargetEngine};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-host";

const USAGE: &str = "usage: bloomery-serve --model host [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--queue-depth Q] [--plan] [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the start screen and how a structured-output
/// request's fixed runs are fed.
pub const ACTS_ON: &[&str] = &[bloomery_levers::SCREEN, bloomery_levers::SCHEMAJUMP];

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the host seat runs no prompt cache";

/// What the host seat serves, closing a refusal of another architecture.
const SERVES: &str = "this build's host seat serves deepseek2 only";

/// The seat's own flags past [`bind::SeatFlags`]: none. `--place` is refused
/// (the host stage names no card), and the slot-file directory with it.
fn parse(args: &[String]) -> Result<bind::SeatFlags, GateError> {
    let a = bind::SeatFlags::parse(args, USAGE, |_, _| Ok(false))?;
    a.one_sequence("host")?;
    if a.place.is_some() {
        return Err(
            "--place names the cards a plan stages on; the host stage runs every layer with no \
             card in the plan, and the host is asked for with -ngl 0 or -dev none"
                .into(),
        );
    }
    if a.slot_save_path.is_some() {
        return Err(format!("--slot-save-path: {NO_CACHE}").into());
    }
    Ok(a)
}

/// The levers (unless `read_levers` is false) and flags [`run`] reads first, checked before the
/// model is fetched ([`bind::check_seat`]).
pub fn check(
    args: &[String],
    read_levers: bool,
) -> Result<bloomery_gpu_gates::bind::CheckedSeat<bind::SeatFlags>, GateError> {
    check_seat(read_levers.then_some(ACTS_ON), args, |args, _| parse(args))
}

/// The file's tensor bytes as the header states them: what the load's mapping
/// holds, the figure the need record and the load record name.
fn resident_bytes(split: &Split) -> Result<u64, GateError> {
    split
        .iter_tensors()
        .map(|(_, t)| t.nbytes)
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| "the file's tensor bytes pass u64".into())
}

/// The refusal of a file the host engine does not serve whole, exit 65 (`unsupported`): an
/// architecture no reader here reads, or one whose reader's rows name what the host engine
/// runs none of ([`coverage::check_host`]), each item on the list.
fn unsupported(path: &Path, split: &Split, detected: Result<Arch, ModelError>) -> Refusal {
    let shown = path.display();
    let arch = match detected {
        Ok(arch) => arch,
        Err(e) => return Refusal::new(Code::Unsupported, format!("{shown}: {e}; {SERVES}")),
    };
    let read = match model::arch::spec(split) {
        Ok(read) => read,
        Err(e) => return Refusal::new(Code::Unsupported, format!("{shown}: {e}; {SERVES}")),
    };
    let missing = coverage::check_host(&read.spec, &read.tensors);
    if missing.is_empty() {
        Refusal::new(
            Code::Unsupported,
            format!(
                "{shown} is a {} file whose rows the host engine covers, and {SERVES}",
                arch.name()
            ),
        )
    } else {
        Refusal::new(
            Code::Unsupported,
            format!(
                "{shown}: the host engine (no card, every layer on the host stage) runs none of: {}",
                PlacementError::Unimplemented(missing)
            ),
        )
    }
}

/// Serves the model, and only it, until the listener or the engine fails; `Ok`
/// carries why the server ended.
pub fn run(args: &[String], side: &side_seat::Flags) -> Result<ServeError, GateError> {
    record::at_main_with_flags(NAME, record::BLOOMERY_SERVE_HOST, |flag| {
        bloomery_gpu_gates::llama_launch::serve_option(args, flag)
    });
    let (a, levers) = check(args, true)?.run();
    arm_screen(levers.screen());
    let path = ref_model_path()?;
    if side.names() {
        return Err(
            "--embd-model/--embd-hf: the embedding model's seat plans on a card, and the host \
             seat's plan holds none"
                .into(),
        );
    }
    let split = model_file::header(&path)?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let detected = Arch::detect(first);
    if detected.as_ref().ok() != Some(&Arch::Deepseek2) {
        return Err(unsupported(&path, &split, detected).into());
    }
    let trained = u64::try_from(generate::trained_ctx(&split).ok_or_else(|| {
        format!(
            "{}: {} is absent or not a count",
            path.display(),
            split.arch_key("context_length")
        )
    })?)?;
    let ctx = match a.ctx {
        Some(asked) => {
            let asked = u64::try_from(asked).map_err(|_| format!("--ctx {asked} passes u64"))?;
            if asked > trained {
                return Err(format!("--ctx {asked} passes the file's trained {trained}").into());
            }
            asked
        }
        None => trained,
    };
    let chat = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.path(),
        "host",
    )?;
    let layers = split.arch_get_u64("block_count").ok_or_else(|| {
        format!(
            "{}: {} is absent or not a count",
            path.display(),
            split.arch_key("block_count")
        )
    })?;
    let bytes = resident_bytes(&split)?;
    let asked = Chosen {
        slot_ctx: ctx,
        slots: 1,
        total: ctx,
        rule: if a.ctx.is_some() {
            ctx_ask::Rule::Set
        } else {
            ctx_ask::Rule::Target
        },
        limit: Limit {
            positions: trained,
            source: "trained",
        },
        plan: (),
    };
    let (slots, from) = slots_given(a.parallel, a.ctx, a.auto.is_some(), ctx_ask::DEFAULT_SLOTS)?;
    eprintln!("{}", asked.line());
    start::note_ctx(ctx, slots as u64, asked.rule.word(), trained);
    start::note_place("host", "no card in the plan");
    Record::new(&record::PLAN_HOST)
        .w("stage", "host")
        .w("layers", &format!("0..{layers}"))
        .w("head", "host")
        .eprint();
    let pool = threads::report().map_err(|e| format!("the worker pool: {e}"))?;
    let need = record::Need {
        bytes,
        classes: NeedClasses {
            cache: bytes,
            ..NeedClasses::default()
        },
        os: 0,
        cache_ram: 0,
        lock: false,
        cards: Vec::new(),
        pool: record::PoolPlace {
            threads: pool.threads,
            cpus: pool.workers,
            caller: pool.caller,
            allowed: pool.allowed,
        },
    };
    record::need(&need).eprint();
    start::note_need(bytes, &[], bytes, 0);
    parallel_line(slots, ctx as usize, from, Pass::One);
    if a.plan_only {
        // The records before the load are out; nothing was opened.
        start::plan_exit();
    }
    // The load: the mapping prefaulted once (a server decodes for hours; a lazy
    // mapping would take a page fault on every expert it first meets), the
    // derived weights and the cache made at the open.
    let t = Instant::now();
    let gguf =
        gguf::Gguf::open_with(&path, true).map_err(|e| format!("{}: {e}", path.display()))?;
    let at = u32::try_from(ctx).map_err(|_| format!("--ctx {ctx} passes u32"))?;
    let target = HostTarget::new(gguf, Some(at)).map_err(|e| format!("{}: {e}", path.display()))?;
    let head_rows = target.head_rows();
    Record::new(&record::LOAD_HOST)
        .w("arch", Arch::Deepseek2.name())
        .u("resident_bytes", bytes)
        .u("ctx", ctx)
        .u("slots", 1)
        .u("layers", layers)
        .f("load_s", t.elapsed().as_secs_f64())
        .eprint();
    let tokenizer = Tokenizer::from_gguf(&path)?;
    let vocab = Arc::new(bind::Vocab::new(tokenizer)?);
    let engine = TargetEngine::new(
        target,
        head_rows,
        vocab,
        format!("{NAME} {}", Arch::Deepseek2.name()),
        EngineProps::default(),
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let server = bind_server(
        engine,
        Listen {
            host: &a.host,
            port: a.port,
            alias: a.alias.unwrap_or(chat.name),
            path: &path,
            template: chat.template,
            preparation: None,
            slot_save_path: None,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
            schema_jump: levers.schema_jump(),
            side: None,
        },
    )?;
    Record::new(&record::LISTENING_HOST)
        .w("place", "host")
        .u("ctx", ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
