//! `bloomery-serve --model qwen3` — the llama-server-compatible HTTP API on
//! the single-card Qwen engines: a qwen3moe file (Qwen3-30B-A3B) or a
//! qwen35moe file (Qwen3.6-35B-A3B), the whole model on device 0.
//!
//!     bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` is `--ctx` under llama-server's spelling, C defaults to
//! 4096, `generate_qwen3moe`'s default. The model is opened as
//! `generate_qwen3moe` opens it — qwen3moe through `Qwen3moeModel::open` at
//! its levers' options, qwen35moe through `Qwen35moeModel::open` with the
//! tensor-core decode flash and the levers' ubatch — in graph mode with the
//! step and every pass captured, so a request's ids are the CLI's: the
//! server feeds the prompt less its last id as one prompt call
//! (`prefill_with`, `auto`: passes up to eight ids, ubatches past that),
//! then steps the last, which `generate_qwen3moe --last-step` does too. Its
//! first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template and `general.name` the alias. One slot; sampling is the sampler
//! crate's chain with no repetition penalty, `temperature <= 0` the
//! engine's argmax.
//!
//! The bodies hold no rollback and the seat no prompt cache: every request
//! prefills its whole prompt from a reset, and a request that shares a
//! prefix with the last is a `cache reuse` note of why it kept none. The
//! `load` record (architecture, resident bytes, context, layers, ubatch,
//! the step graph's nodes, the load's wall) then the `listening` record go
//! to stderr (`record::BLOOMERY_SERVE_QWEN3`). An engine error ends the
//! process with the crash block and exit code 70, as every seat's.
//!
//! The seat sits behind the `deepseek41` feature, the server surface's
//! scope (`bind`, the `serve` and `sampler` crates); it runs no V4.1 code
//! and acts on no parsed lever (`BLOOMERY_QWEN3_UBATCH` is read where the
//! load sizes its arena).

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::Gpu;
use bloomery_gpu::Qwen3moeModel;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::arch::qwen3moe::{Body, Body35, Open35, PrefillPath, Qwen35moeModel};
use bloomery_gpu::model::{ChainBody, GpuModel, StepMode};
use bloomery_gpu_gates::bind::{Seat, SeatEngine, Vocab, sampler_factory};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, ref_model_path};
use gguf::Split;
use model::arch::Arch;
use serve::{CacheNote, EngineProps, FATAL_LINGER, Saved, ServeError, Server, ServerConfig};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-qwen3";

const USAGE: &str = "usage: bloomery-serve --model qwen3 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C]";

/// The context unless `--ctx` says: `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// Why the seat keeps no prefix of the last request.
const NO_KEEP: &str = "the qwen3 seat's bodies hold no rollback: every request prefills from a \
                       reset";

struct Args {
    host: String,
    port: u16,
    ctx: usize,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: CTX,
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
                Ok(n) if n > 0 => a.ctx = n,
                _ => {
                    return Err(
                        format!("{flag} takes a whole number of at least 1, not {v:?}").into(),
                    );
                }
            },
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    Ok(a)
}

/// One of the two bodies the seat serves, opened as `generate_qwen3moe`
/// opens it.
trait Body3: ChainBody + Sized + 'static {
    /// The file's `general.architecture`, as the `load` record names it.
    const ARCH: &'static str;
    /// The model of `file` on device 0 with a `ctx`-row cache.
    fn open(file: Split, ctx: usize) -> Result<GpuModel<Self>, GateError>;
    /// Every pass the prompt call replays, captured after the step.
    fn capture_passes(m: &mut GpuModel<Self>) -> Result<(), GateError>;
    /// The prompt call's ubatch size.
    fn ubatch(m: &GpuModel<Self>) -> Result<usize, GateError>;
    /// `ids` from where the model stands by `auto`; the argmax after the
    /// last.
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32]) -> Result<u32, GateError>;
}

impl Body3 for Body {
    const ARCH: &'static str = "qwen3moe";

    fn open(file: Split, ctx: usize) -> Result<Qwen3moeModel, GateError> {
        Ok(Qwen3moeModel::open(
            Gpu::new()?,
            file,
            Qwen3moeModel::lever_opts(ctx)?,
        )?)
    }

    fn capture_passes(m: &mut Qwen3moeModel) -> Result<(), GateError> {
        m.capture_prefill()?;
        Ok(())
    }

    fn ubatch(m: &Qwen3moeModel) -> Result<usize, GateError> {
        Ok(m.ubatch()?)
    }

    fn prompt(m: &mut Qwen3moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(m.prefill_with(ids, PrefillPath::Auto)?)
    }
}

impl Body3 for Body35 {
    const ARCH: &'static str = "qwen35moe";

    fn open(file: Split, ctx: usize) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx,
            mma: true,
            ubatch: ubatch_size()?,
        };
        Ok(Qwen35moeModel::open(Gpu::new()?, file, o)?)
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

    fn prompt(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<u32, GateError> {
        Ok(m.prefill_with(ids, PrefillPath::Auto)?)
    }
}

/// The model on the engine thread and the positions its cache was sized
/// for.
struct Q3<B: Body3> {
    m: GpuModel<B>,
    ctx: usize,
}

impl<B: Body3> Q3<B> {
    /// The model of the file at `path` in graph mode, its step and passes
    /// captured; the `load` record on stderr.
    fn open(path: &Path, ctx: usize) -> Result<Q3<B>, GateError> {
        let t = Instant::now();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut m = B::open(file, ctx)?;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        B::capture_passes(&mut m)?;
        Record::new(&record::LOAD_QWEN3)
            .w("arch", B::ARCH)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", ctx)
            .u("layers", m.layers().len())
            .u("ubatch", B::ubatch(&m)?)
            .u("graph_nodes", nodes)
            .f("load_s", t.elapsed().as_secs_f64())
            .eprint();
        Ok(Q3 { m, ctx })
    }
}

impl<B: Body3> Seat for Q3<B> {
    fn pos(&self) -> usize {
        self.m.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        B::prompt(&mut self.m, ids)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.m.step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.m.logits_into(row)?)
    }

    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    /// Never asked: [`Seat::keep`] grants no prefix.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Err(format!("a rollback to position {pos}: {NO_KEEP}").into())
    }

    fn keep(&self, n: usize) -> (usize, Option<String>) {
        (0, (n > 0).then(|| NO_KEEP.to_owned()))
    }

    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_KEEP}, and the seat runs no prompt cache").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_KEEP}, and the seat runs no prompt cache").into())
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
    bloomery_levers::at_main(&[])?;
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
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let ctx = a.ctx;
    let open = path.clone();
    let engine = match arch {
        Arch::Qwen3moe => SeatEngine::spawn(
            move || Q3::<Body>::open(&open, ctx),
            ctx,
            vocab,
            "device 0".to_owned(),
            EngineProps::default(),
            0,
        )?,
        Arch::Qwen35moe => SeatEngine::spawn(
            move || Q3::<Body35>::open(&open, ctx),
            ctx,
            vocab,
            "device 0".to_owned(),
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
    let server = Server::bind((a.host.as_str(), a.port), Box::new(engine), config)?;
    Record::new(&record::LISTENING_QWEN3)
        .w("arch", arch.name())
        .u("ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
