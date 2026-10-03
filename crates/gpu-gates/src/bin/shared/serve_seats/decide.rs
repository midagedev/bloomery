//! `bloomery-serve --model decide` — a decision model's HTTP API: one prompt
//! pass a request through the backbone (the model file, whole on one card),
//! the model's head on the host, an answer body out (`serve::decide`).
//!
//!     bloomery-serve [--model decide] (-m PATH | --hf <repo>[:<quant>])
//!                    [--head <weights> [--head-config <file>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C]
//!
//! A decision model is named by its head (`serve::decide::pick`): `--head`,
//! or, under `--hf` with no `--head`, the head repo of the row whose model
//! card the `--hf` repo's card names as the model it quantizes
//! (`model_file::quantized_from`), fetched into the cache with its config.
//! The head's config (`--head-config`, else the row's config file beside
//! the weights) picks the row of [`ROWS`]; a config no row knows is refused
//! by name, listing the rows. The server takes `-m`/`--hf` out before this
//! seat parses (its module doc); `--ctx-size` is `--ctx` under
//! llama-server's spelling, and C defaults to the row's context.
//!
//! Before any load the seat refuses by name a file whose architecture is
//! not one of the row's backbones; then, on a worker thread that owns the
//! card, it opens the head, refuses a head whose width is not the file's
//! `embedding_length`, opens the tokenizer and the backbone ([`BODIES`]),
//! and prints one line on stdout, `bloomery-serve: listening on http://H:P
//! (model …, quant …, head …, row …)`, P the bound port (`--port 0` binds a
//! free one). Requests are POSTed to the row's routes and answered one at a
//! time by the row's own part ([`Decision`]), which runs the backbone
//! through [`Backbone::hidden`] (from a reset). `/props` carries the
//! server's `engine` object, as every seat's (its version names the build's
//! commit), and names the model and head file names, the quant (the GGUF's
//! `general.file_type`, or the census of its tensor types when the file
//! names none), the row and its routes. An unknown flag or a missing value
//! is refused by name; any other flag given twice takes its last value.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
use bloomery_gpu::arch::qwen3moe::{PrefillPath, Qwen35moeModel};
use bloomery_gpu_gates::{GateError, model_file, ref_model_path};
use decision::release;
use gguf::Split;
use serde_json::{Value, json};
use serve::ServeError;
use serve::decide::{Decide, DecideError, DecideServer, Decided, HeadFrom, Row, Seated};
use tokenizer::Tokenizer;

use crate::clef_seat;

const NAME: &str = "bloomery-serve";

const USAGE: &str = "usage: bloomery-serve [--model decide] (-m PATH | --hf <repo>[:<quant>]) \
                     [--head <weights> [--head-config <file>]] [--host H] [--port P] [--ctx C]";

/// A row's own part, opened from its head's weights and config.
pub type Open = fn(&Path, &Path) -> Result<Box<dyn Decision>, GateError>;

/// Where this seat's head comes from: `serve::decide::pick` over [`ROWS`].
pub type Pick = HeadFrom<'static, Open>;

/// The decision models this seat serves, one row each.
pub const ROWS: &[Row<Open>] = &[Row {
    name: release::NAME,
    routes: release::ROUTES,
    ctx: release::CTX,
    backbones: release::BACKBONES,
    head_repo: release::HEAD_REPO,
    head_file: release::HEAD_FILE,
    config_file: release::HEAD_CONFIG,
    knows: release::knows,
    unserved: release::UNSERVED,
    open: clef_seat::open,
}];

/// A decision model's own part, opened: its head, and how a request body
/// becomes an answer.
pub trait Decision {
    /// The hidden width the head reads.
    fn hidden_size(&self) -> usize;
    /// The answer to one request body; the backbone is the seat's.
    fn decide(&mut self, body: &str, on: &mut Backbone) -> Result<Decided, DecideError>;
}

/// The backbone's hidden-state call, which a body with a `prefill_hidden`
/// gives.
pub trait Hidden {
    /// Back to position 0.
    fn reset(&mut self) -> Result<(), GpuError>;
    /// Every position's final-norm hidden state of `ids` from where the
    /// model stands, `ids.len()` rows of its width.
    fn prefill_hidden(&mut self, ids: &[u32]) -> Result<Vec<f32>, GpuError>;
}

impl Hidden for Qwen35moeModel {
    fn reset(&mut self) -> Result<(), GpuError> {
        Qwen35moeModel::reset(self)
    }

    fn prefill_hidden(&mut self, ids: &[u32]) -> Result<Vec<f32>, GpuError> {
        Qwen35moeModel::prefill_hidden(self, ids, PrefillPath::Auto)
    }
}

/// A backbone body opened from a file: its hidden-state call and the file.
type OpenBody = fn(&Path, usize) -> Result<(Box<dyn Hidden>, Split), GateError>;

/// The bodies a backbone can be, by file architecture.
const BODIES: &[(&str, OpenBody)] = &[("qwen35", open_qwen35)];

/// A `qwen35` file whole on one card, prompt ubatches of [`UBATCH`] ids
/// clipped to the context.
fn open_qwen35(path: &Path, ctx: usize) -> Result<(Box<dyn Hidden>, Split), GateError> {
    let (model, split) = crate::qwen35_open::open(path, ctx, UBATCH.min(ctx))?;
    Ok((Box::new(model), split))
}

/// What the seat opened from the model file, lent to the row's own part
/// for each request.
pub struct Backbone {
    body: Box<dyn Hidden>,
    /// The model file (the head's output rows read it).
    pub file: Split,
    pub tokenizer: Tokenizer,
    /// The context the cache was sized for.
    pub ctx: usize,
    /// The server's name for the model (the `--hf` repo, else the file
    /// name): the `model` an answer carries, as llama.cpp's server answers.
    pub name: String,
}

impl Backbone {
    /// Every position's final-norm hidden state of `ids`, from a reset, and
    /// the prompt pass's wall in ms (the reset left out).
    pub fn hidden(&mut self, ids: &[u32]) -> Result<(Vec<f32>, f64), GpuError> {
        self.body.reset()?;
        let t = Instant::now();
        let hidden = self.body.prefill_hidden(ids)?;
        Ok((hidden, t.elapsed().as_secs_f64() * 1e3))
    }
}

struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
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
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    Ok(a)
}

/// `llama_ftype`'s names for the values `general.file_type` takes.
fn ftype_name(v: u64) -> Option<&'static str> {
    Some(match v {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        30 => "IQ4_XS",
        32 => "BF16",
        _ => return None,
    })
}

/// The quant word `/props` gives: `general.file_type`'s name (a value with
/// no name is given as `file_type <n>`), else the census of the tensor
/// types, `<type>×<count>` by type.
fn quant(split: &Split) -> String {
    if let Some(v) = split
        .value("general.file_type")
        .and_then(gguf::Value::as_unsigned)
    {
        return ftype_name(v).map_or_else(|| format!("file_type {v}"), str::to_owned);
    }
    let mut census = BTreeMap::new();
    for (_, t) in split.iter_tensors() {
        *census.entry(t.ty.to_string()).or_insert(0usize) += 1;
    }
    census
        .iter()
        .map(|(ty, n)| format!("{ty}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn base_name(p: &Path) -> String {
    p.file_name().map_or_else(
        || p.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// What the worker opens: the model file, its name, architecture and width,
/// the head's two files, the row's `open` and the context.
struct Load {
    model: PathBuf,
    name: String,
    arch: String,
    width: u64,
    head: PathBuf,
    config: PathBuf,
    open: Open,
    ctx: usize,
}

type Job = (String, Sender<Result<Decided, DecideError>>);

/// The server's side: requests go to the worker thread that owns the card,
/// one at a time.
struct Seat {
    jobs: Sender<Job>,
    props: Value,
}

impl Decide for Seat {
    fn decide(&mut self, body: &str) -> Result<Decided, DecideError> {
        let (tx, rx) = mpsc::channel();
        let gone = || DecideError::Engine("the backbone's worker thread has ended".to_owned());
        self.jobs.send((body.to_owned(), tx)).map_err(|_| gone())?;
        rx.recv().map_err(|_| gone())?
    }

    fn props(&self) -> Value {
        self.props.clone()
    }
}

/// Opens the head, checks its width, opens the tokenizer and the backbone on
/// this thread, reports the open (the quant word, or why not) on `ready`,
/// then answers `jobs` until the server drops its sender.
fn work(l: &Load, ready: &Sender<Result<String, String>>, jobs: &Receiver<Job>) {
    let opened = (|| -> Result<(Box<dyn Decision>, Backbone, String), GateError> {
        let head = (l.open)(&l.head, &l.config)?;
        if u64::try_from(head.hidden_size())? != l.width {
            return Err(format!(
                "{}: the head reads hidden states of width {}, and the backbone {} gives {}",
                l.head.display(),
                head.hidden_size(),
                l.model.display(),
                l.width
            )
            .into());
        }
        let tokenizer = Tokenizer::from_gguf(&l.model)?;
        let (_, open_body) = BODIES
            .iter()
            .find(|(arch, _)| *arch == l.arch)
            .ok_or_else(|| format!("no backbone body opens a {} file", l.arch))?;
        let (body, file) = open_body(&l.model, l.ctx)?;
        let q = quant(&file);
        let on = Backbone {
            body,
            file,
            tokenizer,
            ctx: l.ctx,
            name: l.name.clone(),
        };
        Ok((head, on, q))
    })();
    let (mut head, mut on) = match opened {
        Ok((head, on, q)) => {
            let _ = ready.send(Ok(q));
            (head, on)
        }
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    for (body, reply) in jobs {
        let _ = reply.send(head.decide(&body, &mut on));
    }
}

/// The head's weights and config `from` names, with its row: `--head` and
/// the row its config picks, or the row's head fetched from its repo,
/// whose config that row must know.
fn head_files(from: Pick) -> Result<(&'static Row<Open>, PathBuf, PathBuf), GateError> {
    match from {
        HeadFrom::Given { head, config } => {
            let (row, config) = serve::decide::row_of_config(ROWS, &head, config.as_deref())?;
            Ok((row, head, config))
        }
        HeadFrom::Fetch(row) => {
            let mut files =
                model_file::fetch_exact(row.head_repo, &[row.head_file, row.config_file])?
                    .into_iter();
            let (Some(head), Some(config)) = (files.next(), files.next()) else {
                return Err(format!(
                    "{}: the head's fetch returned fewer than two files",
                    row.head_repo
                )
                .into());
            };
            let text = std::fs::read_to_string(&config)
                .map_err(|e| format!("{}: {e}", config.display()))?;
            (row.knows)(&text).map_err(|e| {
                format!(
                    "{}: the {} row does not know its own repo's head config: {e}",
                    config.display(),
                    row.name
                )
            })?;
            Ok((row, head, config))
        }
    }
}

/// Opens the model `from` names and serves until the listener fails; `Ok`
/// carries why the server ended. `hf` is the repo (`owner/name`) the model
/// file came from under `--hf`: the model's name, else its file name.
pub fn run(args: &[String], from: Pick, hf: Option<&str>) -> Result<ServeError, GateError> {
    bloomery_levers::at_main(&[])?;
    let a = parse_args(args)?;
    let (row, head, config) = head_files(from)?;
    let model = ref_model_path()?;
    let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
    let arch = split.architecture().unwrap_or("<missing>").to_owned();
    if !row.backbones.contains(&arch.as_str()) {
        return Err(format!(
            "{} is a {arch} file, and the {} row's head reads the hidden states of {}",
            model.display(),
            row.name,
            row.backbones.join(" or ")
        )
        .into());
    }
    let width = split
        .arch_get_u64("embedding_length")
        .ok_or_else(|| format!("{}: no {arch}.embedding_length", model.display()))?;
    drop(split);
    let (model_name, head_name) = (base_name(&model), base_name(&head));
    let name = hf.map_or_else(|| model_name.clone(), str::to_owned);
    let addr = format!("{}:{}", a.host, a.port);
    let ctx = a.ctx.unwrap_or(row.ctx);
    let load = Load {
        model,
        name: name.clone(),
        arch,
        width,
        head,
        config,
        open: row.open,
        ctx,
    };
    let (ready_tx, ready_rx) = mpsc::channel();
    let (jobs, jobs_rx) = mpsc::channel::<Job>();
    let worker = std::thread::Builder::new()
        .name("decide-backbone".to_owned())
        .spawn(move || work(&load, &ready_tx, &jobs_rx))?;
    let quant = match ready_rx.recv() {
        Ok(Ok(q)) => q,
        Ok(Err(e)) => return Err(e.into()),
        Err(_) => {
            return Err(match worker.join() {
                Err(_) => "the backbone's worker thread panicked while opening".into(),
                Ok(()) => "the backbone's worker thread ended before opening".into(),
            });
        }
    };
    let props = json!({
        "model": model_name,
        "quant": quant,
        "head": head_name,
        "row": row.name,
        "routes": row.routes,
    });
    let seated = Seated {
        routes: row.routes,
        name,
        n_ctx: ctx,
    };
    let server = DecideServer::bind(addr.as_str(), &seated, Box::new(Seat { jobs, props }))
        .map_err(|e| format!("bind {addr}: {e}"))?;
    // The bound address: `--port 0` binds a free port.
    let bound = server.local_addr()?;
    println!(
        "{NAME}: listening on http://{bound} (model {model_name}, quant {quant}, head {head_name}, row {})",
        row.name
    );
    Ok(ServeError::Io(server.run()))
}
