//! `bloomery_serve_clef` — Clef answering SystemOne requests over HTTP: the
//! decision server of `serve::decide` on Clef's backbone (a `qwen35` file on
//! one card) and its joint schema head (`crates/decision`, on the host).
//!
//!     bloomery_serve_clef (--model <gguf> | --hf <repo>[:<quant>])
//!                         [--head <joint_head.safetensors>] [--head-config <json>]
//!                         [--host H] [--port P] [--ctx C]
//!     bloomery_serve_clef --version
//!
//! The backbone is `--model` (`-m`, `--model-file`), or `--hf`: the repo's
//! GGUF set of that quant tag fetched into the cache (`$BLOOMERY_CACHE`, else
//! `~/.cache/bloomery/hf`; `bloomery_gpu_gates::model_file`, its `hf`
//! records on stderr), e.g. `--hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M`.
//! Under `--hf` with no `--head`, the head and its config
//! (`joint_head.safetensors`, `joint_head_config.json`) are fetched from
//! the release's repo, `Cloudflare/clef-flash`; without `--hf`, `--head` is required.
//! The model named twice (two of its spellings, or one beside `--hf`) is
//! refused by name. `--version` prints this crate's version and the build's
//! commit and exits.
//!
//! Defaults: the head's config `joint_head_config.json` beside `--head`, H
//! `127.0.0.1`, P 8091, C 16384 (the release's `max_length`); prompt ubatches
//! of `UBATCH` ids clipped to C, as `clef_hidden`. Startup opens the backbone
//! and the head once, on a worker thread that owns the card, refuses a head
//! whose `hidden_size` is not the backbone's width, then binds and prints one
//! line, `bloomery_serve_clef: listening on http://H:P (model …, quant …)`,
//! P the bound port (`--port 0` binds a free one).
//!
//! Each request: the decision crate parses, validates and encodes it at the
//! release's `max_length` (an encoding past C is a 400 by name), the backbone
//! runs it from a `reset` (`prompt_ms`), and the head reads the hidden states
//! and the option spans' output rows into the answer body (`head_ms`). A
//! request the decision crate refuses is a 400 with its message; a backbone
//! or head failure is a 500. `/props` names the engine, the build's commit, the
//! model and head file names, and the quant: the GGUF's `general.file_type`,
//! or the census of its tensor types when the file names none. Any other
//! flag given twice takes its last value; an unknown flag or a missing value
//! is refused by name.

#[cfg(not(feature = "clef"))]
fn main() {
    eprintln!("bloomery_serve_clef: built without the `clef` feature; see the module doc.");
    std::process::exit(2);
}

#[cfg(feature = "clef")]
fn main() -> std::process::ExitCode {
    if std::env::args().skip(1).any(|a| a == "--version") {
        println!(
            "{}",
            bloomery_gpu_gates::model_file::version("bloomery_serve_clef")
        );
        return std::process::ExitCode::SUCCESS;
    }
    bloomery_gpu_gates::exit_with("bloomery_serve_clef", run::run())
}

#[cfg(feature = "clef")]
#[path = "shared/clef.rs"]
#[allow(
    dead_code,
    reason = "the server opens the backbone only; the ids reader serves clef_hidden"
)]
mod clef;

#[cfg(feature = "clef")]
mod run {
    use super::clef;
    use bloomery_gpu::arch::qwen3moe::PrefillPath;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu_gates::{GateError, model_file};
    use decision::Error as DError;
    use decision::answer::answer;
    use decision::encode::{MAX_LENGTH, encode};
    use decision::head::ClefHead;
    use decision::render::dumps;
    use decision::request::Request;
    use decision::rows::output_rows;
    use gguf::Split;
    use serde_json::{Value, json};
    use serve::decide::{COMMIT, Decide, DecideError, DecideServer, Decided};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::time::Instant;
    use tokenizer::Tokenizer;

    /// The repo `--hf` fetches the joint head from when `--head` is not
    /// given: the release's.
    const HEAD_REPO: &str = "Cloudflare/clef-flash";

    /// The head's files in [`HEAD_REPO`], the weights first.
    const HEAD_FILES: &[&str] = &["joint_head.safetensors", "joint_head_config.json"];

    /// The spellings of the backbone's path flag.
    const MODEL_FLAGS: &[&str] = &["--model", "-m", "--model-file"];

    struct Args {
        model: PathBuf,
        head: PathBuf,
        head_config: Option<PathBuf>,
        host: String,
        port: u16,
        ctx: usize,
    }

    fn parse() -> Result<Args, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (flags, rest) = model_file::take(&args, MODEL_FLAGS)?;
        let mut a = rest.into_iter();
        let (mut head, mut head_config) = (None, None);
        let (mut host, mut port, mut ctx) = ("127.0.0.1".to_owned(), 8091u16, MAX_LENGTH);
        while let Some(flag) = a.next() {
            let value = a
                .next()
                .ok_or_else(|| format!("{flag}: a value is due after it"))?;
            match flag.as_str() {
                "--head" => head = Some(PathBuf::from(value)),
                "--head-config" => head_config = Some(PathBuf::from(value)),
                "--host" => host = value,
                "--port" => {
                    port = value
                        .parse()
                        .map_err(|e| format!("--port {value:?}: {e}"))?;
                }
                "--ctx" => match value.parse::<usize>() {
                    Ok(n) if n > 0 => ctx = n,
                    _ => {
                        return Err(format!(
                            "--ctx takes a whole number of at least 1, not {value:?}"
                        )
                        .into());
                    }
                },
                _ => return Err(format!("unknown flag {flag:?}").into()),
            }
        }
        if head.is_none() && flags.hf.is_none() {
            return Err(
                format!("--head is required, unless --hf fetches it from {HEAD_REPO}").into(),
            );
        }
        let model = model_file::resolve(&flags, None)?.ok_or_else(|| {
            GateError::from("the backbone is required: --model PATH (-m) or --hf <repo>[:<quant>]")
        })?;
        let head = match head {
            Some(h) => h,
            None => model_file::fetch_exact(HEAD_REPO, HEAD_FILES)?
                .into_iter()
                .next()
                .ok_or("the head's fetch returned no file")?,
        };
        Ok(Args {
            model,
            head,
            head_config,
            host,
            port,
            ctx,
        })
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

    /// The quant word `/props` gives: `general.file_type`'s name (a value
    /// with no name is given as `file_type <n>`), else the census of the
    /// tensor types, `<type>×<count>` by type.
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

    /// Whether a decision crate error is the request's (a 400) or the
    /// engine's (a 500).
    fn classify(e: DError) -> DecideError {
        match e {
            DError::Json { .. }
            | DError::JsonTooDeep(_)
            | DError::JsonNonFinite(_)
            | DError::JsonFloatRange(_)
            | DError::JsonLoneSurrogate(_)
            | DError::Request(_)
            | DError::Media(_)
            | DError::Question { .. }
            | DError::EmptySpan { .. }
            | DError::SchemaTooLong { .. } => DecideError::Refused(e.to_string()),
            _ => DecideError::Engine(e.to_string()),
        }
    }

    /// The card's side: the backbone, the head and the file they read.
    struct Worker {
        engine: bloomery_gpu::arch::qwen3moe::Qwen35moeModel,
        split: Split,
        tokenizer: Tokenizer,
        head: ClefHead,
        ctx: usize,
    }

    impl Worker {
        fn decide(&mut self, body: &str) -> Result<Decided, DecideError> {
            let req = Request::from_json(&decision::json::parse(body).map_err(classify)?)
                .map_err(classify)?;
            let enc = encode(&self.tokenizer, &req, MAX_LENGTH).map_err(classify)?;
            if enc.ids.len() > self.ctx {
                return Err(DecideError::Refused(format!(
                    "the request encodes to {} tokens, past this server's context of {}",
                    enc.ids.len(),
                    self.ctx
                )));
            }
            let engine = |e: bloomery_gpu::GpuError| DecideError::Engine(format!("backbone: {e}"));
            self.engine.reset().map_err(engine)?;
            let t0 = Instant::now();
            let hidden = self
                .engine
                .prefill_hidden(&enc.ids, PrefillPath::Auto)
                .map_err(engine)?;
            let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let split = &self.split;
            let mut stages = Vec::new();
            let logits = self
                .head
                .forward_staged(
                    &hidden,
                    &enc,
                    &mut |ids| output_rows(split, ids).map_err(|e| e.to_string()),
                    &mut stages,
                )
                .map_err(classify)?;
            let body = dumps(&answer(&req, &enc, &logits).map_err(classify)?, false);
            eprintln!(
                "bloomery_serve_clef: n={} prompt={prompt_ms:.1} head {} answer+={:.1}",
                enc.ids.len(),
                decision::head::stage_line(&stages),
                t1.elapsed().as_secs_f64() * 1e3 - stages.iter().map(|s| s.1).sum::<f64>()
            );
            Ok(Decided {
                body,
                prompt_n: enc.ids.len(),
                prompt_ms,
                head_ms: t1.elapsed().as_secs_f64() * 1e3,
            })
        }
    }

    type Job = (String, Sender<Result<Decided, DecideError>>);

    /// The server's side: requests go to the worker thread that owns the
    /// card, one at a time.
    struct Clef {
        jobs: Sender<Job>,
        props: Value,
    }

    impl Decide for Clef {
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

    /// Opens the backbone and the head on this thread, reports the open (the
    /// quant word, or why not) on `ready`, then answers `jobs` until the
    /// server drops its sender.
    fn work(a: &Args, ready: &Sender<Result<String, String>>, jobs: &Receiver<Job>) {
        let opened = (|| -> Result<(Worker, String), GateError> {
            let head = ClefHead::open(&a.head, a.head_config.as_deref())?;
            let tokenizer = Tokenizer::from_gguf(&a.model)?;
            let (engine, split) = clef::open(&a.model, a.ctx, UBATCH.min(a.ctx))?;
            let width = split.arch_get_u64("embedding_length");
            if width != Some(u64::try_from(head.config().hidden_size)?) {
                return Err(format!(
                    "{}: the backbone's width is {width:?}, and the head reads {}",
                    a.model.display(),
                    head.config().hidden_size
                )
                .into());
            }
            let q = quant(&split);
            Ok((
                Worker {
                    engine,
                    split,
                    tokenizer,
                    head,
                    ctx: a.ctx,
                },
                q,
            ))
        })();
        let mut w = match opened {
            Ok((w, q)) => {
                let _ = ready.send(Ok(q));
                w
            }
            Err(e) => {
                let _ = ready.send(Err(e.to_string()));
                return;
            }
        };
        for (body, reply) in jobs {
            let _ = reply.send(w.decide(&body));
        }
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let a = parse()?;
        let (model, head) = (base_name(&a.model), base_name(&a.head));
        let addr = format!("{}:{}", a.host, a.port);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (jobs, jobs_rx) = mpsc::channel::<Job>();
        let worker = std::thread::Builder::new()
            .name("clef-backbone".to_owned())
            .spawn(move || work(&a, &ready_tx, &jobs_rx))?;
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
            "engine": "bloomery",
            "build": COMMIT,
            "model": model,
            "quant": quant,
            "head": head,
        });
        let server = DecideServer::bind(addr.as_str(), Box::new(Clef { jobs, props }))
            .map_err(|e| format!("bind {addr}: {e}"))?;
        // The bound address: `--port 0` binds a free port.
        let bound = server.local_addr()?;
        println!("bloomery_serve_clef: listening on http://{bound} (model {model}, quant {quant})");
        Err(format!("serving {bound}: {}", server.run()).into())
    }
}
