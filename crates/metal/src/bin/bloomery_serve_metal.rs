//! `bloomery-serve-metal` — the llama-server-compatible HTTP API on this
//! Mac's Metal device: a model opened as `generate_metal` opens it, served
//! by `serve::TargetEngine`.
//!
//!     bloomery-serve-metal (-m PATH | --hf REPO) [--host 127.0.0.1]
//!                          [--port 8080] [--ctx C]
//!                          [--chat-template-file PATH] [--alias NAME]
//!                          [--queue-depth Q] [--api-key KEY] [--api-key-file F]
//!                          [--draft lookup | --draft mtp --mtp SIDECAR]
//!
//! The model is a GGUF file's first shard or a checkpoint folder (an MLX
//! folder as downloaded, no `prepare`); `--hf REPO` fetches a repo into the
//! cache and opens it (`bloomery_metal::model_arg`). `--ctx-size` and `-c`
//! are `--ctx` under llama-server's spellings; unset, the cache holds the
//! smaller of 32768 rows and the file's `context_length`, cut to the largest
//! cache whose session fits the device's free bytes (a line on stderr names
//! the cut); an asked count that does not fit is refused by name. The file
//! gives the vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and the alias (`--alias` replaces
//! it): `general.name`, else the folder's name, else the repo's, else the
//! architecture's.
//! Sampling is the sampler crate's chain; `temperature <= 0` is the
//! session's argmax, the ids `generate_metal --tokens` picks for the same
//! prompt ids.
//!
//! One slot: the session is one sequence. A request keeps a prefix back to a
//! checkpoint of the delta layers' recurrent state: a prompt call takes one
//! at its start, every `CHECKPOINT_EVERY` positions, at each message start
//! the server marks inside it (`Target::prompt_marked`: a copy the pass
//! leaves at that row, the call uncut) and at its end. Any other prefix runs
//! from position 0. A prompt span of more than a step's window runs in passes
//! of up to `Session::pass_rows` ids, a shorter one a host-fed step an id. The
//! session opens the copies a marked pass leaves with its rows arena
//! (`MarkCopies::AtOpen`), so the first request pays nothing the later ones
//! do not. No prompt cache, no slot files, no image input.
//!
//! `--draft mtp --mtp SIDECAR` serves with the MTP draft of a Qwen3.5-family
//! file (`generate_metal`'s flags and sidecar, `mlx-community/Qwen3.8-27B-MTP-4bit`):
//! `serve::TargetEngine::drafted` over `runtime::mtp::MtpDraft` with the session
//! as its host behind the width chooser (`BLOOMERY_MTP_WIDTH`, `cost` unset), two
//! rows a pass. A greedy request is verified in passes and keeps the ids the
//! plain server picks, bit for bit. The draft joins a sequence from position 0 or
//! rejoins it at a checkpoint: a prompt call keeps the target's hidden row of the
//! position before each point it takes, so a request that cuts back to a
//! checkpoint keeps drafting (`runtime::mtp::PointRows`, written to the step arena
//! by the cut). A step that reads the logits row (a request that samples) leaves
//! the draft off until the next reset, and a cut to a position whose checkpoint
//! holds no row does too; a line on stderr names why.
//!
//! `--draft lookup` serves with the n-gram lookup draft (`runtime::Lookup`, the
//! one `generate_metal --draft lookup` runs) behind the same chooser and rows: no
//! sidecar, no weights, no device bytes. Its context follows the target by
//! position, so a cut back and a request that samples leave it drafting. The ids
//! are the plain server's, bit for bit.
//!
//! `/props`' `engine.draft` names the draft; each request's passes are one
//! `draft:` line when the next request opens. `--draft mtp` without `--mtp`,
//! `--mtp` without `--draft mtp`, a `--draft` that is neither, and
//! `BLOOMERY_MTP_WIDTH` beside no draft are refused by name, and so is a file the
//! verify pass does not serve (at open, before the server listens). DSpark has no
//! Metal draft model: `--draft dspark` is refused like any other word.
//!
//! `BLOOMERY_SCREEN` (on unset) lets `d` open the server's live view on a
//! terminal (`serve::live`); the start screen is the CUDA seats' alone.
//!
//! A bad flag or value is a named error and exit 64; a refusal of the file,
//! the device or the session is a named error and exit 1; an engine failure
//! while serving ends the run with exit 70. The body is macOS only;
//! elsewhere it refuses and exits 78.

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    cli::run()
}

#[cfg(target_os = "macos")]
mod cli {
    use std::process::ExitCode;
    use std::sync::Arc;
    use std::time::Instant;

    use bloomery_metal::MetalDevice;
    use bloomery_metal::arch::qwen3moe::plan;
    use bloomery_metal::arch::qwen3moe::session::{
        CHECKPOINT_EVERY, MarkCopies, Session, StepMode,
    };
    use bloomery_metal::model_arg;
    use hf::source::Source;
    use model_hf::ModelFile;
    use runtime::mtp::MtpDraft;
    use runtime::width::{Chosen, Mode as WidthMode, Wall};
    use runtime::{Choosing, Lookup, Speculative};
    use serve::flag::{self, ApiKeys};
    use serve::sampling::sampler_factory;
    use serve::vocab::{ChatSurface, TemplateFile, Vocab};
    use serve::{
        DraftProps, DraftWindow, EngineProps, FATAL_LINGER, ServeError, Server, ServerConfig,
        SlotConfig, TargetEngine,
    };

    /// The cache rows' default cap.
    const CTX_DEFAULT: usize = 32768;

    const USAGE: &str = "\
bloomery-serve-metal (-m PATH | --hf REPO) [--host H] [--port P] [--ctx C] [--chat-template-file PATH]
                     [--alias NAME] [--queue-depth Q] [--api-key KEY] [--api-key-file F]
                     [--draft lookup | --draft mtp --mtp SIDECAR]
  --draft lookup            serves the n-gram lookup draft (no sidecar), two rows a pass
  --draft mtp --mtp SIDECAR serves the MTP draft layer of the sidecar's folder, two rows a pass
  PATH is a GGUF file (its first shard) or a checkpoint folder; REPO is fetched into the cache
  --ctx (--ctx-size, -c) defaults to the smaller of 32768 and the file's context_length,
  cut to the largest that fits the device's free bytes";

    /// The model's path flags; `--hf` is the repo's.
    const PATH_FLAGS: &[&str] = &["-m", "--model"];

    struct Args {
        model: Source,
        host: String,
        port: u16,
        ctx: Option<usize>,
        template_file: TemplateFile,
        alias: Option<String>,
        queue_depth: Option<usize>,
        api_keys: ApiKeys,
        /// `--draft`: which draft the server runs beside the session.
        draft: Option<Draft>,
        /// `--mtp`: the sidecar of `--draft mtp`.
        mtp: Option<std::path::PathBuf>,
        /// `BLOOMERY_MTP_WIDTH`: how many of a proposal's ids a pass verifies.
        width: WidthMode,
    }

    /// The drafts the server runs.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Draft {
        /// `--draft mtp --mtp SIDECAR`: the sidecar's draft layer.
        Mtp,
        /// `--draft lookup`: the context's n-grams.
        Lookup,
    }

    fn parse(width: Option<&str>) -> Result<Args, String> {
        let mut model = Vec::new();
        let mut a = Args {
            model: Source::File(String::new()),
            host: "127.0.0.1".to_owned(),
            port: 8080,
            ctx: None,
            template_file: TemplateFile::default(),
            alias: None,
            queue_depth: None,
            api_keys: ApiKeys::default(),
            draft: None,
            mtp: None,
            width: WidthMode::Cost,
        };
        let mut it = std::env::args().skip(1);
        while let Some(f) = it.next() {
            let mut val = || it.next().ok_or_else(|| format!("{f} needs a value"));
            match f.as_str() {
                "-m" | "--model" | "--hf" => model.push((f.clone(), val()?)),
                "--host" => a.host = val()?,
                "--port" => a.port = flag::number(&f, &val()?)?,
                c if flag::CTX.contains(&c) => a.ctx = Some(flag::number(&f, &val()?)?),
                TemplateFile::FLAG => a.template_file.set(&val()?),
                "--alias" => a.alias = Some(val()?),
                "--queue-depth" => a.queue_depth = Some(flag::number(&f, &val()?)?),
                k if flag::KEYS.contains(&k) => a.api_keys.add(&f, &val()?)?,
                "--draft" => match val()?.as_str() {
                    "mtp" => a.draft = Some(Draft::Mtp),
                    "lookup" => a.draft = Some(Draft::Lookup),
                    other => return Err(format!("--draft {other}: lookup or mtp\n{USAGE}")),
                },
                "--mtp" => a.mtp = Some(val()?.into()),
                "--help" | "-h" => return Err(USAGE.to_owned()),
                other => return Err(format!("unknown flag {other}\n{USAGE}")),
            }
        }
        a.model = model_arg::named(&model, PATH_FLAGS, None, "-m PATH or --hf REPO")
            .map_err(|e| format!("{e}\n{USAGE}"))?;
        match (a.draft, &a.mtp) {
            (Some(Draft::Mtp), None) => {
                return Err(format!(
                    "--draft mtp: the draft layer is the sidecar's; give --mtp SIDECAR\n{USAGE}"
                ));
            }
            (None, Some(_)) => {
                return Err(format!(
                    "--mtp: the server drafts with the sidecar only under --draft mtp\n{USAGE}"
                ));
            }
            (Some(Draft::Lookup), Some(_)) => {
                return Err(format!(
                    "--draft lookup: the lookup has no sidecar; --mtp goes with --draft mtp\n{USAGE}"
                ));
            }
            _ => {}
        }
        if let Some(w) = width
            && a.draft.is_none()
        {
            return Err(format!(
                "{}={w}: it picks the width a drafted window verifies; give --draft lookup or mtp",
                bloomery_levers::MTP_WIDTH
            ));
        }
        a.width = WidthMode::of(width)
            .map_err(|_| format!("{}: cost or fixed", bloomery_levers::MTP_WIDTH))?;
        Ok(a)
    }

    pub(super) fn run() -> ExitCode {
        let levers = match bloomery_levers::at_main(&[
            bloomery_levers::SCREEN,
            bloomery_levers::MTP_WIDTH,
        ]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("bloomery-serve-metal: {e}");
                return ExitCode::from(1);
            }
        };
        serve::live::enable(levers.screen());
        let a = match parse(levers.mtp_width()) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("bloomery-serve-metal: {e}");
                return ExitCode::from(64);
            }
        };
        let server = match open(a) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bloomery-serve-metal: {e}");
                return ExitCode::from(1);
            }
        };
        let e = server.run();
        eprintln!("bloomery-serve-metal: {e}");
        match e {
            // EX_SOFTWARE: the engine, not the listener, ended the run.
            ServeError::Engine(_) => ExitCode::from(70),
            ServeError::Io(_)
            | ServeError::Template(_)
            | ServeError::SlotSavePath(_)
            | ServeError::NoStops
            | ServeError::Slots(_) => ExitCode::from(1),
        }
    }

    /// The MTP draft layer of the sidecar at `path` opened beside `session`, as
    /// the window of two rows behind the width chooser the server drafts with.
    /// The draft reads the step's residual, which a token running ahead of the
    /// step would hold, so the session steps host-fed from here on.
    fn open_draft(
        device: &MetalDevice,
        session: &mut Session,
        path: &std::path::Path,
        width: WidthMode,
    ) -> Result<Speculative<runtime::Choosing<MtpDraft<Session>>, 2>, String> {
        session.verify_serves().map_err(|e| e.to_string())?;
        let opened = session.open_mtp(path).map_err(|e| e.to_string())?;
        let with = device.allocated_bytes().map_err(|e| e.to_string())?;
        eprintln!(
            "bloomery-serve-metal: mtp_open weight_bytes={} kv_bytes={} allocated_bytes={with}",
            opened.weight_bytes, opened.kv_bytes
        );
        session
            .set_argmax_mode(StepMode::HostFed)
            .map_err(|e| e.to_string())?;
        let draft = session
            .mtp_draft()
            .and_then(|d| d.choosing(width).map_err(Into::into))
            .map_err(|e| e.to_string())?;
        Ok(Speculative::new(draft))
    }

    /// The n-gram lookup draft opened beside `session`, as the window of two
    /// rows behind the width chooser by the lookup's own rule
    /// ([`Lookup::RULE`]), as `generate_metal --draft lookup` runs it. The
    /// session steps host-fed from here on: the chooser reads those walls as
    /// the plain step's cost, and a verify drops a token a chained step ran
    /// ahead.
    fn open_lookup(
        session: &mut Session,
        width: WidthMode,
    ) -> Result<Speculative<Choosing<Lookup>, 2>, String> {
        session.verify_serves().map_err(|e| e.to_string())?;
        session
            .set_argmax_mode(StepMode::HostFed)
            .map_err(|e| e.to_string())?;
        let draft = <Lookup as Chosen<Session>>::choosing_by(
            Lookup::new(),
            width,
            Lookup::RULE,
            Wall::new(),
        )
        .map_err(|e| e.to_string())?;
        Ok(Speculative::new(draft))
    }

    /// The session served through `window`, which drafts as `props` names.
    fn drafted<W: DraftWindow<Session> + Send + 'static>(
        session: Session,
        window: W,
        vocab: Arc<Vocab>,
        what: String,
        props: DraftProps,
    ) -> Result<Box<dyn serve::Engine>, String> {
        let head_rows = session.vocab();
        Ok(Box::new(TargetEngine::drafted(
            session,
            window,
            head_rows,
            vocab,
            what,
            EngineProps {
                draft: Some(props),
                ..EngineProps::default()
            },
        )?))
    }

    fn open(a: Args) -> Result<Server, String> {
        let model = model_arg::resolve(&a.model)?;
        let path = &model.path;
        let file = ModelFile::open(path).map_err(|e| e.to_string())?;
        let split = file.header();
        let ctx = match a.ctx {
            Some(c) => c,
            None => {
                let unasked = plan::unasked_ctx(split, CTX_DEFAULT, path)?;
                if let Some(note) = unasked.note() {
                    eprintln!("bloomery-serve-metal: {note}");
                }
                unasked.ctx
            }
        };
        let arch = split
            .architecture()
            .ok_or_else(|| format!("{}: the header names no architecture", path.display()))?
            .to_owned();
        let ChatSurface { template, name } = ChatSurface::read(
            path,
            split.value("tokenizer.chat_template"),
            split.value("general.name"),
            a.template_file.path(),
            &model_arg::default_alias(file.name(), model.repo.as_deref(), &arch),
        )?;
        let tok = file.tokenizer().map_err(|e| e.to_string())?;
        let vocab = Arc::new(Vocab::new(tok)?.with_chat_template(&template)?);
        let instance = Session::instance_of_model(&file, ctx).map_err(|e| e.to_string())?;
        if instance.name == bloomery_metal::instance::UNREGISTERED {
            eprintln!(
                "metal: {instance}: no registered instance holds this shape, so no gate checked its output"
            );
        }
        let device = MetalDevice::open(instance).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let mut session = Session::open_model_as(
            &device,
            device.library(),
            &file,
            ctx,
            CHECKPOINT_EVERY,
            MarkCopies::AtOpen,
        )
        .map_err(|e| e.to_string())?;
        // The session holds its own buffers: the file's mapping is not kept for the run.
        drop(file);
        let checkpoints = match session.checkpoint_every() {
            Some(every) => format!(
                "checkpoints every {every} ({} bytes a point, capacity {})",
                session.checkpoint_bytes(),
                session.checkpoint_capacity()
            ),
            None => "no checkpoints (no recurrent store)".to_owned(),
        };
        eprintln!(
            "bloomery-serve-metal: opened {} in {:.1} s, ctx {ctx}, {checkpoints}",
            path.display(),
            t0.elapsed().as_secs_f64(),
        );
        let head_rows = session.vocab();
        let what = format!("metal {arch} session");
        let engine: Box<dyn serve::Engine> = match a.draft {
            None => Box::new(TargetEngine::new(
                session,
                head_rows,
                vocab,
                what,
                EngineProps::default(),
            )?),
            Some(Draft::Mtp) => {
                let path = a
                    .mtp
                    .as_deref()
                    .ok_or("--draft mtp: the draft layer is the sidecar's; give --mtp SIDECAR")?;
                let window = open_draft(&device, &mut session, path, a.width)?;
                drafted(
                    session,
                    window,
                    vocab,
                    what,
                    DraftProps {
                        model: "mtp".to_owned(),
                        n_max: Some(1),
                        kind: Some("mtp".to_owned()),
                        path: Some(path.display().to_string()),
                        device: None,
                    },
                )?
            }
            Some(Draft::Lookup) => {
                let window = open_lookup(&mut session, a.width)?;
                drafted(
                    session,
                    window,
                    vocab,
                    what,
                    DraftProps {
                        model: "lookup".to_owned(),
                        n_max: Some(1),
                        kind: Some("lookup".to_owned()),
                        path: None,
                        device: None,
                    },
                )?
            }
        };
        let config = ServerConfig {
            model_alias: a.alias.unwrap_or(name),
            model_path: path.display().to_string(),
            chat_template: template,
            chat_preparation: None,
            sampler: Some(sampler_factory()),
            fatal_linger: FATAL_LINGER,
            slot_save_path: None,
            api_keys: a.api_keys,
        };
        let slots = SlotConfig {
            parallel: 1,
            queue_depth: a.queue_depth,
            ..SlotConfig::default()
        };
        let server = Server::bind_with((a.host.as_str(), a.port), engine, config, slots)
            .map_err(|e| e.to_string())?;
        let addr = server.local_addr().map_err(|e| e.to_string())?;
        eprintln!("bloomery-serve-metal: listening on http://{addr}");
        Ok(server)
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("bloomery-serve-metal: refused: the Metal server runs on macOS only");
    std::process::ExitCode::from(78)
}
