//! `bloomery-serve-metal` — the llama-server-compatible HTTP API on this
//! Mac's Metal device: a model opened as `generate_metal` opens it, served
//! by `bloomery_metal::server::TargetEngine`.
//!
//!     bloomery-serve-metal (-m PATH | --hf REPO) [--host 127.0.0.1]
//!                          [--port 8080] [--ctx C]
//!                          [--chat-template-file PATH] [--alias NAME]
//!                          [--queue-depth Q] [--api-key KEY] [--api-key-file F]
//!
//! The model is a GGUF file's first shard or a checkpoint folder (an MLX
//! folder as downloaded, no `prepare`); `--hf REPO` fetches a repo into the
//! cache and opens it (`bloomery_metal::model_arg`). `--ctx-size` and `-c`
//! are `--ctx` under llama-server's spellings; unset, the cache holds the
//! smaller of 32768 rows and the file's `context_length`. The file gives the
//! vocabulary, `tokenizer.chat_template` the chat template
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
    use bloomery_metal::arch::qwen3moe::session::{CHECKPOINT_EVERY, MarkCopies, Session};
    use bloomery_metal::model_arg;
    use bloomery_metal::server::TargetEngine;
    use hf::source::Source;
    use model_hf::ModelFile;
    use serve::flag::{self, ApiKeys};
    use serve::sampling::sampler_factory;
    use serve::vocab::{ChatSurface, TemplateFile, Vocab};
    use serve::{EngineProps, FATAL_LINGER, ServeError, Server, ServerConfig, SlotConfig};

    /// The cache rows' default cap.
    const CTX_DEFAULT: usize = 32768;

    const USAGE: &str = "\
bloomery-serve-metal (-m PATH | --hf REPO) [--host H] [--port P] [--ctx C] [--chat-template-file PATH]
                     [--alias NAME] [--queue-depth Q] [--api-key KEY] [--api-key-file F]
  PATH is a GGUF file (its first shard) or a checkpoint folder; REPO is fetched into the cache
  --ctx (--ctx-size, -c) defaults to the smaller of 32768 and the file's context_length";

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
    }

    fn parse() -> Result<Args, String> {
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
                "--help" | "-h" => return Err(USAGE.to_owned()),
                other => return Err(format!("unknown flag {other}\n{USAGE}")),
            }
        }
        a.model = model_arg::named(&model, PATH_FLAGS, None, "-m PATH or --hf REPO")
            .map_err(|e| format!("{e}\n{USAGE}"))?;
        Ok(a)
    }

    pub(super) fn run() -> ExitCode {
        if let Err(e) = bloomery_levers::at_main(&[]) {
            eprintln!("bloomery-serve-metal: {e}");
            return ExitCode::from(1);
        }
        let a = match parse() {
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

    fn open(a: Args) -> Result<Server, String> {
        let model = model_arg::resolve(&a.model)?;
        let path = &model.path;
        let file = ModelFile::open(path).map_err(|e| e.to_string())?;
        let split = file.header();
        let ctx = match a.ctx {
            Some(c) => c,
            None => {
                let train = split
                    .arch_get_u64("context_length")
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        format!(
                            "{}: {} is absent or not a count",
                            path.display(),
                            split.arch_key("context_length")
                        )
                    })?;
                CTX_DEFAULT.min(train)
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
        let session = Session::open_model_as(
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
        let engine = TargetEngine::new(
            session,
            head_rows,
            vocab,
            format!("metal {arch} session"),
            EngineProps::default(),
        )?;
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
        let server = Server::bind_with((a.host.as_str(), a.port), Box::new(engine), config, slots)
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
