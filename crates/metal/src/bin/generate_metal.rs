//! `generate_metal` — the qwen35moe session's decode CLI on Metal: a
//! qwen35moe file opened on this Mac's device, greedy, one token a step, the
//! generated text streamed as ids are picked. The flags spell what the CUDA
//! family's `generate_qwen3moe` spells where the meaning is the same; the
//! file is `$BLOOMERY_REF_MODEL`, as there.
//!
//!     generate_metal (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C]
//!                    [--print-ids]
//!
//! Defaults: N 32, C the smaller of 4096 and the file's `context_length`. A
//! flag given twice takes its last value. `--prompt` tokenizes the text with
//! the file's own vocabulary (`tokenizer`, the specials the header asks for,
//! no chat template — the CUDA CLI runs none either; a server's seat runs
//! the file's). The generation stops at the count, at an end-of-generation
//! id of the vocabulary's own set, or at the cache's last row; the summary
//! lines say which.
//!
//! The text streams through the tokenizer's decoder, so a multi-byte
//! character split across pieces prints once, complete; `--print-ids`
//! prints the generated ids as one comma-separated stdout line instead (the
//! `--tokens` spelling, so a run's output feeds back as an input), for
//! gates. The summary lines go to stderr, `key=value` one a line:
//!
//!     model=<path> open_s=<the open's wall> device_allocated_bytes=<the
//!     driver's live device bytes after the open> prompt_ids=<count>
//!     prompt_s=<the prompt's wall> prompt_tok_s=<rate> decode_tokens=<count>
//!     decode_s=<the loop's wall> decode_tok_s=<rate> stop=<eog|length|ctx>
//!
//! The wall and rate lines are development-loop numbers: no lease, no
//! witness, this Mac whatever else it runs.
//!
//! A bad flag or a bad value is a named error and exit 2; a refusal of the
//! device, the session or the tokenizer, or a fault the device raised, is a
//! named error and exit 1 — never a partial result presented as success.
//! The body is macOS only; elsewhere it refuses and exits 78.

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    cli::run()
}

#[cfg(target_os = "macos")]
mod cli {
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::process::ExitCode;
    use std::time::Instant;

    use bloomery_device::Device;
    use bloomery_metal::arch::qwen3moe::generate::greedy;
    use bloomery_metal::arch::qwen3moe::session::Session;
    use bloomery_metal::device::MetalDevice;

    /// The model file, as the CUDA generate CLIs name it.
    const MODEL_ENV: &str = "BLOOMERY_REF_MODEL";

    /// The cache rows' default cap.
    const CTX_DEFAULT: usize = 4096;

    const USAGE: &str = "\
generate_metal (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C] [--print-ids]
  the file is $BLOOMERY_REF_MODEL; defaults N 32, --ctx the smaller of 4096
  and the file's context_length; --print-ids prints the ids, not the text";

    /// The flags that take a value.
    const VALUE_FLAGS: [&str; 4] = ["--prompt", "--tokens", "-n", "--ctx"];
    /// The flags that stand alone.
    const BARE_FLAGS: [&str; 2] = ["--print-ids", "--help"];

    /// What named the prompt: the text the tokenizer encodes, or the ids as
    /// given. The tokenizer's own failures are the run's, not the usage
    /// parse's, so it opens after the flags are read.
    enum Prompt {
        Text(String),
        Ids(Vec<u32>),
    }

    struct Args {
        prompt: Prompt,
        n: usize,
        ctx: Option<usize>,
        print_ids: bool,
        model: PathBuf,
    }

    /// Every value of flag `name`, in order; a trailing flag without its
    /// value is refused by name.
    fn flags(name: &str) -> Result<Vec<String>, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut out = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if a == name {
                out.push(
                    args.get(i + 1)
                        .ok_or_else(|| format!("{name} needs a value"))?
                        .clone(),
                );
            }
        }
        Ok(out)
    }

    /// The last value of flag `name`, if it was given.
    fn flag(name: &str) -> Result<Option<String>, String> {
        Ok(flags(name)?.pop())
    }

    /// Comma-separated ids, as the CUDA CLIs parse them.
    fn ids_of(s: &str) -> Result<Vec<u32>, String> {
        s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("--tokens {s}: {e}"))
    }

    /// A flag this CLI does not know, if one sits among the arguments.
    fn unknown_flag() -> Option<String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 2;
            } else if BARE_FLAGS.contains(&a.as_str()) {
                i += 1;
            } else {
                return Some(a.clone());
            }
        }
        None
    }

    fn parse() -> Result<Args, String> {
        if let Some(name) = unknown_flag() {
            return Err(format!("{name} is not a flag of this CLI"));
        }
        if std::env::args().any(|a| a == "--help") {
            print!("{USAGE}");
            std::process::exit(0);
        }
        let text = flag("--prompt")?;
        let tokens = flag("--tokens")?;
        if text.is_some() && tokens.is_some() {
            return Err("give --prompt or --tokens, not both".into());
        }
        let prompt = match (&text, &tokens) {
            (Some(t), _) => Prompt::Text(t.clone()),
            (_, Some(s)) => Prompt::Ids(ids_of(s)?),
            _ => return Err("give one of --prompt or --tokens".to_string()),
        };
        let n_gen: usize = match flag("-n")? {
            None => 32,
            Some(s) => s.parse().map_err(|e| format!("-n {s}: {e}"))?,
        };
        if n_gen == 0 {
            return Err(
                "-n 0: a generation of 0 tokens: token 0 comes out of the prompt call".into(),
            );
        }
        let ctx = match flag("--ctx")? {
            None => None,
            Some(s) => {
                let c: usize = s.parse().map_err(|e| format!("--ctx {s}: {e}"))?;
                if c == 0 {
                    return Err(format!("--ctx {s}: a cache of no rows"));
                }
                Some(c)
            }
        };
        let print_ids = std::env::args().any(|a| a == "--print-ids");
        let model = match std::env::var_os(MODEL_ENV) {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => {
                return Err(format!(
                    "no model file: set {MODEL_ENV} to the qwen35moe file (the first shard)"
                ));
            }
        };
        Ok(Args {
            prompt,
            n: n_gen,
            ctx,
            print_ids,
            model,
        })
    }

    /// The cache rows a run holds: the asked count, else the smaller of the
    /// default cap and the file's `context_length` — the key the plan
    /// reader reads, read here once more for the default alone.
    fn ctx_of(a: &Args) -> Result<usize, String> {
        match a.ctx {
            Some(c) => Ok(c),
            None => {
                let split = gguf::Split::open(&a.model)
                    .map_err(|e| format!("{}: {e}", a.model.display()))?;
                let train = split
                    .arch_get_u64("context_length")
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        format!(
                            "{}: {} is absent or not a count",
                            a.model.display(),
                            split.arch_key("context_length")
                        )
                    })?;
                Ok(CTX_DEFAULT.min(train))
            }
        }
    }

    /// The generated text as it completes: the tokenizer's streaming
    /// decoder, flushed a piece.
    struct Stream<'a> {
        decoder: Option<tokenizer::Decoder<'a>>,
    }

    impl Stream<'_> {
        fn id(&mut self, id: u32) {
            let Some(d) = self.decoder.as_mut() else {
                return;
            };
            if let Some(piece) = d.push(id) {
                print!("{piece}");
                let _ = std::io::stdout().flush();
            }
        }

        fn finish(&mut self) {
            if let Some(rest) = self.decoder.as_mut().and_then(|d| d.finish()) {
                print!("{rest}");
            }
            let _ = std::io::stdout().flush();
        }
    }

    pub(super) fn run() -> ExitCode {
        // No lever row acts on this CLI; `at_main` still refuses a retired
        // name that is set, and `--levers` prints the table.
        if let Err(e) = bloomery_levers::at_main(&[]) {
            eprintln!("generate_metal: {e}");
            return ExitCode::from(1);
        }
        let a = match parse() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                return ExitCode::from(2);
            }
        };
        match generate(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                ExitCode::from(1)
            }
        }
    }

    fn generate(a: &Args) -> Result<(), String> {
        // The tokenizer decodes what the run streams and owns the
        // end-of-generation set, so every run opens it, whatever named the
        // prompt.
        let tok = tokenizer::Tokenizer::from_gguf(&a.model).map_err(|e| e.to_string())?;
        let prompt = match &a.prompt {
            Prompt::Text(t) => tok.encode(t, true, false),
            Prompt::Ids(v) => v.clone(),
        };
        if prompt.is_empty() {
            return Err("the prompt has no ids".into());
        }
        let ctx = ctx_of(a)?;
        let device = MetalDevice::open().map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let mut session =
            Session::open(&device, device.library(), &a.model, ctx).map_err(|e| e.to_string())?;
        let open_s = t0.elapsed().as_secs_f64();
        // The census's free bytes are the working set less the driver's own
        // `currentAllocatedSize`, the one count of the live device buffers
        // this process holds; the arena keeps no sum of its own.
        let allocated = device
            .info()
            .map(|i| i.total_bytes - i.free_bytes)
            .map_err(|e| e.to_string())?;
        let mut stream = Stream {
            decoder: (!a.print_ids).then(|| tokenizer::Decoder::new(&tok, true)),
        };
        let out = greedy(&mut session, &prompt, a.n, tok.eog(), |id| stream.id(id))
            .map_err(|e| e.to_string())?;
        if a.print_ids {
            let ids: Vec<String> = out.ids.iter().map(u32::to_string).collect();
            println!("{}", ids.join(","));
        } else {
            stream.finish();
        }
        let prompt_s = out.prompt_wall.as_secs_f64();
        let decode_s = out.decode_wall.as_secs_f64();
        eprintln!("# development-loop numbers: no lease, no witness");
        eprintln!("model={}", a.model.display());
        eprintln!("open_s={open_s:.3}");
        eprintln!("device_allocated_bytes={allocated}");
        eprintln!("prompt_ids={}", prompt.len());
        eprintln!("prompt_s={prompt_s:.3}");
        eprintln!("prompt_tok_s={:.1}", prompt.len() as f64 / prompt_s);
        eprintln!("decode_tokens={}", out.tokens());
        eprintln!("decode_s={decode_s:.3}");
        eprintln!("decode_tok_s={:.1}", out.tokens() as f64 / decode_s);
        eprintln!("stop={}", out.stop.name());
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("generate_metal: refused: the Metal CLI runs on macOS only");
    std::process::ExitCode::from(78)
}
