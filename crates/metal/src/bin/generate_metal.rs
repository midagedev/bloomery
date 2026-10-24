//! `generate_metal` — the qwen35moe session's decode CLI on Metal: a
//! qwen35moe file opened on this Mac's device, greedy, one token a step, the
//! generated text streamed as ids are picked. The flags spell what the CUDA
//! family's `generate_qwen3moe` spells where the meaning is the same; the
//! file is `$BLOOMERY_REF_MODEL`, as there.
//!
//!     generate_metal (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C]
//!                    [--print-ids] [--profile [N]] [--profile-layers]
//!                    [--step-times [N]]
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
//! `--profile [N]` (default N 32; `--profile-layers` implies it) steps N
//! more tokens after the summary, each group of each token on a command
//! buffer of its own with its GPU seconds and its bytes read, and prints a
//! `prof` line a group kind (and a layer, under `--profile-layers`), then
//! the total beside the run's own decode ms a token, so the gap between the
//! groups' GPU time and the real step shows.
//!
//! `--step-times [N]` (default N 32) steps N more tokens in each mode —
//! host-fed, chain, ahead, one id throughout — and prints one `step_times`
//! line a mode: the host's encode µs a token, its GPU start against the host
//! clock at the commit, the GPU's own µs, and the host's µs from the GPU end
//! to the settle's return. What the phases say of a mode the sum against the
//! step's wall does: a host-fed step pays all four serially, a chain or
//! ahead step encodes the next token under the current one's GPU time.
//!
//! The wall and rate lines are development-loop numbers: no lease, no
//! witness, this Mac whatever else it runs. The `prof` lines are too: the
//! profiled token runs one command buffer a group, waited, which the real
//! step's single buffer does not.
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
    use bloomery_metal::arch::qwen3moe::count::{GroupKind, KINDS};
    use bloomery_metal::arch::qwen3moe::generate::greedy;
    use bloomery_metal::arch::qwen3moe::session::{ProfileGroup, Session, StepMode};
    use bloomery_metal::device::MetalDevice;

    /// The model file, as the CUDA generate CLIs name it.
    const MODEL_ENV: &str = "BLOOMERY_REF_MODEL";

    /// The cache rows' default cap.
    const CTX_DEFAULT: usize = 4096;

    const USAGE: &str = "\
generate_metal (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C] [--print-ids] [--profile [N]] [--profile-layers] [--step-times [N]]
  the file is $BLOOMERY_REF_MODEL; defaults N 32, --ctx the smaller of 4096
  and the file's context_length; --print-ids prints the ids, not the text;
  --profile [N] (default 32) steps N more tokens after the summary, each
  group's GPU time and bytes printed; --profile-layers adds a layer each;
  --step-times [N] (default 32) steps N more tokens in each mode, the step
  clock's phases a mode printed";

    /// The flags that take a value.
    const VALUE_FLAGS: [&str; 4] = ["--prompt", "--tokens", "-n", "--ctx"];
    /// The flags that stand alone.
    const BARE_FLAGS: [&str; 3] = ["--print-ids", "--help", "--profile-layers"];
    /// The profile tokens' default count, and the step-times' per mode.
    const PROFILE_DEFAULT: usize = 32;

    /// What named the prompt: the text the tokenizer encodes, or the ids as
    /// given. The tokenizer's own failures are the run's, not the usage
    /// parse's, so it opens after the flags are read.
    enum Prompt {
        Text(String),
        Ids(Vec<u32>),
    }

    /// What `--profile` asked: how many tokens to step past the summary,
    /// and whether a layer's line prints beside each kind's.
    struct Profile {
        n: usize,
        layers: bool,
    }

    struct Args {
        prompt: Prompt,
        n: usize,
        ctx: Option<usize>,
        print_ids: bool,
        profile: Option<Profile>,
        /// `--step-times [N]`: the step clock's tokens a mode.
        step_times: Option<usize>,
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
    /// `--profile` stands alone or takes its count, so it eats the next
    /// argument only when that one names a count.
    fn unknown_flag() -> Option<String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 2;
            } else if ["--profile", "--step-times"].contains(&a.as_str()) {
                i += if args.get(i + 1).is_some_and(|v| v.parse::<usize>().is_ok()) {
                    2
                } else {
                    1
                };
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
        // `--profile` stands alone at its default count or takes one; a last
        // value wins, as for every flag. `--profile-layers` alone implies
        // the profile.
        let mut profile_n: Option<usize> = None;
        {
            let args: Vec<String> = std::env::args().skip(1).collect();
            for (i, a) in args.iter().enumerate() {
                if a != "--profile" {
                    continue;
                }
                let n = args
                    .get(i + 1)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(PROFILE_DEFAULT);
                profile_n = Some(n);
            }
        }
        let layers = std::env::args().any(|a| a == "--profile-layers");
        let profile = match (profile_n, layers) {
            (Some(n), l) => {
                if n == 0 {
                    return Err(format!(
                        "--profile {n}: a profile of no tokens; drop the flag to run none"
                    ));
                }
                Some(Profile { n, layers: l })
            }
            (None, true) => Some(Profile {
                n: PROFILE_DEFAULT,
                layers: true,
            }),
            (None, false) => None,
        };
        // `--step-times` stands alone at its default count or takes one; a
        // last value wins, as for every flag.
        let mut step_times: Option<usize> = None;
        {
            let args: Vec<String> = std::env::args().skip(1).collect();
            for (i, a) in args.iter().enumerate() {
                if a != "--step-times" {
                    continue;
                }
                let n = args
                    .get(i + 1)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(PROFILE_DEFAULT);
                step_times = Some(n);
            }
        }
        if step_times == Some(0) {
            return Err(
                "--step-times 0: a step-times run of no tokens; drop the flag to run none".into(),
            );
        }
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
            profile,
            step_times,
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

    /// One accumulator of a kind's (or a kind and layer's) profiled groups.
    #[derive(Clone, Copy, Default)]
    struct Acc {
        groups: u64,
        gpu_s: f64,
        bytes: u64,
    }

    /// The `prof` lines after a run's summary: `p.n` more tokens stepped,
    /// each group on a command buffer of its own, its GPU seconds and its
    /// bytes read, one line a kind and (under `p.layers`) a layer, then the
    /// total beside the run's own decode ms a token. The ids stop nothing
    /// here: the profile steps on, whatever the run stopped at, until the
    /// count or the cache's last row.
    fn profile_run(
        s: &mut Session,
        p: &Profile,
        mut id: u32,
        decode_ms: f64,
    ) -> Result<(), String> {
        let rows = s.ctx() - s.pos() as usize;
        if rows == 0 {
            eprintln!(
                "prof: refused: the cache holds no row past position {}; give --ctx more rows",
                s.pos()
            );
            return Ok(());
        }
        let steps = p.n.min(rows);
        let layers = s.layer_count();
        // One accumulator a kind and layer (the head's and the embed's sit
        // at `layers`), filled in place: the first token sizes it, the rest
        // allocate nothing.
        let mut acc: Vec<Acc> = vec![Acc::default(); KINDS.len() * (layers + 1)];
        let mut samples: Vec<ProfileGroup> = Vec::new();
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(layers);
        for _ in 0..steps {
            let pos = s.pos();
            id = s
                .step_profile(id, &mut samples)
                .map_err(|e| e.to_string())?
                .id;
            let probe = s.probe().map_err(|e| e.to_string())?;
            ids.clear();
            for l in 0..layers {
                ids.push(probe.router_ids(l).map_err(|e| e.to_string())?);
            }
            for g in &samples {
                let none: &[u32] = &[];
                let routed = match g.kind {
                    GroupKind::GateUp | GroupKind::Down => &ids[g.layer.unwrap_or(0)],
                    _ => none,
                };
                let bytes = s
                    .group_bytes(g, pos, routed)
                    .map_err(|e| e.to_string())?
                    .total();
                let a = &mut acc[kind_index(g.kind) * (layers + 1) + g.layer.unwrap_or(layers)];
                a.groups += 1;
                a.gpu_s += g.gpu_s;
                a.bytes += bytes;
            }
        }
        eprintln!(
            "prof tokens={steps} at {}..={} (development-loop numbers: one command buffer a \
             group, waited; the real step's single buffer is not this)",
            s.pos() - steps as u32,
            s.pos() - 1
        );
        let kind_total = |kind: GroupKind| -> Acc {
            let mut t = Acc::default();
            for l in 0..=layers {
                let a = &acc[kind_index(kind) * (layers + 1) + l];
                t.groups += a.groups;
                t.gpu_s += a.gpu_s;
                t.bytes += a.bytes;
            }
            t
        };
        let line = |op: &str, layer: Option<usize>, a: &Acc| {
            let n = steps as f64;
            let at = layer.map_or_else(String::new, |l| format!(" layer={l}"));
            eprintln!(
                "prof op={op}{at} groups={:.2} gpu_ms={:.3} bytes={:.0} gbps={:.1}",
                a.groups as f64 / n,
                a.gpu_s / n * 1e3,
                a.bytes as f64 / n,
                if a.gpu_s > 0.0 {
                    a.bytes as f64 / a.gpu_s / 1e9
                } else {
                    0.0
                },
            );
        };
        let mut total = Acc::default();
        for &kind in &KINDS {
            let t = kind_total(kind);
            if t.groups == 0 {
                continue;
            }
            line(kind.name(), None, &t);
            total.groups += t.groups;
            total.gpu_s += t.gpu_s;
            total.bytes += t.bytes;
            if p.layers {
                for l in 0..layers {
                    let a = &acc[kind_index(kind) * (layers + 1) + l];
                    if a.groups != 0 {
                        line(kind.name(), Some(l), a);
                    }
                }
            }
        }
        eprintln!(
            "prof total groups={:.2} gpu_ms={:.3} bytes={:.0} gbps={:.1} decode_ms={:.3}",
            total.groups as f64 / steps as f64,
            total.gpu_s / steps as f64 * 1e3,
            total.bytes as f64 / steps as f64,
            if total.gpu_s > 0.0 {
                total.bytes as f64 / total.gpu_s / 1e9
            } else {
                0.0
            },
            decode_ms,
        );
        Ok(())
    }

    /// `kind`'s place in [`KINDS`].
    fn kind_index(kind: GroupKind) -> usize {
        KINDS
            .iter()
            .position(|k| *k == kind)
            .unwrap_or_else(|| panic!("generate_metal: the kind {kind:?} sits in no kind table"))
    }

    /// The `step_times` lines: `n` tokens in each mode past the run —
    /// host-fed and chain greedy, ahead on one id throughout, so every token
    /// ahead is kept — the step clock's phases averaged a mode and printed
    /// as one line, then the count. The lines are development-loop numbers,
    /// like the summary's: the three runs sit at three depths, a mode's
    /// attention reading a few keys more than the mode before it.
    fn step_times_run(s: &mut Session, n: usize, id: u32) -> Result<(), String> {
        let rows = s.ctx() - s.pos() as usize;
        let per = n.min(rows / 3);
        if per == 0 {
            eprintln!(
                "step_times: refused: {} rows past position {} hold no token of each of the \
                 three modes; give --ctx more rows",
                rows,
                s.pos()
            );
            return Ok(());
        }
        s.start_step_clock().map_err(|e| e.to_string())?;
        let mut pick = id;
        for _ in 0..per {
            pick = s
                .step(pick, StepMode::HostFed)
                .map_err(|e| e.to_string())?
                .id;
        }
        for _ in 0..per {
            pick = s.step(pick, StepMode::Chain).map_err(|e| e.to_string())?.id;
        }
        for _ in 0..per {
            s.step(id, StepMode::Ahead { next: Some(id) })
                .map_err(|e| e.to_string())?;
        }
        let clock = s.step_clock().ok_or("the step clock did not run")?;
        for mode in [
            StepMode::HostFed,
            StepMode::Chain,
            StepMode::Ahead { next: None },
        ] {
            let name = match mode {
                StepMode::HostFed => "host",
                StepMode::Chain => "chain",
                StepMode::Ahead { .. } => "ahead",
            };
            let p = clock.phase(mode);
            eprintln!(
                "step_times mode={name} steps={} encode_us={:.1} start_us={:.1} gpu_us={:.1} \
                 wake_us={:.1}",
                p.steps, p.encode_us, p.start_us, p.gpu_us, p.wake_us,
            );
        }
        eprintln!(
            "step_times {per} tokens each of host, chain and ahead (development-loop numbers: \
             no lease, no witness)"
        );
        Ok(())
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
        if let Some(p) = &a.profile {
            let id = *out
                .ids
                .last()
                .ok_or("a profile of no token: the run picked none")?;
            let decode_ms = decode_s / out.tokens() as f64 * 1e3;
            profile_run(&mut session, p, id, decode_ms)?;
        }
        if let Some(n) = a.step_times {
            let id = *out
                .ids
                .last()
                .ok_or("a step-times run of no token: the run picked none")?;
            step_times_run(&mut session, n, id)?;
        }
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("generate_metal: refused: the Metal CLI runs on macOS only");
    std::process::ExitCode::from(78)
}
