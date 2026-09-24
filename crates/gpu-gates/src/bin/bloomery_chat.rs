//! `bloomery-chat` — text in, text out, on the V4.1 engine.
//!
//!     bloomery-chat (--prompt <text> | --prompt-file <path> | --prompt-stdin)
//!                   [-n N] [--place a|gate] [--ctx C] [--no-special]
//!                   [--greedy | [--temp T] [--top-k K] [--top-p P] [--min-p M]
//!                    [--repeat-penalty R] [--repeat-last-n L] [--seed S]]
//!
//! The prompt is tokenized with the vocabulary of `$BLOOMERY_REF_MODEL`'s
//! first shard, with the file's own BOS/EOS rule (`add_special`); special
//! tokens written in the text become their ids unless `--no-special`. No
//! chat template is applied: the text is fed as written. The engine is
//! loaded and captured by [`Generator`], the prompt fed one real step per
//! token, and each next token drawn from the head's logits by the sampler
//! (its defaults unless a flag says otherwise; seed 0), then decoded and
//! written to stdout as it arrives. The run stops after `-n` tokens (default
//! 256), at an end-of-generation token, or when the context is full.
//!
//! `--greedy` is the argmax, and the run checks every choice against the
//! engine's own argmax of the same step: the generated ids are then the ones
//! `generate_ds41 --tokens <the prompt's ids>` prints on its `tokens` line.
//! It refuses the sampling flags beside it.
//!
//! Everything but the text goes to stderr: the `prompt_ids` line, the plan,
//! `load` and `capture` lines, then after the run the `ids` line (every
//! generated id, the end-of-generation one included), a `text_consistent`
//! line (the streamed text against the vocabulary's one-shot decode of the
//! same ids), and the `chat:` summary line.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "bloomery-chat: built without the `deepseek41` feature; see `just gate-gpu-ds41-chat`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("bloomery-chat", drive::run())
}

#[cfg(feature = "deepseek41")]
mod drive {
    use std::io::{Read, Write};

    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::generate::{Generator, OpenArgs, Place};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::workstation;
    use sampler::{Sampler, SamplerParams};
    use tokenizer::{Decoder, Tokenizer};

    const USAGE: &str = "usage: bloomery-chat (--prompt <text> | --prompt-file <path> | \
                         --prompt-stdin) [-n N] [--place a|gate] [--ctx C] [--no-special] \
                         [--greedy | [--temp T] [--top-k K] [--top-p P] [--min-p M] \
                         [--repeat-penalty R] [--repeat-last-n L] [--seed S]]";

    struct Args {
        prompt: Vec<u8>,
        n_gen: usize,
        place: Place,
        ctx: usize,
        parse_special: bool,
        sampling: SamplerParams,
        greedy: bool,
    }

    /// Why the run stopped.
    #[derive(Clone, Copy)]
    enum Stop {
        /// An end-of-generation token was drawn.
        Eog,
        /// `-n` tokens are out.
        Length,
        /// The context has no position left for the next step.
        Ctx,
    }

    impl Stop {
        fn name(self) -> &'static str {
            match self {
                Stop::Eog => "eog",
                Stop::Length => "length",
                Stop::Ctx => "ctx",
            }
        }
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut a = Args {
            prompt: Vec::new(),
            n_gen: 256,
            place: Place::A,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            parse_special: true,
            sampling: SamplerParams::default(),
            greedy: false,
        };
        let (mut sources, mut sampling_flags) = (0, Vec::new());
        let mut it = std::env::args_os().skip(1);
        while let Some(flag) = it.next() {
            let flag = flag.to_string_lossy().into_owned();
            match flag.as_str() {
                "--greedy" => {
                    a.greedy = true;
                    continue;
                }
                "--no-special" => {
                    a.parse_special = false;
                    continue;
                }
                "--prompt-stdin" => {
                    sources += 1;
                    a.prompt.clear();
                    std::io::stdin().read_to_end(&mut a.prompt)?;
                    continue;
                }
                _ => {}
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
            if flag == "--prompt" {
                sources += 1;
                a.prompt = os_bytes(v);
                continue;
            }
            let v = v.to_string_lossy().into_owned();
            let s = &mut a.sampling;
            let sampling = match flag.as_str() {
                "--prompt-file" => {
                    sources += 1;
                    a.prompt = std::fs::read(&v).map_err(|e| format!("{v}: {e}"))?;
                    false
                }
                "-n" => {
                    a.n_gen = v.parse()?;
                    false
                }
                "--place" => {
                    a.place = Place::parse(&v)?;
                    false
                }
                "--ctx" => {
                    a.ctx = v.parse()?;
                    false
                }
                "--temp" => {
                    s.temperature = v.parse()?;
                    true
                }
                "--top-k" => {
                    s.top_k = v.parse()?;
                    true
                }
                "--top-p" => {
                    s.top_p = v.parse()?;
                    true
                }
                "--min-p" => {
                    s.min_p = v.parse()?;
                    true
                }
                "--repeat-penalty" => {
                    s.repeat_penalty = v.parse()?;
                    true
                }
                "--repeat-last-n" => {
                    s.repeat_last_n = v.parse()?;
                    true
                }
                "--seed" => {
                    s.seed = v.parse()?;
                    true
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            };
            if sampling {
                sampling_flags.push(flag);
            }
        }
        if sources != 1 {
            return Err(format!("name the prompt once: {USAGE}").into());
        }
        if a.greedy {
            if !sampling_flags.is_empty() {
                return Err(format!(
                    "--greedy is the argmax; it takes no sampling flag, got {}",
                    sampling_flags.join(" ")
                )
                .into());
            }
            a.sampling = SamplerParams::greedy();
        }
        if a.n_gen == 0 {
            return Err("-n wants at least one generated token".into());
        }
        Ok(a)
    }

    /// An argument's bytes as given (the prompt need not be UTF-8).
    fn os_bytes(v: std::ffi::OsString) -> Vec<u8> {
        use std::os::unix::ffi::OsStringExt;
        v.into_vec()
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        let mut sampler = Sampler::new(a.sampling)?;
        let path = ref_model_path()?;
        let tok = Tokenizer::from_gguf(&path)?;
        let ids = tok.encode(&a.prompt, true, a.parse_special);
        eprintln!("prompt_ids {ids:?}");
        if ids.is_empty() {
            return Err("the prompt encodes to no token".into());
        }

        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        drop(split);
        print_plan(&inputs, a.place, a.ctx)?;
        let want_top_k = inputs.hp.indexer.top_k;
        let pin_main = !std::env::var("BLOOMERY_PIN_MAIN").is_ok_and(|v| v == "0");
        let open = OpenArgs {
            place: a.place,
            ctx: a.ctx,
            mode: StepMode::Graph,
            pin_main,
        };
        let mut g = Generator::open(
            open,
            body::open,
            |m: &Deepseek41Model| {
                let top_k = m.body("bloomery-chat")?.indexer_top_k();
                if top_k != want_top_k {
                    return Err(format!(
                        "the body selects {top_k} rows per stream, the file's top_k is \
                         {want_top_k}: a step past that many visible rows would not be the model's"
                    )
                    .into());
                }
                Ok(format!("layers={} top_k={top_k}", inputs.hp.n_layer))
            },
            &mut std::io::stderr(),
        )?;
        chat(&mut g, &a, &tok, &mut sampler, &ids)
    }

    /// The plan the engine is about to load, on stderr.
    fn print_plan(inputs: &PlanInputs, place: Place, ctx: usize) -> Result<(), GateError> {
        let machine = place.machine()(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?)?;
        let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
        let card = &plan.cards[0];
        eprintln!(
            "plan place={} card={} ctx_max={} card_experts={} ({} B) host_experts={} ({} B) \
             n_l={}..{} on {} layers card_budget={}",
            place.name(),
            machine.cards[0].name,
            plan.ctx_max,
            card.experts,
            card.expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            held.iter().min().copied().unwrap_or(0),
            held.iter().max().copied().unwrap_or(0),
            held.len(),
            plan.card_budget
                .map_or_else(|| "none".to_string(), |b| b.to_string())
        );
        Ok(())
    }

    /// Feed the prompt, then draw, print and feed back one token at a time
    /// until a stop; the closing lines on stderr.
    fn chat(
        g: &mut Generator<body::Body>,
        a: &Args,
        tok: &Tokenizer,
        sampler: &mut Sampler,
        prompt: &[u32],
    ) -> Result<(), GateError> {
        let mut argmax = g.prefill(prompt)?;
        let mut history = prompt.to_vec();
        let mut out: Vec<u32> = Vec::with_capacity(a.n_gen);
        let mut text = String::new();
        let mut dec = Decoder::new(tok, true);
        let mut stdout = std::io::stdout().lock();
        let stop = loop {
            let logits = g.logits()?;
            if logits.len() != tok.n_vocab() {
                return Err(format!(
                    "the head gives {} logits, the vocabulary has {} tokens",
                    logits.len(),
                    tok.n_vocab()
                )
                .into());
            }
            let next = sampler.sample(&logits, &history);
            if a.greedy && next != argmax {
                return Err(format!(
                    "greedy chose {next} at position {}, the engine's argmax is {argmax}",
                    g.pos()
                )
                .into());
            }
            out.push(next);
            history.push(next);
            if tok.eog().contains(&next) {
                break Stop::Eog;
            }
            if let Some(piece) = dec.push(next) {
                stdout.write_all(piece.as_bytes())?;
                stdout.flush()?;
                text.push_str(piece);
            }
            if out.len() == a.n_gen {
                break Stop::Length;
            }
            if g.pos() >= g.ctx_max() {
                break Stop::Ctx;
            }
            argmax = g.step(next)?;
        };
        if let Some(rest) = dec.finish() {
            stdout.write_all(rest.as_bytes())?;
            text.push_str(&rest);
        }
        stdout.write_all(b"\n")?;
        stdout.flush()?;
        drop(stdout);

        let shown = match stop {
            Stop::Eog => &out[..out.len() - 1],
            Stop::Length | Stop::Ctx => &out[..],
        };
        eprintln!("ids {out:?}");
        eprintln!("text_consistent={}", text == tok.decode(shown));
        eprintln!(
            "chat: prompt_tokens={} generated={} stop={}",
            prompt.len(),
            out.len(),
            stop.name()
        );
        Ok(())
    }
}
