//! `bloomery-chat` — text in, text out, on the V4.1 engine.
//!
//!     bloomery-chat (--prompt <text> | --prompt-file <path> | --prompt-stdin)
//!                   [-n N] [--place a|gate|bp] [--ctx C] [--no-special]
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
//! `--place` is `generate_ds41`'s: `bp` loads plan (a) on the A6000 with the
//! 3090 as its expert tier (the `load` line's `cards=` names both, with the
//! tier's experts and bytes; the prompt goes one step per id, as it always
//! does here). A lost tier card is the step's named error and ends the run
//! with a nonzero exit.
//!
//! Everything but the text goes to stderr: the `prompt_ids` line, the plan,
//! `load` and `capture` lines, then after the run the `ids` line (every
//! generated id, the end-of-generation one included), a `text_consistent`
//! line (the streamed text against the vocabulary's one-shot decode of the
//! same ids), and the `chat:` summary line.
//!
//! The levers it acts on are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them — the prompt feed's among them: a prompt is fed one step per id — and
//! a `BLOOMERY_*` name no registry row names; `--levers` prints them with this
//! process's values and exits. The stderr
//! lines are records of the kinds `bloomery_gpu_gates::record` declares;
//! `--records-schema` prints those kinds and exits.

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
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "deepseek41")]
mod drive {
    use std::io::{Read, Write};

    use app::Open;
    use app::arch::deepseek41::Ds41Cfg;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model};
    use bloomery_gpu_gates::generate::{Generator, OpenArgs, Place};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, HOT_LIST, PIN_MAIN, R8,
    };
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::workstation::{self, TierBatchBytes};
    use model::placement::{HotList, PlanLevers};
    use sampler::{Sampler, SamplerParams};
    use tokenizer::{Decoder, Tokenizer};

    use crate::place;

    const USAGE: &str = "usage: bloomery-chat (--prompt <text> | --prompt-file <path> | \
                         --prompt-stdin) [-n N] [--place a|gate|bp] [--ctx C] [--no-special] \
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
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            HOT_LIST,
            CARD_BUDGET,
            PIN_MAIN,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        record::at_main("bloomery-chat", record::BLOOMERY_CHAT);
        let a = parse_args()?;
        let mut cfg = body::OpenCfg::from_levers(&levers)?;
        // The chat feeds its prompt one step per id (`Generator::prefill`), so
        // its load makes no prompt batch's buffers, tiered or not.
        cfg.body.prefill = body::PrefillMode::Steps;
        let ds41 = Ds41Cfg {
            feed: cfg.body.prefill,
            open: cfg,
            card_timing: false,
        };
        let mut sampler = Sampler::new(a.sampling)?;
        let path = ref_model_path()?;
        let tok = Tokenizer::from_gguf(&path)?;
        let ids = tok.encode(&a.prompt, true, a.parse_special);
        Record::new(&record::PROMPT_IDS).list("ids", &ids).eprint();
        if ids.is_empty() {
            return Err("the prompt encodes to no token".into());
        }

        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let tier_batch = place::tier_batch(a.place, &inputs.hp);
        drop(split);
        print_plan(&inputs, a.place, tier_batch, a.ctx, &ds41.open.place)?;
        let want_top_k = inputs.hp.indexer.top_k;
        let open = OpenArgs {
            place: a.place,
            tier_batch,
            ctx: a.ctx,
            mode: StepMode::Graph,
            pin_main: levers.pin_main(),
        };
        let mut g = Generator::open(
            open,
            |file, machine, ctx| {
                let t = std::time::Instant::now();
                let inputs = Body::inputs(&file)?;
                let machine = machine(Body::layer_count(&inputs));
                let plan = Body::plan(&inputs, &machine, ctx, &ds41)?;
                let planned = t.elapsed();
                let mut model = Body::open(file, &inputs, &plan, &ds41)?;
                model.note_load_plan(planned);
                Ok(model)
            },
            |m: &Deepseek41Model, load: Record| {
                let body = m.body("bloomery-chat")?;
                let top_k = body.indexer_top_k();
                let shadow = body.shadow_host();
                if top_k != want_top_k {
                    return Err(format!(
                        "the body selects {top_k} rows per stream, the file's top_k is \
                         {want_top_k}: a step past that many visible rows would not be the model's"
                    )
                    .into());
                }
                let load = load
                    .u("layers", inputs.hp.n_layer)
                    .u("top_k", top_k)
                    .w("shadow", "host")
                    .u("shadow_bytes", shadow.bytes)
                    .u("unified_addressing", shadow.unified_addressing);
                place::with_cards(m, a.place, "bloomery-chat", load)
            },
            &mut std::io::stderr(),
        )?;
        chat(&mut g, &a, &tok, &mut sampler, &ids)
    }

    /// The plan the engine is about to load under the placement's `levers`
    /// (with `tier_batch`, the tier's prompt-batch bytes, on its tier card),
    /// on stderr.
    fn print_plan(
        inputs: &PlanInputs,
        place: Place,
        tier_batch: Option<TierBatchBytes>,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<(), GateError> {
        let machine = place.machine(None, tier_batch)?(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?, levers)?;
        let hot_list = levers.hot.as_ref().map_or("none", HotList::path);
        record::plan(place.name(), &machine, &plan, hot_list).eprint();
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
        // One row for every token: the head refuses a row of another length.
        let mut logits = vec![0.0f32; tok.n_vocab()];
        let stop = loop {
            g.model().logits_into(&mut logits)?;
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
        Record::new(&record::IDS).list("ids", &out).eprint();
        Record::new(&record::TEXT_CONSISTENT)
            .w("consistent", text == tok.decode(shown))
            .eprint();
        Record::new(&record::CHAT)
            .u("prompt_tokens", prompt.len())
            .u("generated", out.len())
            .w("stop", stop.name())
            .eprint();
        Ok(())
    }
}
