//! `generate_kimi_k3` — Kimi K3's end-to-end decode CLI: a prompt of ids (or text, raw, no chat
//! template) walked through the decode step one id at a time, then `-n` greedy ids, printed.
//!
//!     BLOOMERY_MODEL=kimi-k3 generate_kimi_k3 [--tokens a,b,c | --prompt TEXT] [-n N]
//!                            [--ctx C] [--place a] [--mode eager|graph] [--plan]
//!
//! Defaults: `-n 8`, `--ctx 512`, `--place a`, `--mode graph`. The model is `ref_model_path()`. The
//! file is read as the plan reads it ([`PlanInputs::read_without_chat`]: every part the body does
//! not run is refused by name, the template's tool-call parser excepted) and planned on `--place`
//! before anything loads; `--plan` ends there. The body holds the whole model on one card, so
//! `--place a` is the one placement. The released file ends at the plan (its dense trunk alone
//! does not fit the card), and a file the plan passes but whose tensors exceed the card's free
//! bytes is refused by name by the load ([`bloomery_gpu::GpuModel::open`]): the placed load over
//! two stage cards, the host and the NVMe tier does not exist. The prompt walk is the session's
//! ([`app::Prompt`] for the body); the body has no rows pass.
//!
//! The lines on stdout: `load` (the resident bytes, the context, seconds), `ids` (the generated
//! ids) and, with the vocabulary's pieces, `text`. The plan's records go to stderr.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::record::at_main(
        "generate_kimi_k3",
        bloomery_gpu_gates::record::GENERATE_KIMI_K3,
    );
    bloomery_gpu_gates::exit_with("generate_kimi_k3", cli::run())
}

#[cfg(feature = "gpu")]
mod cli {
    use app::Prompt;
    use bloomery_gpu::arch::kimi_k3::Body;
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::need;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
    use gguf::Split;
    use model::arch::kimi_k3::place::PlanInputs;
    use model::placement::PlanLevers;
    use tokenizer::Tokenizer;

    /// The argument after flag `name`: `None` when the flag is absent, an error when it is last.
    fn flag(args: &[String], name: &str) -> Result<Option<String>, GateError> {
        args.iter()
            .position(|a| a == name)
            .map(|i| {
                args.get(i + 1).cloned().ok_or_else(|| {
                    GateError::from(format!("generate_kimi_k3: {name} needs a value"))
                })
            })
            .transpose()
    }

    /// The model of `file` on the card this process holds, with `ctx` positions, stepping in `mode`.
    /// The body's own refusals reach the caller by name (a file past the card's free bytes before
    /// any tensor is read).
    fn open(file: Split, ctx: usize, mode: StepMode) -> Result<GpuModel<Body>, GateError> {
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, ctx)?;
        m.set_mode(mode);
        Ok(m)
    }

    /// `n` greedy ids after `prompt`, from the model's position: the argmax after the prompt
    /// ([`Prompt::prompt`]), then each id fed back one a step.
    fn greedy(m: &mut GpuModel<Body>, prompt: &[u32], n: usize) -> Result<Vec<u32>, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
        }
        Ok(ids)
    }

    /// Every flag this CLI reads; any other argument is refused by name, never ignored.
    const KNOWN: [&str; 7] = [
        "--tokens", "--prompt", "-n", "--ctx", "--place", "--mode", "--plan",
    ];

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            if !KNOWN.contains(&args[i].as_str()) {
                return Err(format!(
                    "generate_kimi_k3: {} is no flag of this CLI (one of {KNOWN:?})",
                    args[i]
                )
                .into());
            }
            i += if args[i] == "--plan" { 1 } else { 2 };
        }
        let plan_only = args.iter().any(|a| a == "--plan");
        let n: usize = flag(&args, "-n")?.map_or(Ok(8), |s| s.parse())?;
        let ctx: usize = flag(&args, "--ctx")?.map_or(Ok(512), |s| s.parse())?;
        let mode = match flag(&args, "--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => {
                return Err(format!("generate_kimi_k3: --mode is eager or graph, not {o}").into());
            }
        };
        let place = Place::parse(flag(&args, "--place")?.as_deref().unwrap_or("a"))?;
        let (tokens, text) = (flag(&args, "--tokens")?, flag(&args, "--prompt")?);
        if tokens.is_some() == text.is_some() && !plan_only {
            return Err("generate_kimi_k3: one prompt: --tokens a,b,c or --prompt TEXT".into());
        }

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read_without_chat(&split)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let census = gpu_census::census()?;
        let chosen = Place::choose_untiered(Some(place), &census, "kimi-k3")?;
        chosen.record().eprint();
        let machine = chosen.place.machine_untiered()?(inputs.model.layers);
        let plan = inputs
            .plan(&machine, ctx as u64, &PlanLevers::default())
            .map_err(|e| format!("--place {}: {e}", chosen.place.name()))?;
        record::plan(chosen.place.name(), &machine, &plan).eprint();
        need::print(&plan, 0, 0, true)?;
        if plan_only {
            return Ok(());
        }
        drop(plan);

        let tok = text
            .is_some()
            .then(|| Tokenizer::from_gguf(&path))
            .transpose()?;
        let ids: Vec<u32> = match (&tokens, &text) {
            (Some(list), _) => list
                .split(',')
                .map(|t| t.trim().parse::<u32>())
                .collect::<Result<_, _>>()?,
            (_, Some(t)) => tok.as_ref().ok_or("no tokenizer")?.encode(t, true, false),
            _ => unreachable!("one prompt by the check above"),
        };
        if ids.is_empty() || n == 0 {
            return Err(format!(
                "generate_kimi_k3: a prompt of {} ids and -n {n}: each is at least 1",
                ids.len()
            )
            .into());
        }
        if ids.len() + n > ctx {
            return Err(format!(
                "generate_kimi_k3: a prompt of {} ids and {n} generated exceed --ctx {ctx}",
                ids.len()
            )
            .into());
        }
        let t = std::time::Instant::now();
        let mut m = open(split, ctx, mode)?;
        println!(
            "load arch={} resident_bytes={} ctx={ctx} layers={} load_s={:.1}",
            <Body as ChainBody>::arch().name(),
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let out = greedy(&mut m, &ids, n)?;
        println!("ids {out:?}");
        if let Some(tok) = tok {
            println!("text {:?}", tok.decode(&out));
        }
        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate_kimi_k3: built without the `gpu` feature");
    std::process::exit(2);
}
