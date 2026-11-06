//! `generate_kimi_k3` — Kimi K3's end-to-end decode CLI: a prompt of ids (or text, raw, no chat
//! template) walked through the decode step one id at a time, then `-n` greedy ids, printed.
//!
//!     BLOOMERY_MODEL=kimi-k3 generate_kimi_k3 [--tokens a,b,c | --prompt TEXT] [-n N]
//!                            [--ctx C] [--place a|bp|gate] [--mode eager|graph] [--plan]
//!                            [--granule BYTES]
//!
//! Defaults: `-n 8`, `--ctx 512`, `--place a`, `--mode graph`. The model is `ref_model_path()`. The
//! file is read as the plan reads it ([`PlanInputs::read_without_chat_for`]: every part the body
//! does not run is refused by name, the template's tool-call parser excepted, and so is a `--ctx`
//! past a K3.1 file's `attention.indexer.top_k`, where its selection would pick among the positions
//! and the body attends them all) and planned on `--place` before anything loads; `--plan` ends
//! there. A placed plan the stage card's dense trunk alone refuses
//! (`PlacementError::CardFreeFloor`) is made again with the head read on the host (no card carries
//! it); the `head on host` line on stderr says so and carries the refusal.
//!
//! `--place a` loads the whole model on one card ([`bloomery_gpu::GpuModel::open`]): a file whose
//! tensors exceed the card's free bytes is refused by name by the load. Any other placement
//! (`bp`: the stage card, the second card as the expert tier and the host and NVMe tiers behind
//! them, `gate`) loads by the plan ([`bloomery_gpu::GpuModel::open_placed`] through
//! `Body::open_placed`): the dense trunk and each layer's card experts on the stage card, the
//! embedding as a host row, the tier cards' experts under the host tier (a node's, with
//! `BLOOMERY_TIER_NODES`, served by `tier_server`). The experts a plan leaves off the cards run
//! on the host tier (`model::ops::ExpertForm::Situ`), the NVMe tier paging what the room cannot
//! hold. The prompt walk is the body's `Prompt` (its ids stepped one at a time; the body has no
//! rows pass).
//!
//! The plan is made under `BLOOMERY_CARD_BUDGET` and `BLOOMERY_CARD_MARGIN` when they are set.
//! `--granule BYTES` counts each card's uploads at that allocator granule instead of the
//! census's (the workstation's 2 MiB when the flag is absent): a fixture's experts fit the shared
//! granules the dense trunk leaves open at 2 MiB, so no budget splits them between the cards, and
//! a finer granule gives the budget a window. A value that is not a power of two of at least 4 KiB
//! is refused by name. It is a plan input only; the driver allocates as it does.
//!
//! The lines on stdout: `load` (the resident bytes, the context, seconds), `ids` (the generated
//! ids), `logits` (an FNV-1a over the last row's bits) and, with the vocabulary's pieces, `text`.
//! The plan's records go to stderr.

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
    use bloomery_gpu_gates::generate::{Place, TierRule};
    use bloomery_gpu_gates::need;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, fnv_row, gpu_census, ref_model_path};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CARD_MARGIN, HOST_LOCK, HOST_POPULATE, HostCfg, TIER_NODES,
    };
    use gguf::Split;
    use model::arch::kimi_k3::place::{PlaceError, PlanInputs, tier_batch};
    use model::placement::{Machine, PlacementError, Plan, PlanLevers};
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

    /// The model of `file` as `plan` places it (card 0 the stage), stepping in `mode`. The body's
    /// own refusals reach the caller by name.
    fn open_placed(
        file: Split,
        (plan, inputs): (&Plan<'_>, &PlanInputs),
        host: HostCfg,
        mode: StepMode,
    ) -> Result<GpuModel<Body>, GateError> {
        let mut m = GpuModel::<Body>::open_placed(file, plan, inputs, 0, host)?;
        m.set_mode(mode);
        Ok(m)
    }

    /// `n` greedy ids after `prompt`, from the model's position: the argmax after the prompt
    /// stepped, then each id fed back one a step.
    fn greedy(m: &mut GpuModel<Body>, prompt: &[u32], n: usize) -> Result<Vec<u32>, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
        }
        Ok(ids)
    }

    /// The tier cards the body serves: one, the stage card's neighbour. No sitting has shown the
    /// tier not slower, so the flag defaults to `a` and the rule never picks.
    const TIER_RULE: TierRule = TierRule {
        tiers: 1,
        break_even: None,
        basis: "",
    };

    /// The levers this CLI acts on: the card budget and margin the plan is made under, the host
    /// set's load and the expert tier's nodes.
    const ACTS_ON: [&str; 6] = [
        CARD_BUDGET,
        CARD_MARGIN,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        TIER_NODES,
    ];

    /// Every flag this CLI reads; any other argument is refused by name, never ignored.
    const KNOWN: [&str; 8] = [
        "--tokens",
        "--prompt",
        "-n",
        "--ctx",
        "--place",
        "--mode",
        "--plan",
        "--granule",
    ];

    /// The plan on `machine` and the machine it is on. A placed plan (`placed` names its place)
    /// the stage card's dense trunk alone refuses (its bytes past the card's free bytes) is made
    /// again on `headless`, the same machine with no card carrying the head, which the host reads.
    fn plan_for<'a>(
        inputs: &'a PlanInputs,
        (machine, headless): (&'a Machine, &'a Machine),
        ctx: u64,
        levers: &PlanLevers,
        placed: Option<&str>,
    ) -> Result<(&'a Machine, Plan<'a>), GateError> {
        match (inputs.plan(machine, ctx, levers), placed) {
            (Err(PlaceError::Placement(PlacementError::CardFreeFloor(floor))), Some(name)) => {
                let on_host = inputs.plan(headless, ctx, levers).map_err(|e| {
                    format!("--place {name}: {floor}; and with the head on the host: {e}")
                })?;
                eprintln!(
                    "head on host: the plan with the head on the stage card was refused ({floor}); \
                     the plan below reads the head on the host"
                );
                Ok((headless, on_host))
            }
            (other, _) => Ok((
                machine,
                other.map_err(|e| format!("--place {}: {e}", placed.unwrap_or("a")))?,
            )),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&ACTS_ON)?;
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
        let inputs = PlanInputs::read_without_chat_for(&split, ctx as u64)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let census = gpu_census::census()?;
        let chosen = Place::choose(Some(place), &census, TIER_RULE, |_| {
            Err("--place unset is never asked: the flag defaults to a".into())
        })?;
        chosen.record().eprint();
        chosen.place.serves("kimi-k3", TIER_RULE.tiers)?;
        let tiered = !chosen.place.tier_cards().is_empty();
        let batch = tiered
            .then(|| tier_batch(&inputs.hp))
            .transpose()
            .map_err(|e| format!("--place {}: {e}", chosen.place.name()))?;
        let mut machine = chosen.place.machine(None, batch)?(inputs.model.layers);
        if let Some(g) = flag(&args, "--granule")? {
            let g: u64 = g.parse()?;
            if !g.is_power_of_two() || g < 4 << 10 {
                return Err(format!(
                    "generate_kimi_k3: --granule {g}: a power of two of at least 4096 bytes (the \
                     default is the workstation's 2 MiB)"
                )
                .into());
            }
            let granule = std::num::NonZeroU64::new(g).ok_or("--granule is nonzero")?;
            for card in machine.cards.iter_mut().chain(machine.tiers.iter_mut()) {
                card.granule_bytes = granule;
            }
        }
        let plan_levers = PlanLevers::from_levers(&levers)?;
        let placed = chosen.place.name() != "a";
        let headless = {
            let mut m = machine.clone();
            m.cards.iter_mut().for_each(|c| c.head = false);
            m
        };
        let (machine, plan) = plan_for(
            &inputs,
            (&machine, &headless),
            ctx as u64,
            &plan_levers,
            placed.then(|| chosen.place.name()),
        )?;
        record::plan(chosen.place.name(), machine, &plan).eprint();
        need::print(&plan, 0, 0, true)?;
        if plan_only {
            return Ok(());
        }

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
        let mut m = if placed {
            open_placed(split, (&plan, &inputs), levers.host(), mode)?
        } else {
            open(split, ctx, mode)?
        };
        println!(
            "load arch={} resident_bytes={} ctx={ctx} layers={} load_s={:.1}",
            <Body as ChainBody>::arch().name(),
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let out = greedy(&mut m, &ids, n)?;
        println!("ids {out:?}");
        println!("logits fnv1a64={:016x}", fnv_row(&m.logits()?));
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
