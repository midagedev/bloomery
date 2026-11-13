//! `generate_exaone` — K-EXAONE-2.0's collecting CLI: the importance matrix of a text corpus off
//! this engine's own prompt batch, the walk the composed head128 program runs
//! ([`bloomery_gpu_flint::prefill`] through [`bloomery_gpu::GpuModel::run_rows`]).
//!
//!     BLOOMERY_MODEL=exaone-moe generate_exaone --imatrix OUT --text FILE [--ctx C] [--chunks N]
//!                            [--save-every K] [--resume] [--arm a|bp] [--plan]
//!
//! Defaults: `--ctx 512`, `--arm a`. The model file is `ref_model_path()`. The file is planned
//! on the arm's machine: `a` as the exaone gates' seat plans it (the description's own room at
//! [`PLAN_ROOM`], the gate card's placed load), `bp` at the host room this machine reads
//! (`BLOOMERY_HOST_ROOM` where set), the room its load checks — the workstation's
//! stage-plus-tier plan with the host tier and the NVMe arena behind it — the plan a bootstrap file needs, its experts nearly
//! all on the host); `--plan` prints the plan record and ends before the load. The corpus is cut
//! and run by the shared collecting walk (`bloomery_gpu_gates::imatrix_walk`, `llama-imatrix`'s
//! own chunking): every chunk from an empty model, the collector attached — the attention
//! projections, the router, the routed stacks' gate and up over every pick and their downs over
//! the card route's own, the shared expert's and the dense lead's gate and up
//! ([`bloomery_gpu::imatrix`]'s module doc names what no walk collects and why). `--save-every K`
//! writes the collection so far every K chunks and `--resume` continues that file, so an
//! hours-long run restarts where it died.
//!
//! The lines on stdout: `load` (the resident bytes, the context, seconds) and the walk's
//! `imatrix` lines. The plan's records go to stderr.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::record::at_main("generate_exaone", &KINDS);
    bloomery_gpu_gates::exit_with("generate_exaone", cli::run())
}

/// What `generate_exaone` prints, in the order it prints them: the plan record.
#[cfg(feature = "flint")]
static KINDS: [&bloomery_record::Kind; 1] = [&bloomery_gpu_gates::record::PLAN];

#[cfg(feature = "flint")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "flint")]
mod cli {
    use crate::lcpp_e2e::Arm;
    use bloomery_gpu::host::served::prompt_pass;
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu_flint::{ExaoneBody, prefill};
    use bloomery_gpu_gates::imatrix_walk;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CARD_MARGIN, HOST_LOCK, HOST_POPULATE, TIER_NODES,
    };
    use gguf::Split;
    use model::arch::flint::place::{self, PlanInputs};
    use model::placement::PlanLevers;
    use model::placement::workstation::{HostRead, PLAN_ROOM};
    use tokenizer::Tokenizer;

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
        "--imatrix",
        "--text",
        "--ctx",
        "--chunks",
        "--save-every",
        "--resume",
        "--arm",
        "--plan",
    ];

    /// The argument after flag `name`: `None` when the flag is absent, an error when it is last.
    fn flag(args: &[String], name: &str) -> Result<Option<String>, GateError> {
        args.iter()
            .position(|a| a == name)
            .map(|i| {
                args.get(i + 1).cloned().ok_or_else(|| {
                    GateError::from(format!("generate_exaone: {name} needs a value"))
                })
            })
            .transpose()
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&ACTS_ON)?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            if !KNOWN.contains(&args[i].as_str()) {
                return Err(format!(
                    "generate_exaone: {} is no flag of this CLI (one of {KNOWN:?})",
                    args[i]
                )
                .into());
            }
            i += if args[i] == "--plan" || args[i] == "--resume" {
                1
            } else {
                2
            };
        }
        let plan_only = args.iter().any(|a| a == "--plan");
        let out = flag(&args, "--imatrix")?
            .ok_or("generate_exaone: --imatrix OUT is required (this CLI only collects)")?;
        let text = flag(&args, "--text")?
            .ok_or("generate_exaone: --text FILE is required (the corpus to collect over)")?;
        let ctx: usize = flag(&args, "--ctx")?.map_or(Ok(512), |s| s.parse())?;
        if ctx == 0 {
            return Err("generate_exaone: --ctx is at least 1".into());
        }
        let chunks: Option<usize> = flag(&args, "--chunks")?.map(|s| s.parse()).transpose()?;
        let save_every: Option<usize> = flag(&args, "--save-every")?
            .map(|s| s.parse())
            .transpose()?;
        if save_every.is_some_and(|k| k == 0) {
            return Err("generate_exaone: --save-every is at least 1".into());
        }
        let resume = args.iter().any(|a| a == "--resume");
        let arm = Arm::parse(flag(&args, "--arm")?.as_deref().unwrap_or("a"))?;

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut inputs = PlanInputs::describe(&split)?;
        inputs.room = match arm {
            Arm::A => (PLAN_ROOM, HostRead::Given),
            Arm::Bp => model::placement::workstation::host_available_read()?,
        };
        let place = PlanLevers::from_levers(&levers)?;
        let machine = arm.machine(inputs.model.layers, place::tier_batch(&inputs)?)?;
        let plan = inputs.plan(&machine, u64::try_from(ctx)?, &place)?;
        record::plan(arm.word(), &machine, &plan).eprint();
        if plan_only {
            return Ok(());
        }
        let tok = Tokenizer::from_gguf(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let t = std::time::Instant::now();
        let mut m = ExaoneBody::open_placed(split, &plan, &inputs, 0, levers.host())?;
        println!(
            "load arch={} resident_bytes={} ctx={ctx} layers={} load_s={:.1}",
            <ExaoneBody as ChainBody>::arch().name(),
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let done = if resume {
            imatrix_walk::resume(m.gpu(), &out)?
        } else {
            0
        };
        imatrix_walk::collect(
            &mut m,
            &tok,
            (&text, &out),
            (ctx, chunks),
            save_every,
            done,
            |m, chunk| {
                m.reset()?;
                prompt_pass(m, |m| {
                    m.run_rows(chunk.len(), "generate_exaone", |gpu, w, body, head, at| {
                        prefill(gpu, w, body, head, chunk, at)
                    })
                })?;
                Ok(())
            },
        )
    }
}

#[cfg(not(feature = "flint"))]
fn main() {
    eprintln!("generate_exaone: built without the `flint` feature");
    std::process::exit(2);
}
