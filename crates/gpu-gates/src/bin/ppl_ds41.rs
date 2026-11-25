//! `ppl_ds41` — the V4.1 loss tool: the engine's own run of a DeepSeek-V4.1-Flash GGUF over a fixed id list, written
//! as an ik KL-divergence base file (`bloomery_gpu_gates::kld`), so that `kld_diff` prints how far one quantization
//! of the model sits from another on the same text, with the one engine that reads every file of the family. Its
//! second mode is the engine's importance-matrix collection of the same file (`--imatrix`, below).
//!
//!     BLOOMERY_V41_MODEL=<first shard> ppl_ds41 (--ids FILE | --ids-of BASE.kld) --run CTX[xCHUNKS]=OUT.kld ...
//!                            [--place PLACE] [--plan-ctx C] [--plan]
//!     kld_diff A.kld B.kld                    mean KLD ± SE, p50/p99/max, same top, PPL of A and of B
//!     BLOOMERY_V41_MODEL=<first shard> ppl_ds41 --imatrix OUT --text FILE [--ctx C] [--chunks N]
//!                            [--save-every K] [--resume] [--place PLACE] [--plan]
//!
//! The command line is `bloomery_gpu_gates::kld::cli`'s (`--ids`, `--ids-of`, `--run`, `--plan-ctx`; `ppl_exaone`
//! reads it too): `--ids-of` takes the ids an existing base file holds, so a file of this tool and the ik base it
//! is compared with share their text. A chunk runs from an empty model one id a step (`shared/ds41_ppl.rs`, the
//! engine's own step and reset: the prompt batch is never used), and a record is written at every position from
//! `CTX/2` through `CTX-2`, scored against the id after it, the ids as given and no BOS put in a chunk's first
//! place (the V4.1 file sets `tokenizer.ggml.add_bos_token` to false). A file is written whole or not at all
//! (`<OUT>.part`, then a rename), and an existing `OUT` is refused before the load.
//!
//! The model opens as `generate_ds41` opens it: `--place` as the serve takes it (unset: the common rule on the cards
//! in view, `a` on one card), the levers at their serving defaults (`BLOOMERY_RESIDENCY` unset resolves against the
//! plan), the plan made at `--plan-ctx` positions (default 8192, so the files of one tool run under one placement;
//! a `CTX` above it is refused), the step captured as a graph. `--plan` prints the `place unset`, `residency lever`
//! and `plan` records and exits before the load: a file the plan refuses is refused there.
//!
//! The lines on stdout: the records above, `load` (the resident bytes, the context, seconds), a progress line every
//! 256 scored positions and a line a chunk (cumulative seconds, so a chunk's own steps a second are the chunk's
//! `CTX - 1` over the difference of two lines), and a `kld_base` line a run.
//!
//! `--imatrix OUT --text FILE` collects the importance matrix the file's quantization is weighted by, in the form
//! and with the flags of `generate_kimi_k3 --imatrix` (`bloomery_gpu_gates::imatrix_walk`: the text tokenized as
//! `llama-imatrix` reads it, `--ctx` ids a chunk, default 512, at least two chunks, `--chunks N` the first N,
//! `--save-every K` and `--resume`). It lives in this tool because this is V4.1's lean open: the residency pre-plan
//! and one load, none of `generate_ds41`'s arms, timing or draft, which a collecting walk of hours has no use for.
//! A chunk is one prompt call (`shared/ds41_imatrix.rs`): the prompt batch is where the collector's feeds sit, the
//! step's captured launches hold none. The load takes the step eager, the plan is made at `--ctx` positions, and the
//! prompt triangle is off (`BLOOMERY_CED` is not read here): a layer then runs every position, so a weight's counts
//! are the rows `llama-imatrix` feeds. The card's routed experts run on the tile path (`Body::set_batch_card_tiles`),
//! which holds the SwiGLU rows as f32 for the down; a layer the common card leg runs feeds them from the leg. The
//! gate and up of every pick are collected whatever tier computes the expert; a down only for the picks the card
//! computed (`BLOOMERY_HOSTSTREAM=on` at the load streams the host tier's experts through the card, so every pick is
//! one). The lines on stdout are the walk's: `load`, an `imatrix chunk=` line a chunk and a closing `imatrix
//! chunks=` line; the collector's own gaps are named in `bloomery_gpu::imatrix`.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("ppl_ds41", tool::run())
}

#[cfg(feature = "deepseek41")]
#[allow(
    dead_code,
    reason = "the V4.1 binaries share the placement rules; each takes the calls it reads"
)]
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_ppl.rs"]
mod ppl;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_imatrix.rs"]
mod imx;

#[cfg(feature = "deepseek41")]
mod tool {
    use app::arch::deepseek41::Ds41Cfg;
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model};
    use bloomery_gpu_deepseek41::swap;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::imatrix_walk;
    use bloomery_gpu_gates::kld::cli::{Args, Spec};
    use bloomery_gpu_gates::need;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, ref_model_path, residency};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CED, ENGRAM_HELPER, EXPERT_RANK, HOST_LANES, HOST_LOCK,
        HOST_POPULATE, HOSTSTREAM, Levers, PIN_MAIN, PREFILL, PREFILL_CUT, PREFILL_GROUP, R8,
        RESIDENCY, ROUTE_CENSUS, ROUTE_MASK, ROUTE_RESIDENT_BONUS, ROUTE_TOPK, ResidencyAt,
        ResidencyPick,
    };
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{Machine, Plan, workstation};
    use std::path::PathBuf;
    use tokenizer::Tokenizer;

    use crate::place;

    /// This tool's flags beside the loss tools' common ones (`kld::cli`).
    const SPEC: Spec = Spec {
        tool: "ppl_ds41",
        own: &["--place"],
        switches: &["--plan"],
        plan_only: Some("--plan"),
    };

    /// The levers the open acts on: `generate_ds41`'s for a plain run, none of its draft's, its timing's or its
    /// slots'.
    const ACTS_ON: &[&str] = &[
        CED,
        PREFILL,
        PREFILL_GROUP,
        PREFILL_CUT,
        ENGRAM_HELPER,
        CARD_BUDGET,
        EXPERT_RANK,
        PIN_MAIN,
        HOST_LANES,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        R8,
        RESIDENCY,
        HOSTSTREAM,
        ROUTE_CENSUS,
        ROUTE_MASK,
        ROUTE_RESIDENT_BONUS,
        ROUTE_TOPK,
    ];

    /// What the open prints of itself: the plan record, the churn pool's and the host need beside it, and whether
    /// the load follows.
    struct Log {
        plan_only: bool,
        /// The residency word as resolved against the pre-plan, for the `residency host` record.
        residency: ResidencyPick,
        /// The pre-plan's per-layer card experts: the open's own plan must equal them.
        plan_n_l: Vec<u64>,
        /// The word the open loads under.
        word: Residency,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            _: &PlanInputs,
            machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            record::plan(place, machine, plan).print();
            // The open's plan is the pre-plan the unset word resolved against; a load whose plan moved under it
            // names itself.
            if plan.n_l != self.plan_n_l {
                return Err(SessionError::Refused(format!(
                    "the plan the residency resolved on holds per-layer card experts {:?}; the open's holds {:?}",
                    self.plan_n_l, plan.n_l
                )));
            }
            let mut pool_bytes = 0;
            if let Some(pool) = swap::churn(plan, 0, self.word)? {
                record::residency_host(self.residency.word, &pool, plan).print();
                pool_bytes = pool.bytes;
            }
            need::print(plan, pool_bytes, 0, false).map_err(SessionError::Caller)?;
            Ok(!self.plan_only)
        }

        fn load(&mut self, _: &Deepseek41Model) -> Result<(), SessionError> {
            Ok(())
        }

        fn capture(&mut self, _: usize) -> Result<(), SessionError> {
            Ok(())
        }

        /// The card-byte invariant's readings, printed and refused as this process's card hold says.
        fn card_watch(&mut self) -> Result<Box<dyn app::CardWatch>, SessionError> {
            bloomery_gpu_gates::card_bytes::watch()
        }
    }

    /// The plan the load will use, made once before the open from the inputs the open plans with (the file's
    /// headers, the placement's machine, the context and the levers), for the unset residency word to resolve against.
    /// Returns the plan's per-layer card experts and the word it left.
    fn residency_at_plan(
        file: &Split,
        machine: impl Fn(usize) -> Machine,
        ctx: usize,
        open: &body::OpenCfg,
        pick: ResidencyPick,
    ) -> Result<(Vec<u64>, ResidencyPick), GateError> {
        const WHAT: &str = "ppl_ds41 residency pre-plan";
        let mut inputs = PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        open.state_room(&mut inputs);
        let machine = machine(inputs.model.layers);
        open.page_inputs(&mut inputs, &machine)?;
        let plan = inputs
            .plan(&machine, u64::try_from(ctx)?, &open.place)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let pick = residency::at_plan(<Body as ChainBody>::UNHOSTED_POOL, &plan, 0, pick)?;
        Ok((plan.n_l, pick))
    }

    /// A loaded model with what the tool prints beside it.
    struct Opened {
        session: Session<Body>,
        path: PathBuf,
    }

    /// What a run asks of the open beyond the placement: `plan_only` ends it after the plan's records, and
    /// `collect` loads for the importance walk (the step eager, the prompt triangle off).
    #[derive(Clone, Copy)]
    struct How {
        plan_only: bool,
        collect: bool,
    }

    /// The load both modes make, as `generate_ds41` opens it; `None` when the plan's records were all that was asked.
    fn open(
        levers: &Levers,
        at_place: Place,
        plan_ctx: usize,
        how: How,
    ) -> Result<Option<Opened>, GateError> {
        let at = ResidencyAt {
            serving_place: at_place != Place::Gate,
            check_finite: false,
            route_trace: false,
            prefill_steps: body::PrefillMode::from_name(levers.prefill())
                == Some(body::PrefillMode::Steps),
        };
        let cfg = body::OpenCfg::from_levers_at(levers, at)?;
        if cfg.body.residency != Residency::Off && at_place == Place::Gate {
            return Err(format!(
                "BLOOMERY_RESIDENCY={}: the residency machine runs under --place a and bp, not gate",
                levers.residency_at(at).word
            )
            .into());
        }
        let pinned = levers.pin_main() && threads::pool().pin_caller();

        let path = ref_model_path()?;
        let t = std::time::Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let tier_batch = place::tier_batch(
            at_place,
            &Hparams::read(&file).map_err(|e| format!("{}: {e}", path.display()))?,
        );
        let machine = at_place.machine(None, tier_batch)?;
        // The room is read once: the pre-plan and the open plan from it, so the headroom the
        // unset word was resolved against is the one the open holds its churn pool to.
        let room = workstation::host_available_read()?;
        let (plan_n_l, pick) = residency_at_plan(
            &file,
            machine,
            plan_ctx,
            &cfg.with_host_reading(room.clone()),
            levers.residency_at(at),
        )?;
        record::residency_lever(pick).print();
        let word = Residency::parse(pick.word)?;
        let mut cfg = body::OpenCfg::from_levers_with(levers, word)?.with_host_reading(room);
        if how.collect {
            // Every layer runs every position, so a weight's counts are the rows fed.
            cfg.body.ced = false;
        }
        let open = OpenArgs {
            place: at_place.name(),
            machine,
            ctx: plan_ctx,
            mode: if how.collect {
                StepMode::Eager
            } else {
                StepMode::Graph
            },
            cfg: Ds41Cfg {
                open: cfg.clone(),
                feed: cfg.body.prefill,
                card_timing: false,
            },
        };
        let mut log = Log {
            plan_only: how.plan_only,
            residency: pick,
            plan_n_l,
            word,
        };
        let Some(loaded) = Loaded::<Body>::open(file, open, &mut log)? else {
            return Ok(None);
        };
        let session = loaded.ready(&mut log)?;
        println!(
            "load place={} model={} resident_bytes={} ctx={plan_ctx} vocab={} residency={} pinned={pinned} load_s={:.1}",
            at_place.name(),
            path.display(),
            session.model().resident_bytes(),
            session.model().n_vocab()?,
            pick.word,
            t.elapsed().as_secs_f64()
        );
        Ok(Some(Opened { session, path }))
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        model::ops::set_host_lanes(levers.host_lanes());
        let raw: Vec<String> = std::env::args().skip(1).collect();
        if raw.iter().any(|a| a == "--imatrix") {
            return collect(&levers, &raw);
        }
        let args = Args::parse(&SPEC, &raw)?;
        let plan_only = args.switch("--plan");
        let plan_ctx = args.plan_ctx();
        let placed = place::choose(args.value("--place").map(place::parse).transpose()?)?;
        placed.record().print();
        // The count and the format before the load; the vocabulary's bound once the head's width is known.
        let ids = if plan_only { None } else { Some(args.ids()?) };
        let how = How {
            plan_only,
            collect: false,
        };
        let Some(mut opened) = open(&levers, placed.place, plan_ctx, how)? else {
            return Ok(());
        };
        let vocab = opened.session.model_mut().n_vocab()?;
        let ids = ids
            .ok_or("ppl_ds41: a load with no ids")?
            .for_vocab(vocab)?;
        args.walk(
            &mut crate::ppl::Steps(opened.session.model_mut()),
            &ids,
            vocab,
        )
    }

    /// Every flag `--imatrix` reads; any other argument is refused by name, never ignored.
    const IMATRIX_KNOWN: [&str; 8] = [
        "--imatrix",
        "--text",
        "--ctx",
        "--chunks",
        "--save-every",
        "--resume",
        "--place",
        "--plan",
    ];

    /// The context of the chunks the derived wall is stated for: the size a quantization's importance matrix is
    /// collected at.
    const WALL_CTX: usize = 2048;
    /// The number of chunks the derived wall is stated for.
    const WALL_CHUNKS: usize = 40;

    /// The argument after flag `name`: `None` when the flag is absent, an error when it is last.
    fn flag(args: &[String], name: &str) -> Result<Option<String>, GateError> {
        args.iter()
            .position(|a| a == name)
            .map(|i| {
                args.get(i + 1)
                    .cloned()
                    .ok_or_else(|| GateError::from(format!("ppl_ds41: {name} needs a value")))
            })
            .transpose()
    }

    /// The importance walk (`--imatrix`, the module doc): the text cut into chunks of `--ctx` ids, each one prompt
    /// call from an empty model, the collector's file written to `OUT`.
    fn collect(levers: &Levers, args: &[String]) -> Result<(), GateError> {
        let mut i = 0;
        while i < args.len() {
            if !IMATRIX_KNOWN.contains(&args[i].as_str()) {
                return Err(format!(
                    "ppl_ds41: {} is no flag of --imatrix (one of {IMATRIX_KNOWN:?})",
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
        let out = flag(args, "--imatrix")?.ok_or("ppl_ds41: --imatrix OUT is required")?;
        let text = flag(args, "--text")?
            .ok_or("ppl_ds41: --text FILE is required (the corpus to collect over)")?;
        let ctx: usize = flag(args, "--ctx")?.map_or(Ok(512), |s| s.parse())?;
        if ctx == 0 {
            return Err("ppl_ds41: --ctx is at least 1".into());
        }
        let chunks: Option<usize> = flag(args, "--chunks")?.map(|s| s.parse()).transpose()?;
        let save_every: Option<usize> =
            flag(args, "--save-every")?.map(|s| s.parse()).transpose()?;
        if save_every.is_some_and(|k| k == 0) {
            return Err("ppl_ds41: --save-every is at least 1".into());
        }
        let resume = args.iter().any(|a| a == "--resume");
        let placed = place::choose(
            flag(args, "--place")?
                .as_deref()
                .map(place::parse)
                .transpose()?,
        )?;
        placed.record().print();
        let how = How {
            plan_only,
            collect: true,
        };
        let Some(mut opened) = open(levers, placed.place, ctx, how)? else {
            return Ok(());
        };
        let tok = Tokenizer::from_gguf(&opened.path)
            .map_err(|e| format!("{}: {e}", opened.path.display()))?;
        let m = opened.session.model_mut();
        crate::imx::ready(m)?;
        let done = if resume {
            imatrix_walk::resume(m.gpu(), &out)?
        } else {
            0
        };
        let (mut walked, mut walk) = (0usize, std::time::Duration::ZERO);
        imatrix_walk::collect(
            m,
            &tok,
            (&text, &out),
            (ctx, chunks),
            save_every,
            done,
            |m, ids| {
                let t = std::time::Instant::now();
                crate::imx::chunk(m, ids)?;
                walk += t.elapsed();
                walked += 1;
                Ok(())
            },
        )?;
        print_rate(walked, ctx, walk);
        Ok(())
    }

    /// What a longer collection costs at the rate this one walked: the walk's tokens per second and the wall of
    /// [`WALL_CHUNKS`] chunks of [`WALL_CTX`] tokens at that rate (a rate measured at another context is the
    /// estimate, not the rate at [`WALL_CTX`]).
    fn print_rate(walked: usize, ctx: usize, walk: std::time::Duration) {
        let rate = (walked * ctx) as f64 / walk.as_secs_f64();
        println!(
            "imatrix walk chunks={walked} tokens={} walk_s={:.3} tok_s={rate:.1} [derived] c{WALL_CTX}x{WALL_CHUNKS} \
             wall_s={:.0} at ctx={ctx}'s rate",
            walked * ctx,
            walk.as_secs_f64(),
            (WALL_CHUNKS * WALL_CTX) as f64 / rate
        );
    }
}

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("ppl_ds41: built without the `deepseek41` feature");
    std::process::exit(2);
}
