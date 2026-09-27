//! Greedy generation with the GLM-5.3-Flash program: the file planned onto
//! one card, the routed experts the card experts read by the expert rule
//! (the hot list `BLOOMERY_HOT_LIST`, or the id prefix) and the rest on the
//! host tier (`model::arch::glm5next::place`), loaded through the session
//! (`app::Loaded`), the prompt fed one step a position, then `-n` greedy
//! steps.
//!
//! `generate_glm5next --tokens a,b,c [-n N] [--ctx C] [--place a|gate]
//! [--mode graph|eager] [--model PATH] [--logits] [--plan]`
//!
//! - `--tokens`: the prompt's ids (the file's own vocabulary, no BOS added).
//! - `--ctx`: the positions the caches hold; the plan refuses more than the
//!   latent layers attend whole. Default 2048.
//! - `--place`: `a` is the serving plan (`workstation::plan_a`, the A6000),
//!   `gate` the gate card's (`workstation::plan_gate`, the 3090). Default
//!   `gate`.
//! - `--model`: the first shard; default the file the reference sets were
//!   dumped from (`refset::arch::glm5next::MODEL`).
//! - `--plan` prints the plan and exits before the load.
//! - `--logits` prints the head's last logits row by its bits after the
//!   tokens.
//!
//! The `load` line's `top_k` is the file's indexer top-k: at the contexts the
//! plan allows every position is within it, so the latent layers attend all
//! of them. The levers it acts on are parsed once, at `main`; every line is a record
//! of a kind `bloomery_gpu_gates::record` declares (`--records-schema`
//! prints them).

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!("generate_glm5next: built without the `glm5next` feature; see `just gen-glm5next`.");
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_glm5next", cli::run())
}

#[cfg(feature = "glm5next")]
mod cli {
    use std::time::Instant;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, OpenArgs, OpenLog, SessionError};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel};
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HOT_LIST, R8};
    use gguf::Split;
    use model::arch::glm5next::place::PlanInputs;
    use model::placement::{Machine, Plan, PlanLevers, workstation};
    use runtime::{Target, Want};

    const ACTS_ON: &[&str] = &[
        HOT_LIST,
        CARD_BUDGET,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        R8,
    ];

    /// The last value of flag `name`, if given.
    fn flag(name: &str) -> Result<Option<String>, GateError> {
        let args: Vec<String> = std::env::args().collect();
        let mut out = None;
        for (i, a) in args.iter().enumerate() {
            if a == name {
                out = Some(
                    args.get(i + 1)
                        .ok_or_else(|| format!("{name} needs a value"))?
                        .clone(),
                );
            }
        }
        Ok(out)
    }

    fn has(name: &str) -> bool {
        std::env::args().any(|a| a == name)
    }

    /// Comma-separated ids.
    fn ids_of(s: &str) -> Result<Vec<u32>, GateError> {
        Ok(s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()?)
    }

    fn mode_name(m: StepMode) -> &'static str {
        match m {
            StepMode::Eager => "eager",
            StepMode::Graph => "graph",
        }
    }

    /// What the open prints: the plan, the load, the capture.
    struct Log {
        /// The file's indexer top-k, which the plan read.
        top_k: usize,
        place: &'static str,
        mode: StepMode,
        ctx: usize,
        stop_at_plan: bool,
        t: Instant,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            inputs: &PlanInputs,
            machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            record::plan(place, machine, plan, "none").print();
            self.top_k = inputs.hp.indexer.top_k;
            Ok(!self.stop_at_plan)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            let b = m.body("generate_glm5next")?;
            Record::new(&record::LOAD_GENERATOR)
                .u("resident_bytes", m.resident_bytes())
                .u("ctx", self.ctx)
                .u("layers", b.kinds().len())
                .u("top_k", self.top_k)
                .w("shadow", "none")
                .u("shadow_bytes", 0)
                .u("unified_addressing", 0)
                .w("prefill", "steps")
                .w("mode", mode_name(self.mode))
                .w("place", self.place)
                .w("pin_main", "off")
                .w("pinned", false)
                .f("load_s", self.t.elapsed().as_secs_f64())
                .print();
            if let Some(h) = b.hybrid().residency() {
                for r in record::host_residency(h) {
                    r.print();
                }
            }
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            Record::new(&record::CAPTURE)
                .u("graph_nodes", nodes)
                .print();
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        record::at_main("generate_glm5next", record::GENERATE_GLM5NEXT);
        let ids = ids_of(&flag("--tokens")?.ok_or("--tokens a,b,c is required")?)?;
        if ids.is_empty() {
            return Err("--tokens holds no id".into());
        }
        let n_gen: usize = flag("-n")?.map_or(Ok(16), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(2048), |s| s.parse())?;
        let (place, machine): (&'static str, fn(usize) -> Machine) =
            match flag("--place")?.as_deref() {
                None | Some("gate") => ("gate", workstation::plan_gate),
                Some("a") => ("a", workstation::plan_a),
                Some(o) => return Err(format!("--place is a or gate, not {o}").into()),
            };
        let mode = match flag("--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => return Err(format!("--mode is eager or graph, not {o}").into()),
        };
        let path = flag("--model")?.unwrap_or_else(|| refset::arch::glm5next::MODEL.to_string());
        if ids.len() + n_gen > ctx {
            return Err(format!(
                "{} prompt ids and {n_gen} generated exceed --ctx {ctx}",
                ids.len()
            )
            .into());
        }
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(&levers)?,
            host: levers.host(),
        };
        let mut log = Log {
            top_k: 0,
            place,
            mode,
            ctx,
            stop_at_plan: has("--plan"),
            t,
        };
        let args = OpenArgs {
            place,
            machine,
            ctx,
            mode,
            cfg,
        };
        let Some(loaded) = Loaded::<Body>::open(file, args, &mut log)? else {
            return Ok(());
        };
        let mut s = loaded.ready(&mut log)?;
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids
            .iter()
            .copied()
            .skip(ids.len().saturating_sub(4))
            .collect();
        Record::new(&record::FED)
            .u("ids", ids.len())
            .list("first", &head)
            .list("last", &tail)
            .u("depth_sequence_from", ids.len())
            .print();
        let t_feed = Instant::now();
        let mut next = s.prompt(&ids, Want::Argmax)?.argmax();
        Record::new(&record::STEP0)
            .u("pos", s.pos() - 1)
            .u("token", next)
            .u("fed", ids.len())
            .f("feed_s", t_feed.elapsed().as_secs_f64())
            .print();
        let mut tokens = vec![next];
        for i in 1..n_gen {
            let pos = s.pos();
            next = s.step(next, Want::Argmax)?.argmax();
            Record::new(&record::STEP)
                .u("i", i)
                .u("pos", pos)
                .u("token", next)
                .print();
            tokens.push(next);
        }
        Record::new(&record::TOKENS).list("tokens", &tokens).print();
        if has("--logits") {
            let row = s.model().logits()?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(&x.0)))
                .map_or(0, |(i, _)| i);
            let fnv = row.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                v.to_bits().to_le_bytes().iter().fold(h, |h, &b| {
                    (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
                })
            });
            Record::new(&record::LOGITS)
                .u("n", row.len())
                .u("argmax", argmax)
                .w("fnv64", format!("{fnv:016x}"))
                .print();
        }
        Ok(())
    }
}
