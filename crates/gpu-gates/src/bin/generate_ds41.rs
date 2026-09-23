//! `generate_ds41` — the thin end-to-end decode CLI of the V4.1 engine, and
//! its timing ruler: `generate`'s shape over `bloomery_gpu_deepseek41::body`.
//!
//!     generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] [-n N]
//!                   [--ctx C] [--place a|gate] [--mode eager|graph]
//!                   [--time [--warm W]]
//!
//! Defaults: prompt 0 (none when `--depth` is given), N 32, C the serving
//! context (`workstation::CTX_MAX`), place `a`, mode graph, W 0. A flag given
//! twice takes its last value, so a recipe's default can be overridden by
//! the arguments after it.
//!
//! Greedy only, one token per step: V4.1 has no batched prefill in the
//! chain, so the prompt is fed one real step per token too. Prompt ids are
//! the V4.1 file's own: row P of `tools/ref/prompts.tsv` as
//! `tools/ref/ik-greedy.sh` tokenized it into
//! `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv` — the ids the step gate's
//! `--greedy` and ik's continuation read.
//!
//! `--place` is the placement the engine loads by: `a` is the serving plan
//! (`workstation::plan_a`, every layer and the head on the A6000, each routed
//! layer's expert prefix the budget allows on the card, the rest on the host
//! tier) and the one the timing runners use; `gate` is the step gate's
//! (`workstation::plan_gate`, the same on the 3090). The card is found by
//! name, so the box's card pin decides which placements can load.
//!
//! `--depth D` stands the run at depth D before its first generated token:
//! the prompt, then the depth tables' sequence (`lcg_prompt` in
//! `tools/ref/lease.sh`) from index `P` on, every id one real step. With no
//! prompt the fed ids are exactly `lcg_prompt D`, the ids the V2-Lite depth
//! runner feeds. There is no synthetic depth: `seed_depth` refuses on this
//! body.
//!
//! Any depth up to `--ctx` runs: every indexer layer selects its stream's
//! list at every position (the identity while the visible rows fit in
//! `top_k`, the indexer's top-k after). The run selects with the file's
//! `top_k`, the model's own; the load line prints it, and a body that
//! loaded with another is refused before the first step.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh generate_ds41 … --time`,
//! `tools/ref/depth-ds41.sh`). It times each generated step host-side around
//! `step(&[tok])`; the argmax readback synchronizes inside, and the host
//! tier's share runs inside it, so the wall time is the whole step. The
//! prompt and the depth ids are fed untimed. Nothing is printed between two
//! timed steps. `--warm W` drops the first W generated steps from the
//! statistics and still prints them, marked. The `SMOKE` footer carries the
//! keys `generate`'s does (`p50_ms=`, `mean_ms=`, `warm=`), so the runners
//! that read one read the other.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("generate_ds41: built without the `deepseek41` feature; see `just gen-ds41`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_ds41", drive::run())
}

#[cfg(feature = "deepseek41")]
mod drive {
    use std::time::Instant;

    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::{GateError, data_dir, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{Machine, workstation};

    const USAGE: &str = "usage: generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] \
                         [-n N] [--ctx C] [--place a|gate] [--mode eager|graph] \
                         [--time [--warm W]]";

    /// Which placement the engine loads by.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Place {
        /// The serving plan, on the A6000.
        A,
        /// The step gate's plan, on the 3090.
        Gate,
    }

    impl Place {
        fn machine(self) -> fn(usize) -> Machine {
            match self {
                Place::A => workstation::plan_a,
                Place::Gate => workstation::plan_gate,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Place::A => "a",
                Place::Gate => "gate",
            }
        }
    }

    /// Where the prompt comes from.
    enum Prompt {
        /// Row 0, or nothing under `--depth`.
        Default,
        Row(u32),
        Ids(Vec<u32>),
    }

    struct Args {
        prompt: Prompt,
        depth: Option<usize>,
        n_gen: usize,
        ctx: usize,
        place: Place,
        mode: StepMode,
        timed: bool,
        warm: Option<usize>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut a = Args {
            prompt: Prompt::Default,
            depth: None,
            n_gen: 32,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            place: Place::A,
            mode: StepMode::Graph,
            timed: false,
            warm: None,
        };
        let (mut row, mut ids) = (None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--time" {
                a.timed = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
            match flag.as_str() {
                "--prompt-id" => row = Some(v.parse()?),
                "--tokens" => {
                    ids = Some(
                        v.split(',')
                            .map(|t| t.trim().parse::<u32>())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                "--depth" => a.depth = Some(v.parse()?),
                "-n" => a.n_gen = v.parse()?,
                "--ctx" => a.ctx = v.parse()?,
                "--warm" => a.warm = Some(v.parse()?),
                "--place" => {
                    a.place = match v.as_str() {
                        "a" => Place::A,
                        "gate" => Place::Gate,
                        other => return Err(format!("--place is a or gate, not {other}").into()),
                    };
                }
                "--mode" => {
                    a.mode = match v.as_str() {
                        "graph" => StepMode::Graph,
                        "eager" => StepMode::Eager,
                        other => {
                            return Err(format!("--mode is eager or graph, not {other}").into());
                        }
                    };
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        a.prompt = match (row, ids) {
            (Some(_), Some(_)) => {
                return Err("--prompt-id and --tokens both name the prompt. Pass one.".into());
            }
            (Some(p), None) => Prompt::Row(p),
            (None, Some(v)) => Prompt::Ids(v),
            (None, None) => Prompt::Default,
        };
        check_counts(&a)?;
        Ok(a)
    }

    /// Refused rather than ignored, as `generate` refuses them: a `--warm`
    /// with nothing to trim, and one that trims every timed step.
    fn check_counts(a: &Args) -> Result<(), GateError> {
        if a.n_gen == 0 {
            return Err("-n wants at least one generated token".into());
        }
        if a.timed && a.n_gen < 2 {
            return Err(
                "--time with -n 1 has no generated step to time: token 0 comes out of the \
                 prompt's own step"
                    .into(),
            );
        }
        match a.warm {
            Some(_) if !a.timed => Err("--warm drops the first W steps from --time's \
                                        statistics, which this run has not asked for. Pass \
                                        both, or neither."
                .into()),
            Some(w) if w >= a.n_gen - 1 => Err(format!(
                "--warm {w} leaves no timed step of the {} that -n {} generates",
                a.n_gen - 1,
                a.n_gen
            )
            .into()),
            _ => Ok(()),
        }
    }

    /// The prompt's ids from `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv`
    /// (`tools/ref/ik-greedy.sh`): the first row's third column.
    fn prompt_row(p: u32) -> Result<Vec<u32>, GateError> {
        let path = data_dir()
            .join("greedy-ds41")
            .join(format!("prompt{p}.tsv"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e} — run just ik-greedy-ds41 {p}", path.display()))?;
        let row = text
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or_else(|| format!("{}: no prompt row", path.display()))?;
        let ids = row
            .split('\t')
            .nth(2)
            .ok_or_else(|| format!("{}: the row has no ids", path.display()))?;
        Ok(ids
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?)
    }

    /// The depth tables' sequence, `lcg_prompt` of `tools/ref/lease.sh`: 100000,
    /// then an LCG walk over [1000, 91000). awk computes it in doubles, so this
    /// does too — past the second id the exact 64-bit LCG is another sequence.
    fn depth_ids(n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        let mut s: f64 = 12345.0;
        for i in 0..n {
            if i == 0 {
                out.push(100_000);
                continue;
            }
            s = (s * 1_103_515_245.0 + 12345.0) % 2_147_483_648.0;
            // An integer-valued double in [0, 90000): the cast is exact.
            out.push(1000 + (s % 90_000.0) as u32);
        }
        out
    }

    /// The ids the run feeds before its first generated token, and how many
    /// of them are the prompt's.
    fn fed_ids(a: &Args) -> Result<(Vec<u32>, usize), GateError> {
        let mut ids = match &a.prompt {
            Prompt::Ids(v) => v.clone(),
            Prompt::Row(p) => prompt_row(*p)?,
            Prompt::Default if a.depth.is_some() => Vec::new(),
            Prompt::Default => prompt_row(0)?,
        };
        let prompt_len = ids.len();
        if let Some(d) = a.depth {
            if d < prompt_len {
                return Err(
                    format!("--depth {d} is shorter than the prompt's {prompt_len} ids").into(),
                );
            }
            ids.extend_from_slice(&depth_ids(d)[prompt_len..]);
        }
        if ids.is_empty() {
            return Err("an empty prompt and no --depth: nothing to feed".into());
        }
        Ok((ids, prompt_len))
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        let (ids, prompt_len) = fed_ids(&a)?;
        let depth = ids.len();
        // Positions 0 .. depth − 1 are the fed ids; the N − 1 feedback steps
        // take depth .. depth + N − 2.
        let fed = depth + a.n_gen - 1;
        if fed > a.ctx {
            return Err(format!(
                "depth {depth} + {} fed tokens exceed --ctx {}",
                a.n_gen - 1,
                a.ctx
            )
            .into());
        }

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        print_plan(&inputs, a.place, a.ctx)?;
        drop(split);

        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut m = body::open(file, a.place.machine(), a.ctx)?;
        m.set_mode(a.mode);
        let top_k = m.body("generate_ds41")?.indexer_top_k();
        if top_k != inputs.hp.indexer.top_k {
            return Err(format!(
                "the body selects {top_k} rows per stream, the file's top_k is {}: a step \
                 past that many visible rows would not be the model's",
                inputs.hp.indexer.top_k
            )
            .into());
        }
        println!(
            "load resident_bytes={} ctx={} layers={} top_k={top_k} mode={} place={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            a.ctx,
            inputs.hp.n_layer,
            mode_name(a.mode),
            a.place.name(),
            t.elapsed().as_secs_f64()
        );
        if a.mode == StepMode::Graph {
            // Captured before the prompt, so the first timed step is a replay.
            println!("capture graph_nodes={}", m.capture_step()?);
        }
        decode(&mut m, &a, &ids, prompt_len)
    }

    /// The plan the engine is about to load: where the experts sit.
    fn print_plan(inputs: &PlanInputs, place: Place, ctx: usize) -> Result<(), GateError> {
        let machine = place.machine()(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?)?;
        let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
        let card = &plan.cards[0];
        println!(
            "plan place={} card={} ctx_max={} card_experts={} ({} B) host_experts={} ({} B) \
             n_l={}..{} on {} layers",
            place.name(),
            machine.cards[0].name,
            plan.ctx_max,
            card.experts,
            card.expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            held.iter().min().copied().unwrap_or(0),
            held.iter().max().copied().unwrap_or(0),
            held.len()
        );
        Ok(())
    }

    fn mode_name(mode: StepMode) -> &'static str {
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        }
    }

    /// Feed `ids` untimed, then the N − 1 feedback steps, timed when asked;
    /// every line after the loop.
    fn decode(
        m: &mut Deepseek41Model,
        a: &Args,
        ids: &[u32],
        prompt_len: usize,
    ) -> Result<(), GateError> {
        let depth = ids.len();
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids.iter().copied().skip(depth.saturating_sub(4)).collect();
        println!("fed ids={depth} first={head:?} last={tail:?} depth_sequence_from={prompt_len}");
        let t = Instant::now();
        let mut next = m.step(ids)?;
        println!(
            "step 0 {} {next} (the {depth} fed steps in {:.1} s, runtime value)",
            m.pos() - 1,
            t.elapsed().as_secs_f64()
        );
        let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(a.n_gen - 1);
        let mut tokens: Vec<u32> = Vec::with_capacity(a.n_gen);
        tokens.push(next);
        for _ in 1..a.n_gen {
            let t0 = Instant::now();
            next = m.step(&[next])?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            rows.push((m.pos() - 1, next, ms));
        }
        let warm = a.warm.unwrap_or(0);
        for (k, (pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            tokens.push(*tok);
            println!("step {i} {pos} {tok}");
            if a.timed {
                let tag = if i <= warm { " warm" } else { "" };
                println!("time step {i}{tag} ms={ms:.4}");
            }
        }
        println!("tokens {tokens:?}");
        if a.timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            println!(
                "SMOKE mode={} place={} prompt_tokens={prompt_len} depth={depth} generated={} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2}",
                mode_name(a.mode),
                a.place.name(),
                a.n_gen,
                counted.len(),
                1e3 / p50
            );
        }
        Ok(())
    }
}
