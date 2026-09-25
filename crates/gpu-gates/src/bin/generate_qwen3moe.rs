//! `generate_qwen3moe` — the decode CLI of the qwen3moe engine: the whole
//! model on one card, greedy, one token per step, and the timing runner's
//! ruler.
//!
//!     generate_qwen3moe (--prompt <text> | --tokens a,b,c | --seed-depth D)
//!                       [-n N] [--ctx C] [--mode eager|graph] [--prefill auto|pass|gemm]
//!                       [--time [--warm W]]
//!
//! Defaults: N 32, C 4096, mode graph, prefill auto, W 0. A flag given twice
//! takes its last value. `--prompt` tokenizes the text with the file's own vocabulary
//! (`tokenizer`, no BOS: the file sets `add_bos_token` false; no
//! chat template) and prints the generated text after the ids.
//!
//! The prompt is prefilled (`Qwen3moeModel::prefill_with` by the `--prefill`
//! path: `auto` takes one pass for a prompt of up to eight ids and GEMM
//! ubatches of up to 512 for a longer one, a tail of up to eight after them
//! one pass; `pass` passes of up to eight positions, the same cache rows and
//! answer as one step per token, in graph mode each a replay of the pass of
//! its size; `gemm` ubatches only); the argmax after its last token is
//! generated token 0, and `N − 1` feedback steps follow.
//! Lines: `prompt_ids`, `load` (with the flash pass: `flash_mma=`), in graph
//! mode `capture graph_nodes=` and `capture prefill_graphs=<n> nodes=<m=1>,…
//! ms= vram_bytes=` (every pass size captured before the prompt: its wall
//! and the card's free bytes it took, runtime values), `step 0 pos tok`
//! (with the units the prompt took: `prefill_steps=`, and their shapes:
//! `plan=ubatch:512x2 pass:1`, a run of equal sizes as `<size>x<k>`), then,
//! all written after the loop,
//! `time prompt n=<P> ms= tok/s= passes=<K> kind=<gemm|prefill>` (`K` the
//! ubatches and passes; `gemm` when a ubatch ran)
//! (the wall of `prefill` through its token's readback, on every run: a
//! runtime value like the `load` line, a measurement only under the lease;
//! the prefill arena is allocated at load and the passes captured before
//! it, so that wall carries neither),
//! per feedback step `step i pos tok` (and `time step i ms=` under
//! `--time`); then
//! `tokens [..]`, `text` for a `--prompt` run, and under `--time` the
//! `SMOKE` footer with `generate`'s keys (`p50_ms=`, `mean_ms=`, `warm=`,
//! `tok/s(p50)=`).
//!
//! `--seed-depth D` stands the model at depth D: `seed_depth(D − 1)` fills
//! the caches with a pattern, and one literal token (id 0) is prefilled
//! after them, so the timed steps run at the positions a D-token prompt
//! leaves. The tokens it prints are meaningless; only the timing is. Its
//! `time prompt` row is that one token's prefill, `n=1`.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh`), never at a bare prompt. The cache's
//! height is `--ctx`: the flash's segment grid is fixed by it, so a timed
//! run names its ctx.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate_qwen3moe: built without the `gpu` feature; see `just gen-qwen3moe`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_qwen3moe", cli::run())
}

#[cfg(feature = "gpu")]
mod cli {
    use bloomery_gpu::Qwen3moeModel;
    use bloomery_gpu::arch::qwen3moe::{PrefillPath, PrefillPlan};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::{GateError, open_split, ref_model_path};
    use model::arch::Arch;
    use std::time::Instant;
    use tokenizer::Tokenizer;

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

    pub fn run() -> Result<(), GateError> {
        let timed = std::env::args().any(|a| a == "--time");
        let n_gen: usize = flag("-n")?.map_or(Ok(32), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(4096), |s| s.parse())?;
        let warm: usize = flag("--warm")?.map_or(Ok(0), |s| s.parse())?;
        let mode = match flag("--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => return Err(format!("--mode is eager or graph, not {o}").into()),
        };
        let path = match flag("--prefill")?.as_deref() {
            None | Some("auto") => PrefillPath::Auto,
            Some("pass") => PrefillPath::Pass,
            Some("gemm") => PrefillPath::Gemm,
            Some(o) => return Err(format!("--prefill is auto, pass or gemm, not {o}").into()),
        };
        let seed_depth: Option<usize> = flag("--seed-depth")?.map(|s| s.parse()).transpose()?;
        let text = flag("--prompt")?;
        let tokens = flag("--tokens")?;
        let sources = usize::from(text.is_some())
            + usize::from(tokens.is_some())
            + usize::from(seed_depth.is_some());
        if sources != 1 {
            return Err("give exactly one of --prompt, --tokens, --seed-depth".into());
        }
        if n_gen == 0 || (timed && n_gen <= warm + 1) {
            return Err(format!("-n {n_gen} leaves no counted step (warm {warm})").into());
        }
        let tok = match &text {
            Some(_) => Some(Tokenizer::from_gguf(ref_model_path()?)?),
            None => None,
        };
        let ids: Vec<u32> = match (&text, &tokens, seed_depth) {
            (Some(t), _, _) => tok.as_ref().ok_or("no tokenizer")?.encode(t, true, false),
            (_, Some(s), _) => s
                .split(',')
                .map(|v| v.trim().parse::<u32>())
                .collect::<Result<_, _>>()?,
            (_, _, Some(_)) => vec![0],
            _ => unreachable!("one source by the check above"),
        };
        if ids.is_empty() {
            return Err("the prompt has no ids".into());
        }
        let depth = seed_depth.unwrap_or(ids.len());
        if depth + n_gen > ctx {
            return Err(format!("depth {depth} + {n_gen} tokens pass --ctx {ctx}").into());
        }
        println!("prompt_ids {ids:?}");
        let t = Instant::now();
        let file = open_split(Arch::Qwen3moe, "gen-qwen3moe")?;
        let mut m = Qwen3moeModel::load_full(file, ctx)?;
        m.set_mode(mode);
        println!(
            "load resident_bytes={} ctx={ctx} layers={} mode={} flash_mma={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.stages()[0].layers().len(),
            if mode == StepMode::Graph {
                "graph"
            } else {
                "eager"
            },
            m.body("generate_qwen3moe")?.flash_mma(),
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            println!("capture graph_nodes={}", m.capture_step()?);
            let gpu = m.stages()[0].gpu();
            let (free0, _) = gpu.mem_info()?;
            let t = Instant::now();
            let nodes = m.capture_prefill()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let (free1, _) = m.stages()[0].gpu().mem_info()?;
            let list: Vec<String> = nodes.iter().map(usize::to_string).collect();
            println!(
                "capture prefill_graphs={} nodes={} ms={ms:.1} vram_bytes={} (runtime values)",
                nodes.len(),
                list.join(","),
                free0.saturating_sub(free1)
            );
        }
        if let Some(d) = seed_depth.filter(|&d| d > 1) {
            m.seed_depth(d - 1)?;
            println!("seed rows={} pos={}", d - 1, m.pos());
        }
        let plan = PrefillPlan::new(ids.len(), path);
        let t = Instant::now();
        let mut next = m.prefill_with(&ids, path)?;
        let prefill_wall = t.elapsed();
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={plan}, {:.2} s, \
             runtime value)",
            m.pos() - 1,
            ids.len(),
            plan.steps.len(),
            prefill_wall.as_secs_f64()
        );
        let mut tokens_out = Vec::with_capacity(n_gen);
        tokens_out.push(next);
        // Every line of the loop is held and written after it: a write is a
        // syscall, and the steps it would separate are the measurement.
        let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(n_gen - 1);
        for _ in 1..n_gen {
            let t0 = Instant::now();
            next = m.step(&[next])?;
            rows.push((m.pos() - 1, next, t0.elapsed().as_secs_f64() * 1e3));
        }
        let prefill_ms = prefill_wall.as_secs_f64() * 1e3;
        println!(
            "time prompt n={} ms={prefill_ms:.4} tok/s={:.2} passes={} kind={}",
            ids.len(),
            ids.len() as f64 * 1e3 / prefill_ms,
            plan.steps.len(),
            plan.kind()
        );
        for (k, &(pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            tokens_out.push(tok);
            println!("step {i} {pos} {tok}");
            if timed {
                let tag = if i <= warm { " warm" } else { "" };
                println!("time step {i}{tag} ms={ms:.4}");
            }
        }
        println!("tokens {tokens_out:?}");
        if let Some(t) = &tok {
            println!("text {:?}", t.decode(&tokens_out));
        }
        if timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={depth} seeded={} generated={n_gen} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2} ctx={ctx}",
                if mode == StepMode::Graph {
                    "graph"
                } else {
                    "eager"
                },
                ids.len(),
                seed_depth.is_some(),
                counted.len(),
                1e3 / p50
            );
        }
        Ok(())
    }
}
