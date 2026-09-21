//! `generate` — the thin end-to-end decode CLI, and the round's A/B ruler.
//!
//! Greedy only, one token at a time, no prefill kernel, no streaming, no
//! chat template, no tokenizer: prompts are pre-tokenized in
//! `tools/ref/prompts.tsv` or given as ids on the command line. The whole
//! job is to drive `GpuModel::step` — every layer plus the head, one
//! position per call — and to print what came out.
//!
//!     generate [--prompt-id <id> | --tokens a,b,c] [-n N] [--ctx C]
//!              [--mode eager|graph] [--time]
//!
//! Defaults: prompt 0, N 32, ctx 512, mode graph.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh generate … --time`), never at a bare prompt. It
//! times each generated step host-side around `step(&[tok])` — the argmax
//! readback synchronizes inside, so the wall time is the whole step. The
//! prompt is fed untimed: it is P steps of the same body and would drag the
//! per-step distribution toward whatever the prompt length happens to be.
//!
//! Which number is the record: the fusion rounds this runner serves compare
//! GRAPH REPLAY µs, so the graph-mode footer is the one that becomes the
//! record and the eager footer prices the host submit path. Two costs sit
//! inside every timed step whichever mode it is — the four
//! `refresh_params` host-to-device copies, each of which synchronizes the
//! stream today, and the argmax readback.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate: built without the `gpu` feature; see `just generate`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::GpuModel;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::StepMode;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::open_model;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::prompts::read_prompts;

/// ik's own decode on this card and this model, so the footer shows the gap
/// and the depth it was read at.
#[cfg(feature = "gpu")]
const IK_REFERENCE: &str =
    "ik 216.6 tok/s at depth 0 (3090, 2026-09-21 morning, n=3; 204.6 at 1024, 189.7 at 4096)";

/// `tools/ref/prompts.tsv`, relative to this crate — the same file both
/// engines read their token ids from.
#[cfg(feature = "gpu")]
fn prompts_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv")
}

#[cfg(feature = "gpu")]
fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("generate: {name} needs a value"))
            .clone()
    })
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let timed = std::env::args().any(|a| a == "--time");
    let mode = match flag_value("--mode").as_deref() {
        None | Some("graph") => StepMode::Graph,
        Some("eager") => StepMode::Eager,
        Some(other) => {
            return Err(format!("generate: --mode is eager or graph, not {other}").into());
        }
    };
    let n_gen: usize = flag_value("-n").map_or(Ok(32), |s| s.parse())?;
    let ctx: usize = flag_value("--ctx").map_or(Ok(512), |s| s.parse())?;

    let tokens: Vec<u32> = match flag_value("--tokens") {
        Some(list) => list
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            let id: usize = flag_value("--prompt-id").map_or(Ok(0), |s| s.parse())?;
            let rows = read_prompts(&prompts_path())?;
            rows.iter()
                .find(|r| r.id == id)
                .ok_or_else(|| format!("generate: no prompt {id} in {}", prompts_path().display()))?
                .tokens
                .clone()
        }
    };
    if tokens.is_empty() {
        return Err("generate: empty prompt".into());
    }
    if tokens.len() + n_gen > ctx {
        return Err(format!(
            "generate: prompt {} + {n_gen} generated tokens exceed --ctx {ctx}",
            tokens.len()
        )
        .into());
    }

    let gguf = open_model()?;
    let mut model = GpuModel::load_full(&gguf, ctx)?;
    model.set_mode(mode);
    println!(
        "load resident_bytes={} ctx={ctx} layers={} mode={}",
        model.resident_bytes(),
        model.stages()[0].layers().len(),
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        }
    );
    if mode == StepMode::Graph {
        // Capture before the prompt so the node count prints once and the
        // first timed step is a replay, not a capture.
        println!("capture graph_nodes={}", model.capture_step()?);
    }

    // The prompt, one token per position; the argmax of its last token is
    // generated token 0, so it costs a `step` of P tokens and is not timed.
    // The N - 1 feedback steps after it are the per-step distribution.
    let mut next = model.step(&tokens)?;
    println!("step 0 {} {next}", model.pos() - 1);
    let mut step_ms: Vec<f64> = Vec::with_capacity(n_gen.saturating_sub(1));
    for i in 1..n_gen {
        let t0 = std::time::Instant::now();
        next = model.step(&[next])?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        step_ms.push(ms);
        println!("step {i} {} {next}", model.pos() - 1);
        if timed {
            println!("time step {i} ms={ms:.4}");
        }
    }

    if timed {
        if step_ms.is_empty() {
            return Err("generate: --time with -n 1 has no generated step to time".into());
        }
        let mut sorted = step_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let p50 = sorted[sorted.len() / 2];
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        // `steps` is the number of TIMED steps: generated token 0 comes out
        // of the prompt's own `step` call and is not one of them, so N
        // generated tokens give N - 1 timed feedback steps.
        println!(
            "SMOKE mode={} prompt_tokens={} generated={n_gen} steps={} p50_ms={p50:.4} \
             mean_ms={mean:.4} tok/s(p50)={:.2}",
            if mode == StepMode::Graph {
                "graph"
            } else {
                "eager"
            },
            tokens.len(),
            step_ms.len(),
            1e3 / p50
        );
        println!("reference {IK_REFERENCE}");
    }
    Ok(())
}
