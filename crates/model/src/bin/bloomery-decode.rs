//! Greedy decode on the reference CPU path, and the first tok/s this engine has.
//!
//! Read the number with its three qualifiers, all of them structural and all of them
//! round 1-5's to remove:
//!
//!   1. **No KV cache.** `attn::block_attn` has prefill semantics, so step `i` re-runs
//!      the whole prefix. Per-step cost therefore *grows* with the output, and the
//!      steady-state tok/s a server would report is not what this measures. The
//!      per-step table below is printed for exactly that reason.
//!   2. **`ops::matmul_q` is the reference implementation** — one thread, dequantizing
//!      a weight row at a time into a scalar dot. `crates/q3k-cpu` holds the fast path
//!      and stage 1 does not call it.
//!   3. **CPU only.** Not one byte of this runs on either card.
//!
//! So this is a floor, not a result: the number to beat is our own, and the first thing
//! that will beat it is the cache.
//!
//! Run it through `tools/ref/decode-measure.sh`, which holds the box lease and records
//! witnesses — a tok/s taken while something else has the machine is not a measurement.
use model::forward::{argmax, forward};
use std::time::Instant;

fn main() {
    let mut model = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".to_string();
    let mut tokens: Vec<u32> = Vec::new();
    let mut n_predict = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" => model = args.next().expect("-m needs a path"),
            "-n" => n_predict = args.next().expect("-n needs a count").parse().unwrap(),
            "--tokens" => {
                tokens = args
                    .next()
                    .expect("--tokens needs a comma-separated list")
                    .split(',')
                    .map(|s| s.trim().parse().expect("token ids are integers"))
                    .collect();
            }
            other => {
                eprintln!("bloomery-decode: unknown argument {other}");
                eprintln!("usage: bloomery-decode -m <gguf> --tokens <id,id,...> -n <count>");
                std::process::exit(2);
            }
        }
    }
    if tokens.is_empty() {
        eprintln!("bloomery-decode: --tokens is required; this tool does not tokenize");
        std::process::exit(2);
    }

    let t_open = Instant::now();
    let g = gguf::Gguf::open(&model).expect("model file");
    println!("model  {model}");
    println!("open   {:?} (mmap, no dequant)", t_open.elapsed());
    println!("prompt {} tokens: {:?}", tokens.len(), tokens);
    println!("\n{:>4} {:>6} {:>12} {:>9}", "step", "ctx", "ms", "tok/s");

    let mut ctx = tokens.clone();
    let mut out = Vec::new();
    let mut total = std::time::Duration::ZERO;
    let mut steps = Vec::new();
    for step in 0..n_predict {
        let t0 = Instant::now();
        let logits = forward(&g, &ctx).expect("forward");
        let dt = t0.elapsed();
        total += dt;
        let next = argmax(&logits.data);
        // The context length at the START of the step is what the step paid for.
        println!(
            "{:>4} {:>6} {:>12.1} {:>9.4}",
            step,
            ctx.len(),
            dt.as_secs_f64() * 1e3,
            1.0 / dt.as_secs_f64()
        );
        steps.push(dt.as_secs_f64() * 1e3);
        ctx.push(next);
        out.push(next);
    }

    println!("\ngenerated {out:?}");
    println!(
        "{} tokens in {:.2} s = {:.4} tok/s (mean {:.1} ms/token)",
        n_predict,
        total.as_secs_f64(),
        n_predict as f64 / total.as_secs_f64(),
        total.as_secs_f64() * 1e3 / n_predict as f64
    );
    // Without a cache the last step is the honest steady-state cost at this length, and
    // the first is the cheapest. Printing both spreads is what stops the mean from
    // reading as a decode rate it is not.
    println!(
        "first step {:.1} ms at ctx {}, last {:.1} ms at ctx {} — that growth is the missing cache",
        steps[0],
        tokens.len(),
        steps[steps.len() - 1],
        ctx.len() - 1
    );
}
