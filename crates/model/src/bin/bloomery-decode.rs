//! Greedy decode on the reference CPU path, and the tok/s this engine has.
//!
//! Two qualifiers left, both structural:
//!
//!   1. **`ops::matmul_q` still materializes f32.** Its output rows run on the
//!      thread pool, but each row is dequantized to f32 and dotted in scalar f32;
//!      the remaining factor is the fused AVX2 int8 kernel (`crates/q3k-cpu` holds
//!      it; stage 1 does not call it), not more threads.
//!   2. **CPU only.** Not one byte of this runs on either card.
//!
//! Decode runs against a `KvCache`, one token per step. **Flat is the claim** —
//! the per-step column below is printed so that an unflat one is visible rather
//! than averaged away.
//!
//! Prefill and decode are timed apart. They are different work: the prefill reads
//! every weight once for `n` tokens, a decode step reads them again for one.
//! Averaging the two is how a prefill-heavy run reports a decode rate it does not
//! have.
//!
//! `--no-cache` keeps the prefix-re-prefill path so both can be measured inside
//! one lease. It exists for that comparison and nothing else; `tests/kv.rs` proves
//! the two agree to the last bit at every split.
//!
//! Run it through `tools/ref/decode-measure.sh`, which holds the box lease and
//! records witnesses — a tok/s taken while something else has the machine is not
//! a measurement.
use model::derived::Derived;
use model::forward::{argmax, forward, new_cache, step};
use std::time::Instant;

fn main() {
    let mut model = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".to_string();
    let mut tokens: Vec<u32> = Vec::new();
    let mut n_predict = 8usize;
    let mut use_cache = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" => model = args.next().expect("-m needs a path"),
            "-n" => n_predict = args.next().expect("-n needs a count").parse().unwrap(),
            "--no-cache" => use_cache = false,
            "--profile" => {
                // The profiler reads BLOOMERY_PROFILE exactly once (OnceLock in
                // `model::profile`), so this must land before any model code runs —
                // which it does: parsing happens before the first `step`. An existing
                // value wins, so `BLOOMERY_PROFILE=2 ... --profile` gets the stage
                // split rather than being clamped back to 1.
                if std::env::var("BLOOMERY_PROFILE")
                    .unwrap_or_default()
                    .is_empty()
                {
                    // SAFETY: no other threads exist yet — the process is single
                    // threaded until the model runs, so nothing can read the
                    // environment concurrently while this writes it.
                    unsafe { std::env::set_var("BLOOMERY_PROFILE", "1") };
                }
            }
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
                eprintln!(
                    "usage: bloomery-decode -m <gguf> --tokens <id,id,...> -n <count> \
                     [--no-cache] [--profile]"
                );
                std::process::exit(2);
            }
        }
    }
    if tokens.is_empty() {
        eprintln!("bloomery-decode: --tokens is required; this tool does not tokenize");
        std::process::exit(2);
    }

    let t_open = Instant::now();
    // This binary owns its main thread, so it takes the dispatcher's cpu slot.
    // `BLOOMERY_PIN_MAIN=0` leaves it floating, for the A/B.
    let pin_main = std::env::var("BLOOMERY_PIN_MAIN").map_or(true, |v| v != "0");
    let pinned = pin_main && threads::pool().pin_caller();
    println!(
        "main   pinned={pinned} threads={}",
        threads::pool().threads()
    );
    let g = gguf::Gguf::open(&model).expect("model file");
    println!("model  {model}");
    println!("open   {:?} (mmap, no dequant)", t_open.elapsed());
    // The wk_b Q8_0 requant, once per model instead of once per step. Its own
    // line for the same reason `open` gets one: it is load-time work, and mixing
    // it into the prefill or decode timings below would misattribute it.
    let t_derived = Instant::now();
    let derived = Derived::new(&g).expect("derived weights");
    println!(
        "derived {:?} ({} blocks, {:.1} MB)",
        t_derived.elapsed(),
        derived.filled_blocks(),
        derived.size_bytes() as f64 / 1e6
    );
    println!("prompt {} tokens: {:?}", tokens.len(), tokens);
    println!(
        "path   {}",
        if use_cache {
            "KV cache, one token per step"
        } else {
            "no cache, the whole prefix re-prefilled every step"
        }
    );

    let mut ctx = tokens.clone();
    let mut out: Vec<u32> = Vec::new();
    let mut steps: Vec<f64> = Vec::new();
    let mut decode_total = std::time::Duration::ZERO;

    // The prefill: with the cache it is one call that leaves the prompt in it; without,
    // there is nothing to leave anywhere and the first "step" pays for it again.
    let mut cache = new_cache(&g).expect("cache");
    let mut next = if use_cache {
        let t0 = Instant::now();
        let logits = step(&g, &tokens, &mut cache, &derived).expect("prefill");
        let dt = t0.elapsed();
        println!(
            "\nprefill {} tokens in {:.1} ms = {:.2} tok/s",
            tokens.len(),
            dt.as_secs_f64() * 1e3,
            tokens.len() as f64 / dt.as_secs_f64()
        );
        // Prefill and decode are different work; the profile tables stay separate too.
        // The reset drops the prefill's accumulators so the decode table is decode only.
        if model::profile::enabled() {
            print!(
                "{}",
                model::profile::report(dt.as_nanos() as u64, "prefill")
            );
            model::profile::reset();
        }
        Some(argmax(&logits.data))
    } else {
        None
    };

    println!("\n{:>4} {:>6} {:>12} {:>9}", "step", "ctx", "ms", "tok/s");
    // Pool protocol counters, snapshotted around the decode loop so the
    // prefill's dispatches stay out of the per-step arithmetic. Whether the
    // workers park between dispatches (futex wake per call) or stay hot on
    // the spin budget is the first fork in attributing the step's
    // orchestration share.
    let pool0 = threads::pool().stats();
    for s in 0..n_predict {
        let ctx_at_start = ctx.len();
        let t0 = Instant::now();
        let token = match next {
            // Cached: the prefill already produced the first token, so this step feeds
            // only that one token and the cache supplies everything before it.
            Some(t) => {
                ctx.push(t);
                out.push(t);
                let logits = step(&g, &[t], &mut cache, &derived).expect("step");
                argmax(&logits.data)
            }
            None => {
                let logits = forward(&g, &ctx).expect("forward");
                let t = argmax(&logits.data);
                ctx.push(t);
                out.push(t);
                t
            }
        };
        let dt = t0.elapsed();
        decode_total += dt;
        steps.push(dt.as_secs_f64() * 1e3);
        println!(
            "{:>4} {:>6} {:>12.1} {:>9.4}",
            s,
            ctx_at_start,
            dt.as_secs_f64() * 1e3,
            1.0 / dt.as_secs_f64()
        );
        if use_cache {
            next = Some(token);
        }
    }

    println!("\ngenerated {out:?}");
    println!(
        "{} decode steps in {:.2} s = {:.4} tok/s (mean {:.1} ms/token)",
        n_predict,
        decode_total.as_secs_f64(),
        n_predict as f64 / decode_total.as_secs_f64(),
        decode_total.as_secs_f64() * 1e3 / n_predict as f64
    );
    // Flatness is the property, so it is reported as a number rather than left to the
    // reader's eye: with a cache the spread is measurement noise, without it the spread
    // IS the missing cache.
    let (lo, hi) = steps
        .iter()
        .fold((f64::MAX, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    println!(
        "per step {:.1}–{:.1} ms across ctx {}–{} — spread {:.1}%",
        lo,
        hi,
        tokens.len(),
        ctx.len() - 1,
        (hi - lo) / lo * 100.0
    );
    if model::profile::enabled() {
        print!(
            "{}",
            model::profile::report(decode_total.as_nanos() as u64, "decode")
        );
        let p1 = threads::pool().stats();
        println!(
            "pool over {n_predict} decode steps: {} dispatches, dispatcher parked {}x, workers parked {}x",
            p1.dispatches - pool0.dispatches,
            p1.dispatcher_parks - pool0.dispatcher_parks,
            p1.worker_parks - pool0.worker_parks,
        );
    }
}
