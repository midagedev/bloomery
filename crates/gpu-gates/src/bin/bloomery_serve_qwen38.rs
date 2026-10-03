//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine: [`qwen38`]'s seat, whose module doc
//! holds the contract (the flags, the records, the keep rule and the exit
//! codes). `bloomery-serve --model qwen38` serves the same seat.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "bloomery-serve-qwen38: built without the `deepseek41` feature; see `just gate-gpu-qwen38-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match qwen38::run(&args) {
        // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
        Ok(serve::ServeError::Engine(f)) => {
            eprintln!("bloomery-serve-qwen38: {f}");
            std::process::ExitCode::from(70)
        }
        Ok(e) => bloomery_gpu_gates::exit_with("bloomery-serve-qwen38", Err(e.into())),
        Err(e) => bloomery_gpu_gates::exit_with("bloomery-serve-qwen38", Err(e)),
    }
}

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen38.rs"]
mod qwen38;

// The seat's MTP draft driver, a sibling of the seat as in `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/drafted.rs"]
mod drafted;

// The seats' shared `--ctx` default rule, a sibling of the seat as in
// `bloomery-serve`.
#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/ctx.rs"]
mod ctx;
