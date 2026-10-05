//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1
//! engine: [`ds41`]'s seat, whose module doc holds the contract (the flags,
//! the records, the prompt cache and the exit codes). `bloomery-serve --model
//! ds41` serves the same seat.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "bloomery-serve-ds41: built without the `deepseek41` feature; see `just weekly-gpu-ds41-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match ds41::run(&args) {
        // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
        Ok(serve::ServeError::Engine(f)) => {
            eprintln!("bloomery-serve-ds41: {f}");
            std::process::ExitCode::from(70)
        }
        Ok(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e.into())),
        Err(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e)),
    }
}

#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/ds41.rs"]
mod ds41;

// The seats' one owner of a round run as passes, the seat naming it beside
// itself (`super::rounds`) as it does under `serve_seats` in the one-binary
// server.
#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/rounds.rs"]
#[allow(
    dead_code,
    reason = "the seat steps its rows plain; the drafted half serves the drafted seats"
)]
mod rounds;

// The per-slot draft table `rounds` names for its drafted half
// (`super::drafted`).
#[cfg(feature = "deepseek41")]
#[path = "shared/serve_seats/drafted.rs"]
#[allow(
    dead_code,
    reason = "only rounds' drafted half names it, which this seat does not run"
)]
mod drafted;

// The shared files the seat names at the crate root (`crate::dspark` and
// kin), as `generate_ds41` includes them.
#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;
