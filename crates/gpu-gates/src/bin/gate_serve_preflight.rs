//! `gate_serve_preflight` — the launcher's checks before the model fetch, held by the shipped server binaries.
//!
//! Each clause starts a seat binary (`bloomery-serve`, and the stand-alone `bloomery-serve-ds41` and
//! `bloomery-serve-qwen38`) in a fresh directory under an offline, empty cache and holds what it refuses (or
//! answers) to its class, its `error` event, its message and the clean state it leaves, before any model path runs.
//! The clauses are this binary's unit tests (`shared/serve_preflight_tests.rs`), so a call is
//! `cargo test --bin gate_serve_preflight`: it compiles this one bin and links no other, where a `--test` target
//! compiles and links every bin of the package and replaces the ones other lanes run.
//!
//! The seats are found one directory above the test executable's (`target/<profile>/<seat>`), built by the recipe
//! beforehand at the feature set each of them is gated at; `bloomery-serve` is a shipped bin, so it is the release
//! list's, with `vision`, and this bin requires it. The host clauses run in `gate-serve-draft-rules` and the `hw_`
//! clauses (the server's start opens the driver) in `gate-gpu-serve-start-glm`.

#[cfg(test)]
#[path = "shared/serve_preflight_tests.rs"]
mod preflight;

/// A run as a program names where its clauses are and ends 64: there is nothing to run but the tests.
fn main() -> std::process::ExitCode {
    eprintln!(
        "gate_serve_preflight: its clauses are this binary's tests (cargo test --bin gate_serve_preflight): \
         `just gate-serve-draft-rules` runs the host ones, `just gate-gpu-serve-start-glm` the hw_ ones"
    );
    std::process::ExitCode::from(64)
}
