//! The row kernels' timing probe: the gated delta step over a chunk of rows
//! and the prefill attention, launched through `Kernels` at the session's
//! shapes, read as microseconds per row of a layer. It is a development
//! probe, not a gate clause: its numbers are this machine's, on no timing
//! runner. The body is macOS only (`metal_rows_time/mac.rs`); elsewhere it
//! refuses and exits 78.

#[cfg(target_os = "macos")]
#[path = "metal_rows_time/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("metal_rows_time: refused: the Metal probe runs on macOS only");
    std::process::ExitCode::from(78)
}
