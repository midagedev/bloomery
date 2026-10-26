//! The Nemotron kernels' gate: the Q5_0 and Q5_1 owners (the mat-vec modes, the
//! segmented mat-vec, the embedding and the GEMM staging's decode) against the
//! f64 oracle, on synthetic rows at the models' real shapes and on the Nano
//! fixture's real tensors. The body is macOS only
//! (`gate_metal_nemotron_kernels/mac.rs`); on another operating system the
//! gate refuses by name and exits 78, the code `tools/mac-gate.sh` uses for
//! "no Metal here".

#[cfg(target_os = "macos")]
#[path = "gate_metal_nemotron_kernels/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_nemotron_kernels: refused: the Metal gate runs on macOS only");
    std::process::ExitCode::from(78)
}
