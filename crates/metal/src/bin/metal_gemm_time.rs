//! The dense GEMM timing probe: the crate's Q4_K, Q6_K and A4 GEMMs,
//! launched through `Kernels::gemm`, at the batched prefill's shapes — the
//! term the design's cost model leaves open, the GEMM's effective TFLOPS F.
//! It is a development probe, not a gate clause: its numbers are this
//! machine's, on no timing runner. The body is macOS only
//! (`metal_gemm_time/mac.rs`); elsewhere it refuses and exits 78.

#[cfg(target_os = "macos")]
#[path = "metal_gemm_time/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("metal_gemm_time: refused: the Metal probe runs on macOS only");
    std::process::ExitCode::from(78)
}
