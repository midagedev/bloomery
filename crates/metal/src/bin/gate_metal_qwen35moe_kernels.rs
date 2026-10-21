//! The qwen35moe kernels' gate: the Q4_K and Q6_K mat-vecs, the norm on load,
//! the residual epilogue and the Q4_K embedding, run on this machine's Metal
//! device against the fixture's real tensors. The body is macOS only
//! (`gate_metal_qwen35moe_kernels/mac.rs`); on another operating system the
//! gate refuses by name and exits 78, the code `tools/mac-gate.sh` uses for
//! "no Metal here".

#[cfg(target_os = "macos")]
#[path = "gate_metal_qwen35moe_kernels/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_qwen35moe_kernels: refused: the Metal gate runs on macOS only");
    std::process::ExitCode::from(78)
}
