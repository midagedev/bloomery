//! The decode-attention timing probe: the flash and the merge, launched
//! through `Kernels` at the session's shapes (the f16 cache, head 256, 16
//! query heads over 2 key heads), keys in {100, 512, 1024, 4096}, read as
//! microseconds per dispatch and K/V gigabytes per second beside the profile
//! the lead took. Phase variants split the flash's time: the same shipped
//! kernels with one phase gated off, compiled from the shipped source beside
//! it, the way the kernel gate compiles its mutants. It is a development
//! probe, not a gate clause: its numbers are this machine's, on no timing
//! runner. The body is macOS only (`metal_attention_time/mac.rs`); elsewhere
//! it refuses and exits 78.

#[cfg(target_os = "macos")]
#[path = "metal_attention_time/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("metal_attention_time: refused: the Metal probe runs on macOS only");
    std::process::ExitCode::from(78)
}
