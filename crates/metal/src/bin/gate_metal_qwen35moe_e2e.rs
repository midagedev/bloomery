//! The qwen35moe end-to-end gate on Metal: the session's tokens on the
//! fixture against ik's reference sets, the token's encoded count, the fault
//! path of a bad token id, and the session as the generation loop's target.
//! The body is macOS only (`gate_metal_qwen35moe_e2e/mac.rs`); on another
//! operating system the gate refuses by name and exits 78, the code
//! `tools/mac-gate.sh` uses for "no Metal here".

#[cfg(target_os = "macos")]
#[path = "gate_metal_qwen35moe_e2e/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_qwen35moe_e2e: refused: the Metal gate runs on macOS only");
    std::process::ExitCode::from(78)
}
