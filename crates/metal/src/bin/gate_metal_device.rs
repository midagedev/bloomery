//! The Metal device gate: the contract's clauses, run on this machine's
//! Metal device. The body is macOS only (`gate_metal_device/mac.rs`); on
//! another operating system the gate refuses by name and exits 78, the code
//! `tools/mac-gate.sh` uses for "no Metal here".

#[cfg(target_os = "macos")]
#[path = "gate_metal_device/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_device: refused: the Metal gate runs on macOS only");
    std::process::ExitCode::from(78)
}
