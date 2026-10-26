//! The Nemotron end-to-end gate on Metal: the session's tokens on the Nano
//! fixture against the fixture's reference set. No session exists yet, so the
//! gate refuses by name and exits 66, never 0: a gate that runs nothing is not
//! green.

fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_nemotron_e2e: refused: no session is registered yet");
    std::process::ExitCode::from(66)
}
