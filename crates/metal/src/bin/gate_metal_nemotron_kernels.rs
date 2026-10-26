//! The Nemotron kernels' gate on Metal: the Mamba-2 step, the Q5_0 and Q5_1
//! owners, the squared ReLU and the sigmoid router against the f64 oracle. No
//! clause is registered yet, so the gate refuses by name and exits 66, never 0:
//! a gate that runs nothing is not green.

fn main() -> std::process::ExitCode {
    eprintln!("gate_metal_nemotron_kernels: refused: no clause is registered yet");
    std::process::ExitCode::from(66)
}
