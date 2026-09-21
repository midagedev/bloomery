//! GPU kernel gate for package P1 (docs/gpu-design.md 작업 꾸러미). The
//! track that owns P1 fills this in; the reference side is
//! `bloomery_gpu_gates` (dequantized f64 dot), the band is `KERNEL_BAND`.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p1: built without the `gpu` feature; see `just gate-gpu-p1`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() {
    eprintln!("gate_p1: not written yet");
    std::process::exit(1);
}
