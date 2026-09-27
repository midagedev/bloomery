//! The launch-cost bench's instrument, for the bins with a `--bench-kernels`
//! arm (`gate_p8`, `gate_gemm`): an arm's eager burst and its graph replay,
//! the `bench op=` row that prints the two, and the fixed fill pattern the
//! arms' weights and activations are made of. Each bin includes it with
//! `#[path]`; timing runs only under `tools/ref/time-gate.sh`'s lease, and
//! nothing here asserts on a result.

use bloomery_gpu::{GpuError, Graph};
use bloomery_gpu_gates::GateError;
use cuda_core::CudaStream;

/// Launches per burst, and nodes per captured graph.
pub const N: usize = 64;
/// Bursts (or graph replays) per arm — the spread of these is printed.
pub const ROUNDS: u32 = 7;
/// Graph launches per round, so the one synchronize is amortized.
pub const GREPS: usize = 4;

/// `enq` issued [`N`] times back to back with one synchronize, after one
/// warm-up burst: µs per launch as (min, mean, max) over [`ROUNDS`] bursts.
pub fn burst(
    stream: &CudaStream,
    enq: &mut dyn FnMut(&CudaStream) -> Result<(), GpuError>,
) -> Result<(f64, f64, f64), GateError> {
    for _ in 0..N {
        enq(stream)?;
    }
    stream.synchronize()?;
    let (mut lo, mut hi, mut sum) = (f64::INFINITY, 0.0f64, 0.0f64);
    for _ in 0..ROUNDS {
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            enq(stream)?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
        lo = lo.min(us);
        hi = hi.max(us);
        sum += us;
    }
    Ok((lo, sum / f64::from(ROUNDS), hi))
}

/// A graph of [`N`] launches replayed [`GREPS`] times per round, after two
/// warm-up replays: µs per node as (min, mean, max) over [`ROUNDS`] rounds.
pub fn replay(stream: &CudaStream, g: &Graph) -> Result<(f64, f64, f64), GateError> {
    for _ in 0..2 {
        g.launch(stream)?;
    }
    stream.synchronize()?;
    let (mut lo, mut hi, mut sum) = (f64::INFINITY, 0.0f64, 0.0f64);
    for _ in 0..ROUNDS {
        let t0 = std::time::Instant::now();
        for _ in 0..GREPS {
            g.launch(stream)?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / (N * GREPS) as f64;
        lo = lo.min(us);
        hi = hi.max(us);
        sum += us;
    }
    Ok((lo, sum / f64::from(ROUNDS), hi))
}

/// A fixed non-trivial u32 pattern — a multiplicative hash of the index,
/// which spreads bits through every byte and every f16 field a K-quant
/// super-block carries. Not a model of any weight distribution: its one job
/// is that no value-dependent decode path (`half_to_f32`'s zero arm above
/// all) is skipped by a buffer of zeros.
pub fn fill_pattern(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9)
        .collect()
}

/// The same pattern as f32 activations, mapped into roughly ±1 so a
/// quantization of it exercises rounding rather than saturation.
pub fn fill_pattern_f32(n: usize) -> Vec<f32> {
    fill_pattern(n)
        .into_iter()
        .map(|w| (w >> 8) as f32 / 8_388_608.0 - 1.0)
        .collect()
}

/// One `bench` row: the eager burst and the graph replay of the same launch,
/// each as min/mean/max over the arm's rounds, with the graph GB/s beside it.
pub fn print_arm(
    name: &str,
    rows: usize,
    bytes: u64,
    nodes: usize,
    eager: (f64, f64, f64),
    graph: (f64, f64, f64),
) {
    let gbps = match (bytes > 0, graph.0 > 0.0) {
        (true, true) => format!("{:.1}", bytes as f64 / graph.0 / 1e3),
        _ => "?".to_string(),
    };
    println!(
        "bench op={name} rows={rows} nodes={nodes} bytes={bytes} \
         eager_us_min={:.3} eager_us_mean={:.3} eager_us_max={:.3} \
         graph_us_min={:.3} graph_us_mean={:.3} graph_us_max={:.3} graph_gbps={gbps}",
        eager.0, eager.1, eager.2, graph.0, graph.1, graph.2
    );
}
