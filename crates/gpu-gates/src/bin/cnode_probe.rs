//! The per-node price of a captured graph, `c_node` of the step cost model in
//! `docs/plan.md`: graphs of N empty `touch` kernels (`Probe::enqueue_touch`,
//! one 32-thread block each), one node per launch, at the node counts of
//! [`NODES`].
//!
//! `--check`: per N, the graph captured, its node count asserted equal to N,
//! the buffer the kernels write zeroed, the graph replayed once and the buffer
//! read back — all of its values must be 1.0, or the replay did not run. Red on
//! either.
//!
//! `--time` (lead-only, under `tools/ref/time-gate.sh`, which owns the lease,
//! the witness blocks and the card pin): the check first, refusing to time if
//! it fails; then per N two warm replays and [`ROUNDS`] rounds of [`REPS`]
//! replays, one synchronize a round, and one `cnode` line with the fastest and
//! the mean round's µs per replay and the fastest round's µs per node.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("cnode_probe: built without the `gpu` feature; see `just time-gpu-cnode`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("cnode_probe", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use bloomery_gpu::probe::Probe;
    use bloomery_gpu::{Gpu, Graph};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};
    use std::time::Instant;

    /// The node counts of the V4.1 decode token's gemv graph that the cost
    /// model's `c_node` row names: 784 with `attn_output_a` as eight grouped
    /// launches per block, 504 with it as one launch.
    const NODES: [usize; 2] = [784, 504];
    /// Timed rounds per graph; the fastest and the mean are printed.
    const ROUNDS: usize = 7;
    /// Replays per round, one synchronize at its end.
    const REPS: usize = 2;
    /// Values one `touch` launch writes: one 32-thread block.
    const TOUCHED: usize = 32;

    enum Mode {
        Check,
        Time,
    }

    fn mode() -> Result<Mode, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["--check"] => Ok(Mode::Check),
            ["--time"] => Ok(Mode::Time),
            _ => Err(
                format!("cnode_probe: want exactly one of --check, --time; got {args:?}").into(),
            ),
        }
    }

    /// Microseconds per replay of `g`: two warm replays, then [`ROUNDS`]
    /// rounds of [`REPS`] replays, one synchronize a round; the fastest round
    /// and the mean.
    fn replay_us(stream: &CudaStream, g: &Graph) -> Result<(f64, f64), GateError> {
        for _ in 0..2 {
            g.launch(stream)?;
        }
        stream.synchronize()?;
        let mut us = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..REPS {
                g.launch(stream)?;
            }
            stream.synchronize()?;
            us.push(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
        }
        let min = us.iter().copied().fold(f64::INFINITY, f64::min);
        Ok((min, us.iter().sum::<f64>() / ROUNDS as f64))
    }

    pub fn run() -> Result<(), GateError> {
        let mode = mode()?;
        let gpu = Gpu::new()?;
        let touch = Probe::load(gpu.context())?;
        let stream = gpu.stream();
        let mut y = DeviceBuffer::<f32>::zeroed(stream, TOUCHED)?;
        let mut graphs = Vec::with_capacity(NODES.len());
        let mut ok = true;
        for n in NODES {
            let g = gpu.capture(|s| (0..n).try_for_each(|_| touch.enqueue_touch(s, &mut y)))?;
            let nodes = g.node_count();
            y.copy_from_host(stream, &[0.0; TOUCHED])?;
            g.launch(stream)?;
            let written = y.to_host_vec(stream)?.iter().filter(|&&v| v == 1.0).count();
            let pass = nodes == n && written == TOUCHED;
            println!(
                "cnode check touch_nodes={n} graph_nodes={nodes} replay_wrote={written}/{TOUCHED} {}",
                verdict(pass)
            );
            ok &= pass;
            graphs.push((n, g));
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: cnode_probe check — every touch graph holds its node count and one replay \
             wrote every value"
        );
        if let Mode::Check = mode {
            return Ok(());
        }
        for (n, g) in &graphs {
            let (min, mean) = replay_us(stream, g)?;
            println!(
                "cnode touch_nodes={n} touch_us_min={min:.1} touch_us_mean={mean:.1} \
                 touch_us_per_node={:.3}",
                min / *n as f64
            );
        }
        Ok(())
    }
}
