//! The mat-vec timing probe's body (macOS only).
//!
//! The method is the probe's (`tools/metal-probe/probe.swift`, `gemvRun`): a
//! matrix is copied into a working set of at least 1 GiB so that a run of
//! dispatches never reads one place twice inside the cache's reach; a command
//! buffer holds enough dispatches to read 768 MiB; two buffers warm the clocks
//! and `--trials` (16) are timed, all committed back to back; a dispatch's time
//! is its buffer's time over the dispatches in it.
//!
//! What differs is the clock. The crate's recording does not give out its
//! command buffer, so a buffer's end is the moment the stream's event recorded
//! after it fires, read by a host that spins on it; a buffer's time is the gap
//! between two such moments (the first timed buffer starts where the last warm
//! one ends). Every buffer is encoded before the first is committed, so the
//! host is never what the device waits for. The signal-only buffer between two
//! timed ones is the stream's own and is in every gap. The figure is the median
//! of the per-buffer rates (see [`Timing`]); `--gaps` prints every gap.
//!
//! Flags: `--only q4k|q6k`, `--trials N`, `--gaps`, `--no-check` (skip the
//! f64 oracle's check of the first dispatch against the band).

use std::process::ExitCode;
use std::time::Instant;

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems, Event, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{BLOCK_VALUES, Kernels, MatvecParams, Mode, Quant};

#[path = "../shared/oracle.rs"]
#[allow(
    dead_code,
    reason = "the probe reads the oracle's reference and band; the gate reads the rest"
)]
mod oracle;

/// The probe's working set, window, warm-up and trial counts.
const WORKING_SET_BYTES: usize = 1 << 30;
const WINDOW_BYTES: usize = 768 << 20;
const WARMUPS: usize = 2;

/// The probe's shapes: (rows, columns) of an expert's down, a layer's
/// projections and the large ones.
const SHAPES: [(usize, usize); 5] = [
    (2048, 2048),
    (4096, 4096),
    (512, 2048),
    (1536, 4096),
    (8192, 4096),
];

/// One matrix of random blocks the probe's way: random bytes, with `d` (and
/// `dmin`) a finite f16 in [0.0005, 0.004].
fn matrix(quant: Quant, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let block = quant.block_bytes();
    let blocks = rows * cols / BLOCK_VALUES;
    let mut rng = oracle::Rng::new(seed);
    let mut bytes = vec![0u8; blocks * block];
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&word[..n]);
    }
    let mut small = || {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::f32_to_f16_bits(0.0005 + 0.0035 * unit).to_le_bytes()
    };
    for b in bytes.chunks_exact_mut(block) {
        match quant {
            Quant::Q4K => {
                b[0..2].copy_from_slice(&small());
                b[2..4].copy_from_slice(&small());
            }
            Quant::Q6K => b[208..210].copy_from_slice(&small()),
        }
    }
    bytes
}

/// One shape's timing: bytes per second over a dispatch, and every timed
/// buffer's gap. The median of the per-buffer rates is the figure: the host
/// that spins on the events wakes late at times, and a late wake stamps two
/// neighbouring buffers at one moment (a gap near zero beside a gap of two
/// buffers), which the median ignores and a mean over the span would not.
struct Timing {
    dispatches: usize,
    copies: usize,
    working_set: usize,
    median: f64,
    /// The gaps within a quarter of the median gap: the buffers read cleanly.
    clean: usize,
    /// Every timed buffer's gap, in seconds.
    gaps: Vec<f64>,
    /// The host's time to encode one dispatch, in seconds.
    encode: f64,
}

fn time_shape(
    device: &MetalDevice,
    kernels: &Kernels,
    quant: Quant,
    rows: usize,
    cols: usize,
    trials: usize,
    check: bool,
) -> Result<Timing, String> {
    let stream = device.stream();
    let one = matrix(
        quant,
        rows,
        cols,
        0x6B70_726F_6265 ^ ((rows as u64) << 24) ^ ((cols as u64) << 4),
    );
    let bytes = one.len();
    let stride = bytes.div_ceil(256) * 256;
    let copies = WORKING_SET_BYTES.div_ceil(stride).max(1);
    let mut all = vec![0u8; copies * stride];
    for c in 0..copies {
        all[c * stride..c * stride + bytes].copy_from_slice(&one);
    }
    let weights = device.upload(stream, &all).map_err(why)?;
    drop(all);
    let x_values = oracle::activations(cols, 0x78, 1.0);
    let x = device.upload(stream, &x_values).map_err(why)?;

    let dispatches = WINDOW_BYTES.div_ceil(bytes).max(30);
    let buffers = WARMUPS + trials;
    let mut y = device
        .zeroed::<f32>(stream, Elems(buffers * dispatches * rows))
        .map_err(why)?;

    // Encode every buffer, then commit them back to back, an event after each.
    let encode_start = Instant::now();
    let mut recordings = Vec::with_capacity(buffers);
    for b in 0..buffers {
        let mut recording = stream.begin().map_err(why)?;
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            for j in 0..dispatches {
                let n = b * dispatches + j;
                kernels
                    .matvec(
                        &mut encoder,
                        MatvecParams {
                            quant,
                            rows,
                            cols,
                            weights: weights
                                .window(Elems((n % copies) * stride), Elems(bytes))
                                .map_err(why)?,
                            x: x.window(Elems(0), Elems(cols)).map_err(why)?,
                            y: y.window_mut(Elems(n * rows), Elems(rows)).map_err(why)?,
                            mode: Mode::Store,
                        },
                    )
                    .map_err(why)?;
            }
            encoder.end();
        }
        recordings.push(recording);
    }
    let encode = encode_start.elapsed().as_secs_f64() / (buffers * dispatches) as f64;
    let mut events = Vec::with_capacity(buffers);
    for recording in recordings {
        stream.submit(recording).map_err(why)?;
        events.push(stream.record().map_err(why)?);
    }
    let mut ends = Vec::with_capacity(buffers);
    for event in &events {
        while !event.done().map_err(why)? {
            std::hint::spin_loop();
        }
        ends.push(Instant::now());
    }
    stream.sync().map_err(why)?;

    let mut out = vec![0f32; buffers * dispatches * rows];
    device
        .read(stream, y.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    let first = &out[..rows];
    for (n, slice) in out.chunks_exact(rows).enumerate() {
        if slice
            .iter()
            .zip(first)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(format!(
                "dispatch {n} read the same weights and gave other bits than dispatch 0"
            ));
        }
    }
    if check {
        let x64: Vec<f64> = x_values.iter().map(|&v| f64::from(v)).collect();
        let reference = oracle::reference(quant, &one, rows, cols, &x64)?;
        let band = oracle::band(quant, cols / BLOCK_VALUES, oracle::Kind::Store);
        oracle::check(first, &reference, band, oracle::Kind::Store, None)?;
    }

    let gaps: Vec<f64> = (WARMUPS..buffers)
        .map(|b| ends[b].duration_since(ends[b - 1]).as_secs_f64())
        .collect();
    let per_buffer: Vec<f64> = gaps
        .iter()
        .map(|&g| bytes as f64 * dispatches as f64 / g / 1e9)
        .collect();
    let median_gbps = median(&per_buffer);
    let median_gap = median(&gaps);
    let clean = gaps
        .iter()
        .filter(|&&g| (g - median_gap).abs() <= 0.25 * median_gap)
        .count();
    Ok(Timing {
        dispatches,
        copies,
        working_set: copies * stride,
        median: median_gbps,
        clean,
        gaps,
        encode,
    })
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The most timed buffers: every buffer, warm-ups included, is encoded before
/// any is committed, and each takes a signal buffer beside it, so both must fit
/// the default queue's 64 command buffers or `begin` blocks.
const MAX_TRIALS: usize = 64 / 2 - WARMUPS;

/// The `--trials` value: a whole number in `1..=MAX_TRIALS`, refused by name
/// otherwise (no trial has no median to take; more would block the queue).
fn trials_of(value: Option<&String>) -> Result<usize, String> {
    match value.map(|v| (v, v.parse::<usize>())) {
        None => Err("--trials takes a value".to_string()),
        Some((_, Ok(n))) if (1..=MAX_TRIALS).contains(&n) => Ok(n),
        Some((value, _)) => Err(format!(
            "--trials takes a whole number in 1..={MAX_TRIALS}, not {value:?}"
        )),
    }
}

/// The `--only` value: one of the probe's weight types, refused by name
/// otherwise (a selection that matches no shape would skip every shape and
/// exit as a success).
fn only_of(value: Option<&String>) -> Result<&str, String> {
    let Some(value) = value else {
        return Err("--only takes a value".to_string());
    };
    if [Quant::Q4K, Quant::Q6K]
        .iter()
        .any(|q| q.name() == value.as_str())
    {
        Ok(value)
    } else {
        Err(format!(
            "--only takes {} or {}, not {value:?}",
            Quant::Q4K.name(),
            Quant::Q6K.name()
        ))
    }
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trials = match args
        .iter()
        .position(|a| a == "--trials")
        .map_or(Ok(16), |at| trials_of(args.get(at + 1)))
    {
        Ok(trials) => trials,
        Err(e) => {
            println!("metal_matvec_time: {e}");
            return ExitCode::from(64);
        }
    };
    let only = match args
        .iter()
        .position(|a| a == "--only")
        .map_or(Ok(None), |at| only_of(args.get(at + 1)).map(Some))
    {
        Ok(only) => only,
        Err(e) => {
            println!("metal_matvec_time: {e}");
            return ExitCode::from(64);
        }
    };
    let check = !args.iter().any(|a| a == "--no-check");
    let show_gaps = args.iter().any(|a| a == "--gaps");

    let device = match MetalDevice::open() {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("metal_matvec_time: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let kernels = match Kernels::new(&device) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("metal_matvec_time: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "info\tdevice={}\tlib={}\tbackend={}\ttrials={trials}\twarmups={WARMUPS}\twindow_bytes={WINDOW_BYTES}\tworking_set_bytes={WORKING_SET_BYTES}",
        info.name,
        device.library().path().name(),
        <MetalDevice as Device>::BACKEND
    );
    let mut failed = false;
    for quant in [Quant::Q4K, Quant::Q6K] {
        if only.is_some_and(|o| o != quant.name()) {
            continue;
        }
        for (rows, cols) in SHAPES {
            match time_shape(&device, &kernels, quant, rows, cols, trials, check) {
                Ok(t) => {
                    if show_gaps {
                        let us: Vec<String> =
                            t.gaps.iter().map(|g| format!("{:.0}", g * 1e6)).collect();
                        println!("gaps_us\t{}", us.join(" "));
                    }
                    let bytes = rows * cols / BLOCK_VALUES * quant.block_bytes();
                    println!(
                        "matvec\ttype={}\trows={rows}\tcols={cols}\tdispatches={}\ttrials={trials}\tclean={}\tbytes={bytes}\tcopies={}\tworking_set_bytes={}\tgbps_median={:.2}\tus_median={:.2}\tencode_us={:.2}",
                        quant.name(),
                        t.dispatches,
                        t.clean,
                        t.copies,
                        t.working_set,
                        t.median,
                        bytes as f64 / t.median / 1e3,
                        t.encode * 1e6,
                    );
                }
                Err(e) => {
                    println!(
                        "matvec\ttype={}\trows={rows}\tcols={cols}\tFAIL\t{e}",
                        quant.name()
                    );
                    failed = true;
                }
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
