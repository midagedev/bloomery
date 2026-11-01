//! The dense GEMM timing probe's body (macOS only).
//!
//! The method is the mat-vec probe's (`metal_matvec_time/mac.rs`): a working
//! set of at least 1 GiB of weight copies, so a timed dispatch never reads a
//! place another one left inside the cache's reach; command buffers of
//! enough dispatches to cover a fixed span of work, each submitted when the
//! one before it has completed (a buffer queued behind another is stamped
//! from the moment the GPU takes it up, which on the M4 Pro is while the
//! one before it still runs: its interval reads twice the buffer's time);
//! each timed buffer's own GPU interval, read after it has completed. A
//! dispatch's time is its buffer's interval over the dispatches in it, and
//! the figure is the median of the timed buffers — two warm-ups first, as
//! the mat-vec probe runs.
//!
//! What is timed: every weight type (Q4_K, Q6_K, MLX's A4), the arms, the
//! batched prefill's dense shapes at the activation row counts the design's
//! cost model names — a layer's projections (8192×2048, 2048×4096), an
//! expert's (512×2048) and the output head's (248320×2048) — one line each:
//!
//! ```text
//! gemm quant=… op=… n=… k=… r=… with its milliseconds, TFLOPS and GB/s
//! ```
//!
//! The arms (`--arm NAME`, repeatable; `f16` and `f32` by default):
//!
//! - `f16`, `f32`: the crate's simdgroup GEMM, launched through
//!   `Kernels::gemm`, on every device (the GEMM path the device's own rule
//!   picks is printed in the header, not timed by these arms).
//! - `tensor16`, `tensor32`: the tensor-op GEMM (`gemm_tensor.metal`, Metal
//!   4's `matmul2d`), launched through `Kernels::gemm` on kernels whose GEMMs
//!   come from that program, on any device that builds it.
//! - `tensor16_relaxed`, `tensor32_relaxed`: the same with the op's relaxed
//!   precision on (MLX's and llama.cpp's setting; the engine keeps it off).
//! - `mlx16`, `mlx32`: MLX's `qmm_t` for 4-bit group-64 weights, A4 only
//!   (`probe_mlx_qmm.metal`, its own library), reading MLX's layout; its
//!   output is checked against a CPU product.
//! - `<variant>_f16`, `<variant>_f32`: the shipped GEMM source with one
//!   edit made (`variants.rs`), in a library of its own.
//!
//! `tflops` is `2·N·K·R / s` on the median. `gbps` counts the bytes one
//! dispatch issues: the weights once, the activations once per weight-tile
//! threadgroup and the output once — the traffic the kernel asks for, not
//! what a cache serves. A config whose buffers this device cannot hold
//! prints a `skipped` line with the reason and costs nothing.
//!
//! This binary is a development-loop instrument. A figure it prints is this
//! machine's, taken with no timing lease, and is not a published number.
//! The kernels gate holds the GEMM's correctness; this probe times, and
//! checks nothing but the reference arm — a launch it could not encode is a
//! named failure.
//!
//! Flags: `--trials N` (the timed buffers per config, at least one),
//! `--warmups N`, `--gflop N` (the work one buffer covers), `--copies N` (the
//! weight copies the dispatches rotate over, at most), `--trace` (each
//! buffer's GPU interval beside the host's own clock), `--warm-ms N`
//! (warm-up buffers for at least this long), `--hash` (a hash of the output
//! the last dispatch wrote), `--arm NAME`,
//! `--only SUBSTRING` (repeatable: the configs whose line would contain it),
//! `--src NAME:ROWS:ACTS:THREADS:ORDER:PATH` (repeatable: the engine
//! library's whole text read from PATH at run time, its tile ROWS weight rows
//! by ACTS activation rows on THREADS threads, the activation tiles (`a`) or
//! the weight tiles (`w`) fastest across the grid, timed as the arms
//! `NAME_f16` and `NAME_f32`; a design is tried this way without a build),
//! `--dump-source
//! PATH` (write the engine library's text, the starting point of a `--src`
//! file, and exit),
//! `--quant NAME` (repeatable), `--shape N:K:R` (repeatable: only these
//! shapes, any N, K and R), `--pipelines` (print each pipeline's thread
//! limit and exit), `--peak` (time the machine's own ruler,
//! `probe_peak.metal`, and exit), `--help`.
//!
//! `--routed` times the experts' grouped GEMM over a route table instead of
//! the dense shapes (`routed.rs`, with its own `--rows`, `--skew` and
//! `--hot`).
//!
//! The header's `gemm:` line names the device's GPU architecture, whether
//! it reports Metal 4's family, the matrix-unit rule's verdict and the GEMM
//! path the engine takes on it.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, Elems, LAYER_NONE, Stream};
use bloomery_metal::encode::{Binding, Bindings, ComputeEncoder};
use bloomery_metal::kernels::{
    GEMM_ACTS, GEMM_ROWS, GEMM_THREADS, GemmMode, GemmParams, GemmX, Kernels, Operand, Quant,
};
use bloomery_metal::library::{SOURCE, TENSOR_SOURCE};
use bloomery_metal::{GpuInterval, Library, MetalBuffer, MetalStream};

#[path = "../shared/oracle.rs"]
#[allow(
    dead_code,
    reason = "the probe reads the oracle's activations and its Rng; the gate reads the rest"
)]
mod oracle;

#[path = "../shared/synthetic.rs"]
#[allow(
    dead_code,
    reason = "the probe reads the weights builder; the gates read their own"
)]
mod synthetic;

mod mlx;
mod routed;
mod variants;

/// The ruler's kernels, compiled into a library of their own.
const PEAK_SOURCE: &str = include_str!("probe_peak.metal");

use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::HOST;

/// The working set the weights rotate through: large enough that a timed
/// dispatch's weights were not just read by another.
const WORKING_SET_BYTES: usize = 1 << 30;

/// The warm-up buffers before the timed ones.
const WARMUPS: usize = 2;

/// The host milliseconds the warm-up buffers run for at least: the GPU's
/// clocks follow the load, and a configuration's own span is short.
const WARM_MS: f64 = 150.0;

/// The timed buffers per configuration.
const TRIALS: usize = 5;

/// How long the host waits for one timed command buffer.
const BUFFER_WAIT: Duration = Duration::from_secs(120);

/// The work one command buffer covers, in GFLOP: enough that a buffer's
/// dispatches amortize the launch, and no more — the probe's own runtime is
/// this times the configurations.
const BUFFER_GFLOP: f64 = 2.0;

/// The shapes the design's cost model names: a layer's projections, an
/// expert's, and the output head's (N weight rows by K columns).
const SHAPES: [(usize, usize); 4] = [(8192, 2048), (2048, 4096), (512, 2048), (248_320, 2048)];

/// The activation row counts the design's per-chunk model names.
const ROWS: [usize; 4] = [32, 128, 512, 2048];

/// The weight rows a reference threadgroup owns (`probe_mlx_qmm.metal`'s
/// `MLXREF_BN`, which is also its activation rows).
const MLX_TILE: usize = 32;

pub fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("metal_gemm_time: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The value after `name`, when the flag is given.
fn flag<'a>(args: &'a [String], name: &str) -> Option<Option<&'a str>> {
    args.iter()
        .position(|a| a == name)
        .map(|at| args.get(at + 1).map(String::as_str))
}

/// The whole number after `name`, `default` when the flag is not given.
fn count_flag(args: &[String], name: &str, min: usize, default: usize) -> Result<usize, String> {
    match flag(args, name) {
        Some(Some(v)) => v
            .parse()
            .ok()
            .filter(|&n| n >= min)
            .ok_or_else(|| format!("{name} takes a whole number of at least {min}")),
        Some(None) => unreachable!("a flag without its value was refused"),
        None => Ok(default),
    }
}

/// The timing method's knobs.
#[derive(Clone, Copy)]
struct Method {
    trials: usize,
    warmups: usize,
    gflop: f64,
    /// The weight copies the dispatches rotate over, at most (the working
    /// set's own count when it is smaller).
    copies: usize,
    /// Print each buffer's GPU interval beside the host's own clock.
    trace: bool,
    /// The host milliseconds of warm-up buffers at least, before the timed
    /// ones.
    warm_ms: f64,
    /// Print a hash of the output the last dispatch wrote, so two arms that
    /// should write the same bits can be seen to.
    hash: bool,
}

/// What is timed at a configuration.
#[derive(Clone, Copy)]
enum Arm {
    /// The crate's simdgroup GEMM, through `Kernels::gemm`.
    Engine(Operand),
    /// The tensor-op GEMM, through `Kernels::gemm`; the op's relaxed
    /// precision on when the flag is.
    Tensor(Operand, bool),
    /// MLX's `qmm_t`, A4 only.
    Mlx(Operand),
    /// The shipped source with a variant's edits.
    Variant(&'static variants::Variant, Operand),
    /// A whole library text read at run time (`--src`).
    Source(&'static SourceArm, Operand),
}

/// A `--src` arm: the library text read from a file, and the tile its
/// kernels own, which sizes the grid.
struct SourceArm {
    name: String,
    text: String,
    rows: usize,
    acts: usize,
    threads: usize,
    /// The activation tiles run fastest across the grid.
    acts_fast: bool,
}

impl SourceArm {
    /// `NAME:ROWS:ACTS:THREADS:ORDER:PATH`, the file read now.
    fn parse(spec: &str) -> Result<&'static Self, String> {
        let parts: Vec<&str> = spec.splitn(6, ':').collect();
        let [name, rows, acts, threads, order, path] = parts[..] else {
            return Err(format!(
                "--src takes NAME:ROWS:ACTS:THREADS:ORDER:PATH, not {spec}"
            ));
        };
        let acts_fast = match order {
            "a" => true,
            "w" => false,
            _ => return Err(format!("--src {name}: ORDER is a or w, not {order}")),
        };
        let whole = |v: &str, what: &str| {
            v.parse::<usize>().ok().filter(|&n| n > 0).ok_or(format!(
                "--src {name}: {what} takes a positive whole number, not {v}"
            ))
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("--src {name}: reading {path}: {e}"))?;
        Ok(Box::leak(Box::new(Self {
            name: name.to_string(),
            text,
            rows: whole(rows, "ROWS")?,
            acts: whole(acts, "ACTS")?,
            threads: whole(threads, "THREADS")?,
            acts_fast,
        })))
    }
}

fn operand_name(operand: Operand) -> &'static str {
    match operand {
        Operand::F16 => "f16",
        Operand::F32 => "f32",
    }
}

impl Arm {
    /// The arm's name in the `op=` field.
    fn name(self) -> String {
        match self {
            Self::Engine(o) => operand_name(o).to_string(),
            Self::Tensor(o, relaxed) => format!(
                "tensor{}{}",
                &operand_name(o)[1..],
                if relaxed { "_relaxed" } else { "" }
            ),
            Self::Mlx(Operand::F16) => "mlx16".to_string(),
            Self::Mlx(Operand::F32) => "mlx32".to_string(),
            Self::Variant(v, o) => format!("{}_{}", v.name, operand_name(o)),
            Self::Source(s, o) => format!("{}_{}", s.name, operand_name(o)),
        }
    }

    /// The arm named `name`: `f16`, `f32`, `tensor16`, `tensor32`,
    /// `tensor16_relaxed`, `tensor32_relaxed`, `mlx16`, `mlx32` or
    /// `<variant>_f16` / `<variant>_f32`.
    fn parse(name: &str, sources: &[&'static SourceArm]) -> Result<Self, String> {
        for (suffix, operand) in [("_f16", Operand::F16), ("_f32", Operand::F32)] {
            if let Some(stem) = name.strip_suffix(suffix)
                && let Some(s) = sources.iter().find(|s| s.name == stem)
            {
                return Ok(Self::Source(s, operand));
            }
        }
        match name {
            "f16" => return Ok(Self::Engine(Operand::F16)),
            "f32" => return Ok(Self::Engine(Operand::F32)),
            "tensor16" => return Ok(Self::Tensor(Operand::F16, false)),
            "tensor32" => return Ok(Self::Tensor(Operand::F32, false)),
            "tensor16_relaxed" => return Ok(Self::Tensor(Operand::F16, true)),
            "tensor32_relaxed" => return Ok(Self::Tensor(Operand::F32, true)),
            "mlx16" => return Ok(Self::Mlx(Operand::F16)),
            "mlx32" => return Ok(Self::Mlx(Operand::F32)),
            _ => {}
        }
        for (suffix, operand) in [("_f16", Operand::F16), ("_f32", Operand::F32)] {
            if let Some(stem) = name.strip_suffix(suffix)
                && let Some(v) = variants::find(stem)
            {
                return Ok(Self::Variant(v, operand));
            }
        }
        let known: Vec<String> = variants::VARIANTS
            .iter()
            .map(|v| v.name.to_string())
            .collect();
        Err(format!(
            "no arm named {name}: f16, f32, tensor16, tensor32, tensor16_relaxed, tensor32_relaxed, mlx16, mlx32, or <variant>_f16 / _f32 of {}",
            known.join(", ")
        ))
    }
}

/// The libraries the arms compile on demand, one per source.
struct Libs {
    mlx: Option<Library>,
    variants: Vec<(&'static str, Library)>,
    sources: Vec<(String, Library)>,
    /// The tensor-op program and the kernels over it, strict then relaxed.
    tensor: [Option<(Library, Kernels)>; 2],
}

/// The tensor-op GEMMs' program, with the op's relaxed precision on when
/// `relaxed` is.
pub(crate) fn tensor_library(device: &MetalDevice, relaxed: bool) -> Result<Library, String> {
    const STRICT: &str = "#define TENSOR_RELAXED false";
    let text = if relaxed {
        if TENSOR_SOURCE.matches(STRICT).count() != 1 {
            return Err(format!(
                "the tensor-op source holds {STRICT:?} other than once"
            ));
        }
        TENSOR_SOURCE.replace(STRICT, "#define TENSOR_RELAXED true")
    } else {
        TENSOR_SOURCE.to_string()
    };
    Library::tensor_from_source_text(device, &text)
        .map_err(|e| format!("the tensor-op GEMMs do not build here: {e}"))
}

impl Libs {
    /// The tensor-op program and the kernels whose GEMMs come from it.
    fn tensor(
        &mut self,
        device: &MetalDevice,
        relaxed: bool,
    ) -> Result<&(Library, Kernels), String> {
        let slot = &mut self.tensor[usize::from(relaxed)];
        if slot.is_none() {
            let library = tensor_library(device, relaxed)?;
            let kernels =
                Kernels::from_library(&device.library().with_tensor_gemms(library.clone()))
                    .map_err(why)?;
            *slot = Some((library, kernels));
        }
        Ok(slot.as_ref().expect("the kernels were just made"))
    }

    fn mlx(&mut self, device: &MetalDevice) -> Result<&Library, String> {
        if self.mlx.is_none() {
            self.mlx = Some(Library::from_source_text(device, mlx::SOURCE).map_err(why)?);
        }
        Ok(self.mlx.as_ref().expect("the library was just made"))
    }

    fn source(&mut self, device: &MetalDevice, s: &'static SourceArm) -> Result<&Library, String> {
        if !self.sources.iter().any(|(name, _)| *name == s.name) {
            let library = Library::from_source_text(device, &s.text).map_err(why)?;
            self.sources.push((s.name.clone(), library));
        }
        Ok(self
            .sources
            .iter()
            .find(|(name, _)| *name == s.name)
            .map(|(_, l)| l)
            .expect("the library was just made"))
    }

    fn variant(
        &mut self,
        device: &MetalDevice,
        v: &'static variants::Variant,
    ) -> Result<&Library, String> {
        if !self.variants.iter().any(|(name, _)| *name == v.name) {
            let text = variants::apply(v, SOURCE)?;
            let library = Library::from_source_text(device, &text).map_err(why)?;
            self.variants.push((v.name, library));
        }
        Ok(self
            .variants
            .iter()
            .find(|(name, _)| *name == v.name)
            .map(|(_, l)| l)
            .expect("the library was just made"))
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|a| a == "--help") {
        println!(
            "metal_gemm_time [--trials N] [--warmups N] [--gflop N] [--arm NAME]... \
             [--only SUBSTRING]... [--quant NAME]... [--shape N:K:R]... [--copies N] [--trace] [--warm-ms N] [--hash] [--pipelines] \
             [--src NAME:ROWS:ACTS:THREADS:ORDER:PATH]...\n\
             metal_gemm_time --dump-source PATH\n\
             metal_gemm_time --routed [--rows U]... [--skew S]... [--arm NAME]... [--hot] [--trials N] [--hash]"
        );
        return Ok(());
    }
    for name in [
        "--trials",
        "--warmups",
        "--gflop",
        "--copies",
        "--warm-ms",
        "--only",
        "--arm",
        "--quant",
        "--shape",
        "--rows",
        "--skew",
        "--src",
        "--dump-source",
    ] {
        if let Some(None) = flag(&args, name) {
            return Err(format!("{name} takes a value"));
        }
    }
    let method = Method {
        trials: count_flag(&args, "--trials", 1, TRIALS)?,
        warmups: count_flag(&args, "--warmups", 0, WARMUPS)?,
        gflop: match flag(&args, "--gflop") {
            Some(Some(v)) => v
                .parse::<f64>()
                .ok()
                .filter(|g| g.is_finite() && *g > 0.0)
                .ok_or("--gflop takes a positive number")?,
            _ => BUFFER_GFLOP,
        },
        copies: count_flag(&args, "--copies", 1, usize::MAX)?,
        trace: args.iter().any(|a| a == "--trace"),
        hash: args.iter().any(|a| a == "--hash"),
        warm_ms: match flag(&args, "--warm-ms") {
            Some(Some(v)) => v
                .parse::<f64>()
                .ok()
                .filter(|g| g.is_finite() && *g >= 0.0)
                .ok_or("--warm-ms takes a number of milliseconds")?,
            _ => WARM_MS,
        },
    };
    let only: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "--only")
        .map(|w| w[1].clone())
        .collect();
    let quants: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "--quant")
        .map(|w| w[1].clone())
        .collect();
    let shapes: Vec<(usize, usize, usize)> = args
        .windows(2)
        .filter(|w| w[0] == "--shape")
        .map(|w| {
            let parts: Vec<usize> = w[1].split(':').filter_map(|p| p.parse().ok()).collect();
            match parts[..] {
                [n, k, r] => Ok((n, k, r)),
                _ => Err(format!("--shape takes N:K:R, not {}", w[1])),
            }
        })
        .collect::<Result<_, _>>()?;
    let routed = args
        .iter()
        .any(|a| a == "--routed")
        .then(|| routed_opts(&args))
        .transpose()?;
    if let Some(Some(path)) = flag(&args, "--dump-source") {
        std::fs::write(path, SOURCE).map_err(|e| format!("--dump-source {path}: {e}"))?;
        return Ok(());
    }
    let sources: Vec<&'static SourceArm> = args
        .windows(2)
        .filter(|w| w[0] == "--src")
        .map(|w| SourceArm::parse(&w[1]))
        .collect::<Result<_, _>>()?;
    let mut arms: Vec<Arm> = if routed.is_some() {
        Vec::new()
    } else {
        args.windows(2)
            .filter(|w| w[0] == "--arm")
            .map(|w| Arm::parse(&w[1], &sources))
            .collect::<Result<_, _>>()?
    };
    if arms.is_empty() && routed.is_none() {
        arms = vec![Arm::Engine(Operand::F16), Arm::Engine(Operand::F32)];
    }

    let device = MetalDevice::open(&HOST).map_err(|e| format!("the Metal device: {e}"))?;
    let info = device.info().map_err(why)?;
    let stream = device.stream();
    println!(
        "metal-gemm-time: device={} backend=metal library={}",
        info.name,
        device.library().path().name()
    );
    let units = device.library().matrix_units();
    println!(
        "gemm: architecture={} metal4={} matrix-units={} engine-path={}",
        units.architecture,
        units.metal4,
        units.present(),
        device.library().gemm_path().name()
    );
    let mut libs = Libs {
        mlx: None,
        variants: Vec::new(),
        sources: Vec::new(),
        tensor: [None, None],
    };
    for arm in &arms {
        if let Arm::Variant(v, _) = arm {
            println!("variant {}: {}", v.name, v.what);
        }
    }
    if args.iter().any(|a| a == "--pipelines") {
        return pipelines(&device, &mut libs, &arms);
    }
    if args.iter().any(|a| a == "--peak") {
        return peaks(&device, stream, method);
    }
    println!(
        "working-set={}MiB trials={} warmups={} warm_ms={} gflop={} copies<={}",
        WORKING_SET_BYTES >> 20,
        method.trials,
        method.warmups,
        method.warm_ms,
        method.gflop,
        method.copies
    );
    // The `f16` and `f32` arms' GEMMs are the simdgroup ones on every device.
    let kernels = Kernels::from_library(&device.library().with_simdgroup_gemms()).map_err(why)?;
    let ctx = Ctx {
        device: &device,
        stream,
        kernels: &kernels,
        method,
    };
    if let Some(opts) = &routed {
        return routed::run(&ctx, opts);
    }
    // The `--shape` list when one is given, any N, K and R; the design's
    // shapes at its row counts otherwise.
    let configs: Vec<(usize, usize, usize)> = if shapes.is_empty() {
        SHAPES
            .iter()
            .flat_map(|&(n, k)| ROWS.iter().map(move |&r| (n, k, r)))
            .collect()
    } else {
        shapes
    };
    for quant in [Quant::Q4K, Quant::Q6K, Quant::A4] {
        if !quants.is_empty() && !quants.iter().any(|q| q == quant.name()) {
            continue;
        }
        for &arm in &arms {
            if matches!(arm, Arm::Mlx(_)) && quant != Quant::A4 {
                continue;
            }
            for &(n, k, r) in &configs {
                let line = format!("quant={} op={} n={n} k={k} r={r}", quant.name(), arm.name());
                if !only.is_empty() && !only.iter().any(|s| line.contains(s)) {
                    continue;
                }
                let cfg = Config { quant, n, k, r };
                one(&ctx, &mut libs, arm, &cfg).map_err(|e| format!("{line}: {e}"))?;
            }
        }
    }
    Ok(())
}

/// The routed mode's flags: `--rows U` and `--skew S` (repeatable), the
/// routed arms by `--arm`, `--hot`.
fn routed_opts(args: &[String]) -> Result<routed::Opts, String> {
    let values = |name: &str| -> Vec<&str> {
        args.windows(2)
            .filter(|w| w[0] == name)
            .map(|w| w[1].as_str())
            .collect()
    };
    let mut rows: Vec<usize> = values("--rows")
        .iter()
        .map(|v| {
            v.parse()
                .ok()
                .filter(|&u: &usize| u > 0)
                .ok_or(format!("--rows takes a positive whole number, not {v}"))
        })
        .collect::<Result<_, _>>()?;
    if rows.is_empty() {
        rows = vec![512, 2048];
    }
    let mut skews: Vec<f64> = values("--skew")
        .iter()
        .map(|v| {
            v.parse()
                .ok()
                .filter(|s: &f64| s.is_finite() && *s >= 0.0)
                .ok_or(format!("--skew takes a number at least 0, not {v}"))
        })
        .collect::<Result<_, _>>()?;
    if skews.is_empty() {
        skews = vec![0.0];
    }
    let mut arms: Vec<String> = values("--arm").iter().map(|v| (*v).to_string()).collect();
    if arms.is_empty() {
        arms = vec!["ship".to_string()];
    }
    Ok(routed::Opts {
        rows,
        skews,
        arms,
        hot: args.iter().any(|a| a == "--hot"),
    })
}

/// Print each arm's pipeline thread limit: the register file's answer to
/// how many threads the compiled kernel can hold in a group.
fn pipelines(device: &MetalDevice, libs: &mut Libs, arms: &[Arm]) -> Result<(), String> {
    for &arm in arms {
        for quant in [Quant::Q4K, Quant::Q6K, Quant::A4] {
            let (library, name): (&Library, String) = match arm {
                Arm::Engine(o) => (device.library(), engine_kernel(quant, o)),
                Arm::Tensor(o, relaxed) => {
                    (&libs.tensor(device, relaxed)?.0, engine_kernel(quant, o))
                }
                Arm::Mlx(o) => {
                    if quant != Quant::A4 {
                        continue;
                    }
                    (libs.mlx(device)?, mlx::kernel_name(o).to_string())
                }
                Arm::Variant(v, o) => (libs.variant(device, v)?, engine_kernel(quant, o)),
                Arm::Source(s, o) => (libs.source(device, s)?, engine_kernel(quant, o)),
            };
            let pipeline = library.pipeline(&name).map_err(why)?;
            println!(
                "pipeline {name} arm={} max_threads={}",
                arm.name(),
                pipeline.max_threads_per_group()
            );
        }
    }
    Ok(())
}

/// The machine's own ruler: kernels with no memory traffic but two small
/// operand loads a round, each timed as the GEMM arms are. The FLOPs are the
/// kernels' own count (`probe_peak.metal`: eight 8x8x8 multiply-accumulates
/// a simdgroup for 1024 rounds).
fn peaks(device: &MetalDevice, stream: &MetalStream, method: Method) -> Result<(), String> {
    const MMA_GROUPS: usize = 512;
    const DISPATCHES: usize = 4;
    let library = Library::from_source_text(device, PEAK_SOURCE).map_err(why)?;
    let ctx = Ctx {
        device,
        stream,
        kernels: &Kernels::from_library(&device.library().with_simdgroup_gemms()).map_err(why)?,
        method,
    };
    let floats = vec![0.0625f32; 512];
    let halves: Vec<u16> = vec![gguf::quant::f32_to_f16_bits(0.0625); 512];
    let in_f32 = device.upload(stream, &floats).map_err(why)?;
    let in_f16 = device.upload(stream, &halves).map_err(why)?;
    let out_elems = MMA_GROUPS * 4 * 8 * 64;
    let mut out = device
        .zeroed::<f32>(stream, Elems(out_elems))
        .map_err(why)?;
    let mma_flops = (MMA_GROUPS * 4) as f64 * 1024.0 * 8.0 * 1024.0;
    for (name, groups, flops, half) in [
        ("probe_peak_mma_f32", MMA_GROUPS, mma_flops, false),
        ("probe_peak_mma_f16", MMA_GROUPS, mma_flops, true),
    ] {
        let pipeline = library.pipeline(name).map_err(why)?;
        let Timed { seconds, wall } = time_dispatches(&ctx, 1, DISPATCHES, |encoder, _| {
            encoder.set_pipeline(&pipeline);
            let y_win = out.window_mut(Elems(0), Elems(out_elems)).map_err(why)?;
            let mut slot = Bindings::new(encoder);
            if half {
                slot.set(Binding::of(&in_f16.whole().map_err(why)?).map_err(why)?)
                    .map_err(why)?;
            } else {
                slot.set(Binding::of(&in_f32.whole().map_err(why)?).map_err(why)?)
                    .map_err(why)?;
            }
            slot.set(Binding::of_mut(&y_win).map_err(why)?)
                .map_err(why)?;
            slot.dispatch_groups(groups, 1, GEMM_THREADS).map_err(why)
        })?;
        println!(
            "peak kind={name} ms={:.3} tflops={:.2} wall_ms={:.3}",
            seconds * 1e3,
            flops / seconds / 1e12,
            wall * 1e3
        );
    }
    Ok(())
}

/// The engine kernel's name for the plain store form.
fn engine_kernel(quant: Quant, operand: Operand) -> String {
    format!("gemm_{}_store_{}", quant.name(), operand_name(operand))
}

/// What every configuration shares.
struct Ctx<'a> {
    device: &'a MetalDevice,
    stream: &'a MetalStream,
    kernels: &'a Kernels,
    method: Method,
}

/// One timed configuration: the weight type and the shape.
struct Config {
    quant: Quant,
    n: usize,
    k: usize,
    r: usize,
}

/// `one` copies of a weight tensor's byte parts, each part padded to a
/// 256-byte stride so every binding's offset is aligned.
struct Pool {
    buffer: MetalBuffer<u8>,
    /// The bytes between one copy's start and the next's.
    stride: usize,
    /// Each part's offset from its copy's start.
    offsets: Vec<usize>,
    /// Each part's length.
    lens: Vec<usize>,
    copies: usize,
}

impl Pool {
    fn new(ctx: &Ctx<'_>, parts: &[&[u8]]) -> Result<Self, String> {
        let mut offsets = Vec::with_capacity(parts.len());
        let mut at = 0usize;
        for part in parts {
            offsets.push(at);
            at += part.len().div_ceil(256) * 256;
        }
        let stride = at;
        let copies = WORKING_SET_BYTES
            .div_ceil(stride)
            .max(1)
            .min(ctx.method.copies);
        let mut bytes = vec![0u8; copies * stride];
        for c in 0..copies {
            for (part, off) in parts.iter().zip(&offsets) {
                bytes[c * stride + off..c * stride + off + part.len()].copy_from_slice(part);
            }
        }
        let buffer = ctx.device.upload(ctx.stream, &bytes).map_err(why)?;
        Ok(Self {
            buffer,
            stride,
            offsets,
            lens: parts.iter().map(|p| p.len()).collect(),
            copies,
        })
    }

    /// Part `part` of copy `copy`.
    fn window(
        &self,
        copy: usize,
        part: usize,
    ) -> Result<bloomery_device::Window<'_, u8, MetalDevice>, String> {
        self.buffer
            .window(
                Elems(copy * self.stride + self.offsets[part]),
                Elems(self.lens[part]),
            )
            .map_err(why)
    }
}

/// Time one configuration of one arm and print its line.
fn one(ctx: &Ctx<'_>, libs: &mut Libs, arm: Arm, cfg: &Config) -> Result<(), String> {
    let Config { quant, n, k, r } = *cfg;
    let op_name = arm.name();
    let skip = |why: String| {
        println!(
            "gemm quant={} op={op_name} n={n} k={k} r={r} skipped ({why})",
            quant.name()
        );
        Ok(())
    };

    let seed = 0x6B65_726E_656C ^ ((n as u64) << 20) ^ ((k as u64) << 8) ^ (r as u64);
    let blocks = synthetic::matrix(quant, n, k, seed);
    let x_values = oracle::activations(r * k, seed ^ 0xA11CE, 1.0);

    // The arm's weights and activations, as its kernel takes them.
    let (pool, x_bytes, mlx_layout) = match arm {
        Arm::Mlx(operand) => {
            let layout = mlx::layout(&blocks, n, k, operand);
            let pool = Pool::new(ctx, &[&layout.codes, &layout.scales, &layout.biases])?;
            (
                pool,
                Some(mlx::activations(&x_values, operand)),
                Some(layout),
            )
        }
        _ => (Pool::new(ctx, &[&blocks])?, None, None),
    };
    let bytes = blocks.len();
    drop(blocks);

    let x_f32 = ctx.device.upload(ctx.stream, &x_values).map_err(why)?;
    let x_half = match &x_bytes {
        Some(b) => Some(ctx.device.upload(ctx.stream, b).map_err(why)?),
        None => None,
    };
    let mut y = match ctx.device.zeroed::<f32>(ctx.stream, Elems(r * n)) {
        Ok(y) => y,
        Err(e) => return skip(format!("the output of {r}x{n} f32: {e}")),
    };

    let dispatches = ((ctx.method.gflop * 1e9 / (2.0 * (n * k * r) as f64)).ceil() as usize)
        .max(1)
        .min(pool.copies);
    let fault = ctx.device.fault_word();

    let timed = match arm {
        Arm::Engine(operand) | Arm::Tensor(operand, _) => {
            let kernels = match arm {
                Arm::Tensor(_, relaxed) => &libs.tensor(ctx.device, relaxed)?.1,
                _ => ctx.kernels,
            };
            time_dispatches(ctx, pool.copies, dispatches, |encoder, copy| {
                kernels
                    .gemm(
                        encoder,
                        GemmParams {
                            quant,
                            rows: n,
                            cols: k,
                            acts: r,
                            weights: pool.window(copy, 0)?,
                            x: GemmX::F32 {
                                x: x_f32.window(Elems(0), Elems(r * k)).map_err(why)?,
                                stride: k,
                            },
                            y: y.window_mut(Elems(0), Elems(r * n)).map_err(why)?,
                            y_stride: n,
                            mode: GemmMode::Store,
                            operand,
                            fault,
                            layer: LAYER_NONE,
                        },
                    )
                    .map_err(|e| format!("the dispatch did not encode: {e}"))
            })?
        }
        Arm::Variant(_, operand) | Arm::Source(_, operand) => {
            let (library, rows, acts, threads, acts_fast) = match arm {
                Arm::Variant(v, _) => (
                    libs.variant(ctx.device, v)?,
                    v.rows,
                    v.acts,
                    GEMM_THREADS,
                    v.acts_fast,
                ),
                Arm::Source(s, _) => (
                    libs.source(ctx.device, s)?,
                    s.rows,
                    s.acts,
                    s.threads,
                    s.acts_fast,
                ),
                _ => unreachable!("the arm is a variant or a source"),
            };
            let pipeline = library
                .pipeline(&engine_kernel(quant, operand))
                .map_err(why)?;
            let words = [k as u32, n as u32, r as u32, k as u32, n as u32, LAYER_NONE];
            let args: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let (groups_w, groups_a) = (n.div_ceil(rows), r.div_ceil(acts));
            let (gx, gy) = if acts_fast {
                (groups_a, groups_w)
            } else {
                (groups_w, groups_a)
            };
            time_dispatches(ctx, pool.copies, dispatches, |encoder, copy| {
                encoder.set_pipeline(&pipeline);
                let weights = pool.window(copy, 0)?;
                let x = x_f32.window(Elems(0), Elems(r * k)).map_err(why)?;
                let y_win = y.window_mut(Elems(0), Elems(r * n)).map_err(why)?;
                let mut slot = Bindings::new(encoder);
                slot.set(Binding::of(&weights).map_err(why)?).map_err(why)?;
                slot.set(Binding::of(&x).map_err(why)?).map_err(why)?;
                slot.set(Binding::of_mut(&y_win).map_err(why)?)
                    .map_err(why)?;
                slot.set(Binding::bytes(&args).map_err(why)?).map_err(why)?;
                slot.set(fault.binding()).map_err(why)?;
                slot.dispatch_groups(gx, gy, threads).map_err(why)
            })?
        }
        Arm::Mlx(operand) => {
            let library = libs.mlx(ctx.device)?;
            let pipeline = library.pipeline(mlx::kernel_name(operand)).map_err(why)?;
            let words = [k as i32, n as i32, r as i32];
            let args: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let x_half = x_half.as_ref().expect("the reference arm's activations");
            let t = mlx::elem_bytes(operand);
            let timed = time_dispatches(ctx, pool.copies, dispatches, |encoder, copy| {
                encoder.set_pipeline(&pipeline);
                let (codes, scales, biases) = (
                    pool.window(copy, 0)?,
                    pool.window(copy, 1)?,
                    pool.window(copy, 2)?,
                );
                let x = x_half.window(Elems(0), Elems(r * k * t)).map_err(why)?;
                let y_win = y.window_mut(Elems(0), Elems(r * n)).map_err(why)?;
                let mut slot = Bindings::new(encoder);
                slot.set(Binding::of(&codes).map_err(why)?).map_err(why)?;
                slot.set(Binding::of(&scales).map_err(why)?).map_err(why)?;
                slot.set(Binding::of(&biases).map_err(why)?).map_err(why)?;
                slot.set(Binding::of(&x).map_err(why)?).map_err(why)?;
                slot.set(Binding::of_mut(&y_win).map_err(why)?)
                    .map_err(why)?;
                slot.set(Binding::bytes(&args).map_err(why)?).map_err(why)?;
                slot.dispatch_groups(n / MLX_TILE, r / MLX_TILE, GEMM_THREADS)
                    .map_err(why)
            })?;
            let mut words = vec![0f32; r * n];
            ctx.device
                .read(
                    ctx.stream,
                    y.window(Elems(0), Elems(r * n)).map_err(why)?,
                    &mut words,
                )
                .map_err(why)?;
            let back: Vec<u8> = words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .take(r * n * t)
                .collect();
            let layout = mlx_layout.as_ref().expect("the reference arm's layout");
            mlx::check(
                layout,
                x_bytes.as_ref().expect("the reference arm's activations"),
                &back,
                n,
                k,
                r,
                operand,
            )?;
            timed
        }
    };
    let Timed { seconds, wall } = timed;

    if ctx.method.hash && !matches!(arm, Arm::Mlx(_)) {
        let mut words = vec![0f32; r * n];
        ctx.device
            .read(
                ctx.stream,
                y.window(Elems(0), Elems(r * n)).map_err(why)?,
                &mut words,
            )
            .map_err(why)?;
        let hash = fnv(0xCBF2_9CE4_8422_2325, &words);
        println!(
            "hash quant={} op={op_name} n={n} k={k} r={r} y_hash={hash:016x}",
            quant.name()
        );
    }
    let flops = 2.0 * (n * k * r) as f64;
    // The traffic one dispatch issues: the weights once, the activations
    // once per weight-tile threadgroup, the output once.
    let (tile, elem) = match arm {
        Arm::Mlx(operand) => (MLX_TILE, mlx::elem_bytes(operand)),
        Arm::Variant(v, _) => (v.rows, 4),
        Arm::Source(s, _) => (s.rows, 4),
        _ => (GEMM_ROWS, 4),
    };
    let issued = bytes + (n / tile) * r * k * elem + r * n * elem;
    println!(
        "gemm quant={} op={op_name} n={n} k={k} r={r} \
         ms={:.3} tflops={:.2} gbps={:.1} wall_ms={:.3}",
        quant.name(),
        seconds * 1e3,
        flops / seconds / 1e12,
        issued as f64 / seconds / 1e9,
        wall * 1e3,
    );
    Ok(())
}

/// Record and submit the buffers one at a time, each when the one before it
/// has completed, and read each timed buffer's own GPU interval: the median
/// seconds per dispatch. The warm-up buffers run at least `warmups` (one at
/// the fewest) and until
/// `warm_ms` of host time has passed, so the clocks have followed the load
/// before the first timed buffer.
fn time_dispatches(
    ctx: &Ctx<'_>,
    copies: usize,
    dispatches: usize,
    mut encode: impl FnMut(&mut ComputeEncoder<'_>, usize) -> Result<(), String>,
) -> Result<Timed, String> {
    let Method {
        trials,
        warmups,
        warm_ms,
        ..
    } = ctx.method;
    let mut gaps = Vec::with_capacity(trials);
    let mut trace = Vec::new();
    let started = Instant::now();
    let mut b = 0usize;
    while gaps.len() < trials {
        // One recording per buffer, its dispatches rotating over the working
        // set's copies: no dispatch reads a place another one just left.
        let mut recording = ctx.stream.begin().map_err(why)?;
        let stamps = recording.stamps();
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            for j in 0..dispatches {
                encode(&mut encoder, (b * dispatches + j) % copies)?;
            }
            encoder.end();
        }
        ctx.stream.submit(recording).map_err(why)?;
        let gpu = match stamps.interval(BUFFER_WAIT) {
            Ok(GpuInterval::Ran(gpu)) => gpu,
            Ok(GpuInterval::StillRunning) => {
                return Err(
                    "the command buffer was still running on the GPU when the wait ended"
                        .to_string(),
                );
            }
            Err(e) => {
                let _drained = ctx.stream.sync();
                return Err(why(e));
            }
        };
        trace.push((gpu, started.elapsed().as_secs_f64()));
        let warm = b < warmups.max(1) || started.elapsed().as_secs_f64() * 1e3 < warm_ms;
        if !warm {
            gaps.push(gpu);
        }
        b += 1;
    }
    if ctx.method.trace {
        let mut last = 0.0;
        for (i, (gpu, done)) in trace.iter().enumerate() {
            println!(
                "trace buffer={i} gpu_ms={:.3} host_done_ms={:.3} host_step_ms={:.3}",
                gpu * 1e3,
                done * 1e3,
                (done - last) * 1e3
            );
            last = *done;
        }
    }
    ctx.stream.sync().map_err(why)?;
    // The host's clock over the timed buffers alone, steps between their
    // completions, as a cross-check of the GPU's stamps.
    let timed_from = trace.len() - trials;
    let wall = (trace[trace.len() - 1].1 - trace[timed_from - 1].1) / (trials * dispatches) as f64;
    let seconds = median(&mut gaps) / dispatches as f64;
    if !(seconds.is_finite() && seconds > 0.0) {
        return Err(format!(
            "the GPU intervals give {seconds:?} s per dispatch, not a positive time"
        ));
    }
    Ok(Timed { seconds, wall })
}

/// What a timed arm measured: the median seconds per dispatch of the
/// buffers' own GPU intervals, and the host's wall seconds per dispatch over
/// the whole run (every buffer, the warm-ups too) as a cross-check of the
/// GPU's stamps.
struct Timed {
    seconds: f64,
    wall: f64,
}

/// The median of `values`, in place.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// The activation rows and weight rows one threadgroup owns, as the host
/// launches them: the probe names the shapes, the geometry constants are the
/// kernel's own, and this is the check that the two agree.
const _: () = assert!(GEMM_ACTS == 64 && GEMM_ROWS == 64);

/// FNV-1a over the words of `v`.
fn fnv(h: u64, v: &[f32]) -> u64 {
    v.iter().fold(h, |h, w| {
        (h ^ u64::from(w.to_bits())).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn shape_requires_exactly_three_whole_numbers() {
        let overflow = format!("{}0", usize::MAX);
        for shape in [
            "1:bad:2:3",
            "bad:1:2:3",
            "1:2:3:bad",
            "1::2:3",
            "1:2:3:",
            "1:2",
            "1:2:3:4",
            "1:bad:3",
            "1:-2:3",
            "1: 2:3",
            &format!("1:{overflow}:2:3"),
        ] {
            let args = ["--shape", "4:5:6", "--shape", shape]
                .map(String::from)
                .to_vec();
            assert_eq!(
                super::run(args),
                Err(format!("--shape takes N:K:R, not {shape}")),
                "every component of every --shape value must parse"
            );
        }
        let largest = format!("{}:2:3", usize::MAX);
        let args = [
            "--shape",
            "1:2:3",
            "--shape",
            &largest,
            "--dump-source",
            "/dev/null",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(
            super::run(args),
            Ok(()),
            "valid repeated shapes must reach the source-only exit without opening a device"
        );
    }
}
