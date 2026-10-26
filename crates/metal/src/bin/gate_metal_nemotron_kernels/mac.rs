//! The Nemotron kernels' gate clauses (macOS only).
//!
//! One clause per line, `PASS`, `SKIP` or `FAIL`, and the run's exit code is
//! 1 when any clause failed, 78 when this machine cannot run the backend (a
//! macOS below the floor, no Metal device), 0 otherwise. The weights are
//! seeded synthetic rows at the models' real shapes, and the Nano fixture's
//! own tensors where the file is present; the oracle is `crates/gguf`'s
//! dequantization of the same blocks with an f64 dot, held to the band
//! `oracle.rs` derives. A clause that needs the fixture and finds none is
//! `SKIP` with the reason; a fixture that is there and does not read is `FAIL`.
//!
//! `--fixture PATH` names the Nano fixture's file (the fixture root's
//! `nemotron_h` directory otherwise); `--list` prints the clause names and
//! exits; `--only CLAUSE` runs one clause; `--mutant NAME` runs the one clause
//! that must catch the named kernel defect (`mutants.rs`) against kernels
//! compiled from the edited source, and a clause that stays green on its
//! mutant is the finding; `--mutants-all` runs every mutant in a child gate of
//! its own and is red when one stays green on its clause.
//!
//! A kernel family adds its clause file, its oracle and its mutant table here:
//! a `#[path]` module line, one `.chain(...)` of its `table` in `main`, and
//! one entry in `mutants::TABLES`.

use std::process::{Command, ExitCode};

use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::HOST;
use bloomery_metal::kernels::{Kernels, MatvecParams, Mode, NormOnLoad, Quant, SegParams, Segment};
use bloomery_metal::library::{Library, SOURCE};
use bloomery_metal::stream::MetalStream;

#[path = "clauses_q5_0.rs"]
mod clauses_q5_0;
#[path = "fixture.rs"]
mod fixture;
#[path = "mutants.rs"]
mod mutants;
#[path = "../shared/oracle.rs"]
#[allow(
    dead_code,
    reason = "the clauses read the oracle's mat-vec reference and band; the other families read the rest"
)]
mod oracle;
#[path = "../shared/synthetic.rs"]
mod synthetic;

/// A clause, ready to run.
type ClauseRun<'a> = Box<dyn Fn() -> Clause + 'a>;

/// A word the buffers around a window hold, so that a write past the window
/// is seen: a quiet NaN, so a row that was never written is not a number
/// either, and an input read from outside its window poisons the output.
pub const SENTINEL_BITS: u32 = 0x7FC0_DEAD;

/// Bytes the weights' buffer holds before and after the window (an odd
/// multiple of two before, so the window's offset is 2-aligned and no
/// further, the alignment the 22-byte Q5_0 blocks are read at), poisoned with
/// 0xFF: a read through the wrong offset reads NaN scales.
const WEIGHT_PAD: usize = 66;
const WEIGHT_TAIL: usize = 258;
/// Values the activation and gain buffers hold before their windows.
const VALUE_PAD: usize = 5;
/// Words the output buffer holds on either side of its window.
const OUT_GUARD: usize = 7;

/// What a launch does with its input and output, as the gate gives it.
pub enum ModeSpec<'a> {
    Store,
    Norm {
        gain: &'a [f32],
        eps: f32,
        layer: u32,
    },
    Residual,
}

/// One mat-vec launch of the gate.
pub struct Launch<'a> {
    pub quant: Quant,
    pub rows: usize,
    pub cols: usize,
    pub weights: &'a [u8],
    pub x: &'a [f32],
    /// The output's contents before the launch (`rows` words).
    pub y0: &'a [f32],
    pub mode: ModeSpec<'a>,
}

/// One segmented mat-vec launch of the gate: the row segments of several
/// tensors of one type and one width, in one dispatch.
pub struct SegLaunch<'a> {
    pub quant: Quant,
    /// The input columns, the same for every segment.
    pub cols: usize,
    /// The segments: the rows of one tensor and the bytes of its weights.
    pub segments: Vec<(usize, &'a [u8])>,
    pub x: &'a [f32],
    /// The output's contents before the launch (every segment's rows).
    pub y0: &'a [f32],
    /// The norm's gain, one value per column.
    pub gain: &'a [f32],
    pub eps: f32,
    pub layer: u32,
}

/// The device, its stream and the kernels a clause runs.
pub struct Harness<'a> {
    pub device: &'a MetalDevice,
    pub stream: &'a MetalStream,
    pub kernels: &'a Kernels,
}

/// What a clause table closes over.
pub struct ClauseCtx<'a> {
    pub harness: &'a Harness<'a>,
    pub fixture: &'a Result<fixture::Fixture, fixture::Unavailable>,
    /// The kernel source this run compiles from: the shipped text, or the
    /// text with a mutant's edits in. A clause that compiles a kernel of its
    /// own appends to this, so its kernel sees the same edits.
    pub source: &'a str,
}

impl ClauseCtx<'_> {
    /// Run a fixture clause: skipped with the reason when the machine has no
    /// fixture, failed with the open error when the file does not read.
    fn with(&self, f: fn(&Harness<'_>, &fixture::Fixture) -> Clause) -> Clause {
        match self.fixture {
            Ok(fixture) => f(self.harness, fixture),
            Err(fixture::Unavailable::Absent(reason)) => Ok(Verdict::Skip(reason.clone())),
            Err(fixture::Unavailable::Broken(e)) => Err(e.clone()),
        }
    }
}

fn padded<T: Copy>(pad: T, before: usize, body: &[T], after: usize) -> Vec<T> {
    let mut out = vec![pad; before];
    out.extend_from_slice(body);
    out.extend(std::iter::repeat_n(pad, after));
    out
}

impl Harness<'_> {
    /// Run `launch` and read the output window back, after checking that the
    /// words around the window are as they were.
    pub fn matvec(&self, launch: &Launch<'_>) -> Result<Vec<f32>, String> {
        let (device, stream) = (self.device, self.stream);
        let sentinel = f32::from_bits(SENTINEL_BITS);
        let weights = padded(0xFFu8, WEIGHT_PAD, launch.weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let x = padded(sentinel, VALUE_PAD, launch.x, 1);
        let x = device.upload(stream, &x).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;
        let gain_buffer = match &launch.mode {
            ModeSpec::Norm { gain, .. } => {
                let gain = padded(sentinel, VALUE_PAD, gain, 1);
                Some(device.upload(stream, &gain).map_err(why)?)
            }
            _ => None,
        };

        let mode = match (&launch.mode, &gain_buffer) {
            (ModeSpec::Store, _) => Mode::Store,
            (ModeSpec::Residual, _) => Mode::Residual,
            (ModeSpec::Norm { eps, layer, .. }, Some(gain)) => Mode::NormOnLoad(NormOnLoad {
                gain: gain
                    .window(Elems(VALUE_PAD), Elems(launch.cols))
                    .map_err(why)?,
                eps: *eps,
                fault: device.fault_word(),
                layer: *layer,
            }),
            (ModeSpec::Norm { .. }, None) => unreachable!("a norm launch made its gain buffer"),
        };
        let params = MatvecParams {
            quant: launch.quant,
            rows: launch.rows,
            cols: launch.cols,
            weights: weights
                .window(Elems(WEIGHT_PAD), Elems(launch.weights.len()))
                .map_err(why)?,
            x: x.window(Elems(VALUE_PAD), Elems(launch.cols))
                .map_err(why)?,
            y: y.window_mut(Elems(OUT_GUARD), Elems(launch.rows))
                .map_err(why)?,
            mode,
        };
        self.encode(|encoder| self.kernels.matvec(encoder, params))?;
        let mut back = vec![0f32; launch.rows + 2 * OUT_GUARD];
        device
            .read(stream, y.whole().map_err(why)?, &mut back)
            .map_err(why)?;
        check_guards(&back, launch.rows)?;
        Ok(back[OUT_GUARD..OUT_GUARD + launch.rows].to_vec())
    }

    /// Run `launch` and read the output window back, after checking that the
    /// words around the window are as they were. The segments' weights share
    /// one buffer and their outputs one window, each at its own offset.
    pub fn seg(&self, launch: &SegLaunch<'_>) -> Result<Vec<f32>, String> {
        let (device, stream) = (self.device, self.stream);
        let sentinel = f32::from_bits(SENTINEL_BITS);
        let block = launch.quant.block_bytes();
        let mut weights: Vec<u8> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let mut out_offset = 0usize;
        for (rows, bytes) in &launch.segments {
            segments.push(Segment {
                cols: launch.cols,
                rows: *rows,
                weight_blocks: weights.len() / block,
                out_offset,
            });
            weights.extend_from_slice(bytes);
            out_offset += rows;
        }
        let weight_bytes = weights.len();
        let rows = out_offset;
        let weights = padded(0xFFu8, WEIGHT_PAD, &weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let x = padded(sentinel, VALUE_PAD, launch.x, 1);
        let x = device.upload(stream, &x).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;
        let gain = padded(sentinel, VALUE_PAD, launch.gain, 1);
        let gain = device.upload(stream, &gain).map_err(why)?;

        let params = SegParams {
            quant: launch.quant,
            segments: &segments,
            weights: weights
                .window(Elems(WEIGHT_PAD), Elems(weight_bytes))
                .map_err(why)?,
            x: x.window(Elems(VALUE_PAD), Elems(launch.cols))
                .map_err(why)?,
            y: y.window_mut(Elems(OUT_GUARD), Elems(rows)).map_err(why)?,
            norm: NormOnLoad {
                gain: gain
                    .window(Elems(VALUE_PAD), Elems(launch.cols))
                    .map_err(why)?,
                eps: launch.eps,
                fault: device.fault_word(),
                layer: launch.layer,
            },
        };
        self.encode(|encoder| self.kernels.matvec_seg(encoder, params))?;
        let mut back = vec![0f32; rows + 2 * OUT_GUARD];
        device
            .read(stream, y.whole().map_err(why)?, &mut back)
            .map_err(why)?;
        check_guards(&back, rows)?;
        Ok(back[OUT_GUARD..OUT_GUARD + rows].to_vec())
    }

    /// One command buffer with one dispatch encoded by `encode`, submitted
    /// whether or not `encode` refused, so a clause sees any work a launcher
    /// encoded before its refusal; the refusal is returned after the submit.
    pub fn encode(
        &self,
        encode: impl FnOnce(&mut bloomery_metal::encode::ComputeEncoder<'_>) -> Result<(), DeviceError>,
    ) -> Result<(), String> {
        let mut recording = self.stream.begin().map_err(why)?;
        let result = {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            let result = encode(&mut encoder);
            encoder.end();
            result
        };
        self.stream.submit(recording).map_err(why)?;
        result.map_err(why)
    }

    /// The fault word, once the stream's work is done; `None` when clean.
    pub fn fault(&self) -> Result<Option<bloomery_device::Fault>, String> {
        self.device.fault_word().read(self.stream).map_err(why)
    }

    /// The fault word, clean.
    pub fn clear_fault(&self) -> Result<(), String> {
        self.device.fault_word().clear(self.stream).map_err(why)
    }
}

/// The words either side of the output window are still the sentinel.
fn check_guards(back: &[f32], rows: usize) -> Result<(), String> {
    let sides = back[..OUT_GUARD].iter().chain(&back[OUT_GUARD + rows..]);
    for (i, v) in sides.enumerate() {
        if v.to_bits() != SENTINEL_BITS {
            return Err(format!(
                "a word {} the output window was written: {v:?}",
                if i < OUT_GUARD { "before" } else { "past" }
            ));
        }
    }
    Ok(())
}

/// The value after the option `name`, when it is given.
pub fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1).map(String::as_str)
}

/// The kernels compiled from `text`, and the library they came from.
fn kernels_of_text(device: &MetalDevice, text: &str) -> Result<(Library, Kernels), String> {
    let library = Library::from_source_text(device, text).map_err(why)?;
    let kernels = Kernels::from_library(&library).map_err(why)?;
    Ok((library, kernels))
}

/// Every mutant of the table, each in a child run of this gate
/// (`--mutant NAME`): its clause must be red and the child's exit code not 0.
/// One that stays green is a defect no clause catches.
fn mutants_all(args: &[String]) -> ExitCode {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            println!("gate_metal_nemotron_kernels --mutants-all: FAIL (the gate's own path: {e})");
            return ExitCode::FAILURE;
        }
    };
    let mut green = Vec::new();
    let mut count = 0usize;
    for mutant in mutants::all() {
        count += 1;
        let mut child = Command::new(&exe);
        child.args(["--mutant", mutant.name]);
        if let Some(path) = flag(args, "--fixture") {
            child.args(["--fixture", path]);
        }
        let out = match child.output() {
            Ok(out) => out,
            Err(e) => {
                println!("mutant {}: the child gate did not start: {e}", mutant.name);
                return ExitCode::FAILURE;
            }
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let red =
            !out.status.success() && text.contains(&format!("clause {}: FAIL", mutant.clause));
        println!(
            "mutant {}: {} on {}",
            mutant.name,
            if red { "red" } else { "NOT RED" },
            mutant.clause
        );
        if !red {
            green.push(mutant.name);
        }
    }
    if green.is_empty() {
        println!(
            "gate_metal_nemotron_kernels --mutants-all: PASS ({count} mutants, each red on its clause)"
        );
        return ExitCode::SUCCESS;
    }
    println!(
        "gate_metal_nemotron_kernels --mutants-all: FAIL (not red: {})",
        green.join(", ")
    );
    ExitCode::FAILURE
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(name) = ["--fixture", "--only", "--mutant"]
        .into_iter()
        .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("gate_metal_nemotron_kernels: {name} takes a value");
        return ExitCode::from(64);
    }
    if args.iter().any(|a| a == "--mutants-all") {
        return mutants_all(&args);
    }
    let device = match MetalDevice::open(&HOST) {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("clause open: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let stream = device.stream();
    // The simdgroup GEMMs, on every device: no clause here runs a GEMM.
    let simdgroup = device.library().with_simdgroup_gemms();
    let kernels = match Kernels::from_library(&simdgroup) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "metal-lib: path={} device={} backend={}",
        device.library().path().name(),
        info.name,
        <MetalDevice as Device>::BACKEND
    );
    let mutant = match flag(&args, "--mutant") {
        Some(name) => match mutants::find(name) {
            Some(mutant) => Some(mutant),
            None => {
                println!("gate_metal_nemotron_kernels: no mutant named {name}");
                return ExitCode::from(64);
            }
        },
        None => None,
    };
    let (run_source, mutated) = match mutant {
        Some(mutant) => match mutants::source_of(mutant) {
            Ok(text) => match kernels_of_text(&device, &text) {
                Ok(mutated) => (text, Some(mutated)),
                Err(e) => {
                    println!("gate_metal_nemotron_kernels: the mutant does not build: {e}");
                    return ExitCode::from(64);
                }
            },
            Err(e) => {
                println!("gate_metal_nemotron_kernels: the mutant does not build: {e}");
                return ExitCode::from(64);
            }
        },
        None => (SOURCE.to_string(), None),
    };
    let harness = Harness {
        device: &device,
        stream,
        kernels: mutated.as_ref().map_or(&kernels, |(_, k)| k),
    };
    if let Some(mutant) = mutant {
        println!("mutant: {} ({})", mutant.name, mutant.why);
    }
    let only = mutant.map(|m| m.clause).or(flag(&args, "--only"));
    let fixture = fixture::open(&args);
    match &fixture {
        Ok(f) => println!("fixture: {}", f.path.display()),
        Err(fixture::Unavailable::Absent(reason)) => println!("fixture: none ({reason})"),
        Err(fixture::Unavailable::Broken(e)) => println!("fixture: {e}"),
    }

    let mut report = Report::new();
    let ctx = ClauseCtx {
        harness: &harness,
        fixture: &fixture,
        source: &run_source,
    };
    // The families in the order their kernels land. A new family adds one
    // line.
    let clauses: Vec<(&'static str, ClauseRun)> = std::iter::empty()
        .chain(clauses_q5_0::table(&ctx))
        .collect();
    if args.iter().any(|a| a == "--list") {
        for (name, _) in &clauses {
            println!("{name}");
        }
        return ExitCode::SUCCESS;
    }
    if let Some(name) = only
        && !clauses.iter().any(|(n, _)| *n == name)
    {
        println!("gate_metal_nemotron_kernels: no clause named {name}");
        return ExitCode::from(64);
    }
    for (name, run) in &clauses {
        if only.is_none_or(|o| o == *name) {
            report.clause(name, run());
        }
    }

    let scope = match (mutant, only) {
        (Some(mutant), _) => format!(" (mutant {})", mutant.name),
        (None, Some(name)) => format!(" (only {name})"),
        (None, None) => String::new(),
    };
    report.finish(&format!("gate_metal_nemotron_kernels{scope}"))
}
