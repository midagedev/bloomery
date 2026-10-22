//! The qwen35moe kernels' gate clauses (macOS only).
//!
//! One clause per line, `PASS`, `SKIP` or `FAIL`, and the run's exit code is
//! 1 when any clause failed, 78 when this machine cannot run the backend (a
//! macOS below the floor, no Metal device), 0 otherwise. The weights are the
//! fixture's own tensors; the oracle is `crates/gguf`'s dequantization of the
//! same blocks with an f64 dot, held to the band `oracle.rs` derives. The
//! clauses:
//!
//! 1. **q4k_matvec**: the Q4_K mat-vec at K = 2048 (`blk.0.attn_qkv`), 4096
//!    (`blk.0.ssm_out`) and 512 (`blk.0.ffn_down_shexp`).
//! 2. **q6k_matvec**: the Q6_K mat-vec on `blk.2.attn_qkv`,
//!    `blk.2.ffn_down_shexp`, `blk.3.attn_v` and rows of `output.weight`.
//! 3. **norm_on_load**: the mat-vec of an RMS-normed input is the host's
//!    `rms_norm` then the mat-vec, and an input the norm cannot take raises
//!    the norm site and writes NaN rows.
//! 4. **residual_epilogue**: `y = y + W . x` in place.
//! 5. **row_tail**: a row count that is not a multiple of the rows of a
//!    threadgroup writes every row and nothing past the last.
//! 6. **row_tail_reads**: those launches read nothing outside their weights'
//!    buffer, as Metal's shader validation sees them in a child run of this
//!    gate (a read nothing stores shows in no output).
//! 7. **metallib_equals_source**: the metallib's outputs and the source's are
//!    the same bits, when both paths exist.
//! 8. **embed**: an id at or past the vocabulary raises the token-id site, and
//!    its row is NaN.
//! 9. **matvec_refusals**: a window that is not the shape's, columns that are
//!    not whole blocks and an empty shape are refused by name, and nothing is
//!    encoded for them.
//! 10. **block_shape**: the block bytes and values the launch arithmetic
//!     uses are gguf's own for Q4_K and Q6_K.
//!
//! `--fixture PATH` names the fixture's trunk file (the Mac's cache copy
//! otherwise); `--only CLAUSE` runs one clause; `--mutant NAME` runs the one
//! clause that must catch the named kernel defect (`mutants.rs`) against
//! kernels compiled from the edited source, and a clause that stays green on
//! its mutant is the finding; `--mutants-all` runs every mutant in a child
//! gate of its own and is red when one stays green on its clause (a mutant
//! that no clause catches is a test that cannot fail); `--child CLAUSE` is
//! `--only` for a run whose output a parent gate reads (`row_tail_reads`).

use std::path::PathBuf;
use std::process::{Command, ExitCode};

use bloomery_device::conformance::{Clause, Report, why};
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{Kernels, MatvecParams, Mode, NormOnLoad, Quant};
use bloomery_metal::library::Library;
use bloomery_metal::stream::MetalStream;

#[path = "clauses.rs"]
mod clauses;
#[path = "fixture.rs"]
mod fixture;
#[path = "mutants.rs"]
mod mutants;
#[path = "../shared/oracle.rs"]
mod oracle;

/// A clause, ready to run.
type ClauseRun<'a> = Box<dyn Fn() -> Clause + 'a>;

/// A word the buffers around a window hold, so that a write past the window
/// is seen: a quiet NaN, so a row that was never written is not a number
/// either, and an input read from outside its window poisons the output.
pub const SENTINEL_BITS: u32 = 0x7FC0_DEAD;

/// Bytes the weights' buffer holds before and after the window (odd multiples
/// of two, so the window's offset is neither zero nor aligned to anything
/// larger than the kernel needs), poisoned with 0xFF: a read through the
/// wrong offset reads NaN scales.
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

/// The device, its stream and the kernels a clause runs.
pub struct Harness<'a> {
    pub device: &'a MetalDevice,
    pub stream: &'a MetalStream,
    pub kernels: &'a Kernels,
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

    /// One command buffer with one dispatch encoded by `encode`, submitted.
    pub fn encode(
        &self,
        encode: impl FnOnce(&mut bloomery_metal::encode::ComputeEncoder<'_>) -> Result<(), DeviceError>,
    ) -> Result<(), String> {
        let mut recording = self.stream.begin().map_err(why)?;
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            encode(&mut encoder).map_err(why)?;
            encoder.end();
        }
        self.stream.submit(recording).map_err(why)
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

/// The kernels compiled from the source with the mutant's edits in.
fn mutated_kernels(device: &MetalDevice, mutant: &mutants::Mutant) -> Result<Kernels, String> {
    let text = mutants::source_of(mutant)?;
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

/// The fixture path: `--fixture PATH`, else the Mac's cache copy; with no
/// `HOME` to find it under, refused by name.
fn fixture_path(args: &[String]) -> Result<PathBuf, String> {
    if let Some(path) = flag(args, "--fixture") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .ok_or("HOME is not set: name the fixture's trunk file with --fixture PATH")?;
    Ok(PathBuf::from(home)
        .join(".cache/bloomery/mac-port/fixture/qwen35moe-fixture-00001-of-00001.gguf"))
}

/// Every mutant of the table, each in a child run of this gate
/// (`--mutant NAME`): its clause must be red and the child's exit code not 0.
/// One that stays green is a defect no clause catches.
fn mutants_all(args: &[String]) -> ExitCode {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            println!("gate_metal_qwen35moe_kernels --mutants-all: FAIL (the gate's own path: {e})");
            return ExitCode::FAILURE;
        }
    };
    let mut green = Vec::new();
    for mutant in mutants::MUTANTS {
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
            "gate_metal_qwen35moe_kernels --mutants-all: PASS ({} mutants, each red on its clause)",
            mutants::MUTANTS.len()
        );
        return ExitCode::SUCCESS;
    }
    println!(
        "gate_metal_qwen35moe_kernels --mutants-all: FAIL (not red: {})",
        green.join(", ")
    );
    ExitCode::FAILURE
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(name) = ["--fixture", "--only", "--mutant", "--child"]
        .into_iter()
        .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("gate_metal_qwen35moe_kernels: {name} takes a value");
        return ExitCode::from(64);
    }
    if args.iter().any(|a| a == "--mutants-all") {
        return mutants_all(&args);
    }
    let device = match MetalDevice::open() {
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
    let kernels = match Kernels::new(&device) {
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
                println!("gate_metal_qwen35moe_kernels: no mutant named {name}");
                return ExitCode::from(64);
            }
        },
        None => None,
    };
    let mutated = match mutant.map(|m| mutated_kernels(&device, m)).transpose() {
        Ok(mutated) => mutated,
        Err(e) => {
            println!("gate_metal_qwen35moe_kernels: the mutant does not build: {e}");
            return ExitCode::from(64);
        }
    };
    let harness = Harness {
        device: &device,
        stream,
        kernels: mutated.as_ref().unwrap_or(&kernels),
    };
    if let Some(mutant) = mutant {
        println!("mutant: {} ({})", mutant.name, mutant.why);
    }
    let only = flag(&args, "--child")
        .or(mutant.map(|m| m.clause))
        .or(flag(&args, "--only"));
    let fixture = fixture_path(&args).and_then(|path| fixture::Fixture::open(&path));
    match &fixture {
        Ok(f) => println!("fixture: {}", f.path.display()),
        Err(e) => println!("fixture: {e}"),
    }

    let mut report = Report::new();
    let with = |f: fn(&Harness<'_>, &fixture::Fixture) -> Clause| match &fixture {
        Ok(fixture) => f(&harness, fixture),
        Err(e) => Err(e.clone()),
    };
    let clauses: [(&'static str, ClauseRun<'_>); 10] = [
        ("q4k_matvec", Box::new(|| with(clauses::q4k_matvec))),
        ("q6k_matvec", Box::new(|| with(clauses::q6k_matvec))),
        ("norm_on_load", Box::new(|| with(clauses::norm_on_load))),
        (
            "residual_epilogue",
            Box::new(|| with(clauses::residual_epilogue)),
        ),
        ("row_tail", Box::new(|| with(clauses::row_tail))),
        (
            "row_tail_reads",
            Box::new(|| clauses::row_tail_reads(&args)),
        ),
        (
            "metallib_equals_source",
            Box::new(|| match &fixture {
                Ok(fixture) => clauses::metallib_equals_source(&device, stream, fixture),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("embed", Box::new(|| with(clauses::embed))),
        (
            "matvec_refusals",
            Box::new(|| clauses::matvec_refusals(&harness)),
        ),
        ("block_shape", Box::new(clauses::block_shape)),
    ];
    if let Some(name) = only
        && !clauses.iter().any(|(n, _)| *n == name)
    {
        println!("gate_metal_qwen35moe_kernels: no clause named {name}");
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
    report.finish(&format!("gate_metal_qwen35moe_kernels{scope}"))
}
