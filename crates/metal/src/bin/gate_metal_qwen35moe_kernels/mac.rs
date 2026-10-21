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
//! 6. **metallib_equals_source**: the metallib's outputs and the source's are
//!    the same bits, when both paths exist.
//! 7. **embed**: an id at or past the vocabulary raises the token-id site, and
//!    its row is NaN.
//! 8. **matvec_refusals**: a window that is not the shape's, columns that are
//!    not whole blocks and an empty shape are refused by name, and nothing is
//!    encoded for them.
//!
//! `--fixture PATH` names the fixture's trunk file (the Mac's cache copy
//! otherwise); `--only CLAUSE` runs one clause; `--mutant NAME` runs the one
//! clause that must catch the named kernel defect (`mutants.rs`) against
//! kernels compiled from the edited source, and a clause that stays green on
//! its mutant is the finding.

use std::path::PathBuf;
use std::process::ExitCode;

use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{Kernels, MatvecParams, Mode, NormOnLoad, Quant};
use bloomery_metal::library::Library;
use bloomery_metal::stream::MetalStream;

#[path = "clauses.rs"]
mod clauses;
#[path = "mutants.rs"]
mod mutants;
#[path = "oracle.rs"]
mod oracle;

/// How a clause that did not fail ended.
pub enum Verdict {
    /// Passed, with a fact worth printing.
    Noted(String),
    /// Could not run here, with the reason.
    Skip(String),
}

/// A clause's result: a verdict, or the reason it failed.
pub type Clause = Result<Verdict, String>;

/// A clause, ready to run.
type ClauseRun<'a> = Box<dyn Fn() -> Clause + 'a>;

/// The gate's own report: one line per clause.
struct Report {
    failures: Vec<&'static str>,
}

impl Report {
    fn clause(&mut self, name: &'static str, result: Clause) {
        match result {
            Ok(Verdict::Noted(note)) => println!("clause {name}: PASS ({note})"),
            Ok(Verdict::Skip(why)) => println!("clause {name}: SKIP ({why})"),
            Err(why) => {
                println!("clause {name}: FAIL ({why})");
                self.failures.push(name);
            }
        }
    }
}

/// The error of a device call, as a clause's text.
pub fn why(e: DeviceError) -> String {
    e.to_string()
}

/// A word the buffers around a window hold, so that a write past the window
/// is seen: a NaN, so a row that was never written is not a number either.
pub const SENTINEL_BITS: u32 = 0xDEAD_BEEF;

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
        y.read(stream, &mut back).map_err(why)?;
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
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1).map(String::as_str)
}

/// The kernels compiled from the source with the mutant's edits in.
fn mutated_kernels(device: &MetalDevice, mutant: &mutants::Mutant) -> Result<Kernels, String> {
    let text = mutants::source_of(mutant)?;
    let library = Library::from_source_text(device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

/// The fixture path: `--fixture PATH`, else the Mac's cache copy.
fn fixture_path(args: &[String]) -> PathBuf {
    if let Some(path) = flag(args, "--fixture") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".cache/bloomery/mac-port/fixture/qwen35moe-fixture-00001-of-00001.gguf")
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let stream = match device.stream() {
        Ok(stream) => stream,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let kernels = match Kernels::new(&device) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "metal-lib: path={} device={} backend={}",
        device.library().path().name(),
        device.info().name,
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
        stream: &stream,
        kernels: mutated.as_ref().unwrap_or(&kernels),
    };
    if let Some(mutant) = mutant {
        println!("mutant: {} ({})", mutant.name, mutant.why);
    }
    let only = mutant.map(|m| m.clause).or(flag(&args, "--only"));
    let path = fixture_path(&args);
    let fixture = oracle::Fixture::open(&path);
    match &fixture {
        Ok(f) => println!("fixture: {}", f.path.display()),
        Err(e) => println!("fixture: {e}"),
    }

    let mut report = Report {
        failures: Vec::new(),
    };
    let with = |f: fn(&Harness<'_>, &oracle::Fixture) -> Clause| match &fixture {
        Ok(fixture) => f(&harness, fixture),
        Err(e) => Err(e.clone()),
    };
    let clauses: [(&'static str, ClauseRun<'_>); 8] = [
        ("q4k_matvec", Box::new(|| with(clauses::q4k_matvec))),
        ("q6k_matvec", Box::new(|| with(clauses::q6k_matvec))),
        ("norm_on_load", Box::new(|| with(clauses::norm_on_load))),
        (
            "residual_epilogue",
            Box::new(|| with(clauses::residual_epilogue)),
        ),
        ("row_tail", Box::new(|| with(clauses::row_tail))),
        (
            "metallib_equals_source",
            Box::new(|| match &fixture {
                Ok(fixture) => clauses::metallib_equals_source(&device, &stream, fixture),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("embed", Box::new(|| with(clauses::embed))),
        (
            "matvec_refusals",
            Box::new(|| clauses::matvec_refusals(&harness)),
        ),
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
    if report.failures.is_empty() {
        println!("gate_metal_qwen35moe_kernels: PASS{scope}");
        return ExitCode::SUCCESS;
    }
    println!(
        "gate_metal_qwen35moe_kernels: FAIL{scope} ({})",
        report.failures.join(", ")
    );
    ExitCode::FAILURE
}
