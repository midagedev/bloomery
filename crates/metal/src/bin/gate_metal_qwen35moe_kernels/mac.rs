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
//! 11. **seg_bit**: every segment of a segmented dispatch is the same bits as
//!     the same tensor run alone: the delta layer's Q4_K group, the attention
//!     layer's group, and the mixed-type layer's Q6_K dispatch.
//! 12. **seg_bounds**: a segment list with mismatched widths, a row count of
//!     zero, more segments than a dispatch runs, or overlapping outputs is
//!     refused by name, and nothing is encoded for it.
//! 13. **headnorm**: the head-normed residual mat-vec against the f64 oracle,
//!     the gated numerator and the heads' partial sums as the inputs.
//! 14. **headnorm_nan**: a partial or a gated numerator that is not finite
//!     raises the gate site and writes NaN.
//! 15. **nsg**: the Q6_K mat-vec at eight SIMD groups in a threadgroup is the
//!     same bits as at two, for every row count the fixture's Q6_K tensors
//!     have.
//! 16. **router_logit**: the router's logits sit inside the f64 band. // K2-REGISTER
//! 17. **router_ids**: the routed ids match the f64 ranking when the 8th-vs-9th margin is wider than the band, and a step inside the band is printed by name. // K2-REGISTER
//! 18. **router_weights**: the top-8 weights and the shared-expert sigmoid match the f64 rule. // K2-REGISTER
//! 19. **router_ticket**: 10000 launches, and each launch's ids are the host's selection from that launch's logits. // K2-REGISTER
//! 20. **router_nan**: a non-finite weight raises the router site and the ids stay in range. // K2-REGISTER
//! 21. **router_bounds**: a shape, a window, an eps or an alignment the kernel does not take is refused by name, and nothing is encoded. // K2-REGISTER
//! 22. **pair**: silu(gate) times up, against the f64 oracle. // K3-REGISTER
//! 23. **pair_id**: an id past the stack raises expert_id and writes NaN. // K3-REGISTER
//! 24. **pair_shared**: slot 8 reads the shared expert, the others the stack. // K3-REGISTER
//! 25. **fold**: the folded down plus the residual, against the f64 oracle. // K3-REGISTER
//! 26. **fold_type**: a shared expert of another quant is refused by name. // K3-REGISTER
//! 27. **experts_bounds**: a launch that is not the shape is refused by name. // K3-REGISTER
//! 28. **delta_step**: one gated delta step from a host state, against f64. // K4-REGISTER
//! 29. **delta_roll**: five steps, each fed by the kernel's own state. // K4-REGISTER
//! 30. **delta_ring**: a predecessor is read only while the position reaches it, and one ring slot is written. // K4-REGISTER
//! 31. **delta_partial**: the partials and the numerator, accepted by the head-norm launch. // K4-REGISTER
//! 32. **delta_nan**: a non-finite lane raises the lane site and writes NaN on that lane. // K4-REGISTER
//! 33. **delta_bounds**: a shape, a position window and a numerator window are refused by name. // K4-REGISTER
//! 34. **attn-f16**: f16 attention and the appended cache match f64 attention, at S = 1 and S = 4. // K5-REGISTER
//! 35. **attn-q8**: the appended q8_0 row matches `gguf::quant::quantize_q8_0`, and attention over it is inside the band. // K5-REGISTER
//! 36. **attn-s**: S = 1 and S = 4 agree inside the band. // K5-REGISTER
//! 37. **attn-reread**: the re-fetch arm matches the tile arm inside the band. // K5-REGISTER
//! 38. **attn-m**: two query positions match the oracle. // K5-REGISTER
//! 39. **attn-fault**: KeyCount, CachePos and KvQuant. // K5-REGISTER
//! 40. **attn-bounds**: a shape or a window the kernel does not take is refused by name. // K5-REGISTER
//! 41. **head_logit**: sample rows of `output.weight` sit inside the Q6_K norm-on-load band. // K6-REGISTER
//! 42. **head_id**: a wide margin on the shipped grid names the f64 argmax, and a bit tie names the smaller index. // K6-REGISTER
//! 43. **head_ticket**: 1000 launches, and each launch's id and candidates are that launch's logits. // K6-REGISTER
//! 44. **head_nan**: a NaN block scale stays NaN, raises the logit site, and the id is the finite argmax. // K6-REGISTER
//! 45. **head_bounds**: a shape, a window, an eps or a SIMD-group count the kernel does not take is refused by name, and nothing is encoded. // K6-REGISTER
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
use bloomery_metal::kernels::{
    HeadNormParams, Kernels, MatvecParams, Mode, NormOnLoad, Quant, SegParams, Segment,
};
use bloomery_metal::library::Library;
use bloomery_metal::stream::MetalStream;

#[path = "clauses.rs"]
mod clauses;
#[path = "clauses_attention.rs"] // K5-REGISTER
mod clauses_attention; // K5-REGISTER
#[path = "clauses_delta.rs"] // K4-REGISTER
mod clauses_delta; // K4-REGISTER
#[path = "clauses_experts.rs"] // K3-REGISTER
mod clauses_experts; // K3-REGISTER
#[path = "clauses_head.rs"] // K6-REGISTER
mod clauses_head; // K6-REGISTER
#[path = "clauses_router.rs"] // K2-REGISTER
mod clauses_router; // K2-REGISTER
#[path = "fixture.rs"]
mod fixture;
#[path = "mutants.rs"]
mod mutants;
#[path = "../shared/oracle.rs"]
mod oracle;
#[path = "../shared/oracle_attention.rs"] // K5-REGISTER
mod oracle_attention;
#[path = "../shared/oracle_delta.rs"] // K4-REGISTER
mod oracle_delta; // K4-REGISTER
#[path = "../shared/oracle_experts.rs"] // K3-REGISTER
mod oracle_experts; // K3-REGISTER
#[path = "../shared/oracle_router.rs"] // K2-REGISTER
mod oracle_router; // K2-REGISTER // K5-REGISTER

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

/// One head-normed residual mat-vec launch of the gate.
pub struct HeadNormLaunch<'a> {
    pub rows: usize,
    pub cols: usize,
    pub weights: &'a [u8],
    /// The gated numerator, one value per column.
    pub g: &'a [f32],
    /// The partial sums of the squares, one row per head.
    pub partials: &'a [f32],
    /// The output's contents before the launch (`rows` words).
    pub y0: &'a [f32],
    pub eps: f32,
    pub layer: u32,
}

/// The device, its stream and the kernels a clause runs.
pub struct Harness<'a> {
    pub device: &'a MetalDevice,
    pub stream: &'a MetalStream,
    pub kernels: &'a Kernels,
    /// The library the kernels came from: a clause that runs the same kernels
    /// with another geometry builds them again from it.
    pub library: &'a Library,
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
            segments,
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

    /// Run `launch` and read the output window back, after checking that the
    /// words around the window are as they were.
    pub fn headnorm(&self, launch: &HeadNormLaunch<'_>) -> Result<Vec<f32>, String> {
        let (device, stream) = (self.device, self.stream);
        let sentinel = f32::from_bits(SENTINEL_BITS);
        let weights = padded(0xFFu8, WEIGHT_PAD, launch.weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let g = padded(sentinel, VALUE_PAD, launch.g, 1);
        let g = device.upload(stream, &g).map_err(why)?;
        let partials = padded(sentinel, VALUE_PAD, launch.partials, 1);
        let partials = device.upload(stream, &partials).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;

        let params = HeadNormParams {
            rows: launch.rows,
            cols: launch.cols,
            weights: weights
                .window(Elems(WEIGHT_PAD), Elems(launch.weights.len()))
                .map_err(why)?,
            g: g.window(Elems(VALUE_PAD), Elems(launch.cols))
                .map_err(why)?,
            partials: partials
                .window(Elems(VALUE_PAD), Elems(launch.partials.len()))
                .map_err(why)?,
            y: y.window_mut(Elems(OUT_GUARD), Elems(launch.rows))
                .map_err(why)?,
            eps: launch.eps,
            fault: device.fault_word(),
            layer: launch.layer,
        };
        self.encode(|encoder| self.kernels.headnorm_resid(encoder, params))?;
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

/// The kernels compiled from the source with the mutant's edits in, and the
/// library they came from.
fn mutated_kernels(
    device: &MetalDevice,
    mutant: &mutants::Mutant,
) -> Result<(Library, Kernels), String> {
    let text = mutants::source_of(mutant)?;
    let library = Library::from_source_text(device, &text).map_err(why)?;
    let kernels = Kernels::from_library(&library).map_err(why)?;
    Ok((library, kernels))
}

/// The fixture path: `--fixture PATH`, else the qwen35moe fixture under the
/// fixture root (`BLOOMERY_FIXTURE_ROOT`), as the reference sets resolve it.
fn fixture_path(args: &[String]) -> Result<PathBuf, String> {
    if let Some(path) = flag(args, "--fixture") {
        return Ok(PathBuf::from(path));
    }
    refset::fixture::first_shard("qwen35moe")
        .map(PathBuf::from)
        .map_err(|e| format!("{e} (name the fixture's trunk file with --fixture PATH)"))
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
    let mutated = match mutant {
        Some(mutant) => match mutated_kernels(&device, mutant) {
            Ok(mutated) => Some(mutated),
            Err(e) => {
                println!("gate_metal_qwen35moe_kernels: the mutant does not build: {e}");
                return ExitCode::from(64);
            }
        },
        None => None,
    };
    let harness = Harness {
        device: &device,
        stream,
        kernels: mutated.as_ref().map_or(&kernels, |(_, k)| k),
        library: mutated.as_ref().map_or(device.library(), |(l, _)| l),
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
    #[rustfmt::skip] // K2-REGISTER
    let clauses: [(&'static str, ClauseRun<'_>); 45] = [ // K2-REGISTER
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
        ("seg_bit", Box::new(|| with(clauses::seg_bit))),
        ("seg_bounds", Box::new(|| with(clauses::seg_bounds))),
        ("headnorm", Box::new(|| with(clauses::headnorm))),
        ("headnorm_nan", Box::new(|| with(clauses::headnorm_nan))),
        ("nsg", Box::new(|| with(clauses::nsg))),
        ("router_logit", Box::new(|| with(clauses_router::router_logit))), // K2-REGISTER
        ("router_ids", Box::new(|| with(clauses_router::router_ids))), // K2-REGISTER
        ("router_weights", Box::new(|| with(clauses_router::router_weights))), // K2-REGISTER
        ("router_ticket", Box::new(|| with(clauses_router::router_ticket))), // K2-REGISTER
        ("router_nan", Box::new(|| with(clauses_router::router_nan))), // K2-REGISTER
        ("router_bounds", Box::new(|| clauses_router::router_bounds(&harness))), // K2-REGISTER
        ("pair", Box::new(|| with(clauses_experts::pair))), // K3-REGISTER
        ("pair_id", Box::new(|| with(clauses_experts::pair_id))), // K3-REGISTER
        ("pair_shared", Box::new(|| with(clauses_experts::pair_shared))), // K3-REGISTER
        ("fold", Box::new(|| with(clauses_experts::fold))), // K3-REGISTER
        ("fold_type", Box::new(|| with(clauses_experts::fold_type))), // K3-REGISTER
        ("experts_bounds", Box::new(|| with(clauses_experts::experts_bounds))), // K3-REGISTER
        ("delta_step", Box::new(|| with(clauses_delta::delta_step))), // K4-REGISTER
        ("delta_roll", Box::new(|| with(clauses_delta::delta_roll))), // K4-REGISTER
        ("delta_ring", Box::new(|| with(clauses_delta::delta_ring))), // K4-REGISTER
        ("delta_partial", Box::new(|| with(clauses_delta::delta_partial))), // K4-REGISTER
        ("delta_nan", Box::new(|| with(clauses_delta::delta_nan))), // K4-REGISTER
        ("delta_bounds", Box::new(|| clauses_delta::delta_bounds(&harness))), // K4-REGISTER
        ("attn-f16", Box::new(|| with(clauses_attention::attn_f16))), // K5-REGISTER
        ("attn-q8", Box::new(|| with(clauses_attention::attn_q8))), // K5-REGISTER
        ("attn-s", Box::new(|| with(clauses_attention::attn_s))), // K5-REGISTER
        ("attn-reread", Box::new(|| with(clauses_attention::attn_reread))), // K5-REGISTER
        ("attn-m", Box::new(|| with(clauses_attention::attn_m))), // K5-REGISTER
        ("attn-fault", Box::new(|| with(clauses_attention::attn_fault))), // K5-REGISTER
        ("attn-bounds", Box::new(|| with(clauses_attention::attn_bounds))), // K5-REGISTER
        ("head_logit", Box::new(|| with(clauses_head::head_logit))), // K6-REGISTER
        ("head_id", Box::new(|| with(clauses_head::head_id))), // K6-REGISTER
        ("head_ticket", Box::new(|| with(clauses_head::head_ticket))), // K6-REGISTER
        ("head_nan", Box::new(|| with(clauses_head::head_nan))), // K6-REGISTER
        ("head_bounds", Box::new(|| clauses_head::head_bounds(&harness))), // K6-REGISTER
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
