//! The qwen35moe kernels' gate clauses (macOS only).
//!
//! One clause per line, `PASS`, `SKIP` or `FAIL`, and the run's exit code is
//! 1 when any clause failed, 78 when this machine cannot run the backend (a
//! macOS below the floor, no Metal device), 0 otherwise. The weights are the
//! fixture's own tensors; the oracle is `crates/gguf`'s dequantization of the
//! same blocks with an f64 dot, held to the band `oracle.rs` derives.
//! `--list` prints the clause names in the order the gate runs them: the
//! mat-vec clauses, then the A4 ones and the requantized A8, A4H and A8H
//! ones, then the token's kernels
//! (delta, router, experts, attention, head), then the prompt pass's, then
//! the tensor-op GEMMs' (`clauses_tensor.rs`).
//!
//! The GEMM clauses outside `clauses_tensor.rs` run the simdgroup GEMMs on
//! every device; the `tensor_*` clauses run the tensor-op GEMMs on every
//! device that builds them. The header names the path the device's own rule
//! picks for the engine.
//!
//! The gate has one row per registered instance ([`INSTANCE_ROWS`]): with no
//! flag it runs every row in order, each on a device opened at its own
//! instance, dropped before the next row opens. A row's clauses size every
//! launch from the geometry its kernels hold (`Harness::geom`), never from a
//! build's host.
//!
//! `--instance NAME` runs one row (an unknown name is refused by name, with
//! the rows); `--fixture PATH` names the fixture's trunk file (the Mac's
//! cache copy of the row's family otherwise) and runs the row whose instance
//! the file selects; `--mlx-fixture PATH` names the MLX-format fixture split
//! the A4 clauses read (the row's `<family>-mlxsplit` copy under the fixture
//! root otherwise: `qwen35moe-mlxsplit`, `qwen3moe-mlxsplit`); `--list`
//! prints the clause names and exits; `--only CLAUSE` runs one clause, in the
//! rows whose table has it; `--mutant NAME` runs the
//! one clause that must catch the named kernel defect (`mutants.rs`) against
//! kernels compiled from the edited source of its row, and a clause that stays
//! green on its mutant is the finding; `--mutants-all` runs every mutant of
//! every row in a child gate of its own and is red when one stays green on its
//! clause (a mutant that no clause catches is a test that cannot fail);
//! `--child CLAUSE` is `--only` for a run whose output a parent gate reads
//! (`row_tail_reads`).

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_device::{Buffer, Device, DeviceError, Elems, FaultWord};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::instance::{
    self, Instance, LFM3_ATTN, LLAMA_ATTN, QWEN3MOE_A3B, QWEN35MOE_A3B, QWEN35MOE_GDN48,
    QWEN35MOE_GQA4, QWEN35MOE_GQA6, REGISTERED,
};
use bloomery_metal::kernels::{
    Geometry, HeadNormParams, Kernels, LoadNorm, MatvecParams, Mode, NormOnLoad, Quant,
    SIMD_GROUPS_PER_GROUP, SegParams, Segment,
};
use bloomery_metal::library::{self, Library};
use bloomery_metal::stream::MetalStream;
use gguf::Split;

#[path = "clauses_a4.rs"]
mod clauses_a4;
#[path = "clauses_apart.rs"]
mod clauses_apart;
#[path = "clauses_attention.rs"]
mod clauses_attention;
#[path = "clauses_delta.rs"]
mod clauses_delta;
#[path = "clauses_embed.rs"]
mod clauses_embed;
#[path = "clauses_experts.rs"]
mod clauses_experts;
#[path = "clauses_f32.rs"]
mod clauses_f32;
#[path = "clauses_gemm.rs"]
mod clauses_gemm;
#[path = "clauses_handon.rs"]
mod clauses_handon;
#[path = "clauses_head.rs"]
mod clauses_head;
#[path = "clauses_matvec.rs"]
mod clauses_matvec;
#[path = "clauses_norm.rs"]
mod clauses_norm;
#[path = "clauses_q5k.rs"]
mod clauses_q5k;
#[path = "clauses_q8_0.rs"]
mod clauses_q8_0;
#[path = "clauses_requant.rs"]
mod clauses_requant;
#[path = "clauses_route.rs"]
mod clauses_route;
#[path = "clauses_router.rs"]
mod clauses_router;
#[path = "clauses_rows.rs"]
mod clauses_rows;
#[path = "clauses_seg.rs"]
mod clauses_seg;
#[path = "clauses_shortconv.rs"]
mod clauses_shortconv;
#[path = "clauses_sites.rs"]
mod clauses_sites;
#[path = "clauses_synth.rs"]
mod clauses_synth;
#[path = "clauses_tensor.rs"]
mod clauses_tensor;
#[path = "clauses_window.rs"]
mod clauses_window;
#[path = "common.rs"]
mod common;
#[path = "fixture.rs"]
mod fixture;
#[path = "../shared/mlx_split.rs"]
mod mlx_split;
#[path = "mutants.rs"]
mod mutants;
#[path = "../shared/oracle.rs"]
mod oracle;
#[path = "../shared/oracle_attention.rs"]
mod oracle_attention;
#[path = "../shared/oracle_delta.rs"]
mod oracle_delta;
#[path = "../shared/oracle_experts.rs"]
mod oracle_experts;
#[path = "../shared/oracle_router.rs"]
mod oracle_router;
#[path = "../shared/oracle_shortconv.rs"]
mod oracle_shortconv;
#[path = "../shared/synthetic.rs"]
mod synthetic;

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
/// The weights' pad for a type whose kernel loads `float4`: the next
/// 16-byte boundary past the same odd-looking offset.
const FLOAT4_WEIGHT_PAD: usize = 80;
/// The weights' pad for a type whose kernels load 32-bit words: the next
/// 4-byte boundary past the same odd-looking offset.
const WORD_WEIGHT_PAD: usize = 68;

/// The weights' pad a launch of `quant` takes: the least aligned offset past
/// the guard its kernels accept, so a read wider than the type's word reads
/// the poisoned pad or is refused.
pub fn weight_pad(quant: Quant) -> usize {
    match quant {
        Quant::F32 => FLOAT4_WEIGHT_PAD,
        Quant::A4 | Quant::A8 | Quant::A4H | Quant::A8H => WORD_WEIGHT_PAD,
        Quant::Q4K | Quant::Q6K | Quant::Q8_0 | Quant::Q5K => WEIGHT_PAD,
    }
}
const WEIGHT_TAIL: usize = 258;
/// Values the activation and gain buffers hold before their windows.
const VALUE_PAD: usize = 5;
/// The same for the kernels that read the activations and the gain as
/// `float4`s: a multiple of four, so the window starts on 16 bytes.
const ACT_PAD: usize = 8;
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
    pub quant: Quant,
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

/// What a clause table closes over.
pub struct ClauseCtx<'a> {
    pub harness: &'a Harness<'a>,
    pub fixture: &'a Result<fixture::Fixture, String>,
    pub args: &'a [String],
    /// The kernel source this run compiles from: the shipped text, or the
    /// text with a mutant's edits in. A clause that compiles a library of its
    /// own appends to this, so its kernel sees the same edits.
    pub source: &'a str,
}

impl<'a> ClauseCtx<'a> {
    /// Run a fixture clause, or the fixture's open error when there is none.
    fn with(&self, f: fn(&Harness<'_>, &fixture::Fixture) -> Clause) -> Clause {
        match self.fixture {
            Ok(fixture) => f(self.harness, fixture),
            Err(e) => Err(e.clone()),
        }
    }

    /// Run a delta clause on the fixture's parameters, or the fixture's open
    /// error when there is none.
    fn with_params(&self, f: fn(&Harness<'_>, clauses_delta::Params<'_>) -> Clause) -> Clause {
        match self.fixture {
            Ok(fixture) => f(self.harness, clauses_delta::Params::Fixture(fixture)),
            Err(e) => Err(e.clone()),
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
    /// The launch geometry of the instance the kernels were made for: every
    /// number a clause sizes a launch, a window or an oracle by.
    pub fn geom(&self) -> &Geometry {
        self.kernels.geom()
    }

    /// Run `launch` and read the output window back, after checking that the
    /// words around the window are as they were.
    pub fn matvec(&self, launch: &Launch<'_>) -> Result<Vec<f32>, String> {
        let (device, stream) = (self.device, self.stream);
        let sentinel = f32::from_bits(SENTINEL_BITS);
        let pad = weight_pad(launch.quant);
        let weights = padded(0xFFu8, pad, launch.weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let x = padded(sentinel, ACT_PAD, launch.x, 1);
        let x = device.upload(stream, &x).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;
        let gain_buffer = match &launch.mode {
            ModeSpec::Norm { gain, .. } => {
                let gain = padded(sentinel, ACT_PAD, gain, 1);
                Some(device.upload(stream, &gain).map_err(why)?)
            }
            _ => None,
        };

        let mode = match (&launch.mode, &gain_buffer) {
            (ModeSpec::Store, _) => Mode::Store,
            (ModeSpec::Residual, _) => Mode::Residual,
            (ModeSpec::Norm { eps, layer, .. }, Some(gain)) => Mode::NormOnLoad(NormOnLoad {
                gain: gain
                    .window(Elems(ACT_PAD), Elems(launch.cols))
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
                .window(Elems(pad), Elems(launch.weights.len()))
                .map_err(why)?,
            x: x.window(Elems(ACT_PAD), Elems(launch.cols)).map_err(why)?,
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
        let pad = weight_pad(launch.quant);
        let weights = padded(0xFFu8, pad, &weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let x = padded(sentinel, ACT_PAD, launch.x, 1);
        let x = device.upload(stream, &x).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;
        let gain = padded(sentinel, ACT_PAD, launch.gain, 1);
        let gain = device.upload(stream, &gain).map_err(why)?;

        let params = SegParams {
            quant: launch.quant,
            segments: &segments,
            weights: weights
                .window(Elems(pad), Elems(weight_bytes))
                .map_err(why)?,
            x: x.window(Elems(ACT_PAD), Elems(launch.cols)).map_err(why)?,
            y: y.window_mut(Elems(OUT_GUARD), Elems(rows)).map_err(why)?,
            norm: LoadNorm::Gain {
                gain: gain
                    .window(Elems(ACT_PAD), Elems(launch.cols))
                    .map_err(why)?,
                eps: launch.eps,
            },
            fault: device.fault_word(),
            layer: launch.layer,
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
        let pad = weight_pad(launch.quant);
        let weights = padded(0xFFu8, pad, launch.weights, WEIGHT_TAIL);
        let weights = device.upload(stream, &weights).map_err(why)?;
        let g = padded(sentinel, ACT_PAD, launch.g, 1);
        let g = device.upload(stream, &g).map_err(why)?;
        let partials = padded(sentinel, VALUE_PAD, launch.partials, 1);
        let partials = device.upload(stream, &partials).map_err(why)?;
        let y_words = padded(sentinel, OUT_GUARD, launch.y0, OUT_GUARD);
        let mut y = device.upload(stream, &y_words).map_err(why)?;

        let params = HeadNormParams {
            quant: launch.quant,
            rows: launch.rows,
            cols: launch.cols,
            weights: weights
                .window(Elems(pad), Elems(launch.weights.len()))
                .map_err(why)?,
            g: g.window(Elems(ACT_PAD), Elems(launch.cols)).map_err(why)?,
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

/// `base` with its GEMMs from the tensor-op program `text`, and its kernels.
fn tensor_kernels_of_text(
    device: &MetalDevice,
    base: &Library,
    text: &str,
) -> Result<(Library, Kernels), String> {
    let tensor = Library::tensor_from_source_text(device, text).map_err(why)?;
    let library = base.with_tensor_gemms(tensor);
    let kernels = Kernels::from_library(&library).map_err(why)?;
    Ok((library, kernels))
}

/// What the gate runs at an instance.
#[derive(Clone, Copy)]
enum Table {
    /// Every clause and both mutant tables: the gate is written at this
    /// instance's fixture.
    Whole,
    /// The clauses of the kernels the instance's geometry shapes (router,
    /// experts, attention, and the prompt pass's rows and routing) and the A4
    /// clauses at its sites (the mat-vec's, the prompt pass's dense and
    /// segmented GEMMs, and the tensor-op GEMMs' A4 clauses), the mutants of
    /// those clauses in both tables (`kept`) and the instance's own (`own`).
    Shaped {
        kept: &'static [&'static str],
        own: &'static [mutants::Mutant],
    },
    /// The clauses of the kernels an instance's delta axes shape, on drawn
    /// parameters (the delta step and rows, the head-normed mat-vecs and
    /// the head norm over rows): an instance that is another registered
    /// instance's but for its delta layer, which no fixture carries. The
    /// mutants of those clauses in both tables (`kept`) and the instance's
    /// own (`own`).
    Delta {
        kept: &'static [&'static str],
        own: &'static [mutants::Mutant],
    },
    /// The clauses of the attention kernels an instance's group size shapes
    /// (the decode flash, the rows flash and the prefill flash) on its own
    /// geometry: an instance that is another registered instance's but for
    /// its attention layers. The mutants of those clauses in both tables
    /// (`kept`) and the instance's own (`own`).
    Attention {
        kept: &'static [&'static str],
        own: &'static [mutants::Mutant],
    },
    /// The clauses of the attention kernels an instance's window and
    /// per-head gate axes shape, with the rope table's: the instance's
    /// flash cuts a window a launch names and gates per head, so the plain
    /// launches of the other clause files refuse it by name. The mutants of
    /// those clauses in both tables (`kept`) and the instance's own (`own`).
    Window {
        kept: &'static [&'static str],
        own: &'static [mutants::Mutant],
    },
    /// The clauses of the attention kernels an instance's head-norm and
    /// pairing axes shape: the instance's flash applies no head norm and
    /// turns neighbour pairs, so the gain launches of the other clause files
    /// refuse it by name. The mutants of those clauses in both tables
    /// (`kept`) and the instance's own (`own`).
    Prologue {
        kept: &'static [&'static str],
        own: &'static [mutants::Mutant],
    },
}

/// A registered instance's row of the gate: the architecture whose fixture
/// its clauses read, and what it runs.
struct InstanceRow {
    instance: &'static Instance,
    fixture: &'static str,
    table: Table,
}

/// The gate's row of each registered instance.
const INSTANCE_ROWS: &[InstanceRow] = &[
    InstanceRow {
        instance: &QWEN35MOE_A3B,
        fixture: "qwen35moe",
        table: Table::Whole,
    },
    InstanceRow {
        instance: &QWEN3MOE_A3B,
        fixture: "qwen3moe",
        table: Table::Shaped {
            kept: mutants::Q30_KEPT,
            own: mutants::Q30_MUTANTS,
        },
    },
    InstanceRow {
        instance: &QWEN35MOE_GDN48,
        fixture: "qwen35moe",
        table: Table::Delta {
            kept: mutants::GDN48_KEPT,
            own: mutants::GDN48_MUTANTS,
        },
    },
    InstanceRow {
        instance: &QWEN35MOE_GQA6,
        fixture: "qwen35moe",
        table: Table::Attention {
            kept: mutants::ATTN_KEPT,
            own: mutants::GQA_MUTANTS,
        },
    },
    InstanceRow {
        instance: &QWEN35MOE_GQA4,
        fixture: "qwen35moe",
        table: Table::Attention {
            kept: mutants::ATTN_KEPT,
            own: mutants::GQA_MUTANTS,
        },
    },
    InstanceRow {
        instance: &LFM3_ATTN,
        fixture: "qwen35moe",
        table: Table::Window {
            kept: mutants::WINDOW_KEPT,
            own: mutants::WINDOW_MUTANTS,
        },
    },
    InstanceRow {
        instance: &LLAMA_ATTN,
        fixture: "qwen35moe",
        table: Table::Prologue {
            kept: mutants::PROLOGUE_KEPT,
            own: mutants::PROLOGUE_MUTANTS,
        },
    },
];

/// Every instance of `registry` has exactly one of `rows`, and every row is
/// of a registered instance: a row the gate does not run is an instance whose
/// launches no clause sizes.
fn rows_cover_registry(rows: &[InstanceRow], registry: &[Instance]) -> Result<(), String> {
    for inst in registry {
        match rows.iter().filter(|r| *r.instance == *inst).count() {
            1 => {}
            0 => return Err(format!("instance {} has no gate row", inst.name)),
            n => return Err(format!("instance {} has {n} gate rows", inst.name)),
        }
    }
    match rows.iter().find(|r| !registry.contains(r.instance)) {
        Some(row) => Err(format!(
            "the gate row of instance {} is not a registered instance",
            row.instance.name
        )),
        None => Ok(()),
    }
}

/// The names of the rows, for a refusal that lists them.
fn row_names() -> String {
    let names: Vec<_> = INSTANCE_ROWS.iter().map(|r| r.instance.name).collect();
    names.join(", ")
}

/// The row of the instance named `name`.
fn row_named(name: &str) -> Result<&'static InstanceRow, String> {
    INSTANCE_ROWS
        .iter()
        .find(|r| r.instance.name == name)
        .ok_or_else(|| format!("no instance named {name} (the rows: {})", row_names()))
}

/// The row of the instance the file at `path` selects: the one its shape
/// matches (`instance::select`).
fn row_of_file(path: &Path) -> Result<&'static InstanceRow, String> {
    let refuse = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let split = Split::open(path).map_err(|e| refuse(&e))?;
    let spec = model::arch::spec(&split).map_err(|e| refuse(&e))?.spec;
    let inst = instance::select(&spec).map_err(|e| refuse(&e))?;
    INSTANCE_ROWS
        .iter()
        .find(|r| *r.instance == *inst)
        .ok_or_else(|| refuse(&format!("instance {} has no gate row", inst.name)))
}

/// The rows a run covers: the one `--instance` names, else the one the
/// `--fixture` file selects (both, when they name the same row), else every
/// row, in order.
fn selected_rows(args: &[String]) -> Result<Vec<&'static InstanceRow>, String> {
    let named = flag(args, "--instance").map(row_named).transpose()?;
    let filed = flag(args, "--fixture")
        .map(|path| row_of_file(Path::new(path)))
        .transpose()?;
    match (named, filed) {
        (Some(n), Some(f)) if *n.instance != *f.instance => Err(format!(
            "--fixture selects instance {}, and --instance names {}",
            f.instance.name, n.instance.name
        )),
        (Some(row), _) | (None, Some(row)) => Ok(vec![row]),
        (None, None) => Ok(INSTANCE_ROWS.iter().collect()),
    }
}

/// The entries whose bodies an instance's geometry shapes.
const INSTANCE_ENTRIES: [&str; 11] = [
    "gqa256_flash_f16",
    "gqa256_flash_q8",
    "gqa256_flash_f16_rows",
    "attn_flash_one_f16",
    "attn_flash_one_f16_rows",
    "attn_norm_rope_rows",
    "attn_prefill_f16",
    "router35",
    "route_rows_select",
    "route_table",
    "combine_rows",
];
/// The instance-shaped entries that take the mat-vec's SIMD group count.
const INSTANCE_MATVEC_ENTRIES: [&str; 3] = ["q4k_pair_swiglu", "q4k_down_fold", "q6k_down_fold"];

/// The row's library compiles at the row's instance, and its instance-shaped
/// entries make pipelines: the source is compiled at run time, so a body that
/// only this instance's `#define`s break is caught here. Every registered
/// instance has a row, so every instance's library is covered.
fn instance_library(h: &Harness<'_>, row: &InstanceRow) -> Clause {
    let name = row.instance.name;
    let library = h.device.library();
    if *library.instance() != *row.instance {
        return Err(format!(
            "the library opened for {name} is instance {}",
            library.instance().name
        ));
    }
    for entry in INSTANCE_ENTRIES {
        library
            .pipeline(entry)
            .map_err(|e| format!("{name}: {entry}: {e}"))?;
    }
    let count = u32::try_from(SIMD_GROUPS_PER_GROUP).map_err(|e| e.to_string())?;
    for entry in INSTANCE_MATVEC_ENTRIES {
        library
            .pipeline_with_simdgroups(entry, count)
            .map_err(|e| format!("{name}: {entry}: {e}"))?;
    }
    Ok(Verdict::Noted(format!(
        "{name} compiles, {} entries make pipelines",
        INSTANCE_ENTRIES.len() + INSTANCE_MATVEC_ENTRIES.len(),
    )))
}

/// The fixture path: `--fixture PATH`, else the row's family's fixture under
/// the fixture root (`BLOOMERY_FIXTURE_ROOT`), as the reference sets resolve
/// it.
fn fixture_path(args: &[String], row: &InstanceRow) -> Result<PathBuf, String> {
    if let Some(path) = flag(args, "--fixture") {
        return Ok(PathBuf::from(path));
    }
    refset::fixture::first_shard(row.fixture)
        .map(PathBuf::from)
        .map_err(|e| format!("{e} (name the fixture's trunk file with --fixture PATH)"))
}

/// The MLX-format fixture split the A4 clauses of `row` read: `--mlx-fixture PATH`,
/// else [`mlx_split::path`] of the row's family.
fn mlx_fixture_path(args: &[String], row: &InstanceRow) -> Result<PathBuf, String> {
    if let Some(path) = flag(args, "--mlx-fixture") {
        return Ok(PathBuf::from(path));
    }
    mlx_split::path(row.fixture)
        .map_err(|e| format!("{e} (name the split with --mlx-fixture PATH)"))
}

/// Every mutant of each row's table, each in a child run of this gate
/// (`--instance NAME --mutant NAME`): its clause must be red and the child's
/// exit code not 0. One that stays green is a defect no clause catches. The
/// parent opens no device, so only one child holds the GPU at a time.
fn mutants_all(args: &[String], rows: &[&InstanceRow]) -> ExitCode {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            println!("gate_metal_qwen35moe_kernels --mutants-all: FAIL (the gate's own path: {e})");
            return ExitCode::FAILURE;
        }
    };
    let mut every_row_red = true;
    for row in rows {
        let instance = row.instance.name;
        let mut green = Vec::new();
        let mut count = 0usize;
        for (mutant, _) in mutants::all(row.table) {
            count += 1;
            let mut child = Command::new(&exe);
            child.args(["--instance", instance, "--mutant", mutant.name]);
            if let Some(path) = flag(args, "--fixture") {
                child.args(["--fixture", path]);
            }
            if let Some(path) = flag(args, "--mlx-fixture") {
                child.args(["--mlx-fixture", path]);
            }
            let out = match child.output() {
                Ok(out) => out,
                Err(e) => {
                    println!(
                        "mutant {} [{instance}]: the child gate did not start: {e}",
                        mutant.name
                    );
                    return ExitCode::FAILURE;
                }
            };
            let text = String::from_utf8_lossy(&out.stdout);
            let red =
                !out.status.success() && text.contains(&format!("clause {}: FAIL", mutant.clause));
            println!(
                "mutant {} [{instance}]: {} on {}",
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
                "gate_metal_qwen35moe_kernels --mutants-all [{instance}]: PASS ({count} mutants, each red on its clause)"
            );
        } else {
            every_row_red = false;
            println!(
                "gate_metal_qwen35moe_kernels --mutants-all [{instance}]: FAIL (not red: {})",
                green.join(", ")
            );
        }
    }
    if every_row_red {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// What one row's run came to.
enum Row {
    /// Every clause the row ran passed, or its clause names were printed.
    Passed,
    /// A clause failed.
    Failed,
    /// `--only` or `--child` (or the mutant's clause) names a clause the
    /// row's table does not have.
    NoClause(String),
    /// The gate ends with this code: no Metal device here, a mutant that
    /// does not build, a mutant the row has none of.
    Ends(u8),
}

/// One row: the device opened at the row's instance, the kernels and the
/// harness of its geometry, the row's fixture, its table. The device and
/// everything it holds drop when this returns, before the next row opens.
fn run_row(args: &[String], row: &InstanceRow) -> Row {
    let inst = row.instance;
    let gate = format!("gate_metal_qwen35moe_kernels [{}]", inst.name);
    let device = match MetalDevice::open(inst) {
        Ok(device) => device,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("clause open: REFUSED ({e})");
            return Row::Ends(78);
        }
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return Row::Failed;
        }
    };
    let stream = device.stream();
    // The simdgroup GEMMs, on every device: the tensor-op ones have clauses
    // of their own.
    let simdgroup = device.library().with_simdgroup_gemms();
    let kernels = match Kernels::from_library(&simdgroup) {
        Ok(kernels) => kernels,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return Row::Failed;
        }
    };
    let info = match device.info() {
        Ok(info) => info,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return Row::Failed;
        }
    };
    println!("instance: {}", inst.name);
    println!(
        "metal-lib: path={} device={} backend={}",
        device.library().path().name(),
        info.name,
        <MetalDevice as Device>::BACKEND
    );
    let units = device.library().matrix_units();
    println!(
        "gemm: architecture={} metal4={} matrix-units={} engine-path={}",
        units.architecture,
        units.metal4,
        units.present(),
        device.library().gemm_path().name()
    );
    let mutant = match flag(args, "--mutant") {
        Some(name) => match mutants::find(name, row.table) {
            Some(mutant) => Some(mutant),
            None => {
                println!("{gate}: no mutant named {name}");
                return Row::Ends(64);
            }
        },
        None => None,
    };
    let (main_text, tensor_text) = match library::sources_of(inst) {
        Ok(texts) => texts,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return Row::Failed;
        }
    };
    let (run_source, mutated) = match mutant {
        Some((mutant, mutants::Program::Main)) => {
            match mutants::source_at(inst, mutant, mutants::Program::Main).and_then(|text| {
                let built = kernels_of_text(&device, &text)?;
                Ok((text, Some(built)))
            }) {
                Ok(built) => built,
                Err(e) => {
                    println!("{gate}: the mutant does not build: {e}");
                    return Row::Ends(64);
                }
            }
        }
        _ => (main_text.to_string(), None),
    };
    let tensor_source = match mutant {
        Some((mutant, mutants::Program::Tensor)) => {
            match mutants::source_at(inst, mutant, mutants::Program::Tensor) {
                Ok(text) => text,
                Err(e) => {
                    println!("{gate}: the mutant does not build: {e}");
                    return Row::Ends(64);
                }
            }
        }
        _ => tensor_text.to_string(),
    };
    let harness = Harness {
        device: &device,
        stream,
        kernels: mutated.as_ref().map_or(&kernels, |(_, k)| k),
        library: mutated.as_ref().map_or(&simdgroup, |(l, _)| l),
    };
    let tensor_built = tensor_kernels_of_text(&device, harness.library, &tensor_source);
    if let (Some((mutant, mutants::Program::Tensor)), Err(e)) = (mutant, &tensor_built) {
        println!("{gate}: the mutant {} does not build: {e}", mutant.name);
        return Row::Ends(64);
    }
    let tensor = tensor_built
        .as_ref()
        .map_err(Clone::clone)
        .map(|(l, k)| Harness {
            device: &device,
            stream,
            kernels: k,
            library: l,
        });
    if let Some((mutant, _)) = mutant {
        println!("mutant: {} ({})", mutant.name, mutant.why);
    }
    let only = flag(args, "--child")
        .or(mutant.map(|(m, _)| m.clause))
        .or(flag(args, "--only"));
    let fixture_at = fixture_path(args, row);
    // A child gate a clause starts (`row_tail_reads`) is told this row's
    // fixture, so the file it selects is this row.
    let mut row_args = args.to_vec();
    if flag(args, "--fixture").is_none()
        && let Ok(path) = &fixture_at
    {
        row_args.extend(["--fixture".to_string(), path.display().to_string()]);
    }
    let fixture = fixture_at.and_then(|path| fixture::Fixture::open(&path));
    match &fixture {
        Ok(f) => println!("fixture: {}", f.path.display()),
        Err(e) => println!("fixture: {e}"),
    }
    // The MLX-format split beside it: the A4 clauses' weights. An A4 clause
    // run without it is red, not skipped — the split is the fixture the
    // family's MLX plan reads.
    let mlx = mlx_fixture_path(args, row).and_then(|path| fixture::Fixture::open(&path));
    match &mlx {
        Ok(f) => println!("mlx fixture: {}", f.path.display()),
        Err(e) => println!("mlx fixture: {e}"),
    }

    let ctx = ClauseCtx {
        harness: &harness,
        fixture: &fixture,
        args: &row_args,
        source: &run_source,
    };
    let clauses = clause_table(row, &ctx, &tensor, &mlx);
    if args.iter().any(|a| a == "--list") {
        for (name, _) in &clauses {
            println!("{name}");
        }
        return Row::Passed;
    }
    if let Some(name) = only
        && !clauses.iter().any(|(n, _)| *n == name)
    {
        return Row::NoClause(name.to_string());
    }
    let mut report = Report::new();
    let mut failed = false;
    for (name, run) in &clauses {
        if only.is_none_or(|o| o == *name) {
            let result = run();
            failed |= result.is_err();
            report.clause(name, result);
        }
    }
    let scope = match (mutant, only) {
        (Some((mutant, _)), _) => format!(" (mutant {})", mutant.name),
        (None, Some(name)) => format!(" (only {name})"),
        (None, None) => String::new(),
    };
    let _ = report.finish(&format!("{gate}{scope}"));
    if failed { Row::Failed } else { Row::Passed }
}

/// The row's clauses in the order the gate runs them. The mat-vec, norm,
/// embed, seg and head-norm families come in gate order, then the token's
/// kernels in the token DAG's. A new kernel adds one line.
fn clause_table<'a>(
    row: &'a InstanceRow,
    ctx: &'a ClauseCtx<'a>,
    tensor: &'a Result<Harness<'a>, String>,
    mlx: &'a Result<fixture::Fixture, String>,
) -> Vec<(&'static str, ClauseRun<'a>)> {
    let instances = || {
        std::iter::once((
            "instances",
            Box::new(move || instance_library(ctx.harness, row)) as ClauseRun<'a>,
        ))
    };
    match row.table {
        Table::Delta { .. } => instances()
            .chain(clauses_delta::wide(ctx))
            .chain(clauses_rows::wide(ctx))
            .collect(),
        Table::Attention { .. } => instances()
            .chain(clauses_attention::table(ctx))
            .chain(clauses_rows::attention(ctx))
            .chain(clauses_window::refusals(ctx))
            .collect(),
        Table::Window { .. } => instances().chain(clauses_window::table(ctx)).collect(),
        Table::Prologue { .. } => instances()
            .chain(clauses_window::prologue_table(ctx))
            .collect(),
        Table::Shaped { .. } => instances()
            .chain(clauses_router::table(ctx))
            .chain(clauses_experts::table(ctx))
            .chain(clauses_attention::table(ctx))
            .chain(clauses_rows::shaped(ctx))
            .chain(clauses_route::shaped(ctx, mlx))
            .chain(clauses_a4::shaped(ctx.harness, mlx))
            .chain(clauses_gemm::shaped(ctx.harness, mlx))
            .chain(clauses_tensor::shaped(tensor, mlx))
            .collect(),
        Table::Whole => front(ctx)
            .into_iter()
            .chain(instances())
            .chain(clauses_a4::table(ctx.harness, mlx))
            .chain(clauses_requant::table(ctx.harness, mlx))
            .chain(clauses_q8_0::table(ctx.harness))
            .chain(clauses_q5k::table(ctx.harness))
            .chain(std::iter::once((
                "f32_seg",
                Box::new(|| clauses_f32::f32_seg(ctx.harness)) as ClauseRun,
            )))
            .chain(clauses_sites::table(ctx.harness))
            .chain(clauses_gemm::table(ctx.harness, ctx.fixture, mlx))
            .chain(clauses_requant::gemm_table(ctx.harness, mlx))
            .chain(clauses_delta::table(ctx))
            .chain(clauses_shortconv::table(ctx))
            .chain(clauses_handon::table(ctx))
            .chain(clauses_router::table(ctx))
            .chain(clauses_experts::table(ctx))
            .chain(clauses_apart::table(ctx))
            .chain(clauses_attention::table(ctx))
            .chain(clauses_head::table(ctx))
            .chain(clauses_rows::table(ctx))
            .chain(clauses_route::table(ctx, mlx))
            .chain(clauses_tensor::table(tensor, ctx.fixture, mlx))
            .chain(clauses_requant::tensor_table(tensor, mlx))
            .collect(),
    }
}

/// The mat-vec, norm, embed, seg and head-norm clauses of the whole table.
fn front<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    vec![
        (
            "q4k_matvec",
            Box::new(|| ctx.with(clauses_matvec::q4k_matvec)) as ClauseRun,
        ),
        (
            "q6k_matvec",
            Box::new(|| ctx.with(clauses_matvec::q6k_matvec)),
        ),
        (
            "norm_on_load",
            Box::new(|| ctx.with(clauses_norm::norm_on_load)),
        ),
        (
            "residual_epilogue",
            Box::new(|| ctx.with(clauses_norm::residual_epilogue)),
        ),
        ("row_tail", Box::new(|| ctx.with(clauses_matvec::row_tail))),
        (
            "row_tail_reads",
            Box::new(|| clauses_matvec::row_tail_reads(ctx.args)),
        ),
        (
            "metallib_equals_source",
            Box::new(|| match ctx.fixture {
                Ok(fixture) => clauses_matvec::metallib_equals_source(
                    ctx.harness.device,
                    ctx.harness.stream,
                    fixture,
                ),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("embed", Box::new(|| ctx.with(clauses_embed::embed))),
        (
            "matvec_refusals",
            Box::new(|| clauses_matvec::matvec_refusals(ctx.harness)),
        ),
        ("block_shape", Box::new(clauses_matvec::block_shape)),
        ("seg_bit", Box::new(|| ctx.with(clauses_seg::seg_bit))),
        ("seg_bounds", Box::new(|| ctx.with(clauses_seg::seg_bounds))),
        ("headnorm", Box::new(|| ctx.with(clauses_norm::headnorm))),
        (
            "headnorm_nan",
            Box::new(|| ctx.with(clauses_norm::headnorm_nan)),
        ),
        ("nsg", Box::new(|| ctx.with(clauses_matvec::nsg))),
        ("nrows", Box::new(|| ctx.with(clauses_matvec::nrows))),
    ]
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(name) = [
        "--instance",
        "--fixture",
        "--mlx-fixture",
        "--only",
        "--mutant",
        "--child",
    ]
    .into_iter()
    .find(|name| args.last().is_some_and(|last| last == name))
    {
        println!("gate_metal_qwen35moe_kernels: {name} takes a value");
        return ExitCode::from(64);
    }
    if let Err(e) = rows_cover_registry(INSTANCE_ROWS, REGISTERED) {
        println!("gate_metal_qwen35moe_kernels: FAIL ({e})");
        return ExitCode::FAILURE;
    }
    let rows = selected_rows(&args).and_then(|rows| {
        if flag(&args, "--mutant").is_some() && rows.len() > 1 {
            return Err(
                "--mutant needs --instance (or a --fixture that selects the row): a mutant edits one instance's source"
                    .to_string(),
            );
        }
        Ok(rows)
    });
    let rows = match rows {
        Ok(rows) => rows,
        Err(e) => {
            println!("gate_metal_qwen35moe_kernels: {e}");
            return ExitCode::from(64);
        }
    };
    if args.iter().any(|a| a == "--mutants-all") {
        return mutants_all(&args, &rows);
    }
    let listing = args.iter().any(|a| a == "--list");
    let mut failed = Vec::new();
    let mut no_clause = None;
    let mut ran = 0usize;
    for row in &rows {
        if listing && rows.len() > 1 {
            println!("[{}]", row.instance.name);
        }
        match run_row(&args, row) {
            Row::Passed => ran += 1,
            Row::Failed => {
                ran += 1;
                failed.push(row.instance.name);
            }
            Row::NoClause(name) => no_clause = Some(name),
            Row::Ends(code) => return ExitCode::from(code),
        }
    }
    if ran == 0
        && let Some(name) = no_clause
    {
        println!("gate_metal_qwen35moe_kernels: no clause named {name}");
        return ExitCode::from(64);
    }
    if !listing && rows.len() > 1 {
        if failed.is_empty() {
            println!(
                "gate_metal_qwen35moe_kernels: PASS ({} instances)",
                rows.len()
            );
        } else {
            println!("gate_metal_qwen35moe_kernels: FAIL ({})", failed.join(", "));
        }
    }
    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
