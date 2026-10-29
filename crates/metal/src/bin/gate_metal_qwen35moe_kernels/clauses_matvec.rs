//! The mat-vec clauses: the two quantized dots against the f64 oracle, the
//! row tails, the library comparison, the host's refusals, the block shape
//! and the SIMD-group geometry.

use std::process::Command;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::router;
use bloomery_metal::kernels::{
    Kernels, MatvecParams, Mode, NormOnLoad, Quant, ROWS_PER_SIMD_GROUP, SIMD_GROUPS_PER_GROUP,
};
use bloomery_metal::library::{Library, sources_of};
use bloomery_metal::stream::MetalStream;
use gguf::quant::GgmlType;

use super::common::{self, BadArg, Edit};
use super::fixture::Fixture;
use super::oracle;
use bloomery_metal::kernels::BLOCK_VALUES;

use super::{Harness, Launch, ModeSpec, flag};

pub fn q4k_matvec(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        common::Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
        common::Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048),
        common::Tensor::q4k("blk.0.ffn_down_shexp.weight", 512, 2048),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B34 + i as u64, 1.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    common::noted(parts)
}

pub fn q6k_matvec(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        common::Tensor::q6k("blk.2.attn_qkv.weight", 2048, 8192),
        common::Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
        common::Tensor::q6k("blk.3.attn_v.weight", 2048, 512),
        common::Tensor::q6k("output.weight", 2048, 4096),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B36 + i as u64, 1.0);
        let worst = common::run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &common::sentinel_rows(t.rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    // The head's last rows: a count that is not a multiple of the rows of a
    // threadgroup, taken from deep in the 248320-row tensor.
    let tail_rows = 1153;
    let head = fixture.rows_at(
        "output.weight",
        GgmlType::Q6_K,
        2048,
        248_320 - tail_rows,
        tail_rows,
    )?;
    let x = oracle::activations(2048, 0x4B3A, 1.0);
    let worst = common::run_checked(
        h,
        &Launch {
            quant: Quant::Q6K,
            rows: tail_rows,
            cols: 2048,
            weights: head,
            x: &x,
            y0: &common::sentinel_rows(tail_rows),
            mode: ModeSpec::Store,
        },
    )
    .map_err(|e| format!("output.weight rows 247167..248320: {e}"))?;
    parts.push(format!("output.weight tail {tail_rows}x2048 {worst:.3}"));
    common::noted(parts)
}

pub fn row_tail(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    for (quant, name) in [
        (Quant::Q4K, "blk.0.attn_qkv.weight"),
        (Quant::Q6K, "blk.2.attn_qkv.weight"),
    ] {
        let t = common::Tensor {
            name,
            quant,
            cols: 2048,
            rows: 8192,
        };
        let all = t.bytes(fixture)?;
        let row_bytes = 2048 / BLOCK_VALUES * quant.block_bytes();
        let x = oracle::activations(2048, 0x4B60, 1.0);
        for rows in counts {
            common::run_checked(
                h,
                &Launch {
                    quant,
                    rows,
                    cols: 2048,
                    weights: &all[..rows * row_bytes],
                    x: &x,
                    y0: &common::sentinel_rows(rows),
                    mode: ModeSpec::Store,
                },
            )
            .map_err(|e| format!("{} with {rows} rows: {e}", quant.name()))?;
        }
    }
    Ok(Verdict::Noted(format!(
        "{} row counts per type, every row written, no word past the last",
        counts.len()
    )))
}

/// What a child run of the gate under Metal's shader validation said on
/// stderr.
struct Validated {
    /// Whether the validation layer announced itself.
    started: bool,
    /// How many accesses it reported.
    reports: usize,
    /// The first report's lines, for a clause that fails on it.
    first: String,
    /// The child's own verdict and last words.
    exit: Result<(), String>,
}

/// The gate's own binary, run as `--child row_tail` with Metal's shader
/// validation on (its reports go to stderr), beside the gate's own `--fixture`
/// and a `--mutant` when one is named.
fn row_tail_under_validation(args: &[String], mutant: Option<&str>) -> Result<Validated, String> {
    let exe = std::env::current_exe().map_err(|e| format!("the gate's own path: {e}"))?;
    let mut child = Command::new(exe);
    child.args(["--child", "row_tail"]);
    if let Some(path) = flag(args, "--fixture") {
        child.args(["--fixture", path]);
    }
    if let Some(name) = mutant {
        child.args(["--mutant", name]);
    }
    child
        .env("MTL_SHADER_VALIDATION", "1")
        .env("MTL_SHADER_VALIDATION_GLOBAL_MEMORY", "1")
        .env("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1")
        .env("MTL_SHADER_VALIDATION_ENABLE_VERBOSE_LOGGING", "1");
    let out = child
        .output()
        .map_err(|e| format!("the child gate did not start: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first = stderr
        .split("=====")
        .find(|block| block.contains("Invalid Metal usage"))
        .map(|block| {
            block
                .lines()
                .filter(|l| !l.is_empty())
                .take(2)
                .collect::<Vec<_>>()
                .join(" / ")
        })
        .unwrap_or_default();
    Ok(Validated {
        started: stderr.contains("Metal GPU Validation Enabled"),
        reports: stderr.matches("Invalid Metal usage").count(),
        first,
        exit: if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "the child exited {}: {}",
                out.status,
                stdout.lines().last().unwrap_or("no output")
            ))
        },
    })
}

/// No launch of `row_tail` reads outside its weights' buffer. The reads of the
/// rows past the last are never stored, so no output shows them; only Metal's
/// shader validation reports an access outside a buffer, and it must be on
/// when the process starts, so the launches run in child gates. The rows are
/// wider than the buffer's padding after the window, so a read of the row past
/// the last leaves the buffer.
///
/// The control: the same child with the `tail_read` mutant, which reads those
/// rows, must be reported; a layer that reports nothing there checked nothing,
/// and the clause is red, not green.
pub fn row_tail_reads(args: &[String]) -> Clause {
    let control = row_tail_under_validation(args, Some("tail_read"))?;
    if !control.started || control.reports == 0 {
        return Err(format!(
            "shader validation {} for the control, a kernel that reads past its weights: nothing \
             was checked",
            if control.started {
                "reported nothing"
            } else {
                "did not start"
            }
        ));
    }
    let own = row_tail_under_validation(args, flag(args, "--mutant"))?;
    if !own.started {
        return Err("shader validation did not start: nothing was checked".to_string());
    }
    if own.reports > 0 {
        return Err(format!(
            "{} accesses outside a buffer reported, the first: {}",
            own.reports, own.first
        ));
    }
    own.exit?;
    Ok(Verdict::Noted(format!(
        "no report in the row-tail launches, {} for the control",
        control.reports
    )))
}

#[cfg(metal_lib = "metallib")]
pub fn metallib_equals_source(
    device: &MetalDevice,
    stream: &MetalStream,
    fixture: &Fixture,
) -> Clause {
    let outputs = |library: &Library| -> Result<Vec<Vec<f32>>, String> {
        let kernels = Kernels::from_library(library).map_err(why)?;
        let h = Harness {
            device,
            stream,
            kernels: &kernels,
            library,
        };
        suite_outputs(&h, fixture)
    };
    let metallib = outputs(&Library::from_metallib(device).map_err(why)?)?;
    let source = outputs(&Library::from_source(device).map_err(why)?)?;
    for (i, (a, b)) in metallib.iter().zip(&source).enumerate() {
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            if x.to_bits() != y.to_bits() {
                return Err(format!(
                    "launch {i} word {j}: the metallib read {x:?}, the source read {y:?}"
                ));
            }
        }
    }
    Ok(Verdict::Noted(format!(
        "{} launches, every word the same bits",
        metallib.len()
    )))
}

#[cfg(not(metal_lib = "metallib"))]
pub fn metallib_equals_source(
    _device: &MetalDevice,
    _stream: &MetalStream,
    _fixture: &Fixture,
) -> Clause {
    Ok(Verdict::Skip(
        "the Metal Toolchain was not installed when this crate was built: no metallib exists; \
         install it and rebuild"
            .to_string(),
    ))
}

/// A short fixed run of every entry, for the comparison of two libraries: the
/// outputs, in launch order.
#[cfg(metal_lib = "metallib")]
fn suite_outputs(h: &Harness<'_>, fixture: &Fixture) -> Result<Vec<Vec<f32>>, String> {
    let eps = fixture.rms_eps()?;
    let mut out = Vec::new();
    for (quant, name, cols) in [
        (Quant::Q4K, "blk.0.attn_qkv.weight", 2048usize),
        (Quant::Q6K, "blk.2.attn_qkv.weight", 2048),
    ] {
        let rows = 67;
        let t = common::Tensor {
            name,
            quant,
            cols,
            rows,
        };
        let weights = t.bytes(fixture)?;
        let x = oracle::activations(cols, 0x51, 1.0);
        let gain = common::gain_for(fixture, Some("blk.0.attn_norm.weight"), cols, 0)?;
        for mode in [
            ModeSpec::Store,
            ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: common::LAYER,
            },
            ModeSpec::Residual,
        ] {
            out.push(h.matvec(&Launch {
                quant,
                rows,
                cols,
                weights,
                x: &x,
                y0: &oracle::activations(rows, 0x52, 1.0),
                mode,
            })?);
        }
    }
    let (table, vocab, ty) = fixture.table("token_embd.weight", 2048)?;
    let quant = Quant::of_ggml(ty)
        .ok_or_else(|| format!("token_embd.weight is {ty:?}, not a table type"))?;
    out.push(super::clauses_embed::run_embed(
        h,
        quant,
        table,
        vocab,
        &[3, 99, 4096],
    )?);
    Ok(out)
}

/// The buffers of the refusal table: the windows' buffers, and the shape every
/// well-formed case of the table shares.
struct Refusals {
    weights: bloomery_metal::MetalBuffer<u8>,
    x: bloomery_metal::MetalBuffer<f32>,
    y: bloomery_metal::MetalBuffer<f32>,
    gain: bloomery_metal::MetalBuffer<f32>,
}

/// The launch of one case: the shape it is told and the length of each window
/// it is given, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    quant: Quant,
    rows: usize,
    cols: usize,
    weights: usize,
    /// The element the weights' window starts at in its buffer.
    weights_at: usize,
    x: usize,
    /// The element the input's window starts at in its buffer.
    x_at: usize,
    y: usize,
    gain: Option<usize>,
    /// The element the gain's window starts at in its buffer.
    gain_at: usize,
}

impl Args {
    const fn whole(rows: usize, cols: usize, bytes: usize) -> Self {
        Self {
            quant: Quant::Q4K,
            rows,
            cols,
            weights: bytes,
            weights_at: 0,
            x: cols,
            x_at: 0,
            y: rows,
            gain: None,
            gain_at: 0,
        }
    }
}

fn refuse(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let mode = match a.gain {
        Some(len) => Mode::NormOnLoad(NormOnLoad {
            gain: b.gain.window(Elems(a.gain_at), Elems(len))?,
            eps: 1e-6,
            fault: h.device.fault_word(),
            layer: common::LAYER,
        }),
        None => Mode::Store,
    };
    h.kernels.matvec(
        encoder,
        MatvecParams {
            quant: a.quant,
            rows: a.rows,
            cols: a.cols,
            weights: b.weights.window(Elems(a.weights_at), Elems(a.weights))?,
            x: b.x.window(Elems(a.x_at), Elems(a.x))?,
            y: b.y.window_mut(Elems(0), Elems(a.y))?,
            mode,
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
    whole: Args,
) -> BadArg<'a, Refusals> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, b, &a, encoder)
        }),
    )
}

/// The launches the host module must refuse: the shape it is told and the
/// length of each window it is given, each refused naming the argument.
///
/// An input or gain window that does not start on 16 bytes is refused too:
/// deleting the `Float4` check of `x` or of `gain` in `matvec.rs`'s `check`
/// turns its case red.
pub fn matvec_refusals(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let rows = 8usize;
    let cols = 512usize;
    let bytes = rows * cols / BLOCK_VALUES * Quant::Q4K.block_bytes();
    let mut bufs = Refusals {
        weights: device.upload(stream, &vec![0u8; bytes + 64]).map_err(why)?,
        x: device.upload(stream, &vec![0f32; cols + 4]).map_err(why)?,
        y: device.upload(stream, &vec![1f32; rows + 4]).map_err(why)?,
        gain: device.upload(stream, &vec![1f32; cols + 4]).map_err(why)?,
    };
    let whole = Args::whole(rows, cols, bytes);
    let cases = vec![
        bad(
            "a short weights window",
            "weights",
            |a| a.weights -= 1,
            h,
            whole,
        ),
        bad(
            "a long weights window",
            "weights",
            |a| a.weights += 1,
            h,
            whole,
        ),
        bad(
            "weights at an odd byte offset",
            "16-bit words",
            |a| a.weights_at = 1,
            h,
            whole,
        ),
        // A4's 36-byte groups total the Q4_K shape's bytes at these rows and
        // columns, so the same window holds them.
        bad(
            "A4 weights two bytes into their buffer",
            "32-bit words",
            |a| {
                a.quant = Quant::A4;
                a.weights_at = 2;
            },
            h,
            whole,
        ),
        bad("a short x window", "x window", |a| a.x -= 1, h, whole),
        bad(
            "an x window one float into its buffer",
            "16-byte aligned",
            |a| a.x_at = 1,
            h,
            whole,
        ),
        bad("a short y window", "y window", |a| a.y -= 1, h, whole),
        bad(
            "a short gain window",
            "gain",
            |a| a.gain = Some(cols - 1),
            h,
            whole,
        ),
        bad(
            "a gain window one float into its buffer",
            "16-byte aligned",
            |a| {
                a.gain = Some(cols);
                a.gain_at = 1;
            },
            h,
            whole,
        ),
        bad(
            "columns that are not whole blocks",
            "whole blocks",
            |a| {
                a.cols = 500;
                a.x = 500;
            },
            h,
            whole,
        ),
        bad(
            "columns past the kernels' signed 32-bit index",
            "signed 32-bit",
            |a| a.cols = (1usize << 31) + BLOCK_VALUES,
            h,
            whole,
        ),
        bad(
            "no rows",
            "empty",
            |a| {
                a.rows = 0;
                a.weights = 0;
                a.y = 0;
            },
            h,
            whole,
        ),
        bad(
            "no columns",
            "empty",
            |a| {
                a.cols = 0;
                a.weights = 0;
                a.x = 0;
            },
            h,
            whole,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| common::still(h, &b.y, |v| *v == 1.0);
    let count = common::refuse_launches(h, "matvec", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let a = whole;
        refuse(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.y, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written"
    )))
}

/// The block shape the kernels are written for is gguf's own row for each
/// type: the bytes are gguf's row by construction ([`Quant::block_bytes`]
/// reads it), and this crate's own count of the values a block holds is
/// held to the same row.
pub fn block_shape() -> Clause {
    for quant in [Quant::Q4K, Quant::Q6K] {
        let ty = quant.ggml();
        if Some(BLOCK_VALUES as u64) != ty.blck_size() {
            return Err(format!(
                "{} blocks hold {BLOCK_VALUES} values here and {:?} in gguf",
                quant.name(),
                ty.blck_size()
            ));
        }
    }
    Ok(Verdict::Noted(
        "Q4_K and Q6_K blocks hold gguf's 256 values; their bytes are gguf's by construction"
            .to_string(),
    ))
}

/// The kernels compiled from the shipped source of the harness's instance
/// with the row count a SIMD group owns replaced by `rows`: the probe sweeps
/// the same edit, and the clause holds the sweep's arithmetic still.
fn kernels_of_rows(h: &Harness<'_>, rows: usize) -> Result<Kernels, String> {
    let (source, _) = sources_of(h.library.instance()).map_err(why)?;
    let find = format!("#define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {ROWS_PER_SIMD_GROUP}");
    let found = source.matches(find.as_str()).count();
    if found != 1 {
        return Err(format!(
            "the geometry define occurs {found} times, not 1; the header moved"
        ));
    }
    let with = format!("#define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {rows}");
    let text = source.replacen(find.as_str(), with.as_str(), 1);
    let library = Library::from_source_text(h.device, &text).map_err(why)?;
    Kernels::with_geometry(&library, rows, SIMD_GROUPS_PER_GROUP, router::SG).map_err(why)
}

/// The mat-vec at two rows a SIMD group is the same bits as at the shipped
/// four: the row count is a geometry, not an arithmetic — a row's lane map
/// and block order do not move, only which rows share a SIMD group.
pub fn nrows(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let two = kernels_of_rows(h, 2)?;
    // A row count the library's source does not define launches rows no
    // kernel writes: the shipped library refuses two.
    if Kernels::with_geometry(h.library, 2, SIMD_GROUPS_PER_GROUP, router::SG).is_ok() {
        return Err(
            "the shipped library took two rows a SIMD group; its source defines \
             another count"
                .to_string(),
        );
    }
    let few = Harness {
        device: h.device,
        stream: h.stream,
        kernels: &two,
        library: h.library,
    };
    let mut parts = Vec::new();
    for (name, quant, cols, rows) in [
        ("blk.0.attn_qkv.weight", Quant::Q4K, 2048usize, 8192usize),
        ("blk.2.attn_qkv.weight", Quant::Q6K, 2048, 8192),
        ("blk.2.ffn_down_shexp.weight", Quant::Q6K, 512, 2048),
    ] {
        let bytes = fixture.rows(name, quant.ggml(), cols, rows)?;
        let x = oracle::activations(cols, 0x5EB, 1.0);
        let many = h.matvec(&Launch {
            quant,
            rows,
            cols,
            weights: bytes,
            x: &x,
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Store,
        })?;
        let few_rows = few.matvec(&Launch {
            quant,
            rows,
            cols,
            weights: bytes,
            x: &x,
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Store,
        })?;
        for (i, (a, b)) in many.iter().zip(few_rows.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "{name} row {i}: four rows a group read {a:?}, two read {b:?}"
                ));
            }
        }
        parts.push(format!("{name} {rows}x{cols}"));
    }
    // A tail: rows that are not a multiple of the four a SIMD group owns, so
    // the last group's live rows are one and its dead rows are never stored.
    let tail = 1153;
    let bytes = fixture.rows_at("output.weight", GgmlType::Q6_K, 2048, 248_320 - tail, tail)?;
    let x = oracle::activations(2048, 0x5EB, 1.0);
    let many = h.matvec(&Launch {
        quant: Quant::Q6K,
        rows: tail,
        cols: 2048,
        weights: bytes,
        x: &x,
        y0: &common::sentinel_rows(tail),
        mode: ModeSpec::Store,
    })?;
    let few_rows = few.matvec(&Launch {
        quant: Quant::Q6K,
        rows: tail,
        cols: 2048,
        weights: bytes,
        x: &x,
        y0: &common::sentinel_rows(tail),
        mode: ModeSpec::Store,
    })?;
    for (i, (a, b)) in many.iter().zip(few_rows.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "the tail row {i}: four rows a group read {a:?}, two read {b:?}"
            ));
        }
    }
    parts.push(format!("output.weight tail {tail}x2048"));
    Ok(Verdict::Noted(format!(
        "{} shapes, every row the same bits at four and two rows a SIMD group",
        parts.len()
    )))
}

/// The Q6_K mat-vec at eight SIMD groups in a threadgroup is the same bits as
/// at two: the count is a geometry, not an arithmetic.
pub fn nsg(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eight = Kernels::with_simdgroups(h.library, 8).map_err(why)?;
    let many = Harness {
        device: h.device,
        stream: h.stream,
        kernels: &eight,
        library: h.library,
    };
    let mut parts = Vec::new();
    for (name, cols, rows) in [
        ("blk.2.attn_qkv.weight", 2048usize, 8192usize),
        ("blk.2.ffn_down_shexp.weight", 512, 2048),
        ("blk.3.attn_v.weight", 2048, 512),
        ("output.weight", 2048, 4096),
    ] {
        let bytes = fixture.rows(name, GgmlType::Q6_K, cols, rows)?;
        let x = oracle::activations(cols, 0x5EA, 1.0);
        let two = h.matvec(&Launch {
            quant: Quant::Q6K,
            rows,
            cols,
            weights: bytes,
            x: &x,
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Store,
        })?;
        let many_rows = many.matvec(&Launch {
            quant: Quant::Q6K,
            rows,
            cols,
            weights: bytes,
            x: &x,
            y0: &common::sentinel_rows(rows),
            mode: ModeSpec::Store,
        })?;
        for (i, (a, b)) in two.iter().zip(many_rows.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "{name} row {i}: two groups read {a:?}, eight read {b:?}"
                ));
            }
        }
        parts.push(format!("{name} {rows}x{cols}"));
    }
    // A tail: rows that are not a multiple of the sixteen a threadgroup of
    // eight SIMD groups owns.
    let tail = 1153;
    let bytes = fixture.rows_at("output.weight", GgmlType::Q6_K, 2048, 248_320 - tail, tail)?;
    let x = oracle::activations(2048, 0x5EA, 1.0);
    let two = h.matvec(&Launch {
        quant: Quant::Q6K,
        rows: tail,
        cols: 2048,
        weights: bytes,
        x: &x,
        y0: &common::sentinel_rows(tail),
        mode: ModeSpec::Store,
    })?;
    let many_rows = many.matvec(&Launch {
        quant: Quant::Q6K,
        rows: tail,
        cols: 2048,
        weights: bytes,
        x: &x,
        y0: &common::sentinel_rows(tail),
        mode: ModeSpec::Store,
    })?;
    for (i, (a, b)) in two.iter().zip(many_rows.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!(
                "the tail row {i}: two groups read {a:?}, eight read {b:?}"
            ));
        }
    }
    parts.push(format!("output.weight tail {tail}x2048"));
    Ok(Verdict::Noted(format!(
        "{} shapes, every row the same bits at two and eight SIMD groups",
        parts.len()
    )))
}
