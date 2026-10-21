//! The clauses of the qwen35moe kernels' gate.

use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{EmbedParams, MatvecParams, Mode, NormOnLoad, Quant};
use bloomery_metal::stream::MetalStream;
use gguf::quant::GgmlType;

use super::oracle::{self, Fixture, Kind};
use super::{Clause, Harness, Launch, ModeSpec, SENTINEL_BITS, Verdict, why};

/// The layer the clauses launch in: a model layer, so a raise is told apart
/// from a launch that names none.
const LAYER: u32 = 5;

/// A tensor a clause reads: its name, the type, the columns and the rows the
/// clause takes of it.
struct Tensor {
    name: &'static str,
    quant: Quant,
    cols: usize,
    rows: usize,
}

impl Tensor {
    const fn q4k(name: &'static str, cols: usize, rows: usize) -> Self {
        Self {
            name,
            quant: Quant::Q4K,
            cols,
            rows,
        }
    }

    const fn q6k(name: &'static str, cols: usize, rows: usize) -> Self {
        Self {
            name,
            quant: Quant::Q6K,
            cols,
            rows,
        }
    }

    fn ggml(&self) -> GgmlType {
        match self.quant {
            Quant::Q4K => GgmlType::Q4_K,
            Quant::Q6K => GgmlType::Q6_K,
        }
    }

    fn bytes<'f>(&self, fixture: &'f Fixture) -> Result<&'f [u8], String> {
        fixture.rows(self.name, self.ggml(), self.cols, self.rows)
    }

    fn label(&self) -> String {
        format!("{} {}x{}", self.name, self.rows, self.cols)
    }
}

/// The gain a norm clause uses for a tensor of `cols` columns: the fixture's
/// own where it has one of that width, else a positive synthetic one.
fn gain_for(
    fixture: &Fixture,
    name: Option<&str>,
    cols: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    match name {
        Some(name) => fixture.f32s(name, cols),
        None => Ok(oracle::activations(cols, seed, 0.5)
            .iter()
            .map(|g| 1.0 + g.abs())
            .collect()),
    }
}

/// Run `launch` and hold the output to the oracle's band: the worst ratio of
/// the error to the band.
fn run_checked(h: &Harness<'_>, launch: &Launch<'_>) -> Result<f64, String> {
    let (effective, kind, base): (Vec<f64>, Kind, Option<&[f32]>) = match &launch.mode {
        ModeSpec::Store => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Store,
            None,
        ),
        ModeSpec::Norm { gain, eps, .. } => {
            (oracle::rms_norm(launch.x, gain, *eps), Kind::Norm, None)
        }
        ModeSpec::Residual => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Residual,
            Some(launch.y0),
        ),
    };
    let reference = oracle::reference(
        launch.quant,
        launch.weights,
        launch.rows,
        launch.cols,
        &effective,
    )?;
    let got = h.matvec(launch)?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let factor = oracle::band_factor(launch.quant, launch.cols / 256, kind);
    oracle::check(&got, &reference, factor, kind, base)
}

fn sentinel_rows(rows: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL_BITS); rows]
}

/// The note a mat-vec clause prints: each tensor's worst error over its band.
fn noted(parts: Vec<String>) -> Clause {
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}",
        parts.join(", ")
    )))
}

pub fn q4k_matvec(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
        Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048),
        Tensor::q4k("blk.0.ffn_down_shexp.weight", 512, 2048),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B34 + i as u64, 1.0);
        let worst = run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &sentinel_rows(t.rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    noted(parts)
}

pub fn q6k_matvec(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        Tensor::q6k("blk.2.attn_qkv.weight", 2048, 8192),
        Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
        Tensor::q6k("blk.3.attn_v.weight", 2048, 512),
        Tensor::q6k("output.weight", 2048, 4096),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B36 + i as u64, 1.0);
        let worst = run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &sentinel_rows(t.rows),
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
    let worst = run_checked(
        h,
        &Launch {
            quant: Quant::Q6K,
            rows: tail_rows,
            cols: 2048,
            weights: head,
            x: &x,
            y0: &sentinel_rows(tail_rows),
            mode: ModeSpec::Store,
        },
    )
    .map_err(|e| format!("output.weight rows 247167..248320: {e}"))?;
    parts.push(format!("output.weight tail {tail_rows}x2048 {worst:.3}"));
    noted(parts)
}

pub fn norm_on_load(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let cases = [
        (
            Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
            Some("blk.0.attn_norm.weight"),
        ),
        (Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048), None),
        (Tensor::q4k("blk.0.ffn_down_shexp.weight", 512, 2048), None),
        (
            Tensor::q6k("blk.3.attn_v.weight", 2048, 512),
            Some("blk.0.attn_norm.weight"),
        ),
        (
            Tensor::q6k("output.weight", 2048, 4096),
            Some("output_norm.weight"),
        ),
    ];
    let mut parts = Vec::new();
    for (i, (t, gain_name)) in cases.iter().enumerate() {
        let gain = gain_for(fixture, *gain_name, t.cols, 0x6A1 + i as u64)?;
        // A mean square far from one, so the scale matters.
        let x = oracle::activations(t.cols, 0x4B40 + i as u64, 3.7);
        let worst = run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &sentinel_rows(t.rows),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }

    // A zero input norms to zero with a positive epsilon: not a fault.
    let t = Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048);
    let gain = gain_for(fixture, None, t.cols, 0x6B0)?;
    run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
            x: &vec![0f32; t.cols],
            y0: &sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {eps}: {e}"))?;

    // A mean square near the epsilon, so the epsilon is part of the scale.
    let small = oracle::activations(t.cols, 0x4B4F, 1.0e-3);
    let worst = run_checked(
        h,
        &Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
            x: &small,
            y0: &sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("an input whose mean square is near eps {eps}: {e}"))?;
    parts.push(format!("small input {worst:.3}"));

    // Input the norm cannot take: each raises the norm site of the layer and
    // writes NaN to every row.
    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(t.cols, 1, 1.0), eps),
        (
            "an infinite input",
            oracle::activations(t.cols, 2, 1.0),
            eps,
        ),
        ("a zero input and a zero epsilon", vec![0f32; t.cols], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[t.cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant: t.quant,
            rows: t.rows,
            cols: t.cols,
            weights: t.bytes(fixture)?,
            x: &x,
            y0: &sentinel_rows(t.rows),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        })?;
        expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;
    noted(parts)
}

/// The fault word names `site` in `LAYER`, and only that site.
fn expect_fault(h: &Harness<'_>, what: &str, site: FaultSite) -> Result<(), String> {
    match h.fault()? {
        Some(f)
            if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            site.name()
        )),
        None => Err(format!("{what} raised nothing, not {}", site.name())),
    }
}

pub fn residual_epilogue(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let tensors = [
        Tensor::q4k("blk.0.attn_qkv.weight", 2048, 8192),
        Tensor::q4k("blk.0.ssm_out.weight", 4096, 2048),
        Tensor::q6k("blk.2.ffn_down_shexp.weight", 512, 2048),
        Tensor::q6k("blk.2.attn_qkv.weight", 2048, 8192),
    ];
    let mut parts = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let x = oracle::activations(t.cols, 0x4B50 + i as u64, 1.0);
        let resid = oracle::activations(t.rows, 0x7E5 + i as u64, 2.0);
        let worst = run_checked(
            h,
            &Launch {
                quant: t.quant,
                rows: t.rows,
                cols: t.cols,
                weights: t.bytes(fixture)?,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("{}: {e}", t.label()))?;
        parts.push(format!("{} {worst:.3}", t.label()));
    }
    noted(parts)
}

pub fn row_tail(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    for (quant, name) in [
        (Quant::Q4K, "blk.0.attn_qkv.weight"),
        (Quant::Q6K, "blk.2.attn_qkv.weight"),
    ] {
        let t = Tensor {
            name,
            quant,
            cols: 2048,
            rows: 8192,
        };
        let all = t.bytes(fixture)?;
        let row_bytes = 2048 / 256 * quant.block_bytes();
        let x = oracle::activations(2048, 0x4B60, 1.0);
        for rows in counts {
            run_checked(
                h,
                &Launch {
                    quant,
                    rows,
                    cols: 2048,
                    weights: &all[..rows * row_bytes],
                    x: &x,
                    y0: &sentinel_rows(rows),
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

#[cfg(metal_lib = "metallib")]
pub fn metallib_equals_source(
    device: &MetalDevice,
    stream: &MetalStream,
    fixture: &Fixture,
) -> Clause {
    use bloomery_metal::kernels::Kernels;
    use bloomery_metal::library::Library;

    let outputs = |library: &Library| -> Result<Vec<Vec<f32>>, String> {
        let kernels = Kernels::from_library(library).map_err(why)?;
        let h = Harness {
            device,
            stream,
            kernels: &kernels,
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
        let t = Tensor {
            name,
            quant,
            cols,
            rows,
        };
        let weights = t.bytes(fixture)?;
        let x = oracle::activations(cols, 0x51, 1.0);
        let gain = gain_for(fixture, Some("blk.0.attn_norm.weight"), cols, 0)?;
        for mode in [
            ModeSpec::Store,
            ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
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
    let (table, vocab) = fixture.table("token_embd.weight", 2048)?;
    out.push(run_embed(h, table, vocab, &[3, 99, 4096])?.0);
    Ok(out)
}

/// Embed `ids` from `table` (`vocab` rows of 2048 columns): the output rows
/// and the fault word's state after the launch.
fn run_embed(
    h: &Harness<'_>,
    table: &[u8],
    vocab: usize,
    ids: &[u32],
) -> Result<(Vec<f32>, Option<bloomery_device::Fault>), String> {
    let (device, stream) = (h.device, h.stream);
    let table = device.upload(stream, table).map_err(why)?;
    let ids_buffer = device.upload(stream, ids).map_err(why)?;
    let marker = 12345.0f32;
    let mut out = device
        .upload(stream, &vec![marker; ids.len() * 2048])
        .map_err(why)?;
    h.clear_fault()?;
    let params = EmbedParams {
        table: table
            .window(Elems(0), Elems(table.len().get()))
            .map_err(why)?,
        vocab,
        cols: 2048,
        ids: ids_buffer.window(Elems(0), Elems(ids.len())).map_err(why)?,
        out: out
            .window_mut(Elems(0), Elems(ids.len() * 2048))
            .map_err(why)?,
        fault: device.fault_word(),
        layer: LAYER,
    };
    h.encode(|encoder| h.kernels.embed(encoder, params))?;
    let mut rows = vec![0f32; ids.len() * 2048];
    out.read(stream, &mut rows).map_err(why)?;
    let fault = h.fault()?;
    h.clear_fault()?;
    Ok((rows, fault))
}

pub fn embed(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let (table, vocab) = fixture.table("token_embd.weight", 2048)?;
    let cols = 2048;

    // Ids in range: the dequantizer's bits, row by row, and a clean word.
    let good = [0u32, 1, 5, 4095, (vocab - 1) as u32];
    let (rows, fault) = run_embed(h, table, vocab, &good)?;
    if let Some(f) = fault {
        return Err(format!("ids in range raised {f}"));
    }
    for (t, &id) in good.iter().enumerate() {
        let want = oracle::q4k_row(table, cols, id as usize)?;
        for (k, (g, w)) in rows[t * cols..(t + 1) * cols].iter().zip(&want).enumerate() {
            if g.to_bits() != w.to_bits() {
                return Err(format!(
                    "id {id} value {k}: got {g:?}, the reader has {w:?}"
                ));
            }
        }
    }

    // Ids at and past the vocabulary: the site is raised, the row is NaN, and
    // the ids around them are still right.
    let mixed = [7u32, vocab as u32, 9, u32::MAX, (vocab - 1) as u32];
    let (rows, fault) = run_embed(h, table, vocab, &mixed)?;
    match fault {
        Some(f)
            if f.layer == LAYER
                && f.code == FaultSite::TokenId as u32
                && f.sites == 1 << (FaultSite::TokenId as u32) => {}
        Some(f) => {
            return Err(format!(
                "ids past the vocabulary raised {f}, not {} in layer {LAYER}",
                FaultSite::TokenId.name()
            ));
        }
        None => return Err("ids past the vocabulary raised nothing".to_string()),
    }
    for (t, &id) in mixed.iter().enumerate() {
        let row = &rows[t * cols..(t + 1) * cols];
        if (id as usize) < vocab {
            let want = oracle::q4k_row(table, cols, id as usize)?;
            if let Some(k) = row
                .iter()
                .zip(&want)
                .position(|(g, w)| g.to_bits() != w.to_bits())
            {
                return Err(format!(
                    "id {id} beside a bad id, value {k}: got {:?}",
                    row[k]
                ));
            }
        } else if let Some(k) = row.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "id {id} (vocab {vocab}) value {k} is {:?}, not NaN",
                row[k]
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "{} ids bit-equal to the reader, ids {} and {} raised token_id with NaN rows",
        good.len(),
        vocab,
        u32::MAX
    )))
}

/// One launch the host module must refuse: the shape it is told, the length of
/// each window it is given, and the words its refusal must name.
struct Refusal {
    what: &'static str,
    rows: usize,
    cols: usize,
    weights: usize,
    x: usize,
    y: usize,
    gain: Option<usize>,
    needle: &'static str,
}

pub fn matvec_refusals(h: &Harness<'_>) -> Clause {
    let device = h.device;
    let stream = h.stream;
    let rows = 8usize;
    let cols = 512usize;
    let bytes = rows * cols / 256 * Quant::Q4K.block_bytes();
    let weights = device.upload(stream, &vec![0u8; bytes + 64]).map_err(why)?;
    let x = device.upload(stream, &vec![0f32; cols + 4]).map_err(why)?;
    let mut y = device.upload(stream, &vec![1f32; rows + 4]).map_err(why)?;
    let gain = device.upload(stream, &vec![1f32; cols + 4]).map_err(why)?;

    let whole = Refusal {
        what: "",
        rows,
        cols,
        weights: bytes,
        x: cols,
        y: rows,
        gain: None,
        needle: "",
    };
    let cases = [
        Refusal {
            what: "a short weights window",
            weights: bytes - 1,
            needle: "weights",
            ..whole
        },
        Refusal {
            what: "a long weights window",
            weights: bytes + 1,
            needle: "weights",
            ..whole
        },
        Refusal {
            what: "a short x window",
            x: cols - 1,
            needle: "x window",
            ..whole
        },
        Refusal {
            what: "a short y window",
            y: rows - 1,
            needle: "y window",
            ..whole
        },
        Refusal {
            what: "a short gain window",
            gain: Some(cols - 1),
            needle: "gain",
            ..whole
        },
        Refusal {
            what: "columns that are not whole blocks",
            cols: 500,
            x: 500,
            needle: "whole blocks",
            ..whole
        },
        Refusal {
            what: "no rows",
            rows: 0,
            weights: 0,
            y: 0,
            needle: "empty",
            ..whole
        },
        Refusal {
            what: "no columns",
            cols: 0,
            weights: 0,
            x: 0,
            needle: "empty",
            ..whole
        },
    ];
    let count = cases.len();
    for case in cases {
        let refused = h.encode(|encoder| {
            let mode = match case.gain {
                Some(len) => Mode::NormOnLoad(NormOnLoad {
                    gain: gain.window(Elems(0), Elems(len))?,
                    eps: 1e-6,
                    fault: device.fault_word(),
                    layer: LAYER,
                }),
                None => Mode::Store,
            };
            h.kernels.matvec(
                encoder,
                MatvecParams {
                    quant: Quant::Q4K,
                    rows: case.rows,
                    cols: case.cols,
                    weights: weights.window(Elems(0), Elems(case.weights))?,
                    x: x.window(Elems(0), Elems(case.x))?,
                    y: y.window_mut(Elems(0), Elems(case.y))?,
                    mode,
                },
            )
        });
        match refused {
            Ok(()) => return Err(format!("{} was accepted", case.what)),
            Err(text) if !text.contains(case.needle) => {
                return Err(format!(
                    "{} was refused without naming {}: {text}",
                    case.what, case.needle
                ));
            }
            Err(_) => {}
        }
    }
    let mut back = vec![0f32; rows + 4];
    y.read(stream, &mut back).map_err(why)?;
    if back.iter().any(|&v| v != 1.0) {
        return Err(format!("a refused launch wrote: {back:?}"));
    }
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written"
    )))
}
