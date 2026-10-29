//! The segmented mat-vec clauses: every segment of one dispatch the same bits
//! as its tensor run alone, and the segment list the launch refuses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::kernels::{BLOCK_VALUES, NormOnLoad, Quant, SegParams, Segment};
use gguf::quant::GgmlType;

use super::clauses_synth;
use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle;
use super::synthetic;
use super::{Harness, Launch, ModeSpec, SENTINEL_BITS, SegLaunch};

/// The groups of the segmented launch: the tensors of one type and one width
/// that a layer's projections are.
struct SegGroup {
    quant: Quant,
    /// The layer's norm gain, the width's gain.
    gain: &'static str,
    /// The tensors: their names and their row counts.
    tensors: Vec<(&'static str, usize)>,
}

/// The four segments of an A4 attention P1 with a per-head gate, at the
/// hidden width and the head counts of a 16-head, 4-KV-head, head-128 layer:
/// q, k, v and the gate, whose 16 rows are the shortest segment a launch runs.
const GATE_P1: [(&str, usize); 4] = [
    ("attn_q", 2048),
    ("attn_k", 512),
    ("attn_v", 512),
    ("attn_gate", 16),
];

/// [`GATE_P1`] on seeded synthetic A4 weights, one dispatch: every segment the
/// same bits as its tensor run alone, the gate's 16 rows among them. The
/// fixture carries no tensor of these shapes, so the weights are drawn.
///
/// Mutant: `seg_weight_offset`, a segment reading the first segment's weights,
/// turns the group red (the groups of [`seg_bit`] before it are red first);
/// `seg_last_segment` does not reach the fourth segment, which no group of
/// [`seg_bit`] but this one has.
fn gate_p1(h: &Harness<'_>, eps: f32) -> Result<String, String> {
    let cols = 2048;
    let gain = clauses_synth::gain(cols, 0x5EA0);
    let x = oracle::activations(cols, 0x5EA1, 1.0);
    let owned: Vec<Vec<u8>> = GATE_P1
        .iter()
        .enumerate()
        .map(|(j, (_, n))| synthetic::matrix(Quant::A4, *n, cols, 0xA4_5E60 + j as u64))
        .collect();
    let segments: Vec<(usize, &[u8])> = GATE_P1
        .iter()
        .zip(&owned)
        .map(|((_, n), w)| (*n, w.as_slice()))
        .collect();
    let rows: usize = GATE_P1.iter().map(|(_, n)| n).sum();
    let got = h.seg(&SegLaunch {
        quant: Quant::A4,
        cols,
        segments: segments.clone(),
        x: &x,
        y0: &common::sentinel_rows(rows),
        gain: &gain,
        eps,
        layer: LAYER,
    })?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the A4 gate P1 raised {fault}"));
    }
    let mut at = 0usize;
    for (j, (n, bytes)) in segments.iter().enumerate() {
        let alone = h.matvec(&Launch {
            quant: Quant::A4,
            rows: *n,
            cols,
            weights: bytes,
            x: &x,
            y0: &common::sentinel_rows(*n),
            mode: ModeSpec::Norm {
                gain: &gain,
                eps,
                layer: LAYER,
            },
        })?;
        if let Some((k, a, b)) = common::first_diff(&got[at..at + n], &alone) {
            return Err(format!(
                "A4 {} segment {j} row {k}: the segment read {a:?}, the tensor alone read {b:?}",
                GATE_P1[j].0
            ));
        }
        at += n;
    }
    Ok(format!(
        "A4 {} {} segments, the last {} rows",
        GATE_P1[0].0,
        GATE_P1.len(),
        GATE_P1[GATE_P1.len() - 1].1
    ))
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone: the kernel does not change the order of a row's sum, and the
/// norm on the load is the norm the tensor alone gets.
pub fn seg_bit(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let cols = 2048;
    let groups = [
        SegGroup {
            quant: Quant::Q4K,
            gain: "blk.0.attn_norm.weight",
            tensors: vec![
                ("blk.0.attn_qkv.weight", 8192),
                ("blk.0.attn_gate.weight", 4096),
            ],
        },
        SegGroup {
            quant: Quant::Q4K,
            gain: "blk.1.attn_norm.weight",
            tensors: vec![
                ("blk.1.attn_q.weight", 8192),
                ("blk.1.attn_k.weight", 512),
                ("blk.1.attn_v.weight", 512),
            ],
        },
        SegGroup {
            quant: Quant::Q6K,
            gain: "blk.2.attn_norm.weight",
            tensors: vec![("blk.2.attn_qkv.weight", 8192)],
        },
    ];
    let mut parts = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let gain = fixture.f32s(group.gain, cols)?;
        let x = oracle::activations(cols, 0x5E6 + i as u64, 1.0);
        let mut segments: Vec<(usize, &[u8])> = Vec::new();
        let mut rows = 0usize;
        for (name, n) in &group.tensors {
            segments.push((*n, fixture.rows(name, group.quant.ggml(), cols, *n)?));
            rows += n;
        }
        let got = h.seg(&SegLaunch {
            quant: group.quant,
            cols,
            segments: segments.clone(),
            x: &x,
            y0: &common::sentinel_rows(rows),
            gain: &gain,
            eps,
            layer: LAYER,
        })?;
        if let Some(fault) = h.fault()? {
            return Err(format!("the segmented launch raised {fault}"));
        }
        let mut at = 0usize;
        for (j, (n, bytes)) in segments.iter().enumerate() {
            let alone = h.matvec(&Launch {
                quant: group.quant,
                rows: *n,
                cols,
                weights: bytes,
                x: &x,
                y0: &common::sentinel_rows(*n),
                mode: ModeSpec::Norm {
                    gain: &gain,
                    eps,
                    layer: LAYER,
                },
            })?;
            for (k, (a, b)) in got[at..at + n].iter().zip(alone.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!(
                        "{} segment {j} row {k}: the segment read {a:?}, the tensor alone read {b:?}",
                        group.tensors[j].0
                    ));
                }
            }
            at += n;
        }
        parts.push(format!(
            "{} {} segments",
            group.tensors[0].0,
            group.tensors.len()
        ));
    }
    parts.push(gate_p1(h, eps)?);
    common::noted(parts)
}

/// The buffers of the refusal table: one weights buffer of sixteen blocks,
/// and the windows every well-formed case of the table shares.
struct Refusals {
    cols: usize,
    bytes: usize,
    weights: bloomery_metal::MetalBuffer<u8>,
    x: bloomery_metal::MetalBuffer<f32>,
    y: bloomery_metal::MetalBuffer<f32>,
    gain: bloomery_metal::MetalBuffer<f32>,
}

/// The segment list of one case, one property of it bad, and the weights
/// window the launch reads.
#[derive(Clone, Copy)]
struct Args {
    segments: [Option<Segment>; 5],
    weights_len: usize,
    y_len: usize,
    /// The element the input's window starts at in its buffer.
    x_at: usize,
    /// The element the gain's window starts at in its buffer.
    gain_at: usize,
}

impl Args {
    /// The table's arguments: the weights window spans the whole buffer, so
    /// a case's own check refuses before the window check can.
    const fn whole(weights_len: usize, y_len: usize) -> Self {
        Self {
            segments: [
                Some(Segment {
                    cols: 2048,
                    rows: 8,
                    weight_blocks: 0,
                    out_offset: 0,
                }),
                None,
                None,
                None,
                None,
            ],
            weights_len,
            y_len,
            x_at: 0,
            gain_at: 0,
        }
    }

    fn list(&self) -> Vec<Segment> {
        self.segments.iter().filter_map(|s| *s).collect()
    }
}

fn refuse(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let segments = a.list();
    h.kernels.matvec_seg(
        encoder,
        SegParams {
            quant: Quant::Q4K,
            segments: &segments,
            weights: b.weights.window(Elems(0), Elems(a.weights_len))?,
            x: b.x.window(Elems(a.x_at), Elems(b.cols))?,
            y: b.y.window_mut(Elems(0), Elems(a.y_len))?,
            norm: NormOnLoad {
                gain: b.gain.window(Elems(a.gain_at), Elems(b.cols))?,
                eps: 1e-6,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
    weights_len: usize,
) -> BadArg<'a, Refusals> {
    let whole = Args::whole(weights_len, 64);
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

/// The bounds of a segment list: a list the kernel cannot run is refused by
/// name, and nothing is encoded for it. A dispatch of the most segments writes
/// every row and nothing past the last. An input or gain window that does not
/// start on 16 bytes is refused: deleting the `Float4` checks in `seg.rs`'s
/// `check` turns its cases red.
pub fn seg_bounds(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let cols = 2048;
    let gain = fixture.f32s("blk.0.attn_norm.weight", cols)?;
    let x = oracle::activations(cols, 0x5E7, 1.0);
    let names = [
        ("blk.0.attn_qkv.weight", 8192usize),
        ("blk.0.attn_gate.weight", 4096),
        ("blk.1.attn_q.weight", 8192),
        // A tail: not a multiple of the rows a threadgroup owns.
        ("blk.1.attn_k.weight", 511),
    ];
    let mut segments: Vec<(usize, &[u8])> = Vec::new();
    let mut rows = 0usize;
    for (name, n) in names {
        segments.push((n, fixture.rows(name, GgmlType::Q4_K, cols, n)?));
        rows += n;
    }
    let got = h.seg(&SegLaunch {
        quant: Quant::Q4K,
        cols,
        segments,
        x: &x,
        y0: &common::sentinel_rows(rows),
        gain: &gain,
        eps,
        layer: LAYER,
    })?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the segmented launch raised {fault}"));
    }
    if let Some(r) = got.iter().position(|v| v.to_bits() == SENTINEL_BITS) {
        return Err(format!("row {r} was not written"));
    }

    // The refusals: a list the kernel cannot run.
    let (device, stream) = (h.device, h.stream);
    let bytes = cols / BLOCK_VALUES * Quant::Q4K.block_bytes();
    let mut bufs = Refusals {
        cols,
        bytes,
        weights: device.upload(stream, &vec![0u8; bytes * 16]).map_err(why)?,
        x: device.upload(stream, &vec![0f32; cols + 4]).map_err(why)?,
        y: device.upload(stream, &vec![1f32; 64]).map_err(why)?,
        gain: device.upload(stream, &vec![1f32; cols + 4]).map_err(why)?,
    };
    let cases = vec![
        bad(
            "segments of different widths",
            "columns",
            |a| {
                a.segments[1] = Some(Segment {
                    cols: cols + BLOCK_VALUES,
                    rows: 8,
                    weight_blocks: 8,
                    out_offset: 8,
                });
            },
            h,
            bytes * 16,
        ),
        bad(
            "a segment of no rows",
            "no rows",
            |a| {
                a.segments[0] = Some(Segment {
                    cols: 2048,
                    rows: 0,
                    weight_blocks: 0,
                    out_offset: 0,
                });
            },
            h,
            bytes * 16,
        ),
        bad(
            "more segments than a dispatch runs",
            "at most",
            |a| {
                for (i, slot) in a.segments.iter_mut().enumerate() {
                    *slot = Some(Segment {
                        cols: 2048,
                        rows: 8,
                        weight_blocks: i * 8,
                        out_offset: i * 8,
                    });
                }
            },
            h,
            bytes * 16,
        ),
        bad(
            "segments whose outputs overlap",
            "overlap",
            |a| {
                a.segments[1] = Some(Segment {
                    cols: 2048,
                    rows: 8,
                    weight_blocks: 8,
                    out_offset: 4,
                });
            },
            h,
            bytes * 16,
        ),
        bad(
            "an x window one float into its buffer",
            "16-byte aligned",
            |a| {
                a.y_len = 8;
                a.x_at = 1;
            },
            h,
            8 * bytes,
        ),
        bad(
            "a gain window one float into its buffer",
            "16-byte aligned",
            |a| {
                a.y_len = 8;
                a.gain_at = 1;
            },
            h,
            8 * bytes,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| common::still(h, &b.y, |v| *v == 1.0);
    let count = common::refuse_launches(h, "matvec seg", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let a = Args::whole(8 * b.bytes, 8);
        refuse(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.y, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written, {rows} rows of a four-segment dispatch written"
    )))
}
