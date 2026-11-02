//! The launch of the M-row mat-vec family: the same weight read multiplied
//! into [`MROWS`] activation rows, the kernels `kernels/matvec_rows.metal`.
//!
//! Four launches: the plain residual mat-vec ([`matvec_m`]), the segmented
//! norm-on-load mat-vec of a layer's projections ([`matvec_seg_m`]), the
//! gate-up pair with its SwiGLU epilogue ([`pair_m`]) and the head's logits
//! with a per-row argmax ([`head_m`]). The segmented launch takes its input as
//! the 1-row launch does, on the load ([`LoadNorm::Gain`]) or already normed by
//! the hand-on that wrote it ([`LoadNorm::Scaled`]: M rows of the stream times
//! the reader's gain, then M scale words); the pair and the head take only the
//! hand-on's rows and scale words, and the hand-on over M rows is
//! [`norm_next_m`]. Row m of any of them is the 1-row kernel's bits on row m's
//! input alone — the same lane map, the same per-lane accumulation order, the
//! same SIMD reduction and the same epilogue; only the weights' read and their
//! unpack is shared, which the kernel's own order argument states. The
//! activations and the outputs are M rows at the width the launch names, the
//! rows arena's layout.
//!
//! Every M but [`MROWS`] is refused by name, as is a window the M rows do not
//! fit; nothing is encoded for a launch that is refused. The family holds A4
//! today, the kind an MLX 4-bit folder of a Bf16 scale pair opens as; another
//! quant gains it by gaining `partial_rows` and the kernel instantiations.

use bloomery_device::{DeviceError, Elems, LAYER_NONE, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::geom::ExpertsGeom;
use crate::library::{Library, Pipeline};

use super::head::{HEAD_SIMD_GROUPS, HEAD_THREADS, MAX_ROWS};
use super::matvec::{LoadNorm, Quant};
use super::seg::Segment;
use super::{Kernels, MAX_SEGMENTS, args_bytes, check_weights_align, check_window_len, pack, word};

pub use crate::geometry::MROWS;

const MATVEC: &str = "matvec mrows";
const SEG: &str = "matvec seg mrows";
const PAIR: &str = "pair mrows";
const HEAD: &str = "head mrows";
const NEXT: &str = "norm next mrows";

fn refuse(what: &'static str, detail: impl Into<String>) -> DeviceError {
    super::range(what, detail)
}

/// `m` refused by name unless it is the one M the family holds entries of.
fn check_m(what: &'static str, m: usize) -> Result<(), DeviceError> {
    if m == MROWS {
        return Ok(());
    }
    Err(refuse(what, format!("M is {m}; the family runs {MROWS}")))
}

/// The one quant whose decode owner holds `partial_rows`, the family's own.
fn a4(what: &'static str, quant: Quant) -> Result<(), DeviceError> {
    if quant == Quant::A4 {
        return Ok(());
    }
    Err(refuse(
        what,
        format!("{} has no M-row entry; the family holds A4", quant.name()),
    ))
}

/// The norm's windows against `m` input rows of `cols` values: a gain of
/// `cols` values on a `float4` boundary, or one scale word a row.
fn check_norm(
    what: &'static str,
    norm: &LoadNorm<'_>,
    cols: usize,
    m: usize,
) -> Result<(), DeviceError> {
    match norm {
        LoadNorm::Gain { .. } => norm.check(what, cols),
        LoadNorm::Scaled { scale } => check_window_len(what, "scale", scale.len(), m),
    }
}

/// A window the kernels read as `float4`s: its first byte is a multiple of 16
/// in its buffer.
fn float4_aligned(what: &'static str, name: &str, start: usize) -> Result<(), DeviceError> {
    check_weights_align(what, name, start, super::WeightAlign::Float4)
}

/// The plain M-row residual mat-vec: `y[m, r] += W[r, :] . x[m, :]`, M rows of
/// `rows` outputs over `cols` columns each.
pub struct MatvecMParams<'w> {
    /// The weight type. A4 today; the launch refuses the others by name.
    pub quant: Quant,
    /// The activation rows, [`MROWS`].
    pub m: usize,
    /// The output rows an activation row takes.
    pub rows: usize,
    /// The input columns, a multiple of the type's block values.
    pub cols: usize,
    /// The weights' bytes: `rows * cols / values_per_block()` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The inputs, `m` rows of `cols` values, starting on a 16-byte boundary
    /// of their buffer.
    pub x: Window<'w, f32, MetalDevice>,
    /// The outputs, `m` rows of `rows` values, which the launch adds into.
    pub y: WindowMut<'w, f32, MetalDevice>,
}

/// The segmented M-row mat-vec: the row segments of up to four tensors of one
/// type and one width, the norm on the load, each activation row's segments
/// at their own output offsets inside that row's outputs.
pub struct SegMParams<'w> {
    /// The weight type. A4 today; the launch refuses the others by name.
    pub quant: Quant,
    /// The activation rows, [`MROWS`].
    pub m: usize,
    /// The segments, one to [`MAX_SEGMENTS`] of them.
    pub segments: &'w [Segment],
    /// The weights of every segment, one buffer.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The inputs, `m` rows of the segments' columns (scaled: each row times
    /// the norm's gain).
    pub x: Window<'w, f32, MetalDevice>,
    /// The outputs, `m` rows of the segments' whole span.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The norm the load applies, or the scale words of an input already
    /// normed: `m` of them, one a row.
    pub norm: LoadNorm<'w>,
    /// The fault word a norm's raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The M-row gate-up pair: the instance's slots, each
/// `silu(gate . x̂[m]) * (up . x̂[m])` over `cols` inputs, `x̂[m]` the normed
/// row m the hand-on wrote. A shared expert of another type than the stacks'
/// has no M-row entry and is refused by name.
pub struct PairMParams<'w> {
    /// The weight type of the stacks. A4 today; the launch refuses the others
    /// by name.
    pub quant: Quant,
    /// The type of the shared expert's gate and up: `quant`, the one type the
    /// M-row pair reads.
    pub shared_quant: Quant,
    /// The activation rows, [`MROWS`].
    pub m: usize,
    /// The slots. A launch runs the instance's (`ExpertsGeom::slots`).
    pub slots: usize,
    /// The expert intermediate width. A launch runs the instance's.
    pub k: usize,
    /// The input columns, a positive multiple of the type's block values.
    pub cols: usize,
    /// Experts in the stack. An id at or past this is `ExpertId`.
    pub n_experts: usize,
    /// The routed ids, int32.
    pub ids: Window<'w, i32, MetalDevice>,
    /// The gate stack: `n_experts` experts of `k` by `cols`.
    pub gate: Window<'w, u8, MetalDevice>,
    /// The up stack, the gate's shape.
    pub up: Window<'w, u8, MetalDevice>,
    /// The shared expert's gate, given exactly when the instance's experts
    /// have a shared slot.
    pub shared_gate: Option<Window<'w, u8, MetalDevice>>,
    /// The shared expert's up, given with the gate.
    pub shared_up: Option<Window<'w, u8, MetalDevice>>,
    /// The inputs, `m` rows of `cols` values, each row times the norm's gain.
    pub x: Window<'w, f32, MetalDevice>,
    /// The `m` scale words of the input, one a row; a NaN word (its hand-on
    /// found the norm unusable) makes every slot of its row NaN.
    pub scale: Window<'w, f32, MetalDevice>,
    /// The outputs: `m` rows at the slots' whole span, slot `s` of row `m` at
    /// `m * slots * out_stride + s * out_stride`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Floats from one slot's output to the next, within an activation row.
    pub out_stride: usize,
    /// The fault word an id past the stack or a norm that cannot take its row
    /// raises into.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The M-row head: M rows of logits over the hand-on's rows, and one argmax a
/// row over that row's logits, each into its own pick word.
pub struct HeadMParams<'w> {
    /// The weight type. A4 today; the launch refuses the others by name.
    pub quant: Quant,
    /// The activation rows, [`MROWS`].
    pub m: usize,
    /// The logits rows an activation row takes: the vocabulary.
    pub rows: usize,
    /// The columns, a positive multiple of the type's block values.
    pub k: usize,
    /// The SIMD groups in each row's argmax threadgroup: the pick's own
    /// count, [`HEAD_SIMD_GROUPS`].
    pub simdgroups: usize,
    /// The weights, `rows * (k / values_per_block())` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The inputs, `m` rows of `k` values, each row times the norm's gain.
    pub x: Window<'w, f32, MetalDevice>,
    /// The `m` scale words of the input, one a row; a NaN word makes every
    /// logit of its row NaN.
    pub scale: Window<'w, f32, MetalDevice>,
    /// The logits, `m` rows of `rows` values.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// The winning row of each activation row, one `u32` each.
    pub step_ids: WindowMut<'w, u32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The plain residual's hand-on over M rows: row `r` of the stream (which the
/// out projection already added into) times the next launch's gain into row
/// `r` of `out`, then the next launch's scale of each row, summed in that
/// launch's own lane order, into the word `m * cols + r` of `out`. Row `r` is
/// the 1-row hand-on's ([`Kernels::norm_next`]) bits on that row alone; a row
/// the norm cannot take raises the norm site in the next launch's layer and
/// hands on a NaN scale, the other rows' words unchanged. The M-row readers
/// take `out`'s rows as `x` and its scale words as their scale.
pub struct NormNextMParams<'w> {
    /// The next launch's weight type, whose lane order the scale is summed in.
    /// A4 today; the launch refuses the others by name.
    pub quant: Quant,
    /// The rows, [`MROWS`].
    pub m: usize,
    /// The stream, `m` rows of the gain's length, on a 16-byte boundary.
    pub x: Window<'w, f32, MetalDevice>,
    /// The next norm's gain: one row's values, on a 16-byte boundary.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The next norm's epsilon.
    pub eps: f32,
    /// The layer the next norm's raise is recorded in.
    pub layer: u32,
    /// The rows times `gain`, then the `m` scale words, on a 16-byte boundary.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
}

struct MatvecMShape {
    nblocks: u32,
    rows: u32,
    groups: usize,
}

/// The plain M-row mat-vec's shape, refused by name as the 1-row launch's is,
/// with the windows holding the M rows.
fn check_matvec(
    p: &MatvecMParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<MatvecMShape, DeviceError> {
    a4(MATVEC, p.quant)?;
    check_m(MATVEC, p.m)?;
    if p.rows == 0 || p.cols == 0 {
        return Err(refuse(
            MATVEC,
            format!("{} rows by {} columns is empty", p.rows, p.cols),
        ));
    }
    if i32::try_from(p.cols).is_err() {
        return Err(refuse(
            MATVEC,
            format!("{} columns pass the kernels' signed 32-bit index", p.cols),
        ));
    }
    if !p.cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(
            MATVEC,
            format!(
                "{} columns are not whole blocks of {}",
                p.cols,
                p.quant.values_per_block()
            ),
        ));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks).ok_or_else(|| {
        refuse(
            MATVEC,
            format!("{} rows of {} columns wrap", p.rows, p.cols),
        )
    })?;
    check_window_len(MATVEC, "weights", p.weights.len(), weight_bytes)?;
    check_window_len(MATVEC, "x", p.x.len(), p.m * p.cols)?;
    check_window_len(MATVEC, "y", p.y.len(), p.m * p.rows)?;
    let weights_start = p.weights.byte_range()?.start.get();
    check_weights_align(MATVEC, "weights", weights_start, p.quant.weight_align())?;
    let x_start = p.x.byte_range()?.start.get();
    float4_aligned(MATVEC, "x", x_start)?;
    Ok(MatvecMShape {
        nblocks: word(MATVEC, "the block count", nblocks)?,
        rows: word(MATVEC, "the row count", p.rows)?,
        groups: p.rows.div_ceil(simdgroups * rows_per_simd_group),
    })
}

/// One segment, as the kernel's argument block holds it.
#[derive(Clone, Copy)]
struct SegArg {
    rows: u32,
    weight_blocks: u32,
    out: u32,
    first_group: u32,
}

struct SegMShape {
    nblocks: u32,
    segments: [SegArg; MAX_SEGMENTS],
    nseg: u32,
    /// The outputs' whole span, the stride between the M rows.
    out_end: u32,
    groups: usize,
}

/// The segmented M-row launch's shape, refused by name as the 1-row
/// segmented launch's is, with the windows holding the M rows.
fn check_seg(
    p: &SegMParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<SegMShape, DeviceError> {
    a4(SEG, p.quant)?;
    check_m(SEG, p.m)?;
    if p.segments.is_empty() {
        return Err(refuse(SEG, "the dispatch has no segment"));
    }
    if p.segments.len() > MAX_SEGMENTS {
        return Err(refuse(
            SEG,
            format!(
                "{} segments, a dispatch runs at most {MAX_SEGMENTS}",
                p.segments.len()
            ),
        ));
    }
    let cols = p.segments[0].cols;
    for (i, s) in p.segments.iter().enumerate() {
        if s.cols != cols {
            return Err(refuse(
                SEG,
                format!(
                    "segment {i} is over {} columns, the first over {cols}",
                    s.cols
                ),
            ));
        }
        if s.rows == 0 {
            return Err(refuse(SEG, format!("segment {i} has no rows")));
        }
    }
    if !cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(
            SEG,
            format!(
                "{cols} columns are not whole blocks of {}",
                p.quant.values_per_block()
            ),
        ));
    }
    let nblocks = cols / p.quant.values_per_block();
    for (j, b) in p.segments.iter().enumerate() {
        for a in &p.segments[..j] {
            if a.out_offset < b.out_offset + b.rows && b.out_offset < a.out_offset + a.rows {
                return Err(refuse(
                    SEG,
                    format!(
                        "the output of the segments at {} and {} overlaps",
                        a.out_offset.min(b.out_offset),
                        a.out_offset.max(b.out_offset)
                    ),
                ));
            }
        }
    }
    let mut weight_end = 0usize;
    for s in p.segments {
        let span = super::weight_bytes(p.quant, s.rows, nblocks)
            .ok_or_else(|| refuse(SEG, format!("a segment of {} rows wraps", s.rows)))?;
        let end = s
            .weight_blocks
            .checked_mul(p.quant.block_bytes())
            .and_then(|at| at.checked_add(span))
            .ok_or_else(|| refuse(SEG, format!("a segment at block {} wraps", s.weight_blocks)))?;
        weight_end = weight_end.max(end);
    }
    check_window_len(SEG, "weights", p.weights.len(), weight_end)?;
    check_window_len(SEG, "x", p.x.len(), p.m * cols)?;
    check_norm(SEG, &p.norm, cols, p.m)?;
    let out_end = p
        .segments
        .iter()
        .map(|s| s.out_offset + s.rows)
        .max()
        .unwrap_or(0);
    check_window_len(SEG, "y", p.y.len(), p.m * out_end)?;
    let weights_start = p.weights.byte_range()?.start.get();
    check_weights_align(SEG, "weights", weights_start, p.quant.weight_align())?;
    let x_start = p.x.byte_range()?.start.get();
    float4_aligned(SEG, "x", x_start)?;

    let mut segments = [SegArg {
        rows: 0,
        weight_blocks: 0,
        out: 0,
        first_group: 0,
    }; MAX_SEGMENTS];
    let mut groups = 0usize;
    for (i, s) in p.segments.iter().enumerate() {
        segments[i] = SegArg {
            rows: word(SEG, "a segment's rows", s.rows)?,
            weight_blocks: word(SEG, "a segment's weight blocks", s.weight_blocks)?,
            out: word(SEG, "a segment's output offset", s.out_offset)?,
            first_group: word(SEG, "a segment's first threadgroup", groups)?,
        };
        groups += s.rows.div_ceil(simdgroups * rows_per_simd_group);
    }
    Ok(SegMShape {
        nblocks: word(SEG, "the block count", nblocks)?,
        segments,
        nseg: word(SEG, "the segment count", p.segments.len())?,
        out_end: word(SEG, "the output span", out_end)?,
        groups,
    })
}

/// The segmented M-row launch's argument words: the header's five and every
/// segment's four.
fn seg_m_args(shape: &SegMShape, layer: u32, eps: f32) -> [u32; 5 + 4 * MAX_SEGMENTS] {
    let mut words = [0u32; 5 + 4 * MAX_SEGMENTS];
    words[0] = shape.nblocks;
    words[1] = layer;
    words[2] = eps.to_bits();
    words[3] = shape.nseg;
    words[4] = shape.out_end;
    for (i, s) in shape.segments.iter().enumerate() {
        let at = 5 + 4 * i;
        words[at] = s.rows;
        words[at + 1] = s.weight_blocks;
        words[at + 2] = s.out;
        words[at + 3] = s.first_group;
    }
    words
}

/// The pair's y window: `m` rows pitched at the slots' whole span, the last
/// row ending at its own last slot's outputs, `None` when it wraps.
fn pair_y_len(m: usize, slots: usize, out_stride: usize, k: usize) -> Option<usize> {
    let row = (slots - 1).checked_mul(out_stride)?.checked_add(k)?;
    row.checked_add((m - 1).checked_mul(slots)?.checked_mul(out_stride)?)
}

struct PairMShape {
    nblocks: u32,
    rows: u32,
    n_experts: u32,
    stride: u32,
    groups_x: usize,
}

/// The M-row pair's shape, refused by name as the 1-row pair's is, with the
/// windows holding the M rows.
fn check_pair(
    g: &ExpertsGeom,
    p: &PairMParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<PairMShape, DeviceError> {
    a4(PAIR, p.quant)?;
    check_m(PAIR, p.m)?;
    if p.shared_quant != p.quant {
        return Err(refuse(
            PAIR,
            format!(
                "stacks of {} and a shared expert of {}: no M-row entry",
                p.quant.name(),
                p.shared_quant.name()
            ),
        ));
    }
    if p.slots != g.slots {
        return Err(refuse(
            PAIR,
            format!("a launch runs {} slots, not {}", g.slots, p.slots),
        ));
    }
    if p.k != g.k {
        return Err(refuse(
            PAIR,
            format!("the expert width is {}, not {}", p.k, g.k),
        ));
    }
    if p.n_experts == 0 {
        return Err(refuse(PAIR, "the expert stack is empty"));
    }
    if p.ids.len().get() == 0 {
        return Err(refuse(PAIR, "the id list is empty"));
    }
    if p.out_stride < g.k {
        return Err(refuse(PAIR, "the slot outputs overlap"));
    }
    if !p.cols.is_multiple_of(p.quant.values_per_block()) || p.cols == 0 {
        return Err(refuse(
            PAIR,
            format!(
                "{} columns are not whole blocks of {}",
                p.cols,
                p.quant.values_per_block()
            ),
        ));
    }
    let shared = match (p.shared_gate, p.shared_up, g.has_shared) {
        (None, None, false) => None,
        (Some(gate), Some(up), true) => Some((gate, up)),
        (None, None, true) => {
            return Err(refuse(
                PAIR,
                "no shared expert, and the experts run a shared slot",
            ));
        }
        (Some(_), Some(_), false) => {
            return Err(refuse(
                PAIR,
                "a shared expert, and the experts run no shared slot",
            ));
        }
        _ => {
            return Err(refuse(
                PAIR,
                "a shared expert's gate without its up, or its up without its gate",
            ));
        }
    };
    let nblocks = p.cols / p.quant.values_per_block();
    let stack = super::weight_bytes(p.quant, g.k, nblocks)
        .and_then(|bytes| bytes.checked_mul(p.n_experts))
        .ok_or_else(|| {
            refuse(
                PAIR,
                format!("{} experts of {} columns wrap", p.n_experts, p.cols),
            )
        })?;
    check_window_len(PAIR, "gate", p.gate.len(), stack)?;
    check_window_len(PAIR, "up", p.up.len(), stack)?;
    if let Some((gate, up)) = shared {
        let shared_bytes = super::weight_bytes(p.quant, g.k, nblocks)
            .ok_or_else(|| refuse(PAIR, "one shared expert wraps".to_string()))?;
        check_window_len(PAIR, "shared gate", gate.len(), shared_bytes)?;
        check_window_len(PAIR, "shared up", up.len(), shared_bytes)?;
        let gate_start = gate.byte_range()?.start.get();
        check_weights_align(PAIR, "shared gate", gate_start, p.quant.weight_align())?;
        let up_start = up.byte_range()?.start.get();
        check_weights_align(PAIR, "shared up", up_start, p.quant.weight_align())?;
    }
    check_window_len(PAIR, "x", p.x.len(), p.m * p.cols)?;
    check_window_len(PAIR, "scale", p.scale.len(), p.m)?;
    check_window_len(PAIR, "ids", p.ids.len(), g.routed_ids)?;
    let y_len = pair_y_len(p.m, g.slots, p.out_stride, g.k)
        .ok_or_else(|| refuse(PAIR, "the output length wraps".to_string()))?;
    check_window_len(PAIR, "y", p.y.len(), y_len)?;
    let gate_start = p.gate.byte_range()?.start.get();
    check_weights_align(PAIR, "gate", gate_start, p.quant.weight_align())?;
    let up_start = p.up.byte_range()?.start.get();
    check_weights_align(PAIR, "up", up_start, p.quant.weight_align())?;
    let x_start = p.x.byte_range()?.start.get();
    float4_aligned(PAIR, "x", x_start)?;
    Ok(PairMShape {
        nblocks: word(PAIR, "the block count", nblocks)?,
        rows: word(PAIR, "the row count", g.k)?,
        n_experts: word(PAIR, "the expert count", p.n_experts)?,
        stride: word(PAIR, "the output stride", p.out_stride)?,
        // A SIMD group owns the launch's weight rows, a gate and an up to an
        // output, in every activation row.
        groups_x: g.k.div_ceil(simdgroups * rows_per_simd_group / 2),
    })
}

/// The logits rows of an activation row, refused by name when the pick cannot
/// read them: none, past the pick's 32-bit loop, or not whole `float4` quads
/// (every activation row's own pick window then starts on 16 bytes).
fn head_rows(rows: usize) -> Result<(), DeviceError> {
    if rows == 0 {
        return Err(refuse(HEAD, "the row count is 0"));
    }
    if rows > MAX_ROWS {
        return Err(refuse(
            HEAD,
            format!("the row count {rows} passes the pick's {MAX_ROWS}"),
        ));
    }
    if !rows.is_multiple_of(4) {
        return Err(refuse(
            HEAD,
            format!("the logits rows are {rows}, the argmax reads them as float4 quads"),
        ));
    }
    Ok(())
}

struct HeadMShape {
    nblocks: u32,
    rows: u32,
    groups: usize,
}

/// The M-row head's shape, refused by name as the 1-row head's is, with the
/// windows holding the M rows. The logits rows a whole number of `float4`
/// quads: every row's own argmax window starts on 16 bytes.
fn check_head(p: &HeadMParams<'_>, rows_per_group: usize) -> Result<HeadMShape, DeviceError> {
    a4(HEAD, p.quant)?;
    check_m(HEAD, p.m)?;
    if p.k == 0 || !p.k.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(
            HEAD,
            format!(
                "K is {}, the kernel reads a positive multiple of {}",
                p.k,
                p.quant.values_per_block()
            ),
        ));
    }
    head_rows(p.rows)?;
    if p.simdgroups != HEAD_SIMD_GROUPS {
        return Err(refuse(
            HEAD,
            format!(
                "the SIMD-group count is {}, the pick runs {HEAD_SIMD_GROUPS}",
                p.simdgroups
            ),
        ));
    }
    let nblocks = p.k / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(HEAD, format!("{} rows of {} columns wrap", p.rows, p.k)))?;
    check_window_len(HEAD, "x", p.x.len(), p.m * p.k)?;
    check_window_len(HEAD, "scale", p.scale.len(), p.m)?;
    check_window_len(HEAD, "logits", p.logits.len(), p.m * p.rows)?;
    check_window_len(HEAD, "step ids", p.step_ids.len(), p.m)?;
    check_window_len(HEAD, "weights", p.weights.len(), weight_bytes)?;
    let weights_start = p.weights.byte_range()?.start.get();
    check_weights_align(HEAD, "weights", weights_start, p.quant.weight_align())?;
    let x_start = p.x.byte_range()?.start.get();
    float4_aligned(HEAD, "x", x_start)?;
    let logits_start = p.logits.byte_range()?.start.get();
    float4_aligned(HEAD, "logits", logits_start)?;
    Ok(HeadMShape {
        nblocks: word(HEAD, "the block count", nblocks)?,
        rows: word(HEAD, "the row count", p.rows)?,
        groups: p.rows.div_ceil(rows_per_group),
    })
}

/// The M-row hand-on's argument words: its columns and rows, the next norm's
/// epsilon and layer.
fn check_next(p: &NormNextMParams<'_>) -> Result<[u32; 4], DeviceError> {
    a4(NEXT, p.quant)?;
    check_m(NEXT, p.m)?;
    let cols = p.gain.len().get();
    if cols == 0 || !cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(
            NEXT,
            format!(
                "cols {cols} are not whole blocks of {}",
                p.quant.values_per_block()
            ),
        ));
    }
    if !p.eps.is_finite() {
        return Err(refuse(
            NEXT,
            format!("the next eps {} is not finite", p.eps),
        ));
    }
    let rows_words =
        p.m.checked_mul(cols)
            .ok_or_else(|| refuse(NEXT, format!("{} rows of {cols} values wrap", p.m)))?;
    check_window_len(NEXT, "x", p.x.len(), rows_words)?;
    check_window_len(NEXT, "out", p.out.len(), rows_words + p.m)?;
    for (name, start) in [
        ("x", p.x.byte_range()?.start.get()),
        ("gain", p.gain.byte_range()?.start.get()),
        ("out", p.out.byte_range()?.start.get()),
    ] {
        float4_aligned(NEXT, name, start)?;
    }
    Ok([
        word(NEXT, "cols", cols)?,
        word(NEXT, "rows", p.m)?,
        p.eps.to_bits(),
        p.layer,
    ])
}

/// The pipelines of the M-row family: one per entry, at [`MROWS`] rows.
pub(crate) struct Pipelines {
    resid: Pipeline,
    seg: Pipeline,
    seg_scaled: Pipeline,
    pair: Pipeline,
    head: Pipeline,
    /// The hand-on over M rows, whose row count is a launch's own.
    next: Pipeline,
}

impl Pipelines {
    /// `simdgroups` is the mat-vec function constant, as every 1-row
    /// family's.
    pub(crate) fn new(library: &Library, simdgroups: u32) -> Result<Self, DeviceError> {
        let of = |name: &str| library.pipeline_with_simdgroups(name, simdgroups);
        Ok(Self {
            resid: of("a4_matvec_resid_m2")?,
            seg: of("a4_matvec_seg_m2")?,
            seg_scaled: of("a4_matvec_seg_scaled_m2")?,
            pair: of("a4_pair_swiglu_scaled_m2")?,
            head: of("a4_head_logits_scaled_m2")?,
            next: library.pipeline("a4_norm_next_m")?,
        })
    }
}

impl Kernels {
    /// Encode the hand-on of `m` rows: one dispatch, a threadgroup a row of
    /// the router's thread count. Nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn norm_next_m(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: NormNextMParams<'_>,
    ) -> Result<(), DeviceError> {
        let args = args_bytes(check_next(&p)?);
        encoder.set_pipeline(&self.matvec_rows.next);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(p.m, 1, self.router_threads())
    }

    /// Encode one plain M-row residual mat-vec dispatch on `encoder`. Nothing
    /// is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec_m(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: MatvecMParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_matvec(&p, self.simdgroups, self.rows_per_simd_group)?;
        let args = args_bytes([
            shape.rows,
            shape.nblocks,
            LAYER_NONE,
            0f32.to_bits(),
            shape.rows,
        ]);
        encoder.set_pipeline(&self.matvec_rows.resid);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        let threads = self.threads_per_group();
        slot.dispatch_groups(shape.groups, 1, threads)
    }

    /// Encode one segmented M-row mat-vec dispatch on `encoder`, the norm on
    /// the load. Nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec_seg_m(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: SegMParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_seg(&p, self.simdgroups, self.rows_per_simd_group)?;
        let args = args_bytes(seg_m_args(&shape, p.layer, p.norm.eps()));
        encoder.set_pipeline(if p.norm.scaled() {
            &self.matvec_rows.seg_scaled
        } else {
            &self.matvec_rows.seg
        });
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(Binding::of(p.norm.window())?)?;
        slot.set(p.fault.binding())?;
        let threads = self.threads_per_group();
        slot.dispatch_groups(shape.groups, 1, threads)
    }

    /// Encode one M-row gate-up pair on `encoder` over the hand-on's rows: the
    /// instance's slots, one dispatch. Nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn pair_m(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: PairMParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_pair(
            &self.geom.experts,
            &p,
            self.simdgroups,
            self.rows_per_simd_group,
        )?;
        // The scaled pair reads no epsilon.
        let args = args_bytes([
            shape.rows,
            shape.nblocks,
            shape.n_experts,
            p.layer,
            shape.stride,
            0f32.to_bits(),
        ]);
        encoder.set_pipeline(&self.matvec_rows.pair);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.gate)?)?;
        slot.set(Binding::of(&p.up)?)?;
        // An instance with no shared slot binds the stacks in their places,
        // which the kernel does not read.
        slot.set(Binding::of(p.shared_gate.as_ref().unwrap_or(&p.gate))?)?;
        slot.set(Binding::of(p.shared_up.as_ref().unwrap_or(&p.up))?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.scale)?)?;
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(
            shape.groups_x,
            self.geom.experts.slots,
            self.threads_per_group(),
        )
    }

    /// Encode the M-row head on `encoder`: the logits dispatch, a barrier
    /// that closes its group, and one argmax an activation row over that
    /// row's logits. Nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head_m(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadMParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_head(&p, self.simdgroups * self.rows_per_simd_group)?;
        // The scaled head reads no epsilon.
        let args = args_bytes([
            shape.rows,
            shape.nblocks,
            p.layer,
            0f32.to_bits(),
            shape.rows,
        ]);
        encoder.set_pipeline(&self.matvec_rows.head);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.scale)?)?;
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(shape.groups, 1, self.threads_per_group())?;
        encoder.memory_barrier();
        // The pick over each row's own logits, the 1-row entry's own
        // threadgroup, one dispatch an activation row, the rows taken from
        // the front as each dispatch consumes its own.
        let mut logits = p.logits;
        let mut picks = p.step_ids;
        for _ in 0..p.m {
            let (row, rest) = logits.split_at(Elems(p.rows))?;
            logits = rest;
            let (one, prest) = picks.split_at(Elems(1))?;
            picks = prest;
            let pick_args = args_bytes([shape.rows, p.layer]);
            encoder.set_pipeline(&self.head_pick);
            let mut slot = Bindings::new(encoder);
            slot.set(Binding::of_mut(&row)?)?;
            slot.set(Binding::of_mut(&one)?)?;
            slot.set(p.fault.binding())?;
            slot.set(Binding::bytes(pack(&pick_args))?)?;
            slot.dispatch_groups(1, 1, HEAD_THREADS)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The family runs [`MROWS`] rows: every other count is refused by name,
    /// saying the count it runs.
    #[test]
    fn the_family_runs_two_rows() {
        assert_eq!(MROWS, 2);
        assert!(check_m(MATVEC, 2).is_ok());
        for m in [0, 1, 3, 4, 5, 8, usize::MAX] {
            match check_m(MATVEC, m) {
                Ok(()) => panic!("M {m} is accepted"),
                Err(DeviceError::Range { what, detail }) => {
                    assert_eq!(what, MATVEC);
                    assert_eq!(detail, format!("M is {m}; the family runs 2"));
                }
                Err(other) => panic!("M {m} is refused as {other}"),
            }
        }
    }

    /// The head's logits rows: whole quads, none past the pick's loop, never
    /// none; every refusal names the count.
    #[test]
    fn the_head_rows_are_whole_quads_inside_the_picks_loop() {
        assert!(head_rows(4).is_ok());
        assert!(head_rows(248_320).is_ok());
        assert!(head_rows(MAX_ROWS - MAX_ROWS % 4).is_ok());
        for (rows, needle) in [
            (0usize, "the row count is 0"),
            (6, "float4 quads"),
            (248_322, "float4 quads"),
            (MAX_ROWS + 4 - MAX_ROWS % 4, "passes the pick's"),
            (usize::MAX, "passes the pick's"),
        ] {
            match head_rows(rows) {
                Ok(()) => panic!("{rows} rows are accepted"),
                Err(DeviceError::Range { what, detail }) => {
                    assert_eq!(what, HEAD);
                    assert!(detail.contains(needle), "{rows} rows: {detail}");
                }
                Err(other) => panic!("{rows} rows are refused as {other}"),
            }
        }
    }

    /// The segmented launch's argument words: the header's five, the stride
    /// between the M rows among them, then each segment's four, the unused
    /// tail of the table zero.
    #[test]
    fn the_segmented_arguments_are_the_header_then_the_segments() {
        let mut segments = [SegArg {
            rows: 0,
            weight_blocks: 0,
            out: 0,
            first_group: 0,
        }; MAX_SEGMENTS];
        segments[0] = SegArg {
            rows: 10,
            weight_blocks: 0,
            out: 0,
            first_group: 0,
        };
        segments[1] = SegArg {
            rows: 6,
            weight_blocks: 800,
            out: 10,
            first_group: 3,
        };
        let shape = SegMShape {
            nblocks: 80,
            segments,
            nseg: 2,
            out_end: 16,
            groups: 5,
        };
        let words = seg_m_args(&shape, 7, 1e-6);
        assert_eq!(
            words[..5],
            [80, 7, 1e-6f32.to_bits(), 2, 16],
            "nblocks, layer, eps, nseg, the output stride"
        );
        assert_eq!(words[5..9], [10, 0, 0, 0]);
        assert_eq!(words[9..13], [6, 800, 10, 3]);
        assert!(words[13..].iter().all(|w| *w == 0));
        assert_eq!(words.len(), 5 + 4 * MAX_SEGMENTS);
    }

    /// The pair's output window: the last activation row ends at its own last
    /// slot's outputs, so the rows pitch the slots' whole span without
    /// holding the tail the stride's rounding leaves.
    #[test]
    fn the_pair_window_holds_the_rows_at_the_slots_span() {
        // One slot of the expert width: the plain M rows of k.
        assert_eq!(pair_y_len(2, 1, 512, 512), Some(1024));
        // A stride above the width: the window holds the gaps the rows pitch
        // over, not the last row's trailing gap.
        assert_eq!(pair_y_len(2, 2, 600, 512), Some(2 * 600 * 2 - (600 - 512)));
        assert_eq!(pair_y_len(2, 9, 512, 512), Some(9 * 512 * 2));
        assert_eq!(pair_y_len(2, 1, usize::MAX, usize::MAX), None);
    }
}
