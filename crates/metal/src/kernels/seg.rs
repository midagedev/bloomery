//! The segmented mat-vec: the row segments of up to four tensors of one type
//! and one width, in one dispatch.
//!
//! A layer's projections share their input, so they share a norm and a gain:
//! one dispatch runs the segments of several of them, and each segment writes
//! its own rows. The rows a segment writes are the rows a launch of that
//! tensor alone would write: the segment's weights sit at their own offset in
//! the weights buffer, its rows at their own offset in the output.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};

use super::geometry::MAX_SEGMENTS;
use super::matvec::{NormOnLoad, Quant};
use super::{Kernels, args_bytes, check_window_len, pack, word};

/// One segment of a segmented mat-vec: the rows of one tensor, and where they
/// sit in the weights and in the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The input columns the segment's rows are over. Every segment of a
    /// dispatch has the same: the dispatch is over one width.
    pub cols: usize,
    /// The segment's output rows.
    pub rows: usize,
    /// The segment's first row's block offset in the weights buffer.
    pub weight_blocks: usize,
    /// The segment's first row's element offset in the output buffer.
    pub out_offset: usize,
}

/// The segmented mat-vec's parameters: the segments, the windows they read and
/// write, and the norm on the load.
pub struct SegParams<'w> {
    /// The weight type every segment has.
    pub quant: Quant,
    /// The segments, one to [`MAX_SEGMENTS`] of them, borrowed from the
    /// launcher's own list: a launch allocates none.
    pub segments: &'w [Segment],
    /// The weights of every segment, one buffer.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input vector, one value per column.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output of every segment, one buffer.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The norm the load applies.
    pub norm: NormOnLoad<'w>,
}

/// One segment, as the kernel's argument block holds it.
#[derive(Clone, Copy)]
struct SegArg {
    /// The segment's output rows.
    rows: u32,
    /// The segment's first row's block offset in the weights.
    weight_blocks: u32,
    /// The segment's first row's element offset in the output.
    out: u32,
    /// The first threadgroup of the segment.
    first_group: u32,
}

/// The shape of a segmented launch, once its windows were checked against it.
struct Shape {
    nblocks: u32,
    /// The segments, as the kernel reads them.
    segments: [SegArg; MAX_SEGMENTS],
    /// How many segments the dispatch runs.
    nseg: u32,
    /// The threadgroups of the dispatch.
    groups: usize,
}

const WHAT: &str = "matvec seg";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape of `p`, refused by name when the segments disagree, when one of
/// them is empty, when there are too many, when their outputs overlap, or when
/// a window is not the shape's.
fn check(
    p: &SegParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<Shape, DeviceError> {
    if p.segments.is_empty() {
        return Err(refuse("the dispatch has no segment".to_string()));
    }
    if p.segments.len() > MAX_SEGMENTS {
        return Err(refuse(format!(
            "{} segments, a dispatch runs at most {MAX_SEGMENTS}",
            p.segments.len()
        )));
    }
    let cols = p.segments[0].cols;
    for (i, s) in p.segments.iter().enumerate() {
        if s.cols != cols {
            return Err(refuse(format!(
                "segment {i} is over {} columns, the first over {cols}",
                s.cols
            )));
        }
        if s.rows == 0 {
            return Err(refuse(format!("segment {i} has no rows")));
        }
    }
    if !cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(format!(
            "{cols} columns are not whole blocks of {}",
            p.quant.values_per_block()
        )));
    }
    let nblocks = cols / p.quant.values_per_block();

    // The output: the segments' ranges must not overlap, or one segment's rows
    // would be another's. A dispatch holds at most MAX_SEGMENTS segments, so
    // the pairwise compare needs no list.
    for (j, b) in p.segments.iter().enumerate() {
        for a in &p.segments[..j] {
            if a.out_offset < b.out_offset + b.rows && b.out_offset < a.out_offset + a.rows {
                let (first, second) = if a.out_offset <= b.out_offset {
                    (a.out_offset, b.out_offset)
                } else {
                    (b.out_offset, a.out_offset)
                };
                return Err(refuse(format!(
                    "the output of the segments at {first} and {second} overlaps"
                )));
            }
        }
    }

    // The weights: the buffer holds every segment's blocks, and the kernels
    // read them as 16-bit words (F32's as `float4`s). A segment's rows are
    // `rows` rows of `nblocks` blocks each.
    let mut weight_end = 0usize;
    for s in p.segments {
        let span = super::weight_bytes(p.quant, s.rows, nblocks)
            .ok_or_else(|| refuse(format!("a segment of {} rows wraps", s.rows)))?;
        let end = s
            .weight_blocks
            .checked_mul(p.quant.block_bytes())
            .and_then(|at| at.checked_add(span))
            .ok_or_else(|| refuse(format!("a segment at block {} wraps", s.weight_blocks)))?;
        weight_end = weight_end.max(end);
    }
    check_window_len(WHAT, "weights", p.weights.len(), weight_end)?;
    check_window_len(WHAT, "x", p.x.len(), cols)?;
    check_window_len(WHAT, "gain", p.norm.gain.len(), cols)?;
    let out_end = p
        .segments
        .iter()
        .map(|s| s.out_offset + s.rows)
        .max()
        .unwrap_or(0);
    check_window_len(WHAT, "y", p.y.len(), out_end)?;

    let start = p.weights.byte_range()?.start.get();
    super::check_weights_align(WHAT, "weights", start, p.quant.weight_align())?;

    // The threadgroups: every segment's own, so a segment's rows are the rows
    // a launch of that segment alone would write.
    let mut segments = [SegArg {
        rows: 0,
        weight_blocks: 0,
        out: 0,
        first_group: 0,
    }; MAX_SEGMENTS];
    let mut groups = 0usize;
    for (i, s) in p.segments.iter().enumerate() {
        segments[i] = SegArg {
            rows: word(WHAT, "a segment's rows", s.rows)?,
            weight_blocks: word(WHAT, "a segment's weight blocks", s.weight_blocks)?,
            out: word(WHAT, "a segment's output offset", s.out_offset)?,
            first_group: word(WHAT, "a segment's first threadgroup", groups)?,
        };
        groups += s.rows.div_ceil(simdgroups * rows_per_simd_group);
    }
    Ok(Shape {
        nblocks: word(WHAT, "the block count", nblocks)?,
        segments,
        nseg: word(WHAT, "the segment count", p.segments.len())?,
        groups,
    })
}

/// The segmented mat-vec's argument words: the header's four and every
/// segment's four. `pack` is the bytes the kernel reads.
fn args_words(shape: &Shape, layer: u32, eps: f32) -> [u32; 4 + 4 * MAX_SEGMENTS] {
    let mut words = [0u32; 4 + 4 * MAX_SEGMENTS];
    words[0] = shape.nblocks;
    words[1] = layer;
    words[2] = eps.to_bits();
    words[3] = shape.nseg;
    for (i, s) in shape.segments.iter().enumerate() {
        let at = 4 + 4 * i;
        words[at] = s.rows;
        words[at + 1] = s.weight_blocks;
        words[at + 2] = s.out;
        words[at + 3] = s.first_group;
    }
    words
}

impl Kernels {
    /// Encode one segmented mat-vec dispatch on `encoder`. The segment list is
    /// checked first; nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec_seg(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: SegParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(&p, self.simdgroups, self.rows_per_simd_group)?;
        let args = args_bytes(args_words(&shape, p.norm.layer, p.norm.eps));
        let pipeline = match p.quant {
            Quant::F32 => &self.f32_seg,
            dense => self
                .dense_family(dense)
                .expect("a type but F32 has a mat-vec family")
                .seg(),
        };
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(Binding::of(&p.norm.gain)?)?;
        slot.set(p.norm.fault.binding())?;
        let threads = self.threads_per_group();
        slot.dispatch_groups(shape.groups, 1, threads)
    }
}
