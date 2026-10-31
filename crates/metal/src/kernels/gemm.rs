//! The launch of the dense quantized GEMM: R activation rows against one
//! quantized weight tensor, or the weight rows of up to four tensors of one
//! type and one width, in one dispatch. The kernels are
//! `kernels/gemm.metal`, one tile body over the quant, the operand type, the
//! activations' type and the epilogue, shared by the plain and the segmented
//! entries the way the mat-vec's row loop is.
//!
//! The activations come as f32 rows, which the kernel casts as it stages
//! them, or on the simdgroup path's f16 operand as half rows
//! [`Kernels::gemm_x`] cast once: the f32 entries re-read and re-check every
//! value once per weight tile, so the GEMMs that share an input read it cast.

use bloomery_device::{Buffer, DeviceError, Elems, Window, WindowMut};

use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{GemmPath, Library, Pipeline};

use super::matvec::Quant;
use super::{
    Kernels, MAX_SEGMENTS, Segment, WeightAlign, args_bytes, check_weights_align, check_window_len,
    pack, range, weight_bytes, word,
};

/// The weight rows one GEMM threadgroup owns, the kernel's own constant
/// (`gemm.metal`'s `GEMM_ROWS`).
pub const GEMM_ROWS: usize = 64;

/// The activation rows one GEMM threadgroup owns (`GEMM_ACTS`).
pub const GEMM_ACTS: usize = 64;

/// The threads in a GEMM threadgroup: four SIMD groups.
pub const GEMM_THREADS: usize = 128;

/// The operand type the GEMM computes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// The weights dequantized to half and the activations cast to half: a
    /// product of two half values accumulates in f32 exactly, and the cast
    /// is checked — an activation that is not finite or past the half range
    /// raises the activation quantizer's fault site and writes NaN rows.
    F16,
    /// Both operands f32: no cast, no check, the decode-level error band.
    F32,
}

/// What the GEMM does with its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    /// `y = W . x`.
    Store,
    /// `y = y + W . x`, in place over `y`: the residual epilogue, the decode
    /// RESID meaning.
    Residual,
}

/// The activation rows a GEMM reads, and the elements between them.
#[derive(Clone, Copy)]
pub enum GemmX<'w> {
    /// f32 rows: the kernel casts them to the operand as it stages them.
    F32 {
        x: Window<'w, f32, MetalDevice>,
        /// The f32 elements between the rows, at least the columns.
        stride: usize,
    },
    /// Half rows (binary16 bits) a [`Kernels::gemm_x`] cast wrote: the
    /// [`Operand::F16`] arm of the simdgroup path only. The window starts on
    /// 8 bytes and the stride is a multiple of four (the staging's `half4`
    /// loads).
    F16 {
        x: Window<'w, u16, MetalDevice>,
        /// The half elements between the rows, at least the columns.
        stride: usize,
    },
}

impl GemmX<'_> {
    /// The elements between the rows.
    fn stride(&self) -> usize {
        match self {
            Self::F32 { stride, .. } | Self::F16 { stride, .. } => *stride,
        }
    }

    /// The window's length in elements.
    fn len(&self) -> usize {
        match self {
            Self::F32 { x, .. } => x.len().get(),
            Self::F16 { x, .. } => x.len().get(),
        }
    }
}

/// The cast [`Kernels::gemm_x`] runs: `acts` f32 rows of `cols` values into
/// the half rows the GEMMs of the same input read.
pub struct GemmXParams<'w> {
    /// The f32 rows.
    pub x: Window<'w, f32, MetalDevice>,
    /// The f32 elements between them, at least `cols`.
    pub x_stride: usize,
    /// The values of a row the GEMMs read: their input columns.
    pub cols: usize,
    /// The rows.
    pub acts: usize,
    /// The operand the GEMMs compute in: only the f16 operand reads half
    /// rows.
    pub operand: Operand,
    /// The half rows' buffer: `acts` rows of `cols` values from its start.
    pub half: &'w mut MetalBuffer<u16>,
    /// The fault word the cast's range check raises into.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The threads of a cast threadgroup (`gemm.metal`'s `GEMM_CAST_THREADS`),
/// four values each.
const GEMM_CAST_THREADS: usize = 128;

/// One dense GEMM: `rows` weight rows over `cols` input columns, `acts`
/// activation rows.
pub struct GemmParams<'w> {
    /// The weight type.
    pub quant: Quant,
    /// The weight rows N: the output's columns.
    pub rows: usize,
    /// The input columns K, a multiple of the type's block values.
    pub cols: usize,
    /// The activation rows R: the output's rows.
    pub acts: usize,
    /// The weights' bytes: `rows * cols / values_per_block()` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The activations, `acts` rows of `cols` values.
    pub x: GemmX<'w>,
    /// The output, `acts` rows of `rows` values.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The f32 elements between the output rows, at least `rows` and a
    /// multiple of four (the edge store writes `float4`).
    pub y_stride: usize,
    /// What the GEMM does with its output.
    pub mode: GemmMode,
    /// The operand type the GEMM computes in.
    pub operand: Operand,
    /// The fault word a raise lands in (the f16 operand's range check).
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in: [`bloomery_device::LAYER_NONE`] for
    /// a direct launch.
    pub layer: u32,
}

/// The segmented GEMM's parameters: the segments and the windows they read
/// and write. Every segment has the same weight type and the same width;
/// its rows are columns of the one R-row output, at its own offset.
pub struct GemmSegParams<'w> {
    /// The weight type every segment has.
    pub quant: Quant,
    /// The segments, one to [`MAX_SEGMENTS`] of them.
    pub segments: &'w [Segment],
    /// The weights of every segment, one buffer.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The activations, `acts` rows of the segments' columns.
    pub x: GemmX<'w>,
    /// The activation rows R, the output's rows.
    pub acts: usize,
    /// The output, `acts` rows of every segment's rows.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The f32 elements between the output rows, at least the segments'
    /// rows and a multiple of four.
    pub y_stride: usize,
    /// What the GEMM does with its output.
    pub mode: GemmMode,
    /// The operand type the GEMM computes in.
    pub operand: Operand,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The GEMM pipelines of one weight type: every epilogue and operand, plain
/// and segmented, and the f16 operand's over half rows where the path has
/// them.
pub(crate) struct GemmPipelines {
    plain: [[Pipeline; 2]; 2],
    seg: [[Pipeline; 2]; 2],
    /// `[plain, segmented][epilogue]` over half rows: the simdgroup path's
    /// (the tensor-op library has no half-row entry).
    half: Option<[[Pipeline; 2]; 2]>,
}

impl GemmPipelines {
    /// The pipelines of `stem`'s family — `gemm_{q4k,q6k,a4}` — suffixed by
    /// the epilogue, the operand and the segmented form, from the library of
    /// the GEMM path `path`.
    pub(crate) fn new(library: &Library, stem: &str, path: GemmPath) -> Result<Self, DeviceError> {
        let one = |suffix: &str| library.pipeline(&format!("{stem}{suffix}"));
        let half = match path {
            GemmPath::Simdgroup => Some([
                [one("_store_h16")?, one("_resid_h16")?],
                [one("_seg_store_h16")?, one("_seg_resid_h16")?],
            ]),
            GemmPath::TensorOps => None,
        };
        Ok(Self {
            plain: [
                [one("_store_f16")?, one("_store_f32")?],
                [one("_resid_f16")?, one("_resid_f32")?],
            ],
            seg: [
                [one("_seg_store_f16")?, one("_seg_store_f32")?],
                [one("_seg_resid_f16")?, one("_seg_resid_f32")?],
            ],
            half,
        })
    }

    /// The pipeline of one launch: `[epilogue][operand]` over f32 rows, the
    /// f16 operand's half-row entry over half rows, segmented or not. Half
    /// rows under the f32 operand, or on a path without their entries, are
    /// refused by name.
    fn of(
        &self,
        seg: bool,
        mode: GemmMode,
        operand: Operand,
        x: &GemmX<'_>,
    ) -> Result<&Pipeline, DeviceError> {
        match x {
            GemmX::F32 { .. } => {
                let table = if seg { &self.seg } else { &self.plain };
                Ok(&table[mode.at()][operand.at()])
            }
            GemmX::F16 { .. } => {
                if operand != Operand::F16 {
                    return Err(refuse(format!(
                        "half activation rows under the {operand:?} operand: only the f16 operand reads them"
                    )));
                }
                let half = self.half.as_ref().ok_or_else(|| {
                    refuse(
                        "half activation rows on the tensor-op path, which has no half-row entry",
                    )
                })?;
                Ok(&half[usize::from(seg)][mode.at()])
            }
        }
    }

    /// Whether the path has the half-row entries.
    fn takes_half(&self) -> bool {
        self.half.is_some()
    }
}

impl GemmMode {
    /// The pipeline table's index of this epilogue.
    const fn at(self) -> usize {
        match self {
            Self::Store => 0,
            Self::Residual => 1,
        }
    }
}

impl Operand {
    /// The pipeline table's index of this operand.
    const fn at(self) -> usize {
        match self {
            Self::F16 => 0,
            Self::F32 => 1,
        }
    }
}

/// The grid of a simdgroup tile launch, `(x, y)`: the weight tiles across
/// x, the fastest axis, and the activation tiles down y, the order
/// `gemm.metal`'s `gemm_kernel` and `gemm_seg_kernel` read their tile from.
/// Every launch of those kernels takes its grid here, the entries other files
/// instantiate from them included (`gemm_routed.metal`'s `gemm_f32w`).
pub(super) const fn simdgroup_grid(groups_rows: usize, groups_acts: usize) -> (usize, usize) {
    (groups_rows, groups_acts)
}

/// The shape of a launch, once its windows were checked against it.
struct Shape {
    /// The weight rows the output's columns cover: one tensor's, or every
    /// segment's side by side.
    rows: u32,
    acts: u32,
    x_stride: u32,
    y_stride: u32,
    /// The activation-tile threadgroups and the weight-tile ones.
    groups_acts: usize,
    groups_rows: usize,
}

const WHAT: &str = "gemm";

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

/// The windows both forms read and write, against the ends the shape needs:
/// the weights' whole span, the last activation row's end (a strided row's
/// tail gap belongs to no row the kernel reads) and the last output row's.
struct Windows<'a> {
    weights: &'a Window<'a, u8, MetalDevice>,
    /// The weights' last byte: every segment's span.
    weight_end: usize,
    x: &'a GemmX<'a>,
    acts: usize,
    cols: usize,
    y: &'a WindowMut<'a, f32, MetalDevice>,
    /// The output's last element: the last row's row-end.
    y_end: usize,
}

/// The windows of one launch, refused by name when one is not the shape's:
/// the lengths, and the alignments the kernel's loads and stores need (the
/// weights as 16-bit words, half rows as `half4`, the output's edge store as
/// `float4`).
fn check_windows(w: &Windows<'_>) -> Result<u32, DeviceError> {
    let x_stride = w.x.stride();
    if x_stride < w.cols {
        return Err(refuse(format!(
            "the x stride {x_stride} is narrower than its row of {} columns",
            w.cols
        )));
    }
    let x_end = (w.acts - 1)
        .checked_mul(x_stride)
        .and_then(|at| at.checked_add(w.cols))
        .ok_or_else(|| refuse(format!("{} rows of stride {x_stride} wrap", w.acts)))?;
    check_window_len(WHAT, "weights", w.weights.len(), w.weight_end)?;
    check_window_len(WHAT, "x", Elems(w.x.len()), x_end)?;
    if let GemmX::F16 { x, stride } = w.x {
        check_half_rows(x, *stride)?;
    }
    check_window_len(WHAT, "y", w.y.len(), w.y_end)?;
    let w_start = w.weights.byte_range()?.start.get();
    check_weights_align(WHAT, "weights", w_start, WeightAlign::Half)?;
    let y_start = w.y.byte_range()?.start.get();
    if !y_start.is_multiple_of(16) {
        return Err(refuse(format!(
            "the y window starts {y_start} bytes into its buffer: the edge store writes float4"
        )));
    }
    word(WHAT, "the activation row count", w.acts)
}

/// Half rows as the staging's `half4` loads need them: the window on 8 bytes
/// and the stride a multiple of four, refused by name otherwise.
fn check_half_rows(x: &Window<'_, u16, MetalDevice>, stride: usize) -> Result<(), DeviceError> {
    let start = x.byte_range()?.start.get();
    if !start.is_multiple_of(8) {
        return Err(refuse(format!(
            "the half rows start {start} bytes into their buffer: the staging loads half4"
        )));
    }
    if !stride.is_multiple_of(4) {
        return Err(refuse(format!(
            "the half rows' stride {stride} is not a multiple of the four the staging loads"
        )));
    }
    Ok(())
}

/// The shape of `p`, refused by name when a window is not the shape's, the
/// columns are not whole blocks, a stride is narrower than its row, or a
/// count passes the kernel's 32-bit words.
fn check(p: &GemmParams<'_>) -> Result<Shape, DeviceError> {
    if p.rows == 0 || p.cols == 0 || p.acts == 0 {
        return Err(refuse(format!(
            "{} weight rows by {} columns by {} activation rows is empty",
            p.rows, p.cols, p.acts
        )));
    }
    if !p.cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(format!(
            "{} columns are not whole blocks of {}",
            p.cols,
            p.quant.values_per_block()
        )));
    }
    if !p.y_stride.is_multiple_of(4) {
        return Err(refuse(format!(
            "the y stride {} is not a multiple of the four elements the edge store writes",
            p.y_stride
        )));
    }
    if p.y_stride < p.rows {
        return Err(refuse(format!(
            "the y stride {} is narrower than its row of {} weight rows",
            p.y_stride, p.rows
        )));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let weight_end = weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
    let y_end = (p.acts - 1)
        .checked_mul(p.y_stride)
        .and_then(|at| at.checked_add(p.rows))
        .ok_or_else(|| refuse(format!("{} rows of stride {} wrap", p.acts, p.y_stride)))?;
    let acts = check_windows(&Windows {
        weights: &p.weights,
        weight_end,
        x: &p.x,
        acts: p.acts,
        cols: p.cols,
        y: &p.y,
        y_end,
    })?;
    Ok(Shape {
        rows: word(WHAT, "the weight row count", p.rows)?,
        acts,
        x_stride: word(WHAT, "the x stride", p.x.stride())?,
        y_stride: word(WHAT, "the y stride", p.y_stride)?,
        groups_acts: p.acts.div_ceil(GEMM_ACTS),
        groups_rows: p.rows.div_ceil(GEMM_ROWS),
    })
}

/// The shape of `p`'s segment table: the segments of one type and one
/// width, their weight spans inside one buffer, their output columns
/// disjoint. A segment's rows are the rows a launch of that segment alone
/// would write, at its own column offset.
fn check_seg(p: &GemmSegParams<'_>) -> Result<Shape, DeviceError> {
    if p.segments.is_empty() {
        return Err(refuse("the dispatch has no segment".to_string()));
    }
    if p.segments.len() > MAX_SEGMENTS {
        return Err(refuse(format!(
            "{} segments, a dispatch runs at most {MAX_SEGMENTS}",
            p.segments.len()
        )));
    }
    if p.acts == 0 {
        return Err(refuse(format!("{} activation rows is empty", p.acts)));
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
    if !p.y_stride.is_multiple_of(4) {
        return Err(refuse(format!(
            "the y stride {} is not a multiple of the four elements the edge store writes",
            p.y_stride
        )));
    }

    // The segments' output columns must not overlap, or one segment's rows
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

    let nblocks = cols / p.quant.values_per_block();
    let mut weight_end = 0usize;
    for s in p.segments {
        let span = weight_bytes(p.quant, s.rows, nblocks)
            .ok_or_else(|| refuse(format!("a segment of {} rows wraps", s.rows)))?;
        let end = s
            .weight_blocks
            .checked_mul(p.quant.block_bytes())
            .and_then(|at| at.checked_add(span))
            .ok_or_else(|| refuse(format!("a segment at block {} wraps", s.weight_blocks)))?;
        weight_end = weight_end.max(end);
    }
    let rows = p
        .segments
        .iter()
        .map(|s| s.out_offset + s.rows)
        .max()
        .unwrap_or(0);
    if p.y_stride < rows {
        return Err(refuse(format!(
            "the y stride {} is narrower than its row of {rows} weight rows",
            p.y_stride
        )));
    }
    let y_end = (p.acts - 1)
        .checked_mul(p.y_stride)
        .and_then(|at| at.checked_add(rows))
        .ok_or_else(|| refuse(format!("{} rows of stride {} wrap", p.acts, p.y_stride)))?;
    let acts = check_windows(&Windows {
        weights: &p.weights,
        weight_end,
        x: &p.x,
        acts: p.acts,
        cols,
        y: &p.y,
        y_end,
    })?;
    Ok(Shape {
        rows: word(WHAT, "the segment row count", rows)?,
        acts,
        x_stride: word(WHAT, "the x stride", p.x.stride())?,
        y_stride: word(WHAT, "the y stride", p.y_stride)?,
        groups_acts: p.acts.div_ceil(GEMM_ACTS),
        groups_rows: p.segments.iter().map(|s| s.rows.div_ceil(GEMM_ROWS)).sum(),
    })
}

/// What the plain and the segmented launches share at the dispatch: the
/// checked windows, the packed argument words and the grid, over the arm
/// both forms take.
struct GemmDispatch<'a> {
    quant: Quant,
    seg: bool,
    mode: GemmMode,
    operand: Operand,
    weights: &'a Window<'a, u8, MetalDevice>,
    x: &'a GemmX<'a>,
    y: &'a WindowMut<'a, f32, MetalDevice>,
    fault: &'a MetalFaultWord,
    /// The kernel's argument words, packed.
    args: &'a [u8],
    groups_acts: usize,
    groups_rows: usize,
}

impl Kernels {
    /// Whether the f16 operand's GEMMs read half rows on this path: the
    /// simdgroup path's do, the tensor-op path has no half-row entry.
    #[must_use]
    pub fn gemm_takes_half(&self) -> bool {
        self.a4_gemm.takes_half()
    }

    /// The activations of the GEMMs that read `p.x`. Where they read half
    /// rows (the f16 operand on a path that takes them), the rows are cast
    /// once into `p.half` — the staging's own checked cast, which the f32
    /// entries run over every value once per weight tile — and a barrier
    /// orders the cast before the GEMMs; every other launch reads `p.x` as
    /// it is and nothing is encoded. A window that is not the shape's is
    /// refused by name before anything is encoded.
    #[must_use = "the GEMMs read the activations this returns"]
    pub fn gemm_x<'w>(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: GemmXParams<'w>,
    ) -> Result<GemmX<'w>, DeviceError> {
        const WHAT_X: &str = "gemm cast";
        if p.cols == 0 || p.acts == 0 {
            return Err(range(
                WHAT_X,
                format!("{} rows of {} columns is empty", p.acts, p.cols),
            ));
        }
        if p.x_stride < p.cols {
            return Err(range(
                WHAT_X,
                format!(
                    "the x stride {} is narrower than its row of {} columns",
                    p.x_stride, p.cols
                ),
            ));
        }
        let x_end = (p.acts - 1)
            .checked_mul(p.x_stride)
            .and_then(|at| at.checked_add(p.cols))
            .ok_or_else(|| {
                range(
                    WHAT_X,
                    format!("{} rows of stride {} wrap", p.acts, p.x_stride),
                )
            })?;
        check_window_len(WHAT_X, "x", p.x.len(), x_end)?;
        if p.operand != Operand::F16 || !self.gemm_takes_half() {
            return Ok(GemmX::F32 {
                x: p.x,
                stride: p.x_stride,
            });
        }
        if !p.cols.is_multiple_of(4) {
            return Err(range(
                WHAT_X,
                format!("{} columns are not whole half4 runs", p.cols),
            ));
        }
        let len = p.acts.checked_mul(p.cols).ok_or_else(|| {
            range(
                WHAT_X,
                format!("{} rows of {} columns wrap", p.acts, p.cols),
            )
        })?;
        let args = args_bytes([
            word(WHAT_X, "the column count", p.cols)?,
            word(WHAT_X, "the row count", p.acts)?,
            word(WHAT_X, "the x stride", p.x_stride)?,
            word(WHAT_X, "the half stride", p.cols)?,
            p.layer,
        ]);
        {
            let half = p.half.window_mut(Elems(0), Elems(len))?;
            check_half_rows(&half.as_read(), p.cols)?;
            encoder.set_pipeline(&self.gemm_cast);
            let mut slot = Bindings::new(encoder);
            slot.set(Binding::of(&p.x)?)?;
            slot.set(Binding::of_mut(&half)?)?;
            slot.set(Binding::bytes(pack(&args))?)?;
            slot.set(p.fault.binding())?;
            slot.dispatch_groups(
                p.cols.div_ceil(4 * GEMM_CAST_THREADS),
                p.acts,
                GEMM_CAST_THREADS,
            )?;
        }
        encoder.memory_barrier();
        let half: &'w MetalBuffer<u16> = p.half;
        Ok(GemmX::F16 {
            x: half.window(Elems(0), Elems(len))?,
            stride: p.cols,
        })
    }

    /// Encode one dense GEMM dispatch on `encoder`. Every window is checked
    /// against the shape first; nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn gemm(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: GemmParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(&p)?;
        let args = args_bytes([
            word(WHAT, "the column count", p.cols)?,
            shape.rows,
            shape.acts,
            shape.x_stride,
            shape.y_stride,
            p.layer,
        ]);
        self.gemm_dispatch(
            encoder,
            &GemmDispatch {
                quant: p.quant,
                seg: false,
                mode: p.mode,
                operand: p.operand,
                weights: &p.weights,
                x: &p.x,
                y: &p.y,
                fault: p.fault,
                args: pack(&args),
                groups_acts: shape.groups_acts,
                groups_rows: shape.groups_rows,
            },
        )
    }

    /// Encode one segmented GEMM dispatch on `encoder`: the weight rows of
    /// up to four tensors of one type and one width, in one dispatch.
    #[must_use = "a refused launch encoded nothing"]
    pub fn gemm_seg(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: GemmSegParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_seg(&p)?;
        // The header's six words, then every segment's four — the table the
        // kernel reads to its `nseg`, the rest zero.
        let mut words = [0u32; 6 + 4 * MAX_SEGMENTS];
        words[0] = word(WHAT, "the column count", p.segments[0].cols)?;
        words[1] = shape.acts;
        words[2] = shape.x_stride;
        words[3] = shape.y_stride;
        words[4] = p.layer;
        words[5] = word(WHAT, "the segment count", p.segments.len())?;
        let mut groups = 0usize;
        for (i, s) in p.segments.iter().enumerate() {
            let at = 6 + 4 * i;
            words[at] = word(WHAT, "a segment's rows", s.rows)?;
            words[at + 1] = word(WHAT, "a segment's weight blocks", s.weight_blocks)?;
            words[at + 2] = word(WHAT, "a segment's output offset", s.out_offset)?;
            words[at + 3] = word(WHAT, "a segment's first threadgroup", groups)?;
            groups += s.rows.div_ceil(GEMM_ROWS);
        }
        self.gemm_dispatch(
            encoder,
            &GemmDispatch {
                quant: p.quant,
                seg: true,
                mode: p.mode,
                operand: p.operand,
                weights: &p.weights,
                x: &p.x,
                y: &p.y,
                fault: p.fault,
                args: pack(&args_bytes(words)),
                groups_acts: shape.groups_acts,
                groups_rows: shape.groups_rows,
            },
        )
    }

    /// The bindings and the dispatch the two forms share: the buffers in the
    /// kernel's slot order, the packed words, the fault word, the grid.
    fn gemm_dispatch(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        d: &GemmDispatch<'_>,
    ) -> Result<(), DeviceError> {
        let family = match d.quant {
            Quant::Q4K => &self.q4k_gemm,
            Quant::Q6K => &self.q6k_gemm,
            Quant::A4 => &self.a4_gemm,
            Quant::A8 => &self.a8_gemm,
            Quant::A4H => &self.a4h_gemm,
            Quant::A8H => &self.a8h_gemm,
            Quant::Q8_0 => &self.q8_0_gemm,
            Quant::Q5K => &self.q5k_gemm,
            // The f32 rows have no tile of this family: a pass runs each
            // segment as the router's f32 GEMM (`Kernels::gemm_f32`).
            Quant::F32 => {
                return Err(refuse(
                    "f32 has no dense GEMM entry; its rows run as the f32 GEMM".to_string(),
                ));
            }
        };
        encoder.set_pipeline(family.of(d.seg, d.mode, d.operand, d.x)?);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(d.weights)?)?;
        match d.x {
            GemmX::F32 { x, .. } => slot.set(Binding::of(x)?)?,
            GemmX::F16 { x, .. } => slot.set(Binding::of(x)?)?,
        }
        slot.set(Binding::of_mut(d.y)?)?;
        slot.set(Binding::bytes(d.args)?)?;
        slot.set(d.fault.binding())?;
        // The simdgroup kernels take the weight tiles across x, the fastest
        // axis, so the threadgroups in flight share one activation tile
        // (`gemm.metal`); the tensor-op kernels take the activation tiles.
        let (x, y) = match self.gemm_path() {
            GemmPath::Simdgroup => simdgroup_grid(d.groups_rows, d.groups_acts),
            GemmPath::TensorOps => (d.groups_acts, d.groups_rows),
        };
        slot.dispatch_groups(x, y, GEMM_THREADS)
    }
}
