//! The launch of the output head: two dispatches, the logits and the pick.
//!
//! The kernels are `kernels/head.metal`. The first is a mat-vec with the norm
//! on the load over `rows` rows, run at the mat-vec geometry
//! ([`Kernels::simdgroups`]), which stores every logit. The second is one
//! threadgroup of [`HEAD_THREADS`] threads, specialized at [`HEAD_SIMD_GROUPS`]
//! SIMD groups and at no other count, which reduces the stored logits to the
//! winning row. The first closes its group on the encoder before the second
//! reads: [`Kernels::head`] encodes both with that barrier between, and a
//! caller that encodes them apart ([`Kernels::head_logits`] then
//! [`Kernels::head_pick`]) closes the group itself. A launch checks the shape
//! and every window's length before it encodes anything, and a launch that is
//! refused encodes nothing.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::Pipeline;

use super::{
    Kernels, LoadNorm, Quant, WeightAlign, args_bytes, check_weights_align, check_window_len, pack,
    word,
};

/// The columns of every row of the head: the hidden width (`Geometry::hidden`).
/// A launch takes any positive multiple of the type's block values. A shim:
/// the build's host instance's. Callers left: the kernel gate and the probes.
pub const HEAD_K: usize = crate::instance::HOST.key.hidden as usize;

/// The SIMD groups in the pick's threadgroup: the most a threadgroup holds, so
/// the one threadgroup keeps the most logit loads in flight.
pub const HEAD_SIMD_GROUPS: usize = 32;

/// The threads in the pick's threadgroup.
pub const HEAD_THREADS: usize = 1024;

const _: () = assert!(HEAD_THREADS == 32 * HEAD_SIMD_GROUPS);

/// The most rows a launch takes: the pick's loop steps a thread's row index
/// four threadgroup widths at a time, and that sum stays a 32-bit word.
const MAX_ROWS: usize = u32::MAX as usize - 4 * HEAD_THREADS;

/// One head launch.
pub struct HeadParams<'w> {
    /// The weight type. Q6_K, A4, A8, Q8_0 and Q5_K have a head entry; Q4_K
    /// and F32 have none and are refused by name.
    pub quant: Quant,
    /// The output rows: the file's vocabulary for the engine's head. A launch
    /// accepts any positive count up to the pick's 32-bit loop.
    pub rows: usize,
    /// The columns, a positive multiple of the type's block values: the
    /// file's hidden width for the engine's head.
    pub k: usize,
    /// The SIMD groups in the pick's threadgroup. The pick runs
    /// [`HEAD_SIMD_GROUPS`] and no other count.
    pub simdgroups: usize,
    /// The weights, `rows * (k / values_per_block())` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input, `k` values, starting on a 16-byte boundary of its buffer.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output norm: its gain, or the scale of the join that applied it.
    pub norm: LoadNorm<'w>,
    /// The logits, `rows` values: the first dispatch's output and the
    /// second's input.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// The winning row, one `u32`.
    pub step_id: WindowMut<'w, u32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

const WHAT: &str = "head";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape of a head launch, once its windows were checked against it.
struct Shape {
    nblocks: u32,
    rows: u32,
    /// The threadgroups of the logits dispatch.
    groups: usize,
}

/// The shape of `p`, refused by name when it is not the kernels' or a window
/// is not that shape's. `rows_per_group` is the geometry of the logits
/// pipelines this launch encodes against.
fn check(p: &HeadParams<'_>, rows_per_group: usize) -> Result<Shape, DeviceError> {
    if p.k == 0 || !p.k.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(format!(
            "K is {}, the kernel reads a positive multiple of {}",
            p.k,
            p.quant.values_per_block()
        )));
    }
    if p.rows == 0 {
        return Err(refuse("the row count is 0".to_string()));
    }
    if p.rows > MAX_ROWS {
        return Err(refuse(format!(
            "the row count {} passes the pick's {MAX_ROWS}",
            p.rows
        )));
    }
    if let LoadNorm::Gain { eps, .. } = p.norm
        && !eps.is_finite()
    {
        return Err(refuse(format!("eps {eps} is not finite")));
    }
    let nblocks = p.k / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.k)))?;
    check_window_len(WHAT, "x", p.x.len(), p.k)?;
    p.norm.check(WHAT, p.k)?;
    check_window_len(WHAT, "logits", p.logits.len(), p.rows)?;
    check_window_len(WHAT, "step id", p.step_id.len(), 1)?;
    check_window_len(WHAT, "weights", p.weights.len(), weight_bytes)?;
    let start = p.weights.byte_range()?.start.get();
    check_weights_align(WHAT, "weights", start, p.quant.weight_align())?;
    // The input is read as `float4`s.
    check_weights_align(
        WHAT,
        "x",
        p.x.byte_range()?.start.get(),
        WeightAlign::Float4,
    )?;
    // The pick reads the logits as `float4`s.
    check_weights_align(
        WHAT,
        "logits",
        p.logits.byte_range()?.start.get(),
        WeightAlign::Float4,
    )?;
    if p.simdgroups != HEAD_SIMD_GROUPS {
        return Err(refuse(format!(
            "the SIMD-group count is {}, the kernel runs {HEAD_SIMD_GROUPS}",
            p.simdgroups
        )));
    }
    Ok(Shape {
        nblocks: word(WHAT, "the block count", nblocks)?,
        rows: word(WHAT, "the row count", p.rows)?,
        groups: p.rows.div_ceil(rows_per_group),
    })
}

impl Kernels {
    /// Encode the head on `encoder`: the logits dispatch, a barrier that
    /// closes its group, and the pick. The shape and every window are checked
    /// first; nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadParams<'_>,
    ) -> Result<(), DeviceError> {
        let (pipeline, shape) = self.head_checked(&p)?;
        self.encode_logits(encoder, pipeline, &p, &shape)?;
        encoder.memory_barrier();
        self.encode_pick(encoder, &p, &shape)
    }

    /// Encode the head's first dispatch alone, the logits. The launch is
    /// checked whole, as [`Kernels::head`]'s is. The pick reads what this
    /// wrote, so the caller closes the group on the encoder before it.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head_logits(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: &HeadParams<'_>,
    ) -> Result<(), DeviceError> {
        let (pipeline, shape) = self.head_checked(p)?;
        self.encode_logits(encoder, pipeline, p, &shape)
    }

    /// Encode the head's second dispatch alone, the pick over the logits the
    /// first one stored. The launch is checked whole, as [`Kernels::head`]'s
    /// is.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head_pick(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: &HeadParams<'_>,
    ) -> Result<(), DeviceError> {
        let (_, shape) = self.head_checked(p)?;
        self.encode_pick(encoder, p, &shape)
    }

    /// The logits pipeline of `p`'s type and its checked shape. A type with no
    /// head entry is refused by name: the entry is absent, not defaulted.
    fn head_checked(&self, p: &HeadParams<'_>) -> Result<(&Pipeline, Shape), DeviceError> {
        let pipeline = match (p.quant, p.norm.scaled()) {
            (Quant::Q6K, false) => &self.head_logits_q6k,
            (Quant::A4, false) => &self.head_logits_a4,
            (Quant::A8, false) => &self.head_logits_a8,
            (Quant::Q8_0, false) => &self.head_logits_q8_0,
            (Quant::Q5K, false) => &self.head_logits_q5k,
            (Quant::Q6K, true) => &self.head_logits_q6k_scaled,
            (Quant::A4, true) => &self.head_logits_a4_scaled,
            (Quant::A8, true) => &self.head_logits_a8_scaled,
            (Quant::Q8_0, true) => &self.head_logits_q8_0_scaled,
            (Quant::Q5K, true) => &self.head_logits_q5k_scaled,
            (Quant::Q4K, _) => {
                return Err(refuse("Q4_K has no head entry".to_string()));
            }
            (Quant::F32, _) => {
                return Err(refuse("F32 has no head entry".to_string()));
            }
        };
        let shape = check(p, self.simdgroups * self.rows_per_simd_group)?;
        Ok((pipeline, shape))
    }

    fn encode_logits(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        pipeline: &Pipeline,
        p: &HeadParams<'_>,
        shape: &Shape,
    ) -> Result<(), DeviceError> {
        let args = args_bytes([shape.rows, shape.nblocks, p.layer, p.norm.eps().to_bits()]);
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(p.norm.window())?)?;
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(shape.groups, 1, self.threads_per_group())
    }

    fn encode_pick(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: &HeadParams<'_>,
        shape: &Shape,
    ) -> Result<(), DeviceError> {
        let args = args_bytes([shape.rows, p.layer]);
        encoder.set_pipeline(&self.head_pick);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(Binding::of_mut(&p.step_id)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(1, 1, HEAD_THREADS)
    }
}
