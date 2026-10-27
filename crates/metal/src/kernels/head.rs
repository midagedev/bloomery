//! The launch of the output head: one Q6_K mat-vec with the norm on the load,
//! and the argmax in the same dispatch.
//!
//! The kernel is `kernels/head.metal`. Its pipeline is specialized at eight
//! SIMD groups and at no other count. A launch checks the shape and every
//! window's length before it encodes anything. The caller stores 0 into the
//! ticket at allocation, and the kernel stores 0 again when it finishes. A
//! launch that is refused encodes nothing.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::ROWS_PER_SIMD_GROUP;
use super::{Kernels, Quant, args_bytes, check_window_len, pack, word};

/// The columns of every row of the head: the hidden width (`Geometry::hidden`).
/// A launch takes any positive multiple of the type's block values. A shim:
/// the build's host instance's. Callers left: the kernel gate and the probes.
pub const HEAD_K: usize = crate::instance::HOST.key.hidden as usize;

/// The SIMD groups in the head's threadgroup.
pub const HEAD_SIMD_GROUPS: usize = 8;

/// The threads in the head's threadgroup.
pub const HEAD_THREADS: usize = 256;

/// The rows one threadgroup owns: the head's SIMD groups of the mat-vec
/// geometry's rows each.
const ROWS_PER_THREADGROUP: usize = HEAD_SIMD_GROUPS * ROWS_PER_SIMD_GROUP;

const _: () = assert!(HEAD_THREADS == 32 * HEAD_SIMD_GROUPS);

/// The threadgroups of a head over `rows` rows, which is also the length of
/// its candidates: one per threadgroup's rows, the last owning the remainder
/// when the count is not a multiple.
#[must_use]
pub const fn head_groups(rows: usize) -> usize {
    rows.div_ceil(ROWS_PER_THREADGROUP)
}

/// One head launch.
pub struct HeadParams<'w> {
    /// The weight type. Q6_K and A4 have a head entry; Q4_K has none and is
    /// refused by name.
    pub quant: Quant,
    /// The output rows: the file's vocabulary for the engine's head. A launch
    /// accepts any positive count.
    pub rows: usize,
    /// The columns, a positive multiple of the type's block values: the
    /// file's hidden width for the engine's head.
    pub k: usize,
    /// The SIMD groups in the threadgroup. The kernel runs
    /// [`HEAD_SIMD_GROUPS`] and no other count.
    pub simdgroups: usize,
    /// The weights, `rows * (k / values_per_block())` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input, `k` values.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output norm's gain, `k` values.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The logits, `rows` values.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// One packed candidate per threadgroup, [`head_groups`]`(rows)` values.
    pub candidates: WindowMut<'w, u64, MetalDevice>,
    /// The ticket counter, one `u32`. The caller stores 0 before the first
    /// launch, and the kernel stores 0 again when it finishes.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
    /// The winning row, one `u32`.
    pub step_id: WindowMut<'w, u32, MetalDevice>,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

const WHAT: &str = "head";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape of `p`, refused by name when it is not the kernel's or a window
/// is not that shape's. `rows_per_threadgroup` is the geometry of the
/// pipelines this launch encodes against.
fn check(p: &HeadParams<'_>, rows_per_threadgroup: usize) -> Result<(usize, usize), DeviceError> {
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
    if !p.eps.is_finite() {
        return Err(refuse(format!("eps {} is not finite", p.eps)));
    }
    let nblocks = p.k / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.k)))?;
    let candidates = p.rows.div_ceil(rows_per_threadgroup);
    check_window_len(WHAT, "x", p.x.len(), p.k)?;
    check_window_len(WHAT, "gain", p.gain.len(), p.k)?;
    check_window_len(WHAT, "logits", p.logits.len(), p.rows)?;
    check_window_len(WHAT, "candidates", p.candidates.len(), candidates)?;
    check_window_len(WHAT, "step id", p.step_id.len(), 1)?;
    check_window_len(WHAT, "ticket", p.ticket.len(), 1)?;
    check_window_len(WHAT, "weights", p.weights.len(), weight_bytes)?;
    if p.simdgroups != HEAD_SIMD_GROUPS {
        return Err(refuse(format!(
            "the SIMD-group count is {}, the kernel runs {HEAD_SIMD_GROUPS}",
            p.simdgroups
        )));
    }
    word(WHAT, "the row count", p.rows)?;
    word(WHAT, "the block count", nblocks)?;
    word(WHAT, "the group count", candidates)?;
    Ok((nblocks, candidates))
}

impl Kernels {
    /// Encode one head dispatch on `encoder`. The shape and every window are
    /// checked first; nothing is encoded for a launch that is refused. The
    /// ticket is the caller's: this launch does not store into it.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadParams<'_>,
    ) -> Result<(), DeviceError> {
        let pipeline = match p.quant {
            Quant::Q6K => &self.head_q6k,
            Quant::A4 => &self.head_a4,
            Quant::Q8_0 => &self.head_q8_0,
            Quant::Q5K => &self.head_q5k,
            // A quant with no head entry is refused by name: the entry is
            // absent, not defaulted.
            Quant::Q4K => {
                return Err(refuse("Q4_K has no head entry".to_string()));
            }
            Quant::F32 => {
                return Err(refuse("F32 has no head entry".to_string()));
            }
        };
        let rows_per_threadgroup = HEAD_SIMD_GROUPS * self.rows_per_simd_group;
        let (nblocks, groups) = check(&p, rows_per_threadgroup)?;
        let args = args_bytes([
            word(WHAT, "the row count", p.rows)?,
            word(WHAT, "the block count", nblocks)?,
            p.layer,
            p.eps.to_bits(),
        ]);
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(Binding::of_mut(&p.candidates)?)?;
        slot.set(Binding::of_mut(&p.ticket)?)?;
        slot.set(Binding::of_mut(&p.step_id)?)?;
        slot.set(p.fault.binding())?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(groups, 1, HEAD_THREADS)
    }
}
