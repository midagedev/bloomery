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

/// The vocabulary the engine's head reads.
pub const HEAD_VOCAB: usize = 248_320;

/// The columns of every row.
pub const HEAD_K: usize = 2048;

/// The SIMD groups in the head's threadgroup.
pub const HEAD_SIMD_GROUPS: usize = 8;

/// The threads in the head's threadgroup.
pub const HEAD_THREADS: usize = 256;

/// The threadgroups of the engine's head.
pub const HEAD_GROUPS: usize = 15_520;

/// The rows one threadgroup owns: eight SIMD groups, two rows each.
const ROWS_PER_THREADGROUP: usize = HEAD_SIMD_GROUPS * ROWS_PER_SIMD_GROUP;

const _: () = assert!(ROWS_PER_THREADGROUP == 16);
const _: () = assert!(HEAD_THREADS == 32 * HEAD_SIMD_GROUPS);
const _: () = assert!(HEAD_GROUPS * ROWS_PER_THREADGROUP == HEAD_VOCAB);

/// One head launch.
pub struct HeadParams<'w> {
    /// The output rows. The engine's head reads [`HEAD_VOCAB`]. A launch
    /// accepts any positive count, so a clause can run a shorter grid.
    pub rows: usize,
    /// The columns, a positive multiple of 256. The engine's head reads
    /// [`HEAD_K`].
    pub k: usize,
    /// The SIMD groups in the threadgroup. The kernel runs
    /// [`HEAD_SIMD_GROUPS`] and no other count.
    pub simdgroups: usize,
    /// The Q6_K weights, `rows * (k / 256) * 210` bytes.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input, `k` values.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output norm's gain, `k` values.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The logits, `rows` values.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// One packed candidate per threadgroup, `rows.div_ceil(16)` values.
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
/// is not that shape's.
fn check(p: &HeadParams<'_>) -> Result<(), DeviceError> {
    if p.k == 0 || !p.k.is_multiple_of(256) {
        return Err(refuse(format!(
            "K is {}, the kernel reads a positive multiple of 256",
            p.k
        )));
    }
    if p.rows == 0 {
        return Err(refuse("the row count is 0".to_string()));
    }
    if !p.eps.is_finite() {
        return Err(refuse(format!("eps {} is not finite", p.eps)));
    }
    let nblocks = p.k / 256;
    let weight_bytes = super::weight_bytes(Quant::Q6K, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.k)))?;
    let candidates = p.rows.div_ceil(ROWS_PER_THREADGROUP);
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
    Ok(())
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
        check(&p)?;
        let nblocks = p.k / 256;
        let groups = p.rows.div_ceil(ROWS_PER_THREADGROUP);
        let args = args_bytes([
            word(WHAT, "the row count", p.rows)?,
            word(WHAT, "the block count", nblocks)?,
            p.layer,
            p.eps.to_bits(),
        ]);
        encoder.set_pipeline(&self.head);
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
