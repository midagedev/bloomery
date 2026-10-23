//! The launch of the output head: one Q6_K mat-vec with the norm on the load,
//! and the argmax in the same dispatch.
//!
//! The kernel is `kernels/head.metal`. Its pipeline is specialized at eight
//! SIMD groups and at no other count. A launch checks the shape and every
//! window's length before it encodes anything, and it stores 0 into the
//! ticket word before the dispatch. A launch that is refused encodes nothing.

use core::sync::atomic::Ordering;

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::ROWS_PER_SIMD_GROUP;
use super::{Kernels, Quant, args_bytes, check_window_len, word};

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
const _: () = assert!(HEAD_VOCAB * (HEAD_K / 256) * Quant::Q6K.block_bytes() == 417_177_600);

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
    /// The ticket counter, one `u32`. The launch stores 0 before the
    /// dispatch, and the kernel stores 0 again when it finishes.
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

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "head",
        detail,
    }
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
    let weight_bytes = p
        .rows
        .checked_mul(nblocks)
        .and_then(|blocks| blocks.checked_mul(Quant::Q6K.block_bytes()))
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.k)))?;
    let candidates = p.rows.div_ceil(ROWS_PER_THREADGROUP);
    check_window_len("head", "x", p.x.len(), p.k)?;
    check_window_len("head", "gain", p.gain.len(), p.k)?;
    check_window_len("head", "logits", p.logits.len(), p.rows)?;
    check_window_len("head", "candidates", p.candidates.len(), candidates)?;
    check_window_len("head", "step id", p.step_id.len(), 1)?;
    check_window_len("head", "ticket", p.ticket.len(), 1)?;
    check_window_len("head", "weights", p.weights.len(), weight_bytes)?;
    if p.simdgroups != HEAD_SIMD_GROUPS {
        return Err(refuse(format!(
            "the SIMD-group count is {}, the kernel runs {HEAD_SIMD_GROUPS}",
            p.simdgroups
        )));
    }
    word("head", "the row count", p.rows)?;
    word("head", "the block count", nblocks)?;
    word("head", "the group count", candidates)?;
    Ok(())
}

/// Store 0 into the ticket word. The window is the caller's, and no device
/// work of this encoder has been submitted.
fn zero_ticket(ticket: &WindowMut<'_, u32, MetalDevice>) -> Result<(), DeviceError> {
    let start = ticket.byte_range()?.start;
    let pages = ticket.raw().window_pages(start, ticket.len())?;
    pages.store(0, 0, Ordering::Relaxed)
}

impl Kernels {
    /// Encode one head dispatch on `encoder`. The shape and every window are
    /// checked first; nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadParams<'_>,
    ) -> Result<(), DeviceError> {
        check(&p)?;
        zero_ticket(&p.ticket)?;
        let nblocks = p.k / 256;
        let groups = p.rows.div_ceil(ROWS_PER_THREADGROUP);
        let args = args_bytes([
            word("head", "the row count", p.rows)?,
            word("head", "the block count", nblocks)?,
            p.layer,
            p.eps.to_bits(),
        ]);
        let threads = groups
            .checked_mul(HEAD_THREADS)
            .ok_or_else(|| refuse(format!("{groups} groups of {HEAD_THREADS} threads wrap")))?;
        encoder.set_pipeline(&self.head);
        encoder.set_binding(0, Binding::of(&p.weights)?)?;
        encoder.set_binding(1, Binding::of(&p.x)?)?;
        encoder.set_binding(2, Binding::of(&p.gain)?)?;
        encoder.set_binding(3, Binding::of_mut(&p.logits)?)?;
        encoder.set_binding(4, Binding::of_mut(&p.candidates)?)?;
        encoder.set_binding(5, Binding::of_mut(&p.ticket)?)?;
        encoder.set_binding(6, Binding::of_mut(&p.step_id)?)?;
        encoder.set_binding(7, p.fault.binding())?;
        encoder.set_binding(8, Binding::bytes(&args)?)?;
        encoder.dispatch(threads, HEAD_THREADS)
    }
}
