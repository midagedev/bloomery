//! The launch of the router: 257 rows of K 2048, split two ways, one dispatch.
//!
//! The kernel is `kernels/router.metal`. A launch checks the shape, every
//! window's length and the alignment of the windows the kernel loads as
//! float4 before it encodes anything. A launch that is refused encodes nothing.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::SIMD_WIDTH;
use super::{Kernels, args_bytes, check_window_len, word};

/// The routed rows, and the id of the shared expert.
pub const EXPERTS: usize = 256;

/// The rows the kernel reads: the routed experts and the shared expert.
pub const ROWS: usize = EXPERTS + 1;

/// The columns of every row.
pub const K: usize = 2048;

/// The splits of one row. Each SIMD group owns one.
pub const SPLIT: usize = 2;

/// The partials of one launch: one per row per split.
pub const PARTIALS: usize = ROWS * SPLIT;

/// The SIMD groups in the router's threadgroup.
pub const SIMD_GROUPS: usize = 4;

/// The threads in the router's threadgroup.
pub const THREADS: usize = SIMD_WIDTH * SIMD_GROUPS;

/// The threadgroups of one launch.
pub const GROUPS: usize = PARTIALS.div_ceil(SIMD_GROUPS);

/// The most routed experts one launch selects.
pub const TOP_K_MAX: usize = 8;

/// The float4s one lane loads from one split.
pub const FLOAT4S: usize = 8;

const _: () = assert!(PARTIALS == ROWS * SPLIT);
const _: () = assert!(GROUPS == 129);
const _: () = assert!(THREADS == 128);
const _: () = assert!(SIMD_WIDTH * FLOAT4S * 4 == K / SPLIT);
const _: () = assert!(SIMD_WIDTH * 16 * 4 == K);

/// One router launch.
pub struct RouterParams<'w> {
    /// The rows, counting the shared expert. The kernel reads [`ROWS`].
    pub rows: usize,
    /// The columns. The kernel reads [`K`].
    pub k: usize,
    /// The routed experts to select, `1..=`[`TOP_K_MAX`]. The output holds
    /// that many ids and weights plus the shared expert.
    pub top_k: usize,
    /// The routed weights, [`EXPERTS`] rows of [`K`], one segment.
    pub routed: Window<'w, f32, MetalDevice>,
    /// The shared expert's weights, [`K`] values, the second segment.
    pub shared: Window<'w, f32, MetalDevice>,
    /// The input, [`K`] values.
    pub x: Window<'w, f32, MetalDevice>,
    /// The norm's gain, [`K`] values.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The split partials, [`PARTIALS`] values. The kernel owns them for the
    /// launch.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The ticket counter, one `u32`. The caller stores 0 before the launch
    /// and the kernel stores 0 again when it finishes.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
    /// The logits, [`ROWS`] values.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// The selected ids and then the shared expert's id, `top_k + 1` values.
    pub ids: WindowMut<'w, u32, MetalDevice>,
    /// The selected weights and then the shared expert's weight, `top_k + 1`
    /// values.
    pub weights: WindowMut<'w, f32, MetalDevice>,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "router",
        detail,
    }
}

fn align16(name: &str, start: usize) -> Result<(), DeviceError> {
    if start.is_multiple_of(16) {
        return Ok(());
    }
    Err(refuse(format!(
        "the {name} window starts {start} bytes into its buffer and is not 16-byte aligned"
    )))
}

/// The shape of `p`, refused by name when it is not the kernel's, a window is
/// not that shape's, eps is not finite, or a float4 window is misaligned.
fn check(p: &RouterParams<'_>) -> Result<(), DeviceError> {
    if p.k != K {
        return Err(refuse(format!("K is {}, the kernel reads {K}", p.k)));
    }
    if p.rows != ROWS {
        return Err(refuse(format!(
            "the row count is {}, the kernel reads {ROWS}",
            p.rows
        )));
    }
    if !(1..=TOP_K_MAX).contains(&p.top_k) {
        return Err(refuse(format!(
            "top-k is {}, the kernel takes 1 to {TOP_K_MAX}",
            p.top_k
        )));
    }
    if !p.eps.is_finite() {
        return Err(refuse(format!("eps {} is not finite", p.eps)));
    }
    check_window_len("router", "routed", p.routed.len(), EXPERTS * K)?;
    check_window_len("router", "shared", p.shared.len(), K)?;
    check_window_len("router", "x", p.x.len(), K)?;
    check_window_len("router", "gain", p.gain.len(), K)?;
    check_window_len("router", "partials", p.partials.len(), PARTIALS)?;
    check_window_len("router", "ticket", p.ticket.len(), 1)?;
    check_window_len("router", "logits", p.logits.len(), ROWS)?;
    check_window_len("router", "ids", p.ids.len(), p.top_k + 1)?;
    check_window_len("router", "weights", p.weights.len(), p.top_k + 1)?;
    align16("routed", p.routed.byte_range()?.start.get())?;
    align16("shared", p.shared.byte_range()?.start.get())?;
    align16("x", p.x.byte_range()?.start.get())?;
    align16("gain", p.gain.byte_range()?.start.get())?;
    Ok(())
}

impl Kernels {
    /// Encode one router dispatch on `encoder`. The shape and every window
    /// are checked first; nothing is encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn router(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouterParams<'_>,
    ) -> Result<(), DeviceError> {
        check(&p)?;
        let top_k = word("router", "top-k", p.top_k)?;
        let groups = word("router", "the group count", GROUPS)?;
        let args = args_bytes([top_k, p.layer, p.eps.to_bits(), groups]);
        encoder.set_pipeline(&self.router);
        encoder.set_binding(0, Binding::of(&p.routed)?)?;
        encoder.set_binding(1, Binding::of(&p.shared)?)?;
        encoder.set_binding(2, Binding::of(&p.x)?)?;
        encoder.set_binding(3, Binding::of(&p.gain)?)?;
        encoder.set_binding(4, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(5, Binding::of_mut(&p.ticket)?)?;
        encoder.set_binding(6, Binding::of_mut(&p.logits)?)?;
        encoder.set_binding(7, Binding::of_mut(&p.ids)?)?;
        encoder.set_binding(8, Binding::of_mut(&p.weights)?)?;
        encoder.set_binding(9, Binding::bytes(&args)?)?;
        encoder.set_binding(10, p.fault.binding())?;
        encoder.dispatch(GROUPS * THREADS, THREADS)
    }
}
