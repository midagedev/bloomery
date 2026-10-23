//! The launch of the router: 257 rows of K 2048, split two ways, one dispatch.
//!
//! The kernel is `kernels/router.metal`. A launch checks the shape, every
//! window's length and the alignment of the windows the kernel loads as
//! float4 before it encodes anything. A launch that is refused encodes nothing.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

pub use super::router_geom::{EXPERTS, K, PARTIALS, ROWS, TOP_K_MAX};
use super::router_geom::{FLOAT4S, GROUPS, SIMD_WIDTH, SPLIT, THREADS};
use super::{Kernels, check_window_len, pack, word};

const WHAT: &str = "router";

const _: () = assert!(PARTIALS == ROWS * SPLIT);
const _: () = assert!(GROUPS == 129);
const _: () = assert!(THREADS == 128);
const _: () = assert!(SIMD_WIDTH * FLOAT4S * 4 == K / SPLIT);
const _: () = assert!(SIMD_WIDTH * FLOAT4S * SPLIT * 4 == K);
const _: () = assert!(SIMD_WIDTH == super::geometry::SIMD_WIDTH);

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

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

fn align16(name: &str, start: usize) -> Result<(), DeviceError> {
    super::check_weights_align(WHAT, name, start, super::WeightAlign::Float4)
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
    check_window_len(WHAT, "routed", p.routed.len(), EXPERTS * K)?;
    check_window_len(WHAT, "shared", p.shared.len(), K)?;
    check_window_len(WHAT, "x", p.x.len(), K)?;
    check_window_len(WHAT, "gain", p.gain.len(), K)?;
    check_window_len(WHAT, "partials", p.partials.len(), PARTIALS)?;
    check_window_len(WHAT, "ticket", p.ticket.len(), 1)?;
    check_window_len(WHAT, "logits", p.logits.len(), ROWS)?;
    check_window_len(WHAT, "ids", p.ids.len(), p.top_k + 1)?;
    check_window_len(WHAT, "weights", p.weights.len(), p.top_k + 1)?;
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
        let top_k = word(WHAT, "top-k", p.top_k)?;
        // Three words: top-k, the layer, the epsilon. The kernel reads the
        // threadgroup count from the grid.
        let packed = [
            top_k.to_ne_bytes(),
            p.layer.to_ne_bytes(),
            p.eps.to_bits().to_ne_bytes(),
        ];
        encoder.set_pipeline(&self.router);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.routed)?)?;
        slot.set(Binding::of(&p.shared)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.partials)?)?;
        slot.set(Binding::of_mut(&p.ticket)?)?;
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(Binding::of_mut(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.weights)?)?;
        slot.set(Binding::bytes(pack(&packed))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(GROUPS, 1, THREADS)
    }
}
