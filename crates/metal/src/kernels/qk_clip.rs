//! The projection epilogues of an Inkling 2 rel-bias visit and the f16 append
//! of its key and value: one dispatch covers the query rows, the key rows, the
//! value rows and the relative rows (`kernels/qk_clip.metal`). The row counts
//! and widths are the family's own (`inkling2_geom`): 24 query heads and 8
//! key heads of 128 values, 16 relative values a query head.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{CLIP_GROUPS, CLIP_THREADS, D_REL, HEAD, KV_HEADS, QUERY_HEADS};

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "qk clip";

/// One epilogue launch: the rows of one token.
pub struct QkClipParams<'w> {
    /// The query rows, `24 · 128`, clipped in place.
    pub q: WindowMut<'w, f32, MetalDevice>,
    /// The key rows, `8 · 128`, clipped in place.
    pub k: WindowMut<'w, f32, MetalDevice>,
    /// The value rows, `8 · 128`, normed in place.
    pub v: WindowMut<'w, f32, MetalDevice>,
    /// The relative rows, `24 · 16`, clipped in place.
    pub r: WindowMut<'w, f32, MetalDevice>,
    /// The query rows' per-channel scale, 128 values.
    pub gq: Window<'w, f32, MetalDevice>,
    /// The relative rows' per-channel scale, 16 values.
    pub gr: Window<'w, f32, MetalDevice>,
    /// The visit's key plane, `8 · ctx · 128` f16 bits, head-major.
    pub cache_k: WindowMut<'w, u16, MetalDevice>,
    /// The visit's value plane, as the key plane.
    pub cache_v: WindowMut<'w, u16, MetalDevice>,
    /// The plane's positions.
    pub ctx: usize,
    /// This token's append position.
    pub pos: u32,
    /// The value rows' RMS epsilon.
    pub v_eps: f32,
    /// The query and key rows' L2 threshold.
    pub qk_thresh: f32,
    /// The relative rows' L2 threshold.
    pub r_thresh: f32,
    /// The windowless visits' log-scaled position factor; 1.0 windowed.
    pub tau: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the epilogue: one dispatch of [`CLIP_GROUPS`] threadgroups, a
    /// SIMD group a row. The shape and every window are checked first; a
    /// refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn qk_clip_append(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: QkClipParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.ctx == 0 {
            return Err(refuse("a plane of 0 positions".to_string()));
        }
        for (name, v) in [
            ("v_eps", p.v_eps),
            ("qk threshold", p.qk_thresh),
            ("r threshold", p.r_thresh),
            ("tau", p.tau),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(refuse(format!("{name} {v} is not finite and positive")));
            }
        }
        check_window_len(WHAT, "q", p.q.len(), QUERY_HEADS * HEAD)?;
        check_window_len(WHAT, "k", p.k.len(), KV_HEADS * HEAD)?;
        check_window_len(WHAT, "v", p.v.len(), KV_HEADS * HEAD)?;
        check_window_len(WHAT, "r", p.r.len(), QUERY_HEADS * D_REL)?;
        check_window_len(WHAT, "gq", p.gq.len(), HEAD)?;
        check_window_len(WHAT, "gr", p.gr.len(), D_REL)?;
        let plane = KV_HEADS
            .checked_mul(p.ctx)
            .and_then(|v| v.checked_mul(HEAD))
            .ok_or_else(|| refuse(format!("a plane of {} positions wraps", p.ctx)))?;
        word(WHAT, "the plane length", plane)?;
        check_window_len(WHAT, "key plane", p.cache_k.len(), plane)?;
        check_window_len(WHAT, "value plane", p.cache_v.len(), plane)?;
        let args = [
            word(WHAT, "ctx", p.ctx)?,
            p.pos,
            p.layer,
            p.v_eps.to_bits(),
            p.qk_thresh.to_bits(),
            p.r_thresh.to_bits(),
            p.tau.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.qk_clip);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.q)?)?;
        slot.set(Binding::of_mut(&p.k)?)?;
        slot.set(Binding::of_mut(&p.v)?)?;
        slot.set(Binding::of_mut(&p.r)?)?;
        slot.set(Binding::of(&p.gq)?)?;
        slot.set(Binding::of(&p.gr)?)?;
        slot.set(Binding::of_mut(&p.cache_k)?)?;
        slot.set(Binding::of_mut(&p.cache_v)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(CLIP_GROUPS, 1, CLIP_THREADS)
    }
}
