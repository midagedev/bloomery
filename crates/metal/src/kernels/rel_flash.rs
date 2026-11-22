//! The relative-position bias attention of one Inkling 2 visit, at a decode
//! step of one query row (`kernels/rel_flash.metal`): the decode attention's
//! segment walk at the group of three, the score of each key gaining
//! `Σ_{d<16} rel_bias[d, live − 1 − key] · r_d` and the softmax's denominator a
//! per-head sink logit that carries no value. No rope runs: the q, k, v and r
//! rows arrive clipped and appended by the epilogue (`qk_clip.rs`).
//!
//! Two dispatches, the segment walk and the fold of the segments' partials
//! with the sink, a memory barrier between: the fold reads the partials
//! through a device fence a dispatch of its own after the walk's. The cache
//! planes are f16 bits head-major `[kv][rows][head]`; a position's row is its
//! position modulo `rows`, so a plane passes its context as `rows` and a
//! windowed visit's ring its own row count.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::attention_geom::SEGMENT_CAP;
use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{D_REL, EXTENT, FLASH_THREADS, HEAD, KV_HEADS, QUERY_HEADS};

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "rel flash";

/// One flash launch.
pub struct RelFlashParams<'w> {
    /// The segments the walk takes, `1..=`[`SEGMENT_CAP`]
    /// (`segments_for_window` names the count a visit's keys want).
    pub segments: usize,
    /// The context the live count is bounded by.
    pub ctx: usize,
    /// The planes' rows per key head.
    pub rows: usize,
    /// The live keys, this position's own among them.
    pub keys: usize,
    /// The last `window` of the live keys attended; 0 for all.
    pub window: usize,
    /// The scores' scale, `1/√head` the caller derives.
    pub scale: f32,
    /// The query rows, `24 · 128`.
    pub q: Window<'w, f32, MetalDevice>,
    /// The key plane, `8 · rows · 128` f16 bits.
    pub cache_k: Window<'w, u16, MetalDevice>,
    /// The value plane, as the key plane.
    pub cache_v: Window<'w, u16, MetalDevice>,
    /// The relative rows, `24 · 16`.
    pub r: Window<'w, f32, MetalDevice>,
    /// The bias table, `16 · 1024`: row `d` holds the distances' columns.
    pub bias: Window<'w, f32, MetalDevice>,
    /// The sinks, one a query head.
    pub sinks: Window<'w, f32, MetalDevice>,
    /// The segments' partials, `24 · segments · 130`.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The attention's output, `24 · 128`.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

/// The partials one launch of `segments` segments writes: a `(max, Σ exp, Σ exp · V)`
/// row of `head + 2` values a (query head, segment).
#[must_use]
pub fn rel_partials_len(segments: usize) -> Option<usize> {
    QUERY_HEADS.checked_mul(segments)?.checked_mul(HEAD + 2)
}

impl Kernels {
    /// Encode the visit's attention: the segment walk, a barrier, the fold. The
    /// shape and every window are checked first; a refused launch encodes
    /// nothing. A live count of zero or past `ctx` is the kernel's to raise.
    #[must_use = "a refused launch encoded nothing"]
    pub fn rel_flash(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RelFlashParams<'_>,
    ) -> Result<(), DeviceError> {
        if !(1..=SEGMENT_CAP).contains(&p.segments) {
            return Err(refuse(format!(
                "{} segments, the kernel takes 1 to {SEGMENT_CAP}",
                p.segments
            )));
        }
        if p.rows == 0 || p.ctx == 0 {
            return Err(refuse(format!(
                "{} rows over a context of {}",
                p.rows, p.ctx
            )));
        }
        if !(p.scale.is_finite() && p.scale > 0.0) {
            return Err(refuse(format!(
                "scale {} is not finite and positive",
                p.scale
            )));
        }
        let plane = KV_HEADS
            .checked_mul(p.rows)
            .and_then(|v| v.checked_mul(HEAD))
            .ok_or_else(|| refuse(format!("planes of {} rows wrap", p.rows)))?;
        word(WHAT, "the plane length", plane)?;
        check_window_len(WHAT, "q", p.q.len(), QUERY_HEADS * HEAD)?;
        check_window_len(WHAT, "key plane", p.cache_k.len(), plane)?;
        check_window_len(WHAT, "value plane", p.cache_v.len(), plane)?;
        check_window_len(WHAT, "r", p.r.len(), QUERY_HEADS * D_REL)?;
        check_window_len(WHAT, "bias", p.bias.len(), D_REL * EXTENT)?;
        check_window_len(WHAT, "sinks", p.sinks.len(), QUERY_HEADS)?;
        let partials = rel_partials_len(p.segments)
            .ok_or_else(|| refuse(format!("{} segments wrap", p.segments)))?;
        check_window_len(WHAT, "partials", p.partials.len(), partials)?;
        check_window_len(WHAT, "out", p.out.len(), QUERY_HEADS * HEAD)?;
        let args = [
            word(WHAT, "segments", p.segments)?,
            word(WHAT, "ctx", p.ctx)?,
            word(WHAT, "rows", p.rows)?,
            word(WHAT, "keys", p.keys)?,
            word(WHAT, "window", p.window)?,
            p.layer,
            p.scale.to_bits(),
        ]
        .map(u32::to_ne_bytes);

        encoder.set_pipeline(&self.ink2.rel_flash);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.q)?)?;
        slot.set(Binding::of(&p.cache_k)?)?;
        slot.set(Binding::of(&p.cache_v)?)?;
        slot.set(Binding::of(&p.r)?)?;
        slot.set(Binding::of(&p.bias)?)?;
        slot.set(Binding::of_mut(&p.partials)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(KV_HEADS * p.segments, 1, FLASH_THREADS)?;

        encoder.memory_barrier();
        encoder.set_pipeline(&self.ink2.rel_fold);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.partials)?)?;
        slot.set(Binding::of(&p.sinks)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(KV_HEADS, 1, FLASH_THREADS)
    }
}
