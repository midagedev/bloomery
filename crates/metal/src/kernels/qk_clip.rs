//! The projection epilogues of an Inkling 2 rel-bias visit and the f16 append
//! of its key and value: one dispatch covers the query rows, the key rows, the
//! value rows and the relative rows (`kernels/qk_clip.metal`). The row counts
//! and widths are the family's own (`inkling2_geom`): 24 query heads and 8
//! key heads of 128 values, 16 relative values a query head.
//!
//! [`Kernels::qk_clip_conv_append`] is the same launch with the key and value
//! short convs run in its tail (`qk_clip_conv_append`): the conv of the raw key
//! and value rows, then the clip of its output, bit for bit what the two conv
//! launches and this one wrote. [`Kernels::qk_clip_append`] stays as the launch
//! it is held to.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{CLIP_GROUPS, CLIP_THREADS, D_REL, HEAD, KV_HEADS, QUERY_HEADS, TAPS};

use super::{
    Kernels, WeightAlign, check_weights_align, check_window_len, conv_scale, pack, range, word,
};

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
    /// The query rows' scale, one a (head, channel): `24 · 128`, head-major as the
    /// file's `attn_q_clip` rows.
    pub gq: Window<'w, f32, MetalDevice>,
    /// The relative rows' scale, one a (head, value): `24 · 16`, head-major as
    /// the file's `attn_rel_clip` rows.
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

/// The key and value short convs a fused launch runs: the raw projection rows,
/// the filters, the head's dynamic taps and the visit's ring.
pub struct KvConvParams<'w> {
    /// The key conv's input, `8 · 128`: the projection's raw row.
    pub k_raw: Window<'w, f32, MetalDevice>,
    /// The value conv's input, `8 · 128`.
    pub v_raw: Window<'w, f32, MetalDevice>,
    /// The key conv's static filters, `8 · 128 · taps`, a channel's taps together, oldest first.
    pub wk: Window<'w, f32, MetalDevice>,
    /// The value conv's static filters.
    pub wv: Window<'w, f32, MetalDevice>,
    /// The key conv's dynamic taps, `8 · taps`, held exactly when the visit is windowless.
    pub dyn_k: Option<Window<'w, f32, MetalDevice>>,
    /// The value conv's dynamic taps, held with the key's.
    pub dyn_v: Option<Window<'w, f32, MetalDevice>>,
    /// The filters' taps: the family's [`TAPS`].
    pub taps: usize,
    /// The visit's conv ring, `ring_rows · ch`, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The ring's rows: at least `taps − 1`.
    pub ring_rows: usize,
    /// The ring's row width.
    pub ch: usize,
    /// The key conv's first channel in the ring's rows.
    pub base_k: usize,
    /// The value conv's first channel.
    pub base_v: usize,
    /// This token's position in the sequence: the convs', not the plane's
    /// append position in `clip.pos` (a windowed plane wraps; the ring does not).
    pub pos: u32,
}

/// One fused epilogue launch: the clip's rows, with `k` and `v` the outputs of
/// the convs of `conv`'s raw rows.
pub struct QkClipConvParams<'w> {
    /// The clip's windows and constants; `k` and `v` are only written.
    pub clip: QkClipParams<'w>,
    /// The convs.
    pub conv: KvConvParams<'w>,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

/// The clip's shape and windows, refused by name, and its argument words.
fn check_clip(p: &QkClipParams<'_>) -> Result<[u32; 7], DeviceError> {
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
    check_window_len(WHAT, "gq", p.gq.len(), QUERY_HEADS * HEAD)?;
    check_window_len(WHAT, "gr", p.gr.len(), QUERY_HEADS * D_REL)?;
    let plane = KV_HEADS
        .checked_mul(p.ctx)
        .and_then(|v| v.checked_mul(HEAD))
        .ok_or_else(|| refuse(format!("a plane of {} positions wraps", p.ctx)))?;
    word(WHAT, "the plane length", plane)?;
    check_window_len(WHAT, "key plane", p.cache_k.len(), plane)?;
    check_window_len(WHAT, "value plane", p.cache_v.len(), plane)?;
    Ok([
        word(WHAT, "ctx", p.ctx)?,
        p.pos,
        p.layer,
        p.v_eps.to_bits(),
        p.qk_thresh.to_bits(),
        p.r_thresh.to_bits(),
        p.tau.to_bits(),
    ])
}

/// Bind the clip's windows (slots 0 to 7).
fn bind_clip(slot: &mut Bindings<'_, '_>, p: &QkClipParams<'_>) -> Result<(), DeviceError> {
    slot.set(Binding::of_mut(&p.q)?)?;
    slot.set(Binding::of_mut(&p.k)?)?;
    slot.set(Binding::of_mut(&p.v)?)?;
    slot.set(Binding::of_mut(&p.r)?)?;
    slot.set(Binding::of(&p.gq)?)?;
    slot.set(Binding::of(&p.gr)?)?;
    slot.set(Binding::of_mut(&p.cache_k)?)?;
    slot.set(Binding::of_mut(&p.cache_v)?)
}

/// The convs' shape and windows, refused by name.
fn check_conv(c: &KvConvParams<'_>) -> Result<(), DeviceError> {
    if c.taps != TAPS {
        return Err(refuse(format!(
            "{} taps, the epilogue is written for {TAPS}",
            c.taps
        )));
    }
    if c.ring_rows < c.taps - 1 {
        return Err(refuse(format!(
            "ring rows {} are fewer than the {} predecessors a token reads",
            c.ring_rows,
            c.taps - 1
        )));
    }
    let width = KV_HEADS * HEAD;
    for base in [c.base_k, c.base_v] {
        if base.checked_add(width).is_none_or(|end| end > c.ch) {
            return Err(refuse(format!(
                "channels {base}..+{width} pass the ring's {}",
                c.ch
            )));
        }
    }
    check_window_len(WHAT, "k raw", c.k_raw.len(), width)?;
    check_window_len(WHAT, "v raw", c.v_raw.len(), width)?;
    check_window_len(WHAT, "wk", c.wk.len(), width * c.taps)?;
    check_window_len(WHAT, "wv", c.wv.len(), width * c.taps)?;
    match (&c.dyn_k, &c.dyn_v) {
        (None, None) => {}
        (Some(k), Some(v)) => {
            check_window_len(WHAT, "dynamic key taps", k.len(), KV_HEADS * c.taps)?;
            check_window_len(WHAT, "dynamic value taps", v.len(), KV_HEADS * c.taps)?;
        }
        _ => {
            return Err(refuse(
                "the key and value convs' dynamic taps are held together".to_string(),
            ));
        }
    }
    let ring = c
        .ring_rows
        .checked_mul(c.ch)
        .ok_or_else(|| refuse(format!("the ring of {} rows wraps", c.ring_rows)))?;
    word(WHAT, "the ring length", ring)?;
    check_window_len(WHAT, "ring", c.ring.len(), ring)?;
    let mut starts = vec![
        ("k raw", c.k_raw.byte_range()?.start.get()),
        ("v raw", c.v_raw.byte_range()?.start.get()),
        ("wk", c.wk.byte_range()?.start.get()),
        ("wv", c.wv.byte_range()?.start.get()),
        ("ring", c.ring.byte_range()?.start.get()),
    ];
    if let (Some(k), Some(v)) = (&c.dyn_k, &c.dyn_v) {
        starts.push(("dynamic key taps", k.byte_range()?.start.get()));
        starts.push(("dynamic value taps", v.byte_range()?.start.get()));
    }
    for (name, start) in starts {
        check_weights_align(WHAT, name, start, WeightAlign::Float4)?;
    }
    Ok(())
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
        let args = check_clip(&p)?.map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.qk_clip);
        let mut slot = Bindings::new(encoder);
        bind_clip(&mut slot, &p)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(CLIP_GROUPS, 1, CLIP_THREADS)
    }

    /// Encode the epilogue with the key and value convs run in its tail: one
    /// dispatch of [`CLIP_GROUPS`] threadgroups, a SIMD group a row, the key
    /// and value rows' groups convolving their raw rows first. The shape and
    /// every window are checked first; a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn qk_clip_conv_append(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: QkClipConvParams<'_>,
    ) -> Result<(), DeviceError> {
        let clip = check_clip(&p.clip)?;
        let c = &p.conv;
        check_conv(c)?;
        let heads = if c.dyn_k.is_some() { KV_HEADS } else { 0 };
        let tail = [
            word(WHAT, "ring rows", c.ring_rows)?,
            word(WHAT, "ch", c.ch)?,
            word(WHAT, "key base", c.base_k)?,
            word(WHAT, "value base", c.base_v)?,
            word(WHAT, "dynamic heads", heads)?,
            c.pos,
            conv_scale(c.taps, heads > 0).to_bits(),
        ];
        let words: Vec<[u8; 4]> = clip.into_iter().chain(tail).map(u32::to_ne_bytes).collect();
        encoder.set_pipeline(&self.ink2.qk_clip_conv);
        let mut slot = Bindings::new(encoder);
        bind_clip(&mut slot, &p.clip)?;
        slot.set(Binding::bytes(words.as_flattened())?)?;
        slot.set(p.clip.faults.binding())?;
        slot.set(Binding::of(&c.k_raw)?)?;
        slot.set(Binding::of(&c.v_raw)?)?;
        slot.set(Binding::of(&c.wk)?)?;
        slot.set(Binding::of(&c.wv)?)?;
        // A static visit binds the filters where the taps would be, as conv_res does.
        slot.set(Binding::of(c.dyn_k.as_ref().unwrap_or(&c.wk))?)?;
        slot.set(Binding::of(c.dyn_v.as_ref().unwrap_or(&c.wv))?)?;
        slot.set(Binding::of_mut(&c.ring)?)?;
        slot.dispatch_groups(CLIP_GROUPS, 1, CLIP_THREADS)
    }
}
