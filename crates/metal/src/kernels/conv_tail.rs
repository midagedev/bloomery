//! The elementwise tails of an Inkling 2 decode visit, run in the launch that
//! makes their input (`kernels/conv_tail.metal`): the attention's post conv
//! and the stream's add as one group, and the experts' sum, the feed-forward's
//! post conv and the stream's add as one. Each channel's value is what the
//! separate launches ([`Kernels::conv_res`], [`Kernels::bias_rows`],
//! [`Kernels::combine_rows`]) wrote.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{TAPS, TOP_K};

use super::rows::check_strided;
use super::{
    Kernels, WeightAlign, check_weights_align, check_window_len, conv_scale, pack, range, word,
};

/// Threads in the tail launches' threadgroup: two SIMD groups, a float4 a thread.
const TAIL_THREADS: usize = 64;

const COMBINE: &str = "combine conv join";
const JOIN: &str = "conv res join";

/// The post conv a tail launch runs: the filters, the ring and the position.
pub struct TailConv<'w> {
    /// The static filters, `cols · taps`, a channel's taps together, oldest first.
    pub w: Window<'w, f32, MetalDevice>,
    /// The filter's taps: the family's [`TAPS`].
    pub taps: usize,
    /// The visit's conv ring, `ring_rows · ch`, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The ring's rows: at least `taps − 1`.
    pub ring_rows: usize,
    /// The ring's row width.
    pub ch: usize,
    /// This conv's first channel in the ring's rows: `base + cols <= ch`.
    pub base: usize,
    /// This token's position.
    pub pos: u32,
}

/// One decode visit's attention post conv and stream add.
pub struct ConvResJoinParams<'w> {
    /// The conv's channels: a multiple of four.
    pub cols: usize,
    /// The conv's input, `cols`.
    pub x: Window<'w, f32, MetalDevice>,
    /// The conv.
    pub conv: TailConv<'w>,
    /// The conv's output, `cols`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The stream's row, `cols`: the conv's output is added into it.
    pub stream: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// One decode visit's expert sum, post conv and stream add.
pub struct CombineConvJoinParams<'w> {
    /// The row's width: a multiple of four.
    pub cols: usize,
    /// The slots a row: the family's [`TOP_K`].
    pub slots: usize,
    /// The slots' rows, in slot order, `y_stride` floats apart.
    pub y: Window<'w, f32, MetalDevice>,
    /// Floats between the slots' rows: at least `cols`, a multiple of four.
    pub y_stride: usize,
    /// The visit's sum row, `cols`: the slots' sum is added into it, and it is the conv's input.
    pub x: WindowMut<'w, f32, MetalDevice>,
    /// The conv.
    pub conv: TailConv<'w>,
    /// The conv's output, `cols`.
    pub conv_out: WindowMut<'w, f32, MetalDevice>,
    /// The stream's row, `cols`: the conv's output is added into it.
    pub stream: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The conv's shape and windows for `cols` channels, refused by name.
fn check_conv(what: &'static str, cols: usize, c: &TailConv<'_>) -> Result<(), DeviceError> {
    if cols == 0 || !cols.is_multiple_of(4) {
        return Err(range(what, format!("{cols} columns are not whole float4s")));
    }
    if c.taps != TAPS {
        return Err(range(
            what,
            format!("{} taps, the tail is written for {TAPS}", c.taps),
        ));
    }
    if c.ring_rows < c.taps - 1 {
        return Err(range(
            what,
            format!(
                "ring rows {} are fewer than the {} predecessors a token reads",
                c.ring_rows,
                c.taps - 1
            ),
        ));
    }
    if !c.ch.is_multiple_of(4) || !c.base.is_multiple_of(4) {
        return Err(range(
            what,
            format!(
                "the ring width {} and the base {} are not whole float4s",
                c.ch, c.base
            ),
        ));
    }
    if c.base.checked_add(cols).is_none_or(|end| end > c.ch) {
        return Err(range(
            what,
            format!("channels {}..+{cols} pass the ring's {}", c.base, c.ch),
        ));
    }
    check_window_len(what, "w", c.w.len(), cols * c.taps)?;
    let ring = c
        .ring_rows
        .checked_mul(c.ch)
        .ok_or_else(|| range(what, format!("the ring of {} rows wraps", c.ring_rows)))?;
    word(what, "the ring length", ring)?;
    check_window_len(what, "ring", c.ring.len(), ring)?;
    for (name, start) in [
        ("w", c.w.byte_range()?.start.get()),
        ("ring", c.ring.byte_range()?.start.get()),
    ] {
        check_weights_align(what, name, start, WeightAlign::Float4)?;
    }
    Ok(())
}

/// A window's start is 16-byte aligned: the tails read and write float4s.
fn check_float4(what: &'static str, name: &str, start: usize) -> Result<(), DeviceError> {
    check_weights_align(what, name, start, WeightAlign::Float4)
}

impl Kernels {
    /// Encode the attention's post conv and the stream's add: one dispatch, a
    /// thread a float4 of the row. The shape and every window are checked
    /// first; a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn conv_res_join(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: ConvResJoinParams<'_>,
    ) -> Result<(), DeviceError> {
        check_conv(JOIN, p.cols, &p.conv)?;
        check_window_len(JOIN, "x", p.x.len(), p.cols)?;
        check_window_len(JOIN, "y", p.y.len(), p.cols)?;
        check_window_len(JOIN, "stream", p.stream.len(), p.cols)?;
        for (name, start) in [
            ("x", p.x.byte_range()?.start.get()),
            ("y", p.y.byte_range()?.start.get()),
            ("stream", p.stream.byte_range()?.start.get()),
        ] {
            check_float4(JOIN, name, start)?;
        }
        let c = &p.conv;
        let words = [
            word(JOIN, "cols", p.cols)?,
            word(JOIN, "ring rows", c.ring_rows)?,
            word(JOIN, "ch", c.ch)?,
            word(JOIN, "base", c.base)?,
            c.pos,
            p.layer,
            conv_scale(c.taps, false).to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.conv_res_join);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&c.w)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of_mut(&c.ring)?)?;
        slot.set(Binding::of_mut(&p.stream)?)?;
        slot.set(Binding::bytes(pack(&words))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups((p.cols / 4).div_ceil(TAIL_THREADS), 1, TAIL_THREADS)
    }

    /// Encode the expert sum, the feed-forward's post conv and the stream's
    /// add: one dispatch, a thread a float4 of the row. The shape and every
    /// window are checked first; a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn combine_conv_join(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: CombineConvJoinParams<'_>,
    ) -> Result<(), DeviceError> {
        check_conv(COMBINE, p.cols, &p.conv)?;
        if p.slots != TOP_K {
            return Err(range(
                COMBINE,
                format!("{} slots, the sum is unrolled over {TOP_K}", p.slots),
            ));
        }
        if !p.y_stride.is_multiple_of(4) {
            return Err(range(
                COMBINE,
                format!("the slot stride {} is not whole float4s", p.y_stride),
            ));
        }
        check_strided(COMBINE, "y", p.y.len(), p.slots, p.y_stride, p.cols)?;
        check_window_len(COMBINE, "x", p.x.len(), p.cols)?;
        check_window_len(COMBINE, "conv out", p.conv_out.len(), p.cols)?;
        check_window_len(COMBINE, "stream", p.stream.len(), p.cols)?;
        for (name, start) in [
            ("y", p.y.byte_range()?.start.get()),
            ("x", p.x.byte_range()?.start.get()),
            ("conv out", p.conv_out.byte_range()?.start.get()),
            ("stream", p.stream.byte_range()?.start.get()),
        ] {
            check_float4(COMBINE, name, start)?;
        }
        let c = &p.conv;
        let words = [
            word(COMBINE, "cols", p.cols)?,
            word(COMBINE, "y stride", p.y_stride)?,
            word(COMBINE, "ring rows", c.ring_rows)?,
            word(COMBINE, "ch", c.ch)?,
            word(COMBINE, "base", c.base)?,
            c.pos,
            p.layer,
            conv_scale(c.taps, false).to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.combine_conv_join);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.y)?)?;
        slot.set(Binding::of_mut(&p.x)?)?;
        slot.set(Binding::of(&c.w)?)?;
        slot.set(Binding::of_mut(&p.conv_out)?)?;
        slot.set(Binding::of_mut(&c.ring)?)?;
        slot.set(Binding::of_mut(&p.stream)?)?;
        slot.set(Binding::bytes(pack(&words))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups((p.cols / 4).div_ceil(TAIL_THREADS), 1, TAIL_THREADS)
    }
}
