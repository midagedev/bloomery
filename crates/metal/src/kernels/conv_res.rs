//! The residual short conv of an Inkling 2 visit: a causal depthwise conv
//! over the last `taps` inputs of each channel, the current input added as a
//! residual and the whole scaled by `1/√((dynamic ? 2 : 1) · taps + 1)`; the
//! windowless layers' k and v convs carry a second set of input-dependent
//! taps (`kernels/conv_res.metal`).
//!
//! The four convs of a visit read and write disjoint channel ranges of one
//! ring of `ch` channels a row: this conv's range starts at channel `base`.
//! The ring holds the input of position `p` in row `p mod rows`; a
//! predecessor before the sequence's first position is 0 whatever the ring
//! holds.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{CONV_THREADS, MAX_TAPS, conv_groups};

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "conv res";

/// One conv launch over `m` tokens from position `pos`.
pub struct ConvResParams<'w> {
    /// The conv's channels.
    pub d: usize,
    /// Filter taps, `2..=`[`MAX_TAPS`].
    pub taps: usize,
    /// The dynamic taps' heads: 0 for a static conv, else a divisor of `d`.
    pub heads: usize,
    /// The ring's rows: at least the `taps − 1` predecessors a token reads (a
    /// channel's thread reads its slots before it writes the call's).
    pub rows: usize,
    /// The ring's row width: the visit's whole channel width.
    pub ch: usize,
    /// This conv's first channel in the ring's rows: `base + d <= ch`.
    pub base: usize,
    /// Tokens in the call.
    pub m: usize,
    /// The position of the call's first token.
    pub pos: u32,
    /// The input rows, `m · d`.
    pub x: Window<'w, f32, MetalDevice>,
    /// The static filters, `d · taps`, a channel's taps together, oldest first.
    pub w: Window<'w, f32, MetalDevice>,
    /// The dynamic taps, `m · heads · taps`; held exactly when `heads` is not 0.
    pub dyn_taps: Option<Window<'w, f32, MetalDevice>>,
    /// The output rows, `m · d`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The visit's ring, `rows · ch`, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The output scale of a conv of `taps` taps: `1/√(taps + 1)` of a static
/// conv, `1/√(2 · taps + 1)` of a dynamic one.
#[must_use]
pub fn conv_scale(taps: usize, dynamic: bool) -> f32 {
    let terms = if dynamic { 2 * taps + 1 } else { taps + 1 };
    (1.0 / (terms as f64).sqrt()) as f32
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the conv: one dispatch over the channels by the call's units
    /// (the first `taps − 1` tokens and the ring store, then a token each).
    /// The shape and every window are checked first; a refused launch
    /// encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn conv_res(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: ConvResParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.d == 0 || p.m == 0 {
            return Err(refuse(format!("{} channels of {} tokens", p.d, p.m)));
        }
        if !(2..=MAX_TAPS).contains(&p.taps) {
            return Err(refuse(format!(
                "{} taps, the kernel takes 2 to {MAX_TAPS}",
                p.taps
            )));
        }
        if p.rows < p.taps - 1 {
            return Err(refuse(format!(
                "ring rows {} are fewer than the {} predecessors a token reads",
                p.rows,
                p.taps - 1
            )));
        }
        if p.base.checked_add(p.d).is_none_or(|end| end > p.ch) {
            return Err(refuse(format!(
                "channels {}..+{} pass the ring's {}",
                p.base, p.d, p.ch
            )));
        }
        if p.heads > 0 && !p.d.is_multiple_of(p.heads) {
            return Err(refuse(format!(
                "{} channels are not whole over {} heads",
                p.d, p.heads
            )));
        }
        let rows = word(WHAT, "m", p.m)?;
        if p.pos.checked_add(rows).is_none() {
            return Err(refuse(format!(
                "positions {}..+{rows} pass the 32-bit position word",
                p.pos
            )));
        }
        let io =
            p.m.checked_mul(p.d)
                .ok_or_else(|| refuse(format!("{} tokens of {} channels wrap", p.m, p.d)))?;
        check_window_len(WHAT, "x", p.x.len(), io)?;
        check_window_len(WHAT, "y", p.y.len(), io)?;
        check_window_len(WHAT, "w", p.w.len(), p.d * p.taps)?;
        let ring = p
            .rows
            .checked_mul(p.ch)
            .ok_or_else(|| refuse(format!("the ring of {} rows wraps", p.rows)))?;
        word(WHAT, "the ring length", ring)?;
        check_window_len(WHAT, "ring", p.ring.len(), ring)?;
        match (&p.dyn_taps, p.heads) {
            (None, 0) => {}
            (Some(taps), heads) if heads > 0 => {
                check_window_len(WHAT, "dynamic taps", taps.len(), p.m * heads * p.taps)?;
            }
            (Some(_), _) => return Err(refuse("dynamic taps of a static conv".to_string())),
            (None, heads) => return Err(refuse(format!("{heads} dynamic heads, no taps"))),
        }
        let units = if p.m > p.taps - 1 {
            1 + p.m - (p.taps - 1)
        } else {
            1
        };
        let words = [
            word(WHAT, "d", p.d)?,
            word(WHAT, "taps", p.taps)?,
            word(WHAT, "heads", p.heads)?,
            word(WHAT, "rows", p.rows)?,
            word(WHAT, "ch", p.ch)?,
            word(WHAT, "base", p.base)?,
            rows,
            p.layer,
            p.pos,
            conv_scale(p.taps, p.heads > 0).to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.conv_res);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.w)?)?;
        slot.set(Binding::of(p.dyn_taps.as_ref().unwrap_or(&p.w))?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::bytes(pack(&words))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(conv_groups(p.d), units, CONV_THREADS)
    }
}
