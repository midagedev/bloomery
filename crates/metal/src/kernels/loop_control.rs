//! The looped core's message controller of Inkling 2: the merge of a message
//! mixed from the previous pass's banked blocks into a sublayer's input,
//! `out = (1 − tanh(F·h)) · cur + tanh(G·h) · msg` with
//! `h = D · [rms(cur); rms(msg)]` (`kernels/loop_control.metal`).
//!
//! One launch is one module of a core layer (its own down, gain and forget
//! rows), a threadgroup of [`CONTROLLER_THREADS`] threads a row. The weights
//! are f32 in the file's own memory order: `down` is `[rank][2n]`, `gain` and
//! `forget` are `[n][rank]`.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{CONTROLLER_RANK, CONTROLLER_THREADS, HIDDEN, controller_holds};

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "loop control";

/// One controller launch over `m` rows.
pub struct LoopControlParams<'w> {
    /// The hidden width. The kernel stages the Mini's [`HIDDEN`]; another
    /// width is refused by name.
    pub n: usize,
    /// The gate rank, [`CONTROLLER_RANK`].
    pub rank: usize,
    /// Rows.
    pub m: usize,
    /// The stream rows, `m · n`.
    pub cur: Window<'w, f32, MetalDevice>,
    /// The message rows, `m · n`.
    pub msg: Window<'w, f32, MetalDevice>,
    /// The down projection, `rank · 2n`.
    pub down: Window<'w, f32, MetalDevice>,
    /// The gain rows, `n · rank`.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The forget rows, `n · rank`.
    pub forget: Window<'w, f32, MetalDevice>,
    /// The merged rows, `m · n`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The RMS epsilon of the two halves' norm.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode one module's controller: one dispatch, a threadgroup a row. The
    /// shape and every window are checked first; a refused launch encodes
    /// nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn loop_control(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: LoopControlParams<'_>,
    ) -> Result<(), DeviceError> {
        controller_holds(p.n).map_err(|e| refuse(e.to_string()))?;
        if p.n != HIDDEN {
            return Err(refuse(format!(
                "hidden {}, the kernel stages {HIDDEN}",
                p.n
            )));
        }
        if p.rank != CONTROLLER_RANK {
            return Err(refuse(format!(
                "rank {}, the kernel reads {CONTROLLER_RANK}",
                p.rank
            )));
        }
        if p.m == 0 {
            return Err(refuse("0 rows".to_string()));
        }
        if !(p.eps.is_finite() && p.eps > 0.0) {
            return Err(refuse(format!("eps {} is not finite and positive", p.eps)));
        }
        let row =
            p.m.checked_mul(p.n)
                .ok_or_else(|| refuse(format!("{} rows wrap", p.m)))?;
        check_window_len(WHAT, "cur", p.cur.len(), row)?;
        check_window_len(WHAT, "msg", p.msg.len(), row)?;
        check_window_len(WHAT, "y", p.y.len(), row)?;
        check_window_len(WHAT, "down", p.down.len(), p.rank * 2 * p.n)?;
        check_window_len(WHAT, "gain", p.gain.len(), p.n * p.rank)?;
        check_window_len(WHAT, "forget", p.forget.len(), p.n * p.rank)?;
        let args = [
            word(WHAT, "n", p.n)?,
            word(WHAT, "rank", p.rank)?,
            word(WHAT, "m", p.m)?,
            p.layer,
            p.eps.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.loop_control);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.cur)?)?;
        slot.set(Binding::of(&p.msg)?)?;
        slot.set(Binding::of(&p.down)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of(&p.forget)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.m, 1, CONTROLLER_THREADS)
    }
}
