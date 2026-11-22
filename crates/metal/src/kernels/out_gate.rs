//! The attention output gate of an Inkling 2 visit: `y = kqv_out ⊙ σ(g)`, one
//! gate value a channel (`kernels/out_gate.metal`), one dispatch.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "out gate";

/// Threads in the gate's threadgroup.
const THREADS: usize = 256;

/// One gate launch over `n` values.
pub struct OutGateParams<'w> {
    /// Values: the hidden width times the tokens.
    pub n: usize,
    /// The attention's output, `n`.
    pub kqv: Window<'w, f32, MetalDevice>,
    /// The gate projection's output, `n`.
    pub g: Window<'w, f32, MetalDevice>,
    /// The gated output, `n`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

impl Kernels {
    /// Encode the gate: one dispatch, a thread a value. The shape and every
    /// window are checked first; a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn out_gate(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: OutGateParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.n == 0 {
            return Err(range(WHAT, "0 values"));
        }
        check_window_len(WHAT, "kqv", p.kqv.len(), p.n)?;
        check_window_len(WHAT, "g", p.g.len(), p.n)?;
        check_window_len(WHAT, "y", p.y.len(), p.n)?;
        let args = [word(WHAT, "n", p.n)?, p.layer].map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.out_gate);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.kqv)?)?;
        slot.set(Binding::of(&p.g)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.n.div_ceil(THREADS), 1, THREADS)
    }
}
