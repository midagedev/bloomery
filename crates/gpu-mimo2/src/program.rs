//! The MiMo layer program of the one-token step ([`StepProgram`]) and the
//! port its host legs go through ([`StepLeg`]): [`runtime::sched::walk`] over
//! the point `(1, 1, Step)`. Each layer's parts, by its programs
//! (`runtime::layer::Layer`), never its number:
//!
//! - the front: the attention sub-layer whole, then either the dense block
//!   whole or the routed block up to its go;
//! - the shadow: none — the routed block has no shared expert and no card
//!   expert, so the card has no work under the host leg;
//! - the back (a routed layer): the wait and the add of the host's sum, then
//!   the host tier told the layer is enqueued.
//!
//! A layer's output goes to its tap, where a gate armed one, once its last
//! add is enqueued. The begin starts from `x`, which the embedding wrote; the
//! last layer's add writes the head's input, and the end is the head.
//!
//! The launches per layer and the step's counts are the description's facts
//! ([`model::arch::mimo2::program`]: `step_launches`, `step_memops`).

use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use runtime::layer::FfnKind;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::Parts;
use crate::{attn, ffn};

/// What the walk's errors name.
const WHAT: &str = "mimo2 Body::enqueue_chain";

/// The step's point, `(1, 1, Step)`.
const STEP: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// Walk the step over `parts`, the host tier opened for it first, into
/// `head`.
pub(crate) fn walk_step(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<HostRun>,
    head: &mut Head,
) -> Result<(), GpuError> {
    let layers = parts.cfg.len();
    let mut port = StepLeg::new(gpu.stream(), hybrid);
    let mut prog = StepProgram {
        gpu,
        w,
        parts,
        head,
    };
    sched::walk(STEP, layers, &mut port, &mut prog)
}

/// The MiMo program over one walk: the body's parts and the head.
struct StepProgram<'s, 'w> {
    gpu: &'w Gpu,
    w: &'w Weights,
    parts: Parts<'s>,
    head: &'s mut Head,
}

impl StepProgram<'_, '_> {
    /// Whether layer `l` is the last: its output is the head's input.
    fn last(&self, l: usize) -> bool {
        l + 1 == self.parts.cfg.len()
    }

    /// Layer `l`'s output copied into its tap when a gate armed them.
    fn tap(&mut self, l: usize) -> Result<(), GpuError> {
        let last = self.last(l);
        let Some(taps) = self.parts.taps.as_deref_mut() else {
            return Ok(());
        };
        let tap = taps.get_mut(l).ok_or(GpuError::State {
            what: WHAT,
            missing: "the layer's tap",
        })?;
        let out = if last {
            &*self.head.input_mut()
        } else {
            &self.parts.s.x
        };
        tap.copy_from_device_async(out, self.gpu.stream())?;
        Ok(())
    }
}

impl<'s> LayerProgram for StepProgram<'s, '_> {
    type Port = StepLeg<'s, HostRun>;

    /// The one row of the walk, whose embedding `refresh` wrote into `x`.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        no_other_row(unit)
    }

    /// The attention sub-layer, then the dense block whole with its tap, or
    /// the routed block up to its go.
    fn front(&mut self, port: &mut StepLeg<'s, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let (gpu, w, l) = (self.gpu, self.w, at.layer);
        attn::attention(gpu, w, &mut self.parts, l)?;
        match self.parts.cfg[l].kind.ffn {
            FfnKind::Dense => {
                let dest = self.last(l).then(|| self.head.input_mut());
                ffn::dense(gpu, w, &mut self.parts, l, dest)?;
                self.tap(l)
            }
            FfnKind::Moe => ffn::front(gpu, w, &mut self.parts, port.hybrid(), l),
        }
    }

    /// A routed layer's wait and add, its tap, and the host tier told the
    /// layer is enqueued (an eager chain is served there).
    fn back(&mut self, port: &mut StepLeg<'s, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let (gpu, l) = (self.gpu, at.layer);
        if !self.parts.cfg[l].kind.host_leg() {
            return Ok(());
        }
        let dest = self.last(l).then(|| self.head.input_mut());
        ffn::back(gpu, &mut self.parts, port.hybrid(), dest)?;
        self.tap(l)?;
        port.hybrid().row_enqueued(l, at.unit)
    }

    /// The head, over the last layer's output.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        no_other_row(unit)?;
        self.head.enqueue(self.gpu, self.w)
    }
}

/// The walk has one row; another unit is refused by name.
fn no_other_row(unit: usize) -> Result<(), GpuError> {
    if unit == 0 {
        return Ok(());
    }
    Err(GpuError::Shape {
        what: WHAT,
        detail: format!("row {unit} of a walk of one row"),
    })
}
