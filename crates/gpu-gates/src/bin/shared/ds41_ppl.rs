//! The V4.1 body as the perplexity walk's model (`bloomery_gpu_gates::kld::PplModel`): one id a step from an
//! empty model, and the logits row that step leaves. `ppl_ds41` (the loss tool) walks a session's model with it,
//! and `gate_deepseek41_step`'s `--kld-walk` clause walks the gate's own load, so one decode path answers both
//! the tool and its gate.

use bloomery_gpu_deepseek41::body::Deepseek41Model;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::kld::PplModel;

/// A loaded V4.1 model behind the walk's three calls. The reset is the session's clear
/// (`app::Session::clear` of one slot): the engine's reset, then the residency machine back to its seed, so a
/// chunk runs from the state the load left whatever the placement moved before it.
pub struct Steps<'a>(pub &'a mut Deepseek41Model);

impl PplModel for Steps<'_> {
    fn reset(&mut self) -> Result<(), GateError> {
        self.0.reset()?;
        self.0.residency_reset()?;
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<(), GateError> {
        self.0.step(&[token])?;
        Ok(())
    }

    fn logits(&mut self) -> Result<Vec<f32>, GateError> {
        Ok(self.0.logits()?)
    }
}
