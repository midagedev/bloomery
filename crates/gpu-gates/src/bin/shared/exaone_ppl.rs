//! K-EXAONE-2.0's body as the perplexity walk's model (`bloomery_gpu_gates::kld::PplModel`): the load of
//! [`crate::seat::open_model`] (the gate placement of the arm on the card in view, no tap armed) run as the serve
//! seat runs its step, the captured graph, and the logits row read back when the walk asks for it. `ppl_exaone`
//! (the loss tool) and `gate_exaone_kld` (its gate) include it by `#[path]` beside `shared/exaone_seat.rs` as
//! `seat`, `shared/lcpp_e2e.rs` and `shared/gate_card.rs`.

use crate::lcpp_e2e::Arm;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_flint::ExaoneModel;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::kld::PplModel;
use std::path::Path;

/// The body behind the walk's calls.
pub struct Body {
    m: ExaoneModel,
    vocab: usize,
}

/// Open `file` at `ctx` positions by the placement of `arm`, the step captured as a graph.
pub fn open(file: &Path, ctx: usize, arm: Arm) -> Result<Body, GateError> {
    let (mut m, dims) = crate::seat::open_model(file, ctx, arm)?;
    m.set_mode(StepMode::Graph);
    Ok(Body {
        m,
        vocab: dims.vocab,
    })
}

impl Body {
    /// Entries of a logits row: the head's width.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The layers the body runs.
    pub fn layers(&self) -> usize {
        self.m.layers().len()
    }

    /// Bytes the load holds on the card.
    pub fn resident_bytes(&self) -> usize {
        self.m.resident_bytes()
    }
}

impl PplModel for Body {
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.m.reset()?)
    }

    fn step(&mut self, token: u32) -> Result<(), GateError> {
        self.m.step(&[token])?;
        Ok(())
    }

    fn logits(&mut self) -> Result<Vec<f32>, GateError> {
        let mut row = vec![0.0; self.vocab];
        self.m.logits_into(&mut row)?;
        Ok(row)
    }
}
