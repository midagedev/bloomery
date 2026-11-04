//! What V4.1's pair rule decides for an expert tier, as the tier gates read it: the layers a tier may
//! run, from the file's layer table, never from layer numbers.

use bloomery_gpu_gates::GateError;
use model::arch::deepseek41::hparams::Hparams;

/// Whether the pair rule lets an expert tier run layer `l`'s routed stacks (`accepts::ds41_pair` at
/// `tier`, over the gate's type and the down's as the placement's own tier rule reads them): the
/// fused layer's Q4_K triple, never the Q5_K down of V4.1's first layers, which a stage card alone
/// runs.
///
/// # Errors
/// A layer the file holds no routed gate, up and down stacks of.
pub fn tier_admits(hp: &Hparams, l: usize) -> Result<bool, GateError> {
    let [gate, _, down] = hp
        .layers
        .get(l)
        .and_then(|k| k.routed_types)
        .ok_or_else(|| format!("layer {l} holds no routed gate, up and down stacks"))?;
    Ok(model::placement::accepts::ds41_pair(gate, down, true))
}
