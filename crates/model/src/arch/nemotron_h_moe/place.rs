//! A `nemotron_h_moe` file planned onto a machine: the family's half of the
//! one path from the file's headers to a whole-card plan, as the Nano's
//! (`nemotron_h::place`). The hyperparameters, every tensor's role, the typed
//! description and each layer's store bytes are read once ([`describe`];
//! [`read`] refuses a file whose features no program runs, every one listed);
//! the plan is the hybrid trunk's ([`PlanInputs::plan`]): the whole model on
//! one card, each routed stack in the format the body's `_sel` launches read.
//!
//! A sequence's stores are an attention layer's K and V planes and a Mamba-2
//! layer's scan state and conv ring; a latent MoE layer holds nothing
//! ([`Holds`]).

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::spec;
pub use crate::arch::hybrid::{Holds, KvLayout, PlaceError, PlanInputs};
use crate::placement::PlacementError;

/// `split`'s hyperparameters, then its tensors' roles, then its description,
/// the first that fails being the error; then a file with a feature the
/// engine does not run ([`PlacementError::Unimplemented`], every one listed).
pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
    describe(split)?.runnable()
}

/// [`read`] without the refusals: the file as it is.
pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
    let hp = Hparams::read(split)?;
    let read = spec::read(split)?;
    Ok(PlanInputs {
        model: read.tensors,
        spec: read.spec,
        kv: kv_layout(&hp),
    })
}

/// The store each layer of `hp` holds: a Mamba-2 layer its scan, an attention
/// layer its key heads' planes, a MoE layer nothing.
#[must_use]
pub fn kv_layout(hp: &Hparams) -> KvLayout {
    KvLayout::of(
        &hp.ssm,
        hp.key_len,
        hp.kinds
            .iter()
            .zip(&hp.kv)
            .map(|(kind, &kv_heads)| match kind {
                Kind::Mamba => Holds::Scan,
                Kind::Attention => Holds::Attention { kv_heads },
                Kind::Moe => Holds::Nothing,
            }),
    )
}
