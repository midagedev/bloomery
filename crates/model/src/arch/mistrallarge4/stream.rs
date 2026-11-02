//! The card bytes a Mistral Large 4 load keeps for its prompt calls' expert stream
//! (`BLOOMERY_XSTREAM=split`): the common rule ([`place_hybrid::stream_ring_bytes`]) over the
//! file's routed stacks at [`STREAM_HALF_SLOTS`] slots a half, beside the ring pass's route GEMM
//! buffers at [`PROMPT_ROWS`] rows in the forms the stacks read. The body sizes its ring from the
//! card bytes the plan leaves free, so a plan that streams reserves them
//! ([`PlanInputs::with_stream`]) and a plan that does not leaves the ring what the card's margin
//! holds.

use super::place::{PROMPT_ROWS, PlanInputs};
use crate::arch::names::routed_stacks;
use crate::arch::place_hybrid::{
    ring_forms_planned, ring_pass_bytes_in, stream_ring_bytes, stream_slot_bytes,
};
use crate::placement::PlacementError;

/// The slots of each half of the expert stream's ring a streaming plan keeps room for
/// [derived: a host expert at the default group's columns an expert takes the host one to two
/// times the lane's copy of one expert (23.5 MB at the pinned rate), so the stream rule's balance
/// streams 0.56 to 0.69 of a layer's 112 host experts, 63 to 77, at one unit and 0.72 to 0.82,
/// 81 to 92, at the default group of two units; a half of 96 holds the larger without capping the
/// rule, and the rule never streams past its balance].
pub const STREAM_HALF_SLOTS: u64 = 96;

/// The card bytes the prompt calls' expert stream takes on a Mistral Large 4 load: the ring's two
/// halves of [`STREAM_HALF_SLOTS`] slots, a slot the widest layer's three stacks per expert
/// ([`stream_slot_bytes`]), and the ring pass's route GEMM buffers over a half
/// ([`ring_pass_bytes_in`], in the forms [`ring_forms_planned`] reads off the routed stacks).
/// Refused by name as the rule refuses.
pub fn stream_bytes(inputs: &PlanInputs) -> Result<u64, PlacementError> {
    let hp = &inputs.hp;
    let half = STREAM_HALF_SLOTS;
    let pass = ring_pass_bytes_in(
        PROMPT_ROWS,
        hp.n_used,
        hp.n_embd,
        hp.expert_ff,
        half as usize,
        ring_forms_planned(&inputs.model, routed_stacks),
    ) as u64;
    stream_ring_bytes(
        &inputs.model,
        routed_stacks,
        (hp.n_expert as u64, half),
        pass,
    )
}

impl PlanInputs {
    /// `self` for a load whose prompt calls stream their host experts (`BLOOMERY_XSTREAM=split`):
    /// the prompt reserve grows by [`stream_bytes`] ([`PlanInputs::streaming`]) so the ring finds
    /// its room on a card the plan otherwise fills with experts, and a slot is the widest routed
    /// layer's expert ([`stream_slot_bytes`]). Refused by name as [`stream_bytes`] refuses.
    pub fn with_stream(self) -> Result<PlanInputs, PlacementError> {
        let ring = stream_bytes(&self)?;
        let slot = stream_slot_bytes(&self.model, routed_stacks, self.hp.n_expert as u64)?;
        Ok(self.streaming(ring, slot))
    }
}
