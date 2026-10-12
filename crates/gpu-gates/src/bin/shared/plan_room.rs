//! The plan inputs of a qwen35moe-type file at an unbounded host room: the one
//! owner of the card probe's inputs for every seat that plans such a file
//! (`serve_seats/qwen38.rs`, `qwen3moe_place.rs`), included by `#[path]` where
//! it is used.

use model::arch::qwen35moe::place::PlanInputs;
use model::placement::workstation::HostRead;

/// `inputs` with the host room unbounded (`HostRead::Given` of every byte):
/// the plan it makes pages no expert to the NVMe tier and keeps the PLE table
/// on the host, so what it holds on the card is the card's alone. A machine
/// whose card shares the host's memory (a unified pool) has no such plan; the
/// pool binds every probe alike.
pub fn unbounded(inputs: &PlanInputs) -> PlanInputs {
    PlanInputs {
        hp: inputs.hp.clone(),
        model: inputs.model.clone(),
        spec: inputs.spec.clone(),
        kv: inputs.kv.clone(),
        room: (u64::MAX, HostRead::Given),
        arena: inputs.arena,
    }
}
