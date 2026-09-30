//! The placement planner as a pure crate: where every tensor of a model
//! lives on a machine — the card byte budgets, the expert rule, the
//! per-device totals and the invariant check — from header facts and device
//! figures alone. `bloomery-model` re-exports the planner at
//! `model::placement`, so a caller names one crate for the model and its
//! placement; what the planner needs of the model's file side reaches it as
//! plain values ([`crate::placement::ModelTensors`], the device figures,
//! the KV bytes).
//!
//! What does not live here, because the Mac cannot run it: the host tier's
//! page locks and residency walks (`bloomery-model`'s
//! `placement::host_lock`), the reading of `/proc/meminfo`
//! (`placement::workstation::host_available`) and the r8 sidecar's writer
//! and checker (`r8file`, which re-exports this crate's [`r8`]: the refusal
//! type the planner's error wraps).

pub mod placement;
pub mod r8;
