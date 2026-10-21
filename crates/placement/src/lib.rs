//! The placement planner as a pure crate: where every tensor of a model
//! lives on a machine — the card byte budgets, the expert rule, the
//! per-device totals and the invariant check — from header facts and device
//! figures alone. `bloomery-model` re-exports the planner at
//! `model::placement`, so a caller names one crate for the model and its
//! placement; what the planner needs of the model's file side reaches it as
//! plain values ([`crate::placement::ModelTensors`], the device figures,
//! the KV bytes). [`kernels`] is the pure table of which common card kernel
//! entry runs each operation on a file tensor of a given type.
//!
//! Card reserve and set-aside sums are checked at their owners and return
//! `None` past u64 bytes, as the complete floor does. Planning refuses those
//! sums with a named [`placement::PlacementError::CardBytesPastU64`]; a
//! whole-load floor cannot turn overflow into a fitting `u64::MAX` sentinel.
//! Shared-pool card totals are checked before host accounting; separate
//! device memories do not accumulate an irrelevant combined card total. The
//! host's reserve sum is checked the same way ([`placement::Host::reserve_bytes`],
//! refused as [`placement::PlacementError::HostBytesPastU64`]), and the host
//! need's terms sum exactly, so a need past u64 bytes is a host over rather
//! than a wrapped small figure.
//!
//! What does not live here, because the Mac cannot run it: the host tier's
//! page locks and residency walks (`bloomery-model`'s
//! `placement::host_lock`) and the r8 sidecar's writer and checker
//! (`r8file`, which re-exports this crate's [`r8`]: the refusal
//! type the planner's error wraps). The host's own memory is read by
//! `bloomery-os`; the arithmetic of the room that reading gives is here
//! ([`placement::workstation::host_room`]).

pub mod accepts;
pub mod kernels;
pub mod placement;
pub mod r8;
pub mod slots;
