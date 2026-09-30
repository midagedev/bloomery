//! The placement planner, re-exported from `bloomery-placement` — a pure
//! crate, so the planner's own tests run on the Mac (`just mac-test`) and
//! every `--plan` question over header facts and device figures is answered
//! without a card. The planner's contract, its types and its rules are
//! documented there ([`bloomery_placement::placement`]); this path is the
//! one the tree's callers and gates name.
//!
//! What the Mac cannot run stays in this crate, beside the re-export:
//! [`host_lock`] (the host tier's page locks, residency walks and page
//! release) and [`workstation`]'s `host_available` (the reading of
//! `/proc/meminfo`).

pub use bloomery_placement::placement::*;
pub mod host_lock;
pub mod workstation;
