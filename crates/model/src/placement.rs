//! The placement planner, re-exported from `bloomery-placement` — a pure
//! crate, so the planner's own tests run on the Mac (`just mac-test`) and
//! every `--plan` question over header facts and device figures is answered
//! without a card. The planner's contract, its types and its rules are
//! documented there ([`bloomery_placement::placement`]); this path is the
//! one the tree's callers and gates name. The resident-slot rules — a
//! sequence's terms and the context split ([`slots`]) — are re-exported
//! beside it, for the seats that split a context among their slots, and so
//! is the card kernel table ([`kernels`]: which common launch runs each
//! operation on a file tensor of a type) for the bodies that route their
//! types through it.
//!
//! What the Mac cannot run stays in this crate, beside the re-export:
//! [`host_lock`] (the host tier's page locks, residency walks and page
//! release) and [`workstation`]'s `host_available` (the reading of
//! `/proc/meminfo`). So does the tail every family's plan ends in — the
//! built plan against its invariants, and how a refusal of them reads
//! ([`checked`], [`joined`]) — one owner beside the planner it checks.

pub use bloomery_placement::placement::*;
pub use bloomery_placement::{kernels, slots};
pub mod host_lock;
pub mod workstation;

/// A built plan against its invariants ([`Plan::violations`]): the plan
/// when none is broken, else every violation it breaks, for the caller's
/// own refusal of them (each family's `Broken`).
pub fn checked<'a>(plan: Plan<'a>) -> Result<Plan<'a>, Vec<Violation>> {
    let broken = plan.violations();
    if broken.is_empty() {
        Ok(plan)
    } else {
        Err(broken)
    }
}

/// The violations, `; `-separated, as a family's `Broken` refusal renders
/// them.
pub fn joined(broken: &[Violation]) -> String {
    let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
    list.join("; ")
}
