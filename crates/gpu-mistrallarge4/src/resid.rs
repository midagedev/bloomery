//! Mistral Large 4's adaptive expert residency (`BLOOMERY_RESIDENCY`). The machine is the common
//! one (`bloomery_gpu::host::swap`): [`Body::open_placed_with`] builds its [`ResidencySpec`] here
//! and starts it as the load's last step, and the model's step and prompt boundaries drive it as
//! they drive every tiered body's. This file holds the family's facts it is built from and the
//! instruments a gate or a server reads it by.
//!
//! - The machine runs over the stage card's routed stacks: each routed layer's card segment is
//!   the id prefix the plan keeps, the first `pinned` of it never a victim and `spares` of it free
//!   slots for flips in flight. A layer the plan gives no card slot takes no part (a Mix file's
//!   layers whose gate and up stacks differ in type hold none), and an expert tier card's experts
//!   stay where the plan put them.
//! - The source is the file's own blocks as the card's stacks hold them: the stacks' types are
//!   read from the file's tensor table layer by layer ([`hosted_stacks`]), so the IQ1_S, IQ1_M,
//!   IQ2_XXS and IQ2_XS stacks of both files move by one path and this crate names no type.
//! - A decode step keeps one row of the router's [`ffn::N_USED`] ids a layer; a prompt call is
//!   one pass that keeps none (`prompt_pass`), so the rule counts decode rows only.
//! - The live delay is the common two passes: a planning pass makes at most the rule's cap of
//!   flips, each one expert's staging and one copy over the card's link in the background copy
//!   stream, which two decode steps of this model's width hold.
//! - The churn pool is hosted ([`ChurnHosting::Hosted`]): the body declares
//!   `UNHOSTED_POOL = NotCarried`, so a word that asks for its pool unhosted is refused by name
//!   and an unset word that does not fit the host climbs to the first pinned count that does.

use std::sync::Arc;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::swap::{PassReport, Residency};
use bloomery_gpu::host::swap_source::{FileSwap, ResidencySpec, hosted_stacks};
use model::arch::mistrallarge4::place::PlanInputs;

use crate::body::{Body, WHAT};
use crate::ffn;

/// The most rows one pass runs: a decode step's one. A drafted pass keeps more and raises it.
pub const MAX_ROWS: usize = 1;

/// What a placed load under `lever` asks the load path for: the common live delay and deadline,
/// the router's pick count, [`MAX_ROWS`], the churn pool hosted, and the file's routed stacks
/// over the `routed_layers` layers past the dense lead.
///
/// # Errors
/// A layer of the file that holds some but not all of its three routed stacks, named.
pub(crate) fn spec(
    lever: Residency,
    inputs: &PlanInputs,
    routed_layers: usize,
) -> Result<ResidencySpec, GpuError> {
    Ok(ResidencySpec::hosted(
        lever,
        ffn::N_USED,
        MAX_ROWS,
        Arc::new(hosted_stacks(&inputs.model, WHAT, (routed_layers, true))?),
    ))
}

impl Body {
    /// Keep every residency boundary's report from now on, for [`Body::take_residency_passes`]: a
    /// binary that prints the `residency pass` records, or a gate that counts flips, asks once,
    /// with the boundaries it takes at most between two takes, `passes`.
    pub fn log_residency(&mut self, passes: usize) {
        self.shell.log_residency(passes);
    }

    /// The residency machine's source, when the load runs one.
    #[must_use]
    pub fn residency_source(&self) -> Option<&FileSwap> {
        self.shell.residency_source()
    }

    /// The residency boundaries' reports since the last take, in order, each with the kind of the
    /// pass it ended.
    pub fn take_residency_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.shell.take_residency_passes()
    }
}
