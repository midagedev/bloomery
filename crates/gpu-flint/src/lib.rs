//! Flint's composition of the common head128 attention, shared routed block and host tier: one
//! program Flint's files and K-EXAONE's `exaone-moe` ones run, the file's router shape picking
//! the compiled instance. Device-linked consumers build this crate with `cargo oxide`.

use models::shape::RouterInst;

pub mod body;
pub mod prefill;
mod step;
pub mod tier;

pub use body::{
    Body, ExaoneBody, ExaoneModel, ExaoneRouter, FlintBody, FlintModel, FlintRouter, Seq,
    StepInput, set_taps,
};
pub use prefill::{prefill, set_prefill_group};

/// Whether the router row `row` a description selects is K-EXAONE's instance
/// (`router::mimo2`): what picks between the program's two bodies where a
/// caller opens by the file alone.
#[must_use]
pub fn is_exaone_router(row: RouterInst) -> bool {
    row == bloomery_gpu_deepseek41::router::mimo2::ROW
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod two_card_tests;
