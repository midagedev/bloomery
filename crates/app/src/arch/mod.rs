//! Each model's session pieces: its [`crate::Open`], [`crate::Prompt`] and
//! [`crate::Keep`], and the drafts that run beside it. A binary turns on the
//! feature of the model it opens.

use bloomery_gpu::model::Rollback;
use bloomery_gpu::{GpuError, GpuModel};
use model::placement::Machine;

#[cfg(feature = "deepseek41")]
pub mod deepseek41;
#[cfg(feature = "glm5next")]
pub mod glm5next;
#[cfg(feature = "mimo2")]
pub mod mimo2;
/// The qwen3moe family's Qwen3.8 session pieces: the body lives in the gpu
/// crate's own qwen3moe arch, so they need no feature of their own.
pub mod qwen3moe;
#[cfg(feature = "solar2")]
pub mod solar2;

/// `ctx` positions as a plan takes them, for a body whose chain runs on one
/// card: refused by name (`what` the body's session) when `machine` puts the
/// layers on any other number of cards or `ctx` passes u64.
pub fn one_card_ctx(what: &'static str, machine: &Machine, ctx: usize) -> Result<u64, GpuError> {
    if machine.cards.len() != 1 {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "the placement puts the layers on {} cards; the chain runs on one",
                machine.cards.len()
            ),
        });
    }
    u64::try_from(ctx).map_err(|_| GpuError::Shape {
        what,
        detail: format!("a context of {ctx} positions passes u64"),
    })
}

/// [`crate::Keep::cut`] of a body whose recurrent state has its history only
/// in checkpoints: nothing to take back at the model's position; back to
/// empty at 0 (a reset); else the checkpoint at `n` ([`Rollback`]), any other
/// position refused by name.
pub fn cut_checkpointed<B: Rollback>(m: &mut GpuModel<B>, n: u32) -> Result<(), GpuError> {
    match n {
        n if n == m.pos() => Ok(()),
        0 => m.reset(),
        n => m.rollback(n),
    }
}
