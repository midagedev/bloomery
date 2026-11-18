//! The placed prompt pass a Kimi K3 load opens beside its step: the widest of the body's pass
//! shapes ([`SHAPES`], rows by the latent mixers' inner rows) whose bytes the stage card's room
//! holds as the prompt reserve, with the expert tier's batch at the shape's rows and the NVMe
//! arena the dial gives, at least its floor ([`PlanInputs::with_pass_arena`]). `generate_kimi_k3 --walk pass` and
//! the Kimi K3 seat of `bloomery-serve` pick their shape here; each plans a shape its own way
//! ([`pick`]'s `attempt`).

use bloomery_gpu::arch::kimi_k3::{CHUNK, PassShape, SHAPES, pass_bytes};
use model::arch::kimi_k3::place::{PlaceError, PlanInputs};
use model::placement::Machine;

use crate::GateError;
use crate::pass_pick;

/// The shape a plan holds, and the narrowest wider shape refused with the bytes it fell short by
/// (`None` when the widest shape of the context fit).
pub type Picked = pass_pick::Picked<PassShape>;

/// Why no shape is picked.
pub use pass_pick::NoShape;

/// `inputs` for the plan of a pass of `shape` at `ctx` positions: the shape's bytes
/// ([`pass_bytes`], the table the load allocates from) the prompt reserve, the NVMe arena the
/// dial gives, at least its floor.
pub fn inputs_for(
    inputs: &PlanInputs,
    ctx: usize,
    shape: PassShape,
) -> Result<PlanInputs, GateError> {
    Ok(inputs
        .clone()
        .with_prompt_reserve(pass_bytes(&inputs.hp, ctx, shape)?)
        .with_pass_arena())
}

/// The widest of [`SHAPES`] whose pass the plan holds at `ctx` ([`pass_pick::pick`]): each shape's
/// [`inputs_for`], planned by `attempt` on the machine `build` makes at the shape's rows (the
/// expert tier's batch at those rows). A shape wider than `ctx` is passed over unless it is
/// [`CHUNK`]'s.
pub fn pick(
    inputs: &PlanInputs,
    ctx: usize,
    build: &dyn Fn(u64) -> Result<Machine, GateError>,
    attempt: &dyn Fn(&PlanInputs, &Machine) -> Result<(), PlaceError>,
) -> Result<Result<Picked, NoShape>, GateError> {
    pass_pick::pick(
        SHAPES.iter().copied().filter(|s| s.rows <= ctx.max(CHUNK)),
        ctx,
        &|s| format!("{} rows ({} latent)", s.rows, s.mla_rows),
        &|s| Ok((inputs_for(inputs, ctx, s)?, build(s.rows as u64)?)),
        attempt,
    )
}
