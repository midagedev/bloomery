//! The placed prompt pass a Hy4-flash placed load opens beside its step: the widest of the body's
//! pass shapes ([`SHAPES`]: the widest walk of several units, then fewer units, then one unit of
//! fewer rows) whose bytes the stage card's room holds as the prompt reserve; a load whose prompt
//! calls stream their host experts (`BLOOMERY_XSTREAM=split`) reserves the stream's ring at its
//! floor beside it: inputs that carry the ring ([`with_ring`]) keep it in every shape's inputs
//! ([`inputs_for`]), and plain inputs take it by [`streaming_for`].
//! `bloomery-serve --model hy4` and `gate_hy4_pp` pick their shape here ([`pick`]'s `attempt` is the
//! caller's plan of a shape). The family hangs no expert tier card, so every shape plans on one
//! machine.

use bloomery_gpu_hy4::{PassShape, SHAPES, pass_bytes};
use model::arch::hy4::place::{PlaceError, PlanInputs, stream_ring};
use model::placement::Machine;

use crate::GateError;
use crate::pass_pick;

/// The shape a plan holds, and the narrowest wider shape refused with the bytes it fell short by
/// (`None` when the widest shape fit).
pub type Picked = pass_pick::Picked<PassShape>;

/// Why no shape is picked.
pub use pass_pick::NoShape;

/// `inputs` for the plan of a pass of `shape` at `ctx` positions: the shape's bytes
/// ([`pass_bytes`]) the prompt reserve, and the stream's ring where `inputs` carry one
/// ([`PlanInputs::with_pass_reserve`]).
pub fn inputs_for(
    inputs: &PlanInputs,
    ctx: usize,
    shape: PassShape,
) -> Result<PlanInputs, GateError> {
    Ok(inputs
        .clone()
        .with_pass_reserve(pass_bytes(&inputs.hp, ctx, shape)?))
}

/// `inputs` for a load whose prompt calls stream their host experts: the stream's ring at its
/// floor ([`stream_ring`], [`PlanInputs::streaming`]), which the pass's inputs keep ([`inputs_for`]).
pub fn with_ring(inputs: PlanInputs) -> Result<PlanInputs, GateError> {
    let ring = stream_ring(&inputs)?;
    Ok(inputs.streaming(ring))
}

/// [`inputs_for`] for a load whose prompt calls stream their host experts: the stream's ring at
/// its floor beside the pass's bytes, `inputs` plain.
pub fn streaming_for(
    inputs: &PlanInputs,
    ctx: usize,
    shape: PassShape,
) -> Result<PlanInputs, GateError> {
    inputs_for(&with_ring(inputs.clone())?, ctx, shape)
}

/// A shape's name in the pick's lines: its rows, and its units when a walk holds several.
#[must_use]
pub fn shape_name(s: PassShape) -> String {
    match s.units {
        1 => format!("{} rows", s.rows),
        u => format!("{u} units of {} rows", s.rows),
    }
}

/// The widest of [`SHAPES`] whose pass the plan holds at `ctx` on `machine` ([`pass_pick::pick`]):
/// each shape's [`inputs_for`], planned by `attempt`.
pub fn pick(
    inputs: &PlanInputs,
    ctx: usize,
    machine: &Machine,
    attempt: &dyn Fn(&PlanInputs, &Machine) -> Result<(), PlaceError>,
) -> Result<Result<Picked, NoShape>, GateError> {
    pass_pick::pick(
        SHAPES,
        ctx,
        &shape_name,
        &|s| Ok((inputs_for(inputs, ctx, s)?, machine.clone())),
        attempt,
    )
}
