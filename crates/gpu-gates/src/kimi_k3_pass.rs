//! The placed prompt pass a Kimi K3 load opens beside its step: the widest of the body's pass
//! shapes ([`SHAPES`], rows by the latent mixers' inner rows) whose bytes the stage card's room
//! holds as the prompt reserve, with the expert tier's batch at the shape's rows and the NVMe
//! arena at its floor ([`PlanInputs::with_arena_at_floor`]). `generate_kimi_k3 --walk pass` and
//! the Kimi K3 seat of `bloomery-serve` pick their shape here; each plans a shape its own way
//! ([`pick`]'s `attempt`).

use std::fmt;

use bloomery_gpu::arch::kimi_k3::{CHUNK, PassShape, SHAPES, pass_bytes};
use model::arch::kimi_k3::place::{PlaceError, PlanInputs};
use model::placement::{Machine, PlacementError};

use crate::GateError;

/// The shape a plan holds, and the narrowest wider shape refused with the bytes it fell short by
/// (`None` when the widest shape of the context fit).
#[derive(Clone, Copy, Debug)]
pub struct Picked {
    pub shape: PassShape,
    pub wider: Option<(PassShape, u64)>,
}

/// Why no shape is picked.
#[derive(Clone, Copy, Debug)]
pub enum NoShape {
    /// Every shape of the context was refused for its bytes: the narrowest, and the bytes it fell
    /// short by.
    Short { narrowest: PassShape, short: u64 },
    /// No shape is narrow enough for a context of this many positions.
    Ctx(usize),
}

impl fmt::Display for NoShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoShape::Short { narrowest, short } => write!(
                f,
                "no pass shape fits the stage card; the narrowest, {} rows ({} latent), is \
                 {short} B short",
                narrowest.rows, narrowest.mla_rows
            ),
            NoShape::Ctx(ctx) => write!(f, "no pass shape of --ctx {ctx}"),
        }
    }
}

/// `inputs` for the plan of a pass of `shape` at `ctx` positions: the shape's bytes
/// ([`pass_bytes`], the table the load allocates from) the prompt reserve, the NVMe arena at its
/// floor.
pub fn inputs_for(
    inputs: &PlanInputs,
    ctx: usize,
    shape: PassShape,
) -> Result<PlanInputs, GateError> {
    Ok(inputs
        .clone()
        .with_prompt_reserve(pass_bytes(&inputs.hp, ctx, shape)?)
        .with_arena_at_floor())
}

/// The widest of [`SHAPES`] whose pass the plan holds at `ctx`: each shape's [`inputs_for`],
/// planned by `attempt` on the machine `build` makes at the shape's rows (the expert tier's batch
/// at those rows). A shape wider than `ctx` is passed over unless it is [`CHUNK`]'s. A shape
/// refused for its reserve ([`PlaceError::PromptOver`]) or for a dense trunk that, with the
/// shape's bytes beside it, passes the card's free bytes ([`PlacementError::CardFreeFloor`])
/// steps down, and the narrowest refused comes back beside the pick. Every shape refused so is
/// [`NoShape`]; any other refusal ends the pick by name.
pub fn pick(
    inputs: &PlanInputs,
    ctx: usize,
    build: &dyn Fn(u64) -> Result<Machine, GateError>,
    attempt: &dyn Fn(&PlanInputs, &Machine) -> Result<(), PlaceError>,
) -> Result<Result<Picked, NoShape>, GateError> {
    let mut wider = None;
    for &shape in SHAPES.iter().filter(|s| s.rows <= ctx.max(CHUNK)) {
        let pass = inputs_for(inputs, ctx, shape)?;
        let machine = build(shape.rows as u64)?;
        match attempt(&pass, &machine) {
            Ok(()) => return Ok(Ok(Picked { shape, wider })),
            Err(PlaceError::PromptOver { short, .. }) => wider = Some((shape, short)),
            // The shape's own bytes beside the dense trunk pass the card's free bytes: a narrower
            // shape holds fewer.
            Err(PlaceError::Placement(PlacementError::CardFreeFloor(f))) => {
                wider = Some((shape, f.over));
            }
            Err(e) => {
                return Err(format!(
                    "the pass of {} rows ({} latent): {e}",
                    shape.rows, shape.mla_rows
                )
                .into());
            }
        }
    }
    Ok(Err(match wider {
        Some((narrowest, short)) => NoShape::Short { narrowest, short },
        None => NoShape::Ctx(ctx),
    }))
}
