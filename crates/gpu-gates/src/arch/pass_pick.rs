//! The widest prompt-pass shape a plan holds: the search every family with a placed prompt pass
//! runs the same way. A family hands the shapes of its pass, widest first, and what planning one
//! takes (its inputs with the shape's bytes as the prompt reserve, the machine at the shape's
//! rows); the pick is the first the stage card's room holds, with the narrowest wider shape
//! refused beside it and the bytes it fell short by.

use std::fmt;

use model::arch::place_hybrid::{PlaceError, PlanInputs};
use model::placement::{Machine, PlacementError};

use crate::GateError;

/// The shape a plan holds, and the narrowest wider shape refused with the bytes it fell short by
/// (`None` when the widest shape tried fit).
#[derive(Clone, Copy, Debug)]
pub struct Picked<S> {
    pub shape: S,
    pub wider: Option<(S, u64)>,
}

/// Why no shape is picked.
#[derive(Clone, Debug)]
pub enum NoShape {
    /// Every shape tried was refused for its bytes: the narrowest (by its label), and the bytes it
    /// fell short by.
    Short { narrowest: String, short: u64 },
    /// No shape is narrow enough for a context of this many positions.
    Ctx(usize),
}

impl fmt::Display for NoShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoShape::Short { narrowest, short } => write!(
                f,
                "no pass shape fits the stage card; the narrowest, {narrowest}, is {short} B short"
            ),
            NoShape::Ctx(ctx) => write!(f, "no pass shape of --ctx {ctx}"),
        }
    }
}

/// What a caller plans a shape's inputs on a machine by: `Ok` when the plan holds the shape.
pub type Attempt<'a, H> = dyn Fn(&PlanInputs<H>, &Machine) -> Result<(), PlaceError> + 'a;

/// The first of `shapes` (widest first) whose pass the plan holds at `ctx`: each shape's inputs and
/// machine from `plan_at`, planned by `attempt`. A shape refused for its reserve
/// ([`PlaceError::PromptOver`]) or for a dense trunk that, with the shape's bytes beside it, passes
/// the card's free bytes ([`PlacementError::CardFreeFloor`]) steps down, and the narrowest refused
/// comes back beside the pick. Every shape refused so is [`NoShape`]; any other refusal ends the
/// pick by name, as `label` names the shape.
pub fn pick<S: Copy, H>(
    shapes: impl IntoIterator<Item = S>,
    ctx: usize,
    label: &dyn Fn(S) -> String,
    plan_at: &dyn Fn(S) -> Result<(PlanInputs<H>, Machine), GateError>,
    attempt: &Attempt<'_, H>,
) -> Result<Result<Picked<S>, NoShape>, GateError> {
    let mut wider: Option<(S, u64)> = None;
    for shape in shapes {
        let (pass, machine) = plan_at(shape)?;
        match attempt(&pass, &machine) {
            Ok(()) => return Ok(Ok(Picked { shape, wider })),
            Err(PlaceError::PromptOver { short, .. }) => wider = Some((shape, short)),
            // The shape's own bytes beside the dense trunk pass the card's free bytes: a narrower
            // shape holds fewer.
            Err(PlaceError::Placement(PlacementError::CardFreeFloor(f))) => {
                wider = Some((shape, f.over));
            }
            Err(e) => return Err(format!("the pass of {}: {e}", label(shape)).into()),
        }
    }
    Ok(Err(match wider {
        Some((narrowest, short)) => NoShape::Short {
            narrowest: label(narrowest),
            short,
        },
        None => NoShape::Ctx(ctx),
    }))
}
