//! The widest prompt-pass shape a plan holds: the search every family with a placed prompt pass
//! runs the same way. A family hands the shapes of its pass, widest first, and what planning one
//! takes (its inputs with the shape's bytes as the prompt reserve, the machine at the shape's
//! rows); the pick is the first the stage card's room holds, with the narrowest wider shape
//! refused beside it and the bytes it fell short by. A shape the card's room does not hold steps
//! to the next, whichever floor names it; every other refusal ends the pick.

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
/// machine from `plan_at`, planned by `attempt`. A shape the card's room does not hold steps down:
/// refused for its reserve ([`PlaceError::PromptOver`]), for a dense trunk that, with the shape's
/// bytes beside it, passes the card's free bytes ([`PlacementError::CardFreeFloor`]), or for a floor
/// that, with them, passes the card's budget ([`PlacementError::CardBudgetFloor`]). The narrowest
/// refused comes back beside the pick. Every shape refused so is [`NoShape`]; any other refusal ends
/// the pick by name, as `label` names the shape.
pub fn pick<S: Copy, H>(
    shapes: impl IntoIterator<Item = S>,
    ctx: usize,
    label: &dyn Fn(S) -> String,
    plan_at: &dyn Fn(S) -> Result<(PlanInputs<H>, Machine), GateError>,
    attempt: &Attempt<'_, H>,
) -> Result<Result<Picked<S>, NoShape>, GateError> {
    step_down(shapes, ctx, label, &|shape| {
        let (pass, machine) = plan_at(shape)?;
        Ok(attempt(&pass, &machine))
    })
}

/// [`pick`] over `tried`, the plan of one shape: `Ok` when it holds the shape, else its refusal.
fn step_down<S: Copy>(
    shapes: impl IntoIterator<Item = S>,
    ctx: usize,
    label: &dyn Fn(S) -> String,
    tried: &dyn Fn(S) -> Result<Result<(), PlaceError>, GateError>,
) -> Result<Result<Picked<S>, NoShape>, GateError> {
    let mut wider: Option<(S, u64)> = None;
    for shape in shapes {
        match tried(shape)? {
            Ok(()) => return Ok(Ok(Picked { shape, wider })),
            Err(PlaceError::PromptOver { short, .. }) => wider = Some((shape, short)),
            // The shape's own bytes beside the dense trunk pass the card's free bytes or its
            // budget: a narrower shape holds fewer.
            Err(PlaceError::Placement(PlacementError::CardFreeFloor(f))) => {
                wider = Some((shape, f.over));
            }
            Err(PlaceError::Placement(PlacementError::CardBudgetFloor {
                budget, floor, ..
            })) => {
                wider = Some((shape, floor.saturating_sub(budget)));
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use model::placement::CardFreeFloor;

    use super::*;

    /// The shapes the scripted plans below walk, named by their position.
    const SHAPES: [u8; 5] = [0, 1, 2, 3, 4];

    fn label(s: u8) -> String {
        format!("shape {s}")
    }

    fn prompt_over(short: u64) -> PlaceError {
        PlaceError::PromptOver {
            card: "card".to_owned(),
            reserve: short,
            left: 0,
            short,
        }
    }

    fn free_floor(over: u64) -> PlaceError {
        PlaceError::Placement(PlacementError::CardFreeFloor(Box::new(CardFreeFloor {
            card: "card".to_owned(),
            unified: false,
            free: 0,
            usable: 0,
            floor: over,
            over,
            dense: 0,
            kv: 0,
            context: 0,
            scratch: 0,
            reserves: 0,
            margin: 0,
            beside: 0,
            held: String::new(),
        })))
    }

    fn budget_floor(budget: u64, floor: u64) -> PlaceError {
        PlaceError::Placement(PlacementError::CardBudgetFloor {
            card: "card".to_owned(),
            budget,
            floor,
            dense: 0,
            kv: 0,
            context: 0,
            scratch: 0,
            reserves: 0,
            margin: 0,
            beside: 0,
        })
    }

    /// What [`step_down`] answers of the scripted shapes.
    type Pick = Result<Result<Picked<u8>, NoShape>, GateError>;

    /// Walks `shapes` with each shape's scripted answer (`None` is a plan that holds it) and
    /// returns the pick and the shapes planned, in order.
    fn walk(shapes: &[u8], mut answers: Vec<Option<PlaceError>>) -> (Pick, Vec<u8>) {
        answers.reverse();
        let (answers, planned) = (RefCell::new(answers), RefCell::new(Vec::new()));
        let picked = step_down(shapes.iter().copied(), 4096, &label, &|shape| {
            planned.borrow_mut().push(shape);
            Ok(answers
                .borrow_mut()
                .pop()
                .expect("a scripted answer")
                .map_or(Ok(()), Err))
        });
        (picked, planned.into_inner())
    }

    /// Each refusal a narrower shape cures steps to the next shape, in the order given, and the
    /// pick is the first the plan holds, with the narrowest refused beside it and its short: the
    /// reserve over, the free bytes over, and the budget under the floor (the floor less the
    /// budget). Mutant: the budget floor ending the pick like any other refusal.
    #[test]
    fn a_floor_over_the_room_steps_to_the_next_shape() {
        let (picked, planned) = walk(
            &SHAPES,
            vec![
                Some(prompt_over(400)),
                Some(free_floor(300)),
                Some(budget_floor(3_516_346_368, 3_516_346_368 + 150)),
                None,
            ],
        );
        let picked = picked.expect("the pick").expect("a shape fits");
        assert_eq!(picked.shape, 3);
        assert_eq!(picked.wider, Some((2, 150)));
        assert_eq!(planned, [0, 1, 2, 3]);
    }

    /// The first shape the plan holds is the pick, with no wider shape refused and no narrower one
    /// planned.
    #[test]
    fn the_first_shape_that_fits_ends_the_walk() {
        let (picked, planned) = walk(&SHAPES, vec![None]);
        let picked = picked.expect("the pick").expect("a shape fits");
        assert_eq!((picked.shape, picked.wider), (0, None));
        assert_eq!(planned, [0]);
    }

    /// Every shape refused for the card's room is no shape, naming the narrowest and its short.
    /// Mutant: the budget floor ending the pick, which returns the first shape's refusal instead.
    #[test]
    fn every_shape_over_the_room_is_no_shape() {
        let (picked, planned) = walk(
            &SHAPES[..3],
            vec![
                Some(budget_floor(100, 900)),
                Some(budget_floor(100, 500)),
                Some(budget_floor(100, 350)),
            ],
        );
        let Ok(Err(NoShape::Short { narrowest, short })) = picked else {
            panic!("expected no shape, got {picked:?}");
        };
        assert_eq!((narrowest.as_str(), short), ("shape 2", 250));
        assert_eq!(planned, [0, 1, 2]);
    }

    /// A refusal no narrower shape cures ends the pick at the shape that met it, named, with no
    /// narrower shape planned. Mutant: every refusal stepping down.
    #[test]
    fn another_refusal_ends_the_pick_by_name() {
        let (picked, planned) = walk(
            &SHAPES,
            vec![Some(budget_floor(100, 200)), Some(PlaceError::CtxZero)],
        );
        let Err(e) = picked else {
            panic!("expected a refusal, got {picked:?}");
        };
        assert_eq!(
            e.to_string(),
            "the pass of shape 1: ctx_max 0: a plan serves at least one position"
        );
        assert_eq!(planned, [0, 1]);
    }

    /// No shape to try is a context no shape is narrow enough for.
    #[test]
    fn no_shapes_is_a_context_no_shape_fits() {
        let (picked, planned) = walk(&[], vec![]);
        assert!(matches!(picked, Ok(Err(NoShape::Ctx(4096)))), "{picked:?}");
        assert!(planned.is_empty());
    }
}
