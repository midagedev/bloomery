//! The hybrid chain's walk: the order of one layer's ops by its kind, with the unit's front ahead of
//! layer 0 and its end after the last layer, written once over [`HybridOps`] for every backend and
//! every family of the trunk.
//!
//! A chain with no host leg walks one unit of `m` rows through [`runtime::sched::walk`]
//! ([`Walk::run`]): each layer is the walk's front, the head its end, and a port that exchanges
//! nothing. A caller that lays its own front and end around the layers (a walk with a host leg
//! splits a layer around it) runs [`layer`] per layer, with [`at`] naming the step.

use super::ops::{HybridOps, Step};
use super::plan::LayerPlan;
use crate::ChainError;
use crate::noleg::NoLeg;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

const WHAT: &str = "hybrid::Walk";

/// Layer `layer` of the chain `plans` at `m` rows: what an op is told of it and the plan it runs, or
/// a named refusal for a layer the chain does not hold.
pub fn at<F>(
    plans: &[LayerPlan<F>],
    layer: usize,
    m: usize,
) -> Result<(Step, &LayerPlan<F>), ChainError> {
    let plan = plans.get(layer).ok_or_else(|| {
        ChainError::shape(
            "hybrid::at",
            format!("layer {layer} of a chain of {} layers", plans.len()),
        )
    })?;
    let last = layer + 1 == plans.len();
    Ok((Step { layer, m, last }, plan))
}

/// One walk of a chain: every layer of `plans` at `m` rows over a backend's ops, then the unit's
/// head.
pub struct Walk<'a, O: HybridOps> {
    ops: &'a mut O,
    plans: &'a [LayerPlan<O::Ffn>],
    m: usize,
}

impl<'a, O: HybridOps> Walk<'a, O> {
    /// The walk of `plans` at `m` rows over `ops`.
    #[must_use]
    pub fn new(ops: &'a mut O, plans: &'a [LayerPlan<O::Ffn>], m: usize) -> Walk<'a, O> {
        Walk { ops, plans, m }
    }

    /// Enqueue the walk `(1, m, Step)` over an arena of `rows` rows: the unit's front, every
    /// layer, the head. A chain of no layer and `m` outside `1..=rows` are refused by name before
    /// any op.
    pub fn run(self, rows: usize) -> Result<(), O::Error> {
        if self.plans.is_empty() {
            return Err(ChainError::shape(WHAT, "a chain of no layer").into());
        }
        let o = Overlap {
            units: 1,
            cols: self.m,
            port: PortKind::Step,
        };
        let layers = self.plans.len();
        sched::walk(o, layers, &mut NoLeg::new(rows), &mut Steps { walk: self })
    }
}

/// A [`Walk`] as the runtime's layer program.
struct Steps<'a, O: HybridOps> {
    walk: Walk<'a, O>,
}

impl<O: HybridOps> LayerProgram for Steps<'_, O> {
    type Port = NoLeg<O::Error>;

    /// The whole layer at `m` rows, the unit's front ahead of layer 0: a chain with no host leg has
    /// no shadow and no back.
    fn front(&mut self, _: &mut Self::Port, here: At) -> Result<(), O::Error> {
        let Walk { ops, plans, m } = &mut self.walk;
        let (step, plan) = at(plans, here.layer, *m)?;
        if step.layer == 0 {
            ops.embed(*m)?;
        }
        layer(*ops, step, plan)
    }

    fn end(&mut self, _: usize) -> Result<(), O::Error> {
        self.walk.ops.head(self.walk.m)
    }
}

/// Layer `step` on its own from the residual rows its caller laid in, no front: the three ops of
/// the kind the plan names, in order.
pub fn layer<O: HybridOps>(
    ops: &mut O,
    step: Step,
    plan: &LayerPlan<O::Ffn>,
) -> Result<(), O::Error> {
    match plan {
        LayerPlan::Mamba(p) => {
            ops.mamba_in(step, p)?;
            ops.mamba_core(step, p)?;
            ops.mamba_out(step, p)
        }
        LayerPlan::Attn(p) => {
            ops.attn_in(step, p)?;
            ops.attn_core(step, p)?;
            ops.attn_out(step, p)
        }
        LayerPlan::Ffn(f) => {
            ops.ffn_in(step, f)?;
            ops.ffn_core(step, f)?;
            ops.ffn_out(step, f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::plan::{LayerPlan, MambaPlan, NopeGqaPlan};
    use super::{HybridOps, Step, Steps, Walk, at, layer};
    use crate::ChainError;
    use crate::noleg::NoLeg;
    use crate::site::SiteTy;
    use model::arch::models::Mamba2;
    use runtime::sched::{self, Overlap, PortKind};

    /// Rows of the arena a test walk's port holds.
    const ARENA_ROWS: usize = 16;

    /// An op list that records each op it is given, as `name(layer)` with `, last` on the chain's
    /// last layer, and every step it was told.
    #[derive(Default)]
    struct Rec {
        log: Vec<String>,
        steps: Vec<Step>,
    }

    impl Rec {
        fn put(&mut self, op: &str, s: Step) -> Result<(), ChainError> {
            let last = if s.last { ", last" } else { "" };
            self.log.push(format!("{op}({}{last})", s.layer));
            self.steps.push(s);
            Ok(())
        }
    }

    impl HybridOps for Rec {
        type Error = ChainError;
        type Ffn = u8;

        fn embed(&mut self, m: usize) -> Result<(), ChainError> {
            self.log.push(format!("embed({m})"));
            Ok(())
        }
        fn mamba_in(&mut self, s: Step, _: &MambaPlan) -> Result<(), ChainError> {
            self.put("mamba_in", s)
        }
        fn mamba_core(&mut self, s: Step, _: &MambaPlan) -> Result<(), ChainError> {
            self.put("mamba_core", s)
        }
        fn mamba_out(&mut self, s: Step, _: &MambaPlan) -> Result<(), ChainError> {
            self.put("mamba_out", s)
        }
        fn attn_in(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), ChainError> {
            self.put("attn_in", s)
        }
        fn attn_core(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), ChainError> {
            self.put("attn_core", s)
        }
        fn attn_out(&mut self, s: Step, _: &NopeGqaPlan) -> Result<(), ChainError> {
            self.put("attn_out", s)
        }
        fn ffn_in(&mut self, s: Step, _: &u8) -> Result<(), ChainError> {
            self.put("ffn_in", s)
        }
        fn ffn_core(&mut self, s: Step, _: &u8) -> Result<(), ChainError> {
            self.put("ffn_core", s)
        }
        fn ffn_out(&mut self, s: Step, _: &u8) -> Result<(), ChainError> {
            self.put("ffn_out", s)
        }
        fn head(&mut self, m: usize) -> Result<(), ChainError> {
            self.log.push(format!("head({m})"));
            Ok(())
        }
    }

    fn mamba() -> LayerPlan<u8> {
        LayerPlan::Mamba(MambaPlan {
            norm: String::new(),
            ssm_in: String::new(),
            ssm_out: String::new(),
            a: String::new(),
            d: String::new(),
            dt_bias: String::new(),
            conv_w: String::new(),
            conv_b: None,
            ssm_norm: String::new(),
            shape: Mamba2 {
                d_inner: 512,
                d_state: 128,
                n_heads: 8,
                head_dim: 64,
                n_groups: 2,
                d_conv: 4,
                d_in_proj: 1544,
            },
            conv_dim: 1024,
            in_ty: SiteTy::Q5_0,
            out_ty: SiteTy::Q8_0,
        })
    }

    fn attn() -> LayerPlan<u8> {
        LayerPlan::Attn(NopeGqaPlan {
            norm: String::new(),
            q: String::new(),
            k: String::new(),
            v: String::new(),
            o: String::new(),
            n_head: 10,
            n_kv: 2,
            q_ty: SiteTy::Q5_0,
            k_ty: SiteTy::Q5_0,
            v_ty: SiteTy::Q5_0,
            o_ty: SiteTy::Q5_0,
        })
    }

    fn ffn() -> LayerPlan<u8> {
        LayerPlan::Ffn(7)
    }

    /// The order of a walk: the unit's front once ahead of layer 0, per layer its kind's three ops
    /// in order, the last layer marked, and the head after it, every op told the unit's rows.
    #[test]
    fn a_walk_orders_the_front_the_layers_and_the_head() {
        let plans = [mamba(), ffn(), attn()];
        for m in [1, ARENA_ROWS] {
            let mut ops = Rec::default();
            Walk::new(&mut ops, &plans, m)
                .run(ARENA_ROWS)
                .unwrap_or_else(|e| panic!("{e}"));
            let (embed, head) = (format!("embed({m})"), format!("head({m})"));
            assert_eq!(
                ops.log,
                [
                    embed.as_str(),
                    "mamba_in(0)",
                    "mamba_core(0)",
                    "mamba_out(0)",
                    "ffn_in(1)",
                    "ffn_core(1)",
                    "ffn_out(1)",
                    "attn_in(2, last)",
                    "attn_core(2, last)",
                    "attn_out(2, last)",
                    head.as_str(),
                ]
            );
            assert!(ops.steps.iter().all(|s| s.m == m), "{:?}", ops.steps);
        }
    }

    /// A kind comes from its plan, not from its place: the same plans in another order walk in that
    /// order, and a one-layer chain is both first and last.
    #[test]
    fn a_layer_runs_the_ops_of_its_plan_whatever_its_index() {
        let mut ops = Rec::default();
        Walk::new(&mut ops, &[attn(), mamba(), ffn()], 1)
            .run(ARENA_ROWS)
            .unwrap_or_else(|e| panic!("{e}"));
        let kinds: Vec<&str> = ops
            .log
            .iter()
            .filter(|l| l.contains("_in("))
            .map(String::as_str)
            .collect();
        assert_eq!(kinds, ["attn_in(0)", "mamba_in(1)", "ffn_in(2, last)"]);

        let mut one = Rec::default();
        Walk::new(&mut one, &[mamba()], 1)
            .run(ARENA_ROWS)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            one.log,
            [
                "embed(1)",
                "mamba_in(0, last)",
                "mamba_core(0, last)",
                "mamba_out(0, last)",
                "head(1)",
            ]
        );
    }

    /// A layer on its own is its kind's three ops with no front and no head, and `at` names the
    /// step: a layer the chain does not hold is refused by name.
    #[test]
    fn a_layer_alone_has_no_front_and_at_names_its_step() {
        let plans = [ffn(), mamba()];
        let mut ops = Rec::default();
        for l in 0..2 {
            let (step, plan) = at(&plans, l, 3).unwrap_or_else(|e| panic!("{e}"));
            layer(&mut ops, step, plan).unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(
            ops.log,
            [
                "ffn_in(0)",
                "ffn_core(0)",
                "ffn_out(0)",
                "mamba_in(1, last)",
                "mamba_core(1, last)",
                "mamba_out(1, last)",
            ]
        );
        let err = at(&plans, 2, 1).err().map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some("hybrid::at: layer 2 of a chain of 2 layers")
        );
    }

    /// A chain of no layer, a unit of no row or wider than the arena, and a point of more than one
    /// unit are each refused by name before any op, with the text a backend's error carries.
    #[test]
    fn what_a_walk_cannot_lay_out_is_refused_by_name() {
        let plans = [mamba()];
        let mut ops = Rec::default();
        let run = |ops: &mut Rec, plans: &[LayerPlan<u8>], m| {
            Walk::new(ops, plans, m)
                .run(ARENA_ROWS)
                .err()
                .map(|e| e.to_string())
        };
        assert_eq!(
            run(&mut ops, &[], 1).as_deref(),
            Some("hybrid::Walk: a chain of no layer")
        );
        assert_eq!(
            run(&mut ops, &plans, ARENA_ROWS + 1).as_deref(),
            Some(
                "chain::NoLeg::open: 1 units of 17 columns; a chain with no host leg walks \
                 one unit of 1..=16 columns (the arena's rows)"
            )
        );
        assert!(run(&mut ops, &plans, 0).is_some_and(|e| e.starts_with("chain::NoLeg: ")));
        let two = Overlap {
            units: 2,
            cols: 1,
            port: PortKind::Step,
        };
        let mut steps = Steps {
            walk: Walk::new(&mut ops, &plans, 1),
        };
        let err = sched::walk(two, 1, &mut NoLeg::new(ARENA_ROWS), &mut steps).err();
        assert_eq!(
            err.map(|e| e.to_string()).as_deref(),
            Some(
                "chain::NoLeg::open: 2 units of 1 columns; a chain with no host leg walks \
                 one unit of 1..=16 columns (the arena's rows)"
            )
        );
        assert!(ops.log.is_empty(), "{:?}", ops.log);
    }
}
