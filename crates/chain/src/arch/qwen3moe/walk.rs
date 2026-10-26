//! The family's walk: the order of one layer's ops — the mixer, then the FFN half — with the unit's
//! front ahead of layer 0 and its end after the last layer, written once over [`ChainOps`] for
//! every backend.
//!
//! A chain with no host leg walks one unit of `m` rows through [`runtime::sched::walk`]
//! ([`Walk::run`]): each layer is the walk's front, the head its end, and a port that exchanges
//! nothing. A caller that lays a copy between a layer's mixer and its FFN half gives it to
//! [`Walk::run_between`]; a caller that runs one layer, or one FFN half, from rows it laid in runs
//! [`layer`] or [`ffn`]; a walk with a host leg runs each layer's [`front`] and then its FFN half
//! around the leg.
//!
//! What this order does not cover, so no backend reads it as covering it: a walk with a host leg
//! splits its FFN half around the leg (the CUDA placed load's router, card experts and combine
//! with the host tier's sum), which [`route`](ChainOps::route) and
//! [`experts`](ChainOps::experts) do not — the card experts of a placed layer run in the leg's
//! shadow, by the dispatch's launches or, for a layer whose stack types only the common card leg
//! runs, by that leg's; the CUDA Qwen3.8 body's walks, which have host legs of their own, and the
//! CUDA whole-card GEMM prefill keep layer programs of their own.

use super::ops::{ChainOps, LayerCtx};
use super::plan::{LayerPlan, MixerPlan};
use crate::ChainError;
use crate::noleg::NoLeg;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

const WHAT: &str = "qwen3moe::Walk";

/// One walk of a chain: every layer of `plans` at `m` rows over a backend's ops, then the unit's
/// head.
pub struct Walk<'a, O: ChainOps> {
    ops: &'a mut O,
    plans: &'a [LayerPlan],
    m: usize,
}

impl<'a, O: ChainOps> Walk<'a, O> {
    /// The walk of `plans` at `m` rows over `ops`.
    #[must_use]
    pub fn new(ops: &'a mut O, plans: &'a [LayerPlan], m: usize) -> Walk<'a, O> {
        Walk { ops, plans, m }
    }

    /// Enqueue the walk `(1, m, Step)` over an arena of `rows` rows: the unit's front, every
    /// layer, the head. A chain of no layer and `m` outside `1..=rows` are refused by name before
    /// any op.
    pub fn run(self, rows: usize) -> Result<(), O::Error> {
        self.run_between(rows, |_, _| Ok(()))
    }

    /// [`Walk::run`], with `between` run on every layer between its mixer and its FFN half: the
    /// rows entering the layer are the residual stream's there (the mixer reads them and writes
    /// the FFN's input; the FFN half's combine is the first op to write the layer's output).
    pub fn run_between<H>(self, rows: usize, between: H) -> Result<(), O::Error>
    where
        H: FnMut(&mut O, LayerCtx<'_>) -> Result<(), O::Error>,
    {
        if self.plans.is_empty() {
            return Err(ChainError::shape(WHAT, "a chain of no layer").into());
        }
        let o = Overlap {
            units: 1,
            cols: self.m,
            port: PortKind::Step,
        };
        let layers = self.plans.len();
        sched::walk(
            o,
            layers,
            &mut NoLeg::new(rows),
            &mut Steps {
                walk: self,
                between,
            },
        )
    }
}

/// A [`Walk`] as the runtime's layer program, with the copy it lays between each mixer and FFN half.
struct Steps<'a, O: ChainOps, H> {
    walk: Walk<'a, O>,
    between: H,
}

impl<O, H> LayerProgram for Steps<'_, O, H>
where
    O: ChainOps,
    H: FnMut(&mut O, LayerCtx<'_>) -> Result<(), O::Error>,
{
    type Port = NoLeg<O::Error>;

    /// The whole layer at `m` rows: a chain with no host leg has no shadow and no back.
    fn front(&mut self, _: &mut Self::Port, at: At) -> Result<(), O::Error> {
        let Walk { ops, plans, m } = &mut self.walk;
        let cx = LayerCtx::of(plans, at.layer, *m)?;
        front(*ops, cx)?;
        (self.between)(*ops, cx)?;
        ffn(*ops, cx)
    }

    fn end(&mut self, _: usize) -> Result<(), O::Error> {
        self.walk.ops.head(self.walk.m)
    }
}

/// What a walk runs of layer `cx` ahead of its FFN half: the unit's front ahead of layer 0, then
/// the mixer the plan names, which leaves the FFN's input.
pub fn front<O: ChainOps>(ops: &mut O, cx: LayerCtx<'_>) -> Result<(), O::Error> {
    if cx.layer == 0 {
        ops.embed(cx.m)?;
    }
    mixer(ops, cx)
}

/// Layer `cx` on its own from the residual rows its caller laid in: the mixer, then the FFN half
/// — what a walk runs of a layer after the unit's front.
pub fn layer<O: ChainOps>(ops: &mut O, cx: LayerCtx<'_>) -> Result<(), O::Error> {
    mixer(ops, cx)?;
    ffn(ops, cx)
}

/// The mixer the plan names: its input side, its middle and its output side.
fn mixer<O: ChainOps>(ops: &mut O, cx: LayerCtx<'_>) -> Result<(), O::Error> {
    match &cx.plan.mixer {
        MixerPlan::Gqa(g) => {
            ops.attn_in(cx, g)?;
            ops.attn_core(cx, g)?;
            ops.attn_out(cx, g)
        }
        MixerPlan::Delta(d) => {
            ops.delta_in(cx, d)?;
            ops.delta_core(cx, d)?;
            ops.delta_out(cx, d)
        }
    }
}

/// One layer's FFN half: the router's pick, then the experts and their combine into the layer's
/// output rows.
pub fn ffn<O: ChainOps>(ops: &mut O, cx: LayerCtx<'_>) -> Result<(), O::Error> {
    let f = &cx.plan.ffn;
    ops.route(cx, f)?;
    ops.experts(cx, f)
}

#[cfg(test)]
mod tests {
    use super::super::plan::{
        DeltaPlan, FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, LayerPlan, MixerPlan, SiteTy,
    };
    use super::{ChainOps, LayerCtx, Steps, Walk, ffn, front, layer};
    use crate::ChainError;
    use crate::linear::{KHeadMap, LinearShape};
    use crate::noleg::NoLeg;
    use runtime::sched::{self, Overlap, PortKind};

    /// Rows of the arena a test walk's port holds.
    const ARENA_ROWS: usize = 16;

    /// An op list that records each op it is given, as `name` or `name(layer)`.
    #[derive(Default)]
    struct Rec {
        log: Vec<String>,
    }

    impl Rec {
        fn put(&mut self, op: &str) -> Result<(), ChainError> {
            self.log.push(op.to_string());
            Ok(())
        }
    }

    impl ChainOps for Rec {
        type Error = ChainError;

        fn embed(&mut self, m: usize) -> Result<(), ChainError> {
            self.put(&format!("embed({m})"))
        }
        fn attn_in(&mut self, _: LayerCtx<'_>, _: &GqaPlan) -> Result<(), ChainError> {
            self.put("attn_in")
        }
        fn attn_core(&mut self, _: LayerCtx<'_>, _: &GqaPlan) -> Result<(), ChainError> {
            self.put("attn_core")
        }
        fn attn_out(&mut self, _: LayerCtx<'_>, _: &GqaPlan) -> Result<(), ChainError> {
            self.put("attn_out")
        }
        fn delta_in(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), ChainError> {
            self.put("delta_in")
        }
        fn delta_core(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), ChainError> {
            self.put("delta_core")
        }
        fn delta_out(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), ChainError> {
            self.put("delta_out")
        }
        fn route(&mut self, _: LayerCtx<'_>, _: &FfnPlan) -> Result<(), ChainError> {
            self.put("route")
        }
        fn experts(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), ChainError> {
            self.put(&format!("experts({})", l.layer))
        }
        fn head(&mut self, m: usize) -> Result<(), ChainError> {
            self.put(&format!("head({m})"))
        }
    }

    fn ffn_plan() -> FfnPlan {
        FfnPlan {
            ffn_norm: String::new(),
            route: FfnRoute::Dense,
            gate: String::new(),
            up: String::new(),
            down: String::new(),
            gate_ty: SiteTy::Q4K,
            up_ty: SiteTy::Q4K,
            down_ty: SiteTy::Q4K,
        }
    }

    fn gqa_layer() -> LayerPlan {
        LayerPlan {
            mixer: MixerPlan::Gqa(GqaPlan {
                kind: GqaKind::Gated256,
                flash: Flash::Group,
                attn_norm: String::new(),
                attn_q: String::new(),
                attn_k: String::new(),
                attn_v: String::new(),
                attn_q_norm: String::new(),
                attn_k_norm: String::new(),
                attn_output: String::new(),
                q_ty: SiteTy::Q4K,
                k_ty: SiteTy::Q4K,
                v_ty: SiteTy::Q4K,
                o_ty: SiteTy::Q4K,
            }),
            ffn: ffn_plan(),
        }
    }

    fn delta_layer() -> LayerPlan {
        LayerPlan {
            mixer: MixerPlan::Delta(DeltaPlan {
                attn_norm: String::new(),
                qkv: String::new(),
                gate: String::new(),
                beta: String::new(),
                alpha: String::new(),
                conv: String::new(),
                dt_bias: String::new(),
                ssm_a: String::new(),
                ssm_norm: String::new(),
                ssm_out: String::new(),
                shape: LinearShape {
                    n_k: 16,
                    n_v: 32,
                    map: KHeadMap::Tiled,
                },
                qkv_ty: SiteTy::Q4K,
                gate_ty: SiteTy::Q4K,
                beta_ty: SiteTy::Q4K,
                alpha_ty: SiteTy::Q4K,
                out_ty: SiteTy::Q4K,
            }),
            ffn: ffn_plan(),
        }
    }

    /// The order of a walk: the unit's front once ahead of layer 0, per layer the mixer's three
    /// ops by its plan, the router's pick and the experts, and the head after the last layer —
    /// the same at every `m`.
    #[test]
    fn a_walk_orders_the_front_the_layers_and_the_head() {
        let plans = [gqa_layer(), delta_layer()];
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
                    "attn_in",
                    "attn_core",
                    "attn_out",
                    "route",
                    "experts(0)",
                    "delta_in",
                    "delta_core",
                    "delta_out",
                    "route",
                    "experts(1)",
                    head.as_str(),
                ]
            );
        }
    }

    /// What a caller lays between a layer's halves runs on every layer, after the mixer's last op
    /// and ahead of the FFN half's first, with the layer's context.
    #[test]
    fn between_runs_after_each_mixer_and_ahead_of_its_ffn() {
        let plans = [delta_layer(), gqa_layer()];
        let mut ops = Rec::default();
        Walk::new(&mut ops, &plans, 1)
            .run_between(ARENA_ROWS, |o: &mut Rec, cx| {
                o.put(&format!("between({}, last {})", cx.layer, cx.last))
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            ops.log,
            [
                "embed(1)",
                "delta_in",
                "delta_core",
                "delta_out",
                "between(0, last false)",
                "route",
                "experts(0)",
                "attn_in",
                "attn_core",
                "attn_out",
                "between(1, last true)",
                "route",
                "experts(1)",
                "head(1)",
            ]
        );
    }

    /// The pieces a caller runs alone: the front holds the unit's front on layer 0 only and ends
    /// before the FFN half, a layer on its own is its mixer then its FFN half with no front, and
    /// the FFN half alone is the pick then the experts.
    #[test]
    fn the_pieces_a_caller_runs_alone() {
        let plans = [gqa_layer(), delta_layer()];
        let at = |l| LayerCtx::of(&plans, l, 3).unwrap_or_else(|e| panic!("{e}"));
        let mut ops = Rec::default();
        front(&mut ops, at(0)).unwrap_or_else(|e| panic!("{e}"));
        front(&mut ops, at(1)).unwrap_or_else(|e| panic!("{e}"));
        layer(&mut ops, at(0)).unwrap_or_else(|e| panic!("{e}"));
        ffn(&mut ops, at(1)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            ops.log,
            [
                "embed(3)",
                "attn_in",
                "attn_core",
                "attn_out",
                "delta_in",
                "delta_core",
                "delta_out",
                "attn_in",
                "attn_core",
                "attn_out",
                "route",
                "experts(0)",
                "route",
                "experts(1)",
            ]
        );
        assert!(!at(0).last && at(1).last);
    }

    /// A layer the chain does not hold, a chain of no layer, a unit of no row or wider than the
    /// arena, and a point of more than one unit are each refused by name before any op, with the
    /// text a backend's error carries.
    #[test]
    fn what_a_walk_cannot_lay_out_is_refused_by_name() {
        let plans = [gqa_layer()];
        let err = LayerCtx::of(&plans, 3, 1).err().map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some("qwen3moe::LayerCtx::of: layer 3 of a chain of 1 layers")
        );
        let mut ops = Rec::default();
        let run = |ops: &mut Rec, plans: &[LayerPlan], m| {
            Walk::new(ops, plans, m)
                .run(ARENA_ROWS)
                .err()
                .map(|e| e.to_string())
        };
        assert_eq!(
            run(&mut ops, &[], 1).as_deref(),
            Some("qwen3moe::Walk: a chain of no layer")
        );
        assert_eq!(
            run(&mut ops, &plans, ARENA_ROWS + 1).as_deref(),
            Some(
                "chain::NoLeg::open: 1 units of 17 columns; a chain with no host leg walks \
                 one unit of 1..=16 columns (the arena's rows)"
            )
        );
        assert!(run(&mut ops, &plans, 0).is_some_and(|e| e.starts_with("chain::NoLeg: ")));
        fn none(_: &mut Rec, _: LayerCtx<'_>) -> Result<(), ChainError> {
            Ok(())
        }
        let two = Overlap {
            units: 2,
            cols: 1,
            port: PortKind::Step,
        };
        let mut steps = Steps {
            walk: Walk::new(&mut ops, &plans, 1),
            between: none,
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
