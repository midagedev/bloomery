//! A model's shape as the kernels are built for it, and the one place a
//! shape is matched to a compiled instance.
//!
//! A kernel whose width sizes a register array, a shared tile or an unrolled
//! lane walk is compiled once per width it serves: an instance. The rows of
//! [`ROUTERS`] and [`GQA`] are those instances, each naming the rule it
//! runs, the width it was built for, the range a runtime argument may take,
//! and the code that owns it. A shape read from a file ([`MoeShape::of`],
//! [`AttnShape::of`]) selects its row here ([`select_router`],
//! [`select_gqa`]); a shape no row serves is refused by name
//! ([`ShapeRefused`]), never mapped to the nearest row.
//!
//! The router crates name each width they compile by its row
//! ([`router_row`]) and hold the row to the body's rule and pick range with a
//! compile-time assert ([`router_row_is`]), so a row and its entries cannot
//! drift apart without a build failing. The coverage check
//! (`model::arch::coverage`) reads the same rows ([`select_router`],
//! [`gqa_row`]).

use std::fmt;

use crate::{Gqa, Moe, Score};

/// How a router turns logits into ids and weights: the numeric contract a
/// router body is written for, apart from its width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterRule {
    pub score: Score,
    /// A selection bias steers the choice only.
    pub bias: bool,
    /// The kept weights are renormalized to sum to one.
    pub norm: bool,
    /// A shared expert whose weight is the sigmoid of one more router row,
    /// run as one more slot after the routed ones.
    pub gated: bool,
}

/// A routed mixture's shape: its rule, the experts it routes over and the
/// experts each token keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeShape {
    pub rule: RouterRule,
    pub experts: u32,
    pub top_k: u32,
}

impl MoeShape {
    /// The shape of a layer's mixture, as its description states it.
    #[must_use]
    pub fn of(m: &Moe) -> MoeShape {
        MoeShape {
            rule: RouterRule {
                score: m.router.score,
                bias: m.router.bias,
                norm: m.router.norm,
                gated: m.shared.is_some_and(|s| s.sigmoid_gate),
            },
            experts: m.experts,
            top_k: m.top_k,
        }
    }
}

/// A router body: one kernel family's code, compiled at one or more widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterBody {
    /// `gpu/src/router.rs` `router_topk`: one thread a token, serial sums.
    Topk,
    /// `gpu/src/arch/qwen3moe/router.rs` `qwen3moe_router_*`: a warp a token,
    /// four named logits a lane.
    Qwen3moe,
    /// `gpu/src/arch/qwen3moe/router.rs` `qwen35moe_router_*`: a warp a token,
    /// the logits in shared memory, the shared expert's gate as the last row.
    Qwen35moe,
    /// `gpu-deepseek41/src/router.rs` `ds41_router*`.
    Ds41,
    /// `gpu-deepseek41/src/experts_mxfp4.rs` `dflash_router`: `ds41_router`'s
    /// body with `norm` a launch argument.
    Dflash,
}

/// One compiled router instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterInst {
    pub body: RouterBody,
    pub rule: RouterRule,
    /// The body takes `norm` as a launch argument: it serves both values of
    /// [`RouterRule::norm`], and the row's own value is not compared.
    pub norm_arg: bool,
    /// Experts each lane of the selecting warp owns (`lane + 32 j`); the
    /// instance routes over `32 · per_lane` experts.
    pub per_lane: u32,
    /// The kept experts a launch may ask for, inclusive. A body whose pick
    /// count is compiled in serves one value.
    pub top_k_min: u32,
    pub top_k_max: u32,
    /// The code that owns the instance.
    pub at: &'static str,
}

impl RouterInst {
    /// Experts the instance routes over.
    #[must_use]
    pub const fn experts(&self) -> u32 {
        32 * self.per_lane
    }

    /// Whether this row's body runs `rule`.
    #[must_use]
    pub const fn runs(&self, rule: RouterRule) -> bool {
        score_eq(self.rule.score, rule.score)
            && self.rule.bias == rule.bias
            && self.rule.gated == rule.gated
            && (self.norm_arg || self.rule.norm == rule.norm)
    }
}

/// The largest pick count a lane-held selection takes: pick `s` lives in
/// lane `s` of the routing warp.
pub const LANE_PICKS: u32 = 32;

const SOFTMAX_TOPK: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: false,
    gated: false,
};
const SOFTMAX_NORM: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: true,
    gated: false,
};
const SOFTMAX_NORM_GATED: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: true,
    gated: true,
};
const BIASED_SQRT_SOFTPLUS: RouterRule = RouterRule {
    score: Score::SqrtSoftplus,
    bias: true,
    norm: true,
    gated: false,
};

/// Every compiled router instance. A body's rows share its rule; a width
/// is a row only while its entries exist.
pub const ROUTERS: &[RouterInst] = &[
    RouterInst {
        body: RouterBody::Topk,
        rule: SOFTMAX_TOPK,
        norm_arg: false,
        per_lane: 2,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/router.rs router_topk",
    },
    RouterInst {
        body: RouterBody::Qwen3moe,
        rule: SOFTMAX_NORM,
        norm_arg: false,
        per_lane: 4,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu/src/arch/qwen3moe/router.rs qwen3moe_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        per_lane: 8,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu/src/arch/qwen3moe/router.rs qwen35moe_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        per_lane: 16,
        top_k_min: 10,
        top_k_max: 10,
        at: "gpu/src/arch/qwen3moe/router.rs qwen35moe_router_*_512",
    },
    RouterInst {
        body: RouterBody::Ds41,
        rule: BIASED_SQRT_SOFTPLUS,
        norm_arg: false,
        per_lane: 12,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu-deepseek41/src/router.rs ds41_router*",
    },
    RouterInst {
        body: RouterBody::Dflash,
        rule: BIASED_SQRT_SOFTPLUS,
        norm_arg: true,
        per_lane: 4,
        top_k_min: 3,
        top_k_max: 3,
        at: "gpu-deepseek41/src/experts_mxfp4.rs dflash_router",
    },
];

/// Why a shape has no instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No router body runs this rule at any width.
    NoRouterRule,
    /// The rule's bodies are built for other expert counts.
    RouterWidth { served: Vec<u32> },
    /// The instance serves another range of kept experts.
    TopK { min: u32, max: u32 },
    /// Query heads that are not a whole multiple of the key heads.
    Group,
    /// No flash is built for this head width.
    HeadWidth { served: Vec<u32> },
    /// The head width's flashes take another query-head group.
    GqaGroup { group: u32, served: Vec<String> },
}

/// A shape read from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Moe(MoeShape),
    Attn(AttnShape),
}

/// A shape no instance serves: the shape, and what refused it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeRefused {
    pub shape: Shape,
    pub why: Refusal,
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shape::Moe(s) => {
                let r = s.rule;
                write!(
                    f,
                    "router {:?}{}{}{}, {} experts, top {}",
                    r.score,
                    if r.bias { " with a selection bias" } else { "" },
                    if r.norm { " renormalized" } else { "" },
                    if r.gated {
                        " with a gated shared expert"
                    } else {
                        ""
                    },
                    s.experts,
                    s.top_k
                )
            }
            Shape::Attn(a) => write!(f, "attention {}/{} heads of {}", a.n_head, a.n_kv, a.head),
        }
    }
}

impl fmt::Display for ShapeRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.shape)?;
        match &self.why {
            Refusal::NoRouterRule => f.write_str("no router body runs this rule"),
            Refusal::RouterWidth { served } => {
                write!(
                    f,
                    "this rule's router bodies are built for {served:?} experts"
                )
            }
            Refusal::TopK { min, max } => {
                write!(f, "the instance keeps {min}..={max} experts a token")
            }
            Refusal::Group => f.write_str("the query heads are not a multiple of the key heads"),
            Refusal::HeadWidth { served } => {
                write!(f, "no flash is built for this head; heads {served:?} are")
            }
            Refusal::GqaGroup { group, served } => write!(
                f,
                "group {group}; this head's flashes take {}",
                served.join(", ")
            ),
        }
    }
}

impl std::error::Error for ShapeRefused {}

/// The router instance that serves `s`: the row whose body runs its rule at
/// its expert count, `top_k` inside the row's range and below the count.
///
/// # Errors
///
/// [`ShapeRefused`] naming `s` when no row runs the rule, none is built for
/// the count, or the row's range does not hold `top_k`.
pub fn select_router(s: MoeShape) -> Result<RouterInst, ShapeRefused> {
    let refuse = |why| ShapeRefused {
        shape: Shape::Moe(s),
        why,
    };
    let rule: Vec<&RouterInst> = ROUTERS.iter().filter(|r| r.runs(s.rule)).collect();
    if rule.is_empty() {
        return Err(refuse(Refusal::NoRouterRule));
    }
    let Some(row) = rule.iter().find(|r| r.experts() == s.experts) else {
        return Err(refuse(Refusal::RouterWidth {
            served: rule.iter().map(|r| r.experts()).collect(),
        }));
    };
    let max = row.top_k_max.min(row.experts());
    if !(row.top_k_min..=max).contains(&s.top_k) {
        return Err(refuse(Refusal::TopK {
            min: row.top_k_min,
            max,
        }));
    }
    Ok(**row)
}

/// Body `body`'s row at `per_lane`: the const form a device crate holds
/// each width it compiles to. Fails the build when the table has no such
/// row.
#[must_use]
pub const fn router_row(body: RouterBody, per_lane: u32) -> RouterInst {
    let mut i = 0;
    while i < ROUTERS.len() {
        let r = ROUTERS[i];
        if body_eq(r.body, body) && r.per_lane == per_lane {
            return r;
        }
        i += 1;
    }
    panic!("no router row for this body at this width")
}

/// Whether row `r` is body `body` at `per_lane`, with rule `rule` and the
/// pick range `min..=max`: the whole of what a device crate compiled in.
#[must_use]
pub const fn router_row_is(
    r: RouterInst,
    rule: RouterRule,
    norm_arg: bool,
    top_k: (u32, u32),
) -> bool {
    score_eq(r.rule.score, rule.score)
        && r.rule.bias == rule.bias
        && r.rule.norm == rule.norm
        && r.rule.gated == rule.gated
        && r.norm_arg == norm_arg
        && r.top_k_min == top_k.0
        && r.top_k_max == top_k.1
}

/// The router rules the device bodies are written for, by body.
pub mod rules {
    use super::RouterRule;

    /// Softmax, the top k, the kept probabilities times a scale.
    pub const SOFTMAX_TOPK: RouterRule = super::SOFTMAX_TOPK;
    /// Softmax, the top k renormalized.
    pub const SOFTMAX_NORM: RouterRule = super::SOFTMAX_NORM;
    /// [`SOFTMAX_NORM`] and a sigmoid-gated shared expert as one more slot.
    pub const SOFTMAX_NORM_GATED: RouterRule = super::SOFTMAX_NORM_GATED;
    /// √softplus scores, a selection bias, the kept scores renormalized.
    pub const BIASED_SQRT_SOFTPLUS: RouterRule = super::BIASED_SQRT_SOFTPLUS;
}

const fn body_eq(a: RouterBody, b: RouterBody) -> bool {
    a as u8 == b as u8
}

const fn score_eq(a: Score, b: Score) -> bool {
    matches!(
        (a, b),
        (Score::Softmax, Score::Softmax)
            | (Score::Sigmoid, Score::Sigmoid)
            | (Score::SqrtSoftplus, Score::SqrtSoftplus)
    )
}

/// A grouped-query attention layer's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub n_head: u32,
    pub n_kv: u32,
    /// Values per head.
    pub head: u32,
}

impl AttnShape {
    /// The shape of a GQA mixer, as its description states it.
    #[must_use]
    pub fn of(g: &Gqa) -> AttnShape {
        AttnShape {
            n_head: g.heads,
            n_kv: g.kv_heads,
            head: g.head_dim,
        }
    }
}

/// Which query-head groups a flash instance takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupRule {
    /// One block a key head: the group is the pack.
    One,
    /// `group / pack` blocks a key head, a launch argument.
    Packs,
}

/// One compiled GQA flash instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqaInst {
    /// Values per head: the tiles' and the register vectors' width.
    pub head: u32,
    /// Query heads one block holds.
    pub pack: u32,
    pub group: GroupRule,
    /// The code that owns the instance.
    pub at: &'static str,
}

impl GqaInst {
    /// Whether this row takes `group` query heads a key head.
    #[must_use]
    pub const fn takes(&self, group: u32) -> bool {
        match self.group {
            GroupRule::One => group == self.pack,
            GroupRule::Packs => group > 0 && group.is_multiple_of(self.pack),
        }
    }

    /// Blocks a key head at `group` query heads a key head.
    #[must_use]
    pub const fn packs(&self, group: u32) -> u32 {
        group / self.pack
    }
}

/// Every compiled GQA flash instance, in the order [`select_gqa`] prefers
/// them: at one head width, a one-block-a-group row before a packed one.
pub const GQA: &[GqaInst] = &[
    GqaInst {
        head: 128,
        pack: 8,
        group: GroupRule::One,
        at: "gpu/src/flash_gqa.rs HEAD, GROUP",
    },
    GqaInst {
        head: 256,
        pack: 8,
        group: GroupRule::One,
        at: "gpu/src/flash_gqa.rs HEAD_256, GROUP (the _256 entries)",
    },
    GqaInst {
        head: 256,
        pack: 4,
        group: GroupRule::Packs,
        at: "gpu/src/flash_gqa.rs PACK_4 (the _256_p4 entries)",
    },
];

/// The first GQA row, in [`GQA`]'s order, built for `s.head` that takes its
/// group.
///
/// # Errors
///
/// [`ShapeRefused`] naming `s` when the heads do not group, no row has its
/// head width, or none of those takes its group.
pub fn select_gqa(s: AttnShape) -> Result<GqaInst, ShapeRefused> {
    let refuse = |why| ShapeRefused {
        shape: Shape::Attn(s),
        why,
    };
    if s.n_kv == 0 || s.n_head == 0 || !s.n_head.is_multiple_of(s.n_kv) {
        return Err(refuse(Refusal::Group));
    }
    let group = s.n_head / s.n_kv;
    let width: Vec<&GqaInst> = GQA.iter().filter(|r| r.head == s.head).collect();
    if width.is_empty() {
        let mut served: Vec<u32> = GQA.iter().map(|r| r.head).collect();
        served.dedup();
        return Err(refuse(Refusal::HeadWidth { served }));
    }
    match gqa_row(s.head, group) {
        Some(r) => Ok(r),
        None => Err(refuse(Refusal::GqaGroup {
            group,
            served: width
                .iter()
                .map(|r| match r.group {
                    GroupRule::One => format!("exactly {}", r.pack),
                    GroupRule::Packs => format!("a multiple of {}", r.pack),
                })
                .collect(),
        })),
    }
}

/// The first row, in [`GQA`]'s order, built for `head` that takes `group`
/// query heads a key head.
#[must_use]
pub fn gqa_row(head: u32, group: u32) -> Option<GqaInst> {
    GQA.iter()
        .find(|r| r.head == head && r.takes(group))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::{
        AttnShape, GQA, GroupRule, MoeShape, ROUTERS, Refusal, RouterRule, Shape, rules,
        select_gqa, select_router,
    };
    use crate::{Act, Moe, Router, Score, Shared};

    fn moe(rule: RouterRule, experts: u32, top_k: u32) -> MoeShape {
        MoeShape {
            rule,
            experts,
            top_k,
        }
    }

    /// The expert counts and pick counts of every model this tree runs or
    /// reads, each with the source that states it.
    fn models() -> Vec<(&'static str, MoeShape)> {
        vec![
            // gpu/src/arch/deepseek2/pins.rs: V2-Lite's expert_count 64, used 6,
            // norm_topk_prob false.
            ("V2-Lite", moe(rules::SOFTMAX_TOPK, 64, 6)),
            // model/src/arch/coverage.rs's V4.1 router row: sqrt-softplus 384/6, bias, norm.
            ("V4.1", moe(rules::BIASED_SQRT_SOFTPLUS, 384, 6)),
            // model/tests/qwen3moe_meta.rs:409 "moe 128/8 … Softmax bias false norm true".
            ("Qwen3", moe(rules::SOFTMAX_NORM, 128, 8)),
            // model/tests/qwen35moe_meta.rs:33 "moe 256/8 … norm true … gate true".
            ("Qwen3.6", moe(rules::SOFTMAX_NORM_GATED, 256, 8)),
            // Qwen3.8 (q38router, 40179b3): 512 experts, 10 used, the gated shared expert.
            ("Qwen3.8", moe(rules::SOFTMAX_NORM_GATED, 512, 10)),
            // model/src/arch/dspark.rs: the draft routes 128/3 by sqrt-softplus with a bias.
            ("DSpark", moe(rules::BIASED_SQRT_SOFTPLUS, 128, 3)),
        ]
    }

    #[test]
    fn every_model_the_tree_runs_selects_its_router() {
        for (name, s) in models() {
            let row = select_router(s).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(row.experts(), s.experts, "{name}");
            assert!(row.runs(s.rule), "{name}");
        }
    }

    /// A draft that does not renormalize runs the same body, `norm` its
    /// launch argument; V4.1's body renormalizes always.
    #[test]
    fn a_norm_argument_row_serves_both_norms_and_only_it() {
        let mut off = rules::BIASED_SQRT_SOFTPLUS;
        off.norm = false;
        assert!(select_router(moe(off, 128, 3)).is_ok());
        let e = select_router(moe(off, 384, 6)).expect_err("V4.1 at norm off");
        assert!(matches!(e.why, Refusal::RouterWidth { ref served } if served == &[128]));
    }

    /// GLM-5.3-Flash (model/tests/glm5next_meta.rs:83, "router: sigmoid, 288
    /// experts, top 8, with a selection bias"): no router body scores with a
    /// sigmoid, so it is refused by name, the rule and the count in the text.
    #[test]
    fn glm_5_3_flash_router_is_refused_by_name() {
        let glm = moe(
            RouterRule {
                score: Score::Sigmoid,
                bias: true,
                norm: true,
                gated: false,
            },
            288,
            8,
        );
        let e = select_router(glm).expect_err("GLM-5.3-Flash selected a router");
        assert_eq!(e.why, Refusal::NoRouterRule);
        let text = e.to_string();
        assert!(
            text.contains("Sigmoid") && text.contains("288 experts"),
            "{text}"
        );
    }

    /// An expert count between two rows of a rule is refused, never taken by
    /// the nearest row.
    #[test]
    fn a_width_between_rows_is_refused() {
        for experts in [384, 257, 255, 1024, 0] {
            let e = select_router(moe(rules::SOFTMAX_NORM_GATED, experts, 8))
                .expect_err("selected between rows");
            assert_eq!(
                e.why,
                Refusal::RouterWidth {
                    served: vec![256, 512]
                },
                "{experts}"
            );
            assert_eq!(
                e.shape,
                Shape::Moe(moe(rules::SOFTMAX_NORM_GATED, experts, 8))
            );
        }
    }

    /// A pick count outside a row's range is refused: zero, past the lane
    /// picks, and any count but the compiled one of a fixed-pick body.
    #[test]
    fn a_top_k_outside_the_row_is_refused() {
        for (s, min, max) in [
            (moe(rules::SOFTMAX_NORM_GATED, 256, 0), 8, 8),
            (moe(rules::SOFTMAX_NORM_GATED, 256, 10), 8, 8),
            (moe(rules::SOFTMAX_NORM_GATED, 512, 8), 10, 10),
            (moe(rules::SOFTMAX_TOPK, 64, 8), 6, 6),
            (moe(rules::BIASED_SQRT_SOFTPLUS, 384, 8), 6, 6),
            (moe(rules::BIASED_SQRT_SOFTPLUS, 128, 6), 3, 3),
        ] {
            let e = select_router(s).expect_err("selected outside the row");
            assert_eq!(e.why, Refusal::TopK { min, max }, "{s:?}");
        }
    }

    /// The shape is read from the description: the gate comes from the
    /// shared expert's `sigmoid_gate`, not from its presence.
    #[test]
    fn the_shape_follows_the_description() {
        let mut m = Moe {
            experts: 256,
            top_k: 8,
            expert_ff: 512,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Softmax,
                bias: false,
                norm: true,
                scale: 1.0,
                hash: false,
            },
            shared: Some(Shared {
                ff: 512,
                act: Act::SwiGlu { limit: None },
                sigmoid_gate: true,
            }),
        };
        assert_eq!(MoeShape::of(&m), moe(rules::SOFTMAX_NORM_GATED, 256, 8));
        if let Some(s) = &mut m.shared {
            s.sigmoid_gate = false;
        }
        assert_eq!(MoeShape::of(&m).rule, rules::SOFTMAX_NORM);
        m.shared = None;
        assert_eq!(MoeShape::of(&m).rule, rules::SOFTMAX_NORM);
    }

    /// Rows are distinct instances: no two rows run one rule at one width.
    #[test]
    fn no_two_router_rows_overlap() {
        for (i, a) in ROUTERS.iter().enumerate() {
            for b in &ROUTERS[i + 1..] {
                let overlap = a.per_lane == b.per_lane && (a.runs(b.rule) || b.runs(a.rule));
                assert!(!overlap, "{} and {}", a.at, b.at);
                assert!(
                    (a.body, a.per_lane) != (b.body, b.per_lane),
                    "{} and {}",
                    a.at,
                    b.at
                );
            }
            assert!(a.top_k_min >= 1 && a.top_k_min <= a.top_k_max, "{}", a.at);
            assert!(a.per_lane >= 1 && a.per_lane <= 32, "{}", a.at);
        }
    }

    fn attn(n_head: u32, n_kv: u32, head: u32) -> AttnShape {
        AttnShape { n_head, n_kv, head }
    }

    /// Qwen3 32/4 × 128 (qwen3moe_meta.rs:409), Qwen3.6 16/2 × 256
    /// (qwen35moe_meta.rs:34), Qwen3.8 24/2 × 256 (q38gqa, 9e5b716).
    #[test]
    fn every_attention_the_tree_runs_selects_its_flash() {
        for (name, s, head, pack, packs) in [
            ("Qwen3", attn(32, 4, 128), 128, 8, 1),
            ("Qwen3.6", attn(16, 2, 256), 256, 8, 1),
            ("Qwen3.8", attn(24, 2, 256), 256, 4, 3),
        ] {
            let row = select_gqa(s).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((row.head, row.pack), (head, pack), "{name}");
            assert_eq!(row.packs(s.n_head / s.n_kv), packs, "{name}");
        }
    }

    #[test]
    fn an_attention_no_flash_takes_is_refused() {
        let e = select_gqa(attn(24, 2, 128)).expect_err("head 128 at group 12");
        assert_eq!(
            e.why,
            Refusal::GqaGroup {
                group: 12,
                served: vec!["exactly 8".to_string()]
            }
        );
        let e = select_gqa(attn(16, 2, 192)).expect_err("head 192");
        assert_eq!(
            e.why,
            Refusal::HeadWidth {
                served: vec![128, 256]
            }
        );
        for s in [attn(24, 5, 256), attn(24, 0, 256), attn(0, 2, 256)] {
            assert_eq!(select_gqa(s).expect_err("ungrouped").why, Refusal::Group);
        }
        let e = select_gqa(attn(12, 2, 256)).expect_err("group 6");
        assert!(e.to_string().contains("group 6"), "{e}");
    }

    #[test]
    fn gqa_rows_are_distinct_and_packed_rows_divide() {
        for (i, a) in GQA.iter().enumerate() {
            for b in &GQA[i + 1..] {
                assert!(
                    (a.head, a.pack) != (b.head, b.pack),
                    "{} and {}",
                    a.at,
                    b.at
                );
            }
            if a.group == GroupRule::Packs {
                assert!(8u32.is_multiple_of(a.pack), "{}", a.at);
            }
        }
    }
}
