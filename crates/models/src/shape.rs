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
    /// `gpu-deepseek41/src/router.rs` `glm5next_router*`: sigmoid scores, the
    /// bias added for the pick only.
    Glm5next,
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
const BIASED_SIGMOID: RouterRule = RouterRule {
    score: Score::Sigmoid,
    bias: true,
    norm: true,
    gated: false,
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
        top_k_min: 1,
        top_k_max: LANE_PICKS,
        at: "gpu/src/arch/qwen3moe/router.rs qwen3moe_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        per_lane: 8,
        top_k_min: 1,
        top_k_max: LANE_PICKS,
        at: "gpu/src/arch/qwen3moe/router.rs qwen35moe_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        per_lane: 16,
        top_k_min: 1,
        top_k_max: LANE_PICKS,
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
    RouterInst {
        body: RouterBody::Glm5next,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        per_lane: 9,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu-deepseek41/src/router.rs glm5next_router*",
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
    /// No flash is built for this value width at this head width: `served` are
    /// the value widths the head's rows carry.
    ValueWidth {
        head: u32,
        value: u32,
        served: Vec<u32>,
    },
    /// The layer attends a window of the positions: the selected row cuts none.
    Window { positions: u32 },
    /// The layer carries per-head softmax sinks: the selected row folds none in.
    Sinks,
    /// The layer scales its value rows: the selected row's append applies no
    /// multiplier.
    ValueScale,
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
            Refusal::ValueWidth {
                head,
                value,
                served,
            } => write!(
                f,
                "no flash is built for a value width of {value} at a head of {head}; \
                 values {served:?} are"
            ),
            Refusal::Window { positions } => {
                write!(f, "this flash cuts no window of {positions} positions")
            }
            Refusal::Sinks => {
                f.write_str("this flash folds no per-head softmax sinks into the softmax")
            }
            Refusal::ValueScale => f.write_str("this flash's append applies no value multiplier"),
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
    /// Sigmoid scores, a selection bias steering the choice only, the kept
    /// unbiased scores renormalized: GLM-5.3's and MiMo-V2's noaux_tc router.
    pub const BIASED_SIGMOID: RouterRule = super::BIASED_SIGMOID;
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
    /// Key and score values per head.
    pub head: u32,
    /// Value values per head: a row serves the width it is built for.
    pub value: u32,
    /// The positions attended; `None`: every position.
    pub window: Option<u32>,
    /// Per-head softmax sinks.
    pub sinks: bool,
    /// The value rows carry a multiplier.
    pub scaled: bool,
}

impl AttnShape {
    /// The shape of a GQA mixer, as its description states it.
    #[must_use]
    pub fn of(g: &Gqa) -> AttnShape {
        AttnShape {
            n_head: g.heads,
            n_kv: g.kv_heads,
            head: g.head_dim,
            value: g.value_dim,
            window: g.window,
            sinks: g.sinks,
            scaled: g.value_scale.is_some(),
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
    /// Key and score values per head: the key tile's width.
    pub head: u32,
    /// Value values per head: the value tile's and the register vectors'
    /// width.
    pub value: u32,
    /// Query heads one block holds.
    pub pack: u32,
    pub group: GroupRule,
    /// The instance cuts a window of the positions (a launch argument).
    pub window: bool,
    /// The instance folds per-head softmax sinks into the softmax.
    pub sinks: bool,
    /// The instance's append multiplies the value rows by the layer's scale.
    pub value_scale: bool,
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
        value: 128,
        pack: 8,
        group: GroupRule::One,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs HEAD, GROUP",
    },
    GqaInst {
        head: 256,
        value: 256,
        pack: 8,
        group: GroupRule::One,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs HEAD_256, GROUP (the _256 entries)",
    },
    GqaInst {
        head: 256,
        value: 256,
        pack: 4,
        group: GroupRule::Packs,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs PACK_4 (the _256_p4 entries)",
    },
    GqaInst {
        head: 256,
        value: 256,
        pack: 2,
        group: GroupRule::Packs,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs PACK_2 (the _256_p2 entries, scalar pass only)",
    },
    GqaInst {
        head: 192,
        value: 128,
        pack: 8,
        group: GroupRule::Packs,
        window: true,
        sinks: true,
        value_scale: true,
        at: "gpu/src/flash_gqa.rs HEAD_K192 (the _k192 entries, scalar pass only; \
             gpu/src/rope_neox.rs neox_append_k192)",
    },
];

/// The first GQA row, in [`GQA`]'s order, built for `s.head` and `s.value`
/// that takes its group and serves its window, sinks and value scale.
///
/// # Errors
///
/// [`ShapeRefused`] naming `s` when the heads do not group, no row has its
/// head width, none of those has its value width, none of those takes its
/// group, or the selected row does not cut its window, fold its sinks or
/// apply its value scale.
pub fn select_gqa(s: AttnShape) -> Result<GqaInst, ShapeRefused> {
    let refuse = |why| ShapeRefused {
        shape: Shape::Attn(s),
        why,
    };
    if s.n_kv == 0 || s.n_head == 0 || !s.n_head.is_multiple_of(s.n_kv) {
        return Err(refuse(Refusal::Group));
    }
    let group = s.n_head / s.n_kv;
    let head: Vec<&GqaInst> = GQA.iter().filter(|r| r.head == s.head).collect();
    if head.is_empty() {
        let mut served: Vec<u32> = GQA.iter().map(|r| r.head).collect();
        served.sort_unstable();
        served.dedup();
        return Err(refuse(Refusal::HeadWidth { served }));
    }
    let width: Vec<&&GqaInst> = head.iter().filter(|r| r.value == s.value).collect();
    if width.is_empty() {
        let mut served: Vec<u32> = head.iter().map(|r| r.value).collect();
        served.sort_unstable();
        served.dedup();
        return Err(refuse(Refusal::ValueWidth {
            head: s.head,
            value: s.value,
            served,
        }));
    }
    let Some(row) = gqa_row_v(s.head, s.value, group) else {
        return Err(refuse(Refusal::GqaGroup {
            group,
            served: width
                .iter()
                .map(|r| match r.group {
                    GroupRule::One => format!("exactly {}", r.pack),
                    GroupRule::Packs => format!("a multiple of {}", r.pack),
                })
                .collect(),
        }));
    };
    if let Some(positions) = s.window
        && !row.window
    {
        return Err(refuse(Refusal::Window { positions }));
    }
    if s.sinks && !row.sinks {
        return Err(refuse(Refusal::Sinks));
    }
    if s.scaled && !row.value_scale {
        return Err(refuse(Refusal::ValueScale));
    }
    Ok(row)
}

/// The first row, in [`GQA`]'s order, built for `head` key values and `value`
/// value values that takes `group` query heads a key head.
#[must_use]
pub fn gqa_row_v(head: u32, value: u32, group: u32) -> Option<GqaInst> {
    GQA.iter()
        .find(|r| r.head == head && r.value == value && r.takes(group))
        .copied()
}

/// [`gqa_row_v`] at a value width equal to the head's: the rows the
/// Qwen bodies launch.
#[must_use]
pub fn gqa_row(head: u32, group: u32) -> Option<GqaInst> {
    gqa_row_v(head, head, group)
}

#[cfg(test)]
mod tests {
    use super::{
        AttnShape, GQA, GroupRule, LANE_PICKS, MoeShape, ROUTERS, Refusal, RouterRule, Shape,
        rules, select_gqa, select_router,
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
    /// experts, top 8, with a selection bias") is the `Glm5next` row, and a
    /// sigmoid rule at another width is still refused by name.
    #[test]
    fn glm_5_3_flash_router_is_its_row() {
        let rule = RouterRule {
            score: Score::Sigmoid,
            bias: true,
            norm: true,
            gated: false,
        };
        let r = select_router(moe(rule, 288, 8)).expect("GLM-5.3-Flash has no router row");
        assert_eq!(r.body, super::RouterBody::Glm5next);
        let e = select_router(moe(rule, 320, 8)).expect_err("a sigmoid router at 320 was selected");
        assert!(matches!(e.why, Refusal::RouterWidth { .. }), "{e}");
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
            (moe(rules::SOFTMAX_NORM_GATED, 256, 0), 1, LANE_PICKS),
            (
                moe(rules::SOFTMAX_NORM_GATED, 256, LANE_PICKS + 1),
                1,
                LANE_PICKS,
            ),
            (moe(rules::SOFTMAX_NORM, 128, 33), 1, LANE_PICKS),
            (moe(rules::SOFTMAX_TOPK, 64, 8), 6, 6),
            (moe(rules::BIASED_SQRT_SOFTPLUS, 384, 8), 6, 6),
            (moe(rules::BIASED_SQRT_SOFTPLUS, 128, 6), 3, 3),
        ] {
            let e = select_router(s).expect_err("selected outside the row");
            assert_eq!(e.why, Refusal::TopK { min, max }, "{s:?}");
        }
        // Every count of a lane-held row selects.
        for k in 1..=LANE_PICKS {
            assert!(select_router(moe(rules::SOFTMAX_NORM_GATED, 256, k)).is_ok());
            assert!(select_router(moe(rules::SOFTMAX_NORM_GATED, 512, k)).is_ok());
            assert!(select_router(moe(rules::SOFTMAX_NORM, 128, k)).is_ok());
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
        AttnShape {
            n_head,
            n_kv,
            head,
            value: head,
            window: None,
            sinks: false,
            scaled: false,
        }
    }

    /// Qwen3 32/4 × 128 (qwen3moe_meta.rs:409), Qwen3.6 16/2 × 256
    /// (qwen35moe_meta.rs:34), Qwen3.8 24/2 × 256 (q38gqa, 9e5b716),
    /// Qwen3.5-9B 24/4 × 256 (the Clef backbone's config).
    #[test]
    fn every_attention_the_tree_runs_selects_its_flash() {
        for (name, s, head, pack, packs) in [
            ("Qwen3", attn(32, 4, 128), 128, 8, 1),
            ("Qwen3.6", attn(16, 2, 256), 256, 8, 1),
            ("Qwen3.8", attn(24, 2, 256), 256, 4, 3),
            ("Qwen3.5-9B", attn(24, 4, 256), 256, 2, 3),
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
        let e = select_gqa(attn(16, 2, 160)).expect_err("head 160");
        assert_eq!(
            e.why,
            Refusal::HeadWidth {
                served: vec![128, 192, 256]
            }
        );
        for s in [attn(24, 5, 256), attn(24, 0, 256), attn(0, 2, 256)] {
            assert_eq!(select_gqa(s).expect_err("ungrouped").why, Refusal::Group);
        }
        let e = select_gqa(attn(24, 8, 256)).expect_err("group 3");
        assert!(e.to_string().contains("group 3"), "{e}");
    }

    fn mimo(n_head: u32, n_kv: u32, window: Option<u32>, sinks: bool) -> AttnShape {
        AttnShape {
            n_head,
            n_kv,
            head: 192,
            value: 128,
            window,
            sinks,
            scaled: true,
        }
    }

    /// MiMo-V2.6-Flash (the GGUF header: `key_length` 192, `value_length` 128,
    /// 64 query heads, `head_count_kv` 4 on the nine full layers and 8 on the
    /// sliding ones, `sliding_window` 128, `attn_sinks` on the sliding layers,
    /// `value_scale` 0.707) selects the K192 row: packs 2 on a full layer,
    /// packs 1 on a sliding one.
    #[test]
    fn mimo_v2_selects_the_k192_row() {
        for (name, s, packs) in [
            ("full", mimo(64, 4, None, false), 2),
            ("sliding", mimo(64, 8, Some(128), true), 1),
        ] {
            let row = select_gqa(s).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((row.head, row.value, row.pack), (192, 128, 8), "{name}");
            assert_eq!(row.packs(s.n_head / s.n_kv), packs, "{name}");
        }
    }

    /// A pair of widths no row holds is refused by name, naming the value
    /// widths the head's rows carry: the K192 row's head with the Qwen value
    /// width, and the row's value width at a head it is not built for.
    #[test]
    fn a_pair_of_widths_no_row_holds_is_refused_by_name() {
        // PIN(2026-10-07): the K192 row serves 192/128, so the pair pinned here is 192/192.
        let s = AttnShape {
            head: 192,
            value: 192,
            ..mimo(64, 4, None, false)
        };
        let e = select_gqa(s).expect_err("K 192 V 192");
        assert_eq!(
            e.why,
            Refusal::ValueWidth {
                head: 192,
                value: 192,
                served: vec![128]
            }
        );
        assert!(
            e.to_string()
                .contains("no flash is built for a value width of 192 at a head of 192"),
            "{e}"
        );
        assert_eq!(e.shape, Shape::Attn(s));
        let s = AttnShape {
            head: 128,
            value: 192,
            ..attn(32, 4, 128)
        };
        assert_eq!(
            select_gqa(s).expect_err("K 128 V 192").why,
            Refusal::ValueWidth {
                head: 128,
                value: 192,
                served: vec![128]
            }
        );
    }

    /// The K192 row takes groups that are multiples of 8 and refuses others
    /// by name; a window, sinks or value scale it does not serve are refused
    /// by the row that was selected.
    #[test]
    fn the_k192_row_takes_its_groups_only() {
        let e = select_gqa(mimo(48, 4, None, false)).expect_err("group 12");
        assert_eq!(
            e.why,
            Refusal::GqaGroup {
                group: 12,
                served: vec!["a multiple of 8".to_string()]
            }
        );
        assert!(select_gqa(mimo(64, 4, None, false)).is_ok());
        assert!(select_gqa(mimo(8, 1, None, false)).is_ok());
        assert_eq!(
            select_gqa(mimo(40, 8, None, false))
                .expect_err("group 5")
                .why,
            Refusal::GqaGroup {
                group: 5,
                served: vec!["a multiple of 8".to_string()]
            }
        );
    }

    /// A window, sinks or a value scale on a shape every flash row serves is
    /// refused by name: the row's kernels run none of the three.
    #[test]
    fn a_window_sinks_or_value_scale_is_refused_by_name() {
        let base = attn(32, 4, 128);
        assert!(select_gqa(base).is_ok());
        let windowed = AttnShape {
            window: Some(128),
            ..base
        };
        let sunk = AttnShape {
            sinks: true,
            ..base
        };
        let scaled = AttnShape {
            scaled: true,
            ..base
        };
        assert_eq!(
            select_gqa(windowed).expect_err("a window").why,
            Refusal::Window { positions: 128 }
        );
        assert_eq!(select_gqa(sunk).expect_err("sinks").why, Refusal::Sinks);
        assert_eq!(
            select_gqa(scaled).expect_err("a value scale").why,
            Refusal::ValueScale
        );
        let e = select_gqa(windowed).expect_err("a window");
        assert!(e.to_string().contains("window of 128 positions"), "{e}");
    }

    /// GLM-5.3's biased sigmoid rule is the shared one the module exports:
    /// the noaux_tc router (sigmoid, the bias steering the choice only, the
    /// kept unbiased scores renormalized) MiMo-V2 runs too.
    #[test]
    fn the_biased_sigmoid_rule_is_glm_5_3_flashs() {
        let rule = RouterRule {
            score: Score::Sigmoid,
            bias: true,
            norm: true,
            gated: false,
        };
        assert_eq!(rules::BIASED_SIGMOID, rule);
        assert!(select_router(moe(rule, 288, 8)).is_ok());
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
