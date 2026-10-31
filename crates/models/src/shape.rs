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

use crate::{Gqa, Moe, RouterNorm, Score};

/// How a router turns logits into ids and weights: the numeric contract a
/// router body is written for, apart from its width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterRule {
    pub score: Score,
    /// A selection bias steers the choice only.
    pub bias: bool,
    /// The divisor policy on the kept, unbiased scores before scaling.
    pub norm: RouterNorm,
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
    /// `gpu/src/route_softmax.rs` `softmax_router_*`: a warp a token,
    /// four named logits a lane.
    Softmax,
    /// `gpu/src/arch/qwen3moe/router.rs` `qwen35moe_router_*`: a warp a token,
    /// the logits in shared memory, the shared expert's gate as the last row.
    Qwen35moe,
    /// `gpu-deepseek41/src/router.rs` `ds41_router*`.
    Ds41,
    /// `gpu-deepseek41/src/experts_mxfp4.rs` `dflash_router`: `ds41_router`'s
    /// body with `norm` a launch argument.
    Dflash,
    /// `gpu/src/route_body.rs` `sigmoid_router_instance!`: sigmoid scores,
    /// the bias added for the pick only ([`rules::BIASED_SIGMOID`]). Its
    /// instances: `gpu-deepseek41/src/router.rs` `glm5next_router*`,
    /// `mimo2_router*`, `sigmoid_320_8_router*`, `sigmoid_128_4_router*`;
    /// `gpu/src/arch/nemotron_h_moe/router.rs` `nemotron_moe_router_*`.
    BiasedSigmoid,
    /// `gpu/src/route_body.rs` `sigmoid_floor_256_6_router*`: sigmoid,
    /// optional selection-only bias and a fixed `2^-14` selected-sum floor.
    SigmoidFloor,
}

impl RouterBody {
    /// Whether the body routes a token in one warp of a softmax top-k
    /// routing: `gpu/src/route_softmax.rs`'s own entries and the gated
    /// shared-expert row built on them. The chain's router dims serve only
    /// these bodies.
    #[must_use]
    pub const fn is_warp_softmax(self) -> bool {
        matches!(self, RouterBody::Softmax | RouterBody::Qwen35moe)
    }
}

/// One compiled router instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterInst {
    pub body: RouterBody,
    pub rule: RouterRule,
    /// The body takes `norm` as a launch argument: it serves
    /// [`RouterNorm::None`] and [`RouterNorm::SelectedSum`] only.
    pub norm_arg: bool,
    /// Experts the instance routes over. Lane `L` of the selecting warp owns
    /// experts `L + 32 j` below this count ([`RouterInst::per_lane`] of them
    /// at most).
    pub experts: u32,
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
        self.experts
    }

    /// Experts the selecting warp's lane 0 owns: `experts / 32` rounded up.
    #[must_use]
    pub const fn per_lane(&self) -> u32 {
        self.experts.div_ceil(32)
    }

    /// Whether this row's body runs `rule`.
    #[must_use]
    pub const fn runs(&self, rule: RouterRule) -> bool {
        score_eq(self.rule.score, rule.score)
            && self.rule.bias == rule.bias
            && self.rule.gated == rule.gated
            && if self.norm_arg {
                matches!(rule.norm, RouterNorm::None | RouterNorm::SelectedSum)
            } else {
                norm_eq(self.rule.norm, rule.norm)
            }
    }
}

/// The largest pick count a lane-held selection takes: pick `s` lives in
/// lane `s` of the routing warp.
pub const LANE_PICKS: u32 = 32;

const SOFTMAX_TOPK: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: RouterNorm::None,
    gated: false,
};
const SOFTMAX_NORM: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: RouterNorm::SelectedSum,
    gated: false,
};
const SOFTMAX_NORM_GATED: RouterRule = RouterRule {
    score: Score::Softmax,
    bias: false,
    norm: RouterNorm::SelectedSum,
    gated: true,
};
const BIASED_SIGMOID: RouterRule = RouterRule {
    score: Score::Sigmoid,
    bias: true,
    norm: RouterNorm::SelectedSum,
    gated: false,
};
const SIGMOID_FLOOR: RouterRule = RouterRule {
    score: Score::Sigmoid,
    bias: false,
    norm: RouterNorm::SelectedSumFloor,
    gated: false,
};
const BIASED_SIGMOID_FLOOR: RouterRule = RouterRule {
    bias: true,
    ..SIGMOID_FLOOR
};
const BIASED_SQRT_SOFTPLUS: RouterRule = RouterRule {
    score: Score::SqrtSoftplus,
    bias: true,
    norm: RouterNorm::SelectedSum,
    gated: false,
};

/// Every compiled router instance. A body's rows share its rule; a width
/// is a row only while its entries exist.
pub const ROUTERS: &[RouterInst] = &[
    RouterInst {
        body: RouterBody::Topk,
        rule: SOFTMAX_TOPK,
        norm_arg: false,
        experts: 64,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/router.rs router_topk",
    },
    RouterInst {
        body: RouterBody::Softmax,
        rule: SOFTMAX_NORM,
        norm_arg: false,
        experts: 128,
        top_k_min: 1,
        top_k_max: LANE_PICKS,
        at: "gpu/src/route_softmax.rs softmax_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        experts: 256,
        top_k_min: 1,
        top_k_max: LANE_PICKS,
        at: "gpu/src/arch/qwen3moe/router.rs qwen35moe_router_*",
    },
    RouterInst {
        body: RouterBody::Qwen35moe,
        rule: SOFTMAX_NORM_GATED,
        norm_arg: false,
        experts: 512,
        top_k_min: 1,
        top_k_max: LANE_PICKS,
        at: "gpu/src/arch/qwen3moe/router.rs qwen35moe_router_*_512",
    },
    RouterInst {
        body: RouterBody::Ds41,
        rule: BIASED_SQRT_SOFTPLUS,
        norm_arg: false,
        experts: 384,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu-deepseek41/src/router.rs ds41_router*",
    },
    RouterInst {
        body: RouterBody::Dflash,
        rule: BIASED_SQRT_SOFTPLUS,
        norm_arg: true,
        experts: 128,
        top_k_min: 3,
        top_k_max: 3,
        at: "gpu-deepseek41/src/experts_mxfp4.rs dflash_router",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 288,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu-deepseek41/src/router.rs glm5next_router*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 256,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu-deepseek41/src/router.rs mimo2_router*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 240,
        top_k_min: 9,
        top_k_max: 9,
        at: "gpu/src/arch/nemotron_h_moe/router.rs nemotron_moe_router_240*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 304,
        top_k_min: 9,
        top_k_max: 9,
        at: "gpu/src/arch/nemotron_h_moe/router.rs nemotron_moe_router_304*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 368,
        top_k_min: 9,
        top_k_max: 9,
        at: "gpu/src/arch/nemotron_h_moe/router.rs nemotron_moe_router_368*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 432,
        top_k_min: 9,
        top_k_max: 9,
        at: "gpu/src/arch/nemotron_h_moe/router.rs nemotron_moe_router_432*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 496,
        top_k_min: 9,
        top_k_max: 9,
        at: "gpu/src/arch/nemotron_h_moe/router.rs nemotron_moe_router_496*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 320,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu-deepseek41/src/router.rs sigmoid_320_8_router*",
    },
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 128,
        top_k_min: 4,
        top_k_max: 4,
        at: "gpu-deepseek41/src/router.rs sigmoid_128_4_router*",
    },
    // Raon-1's router: the second pick count at 128 experts.
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 128,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/route_body.rs sigmoid_128_6_router*",
    },
    // 256 experts, top 6. After MiMo's 256/top-8 row: `router_row` at 256
    // stays that row, and the pick ranges 8..=8 and 6..=6 do not overlap.
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 256,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/route_body.rs sigmoid_256_6_router*",
    },
    RouterInst {
        body: RouterBody::SigmoidFloor,
        rule: SIGMOID_FLOOR,
        norm_arg: false,
        experts: 256,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/route_body.rs sigmoid_floor_256_6_router*",
    },
    RouterInst {
        body: RouterBody::SigmoidFloor,
        rule: BIASED_SIGMOID_FLOOR,
        norm_arg: false,
        experts: 256,
        top_k_min: 6,
        top_k_max: 6,
        at: "gpu/src/route_body.rs sigmoid_floor_256_6_router*",
    },
    // Muse Spark's fixture: 16 experts, the real file's top 8.
    RouterInst {
        body: RouterBody::BiasedSigmoid,
        rule: BIASED_SIGMOID,
        norm_arg: false,
        experts: 16,
        top_k_min: 8,
        top_k_max: 8,
        at: "gpu/src/route_body.rs sigmoid_16_8_router*",
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
                    match r.norm {
                        RouterNorm::None => "",
                        RouterNorm::SelectedSum => " renormalized",
                        RouterNorm::SelectedSumFloor => " normalized by max(selected sum, 2^-14)",
                    },
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
    let width: Vec<&&RouterInst> = rule.iter().filter(|r| r.experts() == s.experts).collect();
    let Some(first) = width.first() else {
        let mut served: Vec<u32> = Vec::new();
        for e in rule.iter().map(|r| r.experts()) {
            if !served.contains(&e) {
                served.push(e);
            }
        }
        return Err(refuse(Refusal::RouterWidth { served }));
    };
    // A width may hold one row a pick count its instances compile.
    let holds = |r: &RouterInst| (r.top_k_min..=r.top_k_max.min(r.experts())).contains(&s.top_k);
    match width.iter().find(|r| holds(r)) {
        Some(row) => Ok(***row),
        None => Err(refuse(Refusal::TopK {
            min: first.top_k_min,
            max: first.top_k_max.min(first.experts()),
        })),
    }
}

/// Body `body`'s row at `experts`: the const form a device crate holds
/// each width it compiles to. Fails the build when the table has no such
/// row.
#[must_use]
pub const fn router_row(body: RouterBody, experts: u32) -> RouterInst {
    let mut i = 0;
    while i < ROUTERS.len() {
        let r = ROUTERS[i];
        if body_eq(r.body, body) && r.experts == experts {
            return r;
        }
        i += 1;
    }
    panic!("no router row for this body at this width")
}

/// Body `body`'s row at `experts` whose pick range holds `top_k`: the const
/// form a device crate holds each (width, pick count) it compiles to, for a
/// body whose instances fix the pick count. Fails the build when the table
/// has no such row.
#[must_use]
pub const fn router_row_k(body: RouterBody, experts: u32, top_k: u32) -> RouterInst {
    let mut i = 0;
    while i < ROUTERS.len() {
        let r = ROUTERS[i];
        if body_eq(r.body, body)
            && r.experts == experts
            && r.top_k_min <= top_k
            && top_k <= r.top_k_max
        {
            return r;
        }
        i += 1;
    }
    panic!("no router row for this body at this width and pick count")
}

/// A compiled binding whose body, rule, expert count and pick count all
/// agree. Unlike [`router_row_k`], distinguishes optional-bias bindings.
#[must_use]
pub const fn router_row_rule(
    body: RouterBody,
    rule: RouterRule,
    experts: u32,
    top_k: u32,
) -> RouterInst {
    let mut i = 0;
    while i < ROUTERS.len() {
        let r = ROUTERS[i];
        if body_eq(r.body, body)
            && r.runs(rule)
            && r.experts == experts
            && r.top_k_min <= top_k
            && top_k <= r.top_k_max
        {
            return r;
        }
        i += 1;
    }
    panic!("no router row for this body, rule, width and pick count")
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
        && norm_eq(r.rule.norm, rule.norm)
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
    /// Sigmoid, no selection bias, selected scores divided by
    /// `max(sum_selected, 2^-14)`.
    pub const SIGMOID_FLOOR: RouterRule = super::SIGMOID_FLOOR;
    /// [`SIGMOID_FLOOR`] with a bias steering selection only.
    pub const BIASED_SIGMOID_FLOOR: RouterRule = super::BIASED_SIGMOID_FLOOR;
}

const fn norm_eq(a: RouterNorm, b: RouterNorm) -> bool {
    matches!(
        (a, b),
        (RouterNorm::None, RouterNorm::None)
            | (RouterNorm::SelectedSum, RouterNorm::SelectedSum)
            | (RouterNorm::SelectedSumFloor, RouterNorm::SelectedSumFloor)
    )
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
/// them: the first row of the head and value width that takes the group and
/// covers the request. A one-block-a-group row stays ahead of a packed row
/// of the same width, and a row that cuts no window stays ahead of the
/// window-and-sinks row of the same pack so a shape that asks for neither
/// keeps the row it had.
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
    // Head 128, value 128, one block of eight, with a window and sinks.
    // After the row above: a shape that asks for neither still takes that
    // row (`gqa_row` is the first match). Full layers launch window 0.
    GqaInst {
        head: 128,
        value: 128,
        pack: 8,
        group: GroupRule::One,
        window: true,
        sinks: true,
        value_scale: false,
        at: "gpu/src/flash_gqa_h128_swa.rs gqa_flash_seg_h128_swa",
    },
    // Head 128, value 128, one block of four, with a window and no sinks: the
    // head-128 flash at a group of four (`gpu/src/attention_h128.rs`'s
    // binding, plane and ring). Full layers launch window 0, so the row covers
    // a layer with a window or without one. It claims no sinks: no layer of
    // this group has them.
    GqaInst {
        head: 128,
        value: 128,
        pack: 4,
        group: GroupRule::One,
        window: true,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa_h128_swa.rs gqa_flash_seg_h128_swa_g4 (the ring: \
             gpu/src/attention_h128_ring.rs gqa_flash_seg_h128_swa_ring_g4)",
    },
    GqaInst {
        head: 128,
        value: 128,
        pack: 5,
        group: GroupRule::One,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs GROUP_5 (gqa_flash_seg_g5, scalar)",
    },
    GqaInst {
        head: 128,
        value: 128,
        pack: 7,
        group: GroupRule::One,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs GROUP_7 (gqa_flash_seg_mma_g7 and its q8 twin, tensor-core pass \
             only; gpu/src/flash_gqa_prefill.rs gqa_prefill_flash_g7)",
    },
    GqaInst {
        head: 128,
        value: 128,
        pack: 16,
        group: GroupRule::One,
        window: false,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs GROUP_16 (two GROUP segment passes a key head, one merge)",
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
    // Muse Spark's head 64: the `_64` entries take a window a layer's pass
    // names (0 for the full layers' no-window rows), no sinks, no value
    // scale. A row that asks for none of those still takes this one
    // (`gqa_row` is the first match).
    GqaInst {
        head: 64,
        value: 64,
        pack: 8,
        group: GroupRule::One,
        window: true,
        sinks: false,
        value_scale: false,
        at: "gpu/src/flash_gqa.rs HEAD_64, GROUP (the _64 entries)",
    },
];

/// The first GQA row, in [`GQA`]'s order, built for `s.head` and `s.value`
/// that takes its group and covers its window, sinks and value scale.
///
/// A row covers a request when it can do everything the request asks. It
/// may also be able to do more: a window-and-sinks row covers a layer that
/// asks for sinks and no window (the launch passes window 0).
///
/// # Errors
///
/// [`ShapeRefused`] naming `s` when the heads do not group, no row has its
/// head width, none of those has its value width, or none of those takes
/// its group. When a row takes the group and none of them covers the
/// request, the refusal is the first such row's, in order: its window, then
/// its sinks, then its value scale.
pub fn select_gqa(s: AttnShape) -> Result<GqaInst, ShapeRefused> {
    select_gqa_in(GQA, s)
}

/// [`select_gqa`] over `table`: a backend that compiles its own flash rows
/// selects from them, and a refusal lists the widths and groups that table
/// serves. `table` is in the order of preference, as [`GQA`] is.
///
/// # Errors
///
/// [`ShapeRefused`] as [`select_gqa`]'s, for `table`'s rows.
pub fn select_gqa_in(table: &[GqaInst], s: AttnShape) -> Result<GqaInst, ShapeRefused> {
    let refuse = |why| ShapeRefused {
        shape: Shape::Attn(s),
        why,
    };
    if s.n_kv == 0 || s.n_head == 0 || !s.n_head.is_multiple_of(s.n_kv) {
        return Err(refuse(Refusal::Group));
    }
    let group = s.n_head / s.n_kv;
    let head: Vec<&GqaInst> = table.iter().filter(|r| r.head == s.head).collect();
    if head.is_empty() {
        let mut served: Vec<u32> = table.iter().map(|r| r.head).collect();
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
    // Capability is "can", so the first group match is not always the row:
    // the head-128 pack of eight that cuts no window is first, and a later
    // row of the same pack covers a window or sinks.
    let covers = |r: &GqaInst| {
        (s.window.is_none() || r.window) && (!s.sinks || r.sinks) && (!s.scaled || r.value_scale)
    };
    if let Some(row) = width.iter().find(|r| r.takes(group) && covers(r)) {
        return Ok(***row);
    }
    let Some(row) = width.iter().find(|r| r.takes(group)) else {
        let mut served = Vec::new();
        for r in &width {
            let text = match r.group {
                GroupRule::One => format!("exactly {}", r.pack),
                GroupRule::Packs => format!("a multiple of {}", r.pack),
            };
            if !served.contains(&text) {
                served.push(text);
            }
        }
        return Err(refuse(Refusal::GqaGroup { group, served }));
    };
    if let Some(positions) = s.window
        && !row.window
    {
        return Err(refuse(Refusal::Window { positions }));
    }
    if s.sinks && !row.sinks {
        return Err(refuse(Refusal::Sinks));
    }
    // The window and the sinks were not the miss, and no row covered the
    // request, so the miss is the value scale.
    if !s.scaled || row.value_scale {
        unreachable!("select_gqa_in: a covering row of this group was not returned");
    }
    Err(refuse(Refusal::ValueScale))
}

/// The first row, in [`GQA`]'s order, built for `head` key values and `value`
/// value values that takes `group` query heads a key head.
#[must_use]
pub const fn gqa_row_v(head: u32, value: u32, group: u32) -> Option<GqaInst> {
    let mut i = 0;
    while i < GQA.len() {
        let r = GQA[i];
        if r.head == head && r.value == value && r.takes(group) {
            return Some(r);
        }
        i += 1;
    }
    None
}

/// [`gqa_row_v`] at a value width equal to the head's: the rows the
/// Qwen bodies launch.
#[must_use]
pub const fn gqa_row(head: u32, group: u32) -> Option<GqaInst> {
    gqa_row_v(head, head, group)
}

#[cfg(test)]
mod tests {
    use super::{
        AttnShape, GQA, GqaInst, GroupRule, LANE_PICKS, MoeShape, ROUTERS, Refusal, RouterBody,
        RouterInst, RouterRule, Shape, gqa_row_v, router_row, router_row_is, router_row_k, rules,
        select_gqa, select_gqa_in, select_router,
    };
    use crate::{Act, Gqa, Moe, OutGate, Rope, RopeMode, Router, RouterNorm, Score, Shared};

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
            // model/tests/nemotron_h_moe_meta.rs SHAPES: Bolt's MoE layers route
            // over 240, 304, 368, 432 or 496 experts, top 9, sigmoid with a bias.
            ("Bolt 240", moe(rules::BIASED_SIGMOID, 240, 9)),
            ("Bolt 304", moe(rules::BIASED_SIGMOID, 304, 9)),
            ("Bolt 368", moe(rules::BIASED_SIGMOID, 368, 9)),
            ("Bolt 432", moe(rules::BIASED_SIGMOID, 432, 9)),
            ("Bolt 496", moe(rules::BIASED_SIGMOID, 496, 9)),
            // Raon-1's header: expert_count 128, expert_used_count 6, gating
            // sigmoid, exp_probs_b, weights_norm true.
            ("Raon-1", moe(rules::BIASED_SIGMOID, 128, 6)),
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
        off.norm = RouterNorm::None;
        assert!(select_router(moe(off, 128, 3)).is_ok());
        let e = select_router(moe(off, 384, 6)).expect_err("V4.1 at norm off");
        assert!(matches!(e.why, Refusal::RouterWidth { ref served } if served == &[128]));
    }

    /// Legacy rows retain their epsilon-added or unnormalized arithmetic;
    /// only the real floor binding accepts the floor policy.
    #[test]
    fn floor_bindings_are_distinct_from_legacy_policies() {
        for row in ROUTERS {
            assert_eq!(
                select_router(moe(row.rule, row.experts(), row.top_k_min)).unwrap(),
                *row
            );
            if row.body == RouterBody::SigmoidFloor {
                assert_eq!(row.rule.norm, RouterNorm::SelectedSumFloor);
                assert!(!row.norm_arg);
            } else {
                assert!(!row.runs(RouterRule {
                    norm: RouterNorm::SelectedSumFloor,
                    ..row.rule
                }));
            }
        }
    }

    #[test]
    fn floor_top_six_serves_both_bias_modes_and_only_its_shape() {
        for bias in [false, true] {
            let rule = RouterRule {
                bias,
                ..rules::SIGMOID_FLOOR
            };
            let row = select_router(moe(rule, 256, 6)).unwrap();
            assert_eq!(row.body, RouterBody::SigmoidFloor);
            assert_eq!(row.rule, rule);
            for top_k in [0, 5, 7, 8, 257] {
                assert_eq!(
                    select_router(moe(rule, 256, top_k)).unwrap_err().why,
                    Refusal::TopK { min: 6, max: 6 }
                );
            }
            assert!(matches!(
                select_router(moe(rule, 128, 6)).unwrap_err().why,
                Refusal::RouterWidth { ref served } if served == &[256]
            ));
            for (score, gated) in [(Score::Softmax, false), (Score::Sigmoid, true)] {
                let wrong = RouterRule {
                    score,
                    gated,
                    ..rule
                };
                assert_eq!(
                    select_router(moe(wrong, 256, 6)).unwrap_err().why,
                    Refusal::NoRouterRule
                );
            }
        }
        let legacy = select_router(moe(rules::BIASED_SIGMOID, 256, 6)).unwrap();
        assert_eq!(legacy.body, RouterBody::BiasedSigmoid);
        assert_eq!(
            select_router(moe(
                RouterRule {
                    norm: RouterNorm::SelectedSum,
                    ..rules::SIGMOID_FLOOR
                },
                256,
                6
            ))
            .unwrap_err()
            .why,
            Refusal::NoRouterRule
        );
    }

    /// GLM-5.3-Flash (model/tests/glm5next_meta.rs:83, "router: sigmoid, 288
    /// experts, top 8, with a selection bias") is the `BiasedSigmoid` row at
    /// 288, and a sigmoid rule at another width is still refused by name.
    #[test]
    fn glm_5_3_flash_router_is_its_row() {
        let rule = RouterRule {
            score: Score::Sigmoid,
            bias: true,
            norm: RouterNorm::SelectedSum,
            gated: false,
        };
        let r = select_router(moe(rule, 288, 8)).expect("GLM-5.3-Flash has no router row");
        assert_eq!(r.body, super::RouterBody::BiasedSigmoid);
        let e = select_router(moe(rule, 352, 8)).expect_err("a sigmoid router at 352 was selected");
        assert!(matches!(e.why, Refusal::RouterWidth { .. }), "{e}");
    }

    /// MiMo-V2.6-Flash (256 experts, top 8, sigmoid with a selection bias,
    /// renormalized) is the `BiasedSigmoid` row at 256, beside GLM's at 288:
    /// both rows resolve through `select_router`, `router_row` and the
    /// body-name test coverage's `routes` applies, each to its own row.
    #[test]
    fn mimo_v2_router_is_its_row_beside_glms() {
        let rule = rules::BIASED_SIGMOID;
        let mimo = select_router(moe(rule, 256, 8)).expect("MiMo has no router row");
        let glm = select_router(moe(rule, 288, 8)).expect("GLM has no router row");
        assert_eq!(mimo.body, super::RouterBody::BiasedSigmoid);
        assert_eq!(glm.body, super::RouterBody::BiasedSigmoid);
        assert_eq!((mimo.experts(), mimo.per_lane()), (256, 8));
        assert_eq!((glm.experts(), glm.per_lane()), (288, 9));
        assert!(mimo.at.contains("mimo2_router"), "{}", mimo.at);
        assert!(glm.at.contains("glm5next_router"), "{}", glm.at);
        // The const form a device crate holds each width by.
        assert_eq!(router_row(super::RouterBody::BiasedSigmoid, 256), mimo);
        assert_eq!(router_row(super::RouterBody::BiasedSigmoid, 288), glm);
        assert!(router_row_is(mimo, rule, false, (8, 8)));
        assert!(router_row_is(glm, rule, false, (8, 8)));
        // `body_eq` tells the rule's body from every other body, and only it.
        for r in ROUTERS {
            assert_eq!(
                super::body_eq(r.body, super::RouterBody::BiasedSigmoid),
                r.at.contains("mimo2_router")
                    || r.at.contains("glm5next_router")
                    || r.at.contains("sigmoid_320_8_router")
                    || r.at.contains("sigmoid_128_4_router")
                    || r.at.contains("sigmoid_128_6_router")
                    || r.at.contains("sigmoid_256_6_router")
                    || r.at.contains("sigmoid_16_8_router")
                    || r.at.contains("nemotron_moe_router"),
                "{}",
                r.at
            );
        }
        // The rows that carry the body: GLM's and MiMo's, then Bolt's five, then
        // the 320-expert instance's, then the two 128-expert ones (top 4, top 6),
        // then the 256-expert top-6 row.
        let widths: Vec<u32> = ROUTERS
            .iter()
            .filter(|r| super::body_eq(r.body, super::RouterBody::BiasedSigmoid))
            .map(RouterInst::experts)
            .collect();
        assert_eq!(
            widths,
            [288, 256, 240, 304, 368, 432, 496, 320, 128, 128, 256, 16]
        );
        // Coverage's `routes(n, &[BiasedSigmoid])`: the rule's shapes pass at
        // both widths, and a width between them is refused. 256 at top 6 is
        // its own row; top 7 matches neither pick range.
        let routes = |s| select_router(s).is_ok_and(|r| r.body == super::RouterBody::BiasedSigmoid);
        assert!(routes(moe(rule, 256, 8)) && routes(moe(rule, 288, 8)));
        assert!(!routes(moe(rule, 272, 8)));
        assert!(routes(moe(rule, 256, 6)));
        assert!(!routes(moe(rule, 256, 7)));
        let e = select_router(moe(rule, 352, 8)).expect_err("a sigmoid router at 352 was selected");
        assert_eq!(
            e.why,
            Refusal::RouterWidth {
                served: vec![288, 256, 240, 304, 368, 432, 496, 320, 128, 16]
            }
        );
    }

    /// The 320-expert sigmoid router (Solar-Open2: top 8, sigmoid with a
    /// selection bias, renormalized) is the `BiasedSigmoid` row at 320: ten
    /// experts a lane, its pick range 8 only.
    #[test]
    fn sigmoid_320_8_router_is_its_row() {
        let rule = rules::BIASED_SIGMOID;
        let r = select_router(moe(rule, 320, 8)).expect("no router row at 320");
        assert_eq!(r.body, super::RouterBody::BiasedSigmoid);
        assert_eq!((r.experts(), r.per_lane()), (320, 10));
        assert!(r.at.contains("sigmoid_320_8_router"), "{}", r.at);
        assert_eq!(router_row(super::RouterBody::BiasedSigmoid, 320), r);
        assert!(router_row_is(r, rule, false, (8, 8)));
        let x = select_router(moe(rule, 320, 9)).expect_err("320 at top 9");
        assert_eq!(x.why, Refusal::TopK { min: 8, max: 8 });
    }

    /// The 128-expert sigmoid router (MiniMax-M3: top 4, sigmoid with a
    /// selection bias, renormalized) is the `BiasedSigmoid` row at 128: four
    /// experts a lane, its pick range 4 only. The `Dflash` row also has 128
    /// experts, under another rule and body: the rule picks the row.
    #[test]
    fn sigmoid_128_4_router_is_its_row() {
        let rule = rules::BIASED_SIGMOID;
        let r = select_router(moe(rule, 128, 4)).expect("no router row at 128");
        assert_eq!(r.body, super::RouterBody::BiasedSigmoid);
        assert_eq!((r.experts(), r.per_lane()), (128, 4));
        assert!(r.at.contains("sigmoid_128_4_router"), "{}", r.at);
        assert_eq!(router_row(super::RouterBody::BiasedSigmoid, 128), r);
        assert!(router_row_is(r, rule, false, (4, 4)));
        for top_k in [3, 5, 8, 9] {
            let x = select_router(moe(rule, 128, top_k)).expect_err("128 at another top k");
            assert_eq!(x.why, Refusal::TopK { min: 4, max: 4 }, "{top_k}");
        }
        let dflash = select_router(moe(rules::BIASED_SQRT_SOFTPLUS, 128, 3)).expect("dflash");
        assert_eq!(dflash.body, super::RouterBody::Dflash);
    }

    /// Raon-1's router (128 experts, top 6) is the second `BiasedSigmoid` row
    /// at 128 beside MiniMax-M3's top 4: the pick count picks the row, the
    /// const form takes it, and a count neither row holds is refused.
    #[test]
    fn sigmoid_128_6_router_is_its_row_beside_128_4() {
        let rule = rules::BIASED_SIGMOID;
        let raon = select_router(moe(rule, 128, 6)).expect("no router row at 128/6");
        assert!(raon.at.contains("sigmoid_128_6_router"), "{}", raon.at);
        assert!(router_row_is(raon, rule, false, (6, 6)));
        let mm = select_router(moe(rule, 128, 4)).expect("no router row at 128/4");
        assert!(mm.at.contains("sigmoid_128_4_router"), "{}", mm.at);
        assert_eq!(router_row_k(super::RouterBody::BiasedSigmoid, 128, 6), raon);
        assert_eq!(router_row_k(super::RouterBody::BiasedSigmoid, 128, 4), mm);
        let x = select_router(moe(rule, 128, 5)).expect_err("128 at top 5");
        assert!(matches!(x.why, Refusal::TopK { .. }), "{x}");
    }

    /// The biased sigmoid at 256 experts and top 6 is its own row, after
    /// MiMo's top 8. Flint's rule is not this one (it divides by the floor
    /// policy), so Flint has no row. `router_row` at 256 stays MiMo's. Top 7
    /// matches neither pick range and is refused with the first row's range.
    #[test]
    fn sigmoid_256_6_router_is_its_row_beside_mimos_top_8() {
        let rule = rules::BIASED_SIGMOID;
        let fl = select_router(moe(rule, 256, 6)).expect("no router row at 256/6");
        assert!(fl.at.contains("sigmoid_256_6_router"), "{}", fl.at);
        assert_eq!((fl.experts(), fl.per_lane()), (256, 8));
        assert!(router_row_is(fl, rule, false, (6, 6)));
        assert_eq!(router_row_k(super::RouterBody::BiasedSigmoid, 256, 6), fl);
        let mimo = select_router(moe(rule, 256, 8)).expect("no router row at 256/8");
        assert!(mimo.at.contains("mimo2_router"), "{}", mimo.at);
        assert_eq!(router_row(super::RouterBody::BiasedSigmoid, 256), mimo);
        let x = select_router(moe(rule, 256, 7)).expect_err("256 at top 7");
        assert_eq!(x.why, Refusal::TopK { min: 8, max: 8 });
    }

    /// Bolt's routers (top 9) select only Bolt's rows: GLM's and MiMo's
    /// widths at top 9 are refused by their pick range, and Bolt's widths at
    /// top 8 by theirs. A count whose lanes do not all own the same number of
    /// experts (240 = 32 · 7.5) is a row: `per_lane` rounds up.
    #[test]
    fn bolt_routers_are_their_rows() {
        let rule = rules::BIASED_SIGMOID;
        for e in [240, 304, 368, 432, 496] {
            let r = select_router(moe(rule, e, 9)).unwrap_or_else(|x| panic!("{e}: {x}"));
            assert!(r.at.contains("nemotron_moe_router"), "{}", r.at);
            assert_eq!(r.per_lane(), e.div_ceil(32));
            assert_eq!(router_row(super::RouterBody::BiasedSigmoid, e), r);
            assert!(router_row_is(r, rule, false, (9, 9)));
            let e8 = select_router(moe(rule, e, 8)).expect_err("Bolt's width at top 8");
            assert_eq!(e8.why, Refusal::TopK { min: 9, max: 9 });
        }
        for e in [256, 288, 320] {
            let x = select_router(moe(rule, e, 9)).expect_err("GLM, MiMo or Solar at top 9");
            assert_eq!(x.why, Refusal::TopK { min: 8, max: 8 });
        }
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
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Softmax,
                bias: false,
                norm: RouterNorm::SelectedSum,
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

    /// No two rows select the same rule, expert width and pick count.
    #[test]
    fn no_two_router_rows_overlap() {
        for (i, a) in ROUTERS.iter().enumerate() {
            for b in &ROUTERS[i + 1..] {
                let picks = a.top_k_min <= b.top_k_max && b.top_k_min <= a.top_k_max;
                let overlap = a.experts == b.experts && (a.runs(b.rule) || b.runs(a.rule)) && picks;
                assert!(!overlap, "{} and {}", a.at, b.at);
            }
            assert!(a.top_k_min >= 1 && a.top_k_min <= a.top_k_max, "{}", a.at);
            assert!(a.per_lane() >= 1 && a.per_lane() <= 32, "{}", a.at);
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
                // PIN(2026-10-29): the head-128 group-of-four row (`GQA`'s
                // pack-4 row at head 128) joins the groups served.
                served: vec![
                    "exactly 8".to_string(),
                    "exactly 4".to_string(),
                    "exactly 5".to_string(),
                    "exactly 7".to_string(),
                    "exactly 16".to_string()
                ]
            }
        );
        let e = select_gqa(attn(16, 2, 160)).expect_err("head 160");
        assert_eq!(
            e.why,
            Refusal::HeadWidth {
                served: vec![64, 128, 192, 256]
            }
        );
        for s in [attn(24, 5, 256), attn(24, 0, 256), attn(0, 2, 256)] {
            assert_eq!(select_gqa(s).expect_err("ungrouped").why, Refusal::Group);
        }
        let e = select_gqa(attn(24, 8, 256)).expect_err("group 3");
        assert!(e.to_string().contains("group 3"), "{e}");
    }

    /// Head 128 at sixteen query heads a key head (Bolt's 32/2) selects the
    /// group-16 row: one call a group, two group-8 passes a key head inside it.
    #[test]
    fn a_head_128_group_of_16_selects_its_row() {
        let row = select_gqa(attn(32, 2, 128)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((row.pack, row.group), (16, GroupRule::One));
        assert!(row.at.contains("GROUP_16"), "{}", row.at);
        assert!(!row.window && !row.sinks && !row.value_scale);
    }

    /// Head 128 at five query heads a key head (10/2) selects the scalar
    /// group-5 row: one block a group, no window, sinks or value scale.
    #[test]
    fn a_head_128_group_of_5_selects_the_scalar_row() {
        let s = attn(10, 2, 128);
        let row = select_gqa(s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(row.pack, 5);
        assert_eq!(row.group, GroupRule::One);
        assert!(!row.window && !row.sinks && !row.value_scale);
        assert_eq!(row.packs(5), 1);
    }

    /// Head 128 at seven query heads a key head (Nanbeige4.3's 56/8) selects
    /// the group-7 row: one block a group, no window, sinks or value scale.
    #[test]
    fn a_head_128_group_of_7_selects_the_g7_row() {
        let s = attn(56, 8, 128);
        let row = select_gqa(s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(row.pack, 7);
        assert_eq!(row.group, GroupRule::One);
        assert!(!row.window && !row.sinks && !row.value_scale);
        assert_eq!(row.packs(7), 1);
        assert!(row.at.contains("GROUP_7"), "{}", row.at);
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

    /// A value scale on head 128 is refused by name: neither pack-8 row
    /// applies one. A window or sinks on head 256, whose rows cut and fold
    /// neither, is still refused by the first row of that width.
    #[test]
    fn a_value_scale_or_an_unserved_window_is_refused_by_name() {
        let base = attn(32, 4, 128);
        assert!(select_gqa(base).is_ok());
        let scaled = AttnShape {
            scaled: true,
            ..base
        };
        assert_eq!(
            select_gqa(scaled).expect_err("a value scale").why,
            Refusal::ValueScale
        );
        let wide = attn(16, 2, 256);
        let windowed = AttnShape {
            window: Some(128),
            ..wide
        };
        let e = select_gqa(windowed).expect_err("a window");
        assert_eq!(e.why, Refusal::Window { positions: 128 });
        assert!(e.to_string().contains("window of 128 positions"), "{e}");
        let sunk = AttnShape {
            sinks: true,
            ..wide
        };
        assert_eq!(select_gqa(sunk).expect_err("sinks").why, Refusal::Sinks);
    }

    /// Head 128 at eight query heads a key head, with sinks, selects the
    /// window row whether or not the layer cuts a window. The same shape
    /// with neither stays on the first pack-8 row, which `gqa_row` returns.
    #[test]
    fn a_head_128_group_of_8_with_sinks_selects_the_window_row() {
        let row_of = |window| {
            select_gqa(AttnShape {
                window,
                sinks: true,
                ..attn(32, 4, 128)
            })
            .unwrap_or_else(|e| panic!("{e}"))
        };
        for (name, window) in [("full", None), ("sliding", Some(513))] {
            let row = row_of(window);
            assert!(row.window && row.sinks && !row.value_scale, "{name}");
            assert_eq!((row.head, row.value, row.pack), (128, 128, 8), "{name}");
            assert_eq!(row.group, GroupRule::One, "{name}");
            assert!(
                row.at.contains("gqa_flash_seg_h128_swa"),
                "{name}: {}",
                row.at
            );
        }
        let plain = select_gqa(attn(32, 4, 128)).unwrap_or_else(|e| panic!("{e}"));
        assert!(!plain.window && !plain.sinks && !plain.value_scale);
        assert!(plain.at.contains("HEAD, GROUP"), "{}", plain.at);
        let first = gqa_row_v(128, 128, 8).expect("the pack-8 row");
        assert!(!first.window && !first.sinks);
        assert!(first.at.contains("HEAD, GROUP"), "{}", first.at);
    }

    /// GLM-5.3's biased sigmoid rule is the shared one the module exports:
    /// the noaux_tc router (sigmoid, the bias steering the choice only, the
    /// kept unbiased scores renormalized) MiMo-V2 runs too.
    #[test]
    fn the_biased_sigmoid_rule_is_glm_5_3_flashs() {
        let rule = RouterRule {
            score: Score::Sigmoid,
            bias: true,
            norm: RouterNorm::SelectedSum,
            gated: false,
        };
        assert_eq!(rules::BIASED_SIGMOID, rule);
        assert!(select_router(moe(rule, 288, 8)).is_ok());
    }

    /// `select_gqa` is `select_gqa_in` over [`GQA`], and each row of the table
    /// selects for the shape it can serve: the first row in the table's order
    /// that has its head and value widths, takes its group and covers its
    /// window, sinks and value scale. Mutant: `select_gqa` selects over
    /// `&GQA[1..]`.
    #[test]
    fn select_gqa_is_select_gqa_in_over_the_table() {
        for row in GQA {
            let group = row.pack;
            let s = AttnShape {
                n_head: 2 * group,
                n_kv: 2,
                head: row.head,
                value: row.value,
                window: row.window.then_some(512),
                sinks: row.sinks,
                scaled: row.value_scale,
            };
            let first = GQA
                .iter()
                .find(|r| {
                    r.head == s.head
                        && r.value == s.value
                        && r.takes(group)
                        && (s.window.is_none() || r.window)
                        && (!s.sinks || r.sinks)
                        && (!s.scaled || r.value_scale)
                })
                .copied()
                .expect("the row itself covers its own shape");
            assert_eq!(select_gqa(s), Ok(first), "{}", row.at);
            assert_eq!(select_gqa_in(GQA, s), select_gqa(s), "{}", row.at);
        }
    }

    /// A table of its own is selected from, and a shape it lacks is refused by
    /// its own widths, groups and window: a row of head 96 is no row of
    /// [`GQA`], so the table selects it and `GQA` refuses the head. Mutant:
    /// `select_gqa_in` reads [`GQA`] where it reads `table`.
    #[test]
    fn select_gqa_in_selects_from_the_table_it_is_given() {
        let row = GqaInst {
            head: 96,
            value: 96,
            pack: 3,
            group: GroupRule::One,
            window: true,
            sinks: false,
            value_scale: false,
            at: "a test row",
        };
        let table = [row];
        let s = attn(6, 2, 96);
        assert_eq!(select_gqa_in(&table, s), Ok(row));
        assert!(matches!(
            select_gqa(s).expect_err("GQA has no head 96").why,
            Refusal::HeadWidth { .. }
        ));
        assert_eq!(
            select_gqa_in(&table, attn(16, 2, 128))
                .expect_err("the table has no head 128")
                .why,
            Refusal::HeadWidth { served: vec![96] }
        );
        assert_eq!(
            select_gqa_in(&table, attn(8, 2, 96))
                .expect_err("the table takes group 3 alone")
                .why,
            Refusal::GqaGroup {
                group: 4,
                served: vec!["exactly 3".to_string()]
            }
        );
        let cuts_none = [GqaInst {
            window: false,
            ..row
        }];
        let windowed = AttnShape {
            window: Some(128),
            ..s
        };
        assert_eq!(
            select_gqa_in(&cuts_none, windowed)
                .expect_err("the row cuts no window")
                .why,
            Refusal::Window { positions: 128 }
        );
    }

    /// LFM3-1.7B's attention as its description states it: 16 query heads over 4 key heads of
    /// 128 values, no sinks, a window of 256 on the sliding layers and none on the full ones.
    fn lfm3(window: Option<u32>) -> Gqa {
        Gqa {
            heads: 16,
            kv_heads: 4,
            head_dim: 128,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: 128,
                base: 1_000_000.0,
                yarn: None,
                factors: window.is_none(),
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::ProjHead,
            select: None,
            window,
            sinks: false,
            value_scale: None,
        }
    }

    /// Both of LFM3's layer kinds, read through the description, select the head-128 group-of-four
    /// row: the sliding layer for its window, the full layer because the row's launch cuts window 0.
    #[test]
    fn lfm3_sliding_and_full_layers_select_the_group_of_four_row() {
        for (name, window) in [("sliding", Some(256)), ("full", None)] {
            let s = AttnShape::of(&lfm3(window));
            let row = select_gqa(s).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                (row.head, row.value, row.pack, row.group),
                (128, 128, 4, GroupRule::One),
                "{name}"
            );
            assert!(row.window && !row.sinks && !row.value_scale, "{name}");
            assert!(
                row.at.contains("gqa_flash_seg_h128_swa_g4"),
                "{name}: {}",
                row.at
            );
            assert_eq!(row.packs(s.n_head / s.n_kv), 1, "{name}");
        }
        let sliding = AttnShape::of(&lfm3(Some(256)));
        assert_eq!((sliding.window, sliding.sinks), (Some(256), false));
        assert_eq!(AttnShape::of(&lfm3(None)).window, None);
    }

    /// The row claims no sinks and no value scale: a group-of-four layer that asks for either is
    /// refused by name, as a window on a row that cuts none is.
    #[test]
    fn the_group_of_four_row_refuses_sinks_and_a_value_scale_by_name() {
        let base = AttnShape::of(&lfm3(Some(256)));
        let sunk = AttnShape {
            sinks: true,
            ..base
        };
        assert_eq!(select_gqa(sunk).expect_err("sinks").why, Refusal::Sinks);
        let scaled = AttnShape {
            scaled: true,
            ..base
        };
        assert_eq!(
            select_gqa(scaled).expect_err("a value scale").why,
            Refusal::ValueScale
        );
    }

    /// The group-of-four row changes no other shape's row: head 128 at eight query heads a key
    /// head with or without a window still takes the pack-8 rows, and head 256 at four takes its
    /// own pack-4 row, one block a key head.
    #[test]
    fn the_group_of_four_row_leaves_the_other_rows_alone() {
        let plain = select_gqa(attn(32, 4, 128)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((plain.pack, plain.window, plain.sinks), (8, false, false));
        assert!(plain.at.contains("HEAD, GROUP"), "{}", plain.at);
        let windowed = select_gqa(AttnShape {
            window: Some(513),
            sinks: true,
            ..attn(32, 4, 128)
        })
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (windowed.pack, windowed.window, windowed.sinks),
            (8, true, true)
        );
        let wide = select_gqa(attn(8, 2, 256)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (wide.head, wide.pack, wide.group),
            (256, 4, GroupRule::Packs)
        );
        assert_eq!(wide.packs(4), 1);
        assert_eq!(
            gqa_row_v(128, 128, 4).map(|r| r.pack),
            Some(4),
            "head 128 at four query heads takes the new row"
        );
        assert_eq!(gqa_row_v(128, 128, 8).map(|r| r.pack), Some(8));
    }

    #[test]
    fn gqa_rows_are_distinct_and_packed_rows_divide() {
        for (i, a) in GQA.iter().enumerate() {
            for b in &GQA[i + 1..] {
                assert!(
                    (a.head, a.value, a.pack, a.window, a.sinks, a.value_scale)
                        != (b.head, b.value, b.pack, b.window, b.sinks, b.value_scale),
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
