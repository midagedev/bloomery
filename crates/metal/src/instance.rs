use core::fmt;
use std::sync::{Mutex, PoisonError};

use crate::geom::{Geometry, KernelRefused};
use models::shape::{self, AttnShape, GqaInst, MoeShape, RouterBody, RouterInst, ShapeRefused};
use models::{DeltaKind, Ffn, GdnGate, Mixer, ModelSpec, OutGate};

/// One instance: the axes its kernels are compiled at, under a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instance {
    /// The instance's name: the stem of the sources `build.rs` writes for
    /// it, and how a refusal names it. [`UNREGISTERED`] for an instance made
    /// at open for a shape no row registers.
    pub name: &'static str,
    pub key: Key,
    /// The cargo feature a build whose host is this instance ([`HOST`]) is
    /// made with; none for the default build's.
    pub host_feature: Option<&'static str>,
    /// Whether a model session opens files at this instance: false for a row
    /// only the kernel gate compiles, whose family has no session yet.
    pub opens: bool,
}

/// The axes an instance's kernels are compiled at: what a file's shape is
/// matched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    /// The residual width: the router's K, a trip count its lanes unroll.
    pub hidden: u32,
    pub ffn: FfnAxes,
    pub attn: AttnAxes,
    /// The gated delta net's axes; none for a file of no delta layer.
    pub delta: Option<DeltaAxes>,
}

/// The FFN every layer of a file has: an instance compiles one kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnAxes {
    /// A routed FFN: the router's picks, then the experts they name.
    Routed(MoeAxes),
    /// A dense FFN of `ff` values: the experts' launches at one slot, expert
    /// 0 of a one-expert stack, at weight 1, and no router.
    Dense { ff: u32 },
}

/// A routed layer's axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeAxes {
    /// The router row of the shape table ([`shape::ROUTERS`]): its rule (a
    /// gated shared expert is one more router row and one more slot) and
    /// the experts it routes over, which a lane holds in registers.
    pub router: RouterInst,
    /// Routed experts a token keeps: the router's picks and the experts'
    /// id slots.
    pub top_k: u32,
    /// One expert's values: the pair's rows and the fold's K. A run-time K
    /// would put a division and a modulo on the fold's address path.
    pub expert_ff: u32,
}

/// An attention layer's axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnAxes {
    /// The flash row of the shape table ([`shape::GQA`]): the head and value
    /// widths, which size the lanes' register vectors, and the group.
    pub row: GqaInst,
    pub n_head: u32,
    pub n_kv: u32,
    /// Values the rope turns in each head.
    pub rot: u32,
    /// `attn_q` writes a gate beside each head's query.
    pub q_gate: bool,
    /// The output gate is one value a query head from a projection of its own
    /// (`OutGate::ProjHead`): exclusive with `q_gate`.
    pub head_gate: bool,
}

/// A gated delta layer's axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaAxes {
    pub value_heads: u32,
    /// Values per key and value head.
    pub head_cols: u32,
    /// Taps of the causal conv.
    pub conv_taps: u32,
}

/// The routed FFN of the 256-expert row: the gated softmax router over 256
/// experts and the shared expert, eight picks of 512 values.
const GATED_ROUTED: MoeAxes = MoeAxes {
    router: shape::router_row(RouterBody::Qwen35moe, 256),
    top_k: 8,
    expert_ff: 512,
};

/// The routed axes an instance of a dense FFN compiles its router's kernels
/// at: every instance's translation unit holds them, and no layer of its
/// files launches them.
const ROUTER_UNLAUNCHED: MoeAxes = GATED_ROUTED;

/// Qwen3.6-35B-A3B's gated delta net.
const QWEN35_GDN: DeltaAxes = DeltaAxes {
    value_heads: 32,
    head_cols: 128,
    conv_taps: 4,
};

/// The delta axes an instance of no delta layer compiles its delta kernels
/// at: every instance's translation unit holds them, and no layer of its
/// files launches them.
const DELTA_UNLAUNCHED: DeltaAxes = QWEN35_GDN;

/// A gated delta net of 48 value heads: three to each of 16 key heads, the
/// value heads as an axis beside Qwen3.6's 32.
const GDN_48: DeltaAxes = DeltaAxes {
    value_heads: 48,
    head_cols: 128,
    conv_taps: 4,
};

/// Qwen3.6-35B-A3B's shape (`qwen35moe`): the gated softmax router over 256
/// experts and the shared expert, the gated query, the gated delta net.
pub const QWEN35MOE_A3B: Instance = Instance {
    name: "qwen35moe_a3b",
    key: Key {
        hidden: 2048,
        ffn: FfnAxes::Routed(GATED_ROUTED),
        attn: AttnAxes {
            row: match shape::gqa_row(256, 16 / 2) {
                Some(r) => r,
                None => panic!("the shape table has no GQA row for this head at this group"),
            },
            n_head: 16,
            n_kv: 2,
            rot: 64,
            q_gate: true,
            head_gate: false,
        },
        delta: Some(QWEN35_GDN),
    },
    host_feature: None,
    opens: true,
};

/// Qwen3-30B-A3B's shape (`qwen3moe`): the softmax router over 128 experts
/// and no shared expert, the plain query of head 128 turned whole, no delta
/// layer.
pub const QWEN3MOE_A3B: Instance = Instance {
    name: "qwen3moe_a3b",
    key: Key {
        hidden: 2048,
        ffn: FfnAxes::Routed(MoeAxes {
            router: shape::router_row(RouterBody::Qwen3moe, 128),
            top_k: 8,
            expert_ff: 768,
        }),
        attn: AttnAxes {
            row: match shape::gqa_row(128, 32 / 4) {
                Some(r) => r,
                None => panic!("the shape table has no GQA row for this head at this group"),
            },
            n_head: 32,
            n_kv: 4,
            rot: 128,
            q_gate: false,
            head_gate: false,
        },
        delta: None,
    },
    host_feature: Some("host-qwen3moe-a3b"),
    opens: true,
};

/// Qwen3.6's shape with the delta net of [`GDN_48`]: the kernel gate's row of
/// the value-head axis, where every other axis is the one its clauses have a
/// fixture for.
pub const QWEN35MOE_GDN48: Instance = Instance {
    name: "qwen35moe_gdn48",
    key: Key {
        delta: Some(GDN_48),
        ..QWEN35MOE_A3B.key
    },
    host_feature: Some("host-qwen35moe-gdn48"),
    opens: true,
};

/// Qwen3.6's shape with the attention layers of a dense Qwen3.8-27B: 24 heads
/// over 4 key heads of 256 values, six query heads a key head. The kernel
/// gate's row of a group size that is not eight; every other axis is the one
/// its clauses have a fixture for.
pub const QWEN35MOE_GQA6: Instance = Instance {
    name: "qwen35moe_gqa6",
    key: Key {
        attn: AttnAxes {
            row: match shape::gqa_row(256, 24 / 4) {
                Some(r) => r,
                None => panic!("the shape table has no GQA row for this head at this group"),
            },
            n_head: 24,
            n_kv: 4,
            rot: 64,
            q_gate: true,
            head_gate: false,
        },
        ..QWEN35MOE_A3B.key
    },
    host_feature: Some("host-qwen35moe-gqa6"),
    opens: true,
};

/// Qwen3.6's shape with the attention layers of a 9B-class model: 16 heads
/// over 4 key heads of 256 values, four query heads a key head.
pub const QWEN35MOE_GQA4: Instance = Instance {
    name: "qwen35moe_gqa4",
    key: Key {
        attn: AttnAxes {
            row: match shape::gqa_row(256, 16 / 4) {
                Some(r) => r,
                None => panic!("the shape table has no GQA row for this head at this group"),
            },
            n_head: 16,
            n_kv: 4,
            rot: 64,
            q_gate: true,
            head_gate: false,
        },
        ..QWEN35MOE_A3B.key
    },
    host_feature: Some("host-qwen35moe-gqa4"),
    opens: true,
};

/// Qwen3.6's shape with the attention layers of LFM3-1.7B: 16 heads over 4
/// key heads of 128 values turned whole (four query heads a key head), a
/// window each layer's launch names, and the output gate one value a head from
/// a projection of its own. The kernel gate's row of the window, per-head gate
/// and head-128 group-4 axes; every other axis is the one its clauses have a
/// fixture for.
pub const LFM3_ATTN: Instance = Instance {
    name: "lfm3_attn",
    key: Key {
        attn: AttnAxes {
            row: GqaInst {
                head: 128,
                value: 128,
                pack: 4,
                group: shape::GroupRule::One,
                window: true,
                sinks: false,
                value_scale: false,
                at: "metal/kernels/attention.metal (a window word, a per-head gate buffer)",
            },
            n_head: 16,
            n_kv: 4,
            rot: 128,
            q_gate: false,
            head_gate: true,
        },
        ..QWEN35MOE_A3B.key
    },
    host_feature: Some("host-lfm3-attn"),
    opens: false,
};

/// Every instance `build.rs` writes and compiles. A shape with no row here
/// is compiled at open when every kernel family takes its geometry
/// ([`select`]), from the source; a row here is the route to its metallib
/// (its router and flash rows in the shape table first).
pub const REGISTERED: &[Instance] = &[
    QWEN35MOE_A3B,
    QWEN3MOE_A3B,
    QWEN35MOE_GDN48,
    QWEN35MOE_GQA6,
    QWEN35MOE_GQA4,
    LFM3_ATTN,
];

/// The name of an instance made at open for a shape no row registers. No
/// registered instance has it.
pub const UNREGISTERED: &str = "unregistered";

/// The instance the host is written at: the launches' grids and the
/// buffers' lengths read its axes as constants (`kernels::*_geom`), so the
/// kernels refuse a library of any other (`Kernels::with_geometry`). A
/// build picks it by feature: Qwen3.6's by default, Qwen3-30B-A3B's under
/// `host-qwen3moe-a3b`, the 48-value-head delta net's under
/// `host-qwen35moe-gdn48`, the six-query-head group's under
/// `host-qwen35moe-gqa6`, the four-query-head group's under `host-qwen35moe-gqa4`,
/// the windowed, per-head gated head-128 attention's under `host-lfm3-attn`.
#[cfg(not(any(
    feature = "host-qwen3moe-a3b",
    feature = "host-qwen35moe-gdn48",
    feature = "host-qwen35moe-gqa6",
    feature = "host-qwen35moe-gqa4",
    feature = "host-lfm3-attn"
)))]
pub const HOST: Instance = QWEN35MOE_A3B;

/// The instance the host is written at (see the default build's).
#[cfg(feature = "host-qwen35moe-gqa6")]
pub const HOST: Instance = QWEN35MOE_GQA6;

/// The instance the host is written at (see the default build's).
#[cfg(feature = "host-qwen35moe-gqa4")]
pub const HOST: Instance = QWEN35MOE_GQA4;

/// The instance the host is written at (see the default build's).
#[cfg(feature = "host-lfm3-attn")]
pub const HOST: Instance = LFM3_ATTN;

/// The instance the host is written at (see the default build's).
#[cfg(feature = "host-qwen3moe-a3b")]
pub const HOST: Instance = QWEN3MOE_A3B;

/// The instance the host is written at (see the default build's).
#[cfg(feature = "host-qwen35moe-gdn48")]
pub const HOST: Instance = QWEN35MOE_GDN48;

impl Key {
    /// The routed axes; none for a dense FFN.
    #[must_use]
    pub const fn routed(&self) -> Option<MoeAxes> {
        match self.ffn {
            FfnAxes::Routed(m) => Some(m),
            FfnAxes::Dense { .. } => None,
        }
    }

    /// Whether the FFN is dense.
    #[must_use]
    pub const fn is_dense(&self) -> bool {
        matches!(self.ffn, FfnAxes::Dense { .. })
    }
}

impl Instance {
    /// The delta axes this instance's delta kernels are compiled at: its
    /// own, or [`DELTA_UNLAUNCHED`] when its files have no delta layer.
    #[must_use]
    pub const fn delta_built(&self) -> DeltaAxes {
        match self.key.delta {
            Some(d) => d,
            None => DELTA_UNLAUNCHED,
        }
    }

    /// The routed axes this instance's router kernels are compiled at: its
    /// own, or [`ROUTER_UNLAUNCHED`] when its files have a dense FFN.
    #[must_use]
    pub const fn router_built(&self) -> MoeAxes {
        match self.key.ffn {
            FfnAxes::Routed(m) => m,
            FfnAxes::Dense { .. } => ROUTER_UNLAUNCHED,
        }
    }
}

impl fmt::Display for Instance {
    /// A registered instance by its name, an unregistered one by its shape.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name == UNREGISTERED {
            write!(f, "an unregistered shape ({})", self.key)
        } else {
            f.write_str(self.name)
        }
    }
}

/// Why a file has no instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The shape table has no row for a layer: its own refusal.
    Row { layer: usize, why: ShapeRefused },
    /// A layer of a kind no instance compiles.
    Kind { layer: usize, what: &'static str },
    /// A layer whose axes are not those of the earlier layers of its kind:
    /// an instance compiles each kind at one shape.
    Layers { layer: usize, what: &'static str },
    /// No layer of a kind every instance compiles: an FFN or attention.
    Missing { what: &'static str },
    /// The file's axes are no registered instance's.
    NoInstance { key: Key },
    /// The file's axes are no registered instance's, and a kernel cannot take
    /// them, so there is nothing to compile. The key is boxed: it would make
    /// every `Refused` as large as the key.
    Geometry { key: Box<Key>, why: KernelRefused },
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hidden {}; ", self.hidden)?;
        match &self.ffn {
            FfnAxes::Routed(m) => write!(
                f,
                "router {:?} of {} experts{}, top {}, expert ff {}",
                m.router.body,
                m.router.experts,
                if m.router.rule.gated {
                    " and a gated shared expert"
                } else {
                    ""
                },
                m.top_k,
                m.expert_ff,
            )?,
            FfnAxes::Dense { ff } => write!(f, "a dense FFN of ff {ff}")?,
        }
        let a = &self.attn;
        write!(
            f,
            "; attention {}/{} heads of {} (value {}), rope {}{}; ",
            a.n_head,
            a.n_kv,
            a.row.head,
            a.row.value,
            a.rot,
            if a.q_gate {
                ", a gated query"
            } else if a.head_gate {
                ", a per-head output gate"
            } else {
                ""
            },
        )?;
        match &self.delta {
            Some(d) => write!(
                f,
                "delta {} value heads of {}, conv {}",
                d.value_heads, d.head_cols, d.conv_taps
            ),
            None => f.write_str("no delta layer"),
        }
    }
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Row { layer, why } => write!(f, "layer {layer}: {why}"),
            Refused::Kind { layer, what } => {
                write!(f, "layer {layer}: {what}, which no instance compiles")
            }
            Refused::Layers { layer, what } => write!(
                f,
                "layer {layer}: its {what} is not an earlier layer's; an instance compiles one"
            ),
            Refused::Missing { what } => {
                write!(f, "no {what} layer; every instance compiles one")
            }
            Refused::NoInstance { key } => {
                write!(f, "no instance is built for {key}; the instances are")?;
                for (i, inst) in REGISTERED.iter().enumerate() {
                    let sep = if i == 0 { " " } else { "; " };
                    write!(f, "{sep}{} ({})", inst.name, inst.key)?;
                }
                Ok(())
            }
            Refused::Geometry { key, why } => {
                write!(
                    f,
                    "no instance is built for {key}, and no kernel takes it: {why}"
                )
            }
        }
    }
}

impl std::error::Error for Refused {}

/// Sets `slot` to `axes`, or holds it to the earlier layer's.
fn agree<T: PartialEq>(
    slot: &mut Option<T>,
    axes: T,
    layer: usize,
    what: &'static str,
) -> Result<(), Refused> {
    match slot {
        None => {
            *slot = Some(axes);
            Ok(())
        }
        Some(s) if *s == axes => Ok(()),
        Some(_) => Err(Refused::Layers { layer, what }),
    }
}

impl Key {
    /// The axes of a file of `hidden` values a row whose layers are
    /// `layers`: each routed layer's router row and each attention layer's
    /// flash row through the shape table's own lookups
    /// ([`shape::select_router`], [`shape::select_gqa`]), the rest as the
    /// description states them. A file's layers are all routed or all dense:
    /// an instance compiles one FFN.
    ///
    /// # Errors
    ///
    /// [`Refused`] naming the layer when the shape table has no row for it,
    /// it is of a kind no instance compiles, or its axes are not the earlier
    /// layers' of its kind; and naming the kind when the file has no FFN or
    /// no attention layer. A file of no delta layer has no delta axes.
    pub fn of<'a>(
        hidden: u32,
        layers: impl IntoIterator<Item = (&'a Mixer, &'a Ffn)>,
    ) -> Result<Key, Refused> {
        let (mut ffn_axes, mut attn, mut delta) = (None, None, None);
        for (layer, (mixer, ffn)) in layers.into_iter().enumerate() {
            match mixer {
                Mixer::Gqa(g) => {
                    if matches!(g.out_gate, OutGate::Proj | OutGate::ProjHead) {
                        return Err(Refused::Kind {
                            layer,
                            what: "an attention output gate on its own projection",
                        });
                    }
                    if g.rope.factors {
                        return Err(Refused::Kind {
                            layer,
                            what: "a rope with frequency factors",
                        });
                    }
                    let row = shape::select_gqa(AttnShape::of(g))
                        .map_err(|why| Refused::Row { layer, why })?;
                    let axes = AttnAxes {
                        row,
                        n_head: g.heads,
                        n_kv: g.kv_heads,
                        rot: g.rope.dims,
                        q_gate: g.out_gate == OutGate::Packed,
                        head_gate: false,
                    };
                    agree(&mut attn, axes, layer, "attention")?;
                }
                Mixer::DeltaRule(d)
                    if matches!(
                        d.kind,
                        DeltaKind::Gdn {
                            gate: GdnGate::Silu,
                            ..
                        }
                    ) =>
                {
                    let axes = DeltaAxes {
                        value_heads: d.v_heads,
                        head_cols: d.d,
                        conv_taps: d.conv,
                    };
                    agree(&mut delta, axes, layer, "gated delta net")?;
                }
                Mixer::DeltaRule(_) => {
                    return Err(Refused::Kind {
                        layer,
                        what: "a delta rule other than the gated delta net with a SiLU gate",
                    });
                }
                other => {
                    return Err(Refused::Kind {
                        layer,
                        what: other.what(),
                    });
                }
            }
            let axes = match ffn {
                Ffn::Moe(m) => {
                    let router = shape::select_router(MoeShape::of(m))
                        .map_err(|why| Refused::Row { layer, why })?;
                    FfnAxes::Routed(MoeAxes {
                        router,
                        top_k: m.top_k,
                        expert_ff: m.expert_ff,
                    })
                }
                Ffn::Dense { ff, .. } => FfnAxes::Dense { ff: *ff },
            };
            agree(&mut ffn_axes, axes, layer, "FFN")?;
        }
        Ok(Key {
            hidden,
            ffn: ffn_axes.ok_or(Refused::Missing { what: "FFN" })?,
            attn: attn.ok_or(Refused::Missing { what: "attention" })?,
            delta,
        })
    }
}

/// The registered instance of `key`.
///
/// # Errors
///
/// [`Refused::NoInstance`] naming `key` and every registered instance when
/// none has it: never the nearest row.
pub fn of_key(key: Key) -> Result<&'static Instance, Refused> {
    registered(&key).ok_or(Refused::NoInstance { key })
}

/// The registered instance of `key`, if any.
fn registered(key: &Key) -> Option<&'static Instance> {
    REGISTERED.iter().find(|inst| inst.key == *key)
}

/// The instances made at open, one per key: a cache, so a key selected twice
/// in a process (the plan's check, then the open's) is one instance for the
/// life of the process.
static OPENED: Mutex<Vec<&'static Instance>> = Mutex::new(Vec::new());

/// The instance a file of `key` runs: the registered instance of its key,
/// else an instance made at open when every kernel family takes the key's
/// geometry ([`Geometry::check`]). The library of an instance made at open
/// is compiled from the source.
///
/// # Errors
///
/// [`Refused::Geometry`] naming the kernel family, the axis and its value
/// when a kernel cannot take the geometry: never the nearest row.
pub fn open_key(key: Key) -> Result<&'static Instance, Refused> {
    if let Some(inst) = registered(&key) {
        return Ok(inst);
    }
    let made = Instance {
        name: UNREGISTERED,
        key,
        host_feature: None,
        opens: true,
    };
    Geometry::of(&made)
        .check()
        .map_err(|why| Refused::Geometry {
            key: Box::new(key),
            why,
        })?;
    let mut opened = OPENED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(inst) = opened.iter().find(|inst| inst.key == key) {
        return Ok(inst);
    }
    let inst: &'static Instance = Box::leak(Box::new(made));
    opened.push(inst);
    Ok(inst)
}

/// The instance a file described by `spec` runs ([`Key::of`], [`open_key`]).
///
/// # Errors
///
/// [`Refused`] when the file's shape has no row in the shape table, is of a
/// kind no kernel runs, or has a geometry no kernel takes.
pub fn select(spec: &ModelSpec) -> Result<&'static Instance, Refused> {
    open_key(Key::of(
        spec.hidden,
        spec.layers.iter().map(|l| (&l.mixer, &l.ffn)),
    )?)
}

#[cfg(test)]
mod tests {
    use models::shape::{Refusal, Shape};
    use models::{
        Act, DeltaRule, GdnGate, Gqa, KHeadMap, Moe, Rope, RopeMode, Router, RouterNorm, Score,
        Shared,
    };

    use super::*;

    /// A Qwen3.6-35B-A3B attention layer, as its reader describes it.
    fn gqa(heads: u32, kv_heads: u32, head_dim: u32, rot: u32) -> Mixer {
        gqa_gated(heads, kv_heads, head_dim, rot, OutGate::Packed)
    }

    /// An attention layer of `out_gate`.
    fn gqa_gated(heads: u32, kv_heads: u32, head_dim: u32, rot: u32, out_gate: OutGate) -> Mixer {
        Mixer::Gqa(Gqa {
            heads,
            kv_heads,
            head_dim,
            value_dim: head_dim,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: rot,
                base: 10_000_000.0,
                yarn: None,
                factors: false,
            },
            qk_norm: true,
            o_norm: false,
            out_gate,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        })
    }

    fn gdn(v_heads: u32) -> Mixer {
        Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Silu,
            },
            k_heads: 16,
            v_heads,
            d: 128,
            conv: 4,
        })
    }

    /// A routed FFN of `experts` softmax-normalized experts, a gated shared
    /// expert when `shared`.
    fn moe(experts: u32, top_k: u32, expert_ff: u32, shared: bool) -> Ffn {
        Ffn::Moe(Moe {
            experts,
            top_k,
            expert_ff,
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
            shared: shared.then_some(Shared {
                ff: expert_ff,
                act: Act::SwiGlu { limit: None },
                sigmoid_gate: true,
            }),
        })
    }

    /// A dense SwiGLU FFN of `ff` values.
    fn dense_ffn(ff: u32) -> Ffn {
        Ffn::Dense {
            ff,
            act: Act::SwiGlu { limit: None },
        }
    }

    /// Qwen3.6's four kinds of layer as its fixture keeps them: delta,
    /// attention, delta, attention.
    fn qwen36() -> Vec<(Mixer, Ffn)> {
        let ffn = || moe(256, 8, 512, true);
        vec![
            (gdn(32), ffn()),
            (gqa(16, 2, 256, 64), ffn()),
            (gdn(32), ffn()),
            (gqa(16, 2, 256, 64), ffn()),
        ]
    }

    /// Qwen3-30B-A3B's layers as its fixture keeps them: four of plain
    /// head-128 attention over 128 routed experts and no shared one.
    fn qwen3_30b() -> Vec<(Mixer, Ffn)> {
        (0..4)
            .map(|_| {
                (
                    gqa_gated(32, 4, 128, 128, OutGate::None),
                    moe(128, 8, 768, false),
                )
            })
            .collect()
    }

    fn key(hidden: u32, layers: &[(Mixer, Ffn)]) -> Result<Key, Refused> {
        Key::of(hidden, layers.iter().map(|(m, f)| (m, f)))
    }

    #[test]
    fn the_registered_rows_are_the_shape_tables() {
        for inst in REGISTERED {
            let m = inst
                .key
                .routed()
                .expect("a registered row is a routed FFN's");
            let s = MoeShape {
                rule: m.router.rule,
                experts: m.router.experts,
                top_k: m.top_k,
            };
            assert_eq!(
                shape::select_router(s),
                Ok(m.router),
                "{}: the router row is the one the table selects at its picks",
                inst.name
            );
            assert_ne!(
                inst.name, UNREGISTERED,
                "the name is an instance made at open's"
            );
            let named = REGISTERED.iter().filter(|i| i.name == inst.name).count();
            let keyed = REGISTERED.iter().filter(|i| i.key == inst.key).count();
            assert_eq!(
                (named, keyed),
                (1, 1),
                "{}: one row a name and a key",
                inst.name
            );
        }
        assert!(
            REGISTERED.contains(&HOST),
            "the host's instance is registered"
        );
        let features: Vec<_> = REGISTERED.iter().map(|i| i.host_feature).collect();
        for (i, f) in features.iter().enumerate() {
            assert!(
                !features[..i].contains(f),
                "{}: one build a host instance",
                REGISTERED[i].name
            );
        }
        let manifest = include_str!("../Cargo.toml");
        for f in features.iter().flatten() {
            assert!(
                manifest.lines().any(|l| l.starts_with(&format!("{f} = "))),
                "{f}: a feature of this crate"
            );
        }
    }

    #[test]
    fn each_shape_selects_its_instance() {
        assert_eq!(
            key(2048, &qwen36()).and_then(of_key),
            Ok(&QWEN35MOE_A3B),
            "Qwen3.6's layers"
        );
        assert_eq!(
            key(2048, &qwen3_30b()).and_then(of_key),
            Ok(&QWEN3MOE_A3B),
            "Qwen3-30B-A3B's layers"
        );
        let gdn48: Vec<_> = qwen36()
            .into_iter()
            .map(|(m, f)| match m {
                Mixer::DeltaRule(_) => (gdn(48), f),
                _ => (m, f),
            })
            .collect();
        assert_eq!(
            key(2048, &gdn48).and_then(of_key),
            Ok(&QWEN35MOE_GDN48),
            "Qwen3.6's layers with 48 delta value heads"
        );
        assert_eq!(QWEN3MOE_A3B.key.delta, None, "no delta layer");
        assert_eq!(
            QWEN3MOE_A3B.delta_built(),
            QWEN35MOE_A3B.delta_built(),
            "an instance of no delta layer compiles the delta kernels at Qwen3.6's axes"
        );
        // A gated query on Qwen3-30B's attention, or its 30B layers with a
        // gated shared expert: no instance.
        let mut gated = qwen3_30b();
        gated[2].0 = gqa_gated(32, 4, 128, 128, OutGate::Packed);
        let got = key(2048, &gated);
        assert_eq!(
            got,
            Err(Refused::Layers {
                layer: 2,
                what: "attention"
            }),
            "one gated layer among plain ones"
        );
        let all_gated: Vec<_> = qwen3_30b()
            .into_iter()
            .map(|(_, f)| (gqa_gated(32, 4, 128, 128, OutGate::Packed), f))
            .collect();
        let got = key(2048, &all_gated).and_then(of_key);
        let Err(e @ Refused::NoInstance { .. }) = &got else {
            panic!("a gated head-128 query: {got:?}");
        };
        assert!(
            e.to_string().contains("a gated query; no delta layer"),
            "{e}"
        );
    }

    #[test]
    fn a_shape_with_no_instance_is_refused_by_name() {
        let layers = qwen36();
        let got = key(2048, &layers).and_then(of_key);
        assert_eq!(
            got,
            Ok(&QWEN35MOE_A3B),
            "Qwen3.6's layers select its instance"
        );

        // Every axis an instance compiles, moved alone: no instance, named.
        let wide = layers
            .iter()
            .map(|(m, _)| (m.clone(), moe(256, 8, 768, true)))
            .collect::<Vec<_>>();
        let narrow = layers
            .iter()
            .map(|(m, f)| match m {
                Mixer::Gqa(_) => (gqa(16, 2, 256, 128), f.clone()),
                _ => (m.clone(), f.clone()),
            })
            .collect::<Vec<_>>();
        let delta = layers
            .iter()
            .map(|(m, f)| match m {
                Mixer::DeltaRule(_) => (gdn(64), f.clone()),
                _ => (m.clone(), f.clone()),
            })
            .collect::<Vec<_>>();
        // Qwen3-30B-A3B's attention: the head-128 row of eight, which the
        // table has and no instance compiles.
        let head128 = layers
            .iter()
            .map(|(m, f)| match m {
                Mixer::Gqa(_) => (gqa(32, 4, 128, 128), f.clone()),
                _ => (m.clone(), f.clone()),
            })
            .collect::<Vec<_>>();
        for (case, hidden, layers, names) in [
            ("hidden", 4096, &layers, "hidden 4096;"),
            ("expert ff", 2048, &wide, "expert ff 768;"),
            ("rope", 2048, &narrow, "rope 128,"),
            ("delta heads", 2048, &delta, "delta 64 value heads"),
            ("head 128", 2048, &head128, "attention 32/4 heads of 128"),
        ] {
            let got = key(hidden, layers).and_then(of_key);
            let Err(e @ Refused::NoInstance { .. }) = &got else {
                panic!("{case}: {got:?} is not the named refusal");
            };
            let text = e.to_string();
            assert!(
                text.starts_with("no instance is built for")
                    && text.contains(names)
                    && text.contains(QWEN35MOE_A3B.name),
                "{case}: {text}"
            );
        }

        // A shape the table has no row for: the table's own refusal, at the
        // layer that asked.
        let mut no_row = layers.clone();
        no_row[1].1 = moe(128, 8, 768, true);
        let got = key(2048, &no_row);
        let Err(Refused::Row { layer: 1, why }) = &got else {
            panic!("a gated router of 128 experts: {got:?}");
        };
        assert!(
            matches!(why.shape, Shape::Moe(_))
                && matches!(why.why, Refusal::RouterWidth { .. })
                && got
                    .as_ref()
                    .unwrap_err()
                    .to_string()
                    .starts_with("layer 1: router"),
            "{why}"
        );
        let mut odd = layers.clone();
        odd[3].0 = gqa(16, 4, 256, 64);
        assert_eq!(
            key(2048, &odd),
            Err(Refused::Layers {
                layer: 3,
                what: "attention"
            }),
            "two attention shapes in one file"
        );
        let no_delta: Vec<_> = layers
            .iter()
            .filter(|(m, _)| matches!(m, Mixer::Gqa(_)))
            .cloned()
            .collect();
        let got = key(2048, &no_delta).and_then(of_key);
        let Err(e @ Refused::NoInstance { key: refused }) = &got else {
            panic!("Qwen3.6's attention with no delta layer: {got:?}");
        };
        assert!(
            refused.delta.is_none()
                && e.to_string()
                    .contains("of 256 (value 256), rope 64, a gated query; no delta layer"),
            "a file of no delta layer: {e}"
        );
        let delta_only: Vec<_> = layers
            .iter()
            .filter(|(m, _)| matches!(m, Mixer::DeltaRule(_)))
            .cloned()
            .collect();
        assert_eq!(
            key(2048, &delta_only),
            Err(Refused::Missing { what: "attention" }),
            "a file of no attention layer"
        );
        let mut dense = layers;
        dense[2].1 = dense_ffn(512);
        assert_eq!(
            key(2048, &dense),
            Err(Refused::Layers {
                layer: 2,
                what: "FFN"
            }),
            "a dense layer among routed ones: an instance compiles one FFN"
        );
    }

    /// Qwen3.8-27B's four kinds of layer as its fixture keeps them: three
    /// delta layers of 48 value heads and one attention layer of 24 heads over
    /// 4, each with a dense FFN of 17,408 values.
    fn qwen38_27b() -> Vec<(Mixer, Ffn)> {
        let mut layers: Vec<(Mixer, Ffn)> = (0..3).map(|_| (gdn(48), dense_ffn(17_408))).collect();
        layers.push((gqa(24, 4, 256, 64), dense_ffn(17_408)));
        layers
    }

    #[test]
    fn a_dense_file_is_one_ffn_axis_and_opens_an_instance_made_at_open() {
        let layers = qwen38_27b();
        let key = key(5120, &layers).expect("the shape table has its rows");
        assert_eq!(key.ffn, FfnAxes::Dense { ff: 17_408 });
        assert!(key.is_dense() && key.routed().is_none());
        assert!(of_key(key).is_err(), "no row registers a dense FFN");
        let text = key.to_string();
        assert!(
            text.starts_with("hidden 5120; a dense FFN of ff 17408; attention 24/4 heads of 256"),
            "{text}"
        );
        let opened = open_key(key).expect("every kernel takes the 27B's geometry");
        assert_eq!(opened.name, UNREGISTERED);
        assert_eq!(opened.key, key);
        let g = Geometry::of(opened);
        assert_eq!(g.check(), Ok(()));
        let e = g.experts;
        assert_eq!(
            (e.slots, e.routed_ids, e.has_shared, e.k, e.dense),
            (1, 1, false, 17_408, true),
            "one slot, expert 0, no shared slot"
        );
        assert_eq!(
            opened.router_built(),
            QWEN35MOE_A3B.router_built(),
            "an instance of a dense FFN compiles the router kernels at Qwen3.6's axes"
        );
        // Every routed row keeps the geometry it had: none is dense.
        for inst in REGISTERED {
            assert!(!Geometry::of(inst).experts.dense, "{}", inst.name);
        }
    }

    #[test]
    fn a_routed_layer_among_dense_ones_is_refused_by_name() {
        let mut layers = qwen38_27b();
        layers[1].1 = moe(256, 8, 512, true);
        assert_eq!(
            key(5120, &layers),
            Err(Refused::Layers {
                layer: 1,
                what: "FFN"
            }),
            "a routed layer among dense ones"
        );
        layers[1].1 = dense_ffn(8192);
        assert_eq!(
            key(5120, &layers),
            Err(Refused::Layers {
                layer: 1,
                what: "FFN"
            }),
            "two dense widths in one file"
        );
    }

    #[test]
    fn a_layer_the_kernels_read_otherwise_is_refused_by_name() {
        // The key records a query gate as present or absent; a gate on a
        // projection of its own is neither.
        let own: Vec<_> = qwen3_30b()
            .into_iter()
            .map(|(_, f)| (gqa_gated(32, 4, 128, 128, OutGate::Proj), f))
            .collect();
        assert_eq!(
            key(2048, &own),
            Err(Refused::Kind {
                layer: 0,
                what: "an attention output gate on its own projection"
            })
        );
        // The delta kernel gates its output with SiLU.
        let mut sigmoid = qwen36();
        sigmoid[0].0 = Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Gdn {
                khead_map: KHeadMap::Tiled,
                gate: GdnGate::Sigmoid,
            },
            k_heads: 16,
            v_heads: 32,
            d: 128,
            conv: 4,
        });
        assert_eq!(
            key(2048, &sigmoid),
            Err(Refused::Kind {
                layer: 0,
                what: "a delta rule other than the gated delta net with a SiLU gate"
            })
        );
    }

    #[test]
    fn a_shape_no_row_registers_opens_when_every_kernel_takes_it() {
        for inst in REGISTERED {
            assert_eq!(
                *open_key(inst.key).expect("a registered key opens"),
                *inst,
                "{}: a registered key opens its row",
                inst.name
            );
        }
        // Qwen3.6's layers at another hidden width: no row, and the kernels
        // take it.
        let wide = key(4096, &qwen36()).expect("the shape table has its rows");
        assert!(of_key(wide).is_err(), "no row has hidden 4096");
        let opened = open_key(wide).expect("every kernel takes hidden 4096");
        assert_eq!(opened.name, UNREGISTERED);
        assert_eq!(opened.key, wide);
        assert_eq!(opened.host_feature, None);
        assert!(
            core::ptr::eq(opened, open_key(wide).expect("opens again")),
            "one instance a key"
        );
        assert!(
            opened
                .to_string()
                .starts_with("an unregistered shape (hidden 4096;"),
            "{opened}"
        );
        assert_eq!(Geometry::of(opened).check(), Ok(()));
    }

    #[test]
    fn a_shape_no_kernel_takes_is_refused_by_name() {
        // Hidden 3136 is no multiple of the router's 1024.
        let odd = key(3136, &qwen36()).expect("the shape table has its rows");
        let got = open_key(odd);
        let Err(Refused::Geometry { key, why }) = &got else {
            panic!("hidden 3136: {got:?}");
        };
        assert_eq!(**key, odd);
        assert_eq!(
            (why.kernel, why.axis, why.value.as_str()),
            ("router", "hidden", "3136")
        );
        let text = got.unwrap_err().to_string();
        assert!(
            text.starts_with("no instance is built for hidden 3136;")
                && text.ends_with(
                    "router: hidden is 3136; it must be a multiple of 1024 \
                     (a float4 a lane over 256 threads)"
                ),
            "{text}"
        );
    }
}
