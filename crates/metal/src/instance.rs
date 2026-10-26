use core::fmt;

use models::shape::{self, AttnShape, GqaInst, MoeShape, RouterBody, RouterInst, ShapeRefused};
use models::{DeltaKind, Ffn, Mixer, ModelSpec, OutGate};

/// One instance: the axes its kernels are compiled at, under a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instance {
    /// The instance's name: the stem of the sources `build.rs` writes for
    /// it, and how a refusal names it.
    pub name: &'static str,
    pub key: Key,
}

/// The axes an instance's kernels are compiled at: what a file's shape is
/// matched on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    /// The residual width: the router's K, a trip count its lanes unroll.
    pub hidden: u32,
    pub moe: MoeAxes,
    pub attn: AttnAxes,
    pub delta: DeltaAxes,
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

/// Qwen3.6-35B-A3B's shape (`qwen35moe`): the gated softmax router over 256
/// experts and the shared expert, the gated query, the gated delta net.
pub const QWEN35MOE_A3B: Instance = Instance {
    name: "qwen35moe_a3b",
    key: Key {
        hidden: 2048,
        moe: MoeAxes {
            router: shape::router_row(RouterBody::Qwen35moe, 256),
            top_k: 8,
            expert_ff: 512,
        },
        attn: AttnAxes {
            row: match shape::gqa_row(256, 16 / 2) {
                Some(r) => r,
                None => panic!("the shape table has no GQA row for this head at this group"),
            },
            n_head: 16,
            n_kv: 2,
            rot: 64,
            q_gate: true,
        },
        delta: DeltaAxes {
            value_heads: 32,
            head_cols: 128,
            conv_taps: 4,
        },
    },
};

/// Every instance `build.rs` writes and compiles. A shape with no row here
/// has no kernels: a file of it is refused by name ([`select`]), and the
/// route to running it is a row here (its router and flash rows in the
/// shape table first).
pub const REGISTERED: &[Instance] = &[QWEN35MOE_A3B];

/// The instance the host is written at: the launches' grids and the
/// buffers' lengths read its axes as constants (`kernels::*_geom`), so the
/// kernels refuse a library of any other (`Kernels::with_geometry`).
pub const HOST: Instance = QWEN35MOE_A3B;

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
    /// No layer of a kind every instance compiles.
    Missing { what: &'static str },
    /// The file's axes are no registered instance's.
    NoInstance { key: Key },
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (m, a, d) = (&self.moe, &self.attn, &self.delta);
        write!(
            f,
            "hidden {}; router {:?} of {} experts{}, top {}, expert ff {}; attention {}/{} heads \
             of {} (value {}), rope {}{}; delta {} value heads of {}, conv {}",
            self.hidden,
            m.router.body,
            m.router.experts,
            if m.router.rule.gated {
                " and a gated shared expert"
            } else {
                ""
            },
            m.top_k,
            m.expert_ff,
            a.n_head,
            a.n_kv,
            a.row.head,
            a.row.value,
            a.rot,
            if a.q_gate { ", a gated query" } else { "" },
            d.value_heads,
            d.head_cols,
            d.conv_taps
        )
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
    /// description states them.
    ///
    /// # Errors
    ///
    /// [`Refused`] naming the layer when the shape table has no row for it,
    /// it is of a kind no instance compiles, or its axes are not the earlier
    /// layers' of its kind; and naming the kind when the file has none of
    /// one.
    pub fn of<'a>(
        hidden: u32,
        layers: impl IntoIterator<Item = (&'a Mixer, &'a Ffn)>,
    ) -> Result<Key, Refused> {
        let (mut moe, mut attn, mut delta) = (None, None, None);
        for (layer, (mixer, ffn)) in layers.into_iter().enumerate() {
            match mixer {
                Mixer::Gqa(g) => {
                    let row = shape::select_gqa(AttnShape::of(g))
                        .map_err(|why| Refused::Row { layer, why })?;
                    let axes = AttnAxes {
                        row,
                        n_head: g.heads,
                        n_kv: g.kv_heads,
                        rot: g.rope.dims,
                        q_gate: g.out_gate == OutGate::Packed,
                    };
                    agree(&mut attn, axes, layer, "attention")?;
                }
                Mixer::DeltaRule(d) if matches!(d.kind, DeltaKind::Gdn { .. }) => {
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
                        what: "a delta rule other than the gated delta net",
                    });
                }
                other => {
                    return Err(Refused::Kind {
                        layer,
                        what: other.what(),
                    });
                }
            }
            let Ffn::Moe(m) = ffn else {
                return Err(Refused::Kind {
                    layer,
                    what: "a dense FFN",
                });
            };
            let router =
                shape::select_router(MoeShape::of(m)).map_err(|why| Refused::Row { layer, why })?;
            let axes = MoeAxes {
                router,
                top_k: m.top_k,
                expert_ff: m.expert_ff,
            };
            agree(&mut moe, axes, layer, "routed FFN")?;
        }
        Ok(Key {
            hidden,
            moe: moe.ok_or(Refused::Missing { what: "routed" })?,
            attn: attn.ok_or(Refused::Missing { what: "attention" })?,
            delta: delta.ok_or(Refused::Missing {
                what: "gated delta net",
            })?,
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
    REGISTERED
        .iter()
        .find(|inst| inst.key == key)
        .ok_or(Refused::NoInstance { key })
}

/// The registered instance a file described by `spec` runs.
///
/// # Errors
///
/// [`Refused`] when the file's shape has no instance ([`Key::of`],
/// [`of_key`]).
pub fn select(spec: &ModelSpec) -> Result<&'static Instance, Refused> {
    of_key(Key::of(
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
            },
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::Packed,
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

    fn key(hidden: u32, layers: &[(Mixer, Ffn)]) -> Result<Key, Refused> {
        Key::of(hidden, layers.iter().map(|(m, f)| (m, f)))
    }

    #[test]
    fn the_registered_rows_are_the_shape_tables() {
        for inst in REGISTERED {
            let m = &inst.key.moe;
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
                Mixer::DeltaRule(_) => (gdn(48), f.clone()),
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
            ("delta heads", 2048, &delta, "delta 48 value heads"),
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
        assert_eq!(
            key(2048, &no_delta),
            Err(Refused::Missing {
                what: "gated delta net"
            }),
            "a file of no delta layer"
        );
        let mut dense = layers.clone();
        dense[2].1 = Ffn::Dense {
            ff: 512,
            act: Act::SwiGlu { limit: None },
        };
        assert_eq!(
            key(2048, &dense),
            Err(Refused::Kind {
                layer: 2,
                what: "a dense FFN"
            }),
            "a dense layer"
        );
    }
}
