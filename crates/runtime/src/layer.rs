//! What a layer's program runs, read off its [`LayerSpec`]: its mixer, its
//! feed-forward block and its residual as the sub-layer programs a body
//! composes ([`Layer`]), and which parts of a walk ([`crate::sched`]) the
//! layer fills — every layer a front, and a layer whose feed-forward block
//! routes a shadow and a back around its host leg ([`Layer::host_leg`]). A
//! body matches on these once at load, never on a layer number, so a model
//! whose kinds interleave another way is the same program over another
//! description.
//!
//! [`hosted`] is the host tier's side of the same reading: the layers whose
//! legs it serves, which a slot map holds as one run of consecutive layers.

use std::fmt;
use std::ops::Range;

use models::{Ffn, LayerSpec, Mixer, Residual};

/// A layer's mixer, as the program that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    Gqa,
    Latent,
    DeltaRule,
}

/// A layer's feed-forward block, as the program that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// One dense SwiGLU block: the card runs it whole.
    Dense,
    /// Routed experts: the router hands the step's slots to the host tier,
    /// whose answer the back joins.
    Moe,
}

/// How a layer's input and output meet the residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualKind {
    Plain,
    /// Hyper-connection streams around every sub-layer.
    Hc,
}

/// One layer's sub-layer programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    pub mixer: MixerKind,
    pub ffn: FfnKind,
    pub residual: ResidualKind,
}

impl Layer {
    /// `spec`'s sub-layer programs.
    #[must_use]
    pub fn of(spec: &LayerSpec) -> Layer {
        Layer {
            mixer: match spec.mixer {
                Mixer::Gqa(_) => MixerKind::Gqa,
                Mixer::Latent(_) => MixerKind::Latent,
                Mixer::DeltaRule(_) => MixerKind::DeltaRule,
            },
            ffn: match spec.ffn {
                Ffn::Dense { .. } => FfnKind::Dense,
                Ffn::Moe(_) => FfnKind::Moe,
            },
            residual: match spec.residual {
                Residual::Plain => ResidualKind::Plain,
                Residual::Hc => ResidualKind::Hc,
            },
        }
    }

    /// Whether the layer hands a leg to the host tier in a body whose host
    /// tier serves the routed experts: its shadow and back then run around
    /// the leg, and a layer without one is its front alone.
    #[must_use]
    pub fn host_leg(self) -> bool {
        self.ffn == FfnKind::Moe
    }
}

/// Why a model's layers have no host run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoRun {
    /// No layer routes.
    None,
    /// The routing layers are not consecutive: the first layer between two
    /// of them that does not route.
    Gap { at: usize, run: Range<usize> },
}

impl fmt::Display for NoRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoRun::None => f.write_str("no layer routes: there is no host leg to serve"),
            NoRun::Gap { at, run } => write!(
                f,
                "the routing layers {}..{} are not one run: layer {at} between them does not \
                 route, and a host tier's slot map holds one run of layers",
                run.start, run.end
            ),
        }
    }
}

/// The layers a host tier serves: those with a host leg ([`Layer::host_leg`]),
/// which must be one run of consecutive layers — the rows of a slot map.
pub fn hosted(layers: &[LayerSpec]) -> Result<Range<usize>, NoRun> {
    let leg = |l: &LayerSpec| Layer::of(l).host_leg();
    let first = layers.iter().position(leg).ok_or(NoRun::None)?;
    let last = layers.iter().rposition(leg).ok_or(NoRun::None)?;
    let run = first..last + 1;
    match (first..=last).find(|&l| !leg(&layers[l])) {
        Some(at) => Err(NoRun::Gap { at, run }),
        None => Ok(run),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::{self, At, Item, LayerProgram, Overlap, Port, PortKind, Refused};
    use crate::state::{Store, StoreRule, every_position, stores};
    use models::{
        Act, DeltaKind, DeltaRule, Latent, LatentOut, LatentUp, Moe, PoolRule, Router, Score,
        Selector, Shared,
    };

    /// GLM-5.3-Flash's 45 trunk layers as `glm5next_meta.rs`'s `VIEW` pins
    /// them: latent attention with a token-pool indexer at 3, 7, …, 43, KDA
    /// elsewhere; a dense block on 0–2, 288 experts routing 8 with a shared
    /// expert after; hyper-connections on every layer.
    fn glm() -> Vec<LayerSpec> {
        let act = Act::SwiGlu { limit: Some(10.0) };
        (0..45)
            .map(|l| LayerSpec {
                mixer: if l % 4 == 3 {
                    Mixer::Latent(Latent {
                        heads: 64,
                        q_lora: 1536,
                        latent: 512,
                        up: LatentUp::Absorbed { qk: 256, v: 256 },
                        rope: None,
                        q_head_norm: false,
                        out: LatentOut::Plain,
                        window: None,
                        sinks: false,
                        compress: None,
                        select: Some(Selector::TokenPool {
                            heads: 32,
                            d: 128,
                            top_k: 2048,
                            pool: 4,
                            rule: PoolRule::Learned { key_eps: 1e-6 },
                        }),
                    })
                } else {
                    Mixer::DeltaRule(DeltaRule {
                        kind: DeltaKind::Kda {
                            gate_lower_bound: -5.0,
                        },
                        k_heads: 64,
                        v_heads: 64,
                        d: 128,
                        conv: 4,
                    })
                },
                ffn: if l < 3 {
                    Ffn::Dense { ff: 12288, act }
                } else {
                    Ffn::Moe(Moe {
                        experts: 288,
                        top_k: 8,
                        expert_ff: 2048,
                        act,
                        router: Router {
                            score: Score::Sigmoid,
                            bias: true,
                            norm: true,
                            scale: 2.5,
                            hash: false,
                        },
                        shared: Some(Shared {
                            ff: 2048,
                            act,
                            sigmoid_gate: false,
                        }),
                    })
                },
                residual: Residual::Hc,
                extras: Vec::new(),
            })
            .collect()
    }

    /// Each GLM layer's programs: the mixer by its own spec, the dense block
    /// on 0–2, the routed one after, the streams everywhere.
    #[test]
    fn glm_layers_by_their_specs() {
        for (l, s) in glm().iter().enumerate() {
            let k = Layer::of(s);
            let mixer = if l % 4 == 3 {
                MixerKind::Latent
            } else {
                MixerKind::DeltaRule
            };
            let ffn = if l < 3 { FfnKind::Dense } else { FfnKind::Moe };
            assert_eq!(
                k,
                Layer {
                    mixer,
                    ffn,
                    residual: ResidualKind::Hc
                },
                "layer {l}"
            );
            assert_eq!(k.host_leg(), l >= 3, "layer {l}");
        }
    }

    /// The host tier serves GLM's routing layers, 3 to 44; a dense layer
    /// among them, or none routing, is refused by name.
    #[test]
    fn glm_hosted_run() {
        let mut g = glm();
        assert_eq!(hosted(&g), Ok(3..45));
        g[20].ffn = Ffn::Dense {
            ff: 12288,
            act: Act::SwiGlu { limit: None },
        };
        assert_eq!(hosted(&g), Err(NoRun::Gap { at: 20, run: 3..45 }));
        let dense: Vec<LayerSpec> = g[..3].to_vec();
        assert_eq!(hosted(&dense), Err(NoRun::None));
    }

    /// GLM's stores: a KDA layer's conv window and recurrent state, a latent
    /// layer's latent rows and pooled keys, each read past any window — so a
    /// prompt call runs every layer at every position.
    #[test]
    fn glm_stores() {
        let g = glm();
        assert_eq!(
            stores(&g[0]),
            [
                (Store::Conv, StoreRule::Window { slots: 3 }),
                (Store::State, StoreRule::Recurrent),
            ]
        );
        assert_eq!(
            stores(&g[3]),
            [
                (Store::Latent, StoreRule::Positional),
                (Store::PooledKeys, StoreRule::Positional),
            ]
        );
        assert!(g.iter().all(every_position));
    }

    /// A port and a program that record the parts a walk asks for, the way a
    /// body fills them: every layer's front, and the shadow and the back of a
    /// layer with a host leg alone.
    struct Rec {
        layers: Vec<Layer>,
        seen: Vec<Item>,
    }

    struct RecPort;

    impl Port for RecPort {
        type Error = Refused;
        const KIND: PortKind = PortKind::Step;

        fn open(&mut self, _o: Overlap) -> Result<(), Refused> {
            Ok(())
        }

        fn refused(why: Refused) -> Refused {
            why
        }
    }

    impl LayerProgram for Rec {
        type Port = RecPort;

        fn front(&mut self, _port: &mut RecPort, at: At) -> Result<(), Refused> {
            self.seen.push(Item::Front(at));
            Ok(())
        }

        fn shadow(&mut self, _port: &mut RecPort, at: At) -> Result<(), Refused> {
            if self.layers[at.layer].host_leg() {
                self.seen.push(Item::Shadow(at));
            }
            Ok(())
        }

        fn back(&mut self, _port: &mut RecPort, at: At) -> Result<(), Refused> {
            if self.layers[at.layer].host_leg() {
                self.seen.push(Item::Back(at));
            }
            Ok(())
        }
    }

    /// The one-token step over GLM's 45 layers: the dense layers are their
    /// fronts alone, every routing layer's shadow and back follow its own
    /// front before the next layer's front, and the legs come in layer
    /// order, 3 to 44 — the order the host tier serves.
    #[test]
    fn glm_step_walk() {
        let layers: Vec<Layer> = glm().iter().map(Layer::of).collect();
        let mut prog = Rec {
            layers: layers.clone(),
            seen: Vec::new(),
        };
        let o = Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        };
        sched::walk(o, layers.len(), &mut RecPort, &mut prog).unwrap();
        let mut want = Vec::new();
        for (l, k) in layers.iter().enumerate() {
            let at = At { unit: 0, layer: l };
            want.push(Item::Front(at));
            if k.host_leg() {
                want.extend([Item::Shadow(at), Item::Back(at)]);
            }
        }
        assert_eq!(prog.seen, want);
        let legs: Vec<usize> = prog
            .seen
            .iter()
            .filter_map(|i| match i {
                Item::Back(at) => Some(at.layer),
                _ => None,
            })
            .collect();
        assert_eq!(legs, (3..45).collect::<Vec<_>>());
    }
}
