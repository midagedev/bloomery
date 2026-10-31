//! What the Qwen3.6 whole-model gate (`gate_qwen35moe_e2e`) reads from the file it opens instead of
//! carrying the real file's numbers: each layer's kind and the types of the sites its launches
//! depend on, and the captured graphs' node counts as sums of per-layer launches under them. The
//! kinds and the types are the file's header (the layer kinds the model crate's header reader
//! gives, the tensor table's types); nothing here reads a loaded body. The family's own names stay
//! with the gate that reads the file: this module takes a layer's kind and a lookup of site
//! types, so it holds only the launch arithmetic and its refusals. The tier the gate runs in, the
//! card budget a plan is made under and the move-proof print belong to `crate::tier`.
//!
//! A layer has one of two launch layouts, each a property of its header and nothing else:
//! - **joined**: Q4_K mixer sites (the `attn_qkv` of a delta layer and the `attn_v` of an
//!   attention layer may be Q6_K) and a shared expert of its routed stacks' types;
//! - **unjoined**: Q8_0 mixer sites but the β and α projections, F32, and a shared expert apart
//!   from the routed stacks (every part of another type than its stack's).
//!
//! A layer that matches neither is a named error, and so is a joined mixer beside an apart
//! shared expert (or the reverse): no count is derived for a layout nobody has counted.

use crate::GateError;
use gguf::quant::GgmlType;

/// Launches of the embedding row.
const EMBED_LAUNCHES: usize = 1;
/// Launches of the decode step's head.
const HEAD_LAUNCHES: usize = 3;
/// Launches of a joined delta layer's mixer: the norm with its quantizer, the q·k·v projection and
/// the z·β·α projection (a Q6_K q·k·v in its own gemv), the conv, the delta step, the gated norm,
/// the q8_1 and `ssm_out` with the residual.
const DELTA_MIXER_LAUNCHES: usize = 8;
/// Launches of a joined attention layer's mixer: the norm with its quantizer, q·k·v, rope, the
/// flash segment pass and merge, the gated q8_1 and the output projection with the residual.
const ATTN_MIXER_LAUNCHES: usize = 7;
/// Launches of a joined layer's feed-forward block: the norm with the gated router, the gate·up
/// over the joined stacks, the q8_1 of the nine slots, the down `_sel` and the combine.
const JOINED_FFN_LAUNCHES: usize = 5;
/// Launches a Q6_K `attn_v` adds to a joined attention layer's mixer at one row: its own gemv.
const ATTN_V_GEMV_LAUNCHES: usize = 1;
/// Launches a Q6_K `attn_qkv` adds to a joined delta layer's mixer in a pass of two or more rows:
/// the token-major copy of its output.
const PASS_QKV_COPY_LAUNCHES: usize = 1;
/// Launches a Q6_K `attn_v` adds to a joined attention layer's mixer in a pass of two or more
/// rows: its gemv and the token-major copy.
const PASS_V_LAUNCHES: usize = 2;
/// Launches of an unjoined layer's mixer, either kind: the norm with its quantizer, each of the
/// layer's projections in its own gemv, and for a delta layer the conv, the delta step, the gated
/// norm and the output projection into `normed` with the residual add; no site is row-major past
/// one row, so a pass holds the same launches as a row.
const UNJOINED_MIXER_LAUNCHES: usize = 10;
/// Launches of an apart shared expert's feed-forward block: the norm and the router on the f32
/// rows its gate and up read, the card list, the routed gate·up, the routed slots' quantizer, the
/// routed down `_sel`, the shared gate and up, the SwiGLU, the shared down and the host-sum
/// combine.
const APART_FFN_LAUNCHES: usize = 11;
/// Launches one busy slot adds to a captured pass of several slots on an attention layer over f16
/// planes (the flash segment pass and merge); its rope with the cache append, and a delta layer's
/// conv and delta step, run once over every slot's rows and add none.
const SLOT_ATTN_LAUNCHES: usize = 2;

use model::arch::qwen35moe::hparams::Kind;

/// A tensor of a layer whose type a launch rule reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    AttnQkv,
    AttnGate,
    SsmAlpha,
    SsmBeta,
    SsmOut,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    GateExps,
    UpExps,
    DownExps,
    GateShexp,
    UpShexp,
    DownShexp,
}

/// The types of a layer's mixer sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mixer {
    Delta {
        qkv: GgmlType,
        gate: GgmlType,
        alpha: GgmlType,
        beta: GgmlType,
        out: GgmlType,
    },
    Attention {
        q: GgmlType,
        k: GgmlType,
        v: GgmlType,
        out: GgmlType,
    },
}

/// Whether a layer's shared expert has the types of its routed stacks (joined into them) or any
/// part of another type (apart).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shared {
    Joined,
    Apart,
}

/// One layer as its header describes it: what the launch rules read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layer {
    mixer: Mixer,
    shared: Shared,
}

impl Layer {
    /// The layer's kind.
    #[must_use]
    fn kind(&self) -> Kind {
        match self.mixer {
            Mixer::Delta { .. } => Kind::DeltaRule,
            Mixer::Attention { .. } => Kind::Attention,
        }
    }
}

/// The launch rule a layer matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// The joined layout; `q6` when the layer's `attn_qkv` (delta) or `attn_v` (attention) is
    /// Q6_K.
    Joined { q6: bool },
    /// The unjoined layout.
    Unjoined,
}

/// The rule `layer` matches, `None` when its sites match none.
fn rule_of(layer: &Layer) -> Option<Rule> {
    use GgmlType::{F32, Q4_K, Q6_K, Q8_0};
    match (layer.mixer, layer.shared) {
        (
            Mixer::Delta {
                qkv,
                gate: Q4_K,
                alpha: Q4_K,
                beta: Q4_K,
                out: Q4_K,
            },
            Shared::Joined,
        ) if matches!(qkv, Q4_K | Q6_K) => Some(Rule::Joined { q6: qkv == Q6_K }),
        (
            Mixer::Attention {
                q: Q4_K,
                k: Q4_K,
                v,
                out: Q4_K,
            },
            Shared::Joined,
        ) if matches!(v, Q4_K | Q6_K) => Some(Rule::Joined { q6: v == Q6_K }),
        (
            Mixer::Delta {
                qkv: Q8_0,
                gate: Q8_0,
                alpha: F32,
                beta: F32,
                out: Q8_0,
            },
            Shared::Apart,
        )
        | (
            Mixer::Attention {
                q: Q8_0,
                k: Q8_0,
                v: Q8_0,
                out: Q8_0,
            },
            Shared::Apart,
        ) => Some(Rule::Unjoined),
        _ => None,
    }
}

/// The layers of a file as its header describes them, each under the launch rule it matches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    layers: Vec<(Layer, Rule)>,
}

impl Shape {
    /// The shape of `layers`.
    ///
    /// # Errors
    /// No layer, or a layer whose sites match no launch rule (the layer and its sites named).
    fn from_layers(layers: Vec<Layer>) -> Result<Shape, GateError> {
        if layers.is_empty() {
            return Err("a file of no layer has no shape".into());
        }
        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(l, layer)| {
                let rule = rule_of(&layer).ok_or_else(|| {
                    format!(
                        "layer {l}: no launch rule for {:?} with its shared expert {:?} (the \
                         joined layout: Q4_K sites, `attn_qkv` or `attn_v` Q4_K or Q6_K, shared \
                         expert joined; the unjoined layout: Q8_0 sites, β and α F32, shared \
                         expert apart)",
                        layer.mixer, layer.shared
                    )
                })?;
                Ok((layer, rule))
            })
            .collect::<Result<_, GateError>>()?;
        Ok(Shape { layers })
    }

    /// The shape of a header: layer `l` has `kinds[l]`, and `ty(l, site)` is the type of its
    /// `site`. A shared expert is apart when any of its three parts has another type than its
    /// routed stack's.
    ///
    /// # Errors
    /// A lookup that fails, or [`Shape::from_layers`]'s.
    pub fn from_header(
        kinds: &[Kind],
        mut ty: impl FnMut(usize, Site) -> Result<GgmlType, GateError>,
    ) -> Result<Shape, GateError> {
        let mut layers = Vec::with_capacity(kinds.len());
        for (l, kind) in kinds.iter().enumerate() {
            let mixer = match kind {
                Kind::DeltaRule => Mixer::Delta {
                    qkv: ty(l, Site::AttnQkv)?,
                    gate: ty(l, Site::AttnGate)?,
                    alpha: ty(l, Site::SsmAlpha)?,
                    beta: ty(l, Site::SsmBeta)?,
                    out: ty(l, Site::SsmOut)?,
                },
                Kind::Attention => Mixer::Attention {
                    q: ty(l, Site::AttnQ)?,
                    k: ty(l, Site::AttnK)?,
                    v: ty(l, Site::AttnV)?,
                    out: ty(l, Site::AttnOutput)?,
                },
            };
            let mut apart = false;
            for (exps, shexp) in [
                (Site::GateExps, Site::GateShexp),
                (Site::UpExps, Site::UpShexp),
                (Site::DownExps, Site::DownShexp),
            ] {
                apart |= ty(l, exps)? != ty(l, shexp)?;
            }
            layers.push(Layer {
                mixer,
                shared: if apart { Shared::Apart } else { Shared::Joined },
            });
        }
        Shape::from_layers(layers)
    }

    /// Layers.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    fn count(&self, kind: Kind) -> usize {
        self.layers
            .iter()
            .filter(|(layer, _)| layer.kind() == kind)
            .count()
    }

    /// Attention layers.
    #[must_use]
    pub fn n_attn(&self) -> usize {
        self.count(Kind::Attention)
    }

    /// Delta layers.
    #[must_use]
    pub fn n_delta(&self) -> usize {
        self.count(Kind::DeltaRule)
    }

    fn q6(&self, kind: Kind) -> usize {
        self.layers
            .iter()
            .filter(|(layer, rule)| {
                layer.kind() == kind && matches!(rule, Rule::Joined { q6: true })
            })
            .count()
    }

    /// Joined delta layers whose `attn_qkv` is Q6_K.
    #[must_use]
    pub fn q6_qkv(&self) -> usize {
        self.q6(Kind::DeltaRule)
    }

    /// Joined attention layers whose `attn_v` is Q6_K.
    #[must_use]
    pub fn q6_v(&self) -> usize {
        self.q6(Kind::Attention)
    }

    /// Per layer, whether its shared expert is apart from its routed stacks.
    #[must_use]
    pub fn apart(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|(layer, _)| layer.shared == Shared::Apart)
            .collect()
    }

    /// The captured decode step's nodes: the embedding row, each layer's mixer and feed-forward
    /// launches (a joined attention layer one more when its `attn_v` is Q6_K), then the head's.
    #[must_use]
    pub fn nodes_decode(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|(layer, rule)| match (rule, layer.kind()) {
                (Rule::Joined { .. }, Kind::DeltaRule) => {
                    DELTA_MIXER_LAUNCHES + JOINED_FFN_LAUNCHES
                }
                (Rule::Joined { q6 }, Kind::Attention) => {
                    ATTN_MIXER_LAUNCHES
                        + JOINED_FFN_LAUNCHES
                        + if *q6 { ATTN_V_GEMV_LAUNCHES } else { 0 }
                }
                (Rule::Unjoined, _) => UNJOINED_MIXER_LAUNCHES + APART_FFN_LAUNCHES,
            })
            .sum();
        EMBED_LAUNCHES + layers + HEAD_LAUNCHES
    }

    /// The nodes of a captured pass of two or more rows without its heads: the embedding row and
    /// each layer's launches, a joined layer's Q6_K site adding its gemv and copy.
    #[must_use]
    pub fn nodes_pass(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|(layer, rule)| match (rule, layer.kind()) {
                (Rule::Joined { q6 }, Kind::DeltaRule) => {
                    DELTA_MIXER_LAUNCHES
                        + JOINED_FFN_LAUNCHES
                        + if *q6 { PASS_QKV_COPY_LAUNCHES } else { 0 }
                }
                (Rule::Joined { q6 }, Kind::Attention) => {
                    ATTN_MIXER_LAUNCHES
                        + JOINED_FFN_LAUNCHES
                        + if *q6 { PASS_V_LAUNCHES } else { 0 }
                }
                (Rule::Unjoined, _) => UNJOINED_MIXER_LAUNCHES + APART_FFN_LAUNCHES,
            })
            .sum();
        EMBED_LAUNCHES + layers
    }

    /// The nodes one more busy slot adds to a captured pass of several slots' rows over f16
    /// planes: the embedding row and each attention layer's sequence-bound launches. Defined for
    /// joined layers only.
    ///
    /// # Errors
    /// A layer that keeps its shared expert apart (the first one named): no slot count is pinned
    /// on such a file.
    pub fn nodes_slot(&self) -> Result<usize, GateError> {
        if let Some(l) = self
            .layers
            .iter()
            .position(|(layer, _)| layer.shared == Shared::Apart)
        {
            return Err(format!(
                "layer {l} keeps its shared expert apart: the slot count is defined for joined \
                 layers only"
            )
            .into());
        }
        Ok(EMBED_LAUNCHES + self.n_attn() * SLOT_ATTN_LAUNCHES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use GgmlType::{F32, Q4_K, Q5_K, Q6_K, Q8_0};

    fn delta(qkv: GgmlType) -> Layer {
        Layer {
            mixer: Mixer::Delta {
                qkv,
                gate: Q4_K,
                alpha: Q4_K,
                beta: Q4_K,
                out: Q4_K,
            },
            shared: Shared::Joined,
        }
    }

    fn attn(v: GgmlType) -> Layer {
        Layer {
            mixer: Mixer::Attention {
                q: Q4_K,
                k: Q4_K,
                v,
                out: Q4_K,
            },
            shared: Shared::Joined,
        }
    }

    fn delta_ud() -> Layer {
        Layer {
            mixer: Mixer::Delta {
                qkv: Q8_0,
                gate: Q8_0,
                alpha: F32,
                beta: F32,
                out: Q8_0,
            },
            shared: Shared::Apart,
        }
    }

    fn attn_ud() -> Layer {
        Layer {
            mixer: Mixer::Attention {
                q: Q8_0,
                k: Q8_0,
                v: Q8_0,
                out: Q8_0,
            },
            shared: Shared::Apart,
        }
    }

    /// The real Q4_K_M file's header (40 layers at interval 4): the 14 delta layers with a Q6_K
    /// `attn_qkv` and the 6 attention layers with a Q6_K `attn_v`.
    fn q4km() -> Vec<Layer> {
        const QKV: [usize; 14] = [0, 1, 2, 4, 10, 13, 16, 22, 25, 28, 34, 36, 37, 38];
        const V: [usize; 6] = [3, 7, 19, 31, 35, 39];
        (0..40)
            .map(|l| {
                if (l + 1) % 4 == 0 {
                    attn(if V.contains(&l) { Q6_K } else { Q4_K })
                } else {
                    delta(if QKV.contains(&l) { Q6_K } else { Q4_K })
                }
            })
            .collect()
    }

    /// The real UD-Q4_K_M file's header: 40 layers at interval 4, every one unjoined.
    fn ud() -> Vec<Layer> {
        (0..40)
            .map(|l| {
                if (l + 1) % 4 == 0 {
                    attn_ud()
                } else {
                    delta_ud()
                }
            })
            .collect()
    }

    /// The Qwen3.6 fixture's header: delta, attention, delta, attention, the `attn_qkv` of the
    /// first delta layer Q4_K and the second Q6_K, the `attn_v` of the first attention layer
    /// Q4_K and the second Q6_K.
    fn fixture() -> Vec<Layer> {
        vec![delta(Q4_K), attn(Q4_K), delta(Q6_K), attn(Q6_K)]
    }

    /// The unjoined variants' header: the fixture's four layers with UD's sites.
    fn variant() -> Vec<Layer> {
        vec![delta_ud(), attn_ud(), delta_ud(), attn_ud()]
    }

    /// The real files' headers give the literals the gate carried; the fixture's and the
    /// variants' give the counts their derivations name. Mutant: any launch constant, a Q6_K rule
    /// (the decode step counting the delta layer's, the pass dropping the attention layer's two)
    /// or the kind split.
    #[test]
    fn the_headers_give_the_real_files_literals_and_the_fixtures_counts() {
        let real = Shape::from_layers(q4km()).expect("q4_k_m");
        assert_eq!(
            (real.n_layer(), real.n_attn(), real.n_delta()),
            (40, 10, 30)
        );
        assert_eq!((real.q6_qkv(), real.q6_v()), (14, 6));
        assert_eq!(real.nodes_decode(), 1 + 30 * 13 + 10 * 12 + 6 + 3);
        assert_eq!(real.nodes_decode(), 520);
        assert_eq!(real.nodes_pass(), 1 + 30 * 13 + 14 + 10 * 12 + 2 * 6);
        assert_eq!(real.nodes_pass(), 537);
        assert_eq!(real.nodes_slot().expect("joined"), 1 + 10 * 2);
        assert_eq!(real.nodes_slot().expect("joined"), 21);
        assert_eq!(real.apart(), vec![false; 40]);

        let ud = Shape::from_layers(ud()).expect("ud");
        assert_eq!((ud.n_layer(), ud.n_attn(), ud.n_delta()), (40, 10, 30));
        assert_eq!((ud.q6_qkv(), ud.q6_v()), (0, 0));
        assert_eq!(ud.nodes_decode(), 1 + 40 * (10 + 11) + 3);
        assert_eq!(ud.nodes_decode(), 844);
        assert_eq!(ud.nodes_pass(), 1 + 40 * 21);
        assert_eq!(ud.nodes_pass(), 841);
        assert_eq!(ud.apart(), vec![true; 40]);

        let fix = Shape::from_layers(fixture()).expect("fixture");
        assert_eq!((fix.n_layer(), fix.n_attn(), fix.n_delta()), (4, 2, 2));
        assert_eq!((fix.q6_qkv(), fix.q6_v()), (1, 1));
        assert_eq!(fix.nodes_decode(), 1 + 2 * 13 + (12 + 13) + 3);
        assert_eq!(fix.nodes_decode(), 55);
        assert_eq!(fix.nodes_pass(), 1 + (13 + 14) + (12 + 14));
        assert_eq!(fix.nodes_pass(), 54);
        assert_eq!(fix.nodes_slot().expect("joined"), 1 + 2 * 2);
        assert_eq!(fix.nodes_slot().expect("joined"), 5);

        let var = Shape::from_layers(variant()).expect("variant");
        assert_eq!(var.nodes_decode(), 1 + 4 * 21 + 3);
        assert_eq!(var.nodes_decode(), 88);
        assert_eq!(var.nodes_pass(), 1 + 4 * 21);
        assert_eq!(var.nodes_pass(), 85);
    }

    /// A site type no rule covers, a joined mixer beside an apart shared expert and the reverse,
    /// a Q6_K site off the two sites that may be, and a file of no layer are each refused by name;
    /// a slot count on an apart layer is refused naming the layer. Mutant: a rule that takes
    /// Q5_K, or one that drops the shared-expert half of the match.
    #[test]
    fn a_layer_no_rule_covers_and_a_slot_count_on_an_apart_layer_are_refused_by_name() {
        let refused = |layers: Vec<Layer>| {
            Shape::from_layers(layers)
                .expect_err("a layer no rule covers")
                .to_string()
        };
        let q5 = refused(vec![delta(Q4_K), delta(Q5_K)]);
        assert!(q5.contains("layer 1: no launch rule"), "{q5}");
        assert!(q5.contains("qkv: Q5_K"), "{q5}");
        let v5 = refused(vec![attn(Q5_K)]);
        assert!(
            v5.contains("layer 0: no launch rule") && v5.contains("v: Q5_K"),
            "{v5}"
        );
        let mut gate6 = delta(Q4_K);
        if let Mixer::Delta { gate, .. } = &mut gate6.mixer {
            *gate = Q6_K;
        }
        assert!(refused(vec![gate6]).contains("gate: Q6_K"));
        let mut joined_ud = delta_ud();
        joined_ud.shared = Shared::Joined;
        assert!(refused(vec![joined_ud]).contains("shared expert Joined"));
        let mut apart_k = attn(Q4_K);
        apart_k.shared = Shared::Apart;
        assert!(refused(vec![apart_k]).contains("shared expert Apart"));
        assert!(refused(vec![]).contains("no layer"));

        let mixed = Shape::from_layers(vec![delta(Q4_K), delta_ud(), attn_ud()]).expect("both");
        let e = mixed.nodes_slot().expect_err("an apart layer").to_string();
        assert!(e.contains("layer 1 keeps its shared expert apart"), "{e}");
    }

    /// A header read through a lookup gives the shape of its layers; the shared expert is apart
    /// when any one part differs from its stack's (the down alone is enough); a failed lookup
    /// ends the read with its own error. Mutant: an apart rule that reads only the gate and up.
    #[test]
    fn a_header_read_gives_the_layers_and_an_apart_rule_over_all_three_parts() {
        let kinds = [
            Kind::DeltaRule,
            Kind::Attention,
            Kind::DeltaRule,
            Kind::Attention,
        ];
        let table = |l: usize, site: Site| {
            let (qkv, v) = (
                if l == 2 { Q6_K } else { Q4_K },
                if l == 3 { Q6_K } else { Q4_K },
            );
            Ok(match site {
                Site::AttnQkv => qkv,
                Site::AttnV => v,
                _ => Q4_K,
            })
        };
        assert_eq!(
            Shape::from_header(&kinds, table).expect("header"),
            Shape::from_layers(vec![delta(Q4_K), attn(Q4_K), delta(Q6_K), attn(Q6_K)])
                .expect("layers")
        );
        let unjoined = |_: usize, site: Site| {
            Ok(match site {
                Site::AttnQkv
                | Site::AttnGate
                | Site::SsmOut
                | Site::AttnQ
                | Site::AttnK
                | Site::AttnV
                | Site::AttnOutput
                | Site::GateShexp
                | Site::UpShexp
                | Site::DownShexp => Q8_0,
                Site::SsmAlpha | Site::SsmBeta => F32,
                Site::GateExps | Site::UpExps => Q4_K,
                Site::DownExps => Q5_K,
            })
        };
        let ud = Shape::from_header(&kinds, unjoined).expect("ud");
        assert_eq!(ud.apart(), [true; 4]);
        assert_eq!((ud.nodes_decode(), ud.nodes_pass()), (88, 85));
        let down_only = |l: usize, site: Site| match site {
            Site::DownShexp => Ok(Q6_K),
            _ => table(l, site),
        };
        let e = Shape::from_header(&kinds, down_only)
            .expect_err("a Q4_K mixer beside a shared down of another type")
            .to_string();
        assert!(
            e.contains("layer 0: no launch rule") && e.contains("shared expert Apart"),
            "{e}"
        );
        let missing = |l: usize, site: Site| match (l, site) {
            (1, Site::AttnK) => Err("the file holds no blk.1.attn_k.weight".into()),
            _ => table(l, site),
        };
        let e = Shape::from_header(&kinds, missing).expect_err("a missing tensor");
        assert_eq!(e.to_string(), "the file holds no blk.1.attn_k.weight");
    }
}
