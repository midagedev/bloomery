//! The card-leg expert tier: a card beside the stage card that holds some of
//! each routed layer's experts and computes the routed slots the slot map
//! sends it, in the host leg's shadow ([`TierExperts`]). Any family whose
//! routed experts run on the common card leg ([`CardLeg`]) uses it; the
//! family supplies the leg's shape (slots a token, tokens a run, widths) and
//! its description, and the node family and open of its own tier
//! ([`LegOpen`], [`open_tier`]).
//!
//! The tier's layer is the card leg ([`CardLeg::enqueue_rows`]) over the
//! tier's stacks: the q8_1 of the activation, the gate·up `_sel` with the
//! routed clamped SwiGLU, the q8_1 of the tier slots' columns and the down
//! `_sel` into the tier's rows; over a prompt batch's block, the same run
//! chunk by chunk of `shape.cap` tokens. Each slot's down output is the one
//! the stage card would write for it, so the card sum over both cards' slots
//! in slot order is the one-card sum over their union. The row's tier image
//! carries the normed activation in f32 ([`TierAct::F32`]): the tier
//! quantizes it with the kernel the stage card quantizes with.

use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use gguf::Split;
use model::arch::models::{Ffn, LayerSpec};
use model::arch::names::routed_stacks;
use model::placement::Plan;

use super::node::{NodeFamily, NodeOpen};
use super::tier::{
    TierAct, TierBlock, TierCard, TierExperts, TierInput, TierIo, TierOpen, TierSet, TierShape,
};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks};
use crate::hybrid::SlotMap;
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{Gpu, GpuError, Window, WindowMut};

/// What the tier's load errors name.
const WHAT: &str = "LegTier";

/// A tier layer's experts, its routed SwiGLU limit and its stacks' names.
struct TierLayer {
    k: usize,
    limit: f32,
    names: [String; 3],
}

impl TierLayer {
    /// The stacks' names, as the card leg reads them.
    fn names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
}

/// The card-leg tier's computation ([`TierExperts`]): per tier layer the card
/// leg over the tier's stacks into the tier's rows; over a prompt batch's
/// block, the same chunk by chunk. Its scratch, one [`LegBufs`] for
/// `shape.cap` tokens that a step's one token and each chunk share, is made
/// at load.
pub struct LegTier {
    leg: CardLeg,
    bufs: LegBufs,
    layers: Range<usize>,
    /// Per layer of `layers`, `None` off the tier.
    cfg: Vec<Option<TierLayer>>,
    shape: LegShape,
}

impl LegTier {
    /// The tier's computation on `gpu` over `set`, whose stacks `w` holds,
    /// for the description `layers`, through the leg of `shape` (`cap` tokens
    /// of `slots_per_col` slots a run, `n` wide, experts `ff` wide). A tier
    /// layer whose three stacks are absent, of a type the card experts do not
    /// read, or of other rows than its experts', and a layer off the tier
    /// with resident stacks, are refused by name. Load-time only.
    pub fn new(
        gpu: &Gpu,
        layers: &[LayerSpec],
        set: &TierSet,
        w: &Weights,
        shape: LegShape,
    ) -> Result<LegTier, GpuError> {
        let run = set.layers();
        let mut cfg = Vec::with_capacity(run.len());
        for l in run.clone() {
            let k = set.on_tier(l)?;
            let names = routed_stacks(l);
            if k == 0 {
                if let Some(n) = names.iter().find(|n| w.get(n).is_some()) {
                    return Err(GpuError::Tensor {
                        what: WHAT,
                        name: n.clone(),
                        need: "no resident rows: the tier holds no expert of its layer",
                    });
                }
                cfg.push(None);
                continue;
            }
            let Some(Ffn::Moe(m)) = layers.get(l).map(|s| &s.ffn) else {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {l}: tier experts off a routed block"),
                });
            };
            let layer = TierLayer {
                k,
                limit: m.act.swiglu_limit(),
                names,
            };
            LegStacks::of(w, layer.names())?.check(k, shape)?;
            cfg.push(Some(layer));
        }
        Ok(LegTier {
            leg: CardLeg::load(gpu)?,
            bufs: LegBufs::new(gpu.stream(), shape)?,
            layers: run,
            cfg,
            shape,
        })
    }
}

/// Layer `layer`'s tier experts of `cfg` (the tier's layers from
/// `layers.start`); a layer the tier holds none of is refused by name as
/// `what`.
fn layer_of<'a>(
    layers: &Range<usize>,
    cfg: &'a [Option<TierLayer>],
    layer: usize,
    what: &'static str,
) -> Result<&'a TierLayer, GpuError> {
    layer
        .checked_sub(layers.start)
        .and_then(|i| cfg.get(i))
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {layer} holds no tier expert"),
        })
}

impl TierExperts for LegTier {
    /// The scratch of a run of `shape.cap` tokens: the leg's buffers.
    fn block_bytes(&self) -> usize {
        self.bufs.bytes()
    }

    /// Per chunk of up to `shape.cap` tokens, the card leg over the tier's
    /// stacks into the chunk's slots of `io.down`.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        const W: &str = "LegTier::enqueue_block";
        let LegTier {
            leg,
            bufs,
            layers,
            cfg,
            shape,
        } = self;
        let LegShape {
            cap,
            slots_per_col,
            n,
            ..
        } = *shape;
        let tl = layer_of(layers, cfg, layer, W)?;
        let stacks = LegStacks::of(weights, tl.names())?;
        for c0 in (0..io.cols).step_by(cap) {
            let cn = cap.min(io.cols - c0);
            let slots = cn * slots_per_col;
            let x = Window::<f32>::of(io.x, c0 * n * size_of::<f32>(), cn * n)?;
            let sel = Window::<u32>::of(io.sel, c0 * slots_per_col * size_of::<u32>(), slots)?;
            let mut down = WindowMut::<f32>::of_mut(
                &mut *io.down,
                c0 * slots_per_col * n * size_of::<f32>(),
                slots * n,
            )?;
            let run = LegRun {
                layer,
                stacks: &stacks,
                normed: &x,
                m: cn,
                sel: &sel,
                n_card: tl.k,
                rule: Act::SwigluClamp { limit: tl.limit },
            };
            leg.enqueue_rows(gpu, bufs, &run, &mut down)?;
        }
        Ok(())
    }

    /// The card leg of one token over the tier's stacks into the row's
    /// routed rows, from the staged activation.
    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        const W: &str = "LegTier::enqueue_layer";
        let tl = layer_of(&self.layers, &self.cfg, layer, W)?;
        let TierInput::F32(x) = io.act else {
            return Err(GpuError::Shape {
                what: W,
                detail: "a q8_1 tier image: the leg tier quantizes the f32 activation itself"
                    .to_string(),
            });
        };
        let stacks = LegStacks::of(weights, tl.names())?;
        let run = LegRun {
            layer,
            stacks: &stacks,
            normed: x,
            m: 1,
            sel: io.sel,
            n_card: tl.k,
            rule: Act::SwigluClamp { limit: tl.limit },
        };
        self.leg.enqueue_rows(gpu, &mut self.bufs, &run, io.rows)
    }
}

/// What a family's tier open fixes beyond its node family: the stage card's
/// open puts it in the node's [`NodeOpen`] ([`open_tier`]) and the node's
/// family reads it back ([`LegOpen::check`]).
#[derive(Clone, Copy, Debug)]
pub struct LegOpen {
    /// Routed experts a layer of the model holds.
    pub n_expert: usize,
    /// Layers of the description.
    pub layers: usize,
    /// The activation's width.
    pub hidden: usize,
    /// Routed slots a token takes.
    pub n_used: usize,
    /// Page rows: the rows in flight.
    pub rows: usize,
}

impl LegOpen {
    /// The tier shape of the open: the f32 activation, one column a row.
    fn shape(&self) -> TierShape {
        TierShape {
            hidden: self.hidden,
            n_used: self.n_used,
            rows: self.rows,
            act: TierAct::F32,
        }
    }

    /// `open` is this tier's open, or a refusal naming `what`.
    pub fn check(&self, what: &'static str, open: &NodeOpen) -> Result<(), GpuError> {
        if open.shape != self.shape()
            || open.set.n_expert() != self.n_expert
            || open.layers != self.layers
            || open.cols != 1
        {
            return Err(GpuError::shape(
                what,
                "the node tier's shape is not this file's",
            ));
        }
        Ok(())
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load, opened in this process or on its node through the common
/// node owner ([`NodeOpen::open_tier`]): its set the map's rows of that
/// tier, the family's tier computation over them (`family`, the open
/// `open`), `card_dontneed` as for the stage card's segments. The stage
/// card's context is current again on return. Load-time only.
pub fn open_tier(
    stage: &Gpu,
    file: &Arc<Split>,
    plan: &Plan<'_>,
    (t, tier): (&TierOpen, usize),
    map: &SlotMap,
    (family, open): (&NodeFamily, LegOpen),
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    if map.n_expert() != open.n_expert {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a map of {} experts; the {} tier is built for {}",
                map.n_expert(),
                family.word,
                open.n_expert
            ),
        ));
    }
    NodeOpen {
        nvme: None,
        family: family.word.to_string(),
        name: t.name.clone(),
        device: t.device,
        card: t.card,
        rows: NodeOpen::rows_of(plan, file, t.card)?,
        set: TierSet::of_map(map, tier)?,
        shape: open.shape(),
        cols: 1,
        layers: open.layers,
        block: (0, 0),
    }
    .open_tier(stage, file, plan, t.node, family, card_dontneed)
}
