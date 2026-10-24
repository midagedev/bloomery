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
//! routed clamped SwiGLU, the selected down form of the tier slots' columns
//! and the down `_sel` into the tier's rows; over a prompt batch's block, the same run
//! chunk by chunk of `shape.cap` tokens. Each slot's down output is the one
//! the stage card would write for it, so the card sum over both cards' slots
//! in slot order is the one-card sum over their union. The row's tier image
//! carries the normed activation in f32 ([`TierAct::F32`]): the tier
//! quantizes it with the kernel the stage card quantizes with.

use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use cuda_core::DeviceBuffer;
use gguf::Split;
use model::ModelError;
use model::arch::models::{Ffn, LayerSpec};
use model::arch::names::routed_stacks;
use model::moe::HostLayer;
use model::placement::accepts::{self, LegArms, Program};
use model::placement::kernels::{self, DownForms};
use model::placement::{ModelTensors, Plan};
use model::r8file::R8Source;

use super::node::{NodeFamily, NodeOpen};
use super::run::{HostRun, HostWidths};
use super::tier::{
    TierAct, TierBlock, TierCard, TierExperts, TierInput, TierIo, TierOpen, TierSet, TierShape,
};
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks};
use crate::hybrid::{Hybrid, SlotMap};
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError, Window, WindowMut};

/// What the tier's load errors name.
const WHAT: &str = "LegTier";

/// What the stage card's side of the tier layers names in its errors.
const SIDE: &str = "TierSide";

/// The selected program's launches admit the same stacks on both cards.
const STACK_ROLES: (&str, &str) = (
    "a resident routed gate or up stack the selected tier gate·up arm runs",
    "a resident routed down stack the selected tier down arm runs",
);

/// The tier the stage card's tier entries serve: they hand one tier its
/// image and join one tier's rows, so a load holds one tier card
/// ([`crate::host::refuse_tier_count`]).
pub const STAGE_TIER: usize = 0;

/// The stage card's side of the tier layers: the slot map's tier view (a row
/// of places per layer at the card copy's row offsets), each row's tier
/// places of its routed slots, and per layer of the description the experts
/// the tier holds. The tier handoff reads the view and writes a row's
/// places; the card sum over both cards' slots reads the places. Built once at
/// load by a body with a tier card.
pub struct TierSide {
    places: DeviceTensor<u32>,
    tsel: Vec<DeviceBuffer<u32>>,
    k: Vec<usize>,
}

impl TierSide {
    /// The side for `map`'s tier [`STAGE_TIER`] over `layers` layers of the
    /// description (a layer outside the map holds no tier expert), `rows`
    /// rows of `n_used` slots. Load-time only.
    pub fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        (rows, n_used): (usize, usize),
    ) -> Result<TierSide, GpuError> {
        let stream = gpu.stream();
        let run = map.layers();
        let k = (0..layers)
            .map(|l| {
                if run.contains(&l) {
                    map.on_tier_of(STAGE_TIER, l)
                } else {
                    Ok(0)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TierSide {
            places: DeviceTensor::upload(
                stream,
                &map.tier_view(STAGE_TIER)?,
                run.len(),
                map.n_expert(),
            )?,
            tsel: (0..rows)
                .map(|_| DeviceBuffer::zeroed(stream, n_used))
                .collect::<Result<Vec<_>, _>>()?,
            k,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    #[must_use]
    pub fn k(&self, l: usize) -> usize {
        self.k.get(l).copied().unwrap_or(0)
    }

    /// Some layer holds a tier expert.
    #[must_use]
    pub fn any(&self) -> bool {
        self.k.iter().any(|&k| k > 0)
    }

    /// The card copy of the map's tier view.
    #[must_use]
    pub fn places(&self) -> &DeviceBuffer<u32> {
        self.places.buf()
    }

    /// Row `row`'s tier places; a row past the load's is refused by name.
    pub fn tsel(&self, row: usize) -> Result<&DeviceBuffer<u32>, GpuError> {
        let n = self.tsel.len();
        self.tsel.get(row).ok_or_else(|| past_rows(n, row))
    }

    /// What row `row`'s tier handoff reads and writes: the tier view and
    /// the row's tier places; a row past the load's is refused by name.
    pub fn handoff_parts(
        &mut self,
        row: usize,
    ) -> Result<(&DeviceBuffer<u32>, &mut DeviceBuffer<u32>), GpuError> {
        let n = self.tsel.len();
        let tsel = self.tsel.get_mut(row).ok_or_else(|| past_rows(n, row))?;
        Ok((self.places.buf(), tsel))
    }

    /// Device bytes the side holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.places.buf().num_bytes() + self.tsel.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }
}

fn past_rows(n: usize, row: usize) -> GpuError {
    GpuError::Shape {
        what: SIDE,
        detail: format!("row {row} of the tier side's {n} rows"),
    }
}

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
    arms: LegArms,
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
        program: Program,
    ) -> Result<LegTier, GpuError> {
        let arms = accepts::leg_arms(program).ok_or(GpuError::state(
            WHAT,
            "the selected program's card leg arms",
        ))?;
        let mut downs = DownForms::NONE;
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
            let stacks = LegStacks::of_arms(w, layer.names(), arms, STACK_ROLES)?;
            stacks.check(k, shape)?;
            let (_, act) = kernels::down(stacks.down_ty()).ok_or(GpuError::state(
                WHAT,
                "the selected tier down's activation form",
            ))?;
            downs = downs.with(act);
            cfg.push(Some(layer));
        }
        Ok(LegTier {
            leg: CardLeg::load(gpu)?,
            bufs: LegBufs::for_downs(gpu.stream(), shape, downs)?,
            layers: run,
            cfg,
            shape,
            arms,
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
            arms,
        } = self;
        let LegShape {
            cap,
            slots_per_col,
            n,
            ..
        } = *shape;
        let tl = layer_of(layers, cfg, layer, W)?;
        let stacks = LegStacks::of_arms(weights, tl.names(), *arms, STACK_ROLES)?;
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
        let stacks = LegStacks::of_arms(weights, tl.names(), self.arms, STACK_ROLES)?;
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

/// The routed block's widths a family's description gives the tier.
#[derive(Clone, Copy, Debug)]
pub struct LegWidths {
    /// Routed experts a layer holds.
    pub n_expert: usize,
    /// Routed slots a token takes.
    pub n_used: usize,
    /// The model's width.
    pub n_embd: usize,
    /// A routed expert's width.
    pub ff: usize,
    /// The program whose card leg runs these experts.
    pub program: Program,
}

impl LegWidths {
    /// The card leg's shape for the tier: up to `cap` tokens of
    /// [`LegWidths::n_used`] slots, `n_embd` wide, through experts `ff` wide.
    #[must_use]
    pub const fn shape(&self, cap: usize) -> LegShape {
        LegShape {
            cap,
            slots_per_col: self.n_used,
            n: self.n_embd,
            ff: self.ff,
        }
    }
}

/// What a family gives the tier node service of its file: its placement's
/// reading of the description, and the host views of its routed layers. The
/// tier's computation is the card leg ([`LegTier`]) whatever the family;
/// [`node_family`] makes the family's [`NodeFamily`] from this.
pub trait LegFamily {
    /// What the node's errors name.
    const WHAT: &'static str;
    /// The family's word on the node protocol ([`NodeFamily::word`]).
    const WORD: &'static str;
    /// The tier page's rows: the rows in flight of a load.
    const ROWS: usize;
    /// Tokens one tier run covers: the card leg's capacity.
    const CHUNK: usize;
    /// The file as the family's placement reads it.
    type Inputs;

    /// `file` read as the family's placement reads it; its refusals are
    /// the family's.
    fn describe(file: &Split) -> Result<Self::Inputs, GpuError>;
    /// The placement's tensors of `inputs`, by role.
    fn into_model(inputs: Self::Inputs) -> ModelTensors;
    /// The description's layers.
    fn layers(inputs: &Self::Inputs) -> &[LayerSpec];
    /// The routed block's widths.
    fn widths(inputs: &Self::Inputs) -> LegWidths;
    /// The host views of the routed layers `run`, each from the family's
    /// host reader; a layer of the run that does not route is refused by
    /// name. Load-time only.
    fn routed(
        src: R8Source<'_>,
        inputs: &Self::Inputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError>;
}

/// The family `F`'s tier node service: the placement tensors of a file, the
/// tier's computation over the stacks the node loaded, and the host run of
/// its routed layers.
#[must_use]
pub const fn node_family<F: LegFamily>() -> NodeFamily {
    NodeFamily {
        word: F::WORD,
        tensors: node_tensors::<F>,
        experts: node_experts::<F>,
        host: node_host::<F>,
    }
}

fn node_tensors<F: LegFamily>(file: &Split) -> Result<ModelTensors, GpuError> {
    F::describe(file).map(F::into_model)
}

/// `file` as `F` reads it, refused unless `open` is a tier open for it.
fn node_inputs<F: LegFamily>(file: &Split, open: &NodeOpen) -> Result<F::Inputs, GpuError> {
    let inputs = F::describe(file)?;
    let w = F::widths(&inputs);
    LegOpen {
        n_expert: w.n_expert,
        layers: F::layers(&inputs).len(),
        hidden: w.n_embd,
        n_used: w.n_used,
        rows: F::ROWS,
    }
    .check(F::WHAT, open)?;
    Ok(inputs)
}

fn node_experts<F: LegFamily>(
    gpu: &Gpu,
    file: &Split,
    _: &ModelTensors,
    w: &Weights,
    open: &NodeOpen,
) -> Result<Box<dyn TierExperts>, GpuError> {
    let inputs = node_inputs::<F>(file, open)?;
    let widths = F::widths(&inputs);
    let shape = widths.shape(F::CHUNK);
    Ok(Box::new(LegTier::new(
        gpu,
        F::layers(&inputs),
        &open.set,
        w,
        shape,
        widths.program,
    )?))
}

fn node_host<F: LegFamily>(file: Arc<Split>, open: &NodeOpen) -> Result<HostRun, GpuError> {
    let inputs = node_inputs::<F>(&file, open)?;
    let run = runtime::layer::hosted(F::layers(&inputs))
        .map_err(|e| GpuError::shape(F::WHAT, format!("the node host run: {e}")))?;
    let w = F::widths(&inputs);
    HostRun::build(
        file,
        run.start,
        false,
        HostWidths {
            embd: w.n_embd,
            ff: w.ff,
            n_used: w.n_used,
        },
        |src| F::routed(src, &inputs, run),
    )
}

/// What a load hangs under its host tier: the machine's expert tier cards,
/// the family that opens them, the open's shape and the batch port's width.
pub struct Hang<'a> {
    /// The expert tier cards of the plan's machine ([`TierOpen::of_machine`]).
    pub tiers: &'a [TierOpen],
    /// The family's node service.
    pub family: &'a NodeFamily,
    /// What the family's open fixes.
    pub open: LegOpen,
    /// Tokens the batch port's blocks hold: the tier's batch legs are made
    /// for them at load.
    pub cols: usize,
    /// The release of the tier's file pages, as for the stage card's
    /// segments.
    pub card_dontneed: bool,
}

/// Open every expert tier card of `hang` for the stage card `gpu`'s load of
/// `plan` and hang them under `hybrid` ([`Hybrid::attach_tiers`]), the tier's
/// batch legs made and their bytes checked against the plan's reserve
/// ([`Hybrid::check_tier_reserves`]): the plan reserves them on the tier card
/// and the host, so they are made at load, not at the first prompt. The one
/// owner of a body's tier hang, after its [`Hybrid`] is made and before its
/// arena leg. Nothing without a tier. Refused by name: more tier cards than
/// the host tier serves, a map that names other tiers. Load-time only.
pub fn hang_tiers(
    (gpu, file, plan): (&Gpu, &Arc<Split>, &Plan<'_>),
    hang: &Hang<'_>,
    hybrid: &mut Hybrid<HostRun>,
) -> Result<(), GpuError> {
    if hang.tiers.is_empty() {
        return Ok(());
    }
    let cards = hang
        .tiers
        .iter()
        .enumerate()
        .map(|(i, t)| {
            open_tier(
                gpu,
                file,
                plan,
                (t, i),
                hybrid.slots(),
                (hang.family, hang.open),
                hang.card_dontneed,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let planned: Vec<usize> = hang.tiers.iter().map(|t| t.card).collect();
    hybrid.attach_tiers(cards, gpu)?;
    hybrid.prepare_batch(gpu.context(), hang.cols)?;
    hybrid.check_tier_reserves(plan, &planned)
}
