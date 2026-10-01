//! Qwen3.8's expert tier ([`crate::host::tier`]): a card beside the stage
//! card that holds some of each routed layer's experts — plan (b′)'s 3090,
//! the next ids after the stage card's prefix — and computes the routed
//! slots the slot map sends it, in the host leg's shadow.
//!
//! The stage card's side is [`TierSide38`]: the slot map's tier view, which
//! the tier handoff (`ds41_ffn_handoff_10_cols_tier`) reads each slot's tier
//! place from, the unit's tier places, which the join
//! (`q38_card_tier_shared_add`, `card38`) reads, and each layer's tier
//! count. A row's tier image carries the normed activation in f32
//! ([`TierAct::F32`]), a column a position: the gate·up `_sel` reads the
//! q8_1 planes of Walk A, which the stage card makes in the shadow after the
//! go, and the tier quantizes the same values with the same kernel. The
//! page's row carries up to [`VERIFY_ROWS`] columns: the step's one and a
//! verify's `Cols(m)`.
//!
//! The tier's own layer ([`Tier38`], the architecture's [`TierExperts`]) is
//! the stage card's card leg over the tier's stacks ([`SelRun`]: the q8_1 of
//! the columns, the gate·up `_sel` of the layer's type with SiLU·mul, the
//! 32-value q8_1 of the tier slots' columns, the down `_sel` of its type)
//! into the row's routed rows, slot-major. Each slot's down output is the one
//! the stage card would write for it, so the join over both cards' slots in
//! slot order is the one-card sum over their union. A prompt batch's block
//! has no tier leg yet: the pass and the ubatch walk refuse a tier map by
//! name (`card38`'s `MapCheck`), so a load with a tier feeds its prompt by
//! steps; [`Tier38::block_bytes`] is the route scratch the plan reserves for
//! that leg on the tier card, which the load does not allocate, and
//! [`Tier38::enqueue_block`] refuses every block by name.

use super::card38::{LegLayer, SelRun, leg_layers};
use super::plan38::geo;
use super::scratch38::VERIFY_ROWS;
use super::swap38::Qwen38Stacks;
use crate::host::tier::{
    TierAct, TierBlock, TierCard, TierExperts, TierInput, TierIo, TierOpen, TierSet, TierShape,
};
use crate::hybrid::SlotMap;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::placement::Plan;

/// What the tier's errors name.
const WHAT: &str = "qwen4exp tier";

/// The tier the stage card's tier entries serve: they hand one tier its
/// image and join one tier's rows, so a load holds one tier card
/// ([`crate::host::refuse_tier_count`]).
pub(super) const STAGE_TIER: usize = 0;

/// The stage card's side of the tier layers: the slot map's tier view (a row
/// of places per layer at the card copy's row offsets), the unit's tier
/// places of its routed slots (ten a column, [`VERIFY_ROWS`] columns), and
/// per layer the experts the tier holds. Built once at load by a body with a
/// tier card.
pub(super) struct TierSide38 {
    places: DeviceTensor<u32>,
    pub(super) tsel: DeviceBuffer<u32>,
    k: Vec<usize>,
}

impl TierSide38 {
    /// The side for `map`'s tier [`STAGE_TIER`] over its layers. Load-time
    /// only.
    pub(super) fn new(gpu: &Gpu, map: &SlotMap) -> Result<TierSide38, GpuError> {
        let stream = gpu.stream();
        let run = map.layers();
        let k = run
            .clone()
            .map(|l| map.on_tier_of(STAGE_TIER, l))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TierSide38 {
            places: DeviceTensor::upload(
                stream,
                &map.tier_view(STAGE_TIER)?,
                run.len(),
                map.n_expert(),
            )?,
            tsel: DeviceBuffer::zeroed(stream, VERIFY_ROWS * geo::N_USED)?,
            k,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(super) fn k(&self, l: usize) -> usize {
        self.k.get(l).copied().unwrap_or(0)
    }

    /// The layers the tier holds experts of.
    pub(super) fn layers(&self) -> usize {
        self.k.iter().filter(|&&k| k > 0).count()
    }

    /// What a tier layer's handoff reads and writes: the card copy of the
    /// map's tier view and the unit's tier places.
    pub(super) fn handoff_parts(&mut self) -> (&DeviceBuffer<u32>, &mut DeviceBuffer<u32>) {
        (self.places.buf(), &mut self.tsel)
    }

    /// Device bytes the side holds.
    pub(super) fn bytes(&self) -> usize {
        self.places.buf().num_bytes() + self.tsel.num_bytes()
    }
}

/// The Qwen3.8 tier card's computation ([`TierExperts`]): per tier layer the
/// stage card's card-leg launches over the tier's stacks into the row's
/// routed rows (module doc). Its scratch, for up to [`VERIFY_ROWS`]
/// columns, is made at load.
pub(super) struct Tier38 {
    run: SelRun,
    /// Per layer of the model, the tier's experts and stacks; `None` off
    /// the tier.
    layers: Vec<Option<LegLayer>>,
    /// The prompt-batch block scratch the plan reserves on the tier card
    /// (the card route's at the load's ubatch): a reserve for the block
    /// leg's route scratch, which this load does not allocate — no block
    /// runs ([`Tier38::enqueue_block`] refuses every one).
    block: usize,
}

impl Tier38 {
    /// The tier's computation on `gpu` over `set`, whose stacks `w` holds
    /// (each tier layer's routed gate, up and down with its experts in tier
    /// slot order), each layer's types as `stacks` lists them, for a model of
    /// `layers` layers; `block` the plan's block scratch for the tier. A tier
    /// layer whose stacks are absent, of a type the leg does not run or of
    /// other rows than its experts', and a layer off the tier with resident
    /// stacks, are refused by name. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        stacks: &Qwen38Stacks,
        set: &TierSet,
        layers: usize,
        block: usize,
    ) -> Result<Tier38, GpuError> {
        let run = set.layers();
        let per = leg_layers(
            w,
            stacks,
            |l| {
                if run.contains(&l) {
                    set.on_tier(l)
                } else {
                    Ok(0)
                }
            },
            layers,
        )?;
        Ok(Tier38 {
            run: SelRun::new(gpu, VERIFY_ROWS)?,
            layers: per,
            block,
        })
    }

    /// Layer `layer`'s tier experts; a layer the tier holds none of is
    /// refused by name.
    fn layer(&self, layer: usize) -> Result<&LegLayer, GpuError> {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} holds no tier expert")))
    }
}

impl TierExperts for Tier38 {
    /// The stage card's card-leg launches of the go's columns over the
    /// tier's stacks but the card sum: the staged activation's q8_1, the
    /// gate·up `_sel`, the q8_1 of the tier slots' columns and the down
    /// `_sel` into the row's routed rows.
    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        let TierInput::F32(x) = io.act else {
            return Err(GpuError::shape(
                WHAT,
                "a q8_1 tier image: Qwen3.8's tier quantizes the f32 activation itself",
            ));
        };
        let Tier38 { run, layers, .. } = self;
        let cl = layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} holds no tier expert")))?;
        run.enqueue(gpu, weights, (layer, cl), x, io.cols, io.sel, io.rows)
    }

    /// Refused by name: a prompt batch's block has no tier leg yet, and the
    /// walks that would hand one over refuse a tier map first (module doc).
    fn enqueue_block(
        &mut self,
        _: &Gpu,
        _: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        self.layer(layer)?;
        Err(GpuError::shape(
            WHAT,
            format!(
                "layer {layer}: a prompt block of {} columns; Qwen3.8's tier serves the step and \
                 the verify, and a load with a tier feeds its prompt by steps",
                io.cols
            ),
        ))
    }

    /// The block scratch the plan reserves on the tier card (the card
    /// route's at the load's ubatch): the reserve for the block leg's route
    /// scratch, not allocated, since no block runs.
    fn block_bytes(&self) -> usize {
        self.block
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load: its card found by name, its routed segments uploaded, its
/// set the map's rows of that tier, Qwen3.8's tier computation over them
/// (each layer's types as `stacks` lists them, `layers` layers, the plan's
/// block scratch `block`), its page rows one of [`VERIFY_ROWS`] columns;
/// `card_dontneed` as for the stage card's segments. The stage card's
/// context is current again on return. Load-time only.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage card, the file, the plan, the tier and its index, the map, the stacks' types, the layers, the block scratch and the page lever (rust-quality R8)"
)]
pub(super) fn open_tier(
    stage: &Gpu,
    file: &Split,
    plan: &Plan<'_>,
    (t, tier): (&TierOpen, usize),
    map: &SlotMap,
    stacks: &Qwen38Stacks,
    layers: usize,
    block: usize,
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    if map.n_expert() != geo::EXPERTS {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a map of {} experts; the tier is built for {}",
                map.n_expert(),
                geo::EXPERTS
            ),
        ));
    }
    let gpu = Gpu::for_card(&t.name)?;
    let w = Weights::load_placed(gpu.stream(), file, plan, t.card, card_dontneed)?;
    let set = TierSet::of_map(map, tier)?;
    let experts = Tier38::new(&gpu, &w, stacks, &set, layers, block)?;
    let card = TierCard::open_cols(
        gpu,
        t.name.clone(),
        w,
        set,
        Box::new(experts),
        TierShape {
            hidden: geo::HIDDEN,
            n_used: geo::N_USED,
            rows: 1,
            act: TierAct::F32,
        },
        VERIFY_ROWS,
    )?;
    stage.context().bind_to_thread()?;
    Ok(card)
}
