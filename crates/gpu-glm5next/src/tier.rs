//! The GLM-5.3-Flash expert tier ([`bloomery_gpu::host::leg_tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//!
//! The stage card's side is [`TierSide`]: the slot map's tier view, which the
//! tier handoff (`ds41_ffn_handoff_8_tier`) reads each slot's tier place
//! from, and each row's tier places, which the tier card sum
//! (`ds41_ffn_card_acc_8_tier`) reads. The row's tier image carries the
//! normed activation in f32: GLM's `_sel` entries read
//! the q8_1 planes of Walk A, which the stage card makes in the shadow,
//! after the go; the tier quantizes the same values with the same kernel.
//!
//! The tier's own layer is the common [`LegTier`] over GLM's leg shape
//! ([`leg_shape`]): [`N_USED`] slots a token, [`CHUNK`] tokens a run. What is
//! GLM's here is the node family ([`NODE_FAMILY`]), which reads the file's
//! description, and the open of the tier card ([`open_tier`]).

use std::sync::Arc;

use bloomery_gpu::card_leg::LegShape;
use bloomery_gpu::host::leg_tier::{self, LegOpen, LegTier};
use bloomery_gpu::host::node::{NodeFamily, NodeOpen};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::tier::{TierCard, TierExperts, TierOpen};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::arch::glm5next::place::PlanInputs;
use model::placement::{ModelTensors, Plan};
use models::LayerSpec;

use crate::body::prefill::CHUNK;

/// What the tier's errors name.
const WHAT: &str = "glm5next tier";

/// GLM's architecture facts for the common tier node service.
pub const NODE_FAMILY: NodeFamily = NodeFamily {
    word: "glm5next",
    tensors: node_tensors,
    experts: node_experts,
    host: node_host,
};

fn node_tensors(file: &Split) -> Result<ModelTensors, GpuError> {
    PlanInputs::describe(file)
        .map(|i| i.model)
        .map_err(|e| GpuError::plan(WHAT, e))
}

fn node_inputs(file: &Split, open: &NodeOpen) -> Result<PlanInputs, GpuError> {
    let inputs = PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))?;
    LegOpen {
        n_expert: inputs.hp.n_expert,
        layers: inputs.spec.layers.len(),
        hidden: inputs.hp.n_embd,
        n_used: inputs.hp.n_used,
        rows: crate::LOAD_ROWS,
    }
    .check(WHAT, open)?;
    Ok(inputs)
}

fn node_experts(
    gpu: &Gpu,
    file: &Split,
    _: &ModelTensors,
    w: &Weights,
    open: &NodeOpen,
) -> Result<Box<dyn TierExperts>, GpuError> {
    let inputs = node_inputs(file, open)?;
    Ok(Box::new(LegTier::new(
        gpu,
        &inputs.spec.layers,
        &open.set,
        w,
        leg_shape(inputs.hp.n_embd, inputs.hp.expert_ff),
    )?))
}

fn node_host(file: Arc<Split>, open: &NodeOpen) -> Result<HostRun, GpuError> {
    let inputs = node_inputs(&file, open)?;
    let run = runtime::layer::hosted(&inputs.spec.layers)
        .map_err(|e| GpuError::shape(WHAT, format!("the node host run: {e}")))?;
    HostRun::build(
        file,
        run.start,
        false,
        HostWidths {
            embd: inputs.hp.n_embd,
            ff: inputs.hp.expert_ff,
            n_used: inputs.hp.n_used,
        },
        |src| crate::host::routed_layers(src, &inputs.hp, run),
    )
}

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(model::arch::glm5next::place::TIER_CHUNK as usize == CHUNK);

/// The tier the stage card's tier entries serve: they hand one tier its
/// image and join one tier's rows, so a load holds one tier card
/// ([`bloomery_gpu::host::refuse_tier_count`]).
pub(crate) const STAGE_TIER: usize = 0;

/// The stage card's side of the tier layers: the slot map's tier view (a row
/// of places per layer at the card copy's row offsets), each row's tier
/// places of its routed slots, and per layer of the description the experts
/// the tier holds. Built once at load by a body with a tier card.
pub(crate) struct TierSide {
    places: DeviceTensor<u32>,
    tsel: Vec<DeviceBuffer<u32>>,
    k: Vec<usize>,
}

impl TierSide {
    /// The side for `map`'s tier [`STAGE_TIER`] over `layers` layers of the
    /// description (a layer outside the map holds no tier expert), `rows`
    /// rows. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        rows: usize,
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
                .map(|_| DeviceBuffer::zeroed(stream, N_USED))
                .collect::<Result<Vec<_>, _>>()?,
            k,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(crate) fn k(&self, l: usize) -> usize {
        self.k.get(l).copied().unwrap_or(0)
    }

    /// The card copy of the map's tier view.
    pub(crate) fn places(&self) -> &DeviceBuffer<u32> {
        self.places.buf()
    }

    /// Row `row`'s tier places; a row past the load's is refused by name.
    pub(crate) fn tsel(&self, row: usize) -> Result<&DeviceBuffer<u32>, GpuError> {
        let n = self.tsel.len();
        self.tsel.get(row).ok_or_else(|| past_rows(n, row))
    }

    /// What row `row`'s tier handoff reads and writes: the tier view and
    /// the row's tier places; a row past the load's is refused by name.
    pub(crate) fn handoff_parts(
        &mut self,
        row: usize,
    ) -> Result<(&DeviceBuffer<u32>, &mut DeviceBuffer<u32>), GpuError> {
        let n = self.tsel.len();
        let tsel = self.tsel.get_mut(row).ok_or_else(|| past_rows(n, row))?;
        Ok((self.places.buf(), tsel))
    }

    /// Device bytes the side holds.
    pub(crate) fn bytes(&self) -> usize {
        self.places.buf().num_bytes() + self.tsel.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }
}

fn past_rows(n: usize, row: usize) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: format!("row {row} of the tier side's {n} rows"),
    }
}

/// The card leg's shape for the tier: up to [`CHUNK`] tokens of
/// [`N_USED`] slots, `n_embd` wide, through experts `ff` wide.
const fn leg_shape(n_embd: usize, ff: usize) -> LegShape {
    LegShape {
        cap: CHUNK,
        slots_per_col: N_USED,
        n: n_embd,
        ff,
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load, opened in this process or on its node through the common
/// owner ([`leg_tier::open_tier`], [`NODE_FAMILY`]): GLM's tier computation
/// over the map's rows of that tier (the description `layers`, `n_embd`
/// wide), its page rows `rows`; `card_dontneed` as for the stage card's
/// segments. The stage card's context is current again on return.
/// Load-time only.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage card, the file, the plan, the tier and its index, the description, the map, the width and rows, and the page lever (rust-quality R8)"
)]
pub(crate) fn open_tier(
    stage: &Gpu,
    file: &Arc<Split>,
    plan: &Plan<'_>,
    t: &TierOpen,
    tier: usize,
    layers: &[LayerSpec],
    map: &SlotMap,
    [n_embd, rows]: [usize; 2],
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    let open = LegOpen {
        n_expert: N_EXPERT,
        layers: layers.len(),
        hidden: n_embd,
        n_used: N_USED,
        rows,
    };
    leg_tier::open_tier(
        stage,
        file,
        plan,
        (t, tier),
        map,
        (&NODE_FAMILY, open),
        card_dontneed,
    )
}
