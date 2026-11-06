//! The Kimi K3 expert tier ([`crate::host::leg_tier`]): a card beside the stage card that holds
//! some of each routed layer's experts and computes the routed slots the slot map sends it, in the
//! host leg's shadow, through the common card leg (the gate·up `_sel` under SiTU, the down `_sel`)
//! in the latent: the tier's activation and its rows are `expert_latent_length` wide.
//!
//! What is K3's here is the family's reading of the file ([`NODE_FAMILY`]) and the tier's share of
//! the experts' sum ([`TierParts`]). A tier slot's row is the expert's down output; the routed
//! sum is linear in the rows, so the stage card sums the tier's slots in slot order
//! (`ds41_ffn_card_acc` over the tier places) and adds that to the card's and the host's
//! sums before the layer's routed norm.
//!
//! The step's tier handoff writes the tier image and its places with the host image
//! ([`handoff_go_tier`]) and the back waits for both ([`Hybrid::enqueue_tier_back`]).

use std::error::Error;
use std::ops::Range;

use cuda_core::DeviceBuffer;
use gguf::Split;
use model::arch::kimi_k3::place::{self, PlanInputs};
use model::arch::models::{Act as ModelAct, LayerSpec, Moe};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::accepts::Program;
use model::r8file::R8Source;

use crate::card_slots::{CardAcc, CardSlotKernels};
use crate::fault::FaultSink;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::leg_tier::{self, LegFamily, LegWidths, STAGE_TIER, TierSide};
use crate::host::node::NodeFamily;
use crate::host::run::HostRun;
use crate::hybrid::{Hybrid, SlotMap};
use crate::kquant::Act;
use crate::{Gpu, GpuError};

/// What the tier's errors name.
const WHAT: &str = "kimi_k3 tier";

/// The tier page's rows: the step's one row in flight.
pub(super) const PAGE_ROWS: usize = place::TIER_ROWS as usize;

/// Tokens one tier run covers: the body walks a prompt one id a step.
const CHUNK: usize = place::TIER_CHUNK as usize;

/// K3's architecture facts for the common tier node service: the file's reading, the latent's
/// widths and the SiTU-gated experts.
struct KimiK3;

impl LegFamily for KimiK3 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "kimi-k3";
    const ROWS: usize = PAGE_ROWS;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

    /// The file as the plan reads it, whatever the coverage list holds: a tier serves experts,
    /// not the template's tool-call parser.
    fn describe(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn into_model(inputs: PlanInputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &PlanInputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    /// The experts run in the latent: the tier's activation and rows are `expert_latent_length`
    /// wide.
    fn widths(inputs: &PlanInputs) -> LegWidths {
        let hp = &inputs.hp;
        LegWidths {
            n_expert: hp.experts.n_expert,
            n_used: hp.experts.n_used,
            n_embd: hp.latent,
            ff: hp.experts.ff,
            program: Program::KimiK3Body,
        }
    }

    /// Refused by name: the host has no SiTU combine, so a node's host run over layers `run`
    /// cannot serve K3's host experts.
    fn routed_views(
        _: R8Source<'_>,
        _: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
        Err(format!(
            "{WHAT}: a node tier's host run over layers {run:?}: the host tier has no SiTU \
             expert form"
        )
        .into())
    }

    /// The block's SiTU at the file's betas; any other activation is refused by name, as is a
    /// beta the card's rule does not run.
    fn tier_act(moe: &Moe) -> Result<Act, GpuError> {
        let ModelAct::Situ {
            gate_beta,
            value_beta,
        } = moe.act
        else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a {} block: the tier runs K3's SiTU experts",
                    moe.act.what()
                ),
            ));
        };
        let act = Act::Situ {
            beta: gate_beta,
            linear_beta: value_beta,
        };
        act.code(WHAT)?;
        Ok(act)
    }
}

/// K3's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<KimiK3>();

/// The stage card's side of a tiered load: the common tier side (the slot map's tier view and the
/// step's tier places).
pub(super) struct TierParts {
    side: TierSide,
}

/// What a tier layer's sum reads: the tier's rows `trows` of one row of `n_used` slots
/// (slot-major, `n` values a slot), the slots' weights and tier places, and the layer's tier
/// experts.
pub(super) struct TierSum<'a> {
    pub(super) trows: &'a DeviceBuffer<f32>,
    pub(super) w: &'a DeviceBuffer<f32>,
    pub(super) tsel: &'a DeviceBuffer<u32>,
    pub(super) n_tier: usize,
    pub(super) n: usize,
    pub(super) n_used: usize,
    pub(super) fault: FaultSink,
}

impl TierParts {
    /// The parts for `map`'s tier over `layers` layers of `n_used` slots a row. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        n_used: usize,
    ) -> Result<TierParts, GpuError> {
        Ok(TierParts {
            side: TierSide::new(gpu, map, layers, (PAGE_ROWS, n_used))?,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(super) fn k(&self, l: usize) -> usize {
        self.side.k(l)
    }

    /// Device bytes the parts hold.
    pub(super) fn bytes(&self) -> usize {
        self.side.bytes()
    }

    /// The step's tier places, as the last handoff wrote them.
    pub(super) fn step_tsel(&self) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.side.tsel(0)
    }

    /// Layer `l`'s step handoff of a tier layer and the go that releases the host and the tier: the
    /// host image, the tier image (the boundary's activation and each slot's tier place) and each
    /// slot's stage place into `sel` (the common any-count tier entry,
    /// [`HandoffKernels::enqueue_handoff_tier_n`]), then the go. One launch and the go.
    pub(super) fn enqueue_step_go(
        &mut self,
        gpu: &Gpu,
        kernels: &HandoffKernels,
        h: &Handoff<'_>,
        hybrid: &mut Hybrid<HostRun>,
        l: usize,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let (tmap, tsel) = self.side.handoff_parts(0)?;
        let targets = hybrid.tier_handoff(0, STAGE_TIER)?;
        kernels.enqueue_handoff_tier_n(
            stream,
            h,
            tmap,
            targets,
            gpu.layer_sink(l)?,
            (sel, tsel),
        )?;
        hybrid.enqueue_tier_go(stream, l, 0)
    }
}

/// The tier's share of a layer's experts' sum: the tier slots' weighted sum in slot order into
/// `acc`. One launch.
pub(super) fn enqueue_tier_sum(
    gpu: &Gpu,
    k: &CardSlotKernels,
    s: &TierSum<'_>,
    acc: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    k.enqueue_card_acc(
        gpu.stream(),
        &CardAcc {
            down: s.trows,
            w: s.w,
            sel: s.tsel,
            n: s.n,
            m: 1,
            n_card: s.n_tier,
            n_used: s.n_used,
            w_pitch: s.n_used,
            fault: s.fault,
        },
        acc,
    )
}
