//! The placed load of an Inkling 2 file: the dense trunk and the stores on the stage card, every
//! routed expert served by the host tier ([`Placed`]), and the token embedding read as a host row.
//!
//! The card leg runs no IQ-quantized stack ([`accepts::leg_arms`]: NVFP4, Q8_0 and MXFP8
//! only), so the Small's experts all sit on the host (the plan's card counts are 0, refused by
//! name otherwise): a layer's block is the router and the handoff of its picks on the card
//! ([`StepWalk`]'s front), the host's work (the step port's serve), and the routed sum back in
//! `moe` where the post conv reads it. The walk is the common step walk's
//! ([`runtime::sched::walk`], one row through the step port), the layers' sequence the
//! whole-card step's: a loopless body's visit is its layer. The prompt is stepped one id at a
//! time; the verify and the prompt pass are the whole-card load's.

use super::{Layer, STEP, VisitStore, Walk};
use crate::host::StepLeg;
use crate::host::placed::{PlacedHost, PlacedSide, SideShape};
use crate::host::refuse_tier_count;
use crate::host::run::{HostRun, HostWidths};
use crate::host::served::{ResidencyParts, TierBody};
use crate::host::shell::TierShell;
use crate::host::swap_source::ResidencyGlue;
use crate::host::tier::TierOpen;
use crate::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use crate::model::{hosted_hybrid_at, stage_run};
use crate::tensor::copy_row;
use crate::{Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::inkling2::hparams::Hparams;
use model::arch::inkling2::place::Inputs;
use model::arch::inkling2::{host, names};
use model::embed::EmbedTable;
use model::placement::Plan;
use runtime::layer::hosted;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

const PLACED: &str = "inkling2 placed load";

/// The step's walk: one row through the host tier's step port.
const STEP_WALK: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The rows in flight a load's host boundary and tier page hold: the step's one.
pub(super) const LOAD_ROWS: usize = 1;

/// What [`Placed::new`] is made from.
pub(super) struct Open<'a> {
    pub(super) file: &'a Arc<Split>,
    pub(super) plan: &'a Plan<'a>,
    pub(super) inputs: &'a Inputs,
    pub(super) host: HostCfg,
    pub(super) set: HostResidency,
    pub(super) glue: ResidencyGlue,
}

/// The token embedding as the host reads it: the table and one dequantized row.
struct EmbedHost {
    table: EmbedTable,
    row: Vec<f32>,
}

/// The placed side of a load: the common host tier and stage-card side, and the token embedding.
/// The host is the first field: its drop stops the residency machine before any card or pinned
/// memory frees.
pub struct Placed {
    pub(super) host: PlacedHost,
    embed: EmbedHost,
}

/// The slot map of a placed load's plan over the routed run: one row of `n_expert` places a
/// routed layer. Refused by name: a plan of other than one stage card running every layer, any
/// expert tier card (the body hangs none), and the map's own refusals.
pub(super) fn placed_map(open: &Open<'_>, card: usize, hp: &Hparams) -> Result<SlotMap, GpuError> {
    stage_run(PLACED, open.plan, card, hp.n_layer)?;
    let tiers = TierOpen::of_machine(open.plan.machine)?;
    refuse_tier_count(PLACED, tiers.len())?;
    if !tiers.is_empty() {
        return Err(GpuError::shape(
            PLACED,
            "an expert tier card: the Inkling 2 body hangs none under its host tier",
        ));
    }
    let run = hosted(&open.inputs.spec.layers)
        .map_err(|e| GpuError::shape(PLACED, format!("the routed layers: {e}")))?;
    SlotMap::of_plan_tiers(open.plan, card, &[], run, hp.n_expert)
}

impl Placed {
    /// The placed side of the load `open` describes, over the slot map `map` of the routed run:
    /// the host tier over every routed layer, the card's side and the embedding table. Refused
    /// by name: a layer with experts on the card (the card leg runs none of the file's quant
    /// types) and a vocabulary that is not the file's. Load-time only.
    pub(super) fn new(gpu: &Gpu, map: SlotMap, open: Open<'_>) -> Result<Placed, GpuError> {
        let Open {
            file,
            plan,
            inputs,
            host: cfg,
            set,
            glue,
        } = open;
        let hp = &inputs.hp;
        let (n_expert, n_used) = (hp.n_expert, hp.n_used[0]);
        let n = hp.n_layer;
        let shape = SideShape {
            n_expert,
            n_used,
            width: hp.n_embd,
        };
        let side = PlacedSide::new(gpu, &map, (n, shape), PLACED, |l, on_card| {
            Err(GpuError::shape(
                PLACED,
                format!(
                    "layer {l}: {on_card} experts on the card, whose leg runs NVFP4, Q8_0 and \
                     MXFP8 stacks only: the file's experts are the host's"
                ),
            ))
        })?;
        let hosted_set = side.hosted_set();
        let hybrid = hosted_hybrid_at(
            (gpu, file, plan),
            (&hosted_set, n, map),
            (cfg, set, &glue),
            (
                HostWidths {
                    embd: hp.n_embd,
                    ff: hp.expert_ff[0],
                    n_used,
                },
                LOAD_ROWS,
            ),
            None,
            |src| {
                hosted_set
                    .iter()
                    .map(|&l| host::layer(src, hp, l))
                    .collect::<Result<Vec<_>, _>>()
            },
        )?;
        let table = EmbedTable::new(Arc::clone(file), &names::token_embd(), hp.n_embd)
            .map_err(|e| GpuError::shape(PLACED, format!("the embedding table: {e}")))?;
        if table.n_vocab() != hp.n_vocab {
            return Err(GpuError::shape(
                PLACED,
                format!(
                    "the embedding has {} rows, the file's vocabulary {}",
                    table.n_vocab(),
                    hp.n_vocab
                ),
            ));
        }
        Ok(Placed {
            host: PlacedHost::new(gpu, TierShell::new(hybrid, glue, hp.n_embd, PLACED), side)?,
            embed: EmbedHost {
                table,
                row: vec![0.0; hp.n_embd],
            },
        })
    }

    /// The host tier's union slabs for the one-column step: the resident path of the host layer
    /// serves the step only through them. Load-time only.
    pub(super) fn prepare_step_union(&mut self) -> Result<(), GpuError> {
        self.host.prepare_step_union()
    }

    /// Device bytes the placed side holds beside the plan's terms for the tier: the slot map's
    /// card copy, the places and the boundary.
    pub(super) fn device_bytes(&self) -> usize {
        self.host.device_bytes()
    }

    /// Whether the host serves a share of layer `l`. A layer past the model panics by name.
    #[must_use]
    pub fn hosted(&self, l: usize) -> bool {
        self.host.side.hosted(l)
    }

    /// The card's experts of layer `l`: 0 on every layer of a load this body makes. A layer past
    /// the model panics by name.
    #[must_use]
    pub fn n_card(&self, l: usize) -> usize {
        self.host.side.n_card(l)
    }

    /// The host tier, for a gate's instrument.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        self.host.hybrid()
    }

    /// The host tier's settling.
    pub(super) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.host.reset(stream)
    }

    /// The step's embedding row of `token` into `emb`, and the position words the card's
    /// embedding kernel writes (`pos`, the live key count `pos + 1`) from the host.
    pub(super) fn refresh(
        &mut self,
        stream: &CudaStream,
        token: u32,
        (emb, pos, n_keys): (
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<u32>,
            &mut DeviceBuffer<u32>,
        ),
        at: u32,
    ) -> Result<(), GpuError> {
        let EmbedHost { table, row } = &mut self.embed;
        table
            .row_into(token, row)
            .map_err(|e| GpuError::shape(STEP, format!("token {token}: {e}")))?;
        emb.copy_from_host(stream, row)?;
        pos.copy_from_host(stream, &[at])?;
        n_keys.copy_from_host(stream, &[at + 1])?;
        Ok(())
    }
}

/// The common host's ([`PlacedHost`]).
impl TierBody for Placed {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.host.hybrid()
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.host.hybrid_mut()
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.host.residency_parts()
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.host.serve_chain(chain)
    }
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's step port: each
/// visit's mixer half and the block's router and handoff (the front), the host's serve, then the
/// routed sum, the post conv and the stream's sum (the back).
pub(super) struct StepWalk<'a> {
    pub(super) walk: Walk<'a>,
    pub(super) layers: &'a [Layer],
    pub(super) stores: &'a mut [VisitStore],
    pub(super) placed: &'a mut PlacedSide,
}

impl<'a> StepWalk<'a> {
    /// Walk every layer through `leg`.
    pub(super) fn run(&mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let n = self.layers.len();
        sched::walk(STEP_WALK, n, leg, self)
    }
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.placed.hosted(at.layer)
    }

    /// The layer's mixer half, the feed-forward front and the router, the normed row into the
    /// boundary and the handoff of the picks with the go that releases the host leg.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let StepWalk {
            walk,
            layers,
            stores,
            placed,
        } = self;
        let layer = &layers[l];
        walk.attn_half(l, (l, layer), &mut stores[l], false)?;
        walk.ffn_front(l, layer, false)?;
        let gpu = walk.cx.gpu;
        let hy = leg.hybrid();
        copy_row(
            gpu.stream(),
            &walk.s.fxn,
            hy.boundary_mut().normed_mut(),
            walk.d.n,
        )?;
        let picks = (walk.s.route.ids(), walk.s.route.weights());
        placed.go(gpu, picks, hy, l)
    }

    /// No card expert runs under the host's: the card's share of every layer is empty.
    fn shadow(&mut self, _leg: &mut StepLeg<'a, HostRun>, _at: At) -> Result<(), GpuError> {
        Ok(())
    }

    /// From the join on: the host's routed sum into `moe`, then the post conv and the stream's
    /// sum.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let StepWalk {
            walk,
            layers,
            stores,
            ..
        } = self;
        let gpu = walk.cx.gpu;
        let stream = gpu.stream();
        let hy = leg.hybrid();
        hy.boundary().enqueue_back_of(stream, 0, l)?;
        copy_row(stream, hy.boundary().hsum_of(0)?, &mut walk.s.moe, walk.d.n)?;
        hy.row_enqueued(l, 0)?;
        walk.ffn_back(l, &layers[l], &mut stores[l])
    }
}
