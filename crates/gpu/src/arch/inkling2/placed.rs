//! The placed load of an Inkling 2 file: the dense trunk and the stores on the stage card, each
//! routed layer's experts split between the card (the plan's share, the id prefix
//! `[0, n_card)`) and the host tier ([`Placed`]), and the token embedding read as a host row.
//!
//! The card leg runs the stack types of [`accepts::leg_arms`] (NVFP4, Q8_0, MXFP8 and the
//! dynamic mixes' i-quant and 32-value ones); a file with a stack of another type (the Small's
//! IQ2_XS and IQ4_XS ones) has none of its experts on the card (the plan's card counts are 0, a
//! share refused by name otherwise). A layer's block is the router and the handoff of its picks
//! on the card ([`StepWalk`]'s front), the card's own slots under the host's work (its shadow,
//! the slots the handoff placed on the card, summed in slot order), the host's work (the step
//! port's serve), and the routed sum back in `moe`, the card's sum and the host's added where the
//! post conv reads it (the routed sum is linear in the experts' rows). The walk is the common step
//! walk's ([`runtime::sched::walk`], one row through the step port) over the file's visits: a
//! looped trunk's stored layer is visited once a pass, its host service asked once a visit, so the
//! go, the join and the host's work are the visit's, the experts and the slot map the layer's.
//! The prompt is stepped one id at a time; the verify and the prompt pass are the whole-card
//! load's.

use super::{Layer, STEP, VisitStore, Walk, layer_card_stacks};
use crate::card_leg::LegSum;
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
use model::arch::models::Visit;
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
    /// by name: a layer with all its experts on the card, an expert tier card and a vocabulary
    /// that is not the file's. Load-time only.
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
            if on_card >= n_expert {
                return Err(GpuError::shape(
                    PLACED,
                    format!(
                        "layer {l}: all {on_card} experts on the card: the placed walk serves \
                         layers the host holds a share of (a load that holds the whole model on \
                         the card is the whole-card load)"
                    ),
                ));
            }
            Ok(())
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

    /// The card's experts of layer `l`: the plan's share, 0 on a layer whose stacks the card leg
    /// does not run. A layer past the model panics by name.
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

/// The captured decode step's walk `(1, 1, Step)` through the host tier's step port, over the
/// file's visits: each visit's mixer half and the block's router and handoff (the front), the
/// card's slots under the host's serve (the shadow), then the routed sum, the post conv and the
/// stream's sum (the back). A walk's layer is the visit's index; the visit's stored layer names
/// the weights, the slot map's row and the host service.
pub(super) struct StepWalk<'a> {
    pub(super) walk: Walk<'a>,
    pub(super) visits: &'a [Visit],
    /// The looped core's first stored layer, where a pass begins; `None` on a loopless file.
    pub(super) loop_core: Option<usize>,
    /// Whether the visits of a pass past the first run their messages.
    pub(super) messages: bool,
    pub(super) layers: &'a [Layer],
    pub(super) stores: &'a mut [VisitStore],
    pub(super) placed: &'a mut PlacedSide,
}

impl<'a> StepWalk<'a> {
    /// Walk every visit through `leg`.
    pub(super) fn run(&mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let n = self.visits.len();
        sched::walk(STEP_WALK, n, leg, self)
    }

    /// The stored layer of the walk's visit `k`.
    fn layer_of(&self, k: usize) -> usize {
        self.visits[k].layer as usize
    }
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.placed.hosted(self.layer_of(at.layer))
    }

    /// The visit's mixer half, the feed-forward front and the router, the normed row into the
    /// boundary and the handoff of the picks with the go that releases the host leg.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let k = at.layer;
        let v = self.visits[k];
        let l = v.layer as usize;
        let StepWalk {
            walk,
            layers,
            stores,
            placed,
            loop_core,
            messages,
            ..
        } = self;
        let layer = &layers[l];
        if *loop_core == Some(l) {
            walk.pass_edge(v.pass)?;
        }
        let msg = *messages && v.pass > 0 && layer.ctrl.is_some();
        walk.attn_half(k, (l, layer), &mut stores[k], msg)?;
        walk.ffn_front(k, layer, msg)?;
        let gpu = walk.cx.gpu;
        let hy = leg.hybrid();
        copy_row(
            gpu.stream(),
            &walk.s.fxn,
            hy.boundary_mut().normed_mut(),
            walk.d.n,
        )?;
        let forced = walk.forced_picks(k)?;
        let picks: (&DeviceBuffer<u32>, &DeviceBuffer<f32>) = match &forced {
            Some((ids, weights)) => (ids, weights),
            None => (walk.s.route.ids(), walk.s.route.weights()),
        };
        placed.go(gpu, picks, hy, l)
    }

    /// The card's slots of the layer, under the host's work: the slots the handoff placed on
    /// the card over the router's normed row, summed in slot order; none on a layer with no
    /// experts on the card.
    fn shadow(&mut self, _leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = self.layer_of(at.layer);
        if self.placed.n_card(l) == 0 {
            return Ok(());
        }
        let StepWalk {
            walk,
            layers,
            placed,
            ..
        } = self;
        let stacks = layer_card_stacks(walk.cx.w, &layers[l])?;
        let forced = walk.forced_picks(at.layer)?;
        let Walk { cx, k, s, act, .. } = walk;
        let super::Scratch {
            fxn,
            leg,
            cacc,
            route,
            ..
        } = &mut **s;
        let Some(run) = placed.card_run(l, &stacks, fxn, *act) else {
            return Ok(());
        };
        k.leg.enqueue(
            cx.gpu,
            leg,
            &run,
            Some(LegSum {
                w: match &forced {
                    Some((_, weights)) => weights,
                    None => route.weights(),
                },
                acc: cacc,
            }),
        )
    }

    /// From the join on: the host's routed sum, the card's added where it holds experts of the
    /// layer, into `moe`, then the post conv and the stream's sum.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let k = at.layer;
        let l = self.layer_of(k);
        let StepWalk {
            walk,
            layers,
            stores,
            placed,
            ..
        } = self;
        let gpu = walk.cx.gpu;
        let stream = gpu.stream();
        let n = walk.d.n;
        let hy = leg.hybrid();
        hy.boundary().enqueue_back_of(stream, 0, l)?;
        let hsum = hy.boundary().hsum_of(0)?;
        if placed.n_card(l) > 0 {
            gpu.elem()
                .enqueue_add(stream, &walk.s.cacc, hsum, n, &mut walk.s.moe)?;
        } else {
            copy_row(stream, hsum, &mut walk.s.moe, n)?;
        }
        hy.row_enqueued(l, 0)?;
        walk.ffn_back(k, &layers[l], &mut stores[k])
    }
}
