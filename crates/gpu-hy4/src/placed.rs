//! A Hy4-flash load whose routed experts the stage card cannot all hold: the common placed side
//! ([`bloomery_gpu::host::placed`]) over the latent's width, the host tier serving each routed
//! layer's experts past the card's share under the gate-first clamp. The stage card holds every
//! layer's dense part and the stores; the embedding and the engram table stay in the file, read
//! on the host as on the whole-card load.
//!
//! A layer the plan holds whole runs the whole-card block ([`crate::step::ffn`]); a routed layer
//! with a host share runs the block in three parts around the host tier's step port, in the host
//! leg's order:
//!
//! - **front**: the layer's engram site and attention sub-layer, the feed-forward fold and norm,
//!   the router on the normed full-width row, the latent down into the boundary's activation (the
//!   host reads the experts' own input, whose width is the latent's), the handoff of the ids,
//!   weights and that row with each slot's place in the card's stacks, and the go;
//! - **shadow**, under the host's work: the shared SiLU expert on the full-width row and the
//!   card's slots over the latent row (a host slot's warps return before their first load),
//!   summed in slot order;
//! - **back**: the wait, the card's and the host's sums added (the routed sum is linear in the
//!   experts' rows), the latent up, the shared expert's output added, the sandwich norm and the
//!   identity mix's post.
//!
//! The sum's order is `card + host`: up to rounding the whole-card layer's, which sums the sixteen
//! slots in one launch. A load opened with its pass runs the prompt as the pass's walk
//! ([`crate::batch::prompt_placed`]) and steps it one id at a time otherwise. Under a residency
//! machine the card's slots are the ones the live map names: the handoff reads the side's map
//! copy, which the machine's flips write ([`bloomery_gpu::host::placed`]).

use bloomery_gpu::card_leg::{LegShape, LegStacks, LegSum};
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::placed::{PlacedHost, PlacedSide, SideShape};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap_source::ResidencyGlue;
use bloomery_gpu::hybrid::{HostResidency, Hybrid, SlotMap};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::model::hosted_hybrid_at;
use bloomery_gpu::routed_block::{RouteIn, RouterEntry};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_levers::HostCfg;
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::names;
use model::placement::Plan;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::Site;

use crate::body::{Body, Ffn, Glu, Layer, Router, WHAT, cx};
use crate::step;

/// What the placed load's refusals name.
const PLACED: &str = "hy4 placed load";

/// The step's walk: one row through the host tier's step port.
const STEP_WALK: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The rows in flight the host boundary holds: the step's one.
pub(crate) const LOAD_ROWS: usize = 1;

/// The placed side of a Hy4 load: the common host tier and stage-card side.
pub type Placed = PlacedHost;

/// What [`load`] is made from.
pub(crate) struct Open<'a> {
    pub(crate) file: &'a Arc<Split>,
    pub(crate) plan: &'a Plan<'a>,
    pub(crate) hp: &'a Hparams,
    pub(crate) host: HostCfg,
    pub(crate) set: HostResidency,
    pub(crate) glue: ResidencyGlue,
}

/// The placed side of the load `open` describes over `layers` and the slot map `map` of the
/// routed run: the host tier over the layers whose card count is under their experts, and the
/// card's side, the residency machine's glue (`open.glue`) beside them. Refused by name: a plan
/// that names an expert tier card (the body hangs none), a layer whose card stacks are not its
/// card count's, and a machine beside a layer the card holds whole ([`PlacedHost::new`]). The
/// host serves its share under the gate-first clamp at the layer's limit
/// ([`crate::tier::host_layer`]). Load-time only.
pub(crate) fn load(
    (gpu, w): (&Gpu, &Weights),
    layers: &[Layer],
    map: SlotMap,
    open: Open<'_>,
) -> Result<Placed, GpuError> {
    let Open {
        file,
        plan,
        hp,
        host,
        set,
        glue,
    } = open;
    if let Some(card) = plan.machine.expert_tiers().next() {
        return Err(GpuError::shape(
            PLACED,
            format!(
                "card {} is an expert tier card: the body serves its host share from the host \
                 tier alone",
                card.name
            ),
        ));
    }
    let (n_expert, n_used) = (hp.experts.n_expert, hp.experts.n_used);
    if (n_expert, n_used) != (Router::N_EXPERT, Router::N_USED) {
        return Err(GpuError::shape(
            PLACED,
            format!(
                "{n_expert} experts, {n_used} a token; the router runs {} and {}",
                Router::N_EXPERT,
                Router::N_USED
            ),
        ));
    }
    let n = layers.len();
    let leg = LegShape {
        cap: 1,
        slots_per_col: n_used,
        n: hp.latent,
        ff: hp.experts.ff,
    };
    let shape = SideShape {
        n_expert,
        n_used,
        width: hp.latent,
    };
    let mut side = PlacedSide::new(gpu, &map, (n, shape), PLACED, |l, on_card| {
        let Ffn::Routed { stacks, .. } = &layers[l].ffn else {
            return Err(GpuError::shape(
                PLACED,
                format!("layer {l}: experts on the card of a layer with no routed block"),
            ));
        };
        LegStacks::of(w, stacks.each_ref().map(String::as_str))?.check(on_card, leg)
    })?;
    // Sixteen slots a token: no per-count handoff entry is built for it, and no tier card holds
    // experts of a hosted layer.
    side.sink_any_count(gpu.stream())?;
    let hosted_set = side.hosted_set();
    let hybrid = hosted_hybrid_at(
        (gpu, file, plan),
        (&hosted_set, n, map),
        (host, set, &glue),
        (
            HostWidths {
                embd: hp.latent,
                ff: hp.experts.ff,
                n_used,
            },
            LOAD_ROWS,
        ),
        None,
        |src| {
            hosted_set
                .iter()
                .map(|&l| crate::tier::host_layer(src, hp, l))
                .collect()
        },
    )?;
    PlacedHost::new(gpu, TierShell::new(hybrid, glue, hp.n_embd, PLACED), side)
}

/// A routed block's parts.
struct Parts<'a> {
    down_lat: &'a Site,
    up_lat: &'a Site,
    stacks: &'a [String; 3],
    shared: &'a Glu,
    clamp: f32,
}

/// Layer `l`'s routed block; a layer with no routed block is refused by name.
fn routed(layers: &[Layer], l: usize) -> Result<Parts<'_>, GpuError> {
    match &layers[l].ffn {
        Ffn::Routed {
            down_lat,
            up_lat,
            stacks,
            shared,
            clamp,
        } => Ok(Parts {
            down_lat,
            up_lat,
            stacks,
            shared,
            clamp: *clamp,
        }),
        Ffn::Dense(_) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a dense block with experts on the host tier"),
        )),
    }
}

/// A hosted layer's front from the feed-forward norm: the router on `s.xn`, the latent down into
/// the boundary's activation, the handoff and the go.
fn front(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    side: &mut PlacedSide,
    hy: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        routes,
        ..
    } = body;
    let down_lat = routed(layers, l)?.down_lat;
    let cx = cx(gpu, w, k, sh);
    let route = routes[l].as_mut().ok_or(GpuError::State {
        what: WHAT,
        missing: "a routed layer's router output",
    })?;
    let r = RouteIn {
        w: w.f32_tensor(WHAT, &names::ffn_gate_inp(l))?,
        bias: Some(w.f32_buf(WHAT, &names::exp_probs_b(l))?),
        scale: sh.weights_scale,
        fault: gpu.layer_sink(l)?,
    };
    k.router.route(gpu.stream(), &s.xn, &r, route)?;
    down_lat.project(&cx, acts, l, &s.xn, hy.boundary_mut().normed_mut())?;
    let picks = (Router::out_ids(route), Router::out_weights(route));
    side.go(gpu, picks, hy, l)
}

/// A hosted layer's card work under the host's: the shared expert on the full-width row into
/// `s.shared` and the card's slots over the latent row the handoff left in the boundary into
/// `s.routed`.
fn shadow(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    side: &PlacedSide,
    hy: &Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let Body {
        sh,
        layers,
        k,
        acts,
        s,
        routes,
        ..
    } = body;
    let Parts {
        stacks,
        shared,
        clamp,
        ..
    } = routed(layers, l)?;
    let cx = cx(gpu, w, k, sh);
    step::glu(gpu, (&cx, acts), l, shared, s, true)?;
    if side.n_card(l) == 0 {
        return Ok(());
    }
    let route = routes[l].as_ref().ok_or(GpuError::State {
        what: WHAT,
        missing: "a routed layer's router output",
    })?;
    let st = LegStacks::of(w, stacks.each_ref().map(String::as_str))?;
    let rule = Act::SwigluGateClamp { limit: clamp };
    let Some(run) = side.card_run(l, &st, hy.boundary().normed(), rule) else {
        return Ok(());
    };
    k.leg.enqueue(
        gpu,
        &mut s.leg,
        &run,
        Some(LegSum {
            w: Router::out_weights(route),
            acc: &mut s.routed,
        }),
    )
}

/// A hosted layer's back from the join on: the wait, the card's and the host's sums into the
/// routed sum, the latent up, the shared expert added, the sandwich norm into `s.out`.
fn back(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    side: &PlacedSide,
    hy: &mut Hybrid<HostRun>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    {
        let Body {
            sh,
            layers,
            k,
            acts,
            s,
            ..
        } = &mut *body;
        let up_lat = routed(layers, l)?.up_lat;
        let cx = cx(gpu, w, k, sh);
        hy.boundary().enqueue_back_of(stream, 0, l)?;
        let hsum = hy.boundary().hsum_of(0)?;
        let sum: &DeviceBuffer<f32> = if side.n_card(l) > 0 {
            gpu.elem()
                .enqueue_add(stream, &s.routed, hsum, sh.latent, &mut s.lat)?;
            &s.lat
        } else {
            hsum
        };
        up_lat.project(&cx, acts, l, sum, &mut s.lat_up)?;
        gpu.elem()
            .enqueue_add(stream, &s.lat_up, &s.shared, sh.hidden, &mut s.f)?;
    }
    step::ffn_norm(gpu, w, body, l)?;
    hy.row_enqueued(l, 0)
}

/// The captured decode step's walk `(1, 1, Step)` through the host tier's step port: each layer's
/// attention sub-layer, then its feed-forward block whole or as the front, shadow and back around
/// the host's go, then the identity mix's post.
struct StepWalk<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    body: &'a mut Body,
    side: &'a mut PlacedSide,
    /// The streams' buffer, as [`step::enqueue`] keeps it.
    cur: usize,
}

/// Walk every layer of `body` through `placed`'s step port, the streams entering in `*cur` and
/// leaving where the last post put them.
pub(crate) fn walk(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    placed: &mut Placed,
    cur: &mut usize,
) -> Result<(), GpuError> {
    let PlacedHost { shell, side, .. } = placed;
    let mut leg = StepLeg::new(gpu.stream(), &mut shell.hybrid);
    let n = body.layers.len();
    let mut prog = StepWalk {
        gpu,
        w,
        body,
        side,
        cur: *cur,
    };
    sched::walk(STEP_WALK, n, &mut leg, &mut prog)?;
    *cur = prog.cur;
    Ok(())
}

impl<'a> LayerProgram for StepWalk<'a> {
    type Port = StepLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    /// A layer with no host share whole; a hosted one up to its go.
    fn front(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let (gpu, w) = (self.gpu, self.w);
        let fault = gpu.layer_sink(l)?;
        step::attn_half(gpu, w, self.body, l, &mut self.cur, fault)?;
        if !self.side.hosted(l) {
            step::ffn(gpu, w, self.body, l, fault)?;
            return step::close(gpu, self.body, l, &mut self.cur);
        }
        front((gpu, w), self.body, self.side, leg.hybrid(), l)
    }

    /// The shared expert and the card's slots of a hosted layer, under the host's work.
    fn shadow(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted(l) {
            return Ok(());
        }
        shadow((self.gpu, self.w), self.body, self.side, leg.hybrid(), l)
    }

    /// A hosted layer from the join on, then the identity mix's post.
    fn back(&mut self, leg: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.side.hosted(l) {
            return Ok(());
        }
        back((self.gpu, self.w), self.body, self.side, leg.hybrid(), l)?;
        step::close(self.gpu, self.body, l, &mut self.cur)
    }
}
