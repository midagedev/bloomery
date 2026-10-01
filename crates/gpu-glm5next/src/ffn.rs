//! The feed-forward blocks of one layer at one token: `x` (the fold of the
//! streams by the block's own mix) in, the update into `out`.
//!
//! A dense block ([`dense`]): `ffn_norm` RMS, the gate·up·SwiGLU with the
//! shared limit (`ds41_shexp_gate_up` over the q8_0 planes), the down
//! projection. Three launches, all in the layer's front.
//!
//! A routed block runs around its host leg; the card computes the experts the
//! slot map puts on it ([`CardExperts`]), the host tier the rest:
//! - [`front`]: `ffn_norm` RMS into the boundary's handoff region, the
//!   sigmoid router over 288 experts with the selection bias (top 8, the
//!   weights from the unbiased scores, normalized and scaled), the handoff
//!   into the host-mapped image — each slot's place in the card's stacks, or
//!   the host mark, from the layer's slot-map row — and the go. Three kernels
//!   and the go.
//! - [`shadow`]: while the host computes, the card's slots — the norm's q8_1
//!   form, the gate·up `_sel` with the routed clamped SwiGLU, the q8_1 of
//!   those slots' columns, the down `_sel` and the card slots' weighted sum
//!   (`ds41_ffn_card_acc_8`) — then the shared expert, gate·up·SwiGLU and
//!   down, and the card sum plus the shared expert's output. Two kernels
//!   without card experts, [`CARD_LAUNCHES`] more with.
//! - [`back`]: the wait, then the host's routed sum plus the shadow's:
//!   `hsum + (card + shexp)`, or `hsum + shexp` on a layer without card
//!   experts (`moe + shexp` as ik adds them). One kernel after the wait.
//!
//! A layer the expert tier card holds experts of (a load with a tier,
//! [`crate::tier`]) runs as a tier layer: the front's handoff also writes the
//! tier's image (the normed row in f32 and each slot's tier place,
//! `FfnKernels::enqueue_handoff_tier`) and the go is the host's and the
//! tier's; the shadow runs the card slots' downs but not their sum; the back
//! waits for the host and the tier, then sums the card's and the tier's rows
//! in one weighted sum (`ds41_ffn_card_acc_8_tier`, the card sum's order with
//! the tier's slots in it), adds the shared expert and the host's sum. The
//! step graph holds the same node count either way.
//!
//! The handoff carries the boundary's eight slots through
//! `FfnKernels::enqueue_handoff`, which picks its entry by the slot count and
//! refuses by name a count it has no entry for. Neither post entry of the
//! V4.1 piece is called: the sum is one add, and `hc_post` is the
//! sub-layer's own.
//!
//! ik divides GLM's eight weights by their bare sum; the router divides by
//! the sum plus `1e-20` (`route_core::renorm_divisor`), under half an ulp of
//! every sum from 2^-42 up. The two agree bit for bit unless all eight
//! sigmoid scores sit below that: a named difference.

use bloomery_gpu::hybrid::{Boundary, Hybrid, SlotMap};
use bloomery_gpu::kquant::{Act, GateUpAct, KquantKernels, SelDown};
use bloomery_gpu::q4k_sel::QuantSel;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{COL_GROUP, DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::chain::ffn::{CardAcc, CardAccTier, FfnBatchKernels, Handoff, Places};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::arch::glm5next::names;
use model::arch::glm5next::place::{KdaLanes, card_routed};
use models::{Ffn, LayerSpec};

use crate::body::{Parts, f32t, f32v, gemv, weight};
use crate::host::GlmHost;
use crate::tensors::{FfnNames, LayerNames, other_kind};
use crate::tier::{STAGE_TIER, TierSide};

/// The dense block's launches.
pub(crate) const DENSE_LAUNCHES: usize = 3;

/// A routed block's launches: the front's three kernels and the go, the
/// shadow's two, the wait and the sum.
pub(crate) const MOE_LAUNCHES: usize = 3 + 1 + 2 + 1 + 1;

/// The shadow's launches more on a layer with card experts: the norm's q8_1,
/// the gate·up, the q8_1 of its card columns, the down, the card sum and the
/// add of the shared expert's output.
pub(crate) const CARD_LAUNCHES: usize = 6;

/// A layer's card experts: how many its stacks hold and the routed SwiGLU
/// limit (`swiglu_clamp_exp`).
#[derive(Clone, Copy, Debug)]
struct CardLayer {
    n_card: usize,
    limit: f32,
}

/// The card's routed experts of the step: the family's kernels, each layer's
/// count and limit from the slot map and the description, and each row's
/// one-token buffers every card layer's shadow writes (a verify's row reads
/// its own card sum in its back, after the other row's shadow). Made at load, after the weights,
/// from the stacks they hold; a layer the map puts experts on holds its three
/// stacks in a format [`card_routed`] names, at `n_card` experts each, and a
/// layer it puts none on holds none — anything else is refused by name.
pub(crate) struct CardExperts {
    kq: KquantKernels,
    batch: FfnBatchKernels,
    /// Per layer of the description, `None` without card experts.
    layers: Vec<Option<CardLayer>>,
    /// Routed expert width: a gate·up slot's rows.
    ff: usize,
    /// Each row's buffers, row 0 the step's: a row a KDA lane of the load.
    rows: Vec<CardRow>,
    /// The stage card's side of the tier layers, on a load with a tier card.
    tier: Option<TierSide>,
}

/// One row's buffers of the card experts' shadow.
struct CardRow {
    /// The norm's q8_1 form, one column.
    act_x: Q8Act,
    /// The gate·up's output, a slot's `ff` rows each.
    h: DeviceBuffer<f32>,
    /// Its card slots' columns in q8_1, one a slot.
    act_h: Q8Act,
    /// The down `_sel`'s output, a slot's `n_embd` rows each.
    down: DeviceBuffer<f32>,
    /// The card slots' weighted sum, and it plus the shared expert's output.
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
}

impl CardRow {
    fn new(stream: &CudaStream, n_embd: usize, ff: usize) -> Result<CardRow, GpuError> {
        Ok(CardRow {
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            h: DeviceBuffer::zeroed(stream, N_USED * ff)?,
            act_h: Q8Act::with_slots(stream, N_USED, ff)?,
            down: DeviceBuffer::zeroed(stream, N_USED * n_embd)?,
            acc: DeviceBuffer::zeroed(stream, n_embd)?,
            pre: DeviceBuffer::zeroed(stream, n_embd)?,
        })
    }

    fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.h.num_bytes()
            + self.act_h.device_bytes()
            + self.down.num_bytes()
            + self.acc.num_bytes()
            + self.pre.num_bytes()
    }
}

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them
    /// and `w` holds them, `n_embd` wide with routed experts `ff` wide, a
    /// row's buffers for each of the load's `lanes`. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        layers: &[LayerSpec],
        map: &SlotMap,
        n_embd: usize,
        ff: usize,
        lanes: KdaLanes,
    ) -> Result<CardExperts, GpuError> {
        let mut per = Vec::with_capacity(layers.len());
        for (l, spec) in layers.iter().enumerate() {
            // A dense block is outside the host run and has no row: no card
            // experts, which the resident-rows check below holds it to. A
            // routed layer the map has no row for is refused by name.
            let n_card = match spec.ffn {
                Ffn::Dense { .. } => 0,
                Ffn::Moe(_) => map.on_card(l)?,
            };
            let names = stack_names(l);
            if n_card == 0 {
                if let Some(n) = names.iter().find(|n| w.get(n).is_some()) {
                    return Err(GpuError::Tensor {
                        what: CARD,
                        name: n.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on the card",
                    });
                }
                per.push(None);
                continue;
            }
            let Ffn::Moe(m) = &spec.ffn else {
                return Err(GpuError::Shape {
                    what: CARD,
                    detail: format!("layer {l}: card experts on a dense block"),
                });
            };
            let models::Act::SwiGlu { limit } = m.act;
            let stacks = CardStacks::of(w, l)?;
            for (st, rpe) in [(stacks.gate, ff), (stacks.up, ff), (stacks.down, n_embd)] {
                if st.w.rows() != n_card * rpe {
                    return Err(GpuError::Tensor {
                        what: CARD,
                        name: st.name.clone(),
                        need: "the rows of the slot map's card experts of its layer",
                    });
                }
            }
            per.push(Some(CardLayer {
                n_card,
                limit: limit.unwrap_or(0.0),
            }));
        }
        let stream = gpu.stream();
        Ok(CardExperts {
            kq: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            batch: FfnBatchKernels::load(gpu.context())?,
            layers: per,
            ff,
            rows: (0..lanes.count())
                .map(|_| CardRow::new(stream, n_embd, ff))
                .collect::<Result<Vec<_>, _>>()?,
            tier: match map.tiers() {
                0 => None,
                _ => Some(TierSide::new(gpu, map, layers.len(), lanes.count())?),
            },
        })
    }

    /// Layer `l` computes experts on the card.
    pub(crate) fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    /// The experts of layer `l` the expert tier holds: 0 without a tier.
    pub(crate) fn tier_k(&self, l: usize) -> usize {
        self.tier.as_ref().map_or(0, |t| t.k(l))
    }

    /// Layer `l` has a card sum: card experts on the stage card or the tier.
    pub(crate) fn sums(&self, l: usize) -> bool {
        self.has(l) || self.tier_k(l) > 0
    }

    /// The stage card's side of the tier layers, on a load with a tier card.
    pub(crate) fn tier(&self) -> Option<&TierSide> {
        self.tier.as_ref()
    }

    /// The stage card's experts of layer `l`: 0 off the card.
    pub(crate) fn n_card(&self, l: usize) -> usize {
        self.layer(l).map_or(0, |c| c.n_card)
    }

    /// The batch kernels the card sums run on.
    pub(crate) fn batch(&self) -> &FfnBatchKernels {
        &self.batch
    }

    fn layer(&self, l: usize) -> Option<CardLayer> {
        self.layers.get(l).copied().flatten()
    }

    /// Row 0's card slots' weighted sum of the last card layer enqueued.
    pub(crate) fn acc(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        Ok(&card_row(&self.rows, 0)?.acc)
    }

    /// The routed experts' width: a gate·up slot's rows.
    pub(crate) fn ff(&self) -> usize {
        self.ff
    }

    /// Device bytes of the buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.rows.iter().map(CardRow::bytes).sum::<usize>()
            + self.tier.as_ref().map_or(0, TierSide::bytes)
    }
}

/// What the card experts' errors name.
const CARD: &str = "glm5next CardExperts";

/// Row `row`'s card buffers; a row past the load's is refused by name.
fn card_row(rows: &[CardRow], row: usize) -> Result<&CardRow, GpuError> {
    rows.get(row).ok_or_else(|| past_rows(rows.len(), row))
}

/// Row `row`'s card buffers, to write; a row past the load's is refused by
/// name.
fn card_row_mut(rows: &mut [CardRow], row: usize) -> Result<&mut CardRow, GpuError> {
    let n = rows.len();
    rows.get_mut(row).ok_or_else(|| past_rows(n, row))
}

fn past_rows(n: usize, row: usize) -> GpuError {
    GpuError::Shape {
        what: CARD,
        detail: format!("row {row} of the card experts' {n} rows"),
    }
}

/// Layer `l`'s routed gate, up and down names.
pub(crate) fn stack_names(l: usize) -> [String; 3] {
    [
        names::ffn_gate_exps(l),
        names::ffn_up_exps(l),
        names::ffn_down_exps(l),
    ]
}

/// One resident routed stack: its name, file type and words.
pub(crate) struct Stack<'w> {
    pub(crate) name: String,
    pub(crate) ty: GgmlType,
    pub(crate) w: &'w DeviceTensor<u32>,
}

/// A layer's three resident routed stacks, each in a format [`card_routed`]
/// names and the gate and up of one type.
pub(crate) struct CardStacks<'w> {
    pub(crate) gate: Stack<'w>,
    pub(crate) up: Stack<'w>,
    pub(crate) down: Stack<'w>,
}

impl<'w> CardStacks<'w> {
    pub(crate) fn of(w: &'w Weights, l: usize) -> Result<CardStacks<'w>, GpuError> {
        let [gate, up, down] = stack_names(l).map(|name| match w.get(&name) {
            Some(DevWeight::KQuant { ty, w, .. }) if card_routed(*ty).is_some() => {
                Ok(Stack { name, ty: *ty, w })
            }
            _ => Err(GpuError::Tensor {
                what: CARD,
                name,
                need: "a resident routed stack of a type the card experts read (Q4_K, Q5_K)",
            }),
        });
        let (gate, up, down) = (gate?, up?, down?);
        if up.ty != gate.ty {
            return Err(GpuError::Tensor {
                what: CARD,
                name: up.name,
                need: "the gate stack's type",
            });
        }
        Ok(CardStacks { gate, up, down })
    }
}

/// A routed layer's names, borrowed out of their [`FfnNames::Moe`].
pub(crate) struct MoeNames<'a> {
    pub norm: &'a str,
    pub router: &'a str,
    pub bias: &'a str,
    pub sh_gate: &'a str,
    pub sh_up: &'a str,
    pub sh_down: &'a str,
}

/// Layer `l`'s routed block's names; another kind's is refused by name.
pub(crate) fn moe_names<'a>(p: &Parts<'a>, l: usize) -> Result<MoeNames<'a>, GpuError> {
    let names: &'a [LayerNames] = p.names;
    match names.get(l).map(|n| &n.ffn) {
        Some(FfnNames::Moe {
            norm,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
        }) => Ok(MoeNames {
            norm,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
        }),
        _ => Err(other_kind("glm5next routed block", l)),
    }
}

/// Enqueue layer `l`'s dense block (module doc).
pub(crate) fn dense(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let Some(FfnNames::Dense {
        norm,
        gate,
        up,
        down,
    }) = p.names.get(l).map(|n| &n.ffn)
    else {
        return Err(other_kind("glm5next dense", l));
    };
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, norm)?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    p.k.experts.enqueue_shexp_gate_up(
        stream,
        weight(w, gate)?,
        weight(w, up)?,
        &s.xn,
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, down, &s.h, &mut s.out)
}

/// Enqueue layer `l`'s routed block up to its go (module doc).
pub(crate) fn front(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let fault = gpu.layer_sink(l)?;
    let n = moe_names(p, l)?;
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, n.norm)?,
        d.rms_eps,
        d.embd,
        1,
        hybrid.boundary_mut().normed_mut(),
    )?;
    let bias = if c.bias { f32v(w, n.bias)? } else { &s.no_bias };
    p.k.router.enqueue_router(
        stream,
        f32t(w, n.router)?,
        hybrid.boundary().normed(),
        bias,
        d.scale,
        &mut s.rout,
        fault,
    )?;
    let h = Handoff {
        ids: &s.rout.ids,
        weights: &s.rout.weights,
        map: p.slots.buf(),
        row_off: c.row_off,
        n_expert: N_EXPERT,
    };
    if let Some(side) = p.card.tier.as_mut().filter(|t| t.k(l) > 0) {
        let (places, tsel) = side.handoff_parts(p.row)?;
        let targets = hybrid.tier_handoff(p.row, STAGE_TIER)?;
        p.k.ffn
            .enqueue_handoff_tier(stream, &h, places, targets, fault, (&mut s.sel, tsel))?;
        return hybrid.enqueue_tier_go(stream, l, p.row);
    }
    let target = hybrid.boundary_mut().handoff_target_of(p.row)?;
    p.k.ffn
        .enqueue_handoff(stream, &h, target, fault, &mut s.sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, p.row)
}

/// Enqueue layer `l`'s card experts, when it has any, and its shared expert
/// in its host leg's shadow, after its [`front`]. On a layer the expert tier
/// holds experts of, the card sum waits for the tier's rows: it runs in the
/// [`back`].
pub(crate) fn shadow(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    let c = p.cfg[l];
    let n = p.d.embd;
    let card = p.card.layer(l);
    let tiered = p.card.tier_k(l) > 0;
    if let Some(cl) = card {
        card_slots(gpu, w, p, boundary, l, cl, tiered)?;
    }
    let sh = moe_names(p, l)?;
    let s = &mut *p.s;
    p.k.experts.enqueue_shexp_gate_up(
        gpu.stream(),
        weight(w, sh.sh_gate)?,
        weight(w, sh.sh_up)?,
        boundary.normed(),
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, sh.sh_down, &s.h, &mut s.sh_y)?;
    if card.is_some() && !tiered {
        let k = card_row_mut(&mut p.card.rows, p.row)?;
        gpu.elem()
            .enqueue_add(gpu.stream(), &k.acc, &s.sh_y, n, &mut k.pre)?;
    }
    Ok(())
}

/// Layer `l`'s card slots into the card sum `acc` (module doc): every slot
/// the handoff placed below `cl.n_card` on the card, the host's left alone;
/// on a `tiered` layer their downs alone, which the [`back`]'s card sum
/// reads beside the tier's rows.
fn card_slots(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
    cl: CardLayer,
    tiered: bool,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n = p.d.embd;
    let fault = gpu.layer_sink(l)?;
    let s = &*p.s;
    let CardExperts {
        kq,
        batch,
        ff,
        rows,
        ..
    } = &mut *p.card;
    let k = card_row_mut(rows, p.row)?;
    let st = CardStacks::of(w, l)?;
    gpu.enqueue_quantize_q8_1_layer(boundary.normed(), &mut k.act_x, l)?;
    let a = GateUpAct {
        wg: st.gate.w,
        wu: st.up.w,
        act: &k.act_x,
        sel: &s.sel,
        n_slots: N_USED,
        rows_per_expert: *ff,
        slots_per_col: N_USED,
        rule: Act::SwigluClamp { limit: cl.limit },
    };
    match st.gate.ty {
        GgmlType::Q4_K => kq.enqueue_gate_up_q4k(stream, &a, fault, &mut k.h)?,
        GgmlType::Q5_K => kq.enqueue_gate_up_q5k(stream, &a, fault, &mut k.h)?,
        _ => return Err(unrun(&st.gate)),
    }
    let q = QuantSel {
        x: &k.h,
        cols: 0..N_USED,
        sel: &s.sel,
        n_card: cl.n_card,
    };
    gpu.q4k_sel()
        .enqueue_quantize_sel(stream, &q, fault, &mut k.act_h)?;
    match st.down.ty {
        GgmlType::Q5_K => {
            let a = SelDown {
                w: st.down.w,
                act: &k.act_h,
                sel: &s.sel,
                n_slots: N_USED,
                rows_per_expert: n,
            };
            kq.enqueue_gemv_q5k_sel(stream, &a, fault, &mut k.down)?;
        }
        GgmlType::Q4_K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
            stream,
            st.down.w,
            &k.act_h,
            &s.sel,
            N_USED,
            n,
            &mut k.down,
        )?,
        _ => return Err(unrun(&st.down)),
    }
    if tiered {
        return Ok(());
    }
    let acc = CardAcc {
        down: &k.down,
        w: &s.rout.weights,
        sel: &s.sel,
        n,
        m: 1,
        n_card: cl.n_card,
        n_used: N_USED,
    };
    batch.enqueue_card_acc(stream, &acc, &mut k.acc)
}

/// A prompt batch's card slots ([`card_rows`]): the batch's `t` normed rows
/// and their routing, the places the call writes, and the buffers its chunks
/// write — each chunk's rows in q8_1 ([`COL_GROUP`] columns), its gate·up
/// rows a slot, per chunk width `c` the q8_1 of its `c · N_USED` slots'
/// columns (`act_h[c - 1]`), its downs — and the card sums, `t` rows. On a
/// `tiered` layer the downs are the whole batch's, slot-major from its first
/// slot, and no card sum is written: the back's sums them with the tier's
/// rows.
pub(crate) struct CardRows<'a> {
    pub normed: &'a DeviceBuffer<f32>,
    pub ids: &'a DeviceBuffer<u32>,
    pub weights: &'a DeviceBuffer<f32>,
    pub t: usize,
    pub sel: &'a mut DeviceBuffer<u32>,
    pub act_x: &'a mut Q8Act,
    pub h: &'a mut DeviceBuffer<f32>,
    pub act_h: &'a mut [Q8Act],
    pub down: &'a mut DeviceBuffer<f32>,
    pub acc: &'a mut DeviceBuffer<f32>,
    pub tiered: bool,
}

/// Layer `l`'s card slots over a prompt batch's `t` tokens into the card
/// sums `io.acc` ([`card_slots`]'s launches): every slot's place from the
/// slot map in one launch (`ds41_ffn_places`, the handoff's rule), then per
/// chunk of up to [`COL_GROUP`] tokens the rows' q8_1 form, the gate·up
/// `_sel` of its slots at `N_USED` a column, the q8_1 of its card slots'
/// columns, the down `_sel` and the card sums — each slot and token what the
/// one-token launches write for it, the host's slots left alone. Refused by
/// name on a layer without card experts.
pub(crate) fn card_rows(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    l: usize,
    io: CardRows<'_>,
) -> Result<(), GpuError> {
    const W: &str = "glm5next card_rows";
    let stream = gpu.stream();
    let n = p.d.embd;
    let fault = gpu.layer_sink(l)?;
    let c = p.cfg[l];
    let k = &*p.card;
    let cl = k.layer(l).ok_or(GpuError::State {
        what: W,
        missing: "card experts on the layer",
    })?;
    let st = CardStacks::of(w, l)?;
    let t = io.t;
    k.batch.enqueue_places(
        stream,
        &Places {
            ids: io.ids,
            n: t * N_USED,
            map: p.slots.buf(),
            row_off: c.row_off,
            n_expert: N_EXPERT,
        },
        fault,
        &mut *io.sel,
    )?;
    let sel: &DeviceBuffer<u32> = &*io.sel;
    for c0 in (0..t).step_by(COL_GROUP) {
        let cn = COL_GROUP.min(t - c0);
        let slots = cn * N_USED;
        let x = span(W, io.normed, c0 * n, cn * n)?;
        let sel_c = span(W, sel, c0 * N_USED, slots)?;
        let w_c = span(W, io.weights, c0 * N_USED, slots)?;
        gpu.enqueue_quantize_q8_1_cols(&x, &mut *io.act_x, cn, l)?;
        let a = GateUpAct {
            wg: st.gate.w,
            wu: st.up.w,
            act: &*io.act_x,
            sel: &sel_c,
            n_slots: slots,
            rows_per_expert: k.ff,
            slots_per_col: N_USED,
            rule: Act::SwigluClamp { limit: cl.limit },
        };
        match st.gate.ty {
            GgmlType::Q4_K => k.kq.enqueue_gate_up_q4k(stream, &a, fault, &mut *io.h)?,
            GgmlType::Q5_K => k.kq.enqueue_gate_up_q5k(stream, &a, fault, &mut *io.h)?,
            _ => return Err(unrun(&st.gate)),
        }
        let act_h = io.act_h.get_mut(cn - 1).ok_or(GpuError::State {
            what: W,
            missing: "the q8_1 columns of a chunk that wide",
        })?;
        let q = QuantSel {
            x: &*io.h,
            cols: 0..slots,
            sel: &sel_c,
            n_card: cl.n_card,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, fault, act_h)?;
        // A tiered layer keeps every chunk's downs for the back's sum.
        let d0 = if io.tiered { c0 * N_USED * n } else { 0 };
        let mut down = span_mut(W, &mut *io.down, d0, slots * n)?;
        match st.down.ty {
            GgmlType::Q5_K => {
                let a = SelDown {
                    w: st.down.w,
                    act: act_h,
                    sel: &sel_c,
                    n_slots: slots,
                    rows_per_expert: n,
                };
                k.kq.enqueue_gemv_q5k_sel(stream, &a, fault, &mut down)?;
            }
            GgmlType::Q4_K => gpu
                .q4k_sel()
                .enqueue_gemv_q4k_sel(stream, st.down.w, act_h, &sel_c, slots, n, &mut down)?,
            _ => return Err(unrun(&st.down)),
        }
        drop(down);
        if io.tiered {
            continue;
        }
        let acc = CardAcc {
            down: &*io.down,
            w: &w_c,
            sel: &sel_c,
            n,
            m: cn,
            n_card: cl.n_card,
            n_used: N_USED,
        };
        k.batch.enqueue_card_acc(
            stream,
            &acc,
            &mut *span_mut(W, &mut *io.acc, c0 * n, cn * n)?,
        )?;
    }
    Ok(())
}

/// A stack whose type [`card_routed`] names but no launch here runs.
pub(crate) fn unrun(st: &Stack<'_>) -> GpuError {
    GpuError::Tensor {
        what: CARD,
        name: st.name.clone(),
        need: "a type the card experts have an entry for in its place (gate·up Q4_K, Q5_K; down Q5_K, Q4_K)",
    }
}

/// Enqueue layer `l`'s wait and the host's routed sum plus its shadow's —
/// the card sum and the shared expert's output, or the latter alone — into
/// `out`, after its [`shadow`]. On a layer the expert tier holds experts of,
/// the wait is the host's and the tier's ([`Hybrid::enqueue_tier_back`]),
/// then the card sum over the stage card's and the tier's slots in slot
/// order (`ds41_ffn_card_acc_8_tier`) and the shared expert's output added
/// to it: the shadow's two launches of a one-card layer, after the wait.
pub(crate) fn back(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<GlmHost>,
    l: usize,
) -> Result<(), GpuError> {
    let n_tier = p.card.tier_k(l);
    if n_tier > 0 {
        return back_tier(gpu, p, hybrid, l, n_tier);
    }
    let boundary = hybrid.boundary();
    let stream = gpu.stream();
    let n = p.d.embd;
    let s = &mut *p.s;
    let shadowed = if p.card.has(l) {
        &card_row(&p.card.rows, p.row)?.pre
    } else {
        &s.sh_y
    };
    boundary.enqueue_back_of(stream, p.row)?;
    gpu.elem()
        .enqueue_add(stream, boundary.hsum_of(p.row)?, shadowed, n, &mut s.out)
}

/// [`back`] of a layer whose `n_tier` experts the expert tier holds.
fn back_tier(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    hybrid: &Hybrid<GlmHost>,
    l: usize,
    n_tier: usize,
) -> Result<(), GpuError> {
    const W: &str = "glm5next back (tier layer)";
    let stream = gpu.stream();
    let (n, row) = (p.d.embd, p.row);
    let n_card = p.card.n_card(l);
    hybrid.enqueue_tier_back(stream, l, row)?;
    let s = &mut *p.s;
    let CardExperts {
        batch, rows, tier, ..
    } = &mut *p.card;
    let tsel = tier
        .as_ref()
        .ok_or(GpuError::State {
            what: W,
            missing: "the stage card's tier side",
        })?
        .tsel(row)?;
    let k = card_row_mut(rows, row)?;
    let acc = CardAccTier {
        down: &k.down,
        trows: hybrid.tier_rows(row)?,
        w: &s.rout.weights,
        sel: &s.sel,
        tsel,
        n,
        m: 1,
        n_card,
        n_tier,
        n_used: N_USED,
    };
    batch.enqueue_card_acc_tier(stream, &acc, &mut k.acc)?;
    gpu.elem()
        .enqueue_add(stream, &k.acc, &s.sh_y, n, &mut k.pre)?;
    gpu.elem().enqueue_add(
        stream,
        hybrid.boundary().hsum_of(row)?,
        &k.pre,
        n,
        &mut s.out,
    )
}
