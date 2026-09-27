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
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::chain::ffn::{CardAcc, FfnBatchKernels, Handoff};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use cuda_core::DeviceBuffer;
use gguf::GgmlType;
use model::arch::glm5next::names;
use model::arch::glm5next::place::card_routed;
use models::{Ffn, LayerSpec};

use crate::body::{Parts, f32t, f32v, gemv, weight};
use crate::host::GlmHost;

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
/// count and limit from the slot map and the description, and the one-token
/// buffers every card layer's shadow writes. Made at load, after the weights,
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

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them
    /// and `w` holds them, `n_embd` wide with routed experts `ff` wide.
    /// Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        layers: &[LayerSpec],
        map: &SlotMap,
        n_embd: usize,
        ff: usize,
    ) -> Result<CardExperts, GpuError> {
        let mut per = Vec::with_capacity(layers.len());
        for (l, spec) in layers.iter().enumerate() {
            let n_card = map.on_card(l);
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
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            h: DeviceBuffer::zeroed(stream, N_USED * ff)?,
            act_h: Q8Act::with_slots(stream, N_USED, ff)?,
            down: DeviceBuffer::zeroed(stream, N_USED * n_embd)?,
            acc: DeviceBuffer::zeroed(stream, n_embd)?,
            pre: DeviceBuffer::zeroed(stream, n_embd)?,
        })
    }

    /// Layer `l` computes experts on the card.
    pub(crate) fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    fn layer(&self, l: usize) -> Option<CardLayer> {
        self.layers.get(l).copied().flatten()
    }

    /// The card slots' weighted sum of the last card layer enqueued.
    pub(crate) fn acc(&self) -> &DeviceBuffer<f32> {
        &self.acc
    }

    /// Device bytes of the buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.h.num_bytes()
            + self.act_h.device_bytes()
            + self.down.num_bytes()
            + self.acc.num_bytes()
            + self.pre.num_bytes()
    }
}

/// What the card experts' errors name.
const CARD: &str = "glm5next CardExperts";

/// Layer `l`'s routed gate, up and down names.
fn stack_names(l: usize) -> [String; 3] {
    [
        names::ffn_gate_exps(l),
        names::ffn_up_exps(l),
        names::ffn_down_exps(l),
    ]
}

/// One resident routed stack: its name, file type and words.
struct Stack<'w> {
    name: String,
    ty: GgmlType,
    w: &'w DeviceTensor<u32>,
}

/// A layer's three resident routed stacks, each in a format [`card_routed`]
/// names and the gate and up of one type.
struct CardStacks<'w> {
    gate: Stack<'w>,
    up: Stack<'w>,
    down: Stack<'w>,
}

impl<'w> CardStacks<'w> {
    fn of(w: &'w Weights, l: usize) -> Result<CardStacks<'w>, GpuError> {
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

/// Enqueue layer `l`'s dense block (module doc).
pub(crate) fn dense(gpu: &Gpu, w: &Weights, p: &mut Parts<'_>, l: usize) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (d, c) = (*p.d, p.cfg[l]);
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::ffn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        &mut s.xn,
    )?;
    p.k.experts.enqueue_shexp_gate_up(
        stream,
        weight(w, &names::ffn_gate(l))?,
        weight(w, &names::ffn_up(l))?,
        &s.xn,
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, &names::ffn_down(l), &s.h, &mut s.out)
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
    let s = &mut *p.s;
    gpu.elem().enqueue_rms_norm(
        stream,
        &s.x,
        f32v(w, &names::ffn_norm(l))?,
        d.rms_eps,
        d.embd,
        1,
        hybrid.boundary_mut().normed_mut(),
    )?;
    let bias = if c.bias {
        f32v(w, &names::exp_probs_b(l))?
    } else {
        &s.no_bias
    };
    p.k.router.enqueue_router(
        stream,
        f32t(w, &names::ffn_gate_inp(l))?,
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
    let target = hybrid.boundary_mut().handoff_target_of(0)?;
    p.k.ffn
        .enqueue_handoff(stream, &h, target, fault, &mut s.sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, 0)
}

/// Enqueue layer `l`'s card experts, when it has any, and its shared expert
/// in its host leg's shadow, after its [`front`].
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
    if let Some(cl) = card {
        card_slots(gpu, w, p, boundary, l, cl)?;
    }
    let s = &mut *p.s;
    p.k.experts.enqueue_shexp_gate_up(
        gpu.stream(),
        weight(w, &names::ffn_gate_shexp(l))?,
        weight(w, &names::ffn_up_shexp(l))?,
        boundary.normed(),
        c.limit,
        &mut s.h,
    )?;
    gemv(gpu, w, &names::ffn_down_shexp(l), &s.h, &mut s.sh_y)?;
    if card.is_some() {
        let k = &mut *p.card;
        gpu.elem()
            .enqueue_add(gpu.stream(), &k.acc, &s.sh_y, n, &mut k.pre)?;
    }
    Ok(())
}

/// Layer `l`'s card slots into the card sum `acc` (module doc): every slot
/// the handoff placed below `cl.n_card` on the card, the host's left alone.
fn card_slots(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
    cl: CardLayer,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n = p.d.embd;
    let fault = gpu.layer_sink(l)?;
    let s = &*p.s;
    let k = &mut *p.card;
    let st = CardStacks::of(w, l)?;
    gpu.enqueue_quantize_q8_1_layer(boundary.normed(), &mut k.act_x, l)?;
    let a = GateUpAct {
        wg: st.gate.w,
        wu: st.up.w,
        act: &k.act_x,
        sel: &s.sel,
        n_slots: N_USED,
        rows_per_expert: k.ff,
        slots_per_col: N_USED,
        rule: Act::SwigluClamp { limit: cl.limit },
    };
    match st.gate.ty {
        GgmlType::Q4_K => k.kq.enqueue_gate_up_q4k(stream, &a, fault, &mut k.h)?,
        GgmlType::Q5_K => k.kq.enqueue_gate_up_q5k(stream, &a, fault, &mut k.h)?,
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
            k.kq.enqueue_gemv_q5k_sel(stream, &a, fault, &mut k.down)?;
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
    let acc = CardAcc {
        down: &k.down,
        w: &s.rout.weights,
        sel: &s.sel,
        n,
        m: 1,
        n_card: cl.n_card,
        n_used: N_USED,
    };
    k.batch.enqueue_card_acc(stream, &acc, &mut k.acc)
}

/// A stack whose type [`card_routed`] names but no launch here runs.
fn unrun(st: &Stack<'_>) -> GpuError {
    GpuError::Tensor {
        what: CARD,
        name: st.name.clone(),
        need: "a type the card experts have an entry for in its place (gate·up Q4_K, Q5_K; down Q5_K, Q4_K)",
    }
}

/// Enqueue layer `l`'s wait and the host's routed sum plus its shadow's —
/// the card sum and the shared expert's output, or the latter alone — into
/// `out`, after its [`shadow`].
pub(crate) fn back(
    gpu: &Gpu,
    p: &mut Parts<'_>,
    boundary: &Boundary,
    l: usize,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n = p.d.embd;
    let s = &mut *p.s;
    let shadowed = if p.card.has(l) { &p.card.pre } else { &s.sh_y };
    boundary.enqueue_back_of(stream, 0)?;
    gpu.elem()
        .enqueue_add(stream, boundary.hsum_of(0)?, shadowed, n, &mut s.out)
}
