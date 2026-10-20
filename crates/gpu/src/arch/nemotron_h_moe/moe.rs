//! The latent MoE layer, Bolt's one layer kind the hybrid trunk does not
//! hold. In llama.cpp's order (`src/models/nemotron-h.cpp` at cd0a5b6a1,
//! `build_ffn_layer`:294-346, `build_moe_ffn` in llama-graph.cpp:2121-2490):
//!
//! 1. the layer's RMS norm of the residual row (`attn_norm`, :224) — the one
//!    row the router, the latent down and the shared expert all read
//!    (`inp_emb = inp_latent = cur`, :304-305);
//! 2. the router on the normed row (:311): the F32 `ffn_gate_inp` rows, the
//!    sigmoid, the selection bias for the pick, the top 9, the unbiased
//!    scores renormalized and scaled ([`super::router`]);
//! 3. `ffn_latent_down` (Q8_0, `hidden → latent`, :307-309);
//! 4. each kept expert on the latent: the up rows of its id (IQ2_XXS,
//!    `latent → w`, the slots' one column), the squared ReLU, the down rows
//!    (`w → latent`), every expert by `iq2_xxs_gemv_sel` (llama-graph.cpp
//!    :2321, :2422-2430, :2435);
//! 5. the slots' downs times their weights, summed in slot order
//!    (:2451-2480);
//! 6. `ffn_latent_up` (Q8_0, `latent → hidden`, :332-334);
//! 7. the shared expert on the normed row: up (IQ2_XXS, `hidden → 5376`), the
//!    squared ReLU, down (:336-341), each a one-expert `iq2_xxs_gemv_sel`;
//! 8. `moe_out + shared` (:344), then the residual add (:243).
//!
//! The expert count and width are the layer's own (its router's rows, its
//! stacks' shape), so one body runs every (count, width) pair of the file.

use super::router::{RouteOut, Routers, TOP_K};
use crate::arch::hybrid::{self, Acts, Cx, Names, Site, residual};
use crate::kquant::SelDown;
use crate::quant_sel::QuantSelKernels;
use crate::site::SiteTy;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::arch::names::blk as layer_name;
use model::arch::nemotron_h_moe::hparams::Hparams;
use model::arch::nemotron_h_moe::names::{
    EXP_PROBS_B, FFN_DOWN_EXPS, FFN_DOWN_SHEXP, FFN_GATE_INP, FFN_LATENT_DOWN, FFN_LATENT_UP,
    FFN_UP_EXPS, FFN_UP_SHEXP,
};

/// The latent pair's site types: Q8_0 through the f32 gemv.
const LATENT_TYPES: &[SiteTy] = &[SiteTy::Q8_0];

/// One latent MoE layer's resident pieces.
pub(super) struct MoeLayer {
    pub(super) norm: String,
    /// The F32 router rows, `n_expert × hidden`.
    pub(super) router: String,
    /// The selection bias, `n_expert` values.
    pub(super) bias: String,
    /// One expert's values (`expert_feed_forward_length` at the layer).
    pub(super) ff: usize,
    pub(super) latent_down: Site,
    pub(super) latent_up: Site,
    /// The experts the card holds of the layer: the router's count on a
    /// whole-card load, the plan's card segment on a placed one (a stack of
    /// `n_card` experts' rows, in the plan's order: slot `s` is the `s`-th
    /// expert the plan puts on the card).
    pub(super) n_card: usize,
    /// The IQ2_XXS stacks of those experts, in the file's blocks; `None`
    /// where `n_card` is 0, which has no tensor.
    up_exps: Option<String>,
    down_exps: Option<String>,
    pub(super) up_shexp: String,
    pub(super) down_shexp: String,
    pub(super) route: RouteOut,
}

/// The widths every MoE layer shares: the latent, the shared expert's and
/// the file's `expert_weights_scale`.
#[derive(Clone, Copy)]
pub(super) struct MoeShape {
    pub(super) latent: usize,
    pub(super) shared_ff: usize,
    pub(super) scale: f32,
}

/// The kernels a MoE layer launches beyond the trunk's.
pub(super) struct MoeKernels {
    pub(super) routers: Routers,
    pub(super) iq: QuantSelKernels,
}

/// The buffers a MoE layer runs over, m = 1.
pub(super) struct MoeScratch {
    /// The latent down's output, the experts' input.
    pub(super) lat: DeviceBuffer<f32>,
    /// The slots' up outputs, `TOP_K × w`.
    pub(super) up: DeviceBuffer<f32>,
    /// The slots' down outputs, `TOP_K × latent`.
    pub(super) down: DeviceBuffer<f32>,
    /// The weighted sum of the slots, `latent`.
    pub(super) moe: DeviceBuffer<f32>,
    /// The shared expert's up output, then its down.
    pub(super) sh_up: DeviceBuffer<f32>,
    pub(super) sh: DeviceBuffer<f32>,
    /// `moe_out + shared`, the layer's delta.
    pub(super) sum: DeviceBuffer<f32>,
    /// The one id of a one-expert stack (the shared expert's).
    pub(super) zero_id: DeviceBuffer<u32>,
}

impl MoeScratch {
    /// The scratch for `sh` over a `hidden`-value row, the widest expert
    /// width `ff`. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        sh: MoeShape,
        hidden: usize,
        ff: usize,
    ) -> Result<MoeScratch, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, n.max(1))?)
        };
        Ok(MoeScratch {
            lat: z(sh.latent)?,
            up: z(TOP_K * ff)?,
            down: z(TOP_K * sh.latent)?,
            moe: z(sh.latent)?,
            sh_up: z(sh.shared_ff)?,
            sh: z(hidden)?,
            sum: z(hidden)?,
            zero_id: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        [
            &self.lat,
            &self.up,
            &self.down,
            &self.moe,
            &self.sh_up,
            &self.sh,
            &self.sum,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.zero_id.num_bytes()
    }
}

/// The resident IQ2_XXS weight `name` of `rows` rows of `k` values, in the
/// file's blocks: its name; any other variant, type or shape is refused by
/// name as `what`'s.
fn iq2_xxs(
    w: &Weights,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
) -> Result<String, GpuError> {
    match w.resident(what, name)? {
        DevWeight::KQuant {
            ty: GgmlType::IQ2_XXS,
            w: t,
            k: got,
        } if t.rows() == rows && *got == k => Ok(name.to_string()),
        _ => Err(GpuError::shape(
            what,
            format!(
                "{name} is not an IQ2_XXS weight of {rows} rows of {k} values in the file's blocks"
            ),
        )),
    }
}

/// The resident stack `name` a launch reads.
pub(super) fn stack<'w>(
    w: &'w Weights,
    what: &'static str,
    name: &str,
) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(what, name)? {
        DevWeight::KQuant { w, .. } => Ok(w),
        _ => Err(GpuError::tensor(
            what,
            name,
            "a weight in the file's blocks",
        )),
    }
}

impl MoeLayer {
    /// Latent MoE layer `l` of `hp` with `n_card` of its experts resident
    /// (the router's count for the whole card), its pre-norm `norm` resolved
    /// by the caller. A layer whose count no router instance is built for,
    /// whose top-k is not [`TOP_K`], whose kept weights are not renormalized
    /// or whose `n_card` is past its experts is refused by name.
    /// Load-time only.
    pub(super) fn load(
        names: &Names<'_>,
        (l, norm): (usize, String),
        hp: &Hparams,
        n_card: usize,
        stream: &CudaStream,
    ) -> Result<MoeLayer, GpuError> {
        let what = names.what;
        let x = hp.experts[l].ok_or_else(|| {
            GpuError::shape(
                what,
                format!("layer {l} is a MoE layer with no experts read"),
            )
        })?;
        if x.top_k != TOP_K || !hp.weights_norm {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {l} keeps {} experts, renormalized {}: the router keeps {TOP_K}, \
                     renormalized",
                    x.top_k, hp.weights_norm
                ),
            ));
        }
        let (hidden, lat, shf, n, ff) = (hp.n_embd, hp.latent, hp.shared_ff, x.n_expert, x.ff);
        if n_card > n {
            return Err(GpuError::shape(
                what,
                format!("layer {l}: {n_card} experts on the card of its {n}"),
            ));
        }
        let router = layer_name(l, FFN_GATE_INP);
        let rows = names.w.f32_tensor(what, &router)?;
        if (rows.rows(), rows.cols()) != (n, hidden) {
            return Err(GpuError::shape(
                what,
                format!("{router} is not {n} f32 rows of {hidden} values"),
            ));
        }
        let w = names.w;
        let layer = MoeLayer {
            router,
            bias: names.vector(l, EXP_PROBS_B, n)?,
            ff,
            latent_down: names.site(
                &layer_name(l, FFN_LATENT_DOWN),
                (lat, hidden),
                LATENT_TYPES,
            )?,
            latent_up: names.site(&layer_name(l, FFN_LATENT_UP), (hidden, lat), LATENT_TYPES)?,
            n_card,
            up_exps: (n_card > 0)
                .then(|| iq2_xxs(w, what, &layer_name(l, FFN_UP_EXPS), (n_card * ff, lat)))
                .transpose()?,
            down_exps: (n_card > 0)
                .then(|| iq2_xxs(w, what, &layer_name(l, FFN_DOWN_EXPS), (n_card * lat, ff)))
                .transpose()?,
            up_shexp: iq2_xxs(w, what, &layer_name(l, FFN_UP_SHEXP), (shf, hidden))?,
            down_shexp: iq2_xxs(w, what, &layer_name(l, FFN_DOWN_SHEXP), (hidden, shf))?,
            route: RouteOut::new(what, stream, n)?,
            norm,
        };
        Ok(layer)
    }

    /// The q8_1 forms (values, columns) the layer's launches read: the latent,
    /// the slots' squared ReLU, the normed row and the shared expert's.
    pub(super) fn forms(&self, hp: &Hparams) -> [(usize, usize); 4] {
        [
            (hp.latent, 1),
            (self.ff, TOP_K),
            (hp.n_embd, 1),
            (hp.shared_ff, 1),
        ]
    }

    pub(super) fn sites(&self) -> [&Site; 2] {
        [&self.latent_down, &self.latent_up]
    }

    /// The routed up stack of the layer's card experts; a layer with none is
    /// refused by name.
    pub(super) fn up(&self) -> Result<&str, GpuError> {
        stack_name(&self.up_exps)
    }

    /// The routed down stack of the layer's card experts; a layer with none
    /// is refused by name.
    pub(super) fn down(&self) -> Result<&str, GpuError> {
        stack_name(&self.down_exps)
    }
}

/// A card stack's name, or the refusal of a layer that holds no expert on
/// the card (a placed load's walks skip such a layer's card slots).
fn stack_name(name: &Option<String>) -> Result<&str, GpuError> {
    name.as_deref().ok_or(GpuError::state(
        "nemotron_h_moe::MoeLayer",
        "a routed stack on the card (a layer with no card expert holds none)",
    ))
}

/// `y = W · col` for the one-expert IQ2_XXS stack `name` of `rows` rows: the
/// shared expert's projections, each `iq2_xxs_gemv_sel` over the stack as its
/// expert 0.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch's stack, activation, id and output, and the layer it raises in"
)]
pub(super) fn one_expert(
    cx: &Cx<'_>,
    k: &MoeKernels,
    l: usize,
    name: &str,
    rows: usize,
    act: &crate::Q8Act,
    zero_id: &DeviceBuffer<u32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let a = SelDown {
        w: stack(cx.w, cx.what, name)?,
        act,
        sel: zero_id,
        n_slots: 1,
        rows_per_expert: rows,
    };
    k.iq.enqueue_gemv_sel(
        cx.gpu.stream(),
        GgmlType::IQ2_XXS,
        &a,
        cx.gpu.layer_sink(l)?,
        y,
    )
}

/// One latent MoE layer (the module doc's steps), `src` to `dst`.
pub(super) fn moe_layer(
    cx: &Cx<'_>,
    (k, sh): (&MoeKernels, MoeShape),
    (l, m): (usize, &mut MoeLayer),
    (t, s, acts): (&mut hybrid::Scratch, &mut MoeScratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let (gpu, what) = (cx.gpu, cx.what);
    let stream = gpu.stream();
    let hidden = cx.sh.hidden;
    let fault = gpu.layer_sink(l)?;
    // One launch writes the normed row the router and the latent read and
    // the q8_1 form the shared expert's up reads, the norm + quantize pair
    // it folds, bit for bit.
    gpu.fused().enqueue_norm_quant(
        stream,
        src,
        cx.f32(&m.norm)?,
        cx.sh.eps,
        acts.q8_1_at(what, hidden, 1)?,
        &mut t.buf.xn,
        fault,
    )?;
    {
        // The shared expert's up consumes the norm's (hidden, 1) q8_1 form
        // here, before the latent's quantize overwrites it: Acts keys forms
        // by (k, m) alone, and the latent width may equal the hidden, making
        // the norm's and the latent's forms one buffer (the latent pair's
        // Q8_0 projections write no q8_1 form).
        let act = acts.q8_1_at(what, hidden, 1)?;
        one_expert(
            cx,
            k,
            l,
            &m.up_shexp,
            sh.shared_ff,
            act,
            &s.zero_id,
            &mut s.sh_up,
        )?;
    }
    k.routers.enqueue(
        stream,
        cx.w.f32_tensor(what, &m.router)?,
        &t.buf.xn,
        cx.f32(&m.bias)?,
        sh.scale,
        &mut m.route,
        fault,
    )?;
    m.latent_down
        .project(&cx.gqa(), acts, l, &t.buf.xn, &mut s.lat)?;
    {
        let act = acts.q8_1_at(what, sh.latent, 1)?;
        gpu.enqueue_quantize_q8_1_layer(&s.lat, act, l)?;
        let up = SelDown {
            w: stack(cx.w, what, m.up()?)?,
            act,
            sel: m.route.ids(),
            n_slots: TOP_K,
            rows_per_expert: m.ff,
        };
        k.iq.enqueue_gemv_sel_cols(stream, GgmlType::IQ2_XXS, &up, TOP_K, fault, &mut s.up)?;
    }
    {
        let act = acts.q8_1_at(what, m.ff, TOP_K)?;
        gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.up, act, l)?;
        let down = SelDown {
            w: stack(cx.w, what, m.down()?)?,
            act,
            sel: m.route.ids(),
            n_slots: TOP_K,
            rows_per_expert: sh.latent,
        };
        k.iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut s.down)?;
    }
    gpu.elem().enqueue_weighted_sum(
        stream,
        &s.down,
        m.route.weights(),
        sh.latent,
        TOP_K as u32,
        1,
        &mut s.moe,
    )?;
    m.latent_up
        .project(&cx.gqa(), acts, l, &s.moe, &mut t.buf.delta)?;
    {
        let act = acts.q8_1_at(what, sh.shared_ff, 1)?;
        gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.sh_up, act, l)?;
        one_expert(cx, k, l, &m.down_shexp, hidden, act, &s.zero_id, &mut s.sh)?;
    }
    gpu.elem()
        .enqueue_add(stream, &t.buf.delta, &s.sh, hidden, &mut s.sum)?;
    residual(cx, src, &s.sum, dst)
}

/// One MoE layer's last routing, read back: the kept expert ids in rank
/// order, their weights, and every expert's unbiased score.
pub struct Routing {
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub scores: Vec<f32>,
}

/// Layer `m`'s last routing (a gate's instrument; it synchronizes `stream`).
pub(super) fn routed(m: &MoeLayer, stream: &CudaStream) -> Result<Routing, GpuError> {
    Ok(Routing {
        ids: m.route.ids().to_host_vec(stream)?,
        weights: m.route.weights().to_host_vec(stream)?,
        scores: m.route.probs().to_host_vec(stream)?,
    })
}
