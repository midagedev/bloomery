//! The MoE layer, the one layer kind the hybrid trunk does not hold. In
//! llama.cpp's order (`src/models/nemotron-h.cpp` at cd0a5b6a1,
//! `build_ffn_layer`:294-346, `build_moe_ffn` in llama-graph.cpp:2121-2490):
//!
//! 1. the layer's RMS norm of the residual row (`attn_norm`, :224) — the one
//!    row the router, the latent down and the shared expert all read
//!    (`inp_emb = inp_latent = cur`, :304-305);
//! 2. the router on the normed row (:311): the F32 `ffn_gate_inp` rows, the
//!    sigmoid, the selection bias for the pick, the top of the file, the
//!    unbiased scores renormalized and scaled ([`super::router`]);
//! 3. `ffn_latent_down` (Q8_0, `hidden → latent`, :307-309), where the file
//!    has a latent; a file without one (Lightning) has the experts act on
//!    the normed row itself;
//! 4. each kept expert on the latent: the up rows of its id (IQ2_XXS,
//!    `latent → w`, the slots' one column), the squared ReLU, the down rows
//!    (`w → latent`), every expert by `iq2_xxs_gemv_sel` (llama-graph.cpp
//!    :2321, :2422-2430, :2435);
//! 5. the slots' downs times their weights, summed in slot order
//!    (:2451-2480);
//! 6. `ffn_latent_up` (Q8_0, `latent → hidden`, :332-334), where the file
//!    has a latent; without one the experts' sum is the hidden row;
//! 7. the shared expert on the normed row: up (IQ2_XXS, `hidden → 5376`), the
//!    squared ReLU, down (:336-341), each a one-expert `iq2_xxs_gemv_sel`; a
//!    file whose shared expert is Q8_0 runs it as a dense site instead;
//! 8. `moe_out + shared` (:344), then the residual add (:243).
//!
//! The expert count and width are the layer's own (its router's rows, its
//! stacks' shape), so one body runs every (count, width) pair of the file.

use super::router::{RouteOut, Routers};
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

/// The shared expert's site types: Q8_0 through the f32 gemv.
const SHARED_TYPES: &[SiteTy] = &[SiteTy::Q8_0];

/// The shared expert's two projections: the IQ2_XXS stack the card's
/// one-expert sel reads (Bolt), or the Q8_0 sites its gemv reads (a file
/// whose shared expert is Q8_0, Lightning).
pub(super) enum Shared {
    /// The IQ2_XXS stacks, in the file's blocks.
    Sel { up: String, down: String },
    /// The Q8_0 sites.
    Site { up: Site, down: Site },
}

/// One MoE layer's resident pieces.
pub(super) struct MoeLayer {
    pub(super) norm: String,
    /// The F32 router rows, `n_expert × hidden`.
    pub(super) router: String,
    /// The selection bias, `n_expert` values.
    pub(super) bias: String,
    /// One expert's values (`expert_feed_forward_length` at the layer).
    pub(super) ff: usize,
    /// The experts a token keeps (`expert_used_count` at the layer).
    pub(super) top_k: usize,
    /// The latent down and up, where the file has a latent.
    pub(super) latent_down: Option<Site>,
    pub(super) latent_up: Option<Site>,
    /// The experts the card holds of the layer: the router's count on a
    /// whole-card load, the plan's card segment on a placed one (a stack of
    /// `n_card` experts' rows, in the plan's order: slot `s` is the `s`-th
    /// expert the plan puts on the card).
    pub(super) n_card: usize,
    /// The IQ2_XXS stacks of those experts, in the file's blocks; `None`
    /// where `n_card` is 0, which has no tensor.
    up_exps: Option<String>,
    down_exps: Option<String>,
    /// The shared expert's two projections.
    pub(super) shared: Shared,
    pub(super) route: RouteOut,
}

/// The widths every MoE layer shares: the latent, the shared expert's, the
/// experts a token keeps and the file's `expert_weights_scale`.
#[derive(Clone, Copy)]
pub(super) struct MoeShape {
    /// The width the routed experts read and write: the latent the file's
    /// down projection makes, or the hidden row where the file has no
    /// latent.
    pub(super) width: usize,
    /// The experts a token keeps (`expert_used_count`).
    pub(super) top_k: usize,
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
    /// The slots' up outputs, `top_k × w`.
    pub(super) up: DeviceBuffer<f32>,
    /// The slots' down outputs, `top_k × width`.
    pub(super) down: DeviceBuffer<f32>,
    /// The weighted sum of the slots, `width`.
    pub(super) moe: DeviceBuffer<f32>,
    /// The shared expert's up output, its relu², then its down.
    pub(super) sh_up: DeviceBuffer<f32>,
    pub(super) sh_relu: DeviceBuffer<f32>,
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
        let width = sh.width;
        Ok(MoeScratch {
            lat: z(width)?,
            up: z(sh.top_k * ff)?,
            down: z(sh.top_k * width)?,
            moe: z(width)?,
            sh_up: z(sh.shared_ff)?,
            sh_relu: z(sh.shared_ff)?,
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
            &self.sh_relu,
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
        let route = RouteOut::new(what, stream, x.n_expert)?;
        hp.refuse_card_keeps(l, route.n_used())
            .map_err(|e| GpuError::shape(what, e))?;
        let (hidden, shf, n, ff) = (hp.n_embd, hp.shared_ff, x.n_expert, x.ff);
        let width = hp.experts_width() as usize;
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
        // The latent pair, where the file has a latent.
        let (latent_down, latent_up) = match hp.latent {
            Some(lat) => (
                Some(names.site(&layer_name(l, FFN_LATENT_DOWN), (lat, hidden), LATENT_TYPES)?),
                Some(names.site(&layer_name(l, FFN_LATENT_UP), (hidden, lat), LATENT_TYPES)?),
            ),
            None => (None, None),
        };
        // The shared expert: the IQ2_XXS stack the card's one-expert sel
        // reads, or the Q8_0 sites its gemv reads.
        let shared = match names
            .file
            .find(&layer_name(l, FFN_UP_SHEXP))
            .map(|(_, t)| t.ty)
        {
            Some(GgmlType::IQ2_XXS) => Shared::Sel {
                up: iq2_xxs(w, what, &layer_name(l, FFN_UP_SHEXP), (shf, hidden))?,
                down: iq2_xxs(w, what, &layer_name(l, FFN_DOWN_SHEXP), (hidden, shf))?,
            },
            Some(GgmlType::Q8_0) => Shared::Site {
                up: names.site(&layer_name(l, FFN_UP_SHEXP), (shf, hidden), SHARED_TYPES)?,
                down: names.site(&layer_name(l, FFN_DOWN_SHEXP), (hidden, shf), SHARED_TYPES)?,
            },
            other => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "layer {l}: the shared expert is {other:?}, neither the iq2_xxs stack the \
                         card's sel reads nor the q8_0 site its gemv reads"
                    ),
                ));
            }
        };
        let layer = MoeLayer {
            router,
            bias: names.vector(l, EXP_PROBS_B, n)?,
            ff,
            top_k: x.top_k,
            latent_down,
            latent_up,
            n_card,
            up_exps: (n_card > 0)
                .then(|| iq2_xxs(w, what, &layer_name(l, FFN_UP_EXPS), (n_card * ff, width)))
                .transpose()?,
            down_exps: (n_card > 0)
                .then(|| iq2_xxs(w, what, &layer_name(l, FFN_DOWN_EXPS), (n_card * width, ff)))
                .transpose()?,
            shared,
            route,
            norm,
        };
        Ok(layer)
    }

    /// The q8_1 forms (values, columns) the layer's launches read.
    pub(super) fn forms(&self, hp: &Hparams) -> Vec<(usize, usize)> {
        forms_of(
            (self.n_card, self.shared_is_sel()),
            (
                hp.experts_width() as usize,
                self.ff,
                hp.n_embd,
                hp.shared_ff,
                hp.top_k_max(),
            ),
        )
    }

    /// Whether the shared expert is an IQ2_XXS stack the card's one-expert
    /// `_sel` launch reads (the q8_1 forms) rather than dense sites.
    pub(super) fn shared_is_sel(&self) -> bool {
        matches!(self.shared, Shared::Sel { .. })
    }

    /// The sites the layer's launches read: the latent pair, where the file
    /// has one, and the shared expert's Q8_0 sites.
    pub(super) fn sites(&self) -> Vec<&Site> {
        let mut sites = Vec::new();
        if let Some(s) = &self.latent_down {
            sites.push(s);
        }
        if let Some(s) = &self.latent_up {
            sites.push(s);
        }
        if let Shared::Site { up, down } = &self.shared {
            sites.push(up);
            sites.push(down);
        }
        sites
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

/// The q8_1 forms (values, columns) a MoE layer's launches read: the
/// experts' input and the slots' squared ReLU where the card holds experts
/// of the layer (`n_card` above 0), and the normed row and the shared
/// expert's where the shared expert is an IQ2_XXS stack (`shared_is_sel`).
/// A launch that reads no q8_1 form asks for none: the form's block is the
/// K-quant's 256 values, and a width no K-quant form takes (a file whose
/// sites are Q8_0 alone) would refuse the load for a form nothing reads.
///
/// The widths are the experts' (`width`, the latent the file's down
/// projection makes or the hidden row where it has none), the slots' squared
/// ReLU (`ff`), the normed row (`hidden`), the shared expert's
/// (`shared_ff`) and the experts a token keeps (`top_k`).
fn forms_of(
    (n_card, shared_is_sel): (usize, bool),
    (width, ff, hidden, shared_ff, top_k): (usize, usize, usize, usize, usize),
) -> Vec<(usize, usize)> {
    let mut forms = Vec::new();
    if n_card > 0 {
        forms.push((width, 1));
        forms.push((ff, top_k));
    }
    if shared_is_sel {
        forms.push((hidden, 1));
        forms.push((shared_ff, 1));
    }
    forms
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

/// One MoE layer (the module doc's steps), `src` to `dst`.
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
    let width = sh.width;
    let top_k = sh.top_k;
    let fault = gpu.layer_sink(l)?;
    // The normed row the router and the latent read. A K-quant site reads
    // the row as the q8_1 form the fused launch writes beside it (the norm +
    // quantize pair it folds, bit for bit); a file whose shared expert is a
    // dense site has no such form, and the plain norm writes the row alone.
    match &m.shared {
        Shared::Sel { .. } => {
            gpu.fused().enqueue_norm_quant(
                stream,
                src,
                cx.f32(&m.norm)?,
                cx.sh.eps,
                acts.q8_1_at(what, hidden, 1)?,
                &mut t.buf.xn,
                fault,
            )?;
        }
        Shared::Site { .. } => {
            hybrid::norm(cx, &m.norm, src, &mut t.buf.xn)?;
        }
    }
    {
        // The shared expert's up consumes the norm's (hidden, 1) q8_1 form
        // here, before the experts' quantize overwrites it: Acts keys forms
        // by (k, m) alone, and the experts' width may equal the hidden,
        // making the norm's and the experts' forms one buffer (the latent
        // pair's Q8_0 projections write no q8_1 form). The Q8_0 site reads
        // the normed row itself.
        match &m.shared {
            Shared::Sel { up, .. } => {
                let act = acts.q8_1_at(what, hidden, 1)?;
                one_expert(cx, k, l, up, sh.shared_ff, act, &s.zero_id, &mut s.sh_up)?;
            }
            Shared::Site { up, .. } => {
                up.project(&cx.gqa(), acts, l, &t.buf.xn, &mut s.sh_up)?;
            }
        }
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
    // The experts' input: the latent the file's down projection makes, or
    // the normed row itself where the file has no latent.
    if let Some(site) = &m.latent_down {
        site.project(&cx.gqa(), acts, l, &t.buf.xn, &mut s.lat)?;
    }
    {
        let act = acts.q8_1_at(what, width, 1)?;
        let input = if m.latent_down.is_some() {
            &s.lat
        } else {
            &t.buf.xn
        };
        gpu.enqueue_quantize_q8_1_layer(input, act, l)?;
        let up = SelDown {
            w: stack(cx.w, what, m.up()?)?,
            act,
            sel: m.route.ids(),
            n_slots: top_k,
            rows_per_expert: m.ff,
        };
        k.iq.enqueue_gemv_sel_cols(stream, GgmlType::IQ2_XXS, &up, top_k, fault, &mut s.up)?;
    }
    {
        let act = acts.q8_1_at(what, m.ff, top_k)?;
        gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.up, act, l)?;
        let down = SelDown {
            w: stack(cx.w, what, m.down()?)?,
            act,
            sel: m.route.ids(),
            n_slots: top_k,
            rows_per_expert: width,
        };
        k.iq.enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut s.down)?;
    }
    gpu.elem().enqueue_weighted_sum(
        stream,
        &s.down,
        m.route.weights(),
        width,
        top_k as u32,
        1,
        &mut s.moe,
    )?;
    // The layer's delta: the latent up's projection, or the experts' sum
    // itself where the file has no latent.
    let delta: &DeviceBuffer<f32> = match &m.latent_up {
        Some(site) => {
            site.project(&cx.gqa(), acts, l, &s.moe, &mut t.buf.delta)?;
            &t.buf.delta
        }
        None => &s.moe,
    };
    {
        match &m.shared {
            Shared::Sel { down, .. } => {
                let act = acts.q8_1_at(what, sh.shared_ff, 1)?;
                gpu.enqueue_quantize_q8_1_layer_relu_sqr(&s.sh_up, act, l)?;
                one_expert(cx, k, l, down, hidden, act, &s.zero_id, &mut s.sh)?;
            }
            Shared::Site { down, .. } => {
                gpu.elem()
                    .enqueue_relu_sqr(stream, &s.sh_up, sh.shared_ff, &mut s.sh_relu)?;
                down.project(&cx.gqa(), acts, l, &s.sh_relu, &mut s.sh)?;
            }
        }
    }
    gpu.elem()
        .enqueue_add(stream, delta, &s.sh, hidden, &mut s.sum)?;
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

#[cfg(test)]
mod tests {
    use super::forms_of;

    /// A layer whose card holds no expert and whose shared expert is a
    /// dense site reads no q8_1 form: the K-quant's 256-value blocks are not
    /// allocated for it, so a width no K-quant form takes (a file whose
    /// sites are Q8_0 alone) does not refuse the load for a form nothing
    /// reads.
    #[test]
    fn a_layer_that_reads_no_q8_1_form_asks_for_none() {
        assert!(forms_of((0, false), (2688, 1856, 2688, 3712, 6)).is_empty());
    }

    /// A layer whose card holds experts asks for the experts' input and the
    /// slots' squared ReLU, and one whose shared expert is an IQ2_XXS stack
    /// asks for the normed row and the shared expert's.
    #[test]
    fn a_layer_asks_for_the_forms_its_launches_read() {
        assert_eq!(
            forms_of((240, true), (1024, 1536, 4096, 5376, 9)),
            vec![(1024, 1), (1536, 9), (4096, 1), (5376, 1)]
        );
        assert_eq!(
            forms_of((240, false), (1024, 1536, 4096, 5376, 9)),
            vec![(1024, 1), (1536, 9)]
        );
    }
}
