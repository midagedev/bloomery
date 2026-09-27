//! One layer alone at one position on streams the caller gives
//! ([`Body::forced_row`]): the teacher-forced arm a gate runs on the
//! reference's own layer inputs, and a layer-by-layer run a gate holds to the
//! chain bit for bit.
//!
//! The row runs the step's launches of that layer in the step's order, eager
//! on the engine stream, and reads back what a gate compares between them:
//! the mixer sub-layer's fold, its output and the streams after it; the
//! feed-forward sub-layer's fold and normed input, the router's results, the
//! routed sum and the shared expert's output, the block's output and the
//! streams after the layer. The layer's store holds what the earlier
//! positions of the same run wrote: the caller resets the model before
//! position 0 of each layer's run.

use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::hc::{HC_STREAMS, HcPostArgs, HcQ8Params, HcQ8PreArgs};
use model::arch::glm5next::names::{self, Sub};
use runtime::layer::{FfnKind, MixerKind};

use crate::body::{Body, Parts, f32v, q8};
use crate::{ffn, kda, mla};

/// What the errors name.
const WHAT: &str = "glm5next Body::forced_row";

/// A routed block's results at one position.
#[derive(Clone, Debug)]
pub struct ForcedRoute {
    /// The router's logits and sigmoid scores, one per expert.
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    /// The selection bias the scores were ranked with (zeros where the file
    /// has none): the ranked value of expert `e` is `probs[e] + bias[e]`.
    pub bias: Vec<f32>,
    /// The chosen experts in slot order and their normalized, scaled
    /// weights.
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    /// The host tier's routed sum and the shared expert's output.
    pub routed: Vec<f32>,
    pub shared: Vec<f32>,
}

/// One layer at one position, read back.
#[derive(Clone, Debug)]
pub struct ForcedRow {
    /// The mixer sub-layer's input (the streams' fold by its own mix), the
    /// mixer's output and the streams after the sub-layer.
    pub mix_in: Vec<f32>,
    pub mix_out: Vec<f32>,
    pub mixed: Vec<f32>,
    /// The feed-forward sub-layer's fold and its normed form.
    pub ffn_in: Vec<f32>,
    pub ffn_normed: Vec<f32>,
    /// A routed block's router and sums; `None` for a dense block.
    pub route: Option<ForcedRoute>,
    /// The block's output and the streams after the layer.
    pub ffn_out: Vec<f32>,
    pub out: Vec<f32>,
}

impl Body {
    /// Layer `l` alone at position `pos` on the streams `x` (`4 · n_embd`,
    /// stream-major, as a layer's tap holds them), eager and blocking. With
    /// `ffn_x`, the feed-forward sub-layer reads those streams in place of
    /// the mixer sub-layer's (its residual too). Refused by name: a layer or
    /// a position outside the load, streams of another length. A fault a
    /// launch raised is the call's error.
    pub fn forced_row(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        pos: u32,
        x: &[f32],
        ffn_x: Option<&[f32]>,
    ) -> Result<ForcedRow, GpuError> {
        let ctx = self.ctx();
        let (mut p, hybrid) = self.parts();
        let stream = gpu.stream();
        let n = p.d.embd;
        let want = HC_STREAMS * n;
        let kind = p
            .cfg
            .get(l)
            .map(|c| c.kind)
            .ok_or_else(|| shape(format!("layer {l} past the {} loaded", p.cfg.len())))?;
        if pos as usize >= ctx {
            return Err(shape(format!("position {pos} in stores of {ctx}")));
        }
        if x.len() != want || ffn_x.is_some_and(|f| f.len() != want) {
            return Err(shape(format!(
                "streams of {} and {:?} values, want {want}",
                x.len(),
                ffn_x.map(<[f32]>::len)
            )));
        }
        p.s.streams[0].copy_from_host(stream, x)?;
        p.s.pos.copy_from_host(stream, &[pos])?;
        p.s.vis.copy_from_host(stream, &[0, pos + 1])?;
        let mut cur = 0;
        hc_in(gpu, w, &mut p, cur, l, Sub::Attn)?;
        let mix_in = p.s.x.to_host_vec(stream)?;
        match kind.mixer {
            MixerKind::DeltaRule => kda::kda(gpu, w, &mut p, l)?,
            MixerKind::Latent => mla::mla(gpu, w, &mut p, l)?,
            MixerKind::Gqa => return Err(shape(format!("layer {l}: a GQA mixer"))),
        }
        let mix_out = p.s.out.to_host_vec(stream)?;
        cur = hc_out(gpu, &mut p, cur)?;
        let mixed = p.s.streams[cur].to_host_vec(stream)?;
        if let Some(f) = ffn_x {
            p.s.streams[cur].copy_from_host(stream, f)?;
        }
        hc_in(gpu, w, &mut p, cur, l, Sub::Ffn)?;
        let ffn_in = p.s.x.to_host_vec(stream)?;
        let (ffn_normed, route) = match kind.ffn {
            FfnKind::Dense => {
                ffn::dense(gpu, w, &mut p, l)?;
                (p.s.xn.to_host_vec(stream)?, None)
            }
            FfnKind::Moe => {
                hybrid.open_step(stream, 1, 1)?;
                ffn::front(gpu, w, &mut p, hybrid, l)?;
                ffn::shadow(gpu, w, &mut p, hybrid.boundary(), l)?;
                ffn::back(gpu, &mut p, hybrid.boundary())?;
                hybrid.row_enqueued(l, 0)?;
                let bias = if p.cfg[l].bias {
                    f32v(w, &names::exp_probs_b(l))?
                } else {
                    &p.s.no_bias
                };
                let r = &p.s.rout;
                let route = ForcedRoute {
                    logits: r.logits.to_host_vec(stream)?,
                    probs: r.probs.to_host_vec(stream)?,
                    bias: bias.to_host_vec(stream)?,
                    ids: r.ids.to_host_vec(stream)?,
                    weights: r.weights.to_host_vec(stream)?,
                    routed: hybrid.boundary().hsum_of(0)?.to_host_vec(stream)?,
                    shared: p.s.sh_y.to_host_vec(stream)?,
                };
                (hybrid.boundary().normed().to_host_vec(stream)?, Some(route))
            }
        };
        let ffn_out = p.s.out.to_host_vec(stream)?;
        cur = hc_out(gpu, &mut p, cur)?;
        let out = p.s.streams[cur].to_host_vec(stream)?;
        if let Some(fault) = gpu.fault()? {
            return Err(GpuError::fault(WHAT, fault));
        }
        Ok(ForcedRow {
            mix_in,
            mix_out,
            mixed,
            ffn_in,
            ffn_normed,
            route,
            ffn_out,
            out,
        })
    }
}

/// A shape the forced row refuses, by name.
fn shape(detail: String) -> GpuError {
    GpuError::Shape { what: WHAT, detail }
}

/// Sub-layer `sub` of layer `l`'s input from stream buffer `cur`: its own
/// mix of the streams (`hc_pre_q8_0`) and their fold by it, into `x` — the
/// step program's launches.
fn hc_in(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    cur: usize,
    l: usize,
    sub: Sub,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let (qs, dd) = q8(w, &names::hc_fn(l, sub))?;
    let params = HcQ8Params {
        qs,
        d: dd,
        scale: f32v(w, &names::hc_scale(l, sub))?,
        base: f32v(w, &names::hc_base(l, sub))?,
        eps: d.hc_eps,
        iters: d.hc_iters,
    };
    let hc = &p.k.hc;
    let s = &mut *p.s;
    let streams = &s.streams[cur];
    hc.enqueue_pre_q8_0(
        stream,
        &HcQ8PreArgs {
            params: &params,
            x: streams,
            tokens: 1,
            rms_eps: d.rms_eps,
        },
        1,
        &mut s.hc_scratch,
        &mut s.mixes,
        &mut s.hc,
    )?;
    hc.enqueue_fold(stream, streams, &s.hc, d.embd, 1, &mut s.x)
}

/// The sub-layer's output into the other stream buffer by its mix
/// (`hc_post`), residual from `cur`; returns the buffer written — the step
/// program's launch.
fn hc_out(gpu: &Gpu, p: &mut Parts<'_>, cur: usize) -> Result<usize, GpuError> {
    let n = p.d.embd;
    let hc = &p.k.hc;
    let s = &mut *p.s;
    let [a, b] = &mut s.streams;
    let (res, next) = if cur == 0 { (&*a, b) } else { (&*b, a) };
    hc.enqueue_post(
        gpu.stream(),
        &HcPostArgs {
            x: &s.out,
            res,
            hc: &s.hc,
            n_embd: n,
            tokens: 1,
        },
        next,
        &mut s.fold,
    )?;
    Ok(cur ^ 1)
}
