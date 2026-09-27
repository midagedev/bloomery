//! The GLM layer program of the one-token step ([`StepProgram`]) and the
//! port its host legs go through ([`HostLeg`]): [`runtime::sched::walk`]
//! over the point `(1, 1, Step)`. Each layer's parts, by its programs
//! (`runtime::layer::Layer`), never its number:
//!
//! - the front: the mixer sub-layer whole — `hc_pre`, the fold, the KDA or
//!   latent mixer, `hc_post` — then the feed-forward sub-layer's `hc_pre`
//!   and fold, and either the dense block and its `hc_post` or the routed
//!   block up to its go;
//! - the shadow (a routed layer): the shared expert;
//! - the back (a routed layer): the wait, the sum and `hc_post`, then the
//!   host tier told the layer is enqueued.
//!
//! A layer's streams go to its tap, where a gate armed one, once its last
//! `hc_post` is enqueued. The begin starts from stream buffer 0, which the
//! embedding wrote; the end is the streams' mean into the head, and the
//! head.
//!
//! Launches per layer: 3 a sub-layer for the streams (`hc_pre`, the fold,
//! `hc_post`), plus the mixer's ([`crate::kda::LAUNCHES`],
//! [`crate::mla::LAUNCHES`]) and the block's ([`crate::ffn::DENSE_LAUNCHES`],
//! [`crate::ffn::MOE_LAUNCHES`]); the head adds the mean and its own three.

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::hc::{HcPostArgs, HcQ8Params, HcQ8PreArgs};
use cuda_core::CudaStream;
use model::arch::glm5next::names::{self, Sub};
use runtime::layer::{FfnKind, MixerKind};
use runtime::sched::{self, At, LayerProgram, Overlap, Port, PortKind, Refused};

use crate::body::{Parts, f32v, q8};
use crate::host::GlmHost;
use crate::{ffn, kda, mla};

/// What the walk's errors name.
const WHAT: &str = "glm5next Body::enqueue_chain";

/// The launches of the step's captured chain: every layer's
/// ([`layer_launches`]) and the head's four.
#[must_use]
pub fn step_launches(mixers: &[MixerKind], ffns: &[FfnKind]) -> usize {
    mixers
        .iter()
        .zip(ffns)
        .map(|(&m, &f)| layer_launches(m, f))
        .sum::<usize>()
        + 4
}

/// One layer's launches: two sub-layers' three for the streams, its mixer's
/// and its block's.
#[must_use]
pub fn layer_launches(mixer: MixerKind, ffn: FfnKind) -> usize {
    let mix = match mixer {
        MixerKind::DeltaRule => kda::LAUNCHES,
        MixerKind::Latent => mla::LAUNCHES,
        MixerKind::Gqa => 0,
    };
    let block = match ffn {
        FfnKind::Dense => ffn::DENSE_LAUNCHES,
        FfnKind::Moe => ffn::MOE_LAUNCHES,
    };
    2 * 3 + mix + block
}

/// Walk the step over `parts`, the host tier opened for it first, into
/// `head`.
pub(crate) fn walk_step(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    head: &mut Head,
) -> Result<(), GpuError> {
    let layers = parts.cfg.len();
    let o = Overlap {
        units: 1,
        cols: 1,
        port: PortKind::Step,
    };
    let mut port = HostLeg {
        stream: gpu.stream(),
        hybrid,
    };
    let mut prog = StepProgram {
        gpu,
        w,
        parts,
        cur: 0,
        head,
    };
    sched::walk(o, layers, &mut port, &mut prog)
}

/// The step's host leg through the body's host tier.
pub(crate) struct HostLeg<'a> {
    stream: &'a CudaStream,
    hybrid: &'a mut Hybrid<GlmHost>,
}

impl Port for HostLeg<'_> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Opens the host tier's step port on the point's rows
    /// ([`Hybrid::open_step`]): one row of one column; any other point is
    /// refused by name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.hybrid.open_step(self.stream, o.units, o.cols)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT,
            detail: why.to_string(),
        }
    }
}

/// The GLM program over one walk of the step: the body's parts, the stream
/// buffer the next sub-layer reads, the head.
struct StepProgram<'s, 'w> {
    gpu: &'w Gpu,
    w: &'w Weights,
    parts: Parts<'s>,
    cur: usize,
    head: &'s mut Head,
}

impl StepProgram<'_, '_> {
    /// Sub-layer `sub` of layer `l`'s input: its own mix of the streams
    /// (`hc_pre_q8_0`) and their fold by it, into `x`.
    fn hc_in(&mut self, l: usize, sub: Sub) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let stream = gpu.stream();
        let d = *self.parts.d;
        let (qs, dd) = q8(w, &names::hc_fn(l, sub))?;
        let params = HcQ8Params {
            qs,
            d: dd,
            scale: f32v(w, &names::hc_scale(l, sub))?,
            base: f32v(w, &names::hc_base(l, sub))?,
            eps: d.hc_eps,
            iters: d.hc_iters,
        };
        let hc = &self.parts.k.hc;
        let s = &mut *self.parts.s;
        let streams = &s.streams[self.cur];
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

    /// The sub-layer's output `out` into the next streams by its mix
    /// (`hc_post`); the fold it writes beside them is never read.
    fn hc_out(&mut self) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        let n = self.parts.d.embd;
        let hc = &self.parts.k.hc;
        let cur = self.cur;
        let s = &mut *self.parts.s;
        let [a, b] = &mut s.streams;
        let (res, next) = if cur == 0 { (&*a, b) } else { (&*b, a) };
        hc.enqueue_post(
            stream,
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
        self.cur ^= 1;
        Ok(())
    }
}

impl<'s> LayerProgram for StepProgram<'s, '_> {
    type Port = HostLeg<'s>;

    /// Stream buffer 0, which the embedding wrote.
    fn begin(&mut self, _unit: usize) -> Result<(), GpuError> {
        self.cur = 0;
        Ok(())
    }

    /// The mixer sub-layer, then the block's input and the dense block with
    /// its `hc_post`, or the routed block up to its go.
    fn front(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let (gpu, w, l) = (self.gpu, self.w, at.layer);
        let kind = self.parts.cfg[l].kind;
        self.hc_in(l, Sub::Attn)?;
        match kind.mixer {
            MixerKind::DeltaRule => kda::kda(gpu, w, &mut self.parts, l)?,
            MixerKind::Latent => mla::mla(gpu, w, &mut self.parts, l)?,
            MixerKind::Gqa => {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {l}: a GQA mixer, which glm5next has none of"),
                });
            }
        }
        self.hc_out()?;
        self.hc_in(l, Sub::Ffn)?;
        match kind.ffn {
            FfnKind::Dense => {
                ffn::dense(gpu, w, &mut self.parts, l)?;
                self.hc_out()?;
                self.parts.tap(gpu, l, self.cur)
            }
            FfnKind::Moe => ffn::front(gpu, w, &mut self.parts, port.hybrid, l),
        }
    }

    /// A routed layer's shared expert under its host leg.
    fn shadow(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        if !self.parts.cfg[l].kind.host_leg() {
            return Ok(());
        }
        ffn::shadow(self.gpu, self.w, &mut self.parts, port.hybrid.boundary(), l)
    }

    /// A routed layer's wait, sum and `hc_post`, its tap, and the host tier
    /// told the layer is enqueued (an eager chain is served there).
    fn back(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let (gpu, l) = (self.gpu, at.layer);
        if !self.parts.cfg[l].kind.host_leg() {
            return Ok(());
        }
        ffn::back(gpu, &mut self.parts, port.hybrid.boundary())?;
        self.hc_out()?;
        self.parts.tap(gpu, l, self.cur)?;
        port.hybrid.row_enqueued(l, 0)
    }

    /// The streams' mean into the head, and the head.
    fn end(&mut self, _unit: usize) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let n = self.parts.d.embd;
        let s = &self.parts.s.streams[self.cur];
        self.parts
            .k
            .hc
            .enqueue_mean(gpu.stream(), s, n, 0, self.head.input_mut())?;
        self.head.enqueue(gpu, w)
    }
}
