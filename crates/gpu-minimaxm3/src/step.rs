//! The one-token step: `runtime::sched::walk` at `(1, 1, Step)` over the layers' programs. The front is the layer's
//! attention sub-layer (dense over every position, or sparse over the blocks its indexer picks) and then its
//! feed-forward block: a dense block whole, or a routed block up to its go; the shadow the shared expert and the
//! card experts; the back the host sum's add.
//!
//! Each layer's parts, by its programs (`runtime::layer::Layer`), never its number:
//! - the front: the attention sub-layer whole (its output is the residual after it, `x1`), then a dense block to
//!   the layer's output ([`ffn::dense`]) or the routed block up to its go ([`ffn::front`]);
//! - the shadow: the shared expert and the card experts, when the slot map puts any on the card ([`ffn::shadow`]);
//! - the back: the wait and the add of the host's sum to the shadow's, then the host tier told the layer is
//!   enqueued ([`ffn::back`]).
//!
//! The attention sub-layer is the launches below, in this order: the `attn_norm` RMS, the q, k and v projections;
//! on a sparse layer the index heads' two projections, their head kernel (norm, partial turn, append of the index
//! key), the block scores and the pick; the main head kernel (q/k norm, partial turn, append of K and V); the
//! flash over every position or over the picked blocks; the output projection and the residual add.
//!
//! A layer's output goes to its tap, where a gate armed one, once its last add is enqueued. The begin starts from
//! `x`, which the embedding wrote together with the position words; the last layer's add writes the head's input,
//! and the end is the head.

use bloomery_gpu::arch::hybrid::GqaCx;
use bloomery_gpu::flash_gqa::{FlashGqaKernels, GROUP_16, GqaArgs};
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::rope_neox::{PartialNeoxArgs, RopeNeoxKernels};
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::site::KGemvKernels;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use runtime::layer::{FfnKind, Layer};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::attn::{
    AttnArgs, IndexArgs, MsaKernels, MsaShape, PickArgs, ScoreArgs, attn_scale, blocks_for,
    enqueue_attention, enqueue_index,
};
use crate::body::{Body, Dims, Scratch, StepInput, Store, WHAT as BODY};
use crate::ffn::{self, Routed};

/// The step flash scores on the tensor cores from the query rounded to f16, which is the prefill flash's score pass
/// bit for bit: a position's bits do not depend on which of the two ran it.
pub(crate) const FLASH_MMA: bool = true;

/// What the walk's errors name.
const WHAT: &str = "minimaxm3 enqueue_step";

/// The step's point, `(1, 1, Step)`.
const STEP: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The kernels the layers launch besides `Gpu`'s own, loaded once.
pub struct Kernels {
    pub kgemv: KGemvKernels,
    pub g32: Gemm32Kernels,
    pub flash: FlashGqaKernels,
    pub rope: RopeNeoxKernels,
    pub msa: MsaKernels,
}

impl Kernels {
    /// Load the kernels into `gpu`'s context. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            rope: RopeNeoxKernels::load(ctx)?,
            msa: MsaKernels::load(ctx)?,
        })
    }

    /// Device bytes held beside the modules.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.kgemv.device_bytes()
    }
}

/// Enqueue one step of `input` into `head` (eager or under capture). The embedding's row is in `x` and the position
/// words are written (`refresh`); the position and the token are checked against the file's bounds here.
pub fn enqueue_step(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    input: StepInput,
    head: &mut Head,
) -> Result<(), GpuError> {
    let Body {
        shell,
        dims,
        cfg,
        stores,
        s,
        k,
        rope,
        shape: msa,
        ffn,
        ..
    } = body;
    if input.pos as usize >= dims.ctx {
        return Err(shape(format!(
            "position {} at or past the load's context {}",
            input.pos, dims.ctx
        )));
    }
    if input.token as usize >= dims.vocab {
        return Err(shape(format!(
            "token {} past the vocabulary of {}",
            input.token, dims.vocab
        )));
    }
    let cx = GqaCx {
        what: WHAT,
        gpu,
        w,
        kgemv: &k.kgemv,
        g32: &k.g32,
        flash: &k.flash,
        mma: FLASH_MMA,
        out_gate: None,
        hidden: dims.embd,
        eps: dims.rms_eps,
        n_kv: dims.kv_heads,
        ctx: dims.ctx,
    };
    let layers = cfg.len();
    let mut port = StepLeg::new(gpu.stream(), &mut shell.hybrid);
    let mut prog = StepProgram {
        gpu,
        w,
        cx,
        k,
        rope,
        msa,
        dims,
        cfg,
        stores,
        s,
        ffn,
        taps: shell.taps.as_deref_mut(),
        head,
    };
    sched::walk(STEP, layers, &mut port, &mut prog)
}

/// The MiniMax program over one walk: the body's parts and the head.
struct StepProgram<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    cx: GqaCx<'a>,
    k: &'a Kernels,
    rope: &'a RopeRows,
    msa: &'a MsaShape,
    dims: &'a Dims,
    cfg: &'a [Layer],
    stores: &'a mut [Store],
    s: &'a mut Scratch,
    ffn: &'a mut Routed,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
    head: &'a mut Head,
}

impl StepProgram<'_> {
    /// Whether layer `l` is the last: its output is the head's input.
    fn last(&self, l: usize) -> bool {
        l + 1 == self.cfg.len()
    }

    /// Layer `l`'s output copied into its tap when a gate armed them.
    fn tap(&mut self, l: usize) -> Result<(), GpuError> {
        let last = self.last(l);
        let Some(taps) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        let tap = taps.get_mut(l).ok_or(GpuError::State {
            what: WHAT,
            missing: "the layer's tap",
        })?;
        let out = if last {
            &*self.head.input_mut()
        } else {
            &self.s.x
        };
        tap.copy_from_device_async(out, self.gpu.stream())?;
        Ok(())
    }

    /// Layer `l`'s attention sub-layer: `x1 = x + attention(rms_norm(x))` (module doc).
    fn attend(&mut self, l: usize) -> Result<(), GpuError> {
        let parts = AttendParts {
            gpu: self.gpu,
            w: self.w,
            k: self.k,
            rope: self.rope,
            msa: self.msa,
            dims: self.dims,
            cx: &self.cx,
        };
        attend_layer(&parts, self.stores, self.s, l)
    }
}

/// What a layer's attention sub-layer reads and does not change: the step's program and the
/// forced row ([`crate::forced`]) each lend theirs.
#[derive(Clone, Copy)]
pub(crate) struct AttendParts<'a, 'c> {
    pub(crate) gpu: &'a Gpu,
    pub(crate) w: &'a Weights,
    pub(crate) k: &'a Kernels,
    pub(crate) rope: &'a RopeRows,
    pub(crate) msa: &'a MsaShape,
    pub(crate) dims: &'a Dims,
    pub(crate) cx: &'a GqaCx<'c>,
}

/// Layer `l`'s attention sub-layer: `x1 = x + attention(rms_norm(x))` (module doc), over the pieces
/// the step's program and the forced row ([`crate::forced`]) share.
pub(crate) fn attend_layer(
    parts: &AttendParts<'_, '_>,
    stores: &mut [Store],
    s: &mut Scratch,
    l: usize,
) -> Result<(), GpuError> {
    let AttendParts {
        gpu,
        w,
        k,
        rope,
        msa: msa_shape,
        dims,
        cx,
    } = *parts;
    let d = *dims;
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let Some(Store {
        gqa,
        q_norm,
        k_norm,
        index,
    }) = stores.get_mut(l)
    else {
        return Err(shape(format!("layer {l}: no store")));
    };
    let Scratch {
        x,
        x1,
        xn,
        delta,
        pos,
        n_keys,
        q_raw,
        q,
        k: key,
        v,
        attn,
        part_v,
        part_ms,
        iq_raw,
        iq,
        ik,
        zeros,
        sink,
        msa,
        acts,
        forced_x: _,
    } = &mut *s;
    let table = &rope.table;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        w.f32_buf(WHAT, &gqa.norm)?,
        d.rms_eps,
        d.embd,
        1,
        xn,
    )?;
    gqa.q.project(cx, acts, l, xn, q_raw)?;
    gqa.k.project(cx, acts, l, xn, key)?;
    gqa.v.project(cx, acts, l, xn, v)?;
    if let Some(ix) = index.as_mut() {
        ix.q.project(cx, acts, l, xn, iq_raw)?;
        ix.k.project(cx, acts, l, xn, ik)?;
        // SAFETY: the step holds one token, so the head kernel's distinct-position contract holds.
        unsafe {
            enqueue_index(
                &k.rope,
                stream,
                IndexArgs {
                    proj_q: &*iq_raw,
                    iq,
                    proj_k: ik,
                    zeros: &*zeros,
                    gain_q: w.f32_buf(WHAT, &ix.q_norm)?,
                    gain_k: w.f32_buf(WHAT, &ix.k_norm)?,
                    table,
                    pos: &*pos,
                    eps: d.rms_eps,
                    ctx: d.ctx,
                    m: 1,
                    fault,
                    keys: &mut ix.keys,
                    sink,
                },
            )?;
        }
        k.msa.enqueue_score(
            stream,
            msa_shape,
            ScoreArgs {
                iq: &*iq,
                ik: &ix.keys,
                n_keys: &*n_keys,
                ctx: d.ctx,
                m: 1,
                blocks: blocks_for(d.ctx),
                fault,
            },
            msa,
        )?;
        k.msa.enqueue_pick(
            stream,
            msa_shape,
            PickArgs {
                n_keys: &*n_keys,
                ctx: d.ctx,
                m: 1,
                fault,
            },
            msa,
        )?;
    }
    // SAFETY: the step holds one token, so the head kernel's distinct-position contract holds.
    unsafe {
        k.rope.enqueue_head_norm_neox_append_128_partial(
            stream,
            PartialNeoxArgs {
                qg: &*q_raw,
                q,
                k: key,
                v: &*v,
                gq: w.f32_buf(WHAT, q_norm)?,
                gk: w.f32_buf(WHAT, k_norm)?,
                table,
                pos: &*pos,
                eps: d.rms_eps,
                n_head: d.heads,
                n_kv: d.kv_heads,
                ctx: d.ctx,
                m: 1,
                fault,
                cache_k: gqa.kc.buf_mut(),
                cache_v: gqa.vc.buf_mut(),
            },
        )?;
    }
    if index.is_some() {
        enqueue_attention(
            &k.flash,
            stream,
            msa_shape,
            msa,
            AttnArgs {
                q: &*q,
                kc: gqa.kc.buf(),
                vc: gqa.vc.buf(),
                ctx: d.ctx,
                m: 1,
                part_v,
                part_ms,
                fault,
                y: &mut *attn,
            },
        )?;
    } else {
        k.flash.enqueue_pass::<GROUP_16>(
            stream,
            GqaArgs {
                q: &*q,
                kc: gqa.kc.buf(),
                vc: gqa.vc.buf(),
                n_keys: &*n_keys,
                scale: attn_scale(),
                n_kv: d.kv_heads,
                ctx: d.ctx,
                m: 1,
                part_v,
                part_ms,
                fault,
                y: &mut *attn,
            },
            FLASH_MMA,
        )?;
    }
    gqa.o.project(cx, acts, l, attn, delta)?;
    gpu.elem().enqueue_add(stream, x, delta, d.embd, x1)
}

impl<'a> LayerProgram for StepProgram<'a> {
    type Port = StepLeg<'a, HostRun>;

    /// The one row of the walk, whose embedding `refresh` wrote into `x`.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        no_other_row(unit)
    }

    /// The attention sub-layer, then a dense block whole or the routed block up to its go.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        self.attend(l)?;
        match self.cfg[l].ffn {
            FfnKind::Moe => {
                let hybrid = port.hybrid();
                ffn::front(self.gpu, self.w, self.ffn, &self.s.x1, (hybrid, l))
            }
            FfnKind::Dense => {
                let dest = if self.last(l) {
                    self.head.input_mut()
                } else {
                    &mut self.s.x
                };
                let (x1, xn, delta) = (&self.s.x1, &mut self.s.xn, &mut self.s.delta);
                ffn::dense(self.gpu, self.w, self.ffn, l, (x1, xn, delta), dest)?;
                self.tap(l)
            }
        }
    }

    /// A routed layer's shared expert and card experts under its host leg.
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].host_leg() {
            return Ok(());
        }
        let boundary = port.hybrid().boundary();
        ffn::shadow(self.gpu, self.w, self.ffn, &self.s.x1, (boundary, l))
    }

    /// A routed layer's wait and add, its tap, and the host tier told the layer is enqueued (an eager chain is
    /// served there).
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].host_leg() {
            return Ok(());
        }
        let dest = if self.last(l) {
            self.head.input_mut()
        } else {
            &mut self.s.x
        };
        ffn::back(self.gpu, self.ffn, port.hybrid(), l, dest)?;
        self.tap(l)?;
        port.hybrid().row_enqueued(l, at.unit)
    }

    /// The head, over the last layer's output.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        no_other_row(unit)?;
        self.head.enqueue(self.gpu, self.w)
    }
}

/// A shape the program has no launch for, by name.
fn shape(detail: String) -> GpuError {
    GpuError::Shape { what: WHAT, detail }
}

/// The walk has one row; another unit is refused by name.
fn no_other_row(unit: usize) -> Result<(), GpuError> {
    if unit == 0 {
        return Ok(());
    }
    Err(shape(format!(
        "row {unit} of a walk of one row ({BODY}'s step)"
    )))
}
