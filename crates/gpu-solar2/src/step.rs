//! The one-token step: `runtime::sched::walk` at `(1, 1, Step)` over the
//! layers' programs. The front is the layer's mixer by its kind (`Gqa`:
//! `hybrid::gqa_layer::<8>`; `DeltaRule`: `mixer_kda::enqueue` over
//! [`crate::kda::Sites`]) and the routed block to its go; the shadow the
//! shared expert and the card experts; the back the host sum's add.
//!
//! Each layer's parts, by its programs (`runtime::layer::Layer`), never its
//! number:
//! - the front: the mixer sub-layer whole (its output is the residual after
//!   the mixer `x1`), then the routed block up to its go ([`ffn::front`]);
//! - the shadow: the shared expert and the card experts, when the slot map
//!   puts any on the card ([`ffn::shadow`]);
//! - the back: the wait and the add of the host's sum to the shadow's, then
//!   the host tier told the layer is enqueued ([`ffn::back`]).
//!
//! A layer's output goes to its tap, where a gate armed one, once its last
//! add is enqueued. The begin starts from `x`, which the embedding wrote
//! together with the position words; the last layer's add writes the head's
//! input, and the end is the head.

use bloomery_gpu::arch::hybrid::{GqaBufs, GqaCx, gqa_layer};
use bloomery_gpu::flash_gqa::{FlashGqaKernels, GROUP};
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::linear::LinearKernels;
use bloomery_gpu::linear::mixer_kda::{self, KdaBufs, KdaState, KdaVecs};
use bloomery_gpu::site::KGemvKernels;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use runtime::layer::{FfnKind, MixerKind};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use crate::body::{Body, Dims, LayerCfg, Scratch, StepInput, Store, WORD_ROWS};
use crate::ffn::{self, Routed};
use crate::kda::Sites;

/// The step flash scores on the tensor cores from the query rounded to f16,
/// which is the prefill flash's score pass bit for bit: a position's bits do not
/// depend on which of the two ran it.
const FLASH_MMA: bool = true;

/// What the walk's errors name.
const WHAT: &str = "solar2 enqueue_step";

/// The step's point, `(1, 1, Step)`.
const STEP: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// The kernels the mixers launch besides `Gpu`'s own, loaded once, and the
/// committed lane word of the KDA stores.
pub struct Kernels {
    pub linear: LinearKernels,
    pub kgemv: KGemvKernels,
    pub g32: Gemm32Kernels,
    pub flash: FlashGqaKernels,
    /// The one lane every KDA store holds: the file has no NextN head, so
    /// the committed lane is always 0.
    pub lane: DeviceBuffer<u32>,
}

impl Kernels {
    /// Load the kernels into `gpu`'s context and zero the lane word.
    /// Load-time only.
    pub fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            linear: LinearKernels::load(ctx)?,
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            lane: DeviceBuffer::zeroed(gpu.stream(), 1)?,
        })
    }

    /// Device bytes held beside the modules.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.kgemv.device_bytes() + self.lane.num_bytes()
    }
}

/// Enqueue one step of `input` into `head` (eager or under capture). The
/// embedding's row is in `x` and the position words are written
/// (`refresh`); the position and the token are checked against the file's
/// bounds here.
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
    if dims.heads != GROUP * dims.kv_heads {
        return Err(shape(format!(
            "{} query heads over {} key heads: the GQA layer runs {GROUP} a key head",
            dims.heads, dims.kv_heads
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
        out_gate: Some(&k.linear.out_gate),
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
        lin: &k.linear,
        lane: &k.lane,
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

/// The Solar program over one walk: the body's parts and the head.
struct StepProgram<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    cx: GqaCx<'a>,
    lin: &'a LinearKernels,
    lane: &'a DeviceBuffer<u32>,
    dims: &'a Dims,
    cfg: &'a [LayerCfg],
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

    /// Layer `l`'s mixer sub-layer: `x1 = x + mixer(rms_norm(x))`.
    fn mixer(&mut self, l: usize) -> Result<(), GpuError> {
        match self.cfg[l].kind.mixer {
            MixerKind::Gqa => self.gqa(l),
            MixerKind::DeltaRule => self.kda(l),
            other => Err(shape(format!(
                "layer {l}: a {other:?} mixer; this program runs GQA and KDA"
            ))),
        }
    }

    /// A NoPE GQA layer with its out gate.
    fn gqa(&mut self, l: usize) -> Result<(), GpuError> {
        let Some(Store::Gqa(layer)) = self.stores.get_mut(l) else {
            return Err(wrong_store(l, "GQA"));
        };
        let Scratch {
            x,
            x1,
            xn,
            words,
            n_keys,
            gqa,
            delta,
            acts,
            ..
        } = &mut *self.s;
        let [q, k, v, g, attn, part_v, part_ms] = gqa;
        let bufs = GqaBufs {
            words: &*words,
            row_word: WORD_ROWS,
            n_keys: &*n_keys,
            xn,
            q,
            k,
            v,
            g: Some(g),
            attn,
            part_v,
            part_ms,
            delta,
        };
        gqa_layer::<GROUP>(&self.cx, (l, layer), (bufs, acts), (&*x, x1))
    }

    /// A KDA layer: the mixer's output into `delta`, the residual add into `x1`.
    fn kda(&mut self, l: usize) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let Some(Store::Kda {
            sites,
            state,
            stamp,
            ring,
        }) = self.stores.get_mut(l)
        else {
            return Err(wrong_store(l, "KDA"));
        };
        let Scratch {
            x,
            x1,
            xn,
            pos,
            kda,
            delta,
            acts,
            ..
        } = &mut *self.s;
        let [qkv, fa, ga, beta_raw, f, z, conv, beta, decay, o, gated] = kda;
        let [norm, conv_w, dt_bias, a, gate_norm] = sites.vecs.each_ref();
        let vecs = KdaVecs {
            norm: w.f32_buf(WHAT, norm)?,
            conv: w.f32_buf(WHAT, conv_w)?,
            dt_bias: w.f32_buf(WHAT, dt_bias)?,
            a: w.f32_buf(WHAT, a)?,
            gate_norm: w.f32_buf(WHAT, gate_norm)?,
        };
        let mut proj = Sites {
            cx: &self.cx,
            sites: &*sites,
            shape: self.dims.kda.shape,
            acts,
        };
        let st = KdaState {
            state,
            stamp,
            ring,
            lane: self.lane,
            lanes: 1,
            each: false,
            row: 0,
        };
        let bufs = KdaBufs {
            x: &*x,
            pos: &*pos,
            xn,
            qkv,
            fa,
            ga,
            beta_raw,
            f,
            z,
            conv,
            beta,
            decay,
            o,
            gated,
            out: &mut *delta,
        };
        let m = 1;
        mixer_kda::enqueue(
            gpu,
            gpu.stream(),
            self.lin,
            &mut proj,
            (l, m, &self.dims.kda),
            &vecs,
            st,
            bufs,
        )?;
        gpu.elem()
            .enqueue_add(gpu.stream(), &*x, &*delta, self.dims.embd, x1)
    }
}

impl<'a> LayerProgram for StepProgram<'a> {
    type Port = StepLeg<'a, HostRun>;

    /// The one row of the walk, whose embedding `refresh` wrote into `x`.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        no_other_row(unit)
    }

    /// The mixer sub-layer, then the routed block up to its go.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        self.mixer(l)?;
        match self.cfg[l].kind.ffn {
            FfnKind::Moe => {
                let hybrid = port.hybrid();
                ffn::front(self.gpu, self.w, self.ffn, &self.s.x1, (hybrid, l))
            }
            FfnKind::Dense => Err(shape(format!(
                "layer {l}: a dense block; this program runs routed blocks only"
            ))),
        }
    }

    /// A routed layer's shared expert and card experts under its host leg.
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].kind.host_leg() {
            return Ok(());
        }
        let boundary = port.hybrid().boundary();
        ffn::shadow(self.gpu, self.w, self.ffn, &self.s.x1, (boundary, l))
    }

    /// A routed layer's wait and add, its tap, and the host tier told the
    /// layer is enqueued (an eager chain is served there).
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        no_other_row(at.unit)?;
        let l = at.layer;
        if !self.cfg[l].kind.host_leg() {
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

/// Layer `l`'s store is not the kind its description says.
fn wrong_store(l: usize, kind: &str) -> GpuError {
    shape(format!("layer {l}: no {kind} store for a {kind} mixer"))
}

/// The walk has one row; another unit is refused by name.
fn no_other_row(unit: usize) -> Result<(), GpuError> {
    if unit == 0 {
        return Ok(());
    }
    Err(shape(format!("row {unit} of a walk of one row")))
}
