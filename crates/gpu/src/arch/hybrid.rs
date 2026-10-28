//! The Nemotron-H hybrid trunk's card half, which the `nemotron_h` and
//! `nemotron_h_moe` bodies run their Mamba-2 and attention layers through
//! (the reader's twin is `model::arch::hybrid`; the dialect's authority is
//! llama.cpp's `src/models/nemotron-h.cpp`). A family's own feed-forward
//! layer (the dense MLP, the latent MoE), its token table and its layer list
//! stay in the family.
//!
//! What is here: a layer's projection sites ([`Site`], launched by the type
//! the file gives it through [`crate::site`]) and the quantized activation
//! forms they read ([`Acts`]); the two layer kinds' resident pieces
//! ([`MambaLayer`], [`GqaLayer`]), resolved at load by [`Names`]; the trunk's
//! widths ([`Shape`]), kernels ([`Kernels`]) and step scratch ([`Scratch`]);
//! the launches of one layer of each kind ([`mamba_layer`], [`gqa_layer`])
//! and the pre-norm and residual add every layer brackets its block with
//! ([`norm`], [`residual`]); and the step's word record ([`Words`]: the
//! position, the token and one plane row a key head), which a body refreshes
//! before every enqueue or replay, so one captured graph serves every
//! position.
//!
//! The attention layer is the NoPE GQA layer any family with one runs
//! ([`GqaCx`], [`GqaLayer`], [`GqaBufs`]): Nemotron-H's has no out gate
//! ([`GqaGate::None`]), Solar-Open2's has a projection of its own
//! ([`GqaGate::Proj`]). A prompt batch runs it over `m` rows in one launch a
//! site ([`gqa_rows`]); the K/V append (`append_plane`) and the wide
//! projection (`project_wide`) are the pieces the Nemotron-H prompt passes
//! share with it.

use crate::fault::FaultSink;
use crate::flash::PlaneAppend;
use crate::flash_gqa::{FlashGqaKernels, GROUP, GqaArgs, HEAD, partials_ms_len, partials_v_len};
use crate::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs};
use crate::gemm::{Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
use crate::linear::out_gate::{OutGateArgs, OutGateKernels};
use crate::mamba2::{Mamba2Kernels, Mamba2Scratch, StepArgs};
use crate::q5::Q8Blocks32;
use crate::recur::RecStore;
use crate::site::{self, Form, KGemvKernels, SiteTy, SiteWeight, WideActs, WideKernels};
use crate::tensor::{Pitched, dtod_2d};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act, Window};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::hybrid::{
    ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_A, SSM_CONV, SSM_CONV_BIAS, SSM_D, SSM_DT_BIAS,
    SSM_IN, SSM_NORM, SSM_OUT, Ssm,
};
use model::arch::names::blk as layer_name;
use model::mamba2::{Checked, Geom};
use runtime::stores;

use crate::linear::CONV_TAPS;

// The scan's ring is indexed by the linear module's rows; the plan counts the ring by
// `stores::conv_ring_rows`.
const _: () = assert!(crate::linear::RING_ROWS == stores::conv_ring_rows(CONV_TAPS));

/// The input of one decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeInput {
    pub token: u32,
    pub pos: u32,
}

/// The step's word record: the position, the token, then one row word a key
/// head ([`ROW_WORDS`]).
pub(crate) const WORD_POS: usize = 0;
pub(crate) const WORD_TOKEN: usize = 1;
pub(crate) const ROW_WORDS: usize = 2;

/// One projection of a layer: the file tensor's name, the launch its type
/// selects and the width of the rows it reads.
pub struct Site {
    pub(crate) name: String,
    pub(crate) ty: SiteTy,
    pub(crate) k: usize,
}

pub(crate) struct MambaLayer {
    pub(crate) norm: String,
    pub(crate) ssm_in: Site,
    pub(crate) ssm_out: Site,
    pub(crate) a: String,
    pub(crate) d: String,
    pub(crate) dt_bias: String,
    pub(crate) conv_w: String,
    pub(crate) conv_b: String,
    pub(crate) ssm_norm: String,
    pub(crate) store: RecStore,
}

impl MambaLayer {
    pub(crate) fn sites(&self) -> [&Site; 2] {
        [&self.ssm_in, &self.ssm_out]
    }
}

/// The widths the scratch is cut for and the chain runs at.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) hidden: usize,
    pub(crate) eps: f32,
    pub(crate) mamba: Checked,
    pub(crate) n_kv: usize,
    pub(crate) ctx: usize,
}

/// The kernels the trunk's layers launch beyond the `Gpu`'s own.
pub(crate) struct Kernels {
    pub(crate) mamba: Mamba2Kernels,
    pub(crate) flash: FlashGqaKernels,
    pub(crate) g32: Gemm32Kernels,
    pub(crate) kgemv: KGemvKernels,
    /// The grouped GEMM a prompt pass's K-quant sites run on.
    pub(crate) gemm: GemmKernels,
}

impl Kernels {
    /// Load the five modules on `gpu`. Load-time only.
    pub(crate) fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            mamba: Mamba2Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            kgemv: KGemvKernels::load(gpu)?,
            gemm: GemmKernels::load(ctx)?,
        })
    }
}

/// The quantized activation forms the sites read: one of each form a site's
/// type reads, at each width a site reads it at, and the q8_1 forms of more
/// than one column a family's own launches read ([`Acts::q8_1_at`]).
pub struct Acts {
    blocks32: Vec<Q8Blocks32>,
    q8_1: Vec<Q8Act>,
}

impl Acts {
    /// The forms `sites` read. Load-time only.
    pub fn new<'s>(
        stream: &CudaStream,
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<Acts, GpuError> {
        let mut acts = Acts {
            blocks32: Vec::new(),
            q8_1: Vec::new(),
        };
        for s in sites {
            match s.reads() {
                Form::Q8Blocks32 if !acts.blocks32.iter().any(|a| a.k() == s.k) => {
                    acts.blocks32.push(Q8Blocks32::new(stream, s.k, 1)?);
                }
                Form::Q8x128 => {
                    acts.add_q8_1(stream, s.k, 1)?;
                }
                _ => {}
            }
        }
        Ok(acts)
    }

    /// A q8_1 form of `m` columns of `k` values, unless one is there.
    /// Load-time only.
    pub(crate) fn add_q8_1(
        &mut self,
        stream: &CudaStream,
        k: usize,
        m: usize,
    ) -> Result<(), GpuError> {
        if !self.q8_1.iter().any(|a| a.k() == k && a.m() == m) {
            // More than one column is a slot form: column `s` is slot `s`'s.
            self.q8_1.push(if m == 1 {
                Q8Act::with_k(stream, m, k)?
            } else {
                Q8Act::with_slots(stream, m, k)?
            });
        }
        Ok(())
    }

    /// The q8_1 form of `m` columns of `k` values, which the load made.
    pub fn q8_1_at(
        &mut self,
        what: &'static str,
        k: usize,
        m: usize,
    ) -> Result<&mut Q8Act, GpuError> {
        self.q8_1
            .iter_mut()
            .find(|a| a.k() == k && a.m() == m)
            .ok_or_else(|| GpuError::state(what, "a q8_1 form at the launch's width and columns"))
    }

    /// Device bytes of the forms.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.blocks32
            .iter()
            .map(|a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
            .sum::<usize>()
            + self.q8_1.iter().map(Q8Act::device_bytes).sum::<usize>()
    }
}

/// The rows a chain runs over, `rows` of each width: the step's one row
/// ([`Scratch`]) and a prompt pass's rows (the Nano's `prefill`), with the
/// flash partials and the embedding's row outputs. `up` and `act` are the
/// dense MLP's (`ff` 0 where a family has none).
pub(crate) struct RowBuffers {
    pub(crate) xn: DeviceBuffer<f32>,
    pub(crate) delta: DeviceBuffer<f32>,
    pub(crate) zxbcdt: DeviceBuffer<f32>,
    pub(crate) y: DeviceBuffer<f32>,
    pub(crate) q: DeviceBuffer<f32>,
    pub(crate) k: DeviceBuffer<f32>,
    pub(crate) v: DeviceBuffer<f32>,
    pub(crate) attn: DeviceBuffer<f32>,
    pub(crate) up: DeviceBuffer<f32>,
    pub(crate) act: DeviceBuffer<f32>,
    pub(crate) part_v: DeviceBuffer<f32>,
    pub(crate) part_ms: DeviceBuffer<f32>,
    /// The embedding's outputs: the position word the scan reads and the live
    /// key count the flash reads, one of each a row.
    pub(crate) pos: DeviceBuffer<u32>,
    pub(crate) n_keys: DeviceBuffer<u32>,
}

impl RowBuffers {
    /// `rows` rows of each width for `sh`, `n_head` query heads and the
    /// widest `ff` an MLP layer has, the flash partials at the rows a flash
    /// run takes. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        rows: usize,
        sh: &Shape,
        (n_head, ff): (usize, usize),
    ) -> Result<RowBuffers, GpuError> {
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, rows * n.max(1))?)
        };
        let kv_width = sh.n_kv * HEAD;
        let attn_rows = rows.min(crate::model::MAX_PASS_ROWS);
        Ok(RowBuffers {
            xn: z(sh.hidden)?,
            delta: z(sh.hidden)?,
            zxbcdt: z(sh.mamba.in_proj_width())?,
            y: z(sh.mamba.geom().d_inner)?,
            q: z(n_head * HEAD)?,
            k: z(kv_width)?,
            v: z(kv_width)?,
            attn: z(n_head * HEAD)?,
            up: z(ff)?,
            act: z(ff)?,
            part_v: DeviceBuffer::zeroed(stream, partials_v_len(attn_rows, n_head))?,
            part_ms: DeviceBuffer::zeroed(stream, partials_ms_len(attn_rows, n_head))?,
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        [
            &self.xn,
            &self.delta,
            &self.zxbcdt,
            &self.y,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.up,
            &self.act,
            &self.part_v,
            &self.part_ms,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }
}

/// The buffers one step runs over, m = 1.
pub(crate) struct Scratch {
    pub(crate) buf: RowBuffers,
    pub(crate) mamba: Mamba2Scratch,
    /// The record [`WORD_POS`], [`WORD_TOKEN`] and [`ROW_WORDS`] name.
    pub(crate) words: DeviceBuffer<u32>,
}

impl Scratch {
    pub(crate) fn bytes(&self) -> usize {
        self.buf.bytes() + self.words.num_bytes() + self.mamba.device_bytes()
    }

    /// The step's rows as a GQA layer's buffers: the record's row words from
    /// [`ROW_WORDS`], and no gate rows (the trunk's attention layers have no
    /// out gate).
    pub(crate) fn gqa(&mut self) -> GqaBufs<'_> {
        let RowBuffers {
            xn,
            delta,
            q,
            k,
            v,
            attn,
            part_v,
            part_ms,
            n_keys,
            ..
        } = &mut self.buf;
        GqaBufs {
            words: &self.words,
            row_word: ROW_WORDS,
            n_keys,
            xn,
            q,
            k,
            v,
            g: None,
            attn,
            part_v,
            part_ms,
            delta,
        }
    }

    /// The scratch for `sh`, `n_head` query heads and the widest `ff` an MLP
    /// layer has. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        sh: &Shape,
        n_head: usize,
        ff: usize,
    ) -> Result<Scratch, GpuError> {
        Ok(Scratch {
            buf: RowBuffers::new(stream, 1, sh, (n_head, ff))?,
            mamba: Mamba2Scratch::new(stream, sh.mamba)?,
            words: DeviceBuffer::zeroed(stream, ROW_WORDS + sh.n_kv)?,
        })
    }
}

/// The step record's host words, rewritten by every refresh.
pub(crate) struct Words {
    host: Vec<u32>,
}

impl Words {
    /// The record of `n_kv` key heads. Load-time only.
    pub(crate) fn new(n_kv: usize) -> Words {
        Words {
            host: vec![0; ROW_WORDS + n_kv],
        }
    }

    /// The record in one copy into `dev`: the position, the token and, for
    /// each key head, the row `h · ctx + pos` its plane's append lands at.
    /// `what` names the refusal of a row that does not fit a u32 word.
    pub(crate) fn refresh(
        &mut self,
        what: &'static str,
        stream: &CudaStream,
        sh: &Shape,
        input: &DecodeInput,
        dev: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let DecodeInput { token, pos } = *input;
        let ctx = sh.ctx;
        self.host[WORD_POS] = pos;
        self.host[WORD_TOKEN] = token;
        for h in 0..sh.n_kv {
            let row = h
                .checked_mul(ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        what,
                        format!("row {h} · {ctx} + {pos} does not fit a u32 row word"),
                    )
                })?;
            self.host[ROW_WORDS + h] = row;
        }
        Ok(dev.copy_from_host(stream, &self.host)?)
    }
}

/// What every launch of a step reads: the card, the resident weights, the
/// trunk's kernels and the widths; `what` names the step in a refusal.
pub(crate) struct Cx<'a> {
    pub(crate) what: &'static str,
    pub(crate) gpu: &'a Gpu,
    pub(crate) w: &'a Weights,
    pub(crate) k: &'a Kernels,
    pub(crate) sh: Shape,
}

impl<'a> Cx<'a> {
    /// The resident f32 vector `name`.
    pub(crate) fn f32(&self, name: &str) -> Result<&'a DeviceBuffer<f32>, GpuError> {
        self.w.f32_buf(self.what, name)
    }

    /// The site and flash kernels and the widths of this step, as a GQA
    /// layer's launches (and every [`Site`]'s) read them; the trunk's flash
    /// scores on the f32 dot.
    pub(crate) fn gqa(&self) -> GqaCx<'a> {
        let k: &'a Kernels = self.k;
        GqaCx {
            what: self.what,
            gpu: self.gpu,
            w: self.w,
            kgemv: &k.kgemv,
            g32: &k.g32,
            flash: &k.flash,
            mma: false,
            out_gate: None,
            hidden: self.sh.hidden,
            eps: self.sh.eps,
            n_kv: self.sh.n_kv,
            ctx: self.sh.ctx,
        }
    }
}

impl Site {
    /// The site of the file tensor `name`, launched by the type `ty`, over
    /// rows of `k` values.
    #[must_use]
    pub fn new(name: String, ty: SiteTy, k: usize) -> Site {
        Site { name, ty, k }
    }

    /// The site `name` of `rows` rows of `k` values in `file`: its type read from the header (one of `allowed`) and
    /// checked against the resident weight, refused by name as `what`'s.
    pub fn of_file(
        (file, w): (&Split, &Weights),
        what: &'static str,
        name: String,
        (rows, k): (usize, usize),
        allowed: &[SiteTy],
    ) -> Result<Site, GpuError> {
        let ty = site::file_site(file, what, &name, (rows, k), allowed)?;
        site::site(w, what, &name, (rows, k), ty)?;
        Ok(Site::new(name, ty, k))
    }

    /// The type that selects the launch.
    #[must_use]
    pub fn ty(&self) -> SiteTy {
        self.ty
    }

    /// The file tensor's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The width of the rows the site reads.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// The activation form the site's launch reads at its width ([`site::reads_at`]).
    #[must_use]
    pub fn reads(&self) -> Form {
        site::reads_at(self.ty, self.k)
    }

    /// `y = W · act` over the site's already-quantized q8_1 form, raising
    /// into layer `layer`'s fault sink — the gemv half of the q8_1 arm of
    /// [`Site::project`], which a caller that quantized by other means
    /// ([`mamba_layer`]'s fused norm, or one quantization serving several
    /// sites of one input) launches alone.
    pub fn kgemv(
        &self,
        cx: &GqaCx<'_>,
        act: &Q8Act,
        layer: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let key = (self.ty, cx.w, self.name.as_str());
        site::kgemv(cx.gpu, cx.kgemv, key, act, cx.gpu.layer_sink(layer)?, y)
    }

    /// `y = W · x` for this site over the first `k` values of `x`, raising
    /// into layer `layer`'s fault sink.
    pub fn project(
        &self,
        cx: &GqaCx<'_>,
        acts: &mut Acts,
        layer: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let gpu = cx.gpu;
        let key = (self.ty, cx.w, self.name.as_str());
        match self.reads() {
            // Q8_0's form: the f32 rows on this gemv arm.
            Form::F32 | Form::Q8x32 => site::gemv(gpu, cx.g32, key, x, 1, y),
            Form::Q8Blocks32 => {
                let act = acts
                    .blocks32
                    .iter_mut()
                    .find(|a| a.k() == self.k)
                    .ok_or_else(|| {
                        GpuError::state(cx.what, "a Q8Blocks32 form at the site's width")
                    })?;
                let fault = gpu.layer_sink(layer)?;
                gpu.q5().enqueue_quantize_q8(gpu.stream(), x, act, fault)?;
                match self.ty {
                    SiteTy::Nvfp4 => site::nvfp4_sel32_gemv(gpu, cx.kgemv, key, act, fault, y),
                    _ => site::blocks32_gemv(gpu, key, act, y),
                }
            }
            Form::Q8x128 => {
                let act = acts.q8_1_at(cx.what, self.k, 1)?;
                gpu.enqueue_quantize_q8_1_layer(x, act, layer)?;
                self.kgemv(cx, act, layer, y)
            }
        }
    }
}

/// `y = rms_norm(x, name)` over `m` rows of `hidden` at `eps`: the step's
/// one row and a prompt pass's rows.
pub(crate) fn norm_rows(
    (gpu, w): (&Gpu, &Weights),
    (eps, hidden, m): (f32, usize, usize),
    name: &str,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let gain = w.f32_buf("hybrid::norm_rows", name)?;
    gpu.elem()
        .enqueue_rms_norm(gpu.stream(), x, gain, eps, hidden, m, y)
}

/// `dst = src + delta` over `n` values, the layer's residual.
pub(crate) fn residual_rows(
    gpu: &Gpu,
    n: usize,
    src: &DeviceBuffer<f32>,
    delta: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gpu.elem().enqueue_add(gpu.stream(), src, delta, n, dst)
}

/// [`norm_rows`] of the step's one row.
pub(crate) fn norm(
    cx: &Cx<'_>,
    name: &str,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    norm_rows((cx.gpu, cx.w), (cx.sh.eps, cx.sh.hidden, 1), name, x, y)
}

/// [`residual_rows`] of the step's one row.
pub(crate) fn residual(
    cx: &Cx<'_>,
    src: &DeviceBuffer<f32>,
    delta: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    residual_rows(cx.gpu, cx.sh.hidden, src, delta, dst)
}

/// One Mamba-2 layer: the `ssm_in` output is `[z | xBC | dt]`, three views of
/// the one buffer at `0`, `d_inner` and `d_inner + conv_dim` values (llama.cpp
/// `build_mamba2_layer`'s views at `0`, `d_inner`, `2·d_inner + 2·n_group·d_state`).
pub(crate) fn mamba_layer(
    cx: &Cx<'_>,
    (l, m): (usize, &mut MambaLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let sh = cx.sh;
    let site_cx = cx.gqa();
    let (inner, conv_dim, heads) = (
        sh.mamba.geom().d_inner,
        sh.mamba.conv_dim(),
        sh.mamba.geom().n_heads,
    );
    if m.ssm_in.reads() == Form::Q8x128 {
        // The fused launch writes the normed row into `xn` and the q8_1 form
        // `ssm_in`'s gemv reads — the norm + quantize pair it folds, bit for
        // bit; `xn` has no other reader on this path.
        let act = acts.q8_1_at(cx.what, m.ssm_in.k, 1)?;
        cx.gpu.fused().enqueue_norm_quant(
            cx.gpu.stream(),
            src,
            cx.f32(&m.norm)?,
            sh.eps,
            act,
            &mut s.buf.xn,
            cx.gpu.layer_sink(l)?,
        )?;
        m.ssm_in.kgemv(&site_cx, act, l, &mut s.buf.zxbcdt)?;
    } else {
        norm(cx, &m.norm, src, &mut s.buf.xn)?;
        m.ssm_in
            .project(&site_cx, acts, l, &s.buf.xn, &mut s.buf.zxbcdt)?;
    }
    let f32s = size_of::<f32>();
    let z = Window::<f32>::of(&s.buf.zxbcdt, 0, inner)?;
    let xbc = Window::<f32>::of(&s.buf.zxbcdt, inner * f32s, conv_dim)?;
    let dt = Window::<f32>::of(&s.buf.zxbcdt, (inner + conv_dim) * f32s, heads)?;
    cx.k.mamba.enqueue_step(
        cx.gpu.stream(),
        StepArgs {
            shape: sh.mamba,
            xbc: &xbc,
            dt: &dt,
            dt_bias: cx.f32(&m.dt_bias)?,
            z: &z,
            a: cx.f32(&m.a)?,
            d: cx.f32(&m.d)?,
            conv_w: cx.f32(&m.conv_w)?,
            conv_b: cx.f32(&m.conv_b)?,
            norm: Some(cx.f32(&m.ssm_norm)?),
            eps: sh.eps,
            pos: &s.buf.pos,
            pos_at: 0,
            fault: cx.gpu.layer_sink(l)?,
            y: &mut s.buf.y,
            ring: &mut m.store.ring,
            state: &mut m.store.state,
            scratch: &mut s.mamba,
        },
    )?;
    m.ssm_out
        .project(&site_cx, acts, l, &s.buf.y, &mut s.buf.delta)?;
    residual(cx, src, &s.buf.delta, dst)
}

/// The scan's checked widths of the Mamba-2 widths `s` ([`crate::mamba2::check`]),
/// refused by name as `what`'s.
pub(crate) fn mamba_shape(what: &'static str, s: &Ssm) -> Result<Checked, GpuError> {
    crate::mamba2::check(
        what,
        Geom {
            d_inner: s.d_inner,
            n_heads: s.n_heads,
            n_groups: s.n_groups,
            d_state: s.d_state,
            d_conv: s.d_conv,
        },
    )
}

/// The key heads of the attention layers, `kv` each layer's (`l`, count),
/// which the body takes equal across layers: the planes, the row words and
/// the flash group are cut for one. Heads of [`HEAD`] values and `group`
/// query heads a key head (the group the body's flash pass takes) are what
/// it runs; any other is refused by name.
pub(crate) fn attention_heads(
    what: &'static str,
    kv: impl Iterator<Item = (usize, usize)>,
    (n_head, key_len, value_len): (usize, usize, usize),
    group: usize,
) -> Result<usize, GpuError> {
    let mut found = None;
    for (l, kv) in kv {
        if *found.get_or_insert(kv) != kv {
            return Err(GpuError::shape(
                what,
                format!("attention layer {l} has {kv} key heads, an earlier one {found:?}"),
            ));
        }
        if key_len != HEAD || value_len != HEAD || n_head != kv * group {
            return Err(GpuError::shape(
                what,
                format!(
                    "attention of {n_head} heads over {kv} key heads of {key_len}/{value_len} \
                     values: the flash runs {group} query heads a key head of {HEAD} values"
                ),
            ));
        }
    }
    Ok(found.unwrap_or(0))
}

/// The derived conv bias of a file whose Mamba-2 layers hold none.
const ZERO_CONV_BIAS: &str = "derived.nemotron_h.zero_conv_bias";

/// The file and its resident weights, what a load resolves names against;
/// `what` names the load in a refusal and `types` are the site types its
/// projections may be.
pub(crate) struct Names<'a> {
    pub(crate) what: &'static str,
    pub(crate) file: &'a Split,
    pub(crate) w: &'a Weights,
    pub(crate) types: &'static [SiteTy],
}

/// The zero conv bias of a file whose Mamba-2 layers `mamba_layers` hold none
/// of, made once and filed into `w` before any name is resolved
/// ([`Names::mamba`] reads it): one zero vector of `len` values every such
/// layer shares. A file that holds every layer's bias adds nothing.
pub(crate) fn derive_zero_conv_bias(
    w: &mut Weights,
    file: &Split,
    mamba_layers: impl IntoIterator<Item = usize>,
    len: usize,
    stream: &CudaStream,
) -> Result<(), GpuError> {
    let lacks = mamba_layers
        .into_iter()
        .any(|l| file.find(&layer_name(l, SSM_CONV_BIAS)).is_none());
    if lacks && w.get(ZERO_CONV_BIAS).is_none() {
        let t = DeviceTensor::zeroed(stream, 1, len)?;
        w.insert_derived(ZERO_CONV_BIAS.to_owned(), DevWeight::F32 { w: t, k: len })?;
    }
    Ok(())
}

impl Names<'_> {
    /// Site `name` of `rows` rows of `k` values: its type from the file's
    /// header (one of `allowed`), checked against the resident weight.
    pub(crate) fn site(
        &self,
        name: &str,
        (rows, k): (usize, usize),
        allowed: &[SiteTy],
    ) -> Result<Site, GpuError> {
        let ty = site::file_site(self.file, self.what, name, (rows, k), allowed)?;
        site::site(self.w, self.what, name, (rows, k), ty)?;
        Ok(Site {
            name: name.to_string(),
            ty,
            k,
        })
    }

    /// The projection `stem` of layer `l`, of one of the load's `types`.
    pub(crate) fn proj(
        &self,
        l: usize,
        stem: &str,
        shape: (usize, usize),
    ) -> Result<Site, GpuError> {
        self.site(&layer_name(l, stem), shape, self.types)
    }

    /// Layer `l`'s conv bias of `len` values: the file's, or where the file
    /// holds none (the bias is optional) the zero vector every such layer
    /// reads ([`derive_zero_conv_bias`], resident from load).
    fn conv_bias(&self, l: usize, len: usize) -> Result<String, GpuError> {
        if self.file.find(&layer_name(l, SSM_CONV_BIAS)).is_some() {
            return self.vector(l, SSM_CONV_BIAS, len);
        }
        site::vector(self.w, self.what, ZERO_CONV_BIAS.to_owned(), len)
    }

    /// The f32 vector `stem` of layer `l`, `len` values: its name.
    pub(crate) fn vector(&self, l: usize, stem: &str, len: usize) -> Result<String, GpuError> {
        site::vector(self.w, self.what, layer_name(l, stem), len)
    }

    /// Mamba-2 layer `l` of the widths `s` on a `hidden`-value row, its
    /// pre-norm `norm` resolved by the caller, and its scan store.
    pub(crate) fn mamba(
        &mut self,
        (l, norm): (usize, String),
        s: &Ssm,
        hidden: usize,
        stream: &CudaStream,
    ) -> Result<MambaLayer, GpuError> {
        let lens = stores::mamba2_lens(s.d_inner, s.d_state, CONV_TAPS, s.conv_dim);
        Ok(MambaLayer {
            ssm_in: self.proj(l, SSM_IN, (s.d_in_proj, hidden))?,
            ssm_out: self.proj(l, SSM_OUT, (hidden, s.d_inner))?,
            a: self.vector(l, SSM_A, s.n_heads)?,
            d: self.vector(l, SSM_D, s.n_heads)?,
            dt_bias: self.vector(l, SSM_DT_BIAS, s.n_heads)?,
            conv_w: self.vector(l, SSM_CONV, s.d_conv * s.conv_dim)?,
            conv_b: self.conv_bias(l, s.conv_dim)?,
            ssm_norm: self.vector(l, SSM_NORM, s.d_inner)?,
            store: RecStore::new(stream, lens, 1)?,
            norm,
        })
    }

    /// Attention layer `l` of `n_head` query heads over `kv` key heads of
    /// [`HEAD`] values on a `hidden`-value row, its pre-norm `norm` resolved
    /// by the caller, no out gate, and its K/V planes of `ctx` rows a key
    /// head.
    pub(crate) fn attn(
        &mut self,
        (l, norm): (usize, String),
        (n_head, kv): (usize, usize),
        (hidden, ctx): (usize, usize),
        stream: &CudaStream,
    ) -> Result<GqaLayer, GpuError> {
        let width = kv * HEAD;
        Ok(GqaLayer {
            q: self.proj(l, ATTN_Q, (n_head * HEAD, hidden))?,
            k: self.proj(l, ATTN_K, (width, hidden))?,
            v: self.proj(l, ATTN_V, (width, hidden))?,
            o: self.proj(l, ATTN_OUTPUT, (hidden, n_head * HEAD))?,
            gate: GqaGate::None,
            kc: DeviceTensor::zeroed(stream, kv * ctx, HEAD)?,
            vc: DeviceTensor::zeroed(stream, kv * ctx, HEAD)?,
            norm,
        })
    }
}

/// What a NoPE GQA layer's launches read, apart from a family's other
/// kernels: the card, the resident weights, the site and flash kernels and
/// the widths. Nemotron-H's `Cx` lends one (`Cx::gqa`); a family with no
/// Mamba-2 layer (Solar-Open2) makes its own.
pub struct GqaCx<'a> {
    pub what: &'static str,
    pub gpu: &'a Gpu,
    pub w: &'a Weights,
    pub kgemv: &'a KGemvKernels,
    pub g32: &'a Gemm32Kernels,
    pub flash: &'a FlashGqaKernels,
    /// The step flash's score pass: the tensor cores from the query rounded
    /// to f16 (the prefill flash's scores, bit for bit) or the f32 dot
    /// (group 5 has only that one).
    pub mma: bool,
    /// The out gate's kernels; a layer with a [`GqaGate::Proj`] given none is
    /// refused by name.
    pub out_gate: Option<&'a OutGateKernels>,
    pub hidden: usize,
    pub eps: f32,
    pub n_kv: usize,
    pub ctx: usize,
}

/// A GQA layer's out gate: the description's `models::OutGate`, resolved at
/// load. `Proj` is a separate projection of the normed input,
/// `attn · σ(W_g · xn)` before the output projection (Solar-Open2's
/// `attn_gate`, 4096 → heads·128); the HEAD-128 separate-gate entry
/// (`linear::out_gate`) launches it.
pub enum GqaGate {
    None,
    Proj(Site),
}

/// One NoPE GQA layer's resident pieces: the four sites, the out gate beside
/// them, and the K/V planes.
pub struct GqaLayer {
    pub norm: String,
    pub q: Site,
    pub k: Site,
    pub v: Site,
    pub o: Site,
    pub gate: GqaGate,
    /// The head-major planes: head `h`, position `p` at row `h · ctx + p`.
    pub kc: DeviceTensor<u16>,
    pub vc: DeviceTensor<u16>,
}

impl GqaLayer {
    /// Every site the layer launches, for the load's [`Acts::new`].
    #[must_use]
    pub fn sites(&self) -> Vec<&Site> {
        let mut sites = vec![&self.q, &self.k, &self.v, &self.o];
        if let GqaGate::Proj(g) = &self.gate {
            sites.push(g);
        }
        sites
    }

    /// The K/V planes' bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.kc.buf().num_bytes() + self.vc.buf().num_bytes()
    }
}

/// The rows a GQA layer's launches write over `m` rows (the step's one, a
/// prompt batch's `m`), each at least `m` rows of its width; `words` the
/// step's or the batch's word record, whose row words (one a key head, the
/// plane row `h · ctx + pos` the first row's append lands at) start at word
/// `row_word`.
pub struct GqaBufs<'a> {
    pub words: &'a DeviceBuffer<u32>,
    pub row_word: usize,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub xn: &'a mut DeviceBuffer<f32>,
    pub q: &'a mut DeviceBuffer<f32>,
    pub k: &'a mut DeviceBuffer<f32>,
    pub v: &'a mut DeviceBuffer<f32>,
    /// The gate's rows (heads · [`HEAD`] values a row): unread under
    /// [`GqaGate::None`], and a [`GqaGate::Proj`] layer given none is refused
    /// by name.
    pub g: Option<&'a mut DeviceBuffer<f32>>,
    pub attn: &'a mut DeviceBuffer<f32>,
    pub part_v: &'a mut DeviceBuffer<f32>,
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub delta: &'a mut DeviceBuffer<f32>,
}

/// What a prompt batch's GQA launches read beyond [`GqaCx`]: the grouped
/// GEMM and the prefill flash, the batch's one-expert table (filled for its
/// `m` rows by `GemmKernels::enqueue_route_dense`), and the quantized forms of
/// the rows a site reads — `xn` of the normed rows (`hidden` values a row),
/// `o` of the flash output (heads · [`HEAD`] values a row).
pub struct GqaWide<'a> {
    pub gemm: &'a GemmKernels,
    pub prefill: &'a FlashGqaPrefill,
    pub route: &'a GemmRoute,
    pub xn: &'a mut WideActs,
    pub o: &'a mut WideActs,
}

/// Key head `h`'s row of `buf`, the view an append reads.
fn head_rows<'b>(
    buf: &'b DeviceBuffer<f32>,
) -> impl FnMut(usize) -> Result<Window<'b, f32>, GpuError> {
    move |h| Window::<f32>::of(buf, h * HEAD * size_of::<f32>(), HEAD)
}

/// The gate's site and rows, or `None` under [`GqaGate::None`]; a gated
/// layer given no rows is refused by name.
fn gate_rows<'a>(
    what: &'static str,
    gate: &'a GqaGate,
    g: Option<&'a mut DeviceBuffer<f32>>,
) -> Result<Option<(&'a Site, &'a mut DeviceBuffer<f32>)>, GpuError> {
    match (gate, g) {
        (GqaGate::None, _) => Ok(None),
        (GqaGate::Proj(site), Some(g)) => Ok(Some((site, g))),
        (GqaGate::Proj(site), None) => Err(GpuError::shape(
            what,
            format!("the out gate {} has no rows to project into", site.name),
        )),
    }
}

/// `y = attn · σ(g)` over `rows` rows of `heads` heads of [`HEAD`] values,
/// raising into `fault`: the HEAD-128 separate-gate entry
/// (`linear::out_gate`).
fn out_gate(
    cx: &GqaCx<'_>,
    (attn, g): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    (rows, heads): (usize, usize),
    fault: FaultSink,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let kernels = cx
        .out_gate
        .ok_or_else(|| GpuError::state(cx.what, "the out gate's kernels (GqaCx::out_gate)"))?;
    kernels.enqueue_out_gate(
        cx.gpu.stream(),
        OutGateArgs {
            attn,
            gate: g,
            n_head: heads,
            m: rows,
            fault,
            y,
        },
    )
}

/// One NoPE GQA layer of `PACK` query heads a key head at one row: the
/// norm, q·k·v (and the gate's projection), the plane appends, the flash
/// pass, the out gate when the layer has one, `o`, the residual add of
/// `src` into `dst`. The gate multiplies into `q`'s rows, which nothing reads
/// after the flash pass; `o` reads them in place of `attn`.
pub fn gqa_layer<const PACK: usize>(
    cx: &GqaCx<'_>,
    (l, a): (usize, &mut GqaLayer),
    (b, acts): (GqaBufs<'_>, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let GqaBufs {
        words,
        row_word,
        n_keys,
        xn,
        q,
        k,
        v,
        g,
        attn,
        part_v,
        part_ms,
        delta,
    } = b;
    let fault = cx.gpu.layer_sink(l)?;
    let mut gate = gate_rows(cx.what, &a.gate, g)?;
    norm_rows((cx.gpu, cx.w), (cx.eps, cx.hidden, 1), &a.norm, src, xn)?;
    a.q.project(cx, acts, l, xn, q)?;
    a.k.project(cx, acts, l, xn, k)?;
    a.v.project(cx, acts, l, xn, v)?;
    if let Some((site, g)) = gate.as_mut() {
        site.project(cx, acts, l, xn, g)?;
    }
    let at = PlaneAppend {
        words,
        first_word: row_word,
        heads: cx.n_kv,
        m: 1,
        fault,
    };
    let flash = cx.gpu.flash();
    flash.enqueue_kv_append_planes(cx.gpu.stream(), &at, head_rows(k), &mut a.kc)?;
    flash.enqueue_kv_append_planes(cx.gpu.stream(), &at, head_rows(v), &mut a.vc)?;
    cx.flash.enqueue_pass::<PACK>(
        cx.gpu.stream(),
        GqaArgs {
            q: &*q,
            kc: a.kc.buf(),
            vc: a.vc.buf(),
            n_keys,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_kv: cx.n_kv,
            ctx: cx.ctx,
            m: 1,
            part_v,
            part_ms,
            fault,
            y: &mut *attn,
        },
        cx.mma,
    )?;
    let o_in: &DeviceBuffer<f32> = match &gate {
        Some((_, g)) => {
            out_gate(cx, (&*attn, &**g), (1, PACK * cx.n_kv), fault, &mut *q)?;
            &*q
        }
        None => &*attn,
    };
    a.o.project(cx, acts, l, o_in, delta)?;
    residual_rows(cx.gpu, cx.hidden, src, delta, dst)
}

/// [`gqa_layer`] over a prompt batch's `m` rows in position order: the
/// projections by `project_wide` (one quantization of the input and one GEMM
/// a site), the appends of every row (`append_plane`; the regrouped K and V
/// rows live in `attn`, which the flash writes last), then the prefill flash
/// over the causal mask, and the gate and `o` as in [`gqa_layer`]. `PACK` is
/// the prefill flash's group ([`GROUP`]).
pub fn gqa_rows<const PACK: usize>(
    cx: &GqaCx<'_>,
    (l, a): (usize, &mut GqaLayer),
    m: usize,
    (b, wide): (GqaBufs<'_>, GqaWide<'_>),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    const { assert!(PACK == GROUP) };
    let GqaBufs {
        words,
        row_word,
        n_keys,
        xn,
        q,
        k,
        v,
        g,
        attn,
        delta,
        ..
    } = b;
    let GqaWide {
        gemm,
        prefill,
        route,
        xn: xn_acts,
        o: o_acts,
    } = wide;
    let (gpu, f) = (cx.gpu, size_of::<f32>());
    let fault = gpu.layer_sink(l)?;
    let (n_head, kv_width) = (PACK * cx.n_kv, cx.n_kv * HEAD);
    let q_width = n_head * HEAD;
    if attn.len() < 2 * m * kv_width {
        return Err(GpuError::shape(
            cx.what,
            format!(
                "attn holds {} values; the K and V rows of {m} rows regroup into it, \
                 {} values",
                attn.len(),
                2 * m * kv_width
            ),
        ));
    }
    let mut gate = gate_rows(cx.what, &a.gate, g)?;
    let wc = WideCx {
        what: cx.what,
        gpu,
        w: cx.w,
        wk: WideKernels {
            gemm,
            g32: Some(cx.g32),
        },
        m,
    };
    norm_rows((gpu, cx.w), (cx.eps, cx.hidden, m), &a.norm, src, xn)?;
    let (qo, ko, vo) = (&mut *q, &mut *k, &mut *v);
    match gate.as_mut() {
        None => project_wide(
            &wc,
            (xn_acts, route),
            l,
            xn,
            &mut [
                (&a.q, q_width, qo),
                (&a.k, kv_width, ko),
                (&a.v, kv_width, vo),
            ],
        )?,
        Some((site, g)) => project_wide(
            &wc,
            (xn_acts, route),
            l,
            xn,
            &mut [
                (&a.q, q_width, qo),
                (&a.k, kv_width, ko),
                (&a.v, kv_width, vo),
                (*site, q_width, &mut **g),
            ],
        )?,
    }
    let at = PlaneAppend {
        words,
        first_word: row_word,
        heads: cx.n_kv,
        m,
        fault,
    };
    append_plane(gpu, &at, &*k, (&mut *attn, 0), &mut a.kc)?;
    append_plane(gpu, &at, &*v, (&mut *attn, m * kv_width * f), &mut a.vc)?;
    prefill.enqueue(
        gpu.stream(),
        GqaPrefillArgs {
            q: &*q,
            kc: a.kc.buf(),
            vc: a.vc.buf(),
            n_keys,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_head,
            n_kv: cx.n_kv,
            ctx: cx.ctx,
            t: m,
            fault,
            y: &mut *attn,
        },
    )?;
    let o_in: &DeviceBuffer<f32> = match &gate {
        Some((_, g)) => {
            out_gate(cx, (&*attn, &**g), (m, n_head), fault, &mut *q)?;
            &*q
        }
        None => &*attn,
    };
    project_wide(
        &wc,
        (o_acts, route),
        l,
        o_in,
        &mut [(&a.o, cx.hidden, delta)],
    )?;
    residual_rows(gpu, m * cx.hidden, src, delta, dst)
}

/// What a batch's wide projections read: the card, the resident weights, the
/// wide kernels and the batch's `m` rows; `what` names them in a refusal.
pub struct WideCx<'a> {
    pub what: &'static str,
    pub gpu: &'a Gpu,
    pub w: &'a Weights,
    pub wk: WideKernels<'a>,
    pub m: usize,
}

/// The wide forms [`project_wide`] quantizes into for the `sites` of the
/// model that read `k` values: `cols` columns of `k` values, one of each form
/// the types of those sites read ([`WideActs::wants_of`], less the f32 rows,
/// which are the caller's `x`); a site of another width is not counted.
/// Load-time only.
pub fn wide_acts_at<'s>(
    stream: &CudaStream,
    sites: impl IntoIterator<Item = &'s Site>,
    (cols, k): (usize, usize),
) -> Result<WideActs, GpuError> {
    let tys: Vec<SiteTy> = sites
        .into_iter()
        .filter(|s| s.k == k)
        .map(|s| s.ty)
        .collect();
    WideActs::new(stream, WideActs::wants_of(&tys).without_f32(), cols, k)
}

/// `y = W · x` for each of `sites` over the batch's `m` rows of `x`, `rows`
/// output values a row each: `x` quantized once into the forms the sites
/// read (`acts`, cut for their width), each site one GEMM on the batch's
/// one-expert table.
pub fn project_wide(
    cx: &WideCx<'_>,
    (acts, route): (&mut WideActs, &GemmRoute),
    layer: usize,
    x: &DeviceBuffer<f32>,
    sites: &mut [(&Site, usize, &mut DeviceBuffer<f32>)],
) -> Result<(), GpuError> {
    let Some(k) = sites.first().map(|s| s.0.k) else {
        return Ok(());
    };
    if let Some(s) = sites.iter().find(|s| s.0.k != k) {
        return Err(GpuError::shape(
            cx.what,
            format!(
                "sites of one input read one width: {} reads {}, not {k}",
                s.0.name, s.0.k
            ),
        ));
    }
    let tys: Vec<SiteTy> = sites.iter().map(|s| s.0.ty).collect();
    acts.quantize(cx.gpu, &cx.wk, cx.gpu.layer_sink(layer)?, x, cx.m, &tys)?;
    let input = acts.input(x);
    for (s, rows, y) in sites.iter_mut() {
        site::gemm(
            cx.gpu,
            &cx.wk,
            (s.ty, cx.w, SiteWeight::Resident(&s.name)),
            *rows,
            &input,
            (route, GemmInput::PerSlot),
            cx.m,
            y,
        )?;
    }
    Ok(())
}

/// Append a batch's `at.m` token-major rows of `rows` (`at.heads · HEAD`
/// values a row) to `plane`: each key head's rows are regrouped into
/// `scratch` from byte `off` (`at.m · HEAD` values a head, heads in turn),
/// which the append reads, then land at the plane rows `at`'s words name.
pub(crate) fn append_plane(
    gpu: &Gpu,
    at: &PlaneAppend<'_>,
    rows: &DeviceBuffer<f32>,
    (scratch, off): (&mut DeviceBuffer<f32>, usize),
    plane: &mut DeviceTensor<u16>,
) -> Result<(), GpuError> {
    let (m, f) = (at.m, size_of::<f32>());
    let shape = Pitched {
        width: HEAD * f,
        rows: m,
        dev_pitch: at.heads * HEAD * f,
        host_pitch: HEAD * f,
    };
    for h in 0..at.heads {
        dtod_2d(
            gpu.stream(),
            (rows, h * HEAD * f),
            (&mut *scratch, off + h * m * HEAD * f),
            shape,
        )?;
    }
    let scratch: &DeviceBuffer<f32> = scratch;
    gpu.flash().enqueue_kv_append_planes(
        gpu.stream(),
        at,
        |h| Window::<f32>::of(scratch, off + h * m * HEAD * f, m * HEAD),
        plane,
    )
}
