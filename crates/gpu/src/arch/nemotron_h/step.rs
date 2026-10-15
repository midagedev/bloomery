//! The launches of one decode step: the layer kinds, the sites' activation
//! forms and the scratch the chain runs over, and the enqueue of the whole
//! chain (async throughout: nothing here allocates, frees or synchronizes, so
//! it is both the eager body and what a capture records).

use super::body::Body;
use crate::elem::EmbedRowsArgs;
use crate::flash::PlaneAppend;
use crate::flash_gqa::{FlashGqaKernels, GROUP_5, GqaArgs, HEAD, partials_ms_len, partials_v_len};
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::head::Head;
use crate::mamba2::{Mamba2Kernels, Mamba2Scratch, StepArgs};
use crate::q5::Q8Blocks32;
use crate::recur::RecStore;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act, Window};
use cuda_core::DeviceBuffer;
use model::mamba2::Checked;

const WHAT: &str = "nemotron_h::Body::enqueue_chain";

/// The file types a site launches: Q5_0 through its packed 32-value gemv,
/// Q8_0 through the f32 one, Q4_K and Q6_K through theirs on the q8_1 rows.
pub(super) const SITE_TYPES: &[SiteTy] = &[SiteTy::Q5_0, SiteTy::Q8_0, SiteTy::Q4K, SiteTy::Q6K];

/// The step's word record: the position, the token, then one row word a key
/// head ([`ROW_WORDS`]).
pub(super) const WORD_POS: usize = 0;
pub(super) const WORD_TOKEN: usize = 1;
pub(super) const ROW_WORDS: usize = 2;

/// One projection of a layer: the file tensor's name, the launch its type
/// selects and the width of the rows it reads.
pub(super) struct Site {
    pub(super) name: String,
    pub(super) ty: SiteTy,
    pub(super) k: usize,
}

pub(super) struct MambaLayer {
    pub(super) norm: String,
    pub(super) ssm_in: Site,
    pub(super) ssm_out: Site,
    pub(super) a: String,
    pub(super) d: String,
    pub(super) dt_bias: String,
    pub(super) conv_w: String,
    pub(super) conv_b: String,
    pub(super) ssm_norm: String,
    pub(super) store: RecStore,
}

pub(super) struct AttnLayer {
    pub(super) norm: String,
    pub(super) q: Site,
    pub(super) k: Site,
    pub(super) v: Site,
    pub(super) o: Site,
    /// The head-major planes: head `h`, position `p` at row `h · ctx + p`.
    pub(super) kc: DeviceTensor<u16>,
    pub(super) vc: DeviceTensor<u16>,
}

pub(super) struct MlpLayer {
    pub(super) norm: String,
    pub(super) up: Site,
    pub(super) down: Site,
    pub(super) ff: usize,
}

/// A layer, by the kind the file's per-layer arrays give it.
pub(super) enum Layer {
    Mamba(MambaLayer),
    Attn(AttnLayer),
    Mlp(MlpLayer),
}

impl Layer {
    pub(super) fn sites(&self) -> Vec<&Site> {
        match self {
            Layer::Mamba(m) => vec![&m.ssm_in, &m.ssm_out],
            Layer::Attn(a) => vec![&a.q, &a.k, &a.v, &a.o],
            Layer::Mlp(m) => vec![&m.up, &m.down],
        }
    }
}

/// The widths the scratch is cut for and the chain runs at.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub(super) hidden: usize,
    pub(super) eps: f32,
    pub(super) mamba: Checked,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
}

/// The kernels the chain launches beyond the `Gpu`'s own.
pub(super) struct Kernels {
    pub(super) mamba: Mamba2Kernels,
    pub(super) flash: FlashGqaKernels,
    pub(super) g32: Gemm32Kernels,
    pub(super) kgemv: KGemvKernels,
    /// The grouped GEMM a prompt pass's K-quant sites run on.
    pub(super) gemm: GemmKernels,
}

/// The quantized activation forms the sites read: one of each form a site's
/// type reads, at each width a site reads it at.
pub(super) struct Acts {
    blocks32: Vec<Q8Blocks32>,
    q8_1: Vec<Q8Act>,
}

impl Acts {
    /// The forms `sites` read. Load-time only.
    pub(super) fn new<'s>(
        stream: &cuda_core::CudaStream,
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<Acts, GpuError> {
        let mut acts = Acts {
            blocks32: Vec::new(),
            q8_1: Vec::new(),
        };
        for s in sites {
            match s.ty {
                SiteTy::Q5_0 if !acts.blocks32.iter().any(|a| a.k() == s.k) => {
                    acts.blocks32.push(Q8Blocks32::new(stream, s.k, 1)?);
                }
                SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K | SiteTy::Nvfp4
                    if !acts.q8_1.iter().any(|a| a.k() == s.k) =>
                {
                    acts.q8_1.push(Q8Act::with_k(stream, 1, s.k)?);
                }
                _ => {}
            }
        }
        Ok(acts)
    }

    pub(super) fn bytes(&self) -> usize {
        self.blocks32
            .iter()
            .map(|a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
            .sum::<usize>()
            + self.q8_1.iter().map(Q8Act::device_bytes).sum::<usize>()
    }
}

/// The rows a chain runs over, `rows` of each width: the step's one row
/// ([`Scratch`]) and a prompt pass's rows ([`super::prefill`]), with the
/// flash partials and the embedding's row outputs.
pub(super) struct RowBuffers {
    pub(super) xn: DeviceBuffer<f32>,
    pub(super) delta: DeviceBuffer<f32>,
    pub(super) zxbcdt: DeviceBuffer<f32>,
    pub(super) y: DeviceBuffer<f32>,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) attn: DeviceBuffer<f32>,
    pub(super) up: DeviceBuffer<f32>,
    pub(super) act: DeviceBuffer<f32>,
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The embedding's outputs: the position word the scan reads and the live
    /// key count the flash reads, one of each a row.
    pub(super) pos: DeviceBuffer<u32>,
    pub(super) n_keys: DeviceBuffer<u32>,
}

impl RowBuffers {
    /// `rows` rows of each width for `sh`, `n_head` query heads and the
    /// widest `ff` an MLP layer has, the flash partials at the rows a flash
    /// run takes. Load-time only.
    pub(super) fn new(
        stream: &cuda_core::CudaStream,
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

    pub(super) fn bytes(&self) -> usize {
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
pub(super) struct Scratch {
    pub(super) buf: RowBuffers,
    pub(super) mamba: Mamba2Scratch,
    /// The record [`WORD_POS`], [`WORD_TOKEN`] and [`ROW_WORDS`] name.
    pub(super) words: DeviceBuffer<u32>,
}

impl Scratch {
    pub(super) fn bytes(&self) -> usize {
        self.buf.bytes() + self.words.num_bytes() + self.mamba.device_bytes()
    }

    /// The scratch for `sh`, the widest `ff` an MLP layer has and the
    /// projection widths `sh`'s layers read. Load-time only.
    pub(super) fn new(
        stream: &cuda_core::CudaStream,
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

/// What every launch of a step reads: the card, the resident weights, the
/// kernels and the widths.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    sh: Shape,
}

impl<'a> Cx<'a> {
    /// The resident f32 vector `name`.
    fn f32(&self, name: &str) -> Result<&'a DeviceBuffer<f32>, GpuError> {
        self.w.f32_buf(WHAT, name)
    }
}

impl Site {
    /// `y = W · x` for this site over the first `k` values of `x`, raising
    /// into layer `layer`'s fault sink.
    fn project(
        &self,
        cx: &Cx<'_>,
        acts: &mut Acts,
        layer: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let gpu = cx.gpu;
        let key = (self.ty, cx.w, self.name.as_str());
        match self.ty {
            SiteTy::Q8_0 | SiteTy::F32 => site::gemv(gpu, &cx.k.g32, key, x, 1, y),
            SiteTy::Q5_0 => {
                let act = acts
                    .blocks32
                    .iter_mut()
                    .find(|a| a.k() == self.k)
                    .ok_or_else(|| {
                        GpuError::state(WHAT, "a Q8Blocks32 form at the site's width")
                    })?;
                gpu.q5()
                    .enqueue_quantize_q8(gpu.stream(), x, act, gpu.layer_sink(layer)?)?;
                site::blocks32_gemv(gpu, key, act, y)
            }
            SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K | SiteTy::Nvfp4 => {
                let act = acts
                    .q8_1
                    .iter_mut()
                    .find(|a| a.k() == self.k)
                    .ok_or_else(|| GpuError::state(WHAT, "a q8_1 form at the site's width"))?;
                gpu.enqueue_quantize_q8_1_layer(x, act, layer)?;
                site::kgemv(gpu, &cx.k.kgemv, key, act, gpu.layer_sink(layer)?, y)
            }
        }
    }
}

/// `y = rms_norm(x, name)` over `m` rows of `hidden` at `eps`: the step's
/// one row and a prompt pass's rows ([`super::prefill`]).
pub(super) fn norm_rows(
    (gpu, w): (&Gpu, &Weights),
    (eps, hidden, m): (f32, usize, usize),
    name: &str,
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let gain = w.f32_buf(WHAT, name)?;
    gpu.elem()
        .enqueue_rms_norm(gpu.stream(), x, gain, eps, hidden, m, y)
}

/// `dst = src + delta` over `n` values, the layer's residual.
pub(super) fn residual_rows(
    gpu: &Gpu,
    n: usize,
    src: &DeviceBuffer<f32>,
    delta: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gpu.elem().enqueue_add(gpu.stream(), src, delta, n, dst)
}

/// One Mamba-2 layer: the `ssm_in` output is `[z | xBC | dt]`, three views of
/// the one buffer at `0`, `d_inner` and `d_inner + conv_dim` values (llama.cpp
/// `build_mamba2_layer`'s views at `0`, `d_inner`, `2·d_inner + 2·n_group·d_state`).
fn mamba_layer(
    cx: &Cx<'_>,
    (l, m): (usize, &mut MambaLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let sh = cx.sh;
    let (inner, conv_dim, heads) = (
        sh.mamba.geom().d_inner,
        sh.mamba.conv_dim(),
        sh.mamba.geom().n_heads,
    );
    norm_rows(
        (cx.gpu, cx.w),
        (cx.sh.eps, cx.sh.hidden, 1),
        &m.norm,
        src,
        &mut s.buf.xn,
    )?;
    m.ssm_in
        .project(cx, acts, l, &s.buf.xn, &mut s.buf.zxbcdt)?;
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
    m.ssm_out.project(cx, acts, l, &s.buf.y, &mut s.buf.delta)?;
    residual_rows(cx.gpu, cx.sh.hidden, src, &s.buf.delta, dst)
}

/// Key head `h`'s row of `buf`, the view an append reads.
fn head_rows<'b>(
    buf: &'b DeviceBuffer<f32>,
) -> impl FnMut(usize) -> Result<Window<'b, f32>, GpuError> {
    move |h| Window::<f32>::of(buf, h * HEAD * size_of::<f32>(), HEAD)
}

/// One attention layer: no rope and no qk norm, so the projections are the
/// flash's inputs and the planes' rows as they stand.
fn attn_layer(
    cx: &Cx<'_>,
    (l, a): (usize, &mut AttnLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let sh = cx.sh;
    let fault = cx.gpu.layer_sink(l)?;
    norm_rows(
        (cx.gpu, cx.w),
        (cx.sh.eps, cx.sh.hidden, 1),
        &a.norm,
        src,
        &mut s.buf.xn,
    )?;
    a.q.project(cx, acts, l, &s.buf.xn, &mut s.buf.q)?;
    a.k.project(cx, acts, l, &s.buf.xn, &mut s.buf.k)?;
    a.v.project(cx, acts, l, &s.buf.xn, &mut s.buf.v)?;
    let at = PlaneAppend {
        words: &s.words,
        first_word: ROW_WORDS,
        heads: sh.n_kv,
        m: 1,
        fault,
    };
    let flash = cx.gpu.flash();
    flash.enqueue_kv_append_planes(cx.gpu.stream(), &at, head_rows(&s.buf.k), &mut a.kc)?;
    flash.enqueue_kv_append_planes(cx.gpu.stream(), &at, head_rows(&s.buf.v), &mut a.vc)?;
    cx.k.flash.enqueue_pass::<GROUP_5>(
        cx.gpu.stream(),
        GqaArgs {
            q: &s.buf.q,
            kc: a.kc.buf(),
            vc: a.vc.buf(),
            n_keys: &s.buf.n_keys,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_kv: sh.n_kv,
            ctx: sh.ctx,
            m: 1,
            part_v: &mut s.buf.part_v,
            part_ms: &mut s.buf.part_ms,
            fault,
            y: &mut s.buf.attn,
        },
        false,
    )?;
    a.o.project(cx, acts, l, &s.buf.attn, &mut s.buf.delta)?;
    residual_rows(cx.gpu, cx.sh.hidden, src, &s.buf.delta, dst)
}

/// One MLP layer: `down(relu²(up(x)))`, no gate.
fn mlp_layer(
    cx: &Cx<'_>,
    (l, m): (usize, &MlpLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    norm_rows(
        (cx.gpu, cx.w),
        (cx.sh.eps, cx.sh.hidden, 1),
        &m.norm,
        src,
        &mut s.buf.xn,
    )?;
    m.up.project(cx, acts, l, &s.buf.xn, &mut s.buf.up)?;
    cx.gpu
        .elem()
        .enqueue_relu_sqr(cx.gpu.stream(), &s.buf.up, m.ff, &mut s.buf.act)?;
    m.down.project(cx, acts, l, &s.buf.act, &mut s.buf.delta)?;
    residual_rows(cx.gpu, cx.sh.hidden, src, &s.buf.delta, dst)
}

impl Body {
    /// The whole chain: the embedding, every layer, then `head`. The residual
    /// stream alternates between the head's input and the body's own buffer,
    /// each layer writing the one it did not read, so the last layer's sum is
    /// the head's input whatever the layer count's parity.
    pub(super) fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
    ) -> Result<(), GpuError> {
        let cx = Cx {
            gpu,
            w,
            k: &self.k,
            sh: self.sh,
        };
        let table = match w.get(&self.embed) {
            Some(DevWeight::Q5_0 { w, .. }) => w,
            _ => return Err(GpuError::tensor(WHAT, &self.embed, "a Q5_0 table")),
        };
        let hin = head.input_mut();
        let mut in_head = self.layers.len().is_multiple_of(2);
        {
            let ids = Window::<u32>::of(&self.s.words, WORD_TOKEN * size_of::<u32>(), 1)?;
            let pos0 = Window::<u32>::of(&self.s.words, WORD_POS * size_of::<u32>(), 1)?;
            let first = if in_head { &mut *hin } else { &mut self.r };
            gpu.elem().enqueue_embed_rows_q5_0(
                gpu.stream(),
                EmbedRowsArgs {
                    w: table,
                    ids: &ids,
                    pos0: &pos0,
                    first: 0,
                    y: first,
                    pos: &mut self.s.buf.pos,
                    n_keys: &mut self.s.buf.n_keys,
                },
                self.sh.hidden,
            )?;
        }
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let (src, dst) = if in_head {
                (&*hin, &mut self.r)
            } else {
                (&self.r as &DeviceBuffer<f32>, &mut *hin)
            };
            let scratch = (&mut self.s, &mut self.acts);
            match layer {
                Layer::Mamba(m) => mamba_layer(&cx, (l, m), scratch, (src, dst))?,
                Layer::Attn(a) => attn_layer(&cx, (l, a), scratch, (src, dst))?,
                Layer::Mlp(m) => mlp_layer(&cx, (l, m), scratch, (src, dst))?,
            }
            in_head = !in_head;
        }
        head.enqueue(gpu, w)
    }
}
