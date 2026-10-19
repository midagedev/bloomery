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
//! ([`MambaLayer`], [`AttnLayer`]), resolved at load by [`Names`]; the trunk's
//! widths ([`Shape`]), kernels ([`Kernels`]) and step scratch ([`Scratch`]);
//! the launches of one layer of each kind ([`mamba_layer`], [`attn_layer`])
//! and the pre-norm and residual add every layer brackets its block with
//! ([`norm`], [`residual`]); and the step's word record ([`Words`]: the
//! position, the token and one plane row a key head), which a body refreshes
//! before every enqueue or replay, so one captured graph serves every
//! position.

use crate::flash::PlaneAppend;
use crate::flash_gqa::{FlashGqaKernels, GqaArgs, HEAD, partials_ms_len, partials_v_len};
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::mamba2::{Mamba2Kernels, Mamba2Scratch, StepArgs};
use crate::q5::Q8Blocks32;
use crate::recur::RecStore;
use crate::site::{self, Form, KGemvKernels, SiteTy};
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
pub(crate) struct Site {
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

pub(crate) struct AttnLayer {
    pub(crate) norm: String,
    pub(crate) q: Site,
    pub(crate) k: Site,
    pub(crate) v: Site,
    pub(crate) o: Site,
    /// The head-major planes: head `h`, position `p` at row `h · ctx + p`.
    pub(crate) kc: DeviceTensor<u16>,
    pub(crate) vc: DeviceTensor<u16>,
}

impl MambaLayer {
    pub(crate) fn sites(&self) -> [&Site; 2] {
        [&self.ssm_in, &self.ssm_out]
    }
}

impl AttnLayer {
    pub(crate) fn sites(&self) -> [&Site; 4] {
        [&self.q, &self.k, &self.v, &self.o]
    }

    /// The K/V planes' bytes.
    pub(crate) fn bytes(&self) -> usize {
        self.kc.buf().num_bytes() + self.vc.buf().num_bytes()
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
pub(crate) struct Acts {
    blocks32: Vec<Q8Blocks32>,
    q8_1: Vec<Q8Act>,
}

impl Acts {
    /// The forms `sites` read. Load-time only.
    pub(crate) fn new<'s>(
        stream: &CudaStream,
        sites: impl Iterator<Item = &'s Site>,
    ) -> Result<Acts, GpuError> {
        let mut acts = Acts {
            blocks32: Vec::new(),
            q8_1: Vec::new(),
        };
        for s in sites {
            match s.ty.reads() {
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
    pub(crate) fn q8_1_at(
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

    pub(crate) fn bytes(&self) -> usize {
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
}

impl Site {
    /// `y = W · act` over the site's already-quantized q8_1 form, raising
    /// into layer `layer`'s fault sink — the gemv half of the q8_1 arm of
    /// [`Site::project`], which a caller that quantized by other means
    /// ([`mamba_layer`]'s fused norm) launches alone.
    fn kgemv(
        &self,
        cx: &Cx<'_>,
        act: &Q8Act,
        layer: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let key = (self.ty, cx.w, self.name.as_str());
        site::kgemv(cx.gpu, &cx.k.kgemv, key, act, cx.gpu.layer_sink(layer)?, y)
    }

    /// `y = W · x` for this site over the first `k` values of `x`, raising
    /// into layer `layer`'s fault sink.
    pub(crate) fn project(
        &self,
        cx: &Cx<'_>,
        acts: &mut Acts,
        layer: usize,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let gpu = cx.gpu;
        let key = (self.ty, cx.w, self.name.as_str());
        match self.ty.reads() {
            // Q8_0's form: the f32 rows on this gemv arm.
            Form::F32 | Form::Q8x32 => site::gemv(gpu, &cx.k.g32, key, x, 1, y),
            Form::Q8Blocks32 => {
                let act = acts
                    .blocks32
                    .iter_mut()
                    .find(|a| a.k() == self.k)
                    .ok_or_else(|| {
                        GpuError::state(cx.what, "a Q8Blocks32 form at the site's width")
                    })?;
                gpu.q5()
                    .enqueue_quantize_q8(gpu.stream(), x, act, gpu.layer_sink(layer)?)?;
                site::blocks32_gemv(gpu, key, act, y)
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
    let (inner, conv_dim, heads) = (
        sh.mamba.geom().d_inner,
        sh.mamba.conv_dim(),
        sh.mamba.geom().n_heads,
    );
    if m.ssm_in.ty.reads() == Form::Q8x128 {
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
        m.ssm_in.kgemv(cx, act, l, &mut s.buf.zxbcdt)?;
    } else {
        norm(cx, &m.norm, src, &mut s.buf.xn)?;
        m.ssm_in
            .project(cx, acts, l, &s.buf.xn, &mut s.buf.zxbcdt)?;
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
    m.ssm_out.project(cx, acts, l, &s.buf.y, &mut s.buf.delta)?;
    residual(cx, src, &s.buf.delta, dst)
}

/// Key head `h`'s row of `buf`, the view an append reads.
fn head_rows<'b>(
    buf: &'b DeviceBuffer<f32>,
) -> impl FnMut(usize) -> Result<Window<'b, f32>, GpuError> {
    move |h| Window::<f32>::of(buf, h * HEAD * size_of::<f32>(), HEAD)
}

/// One attention layer of `PACK` query heads a key head (a group the flash
/// family's `enqueue_pass` takes): no rope and no qk norm, so the
/// projections are the flash's inputs and the planes' rows as they stand.
pub(crate) fn attn_layer<const PACK: usize>(
    cx: &Cx<'_>,
    (l, a): (usize, &mut AttnLayer),
    (s, acts): (&mut Scratch, &mut Acts),
    (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let sh = cx.sh;
    let fault = cx.gpu.layer_sink(l)?;
    norm(cx, &a.norm, src, &mut s.buf.xn)?;
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
    cx.k.flash.enqueue_pass::<PACK>(
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
        let got = self.w.f32_buf(self.what, ZERO_CONV_BIAS)?.len();
        if got != len {
            return Err(GpuError::shape(
                self.what,
                format!("{ZERO_CONV_BIAS} holds {got} values, the layer reads {len}"),
            ));
        }
        Ok(ZERO_CONV_BIAS.to_owned())
    }

    /// The f32 vector `stem` of layer `l`, `len` values: its name.
    pub(crate) fn vector(&self, l: usize, stem: &str, len: usize) -> Result<String, GpuError> {
        let name = layer_name(l, stem);
        let got = self.w.f32_buf(self.what, &name)?.len();
        if got != len {
            return Err(GpuError::shape(
                self.what,
                format!("{name} holds {got} values, the layer reads {len}"),
            ));
        }
        Ok(name)
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
    /// by the caller, and its K/V planes of `ctx` rows a key head.
    pub(crate) fn attn(
        &mut self,
        (l, norm): (usize, String),
        (n_head, kv): (usize, usize),
        (hidden, ctx): (usize, usize),
        stream: &CudaStream,
    ) -> Result<AttnLayer, GpuError> {
        let width = kv * HEAD;
        Ok(AttnLayer {
            q: self.proj(l, ATTN_Q, (n_head * HEAD, hidden))?,
            k: self.proj(l, ATTN_K, (width, hidden))?,
            v: self.proj(l, ATTN_V, (width, hidden))?,
            o: self.proj(l, ATTN_OUTPUT, (hidden, n_head * HEAD))?,
            kc: DeviceTensor::zeroed(stream, kv * ctx, HEAD)?,
            vc: DeviceTensor::zeroed(stream, kv * ctx, HEAD)?,
            norm,
        })
    }
}
