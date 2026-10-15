//! The prompt path: a prompt runs layer-major over passes of up to
//! [`PASS_ROWS`] ids, each projection one GEMM over the pass's rows
//! ([`site::gemm`] on the pass's one-expert table), each Mamba-2 layer one
//! pass over its tokens ([`Mamba2Kernels::enqueue_steps`]), each attention
//! layer the pass's K/V rows appended to its planes and the group-5 flash in
//! runs of [`MAX_PASS_ROWS`] rows, each row over the keys up to its own
//! position. The head reads the last row of the prompt's last pass only.
//!
//! After [`GpuModel::prefill_with`] the stores (the scan states, the conv
//! rings and the K/V planes) and the position are those of the prompt's ids
//! run as steps; the next [`GpuModel::step`] continues from them.
//!
//! The pass's rows live in [`Prompt`], cut at load for the most rows a pass
//! takes, so a pass allocates nothing. The GEMM sums each output in another
//! order than the step's gemv, so a pass's rows agree with the steps' to the
//! GEMM's band, not bit for bit.

use super::body::Body;
use super::step::{Layer, Site, norm_rows, residual_rows};
use crate::elem::EmbedRowsArgs;
use crate::flash::PlaneAppend;
use crate::flash_gqa::{GROUP_5, GqaArgs, HEAD, partials_ms_len, partials_v_len};
use crate::gemm::{GemmInput, GemmRoute};
use crate::head::Head;
use crate::model::{GpuModel, MAX_PASS_ROWS};
use crate::site::{self, SiteTy, SiteWeight, WideActs, WideKernels};
use crate::tensor::{Pitched, WindowMut, dtod_2d};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Window, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};

const WHAT: &str = "nemotron_h::prefill";

/// The most ids one pass of [`GpuModel::prefill`] takes.
pub const PASS_ROWS: usize = 512;

/// The pass's word record: the first position, then one row word a key head
/// (head `h` at `P_ROWS + h`, the plane row `h · ctx + pos` the pass's first
/// K/V row lands at), then the pass's ids from [`Prompt::ids_at`].
const P_POS: usize = 0;
const P_ROWS: usize = 1;

/// One input width's quantized forms: the width and its forms.
struct ActAt {
    k: usize,
    acts: WideActs,
}

/// The buffers one pass runs over, `rows` rows each.
pub(super) struct Prompt {
    rows: usize,
    /// The query heads' width a row, `n_head · HEAD`.
    q_width: usize,
    /// The pass's two residual streams, each `rows` rows of `hidden`.
    xa: DeviceBuffer<f32>,
    xb: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    delta: DeviceBuffer<f32>,
    zxbcdt: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    /// The pass's K and V rows regrouped head-major: head `h`'s rows at
    /// `h · rows · HEAD`, what the plane append reads.
    kh: DeviceBuffer<f32>,
    vh: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// Each row's position and live key count, the embedding's outputs.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    /// The word record ([`P_POS`], [`P_ROWS`], the ids) and its host copy.
    words: DeviceBuffer<u32>,
    host_words: Vec<u32>,
    /// The pass's one-expert table: slot `s` reads column `s`.
    route: GemmRoute,
    acts: Vec<ActAt>,
}

/// The widths a pass's buffers are cut for.
pub(super) struct PromptShape {
    pub(super) hidden: usize,
    pub(super) ld: usize,
    pub(super) inner: usize,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ff: usize,
}

impl Prompt {
    /// The buffers for passes of up to `rows` rows over `sh`, and the
    /// quantized forms each width `layers`' sites read. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        rows: usize,
        sh: &PromptShape,
        layers: &[Layer],
    ) -> Result<Prompt, GpuError> {
        if rows == 0 {
            return Err(GpuError::shape(WHAT, "a pass of 0 rows"));
        }
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, rows * n.max(1))?)
        };
        let kv_width = sh.n_kv * HEAD;
        let attn_rows = rows.min(MAX_PASS_ROWS);
        let mut acts: Vec<ActAt> = Vec::new();
        let mut forms: Vec<(usize, bool, bool)> = Vec::new();
        for s in layers.iter().flat_map(Layer::sites) {
            let (q128, q32) = WideActs::wants(s.ty);
            match forms.iter_mut().find(|f| f.0 == s.k) {
                Some(f) => {
                    f.1 |= q128;
                    f.2 |= q32;
                }
                None => forms.push((s.k, q128, q32)),
            }
        }
        for (k, q128, q32) in forms {
            acts.push(ActAt {
                k,
                acts: WideActs::new(stream, (q128, q32), rows, k)?,
            });
        }
        let n_words = P_ROWS + sh.n_kv + rows;
        Ok(Prompt {
            rows,
            q_width: sh.n_head * HEAD,
            xa: z(sh.hidden)?,
            xb: z(sh.hidden)?,
            xn: z(sh.hidden)?,
            delta: z(sh.hidden)?,
            zxbcdt: z(sh.ld)?,
            y: z(sh.inner)?,
            q: z(sh.n_head * HEAD)?,
            k: z(kv_width)?,
            v: z(kv_width)?,
            kh: z(kv_width)?,
            vh: z(kv_width)?,
            attn: z(sh.n_head * HEAD)?,
            up: z(sh.ff)?,
            act: z(sh.ff)?,
            part_v: DeviceBuffer::zeroed(stream, partials_v_len(attn_rows, sh.n_head))?,
            part_ms: DeviceBuffer::zeroed(stream, partials_ms_len(attn_rows, sh.n_head))?,
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
            words: DeviceBuffer::zeroed(stream, n_words)?,
            host_words: vec![0; n_words],
            route: GemmRoute::new(stream, rows, 1)?,
            acts,
        })
    }

    /// The device bytes the pass's buffers hold.
    pub(super) fn bytes(&self) -> usize {
        [
            &self.xa,
            &self.xb,
            &self.xn,
            &self.delta,
            &self.zxbcdt,
            &self.y,
            &self.q,
            &self.k,
            &self.v,
            &self.kh,
            &self.vh,
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
            + self.words.num_bytes()
            + self.route.bytes()
            + self.acts.iter().map(|a| a.acts.bytes()).sum::<usize>()
    }

    /// The word index the pass's ids start at.
    fn ids_at(n_kv: usize) -> usize {
        P_ROWS + n_kv
    }
}

/// What every launch of a pass reads.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    wk: WideKernels<'a>,
    m: usize,
}

/// The forms of width `k` in `acts`, or a named refusal.
fn acts_at(acts: &mut [ActAt], k: usize) -> Result<&mut WideActs, GpuError> {
    acts.iter_mut()
        .find(|a| a.k == k)
        .map(|a| &mut a.acts)
        .ok_or_else(|| {
            GpuError::shape(WHAT, format!("no quantized forms of width {k} in the pass"))
        })
}

/// `y = W · x` for each of `sites` over the pass's `m` rows of `x`, `rows`
/// output rows each: `x` quantized once into the forms the sites read.
fn project(
    cx: &Cx<'_>,
    (acts, route): (&mut [ActAt], &GemmRoute),
    layer: usize,
    x: &DeviceBuffer<f32>,
    sites: &mut [(&Site, usize, &mut DeviceBuffer<f32>)],
) -> Result<(), GpuError> {
    let Some(k) = sites.first().map(|s| s.0.k) else {
        return Ok(());
    };
    if let Some(s) = sites.iter().find(|s| s.0.k != k) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "sites of one input read one width: {} reads {}, not {k}",
                s.0.name, s.0.k
            ),
        ));
    }
    let tys: Vec<SiteTy> = sites.iter().map(|s| s.0.ty).collect();
    let a = acts_at(acts, k)?;
    a.quantize(cx.gpu, &cx.wk, cx.gpu.layer_sink(layer)?, x, cx.m, &tys)?;
    let input = a.input(x);
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

impl Body {
    /// One pass of `ids` (at most the prompt buffers' rows) from position
    /// `pos`, layer-major; with `head`, the head over the pass's last row.
    fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (ids, pos): (&[u32], u32),
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let sh = self.sh;
        let n_kv = sh.n_kv;
        let stream = gpu.stream();
        let p = &mut self.prompt;
        let q_width = p.q_width;
        if m == 0 || m > p.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} ids; the prompt buffers hold 1..={}", p.rows),
            ));
        }
        p.host_words[P_POS] = pos;
        for h in 0..n_kv {
            p.host_words[P_ROWS + h] = h
                .checked_mul(sh.ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("row {h} · {} + {pos} does not fit a u32 row word", sh.ctx),
                    )
                })?;
        }
        let at = Prompt::ids_at(n_kv);
        p.host_words[at..at + m].copy_from_slice(ids);
        p.words.copy_from_host(stream, &p.host_words)?;
        let table = match w.get(&self.embed) {
            Some(DevWeight::Q5_0 { w, .. }) => w,
            _ => return Err(GpuError::tensor(WHAT, &self.embed, "a Q5_0 table")),
        };
        {
            let ids_w = Window::<u32>::of(&p.words, at * size_of::<u32>(), m)?;
            let pos0 = Window::<u32>::of(&p.words, P_POS * size_of::<u32>(), 1)?;
            gpu.elem().enqueue_embed_rows_q5_0(
                stream,
                EmbedRowsArgs {
                    w: table,
                    ids: &ids_w,
                    pos0: &pos0,
                    first: 0,
                    y: &mut p.xa,
                    pos: &mut p.pos,
                    n_keys: &mut p.n_keys,
                },
                sh.hidden,
            )?;
        }
        self.k
            .gemm
            .enqueue_route_dense(stream, m, &mut p.route, gpu.unlabelled_sink())?;
        let cx = Cx {
            gpu,
            w,
            wk: WideKernels {
                gemm: &self.k.gemm,
                g32: Some(&self.k.g32),
            },
            m,
        };
        let mut in_a = true;
        for (l, layer) in self.layers.iter_mut().enumerate() {
            let Prompt {
                xa,
                xb,
                xn,
                delta,
                zxbcdt,
                y,
                q,
                k,
                v,
                kh,
                vh,
                attn,
                up,
                act,
                part_v,
                part_ms,
                pos: rows_pos,
                n_keys,
                words,
                route,
                acts,
                ..
            } = p;
            let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) =
                if in_a { (xa, xb) } else { (xb, xa) };
            let fault = gpu.layer_sink(l)?;
            match layer {
                Layer::Mamba(mm) => {
                    let g = sh.mamba.geom();
                    let (inner, conv_dim) = (g.d_inner, sh.mamba.conv_dim());
                    let ld = sh.mamba.in_proj_width();
                    norm_rows((gpu, w), (sh.eps, sh.hidden, m), &mm.norm, src, xn)?;
                    project(&cx, (acts, route), l, xn, &mut [(&mm.ssm_in, ld, zxbcdt)])?;
                    self.k.mamba.enqueue_steps(
                        stream,
                        crate::mamba2::StepsArgs {
                            shape: sh.mamba,
                            u: m,
                            rows: zxbcdt,
                            ld,
                            z0: 0,
                            x0: inner,
                            dt0: inner + conv_dim,
                            dt_bias: w.f32_buf(WHAT, &mm.dt_bias)?,
                            a: w.f32_buf(WHAT, &mm.a)?,
                            d: w.f32_buf(WHAT, &mm.d)?,
                            conv_w: w.f32_buf(WHAT, &mm.conv_w)?,
                            conv_b: w.f32_buf(WHAT, &mm.conv_b)?,
                            norm: Some(w.f32_buf(WHAT, &mm.ssm_norm)?),
                            eps: sh.eps,
                            pos: rows_pos,
                            pos_at: 0,
                            fault,
                            y,
                            ring: &mut mm.store.ring,
                            state: &mut mm.store.state,
                        },
                    )?;
                    project(
                        &cx,
                        (acts, route),
                        l,
                        y,
                        &mut [(&mm.ssm_out, sh.hidden, delta)],
                    )?;
                }
                Layer::Attn(a) => {
                    let kv_width = n_kv * HEAD;
                    norm_rows((gpu, w), (sh.eps, sh.hidden, m), &a.norm, src, xn)?;
                    project(
                        &cx,
                        (acts, route),
                        l,
                        xn,
                        &mut [(&a.q, q_width, q), (&a.k, kv_width, k), (&a.v, kv_width, v)],
                    )?;
                    let f = size_of::<f32>();
                    for h in 0..n_kv {
                        let shape = Pitched {
                            width: HEAD * f,
                            rows: m,
                            dev_pitch: kv_width * f,
                            host_pitch: HEAD * f,
                        };
                        dtod_2d(stream, (k, h * HEAD * f), (kh, h * m * HEAD * f), shape)?;
                        dtod_2d(stream, (v, h * HEAD * f), (vh, h * m * HEAD * f), shape)?;
                    }
                    let append = PlaneAppend {
                        words,
                        first_word: P_ROWS,
                        heads: n_kv,
                        m,
                        fault,
                    };
                    let flash = gpu.flash();

                    flash.enqueue_kv_append_planes(
                        stream,
                        &append,
                        |h| Window::<f32>::of(kh, h * m * HEAD * f, m * HEAD),
                        &mut a.kc,
                    )?;
                    flash.enqueue_kv_append_planes(
                        stream,
                        &append,
                        |h| Window::<f32>::of(vh, h * m * HEAD * f, m * HEAD),
                        &mut a.vc,
                    )?;
                    let mut r0 = 0;
                    while r0 < m {
                        let mm = (m - r0).min(MAX_PASS_ROWS);
                        let qw = Window::<f32>::of(q, r0 * q_width * f, mm * q_width)?;
                        let nk = Window::<u32>::of(n_keys, r0 * size_of::<u32>(), mm)?;
                        let mut yw =
                            WindowMut::<f32>::of_mut(attn, r0 * q_width * f, mm * q_width)?;
                        self.k.flash.enqueue_pass::<GROUP_5>(
                            stream,
                            GqaArgs {
                                q: &qw,
                                kc: a.kc.buf(),
                                vc: a.vc.buf(),
                                n_keys: &nk,
                                scale: 1.0 / (HEAD as f32).sqrt(),
                                n_kv,
                                ctx: sh.ctx,
                                m: mm,
                                part_v,
                                part_ms,
                                fault,
                                y: &mut yw,
                            },
                            false,
                        )?;
                        r0 += mm;
                    }
                    project(&cx, (acts, route), l, attn, &mut [(&a.o, sh.hidden, delta)])?;
                }
                Layer::Mlp(ml) => {
                    norm_rows((gpu, w), (sh.eps, sh.hidden, m), &ml.norm, src, xn)?;
                    project(&cx, (acts, route), l, xn, &mut [(&ml.up, ml.ff, up)])?;
                    gpu.elem().enqueue_relu_sqr(stream, up, m * ml.ff, act)?;
                    project(
                        &cx,
                        (acts, route),
                        l,
                        act,
                        &mut [(&ml.down, sh.hidden, delta)],
                    )?;
                }
            }
            residual_rows(gpu, m * sh.hidden, src, delta, dst)?;
            in_a = !in_a;
        }
        if let Some(head) = head {
            let last = if in_a { &p.xa } else { &p.xb };
            let row = Window::<f32>::of(last, (m - 1) * sh.hidden * size_of::<f32>(), sh.hidden)?;
            head.input_mut().copy_from_device_async(&row, stream)?;
            head.enqueue(gpu, w)?;
        }
        Ok(())
    }
}

impl GpuModel<Body> {
    /// [`GpuModel::prefill_with`] in passes of the most ids the prompt
    /// buffers hold: [`PASS_ROWS`], or the context where that is less.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        let rows = self.body(WHAT)?.prompt.rows;
        self.prefill_with(ids, rows)
    }

    /// Feed `ids` through the chain in passes of up to `rows` ids (module
    /// doc) and return the greedy next token after the last one. Positions
    /// continue from wherever the model stands, and the stores after it are
    /// those of the ids run as steps. `rows` past the prompt buffers' rows,
    /// or 0, and an empty `ids` are refused by name. The passes run eager in
    /// either step mode; a fault any pass raised is read behind the last
    /// pass's head (`GpuModel::run_rows`, the call-end read).
    pub fn prefill_with(&mut self, ids: &[u32], rows: usize) -> Result<u32, GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let cap = self.body(WHAT)?.prompt.rows;
        if rows == 0 || rows > cap {
            return Err(GpuError::shape(
                WHAT,
                format!("passes of {rows} ids; the prompt buffers hold 1..={cap}"),
            ));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "ids", ids.len())? - 1, WHAT)?;
        let n = ids.len().div_ceil(rows);
        let mut next = None;
        for (i, pass) in ids.chunks(rows).enumerate() {
            let last = i + 1 == n;
            next = self.run_rows(pass.len(), WHAT, |gpu, w, body, head, pos| {
                body.enqueue_pass(gpu, w, (pass, pos), last.then_some(head))?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
    }
}
