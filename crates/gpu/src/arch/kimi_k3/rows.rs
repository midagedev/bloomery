//! Kimi K3's pass of rows: 1 to [`PASS_ROWS`] consecutive positions through the chain as one
//! walk, each row the one-token step's bits ([`Body::walk`]'s launches at `m` rows), the state
//! book advanced as that many steps would.
//!
//! A child module of the step's, so it reads the step's private layers and scratch; the pass's own
//! buffers are [`Pass`], made once beside the model and lent to every walk. What the step's
//! launches become at `m` rows, and why each row is the step's bits:
//!
//! - the embedding writes every row's position and live key count, as the step's does for one;
//! - the AttnRes mix is one block a row ([`crate::attn_res`]: banks `stride` apart, bank `b`'s
//!   row `t` at `b · stride + t · n`), so the banked inputs of a pass are `m` rows a bank and the
//!   mix of a row reads only that row's;
//! - a delta layer runs its rows through the mixer's own recurrence in position order on the
//!   one-lane store in place ([`super::kda::rows`]); a latent layer appends its rows to the cache
//!   and attends each to the keys up to its own position ([`mla::rows`]), and on a load that
//!   selects ([`super::index`]) runs its rows' projections and index keys at once and each row's
//!   pick and core in position order, the step's launches at that row
//!   ([`super::index::attend_rows_sel`], over the rows' count words [`super::index::pass_words`]);
//! - the feed-forward block runs the routed block's rows ([`ffn::ffn_rows`]): the router a row,
//!   the card leg's slots local to a row;
//! - every projection is each row's one-row launch: the K-quant m-column gemv, the f32 and q8_0
//!   m-column launches, and the one-row launch on each row where a site has no m-column launch
//!   ([`crate::arch::rows_kit::project_rows`]); the norms and adds are element-wise or a block
//!   a row.
//!
//! One row is the step itself (`kda::rows`, `mla::rows` and the block at one row run the step's
//! launches), so a pass of one is a step in the pass's buffers.
//!
//! The reductions stay in the step's order: no launch sums across rows, and none changes the
//! order within a row. A wide unit (a GEMM over more rows than a pass) would sum a row's products
//! in the GEMM's tile order, which is not the step's; it is not here.

use super::super::kda;
use super::super::moe::MoeDims;
use super::super::state::Lane;
use super::index::{self, WORDS};
use super::{
    Body, Instance, KdaBuffers, Kernels, Layer, Mixer, MixerIo, MlaBuffers, Scratch, step_cx,
};
use crate::arch::gated_mla::{self as mla, MlaBufs};
use crate::arch::hybrid::{Acts, GqaCx};
use crate::arch::routed_ffn::{self as ffn, Ffn, FfnRows};
use crate::arch::rows_kit::{RowKernels, RowsKit, add_row_forms};
use crate::attn_res::AttnResArgs;
use crate::elem::EmbedRowsArgs;
use crate::head::Head;
use crate::kquant::Act;
use crate::latent_gated::{self, MlaDims};
use crate::linear::mixer_kda::{KdaBufs, KdaDims};
use crate::linear::{HEAD, PASS_ROWS};
use crate::model::GpuModel;
use crate::route_body::{sigmoid_8_2, sigmoid_896_16};
use crate::tensor::{Window, WindowMut, copy_row};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, LAYER_HEAD, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::kimi_k3::hparams::{INDEX_DIM, INDEX_HEADS};
use model::arch::kimi_k3::names;

/// What the refusals of this file name.
const ROWS: &str = "kimi_k3 rows";

/// Where a pass's final mix lands, after the last layer.
pub enum Tail<'a> {
    /// The mix of every row into `head`, a head of the pass's rows, and the head's launches: each
    /// row's logits and argmax.
    Head(&'a mut Head),
    /// The mix of every row, the last row's into `head`, a head of one row, and the head's
    /// launches: the last row's logits and argmax.
    Last(&'a mut Head),
    /// No head: the pass leaves the stores and caches for the next.
    Open,
}

/// The ids and the first position of a pass, staged on the card by one copy, the embedding's
/// per-row positions and live key counts the layers read, and on a load that selects the rows'
/// count words ([`index::pass_words`]), staged by one copy more.
struct PassIn {
    host: Vec<u32>,
    /// [`PASS_ROWS`] ids, then the first position.
    buf: DeviceBuffer<u32>,
    pos: DeviceBuffer<u32>,
    keys: DeviceBuffer<u32>,
    words_host: Vec<u32>,
    /// `[PASS_ROWS][WORDS]`.
    words: DeviceBuffer<u32>,
}

impl PassIn {
    fn new(stream: &CudaStream) -> Result<PassIn, GpuError> {
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        Ok(PassIn {
            host: vec![0; PASS_ROWS + 1],
            buf: u(PASS_ROWS + 1)?,
            pos: u(PASS_ROWS)?,
            keys: u(PASS_ROWS)?,
            words_host: vec![0; PASS_ROWS * WORDS],
            words: u(PASS_ROWS * WORDS)?,
        })
    }

    /// The count words of `m` rows from position `pos` over lists of `top_k` rows, in one copy.
    fn stage_words(
        &mut self,
        stream: &CudaStream,
        (pos, m): (u32, usize),
        top_k: u32,
    ) -> Result<(), GpuError> {
        let host = &mut self.words_host[..m * WORDS];
        index::pass_words(host, pos, top_k)?;
        let mut dst = WindowMut::<u32>::of_mut(&mut self.words, 0, m * WORDS)?;
        Ok(dst.copy_from_host(stream, host)?)
    }

    /// `ids` and the position `pos` of the first, in one copy that waits for the walks before it.
    fn stage(&mut self, stream: &CudaStream, ids: &[u32], pos: u32) -> Result<(), GpuError> {
        self.host.fill(0);
        self.host[..ids.len()].copy_from_slice(ids);
        self.host[PASS_ROWS] = pos;
        Ok(self.buf.copy_from_host(stream, &self.host)?)
    }

    fn bytes(&self) -> usize {
        [&self.buf, &self.pos, &self.keys, &self.words]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum()
    }
}

/// The routed block's buffers over a pass's rows, at the router instance the body was loaded for.
enum PassFfn {
    E896(FfnRows<sigmoid_896_16::RouterKernels>),
    E8(FfnRows<sigmoid_8_2::RouterKernels>),
}

impl PassFfn {
    fn bytes(&self) -> usize {
        match self {
            PassFfn::E896(p) => p.bytes(),
            PassFfn::E8(p) => p.bytes(),
        }
    }

    /// The block of layer `l` over `m` rows of `xn` into `y`, 2 to [`PASS_ROWS`]: the router
    /// instance of `router` and of these buffers must be the same.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's inputs: the router, the launch context, the forms, the rule, the \
                  layer, the rows, the output and the kit"
    )]
    fn run(
        &mut self,
        router: &Instance,
        cx: &GqaCx<'_>,
        acts: &mut Acts,
        rule: (Act, f32),
        (l, f): (usize, &Ffn),
        (m, xn): (usize, &DeviceBuffer<f32>),
        y: &mut DeviceBuffer<f32>,
        kit: RowsKit<'_>,
    ) -> Result<(), GpuError> {
        match (router, self) {
            (Instance::E896(r), PassFfn::E896(p)) => {
                ffn::ffn_rows(cx, &r.k, acts, rule, (l, f), m, xn, p, y, kit)
            }
            (Instance::E8(r), PassFfn::E8(p)) => {
                ffn::ffn_rows(cx, &r.k, acts, rule, (l, f), m, xn, p, y, kit)
            }
            _ => Err(GpuError::shape(
                ROWS,
                "a pass made for the other router instance than the body's",
            )),
        }
    }
}

/// A pass's own buffers: [`PASS_ROWS`] rows of each width the step has one of, the banked block
/// inputs of every row, the stage a row-major projection's columns land in, the routed block's
/// rows, and on a load that selects the rows' index queries and weights. Made once
/// ([`Pass::new`], load-time) and lent to every walk; it holds no state a walk leaves for the
/// next, so a model's reset needs nothing of it.
pub struct Pass {
    s: Scratch,
    /// `banks · PASS_ROWS · n` values: bank `b`'s row `t` at `b · PASS_ROWS · n + t · n`.
    banks: DeviceBuffer<f32>,
    stage: DeviceBuffer<f32>,
    rk: RowKernels,
    ffn: PassFfn,
    input: PassIn,
    /// `[PASS_ROWS][heads · d]` and `[PASS_ROWS][heads]` on a load that selects, one value
    /// otherwise.
    iq: DeviceBuffer<f32>,
    iw: DeviceBuffer<f32>,
    n: usize,
}

/// The step's scratch at `rows` rows: the widths of [`Scratch::new`], each `rows` times over.
fn scratch(
    stream: &CudaStream,
    n: usize,
    (kd, md): (&KdaDims, &MlaDims),
    ctx: usize,
    rows: usize,
) -> Result<Scratch, GpuError> {
    let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, (rows * len).max(1));
    let lin = kd.shape;
    let (channels, v, n_v) = (lin.channels(), lin.n_v * HEAD, lin.n_v);
    let (part_v, part_ms) = latent_gated::partials_lens_rows(md, ctx, rows);
    let (heads, row) = (md.heads, md.row());
    Ok(Scratch {
        sum: [z(n)?, z(n)?],
        mixed: z(n)?,
        update: z(n)?,
        xn: z(n)?,
        ffn_xn: z(n)?,
        kda: KdaBuffers {
            qkv: z(channels)?,
            fa: z(HEAD)?,
            ga: z(1)?,
            beta_raw: z(n_v)?,
            f: z(v)?,
            z: z(v)?,
            conv: z(channels)?,
            beta: z(n_v)?,
            decay: z(v)?,
            o: z(v)?,
            gated: z(v)?,
        },
        mla: MlaBuffers {
            sel: None,
            qa: z(md.q_lora)?,
            qr: z(md.q_lora)?,
            kv: z(row)?,
            gate: z(heads * md.v_head)?,
            q: z(heads * md.head_k())?,
            qabs: z(heads * row)?,
            att: z(heads * crate::latent::LATENT)?,
            av: z(heads * md.v_head)?,
            gated: z(heads * md.v_head)?,
            part_v: DeviceBuffer::zeroed(stream, part_v.max(1))?,
            part_ms: DeviceBuffer::zeroed(stream, part_ms.max(1))?,
        },
    })
}

impl Pass {
    /// The buffers for passes of `body`'s model, and the q8_1 forms of 2 to [`PASS_ROWS`] columns
    /// its K-quant sites read, which the body's activation forms gain here. Load-time only.
    ///
    /// # Errors
    /// [`GpuError`] for a placed body, whose passes are the placed pass's ([`super::pass`]), and
    /// from the allocations.
    pub fn new(gpu: &Gpu, body: &mut Body) -> Result<Pass, GpuError> {
        if body.placed().is_some() {
            return Err(GpuError::shape(
                ROWS,
                "a placed body runs no card-alone rows pass: its passes are the placed pass's",
            ));
        }
        let stream = gpu.stream();
        let Body {
            hp,
            kda: kd,
            mla: md,
            layers,
            acts,
            router,
            ctx,
            s: step,
            ..
        } = body;
        let selects = step.mla.sel.is_some();
        let pick = |width: usize| {
            DeviceBuffer::<f32>::zeroed(stream, if selects { PASS_ROWS * width } else { 1 })
        };
        add_row_forms(stream, acts, layers.iter().flat_map(Layer::sites))?;
        let n = hp.n_embd;
        let moe = MoeDims::of(hp);
        let ff = hp.dense_ff.max(moe.shared_ff);
        let shape = moe.shape();
        let bank_slots = hp.n_layer.div_ceil(hp.attn_res_block);
        let wide = n.max(moe.latent).max(ff);
        let stage = [kda::stage_len(kd), mla::stage_len(md), PASS_ROWS * wide]
            .into_iter()
            .max()
            .unwrap_or(1);
        let ffn = match router {
            Instance::E896(_) => PassFfn::E896(FfnRows::new(stream, n, (ff, shape), PASS_ROWS)?),
            Instance::E8(_) => PassFfn::E8(FfnRows::new(stream, n, (ff, shape), PASS_ROWS)?),
        };
        Ok(Pass {
            s: scratch(stream, n, (kd, md), *ctx, PASS_ROWS)?,
            banks: DeviceBuffer::zeroed(stream, bank_slots * PASS_ROWS * n)?,
            stage: DeviceBuffer::zeroed(stream, stage)?,
            rk: RowKernels::load(gpu.context())?,
            ffn,
            input: PassIn::new(stream)?,
            iq: pick(INDEX_HEADS * INDEX_DIM)?,
            iw: pick(INDEX_HEADS)?,
            n,
        })
    }

    /// Device bytes the pass holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.s.bytes()
            + self.banks.num_bytes()
            + self.stage.num_bytes()
            + self.ffn.bytes()
            + self.input.bytes()
            + self.iq.num_bytes()
            + self.iw.num_bytes()
    }
}

/// What a pass's latent layer selects with on a load that selects: the rows' index queries and
/// weights (`[m][heads · d]`, `[m][heads]`), their count words (`[m][WORDS]`) and the step's
/// selection buffers, which each row reuses in turn.
pub(super) struct PassSel<'a> {
    pub(super) iq: &'a mut DeviceBuffer<f32>,
    pub(super) iw: &'a mut DeviceBuffer<f32>,
    pub(super) words: &'a DeviceBuffer<u32>,
    pub(super) sel: &'a mut index::Sel,
}

/// Layer `l`'s mixer over the pass's rows `io.input`: the delta layer's rows on its one-lane store
/// in place, or the gated latent layer's on its cache, over the rows its indexer picks for each
/// row on a load that selects (`pick`, module doc); the updates land in `io.out`. A latent layer
/// whose `pick` and the load's indexer disagree is refused by name.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch context, the kit, the layer, the dims, the buffers and the selection, \
              each named as the step's mixer names it (rust-quality R8)"
)]
pub(super) fn run_mixer_rows(
    cx: &GqaCx<'_>,
    (k, acts, lane): (&Kernels, &mut Acts, &Lane),
    kit: RowsKit<'_>,
    (l, m, mixer): (usize, usize, &mut Mixer),
    (kd, md): (&KdaDims, &MlaDims),
    io: MixerIo<'_>,
    pick: Option<PassSel<'_>>,
) -> Result<(), GpuError> {
    let MixerIo {
        input,
        pos,
        keys,
        xn,
        kb,
        mb,
        out,
    } = io;
    match mixer {
        Mixer::Kda { sites, store } => {
            let st = lane.state(store)?;
            let bufs = KdaBufs {
                x: input,
                pos,
                xn,
                qkv: &mut kb.qkv,
                fa: &mut kb.fa,
                ga: &mut kb.ga,
                beta_raw: &mut kb.beta_raw,
                write_raw: None,
                f: &mut kb.f,
                z: &mut kb.z,
                conv: &mut kb.conv,
                beta: &mut kb.beta,
                decay: &mut kb.decay,
                o: &mut kb.o,
                gated: &mut kb.gated,
                out,
            };
            kda::rows(cx, kit, acts, sites, (l, kd, m), st, bufs)
        }
        Mixer::Mla {
            sites,
            cache,
            index,
        } => {
            let mut bufs = MlaBufs {
                x: input,
                pos,
                n_keys: keys,
                xn: &mut *xn,
                qa: &mut mb.qa,
                qr: &mut mb.qr,
                kv: &mut mb.kv,
                gate: &mut mb.gate,
                q: &mut mb.q,
                qabs: &mut mb.qabs,
                att: &mut mb.att,
                av: &mut mb.av,
                gated: &mut mb.gated,
                part_v: &mut mb.part_v,
                part_ms: &mut mb.part_ms,
                out,
                cache,
            };
            let mut kit = kit;
            let sites = &**sites;
            match (index.as_deref_mut(), k.index.as_ref(), pick) {
                (_, None, None) => mla::rows(cx, &k.mla, acts, (sites, md), (l, m), bufs, kit),
                (Some(ix), Some(ik), Some(p)) => {
                    let PassSel { iq, iw, words, sel } = p;
                    mla::front_rows(cx, acts, (sites, md), (l, m), &mut bufs, &mut kit)?;
                    let ins = (&*bufs.xn, &*bufs.qr, bufs.pos);
                    let into = (&mut *iq, &mut *iw);
                    ix.project_pass(cx, ik, (acts, &mut kit), (l, m), ins, bufs.qa, into)?;
                    let mla::Core { a, s, cache, y } =
                        mla::core(cx.w, cx.gpu.layer_sink(l)?, sites, &mut bufs)?;
                    let pick = sel.pass_pick(ix.keys(), (&*iq, &*iw), words);
                    index::attend_rows_sel(cx.gpu, (&k.mla, ik), md, m, a, s, cache, pick, y)?;
                    mla::out_rows(cx, acts, (sites, md), (l, m), &mut bufs, &mut kit)
                }
                _ => Err(GpuError::shape(
                    ROWS,
                    format!(
                        "layer {l}: a pass's selection inputs and the load's indexer disagree \
                         (a load that selects runs every pass row over its own list)"
                    ),
                )),
            }
        }
    }
}

impl Body {
    /// The pass at positions `pos .. pos + ids.len()` of the sequence the stores stand at:
    /// `ids.len()` is 1 to [`PASS_ROWS`], each row the one-token step's bits, the stores and
    /// caches left as that many steps leave them and the position book (the next step's) on
    /// `pos + ids.len()`. Asynchronous as [`Body::walk`] is, with the same fault words. `tail`
    /// says where the last layer's output goes.
    ///
    /// # Errors
    /// [`GpuError`], before anything is enqueued, for no ids or more than [`PASS_ROWS`], a `pos`
    /// the stores do not stand at, rows past the caches, taps armed (a pass does not tap),
    /// a [`Tail::Head`] of another row count than the pass's or a [`Tail::Last`] of more than one
    /// row, and a `pass` made for another model; from the launches otherwise.
    pub fn rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        pass: &mut Pass,
        (ids, pos): (&[u32], u32),
        tail: Tail<'_>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        if !(1..=PASS_ROWS).contains(&m) {
            return Err(GpuError::shape(
                ROWS,
                format!("a pass of {m} rows; a pass runs 1 to {PASS_ROWS}"),
            ));
        }
        let last = launch_u32(ROWS, "rows", m)?;
        let end = pos.checked_add(last).filter(|&e| e as usize <= self.ctx);
        let Some(end) = end else {
            return Err(GpuError::shape(
                ROWS,
                format!(
                    "a pass of {m} rows at position {pos} in caches of {}",
                    self.ctx
                ),
            ));
        };
        if pos != self.held {
            return Err(GpuError::shape(
                ROWS,
                format!(
                    "a pass at position {pos}; the stores stand at {} (a call that failed \
                     part-way: reset)",
                    self.held
                ),
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::shape(
                ROWS,
                "taps are armed: a pass of rows does not tap",
            ));
        }
        if pass.n != self.hp.n_embd {
            return Err(GpuError::shape(
                ROWS,
                format!(
                    "a pass made for rows of {}, the body's are {}",
                    pass.n, self.hp.n_embd
                ),
            ));
        }
        match &tail {
            Tail::Head(h) if h.m() != m => {
                return Err(GpuError::shape(
                    ROWS,
                    format!("a head of {} rows under a pass of {m}", h.m()),
                ));
            }
            Tail::Last(h) if h.m() != 1 => {
                return Err(GpuError::shape(
                    ROWS,
                    format!("a head of {} rows on the last row alone", h.m()),
                ));
            }
            _ => {}
        }
        pass.input.stage(gpu.stream(), ids, pos)?;
        if let Some(sel) = self.s.mla.sel.as_ref() {
            pass.input
                .stage_words(gpu.stream(), (pos, m), sel.top_k())?;
        }
        self.walk_rows(gpu, w, pass, m, tail)?;
        self.held = end;
        Ok(())
    }

    /// The pass's launches in the step's order ([`Body::walk`]): the embedding, every layer, the
    /// final mix, the head.
    fn walk_rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        pass: &mut Pass,
        m: usize,
        tail: Tail<'_>,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let Body {
            hp,
            kda: kd,
            mla: md,
            act,
            layers,
            k,
            acts,
            lane,
            router,
            ctx,
            out_score,
            s: step,
            ..
        } = self;
        let Pass {
            s,
            banks,
            stage,
            rk,
            ffn: pass_ffn,
            input,
            iq,
            iw,
            ..
        } = pass;
        let n = hp.n_embd;
        let eps = hp.rms_eps;
        let scale = hp.experts.weights_scale;
        let stride = PASS_ROWS * n;
        let mut cx = step_cx(gpu, w, k, hp, *ctx);
        cx.what = ROWS;
        let Scratch {
            sum,
            mixed,
            update,
            xn,
            ffn_xn,
            kda: kb,
            mla: mb,
        } = s;
        let [pa, pb] = sum;
        embed_rows(gpu, w, (input, m), pa)?;
        let (pos, keys, words) = (&input.pos, &input.keys, &input.words);
        let mut pushed = 0usize;
        for (l, layer) in layers.iter_mut().enumerate() {
            let fault = gpu.layer_sink(l)?;
            let [attn_score, ffn_score] = &layer.scores;
            // 1. The mixer's input: the mix over the banks, or the sums themselves with no bank
            // yet.
            let input: &DeviceBuffer<f32> = if pushed > 0 {
                k.attn_res.enqueue(
                    stream,
                    AttnResArgs {
                        banks: &*banks,
                        n_banks: pushed,
                        stride,
                        cur: &*pa,
                        score: w.f32_buf(ROWS, attn_score)?,
                        eps,
                        n,
                        m,
                        fault,
                    },
                    mixed,
                )?;
                &*mixed
            } else {
                &*pa
            };
            // 2. A banking layer banks every row's raw sum.
            if layer.banks {
                let mut slot = WindowMut::<f32>::of_mut(&mut *banks, pushed * stride * 4, m * n)?;
                copy_row(stream, &*pa, &mut slot, m * n)?;
                pushed += 1;
            }
            // 3. The mixer's rows. A banking layer's updates are the next sums: they land in the
            // other sum buffer.
            let out: &mut DeviceBuffer<f32> = if layer.banks { &mut *pb } else { &mut *update };
            let io = MixerIo {
                input,
                pos,
                keys,
                xn: &mut *xn,
                kb: &mut *kb,
                mb: &mut *mb,
                out,
            };
            run_mixer_rows(
                &cx,
                (k, &mut *acts, &*lane),
                RowsKit {
                    linear: &k.linear,
                    rows: &*rk,
                    stage: &mut *stage,
                },
                (l, m, &mut layer.mixer),
                (kd, md),
                io,
                step.mla.sel.as_mut().map(|sel| PassSel {
                    iq: &mut *iq,
                    iw: &mut *iw,
                    words,
                    sel,
                }),
            )?;
            // 4. The sums after the mixer: its updates alone on a banking layer, else the sums
            // plus them.
            if !layer.banks {
                gpu.elem().enqueue_add(stream, &*pa, &*update, m * n, pb)?;
            }
            // 5. The block: the mix of the banks and the sums, the layer's norm, the block.
            k.attn_res.enqueue(
                stream,
                AttnResArgs {
                    banks: &*banks,
                    n_banks: pushed,
                    stride,
                    cur: &*pb,
                    score: w.f32_buf(ROWS, ffn_score)?,
                    eps,
                    n,
                    m,
                    fault,
                },
                mixed,
            )?;
            gpu.elem().enqueue_rms_norm(
                stream,
                &*mixed,
                w.f32_buf(ROWS, &layer.ffn_norm)?,
                eps,
                n,
                m,
                ffn_xn,
            )?;
            if m == 1 {
                router.run(&cx, acts, (*act, scale), (l, &layer.ffn), &*ffn_xn, update)?;
            } else {
                pass_ffn.run(
                    router,
                    &cx,
                    acts,
                    (*act, scale),
                    (l, &layer.ffn),
                    (m, &*ffn_xn),
                    update,
                    RowsKit {
                        linear: &k.linear,
                        rows: &*rk,
                        stage: &mut *stage,
                    },
                )?;
            }
            gpu.elem().enqueue_add(stream, &*pb, &*update, m * n, pa)?;
        }
        let mix_into = |y: &mut DeviceBuffer<f32>| {
            k.attn_res.enqueue(
                stream,
                AttnResArgs {
                    banks: &*banks,
                    n_banks: pushed,
                    stride,
                    cur: &*pa,
                    score: out_score
                        .as_ref()
                        .unwrap_or(w)
                        .f32_buf(ROWS, names::OUTPUT_RES_SCORE)?,
                    eps,
                    n,
                    m,
                    fault: gpu.fault_sink(LAYER_HEAD),
                },
                y,
            )
        };
        match tail {
            Tail::Head(head) => {
                mix_into(head.input_mut())?;
                head.enqueue(gpu, w)
            }
            Tail::Last(head) => {
                mix_into(mixed)?;
                let last = Window::<f32>::of(&*mixed, (m - 1) * n * 4, n)?;
                copy_row(stream, &last, head.input_mut(), n)?;
                head.enqueue(gpu, w)
            }
            Tail::Open => Ok(()),
        }
    }

    /// The prompt's chunks of up to [`PASS_ROWS`] ids from `pos`, in order, each a pass; the head
    /// on the last row alone. Returns `true`: the call's last pass enqueues the head.
    fn prompt_walk(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        pass: &mut Pass,
        head: &mut Head,
        (ids, pos): (&[u32], u32),
    ) -> Result<bool, GpuError> {
        let n = ids.len();
        for (c, chunk) in ids.chunks(PASS_ROWS).enumerate() {
            let at = c * PASS_ROWS;
            let tail = if at + chunk.len() == n {
                Tail::Last(&mut *head)
            } else {
                Tail::Open
            };
            let first = pos + launch_u32(ROWS, "row", at)?;
            self.rows(gpu, w, pass, (chunk, first), tail)?;
        }
        Ok(true)
    }
}

/// The embedding of the pass's `m` staged ids into `y`, and the positions and live key counts of
/// the rows.
fn embed_rows(
    gpu: &Gpu,
    w: &Weights,
    (input, m): (&mut PassIn, usize),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let embd = names::token_embd();
    let Some(DevWeight::KQuant { ty, w: table, .. }) = w.get(&embd) else {
        return Err(GpuError::tensor(ROWS, &embd, "K-quant rows"));
    };
    let PassIn { buf, pos, keys, .. } = input;
    let ids = Window::<u32>::of(&*buf, 0, m)?;
    let pos0 = Window::<u32>::of(&*buf, PASS_ROWS * 4, 1)?;
    gpu.elem().enqueue_embed_rows_kquant(
        gpu.stream(),
        *ty,
        EmbedRowsArgs {
            w: table,
            ids: &ids,
            pos0: &pos0,
            first: 0,
            y,
            pos,
            n_keys: keys,
        },
    )
}

impl GpuModel<Body> {
    /// Run `ids` from the model's position as passes of up to [`PASS_ROWS`] ids and return the
    /// argmax after the last: every row the one-token step's bits, the head on the last row
    /// alone. Refused by name with no ids, while a verify waits and as [`Body::rows`] refuses.
    ///
    /// # Errors
    /// [`GpuError`] as above, and the head's fault.
    pub fn prompt_rows(&mut self, pass: &mut Pass, ids: &[u32]) -> Result<u32, GpuError> {
        const WHAT: &str = "kimi_k3 GpuModel::prompt_rows";
        self.run_rows(ids.len(), WHAT, |gpu, w, body, head, pos| {
            body.prompt_walk(gpu, w, pass, head, (ids, pos))
        })?
        .ok_or(GpuError::state(
            WHAT,
            "the head's token of the call's last row",
        ))
    }
}
