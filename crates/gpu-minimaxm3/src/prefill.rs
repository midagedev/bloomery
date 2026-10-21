//! The prompt batch: a prompt of `P` ids fed in units of at most [`T_MAX`]
//! positions ([`prefill`]), which leaves the model where `P` decode steps over
//! the same ids leave it in everything a later step reads: every layer's key
//! and value rows at those positions, every sparse layer's index keys, and the
//! last position's logits.
//!
//! The walk is the common owner's ([`prompt_walk`]): the units of a group, the
//! host staging of their rows, the routed block over the unit's slots, the
//! head and the fault word. This file holds what MiniMax-M3's architecture
//! decides: the attention over the unit's rows (the per-head q and k norms,
//! the partial NEOX rope, the indexer and its block-max selection) and the
//! buffers its launches read.
//!
//! A sparse layer's index keys are the unit's rows' first: every row's index
//! key is appended before any row is scored, so a row's score sees the keys of
//! the rows before it and none after it (its live count is its position plus
//! one, as the step's is). The dense layers attend every position; the sparse
//! ones the blocks their indexer picks.

use bloomery_gpu::arch::hybrid::WideCx;
use bloomery_gpu::flash_gqa::{GROUP_16, HEAD};
use bloomery_gpu::gemm::{GemmKernels, GemmRoute};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::rope_neox::PartialNeoxArgs;
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::site::{WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::prompt_walk::{
    self, Batch as Walk, BatchAttention, BatchBufs, UnitRows,
};
use cuda_core::DeviceBuffer;
use model::embed::EmbedTable;
use runtime::layer::{FfnKind, Layer};

use crate::attn::{
    AttnArgs, IndexArgs, MsaScratch, MsaShape, PickArgs, ScoreArgs, attn_scale, blocks_for,
    enqueue_attention, enqueue_index, partials_lens,
};
use crate::body::{Body, Dims, Store};
use crate::ffn::Routed;
use crate::step::{FLASH_MMA, Kernels};

/// What this file's refusals name.
const WHAT: &str = "minimaxm3 prefill";

/// The most positions a unit of the prompt batch holds.
///
/// The plan's scratch term (67,108,864 B on the gate card's plan) holds the
/// step's scratch and the unit's rows: a row's scratch is 2,932,000 B, of
/// which the dense layers' flash partials are 2,621,440 B (the decode flash
/// cuts every row's keys into 80 segments a head, so a row's partials are
/// `64 · 80 · 128` f32). Sixteen rows take 46,912,000 B and the step's
/// scratch the rest of the term; thirty-two rows do not fit beside it.
pub const T_MAX: usize = 16;

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// The prompt batch's rows: the normed rows, the projections' rows, the
/// flash's output and partials, the index heads' rows and the selection's
/// scratch, at [`T_MAX`] rows.
pub struct BatchRows {
    pub xn: DeviceBuffer<f32>,
    /// A sub-layer's output before its residual add.
    pub delta: DeviceBuffer<f32>,
    /// The query projection, its normed and turned heads, the key and value rows, the flash's output.
    pub q_raw: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub attn: DeviceBuffer<f32>,
    /// The flash's partials: the decode flash's over every position, the listed flash's over the
    /// picked blocks.
    pub part_v: DeviceBuffer<f32>,
    pub part_ms: DeviceBuffer<f32>,
    /// The index heads' projection, their normed and turned heads, the index key row.
    pub iq_raw: DeviceBuffer<f32>,
    pub iq: DeviceBuffer<f32>,
    pub ik: DeviceBuffer<f32>,
    /// The value row the index head kernel writes where nothing reads it.
    pub zeros: DeviceBuffer<f32>,
}

impl BatchRows {
    /// `rows` rows of every width the attention reads, over a cache of `ctx` positions. Load-time
    /// only.
    fn new(
        stream: &cuda_core::CudaStream,
        (d, shape): (&Dims, &MsaShape),
        rows: usize,
    ) -> Result<BatchRows, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, rows * n);
        let (q_width, kv_width) = (d.heads * HEAD, d.kv_heads * HEAD);
        let (msa_v, msa_ms) = partials_lens(shape, rows);
        Ok(BatchRows {
            xn: z(d.embd)?,
            delta: z(d.embd)?,
            q_raw: z(q_width)?,
            q: z(q_width)?,
            k: z(kv_width)?,
            v: z(kv_width)?,
            attn: z(q_width)?,
            // The flash's partials are the rows' own (the length is over the rows already).
            part_v: DeviceBuffer::zeroed(
                stream,
                bloomery_gpu::flash_gqa::partials_v_len(rows, d.heads).max(msa_v),
            )?,
            part_ms: DeviceBuffer::zeroed(
                stream,
                bloomery_gpu::flash_gqa::partials_ms_len(rows, d.heads).max(msa_ms),
            )?,
            iq_raw: z(crate::attn::HEADS * crate::attn::DIM)?,
            iq: z(crate::attn::HEADS * crate::attn::DIM)?,
            ik: z(crate::attn::DIM)?,
            zeros: z(crate::attn::DIM)?,
        })
    }

    /// Device bytes of the rows and the selection's scratch.
    #[must_use]
    pub fn bytes(&self) -> usize {
        [
            &self.xn,
            &self.delta,
            &self.q_raw,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.part_v,
            &self.part_ms,
            &self.iq_raw,
            &self.iq,
            &self.ik,
            &self.zeros,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
    }
}

/// The prompt batch's buffers: the rows' own, the wide forms the projections
/// read, the dense one-expert table the wide GEMMs route on and the grouped
/// GEMM's kernels. Made once, so that a timed prompt allocates nothing.
pub struct Batch {
    pub rows: BatchRows,
    /// The selection's block scores, kept blocks and key counts.
    pub msa: MsaScratch,
    /// The wide forms at the model width: the q, k and v projections read them.
    pub wide: WideActs,
    /// The wide forms at the output projection's width.
    pub arm: WideActs,
    /// The dense one-expert table, filled for the unit's rows.
    pub route: GemmRoute,
    pub gemm: GemmKernels,
}

impl Batch {
    /// The buffers of a batch of `rows` rows a unit holds. Load-time only.
    pub fn new(gpu: &Gpu, (d, shape): (&Dims, &MsaShape), rows: usize) -> Result<Batch, GpuError> {
        let stream = gpu.stream();
        let q_width = d.heads * HEAD;
        Ok(Batch {
            rows: BatchRows::new(stream, (d, shape), rows)?,
            msa: MsaScratch::new(stream, shape, rows, d.ctx)?,
            wide: WideActs::new(stream, (false, true), rows, d.embd)?,
            arm: WideActs::new(stream, (false, true), rows, q_width)?,
            route: GemmRoute::new(stream, rows, 1)?,
            gemm: GemmKernels::load(gpu.context())?,
        })
    }

    /// Device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.bytes()
            + self.msa.scores.num_bytes()
            + self.msa.ids.num_bytes()
            + self.msa.n_sel.num_bytes()
            + self.wide.bytes()
            + self.arm.bytes()
            + self.route.bytes()
    }
}

/// Set the prompt group to `g` units a walk: a group's units run their layers
/// together, so the card work of one unit overlaps the previous one's host
/// serve; 1 runs each unit alone, and every `g` writes the same bits.
/// Refused by name for a `g` outside 1 to the lever's most. Returns whether it
/// made any.
pub fn set_prefill_group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    prompt_walk::check_group(g, prompt_walk::GROUP_MAX).map_err(|r| GpuError::Shape {
        what: prompt_walk::GROUP_LEVER,
        detail: r.to_string(),
    })?;
    body.units.set_group(gpu, body.dims.embd, g)
}

/// Feed `ids` from position `at` in units of up to [`T_MAX`] positions and
/// enqueue the head after the last: the pass
/// [`GpuModel::run_rows`](bloomery_gpu::GpuModel::run_rows) runs, returning
/// whether the head was enqueued (always, on success). Refused by name
/// before anything runs: for no id and past the stores' positions.
pub fn prefill(
    gpu: &Gpu,
    w: &Weights,
    body: &mut Body,
    head: &mut Head,
    ids: &[u32],
    at: u32,
) -> Result<bool, GpuError> {
    let p = parts(body);
    let mut attention = Attention {
        gpu,
        w,
        dims: p.dims,
        cfg: p.cfg,
        stores: p.stores,
        k: p.k,
        rope: p.rope,
        shape: p.shape,
        batch: p.batch,
        sink: p.sink,
    };
    let mut batch = Walk {
        gpu,
        w,
        bufs: p.units,
        attention: &mut attention,
        ffn: p.ffn,
        hybrid: p.hybrid,
        embed: p.embed,
        head,
        taps: p.taps,
        embd: p.dims.embd,
        ctx: p.dims.ctx,
        layers: p.cfg.len(),
    };
    let mut at = at;
    batch.prompt(ids, &mut at)?;
    *p.at = (*p.at).max(at);
    Ok(true)
}

/// The body's fields the batch walks, split into the disjoint borrows the walk
/// holds at once.
struct Parts<'a> {
    hybrid: &'a mut Hybrid<HostRun>,
    dims: &'a Dims,
    cfg: &'a [Layer],
    stores: &'a mut [Store],
    batch: &'a mut Batch,
    /// The units of a group: the walker's own buffers.
    units: &'a mut BatchBufs,
    k: &'a Kernels,
    rope: &'a RopeRows,
    shape: &'a MsaShape,
    /// The plane the index head kernel writes where nothing reads it.
    sink: &'a mut DeviceBuffer<u16>,
    ffn: &'a mut Routed,
    at: &'a mut u32,
    embed: &'a EmbedTable,
    taps: Option<&'a mut [DeviceBuffer<f32>]>,
}

/// `body` as the walk's [`Parts`].
fn parts(body: &mut Body) -> Parts<'_> {
    let Body {
        shell,
        dims,
        cfg,
        stores,
        s,
        batch,
        units,
        k,
        rope,
        shape,
        ffn,
        at,
        embd,
        ..
    } = body;
    Parts {
        taps: shell.taps.as_deref_mut(),
        hybrid: &mut shell.hybrid,
        dims,
        cfg,
        stores,
        batch,
        units,
        k,
        rope,
        shape,
        sink: &mut s.sink,
        ffn,
        at,
        embed: &embd.table,
    }
}

/// The attention over a unit's rows: what MiniMax-M3's architecture decides,
/// as the walk's attention.
struct Attention<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    dims: &'a Dims,
    cfg: &'a [Layer],
    stores: &'a mut [Store],
    k: &'a Kernels,
    rope: &'a RopeRows,
    shape: &'a MsaShape,
    batch: &'a mut Batch,
    sink: &'a mut DeviceBuffer<u16>,
}

impl Attention<'_> {
    /// The projections' launch context over `m` rows.
    fn cx<'a>(
        gpu: &'a Gpu,
        w: &'a Weights,
        gemm: &'a GemmKernels,
        g32: &'a bloomery_gpu::gemm::Gemm32Kernels,
        m: usize,
    ) -> WideCx<'a> {
        WideCx {
            what: WHAT,
            gpu,
            w,
            wk: WideKernels {
                gemm,
                g32: Some(g32),
            },
            m,
        }
    }

    /// The unit's rows through the layer's attention: the norm, the q, k and v
    /// projections, the index heads' on a sparse layer, the head kernel, the
    /// flash and the output projection, then the residual add.
    fn attend(&mut self, l: usize, m: usize, rows: UnitRows<'_>) -> Result<(), GpuError> {
        let d = *self.dims;
        let stream = self.gpu.stream();
        let fault = self.gpu.layer_sink(l)?;
        let Some(store) = self.stores.get(l) else {
            return Err(shape(format!("layer {l}: no store")));
        };
        let q_width = d.heads * HEAD;
        let kv_width = d.kv_heads * HEAD;
        // The dense one-expert table the wide GEMMs route on, filled for the unit's rows.
        self.batch.gemm.enqueue_route_dense(
            stream,
            m,
            &mut self.batch.route,
            self.gpu.unlabelled_sink(),
        )?;
        self.gpu.elem().enqueue_rms_norm(
            stream,
            rows.x,
            self.w.f32_buf(WHAT, &store.gqa.norm)?,
            d.rms_eps,
            d.embd,
            m,
            &mut self.batch.rows.xn,
        )?;
        let cx = Self::cx(self.gpu, self.w, &self.batch.gemm, &self.k.g32, m);
        bloomery_gpu::arch::hybrid::project_wide(
            &cx,
            (&mut self.batch.wide, &self.batch.route),
            l,
            &self.batch.rows.xn,
            &mut [
                (&store.gqa.q, q_width, &mut self.batch.rows.q_raw),
                (&store.gqa.k, kv_width, &mut self.batch.rows.k),
                (&store.gqa.v, kv_width, &mut self.batch.rows.v),
            ],
        )?;
        if let Some(ix) = store.index.as_ref() {
            // The index heads' projections: the query over the indexer's heads and the key over
            // its width, both over the normed rows.
            bloomery_gpu::arch::hybrid::project_wide(
                &cx,
                (&mut self.batch.wide, &self.batch.route),
                l,
                &self.batch.rows.xn,
                &mut [
                    (
                        &ix.q,
                        crate::attn::HEADS * crate::attn::DIM,
                        &mut self.batch.rows.iq_raw,
                    ),
                    (&ix.k, crate::attn::DIM, &mut self.batch.rows.ik),
                ],
            )?;
        }
        if let Some(ix) = self.stores[l].index.as_mut() {
            // The index heads' rows: the norms, the turned values and the index key, appended at
            // the rows' positions. Every row's key is appended before any row is scored.
            let gain_q = self.w.f32_buf(WHAT, &ix.q_norm)?;
            let gain_k = self.w.f32_buf(WHAT, &ix.k_norm)?;
            let table = &self.rope.table;
            let BatchRows {
                iq_raw,
                iq,
                ik,
                zeros,
                ..
            } = &mut self.batch.rows;
            // SAFETY: the unit's positions are distinct (the head kernel's contract).
            unsafe {
                enqueue_index(
                    &self.k.rope,
                    stream,
                    IndexArgs {
                        proj_q: iq_raw,
                        iq,
                        proj_k: ik,
                        zeros,
                        gain_q,
                        gain_k,
                        table,
                        pos: rows.pos,
                        eps: d.rms_eps,
                        ctx: d.ctx,
                        m,
                        fault,
                        keys: &mut ix.keys,
                        sink: &mut *self.sink,
                    },
                )?;
            }
            self.k.msa.enqueue_score(
                stream,
                self.shape,
                ScoreArgs {
                    iq: &self.batch.rows.iq,
                    ik: &ix.keys,
                    n_keys: rows.n_keys,
                    ctx: d.ctx,
                    m,
                    blocks: blocks_for(d.ctx),
                    fault,
                },
                &mut self.batch.msa,
            )?;
            self.k.msa.enqueue_pick(
                stream,
                self.shape,
                PickArgs {
                    n_keys: rows.n_keys,
                    ctx: d.ctx,
                    m,
                    fault,
                },
                &mut self.batch.msa,
            )?;
        }
        {
            let store = &mut self.stores[l];
            let gq = self.w.f32_buf(WHAT, &store.q_norm)?;
            let gk = self.w.f32_buf(WHAT, &store.k_norm)?;
            let table = &self.rope.table;
            let BatchRows {
                q_raw,
                q,
                k,
                v,
                part_v,
                part_ms,
                attn,
                ..
            } = &mut self.batch.rows;
            // SAFETY: the unit's positions are distinct (the head kernel's contract).
            unsafe {
                self.k.rope.enqueue_head_norm_neox_append_128_partial(
                    stream,
                    PartialNeoxArgs {
                        qg: q_raw,
                        q,
                        k,
                        v,
                        gq,
                        gk,
                        table,
                        pos: rows.pos,
                        eps: d.rms_eps,
                        n_head: d.heads,
                        n_kv: d.kv_heads,
                        ctx: d.ctx,
                        m,
                        fault,
                        cache_k: store.gqa.kc.buf_mut(),
                        cache_v: store.gqa.vc.buf_mut(),
                    },
                )?;
            }
            let store = &*store;
            if store.index.is_some() {
                enqueue_attention(
                    &self.k.flash,
                    stream,
                    self.shape,
                    &self.batch.msa,
                    AttnArgs {
                        q,
                        kc: store.gqa.kc.buf(),
                        vc: store.gqa.vc.buf(),
                        ctx: d.ctx,
                        m,
                        part_v,
                        part_ms,
                        fault,
                        y: attn,
                    },
                )?;
            } else {
                self.k.flash.enqueue_pass::<GROUP_16>(
                    stream,
                    bloomery_gpu::flash_gqa::GqaArgs {
                        q,
                        kc: store.gqa.kc.buf(),
                        vc: store.gqa.vc.buf(),
                        n_keys: rows.n_keys,
                        scale: attn_scale(),
                        n_kv: d.kv_heads,
                        ctx: d.ctx,
                        m,
                        part_v,
                        part_ms,
                        fault,
                        y: attn,
                    },
                    FLASH_MMA,
                )?;
            }
        }
        let store = &self.stores[l];
        let cx = Self::cx(self.gpu, self.w, &self.batch.gemm, &self.k.g32, m);
        bloomery_gpu::arch::hybrid::project_wide(
            &cx,
            (&mut self.batch.arm, &self.batch.route),
            l,
            &self.batch.rows.attn,
            &mut [(&store.gqa.o, d.embd, &mut self.batch.rows.delta)],
        )?;
        self.gpu
            .elem()
            .enqueue_add(stream, rows.x, &self.batch.rows.delta, m * d.embd, rows.x1)
    }
}

impl BatchAttention for Attention<'_> {
    /// The attention over the unit's rows.
    fn attend(
        &mut self,
        l: usize,
        _u: usize,
        m: usize,
        rows: UnitRows<'_>,
    ) -> Result<(), GpuError> {
        self.attend(l, m, rows)
    }

    /// The layer's feed-forward, as its description names it.
    fn ffn(&self, l: usize) -> FfnKind {
        self.cfg[l].ffn
    }
}
