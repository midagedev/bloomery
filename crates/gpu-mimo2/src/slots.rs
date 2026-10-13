//! The pass of several slots' rows ([`SlotRows`], `GpuModel::step_slots`): the
//! step's layer program over `R` columns in one walk, `(1, R, Step)` through
//! the step port's `Cols(R)` chain — the template is the Qwen3.8 body's
//! pass, not the pair walk. Per layer:
//! - the row-wise launches once over all `R` rows: the norms, the q·k·v and
//!   output projections (`q8_0_gemv_mcol`, each column the one-column gemv's
//!   bits), the dense block (the common gated block over the columns: a q8_0
//!   block in one launch pair, a K-quant block a column at a time), the
//!   router over the rows (`enqueue_router_rows`, what `R` router launches
//!   write) and the adds;
//! - the launches bound to one sequence's stores once per busy slot, over
//!   that slot's rows and stores: the rope-and-append and the K192 flash with
//!   its merge ([`SEQ_LAUNCHES`]) — K192 with sinks and a window has no row
//!   table, so one slot's launches cannot serve another's stores;
//! - the host tier through one union call: one handoff launch a column into
//!   the boundary's row of `R` columns, one go, one wait, the card experts
//!   over all `R` rows in the host leg's shadow.
//!
//! A pass of one row is the step as it runs: the step's walk and launches
//! over the slot's own stores ([`Body::parts_on`]). A pass of several rows
//! needs the load planned for several sequences, which made its buffers
//! ([`PassBufs`]).
//!
//! The embedding is a host row (module doc of [`super`]): [`Body::plan`]
//! writes each row's embedding into its column of `x` and each row's position
//! and live key count, before the pass.

use std::ops::Range;

use bloomery_gpu::flash_gqa::{GqaK192Args, HEAD, HEAD_K192};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::StepLeg;
use bloomery_gpu::host::handoff::Handoff;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::{Chain, HandoffTarget};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::model::{ChainBody, RowHeads, SlotRange, SlotRows};
use bloomery_gpu::rope_neox::K192Args;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, DeviceTensor, Gpu, GpuError};
use bloomery_gpu_deepseek41::block::{Block, BlockIo, BlockKernels, BlockPlan, enqueue_block_cols};
use bloomery_gpu_deepseek41::router::mimo2::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::mimo2::program as facts;
use runtime::layer::FfnKind;
use runtime::prompt::chunks;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

use super::prefill::FLASH_ROWS;
use super::{
    Body, Dims, Kernels, LayerCfg, LayerNames, RopeBase, Scratch, Seq, Store, WHAT, shape,
};
use crate::attn::{enqueue_append, enqueue_pass};
use crate::card::{CardExperts, Io};
use crate::ffn::FfnNames;
use crate::program::walk_step;

/// A shape refused by name in `what`.
fn shape_in(what: &'static str, detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what,
        detail: detail.into(),
    }
}

/// State `missing` refused by name in `what`.
fn state_in(what: &'static str, missing: &'static str) -> GpuError {
    GpuError::State { what, missing }
}

/// The most rows one pass takes: the seat's default slots, one row each. The
/// projections run in one column group and the flash of a slot in one launch,
/// so a pass is bound by both.
pub(crate) const MAX_ROWS: usize = 2;
const _: () = assert!(MAX_ROWS <= COL_GROUP && MAX_ROWS <= FLASH_ROWS);

/// The launches of an attention sub-layer bound to one sequence's stores,
/// whatever the rows of its slot: the rope-and-append, the flash's segment
/// pass and its merge. The other four of [`facts::ATTN_LAUNCHES`] are
/// row-wise.
pub(crate) const SEQ_LAUNCHES: usize = 3;
const _: () = assert!(SEQ_LAUNCHES < facts::ATTN_LAUNCHES);

/// What the pass's refusals name.
const PASS: &str = "mimo2 Body::enqueue_slots";

/// The buffers of a pass of [`MAX_ROWS`] rows, beside the step's: the step's
/// scratch widened to the pass's rows (the flash's partials among them, shared
/// by the slots' launches one after another), the routed block's normed rows
/// and the router's outputs for every row, and the host staging of a plan.
pub(crate) struct PassBufs {
    pub s: Scratch,
    /// A routed layer's normed rows: the handoffs' and the card experts' input.
    normed: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The plan's host values: the rows' embeddings, positions and live key
    /// counts.
    host: Staging,
}

/// A plan's host values for a pass.
struct Staging {
    x: Vec<f32>,
    pos: Vec<u32>,
    keys: Vec<u32>,
}

impl PassBufs {
    /// The buffers of a pass of `rows` rows. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        d: &Dims,
        widths: (usize, usize),
        blocks: impl IntoIterator<Item = (BlockPlan, usize)>,
        rows: usize,
    ) -> Result<PassBufs, GpuError> {
        Ok(PassBufs {
            s: Scratch::new(stream, d, widths, blocks, rows)?,
            normed: DeviceBuffer::zeroed(stream, rows * d.embd)?,
            probs: DeviceBuffer::zeroed(stream, rows * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, rows * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, rows * N_USED)?,
            host: Staging {
                x: vec![0.0; rows * d.embd],
                pos: vec![0; rows],
                keys: vec![0; rows],
            },
        })
    }

    /// Device bytes of the buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.s.bytes()
            + self.normed.num_bytes()
            + self.probs.num_bytes()
            + self.ids.num_bytes()
            + self.weights.num_bytes()
    }
}

/// The rows of a pass of `rows`, once they tile `0..R` in order with no empty
/// range; else refused by name.
fn pass_rows(rows: &[SlotRange]) -> Result<usize, GpuError> {
    let mut at = 0;
    for r in rows {
        if r.rows.start != at || r.rows.is_empty() {
            return Err(shape_in(
                PASS,
                format!(
                    "slot {}'s rows {:?}; the ranges tile the pass in order from row 0",
                    r.slot, r.rows
                ),
            ));
        }
        at = r.rows.end;
    }
    if at == 0 {
        return Err(shape_in(PASS, "a pass of no row"));
    }
    Ok(at)
}

impl SlotRows for Body {
    const MAX_ROWS: usize = MAX_ROWS;
    const HEADS: RowHeads = RowHeads::One;

    /// The step's chain at one row, the step port's columns at more.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        match rows.last().map_or(0, |r| r.rows.end) {
            1 => Chain::Step,
            m => Chain::Cols(m),
        }
    }

    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        _parked: &mut [&mut Seq],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        self.plan(stream, rows, ids)
    }

    /// The pass into `heads[0]`, a head of its rows; any other heads are
    /// refused by name.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Seq],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        match heads {
            [head] => self.enqueue_pass(gpu, w, head, parked, rows),
            _ => Err(shape_in(
                PASS,
                format!(
                    "{} heads; a pass of several slots runs into one head of its rows",
                    heads.len()
                ),
            )),
        }
    }
}

impl Body {
    /// Plan the pass of `rows` over `ids` (every row's token): each row's
    /// embedding, position and live key count into the buffers the pass reads
    /// — a pass of one row exactly as a step's input
    /// ([`ChainBody::decode_input`], [`ChainBody::refresh`]). Refused by name
    /// before anything moves: the layer taps armed, a route trace attached, a
    /// pass of several rows on a load planned for one sequence, a row at a
    /// position past the stores, an id past the vocabulary.
    fn plan(
        &mut self,
        stream: &CudaStream,
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        let total = pass_rows(rows)?;
        if ids.len() != total {
            return Err(shape_in(
                PASS,
                format!("{} ids for a pass of {total} rows", ids.len()),
            ));
        }
        if self.taps.is_some() {
            return Err(state_in(
                PASS,
                "layer taps off (a pass of several slots writes none)",
            ));
        }
        if self.hybrid.route_traced() {
            return Err(shape_in(
                PASS,
                "a pass of several slots with a route trace attached: the trace records one \
                 sequence's positions",
            ));
        }
        self.pending = None;
        if total == 1 {
            let input = self.decode_input(ids[0], rows[0].pos0)?;
            if rows[0].slot != 0 {
                // Another slot's row. The live sequence does not stand past it.
                self.pending = None;
            }
            return self.refresh(stream, &input);
        }
        let Body {
            embd,
            pass,
            ctx,
            dims,
            ..
        } = self;
        let bufs = pass.as_mut().ok_or_else(|| {
            shape_in(
                PASS,
                format!("a pass of {total} rows on a load planned for one sequence"),
            )
        })?;
        let n = dims.embd;
        for r in rows {
            let last = r.pos0 as usize + r.rows.len();
            if last > *ctx {
                return Err(shape_in(
                    PASS,
                    format!(
                        "slot {}'s rows {:?} from position {} in stores of {ctx}",
                        r.slot, r.rows, r.pos0
                    ),
                ));
            }
            for (i, row) in r.rows.clone().enumerate() {
                let pos = r.pos0 + u32::try_from(i).map_err(|_| shape("a slot's row past u32"))?;
                bufs.host.pos[row] = pos;
                bufs.host.keys[row] = pos + 1;
                embd.fill_into(ids[row], &mut bufs.host.x[row * n..][..n])?;
            }
        }
        let s = &mut bufs.s;
        span_mut(PASS, &mut s.x, 0, total * n)?
            .copy_from_host(stream, &bufs.host.x[..total * n])?;
        span_mut(PASS, &mut s.pos, 0, total)?.copy_from_host(stream, &bufs.host.pos[..total])?;
        span_mut(PASS, &mut s.n_keys, 0, total)?
            .copy_from_host(stream, &bufs.host.keys[..total])?;
        Ok(())
    }

    /// Enqueue the pass of `rows` into `head`, a head of its rows. A pass of
    /// one row is the step's walk over its slot's stores; a pass of several
    /// is [`PassProgram`]'s over all of them, each busy slot's
    /// sequence-bound launches over its own rows and stores (`parked` holds
    /// every busy slot's sequence but the live one's, in `rows` order).
    fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        parked: &mut [&mut Seq],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        let total = pass_rows(rows)?;
        if head.m() != total {
            return Err(shape_in(
                PASS,
                format!("a pass of {total} rows into a head of {} rows", head.m()),
            ));
        }
        let mut others = parked.iter_mut();
        let no_home = |slot: usize| {
            shape_in(
                PASS,
                format!("slot {slot}'s parked sequence, or slot 0 twice"),
            )
        };
        if total == 1 {
            let slot = rows[0].slot;
            {
                let stores = match slot {
                    0 => None,
                    _ => Some(
                        others
                            .next()
                            .ok_or_else(|| no_home(slot))?
                            .stores
                            .as_mut_slice(),
                    ),
                };
                let (parts, hybrid) = self.parts_on(stores);
                walk_step(gpu, w, parts, hybrid, head)?;
            }
            if slot == 0 {
                self.commit_pending();
            }
            return Ok(());
        }
        let live_end = rows.iter().find(|r| r.slot == 0).map(|r| {
            r.pos0
                .saturating_add(u32::try_from(r.rows.len()).unwrap_or(u32::MAX))
        });
        {
            let Body {
                hybrid,
                cfg,
                names,
                dims,
                k,
                pass,
                card,
                stores,
                ropes,
                slots,
                ctx,
                ..
            } = self;
            let b = pass.as_mut().ok_or_else(|| {
                shape_in(
                    PASS,
                    format!("a pass of {total} rows on a load planned for one sequence"),
                )
            })?;
            let mut live = Some(stores.as_mut_slice());
            let mut seqs = Vec::with_capacity(rows.len());
            for r in rows {
                let st = match r.slot {
                    0 => live.take().ok_or_else(|| no_home(0))?,
                    s => others
                        .next()
                        .ok_or_else(|| no_home(s))?
                        .stores
                        .as_mut_slice(),
                };
                seqs.push(SlotStores {
                    rows: r.rows.clone(),
                    stores: st,
                });
            }
            let mut prog = PassProgram {
                gpu,
                w,
                k,
                d: dims,
                cfg,
                names,
                b,
                card,
                ropes,
                slots,
                ctx: *ctx,
                seqs,
                head,
                m: total,
            };
            let mut leg = StepLeg::new(gpu.stream(), hybrid);
            prog.walk(&mut leg)?;
        }
        if let Some(end) = live_end {
            self.at = end;
            self.pending = None;
        }
        Ok(())
    }

    /// The launches of the captured pass of `key` (each `(slot, rows)`, in
    /// pass order), derived from the layer table: a pass of one row is the
    /// step's ([`facts::step_launches`]); a pass of `m` rows runs each layer's
    /// row-wise launches once (the attention's four, the dense block's
    /// four, the routed block's six with the router's second launch and a
    /// handoff a column, and the card experts' six) and the
    /// [`SEQ_LAUNCHES`] once a busy slot, then the head's.
    #[must_use]
    pub fn slots_launches(&self, key: &[(usize, usize)]) -> usize {
        let m: usize = key.iter().map(|&(_, rows)| rows).sum();
        let blocks = self.blocks();
        if m <= 1 {
            let extra: usize = self.block_plans().iter().map(|p| p.extra_launches()).sum();
            return facts::step_launches(&blocks) + extra;
        }
        let attn = facts::ATTN_LAUNCHES - SEQ_LAUNCHES + SEQ_LAUNCHES * key.len();
        let layers: usize = blocks
            .iter()
            .zip(&self.cfg)
            .map(|(&(ffn, card), c)| {
                attn + match ffn {
                    FfnKind::Dense => {
                        facts::DENSE_LAUNCHES - 2 + c.block.map_or(2, |p| p.column_launches(m))
                    }
                    FfnKind::Moe => {
                        facts::MOE_LAUNCHES + m + if card { facts::CARD_LAUNCHES } else { 0 }
                    }
                }
            })
            .sum();
        layers + facts::HEAD_LAUNCHES
    }
}

/// One busy slot's rows of a pass and its sequence's stores.
struct SlotStores<'s> {
    rows: Range<usize>,
    stores: &'s mut [Store],
}

/// The MiMo layer program of a pass of `m` rows: the parts of the step's
/// program over the pass's buffers, the busy slots' stores and the head of
/// its rows.
struct PassProgram<'s> {
    gpu: &'s Gpu,
    w: &'s Weights,
    k: &'s Kernels,
    d: &'s Dims,
    cfg: &'s [LayerCfg],
    names: &'s [LayerNames],
    b: &'s mut PassBufs,
    card: &'s mut CardExperts,
    ropes: &'s [RopeBase],
    slots: &'s DeviceTensor<u32>,
    ctx: usize,
    seqs: Vec<SlotStores<'s>>,
    head: &'s mut Head,
    m: usize,
}

impl<'s> PassProgram<'s> {
    /// Walk every layer at `m` columns through `leg`, then the head.
    fn walk(&mut self, leg: &mut StepLeg<'s, HostRun>) -> Result<(), GpuError> {
        let o = Overlap {
            units: 1,
            cols: self.m,
            port: PortKind::Step,
        };
        let layers = self.cfg.len();
        sched::walk(o, layers, leg, self)
    }

    /// Whether layer `l` is the last: its output is the head's input.
    fn last(&self, l: usize) -> bool {
        l + 1 == self.cfg.len()
    }

    /// Layer `l`'s attention over the pass's rows: `x` in, `x1` out.
    fn attention(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "mimo2 pass attention";
        let (gpu, w, m) = (self.gpu, self.w, self.m);
        let stream = gpu.stream();
        let (d, c) = (*self.d, self.cfg[l]);
        let a = c.attn;
        let n = &self.names[l].attn;
        let fault = gpu.layer_sink(l)?;
        let s = &mut self.b.s;
        let width = d.heads * HEAD_K192 + a.kv_heads * (HEAD_K192 + HEAD);
        let (q_row, y_row) = (d.heads * HEAD_K192, d.heads * HEAD);
        gpu.elem().enqueue_rms_norm(
            stream,
            &s.x,
            w.f32_buf(WHAT, &n.norm)?,
            d.rms_eps,
            d.embd,
            m,
            &mut s.xn,
        )?;
        for (c0, cn) in chunks(m, COL_GROUP) {
            let xs = span(W, &s.xn, c0 * d.embd, cn * d.embd)?;
            let mut ys = span_mut(W, &mut s.qkv, c0 * width, cn * width)?;
            w.q8_gemv_mcol(gpu, WHAT, &n.qkv, &xs, cn, &mut ys)?;
        }
        let sinks = n
            .sinks
            .as_deref()
            .map(|name| w.f32_buf(WHAT, name))
            .transpose()?;
        for sq in &mut self.seqs {
            let (r0, mj) = (sq.rows.start, sq.rows.len());
            let store = &mut sq.stores[l];
            let rows = store.rows;
            {
                let mut qkv = span_mut(W, &mut s.qkv, r0 * width, mj * width)?;
                let mut q = span_mut(W, &mut s.q, r0 * q_row, mj * q_row)?;
                let pos = span(W, &s.pos, r0, mj)?;
                enqueue_append(
                    &self.k.rope,
                    stream,
                    K192Args {
                        qkv: &mut qkv,
                        q: &mut q,
                        table: &self.ropes[c.rope].rows.table,
                        pos: &pos,
                        v_scale: a.v_scale,
                        n_head: d.heads,
                        n_kv: a.kv_heads,
                        ctx: self.ctx,
                        m: mj,
                        fault,
                        cache_k: &mut store.k,
                        cache_v: &mut store.v,
                    },
                    rows,
                )?;
            }
            let q = span(W, &s.q, r0 * q_row, mj * q_row)?;
            let keys = span(W, &s.n_keys, r0, mj)?;
            let mut y = span_mut(W, &mut s.y, r0 * y_row, mj * y_row)?;
            enqueue_pass(
                &self.k.flash,
                stream,
                GqaK192Args {
                    q: &q,
                    kc: &store.k,
                    vc: &store.v,
                    n_keys: &keys,
                    scale: d.score_scale,
                    n_kv: a.kv_heads,
                    ctx: self.ctx,
                    m: mj,
                    window: a.window,
                    sinks,
                    part_v: &mut s.part_v,
                    part_ms: &mut s.part_ms,
                    fault,
                    y: &mut y,
                },
                d.heads,
                rows,
            )?;
        }
        for (c0, cn) in chunks(m, COL_GROUP) {
            let ys = span(W, &s.y, c0 * y_row, cn * y_row)?;
            let mut out = span_mut(W, &mut s.out, c0 * d.embd, cn * d.embd)?;
            w.q8_gemv_mcol(gpu, WHAT, &n.o, &ys, cn, &mut out)?;
        }
        gpu.elem()
            .enqueue_add(stream, &s.x, &s.out, m * d.embd, &mut s.x1)
    }

    /// Layer `l`'s dense block over the pass's rows: `x1` in, the layer's
    /// output out (`x`, or the head's input after the last layer). The common
    /// gated block runs the columns: a q8_0 block in one launch pair, a
    /// K-quant block a column at a time.
    fn dense(&mut self, l: usize) -> Result<(), GpuError> {
        const W: &str = "mimo2 pass dense";
        let (gpu, w, m) = (self.gpu, self.w, self.m);
        let stream = gpu.stream();
        let (d, c) = (*self.d, self.cfg[l]);
        let FfnNames::Dense {
            norm,
            gate,
            up,
            down,
        } = &self.names[l].ffn
        else {
            return Err(shape(format!(
                "layer {l}: the names of another block kind than the dense block"
            )));
        };
        let plan = c
            .block
            .ok_or_else(|| shape(format!("layer {l}: a dense block without its plan")))?;
        let last = self.last(l);
        let kernels = BlockKernels {
            experts: &self.k.experts,
            kquant: self.k.block.as_ref(),
        };
        let blk = Block {
            plan,
            names: [gate, up, down],
            ff: c.ffn.ff,
            limit: c.ffn.limit,
            layer: l,
        };
        let s = &mut self.b.s;
        gpu.elem().enqueue_rms_norm(
            stream,
            &s.x1,
            w.f32_buf(WHAT, norm)?,
            d.rms_eps,
            d.embd,
            m,
            &mut s.xn,
        )?;
        for (c0, cn) in chunks(m, COL_GROUP) {
            let xs = span(W, &s.xn, c0 * d.embd, cn * d.embd)?;
            let mut ys = span_mut(W, &mut s.out, c0 * d.embd, cn * d.embd)?;
            let io = BlockIo {
                x: &xs,
                h: &mut s.h,
                y: &mut ys,
            };
            enqueue_block_cols(gpu, w, &kernels, &mut s.block, &blk, cn, io)?;
        }
        let dest = if last {
            self.head.input_mut()
        } else {
            &mut s.x
        };
        gpu.elem()
            .enqueue_add(stream, &s.x1, &s.out, m * d.embd, dest)
    }

    /// Layer `l`'s routed block up to its go over the pass's rows: the norm
    /// into the pass's normed rows, the router over every row, one handoff a
    /// column into the boundary's image (each column's routing, activation
    /// and places at its offsets), and the go.
    fn route(&mut self, port: &mut StepLeg<'s, HostRun>, l: usize) -> Result<(), GpuError> {
        const W: &str = "mimo2 pass route";
        let (gpu, w, m) = (self.gpu, self.w, self.m);
        let stream = gpu.stream();
        let (d, c) = (*self.d, self.cfg[l]);
        let fault = gpu.layer_sink(l)?;
        let FfnNames::Moe { norm, router, bias } = &self.names[l].ffn else {
            return Err(shape(format!(
                "layer {l}: the names of another block kind than the routed block"
            )));
        };
        let PassBufs {
            s,
            normed,
            probs,
            ids,
            weights,
            ..
        } = &mut *self.b;
        gpu.elem().enqueue_rms_norm(
            stream,
            &s.x1,
            w.f32_buf(WHAT, norm)?,
            d.rms_eps,
            d.embd,
            m,
            normed,
        )?;
        let bias = match bias {
            Some(name) => w.f32_buf(WHAT, name)?,
            None => &s.no_bias,
        };
        self.k.router.enqueue_router_rows(
            stream,
            w.f32_tensor(WHAT, router)?,
            normed,
            bias,
            d.scale,
            m,
            probs,
            ids,
            weights,
            fault,
        )?;
        for col in 0..m {
            let col_ids = span(W, ids, col * N_USED, N_USED)?;
            let col_weights = span(W, weights, col * N_USED, N_USED)?;
            let x = span(W, normed, col * d.embd, d.embd)?;
            let mut sel = span_mut(W, &mut s.sel, col * N_USED, N_USED)?;
            let h = Handoff {
                ids: &col_ids,
                weights: &col_weights,
                map: self.slots.buf(),
                row_off: c.row_off,
                n_expert: N_EXPERT,
            };
            let t = port.hybrid().boundary_mut().handoff_target_of(0)?;
            let mut layout = t.layout;
            layout.ids += col * N_USED;
            layout.weights += col * N_USED;
            layout.x += col * d.embd;
            let target = HandoffTarget {
                x: &x,
                seq: t.seq,
                image: t.image,
                layout,
            };
            self.k
                .handoff
                .enqueue_handoff(stream, &h, target, fault, &mut sel)?;
        }
        port.hybrid().boundary().enqueue_go_of(stream, l, 0)
    }
}

impl<'s> LayerProgram for PassProgram<'s> {
    type Port = StepLeg<'s, HostRun>;

    /// The one unit of the walk, whose embeddings `plan` wrote into `x`.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        one_unit(unit)
    }

    /// The attention sub-layer, then the dense block whole, or the routed
    /// block up to its go.
    fn front(&mut self, port: &mut StepLeg<'s, HostRun>, at: At) -> Result<(), GpuError> {
        one_unit(at.unit)?;
        let l = at.layer;
        self.attention(l)?;
        match self.cfg[l].kind.ffn {
            FfnKind::Dense => self.dense(l),
            FfnKind::Moe => self.route(port, l),
        }
    }

    /// A routed layer's card experts over the pass's rows, under its host leg.
    fn shadow(&mut self, _port: &mut StepLeg<'s, HostRun>, at: At) -> Result<(), GpuError> {
        one_unit(at.unit)?;
        let (gpu, w, l, m) = (self.gpu, self.w, at.layer, self.m);
        if !self.cfg[l].kind.host_leg() || !self.card.has(l) {
            return Ok(());
        }
        let rule = Act::SwigluClamp {
            limit: self.cfg[l].ffn.limit,
        };
        let b = &*self.b;
        let io = Io {
            normed: &b.normed,
            weights: &b.weights,
            x1: &b.s.x1,
            m,
        };
        self.card.enqueue_pass(gpu, w, l, rule, &io, &b.s.sel)
    }

    /// A routed layer's wait and the add of the host's sums to the shadow's,
    /// and the host tier told the layer is enqueued (an eager pass is served
    /// there).
    fn back(&mut self, port: &mut StepLeg<'s, HostRun>, at: At) -> Result<(), GpuError> {
        one_unit(at.unit)?;
        let (gpu, l, m) = (self.gpu, at.layer, self.m);
        if !self.cfg[l].kind.host_leg() {
            return Ok(());
        }
        let stream = gpu.stream();
        let s = &mut self.b.s;
        let boundary = port.hybrid().boundary();
        boundary.enqueue_back_of(stream, 0)?;
        let shadowed = if self.card.has(l) {
            self.card.pass_pre()?
        } else {
            &s.x1
        };
        let dest = if l + 1 == self.cfg.len() {
            self.head.input_mut()
        } else {
            &mut s.x
        };
        gpu.elem().enqueue_add(
            stream,
            boundary.hsum_of(0)?,
            shadowed,
            m * self.d.embd,
            dest,
        )?;
        port.hybrid().row_enqueued(l, 0)
    }

    /// The head, over the last layer's rows.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        one_unit(unit)?;
        self.head.enqueue(self.gpu, self.w)
    }
}

/// The walk has one unit; another is refused by name.
fn one_unit(unit: usize) -> Result<(), GpuError> {
    if unit == 0 {
        return Ok(());
    }
    Err(shape_in(PASS, format!("unit {unit} of a walk of one unit")))
}
