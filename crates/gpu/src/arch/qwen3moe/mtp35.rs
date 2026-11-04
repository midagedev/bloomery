//! Qwen3.6's next-token (MTP) layer on the card: one draft step through
//! [`Body35`](super::Body35)'s own layer path.
//!
//! The file's tail block `blk.{n}` (`n` the trunk's layer count) is a
//! full-attention layer with the trunk's shapes beside four stems of its own:
//! `nextn.{enorm, hnorm, eh_proj, shared_head_norm}`. It borrows `token_embd`
//! and `output` from the target, and one step is
//!
//! ```text
//! x      = eh_proj · [ enorm(embed(t)) | hnorm(h) ]     one stream
//! x      = layer(x)             the trunk's attention + MoE, its own K/V
//! h'     = shared_head_norm(x)                          the next step's h
//! token  = argmax(output · h')
//! ```
//!
//! `h` is the target's post-`output_norm` row of the position before `t`. The
//! step's input join is the crate's `mtp_input` kernel at one stream; the
//! join's projection runs through the trunk's site owner
//! ([`dispatch::site_gemv`](super::dispatch::site_gemv)) at `eh_proj`'s own
//! resident type, a K-quant reading the pack's q8_1 rows like any trunk site;
//! the layer is the chain's walk of one layer (`walk::layer`) over the
//! family's ops ([`Program`](super::program::Program)) with a plan, weights
//! and store of the draft's own and an arena of one row, so attention, the
//! router, the routed and the shared experts are the trunk's launches; the
//! head is [`Head`] with the draft's gain over the target's own `output`
//! buffer, read where it sits. A step is eager and enqueues the same stream
//! as the target.
//!
//! The draft's K/V rows are its own: a step at position `p` appends row `p`
//! and attends rows `0..=p`, so a caller that wants the layer's attention to
//! see the prompt feeds every position in order. The target's matrices are
//! never copied: the draft records their addresses at open and refuses a step
//! by name when either has moved.
//!
//! A walk is eager. A captured walk is refused by name. A store walk
//! ([`WalkMode::Store`], the prompt's warmup) takes up to [`MTP35_STORE_ROWS`]
//! rows at once and runs them through the layer's cache append alone
//! (`walk::store` over [`StoreOps`]): the join, `eh_proj`, the attention's
//! input side — the norm, k and v — and the append at no query head; no
//! query, flash, FFN half or head. A later row reads a row's keys and values
//! and nothing else of it, and each op computes a row's values from that
//! row's inputs alone. The projections are the trunk's wide arm
//! ([`wide::gemm_site`]: one GEMM of the weight's own type, which reads each
//! weight once for the walk), over the walk's own buffers ([`StoreArena`]) and
//! a one-expert table; against an eager walk, which reads the f32 rows
//! through the gemv arm, the two stores agree to the error of the projections'
//! quantized activations and are not bit-equal. A step walk
//! ([`MtpArena::Step`]) reads the target head's normed row on the card,
//! copied into the draft's own row on the engine stream: no readback.
//! `held` counts the positions the draft's store holds; a walk from position
//! 0 replaces that count. The body's reset does not reach it.

use super::PrefillPath;
use super::body::Kernels;
use super::dispatch::{self, Ctx, PassCtx, embed_into};
use super::mtp38::BorrowedPlanes;
use super::plan::{DeltaPlan, FfnPlan, GqaKind, GqaPlan, LayerPlan, MixerPlan};
use super::program::{OneLayer, Program, Tail};
use super::scratch::{
    Append256, Arena, Dims, Forms, IN_IDS, IN_POS0, Inbox, Io, KvPlanes, StepParams, put_input,
};
use super::wide::{gemm_site, wide_kernels};
use crate::draft::{self, DraftHead, Feed, Hidden, WalkBounds, WalkMode};
use crate::gemm::{GemmInput, GemmRoute};
use crate::head::{Head, HeadNorm};
use crate::kv_format::KvQ8;
use crate::model::lookup::f32_gain;
use crate::mtp::{MtpInputArgs, MtpKernels};
use crate::site::{self, SiteTy, WideActs};
use crate::tensor::{Q8Act, Window, WindowMut};
use crate::weights::{DevWeight, HEAD_TENSOR, Weights};
use crate::{FaultSink, Gpu, GpuError, launch_u32};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::walk;
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::models::MtpDraft;
use model::arch::qwen3moe::names::token_embd;
use runtime::prompt::CallEnd;

const WHAT: &str = "qwen35moe Mtp35";

/// The draft's input join is one stream: the hidden row's width is the
/// embedding's.
const STREAMS: usize = 1;

/// The most rows one store walk takes: the columns of its GEMMs and the rows of
/// its buffers ([`StoreArena`]).
pub const MTP35_STORE_ROWS: usize = 256;

/// The draft layer's tensor names in its file.
struct Names {
    eh: String,
    enorm: String,
    hnorm: String,
}

impl Names {
    fn of(index: usize) -> Names {
        let blk = |stem: &str| super::body35::blk(index, stem);
        Names {
            eh: blk(model::arch::qwen35moe::mtp::EH_PROJ),
            enorm: blk(model::arch::qwen35moe::mtp::ENORM),
            hnorm: blk(model::arch::qwen35moe::mtp::HNORM),
        }
    }
}

/// The qwen35moe next-token layer, resident beside its target: its own
/// weights, plan, K/V planes, one-row arena and head scratch, and the
/// addresses of the two target matrices it reads in place.
pub struct Mtp35 {
    index: usize,
    hidden: usize,
    ctx: usize,
    vocab: usize,
    eps: f32,
    names: Names,
    /// The file's tail block `blk.{index}`: the layer's tensors and the four
    /// stems of its own, the router joined with the shared expert's gate.
    w: Weights,
    plan: LayerPlan,
    kv: KvPlanes,
    s: Arena,
    sp: StepParams,
    /// The embedding row of the step's token, and the join's pack
    /// `[enorm(e) | hnorm(h)]`.
    emb: DeviceBuffer<f32>,
    pack: DeviceBuffer<f32>,
    /// The pack's q8_1 form, [`Mtp35::enqueue`]'s `eh_proj` input when the
    /// site reads the q8_1 rows; unread, like the trunk sites' twin
    /// activations, at any other type.
    act: Q8Act,
    /// `eh_proj`'s site type, read from its resident form at the open.
    eh_ty: SiteTy,
    /// One target row copied out of an arena, and that row after
    /// `output_norm`, which [`Mtp35::enqueue`] reads.
    raw: DeviceBuffer<f32>,
    h_row: DeviceBuffer<f32>,
    head: Head,
    mtp: MtpKernels,
    /// A store walk's buffers.
    store: StoreArena,
    /// `token_embd` and the head matrix of the target, at open.
    borrowed: [BorrowedPlanes; 2],
    /// Positions of this sequence the store holds: the first it does not.
    /// A walk may start at or below it. A walk from position 0 replaces it.
    held: u32,
}

/// Where a walk reads the target's final hidden rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpArena {
    /// The head's last position, one row, read on the card.
    Step,
    /// The pass arena.
    Pass,
    /// The prompt's ubatch arena.
    Ubatch,
}

impl WalkBounds for Mtp35 {
    const WALK_ROWS: usize = 2;
    /// The layer captures no walk.
    const GRAPH_ROWS: usize = 0;
    const STORE_ROWS: usize = MTP35_STORE_ROWS;
}

/// A store walk's own buffers for up to [`MTP35_STORE_ROWS`] rows,
/// token-major: the input record (the walk's first position, then its ids),
/// the target's hidden rows after `output_norm`, the embedding rows with
/// their positions and live key counts, the join's pack and its quantized
/// forms, `eh_proj`'s rows (the layer's input), the attention-normed rows
/// with their forms, k and v, and the one-expert table every projection
/// reads. `qg` and `q` stand in for the append's query arguments at no query
/// head, where the launch reads and writes neither.
struct StoreArena {
    inbox: Inbox,
    h: DeviceBuffer<f32>,
    emb: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    pack: DeviceBuffer<f32>,
    pack_acts: WideActs,
    x: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    normed_acts: WideActs,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    rows: GemmRoute,
    qg: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
}

impl StoreArena {
    /// The buffers for `d`'s shapes, `eh_proj` read at `eh_ty` and the
    /// attention's k and v at `g`'s types. Load-time only.
    fn new(
        stream: &CudaStream,
        d: &Dims,
        eh_ty: SiteTy,
        g: &GqaPlan,
    ) -> Result<StoreArena, GpuError> {
        let r = MTP35_STORE_ROWS;
        let (h, kv) = (d.hidden, d.kv_len());
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let u = |n: usize| DeviceBuffer::<u32>::zeroed(stream, n);
        // The f32 rows a site of an f32 type reads are `pack` and `normed`
        // themselves, so the forms hold none.
        let pack_forms = WideActs::wants(eh_ty).without_f32();
        let kv_forms = WideActs::wants_of(&[g.k_ty, g.v_ty]).without_f32();
        Ok(StoreArena {
            inbox: Inbox::new(stream, IN_IDS + r)?,
            h: f(r * h)?,
            emb: f(r * h)?,
            pos: u(r)?,
            n_keys: u(r)?,
            pack: f(r * 2 * h)?,
            pack_acts: WideActs::new(stream, pack_forms, r, 2 * h)?,
            x: f(r * h)?,
            normed: f(r * h)?,
            normed_acts: WideActs::new(stream, kv_forms, r, h)?,
            k: f(r * kv)?,
            v: f(r * kv)?,
            rows: GemmRoute::new(stream, r, 1)?,
            qg: f(1)?,
            q: f(1)?,
        })
    }

    /// Device bytes.
    fn bytes(&self) -> usize {
        let f32s = [
            &self.h,
            &self.emb,
            &self.pack,
            &self.x,
            &self.normed,
            &self.k,
            &self.v,
            &self.qg,
            &self.q,
        ];
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + self.inbox.bytes()
            + self.pack_acts.bytes()
            + self.normed_acts.bytes()
            + self.rows.bytes()
    }
}

/// Device addresses held by [`BorrowedPlanes`]: Q8_0 or MXFP8 planes,
/// or BF16 words and a zero
/// ([`super::mtp38::planes`]), a K-quant weight's one plane and a zero.
pub fn addresses(dw: &DevWeight) -> Option<[u64; 2]> {
    match dw {
        DevWeight::KQuant { w, .. } => Some([w.buf().cu_deviceptr(), 0]),
        _ => super::mtp38::planes(dw),
    }
}

/// `name` of the target's weights `target`, the matrix of `hidden` columns and
/// `vocab` rows the draft reads in place, with its addresses.
fn borrow(
    target: &Weights,
    name: &str,
    hidden: usize,
    vocab: usize,
) -> Result<BorrowedPlanes, GpuError> {
    let dw = target
        .get(name)
        .ok_or_else(|| GpuError::tensor(WHAT, name, "resident on the target's card"))?;
    if (dw.k(), dw.rows()) != (hidden, vocab) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the target's {name} is {} rows of {} values; the draft reads [{hidden}, {vocab}]",
                dw.rows(),
                dw.k()
            ),
        ));
    }
    let planes = addresses(dw).ok_or_else(|| {
        GpuError::tensor(
            WHAT,
            name,
            "Q8_0 or MXFP8 planes, BF16 words, or a K-quant word plane",
        )
    })?;
    Ok(BorrowedPlanes {
        name: name.to_string(),
        planes,
    })
}

impl Mtp35 {
    /// The draft over its resident weights `w` and `plan` (the tail block of
    /// the target's file, planned and joined like a trunk layer), beside the
    /// target's `target` weights, with `ctx` K/V rows of the format `kv` and
    /// the arena of `d`'s shapes. Load-time only.
    pub(super) fn open(
        gpu: &Gpu,
        target: &Weights,
        draft: &MtpDraft,
        (w, plan): (Weights, LayerPlan),
        (d, kv): (Dims, KvQ8),
    ) -> Result<Mtp35, GpuError> {
        let stream = gpu.stream();
        let index = draft.index as usize;
        let (hidden, vocab) = (draft.hidden as usize, draft.vocab as usize);
        let names = Names::of(index);
        if d.hidden != hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the arena's width is {}, the next-token layer's {hidden}",
                    d.hidden
                ),
            ));
        }
        let borrowed = [
            borrow(target, &token_embd(), hidden, vocab)?,
            borrow(target, HEAD_TENSOR, hidden, vocab)?,
        ];
        // The join's projection site, at its resident form's type: the type
        // a file leaves `eh_proj` decides the form the step feeds it, so it
        // is read once here, like the head's own output form.
        let eh_ty = site::resident_type(&w, WHAT, &names.eh)?;
        let MixerPlan::Gqa(g) = &plan.mixer else {
            return Err(GpuError::shape(
                WHAT,
                "the next-token layer's mixer is not attention",
            ));
        };
        if g.kind != GqaKind::Gated256 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the next-token layer's attention is {:?}; the store walk runs the head-256 \
                     gated append",
                    g.kind
                ),
            ));
        }
        let store = StoreArena::new(stream, &d, eh_ty, g)?;
        let forms = Forms::of(std::slice::from_ref(&plan), &d);
        let head = Head::with_gain(
            gpu,
            (
                &w,
                &super::body35::blk(index, model::arch::qwen35moe::mtp::SHARED_HEAD_NORM),
            ),
            target,
            draft.rms_eps,
            1,
            HeadNorm::Rms,
        )?;
        Ok(Mtp35 {
            index,
            hidden,
            ctx: d.ctx,
            vocab,
            eps: draft.rms_eps,
            kv: KvPlanes::new(stream, &d, kv)?,
            s: Arena::with(stream, d, 1, forms)?,
            sp: StepParams::new(stream, false)?,
            emb: DeviceBuffer::zeroed(stream, hidden)?,
            pack: DeviceBuffer::zeroed(stream, 2 * STREAMS * hidden)?,
            act: Q8Act::with_k(stream, 1, 2 * STREAMS * hidden)?,
            raw: DeviceBuffer::zeroed(stream, hidden)?,
            h_row: DeviceBuffer::zeroed(stream, hidden)?,
            mtp: MtpKernels::load(gpu.context())?,
            store,
            names,
            w,
            plan,
            eh_ty,
            head,
            borrowed,
            held: 0,
        })
    }

    /// The layer's index in the file.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Values of one hidden row the walk reads.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Positions of this sequence the store holds: the first it does not.
    #[must_use]
    pub fn held(&self) -> u32 {
        self.held
    }

    /// The draft's own weights, by file name.
    #[must_use]
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// Card bytes the draft holds as the load's plan counts them: its weights,
    /// its K/V planes and the one f32 residual stream its layer writes and
    /// its head reads. The target's two matrices are not here, and neither is
    /// the step's scratch ([`Mtp35::scratch_bytes`]).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.resident_bytes() + self.kv.bytes() + self.hidden * size_of::<f32>()
    }

    /// Card bytes of the step's scratch: the one-row arena, the embedding row,
    /// the join's pack and its q8_1 form, the copied target row and its normed
    /// row, the head's rows, activations and logits, and a store walk's
    /// buffers ([`StoreArena`]), less the residual stream counted in
    /// [`Mtp35::resident_bytes`].
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.store.bytes()
            + self.emb.num_bytes()
            + self.pack.num_bytes()
            + self.act.device_bytes()
            + self.raw.num_bytes()
            + self.h_row.num_bytes()
            + self.sp.bytes()
            + self.head.resident_bytes()
            - self.hidden * size_of::<f32>()
    }

    /// The target matrices the draft reads in place, `token_embd` then the
    /// head's, each with the addresses it had at open.
    #[must_use]
    pub fn borrowed_planes(&self) -> &[BorrowedPlanes; 2] {
        &self.borrowed
    }

    /// The name of the gain tensor the draft's head norms by: the layer's own
    /// `nextn.shared_head_norm`.
    #[must_use]
    pub fn head_gain(&self) -> &str {
        self.head.gain()
    }

    /// The store's keys and values at positions `0..n`, position-major:
    /// `[n][n_kv][head]` f16 bits each (the store's own layout is
    /// `[n_kv][ctx][head]`). Blocking; gate use.
    pub fn store_host(
        &self,
        stream: &CudaStream,
        n: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
        if n > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("positions 0..{n} of a store of {}", self.ctx),
            ));
        }
        let (ctx, head, n_kv) = (self.ctx, self.s.dims.head, self.s.dims.n_kv);
        let plane = |b: &DeviceBuffer<u16>| -> Result<Vec<u16>, GpuError> {
            let heads = (0..n_kv)
                .map(|kv| {
                    let at = kv * ctx * head * size_of::<u16>();
                    Ok(Window::<u16>::of(b, at, n * head)?.to_host_vec(stream)?)
                })
                .collect::<Result<Vec<_>, GpuError>>()?;
            Ok((0..n)
                .flat_map(|p| heads.iter().map(move |h| &h[p * head..(p + 1) * head]))
                .flatten()
                .copied()
                .collect())
        };
        let (k, v) = self.kv.f16(WHAT)?;
        Ok((plane(k)?, plane(v)?))
    }

    /// The last step's logits (`vocab` f32). Blocking read; gate/debug use.
    pub fn logits_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        self.head.logits_to_host(gpu)
    }

    /// Enqueue one step of the draft for token `t` at position `pos` over
    /// the target's hidden row `h` (`hidden` f32 on the card, the row of the
    /// position before `t`): the join, `eh_proj`, the layer at `pos` over the
    /// draft's K/V rows `0..=pos`, and the head, whose token and logits the
    /// caller reads back. Eager; the target's `target` weights, kernels and
    /// rope table are the ones its own layers launch with.
    pub(super) fn enqueue(
        &mut self,
        gpu: &Gpu,
        target: &Weights,
        (k, table, mma): (&Kernels, &DeviceBuffer<f32>, bool),
        (t, pos): (u32, usize),
        h: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        super::refuse_past_vocab(WHAT, &[t], self.vocab)?;
        if pos >= self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("position {pos} on a draft of {} K/V rows", self.ctx),
            ));
        }
        if h.len() < self.hidden {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a hidden row of {} values; the draft reads {}",
                    h.len(),
                    self.hidden
                ),
            ));
        }
        for b in &self.borrowed {
            super::mtp38::at_open(target, b, WHAT)?;
        }
        let stream = gpu.stream();
        self.sp.write(stream, t, launch_u32(WHAT, "pos", pos)?)?;
        embed_into(
            gpu,
            target,
            k,
            &self.sp.io(),
            (&mut self.emb, &mut self.s.pos, &mut self.s.n_keys),
        )?;
        self.mtp.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &self.emb,
                h,
                enorm: f32_gain(&self.w, &self.names.enorm)?,
                hnorm: f32_gain(&self.w, &self.names.hnorm)?,
                hidden: self.hidden,
                eps: self.eps,
                m: 1,
                streams: STREAMS,
                fault: gpu.layer_sink(self.index)?,
                out: &mut self.pack,
            },
        )?;
        let c = PassCtx {
            gpu,
            w: &self.w,
            plans: std::slice::from_ref(&self.plan),
            k,
            mma,
            eps: self.eps,
            table,
        };
        // The projection is the trunk's site dispatch at the weight's own
        // type: the pack's quantizer runs only where that type reads the
        // q8_1 rows, and the gemv takes the form it names.
        if dispatch::reads_q8_1(self.eh_ty) {
            gpu.enqueue_quantize_q8_1_layer(&self.pack, &mut self.act, self.index)?;
        }
        dispatch::site_gemv(
            &Ctx::new(
                gpu,
                &self.w,
                (&self.plan, self.index),
                k,
                mma,
                self.eps,
                table,
            )?,
            (self.eh_ty, &self.names.eh),
            self.hidden,
            (&self.act, &self.pack),
            1,
            None,
            &mut self.s.x,
        )?;
        let cx = LayerCtx {
            layer: self.index,
            plan: &self.plan,
            m: 1,
            last: true,
        };
        let mut layer = Program {
            c: &c,
            stores: &mut OneLayer {
                index: self.index,
                kv: &mut self.kv,
            },
            s: &mut self.s,
            io: &self.sp.io(),
            m: 1,
            tail: Tail::Into(self.head.input_mut()),
        };
        walk::layer(&mut layer, cx)?;
        self.head.enqueue_over(gpu, &self.w, target)
    }

    /// The last step's argmax token (first index on a tie): a blocking read,
    /// where a fault any launch of the step raised is [`GpuError::Fault`].
    pub fn token(&self, gpu: &Gpu) -> Result<u32, GpuError> {
        self.head.token(gpu)
    }

    /// [`Mtp35::enqueue`] over the resident normed row. `enqueue` reads that
    /// row and does not write it, and the row stays allocated for the call.
    fn enqueue_held(
        &mut self,
        gpu: &Gpu,
        target: &Weights,
        kernels: (&Kernels, &DeviceBuffer<f32>, bool),
        feed: (u32, usize),
    ) -> Result<(), GpuError> {
        // The pointer is taken before `enqueue` borrows `self`, so the two
        // borrows of the row do not alias in the checker.
        let h = std::ptr::from_ref(&self.h_row);
        // SAFETY: `enqueue` reads `h` and does not write `h_row`. `h_row`
        // stays allocated.
        self.enqueue(gpu, target, kernels, feed, unsafe { &*h })
    }
}

impl Mtp35 {
    /// Enqueue a store walk of `tokens` at positions `pos0 ..`, their hidden
    /// rows the store arena's `h`: the embedding rows, the join, `eh_proj`
    /// as one GEMM over the rows, then `walk::store` over [`StoreOps`] — the
    /// attention's norm, k and v as GEMMs, and the append at no query head
    /// into the draft's K/V planes. The head does not run and the last step's
    /// token and logits stay. Each row's keys and values are its inputs'
    /// alone, so a row's stored bits do not depend on the walk it lands in.
    /// The target's `target` weights, kernels and rope table are the ones its
    /// own layers launch with.
    pub(super) fn enqueue_store(
        &mut self,
        gpu: &Gpu,
        target: &Weights,
        (k, table): (&Kernels, &DeviceBuffer<f32>),
        (tokens, pos0): (&[u32], usize),
    ) -> Result<(), GpuError> {
        let n = tokens.len();
        super::refuse_past_vocab(WHAT, tokens, self.vocab)?;
        if pos0.saturating_add(n) > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "positions {pos0}..{} on a draft of {} K/V rows",
                    pos0.saturating_add(n),
                    self.ctx
                ),
            ));
        }
        for b in &self.borrowed {
            super::mtp38::at_open(target, b, WHAT)?;
        }
        let stream = gpu.stream();
        let sink = gpu.layer_sink(self.index)?;
        let d = self.s.dims;
        let Mtp35 {
            index,
            hidden,
            eps,
            names,
            w,
            plan,
            kv,
            eh_ty,
            mtp,
            store,
            ..
        } = self;
        let (hidden, eps) = (*hidden, *eps);
        let (w, plan, names): (&Weights, &LayerPlan, &Names) = (w, plan, names);
        let StoreArena {
            inbox,
            h,
            emb,
            pos,
            n_keys,
            pack,
            pack_acts,
            x,
            rows,
            ..
        } = &mut *store;

        // The walk's first position and its ids, then the embedding rows
        // with their positions and live key counts.
        put_input(inbox.host_mut()?, tokens, launch_u32(WHAT, "pos", pos0)?)?;
        inbox.upload(stream, IN_IDS + n)?;
        {
            let ids = Window::<u32>::of(inbox.dev(), IN_IDS * size_of::<u32>(), n)?;
            let first = Window::<u32>::of(inbox.dev(), IN_POS0 * size_of::<u32>(), 1)?;
            let io = Io {
                ids: &ids,
                pos0: &first,
                first: 0,
                lane: None,
            };
            embed_into(gpu, target, k, &io, (emb, pos, n_keys))?;
        }

        // The join of every row, then `eh_proj` over them as one GEMM into
        // the layer's input rows.
        mtp.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: emb,
                h,
                enorm: f32_gain(w, &names.enorm)?,
                hnorm: f32_gain(w, &names.hnorm)?,
                hidden,
                eps,
                m: n,
                streams: STREAMS,
                fault: sink,
                out: pack,
            },
        )?;
        let wk = wide_kernels(k);
        pack_acts.quantize(gpu, &wk, sink, pack, n, &[*eh_ty])?;
        k.gemm
            .enqueue_route_dense(stream, n, rows, gpu.unlabelled_sink())?;
        let input = pack_acts.input(pack);
        gemm_site(
            (gpu, w, k),
            (*eh_ty, &names.eh),
            hidden,
            &input,
            (rows, GemmInput::PerSlot),
            n,
            x,
        )?;

        let mut ops = StoreOps {
            gpu,
            w,
            k,
            table,
            eps,
            sink,
            d,
            kv,
            s: store,
        };
        let cx = LayerCtx {
            layer: *index,
            plan,
            m: n,
            last: true,
        };
        walk::store(&mut ops, cx)
    }
}

/// The ops of a store walk ([`walk::store`]) over the draft layer: the
/// attention's input side without its query, and the cache append. Every other
/// op of a layer is no part of one and is refused by name.
struct StoreOps<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    table: &'a DeviceBuffer<f32>,
    eps: f32,
    sink: FaultSink,
    d: Dims,
    kv: &'a mut KvPlanes,
    s: &'a mut StoreArena,
}

/// The refusal of op `op` in a store walk.
fn outside_store(op: &'static str) -> GpuError {
    GpuError::shape(
        WHAT,
        format!("{op} in a store walk: it runs the attention's input side and its append alone"),
    )
}

impl ChainOps for StoreOps<'_> {
    type Error = GpuError;

    fn embed(&mut self, _: usize) -> Result<(), GpuError> {
        Err(outside_store("the unit's front"))
    }

    /// The layer's input rows (`eh_proj`'s) normed and quantized into the
    /// forms k and v read, then k and v as the trunk's wide arm runs them. No
    /// query: nothing reads one of a store walk's rows.
    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let (gpu, w, k, m) = (self.gpu, self.w, self.k, l.m);
        let StoreArena {
            x,
            normed,
            normed_acts,
            k: kr,
            v,
            rows,
            ..
        } = &mut *self.s;
        gpu.elem().enqueue_rms_norm(
            gpu.stream(),
            x,
            f32_gain(w, &g.attn_norm)?,
            self.eps,
            self.d.hidden,
            m,
            normed,
        )?;
        normed_acts.quantize(
            gpu,
            &wide_kernels(k),
            self.sink,
            normed,
            m,
            &[g.k_ty, g.v_ty],
        )?;
        let input = normed_acts.input(normed);
        let table = (&*rows, GemmInput::PerSlot);
        let kv_len = self.d.kv_len();
        gemm_site(
            (gpu, w, k),
            (g.k_ty, &g.attn_k),
            kv_len,
            &input,
            table,
            m,
            kr,
        )?;
        gemm_site(
            (gpu, w, k),
            (g.v_ty, &g.attn_v),
            kv_len,
            &input,
            table,
            m,
            v,
        )
    }

    fn attn_core(&mut self, _: LayerCtx<'_>, _: &GqaPlan) -> Result<(), GpuError> {
        Err(outside_store("the attention's flash"))
    }

    /// The keys' norm and turn by the rows' positions and the append of keys
    /// and values into the draft's planes, at no query head.
    fn attn_append(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let StoreArena {
            pos,
            k: kr,
            v,
            qg,
            q,
            ..
        } = &mut *self.s;
        // SAFETY: the rows are one embedding's consecutive positions below
        // the planes' rows (`enqueue_store` refuses a walk past them), so
        // they are pairwise distinct and cache-valid; the planes are this
        // draft's own.
        unsafe {
            self.kv.append_256(
                &self.k.neox,
                self.gpu.stream(),
                Append256 {
                    qg,
                    q,
                    k: kr,
                    v,
                    gq: f32_gain(self.w, &g.attn_q_norm)?,
                    gk: f32_gain(self.w, &g.attn_k_norm)?,
                    table: self.table,
                    pos,
                    eps: self.eps,
                    n_head: 0,
                    n_kv: self.d.n_kv,
                    ctx: self.d.ctx,
                    m: l.m,
                    fault: self.sink,
                },
            )
        }
    }

    fn attn_out(&mut self, _: LayerCtx<'_>, _: &GqaPlan) -> Result<(), GpuError> {
        Err(outside_store("the attention's output side"))
    }

    fn delta_in(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(outside_store("a delta layer"))
    }

    fn delta_core(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(outside_store("a delta layer"))
    }

    fn delta_out(&mut self, _: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), GpuError> {
        Err(outside_store("a delta layer"))
    }

    fn route(&mut self, _: LayerCtx<'_>, _: &FfnPlan) -> Result<(), GpuError> {
        Err(outside_store("the FFN's router"))
    }

    fn experts(&mut self, _: LayerCtx<'_>, _: &FfnPlan) -> Result<(), GpuError> {
        Err(outside_store("the FFN's experts"))
    }

    fn head(&mut self, _: usize) -> Result<(), GpuError> {
        Err(outside_store("the head"))
    }
}

fn no_layer() -> GpuError {
    GpuError::state(
        WHAT,
        "the load carries no next-token layer (open it with GpuModel::open_mtp)",
    )
}

/// Copy arena row `row` and RMS-norm it by the target's `output_norm` into
/// the draft's normed row.
fn norm_row(
    mtp: &mut Mtp35,
    gpu: &Gpu,
    w: &Weights,
    arena: &Arena,
    eps: f32,
    row: usize,
) -> Result<(), GpuError> {
    let hidden = mtp.hidden;
    if row >= arena.rows {
        return Err(GpuError::shape(
            WHAT,
            format!("hidden row {row} of a {}-row arena", arena.rows),
        ));
    }
    let src = Window::<f32>::of(&arena.x, row * hidden * size_of::<f32>(), hidden)?;
    mtp.raw.copy_from_device_async(&src, gpu.stream())?;
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        &mtp.raw,
        f32_gain(w, "output_norm.weight")?,
        eps,
        hidden,
        1,
        &mut mtp.h_row,
    )
}

/// The first `n` rows of `src` from row `first`, each RMS-normed by the
/// target's `output_norm` into the store arena's `h`.
fn norm_rows(
    mtp: &mut Mtp35,
    gpu: &Gpu,
    w: &Weights,
    (src, eps): (&Arena, f32),
    (first, n): (usize, usize),
) -> Result<(), GpuError> {
    let hidden = mtp.hidden;
    if first.saturating_add(n) > src.rows {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "hidden rows {first}..{} of a {}-row arena",
                first.saturating_add(n),
                src.rows
            ),
        ));
    }
    let rows = Window::<f32>::of(&src.x, first * hidden * size_of::<f32>(), n * hidden)?;
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        &rows,
        f32_gain(w, "output_norm.weight")?,
        eps,
        hidden,
        n,
        &mut mtp.store.h,
    )
}

/// One eager walk of the file's next-token layer, no readback. A captured
/// walk is refused by name. A store walk runs its rows through the layer's
/// cache append alone, all at once ([`Mtp35::enqueue_store`]). A poisoned
/// model is refused; a fault the walk raises is [`GpuModel::note_fault`]'s.
pub fn mtp_walk(
    m: &mut super::Qwen35moeModel,
    feed: Feed<'_, MtpArena>,
    head: DraftHead,
    mode: WalkMode,
) -> Result<(), GpuError> {
    m.refuse_if_poisoned(WHAT)?;
    let r = walk_rows(m, feed, head, mode);
    m.note_fault(WHAT, r)
}

/// One window's chain: the refresh walked, then the head's argmax into
/// `out`'s first place. An own row, a probability buffer, a store walk and a
/// captured walk are refused by name. The proposal is one id.
pub fn mtp_chain(
    m: &mut super::Qwen35moeModel,
    refresh: Feed<'_, MtpArena>,
    own: usize,
    head: DraftHead,
    mode: WalkMode,
    out: &mut [u32],
    p: Option<&mut [f32]>,
) -> Result<usize, GpuError> {
    m.refuse_if_poisoned(WHAT)?;
    let r = chain_rows(m, refresh, own, head, mode, out, p);
    m.note_fault(WHAT, r)
}

fn chain_rows(
    m: &mut super::Qwen35moeModel,
    refresh: Feed<'_, MtpArena>,
    own: usize,
    head: DraftHead,
    mode: WalkMode,
    out: &mut [u32],
    p: Option<&mut [f32]>,
) -> Result<usize, GpuError> {
    if own != 0 {
        return Err(draft::refuse_own(
            WHAT,
            &format!("a chain of {own} own rows"),
        ));
    }
    if p.is_some() {
        return Err(GpuError::shape(
            WHAT,
            "a chain's probabilities: the next-token layer's head reads back its ids alone",
        ));
    }
    if mode == WalkMode::Store {
        return Err(GpuError::shape(
            WHAT,
            "a chain in the store walk's mode: it reads the head",
        ));
    }
    if out.is_empty() {
        return Err(GpuError::shape(WHAT, "a proposal of 1 id into 0 places"));
    }
    walk_rows(m, refresh, head, mode)?;
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let id = body.mtp.as_ref().ok_or_else(no_layer)?.token(gpu)?;
    out[0] = id;
    Ok(1)
}

fn walk_rows(
    m: &mut super::Qwen35moeModel,
    feed: Feed<'_, MtpArena>,
    head: DraftHead,
    mode: WalkMode,
) -> Result<(), GpuError> {
    if mode == WalkMode::Graph {
        return Err(GpuError::shape(
            WHAT,
            "a captured walk: the next-token layer runs eager",
        ));
    }
    if feed.media.is_some() {
        return Err(GpuError::shape(
            WHAT,
            "a media load: its sequences' rotation tables are their own, and the draft reads the \
             plain table",
        ));
    }
    head.check_opened(WHAT, false)?;
    let Feed {
        tokens,
        pos0,
        hidden,
        media: _,
    } = feed;
    let n = tokens.len();
    draft::refuse_rows::<Mtp35>(WHAT, mode, n)?;
    let width = {
        let (_, _, body) = m.body_parts(WHAT)?;
        body.mtp.as_ref().ok_or_else(no_layer)?.hidden
    };
    if let Hidden::Host(v) = hidden
        && v.len() != n * width
    {
        return Err(GpuError::shape(
            WHAT,
            format!("{} hidden values for {n} rows of {width}", v.len()),
        ));
    }
    let (gpu, w, body, step_row) = match hidden {
        Hidden::Target {
            walk: MtpArena::Step,
            first,
        } => {
            if first != 0 || n != 1 {
                return Err(GpuError::shape(
                    WHAT,
                    "the step arena holds the head's last position only: a walk reads one row \
                     from its row 0",
                ));
            }
            let (gpu, w, body, row) = m.body_parts_normed(WHAT)?;
            if row.len() != width {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the head's normed row is {} values; the draft reads {width}",
                        row.len()
                    ),
                ));
            }
            (gpu, w, body, Some(row))
        }
        _ => {
            let (gpu, w, body) = m.body_parts(WHAT)?;
            (gpu, w, body, None)
        }
    };
    let Body35Fields {
        eps,
        u,
        a,
        mtp,
        k,
        rope,
        seq_rope,
        mma,
    } = split_body(body)?;
    if pos0 == 0 {
        mtp.held = 0;
    }
    if usize::try_from(pos0).unwrap_or(usize::MAX) > mtp.held as usize {
        let held = mtp.held;
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a walk from position {pos0}: the draft's store holds this sequence's \
                 positions below {held}, and the walk would read the keys between"
            ),
        ));
    }
    let table = crate::media::rope_table(rope, seq_rope.as_ref());
    let mma = *mma;
    let eps = *eps;
    if mode == WalkMode::Store {
        match hidden {
            Hidden::Host(v) => {
                let mut rows = WindowMut::<f32>::of_mut(&mut mtp.store.h, 0, n * width)?;
                rows.copy_from_host(gpu.stream(), v)?;
            }
            Hidden::Target {
                walk: MtpArena::Step,
                ..
            } => {
                let row = step_row
                    .ok_or_else(|| GpuError::state(WHAT, "the head's normed row of the step"))?;
                let mut rows = WindowMut::<f32>::of_mut(&mut mtp.store.h, 0, width)?;
                rows.copy_from_device_async(row, gpu.stream())?;
            }
            Hidden::Target {
                walk: MtpArena::Pass,
                first,
            } => norm_rows(mtp, gpu, w, (a, eps), (first, n))?,
            Hidden::Target {
                walk: MtpArena::Ubatch,
                first,
            } => norm_rows(mtp, gpu, w, (u, eps), (first, n))?,
        }
        mtp.enqueue_store(gpu, w, (k, table), (tokens, pos0 as usize))?;
    } else {
        for i in 0..n {
            let pos = (pos0 as usize).checked_add(i).ok_or_else(|| {
                GpuError::shape(WHAT, format!("position {pos0} plus {i} passes usize"))
            })?;
            let t = tokens[i];
            match hidden {
                Hidden::Host(v) => {
                    mtp.h_row
                        .copy_from_host(gpu.stream(), &v[i * width..(i + 1) * width])?;
                }
                Hidden::Target {
                    walk: MtpArena::Step,
                    ..
                } => {
                    let row = step_row.ok_or_else(|| {
                        GpuError::state(WHAT, "the head's normed row of the step")
                    })?;
                    mtp.h_row.copy_from_device_async(row, gpu.stream())?;
                }
                Hidden::Target {
                    walk: MtpArena::Pass,
                    first,
                } => norm_row(mtp, gpu, w, a, eps, first + i)?,
                Hidden::Target {
                    walk: MtpArena::Ubatch,
                    first,
                } => norm_row(mtp, gpu, w, u, eps, first + i)?,
            }
            mtp.enqueue_held(gpu, w, (k, table, mma), (t, pos))?;
        }
    }
    let n_u = u32::try_from(n)
        .map_err(|_| GpuError::shape(WHAT, format!("a walk of {n} rows passes u32")))?;
    let end = pos0.checked_add(n_u).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("a walk of {n} rows from position {pos0} passes u32"),
        )
    })?;
    mtp.held = mtp.held.max(end);
    Ok(())
}

/// The body's fields a walk borrows apart: the arenas, the draft, the rope.
struct Body35Fields<'a> {
    eps: &'a f32,
    u: &'a Arena,
    a: &'a Arena,
    mtp: &'a mut Mtp35,
    k: &'a Kernels,
    rope: &'a super::scratch::RopeRows,
    seq_rope: &'a Option<crate::media::SeqRope>,
    mma: &'a bool,
}

fn split_body(body: &mut super::body35::Body35) -> Result<Body35Fields<'_>, GpuError> {
    let super::body35::Body35 {
        eps,
        u,
        a,
        mtp,
        k,
        rope,
        seq_rope,
        mma,
        ..
    } = body;
    Ok(Body35Fields {
        eps,
        u,
        a,
        mtp: mtp.as_mut().ok_or_else(no_layer)?,
        k,
        rope,
        seq_rope,
        mma,
    })
}

/// A prompt call's tap ([`prompt_with35`]): called after each unit with the
/// arena the unit's rows sit in, the unit's first position and its rows.
pub type Prompt35Sink<'a> =
    dyn FnMut(&mut super::Qwen35moeModel, MtpArena, u32, usize) -> Result<(), GpuError> + 'a;

/// The prompt call on `path` ending as `end` asks
/// ([`super::Qwen35moeModel::prefill_call`]), one residency pass, and `sink`
/// after each unit with the ubatch arena the unit's rows sit in.
pub fn prompt_with35(
    m: &mut super::Qwen35moeModel,
    ids: &[u32],
    path: PrefillPath,
    end: CallEnd,
    sink: Option<&mut Prompt35Sink<'_>>,
) -> Result<Option<u32>, GpuError> {
    crate::host::served::prompt_pass(m, |m| match sink {
        None => m.prefill_hooked(ids, path, None, end),
        Some(sink) => {
            let mut hook = |model: &mut super::Qwen35moeModel, first: u32, rows: usize| {
                sink(model, MtpArena::Ubatch, first, rows)
            };
            m.prefill_hooked(ids, path, Some(&mut hook), end)
        }
    })
}
