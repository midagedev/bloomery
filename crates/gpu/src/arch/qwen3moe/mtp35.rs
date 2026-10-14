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
//! layer is [`dispatch::layer`](super::dispatch::layer) over a plan, weights
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

use super::body::{Kernels, KvQ8};
use super::dispatch::{self, Ctx, embed_into};
use super::mtp38::BorrowedPlanes;
use super::plan::LayerPlan;
use super::scratch::{Arena, Dims, Forms, KvPlanes, StepParams, StoreMut};
use crate::head::{Head, HeadNorm};
use crate::model::lookup::f32_gain;
use crate::mtp::{MtpInputArgs, MtpKernels};
use crate::weights::{DevWeight, HEAD_TENSOR, Weights};
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::DeviceBuffer;
use model::arch::models::MtpDraft;
use model::arch::qwen3moe::names::token_embd;

const WHAT: &str = "qwen35moe Mtp35";

/// The draft's input join is one stream: the hidden row's width is the
/// embedding's.
const STREAMS: usize = 1;

/// The draft layer's tensor names in its file.
struct Names {
    eh: String,
    enorm: String,
    hnorm: String,
    head_gain: String,
}

impl Names {
    fn of(index: usize) -> Names {
        let blk = |stem: &str| super::body35::blk(index, stem);
        Names {
            eh: blk(model::arch::qwen35moe::mtp::EH_PROJ),
            enorm: blk(model::arch::qwen35moe::mtp::ENORM),
            hnorm: blk(model::arch::qwen35moe::mtp::HNORM),
            head_gain: blk(model::arch::qwen35moe::mtp::SHARED_HEAD_NORM),
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
    head: Head,
    mtp: MtpKernels,
    /// `token_embd` and the head matrix of the target, at open.
    borrowed: [BorrowedPlanes; 2],
}

/// A weight's device addresses as a [`BorrowedPlanes`] holds them: a Q8_0
/// weight's two planes ([`super::mtp38::planes`]), a K-quant weight's one
/// plane and a zero.
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
            "Q8_0 planes or a K-quant word plane (the head's forms)",
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
        let forms = Forms::of(std::slice::from_ref(&plan), &d);
        let head = Head::with_gain(
            gpu,
            (&w, names.head_gain.as_str()),
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
            mtp: MtpKernels::load(gpu.context())?,
            names,
            w,
            plan,
            head,
            borrowed,
        })
    }

    /// The layer's index in the file.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
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
    /// the join's pack and the head's rows, activations and logits, less the
    /// residual stream counted in [`Mtp35::resident_bytes`].
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.emb.num_bytes()
            + self.pack.num_bytes()
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
        self.w
            .q8_gemv(gpu, WHAT, &self.names.eh, &self.pack, &mut self.s.x)?;
        let c = Ctx::new(
            gpu,
            &self.w,
            (&self.plan, self.index),
            k,
            mma,
            self.eps,
            table,
        )?;
        dispatch::layer(
            &c,
            StoreMut::Kv(&mut self.kv),
            &mut self.s,
            &self.sp.io(),
            1,
            false,
            Some(self.head.input_mut()),
        )?;
        self.head.enqueue_over(gpu, &self.w, target)
    }

    /// The last step's argmax token (first index on a tie): a blocking read,
    /// where a fault any launch of the step raised is [`GpuError::Fault`].
    pub fn token(&self, gpu: &Gpu) -> Result<u32, GpuError> {
        self.head.token(gpu)
    }
}
