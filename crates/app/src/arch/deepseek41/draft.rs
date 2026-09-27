//! The DSpark draft a V4.1 session drives: the draft body
//! (`bloomery_gpu_deepseek41::draft::DraftBody`) on a card of its own, fed
//! the target's feature tap (`body::attach_features`, `Body::read_features`)
//! in the order the draft module's contract fixes:
//!
//! 1. [`Draft::prompt`]: the draft reset, then the fed positions' features
//!    appended in order — every position's when the prompt is stepped, the
//!    last [`CardDraft::window`]'s when it is batched (skipped past the rest);
//! 2. per pass: the proposal after the token at the target's position, the
//!    target's verify of the pair, and the features of the rows it keeps
//!    appended before the rest is taken back.
//!
//! The proposal changes which passes run, never a token: every kept token is
//! the target's own argmax.
//!
//! The draft's card may be the target's own (the gate runs both on one card
//! under a card budget); the two then share the card's primary context.
//! Every call binds the draft's context and binds the target's back before it
//! returns, so the target's next launch never runs under the draft's context.
//! A device fault the draft raises comes back as the draft's
//! `GpuError::Fault` — from a proposal's readback, or from
//! [`CardDraft::check_fault`] after the last append — and ends the run: no
//! pass falls back to a plain step.

use std::sync::Arc;

use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::body::{self, Body};
use bloomery_gpu_deepseek41::draft::DraftBody;
use bloomery_gpu_deepseek41::draft::kv::GROUP;
use bloomery_gpu_deepseek41::draft::load::DraftWeights;
use cuda_core::CudaContext;
use gguf::Split;
use model::arch::deepseek41::spec::draft_of;
use model::arch::dspark::DraftHparams;
use runtime::{Draft, Program, TapNeed, Tapped, Target};

use crate::{Loaded, Session, SessionError};

const WHAT: &str = "dspark loop";

/// A draft body `D` on its own card, and the target's context to hand back.
pub struct CardDraft<D> {
    /// Declared first: dropped before its card's `Gpu`.
    body: D,
    gpu: Gpu,
    target: Arc<CudaContext>,
    card: &'static str,
    /// Prompt features waiting for a whole group, `GROUP` rows at most.
    pending: Vec<f32>,
    width: usize,
}

impl CardDraft<DraftBody> {
    /// The draft of `draft` on card `card`, borrowing `target_file`'s head and
    /// embedding rows, and the loaded target's feature tap built for the
    /// draft's target layers — before the target captures anything, which
    /// [`Loaded`] guarantees. Load-time only.
    pub fn open(
        target: &mut Loaded<Body>,
        draft: &Split,
        hp: &DraftHparams,
        target_file: Arc<Split>,
        card: &'static str,
    ) -> Result<CardDraft<DraftBody>, SessionError> {
        let m = target.model_mut();
        let desc = draft_of(hp)
            .map_err(|e| SessionError::Refused(format!("the draft's description: {e}")))?;
        let Program::Block(spec) = runtime::program(&desc)
            .map_err(|e| SessionError::Refused(format!("the draft's description: {e}")))?;
        let taps: Vec<usize> = spec.target_layers.iter().map(|&l| l as usize).collect();
        body::attach_features(m, &taps)?;
        let ctx = m.gpu().context().clone();
        let width = m.body(WHAT)?.feature_width();
        let gpu = Gpu::for_card(card)?;
        let loaded = (|| {
            let w = DraftWeights::load(gpu.stream(), draft, &target_file)?;
            DraftBody::new(&gpu, w, target_file)
        })();
        ctx.bind_to_thread().map_err(GpuError::from)?;
        let body = loaded?;
        let reads = body.weights().hp().target_layers.len() * body.weights().hp().n_embd;
        if reads != width {
            return Err(SessionError::Refused(format!(
                "the draft reads {reads} features a row, the target's tap writes {width}"
            )));
        }
        Ok(CardDraft {
            body,
            gpu,
            target: ctx,
            card,
            pending: Vec::with_capacity(GROUP * width),
            width,
        })
    }

    /// The card the draft runs on.
    #[must_use]
    pub fn card(&self) -> &'static str {
        self.card
    }

    /// `(free, total)` device bytes of the draft's card.
    pub fn mem_info(&mut self) -> Result<(usize, usize), GpuError> {
        self.on_card(|gpu, _| gpu.mem_info())
    }

    /// Positions appended so far.
    #[must_use]
    pub fn committed(&self) -> u32 {
        self.body.committed()
    }

    /// The draft's window, from its file: the trailing positions of a prompt
    /// whose features it keeps.
    #[must_use]
    pub fn window(&self) -> usize {
        self.body.weights().hp().window
    }

    /// Commit the positions before `to` without their features, the queued
    /// ones appended first: a batched prompt hands over only the rows the
    /// draft's window keeps.
    fn skip_to(&mut self, to: u32) -> Result<(), GpuError> {
        if to > self.body.committed() + (self.pending.len() / self.width) as u32 {
            self.flush()?;
            self.body.skip(to)?;
        }
        Ok(())
    }

    /// `f` under the draft's context, the target's bound again after it.
    fn on_card<T>(
        &mut self,
        f: impl FnOnce(&Gpu, &mut DraftBody) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        self.gpu.context().bind_to_thread()?;
        let r = f(&self.gpu, &mut self.body);
        self.target.bind_to_thread()?;
        r
    }

    /// Forget the sequence.
    fn reset(&mut self) -> Result<(), GpuError> {
        self.pending.clear();
        self.on_card(|gpu, body| body.reset(gpu))
    }

    /// Queue a fed position's features; a whole group is appended at once,
    /// through its captured graph. [`CardDraft::flush`] appends the rest.
    fn feed(&mut self, feats: &[f32]) -> Result<(), GpuError> {
        self.pending.extend_from_slice(feats);
        if self.pending.len() == GROUP * self.width {
            self.flush()?;
        }
        Ok(())
    }

    /// Append the queued fed positions.
    fn flush(&mut self) -> Result<(), GpuError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let rows = std::mem::take(&mut self.pending);
        let r = self.append(&rows);
        self.pending = rows;
        self.pending.clear();
        r
    }

    /// Append the features of the next committed positions (whole rows).
    fn append(&mut self, feats: &[f32]) -> Result<(), GpuError> {
        self.on_card(|gpu, body| body.append(gpu, feats).map(|_| ()))
    }

    /// The draft card's fault word, read after the last append: raised is
    /// the draft's `GpuError::Fault`.
    pub fn check_fault(&mut self) -> Result<(), GpuError> {
        match self.on_card(|gpu, _| gpu.fault())? {
            None => Ok(()),
            Some(fault) => Err(GpuError::fault(WHAT, fault)),
        }
    }
}

/// One id a proposal: the V4.1 pair pass verifies two rows.
impl Draft<Session<Body>> for CardDraft<DraftBody> {
    const WIDTH: usize = 1;
    const TAPS: TapNeed = TapNeed::Layers;

    /// The draft reset, the prompt through the target's tapped call, each
    /// position's features into the draft as the call hands them over, and
    /// the rest appended.
    fn prompt(&mut self, t: &mut Session<Body>, ids: &[u32]) -> Result<u32, SessionError> {
        self.reset()?;
        let (window, width) = (self.window(), self.width);
        let next = t.prompt_tapped(ids, window, &mut |first, rows| {
            self.skip_to(first)?;
            rows.chunks_exact(width).try_for_each(|row| self.feed(row))
        })?;
        self.flush()?;
        Ok(next)
    }

    fn begin(
        &mut self,
        _t: &Session<Body>,
        _prompt: &[u32],
        _first: u32,
    ) -> Result<(), SessionError> {
        Ok(())
    }

    /// The draft's id after `last`, the token at its committed position,
    /// which must be the target's.
    fn propose(
        &mut self,
        t: &mut Session<Body>,
        last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        let pos = t.pos();
        if self.committed() != pos {
            return Err(SessionError::Refused(format!(
                "dspark loop: the draft has {} positions, the target stands at {pos}",
                self.committed()
            )));
        }
        let width = <Self as Draft<Session<Body>>>::WIDTH;
        let ids = self.on_card(|gpu, body| body.propose(gpu, last, width))?;
        match ids.as_slice() {
            &[d] => {
                out[0] = d;
                Ok(1)
            }
            other => Err(SessionError::Gpu(GpuError::Shape {
                what: WHAT,
                detail: format!("a proposal of width {width} returned {} ids", other.len()),
            })),
        }
    }

    /// The kept rows' features appended, read before the rest is taken back.
    fn accept(
        &mut self,
        t: &mut Session<Body>,
        _rows: &[u32],
        _out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        let feats = t.taps(accepted)?;
        Ok(self.append(feats)?)
    }

    /// Never reached: every pass proposes.
    fn stepped(
        &mut self,
        _t: &mut Session<Body>,
        _last: u32,
        _next: u32,
    ) -> Result<(), SessionError> {
        Err(SessionError::Refused(
            "dspark loop: a pass stepped without a proposal, and the draft has no features for it"
                .into(),
        ))
    }
}
