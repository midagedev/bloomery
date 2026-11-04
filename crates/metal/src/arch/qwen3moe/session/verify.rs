//! The verify pass of a session: `m` ids, two to [`VERIFY_ROWS`], as one pass
//! whose row `r` is the bits of the step at position `pos + r`, and the commit
//! that keeps the first rows of it.
//!
//! The walk is the session's own ([`Walk`] over the chain plan) with the
//! verify's arms of the ops ([`VerifyOps`]) in place of the pass's: the M-row
//! mat-vecs, the decode flash's rows launch and the delta rule that leaves
//! the lane ring, so a row of the verify and the step at its position leave
//! the same logits, the same pick and the same keys, values and delta state
//! (`ops/verify.rs` says which launch is which). It is one command buffer,
//! committed once and waited once, as the step's is: every group closes with
//! a barrier, so a buffer boundary between layers would order nothing more and
//! leave the card idle across each.
//!
//! A verify stands the session `m` positions on and leaves it pending: the
//! position is `pos + m` and the delta state of row `r`, the lane of position
//! `pos + r + 1`, holds what the row left. [`Session::commit_rows`] keeps the
//! first `accepted` of the rows by standing the session at `pos + accepted`,
//! where the lane of that position is the state after row `accepted - 1`: a
//! position moves, and no launch runs and no byte is copied. The keys and
//! values of the rows past it are rows no position reads until a step at their
//! position appends over them.
//!
//! The token ahead of a chain step is waited out and dropped when a verify
//! starts, as a pass drops it: it is the token at `pos` embedding the pick
//! its predecessor made, and a verify runs that position itself as its row 0,
//! with the id its caller names. A step followed by a verify therefore spends
//! one token's device time on the dropped token ([`StepStats::discarded`]).
//!
//! Until [`Session::commit_rows`] a step, a pass, another verify, a cut, a
//! checkpoint, a probe and a read of the logits row are refused by name, and
//! the refused call leaves the verify committable; a reset ends it.
//!
//! Two instruments run the same verify and leave it as committable as a verify
//! does. [`Session::verify_profile`] puts each group of the walk on a command
//! buffer of its own, waited, its GPU seconds read, as the step's profile
//! does ([`Session::step_profile`]): the groups' own times, none of the
//! buffers' overlap. [`Session::verify_traced`] runs the verify as it is run
//! and stamps its buffer on the host (begin, commit) and on the GPU (start,
//! end) on one clock, beside what the buffer encoded ([`VerifyTrace`]): what
//! the host's encode and the card's wake add to the groups' times.

use core::sync::atomic::Ordering;

use bloomery_device::{Elems, Stream};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, FfnRoute, GqaPlan};
use chain::arch::qwen3moe::walk::Walk;

use super::super::arena::{Arena, MixerBufs, slot_of};
use super::super::count::GroupKind;
use super::super::lanes::{self, VERIFY_ROWS};
use super::super::ops::{Half, Part, VerifyOps, serves};
use super::super::rows::{RowsArena, VerifyBufs};
use super::{
    Arms, LayerBuffers, Layered, ProfileGroup, STAMP_WAIT, Session, SessionError, Traced, refuse,
};
use crate::encode::{ComputeEncoder, EncodeCount};
use crate::kernels::{Kernels, headnorm_m_serves};
use crate::residency::Alloc;
use crate::stream::{GpuInterval, MetalStream, Settled, host_secs};

/// The state a verify's ops ([`VerifyOps`]) are made from, in the walk's order
/// on one command buffer ([`LayerBuffers::whole`]), the embed and the head with
/// the layers' groups.
struct VerifyArms<'s> {
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    rows: &'s mut RowsArena,
    from: u32,
}

impl Arms for VerifyArms<'_> {
    fn with_ops<R>(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        f: impl FnOnce(&mut dyn ChainOps<Error = SessionError>) -> R,
    ) -> R {
        f(&mut VerifyOps::new(
            enc,
            self.arena,
            self.rows,
            self.kernels,
            self.from,
        ))
    }
}

/// What a verify settles before its walk, for its end.
struct Started {
    /// The position of row 0.
    from: u32,
    /// The rows.
    m: usize,
    /// The commit's plan.
    plan: lanes::Verify,
    /// The step record slot of the last row, where the verify's fault lands.
    slot: usize,
}

/// The profiled verify's ops: each group of [`VerifyOps`] on a command buffer
/// of its own, waited, its GPU seconds read, one sample appended. A buffer
/// opens at the stream's token boundary, so each group sees what the one
/// before it wrote.
struct VerifyProfiler<'s> {
    stream: &'s MetalStream,
    kernels: &'s Kernels,
    arena: &'s mut Arena,
    rows: &'s mut RowsArena,
    from: u32,
    /// The last group's settle: the head's, once the walk is done.
    settled: Option<Settled>,
    out: &'s mut Vec<ProfileGroup>,
}

impl VerifyProfiler<'_> {
    /// `kind` of `layer` (`part` of it, for a group a row runs or the hand-on
    /// after a half) over the verify's `m` rows, alone on a command buffer.
    fn one(
        &mut self,
        kind: GroupKind,
        layer: Option<usize>,
        m: usize,
        part: Part,
    ) -> Result<(), SessionError> {
        let mut recording = self.stream.begin()?;
        let mut enc = recording.token_encoder()?;
        VerifyOps::new(&mut enc, self.arena, self.rows, self.kernels, self.from).group(
            kind,
            layer.unwrap_or(0),
            m,
            part,
        )?;
        enc.end();
        let stamps = recording.stamps();
        let signalled = self.stream.submit_signalled(recording)?;
        self.settled = Some(self.stream.settle(signalled)?);
        // The settle above waited the buffer out, so its stamps are the
        // driver's record of a completed run; anything else is a refusal.
        let gpu_s = match stamps.interval(STAMP_WAIT)? {
            GpuInterval::Ran(s) => s,
            GpuInterval::StillRunning => {
                return Err(refuse(format!(
                    "{kind:?} layer {layer:?}: its buffer still runs after its settle"
                )));
            }
        };
        self.out.push(ProfileGroup { kind, layer, gpu_s });
        Ok(())
    }

    /// The hand-on group after `half` of layer `l`, when the layer hands on:
    /// a layer that does not has no such group.
    fn hand_on(&mut self, l: LayerCtx<'_>, half: Half) -> Result<(), SessionError> {
        if !self.arena.hands_on(l.layer) {
            return Ok(());
        }
        self.one(GroupKind::HandOn, Some(l.layer), l.m, Part::Half(half))
    }
}

impl ChainOps for VerifyProfiler<'_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(GroupKind::Embed, None, m, Part::Whole)
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnP1, Some(l.layer), l.m, Part::Whole)?;
        if g.bias.is_some() {
            self.one(GroupKind::AttnBias, Some(l.layer), l.m, Part::Whole)?;
        }
        Ok(())
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::AttnFlash, Some(l.layer), l.m, Part::Whole)
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyProfiler::attn_out")?;
        self.one(GroupKind::AttnOut, Some(l.layer), l.m, Part::Whole)?;
        self.hand_on(l, Half::Mixer)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaP1, Some(l.layer), l.m, Part::Whole)
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        self.one(GroupKind::DeltaCore, Some(l.layer), l.m, Part::Whole)
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyProfiler::delta_out")?;
        // The delta out is one group where the M-row head-normed entry serves
        // the layer's out type at the verify's rows (the other M-row launches'
        // own shape), else a group a row, each launch timed on its own buffer.
        let out = match &self.arena.layers[l.layer].mixer {
            MixerBufs::Delta(d) => d.out_quant,
            MixerBufs::Attention(_) | MixerBufs::ShortConv(_) => {
                return Err(refuse(format!(
                    "layer {}: a delta op on a layer of another mixer",
                    l.layer
                )));
            }
        };
        if headnorm_m_serves(out, l.m) {
            self.one(GroupKind::DeltaOut, Some(l.layer), l.m, Part::Whole)?;
        } else {
            for row in 0..l.m {
                self.one(GroupKind::DeltaOut, Some(l.layer), l.m, Part::Row(row))?;
            }
        }
        self.hand_on(l, Half::Mixer)
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        match f.route {
            // A dense FFN has no router and no group of its own.
            FfnRoute::Dense => Ok(()),
            FfnRoute::Router { .. } => Err(refuse(format!(
                "layer {}: a routed FFN over a verify, whose experts have no M-slot entry",
                l.layer
            ))),
        }
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyProfiler::experts")?;
        match f.route {
            FfnRoute::Dense => {
                self.one(GroupKind::GateUp, Some(l.layer), l.m, Part::Whole)?;
                self.one(GroupKind::Down, Some(l.layer), l.m, Part::Whole)?;
                self.hand_on(l, Half::Ffn)
            }
            FfnRoute::Router { .. } => Err(refuse(format!(
                "layer {}: a routed FFN over a verify, whose experts have no M-slot entry",
                l.layer
            ))),
        }
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.one(GroupKind::Head, None, m, Part::Whole)
    }
}

/// One command buffer of a traced verify, stamped by the host and by the GPU
/// on one clock: seconds on the host clock the GPU's stamps read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BufferStamp {
    /// When the host began the buffer.
    pub opened: f64,
    /// When the host's commit of it returned.
    pub committed: f64,
    /// The buffer's `GPUStartTime`.
    pub start: f64,
    /// The buffer's `GPUEndTime`.
    pub end: f64,
}

impl BufferStamp {
    /// The host's seconds from beginning the buffer to its commit's return.
    #[must_use]
    pub fn encode(&self) -> f64 {
        self.committed - self.opened
    }

    /// The GPU's seconds on the buffer.
    #[must_use]
    pub fn gpu(&self) -> f64 {
        self.end - self.start
    }

    /// The seconds from the host's commit to the GPU's end of the buffer.
    #[must_use]
    pub fn commit_to_end(&self) -> f64 {
        self.end - self.committed
    }
}

/// What one traced verify did ([`Session::verify_traced`]): the host's clock
/// at the call and at the wake of its one wait, the host's waits between
/// them, the buffers in commit order and what they encoded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VerifyTrace {
    /// The host clock when the call began.
    pub called: f64,
    /// The host clock when the call's wait on the last buffer returned.
    pub woke: f64,
    /// The host's waits on the stream from the walk's first buffer to the
    /// wake: the settle of the last buffer, and nothing before it.
    pub waits: u64,
    /// The buffers: one for a verify.
    pub buffers: Vec<BufferStamp>,
    /// What the buffers encoded, summed: the groups, dispatches and barriers
    /// of the verify.
    pub count: EncodeCount,
}

/// A verify's wall split by what ran in it ([`VerifyTrace::summary`]): the
/// terms add to `wall` (`lead + gpu + gaps + tail`), with the gaps signed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceSummary {
    /// The buffers.
    pub buffers: usize,
    /// The call's seconds from entry to the wake.
    pub wall: f64,
    /// From the call to the first buffer's GPU start: the refusals' checks,
    /// the token ahead's wait, the whole verify's encode and the commit to
    /// start.
    pub lead: f64,
    /// The GPU's seconds on the buffers, summed.
    pub gpu: f64,
    /// The GPU's seconds between one buffer's end and the next's start,
    /// summed; negative where a buffer started before the one before it ended.
    pub gaps: f64,
    /// From the last buffer's end to the wake.
    pub tail: f64,
    /// The longest gap.
    pub gap_max: f64,
    /// The buffers that started before the one before them ended.
    pub overlapped: usize,
    /// The buffers whose commit returned after the one before them ended: the
    /// GPU had no next buffer to run, so the host was the pace there.
    pub host_late: usize,
    /// Those buffers' seconds between the end before and their commit.
    pub host_late_s: f64,
    /// The host's seconds encoding and committing the buffers, summed.
    pub encode: f64,
    /// The longest buffer's encode and commit.
    pub encode_max: f64,
}

impl VerifyTrace {
    /// The trace of a settled walk: `traced` buffers' GPU stamps read now.
    fn read(
        called: f64,
        woke: f64,
        waits: u64,
        traced: Vec<Traced>,
        count: EncodeCount,
    ) -> Result<Self, SessionError> {
        let buffers = traced
            .into_iter()
            .map(|t| {
                let times = t.stamps.times(STAMP_WAIT)?;
                Ok(BufferStamp {
                    opened: t.opened,
                    committed: t.committed,
                    start: times.start,
                    end: times.end,
                })
            })
            .collect::<Result<Vec<_>, SessionError>>()?;
        Ok(Self {
            called,
            woke,
            waits,
            buffers,
            count,
        })
    }

    /// The wall split by what ran; `None` for a trace of no buffer.
    #[must_use]
    pub fn summary(&self) -> Option<TraceSummary> {
        let (first, last) = (self.buffers.first()?, self.buffers.last()?);
        let mut s = TraceSummary {
            buffers: self.buffers.len(),
            wall: self.woke - self.called,
            lead: first.start - self.called,
            gpu: 0.0,
            gaps: 0.0,
            tail: self.woke - last.end,
            gap_max: f64::NEG_INFINITY,
            overlapped: 0,
            host_late: 0,
            host_late_s: 0.0,
            encode: 0.0,
            encode_max: 0.0,
        };
        let mut before: Option<f64> = None;
        for b in &self.buffers {
            s.gpu += b.gpu();
            s.encode += b.encode();
            s.encode_max = s.encode_max.max(b.encode());
            if let Some(end) = before {
                let gap = b.start - end;
                s.gaps += gap;
                s.gap_max = s.gap_max.max(gap);
                s.overlapped += usize::from(gap < 0.0);
                if b.committed > end {
                    s.host_late += 1;
                    s.host_late_s += b.committed - end;
                }
            }
            before = Some(b.end);
        }
        if s.buffers == 1 {
            s.gap_max = 0.0;
        }
        Some(s)
    }
}

impl Session {
    /// Whether the verify pass serves this session's file: every layer's
    /// mixer, projections and FFN, and the head. What it does not serve is
    /// refused by name, the layer and the reason; a verify refuses the same
    /// before it encodes anything.
    ///
    /// # Errors
    ///
    /// [`SessionError::Shape`] naming the first thing the pass cannot run.
    pub fn verify_serves(&self) -> Result<(), SessionError> {
        serves(&self.arena)
    }

    /// The verify that waits for its commit: the position of its row 0 and
    /// its rows.
    #[must_use]
    pub fn verify_pending(&self) -> Option<(u32, usize)> {
        self.pending.map(|p| (p.pos0(), p.rows()))
    }

    /// `Ok` when no verify waits for its commit, else the refusal of `what`
    /// by name.
    pub(super) fn not_pending(&self, what: &str) -> Result<(), SessionError> {
        match self.pending {
            None => Ok(()),
            Some(p) => Err(refuse(format!(
                "{what} at position {}: a verify of {} rows from position {} waits for its \
                 commit, which keeps 1 to {} of them",
                self.pos,
                p.rows(),
                p.pos0(),
                p.rows()
            ))),
        }
    }

    /// `ids` as one verify at the next positions: row `r` is `ids[r]` at
    /// position `pos + r`, and `picks[r]` is the head's pick after it. Every
    /// row is the bits of the step at its position, the logits of a row
    /// readable until the commit ([`Session::verify_row_logits`]). The session
    /// stands `ids.len()` positions on, pending its [`Session::commit_rows`].
    ///
    /// Refused by name, with nothing run and the session as it was: fewer
    /// than two ids or more than [`VERIFY_ROWS`], a verify while another
    /// waits for its commit or the session is poisoned, rows past the cache
    /// or the 32-bit position, and a file whose layers the pass does not serve
    /// ([`Session::verify_serves`]). A token running ahead is waited out and
    /// dropped. A refusal in the walk comes before the verify's one commit:
    /// nothing ran and the stores stay as they were. A failure at the settle
    /// leaves the stores as the buffer left them and the session unmoved: the
    /// caller resets.
    pub(in super::super) fn verify_rows(
        &mut self,
        ids: &[u32],
        picks: &mut [u32],
    ) -> Result<(), SessionError> {
        self.verify_layered(ids, picks, None)
    }

    /// [`Session::verify_rows`] with its command buffer stamped: `trace` is
    /// replaced by what this verify's buffer did on the host and the GPU and
    /// what it encoded. The verify is the one [`Session::verify_rows`] runs,
    /// and it waits for its commit as that does; the clock reads and the
    /// stamps' handle are all the trace adds.
    ///
    /// # Errors
    ///
    /// As [`Session::verify_rows`], and [`SessionError::Device`] when a
    /// buffer's GPU stamps cannot be read; the verify has run and waits for
    /// its commit then.
    pub fn verify_traced(
        &mut self,
        ids: &[u32],
        picks: &mut [u32],
        trace: &mut VerifyTrace,
    ) -> Result<(), SessionError> {
        self.verify_layered(ids, picks, Some(trace))
    }

    /// [`Session::verify_rows`] with each group of the walk on a command
    /// buffer of its own, waited before the next is committed, its GPU seconds
    /// read: one sample a group in the walk's order, appended to `out`
    /// (cleared first). The delta layers' out projection runs a row at a time,
    /// so it is a group a row. The groups are the unprofiled verify's, in its
    /// order, so it leaves the same bits and the same pending commit; a
    /// refusal or a failure inside the walk leaves the stores as the groups
    /// that ran left them, and the caller resets.
    ///
    /// # Errors
    ///
    /// As [`Session::verify_rows`].
    pub fn verify_profile(
        &mut self,
        ids: &[u32],
        picks: &mut [u32],
        out: &mut Vec<ProfileGroup>,
    ) -> Result<(), SessionError> {
        out.clear();
        let started = self.verify_start(ids, picks)?;
        let cap = self.rows.cap;
        let mut ops = VerifyProfiler {
            stream: &self.stream,
            kernels: &self.kernels,
            arena: &mut self.arena,
            rows: &mut self.rows,
            from: started.from,
            settled: None,
            out,
        };
        let walked = Walk::new(&mut ops, &self.plans, started.m).run(cap);
        let settled = ops.settled.take();
        if let Err(e) = walked {
            if settled.is_some() {
                let _ = self.stream.sync();
            }
            return Err(e);
        }
        let settled = settled.ok_or_else(|| refuse("a profiled verify settled no group"))?;
        self.verify_end(&started, &settled, picks)
    }

    /// The verify as the session runs it, one command buffer, stamping it when
    /// `trace` is given.
    fn verify_layered(
        &mut self,
        ids: &[u32],
        picks: &mut [u32],
        trace: Option<&mut VerifyTrace>,
    ) -> Result<(), SessionError> {
        let called = host_secs();
        let started = self.verify_start(ids, picks)?;
        let waits = self.stream.waits();
        let cap = self.rows.cap;
        let buffers = LayerBuffers::whole(&self.stream);
        let mut ops = Layered {
            buffers: if trace.is_some() {
                buffers.traced()
            } else {
                buffers
            },
            arms: VerifyArms {
                kernels: &self.kernels,
                arena: &mut self.arena,
                rows: &mut self.rows,
                from: started.from,
            },
        };
        let walked = Walk::new(&mut ops, &self.plans, started.m)
            .run(cap)
            .and_then(|()| ops.buffers.finish());
        let LayerBuffers {
            last,
            count,
            trace: traced,
            ..
        } = ops.buffers;
        // A refusal in the walk comes before the one commit: nothing ran.
        walked?;
        let signalled = last.ok_or_else(|| refuse("a verify committed no buffer"))?;
        let settled = self.stream.settle(signalled)?;
        let woke = host_secs();
        let waited = self.stream.waits() - waits;
        self.verify_end(&started, &settled, picks)?;
        if let (Some(trace), Some(traced)) = (trace, traced) {
            *trace = VerifyTrace::read(called, woke, waited, traced, count)?;
        }
        Ok(())
    }

    /// Everything a verify settles before its walk: the refusals by name, the
    /// token ahead dropped, the verify's buffers open, the ids written.
    fn verify_start(&mut self, ids: &[u32], picks: &[u32]) -> Result<Started, SessionError> {
        self.ready()?;
        let m = ids.len();
        if !(2..=VERIFY_ROWS).contains(&m) || picks.len() != m {
            return Err(refuse(format!(
                "a verify of {m} ids into {} picks; a verify takes 2 to {VERIFY_ROWS}",
                picks.len()
            )));
        }
        let from = self.pos;
        let plan = lanes::Verify::at(from, m).map_err(|e| refuse(e.to_string()))?;
        if from as usize + m > self.arena.ctx {
            return Err(refuse(format!(
                "a verify of {m} ids at position {from}: the cache holds {} rows",
                self.arena.ctx
            )));
        }
        self.verify_serves()?;
        if let Some(a) = self.ahead.take() {
            self.drop_ahead(a)?;
        }
        self.open_verify_buffers()?;
        let slot =
            slot_of(from + u32::try_from(m).map_err(|_| refuse("a verify's rows fit u32"))? - 1);
        self.rows.ids.write_at(Elems(0), ids)?;
        Ok(Started {
            from,
            m,
            plan,
            slot,
        })
    }

    /// A settled verify's end: its fault word (a fault poisons the session),
    /// its picks, the position moved and the commit pending.
    fn verify_end(
        &mut self,
        started: &Started,
        settled: &Settled,
        picks: &mut [u32],
    ) -> Result<(), SessionError> {
        let m = started.m;
        self.stats.passes += 1;
        if let Some(fault) = self.arena.faults[started.slot].read_settled(settled)? {
            self.poisoned = Some(fault);
            return Err(SessionError::Fault(fault));
        }
        let verify = self
            .rows
            .verify
            .as_ref()
            .ok_or_else(|| refuse("a verify settled with no verify buffers open"))?;
        for (r, pick) in picks.iter_mut().enumerate() {
            *pick = verify.picks.pages().load(r, Ordering::Relaxed)?;
        }
        self.pos =
            started.from + u32::try_from(m).map_err(|_| refuse("a verify's rows fit u32"))?;
        self.pending = Some(started.plan);
        Ok(())
    }

    /// Keep the first `accepted` rows of the verify that waits: the session
    /// stands at `pos0 + accepted`, the position whose lane holds the delta
    /// state after row `accepted - 1`. No launch runs and no byte is copied.
    ///
    /// Refused by name, the verify still waiting, for none or for `accepted`
    /// of 0 or past its rows; and when no verify waits.
    pub(in super::super) fn commit_rows(&mut self, accepted: usize) -> Result<(), SessionError> {
        let Some(plan) = self.pending else {
            return Err(refuse(format!(
                "a commit of {accepted} rows at position {}: no verify waits for its commit",
                self.pos
            )));
        };
        let kept = plan.commit(accepted).map_err(|e| refuse(e.to_string()))?;
        self.pos = kept.pos;
        self.pending = None;
        Ok(())
    }

    /// Row `r`'s logits of the verify that waits for its commit: the head's
    /// own pages, zero copy, valid until the commit. Refused by name when no
    /// verify waits and for a row it did not run.
    ///
    /// # Errors
    ///
    /// [`SessionError::Shape`] naming the refusal; [`SessionError::Device`]
    /// when the stream's wait fails.
    pub fn verify_row_logits(&self, r: usize) -> Result<&[f32], SessionError> {
        let Some(plan) = self.pending else {
            return Err(refuse(format!(
                "the logits of verify row {r}: no verify waits for its commit"
            )));
        };
        if r >= plan.rows() {
            return Err(refuse(format!(
                "the logits of row {r} of a verify of {} rows",
                plan.rows()
            )));
        }
        let verify = self
            .rows
            .verify
            .as_ref()
            .ok_or_else(|| refuse("a pending verify with no verify buffers open"))?;
        self.stream.sync()?;
        let vocab = self.arena.vocab;
        // SAFETY: the verify settled before it returned and nothing runs
        // until its commit (a step, a pass and a second verify are refused
        // while it waits), so no device work writes these pages; the row
        // borrows the session, whose every launch goes through `&mut self`.
        let all = unsafe { verify.logits.pages().as_slice() };
        all.get(r * vocab..(r + 1) * vocab).ok_or_else(|| {
            refuse(format!(
                "the logits of verify row {r} lie past their buffer"
            ))
        })
    }

    /// The verify's own buffers, opened by the first verify: they join the
    /// stream's residency set as the pass's snapshot copies do.
    fn open_verify_buffers(&mut self) -> Result<(), SessionError> {
        if self.rows.verify.is_some() {
            return Ok(());
        }
        let residency = self
            .stream
            .residency()
            .ok_or_else(|| refuse("the stream has no residency set"))?;
        let staged = residency.stage();
        let bufs = VerifyBufs::open(Alloc::new(&self.device, &staged), &self.arena, VERIFY_ROWS)?;
        staged.commit();
        self.rows.verify = Some(bufs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(opened: f64, committed: f64, start: f64, end: f64) -> BufferStamp {
        BufferStamp {
            opened,
            committed,
            start,
            end,
        }
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    /// Four buffers: the second starts after a gap, the third's commit comes
    /// after the second ended, the fourth starts before the third ended.
    fn trace() -> VerifyTrace {
        VerifyTrace {
            called: 0.0,
            woke: 0.052,
            waits: 1,
            buffers: vec![
                stamp(0.001, 0.002, 0.003, 0.013),
                stamp(0.002, 0.004, 0.0135, 0.0235),
                stamp(0.014, 0.030, 0.031, 0.041),
                stamp(0.005, 0.010, 0.0405, 0.0505),
            ],
            ..VerifyTrace::default()
        }
    }

    /// The summary's lead, GPU seconds, gaps and tail add to the call's wall,
    /// the gaps signed: the wall is accounted for with nothing left over.
    ///
    /// Mutant: the lead read from the first buffer's end, a gap taken
    /// unsigned or the tail from the last buffer's start.
    #[test]
    fn the_terms_of_a_traced_wall_add_to_it() {
        let s = trace().summary().expect("a trace of four buffers");
        assert_eq!(s.buffers, 4);
        assert!(near(s.wall, 0.052));
        assert!(near(s.lead, 0.003));
        assert!(near(s.gpu, 0.040));
        assert!(near(s.gaps, 0.0075));
        assert!(near(s.tail, 0.0015));
        assert!(near(s.lead + s.gpu + s.gaps + s.tail, s.wall));
    }

    /// The overlap, the late host and the encode are counted from the
    /// buffers' own stamps: a buffer that began before the one before it
    /// ended, one committed after it ended, and the host's seconds a buffer.
    ///
    /// Mutant: a late commit judged against the buffer's own start, an
    /// equal clock counted as late, or the encode taken from the start.
    #[test]
    fn the_overlap_the_late_host_and_the_encode_are_counted() {
        let s = trace().summary().expect("a trace of four buffers");
        assert_eq!(s.overlapped, 1);
        assert!(near(s.gap_max, 0.0075));
        assert_eq!(s.host_late, 1);
        assert!(near(s.host_late_s, 0.0065));
        assert!(near(s.encode, 0.001 + 0.002 + 0.016 + 0.005));
        assert!(near(s.encode_max, 0.016));
        let mut t = trace();
        // A commit that returned the instant the buffer before ended is on
        // time.
        t.buffers[2].committed = t.buffers[1].end;
        assert_eq!(t.summary().expect("a trace").host_late, 0);
    }

    /// A trace of no buffer has no summary, and one buffer has no gap.
    #[test]
    fn a_trace_of_no_buffer_has_no_summary_and_one_has_no_gap() {
        assert_eq!(VerifyTrace::default().summary(), None);
        let one = VerifyTrace {
            called: 1.0,
            woke: 1.5,
            waits: 1,
            buffers: vec![stamp(1.1, 1.2, 1.3, 1.4)],
            ..VerifyTrace::default()
        };
        let s = one.summary().expect("a trace of one buffer");
        assert!(near(s.gaps, 0.0) && near(s.gap_max, 0.0));
        assert_eq!((s.overlapped, s.host_late), (0, 0));
        assert!(near(s.lead + s.gpu + s.gaps + s.tail, s.wall));
    }
}
