//! A DSpark draft on a card of its own, for any target that taps its rows:
//! [`CardDspark`] drives the common owner ([`DsparkBody`]) over a family's
//! layers, fed the target's tapped rows in the order the owner's sequence
//! fixes:
//!
//! 1. [`Draft::prompt`]: the draft reset, then the fed positions' rows
//!    appended in order — every position's when the prompt is stepped, the
//!    last window's when it is batched (skipped past the rest);
//! 2. per pass: the proposal after the token at the target's position, the
//!    target's verify of its rows, and the rows it keeps appended before the
//!    rest is taken back; a step's one row after it.
//!
//! The proposal changes which passes run, never a token: every kept token is
//! the target's own argmax.
//!
//! A server keeps one target across requests, which cut it back and put saved
//! states into it. The draft then follows the target only while every
//! position the target ran reached it: [`CardDspark::forget`] marks a target
//! moved elsewhere, and [`CardDspark::feed_call`] continues a draft that
//! follows and starts over one that does not, the positions before the call
//! skipped. A draft started over proposes only once its window's rows are all
//! appended: a call that starts past position 0 must carry a whole window.
//!
//! The draft's card may be the target's own (the gate runs both on one card
//! under a card budget); the two then share the card's primary context. Every
//! call binds the draft's context and binds the target's back before it
//! returns, so the target's next launch never runs under the draft's context.
//! A device fault the draft raises comes back as the draft's
//! `GpuError::Fault` — from a proposal's readback, or from
//! [`CardDspark::check_fault`] after the last append — and ends the run: no
//! pass falls back to a plain step.

use std::sync::Arc;

use bloomery_gpu::dspark::{CudaContext, DraftLayers, DsparkBody};
use bloomery_gpu::model::ChainBody;
use bloomery_gpu::{Gpu, GpuError};
use model::placement::workstation::DeviceId;
use runtime::dspark::DsparkShape;
use runtime::{Draft, TapNeed, Tapped, TappedPrompt, Target, Verify};

use crate::{Session, SessionError};

const WHAT: &str = "dspark";

/// A DSpark draft body over `L` on its own card, and the target's context to
/// hand back.
pub struct CardDspark<L: DraftLayers> {
    /// Declared first: dropped before its card's `Gpu`.
    body: DsparkBody<L>,
    gpu: Gpu,
    target: Arc<CudaContext>,
    card: &'static str,
    shape: DsparkShape,
    /// Fed rows waiting for a whole group, `L::APPEND_GROUP` rows at most.
    pending: Vec<f32>,
    /// Floats of one tapped row.
    width: usize,
    /// Every position the target ran since the draft's reset reached it.
    follows: bool,
    /// The device bytes the draft's load took on its card.
    resident: usize,
}

impl<L: DraftLayers> CardDspark<L> {
    /// The draft `body` on `gpu`, its card `card`, drafting for a target `B`
    /// whose context is `target` and whose verify pass runs
    /// [`Verify::MAX_ROWS`] rows. `resident` is what the load took on the
    /// card. Refused by name when a proposal that fills that verify is a
    /// block wider than the layers hold, or the target's verify holds no
    /// proposal. `body` is built from `shape`; the target's taps are set at
    /// the first prompt call.
    pub fn new<B>(
        body: DsparkBody<L>,
        gpu: Gpu,
        target: Arc<CudaContext>,
        card: &'static str,
        shape: DsparkShape,
        resident: usize,
    ) -> Result<CardDspark<L>, SessionError>
    where
        B: ChainBody,
        Session<B>: Verify,
    {
        let verify_rows = <Session<B> as Verify>::MAX_ROWS;
        let width = shape.width(verify_rows);
        let rows = shape.rows(width);
        if width == 0 || rows > L::MAX_ROWS {
            return Err(SessionError::Refused(format!(
                "{WHAT}: the target verifies {verify_rows} rows, a proposal of {width} ids; its \
                 block of {rows} rows needs 1..={} (the layers' rows)",
                L::MAX_ROWS
            )));
        }
        let tap = shape.taps.len() * shape.hidden;
        Ok(CardDspark {
            body,
            gpu,
            target,
            card,
            shape,
            pending: Vec::with_capacity(L::APPEND_GROUP * tap),
            width: tap,
            follows: true,
            resident,
        })
    }

    /// The device bytes the draft's load took on its card: its weights,
    /// buffers and captured graphs, read as the card's free bytes before and
    /// after it.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.resident
    }

    /// The target now stands elsewhere than the positions the draft was fed
    /// (a cut, a saved state put back): the next [`CardDspark::feed_call`]
    /// starts the draft over.
    pub fn forget(&mut self) {
        self.follows = false;
    }

    /// Start the draft over at position 0, for a target that was reset.
    pub fn restart(&mut self) -> Result<(), GpuError> {
        self.reset()?;
        self.follows = true;
        Ok(())
    }

    /// The prompt call of `ids` from where `t` stands, each position's tapped
    /// row into the draft as the call hands it over: after the positions the
    /// draft holds when it follows `t` there, else after a reset that skips
    /// every position before the call. The target's taps are the draft's
    /// ([`TappedPrompt::set_taps`]), and a tap row of another width than the
    /// draft reads is refused by name. The argmax after the last id.
    pub fn feed_call<B>(&mut self, t: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError>
    where
        B: ChainBody,
        Session<B>: TappedPrompt<Error = SessionError>,
    {
        t.set_taps(&self.shape.taps)?;
        if t.tap_width() != self.width {
            return Err(SessionError::Refused(format!(
                "{WHAT}: the draft reads {} values a row, the target's tap writes {}",
                self.width,
                t.tap_width()
            )));
        }
        if !self.follows(t.pos()) {
            self.restart()?;
        }
        let (window, width) = (self.shape.window.unwrap_or(ids.len()), self.width);
        let next = t.prompt_tapped(ids, window, &mut |first, rows| {
            self.skip_to(first)?;
            rows.chunks_exact(width)
                .try_for_each(|row| self.feed(row))?;
            Ok(())
        })?;
        self.flush()?;
        Ok(next)
    }

    /// Whether the draft holds every position the target ran before `pos`,
    /// the target's position: it can propose there with no prompt call first.
    #[must_use]
    pub fn follows(&self, pos: u32) -> bool {
        self.follows && self.committed() == pos
    }

    /// Refused unless the draft holds every position before `pos`, the
    /// target's.
    fn following(&self, pos: u32) -> Result<(), SessionError> {
        if self.follows(pos) {
            return Ok(());
        }
        Err(SessionError::Refused(format!(
            "{WHAT}: the draft {} {} positions, the target stands at {pos}",
            if self.follows {
                "has"
            } else {
                "was left behind at"
            },
            self.committed()
        )))
    }

    /// The card the draft runs on.
    #[must_use]
    pub fn card(&self) -> &'static str {
        self.card
    }

    /// The device the draft runs on, as this process's census names it.
    pub fn device(&self) -> Result<DeviceId, GpuError> {
        self.gpu.device_id()
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
    /// whose rows it keeps, `None` for the whole context.
    #[must_use]
    pub fn window(&self) -> Option<usize> {
        self.shape.window
    }

    /// Commit the positions before `to` without their rows, the queued ones
    /// appended first: a batched prompt hands over only the rows the draft's
    /// window keeps.
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
        f: impl FnOnce(&Gpu, &mut DsparkBody<L>) -> Result<T, GpuError>,
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

    /// Queue a fed position's row; a whole group is appended at once, through
    /// its captured graph. [`CardDspark::flush`] appends the rest.
    fn feed(&mut self, row: &[f32]) -> Result<(), GpuError> {
        self.pending.extend_from_slice(row);
        if self.pending.len() == L::APPEND_GROUP * self.width {
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

    /// Append the tapped rows of the next committed positions (whole rows).
    fn append(&mut self, rows: &[f32]) -> Result<(), GpuError> {
        self.on_card(|gpu, body| {
            let first = body.committed();
            body.append_as(gpu, first, rows).map(|_| ())
        })
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

/// A proposal of up to the target's verify rows less one: the ids of the
/// block the layers hold.
impl<B, L> Draft<Session<B>> for CardDspark<L>
where
    B: ChainBody,
    Session<B>: TappedPrompt<Error = SessionError>,
    L: DraftLayers,
{
    const WIDTH: usize = <Session<B> as Verify>::MAX_ROWS - 1;
    const TAPS: TapNeed = TapNeed::Layers;

    /// The draft reset, then [`CardDspark::feed_call`].
    fn prompt(&mut self, t: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError> {
        self.restart()?;
        self.feed_call(t, ids)
    }

    fn begin(&mut self, _t: &Session<B>, _prompt: &[u32], _first: u32) -> Result<(), SessionError> {
        Ok(())
    }

    /// The draft's ids after `last`, the token at its committed position,
    /// which must be the target's.
    fn propose(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        self.propose_with(t, last, out, &mut [])
    }

    /// [`Draft::propose`] with each id's probability under the draft.
    fn propose_p(
        &mut self,
        t: &mut Session<B>,
        last: u32,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, SessionError> {
        self.propose_with(t, last, out, p)
    }

    /// The kept rows' taps appended, read before the rest is taken back.
    fn accept(
        &mut self,
        t: &mut Session<B>,
        _rows: &[u32],
        _out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        let rows = t.taps(accepted)?;
        Ok(self.append(rows)?)
    }

    /// The step's one row appended: the tap of `last`'s position.
    fn stepped(&mut self, t: &mut Session<B>, _last: u32, _next: u32) -> Result<(), SessionError> {
        let pos = t.pos().checked_sub(1).ok_or_else(|| {
            SessionError::Refused(format!("{WHAT}: a step that left the target at 0"))
        })?;
        if !self.follows(pos) {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a step at position {pos}, and the draft {} {} positions",
                if self.follows {
                    "has"
                } else {
                    "was left behind at"
                },
                self.committed()
            )));
        }
        let row = t.taps(1)?;
        Ok(self.append(row)?)
    }
}

impl<L: DraftLayers> CardDspark<L> {
    /// [`Draft::propose`] and [`Draft::propose_p`]: the proposal the target's
    /// verify holds, `p` empty for no probabilities.
    fn propose_with<B>(
        &mut self,
        t: &Session<B>,
        last: u32,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, SessionError>
    where
        B: ChainBody,
        Session<B>: Verify,
    {
        self.following(t.pos())?;
        let width = self.shape.width(<Session<B> as Verify>::MAX_ROWS);
        let pos = self.body.committed();
        let room = |have: usize| {
            SessionError::Refused(format!(
                "{WHAT}: a proposal of {width} ids with room for {have}"
            ))
        };
        let (out_len, p_len) = (out.len(), p.len());
        let out = out.get_mut(..width).ok_or_else(|| room(out_len))?;
        let p = if p_len == 0 {
            p
        } else {
            p.get_mut(..width).ok_or_else(|| room(p_len))?
        };
        Ok(self.on_card(|gpu, body| body.propose_as(gpu, pos, last, width, out, p))?)
    }
}
