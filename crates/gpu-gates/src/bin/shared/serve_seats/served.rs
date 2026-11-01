//! The draft a serve seat serves, in one place. A seat holds a [`ServedDraft`]
//! beside its session and forwards to it the [`Seat`](bloomery_gpu_gates::bind::Seat)
//! calls that move positions: the prompt call, a step, a pass, a reset, a cut,
//! the keep rule and `/props`' draft row. The seat's own plain path stays the
//! seat's: every method that has none to run says so by its `None`.
//!
//! The DSpark arm is [`ServedSpark`], which reaches its draft through [`Spark`]
//! alone: the calls a seat makes of the draft besides `runtime::Draft`, which
//! the width chooser (`BLOOMERY_MTP_WIDTH`) and the session's verify pass
//! drive. It stores the window behind a trait object (`Driven`), so a seat is
//! generic over its body and nothing else: a body that cannot verify rows (a
//! qwen3moe file's) holds an owner that is only ever off. It needs no slots
//! (the DSpark draft serves one, [`draft_slots`]), so a body of one sequence
//! serves it alone, without the lookup arm.
//!
//! The card draft's own adapter is one impl here too ([`Spark`] over
//! [`CardDspark`], every seat's): a seat names its draft's layers, never a
//! copy of these calls.
//!
//! The lookup arm holds the serve seats' one owner of the lookup draft,
//! [`Lookups`]: a context and a width chooser a resident slot, the context
//! one the V4.1 seat rebuilds from its body's token history when it falls
//! behind.
//!
//! The rule that a DSpark draft serves one slot is [`draft_slots`], and the
//! qwen3 seat's draft word is [`word`].

use std::path::{Path, PathBuf};

use app::dspark::CardDspark;
use app::{Keep, Prompt, RowsLog, Session, SessionError};
use bloomery_gpu::GpuError;
use bloomery_gpu::dspark::DraftLayers;
use bloomery_gpu::model::{ChainBody, Rollback, Rows, Slots};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{SlotsFrom, nvidia_smi_index};
use bloomery_levers::Levers;
use model::placement::workstation::DeviceId;
use runtime::width::{Choosing, Chosen, Mode as WidthMode};
use runtime::{Advance, Committed, Draft as RtDraft, Speculative, Widths, Window};
use serve::{DeviceProps, DraftProps, Drafted, EngineProps, PlacementProps};

use super::drafted::{History, LOOKUP_ROWS, Lookups, Taking, drafted, lookup_props};
use crate::draft::Draft;

/// The class `/props` files the draft's resident bytes under.
const DRAFT_CLASS: &str = "draft";

/// The refusal of an image under the DSpark draft: its window is fed by the
/// prompt call's features, which a media call hands none.
#[cfg(feature = "vision")]
const MEDIA_UNDER_DSPARK: &str = "an image under BLOOMERY_DRAFT=dspark: the draft's window is fed \
     by the prompt call's features, which a media call hands none; serve the lookup draft, or none";

/// The calls a seat makes of a DSpark draft of the session over body `B`,
/// besides `runtime::Draft` (which the chooser and the verify pass drive).
pub trait Spark<B>: RtDraft<Session<B>> + Sized + 'static
where
    B: Prompt + Keep + Rows + Rollback,
{
    /// The prompt call of `ids` from where `s` stands, each position's
    /// features into the draft as the call hands them over; the argmax after
    /// the last id.
    fn feed_call(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError>;

    /// The target now stands elsewhere than the positions the draft was fed:
    /// the next [`Spark::feed_call`] starts the draft over.
    fn forget(&mut self);

    /// Start the draft over at position 0, for a target that was reset.
    fn restart(&mut self) -> Result<(), GpuError>;

    /// The trailing positions of a prompt whose features the draft keeps.
    fn window(&self) -> usize;

    /// Whether the draft holds every position the target ran before `pos`.
    fn follows(&self, pos: u32) -> bool;

    /// The device bytes the draft's load took on its card.
    fn resident_bytes(&self) -> usize;

    /// The device the draft runs on, as this process's census names it.
    fn device(&self) -> Result<DeviceId, GpuError>;

    /// The card the draft runs on.
    fn card(&self) -> &'static str;
}

/// The DSpark draft of any body that taps its rows: the card draft the session drives, as
/// the owner calls it. One impl for every seat ([`CardDspark`] is the one owner of the card
/// draft), so a new body takes the draft by naming its layers, never by copying this.
///
/// A draft with no window keeps every position, so [`Spark::window`] answers `usize::MAX`: a
/// cut then leaves a prompt call that starts the draft over at 0 and feeds every id. The two
/// families whose draft file always carries one (V4.1's `DraftHparams::read` and DFlash2's
/// `Hparams::read` refuse a missing `attention.sliding_window` by name) read the same way
/// here: the file's own reader has already refused the case that has no window.
impl<B, L> Spark<B> for CardDspark<L>
where
    B: Prompt + Keep + Rows + Rollback + 'static,
    Session<B>: runtime::TappedPrompt<Error = SessionError>,
    L: DraftLayers + 'static,
{
    fn feed_call(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError> {
        CardDspark::<L>::feed_call(self, s, ids)
    }

    fn forget(&mut self) {
        CardDspark::<L>::forget(self);
    }

    fn restart(&mut self) -> Result<(), GpuError> {
        CardDspark::<L>::restart(self)
    }

    fn window(&self) -> usize {
        CardDspark::<L>::window(self).unwrap_or(usize::MAX)
    }

    fn follows(&self, pos: u32) -> bool {
        CardDspark::<L>::follows(self, pos)
    }

    fn resident_bytes(&self) -> usize {
        CardDspark::<L>::resident_bytes(self)
    }

    fn device(&self) -> Result<DeviceId, GpuError> {
        CardDspark::<L>::device(self)
    }

    fn card(&self) -> &'static str {
        CardDspark::<L>::card(self)
    }
}

/// A draft's window behind the width chooser, as the owner drives it. One
/// impl, over every [`Spark`] draft and its pass width.
trait Driven<B: ChainBody + 'static> {
    /// The width chooser's record for the last request, then the prompt call.
    fn prompt_call(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError>;
    fn stepped(&mut self, s: &mut Session<B>, last: u32, next: u32) -> Result<(), SessionError>;
    fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Committed, SessionError>;
    /// The most rows one pass verifies.
    fn rows(&self) -> usize;
    /// The width chooser's record for the last request, at the server's stop.
    fn close(&mut self);
    fn forget(&mut self);
    fn restart(&mut self) -> Result<(), GpuError>;
    fn window(&self) -> usize;
    fn follows(&self, pos: u32) -> bool;
    fn resident_bytes(&self) -> usize;
    fn device(&self) -> Result<DeviceId, GpuError>;
    fn card(&self) -> &'static str;
}

impl<B, D, const M: usize> Driven<B> for Speculative<Choosing<D>, M>
where
    B: Prompt + Keep + Rows + Rollback + 'static,
    D: Spark<B>,
    Window<M>: Widths,
{
    fn prompt_call(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, SessionError> {
        super::drafted::print_widths(self);
        self.draft_mut().draft_mut().feed_call(s, ids)
    }

    fn stepped(&mut self, s: &mut Session<B>, last: u32, next: u32) -> Result<(), SessionError> {
        RtDraft::stepped(self.draft_mut(), s, last, next)
    }

    fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Committed, SessionError> {
        Advance::pass(self, s, last, out)
    }

    fn rows(&self) -> usize {
        M
    }

    fn close(&mut self) {
        super::drafted::print_widths(self);
    }

    fn forget(&mut self) {
        self.draft_mut().draft_mut().forget();
    }

    fn restart(&mut self) -> Result<(), GpuError> {
        self.draft_mut().draft_mut().restart()
    }

    fn window(&self) -> usize {
        self.draft().draft().window()
    }

    fn follows(&self, pos: u32) -> bool {
        self.draft().draft().follows(pos)
    }

    fn resident_bytes(&self) -> usize {
        self.draft().draft().resident_bytes()
    }

    fn device(&self) -> Result<DeviceId, GpuError> {
        self.draft().draft().device()
    }

    fn card(&self) -> &'static str {
        self.draft().draft().card()
    }
}

/// The DSpark draft a seat serves: its window behind the width chooser and the
/// draft's file for `/props`.
pub struct ServedSpark<B: ChainBody + 'static> {
    window: Box<dyn Driven<B>>,
    file: Option<PathBuf>,
}

impl<B: ChainBody + 'static> ServedSpark<B> {
    /// `draft` on `s`, behind the width chooser of `mode`, its verify pass of
    /// every width up to `M` = the draft's width + 1 captured now (`log` told
    /// of each); `file` is the draft's file for `/props`.
    pub fn open<D, const M: usize>(
        s: &mut Session<B>,
        draft: D,
        mode: WidthMode,
        log: &mut impl RowsLog,
        file: Option<PathBuf>,
    ) -> Result<Self, SessionError>
    where
        B: Prompt + Keep + Rows + Rollback,
        D: Spark<B>,
        Window<M>: Widths,
    {
        let spec = s.with_draft::<Choosing<D>, M>(behind::<B, D>(draft, mode)?, log)?;
        Ok(ServedSpark {
            window: Box::new(spec),
            file,
        })
    }

    /// The draft's own prompt call of `ids`, fed as the call hands its
    /// features over. The argmax after the last id.
    pub fn prefill(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<u32, GateError> {
        Ok(self.window.prompt_call(s, ids)?)
    }

    /// The seat's plain step of `last` took `next`: the draft takes in the
    /// position.
    pub fn stepped(&mut self, s: &mut Session<B>, last: u32, next: u32) -> Result<(), GateError> {
        Ok(self.window.stepped(s, last, next)?)
    }

    /// One drafted pass from `last`: the kept ids into `out` and what the
    /// draft proposed and kept.
    pub fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        Ok(drafted(self.window.pass(s, last, out)?))
    }

    /// The most rows [`Seat::pass`](bloomery_gpu_gates::bind::Seat::pass) runs.
    pub fn rows(&self) -> usize {
        self.window.rows()
    }

    /// The target stands elsewhere than the draft was fed: the draft starts
    /// over at its next call.
    pub fn moved(&mut self) {
        self.window.forget();
    }

    /// The target was reset: the draft starts over with it.
    pub fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.window.restart()?)
    }

    /// The server's stop: the width chooser's record of the last request
    /// printed, since no prompt call follows it.
    pub fn close(&mut self) {
        self.window.close();
    }

    /// The positions a request that starts the draft over feeds it before its
    /// first proposal.
    pub fn window(&self) -> usize {
        self.window.window()
    }

    /// Whether the draft holds every position the target ran before `pos`.
    #[allow(
        dead_code,
        reason = "the seat that checks its draft's position before it proposes reads it; a binary that includes this file and serves no such seat reads it nowhere"
    )]
    pub fn follows(&self, pos: u32) -> bool {
        self.window.follows(pos)
    }

    /// The longest prefix of at most `n` positions a cut keeps, `body` being
    /// the body's rule at the target's position `pos`: a cut anywhere but
    /// where the draft stands also leaves the request's prompt call a window.
    pub fn keep(
        &self,
        n: usize,
        pos: u32,
        body: impl FnOnce(usize) -> (usize, Option<String>),
    ) -> (usize, Option<String>) {
        if u32::try_from(n).is_ok_and(|n| n == pos && self.window.follows(n)) {
            return body(n);
        }
        let window = self.window.window();
        let (k, why) = body(n.saturating_sub(window));
        let why = format!(
            "the DSpark draft starts over, so the prompt call carries its window of \
             {window} positions: cut to at most {}{}",
            n.saturating_sub(window),
            why.map(|w| format!(", then {w}")).unwrap_or_default()
        );
        (k, Some(why))
    }

    /// `engine.draft`, and the draft's resident bytes in its card's placement
    /// row, the class `draft` (a row of its own when the target has no layer
    /// there). The card's index missing leaves the device out.
    pub fn props(&self, mut p: EngineProps) -> EngineProps {
        let ServedSpark { window, file } = self;
        let device = window
            .device()
            .ok()
            .and_then(|dev| nvidia_smi_index(window.card(), Some(dev)).ok())
            .map(|i| format!("GPU{i}"));
        let name = file
            .as_deref()
            .and_then(Path::file_name)
            .map(|f| f.to_string_lossy().into_owned());
        if let (Some(dev), Some(pl)) = (&device, p.placement.as_mut()) {
            add_draft_bytes(pl, dev, window.resident_bytes() as u64);
        }
        p.draft = Some(DraftProps {
            model: name.unwrap_or_else(|| Draft::Dspark.name().to_owned()),
            n_max: Some(window.rows() as u64 - 1),
            kind: Some(Draft::Dspark.name().to_owned()),
            path: file.as_deref().map(|p| p.display().to_string()),
            device,
        });
        p
    }
}

/// The draft a seat serves: none, the DSpark draft or the lookup, each
/// verified by the session's pass behind the width chooser.
pub struct ServedDraft<B: ChainBody + 'static> {
    dspark: Option<ServedSpark<B>>,
    lookup: Option<Lookups<Session<B>>>,
}

impl<B> ServedDraft<B>
where
    B: Prompt + Keep + Slots + 'static,
    B::Seq: 'static,
{
    /// No draft: every call is the seat's plain one.
    pub fn off() -> Self {
        ServedDraft {
            dspark: None,
            lookup: None,
        }
    }

    /// The lookup draft on each of `slots` resident slots of `s`
    /// ([`Lookups::open`]), each behind a width chooser of `mode`, each served
    /// slot's verify pass captured now (`log` told of each); `history` the
    /// body's token history, where it keeps one.
    pub fn lookup(
        s: &mut Session<B>,
        slots: usize,
        mode: WidthMode,
        log: &mut impl RowsLog,
        history: Option<History<Session<B>>>,
    ) -> Result<Self, GateError>
    where
        B: Rows + Rollback,
    {
        Ok(ServedDraft {
            dspark: None,
            lookup: Some(Lookups::open(s, slots, mode, log, history)?),
        })
    }

    /// `draft` on `s`, behind the width chooser of `mode`, its verify pass of
    /// every width up to `M` = the draft's width + 1 captured now (`log` told
    /// of each); `file` is the draft's file for `/props`.
    pub fn dspark<D, const M: usize>(
        s: &mut Session<B>,
        draft: D,
        mode: WidthMode,
        log: &mut impl RowsLog,
        file: Option<PathBuf>,
    ) -> Result<Self, SessionError>
    where
        B: Prompt + Keep + Rows + Rollback,
        D: Spark<B>,
        Window<M>: Widths,
    {
        Ok(ServedDraft {
            dspark: Some(ServedSpark::open::<D, M>(s, draft, mode, log, file)?),
            lookup: None,
        })
    }

    /// The most rows [`Seat::pass`](bloomery_gpu_gates::bind::Seat::pass) runs.
    pub fn rows(&self) -> usize {
        match (&self.dspark, &self.lookup) {
            (Some(d), _) => d.rows(),
            (None, Some(_)) => LOOKUP_ROWS,
            (None, None) => 1,
        }
    }

    /// Whether the draft's state is per slot (`Seat::slot_drafts`): the
    /// lookup's is, a context a slot.
    pub fn slot_drafts(&self) -> bool {
        self.lookup.is_some()
    }

    /// The prompt call of `ids`: under the DSpark draft the draft's own, fed
    /// as the call hands its features over; else `plain`, the seat's own,
    /// its ids fed to the lookup's context of the selected slot when the
    /// lookup runs. The argmax after the last id.
    pub fn prefill(
        &mut self,
        s: &mut Session<B>,
        ids: &[u32],
        plain: impl FnOnce(&mut Session<B>) -> Result<u32, GateError>,
    ) -> Result<u32, GateError> {
        match (&mut self.dspark, &mut self.lookup) {
            (Some(d), _) => d.prefill(s, ids),
            (None, Some(l)) => {
                let slot = s.selected();
                l.prompt_call(s, slot, ids, plain)
            }
            (None, None) => plain(s),
        }
    }

    /// The prompt call of `ids` that holds an image, `call` the seat's own:
    /// refused by name under the DSpark draft, its ids fed to the lookup's
    /// context as [`ServedDraft::prefill`]'s.
    #[cfg(feature = "vision")]
    pub fn prefill_media(
        &mut self,
        s: &mut Session<B>,
        ids: &[u32],
        call: impl FnOnce(&mut Session<B>) -> Result<u32, GateError>,
    ) -> Result<u32, GateError> {
        if self.dspark.is_some() {
            return Err(MEDIA_UNDER_DSPARK.into());
        }
        self.prefill(s, ids, call)
    }

    /// The seat's plain step of `last` took `next`: the draft takes in the
    /// position.
    pub fn stepped(&mut self, s: &mut Session<B>, last: u32, next: u32) -> Result<(), GateError> {
        match (&mut self.dspark, &mut self.lookup) {
            (Some(d), _) => d.stepped(s, last, next),
            (None, Some(l)) => l.stepped(s, s.selected(), last, next),
            (None, None) => Ok(()),
        }
    }

    /// One drafted pass from `last`: the kept ids into `out` and what the
    /// draft proposed and kept. `None` without a draft: the seat's plain step
    /// is the pass.
    pub fn pass(
        &mut self,
        s: &mut Session<B>,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Option<Drafted>, GateError> {
        match (&mut self.dspark, &mut self.lookup) {
            (Some(d), _) => Ok(Some(d.pass(s, last, out)?)),
            (None, Some(l)) => {
                let slot = s.selected();
                Ok(Some(l.pass(s, slot, last, Taking::Argmax, out)?))
            }
            (None, None) => Ok(None),
        }
    }

    /// The target stands elsewhere than the draft was fed: another slot
    /// selected, or a cut. The DSpark draft starts over at its next call;
    /// the lookup's contexts follow their slots by position and need
    /// nothing.
    pub fn moved(&mut self) {
        if let Some(d) = &mut self.dspark {
            d.moved();
        }
    }

    /// `slot`'s sequence was put back from a saved state: the DSpark draft
    /// starts over at its next call, and the lookup's context of the slot
    /// starts over (from the body's history where it keeps one).
    pub fn replaced(&mut self, slot: usize) -> Result<(), GateError> {
        self.moved();
        if let Some(l) = &mut self.lookup {
            l.forget(slot)?;
        }
        Ok(())
    }

    /// The target's `slot` was reset: the draft starts over with it.
    pub fn reset(&mut self, slot: usize) -> Result<(), GateError> {
        if let Some(d) = &mut self.dspark {
            d.reset()?;
        }
        if let Some(l) = &mut self.lookup {
            l.forget(slot)?;
        }
        Ok(())
    }

    /// The server's stop: the width chooser's record of each slot's last
    /// request printed, the DSpark draft's or the lookup's.
    pub fn close(&mut self) {
        if let Some(d) = &mut self.dspark {
            d.close();
        }
        if let Some(l) = &mut self.lookup {
            l.close();
        }
    }

    /// The draft's window: the positions a request that starts the draft over
    /// feeds it before its first proposal; 0 without one.
    pub fn window(&self) -> usize {
        self.dspark.as_ref().map_or(0, ServedSpark::window)
    }

    /// The longest prefix of at most `n` positions a cut keeps, `body` being
    /// the body's rule at the target's position `pos`. Under the DSpark draft,
    /// a cut anywhere but where the draft stands also leaves the request's
    /// prompt call a window.
    pub fn keep(
        &self,
        n: usize,
        pos: u32,
        body: impl FnOnce(usize) -> (usize, Option<String>),
    ) -> (usize, Option<String>) {
        match &self.dspark {
            Some(d) => d.keep(n, pos, body),
            None => body(n),
        }
    }

    /// `engine.draft`, and the draft's resident bytes in its card's placement
    /// row, the class `draft` (a row of its own when the target has no layer
    /// there). The card's index missing leaves the device out.
    pub fn props(&self, p: EngineProps) -> EngineProps {
        if self.lookup.is_some() {
            return lookup_props(p);
        }
        match &self.dspark {
            Some(d) => d.props(p),
            None => p,
        }
    }
}

/// `d` behind the width chooser of `mode` ([`runtime::width`]): a seat's
/// drafts are the chooser's, whatever their own width.
pub fn behind<B, D>(d: D, mode: WidthMode) -> Result<Choosing<D>, SessionError>
where
    B: Prompt + Keep + Rows + Rollback,
    D: RtDraft<Session<B>>,
{
    Ok(<D as Chosen<Session<B>>>::choosing(d, mode)?)
}

/// `BLOOMERY_DRAFT` as the qwen3 seat reads it. Unset and `off` are
/// [`Word::Off`] here; the seat decides afterwards whether a qwen35moe file's
/// in-file layer drafts. `lookup` and `dspark` stay the served drafts
/// ([`crate::draft::Draft`]). (`Draft::from_levers` is the V4.1 binaries'
/// parse, which refuses `off` and registers the V4.1 draft record; this one
/// needs neither.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// Unset or `off`: the seat's plain path, or a qwen35moe file's in-file
    /// layer when that seat's default says so.
    Off,
    /// The n-gram draft.
    Lookup,
    /// The DSpark draft.
    Dspark,
    /// The file's own next-token layer.
    Mtp,
}

/// [`Word`] from the levers. A word this seat does not take is refused by name.
#[allow(
    dead_code,
    reason = "the qwen3 seat's parse; the V4.1 binary that includes this file serves no such seat"
)]
pub fn word(levers: &Levers) -> Result<Word, GateError> {
    match levers.draft() {
        None | Some("off") => Ok(Word::Off),
        Some("lookup") => Ok(Word::Lookup),
        Some("dspark") => Ok(Word::Dspark),
        Some("mtp") => Ok(Word::Mtp),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT is lookup, dspark, mtp, off or unset, not {other:?}"
        )
        .into()),
    }
}

/// The slot count the DSpark draft decides, when it does: its window is one
/// sequence's, on its own card, and a select would start it over. Several
/// slots (`parallel` above 1) are refused by name; none given is one slot
/// ([`SlotsFrom::Dspark`]); `None` where `draft` decides nothing, or the flag
/// names one slot. `lookup` is whether the seat serves the lookup draft too,
/// which the refusal then offers.
pub fn draft_slots(
    parallel: Option<usize>,
    draft: Draft,
    lookup: bool,
) -> Result<Option<(usize, SlotsFrom)>, GateError> {
    if draft != Draft::Dspark {
        return Ok(None);
    }
    match parallel {
        Some(n) if n > 1 => Err(format!(
            "--parallel {n} under BLOOMERY_DRAFT=dspark: the draft's window is one \
             sequence's, on its own card, and a select would start it over; serve one \
             slot{}",
            if lookup { ", or the lookup draft" } else { "" }
        )
        .into()),
        Some(_) => Ok(None),
        None => Ok(Some((1, SlotsFrom::Dspark))),
    }
}

/// `bytes` of the draft on `device`: into that card's row, or a row of its
/// own before the host's.
fn add_draft_bytes(p: &mut PlacementProps, device: &str, bytes: u64) {
    if let Some(d) = p.devices.iter_mut().find(|d| d.device == device) {
        *d.class_bytes.entry(DRAFT_CLASS.to_owned()).or_default() += bytes;
        return;
    }
    let at = p
        .devices
        .iter()
        .position(|d| d.device == "CPU")
        .unwrap_or(p.devices.len());
    p.devices.insert(
        at,
        DeviceProps {
            device: device.to_owned(),
            class_bytes: [(DRAFT_CLASS.to_owned(), bytes)].into(),
            layers: None,
        },
    );
}
