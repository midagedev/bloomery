//! The draft a serve seat serves, in one place. A seat holds a [`ServedDraft`]
//! beside its session and forwards to it the [`Seat`](bloomery_gpu_gates::bind::Seat)
//! calls that move positions: the prompt call, a step, a pass, a reset, a cut,
//! the keep rule and `/props`' draft row. The seat's own plain path stays the
//! seat's: every method that has none to run says so by its `None`.
//!
//! The DSpark arm reaches its draft through [`Spark`] alone: the calls a seat
//! makes of the draft besides `runtime::Draft`, which the width chooser
//! (`BLOOMERY_MTP_WIDTH`) and the session's verify pass drive. The owner
//! stores the window behind a trait object (`Driven`), so a seat is generic
//! over its body and nothing else: a body that cannot verify rows (a
//! qwen3moe file's) holds an owner that is only ever off.
//!
//! The lookup draft is not here: it rebuilds its tables from the target's token
//! history, which only the V4.1 body keeps (`Body::history`), so it stays an
//! arm of the V4.1 seat beside the owner.
//!
//! The rule that a DSpark draft serves one slot is [`draft_slots`], and the
//! qwen3 seat's draft word is [`word`].

use std::path::{Path, PathBuf};

use app::{Keep, Prompt, RowsLog, Session, SessionError};
use bloomery_gpu::GpuError;
use bloomery_gpu::model::{ChainBody, Rollback, Rows};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::bind::{SlotsFrom, nvidia_smi_index};
use bloomery_levers::Levers;
use model::placement::workstation::DeviceId;
use runtime::width::{Choosing, Chosen, Mode as WidthMode};
use runtime::{Advance, Committed, Draft as RtDraft, Speculative, Widths, Window};
use serve::{DeviceProps, DraftProps, Drafted, EngineProps, PlacementProps};

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
struct Dspark<B: ChainBody + 'static> {
    window: Box<dyn Driven<B>>,
    file: Option<PathBuf>,
}

/// The draft a seat serves besides its lookup: none, or the DSpark draft
/// verified by the session's pass behind the width chooser.
pub struct ServedDraft<B: ChainBody + 'static> {
    dspark: Option<Dspark<B>>,
}

impl<B: ChainBody + 'static> ServedDraft<B> {
    /// No draft: every call is the seat's plain one.
    pub fn off() -> Self {
        ServedDraft { dspark: None }
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
        let spec = s.with_draft::<Choosing<D>, M>(behind::<B, D>(draft, mode)?, log)?;
        Ok(ServedDraft {
            dspark: Some(Dspark {
                window: Box::new(spec),
                file,
            }),
        })
    }

    /// The most rows [`Seat::pass`](bloomery_gpu_gates::bind::Seat::pass) runs.
    pub fn rows(&self) -> usize {
        self.dspark.as_ref().map_or(1, |d| d.window.rows())
    }

    /// The prompt call of `ids`, when a draft takes it: the draft fed as the
    /// call hands its features over, the argmax after the last id. `None`
    /// without one: the seat's plain prompt call is the call.
    pub fn prefill(&mut self, s: &mut Session<B>, ids: &[u32]) -> Result<Option<u32>, GateError> {
        match &mut self.dspark {
            None => Ok(None),
            Some(d) => Ok(Some(d.window.prompt_call(s, ids)?)),
        }
    }

    /// An image in the prompt call is refused by name under the DSpark draft.
    #[cfg(feature = "vision")]
    pub fn refuse_media(&self) -> Result<(), GateError> {
        match self.dspark {
            None => Ok(()),
            Some(_) => Err(MEDIA_UNDER_DSPARK.into()),
        }
    }

    /// The seat's plain step of `last` took `next`: the draft takes in the
    /// position.
    pub fn stepped(&mut self, s: &mut Session<B>, last: u32, next: u32) -> Result<(), GateError> {
        match &mut self.dspark {
            None => Ok(()),
            Some(d) => Ok(d.window.stepped(s, last, next)?),
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
        match &mut self.dspark {
            None => Ok(None),
            Some(d) => Ok(Some(drafted(d.window.pass(s, last, out)?))),
        }
    }

    /// The target stands elsewhere than the draft was fed.
    pub fn moved(&mut self) {
        if let Some(d) = &mut self.dspark {
            d.window.forget();
        }
    }

    /// The target was reset: the draft starts over with it.
    pub fn reset(&mut self) -> Result<(), GateError> {
        if let Some(d) = &mut self.dspark {
            d.window.restart()?;
        }
        Ok(())
    }

    /// The draft's window: the positions a request that starts the draft over
    /// feeds it before its first proposal; 0 without one.
    pub fn window(&self) -> usize {
        self.dspark.as_ref().map_or(0, |d| d.window.window())
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
        let Some(d) = &self.dspark else {
            return body(n);
        };
        if u32::try_from(n).is_ok_and(|n| n == pos && d.window.follows(n)) {
            return body(n);
        }
        let window = d.window.window();
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
        let Some(Dspark { window, file }) = &self.dspark else {
            return p;
        };
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

/// `d` behind the width chooser of `mode` ([`runtime::width`]): a seat's
/// drafts are the chooser's, whatever their own width.
pub fn behind<B, D>(d: D, mode: WidthMode) -> Result<Choosing<D>, SessionError>
where
    B: Prompt + Keep + Rows + Rollback,
    D: RtDraft<Session<B>>,
{
    Ok(<D as Chosen<Session<B>>>::choosing(d, mode)?)
}

/// A pass's draft counts: the proposal's ids, the ones kept.
pub fn drafted(c: Committed) -> Drafted {
    if c.proposed {
        Drafted {
            proposed: c.rows - 1,
            accepted: c.kept - 1,
        }
    } else {
        Drafted::default()
    }
}

/// `BLOOMERY_DRAFT` as the levers hold it for a seat that takes `off`: unset
/// and `off` are the plain path, `lookup` and `dspark` the served drafts.
/// (`Draft::from_levers` is the V4.1 binaries' parse, which refuses `off` and
/// registers the V4.1 draft record; this one needs neither.)
#[allow(
    dead_code,
    reason = "the qwen3 seat's parse; the V4.1 binary that includes this file serves no such seat"
)]
pub fn word(levers: &Levers) -> Result<Draft, GateError> {
    match levers.draft() {
        None | Some("off") => Ok(Draft::Off),
        Some("lookup") => Ok(Draft::Lookup),
        Some("dspark") => Ok(Draft::Dspark),
        Some(other) => {
            Err(format!("BLOOMERY_DRAFT is lookup, dspark, off or unset, not {other:?}").into())
        }
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
