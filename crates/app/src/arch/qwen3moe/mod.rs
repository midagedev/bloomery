//! The Qwen3.8-Flash-Next (qwen4exp) session: [`Body38`] behind the
//! session's traits, and the MTP draft a generation drives beside it
//! ([`MtpDraft`], the runtime rule's `Draft` impl over the draft layer
//! [`Mtp38`] carries on the target's own card).
//!
//! The MTP layer's row at position `q` reads the token at `q` and the
//! target's hidden row at `q − 1` (ik's pairing, the `mtp-qwen4exp` set's
//! shift 1), and predicts the token at `q + 1`. A generation drives it in
//! windows of four rows:
//!
//! - the prompt call ([`MtpDraft::prompt`]) walks the draft over the prompt
//!   as its units complete — a unit's arena holds its rows' hidden rows
//!   until the next unit overwrites them — filling the draft's store from
//!   position 0 (the row at 0 reads a zero hidden row: the target holds
//!   nothing before it) and leaving the last unit's last row for
//!   [`MtpDraft`'s `begin`](Draft::begin), whose walk is the next window's
//!   anchor;
//! - each window's [`MtpDraft::propose`] runs one chain
//!   ([`GpuModel::mtp_chain`], one readback): the refresh — the rows the
//!   target kept, at the positions after the verify's first, each with the
//!   hidden row the verify's row before it wrote, the last row the target's
//!   own next token (the anchor, whose prediction is the first proposal) —
//!   then two own walks, each reading the walk before it on the card;
//! - [`MtpDraft::accept`] records the refresh before the commit takes the
//!   rejected rows back, the verify's arena still holding every row's hidden
//!   row.
//!
//! The proposal changes which passes run, never a token: every kept token is
//! the target's own argmax. The draft never moves the target: its walks
//! write only the draft's store and arena, and the target's position moves
//! by the rule's verify and commit alone.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{
    Body38, MTP_ROWS, MtpFeed, MtpHead, MtpHidden, MtpMode, Prompt38, Qwen38Model, TargetRows,
};
use runtime::{Draft, Speculative, TapNeed, Tapped, Target};

use crate::{Keep, Prompt, Session, SessionError};

const WHAT: &str = "qwen4exp session";

/// How the session feeds a prompt and runs the draft's walks.
#[derive(Clone, Debug)]
pub struct Q38Cfg {
    /// The prompt path (`--prefill`): `auto` the default.
    pub prompt: Prompt38,
    /// The draft's walks captured (`graph`) or enqueued launch by launch
    /// (`eager`).
    pub draft: MtpMode,
}

impl Prompt for Body38 {
    /// The prompt call by the configured path — `auto` resolving by the
    /// prompt's length — the argmax after the last id.
    fn prompt(m: &mut Qwen38Model, ids: &[u32]) -> Result<u32, GpuError> {
        m.prompt38(ids, Prompt38::Auto)
    }
}

impl Keep for Body38 {
    /// Every position when `n` reaches the model's, else nothing: the
    /// recurrent state (the GDN lanes, the PLE hash history) keeps no
    /// earlier position — a verify's commit aside, which [`Keep::cut`]
    /// serves through the body's own rule.
    fn keepable(m: &Qwen38Model, n: u32) -> u32 {
        let pos = m.pos();
        if n >= pos { pos } else { 0 }
    }

    /// The body's commit: nothing to take back at the model's position, the
    /// waiting verify's kept rows anywhere past its first, refused by name
    /// elsewhere.
    fn cut(m: &mut Qwen38Model, n: u32) -> Result<(), GpuError> {
        m.rollback(n)
    }
}

impl Tapped for Session<Body38> {
    fn tap_width(&self) -> usize {
        self.model().body(WHAT).map_or(0, Body38::mtp_tap_width)
    }

    /// The last call's first `rows` final hidden rows ([`Body38::target_streams`]
    /// of the arena it walked), four streams of the model's width a row,
    /// kept for the borrow. Blocking.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        let walk = self.model().body(WHAT)?.last_walk();
        let v = self.taps_at(walk, rows)?;
        self.tapped = v;
        Ok(&self.tapped)
    }
}

impl Session<Body38> {
    /// The last call's first `rows` final hidden rows of the arena `walk`
    /// ([`Body38::target_streams`]). Blocking; gate use.
    pub fn taps_at(&mut self, walk: TargetRows, rows: usize) -> Result<Vec<f32>, SessionError> {
        let (gpu, _, body) = self.model_mut().body_parts(WHAT)?;
        Ok(body.target_streams(gpu, walk, rows)?)
    }
}

/// A generation's windows over the draft: the speculative advance a caller
/// drives a [`Session<Body38>`] with, `M` = the draft's four rows.
pub type Drafted38 = Speculative<MtpDraft, 4>;

/// The refresh a window's chain opens with: its rows' tokens, their first
/// position, and where their hidden rows sit — the arena `walk`'s rows
/// `first` on, one row a walk row.
struct Refresh {
    tokens: Vec<u32>,
    pos0: u32,
    walk: TargetRows,
    first: usize,
}

impl Refresh {
    /// The refresh's feed.
    fn feed(&self) -> MtpFeed<'_> {
        MtpFeed::Rows {
            tokens: &self.tokens,
            pos0: self.pos0,
            hidden: MtpHidden::Target {
                walk: self.walk,
                first: self.first,
            },
        }
    }
}

/// Qwen3.8's MTP draft over a loaded [`Mtp38`] (the module doc): the host
/// policy of the windows — which head the load opened, how the prompt is
/// fed, the refresh the next chain opens with — the walks themselves on the
/// card.
pub struct MtpDraft {
    cfg: Q38Cfg,
    head: MtpHead,
    /// The next window's refresh; `None` between a prompt call and its
    /// `begin`.
    next: Option<Refresh>,
    /// The last prompt unit's arena and rows, for [`Draft::begin`]'s
    /// anchor.
    last_unit: Option<(TargetRows, usize)>,
    /// A zero hidden row: the row at position 0's input.
    zeros: Vec<f32>,
}

impl MtpDraft {
    /// The draft over the model `m` loaded beside its target
    /// ([`Body38::open_placed_mtp`]): the head it opened (a row list or the
    /// full vocabulary) and the paths `cfg` names. Load-time only.
    pub fn open(m: &Qwen38Model, cfg: Q38Cfg) -> Result<MtpDraft, SessionError> {
        let body = m.body(WHAT)?;
        let head = if body.mtp().is_some_and(|d| d.head_map().is_some()) {
            MtpHead::Rows
        } else {
            MtpHead::Full
        };
        Ok(MtpDraft {
            head,
            next: None,
            last_unit: None,
            zeros: vec![0.0; body.mtp_tap_width()],
            cfg,
        })
    }

    /// The head the load opened.
    #[must_use]
    pub fn head(&self) -> MtpHead {
        self.head
    }

    /// The draft's own state started over: no refresh waits, the model's
    /// reset having emptied the draft's store. The session's reset calls it.
    pub fn restart(&mut self) {
        self.next = None;
        self.last_unit = None;
    }

    /// One run of the warmup: `tokens` at `pos0`, their hidden rows the
    /// arena `walk`'s from `first` on, walked eagerly with no readback — a
    /// fault stays on the fault word, which the next readback (the first
    /// chain's) names.
    fn warm(
        &self,
        m: &mut Qwen38Model,
        walk: TargetRows,
        pos0: u32,
        first: usize,
        tokens: &[u32],
    ) -> Result<(), GpuError> {
        for (i, run) in tokens.chunks(MTP_ROWS).enumerate() {
            m.mtp_walk(
                MtpFeed::Rows {
                    tokens: run,
                    pos0: pos0
                        + u32::try_from(i * MTP_ROWS)
                            .expect("a prompt's positions lie below its context"),
                    hidden: MtpHidden::Target {
                        walk,
                        first: first + i * MTP_ROWS,
                    },
                },
                self.head,
                MtpMode::Eager,
            )?;
        }
        Ok(())
    }
}

impl Draft<Session<Body38>> for MtpDraft {
    /// The most ids a proposal holds: a verify of four rows.
    const WIDTH: usize = 3;
    const TAPS: TapNeed = TapNeed::Final;

    /// The prompt through the target's own call, the draft walked over its
    /// units as they complete (the module doc): position 0 first — the
    /// target holds no hidden row before it — then every unit's rows from
    /// its second on, each with the hidden row its own arena holds for the
    /// position before it, the last unit's last row left for `begin`.
    fn prompt(&mut self, t: &mut Session<Body38>, ids: &[u32]) -> Result<u32, SessionError> {
        self.next = None;
        self.last_unit = None;
        let (path, head) = (self.cfg.prompt, self.head);
        let end_of_prompt = t.pos() + ids.len() as u32;
        let mut first_unit = true;
        let mut sink = |m: &mut Qwen38Model,
                        walk: TargetRows,
                        first: u32,
                        rows: usize|
         -> Result<(), GpuError> {
            self.last_unit = Some((walk, rows));
            // The row at `first + i` reads the target's hidden row at
            // `first + i − 1`: the unit's own arena holds it from its second
            // row on; the unit's first row was walked by the unit before (or
            // is position 0's zero-hidden row below), save the last unit's
            // last row — the next window's anchor, `begin`'s.
            let last = first + rows as u32 == end_of_prompt;
            if first_unit {
                first_unit = false;
                m.mtp_walk(
                    MtpFeed::Rows {
                        tokens: &ids[..1],
                        pos0: first,
                        hidden: MtpHidden::Host(&self.zeros),
                    },
                    head,
                    MtpMode::Eager,
                )?;
            }
            let end = first + rows as u32 + 1 - u32::from(last);
            if end > first + 1 {
                let at =
                    usize::try_from(first + 1).expect("a prompt's positions lie below its context");
                let to = usize::try_from(end).expect("a prompt fits usize");
                self.warm(m, walk, first + 1, 1, &ids[at..to])?;
            }
            Ok(())
        };
        let next = t.model_mut().prompt38_with(ids, path, Some(&mut sink))?;
        Ok(next)
    }

    /// The next window's anchor: the target's own token `first` at the
    /// prompt's end, with the hidden row the prompt's last position wrote.
    fn begin(
        &mut self,
        t: &Session<Body38>,
        _prompt: &[u32],
        first: u32,
    ) -> Result<(), SessionError> {
        let (walk, rows) = self.last_unit.ok_or_else(|| {
            SessionError::Refused(format!("{WHAT}: a prompt the draft never saw"))
        })?;
        self.next = Some(Refresh {
            tokens: vec![first],
            pos0: t.pos(),
            walk,
            first: rows - 1,
        });
        Ok(())
    }

    /// One chain, one readback (the module doc): the recorded refresh's
    /// walk — its last row the target's own next token, whose prediction is
    /// the first proposal — then own walks while the context holds them.
    fn propose(
        &mut self,
        t: &mut Session<Body38>,
        _last: u32,
        out: &mut [u32],
    ) -> Result<usize, SessionError> {
        let Some(r) = self.next.take() else {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a proposal before the draft's refresh (its prompt call, or the accept \
                 before it)"
            )));
        };
        let end = r.pos0 as usize + r.tokens.len();
        let here = t.pos() as usize;
        if end != here {
            return Err(SessionError::Refused(format!(
                "{WHAT}: a refresh of {} rows ending at {end}, where the target stands at {here}",
                r.tokens.len()
            )));
        }
        let own = (Self::WIDTH - 1).min(t.ctx() as usize - end);
        let d = t
            .model_mut()
            .mtp_chain(r.feed(), own, self.head, self.cfg.draft)?;
        out[..d.tokens.len()].copy_from_slice(&d.tokens);
        Ok(d.tokens.len())
    }

    /// The next refresh recorded before the commit takes the rejected rows
    /// back: the kept rows at the positions after the verify's first, each
    /// with the hidden row the verify's row before it wrote, the last row
    /// the target's own next token.
    fn accept(
        &mut self,
        t: &mut Session<Body38>,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        let p0 = t.pos() - rows.len() as u32;
        let last = out.get(accepted - 1).copied().ok_or_else(|| {
            SessionError::Refused(format!(
                "{WHAT}: a verify that kept {accepted} rows read back no argmax"
            ))
        })?;
        self.next = Some(Refresh {
            tokens: rows[1..accepted].iter().copied().chain([last]).collect(),
            pos0: p0 + 1,
            walk: TargetRows::Pass,
            first: 0,
        });
        Ok(())
    }

    /// A plain step's row — the token `next` at the position the step's
    /// argmax names, with the hidden row the step wrote — as the next
    /// refresh.
    fn stepped(
        &mut self,
        t: &mut Session<Body38>,
        _last: u32,
        next: u32,
    ) -> Result<(), SessionError> {
        self.next = Some(Refresh {
            tokens: vec![next],
            pos0: t.pos(),
            walk: TargetRows::Step,
            first: 0,
        });
        Ok(())
    }
}
