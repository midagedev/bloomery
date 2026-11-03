//! The session as the host of the MTP window ([`runtime::mtp::Host`]): the
//! window's policy and its draft state machine are `runtime::mtp`'s, and the
//! session runs the walks they ask for on the draft layer it opened
//! ([`Session::open_mtp`]).
//!
//! A walk is a row at a time in one command buffer, each row the single-row
//! draft step of the layer (`super::encode_rows`) reading the id in its own
//! word and the hidden row its place names: a host-written row, the arena's
//! `x` (the row a settled host-fed step leaves, [`MtpArena::Step`]) or a row
//! of the rows arena's residual (a prompt pass's or a verify's,
//! [`MtpArena::Rows`]). A store walk stops at the layer's cache append: the
//! join, the fusion projection, the attention's input side and the keys and
//! values, no flash output, FFN half or head. A chain's refresh is a store
//! walk over every row but its last, then the last row whole, whose head's
//! pick is the one readback.
//!
//! What the Metal host refuses by name, where the shared window may ask:
//!
//! - an own walk after the refresh and a probability beside the pick: the
//!   proposal is one id ([`Host::WIDTH`]), so the window's chain has no own
//!   row, and the head's readback is the pick alone ([`Host::PROBS`]);
//! - a captured walk: the draft's walks are eager;
//! - a walk while a token runs ahead: the step arena's residual is that
//!   token's. A caller that drafts steps host-fed ([`Session::set_argmax_mode`]);
//!   a prompt call run with a sink steps its ids that way;
//! - a prompt call that is not read at its end ([`CallEnd`]), and the copy of
//!   a shared prompt arena's row to the step arena: the Metal session holds one
//!   sequence, so no call switches it with a prompt's waiting row in the rows
//!   arena.
//!
//! The rows arena is both a verify's arena and a prompt pass's: the window
//! reads a verify's rows before the next verify or pass overwrites them, as
//! it does a pass unit's.

use core::sync::atomic::Ordering;

use bloomery_device::Stream;
use runtime::mtp::{Chain, Exec, Hidden, Host, HostSink, MtpDraft, Walk, WalkMode};
use runtime::prompt::CallEnd;

use super::{DraftRow, HiddenAt, RowKind, Session, SessionError, mtp_refuse};
use crate::arch::qwen3moe::arena::slot_of;
use crate::arch::qwen3moe::lanes::VERIFY_ROWS;
use crate::arch::qwen3moe::target;
use crate::kernels::HiddenRow;

/// The most rows one store walk takes: the draft's ids hold a word a row, and
/// a prompt unit's rows are walked in runs of this many, each one command
/// buffer and one wait.
pub(super) const STORE_ROWS: usize = 64;

const _: () = {
    // The session is a host the window accepts: the widths fit together, and
    // the draft's ids hold a word for the rows of a walk of either kind.
    let () = <Session as Host>::FITS;
    assert!(
        <Session as Host>::STORE_ROWS >= <Session as Host>::WALK_ROWS,
        "the draft's ids hold a word a row of the widest walk"
    );
};

/// A target arena whose final hidden rows a draft walk reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpArena {
    /// The step's: the arena's `x`, the one row a settled step leaves.
    Step,
    /// The rows arena's residual: the final rows of a verify or of a prompt
    /// pass, a row each.
    Rows,
}

/// The positions of the current sequence the draft's store holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Held(usize);

impl Held {
    /// The count.
    pub(super) fn get(self) -> usize {
        self.0
    }

    /// The store holds nothing: the session's reset.
    pub(super) fn empty(&mut self) {
        self.0 = 0;
    }

    /// The session was cut back to `pos`: no position past it is held.
    pub(super) fn cut_to(&mut self, pos: u32) {
        self.0 = self.0.min(pos as usize);
    }

    /// A walk may start at `pos0`: at position 0, which replaces the count,
    /// or at or below the positions held. A start past them is refused by name
    /// and the count stays.
    pub(super) fn start(&mut self, pos0: u32) -> Result<(), SessionError> {
        if pos0 == 0 {
            self.0 = 0;
        }
        if pos0 as usize > self.0 {
            return Err(mtp_refuse(format!(
                "a walk from position {pos0}: the draft's store holds this sequence's positions \
                 below {}, and the walk would read the keys between",
                self.0
            )));
        }
        Ok(())
    }

    /// A walk of `rows` rows from `pos0` ran: the store holds them too.
    pub(super) fn walked(&mut self, pos0: u32, rows: usize) {
        self.0 = self.0.max(pos0 as usize + rows);
    }
}

/// Where row `j` of a walk reads its hidden row, the walk's `n` rows from row
/// `first` of the target's arena `walk` (the rows arena holds `cap` rows), or
/// the refusal by name of rows the arena does not hold.
fn hidden_at(
    walk: MtpArena,
    first: usize,
    j: usize,
    n: usize,
    cap: usize,
) -> Result<HiddenAt, SessionError> {
    match walk {
        MtpArena::Step if first == 0 && n == 1 => Ok(HiddenAt::Step),
        MtpArena::Step => Err(mtp_refuse(format!(
            "a walk of {n} rows from row {first} of the step's arena, which holds one row"
        ))),
        MtpArena::Rows if first.checked_add(n).is_some_and(|end| end <= cap) => {
            Ok(HiddenAt::Rows(first + j))
        }
        MtpArena::Rows => Err(mtp_refuse(format!(
            "a walk of {n} rows from row {first} of the rows arena, which holds {cap} rows"
        ))),
    }
}

/// The rows of a walk of `tokens` from `pos0`: row `j` the token at
/// `pos0 + j` beside the hidden row `hidden` names, every row a store row but
/// the last when `whole` asks the last one whole.
fn rows_of(
    pos0: u32,
    n: usize,
    hidden: impl Fn(usize) -> Result<HiddenAt, SessionError>,
    reading: HiddenRow,
    whole: bool,
) -> Result<Vec<DraftRow>, SessionError> {
    (0..n)
        .map(|j| {
            Ok(DraftRow {
                id: j,
                pos: pos0 + u32::try_from(j).expect("a walk's rows fit u32"),
                hidden: hidden(j)?,
                reading,
                kind: if whole && j + 1 == n {
                    RowKind::Full
                } else {
                    RowKind::Store
                },
            })
        })
        .collect()
}

/// A host's walks as the executor a prompt call's sink runs them on.
struct SessionExec<'a>(&'a mut Session);

impl Exec<MtpArena> for SessionExec<'_> {
    type Err = SessionError;

    fn walk(&mut self, w: &Walk<'_, MtpArena>) -> Result<(), SessionError> {
        Host::walk(self.0, w, ())
    }

    fn held(&mut self) -> Result<usize, SessionError> {
        Host::held(self.0)
    }
}

impl Session {
    /// Whether a sidecar is open as the draft layer.
    #[must_use]
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// The hidden row a walk over the target's rows joins: the row before the
    /// target's output norm or after it. [`HiddenRow::Post`] unless set; a
    /// draft step the caller makes names its own.
    pub fn set_mtp_reading(&mut self, reading: HiddenRow) -> Result<(), SessionError> {
        self.mtp_ref()?;
        self.mtp.as_mut().expect("checked above").reading = reading;
        Ok(())
    }

    /// The window's draft over the opened sidecar: `runtime::mtp::MtpDraft`
    /// with this session as its host, its chains eager. Drive it as
    /// `Speculative<Choosing<MtpDraft<Session>>, 2>`.
    pub fn mtp_draft(&self) -> Result<MtpDraft<Session>, SessionError> {
        self.mtp_ref()?;
        Ok(MtpDraft::new((), (), WalkMode::Eager, self.arena.hidden))
    }

    /// A draft walk's preconditions: the session is not poisoned, no verify
    /// waits for its commit and no token runs ahead; the stream settled.
    fn draft_ready(&self, what: &str) -> Result<(), SessionError> {
        if let Some(fault) = self.poisoned {
            return Err(SessionError::Poisoned(fault));
        }
        self.not_pending(what)?;
        if self.ahead.is_some() {
            return Err(mtp_refuse(format!(
                "{what} while a token runs ahead: the step arena's residual is that token's; \
                 step host-fed (`Session::set_argmax_mode`)"
            )));
        }
        self.stream.sync()?;
        Ok(())
    }

    /// `tokens` at `pos0 ..` walked on the draft, every row a store row but
    /// the last when `whole`; `hidden` names the rows' hidden rows. Returns
    /// the last row's position.
    fn draft_rows(
        &mut self,
        tokens: &[u32],
        pos0: u32,
        hidden: Hidden<'_, MtpArena>,
        whole: bool,
    ) -> Result<u32, SessionError> {
        let n = tokens.len();
        if n == 0 {
            return Err(mtp_refuse(format!(
                "a walk of no rows from position {pos0}"
            )));
        }
        let last = u32::try_from(n)
            .ok()
            .and_then(|n| pos0.checked_add(n))
            .filter(|&end| end as usize <= self.arena.ctx)
            .map(|end| end - 1)
            .ok_or_else(|| {
                mtp_refuse(format!(
                    "a walk of {n} rows from position {pos0}: the cache holds {} rows",
                    self.arena.ctx
                ))
            })?;
        self.draft_ready("a draft walk")?;
        let reading = self.mtp_ref()?.reading;
        let cap = self.rows.cap;
        let planned = match hidden {
            Hidden::Host(row) => {
                let h = self.arena.hidden;
                if n != 1 || row.len() != h {
                    return Err(mtp_refuse(format!(
                        "a walk of {n} rows over {} host values; one row of the model's {h} values \
                         is what the host writes",
                        row.len()
                    )));
                }
                rows_of(pos0, n, |_| Ok(HiddenAt::Own(0)), reading, whole)?
            }
            Hidden::Target { walk, first } => rows_of(
                pos0,
                n,
                |j| hidden_at(walk, first, j, n, cap),
                reading,
                whole,
            )?,
        };
        let mtp = self.mtp.as_mut().expect("checked above");
        mtp.held.start(pos0)?;
        if let Hidden::Host(row) = hidden {
            mtp.bufs.hid.write(row)?;
        }
        for (j, &id) in tokens.iter().enumerate() {
            mtp.bufs.ids.pages().store(j, id, Ordering::Relaxed)?;
        }
        self.submit_rows(&planned)?;
        self.mtp
            .as_mut()
            .expect("checked above")
            .held
            .walked(pos0, n);
        Ok(last)
    }
}

impl Host for Session {
    /// One id a proposal: the draft layer's own hidden row is not fed to a
    /// walk after it.
    const WIDTH: usize = 1;
    const WALK_ROWS: usize = VERIFY_ROWS;
    const STORE_ROWS: usize = STORE_ROWS;

    type Arena = MtpArena;
    const STEP_ARENA: MtpArena = MtpArena::Step;
    const VERIFY_ARENA: MtpArena = MtpArena::Rows;
    type Path = ();
    type Head = ();
    type Cause = SessionError;

    fn held(&self) -> Result<usize, SessionError> {
        Ok(self.mtp_ref()?.held.get())
    }

    fn walk(&mut self, w: &Walk<'_, MtpArena>, (): ()) -> Result<(), SessionError> {
        if w.mode != WalkMode::Store {
            return Err(mtp_refuse(format!(
                "a {:?} walk: the walks the session runs with no readback are the store walk's, \
                 a row's keys and values; a chain reads the head",
                w.mode
            )));
        }
        if w.tokens.len() > STORE_ROWS {
            return Err(mtp_refuse(format!(
                "a store walk of {} rows; it takes at most {STORE_ROWS}",
                w.tokens.len()
            )));
        }
        self.draft_rows(w.tokens, w.pos0, w.hidden, false)?;
        Ok(())
    }

    fn chain(
        &mut self,
        c: &Chain<MtpArena>,
        (): (),
        out: &mut [u32],
        p: Option<&mut [f32]>,
    ) -> Result<usize, SessionError> {
        if c.own != 0 {
            return Err(mtp_refuse(format!(
                "a chain of {} own rows: the proposal holds one id, so a walk after the refresh \
                 has no hidden row of the draft's own to read",
                c.own
            )));
        }
        if p.is_some() {
            return Err(mtp_refuse(
                "a chain's probabilities: the draft's head reads back its pick alone",
            ));
        }
        match c.mode {
            WalkMode::Eager => {}
            WalkMode::Store => {
                return Err(mtp_refuse(
                    "a chain in the store walk's mode: it reads the head",
                ));
            }
            WalkMode::Graph => {
                return Err(mtp_refuse("a captured walk: the draft's walks are eager"));
            }
        }
        if out.is_empty() {
            return Err(mtp_refuse("a proposal of 1 id into 0 places"));
        }
        let r = &c.refresh;
        if r.tokens.len() > Self::WALK_ROWS {
            return Err(mtp_refuse(format!(
                "a refresh of {} rows; a walk takes at most {}",
                r.tokens.len(),
                Self::WALK_ROWS
            )));
        }
        let last = self.draft_rows(
            &r.tokens,
            r.pos0,
            Hidden::Target {
                walk: r.walk,
                first: r.first,
            },
            true,
        )?;
        let id = self
            .arena
            .step_id
            .pages()
            .load(slot_of(last), Ordering::Relaxed)?;
        if id as usize >= self.arena.vocab {
            return Err(mtp_refuse(format!(
                "the draft picked id {id} at position {last}; the vocabulary holds {}",
                self.arena.vocab
            )));
        }
        out[0] = id;
        Ok(1)
    }

    fn prompt_with(
        &mut self,
        ids: &[u32],
        (): (),
        end: CallEnd,
        (): (),
        sink: Option<&mut HostSink<'_, Self>>,
    ) -> Result<Option<u32>, SessionError> {
        if end != CallEnd::Argmax {
            return Err(mtp_refuse(format!(
                "a prompt call ending {end:?}: the session's prompt call reads the argmax after \
                 its last id"
            )));
        }
        self.mtp_ref()?;
        let pick = match sink {
            None => target::prompt_call(self, ids, &[], None)?,
            Some(sink) => {
                let mut unit = |s: &mut Session, arena: MtpArena, first: u32, rows: usize| {
                    sink(&mut SessionExec(s), arena, first, rows)
                };
                target::prompt_call(self, ids, &[], Some(&mut unit))?
            }
        };
        Ok(Some(pick))
    }

    fn retain_prompt_row(
        &mut self,
        walk: MtpArena,
        first: usize,
        pos: u32,
    ) -> Result<(), SessionError> {
        Err(mtp_refuse(format!(
            "keeping row {first} of {walk:?}'s arena at position {}: the session holds one \
             sequence, and no caller switches it with a prompt's waiting row in the rows arena",
            pos.saturating_sub(1)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn said(e: SessionError) -> String {
        e.to_string()
    }

    /// The step's arena holds the one row a settled step leaves: a walk of
    /// that row reads `x`, and a walk of more rows or from another row is
    /// refused by name.
    ///
    /// Mutant: the step's arena mapped to a rows-arena row, or a second row
    /// of it admitted.
    #[test]
    fn the_step_arena_is_one_row_of_x() {
        assert_eq!(
            hidden_at(MtpArena::Step, 0, 0, 1, 8).map_err(said),
            Ok(HiddenAt::Step)
        );
        for (first, n) in [(0, 2), (1, 1)] {
            let why = hidden_at(MtpArena::Step, first, 0, n, 8).map_err(said);
            assert!(
                why.as_ref()
                    .is_err_and(|e| e.contains("step's arena, which holds one row")),
                "{why:?}"
            );
        }
    }

    /// The rows arena's row `j` of a walk from row `first` is row
    /// `first + j`, and a walk past the arena's rows is refused by name.
    ///
    /// Mutant: the row taken as `j` alone, or the bound off by one.
    #[test]
    fn the_rows_arena_maps_each_row_of_a_walk_to_its_own_row() {
        assert_eq!(
            hidden_at(MtpArena::Rows, 3, 0, 2, 8).map_err(said),
            Ok(HiddenAt::Rows(3))
        );
        assert_eq!(
            hidden_at(MtpArena::Rows, 3, 1, 2, 8).map_err(said),
            Ok(HiddenAt::Rows(4))
        );
        assert_eq!(
            hidden_at(MtpArena::Rows, 6, 1, 2, 8).map_err(said),
            Ok(HiddenAt::Rows(7))
        );
        let why = hidden_at(MtpArena::Rows, 7, 0, 2, 8).map_err(said);
        assert!(
            why.as_ref()
                .is_err_and(|e| e.contains("rows arena, which holds 8 rows")),
            "{why:?}"
        );
        let why = hidden_at(MtpArena::Rows, usize::MAX, 0, 2, 8).map_err(said);
        assert!(why.is_err(), "{why:?}");
    }

    /// The window's refusal is the session's, by name: the text the window
    /// gave is the text the session reports.
    #[test]
    fn the_windows_refusal_is_the_sessions_by_name() {
        let e = SessionError::from(runtime::mtp::Refused(
            "MTP window: a proposal into no room".into(),
        ));
        assert!(
            e.to_string()
                .contains("MTP window: a proposal into no room"),
            "{e}"
        );
    }

    /// A chain's rows are store rows but the last, which is whole and carries
    /// the head; a walk's rows are all store rows; each row has its own word of
    /// the ids and the next position.
    ///
    /// Mutant: the whole row put first, or every row of a chain whole.
    #[test]
    fn a_chains_last_row_alone_is_whole() {
        let hidden = |j| hidden_at(MtpArena::Rows, 0, j, 2, 8);
        let chain = rows_of(10, 2, hidden, HiddenRow::Post, true).expect("two rows");
        assert_eq!(
            chain
                .iter()
                .map(|r| (r.id, r.pos, r.hidden, r.kind))
                .collect::<Vec<_>>(),
            [
                (0, 10, HiddenAt::Rows(0), RowKind::Store),
                (1, 11, HiddenAt::Rows(1), RowKind::Full),
            ]
        );
        let walk = rows_of(10, 2, hidden, HiddenRow::Pre, false).expect("two rows");
        assert!(walk.iter().all(|r| r.kind == RowKind::Store));
        assert!(walk.iter().all(|r| r.reading == HiddenRow::Pre));
        let one = rows_of(4, 1, |_| Ok(HiddenAt::Step), HiddenRow::Post, true).expect("one row");
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].kind, RowKind::Full);
    }

    /// The store the draft holds: a walk raises it to the end of its rows
    /// (never lowers it), may start at or below it and never past it, a start
    /// at position 0 replaces it, a cut lowers it to the position and a reset
    /// empties it.
    ///
    /// Mutant: the count set to a walk's end where it was higher, a start past
    /// the count admitted, a cut that leaves it, or a reset that does not
    /// reach it.
    #[test]
    fn the_held_count_follows_walks_cuts_and_resets() {
        let mut held = Held::default();
        assert_eq!(held.get(), 0);
        // The prompt: position 0 then runs of rows, each from the held end.
        held.start(0).expect("a start at 0");
        held.walked(0, 1);
        held.start(1).expect("the next start");
        held.walked(1, 5);
        assert_eq!(held.get(), 6);
        // A walk past the held positions is refused and leaves the count.
        let why = held.start(7).map_err(said);
        assert!(
            why.as_ref()
                .is_err_and(|e| e.contains("holds this sequence's positions below 6")),
            "{why:?}"
        );
        assert_eq!(held.get(), 6);
        // A chain's refresh of two rows from below the end walks the end on.
        held.start(5)
            .expect("a refresh from the last held position");
        held.walked(5, 2);
        assert_eq!(held.get(), 7);
        // A walk that ends below the count does not lower it.
        held.start(3).expect("a start below the count");
        held.walked(3, 1);
        assert_eq!(held.get(), 7);
        // A cut holds nothing past its position, and a cut past the count
        // holds what there was.
        held.cut_to(4);
        assert_eq!(held.get(), 4);
        held.cut_to(9);
        assert_eq!(held.get(), 4);
        assert!(held.start(5).is_err());
        // A walk from 0 replaces the count, so the prompt of the next sequence
        // starts over whatever the store held.
        held.start(0).expect("a start at 0");
        assert_eq!(held.get(), 0);
        held.walked(0, 3);
        held.empty();
        assert_eq!(held.get(), 0);
        assert!(held.start(1).is_err());
    }
}
