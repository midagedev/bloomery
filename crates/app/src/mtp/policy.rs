//! The MTP window's host policy over plain values: [`Policy`] holds what the
//! window knows between two calls (the rows waiting for the draft's next
//! walk, the last prompt unit, why the draft skips, the last join) and decides
//! every call from positions, the context, token ids and arena kinds alone —
//! which waiting rows a call walks and in which mode, the anchor a step
//! records, the join checks, the refusals and the sequence a park keeps.
//!
//! The walks themselves are the executor's ([`Exec`]): a call issues them one
//! at a time, in the order the window runs them, and the state the call
//! mutates between two walks is the state the executor sees when the second
//! one fails. The card's executor is `MtpDraft`'s (`super::Card`); the unit
//! tests below run the same calls against a model of the body's arena and
//! store records, with no card.

use std::fmt;

use super::{Join, Parked, WHAT, WalkMode};

/// A join the draft cannot make: no rows of the held sequence wait for it.
const NOTHING_WAITS: &str = "no rows of the held sequence wait for the draft: it did not walk \
                             the sequence the target holds";
/// A join the draft cannot make: the waiting rows end elsewhere.
const ENDS_ELSEWHERE: &str = "the rows waiting for the draft do not end at the call's first \
                              position";
/// A join the draft cannot make: the store is behind the waiting rows.
const STORE_BEHIND: &str = "the draft's store does not hold the positions below the rows \
                            waiting for it";
/// A join the draft cannot make: the sequence was parked with its waiting
/// rows in a prompt call's arena.
const PROMPT_ARENA: &str = "the rows waiting for the draft sat in a prompt call's arena when the \
                            sequence was parked, and a parked sequence keeps only the step's and \
                            the verify's";

/// A call the window refuses, by name: the text the session reports.
#[derive(Debug)]
pub(super) struct Refused(pub(super) String);

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// How a call that runs walks ends when it fails: the window's refusal, or
/// the executor's own error from a walk.
#[derive(Debug)]
pub(super) enum Fail<E> {
    Refused(Refused),
    Exec(E),
}

/// Where a walk's hidden rows come from; `A` is the body's arena kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Source<A> {
    /// One zero row: position 0's input, the target holding nothing before
    /// it.
    Zero,
    /// The target's final hidden rows in the arena `walk`, from its row
    /// `first` on, as the target's last call of that arena left them.
    Target { walk: A, first: usize },
}

/// One walk of the draft with no readback, as the window decides it: `tokens`
/// at positions `pos0 ..`, each beside its hidden row, run in `mode`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Walk<'a, A> {
    pub(super) tokens: &'a [u32],
    pub(super) pos0: u32,
    pub(super) hidden: Source<A>,
    pub(super) mode: WalkMode,
}

/// What a call's walks run on, and the one question a join asks of it.
pub(super) trait Exec<A> {
    type Err;

    /// Run `w`; its error ends the call.
    fn walk(&mut self, w: &Walk<'_, A>) -> Result<(), Self::Err>;

    /// The positions of the held sequence the draft's store holds: a walk
    /// may start at or below it, never past it.
    fn held(&mut self) -> Result<usize, Self::Err>;
}

/// The window's constants as plain values: what the decisions read of a body.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape<A> {
    /// The arena a plain step leaves its row in.
    pub(super) step: A,
    /// The arena a verify leaves its rows in.
    pub(super) verify: A,
    /// How the walks with a head and the chains run.
    pub(super) mode: WalkMode,
    /// The most ids a proposal holds.
    pub(super) width: usize,
    /// The most rows one store walk takes.
    pub(super) store_rows: usize,
}

/// The refresh a window's chain opens with: its rows' tokens, their first
/// position, and where their hidden rows sit — the arena `walk`'s rows
/// `first` on, one row a walk row.
#[derive(Clone, Debug)]
pub(super) struct Refresh<A> {
    pub(super) tokens: Vec<u32>,
    pub(super) pos0: u32,
    pub(super) walk: A,
    pub(super) first: usize,
}

impl<A: Copy> Refresh<A> {
    /// The refresh's rows as a walk in `mode`.
    fn as_walk(&self, mode: WalkMode) -> Walk<'_, A> {
        Walk {
            tokens: &self.tokens,
            pos0: self.pos0,
            hidden: Source::Target {
                walk: self.walk,
                first: self.first,
            },
            mode,
        }
    }

    /// The refresh's feed.
    pub(super) fn feed(&self) -> super::Feed<'_, A> {
        super::Feed {
            tokens: &self.tokens,
            pos0: self.pos0,
            hidden: super::Hidden::Target {
                walk: self.walk,
                first: self.first,
            },
        }
    }
}

/// A window's chain as decided: the refresh to walk, its last row the token
/// at the target's position, then `own` walks, all in `mode`.
#[derive(Debug)]
pub(super) struct Chain<A> {
    pub(super) refresh: Refresh<A>,
    pub(super) own: usize,
    pub(super) mode: WalkMode,
}

/// A prompt call's feed of the draft, one unit at a time ([`Policy::unit`]).
#[derive(Debug)]
pub(super) struct Call<'a> {
    ids: &'a [u32],
    start: u32,
    end: u32,
    first_unit: bool,
}

/// The window's host policy (the module doc).
#[derive(Debug)]
pub(super) struct Policy<A> {
    shape: Shape<A>,
    /// The next window's refresh; `None` between a prompt call and its
    /// `begin`.
    next: Option<Refresh<A>>,
    /// The last prompt unit's arena and rows, for [`Policy::begin`]'s anchor.
    last_unit: Option<(A, usize)>,
    /// The last join to a held sequence a call continued, until taken.
    joined: Option<Join>,
    /// Why the draft proposes nothing until the next prompt call from
    /// position 0 or restart; `None` while it drafts.
    skip: Option<&'static str>,
}

impl<A: Copy + Eq + fmt::Debug> Policy<A> {
    pub(super) fn new(shape: Shape<A>) -> Self {
        Policy {
            shape,
            next: None,
            last_unit: None,
            joined: None,
            skip: None,
        }
    }

    /// Started over: no refresh waits, no skip.
    pub(super) fn restart(&mut self) {
        self.next = None;
        self.last_unit = None;
        self.joined = None;
        self.skip = None;
    }

    pub(super) fn skipping(&self) -> bool {
        self.skip.is_some()
    }

    pub(super) fn take_joined(&mut self) -> Option<Join> {
        self.joined.take()
    }

    /// What the window holds of the target's sequence between two calls, for
    /// the caller's sequence state ([`Policy::unpark`] puts it back): the rows
    /// waiting for its next walk when they sit in the step's or the verify's
    /// arena, or why it proposes nothing — its own skip, or rows a prompt
    /// call's arena holds, which a sequence state does not keep.
    pub(super) fn park(&self) -> Parked<A> {
        let Shape { step, verify, .. } = self.shape;
        let kept = |walk: A| walk == step || walk == verify;
        let skipping = |why| Parked {
            next: None,
            last_unit: None,
            skip: Some(why),
        };
        match (self.skip, &self.next, self.last_unit) {
            (Some(why), _, _) => skipping(why),
            (None, Some(r), _) if kept(r.walk) => Parked {
                next: Some(r.clone()),
                last_unit: None,
                skip: None,
            },
            (None, None, Some((walk, rows))) if kept(walk) => Parked {
                next: None,
                last_unit: Some((walk, rows)),
                skip: None,
            },
            (None, Some(_), _) | (None, None, Some(_)) => skipping(PROMPT_ARENA),
            (None, None, None) => Parked {
                next: None,
                last_unit: None,
                skip: None,
            },
        }
    }

    /// `p`, which [`Policy::park`] took, back in place.
    pub(super) fn unpark(&mut self, p: &Parked<A>) {
        self.next = p.next.clone();
        self.last_unit = p.last_unit;
        self.skip = p.skip;
        self.joined = None;
    }

    /// Before a plain step of `last` at the target's position `pos`: a
    /// refresh an earlier step or window left waiting reads an arena the step
    /// may overwrite (its own), so it is walked now, its last row `last`. A
    /// draft that cannot walk it skips, the join naming why. A prompt call's
    /// anchor is left for [`Policy::stepped`]. A server seat calls this
    /// before each of its steps, and the runtime's width chooser — whose
    /// held pass steps the target with no proposal of the draft's — before
    /// each pass it holds back.
    pub(super) fn before_step<X: Exec<A>>(
        &mut self,
        x: &mut X,
        pos: u32,
        last: u32,
    ) -> Result<(), X::Err> {
        if self.skip.is_some() || self.next.is_none() {
            return Ok(());
        }
        if let Err(why) = self.catch_up(x, pos, last)? {
            self.skip_from(pos, why);
        }
        Ok(())
    }

    /// The rows an earlier call left waiting — the refresh a window or a
    /// step recorded, or a prompt call's anchor — walked with `token` as
    /// their last row, the row at the target's position `here`, so the store
    /// holds every position through it: the rows walked, or why the draft
    /// cannot walk them. Nothing has run on the target since they were
    /// recorded (the caller's contract), so their arena still holds their
    /// hidden rows.
    fn catch_up<X: Exec<A>>(
        &mut self,
        x: &mut X,
        here: u32,
        token: u32,
    ) -> Result<Result<usize, &'static str>, X::Err> {
        let mut r = match (self.next.take(), self.last_unit.take()) {
            (Some(r), _) => r,
            (None, Some((walk, rows))) => Refresh {
                tokens: vec![token],
                pos0: here,
                walk,
                first: rows - 1,
            },
            (None, None) => return Ok(Err(NOTHING_WAITS)),
        };
        if r.pos0 as usize + r.tokens.len() != here as usize + 1 {
            return Ok(Err(ENDS_ELSEWHERE));
        }
        if r.pos0 as usize > x.held()? {
            return Ok(Err(STORE_BEHIND));
        }
        if let Some(l) = r.tokens.last_mut() {
            *l = token;
        }
        x.walk(&r.as_walk(self.shape.mode))?;
        Ok(Ok(r.tokens.len()))
    }

    /// The draft proposes nothing from the call at `start` on, for `why`.
    fn skip_from(&mut self, start: u32, why: &'static str) {
        self.next = None;
        self.last_unit = None;
        self.skip = Some(why);
        self.joined = Some(Join {
            start,
            caught_up: 0,
            skipped: Some(why),
        });
    }

    /// A prompt call of `ids` from the position `start`: the window's state
    /// for it — position 0 starts over, a call that continues the held
    /// sequence joins the draft to it (walking the rows the last call left
    /// waiting, the call's first id their last), and one the draft cannot
    /// join skips. `Some` is the feed of the call's units ([`Policy::unit`]);
    /// `None` is a plain call of the target's, the draft untouched or
    /// skipping.
    pub(super) fn prompt_start<'a, X: Exec<A>>(
        &mut self,
        x: &mut X,
        start: u32,
        ids: &'a [u32],
    ) -> Result<Option<Call<'a>>, X::Err> {
        let Some(&id) = ids.first() else {
            return Ok(None);
        };
        if start == 0 {
            self.next = None;
            self.last_unit = None;
            self.joined = None;
            self.skip = None;
        } else if let Some(why) = self.skip {
            self.skip_from(start, why);
        } else {
            match self.catch_up(x, start, id)? {
                Ok(rows) => {
                    self.joined = Some(Join {
                        start,
                        caught_up: rows,
                        skipped: None,
                    });
                }
                Err(why) => self.skip_from(start, why),
            }
        }
        if self.skip.is_some() {
            return Ok(None);
        }
        Ok(Some(Call {
            ids,
            start,
            end: start + ids.len() as u32,
            first_unit: start == 0,
        }))
    }

    /// One unit of the prompt call `call` completed: `rows` rows from the
    /// position `first`, their final hidden rows in the arena `walk`. The
    /// draft walks over them through the store's append alone, filling its
    /// store; the row at `first + i` reads the target's hidden row at
    /// `first + i − 1`, which the unit's own arena holds from its second row
    /// on. The unit's first row was walked by the unit before (or is position
    /// 0's zero-hidden row, or the join's last row), save the last unit's last
    /// row — the next window's anchor, [`Policy::begin`]'s.
    pub(super) fn unit<X: Exec<A>>(
        &mut self,
        call: &mut Call<'_>,
        x: &mut X,
        walk: A,
        first: u32,
        rows: usize,
    ) -> Result<(), X::Err> {
        self.last_unit = Some((walk, rows));
        let last = first + rows as u32 == call.end;
        if call.first_unit {
            call.first_unit = false;
            x.walk(&Walk {
                tokens: &call.ids[..1],
                pos0: first,
                hidden: Source::Zero,
                mode: WalkMode::Store,
            })?;
        }
        let end = first + rows as u32 + 1 - u32::from(last);
        if end > first + 1 {
            let at = usize::try_from(first + 1 - call.start)
                .expect("a prompt's positions lie below its context");
            let to = usize::try_from(end - call.start).expect("a prompt fits usize");
            // Position first + 1 + i reads the unit's row i, the hidden row
            // of the position before it.
            self.warm(x, walk, first + 1, &call.ids[at..to])?;
        }
        Ok(())
    }

    /// One run of the warmup: `tokens` at `pos0`, their hidden rows the arena
    /// `walk`'s from its row 0 on, walked through the store's append alone
    /// ([`WalkMode::Store`]) in runs of at most the body's store rows.
    fn warm<X: Exec<A>>(
        &self,
        x: &mut X,
        walk: A,
        pos0: u32,
        tokens: &[u32],
    ) -> Result<(), X::Err> {
        let rows = self.shape.store_rows;
        for (i, run) in tokens.chunks(rows).enumerate() {
            x.walk(&Walk {
                tokens: run,
                pos0: pos0
                    + u32::try_from(i * rows).expect("a prompt's positions lie below its context"),
                hidden: Source::Target {
                    walk,
                    first: i * rows,
                },
                mode: WalkMode::Store,
            })?;
        }
        Ok(())
    }

    /// The next window's anchor: the target's own token `first` at the
    /// prompt's end `pos`, with the hidden row the prompt's last position
    /// wrote — unless a step since the prompt call already recorded one (the
    /// server's cut: the prompt less its last id, then the last id as a step,
    /// [`Policy::stepped`] recording the anchor the step's window proposes
    /// from).
    pub(super) fn begin(&mut self, pos: u32, first: u32) -> Result<(), Refused> {
        if self.skip.is_some() || self.next.is_some() {
            return Ok(());
        }
        let (walk, rows) = self
            .last_unit
            .ok_or_else(|| Refused(format!("{WHAT}: a prompt the draft never saw")))?;
        self.next = Some(Refresh {
            tokens: vec![first],
            pos0: pos,
            walk,
            first: rows - 1,
        });
        Ok(())
    }

    /// The window's chain at the target's position `here` of a context of
    /// `ctx`, into `room` ids: the refresh taken, its last row `last`, the
    /// token at `here`, and the own walks the room and the verify's rows
    /// below `ctx` allow; `None` while the draft skips.
    pub(super) fn chain(
        &mut self,
        here: u32,
        ctx: u32,
        last: u32,
        room: usize,
    ) -> Result<Option<Chain<A>>, Refused> {
        if self.skip.is_some() {
            return Ok(None);
        }
        let Some(mut refresh) = self.next.take() else {
            return Err(Refused(format!(
                "{WHAT}: a proposal before the draft's refresh (its prompt call, or the accept \
                 before it)"
            )));
        };
        // The refresh's last row is the token at the target's position, which
        // the target has not run yet: the refresh ends one past it.
        let end = refresh.pos0 as usize + refresh.tokens.len();
        let here = here as usize;
        if end != here + 1 {
            return Err(Refused(format!(
                "{WHAT}: a refresh of {} rows ending at {end}, where the target stands at {here} \
                 (its next token's row ends at {})",
                refresh.tokens.len(),
                here + 1
            )));
        }
        // The verify of the proposal runs a row at every position through
        // its last id: `here` (the refresh's last row) through
        // `here + own + 1`, one past the last own walk, so the narrowest
        // chain (`own` 0) verifies two rows. A chain whose narrowest verify
        // does not fit the context is refused by name, not clamped: every
        // caller gates its pass by the rows it runs before the draft is
        // asked, so a chain this close to the end is that contract's breach.
        if end >= ctx as usize {
            return Err(Refused(format!(
                "{WHAT}: a chain at position {here} of a context of {ctx}: its narrowest verify \
                 would run the rows {here} and {}, past the context's end",
                here + 1
            )));
        }
        let room = room.min(self.shape.width);
        if room == 0 {
            return Err(Refused(format!("{WHAT}: a proposal into no room")));
        }
        if let Some(l) = refresh.tokens.last_mut() {
            *l = last;
        }
        let own = (room - 1).min(ctx as usize - 1 - end);
        Ok(Some(Chain {
            refresh,
            own,
            mode: self.shape.mode,
        }))
    }

    /// The next refresh recorded for a verify of `rows` that ran from position
    /// `p0`, keeping its first `accepted` rows, `out` the id the pass took at
    /// each, before the commit takes the rest back: the kept rows at the
    /// positions after the verify's first, each with the hidden row the
    /// verify's row before it wrote, the last row the id the pass took after
    /// the last kept row — the argmax on a greedy pass, the draw on a sampled
    /// one.
    pub(super) fn record(
        &mut self,
        p0: u32,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), Refused> {
        let last = accepted
            .checked_sub(1)
            .and_then(|k| out.get(k))
            .copied()
            .ok_or_else(|| {
                Refused(format!(
                    "{WHAT}: a verify that kept {accepted} rows took no id"
                ))
            })?;
        self.next = Some(Refresh {
            tokens: rows[1..accepted].iter().copied().chain([last]).collect(),
            pos0: p0 + 1,
            walk: self.shape.verify,
            first: 0,
        });
        Ok(())
    }

    /// A plain step of `last` that left the target at `pos`, `next` the id
    /// taken after it: the rows that waited for the next chain — the
    /// refresh an accept recorded, after a prompt call with no `begin` the
    /// anchor (`last` at the prompt's end), or on a first step at position
    /// 0 its row beside a zero hidden row — walked now, since the step's
    /// row follows them and the store must hold every position below the
    /// next chain's; then `next` at the position after the step, with the
    /// hidden row the step wrote, as the next refresh, its rows left for
    /// the call that walks them — the chooser's hook
    /// ([`Policy::before_step`], which the runtime calls before a pass it
    /// holds back) or a seat's step before the next plain step, the next
    /// chain for its head, a continuing prompt's join — so the store never
    /// holds a position past the target's, which a sequence state needs.
    /// Refused by name when the waiting rows read the step's own arena,
    /// which the step has overwritten.
    pub(super) fn stepped<X: Exec<A>>(
        &mut self,
        x: &mut X,
        pos: u32,
        last: u32,
        next: u32,
    ) -> Result<(), Fail<X::Err>> {
        if self.skip.is_some() {
            return Ok(());
        }
        let waiting = match (self.next.take(), self.last_unit.take()) {
            (Some(r), _) => Some(r),
            (None, Some((walk, rows))) => Some(Refresh {
                tokens: vec![last],
                pos0: pos - 1,
                walk,
                first: rows - 1,
            }),
            (None, None) => {
                // A one-id prompt feeds no prompt call: the step ran position
                // 0, whose row reads a zero hidden row, as the prompt call's
                // first row does.
                if pos == 1 {
                    x.walk(&Walk {
                        tokens: &[last],
                        pos0: 0,
                        hidden: Source::Zero,
                        mode: self.shape.mode,
                    })
                    .map_err(Fail::Exec)?;
                }
                None
            }
        };
        if let Some(r) = waiting {
            if r.walk == self.shape.step {
                return Err(Fail::Refused(Refused(format!(
                    "{WHAT}: a step after rows whose hidden rows the step's own arena held (a \
                     prompt fed by steps): the step overwrote them (rows at position {})",
                    r.pos0
                ))));
            }
            x.walk(&r.as_walk(self.shape.mode)).map_err(Fail::Exec)?;
        }
        self.next = Some(Refresh {
            tokens: vec![next],
            pos0: pos,
            walk: self.shape.step,
            first: 0,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena kinds of a body whose prompt runs through ubatches or
    /// passes, its verify through the pass arena and its plain step through
    /// the step's: the shape Qwen3.8's window has.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ar {
        Step = 0,
        Pass = 1,
        Ubatch = 2,
    }

    const SHAPE: Shape<Ar> = Shape {
        step: Ar::Step,
        verify: Ar::Pass,
        mode: WalkMode::Graph,
        width: 3,
        store_rows: 4,
    };

    /// The prompt every test feeds: nine ids at positions 0..9.
    const IDS: [u32; 9] = [10, 11, 12, 13, 14, 15, 16, 17, 18];

    const HEAD: WalkMode = WalkMode::Graph;
    const STORE: WalkMode = WalkMode::Store;

    /// One walk as the card ran it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Seen {
        mode: WalkMode,
        pos0: u32,
        tokens: Vec<u32>,
        hidden: Source<Ar>,
    }

    fn seen(mode: WalkMode, pos0: u32, tokens: &[u32], hidden: Source<Ar>) -> Seen {
        Seen {
            mode,
            pos0,
            tokens: tokens.to_vec(),
            hidden,
        }
    }

    fn rows_of(walk: Ar, first: usize) -> Source<Ar> {
        Source::Target { walk, first }
    }

    /// The rows an arena holds, row `r` the position `first + r`: the body's
    /// record of what the call that last wrote the arena left there.
    #[derive(Clone, Copy, Debug, Default)]
    struct Held {
        first: u32,
        rows: usize,
    }

    /// The target and the draft's store as the body records them, with no
    /// card: a walk is refused by name when it starts past the store or
    /// reads rows its arena no longer holds at the positions the walk needs,
    /// exactly the two refusals the body makes, and it leaves the store
    /// holding the positions through its last row.
    struct Card {
        pos: u32,
        ctx: u32,
        store: usize,
        arenas: [Held; 3],
        seen: Vec<Seen>,
        /// The walk that fails: the number of walks to run before it.
        fail_after: Option<usize>,
    }

    impl Card {
        fn new(ctx: u32) -> Card {
            Card {
                pos: 0,
                ctx,
                store: 0,
                arenas: [Held::default(); 3],
                seen: Vec::new(),
                fail_after: None,
            }
        }

        /// A call of the target that wrote `rows` rows of `arena` from the
        /// position it stood at.
        fn ran(&mut self, arena: Ar, rows: usize) {
            self.arenas[arena as usize] = Held {
                first: self.pos,
                rows,
            };
            self.pos += u32::try_from(rows).unwrap();
            assert!(self.pos <= self.ctx, "the target ran past its context");
        }

        /// The commit of a verify from `p0` keeping `kept` rows: the position
        /// taken back to, and the arenas' and the store's rows past it cut.
        fn commit(&mut self, p0: u32, kept: usize) {
            let pos = p0 + u32::try_from(kept).unwrap();
            self.pos = pos;
            for h in &mut self.arenas {
                h.rows = h.rows.min(pos.saturating_sub(h.first) as usize);
            }
            self.store = self.store.min(pos as usize);
        }

        /// The walks run since the last take.
        fn take(&mut self) -> Vec<Seen> {
            std::mem::take(&mut self.seen)
        }
    }

    impl Exec<Ar> for Card {
        type Err = String;

        fn walk(&mut self, w: &Walk<'_, Ar>) -> Result<(), String> {
            if let Some(n) = &mut self.fail_after {
                if *n == 0 {
                    return Err("the card refused the walk".to_string());
                }
                *n -= 1;
            }
            let m = w.tokens.len();
            if w.pos0 as usize > self.store {
                return Err(format!(
                    "a walk from position {}: the draft's store holds the positions below {}",
                    w.pos0, self.store
                ));
            }
            match w.hidden {
                Source::Zero if w.pos0 == 0 && m == 1 => {}
                Source::Zero => {
                    return Err(format!(
                        "a zero hidden row for {m} rows from position {}",
                        w.pos0
                    ));
                }
                Source::Target { walk, first } => {
                    let at = w
                        .pos0
                        .checked_sub(1)
                        .ok_or("the target holds no row before position 0")?;
                    let h = self.arenas[walk as usize];
                    if first + m > h.rows || h.first as usize + first != at as usize {
                        return Err(format!(
                            "rows {first}..{} of the {walk:?} arena as positions {at}..{} for a \
                             walk from {}: the arena holds {}..{}",
                            first + m,
                            at as usize + m,
                            w.pos0,
                            h.first,
                            h.first as usize + h.rows
                        ));
                    }
                }
            }
            self.store = w.pos0 as usize + m;
            self.seen.push(seen(w.mode, w.pos0, w.tokens, w.hidden));
            Ok(())
        }

        fn held(&mut self) -> Result<usize, String> {
            Ok(self.store)
        }
    }

    fn fail(f: Fail<String>) -> String {
        match f {
            Fail::Refused(r) => r.to_string(),
            Fail::Exec(e) => e,
        }
    }

    /// What a seat or the runtime's generation loop calls on the draft, in
    /// the order each call runs, over the policy and the card model.
    struct Rig {
        p: Policy<Ar>,
        c: Card,
        /// The token at the target's position.
        last: u32,
    }

    impl Rig {
        fn new(ctx: u32) -> Rig {
            Rig {
                p: Policy::new(SHAPE),
                c: Card::new(ctx),
                last: 0,
            }
        }

        /// The prompt call of `ids` from the target's position, its units of
        /// `unit` rows through `arena`, as the body runs them: each unit
        /// written by the target, then handed to the draft.
        fn prompt(&mut self, ids: &[u32], unit: usize, arena: Ar) -> Result<(), String> {
            let start = self.c.pos;
            let mut call = self.p.prompt_start(&mut self.c, start, ids)?;
            for run in ids.chunks(unit) {
                let first = self.c.pos;
                self.c.ran(arena, run.len());
                if let Some(call) = &mut call {
                    self.p.unit(call, &mut self.c, arena, first, run.len())?;
                }
            }
            self.last = 1000 + self.c.pos;
            Ok(())
        }

        /// `Draft::begin` with the prompt's argmax, the token at the
        /// target's position.
        fn begin(&mut self) -> Result<(), String> {
            self.p
                .begin(self.c.pos, self.last)
                .map_err(|e| e.to_string())
        }

        /// The server's step and a pass the width chooser holds back — the
        /// same order: the draft's waiting rows walked first (the seat's
        /// `before_step`, the chooser's `before_plain` hook), then the
        /// target's step, then the draft told (`stepped`, `held`).
        fn plain_step(&mut self) -> Result<(), String> {
            let (pos, last) = (self.c.pos, self.last);
            self.p.before_step(&mut self.c, pos, last)?;
            self.c.ran(Ar::Step, 1);
            let pos = self.c.pos;
            let next = 1000 + pos;
            self.p.stepped(&mut self.c, pos, last, next).map_err(fail)?;
            self.last = next;
            Ok(())
        }

        /// A drafted pass that keeps `kept` rows (all of them when `None`):
        /// the chain, the verify, the accept's record, the commit.
        fn drafted_pass(&mut self, kept: Option<usize>) -> Result<(), String> {
            let (p0, ctx, last) = (self.c.pos, self.c.ctx, self.last);
            let chain = self
                .p
                .chain(p0, ctx, last, SHAPE.width)
                .map_err(|e| e.to_string())?
                .ok_or("a drafted pass of a draft that skips")?;
            self.c.walk(&chain.refresh.as_walk(chain.mode))?;
            let end = chain.refresh.pos0 as usize + chain.refresh.tokens.len();
            self.c.store = end + chain.own;
            let mut rows = vec![last];
            rows.extend((0..=chain.own as u32).map(|i| 500 + i));
            self.c.ran(Ar::Pass, rows.len());
            let kept = kept.map_or(rows.len(), |k| k.min(rows.len()));
            let out: Vec<u32> = (0..rows.len())
                .map(|i| match rows.get(i + 1) {
                    Some(&id) if i + 1 < kept => id,
                    _ => 2000 + u32::try_from(i).unwrap(),
                })
                .collect();
            self.p
                .record(p0, &rows, &out, kept)
                .map_err(|e| e.to_string())?;
            self.c.commit(p0, kept);
            self.last = out[kept - 1];
            Ok(())
        }

        /// The refresh the next chain opens with.
        fn next(&self) -> &Refresh<Ar> {
            self.p.next.as_ref().expect("a refresh waits")
        }
    }

    /// A rig after the prompt of [`IDS`] in units of four rows through the
    /// ubatch arena.
    fn after_prompt(ctx: u32) -> Rig {
        let mut rig = Rig::new(ctx);
        rig.prompt(&IDS, 4, Ar::Ubatch).unwrap();
        rig.c.take();
        rig
    }

    /// The walks a prompt call runs to fill the store: position 0's row
    /// beside the zero row, then every row after it but the last through the
    /// unit that holds its hidden row, in runs of the store's rows.
    #[test]
    fn prompt_fills_the_store_and_leaves_the_last_row_for_begin() {
        let mut rig = Rig::new(64);
        rig.prompt(&IDS, 4, Ar::Ubatch).unwrap();
        assert_eq!(
            rig.c.take(),
            [
                seen(STORE, 0, &[10], Source::Zero),
                seen(STORE, 1, &[11, 12, 13, 14], rows_of(Ar::Ubatch, 0)),
                seen(STORE, 5, &[15, 16, 17, 18], rows_of(Ar::Ubatch, 0)),
            ]
        );
        assert_eq!((rig.c.store, rig.p.last_unit), (9, Some((Ar::Ubatch, 1))));
        assert!(rig.p.next.is_none());

        let mut rig = Rig::new(64);
        rig.prompt(&IDS, 9, Ar::Ubatch).unwrap();
        assert_eq!(
            rig.c.take(),
            [
                seen(STORE, 0, &[10], Source::Zero),
                seen(STORE, 1, &[11, 12, 13, 14], rows_of(Ar::Ubatch, 0)),
                seen(STORE, 5, &[15, 16, 17, 18], rows_of(Ar::Ubatch, 4)),
            ]
        );
    }

    /// The server's cut: the prompt call, then its last id as one plain
    /// step, then a pass the width chooser holds back (its hook walks the
    /// anchor the cut step recorded, before its step overwrites the arena
    /// the anchor reads), then a drafted pass. After every call the store
    /// holds no position past the target's — what a sequence state needs.
    #[test]
    fn a_held_pass_after_the_cut_step_takes_the_position_in() {
        let mut rig = after_prompt(64);

        rig.plain_step().unwrap_or_else(|e| panic!("{e}"));
        let cut = rig.c.take();
        let anchor = rig.next().clone();
        // The store never holds a position past the target's: at the cut
        // step's end it stands exactly at it.
        assert_eq!((rig.c.store, rig.c.pos), (10, 10));
        rig.plain_step().unwrap_or_else(|e| panic!("{e}"));
        let held = rig.c.take();
        assert_eq!((rig.c.store, rig.c.pos), (11, 11));
        rig.drafted_pass(None).unwrap_or_else(|e| panic!("{e}"));
        let drafted = rig.c.take();

        assert_eq!(cut, [seen(HEAD, 9, &[1009], rows_of(Ar::Ubatch, 0))]);
        assert_eq!((anchor.pos0, anchor.walk), (10, Ar::Step));
        assert_eq!(held, [seen(HEAD, 10, &[1010], rows_of(Ar::Step, 0))]);
        assert_eq!(drafted, [seen(HEAD, 11, &[1011], rows_of(Ar::Step, 0))]);
    }

    /// The generation loop's start: the prompt call, then `begin` with the
    /// prompt's argmax, then drafted passes. `begin` records the last unit's
    /// last row as the anchor, the first chain walks it, and each pass's
    /// accept records the next refresh from the verify's arena.
    #[test]
    fn begin_then_drafted_passes() {
        let mut rig = after_prompt(64);
        rig.begin().unwrap();
        let anchor = rig.next();
        assert_eq!(
            (&anchor.tokens[..], anchor.pos0, anchor.walk, anchor.first),
            (&[1009][..], 9, Ar::Ubatch, 0)
        );

        rig.drafted_pass(None).unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 9, &[1009], rows_of(Ar::Ubatch, 0))]
        );
        let r = rig.next();
        assert_eq!(
            (&r.tokens[..], r.pos0, r.walk),
            (&[500, 501, 502, 2003][..], 10, Ar::Pass)
        );

        rig.drafted_pass(Some(2)).unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 10, &[500, 501, 502, 2003], rows_of(Ar::Pass, 0))]
        );
        let r = rig.next();
        assert_eq!(
            (&r.tokens[..], r.pos0, r.walk),
            (&[500, 2001][..], 14, Ar::Pass)
        );
    }

    /// `begin` leaves the anchor a step already recorded, and refuses by
    /// name a prompt the draft never saw.
    #[test]
    fn begin_keeps_the_cut_steps_anchor_and_refuses_an_unseen_prompt() {
        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        let anchor = rig.next().clone();
        rig.begin().unwrap();
        let kept = rig.next();
        assert_eq!(
            (&kept.tokens, kept.pos0, kept.walk),
            (&anchor.tokens, anchor.pos0, anchor.walk)
        );

        let mut fresh = Rig::new(64);
        let why = fresh.begin().unwrap_err();
        assert!(why.ends_with("a prompt the draft never saw"), "{why}");
    }

    /// Two of the server's steps in a row: the second one's waiting rows
    /// (the first one's anchor, still in the step's arena) are walked before
    /// its step overwrites them, and its own anchor waits for the next call.
    #[test]
    fn two_plain_steps_in_a_row() {
        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        rig.c.take();
        rig.plain_step().unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 10, &[1010], rows_of(Ar::Step, 0))]
        );
        assert_eq!(
            (rig.next().pos0, rig.next().walk, rig.c.store, rig.c.pos),
            (11, Ar::Step, 11, 11)
        );
    }

    /// A held pass in the middle of a generation: the refresh the accept
    /// recorded (in the verify's arena) is walked before the step, and a
    /// second held pass in a row walks the first one's anchor the same way.
    #[test]
    fn a_held_pass_mid_generation() {
        let mut rig = after_prompt(64);
        rig.begin().unwrap();
        rig.drafted_pass(None).unwrap();
        rig.c.take();

        rig.plain_step().unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 10, &[500, 501, 502, 2003], rows_of(Ar::Pass, 0))]
        );
        rig.plain_step().unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 14, &[1014], rows_of(Ar::Step, 0))]
        );
        rig.drafted_pass(None).unwrap();
    }

    /// A step that ends at the context: its row is left unwalked, since no
    /// chain runs there, and the refresh it records waits for a call that
    /// never comes.
    #[test]
    fn a_step_past_the_context_leaves_its_row_unwalked() {
        let mut rig = after_prompt(11);
        rig.plain_step().unwrap();
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 9, &[1009], rows_of(Ar::Ubatch, 0))]
        );
        rig.plain_step().unwrap();
        assert_eq!(rig.c.pos, 11);
        assert_eq!(
            rig.c.take(),
            [seen(HEAD, 10, &[1010], rows_of(Ar::Step, 0))]
        );
        let r = rig.next();
        assert_eq!((r.pos0, r.walk), (11, Ar::Step));
    }

    /// A prompt fed by steps leaves its last row in the step's arena, which
    /// the first step overwrites before anything walks it: the one refusal
    /// left, by name, with the rows' position.
    #[test]
    fn a_prompt_fed_by_steps_is_refused_at_the_first_step() {
        let mut rig = Rig::new(64);
        rig.prompt(&IDS, 1, Ar::Step).unwrap();
        let why = rig.plain_step().unwrap_err();
        assert!(
            why.contains(
                "a step after rows whose hidden rows the step's own arena held (a prompt fed by \
                 steps): the step overwrote them (rows at position 9)"
            ),
            "{why}"
        );
        assert!(rig.p.next.is_none() && rig.p.last_unit.is_none());
    }

    /// A draft that skips does nothing for any call, proposes nothing, and
    /// starts over at a prompt call from position 0.
    #[test]
    fn a_skipping_draft_does_nothing_until_a_prompt_from_position_0() {
        let mut rig = Rig::new(64);
        rig.c.pos = 5;
        rig.c.store = 5;
        rig.prompt(&IDS[..3], 3, Ar::Ubatch).unwrap();
        assert!(rig.p.skipping());
        rig.c.take();

        rig.plain_step().unwrap();
        rig.plain_step().unwrap();
        assert!(rig.c.take().is_empty());
        assert!(rig.p.next.is_none() && rig.p.last_unit.is_none());
        assert!(
            rig.p
                .chain(rig.c.pos, rig.c.ctx, rig.last, 3)
                .unwrap()
                .is_none()
        );
        assert_eq!(rig.p.park().skip, rig.p.skip);

        rig.c.pos = 0;
        rig.prompt(&IDS, 4, Ar::Ubatch).unwrap();
        assert!(!rig.p.skipping());
    }

    /// A prompt call that continues the held sequence walks the rows the
    /// last call left waiting, its first id their last row, before its own
    /// units, which carry no zero row.
    #[test]
    fn a_continuing_prompt_walks_the_waiting_rows_with_its_first_id() {
        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        rig.c.take();
        rig.prompt(&[20, 21, 22], 3, Ar::Ubatch).unwrap();
        assert_eq!(
            rig.c.take(),
            [
                seen(HEAD, 10, &[20], rows_of(Ar::Step, 0)),
                seen(STORE, 11, &[21, 22], rows_of(Ar::Ubatch, 0)),
            ]
        );
        assert_eq!(
            rig.p.take_joined(),
            Some(Join {
                start: 10,
                caught_up: 1,
                skipped: None
            })
        );
        assert!(rig.p.take_joined().is_none());
    }

    /// A join the draft cannot make skips for a named reason, each of the
    /// three: nothing waits, the waiting rows end elsewhere, the store is
    /// behind them.
    #[test]
    fn a_join_the_draft_cannot_make_skips_for_its_reason() {
        let mut rig = Rig::new(64);
        rig.c.pos = 5;
        rig.prompt(&IDS[..2], 2, Ar::Ubatch).unwrap();
        let got = rig.p.take_joined().unwrap();
        assert_eq!(
            (got.start, got.caught_up, got.skipped),
            (5, 0, Some(NOTHING_WAITS))
        );

        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        rig.c.pos += 2;
        rig.prompt(&IDS[..2], 2, Ar::Ubatch).unwrap();
        let got = rig.p.take_joined().unwrap();
        assert_eq!((got.start, got.skipped), (12, Some(ENDS_ELSEWHERE)));

        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        rig.c.store = 5;
        rig.prompt(&IDS[..2], 2, Ar::Ubatch).unwrap();
        let got = rig.p.take_joined().unwrap();
        assert_eq!((got.start, got.skipped), (10, Some(STORE_BEHIND)));
        assert!(rig.p.skipping());
    }

    /// The chain takes the refresh with its last row the token at the
    /// target's position, and runs as many own walks as the room, the width
    /// and the context allow: a proposal of `own + 1` ids verifies `own + 2`
    /// rows through `here + own + 1`, so the context's arm leaves that last
    /// row at the context's last — one shallower a position the nearer the
    /// target stands to the end.
    #[test]
    fn a_chain_takes_the_refresh_and_caps_its_own_walks() {
        for (room, ctx, own) in [
            (8, 64, 2),
            (2, 64, 1),
            (1, 64, 0),
            (8, 13, 2),
            (8, 12, 1),
            (8, 11, 0),
        ] {
            let mut rig = after_prompt(ctx);
            rig.begin().unwrap();
            let chain = rig
                .p
                .chain(9, ctx, 77, room)
                .unwrap()
                .expect("a draft that drafts");
            assert_eq!(
                (&chain.refresh.tokens[..], chain.own, chain.mode),
                (&[77][..], own, HEAD),
                "room {room}, ctx {ctx}"
            );
            assert!(rig.p.next.is_none());
        }
    }

    /// The chain's refusals, by name: no refresh waits, the refresh ends
    /// where the target is not, no room.
    #[test]
    fn a_chain_refuses_by_name() {
        let refused = |rig: &mut Rig, here, room| {
            rig.p
                .chain(here, 64, 77, room)
                .map(|_| ())
                .unwrap_err()
                .to_string()
        };

        let mut rig = after_prompt(64);
        let why = refused(&mut rig, 9, 3);
        assert!(
            why.ends_with(
                "a proposal before the draft's refresh (its prompt call, or the accept before it)"
            ),
            "{why}"
        );

        rig.begin().unwrap();
        let why = refused(&mut rig, 7, 3);
        assert!(
            why.ends_with(
                "a refresh of 1 rows ending at 10, where the target stands at 7 (its next \
                 token's row ends at 8)"
            ),
            "{why}"
        );

        rig.begin().unwrap();
        let why = refused(&mut rig, 9, 0);
        assert!(why.ends_with("a proposal into no room"), "{why}");
    }

    /// A chain whose narrowest verify does not fit the context: at the
    /// context's last row (the refresh ends at the context, so even a
    /// proposal of one id verifies a row past it) and at its end, refused by
    /// name — a clamp would hand the caller a proposal it cannot verify.
    #[test]
    fn a_chain_past_the_contexts_last_row_is_refused() {
        let mut rig = after_prompt(10);
        rig.begin().unwrap();
        let why = rig.p.chain(9, 10, 77, 3).unwrap_err().to_string();
        assert!(
            why.ends_with(
                "a chain at position 9 of a context of 10: its narrowest verify would run the \
                 rows 9 and 10, past the context's end"
            ),
            "{why}"
        );

        let mut rig = after_prompt(9);
        rig.begin().unwrap();
        let why = rig.p.chain(9, 9, 77, 3).unwrap_err().to_string();
        assert!(
            why.ends_with(
                "a chain at position 9 of a context of 9: its narrowest verify would run the \
                 rows 9 and 10, past the context's end"
            ),
            "{why}"
        );
    }

    /// A verify that kept no row took no id: refused by name.
    #[test]
    fn a_record_of_no_kept_row_is_refused() {
        let mut rig = after_prompt(64);
        let why = rig
            .p
            .record(9, &[1, 2], &[2, 3], 0)
            .unwrap_err()
            .to_string();
        assert!(
            why.ends_with("a verify that kept 0 rows took no id"),
            "{why}"
        );
        let why = rig.p.record(9, &[1, 2], &[2], 2).unwrap_err().to_string();
        assert!(
            why.ends_with("a verify that kept 2 rows took no id"),
            "{why}"
        );
    }

    /// A sequence state keeps the rows waiting in the step's and the verify's
    /// arenas, and parks a draft whose rows sit in a prompt call's arena as
    /// skipping, by name; the put-back restores what was kept and clears the
    /// join.
    #[test]
    fn park_keeps_the_step_and_verify_arenas_only() {
        let mut rig = after_prompt(64);
        rig.p.joined = Some(Join {
            start: 3,
            caught_up: 0,
            skipped: None,
        });

        // The prompt's last unit waits in a prompt arena.
        assert_eq!(rig.p.park().skip, Some(PROMPT_ARENA));
        rig.begin().unwrap();
        assert_eq!(rig.p.park().skip, Some(PROMPT_ARENA));

        rig.drafted_pass(None).unwrap();
        let verify = rig.p.park();
        assert!(verify.skip.is_none() && verify.next.as_ref().unwrap().walk == Ar::Pass);
        rig.plain_step().unwrap();
        let step = rig.p.park();
        assert!(step.skip.is_none() && step.next.as_ref().unwrap().walk == Ar::Step);

        rig.p.restart();
        assert!(rig.p.next.is_none() && rig.p.joined.is_none());
        rig.p.unpark(&verify);
        assert_eq!(rig.next().walk, Ar::Pass);
        rig.p.unpark(&step);
        assert!(rig.next().walk == Ar::Step && rig.p.joined.is_none());

        let mut by_steps = Rig::new(64);
        by_steps.prompt(&IDS, 1, Ar::Step).unwrap();
        let parked = by_steps.p.park();
        assert_eq!((parked.skip, parked.last_unit), (None, Some((Ar::Step, 1))));

        assert_eq!(Rig::new(64).p.park().skip, None);
        let mut skipping = Rig::new(64);
        skipping.p.skip = Some(NOTHING_WAITS);
        assert_eq!(skipping.p.park().skip, Some(NOTHING_WAITS));
    }

    /// A walk that fails ends the call with the error, the target unmoved
    /// (the hook's walk runs before the step) and the rows it was to walk
    /// taken without being walked: the draft holds nothing waiting, and
    /// nothing it skips — the next chain refuses by name.
    #[test]
    fn a_failed_waiting_rows_walk_leaves_the_target_unmoved() {
        let mut rig = after_prompt(64);
        rig.plain_step().unwrap();
        rig.c.take();
        rig.c.fail_after = Some(0);
        let why = rig.plain_step().unwrap_err();
        assert_eq!(why, "the card refused the walk");
        assert_eq!(rig.c.pos, 10);
        assert!(rig.p.next.is_none() && rig.p.last_unit.is_none());
        assert!(!rig.p.skipping());
    }

    /// No call order the seats and the generation loop can produce leaves
    /// the draft's store holding a position at or past the target's — what
    /// a sequence state needs to save the draft's side — and none ends in a
    /// refusal: every sequence of up to six calls from a plain step (the
    /// seat's step and a pass the width chooser holds back, one order) and
    /// a drafted pass (keeping one row or all) over a prompt, started by
    /// `begin` or by the server's cut step, runs without a refusal, each of
    /// its walks one the card model accepts, and after every call the store
    /// holds no position past the target's. The runtime's own plain loop —
    /// a step with no walk before it, which is what refused here once —
    /// never follows a cut step (its steps come only after the draft's own
    /// proposal), so it is not among the calls.
    ///
    /// The contexts the sequences run at are one the calls cannot reach the
    /// end of and one they can (the prompt's nine ids plus three rows):
    /// there a drafted pass runs while its narrowest verify fits the
    /// context — the policy's own bound, which the runtime's `Stop` and the
    /// server's pass gate never test, both gating a pass by its widest
    /// rows — and no call runs at the context's end, where the callers
    /// stop, so the end-of-context chains are drawn with the calls the
    /// callers make around them.
    #[test]
    fn no_call_sequence_leaves_the_store_past_the_target_or_refuses() {
        #[derive(Clone, Copy, Debug)]
        enum Call {
            Plain,
            Drafted(Option<usize>),
        }
        const CALLS: [Call; 3] = [Call::Plain, Call::Drafted(Some(1)), Call::Drafted(None)];

        fn run(rig: &mut Rig, calls: &[Call]) -> Result<(), String> {
            for (i, call) in calls.iter().enumerate() {
                let (pos, ctx) = (rig.c.pos as usize, rig.c.ctx as usize);
                if pos >= ctx {
                    // The context's end: the callers stop here (`Stop`'s Ctx,
                    // the server's truncated turn).
                    return Ok(());
                }
                // A drafted pass runs while its narrowest verify fits the
                // context; closer to the end the caller steps.
                let call = match *call {
                    Call::Drafted(_) if pos + 2 > ctx => Call::Plain,
                    c => c,
                };
                match call {
                    Call::Plain => rig.plain_step(),
                    Call::Drafted(kept) => rig.drafted_pass(kept),
                }
                .map_err(|e| format!("call {} ({call:?}): {e}", i + 1))?;
                if rig.c.store > rig.c.pos as usize {
                    return Err(format!(
                        "call {} ({call:?}): the store holds {} positions, the target at {}",
                        i + 1,
                        rig.c.store,
                        rig.c.pos
                    ));
                }
            }
            Ok(())
        }

        fn each(len: usize, tail: &mut Vec<Call>, f: &mut dyn FnMut(&[Call])) {
            f(tail);
            if tail.len() < len {
                for c in CALLS {
                    tail.push(c);
                    each(len, tail, f);
                    tail.pop();
                }
            }
        }

        let mut count = 0usize;
        for ctx in [64, 12] {
            for arena in [Ar::Ubatch, Ar::Pass] {
                for begun in [true, false] {
                    let (first, len) = if begun {
                        (&[][..], 6)
                    } else {
                        (&[Call::Plain][..], 5)
                    };
                    each(len, &mut Vec::new(), &mut |calls| {
                        let mut rig = Rig::new(ctx as u32);
                        rig.prompt(&IDS, 4, arena).unwrap();
                        if begun {
                            rig.begin().unwrap();
                        }
                        let all: Vec<Call> = first.iter().chain(calls).copied().collect();
                        if let Err(why) = run(&mut rig, &all) {
                            panic!("ctx {ctx}, {arena:?} prompt, begun {begun}, {all:?}: {why}");
                        }
                        count += 1;
                    });
                }
            }
        }
        assert_eq!(count, 2 * 2 * (1093 + 364));
    }
}
