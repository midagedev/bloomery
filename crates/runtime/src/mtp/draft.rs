//! [`MtpDraft`]: the runtime rule's [`Draft`] over a [`Host`]'s draft layer,
//! the window's host policy ([`Policy`]) driving the walks the host runs.

use super::host::{Host, On, Unit};
use super::{Chain, Exec, Fail, Join, Parked, Policy, Refused, Shape, WHAT, WalkMode};
use crate::prompt::CallEnd;
use crate::{Draft, TapNeed};

/// One drafted window as the draft saw it ([`MtpDraft::keep_windows`]): the
/// target's position before its verify, the proposal's ids and each one's
/// probability among the head's rows, how many of those ids the pass
/// verified (the width chooser's cut; the proposal's whole length when
/// nothing cut it) and how many of them the target kept.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowDraft {
    pub pos: u32,
    pub ids: Vec<u32>,
    pub p: Vec<f32>,
    /// The ids the pass verified, at most the proposal's length.
    pub width: usize,
    pub accepted: usize,
}

/// What [`MtpDraft::keep_windows`] keeps: the last proposal until its
/// accept, and every window since the last take.
struct Windows {
    pending: Option<(Vec<u32>, Vec<f32>)>,
    kept: Vec<WindowDraft>,
}

/// The MTP draft over a host's draft layer (the module doc): the host
/// policy of the windows ([`Policy`]: which rows wait, which walks run, the
/// refresh the next chain opens with) and the host that runs the walks —
/// which head its load opened, how the prompt is fed, the walks themselves. A
/// caller drives it as `Speculative<MtpDraft<H>, M>` with `M` = [`Host::WIDTH`]
/// + 1, which the target's `with_draft` holds to the draft's width.
pub struct MtpDraft<H: Host> {
    /// The prompt path.
    path: H::Path,
    head: H::Head,
    /// What the window knows between two calls, and every decision a call
    /// makes of it.
    policy: Policy<H::Arena>,
    /// The windows kept under [`MtpDraft::keep_windows`]; `None`, nothing
    /// is kept.
    windows: Option<Windows>,
    /// The last chain's probabilities, when it read them: a host whose
    /// readback holds them ([`Host::PROBS`]) behind the width chooser, or the
    /// kept windows.
    p: Vec<f32>,
}

impl<H: Host> MtpDraft<H> {
    /// The draft on a host whose load opened `head` (a row list or the full
    /// vocabulary), the prompt fed by `path`, the chains run in `mode`,
    /// `hidden_width` the values of one hidden row a walk reads. Load-time
    /// only.
    pub fn new(path: H::Path, head: H::Head, mode: WalkMode, hidden_width: usize) -> Self {
        let () = H::FITS;
        MtpDraft {
            path,
            head,
            policy: Policy::new(Shape {
                step: H::STEP_ARENA,
                verify: H::VERIFY_ARENA,
                mode,
                width: H::WIDTH,
                store_rows: H::STORE_ROWS,
                hidden_width,
            }),
            windows: None,
            p: vec![0.0; H::WIDTH],
        }
    }

    /// Keep every drafted window from here on ([`WindowDraft`]): the chain
    /// hands back each id's probability, which its one readback already
    /// holds. Load-time only.
    pub fn keep_windows(&mut self) {
        self.windows = Some(Windows {
            pending: None,
            kept: Vec::new(),
        });
    }

    /// The windows kept since the last take, in order; empty when
    /// [`MtpDraft::keep_windows`] was never called.
    pub fn take_windows(&mut self) -> Vec<WindowDraft> {
        self.windows
            .as_mut()
            .map(|w| std::mem::take(&mut w.kept))
            .unwrap_or_default()
    }

    /// The join of the last call that continued a held sequence, once: a
    /// second take is `None` until another call joins.
    pub fn take_joined(&mut self) -> Option<Join> {
        self.policy.take_joined()
    }

    /// The head the load opened.
    #[must_use]
    pub fn head(&self) -> H::Head {
        self.head
    }

    /// The draft's own state started over: no refresh waits, no skip, the
    /// target's reset having emptied the draft's store. The session's reset
    /// calls it.
    pub fn restart(&mut self) {
        self.policy.restart();
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// Whether the draft proposes nothing until the next prompt call from
    /// position 0 or a restart ([`Join::skipped`] names why).
    #[must_use]
    pub fn skipping(&self) -> bool {
        self.policy.skipping()
    }

    /// What the draft holds of the target's sequence between two calls, for
    /// the caller's sequence state ([`MtpDraft::unpark`] puts it back): the
    /// rows waiting for its next walk when they sit in the step's or the
    /// verify's arena ([`Host::STEP_ARENA`], [`Host::VERIFY_ARENA`]), or why
    /// it proposes nothing — its own skip, or rows a prompt call's arena
    /// holds, which a sequence state does not keep. The caller's state keeps
    /// the draft's store and those two arenas' rows with the positions they
    /// hold, so the walk after the put-back reads the rows it would have
    /// read; a host whose walk does not refuse an arena that holds other
    /// positions must not park.
    #[must_use]
    pub fn park(&self) -> Parked<H::Arena> {
        self.policy.park()
    }

    /// `p`, which [`MtpDraft::park`] took, back in place once the target's
    /// sequence stands where it stood then (the caller's state put back, the
    /// draft's store and the two arenas with it): the next call walks the
    /// waiting rows as it would have, or skips for the parked why, its
    /// [`Join`] naming it.
    pub fn unpark(&mut self, p: &Parked<H::Arena>) {
        self.policy.unpark(p);
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// The target cut back to `pos`, or a state put back as its first `pos`
    /// positions, where the host rejoined the draft to the sequence: its
    /// store holds every position below `pos` and the step's arena
    /// ([`Host::STEP_ARENA`]) the target's hidden row at `pos - 1`. The next
    /// call walks that row with the id it feeds at `pos` and drafts on, as
    /// after a plain step at `pos - 1`; nothing is skipped, and the next
    /// [`Join`] is the first since.
    pub fn rejoin(&mut self, pos: u32) {
        self.policy.rejoin(pos);
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// The window's policy, for a caller that runs the decisions itself on an
    /// executor of its own (`before_step`, `chain`, `prompt_start`, `unit`):
    /// the state a restart, an unpark or a rejoin also clears from the kept
    /// windows stays with [`MtpDraft`]'s own calls.
    pub fn policy_mut(&mut self) -> &mut Policy<H::Arena> {
        &mut self.policy
    }

    /// Whether the draft keeps its windows ([`MtpDraft::keep_windows`]): a
    /// chain then reads each id's probability.
    #[must_use]
    pub fn keeps_windows(&self) -> bool {
        self.windows.is_some()
    }

    /// The place a chain writes each id's probability to when the draft keeps
    /// its windows; `None` otherwise.
    pub fn kept_probs(&mut self) -> Option<&mut [f32]> {
        self.windows.is_some().then_some(&mut self.p[..])
    }

    /// The proposal `ids` a chain made, its probabilities in the draft's
    /// own `p`, kept until its accept under [`MtpDraft::keep_windows`].
    pub fn proposed(&mut self, ids: &[u32]) {
        if let Some(w) = &mut self.windows {
            w.pending = Some((ids.to_vec(), self.p[..ids.len()].to_vec()));
        }
    }

    /// The prompt call's one core ([`Draft::prompt`], and a host's own call
    /// that walks the draft on an executor of its own, a call with media rows
    /// among them): `x` is the executor of the call's walks, handed back to
    /// `call` — the target's prompt call of `ids` from `start` — with the
    /// sink that walks the draft over each unit as it completes, or `None`
    /// when the draft cannot join and the call is the target's alone. The
    /// join's walk runs on `x` before `call` does.
    ///
    /// # Errors
    ///
    /// The executor's, from the join's walk, a unit's, or `call`'s own.
    pub fn prompt_joined<X, R>(
        &mut self,
        x: &mut X,
        start: u32,
        ids: &[u32],
        call: impl FnOnce(
            &mut X,
            Option<&mut Unit<'_, H::Arena, <X as Exec<H::Arena>>::Err>>,
        ) -> Result<R, <X as Exec<H::Arena>>::Err>,
    ) -> Result<R, <X as Exec<H::Arena>>::Err>
    where
        X: Exec<H::Arena>,
    {
        let Some(mut join) = self.policy.prompt_start(x, start, ids)? else {
            return call(x, None);
        };
        let policy = &mut self.policy;
        let mut sink =
            |u: &mut dyn Exec<H::Arena, Err = <X as Exec<H::Arena>>::Err>,
             walk: H::Arena,
             first: u32,
             rows: usize| { policy.unit(&mut join, u, walk, first, rows) };
        call(x, Some(&mut sink))
    }
}

impl<H: Host> MtpDraft<H> {
    /// Preserve the prompt's waiting row before another resident sequence
    /// can overwrite its shared arena. The target is never run again.
    ///
    /// # Errors
    ///
    /// The host's copy.
    pub fn retain_prompt(&mut self, t: &mut H) -> Result<(), H::Error> {
        let pos = t.pos();
        self.policy
            .retain_prompt(pos, |walk, first, pos| {
                t.retain_prompt_row(walk, first, pos)
            })
            .map_err(Into::into)
    }

    /// [`Draft::prompt`] ending as `end` asks: the draft walked over the
    /// call's units as they complete whatever its end, the argmax after the
    /// last when the call runs the head.
    ///
    /// # Errors
    ///
    /// The host's, from the join's walk, a unit's or the call's own.
    pub fn prompt_call(
        &mut self,
        t: &mut H,
        ids: &[u32],
        end: CallEnd,
    ) -> Result<Option<u32>, H::Error> {
        let (path, head, start) = (self.path, self.head, t.pos());
        t.prompt_scope(|t| {
            self.prompt_joined(&mut On { t, head }, start, ids, |on, sink| {
                on.t.prompt_with(ids, path, end, head, sink)
            })
        })
        .map_err(Into::into)
    }

    /// Before a plain step of `last` at the target's position: a refresh an
    /// earlier step or window left waiting reads an arena the step may
    /// overwrite (its own), so it is walked now, its last row `last` — a
    /// first step that continues the held sequence joins it here. A draft
    /// that cannot walk it skips, the [`Join`] naming why
    /// ([`MtpDraft::take_joined`]). A prompt call's anchor is left for
    /// [`Draft::stepped`]. A server seat calls it before each of its steps,
    /// and the width chooser makes the same walk through
    /// [`Draft::before_plain`] before each pass it holds back.
    ///
    /// # Errors
    ///
    /// The walk's.
    pub fn before_step(&mut self, t: &mut H, last: u32) -> Result<(), H::Error> {
        let pos = t.pos();
        let mut on = On { t, head: self.head };
        self.policy
            .before_step(&mut on, pos, last)
            .map_err(Into::into)
    }

    /// The chain `c` the policy decided, run on the target's selected
    /// sequence ([`Host::chain`]) into `out`, with each id's probability into
    /// the draft's own `p` when `read_p`, and kept ([`MtpDraft::proposed`]).
    ///
    /// # Errors
    ///
    /// The host's.
    pub fn run_chain(
        &mut self,
        t: &mut H,
        c: &Chain<H::Arena>,
        out: &mut [u32],
        read_p: bool,
    ) -> Result<usize, H::Error> {
        let n = t
            .chain(c, self.head, out, read_p.then_some(&mut self.p[..]))
            .map_err(Into::<H::Error>::into)?;
        self.proposed(&out[..n]);
        Ok(n)
    }
}

impl<H: Host> MtpDraft<H>
where
    H::Error: From<Refused>,
{
    /// The next refresh recorded for a verify of `rows` that ran from
    /// position `p0`, keeping its first `accepted` rows, `out` the id the
    /// pass took at each ([`Draft::accept`]), before the commit takes the
    /// rest back: the kept rows at the positions after the verify's first,
    /// each with the hidden row the verify's row before it wrote
    /// ([`Host::VERIFY_ARENA`]), the last row the id the pass took after
    /// the last kept row — the argmax on a greedy pass, the draw on a
    /// sampled one. What [`Draft::accept`] records, `p0` given by the caller
    /// that ran the verify.
    ///
    /// # Errors
    ///
    /// Refused by name: a verify that kept no row, and a window kept whose
    /// proposal the draft did not keep.
    pub fn record(
        &mut self,
        p0: u32,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), H::Error> {
        self.policy.record(p0, rows, out, accepted)?;
        if let Some(w) = &mut self.windows {
            let (ids, p) = w.pending.take().ok_or_else(|| {
                Refused(format!(
                    "{WHAT}: an accept of a window whose proposal the draft did not keep"
                ))
            })?;
            w.kept.push(WindowDraft {
                pos: p0,
                ids,
                p,
                width: rows.len() - 1,
                accepted: accepted - 1,
            });
        }
        Ok(())
    }

    /// A plain row of `last` that a pass of several slots ran at the target's
    /// position `p0`, `next` the id taken after it, its rows waiting walked
    /// before the pass ([`MtpDraft::before_step`]): `next` at `p0 + 1`, with
    /// the hidden row the pass's row wrote ([`Host::VERIFY_ARENA`]), as
    /// the next refresh, the refresh [`Draft::stepped`] records after a step
    /// of its own arena. A window of depth zero in a pass of several slots
    /// and a seat's round of plain steps run as one pass both end here.
    ///
    /// # Errors
    ///
    /// Refused by name as [`MtpDraft::record`] refuses a verify that kept no
    /// row.
    pub fn passed_plain(&mut self, p0: u32, last: u32, next: u32) -> Result<(), H::Error> {
        Ok(self.policy.record(p0, &[last], &[next], 1)?)
    }

    /// One chain, one readback (the module doc): the recorded refresh's
    /// walk — its last row `last`, the token at the target's position, whose
    /// prediction is the first proposal — then own walks while the verify of
    /// the proposal fits the context and the proposal fits `out`: at most
    /// `out.len()` ids, so a caller caps a window's depth by the room it
    /// hands. No room is refused by name. The refresh's last row is `last`
    /// whatever the call that recorded it took there: a step tells the draft
    /// its argmax, and a sampled request feeds its draw. With `read_p`, each
    /// id's probability as the same readback holds it, into the draft's own
    /// `p`.
    fn chain(
        &mut self,
        t: &mut H,
        last: u32,
        out: &mut [u32],
        read_p: bool,
    ) -> Result<usize, H::Error> {
        match self.policy.chain(t.pos(), t.ctx(), last, out.len())? {
            None => Ok(0),
            Some(c) => self.run_chain(t, &c, out, read_p),
        }
    }
}

impl<H: Host> Draft<H> for MtpDraft<H>
where
    H::Error: From<Refused>,
{
    /// The most ids a proposal holds: the host's [`Host::WIDTH`].
    const WIDTH: usize = H::WIDTH;
    const TAPS: TapNeed = TapNeed::Final;
    /// Each id past the first is one of the chain's own walks, which stop at
    /// the room the proposal is handed.
    const CHAIN: bool = true;

    /// The prompt through the target's own call, the draft walked over its
    /// units as they complete (the module doc): position 0 first — the
    /// target holds no hidden row before it — or, for a call that continues
    /// the held sequence, the rows the last call left waiting, the call's
    /// first id their last ([`Join`]); then every unit's rows from its second
    /// on, each with the hidden row its own arena holds for the position
    /// before it, the last unit's last row left for `begin`. A draft that
    /// cannot join skips: the target's call alone.
    fn prompt(&mut self, t: &mut H, ids: &[u32]) -> Result<u32, H::Error> {
        let next = self.prompt_call(t, ids, CallEnd::Argmax)?;
        Ok(next.ok_or_else(|| {
            Refused(format!(
                "{WHAT}: a prompt call that ends at the argmax gave no head token after the prompt"
            ))
        })?)
    }

    /// The next window's anchor: the target's own token `first` at the
    /// prompt's end, with the hidden row the prompt's last position wrote —
    /// unless a step since the prompt call already recorded one (the
    /// server's cut: the prompt less its last id, then the last id as a
    /// step, [`Draft::stepped`] recording the anchor the step's window
    /// proposes from).
    fn begin(&mut self, t: &H, _prompt: &[u32], first: u32) -> Result<(), H::Error> {
        Ok(self.policy.begin(t.pos(), first)?)
    }

    /// One chain, one readback ([`Host::chain`]); each id's
    /// probability read back only for the kept windows
    /// ([`MtpDraft::keep_windows`]).
    fn propose(&mut self, t: &mut H, last: u32, out: &mut [u32]) -> Result<usize, H::Error> {
        self.chain(t, last, out, self.windows.is_some())
    }

    /// The same chain with each proposed id's probability among the head's
    /// rows written to `p` beside it, from the readback that already holds
    /// it ([`Host::PROBS`]); a host whose readback holds none reports
    /// certainty, every 1.
    fn propose_p(
        &mut self,
        t: &mut H,
        last: u32,
        out: &mut [u32],
        p: &mut [f32],
    ) -> Result<usize, H::Error> {
        let n = self.chain(t, last, out, H::PROBS || self.windows.is_some())?;
        let places = p.len();
        let p = p.get_mut(..n).ok_or_else(|| {
            Refused(format!(
                "{WHAT}: a proposal of {n} ids into {places} probabilities"
            ))
        })?;
        if H::PROBS {
            p.copy_from_slice(&self.p[..n]);
        } else {
            p.fill(1.0);
        }
        Ok(n)
    }

    /// The last proposal was never verified: its kept window, if any, is
    /// dropped, so the next accept records the window that ran.
    fn unproposed(&mut self) {
        if let Some(w) = &mut self.windows {
            w.pending = None;
        }
    }

    /// The next refresh recorded before the commit takes the rejected rows
    /// back ([`MtpDraft::record`]), the verify's first position the one its
    /// rows stood the target past.
    fn accept(
        &mut self,
        t: &mut H,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), H::Error> {
        self.record(t.pos() - rows.len() as u32, rows, out, accepted)
    }

    /// A plain step on `last`: the rows that waited for the next chain —
    /// the refresh an accept recorded, after a prompt call with no `begin`
    /// the anchor (`last` at the prompt's end), or on a first step at
    /// position 0 its row beside a zero hidden row — walked now, since
    /// the step's row follows them and the store must hold every position
    /// below the next chain's; then `next`, the id taken after the step, at
    /// the position after it, with the hidden row the step wrote
    /// ([`Host::STEP_ARENA`]), as the next refresh, its rows left for
    /// the call that walks them — the width chooser's hook
    /// ([`Draft::before_plain`]) or a seat's step before the next plain
    /// step, the next chain for its head, a continuing prompt's join — so
    /// the store never holds a position past the target's, which a
    /// sequence state needs. Refused by name when the waiting rows read
    /// the step's own arena, which the step has overwritten.
    fn stepped(&mut self, t: &mut H, last: u32, next: u32) -> Result<(), H::Error> {
        let pos = t.pos();
        let mut on = On { t, head: self.head };
        self.policy
            .stepped(&mut on, pos, last, next)
            .map_err(Fail::lift)
    }

    /// The waiting rows walked before the chooser's held pass steps the
    /// target ([`MtpDraft::before_step`]: the step would overwrite the
    /// arena they read); nothing waits, and this does nothing, on every
    /// pass that runs a proposal.
    fn before_plain(&mut self, t: &mut H, last: u32) -> Result<(), H::Error> {
        self.before_step(t, last)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::fmt;
    use std::rc::Rc;

    use super::*;
    use crate::mock::{Mock, MockError};
    use crate::mtp::{Hidden, HostSink, Walk};
    use crate::{Out, Target, Verify, Want};

    /// The arena kinds of a host whose prompt runs through ubatches, its verify
    /// through the pass arena and its plain step through the step's.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ar {
        Step,
        Pass,
        Ubatch,
    }

    /// The head a host's chains run.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Head(u8);

    /// Where a walk's hidden rows come from, as the host saw them.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Rd {
        /// Rows from the caller: how many values.
        Host(usize),
        /// The arena `walk`'s rows from its row `first` on.
        Target(Ar, usize),
    }

    /// One call a host heard, in order.
    #[derive(Clone, Debug, PartialEq)]
    enum Ev {
        /// The prompt scope entered (`true`) or left (`false`).
        Scope(bool),
        Held,
        Walk {
            head: Option<Head>,
            mode: WalkMode,
            pos0: u32,
            tokens: Vec<u32>,
            hidden: Rd,
        },
        Chain {
            refresh: Vec<u32>,
            own: usize,
            head: Head,
            read_p: bool,
        },
        Prompt {
            ids: usize,
            end: CallEnd,
            sink: bool,
        },
        Retain(Ar, usize, u32),
    }

    fn walk(head: Option<Head>, mode: WalkMode, pos0: u32, tokens: &[u32], hidden: Rd) -> Ev {
        Ev::Walk {
            head,
            mode,
            pos0,
            tokens: tokens.to_vec(),
            hidden,
        }
    }

    const STORE: WalkMode = WalkMode::Store;
    const HEAD: Head = Head(7);
    /// The prompt every test feeds: nine ids at positions 0..9.
    const IDS: [u32; 9] = [10, 11, 12, 13, 14, 15, 16, 17, 18];
    /// Values of one hidden row the host's draft reads.
    const HIDDEN: usize = 2;

    /// What a host's calls fail with before the target wraps it.
    #[derive(Debug, PartialEq)]
    struct Cause(&'static str);

    #[derive(Debug)]
    enum RigError {
        Mock(MockError),
        Refused(String),
        Cause(&'static str),
    }

    impl fmt::Display for RigError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                RigError::Mock(e) => e.fmt(f),
                RigError::Refused(s) => f.write_str(s),
                RigError::Cause(s) => f.write_str(s),
            }
        }
    }

    impl std::error::Error for RigError {}

    impl From<MockError> for RigError {
        fn from(e: MockError) -> RigError {
            RigError::Mock(e)
        }
    }

    impl From<Refused> for RigError {
        fn from(e: Refused) -> RigError {
            RigError::Refused(e.0)
        }
    }

    impl From<Cause> for RigError {
        fn from(e: Cause) -> RigError {
            RigError::Cause(e.0)
        }
    }

    /// What the host and the executors it hands a prompt call's sink share:
    /// the log of what they heard, the positions the draft's store holds, and
    /// the walk that fails.
    #[derive(Default)]
    struct Shared {
        log: RefCell<Vec<Ev>>,
        store: Cell<usize>,
        fail_walks: Cell<bool>,
    }

    impl Shared {
        /// A walk as the host runs it: refused when it starts past the store
        /// or when told to fail, else the store holds the positions through
        /// its last row.
        fn walk(&self, head: Option<Head>, w: &Walk<'_, Ar>) -> Result<(), Cause> {
            if self.fail_walks.get() {
                return Err(Cause("the host refused the walk"));
            }
            if w.pos0 as usize > self.store.get() {
                return Err(Cause("a walk from past the draft's store"));
            }
            let hidden = match w.hidden {
                Hidden::Host(v) => Rd::Host(v.len()),
                Hidden::Target { walk, first } => Rd::Target(walk, first),
            };
            self.log
                .borrow_mut()
                .push(walk(head, w.mode, w.pos0, w.tokens, hidden));
            self.store.set(w.pos0 as usize + w.tokens.len());
            Ok(())
        }

        fn take(&self) -> Vec<Ev> {
            std::mem::take(&mut *self.log.borrow_mut())
        }
    }

    /// The walks a prompt call's sink runs on: an executor of one unit.
    struct Unit4(Rc<Shared>);

    impl Exec<Ar> for Unit4 {
        type Err = Cause;

        fn walk(&mut self, w: &Walk<'_, Ar>) -> Result<(), Cause> {
            self.0.walk(None, w)
        }

        fn held(&mut self) -> Result<usize, Cause> {
            Ok(self.0.store.get())
        }
    }

    /// A host with no card: the mock target (greedy, with a position and a
    /// context) beside a draft whose walks, chains and prompt units are
    /// recorded. `PROBS` is whether its chain's readback holds probabilities.
    struct Rig<const PROBS: bool> {
        target: Mock,
        shared: Rc<Shared>,
        /// A prompt call that ends at the argmax gives none.
        no_token: bool,
    }

    impl<const PROBS: bool> Rig<PROBS> {
        fn new() -> Self {
            Rig {
                target: Mock::new(),
                shared: Rc::new(Shared::default()),
                no_token: false,
            }
        }

        fn take(&self) -> Vec<Ev> {
            self.shared.take()
        }
    }

    impl<const PROBS: bool> Target for Rig<PROBS> {
        type Error = RigError;

        fn pos(&self) -> u32 {
            self.target.pos()
        }

        fn ctx(&self) -> u32 {
            self.target.ctx()
        }

        fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, RigError> {
            Ok(self.target.prompt(ids, want)?)
        }

        fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, RigError> {
            Ok(self.target.step(id, want)?)
        }

        fn keepable(&self, n: u32) -> u32 {
            self.target.keepable(n)
        }

        fn cut(&mut self, n: u32) -> Result<(), RigError> {
            Ok(self.target.cut(n)?)
        }

        fn reset(&mut self) -> Result<(), RigError> {
            Ok(self.target.reset()?)
        }
    }

    impl<const PROBS: bool> Verify for Rig<PROBS> {
        const MAX_ROWS: usize = 4;

        fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], RigError> {
            Ok(self.target.verify(rows)?)
        }

        fn commit(&mut self, accepted: usize) -> Result<(), RigError> {
            Ok(self.target.commit(accepted)?)
        }
    }

    impl<const PROBS: bool> Host for Rig<PROBS> {
        const WIDTH: usize = 3;
        const WALK_ROWS: usize = 4;
        const PROBS: bool = PROBS;

        type Arena = Ar;
        const STEP_ARENA: Ar = Ar::Step;
        const VERIFY_ARENA: Ar = Ar::Pass;
        type Path = ();
        type Head = Head;
        type Cause = Cause;

        fn held(&self) -> Result<usize, Cause> {
            self.shared.log.borrow_mut().push(Ev::Held);
            Ok(self.shared.store.get())
        }

        fn walk(&mut self, w: &Walk<'_, Ar>, head: Head) -> Result<(), Cause> {
            self.shared.walk(Some(head), w)
        }

        fn chain(
            &mut self,
            c: &Chain<Ar>,
            head: Head,
            out: &mut [u32],
            p: Option<&mut [f32]>,
        ) -> Result<usize, Cause> {
            let n = 1 + c.own;
            self.shared.log.borrow_mut().push(Ev::Chain {
                refresh: c.refresh.tokens.clone(),
                own: c.own,
                head,
                read_p: p.is_some(),
            });
            for (i, o) in out[..n].iter_mut().enumerate() {
                *o = 100 + i as u32;
            }
            if let Some(p) = p {
                if !PROBS {
                    return Err(Cause(
                        "a chain's probabilities: the readback holds ids alone",
                    ));
                }
                for (i, v) in p[..n].iter_mut().enumerate() {
                    *v = 0.5 - i as f32 / 10.0;
                }
            }
            Ok(n)
        }

        fn prompt_with(
            &mut self,
            ids: &[u32],
            _path: (),
            end: CallEnd,
            _head: Head,
            sink: Option<&mut HostSink<'_, Self>>,
        ) -> Result<Option<u32>, Cause> {
            self.shared.log.borrow_mut().push(Ev::Prompt {
                ids: ids.len(),
                end,
                sink: sink.is_some(),
            });
            let mut sink = sink;
            let mut argmax = 0;
            for run in ids.chunks(4) {
                let first = self.target.pos();
                argmax = self
                    .target
                    .prompt(run, Want::Argmax)
                    .map_err(|_| Cause("the target's prompt failed"))?
                    .argmax();
                if let Some(sink) = sink.as_deref_mut() {
                    sink(
                        &mut Unit4(self.shared.clone()),
                        Ar::Ubatch,
                        first,
                        run.len(),
                    )?;
                }
            }
            Ok((end.head() && !self.no_token).then_some(argmax))
        }

        fn retain_prompt_row(&mut self, walk: Ar, first: usize, pos: u32) -> Result<(), Cause> {
            self.shared
                .log
                .borrow_mut()
                .push(Ev::Retain(walk, first, pos));
            Ok(())
        }

        fn prompt_scope<R>(&mut self, call: impl FnOnce(&mut Self) -> R) -> R {
            self.shared.log.borrow_mut().push(Ev::Scope(true));
            let r = call(self);
            self.shared.log.borrow_mut().push(Ev::Scope(false));
            r
        }
    }

    fn draft<const PROBS: bool>() -> MtpDraft<Rig<PROBS>> {
        MtpDraft::new((), HEAD, WalkMode::Graph, HIDDEN)
    }

    /// The walks a prompt of [`IDS`] in units of four rows runs to fill the
    /// store: position 0's row beside the zero row, then every row after it
    /// but the last through the unit that holds its hidden row.
    fn fill() -> Vec<Ev> {
        let w = |pos0, tokens: &[u32]| walk(None, STORE, pos0, tokens, Rd::Target(Ar::Ubatch, 0));
        vec![
            walk(None, STORE, 0, &[10], Rd::Host(HIDDEN)),
            w(1, &[11, 12, 13, 14]),
            w(5, &[15, 16, 17, 18]),
        ]
    }

    /// A draft after the prompt of [`IDS`] and its `begin`, the host's log
    /// taken.
    fn begun<const PROBS: bool>() -> (MtpDraft<Rig<PROBS>>, Rig<PROBS>, u32) {
        let (mut d, mut t) = (draft::<PROBS>(), Rig::<PROBS>::new());
        let first = d.prompt(&mut t, &IDS).unwrap();
        d.begin(&t, &IDS, first).unwrap();
        t.take();
        (d, t, first)
    }

    #[test]
    fn a_prompt_call_walks_the_draft_over_each_unit() {
        let (mut d, mut t) = (draft::<false>(), Rig::<false>::new());
        let first = d.prompt(&mut t, &IDS).unwrap();
        let mut want = vec![
            Ev::Scope(true),
            Ev::Prompt {
                ids: 9,
                end: CallEnd::Argmax,
                sink: true,
            },
        ];
        want.splice(2..2, fill());
        want.push(Ev::Scope(false));
        assert_eq!(t.take(), want);
        let target_alone = Mock::new().prompt(&IDS, Want::Argmax).unwrap().argmax();
        assert_eq!((first, t.pos()), (target_alone, 9));
        d.begin(&t, &IDS, first).unwrap();
    }

    #[test]
    fn a_prompt_call_that_ends_unread_gives_no_token_and_still_fills_the_store() {
        let (mut d, mut t) = (draft::<false>(), Rig::<false>::new());
        assert_eq!(d.prompt_call(&mut t, &IDS, CallEnd::Unread).unwrap(), None);
        let log = t.take();
        assert!(log.contains(&Ev::Prompt {
            ids: 9,
            end: CallEnd::Unread,
            sink: true
        }));
        assert_eq!(
            log.iter().filter(|e| matches!(e, Ev::Walk { .. })).count(),
            3
        );
        t.no_token = true;
        let e = draft::<false>().prompt(&mut t, &IDS).unwrap_err();
        assert!(
            matches!(&e, RigError::Refused(s) if s.starts_with("MTP window:") && s.contains("head token")),
            "{e}"
        );
    }

    #[test]
    fn a_held_pass_walks_the_waiting_rows() {
        let (mut d, mut t, first) = begun::<false>();
        d.before_plain(&mut t, first).unwrap();
        assert_eq!(
            t.take(),
            [
                Ev::Held,
                walk(Some(HEAD), STORE, 9, &[first], Rd::Target(Ar::Ubatch, 0))
            ]
        );
        // Nothing waits any more: a second hook walks nothing.
        d.before_plain(&mut t, first).unwrap();
        assert_eq!(t.take(), []);
    }

    #[test]
    fn a_continuing_prompt_joins_the_draft_inside_the_prompt_scope() {
        let (mut d, mut t, first) = begun::<false>();
        let more = [first, 21, 22];
        d.prompt_call(&mut t, &more, CallEnd::Argmax).unwrap();
        assert_eq!(
            t.take(),
            [
                Ev::Scope(true),
                Ev::Held,
                walk(Some(HEAD), STORE, 9, &[first], Rd::Target(Ar::Ubatch, 0)),
                Ev::Prompt {
                    ids: 3,
                    end: CallEnd::Argmax,
                    sink: true
                },
                walk(None, STORE, 10, &[21, 22], Rd::Target(Ar::Ubatch, 0)),
                Ev::Scope(false),
            ]
        );
        let join = d.take_joined().unwrap();
        assert_eq!((join.start, join.caught_up, join.skipped), (9, 1, None));
    }

    #[test]
    fn a_drafted_window_is_kept_with_its_probabilities_and_what_the_target_took() {
        let (mut d, mut t, first) = begun::<true>();
        d.keep_windows();
        let (mut out, mut p) = ([0u32; 3], [0f32; 3]);
        let n = d.propose_p(&mut t, first, &mut out, &mut p).unwrap();
        assert_eq!((n, out, p), (3, [100, 101, 102], [0.5, 0.4, 0.3]));
        assert_eq!(
            t.take(),
            [Ev::Chain {
                refresh: vec![first],
                own: 2,
                head: HEAD,
                read_p: true
            }]
        );
        let rows = [first, 100, 101, 102];
        let taken = t.verify(rows).unwrap();
        d.accept(&mut t, &rows, &taken, 3).unwrap();
        t.commit(3).unwrap();
        assert_eq!(
            d.take_windows(),
            [WindowDraft {
                pos: 9,
                ids: vec![100, 101, 102],
                p: vec![0.5, 0.4, 0.3],
                width: 3,
                accepted: 2,
            }]
        );
        assert!(d.take_windows().is_empty());
        // The window kept is the one the accept saw: an accept with no
        // proposal behind it is refused by name.
        let e = d.accept(&mut t, &[first, 100], &[100, 101], 2).unwrap_err();
        assert!(
            matches!(&e, RigError::Refused(s) if s.contains("proposal the draft did not keep")),
            "{e}"
        );
    }

    #[test]
    fn a_proposal_reads_probabilities_only_where_the_readback_holds_them() {
        let (mut d, mut t, first) = begun::<false>();
        let (mut out, mut p) = ([0u32; 3], [0f32; 3]);
        assert_eq!(d.propose_p(&mut t, first, &mut out, &mut p).unwrap(), 3);
        assert_eq!(p, [1.0; 3]);
        assert!(matches!(
            t.take().as_slice(),
            [Ev::Chain { read_p: false, .. }]
        ));
        let (mut d, mut t, first) = begun::<true>();
        d.propose_p(&mut t, first, &mut out, &mut p).unwrap();
        assert_eq!(p, [0.5, 0.4, 0.3]);
        // The room the caller hands caps the chain's own walks.
        let (mut d, mut t, first) = begun::<true>();
        let mut short = [0u32; 2];
        assert_eq!(d.propose(&mut t, first, &mut short).unwrap(), 2);
        assert!(matches!(t.take().as_slice(), [Ev::Chain { own: 1, .. }]));
        let mut p2 = [0f32; 1];
        let (mut d, mut t, first) = begun::<true>();
        let e = d.propose_p(&mut t, first, &mut short, &mut p2).unwrap_err();
        assert!(
            matches!(&e, RigError::Refused(s) if s.contains("2 ids into 1 probabilities")),
            "{e}"
        );
        // A host whose readback holds ids alone refuses the kept windows' probabilities.
        let (mut d, mut t, first) = begun::<false>();
        d.keep_windows();
        let e = d.propose(&mut t, first, &mut out).unwrap_err();
        assert!(
            matches!(e, RigError::Cause(s) if s.contains("ids alone")),
            "{e}"
        );
    }

    #[test]
    fn a_failed_walk_and_a_refusal_both_end_the_call_as_the_hosts_error() {
        let (mut d, mut t, first) = begun::<false>();
        t.shared.fail_walks.set(true);
        let e = d.before_plain(&mut t, first).unwrap_err();
        assert!(
            matches!(e, RigError::Cause("the host refused the walk")),
            "{e}"
        );
        let (mut d, mut t, first) = begun::<false>();
        t.shared.fail_walks.set(true);
        let e = d.stepped(&mut t, first, 1).unwrap_err();
        assert!(matches!(e, RigError::Cause(_)), "{e}");

        // A refusal of the policy is the host's error through `From<Refused>`.
        let (mut d, mut t) = (draft::<false>(), Rig::<false>::new());
        let e = d.begin(&t, &IDS, 1).unwrap_err();
        assert!(
            matches!(&e, RigError::Refused(s) if s.contains("a prompt the draft never saw")),
            "{e}"
        );
        let first = d.prompt(&mut t, &IDS).unwrap();
        d.begin(&t, &IDS, first).unwrap();
        t.target.step(first, Want::Argmax).unwrap();
        d.stepped(&mut t, first, 4).unwrap();
        t.target.step(4, Want::Argmax).unwrap();
        let e = d.stepped(&mut t, 4, 2).unwrap_err();
        assert!(
            matches!(&e, RigError::Refused(s) if s.contains("the step's own arena")),
            "{e}"
        );
    }

    #[test]
    fn a_parked_draft_keeps_what_waits_and_an_unpark_drops_the_kept_proposal() {
        let (mut d, mut t, first) = begun::<true>();
        d.keep_windows();
        let parked = d.park();
        // A prompt call's anchor sits in a prompt arena, which a park does not keep.
        assert!(parked.next.is_none() && parked.skip.is_some());
        let mut copied = Vec::new();
        d.retain_prompt(&mut t).unwrap();
        copied.extend(t.take());
        assert_eq!(copied, [Ev::Retain(Ar::Ubatch, 0, 9)]);
        let mut out = [0u32; 3];
        d.propose(&mut t, first, &mut out).unwrap();
        assert!(d.keeps_windows());
        d.unpark(&parked);
        d.restart();
        assert!(d.take_windows().is_empty());
    }

    /// The walks a call's own executor ran, as the host's own sink would see
    /// them: the join's and each unit's, on the one executor.
    struct Own(Vec<Ev>);

    impl Exec<Ar> for Own {
        type Err = Cause;

        fn walk(&mut self, w: &Walk<'_, Ar>) -> Result<(), Cause> {
            let hidden = match w.hidden {
                Hidden::Host(v) => Rd::Host(v.len()),
                Hidden::Target { walk, first } => Rd::Target(walk, first),
            };
            self.0.push(walk(None, w.mode, w.pos0, w.tokens, hidden));
            Ok(())
        }

        fn held(&mut self) -> Result<usize, Cause> {
            Ok(9)
        }
    }

    #[test]
    fn a_call_on_an_executor_of_its_own_joins_and_walks_each_unit_on_it() {
        let mut d = draft::<false>();
        let mut x = Own(Vec::new());
        let ids = [10, 11, 12, 13, 14];
        let ran = d
            .prompt_joined(&mut x, 0, &ids, |x, sink| {
                let sink = sink.expect("a draft at position 0 joins");
                sink(x, Ar::Ubatch, 0, 3)?;
                sink(x, Ar::Ubatch, 3, 2)?;
                Ok(2)
            })
            .unwrap();
        assert_eq!(ran, 2);
        assert_eq!(
            x.0,
            [
                walk(None, STORE, 0, &[10], Rd::Host(HIDDEN)),
                walk(None, STORE, 1, &[11, 12, 13], Rd::Target(Ar::Ubatch, 0)),
                walk(None, STORE, 4, &[14], Rd::Target(Ar::Ubatch, 0)),
            ]
        );
        // A draft that cannot join is handed no sink: the call is the target's alone.
        let mut d = draft::<false>();
        let mut x = Own(Vec::new());
        let none = d
            .prompt_joined(&mut x, 5, &ids, |_, sink| Ok(sink.is_none()))
            .unwrap();
        assert!(none && x.0.is_empty());
    }
}
