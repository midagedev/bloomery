//! Drafts: a proposal of the next ids, verified by one pass of the target.
//!
//! A proposal holds 0 to [`Draft::WIDTH`] ids. None is one step of the
//! target, the plain pass's own call; `n` ids are one verify of `n + 1` rows,
//! dispatched by width ([`Window`]) so that a target's `verify::<R>` is
//! instantiated only for the widths its draft can ask.

use std::fmt;

use models::{BlockDraft, DraftSpec};

use crate::stores::PASS_ROWS;
use crate::{Advance, Committed, Target, Verify, Want};

/// Which hidden rows of the target a draft reads. The rows themselves — which
/// layers, how wide — are the draft file's and the target's to agree on at
/// load; this is only the kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapNeed {
    /// The token history alone.
    None,
    /// The final hidden row before the head, per position (an MTP draft).
    Final,
    /// The rows entering the layers the draft names, per position (a block
    /// draft's target layers).
    Layers,
}

/// A target whose calls leave hidden rows a draft reads ([`TapNeed`]).
pub trait Tapped: Verify {
    /// Floats one position's tapped row holds.
    fn tap_width(&self) -> usize;

    /// The tapped rows of the last call's first `rows` positions, one
    /// [`Tapped::tap_width`] row each: a verify's rows, or a step's one.
    fn taps(&mut self, rows: usize) -> Result<&[f32], Self::Error>;
}

/// A source of proposals for a target `T`.
pub trait Draft<T: Verify> {
    /// The most ids a proposal holds; a verify of `n` of them runs `n + 1`
    /// rows, the token at the target's position first.
    const WIDTH: usize;

    /// What the draft reads of the target besides the tokens.
    const TAPS: TapNeed;

    /// Holds when a proposal and the token before it fit one verify of `T`.
    /// [`Speculative`]'s pass reads it, so a draft wider than
    /// `T::MAX_ROWS − 1` fails to compile where it is driven.
    const FITS: () = assert!(
        Self::WIDTH >= 1 && Self::WIDTH < T::MAX_ROWS,
        "a draft's WIDTH + 1 rows must fit one verify of the target (Verify::MAX_ROWS)"
    );

    /// Feed the prompt `ids` through `t` from where it stands, with whatever
    /// the draft takes of it; the argmax after the last id. A draft that reads
    /// only tokens takes nothing from the call: the target's own prompt.
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        Ok(t.prompt(ids, Want::Argmax)?.argmax())
    }

    /// Generation begins after `prompt`, whose argmax `first` is generated
    /// token 0. A generation is its own: `prompt` then `begin` start the draft
    /// over, and after `begin` it holds what this prompt call and `first` gave
    /// it and nothing of the positions before the call — a target that
    /// continues from earlier turns is drafted from this turn alone, which
    /// changes how many rows a pass keeps and never a token.
    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error>;

    /// Write up to [`Draft::WIDTH`] ids to follow `last`, the token at the
    /// target's position, into the front of `out` (`WIDTH` long) and return
    /// how many: 0 is no proposal, and the pass is then one step; `n` is a
    /// verify of `n + 1` rows. A draft may run its own program on the
    /// target's buffers, and leaves the target at the position it found it:
    /// a draft that moves it, or proposes more than `WIDTH` ids, is broken,
    /// and the pass panics by name.
    fn propose(&mut self, t: &mut T, last: u32, out: &mut [u32]) -> Result<usize, T::Error>;

    /// The verify of `rows` keeps its first `accepted` rows; `out` holds one
    /// id a row, the id the pass took at each kept row ([`Pick`]: the
    /// argmax on a greedy pass, the draw on a sampled one) and the argmax
    /// past them. Runs before [`Verify::commit`] takes the rest back, so the
    /// target's taps still hold every row. `rows` is the proposal's `n + 1`,
    /// fewer than `WIDTH + 1` when the draft proposed fewer.
    fn accept(
        &mut self,
        t: &mut T,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error>;

    /// A pass with no proposal stepped `last`; the id it took after it is
    /// `next`.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error>;

    /// [`Draft::stepped`] of a pass whose proposal was held back (a closed
    /// [`crate::Gated`]): the draft still takes in the position — a window it
    /// keeps cannot be rebuilt — but may queue the work, as long as its next
    /// [`Draft::propose`] sees every position before it. The default is
    /// [`Draft::stepped`].
    fn held(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        self.stepped(t, last, next)
    }
}

/// The verify widths of a pass of at most `M` rows: 2 to `M`, the rows of a
/// proposal of 1 to `M − 1` ids. [`Widths`] is implemented for `M` of 2 to
/// [`PASS_ROWS`] and no other, so a wider window does not compile.
#[derive(Clone, Copy, Debug)]
pub struct Window<const M: usize>;

/// What a caller does with a verify of `R` rows, `R` one width of a
/// [`Window`]: run it, or capture it.
pub trait Width {
    type Error;

    /// Act on the width `R`.
    fn run<const R: usize>(&mut self) -> Result<(), Self::Error>;
}

/// A window's widths, each an instance of [`Width::run`] of its own.
#[diagnostic::on_unimplemented(
    message = "`{Self}`: a verify window keys 2 to `runtime::stores::PASS_ROWS` rows",
    label = "no pass of this many rows"
)]
pub trait Widths {
    /// The window's widest pass.
    const ROWS: usize;

    /// `w` of every width, 2 to [`Widths::ROWS`], ascending.
    fn each<W: Width>(w: &mut W) -> Result<(), W::Error>;

    /// `w` of the width `rows`; `None` when `rows` is not one of the window's.
    fn one<W: Width>(rows: usize, w: &mut W) -> Option<Result<(), W::Error>>;
}

/// `Widths` for each window, its widths listed in full: an arm names only
/// widths up to its own `M`, which a target of `MAX_ROWS` ≥ `M` compiles.
macro_rules! widths {
    ($($m:literal: $($r:literal)+;)+) => {$(
        const _: () = assert!($m <= PASS_ROWS, "a window wider than a pass");

        impl Widths for Window<$m> {
            const ROWS: usize = $m;

            fn each<W: Width>(w: &mut W) -> Result<(), W::Error> {
                $(w.run::<$r>()?;)+
                Ok(())
            }

            fn one<W: Width>(rows: usize, w: &mut W) -> Option<Result<(), W::Error>> {
                match rows {
                    $($r => Some(w.run::<$r>()),)+
                    _ => None,
                }
            }
        }
    )+};
}

widths! {
    2: 2;
    3: 2 3;
    4: 2 3 4;
    5: 2 3 4 5;
    6: 2 3 4 5 6;
    7: 2 3 4 5 6 7;
    8: 2 3 4 5 6 7 8;
}

/// The widest window is a pass's own width.
const _: () = assert!(<Window<{ PASS_ROWS }> as Widths>::ROWS == PASS_ROWS);

/// A draft's proposals, verified in passes of at most `M` rows: `M` is the
/// draft's [`Draft::WIDTH`] + 1, which its pass checks at compile time.
#[derive(Debug)]
pub struct Speculative<D, const M: usize> {
    draft: D,
}

impl<D, const M: usize> Speculative<D, M> {
    /// Verify `draft`'s proposals in passes of `M` rows.
    pub fn new(draft: D) -> Speculative<D, M> {
        Speculative { draft }
    }

    /// The draft.
    pub fn draft(&self) -> &D {
        &self.draft
    }

    /// See [`Speculative::draft`].
    pub fn draft_mut(&mut self) -> &mut D {
        &mut self.draft
    }
}

/// The rows of a greedy verify that are kept: row 0 always, and row `r + 1`
/// while row `r`'s argmax is the proposal's id at `r + 1`. `rows` and `out`
/// have one entry a row; `rows` holds at least row 0 (an empty one panics).
pub fn accepted_rows(rows: &[u32], out: &[u32]) -> usize {
    1 + rows[1..]
        .iter()
        .zip(out)
        .take_while(|(d, o)| d == o)
        .count()
}

/// A target that reads back the logits of its last call's rows.
pub trait RowLogits: Target {
    /// Row `r`'s logits (one a vocabulary id): a step's or a prompt call's
    /// row 0, or row `r` of a verify waiting for its commit. Refused by name
    /// for a row the call did not run, and for a row holding a NaN.
    fn row_logits(&mut self, r: usize) -> Result<&[f32], Self::Error>;
}

/// Which id a pass takes at each row it ran, asked row by row in position
/// order and never past the first row whose id is not the proposal's next.
pub trait Pick<T: Target> {
    /// The id taken at row `r` of `t`'s last call (a step's row 0, or row
    /// `r` of a verify waiting for its commit), whose argmax is `argmax`.
    fn pick(&mut self, t: &mut T, r: usize, argmax: u32) -> Result<u32, T::Error>;
}

/// The greedy pick: each row's argmax. It reads nothing back.
#[derive(Clone, Copy, Debug, Default)]
pub struct Argmax;

impl<T: Target> Pick<T> for Argmax {
    #[inline]
    fn pick(&mut self, _t: &mut T, _r: usize, argmax: u32) -> Result<u32, T::Error> {
        Ok(argmax)
    }
}

/// The sampled pick: each row's id drawn by the closure from that row's
/// logits ([`RowLogits::row_logits`]), one call a taken id.
pub struct Sample<F>(pub F);

impl<T: RowLogits, F: FnMut(&[f32]) -> u32> Pick<T> for Sample<F> {
    fn pick(&mut self, t: &mut T, r: usize, _argmax: u32) -> Result<u32, T::Error> {
        Ok((self.0)(t.row_logits(r)?))
    }
}

/// One verify of the proposal in `rows`, [`Width::run`] at the proposal's
/// width: each row's id taken by `pick` in position order, the rows through
/// the first whose id is not the proposal's next kept, the draft hearing of
/// them before the rest is taken back.
struct VerifyRows<'a, T, D, P> {
    t: &'a mut T,
    draft: &'a mut D,
    pick: &'a mut P,
    /// The token at the target's position, then the proposal; at least the
    /// width's rows.
    rows: &'a [u32],
    out: &'a mut Vec<u32>,
    kept: usize,
}

impl<T: Verify, D: Draft<T>, P: Pick<T>> Width for VerifyRows<'_, T, D, P> {
    type Error = T::Error;

    fn run<const R: usize>(&mut self) -> Result<(), T::Error> {
        let rows: [u32; R] = self.rows[..R]
            .try_into()
            .expect("a slice of R ids is an array of R");
        let argmax = self.t.verify(rows)?;
        let mut taken = argmax;
        let mut kept = R;
        for (r, (id, &a)) in taken.iter_mut().zip(&argmax).enumerate() {
            *id = self.pick.pick(self.t, r, a)?;
            if rows.get(r + 1).is_some_and(|&d| d != *id) {
                kept = r + 1;
                break;
            }
        }
        self.draft.accept(self.t, &rows, &taken, kept)?;
        self.t.commit(kept)?;
        self.out.extend_from_slice(&taken[..kept]);
        self.kept = kept;
        Ok(())
    }
}

impl<T: Verify, D: Draft<T>, const M: usize> Advance<T> for Speculative<D, M>
where
    Window<M>: Widths,
{
    const ROWS: usize = M;

    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        self.draft.prompt(t, ids)
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.draft.begin(t, prompt, first)
    }

    /// Propose, verify the `n` ids behind `last` in one pass of `n + 1`
    /// rows, keep the rows the target agrees with, the draft hearing of them
    /// before the rest is taken back; with no proposal, one step: the greedy
    /// pass, [`Speculative::pass_picking`] by [`Argmax`].
    ///
    /// # Panics
    ///
    /// As [`Speculative::pass_picking`].
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        self.pass_picking(t, last, &mut Argmax, out)
    }
}

impl<D, const M: usize> Speculative<D, M>
where
    Window<M>: Widths,
{
    /// One pass whose ids `pick` takes, by llama.cpp's
    /// `common_sampler_sample_and_accept_n`: propose `n` ids behind `last`
    /// and verify them in one pass of `n + 1` rows; then, row by row in
    /// position order, the id `pick` takes is appended to `out`, until the
    /// first that is not the proposal's next id, or the last row. Those rows
    /// are kept, the draft hearing of the taken ids before the rest is taken
    /// back. With no proposal, one step and its row's id. Each row is picked
    /// once, before the draft hears of it.
    ///
    /// # Panics
    ///
    /// When the draft proposes more than its [`Draft::WIDTH`] ids, or moves
    /// the target's position.
    pub fn pass_picking<T: Verify, P: Pick<T>>(
        &mut self,
        t: &mut T,
        last: u32,
        pick: &mut P,
        out: &mut Vec<u32>,
    ) -> Result<Committed, T::Error>
    where
        D: Draft<T>,
    {
        let () = D::FITS;
        const {
            assert!(
                M == D::WIDTH + 1,
                "Speculative<D, M> verifies D::WIDTH + 1 rows"
            );
        }
        let pos = t.pos();
        let mut rows = [0u32; M];
        rows[0] = last;
        let n = self.draft.propose(t, last, &mut rows[1..])?;
        let now = t.pos();
        assert!(
            now == pos,
            "a draft's proposal moved the target from position {pos} to {now}"
        );
        if n == 0 {
            let argmax = t.step(last, Want::Argmax)?.argmax();
            let next = pick.pick(t, 0, argmax)?;
            self.draft.stepped(t, last, next)?;
            out.push(next);
            return Ok(Committed {
                pos,
                kept: 1,
                rows: 1,
                proposed: false,
            });
        }
        let mut v = VerifyRows {
            t,
            draft: &mut self.draft,
            pick,
            rows: &rows,
            out,
            kept: 0,
        };
        let Some(ran) = Window::<M>::one(n + 1, &mut v) else {
            panic!(
                "a draft of width {} proposed {n} ids: a pass verifies at most {M} rows",
                D::WIDTH
            );
        };
        ran?;
        Ok(Committed {
            pos,
            kept: v.kept,
            rows: n + 1,
            proposed: true,
        })
    }
}

/// The program a draft's description is driven by, by its kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Program<'a> {
    /// A block draft on a card of its own, fed the target's layer taps.
    Block(&'a BlockDraft),
}

/// A draft kind whose program is not built: refused by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotBuilt {
    /// The kind, as the description names it.
    pub kind: &'static str,
}

impl fmt::Display for NotBuilt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the {} draft program is not built yet", self.kind)
    }
}

impl std::error::Error for NotBuilt {}

/// The program `spec` is driven by: a block draft's, and a next-token
/// (MTP) draft refused by name.
pub fn program(spec: &DraftSpec) -> Result<Program<'_>, NotBuilt> {
    match spec {
        DraftSpec::Block(b) => Ok(Program::Block(b)),
        DraftSpec::Mtp(_) => Err(NotBuilt { kind: "MTP" }),
    }
}

#[cfg(test)]
mod tests {
    use super::{NotBuilt, Program, Width, Widths, Window, accepted_rows, program};
    use crate::mock::{Call, Mock, MockError, Quiet};
    use crate::{
        Advance, Committed, Draft, Lookup, Out, PassSink, Plain, RowLogits, Sample, Speculative,
        Stop, StopReason, TapNeed, Verify, Want, generate,
    };

    /// Row 0 always; then while the argmax of row r is row r + 1's id.
    #[test]
    fn accepted_rows_arithmetic() {
        assert_eq!(accepted_rows(&[5, 7], &[7, 3]), 2);
        assert_eq!(accepted_rows(&[5, 7], &[8, 3]), 1);
        assert_eq!(accepted_rows(&[5, 7, 9, 2], &[7, 9, 1, 4]), 3);
        assert_eq!(accepted_rows(&[5, 7, 9, 2], &[7, 9, 2, 4]), 4);
        assert_eq!(accepted_rows(&[5, 7, 9, 2], &[6, 9, 2, 4]), 1);
    }

    /// Lossless: the lookup's passes emit the plain run's tokens, and every
    /// kept token is the target's own argmax at its position.
    #[test]
    fn lookup_emits_the_plain_tokens() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(60, 1000).unwrap();
        let plain = run(&mut Plain, &prompt, &stop);
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        let drafted = run(&mut spec, &prompt, &stop);
        assert_eq!(plain.0.stop, StopReason::Length);
        assert_eq!(drafted.0.tokens[..60], plain.0.tokens[..]);
        assert!(drafted.0.tokens.len() <= 61);
        // Both branches ran: some proposals kept, some taken back.
        let verifies = drafted
            .1
            .iter()
            .filter(|c| matches!(c, Call::Verify(..)))
            .count();
        let commits: Vec<usize> = drafted
            .1
            .iter()
            .filter_map(|c| match c {
                Call::Commit(k) => Some(*k),
                _ => None,
            })
            .collect();
        assert_eq!(commits.len(), verifies);
        assert!(commits.contains(&1) && commits.contains(&2), "{commits:?}");
    }

    /// Only kept tokens enter the lookup's context: after the run it holds
    /// the prompt and exactly the tokens the passes kept.
    #[test]
    fn lookup_context_is_the_kept_tokens() {
        let prompt = [4, 4, 1, 2, 4];
        let stop = Stop::new(40, 1000).unwrap();
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        let (out, _) = run(&mut spec, &prompt, &stop);
        let want: Vec<u32> = prompt.iter().copied().chain(out.tokens).collect();
        assert_eq!(spec.draft().context(), &want[..]);
    }

    /// A rejected row is taken back: the target stands one position past
    /// each kept token, and the commit follows every verify at once.
    #[test]
    fn reject_takes_the_row_back() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(50, 1000).unwrap();
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        let (out, calls) = run(&mut spec, &prompt, &stop);
        let mut pos = prompt.len();
        for c in &calls[1..] {
            match *c {
                Call::Step(p) => {
                    assert_eq!(p as usize, pos);
                    pos += 1;
                }
                Call::Verify(p, m) => {
                    assert_eq!(p as usize, pos);
                    pos += m;
                }
                Call::Commit(k) => pos = pos - 2 + k,
                Call::Prompt(..) => panic!("a second prompt call"),
            }
        }
        // Every kept token after token 0 advanced one position.
        assert_eq!(pos, prompt.len() + out.tokens.len() - 1);
        for w in calls.windows(2) {
            if matches!(w[0], Call::Verify(..)) {
                assert!(matches!(w[1], Call::Commit(_)), "{w:?}");
            }
        }
    }

    /// A failed prompt call is returned once, not fed again: the advance
    /// makes one call and passes the error on.
    #[test]
    fn failed_prompt_is_not_refed() {
        let mut t = Mock::new().fail_prompt();
        assert!(Plain.prompt(&mut t, &[1, 2, 3]).is_err());
        assert_eq!(t.calls(), &[Call::Prompt(0, 3)]);
        let mut t = Mock::new().fail_prompt();
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        assert!(spec.prompt(&mut t, &[1, 2, 3]).is_err());
        assert_eq!(t.calls(), &[Call::Prompt(0, 3)]);
    }

    /// A pass that fails ends the loop: no call after it.
    #[test]
    fn failed_pass_ends_the_loop() {
        let mut t = Mock::new().fail_at(9);
        let first = Plain.prompt(&mut t, &[1, 2, 3]).unwrap();
        let stop = Stop::new(50, 1000).unwrap();
        let r = generate(&mut t, &mut Plain, &[1, 2, 3], first, &stop, &mut Quiet);
        assert!(r.is_err());
        assert_eq!(t.calls().last(), Some(&Call::Step(9)));
        assert_eq!(t.pos(), 9);
    }

    /// An end-of-generation id kept at a pass's first row ends the
    /// generation there: the tokens stop at it, as the plain run's do.
    #[test]
    fn eog_inside_a_pass_ends_it() {
        let prompt = [1, 2, 3, 1, 2];
        let (plain, _) = run(&mut Plain, &prompt, &Stop::new(60, 1000).unwrap());
        // Every oracle pass keeps two rows, tokens 2k + 1 and 2k + 2: an id
        // first seen at an odd index is kept one row before its pass's end.
        let t = &plain.tokens;
        let (j, eog) = (1..t.len())
            .step_by(2)
            .map(|j| (j, t[j]))
            .find(|&(j, id)| !t[..j].contains(&id))
            .expect("an id first seen at an odd index");
        let stop = Stop::new(60, 1000).unwrap().with_eog(&[eog]);
        let (want, _) = run(&mut Plain, &prompt, &stop);
        assert_eq!((want.stop, want.tokens.len()), (StopReason::Eog, j + 1));
        let mut spec = Speculative::<Oracle, 2>::new(Oracle);
        let (got, calls) = run(&mut spec, &prompt, &stop);
        assert!(calls.contains(&Call::Commit(2)), "{calls:?}");
        assert_eq!(got.stop, StopReason::Eog);
        assert_eq!(got.tokens, want.tokens);
    }

    /// A pass whose rows would pass the caches' end is not run: the
    /// generation stops at the context, where the plain loop stops one
    /// position later.
    #[test]
    fn a_pass_past_the_context_is_not_run() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(1000, 20).unwrap();
        let mut t = Mock::new().with_ctx(20);
        let mut spec = Speculative::<Oracle, 2>::new(Oracle);
        let first = spec.prompt(&mut t, &prompt).unwrap();
        let out = generate(&mut t, &mut spec, &prompt, first, &stop, &mut Quiet)
            .expect("the loop stops before the target refuses");
        assert_eq!((out.stop, t.pos()), (StopReason::Ctx, 19));
        let mut t = Mock::new().with_ctx(20);
        let first = Plain.prompt(&mut t, &prompt).unwrap();
        let out = generate(&mut t, &mut Plain, &prompt, first, &stop, &mut Quiet).unwrap();
        assert_eq!((out.stop, t.pos()), (StopReason::Ctx, 20));
    }

    /// A new prompt starts the lookup over: after a reset target's second
    /// generation its context is that generation's prompt and kept tokens.
    #[test]
    fn a_new_prompt_starts_the_lookup_over() {
        let stop = Stop::new(30, 1000).unwrap();
        let mut t = Mock::new();
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        for prompt in [[4, 4, 1, 2, 4], [3, 0, 3, 1, 1]] {
            t.reset().unwrap();
            let first = spec.prompt(&mut t, &prompt).unwrap();
            let out = generate(&mut t, &mut spec, &prompt, first, &stop, &mut Quiet).unwrap();
            let want: Vec<u32> = prompt.iter().copied().chain(out.tokens).collect();
            assert_eq!(spec.draft().context(), &want[..]);
        }
    }

    /// A draft that proposes the mock's own next token: every pass verifies
    /// two rows and keeps both.
    struct Oracle;

    impl Draft<Mock> for Oracle {
        const WIDTH: usize = 1;
        const TAPS: TapNeed = TapNeed::None;

        fn begin(&mut self, _t: &Mock, _prompt: &[u32], _first: u32) -> Result<(), MockError> {
            Ok(())
        }

        fn propose(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, MockError> {
            out[0] = t.next_after(last);
            Ok(1)
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            _rows: &[u32],
            _out: &[u32],
            _accepted: usize,
        ) -> Result<(), MockError> {
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Mock, _last: u32, _next: u32) -> Result<(), MockError> {
            Ok(())
        }
    }

    /// A draft of width 3 whose proposals hold, pass by pass, the counts of
    /// `plan` in turn: the target's own next token first, then ids the
    /// target may or may not agree with.
    struct Scripted {
        plan: Vec<usize>,
        next: usize,
        stepped: usize,
    }

    impl Scripted {
        fn new(plan: &[usize]) -> Scripted {
            Scripted {
                plan: plan.to_vec(),
                next: 0,
                stepped: 0,
            }
        }
    }

    impl Draft<Mock> for Scripted {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::None;

        fn begin(&mut self, _t: &Mock, _prompt: &[u32], _first: u32) -> Result<(), MockError> {
            Ok(())
        }

        fn propose(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, MockError> {
            let n = self.plan[self.next % self.plan.len()];
            self.next += 1;
            let first = t.next_after(last);
            for (i, o) in out.iter_mut().enumerate().take(n) {
                *o = (first + u32::try_from(i).unwrap()) % 5;
            }
            Ok(n)
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            _rows: &[u32],
            _out: &[u32],
            _accepted: usize,
        ) -> Result<(), MockError> {
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Mock, _last: u32, _next: u32) -> Result<(), MockError> {
            self.stepped += 1;
            Ok(())
        }
    }

    /// A draft of width 3 whose proposals are the target's own greedy ids
    /// after `last`, counts by `plan` in turn, that keeps every id it hears:
    /// each accept's kept ids and each step's next, and how many windows it
    /// heard kept whole and cut short.
    struct Heard {
        plan: Vec<usize>,
        next: usize,
        heard: Vec<u32>,
        whole: usize,
        cut: usize,
    }

    impl Heard {
        fn new(plan: &[usize]) -> Heard {
            Heard {
                plan: plan.to_vec(),
                next: 0,
                heard: Vec::new(),
                whole: 0,
                cut: 0,
            }
        }
    }

    impl Draft<Mock> for Heard {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::None;

        fn begin(&mut self, _t: &Mock, _prompt: &[u32], _first: u32) -> Result<(), MockError> {
            Ok(())
        }

        fn propose(
            &mut self,
            t: &mut Mock,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, MockError> {
            let n = self.plan[self.next % self.plan.len()];
            self.next += 1;
            out[..n].copy_from_slice(&t.greedy_after(last, n));
            Ok(n)
        }

        fn accept(
            &mut self,
            _t: &mut Mock,
            rows: &[u32],
            out: &[u32],
            accepted: usize,
        ) -> Result<(), MockError> {
            self.heard.extend_from_slice(&out[..accepted]);
            if accepted == rows.len() {
                self.whole += 1;
            } else {
                self.cut += 1;
            }
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Mock, _last: u32, next: u32) -> Result<(), MockError> {
            self.heard.push(next);
            Ok(())
        }
    }

    /// One seeded draw from `row` at temperature 0.8, a repeat of the
    /// history's last id held back by 0.5, so a draw reads the ids before it:
    /// xorshift, one step of the generator a call.
    fn draw(rng: &mut u64, row: &[f32], history: &[u32]) -> u32 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        let u = (*rng >> 40) as f32 / (1u64 << 24) as f32;
        let last = history.last().copied();
        let w: Vec<f32> = (0u32..)
            .zip(row)
            .map(|(v, &l)| ((l - if Some(v) == last { 0.5 } else { 0.0 }) / 0.8).exp())
            .collect();
        let total: f32 = w.iter().sum();
        let mut acc = 0.0;
        for (v, &x) in (0u32..).zip(&w) {
            acc += x / total;
            if u < acc {
                return v;
            }
        }
        u32::try_from(w.len() - 1).unwrap()
    }

    /// The generator's state for `seed`: never the fixed point 0.
    fn rng_of(seed: u64) -> u64 {
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
    }

    /// The prompt's row's draw, then plain steps, each step's row drawn,
    /// until `n` ids.
    fn plain_sampled(prompt: &[u32], seed: u64, n: usize) -> Vec<u32> {
        let mut t = Mock::new();
        let mut rng = rng_of(seed);
        let mut ids = Vec::new();
        t.prompt(prompt, Want::Argmax).unwrap();
        while ids.len() < n {
            if let Some(&last) = ids.last() {
                t.step(last, Want::Argmax).unwrap();
            }
            let id = draw(&mut rng, t.row_logits(0).unwrap(), &ids);
            ids.push(id);
        }
        ids
    }

    /// The prompt's row's draw, then sampled passes of `spec` until `n` ids
    /// at least: the ids, the draws made, and the target.
    fn drafted_sampled<D: Draft<Mock>, const M: usize>(
        spec: &mut Speculative<D, M>,
        prompt: &[u32],
        seed: u64,
        n: usize,
    ) -> (Vec<u32>, usize, Mock)
    where
        Window<M>: Widths,
    {
        let mut t = Mock::new();
        let mut rng = rng_of(seed);
        let mut ids = Vec::new();
        spec.prompt(&mut t, prompt).unwrap();
        let first = draw(&mut rng, t.row_logits(0).unwrap(), &ids);
        ids.push(first);
        spec.begin(&t, prompt, first).unwrap();
        let mut draws = 1;
        while ids.len() < n {
            let (from, last) = (ids.len(), ids[ids.len() - 1]);
            let mut out = Vec::new();
            let mut pick = Sample(|row: &[f32]| {
                draws += 1;
                let id = draw(&mut rng, row, &ids);
                ids.push(id);
                id
            });
            spec.pass_picking(&mut t, last, &mut pick, &mut out)
                .unwrap();
            assert_eq!(out, ids[from..], "a pass's ids are the ones it took");
        }
        (ids, draws, t)
    }

    /// Identity: over many seeds, sampled passes take the ids plain sampled
    /// steps take, one draw a taken id, with windows both kept whole and
    /// cut short at a draw that is not the proposal's id.
    #[test]
    fn a_sampled_pass_takes_the_plain_steps_ids() {
        let prompt = [1, 2, 3, 1, 2];
        let (mut whole, mut cut, mut off_greedy) = (0, 0, 0);
        let greedy = run(&mut Plain, &prompt, &Stop::new(48, 1000).unwrap())
            .0
            .tokens;
        for seed in 0..64 {
            let want = plain_sampled(&prompt, seed, 48);
            let mut spec = Speculative::<Heard, 4>::new(Heard::new(&[3, 1, 0, 2]));
            let (got, draws, _) = drafted_sampled(&mut spec, &prompt, seed, 48);
            assert_eq!(got[..48], want[..], "seed {seed}");
            assert_eq!(draws, got.len(), "seed {seed}: one draw a taken id");
            whole += spec.draft().whole;
            cut += spec.draft().cut;
            off_greedy += usize::from(got[..48] != greedy[..]);
        }
        assert!(whole > 0 && cut > 0, "whole {whole}, cut {cut}");
        assert!(off_greedy > 0, "no seed drew off the greedy ids");
    }

    /// The draft hears the taken ids: every accept's kept ids and every
    /// step's next, in order, are the ids the passes took after the first —
    /// a cut window's last kept id the draw, not its row's argmax.
    #[test]
    fn a_draft_hears_the_taken_ids() {
        let prompt = [1, 2, 3, 1, 2];
        for seed in 0..16 {
            let mut spec = Speculative::<Heard, 4>::new(Heard::new(&[3, 1, 0, 2]));
            let (got, _, _) = drafted_sampled(&mut spec, &prompt, seed, 40);
            assert!(spec.draft().cut > 0, "seed {seed}: no window was cut");
            assert_eq!(spec.draft().heard, got[1..], "seed {seed}");
        }
    }

    /// The greedy pick is the pass by each row's argmax and reads no row:
    /// [`Advance::pass`] gives the ids, the target's calls and what the draft
    /// hears that a sampled pick taking each row's largest logit gives, and
    /// reads nothing back while that one reads every row it takes.
    #[test]
    fn the_greedy_pick_reads_no_row() {
        let prompt = [1, 2, 3, 1, 2];
        let drive = |sampled: bool| {
            let mut t = Mock::new();
            let mut spec = Speculative::<Heard, 4>::new(Heard::new(&[3, 1, 0, 2]));
            let mut ids = vec![spec.prompt(&mut t, &prompt).unwrap()];
            spec.begin(&t, &prompt, ids[0]).unwrap();
            let mut largest = Sample(|row: &[f32]| {
                (0u32..)
                    .zip(row)
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |b, (v, &l)| if l > b.1 { (v, l) } else { b },
                    )
                    .0
            });
            while ids.len() < 40 {
                let last = ids[ids.len() - 1];
                if sampled {
                    spec.pass_picking(&mut t, last, &mut largest, &mut ids)
                } else {
                    Advance::pass(&mut spec, &mut t, last, &mut ids)
                }
                .unwrap();
            }
            let reads = t.logit_reads();
            (ids, t.calls().to_vec(), spec.draft().heard.clone(), reads)
        };
        let (ids, calls, heard, reads) = drive(false);
        let (s_ids, s_calls, s_heard, s_reads) = drive(true);
        assert_eq!((&ids, &calls, &heard), (&s_ids, &s_calls, &s_heard));
        assert_eq!(reads, 0, "the greedy pass read {reads} rows back");
        assert!(
            s_reads >= ids.len() - 1,
            "{s_reads} reads for {} ids",
            ids.len()
        );
    }

    /// A sink that keeps every pass's [`Committed`].
    #[derive(Default)]
    struct Kept(Vec<Committed>);

    impl PassSink<Mock> for Kept {
        type Error = MockError;

        fn begin(&mut self, _t: &Mock) -> Result<(), MockError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _t: &Mock,
            c: &Committed,
            _tokens: &[u32],
            _wall: std::time::Duration,
        ) -> Result<(), MockError> {
            self.0.push(*c);
            Ok(())
        }
    }

    /// The rows each pass ran, read off the target's calls: a step's 1, a
    /// verify's `M`.
    fn pass_rows(calls: &[Call]) -> Vec<usize> {
        calls
            .iter()
            .filter_map(|c| match *c {
                Call::Step(_) => Some(1),
                Call::Verify(_, m) => Some(m),
                Call::Prompt(..) | Call::Commit(_) => None,
            })
            .collect()
    }

    /// A proposal of n ids is one verify of n + 1 rows, whatever the
    /// window's widest: n = 3, 1, 0, 2 in turn run 4, 2, 1 and 3 rows, each
    /// pass's `Committed` says so, and the tokens are the plain run's.
    #[test]
    fn a_proposal_of_n_verifies_n_plus_one_rows() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(40, 1000).unwrap();
        let (plain, _) = run(&mut Plain, &prompt, &stop);
        let plan = [3, 1, 0, 2];
        let mut spec = Speculative::<Scripted, 4>::new(Scripted::new(&plan));
        let mut t = Mock::new();
        let first = spec.prompt(&mut t, &prompt).unwrap();
        let mut sink = Kept::default();
        let out = generate(&mut t, &mut spec, &prompt, first, &stop, &mut sink).unwrap();
        assert_eq!(out.tokens[..40], plain.tokens[..]);
        let want: Vec<usize> = (0..sink.0.len()).map(|i| plan[i % 4] + 1).collect();
        assert_eq!(pass_rows(t.calls()), want);
        let rows: Vec<usize> = sink.0.iter().map(|c| c.rows).collect();
        assert_eq!(rows, want);
        assert!(sink.0.iter().all(|c| c.proposed == (c.rows > 1)));
        assert!(sink.0.iter().all(|c| (1..=c.rows).contains(&c.kept)));
    }

    /// No proposal is the plain pass itself: the same calls in the same
    /// order and the same tokens as [`Plain`], every token heard by
    /// [`Draft::stepped`], and no verify of one row.
    #[test]
    fn no_proposal_is_the_plain_pass() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(30, 1000).unwrap();
        let (plain, plain_calls) = run(&mut Plain, &prompt, &stop);
        let mut spec = Speculative::<Scripted, 4>::new(Scripted::new(&[0]));
        let (got, calls) = run(&mut spec, &prompt, &stop);
        assert_eq!(calls, plain_calls);
        assert_eq!(got, plain);
        assert_eq!(spec.draft().stepped, plain.tokens.len() - 1);
    }

    /// A draft that proposes past its width is broken, and the pass says so
    /// by name instead of verifying some other number of rows.
    #[test]
    #[should_panic(expected = "a draft of width 3 proposed 4 ids")]
    fn a_proposal_past_the_width_panics() {
        let prompt = [1, 2, 3, 1, 2];
        let mut spec = Speculative::<Scripted, 4>::new(Scripted::new(&[4]));
        run(&mut spec, &prompt, &Stop::new(10, 1000).unwrap());
    }

    /// A draft that runs the target's step while proposing leaves it a
    /// position on: the pass says so by name instead of verifying from there.
    #[test]
    #[should_panic(expected = "a draft's proposal moved the target from position 5 to 6")]
    fn a_proposal_that_moves_the_target_panics() {
        struct Mover;
        impl Draft<Mock> for Mover {
            const WIDTH: usize = 1;
            const TAPS: TapNeed = TapNeed::None;
            fn begin(&mut self, _t: &Mock, _p: &[u32], _f: u32) -> Result<(), MockError> {
                Ok(())
            }
            fn propose(
                &mut self,
                t: &mut Mock,
                last: u32,
                out: &mut [u32],
            ) -> Result<usize, MockError> {
                out[0] = t.step(last, Want::Argmax)?.argmax();
                Ok(1)
            }
            fn accept(
                &mut self,
                _t: &mut Mock,
                _rows: &[u32],
                _out: &[u32],
                _accepted: usize,
            ) -> Result<(), MockError> {
                Ok(())
            }
            fn stepped(&mut self, _t: &mut Mock, _l: u32, _n: u32) -> Result<(), MockError> {
                Ok(())
            }
        }
        let mut spec = Speculative::<Mover, 2>::new(Mover);
        run(&mut spec, &[1, 2, 3, 1, 2], &Stop::new(10, 1000).unwrap());
    }

    /// The widths a window visits: every one from 2 to its widest for a
    /// capture, the one asked for a pass, none outside it.
    #[test]
    fn a_window_visits_its_widths() {
        struct Seen(Vec<usize>);
        impl Width for Seen {
            type Error = ();
            fn run<const R: usize>(&mut self) -> Result<(), ()> {
                self.0.push(R);
                Ok(())
            }
        }
        let mut s = Seen(Vec::new());
        Window::<4>::each(&mut s).unwrap();
        assert_eq!(s.0, [2, 3, 4]);
        let mut s = Seen(Vec::new());
        Window::<2>::each(&mut s).unwrap();
        assert_eq!(s.0, [2]);
        let mut s = Seen(Vec::new());
        assert_eq!(Window::<4>::one(3, &mut s), Some(Ok(())));
        assert_eq!(s.0, [3]);
        assert!(Window::<4>::one(1, &mut s).is_none());
        assert!(Window::<4>::one(5, &mut s).is_none());
        assert_eq!(s.0, [3]);
    }

    /// A target whose passes hold two rows at most, refusing a wider one at
    /// compile time as the card's bodies do: a draft of width 1 drives it,
    /// which compiles only when the pass instantiates no verify wider than
    /// the draft's own.
    struct Pair(Mock);

    impl Target for Pair {
        type Error = MockError;

        fn pos(&self) -> u32 {
            self.0.pos()
        }

        fn ctx(&self) -> u32 {
            self.0.ctx()
        }

        fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, MockError> {
            self.0.prompt(ids, want)
        }

        fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, MockError> {
            self.0.step(id, want)
        }

        fn keepable(&self, n: u32) -> u32 {
            self.0.keepable(n)
        }

        fn cut(&mut self, n: u32) -> Result<(), MockError> {
            self.0.cut(n)
        }

        fn reset(&mut self) -> Result<(), MockError> {
            self.0.reset()
        }
    }

    impl Verify for Pair {
        const MAX_ROWS: usize = 2;

        fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], MockError> {
            const { assert!(M <= 2, "a pair target verifies two rows at most") };
            self.0.verify(rows)
        }

        fn commit(&mut self, accepted: usize) -> Result<(), MockError> {
            self.0.commit(accepted)
        }
    }

    impl PassSink<Pair> for Quiet {
        type Error = MockError;

        fn begin(&mut self, _t: &Pair) -> Result<(), MockError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _t: &Pair,
            _c: &Committed,
            _tokens: &[u32],
            _wall: std::time::Duration,
        ) -> Result<(), MockError> {
            Ok(())
        }
    }

    /// A pair target runs a lookup: the pass instantiates the widths of its
    /// draft's window alone, so the build of this test is the proof.
    #[test]
    fn a_pair_target_compiles_only_its_width() {
        let prompt = [1, 2, 3, 1, 2];
        let stop = Stop::new(20, 1000).unwrap();
        let (plain, _) = run(&mut Plain, &prompt, &stop);
        let mut t = Pair(Mock::new());
        let mut spec = Speculative::<Lookup, 2>::new(Lookup::new());
        let first = spec.prompt(&mut t, &prompt).unwrap();
        let out = generate(&mut t, &mut spec, &prompt, first, &stop, &mut Quiet).unwrap();
        assert_eq!(out.tokens[..20], plain.tokens[..]);
    }

    /// A block draft is driven by its program; an MTP draft is refused by
    /// name, not taken for a block draft.
    #[test]
    fn a_draft_program_by_kind() {
        use models::{
            Act, BlockDraft, DraftSpec, Ffn, Gqa, HcKind, HcSpec, HeadRows, LayerSpec, Mixer,
            MtpDraft, MtpHeadNorm, MtpInput, MtpSource, Residual, Rope, RopeMode,
        };
        let block = BlockDraft {
            hidden: 64,
            vocab: 100,
            rms_eps: 1e-6,
            hc: HcSpec {
                streams: 4,
                kind: HcKind::Gated { rank: 8 },
            },
            layers: Vec::new(),
            width: 2,
            target_layers: vec![1],
            mask_token: 99,
            markov_rank: 4,
        };
        let spec = DraftSpec::Block(block.clone());
        assert_eq!(program(&spec), Ok(Program::Block(&block)));
        let layer = LayerSpec {
            mixer: Mixer::Gqa(Gqa {
                heads: 4,
                kv_heads: 2,
                head_dim: 16,
                rope: Rope {
                    mode: RopeMode::Neox,
                    dims: 16,
                    base: 10_000.0,
                    yarn: None,
                },
                qk_norm: true,
                out_gate: true,
                select: None,
            }),
            ffn: Ffn::Dense {
                ff: 128,
                act: Act::SwiGlu { limit: None },
            },
            residual: Residual::Plain,
            extras: Vec::new(),
        };
        let mtp = DraftSpec::Mtp(Box::new(MtpDraft {
            source: MtpSource::InFile { layer: 2 },
            layer,
            index: 2,
            hidden: 64,
            vocab: 100,
            rms_eps: 1e-6,
            hc: None,
            input: MtpInput::HeadRow,
            head_norm: MtpHeadNorm::Rms,
            head_rows: HeadRows::Full,
        }));
        let e = program(&mtp).unwrap_err();
        assert_eq!(e, NotBuilt { kind: "MTP" });
        assert_eq!(e.to_string(), "the MTP draft program is not built yet");
    }

    use crate::Target;

    fn run<A: Advance<Mock>>(
        a: &mut A,
        prompt: &[u32],
        stop: &Stop,
    ) -> (crate::GenOutcome, Vec<Call>) {
        let mut t = Mock::new();
        let first = a.prompt(&mut t, prompt).unwrap();
        let out = generate(&mut t, a, prompt, first, stop, &mut Quiet).unwrap();
        (out, t.calls().to_vec())
    }
}
