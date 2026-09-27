//! Drafts: a proposal of the next ids, verified by one pass of the target.

use crate::{Advance, Committed, Verify, Want};

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
    /// Ids a proposal holds; the verify runs `WIDTH + 1` rows, the token at
    /// the target's position first.
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
    /// token 0.
    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error>;

    /// Write the [`Draft::WIDTH`] ids to follow `last`, the token at the
    /// target's position, into `out`; `false` is no proposal, and the pass is
    /// then one step.
    fn propose(&mut self, t: &T, last: u32, out: &mut [u32]) -> Result<bool, T::Error>;

    /// The verify of `rows` read back `out`, one argmax a row, and keeps its
    /// first `accepted` rows; runs before [`Verify::commit`] takes the rest
    /// back, so the target's taps still hold every row.
    fn accept(
        &mut self,
        t: &mut T,
        rows: &[u32],
        out: &[u32],
        accepted: usize,
    ) -> Result<(), T::Error>;

    /// A pass with no proposal stepped `last`; its argmax is `next`.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error>;
}

/// A draft's proposals, verified `M` rows at a time: `M` is the draft's
/// [`Draft::WIDTH`] + 1, which its pass checks at compile time.
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
/// have one entry a row.
pub(crate) fn accepted_rows(rows: &[u32], out: &[u32]) -> usize {
    1 + rows[1..]
        .iter()
        .zip(out)
        .take_while(|(d, o)| d == o)
        .count()
}

impl<T: Verify, D: Draft<T>, const M: usize> Advance<T> for Speculative<D, M> {
    fn prompt(&mut self, t: &mut T, ids: &[u32]) -> Result<u32, T::Error> {
        self.draft.prompt(t, ids)
    }

    fn begin(&mut self, t: &T, prompt: &[u32], first: u32) -> Result<(), T::Error> {
        self.draft.begin(t, prompt, first)
    }

    /// Propose, verify the proposal behind `last` in one pass, keep the rows
    /// the target agrees with, the draft hearing of them before the rest is
    /// taken back; with no proposal, one step.
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
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
        if !self.draft.propose(t, last, &mut rows[1..])? {
            let next = t.step(last, Want::Argmax)?.argmax();
            self.draft.stepped(t, last, next)?;
            out.push(next);
            return Ok(Committed {
                pos,
                kept: 1,
                rows: 1,
                proposed: false,
            });
        }
        let argmax = t.verify(rows)?;
        let accepted = accepted_rows(&rows, &argmax);
        self.draft.accept(t, &rows, &argmax, accepted)?;
        t.commit(accepted)?;
        out.extend_from_slice(&argmax[..accepted]);
        Ok(Committed {
            pos,
            kept: accepted,
            rows: M,
            proposed: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::accepted_rows;
    use crate::mock::{Call, Mock};
    use crate::{Advance, Lookup, Plain, Speculative, Stop, StopReason, generate};

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
        let r = generate(
            &mut t,
            &mut Plain,
            &[1, 2, 3],
            first,
            &stop,
            &mut crate::mock::Quiet,
        );
        assert!(r.is_err());
        assert_eq!(t.calls().last(), Some(&Call::Step(9)));
        assert_eq!(t.pos(), 9);
    }

    use crate::Target;

    fn run<A: Advance<Mock>>(
        a: &mut A,
        prompt: &[u32],
        stop: &Stop,
    ) -> (crate::GenOutcome, Vec<Call>) {
        let mut t = Mock::new();
        let first = a.prompt(&mut t, prompt).unwrap();
        let out = generate(&mut t, a, prompt, first, stop, &mut crate::mock::Quiet).unwrap();
        (out, t.calls().to_vec())
    }
}
