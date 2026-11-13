//! The server's engine over a generation target: [`TargetEngine`] implements
//! `serve::Engine` over any [`runtime::Target`] beside the file's
//! vocabulary, so a request's ids are the ones the target's own loop picks for
//! the same prompt. Host code, every OS: the Metal session and the host
//! engine are served through it.
//!
//! What it maps:
//!
//! - `prefill` is the target's prompt call, its pick dropped, and one line
//!   on stderr naming its positions, its ids, the marks it kept and its
//!   wall; `next` is one step, [`Want::Argmax`] for a greedy request (the
//!   session chains the pick into the token ahead) and [`Want::Logits`] for
//!   one that samples, the row copied into the server's buffer;
//! - `prefill_marked` is that call with the server's marks handed to the
//!   target ([`Target::prompt_marked`]), and `prefill_splits` cuts nowhere:
//!   the target keeps each mark as a point inside the call, so the start of
//!   each chat message is one without a call of its own. A cut is not
//!   bit-neutral here — a part of at most a step's window steps where the
//!   one call passes — so the server asks for none;
//! - `keepable`, `cut`, `reset` and `ctx_max` are the target's own rules: a
//!   prefix the target cannot keep is a reset, and why is the target's
//!   [`runtime::seqstate::Why`] word.
//!
//! A drafted engine ([`TargetEngine::drafted`]) is the same engine over a
//! [`DraftWindow`]: the prompt call, the steps and the passes go through the
//! window, and `advance` is one pass of it, [`Engine::advance_rows`] rows at
//! most. The window's draft proposes and the target verifies, so a kept id is
//! the target's own argmax and a request's ids are the plain engine's. The
//! draft joins a sequence only where it holds the rows below the position: a
//! reset starts it over, a cut to a position it cannot rejoin and a step
//! that reads the logits row turn it off until the next reset, every call
//! then the target's own.
//!
//! What it refuses, by name: a target whose head rows are not the
//! vocabulary's count, at construction, so a sampled row is never cut or
//! padded.

use std::fmt::Display;
use std::sync::Arc;
use std::time::Instant;

use crate::media::MediaFeed;
use crate::{Drafted, Engine, EngineError, EngineProps, Tokenizer};
use runtime::mtp::{Host, MtpDraft, Parked, Refused};
use runtime::prompt::CallEnd;
use runtime::{
    Advance, Choosing, Committed, Draft, Out, Plain, Speculative, Target, Want, WidthError, Widths,
    Window,
};

/// The draft a [`TargetEngine`] runs beside its target: what the engine's
/// calls do to it and what a pass is. [`Undrafted`] is no draft, one step a
/// pass.
pub trait DraftWindow<T: Target> {
    /// The most rows one pass runs.
    const ROWS: usize;

    /// The prompt call of `ids` from the target's position, a point at each
    /// of `marks` ([`Target::prompt_marked`]'s contract), its pick dropped.
    ///
    /// # Errors
    ///
    /// The target's, or the draft's.
    fn prompt(&mut self, t: &mut T, ids: &[u32], marks: &[u32]) -> Result<(), T::Error>;

    /// Before a plain step of `last`: the rows the draft left waiting, which
    /// the step would overwrite, are walked.
    ///
    /// # Errors
    ///
    /// The target's, or the draft's.
    fn before_step(&mut self, t: &mut T, last: u32) -> Result<(), T::Error>;

    /// After a plain step of `last` that took `next`.
    ///
    /// # Errors
    ///
    /// The target's, or the draft's.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error>;

    /// One greedy pass from `last`, the tokens it keeps appended to `out`.
    ///
    /// # Errors
    ///
    /// The target's, or the draft's.
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error>;

    /// The target was reset: the draft starts over and drafts again.
    fn reset(&mut self);

    /// The draft proposes nothing, and every call is the target's own, until
    /// the next reset: the target moved where the draft holds no rows to
    /// rejoin at, or a step read the logits row.
    fn skip(&mut self, why: &'static str);

    /// How the last prompt call joined the draft to the sequence, once.
    fn joined(&mut self) -> Option<String>;
}

/// No draft: the target's own calls, one step a pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct Undrafted;

impl<T: Target> DraftWindow<T> for Undrafted {
    const ROWS: usize = 1;

    fn prompt(&mut self, t: &mut T, ids: &[u32], marks: &[u32]) -> Result<(), T::Error> {
        t.prompt_marked(ids, marks, Want::Argmax).map(|_| ())
    }

    fn before_step(&mut self, _t: &mut T, _last: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn stepped(&mut self, _t: &mut T, _last: u32, _next: u32) -> Result<(), T::Error> {
        Ok(())
    }

    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        Plain.pass(t, last, out)
    }

    fn reset(&mut self) {}

    fn skip(&mut self, _why: &'static str) {}

    fn joined(&mut self) -> Option<String> {
        None
    }
}

/// Why a cut leaves the MTP draft off: the draft's store holds the positions
/// below the cut, and the target's hidden row before it is gone.
const CUT_SKIP: &str =
    "a cut to a position the draft cannot rejoin (it keeps no hidden row at a checkpoint)";

/// Why a step that reads the logits row leaves the MTP draft off.
const ROW_SKIP: &str = "a step that reads the logits row (a request that samples or asks for probabilities) is not drafted";

/// The MTP window of a [`Host`] behind the width chooser: the draft layer
/// proposes the ids a pass verifies, `M` rows at most.
impl<H, const M: usize> DraftWindow<H> for Speculative<Choosing<MtpDraft<H>>, M>
where
    H: Host,
    H::Error: From<Refused> + From<WidthError>,
    Window<M>: Widths,
{
    const ROWS: usize = M;

    /// The call walks the draft over each unit as it completes
    /// ([`MtpDraft::prompt_call_marked`]), behind the chooser's new
    /// generation.
    fn prompt(&mut self, t: &mut H, ids: &[u32], marks: &[u32]) -> Result<(), H::Error> {
        let pick = self
            .draft_mut()
            .prompt_by(|d| d.prompt_call_marked(t, ids, marks, CallEnd::Argmax))?;
        pick.map(|_| ()).ok_or_else(|| {
            Refused("MTP window: a prompt call that ends at the argmax gave no head token".into())
                .into()
        })
    }

    fn before_step(&mut self, t: &mut H, last: u32) -> Result<(), H::Error> {
        let chooser = self.draft_mut();
        chooser.forget_pass();
        chooser.draft_mut().before_step(t, last)
    }

    fn stepped(&mut self, t: &mut H, last: u32, next: u32) -> Result<(), H::Error> {
        Draft::stepped(self.draft_mut(), t, last, next)
    }

    /// The pass lands at its end, so the time between two of the server's
    /// calls is no part of its wall.
    fn pass(&mut self, t: &mut H, last: u32, out: &mut Vec<u32>) -> Result<Committed, H::Error> {
        let c = Advance::pass(self, t, last, out)?;
        self.draft_mut().finish_pass(t.pos())?;
        Ok(c)
    }

    fn reset(&mut self) {
        let chooser = self.draft_mut();
        chooser.forget_pass();
        chooser.draft_mut().restart();
    }

    fn skip(&mut self, why: &'static str) {
        let chooser = self.draft_mut();
        chooser.forget_pass();
        chooser.draft_mut().unpark(&Parked {
            next: None,
            last_unit: None,
            skip: Some(why),
        });
    }

    fn joined(&mut self) -> Option<String> {
        self.draft_mut()
            .draft_mut()
            .take_joined()
            .map(|j| match j.skipped {
                Some(why) => format!("draft skipped from {}: {why}", j.start),
                None => format!(
                    "draft joined at {} (caught up {} rows)",
                    j.start, j.caught_up
                ),
            })
    }
}

/// What the passes of one request drafted, for the line the next request
/// opens with.
#[derive(Clone, Copy, Debug, Default)]
struct Passes {
    passes: u64,
    proposed: u64,
    kept: u64,
}

/// A target served with its vocabulary, and its draft window when it has one.
pub struct TargetEngine<T, W = Undrafted> {
    target: T,
    window: W,
    vocab: Arc<dyn Tokenizer>,
    /// What a crash report names before the position: the backend and the
    /// architecture.
    what: String,
    props: EngineProps,
    passes: Passes,
}

impl<T: Target> TargetEngine<T> {
    /// `target` served with `vocab`; `head_rows` is the target's logits row
    /// length, refused when it is not the vocabulary's count.
    pub fn new(
        target: T,
        head_rows: usize,
        vocab: Arc<dyn Tokenizer>,
        what: String,
        props: EngineProps,
    ) -> Result<TargetEngine<T>, String> {
        TargetEngine::drafted(target, Undrafted, head_rows, vocab, what, props)
    }
}

impl<T: Target, W: DraftWindow<T>> TargetEngine<T, W> {
    /// [`TargetEngine::new`] with `window` drafting beside the target.
    pub fn drafted(
        target: T,
        window: W,
        head_rows: usize,
        vocab: Arc<dyn Tokenizer>,
        what: String,
        props: EngineProps,
    ) -> Result<TargetEngine<T, W>, String> {
        if head_rows != vocab.n_vocab() {
            return Err(format!(
                "{what}: the head writes {head_rows} logits a row, the vocabulary holds {} ids",
                vocab.n_vocab()
            ));
        }
        Ok(TargetEngine {
            target,
            window,
            vocab,
            what,
            props,
            passes: Passes::default(),
        })
    }

    /// The line a drafted request's passes leave, printed once as the next
    /// request opens.
    fn say_passes(&mut self) {
        let p = std::mem::take(&mut self.passes);
        if p.passes > 0 {
            eprintln!(
                "{}: draft: {} passes, {} ids proposed, {} kept",
                self.what, p.passes, p.proposed, p.kept
            );
        }
    }
}

fn engine_error(e: impl Display) -> EngineError {
    EngineError(e.to_string())
}

impl<T: Target + Send, W: DraftWindow<T> + Send> Engine for TargetEngine<T, W> {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.vocab)
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.prefill_marked(ids, &[], &[])
    }

    /// The prompt call with a point at each mark ([`Target::prompt_marked`]).
    /// A mark that is no `u32` position is the server's error, by name; the
    /// target refuses one out of order or outside the call. The default
    /// refuses a call that carries images: this engine takes none.
    fn prefill_marked(
        &mut self,
        ids: &[u32],
        media: &[MediaFeed],
        marks: &[usize],
    ) -> Result<(), EngineError> {
        if !media.is_empty() {
            return self.prefill_media(ids, media);
        }
        if ids.is_empty() {
            return Ok(());
        }
        let at = marks
            .iter()
            .map(|&u| u32::try_from(u))
            .collect::<Result<Vec<u32>, _>>()
            .map_err(|_| {
                EngineError(format!("{}: a prompt call marked at {marks:?}", self.what))
            })?;
        self.say_passes();
        let from = self.target.pos();
        let t0 = Instant::now();
        self.window
            .prompt(&mut self.target, ids, &at)
            .map_err(engine_error)?;
        let joined = self
            .window
            .joined()
            .map_or_else(String::new, |j| format!(", {j}"));
        eprintln!(
            "{}: prompt call {from}..{}: {} ids, points at {at:?}, in {:.1} ms{joined}",
            self.what,
            self.target.pos(),
            ids.len(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    /// One step. A step that reads the logits row turns the draft off until
    /// the next reset; a greedy step is told to the draft, after the rows it
    /// left waiting are walked.
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let want = match logits_out {
            Some(_) => {
                self.window.skip(ROW_SKIP);
                Want::Logits
            }
            None => Want::Argmax,
        };
        self.window
            .before_step(&mut self.target, last)
            .map_err(engine_error)?;
        let out = self.target.step(last, want).map_err(engine_error)?;
        let argmax = out.argmax();
        if let (Some(dst), Out::Logits { row, .. }) = (logits_out, &out) {
            if dst.len() != row.len() {
                return Err(EngineError(format!(
                    "{}: a logits buffer of {} for a row of {}",
                    self.what,
                    dst.len(),
                    row.len()
                )));
            }
            dst.copy_from_slice(row);
        }
        self.window
            .stepped(&mut self.target, last, argmax)
            .map_err(engine_error)?;
        Ok(argmax)
    }

    /// One pass of the draft window: `last` and what the draft proposes,
    /// verified together.
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        let c = self
            .window
            .pass(&mut self.target, last, out)
            .map_err(engine_error)?;
        let drafted = if c.proposed {
            Drafted {
                proposed: c.rows - 1,
                accepted: c.kept - 1,
            }
        } else {
            Drafted::default()
        };
        self.passes.passes += 1;
        self.passes.proposed += drafted.proposed as u64;
        self.passes.kept += drafted.accepted as u64;
        Ok(drafted)
    }

    fn advance_rows(&self) -> usize {
        W::ROWS
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.say_passes();
        self.target.reset().map_err(engine_error)?;
        self.window.reset();
        Ok(())
    }

    fn keepable(&self, n: usize) -> usize {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        self.target.keepable(n) as usize
    }

    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        let n = u32::try_from(n)
            .map_err(|_| EngineError(format!("{}: a cut to position {n}", self.what)))?;
        let from = self.target.pos();
        self.target.cut(n).map_err(engine_error)?;
        // A cut to the position moves nothing, and a cut to 0 is the reset.
        match self.target.pos() {
            0 => self.window.reset(),
            at if at != from => self.window.skip(CUT_SKIP),
            _ => {}
        }
        Ok(())
    }

    fn keep_limit(&self, n: usize) -> Option<String> {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        let kept = self.target.kept(n);
        (kept.at < n).then(|| {
            format!(
                "{} keeps {} of {n} positions: {}",
                self.what,
                kept.at,
                kept.why.code()
            )
        })
    }

    fn ctx_max(&self) -> usize {
        self.target.ctx() as usize
    }

    fn describe(&self) -> String {
        format!("{} at position {}", self.what, self.target.pos())
    }

    fn props_engine(&self) -> EngineProps {
        self.props.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::Arc;

    use crate::{Engine, EngineProps, MockTokenizer, Tokenizer};
    use runtime::{Committed, Out, Target, Want};

    use super::{DraftWindow, TargetEngine};

    /// A target whose every step writes a row of `row` zeros and picks 0;
    /// a prompt of no ids is refused, as the session's is. `calls` holds
    /// each prompt's id count.
    struct Rows {
        row: Vec<f32>,
        pos: u32,
        calls: Vec<usize>,
    }

    impl Target for Rows {
        type Error = io::Error;

        fn pos(&self) -> u32 {
            self.pos
        }

        fn ctx(&self) -> u32 {
            64
        }

        fn prompt(&mut self, ids: &[u32], _want: Want) -> Result<Out<'_>, io::Error> {
            if ids.is_empty() {
                return Err(io::Error::other("a prompt of no ids"));
            }
            self.pos += ids.len() as u32;
            self.calls.push(ids.len());
            Ok(Out::Argmax(0))
        }

        fn step(&mut self, _id: u32, want: Want) -> Result<Out<'_>, io::Error> {
            self.pos += 1;
            Ok(match want {
                Want::Argmax => Out::Argmax(0),
                Want::Logits => Out::Logits {
                    argmax: 0,
                    row: &self.row,
                },
            })
        }

        fn keepable(&self, n: u32) -> u32 {
            if n >= self.pos { self.pos } else { 0 }
        }

        fn cut(&mut self, n: u32) -> Result<(), io::Error> {
            self.pos = n;
            Ok(())
        }

        fn reset(&mut self) -> Result<(), io::Error> {
            self.pos = 0;
            Ok(())
        }
    }

    /// A window that tells what the engine called, and runs the target as a
    /// pass of `commit` says: its rows, the rows it kept and whether it
    /// proposed.
    struct Log {
        calls: Vec<String>,
        commit: Committed,
    }

    impl DraftWindow<Rows> for Log {
        const ROWS: usize = 2;

        fn prompt(&mut self, t: &mut Rows, ids: &[u32], marks: &[u32]) -> Result<(), io::Error> {
            self.calls.push(format!("prompt {} {marks:?}", ids.len()));
            t.prompt(ids, Want::Argmax).map(|_| ())
        }

        fn before_step(&mut self, _t: &mut Rows, last: u32) -> Result<(), io::Error> {
            self.calls.push(format!("before {last}"));
            Ok(())
        }

        fn stepped(&mut self, _t: &mut Rows, last: u32, next: u32) -> Result<(), io::Error> {
            self.calls.push(format!("stepped {last} {next}"));
            Ok(())
        }

        fn pass(
            &mut self,
            t: &mut Rows,
            last: u32,
            out: &mut Vec<u32>,
        ) -> Result<Committed, io::Error> {
            self.calls.push(format!("pass {last}"));
            t.step(last, Want::Argmax)?;
            out.resize(out.len() + self.commit.kept, 0);
            Ok(self.commit)
        }

        fn reset(&mut self) {
            self.calls.push("reset".to_owned());
        }

        fn skip(&mut self, why: &'static str) {
            self.calls.push(format!("skip {}", &why[..5]));
        }

        fn joined(&mut self) -> Option<String> {
            None
        }
    }

    fn drafted(commit: Committed) -> TargetEngine<Rows, Log> {
        let n = MockTokenizer.n_vocab();
        let target = Rows {
            row: vec![0.0; n],
            pos: 0,
            calls: Vec::new(),
        };
        let window = Log {
            calls: Vec::new(),
            commit,
        };
        TargetEngine::drafted(
            target,
            window,
            n,
            Arc::new(MockTokenizer),
            "rows".to_owned(),
            EngineProps::default(),
        )
        .expect("a head of the vocabulary's rows")
    }

    fn committed(kept: usize, rows: usize, proposed: bool) -> Committed {
        Committed {
            pos: 0,
            kept,
            rows,
            proposed,
        }
    }

    /// A drafted engine's prompt call is the window's, its marks handed on in
    /// one call; a step is walked before and told after, except that a step
    /// that reads the logits row turns the draft off first.
    ///
    /// Mutant: the prompt run on the target alone, a mark dropped, the step
    /// told to the draft before the walk, or the row's step left drafting.
    #[test]
    fn a_drafted_prompt_and_step_go_through_the_window() {
        let mut e = drafted(committed(1, 1, false));
        e.prefill_marked(&[1; 10], &[], &[2, 7])
            .expect("a call marked inside it");
        assert_eq!(e.window.calls, ["prompt 10 [2, 7]"]);
        assert_eq!(
            e.target.calls,
            [10],
            "the window ran the target's call once"
        );
        e.window.calls.clear();
        assert_eq!(e.next(5, None).expect("a greedy step"), 0);
        let mut buf = vec![1.0; MockTokenizer.n_vocab()];
        assert_eq!(e.next(6, Some(&mut buf)).expect("a step with a row"), 0);
        assert_eq!(
            e.window.calls,
            [
                "before 5",
                "stepped 5 0",
                "skip a ste",
                "before 6",
                "stepped 6 0"
            ]
        );
        assert_eq!(e.advance_rows(), 2);
        assert_eq!(
            engine(MockTokenizer.n_vocab(), MockTokenizer.n_vocab())
                .unwrap()
                .advance_rows(),
            1
        );
    }

    /// A pass tells the server what its draft proposed and the target kept:
    /// a window kept whole, a window that kept one row, and a pass that
    /// proposed nothing.
    ///
    /// Mutant: the counts off by one, or a pass with no proposal counted as
    /// one.
    #[test]
    fn a_pass_tells_what_the_draft_proposed_and_the_target_kept() {
        let mut out = Vec::new();
        for (c, want) in [
            (committed(2, 2, true), (1, 1, 2)),
            (committed(1, 2, true), (1, 0, 1)),
            (committed(1, 1, false), (0, 0, 1)),
        ] {
            let mut e = drafted(c);
            out.clear();
            let d = e.advance(3, &mut out).expect("a pass");
            assert_eq!((d.proposed, d.accepted, out.len()), want);
            assert_eq!(e.window.calls, ["pass 3"]);
        }
    }

    /// A reset starts the draft over, a cut to the position moves nothing, a
    /// cut to 0 is a reset and a cut to a position between leaves the draft
    /// off.
    ///
    /// Mutant: a cut that leaves the draft on, a cut to the position that
    /// turns it off, or a cut to 0 that does not start it over.
    #[test]
    fn a_reset_restarts_the_draft_and_a_cut_leaves_it_off() {
        let mut e = drafted(committed(1, 1, false));
        e.prefill(&[1; 10]).expect("a prompt");
        e.window.calls.clear();
        e.cut(10).expect("a cut to the position");
        assert_eq!(e.window.calls, Vec::<String>::new());
        e.cut(4).expect("a cut inside");
        assert_eq!(e.window.calls, ["skip a cut"]);
        e.window.calls.clear();
        e.cut(0).expect("a cut to 0");
        assert_eq!(e.window.calls, ["reset"]);
        e.window.calls.clear();
        e.reset().expect("a reset");
        assert_eq!(e.window.calls, ["reset"]);
    }

    fn engine(head: usize, row: usize) -> Result<TargetEngine<Rows>, String> {
        let target = Rows {
            row: vec![0.0; row],
            pos: 0,
            calls: Vec::new(),
        };
        TargetEngine::new(
            target,
            head,
            Arc::new(MockTokenizer),
            "rows".to_owned(),
            EngineProps::default(),
        )
    }

    #[test]
    fn refusals_by_name() {
        let n = MockTokenizer.n_vocab();
        let e = engine(n + 1, n + 1).err().unwrap_or_default();
        assert!(e.contains("logits a row"), "{e}");

        // A row of another length than the buffer is refused, never cut.
        let mut e = engine(n, n + 1).expect("a head of the vocabulary's rows");
        let mut buf = vec![0.0; n];
        let refused = e.next(1, Some(&mut buf)).expect_err("a row of n + 1");
        assert!(refused.0.contains("a logits buffer"), "{}", refused.0);

        // No ids is no call of the target, whose prompt refuses them.
        e.prefill(&[]).expect("an empty prefill is a no-op");
    }

    #[test]
    fn marks_reach_the_target_in_one_call() {
        let n = MockTokenizer.n_vocab();
        let mut e = engine(n, n).expect("a head of the vocabulary's rows");
        let marks = [2usize, 7];
        assert_eq!(
            Engine::prefill_splits(&e, 0, 10, &marks),
            Vec::<usize>::new()
        );

        // The engine hands the marks to the target's prompt; this target
        // keeps them by the default, a prompt a part.
        e.prefill_marked(&[1; 10], &[], &marks)
            .expect("a call marked inside it");
        assert_eq!(e.target.calls, [2, 5, 3]);
        e.prefill(&[1; 4]).expect("a call with no marks");
        assert_eq!(e.target.calls, [2, 5, 3, 4]);

        // A mark that is no position of the target is the server's error.
        let refused = e
            .prefill_marked(&[1; 3], &[], &[usize::MAX])
            .expect_err("a mark past u32");
        assert!(refused.0.contains("marked at"), "{}", refused.0);
    }
}
