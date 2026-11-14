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
//! reset starts it over, and a cut to a position the draft rejoins keeps it
//! ([`DraftWindow::rejoins`]: the MTP draft where the checkpoint carries the
//! target's hidden row, the lookup anywhere, its context following the
//! target by positions). A cut to a position it cannot rejoin and a step
//! that reads the logits row turn the MTP draft off until the next reset,
//! every call then the target's own.
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
    Advance, Choosing, Committed, Draft, Lookup, Out, Plain, Speculative, Target, Verify, Want,
    WidthError, Widths, Window,
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

    /// Before the target is cut to `n`, a position below the one it stands
    /// at: whether the draft keeps drafting after the cut. Asked before the
    /// cut moves the target, which a host may need to see as it stands.
    fn rejoins(&self, t: &T, n: u32) -> bool;

    /// The target was cut back to `at`, a position below the one it stood
    /// at: the draft rejoins there when [`DraftWindow::rejoins`] held, and is
    /// off until the next reset otherwise.
    fn cut(&mut self, at: u32, rejoins: bool);

    /// The draft proposes nothing, and every call is the target's own, until
    /// the next reset: a step read the logits row, or the target moved where
    /// the draft holds no rows to rejoin at.
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

    fn rejoins(&self, _t: &T, _n: u32) -> bool {
        true
    }

    fn cut(&mut self, _at: u32, _rejoins: bool) {}

    fn skip(&mut self, _why: &'static str) {}

    fn joined(&mut self) -> Option<String> {
        None
    }
}

/// Why a cut leaves the MTP draft off: the draft rejoins a cut only where its
/// store holds the positions below it and the checkpoint there carries the
/// target's hidden row of the position before it.
const CUT_SKIP: &str = "a cut to a position the draft cannot rejoin (the checkpoint there carries no hidden row, or the draft's store stops short of it)";

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

    /// The host kept the draft's side of the cut ([`Host::rejoins`]).
    fn rejoins(&self, t: &H, n: u32) -> bool {
        t.rejoins(n)
    }

    /// The next call walks the row the host left in the step arena with its
    /// first id ([`MtpDraft::rejoin`]).
    fn cut(&mut self, at: u32, rejoins: bool) {
        if rejoins {
            let chooser = self.draft_mut();
            chooser.forget_pass();
            chooser.draft_mut().rejoin(at);
        } else {
            self.skip(CUT_SKIP);
        }
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

/// The lookup window of a [`Verify`] target behind the width chooser: the
/// context's n-grams propose the one id a pass verifies, `M` rows at most
/// (`Lookup::WIDTH + 1`). It needs no weights and no draft side, so no call
/// ever turns it off: a context follows the target by the positions the
/// calls stand at ([`Lookup::stand`]), and a cut back, a reset's fresh
/// prompt and a step that reads the logits row (the id the request draws
/// takes the argmax's place) need nothing told.
impl<T, const M: usize> DraftWindow<T> for Speculative<Choosing<Lookup>, M>
where
    T: Verify,
    T::Error: From<WidthError>,
    Window<M>: Widths,
{
    const ROWS: usize = M;

    /// The call's ids enter the context from the call's position, the
    /// context first made what it holds below that position; the call's own
    /// pick is dropped, the step that follows feeds the id the request holds.
    fn prompt(&mut self, t: &mut T, ids: &[u32], marks: &[u32]) -> Result<(), T::Error> {
        let pos = t.pos() as usize;
        self.draft_mut().prompt_by(|lookup| {
            t.prompt_marked(ids, marks, Want::Argmax)?;
            if let Some((&first, rest)) = ids.split_first() {
                lookup.stand(pos, first);
                for &id in rest {
                    lookup.push(id);
                }
            }
            Ok(())
        })
    }

    fn before_step(&mut self, _t: &mut T, _last: u32) -> Result<(), T::Error> {
        self.draft_mut().forget_pass();
        Ok(())
    }

    /// The step ran `last` at the position before the target's: the context
    /// stands there and takes `next`.
    fn stepped(&mut self, t: &mut T, last: u32, next: u32) -> Result<(), T::Error> {
        let lookup = self.draft_mut().draft_mut();
        lookup.stand((t.pos() as usize).saturating_sub(1), last);
        lookup.push(next);
        Ok(())
    }

    /// The context stands at the target's position with `last`, then the
    /// pass lands at its end, so the time between two of the server's calls
    /// is no part of its wall.
    fn pass(&mut self, t: &mut T, last: u32, out: &mut Vec<u32>) -> Result<Committed, T::Error> {
        self.draft_mut().draft_mut().stand(t.pos() as usize, last);
        let c = Advance::pass(self, t, last, out)?;
        self.draft_mut().finish_pass(t.pos())?;
        Ok(c)
    }

    fn reset(&mut self) {
        let chooser = self.draft_mut();
        chooser.forget_pass();
        chooser.draft_mut().reset();
    }

    /// A context follows the target by position: a cut back is always
    /// rejoined.
    fn rejoins(&self, _t: &T, _n: u32) -> bool {
        true
    }

    fn cut(&mut self, _at: u32, _rejoins: bool) {
        self.draft_mut().forget_pass();
    }

    /// The lookup is never off: a request that reads the logits row is
    /// drafted again from its next greedy call.
    fn skip(&mut self, _why: &'static str) {}

    fn joined(&mut self) -> Option<String> {
        None
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

    /// The target the engine serves.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// The draft window beside the target.
    pub fn window(&self) -> &W {
        &self.window
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
        let rejoins = self.window.rejoins(&self.target, n);
        self.target.cut(n).map_err(engine_error)?;
        // A cut to the position moves nothing, and a cut to 0 is the reset.
        match self.target.pos() {
            0 => self.window.reset(),
            at if at != from => self.window.cut(at, rejoins),
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
    use runtime::width::{Chosen, Mode};
    use runtime::{Committed, Lookup, Out, Speculative, Target, Verify, Want, WidthError};

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
        /// Whether the window rejoins a cut, as a host that kept the draft's
        /// side of it says; asked while the target stands above the cut.
        rejoin: bool,
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

        fn rejoins(&self, t: &Rows, n: u32) -> bool {
            self.rejoin && t.pos > n
        }

        fn cut(&mut self, at: u32, rejoins: bool) {
            self.calls.push(format!("cut {at} {rejoins}"));
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
            rejoin: true,
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
    /// cut to 0 is a reset, and a cut to a position between tells the window
    /// where the target stands now and whether the draft rejoins there, asked
    /// before the target moves.
    ///
    /// Mutant: a cut to the position that reaches the window, a cut to 0 that
    /// does not start it over, the window asked after the cut (the target
    /// then stands at the cut, and the host's own state is gone), or the
    /// answer dropped.
    #[test]
    fn a_reset_restarts_the_draft_and_a_cut_tells_whether_it_rejoins() {
        let mut e = drafted(committed(1, 1, false));
        e.prefill(&[1; 10]).expect("a prompt");
        e.window.calls.clear();
        e.cut(10).expect("a cut to the position");
        assert_eq!(e.window.calls, Vec::<String>::new());
        e.cut(4).expect("a cut inside");
        assert_eq!(e.window.calls, ["cut 4 true"]);
        e.window.calls.clear();
        e.window.rejoin = false;
        e.prefill(&[1; 6]).expect("a prompt");
        e.window.calls.clear();
        e.cut(7)
            .expect("a cut inside, where the draft cannot rejoin");
        assert_eq!(e.window.calls, ["cut 7 false"]);
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

    /// A target whose argmax after token `x` is `(x + 1) % 5`: the cycle a
    /// lookup learns after one lap.
    struct Cycle {
        pos: u32,
        row: Vec<f32>,
        rows: usize,
    }

    #[derive(Debug)]
    struct CycleError(String);

    impl std::fmt::Display for CycleError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for CycleError {}

    impl From<WidthError> for CycleError {
        fn from(e: WidthError) -> CycleError {
            CycleError(e.to_string())
        }
    }

    fn after(x: u32) -> u32 {
        (x + 1) % 5
    }

    impl Target for Cycle {
        type Error = CycleError;

        fn pos(&self) -> u32 {
            self.pos
        }

        fn ctx(&self) -> u32 {
            64
        }

        fn prompt(&mut self, ids: &[u32], _want: Want) -> Result<Out<'_>, CycleError> {
            self.pos += ids.len() as u32;
            Ok(Out::Argmax(after(ids[ids.len() - 1])))
        }

        fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, CycleError> {
            self.pos += 1;
            Ok(match want {
                Want::Argmax => Out::Argmax(after(id)),
                Want::Logits => Out::Logits {
                    argmax: after(id),
                    row: &self.row,
                },
            })
        }

        fn keepable(&self, n: u32) -> u32 {
            n.min(self.pos)
        }

        fn cut(&mut self, n: u32) -> Result<(), CycleError> {
            self.pos = n;
            Ok(())
        }

        fn reset(&mut self) -> Result<(), CycleError> {
            self.pos = 0;
            Ok(())
        }
    }

    impl Verify for Cycle {
        const MAX_ROWS: usize = 4;

        fn verify<const M: usize>(&mut self, rows: [u32; M]) -> Result<[u32; M], CycleError> {
            self.pos += M as u32;
            self.rows = M;
            Ok(rows.map(after))
        }

        fn commit(&mut self, accepted: usize) -> Result<(), CycleError> {
            self.pos -= (self.rows - accepted) as u32;
            Ok(())
        }
    }

    /// The lookup window's context follows the target by position through a
    /// prompt call, a greedy step, a kept pass, a cut back, a step that reads
    /// the logits row (the id the request draws takes the argmax's place) and
    /// a reset, and the draft is never turned off: a request cut to a point
    /// proposes as the fresh one did, and a greedy pass after a sampled
    /// request still proposes.
    ///
    /// Mutant: a context that is not stood at the call's position (a prompt
    /// after a cut keeps the cut's tail), a cut or a row step that empties it,
    /// the argmax kept where the request drew another id, a skip that turns
    /// the draft off, or a reset that keeps it.
    #[test]
    fn a_lookup_window_follows_the_target_and_is_never_off() {
        let n = MockTokenizer.n_vocab();
        let window = Speculative::<_, 2>::new(
            Chosen::<Cycle>::choosing(Lookup::new(), Mode::Fixed).expect("the lookup's rule"),
        );
        let target = Cycle {
            pos: 0,
            row: vec![0.0; n],
            rows: 0,
        };
        let mut e = TargetEngine::drafted(
            target,
            window,
            n,
            Arc::new(MockTokenizer),
            "cycle".to_owned(),
            EngineProps::default(),
        )
        .expect("a head of the vocabulary's rows");
        let context = |e: &TargetEngine<Cycle, Speculative<runtime::Choosing<Lookup>, 2>>| {
            e.window.draft().draft().context().to_vec()
        };
        let history = [0, 1, 2, 3, 4, 0, 1, 2];
        let mut out = Vec::new();

        // The server's request: the prompt less its last id, then that id as a step.
        e.prefill(&history).expect("a prompt call");
        assert_eq!(context(&e), history);
        assert_eq!(e.next(3, None).expect("the last id's step"), 4);
        let d = e.advance(4, &mut out).expect("a pass");
        assert_eq!(
            (d.proposed, d.accepted, out.as_slice()),
            (1, 1, &[0, 1][..])
        );
        assert_eq!(context(&e), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]);

        // A cut back to a point, the call after it from there: the context is the history.
        e.cut(4).expect("a cut to a point");
        e.prefill(&history[4..]).expect("the call after the cut");
        assert_eq!(context(&e), history);
        assert_eq!(e.next(3, None).expect("the last id's step"), 4);
        out.clear();
        let d = e.advance(4, &mut out).expect("a pass after the cut");
        assert_eq!(
            (d.proposed, d.accepted, out.as_slice()),
            (1, 1, &[0, 1][..]),
            "a request cut to a point drafts as the fresh one did"
        );

        // A request that reads the logits row, then draws another id than the argmax.
        e.cut(8).expect("a cut to the point");
        let mut row = vec![0.0; n];
        assert_eq!(e.next(3, Some(&mut row)).expect("a step with a row"), 4);
        assert_eq!(context(&e), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
        assert_eq!(e.next(2, None).expect("the drawn id's step"), 3);
        assert_eq!(context(&e), [0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 3]);
        out.clear();
        let d = e.advance(3, &mut out).expect("a pass after a sampled step");
        assert_eq!(d.proposed, 1, "the lookup is not turned off by a row step");

        // A reset empties the context.
        e.reset().expect("a reset");
        assert_eq!(context(&e), Vec::<u32>::new());
    }
}
