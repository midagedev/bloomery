//! The server's engine over a Metal session: [`TargetEngine`] implements
//! `serve::Engine` over any [`runtime::Target`] beside the file's
//! vocabulary, so a request's ids are the ones `generate_metal` picks for the
//! same prompt.
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
//! What it refuses, by name: a target whose head rows are not the
//! vocabulary's count, at construction, so a sampled row is never cut or
//! padded.

use std::fmt::Display;
use std::sync::Arc;
use std::time::Instant;

use runtime::{Out, Target, Want};
use serve::media::MediaFeed;
use serve::{Engine, EngineError, EngineProps, Tokenizer};

/// A target served with its vocabulary.
pub struct TargetEngine<T> {
    target: T,
    vocab: Arc<dyn Tokenizer>,
    /// What a crash report names before the position: the backend and the
    /// architecture.
    what: String,
    props: EngineProps,
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
        if head_rows != vocab.n_vocab() {
            return Err(format!(
                "{what}: the head writes {head_rows} logits a row, the vocabulary holds {} ids",
                vocab.n_vocab()
            ));
        }
        Ok(TargetEngine {
            target,
            vocab,
            what,
            props,
        })
    }
}

fn engine_error(e: impl Display) -> EngineError {
    EngineError(e.to_string())
}

impl<T: Target + Send> Engine for TargetEngine<T> {
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
        let from = self.target.pos();
        let t0 = Instant::now();
        self.target
            .prompt_marked(ids, &at, Want::Argmax)
            .map_err(engine_error)?;
        eprintln!(
            "{}: prompt call {from}..{}: {} ids, points at {at:?}, in {:.1} ms",
            self.what,
            self.target.pos(),
            ids.len(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let want = match logits_out {
            Some(_) => Want::Logits,
            None => Want::Argmax,
        };
        let out = self.target.step(last, want).map_err(engine_error)?;
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
        Ok(out.argmax())
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.target.reset().map_err(engine_error)
    }

    fn keepable(&self, n: usize) -> usize {
        let n = u32::try_from(n).unwrap_or(u32::MAX);
        self.target.keepable(n) as usize
    }

    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        let n = u32::try_from(n)
            .map_err(|_| EngineError(format!("{}: a cut to position {n}", self.what)))?;
        self.target.cut(n).map_err(engine_error)
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

    use runtime::{Out, Target, Want};
    use serve::{Engine, EngineProps, MockTokenizer, Tokenizer};

    use super::TargetEngine;

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
