//! The skeleton of a streamed restore: a state read from a stream and put on
//! a model as it is read, never held whole on the host. Every owner of a
//! state (the plane bodies, Qwen3.8, DeepSeek V4.1) restores through
//! [`restore_streamed`]; each gives only what its architecture forces, the
//! three parts the skeleton runs in order:
//!
//! 1. `head`: the header and every part of the stream that can be read and
//!    checked before the first array, against the model as it stands. A
//!    refusal or a failure of the stream here leaves the model untouched.
//! 2. the model is reset.
//! 3. `stream`: the arrays, read a bounded chunk at a time onto the card
//!    ([`Take::to_card`]), and any part of the stream that follows them.
//! 4. the stream's end ([`Take::end`]): a byte past the state is refused.
//! 5. `stand`: the model stands at the state's positions, which only a
//!    stream that ended where it must reaches.
//!
//! A failure from the reset on (a refusal, a failing stream or card, a short
//! stream, a byte past the state) leaves the model reset, never a part of the
//! state claimed whole. The failure that ended the restore is the one
//! returned; a reset that fails too is named beside it ([`Fail::Reset`]).

use super::bytes::{Fail, Take};
use crate::model::ChainBody;
use crate::{GpuError, GpuModel};

/// What a restore resets: the model a failed restore leaves empty.
pub trait Resettable {
    /// The model empty, as [`GpuModel::reset`] leaves it.
    fn reset(&mut self) -> Result<(), GpuError>;
}

impl<B: ChainBody> Resettable for GpuModel<B> {
    fn reset(&mut self) -> Result<(), GpuError> {
        GpuModel::reset(self)
    }
}

/// `r`'s state put on `m` ([the module doc](self)): `head` reads and checks
/// what precedes the arrays and gives the state's own description `H`;
/// `stream` takes the arrays from `r` onto the reset model, and whatever
/// follows them, which it gives as `S`; `stand` puts the model at the state's
/// positions from `H` and `S` and gives what the restore returns.
pub fn restore_streamed<M, H, S, R>(
    m: &mut M,
    r: &mut Take<'_>,
    head: impl FnOnce(&mut M, &mut Take<'_>) -> Result<H, Fail>,
    stream: impl FnOnce(&mut M, &mut Take<'_>, &H) -> Result<S, Fail>,
    stand: impl FnOnce(&mut M, H, S) -> Result<R, Fail>,
) -> Result<R, Fail>
where
    M: Resettable,
{
    let head = head(m, r)?;
    m.reset().map_err(Fail::card)?;
    let done = stream(m, r, &head).and_then(|side| {
        r.end()?;
        stand(m, head, side)
    });
    done.map_err(|e| e.with_reset(m.reset()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that records what the skeleton does to it, and whose `n`th
    /// reset (1-based) fails when asked to.
    #[derive(Default)]
    struct Mock {
        log: Vec<&'static str>,
        resets: u32,
        fails_reset: Option<u32>,
    }

    fn driver() -> GpuError {
        GpuError::Protocol {
            what: "a test reset",
            detail: "the card went".to_owned(),
        }
    }

    impl Resettable for Mock {
        fn reset(&mut self) -> Result<(), GpuError> {
            self.resets += 1;
            self.log.push("reset");
            if self.fails_reset == Some(self.resets) {
                return Err(driver());
            }
            Ok(())
        }
    }

    /// A stream of `bytes`, which the arrays part of a test restore takes
    /// `taken` of.
    fn run(
        m: &mut Mock,
        bytes: &[u8],
        taken: usize,
        head: Result<(), Fail>,
        stream: Result<(), Fail>,
        stand: Result<u32, Fail>,
    ) -> Result<u32, Fail> {
        let mut input = bytes;
        let mut r = Take::new(&mut input);
        restore_streamed(
            m,
            &mut r,
            |m, _| {
                m.log.push("head");
                head
            },
            |m, r, _| {
                m.log.push("stream");
                r.vals::<u8>(taken, "the arrays")?;
                stream
            },
            |m, (), ()| {
                m.log.push("stand");
                stand
            },
        )
    }

    fn short() -> Fail {
        crate::checkpoint::bytes::refused("a short stream: it ends inside the arrays")
    }

    fn pipe() -> Fail {
        Fail::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "the pipe closed",
        ))
    }

    /// A state that restores resets the model once, between the head and the
    /// arrays, and stands it last, after the stream's end.
    #[test]
    fn a_restore_resets_between_the_head_and_the_arrays_and_stands_last() {
        let mut m = Mock::default();
        let n = run(&mut m, b"abcd", 4, Ok(()), Ok(()), Ok(7)).expect("a whole state");
        assert_eq!(n, 7);
        assert_eq!(m.log, ["head", "reset", "stream", "stand"]);
    }

    /// A refusal or a failing stream in the head leaves the model untouched:
    /// no reset, no array, no stand.
    #[test]
    fn a_failure_in_the_head_leaves_the_model_untouched() {
        for fail in [short as fn() -> Fail, pipe] {
            let mut m = Mock::default();
            let said = fail().to_string();
            let e = run(&mut m, b"abcd", 4, Err(fail()), Ok(()), Ok(7)).expect_err("the head");
            assert_eq!(e.to_string(), said);
            assert_eq!(m.log, ["head"]);
        }
    }

    /// A failure after the reset (the arrays, a stream that ends early, the
    /// model's stand) leaves the model reset and is the failure returned,
    /// whatever its kind.
    #[test]
    fn a_failure_after_the_reset_leaves_the_model_reset_and_is_returned_whole() {
        for fail in [short as fn() -> Fail, pipe] {
            let said = fail().to_string();
            let mut m = Mock::default();
            let e = run(&mut m, b"abcd", 4, Ok(()), Err(fail()), Ok(7)).expect_err("the arrays");
            assert_eq!(e.to_string(), said);
            assert_eq!(m.log, ["head", "reset", "stream", "reset"]);
            let mut m = Mock::default();
            let e = run(&mut m, b"abcd", 4, Ok(()), Ok(()), Err(fail())).expect_err("the stand");
            assert_eq!(e.to_string(), said);
            assert_eq!(m.log, ["head", "reset", "stream", "stand", "reset"]);
        }
        let mut m = Mock::default();
        let e = run(&mut m, b"abc", 4, Ok(()), Ok(()), Ok(7)).expect_err("a short stream");
        assert!(e.to_string().contains("a short stream"), "{e}");
        assert_eq!(m.log, ["head", "reset", "stream", "reset"]);
    }

    /// A byte past the state is refused by name before the model stands, and
    /// leaves it reset.
    #[test]
    fn a_stream_that_goes_on_past_the_state_is_refused_and_the_model_reset() {
        let mut m = Mock::default();
        let e = run(&mut m, b"abcde", 4, Ok(()), Ok(()), Ok(7)).expect_err("a byte too many");
        match e {
            Fail::Refused(e) => assert!(e.to_string().contains("trailing bytes"), "{e}"),
            other => panic!("{other}: not a refusal by name"),
        }
        assert_eq!(m.log, ["head", "reset", "stream", "reset"]);
    }

    /// A reset that fails after a failed restore does not replace the
    /// restore's failure: both are named, the restore's whole; a reset that
    /// fails before the arrays is the card's failure and nothing streams.
    #[test]
    fn a_failing_reset_names_the_failure_that_called_for_it() {
        let mut m = Mock {
            fails_reset: Some(2),
            ..Mock::default()
        };
        let e = run(&mut m, b"abcd", 4, Ok(()), Err(short()), Ok(7)).expect_err("the arrays");
        match e {
            Fail::Reset { restore, reset } => {
                assert!(restore.to_string().contains("a short stream"), "{restore}");
                assert!(matches!(reset, GpuError::Protocol { .. }), "{reset}");
            }
            other => panic!("{other}: the failed reset is not named"),
        }
        let mut m = Mock {
            fails_reset: Some(1),
            ..Mock::default()
        };
        let e = run(&mut m, b"abcd", 4, Ok(()), Ok(()), Ok(7)).expect_err("the first reset");
        assert!(matches!(e, Fail::Card(GpuError::Protocol { .. })), "{e}");
        assert_eq!(m.log, ["head", "reset"]);
    }
}
