//! ERNIE 5.5's prompt calls on the common host streaming
//! ([`PromptStream`](bloomery_gpu::host::prompt_stream::PromptStream), the `BLOOMERY_XSTREAM`
//! lever): the body hands the common owners its own facts as data and holds no stream code of its
//! own.
//!
//! A call opens inside the pass its boundary opened ([`crate::pass::prompt`]); each unit of the
//! pass is one walk of one unit ([`crate::placed::walk_pass`]). At a hosted layer's shadow the
//! walk picks ([`PromptStream::unit_pick`](bloomery_gpu::host::prompt_stream::PromptStream::unit_pick)):
//! the residency machine moves the host experts the unit's rows route to
//! at least [`ADMIT_FLOOR`] times into the card's pool, in place of its coldest residents, and the
//! host's union leaves them out. The card's slots then run under the moved map, each behind its
//! landing batch's event ([`crate::moe::Moe::enqueue_card_rows`]). The pick stays for the decode
//! after a call at position 0 and returns to the call's start otherwise
//! ([`Body::stream_begin`]).
//!
//! The experts are MXFP4 stacks and the common route GEMM has no entry for them, so the ring of
//! `split` cannot hold a streamed expert's slots: `split` is refused in the form refusal the
//! unset rule falls back from to `admit` (`shared/xstream_rule.rs`), with no ring reserved.

use bloomery_gpu::host::prompt_stream::ADMIT_FLOOR;
use bloomery_gpu::host::swap::{CallPick, CallReport};
use bloomery_gpu::host::xstream::{XMode, XSTREAM_FORM, XStreamBody};
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;

use crate::body::Body;

/// Why `split` is refused: the stream's ring holds a streamed expert's three matrices and runs
/// them as the common route GEMM, which names no entry for MXFP4 stacks. The form refusal
/// ([`XSTREAM_FORM`]) is the one the unset rule reads as "no ring for this layout" and answers
/// with `admit`; nothing is reserved or started when it is returned.
fn ring_refusal() -> GpuError {
    GpuError::shape(
        XSTREAM_FORM,
        "ernie55 routed experts are MXFP4 stacks, which the route GEMM has no entry for, so the \
         expert stream has no ring form for them; `admit` moves them into the card's pool",
    )
}

impl Body {
    /// Open the prompt call's streaming over `n` ids from position `start`
    /// ([`PromptStream::begin`](bloomery_gpu::host::prompt_stream::PromptStream::begin), every
    /// unit of the pass being a walk of a placed load): whether the call returns to its start,
    /// which is a call that continues a sequence holding decode heat. `false` and nothing more
    /// on a whole-card load or while the lever is off, which leaves the rule's heat as the steps
    /// made it.
    pub(crate) fn stream_begin(
        &mut self,
        stream: &CudaStream,
        n: usize,
        start: u32,
    ) -> Result<bool, GpuError> {
        if self.xs.stream.mode() == XMode::Off {
            return Ok(false);
        }
        let layers = self.sh.n_layer;
        let Some(p) = self.placed.as_mut() else {
            return Ok(false);
        };
        // Read once: the call keeps no row, so the history does not move until its end.
        let history = p.shell.hybrid.call_history(start);
        self.xs.stream.begin(
            &mut p.shell.hybrid,
            stream,
            true,
            n,
            start as usize + n,
            history,
            ADMIT_FLOOR,
            0.0,
            layers,
        )?;
        Ok(history)
    }

    /// End the prompt call's streaming
    /// ([`PromptStream::end`](bloomery_gpu::host::prompt_stream::PromptStream::end)), its
    /// placement `kept` for the decode after it; nothing on a whole-card load or a call that did
    /// not stream.
    pub(crate) fn stream_end(&mut self, stream: &CudaStream, kept: bool) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(p) => self.xs.stream.end(&mut p.shell.hybrid, stream, kept),
            None => Ok(()),
        }
    }

    /// The last streaming prompt call's picks, each with its walk, and its end; empty and `None`
    /// after a call that did not stream. Taken
    /// ([`PromptStream::take_pick_records`](bloomery_gpu::host::prompt_stream::PromptStream::take_pick_records)):
    /// a second read is empty.
    pub fn take_stream_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        self.xs.stream.take_pick_records()
    }
}

impl XStreamBody for Body {
    /// What the next prompt calls move (`BLOOMERY_XSTREAM`): `off` until set here. `Split` is
    /// refused in the form refusal ([`ring_refusal`]) before anything starts. Refused by name:
    /// `admit` or `split` on a load made without the prompt pass (no walk to pick in) or on a
    /// whole-card load, with no residency machine, and any change while a call streams.
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        const WHAT_S: &str = "ernie55::Body::set_xstream";
        if mode != XMode::Off && self.pass.is_none() {
            return Err(GpuError::State {
                what: WHAT_S,
                missing: "a load opened with the prompt pass (the walks the stream serves)",
            });
        }
        let tier = self.placed.as_mut().map(|p| &mut p.shell.hybrid);
        self.xs
            .set_xstream(tier, (gpu, WHAT_S), mode, |_| Err(ring_refusal()))
    }

    /// The load line's `xstream=` word
    /// ([`PromptStream::word`](bloomery_gpu::host::prompt_stream::PromptStream::word)); the mode
    /// alone on a whole-card load.
    fn xstream_word(&self) -> String {
        match &self.placed {
            Some(p) => self.xs.stream.word(&p.shell.hybrid),
            None => self.xs.stream.mode().word().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::xstream::XSTREAM_FORM;

    use super::ring_refusal;

    /// `split` is refused in the form refusal the unset rule falls back from to `admit`: any other
    /// refusal fails the default load. Mutant: `ring_refusal` names another `what`.
    #[test]
    fn split_is_refused_in_the_form_the_unset_rule_answers_with_admit() {
        assert!(
            matches!(
                ring_refusal(),
                GpuError::Shape {
                    what: XSTREAM_FORM,
                    ..
                }
            ),
            "{}",
            ring_refusal()
        );
    }
}
