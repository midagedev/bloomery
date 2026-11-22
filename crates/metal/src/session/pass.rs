//! The pass vocabulary and the pass's command-buffer machinery: how a prompt
//! call's ids run as one pass (every layer once over all the rows), what a
//! pass returns ([`PassOut`], [`Passed`]), the copies a pass leaves after
//! rows inside it ([`StateOut`]), how long an open's parts took
//! ([`OpenSplit`]), when the mark copies open ([`MarkCopies`]) — and
//! [`LayerBuffers`], the buffers a walk's ops encode on: a pass takes one a
//! layer, a verify one for the whole walk, each committed with a signal and
//! waited only at the last, so the host encodes a layer while the card runs
//! the one before it and no host wait sits inside the pass.

use core::time::Duration;

use bloomery_device::DeviceError;

use crate::encode::EncodeCount;
use crate::stream::{GpuStamps, MetalStream, Signalled, TokenRecording, host_secs};

/// When a session opens the copies a pass leaves after rows inside it
/// ([`StateOut`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkCopies {
    /// With the session, beside the rows arena: a server's session, whose
    /// chat calls mark their messages.
    AtOpen,
    /// On the first pass that takes one: a session that never marks never
    /// holds them.
    OnFirstMark,
}

/// How long each part of a session's open took, on the host's clock. The
/// pipelines are built beside the other three, which run one after another:
/// the open's wall is the longer of the two, not the sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenSplit {
    /// The plan read from the file and the weights copied into the arena.
    pub load: Duration,
    /// The rows arena's buffers allocated, and the mark copies' when the
    /// open makes them.
    pub rows: Duration,
    /// The residency set's commit: the driver wiring every buffer the open
    /// made.
    pub residency: Duration,
    /// The kernels' pipelines built, the prefill attention's among them.
    pub pipelines: Duration,
}

/// What a pass returns of its head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassOut {
    /// No head: the rows' picks are nobody's. The pass returns the id it was
    /// last fed.
    None,
    /// The head over the last row: its pick, and its logits row in the
    /// session's `logits_row`. (The verify arm's every-row head is its own
    /// variant.)
    Last,
}

/// The copies of the recurrent state a pass leaves, named by the rows they
/// follow: each recurrent layer's state lane and ring after that row, the
/// bits a pass ending there leaves in its stores, which the session's
/// checkpoint-from-a-copy takes as the point at the position after the row. A
/// prompt call names the message marks inside a pass here.
#[derive(Clone, Copy, Debug, Default)]
pub struct StateOut<'a> {
    /// The rows after which a copy is taken, ascending, each a row of the
    /// pass; at most [`SNAPS`](crate::kernels::SNAPS). None takes none.
    pub after_rows: &'a [u32],
}

/// What a pass returns: the head's pick, or the id it was last fed, and what
/// its ops encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Passed {
    pub id: u32,
    pub count: EncodeCount,
}

/// The command buffers of a walk: the buffer being encoded, and the signal of
/// the last one committed. A pass takes one a layer ([`LayerBuffers::new`]): a
/// buffer is committed with a signal and not waited when the walk reaches the
/// next layer, so the host encodes a layer while the card runs the one before
/// it; each buffer opens at the stream's token boundary, which the buffer
/// ahead signals at its end, so it sees what that one wrote. A verify takes
/// one for the whole walk ([`LayerBuffers::whole`]), as a step does: the groups
/// close with barriers, so a boundary between layers orders nothing more and
/// costs the card an idle gap each. The walk settles the last signal once.
pub(crate) struct LayerBuffers<'s> {
    pub(crate) stream: &'s MetalStream,
    /// Whether every op encodes on the one buffer, whatever its layer.
    pub(crate) whole: bool,
    /// The buffer being encoded, not yet committed.
    pub(crate) open: Option<OpenLayer<'s>>,
    /// The last committed buffer's signal.
    pub(crate) last: Option<Signalled>,
    /// What the committed buffers encoded.
    pub(crate) count: EncodeCount,
    /// Each committed buffer's host stamps and its GPU stamps handle, when a
    /// caller traces the walk ([`LayerBuffers::traced`]); `None` reads no clock.
    pub(crate) trace: Option<Vec<Traced>>,
}

/// A layer's buffer being encoded.
pub(crate) struct OpenLayer<'s> {
    /// The layer its ops name; the embed opens a buffer before one does.
    pub(crate) layer: Option<usize>,
    pub(crate) recording: TokenRecording<'s>,
    /// The host clock when the buffer began, read only in a traced walk.
    pub(crate) opened: Option<f64>,
}

/// A committed buffer of a traced walk: the host clock at its begin and at
/// its commit's return, and the handle its GPU stamps are read through once
/// the walk settles.
pub(crate) struct Traced {
    pub(crate) opened: f64,
    pub(crate) committed: f64,
    pub(crate) stamps: GpuStamps,
}

impl<'s> LayerBuffers<'s> {
    /// A buffer a layer.
    pub(crate) fn new(stream: &'s MetalStream) -> Self {
        Self {
            stream,
            whole: false,
            open: None,
            last: None,
            count: EncodeCount::default(),
            trace: None,
        }
    }

    /// One buffer for the whole walk.
    pub(crate) fn whole(stream: &'s MetalStream) -> Self {
        Self {
            whole: true,
            ..Self::new(stream)
        }
    }

    /// These buffers, stamping every buffer they commit.
    pub(crate) fn traced(self) -> Self {
        Self {
            trace: Some(Vec::new()),
            ..self
        }
    }

    /// The buffer an op of `layer` encodes on. The open one when the walk is
    /// one buffer, when it is `layer`'s, or when either names no layer;
    /// otherwise the open buffer is committed and the next one opened.
    pub(crate) fn at(
        &mut self,
        layer: Option<usize>,
    ) -> Result<&mut TokenRecording<'s>, DeviceError> {
        let open = match self.open.take() {
            Some(mut open)
                if self.whole || layer.is_none() || open.layer.is_none() || open.layer == layer =>
            {
                open.layer = open.layer.or(layer);
                open
            }
            behind => {
                if let Some(behind) = behind {
                    self.commit(behind)?;
                }
                let opened = self.trace.is_some().then(host_secs);
                OpenLayer {
                    layer,
                    recording: self.stream.begin()?.open_token_encoder()?,
                    opened,
                }
            }
        };
        Ok(&mut self.open.insert(open).recording)
    }

    /// Commit `open` with a signal, unwaited.
    pub(crate) fn commit(&mut self, open: OpenLayer<'s>) -> Result<(), DeviceError> {
        let OpenLayer {
            recording, opened, ..
        } = open;
        let (recording, count) = recording.finish();
        let stamps = opened.map(|opened| (opened, recording.stamps()));
        // A buffer's own failure is named at the last signal's settle: the
        // driver reports a failed buffer there, whichever buffer it was.
        self.last = Some(self.stream.submit_signalled(recording)?);
        self.count += count;
        if let (Some(trace), Some((opened, stamps))) = (self.trace.as_mut(), stamps) {
            trace.push(Traced {
                opened,
                committed: host_secs(),
                stamps,
            });
        }
        Ok(())
    }

    /// Commit the buffer still open, the walk done.
    pub(crate) fn finish(&mut self) -> Result<(), DeviceError> {
        match self.open.take() {
            Some(open) => self.commit(open),
            None => Ok(()),
        }
    }
}
