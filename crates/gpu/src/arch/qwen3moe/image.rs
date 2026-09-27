//! The prompt image: one input record for a whole prompt — the position of
//! its first token, then its ids, and on a chain with recurrent layers the
//! lane word after the ids' room — written into pinned host words and
//! copied to the card once per prompt, asynchronously, as far as the prompt
//! reaches. A unit of the prompt (a ubatch, or a walk of the layer program)
//! reads a window of the ids ([`PromptImage::windows`]); its embedding launch
//! derives its rows' positions and live key counts from the position word
//! and the window's offset.
//!
//! Layout, in u32 words: [`IN_POS0`] the prompt's first position, the ids
//! from [`IN_IDS`] with room for `cap` of them, and — when the image has one
//! — the lane word at `IN_IDS + cap`. The lane word is written and copied
//! once, at load; a prompt copies its first `n + 1` words.

use super::scratch::{IN_IDS, IN_POS0, Inbox, Io, LANE, param_view, put_input};
use crate::GpuError;
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::time::{Duration, Instant};

const WHAT: &str = "qwen3moe::image";

/// One prompt image's write: its tokens, the bytes copied to the card, the
/// host's fill of them and the enqueue of their copy, which does not wait
/// for it: the launches behind it do.
#[derive(Clone, Copy, Debug)]
pub struct ImageWrite {
    pub tokens: usize,
    pub bytes: usize,
    pub fill: Duration,
    pub copy: Duration,
}

/// Word offset of an image's lane word: after the ids' room of `cap`.
fn lane_at(cap: usize) -> usize {
    IN_IDS + cap
}

/// Words of an image with room for `cap` ids, with the lane word when
/// `lane`.
fn image_words(cap: usize, lane: bool) -> usize {
    lane_at(cap) + usize::from(lane)
}

/// The prompt image (module doc).
pub(super) struct PromptImage {
    /// The record, with room for a prompt of `cap` tokens.
    inbox: Inbox,
    /// Tokens the image has room for, and positions the rope table holds:
    /// the cache's rows.
    cap: usize,
    /// Whether the record carries the lane word.
    lane: bool,
    /// Tokens of the prompt the image holds (0 while it holds none), and the
    /// position of its first.
    len: usize,
    pub(super) pos0: u32,
    /// The last prompt's write.
    pub(super) last: Option<ImageWrite>,
}

/// A unit's windows onto the image: its ids, the prompt's position word,
/// the lane word when the image has one, and the ids' offset in the prompt.
/// Owned, so that the arena can be borrowed beside them.
pub(super) struct ImageWindows {
    ids: ManuallyDrop<DeviceBuffer<u32>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    lane: Option<ManuallyDrop<DeviceBuffer<u32>>>,
    first: usize,
}

impl ImageWindows {
    /// The unit's input, as its first launch reads it: row `t` is position
    /// `pos0 + first + t`.
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.ids,
            pos0: &self.pos0,
            first: self.first,
            lane: self.lane.as_deref(),
        }
    }
}

impl PromptImage {
    /// The image for a cache of `cap` rows, with the lane word [`LANE`]
    /// written and copied when `lane`. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        cap: usize,
        lane: bool,
    ) -> Result<PromptImage, GpuError> {
        let mut inbox = Inbox::new(stream, image_words(cap, lane))?;
        if lane {
            inbox.host_mut()?[lane_at(cap)] = LANE;
            inbox.upload(stream, image_words(cap, lane))?;
        }
        Ok(PromptImage {
            inbox,
            cap,
            lane,
            len: 0,
            pos0: 0,
            last: None,
        })
    }

    /// Write the record of `tokens` at positions `pos0 ..` — its first
    /// `n + 1` words — and enqueue their copy to the card. Asynchronous: the
    /// units behind it read it. A prompt whose positions pass the cache's
    /// rows, the rope table's too, is refused.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        self.len = 0;
        let n = tokens.len();
        let end = pos0 as usize + n;
        if n == 0 || end > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} tokens at positions {pos0}..{end}: the image and the rope table hold \
                     positions 0..{}",
                    self.cap
                ),
            ));
        }
        let t0 = Instant::now();
        put_input(self.inbox.host_mut()?, tokens, pos0)?;
        let t1 = Instant::now();
        self.inbox.upload(stream, IN_IDS + n)?;
        let t2 = Instant::now();
        self.len = n;
        self.pos0 = pos0;
        self.last = Some(ImageWrite {
            tokens: n,
            bytes: (IN_IDS + n) * size_of::<u32>(),
            fill: t1 - t0,
            copy: t2 - t1,
        });
        Ok(())
    }

    /// Windows onto the prompt's `tokens` in the image, its position word
    /// and its lane word: row `t` of the unit is position `pos0 +
    /// tokens.start + t`.
    pub(super) fn windows(&self, tokens: Range<usize>) -> Result<ImageWindows, GpuError> {
        if tokens.is_empty() || tokens.end > self.len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "tokens {tokens:?} are not inside the {}-token prompt in the image",
                    self.len
                ),
            ));
        }
        // SAFETY: tokens.end <= len <= cap, so the ids lie inside the
        // record's `IN_IDS + cap` words, and so do its position word and,
        // when the record has one, its lane word at `IN_IDS + cap`; the
        // windows live for one unit, while the image stays in place (it is
        // allocated only at load).
        let (ids, pos0, lane) = unsafe {
            (
                param_view::<u32>(self.inbox.dev(), IN_IDS + tokens.start, tokens.len()),
                param_view::<u32>(self.inbox.dev(), IN_POS0, 1),
                self.lane
                    .then(|| param_view::<u32>(self.inbox.dev(), lane_at(self.cap), 1)),
            )
        };
        Ok(ImageWindows {
            ids,
            pos0,
            lane,
            first: tokens.start,
        })
    }

    /// Device bytes of the record.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}
