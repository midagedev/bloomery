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
//!
//! A media call's rows ride beside it ([`MediaRows`]): the rows of the call's
//! spans, widened to f32 in pinned host memory once before its first unit,
//! and per unit the runs of positions whose rows the unit's first layer
//! copies over the embeddings it just wrote.

use super::scratch::{IN_IDS, IN_POS0, Inbox, Io, LANE, param_view, put_input};
use crate::media::{ImageSpan, widen_bf16};
use crate::tensor::{Window, WindowMut};
use crate::{GpuError, capturing};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
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

/// A span of the call as staged: its rows' positions in the sequence and the
/// offset of its first row in the staging buffer.
struct Staged {
    at: Range<usize>,
    from: usize,
}

/// The positions of one unit whose rows are media: `rows` rows from unit row
/// `row`, read from the staging buffer at `from`.
struct Run {
    row: usize,
    rows: usize,
    from: usize,
}

/// The runs of a unit of the positions `unit` over the staged `spans`
/// (`width` values a row): each span's rows inside the unit, at their rows
/// of the unit. A span the unit's boundary cuts gives each of its units its
/// own rows only.
fn runs_of<'a>(
    spans: &'a [Staged],
    unit: &'a Range<usize>,
    width: usize,
) -> impl Iterator<Item = Run> + 'a {
    spans.iter().filter_map(move |st| {
        let (lo, hi) = (st.at.start.max(unit.start), st.at.end.min(unit.end));
        (lo < hi).then(|| Run {
            row: lo - unit.start,
            rows: hi - lo,
            from: st.from + (lo - st.at.start) * width,
        })
    })
}

/// A media call's rows on their way into the unit arena (module doc). The
/// staging buffer is pinned host memory, written once per call by
/// [`MediaRows::begin`], every span's rows after the one before; each unit's
/// runs ([`MediaRows::plan`]) are copied to the arena's `x` by
/// [`MediaRows::write`], one asynchronous copy a run, in the unit's first
/// layer behind its embedding. A call is eager: no copy is ever recorded into
/// a graph, which would bake the staging address.
pub(super) struct MediaRows {
    host: PinnedHostBuffer<f32>,
    /// Recorded behind each unit's copies: the staging buffer is not written
    /// or freed while a copy still reads it.
    copied: CudaEvent,
    /// Values of a row.
    width: usize,
    spans: Vec<Staged>,
    /// The positions of the unit planned last and the runs of media rows
    /// inside it.
    unit: Range<usize>,
    runs: Vec<Run>,
    /// An armed instrument: each unit's rows as layer 0 reads them, `(first
    /// position, rows)`, read back eagerly after the media rows are in.
    tap: Option<Vec<(usize, Vec<f32>)>>,
}

impl MediaRows {
    /// No rows staged, for rows of `width` values; the buffer grows with the
    /// first call that needs it.
    pub(super) fn new(stream: &CudaStream, width: usize) -> Result<MediaRows, GpuError> {
        let ctx = stream.context();
        Ok(MediaRows {
            host: PinnedHostBuffer::zeroed(ctx, 0)?,
            copied: ctx.new_event(None)?,
            width,
            spans: Vec::new(),
            unit: 0..0,
            runs: Vec::new(),
            tap: None,
        })
    }

    /// Stage `spans`' rows of a call whose first position is `first`: each
    /// span's bf16 row widened to f32 (bf16 is the high half of an f32), span
    /// after span. Waits for the copies of the call before, and holds no span
    /// of it.
    pub(super) fn begin(
        &mut self,
        stream: &CudaStream,
        first: usize,
        spans: &[ImageSpan<'_>],
    ) -> Result<(), GpuError> {
        self.copied.synchronize()?;
        self.end();
        let total: usize = spans.iter().map(|s| s.span.rows.len()).sum();
        if total > self.host.len() {
            self.host = PinnedHostBuffer::zeroed(stream.context(), total)?;
        }
        let mut from = 0;
        for s in spans {
            let rows = s.span.rows;
            widen_bf16(rows, &mut self.host.as_mut_slice()[from..from + rows.len()]);
            self.spans.push(Staged {
                at: first + s.span.at.start..first + s.span.at.end,
                from,
            });
            from += rows.len();
        }
        Ok(())
    }

    /// The call is over: no span, no run.
    pub(super) fn end(&mut self) {
        self.spans.clear();
        self.runs.clear();
    }

    /// The runs of a unit of the positions `unit`: each staged span's rows
    /// inside it, at their rows of the unit.
    pub(super) fn plan(&mut self, unit: Range<usize>) {
        self.runs.clear();
        self.runs.extend(runs_of(&self.spans, &unit, self.width));
        self.unit = unit;
    }

    /// Copy the planned runs over the rows of `x` (the arena's residual,
    /// `width` values a row), each in one asynchronous copy on `stream`;
    /// then, with the tap armed, read the unit's rows back. Refused by name
    /// on a capturing stream.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT_W: &str = "qwen3moe::MediaRows::write";
        if self.runs.is_empty() && self.tap.is_none() {
            return Ok(());
        }
        if capturing(stream)? {
            return Err(GpuError::state(
                WHAT_W,
                "an eager unit: a captured graph would bake the staged rows' address",
            ));
        }
        for r in &self.runs {
            let src = self
                .host
                .as_slice()
                .get(r.from..r.from + r.rows * self.width)
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT_W,
                        format!(
                            "{} rows of {} from value {} of {} staged",
                            r.rows,
                            self.width,
                            r.from,
                            self.host.len()
                        ),
                    )
                })?;
            let mut dst = WindowMut::<f32>::of_mut(
                x,
                r.row * self.width * size_of::<f32>(),
                r.rows * self.width,
            )?;
            // SAFETY: `src` is pinned memory this staging buffer owns; it is
            // neither written ([`MediaRows::begin`] waits on `copied`, recorded
            // behind this copy) nor freed (the drop waits on it) while the
            // copy reads it, so the copy reads it whole.
            unsafe { dst.copy_from_host_async_unchecked(stream, src)? };
            self.copied.record(stream)?;
        }
        if let Some(tap) = self.tap.as_mut() {
            let rows = Window::<f32>::of(x, 0, self.unit.len() * self.width)?;
            tap.push((self.unit.start, rows.to_host_vec(stream)?));
        }
        Ok(())
    }

    /// Arm the tap, or disarm it and drop what it kept.
    pub(super) fn set_tap(&mut self, on: bool) {
        self.tap = on.then(Vec::new);
    }

    /// What the tap kept since the last call, in unit order; `None` unarmed.
    pub(super) fn take_tap(&mut self) -> Option<Vec<(usize, Vec<f32>)>> {
        self.tap.as_mut().map(std::mem::take)
    }
}

impl Drop for MediaRows {
    fn drop(&mut self) {
        // The pinned rows outlive any copy still reading them. A failure on
        // the drop path is unreportable and ignored.
        let _ = self.copied.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span that a unit boundary cuts gives each unit its own rows: the
    /// rows of the span's positions inside the unit, at the unit's row of
    /// that position, read from the staging buffer where that position's row
    /// was staged. Spans of rows of 4 values: one at positions 10..30 staged
    /// from value 0, one at 34..36 staged from value 80 (20 rows of 4 before
    /// it).
    #[test]
    fn a_span_cut_by_units_gives_each_unit_its_own_rows() {
        let spans = [
            Staged {
                at: 10..30,
                from: 0,
            },
            Staged {
                at: 34..36,
                from: 80,
            },
        ];
        let runs = |unit: Range<usize>| -> Vec<(usize, usize, usize)> {
            runs_of(&spans, &unit, 4)
                .map(|r| (r.row, r.rows, r.from))
                .collect()
        };
        // The unit that ends inside the first span: its last 6 rows.
        assert_eq!(runs(0..16), [(10, 6, 0)]);
        // The next unit starts inside it: its first 14 rows, from position
        // 16's row (6 rows in).
        assert_eq!(runs(16..32), [(0, 14, 24)]);
        // A unit wholly inside the span.
        assert_eq!(runs(12..14), [(0, 2, 8)]);
        // The unit that holds the second span, none of the first.
        assert_eq!(runs(32..40), [(2, 2, 80)]);
        // Units outside both spans.
        assert!(runs(30..34).is_empty());
        assert!(runs(36..44).is_empty());
        // A unit that holds both spans' rows, in order.
        assert_eq!(runs(0..40), [(10, 20, 0), (34, 2, 80)]);
    }
}
