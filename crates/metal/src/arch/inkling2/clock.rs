//! The step clock of the Inkling 2 session: what a run's host-fed steps
//! spent, phase by phase, read from the GPU's own stamps and the host's
//! clock.
//!
//! A step is host-fed: the host embeds the id, encodes the first command
//! buffer of the token, commits it, encodes and commits the next two under
//! the first one's GPU time, and waits for the last. So a token's wall is
//! the card's busy time in its buffers, the idle between a buffer's end and
//! the next one's start (the host's commit not yet there, or the boundary
//! event not yet signalled), and the gap between two tokens — wake from the
//! wait, read the pick, embed, zero, encode the first buffer, commit, and
//! the card's start — which is the price of having no token ahead
//! (`Target::step`). [`StepClock`] measures each of those terms.
//!
//! The common step clock ([`crate::session::clock::StepClock`]) holds one
//! buffer a token and a chain; this one holds the buffers of a token. A
//! session whose clock is off reads no stamp and no host second.

use bloomery_device::DeviceError;

use crate::session::clock::{STAMP_WAIT, StepPhase};
use crate::stream::{GpuStamps, host_secs};

/// One command buffer of a traced token: its stamps and the host's second at
/// its commit.
pub(super) struct Chunk {
    stamps: GpuStamps,
    committed: f64,
}

/// What a step's walk records while the clock runs: the host's seconds at
/// the step's start and at the wait's return, and every command buffer.
pub(super) struct Trace {
    start: f64,
    wake: f64,
    chunks: Vec<Chunk>,
}

impl Trace {
    pub(super) fn new() -> Self {
        Self {
            start: 0.0,
            wake: 0.0,
            chunks: Vec::with_capacity(3),
        }
    }

    /// The step begins now.
    pub(super) fn begin(&mut self) {
        self.start = host_secs();
    }

    /// A command buffer was committed now.
    pub(super) fn committed(&mut self, stamps: GpuStamps) {
        self.chunks.push(Chunk {
            stamps,
            committed: host_secs(),
        });
    }

    /// The wait for the last buffer returned now.
    pub(super) fn woke(&mut self) {
        self.wake = host_secs();
    }
}

/// One step's terms, seconds.
struct Step {
    /// The host from the step's start to the first buffer's commit: embed,
    /// zero, the first encode.
    pre: f64,
    /// The host from the step's start to the last buffer's commit.
    encode: f64,
    /// The first buffer's GPU start after its commit.
    start: f64,
    /// The card's busy seconds in the token's buffers.
    gpu: f64,
    /// The card's idle seconds between the token's buffers.
    gaps: f64,
    /// The host's seconds from the last buffer's GPU end to the wait's return.
    wake: f64,
    /// The step's wall, start to the wait's return.
    wall: f64,
    first_start: f64,
    last_end: f64,
    chunks: usize,
}

impl Step {
    fn of(t: &Trace) -> Result<Step, DeviceError> {
        let times: Vec<_> = t
            .chunks
            .iter()
            .map(|c| c.stamps.times(STAMP_WAIT))
            .collect::<Result<_, _>>()?;
        let (Some(c0), Some(cn)) = (t.chunks.first(), t.chunks.last()) else {
            return Err(DeviceError::backend(
                "step clock",
                "a clocked step committed no command buffer".to_string(),
            ));
        };
        let (first, last) = (times[0], times[times.len() - 1]);
        let gpu = times.iter().map(|g| g.end - g.start).sum();
        let gaps = times.windows(2).map(|w| w[1].start - w[0].end).sum();
        Ok(Step {
            pre: c0.committed - t.start,
            encode: cn.committed - t.start,
            start: first.start - c0.committed,
            gpu,
            gaps,
            wake: t.wake - last.end,
            wall: t.wake - t.start,
            first_start: first.start,
            last_end: last.end,
            chunks: times.len(),
        })
    }
}

/// Seconds summed over a run's steps.
#[derive(Clone, Copy, Default)]
struct Sums {
    pre: f64,
    encode: f64,
    start: f64,
    gpu: f64,
    gaps: f64,
    wake: f64,
    wall: f64,
    chunks: f64,
    /// The card's idle between a step's last end and the next step's first
    /// start, summed over the steps that had a step before them.
    between: f64,
}

/// The step clock: what a run's host-fed steps spent.
pub struct StepClock {
    sums: Sums,
    steps: u64,
    between_steps: u64,
    /// The last settled step's GPU end, for the gap to the next one.
    last_end: Option<f64>,
}

/// The averaged terms of a run's steps, µs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepTerms {
    /// The settled steps the averages hold.
    pub steps: u64,
    /// The command buffers a token ran in.
    pub chunks: f64,
    /// The host from a step's start to its first commit: the embed, the zero
    /// and the first buffer's encode, before the card can start.
    pub pre_us: f64,
    /// The card's idle between a token's buffers.
    pub gap_us: f64,
    /// The card's idle between a token's last buffer and the next token's
    /// first, over the tokens that followed one: the host-fed gap.
    pub between_us: f64,
    /// The step's wall, start to the wait's return.
    pub wall_us: f64,
}

impl StepClock {
    pub(super) fn new() -> Self {
        Self {
            sums: Sums::default(),
            steps: 0,
            between_steps: 0,
            last_end: None,
        }
    }

    /// The step `t` traced, settled: its stamps read.
    pub(super) fn settled(&mut self, t: &Trace) -> Result<(), DeviceError> {
        let s = Step::of(t)?;
        let a = &mut self.sums;
        a.pre += s.pre;
        a.encode += s.encode;
        a.start += s.start;
        a.gpu += s.gpu;
        a.gaps += s.gaps;
        a.wake += s.wake;
        a.wall += s.wall;
        a.chunks += s.chunks as f64;
        if let Some(end) = self.last_end {
            a.between += s.first_start - end;
            self.between_steps += 1;
        }
        self.last_end = Some(s.last_end);
        self.steps += 1;
        Ok(())
    }

    /// A step that did not run in the clock's sequence (a pass, a reset, a
    /// profile): the gap to the next step is not one.
    pub(super) fn broken(&mut self) {
        self.last_end = None;
    }

    /// The common step vocabulary's phases of the host-fed mode: the
    /// encode of a token (the host's whole encode, its first buffer's alone
    /// is [`StepTerms::pre_us`]), the first buffer's GPU start after its
    /// commit, the card's busy µs, the host's µs from the last end to the
    /// wait's return.
    #[must_use]
    pub fn phase(&self) -> StepPhase {
        let n = self.steps as f64;
        let avg = |s: f64| if self.steps == 0 { 0.0 } else { s / n * 1e6 };
        StepPhase {
            steps: self.steps,
            encode_us: avg(self.sums.encode),
            start_us: avg(self.sums.start),
            gpu_us: avg(self.sums.gpu),
            wake_us: avg(self.sums.wake),
        }
    }

    /// The terms the common phases do not hold.
    #[must_use]
    pub fn terms(&self) -> StepTerms {
        let n = self.steps as f64;
        let avg = |s: f64| if self.steps == 0 { 0.0 } else { s / n * 1e6 };
        StepTerms {
            steps: self.steps,
            chunks: if self.steps == 0 {
                0.0
            } else {
                self.sums.chunks / n
            },
            pre_us: avg(self.sums.pre),
            gap_us: avg(self.sums.gaps),
            between_us: if self.between_steps == 0 {
                0.0
            } else {
                self.sums.between / self.between_steps as f64 * 1e6
            },
            wall_us: avg(self.sums.wall),
        }
    }
}
