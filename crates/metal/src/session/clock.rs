//! The step vocabulary every family's session steps in: how a token reaches
//! the device ([`StepMode`]), what a step returns ([`Stepped`]), what the
//! steps did with the token ahead ([`StepStats`]), and the step clock
//! ([`StepClock`]), the coarser instrument a profiling run reads a mode's
//! averaged phases through. The tokens themselves — the one running ahead,
//! the one encoded and parked — are the machinery's own words
//! ([`Ahead`], [`Primed`]), held by a family's session through its step
//! protocol.

use core::time::Duration;
use std::sync::{Mutex, PoisonError};

use bloomery_device::DeviceError;

use crate::encode::EncodeCount;
use crate::stream::{GpuStamps, Parked, Signalled};

/// How long a profiled group's stamps wait for the completed status its
/// settle already implied: the buffer's signal fires before the driver
/// reports it complete, so the stamps give the report a moment.
pub(crate) const STAMP_WAIT: Duration = Duration::from_secs(1);

/// What a step returns: the head's pick and what the token encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stepped {
    pub id: u32,
    pub count: EncodeCount,
}

/// How a step's token reaches the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepMode {
    /// The id is written into the host id slot; the step commits its token
    /// and waits for it, and nothing runs ahead. The next position's
    /// host-fed token is encoded during the wait and committed by the next
    /// host-fed call, its id written first.
    HostFed,
    /// As host-fed, unless a token runs ahead with this id; either way the
    /// next token is committed before this one is waited, embedding this
    /// token's pick on the device.
    Chain,
    /// The ids are known ahead of the calls: a prompt's. The token this call
    /// names lands with no head — its pick is nobody's input, the call after
    /// names its id — and with `next` named, that token is committed before
    /// this one is waited, embedding `next` from this token's record slot,
    /// which the host writes for it. A chain's token ahead reads the pick
    /// this token's head would have written there; this token writes no
    /// head, so the slot is the host's. `next` of `None` commits nothing
    /// ahead: the last row of a span a checkpoint ends, or the row before
    /// one whose pick is read.
    Ahead { next: Option<u32> },
}

/// What the session's steps did with the token ahead, the host's waits, the
/// command buffers committed and the passes run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStats {
    /// The host's waits on the session's stream: every settle and sync.
    pub waits: u64,
    /// The command buffers committed on the session's stream.
    pub submits: u64,
    /// Tokens committed ahead of the call that names them.
    pub ahead: u64,
    /// Tokens ahead a call kept: its id was their embedded pick.
    pub kept: u64,
    /// Tokens ahead waited out and dropped: a broken promise, a fault, a
    /// reset, a traced step, a checkpoint or a cut.
    pub discarded: u64,
    /// Steps at the cache's last row that would commit a token ahead, and
    /// commit none.
    pub last_row: u64,
    /// Host-fed tokens committed from the encode the step before made.
    pub primed: u64,
    /// Passes that ran to their settle.
    pub passes: u64,
}

/// The token committed ahead of its call.
pub(crate) struct Ahead {
    pub(crate) signalled: Signalled,
    pub(crate) count: EncodeCount,
    pub(crate) pos: u32,
    /// The id its embed reads: the pick of the token before it (a chain's,
    /// learned at that token's land) or the id its committing call named (an
    /// ahead's, which knew it).
    pub(crate) embeds: u32,
    /// Whether the token's encode carried the head: a chain's does, its pick
    /// the next promise; an ahead's does not, the call after names its id.
    pub(crate) head: bool,
}

/// The next host-fed token, encoded during a host-fed step's wait and held
/// off the queue until the call that names its id.
pub(crate) struct Primed {
    pub(crate) parked: Parked,
    pub(crate) count: EncodeCount,
    pub(crate) pos: u32,
    /// The encode's own seconds, when the step clock ran at the encode.
    pub(crate) encode_s: Option<f64>,
}

/// The host's encode of a committed token, as the clock counts it.
#[derive(Clone, Copy)]
pub(crate) enum Encoded {
    /// Encoded at the commit, from this host second on.
    Since(f64),
    /// Encoded and parked a step before, in this many seconds.
    Took(f64),
}

/// One committed token the clock holds until its settle: its identity on the
/// stream, its stamps (a command buffer's handle, which only a lock may
/// carry), and the host's seconds around its encode and commit.
pub(crate) struct PendingToken {
    pub(crate) value: u64,
    pub(crate) stamps: GpuStamps,
    pub(crate) encode_s: f64,
    pub(crate) submitted: f64,
}

/// How many step modes a clock holds a row for.
const MODES: usize = 3;

/// A mode's place in a step clock's rows.
pub(crate) fn row_of(mode: StepMode) -> usize {
    match mode {
        StepMode::HostFed => 0,
        StepMode::Chain => 1,
        StepMode::Ahead { .. } => 2,
    }
}

/// One mode's phases, summed; the averages are [`StepPhase`].
#[derive(Clone, Copy, Default)]
pub(crate) struct Phases {
    pub(crate) encode_s: f64,
    pub(crate) start_s: f64,
    pub(crate) gpu_s: f64,
    pub(crate) wake_s: f64,
}

/// One mode's averaged step phases, as a run's steps spent them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepPhase {
    /// The settled steps the averages hold.
    pub steps: u64,
    /// The host's encode of a token, µs: the encoder's begin to the commit,
    /// or to the park for a token a host-fed step encoded during its wait,
    /// which no step waits on.
    pub encode_us: f64,
    /// A token's GPU start against the host clock at its commit, µs.
    pub start_us: f64,
    /// The GPU's own µs on a token.
    pub gpu_us: f64,
    /// The host's µs from a token's GPU end to the settle's return.
    pub wake_us: f64,
}

/// The step clock: what a run's steps spent, by the mode of the call that
/// settled each token. A session starts one; a session whose clock is off
/// times nothing, and a step of it costs no read.
pub struct StepClock {
    /// The committed tokens whose settle has not come, the run-ahead one
    /// deep: two stand at once, the token a call settles and the one it
    /// committed ahead of it. The stamps' buffer handle is not `Sync`, so
    /// the lock carries it and it never leaves the clock.
    pending: Mutex<Vec<PendingToken>>,
    /// One row a mode, [`StepMode`]'s order.
    rows: [Phases; MODES],
    steps: [u64; MODES],
}

impl StepClock {
    pub(crate) fn new() -> Self {
        Self {
            // The run-ahead holds two; a step's commits never grow it.
            pending: Mutex::new(Vec::with_capacity(2)),
            rows: [Phases::default(); MODES],
            steps: [0; MODES],
        }
    }

    /// The registry, taken as it stands: every change to it is one push or
    /// one removal, so a panic while it was held left it whole.
    fn registry(&self) -> std::sync::MutexGuard<'_, Vec<PendingToken>> {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The token `commit` clocked, held until its settle.
    pub(crate) fn committed(&self, token: PendingToken) {
        self.registry().push(token);
    }

    /// The token of signal `value`, taken from the registry: `None` when the
    /// clock holds none (it was committed before the clock started).
    pub(crate) fn take(&self, value: u64) -> Option<PendingToken> {
        let mut pending = self.registry();
        let at = pending.iter().position(|t| t.value == value)?;
        Some(pending.swap_remove(at))
    }

    /// The token of signal `value` whose settle will not come — dropped,
    /// reset — its clock dropped with it.
    pub(crate) fn dropped(&self, value: u64) {
        let mut pending = self.registry();
        if let Some(at) = pending.iter().position(|t| t.value == value) {
            pending.swap_remove(at);
        }
    }

    /// The settled token's phases under `mode`'s row: its stamps read (the
    /// settle already implied them complete) and `wake` the host's clock at
    /// the record.
    pub(crate) fn settled(
        &mut self,
        mode: StepMode,
        token: PendingToken,
        wake: f64,
    ) -> Result<(), DeviceError> {
        let stamps = token.stamps.times(STAMP_WAIT)?;
        let row = &mut self.rows[row_of(mode)];
        row.encode_s += token.encode_s;
        row.start_s += stamps.start - token.submitted;
        row.gpu_s += stamps.end - stamps.start;
        row.wake_s += wake - stamps.end;
        self.steps[row_of(mode)] += 1;
        Ok(())
    }

    /// `mode`'s averaged phases; zeros when no step of the mode settled.
    #[must_use]
    pub fn phase(&self, mode: StepMode) -> StepPhase {
        let n = self.steps[row_of(mode)];
        let p = &self.rows[row_of(mode)];
        let avg = |s: f64| if n == 0 { 0.0 } else { s / n as f64 * 1e6 };
        StepPhase {
            steps: n,
            encode_us: avg(p.encode_s),
            start_us: avg(p.start_s),
            gpu_us: avg(p.gpu_s),
            wake_us: avg(p.wake_s),
        }
    }
}
