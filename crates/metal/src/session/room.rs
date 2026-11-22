//! The room rule: the bytes a session asks of the driver held to the
//! device's free bytes, generic over the family's byte plan
//! ([`SessionBytes`]). The one refusal the header alone cannot decide: the
//! census reads the free bytes and hands them over (the open its reading,
//! the plan the same figure).

/// A pass-through page: Apple silicon's, the size a Metal buffer's bytes round
/// up to in the driver's count.
pub const PAGE: u64 = 16 * 1024;

/// One class of buffers the open makes: how many, the bytes asked of the
/// driver, and the bytes its pages count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub buffers: u64,
    pub requested: u64,
    pub paged: u64,
}

impl Tally {
    /// Adds one buffer of `bytes`: the family's byte plan calls this, a
    /// macOS module.
    #[cfg(target_os = "macos")]
    pub(crate) fn add(&mut self, bytes: u64) {
        self.buffers += 1;
        self.requested += bytes;
        self.paged += bytes.div_ceil(PAGE) * PAGE;
    }
}

/// The bytes a session makes buffers of, by class: the file's weights, the
/// attention caches, the recurrent stores, the token arena's other buffers
/// and the rows arena, which the open makes, and the copies the session
/// makes after it: a pass's mark copies and the checkpoints the budget
/// grants at the cache's rows (a buffer a store a point, made a point at a
/// time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bytes {
    pub weights: Tally,
    pub kv: Tally,
    pub state: Tally,
    pub scratch: Tally,
    pub rows: Tally,
    pub copies: Tally,
}

impl Bytes {
    /// The classes the open makes buffers of, together: every class but the
    /// copies.
    #[must_use]
    pub fn total(&self) -> Tally {
        sum(&[self.weights, self.kv, self.state, self.scratch, self.rows])
    }

    /// Every class the session can hold, the copies among them: the room
    /// rule's figure.
    #[must_use]
    pub fn held(&self) -> Tally {
        sum(&[self.total(), self.copies])
    }
}

/// The tallies `parts` summed.
fn sum(parts: &[Tally]) -> Tally {
    parts.iter().fold(Tally::default(), |a, t| Tally {
        buffers: a.buffers + t.buffers,
        requested: a.requested + t.requested,
        paged: a.paged + t.paged,
    })
}

/// What the room rule reads of a family's file: its byte plan at a cache
/// size, the check that prices another cache size, and the refusal the rule
/// words. Derived from what the rule calls: a family's checked file answers
/// its bytes with the checkpoints priced, and the file itself answers the
/// check of another cache size.
pub trait SessionBytes {
    /// The family's checked file: what its check returned.
    type Checked;

    /// The family's refusals and failures, as its session words them.
    type Error;

    /// The family's check of this file at a cache of `rows` rows: the
    /// checked file when it passes, `None` when the check refuses it.
    fn check_at(&self, rows: usize) -> Option<Self::Checked>;

    /// The bytes a session with checkpoints every `every` positions makes
    /// buffers of, every class it holds, the copies among them.
    fn bytes_at(checked: &Self::Checked, every: u32) -> Result<Bytes, Self::Error>;

    /// The checked file's cache rows.
    fn ctx(checked: &Self::Checked) -> usize;

    /// The refusal of the rule's wording `detail`, as the family's session
    /// words its refusals.
    fn refuse_room(detail: String) -> Self::Error;
}

/// The open's room rule: the bytes the session asks of the driver — every
/// class [`Bytes`] counts, the open's buffers and the copies it makes later
/// with checkpoints every `every` positions, the requested total — held to
/// `free`, the device's free bytes as the census reads them (the working set
/// less this process's allocation and the host's available memory, the
/// smaller). The requested side is the ask itself; the paged figure is an
/// upper bound the driver's own count can fall short of, and a rule that read
/// it would refuse loads the driver holds. A load that passes the room is
/// refused by name, with the largest cache of rows that fits taken from the
/// byte formula itself, before any buffer is made.
pub fn room<F: SessionBytes>(
    file: &F,
    checked: &F::Checked,
    every: u32,
    free: u64,
) -> Result<(), F::Error> {
    let bytes = F::bytes_at(checked, every)?;
    if fits_room(&bytes, free) {
        return Ok(());
    }
    let ctx = F::ctx(checked);
    Err(F::refuse_room(room_words(
        &bytes,
        free,
        largest_ctx_of(file, ctx, every, free),
    )))
}

/// Whether a session of `bytes` fits `free` bytes of the device's room: the
/// requested total of every class it holds, the copies among them, against
/// the free bytes. The room rule's one comparison; [`room`] words its refusal.
#[must_use]
pub fn fits_room(bytes: &Bytes, free: u64) -> bool {
    bytes.held().requested <= free
}

/// The cache rows of an open that asked for none, at most `cap`: `cap` when
/// its session fits `free` bytes of the device's room (checkpoints at `every`
/// positions), else the largest cache below it that fits. `cap` again when
/// even one row does not fit, or the file has another refusal, so the open
/// refuses it by name.
#[must_use]
pub fn unasked_ctx<F: SessionBytes>(file: &F, cap: usize, every: u32, free: u64) -> usize {
    largest_ctx_of(file, cap, every, free).unwrap_or(cap)
}

/// The words of a session that asks `bytes` of the driver against `free`
/// bytes of room: the requested total, the room, and the largest cache of
/// rows that fits when [`largest_ctx_of`] found one.
#[must_use]
pub fn room_words(bytes: &Bytes, free: u64, largest_ctx: Option<usize>) -> String {
    let asked = bytes.held().requested;
    match largest_ctx {
        Some(rows) => format!(
            "the load asks {asked} bytes of the driver and the device's free bytes are {free}; a \
             cache of {rows} rows fits"
        ),
        None => format!(
            "the load asks {asked} bytes of the driver and the device's free bytes are {free}"
        ),
    }
}

/// The largest cache of at most `ctx` rows whose session fits `free` bytes:
/// the byte formula itself at each candidate, exact because every class that
/// scales with the cache's rows never shrinks as the rows grow. `None` when
/// even one row does not fit.
#[must_use]
pub fn largest_ctx_of<F: SessionBytes>(
    file: &F,
    ctx: usize,
    every: u32,
    free: u64,
) -> Option<usize> {
    let fits = |rows: usize| {
        file.check_at(rows)
            .is_some_and(|checked| F::bytes_at(&checked, every).is_ok_and(|b| fits_room(&b, free)))
    };
    if !fits(1) {
        return None;
    }
    let (mut lo, mut hi) = (1, ctx);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}
