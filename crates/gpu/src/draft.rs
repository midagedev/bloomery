//! The types a draft walk is given: how it runs ([`WalkMode`]), which
//! projection its head runs ([`DraftHead`]), where its hidden rows come from
//! ([`Hidden`]), the rows it walks ([`Feed`], [`DraftFeed`]), the positions
//! its target arenas hold ([`Held`]) and the names of its row bounds
//! ([`WalkBounds`]). Each draft program — Qwen3.8's MTP layer and GLM's
//! NextN layer — walks these.

use crate::GpuError;
use crate::media::CallRows;
use crate::model::StepMode;

/// Copy one final hidden row to row 0 of a sequence-owned arena, ordered on
/// `stream`. The caller validates the source's position record first.
pub fn retain_hidden_row(
    stream: &cuda_core::CudaStream,
    source: &cuda_core::DeviceBuffer<f32>,
    first: usize,
    width: usize,
    target: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let byte_off = first
        .checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| GpuError::shape("draft::retain_hidden_row", "hidden row offset overflow"))?;
    let source = crate::tensor::Window::<f32>::of(source, byte_off, width)?;
    let mut target = crate::tensor::WindowMut::<f32>::of_mut(target, 0, width)?;
    target.copy_from_device_async(stream, &source)?;
    Ok(())
}

/// How a walk runs: the program's own walk, or its chain of walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkMode {
    /// Enqueued launch by launch.
    Eager,
    /// Replayed from its capture, captured on first use.
    Graph,
    /// Enqueued launch by launch through the store's append and no further:
    /// the rows' keys and values, which are all a later row reads of them (a
    /// row's input is its token and the target's hidden row, never the
    /// draft's output). No flash, feed-forward block or head, no readback
    /// and no own row: the prompt's warmup. The store's bits are a
    /// [`WalkMode::Eager`] walk's. A body with no such walk refuses it by
    /// name; a chain never takes it.
    Store,
}

impl From<StepMode> for WalkMode {
    /// A step's mode as a walk's: the draft's walks with a head run as the
    /// target's steps do.
    fn from(m: StepMode) -> WalkMode {
        match m {
            StepMode::Eager => WalkMode::Eager,
            StepMode::Graph => WalkMode::Graph,
        }
    }
}

/// Which projection a draft's head runs. Whether the load opened a row list
/// is the load's fact; a program that opens none holds [`DraftHead::Full`]
/// alone and refuses the other by name ([`DraftHead::check_opened`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftHead {
    /// The target's `output`: every token of the vocabulary.
    Full,
    /// The load's row list: the target's `output` rows of its ids.
    Rows,
}

impl DraftHead {
    /// Refused by name: the row-list head on a load that opened the full head
    /// alone (`opened_rows` false).
    ///
    /// # Errors
    /// [`GpuError::State`] naming `what` for [`DraftHead::Rows`] when the load
    /// opened no row list.
    pub fn check_opened(self, what: &'static str, opened_rows: bool) -> Result<(), GpuError> {
        if self == DraftHead::Rows && !opened_rows {
            return Err(GpuError::state(
                what,
                "a row-list head: this draft was opened with the full head",
            ));
        }
        Ok(())
    }
}

/// Where a walk's hidden rows come from; `A` is the arena kind the caller's
/// body names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hidden<'a, A> {
    /// Rows written from the host, as many values a row as the caller's body
    /// says: the row at position 0 reads a zero row, the target holding
    /// nothing before it.
    Host(&'a [f32]),
    /// The target's final hidden rows in the arena `walk`, from its row
    /// `first` on, as the target's last call of that arena left them.
    Target { walk: A, first: usize },
}

/// A walk's rows: `tokens` at positions `pos0 ..`, each beside its hidden
/// row (the target's at the position before it). `media` is the prompt
/// call's span rows by absolute position: a row the lookup holds replaces
/// the token's embedding row at its position, as the target's media call
/// took it. Only a prompt call that holds an image carries it; a body whose
/// walk cannot take it refuses it by name.
#[derive(Clone, Copy, Debug)]
pub struct Feed<'a, A> {
    pub tokens: &'a [u32],
    pub pos0: u32,
    pub hidden: Hidden<'a, A>,
    pub media: Option<&'a CallRows<'a>>,
}

/// What a draft program is fed one walk: the rows of a [`Feed`], or the
/// draft's own last row.
#[derive(Clone, Copy, Debug)]
pub enum DraftFeed<'a, A> {
    /// `tokens` at positions `pos0 ..`, each beside its hidden row.
    Rows(Feed<'a, A>),
    /// One row at `pos0`: the last walk's last row's token and streams. A
    /// program whose proposal holds one id has no such walk and refuses it by
    /// name ([`DraftFeed::target_rows`]).
    Own { pos0: u32 },
}

impl<'a, A> DraftFeed<'a, A> {
    /// The position of the feed's first row.
    #[must_use]
    pub fn pos0(&self) -> u32 {
        match self {
            DraftFeed::Rows(f) => f.pos0,
            DraftFeed::Own { pos0 } => *pos0,
        }
    }

    /// How many rows the feed walks: its tokens, or the one own row.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            DraftFeed::Rows(f) => f.tokens.len(),
            DraftFeed::Own { .. } => 1,
        }
    }

    /// The feed's rows, for a program that walks the target's rows alone: the
    /// own row is refused by name ([`refuse_own`]).
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming `what` for [`DraftFeed::Own`].
    pub fn target_rows(self, what: &'static str) -> Result<Feed<'a, A>, GpuError> {
        match self {
            DraftFeed::Rows(f) => Ok(f),
            DraftFeed::Own { pos0 } => Err(refuse_own(
                what,
                &format!("a walk of the draft's own row at position {pos0}"),
            )),
        }
    }
}

/// The refusal by name of the draft's own row ([`DraftFeed::Own`]) by a
/// program whose walks read the target's hidden rows alone, `detail` naming
/// the call.
#[must_use]
pub fn refuse_own(what: &'static str, detail: &str) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "{detail}: this program's walk reads the target's hidden rows and its proposal holds \
             one id, so it has no own row (DraftFeed::Own)"
        ),
    )
}

/// The names of a draft program's row bounds: the most rows one walk takes
/// in each mode. A program implements it with its own values; the names, and
/// the refusal of a walk past them ([`refuse_rows`]), are this module's.
pub trait WalkBounds {
    /// The most rows one eager walk takes.
    const WALK_ROWS: usize;
    /// The most rows a captured walk takes; 0 for a program that captures
    /// none.
    const GRAPH_ROWS: usize;
    /// The most rows one store walk ([`WalkMode::Store`]) takes.
    const STORE_ROWS: usize;
}

/// The most rows one walk of `P` takes in `mode`.
#[must_use]
pub const fn most_rows<P: WalkBounds>(mode: WalkMode) -> usize {
    match mode {
        WalkMode::Eager => P::WALK_ROWS,
        WalkMode::Graph => P::GRAPH_ROWS,
        WalkMode::Store => P::STORE_ROWS,
    }
}

/// Refused by name: a walk of `m` rows outside `1..=` [`most_rows`] of `P`'s
/// `mode`.
///
/// # Errors
/// [`GpuError::Shape`] naming `what`.
pub fn refuse_rows<P: WalkBounds>(
    what: &'static str,
    mode: WalkMode,
    m: usize,
) -> Result<(), GpuError> {
    let most = most_rows::<P>(mode);
    if !(1..=most).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("a walk of {m} rows; the {mode:?} walk takes 1..={most}"),
        ));
    }
    Ok(())
}

/// The positions a target arena's rows hold, row `r` position `first + r`:
/// what the call that last wrote the arena left there, less the positions a
/// cut took back since. No rows: the arena holds no position of the
/// sequence. A [`Hidden::Target`] feed must read the positions recorded here
/// rather than stale rows left in the arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Held {
    first: u32,
    rows: u32,
}

impl Held {
    /// No position.
    pub const NONE: Held = Held { first: 0, rows: 0 };

    /// Rows `0 .. rows` at positions `first ..`. Construction does not read
    /// rows; [`Held::holds`] refuses a read past the position width.
    #[must_use]
    pub const fn at(first: u32, rows: u32) -> Held {
        Held { first, rows }
    }

    /// The positions from `pos` on taken back.
    pub fn cut(&mut self, pos: u32) {
        self.rows = self.rows.min(pos.saturating_sub(self.first));
    }

    /// The first position and the number of rows it holds.
    #[must_use]
    pub const fn parts(self) -> (u32, u32) {
        (self.first, self.rows)
    }

    /// Whether rows `row .. row + m` hold positions `at .. at + m`.
    /// Row arithmetic must fit `usize`, and every read position must fit
    /// `u32`. An empty range may name the boundary just after the held rows.
    #[must_use]
    pub fn holds(self, row: usize, m: usize, at: u32) -> bool {
        let Some(end) = row.checked_add(m) else {
            return false;
        };
        let (Ok(row), Ok(m), Ok(end)) = (u32::try_from(row), u32::try_from(m), u32::try_from(end))
        else {
            return false;
        };
        end <= self.rows
            && self.first.checked_add(row) == Some(at)
            && (m == 0 || at.checked_add(m - 1).is_some())
    }

    /// The positions held, for a refusal's words.
    #[must_use]
    pub fn shown(self) -> String {
        match self.rows {
            0 => "no position".to_string(),
            rows => format!(
                "positions {}..{}",
                self.first,
                u64::from(self.first) + u64::from(rows)
            ),
        }
    }

    /// Check the target rows of a [`Hidden::Target`] feed before any copy.
    /// `rows` is the arena's row capacity, and `pos0` the walk's first token
    /// position: its hidden rows must start at position `pos0 - 1`.
    ///
    /// # Errors
    /// [`GpuError::Shape`] naming `what` and `walk` for rows outside the
    /// arena, position 0 (which needs a host zero row), or positions the
    /// arena does not hold, including overflowing row or position ranges.
    pub fn check_read(
        self,
        what: &'static str,
        walk: impl std::fmt::Debug,
        rows: usize,
        (first, m): (usize, usize),
        pos0: u32,
    ) -> Result<(), GpuError> {
        let Some(end) = first.checked_add(m).filter(|&end| end <= rows) else {
            return Err(GpuError::shape(
                what,
                format!("{m} rows from row {first} of the {walk:?} arena's {rows}"),
            ));
        };
        let at = pos0.checked_sub(1).ok_or_else(|| {
            GpuError::shape(
                what,
                format!(
                    "the {walk:?} arena's rows for a walk at position 0: the target \
                     holds no row before it (a host zero row is position 0's)"
                ),
            )
        })?;
        if !self.holds(first, m, at) {
            return Err(GpuError::shape(
                what,
                format!(
                    "rows {first}..{end} of the {walk:?} arena as positions {at}..{} for \
                     a walk from {pos0}: the arena holds {}",
                    u128::from(at) + m as u128,
                    self.shown()
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Held;
    use crate::GpuError;

    #[test]
    fn held_rows_respect_row_and_position_widths_and_cuts() {
        let held = Held::at(1, 1);
        assert!(held.holds(0, 1, 1));
        assert!(!held.holds(usize::MAX, 1, 0));
        assert!(!held.holds(1, usize::MAX, 2));
        assert!(held.holds(1, 0, 2));
        assert!(!held.holds(2, 0, 3));
        assert!(Held::NONE.holds(0, 0, 0));
        assert!(!Held::NONE.holds(0, 1, 0));
        assert!(Held::at(u32::MAX, 1).holds(0, 1, u32::MAX));
        assert!(!Held::at(u32::MAX, 2).holds(0, 2, u32::MAX));
        assert!(!Held::at(u32::MAX, 2).holds(1, 1, 0));
        assert!(Held::at(u32::MAX, 2).shown().contains("4294967297"));
        if let Ok(past_u32) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(!Held::at(0, u32::MAX).holds(past_u32, 0, 0));
            assert!(!Held::at(0, u32::MAX).holds(0, past_u32, 0));
        }
        let mut held = Held::at(5, 3);
        held.cut(7);
        assert_eq!(held.parts(), (5, 2));
        assert!(held.holds(0, 2, 5));
        assert!(!held.holds(2, 1, 7));
        held.cut(4);
        assert_eq!(held.parts(), (5, 0));
        held.cut(9);
        assert_eq!(held.parts(), (5, 0));
    }

    #[test]
    fn target_reads_refuse_unheld_or_overflowing_ranges_before_copy() {
        const WHAT: &str = "draft read";
        let held = Held::at(1, 1);
        let e = held
            .check_read(WHAT, "Step", 0, (0, 1), 2)
            .expect_err("a held row outside the arena");
        assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        held.check_read(WHAT, "Step", 2, (0, 1), 2)
            .expect("the held row");
        for (first, m) in [
            (0, 3),
            (usize::MAX, 1),
            (1, usize::MAX),
            (usize::MAX, usize::MAX),
        ] {
            let e = held
                .check_read(WHAT, "Step", 2, (first, m), 1)
                .expect_err("rows outside the arena");
            assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        }
        let e = held
            .check_read(WHAT, "Step", usize::MAX, (0, usize::MAX), 2)
            .expect_err("a position end past usize");
        assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
        for (held, rows, range, pos0) in [
            (held, 2, (0, 1), 0),
            (Held::NONE, 2, (0, 1), 2),
            (held, 2, (1, 1), 3),
            (Held::at(u32::MAX - 1, 3), 3, (0, 3), u32::MAX),
        ] {
            let e = held
                .check_read(WHAT, "Step", rows, range, pos0)
                .expect_err("a target read the arena cannot provide");
            assert!(matches!(e, GpuError::Shape { what: WHAT, .. }), "{e}");
            if pos0 == u32::MAX {
                assert!(e.to_string().contains("4294967297"));
            }
        }
        let mut held = Held::at(5, 2);
        held.cut(6);
        held.check_read(WHAT, "Pair", 2, (0, 1), 6)
            .expect("the kept row remains readable");
        assert!(held.check_read(WHAT, "Pair", 2, (0, 2), 6).is_err());
    }
}
