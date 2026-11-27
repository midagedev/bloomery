//! The recall draft's one constructor, for the seats whose body verifies eight
//! rows (the qwen3 seat's `Body35` and the Inkling 2 seat). It sits in a file
//! of its own, which only the binary that serves those seats includes: the
//! draft's window is `RECALL_ROWS` rows, a body whose pass holds fewer does
//! not compile it ([`runtime::Draft::FITS`]), and the per-model binaries of
//! the V4.1 and qwen38 seats include the table and the served draft without
//! this.

use app::{Keep, Prompt, RowsLog, Session};
use bloomery_gpu::model::{Rollback, Rows, Slots};
use bloomery_gpu_gates::GateError;
use runtime::Recall;
use runtime::width::Mode as WidthMode;

use super::drafted::{History, HistoryDrafts};
use super::served::ServedDraft;

/// The rows a recall pass verifies on a body whose pass holds eight rows
/// ([`runtime::Verify::MAX_ROWS`]): the token at the target's position and the
/// recall draft's seven proposed ids. A seat names this pair beside the
/// body's own limit, so a body of fewer rows fails to compile where it
/// would drive it ([`runtime::Draft::FITS`]), and the draft word is refused there by
/// name.
pub(super) const RECALL_ROWS: usize = 8;

/// The most ids a recall proposal holds on a body of [`RECALL_ROWS`] rows.
pub(super) const RECALL_WIDTH: usize = RECALL_ROWS - 1;

impl<B> ServedDraft<B>
where
    B: Prompt + Keep + Slots + 'static,
    B::Seq: 'static,
{
    /// The recall draft on each of `slots` resident slots of `s` (its ladder
    /// the served one, [`RECALL_ROWS`] rows a pass), as
    /// [`ServedDraft::lookup`] opens the lookup: each behind a width chooser
    /// of `mode`, each served slot's verify pass captured now (`log` told of
    /// each); `history` the body's token history, where it keeps one. A seat
    /// whose body's pass holds fewer rows refuses the draft word by name
    /// instead.
    pub fn recall(
        s: &mut Session<B>,
        slots: usize,
        mode: WidthMode,
        log: &mut impl RowsLog,
        history: Option<History<Session<B>>>,
    ) -> Result<Self, GateError>
    where
        B: Rows + Rollback,
    {
        let table = HistoryDrafts::open::<Recall<RECALL_WIDTH>, RECALL_ROWS>(
            s,
            slots,
            mode,
            log,
            history,
            Recall::served,
            Recall::<RECALL_WIDTH>::RULE,
        )?;
        Ok(Self::of_history(table))
    }
}
