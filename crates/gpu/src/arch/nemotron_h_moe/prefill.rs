//! The prompt path: [`GpuModel::prefill_with`] feeds a prompt in passes of up
//! to [`PASS_ROWS`] ids. Each pass runs the ids as steps
//! ([`GpuModel::step`]), so the stores and the position after a prompt are
//! those of its ids run as steps.

use super::body::Body;
use crate::GpuError;
use crate::model::GpuModel;

const WHAT: &str = "nemotron_h_moe::prefill";

/// The most ids one pass of [`GpuModel::prefill`] takes.
pub const PASS_ROWS: usize = 512;

impl GpuModel<Body> {
    /// [`GpuModel::prefill_with`] in passes of [`PASS_ROWS`] ids.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        self.prefill_with(ids, PASS_ROWS)
    }

    /// Feed `ids` through the chain in passes of up to `rows` ids and return
    /// the greedy next token after the last one. Positions continue from
    /// wherever the model stands. `rows` of 0 or past [`PASS_ROWS`], and an
    /// empty `ids`, are refused by name.
    pub fn prefill_with(&mut self, ids: &[u32], rows: usize) -> Result<u32, GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        if rows == 0 || rows > PASS_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!("passes of {rows} ids; a pass holds 1..={PASS_ROWS}"),
            ));
        }
        let mut next = None;
        for pass in ids.chunks(rows) {
            next = Some(self.step(pass)?);
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
    }
}
