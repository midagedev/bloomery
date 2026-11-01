//! The production taps of a Qwen3.6 body: the residual rows entering the
//! layers a block draft reads (HF's `hidden_states[L]`; entering layer 0 is
//! the embedding row), laid into one device buffer by every walk of the
//! whole-card load — the decode step, a pass of rows, each prompt unit —
//! and read back by the session ([`GpuModel::draft_taps`]) or, for a prompt
//! call, after each unit that holds a position of its window
//! ([`GpuModel::prefill_tapped`]).
//!
//! They are not the gate instrument ([`GpuModel::set_layer_taps`], every
//! layer's output of the decode chain, eager): the two are independent, a
//! step runs with both set, the instrument keeps refusing a prompt unit by
//! name, and these taps refuse a placed load and a pass of several slots.
//!
//! The buffer and its layout are the common [`DraftTaps`]'s, here with the
//! ubatch arena's rows a tap. A walk lays a tapped layer's rows in between
//! its mixer and its FFN half (`Program::walk_with`, over the chain walk's
//! `Walk::run_between`).

pub(super) use crate::draft_taps::DraftTaps;
pub use crate::draft_taps::TapRowsSink;

use super::body35::Body35;
use super::prefill::PrefillPath;
use crate::model::GpuModel;
use crate::{GpuError, launch_u32};
use runtime::prompt::CallEnd;

const WHAT: &str = "qwen35moe::draft_taps";

impl GpuModel<Body35> {
    /// Tap the rows entering `layers` (strictly rising, each below the
    /// layer count; empty stops tapping) from the next walk on: every
    /// walk — a step, a pass of rows, each prompt unit — lays them into a
    /// buffer of the ubatch arena's rows a layer, allocated here against the
    /// card's free bytes with [`FIT_RESERVE`](crate::model::fit::FIT_RESERVE)
    /// left. The captured chains are dropped first (a capture records the
    /// copies into the buffer of its time), and the next graph step or pass
    /// recaptures with the copies. A
    /// placed load, whose walks run through the host tier's legs, is
    /// refused by name; so is a pass of several slots while taps are set. A
    /// refused call leaves taps off. Load-time allocation.
    pub fn set_draft_taps(&mut self, layers: &[usize]) -> Result<(), GpuError> {
        const WHAT_S: &str = "qwen35moe::set_draft_taps";
        self.drop_captures();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        body.draft = None;
        if layers.is_empty() {
            return Ok(());
        }
        if body.placed.is_some() {
            return Err(GpuError::shape(
                WHAT_S,
                "taps on a placed load: its walks run through the host tier's legs, which lay no \
                 rows in",
            ));
        }
        let (hidden, rows_cap) = (body.s.dims.hidden, body.u.rows.max(body.a.rows));
        let free = gpu.alloc_room()?;
        body.draft = Some(DraftTaps::new(
            gpu.stream(),
            layers,
            body.plans.len(),
            (hidden, rows_cap),
            free,
        )?);
        Ok(())
    }

    /// Floats one position's tapped row holds: the tapped layers times the
    /// body's width; 0 with no taps set.
    #[must_use]
    pub fn draft_tap_width(&self) -> usize {
        self.body(WHAT)
            .map_or(0, |b| b.draft.as_ref().map_or(0, DraftTaps::width))
    }

    /// The first `rows` rows of the last walk — a step's one, a verify's
    /// `m`, a prompt unit's — one [`GpuModel::draft_tap_width`] row each.
    /// Blocking. Refused by name with no taps set, and for more rows than
    /// the last walk laid in.
    pub fn draft_taps(&mut self, rows: usize) -> Result<Vec<f32>, GpuError> {
        self.draft_tap_rows(0, rows)
    }

    fn draft_tap_rows(&mut self, from: usize, rows: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT_T: &str = "qwen35moe::draft_taps";
        let (gpu, _, body) = self.body_parts(WHAT_T)?;
        body.draft
            .as_ref()
            .ok_or(GpuError::state(WHAT_T, "taps set (set_draft_taps)"))?
            .read(gpu.stream(), from, rows)
    }

    /// [`GpuModel::prefill_with`] with the tapped rows of the call's last
    /// `window` positions (all of them when the call is shorter) handed to
    /// `sink` as `(first position, rows)`, one [`GpuModel::draft_tap_width`]
    /// row a position, after each unit that holds any of them — in order,
    /// no position twice or missed. The units are the plain call's, so its
    /// cache, state and argmax are bit for bit. Refused by name with no taps
    /// set; a sink that fails ends the call with its error.
    pub fn prefill_tapped(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
        window: usize,
        sink: &mut TapRowsSink<'_>,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen35moe::prefill_tapped";
        if self.draft_tap_width() == 0 {
            return Err(GpuError::state(WHAT_P, "taps set (set_draft_taps)"));
        }
        let end = self.pos() + launch_u32(WHAT_P, "tokens", tokens.len())?;
        let held = end - launch_u32(WHAT_P, "window", window.min(tokens.len()))?;
        let mut unit = |m: &mut GpuModel<Body35>, first: u32, rows: usize| {
            let past = first + launch_u32(WHAT_P, "rows", rows)?;
            if past <= held {
                return Ok(());
            }
            let from = first.max(held);
            let skip = (from - first) as usize;
            let rows = m.draft_tap_rows(skip, rows - skip)?;
            sink(from, &rows)
        };
        let next = self.prefill_hooked(tokens, path, Some(&mut unit), CallEnd::Argmax)?;
        next.ok_or(GpuError::state(WHAT_P, "a token read after the last unit"))
    }
}
