//! The Inkling 2 session: [`Body`] behind the session's traits — its prompt as the rows passes
//! ([`GpuModel::prompt_pass`]: the stores left where the ids' steps leave them, the head's argmax
//! after the last row the call's; a load with no pass steps its ids) and its keep rule (the positions its rollback takes back). The
//! whole-card body opens by its own constructor ([`GpuModel::open`]), so the session plans
//! nothing. The body lives in the gpu crate's own inkling2 arch, so it needs no feature.
//!
//! A block draft reads the rows entering the visits its file names and the final normed row
//! ([`Tapped`], [`TappedPrompt`]): the seat attaches the body's draft taps at the open, before
//! the first capture, and the session hands the walks' rows back.

use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::{GpuError, GpuModel};

use crate::{Keep, Prompt, Session, SessionError, cut_positions, keep_down_to};
use runtime::{TapSink, Tapped, TappedPrompt};

const WHAT: &str = "inkling2 session";

impl Prompt for Body {
    /// The ids as passes of rows ([`GpuModel::prompt_pass`]), one readback after the last; a load
    /// with no pass (a placed one the card had no room for the pass's buffers beside) steps the
    /// ids one at a time.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        if m.body(WHAT)?.pass_rows() == 0 {
            return m.step(ids);
        }
        m.prompt_pass(ids)
    }
}

impl Keep for Body {
    /// Every store is positional, so a cut keeps `n` down to [`Body::rollback_floor`], the written
    /// end less the conv ring's reach, and nothing below it ([`keep_down_to`]).
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        match m.body(WHAT) {
            Ok(body) => keep_down_to(m, n, body.rollback_floor()),
            Err(_) => 0,
        }
    }

    /// Back to empty at 0 — a reset; else the model's rollback, which the body refuses by name
    /// below its floor.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_positions(m, n)
    }
}

impl Tapped for Session<Body> {
    /// The tapped visits ([`TappedPrompt::set_taps`]) times the stream's width; 0 with no taps
    /// set.
    fn tap_width(&self) -> usize {
        self.model().draft_tap_width().unwrap_or(0)
    }

    /// The last walk's first `rows` positions' rows entering the tapped visits, then the final
    /// normed row, in the order the taps were named: a verify's rows, a prompt pass's, a step's
    /// one. Blocking. Refused by name for more rows than the last walk laid in.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        self.tapped = self.model_mut().draft_taps_rows(rows)?;
        Ok(&self.tapped)
    }
}

impl TappedPrompt for Session<Body> {
    /// The taps are attached at the open for the draft's visits (`set_draft_taps`), before the
    /// target's first capture, since setting them drops the captures; a different set is refused
    /// by name.
    fn set_taps(&mut self, visits: &[usize]) -> Result<(), SessionError> {
        match self.model().draft_layers()? {
            Some(held) if held == visits => Ok(()),
            Some(held) => Err(SessionError::Refused(format!(
                "{WHAT}: the draft's taps read the visits {held:?}, not {visits:?}"
            ))),
            None => Err(SessionError::Refused(format!(
                "{WHAT}: no draft taps: the open attaches them before the target's first capture"
            ))),
        }
    }

    /// [`Target::prompt`](runtime::Target::prompt)'s passes with the tapped rows of each pass's
    /// last `window` positions handed to `sink` (the first position they hold, then one row a
    /// position) as the pass ends: the stores and the argmax are the plain call's, bit for bit.
    /// A sink's error ends the call and comes back as it was.
    fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: TapSink<'_, SessionError>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt_tapped")?;
        let _busy = self.model.prompt_busy();
        let mut refused = None;
        let ran = self
            .model_mut()
            .prompt_pass_tapped(ids, window, &mut |first, rows| {
                sink(first, rows).map_err(|e| {
                    refused = Some(e);
                    GpuError::State {
                        what: WHAT,
                        missing: "a tap sink that takes the rows (its error follows)",
                    }
                })
            });
        // The sink's own error first: the call's echo of it is a placeholder.
        if let Some(e) = refused {
            return Err(e);
        }
        Ok(ran?)
    }
}
