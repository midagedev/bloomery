//! The Raon-1-mini session: [`Body`] behind the session's traits — its
//! prompt as passes of rows ([`GpuModel::prompt_rows`]) and the call's
//! checkpoints, its keep rule, and the taps a block draft reads
//! ([`GpuModel::set_draft_taps`]), so the common drafted path
//! ([`crate::dspark::CardDspark`]) drives it. The body lives in the gpu crate's
//! own raon1 arch, so it needs no feature.
//!
//! An RDA layer's state and conv ring hold the position they stand at alone, so
//! a cut keeps every held position, the empty model, or a checkpoint: each
//! prompt call takes one at its start and at its end ([`Body::kept`]), which a
//! request that extends the last prompt returns to.
//!
//! A session that verifies drafted rows opens its model with
//! [`GpuModel::open_verifying`], whose RDA stores hold a lane a verify row.

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::arch::raon1::Body;
use bloomery_gpu::checkpoint::prompt_marked;
use bloomery_gpu::host::served::prompt_pass;
use runtime::seqstate::Kept;
use runtime::{TapSink, Tapped, TappedPrompt, Want};

use super::cut_checkpointed;
use crate::{Keep, Prompt, Session, SessionError};

const WHAT: &str = "raon1 session";

impl Prompt for Body {
    /// Passes of up to the verify's rows, each row a step's bits, the head
    /// on the last ([`GpuModel::prompt_rows`]), and the call's checkpoints
    /// ([`prompt_marked`]). The call is one residency pass ([`prompt_pass`]).
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        prompt_marked(m, ids, |m, seg| prompt_pass(m, |m| m.prompt_rows(seg)))
    }
}

impl Keep for Body {
    /// [`Body::kept`]: every position, the empty model, or the nearest
    /// checkpoint at or below `n`.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        <Body as Keep>::kept(m, n).at
    }

    /// [`Body::kept`].
    fn kept(m: &GpuModel<Body>, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// [`cut_checkpointed`]: the checkpoints are the only positions a cut
    /// takes back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_checkpointed(m, n)
    }
}

impl Tapped for Session<Body> {
    /// The tapped layers ([`TappedPrompt::set_taps`]) times the streams'
    /// width; 0 with no taps set.
    fn tap_width(&self) -> usize {
        self.model().draft_tap_width()
    }

    /// The last call's first `rows` positions' rows entering the tapped
    /// layers, in the order they were named: a verify's rows, or a step's
    /// one. Blocking. Refused by name for more rows than the last walk laid
    /// in.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        self.tapped = self.model_mut().draft_taps(rows)?;
        Ok(&self.tapped)
    }
}

impl TappedPrompt for Session<Body> {
    /// Tap the rows entering `layers` from the next call on, in every walk:
    /// the step, a verify's pass, each prompt chunk. Refused by name past
    /// the layer count and for a card without the bytes.
    fn set_taps(&mut self, layers: &[usize]) -> Result<(), SessionError> {
        self.idle("set_taps")?;
        Ok(self.model_mut().set_draft_taps(layers)?)
    }

    /// [`Target::prompt`](runtime::Target::prompt)'s passes with the tapped
    /// rows of the call's last `window` positions handed to `sink` after
    /// each group that holds any of them: the stores and the argmax are the
    /// plain call's, bit for bit, and so are its checkpoints. The body spaces no
    /// inner mark, so the call is one segment and `window` counts from its end.
    fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: TapSink<'_, SessionError>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt_tapped")?;
        let _busy = self.model.prompt_busy();
        let mut refused = None;
        let ran = prompt_marked(&mut self.model, ids, |m, seg| {
            prompt_pass(m, |m| {
                m.prompt_rows_tapped(seg, window, &mut |first, rows| {
                    sink(first, rows).map_err(|e| {
                        refused = Some(e);
                        GpuError::State {
                            what: WHAT,
                            missing: "a tap sink that takes the rows (its error follows)",
                        }
                    })
                })
            })
        });
        // The sink's own error first: the call's echo of it is a placeholder.
        if let Some(e) = refused {
            return Err(e);
        }
        Ok(self.read(ran?, Want::Argmax)?.argmax())
    }
}
