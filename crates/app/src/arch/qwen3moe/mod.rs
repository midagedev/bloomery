//! The Qwen3.8-Flash-Next (qwen4exp) session: [`Body38`] behind the
//! session's traits, and its [`MtpBody`](crate::mtp::MtpBody) over the
//! draft layer [`Mtp38`](bloomery_gpu::arch::qwen3moe::Mtp38) carries on
//! the target's own card, which the shared window
//! [`MtpDraft`](crate::mtp::MtpDraft) drives in windows of [`Drafted38`]'s
//! rows.
//!
//! The MTP layer's row at position `q` reads the token at `q` and the
//! target's hidden row at `q − 1` (ik's pairing, the `mtp-qwen4exp` set's
//! shift 1), and predicts the token at `q + 1`.

use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38, Qwen38Model, TargetRows};
use bloomery_gpu::model::StepMode;
use runtime::seqstate::Kept;
use runtime::{Speculative, Tapped};

use crate::mtp::{MtpBody, MtpDraft};
use crate::{Keep, Prompt, Session, SessionError};
mod mtp;

const WHAT: &str = "qwen4exp session";

/// How the session feeds a prompt and runs the draft's walks.
#[derive(Clone, Debug)]
pub struct Q38Cfg {
    /// The prompt path (`--prefill`): `auto` the default.
    pub prompt: Prompt38,
    /// The draft's walks with a head and its chains captured (`graph`) or
    /// enqueued launch by launch (`eager`); the prompt's warmup walks the
    /// store alone either way.
    pub draft: StepMode,
}

impl Prompt for Body38 {
    /// The prompt call by the configured path — `auto` resolving by the
    /// prompt's length — the argmax after the last id.
    fn prompt(m: &mut Qwen38Model, ids: &[u32]) -> Result<u32, GpuError> {
        m.prompt38(ids, Prompt38::Auto)
    }
}

impl Keep for Body38 {
    /// [`Body38::kept`]'s position: every position, the empty model, the
    /// nearest checkpoint at or below `n` a prompt call took, or a waiting
    /// verify's rows.
    fn keepable(m: &Qwen38Model, n: u32) -> u32 {
        <Body38 as Keep>::kept(m, n).at
    }

    /// [`Body38::kept`]: what a cut keeps and why — a checkpoint's position,
    /// or nothing with the reason a delta layer holds no earlier state.
    fn kept(m: &Qwen38Model, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// Back to empty at 0 — a reset, the residency where use has taken it;
    /// else the body's commit ([`Rollback`](bloomery_gpu::model::Rollback)):
    /// a waiting verify's kept rows, or the checkpoint at `n`, any other
    /// position refused by name.
    fn cut(m: &mut Qwen38Model, n: u32) -> Result<(), GpuError> {
        match n {
            0 => m.reset(),
            n => m.rollback(n),
        }
    }
}

impl Tapped for Session<Body38> {
    fn tap_width(&self) -> usize {
        self.model().body(WHAT).map_or(0, Body38::mtp_tap_width)
    }

    /// The last call's first `rows` final hidden rows ([`Body38::target_streams`]
    /// of the arena it walked), four streams of the model's width a row,
    /// kept for the borrow. Blocking.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        let walk = self.model().body(WHAT)?.last_walk();
        let v = self.taps_at(walk, rows)?;
        self.tapped = v;
        Ok(&self.tapped)
    }
}

impl Session<Body38> {
    /// The last call's first `rows` final hidden rows of the arena `walk`
    /// ([`Body38::target_streams`]). Blocking; gate use.
    pub fn taps_at(&mut self, walk: TargetRows, rows: usize) -> Result<Vec<f32>, SessionError> {
        let (gpu, _, body) = self.model_mut().body_parts(WHAT)?;
        Ok(body.target_streams(gpu, walk, rows)?)
    }
}

/// A generation's windows over the draft: the speculative advance a caller
/// drives a [`Session<Body38>`] with, `M` = the draft's
/// [`MtpBody::VERIFY_ROWS`].
pub type Drafted38 = Speculative<MtpDraft<Body38>, { <Body38 as MtpBody>::VERIFY_ROWS }>;
