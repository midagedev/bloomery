//! The Qwen3.8-Flash-Next (qwen4exp) session: [`Body38`] behind the
//! session's traits, and its [`MtpBody`](crate::mtp::MtpBody) over the
//! draft layer [`Mtp38`](bloomery_gpu::arch::qwen3moe::Mtp38) carries on
//! the target's own card, which the shared window
//! [`MtpDraft`](crate::mtp::MtpDraft) drives in windows of [`Drafted38`]'s
//! rows.
//!
//! The plain attention body of a qwen3moe file ([`Body`]) sits behind the
//! same traits: its prompt schedule ([`GEMM_FROM`]; wide units on a placed
//! load) and its keep rule (every held position — the caches are
//! per-position). So does the Qwen3.6 body
//! of a qwen35moe file ([`Body35`]): the same schedule's wide half
//! (`PrefillPath::Wide`) and the checkpoints' keep rule — the nearest
//! checkpoint a marked prompt call took at or below the cut. Its in-file
//! next-token layer is [`mtp35`]'s [`MtpBody`](crate::mtp::MtpBody), one id
//! a proposal, driven by the shared window.
//!
//! A Qwen3.6 body opened with media
//! ([`Open35::media`](bloomery_gpu::arch::qwen3moe::Open35::media)) is a
//! [`MediaBody`](bloomery_gpu::media::MediaBody), so its session takes
//! images through the common [`Session::prompt_media`]; no entry of its own.
//!
//! The MTP layer's row at position `q` reads the token at `q` and the
//! target's hidden row at `q − 1` (ik's pairing, the `mtp-qwen4exp` set's
//! shift 1), and predicts the token at `q + 1`.

use bloomery_gpu::GpuError;
use bloomery_gpu::Qwen3moeModel;
use bloomery_gpu::arch::qwen3moe::{
    Body, Body35, Body38, PrefillPath, Prompt38, Qwen35moeModel, Qwen38Model, TargetRows,
};
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::model::StepMode;
use bloomery_gpu::route_softmax::MAX_TOKENS;
use runtime::seqstate::Kept;
use runtime::{Speculative, TapSink, Tapped, TappedPrompt, Want};

use crate::mtp::{MtpBody, MtpDraft};
use crate::{Keep, Prompt, Session, SessionError, cut_positions, keep_positions};
mod mtp;
mod mtp35;
pub use mtp35::Prompt35;

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
        cut_positions(m, n)
    }
}

/// The fewest rows a qwen3moe prompt call runs as the ubatch walk: one past
/// a pass. A call this long and longer writes each row the bits the walk
/// writes wherever the call is cut — `Body`'s GEMM because a token's values
/// do not depend on the ubatch it lands in, `Body35`'s
/// [`Wide`](PrefillPath::Wide) because it runs no unit small enough to fall
/// to the gemv arm — and a shorter call runs passes, bit-equal to one step
/// a row. So a request that keeps a prefix of what the model holds leaves a
/// call whose rows are a whole fresh run's as long as it holds at least
/// this many ids.
pub const GEMM_FROM: usize = MAX_TOKENS + 1;

impl Prompt for Body {
    /// The GEMM ubatch walk from [`GEMM_FROM`] rows on, passes below (the
    /// `Auto` of a prompt shorter than one ubatch, resolved without the
    /// tail pass): the argmax after the last id. A placed load has no GEMM
    /// walk (`prefill_plan` refuses it by name): its prompt runs the wide
    /// units through the host tier's batch port from [`GEMM_FROM`] rows on, as
    /// [`Body35`]'s does, and passes below.
    fn prompt(m: &mut Qwen3moeModel, ids: &[u32]) -> Result<u32, GpuError> {
        let placed = m.body("app::Prompt for Body")?.placed().is_some();
        let path = match (ids.len() >= GEMM_FROM, placed) {
            (false, _) => PrefillPath::Pass,
            (true, false) => PrefillPath::Gemm,
            (true, true) => PrefillPath::Wide,
        };
        m.prefill_with(ids, path)
    }
}

impl Keep for Body {
    /// Every held position: the caches are per-position, and
    /// [`Rollback`](bloomery_gpu::model::Rollback) takes a position back
    /// without device work.
    fn keepable(m: &Qwen3moeModel, n: u32) -> u32 {
        keep_positions(m, n)
    }

    /// Back to empty at 0 — a reset; else the model's rollback
    /// ([`Rollback`](bloomery_gpu::model::Rollback)), which the caches take
    /// as given.
    fn cut(m: &mut Qwen3moeModel, n: u32) -> Result<(), GpuError> {
        cut_positions(m, n)
    }
}

impl Prompt for Body35 {
    /// The wide ubatch walk from [`GEMM_FROM`] rows on — a call this long
    /// and longer runs no unit under the gemv arm's cut, so its rows are
    /// the same bits wherever the call is cut — and passes below, bit-equal
    /// to one step a row. The model takes the checkpoints of the call's
    /// marks only while a caller armed them
    /// ([`Body35::set_checkpoints`](bloomery_gpu::arch::qwen3moe::Body35)).
    /// The call is one residency pass ([`prompt_pass`]): a load with a machine takes
    /// its boundary before the first unit, none inside.
    fn prompt(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<u32, GpuError> {
        prompt_pass(m, |m| m.prefill_with(ids, path35(ids.len())))
    }
}

/// The units a Qwen3.6 prompt call of `n` ids runs as: the wide ubatch walk
/// from [`GEMM_FROM`] ids on, passes below. The drafted prompt uses the same
/// choice ([`mtp35`]): the window stores one path at open, and this one
/// depends on the length.
pub(super) fn path35(n: usize) -> PrefillPath {
    if n >= GEMM_FROM {
        PrefillPath::Wide
    } else {
        PrefillPath::Pass
    }
}

impl Keep for Body35 {
    /// [`Body35::kept`]'s position: every position, the empty model, or the
    /// nearest checkpoint a marked prompt call took at or below `n` — and,
    /// on a load that verifies drafted rows, any position past the first of
    /// the verify waiting for its commit.
    fn keepable(m: &Qwen35moeModel, n: u32) -> u32 {
        <Body35 as Keep>::kept(m, n).at
    }

    /// [`Body35::kept`]: what a cut keeps and why — a checkpoint's
    /// position, a waiting verify's row (the commit moves the lane word and
    /// copies nothing), or nothing with the reason a delta layer holds no
    /// earlier state but a copy.
    fn kept(m: &Qwen35moeModel, n: u32) -> Kept {
        m.body(WHAT)
            .map_or_else(|_| Kept::rule(n, m.pos(), 0), |b| b.kept(n, m.pos()))
    }

    /// Back to empty at 0 — a reset; else the model's rollback
    /// ([`Rollback`](bloomery_gpu::model::Rollback)): the checkpoint at
    /// `n`, a row of the verify waiting on a load of several lanes (its
    /// commit), any other position refused by name.
    fn cut(m: &mut Qwen35moeModel, n: u32) -> Result<(), GpuError> {
        cut_positions(m, n)
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

impl Tapped for Session<Body35> {
    /// The tapped layers ([`TappedPrompt::set_taps`]) times the model's
    /// width; 0 with no taps set.
    fn tap_width(&self) -> usize {
        self.model().draft_tap_width()
    }

    /// The last call's first `rows` positions' rows entering the tapped
    /// layers, in the order they were named: a verify's rows, or a step's
    /// one. Blocking.
    /// Refused by name for more rows than the last walk laid in.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        self.tapped = self.model_mut().draft_taps(rows)?;
        Ok(&self.tapped)
    }
}

impl TappedPrompt for Session<Body35> {
    /// Tap the rows entering `layers` from the next call on, in every walk
    /// of the call: the step, a verify's pass, each prompt unit. The
    /// captured chains are dropped and recaptured with the copies at their
    /// next graph run. Empty stops tapping. Refused by name on a placed
    /// load, past the layer count, and for a card without the bytes.
    fn set_taps(&mut self, layers: &[usize]) -> Result<(), SessionError> {
        self.idle("set_taps")?;
        Ok(self.model_mut().set_draft_taps(layers)?)
    }

    /// [`Target::prompt`]'s units (`Prompt for Body35`) with the tapped
    /// rows of the call's last `window` positions handed to `sink` after
    /// each unit that holds any of them: the cache, the recurrent state and
    /// the argmax are the plain call's, bit for bit.
    fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: TapSink<'_, SessionError>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt_tapped")?;
        let _busy = self.model.prompt_busy();
        let mut refused = None;
        let ran = prompt_pass(&mut self.model, |m| {
            m.prefill_tapped(ids, path35(ids.len()), window, &mut |first, rows| {
                sink(first, rows).map_err(|e| {
                    refused = Some(e);
                    GpuError::State {
                        what: WHAT,
                        missing: "a tap sink that takes the rows (its error follows)",
                    }
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
