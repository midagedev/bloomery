//! The V4.1 session: [`Body`] behind the session's traits, planned once by its
//! placement and loaded by gpumodel's constructor ([`Body::open_placed_tiered`]),
//! with the plan's expert tier card hung under the host tier when the
//! placement names one.

mod draft;

pub use draft::{Ds41Draft, open_draft};

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu_deepseek41::body::{
    self, Body, BodyMeta, FeatureRows, FeatureSink, OpenCfg, PrefillMode, TierOpen,
};
use gguf::Split;
use model::arch::deepseek41::place::PlanInputs;
use model::placement::{Machine, Plan};
use runtime::swaprule::KeptRows;
use runtime::{Tapped, TappedPrompt, Target};

use crate::{Keep, Open, Prompt, Session, SessionError};

const WHAT: &str = "deepseek41 session";

/// How the V4.1 session opens and feeds a prompt.
#[derive(Clone, Debug)]
pub struct Ds41Cfg {
    /// The body's and the placement's levers.
    pub open: OpenCfg,
    /// The prompt schedule the open prepares for: the body's own
    /// (`BLOOMERY_PREFILL`), or steps for a caller that feeds the prompt
    /// outside the prompt call (the finite probe).
    pub feed: PrefillMode,
    /// Time each layer-batch's card work with events (`BLOOMERY_STEP_STATS`);
    /// the batch's buffers are made with the events.
    pub card_timing: bool,
}

impl Open for Body {
    type Cfg = Ds41Cfg;
    type Inputs = PlanInputs;

    fn inputs(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn layer_count(inputs: &PlanInputs) -> usize {
        inputs.model.layers
    }

    /// The plan on `machine`, refused unless its layers sit on one card: the
    /// chain runs on one.
    fn plan<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        ctx: usize,
        cfg: &Ds41Cfg,
    ) -> Result<Plan<'a>, GpuError> {
        if machine.cards.len() != 1 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the placement puts the layers on {} cards; the chain runs on one",
                    machine.cards.len()
                ),
            });
        }
        let ctx = u64::try_from(ctx).map_err(|_| GpuError::Shape {
            what: WHAT,
            detail: format!("a context of {ctx} positions passes u64"),
        })?;
        inputs
            .plan(machine, ctx, &cfg.open.place)
            .map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The stage card, card 0, and the plan's expert tier cards
    /// ([`TierOpen::of_machine`]).
    fn open(
        file: Split,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        cfg: &Ds41Cfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: cfg.open.body,
        };
        Body::open_placed_tiered(file, plan, 0, TierOpen::of_machine(plan.machine), &meta)
    }

    /// The batch's buffers under the batched schedule (with the card-timing
    /// events when asked); none under steps.
    fn prepare(m: &mut GpuModel<Body>, cfg: &Ds41Cfg) -> Result<bool, GpuError> {
        if cfg.feed != PrefillMode::Batch {
            return Ok(false);
        }
        body::prepare_prefill(m)?;
        if cfg.card_timing {
            let (gpu, _, b) = m.body_parts(WHAT)?;
            b.set_prefill_card_timing(gpu, true)?;
        }
        Ok(true)
    }
}

impl Prompt for Body {
    /// Under the body's `BLOOMERY_PREFILL`: a text call of at most
    /// [`body::PAIR_BELOW`] ids as pair replays (and one one-row replay when
    /// the count is odd), each kept as a prompt with no row; a longer call's
    /// batches ([`body::prefill`], one residency pass: [`prompt_pass`]); or
    /// one step per id with one readback after the last, the host tier's
    /// route trace told the call's positions first. A batched call while a
    /// route trace is attached is refused by name, the short walk included.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        match m.body(WHAT)?.prefill_mode() {
            PrefillMode::Batch if m.body(WHAT)?.hybrid().route_traced() => Err(GpuError::Shape {
                what: WHAT,
                detail: "a batched prompt call while a route trace is attached: the trace \
                         records the step feed (BLOOMERY_PREFILL=steps)"
                    .to_string(),
            }),
            PrefillMode::Batch if !ids.is_empty() && ids.len() <= body::PAIR_BELOW => {
                prompt_pairs(m, ids, None)
            }
            PrefillMode::Batch => prompt_pass(m, |m| body::prefill(m, ids)),
            PrefillMode::Steps => {
                let pos = m.pos();
                m.body_parts(WHAT)?
                    .2
                    .hybrid_mut()
                    .route_prompt(pos, ids.len())?;
                m.step(ids)
            }
        }
    }
}

impl Keep for Body {
    /// [`Body::keep_point`].
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        m.body(WHAT)
            .map_or(0, |b| u32::try_from(b.keep_point(n as usize)).unwrap_or(0))
    }

    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        m.rollback(n)
    }
}

impl Tapped for Session<Body> {
    fn tap_width(&self) -> usize {
        self.model().body(WHAT).map_or(0, Body::feature_width)
    }

    /// [`Body::read_features`] of the last call's first `rows` rows.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        let (gpu, _, b) = self.model_mut().body_parts(WHAT)?;
        Ok(b.read_features(gpu, rows)?.values)
    }
}

impl TappedPrompt for Session<Body> {
    /// The tap is built at the draft's open ([`open_draft`]) for its layers;
    /// a different set is refused by name.
    fn set_taps(&mut self, layers: &[usize]) -> Result<(), SessionError> {
        match self.model().body(WHAT)?.feature_layers() {
            Some(held) if held == layers => Ok(()),
            Some(held) => Err(SessionError::Refused(format!(
                "{WHAT}: the feature tap reads the layers {held:?}, not {layers:?}"
            ))),
            None => Err(SessionError::Refused(format!(
                "{WHAT}: no feature tap: open_draft attaches it before the target's first capture"
            ))),
        }
    }

    /// [`Target::prompt`] with the feature tap's rows handed to `sink` (the
    /// first position they hold, then one row a position): under the batched
    /// schedule the rows of the call's last `window` positions, per batch
    /// ([`body::prefill_with`]), or one call of those rows when the call is
    /// the short pair walk ([`body::PAIR_BELOW`]); under steps every
    /// position's, after its step, the host tier's route trace told the
    /// call's positions first as [`Prompt::prompt`] tells it. The argmax
    /// after the last id. A sink's error ends the call and comes back as it
    /// was.
    fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: &mut dyn FnMut(u32, &[f32]) -> Result<(), SessionError>,
    ) -> Result<u32, SessionError> {
        let mut refused = None;
        let called = self.feed_tapped(ids, window, &mut |first, rows| {
            sink(first, rows).map_err(|e| {
                refused = Some(e);
                GpuError::State {
                    what: WHAT,
                    missing: "a feature sink that took the rows",
                }
            })
        });
        match refused {
            Some(e) => Err(e),
            None => called,
        }
    }
}

impl Session<Body> {
    /// The prompt call of [`TappedPrompt::prompt_tapped`], its sink's errors
    /// the body's.
    fn feed_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: FeatureSink<'_>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt")?;
        let _busy = self.model().prompt_busy();
        match self.model().body(WHAT)?.prefill_mode() {
            PrefillMode::Batch if !ids.is_empty() && ids.len() <= body::PAIR_BELOW => Ok(
                prompt_pairs(self.model_mut(), ids, Some(FeatureRows { window, sink }))?,
            ),
            PrefillMode::Batch => {
                let rows = FeatureRows { window, sink };
                Ok(prompt_pass(self.model_mut(), |m| {
                    body::prefill_with(m, ids, Some(rows))
                })?)
            }
            PrefillMode::Steps => {
                let pos = self.pos();
                self.model_mut()
                    .body_parts(WHAT)?
                    .2
                    .hybrid_mut()
                    .route_prompt(pos, ids.len())?;
                let mut next = None;
                for &id in ids {
                    let pos = self.pos();
                    next = Some(self.model_mut().step(&[id])?);
                    let (gpu, _, b) = self.model_mut().body_parts(WHAT)?;
                    sink(pos, b.read_features(gpu, 1)?.values)?;
                }
                next.ok_or_else(|| SessionError::Refused("a prompt of no ids".into()))
            }
        }
    }
}

/// A text batch of at most [`body::PAIR_BELOW`] ids: `⌈n/2⌉` pair replays and
/// one one-row replay when `n` is odd, each ended with a prompt keep of no
/// row. The argmax is the last replay's last row. `features`, when set, gets
/// one call of the call's last `window` positions (all of them in a shorter
/// call) — the batch's last-window rule, not one call per replay. An empty
/// `ids` is the batch's "a prompt of no ids". A failure takes the call back
/// with [`body::take_back`], unless it faulted.
fn prompt_pairs(
    m: &mut GpuModel<Body>,
    ids: &[u32],
    features: Option<FeatureRows<'_>>,
) -> Result<u32, GpuError> {
    if ids.is_empty() {
        return Err(GpuError::Shape {
            what: PREFILL_WHAT,
            detail: "a prompt of no ids".to_string(),
        });
    }
    if features.is_some() && m.body(WHAT)?.feature_layers().is_none() {
        return Err(GpuError::State {
            what: PREFILL_WHAT,
            missing: "a feature tap (attach_features)",
        });
    }
    let first = m.pos() as usize;
    let walked = walk_pairs(m, ids, features);
    match walked {
        Ok(token) => Ok(token),
        Err(e) => {
            // The walk's own error wins, as `prompt_pass` returns the call's
            // error when the keep after it also fails.
            let _ = close_open_prompt(m);
            Err(body::take_back(m, first, e))
        }
    }
}

/// [`prompt_pairs`] once the tap check has passed. `ids` is not empty.
fn walk_pairs(
    m: &mut GpuModel<Body>,
    ids: &[u32],
    mut features: Option<FeatureRows<'_>>,
) -> Result<u32, GpuError> {
    let first = m.pos() as usize;
    let end = first + ids.len();
    let from = match features.as_ref() {
        Some(f) => end.saturating_sub(f.window).max(first),
        None => usize::MAX,
    };
    let mut kept = Vec::new();
    let mut kept_at = None;
    let mut token = 0;
    let mut i = 0;
    while i < ids.len() {
        let rows = if i + 1 < ids.len() {
            let out = m.step_rows([ids[i], ids[i + 1]])?;
            token = out[1];
            i += 2;
            if i == ids.len() {
                // Row 1 of this last pair is position n - 1. The step head
                // holds row 0 until this copy.
                m.settle_row(1)?;
            }
            2
        } else {
            token = m.replay_row(ids[i])?;
            i += 1;
            1
        };
        m.keep_rows(KeptRows::prefix(0), PassKind::Prompt)?;
        if features.is_some() {
            copy_kept(m, rows, from, &mut kept, &mut kept_at)?;
        }
    }
    if let (Some(f), Some(at)) = (features.as_mut(), kept_at) {
        (f.sink)(at, &kept)?;
    }
    Ok(token)
}

/// The last pass's feature rows that sit at or past `from`, appended to
/// `kept`. Row `r` is position `pos + r` of that pass.
fn copy_kept(
    m: &mut GpuModel<Body>,
    rows: usize,
    from: usize,
    kept: &mut Vec<f32>,
    kept_at: &mut Option<u32>,
) -> Result<(), GpuError> {
    let (gpu, _, b) = m.body_parts(WHAT)?;
    let feat = b.read_features(gpu, rows)?;
    for r in 0..rows {
        let pos = feat.pos as usize + r;
        if pos < from {
            continue;
        }
        if kept_at.is_none() {
            *kept_at = Some(u32::try_from(pos).map_err(|_| GpuError::Shape {
                what: WHAT,
                detail: format!("position {pos} passes u32"),
            })?);
        }
        let row = feat.row(r).ok_or(GpuError::Shape {
            what: WHAT,
            detail: format!("feature row {r} of a pass of {rows}"),
        })?;
        kept.extend_from_slice(row);
    }
    Ok(())
}

/// Close a pass this walk opened and has not kept, as a prompt with no row.
/// Without a machine a keep on a closed call changes nothing; with one, a
/// keep is given only while a pass is open.
fn close_open_prompt(m: &mut GpuModel<Body>) -> Result<(), GpuError> {
    let unkept = m.body(WHAT)?.hybrid().swap().is_none_or(|s| s.pass_open());
    if unkept {
        m.keep_rows(KeptRows::prefix(0), PassKind::Prompt)?;
    }
    Ok(())
}

/// The name [`body::prefill`]'s errors use, so a short call that fails says
/// the same thing the batch says.
const PREFILL_WHAT: &str = "deepseek41 prefill";
