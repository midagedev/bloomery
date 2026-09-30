//! The V4.1 session: [`Body`] behind the session's traits, planned once by its
//! placement and loaded by gpumodel's constructor ([`Body::open_placed_tiered`]),
//! with the plan's expert tier card hung under the host tier when the
//! placement names one.

mod draft;

pub use draft::CardDraft;

use bloomery_gpu::GpuError;
use bloomery_gpu::GpuModel;
use bloomery_gpu::host::PassKind;
use bloomery_gpu_deepseek41::body::{
    self, Body, BodyMeta, FeatureRows, FeatureSink, OpenCfg, PrefillMode, TierOpen,
};
use gguf::Split;
use model::arch::deepseek41::place::PlanInputs;
use model::placement::{Machine, Plan};
use runtime::{Tapped, Target};

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

    /// The stage card, card 0, and the plan's expert tier card when it
    /// names one ([`tier_of`]).
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
        Body::open_placed_tiered(file, plan, 0, tier_of(plan.machine)?, &meta)
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

/// The expert tier card `machine` names, as the body opens it: its index in
/// [`Machine::all_cards`] (after the one stage card) and its name, by which
/// the card is found. `None` without a tier; more than one is refused by
/// name, since the host tier holds one.
pub fn tier_of(machine: &Machine) -> Result<Option<TierOpen>, GpuError> {
    match machine.tiers.as_slice() {
        [] => Ok(None),
        [t] => Ok(Some(TierOpen {
            card: machine.cards.len(),
            name: t.name.clone(),
        })),
        more => Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the placement names {} expert tier cards; the host tier holds one",
                more.len()
            ),
        }),
    }
}

impl Prompt for Body {
    /// Under the body's `BLOOMERY_PREFILL`: the prompt call's batches
    /// ([`body::prefill`], one residency pass: [`call`]), or one step per id
    /// with one readback after the last, the host tier's route trace told the
    /// call's positions first. A batched call while a route trace is attached
    /// is refused by name.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        match m.body(WHAT)?.prefill_mode() {
            PrefillMode::Batch if m.body(WHAT)?.hybrid().route_traced() => Err(GpuError::Shape {
                what: WHAT,
                detail: "a batched prompt call while a route trace is attached: the trace \
                         records the step feed (BLOOMERY_PREFILL=steps)"
                    .to_string(),
            }),
            PrefillMode::Batch => call(m, |m| body::prefill(m, ids)),
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

/// A prompt call `run` of `m` as one residency pass: its boundary before it
/// and none inside ([`GpuModel::pass_boundary`]), so the slot map is one map
/// for the whole call, and 0 rows kept after it, whatever it returned — the
/// residency rule counts decode rows only, and the batch service notes no
/// id. Nothing more on a load with no residency machine.
fn call<T>(
    m: &mut GpuModel<Body>,
    run: impl FnOnce(&mut GpuModel<Body>) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    m.pass_boundary()?;
    let r = run(m);
    let kept = m.keep_rows(0, PassKind::Prompt);
    // The call's own error first: a keep refused after a failed call is its echo.
    let v = r?;
    kept?;
    Ok(v)
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

impl Session<Body> {
    /// [`Target::prompt`] with the feature tap's rows handed to `sink` (the
    /// first position they hold, then one row a position): under the batched
    /// schedule the rows of the call's last `window` positions, per batch
    /// ([`body::prefill_with`]); under steps every position's, after its
    /// step, the host tier's route trace told the call's positions first as
    /// [`Prompt::prompt`] tells it. The argmax after the last id.
    pub fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: FeatureSink<'_>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt")?;
        match self.model().body(WHAT)?.prefill_mode() {
            PrefillMode::Batch => {
                let rows = FeatureRows { window, sink };
                Ok(call(self.model_mut(), |m| {
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
