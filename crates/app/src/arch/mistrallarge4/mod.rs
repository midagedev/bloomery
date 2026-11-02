//! The Mistral Large 4 session: [`Body`] behind the session's traits, planned
//! once by its placement (the routed experts the card holds a prefix of by
//! the expert rule, the rest on the host tier, the next ids of each card
//! layer on the expert tier card when the placement names one) and loaded by
//! the body's own placed load ([`Body::open_placed_with`]), under the
//! configuration's residency ([`HybridCfg::residency`]).
//!
//! A prompt is fed in batches of rows through the common walk
//! ([`bloomery_gpu_mistrallarge4::prefill::prefill`]), one residency pass a
//! call, and leaves the bits the decode steps leave: every owner layer's
//! latent rows at those positions, every full indexer layer's index keys, and
//! the last position's logits. [`PROMPT_FEED`] is the word the server's `load`
//! record gives for it.
//!
//! A block draft reads the rows entering the layers its file names: the body's
//! feature tap ([`bloomery_gpu_mistrallarge4::features`]), attached before the
//! first capture, lays them in the step, a verify's rows and the prompt call,
//! and the session hands them back ([`Tapped`], [`TappedPrompt`]).
//!
//! Every store is a plane of positions a later position never reads before it
//! writes, so a cut keeps every held position. The plan counts the configured
//! resident sequences ([`HybridCfg::slots`]); the server parks them through
//! the body's actual `Slots` ([`crate::Session::add_slots`]).
//!
//! The session's shape is the common owner's ([`super::Batch`]): what the
//! placement plans, how the prompt runs and how a cut is kept are shared with
//! the other families that run one chain on one card. This module holds what
//! Mistral Large 4's architecture forces: the plan inputs, the load and the
//! batch.

use std::num::NonZeroUsize;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_mistrallarge4::body::Body;
use bloomery_gpu_mistrallarge4::features::FeatureRows;
use bloomery_gpu_mistrallarge4::prefill::{prefill, prefill_with, set_prefill_group};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::mistrallarge4::place::{self, PlanInputs};
use model::placement::Plan;
use runtime::{TapSink, Tapped, TappedPrompt};

use super::{Batch, HybridCfg, ResidentSequences};
use crate::{Session, SessionError};

pub mod dspark;

/// The session's name in its refusals.
const WHAT: &str = "mistrallarge4 session";

/// How the prompt is fed, the word of the server's `load` record: in batches
/// of rows through the common walk ([`Batch::batch`]).
pub const PROMPT_FEED: &str = "batch";

/// A Mistral Large 4 session: the body's stores per position, its resident
/// sequences parked through its `Slots`.
pub type MistralLarge4Session = crate::Session<Body>;

impl Batch for Body {
    type Inputs = PlanInputs;
    type Sequences = ResidentSequences;

    const WHAT: &'static str = self::WHAT;

    /// The family's own reading ([`PlanInputs::describe`]), not the strict
    /// one: the coverage list still names the template's parsers, which the
    /// server runs.
    fn read(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The prompt group's units, records and lists priced into the plan's
    /// prompt reserve.
    fn configure(inputs: &mut PlanInputs, cfg: &HybridCfg) -> Result<(), GpuError> {
        place::configure_group(inputs, cfg.group).map_err(|e| GpuError::plan(WHAT, e))
    }

    /// The body's placed load on the plan's one card, under `residency`. The
    /// plan counts `_slots` sequences' stores; the session parks them after
    /// the load.
    fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        host: HostCfg,
        residency: Residency,
        _slots: NonZeroUsize,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_with(file, plan, inputs, 0, host, residency)
    }

    fn batch(
        gpu: &Gpu,
        w: &Weights,
        body: &mut Body,
        head: &mut Head,
        ids: &[u32],
        at: u32,
    ) -> Result<bool, GpuError> {
        prefill(gpu, w, body, head, ids, at)
    }

    fn group(m: &mut GpuModel<Body>, g: usize) -> Result<bool, GpuError> {
        set_prefill_group(m, g)
    }
}

impl Tapped for Session<Body> {
    fn tap_width(&self) -> usize {
        self.model().body(WHAT).map_or(0, Body::feature_width)
    }

    /// [`Body::read_features`] of the last step's or verify's first `rows` rows, kept for the
    /// borrow.
    fn taps(&mut self, rows: usize) -> Result<&[f32], SessionError> {
        let read = {
            let (gpu, _, body) = self.model_mut().body_parts(WHAT)?;
            body.read_features(gpu, rows)?
        };
        self.tapped = read;
        Ok(&self.tapped)
    }
}

impl TappedPrompt for Session<Body> {
    /// The tap is attached at the open for the draft's layers
    /// ([`bloomery_gpu_mistrallarge4::features::attach_features`]), before the first capture; a
    /// different set is refused by name.
    fn set_taps(&mut self, layers: &[usize]) -> Result<(), SessionError> {
        match self.model().body(WHAT)?.feature_layers() {
            Some(held) if held == layers => Ok(()),
            Some(held) => Err(SessionError::Refused(format!(
                "{WHAT}: the feature tap reads the layers {held:?}, not {layers:?}"
            ))),
            None => Err(SessionError::Refused(format!(
                "{WHAT}: no feature tap: the open attaches it before the target's first capture"
            ))),
        }
    }

    /// [`Target::prompt`]'s batches (one residency pass) with the tapped rows of the call's last
    /// `window` positions handed to `sink` (the first position they hold, then one row a
    /// position) after each group of units that holds any of them
    /// ([`prefill_with`]): the cache, the stores and the argmax are the plain call's, bit for
    /// bit. A sink's error ends the call and comes back as it was.
    fn prompt_tapped(
        &mut self,
        ids: &[u32],
        window: usize,
        sink: TapSink<'_, SessionError>,
    ) -> Result<u32, SessionError> {
        self.idle("prompt_tapped")?;
        let _busy = self.model().prompt_busy();
        let mut refused = None;
        let ran = prompt_pass(self.model_mut(), |m| {
            m.run_rows(ids.len(), WHAT, |gpu, w, body, head, pos| {
                let mut forward = |first: u32, rows: &[f32]| -> Result<(), GpuError> {
                    sink(first, rows).map_err(|e| {
                        refused = Some(e);
                        GpuError::State {
                            what: WHAT,
                            missing: "a tap sink that takes the rows (its error follows)",
                        }
                    })
                };
                let rows = FeatureRows {
                    window,
                    sink: &mut forward,
                };
                prefill_with(gpu, w, body, head, ids, pos, Some(rows))
            })
        });
        // The sink's own error first: the call's echo of it is a placeholder.
        if let Some(e) = refused {
            return Err(e);
        }
        ran?.ok_or_else(|| {
            SessionError::Refused(format!("{WHAT}: the head's token after the prompt"))
        })
    }
}
