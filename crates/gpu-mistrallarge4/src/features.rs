//! The feature tap a block draft reads: the rows entering the layers the draft's file names
//! (`target_layers`, the last layer's output when the file names the layer count), laid by the
//! body's walks and read back by its session. The common owner is
//! [`bloomery_gpu::draft_taps::DraftTaps`] (a slot a tapped layer, a row a position); this file
//! holds the two kinds of walk this family lays rows in:
//!
//! - the step and a verify's rows: one copy node a tapped layer a row, in the step's launches
//!   (`crate::step`), so they are captured with the chain and replayed with it. The tap's buffer
//!   is made before the first capture ([`attach_features`]) and its rows are the pass's
//!   ([`PASS_ROWS`]): row `r` holds the rows of position `pos + r`;
//! - a prompt call's units: the same copies, one a tapped layer a unit, in the common walk
//!   (`bloomery_gpu_deepseek41::prompt_walk::Entering`), into a buffer of the group's rows, and
//!   handed to the caller's sink after each group ([`FeatureRows`]). The buffer is remade with the
//!   group ([`crate::prefill::set_prefill_group`]).
//!
//! A tapped layer's rows are copies of the running row the walk already holds, so a tapped call is
//! the plain call's launches plus the copies: its cache, its state and its logits are bit for bit
//! the plain call's.

use std::ops::Range;

use bloomery_gpu::draft_taps::{DraftTaps, TapRowsSink};
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::prompt_walk::{Entering, group_sets};
use cuda_core::{CudaStream, DeviceBuffer};

use crate::body::Body;
use crate::body::rows::PASS_ROWS;
use crate::prefill::T_MAX;

/// What the tap's refusals name.
const WHAT: &str = "mistrallarge4 features";

/// The tapped layers' buffers: the step's and the prompt group's.
pub(crate) struct FeatureTap {
    /// The step's and a verify's rows, [`PASS_ROWS`] a slot.
    pub(crate) step: DraftTaps,
    /// A prompt group's units' rows, `group_sets(g) · T_MAX` a slot; `None` while the group is
    /// being remade.
    pub(crate) batch: Option<DraftTaps>,
    /// The count of hidden states the body lays in (the row entering each layer, and the last
    /// layer's output) and the width of a row.
    states: usize,
    hidden: usize,
    /// The units a group walks the prompt buffer was made for.
    sets: usize,
}

impl FeatureTap {
    /// The buffers for `layers` of a body of `n_layer` layers and `hidden` values a row, the
    /// prompt buffer for a group of `group` units. Refused by name: layers out of order or past
    /// the layer count (the count itself taps the last layer's output), and bytes the card's
    /// free room does not hold.
    fn new(
        gpu: &Gpu,
        layers: &[usize],
        (n_layer, hidden): (usize, usize),
        group: usize,
    ) -> Result<FeatureTap, GpuError> {
        let states = n_layer + 1;
        let step = DraftTaps::new(
            gpu.stream(),
            layers,
            states,
            (hidden, PASS_ROWS),
            gpu.alloc_room()?,
        )?;
        let sets = group_sets(group);
        let batch = DraftTaps::new(
            gpu.stream(),
            layers,
            states,
            (hidden, sets * T_MAX),
            gpu.alloc_room()?,
        )?;
        Ok(FeatureTap {
            step,
            batch: Some(batch),
            states,
            hidden,
            sets,
        })
    }

    /// The prompt buffer remade for a group of `group` units (between calls); nothing when it
    /// holds that many already.
    pub(crate) fn set_group(&mut self, gpu: &Gpu, group: usize) -> Result<(), GpuError> {
        let sets = group_sets(group);
        if sets == self.sets && self.batch.is_some() {
            return Ok(());
        }
        let layers = self.step.layers().to_vec();
        // The old buffer is freed first: the new one is sized against the room that leaves.
        self.batch = None;
        self.batch = Some(DraftTaps::new(
            gpu.stream(),
            &layers,
            self.states,
            (self.hidden, sets * T_MAX),
            gpu.alloc_room()?,
        )?);
        self.sets = sets;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.step.bytes() + self.batch.as_ref().map_or(0, DraftTaps::bytes)
    }
}

/// Build the feature tap a draft reads ([`Body::read_features`]) for `layers` — a draft's
/// `target_layers`, strictly rising, each at most the layer count (the count itself is the last
/// layer's output) — beside the step and the verify's rows. Before the first capture: a graph
/// captured without the tap would never write the rows its readback returns, so a model that
/// holds one is refused. Load-time only.
pub fn attach_features(m: &mut GpuModel<Body>, layers: &[usize]) -> Result<(), GpuError> {
    const WHAT_A: &str = "mistrallarge4 attach_features";
    if m.has_capture() {
        return Err(GpuError::State {
            what: WHAT_A,
            missing: "a model with no captured graph: attach the tap before the first capture",
        });
    }
    let (gpu, _, body) = m.body_parts(WHAT_A)?;
    body.attach_features(gpu, layers)
}

impl Body {
    /// [`attach_features`] on the body: refused when a tap exists already and while the
    /// per-layer taps are armed (a gate's span and injected rows would leave copies standing on
    /// rows the walk did not enter).
    fn attach_features(&mut self, gpu: &Gpu, layers: &[usize]) -> Result<(), GpuError> {
        const WHAT_A: &str = "mistrallarge4 Body::attach_features";
        if self.feat.is_some() || self.taps.is_some() {
            return Err(GpuError::State {
                what: WHAT_A,
                missing: "a body with one feature tap at most and the per-layer taps off",
            });
        }
        self.feat = Some(FeatureTap::new(
            gpu,
            layers,
            (self.sh.n_layer, self.sh.hidden),
            self.units.group(),
        )?);
        Ok(())
    }

    /// The layers the feature tap reads the input of, in its order (the layer count is the last
    /// layer's output); `None` without a tap.
    #[must_use]
    pub fn feature_layers(&self) -> Option<&[usize]> {
        self.feat.as_ref().map(|f| f.step.layers())
    }

    /// f32 values of one position's features: tapped layers × the model's width; 0 without a
    /// tap.
    #[must_use]
    pub fn feature_width(&self) -> usize {
        self.feat.as_ref().map_or(0, |f| f.step.width())
    }

    /// Device bytes of the feature tap's buffers, the step's and the prompt group's; 0 without
    /// a tap.
    #[must_use]
    pub fn feature_bytes(&self) -> usize {
        self.feat.as_ref().map_or(0, FeatureTap::bytes)
    }

    /// The first `rows` rows of the last step or verify pass, one [`Body::feature_width`] row
    /// each, a position's tapped layers in the order they were named: row `r` holds the features
    /// of position `pos + r`. Blocking on the engine stream. Refused by name without a tap, for
    /// more rows than the last walk laid in, and after a prompt call (which lays its rows in the
    /// prompt buffer, [`FeatureRows`]).
    pub fn read_features(&self, gpu: &Gpu, rows: usize) -> Result<Vec<f32>, GpuError> {
        let f = self.feat.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "a feature tap (attach_features)",
        })?;
        f.step.read(gpu.stream(), 0, rows)
    }
}

/// The feature rows a tapped prompt call hands over: those of its last `window` positions (all
/// of them when the call is shorter), as `(first position, rows)` after each group of units that
/// holds any of them, one [`Body::feature_width`] row a position, in order, no position twice or
/// missed.
pub struct FeatureRows<'a, 'b> {
    /// The trailing positions of the call whose rows are handed over.
    pub window: usize,
    /// Where the rows go: the first position they hold, then the rows.
    pub sink: &'a mut TapRowsSink<'b>,
}

impl<'a, 'b> FeatureRows<'a, 'b> {
    /// The reader of a call of `n` ids from position `at`, laying its rows in `taps`. Refused by
    /// name for positions past a u32.
    pub(crate) fn reader<'t>(
        self,
        taps: &'t mut DraftTaps,
        (at, n): (u32, usize),
    ) -> Result<Call<'t, 'a, 'b>, GpuError> {
        let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
        let len = u32::try_from(n).map_err(|_| shape(format!("{n} ids past a u32")))?;
        let kept = u32::try_from(self.window.min(n))
            .map_err(|_| shape("a window past a u32".to_string()))?;
        let end = at
            .checked_add(len)
            .ok_or_else(|| shape(format!("{n} ids from position {at} past a u32")))?;
        Ok(Call {
            taps,
            held: end - kept,
            sink: self.sink,
            units: Vec::new(),
        })
    }
}

/// A tapped call's reader of the walk's rows ([`Entering`]): each unit's rows copied into the
/// tap at the layers it names, and the kept positions handed to the sink once the group's
/// launches are enqueued.
pub(crate) struct Call<'t, 'a, 'b> {
    taps: &'t mut DraftTaps,
    /// The first position the sink is handed.
    held: u32,
    sink: &'a mut TapRowsSink<'b>,
    /// The group's units: their positions and the row each one's rows begin at in the tap.
    units: Vec<(Range<u32>, usize)>,
}

impl Entering for Call<'_, '_, '_> {
    fn group(&mut self, runs: &[Range<u32>]) -> Result<(), GpuError> {
        self.units.clear();
        let mut at = 0usize;
        for run in runs {
            self.units.push((run.clone(), at));
            at += (run.end - run.start) as usize;
        }
        self.taps.plan(at);
        Ok(())
    }

    fn enter(
        &mut self,
        stream: &CudaStream,
        l: usize,
        u: usize,
        (x, t): (&DeviceBuffer<f32>, usize),
    ) -> Result<(), GpuError> {
        let Some(slot) = self.taps.slot(l) else {
            return Ok(());
        };
        let (run, at) = self.units.get(u).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("unit {u} of a group of {}", self.units.len()),
        })?;
        // A unit wholly before the window has nothing to hand over.
        if run.end <= self.held {
            return Ok(());
        }
        self.taps.enter_at(stream, slot, x, (*at, t))
    }

    fn done(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        for (run, at) in &self.units {
            if run.end <= self.held {
                continue;
            }
            let from = run.start.max(self.held);
            let skip = (from - run.start) as usize;
            let rows = (run.end - from) as usize;
            let tapped = self.taps.read(stream, at + skip, rows)?;
            (self.sink)(from, &tapped)?;
        }
        Ok(())
    }
}
