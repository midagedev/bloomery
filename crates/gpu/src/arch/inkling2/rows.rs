//! The Inkling 2 verify: the step widened to `m` rows ([`Rows`]), 2 to [`VERIFY_ROWS`] a call.
//! A verify walks the prompt pass's rows ([`super`]) from the held position with the step's
//! kernel at every node: the mixes as the step's launch pair over the rows ([`MixAt::mix_step`], a
//! query a job), the controllers as the step's launch pair at the rows (a block one row's), the
//! input norms and projections as the step's `norm_proj` launch over the rows (its rows entry) or,
//! for a layer the step projects a site at a time, the norms and the step's gemvs over the rows, the
//! convs on the visit's conv ring (a row reads the rows before it, this call's too), the
//! epilogue's append of every row's keys before the step's rel-bias flash reads each row's live
//! keys, the router's rows, the routed rows on the step's card leg (a run's slots naming one
//! expert read it once) and the head once over every row ([`RowHeads::One`]). Each row is the
//! step's bits at its position, so a verify of `m` rows is `m` steps in turn, and it leaves every
//! store, the held position and the written end where they leave them; a rollback after it is the
//! step's.
//!
//! A verify takes no forced picks, no armed stage taps ([`Body::arm_taps`]) and no importance
//! collector: each is refused by name. A block draft's taps it lays ([`super::pass`]).

use super::super::route::N_USED;
use super::super::{Body, DOWN_ROLE, Dims, GATE_UP_ROLE, HEAD, Layer, MIX_EPS, head_logits};
use super::{Bufs, MixAt, RowWalk, Tail};
use crate::attn_res::AttnResNormedArgs;
use crate::attn_res_step::MixStepScratch;
use crate::card_leg::{LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::flash_gqa::window_segments;
use crate::head::Head;
use crate::hybrid::Chain;
use crate::model::{GpuModel, MAX_PASS_ROWS, RowHeads, Rows};
use crate::tensor::Window;
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::inkling2::names;
use model::placement::accepts::{self, Program};
use model::placement::kernels::DownAct;

const WHAT: &str = "inkling2 verify";

/// The most rows a verify walks: a draft block, its anchor and the drafted ids.
pub(in crate::arch::inkling2) const VERIFY_ROWS: usize = MAX_PASS_ROWS;
// `GpuModel::capture_verify` names each row count.
const _: () = assert!(VERIFY_ROWS == 8);

/// A verify's own buffers beside the prompt pass's, whose rows it walks: the step's flash
/// partials and card leg at [`VERIFY_ROWS`] rows, and its mixes' statistics and tickets at as many
/// (at the small pass's rows when those are more: the pass's mixes run the step's pair on them too).
pub(in crate::arch::inkling2) struct VerifyBufs {
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    pub(super) mix: MixStepScratch,
    leg: LegBufs,
}

impl VerifyBufs {
    /// The buffers of a load of dims `d`. Load-time only.
    pub(in crate::arch::inkling2) fn new(
        stream: &CudaStream,
        d: &Dims,
    ) -> Result<VerifyBufs, GpuError> {
        let segs = window_segments(d.window).max(window_segments(0));
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len)?)
        };
        let shape = LegShape {
            cap: VERIFY_ROWS,
            slots_per_col: N_USED,
            n: d.n,
            ff: d.ff,
        };
        Ok(VerifyBufs {
            part_v: z(VERIFY_ROWS * d.n_head * segs * HEAD)?,
            part_ms: z(VERIFY_ROWS * d.n_head * segs * 2)?,
            mix: MixStepScratch::rows(stream, (d.n, VERIFY_ROWS.max(super::SMALL_ROWS)))?,
            leg: LegBufs::for_down(stream, shape, DownAct::Q8Blocks32)?,
        })
    }

    /// Device bytes the buffers take.
    pub(in crate::arch::inkling2) fn bytes(&self) -> usize {
        self.part_v.num_bytes() + self.part_ms.num_bytes() + self.mix.bytes() + self.leg.bytes()
    }
}

impl Body {
    /// A verify of `m` rows on this load now, or why not by name: a count outside 2 to the rows
    /// a verify and the pass hold, forced picks, armed taps, an importance collector.
    fn verify_refusal(&self, m: usize) -> Result<(), GpuError> {
        let pass = self.pass.as_ref().ok_or(GpuError::state(
            WHAT,
            "a whole-card load (a placed load has no verify: its prompt and drafts are steps)",
        ))?;
        let most = VERIFY_ROWS.min(pass.rows);
        if !(2..=most).contains(&m) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a verify of {m} rows: 2 to {most} (a draft block's {VERIFY_ROWS}, the \
                     pass's {} at this context)",
                    pass.rows
                ),
            ));
        }
        if self.forced.is_some() || pass.forced.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a verify under forced picks: unforce them first",
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a verify with the taps armed: disarm them first",
            ));
        }
        if crate::imatrix::collector().is_some() {
            return Err(GpuError::state(
                WHAT,
                "a verify under an importance collector, which reads the step's sites",
            ));
        }
        Ok(())
    }
}

impl Rows for Body {
    const MAX_ROWS: usize = VERIFY_ROWS;
    /// No host tier serves an Inkling 2 replay; the verify is the step's one-token layout at `m`
    /// rows.
    const CHAIN: Chain = Chain::Step;
    /// The head's rows are independent: one read of `output` for the verify.
    const HEADS: RowHeads = RowHeads::One;

    /// The verify's ids, ring slots, first position and global factors staged as a prompt
    /// pass's, a draft's taps told the rows the walk lays, and the held position and the written
    /// end moved as `tokens.len()` steps move them. Refused by name off the held position and as
    /// [`Body::verify_refusal`] says.
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.verify_refusal(tokens.len())?;
        if pos != self.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a verify at position {pos}; the stores stand at {}",
                    self.held
                ),
            ));
        }
        if let Some(t) = self.draft.as_mut() {
            t.plan(tokens.len());
        }
        let pass = self
            .pass
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a whole-card load"))?;
        let (rows, ring) = (pass.rows, self.ring);
        let (floor, alpha) = (self.hp.log_floor, self.hp.log_alpha);
        pass.b
            .stage(stream, (tokens, pos, rows), (ring, floor, alpha))?;
        self.held = pos + launch_u32(WHAT, "rows", tokens.len())?;
        self.end = self.end.max(self.held);
        Ok(())
    }

    /// The walk of the staged rows into `heads[0]`, the head of the verify's rows.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        let head = heads
            .first_mut()
            .ok_or(GpuError::state(WHAT, "the verify's head of its rows"))?;
        let m = head.m();
        self.verify_refusal(m)?;
        self.walk_rows(gpu, w, head, (m, Tail::Verify))
    }
}

impl GpuModel<Body> {
    /// Capture the verify of every row count, 2 to [`VERIFY_ROWS`], each into its own graph
    /// beside the step's, and return their node counts in that order: a load that serves a
    /// draft captures them once, after the step's.
    pub fn capture_verify(&mut self) -> Result<[usize; VERIFY_ROWS - 1], GpuError> {
        Ok([
            self.capture_rows::<2>()?,
            self.capture_rows::<3>()?,
            self.capture_rows::<4>()?,
            self.capture_rows::<5>()?,
            self.capture_rows::<6>()?,
            self.capture_rows::<7>()?,
            self.capture_rows::<8>()?,
        ])
    }
}

/// One mix of the step's pair over the verify's rows: the banked blocks and the stream, the query
/// row (`j` of its table) and the rows it writes. A job is one query, so each query's rows land
/// in their own buffer, as the controllers and the norms read them.
pub(super) struct StepMix<'a> {
    banks: (&'a DeviceBuffer<f32>, usize),
    cur: &'a DeviceBuffer<f32>,
    query: (&'a DeviceBuffer<f32>, usize),
    y: &'a mut DeviceBuffer<f32>,
}

impl<'a> StepMix<'a> {
    /// The mix over `banks` and the stream by query row `j` of the query table, into `y`'s rows.
    pub(super) fn new(
        banks: (&'a DeviceBuffer<f32>, usize),
        cur: &'a DeviceBuffer<f32>,
        query: (&'a DeviceBuffer<f32>, usize),
        y: &'a mut DeviceBuffer<f32>,
    ) -> StepMix<'a> {
        StepMix {
            banks,
            cur,
            query,
            y,
        }
    }
}

/// Query row `j` of a query table: its `n` values.
fn query_row(buf: &DeviceBuffer<f32>, (j, n): (usize, usize)) -> Result<Window<'_, f32>, GpuError> {
    Window::<f32>::of(buf, j * n * 4, n)
}

/// `m`'s mix at the one query row `query`: the visit's pitch, row count, scale and epsilon and
/// its fault sink.
fn step_args<'a>(
    m: &StepMix<'a>,
    query: &'a DeviceBuffer<f32>,
    at: &MixAt<'_>,
) -> AttnResNormedArgs<'a> {
    AttnResNormedArgs {
        banks: m.banks.0,
        n_banks: m.banks.1,
        stride: at.pitch,
        cur: m.cur,
        queries: query,
        nq: 1,
        scale: at.scale,
        eps: MIX_EPS,
        n: at.n,
        m: at.m,
        fault: at.fault,
    }
}

impl MixAt<'_> {
    /// The step's launch pair over the rows ([`crate::attn_res_step`]): the mix `a` and, beside
    /// it, the independent mix `b` into their outputs, each row of an output the step mix's bits
    /// at that row ([`MixAt::mix`] writes the same row, a query a launch). The banks are the
    /// pass's pitch apart, the statistics and tickets `scratch`'s, a row each.
    pub(super) fn mix_step(
        &self,
        (scratch, a, b): (&mut MixStepScratch, StepMix<'_>, Option<StepMix<'_>>),
    ) -> Result<(), GpuError> {
        let qa = query_row(a.query.0, (a.query.1, self.n))?;
        let a_args = step_args(&a, &qa, self);
        let StepMix { y: y_a, .. } = a;
        let Some(b) = b else {
            return self
                .step
                .enqueue_rows(self.stream, (a_args, y_a), None, scratch);
        };
        let qb = query_row(b.query.0, (b.query.1, self.n))?;
        let b_args = step_args(&b, &qb, self);
        let StepMix { y: y_b, .. } = b;
        self.step
            .enqueue_rows(self.stream, (a_args, y_a), Some((b_args, y_b)), scratch)
    }
}

impl RowWalk<'_> {
    /// The verify's routed rows on the step's card leg: every row's slots at the router's picks
    /// (a run's slots naming one expert read it once), each row's weighted sum into `moe`.
    pub(super) fn shadow_rows(&mut self, slot: usize, layer: &Layer) -> Result<(), GpuError> {
        let (gpu, w, k, m, rule) = (self.gpu, self.w, self.k, self.m, self.act);
        let arms = accepts::leg_arms(Program::Inkling2Body)
            .ok_or_else(|| GpuError::state(WHAT, "the inkling2 program's card leg"))?;
        let names = layer
            .stacks
            .as_ref()
            .ok_or(GpuError::state(WHAT, "the card's routed stacks"))?
            .each_ref()
            .map(String::as_str);
        let stacks = LegStacks::of_arms(w, names, arms, (GATE_UP_ROLE, DOWN_ROLE))?;
        let v = self
            .verify
            .as_deref_mut()
            .ok_or(GpuError::state(WHAT, "a verify's buffers"))?;
        let Bufs {
            fxn,
            picks,
            weights,
            moe,
            ..
        } = &mut *self.b;
        let run = LegRun {
            layer: slot,
            stacks: &stacks,
            normed: fxn,
            m,
            sel: picks,
            n_card: self.d.n_expert,
            rule,
        };
        k.leg.enqueue(
            gpu,
            &mut v.leg,
            &run,
            Some(LegSum {
                w: weights,
                acc: moe,
            }),
        )
    }

    /// The head's mix of the banked blocks and the stream over every row into the head's rows
    /// (the step's pair), then the head over them ([`head_logits`]).
    pub(super) fn finish_rows(
        &mut self,
        (n_visits, pad): (usize, &DeviceBuffer<f32>),
        head: &mut Head,
        logits: (f32, usize, usize),
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let mx = self.mixes(gpu.layer_sink(n_visits)?);
        let b = &*self.b;
        let v = self
            .verify
            .as_deref_mut()
            .ok_or(GpuError::state(WHAT, "a verify's buffers"))?;
        mx.mix_step((
            &mut v.mix,
            StepMix::new(
                (&b.banks, self.pushed),
                &b.x,
                (w.f32_buf(WHAT, names::OUTPUT_RES_QUERY)?, 0),
                head.input_mut(),
            ),
            None,
        ))?;
        head_logits((gpu, w), head, pad, logits, self.m)
    }
}
