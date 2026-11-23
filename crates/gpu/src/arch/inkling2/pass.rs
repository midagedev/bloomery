//! The Inkling 2 prompt pass: the step's walk ([`super`], its module doc's steps) over a prompt's
//! rows, up to [`place::PASS_ROWS`] a pass, the card holding every layer. Each visit runs its
//! mixers at the pass's rows and leaves its stores where that many steps leave them:
//!
//! - the mixes (`attn_res::enqueue_normed`, one query a launch, the banks a pass's rows apart),
//!   the controllers in their rows forms (`loop_control::enqueue_attn_rows`, `enqueue_rows`), the
//!   norms, the output gate and the joins over the rows;
//! - the projections of one input as one wide GEMM over the rows each ([`project_wide`]: the
//!   input quantized once into q8 blocks of 32, the dense table over the rows);
//! - the k, v and post convs over the rows on the visit's conv ring, which carries the last
//!   inputs across rows, passes and calls (`conv_res`);
//! - the epilogue's append of the rows' keys at their positions (a windowed visit's ring slots,
//!   [`super::state`]: the ring holds the window and a pass's rows) before the rel-bias prefill
//!   flash reads them over the cached prefix and the rows (`rel_bias_prefill`);
//! - the feed-forward block in the step's three parts over the rows ([`RowWalk::ffn_front`]: the
//!   mix, the controller, the norm and the router's rows; [`RowWalk::ffn_shadow`]: the routed rows
//!   on the grouped route GEMM, the table, gate·up, SiTU and down, then the card sum;
//!   [`RowWalk::ffn_back`]: the post conv and the stream's sum): a placed body attaches its host
//!   tier between them. Every visit routes its rows anew.
//!
//! After the last visit the head's mix runs over the rows and the head over the last row, so the
//! call's argmax and logits are the last position's, the first decode step's input.
//!
//! The order is the step's where the kernels are the step's (the mixes, the convs, the
//! epilogue, the controllers' rows forms, the router's rows, the output gate), and the prompt
//! kernels' elsewhere: the wide projections read q8 blocks of 32 where the step's gemv reads the
//! f32 row, the prefill flash sums its keys in its own tiles, and the route GEMMs read q8 blocks
//! of 32 where the step's leg reads q8_1 at one scale a 128 values.
//!
//! The buffers are [`place::pass_bytes`] at the load's rows, the plan's reserve; the load refuses
//! by name a pass that takes more.
//!
//! A verify ([`rows`]) walks the same rows with the step's kernels where the prompt's differ: the
//! projections as the step's gemvs over the rows, the attention as the step's rel-bias flash at
//! the rows' causal offsets, the routed rows on the step's card leg, and the head over every row;
//! so each row is a step's bits.

use super::route::N_USED;
use super::state::{self, VisitStore};
use super::{
    Body, DOWN_ROLE, Dims, GATE_UP_ROLE, HEAD, Kernels, Layer, MIX_EPS, MSG_ATTN, MSG_FFN, WHAT,
    conv_scale, head_logits, tap_widths,
};
use crate::arch::hybrid::{Site, WideCx, project_wide};
use crate::attn_res::{AttnResKernels, AttnResNormedArgs};
use crate::card_leg::{LegStacks, RouteIo};
use crate::card_slots::{CardAcc, CardSlotKernels};
use crate::fault::FaultSink;
use crate::gemm::{GemmKernels, GemmRoute};
use crate::head::Head;
use crate::kquant::Act;
use crate::linear::conv_res::ConvResArgs;
use crate::linear::out_gate::OutGateArgs;
use crate::loop_control::{LoopControlArgs, MAX_MODULES, scratch_len};
use crate::model::GpuModel;
use crate::q38::EmbedIo;
use crate::qk_clip::QkClipArgs;
use crate::rel_bias::{REL, RelBiasArgs};
use crate::rel_bias_prefill::{RelBiasPrefillKernels, RelPrefillArgs};
use crate::ring_gemm::Rows;
use crate::routed_block::{RouteIn, RoutedRows};
use crate::site::{self, SiteTy, WideActs, WideKernels};
use crate::tensor::{Window, WindowMut, copy_row};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use chain::linear::Ring;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::inkling2::hparams::Hparams;
use model::arch::inkling2::names;
use model::arch::inkling2::place::{self, PassShape};

#[path = "rows.rs"]
pub(super) mod rows;
use model::arch::models::OutGate;
use model::placement::accepts::{self, Program};

const CALL: &str = "inkling2 prompt pass";
/// The message mix's rows: q, ffn, k and v.
const MSGS: usize = 4;

/// The prompt pass's shape for `hp` over `file`'s site types ([`place::pass_shape`]).
pub(super) fn shape_of(hp: &Hparams, file: &Split) -> Result<PassShape, GpuError> {
    place::pass_shape(hp, |name| file.find(name).map(|(_, t)| t.ty))
        .map_err(|e| GpuError::plan(WHAT, e))
}

/// The pass's kernels beside the step's.
struct PassKernels {
    prefill: RelBiasPrefillKernels,
    gemm: GemmKernels,
    slots: CardSlotKernels,
}

/// The message path's rows, a looped trunk's alone.
struct LoopRows {
    msg: [DeviceBuffer<f32>; MSGS],
    bank_last: DeviceBuffer<f32>,
    prelude: DeviceBuffer<f32>,
    /// The message bank: the banks a pass banked past its first, a pass's rows apart.
    msg_bank: DeviceBuffer<f32>,
    /// The controllers' down rows.
    h: DeviceBuffer<f32>,
}

/// The pass's buffers, each `rows` rows: the terms of [`place::pass_bytes`] in its order.
struct Bufs {
    /// The stream; `o` is the embedding's rows before the stream's norm.
    x: DeviceBuffer<f32>,
    mix: [DeviceBuffer<f32>; 3],
    xn: [DeviceBuffer<f32>; 3],
    o: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    /// The feed-forward mix, and the head's mix after the last visit.
    fmix: DeviceBuffer<f32>,
    fxn: DeviceBuffer<f32>,
    /// The routed sum, and the ffn controller's output before it.
    moe: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    /// The banked block inputs: bank `i`'s row `t` at `(i · rows + t) · n`.
    banks: DeviceBuffer<f32>,
    lp: Option<LoopRows>,
    q: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    k_raw: DeviceBuffer<f32>,
    v_raw: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    k_taps: DeviceBuffer<f32>,
    v_taps: DeviceBuffer<f32>,
    /// The global visits' factor a row, and the windowed visits' 1.0 a row.
    tau: DeviceBuffer<f32>,
    ones: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    weights: DeviceBuffer<f32>,
    /// Every slot's down row, the slots of a row together.
    ys: DeviceBuffer<f32>,
    /// The ids (`[0, rows)`), the windowed visits' ring slots (`[rows, 2 · rows)`) and the
    /// pass's first position (`2 · rows`).
    words: DeviceBuffer<u32>,
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
    picks: DeviceBuffer<u32>,
    wide_n: WideActs,
    wide_q: WideActs,
    dense: GemmRoute,
    /// The routed rows' inputs in the gate·up's form and the SwiGLU rows in the down's.
    x_rows: Rows,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    h_rows: Rows,
    route: GemmRoute,
}

impl Bufs {
    fn new(
        stream: &CudaStream,
        s: &PassShape,
        rows: usize,
        (inner, outer): (&[SiteTy], &[SiteTy]),
    ) -> Result<Bufs, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len)?)
        };
        let w = |len: usize| -> Result<DeviceBuffer<u32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len)?)
        };
        let (n, q, kv) = (s.n, s.n_head * s.head, s.n_kv * s.head);
        let rn = rows * n;
        let slots = rows * s.n_used;
        let lp = if s.looped {
            Some(LoopRows {
                msg: [z(rn)?, z(rn)?, z(rn)?, z(rn)?],
                bank_last: z(rn)?,
                prelude: z(rn)?,
                msg_bank: z(s.banks * rn)?,
                h: z(scratch_len(MAX_MODULES, rows, s.rank))?,
            })
        } else {
            None
        };
        Ok(Bufs {
            x: z(rn)?,
            mix: [z(rn)?, z(rn)?, z(rn)?],
            xn: [z(rn)?, z(rn)?, z(rn)?],
            o: z(rn)?,
            attn_out: z(rn)?,
            ffn_inp: z(rn)?,
            fmix: z(rn)?,
            fxn: z(rn)?,
            moe: z(rn)?,
            ffn_out: z(rn)?,
            banks: z(s.banks * rn)?,
            lp,
            q: z(rows * q)?,
            gate: z(rows * q)?,
            attn: z(rows * q)?,
            gated: z(rows * q)?,
            k_raw: z(rows * kv)?,
            v_raw: z(rows * kv)?,
            k: z(rows * kv)?,
            v: z(rows * kv)?,
            r: z(rows * s.n_head * s.rel)?,
            sinks: z(rows * s.n_head)?,
            k_taps: z(rows * s.n_kv * s.taps)?,
            v_taps: z(rows * s.n_kv * s.taps)?,
            tau: z(rows)?,
            ones: DeviceBuffer::from_host(stream, &vec![1.0f32; rows])?,
            probs: z(rows * s.n_expert)?,
            weights: z(slots)?,
            ys: z(slots * n)?,
            words: w(2 * rows + 1)?,
            pos: w(rows)?,
            n_keys: w(rows)?,
            picks: w(slots)?,
            wide_n: WideActs::new(stream, WideActs::wants_of(inner).without_f32(), rows, n)?,
            wide_q: WideActs::new(stream, WideActs::wants_of(outer).without_f32(), rows, q)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            x_rows: Rows::new(stream, rows, n, s.forms.gate_up32)?,
            g: z(slots * s.ff)?,
            u: z(slots * s.ff)?,
            h_rows: Rows::new(stream, slots, s.ff, s.forms.down32)?,
            route: GemmRoute::new(stream, slots, s.n_expert)?,
        })
    }

    /// Device bytes the buffers take: [`place::pass_bytes`]'s terms.
    fn bytes(&self) -> usize {
        let f32s = [
            &self.x,
            &self.o,
            &self.attn_out,
            &self.ffn_inp,
            &self.fmix,
            &self.fxn,
            &self.moe,
            &self.ffn_out,
            &self.banks,
            &self.q,
            &self.gate,
            &self.attn,
            &self.gated,
            &self.k_raw,
            &self.v_raw,
            &self.k,
            &self.v,
            &self.r,
            &self.sinks,
            &self.k_taps,
            &self.v_taps,
            &self.tau,
            &self.ones,
            &self.probs,
            &self.weights,
            &self.ys,
            &self.g,
            &self.u,
        ];
        let lp = self.lp.as_ref().map_or(0, |lp| {
            lp.msg.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
                + lp.bank_last.num_bytes()
                + lp.prelude.num_bytes()
                + lp.msg_bank.num_bytes()
                + lp.h.num_bytes()
        });
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self
                .mix
                .iter()
                .chain(&self.xn)
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + lp
            + [&self.words, &self.pos, &self.n_keys, &self.picks]
                .iter()
                .map(|b| b.num_bytes())
                .sum::<usize>()
            + self.wide_n.bytes()
            + self.wide_q.bytes()
            + self.dense.bytes()
            + self.x_rows.bytes()
            + self.h_rows.bytes()
            + self.route.bytes()
    }
}

/// Teacher-forced picks of one call: every visit's routed ids and weights at each of the call's
/// `rows` positions, visit-major then position-major, the router's top count a position.
struct PassForced {
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    rows: usize,
}

/// The prompt pass's buffers and kernels: one set a load, [`place::pass_bytes`] at `rows`.
pub(super) struct Pass {
    rows: usize,
    made: usize,
    k: PassKernels,
    b: Bufs,
    forced: Option<PassForced>,
}

impl Pass {
    /// The pass of shape `s` at `rows` rows over `layers`' sites, its routed rows in the form the
    /// stacks' route GEMMs read ([`PassShape::forms`]). Load-time only. Refused by name: buffers
    /// past the formula's bytes.
    pub(super) fn new(
        gpu: &Gpu,
        (s, rows): (&PassShape, usize),
        layers: &[Layer],
    ) -> Result<Pass, GpuError> {
        let inner: Vec<SiteTy> = layers
            .iter()
            .flat_map(|l| {
                let mut tys = vec![l.q.ty, l.k.ty, l.v.ty, l.rel.ty, l.sink.ty, l.gate.ty];
                tys.extend(l.taps.iter().flat_map(|t| t.iter().map(|s| s.ty)));
                tys
            })
            .collect();
        let outer: Vec<SiteTy> = layers.iter().map(|l| l.o.ty).collect();
        let b = Bufs::new(gpu.stream(), s, rows, (&inner, &outer))?;
        let made = b.bytes();
        let formula = place::pass_bytes(s, rows);
        if u64::try_from(made).map_or(true, |m| m > formula) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the pass's buffers take {made} B; its formula (pass_bytes) counts {formula}"
                ),
            ));
        }
        let ctx = gpu.context();
        Ok(Pass {
            rows,
            made,
            k: PassKernels {
                prefill: RelBiasPrefillKernels::load(ctx)?,
                gemm: GemmKernels::load(ctx)?,
                slots: CardSlotKernels::load(ctx)?,
            },
            b,
            forced: None,
        })
    }

    /// Device bytes the pass's buffers take.
    pub(super) fn bytes(&self) -> usize {
        self.made
    }
}

impl Bufs {
    /// The pass's words: its ids, the windowed visits' ring slots under rings of `ring` rows (0:
    /// none rings), its first position; and every row's global factor. Copies on `stream`.
    fn stage(
        &mut self,
        stream: &CudaStream,
        (ids, pos0, rows): (&[u32], u32, usize),
        (ring, floor, alpha): (u32, u32, f32),
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let at = |t: usize| pos0 + t as u32;
        WindowMut::<u32>::of_mut(&mut self.words, 0, m)?.copy_from_host(stream, ids)?;
        if ring > 0 {
            let slots: Vec<u32> = (0..m).map(|t| at(t) % ring).collect();
            WindowMut::<u32>::of_mut(&mut self.words, rows * 4, m)?
                .copy_from_host(stream, &slots)?;
        }
        WindowMut::<u32>::of_mut(&mut self.words, 2 * rows * 4, 1)?
            .copy_from_host(stream, &[pos0])?;
        let tau: Vec<f32> = (0..m).map(|t| state::tau_of(at(t), floor, alpha)).collect();
        WindowMut::<f32>::of_mut(&mut self.tau, 0, m)?.copy_from_host(stream, &tau)?;
        Ok(())
    }
}

/// One pass of a call: its ids, its first position, its first row in the call, whether it is
/// the call's last, and the call's rows whose logits a gate keeps (this pass's rows of them).
struct PassAt<'c> {
    ids: &'c [u32],
    pos0: u32,
    first: usize,
    last: bool,
    scored: Vec<usize>,
}

impl GpuModel<Body> {
    /// The prompt `ids` through the model from the stores' position as passes of up to
    /// [`place::PASS_ROWS`] rows: the stores and the position stand where `ids.len()` steps leave
    /// them, and the return is the greedy id after the last row (its logits are the model's,
    /// [`GpuModel::logits`]). A position the stores do not stand at, a context the rows pass, a
    /// call the armed forced picks were not sized for and a fault are refused by name; a call
    /// that fails part-way leaves the stores to a reset.
    pub fn prompt_pass(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        self.prompt_pass_scored(ids, &[]).map(|(id, _)| id)
    }

    /// [`GpuModel::prompt_pass`], keeping the logits rows at the call's rows `scored` (each a
    /// head launch and a blocking read): a gate's view of the rows.
    pub fn prompt_pass_scored(
        &mut self,
        ids: &[u32],
        scored: &[usize],
    ) -> Result<(u32, Vec<Vec<f32>>), GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(CALL, "a prompt of no id"));
        }
        if let Some(bad) = scored.iter().find(|&&r| r >= ids.len()) {
            return Err(GpuError::shape(
                CALL,
                format!("a kept row {bad} of a call of {} ids", ids.len()),
            ));
        }
        let n = u32::try_from(ids.len()).map_err(|_| GpuError::shape(CALL, "ids past u32"))?;
        let mut kept = Vec::with_capacity(scored.len());
        let token = self.run_rows(ids.len(), CALL, |gpu, w, body, head, pos| {
            if pos != body.held {
                return Err(GpuError::shape(
                    CALL,
                    format!(
                        "a call at position {pos}; the stores stand at {} (reset after a call \
                         that failed part-way)",
                        body.held
                    ),
                ));
            }
            let pass = body.pass.as_ref().ok_or(GpuError::state(
                CALL,
                "a whole-card load (a placed load's prompt is stepped)",
            ))?;
            if let Some(f) = &pass.forced
                && f.rows != ids.len()
            {
                return Err(GpuError::shape(
                    CALL,
                    format!(
                        "forced picks of {} positions under a call of {}",
                        f.rows,
                        ids.len()
                    ),
                ));
            }
            body.held = u32::MAX;
            let rows = pass.rows;
            let passes = ids.len().div_ceil(rows);
            for (i, chunk) in ids.chunks(rows).enumerate() {
                let first = i * rows;
                let at = PassAt {
                    ids: chunk,
                    pos0: pos + first as u32,
                    first,
                    last: i + 1 == passes,
                    scored: scored
                        .iter()
                        .filter(|&&r| (first..first + chunk.len()).contains(&r))
                        .map(|r| r - first)
                        .collect(),
                };
                body.run_pass(gpu, w, head, &at, &mut kept)?;
            }
            body.held = pos + n;
            body.end = body.end.max(body.held);
            Ok(true)
        })?;
        let token =
            token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))?;
        Ok((token, kept))
    }
}

impl Body {
    /// Rows a pass holds.
    #[must_use]
    pub fn pass_rows(&self) -> usize {
        self.pass.as_ref().map_or(0, |p| p.rows)
    }

    /// Device bytes the prompt pass's buffers take: what a plan reserves for it
    /// ([`place::pass_bytes`]).
    #[must_use]
    pub fn pass_made_bytes(&self) -> usize {
        self.pass.as_ref().map_or(0, Pass::bytes)
    }

    /// Route every visit's rows of the next prompt pass by `ids` and `weights` (visit-major, then
    /// the call's `rows` positions, the router's top count a position, the weights as the router
    /// leaves them) instead of the router's picks, until [`Body::unforce_pass_picks`]: a gate's
    /// teacher-forced pass. Allocates. Refused by name: other lengths, an id past the experts.
    pub fn force_pass_picks(
        &mut self,
        gpu: &Gpu,
        (ids, weights): (&[u32], &[f32]),
        rows: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "inkling2::Body::force_pass_picks";
        let len = self.visits.len() * rows * N_USED;
        if ids.len() != len || weights.len() != len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} ids and {} weights; {} visits of {rows} positions take {len} each",
                    ids.len(),
                    weights.len(),
                    self.visits.len()
                ),
            ));
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.d.n_expert) {
            return Err(GpuError::shape(
                WHAT,
                format!("expert {bad} of {}", self.d.n_expert),
            ));
        }
        let stream = gpu.stream();
        let pass = self
            .pass
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a whole-card load"))?;
        pass.forced = Some(PassForced {
            ids: DeviceBuffer::from_host(stream, ids)?,
            weights: DeviceBuffer::from_host(stream, weights)?,
            rows,
        });
        Ok(())
    }

    /// Route the passes by the router's picks again ([`Body::force_pass_picks`]).
    pub fn unforce_pass_picks(&mut self) {
        if let Some(p) = self.pass.as_mut() {
            p.forced = None;
        }
    }

    /// One pass of a call: the words staged, the visits over its rows, the head's mix, the kept
    /// rows' logits and, on the call's last pass, the head over its last row.
    fn run_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        at: &PassAt<'_>,
        kept: &mut Vec<Vec<f32>>,
    ) -> Result<(), GpuError> {
        let pass = self
            .pass
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a whole-card load"))?;
        let (rows, ring) = (pass.rows, self.ring);
        let (floor, alpha) = (self.hp.log_floor, self.hp.log_alpha);
        pass.b
            .stage(gpu.stream(), (at.ids, at.pos0, rows), (ring, floor, alpha))?;
        self.walk_rows(gpu, w, head, (at.ids.len(), Tail::Pass { at, kept }))
    }

    /// The visits over the `m` staged rows, then `tail`'s head.
    fn walk_rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut Head,
        (m, tail): (usize, Tail<'_, '_>),
    ) -> Result<(), GpuError> {
        let Body {
            hp,
            d,
            visits,
            layers,
            stores,
            k,
            messages,
            taps,
            pass,
            ctx,
            s,
            verify,
            ..
        } = self;
        let Pass {
            rows,
            k: pk,
            b,
            forced,
            ..
        } = pass
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a whole-card load"))?;
        let (forced, verify) = match &tail {
            Tail::Pass { at, .. } => (forced.as_ref().map(|f| (f, at.first)), None),
            Tail::Verify => (
                None,
                Some(
                    verify
                        .as_mut()
                        .ok_or(GpuError::state(WHAT, "a whole-card load"))?,
                ),
            ),
        };
        let mut r = RowWalk {
            gpu,
            w,
            k,
            pk,
            b,
            forced,
            verify,
            d: *d,
            m,
            cap: *rows,
            ctx: *ctx,
            eps: hp.rms_eps,
            clips: (hp.qk_clip, hp.rel_clip),
            act: Act::Situ {
                beta: hp.situ.gate_beta,
                linear_beta: hp.situ.value_beta,
            },
            route_scale: hp.weights_scale,
            mix_scale: 1.0 / (d.n as f32).sqrt(),
            block: hp.block,
            pushed: 0,
            prelude: 0,
            n_msg: 0,
            banked: false,
        };
        r.embed()?;
        let core = hp.loop_.filter(|lp| lp.passes > 1);
        for (slot, v) in visits.iter().enumerate() {
            let l = v.layer as usize;
            let layer = &layers[l];
            if let Some(lp) = core
                && l == lp.core_start
            {
                r.pass_edge(v.pass)?;
            }
            let msg = *messages && v.pass > 0 && layer.ctrl.is_some();
            r.attn_half(slot, (l, layer), &mut stores[slot], msg)?;
            r.ffn_front(slot, layer, msg)?;
            r.ffn_shadow(slot, layer)?;
            r.ffn_back(slot, layer, &mut stores[slot])?;
            if let Some(t) = taps.as_mut() {
                r.tap(t, slot)?;
            }
        }
        let logits = (hp.softcap, hp.n_vocab_unpadded, hp.n_vocab);
        match tail {
            Tail::Pass { at, kept } => r.finish((visits.len(), &s.pad), head, logits, at, kept),
            Tail::Verify => r.finish_rows((visits.len(), &s.pad), head, logits),
        }
    }
}

/// What a walk over rows ends with: a prompt pass's head over its kept rows and, on the call's
/// last pass, its last row; or a verify's head over every row ([`rows`]).
enum Tail<'t, 'c> {
    Pass {
        at: &'t PassAt<'c>,
        kept: &'t mut Vec<Vec<f32>>,
    },
    Verify,
}

/// One walk's borrowed parts and its counters (the step's [`super::Walk`]'s, over rows).
struct RowWalk<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    k: &'a Kernels,
    pk: &'a PassKernels,
    b: &'a mut Bufs,
    /// The forced picks and this pass's first row in the call.
    forced: Option<(&'a PassForced, usize)>,
    /// A verify's own buffers; none on a prompt pass.
    verify: Option<&'a mut rows::VerifyBufs>,
    d: Dims,
    /// The pass's rows, and the rows the buffers hold (the banks' pitch).
    m: usize,
    cap: usize,
    ctx: usize,
    eps: f32,
    clips: (f32, f32),
    act: Act,
    route_scale: f32,
    mix_scale: f32,
    block: usize,
    pushed: usize,
    prelude: usize,
    n_msg: usize,
    banked: bool,
}

/// `src`'s values `[from, from + len)` into `dst` from `to`, one copy; nothing for none.
fn copy_span(
    stream: &CudaStream,
    (src, from): (&DeviceBuffer<f32>, usize),
    (dst, to): (&mut DeviceBuffer<f32>, usize),
    len: usize,
) -> Result<(), GpuError> {
    if len == 0 {
        return Ok(());
    }
    let src = Window::<f32>::of(src, from * 4, len)?;
    let mut dst = WindowMut::<f32>::of_mut(dst, to * 4, len)?;
    copy_row(stream, &src, &mut dst, len)
}

/// What every normed mix of a visit shares: the kernels, the rows and the banks' pitch, the
/// score scale, the visit's fault sink.
#[derive(Clone, Copy)]
struct MixAt<'a> {
    k: &'a AttnResKernels,
    stream: &'a CudaStream,
    n: usize,
    m: usize,
    pitch: usize,
    scale: f32,
    fault: FaultSink,
}

impl MixAt<'_> {
    /// Query `j` of `queries`' normed mix of `n_banks` banks and the rows `cur` into `y`: one
    /// launch (a query's rows are its own, so one query a launch is the step's bits).
    fn mix(
        &self,
        (banks, n_banks): (&DeviceBuffer<f32>, usize),
        cur: &DeviceBuffer<f32>,
        (queries, j): (&DeviceBuffer<f32>, usize),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let query = Window::<f32>::of(queries, j * self.n * 4, self.n)?;
        self.k.enqueue_normed(
            self.stream,
            AttnResNormedArgs {
                banks,
                n_banks,
                stride: self.pitch,
                cur,
                queries: &query,
                nq: 1,
                scale: self.scale,
                eps: MIX_EPS,
                n: self.n,
                m: self.m,
                fault: self.fault,
            },
            y,
        )
    }
}

impl<'a> RowWalk<'a> {
    /// The visit's mixes' shared arguments under `fault`.
    fn mixes(&self, fault: FaultSink) -> MixAt<'a> {
        let (k, gpu): (&'a Kernels, &'a Gpu) = (self.k, self.gpu);
        MixAt {
            k: &k.mix,
            stream: gpu.stream(),
            n: self.d.n,
            m: self.m,
            pitch: self.cap * self.d.n,
            scale: self.mix_scale,
            fault,
        }
    }

    /// The rows' embeddings, normed by the embedding norm into the stream, each row's position
    /// and live key count beside them; the dense table over the rows.
    fn embed(&mut self) -> Result<(), GpuError> {
        let (gpu, w, m, n) = (self.gpu, self.w, self.m, self.d.n);
        let b = &mut *self.b;
        let ids = Window::<u32>::of(&b.words, 0, m)?;
        let pos0 = Window::<u32>::of(&b.words, 2 * self.cap * 4, 1)?;
        self.k.q38.enqueue_embed_lookup(
            gpu,
            CALL,
            (w, "token_embd.weight"),
            EmbedIo {
                ids: &ids,
                pos0: &pos0,
                first: 0,
                width: n,
                y: &mut b.o,
                pos: &mut b.pos,
                n_keys: &mut b.n_keys,
            },
        )?;
        gpu.elem().enqueue_rms_norm(
            gpu.stream(),
            &b.o,
            w.f32_buf(CALL, names::TOKEN_EMBD_NORM)?,
            self.eps,
            n,
            m,
            &mut b.x,
        )?;
        if self.verify.is_some() {
            return Ok(());
        }
        self.pk
            .gemm
            .enqueue_route_dense(gpu.stream(), m, &mut b.dense, gpu.unlabelled_sink())
    }

    /// The step's pass edge over the rows ([`super::Walk::pass_edge`]).
    fn pass_edge(&mut self, pass: u32) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        let mn = self.m * self.d.n;
        let pitch = self.cap * self.d.n;
        let b = &mut *self.b;
        let lp =
            b.lp.as_mut()
                .ok_or(GpuError::state(CALL, "a looped trunk's message rows"))?;
        if pass == 0 {
            self.prelude = self.pushed;
            return copy_span(stream, (&b.x, 0), (&mut lp.prelude, 0), mn);
        }
        let first = self.prelude + 1;
        if self.pushed < first {
            return Err(GpuError::shape(
                CALL,
                format!(
                    "pass {pass} after {} banked blocks over a prelude of {}: a pass banks its \
                     first layer's input",
                    self.pushed, self.prelude
                ),
            ));
        }
        self.n_msg = self.pushed - first;
        copy_span(
            stream,
            (&b.banks, first * pitch),
            (&mut lp.msg_bank, 0),
            self.n_msg * pitch,
        )?;
        copy_span(stream, (&b.x, 0), (&mut lp.bank_last, 0), mn)?;
        self.pushed = self.prelude;
        copy_span(stream, (&lp.prelude, 0), (&mut b.x, 0), mn)
    }

    /// The visit's attention half over the rows ([`super::Walk::attn_half`]): where the messages
    /// run, the controllers write the inputs' rows into `xn` and the norms into `mix`; elsewhere
    /// the norms read the mix (or the stream) into `xn`.
    #[allow(
        clippy::too_many_lines,
        reason = "one sublayer in the reference's order: the messages, the inputs, the \
                  projections, the convs, the epilogue, the attention and the join"
    )]
    fn attn_half(
        &mut self,
        slot: usize,
        (l, layer): (usize, &Layer),
        store: &mut VisitStore,
        msg: bool,
    ) -> Result<(), GpuError> {
        let (gpu, w, k, pk) = (self.gpu, self.w, self.k, self.pk);
        let stream = gpu.stream();
        let d = self.d;
        let (n, m, cap) = (d.n, self.m, self.cap);
        let (qw, kvw) = (d.n_head * HEAD, d.n_kv * HEAD);
        let fault = gpu.layer_sink(slot)?;
        let mx = self.mixes(fault);
        let ctrl = match (msg, layer.ctrl.as_ref()) {
            (true, Some(c)) => Some(c),
            (true, None) => return Err(GpuError::state(CALL, "a core layer's controllers")),
            (false, _) => None,
        };
        let (pushed, n_msg) = (self.pushed, self.n_msg);
        let wc = WideCx {
            what: CALL,
            gpu,
            w,
            wk: WideKernels {
                gemm: &pk.gemm,
                g32: Some(&k.g32),
            },
            m,
        };
        let Bufs {
            x,
            mix,
            xn,
            banks,
            lp,
            q,
            gate,
            attn,
            gated,
            k_raw,
            v_raw,
            k: kk,
            v: vv,
            r,
            sinks,
            k_taps,
            v_taps,
            tau,
            ones,
            words,
            pos,
            n_keys,
            wide_n,
            wide_q,
            dense,
            o,
            attn_out,
            ffn_inp,
            ..
        } = &mut *self.b;
        // The message mixes (q, ffn, k, v) over the bank, then the attention mix's queries.
        if let Some(c) = ctrl {
            let LoopRows {
                msg: msgs,
                bank_last,
                msg_bank,
                ..
            } = lp
                .as_mut()
                .ok_or(GpuError::state(CALL, "a looped trunk's message rows"))?;
            for (j, y) in msgs.iter_mut().enumerate() {
                mx.mix((msg_bank, n_msg), bank_last, (&c.queries, j), y)?;
            }
        }
        if let Some((query, nq)) = &layer.attn_query {
            let queries = w.f32_buf(CALL, query)?;
            for (j, y) in mix.iter_mut().enumerate().take(*nq) {
                mx.mix((banks, pushed), x, (queries, j), y)?;
            }
        }
        let n_in = if layer.windowed && ctrl.is_none() {
            1
        } else {
            3
        };
        let input = |mix: &'_ [DeviceBuffer<f32>; 3], j: usize| -> usize {
            let _ = mix;
            if layer.windowed { 0 } else { j }
        };
        if let Some(c) = ctrl {
            let LoopRows { msg: msgs, h, .. } = lp
                .as_mut()
                .ok_or(GpuError::state(CALL, "a looped trunk's message rows"))?;
            {
                let cur = |j: usize| -> &DeviceBuffer<f32> {
                    match layer.attn_query {
                        Some(_) => &mix[input(mix, j)],
                        None => x,
                    }
                };
                let args = |j: usize| -> Result<LoopControlArgs<'_>, GpuError> {
                    let [down, gain, forget] = &c.w[j];
                    Ok(LoopControlArgs {
                        cur: cur(j),
                        msg: &msgs[MSG_ATTN[j]],
                        down: w.f32_buf(CALL, down)?,
                        gain: w.f32_buf(CALL, gain)?,
                        forget: w.f32_buf(CALL, forget)?,
                        n,
                        rank: d.rank,
                        m,
                        eps: MIX_EPS,
                        fault,
                    })
                };
                let [y0, y1, y2] = &mut *xn;
                k.ctrl.enqueue_attn_rows(
                    stream,
                    [args(0)?, args(1)?, args(2)?],
                    h,
                    [y0, y1, y2],
                )?;
            }
            for (j, y) in mix.iter_mut().enumerate() {
                gpu.elem().enqueue_rms_norm(
                    stream,
                    &xn[j],
                    w.f32_buf(CALL, &layer.norms[j])?,
                    self.eps,
                    n,
                    m,
                    y,
                )?;
            }
        } else {
            for (j, y) in xn.iter_mut().enumerate().take(n_in) {
                let src = match layer.attn_query {
                    Some(_) => &mix[input(mix, j)],
                    None => &*x,
                };
                gpu.elem().enqueue_rms_norm(
                    stream,
                    src,
                    w.f32_buf(CALL, &layer.norms[j])?,
                    self.eps,
                    n,
                    m,
                    y,
                )?;
            }
        }
        // The projections of the normed inputs, one wide GEMM a site over the rows: q, the
        // relative rows, the sinks and the gate read the q input, k and its taps the k input, v
        // and its taps the v input (one input's quantization where the three are one).
        let normed: &[DeviceBuffer<f32>; 3] = if ctrl.is_some() { mix } else { xn };
        let tw = d.n_kv * d.taps;
        let dense = &*dense;
        if self.verify.is_some() {
            // The step's sites (`norm_proj`'s outputs are these gemvs' bits) over the rows.
            let (ik, iv) = if n_in == 1 { (0, 0) } else { (1, 2) };
            let gemv = |site: &Site, x: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>| {
                site::gemv(gpu, &k.g32, (site.ty, w, &site.name), x, m, y)
            };
            gemv(&layer.q, &normed[0], q)?;
            gemv(&layer.k, &normed[ik], k_raw)?;
            gemv(&layer.v, &normed[iv], v_raw)?;
            gemv(&layer.rel, &normed[0], r)?;
            gemv(&layer.sink, &normed[0], sinks)?;
            gemv(&layer.gate, &normed[0], gate)?;
            if let Some([tk, tv]) = &layer.taps {
                gemv(tk, &normed[ik], k_taps)?;
                gemv(tv, &normed[iv], v_taps)?;
            }
        } else {
            let mut qs = vec![
                (&layer.q, qw, &mut *q),
                (&layer.rel, d.n_head * REL, &mut *r),
                (&layer.sink, d.n_head, &mut *sinks),
                (&layer.gate, qw, &mut *gate),
            ];
            let mut ks = vec![(&layer.k, kvw, &mut *k_raw)];
            let mut vs = vec![(&layer.v, kvw, &mut *v_raw)];
            if let Some([tk, tv]) = &layer.taps {
                ks.push((tk, tw, &mut *k_taps));
                vs.push((tv, tw, &mut *v_taps));
            }
            if n_in == 1 {
                qs.extend(ks);
                qs.extend(vs);
                project_wide(&wc, (wide_n, dense), slot, &normed[0], &mut qs)?;
            } else {
                project_wide(&wc, (wide_n, dense), slot, &normed[0], &mut qs)?;
                project_wide(&wc, (wide_n, dense), slot, &normed[1], &mut ks)?;
                project_wide(&wc, (wide_n, dense), slot, &normed[2], &mut vs)?;
            }
        }
        // The k and v short convs over the rows on the visit's ring.
        let ring = Ring::new(d.taps);
        let (heads, dynamic) = if layer.windowed {
            (0, false)
        } else {
            (d.n_kv, true)
        };
        for (j, (xin, taps, y)) in [(&*k_raw, &*k_taps, &mut *kk), (&*v_raw, &*v_taps, &mut *vv)]
            .into_iter()
            .enumerate()
        {
            k.conv.enqueue(
                stream,
                ConvResArgs {
                    ring,
                    d: kvw,
                    base: d.bases[j],
                    ch: d.ch,
                    heads,
                    m,
                    x: xin,
                    weights: w.f32_buf(CALL, &layer.convs[j])?,
                    dynamic: taps,
                    pos,
                    fault,
                    scale: conv_scale(d.taps, dynamic),
                    y,
                    state: &mut store.conv,
                },
            )?;
        }
        // The epilogue appends the rows' keys at their positions (a ring's slots) before the
        // prefill flash reads them.
        {
            let slots = Window::<u32>::of(words, cap * 4, m)?;
            let at: &DeviceBuffer<u32> = if store.shape.ring { &slots } else { pos };
            k.clip.enqueue(
                stream,
                QkClipArgs {
                    q,
                    k: kk,
                    v: vv,
                    r,
                    gq: w.f32_buf(CALL, &layer.q_clip)?,
                    gr: w.f32_buf(CALL, &layer.rel_clip)?,
                    tau: if layer.windowed { ones } else { tau },
                    pos: at,
                    v_eps: self.eps,
                    qk_thresh: self.clips.0,
                    r_thresh: self.clips.1,
                    n_head: d.n_head,
                    n_kv: d.n_kv,
                    ctx: store.shape.rows,
                    m,
                    fault,
                    cache_k: &mut store.kc,
                    cache_v: &mut store.vc,
                },
            )?;
        }
        let (scale, ring_rows) = (
            1.0 / (HEAD as f32).sqrt(),
            store.shape.ring.then_some(store.shape.rows),
        );
        let window = if layer.windowed { d.window } else { 0 };
        match self.verify.as_deref_mut() {
            Some(v) => k.rel.enqueue(
                stream,
                RelBiasArgs {
                    q,
                    kc: &store.kc,
                    vc: &store.vc,
                    n_keys,
                    r,
                    bias: &layer.bias,
                    extent: layer.extent,
                    scale,
                    n_kv: d.n_kv,
                    ctx: self.ctx,
                    ring_rows,
                    m,
                    window,
                    sinks,
                    part_v: &mut v.part_v,
                    part_ms: &mut v.part_ms,
                    fault,
                    y: attn,
                },
                d.n_head,
            )?,
            None => pk.prefill.enqueue(
                stream,
                RelPrefillArgs {
                    q,
                    kc: &store.kc,
                    vc: &store.vc,
                    n_keys,
                    r,
                    bias: &layer.bias,
                    extent: layer.extent,
                    scale,
                    n_kv: d.n_kv,
                    ctx: self.ctx,
                    ring_rows,
                    m,
                    window,
                    sinks,
                    fault,
                    y: attn,
                },
                d.n_head,
            )?,
        }
        k.out_gate.enqueue_out_gate(
            stream,
            OutGateArgs {
                kind: OutGate::Proj,
                attn,
                gate,
                n_head: d.n_head,
                m,
                fault,
                y: gated,
            },
        )?;
        if self.verify.is_some() {
            site::gemv(gpu, &k.g32, (layer.o.ty, w, &layer.o.name), gated, m, o)?;
        } else {
            project_wide(
                &wc,
                (wide_q, dense),
                slot,
                gated,
                &mut [(&layer.o, n, &mut *o)],
            )?;
        }
        k.conv.enqueue(
            stream,
            ConvResArgs {
                ring,
                d: n,
                base: d.bases[2],
                ch: d.ch,
                heads: 0,
                m,
                x: o,
                weights: w.f32_buf(CALL, &layer.convs[2])?,
                dynamic: k_taps,
                pos,
                fault,
                scale: conv_scale(d.taps, false),
                y: attn_out,
                state: &mut store.conv,
            },
        )?;
        // The block join: a block's first layer banks the stream's rows and its attention output
        // is the new stream; any other adds the two.
        self.banked = l % self.block == 0;
        if self.banked {
            copy_span(stream, (x, 0), (banks, pushed * cap * n), m * n)?;
            self.pushed += 1;
            Ok(())
        } else {
            gpu.elem().enqueue_add(stream, x, attn_out, m * n, ffn_inp)
        }
    }

    /// The feed-forward block's front over the rows: the mix, the ffn controller where the
    /// messages run (into `moe`, free until the routed sum), the norm and the router's rows.
    fn ffn_front(&mut self, slot: usize, layer: &Layer, msg: bool) -> Result<(), GpuError> {
        let (gpu, w, k) = (self.gpu, self.w, self.k);
        let stream = gpu.stream();
        let (n, m) = (self.d.n, self.m);
        let fault = gpu.layer_sink(slot)?;
        let mx = self.mixes(fault);
        let Bufs {
            banks,
            attn_out,
            ffn_inp,
            fmix,
            fxn,
            moe,
            lp,
            probs,
            weights,
            picks,
            ..
        } = &mut *self.b;
        let cur = if self.banked { &*attn_out } else { &*ffn_inp };
        mx.mix(
            (banks, self.pushed),
            cur,
            (w.f32_buf(CALL, &layer.ffn_query)?, 0),
            fmix,
        )?;
        let normed_in: &DeviceBuffer<f32> = match (msg, layer.ctrl.as_ref()) {
            (true, Some(c)) => {
                let LoopRows { msg: msgs, h, .. } = lp
                    .as_mut()
                    .ok_or(GpuError::state(CALL, "a looped trunk's message rows"))?;
                let [down, gain, forget] = &c.w[3];
                k.ctrl.enqueue_rows(
                    stream,
                    LoopControlArgs {
                        cur: fmix,
                        msg: &msgs[MSG_FFN],
                        down: w.f32_buf(CALL, down)?,
                        gain: w.f32_buf(CALL, gain)?,
                        forget: w.f32_buf(CALL, forget)?,
                        n,
                        rank: self.d.rank,
                        m,
                        eps: MIX_EPS,
                        fault,
                    },
                    h,
                    moe,
                )?;
                moe
            }
            (true, None) => return Err(GpuError::state(CALL, "a core layer's controllers")),
            (false, _) => fmix,
        };
        gpu.elem().enqueue_rms_norm(
            stream,
            normed_in,
            w.f32_buf(CALL, &layer.ffn_norm)?,
            self.eps,
            n,
            m,
            fxn,
        )?;
        let rin = RouteIn {
            w: w.f32_tensor(CALL, &layer.router)?,
            bias: Some(w.f32_buf(CALL, &layer.probs_b)?),
            scale: self.route_scale,
            fault,
        };
        k.router.route_rows(
            (stream, gpu, &k.g32),
            fxn,
            &rin,
            m,
            RoutedRows {
                probs,
                logits: None,
                ids: picks,
                weights,
            },
        )
    }

    /// The feed-forward block's shadow over the rows: the routed rows on the grouped route GEMM
    /// over the card's stacks (every expert resident, so a pick's id is its place) — the table,
    /// the rows in q8 blocks of 32, gate·up, SiTU, down — and the card sum of each row's slots.
    fn ffn_shadow(&mut self, slot: usize, layer: &Layer) -> Result<(), GpuError> {
        if self.verify.is_some() {
            return self.shadow_rows(slot, layer);
        }
        let (gpu, w, k, pk) = (self.gpu, self.w, self.k, self.pk);
        let stream = gpu.stream();
        let (n, m) = (self.d.n, self.m);
        let fault = gpu.layer_sink(slot)?;
        let arms = accepts::leg_arms(Program::Inkling2Body)
            .ok_or_else(|| GpuError::state(CALL, "the inkling2 program's card leg"))?;
        let names = layer
            .stacks
            .as_ref()
            .ok_or(GpuError::state(CALL, "the card's routed stacks"))?
            .each_ref()
            .map(String::as_str);
        let stacks = LegStacks::of_arms(w, names, arms, (GATE_UP_ROLE, DOWN_ROLE))?;
        let block = stacks.block(self.d.ff, n)?;
        let slots = m * N_USED;
        let forced = self
            .forced
            .map(|(f, first)| -> Result<_, GpuError> {
                let at = (slot * f.rows + first) * N_USED * 4;
                Ok((
                    Window::<u32>::of(&f.ids, at, slots)?,
                    Window::<f32>::of(&f.weights, at, slots)?,
                ))
            })
            .transpose()?;
        let Bufs {
            fxn,
            picks,
            weights,
            x_rows,
            g,
            u,
            h_rows,
            ys,
            route,
            moe,
            ..
        } = &mut *self.b;
        let (sel, wts): (&DeviceBuffer<u32>, &DeviceBuffer<f32>) = match &forced {
            Some((ids, ws)) => (ids, ws),
            None => (picks, weights),
        };
        pk.gemm.enqueue_route(stream, sel, slots, route, fault)?;
        x_rows.enqueue_quantize(gpu, &k.g32, fxn, m, fault)?;
        let (x128, x32) = x_rows.parts();
        let (h128, h32) = h_rows.parts_mut();
        block.enqueue_rule(
            gpu,
            (Some(&pk.gemm), &k.g32),
            (route, N_USED),
            slots,
            (
                RouteIo {
                    x128,
                    x32,
                    g,
                    u,
                    h128,
                    h32,
                    y: ys,
                },
                self.act,
            ),
            fault,
        )?;
        pk.slots.enqueue_card_acc(
            stream,
            &CardAcc {
                down: ys,
                w: wts,
                sel,
                n,
                m,
                n_card: self.d.n_expert,
                n_used: N_USED,
                w_pitch: N_USED,
                fault,
            },
            moe,
        )
    }

    /// The feed-forward block's back over the rows: the post conv of the routed sum, the
    /// stream's sum.
    fn ffn_back(
        &mut self,
        slot: usize,
        layer: &Layer,
        store: &mut VisitStore,
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let stream = gpu.stream();
        let (d, m) = (self.d, self.m);
        let b = &mut *self.b;
        self.k.conv.enqueue(
            stream,
            ConvResArgs {
                ring: Ring::new(d.taps),
                d: d.n,
                base: d.bases[3],
                ch: d.ch,
                heads: 0,
                m,
                x: &b.moe,
                weights: w.f32_buf(CALL, &layer.convs[3])?,
                dynamic: &b.k_taps,
                pos: &b.pos,
                fault: gpu.layer_sink(slot)?,
                scale: conv_scale(d.taps, false),
                y: &mut b.ffn_out,
                state: &mut store.conv,
            },
        )?;
        let ffn_inp = if self.banked { &b.attn_out } else { &b.ffn_inp };
        gpu.elem()
            .enqueue_add(stream, ffn_inp, &b.ffn_out, m * d.n, &mut b.x)
    }

    /// Visit `slot`'s [`super::TAP_STAGES`] at the pass's last row into its block of `t`.
    fn tap(&mut self, t: &mut DeviceBuffer<f32>, slot: usize) -> Result<(), GpuError> {
        let stream = self.gpu.stream();
        let b = &*self.b;
        let ffn_inp = if self.banked { &b.attn_out } else { &b.ffn_inp };
        let rows = [
            &b.q,
            &b.k,
            &b.v,
            &b.r,
            &b.attn,
            &b.attn_out,
            ffn_inp,
            &b.moe,
            &b.ffn_out,
            &b.x,
        ];
        let widths = tap_widths(&self.d);
        let block: usize = widths.iter().sum();
        let mut at = slot * block;
        for (src, w) in rows.into_iter().zip(widths) {
            copy_span(stream, (src, (self.m - 1) * w), (t, at), w)?;
            at += w;
        }
        Ok(())
    }

    /// The head's mix of the banked blocks and the stream over the rows, then the head over each
    /// kept row (its logits read back) and, on the call's last pass, over the last row.
    fn finish(
        &mut self,
        (n_visits, pad): (usize, &DeviceBuffer<f32>),
        head: &mut Head,
        logits: (f32, usize, usize),
        at: &PassAt<'_>,
        kept: &mut Vec<Vec<f32>>,
    ) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let stream = gpu.stream();
        let n = self.d.n;
        let mx = self.mixes(gpu.layer_sink(n_visits)?);
        let b = &mut *self.b;
        mx.mix(
            (&b.banks, self.pushed),
            &b.x,
            (w.f32_buf(CALL, names::OUTPUT_RES_QUERY)?, 0),
            &mut b.fmix,
        )?;
        let row = |t: usize, head: &mut Head| -> Result<(), GpuError> {
            copy_span(stream, (&b.fmix, t * n), (head.input_mut(), 0), n)?;
            head_logits((gpu, w), head, pad, logits, 1)
        };
        for &t in &at.scored {
            row(t, head)?;
            kept.push(head.logits_to_host(gpu)?);
        }
        if at.last {
            row(self.m - 1, head)?;
        }
        Ok(())
    }
}
