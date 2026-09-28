//! Qwen3.8's layer program: [`runtime::sched::walk`] over one unit of `m`
//! columns — the decode step `(1, 1, Step)` through the host tier's step port
//! ([`Step38`], captured), a verify `(1, 2 <= m <= 4, Step)` through the same
//! port's `Cols(m)` chain ([`Verify38`], captured), an eager pass `(1, m <=
//! 8, Batch)` through its batch port ([`Pass38`]). The walks share every
//! card launch ([`Parts38`]); they differ in how a layer's routed experts
//! reach the host and come back, and a verify keeps each row's delta state
//! in a lane of its own where the step and the pass write the last row's
//! back into the committed lane. A prompt's ubatch walk `(1, m, Batch)` over
//! the ubatch's rows is `wide38`'s, its launches the wide family's.
//!
//! A layer's parts, by its plan ([`Layer38`]), never its number:
//! - the front: the embedding in front of layer 0; the attention site's mix
//!   (its norm first applies the previous layer's combine in place; layer 0
//!   starts every stream from the embedding); on the PLE layer the previous
//!   combine alone, the PLE site into the other stream buffer, then a plain
//!   mix; the mixer — the gated delta rule ([`GdnPlan`]) or the selecting
//!   attention ([`QsaPlan`]); the feed-forward site's mix, its combine of the
//!   mixer's output first; the router; the host leg's send (the step's
//!   handoff and go, the pass's download);
//! - the shadow: on a layer with card experts the card leg ([`Card38`]:
//!   the routed experts the slot map puts on the card, from the places the
//!   step's and the verify's handoff wrote beside the ids, or the pass's
//!   places launch in its front), then the shared expert;
//! - the back: the join (the step's wait; the pass's serve and upload are the
//!   walk's), and the host's routed sum, the card's on a layer with card
//!   experts, plus the gated shared expert into `y`, the next site's combine
//!   input.
//!
//! After the last layer the head's mix (its norm applies the last combine)
//! writes the head's input. Launches of the captured step
//! ([`step_launches`]): the embedding; per layer the two mixes' three, the
//! mixer's ([`GDN_LAUNCHES`] or [`QSA_LAUNCHES`]), the PLE site's
//! [`PLE_LAUNCHES`] on its layer, the block's [`FFN_LAUNCHES`] — two of
//! them the go and the wait, stream memory-operation batches
//! ([`STEP_MEMOPS`] a layer) — and the card leg's [`CARD_LAUNCHES`] on a
//! layer with card experts; then the head's mix, projection and argmax.

use super::body::ATTN_SCALE_256;
use super::card38::{CARD_LAUNCHES, Card38};
use super::plan38::{GDN, GdnPlan, HcSite, Layer38, Mixer38, QsaPlan, geo, head_site};
use super::proj::ProjKernels;
use super::router::gated;
use super::scratch::{Io, f32_view};
use super::scratch38::{Arena38, Store38, Taps38};
use crate::fault::{FaultSink, LAYER_HEAD};
use crate::flash_gqa::{FlashGqaKernels, GqaSelArgs};
use crate::flash_gqa_prefill::FlashGqaPrefill;
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::hc_gated::{Before, HcGatedKernels, HcScratch, HcWideKernels, MixArgs, SiteWeights};
use crate::head::Head;
use crate::host::handoff::{Handoff, HandoffKernels, Places};
use crate::host::run::HostRun;
use crate::host::{BatchLeg, StepLeg};
use crate::linear::conv::ConvArgs;
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::linear::norm_gate::NormGateArgs;
use crate::linear::{self, LinearKernels};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::ple::{PleConvArgs, PleGateArgs, PleKernels};
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::q38::{
    CardSharedAddArgs, EmbedQ8Args, KeyAppendArgs, OutGateArgs, Q38Kernels, SharedAddArgs,
};
use crate::qsa::{PoolArgs, QsaKernels, SelectArgs};
use crate::rope_neox::{PartialNeoxArgs, RopeNeoxKernels};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

/// What the walks' errors name.
const WHAT: &str = "qwen4exp program";

/// The selected flash's pass: the tensor-core one.
const MMA: bool = true;

/// A delta layer's mixer launches at one row: the q·k·v, `z` and joined
/// β·α projections, the conv, the delta step, the gated norm and the output
/// projection.
pub(super) const GDN_LAUNCHES: usize = 7;
/// A selecting layer's mixer launches at one row: q, k and v; the indexer
/// key's projection, its append and the pool; the indexer query's projection
/// and the selection's two; the q/k norm, turn and append; the selected
/// flash's two; the gate and the output projection.
pub(super) const QSA_LAUNCHES: usize = 14;
/// The PLE site's launches: the previous combine, the key and value
/// projections, the gate and the conv.
pub(super) const PLE_LAUNCHES: usize = 5;
/// A layer's two mixes.
pub(super) const MIX_LAUNCHES: usize = 2 * 3;
/// The block's launches in the step: the router, the handoff, the go, the
/// shared expert's four, the wait and the gated sum.
pub(super) const FFN_LAUNCHES: usize = 1 + 1 + 1 + 4 + 1 + 1;
/// Of the block's launches, the stream memory-operation batches: the go and
/// the wait.
pub(super) const STEP_MEMOPS: usize = 2;
/// The head: its mix, the projection and the argmax.
pub(super) const HEAD_LAUNCHES: usize = 3 + 2;

/// A delta layer's launches past [`GDN_LAUNCHES`] at more than one row: β
/// and α copied token-major out of the joined projection.
pub(super) const GDN_ROWS_LAUNCHES: usize = 2;
/// A selecting layer's launches past [`QSA_LAUNCHES`] at more than one row:
/// the indexer queries copied token-major.
pub(super) const QSA_ROWS_LAUNCHES: usize = 1;

/// The captured decode step's launches for `plans` with `card`'s card
/// experts (module doc).
pub(super) fn step_launches(plans: &[Layer38], card: &Card38) -> usize {
    walk_launches(plans, card, 1)
}

/// The captured verify's launches for `plans` with `card`'s card experts at
/// `m >= 2` rows: the step's, plus each mixer's token-major copies
/// ([`GDN_ROWS_LAUNCHES`], [`QSA_ROWS_LAUNCHES`]); the handoff, the go, the
/// wait, the card leg and the head are one each whatever `m`.
pub(super) fn verify_launches(plans: &[Layer38], card: &Card38, m: usize) -> usize {
    walk_launches(plans, card, m)
}

fn walk_launches(plans: &[Layer38], card: &Card38, m: usize) -> usize {
    let rows = m > 1;
    1 + plans
        .iter()
        .enumerate()
        .map(|(l, p)| {
            let mixer = match p.mixer {
                Mixer38::Gdn(_) => GDN_LAUNCHES + usize::from(rows) * GDN_ROWS_LAUNCHES,
                Mixer38::Qsa(_) => QSA_LAUNCHES + usize::from(rows) * QSA_ROWS_LAUNCHES,
            };
            MIX_LAUNCHES
                + mixer
                + FFN_LAUNCHES
                + p.ple.as_ref().map_or(0, |_| PLE_LAUNCHES)
                + usize::from(card.has(l)) * CARD_LAUNCHES
        })
        .sum::<usize>()
        + HEAD_LAUNCHES
}

/// Every module a Qwen3.8 walk launches besides the `Gpu`'s own.
pub(super) struct Kernels38 {
    pub(super) q38: Q38Kernels,
    pub(super) hc: HcGatedKernels,
    pub(super) ple: PleKernels,
    pub(super) qsa: QsaKernels,
    pub(super) neox: RopeNeoxKernels,
    pub(super) flash: FlashGqaKernels,
    pub(super) linear: LinearKernels,
    pub(super) router: gated::RouterKernels,
    pub(super) handoff: HandoffKernels,
    /// The token-major copy of a row-major gemv output.
    pub(super) proj: ProjKernels,
    /// The ubatch walk's: the one-expert table's fill, the 32-value GEMM
    /// family, the prefill flash and the wide mixes.
    pub(super) gemm: GemmKernels,
    pub(super) g32: Gemm32Kernels,
    pub(super) prefill: FlashGqaPrefill,
    pub(super) hcw: HcWideKernels,
}

impl Kernels38 {
    /// Load every module into `gpu`'s context. Load-time only.
    pub(super) fn load(gpu: &Gpu) -> Result<Kernels38, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels38 {
            q38: Q38Kernels::load(ctx)?,
            hc: HcGatedKernels::load(ctx)?,
            ple: PleKernels::load(ctx)?,
            qsa: QsaKernels::load(ctx)?,
            neox: RopeNeoxKernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            router: gated::RouterKernels::load(ctx)?,
            handoff: HandoffKernels::load(ctx)?,
            proj: ProjKernels::load(ctx)?,
            gemm: GemmKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            prefill: FlashGqaPrefill::load(ctx)?,
            hcw: HcWideKernels::load(ctx)?,
        })
    }
}

/// The resident Q8_0 planes of `name`.
pub(super) fn q8<'w>(
    w: &'w Weights,
    name: &str,
) -> Result<(&'w DeviceTensor<u32>, &'w DeviceTensor<u16>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { qs, d, .. }) => Ok((qs, d)),
        Some(_) => Err(GpuError::tensor(WHAT, name, "Q8_0 (the q8f32 planes)")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}

/// What every launch of a walk reads besides the arena and the stores: the
/// engine, the weights, the kernels, the norms' epsilon, the rope table and
/// the stores' positions.
pub(super) struct Ctx38<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) k: &'a Kernels38,
    pub(super) eps: f32,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) ctx: usize,
}

impl Ctx38<'_> {
    /// `y = W · x` for the Q8_0 weight `name` over `m` columns, token-major.
    fn q8_gemv(
        &self,
        name: &str,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (qs, d) = q8(self.w, name)?;
        let (q, stream) = (self.gpu.q8f32(), self.gpu.stream());
        if m == 1 {
            q.enqueue_q8_0_gemv(stream, qs, d, x, 1, y)
        } else {
            q.enqueue_q8_0_gemv_mcol(
                stream,
                Q8_0GemvMcolArgs {
                    qs,
                    d,
                    x,
                    m,
                    out: GemvOut::TokenMajor,
                    y,
                },
            )
        }
    }

    /// `y = W · x` for the F32 weight `name` over `m` columns, row-major
    /// (`[rows][m]`).
    fn f32_gemv(
        &self,
        name: &str,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.gpu
            .q8f32()
            .enqueue_f32_gemv(self.gpu.stream(), f32_tensor(self.w, name)?, x, m, y)
    }

    /// The mix of `site` over `m` columns of `res`, after `before`, into
    /// `mixed`.
    #[allow(
        clippy::too_many_arguments,
        reason = "one site's streams, rule, width, sink and two outputs (rust-quality R8)"
    )]
    fn mix(
        &self,
        site: &HcSite,
        res: &mut DeviceBuffer<f32>,
        before: Before<'_>,
        m: usize,
        fault: FaultSink,
        scratch: &mut HcScratch,
        mixed: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let w = self.w;
        let (down_qs, down_d) = q8(w, &site.down)?;
        let (up_qs, up_d) = q8(w, &site.up)?;
        let inject = match &site.inject {
            Some(n) => Some(f32_tensor(w, n)?),
            None => None,
        };
        self.k.hc.enqueue_mix(
            self.gpu.stream(),
            MixArgs {
                res,
                before,
                w: SiteWeights {
                    gamma: f32_gain(w, &site.norm)?,
                    down_qs,
                    down_d,
                    up_qs,
                    up_d,
                    inject,
                },
                eps: self.eps,
                m,
                fault,
                scratch,
                mixed,
            },
        )
    }
}

/// The parts of the body one walk writes: the plans, the stores, the PLE
/// ring, the arena, the unit's input record, its width, the stream buffer
/// the next site reads, the layer taps when a gate armed them, the slot
/// map's card copy and the card leg.
pub(super) struct Parts38<'a> {
    pub(super) c: Ctx38<'a>,
    pub(super) plans: &'a [Layer38],
    pub(super) stores: &'a mut [Store38],
    pub(super) ple_ring: &'a mut DeviceBuffer<f32>,
    pub(super) s: &'a mut Arena38,
    pub(super) io: &'a Io<'a>,
    pub(super) m: usize,
    pub(super) cur: usize,
    pub(super) taps: Option<&'a mut Taps38>,
    pub(super) slots: &'a DeviceTensor<u32>,
    pub(super) card: &'a mut Card38,
    /// A verify's row mode: each row's delta state into a lane of its own
    /// (`linear::delta`'s `gdn_delta_lanes`); else the last row's back into
    /// the committed lane.
    pub(super) each: bool,
}

/// Layer `l`'s plan.
fn plan_of(plans: &[Layer38], l: usize) -> Result<&Layer38, GpuError> {
    plans
        .get(l)
        .ok_or(GpuError::state(WHAT, "a plan for every layer"))
}

/// The streams `res` into layer `l`'s tap, when a gate armed the taps.
fn tap(
    taps: &mut Option<&mut Taps38>,
    res: &DeviceBuffer<f32>,
    l: usize,
    gpu: &Gpu,
) -> Result<(), GpuError> {
    if let Some(t) = taps.as_deref_mut() {
        let tap = t
            .out
            .get_mut(l)
            .ok_or(GpuError::state(WHAT, "a tap for every layer"))?;
        tap.copy_from_device_async(res, gpu.stream())?;
    }
    Ok(())
}

impl Parts38<'_> {
    /// Layer `l`'s front up to its feed-forward site: the embedding in front
    /// of layer 0, the attention site (with the PLE site on its layer), the
    /// mixer into `y` (module doc).
    fn front(&mut self, l: usize) -> Result<(), GpuError> {
        let (plans, gpu, m) = (self.plans, self.c.gpu, self.m);
        let p = plan_of(plans, l)?;
        let sink = gpu.layer_sink(l)?;
        let stream = gpu.stream();
        let c = &self.c;
        let s = &mut *self.s;
        if l == 0 {
            let (qs, d) = q8(c.w, &model::arch::qwen35moe::names::token_embd())?;
            c.k.q38.enqueue_embed_rows(
                stream,
                EmbedQ8Args {
                    qs,
                    d,
                    ids: self.io.ids,
                    pos0: self.io.pos0,
                    first: self.io.first,
                    fault: gpu.unlabelled_sink(),
                    y: &mut s.emb,
                    pos: &mut s.pos,
                    n_keys: &mut s.n_keys,
                },
            )?;
            c.mix(
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Init { y: &s.emb },
                m,
                sink,
                &mut s.hc,
                &mut s.mixed,
            )?;
        } else if let Some(pp) = &p.ple {
            let prev = gpu.layer_sink(l - 1)?;
            c.k.hc
                .enqueue_combine(stream, &mut s.res[self.cur], &s.y, m, prev, &s.hc)?;
            tap(&mut self.taps, &s.res[self.cur], l - 1, gpu)?;
            let w = c.w;
            let pl = &mut s.ple;
            c.q8_gemv(&pp.key, &pl.e, m, &mut pl.key)?;
            c.q8_gemv(&pp.value, &pl.e, m, &mut pl.value)?;
            let [r0, r1] = &mut s.res;
            let (x, out) = if self.cur == 0 {
                (&*r0, r1)
            } else {
                (&*r1, r0)
            };
            c.k.ple.enqueue_gate(
                stream,
                PleGateArgs {
                    key: &pl.key,
                    value: &pl.value,
                    x,
                    gain_key: f32_gain(w, &pp.norm_key)?,
                    gain_query: f32_gain(w, &pp.norm_query)?,
                    gain_conv: f32_gain(w, &pp.norm_conv)?,
                    eps: c.eps,
                    hc: geo::STREAMS,
                    m,
                    fault: sink,
                    gv: &mut pl.gv,
                    ngv: &mut pl.ngv,
                    gate: &mut pl.gate,
                },
            )?;
            c.k.ple.enqueue_conv(
                stream,
                PleConvArgs {
                    ngv: &pl.ngv,
                    gv: &pl.gv,
                    x,
                    w: f32_gain(w, &pp.conv)?,
                    pos: &s.pos,
                    taps: geo::PLE_TAPS,
                    dilation: geo::PLE_DILATION,
                    hc: geo::STREAMS,
                    m,
                    fault: sink,
                    out,
                    ring: &mut *self.ple_ring,
                },
            )?;
            self.cur ^= 1;
            c.mix(
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Plain,
                m,
                sink,
                &mut s.hc,
                &mut s.mixed,
            )?;
        } else {
            c.mix(
                &p.attn_hc,
                &mut s.res[self.cur],
                Before::Combine { y: &s.y },
                m,
                sink,
                &mut s.hc,
                &mut s.mixed,
            )?;
            tap(&mut self.taps, &s.res[self.cur], l - 1, gpu)?;
        }
        let store = self
            .stores
            .get_mut(l)
            .ok_or(GpuError::state(WHAT, "a store for every layer"))?;
        match (&p.mixer, store) {
            (Mixer38::Gdn(g), Store38::Rec { rec, stamp }) => {
                let lane = self.io.lane.ok_or(GpuError::state(
                    WHAT,
                    "the record's lane word (a chain with delta layers carries one)",
                ))?;
                gdn(c, g, (rec, stamp, lane), s, (m, self.each), sink)
            }
            (Mixer38::Qsa(q), Store38::Qsa { kv, raw, pooled }) => {
                qsa(c, q, (kv, raw, pooled), s, m, sink)
            }
            _ => Err(GpuError::shape(
                WHAT,
                format!("layer {l}'s plan and store are of two kinds"),
            )),
        }
    }

    /// Layer `l`'s feed-forward site: the mix of the streams, the mixer's
    /// output combined first, into `into` (`None`: the arena's `ffn_x`).
    fn ffn_mix(&mut self, l: usize, into: Option<&mut DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let s = &mut *self.s;
        let x = match into {
            Some(x) => x,
            None => &mut s.ffn_x,
        };
        self.c.mix(
            &p.ffn_hc,
            &mut s.res[self.cur],
            Before::Combine { y: &s.y },
            self.m,
            sink,
            &mut s.hc,
            x,
        )
    }

    /// Layer `l`'s router over `x` (`None`: the arena's `ffn_x`): the routed
    /// slots, then the shared expert's with the sigmoid of its gate; its
    /// logits and ids into layer `l`'s route taps when a gate armed them.
    fn route(&mut self, l: usize, x: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let stream = self.c.gpu.stream();
        let s = &mut *self.s;
        let x = x.unwrap_or(&s.ffn_x);
        self.c.k.router.enqueue_fused(
            stream,
            f32_tensor(self.c.w, &p.ffn.router)?,
            x,
            self.m,
            sink,
            &mut s.route,
        )?;
        if let Some(t) = self.taps.as_deref_mut() {
            let missing = || GpuError::state(WHAT, "a route tap for every layer");
            t.logits
                .get_mut(l)
                .ok_or_else(missing)?
                .copy_from_device_async(&s.route.logits, stream)?;
            t.ids
                .get_mut(l)
                .ok_or_else(missing)?
                .copy_from_device_async(&s.route.ids, stream)?;
        }
        Ok(())
    }

    /// Layer `l`'s shared expert over `x` (`None`: the arena's `ffn_x`) into
    /// `sh_y`: gate and up, SwiGLU, down.
    fn shared(&mut self, l: usize, x: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let (c, m) = (&self.c, self.m);
        let s = &mut *self.s;
        let x = x.unwrap_or(&s.ffn_x);
        c.q8_gemv(&p.ffn.gate_sh, x, m, &mut s.sh_g)?;
        c.q8_gemv(&p.ffn.up_sh, x, m, &mut s.sh_u)?;
        c.gpu
            .elem()
            .enqueue_swiglu(c.gpu.stream(), &s.sh_g, &s.sh_u, geo::FF * m, &mut s.sh_h)?;
        c.q8_gemv(&p.ffn.down_sh, &s.sh_h, m, &mut s.sh_y)
    }

    /// Layer `l`'s card leg over `x` (`None`: the arena's `ffn_x`), when it
    /// has card experts: the places in the arena's `sel`, the router's
    /// weights, into the card sum ([`Card38::enqueue`]).
    fn card(&mut self, l: usize, x: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        if !self.card.has(l) {
            return Ok(());
        }
        let s = &*self.s;
        let x = x.unwrap_or(&s.ffn_x);
        self.card
            .enqueue(&self.c, l, x, self.m, &s.sel, &s.route.weights)
    }

    /// Layer `l`'s block output into `y`: the host's routed sum `hsum`, on a
    /// layer with card experts the card's, plus the shared expert's output
    /// times its gate weight.
    fn shared_add(&mut self, l: usize, hsum: &DeviceBuffer<f32>) -> Result<(), GpuError> {
        let sink = self.c.gpu.layer_sink(l)?;
        let s = &mut *self.s;
        let (stream, q38) = (self.c.gpu.stream(), &self.c.k.q38);
        let (slot, slots, n, m) = (geo::N_USED, geo::N_USED + 1, geo::HIDDEN, self.m);
        if self.card.has(l) {
            q38.enqueue_card_shared_add(
                stream,
                CardSharedAddArgs {
                    hsum,
                    acc: self.card.acc()?,
                    sh: &s.sh_y,
                    w: &s.route.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut s.y,
                },
            )
        } else {
            q38.enqueue_shared_add(
                stream,
                SharedAddArgs {
                    hsum,
                    sh: &s.sh_y,
                    w: &s.route.weights,
                    slot,
                    slots,
                    n,
                    m,
                    fault: sink,
                    y: &mut s.y,
                },
            )
        }
    }

    /// The head's mix over the unit's columns, the last layer's combine
    /// first, into `into` (`None`: the arena's `mixed`); the last layer's tap
    /// after it.
    fn head_mix(&mut self, into: Option<&mut DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let gpu = self.c.gpu;
        let fault = gpu.fault_sink(LAYER_HEAD);
        let s = &mut *self.s;
        let out = match into {
            Some(x) => x,
            None => &mut s.mixed,
        };
        self.c.mix(
            &head_site(),
            &mut s.res[self.cur],
            Before::Combine { y: &s.y },
            self.m,
            fault,
            &mut s.hc,
            out,
        )?;
        let last = self.plans.len().saturating_sub(1);
        tap(&mut self.taps, &s.res[self.cur], last, gpu)
    }
}

/// A delta layer's mixer at `m` rows over its store `r`, its lanes' stamps
/// and the lane word, in row mode when `each`, the attention site's mix in
/// `s.mixed`, its output projection into `s.y`.
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    (r, stamp, lane): (
        &mut super::scratch::RecStore,
        &mut DeviceBuffer<u32>,
        &DeviceBuffer<u32>,
    ),
    s: &mut Arena38,
    (m, each): (usize, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let nv = GDN.n_v;
    let Arena38 {
        mixed,
        gdn: g,
        pos,
        attn,
        y,
        ..
    } = s;
    c.q8_gemv(&gp.qkv, mixed, m, &mut g.x)?;
    c.q8_gemv(&gp.z, mixed, m, &mut g.z)?;
    c.f32_gemv(&gp.beta_alpha, mixed, m, &mut g.ba)?;
    // SAFETY: the joined output is `2·nv` rows of `m` columns inside `ba`
    // (the arena's `2·nv` a row, `m` at most its rows); β's rows are the
    // first `nv·m` values and α's the next, and `ba` stays in place while
    // the windows live (this layer's launches).
    let (wb, wa) = unsafe { (f32_view(&g.ba, 0, nv * m), f32_view(&g.ba, nv * m, nv * m)) };
    let (b, a): (&DeviceBuffer<f32>, &DeviceBuffer<f32>) = if m == 1 {
        (&wb, &wa)
    } else {
        c.k.proj.enqueue_token_major(stream, &wb, nv, m, &mut g.b)?;
        c.k.proj.enqueue_token_major(stream, &wa, nv, m, &mut g.a)?;
        (&g.b, &g.a)
    };
    let lin = &c.k.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: &g.x,
            b_raw: b,
            a_raw: a,
            w: f32_gain(w, &gp.conv)?,
            dt_bias: f32_gain(w, &gp.dt_bias)?,
            ssm_a: f32_gain(w, &gp.ssm_a)?,
            pos: &*pos,
            shape: GDN,
            eps: c.eps,
            m,
            fault: sink,
            y: &mut g.conv,
            beta: &mut g.beta,
            decay: &mut g.decay,
            ring: &mut r.ring,
        },
    )?;
    lin.delta.enqueue_delta_lanes(
        stream,
        DeltaLanesArgs {
            delta: DeltaArgs {
                qkv: &g.conv,
                beta: &g.beta,
                decay: &g.decay,
                lane,
                lane_at: 0,
                lanes: r.lanes,
                shape: GDN,
                m,
                fault: sink,
                o: &mut g.o,
                state: &mut r.state,
            },
            each,
            pos: &*pos,
            stamp,
        },
    )?;
    lin.norm_gate.enqueue_norm_gate_sigmoid(
        stream,
        NormGateArgs {
            o: &g.o,
            z: &g.z,
            w: f32_gain(w, &gp.ssm_norm)?,
            eps: c.eps,
            n_v: nv,
            m,
            fault: sink,
            y: &mut *attn,
        },
    )?;
    debug_assert_eq!(nv * linear::HEAD, geo::ATTN);
    c.q8_gemv(&gp.ssm_out, &*attn, m, y)
}

/// A selecting attention layer's mixer at `m` rows over its store (the K/V
/// planes, the raw and pooled indexer keys), the attention site's mix in
/// `s.mixed`, its output projection into `s.y`.
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (kv, raw, pooled): (
        &mut super::scratch::KvPlanes,
        &mut DeviceBuffer<u16>,
        &mut DeviceBuffer<u16>,
    ),
    s: &mut Arena38,
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let Arena38 {
        mixed,
        qsa: q,
        pos,
        n_keys,
        flash,
        attn,
        y,
        ..
    } = s;
    c.q8_gemv(&qp.q, mixed, m, &mut q.qg)?;
    c.q8_gemv(&qp.k, mixed, m, &mut q.k)?;
    c.q8_gemv(&qp.v, mixed, m, &mut q.v)?;
    c.f32_gemv(&qp.idx_k, mixed, m, &mut q.kr)?;
    c.k.q38.enqueue_key_append(
        stream,
        KeyAppendArgs {
            kr: &q.kr,
            pos: &*pos,
            m,
            ctx,
            fault: sink,
            raw: &mut *raw,
        },
    )?;
    c.k.qsa.enqueue_pool(
        stream,
        PoolArgs {
            raw: &*raw,
            gain: f32_gain(w, &qp.idx_k_norm)?,
            table: c.table,
            n_keys: &*n_keys,
            eps: c.eps,
            ctx,
            m,
            fault: sink,
            pooled: &mut *pooled,
        },
    )?;
    c.f32_gemv(&qp.idx_q, mixed, m, &mut q.qr)?;
    let qsel: &DeviceBuffer<f32> = if m == 1 {
        &q.qr
    } else {
        c.k.proj
            .enqueue_token_major(stream, &q.qr, geo::IDX_HEADS * geo::IDX_DIM, m, &mut q.qi)?;
        &q.qi
    };
    c.k.qsa.enqueue_select(
        stream,
        SelectArgs {
            q: qsel,
            gain: f32_gain(w, &qp.idx_q_norm)?,
            table: c.table,
            n_keys: &*n_keys,
            pooled: &*pooled,
            eps: c.eps,
            ctx,
            kept: geo::KEPT,
            m,
            fault: sink,
            scratch: &mut q.sel,
        },
    )?;
    c.k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
            qg: &q.qg,
            q: &mut q.q,
            k: &mut q.k,
            v: &q.v,
            gq: f32_gain(w, &qp.q_norm)?,
            gk: f32_gain(w, &qp.k_norm)?,
            table: c.table,
            pos: &*pos,
            eps: c.eps,
            n_head: geo::N_HEAD,
            n_kv: geo::N_KV,
            ctx,
            m,
            fault: sink,
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    let width = q.sel.width();
    c.k.flash.enqueue_pass_256_p4_sel(
        stream,
        GqaSelArgs {
            q: &q.q,
            kc: &kv.k,
            vc: &kv.v,
            list: &q.sel.list,
            n_sel: &q.sel.n_sel,
            width,
            scale: ATTN_SCALE_256,
            n_kv: geo::N_KV,
            ctx,
            m,
            part_v: &mut q.part_v,
            part_ms: &mut q.part_ms,
            fault: sink,
            y: &mut *flash,
        },
        geo::N_HEAD,
        MMA,
    )?;
    c.k.q38.enqueue_out_gate(
        stream,
        OutGateArgs {
            attn: &*flash,
            qg: &q.qg,
            n_head: geo::N_HEAD,
            m,
            fault: sink,
            y: &mut *attn,
        },
    )?;
    c.q8_gemv(&qp.out, &*attn, m, y)
}

/// The captured decode step's walk `(1, 1, Step)`, into `head`.
pub(super) struct Step38<'a> {
    pub(super) p: Parts38<'a>,
    pub(super) head: &'a mut Head,
}

impl<'a> Step38<'a> {
    /// Walk every layer through `leg`, then the head.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let o = Overlap {
            units: 1,
            cols: 1,
            port: PortKind::Step,
        };
        let layers = self.p.plans.len();
        sched::walk(o, layers, leg, &mut self)
    }
}

impl<'a> LayerProgram for Step38<'a> {
    type Port = StepLeg<'a, HostRun>;

    /// The layer up to its router, the mix into the boundary's activation,
    /// then the handoff into the page and the go.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.front(l)?;
        let hy = port.hybrid();
        self.p.ffn_mix(l, Some(hy.boundary_mut().normed_mut()))?;
        self.p.route(l, Some(hy.boundary().normed()))?;
        let sink = self.p.c.gpu.layer_sink(l)?;
        let stream = self.p.c.gpu.stream();
        let s = &mut *self.p.s;
        let h = Handoff {
            ids: &s.route.ids,
            weights: &s.route.weights,
            map: self.p.slots.buf(),
            row_off: l * geo::EXPERTS,
            n_expert: geo::EXPERTS,
        };
        let target = hy.boundary_mut().handoff_target_of(0)?;
        self.p
            .c
            .k
            .handoff
            .enqueue_handoff(stream, &h, target, sink, &mut s.sel)?;
        hy.boundary().enqueue_go_of(stream, l, 0)
    }

    /// The card leg, then the shared expert, over the boundary's activation.
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let x = port.hybrid().boundary().normed();
        self.p.card(at.layer, Some(x))?;
        self.p.shared(at.layer, Some(x))
    }

    /// The wait, the gated sum, and the host tier told the layer is enqueued
    /// (an eager step is served there).
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let stream = self.p.c.gpu.stream();
        {
            let b = port.hybrid().boundary();
            b.enqueue_back_of(stream, 0)?;
            self.p.shared_add(l, b.hsum_of(0)?)?;
        }
        port.hybrid().row_enqueued(l, 0)
    }

    /// The head's mix into its input, then the head.
    fn end(&mut self, _unit: usize) -> Result<(), GpuError> {
        self.p.head_mix(Some(self.head.input_mut()))?;
        self.head.enqueue(self.p.c.gpu, self.p.c.w)
    }
}

/// The captured verify's walk `(1, m, Step)` over `m` consecutive
/// positions: every layer through the step port's `Cols(m)` chain — one
/// handoff launch writing the `m` columns' image, one go, one union call on
/// the host, one wait — each delta row's state into a lane of its own, then
/// the head's mix over the `m` columns into `head`, a head of `m` rows (one
/// projection and one argmax for every row). Each row is bit for bit its
/// step.
pub(super) struct Verify38<'a> {
    pub(super) p: Parts38<'a>,
    pub(super) head: &'a mut Head,
}

impl<'a> Verify38<'a> {
    /// Walk every layer at the unit's `m` columns through `leg`, then the
    /// head. A head of other rows than the walk's, or a walk not in row
    /// mode, is refused by name before any launch.
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let m = self.p.m;
        if self.head.m() != m || !self.p.each {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a verify of {m} rows (row mode {}) into a head of {} rows",
                    self.p.each,
                    self.head.m()
                ),
            ));
        }
        let o = Overlap {
            units: 1,
            cols: m,
            port: PortKind::Step,
        };
        let layers = self.p.plans.len();
        sched::walk(o, layers, leg, &mut self)
    }
}

impl<'a> LayerProgram for Verify38<'a> {
    type Port = StepLeg<'a, HostRun>;

    /// The layer up to its router over the `m` columns, the mix into the
    /// boundary's activations, then the `m` columns' handoff into the page
    /// in one launch and the go.
    fn front(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.front(l)?;
        let hy = port.hybrid();
        self.p.ffn_mix(l, Some(hy.boundary_mut().normed_mut()))?;
        self.p.route(l, Some(hy.boundary().normed()))?;
        let sink = self.p.c.gpu.layer_sink(l)?;
        let stream = self.p.c.gpu.stream();
        let m = self.p.m;
        let s = &mut *self.p.s;
        let h = Handoff {
            ids: &s.route.ids,
            weights: &s.route.weights,
            map: self.p.slots.buf(),
            row_off: l * geo::EXPERTS,
            n_expert: geo::EXPERTS,
        };
        let target = hy.boundary_mut().handoff_target_of(0)?;
        self.p.c.k.handoff.enqueue_handoff_cols(
            stream,
            &h,
            geo::N_USED + 1,
            m,
            target,
            sink,
            &mut s.sel,
        )?;
        hy.boundary().enqueue_go_of(stream, l, 0)
    }

    /// The card leg, then the shared expert, over the boundary's
    /// activations.
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let x = port.hybrid().boundary().normed();
        self.p.card(at.layer, Some(x))?;
        self.p.shared(at.layer, Some(x))
    }

    /// The wait, the gated sum over the `m` columns' host sums, and the host
    /// tier told the layer is enqueued (an eager verify is served there).
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        let stream = self.p.c.gpu.stream();
        {
            let b = port.hybrid().boundary();
            b.enqueue_back_of(stream, 0)?;
            self.p.shared_add(l, b.hsum_of(0)?)?;
        }
        port.hybrid().row_enqueued(l, 0)
    }

    /// The head's mix over the `m` columns into the head's input, then the
    /// head.
    fn end(&mut self, _unit: usize) -> Result<(), GpuError> {
        self.p.head_mix(Some(self.head.input_mut()))?;
        self.head.enqueue(self.p.c.gpu, self.p.c.w)
    }
}

/// An eager pass's walk `(1, m, Batch)`: every layer through the batch port;
/// the head is the caller's ([`Pass38::head`]).
pub(super) struct Pass38<'a> {
    pub(super) p: Parts38<'a>,
}

impl<'a> Pass38<'a> {
    /// Walk every layer at the unit's `m` columns through `leg`.
    pub(super) fn walk(&mut self, leg: &mut BatchLeg<'a, HostRun>) -> Result<(), GpuError> {
        let o = Overlap {
            units: 1,
            cols: self.p.m,
            port: PortKind::Batch,
        };
        let layers = self.p.plans.len();
        sched::walk(o, layers, leg, self)
    }

    /// After the walk: the head's mix over the unit's columns, its last row
    /// into `head`'s input, and the head. A unit of no row, or of more rows
    /// than the arena holds, is refused by name before any launch.
    pub(super) fn head(&mut self, head: &mut Head) -> Result<(), GpuError> {
        let (m, h, gpu) = (self.p.m, geo::HIDDEN, self.p.c.gpu);
        if m == 0 || m > self.p.s.rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a head after a unit of {m} rows on a {}-row arena",
                    self.p.s.rows
                ),
            ));
        }
        self.p.head_mix(None)?;
        // SAFETY: 1 <= m <= rows (checked above), so row m − 1 spans `hidden`
        // values inside `mixed` (`rows · hidden`), which stays in place while
        // the window lives (one copy).
        let row = unsafe { f32_view(&self.p.s.mixed, (m - 1) * h, h) };
        head.input_mut()
            .copy_from_device_async(&row, gpu.stream())?;
        head.enqueue(gpu, self.p.c.w)
    }
}

impl<'a> LayerProgram for Pass38<'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// The layer up to its router, the mix into the arena's `ffn_x`, then
    /// the download of the unit's activations and routed slots (the router's
    /// `N_USED + 1` a token, the routed `N_USED` of them), and on a layer
    /// with card experts each routed slot's place into the arena's `sel`
    /// from the slot map's card copy.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.front(l)?;
        self.p.ffn_mix(l, None)?;
        self.p.route(l, None)?;
        let key = port.key(at);
        let stream = self.p.c.gpu.stream();
        let s = &mut *self.p.s;
        port.hybrid().enqueue_download_pitched(
            stream,
            [&s.ffn_x, &s.route.weights],
            &s.route.ids,
            geo::N_USED + 1,
            key,
        )?;
        if !self.p.card.has(l) {
            return Ok(());
        }
        let p = Places {
            ids: &s.route.ids,
            map: self.p.slots.buf(),
            row_off: l * geo::EXPERTS,
            n_expert: geo::EXPERTS,
        };
        self.p.c.k.handoff.enqueue_places_cols(
            stream,
            &p,
            geo::N_USED + 1,
            self.p.m,
            self.p.c.gpu.layer_sink(l)?,
            &mut s.sel,
        )
    }

    /// The card leg, then the shared expert, over the arena's `ffn_x`.
    fn shadow(&mut self, _: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        self.p.card(at.layer, None)?;
        self.p.shared(at.layer, None)
    }

    /// The gated sum over the host sums the walk's serve uploaded.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        self.p.shared_add(at.layer, port.hsum())
    }
}
