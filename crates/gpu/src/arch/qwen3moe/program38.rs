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
//! On a load with an expert tier (`tier38`) the step and the verify serve
//! its layers: a tier layer's handoff is `ds41_ffn_handoff_10_cols_tier`
//! (the step's of one column), which also writes the tier's places and the
//! normed rows into the row's tier image, its go and its wait the tier's
//! too ([`crate::hybrid::Hybrid::enqueue_tier_go`], `_back`), its card leg
//! the launches but the card sum, and its back the join over both cards'
//! slots in slot order with the combine in one launch
//! (`q38_card_tier_shared_add`). The pass has no tier leg (`card38`'s map
//! check refuses it first).
//!
//! After the last layer the head's mix (its norm applies the last combine)
//! writes the head's input. Launches of the captured step
//! ([`step_launches`]): the embedding; per layer the two mixes' three, the
//! mixer's ([`GDN_LAUNCHES`] or [`QSA_LAUNCHES`]), the PLE site's
//! [`PLE_LAUNCHES`] on its layer, the block's [`FFN_LAUNCHES`] — two of
//! them the go and the wait, stream memory-operation batches
//! ([`STEP_MEMOPS`] a layer) — and the card leg's [`CARD_LAUNCHES`] on a
//! layer with card experts ([`TIER_CARD_LAUNCHES`] on a tier layer); then
//! the head's mix, projection and argmax.

use super::body::ATTN_SCALE_256;
use super::card38::{CARD_LAUNCHES, Card38, TIER_CARD_LAUNCHES};
use super::plan38::{GDN, GdnPlan, HcSite, Layer38, Mixer38, PlePlan, QsaPlan, geo, head_site};
use super::proj::ProjKernels;
use super::router::gated;
use super::scratch::{Io, f32_view};
use super::scratch38::{Arena38, Store38, Taps38};
use super::tier38::{STAGE_TIER, TierSide38};
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
use crate::mtp::MtpKernels;
use crate::ple::{PleConvArgs, PleGateArgs, PleKernels};
use crate::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use crate::q38::{
    CardSharedAddArgs, EmbedQ8Args, KeyAppendArgs, OutGateArgs, Q38Kernels, SharedAddArgs,
};
use crate::qsa::{PoolArgs, QsaKernels, QsaScratch, SelectArgs};
use crate::rope_neox::{PartialNeoxArgs, RopeNeoxKernels};
use crate::tensor::{DeviceTensor, Window, WindowMut};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};
use std::ops::Range;

/// What the walks' errors name.
const WHAT: &str = "qwen4exp program";

/// The draft walk's dense flash pass: the tensor-core one.
pub(super) const MMA: bool = true;

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

/// The head's launches in a captured walk for its borrowed form — the one
/// owner of the form is `model::arch::qwen35moe::mtp` (`head_kind`, its
/// `output_form` the borrow check asks): [`HEAD_LAUNCHES`] for the Q8_0
/// planes, whose projection reads the normed f32 rows; a Q6_K word plane's
/// walk quantizes its input rows to q8_1 first, one launch more
/// (`Head::enqueue`'s `enqueue_quantize_q8_1_head`).
fn head_launches(head: model::arch::qwen35moe::mtp::BorrowedHead) -> usize {
    HEAD_LAUNCHES
        + usize::from(matches!(
            head,
            model::arch::qwen35moe::mtp::BorrowedHead::Q6K
        ))
}

/// A delta layer's launches past [`GDN_LAUNCHES`] at more than one row: β
/// and α copied token-major out of the joined projection.
pub(super) const GDN_ROWS_LAUNCHES: usize = 2;
/// A selecting layer's launches past [`QSA_LAUNCHES`] at more than one row:
/// the indexer queries copied token-major.
pub(super) const QSA_ROWS_LAUNCHES: usize = 1;

/// The captured decode step's launches for `plans` with `card`'s card
/// experts and `tier`'s tier layers (module doc).
pub(super) fn step_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    head: model::arch::qwen35moe::mtp::BorrowedHead,
) -> usize {
    walk_launches(plans, card, tier, 1, head)
}

/// The captured verify's launches for `plans` with `card`'s card experts and
/// `tier`'s tier layers at `m >= 2` rows: the step's, plus each mixer's
/// token-major copies ([`GDN_ROWS_LAUNCHES`], [`QSA_ROWS_LAUNCHES`]); the
/// handoff, the go, the wait, the card leg and the head are one each
/// whatever `m`.
pub(super) fn verify_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    m: usize,
    head: model::arch::qwen35moe::mtp::BorrowedHead,
) -> usize {
    walk_launches(plans, card, tier, m, head)
}

fn walk_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    m: usize,
    head: model::arch::qwen35moe::mtp::BorrowedHead,
) -> usize {
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
                + match (card.has(l), tier.is_some_and(|t| t.k(l) > 0)) {
                    (true, true) => TIER_CARD_LAUNCHES,
                    (true, false) => CARD_LAUNCHES,
                    (false, _) => 0,
                }
        })
        .sum::<usize>()
        + head_launches(head)
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
    /// The MTP draft layer's input pack and head argmax (`mtp38`).
    pub(super) mtp: MtpKernels,
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
            mtp: MtpKernels::load(ctx)?,
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
    pub(super) fn q8_gemv(
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
    pub(super) fn mix(
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

/// The parts of the body one walk writes: the plans, the sequences its
/// unit binds ([`Seqs38`]), the arena, its width, the stream buffer the
/// next site reads, the layer taps when a gate armed them, the slot map's
/// card copy and the card leg.
pub(super) struct Parts38<'a> {
    pub(super) c: Ctx38<'a>,
    pub(super) plans: &'a [Layer38],
    pub(super) seqs: Seqs38<'a>,
    pub(super) s: &'a mut Arena38,
    pub(super) m: usize,
    pub(super) cur: usize,
    pub(super) taps: Option<&'a mut Taps38>,
    pub(super) slots: &'a DeviceTensor<u32>,
    pub(super) card: &'a mut Card38,
    /// The stage card's tier side on a load with an expert tier.
    pub(super) tier: Option<&'a mut TierSide38>,
    /// A verify's row mode: each row's delta state into a lane of its own
    /// (`linear::delta`'s `gdn_delta_lanes`); else the last row's back into
    /// the committed lane.
    pub(super) each: bool,
}

/// The sequences a walk's unit binds: one over every row of the unit — the
/// step's, a verify's, a pass's — its stores and input record handed to the
/// launches whole; or several, each over its own rows of the unit (a pass of
/// several slots), each sequence-bound launch run once a sequence over its
/// rows' windows and its own stores.
pub(super) enum Seqs38<'a> {
    One {
        stores: &'a mut [Store38],
        ple_ring: &'a mut DeviceBuffer<f32>,
        io: &'a Io<'a>,
    },
    Slots(Vec<SlotSeq38<'a>>),
}

/// One busy slot of a pass of several: its stores, its PLE ring, its input
/// record (its first position, its ids and its lane word) and the rows of
/// the pass it holds.
pub(super) struct SlotSeq38<'a> {
    pub(super) stores: &'a mut [Store38],
    pub(super) ple_ring: &'a mut DeviceBuffer<f32>,
    pub(super) io: Io<'a>,
    pub(super) rows: Range<usize>,
}

/// The launches one more sequence's rows add to a pass: its embedding,
/// the PLE site's conv, each delta layer's [`GDN_SEQ_LAUNCHES`] and each
/// selecting layer's [`QSA_SEQ_LAUNCHES`] — every launch that binds a
/// sequence's stores or its input record.
pub(super) fn seq_launches(plans: &[Layer38]) -> usize {
    1 + plans
        .iter()
        .map(|p| {
            let mixer = match p.mixer {
                Mixer38::Gdn(_) => GDN_SEQ_LAUNCHES,
                Mixer38::Qsa(_) => QSA_SEQ_LAUNCHES,
            };
            mixer + usize::from(p.ple.is_some())
        })
        .sum::<usize>()
}

/// A delta layer's launches bound to a sequence: the conv over its ring and
/// the delta step over its state, stamps and lane word.
pub(super) const GDN_SEQ_LAUNCHES: usize = 2;
/// A selecting layer's launches bound to a sequence: the indexer key's
/// projection (its rows row-major, so a sequence's own), its append and the
/// pool; the selection's two; the q/k norm, turn and append; the selected
/// flash's two.
pub(super) const QSA_SEQ_LAUNCHES: usize = 1 + 1 + 1 + 2 + 1 + 2;

/// Rows `r` of `parent`, `width` values a row, shared.
fn rows_of<'b, T>(
    parent: &'b DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<Window<'b, T>, GpuError> {
    Window::of(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// Rows `r` of `parent`, `width` values a row, to write through.
fn rows_mut<'b, T>(
    parent: &'b mut DeviceBuffer<T>,
    r: &Range<usize>,
    width: usize,
) -> Result<WindowMut<'b, T>, GpuError> {
    WindowMut::of_mut(parent, r.start * width * size_of::<T>(), r.len() * width)
}

/// Layer `l`'s delta store of `stores`: its state, ring and stamps.
fn rec_of(
    stores: &mut [Store38],
    l: usize,
) -> Result<(&mut super::scratch::RecStore, &mut DeviceBuffer<u32>), GpuError> {
    match stores.get_mut(l) {
        Some(Store38::Rec { rec, stamp }) => Ok((rec, stamp)),
        Some(Store38::Qsa { .. }) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}'s plan and store are of two kinds"),
        )),
        None => Err(GpuError::state(WHAT, "a store for every layer")),
    }
}

/// Layer `l`'s selecting store of `stores`: its K/V planes, raw and pooled
/// keys.
fn qsa_of(
    stores: &mut [Store38],
    l: usize,
) -> Result<
    (
        &mut super::scratch::KvPlanes,
        &mut DeviceBuffer<u16>,
        &mut DeviceBuffer<u16>,
    ),
    GpuError,
> {
    match stores.get_mut(l) {
        Some(Store38::Qsa { kv, raw, pooled }) => Ok((kv, raw, pooled)),
        Some(Store38::Rec { .. }) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}'s plan and store are of two kinds"),
        )),
        None => Err(GpuError::state(WHAT, "a store for every layer")),
    }
}

/// The lane word of a record, which every chain with delta layers carries.
fn lane_of<'b>(io: &Io<'b>) -> Result<&'b DeviceBuffer<u32>, GpuError> {
    io.lane.ok_or(GpuError::state(
        WHAT,
        "the record's lane word (a chain with delta layers carries one)",
    ))
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
    /// mixer into `y` (module doc). Each launch bound to a sequence runs once
    /// a sequence of the unit ([`Seqs38`]).
    fn front(&mut self, l: usize) -> Result<(), GpuError> {
        let (plans, gpu, m) = (self.plans, self.c.gpu, self.m);
        let p = plan_of(plans, l)?;
        let sink = gpu.layer_sink(l)?;
        let stream = gpu.stream();
        let c = &self.c;
        let s = &mut *self.s;
        if l == 0 {
            embed(c, &self.seqs, s)?;
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
            let x = &s.res[self.cur];
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
            ple_conv(c, pp, &mut self.seqs, s, (self.cur, m), sink)?;
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
        match &p.mixer {
            Mixer38::Gdn(g) => gdn(c, g, l, &mut self.seqs, s, (m, self.each), sink),
            Mixer38::Qsa(q) => qsa(c, q, l, &mut self.seqs, s, m, sink),
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
        if self.tier_k(l) > 0 {
            return self.card.enqueue_rows(&self.c, l, x, self.m, &s.sel);
        }
        self.card
            .enqueue(&self.c, l, x, self.m, &s.sel, &s.route.weights)
    }

    /// The tier's experts of layer `l`: 0 off the tier and on a load
    /// without one.
    fn tier_k(&self, l: usize) -> usize {
        self.tier.as_deref().map_or(0, |t| t.k(l))
    }

    /// Layer `l`'s handoff of the unit's `m` columns into the row's image
    /// with the router's slots (eleven a column, the shared expert's last)
    /// and the go: on a tier layer `ds41_ffn_handoff_10_cols_tier` with the
    /// tier's go, else `plain` — the walk's own launch — and the plain go.
    fn handoff_go(
        &mut self,
        hy: &mut crate::hybrid::Hybrid<HostRun>,
        l: usize,
        plain: impl FnOnce(
            &HandoffKernels,
            &Handoff<'_>,
            crate::host::step::HandoffTarget<'_>,
            &mut DeviceBuffer<u32>,
        ) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let sink = self.c.gpu.layer_sink(l)?;
        let stream = self.c.gpu.stream();
        let (m, s) = (self.m, &mut *self.s);
        let h = Handoff {
            ids: &s.route.ids,
            weights: &s.route.weights,
            map: self.slots.buf(),
            row_off: l * geo::EXPERTS,
            n_expert: geo::EXPERTS,
        };
        let tier = self.tier.as_deref_mut().filter(|t| t.k(l) > 0);
        let Some(t) = tier else {
            let target = hy.boundary_mut().handoff_target_of(0)?;
            plain(&self.c.k.handoff, &h, target, &mut s.sel)?;
            return hy.boundary().enqueue_go_of(stream, l, 0);
        };
        let targets = hy.tier_handoff(0, STAGE_TIER)?;
        let (tmap, tsel) = t.handoff_parts();
        self.c.k.handoff.enqueue_handoff_cols_tier(
            stream,
            &h,
            (geo::N_USED + 1, m),
            tmap,
            targets,
            sink,
            (&mut s.sel, tsel),
        )?;
        hy.enqueue_tier_go(stream, l, 0)
    }

    /// Layer `l`'s back through the step port over row 0: the wait — the
    /// host's, and on a tier layer the tier's too — then the block's output
    /// into `y`: on a tier layer the join over both cards' slots with the
    /// combine ([`Card38::enqueue_tier_join`]), else the shared add over the
    /// host's sum ([`Parts38::shared_add`]).
    fn step_back(
        &mut self,
        hy: &mut crate::hybrid::Hybrid<HostRun>,
        l: usize,
    ) -> Result<(), GpuError> {
        let stream = self.c.gpu.stream();
        let n_tier = self.tier_k(l);
        if n_tier == 0 {
            let b = hy.boundary();
            b.enqueue_back_of(stream, 0)?;
            return self.shared_add(l, b.hsum_of(0)?);
        }
        hy.enqueue_tier_back(stream, l, 0)?;
        let (hy, s) = (&*hy, &mut *self.s);
        let tsel = &self
            .tier
            .as_deref()
            .ok_or(GpuError::state(WHAT, "the tier side of a tier layer"))?
            .tsel;
        self.card.enqueue_tier_join(
            &self.c,
            (l, self.m),
            (&s.sel, tsel, n_tier),
            hy.tier_rows(0)?,
            (hy.boundary().hsum_of(0)?, &s.sh_y, &s.route.weights),
            &mut s.y,
        )
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

/// Each sequence's embedding of its ids into its rows of `s` — `emb`, and
/// each row's position and live key count — from its own input record.
fn embed(c: &Ctx38<'_>, seqs: &Seqs38<'_>, s: &mut Arena38) -> Result<(), GpuError> {
    let (qs, d) = q8(c.w, &model::arch::qwen35moe::names::token_embd())?;
    let (gpu, stream) = (c.gpu, c.gpu.stream());
    let Arena38 {
        emb, pos, n_keys, ..
    } = s;
    match seqs {
        Seqs38::One { io, .. } => c.k.q38.enqueue_embed_rows(
            stream,
            EmbedQ8Args {
                qs,
                d,
                ids: io.ids,
                pos0: io.pos0,
                first: io.first,
                fault: gpu.unlabelled_sink(),
                y: emb,
                pos,
                n_keys,
            },
        ),
        Seqs38::Slots(seqs) => {
            for q in seqs {
                let r = &q.rows;
                c.k.q38.enqueue_embed_rows(
                    stream,
                    EmbedQ8Args {
                        qs,
                        d,
                        ids: q.io.ids,
                        pos0: q.io.pos0,
                        first: q.io.first,
                        fault: gpu.unlabelled_sink(),
                        y: &mut *rows_mut(emb, r, geo::HIDDEN)?,
                        pos: &mut *rows_mut(pos, r, 1)?,
                        n_keys: &mut *rows_mut(n_keys, r, 1)?,
                    },
                )?;
            }
            Ok(())
        }
    }
}

/// The PLE site's conv over the gate's rows (`ngv`, `gv`): each
/// sequence's rows of the streams `res[cur]` into the other stream buffer,
/// over its own ring.
fn ple_conv(
    c: &Ctx38<'_>,
    pp: &PlePlan,
    seqs: &mut Seqs38<'_>,
    s: &mut Arena38,
    (cur, m): (usize, usize),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (f32_gain(c.w, &pp.conv)?, c.gpu.stream());
    let Arena38 {
        res: [r0, r1],
        ple: pl,
        pos,
        ..
    } = s;
    let (x, out) = if cur == 0 { (&*r0, r1) } else { (&*r1, r0) };
    match seqs {
        Seqs38::One { ple_ring, .. } => c.k.ple.enqueue_conv(
            stream,
            PleConvArgs {
                ngv: &pl.ngv,
                gv: &pl.gv,
                x,
                w,
                pos,
                taps: geo::PLE_TAPS,
                dilation: geo::PLE_DILATION,
                hc: geo::STREAMS,
                m,
                fault: sink,
                out,
                ring: ple_ring,
            },
        ),
        Seqs38::Slots(seqs) => {
            let wide = geo::STREAMS * geo::HIDDEN;
            for q in seqs {
                let r = &q.rows;
                c.k.ple.enqueue_conv(
                    stream,
                    PleConvArgs {
                        ngv: &*rows_of(&pl.ngv, r, wide)?,
                        gv: &*rows_of(&pl.gv, r, wide)?,
                        x: &*rows_of(x, r, wide)?,
                        w,
                        pos: &*rows_of(pos, r, 1)?,
                        taps: geo::PLE_TAPS,
                        dilation: geo::PLE_DILATION,
                        hc: geo::STREAMS,
                        m: r.len(),
                        fault: sink,
                        out: &mut *rows_mut(out, r, wide)?,
                        ring: &mut *q.ple_ring,
                    },
                )?;
            }
            Ok(())
        }
    }
}

/// The rows a delta layer's launches bound to one sequence read and write:
/// the conv's inputs (`x`, β, α, each row's position) and outputs, and the
/// delta step's output `o`.
struct GdnIo<'b> {
    x: &'b DeviceBuffer<f32>,
    b: &'b DeviceBuffer<f32>,
    a: &'b DeviceBuffer<f32>,
    pos: &'b DeviceBuffer<u32>,
    conv: &'b mut DeviceBuffer<f32>,
    beta: &'b mut DeviceBuffer<f32>,
    decay: &'b mut DeviceBuffer<f32>,
    o: &'b mut DeviceBuffer<f32>,
}

/// A delta layer's mixer at `m` rows, layer `l` of each sequence of
/// `seqs`, the attention site's mix in `s.mixed`, its output projection into
/// `s.y`: the projections over every row ([`gdn_in`]), each sequence's conv
/// and delta step over its rows and its store, in row mode when `each`
/// ([`gdn_seq`]), then the gated norm and the output projection over every
/// row ([`gdn_out`]).
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    l: usize,
    seqs: &mut Seqs38<'_>,
    s: &mut Arena38,
    (m, each): (usize, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    gdn_in(c, gp, s, m)?;
    let nv = GDN.n_v;
    {
        let Arena38 { gdn: g, pos, .. } = &mut *s;
        let (wb, wa) = beta_alpha_rows(&g.ba, m)?;
        let (b, a): (&DeviceBuffer<f32>, &DeviceBuffer<f32>) =
            if m == 1 { (&*wb, &*wa) } else { (&g.b, &g.a) };
        match seqs {
            Seqs38::One { stores, io, .. } => {
                let lane = lane_of(io)?;
                let (rec, stamp) = rec_of(stores, l)?;
                gdn_seq(
                    c,
                    gp,
                    (rec, stamp, lane),
                    GdnIo {
                        x: &g.x,
                        b,
                        a,
                        pos,
                        conv: &mut g.conv,
                        beta: &mut g.beta,
                        decay: &mut g.decay,
                        o: &mut g.o,
                    },
                    (m, each),
                    sink,
                )?;
            }
            Seqs38::Slots(seqs) => {
                let ch = GDN.channels();
                for q in seqs {
                    let r = &q.rows;
                    let lane = lane_of(&q.io)?;
                    let (rec, stamp) = rec_of(q.stores, l)?;
                    gdn_seq(
                        c,
                        gp,
                        (rec, stamp, lane),
                        GdnIo {
                            x: &*rows_of(&g.x, r, ch)?,
                            b: &*rows_of(b, r, nv)?,
                            a: &*rows_of(a, r, nv)?,
                            pos: &*rows_of(pos, r, 1)?,
                            conv: &mut *rows_mut(&mut g.conv, r, ch)?,
                            beta: &mut *rows_mut(&mut g.beta, r, nv)?,
                            decay: &mut *rows_mut(&mut g.decay, r, nv)?,
                            o: &mut *rows_mut(&mut g.o, r, nv * linear::HEAD)?,
                        },
                        (r.len(), each),
                        sink,
                    )?;
                }
            }
        }
    }
    gdn_out(c, gp, s, m, sink)
}

/// A delta layer's projections over every row of `s.mixed`: q·k·v, `z`
/// and the joined β·α (row-major `[2·n_v][m]`), β and α then copied
/// token-major at more than one row.
fn gdn_in(c: &Ctx38<'_>, gp: &GdnPlan, s: &mut Arena38, m: usize) -> Result<(), GpuError> {
    let stream = c.gpu.stream();
    let nv = GDN.n_v;
    let Arena38 { mixed, gdn: g, .. } = s;
    c.q8_gemv(&gp.qkv, mixed, m, &mut g.x)?;
    c.q8_gemv(&gp.z, mixed, m, &mut g.z)?;
    c.f32_gemv(&gp.beta_alpha, mixed, m, &mut g.ba)?;
    if m > 1 {
        let (wb, wa) = beta_alpha_rows(&g.ba, m)?;
        c.k.proj.enqueue_token_major(stream, &wb, nv, m, &mut g.b)?;
        c.k.proj.enqueue_token_major(stream, &wa, nv, m, &mut g.a)?;
    }
    Ok(())
}

/// β's and α's rows of the joined projection `ba` over `m` columns
/// (row-major `[2·n_v][m]`): its first `n_v·m` values, then the next.
fn beta_alpha_rows(
    ba: &DeviceBuffer<f32>,
    m: usize,
) -> Result<(Window<'_, f32>, Window<'_, f32>), GpuError> {
    let n = GDN.n_v * m;
    Ok((
        Window::of(ba, 0, n)?,
        Window::of(ba, n * size_of::<f32>(), n)?,
    ))
}

/// A delta layer's launches bound to one sequence, over `io`'s `m` rows:
/// the conv over the store's ring `r.ring`, then the delta step over its
/// state, its lanes' stamps and the lane word `lane`, in row mode when
/// `each`.
fn gdn_seq(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    (r, stamp, lane): (
        &mut super::scratch::RecStore,
        &mut DeviceBuffer<u32>,
        &DeviceBuffer<u32>,
    ),
    io: GdnIo<'_>,
    (m, each): (usize, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let GdnIo {
        x,
        b,
        a,
        pos,
        conv,
        beta,
        decay,
        o,
    } = io;
    let lin = &c.k.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x,
            b_raw: b,
            a_raw: a,
            w: f32_gain(w, &gp.conv)?,
            dt_bias: f32_gain(w, &gp.dt_bias)?,
            ssm_a: f32_gain(w, &gp.ssm_a)?,
            pos,
            shape: GDN,
            eps: c.eps,
            m,
            fault: sink,
            y: &mut *conv,
            beta: &mut *beta,
            decay: &mut *decay,
            ring: &mut r.ring,
        },
    )?;
    lin.delta.enqueue_delta_lanes(
        stream,
        DeltaLanesArgs {
            delta: DeltaArgs {
                qkv: conv,
                beta,
                decay,
                lane,
                lane_at: 0,
                lanes: r.lanes,
                shape: GDN,
                m,
                fault: sink,
                o,
                state: &mut r.state,
            },
            each,
            pos,
            stamp,
        },
    )
}

/// A delta layer's gated norm and output projection over every row, into
/// `s.y`.
fn gdn_out(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    s: &mut Arena38,
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let nv = GDN.n_v;
    let Arena38 {
        gdn: g, attn, y, ..
    } = s;
    c.k.linear.norm_gate.enqueue_norm_gate_sigmoid(
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

/// The rows a selecting layer's indexer keys of one sequence read and
/// write: the mix, the raw keys' projection, each row's position and live
/// key count.
struct KeysIo<'b> {
    x: &'b DeviceBuffer<f32>,
    kr: &'b mut DeviceBuffer<f32>,
    pos: &'b DeviceBuffer<u32>,
    n_keys: &'b DeviceBuffer<u32>,
}

/// The rows a selecting layer's attention over one sequence reads and
/// writes: the selection's queries, each row's live key count and
/// position, the query projection's `[q | gate]` rows, the normed queries,
/// the keys and values, and the flash's output.
struct AttendIo<'b> {
    qsel: &'b DeviceBuffer<f32>,
    n_keys: &'b DeviceBuffer<u32>,
    qg: &'b DeviceBuffer<f32>,
    q: &'b mut DeviceBuffer<f32>,
    k: &'b mut DeviceBuffer<f32>,
    v: &'b DeviceBuffer<f32>,
    pos: &'b DeviceBuffer<u32>,
    y: &'b mut DeviceBuffer<f32>,
}

/// A selecting attention layer's mixer at `m` rows, layer `l` of each
/// sequence of `seqs`, the attention site's mix in `s.mixed`, its output
/// projection into `s.y`: q, k and v over every row ([`qsa_qkv`]); each
/// sequence's indexer keys ([`qsa_keys`]) and attention ([`qsa_attend`])
/// over its rows and its store, the indexer queries over every row between
/// ([`qsa_query`]: after one sequence's keys, before several sequences'
/// keys); then the gate and the output projection over every row
/// ([`qsa_out`]).
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    l: usize,
    seqs: &mut Seqs38<'_>,
    s: &mut Arena38,
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    qsa_qkv(c, qp, s, m)?;
    match seqs {
        Seqs38::One { stores, .. } => {
            let (kv, raw, pooled) = qsa_of(stores, l)?;
            {
                let Arena38 {
                    mixed,
                    qsa: qa,
                    pos,
                    n_keys,
                    ..
                } = &mut *s;
                let keys = KeysIo {
                    x: mixed,
                    kr: &mut qa.kr,
                    pos,
                    n_keys,
                };
                qsa_keys(c, qp, (raw, pooled), keys, m, sink)?;
            }
            qsa_query(c, qp, s, m)?;
            let Arena38 {
                qsa: qa,
                pos,
                n_keys,
                flash,
                ..
            } = &mut *s;
            let qsel: &DeviceBuffer<f32> = if m == 1 { &qa.qr } else { &qa.qi };
            let io = AttendIo {
                qsel,
                n_keys,
                qg: &qa.qg,
                q: &mut qa.q,
                k: &mut qa.k,
                v: &qa.v,
                pos,
                y: flash,
            };
            let scratch = (&mut qa.sel, &mut qa.part_v, &mut qa.part_ms);
            qsa_attend(c, qp, (kv, pooled), io, scratch, m, sink)?;
        }
        Seqs38::Slots(seqs) => {
            qsa_query(c, qp, s, m)?;
            let (h, d) = (geo::HIDDEN, geo::IDX_DIM);
            let Arena38 {
                mixed,
                qsa: qa,
                pos,
                n_keys,
                flash,
                ..
            } = &mut *s;
            for q in seqs {
                let r = &q.rows;
                let (kv, raw, pooled) = qsa_of(q.stores, l)?;
                let (pos, n_keys) = (rows_of(pos, r, 1)?, rows_of(n_keys, r, 1)?);
                let keys = KeysIo {
                    x: &*rows_of(mixed, r, h)?,
                    kr: &mut *rows_mut(&mut qa.kr, r, d)?,
                    pos: &pos,
                    n_keys: &n_keys,
                };
                qsa_keys(c, qp, (raw, pooled), keys, r.len(), sink)?;
                let qsel: &DeviceBuffer<f32> = if m == 1 { &qa.qr } else { &qa.qi };
                let io = AttendIo {
                    qsel: &*rows_of(qsel, r, geo::IDX_HEADS * d)?,
                    n_keys: &n_keys,
                    qg: &*rows_of(&qa.qg, r, geo::Q_ROWS)?,
                    q: &mut *rows_mut(&mut qa.q, r, geo::ATTN)?,
                    k: &mut *rows_mut(&mut qa.k, r, geo::KV)?,
                    v: &*rows_of(&qa.v, r, geo::KV)?,
                    pos: &pos,
                    y: &mut *rows_mut(flash, r, geo::ATTN)?,
                };
                let scratch = (&mut qa.sel, &mut qa.part_v, &mut qa.part_ms);
                qsa_attend(c, qp, (kv, pooled), io, scratch, r.len(), sink)?;
            }
        }
    }
    qsa_out(c, qp, s, m, sink)
}

/// A selecting layer's q (with each head's gate), k and v over every row
/// of `s.mixed`.
fn qsa_qkv(c: &Ctx38<'_>, qp: &QsaPlan, s: &mut Arena38, m: usize) -> Result<(), GpuError> {
    let Arena38 { mixed, qsa: qa, .. } = s;
    c.q8_gemv(&qp.q, mixed, m, &mut qa.qg)?;
    c.q8_gemv(&qp.k, mixed, m, &mut qa.k)?;
    c.q8_gemv(&qp.v, mixed, m, &mut qa.v)
}

/// A selecting layer's indexer keys of one sequence's `m` rows: their
/// projection (row-major `[IDX_DIM][m]`, so a sequence's own launch), the
/// raw keys appended to the store at the rows' positions, then the pools
/// those rows complete.
fn qsa_keys(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (raw, pooled): (&mut DeviceBuffer<u16>, &mut DeviceBuffer<u16>),
    io: KeysIo<'_>,
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let KeysIo { x, kr, pos, n_keys } = io;
    c.f32_gemv(&qp.idx_k, x, m, kr)?;
    c.k.q38.enqueue_key_append(
        stream,
        KeyAppendArgs {
            kr,
            pos,
            m,
            ctx,
            fault: sink,
            raw: &mut *raw,
        },
    )?;
    c.k.qsa.enqueue_pool(
        stream,
        PoolArgs {
            raw,
            gain: f32_gain(w, &qp.idx_k_norm)?,
            table: c.table,
            n_keys,
            eps: c.eps,
            ctx,
            m,
            fault: sink,
            pooled,
        },
    )
}

/// A selecting layer's indexer queries over every row of `s.mixed`
/// (row-major), copied token-major at more than one row.
fn qsa_query(c: &Ctx38<'_>, qp: &QsaPlan, s: &mut Arena38, m: usize) -> Result<(), GpuError> {
    let Arena38 { mixed, qsa: qa, .. } = s;
    c.f32_gemv(&qp.idx_q, mixed, m, &mut qa.qr)?;
    if m > 1 {
        let stream = c.gpu.stream();
        c.k.proj.enqueue_token_major(
            stream,
            &qa.qr,
            geo::IDX_HEADS * geo::IDX_DIM,
            m,
            &mut qa.qi,
        )?;
    }
    Ok(())
}

/// A selecting layer's attention over one sequence's `m` rows and its
/// K/V planes: the selection of each row's pooled keys (into `sel` from its
/// first row), the q/k norm and turn with the K/V append at the rows'
/// positions, then the selected flash over the selection's lists (its
/// partials from their first row) into `io.y`.
fn qsa_attend(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    (kv, pooled): (&mut super::scratch::KvPlanes, &mut DeviceBuffer<u16>),
    io: AttendIo<'_>,
    (sel, part_v, part_ms): (
        &mut QsaScratch,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let AttendIo {
        qsel,
        n_keys,
        qg,
        q,
        k,
        v,
        pos,
        y,
    } = io;
    c.k.qsa.enqueue_select(
        stream,
        SelectArgs {
            q: qsel,
            gain: f32_gain(w, &qp.idx_q_norm)?,
            table: c.table,
            n_keys,
            pooled: &*pooled,
            eps: c.eps,
            ctx,
            kept: geo::KEPT,
            m,
            fault: sink,
            scratch: &mut *sel,
        },
    )?;
    // The selecting store is f16 (the qwen38 family's stores carry no q8_0
    // form), so the family's shared append runs its f16 arm.
    let (kc, vc) = kv.f16_mut("qwen38::attention")?;
    c.k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
            qg,
            q: &mut *q,
            k,
            v,
            gq: f32_gain(w, &qp.q_norm)?,
            gk: f32_gain(w, &qp.k_norm)?,
            table: c.table,
            pos,
            eps: c.eps,
            n_head: geo::N_HEAD,
            n_kv: geo::N_KV,
            ctx,
            m,
            fault: sink,
            cache_k: kc,
            cache_v: vc,
        },
    )?;
    let (kc, vc) = kv.f16("qwen38::attention")?;
    let width = sel.width();
    c.k.flash.enqueue_pass_256_p12_sel(
        stream,
        GqaSelArgs {
            q,
            kc,
            vc,
            list: &sel.list,
            n_sel: &sel.n_sel,
            width,
            scale: ATTN_SCALE_256,
            n_kv: geo::N_KV,
            ctx,
            m,
            part_v,
            part_ms,
            fault: sink,
            y,
        },
        geo::N_HEAD,
    )
}

/// A selecting layer's gate over every row's attention and its output
/// projection, into `s.y`.
fn qsa_out(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    s: &mut Arena38,
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let Arena38 {
        qsa: qa,
        flash,
        attn,
        y,
        ..
    } = s;
    c.k.q38.enqueue_out_gate(
        c.gpu.stream(),
        OutGateArgs {
            attn: &*flash,
            qg: &qa.qg,
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
        self.p.handoff_go(hy, l, |k, h, target, sel| {
            k.enqueue_handoff(stream, h, target, sink, sel)
        })
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
        self.p.step_back(port.hybrid(), l)?;
        port.hybrid().row_enqueued(l, 0)
    }

    /// The head's mix into its input, then the head.
    fn end(&mut self, _unit: usize) -> Result<(), GpuError> {
        self.p.head_mix(Some(self.head.input_mut()))?;
        self.head.enqueue(self.p.c.gpu, self.p.c.w)
    }
}

/// The captured verify's walk `(1, m, Step)` over `m` rows — a verify's
/// consecutive positions of one sequence, or a pass of several slots'
/// rows, each slot's consecutive positions ([`Seqs38::Slots`]): every
/// layer through the step port's `Cols(m)` chain — one handoff launch
/// writing the `m` columns' image, one go, one union call on the host, one
/// wait; at one row the step's chain and handoff — each delta row's state
/// into a lane of its own, then the head's mix over the `m` columns into
/// `head`, a head of `m` rows (one projection and one argmax for every
/// row), and each of `copies` after it. Each row is bit for bit its step.
pub(super) struct Verify38<'a> {
    pub(super) p: Parts38<'a>,
    pub(super) head: &'a mut Head,
    pub(super) copies: Vec<RowsCopy<'a>>,
}

/// A parked slot's rows of a pass of several slots copied, once the head's
/// mix has made the walk's final streams, to the slot's own final-stream
/// rows from row 0: where its draft's walk reads the target's hidden rows
/// of its last call, as after a verify of its rows alone.
pub(super) struct RowsCopy<'a> {
    pub(super) rows: Range<usize>,
    pub(super) to: &'a mut DeviceBuffer<f32>,
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
        self.p.handoff_go(hy, l, |k, h, target, sel| match m {
            1 => k.enqueue_handoff(stream, h, target, sink, sel),
            m => k.enqueue_handoff_cols(stream, h, geo::N_USED + 1, m, target, sink, sel),
        })
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
        self.p.step_back(port.hybrid(), l)?;
        port.hybrid().row_enqueued(l, 0)
    }

    /// The head's mix over the `m` columns into the head's input, then the
    /// head, then the copies of the parked slots' rows.
    fn end(&mut self, _unit: usize) -> Result<(), GpuError> {
        self.p.head_mix(Some(self.head.input_mut()))?;
        self.head.enqueue(self.p.c.gpu, self.p.c.w)?;
        let (wide, stream) = (geo::STREAMS * geo::HIDDEN, self.p.c.gpu.stream());
        let res = &self.p.s.res[self.p.cur];
        for c in &mut self.copies {
            let from = rows_of(res, &c.rows, wide)?;
            rows_mut(c.to, &(0..c.rows.len()), wide)?.copy_from_device_async(&from, stream)?;
        }
        Ok(())
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
