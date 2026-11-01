//! Qwen3.8's layer program: [`runtime::sched::walk`] over one unit of `m`
//! columns — the decode step `(1, 1, Step)` through the host tier's step port
//! ([`Step38`], captured), a verify `(1, 2 <= m <= 4, Step)` through the same
//! port's `Cols(m)` chain ([`Verify38`], captured), an eager pass `(1, m <=
//! 8, Batch)` through its batch port ([`Pass38`]). The walks share every
//! card launch ([`Parts38`]); they differ in how a layer's routed experts
//! reach the host and come back, and a verify records each row's delta
//! update in the store's record, leaving the committed lane for its commit
//! to replay into, where the step and the pass write the last row's state
//! back into the lane. A prompt's ubatch walk `(1, m, Batch)` over
//! the ubatch's rows is `wide38`'s, its launches the wide family's.
//!
//! A layer's parts, by its plan ([`Layer38`]), never its number:
//! - the front: the embedding in front of layer 0; the attention site's mix
//!   (its norm first applies the previous layer's combine in place; layer 0
//!   starts every stream from the embedding); on the PLE layer the previous
//!   combine alone, the PLE site into the other stream buffer, then a plain
//!   mix; the mixer — the gated delta rule ([`GdnPlan`]) or the selecting
//!   attention ([`QsaPlan`]), the projection it reads last (the delta rule's
//!   `z`, the attention's q) on the walk's [`Branch`] beside its other
//!   launches up to that read; the feed-forward site's mix, its combine of the
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
//! A layer the load runs without a host leg ([`Card38::hostless`]: every
//! routed expert on the card, no residency machine, no route trace) has
//! none in the step, the verify or the pass: its handoff writes the row's
//! image and the card's places and no go follows (the pass downloads
//! nothing and the walk serves nothing), its back has no wait, the host is
//! not told of it, and its back sums the retained card rows with the gated
//! shared expert in one launch, including the empty host list's zero addition.
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
//! layer with card experts ([`TIER_CARD_LAUNCHES`] on a tier or hostless layer); then
//! the head's mix ([`HEAD_MIX_LAUNCHES`]) and the head's own launches
//! ([`HeadLaunches`]). A layer without a host leg
//! ([`Card38::hostless`]) has no go and no wait.

use super::body::ATTN_SCALE_256;
use super::card38::{CARD_LAUNCHES, Card38, TIER_CARD_LAUNCHES};
use super::plan38::{
    GDN, GDN_GATE, GdnPlan, HcSite, Layer38, Mixer38, PlePlan, QsaPlan, geo, head_site,
};
use super::router::gated;
use super::scratch::{Append256, Io, cells_view};
use super::scratch38::{Arena38, Store38, Taps38};
use super::tier38::{STAGE_TIER, TierSide38};
use crate::card_slots::CardSlotKernels;
use crate::fault::{FaultSink, LAYER_HEAD};
use crate::flash_gqa::{FlashGqaKernels, GqaListArgs, GqaSelArgs, GqaSelSeqsArgs};
use crate::flash_gqa_prefill::FlashGqaPrefill;
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::graph::{Forked, StagedRows};
use crate::hc_gated::{
    Before, HcGatedKernels, HcMatrix, HcScratch, HcWideKernels, MixArgs, SiteWeights, Streams,
};
use crate::head::{Head, HeadLaunches};
use crate::host::handoff::{Handoff, HandoffKernels, Places};
use crate::host::run::HostRun;
use crate::host::{BatchLeg, StepLeg};
use crate::kv_format::KvQ8;
use crate::linear::conv::{ConvArgs, ConvRows};
use crate::linear::delta::{DeltaArgs, DeltaLanesArgs};
use crate::linear::norm_gate::NormGateArgs;
use crate::linear::seqs::{GdnSeq, GdnSeqsArgs};
use crate::linear::{self, LinearKernels};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::mtp::MtpKernels;
use crate::ple::{PleConvArgs, PleGateArgs, PleKernels};
use crate::q38::{CardSharedAddArgs, EmbedIo, Q38Kernels, SharedAddArgs};
use crate::qsa::{
    KeyAppendArgs, PoolArgs, PoolSeqsArgs, QsaKernels, QsaSeq, SelectArgs, SelectSeqsArgs,
};
use crate::rope_neox::{PartialNeoxSeqsArgs, RopeNeoxKernels};
use crate::rows::RowKernels;
use crate::site::{self, ByName, ByNameKernels, Out};
use crate::tensor::{DeviceTensor, Window, WindowMut};
use crate::weights::Weights;
use crate::{Branch, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
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
/// flash's two, its merge under the gate; the output projection.
pub(super) const QSA_LAUNCHES: usize = 13;
/// The PLE site's launches: the previous combine, the key and value
/// projections, the gate and the conv.
pub(super) const PLE_LAUNCHES: usize = 5;
/// A layer's two mixes.
pub(super) const MIX_LAUNCHES: usize = 2 * 3;
/// The block's launches in the step: the router, the handoff, the go, the
/// shared expert's four, the wait and the gated sum.
pub(super) const FFN_LAUNCHES: usize = 1 + 1 + 1 + 4 + 1 + 1;
/// Of the block's launches, the stream memory-operation batches: the go and
/// the wait, which a layer without a host leg ([`Card38::hostless`]) goes
/// without.
pub(super) const STEP_MEMOPS: usize = 2;
/// The head's mix: the last layer's combine and the final norm, into the
/// head's input. The head's own launches are [`HeadLaunches::at`]'s.
pub(super) const HEAD_MIX_LAUNCHES: usize = 3;

// A layer's launches past those the constants count are its by-name
// projections' ([`Layer38::by_extra`]): the quantization of each K-quant
// site's input rows, and past one row the token-major copy of a row-major
// type's result ([`site::by_name_extra`], from the resident types).

/// The captured decode step's launches for `plans` with `card`'s card
/// experts and `tier`'s tier layers (module doc).
pub(super) fn step_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    head: HeadLaunches,
) -> usize {
    walk_launches(plans, card, tier, 1, head)
}

/// The captured verify's launches for `plans` with `card`'s card experts and
/// `tier`'s tier layers at `m >= 2` rows: the step's, a delta layer's
/// joined β·α projection one launch at every `m`; the handoff, the go, the
/// wait and the card leg are one each whatever `m`, and the head's are
/// [`HeadLaunches::at`]'s.
pub(super) fn verify_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    m: usize,
    head: HeadLaunches,
) -> usize {
    walk_launches(plans, card, tier, m, head)
}

/// Layer `l`'s card leg retains its down rows for the join instead of
/// summing them ([`TIER_CARD_LAUNCHES`]): a tier layer of `tier` or a
/// layer `card`'s load runs without a host leg ([`Card38::hostless`]).
fn joins_rows(card: &Card38, tier: Option<&TierSide38>, l: usize) -> bool {
    tier.is_some_and(|t| t.k(l) > 0) || card.hostless(l)
}

fn walk_launches(
    plans: &[Layer38],
    card: &Card38,
    tier: Option<&TierSide38>,
    m: usize,
    head: HeadLaunches,
) -> usize {
    1 + plans
        .iter()
        .enumerate()
        .map(|(l, p)| {
            let mixer = match p.mixer {
                Mixer38::Gdn(_) => GDN_LAUNCHES,
                Mixer38::Qsa(_) => QSA_LAUNCHES,
            };
            let ffn = if card.hostless(l) {
                FFN_LAUNCHES - STEP_MEMOPS
            } else {
                FFN_LAUNCHES
            };
            MIX_LAUNCHES
                + mixer
                + p.by_extra(m)
                + ffn
                + p.ple.as_ref().map_or(0, |_| PLE_LAUNCHES)
                + match (card.has(l), joins_rows(card, tier, l)) {
                    (true, true) => TIER_CARD_LAUNCHES,
                    (true, false) => CARD_LAUNCHES,
                    (false, _) => 0,
                }
        })
        .sum::<usize>()
        + HEAD_MIX_LAUNCHES
        + head.at(m)
}

/// Every module a Qwen3.8 walk launches besides the `Gpu`'s own.
pub(super) struct Kernels38 {
    pub(super) q38: Q38Kernels,
    /// The card slots' places and sum.
    pub(super) slots: CardSlotKernels,
    pub(super) hc: HcGatedKernels,
    pub(super) ple: PleKernels,
    /// The PLE table's row decode.
    pub(super) rows: RowKernels,
    pub(super) qsa: QsaKernels,
    pub(super) neox: RopeNeoxKernels,
    pub(super) flash: FlashGqaKernels,
    pub(super) linear: LinearKernels,
    pub(super) router: gated::RouterKernels,
    pub(super) handoff: HandoffKernels,
    /// The ubatch walk's: the one-expert table's fill, the 32-value GEMM
    /// family, the prefill flash and the wide mixes.
    pub(super) gemm: GemmKernels,
    pub(super) g32: Gemm32Kernels,
    pub(super) prefill: FlashGqaPrefill,
    pub(super) hcw: HcWideKernels,
    /// The MTP draft layer's input pack and head argmax (`mtp38`).
    pub(super) mtp: MtpKernels,
    /// The K-quant gemvs and the token-major copy a by-name projection of a
    /// site that reads q8_1 rows launches; `None` on a load whose sites all
    /// read the f32 rows or q8 blocks of 32.
    pub(super) by: Option<ByNameKernels>,
}

impl Kernels38 {
    /// Load every module into `gpu`'s context, the by-name projections' when
    /// `by_name` (a site reads q8_1 rows). Load-time only.
    pub(super) fn load(gpu: &Gpu, by_name: bool) -> Result<Kernels38, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels38 {
            q38: Q38Kernels::load(ctx)?,
            slots: CardSlotKernels::load(ctx)?,
            hc: HcGatedKernels::load(ctx)?,
            ple: PleKernels::load(ctx)?,
            rows: RowKernels::load(ctx)?,
            qsa: QsaKernels::load(ctx)?,
            neox: RopeNeoxKernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            router: gated::RouterKernels::load(ctx)?,
            handoff: HandoffKernels::load(ctx)?,
            gemm: GemmKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            prefill: FlashGqaPrefill::load(ctx)?,
            hcw: HcWideKernels::load(ctx)?,
            mtp: MtpKernels::load(ctx)?,
            by: by_name.then(|| ByNameKernels::load(gpu)).transpose()?,
        })
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
    /// `y = W · x` for the dense weight `name` over `m` columns of the f32
    /// rows `x`, token-major, at the site type of its resident form
    /// ([`site::resident_gemv_cols`]): the draft layer's sites, which
    /// `Mtp38::arm` checks read the f32 rows or q8 planes (`site::BY_NAME`);
    /// a K-quant site is [`Ctx38::project`]'s.
    pub(super) fn dense_gemv(
        &self,
        name: &str,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        site::resident_gemv_cols_on(
            self.gpu,
            self.gpu.stream(),
            Some(&self.k.g32),
            (self.w, name),
            x,
            m,
            y,
        )
    }

    /// `out = W · x` for the dense weight `name` over `m` columns, token-major:
    /// the common dense site at the type of its resident form, by name
    /// ([`ByName::gemv_on`]; `plan38::plans` checks it is one). `bn` holds the
    /// forms of the stream the projection runs on, `sink` the layer's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the forms, the site, the rows, their count, the sink and the output (rust-quality R8)"
    )]
    pub(super) fn project(
        &self,
        bn: &mut ByName,
        name: &str,
        x: &DeviceBuffer<f32>,
        m: usize,
        sink: FaultSink,
        out: Out<'_>,
    ) -> Result<(), GpuError> {
        self.project_on(self.gpu.stream(), bn, name, x, m, sink, out)
    }

    /// [`Ctx38::project`] on `stream`, which the caller orders before the
    /// output's readers; `bn` is the one that stream's projections share.
    #[allow(
        clippy::too_many_arguments,
        reason = "the stream, the forms, the site, the rows, their count, the sink and the output (rust-quality R8)"
    )]
    pub(super) fn project_on(
        &self,
        stream: &CudaStream,
        bn: &mut ByName,
        name: &str,
        x: &DeviceBuffer<f32>,
        m: usize,
        sink: FaultSink,
        out: Out<'_>,
    ) -> Result<(), GpuError> {
        bn.gemv_on(
            self.gpu,
            stream,
            (self.k.by.as_ref(), Some(&self.k.g32)),
            (self.w, name),
            sink,
            x,
            m,
            out,
        )
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
                    down: HcMatrix::of(w, WHAT, &site.down)?,
                    up: HcMatrix::of(w, WHAT, &site.up)?,
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
    /// A verify's row mode: each row's delta update into the store's record
    /// (`linear::delta`'s `enqueue_delta_recovery`); else the last row's state
    /// back into the committed lane.
    pub(super) each: bool,
}

/// The sequences a walk's unit binds: one over every row of the unit — the
/// step's, a verify's, a pass's — its stores and input record handed to the
/// launches whole; or several, each over its own rows of the unit (a pass of
/// several slots), each sequence-bound launch run over each sequence's rows
/// and its own stores: once over every sequence where a launch takes them
/// all ([`seq_launches`] names the rest).
pub(super) enum Seqs38<'a> {
    One {
        stores: &'a mut [Store38],
        ple_ring: &'a mut DeviceBuffer<f32>,
        io: &'a Io<'a>,
    },
    Slots(Vec<SlotSeq38<'a>>),
}

/// One busy slot of a pass of several: its stores, its PLE ring, its input
/// record (its first position, its ids and its lane word), the rows of
/// the pass it holds and the rotation table its rope launches read.
pub(super) struct SlotSeq38<'a> {
    pub(super) stores: &'a mut [Store38],
    pub(super) ple_ring: &'a mut DeviceBuffer<f32>,
    pub(super) io: Io<'a>,
    pub(super) rows: Range<usize>,
    /// The rotation table the slot's rope launches read: its own on a media
    /// load, else the load's.
    pub(super) table: &'a DeviceBuffer<f32>,
}

/// The launches one more sequence's rows add to a pass of several slots
/// over a cache in the format `kv`: its embedding, the PLE site's conv, and
/// on a q8_0 cache each selecting layer's [`QSA_Q8_SEQ_LAUNCHES`]. Every other
/// launch that binds a sequence's stores runs once over every sequence's rows
/// (`crate::seq_rows`): the delta layers' conv and step, the selecting
/// layers' pool, key append and selection, and on an f16 cache their q/k
/// norm, turn and append and the selected flash.
pub(super) fn seq_launches(plans: &[Layer38], kv: KvQ8) -> usize {
    1 + plans
        .iter()
        .map(|p| {
            let per_seq = match (&p.mixer, kv) {
                (Mixer38::Qsa(_), KvQ8::Q8) => QSA_Q8_SEQ_LAUNCHES,
                _ => 0,
            };
            per_seq + usize::from(p.ple.is_some())
        })
        .sum::<usize>()
}

/// A selecting layer's launches bound to one sequence's q8_0 cache, which no
/// launch over several sequences reads: the q/k norm, turn and append, and
/// the selected flash's two.
pub(super) const QSA_Q8_SEQ_LAUNCHES: usize = 1 + 2;

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

/// A delta store's state and ring, stamps and verify record, lent for one
/// launch.
type RecOf<'a> = (
    &'a mut super::scratch::RecStore,
    &'a mut DeviceBuffer<u32>,
    &'a mut DeviceBuffer<f32>,
);

/// Layer `l`'s delta store of `stores`: its state, ring, stamps and verify
/// record.
fn rec_of(stores: &mut [Store38], l: usize) -> Result<RecOf<'_>, GpuError> {
    match stores.get_mut(l) {
        Some(Store38::Rec { rec, stamp, record }) => Ok((rec, stamp, record)),
        Some(Store38::Qsa { .. }) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}'s plan and store are of two kinds"),
        )),
        None => Err(GpuError::state(WHAT, "a store for every layer")),
    }
}

/// Layer `l`'s selecting store of `stores`: its K/V planes, raw key ring and
/// pooled keys.
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
        Some(Store38::Qsa { kv, ring, pooled }) => Ok((&mut **kv, ring, &mut **pooled)),
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
    /// On layer `l` with the PLE site, the walk's rows of `arrival`'s row 0
    /// enqueued over the arena's PLE values ([`StagedRows::enqueue_prefix`]):
    /// the rows the host delivers after the launch reach the site before it
    /// reads them.
    fn arrive(&mut self, arrival: Option<&StagedRows<f32>>, l: usize) -> Result<(), GpuError> {
        if let Some(arrival) = arrival
            && plan_of(self.plans, l)?.ple.is_some()
        {
            let len = self.m * geo::HIDDEN;
            arrival.enqueue_prefix(self.c.gpu.stream(), 0, len, &mut self.s.ple.e, 0)?;
        }
        Ok(())
    }

    /// Layer `l`'s front up to its feed-forward site: the embedding in front
    /// of layer 0, the attention site (with the PLE site on its layer), the
    /// mixer into `y`, one of its projections on `branch` (module doc). Each
    /// launch bound to a sequence runs once a sequence of the unit
    /// ([`Seqs38`]).
    fn front(&mut self, branch: &mut Branch, l: usize) -> Result<(), GpuError> {
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
            c.project(&mut s.by, &pp.key, &pl.e, m, sink, Out::whole(&mut pl.key))?;
            c.project(
                &mut s.by,
                &pp.value,
                &pl.e,
                m,
                sink,
                Out::whole(&mut pl.value),
            )?;
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
            Mixer38::Gdn(g) => gdn(c, g, branch, &mut self.seqs, s, (l, m, self.each), sink),
            Mixer38::Qsa(q) => qsa(c, q, branch, &mut self.seqs, s, (l, m), sink),
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
            gated::RouterWeight::of(self.c.w, WHAT, &p.ffn.router)?,
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
    fn shared(
        &mut self,
        (stream, forked): (&CudaStream, bool),
        l: usize,
        x: Option<&DeviceBuffer<f32>>,
    ) -> Result<(), GpuError> {
        let p = plan_of(self.plans, l)?;
        let sink = self.c.gpu.layer_sink(l)?;
        let (c, m) = (&self.c, self.m);
        let s = &mut *self.s;
        let x = x.unwrap_or(&s.ffn_x);
        let bn = if forked { &mut s.by_fork } else { &mut s.by };
        let f = &p.ffn;
        c.project_on(stream, bn, &f.gate_sh, x, m, sink, Out::whole(&mut s.sh_g))?;
        c.project_on(stream, bn, &f.up_sh, x, m, sink, Out::whole(&mut s.sh_u))?;
        c.gpu
            .elem()
            .enqueue_swiglu(stream, &s.sh_g, &s.sh_u, geo::FF * m, &mut s.sh_h)?;
        c.project_on(
            stream,
            bn,
            &f.down_sh,
            &s.sh_h,
            m,
            sink,
            Out::whole(&mut s.sh_y),
        )
    }

    /// Layer `l`'s shadow over `x` (`None`: the arena's `ffn_x`): the shared
    /// expert on `branch` beside the card leg on the main stream, joined
    /// before the main stream's next launch, so the block's sum reads both.
    /// A layer without card experts has no card leg to run beside, so its
    /// shared expert runs on the main stream and the layer forks nothing.
    /// Every walk here holds at most [`super::scratch38::PASS_ROWS`] rows.
    fn beside(
        &mut self,
        branch: &mut Branch,
        l: usize,
        x: Option<&DeviceBuffer<f32>>,
    ) -> Result<(), GpuError> {
        if !self.card.has(l) {
            let gpu = self.c.gpu;
            return self.shared((gpu.stream(), false), l, x);
        }
        let fork = branch.fork()?;
        self.shared((fork.stream(), true), l, x)?;
        self.card(l, x)?;
        fork.join()
    }

    /// Layer `l`'s card leg over `x` (`None`: the arena's `ffn_x`): the
    /// places in the arena's `sel`, the router's weights, into the card sum
    /// ([`Card38::enqueue`]), or retained down rows for a tier or hostless
    /// layer's join. Refuses a layer without card experts by name.
    fn card(&mut self, l: usize, x: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let s = &*self.s;
        let x = x.unwrap_or(&s.ffn_x);
        if joins_rows(self.card, self.tier.as_deref(), l) {
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
    /// A layer without a host leg ([`Card38::hostless`]) has no go.
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
            if self.card.hostless(l) {
                return Ok(());
            }
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
    /// host's sum ([`Parts38::shared_add`]). A layer without a host leg
    /// ([`Card38::hostless`]) has no wait and fuses its card sum and shared add.
    fn step_back(
        &mut self,
        hy: &mut crate::hybrid::Hybrid<HostRun>,
        l: usize,
    ) -> Result<(), GpuError> {
        let stream = self.c.gpu.stream();
        let n_tier = self.tier_k(l);
        if n_tier == 0 {
            if self.card.hostless(l) {
                return self.shared_add(l, None);
            }
            let b = hy.boundary();
            b.enqueue_back_of(stream, 0, l)?;
            return self.shared_add(l, Some(b.hsum_of(0)?));
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

    /// Layer `l`'s block output into `y`: the host's routed sum `hsum`
    /// (`None`: a fused hostless join), on a layer with card
    /// experts the card's, plus the shared expert's output times its gate
    /// weight.
    fn shared_add(&mut self, l: usize, hsum: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        let Some(hsum) = hsum else {
            let s = &mut *self.s;
            return self.card.enqueue_hostless_join(
                &self.c,
                (l, self.m),
                (&s.sel, &s.sh_y, &s.route.weights),
                &mut s.y,
            );
        };
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
    const WHAT: &str = "qwen3moe::program38::embed";
    let name = model::arch::qwen35moe::names::token_embd();
    let Arena38 {
        emb, pos, n_keys, ..
    } = s;
    match seqs {
        Seqs38::One { io, .. } => c.k.q38.enqueue_embed_lookup(
            c.gpu,
            WHAT,
            (c.w, &name),
            EmbedIo {
                ids: io.ids,
                pos0: io.pos0,
                first: io.first,
                width: geo::HIDDEN,
                y: emb,
                pos,
                n_keys,
            },
        ),
        Seqs38::Slots(seqs) => {
            for q in seqs {
                let r = &q.rows;
                c.k.q38.enqueue_embed_lookup(
                    c.gpu,
                    WHAT,
                    (c.w, &name),
                    EmbedIo {
                        ids: q.io.ids,
                        pos0: q.io.pos0,
                        first: q.io.first,
                        width: geo::HIDDEN,
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
                out: Streams::F32(out),
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
                        out: Streams::F32(&mut *rows_mut(out, r, wide)?),
                        ring: &mut *q.ple_ring,
                    },
                )?;
            }
            Ok(())
        }
    }
}

/// The rows a delta layer's launches bound to one sequence read and write:
/// the conv's inputs (`x`, the joined β·α rows `ba`, each row's position)
/// and outputs, and the delta step's output `o`.
struct GdnIo<'b> {
    x: &'b DeviceBuffer<f32>,
    ba: &'b DeviceBuffer<f32>,
    pos: &'b DeviceBuffer<u32>,
    conv: &'b mut DeviceBuffer<f32>,
    beta: &'b mut DeviceBuffer<f32>,
    decay: &'b mut DeviceBuffer<f32>,
    o: &'b mut DeviceBuffer<f32>,
}

/// A delta layer's mixer at `m` rows, layer `l` of each sequence of
/// `seqs`, the attention site's mix in `s.mixed`, its output projection into
/// `s.y`: the projections over every row, `z` on `branch` ([`gdn_in`]), the
/// conv and delta step over each sequence's rows and its store, in row mode
/// when `each`, else in place ([`gdn_seq`]; several sequences' in one launch
/// each, `linear::seqs`), the join, then the gated norm and the output
/// projection over every row ([`gdn_out`]).
fn gdn(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    branch: &mut Branch,
    seqs: &mut Seqs38<'_>,
    s: &mut Arena38,
    (l, m, each): (usize, usize, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let fork = gdn_in(c, gp, branch, s, (m, sink))?;
    {
        let Arena38 { gdn: g, pos, .. } = &mut *s;
        match seqs {
            Seqs38::One { stores, io, .. } => {
                let lane = lane_of(io)?;
                gdn_seq(
                    c,
                    gp,
                    (rec_of(stores, l)?, lane),
                    GdnIo {
                        x: &g.x,
                        ba: &g.ba,
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
                let mut recs = Vec::with_capacity(seqs.len());
                for q in seqs.iter_mut() {
                    let (rec, stamp, record) = rec_of(q.stores, l)?;
                    recs.push(GdnSeq {
                        rows: q.rows.clone(),
                        lane: lane_of(&q.io)?,
                        rec,
                        stamped: Some((stamp, each.then_some(record))),
                    });
                }
                let (b, a) = beta_alpha_rows(&g.ba, m)?;
                let w = c.w;
                linear::seqs::enqueue_gdn_seqs(
                    &c.k.linear,
                    c.gpu.stream(),
                    GdnSeqsArgs {
                        x: &g.x,
                        b_raw: &b,
                        a_raw: &a,
                        ba_pitch: BA_PITCH,
                        w: f32_gain(w, &gp.conv)?,
                        dt_bias: f32_gain(w, &gp.dt_bias)?,
                        ssm_a: f32_gain(w, &gp.ssm_a)?,
                        pos,
                        shape: GDN,
                        eps: c.eps,
                        fault: sink,
                        conv: &mut g.conv,
                        beta: &mut g.beta,
                        decay: &mut g.decay,
                        o: &mut g.o,
                    },
                    &mut recs,
                )?;
            }
        }
    }
    fork.join()?;
    gdn_out(c, gp, s, m, sink)
}

/// A delta layer's projections over every row of `s.mixed`: q·k·v, then
/// `branch` forked, the joined β·α into `ba` in one launch
/// ([`beta_alpha_rows`]) and `z` on the branch, beside the conv and the
/// delta step. The fork returned is joined before the gated norm reads `z`.
fn gdn_in<'b>(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    branch: &'b mut Branch,
    s: &mut Arena38,
    (m, sink): (usize, FaultSink),
) -> Result<Forked<'b>, GpuError> {
    let Arena38 {
        mixed,
        gdn: g,
        by,
        by_fork,
        ..
    } = s;
    c.project(by, &gp.qkv, mixed, m, sink, Out::whole(&mut g.x))?;
    let fork = branch.fork()?;
    if m == 1 {
        c.project(by, &gp.beta_alpha, mixed, 1, sink, Out::whole(&mut g.ba))?;
    } else {
        let g32 = Some(&c.k.g32);
        site::resident_gemv_cols(c.gpu, g32, (c.w, &gp.beta_alpha), mixed, m, &mut g.ba)?;
    }
    c.project_on(
        fork.stream(),
        by_fork,
        &gp.z,
        mixed,
        m,
        sink,
        Out::whole(&mut g.z),
    )?;
    Ok(fork)
}

/// Values a row of the joined β·α projection holds: β's `n_v`, then α's.
pub(super) const BA_PITCH: usize = 2 * GDN.n_v;

/// β's and α's values of the joined projection `ba` over `m` rows,
/// token-major, [`BA_PITCH`] values a row: the window from its first β and
/// the one from its first α, each to the last row's last value of its own,
/// as the conv reads them (`ConvArgs::ba_pitch`).
pub(super) fn beta_alpha_rows(
    ba: &DeviceBuffer<f32>,
    m: usize,
) -> Result<(Window<'_, f32>, Window<'_, f32>), GpuError> {
    let nv = GDN.n_v;
    let n = (BA_PITCH * m).saturating_sub(nv);
    Ok((
        Window::of(ba, 0, n)?,
        Window::of(ba, nv * size_of::<f32>(), n)?,
    ))
}

/// A delta layer's launches bound to one sequence, over `io`'s `m` rows:
/// the conv over the store's ring `r.ring`, then the delta step over its
/// state, its lane's stamp and the lane word `lane`: in place, or a verify
/// recorded into the store's record when `each`.
fn gdn_seq(
    c: &Ctx38<'_>,
    gp: &GdnPlan,
    ((r, stamp, record), lane): (RecOf<'_>, &DeviceBuffer<u32>),
    io: GdnIo<'_>,
    (m, each): (usize, bool),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let GdnIo {
        x,
        ba,
        pos,
        conv,
        beta,
        decay,
        o,
    } = io;
    let (b, a) = beta_alpha_rows(ba, m)?;
    let lin = &c.k.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: ConvRows::F32(x),
            b_raw: &b,
            a_raw: &a,
            ba_pitch: BA_PITCH,
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
    let args = DeltaLanesArgs {
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
    };
    if each {
        lin.delta
            .enqueue_delta_recovery(stream, args, r.carry.as_ref(), record)
    } else {
        lin.delta
            .enqueue_delta_lanes(stream, args, r.carry.as_ref())
    }
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
        gdn: g,
        attn,
        y,
        by,
        ..
    } = s;
    c.k.linear.norm_gate.enqueue_norm_gate(
        stream,
        NormGateArgs {
            o: &g.o,
            z: &g.z,
            w: f32_gain(w, &gp.ssm_norm)?,
            gate: GDN_GATE,
            eps: c.eps,
            n_v: nv,
            m,
            fault: sink,
            y: &mut *attn,
        },
    )?;
    debug_assert_eq!(nv * linear::HEAD, geo::ATTN);
    c.project(by, &gp.ssm_out, &*attn, m, sink, Out::whole(y))
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

/// The rows a selecting layer's attention over its K/V planes reads and
/// writes: the query projection's `[q | gate]` rows, the normed queries,
/// the keys and values, each row's position, the selection's lists of
/// `width` a row and lengths, and the flash's output under its gate.
struct AttendIo<'b> {
    qg: &'b DeviceBuffer<f32>,
    q: &'b mut DeviceBuffer<f32>,
    k: &'b mut DeviceBuffer<f32>,
    v: &'b DeviceBuffer<f32>,
    pos: &'b DeviceBuffer<u32>,
    list: &'b DeviceBuffer<u32>,
    width: usize,
    n_sel: &'b DeviceBuffer<u32>,
    y: &'b mut DeviceBuffer<f32>,
}

/// A selecting attention layer's mixer at `m` rows, layer `l` of each
/// sequence of `seqs`, the attention site's mix in `s.mixed`, its output
/// projection into `s.y`: q, k and v over every row, q on `branch`
/// ([`qsa_qkv`]); the indexer keys' projection over every row, the pools the
/// rows complete and the keys' append to each sequence's ring ([`qsa_keys`]
/// for one sequence, one launch each over several, `crate::seq_rows`); the
/// indexer queries over every row ([`qsa_query`]) and the join; each row's
/// selection over its sequence's pooled keys; then the q/k norm, turn and
/// append and the selected flash over each sequence's K/V planes
/// ([`kv_attend`]; over several f16 caches one launch each), under each
/// head's gate into `s.attn`; then the output projection over every row.
fn qsa(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    branch: &mut Branch,
    seqs: &mut Seqs38<'_>,
    s: &mut Arena38,
    (l, m): (usize, usize),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let fork = qsa_qkv(c, qp, branch, s, (m, sink))?;
    match seqs {
        Seqs38::One { stores, .. } => {
            let (kv, ring, pooled) = qsa_of(stores, l)?;
            {
                let Arena38 {
                    mixed,
                    qsa: qa,
                    pos,
                    n_keys,
                    by,
                    ..
                } = &mut *s;
                let keys = KeysIo {
                    x: mixed,
                    kr: &mut qa.kr,
                    pos,
                    n_keys,
                };
                qsa_keys(c, qp, by, (ring, pooled), keys, (m, sink))?;
            }
            qsa_query(c, qp, s, (m, sink))?;
            fork.join()?;
            let Arena38 {
                qsa: qa,
                pos,
                n_keys,
                attn,
                ..
            } = &mut *s;
            c.k.qsa.enqueue_select(
                c.gpu.stream(),
                SelectArgs {
                    q: &qa.qi,
                    gain: f32_gain(c.w, &qp.idx_q_norm)?,
                    table: c.table,
                    n_keys,
                    pooled: &*pooled,
                    eps: c.eps,
                    ctx: c.ctx,
                    kept: geo::KEPT,
                    m,
                    fault: sink,
                    scratch: &mut qa.sel,
                },
            )?;
            let io = AttendIo {
                qg: &qa.qg,
                q: &mut qa.q,
                k: &mut qa.k,
                v: &qa.v,
                pos,
                list: &qa.sel.list,
                width: qa.sel.width(),
                n_sel: &qa.sel.n_sel,
                y: attn,
            };
            // SAFETY: the front embedding generates consecutive positions within this
            // sequence's window; this call selects only that sequence's cache planes.
            unsafe { kv_attend(c, qp, kv, io, (&mut qa.part_v, &mut qa.part_ms), m, sink) }?;
        }
        Seqs38::Slots(seqs) => qsa_slots(c, qp, fork, seqs, s, (l, m), sink)?,
    }
    let Arena38 { attn, y, by, .. } = s;
    c.project(by, &qp.out, &*attn, m, sink, Out::whole(y))
}

/// [`qsa`]'s launches bound to the sequences of a pass of several, `fork`
/// the branch [`qsa_qkv`] forked: the indexer keys' projection over every
/// row, then over every sequence's rows in one launch each, each over its
/// own store and rotation table (`crate::seq_rows`), the pools, the keys'
/// append, the indexer queries' projection and the join, and the selection;
/// then on f16 caches the q/k norm, turn and append and the selected flash in
/// one launch each, on q8_0 caches [`kv_attend`] once a sequence over its
/// rows' windows.
fn qsa_slots(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    fork: Forked<'_>,
    seqs: &mut [SlotSeq38<'_>],
    s: &mut Arena38,
    (l, m): (usize, usize),
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream) = (c.w, c.gpu.stream());
    let mut keys = Vec::with_capacity(seqs.len());
    let mut planes = Vec::with_capacity(seqs.len());
    for q in seqs.iter_mut() {
        let (kv, ring, pooled) = qsa_of(q.stores, l)?;
        keys.push(QsaSeq {
            rows: q.rows.clone(),
            ring,
            pooled,
            rope: q.table,
        });
        planes.push((q.rows.clone(), kv, q.table));
    }
    {
        let Arena38 {
            mixed,
            qsa: qa,
            pos,
            n_keys,
            by,
            ..
        } = &mut *s;
        c.project(by, &qp.idx_k, mixed, m, sink, Out::whole(&mut qa.kr))?;
        let k = &c.k.qsa;
        k.enqueue_pool_seqs(
            stream,
            PoolSeqsArgs {
                kr: &qa.kr,
                gain: f32_gain(w, &qp.idx_k_norm)?,
                n_keys,
                eps: c.eps,
                ctx: c.ctx,
                fault: sink,
            },
            &keys,
        )?;
        k.enqueue_key_append_seqs(stream, (&qa.kr, pos), (c.ctx, sink), &keys)?;
    }
    qsa_query(c, qp, s, (m, sink))?;
    fork.join()?;
    let Arena38 {
        qsa: qa,
        pos,
        n_keys,
        attn,
        ..
    } = &mut *s;
    c.k.qsa.enqueue_select_seqs(
        stream,
        SelectSeqsArgs {
            q: &qa.qi,
            gain: f32_gain(w, &qp.idx_q_norm)?,
            n_keys,
            eps: c.eps,
            ctx: c.ctx,
            kept: geo::KEPT,
            fault: sink,
            scratch: &mut qa.sel,
        },
        &keys,
    )?;
    drop(keys);
    let (sel, width) = (&qa.sel, qa.sel.width());
    if planes.iter().all(|(_, kv, _)| kv.seqs_serve()) {
        let kv = planes
            .iter_mut()
            .map(|(rows, kv, rope)| kv.kv_seq(rows.clone(), rope))
            .collect::<Result<Vec<_>, GpuError>>()?;
        // SAFETY: the front embedding generates consecutive positions within
        // each sequence's window, and each sequence appends into its own
        // planes.
        unsafe {
            c.k.neox.enqueue_head_norm_neox_append_256_seqs(
                stream,
                &kv,
                PartialNeoxSeqsArgs {
                    qg: &qa.qg,
                    q: &mut qa.q,
                    k: &mut qa.k,
                    v: &qa.v,
                    gq: f32_gain(w, &qp.q_norm)?,
                    gk: f32_gain(w, &qp.k_norm)?,
                    pos,
                    eps: c.eps,
                    n_head: geo::N_HEAD,
                    n_kv: geo::N_KV,
                    ctx: c.ctx,
                    fault: sink,
                },
            )
        }?;
        return c.k.flash.enqueue_pass_256_p12_sel_seqs(
            stream,
            &kv,
            GqaSelSeqsArgs {
                q: &qa.q,
                list: &sel.list,
                n_sel: &sel.n_sel,
                width,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx: c.ctx,
                fault: sink,
                y: attn,
                part_v: &mut qa.part_v,
                part_ms: &mut qa.part_ms,
                gate: Some(&qa.qg),
            },
            geo::N_HEAD,
        );
    }
    for (r, kv, rope) in &mut planes {
        // The slot's own rotation table: an image shifts one sequence's rows
        // and no other's.
        let c = &Ctx38 { table: rope, ..*c };
        let pos = rows_of(pos, r, 1)?;
        let io = AttendIo {
            qg: &*rows_of(&qa.qg, r, geo::Q_ROWS)?,
            q: &mut *rows_mut(&mut qa.q, r, geo::ATTN)?,
            k: &mut *rows_mut(&mut qa.k, r, geo::KV)?,
            v: &*rows_of(&qa.v, r, geo::KV)?,
            pos: &pos,
            list: &*rows_of(&sel.list, r, width)?,
            width,
            n_sel: &*rows_of(&sel.n_sel, r, 1)?,
            y: &mut *rows_mut(attn, r, geo::ATTN)?,
        };
        let parts = (&mut qa.part_v, &mut qa.part_ms);
        // SAFETY: the front embedding generates consecutive positions within this
        // sequence's window; this call selects only that sequence's cache planes.
        unsafe { kv_attend(c, qp, kv, io, parts, r.len(), sink) }?;
    }
    Ok(())
}

/// A selecting layer's q (with each head's gate), k and v over every row
/// of `s.mixed`: `branch` forked, k and v, then q on the branch, beside
/// them and the indexer's keys and queries. The fork returned is joined
/// before the attention reads q.
fn qsa_qkv<'b>(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    branch: &'b mut Branch,
    s: &mut Arena38,
    (m, sink): (usize, FaultSink),
) -> Result<Forked<'b>, GpuError> {
    let Arena38 {
        mixed,
        qsa: qa,
        by,
        by_fork,
        ..
    } = s;
    let fork = branch.fork()?;
    c.project(by, &qp.k, mixed, m, sink, Out::whole(&mut qa.k))?;
    c.project(by, &qp.v, mixed, m, sink, Out::whole(&mut qa.v))?;
    c.project_on(
        fork.stream(),
        by_fork,
        &qp.q,
        mixed,
        m,
        sink,
        Out::whole(&mut qa.qg),
    )?;
    Ok(fork)
}

/// A selecting layer's indexer keys of one sequence's `m` rows: their
/// projection (token-major, a sequence's own launch), the pools those rows
/// complete, then the rows' raw keys appended to the store's ring.
fn qsa_keys(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    bn: &mut ByName,
    (ring, pooled): (&mut DeviceBuffer<u16>, &mut DeviceBuffer<u16>),
    io: KeysIo<'_>,
    (m, sink): (usize, FaultSink),
) -> Result<(), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let KeysIo { x, kr, pos, n_keys } = io;
    c.project(bn, &qp.idx_k, x, m, sink, Out::whole(&mut *kr))?;
    c.k.qsa.enqueue_pool(
        stream,
        PoolArgs {
            kr: &*kr,
            ring: &*ring,
            gain: f32_gain(w, &qp.idx_k_norm)?,
            table: c.table,
            n_keys,
            eps: c.eps,
            ctx,
            m,
            fault: sink,
            pooled,
        },
    )?;
    c.k.qsa.enqueue_key_append(
        stream,
        KeyAppendArgs {
            kr: &*kr,
            pos,
            m,
            ctx,
            fault: sink,
            ring,
        },
    )
}

/// A selecting layer's indexer queries over every row of `s.mixed` into
/// `qi`, token-major.
fn qsa_query(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    s: &mut Arena38,
    (m, sink): (usize, FaultSink),
) -> Result<(), GpuError> {
    let Arena38 {
        mixed, qsa: qa, by, ..
    } = s;
    c.project(by, &qp.idx_q, mixed, m, sink, Out::whole(&mut qa.qi))
}

/// A selecting layer's attention over one sequence's `m` rows and its K/V
/// planes, past their selection: the q/k norm and turn with the K/V append
/// at the rows' positions, then the selected flash over the selection's
/// lists (its partials `part_v` and `part_ms` from their first row) into
/// `io.y` under each head's sigmoid gate from `io.qg`.
///
/// # Safety
///
/// Among the first `m` positions in `io.pos`, those below `c.ctx` are
/// pairwise distinct.
unsafe fn kv_attend(
    c: &Ctx38<'_>,
    qp: &QsaPlan,
    kv: &mut super::scratch::KvPlanes,
    io: AttendIo<'_>,
    (part_v, part_ms): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    m: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (w, stream, ctx) = (c.w, c.gpu.stream(), c.ctx);
    let AttendIo {
        qg,
        q,
        k,
        v,
        pos,
        list,
        width,
        n_sel,
        y,
    } = io;
    // SAFETY: the enclosing helper requires distinct cache-valid positions.
    unsafe {
        kv.append_256(
            &c.k.neox,
            stream,
            Append256 {
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
            },
        )
    }?;
    c.k.flash.enqueue_pass_256_p12_sel(
        stream,
        GqaSelArgs {
            attn: GqaListArgs {
                q,
                kv: kv.planes(),
                list,
                n_sel,
                width,
                scale: ATTN_SCALE_256,
                n_kv: geo::N_KV,
                ctx,
                m,
                fault: sink,
                y,
            },
            part_v,
            part_ms,
            gate: Some(qg),
        },
        geo::N_HEAD,
    )
}

/// The captured decode step's walk `(1, 1, Step)`, into `head`. Its PLE
/// rows' `arrival` (one staged row's first position) is enqueued at its PLE
/// site ([`StagedRows::enqueue`]), as a verify's is.
pub(super) struct Step38<'a> {
    pub(super) branch: &'a mut Branch,
    pub(super) p: Parts38<'a>,
    pub(super) head: &'a mut Head,
    pub(super) arrival: &'a StagedRows<f32>,
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
        self.p.arrive(Some(self.arrival), l)?;
        self.p.front(self.branch, l)?;
        let hy = port.hybrid();
        self.p.ffn_mix(l, Some(hy.boundary_mut().normed_mut()))?;
        self.p.route(l, Some(hy.boundary().normed()))?;
        let sink = self.p.c.gpu.layer_sink(l)?;
        let stream = self.p.c.gpu.stream();
        self.p.handoff_go(hy, l, |k, h, target, sel| {
            k.enqueue_handoff(stream, h, target, sink, sel)
        })
    }

    /// The shared expert beside the card leg over the boundary's activation
    /// ([`Parts38::beside`]).
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let x = port.hybrid().boundary().normed();
        self.p.beside(self.branch, at.layer, Some(x))
    }

    /// The wait, the gated sum, and the host tier told the layer is enqueued
    /// (an eager step is served there), on a layer with a host leg.
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.step_back(port.hybrid(), l)?;
        if self.p.card.hostless(l) {
            return Ok(());
        }
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
/// wait; at one row the step's chain and handoff — each delta row's update
/// into its store's record, then the head's mix over the `m` columns into
/// `head`, a head of `m` rows (one projection and one argmax for every
/// row), and each of `copies` after it. Each row is bit for bit its step.
/// A verify's walk enqueues its PLE rows' `arrival` (one staged row of its
/// rows' values) at its PLE site, over the arena's rows
/// ([`StagedRows::enqueue`]); a pass of several slots has its rows in the
/// arena before its launch.
pub(super) struct Verify38<'a> {
    pub(super) branch: &'a mut Branch,
    pub(super) p: Parts38<'a>,
    pub(super) head: &'a mut Head,
    pub(super) copies: Vec<RowsCopy<'a>>,
    pub(super) arrival: Option<&'a StagedRows<f32>>,
}

/// The nodes a step's or a verify's PLE arrival adds to its capture
/// ([`StagedRows::enqueue`]): the flag's wait (one batch memory operation)
/// and the rows' copy.
pub(super) const PLE_ARRIVAL_NODES: usize = 2;

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
    /// head. A head of other rows than the walk's, or a verify of one
    /// sequence not in row mode, is refused by name before any launch: a
    /// pass of several slots runs in place when its commit keeps every row
    /// (`Body38::walk_slots`).
    pub(super) fn walk(mut self, leg: &mut StepLeg<'a, HostRun>) -> Result<(), GpuError> {
        let m = self.p.m;
        let one = matches!(self.p.seqs, Seqs38::One { .. });
        if self.head.m() != m || (one && !self.p.each) {
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
        self.p.arrive(self.arrival, l)?;
        self.p.front(self.branch, l)?;
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

    /// The shared expert beside the card leg over the boundary's
    /// activations ([`Parts38::beside`]).
    fn shadow(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let x = port.hybrid().boundary().normed();
        self.p.beside(self.branch, at.layer, Some(x))
    }

    /// The wait, the gated sum over the `m` columns' host sums, and the host
    /// tier told the layer is enqueued (an eager verify is served there), on
    /// a layer with a host leg.
    fn back(&mut self, port: &mut StepLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.step_back(port.hybrid(), l)?;
        if self.p.card.hostless(l) {
            return Ok(());
        }
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
/// the head and the final rows are the caller's ([`Pass38::head`],
/// [`Pass38::finals`]).
pub(super) struct Pass38<'a> {
    pub(super) branch: &'a mut Branch,
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
        let row = unsafe { cells_view(&self.p.s.mixed, (m - 1) * h, h) };
        head.input_mut()
            .copy_from_device_async(&row, gpu.stream())?;
        head.enqueue(gpu, self.p.c.w)
    }

    /// After the walk of a prompt call's pass that runs no head, when the
    /// call's tap reads its rows ([`super::body38::Prompt38Sink`]): the last
    /// layer's combine alone over the unit's columns, the one the head's mix
    /// applies first ([`Parts38::head_mix`]), so each row holds the target's
    /// final streams. One launch.
    pub(super) fn finals(&mut self) -> Result<(), GpuError> {
        let p = &mut self.p;
        let gpu = p.c.gpu;
        let fault = gpu.layer_sink(p.plans.len().saturating_sub(1))?;
        let s = &mut *p.s;
        p.c.k
            .hc
            .enqueue_combine(gpu.stream(), &mut s.res[p.cur], &s.y, p.m, fault, &s.hc)
    }
}

impl<'a> LayerProgram for Pass38<'a> {
    type Port = BatchLeg<'a, HostRun>;

    /// A layer the load runs without a host leg ([`Card38::hostless`]) has
    /// none in the pass either: the host run holds no view of it.
    fn host_leg(&self, at: At) -> bool {
        !self.p.card.hostless(at.layer)
    }

    /// The layer up to its router, the mix into the arena's `ffn_x`, then
    /// the download of the unit's activations and routed slots (the router's
    /// `N_USED + 1` a token, the routed `N_USED` of them; none on a layer
    /// without a host leg), and on a layer with card experts each routed
    /// slot's place into the arena's `sel` from the slot map's card copy.
    fn front(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let l = at.layer;
        self.p.front(self.branch, l)?;
        self.p.ffn_mix(l, None)?;
        self.p.route(l, None)?;
        let key = port.key(at);
        let stream = self.p.c.gpu.stream();
        let s = &mut *self.p.s;
        if !self.p.card.hostless(l) {
            port.hybrid().enqueue_download_pitched(
                stream,
                [&s.ffn_x, &s.route.weights],
                &s.route.ids,
                geo::N_USED + 1,
                key,
            )?;
        }
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

    /// The shared expert beside the card leg over the arena's `ffn_x`
    /// ([`Parts38::beside`]).
    fn shadow(&mut self, _: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        self.p.beside(self.branch, at.layer, None)
    }

    /// The gated sum over the host sums the walk's serve uploaded, or the
    /// fused card sum on a layer without a host leg.
    fn back(&mut self, port: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        let hsum = (!self.p.card.hostless(at.layer)).then(|| port.hsum());
        self.p.shared_add(at.layer, hsum)
    }
}
