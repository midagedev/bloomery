//! The step port's card leg over the NVMe tier's arena ([`ArenaLeg`]): on a
//! pinned arena the card reads, a service's routed ids the arena serves are
//! computed on the card from their slots instead of by the host.
//!
//! The service claims those ids and queues their fills
//! ([`NvTier::issue_card_hold`]), computes the row's other host slots into
//! the row's sum under the fills, joins the fills into a hold
//! ([`super::nvtier::CardTicket::finish`]: the ids filled and pinned) and
//! writes, for each of the row's routed slots, the word offsets of its
//! expert's gate, up and down parts in the arena and its place (0, the
//! arena's, or [`HOST`]) into the leg's own mapped tables. After the row's
//! wait the card runs, on the stream, the arena entries over those tables and
//! adds the arena slots' weighted sum to the row's sum, which the family's
//! combine then reads. A slot whose place is [`HOST`] is skipped by every
//! launch. The hold lasts until the row's next go, which the card enqueues
//! after these launches, so no claim takes a slot the card may still read.
//!
//! The card quantizes the activation as q8_1 (an f32 scale per 128 values),
//! where the host leg's K-quant dots take Q8_2X4 (a bf16 scale per 32), and
//! sums the held ids in its own order, so a step's logits are not the host
//! leg's bit for bit.
//!
//! The leg is one rule, not a device bit: it runs where the arena gives the
//! card an address (a pinned arena) and the page carries one column a row;
//! a load with expert tier cards keeps the host leg (their back is the
//! tier's), and so does a load whose residency machine runs (it frees arena
//! slots between passes, which a hold refuses). Every layer the arena covers
//! must have gate·up and down types the arena entries take, its parts on
//! 4-byte words and a slot for each id a pass's rows can hold at once, or
//! the leg is off for the load, by name ([`LegOff`]).

use std::sync::{Arc, Mutex};

use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::moe::HostLayer;
use model::ops::{ExpertForm, RowLayout};

use super::nvtier::NvTier;
use super::page::MAX_ROWS;
use super::slots::HOST;
use super::step::Boundary;
use crate::card_slots::{CardAcc, CardSlotKernels};
use crate::elem::ElemKernels;
use crate::fault::FaultSink;
use crate::graph::MappedHost;
use crate::kquant::Act;
use crate::kquant::sel::{ARENA_PARTS, ARENA_SKIP, GateUpAt, SelDownAt};
use crate::q4k_sel::{Q4kSelKernels, QuantSel};
use crate::q6k_sel::Q6kSelKernels;
use crate::quant_sel::QuantSelKernels;
use crate::tensor::WindowMut;
use crate::{Gpu, GpuError, Q8Act};

/// What the leg's errors name.
const WHAT: &str = "ArenaLeg";

/// Why a load runs no card leg over its arena: the `nvtier` record's
/// `card_leg` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegOff {
    /// The load never asked for a leg: its family attaches none.
    Unattached,
    /// The arena is anonymous memory: the card has no address of it.
    Anon,
    /// The load has expert tier cards, whose back runs the tier's leg.
    ExpertTier,
    /// The page carries more than one column a row.
    Cols,
    /// A covered layer's stacks are of a type no arena entry takes, read
    /// from the r8 sidecar, or of a form no gate·up rule runs (relu²).
    Types,
    /// A covered layer's part does not start on a 4-byte word in its slot,
    /// where the arena entries read it.
    Words,
    /// A covered layer has fewer slots than the ids a pass's rows hold at
    /// once (each row's routed width).
    Slots,
    /// The residency machine runs: it frees slots between passes.
    Residency,
}

impl LegOff {
    /// Every reason, in declaration order.
    pub const ALL: [LegOff; 8] = [
        LegOff::Unattached,
        LegOff::Anon,
        LegOff::ExpertTier,
        LegOff::Cols,
        LegOff::Types,
        LegOff::Words,
        LegOff::Slots,
        LegOff::Residency,
    ];

    /// The record's word.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            LegOff::Unattached => "unattached",
            LegOff::Anon => "off_anon",
            LegOff::ExpertTier => "off_expert_tier",
            LegOff::Cols => "off_cols",
            LegOff::Types => "off_types",
            LegOff::Words => "off_words",
            LegOff::Slots => "off_slots",
            LegOff::Residency => "off_residency",
        }
    }
}

/// The gate·up arena entry a layer's stacks take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GateUpAtEntry {
    Q4k,
    Q5k,
}

/// The down arena entry a layer's stack takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DownAtEntry {
    Q5k,
    Q6k,
}

/// One covered layer as the leg launches it.
#[derive(Clone, Copy)]
pub(super) struct LegLayer {
    pub(super) gate_up: GateUpAtEntry,
    pub(super) down: DownAtEntry,
    pub(super) rule: Act,
    pub(super) fault: FaultSink,
}

impl LegLayer {
    /// The entries for `layer`'s `[gate, up, down]` stack types; `None` when
    /// an arena entry is missing for one of them, a stack is read in
    /// another row layout than its file's (the r8 sidecar's), whose bytes
    /// the arena's slots then hold, or the layer's form has no gate·up rule
    /// ([`rule_of`]).
    fn of(host: &HostLayer, fault: FaultSink) -> Option<LegLayer> {
        if host.layouts().iter().any(|l| *l != RowLayout::Rows) {
            return None;
        }
        let [g, u, d] = host.stacks().map(|s| s.info().ty);
        let gate_up = match (g, u) {
            (GgmlType::Q4_K, GgmlType::Q4_K) => GateUpAtEntry::Q4k,
            (GgmlType::Q5_K, GgmlType::Q5_K) => GateUpAtEntry::Q5k,
            _ => return None,
        };
        let down = match d {
            GgmlType::Q5_K => DownAtEntry::Q5k,
            GgmlType::Q6_K => DownAtEntry::Q6k,
            _ => return None,
        };
        Some(LegLayer {
            gate_up,
            down,
            rule: rule_of(host.form())?,
            fault,
        })
    }
}

/// The gate·up rule the card leg's kernels run for a layer of `form`: the
/// gated forms'; `None` for relu², whose `relu(up·x)²` no gate·up kernel
/// runs, so its layer keeps the host leg.
fn rule_of(form: ExpertForm) -> Option<Act> {
    match form {
        ExpertForm::GatedOai { alpha, limit } => Some(Act::SwigluOai { alpha, limit }),
        ExpertForm::Gated { limit } => Some(Act::SwigluClamp { limit }),
        ExpertForm::Relu2 => None,
    }
}

/// Byte offsets of a row's tables in the leg's mapped page: the offsets
/// (`ARENA_PARTS` u64 a slot) then the places (one u32 a slot), each row at a
/// 256-byte boundary.
#[derive(Clone, Copy)]
struct TableAt {
    offs: usize,
    places: usize,
}

/// One row's windows the launches read: its tables, and in the boundary's
/// page its image's activation and routing weights and its sum.
struct RowWindows {
    at: TableAt,
    offs: WindowMut<'static, u64>,
    places: WindowMut<'static, u32>,
    x: WindowMut<'static, f32>,
    w: WindowMut<'static, f32>,
    hsum: WindowMut<'static, f32>,
}

/// The buffers the launches write, and per row a second window of its sum,
/// which the last launch writes the sum back through.
struct Scratch {
    act_x: Q8Act,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    down: DeviceBuffer<f32>,
    acc: DeviceBuffer<f32>,
    sums: Vec<WindowMut<'static, f32>>,
}

/// The card leg over the arena (module doc). Made at load
/// ([`ArenaLeg::new`]); its tables are written by the step port's services
/// and read by the launches [`ArenaLeg::enqueue`] puts after a row's wait.
///
/// Every window it holds views memory whose address a captured graph bakes
/// in, so the leg is made at load only, before any capture, and lives as
/// long as the load: a load whose residency machine runs builds none
/// ([`super::HostTier::attach_arena_leg`]), and the machine's start refuses
/// a load that has one.
pub struct ArenaLeg {
    /// Field order is drop order: the windows go before the tier and the
    /// tables whose memory they view.
    arena: WindowMut<'static, u32>,
    rows: Vec<RowWindows>,
    /// The x's place, 0: the activation is quantized as the arena's column.
    x_place: DeviceBuffer<u32>,
    tables: MappedHost,
    tier: Arc<NvTier>,
    layers: Vec<Option<LegLayer>>,
    /// The launches write the scratch, but they are enqueued through the
    /// boundary's shared borrow every family's chain holds
    /// ([`Boundary::enqueue_back_of`]); one stream enqueues them, so the
    /// lock is never contended.
    scratch: Mutex<Scratch>,
    kquant: QuantSelKernels,
    q6k: Q6kSelKernels,
    q4k: Q4kSelKernels,
    slots: CardSlotKernels,
    elem: ElemKernels,
    hidden: usize,
    ff: usize,
    n_used: usize,
}

impl ArenaLeg {
    /// The leg over `tier`'s arena for `boundary`'s rows, its layers the
    /// host run's `layers` (by model layer, `None` where a layer carries no
    /// host share), each of `ff` routed rows; `Ok(Err(why))` when the leg cannot run on this load (module
    /// doc). Load-time only, before any capture.
    pub(crate) fn new(
        gpu: &Gpu,
        boundary: &Boundary,
        tier: &Arc<NvTier>,
        layers: &[Option<HostLayer>],
        ff: usize,
    ) -> Result<Result<ArenaLeg, LegOff>, GpuError> {
        if tier.card_arena().is_none() {
            return Ok(Err(LegOff::Anon));
        }
        let layout = boundary.layout();
        if layout.cols() != 1 {
            return Ok(Err(LegOff::Cols));
        }
        let held = boundary.rows() * layout.handoff().n_used;
        let mut by_layer = vec![None; layers.len()];
        for (l, host) in layers.iter().enumerate() {
            let Some(host) = host else {
                continue;
            };
            if !tier.covers(l) {
                continue;
            }
            let Some(leg) = LegLayer::of(host, gpu.layer_sink(l)?) else {
                return Ok(Err(LegOff::Types));
            };
            if !tier.parts_on_words(l) {
                return Ok(Err(LegOff::Words));
            }
            if tier.layer_slots(l) < held {
                return Ok(Err(LegOff::Slots));
            }
            by_layer[l] = Some(leg);
        }
        ArenaLeg::build(gpu, boundary, tier, by_layer, ff).map(Ok)
    }

    /// The leg of [`ArenaLeg::new`] over the layers `layers` gives (by model
    /// layer, `None` for a layer the leg does not run), each of `ff` routed
    /// rows, on a pinned arena whose covered layers' parts start on words
    /// and hold every id `boundary`'s rows can hold at once: the load's
    /// checks are the caller's. Load-time only, before any capture.
    pub(super) fn build(
        gpu: &Gpu,
        boundary: &Boundary,
        tier: &Arc<NvTier>,
        layers: Vec<Option<LegLayer>>,
        ff: usize,
    ) -> Result<ArenaLeg, GpuError> {
        let (dev0, words) = tier
            .card_arena()
            .ok_or(GpuError::state(WHAT, "a pinned arena's card address"))?;
        let h = boundary.layout().handoff();
        let (hidden, n_used) = (h.hidden, h.n_used);
        let offs_bytes = (8 * ARENA_PARTS * n_used).next_multiple_of(256);
        let row_bytes = offs_bytes + (4 * n_used).next_multiple_of(256);
        let ctx = gpu.context();
        let tables = MappedHost::new(
            ctx,
            row_bytes * MAX_ROWS,
            "cuMemHostAlloc (arena leg tables)",
        )?;
        let page = boundary.page();
        let mut rows = Vec::with_capacity(boundary.rows());
        let mut sums = Vec::with_capacity(boundary.rows());
        for r in 0..boundary.rows() {
            let base = r * row_bytes;
            let at = TableAt {
                offs: base,
                places: base + offs_bytes,
            };
            let (image, hsum) = boundary.row_offsets(r)?;
            // SAFETY: the row's offsets, `ARENA_PARTS · n_used` u64 at a
            // 256-byte offset, lie inside `tables` (sized for MAX_ROWS rows);
            // the tables move into the leg beside the window, and the leg is
            // made at load and lives as long as the load (`ArenaLeg`'s doc),
            // so they outlive it and every graph that captured a launch over
            // it.
            let offs =
                unsafe { WindowMut::from_raw(tables.dev_at(at.offs), ARENA_PARTS * n_used, ctx) };
            // SAFETY: as for the offsets, for the row's `n_used` u32 places.
            let places = unsafe { WindowMut::from_raw(tables.dev_at(at.places), n_used, ctx) };
            // SAFETY: the image's activation, `hidden` f32 at word `h.x`,
            // lies inside the boundary's page (its layout places it); the leg
            // sits in the boundary, so the page outlives the window and every
            // graph that captured a launch over it.
            let x = unsafe { WindowMut::from_raw(page.dev_at(image + 4 * h.x), hidden, ctx) };
            // SAFETY: as for the activation, for the image's `n_used` routing
            // weights at word `h.weights`.
            let w = unsafe { WindowMut::from_raw(page.dev_at(image + 4 * h.weights), n_used, ctx) };
            // SAFETY: as for the activation, for the row's sum, `hidden` f32
            // at `hsum`.
            let hsum_w = unsafe { WindowMut::from_raw(page.dev_at(hsum), hidden, ctx) };
            // SAFETY: as for the sum, a second window of the same span: the
            // add reads value `i` and writes value `i` back, one thread a
            // value.
            let sum = unsafe { WindowMut::from_raw(page.dev_at(hsum), hidden, ctx) };
            rows.push(RowWindows {
                at,
                offs,
                places,
                x,
                w,
                hsum: hsum_w,
            });
            sums.push(sum);
        }
        // SAFETY: the arena's `words` words from `dev0` are the tier's pinned
        // allocation the card maps; the leg holds the tier and is made at
        // load and lives as long as the load, so the memory outlives the
        // window and every graph that captured a launch over it.
        let arena = unsafe { WindowMut::from_raw(dev0, words, ctx) };
        let stream = gpu.stream();
        let word = gpu.fault_word();
        let mut leg = ArenaLeg {
            arena,
            rows,
            x_place: DeviceBuffer::from_host(stream, &[0u32])?,
            tables,
            tier: Arc::clone(tier),
            layers,
            scratch: Mutex::new(Scratch {
                act_x: Q8Act::with_k(stream, 1, hidden)?,
                h: DeviceBuffer::zeroed(stream, n_used * ff)?,
                act_h: Q8Act::with_slots(stream, n_used, ff)?,
                down: DeviceBuffer::zeroed(stream, n_used * hidden)?,
                acc: DeviceBuffer::zeroed(stream, hidden)?,
                sums,
            }),
            kquant: QuantSelKernels::load(ctx, word)?,
            q6k: Q6kSelKernels::load(ctx, word)?,
            q4k: Q4kSelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
            elem: ElemKernels::load(ctx, word)?,
            hidden,
            ff,
            n_used,
        };
        for r in 0..leg.rows.len() {
            leg.skip_row(r)?;
        }
        Ok(leg)
    }

    pub(crate) fn device_bytes(&self) -> usize {
        let s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        self.x_place.num_bytes()
            + s.act_x.device_bytes()
            + s.h.num_bytes()
            + s.act_h.device_bytes()
            + s.down.num_bytes()
            + s.acc.num_bytes()
    }

    /// The tier whose arena the leg reads.
    pub(crate) fn tier(&self) -> &Arc<NvTier> {
        &self.tier
    }

    /// Whether the leg computes layer `layer`'s arena ids.
    pub(crate) fn runs(&self, layer: usize) -> bool {
        self.layers.get(layer).is_some_and(Option::is_some)
    }

    fn row(&self, row: usize) -> Result<&RowWindows, GpuError> {
        self.rows
            .get(row)
            .ok_or_else(|| GpuError::shape(WHAT, format!("row {row} of {} rows", self.rows.len())))
    }

    /// Every slot of row `row` skipped: no launch reads an arena part for it.
    pub(crate) fn skip_row(&mut self, row: usize) -> Result<(), GpuError> {
        let at = self.row(row)?.at;
        for s in 0..self.n_used {
            self.write_slot(at, s, None);
        }
        Ok(())
    }

    /// Slot `slot` of row `row` read from the arena at `offs` (word offsets
    /// of its gate, up and down parts), or skipped when `None`.
    pub(crate) fn set_slot(
        &mut self,
        row: usize,
        slot: usize,
        offs: Option<[u64; ARENA_PARTS]>,
    ) -> Result<(), GpuError> {
        let at = self.row(row)?.at;
        if slot >= self.n_used {
            return Err(GpuError::shape(
                WHAT,
                format!("slot {slot} of {} slots", self.n_used),
            ));
        }
        self.write_slot(at, slot, offs);
        Ok(())
    }

    fn write_slot(&mut self, at: TableAt, s: usize, offs: Option<[u64; ARENA_PARTS]>) {
        let (words, place) = match offs {
            Some(o) => (o, 0u32),
            None => ([ARENA_SKIP; ARENA_PARTS], HOST),
        };
        for (p, w) in words.iter().enumerate() {
            let o = self
                .tables
                .host_at(at.offs + 8 * (ARENA_PARTS * s + p))
                .cast::<u64>();
            // SAFETY: the row's offsets lie inside `tables` (laid out by
            // `build`) and `s < n_used`, so this u64 does, 8-aligned at a
            // 256-byte row offset; `&mut self` makes this the only host
            // access, and the card reads a row's tables only after the
            // service's signal publishes them, until the row's next go,
            // before which no service writes them.
            unsafe { o.write_volatile(*w) };
        }
        let o = self.tables.host_at(at.places + 4 * s).cast::<u32>();
        // SAFETY: as for the offsets, for the slot's u32 place.
        unsafe { o.write_volatile(place) };
    }

    /// Enqueue on `stream` the leg of row `row` at layer `layer` (module
    /// doc): the activation's q8_1, the gate·up arena entry, the arena
    /// slots' q8_1, the down arena entry, their weighted sum, and the sum
    /// added to the row's host sum. Nothing for a layer the leg does not
    /// run. Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue(
        &self,
        stream: &CudaStream,
        row: usize,
        layer: usize,
    ) -> Result<(), GpuError> {
        let Some(Some(l)) = self.layers.get(layer) else {
            return Ok(());
        };
        let r = self.row(row)?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let Scratch {
            act_x,
            h,
            act_h,
            down,
            acc,
            sums,
        } = &mut *s;
        let x = QuantSel {
            x: &r.x,
            cols: 0..1,
            sel: &self.x_place,
            n_card: 1,
        };
        self.q4k.enqueue_quantize_sel(stream, &x, l.fault, act_x)?;
        let a = GateUpAt {
            arena: &self.arena,
            offs: &r.offs,
            act: act_x,
            n_slots: self.n_used,
            rows_per_expert: self.ff,
            slots_per_col: self.n_used,
            rule: l.rule,
        };
        match l.gate_up {
            GateUpAtEntry::Q4k => self.kquant.enqueue_gate_up_q4k_at(stream, &a, l.fault, h)?,
            GateUpAtEntry::Q5k => self.kquant.enqueue_gate_up_q5k_at(stream, &a, l.fault, h)?,
        }
        let q = QuantSel {
            x: h,
            cols: 0..self.n_used,
            sel: &r.places,
            n_card: 1,
        };
        self.q4k.enqueue_quantize_sel(stream, &q, l.fault, act_h)?;
        let d = SelDownAt {
            arena: &self.arena,
            offs: &r.offs,
            act: act_h,
            n_slots: self.n_used,
            rows_per_expert: self.hidden,
        };
        match l.down {
            DownAtEntry::Q5k => self
                .kquant
                .enqueue_gemv_q5k_sel_at(stream, &d, l.fault, down)?,
            DownAtEntry::Q6k => self
                .q6k
                .enqueue_gemv_q6k_sel_at(stream, &d, l.fault, down)?,
        }
        let c = CardAcc {
            down,
            w: &r.w,
            sel: &r.places,
            n: self.hidden,
            m: 1,
            n_card: 1,
            n_used: self.n_used,
            w_pitch: self.n_used,
            fault: l.fault,
        };
        self.slots.enqueue_card_acc(stream, &c, acc)?;
        self.elem
            .enqueue_add(stream, &r.hsum, acc, self.hidden, &mut sums[row])
    }
}

#[cfg(test)]
mod tests {
    use super::rule_of;
    use crate::kquant::Act;
    use model::ops::ExpertForm;

    /// A gated layer runs its own rule on the card leg, and a relu² layer
    /// none: the leg's gate·up kernels would read its up as a gate.
    /// FAIL-first: a relu² layer took the clamped SwiGLU.
    #[test]
    fn a_relu2_layer_has_no_card_leg_rule() {
        assert_eq!(
            rule_of(ExpertForm::Gated { limit: 7.0 }),
            Some(Act::SwigluClamp { limit: 7.0 })
        );
        assert_eq!(
            rule_of(ExpertForm::GatedOai {
                alpha: 1.702,
                limit: 7.0
            }),
            Some(Act::SwigluOai {
                alpha: 1.702,
                limit: 7.0
            })
        );
        assert_eq!(rule_of(ExpertForm::Relu2), None);
    }
}
