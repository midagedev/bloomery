//! Qwen3.8's card leg: the routed experts of a layer the slot map puts on
//! the card, computed in the host leg's shadow ([`Card38`]), and the map
//! check every walk runs at its entry ([`MapCheck`]).
//!
//! The captured step, the verify and the eager pass run the card leg: on a
//! layer with card experts the shadow quantizes the unit's normed rows to
//! q8_1, runs the Q4_K gate·up `_sel` with SiLU·mul over the unit's ten
//! slots a column, the 32-value q8_1 of the card slots' columns, the Q5_1
//! down `_sel` and the card slots' weighted sum (`q38_card_acc`) —
//! [`CARD_LAUNCHES`] — and the back combines `(hsum + acc) + sh·w`
//! (`q38_card_shared_add`). The ubatch walk runs the card route
//! (`wide38`'s, over the same stacks through the grouped GEMMs) on a layer
//! with card experts and the same back. A layer without card experts
//! launches none of them and keeps `q38_shared_add`. The slot places each
//! launch reads are the ones the walk's handoff or places launch wrote
//! beside the ids from the slot map's card copy, and the host tier skips
//! the same slots by the same map, so which expert runs where has one
//! owner.
//!
//! Each layer's card count is the map's ([`SlotMap::on_card`]), read once at
//! load, where each of the layer's three routed stacks is held to that
//! count's rows ([`Card38::new`]): a stack of other rows would leave slots
//! that neither side sums.
//!
//! No walk has a tier leg: a map with an expert on the tier card is refused
//! by name at every walk's entry — the card copy marks those experts
//! [`crate::hybrid::HOST`] and the host skips them, so a walk that ran
//! would leave them out of the sum. The check's answer is read when the map
//! is set — at the load, when a gate plants one and at reset — so a walk
//! reads one field.

use super::plan38::geo;
use super::program38::Ctx38;
use super::scratch38::PASS_ROWS;
use crate::hybrid::SlotMap;
use crate::kquant::{Act, GateUpAct, KquantKernels};
use crate::q4k_sel::QuantSel;
use crate::q5::Q8Blocks32;
use crate::q5_1_sel::{Q51SelDown, Q51SelKernels};
use crate::q38::{CardAccArgs, SLOTS};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, Q8Act};
use cuda_core::DeviceBuffer;
use gguf::GgmlType;

/// What the map check's refusals name.
const WHAT: &str = "qwen4exp card leg";

/// The card leg's launches on a layer with card experts: the normed rows'
/// q8_1, the gate·up, the q8_1 of the card slots' columns, the down and the
/// card sum.
pub(super) const CARD_LAUNCHES: usize = 5;

const _: () = assert!(SLOTS == geo::N_USED);

/// A walk, as the map check names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Walk38 {
    Step,
    Verify,
    Pass,
    Ubatch,
}

impl Walk38 {
    fn name(self) -> &'static str {
        match self {
            Walk38::Step => "step",
            Walk38::Verify => "verify",
            Walk38::Pass => "pass",
            Walk38::Ubatch => "ubatch",
        }
    }
}

/// A layer of a slot map with routed experts on one device, and how many.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct First {
    layer: usize,
    on: usize,
}

/// A slot map's answer to the walks' check: its first layer with routed
/// experts on the tier card.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MapCheck {
    tier: Option<First>,
}

impl MapCheck {
    /// `map`'s answer. Reads one count a layer.
    pub(super) fn of(map: &SlotMap) -> Result<MapCheck, GpuError> {
        let mut out = MapCheck::default();
        for l in map.layers() {
            let tier = map.on_tier(l)?;
            if tier > 0 && out.tier.is_none() {
                out.tier = Some(First { layer: l, on: tier });
            }
        }
        Ok(out)
    }

    /// Refused by name, before anything moves: a map with an expert on the
    /// tier card at any walk; the error names the walk, the layer and its
    /// count. Allocates nothing unless it refuses.
    pub(super) fn refuse(self, walk: Walk38) -> Result<(), GpuError> {
        let Some(first) = self.tier else {
            return Ok(());
        };
        Err(GpuError::shape(
            WHAT,
            format!(
                "the {} walk has no tier leg: layer {} holds {} routed experts on the tier card",
                walk.name(),
                first.layer,
                first.on
            ),
        ))
    }
}

/// A layer's card experts: how many each of its routed stacks holds, and
/// the stacks' names.
struct CardLayer {
    n_card: usize,
    gate: String,
    up: String,
    down: String,
}

/// A layer's card experts as the ubatch route reads them: their count and
/// the three resident stacks ([`Card38::stacks`]).
pub(super) struct Stacks38<'w> {
    pub(super) n_card: usize,
    pub(super) gate: &'w DeviceTensor<u32>,
    pub(super) up: &'w DeviceTensor<u32>,
    pub(super) down: &'w DeviceTensor<u32>,
}

/// The leg's modules and the buffers every walk's card layers write, for up
/// to [`PASS_ROWS`] columns.
struct CardRun {
    kq: KquantKernels,
    q51: Q51SelKernels,
    /// The normed rows' q8_1 form, a column a token.
    act_x: Q8Act,
    /// The gate·up's output, [`geo::FF`] values a slot.
    h: DeviceBuffer<f32>,
    /// The card slots' columns in 32-value q8_1, per unit width `m` the
    /// `10·m` columns of `act_h[m − 1]` (the down reads one column a slot).
    act_h: Vec<Q8Blocks32>,
    /// The down's output, [`geo::HIDDEN`] values a slot.
    down: DeviceBuffer<f32>,
    /// The card slots' weighted sum, a row a token.
    acc: DeviceBuffer<f32>,
}

/// Qwen3.8's card leg: each layer's card experts, and, when any layer has
/// some, the modules and buffers the leg runs on (module doc).
pub(super) struct Card38 {
    layers: Vec<Option<CardLayer>>,
    run: Option<CardRun>,
}

/// Layer `l`'s routed gate, up and down names.
fn stack_names(l: usize) -> [String; 3] {
    [
        model::arch::qwen35moe::names::ffn_gate_exps(l),
        model::arch::qwen35moe::names::ffn_up_exps(l),
        model::arch::qwen35moe::names::ffn_down_exps(l),
    ]
}

/// The resident stack `name` of type `ty` holding `rows` rows, else refused
/// by name.
fn stack<'w>(
    w: &'w Weights,
    name: &str,
    ty: GgmlType,
    rows: usize,
) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant { ty: t, w, .. }) if *t == ty && w.rows() == rows => Ok(w),
        _ => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: "a resident routed stack of the slot map's card experts of its layer: the \
                   gate and up Q4_K, the down Q5_1, each that count's rows",
        }),
    }
}

impl Card38 {
    /// The card experts of `layers` layers as `map` places them and `w`
    /// holds them. A layer the map puts experts on holds its three routed
    /// stacks as [`stack`] names them, at its count's rows; a layer it puts
    /// none on holds none. Anything else is refused by name. The modules
    /// and buffers are made only when a layer has card experts. Load-time
    /// only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        map: &SlotMap,
        layers: usize,
    ) -> Result<Card38, GpuError> {
        let mut per = Vec::with_capacity(layers);
        for l in 0..layers {
            let n_card = map.on_card(l)?;
            let [gate, up, down] = stack_names(l);
            if n_card == 0 {
                if let Some(n) = [&gate, &up, &down].into_iter().find(|n| w.get(n).is_some()) {
                    return Err(GpuError::Tensor {
                        what: WHAT,
                        name: n.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on \
                               the card",
                    });
                }
                per.push(None);
                continue;
            }
            stack(w, &gate, GgmlType::Q4_K, n_card * geo::FF)?;
            stack(w, &up, GgmlType::Q4_K, n_card * geo::FF)?;
            stack(w, &down, GgmlType::Q5_1, n_card * geo::HIDDEN)?;
            per.push(Some(CardLayer {
                n_card,
                gate,
                up,
                down,
            }));
        }
        let run = if per.iter().any(Option::is_some) {
            let stream = gpu.stream();
            let slots = PASS_ROWS * SLOTS;
            Some(CardRun {
                kq: KquantKernels::load(gpu.context(), gpu.fault_word())?,
                q51: Q51SelKernels::load(gpu.context(), gpu.fault_word())?,
                act_x: Q8Act::with_k(stream, PASS_ROWS, geo::HIDDEN)?,
                h: DeviceBuffer::zeroed(stream, slots * geo::FF)?,
                act_h: (1..=PASS_ROWS)
                    .map(|m| Q8Blocks32::with_slots(stream, geo::FF, m * SLOTS))
                    .collect::<Result<_, _>>()?,
                down: DeviceBuffer::zeroed(stream, slots * geo::HIDDEN)?,
                acc: DeviceBuffer::zeroed(stream, PASS_ROWS * geo::HIDDEN)?,
            })
        } else {
            None
        };
        Ok(Card38 { layers: per, run })
    }

    /// Layer `l` has card experts.
    pub(super) fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    fn layer(&self, l: usize) -> Option<&CardLayer> {
        self.layers.get(l).and_then(Option::as_ref)
    }

    /// Layers with card experts.
    pub(super) fn card_layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// The distinct card counts of the layers with card experts, ascending:
    /// what the ubatch route builds its route tables over. The placement's
    /// spread keeps every eligible layer's count within one, so a plan of
    /// this program holds at most two.
    pub(super) fn card_counts(&self) -> Vec<usize> {
        let mut counts: Vec<usize> = self.layers.iter().flatten().map(|c| c.n_card).collect();
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    /// Layer `l`'s card experts for the ubatch route (`wide38`'s): their
    /// count, the three routed stacks the grouped GEMMs read (the gate and
    /// up Q4_K at `n_card · FF` rows, the down Q5_1 at `n_card · HIDDEN`),
    /// refused by name as [`Card38::new`] refuses. The leg's own buffers are
    /// not touched — the route owns its own.
    pub(super) fn stacks<'w>(&self, w: &'w Weights, l: usize) -> Result<Stacks38<'w>, GpuError> {
        let cl = self.layer(l).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("a card route on layer {l}, which has no card experts"),
            )
        })?;
        Ok(Stacks38 {
            n_card: cl.n_card,
            gate: stack(w, &cl.gate, GgmlType::Q4_K, cl.n_card * geo::FF)?,
            up: stack(w, &cl.up, GgmlType::Q4_K, cl.n_card * geo::FF)?,
            down: stack(w, &cl.down, GgmlType::Q5_1, cl.n_card * geo::HIDDEN)?,
        })
    }

    /// The card slots' weighted sum the last card layer enqueued wrote, a
    /// row a token; refused by name when no layer has card experts.
    pub(super) fn acc(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.run
            .as_ref()
            .map(|r| &r.acc)
            .ok_or(GpuError::state(WHAT, "a layer with card experts"))
    }

    /// Device bytes of the buffers.
    pub(super) fn bytes(&self) -> usize {
        self.run.as_ref().map_or(0, |r| {
            r.act_x.device_bytes()
                + r.h.num_bytes()
                + r.act_h
                    .iter()
                    .map(|a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
                    .sum::<usize>()
                + r.down.num_bytes()
                + r.acc.num_bytes()
        })
    }

    /// Enqueue layer `l`'s card leg over a unit of `m` columns (module doc):
    /// `x` the normed rows (`[m][HIDDEN]`), `sel` the slots' places (ten a
    /// column), `weights` the router's (eleven a column: the routed slots,
    /// then the shared expert's gate), into the card sum ([`Card38::acc`]).
    /// [`CARD_LAUNCHES`] launches. A layer without card experts, and `m`
    /// outside `1..=PASS_ROWS`, are refused by name. Asynchronous,
    /// allocation-free, capturable.
    pub(super) fn enqueue(
        &mut self,
        c: &Ctx38<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        m: usize,
        sel: &DeviceBuffer<u32>,
        weights: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (Some(cl), Some(r)) = (
            self.layers.get(l).and_then(Option::as_ref),
            self.run.as_mut(),
        ) else {
            return Err(GpuError::shape(
                WHAT,
                format!("a card leg on layer {l}, which has no card experts"),
            ));
        };
        let act_h = m
            .checked_sub(1)
            .and_then(|i| r.act_h.get_mut(i))
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("a unit of {m} columns; the leg takes 1..={PASS_ROWS}"),
                )
            })?;
        let (gpu, w) = (c.gpu, c.w);
        let stream = gpu.stream();
        let sink = gpu.layer_sink(l)?;
        let slots = m * SLOTS;
        let n_card = cl.n_card;
        gpu.enqueue_quantize_q8_1_cols(x, &mut r.act_x, m, l)?;
        r.kq.enqueue_gate_up_q4k(
            stream,
            &GateUpAct {
                wg: stack(w, &cl.gate, GgmlType::Q4_K, n_card * geo::FF)?,
                wu: stack(w, &cl.up, GgmlType::Q4_K, n_card * geo::FF)?,
                act: &r.act_x,
                sel,
                n_slots: slots,
                rows_per_expert: geo::FF,
                slots_per_col: SLOTS,
                rule: Act::SiluMul,
            },
            sink,
            &mut r.h,
        )?;
        gpu.q5().enqueue_quantize_q8_sel(
            stream,
            &QuantSel {
                x: &r.h,
                cols: 0..slots,
                sel,
                n_card,
            },
            act_h,
            sink,
        )?;
        r.q51.enqueue_gemv_q5_1_sel(
            stream,
            &Q51SelDown {
                w: stack(w, &cl.down, GgmlType::Q5_1, n_card * geo::HIDDEN)?,
                act: act_h,
                sel,
                n_slots: slots,
                rows_per_expert: geo::HIDDEN,
            },
            sink,
            &mut r.down,
        )?;
        c.k.q38.enqueue_card_acc(
            stream,
            CardAccArgs {
                down: &r.down,
                w: weights,
                sel,
                n: geo::HIDDEN,
                m,
                n_card,
                fault: sink,
                acc: &mut r.acc,
            },
        )
    }
}
