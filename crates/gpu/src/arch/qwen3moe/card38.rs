//! Qwen3.8's card leg: the routed experts of a layer the slot map puts on
//! the card, computed in the host leg's shadow ([`Card38`]), and the map
//! check every walk runs at its entry ([`MapCheck`]).
//!
//! The captured step, the verify and the eager pass run the card leg: on a
//! layer with card experts the shadow quantizes the unit's normed rows to
//! q8_1, runs the gate·up `_sel` of the layer's type (Q4_K or Q5_K) with
//! SiLU·mul over the unit's ten slots a column, the 32-value q8_1 of the
//! card slots' columns, the down `_sel` of its type (Q5_1 or Q8_0, the
//! file's blocks) and the card slots' weighted sum (`q38_card_acc`) —
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
//! count's rows and to the type the plan lists for it ([`Qwen38Stacks`], the
//! residency's source of the same types) ([`Card38::new`]): a stack of other
//! rows would leave slots that neither side sums, and one of another type
//! would be read in a layout it is not in.
//!
//! On a layer an expert tier holds experts of (a two-card load, `tier38`)
//! the step's and the verify's card leg runs the same launches but the card
//! sum ([`TIER_CARD_LAUNCHES`]): the tier's rows land only with the wait, so
//! the back sums the card's and the tier's slots in slot order and combines
//! in one launch (`q38_card_tier_shared_add`, [`Card38::enqueue_tier_join`]).
//! The tier card runs the same four launches over its own stacks
//! ([`SelRun`], the one owner of that sequence).
//!
//! The ubatch walk's card route on such a layer keeps its card slots' down
//! rows for the whole unit and leaves the sum to the join after the host's
//! wait for the tier (`wide38`, `q38_card_tier_acc`).
//!
//! The pass has no tier leg, and no walk has one on a load that hung no
//! tier: a map with an expert on the tier card there is refused by name at
//! the walk's entry — the card copy marks those experts
//! [`crate::hybrid::HOST`] and the host skips them, so a walk that ran would
//! leave them out of the sum. The check's answer is read
//! when the map is set — at the load, when a gate plants one and at reset —
//! so a walk reads one field.

use super::plan38::geo;
use super::program38::Ctx38;
use super::scratch38::PASS_ROWS;
use super::swap38::Qwen38Stacks;
use crate::hybrid::SlotMap;
use crate::kquant::{Act, GateUpAct, KquantKernels};
use crate::q4k_sel::QuantSel;
use crate::q5::Q8Blocks32;
use crate::q5_1_sel::{Q51SelDown, Q51SelKernels};
use crate::q8_0_sel32::{Q80SelDown, Q80SelKernels};
use crate::q38::{CardAccArgs, CardTierSharedAddArgs, SLOTS};
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
/// The card leg's launches on a layer an expert tier holds experts of: those
/// of [`CARD_LAUNCHES`] but the card sum, which moves into the join.
pub(super) const TIER_CARD_LAUNCHES: usize = CARD_LAUNCHES - 1;

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
/// experts on the tier card, and whether the load hung a tier card whose
/// leg the step and the verify run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MapCheck {
    tier: Option<First>,
    legs: bool,
}

impl MapCheck {
    /// `map`'s answer on a load that hung a tier card when `legs`. Reads one
    /// count a layer.
    pub(super) fn of(map: &SlotMap, legs: bool) -> Result<MapCheck, GpuError> {
        let mut out = MapCheck { tier: None, legs };
        for l in map.layers() {
            let tier = map.on_tier(l)?;
            if tier > 0 && out.tier.is_none() {
                out.tier = Some(First { layer: l, on: tier });
            }
        }
        Ok(out)
    }

    /// Refused by name, before anything moves: a map with an expert on the
    /// tier card at the pass, and at every walk on a load that hung no tier;
    /// the error names the walk, the layer and its count, and on a tier load
    /// the way its prompt runs. Allocates nothing unless it refuses.
    pub(super) fn refuse(self, walk: Walk38) -> Result<(), GpuError> {
        let Some(first) = self.tier else {
            return Ok(());
        };
        if walk != Walk38::Pass && self.legs {
            return Ok(());
        }
        let way = if self.legs {
            "; a load with an expert tier feeds a prompt by ubatches (Prompt38::Gemm, which \
             --prefill auto resolves to at every length there) or by steps"
        } else {
            ""
        };
        Err(GpuError::shape(
            WHAT,
            format!(
                "the {} walk has no tier leg: layer {} holds {} routed experts on the tier \
                 card{way}",
                walk.name(),
                first.layer,
                first.on
            ),
        ))
    }
}

/// A layer's experts on one card, the stage's or a tier's: how many each of
/// its routed stacks holds, the stacks' names, and the gate·up's and the
/// down's types.
pub(super) struct LegLayer {
    n_card: usize,
    gate: String,
    up: String,
    down: String,
    gate_up_ty: GgmlType,
    down_ty: GgmlType,
}

impl LegLayer {
    /// The experts' count.
    pub(super) fn n(&self) -> usize {
        self.n_card
    }

    /// The layer's experts on this card as a grouped route reads them, from
    /// `w`, that card's weights ([`Stacks38`]): their count, the three routed
    /// stacks at their rows and their types, refused by name as
    /// [`leg_layers`] refuses.
    pub(super) fn stacks<'w>(&self, w: &'w Weights) -> Result<Stacks38<'w>, GpuError> {
        Ok(Stacks38 {
            n_card: self.n_card,
            gate: stack(w, &self.gate, self.gate_up_ty, self.n_card * geo::FF)?,
            up: stack(w, &self.up, self.gate_up_ty, self.n_card * geo::FF)?,
            down: stack(w, &self.down, self.down_ty, self.n_card * geo::HIDDEN)?,
            gate_up_ty: self.gate_up_ty,
            down_ty: self.down_ty,
        })
    }
}

/// A layer's card experts as the ubatch route reads them: their count, the
/// three resident stacks ([`Card38::stacks`]) and their types.
pub(super) struct Stacks38<'w> {
    pub(super) n_card: usize,
    pub(super) gate: &'w DeviceTensor<u32>,
    pub(super) up: &'w DeviceTensor<u32>,
    pub(super) down: &'w DeviceTensor<u32>,
    /// The gate's and the up's type: Q4_K or Q5_K.
    pub(super) gate_up_ty: GgmlType,
    /// The down's type: Q5_1 or Q8_0.
    pub(super) down_ty: GgmlType,
}

/// The four `_sel` launches a card runs over a unit of `m` columns of its
/// own experts of a layer — the normed rows' q8_1, the gate·up with
/// SiLU·mul, the 32-value q8_1 of its slots' columns, the down — and their
/// modules and buffers, for up to `cols` columns: the stage card's card leg
/// and the tier card's layer ([`super::tier38`]) both run it.
pub(super) struct SelRun {
    kq: KquantKernels,
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    /// The normed rows' q8_1 form, a column a token.
    act_x: Q8Act,
    /// The gate·up's output, [`geo::FF`] values a slot.
    h: DeviceBuffer<f32>,
    /// The card slots' columns in 32-value q8_1, per unit width `m` the
    /// `10·m` columns of `act_h[m − 1]` (the down reads one column a slot).
    act_h: Vec<Q8Blocks32>,
}

impl SelRun {
    /// The modules on `gpu` and the buffers for units of up to `cols`
    /// columns. Load-time only.
    pub(super) fn new(gpu: &Gpu, cols: usize) -> Result<SelRun, GpuError> {
        let stream = gpu.stream();
        Ok(SelRun {
            kq: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            q51: Q51SelKernels::load(gpu.context(), gpu.fault_word())?,
            q80: Q80SelKernels::load(gpu.context(), gpu.fault_word())?,
            act_x: Q8Act::with_k(stream, cols, geo::HIDDEN)?,
            h: DeviceBuffer::zeroed(stream, cols * SLOTS * geo::FF)?,
            act_h: (1..=cols)
                .map(|m| Q8Blocks32::with_slots(stream, geo::FF, m * SLOTS))
                .collect::<Result<_, _>>()?,
        })
    }

    /// Device bytes of the buffers.
    pub(super) fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.h.num_bytes()
            + self
                .act_h
                .iter()
                .map(|a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
                .sum::<usize>()
    }

    /// Enqueue on `gpu`'s stream layer `l`'s four launches over `cl`'s
    /// stacks in `w`: `x` the normed rows (`[m][HIDDEN]`), `sel` the slots'
    /// places (ten a column; below `cl`'s count this card's, else
    /// [`crate::hybrid::HOST`] or a fault), the down's outputs into `down`
    /// slot-major ([`geo::HIDDEN`] a slot, this card's slots only). `m`
    /// outside the buffers' `1..=cols` is refused by name. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, its weights, the layer and its stacks, the unit, its places and the output (rust-quality R8)"
    )]
    pub(super) fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, cl): (usize, &LegLayer),
        x: &DeviceBuffer<f32>,
        m: usize,
        sel: &DeviceBuffer<u32>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let cols = self.act_h.len();
        let act_h = m
            .checked_sub(1)
            .and_then(|i| self.act_h.get_mut(i))
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("a unit of {m} columns; the leg takes 1..={cols}"),
                )
            })?;
        let stream = gpu.stream();
        let sink = gpu.layer_sink(l)?;
        let slots = m * SLOTS;
        let n_card = cl.n_card;
        gpu.enqueue_quantize_q8_1_cols(x, &mut self.act_x, m, l)?;
        let gu = GateUpAct {
            wg: stack(w, &cl.gate, cl.gate_up_ty, n_card * geo::FF)?,
            wu: stack(w, &cl.up, cl.gate_up_ty, n_card * geo::FF)?,
            act: &self.act_x,
            sel,
            n_slots: slots,
            rows_per_expert: geo::FF,
            slots_per_col: SLOTS,
            rule: Act::SiluMul,
        };
        if cl.gate_up_ty == GgmlType::Q5_K {
            self.kq
                .enqueue_gate_up_q5k(stream, &gu, sink, &mut self.h)?;
        } else {
            self.kq
                .enqueue_gate_up_q4k(stream, &gu, sink, &mut self.h)?;
        }
        gpu.q5().enqueue_quantize_q8_sel(
            stream,
            &QuantSel {
                x: &self.h,
                cols: 0..slots,
                sel,
                n_card,
            },
            act_h,
            sink,
        )?;
        let wd = stack(w, &cl.down, cl.down_ty, n_card * geo::HIDDEN)?;
        if cl.down_ty == GgmlType::Q8_0 {
            self.q80.enqueue_gemv_q8_0_sel32(
                stream,
                &Q80SelDown {
                    w: wd,
                    act: act_h,
                    sel,
                    n_slots: slots,
                    rows_per_expert: geo::HIDDEN,
                },
                sink,
                down,
            )
        } else {
            self.q51.enqueue_gemv_q5_1_sel(
                stream,
                &Q51SelDown {
                    w: wd,
                    act: act_h,
                    sel,
                    n_slots: slots,
                    rows_per_expert: geo::HIDDEN,
                },
                sink,
                down,
            )
        }
    }
}

/// The leg's launches and the buffers every walk's card layers write, for up
/// to [`PASS_ROWS`] columns.
struct CardRun {
    sel: SelRun,
    /// The down's output, [`geo::HIDDEN`] values a slot.
    down: DeviceBuffer<f32>,
    /// The card slots' weighted sum, a row a token.
    acc: DeviceBuffer<f32>,
}

/// Qwen3.8's card leg: each layer's card experts, and, when any layer has
/// some, the modules and buffers the leg runs on (module doc).
pub(super) struct Card38 {
    layers: Vec<Option<LegLayer>>,
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
            need: "a resident routed stack of the slot map's experts of its layer on this card, \
                   in the file's blocks of the type the plan lists for it (the gate and up Q4_K \
                   or Q5_K, the down Q5_1 or Q8_0), each that count's rows",
        }),
    }
}

/// Per layer of `layers`, its experts on one card as `on(l)` counts them and
/// `w` holds them, each layer's types as `stacks` lists them: a layer with
/// experts there holds its three routed stacks as [`stack`] names them, at
/// its count's rows and in types the leg runs ([`leg_types`]); a layer with
/// none holds none. Anything else is refused by name. Load-time only.
pub(super) fn leg_layers(
    w: &Weights,
    stacks: &Qwen38Stacks,
    on: impl Fn(usize) -> Result<usize, GpuError>,
    layers: usize,
) -> Result<Vec<Option<LegLayer>>, GpuError> {
    let mut per = Vec::with_capacity(layers);
    for l in 0..layers {
        let n_card = on(l)?;
        let [gate, up, down] = stack_names(l);
        if n_card == 0 {
            if let Some(n) = [&gate, &up, &down].into_iter().find(|n| w.get(n).is_some()) {
                return Err(GpuError::Tensor {
                    what: WHAT,
                    name: n.clone(),
                    need: "no resident rows: the slot map puts no expert of its layer on this \
                           card",
                });
            }
            per.push(None);
            continue;
        }
        let (gate_up_ty, down_ty) = leg_types(stacks, l)?;
        stack(w, &gate, gate_up_ty, n_card * geo::FF)?;
        stack(w, &up, gate_up_ty, n_card * geo::FF)?;
        stack(w, &down, down_ty, n_card * geo::HIDDEN)?;
        per.push(Some(LegLayer {
            n_card,
            gate,
            up,
            down,
            gate_up_ty,
            down_ty,
        }));
    }
    Ok(per)
}

/// Layer `l`'s gate·up and down types as `stacks` lists them, refused by
/// name unless the card leg runs them: the gate and up Q4_K or Q5_K, the
/// down Q5_1 or Q8_0.
fn leg_types(stacks: &Qwen38Stacks, l: usize) -> Result<(GgmlType, GgmlType), GpuError> {
    match stacks.pair(l) {
        Some(p @ (GgmlType::Q4_K | GgmlType::Q5_K, GgmlType::Q5_1 | GgmlType::Q8_0)) => Ok(p),
        other => Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}'s card experts in gate·up and down types {other:?}: the card leg runs a \
                 Q4_K or Q5_K gate and up and a Q5_1 or Q8_0 down"
            ),
        )),
    }
}

impl Card38 {
    /// The card experts of `layers` layers as `map` places them and `w`
    /// holds them, each layer's types as `stacks` lists them (the plan's). A
    /// layer the map puts experts on holds its three routed stacks as
    /// [`stack`] names them, at its count's rows and in those types, which
    /// the leg runs ([`leg_types`]); a layer it puts none on holds none.
    /// Anything else is refused by name. The modules and buffers are made
    /// only when a layer has card experts. Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        stacks: &Qwen38Stacks,
        map: &SlotMap,
        layers: usize,
    ) -> Result<Card38, GpuError> {
        let per = leg_layers(w, stacks, |l| map.on_card(l), layers)?;
        let run = if per.iter().any(Option::is_some) {
            let stream = gpu.stream();
            let slots = PASS_ROWS * SLOTS;
            Some(CardRun {
                sel: SelRun::new(gpu, PASS_ROWS)?,
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

    /// Layer `l`'s card experts: 0 on a layer without them.
    pub(super) fn n_card(&self, l: usize) -> usize {
        self.layer(l).map_or(0, |c| c.n_card)
    }

    fn layer(&self, l: usize) -> Option<&LegLayer> {
        self.layers.get(l).and_then(Option::as_ref)
    }

    /// Layers with card experts.
    pub(super) fn card_layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// The card layers' stack types with the layers of each, in layer order
    /// of first use: `<gate·up>/<down>:<layers>` joined by `,`, the layers as
    /// ascending runs (`q4_K/q5_1:0-1,3,5-29,…;q5_K/q8_0:2;…`, runs `,`
    /// and types `;`). `none` when no layer has card experts. Load-time
    /// telemetry: the load line prints it.
    pub(super) fn card_stacks(&self) -> String {
        let mut groups: Vec<((GgmlType, GgmlType), Vec<usize>)> = Vec::new();
        for (l, cl) in self.layers.iter().enumerate() {
            let Some(cl) = cl else { continue };
            let key = (cl.gate_up_ty, cl.down_ty);
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, ls)) => ls.push(l),
                None => groups.push((key, vec![l])),
            }
        }
        if groups.is_empty() {
            return "none".to_string();
        }
        let runs = |ls: &[usize]| {
            let mut out: Vec<String> = Vec::new();
            let mut i = 0;
            while i < ls.len() {
                let mut j = i;
                while j + 1 < ls.len() && ls[j + 1] == ls[j] + 1 {
                    j += 1;
                }
                out.push(if i == j {
                    ls[i].to_string()
                } else {
                    format!("{}-{}", ls[i], ls[j])
                });
                i = j + 1;
            }
            out.join(",")
        };
        groups
            .iter()
            .map(|((gu, d), ls)| format!("{gu}/{d}:{}", runs(ls)))
            .collect::<Vec<_>>()
            .join(";")
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
    /// up at `n_card · FF` rows, the down at `n_card · HIDDEN`) and their
    /// types, refused by name as [`Card38::new`] refuses. The leg's own buffers are
    /// not touched — the route owns its own.
    pub(super) fn stacks<'w>(&self, w: &'w Weights, l: usize) -> Result<Stacks38<'w>, GpuError> {
        self.layer(l)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("a card route on layer {l}, which has no card experts"),
                )
            })?
            .stacks(w)
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
            r.sel.bytes() + r.down.num_bytes() + r.acc.num_bytes()
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
        self.enqueue_rows(c, l, x, m, sel)?;
        let (Some(cl), Some(r)) = (
            self.layers.get(l).and_then(Option::as_ref),
            self.run.as_mut(),
        ) else {
            return Err(no_card(l));
        };
        c.k.q38.enqueue_card_acc(
            c.gpu.stream(),
            CardAccArgs {
                down: &r.down,
                w: weights,
                sel,
                n: geo::HIDDEN,
                m,
                n_card: cl.n_card,
                fault: c.gpu.layer_sink(l)?,
                acc: &mut r.acc,
            },
        )
    }

    /// [`Card38::enqueue`] but the card sum: the card slots' down outputs
    /// stay in the leg's rows for a tier layer's join
    /// ([`Card38::enqueue_tier_join`]). [`TIER_CARD_LAUNCHES`] launches.
    pub(super) fn enqueue_rows(
        &mut self,
        c: &Ctx38<'_>,
        l: usize,
        x: &DeviceBuffer<f32>,
        m: usize,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let (Some(cl), Some(r)) = (
            self.layers.get(l).and_then(Option::as_ref),
            self.run.as_mut(),
        ) else {
            return Err(no_card(l));
        };
        r.sel.enqueue(c.gpu, c.w, (l, cl), x, m, sel, &mut r.down)
    }

    /// Enqueue the join of tier layer `l` over the unit's `m` columns, after
    /// the wait: the card's slots (`sel`, the leg's rows of
    /// [`Card38::enqueue_rows`]) and the tier's (`tsel`, below `n_tier`, the
    /// tier's rows `trows`) summed in slot order by the router's `weights`,
    /// then `(hsum + sum) + sh · w` into `y` (`q38_card_tier_shared_add`).
    /// One launch. A layer without card experts is refused by name.
    #[allow(
        clippy::too_many_arguments,
        reason = "the walk's context, the layer and its width, both cards' places and rows, and the combine's inputs and output (rust-quality R8)"
    )]
    pub(super) fn enqueue_tier_join(
        &self,
        c: &Ctx38<'_>,
        (l, m): (usize, usize),
        (sel, tsel, n_tier): (&DeviceBuffer<u32>, &DeviceBuffer<u32>, usize),
        trows: &DeviceBuffer<f32>,
        (hsum, sh, weights): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (Some(cl), Some(r)) = (self.layer(l), self.run.as_ref()) else {
            return Err(no_card(l));
        };
        c.k.q38.enqueue_card_tier_shared_add(
            c.gpu.stream(),
            CardTierSharedAddArgs {
                down: &r.down,
                trows,
                w: weights,
                sel,
                tsel,
                hsum,
                sh,
                n: geo::HIDDEN,
                m,
                n_card: cl.n_card,
                n_tier,
                fault: c.gpu.layer_sink(l)?,
                y,
            },
        )
    }
}

/// The refusal of a card leg on layer `l`, which has no card experts.
fn no_card(l: usize) -> GpuError {
    GpuError::shape(
        WHAT,
        format!("a card leg on layer {l}, which has no card experts"),
    )
}
