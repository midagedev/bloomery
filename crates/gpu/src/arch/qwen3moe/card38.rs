//! Qwen3.8's card leg: the routed experts of a layer the slot map puts on
//! the card, computed in the host leg's shadow ([`Card38`]), and the map
//! check every walk runs at its entry ([`MapCheck`]).
//!
//! The captured step, the verify and the eager pass run the card leg: on a
//! layer with card experts the shadow quantizes the unit's normed rows to
//! q8_1, runs the gate·up `_sel` of the layer's type (Q4_K, Q5_K, IQ3_XXS
//! or IQ4_XS in the file's blocks, NVFP4 in its planes) with
//! SiLU·mul over the unit's ten slots a column, the 32-value q8_1 of the
//! card slots' columns, the down `_sel` of its type (Q5_1, Q8_0 or IQ4_NL,
//! the file's blocks, or NVFP4's planes) — the four launches of the common card leg
//! (`crate::card_leg`) — and the card slots' weighted sum (`ds41_ffn_card_acc_10`,
//! this family's own: ten slots at a weight pitch of eleven, and a place in
//! `[n_card, HOST)` is a fault the leg's sum does not raise) —
//! [`CARD_LAUNCHES`] — and the back combines `(hsum + acc) + sh·w`
//! (`q38_card_shared_add`). The ubatch walk runs the card route
//! (`wide38`'s, over the same stacks through the grouped GEMMs) on a layer
//! with card experts and the same back. On a layer without a host leg
//! ([`Card38::hostless`]) the step, the verify, the pass and the ubatch
//! walk retain the down rows, and their sum and the combine are one launch
//! (`card_sum_join`). A layer without card experts
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
//! ([`SelRun`]: the common card leg, `crate::card_leg`).
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
use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, RouteW, StackW};
use crate::card_slots::{CardAcc, CardJoin, DownRows, SharedJoin};
use crate::hybrid::SlotMap;
use crate::kquant::Act;
use crate::q38::{CardTierSharedAddArgs, SLOTS, W_PITCH};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::GgmlType;
use model::arch::names::routed_stacks;
use model::placement::accepts::{self, Program};
use model::placement::kernels::DownAct;

/// What the map check's refusals name.
const WHAT: &str = "qwen4exp card leg";

/// The card leg's launches on a layer with card experts: the normed rows'
/// q8_1, the gate·up, the q8_1 of the card slots' columns, the down and the
/// card sum.
pub(super) const CARD_LAUNCHES: usize = 5;
/// The card leg's launches on a hostless or expert-tier layer: those
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

    /// The layer's three routed stacks on this card, from `w`, as the common
    /// card leg reads them ([`LegStacks::of`]): the leg's launches, and the
    /// forms the expert stream's ring holds a half of host experts in for
    /// the ubatch route's ring pass (`crate::card_leg::with_ring`).
    pub(super) fn leg_stacks<'w>(&'w self, w: &'w Weights) -> Result<LegStacks<'w>, GpuError> {
        LegStacks::of(w, [&self.gate, &self.up, &self.down])
    }
}

/// A layer's card experts as the ubatch route reads them: their count, the
/// three resident stacks ([`Card38::stacks`]) and their types.
pub(super) struct Stacks38<'w> {
    pub(super) n_card: usize,
    pub(super) gate: StackW<'w>,
    pub(super) up: StackW<'w>,
    pub(super) down: StackW<'w>,
    /// The gate's and the up's type: Q4_K, Q5_K, IQ3_XXS, IQ4_XS or NVFP4.
    pub(super) gate_up_ty: GgmlType,
    /// The down's type: Q5_1, Q8_0, IQ4_NL or NVFP4.
    pub(super) down_ty: GgmlType,
}

/// The card leg ([`CardLeg`]) over a unit of `m` columns of a card's own
/// experts of a layer, with its modules and buffers for up to `cols` columns:
/// the stage card's card leg and the tier card's layer ([`super::tier38`])
/// both run it. The down's activation is the 32-value q8_1 form every type
/// Qwen3.8's downs have reads.
pub(super) struct SelRun {
    leg: CardLeg,
    bufs: LegBufs,
}

impl SelRun {
    /// The modules on `gpu` and the buffers for units of up to `cols`
    /// columns. Load-time only.
    pub(super) fn new(gpu: &Gpu, cols: usize) -> Result<SelRun, GpuError> {
        let shape = LegShape {
            cap: cols,
            slots_per_col: SLOTS,
            n: geo::HIDDEN,
            ff: geo::FF,
        };
        Ok(SelRun {
            leg: CardLeg::load(gpu)?,
            bufs: LegBufs::for_down(gpu.stream(), shape, DownAct::Q8Blocks32)?,
        })
    }

    /// Device bytes of the buffers.
    pub(super) fn bytes(&self) -> usize {
        self.bufs.bytes()
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
        let stacks = cl.leg_stacks(w)?;
        stacks.check(cl.n_card, self.bufs.shape())?;
        let run = LegRun {
            layer: l,
            stacks: &stacks,
            normed: x,
            m,
            sel,
            n_card: cl.n_card,
            rule: Act::SiluMul,
        };
        self.leg.enqueue_rows(gpu, &mut self.bufs, &run, down)
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
    /// Per layer, whether the step and the verify run it without a host leg
    /// ([`Card38::hostless`]).
    hostless: Vec<bool>,
    /// [`Card38::blocks32_gate_up`].
    blocks32_gate_up: bool,
    run: Option<CardRun>,
}

/// The resident stack `name` of type `ty` holding `rows` rows, else refused
/// by name.
fn stack<'w>(
    w: &'w Weights,
    name: &str,
    ty: GgmlType,
    rows: usize,
) -> Result<StackW<'w>, GpuError> {
    let held = match w.get(name) {
        Some(DevWeight::KQuant { ty: t, w, .. }) if *t == ty => Some(StackW::Words(w)),
        Some(DevWeight::Nvfp4Stack { .. }) if ty == GgmlType::NVFP4 => {
            Some(StackW::Nvfp4(w.nvfp4_stack(WHAT, name)?))
        }
        _ => None,
    };
    match held {
        Some(st) if st.rows() == rows => Ok(st),
        _ => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: "a resident routed stack of the slot map's experts of its layer on this card, \
                   of the type the plan lists for it (the gate and up Q4_K, Q5_K, IQ3_XXS or \
                   IQ4_XS, the down Q5_1, Q8_0 or IQ4_NL, in the file's blocks; NVFP4 in its \
                   planes), each that count's rows",
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
        let [gate, up, down] = routed_stacks(l);
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

/// Layer `l`'s gate·up and down types as the card holds the stacks `stacks`
/// lists, refused by name unless the card leg runs them ([`accepts::leg_runs`]
/// of Qwen3.8's program: the table checked every layer the plan cards when it
/// was built). A checkpoint folder's NVFP4 rows load as NVFP4's planes.
pub(super) fn leg_types(stacks: &Qwen38Stacks, l: usize) -> Result<(GgmlType, GgmlType), GpuError> {
    let held = |ty| match ty {
        GgmlType::NVFP4_ROWS => GgmlType::NVFP4,
        ty => ty,
    };
    match stacks.pair(l) {
        Some((gate_up, down))
            if accepts::leg_runs(Program::Qwen38Body, [gate_up, gate_up, down]) == Some(true) =>
        {
            Ok((held(gate_up), held(down)))
        }
        other => Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}'s card experts in gate·up and down types {other:?}: the card leg \
                 does not run them (swap38::stacks names its types)"
            ),
        )),
    }
}

/// Whether a load's card gate and up are read by the 32-value GEMM in the
/// ubatch route, from each card layer's `(layer, read by it)`: the route
/// holds the run's rows in one input form, the q8_1 rows of the grouped GEMM
/// or the 32-value rows, and the plan counts a run of the q8_1 form, the
/// larger, unless the native form holds them (`place::RouteForm`). A load
/// whose card layers read both is refused by name, naming a layer of each;
/// one with no card layer reads the q8_1 form.
fn one_gate_up_form(forms: &[(usize, bool)]) -> Result<bool, GpuError> {
    let Some(&(first, blocks32)) = forms.first() else {
        return Ok(false);
    };
    match forms.iter().find(|&&(_, b)| b != blocks32) {
        None => Ok(blocks32),
        Some(&(l, _)) => {
            let [a, b] = if blocks32 { [first, l] } else { [l, first] };
            Err(GpuError::shape(
                WHAT,
                format!(
                    "card gate and up stacks the ubatch route reads by two GEMMs (layer {a} by \
                     the 32-value GEMM, layer {b} by the grouped GEMM): the route holds the \
                     run's rows in one form, the one the plan counts"
                ),
            ))
        }
    }
}

impl Card38 {
    /// The card experts of `layers` layers as `map` places them and `w`
    /// holds them, each layer's types as `stacks` lists them (the plan's). A
    /// layer the map puts experts on holds its three routed stacks as
    /// [`stack`] names them, at its count's rows and in those types, which
    /// the leg runs ([`leg_types`]); a layer it puts none on holds none.
    /// Anything else is refused by name. The modules and buffers are made
    /// only when a layer has card experts. `keep_host`: the load runs every
    /// layer's host leg ([`Card38::hostless`]). Load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        w: &Weights,
        stacks: &Qwen38Stacks,
        map: &SlotMap,
        layers: usize,
        keep_host: bool,
    ) -> Result<Card38, GpuError> {
        let per = leg_layers(w, stacks, |l| map.on_card(l), layers)?;
        let hostless: Vec<bool> = per
            .iter()
            .map(|c| !keep_host && c.as_ref().is_some_and(|c| c.n_card == geo::EXPERTS))
            .collect();
        let mut forms = Vec::new();
        for (l, c) in per.iter().enumerate() {
            if let Some(c) = c {
                let route = c.stacks(w)?.gate.route(c.gate_up_ty);
                forms.push((l, matches!(route, Some(RouteW::Blocks32(_)))));
            }
        }
        let blocks32_gate_up = one_gate_up_form(&forms)?;
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
        Ok(Card38 {
            layers: per,
            hostless,
            blocks32_gate_up,
            run,
        })
    }

    /// Layer `l` has card experts.
    pub(super) fn has(&self, l: usize) -> bool {
        self.layer(l).is_some()
    }

    /// The step and the verify run layer `l` without a host leg: no go, no
    /// wait, the host not told of the row, and a fused card sum and shared
    /// add. The load decides it once ([`Card38::new`]):
    /// every routed expert of the layer is on the card, and `keep_host` is
    /// false, which the load sets when a residency machine runs or the host
    /// service records a route trace. It holds for the load's life: with no
    /// machine the load's slot map is every walk's (the machine is what
    /// moves an expert to the host), and on a load with such a layer the
    /// host tier refuses a machine started later
    /// ([`crate::host::HostTier::start_swap`]) and a route trace, which the
    /// host service writes ([`crate::host::HostTier::attach_route_trace`]).
    /// The prompt walk runs such a layer without one too (no download, no
    /// service, no host sum: its card route writes the block's output), and
    /// the host run holds no view of it (`crate::model::host_run` over the
    /// other layers), so its stacks may be of a type no host kernel reads.
    /// `l` must be a layer of the load.
    pub(super) fn hostless(&self, l: usize) -> bool {
        assert!(
            l < self.hostless.len(),
            "Card38::hostless: layer {l} of a load of {} layers",
            self.hostless.len()
        );
        self.hostless[l]
    }

    /// Every layer of the load runs without a host leg ([`Card38::hostless`]).
    pub(super) fn all_hostless(&self) -> bool {
        self.hostless.iter().all(|&h| h)
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

    /// The load's card gate and up are read by the 32-value GEMM in the
    /// ubatch route ([`StackW::route`]), over the 32-value activation; else
    /// by the grouped GEMM over the q8_1 rows ([`one_gate_up_form`]).
    pub(super) fn blocks32_gate_up(&self) -> bool {
        self.blocks32_gate_up
    }

    pub(super) fn routed_types(&self) -> impl Iterator<Item = (GgmlType, GgmlType)> + '_ {
        self.layers
            .iter()
            .flatten()
            .map(|layer| (layer.gate_up_ty, layer.down_ty))
    }

    /// Layer `l`'s card stacks as the common card leg reads them
    /// ([`LegLayer::leg_stacks`]), refused by name on a layer with no card
    /// experts.
    pub(super) fn leg_stacks<'w>(
        &'w self,
        w: &'w Weights,
        l: usize,
    ) -> Result<LegStacks<'w>, GpuError> {
        self.card_layer(l)?.leg_stacks(w)
    }

    /// Layer `l`'s card experts for the ubatch route (`wide38`'s): their
    /// count, the three routed stacks the grouped GEMMs read (the gate and
    /// up at `n_card · FF` rows, the down at `n_card · HIDDEN`) and their
    /// types, refused by name as [`Card38::new`] refuses. The leg's own buffers are
    /// not touched — the route owns its own.
    pub(super) fn stacks<'w>(&self, w: &'w Weights, l: usize) -> Result<Stacks38<'w>, GpuError> {
        self.card_layer(l)?.stacks(w)
    }

    /// Layer `l`'s card experts, refused by name on a layer with none.
    fn card_layer(&self, l: usize) -> Result<&LegLayer, GpuError> {
        self.layer(l).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("a card route on layer {l}, which has no card experts"),
            )
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
        let a = CardAcc {
            down: &r.down,
            w: weights,
            sel,
            n: geo::HIDDEN,
            m,
            n_card: cl.n_card,
            n_used: SLOTS,
            w_pitch: W_PITCH,
            fault: c.gpu.layer_sink(l)?,
        };
        c.k.slots.enqueue_card_acc(c.gpu.stream(), &a, &mut r.acc)
    }

    /// [`Card38::enqueue`] but the card sum: the card slots' down outputs
    /// stay in the leg's rows for a tier or hostless layer's join
    /// ([`Card38::enqueue_tier_join`], [`Card38::enqueue_hostless_join`]).
    /// [`TIER_CARD_LAUNCHES`] launches.
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

    /// Sum a hostless layer's retained down rows and add its gated shared
    /// expert in one launch, with the same slot order as the card sum.
    pub(super) fn enqueue_hostless_join(
        &self,
        c: &Ctx38<'_>,
        (l, m): (usize, usize),
        (sel, sh, weights): (&DeviceBuffer<u32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (Some(cl), Some(r)) = (self.layer(l), self.run.as_ref()) else {
            return Err(no_card(l));
        };
        let a = CardJoin {
            down: DownRows::F32(&r.down),
            w: weights,
            sel,
            n: geo::HIDDEN,
            m,
            n_card: cl.n_card,
            n_used: SLOTS,
            w_pitch: W_PITCH,
            shared: SharedJoin {
                sh,
                gate: Some(SLOTS),
            },
            fault: c.gpu.layer_sink(l)?,
        };
        c.k.slots.enqueue_card_join(c.gpu.stream(), &a, y)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A load reads its card gate and up by one GEMM: none, all grouped and
    /// all 32-value pass as such, and a mix is refused naming a layer of
    /// each. Mutants: a mix taken as either form, or the layers swapped in
    /// the refusal.
    #[test]
    fn a_load_reads_its_card_gate_up_in_one_form() {
        assert!(!one_gate_up_form(&[]).expect("no card layer"));
        assert!(!one_gate_up_form(&[(0, false), (3, false)]).expect("all grouped"));
        assert!(one_gate_up_form(&[(1, true), (2, true)]).expect("all 32-value"));
        for forms in [[(4, true), (7, false)], [(7, false), (4, true)]] {
            let err = one_gate_up_form(&forms).expect_err("a mix").to_string();
            assert!(
                err.contains("layer 4 by the 32-value GEMM, layer 7 by the grouped GEMM"),
                "{err}"
            );
        }
    }
}
