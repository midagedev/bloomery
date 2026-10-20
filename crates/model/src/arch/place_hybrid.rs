//! A hybrid file planned onto a machine: the one path from a family's headers to a plan that keeps
//! its invariants, for every family whose body mixes with GQA layers (a plane, or a ring for a window
//! layer) and delta-rule layers (a state and a conv ring), routes every layer's experts and puts the
//! rest of them on the host. A family reads its headers into a [`ModelSpec`] and its tensors' roles
//! ([`PlanInputs::new`]); what the layers hold is read off the description ([`KvLayout::of`]), the
//! program off its architecture ([`program_of`]), and the placement runs the expert rule on the
//! layers whose stacks the card experts read ([`placement::plan_routed`]). What stays in a family is
//! what an architecture fact forces: the keys it reads, the layers it refuses, the engine's name in
//! the refusals that name it.
//!
//! The plan refuses by name a machine that hangs an expert tier card, a context of no position and
//! a host room that cannot hold the host experts: the bodies this plans serve no NVMe expert tier, so
//! a plan that pages one ([`PlaceError::NoNvmeTier`]) would load experts nothing serves.
//!
//! A body that makes buffers beside the plan's terms on its card (the prompt batch's rows) states
//! their bytes in [`PlanInputs::prompt_reserve`]: the expert rule spreads within the card's budget
//! less them ([`placement::plan_routed_reserving`]) and a card that cannot hold them beside its
//! other terms is refused by name ([`PlaceError::PromptOver`]). The family that makes the buffers
//! owns the formula of their bytes; the body holds its allocation to the same figure.

use std::num::NonZeroUsize;
use std::ops::Range;

use bloomery_placement::slots::{SeqTerms, Stores};
use gguf::GgmlType;
use models::{DeltaRule, Mixer, ModelSpec, RopeMode};
use runtime::stores::{self, Rows};

use crate::arch::coverage;
use crate::placement::accepts::{program_of, routed_rule};
use crate::placement::ctx_ask::{PlanRefusal, Short, Side};
use crate::placement::workstation::{self, HostNeed, HostRead};
use crate::placement::{
    self, Card, CardFormat, CardTotals, KvBytes, Machine, ModelTensors, PlacementError, Plan,
    PlanLevers, Unimplemented, Violation, checked_beside, joined,
};

// The prompt batch a window ring keeps room for is the batch the card walks the prompt in.
const _: () = assert!(stores::PROMPT_BATCH == crate::moe::UNION_MAX_COLS as u64);

/// The routed stacks the card experts read, each as the file stores it ([`CardFormat::KQuant`]):
/// the types the common card leg runs ([`placement::kernels::card_runs`]). The rule reads a stack's
/// type alone; a layer with a stack of any other type keeps its experts on the host.
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    placement::kernels::card_runs(ty).then_some(CardFormat::KQuant)
}

/// What a plan of a hybrid file is made from, read from its headers. `H` is the family's own
/// hyperparameters, carried for its body and never read here.
#[derive(Clone, Debug)]
pub struct PlanInputs<H> {
    /// The file's hyperparameters.
    pub hp: H,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's stores.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the reading that decided
    /// ([`workstation::host_available_read`]): the routed experts the room cannot hold follow it to
    /// the NVMe tier ([`placement::expert_nvme_tier`]), which these bodies do not serve, so such a
    /// plan is refused ([`PlaceError::NoNvmeTier`]). A gate that plans an arm by a room of its own
    /// sets it, read [`HostRead::Given`].
    ///
    /// The plan is made at [`workstation::plan_room`]: a read room less
    /// [`workstation::HOST_DRIFT`], a given room as given.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it as its `given`. A
    /// gate that plans an arm by an arena of its own sets it.
    pub arena: Option<u64>,
    /// The card bytes the load makes beside the plan's terms, its prompt batch's buffers: every
    /// card keeps them out of the expert budget ([`placement::plan_routed_reserving`]), and the
    /// body refuses a load whose made bytes pass them. 0 for a family whose body makes none.
    pub prompt_reserve: u64,
    /// The engine the refusals that name one name.
    engine: &'static str,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// A machine with an expert tier card: this program hangs no expert tier on its card experts.
    #[error(
        "the machine has the expert tier card {card} (tier {tier}); the program this plan is \
         for hangs no expert tier"
    )]
    Tier {
        /// The tier card's name.
        card: String,
        /// Its index among the machine's tiers.
        tier: usize,
    },
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
    /// A card whose plan leaves less beside it than the prompt batch's buffers
    /// ([`PlanInputs::prompt_reserve`]) take: the dense granules, the cache, context, scratch,
    /// reserves and margin already pass what the card holds less them, so no expert placement
    /// makes room.
    #[error(
        "card {card}: the prompt batch's buffers take {reserve} B, and the plan leaves {left} B \
         beside its dense granules, cache, context, scratch, reserves and margin ({short} B short)"
    )]
    PromptOver {
        /// The card's name.
        card: String,
        /// The bytes the load makes beside the plan's terms.
        reserve: u64,
        /// The bytes the card has left beside the plan's terms.
        left: u64,
        /// `reserve - left`.
        short: u64,
    },
    /// The host room cannot hold the host experts, so the plan pages some to the NVMe expert tier.
    /// The body builds its host run without that tier (`HostRun::build`, not the common `host_run`
    /// that attaches it), so the paged experts would load into nothing that serves them.
    #[error(
        "the host need {need} B passes the room, so the plan pages {paged} B of routed experts \
         to the NVMe expert tier{room}, and the {engine} engine serves no NVMe expert tier; free \
         host memory (the room is the host's available bytes, or BLOOMERY_HOST_ROOM where set)"
    )]
    NoNvmeTier {
        /// The routed-expert bytes the plan leaves on the NVMe tier.
        paged: u64,
        /// The host need before any expert is paged ([`HostNeed::bytes`]).
        need: u64,
        /// The room the plan was made at, as the plan names it
        /// ([`placement::HostTotals::room_clause`]).
        room: String,
        /// The engine that serves no such tier.
        engine: &'static str,
    },
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            _ => None,
        }
    }

    /// The card's reading, and the host's: a longer context leaves more routed experts to the
    /// host, and the body serves no NVMe tier past it ([`PlaceError::NoNvmeTier`]), so a smaller
    /// context is what that refusal asks for. The host held its need less the bytes it paged.
    fn short(&self) -> Option<Short> {
        match self {
            Self::NoNvmeTier { paged, need, .. } => Some(Short {
                on: Side::Host,
                need: *need,
                room: need.saturating_sub(*paged),
            }),
            Self::PromptOver { reserve, left, .. } => Some(Short {
                on: Side::Card,
                need: *reserve,
                room: *left,
            }),
            _ => self.card_short(),
        }
    }
}

impl<H> PlanInputs<H> {
    /// The inputs of a file whose `hp`, roles `model` and description `spec` a family read: what
    /// each trunk layer holds ([`KvLayout::of`]), then the host's room
    /// ([`workstation::host_available_read`]) and the arena's given bytes, a room that cannot be
    /// read refused by name. `engine` is the name the refusals that name one use.
    pub fn new(
        engine: &'static str,
        hp: H,
        model: ModelTensors,
        spec: ModelSpec,
    ) -> Result<PlanInputs<H>, PlacementError> {
        let kv = KvLayout::of(spec.layers.iter().map(|l| &l.mixer))?;
        let room = workstation::host_available_read().map_err(PlacementError::HostRoom)?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
            prompt_reserve: 0,
            engine,
        })
    }

    /// `self` with a load that makes `bytes` of card buffers beside the plan's terms
    /// ([`PlanInputs::prompt_reserve`]).
    #[must_use]
    pub fn with_prompt_reserve(self, bytes: u64) -> PlanInputs<H> {
        PlanInputs {
            prompt_reserve: bytes,
            ..self
        }
    }

    /// `self`, or — for a file with a feature the engine does not run —
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<PlanInputs<H>, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file the engine does not run, as [`coverage::check`] lists them. Empty
    /// for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under the placement's
    /// `levers`: the expert rule on the layers whose routed stacks [`card_routed`] runs, each such
    /// layer's id prefix on the card as far as its budget holds them, the rest on the host; refused
    /// by name a context of no position, a machine that hangs an expert tier card, a context whose
    /// stores do not fit the card and a host room under the host experts' need
    /// ([`PlanInputs::plan_with_slots`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of `ctx_max` positions each:
    /// every layer's stores counted `slots` times on the card ([`SeqTerms::slots_of`] over
    /// [`PlanInputs::seq_terms`]), refused as [`PlanInputs::plan`] refuses. A sequence holds nothing
    /// beside its stores, so the card's kv class is `slots` sequences' stores.
    ///
    /// The host experts are held to the room the inputs name ([`PlanInputs::room`]): a room under
    /// the host need runs the common NVMe tier split ([`placement::expert_nvme_tier`], which refuses
    /// a room under the tier's floor by name), and a plan that comes out paging any routed expert is
    /// refused ([`PlaceError::NoNvmeTier`]), for the body serves no such tier.
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        if let Some((tier, card)) = machine.tiers.iter().enumerate().next() {
            return Err(PlaceError::Tier {
                card: card.name.clone(),
                tier,
            });
        }
        let program = program_of(self.spec.arch)
            .ok_or_else(|| PlacementError::Unimplemented(self.unimplemented()))?;
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let mut plan = placement::plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &kv,
            levers,
            routed_rule(program),
            self.prompt_reserve,
        )?;
        let need = HostNeed::of(&plan, 0).bytes();
        let room = plan.plan_at(&self.room);
        placement::expert_nvme_tier(&mut plan, room, self.arena)?;
        if plan.host.nvme_expert_bytes > 0 {
            return Err(PlaceError::NoNvmeTier {
                paged: plan.host.nvme_expert_bytes,
                need,
                room: plan.host.room_clause(),
                engine: self.engine,
            });
        }
        let plan = checked_beside(plan, self.prompt_reserve).map_err(PlaceError::Broken)?;
        for (card, totals) in machine.cards.iter().zip(&plan.cards) {
            let limit = plan.usable_bytes(card).saturating_sub(card.margin_bytes);
            refuse_reserve(card, card_terms(totals), limit, self.prompt_reserve)?;
        }
        Ok(plan)
    }

    /// What one resident sequence of a load of the file holds on its card ([`SeqTerms`]).
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        self.kv.seq_terms()
    }
}

/// The prompt batch's `reserve` beside `total` bytes of a plan's terms on `card`, `limit` its
/// usable bytes less its margin: refused by name when the terms fit alone and the reserve does not
/// fit beside them (terms past the limit are the plan's own violation).
fn refuse_reserve(card: &Card, total: u64, limit: u64, reserve: u64) -> Result<(), PlaceError> {
    match limit.checked_sub(total) {
        Some(left) if reserve > left => Err(PlaceError::PromptOver {
            card: card.name.clone(),
            reserve,
            left,
            short: reserve - left,
        }),
        _ => Ok(()),
    }
}

/// A plan's terms on a card: its granules, cache, scratch, context and reserves.
fn card_terms(t: &CardTotals) -> u64 {
    t.dense_bytes
        + t.expert_bytes
        + t.rounding_bytes
        + t.kv_bytes
        + t.scratch_bytes
        + t.context_bytes
        + t.reserve_bytes
}

/// What one trunk layer holds per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// A GQA layer: K and V of `row` bytes a position, over its `window` positions when it has one.
    Attention { row: u64, window: Option<u64> },
    /// A delta-rule layer: its state, stamp and conv ring, whatever the context.
    State(u64),
    /// A layer with no token mixer.
    Nothing,
}

/// One file's per-layer stores, read off its description. A full GQA layer holds a plane of every
/// position. A window layer holds that plane under [`Rows::Plane`], and
/// [`runtime::stores::ring_rows`] rows under [`Rows::Ring`]; each row is the layer's key heads' key
/// and value rows in f16. A delta-rule layer holds a state of `v_heads` heads of `d × d` f32, one
/// stamped lane, and the conv ring (`runtime::stores::recurrent_bytes`, `delta_lane_bytes`).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer.
    holds: Vec<Hold>,
    /// Which row count a window layer uses.
    rows: Rows,
    /// The rope rows the body copies to its card, in bytes a position: one table of `dims` f32
    /// values a position for each distinct rope (base and dims) the rotating layers turn by.
    rope_row: u64,
}

impl KvLayout {
    /// The layout of the trunk layers `mixers` name, window layers on the ring; a mixer whose store
    /// this owner does not size (a latent or Mamba-2 layer, a selecting GQA layer) is refused by
    /// name with its layer.
    pub fn of<'a>(mixers: impl IntoIterator<Item = &'a Mixer>) -> Result<KvLayout, PlacementError> {
        let mut holds = Vec::new();
        let mut ropes: Vec<(u32, u32)> = Vec::new();
        for (layer, mixer) in mixers.into_iter().enumerate() {
            holds.push(match mixer {
                Mixer::Gqa(g) if g.select.is_none() => {
                    if g.rope.mode != RopeMode::None {
                        let rope = (g.rope.base.to_bits(), g.rope.dims);
                        if !ropes.contains(&rope) {
                            ropes.push(rope);
                        }
                    }
                    Hold::Attention {
                        row: u64::from(g.kv_heads)
                            * (u64::from(g.head_dim) + u64::from(g.value_dim))
                            * stores::F16_BYTES,
                        window: g.window.map(u64::from),
                    }
                }
                Mixer::DeltaRule(d) => Hold::State(state_bytes(d)),
                Mixer::None => Hold::Nothing,
                other => {
                    return Err(PlacementError::Unimplemented(vec![Unimplemented {
                        layer: Some(layer),
                        feature: format!(
                            "{}: this plan sizes the stores of a plain GQA layer and a \
                             delta-rule layer",
                            other.what()
                        ),
                    }]));
                }
            });
        }
        Ok(KvLayout {
            holds,
            rows: Rows::Ring,
            rope_row: ropes
                .iter()
                .map(|&(_, dims)| u64::from(dims) * stores::F32_BYTES)
                .sum(),
        })
    }

    /// The same layout with the window layers on `rows`.
    #[must_use]
    pub fn with_rows(self, rows: Rows) -> KvLayout {
        KvLayout { rows, ..self }
    }

    /// The bytes of layers `layers` at `ctx_max` positions: what a card that runs them holds beside
    /// its weights ([`KvBytes::layer_bytes`] summed).
    #[must_use]
    pub fn bytes(&self, layers: Range<usize>, ctx_max: u64) -> u64 {
        layers.map(|l| self.layer_bytes(l, ctx_max)).sum()
    }

    /// The rows layer `layer` holds at `ctx_max`. A layer that keeps no positions, or is past the
    /// trunk, holds none.
    #[must_use]
    pub fn rows_at(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.holds.get(layer) {
            Some(Hold::Attention { window, .. }) => self.rows.count(*window, ctx_max),
            _ => 0,
        }
    }

    /// One resident sequence's stores over the trunk, nothing beside them.
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: self,
                count: self.holds.len(),
            },
            draft: None,
            beside: 0,
        }
    }
}

/// A delta-rule layer's state, one stamped lane and conv ring in bytes.
fn state_bytes(d: &DeltaRule) -> u64 {
    let (v_heads, k_heads) = (d.v_heads as usize, d.k_heads as usize);
    let (state, conv) = (d.d as usize, d.conv as usize);
    stores::recurrent_bytes(v_heads, k_heads, state, conv)
        + stores::delta_lane_bytes(v_heads, state, 1)
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.holds.get(layer) {
            Some(Hold::Attention { row, .. }) => self.rows_at(layer, ctx_max).saturating_mul(*row),
            Some(Hold::State(bytes)) => *bytes,
            Some(Hold::Nothing) | None => 0,
        }
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        self.rope_row * ctx_max
    }
}

#[cfg(test)]
mod tests {
    use models::{DeltaKind, Gqa, KdaGate, Mamba2, OutGate, Rope};
    use runtime::stores::ring_rows;

    use super::*;

    /// A GQA layer of `kv` key heads, key width `k` and value width `v`, a window of `window`
    /// positions or none, and a rope of `rope` (base, dims) or none.
    fn gqa(kv: u32, (k, v): (u32, u32), window: Option<u32>, rope: Option<(f32, u32)>) -> Mixer {
        Mixer::Gqa(Gqa {
            heads: kv * 4,
            kv_heads: kv,
            head_dim: k,
            value_dim: v,
            rope: rope.map_or(
                Rope {
                    mode: RopeMode::None,
                    dims: 0,
                    base: 0.0,
                    yarn: None,
                },
                |(base, dims)| Rope {
                    mode: RopeMode::Neox,
                    dims,
                    base,
                    yarn: None,
                },
            ),
            qk_norm: false,
            out_gate: OutGate::None,
            select: None,
            window,
            sinks: false,
            value_scale: None,
        })
    }

    /// A layer's rows: its own key-head count's key and value rows in f16 a position. Layers 1 and
    /// 2 are a window of 8; 0 and 3 are full. At 9 positions the ring still holds every position,
    /// because [`ring_rows`] pads the window plus one prompt batch past 9. Past that pad the ring
    /// stays at the pad and the plane grows with the context. A layer past the trunk holds nothing.
    /// The widths are key 16 and value 8 values, 2 key heads on a full layer and 1 on a window one.
    #[test]
    fn layer_bytes() {
        let pad = ring_rows(8, u64::MAX);
        assert!(pad > 9, "the ring pad is past the 9-position case");
        let mixers = [
            gqa(2, (16, 8), None, None),
            gqa(1, (16, 8), Some(8), None),
            gqa(1, (16, 8), Some(8), None),
            gqa(2, (16, 8), None, None),
        ];
        let ring = KvLayout::of(&mixers).expect("GQA layers are sized");
        let plane = ring.clone().with_rows(Rows::Plane);
        for kv in [&ring, &plane] {
            assert_eq!(kv.layer_bytes(0, 9), 9 * 2 * 48);
            assert_eq!(kv.layer_bytes(1, 9), 9 * 48);
            assert_eq!(kv.layer_bytes(4, 9), 0);
            assert_eq!(kv.bytes(0..4, 1), 6 * 48);
            assert_eq!(kv.bytes(0..4, 9), 9 * 6 * 48);
        }
        let past = pad + 32;
        assert_eq!(ring.rows_at(1, past), pad);
        assert_eq!(plane.rows_at(1, past), past);
        assert_eq!(ring.layer_bytes(1, past), pad * 48);
        assert_eq!(plane.layer_bytes(1, past), past * 48);
        assert_eq!(ring.layer_bytes(0, past), past * 2 * 48);
        assert_eq!(plane.layer_bytes(0, past), past * 2 * 48);
    }

    /// The published MiMo trunk: nine full layers of 4 key heads and thirty-nine window layers of
    /// 8, each value 320 f16 bytes a head. A position is 222,720 B on either term, because the ring
    /// is the context. At the gate context the plane is that figure times 4,160 and the ring is the
    /// full layers' plane plus 768 window rows.
    #[test]
    fn the_ring_bytes_at_the_gate_context() {
        let full = [0usize, 5, 11, 17, 23, 29, 35, 41, 47];
        let mixers: Vec<Mixer> = (0..48)
            .map(|l| {
                if full.contains(&l) {
                    gqa(4, (192, 128), None, None)
                } else {
                    gqa(8, (192, 128), Some(128), None)
                }
            })
            .collect();
        let ring = KvLayout::of(&mixers).expect("GQA layers are sized");
        let plane = ring.clone().with_rows(Rows::Plane);
        assert_eq!(plane.bytes(0..48, 1), 222_720);
        assert_eq!(ring.bytes(0..48, 1), 222_720);
        assert_eq!(plane.bytes(0..48, 4_160), 4_160 * 222_720);
        assert_eq!(ring.bytes(0..48, 4_160), 249_200_640);
        assert_eq!(ring_rows(128, 4_160), 768);
    }

    /// A delta-rule layer at Solar-Open2's published widths (64 heads of 128, conv 4): the state
    /// 64 × 128 × 128 f32 = 4,194,304 B, the conv ring (3 + 8 rows) of the 3 × 64 × 128 q, k and
    /// v channels in f32 = 1,081,344 B, and one lane's u32 stamp = 4 B [derived]: the same on every
    /// context. Mutant: the stamp or the pass rows left out of the ring.
    #[test]
    fn a_delta_rule_layer_holds_its_state_whatever_the_context() {
        let kda = Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Kda {
                gate: KdaGate::Softplus,
                beta_scale: 2.0,
            },
            k_heads: 64,
            v_heads: 64,
            d: 128,
            conv: 4,
        });
        let kv = KvLayout::of([&kda, &gqa(8, (128, 128), None, None)]).expect("both are sized");
        assert_eq!(kv.layer_bytes(0, 1), 4_194_304 + 1_081_344 + 4);
        assert_eq!(kv.layer_bytes(0, 1 << 20), kv.layer_bytes(0, 1));
        assert_eq!(kv.rows_at(0, 4096), 0);
        assert_eq!(kv.layer_bytes(1, 4096), 4096 * 2 * 8 * 128 * 2);
    }

    /// One sequence's terms are the layout's stores over the trunk with nothing beside them, and
    /// `slots` sequences count each layer `slots` times — the card's kv class of a load that serves
    /// them.
    #[test]
    fn seq_terms_count_every_sequence() {
        let mixers = [
            gqa(2, (16, 8), None, None),
            gqa(1, (16, 8), Some(8), None),
            gqa(1, (16, 8), Some(8), None),
            gqa(2, (16, 8), None, None),
        ];
        let kv = KvLayout::of(&mixers).expect("GQA layers are sized");
        let terms = kv.seq_terms();
        assert_eq!(terms.bytes(9, 1), kv.bytes(0..4, 9));
        assert_eq!(terms.bytes(9, 3), 3 * kv.bytes(0..4, 9));
        assert_eq!(terms.plan_kv(9, 3), 3 * kv.bytes(0..4, 9));
        let three = terms.slots_of(3);
        for l in 0..4 {
            assert_eq!(three.layer_bytes(l, 9), 3 * kv.layer_bytes(l, 9));
        }
        assert_eq!(three.layer_bytes(4, 9), 0);
    }

    /// The stage's rope rows: one table a distinct rope the rotating layers turn by, `dims` f32
    /// values a position, as the body copies them to its card. Two bases are two tables, the same
    /// base on every layer is one, and a layer that turns nothing adds none.
    #[test]
    fn the_stage_rope_rows_are_a_table_a_rope() {
        let rope = |bases: [f32; 3]| {
            let mixers = bases.map(|b| gqa(1, (16, 8), None, (b > 0.0).then_some((b, 4))));
            KvLayout::of(&mixers)
                .expect("GQA layers are sized")
                .stage_position_bytes(10)
        };
        assert_eq!(rope([1e6, 1e7, 1e6]), 2 * 4 * 4 * 10);
        assert_eq!(rope([1e7, 1e7, 1e7]), 4 * 4 * 10);
        assert_eq!(rope([0.0, 0.0, 0.0]), 0);
        assert_eq!(rope([0.0, 1e7, 0.0]), 4 * 4 * 10);
    }

    /// A mixer whose store this owner does not size is refused by name with its layer, never read
    /// as a layer that holds nothing.
    #[test]
    fn a_mixer_with_no_sized_store_is_refused_by_name() {
        let mamba = Mixer::Mamba2(Mamba2 {
            d_inner: 8,
            d_state: 4,
            n_heads: 2,
            head_dim: 4,
            n_groups: 1,
            d_conv: 4,
            d_in_proj: 24,
        });
        let err = KvLayout::of([&gqa(1, (16, 8), None, None), &mamba]).expect_err("refused");
        let PlacementError::Unimplemented(items) = err else {
            panic!("{err:?}, not the unimplemented list");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].layer, Some(1));
        assert!(items[0].feature.contains("a Mamba-2 mixer"), "{items:?}");
    }

    /// A stack is card-routed when the common kernel table gives its type the entries this
    /// program's leg launches, and not otherwise: the types the table runs on the card for other
    /// families (Q8_0 and IQ4_NL) and the ones it has no row for (Q2_K, the floats) keep their
    /// layer's experts on the host.
    // PIN(2026-10-11): card_runs now runs Q4_K/Q5_K through the common leg.
    // PIN(2026-10-11): the leg also runs IQ2_XS, IQ3_XXS and IQ4_XS (the raw-block entries of
    // `iq_sel.rs`), so those three are card-routed too.
    // PIN(2026-10-12): the leg also runs IQ3_S (`iq3_s_gate_up_sel`, `iq3_s_gemv_sel`), so it is
    // card-routed too.
    #[test]
    fn only_the_stacks_the_leg_runs_are_card_routed() {
        for ty in [
            GgmlType::MXFP4,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            assert_eq!(card_routed(ty), Some(CardFormat::KQuant), "{ty}");
        }
        for ty in [
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::F32,
            GgmlType::BF16,
        ] {
            assert_eq!(card_routed(ty), None, "{ty}");
        }
    }

    /// A host room under the host experts' need is refused by name: the need, the bytes the plan
    /// pages, the room the plan was made at, and the engine that serves no NVMe expert tier.
    #[test]
    fn a_room_under_the_host_need_is_refused_by_name() {
        let err = PlaceError::NoNvmeTier {
            paged: 1_073_741_824,
            need: 150_000_000_000,
            room: "; the plan was made at a reading of 149000000000 B (plan room 149000000000 B)"
                .to_owned(),
            engine: "MiMo",
        };
        assert_eq!(
            err.to_string(),
            "the host need 150000000000 B passes the room, so the plan pages 1073741824 B of \
             routed experts to the NVMe expert tier; the plan was made at a reading of \
             149000000000 B (plan room 149000000000 B), and the MiMo engine serves no NVMe \
             expert tier; free host memory (the room is the host's available bytes, or \
             BLOOMERY_HOST_ROOM where set)"
        );
    }
}
