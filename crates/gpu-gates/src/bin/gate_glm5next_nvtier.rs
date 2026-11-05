//! GPU gate for GLM-5.3-Flash's attach of the NVMe expert tier (the split dial's arena, `Body::open_placed`'s host run
//! through the common `host_run`): the six clauses of `shared/nvtier_family.rs` (`plan`, `attach`, `bits`, `card`,
//! `refuse`, `residency`) on the gate placement — in the fixture tier the plain fixture, planned under a card budget
//! derived from the plan itself (the Tiers paragraph), not the header's — a paged arm against an all-RAM arm on the
//! same file, the residency off. GLM's load attaches the step port's card leg, so the paged arm runs on both arena
//! memories: the host leg's (`anon`) for `bits` and the card leg's (`pinned`) for `card`.
//!
//! Each arm's host room and arena reach the plan through `PlanInputs` (`room`, `arena`), so the gate sets no
//! environment beyond the paged arm's child's arena memory: the all-RAM arm's room is the plan's need, the paged
//! arm's is the split's floor F (read off the `HostRoomFloor` refusal) plus an arena of the slot sets the family
//! sizes from the file (`paged_slots`: at most [`nvtier_family::SLOTS`], fewer where the need leaves less above F).
//! One prompt call of [`PROMPT`] prose ids, then [`STEPS`] greedy steps, on each arm; one load at a time.
//!
//! GLM's own refusal, beside the family's (`refuse nextn-room`): the draft layer's routed experts are the host's
//! beside the trunk's, a plan of their own, so the NextN plan splits the trunk at the room less them
//! (`PlanInputs::plan_nextn_slots`). The NextN plan at the paged arm's own floor plus the arena (`F_n + A`) is
//! refused by name — the trunk gets `F_n + A − D` — and at `F_n + A + D` it plans, pages routed experts through the
//! same arena. GLM has no r8 sidecar, so the family's r8 refusal has no counterpart.
//!
//! Tiers: every clause is self-consistency (the arms and the plan's own numbers), so it runs on a fixture and on the
//! real file alike. The fixture tier's card budget is the least one its plan accepts, read off the plan itself
//! ([`floor_budget`]): the floor a zero-budget probe is refused by name with (`PlacementError::CardBudgetFloor`: the
//! card's dense granules, its KV, context, scratch, reserves, margin and what the caller holds beside the plan), the
//! budget that holds no routed expert on the card — so every routed layer's experts stay on the host, the tier pages
//! them, and every layer stays one the card leg runs (the fixture's stacks are the arena entries' own types: Q4_K and
//! Q5_K gate·up, Q5_K and Q6_K down). The header's recorded budget instead cards half the experts of every
//! card-eligible layer, under which the host holds the whole of its share and nothing pages. The NextN plan reserves
//! the draft's card bytes and arena out of the same budget, so its probes carry their own least budget, derived the
//! same way ([`Glm::place_nextn`]). The floor holds three times the heaviest layer's host-served experts
//! (`ArenaDial::floor`, `base + 3 W`), so the need may leave fewer than [`nvtier_family::SLOTS`] slot sets above it
//! and the arena is what fits. No fixture variant lists this gate: the paged layers come from the budget, not from a
//! type map, and a variant whose stacks the arena entries do not take would fail `card` red (its leg `off_types`),
//! not defer.

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_nvtier", gate::run())
}

#[cfg(feature = "glm5next")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
#[path = "shared/nvtier_family.rs"]
mod nvtier_family;

#[cfg(feature = "glm5next")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "glm5next")]
mod gate {
    use std::sync::Arc;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, Open, Session};
    use bloomery_gpu::host::nvtier::NvTier;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::residency::PlanRoom;
    use bloomery_gpu_gates::residency38::{GLM_CARD, Seqs, reserve_checkpoints};
    use bloomery_gpu_gates::{Fnv1a64, GateError, prose_ids, record, tier};
    use bloomery_gpu_glm5next::{Body, PrefillMode, checkpoint_price};
    use bloomery_levers::{GlmAt, GlmWhy, HostCfg, glm_residency_at_plan, glm_unset};
    use gguf::Split;
    use model::arch::glm5next::place::{NextnInputs, PlaceError, PlanInputs};
    use model::placement::workstation::HostRead;
    use model::placement::{Machine, PlacementError, PlanLevers};
    use runtime::{Out, Target, Want};

    use crate::glm5next_tier;
    use crate::nvtier_family::{Arm, Family, PlanFacts, Pos, Refusal, Unset};
    use crate::quiet::Quiet;

    const NAME: &str = "gate_glm5next_nvtier";
    /// The positions the stores hold: the e2e and MTP gates' context, the one the fixture's card budget is planned at.
    const CTX: usize = 3136;
    /// The prompt call's ids: one batch.
    const PROMPT: usize = 512;
    /// Greedy steps after the call.
    const STEPS: usize = 8;
    /// A room the NextN probe adds to the draft's bytes: under every file's floor beside them.
    const PROBE_ROOM: u64 = 1 << 30;

    /// GLM's file and placement, as the family's closures read them.
    struct Glm {
        path: String,
        /// The trunk's plan levers: in the fixture tier the least card budget its plan accepts
        /// (no routed expert on the card, so the tier has host-served experts to page), in the
        /// real tier the caller's.
        place: PlanLevers,
        /// [`Glm::place`] for the NextN probes: the NextN plan holds the draft's card bytes and
        /// arena beside the plan, so in the fixture tier its least budget is its own floor's, not
        /// the trunk's.
        place_nextn: PlanLevers,
        host: HostCfg,
    }

    impl Glm {
        /// The inputs of `file` with `arm`'s room and arena.
        fn inputs(file: &Split, arm: Arm) -> Result<PlanInputs, GateError> {
            let mut inputs = PlanInputs::read(file)?;
            inputs.room = (arm.room, HostRead::Given);
            inputs.arena = arm.arena;
            Ok(inputs)
        }

        /// The gate placement of `inputs`' layers, its checkpoints reserved once at the gate's context.
        fn machine(inputs: &PlanInputs) -> Result<Machine, GateError> {
            let mut machine = crate::gate_card::plan_gate(inputs.model.layers);
            reserve_checkpoints(
                &mut machine,
                Seqs::priced(1, checkpoint_price(inputs)?),
                u64::try_from(CTX)?,
            )?;
            Ok(machine)
        }

        /// The open's configuration: the host set as the levers say, the r8 pair off.
        fn cfg(&self) -> GlmCfg {
            let mut host = self.host;
            host.r8 = false;
            GlmCfg {
                place: self.place.clone(),
                host,
                prefill: PrefillMode::Batch,
                group: 1,
            }
        }

        /// The NextN plan of `file` at `room` and `arena`: its target's plan facts and the draft layer's routed
        /// expert bytes, or the planner's own refusal.
        fn nextn(
            &self,
            file: &Split,
            nextn: &NextnInputs,
            arm: Arm,
        ) -> Result<Result<(PlanFacts, u64), PlacementError>, GateError> {
            let inputs = Glm::inputs(file, arm)?;
            let machine = Glm::machine(&inputs)?;
            match inputs.plan_nextn(&machine, u64::try_from(CTX)?, &self.place_nextn, nextn) {
                Ok(plan) => Ok(Ok((
                    PlanFacts::of(&plan.plan, GLM_CARD)?,
                    plan.nextn.host.expert_bytes,
                ))),
                Err(PlaceError::Placement(e)) => Ok(Err(e)),
                Err(e) => Err(e.into()),
            }
        }

        /// `refuse nextn-room`: see the module doc.
        fn nextn_room(&self, paged: Arm, slots: u64) -> Result<(), GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let nextn = NextnInputs::read(&PlanInputs::read(&file)?)?;
            let arena = paged.arena;
            // D: the draft layer's routed experts, from the NextN plan at a room none of them passes.
            let (_, d) = self
                .nextn(
                    &file,
                    &nextn,
                    Arm {
                        room: u64::MAX,
                        arena: None,
                    },
                )?
                .map_err(|e| format!("the NextN plan at no room limit: {e}"))?;
            // F_n: the floor the NextN plan's own split refuses a low room with, under the given arena.
            let floor = match self.nextn(
                &file,
                &nextn,
                Arm {
                    room: d + PROBE_ROOM,
                    arena,
                },
            )? {
                Err(PlacementError::HostRoomFloor { floor, .. }) => floor,
                other => {
                    return Err(format!(
                        "nextn room: the probe at D + {PROBE_ROOM} B is not the floor's refusal: {}",
                        other.map_or_else(|e| e.to_string(), |_| "planned".to_string())
                    )
                    .into());
                }
            };
            let a = arena.ok_or("nextn room: the paged arm has no given arena")?;
            let with = Arm {
                room: floor + a + d,
                arena,
            };
            let (facts, _) = self.nextn(&file, &nextn, with)?.map_err(|e| {
                format!(
                    "nextn room: no NextN plan at F_n + A + D ({} B): {e}",
                    with.room
                )
            })?;
            println!(
                "nextn room: draft D {d} B, NextN floor F_n {floor} B (the trunk's arm room {} B); at F_n + A + D \
                 = {} B the plan pages {} B through an arena of {} B = {} slots",
                paged.room, with.room, facts.paged, facts.arena, facts.arena_slots
            );
            if facts.paged == 0 || facts.arena != a || facts.arena_slots != slots {
                return Err(format!(
                    "nextn room: the NextN plan at F_n + A + D pages {} B through an arena of {} B = {} slots, not \
                     {a} B = {slots} slots",
                    facts.paged, facts.arena, facts.arena_slots
                )
                .into());
            }
            // The draft's bytes leave the trunk F_n + A − D: under the floor, so the plan is refused by name.
            let without = Arm {
                room: floor + a,
                arena,
            };
            match self.nextn(&file, &nextn, without)? {
                Err(
                    e @ (PlacementError::HostRoomFloor { .. } | PlacementError::NvTierArena { .. }),
                ) => Err(e.into()),
                Err(e) => {
                    println!("nextn room: the plan at F_n + A: {e}");
                    Err("a refusal that is not the room's".into())
                }
                Ok(_) => Ok(()),
            }
        }
    }

    /// The position's argmax and logits digest.
    fn pos_of(out: Out<'_>) -> Result<Pos, GateError> {
        match out {
            Out::Logits { argmax, row } => Ok(Pos {
                argmax,
                fnv: Fnv1a64::default().f32s(row).value(),
            }),
            Out::Argmax(_) => Err("a call asked for its logits returned the argmax alone".into()),
        }
    }

    /// `why` in the names the family gate's table uses ([`Unset`]): the word kept as the rule's `Serving` is `place`,
    /// the word the card slots moved to another P is `shrunk`, the host's pool `pool_shrunk`, the other whys keep
    /// their meaning.
    fn why_name(why: GlmWhy) -> &'static str {
        match why {
            GlmWhy::Serving => "place",
            GlmWhy::Gate => "gate",
            GlmWhy::Nextn { .. } => "nextn",
            GlmWhy::Ctx { .. } => "ctx",
            GlmWhy::PrefillSteps => "prefill_steps",
            GlmWhy::NoCardExperts => "no_card_experts",
            GlmWhy::NoRoom { .. } => "no_room",
            GlmWhy::Moved { .. } => "shrunk",
            GlmWhy::Unhosted(_) => "pool_unhosted",
            GlmWhy::PoolShrunk { .. } => "pool_shrunk",
            GlmWhy::PoolUnhostable(_) => "pool_unhostable",
        }
    }

    impl Family for Glm {
        type Session = Session<Body>;

        fn plan(&self, arm: Arm) -> Result<Result<PlanFacts, PlacementError>, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = Glm::inputs(&file, arm)?;
            let machine = Glm::machine(&inputs)?;
            match inputs.plan(&machine, u64::try_from(CTX)?, &self.place) {
                Ok(plan) => Ok(Ok(PlanFacts::of(&plan, GLM_CARD)?)),
                Err(PlaceError::Placement(e)) => Ok(Err(e)),
                Err(e) => Err(e.into()),
            }
        }

        fn load(&self, arm: Arm, _r8: bool) -> Result<Session<Body>, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = Glm::inputs(&file, arm)?;
            let machine = Glm::machine(&inputs)?;
            let cfg = self.cfg();
            let plan = <Body as Open>::plan(&inputs, &machine, CTX, &cfg)?;
            let mut model = <Body as Open>::open(file, &inputs, &plan, &cfg)?;
            model.set_mode(StepMode::Graph);
            let at = u32::try_from(plan.ctx_max)?;
            Ok(Loaded::<Body>::from_model(model, cfg, at).ready(&mut Quiet)?)
        }

        fn tier(&self, s: &Session<Body>) -> Result<Option<Arc<NvTier>>, GateError> {
            Ok(s.model().body(NAME)?.nvme_tier().cloned())
        }

        fn run(&self, s: &mut Session<Body>) -> Result<Vec<Pos>, GateError> {
            let ids = prose_ids("glm5next", PROMPT)?;
            let mut out = Vec::with_capacity(STEPS + 1);
            out.push(pos_of(s.prompt(&ids, Want::Logits)?)?);
            for _ in 0..STEPS {
                let next = out[out.len() - 1].argmax;
                out.push(pos_of(s.step(next, Want::Logits)?)?);
            }
            Ok(out)
        }

        fn eager(&self, s: &mut Session<Body>) -> Result<(), GateError> {
            s.clear()?;
            s.model_mut().set_mode(StepMode::Eager);
            Ok(())
        }

        /// The placed load (`Body::load_placed`) attaches the step port's card leg after its tier cards.
        fn card_leg(&self) -> bool {
            true
        }

        fn finish(&self, mut s: Session<Body>) -> Result<(), GateError> {
            s.clear()?;
            drop(s);
            Ok(())
        }

        /// The seat's own two calls: `glm_unset` on a serving placement, then `glm_residency_at_plan` on the plan
        /// (`serve_seats/glm.rs`, `residency_at`), the why in the family gate's names ([`why_name`]).
        fn unset_residency(&self, arm: Arm) -> Result<(String, &'static str), GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = Glm::inputs(&file, arm)?;
            let machine = Glm::machine(&inputs)?;
            let plan = inputs.plan(&machine, u64::try_from(CTX)?, &self.place)?;
            let unset = glm_unset(GlmAt {
                serving_place: true,
                nextn_layers: 1,
                need: 0,
                ctx: 0,
                prefill_steps: false,
            })
            .residency;
            let room = PlanRoom::new(&plan, GLM_CARD, 0);
            let pick = glm_residency_at_plan(
                <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
                unset,
                room.card_experts(),
                |pinned| room.pool_bytes(pinned),
                room.headroom(),
                room.mem_left()?,
            )
            .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?;
            Ok((pick.word.to_string(), why_name(pick.why)))
        }

        /// The rule is `glm_residency_at_plan` (`crates/levers/src/lib.rs:2134`), which runs `room_for` (`:1279`)
        /// with no clause of its own beside the pass-through of a word that is not `mid` (`:2141`): the table of
        /// [`Unset::serving`] holds as it is, its `place` and `shrunk` being the rule's `Serving` and `Moved`.
        /// On the 3090 the plan holds no routed expert on the card, so the first clause (`:1287`) answers `off`,
        /// `no_card_experts`, whatever the arena (the fixture tier's derived budget holds none either,
        /// [`Glm::place`]); on the A6000 the card experts, the pool and the headroom decide.
        fn expected_residency(&self, plan: &PlanFacts) -> Unset {
            Unset::serving(&plan.room)
        }

        fn refusals(&self, paged: Arm, slots: u64) -> Vec<Refusal<'_>> {
            vec![Refusal {
                what: "nextn-room",
                holds: "floor",
                attempt: Box::new(move || self.nextn_room(paged, slots)),
            }]
        }
    }

    /// The least card budget the plan `make` builds accepts: the floor its zero-budget probe is refused by name
    /// with ([`PlacementError::CardBudgetFloor`]'s `floor`: the card's dense granules, its KV, context, scratch,
    /// reserves, margin and what the caller holds beside the plan). At it the plan holds no routed expert on the
    /// card, every routed layer's experts the host's — the placement a paged arm pages through the tier.
    ///
    /// # Errors
    /// A probe that plans at a zero budget, or is refused for a reason other than the floor's, each by name.
    fn floor_budget(
        make: impl FnOnce(&PlanLevers) -> Result<(), PlaceError>,
    ) -> Result<u64, GateError> {
        match make(&PlanLevers {
            card_budget_bytes: Some(0).into(),
            ..PlanLevers::default()
        }) {
            Err(PlaceError::Placement(PlacementError::CardBudgetFloor { floor, .. })) => Ok(floor),
            Err(e) => Err(format!("the zero card budget's probe: {e}").into()),
            Ok(()) => Err(
                "the zero card budget's probe planned: the card holds routed experts at no budget"
                    .into(),
            ),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&tier::acts_on(&[])?)?;
        record::at_main(NAME, record::GATE_NVTIER);
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        let path = glm5next_tier::model_path()?;
        let mut place = glm5next_tier::plan_levers(&levers, 0)?;
        // The fixture tier plans under the least budget its own plan accepts, not the header's: the
        // header's cards half the experts of every card-eligible layer, under which the host holds
        // the whole of its share and nothing pages. The NextN plan holds the draft's card bytes and
        // arena beside the plan, so its least budget is its own.
        let mut place_nextn = place.clone();
        if tier::Tier::from_env()? == tier::Tier::Fixture {
            let file = Split::open(&path).map_err(|e| format!("open: {e}"))?;
            let probe = Glm::inputs(
                &file,
                Arm {
                    room: u64::MAX,
                    arena: None,
                },
            )?;
            let machine = Glm::machine(&probe)?;
            let ctx = u64::try_from(CTX)?;
            let budget = floor_budget(|l| probe.plan(&machine, ctx, l).map(|_| ()))?;
            let nextn = NextnInputs::read(&probe)?;
            let nextn_budget =
                floor_budget(|l| probe.plan_nextn(&machine, ctx, l, &nextn).map(|_| ()))?;
            place.card_budget_bytes = Some(budget).into();
            place_nextn.card_budget_bytes = Some(nextn_budget).into();
            println!(
                "{NAME}: the fixture tier's card budget: the plan's floor {budget} B, the NextN plan's \
                 {nextn_budget} B, no routed expert on the card"
            );
        }
        crate::nvtier_family::gate(
            NAME,
            &Glm {
                path,
                place,
                place_nextn,
                host: levers.host(),
            },
        )
    }
}
