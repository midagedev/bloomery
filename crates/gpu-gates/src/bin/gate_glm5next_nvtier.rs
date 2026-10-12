//! GPU gate for GLM-5.3-Flash's attach of the NVMe expert tier (the split dial's arena, `Body::open_placed`'s host run
//! through the common `host_run`): the five clauses of `shared/nvtier_family.rs` (`plan`, `attach`, `bits`, `refuse`,
//! `residency`) on the gate placement — in the fixture tier the `q6kdown` variant (`tools/fixture-variants.tsv`: Q6_K
//! routed downs, every routed layer on the host), planned under the header's card budget — a paged arm against an
//! all-RAM arm on the same file, the residency off.
//!
//! Each arm's host room and arena reach the plan through `PlanInputs` (`room`, `arena`), so the gate sets no
//! environment: the all-RAM arm's room is the plan's need, the paged arm's is the split's floor F (read off the
//! `HostRoomFloor` refusal) plus an arena of the slot sets the family sizes from the file (`paged_slots`: at most
//! [`nvtier_family::SLOTS`], fewer where the need leaves less above F). One prompt call of [`PROMPT`] prose ids, then
//! [`STEPS`] greedy steps, on each arm; one load at a time.
//!
//! GLM's own refusal, beside the family's (`refuse nextn-room`): the draft layer's routed experts are the host's
//! beside the trunk's, a plan of their own, so the NextN plan splits the trunk at the room less them
//! (`PlanInputs::plan_nextn_slots`). The NextN plan at the paged arm's own floor plus the arena (`F_n + A`) is
//! refused by name — the trunk gets `F_n + A − D` — and at `F_n + A + D` it plans, pages routed experts through the
//! same arena. GLM has no r8 sidecar, so the family's r8 refusal has no counterpart.
//!
//! Tiers: every clause is self-consistency (the arms and the plan's own numbers), so it runs on a fixture and on the
//! real file alike, except on a fixture that keeps no more than one routed layer on the host (the plain fixture): the
//! clauses that read the paged experts are left to the real tier by name (`no-host-layer`, [`tier::host_clause`]), and
//! the `q6kdown` variant, whose four routed layers all stay on the host, runs them: its floor holds three times the
//! heaviest layer's host-served experts, so its need leaves fewer than [`nvtier_family::SLOTS`] slot sets above it and
//! the arena is what fits. A variant whose routed layers the card reads (`iq4xs`, `q2kxl`) is refused by name there,
//! so no such variant lists this gate.

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
    use bloomery_gpu_gates::{Fnv1a64, GateError, prose_ids, tier};
    use bloomery_gpu_glm5next::{Body, PrefillMode, checkpoint_price};
    use bloomery_levers::{GlmAt, GlmWhy, HostCfg, glm_residency_at_plan, glm_unset};
    use gguf::Split;
    use model::arch::glm5next::hparams::Hparams;
    use model::arch::glm5next::place::{NextnInputs, PlaceError, PlanInputs};
    use model::arch::names::routed_stacks;
    use model::placement::accepts::{self, Program};
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
        place: PlanLevers,
        host: HostCfg,
    }

    impl Glm {
        /// The inputs of `file` with `arm`'s room and arena.
        fn inputs(&self, file: &Split, arm: Arm) -> Result<PlanInputs, GateError> {
            let mut inputs = PlanInputs::read(file)?;
            inputs.room = (arm.room, HostRead::Given);
            inputs.arena = arm.arena;
            Ok(inputs)
        }

        /// The trunk's routed layers whose experts stay on the host, from the file's types: a layer
        /// whose three stacks the card leg does not run (`accepts::leg_runs`, the arms the plan's
        /// rule reads).
        fn host_layers(file: &Split, hp: &Hparams) -> usize {
            (hp.dense_lead..hp.n_trunk)
                .filter(|&l| {
                    let ty = routed_stacks(l).map(|name| file.find(&name).map(|(_, info)| info.ty));
                    !matches!(ty, [Some(gate), Some(up), Some(down)]
                        if accepts::leg_runs(Program::Glm5nextBody, [gate, up, down]) == Some(true))
                })
                .count()
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
            let inputs = self.inputs(file, arm)?;
            let machine = Glm::machine(&inputs)?;
            match inputs.plan_nextn(&machine, u64::try_from(CTX)?, &self.place, nextn) {
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
            let inputs = self.inputs(&file, arm)?;
            let machine = Glm::machine(&inputs)?;
            match inputs.plan(&machine, u64::try_from(CTX)?, &self.place) {
                Ok(plan) => Ok(Ok(PlanFacts::of(&plan, GLM_CARD)?)),
                Err(PlaceError::Placement(e)) => Ok(Err(e)),
                Err(e) => Err(e.into()),
            }
        }

        fn load(&self, arm: Arm, _r8: bool) -> Result<Session<Body>, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = self.inputs(&file, arm)?;
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

        fn finish(&self, mut s: Session<Body>) -> Result<(), GateError> {
            s.clear()?;
            drop(s);
            Ok(())
        }

        /// The seat's own two calls: `glm_unset` on a serving placement, then `glm_residency_at_plan` on the plan
        /// (`serve_seats/glm.rs`, `residency_at`), the why in the family gate's names ([`why_name`]).
        fn unset_residency(&self, arm: Arm) -> Result<(String, &'static str), GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = self.inputs(&file, arm)?;
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
        /// `no_card_experts`, whatever the arena; on the A6000 the card experts, the pool and the headroom decide.
        fn expected_residency(&self, plan: &PlanFacts) -> Unset {
            Unset::serving(&plan.room)
        }

        /// More than one routed layer stays on the host, so the tier has something to page: an
        /// approximation of the plan's own condition, whose miss is a red clause, never a deferral.
        fn host_layers_enough(&self) -> Result<bool, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let hp = PlanInputs::read(&file)?.hp;
            Ok(Glm::host_layers(&file, &hp) > 1)
        }

        fn refusals(&self, paged: Arm, slots: u64) -> Vec<Refusal<'_>> {
            vec![Refusal {
                what: "nextn-room",
                holds: "floor",
                attempt: Box::new(move || self.nextn_room(paged, slots)),
            }]
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&tier::acts_on(&[])?)?;
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        let path = glm5next_tier::model_path()?;
        let place = glm5next_tier::plan_levers(&levers, 0)?;
        crate::nvtier_family::gate(
            NAME,
            &Glm {
                path,
                place,
                host: levers.host(),
            },
        )
    }
}
