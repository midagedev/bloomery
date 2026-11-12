//! GPU gate for V4.1's attach of the NVMe expert tier (the split dial's arena, `body::open_placed_tiered`'s host run
//! through the common `host_run`): the seven clauses of `shared/nvtier_family.rs` (`plan`, `attach`, `bits`, `card`,
//! `refuse`, `residency`, `rank`) on V4.1's placement (a) — the fixture in the fixture tier, planned under its
//! header's card budget — a paged arm against an all-RAM arm on the same file, the residency `off`, the r8 pair off.
//! V4.1's load attaches no card leg, so `card` holds its tier `unattached`; `rank` writes a router rank over the
//! file's routed layers and runs the paged arm under it against the same arm without it.
//!
//! Each arm's host room and arena reach the plan through `PlanInputs` (`room`, `arena`), so the gate sets no
//! environment: the all-RAM arm's room is the plan's need, the paged arm's is the split's floor F (read off the
//! `HostRoomFloor` refusal) plus an arena of the slot sets the family sizes from the file (`paged_slots`: at most
//! [`nvtier_family::SLOTS`], fewer where the need leaves less above F). One prompt call of [`PROMPT`] prose ids, then
//! [`STEPS`] greedy steps, on each arm; one load at a time.
//!
//! V4.1's own refusal, beside the family's: an r8 pair that reads the file's sidecar on the paged plan, refused at
//! the load (`bloomery_gpu::model::nvme_tier`).
//!
//! Tiers: every clause is self-consistency (the arms and the plan's own numbers), so it runs on a fixture and on the
//! real file alike.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_nvtier", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the levers and the clause tags; the triangle's facts serve the other gates"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
#[path = "shared/nvtier_family.rs"]
mod nvtier_family;

#[cfg(feature = "deepseek41")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;

    use app::arch::deepseek41::Ds41Cfg;
    use app::{Loaded, Open, Session};
    use bloomery_gpu::host::nvtier::NvTier;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{Body, OpenCfg, PrefillMode};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::prompts::{HEAD_MARGIN_V41, MARGIN_FLOOR};
    use bloomery_gpu_gates::{Fnv1a64, GateError, prose_ids, record, ref_model_path, tier};
    use bloomery_levers::{EXPERT_RANK, ResidencyAt, residency_unset};
    use gguf::Split;
    use model::arch::deepseek41::place::{PlaceError, PlanInputs};
    use model::placement::workstation::{self, HostRead};
    use model::placement::{Machine, PlacementError};
    use runtime::{Out, Target, Want};

    use crate::nvtier_family::{Arm, Family, PlanFacts, Pos, Refusal, Rows, Unset};
    use crate::quiet::Quiet;

    const NAME: &str = "gate_ds41_nvtier";
    /// The prompt call's ids: one batch.
    const PROMPT: usize = 512;
    /// Greedy steps after the call.
    const STEPS: usize = 8;
    /// The rank clause's prompt positions, then the same greedy steps: the
    /// tier gate's own rank clause size, so the run the child leaves beside
    /// its rank file stays a small file.
    const RANK_P: usize = 64;

    /// V4.1's file and placement, as the family's closures read them.
    struct Ds41<M> {
        path: PathBuf,
        cfg: OpenCfg,
        machine: M,
        /// The rank walk's head margin: the real tier's [`HEAD_MARGIN_V41`],
        /// the fixture tier's the e2e floor.
        head: f32,
    }

    impl<M: Fn(usize) -> Machine + Copy> Ds41<M> {
        /// The inputs of `file` with `arm`'s room and arena.
        fn inputs(&self, file: &Split, arm: Arm) -> Result<PlanInputs, GateError> {
            let mut inputs = PlanInputs::read(file)?;
            inputs.room = (arm.room, HostRead::Given);
            inputs.arena = arm.arena;
            Ok(inputs)
        }

        /// The open's configuration, the r8 pair as `r8` says.
        fn ds41(&self, r8: bool) -> Ds41Cfg {
            let mut open = self.cfg.clone();
            open.body.host.r8 = r8;
            Ds41Cfg {
                feed: open.body.prefill,
                open,
                card_timing: false,
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

    impl<M: Fn(usize) -> Machine + Copy> Family for Ds41<M> {
        type Session = Session<Body>;

        fn plan(&self, arm: Arm) -> Result<Result<PlanFacts, PlacementError>, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = self.inputs(&file, arm)?;
            let machine = (self.machine)(inputs.model.layers);
            match inputs.plan(&machine, workstation::CTX_MAX, &self.cfg.place) {
                Ok(plan) => Ok(Ok(PlanFacts::of(&plan, 0)?)),
                Err(PlaceError::Placement(e)) => Ok(Err(e)),
                Err(e) => Err(e.into()),
            }
        }

        fn load(&self, arm: Arm, r8: bool) -> Result<Session<Body>, GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = self.inputs(&file, arm)?;
            let machine = (self.machine)(inputs.model.layers);
            let cfg = self.ds41(r8);
            let ctx = usize::try_from(workstation::CTX_MAX)?;
            let plan = <Body as Open>::plan(&inputs, &machine, ctx, &cfg)?;
            let mut model = <Body as Open>::open(file, &inputs, &plan, &cfg)?;
            model.set_mode(StepMode::Graph);
            let at = u32::try_from(plan.ctx_max)?;
            Ok(Loaded::<Body>::from_model(model, cfg, at).ready(&mut Quiet)?)
        }

        fn tier(&self, s: &Session<Body>) -> Result<Option<Arc<NvTier>>, GateError> {
            Ok(s.model().body(NAME)?.nvme_tier().cloned())
        }

        fn run(&self, s: &mut Session<Body>) -> Result<Vec<Pos>, GateError> {
            let ids = prose_ids("engram", PROMPT)?;
            let mut out = Vec::with_capacity(STEPS + 1);
            out.push(pos_of(s.prompt(&ids, Want::Logits)?)?);
            for _ in 0..STEPS {
                let next = out[out.len() - 1].argmax;
                out.push(pos_of(s.step(next, Want::Logits)?)?);
            }
            Ok(out)
        }

        fn takes_rank(&self) -> bool {
            true
        }

        /// The rank walk's head margin, parsed once at `run` from the tier: the real tier's
        /// [`HEAD_MARGIN_V41`], the fixture tier's the e2e floor.
        fn head_margin(&self) -> f32 {
            self.head
        }

        fn rank_child(&self, cmd: &mut Command, rank: &Path) -> Result<(), GateError> {
            cmd.env(EXPERT_RANK, rank);
            Ok(())
        }

        fn write_rank(&self, dir: &Path) -> Result<PathBuf, GateError> {
            // A rank the lever already named would rank the parent's own arm
            // too, so the clause's comparison would be the rank against
            // itself.
            if let Some(rank) = self.cfg.place.expert_rank.as_deref() {
                return Err(format!(
                    "{EXPERT_RANK} names {}: the rank clause sets it for its child alone",
                    rank.path()
                )
                .into());
            }
            // The stride order the tier gate's rank clause writes: the
            // smallest stride of 3 or more with no factor in common with the
            // expert count, offset by the layer, so a layer's first ids are
            // neither its id prefix nor the same ids of another layer.
            fn gcd(a: u64, b: u64) -> u64 {
                if b == 0 { a } else { gcd(b, a % b) }
            }
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = PlanInputs::read(&file)?;
            let n = inputs.model.experts;
            let stride = (3..n)
                .find(|&s| gcd(s, n) == 1)
                .ok_or_else(|| format!("{n} experts have no stride of 3 or more coprime to it"))?;
            let mut layers = BTreeSet::new();
            for t in &inputs.model.tensors {
                if t.role == model::placement::Role::RoutedExperts {
                    layers
                        .insert(t.layer.ok_or_else(|| {
                            format!("{}: a routed stack without a layer", t.name)
                        })?);
                }
            }
            if layers.is_empty() {
                return Err("the model has no routed layer".into());
            }
            let mut text = format!(
                "# router-rank\tn_expert={n}\tn_used={}\tsets=gate-nvtier\tpositions=1\n",
                u64::try_from(inputs.hp.experts.n_used)?
            );
            for layer in layers {
                let ids: Vec<String> = (0..n)
                    .map(|i| ((i * stride + layer as u64) % n).to_string())
                    .collect();
                text.push_str(&format!("{layer}\t{}\n", ids.join(",")));
            }
            let rank = dir.join("rank.tsv");
            std::fs::write(&rank, text)?;
            Ok(rank)
        }

        fn run_rows(&self, s: &mut Session<Body>) -> Result<Rows, GateError> {
            // The rank clause's run shape: one decode step a position over
            // the prompt ids, then the greedy steps — every position a row
            // the band walks. Each prompt position records the id that was
            // fed as its token — both arms feed the same ids, so the walk
            // judges every prompt row on the same context — save the last,
            // which records its argmax, the seed the greedy steps run on: a
            // parted seed ends the walk before the greedy rows instead of
            // judging rows that ran on other contexts.
            let row_of = |out: Out<'_>| -> Result<(u32, Vec<f32>), GateError> {
                match out {
                    Out::Logits { argmax, row } => Ok((argmax, row.to_vec())),
                    Out::Argmax(_) => {
                        Err("a call asked for its logits returned the argmax alone".into())
                    }
                }
            };
            let ids = prose_ids("engram", RANK_P)?;
            let mut run = Rows::default();
            let mut seed = 0;
            for (k, &id) in ids.iter().enumerate() {
                let (argmax, row) = row_of(s.step(id, Want::Logits)?)?;
                seed = argmax;
                run.push(if k + 1 == ids.len() { argmax } else { id }, row);
            }
            for _ in 0..STEPS {
                let (token, row) = row_of(s.step(seed, Want::Logits)?)?;
                seed = token;
                run.push(token, row);
            }
            Ok(run)
        }

        fn finish(&self, mut s: Session<Body>) -> Result<(), GateError> {
            s.clear()?;
            drop(s);
            Ok(())
        }

        fn unset_residency(&self, arm: Arm) -> Result<(String, &'static str), GateError> {
            let file = Split::open(&self.path).map_err(|e| format!("open: {e}"))?;
            let inputs = self.inputs(&file, arm)?;
            let machine = (self.machine)(inputs.model.layers);
            let plan = inputs.plan(&machine, workstation::CTX_MAX, &self.cfg.place)?;
            let unset = residency_unset(ResidencyAt {
                serving_place: true,
                ..ResidencyAt::FIXED
            });
            let pick = bloomery_gpu_gates::residency::at_plan(
                <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
                &plan,
                0,
                unset,
            )?;
            Ok((pick.word.to_string(), pick.why.name()))
        }

        /// The rule is `residency_at_plan` (`crates/levers/src/lib.rs:1379`) on the serving word, through
        /// `room_for` (`:1279`). A routed stack the card holds whole puts its experts past P in the churn pool
        /// (`crates/placement/src/placement/churn.rs:46-49`) against a headroom the split's floor leaves near 0:
        /// `pool_unhostable`, an unset word climbing on every body. A plan whose every
        /// card-expert layer pages has an empty pool and keeps the word.
        fn expected_residency(&self, plan: &PlanFacts) -> Unset {
            Unset::serving(&plan.room)
        }

        fn refusals(&self, paged: Arm, _slots: u64) -> Vec<Refusal<'_>> {
            vec![Refusal {
                what: "r8",
                holds: "a paged plan under an r8 pair that reads a sidecar",
                attempt: Box::new(move || self.load(paged, true).and_then(|s| self.finish(s))),
            }]
        }
    }

    pub fn run() -> Result<(), GateError> {
        // The rank clause's child parses the rank its parent set in its
        // environment alone.
        let levers = bloomery_levers::at_main(&crate::ds41_tier::acts_on(&[EXPERT_RANK])?)?;
        record::at_main(NAME, record::GATE_NVTIER);
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.residency = Residency::parse("off")?;
        cfg.body.prefill = PrefillMode::Batch;
        let path = ref_model_path()?;
        let head = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // The fixture tier plans under its header's card budget, the lever held equal to it; the real tier sets
        // none.
        cfg.place = tier::plan_levers(&head, &levers, 0)?;
        drop(head);
        let machine = Place::A.machine(None, None)?;
        let head = match tier::Tier::from_env()? {
            tier::Tier::Real => HEAD_MARGIN_V41,
            tier::Tier::Fixture => MARGIN_FLOOR,
        };
        crate::nvtier_family::gate(
            NAME,
            &Ds41 {
                path,
                cfg,
                machine,
                head,
            },
        )
    }
}
