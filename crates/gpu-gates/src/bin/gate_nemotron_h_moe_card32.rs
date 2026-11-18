//! The nemotron_h_moe gate of the card's 32-value routed experts: a file whose
//! routed up and down stacks are of a 32-value-block type (Nemotron 3.5
//! Lightning's Q5_0 ups beside Q5_1 and Q8_0 downs) runs its card
//! experts through the card leg's relu² form in the step and the 32-value
//! GEMMs in the prompt pass (`bloomery_gpu::arch::nemotron_h_moe`), where the
//! host tier runs them in relu² through its own kernels. A file none of whose routed up
//! stacks has a card form (Lightning's MXFP4_MOE: MXFP4 ups) is the gate's second kind: its
//! plan keeps every expert on the host, and the clause reads that placed load beside the host
//! arm's. The fixture tier runs the Lightning-shaped fixture (`nemotron-light`, the `nemlight`
//! directory: no latent, 128 experts of 1,856 at the top 6, the UD-Q4_K_XL file's Q5_0, Q5_1 and
//! Q8_0 stacks and Q8_0 mixers) and its `nemmx` variant (MXFP4 ups beside Q5_1 downs); the real
//! tier runs the file `BLOOMERY_REF_MODEL` names. A file whose routed up stacks are neither all
//! of a 32-value type nor all without a card form ends the gate by name before the card loads.
//!
//! Three placed loads of a 32-value file, one after another on one card, each released before the next
//! opens: `host` (every expert on the host, the path the file ran before the card read its
//! types), `card` (the gate plan's card, `gate_card`: the planner's experts on the card, the rest
//! on the host, so every layer runs the card's share beside the host's) and `whole` (the card in
//! view's own room, `gate_card::room`, where the plan holds every expert on the card and every
//! layer runs whole, the path the seat serves on a card with the room). The `whole` arm needs a
//! card whose plan holds every expert: where the plan of the card in view does not, the clauses
//! that read that arm are left by name (`tier::room_clause`, `no-card-room`) and the other arms
//! run.
//!
//! The clauses, each printing lines ending in PASS or FAIL (a step that raises the card's fault
//! word is a FAIL naming the fault):
//!
//! - `card32_split_near_host` (self-consistency): the `card` plan holds an expert of some routed
//!   layer on the card and leaves one on the host, or the clause fails as vacuous (the real
//!   tier's room splits the file; the fixture fits the card whole, so the arm caps the card's
//!   experts at half of the layers' whose stacks a card form runs, `card_experts_cap`: every
//!   such layer keeps half its experts on each side). Each arm's
//!   eager and replayed walks are bit-identical, and the `card` arm's last-position logits keep
//!   [`moe_arm::COSINE_FLOOR`] against the `host` arm's, the argmax the same wherever the host
//!   arm's top-1 margin passes twice the max |Δ|: the card's relu² and the host's differ by their
//!   activations' rounding and their sums' order alone.
//! - `card32_whole_near_host` (self-consistency): the `whole` plan holds every expert on the
//!   card (the plan is read before the arm loads, and a card whose plan keeps an expert on the host
//!   leaves the clause by name, `no-card-room`); its eager and replayed walks are bit-identical, and its
//!   logits keep the same floor and argmax band against the `host` arm's.
//! - `card32_host_only_near_host` (self-consistency): a file no card form runs: the placed plan
//!   holds no expert on the card (the coverage pin: a layer of a type with no card form keeps
//!   its experts on the host), its eager and replayed walks are bit-identical, and its
//!   last-position logits keep the same floor and argmax band against the `host` arm's.
//! - `card32_prompt_near_steps` (self-consistency): on the `card` and `whole` arms (the placed
//!   arm of a file no card form runs), the prompt
//!   path over the walk at passes of [`moe_arm::COLS`] rows and at the load's pass rows (the
//!   32-value GEMMs at both) keeps the same floor and argmax band against the arm's own steps.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_h_moe_card32", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the arms read its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the arm harness the card gates share; each takes the items it reads"
)]
#[path = "shared/nemotron_h_moe_arm.rs"]
mod moe_arm;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::walk;
    use crate::gate_card;
    use crate::moe_arm::{
        CTX_SPARE, PROMPT_LEN, Premise, Words, arm, last_logits, near_host, prompt_arm, whole_fits,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split};
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::Hparams;
    use model::arch::nemotron_h_moe::place::{self as moe_place, Forms};
    use model::placement::PlanLevers;
    use tier::Tier;

    const NAME: &str = "gate_nemotron_h_moe_card32";

    /// The recipe the errors name.
    const RECIPE: &str = "gate-gpu-nemotron-h-moe-card32";

    const SPLIT: &str = "card32_split_near_host";
    const WHOLE: &str = "card32_whole_near_host";
    const HOST_ONLY: &str = "card32_host_only_near_host";
    const PROMPT: &str = "card32_prompt_near_steps";
    /// The prompt clause's part on the `whole` arm, left where the card has no room for the arm.
    const PROMPT_WHOLE: &str = "card32_prompt_near_steps (whole arm)";

    /// What the split arm is called in the clauses' lines.
    const SPLIT_ARM: &str = "card";

    /// What the placed arm of a file no card form runs is called.
    const PLACED_ARM: &str = "placed";

    /// The two kinds of file the gate runs, by the forms its routed up stacks take.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        /// Every routed up stack of a 32-value-block type: the card holds some experts.
        Card32,
        /// No routed up stack has a card form: every expert stays on the host.
        HostOnly,
    }

    /// The kind of a file whose routed up stacks take `forms`, or the refusal naming them.
    fn kind_of(forms: Forms) -> Result<Kind, GateError> {
        match forms {
            Forms {
                iq2: false,
                sel32: true,
                fp4: false,
            } => Ok(Kind::Card32),
            Forms {
                iq2: false,
                sel32: false,
                fp4: false,
            } => Ok(Kind::HostOnly),
            _ => Err(format!(
                "{NAME} runs a file whose routed stacks are all of a 32-value-block type or none \
                 of them has a card form; this one's are {forms:?} — name such a file (Nemotron \
                 3.5 Lightning) in BLOOMERY_REF_MODEL"
            )
            .into()),
        }
    }

    /// The split arm's levers: the real tier's room splits the file by itself; the fixture
    /// fits the card whole, so the card's experts are capped at half of those a card form runs
    /// (every such layer then keeps half its experts on each side).
    fn split_levers(kind: Kind, file: &gguf::Split, hp: &Hparams) -> Result<PlanLevers, GateError> {
        let cap = match (Tier::from_env()?, kind) {
            (Tier::Fixture, Kind::Card32) => Some(moe_place::card_runs_experts(file, hp) / 2),
            _ => None,
        };
        Ok(PlanLevers {
            card_experts_cap: cap,
            ..PlanLevers::default()
        })
    }

    /// An arm's model: the file the gate runs, a split opened afresh for each load.
    fn model_file() -> Result<bloomery_gpu::folder::Model, GateError> {
        Ok(open_split(Arch::NemotronHMoe, RECIPE)?.into())
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::NemotronHMoe, RECIPE)?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h_moe header: {e}"))?;
        let kind = kind_of(moe_place::forms_of(&file, &hp))?;
        let split_plan = split_levers(kind, &file, &hp)?;
        let inputs = moe_place::read(&file)?;
        drop(file);
        let ids = walk(PROMPT_LEN, hp.n_vocab);
        let ctx = u64::try_from(PROMPT_LEN + CTX_SPARE)?;
        let (split_clause, split_arm, split_premise) = match kind {
            Kind::Card32 => (SPLIT, SPLIT_ARM, Premise::Split),
            Kind::HostOnly => (HOST_ONLY, PLACED_ARM, Premise::NoCard),
        };
        let split = tier::run_clause(split_clause, Tag::SelfConsistency)?;
        let prompt = tier::run_clause(PROMPT, Tag::SelfConsistency)?;
        let gate = gate_card::card()?;
        let room = gate_card::room()?;
        // The `whole` arm's premise, read from the plan before any load: a card whose plan keeps
        // an expert on the host leaves the clauses that read the arm. A file no card form runs
        // has no such arm.
        let (whole, fits) = match kind {
            Kind::Card32 => {
                let fits = whole_fits(&inputs, (&hp, ctx), room)?;
                println!(
                    "{WHOLE}: the plan of the card in view ({} bytes) holds every expert on the \
                     card: {fits}",
                    room.total_bytes
                );
                (tier::room_clause(WHOLE, fits)?, fits)
            }
            Kind::HostOnly => (false, false),
        };
        if kind == Kind::Card32 && prompt && !fits {
            // The prompt clause reads the `whole` arm too: that part is left by name.
            tier::room_clause(PROMPT_WHOLE, false)?;
        }
        let words = Words::clause();
        // Each arm is released before the next opens: no two share the card.
        let host_last = if split || whole {
            last_logits(arm(
                &inputs,
                (&hp, ctx),
                (split_clause, "host", Premise::Host, words),
                (gate, PlanLevers::default(), &model_file, &ids),
            )?)
        } else {
            None
        };
        let mut ok = true;
        for (clause, name, premise, spec, levers, on, prompt_on) in [
            (
                split_clause,
                split_arm,
                split_premise,
                gate,
                split_plan,
                split,
                prompt,
            ),
            (
                WHOLE,
                "whole",
                Premise::Whole,
                room,
                PlanLevers::default(),
                whole,
                prompt && fits,
            ),
        ] {
            if !on && !prompt_on {
                continue;
            }
            let run = arm(
                &inputs,
                (&hp, ctx),
                (clause, name, premise, words),
                (spec, levers, &model_file, &ids),
            )?;
            if on {
                ok &= near_host(
                    clause,
                    name,
                    run.as_ref().map(|(_, steps)| steps),
                    host_last.as_deref(),
                );
            }
            if prompt_on {
                ok &= prompt_arm(PROMPT, name, run, &ids)?;
            }
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the card's 32-value routed experts (or, for a file no card form runs, \
             the placed plan's all-host experts): the split and whole arms' logits keep the floor \
             and the argmax band against every expert on the host, each arm's eager and replayed \
             walks are bit-identical, and the prompt path keeps the floor against each arm's steps"
        );
        Ok(())
    }
}
