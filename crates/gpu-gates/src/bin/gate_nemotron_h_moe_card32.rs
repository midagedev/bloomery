//! The nemotron_h_moe gate of the card's 32-value routed experts: a file whose
//! routed up and down stacks are of a 32-value-block type (Nemotron 3.5
//! Lightning's Q5_0 ups beside Q5_1 and Q8_0 downs) runs its card
//! experts through the card leg's relu² form in the step and the 32-value
//! GEMMs in the prompt pass (`bloomery_gpu::arch::nemotron_h_moe`), where the
//! host tier runs them in relu² through its own kernels. The real tier only:
//! the fixture is Bolt's, whose stacks are IQ2_XXS. The file is the one
//! `BLOOMERY_REF_MODEL` names; a file whose routed stacks are not all of a
//! 32-value type ends the gate by name before the card loads.
//!
//! Three placed loads of the file, one after another on one card, each released before the next
//! opens: `host` (every expert on the host, the path the file ran before the card read its
//! types), `card` (the gate plan's card, `gate_card`: the planner's experts on the card, the rest
//! on the host, so every layer runs the card's share beside the host's) and `whole` (the
//! A6000's room, where the plan holds every expert on the card and every layer runs whole, the
//! path the seat serves on the A6000). The `whole` arm needs the A6000's room, so the recipe
//! takes that card; another card in view ends the gate by name.
//!
//! The clauses, each printing lines ending in PASS or FAIL (a step that raises the card's fault
//! word is a FAIL naming the fault):
//!
//! - `card32_split_near_host` (self-consistency): the `card` plan holds an expert of some routed
//!   layer on the card and leaves one on the host, or the clause fails as vacuous. Each arm's
//!   eager and replayed walks are bit-identical, and the `card` arm's last-position logits keep
//!   [`moe_arm::COSINE_FLOOR`] against the `host` arm's, the argmax the same wherever the host
//!   arm's top-1 margin passes twice the max |Δ|: the card's relu² and the host's differ by their
//!   activations' rounding and their sums' order alone.
//! - `card32_whole_near_host` (self-consistency): the `whole` plan holds every expert on the
//!   card, or the clause fails as vacuous; its eager and replayed walks are bit-identical, and its
//!   logits keep the same floor and argmax band against the `host` arm's.
//! - `card32_prompt_near_steps` (self-consistency): on the `card` and `whole` arms, the prompt
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
        CTX_SPARE, PROMPT_LEN, Premise, Words, arm, last_logits, near_host, prompt_arm,
    };
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split};
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::Hparams;
    use model::arch::nemotron_h_moe::place::{self as moe_place, Forms};
    use model::placement::PlanLevers;
    use model::placement::workstation::{self, CardSpec};

    const NAME: &str = "gate_nemotron_h_moe_card32";

    /// The recipe the errors name.
    const RECIPE: &str = "gate-gpu-nemotron-h-moe-card32";

    const SPLIT: &str = "card32_split_near_host";
    const WHOLE: &str = "card32_whole_near_host";
    const PROMPT: &str = "card32_prompt_near_steps";

    /// What the split arm is called in the clauses' lines.
    const SPLIT_ARM: &str = "card";

    /// An arm's model: the file the gate runs, a split opened afresh for each load.
    fn model_file() -> Result<bloomery_gpu::folder::Model, GateError> {
        Ok(open_split(Arch::NemotronHMoe, RECIPE)?.into())
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::NemotronHMoe, RECIPE)?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h_moe header: {e}"))?;
        let forms = moe_place::forms_of(&file, &hp);
        if forms
            != (Forms {
                iq2: false,
                sel32: true,
                fp4: false,
            })
        {
            return Err(format!(
                "{NAME} runs a file whose routed stacks are all of a 32-value-block type; this \
                 one's are {forms:?} — name such a file (Nemotron 3.5 Lightning) in \
                 BLOOMERY_REF_MODEL"
            )
            .into());
        }
        let inputs = moe_place::read(&file)?;
        drop(file);
        let ids = walk(PROMPT_LEN, hp.n_vocab);
        let ctx = u64::try_from(PROMPT_LEN + CTX_SPARE)?;
        let split = tier::run_clause(SPLIT, Tag::SelfConsistency)?;
        let whole = tier::run_clause(WHOLE, Tag::SelfConsistency)?;
        let prompt = tier::run_clause(PROMPT, Tag::SelfConsistency)?;
        let gate = gate_card::card()?;
        if gate.name != workstation::A6000.name {
            return Err(format!(
                "{NAME}'s `whole` arm holds the whole file on the card, which only the A6000's \
                 room does; the card in view is the {}: run it with BLOOMERY_GATE_CARD=a6000",
                gate.name
            )
            .into());
        }
        let room = CardSpec {
            name: gate.name,
            ..workstation::A6000
        };
        let words = Words::clause();
        // Each arm is released before the next opens: no two share the card.
        let host_last = if split || whole {
            last_logits(arm(
                &inputs,
                (&hp, ctx),
                (SPLIT, "host", Premise::Host, words),
                (gate, PlanLevers::default(), &model_file, &ids),
            )?)
        } else {
            None
        };
        let mut ok = true;
        for (clause, name, premise, spec, on) in [
            (SPLIT, SPLIT_ARM, Premise::Split, gate, split),
            (WHOLE, "whole", Premise::Whole, room, whole),
        ] {
            if !on && !prompt {
                continue;
            }
            let run = arm(
                &inputs,
                (&hp, ctx),
                (clause, name, premise, words),
                (spec, PlanLevers::default(), &model_file, &ids),
            )?;
            if on {
                ok &= near_host(
                    clause,
                    name,
                    run.as_ref().map(|(_, steps)| steps),
                    host_last.as_deref(),
                );
            }
            if prompt {
                ok &= prompt_arm(PROMPT, name, run, &ids)?;
            }
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the card's 32-value routed experts: the split and whole arms' logits \
             keep the floor and the argmax band against every expert on the host, each arm's \
             eager and replayed walks are bit-identical, and the prompt path keeps the floor \
             against each arm's steps"
        );
        Ok(())
    }
}
