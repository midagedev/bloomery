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
//!   [`COSINE_FLOOR`] against the `host` arm's, the argmax the same wherever the host arm's top-1
//!   margin passes twice the max |Δ|: the card's relu² and the host's differ by their
//!   activations' rounding and their sums' order alone.
//! - `card32_whole_near_host` (self-consistency): the `whole` plan holds every expert on the
//!   card, or the clause fails as vacuous; its eager and replayed walks are bit-identical, and its
//!   logits keep the same floor and argmax band against the `host` arm's.
//! - `card32_prompt_near_steps` (self-consistency): on the `card` and `whole` arms, the prompt
//!   path over the walk at passes of [`COLS`] rows and at the load's pass rows (the 32-value
//!   GEMMs at both) keeps the same floor and argmax band against the arm's own steps.

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
mod gate {
    use crate::e2e::{Run, argmax, cosine, fresh, max_delta, or_fault, same, second, walk};
    use crate::gate_card;
    use bloomery_gpu::NemotronHMoeModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::Hparams;
    use model::arch::nemotron_h_moe::place::{self as moe_place, Experts, Forms};
    use model::placement::PlanLevers;
    use model::placement::workstation::{self, CardSpec};

    type Model = NemotronHMoeModel;

    const NAME: &str = "gate_nemotron_h_moe_card32";

    /// The recipe the errors name.
    const RECIPE: &str = "gate-gpu-nemotron-h-moe-card32";

    const SPLIT: &str = "card32_split_near_host";
    const WHOLE: &str = "card32_whole_near_host";
    const PROMPT: &str = "card32_prompt_near_steps";

    /// The cosine the card arm's logits keep against the host arm's, and the
    /// prompt path's against the steps': the e2e gate's floor.
    const COSINE_FLOOR: f64 = 0.999;

    /// Tokens of the walk both clauses run.
    const PROMPT_LEN: usize = 16;

    /// Rows of the K/V planes a key head holds beyond the walk.
    const CTX_SPARE: usize = 16;

    /// The prompt path's short passes: the multi-column gemvs' run, below which the IQ2_XXS
    /// stacks take their `_sel` fallback and the 32-value ones still their GEMMs.
    const COLS: usize = 8;

    /// No lever is read: the arms plan and load by their own terms alone, so a lever left in the
    /// runner's environment cannot move them.
    const HOST: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// `got`'s last logits against `want`'s: the floor, and the argmax wherever `want`'s top-1
    /// margin passes twice the max |Δ|. One line naming `what`.
    fn near(clause: &str, what: &str, got: &[f32], want: &[f32]) -> bool {
        let cos = cosine(got, want);
        let delta = max_delta(got, want);
        let top = argmax(want);
        let margin = f64::from(want[top as usize] - want[second(want, top) as usize]);
        let decided = margin > 2.0 * delta;
        let argmax_ok = !decided || argmax(got) == top;
        let pass = cos >= COSINE_FLOOR && argmax_ok;
        println!(
            "{clause}: {what}: cosine {cos:.6} (floor {COSINE_FLOOR}), max |delta| {delta:.4}, \
             top-1 margin {margin:.4} ({}), argmax {} vs {top} {}",
            if decided {
                "decides"
            } else {
                "within twice the delta"
            },
            argmax(got),
            verdict(pass)
        );
        pass
    }

    /// What an arm's plan must hold for its clause to read anything: the `card` arm a layer with
    /// experts on both sides, the `whole` arm every expert on the card.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Premise {
        Host,
        Split,
        Whole,
    }

    /// One arm: the plan of `premise` on `spec`'s room, its placed load, the eager walk and its
    /// replay. `None` after a line naming the premise the plan misses, the fault a launch raised
    /// or the bits that differ.
    fn arm(
        inputs: &moe_place::PlanInputs,
        (hp, ctx): (&Hparams, u64),
        (clause, name, premise): (&str, &str, Premise),
        (spec, ids): (CardSpec, &[u32]),
    ) -> Result<Option<(Model, Run)>, GateError> {
        let experts = if premise == Premise::Host {
            Experts::Host
        } else {
            Experts::Placed
        };
        // The planner's layer count: the trunk, not `block_count`, which holds the file's next-token
        // block too.
        let layers = inputs.model.layers;
        let machine = workstation::placed_on(spec, layers, workstation::SCRATCH);
        let plan = inputs.plan_with(&machine, ctx, &PlanLevers::default(), experts)?;
        let card: Vec<(usize, u64, u64)> = (0..layers)
            .filter_map(|l| Some((l, plan.n_l[l], hp.experts[l]?.n_expert as u64)))
            .collect();
        println!(
            "{clause}: arm {name} plan on {} bytes: {} experts on the host, card experts by layer \
             (layer, on card, of) {card:?}",
            spec.total_bytes, plan.host.experts
        );
        let holds = match premise {
            Premise::Host => true,
            Premise::Split => {
                card.iter().any(|&(_, on, _)| on > 0) && card.iter().any(|&(_, on, of)| on < of)
            }
            Premise::Whole => card.iter().all(|&(_, on, of)| on == of),
        };
        if !holds {
            println!(
                "{clause}: arm {name}: the plan misses the arm's premise (a layer with experts on \
                 both sides for `card`, every expert on the card for `whole`) FAIL"
            );
            return Ok(None);
        }
        let mut m = Model::open_placed(open_split(Arch::NemotronHMoe, RECIPE)?, &plan, 0, HOST)?;
        m.set_mode(StepMode::Eager);
        let eager = fresh(clause, "eager", &mut m, ids)?;
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        let replay = fresh(clause, "replay", &mut m, ids)?;
        let (Some(eager), Some(replay)) = (eager, replay) else {
            return Ok(None);
        };
        let bits = same(&eager, &replay);
        println!(
            "{clause}: arm {name}: eager and replayed walks bit-identical {bits} {}",
            verdict(bits)
        );
        Ok(bits.then_some((m, replay)))
    }

    /// The prompt clause on arm `name`'s model `m`, its steps `steps` over `ids`.
    fn prompt_near_steps(
        name: &str,
        m: &mut Model,
        ids: &[u32],
        steps: &Run,
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "card32_prompt_near_steps";
        let want = steps.logits.last().ok_or("an arm's walk is empty")?;
        let cap = m.pass_rows()?;
        let mut pass = true;
        for rows in [COLS.min(cap), cap] {
            m.reset()?;
            let Some(_) = or_fault(CLAUSE, "the prompt", m.prefill_with(ids, rows))? else {
                pass = false;
                continue;
            };
            let got = m.logits()?;
            pass &= near(
                CLAUSE,
                &format!("arm {name}, passes of {rows} rows"),
                &got,
                want,
            );
        }
        Ok(pass)
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
        // Each arm is released before the next opens: no two share the card.
        let host_last = if split || whole {
            arm(
                &inputs,
                (&hp, ctx),
                (SPLIT, "host", Premise::Host),
                (gate, &ids),
            )?
            .and_then(|(m, run)| {
                drop(m);
                run.logits.last().cloned()
            })
        } else {
            None
        };
        let mut ok = true;
        for (clause, name, premise, spec, on) in [
            (SPLIT, "card", Premise::Split, gate, split),
            (WHOLE, "whole", Premise::Whole, room, whole),
        ] {
            if !on && !prompt {
                continue;
            }
            let run = arm(&inputs, (&hp, ctx), (clause, name, premise), (spec, &ids))?;
            if on {
                let pass = match (&run, &host_last) {
                    (Some((_, steps)), Some(want)) => steps
                        .logits
                        .last()
                        .is_some_and(|got| near(clause, &format!("{name} vs host"), got, want)),
                    _ => false,
                };
                if !pass {
                    println!("clause {clause} FAIL");
                }
                ok &= pass;
            }
            if prompt {
                let pass = match run {
                    Some((mut m, steps)) => prompt_near_steps(name, &mut m, &ids, &steps)?,
                    None => false,
                };
                if !pass {
                    println!("clause {PROMPT} FAIL (arm {name})");
                }
                ok &= pass;
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
