//! The nemotron_h_moe gate of a checkpoint folder: Nemotron 3.5 Lightning's ModelOpt checkpoint
//! (routed experts NVFP4 planes with a tensor scale an expert, every dense projection a Q8_0 or
//! NVFP4 site, as `model_hf`'s map reads it) opened as stored and run end to end on the card
//! (`NemotronHMoeModel::open_placed` over a `bloomery_gpu::folder::Model`), where the host tier
//! runs the same experts in relu² from the folder's own bytes. The real tier only: the folder
//! is the real checkpoint. The folder is the directory `BLOOMERY_REF_MODEL` names; a path that
//! is a file, a folder of another architecture, or one whose routed stacks are not all NVFP4
//! planes ends the gate by name before the card loads. `--twin <file>` names the GGUF twin
//! (Lightning's UD-Q4_K_XL) the diagnostic reads; without it that clause prints that it compared
//! nothing.
//!
//! Three placed loads of the folder, one after another on one card, each released before the next
//! opens: `host` (every expert on the host, read from the folder in place: the reference), `split`
//! (the whole plan's card bytes less half of its routed experts', so about half of them stay on the
//! host) and `whole` (the A6000's room, where the plan holds every expert on the card and every
//! layer runs whole). The `whole` and `split` arms need the A6000's room, so the recipe takes that
//! card; another card in view ends the gate by name. The 3090's room is not the split's: the
//! design's byte count has the folder whole there, and the run prints whether the 3090's room holds
//! every expert.
//!
//! The band is the card32 gate's: the card's relu² form and the host's differ by their
//! activations' rounding and their sums' order alone, in the folder's NVFP4 planes as in the
//! 32-value stacks, so the floor is [`moe_arm::COSINE_FLOOR`] and the argmax is held wherever the
//! host arm's top-1 margin passes twice the max |Δ| [derived from that precedent, not tuned; the
//! first box run prints each cosine beside the floor].
//!
//! The clauses, each printing lines ending in PASS or FAIL (a step that raises the card's fault
//! word is a FAIL naming the fault). A mutant that moves every arm's bits alike (a scale the map
//! drops, the host's and the card's relu² together) leaves these clauses green: only
//! `folder_near_twin` sees it, as a number.
//!
//! - `folder_split_near_host` (self-consistency): the `split` plan holds an expert of some routed
//!   layer on the card and leaves one on the host, or the clause fails as vacuous. Each arm's eager
//!   and replayed walks are bit-identical, and the `split` arm's last-position logits keep the
//!   floor against the `host` arm's, the argmax the same wherever the host arm's top-1 margin
//!   passes twice the max |Δ|. Mutant (red, the card side only): the stack's per-expert tensor
//!   scales uploaded as 1.0 (`parts.rs`' part `scale`), or the card leg's relu² read as relu; the
//!   `host` arm, which reads the folder's scales in place and runs the host's relu², does not
//!   move.
//! - `folder_whole_near_host` (self-consistency): the `whole` plan holds every expert on the card,
//!   or the clause fails as vacuous; its eager and replayed walks are bit-identical and its logits
//!   keep the same floor and argmax band against the `host` arm's. Mutant (red): the same two.
//! - `folder_prompt_near_steps` (self-consistency): on the `split` and `whole` arms, the prompt
//!   path over the walk at passes of [`moe_arm::COLS`] rows and at the load's pass rows (the NVFP4
//!   GEMMs at both) keeps the same floor and argmax band against the arm's own steps. Mutant
//!   (red): the prompt route's restore over the stack's scales given 1.0 while the step keeps
//!   them, or the pass's relu² read as relu where the step reads relu².
//! - `folder_near_twin` (a diagnostic: it prints and decides nothing): the folder's last-position
//!   logits on the `whole` arm against the twin's on its own whole placed load, over the same ids.
//!   The two files round the same weights differently (NVFP4 against Q5_0, Q5_1 and Q8_0), so no
//!   band is derived: the cosine, the max |Δ| and the two argmaxes are printed. A mutant common to
//!   every arm (an expert scale the map drops, relu² read as relu in both legs) moves the cosine
//!   from its quant-noise level toward 0.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_h_moe_folder", gate::run())
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
#[allow(
    dead_code,
    reason = "the seats' folder open serves the gates too; the gate opens one folder and its model"
)]
#[path = "shared/serve_seats/folder.rs"]
mod folder;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};

    use crate::e2e::{argmax, cosine, max_delta, walk};
    use crate::moe_arm::{
        CTX_SPARE, PROMPT_LEN, Premise, Words, arm, last_logits, near_host, prompt_arm,
    };
    use crate::{folder, gate_card};
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
    use gguf::Split;
    use model::arch::nemotron_h_moe::hparams::Hparams;
    use model::arch::nemotron_h_moe::place::{self as moe_place, Experts, Forms};
    use model::placement::PlanLevers;
    use model::placement::workstation::{self, CardSpec};

    const NAME: &str = "gate_nemotron_h_moe_folder";

    /// The recipe the errors name.
    const RECIPE: &str = "gate-gpu-nemotron-h-moe-folder";

    const SPLIT: &str = "folder_split_near_host";
    const WHOLE: &str = "folder_whole_near_host";
    const PROMPT: &str = "folder_prompt_near_steps";
    const TWIN: &str = "folder_near_twin";

    /// What the split arm is called in the clauses' lines.
    const SPLIT_ARM: &str = "split";

    /// The model at `path`: the checkpoint folder a directory is (the map its config names, held to
    /// this family's architecture), or the GGUF split a file is.
    fn model_at(path: &Path) -> Result<bloomery_gpu::folder::Model, GateError> {
        if path.is_dir() {
            let f = folder::open(path, refset::arch::nemotron_h_moe::ARCH)?;
            return folder::model(f.header, f.source);
        }
        Ok(Split::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?
            .into())
    }

    /// The header of the model at `path` (a split's first shard, a folder's located header).
    fn header_at(path: &Path) -> Result<Split, GateError> {
        if path.is_dir() {
            return Ok(folder::open(path, refset::arch::nemotron_h_moe::ARCH)?.header);
        }
        Ok(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?)
    }

    /// The plan levers of the `split` arm on `spec`'s room: the card budget is the bytes the whole
    /// plan puts on the card less half of its routed experts', so the plan keeps about half the
    /// experts on the card (less the card's margin, which stays free) and leaves the rest to the
    /// host.
    fn half_room(
        inputs: &moe_place::PlanInputs,
        spec: CardSpec,
        ctx: u64,
    ) -> Result<PlanLevers, GateError> {
        let machine = workstation::placed_on(spec, inputs.model.layers, workstation::SCRATCH);
        let whole = inputs.plan_with(&machine, ctx, &PlanLevers::default(), Experts::Placed)?;
        let card = whole.cards.first().ok_or("a plan of no card")?;
        Ok(PlanLevers {
            card_budget_bytes: Some(card.bytes() - card.expert_bytes / 2).into(),
        })
    }

    /// The model a header describes, its `Hparams` and plan inputs, refused by name unless its routed
    /// up stacks all take `want`'s forms (`what` names the file in the refusal).
    fn described(
        header: &Split,
        want: Forms,
        what: &str,
    ) -> Result<(Hparams, moe_place::PlanInputs), GateError> {
        let hp =
            Hparams::read(header).map_err(|e| format!("{what}: nemotron_h_moe header: {e}"))?;
        let forms = moe_place::forms_of(header, &hp);
        if forms != want {
            return Err(format!(
                "{NAME} runs {what} whose routed stacks take {want:?}; this one's are {forms:?}"
            )
            .into());
        }
        let inputs = moe_place::read(header).map_err(|e| format!("{what}: {e}"))?;
        Ok((hp, inputs))
    }

    /// The path `--twin` names; `None` without the flag. Any other argument is refused by name.
    fn twin_arg() -> Result<Option<PathBuf>, GateError> {
        let mut args = std::env::args().skip(1);
        let mut twin = None;
        while let Some(a) = args.next() {
            match a.as_str() {
                "--twin" if twin.is_none() => {
                    twin = Some(PathBuf::from(
                        args.next().ok_or("--twin takes the twin file's path")?,
                    ));
                }
                other => {
                    return Err(format!(
                        "{NAME}: an argument it does not take: {other} (usage: {NAME} [--twin FILE])"
                    )
                    .into());
                }
            }
        }
        Ok(twin)
    }

    pub fn run() -> Result<(), GateError> {
        let path = ref_model_path()?;
        if !path.is_dir() {
            return Err(format!(
                "{NAME} runs a checkpoint folder; {} is a file: name the folder in \
                 BLOOMERY_REF_MODEL (`just {RECIPE}`)",
                path.display()
            )
            .into());
        }
        let twin_path = twin_arg()?;
        let folder_forms = Forms {
            iq2: false,
            sel32: true,
            fp4: true,
        };
        let (hp, inputs) = described(&header_at(&path)?, folder_forms, "a folder")?;
        // The twin's header is read before any card work, so a file that is not Lightning's
        // 32-value twin ends the gate here.
        let twin = twin_path
            .map(|p| -> Result<_, GateError> {
                let (thp, tinputs) = described(
                    &header_at(&p)?,
                    Forms {
                        iq2: false,
                        sel32: true,
                        fp4: false,
                    },
                    "a twin",
                )?;
                if thp.n_vocab != hp.n_vocab {
                    return Err(format!(
                        "{NAME}: the twin's vocabulary is {} ids and the folder's {}: they are not \
                         the same model",
                        thp.n_vocab, hp.n_vocab
                    )
                    .into());
                }
                Ok((p, thp, tinputs))
            })
            .transpose()?;
        let ids = walk(PROMPT_LEN, hp.n_vocab);
        let ctx = u64::try_from(PROMPT_LEN + CTX_SPARE)?;
        let split = tier::run_clause(SPLIT, Tag::SelfConsistency)?;
        let whole = tier::run_clause(WHOLE, Tag::SelfConsistency)?;
        let prompt = tier::run_clause(PROMPT, Tag::SelfConsistency)?;
        let gate = gate_card::card()?;
        if gate.name != workstation::A6000.name {
            return Err(format!(
                "{NAME}'s `whole` arm holds the whole folder on the card, which the A6000's room \
                 does by its plan; the card in view is the {}: run it with BLOOMERY_GATE_CARD=a6000",
                gate.name
            )
            .into());
        }
        let room = CardSpec {
            name: gate.name,
            ..workstation::A6000
        };
        let source = || model_at(&path);
        let words = Words::clause();
        // The 3090's room, planned and not loaded: whether it holds every expert of the folder.
        {
            let machine = workstation::placed_on(gate, inputs.model.layers, workstation::SCRATCH);
            match inputs.plan_with(&machine, ctx, &PlanLevers::default(), Experts::Placed) {
                Ok(plan) => println!(
                    "{WHOLE}: the 3090's room ({} bytes) holds every expert of the folder: {} \
                     ({} experts left on the host)",
                    gate.total_bytes,
                    plan.host.experts == 0,
                    plan.host.experts
                ),
                Err(e) => println!(
                    "{WHOLE}: the 3090's room ({} bytes) plans no load of the folder: {e}",
                    gate.total_bytes
                ),
            }
        }
        // Each arm is released before the next opens: no two share the card.
        let host_last = if split || whole {
            last_logits(arm(
                &inputs,
                (&hp, ctx),
                (SPLIT, "host", Premise::Host, words),
                (room, PlanLevers::default(), &source, &ids),
            )?)
        } else {
            None
        };
        let mut ok = true;
        let mut whole_last = None;
        for (clause, name, premise, on) in [
            (SPLIT, SPLIT_ARM, Premise::Split, split),
            (WHOLE, "whole", Premise::Whole, whole),
        ] {
            if !on && !prompt && !(twin.is_some() && premise == Premise::Whole) {
                continue;
            }
            let levers = if premise == Premise::Split {
                half_room(&inputs, room, ctx)?
            } else {
                PlanLevers::default()
            };
            let run = arm(
                &inputs,
                (&hp, ctx),
                (clause, name, premise, words),
                (room, levers, &source, &ids),
            )?;
            if on {
                ok &= near_host(
                    clause,
                    name,
                    run.as_ref().map(|(_, steps)| steps),
                    host_last.as_deref(),
                );
            }
            if premise == Premise::Whole {
                whole_last = run
                    .as_ref()
                    .and_then(|(_, steps)| steps.logits.last().cloned());
            }
            if prompt {
                ok &= prompt_arm(PROMPT, name, run, &ids)?;
            }
        }
        match (twin, whole_last) {
            (Some((p, thp, tinputs)), Some(got)) => {
                let twin_source = || model_at(&p);
                let diagnostic = Words {
                    pass: "same",
                    fail: "differ; skipped",
                };
                let run = arm(
                    &tinputs,
                    (&thp, ctx),
                    (TWIN, "twin", Premise::Whole, diagnostic),
                    (room, PlanLevers::default(), &twin_source, &ids),
                )?;
                match last_logits(run) {
                    Some(want) => println!(
                        "{TWIN}: diagnostic, no verdict (the folder's NVFP4 and the twin's 32-value \
                         types round the same weights differently): whole vs twin cosine {:.6}, \
                         max |delta| {:.4}, argmax {} vs {}",
                        cosine(&got, &want),
                        max_delta(&got, &want),
                        argmax(&got),
                        argmax(&want)
                    ),
                    None => println!("{TWIN}: diagnostic, no verdict: the twin arm did not run"),
                }
            }
            (Some(_), None) => {
                println!("{TWIN}: diagnostic, no verdict: the folder's `whole` arm did not run");
            }
            (None, _) => println!("{TWIN}: diagnostic, no verdict: no --twin, nothing compared"),
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the checkpoint folder's NVFP4 routed experts: the split and whole arms' \
             logits keep the floor and the argmax band against every expert on the host, each arm's \
             eager and replayed walks are bit-identical, and the prompt path keeps the floor \
             against each arm's steps"
        );
        Ok(())
    }
}
