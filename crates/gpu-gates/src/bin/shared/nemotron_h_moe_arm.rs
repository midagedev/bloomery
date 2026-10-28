//! The arm harness of the nemotron_h_moe card gates (`gate_nemotron_h_moe_card32`,
//! `gate_nemotron_h_moe_folder`): one placed load of a model on one card, its eager walk and its
//! replay, and the two readings every arm gets, its last logits against the host arm's
//! ([`near_host`]) and its prompt path against its own steps ([`prompt_arm`]). A gate includes it
//! by `#[path]` (as `shared/e2e.rs` is) and holds what differs: where the model comes from, the
//! plan's levers, the cards, and its own clauses.

use crate::e2e::{Run, argmax, cosine, fresh, max_delta, or_fault, same, second};
use bloomery_gpu::NemotronHMoeModel;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::{GateError, verdict};
use model::arch::nemotron_h_moe::hparams::Hparams;
use model::arch::nemotron_h_moe::place::{self as moe_place, Experts};
use model::placement::PlanLevers;
use model::placement::workstation::{self, CardSpec};

pub type Model = NemotronHMoeModel;

/// The cosine the card arms' logits keep against the host arm's, and the prompt path's against
/// the steps': the e2e gate's floor.
pub const COSINE_FLOOR: f64 = 0.999;

/// Tokens of the walk every clause runs.
pub const PROMPT_LEN: usize = 16;

/// Rows of the K/V planes a key head holds beyond the walk.
pub const CTX_SPARE: usize = 16;

/// The prompt path's short passes: the multi-column gemvs' run, below which the IQ2_XXS stacks
/// take their `_sel` fallback and the 32-value and NVFP4 ones still their GEMMs.
pub const COLS: usize = 8;

/// No lever is read: the arms plan and load by their own terms alone, so a lever left in the
/// runner's environment cannot move them.
const HOST: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
    populate: false,
    lock: false,
    card_dontneed: false,
    r8: false,
    route_trace: false,
};

/// Where an arm's model comes from: a fresh [`bloomery_gpu::folder::Model`] each call, a load
/// consuming it.
pub type Source<'a> = &'a dyn Fn() -> Result<bloomery_gpu::folder::Model, GateError>;

/// What an arm's plan must hold for its clause to read anything: the `Split` arm a layer with
/// experts on both sides, the `Whole` arm every expert on the card.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Premise {
    Host,
    Split,
    Whole,
}

/// The words an arm's lines are written in.
#[derive(Clone, Copy)]
pub struct Words<'a> {
    /// What the premise and bits lines end in when the arm holds.
    pub pass: &'a str,
    /// What they end in when it does not.
    pub fail: &'a str,
}

impl Words<'_> {
    /// A clause's words: its verdict.
    pub fn clause() -> Self {
        Words {
            pass: verdict(true),
            fail: verdict(false),
        }
    }
}

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

/// One arm: the plan of `premise` on `spec`'s room under `levers`, its placed load of the model
/// `source` gives, the eager walk and its replay. `None` after a line naming the premise the plan
/// misses, the fault a launch raised or the bits that differ; `words` are the words the premise
/// and bits lines are written in: a clause's verdict, or a diagnostic's plain words.
pub fn arm(
    inputs: &moe_place::PlanInputs,
    (hp, ctx): (&Hparams, u64),
    (clause, name, premise, words): (&str, &str, Premise, Words<'_>),
    (spec, levers, source, ids): (CardSpec, PlanLevers, Source<'_>, &[u32]),
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
    let plan = inputs.plan_with(&machine, ctx, &levers, experts)?;
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
             both sides for the split arm, every expert on the card for `whole`) {}",
            words.fail
        );
        return Ok(None);
    }
    let mut m = Model::open_placed(source()?, &plan, 0, HOST)?;
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
        if bits { words.pass } else { words.fail }
    );
    Ok(bits.then_some((m, replay)))
}

/// The last logits of a finished arm, its model released first: no two arms share the card.
pub fn last_logits(run: Option<(Model, Run)>) -> Option<Vec<f32>> {
    run.and_then(|(m, steps)| {
        drop(m);
        steps.logits.last().cloned()
    })
}

/// The `near` reading of arm `name`'s `steps` against the host arm's last logits `host`, and the
/// clause's FAIL line when it does not hold; an arm or a host arm that did not run fails.
pub fn near_host(clause: &str, name: &str, steps: Option<&Run>, host: Option<&[f32]>) -> bool {
    let pass = match (steps, host) {
        (Some(steps), Some(want)) => steps
            .logits
            .last()
            .is_some_and(|got| near(clause, &format!("{name} vs host"), got, want)),
        _ => false,
    };
    if !pass {
        println!("clause {clause} FAIL");
    }
    pass
}

/// The prompt clause on arm `name`'s `run` over `ids`: the prompt path at passes of [`COLS`] rows
/// and at the load's pass rows keeps the floor and the argmax band against the arm's own steps. The
/// model is released at the end; an arm that did not run fails.
pub fn prompt_arm(
    clause: &str,
    name: &str,
    run: Option<(Model, Run)>,
    ids: &[u32],
) -> Result<bool, GateError> {
    let pass = match run {
        Some((mut m, steps)) => prompt_near_steps(clause, name, &mut m, ids, &steps)?,
        None => false,
    };
    if !pass {
        println!("clause {clause} FAIL (arm {name})");
    }
    Ok(pass)
}

/// The prompt clause on arm `name`'s model `m`, its steps `steps` over `ids`.
fn prompt_near_steps(
    clause: &str,
    name: &str,
    m: &mut Model,
    ids: &[u32],
    steps: &Run,
) -> Result<bool, GateError> {
    let want = steps.logits.last().ok_or("an arm's walk is empty")?;
    let cap = m.pass_rows()?;
    let mut pass = true;
    for rows in [COLS.min(cap), cap] {
        m.reset()?;
        let Some(_) = or_fault(clause, "the prompt", m.prefill_with(ids, rows))? else {
            pass = false;
            continue;
        };
        let got = m.logits()?;
        pass &= near(
            clause,
            &format!("arm {name}, passes of {rows} rows"),
            &got,
            want,
        );
    }
    Ok(pass)
}
