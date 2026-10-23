//! `g_generate`: the generation core (`arch::qwen3moe::generate::greedy`)
//! held to the session's own steps.
//!
//! - the ids: the core over the batch set's first ids emits the ids the
//!   session's plain host-fed steps pick from them. The reference is the
//!   plain session's own steps, which no other clause pins for this prompt
//!   (`t_target` prompts with the first id alone; `a_free` holds the fixed
//!   feed to ik), and the count stop ends the run at the budget.
//! - the end-of-generation stop: an end-id set holding the id the reference
//!   picks at step `k` stops the run at `k` — the first step whose pick the
//!   set holds, none before it — with the run's ids the reference's first
//!   `k + 1`.
//! - the sink: every generated id, in order, reaches the caller's sink as
//!   it is picked — in every arm.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::generate::{Generated, greedy};
use bloomery_metal::arch::qwen3moe::session::{Session, StepMode};
use runtime::StopReason;

use super::says;
use super::sets::{self, Set};

/// The prompt's ids: the batch set's first three (the set holds five; three
/// leaves the run and every arm well inside the cache).
const PROMPT: usize = 3;

/// The tokens the runs generate, past the session's overlap.
const GREEDY: usize = 16;

/// The count-stop arm's budget, under `GREEDY` so the count ends the run
/// before the reference runs out.
const COUNT: usize = 5;

/// The reference: the session reset, `prompt` fed host-fed one step each,
/// then `n` greedy picks, each fed the pick before it.
fn plain_picks(s: &mut Session, prompt: &[u32], n: usize) -> Result<Vec<u32>, String> {
    s.reset().map_err(says)?;
    for &id in &prompt[..prompt.len() - 1] {
        s.step(id, StepMode::HostFed).map_err(says)?;
    }
    let mut id = prompt[prompt.len() - 1];
    let mut picks = Vec::with_capacity(n);
    for _ in 0..n {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        picks.push(id);
    }
    Ok(picks)
}

/// One run of the core with its sink recorded, from a reset.
fn run_core(
    s: &mut Session,
    prompt: &[u32],
    budget: usize,
    eog: &[u32],
) -> Result<(Generated, Vec<u32>), String> {
    s.reset().map_err(says)?;
    let mut seen = Vec::with_capacity(budget);
    let out = greedy(s, prompt, budget, eog, |id| seen.push(id)).map_err(|e| e.to_string())?;
    Ok((out, seen))
}

pub fn g_generate(s: &mut Session) -> Clause {
    let man = sets::open(Set::Batch)?;
    let (_, toks, _) = man
        .step()
        .map_err(|e| format!("the batch set's tokens: {e}"))?;
    let toks = toks.to_vec();
    if toks.len() < PROMPT {
        return Err(format!(
            "the batch set holds {} tokens; the clause reads at least {PROMPT}",
            toks.len()
        ));
    }
    let prompt = &toks[..PROMPT];

    // The count arm: the budget's ids, the reference's own, ended by the
    // count with the sink having seen every one.
    let want = plain_picks(s, prompt, GREEDY)?;
    let (out, seen) = run_core(s, prompt, GREEDY, &[])?;
    if let Some(at) = first_diff(&out.ids, &want) {
        return Err(format!(
            "the run's token {at} is {} where the session's own steps pick {}",
            out.ids.get(at).copied().unwrap_or(0),
            want.get(at).copied().unwrap_or(0),
        ));
    }
    if out.stop != StopReason::Length {
        return Err(format!(
            "the run of {GREEDY} ended by {:?}; length at the budget is expected",
            out.stop
        ));
    }
    if seen != out.ids {
        return Err(format!(
            "the sink saw {} ids for the run's {}; the first difference at {}",
            seen.len(),
            out.ids.len(),
            first_diff(&seen, &out.ids).unwrap_or(0)
        ));
    }

    // The count at a smaller budget: a second budget ends its own run.
    let (small, seen) = run_core(s, prompt, COUNT, &[])?;
    if small.ids != want[..COUNT] || small.stop != StopReason::Length {
        return Err(format!(
            "a budget of {COUNT} ended by {:?} with {} ids; length with the first {COUNT} of the \
             reference's is expected",
            small.stop,
            small.ids.len()
        ));
    }
    if seen != small.ids {
        return Err("the sink did not see the smaller budget's ids in order".into());
    }

    // The end-of-generation arm: the id the reference picks at step k, the
    // first step whose pick no earlier step picked, must end the run at k.
    let k = (1..want.len())
        .find(|&i| !want[..i].contains(&want[i]))
        .ok_or("every pick repeats an earlier one; no step k holds a new id")?;
    let (eog, seen) = run_core(s, prompt, GREEDY, &[want[k]])?;
    if eog.ids != want[..=k] || eog.stop != StopReason::Eog {
        return Err(format!(
            "an end-id set holding the pick at step {k} ended by {:?} after {} ids",
            eog.stop,
            eog.ids.len()
        ));
    }
    if seen != eog.ids {
        return Err("the sink did not see the end-of-generation run's ids in order".into());
    }

    Ok(Verdict::Noted(format!(
        "greedy over the batch set's first {PROMPT} ids emitted the session's own picks, the \
         count stop at {COUNT} and at {GREEDY}, the end-id stop at the step whose pick it holds \
         ({k}), and the sink saw every id in order"
    )))
}

/// The first index `a` and `b` differ at.
fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}
