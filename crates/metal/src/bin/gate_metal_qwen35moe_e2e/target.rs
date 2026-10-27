//! `t_target`: the session as the generation loop's target
//! (`runtime::Target`), held to the session's own steps.
//!
//! - greedy: `runtime::generate` over `Plain` and a token-count `Stop`,
//!   prompted with the batch set's first id, emits the ids the session's
//!   plain host-fed steps pick one at a time from that id — the reference is
//!   the plain session's own steps, which no other clause pins (`a_free`
//!   holds the fixed feed to ik, `b_chain` the chain's picks against the
//!   host-fed ones) — and every token the loop's argmax steps chain ahead is
//!   kept, the loop feeding each pick as the next id.
//! - the row: a `Want::Logits` step's row equals `Node::Logits` read through
//!   `probe` after the same step, bit for bit, and the row's own argmax (the
//!   first of its largest values, the head's tie rule) is the call's.
//! - the prompt: the batch set's ids through the target leave the same
//!   position and the same last logits bits as the same ids stepped host-fed
//!   one at a time.
//! - the refusals: a prompt of no ids, a prompt past the cache's last row, a
//!   cut between 0 and the position — each refused by name, the position
//!   unmoved (a partial prompt would move it), and the session still steps
//!   the pick a fresh replay of the same prefix steps. The rule's granted
//!   arms beside them: a cut to the position does nothing, a cut to 0
//!   resets. A file with no recurrent store keeps any position, so its cut
//!   between 0 and the position is granted instead: the position moves to it
//!   and the session steps the pick a fresh replay of the prefix up to it
//!   steps.
//! - the cut rule: `kept` (and `keepable` as its `at`) for 0, a position
//!   between, the position, and past it. A file with no recurrent store
//!   answers each with its own rule: the smaller of the ask and the position.
//! - no host allocation call in a steady target step, for either read-back
//!   (`d_count`'s counter and its first-call exception).

use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::{Node, Session, SessionError, StepMode};
use runtime::seqstate::{Kept, Why};
use runtime::{Advance, Committed, Out, PassSink, Plain, Stop, StopReason, Target, Want};

use super::says;
use super::sets::{self, Set};

/// The tokens `generate` runs: the prompt's id and the picks, past the
/// session's overlap, well inside the cache.
const GREEDY: usize = 24;

/// The tokens the row, refusal and cut clauses feed first: a position with a
/// middle to cut to. The batch set holds five.
const PREFIX: usize = 3;

/// The steady target steps each read-back's allocation count runs.
const STEADY: usize = 8;

/// A [`PassSink`] that hears nothing.
pub(super) struct Quiet;

impl PassSink<Session> for Quiet {
    type Error = SessionError;

    fn begin(&mut self, _t: &Session) -> Result<(), SessionError> {
        Ok(())
    }

    fn pass(
        &mut self,
        _t: &Session,
        _c: &Committed,
        _tokens: &[u32],
        _wall: Duration,
    ) -> Result<(), SessionError> {
        Ok(())
    }
}

/// The session reset and `ids` fed host-fed, one step each; its position.
fn fed(s: &mut Session, ids: &[u32]) -> Result<u32, String> {
    s.reset().map_err(says)?;
    for &id in ids {
        s.step(id, StepMode::HostFed).map_err(says)?;
    }
    Ok(s.pos())
}

/// The first index of `row`'s largest values.
fn first_max(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// The target's greedy loop against the session's own steps.
fn greedy(s: &mut Session, first: u32) -> Result<String, String> {
    s.reset().map_err(says)?;
    let mut plain = Plain;
    let prompt = [first];
    let picked = plain
        .prompt(s, &prompt)
        .map_err(|e| format!("the prompt's argmax: {}", says(e)))?;

    // The reference: the same id and each pick host-fed, one step each; its
    // first pick is the prompt's own.
    s.reset().map_err(says)?;
    let mut ref_picks = Vec::with_capacity(GREEDY);
    let mut id = first;
    for _ in 0..GREEDY {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        ref_picks.push(id);
    }

    // The loop over the target: the prompt again, then the passes.
    s.reset().map_err(says)?;
    let again = plain
        .prompt(s, &prompt)
        .map_err(|e| format!("the prompt's argmax, again: {}", says(e)))?;
    if again != picked {
        return Err(format!(
            "the prompt picked {picked}, then {again} after a reset"
        ));
    }
    let stop = Stop::new(GREEDY, Target::ctx(s)).map_err(|e| format!("the stop rule: {e}"))?;
    let before = s.stats();
    let outcome =
        runtime::generate(s, &mut plain, &prompt, picked, &stop, &mut Quiet).map_err(says)?;
    let moved = super::chain::moved(s, before);
    let passes = GREEDY - 1;
    if outcome.tokens != ref_picks {
        let at = outcome
            .tokens
            .iter()
            .zip(&ref_picks)
            .position(|(a, b)| a != b)
            .unwrap_or(outcome.tokens.len().min(ref_picks.len()));
        return Err(format!(
            "the loop's token {at} is {:?} where the session's own steps pick {:?}",
            outcome.tokens.get(at),
            ref_picks.get(at)
        ));
    }
    if outcome.stop != StopReason::Length || outcome.passes != passes {
        return Err(format!(
            "the loop ended by {:?} after {} passes; length after {passes} is expected",
            outcome.stop, outcome.passes
        ));
    }
    if moved.ahead != passes as u64
        || moved.kept != (passes - 1) as u64
        || moved.discarded != 0
        || moved.last_row != 0
    {
        return Err(format!(
            "{passes} argmax steps chained {} ahead, kept {}, dropped {}, {} at the last row; \
             {} ahead, {} kept, 0 dropped and none at the last row are expected",
            moved.ahead,
            moved.kept,
            moved.discarded,
            moved.last_row,
            passes,
            passes - 1
        ));
    }
    Ok(format!(
        "generate over Plain and a count stop emitted the session's own {} picks; {} argmax \
         steps chained {} ahead, all kept but the last",
        GREEDY, passes, moved.ahead
    ))
}

/// A `Want::Logits` step's row against the probe's, and the call's argmax.
fn the_row(s: &mut Session, toks: &[u32]) -> Result<String, String> {
    let prefix = &toks[..PREFIX];
    fed(s, prefix)?;
    let id = toks[PREFIX];
    let (argmax, bits) = {
        let out = Target::step(s, id, Want::Logits).map_err(says)?;
        match &out {
            Out::Logits { argmax, row } => {
                (*argmax, row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            }
            Out::Argmax(_) => return Err("a Want::Logits step returned the argmax alone".into()),
        }
    };
    let probe = s.probe().map_err(says)?;
    let node = probe.read(Node::Logits).map_err(says)?;
    if node.len() != bits.len() {
        return Err(format!(
            "a Want::Logits step returned a row of {} values; the head's holds {}",
            bits.len(),
            node.len()
        ));
    }
    let differs = bits
        .iter()
        .zip(&node)
        .filter(|(a, b)| **a != b.to_bits())
        .count();
    if differs != 0 {
        return Err(format!(
            "a Want::Logits step's row differs from Node::Logits after the same step in \
             {differs} of {} values",
            node.len()
        ));
    }
    let own = first_max(&node);
    if argmax as usize != own {
        return Err(format!(
            "the step's argmax {argmax} is not the row's own {own}"
        ));
    }
    Ok(format!(
        "a Want::Logits step's row of {} values equals Node::Logits after the same step, bit \
         for bit, and its argmax is the row's own",
        node.len()
    ))
}

/// The target's prompt against the same ids stepped.
fn the_prompt(s: &mut Session, toks: &[u32]) -> Result<String, String> {
    s.reset().map_err(says)?;
    let bits = {
        let out = Target::prompt(s, toks, Want::Logits).map_err(says)?;
        match &out {
            Out::Logits { argmax: _, row } => row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            Out::Argmax(_) => return Err("a Want::Logits prompt returned the argmax alone".into()),
        }
    };
    let pos_p = s.pos();
    let pos_s = fed(s, toks)?;
    let node = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    if pos_p != pos_s || pos_s != toks.len() as u32 {
        return Err(format!(
            "the prompt left the session at {pos_p}, the steps at {pos_s}, for {} ids",
            toks.len()
        ));
    }
    if node.len() != bits.len() || bits.iter().zip(&node).any(|(a, b)| *a != b.to_bits()) {
        return Err(
            "the prompt's last logits row differs from the stepped run's, bit for bit".into(),
        );
    }
    Ok(format!(
        "a prompt of {} ids left the session at {pos_p} with the stepped run's last logits, \
         bit for bit",
        toks.len()
    ))
}

/// That the session still steps the pick a fresh replay of `prefix` steps.
fn usable_after(s: &mut Session, prefix: &[u32], id: u32) -> Result<(), String> {
    let pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
    fed(s, prefix)?;
    let fresh = s.step(id, StepMode::HostFed).map_err(says)?.id;
    if pick != fresh {
        return Err(format!(
            "after the refusal the session stepped {pick}; a fresh replay steps {fresh}"
        ));
    }
    Ok(())
}

/// The refusals by name, and the rule's granted arms.
fn the_refusals(s: &mut Session, toks: &[u32]) -> Result<String, String> {
    let prefix = &toks[..PREFIX];
    let next = toks[PREFIX];

    // A prompt of no ids.
    fed(s, prefix)?;
    let err = match Target::prompt(s, &[], Want::Argmax) {
        Err(e) => says(e),
        Ok(_) => return Err("a prompt of no ids ran".into()),
    };
    if !err.contains("no ids") {
        return Err(format!("a prompt of no ids refused as: {err}"));
    }
    if s.pos() != PREFIX as u32 {
        return Err(format!(
            "a refused prompt moved the position to {}",
            s.pos()
        ));
    }
    usable_after(s, prefix, next)?;

    // A prompt past the cache's last row: one id more than rows remain.
    fed(s, prefix)?;
    let rows = s.ctx() - s.pos() as usize;
    let past = vec![next; rows + 1];
    let err = match Target::prompt(s, &past, Want::Argmax) {
        Err(e) => says(e),
        Ok(_) => return Err("a prompt past the cache's last row ran".into()),
    };
    if !err.contains("the cache holds") {
        return Err(format!("a prompt past the cache refused as: {err}"));
    }
    if s.pos() != PREFIX as u32 {
        return Err(format!(
            "a refused prompt moved the position to {}",
            s.pos()
        ));
    }
    usable_after(s, prefix, next)?;

    // A cut between 0 and the position. The prefix was stepped, not prompted,
    // so no checkpoint stands below it.
    fed(s, prefix)?;
    let held = s.pos();
    if super::row::current().recurrent {
        let err = match Target::cut(s, 1) {
            Err(e) => says(e),
            Ok(()) => return Err("a cut to position 1 ran".into()),
        };
        if !err.contains("no checkpoint") {
            return Err(format!("a cut to a middle position refused as: {err}"));
        }
        if s.pos() != held {
            return Err(format!("a refused cut moved the position to {}", s.pos()));
        }
        usable_after(s, prefix, next)?;
    } else {
        // No recurrent store: the caches hold every position below the cut.
        Target::cut(s, 1).map_err(|e| format!("a cut to position 1: {}", says(e)))?;
        if s.pos() != 1 {
            return Err(format!(
                "a cut to position 1 left the position at {}",
                s.pos()
            ));
        }
        usable_after(s, &prefix[..1], prefix[1])?;
    }

    // The granted arms: the position itself does nothing, 0 resets.
    fed(s, prefix)?;
    if s.pos() != held {
        return Err(format!(
            "a replay of the prefix left the session at {}",
            s.pos()
        ));
    }
    Target::cut(s, held).map_err(says)?;
    if s.pos() != held {
        return Err(format!("a cut to the position moved it to {}", s.pos()));
    }
    usable_after(s, prefix, next)?;
    Target::cut(s, 0).map_err(says)?;
    if s.pos() != 0 {
        return Err(format!("a cut to 0 left the position at {}", s.pos()));
    }
    let pick = s.step(next, StepMode::HostFed).map_err(says)?.id;
    s.reset().map_err(says)?;
    let fresh = s.step(next, StepMode::HostFed).map_err(says)?.id;
    if pick != fresh {
        return Err(format!(
            "after a cut to 0 the session stepped {pick}; a fresh session steps {fresh}"
        ));
    }
    let middle = if super::row::current().recurrent {
        "refused by name, the position unmoved, the session stepping on"
    } else {
        "granted (the file keeps any position), the session stepping a fresh replay's pick"
    };
    Ok(format!(
        "a prompt of no ids and a prompt past the cache's last row refused by name, the position \
        unmoved, the session stepping on; a cut to a middle position {middle}; a cut to the \
        position did nothing and a cut to 0 reset to step a fresh session's pick"
    ))
}

/// `kept` and `keepable` for the rule's four asks.
fn the_rule(s: &mut Session, toks: &[u32]) -> Result<String, String> {
    fed(s, &toks[..PREFIX])?;
    let held = s.pos();
    // With a recurrent store the ledger answers: nothing for 0, nothing below
    // the first checkpoint, the held positions from the position up. With
    // none, the file's own rule keeps the smaller of the ask and the position.
    let recurrent = super::row::current().recurrent;
    let kept = |asked: u32, at: u32, why: Why| Kept {
        asked,
        held,
        at: if recurrent { at } else { asked.min(held) },
        why: if recurrent { why } else { Why::Rule },
    };
    let cases = [
        (0, kept(0, 0, Why::Empty)),
        (1, kept(1, 0, Why::Missed { lost: None })),
        (held, kept(held, held, Why::Current)),
        (held + 3, kept(held + 3, held, Why::Current)),
    ];
    for (n, want) in cases {
        let got = Target::kept(s, n);
        if got != want {
            return Err(format!("kept({n}) says {got:?}; {want:?} is expected"));
        }
        if Target::keepable(s, n) != want.at {
            return Err(format!(
                "keepable({n}) says {}; kept's {} is expected",
                Target::keepable(s, n),
                want.at
            ));
        }
    }
    Ok(if recurrent {
        format!(
            "kept for 0 (empty), a middle position (no checkpoint), {held} and past it (current), \
             keepable each one's at"
        )
    } else {
        format!(
            "kept for 0, a middle position, {held} and past it is the smaller of the ask and \
             {held} (the file's rule), keepable each one's at"
        )
    })
}

/// No host allocation call in a steady target step, either read-back.
fn no_alloc(s: &mut Session, seed: u32) -> Result<String, String> {
    s.reset().map_err(says)?;
    // The first call may warm the driver's paths; a later one is steady.
    let mut id = Target::step(s, seed, Want::Argmax).map_err(says)?.argmax();
    for call in 0..STEADY {
        let calls0 = super::HOST_ALLOC_CALLS.load(Relaxed);
        let out = Target::step(s, id, Want::Argmax).map_err(says)?;
        let calls = super::HOST_ALLOC_CALLS.load(Relaxed) - calls0;
        id = out.argmax();
        if calls != 0 {
            return Err(format!(
                "an argmax target step made {calls} host allocation calls (call {call}); a \\
                 steady step makes none"
            ));
        }
    }
    let mut id = match Target::step(s, id, Want::Logits).map_err(says)? {
        Out::Logits { argmax, .. } => argmax,
        Out::Argmax(_) => return Err("a Want::Logits step returned the argmax alone".into()),
    };
    for call in 0..STEADY {
        let calls0 = super::HOST_ALLOC_CALLS.load(Relaxed);
        let out = Target::step(s, id, Want::Logits).map_err(says)?;
        let calls = super::HOST_ALLOC_CALLS.load(Relaxed) - calls0;
        match &out {
            Out::Logits { argmax, .. } => id = *argmax,
            Out::Argmax(_) => return Err("a Want::Logits step returned the argmax alone".into()),
        }
        if calls != 0 {
            return Err(format!(
                "a logits target step made {calls} host allocation calls (call {call}); a \
                 steady step makes none"
            ));
        }
    }
    Ok(format!(
        "no host allocation call in {STEADY} steady argmax and {STEADY} steady logits steps"
    ))
}

pub fn t_target(s: &mut Session) -> Clause {
    let man = sets::open(Set::Batch)?;
    let (_, toks, _) = man
        .step()
        .map_err(|e| format!("the batch set's tokens: {e}"))?;
    let toks = toks.to_vec();
    let first = *toks.first().ok_or("the batch set holds no token")?;
    if toks.len() <= PREFIX {
        return Err(format!(
            "the batch set holds {} tokens; the clause reads at least {}",
            toks.len(),
            PREFIX + 1
        ));
    }
    let parts = [
        greedy(s, first)?,
        the_row(s, &toks)?,
        the_prompt(s, &toks)?,
        the_refusals(s, &toks)?,
        the_rule(s, &toks)?,
        no_alloc(s, first)?,
    ];
    Ok(Verdict::Noted(parts.join("; ")))
}
