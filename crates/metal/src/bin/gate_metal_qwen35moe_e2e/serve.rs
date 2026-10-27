//! `s_serve`: the server's engine (`bloomery_metal::server::TargetEngine`)
//! over a session of its own, held to the gate's session's host-fed steps.
//!
//! - greedy: `prefill` of the prompt less its last id, then `next` of the
//!   last and of each pick with no row asked, emits the picks the gate's
//!   session steps host-fed one at a time from the same ids — what a greedy
//!   request of `bloomery-serve-metal` returns.
//! - the row: a `next` that asks for the row (a sampled request) writes the
//!   gate's session's logits row after the same steps, bit for bit, and
//!   returns that step's pick.
//! - the keep rule: the position is kept, the prefill's end is a checkpoint,
//!   and a position below that point is not (the reuse note names why); a
//!   cut to the position continues the sequence; a cut to 0 starts it again,
//!   and the same prompt picks the same id. A file with no recurrent store
//!   keeps every position up to the held one and has nothing to note.

use std::path::Path;
use std::sync::Arc;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::{Session, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::server::TargetEngine;
use serve::vocab::Vocab;
use serve::{Engine, EngineProps};

use super::sets::{self, Set};
use super::{CTX, says};

/// The prompt's ids, from the batch set: a position with one below it.
const PROMPT: usize = 3;

/// The picks the greedy part compares, well inside the cache.
const PICKS: usize = 16;

/// The gate session's picks after `prompt`, host-fed one step each, and its
/// logits row after the prompt's last id.
fn reference(s: &mut Session, prompt: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    s.reset().map_err(says)?;
    let mut id = 0;
    for &p in prompt {
        id = s.step(p, StepMode::HostFed).map_err(says)?.id;
    }
    let row = s
        .logits_row()
        .map_err(says)?
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let mut picks = vec![id];
    while picks.len() < PICKS {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
        picks.push(id);
    }
    Ok((picks, row))
}

fn engine_says(e: serve::EngineError) -> String {
    e.0
}

pub fn s_serve(s: &mut Session, device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let man = sets::open(Set::Batch)?;
    let (_, toks, _) = man
        .step()
        .map_err(|e| format!("the batch set's tokens: {e}"))?;
    let prompt = toks
        .get(..PROMPT)
        .ok_or_else(|| format!("the batch set holds {} tokens", toks.len()))?
        .to_vec();
    let (lead, last) = (&prompt[..PROMPT - 1], prompt[PROMPT - 1]);
    let (want, row) = reference(s, &prompt)?;

    let tok = tokenizer::Tokenizer::from_gguf(path).map_err(|e| e.to_string())?;
    let vocab = Arc::new(Vocab::new(tok)?);
    let session = Session::open(device, library, path, CTX).map_err(says)?;
    let head_rows = session.vocab();
    let mut e = TargetEngine::new(
        session,
        head_rows,
        vocab,
        "the gate's engine".to_owned(),
        EngineProps::default(),
    )?;

    // Greedy, as a request with no sampler runs it.
    e.prefill(lead).map_err(engine_says)?;
    let mut got = vec![e.next(last, None).map_err(engine_says)?];
    while got.len() < PICKS {
        let id = *got.last().expect("got holds the first pick");
        got.push(e.next(id, None).map_err(engine_says)?);
    }
    if got != want {
        let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!(
            "the engine's pick {at} is {} where the session's host-fed steps pick {}",
            got[at], want[at]
        ));
    }

    // The row, as a sampled request asks it.
    e.reset().map_err(engine_says)?;
    e.prefill(lead).map_err(engine_says)?;
    let mut buf = vec![f32::NAN; row.len()];
    let pick = e.next(last, Some(&mut buf)).map_err(engine_says)?;
    let differs = buf
        .iter()
        .zip(&row)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    if differs != 0 || pick != want[0] {
        return Err(format!(
            "a next that asks for the row wrote {differs} of {} values unlike the session's \
             row and picked {pick}; {} is the session's",
            row.len(),
            want[0]
        ));
    }

    // The keep rule at the prompt's end. With a recurrent store `prefill(lead)`
    // is 2 ids, so a point stands at 2 and position 1 has none; the following
    // `next` does not take one. Without one every position up to the held one
    // is kept.
    let held = PROMPT;
    let at_point = PROMPT - 1;
    let below = if super::row::current().recurrent {
        0
    } else {
        1
    };
    if e.keepable(held) != held || e.keepable(at_point) != at_point || e.keepable(1) != below {
        return Err(format!(
            "keepable({held}) is {}, keepable({at_point}) is {}, keepable(1) is {}; {below} is \
             expected for the last",
            e.keepable(held),
            e.keepable(at_point),
            e.keepable(1)
        ));
    }
    let note = e.keep_limit(1);
    let noted = if super::row::current().recurrent {
        note.as_deref().is_some_and(|n| n.contains("no-checkpoint"))
    } else {
        note.is_none()
    };
    if !noted || e.keep_limit(at_point).is_some() || e.keep_limit(held).is_some() {
        return Err(format!(
            "the reuse note at 1 is {note:?}, at {at_point} {:?}, at {held} {:?}",
            e.keep_limit(at_point),
            e.keep_limit(held)
        ));
    }
    e.cut(held).map_err(engine_says)?;
    let on = e.next(want[0], None).map_err(engine_says)?;
    e.cut(0).map_err(engine_says)?;
    e.prefill(lead).map_err(engine_says)?;
    let again = e.next(last, None).map_err(engine_says)?;
    if on != want[1] || again != want[0] {
        return Err(format!(
            "after a cut to the position the next pick is {on} ({} expected); after a cut to 0 \
             the prompt picks {again} ({} expected)",
            want[1], want[0]
        ));
    }
    Ok(Verdict::Noted(format!(
        "the engine's {PICKS} greedy picks are the session's host-fed ones; a sampled next \
         writes the session's row of {} values bit for bit; keepable {held} -> {held}, \
         {at_point} -> {at_point} (the prefill's end), 1 -> {below} ({}); a cut to the \
         position continues, a cut to 0 starts again",
        row.len(),
        if below == 0 {
            "no-checkpoint"
        } else {
            "kept, nothing to note"
        }
    )))
}
