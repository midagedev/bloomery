//! `k_keep`: checkpoints of the delta layers' recurrent state.
//!
//! A session opened with a checkpoint every 2 positions, over the d1k set's
//! prompt (the batch set holds five tokens):
//!
//! - a prompt leaves points at exactly [`runtime::seqstate::marks`];
//! - `kept` for a point, the position between two (the point below), 0 and
//!   the position;
//! - a cut to an even point and to an odd point, then the rest of the prompt,
//!   matches a fresh prompt's logits and the next greedy picks;
//! - a cut while a token runs ahead matches that fresh prompt;
//! - a cut where no point stands is refused by name, the position unmoved,
//!   the next step the replay's;
//! - the engine keeps a prefill's end, and a second request that shares it
//!   matches a fresh run of that request;
//! - `prefill_splits` returns every mark, and a call ending at one makes it
//!   a point.
//!
//! The prompt is 16 ids. The conv ring has 11 rows, and position 4 reads the
//! slot the token at 12 writes, so a cut that skips the ring is not the fresh
//! prompt.

use std::path::Path;
use std::sync::Arc;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::Session;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::server::TargetEngine;
use runtime::seqstate::{self, Kept, Why};
use runtime::{Out, Target, Want};
use serve::vocab::Vocab;
use serve::{Engine, EngineProps, Tokenizer};

use super::sets::{self, Set};
use super::{CTX, says};

/// Ids of the prompt. 16 so a token past the cut has written a conv-ring
/// slot the cut's position reads again.
const P: usize = 16;

/// The gate's checkpoint spacing: short enough that the prompt holds points
/// of both lane parities.
const EVERY: u32 = 2;

/// Greedy picks compared after a restored prompt.
const GREEDY: usize = 8;

/// Chain steps before a cut, so a token runs ahead.
const AHEAD: usize = 3;

const MARKER: &str = "<|im_start|>";

fn engine_says(e: serve::EngineError) -> String {
    e.0
}

fn prompt_bits(s: &mut Session, ids: &[u32]) -> Result<(Vec<u32>, u32), String> {
    match Target::prompt(s, ids, Want::Logits).map_err(says)? {
        Out::Logits { argmax, row } => Ok((row.iter().map(|v| v.to_bits()).collect(), argmax)),
        Out::Argmax(_) => Err("a Want::Logits prompt returned the argmax alone".into()),
    }
}

fn greedy(s: &mut Session, mut id: u32, n: usize) -> Result<Vec<u32>, String> {
    let mut picks = Vec::with_capacity(n);
    for _ in 0..n {
        id = Target::step(s, id, Want::Argmax).map_err(says)?.argmax();
        picks.push(id);
    }
    Ok(picks)
}

fn bits_differ(got: &[u32], want: &[u32]) -> usize {
    got.iter().zip(want).filter(|(a, b)| *a != *b).count() + got.len().abs_diff(want.len())
}

fn kept_is(s: &Session, n: u32, at: u32, why: Why) -> Result<(), String> {
    let want = Kept {
        asked: n,
        held: s.pos(),
        at,
        why,
    };
    let got = Target::kept(s, n);
    if got != want {
        return Err(format!("kept({n}) says {got:?}; {want:?} is expected"));
    }
    if Target::keepable(s, n) != at {
        return Err(format!(
            "keepable({n}) says {}; kept's {at} is expected",
            Target::keepable(s, n)
        ));
    }
    Ok(())
}

/// `cut` to `n`, then the prompt's tail, against the fresh row and picks.
fn cut_then_tail(
    s: &mut Session,
    ids: &[u32],
    n: u32,
    row: &[u32],
    picks: &[u32],
) -> Result<(), String> {
    Target::cut(s, n).map_err(says)?;
    if s.pos() != n {
        return Err(format!("a cut to {n} left the session at {}", s.pos()));
    }
    let (got_row, argmax) = prompt_bits(s, &ids[n as usize..])?;
    let differs = bits_differ(&got_row, row);
    if differs != 0 {
        return Err(format!(
            "after a cut to {n} the prompt's row differs in {differs} of {} values",
            row.len()
        ));
    }
    let got = greedy(s, argmax, GREEDY)?;
    if got != picks {
        let at = got.iter().zip(picks).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!(
            "after a cut to {n} greedy pick {at} is {} where a fresh prompt picks {}",
            got[at], picks[at]
        ));
    }
    Ok(())
}

/// The file's message start, marked as the server marks it
/// ([`Vocab::with_chat_template`]), opens a second message at 3, and that
/// call's start is a point.
fn message_start(path: &Path, s: &mut Session, ids: &[u32]) -> Result<String, String> {
    let tok = tokenizer::Tokenizer::from_gguf(path).map_err(|e| e.to_string())?;
    let vocab = Vocab::new(tok)?.with_chat_template(MARKER)?;
    let start = vocab.user_start();
    let &[id] = start.as_slice() else {
        return Err(format!(
            "the file's {MARKER} marks {start:?}; one special token is expected"
        ));
    };
    let mut msg = ids.to_vec();
    msg[0] = id;
    msg[3] = id;
    s.reset().map_err(says)?;
    Target::prompt(s, &msg[..3], Want::Argmax).map_err(says)?;
    Target::prompt(s, &msg[3..], Want::Argmax).map_err(says)?;
    if !s.checkpoint_positions().contains(&3) {
        return Err(format!(
            "a second message at 3 left points {:?}",
            s.checkpoint_positions()
        ));
    }
    Ok(format!("{MARKER} is id {id}"))
}

fn engine_keeps(
    device: &MetalDevice,
    library: &Library,
    path: &Path,
    ids: &[u32],
) -> Result<(), String> {
    let tok = tokenizer::Tokenizer::from_gguf(path).map_err(|e| e.to_string())?;
    let vocab = Arc::new(Vocab::new(tok)?);
    let session = Session::open_with(device, library, path, CTX, EVERY).map_err(says)?;
    let head_rows = session.vocab();
    let mut e = TargetEngine::new(
        session,
        head_rows,
        vocab,
        "k_keep".to_owned(),
        EngineProps::default(),
    )?;
    let marks = [3usize, 7];
    let splits = Engine::prefill_splits(&e, 0, ids.len(), &marks);
    if splits != marks {
        return Err(format!("prefill_splits of {marks:?} returned {splits:?}"));
    }

    let lead = &ids[..4];
    e.prefill(lead).map_err(engine_says)?;
    let mut id = ids[4];
    let mut picks = Vec::new();
    for _ in 0..4 {
        id = e.next(id, None).map_err(engine_says)?;
        picks.push(id);
    }
    if e.keepable(lead.len()) != lead.len() {
        return Err(format!(
            "keepable({}) is {}; the prefill's end is a checkpoint",
            lead.len(),
            e.keepable(lead.len())
        ));
    }
    let req2: Vec<u32> = lead
        .iter()
        .copied()
        .chain(picks[..2].iter().copied())
        .collect();
    e.cut(lead.len()).map_err(engine_says)?;
    let rest = &req2[lead.len()..];
    e.prefill(&rest[..rest.len() - 1]).map_err(engine_says)?;
    let mut got = vec![
        e.next(*rest.last().expect("the tail holds a last id"), None)
            .map_err(engine_says)?,
    ];
    for _ in 0..3 {
        let id = *got.last().expect("got holds a pick");
        got.push(e.next(id, None).map_err(engine_says)?);
    }

    e.reset().map_err(engine_says)?;
    e.prefill(&req2[..req2.len() - 1]).map_err(engine_says)?;
    let mut want = vec![
        e.next(*req2.last().expect("request 2 holds a last id"), None)
            .map_err(engine_says)?,
    ];
    for _ in 0..3 {
        let id = *want.last().expect("want holds a pick");
        want.push(e.next(id, None).map_err(engine_says)?);
    }
    if got != want {
        let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
        return Err(format!(
            "request 2's pick {at} is {} where a fresh engine picks {}",
            got[at], want[at]
        ));
    }
    Ok(())
}

pub fn k_keep(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let man = sets::open(Set::D1k)?;
    let (_, _, prompt) = man
        .step()
        .map_err(|e| format!("the d1k set's tokens: {e}"))?;
    if prompt.len() < P + 1 {
        return Err(format!(
            "the d1k prompt holds {} tokens; {P} and one more are needed",
            prompt.len()
        ));
    }
    let ids = prompt[..P].to_vec();
    let after = prompt[P];
    let mut s = Session::open_with(device, library, path, CTX, EVERY).map_err(says)?;
    let bytes = s.checkpoint_bytes();

    let (row, argmax) = prompt_bits(&mut s, &ids)?;
    let marks = seqstate::marks(0, P as u32, EVERY);
    let points = s.checkpoint_positions();
    if points != marks {
        return Err(format!(
            "points are {points:?}; {marks:?} is marks(0, {P}, {EVERY})"
        ));
    }
    let held = s.pos();
    if held != P as u32 {
        return Err(format!(
            "the prompt left the session at {held}; {P} is expected"
        ));
    }
    kept_is(&s, 4, 4, Why::Checkpoint { lost: None })?;
    kept_is(&s, 5, 4, Why::Checkpoint { lost: None })?;
    kept_is(&s, 0, 0, Why::Empty)?;
    kept_is(&s, held, held, Why::Current)?;
    let picks = greedy(&mut s, argmax, GREEDY)?;

    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids, Want::Argmax).map_err(says)?;
    cut_then_tail(&mut s, &ids, 4, &row, &picks)?;

    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids[..5], Want::Argmax).map_err(says)?;
    Target::prompt(&mut s, &ids[5..], Want::Argmax).map_err(says)?;
    cut_then_tail(&mut s, &ids, 5, &row, &picks)?;

    s.reset().map_err(says)?;
    let mut id = Target::prompt(&mut s, &ids, Want::Argmax)
        .map_err(says)?
        .argmax();
    let at_prompt = s.pos();
    for _ in 0..AHEAD {
        id = Target::step(&mut s, id, Want::Argmax)
            .map_err(says)?
            .argmax();
    }
    let ahead = u32::try_from(AHEAD).expect("AHEAD fits u32");
    if s.pos() != at_prompt + ahead {
        return Err(format!(
            "the {AHEAD} steps ahead (last pick {id}) left the session at {}; {} was expected",
            s.pos(),
            at_prompt + ahead
        ));
    }
    cut_then_tail(&mut s, &ids, 4, &row, &picks)?;

    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids, Want::Argmax).map_err(says)?;
    let held = s.pos();
    let err = match Target::cut(&mut s, 3) {
        Err(e) => says(e),
        Ok(()) => return Err("a cut to 3, where no point stands, ran".into()),
    };
    if !err.contains("no checkpoint") {
        return Err(format!("a cut to 3 refused as: {err}"));
    }
    if s.pos() != held {
        return Err(format!("a refused cut moved the position to {}", s.pos()));
    }
    let pick = Target::step(&mut s, after, Want::Argmax)
        .map_err(says)?
        .argmax();
    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids, Want::Argmax).map_err(says)?;
    let fresh = Target::step(&mut s, after, Want::Argmax)
        .map_err(says)?
        .argmax();
    if pick != fresh {
        return Err(format!(
            "after the refused cut the session stepped {pick}; a fresh replay steps {fresh}"
        ));
    }

    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids[..3], Want::Argmax).map_err(says)?;
    Target::prompt(&mut s, &ids[3..], Want::Argmax).map_err(says)?;
    if !s.checkpoint_positions().contains(&3) {
        return Err(format!(
            "a call ending at 3 left points {:?}",
            s.checkpoint_positions()
        ));
    }
    s.reset().map_err(says)?;
    Target::prompt(&mut s, &ids, Want::Argmax).map_err(says)?;
    if s.checkpoint_positions().contains(&3) {
        return Err(format!(
            "an unsplit prompt took a point at 3: {:?}",
            s.checkpoint_positions()
        ));
    }

    engine_keeps(device, library, path, &ids)?;
    let marked = message_start(path, &mut s, &ids)?;
    Ok(Verdict::Noted(format!(
        "points {marks:?}; a cut to 4 and to 5, and a cut with a token ahead, match the fresh \
         prompt's row and {GREEDY} picks; a cut to 3 is refused; a call ending at 3 is a point; \
         the engine keeps a prefill of 4; {bytes} bytes a point; {marked}"
    )))
}
