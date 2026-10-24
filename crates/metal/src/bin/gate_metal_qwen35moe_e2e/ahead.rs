//! `h_prompt`: the prompt loop's known-id run-ahead
//! ([`StepMode::Ahead`](bloomery_metal::arch::qwen3moe::session::StepMode)),
//! held to the host-fed steps it replaces over the d1k set's prompt.
//!
//! - the counts: an ahead step encodes the head's group and dispatch less —
//!   its barriers whole, the head closing none — and the prompt's last row,
//!   host-fed, the full token.
//! - the prompt: the ids through the target leave the same position, pick,
//!   logits row bits, delta lanes, key and value rows and greedy picks as
//!   the same ids stepped host-fed one at a time, on a session whose
//!   checkpoints the prompt crosses (every second position) and one it does
//!   not; `StepStats` shows the ids ahead all kept, their count the loop's
//!   own rule (mirrored below for the expectation).
//! - the checkpoint: the token ahead a mark's take waits out is dropped —
//!   the rule the cut, fault and reset paths share — and the prompt on from
//!   the mark, and a cut back to it, match the host-fed run: the point the
//!   take left is the state the mark names.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::count::{Layer, LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Node, Session, StepMode, StepStats, Stepped};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use runtime::seqstate;
use runtime::{Out, Target, Want};

use super::sets::{self, Set};
use super::{CTX, says};

/// The prompt's ids: the d1k set's first `P`, whose second-to-last is not 0
/// (the stale record slot a headless last row would read back instead of a
/// pick).
const P: usize = 16;

/// The crossing session's checkpoint spacing: a mark every second position.
const EVERY: u32 = 2;

/// The other session's spacing: the prompt crosses no inner mark of it.
const WIDE: u32 = 256;

/// The greedy picks compared after each prompt.
const GREEDY: usize = 8;

/// The ahead steps before the checkpoint the discard arm takes.
const LEAD: usize = 3;

/// The f32s as their bits.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// What a prompt left: its position, pick and logits row, each delta layer's
/// lane the next token reads, every attention layer's key and value rows the
/// prompt wrote, and the greedy picks that follow.
struct After {
    pos: u32,
    pick: u32,
    logits: Vec<u32>,
    lanes: Vec<Vec<u32>>,
    kv: Vec<Vec<u32>>,
    picks: Vec<u32>,
}

/// `s`'s state at the prompt's end, `pick` its last row's and `logits` that
/// row's bits, with `GREEDY` greedy picks taken after it.
fn after(s: &mut Session, pick: u32, logits: Vec<u32>) -> Result<After, String> {
    let pos = s.pos();
    let probe = s.probe().map_err(says)?;
    let mut lanes = Vec::new();
    let mut kv = Vec::new();
    for (l, layer) in s.token_layers().iter().enumerate() {
        match layer {
            Layer::Delta { .. } => {
                lanes.push(bits(&probe.read(Node::DeltaState(l)).map_err(says)?));
            }
            Layer::Attention { .. } => {
                for row in 0..pos {
                    kv.push(bits(&probe.read(Node::KeyRow(l, row)).map_err(says)?));
                    kv.push(bits(&probe.read(Node::ValueRow(l, row)).map_err(says)?));
                }
            }
        }
    }
    let mut id = pick;
    let mut picks = Vec::with_capacity(GREEDY);
    for _ in 0..GREEDY {
        id = Target::step(s, id, Want::Argmax).map_err(says)?.argmax();
        picks.push(id);
    }
    Ok(After {
        pos,
        pick,
        logits,
        lanes,
        kv,
        picks,
    })
}

/// The first way `a` and `b` differ, by name.
fn differs(what: &str, a: &After, b: &After) -> Option<String> {
    if a.pos != b.pos {
        return Some(format!("{what}: the positions {} and {}", a.pos, b.pos));
    }
    if a.pick != b.pick {
        return Some(format!("{what}: the picks {} and {}", a.pick, b.pick));
    }
    if a.logits != b.logits {
        return Some(format!("{what}: the logits rows differ"));
    }
    if let Some(l) = a.lanes.iter().zip(&b.lanes).position(|(x, y)| x != y) {
        return Some(format!("{what}: delta layer {l}'s lane differs"));
    }
    if let Some(at) = a.kv.iter().zip(&b.kv).position(|(x, y)| x != y) {
        return Some(format!("{what}: cache row pair {at} differs"));
    }
    if a.picks != b.picks {
        return Some(format!("{what}: the greedy picks part"));
    }
    None
}

/// The prompt host-fed, one step each, reset first: the reference.
fn host_fed(s: &mut Session, ids: &[u32]) -> Result<After, String> {
    s.reset().map_err(says)?;
    let mut pick = 0;
    for &id in ids {
        pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
    }
    let probe = s.probe().map_err(says)?;
    let logits = bits(&probe.read(Node::Logits).map_err(says)?);
    after(s, pick, logits)
}

/// The prompt through the target from where the session stands: what it
/// left.
fn ahead_on(s: &mut Session, ids: &[u32]) -> Result<After, String> {
    let (pick, logits) = match Target::prompt(s, ids, Want::Logits).map_err(says)? {
        Out::Logits { argmax, row } => (argmax, bits(row)),
        Out::Argmax(_) => return Err("a Want::Logits prompt returned the argmax alone".into()),
    };
    after(s, pick, logits)
}

/// The prompt through the target, reset first: what it left and the stats it
/// moved — the prompt's own, the greedy tail `after` takes not among them.
fn ahead(s: &mut Session, ids: &[u32]) -> Result<(After, StepStats), String> {
    s.reset().map_err(says)?;
    let before = s.stats();
    let (pick, logits) = match Target::prompt(s, ids, Want::Logits).map_err(says)? {
        Out::Logits { argmax, row } => (argmax, bits(row)),
        Out::Argmax(_) => return Err("a Want::Logits prompt returned the argmax alone".into()),
    };
    let moved = super::chain::moved(s, before);
    Ok((after(s, pick, logits)?, moved))
}

/// The tokens the prompt loop commits ahead over `ids` from `from` between
/// `marks`: its own rule, mirrored — every id but the last of each span, and
/// but the row before the prompt's last, whose next runs the head.
fn ahead_calls(ids: &[u32], marks: &[u32], from: u32) -> u64 {
    let last = ids.len() - 1;
    let mut at = from;
    let mut n = 0;
    for &m in marks {
        for i in (at - from) as usize..(m - from) as usize {
            if i + 1 < (m - from) as usize && i + 1 != last {
                n += 1;
            }
        }
        at = m;
    }
    n
}

/// The stats the ahead prompt owes: `n` tokens ahead, all kept, none dropped
/// and none at the cache's last row.
fn stats_are(moved: &StepStats, n: u64, what: &str) -> Result<(), String> {
    if moved.ahead != n || moved.kept != n || moved.discarded != 0 || moved.last_row != 0 {
        return Err(format!(
            "{what}: {} tokens ahead, {} kept, {} dropped, {} at the last row; {n} ahead, all \
             kept, none dropped and none at the last row are expected",
            moved.ahead, moved.kept, moved.discarded, moved.last_row
        ));
    }
    Ok(())
}

/// One step's encoded count against the expectation.
fn count_is(
    stepped: &Stepped,
    groups: usize,
    dispatches: usize,
    barriers: usize,
    what: &str,
) -> Result<(), String> {
    let c = stepped.count;
    if c.groups != groups || c.dispatches != dispatches || c.barriers != barriers {
        return Err(format!(
            "{what} encoded {} groups, {} dispatches and {} barriers; {groups}, {dispatches}              and {barriers} are expected",
            c.groups, c.dispatches, c.barriers
        ));
    }
    Ok(())
}

/// The ahead steps' counts: a headless token is the head's group and
/// dispatch short, its barriers whole — the head closes none — and the
/// host-fed step after them the full token.
fn counts(s: &mut Session, ids: &[u32]) -> Result<String, String> {
    let full = token_count(s.token_layers(), LiveKeys::Within64, Ticket::Folded);
    // The head is one group and one dispatch and closes no barrier, so the
    // headless token's barriers are the full token's: one a group but the
    // head's.
    let barriers = (full.groups - 1) as usize;
    let groups = (full.groups - 1) as usize;
    let dispatches = (full.dispatches - 1) as usize;
    let (full_groups, full_dispatches) = (full.groups as usize, full.dispatches as usize);
    s.reset().map_err(says)?;
    for i in 0..LEAD {
        let stepped = s
            .step(
                ids[i],
                StepMode::Ahead {
                    next: Some(ids[i + 1]),
                },
            )
            .map_err(says)?;
        count_is(
            &stepped,
            groups,
            dispatches,
            barriers,
            &format!("ahead step {i}"),
        )?;
    }
    let stepped = s
        .step(ids[LEAD], StepMode::Ahead { next: None })
        .map_err(says)?;
    count_is(
        &stepped,
        groups,
        dispatches,
        barriers,
        "an ahead step with no next",
    )?;
    let stepped = s.step(ids[LEAD + 1], StepMode::HostFed).map_err(says)?;
    count_is(
        &stepped,
        full_groups,
        full_dispatches,
        barriers,
        "the host-fed step",
    )?;
    Ok(format!(
        "an ahead step encodes {groups} groups and {dispatches} dispatches, the head's own \
         skipped, its barriers whole; the host-fed step {full_groups} and {full_dispatches}"
    ))
}

/// A checkpoint with a token ahead: the take waits the token out and drops
/// it, and the prompt on from the mark, and a cut back to it, match the
/// host-fed run.
fn checkpoint_ahead(s: &mut Session, ids: &[u32], want: &After) -> Result<String, String> {
    let at = LEAD as u32;
    s.reset().map_err(says)?;
    let before = s.stats();
    for i in 0..LEAD {
        s.step(
            ids[i],
            StepMode::Ahead {
                next: Some(ids[i + 1]),
            },
        )
        .map_err(says)?;
    }
    if s.pos() != at {
        return Err(format!(
            "{} ahead steps left the session at {}; {at} is expected",
            LEAD,
            s.pos()
        ));
    }
    Session::checkpoint(s).map_err(says)?;
    let moved = super::chain::moved(s, before);
    if moved.ahead != LEAD as u64
        || moved.kept != (LEAD - 1) as u64
        || moved.discarded != 1
        || moved.last_row != 0
    {
        return Err(format!(
            "the mark's take with a token ahead: {} ahead, {} kept, {} dropped; {LEAD} ahead, \
             {} kept and 1 dropped are expected",
            moved.ahead,
            moved.kept,
            moved.discarded,
            LEAD - 1
        ));
    }
    // On from the mark, then a cut back to it and again: both end where the
    // host-fed run ended, with its greedy picks.
    let on = ahead_on(s, &ids[at as usize..])?;
    if let Some(d) = differs("the prompt on from the mark", &on, want) {
        return Err(d);
    }
    Target::cut(s, at).map_err(says)?;
    let back = ahead_on(s, &ids[at as usize..])?;
    if let Some(d) = differs("the prompt after a cut back to the mark", &back, want) {
        return Err(d);
    }
    Ok(
        "the mark's take dropped the token ahead and left the point the mark names: the prompt \
         on from it, and after a cut back to it, matched the host-fed run"
            .to_string(),
    )
}

pub fn h_prompt(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
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
    if ids[P - 2] == 0 {
        return Err(
            "the set's second-to-last id is 0; the clause reads one a stale record slot cannot \
             stand in for a pick"
                .into(),
        );
    }
    let mut wide = Session::open_with(device, library, path, CTX, WIDE).map_err(says)?;
    let mut near = Session::open_with(device, library, path, CTX, EVERY).map_err(says)?;

    let reference = host_fed(&mut wide, &ids)?;
    let counted = counts(&mut wide, &ids)?;

    let (got, moved) = ahead(&mut wide, &ids)?;
    if let Some(d) = differs("the ahead prompt", &got, &reference) {
        return Err(d);
    }
    stats_are(
        &moved,
        ahead_calls(&ids, &seqstate::marks(0, P as u32, WIDE), 0),
        "the ahead prompt",
    )?;

    let (got, moved) = ahead(&mut near, &ids)?;
    if let Some(d) = differs("the ahead prompt across marks", &got, &reference) {
        return Err(d);
    }
    stats_are(
        &moved,
        ahead_calls(&ids, &seqstate::marks(0, P as u32, EVERY), 0),
        "the ahead prompt across marks",
    )?;

    let checked = checkpoint_ahead(&mut near, &ids, &reference)?;
    Ok(Verdict::Noted(format!(
        "{counted}; the ahead prompt left the host-fed run's position, pick, logits row, {} \
         delta lanes and {} cache rows bit for bit and its {} greedy picks, with a mark crossed \
         and without; {checked}",
        reference.lanes.len(),
        reference.kv.len(),
        GREEDY
    )))
}
