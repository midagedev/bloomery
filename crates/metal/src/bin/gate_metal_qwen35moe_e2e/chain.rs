//! The run-ahead clauses: chain steps, each committing the next token before
//! it waits, held to the host-fed steps they replace.
//!
//! - `b_chain`: from one prompt, the greedy picks of chain steps to the
//!   cache's last row equal the host-fed steps' picks, the last logits bit
//!   for bit; every token ahead is kept, and the last call commits nothing
//!   ahead.
//! - `c_resume`: a broken promise. Both arms run the prompt and `RESUME_AT`
//!   greedy steps, then step an id that is not the pick, then run greedy to
//!   the last row: arm A host-fed throughout, arm B in chain steps with the
//!   broken promise host-fed, which drops the token ahead. From the broken
//!   promise on, the picks are equal, and so is every recurrent layer's store
//!   the next token reads (a delta layer's lane, a short conv layer's ring),
//!   bit for bit, after the broken promise and at the end.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::count::Layer;
use bloomery_metal::arch::qwen3moe::session::{Node, Session, StepMode, StepStats};

use super::keep;
use super::says;

/// The prompt both clauses start from after the row's seed: ids the
/// vocabulary holds. Neither clause reads a reference set; each arm is held to
/// the other.
const PROMPT_AFTER_SEED: [u32; 3] = [1000, 2000, 3000];

/// The greedy steps `c_resume` runs before the broken promise: an even
/// count, so the dropped token's lane parity is the prompt's.
const RESUME_AT: usize = 8;

/// The prompt host-fed, the session reset first; the last pick is returned.
fn prompt(s: &mut Session) -> Result<u32, String> {
    s.reset().map_err(says)?;
    let mut pick = 0;
    let seed = super::row::current().seed;
    for id in std::iter::once(seed).chain(PROMPT_AFTER_SEED) {
        pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
    }
    Ok(pick)
}

/// Greedy steps in `mode` from `id` until the cache's last row is stepped;
/// the picks, in order.
fn greedy(s: &mut Session, mut id: u32, mode: StepMode) -> Result<Vec<u32>, String> {
    let mut picks = Vec::with_capacity(s.ctx());
    while (s.pos() as usize) < s.ctx() {
        id = s.step(id, mode).map_err(says)?.id;
        picks.push(id);
    }
    Ok(picks)
}

/// `n` greedy steps in `mode` from `id`; the picks, in order.
fn greedy_n(s: &mut Session, mut id: u32, mode: StepMode, n: usize) -> Result<Vec<u32>, String> {
    let mut picks = Vec::with_capacity(n);
    for _ in 0..n {
        id = s.step(id, mode).map_err(says)?.id;
        picks.push(id);
    }
    Ok(picks)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The first index where two pick lists part, or their lengths.
fn parted(a: &[u32], b: &[u32]) -> Option<String> {
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        return Some(format!("step {i}: {} against {}", a[i], b[i]));
    }
    (a.len() != b.len()).then(|| format!("{} picks against {}", a.len(), b.len()))
}

/// The session's stats moved from `before`.
pub fn moved(s: &Session, before: StepStats) -> StepStats {
    let now = s.stats();
    StepStats {
        waits: now.waits - before.waits,
        submits: now.submits - before.submits,
        ahead: now.ahead - before.ahead,
        kept: now.kept - before.kept,
        discarded: now.discarded - before.discarded,
        last_row: now.last_row - before.last_row,
        passes: now.passes - before.passes,
    }
}

/// What the last step left: the logits, and each recurrent layer's store the
/// next token reads: a delta layer's state lane, a short conv layer's ring.
struct Snapshot {
    logits: Vec<f32>,
    lanes: Vec<(usize, Vec<f32>)>,
}

fn snapshot(s: &Session) -> Result<Snapshot, String> {
    let probe = s.probe().map_err(says)?;
    let logits = probe.read(Node::Logits).map_err(says)?;
    let lanes = s
        .token_layers()
        .iter()
        .enumerate()
        .filter(|(_, layer)| keep::is_recurrent(layer))
        .map(|(l, layer)| {
            let node = match layer {
                Layer::ShortConv { .. } => Node::ConvRing(l),
                _ => Node::DeltaState(l),
            };
            Ok((l, probe.read(node).map_err(says)?))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Snapshot { logits, lanes })
}

/// The two snapshots' first difference, by name.
fn differs(what: &str, a: &Snapshot, b: &Snapshot) -> Option<String> {
    if !bits_equal(&a.logits, &b.logits) {
        return Some(format!("{what}: the logits differ"));
    }
    a.lanes
        .iter()
        .zip(&b.lanes)
        .find(|((_, x), (_, y))| !bits_equal(x, y))
        .map(|((l, _), _)| format!("{what}: layer {l}'s store the next token reads differs"))
}

pub fn b_chain(s: &mut Session) -> Clause {
    let start = prompt(s)?;
    let host = greedy(s, start, StepMode::HostFed)?;
    let host_logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;

    let start_b = prompt(s)?;
    if start_b != start {
        return Err(format!("the prompt picked {start}, then {start_b}"));
    }
    let before = s.stats();
    let chain = greedy(s, start, StepMode::Chain)?;
    let stats = moved(s, before);
    let chain_logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    if let Some(at) = parted(&host, &chain) {
        return Err(format!("chain picks part from host-fed ones at {at}"));
    }
    if !bits_equal(&host_logits, &chain_logits) {
        return Err("the last logits differ between host-fed and chain steps".to_string());
    }
    let calls = chain.len() as u64;
    if stats.last_row != 1 || stats.ahead != calls - 1 || stats.kept != calls - 1 {
        return Err(format!(
            "{calls} chain calls: {} tokens ahead, {} kept, {} dropped, {} at the last row; \
             {} ahead, {} kept, 0 dropped and 1 at the last row are expected",
            stats.ahead,
            stats.kept,
            stats.discarded,
            stats.last_row,
            calls - 1,
            calls - 1
        ));
    }
    if stats.discarded != 0 {
        return Err(format!(
            "{} tokens ahead dropped in a greedy chain",
            stats.discarded
        ));
    }
    if stats.waits != calls {
        return Err(format!("{calls} chain calls waited {} times", stats.waits));
    }
    Ok(Verdict::Noted(format!(
        "{calls} chain calls from a {}-id prompt to the last row: picks and last logits equal \
         the host-fed steps'; {} tokens ahead, all kept; call {calls}, at row {}, took the \
         last-row rule",
        PROMPT_AFTER_SEED.len() + 1,
        stats.ahead,
        s.ctx()
    )))
}

pub fn c_resume(s: &mut Session) -> Clause {
    let vocab = u32::try_from(s.vocab()).map_err(|_| "the vocabulary does not fit u32")?;

    // Arm A: host-fed throughout.
    let start = prompt(s)?;
    let lead_a = greedy_n(s, start, StepMode::HostFed, RESUME_AT)?;
    let promised = *lead_a
        .last()
        .ok_or("no greedy step before the broken promise")?;
    let x = (promised + 1) % vocab;
    let first_a = s.step(x, StepMode::HostFed).map_err(says)?.id;
    let after_a = snapshot(s)?;
    let rest_a = greedy(s, first_a, StepMode::HostFed)?;
    let end_a = snapshot(s)?;

    // Arm B: chain steps; the broken promise host-fed.
    let start_b = prompt(s)?;
    if start_b != start {
        return Err(format!("the prompt picked {start}, then {start_b}"));
    }
    let before = s.stats();
    let lead_b = greedy_n(s, start, StepMode::Chain, RESUME_AT)?;
    if let Some(at) = parted(&lead_a, &lead_b) {
        return Err(format!(
            "before the broken promise, chain picks part at {at}"
        ));
    }
    let first_b = s.step(x, StepMode::HostFed).map_err(says)?.id;
    let after_b = snapshot(s)?;
    let rest_b = greedy(s, first_b, StepMode::Chain)?;
    let end_b = snapshot(s)?;
    let stats = moved(s, before);

    if first_a != first_b {
        return Err(format!(
            "the broken promise's id {x} picked {first_a} host-fed, {first_b} after the drop"
        ));
    }
    if after_a.lanes.is_empty() == super::row::current().recurrent {
        return Err(format!(
            "row {} says recurrent = {}; the token holds {} recurrent stores",
            super::row::current().name,
            super::row::current().recurrent,
            after_a.lanes.len()
        ));
    }
    if let Some(d) = differs("after the broken promise", &after_a, &after_b) {
        return Err(d);
    }
    if let Some(at) = parted(&rest_a, &rest_b) {
        return Err(format!("after the broken promise, picks part at {at}"));
    }
    if let Some(d) = differs("at the last row", &end_a, &end_b) {
        return Err(d);
    }
    if stats.discarded != 1 {
        return Err(format!(
            "{} tokens ahead dropped; the broken promise drops one",
            stats.discarded
        ));
    }
    Ok(Verdict::Noted(format!(
        "{RESUME_AT} chain steps, then id {x} for the promised {promised}: one token ahead \
         dropped; the next {} picks, the logits and {} recurrent stores equal the host-fed arm's \
         after the broken promise and at the last row",
        rest_a.len() + 1,
        after_a.lanes.len()
    )))
}
