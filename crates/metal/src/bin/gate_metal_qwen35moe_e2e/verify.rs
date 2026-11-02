//! The verify clauses: a pass of two rows (`Session::verify_rows`, through
//! `runtime::Verify`) whose row `r` is the bits of the plain step at the position
//! after `r` rows, and the commit that keeps the first rows of it by moving the
//! position. They mirror the CUDA family's `gate_qwen35moe_verify` where the Metal
//! session has the same contract; a file whose layers the pass does not serve
//! (`Session::verify_serves`: a routed FFN, a short conv layer, a sandwich
//! residual, a weight type with no M-row mat-vec) prints each of them as SKIP
//! with the pass's own reason, except `v_refused`, which holds the refusal.
//!
//! - `v_forced`: from a plain run's states, `verify` then `commit(a)` for every
//!   accepted `a` of 1 and 2, at both parities of the position, after host-fed and
//!   chain steps (the token a chain step ran ahead is dropped), with the
//!   second row the plain run's next id or a wrong one, and a chain of commits:
//!   each row's pick and logits equal the plain step's bit for bit, the
//!   verify stands two positions on and a commit moves it back with no launch, no
//!   wait and no copy, the delta lane and the keys and values below the committed
//!   position equal the plain run's, and the plain steps after it equal the plain
//!   run's picks, logits and every store (the conv ring too, which the step after
//!   the commit rewrites).
//! - `v_count`: a verify of two rows is one command buffer and one wait, as a step is,
//!   and encodes the groups, dispatches and barriers the count model gives for the
//!   file's layers (`count::verify_count`), held first to the row's own count derived on
//!   paper where it has one: the buffers and waits as the session runs the verify, the
//!   groups, dispatches and barriers in the verify's trace, the buffer the trace stamps
//!   the one.
//! - `v_segments`: the rows of a verify that straddle the one-pass flash's key
//!   ceiling are two launches of the flash, each with its own segment count: a
//!   session deep enough, verifies from `ceiling - 2` through `ceiling` keep the
//!   plain steps' logits.
//! - `v_lookup`: `runtime::Speculative` of a scripted draft (a right proposal, a
//!   wrong one, none, in turn) and of `runtime::Lookup` behind the width chooser,
//!   in both modes (`fixed`: every proposal verified; `cost`: the gate by the
//!   walls it reads), over 64 tokens pick the plain greedy run's ids; the
//!   scripted draft drives both commits and the steps between them, and the
//!   lookup's accept counts and the passes the cost gate ran closed are
//!   printed.
//! - `v_cut`: a commit then `cut` keeps the cut's contract: after a verify and its
//!   commit, a cut back to the prompt's mark or its end leaves what a fresh run
//!   of the ids leaves, and the plain steps from it are the plain run's.
//! - `v_refused`: a step, a prompt, a pass, another verify, a cut, a checkpoint,
//!   a probe and a read of the logits row before the commit are each refused by
//!   name and leave the verify committable; a commit of none, of more rows than it
//!   ran or of no verify is refused by name; a verify that does not fit the cache
//!   and, on a file the pass does not serve, any verify is refused by name and
//!   leaves the session as it was.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::count::{Layer, verify_count};
use bloomery_metal::arch::qwen3moe::generate::{greedy, greedy_lookup};
use bloomery_metal::arch::qwen3moe::session::{
    Node, Session, SessionError, StepMode, StepStats, VerifyTrace,
};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::EncodeCount;
use bloomery_metal::kernels::ONE_PASS_KEYS;
use bloomery_metal::library::Library;
use runtime::prompt::PrefillMode;
use runtime::width::Mode as WidthMode;
use runtime::{Advance, Draft, Speculative, TapNeed, Target, Verify, Want};

use super::chain::{bits_equal, moved, prompt};
use super::{keep, row, says, sets};

/// The rows of a verify: the pass's two.
const ROWS: usize = 2;

/// What every refusal of a call before the commit says.
const WAITS: &str = "waits for its commit";

/// The plain steps `v_forced` records: the arms' longest chain of commits, the
/// steps after the last one and the row the right draft peeks at.
const REF_STEPS: usize = 20;

/// The plain steps an arm runs after its last commit.
const AFTER: usize = 8;

/// A verify of `rows`, the session's refusal as a string.
fn verify(s: &mut Session, rows: [u32; ROWS]) -> Result<[u32; ROWS], String> {
    Verify::verify::<ROWS>(s, rows).map_err(says)
}

/// The session's refusal of a verify when the pass does not serve its file:
/// a clause that needs one is skipped with this reason.
fn unserved(s: &Session) -> Option<Verdict> {
    s.verify_serves()
        .err()
        .map(|e| Verdict::Skip(format!("the verify pass does not serve this file: {e}")))
}

/// What a session holds that a later token reads: each delta layer's live lane
/// and conv ring, each attention layer's keys and values below the position.
struct Stores {
    pos: u32,
    lanes: Vec<Vec<f32>>,
    rings: Vec<Vec<f32>>,
    kv: Vec<Vec<f32>>,
}

/// The session's stores at its position, after the last call settled.
fn stores(s: &Session) -> Result<Stores, String> {
    let pos = s.pos();
    let probe = s.probe().map_err(says)?;
    let (mut lanes, mut rings, mut kv) = (Vec::new(), Vec::new(), Vec::new());
    for (l, layer) in s.token_layers().iter().enumerate() {
        match layer {
            Layer::Delta { .. } => {
                lanes.push(probe.read(Node::DeltaState(l)).map_err(says)?);
                rings.push(probe.read(Node::DeltaRing(l)).map_err(says)?);
            }
            Layer::ShortConv { .. } => rings.push(probe.read(Node::ConvRing(l)).map_err(says)?),
            Layer::Attention { .. } => {
                let mut rows = Vec::new();
                for row in 0..pos {
                    rows.extend(probe.read(Node::KeyRow(l, row)).map_err(says)?);
                    rows.extend(probe.read(Node::ValueRow(l, row)).map_err(says)?);
                }
                kv.push(rows);
            }
        }
    }
    Ok(Stores {
        pos,
        lanes,
        rings,
        kv,
    })
}

/// The first way two stores differ, by name; the conv rings only when `rings`:
/// the ring slot of a row the commit took back holds the row's input until the
/// next step at its position rewrites it.
fn differs(what: &str, a: &Stores, b: &Stores, rings: bool) -> Option<String> {
    if a.pos != b.pos {
        return Some(format!("{what}: positions {} and {}", a.pos, b.pos));
    }
    let first = |name: &str, x: &[Vec<f32>], y: &[Vec<f32>]| {
        x.iter()
            .zip(y)
            .position(|(p, q)| !bits_equal(p, q))
            .map(|i| format!("{what}: store {i} of the {name}s differs"))
    };
    first("delta lane", &a.lanes, &b.lanes)
        .or_else(|| first("keys and values", &a.kv, &b.kv))
        .or_else(|| {
            rings
                .then(|| first("conv ring", &a.rings, &b.rings))
                .flatten()
        })
}

/// One plain step of the reference run.
struct RefStep {
    pick: u32,
    logits: Vec<f32>,
    after: Stores,
}

/// The plain run the arms are held to: `ids[j]` is fed at position `base + j`
/// (`base` the prompt's length), and `steps[j]` is what that step left.
struct Reference {
    base: u32,
    ids: Vec<u32>,
    steps: Vec<RefStep>,
}

/// The prompt, then `n` host-fed plain steps, each recorded.
fn reference(s: &mut Session, n: usize) -> Result<Reference, String> {
    let mut id = prompt(s)?;
    let mut run = Reference {
        base: s.pos(),
        ids: Vec::new(),
        steps: Vec::new(),
    };
    for _ in 0..n {
        let pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
        let logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
        run.steps.push(RefStep {
            pick,
            logits,
            after: stores(s)?,
        });
        run.ids.push(id);
        id = pick;
    }
    run.ids.push(id);
    Ok(run)
}

/// One verify and its commit: the second row the run's next id (`right`) or
/// another, and how many rows the commit keeps.
#[derive(Clone, Copy)]
struct Round {
    accepted: usize,
    right: bool,
}

/// A run from the prompt: `lead` plain steps in `mode`, then the rounds in
/// order, then [`AFTER`] plain steps.
struct Arm {
    name: &'static str,
    lead: usize,
    mode: StepMode,
    rounds: &'static [Round],
}

const fn round(accepted: usize, right: bool) -> Round {
    Round { accepted, right }
}

const ARMS: [Arm; 7] = [
    Arm {
        name: "even both kept",
        lead: 0,
        mode: StepMode::HostFed,
        rounds: &[round(2, true)],
    },
    Arm {
        name: "even one kept of a wrong draft",
        lead: 0,
        mode: StepMode::HostFed,
        rounds: &[round(1, false)],
    },
    Arm {
        name: "even one kept of a right draft",
        lead: 0,
        mode: StepMode::HostFed,
        rounds: &[round(1, true)],
    },
    Arm {
        name: "odd after a chain step, both kept",
        lead: 1,
        mode: StepMode::Chain,
        rounds: &[round(2, true)],
    },
    Arm {
        name: "odd after a chain step, one kept of a wrong draft",
        lead: 1,
        mode: StepMode::Chain,
        rounds: &[round(1, false)],
    },
    Arm {
        name: "odd after a host-fed step, one kept of a right draft",
        lead: 1,
        mode: StepMode::HostFed,
        rounds: &[round(1, true)],
    },
    Arm {
        name: "five verifies in a row",
        lead: 0,
        mode: StepMode::HostFed,
        rounds: &[
            round(2, true),
            round(1, false),
            round(2, true),
            round(1, true),
            round(2, true),
        ],
    },
];

/// The id that is not `id`, in the vocabulary.
fn other(id: u32, vocab: usize) -> u32 {
    let next = (id as usize + 1) % vocab;
    u32::try_from(next).expect("an id of the vocabulary fits u32")
}

/// The plain step of `id` against the reference's step `j`: the pick, the
/// logits row and, past the pick, every store the next token reads.
fn plain_against(s: &mut Session, r: &Reference, j: usize, what: &str) -> Result<(), String> {
    let pick = s.step(r.ids[j], StepMode::HostFed).map_err(says)?.id;
    let step = &r.steps[j];
    if pick != step.pick {
        return Err(format!(
            "{what}: the plain step {j} picked {pick}, the plain run's {}",
            step.pick
        ));
    }
    let logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    if !bits_equal(&logits, &step.logits) {
        return Err(format!("{what}: the plain step {j}'s logits differ"));
    }
    if let Some(d) = differs(
        &format!("{what}, plain step {j}"),
        &stores(s)?,
        &step.after,
        true,
    ) {
        return Err(d);
    }
    Ok(())
}

/// One arm against the reference; the note it adds to the verdict.
fn run_arm(s: &mut Session, r: &Reference, arm: &Arm) -> Result<String, String> {
    let vocab = s.vocab();
    let what = arm.name;
    let first = prompt(s)?;
    if first != r.ids[0] {
        return Err(format!(
            "{what}: the prompt picked {first}, then {}",
            r.ids[0]
        ));
    }
    for j in 0..arm.lead {
        let pick = s.step(r.ids[j], arm.mode).map_err(says)?.id;
        if pick != r.ids[j + 1] {
            return Err(format!("{what}: lead step {j} picked {pick}"));
        }
    }
    let mut j = arm.lead;
    let mut dropped = 0u64;
    for (n, round) in arm.rounds.iter().enumerate() {
        let at = s.pos();
        let before = s.stats();
        let draft = if round.right {
            r.ids[j + 1]
        } else {
            other(r.ids[j + 1], vocab)
        };
        let picks = verify(s, [r.ids[j], draft]).map_err(|e| format!("{what}: {e}"))?;
        if s.pos() != at + ROWS as u32 || s.verify_pending() != Some((at, ROWS)) {
            return Err(format!(
                "{what}: a verify from {at} stands at {} with {:?} pending; {} and {:?} are expected",
                s.pos(),
                s.verify_pending(),
                at + ROWS as u32,
                Some((at, ROWS))
            ));
        }
        let ran = moved(s, before);
        if ran.passes != 1 {
            return Err(format!("{what}: a verify ran {} passes", ran.passes));
        }
        dropped += ran.discarded;
        // Row 0 is the plain step at the position; row 1 is too when its id is
        // the one the plain run feeds there.
        let rows: &[(usize, usize)] = if round.right {
            &[(0, j), (1, j + 1)]
        } else {
            &[(0, j)]
        };
        for &(row, step) in rows {
            if picks[row] != r.steps[step].pick {
                return Err(format!(
                    "{what}: verify {n} row {row} picked {}, the plain step {step}'s {}",
                    picks[row], r.steps[step].pick
                ));
            }
            let got = s.verify_row_logits(row).map_err(says)?;
            if !bits_equal(got, &r.steps[step].logits) {
                return Err(format!(
                    "{what}: verify {n} row {row}'s logits differ from the plain step {step}'s"
                ));
            }
        }
        let launched = s.stats();
        Verify::commit(s, round.accepted).map_err(says)?;
        let after = moved(s, launched);
        if after.submits != 0 || after.waits != 0 || after.passes != 0 {
            return Err(format!(
                "{what}: a commit submitted {} buffers, waited {} times and ran {} passes; it \
                 launches nothing",
                after.submits, after.waits, after.passes
            ));
        }
        j += round.accepted;
        if s.pos() != at + round.accepted as u32 || s.verify_pending().is_some() {
            return Err(format!(
                "{what}: a commit of {} from {at} stands at {} with {:?} pending",
                round.accepted,
                s.pos(),
                s.verify_pending()
            ));
        }
        if let Some(d) = differs(
            &format!("{what}, after commit {n} of {}", round.accepted),
            &stores(s)?,
            &r.steps[j - 1].after,
            false,
        ) {
            return Err(d);
        }
    }
    for t in 0..AFTER {
        plain_against(s, r, j + t, what)?;
    }
    let expect = u64::from(arm.lead > 0 && arm.mode == StepMode::Chain);
    if dropped != expect {
        return Err(format!(
            "{what}: {dropped} tokens ahead dropped by the verifies; {expect} is expected"
        ));
    }
    Ok(format!(
        "{what} (from position {}, commits {:?})",
        r.base as usize + arm.lead,
        arm.rounds.iter().map(|r| r.accepted).collect::<Vec<_>>()
    ))
}

/// (a): the forced rows.
pub fn v_forced(s: &mut Session) -> Clause {
    if let Some(skip) = unserved(s) {
        return Ok(skip);
    }
    let r = reference(s, REF_STEPS)?;
    let mut notes = Vec::new();
    for arm in &ARMS {
        notes.push(run_arm(s, &r, arm)?);
    }
    Ok(Verdict::Noted(format!(
        "{} arms from a {}-id prompt: every verify row's pick and logits equal the plain \
         step's bit for bit, a commit moves the position with no submit, wait or pass, the \
         lane, keys and values below it are the plain run's, and {AFTER} plain steps after \
         the last commit equal the plain run's picks, logits and stores: {}",
        ARMS.len(),
        r.base,
        notes.join("; ")
    )))
}

/// (b): a verify's structure. From a prompt a verify of [`ROWS`] ids, as the
/// session runs it and as its trace runs it, is one command buffer and one
/// wait, and encodes the groups, dispatches and barriers the count model gives
/// for the file's layers, one launch of the flash a layer (the prompt and the
/// verifies stay inside one segment count of keys).
pub fn v_count(s: &mut Session) -> Clause {
    if let Some(skip) = unserved(s) {
        return Ok(skip);
    }
    let rows = u32::try_from(ROWS).map_err(|_| "the verify's rows fit u32")?;
    let want = verify_count(s.token_layers(), rows, 1);
    // The row's own count, derived on paper from the layer list: the count
    // model and the encoder agree below only while both are right.
    if let Some(paper) = row::current().verify
        && (want.groups, want.dispatches) != (paper.groups, paper.dispatches)
    {
        return Err(format!(
            "the count model gives {} groups and {} dispatches for a verify of the file's layers; \
            the row derives {} and {} on paper",
            want.groups, want.dispatches, paper.groups, paper.dispatches
        ));
    }
    let groups = want.groups as usize;
    let want = EncodeCount {
        groups,
        dispatches: want.dispatches as usize,
        barriers: groups - 1,
    };
    let x0 = prompt(s)?;
    let ids = [x0, other(x0, s.vocab())];
    let before = s.stats();
    verify(s, ids)?;
    buffers_and_waits("a verify", moved(s, before))?;
    Verify::commit(s, ROWS).map_err(says)?;
    let before = s.stats();
    let (mut picks, mut trace) = ([0u32; ROWS], VerifyTrace::default());
    s.verify_traced(&ids, &mut picks, &mut trace)
        .map_err(says)?;
    buffers_and_waits("a traced verify", moved(s, before))?;
    Verify::commit(s, ROWS).map_err(says)?;
    if trace.buffers.len() != 1 || trace.waits != 1 {
        return Err(format!(
            "a traced verify stamped {} buffers and the host waited {} times between its first \
            commit and its wake; one and one are expected",
            trace.buffers.len(),
            trace.waits
        ));
    }
    if trace.count != want {
        return Err(format!(
            "a verify of {ROWS} ids encoded {} groups, {} dispatches and {} barriers; the count \
            model gives {} groups, {} dispatches and {} barriers",
            trace.count.groups,
            trace.count.dispatches,
            trace.count.barriers,
            want.groups,
            want.dispatches,
            want.barriers
        ));
    }
    Ok(Verdict::Noted(format!(
        "a verify of {ROWS} ids over {} layers is one command buffer and one wait, as run and \
        traced, and encodes {} groups, {} dispatches and {} barriers",
        s.layer_count(),
        want.groups,
        want.dispatches,
        want.barriers
    )))
}

/// A verify's one command buffer, one wait and one pass, no token ahead
/// dropped: what the session's stats moved by across the call `what`.
fn buffers_and_waits(what: &str, ran: StepStats) -> Result<(), String> {
    if ran.submits != 1 || ran.waits != 1 || ran.passes != 1 || ran.discarded != 0 {
        return Err(format!(
            "{what} of {ROWS} ids committed {} command buffers, waited {} times, ran {} passes \
            and dropped {} tokens ahead; one buffer, one wait, one pass and none dropped are \
            expected",
            ran.submits, ran.waits, ran.passes, ran.discarded
        ));
    }
    Ok(())
}

/// The cache rows the deep session holds: past the one-pass flash's ceiling,
/// with room for the last verify.
const DEEP_CTX: usize = ONE_PASS_KEYS + 8;

/// The positions `v_segments` verifies from: the rows of the first hold
/// `ONE_PASS_KEYS - 1` and `ONE_PASS_KEYS` keys, the second's one and the
/// ceiling's, the third's the two past it.
const DEEP_FROM: [usize; 3] = [ONE_PASS_KEYS - 2, ONE_PASS_KEYS - 1, ONE_PASS_KEYS];

/// (a) past the one-pass flash's ceiling of keys.
pub fn v_segments(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let mut s = Session::open(device, library, path, DEEP_CTX).map_err(says)?;
    if let Some(skip) = unserved(&s) {
        return Ok(skip);
    }
    // The plain run to the last row of the last verify, the logits and picks
    // kept at the positions the verifies read.
    let last = DEEP_FROM[2] + ROWS;
    let mut id = prompt(&mut s)?;
    let mut ids = Vec::new();
    let mut seen: Vec<(usize, u32, Vec<f32>)> = Vec::new();
    while (s.pos() as usize) < last {
        let at = s.pos() as usize;
        ids.push((at, id));
        let pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
        if at >= DEEP_FROM[0] {
            let logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
            seen.push((at, pick, logits));
        }
        id = pick;
    }
    let id_at = |at: usize| ids.iter().find(|(p, _)| *p == at).map(|(_, id)| *id);
    let seen_at = |at: usize| seen.iter().find(|(p, ..)| *p == at);
    // The arm: plain to the first position, then a verify from each, one row
    // kept, the last two.
    s.reset().map_err(says)?;
    prompt(&mut s)?;
    let mut feed = ids[0].1;
    while (s.pos() as usize) < DEEP_FROM[0] {
        feed = s.step(feed, StepMode::HostFed).map_err(says)?.id;
    }
    let mut notes = Vec::new();
    for from in DEEP_FROM {
        if s.pos() as usize != from {
            return Err(format!("the arm stands at {}, not {from}", s.pos()));
        }
        let rows = [
            id_at(from).ok_or("no id at the verify's first row")?,
            id_at(from + 1).ok_or("no id at the verify's second row")?,
        ];
        let picks = verify(&mut s, rows)?;
        for (r, got) in picks.iter().enumerate() {
            let (_, pick, logits) = seen_at(from + r).ok_or("no plain row kept")?;
            if got != pick {
                return Err(format!(
                    "a verify from {from}: row {r} picked {got}, the plain step's {pick}"
                ));
            }
            if !bits_equal(s.verify_row_logits(r).map_err(says)?, logits) {
                return Err(format!(
                    "a verify from {from}: row {r}'s logits differ from the plain step's at {}",
                    from + r
                ));
            }
        }
        let kept = if from == DEEP_FROM[2] { 2 } else { 1 };
        Verify::commit(&mut s, kept).map_err(says)?;
        notes.push(format!(
            "{from}..{}: {} and {} keys",
            from + 1,
            from + 1,
            from + 2
        ));
    }
    Ok(Verdict::Noted(format!(
        "a session of {DEEP_CTX} rows, verifies from {DEEP_FROM:?} (rows of {}): each row's pick \
         and logits equal the plain step's at its position bit for bit, across the one-pass \
         flash's {ONE_PASS_KEYS}-key ceiling",
        notes.join("; ")
    )))
}

/// The draft of the clause's second part: a proposal that is the plain run's
/// next id, then a wrong one, then the next id again, then none, then the next
/// id, so a pass keeps both rows, one row and takes a step in turn.
struct Scripted<'a> {
    want: &'a [u32],
    vocab: usize,
    emitted: usize,
    pass: usize,
}

impl Draft<Session> for Scripted<'_> {
    const WIDTH: usize = 1;
    const TAPS: TapNeed = TapNeed::None;

    fn begin(&mut self, _: &Session, _: &[u32], _: u32) -> Result<(), SessionError> {
        self.emitted = 1;
        self.pass = 0;
        Ok(())
    }

    fn propose(&mut self, _: &mut Session, _: u32, out: &mut [u32]) -> Result<usize, SessionError> {
        let turn = self.pass % 5;
        self.pass += 1;
        let Some(&next) = self.want.get(self.emitted) else {
            return Ok(0);
        };
        match turn {
            3 => Ok(0),
            1 => {
                out[0] = other(next, self.vocab);
                Ok(1)
            }
            _ => {
                out[0] = next;
                Ok(1)
            }
        }
    }

    fn accept(
        &mut self,
        _: &mut Session,
        _: &[u32],
        _: &[u32],
        accepted: usize,
    ) -> Result<(), SessionError> {
        self.emitted += accepted;
        Ok(())
    }

    fn stepped(&mut self, _: &mut Session, _: u32, _: u32) -> Result<(), SessionError> {
        self.emitted += 1;
        Ok(())
    }
}

/// The ids a run asks for.
const GEN: usize = 64;

/// The cache rows the lookup clause's session holds: the longest prompt, the
/// run and a verify's last row.
const LOOKUP_CTX: usize = 256;

/// The scripted draft's run of `prompt`: its ids and the commits it made.
fn scripted_run(
    s: &mut Session,
    prompt: &[u32],
    want: &[u32],
) -> Result<(Vec<u32>, [usize; 3]), String> {
    s.reset().map_err(says)?;
    let mut spec: Speculative<Scripted, 2> = Speculative::new(Scripted {
        want,
        vocab: s.vocab(),
        emitted: 0,
        pass: 0,
    });
    let first = Advance::prompt(&mut spec, s, prompt).map_err(says)?;
    Advance::begin(&mut spec, s, prompt, first).map_err(says)?;
    let mut tokens = vec![first];
    // Both rows kept, one row of a proposal kept, passes with no proposal.
    let mut counts = [0usize; 3];
    while tokens.len() < GEN {
        let last = tokens[tokens.len() - 1];
        let c = Advance::pass(&mut spec, s, last, &mut tokens).map_err(says)?;
        match (c.proposed, c.kept) {
            (true, 2) => counts[0] += 1,
            (true, 1) => counts[1] += 1,
            (false, 1) => counts[2] += 1,
            other => return Err(format!("a pass kept {other:?}")),
        }
    }
    Ok((tokens, counts))
}

/// (b): the drafts' ids are the plain run's.
pub fn v_lookup(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let mut s = Session::open(device, library, path, LOOKUP_CTX).map_err(says)?;
    if let Some(skip) = unserved(&s) {
        return Ok(skip);
    }
    let seg = sets::prompt_ids(48)?;
    let prompts = [
        ("twice", [seg.clone(), seg.clone()].concat()),
        ("start", [seg.clone(), seg[..24].to_vec()].concat()),
        ("four", seg[..16].repeat(4)),
    ];
    let mut lines = Vec::new();
    let (mut proposals, mut accepted, mut rejected) = (0usize, 0usize, 0usize);
    let (mut script_both, mut script_one, mut script_step) = (0usize, 0usize, 0usize);
    for (name, p) in &prompts {
        s.reset().map_err(says)?;
        let plain = greedy(&mut s, p, GEN, &[], |_| {}).map_err(|e| e.to_string())?;
        if plain.ids.len() != GEN {
            return Err(format!(
                "{name}: the plain run picked {} ids",
                plain.ids.len()
            ));
        }
        let (tokens, counts) = scripted_run(&mut s, p, &plain.ids)?;
        if tokens[..GEN] != plain.ids[..] {
            let at = tokens.iter().zip(&plain.ids).position(|(a, b)| a != b);
            return Err(format!(
                "{name}: the scripted draft's ids part from the plain run's at {at:?}"
            ));
        }
        if counts.contains(&0) {
            return Err(format!(
                "{name}: the scripted draft kept {} both-row, {} one-row and {} step passes; each \
                 kind is expected",
                counts[0], counts[1], counts[2]
            ));
        }
        script_both += counts[0];
        script_one += counts[1];
        script_step += counts[2];
        // Every proposal verified, then the gate by the walls: each run's ids are the
        // plain run's, because the chooser changes which rows a pass runs, never a token.
        let mut modes = Vec::new();
        for mode in [WidthMode::Fixed, WidthMode::Cost] {
            s.reset().map_err(says)?;
            let drafted = greedy_lookup(&mut s, p, GEN, &[], mode, |_| {})
                .map_err(|e| format!("{name}: {} width: {e}", mode.word()))?;
            if drafted.ids != plain.ids {
                let at = drafted.ids.iter().zip(&plain.ids).position(|(a, b)| a != b);
                return Err(format!(
                    "{name}: the lookup's ids under the {} width part from the plain run's at \
                     {at:?} ({} ids against {})",
                    mode.word(),
                    drafted.ids.len(),
                    plain.ids.len()
                ));
            }
            let t = drafted.tally();
            if mode == WidthMode::Fixed {
                proposals += t.proposals;
                accepted += t.accepted;
                rejected += t.partial;
            }
            let closed = drafted
                .width
                .as_ref()
                .and_then(|w| w.counted.as_ref())
                .map_or_else(String::new, |c| format!(", closed {}", c.tally.closed));
            modes.push(format!(
                "{}: proposals {}, accepted {}, partial commits {}, steps {}{closed}",
                mode.word(),
                t.proposals,
                t.accepted,
                t.partial,
                t.steps
            ));
        }
        lines.push(format!("{name} ({} ids): {}", p.len(), modes.join("; ")));
    }
    // The lookup's proposals and accepts (counted under the fixed width, which verifies every
    // one) come from the model's own repeats, so a cut whose greedy run repeats one id takes no
    // proposal back; the scripted draft above holds every kind of pass on any file, and the
    // lookup holds the real draft's ids to the plain run's under both widths.
    if proposals == 0 || accepted == 0 {
        return Err(format!(
            "the lookup's passes: {proposals} proposals, {accepted} accepted, {rejected} partly \
             committed over the prompts; a proposal kept whole is expected ({})",
            lines.join("; ")
        ));
    }
    Ok(Verdict::Noted(format!(
        "{GEN} ids of each prompt equal the plain greedy run's through the scripted draft \
         ({script_both} passes kept both rows, {script_one} one row, {script_step} stepped) and \
         through the lookup under both widths, the cost gate's closed passes counted: {}",
        lines.join("; ")
    )))
}

/// The ids of the prompt `v_cut` marks and the mark inside it.
const CUT_IDS: usize = 40;
const CUT_AT: u32 = 25;

/// The plain steps after a cut.
const CUT_STEPS: usize = 6;

/// (d): a cut after a commit.
pub fn v_cut(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = sets::prompt_ids(CUT_IDS)?;
    let mut s = Session::open_with(device, library, path, ids.len() + 12, 2048).map_err(says)?;
    s.set_prefill(PrefillMode::Steps);
    if let Some(skip) = unserved(&s) {
        return Ok(skip);
    }
    let end = ids.len() as u32;
    let (logits0, x0) = keep::prompt_bits_marked(&mut s, &ids, &[CUT_AT])?;
    // A file with a recurrent store cuts only to a point: the mark, and the call's
    // end where the session keeps one. A file with none keeps any position up to
    // the cut.
    let recurrent = super::row::current().recurrent;
    let held = s.checkpoint_positions();
    if recurrent && !held.contains(&CUT_AT) {
        return Err(format!(
            "the prompt marked at {CUT_AT} left the checkpoints {held:?}: none at the mark"
        ));
    }
    let cuts: Vec<u32> = [CUT_AT, end]
        .into_iter()
        .filter(|p| !recurrent || held.contains(p))
        .collect();
    // The plain run from the prompt's end.
    let mut want = Vec::new();
    let mut id = x0;
    for _ in 0..CUT_STEPS {
        let pick = s.step(id, StepMode::HostFed).map_err(says)?.id;
        let logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
        want.push((id, pick, logits));
        id = pick;
    }
    let vocab = s.vocab();
    let mut notes = Vec::new();
    for (accepted, right) in [(2usize, true), (1, true), (1, false)] {
        for &to in &cuts {
            s.reset().map_err(says)?;
            let (logits, pick) = keep::prompt_bits_marked(&mut s, &ids, &[CUT_AT])?;
            if (&logits, pick) != (&logits0, x0) {
                return Err("the marked prompt again left another pick or logits row".into());
            }
            let draft = if right {
                want[0].1
            } else {
                other(want[0].1, vocab)
            };
            let what = format!(
                "commit {accepted} of a {} draft, cut to {to}",
                if right { "right" } else { "wrong" }
            );
            verify(&mut s, [x0, draft])?;
            Verify::commit(&mut s, accepted).map_err(says)?;
            if s.pos() != end + accepted as u32 {
                return Err(format!("{what}: the commit stands at {}", s.pos()));
            }
            if recurrent && to != end {
                // A position the verify's rows hold has no point: refused by
                // name, the session unmoved.
                let before = s.pos();
                match Target::cut(&mut s, end + 1) {
                    Ok(()) if before == end + 1 => {}
                    Ok(()) => {
                        return Err(format!(
                            "{what}: a cut to {} with no point was taken",
                            end + 1
                        ));
                    }
                    Err(e) if s.pos() == before && !e.to_string().is_empty() => {}
                    Err(e) => {
                        return Err(format!("{what}: the refused cut moved the session: {e}"));
                    }
                }
            }
            Target::cut(&mut s, to).map_err(says)?;
            if s.pos() != to {
                return Err(format!("{what}: the cut stands at {}", s.pos()));
            }
            if to < end {
                let (logits, pick) = keep::prompt_bits(&mut s, &ids[to as usize..])?;
                if (&logits, pick) != (&logits0, x0) {
                    return Err(format!(
                        "{what}: the ids continued from {to} leave another pick or logits row than \
                         the fresh run's"
                    ));
                }
            }
            for (t, (id, pick, logits)) in want.iter().take(CUT_STEPS).enumerate() {
                let got = s.step(*id, StepMode::HostFed).map_err(says)?.id;
                if got != *pick {
                    return Err(format!(
                        "{what}: plain step {t} picked {got}, the plain run's {pick}"
                    ));
                }
                let row = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
                if !bits_equal(&row, logits) {
                    return Err(format!("{what}: plain step {t}'s logits differ"));
                }
            }
        }
        notes.push(format!(
            "commit {accepted} of a {} draft",
            if right { "right" } else { "wrong" }
        ));
    }
    Ok(Verdict::Noted(format!(
        "a {CUT_IDS}-id prompt marked at {CUT_AT}: after {} a cut to each of the points {cuts:?} \
         leaves the fresh run's pick and logits row where ids continue, a cut to a position no \
         point holds is refused, and {CUT_STEPS} plain steps after each cut equal the plain \
         run's",
        notes.join(", ")
    )))
}

/// Whether `r` is a refusal by name that says the verify waits for its commit.
fn waits<T>(what: &str, r: Result<T, SessionError>, s: &Session, pos: u32) -> Result<(), String> {
    match r {
        Ok(_) => Err(format!("{what} ran while a verify waits for its commit")),
        Err(e) if e.to_string().contains(WAITS) => {
            if s.pos() != pos || s.verify_pending().is_none() {
                return Err(format!("{what} was refused and moved the session"));
            }
            Ok(())
        }
        Err(e) => Err(format!(
            "{what} was refused as {e}, which does not say the verify {WAITS}"
        )),
    }
}

/// A refusal that says `needle`.
fn refused<T>(what: &str, r: Result<T, SessionError>, needle: &str) -> Result<(), String> {
    match r {
        Ok(_) => Err(format!("{what} was accepted")),
        Err(e) if e.to_string().contains(needle) => Ok(()),
        Err(e) => Err(format!(
            "{what} was refused as {e}, which does not say {needle:?}"
        )),
    }
}

/// (e): what a verify refuses and what it refuses before its commit.
pub fn v_refused(s: &mut Session) -> Clause {
    s.reset().map_err(says)?;
    let x0 = prompt(s)?;
    if let Err(why) = s.verify_serves() {
        // A file the pass does not serve: every verify is refused by name and
        // the session is as a plain run leaves it.
        let at = s.pos();
        refused(
            "a verify of an unserved file",
            Verify::verify::<ROWS>(s, [x0, x0]),
            "does not serve",
        )?;
        if s.pos() != at || s.verify_pending().is_some() {
            return Err("a refused verify moved the session".into());
        }
        s.step(x0, StepMode::HostFed).map_err(says)?;
        return Ok(Verdict::Noted(format!(
            "the pass does not serve this file ({why}): a verify is refused by name, the \
             session unmoved and a plain step runs after it"
        )));
    }
    let at = s.pos();
    verify(s, [x0, x0])?;
    let stand = at + ROWS as u32;
    waits("a host-fed step", s.step(x0, StepMode::HostFed), s, stand)?;
    waits("a chain step", s.step(x0, StepMode::Chain), s, stand)?;
    waits(
        "a one-id prompt",
        Target::prompt(s, &[x0], Want::Argmax).map(|_| ()),
        s,
        stand,
    )?;
    let nine: Vec<u32> = vec![x0; 9];
    waits(
        "a prompt of nine ids",
        Target::prompt(s, &nine, Want::Argmax).map(|_| ()),
        s,
        stand,
    )?;
    waits(
        "a second verify",
        Verify::verify::<ROWS>(s, [x0, x0]),
        s,
        stand,
    )?;
    waits("a cut", Target::cut(s, 2), s, stand)?;
    waits("a checkpoint", s.checkpoint(), s, stand)?;
    waits("a probe", s.probe().map(|_| ()), s, stand)?;
    waits(
        "a read of the logits row",
        s.logits_row().map(|_| ()),
        s,
        stand,
    )?;
    refused(
        "a commit of none",
        Verify::commit(s, 0),
        "commit keeping 0 of",
    )?;
    refused(
        "a commit of more rows than the verify ran",
        Verify::commit(s, ROWS + 1),
        "commit keeping 3 of",
    )?;
    waits(
        "a step after the refused commits",
        s.step(x0, StepMode::HostFed),
        s,
        stand,
    )?;
    Verify::commit(s, ROWS).map_err(says)?;
    refused(
        "a commit with no verify",
        Verify::commit(s, 1),
        "no verify waits",
    )?;
    if s.pos() != stand {
        return Err(format!("the commit of both rows stands at {}", s.pos()));
    }
    s.step(x0, StepMode::HostFed).map_err(says)?;

    // The cache's last rows: a verify whose rows fit the cache is taken, and
    // one that does not is refused by name with the session unmoved.
    s.reset().map_err(says)?;
    prompt(s)?;
    let mut id = x0;
    while (s.pos() as usize) + ROWS < s.ctx() {
        id = s.step(id, StepMode::HostFed).map_err(says)?.id;
    }
    verify(s, [id, id])?;
    Verify::commit(s, 1).map_err(says)?;
    let last = s.pos();
    refused(
        "a verify at the cache's last row",
        Verify::verify::<ROWS>(s, [id, id]),
        "the cache holds",
    )?;
    if s.pos() != last || s.verify_pending().is_some() {
        return Err("a verify past the cache moved the session".into());
    }
    s.step(id, StepMode::HostFed).map_err(says)?;
    Ok(Verdict::Noted(format!(
        "a verify waits for its commit: a host-fed and a chain step, a one-id and a nine-id \
         prompt, a second verify, a cut, a checkpoint, a probe and the logits row are each \
         refused by name with the verify still committable; a commit of 0, of {} rows and of no \
         verify is refused by name; a verify of the last two rows is taken and one at the last \
         row is refused with the session unmoved",
        ROWS + 1
    )))
}
