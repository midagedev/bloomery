//! The body/session slot harness: the resident-slot contracts every body
//! with [`Slots`] answers ([`GpuModel::add_slots`],
//! [`GpuModel::select_slot`]), written once and driven from each body's e2e
//! gate through its [`SlotsAdapter`]. The clause names are the harness's, so
//! every body prints the same `check slots H<n> …` lines:
//!
//! - H1 interleave: [`STREAMS`] streams, each run solo on the load's one
//!   sequence first, then together on their own slots — every slot rewound,
//!   each slot's prompt, then a step a slot a round, a select between every
//!   token, in server order: every slot's ids, last logits (bit for bit),
//!   state hash ([`SlotsAdapter::state_hash`]) and position its solo run's.
//! - H3 bytes: `resident_bytes` grows by exactly `seq_bytes` a slot added,
//!   and `seq_bytes` equals its derivation from the owners of its terms
//!   ([`SlotsAdapter::seq_bytes_derived`]) and, where the body's load builds
//!   one, the plan's sequence descriptor ([`SlotsAdapter::seq_terms_bytes`]).
//! - H4 reset: a reset of the last slot rewinds it alone — its prompt run
//!   again gives its solo run's first ids, and every other slot's
//!   continuation is its solo run's.
//! - H6 refusals: a select out of range, an `add_slots` under the slots held
//!   and one of 0 are refused by name, the selection and the count kept.
//! - H7 captures: every slot holds its own captures — an added slot none
//!   before its first step while slot 0 holds its solo runs', each slot
//!   some after the interleave — and none after a `set_mode` round trip.
//!
//! The harness's next contracts, once the model runs several slots' rows in
//! one pass (`GpuModel::step_slots`):
//! - H2: one pass over N slots gives each slot's solo steps bit for bit —
//!   ids, logits and stores.
//! - H5: a fault on a slot, or in a pass over a slot set, refuses every
//!   other slot naming it, and only its own reset lifts it (the adapter's
//!   fault planter comes with it).
//!
//! The order is the gate's: [`interleave`] runs H3, H6, H1 and H7 and hands
//! back the model with every slot on its solo path ([`Interleaved`]); the
//! body gate runs its own fact clauses there, moving at most the last slot
//! off its path and reading any other slot's continuation through
//! [`Interleaved::continues`]; [`Interleaved::finish`] runs H4 last, since a
//! reset that reaches past its slot moves the slots those clauses read. A
//! call that fails inside a contract is that contract's red with the error
//! printed, so every contract prints its line whatever an earlier one left.

use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel, Slots};

use crate::{GateError, bits_equal, verdict};

/// Streams the harness runs, one a slot: two, the fewest that interleave.
pub const STREAMS: usize = 2;

const H1: &str = "slots H1 interleave";
const H3: &str = "slots H3 bytes";
const H4: &str = "slots H4 reset";
const H6: &str = "slots H6 refusals";
const H7: &str = "slots H7 captures";

/// What a body's e2e gate hands the harness: how its body loads, rewinds,
/// prompts and is read.
pub trait SlotsAdapter {
    /// The body under test. A parked sequence sits in the model's lot until
    /// the model drops, so it borrows nothing.
    type Body: Slots<Seq: 'static>;

    /// Greedy steps a stream takes past its prompt in the interleave.
    const STEPS: usize;
    /// Greedy steps of one continuation check ([`Interleaved::continues`]).
    const TAIL: usize;

    /// The body loaded to serve `slots` sequences, its one slot live: the
    /// load's plan counts `slots`, and [`GpuModel::add_slots`] makes the
    /// rest.
    fn open(&self, slots: usize) -> Result<GpuModel<Self::Body>, GateError>;

    /// The selected slot back to the state a prompt starts from:
    /// [`GpuModel::reset`], and any rows no call writes that
    /// [`SlotsAdapter::state_hash`] reads set to the gate's fill.
    fn rewind(&self, m: &mut GpuModel<Self::Body>) -> Result<(), GateError>;

    /// Stream `stream`'s prompt (`0..STREAMS`) on the selected slot from
    /// where it stands, by the path the body's gate gives that stream: its
    /// argmax. Another stream is refused by name.
    fn prompt(&self, m: &mut GpuModel<Self::Body>, stream: usize) -> Result<u32, GateError>;

    /// The selected slot's whole per-sequence state, digested: every store a
    /// call writes and a later call reads.
    fn state_hash(&self, m: &mut GpuModel<Self::Body>) -> Result<u64, GateError>;

    /// One sequence's device bytes, derived from the owners of its terms.
    fn seq_bytes_derived(&self, m: &GpuModel<Self::Body>) -> Result<Derived, GateError>;

    /// One sequence's device bytes as the plan's sequence descriptor counts
    /// them (`SeqTerms::bytes(ctx_max, 1)`), which [`Slots::seq_bytes`] must
    /// equal; `None` where the body's load builds no descriptor, and H3
    /// names the equality skipped.
    fn seq_terms_bytes(&self, _m: &GpuModel<Self::Body>) -> Result<Option<usize>, GateError> {
        Ok(None)
    }
}

/// A sequence's device bytes as a derivation gives them, and the
/// derivation's terms in words for H3's line.
pub struct Derived {
    pub bytes: usize,
    pub terms: String,
}

/// One stream's solo run on the load's one sequence: its prompt's argmax
/// and then `STEPS + 2·TAIL` greedy ids (the interleave's, the body gate's
/// continuation, H4's), and after the interleave's steps the last logits,
/// the state hash and the position.
struct Solo {
    ids: Vec<u32>,
    logits: Vec<f32>,
    hash: u64,
    pos: u32,
}

/// A slot's stream: its solo run, and the ids the slot has given since its
/// last rewind — the last of them the next step's input.
struct Stream {
    solo: Solo,
    given: Vec<u32>,
}

/// The model after [`interleave`]: every slot on its stream's solo path as
/// far as the contracts run so far show. The body gate's own clauses run
/// here (module doc), then [`Interleaved::finish`].
pub struct Interleaved<'a, A: SlotsAdapter> {
    a: &'a A,
    model: GpuModel<A::Body>,
    streams: Vec<Stream>,
    ok: bool,
}

/// H3, H6, H1 and H7 over `a`'s body loaded for [`STREAMS`] slots: every
/// stream's solo run on the load's one sequence first, then the slots added
/// and interleaved (module doc). Each contract prints its `check` line; the
/// verdict over all of them is [`Interleaved::finish`]'s.
pub fn interleave<A, B>(a: &A) -> Result<Interleaved<'_, A>, GateError>
where
    A: SlotsAdapter<Body = B>,
    B: Slots<Seq: 'static>,
{
    const { assert!(A::STEPS >= 1 && A::TAIL >= 1) };
    let mut m = a.open(STREAMS)?;
    m.set_mode(StepMode::Graph);
    let mut streams = Vec::with_capacity(STREAMS);
    for s in 0..STREAMS {
        streams.push(Stream {
            solo: solo(a, &mut m, s)?,
            given: Vec::new(),
        });
    }
    let mut ok = check(H3, bytes(a, &mut m));
    ok &= check(H6, refusals(&mut m));
    let added = own_captures(&mut m);
    ok &= check(H1, together(a, &mut m, &mut streams));
    ok &= check(H7, captures(&mut m, added));
    Ok(Interleaved {
        a,
        model: m,
        streams,
        ok,
    })
}

impl<A, B> Interleaved<'_, A>
where
    A: SlotsAdapter<Body = B>,
    B: Slots<Seq: 'static>,
{
    /// The model, for the body gate's own clauses.
    pub fn model(&mut self) -> &mut GpuModel<A::Body> {
        &mut self.model
    }

    /// The last id `slot` gave: its next step's input.
    pub fn last(&self, slot: usize) -> Result<u32, GateError> {
        let stream = self
            .streams
            .get(slot)
            .ok_or_else(|| format!("slot {slot} of {STREAMS}"))?;
        Ok(*stream
            .given
            .last()
            .ok_or_else(|| format!("slot {slot} has given no id"))?)
    }

    /// `slot` selected and stepped [`SlotsAdapter::TAIL`] greedy steps on
    /// from its last id: whether they are its solo run's next ids.
    pub fn continues(&mut self, slot: usize) -> Result<bool, GateError> {
        let tail = A::TAIL;
        let stream = self
            .streams
            .get_mut(slot)
            .ok_or_else(|| format!("slot {slot} of {STREAMS}"))?;
        let from = stream.given.len();
        let want = stream.solo.ids.get(from..from + tail).ok_or_else(|| {
            format!("slot {slot}'s solo run holds no {tail} ids past the {from} it has given")
        })?;
        self.model.select_slot(slot)?;
        greedy(&mut self.model, &mut stream.given, tail)?;
        Ok(stream.given[from..] == *want)
    }

    /// H4, and the harness's verdict over every contract it ran. The model
    /// drops here, its added sequences with it.
    pub fn finish(mut self) -> Result<bool, GateError> {
        let reset = self.reset_isolation();
        Ok(check(H4, reset) && self.ok)
    }

    /// H4: the last slot rewound and its prompt run again, then every other
    /// slot's continuation.
    fn reset_isolation(&mut self) -> Result<(bool, String), GateError> {
        let j = STREAMS - 1;
        self.model.select_slot(j)?;
        self.a.rewind(&mut self.model)?;
        let stream = &mut self.streams[j];
        stream.given = vec![self.a.prompt(&mut self.model, j)?];
        greedy(&mut self.model, &mut stream.given, A::TAIL)?;
        let rewound = stream.given[..] == stream.solo.ids[..=A::TAIL];
        let mut off = Vec::new();
        for s in 0..j {
            if !self.continues(s)? {
                off.push(s.to_string());
            }
        }
        let pass = rewound && off.is_empty();
        Ok((
            pass,
            format!(
                "slot {j} rewound and prompted again: its first {} ids {} its solo run's; every \
                 other slot's next {} ids {}",
                1 + A::TAIL,
                if rewound { "are" } else { "are not" },
                A::TAIL,
                if off.is_empty() {
                    "its solo run's continuation".to_string()
                } else {
                    format!(
                        "differ from its solo run's continuation for slot {}",
                        off.join(", ")
                    )
                },
            ),
        ))
    }
}

/// A contract's line, `check <name>: <detail> PASS|FAIL`, and its verdict;
/// a call that failed inside the contract is its red, the error the detail.
fn check(name: &str, arm: Result<(bool, String), GateError>) -> bool {
    let (pass, detail) = arm.unwrap_or_else(|e| (false, format!("a call failed: {e}")));
    println!("check {name}: {detail} {}", verdict(pass));
    pass
}

/// `n` greedy steps on the selected slot as it stands, each fed the id
/// before it, appended to `ids` (its last id the first step's input).
fn greedy<B: ChainBody>(
    m: &mut GpuModel<B>,
    ids: &mut Vec<u32>,
    n: usize,
) -> Result<(), GateError> {
    for _ in 0..n {
        let last = *ids.last().ok_or("greedy: no id to step from")?;
        ids.push(m.step(&[last])?);
    }
    Ok(())
}

/// Stream `stream` alone on the load's one sequence, from a rewind
/// ([`Solo`]).
fn solo<A: SlotsAdapter>(
    a: &A,
    m: &mut GpuModel<A::Body>,
    stream: usize,
) -> Result<Solo, GateError> {
    a.rewind(m)?;
    let mut ids = vec![a.prompt(m, stream)?];
    greedy(m, &mut ids, A::STEPS)?;
    let (logits, hash, pos) = (m.logits()?, a.state_hash(m)?, m.pos());
    greedy(m, &mut ids, 2 * A::TAIL)?;
    Ok(Solo {
        ids,
        logits,
        hash,
        pos,
    })
}

/// H3: the slots added, `resident_bytes` read around the call.
fn bytes<A, B>(a: &A, m: &mut GpuModel<B>) -> Result<(bool, String), GateError>
where
    A: SlotsAdapter<Body = B>,
    B: Slots<Seq: 'static>,
{
    let before = m.resident_bytes();
    m.add_slots(STREAMS)?;
    let grown = i128::try_from(m.resident_bytes())? - i128::try_from(before)?;
    let seq = m.body("slots_gate")?.seq_bytes();
    let added = STREAMS - 1;
    let derived = a.seq_bytes_derived(m)?;
    let terms = a.seq_terms_bytes(m)?;
    let pass = grown == i128::try_from(added * seq)?
        && seq == derived.bytes
        && terms.is_none_or(|t| t == seq);
    Ok((
        pass,
        format!(
            "{added} added sequence(s) grew resident_bytes by {grown}, seq_bytes {seq} a slot; the \
             derived {} ({}); the plan's sequence descriptor {}",
            derived.bytes,
            derived.terms,
            terms.map_or_else(
                || "not built by this load, its equality skipped".to_string(),
                |t| t.to_string()
            ),
        ),
    ))
}

/// H6: the refusals, each by its entry's name and words.
fn refusals<B: Slots<Seq: 'static>>(m: &mut GpuModel<B>) -> Result<(bool, String), GateError> {
    let (selected, held) = (m.selected(), m.slots());
    let select = refused(
        m.select_slot(held),
        "GpuModel::select_slot",
        &format!("serves 0..{held}"),
    );
    let under = refused(
        m.add_slots(held - 1),
        "GpuModel::add_slots",
        &format!("already serves {held}"),
    );
    let zero = refused(
        m.add_slots(0),
        "GpuModel::add_slots",
        "a slot count of at least 1",
    );
    let kept = m.selected() == selected && m.slots() == held;
    Ok((
        select && under && zero && kept,
        format!(
            "select {held} of {held} slots named {select}, add_slots({}) named {under}, \
             add_slots(0) named {zero}; the selection and the count kept {kept}",
            held - 1
        ),
    ))
}

/// Whether `r` is `entry`'s refusal by shape, its words holding `says`.
fn refused(r: Result<(), GpuError>, entry: &str, says: &str) -> bool {
    match r {
        Err(GpuError::Shape { what, detail }) => what == entry && detail.contains(says),
        _ => false,
    }
}

/// H7's first reading, before any added slot steps: whether every added
/// slot, selected, holds no capture, and whether slot 0 holds its solo
/// runs'.
fn own_captures<B: Slots<Seq: 'static>>(m: &mut GpuModel<B>) -> Result<(bool, bool), GateError> {
    let mut none = true;
    for s in 1..STREAMS {
        m.select_slot(s)?;
        none &= !m.has_capture();
    }
    m.select_slot(0)?;
    Ok((none, m.has_capture()))
}

/// H1: every slot rewound, each slot's prompt, then the rounds.
fn together<A, B>(
    a: &A,
    m: &mut GpuModel<B>,
    streams: &mut [Stream],
) -> Result<(bool, String), GateError>
where
    A: SlotsAdapter<Body = B>,
    B: Slots<Seq: 'static>,
{
    // Every slot rewound before any prompt, slot 0 last: the interleave's
    // start does not lean on a reset acting on its own slot alone (H4).
    for s in (0..STREAMS).rev() {
        m.select_slot(s)?;
        a.rewind(m)?;
    }
    for (s, stream) in streams.iter_mut().enumerate() {
        m.select_slot(s)?;
        stream.given = vec![a.prompt(m, s)?];
    }
    // The last logits are read inside the final round: the head is the
    // model's one, so the other slot's step replaces what it holds.
    let mut logits = vec![Vec::new(); STREAMS];
    for r in 0..A::STEPS {
        for (s, stream) in streams.iter_mut().enumerate() {
            m.select_slot(s)?;
            greedy(m, &mut stream.given, 1)?;
            if r + 1 == A::STEPS {
                logits[s] = m.logits()?;
            }
        }
    }
    let mut off = Vec::new();
    for (s, (stream, logits)) in streams.iter().zip(&logits).enumerate() {
        m.select_slot(s)?;
        let solo = &stream.solo;
        let parts = [
            ("ids", stream.given[..] == solo.ids[..=A::STEPS]),
            ("logits", bits_equal(logits, &solo.logits)),
            ("state", a.state_hash(m)? == solo.hash),
            ("position", m.pos() == solo.pos),
        ];
        off.extend(
            parts
                .into_iter()
                .filter(|&(_, same)| !same)
                .map(|(part, _)| format!("slot {s} {part}")),
        );
    }
    let pass = off.is_empty();
    Ok((
        pass,
        format!(
            "{} interleaved steps a slot against each stream's solo run: {}",
            A::STEPS,
            if pass {
                "every slot's ids, last logits, state hash and position bit for bit".to_string()
            } else {
                format!("differs in {}", off.join(", "))
            }
        ),
    ))
}

/// H7: `added` ([`own_captures`]), every slot's captures after the
/// interleave, then a `set_mode` round trip.
fn captures<B: Slots<Seq: 'static>>(
    m: &mut GpuModel<B>,
    added: Result<(bool, bool), GateError>,
) -> Result<(bool, String), GateError> {
    let (added_none, home) = added?;
    let mut held = true;
    for s in 0..STREAMS {
        m.select_slot(s)?;
        held &= m.has_capture();
    }
    m.set_mode(StepMode::Eager);
    m.set_mode(StepMode::Graph);
    let mut gone = true;
    for s in 0..STREAMS {
        m.select_slot(s)?;
        gone &= !m.has_capture();
    }
    Ok((
        added_none && home && held && gone,
        format!(
            "before its first step every added slot held none {added_none} and slot 0 its solo \
             runs' {home}; every slot held its own after the interleave {held}; none after a \
             set_mode round trip {gone}"
        ),
    ))
}
