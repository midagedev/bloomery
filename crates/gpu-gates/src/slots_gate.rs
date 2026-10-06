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
//! - H2 one pass, for a body that runs several slots' rows as one pass
//!   ([`SlotRows`], [`Interleaved::one_pass`]): after H1, [`SlotsAdapter::TAIL`]
//!   passes of one row a slot ([`GpuModel::step_slots`]), the first half
//!   captured and the rest eager — every slot's ids, last logits, state hash
//!   and position its solo run's; and the captured pass of one row a slot
//!   holds exactly the nodes of the pass of as many rows on one slot plus
//!   each added slot's sequence-bound launches
//!   ([`PassAdapter::added_slot_launches`]), each count the body's own
//!   ([`PassAdapter::slots_launches`]).
//! - H3 bytes: `resident_bytes` grows by exactly `seq_bytes` a slot added,
//!   and `seq_bytes` equals its derivation from the owners of its terms
//!   ([`SlotsAdapter::seq_bytes_derived`]) and, where the body's load builds
//!   one, the plan's sequence descriptor ([`SlotsAdapter::seq_terms_bytes`]).
//! - H4 reset: a reset of the last slot rewinds it alone — its prompt run
//!   again gives its solo run's first ids, and every other slot's
//!   continuation is its solo run's.
//! - H5 poison: a failed round of both slots — the host service's refusal
//!   (a), a card fault the readback carries (b) — poisons every slot it ran
//!   on: every call refused by name, the tier's own refusal standing with
//!   it, a reset of one slot of the set alone lifting nothing, and after each
//!   slot's own reset both slots running their solo paths from their
//!   prompts again ([`Interleaved::poisons`], the adapter's planter
//!   [`SlotsAdapter::plant_refusal`] and window read
//!   [`SlotsAdapter::tier_poisoned`]); a prompt call on slot 0 refused by
//!   the host poisons the same way, slot 0 alone (c); and the load of one
//!   sequence keeps the load's own rule — its step's host refusal standing
//!   until its reset, the next step refused by name,
//!   the reset lifting it, the fresh-context prompt bit for bit
//!   ([`one_slot`]). A probe step that never returns is its clause's red
//!   line, by name ([`watched`]).
//! - H6 refusals: a select out of range, an `add_slots` under the slots held
//!   and one of 0 are refused by name, the selection and the count kept.
//! - H7 captures: every slot holds its own captures — an added slot none
//!   before its first step while slot 0 holds its solo runs', each slot
//!   some after the interleave — and none after a `set_mode` round trip.
//!
//! The order is the gate's: [`interleave`] runs H3, H6, H1 and H7 and hands
//! back the model with every slot on its solo path ([`Interleaved`]); a
//! [`SlotRows`] body's gate runs H2 there ([`Interleaved::one_pass`]), which
//! keeps every slot on its path; the body gate runs its own fact clauses
//! there, moving at most the last slot off its path and reading any other
//! slot's continuation through [`Interleaved::continues`];
//! [`Interleaved::finish`] runs H4, then H5 — both its resets move both
//! slots off their paths, so nothing may read a path behind it. A call that
//! fails inside a contract is that contract's red with the error printed,
//! so every contract prints its line whatever an earlier one left.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use bloomery_gpu::model::{ChainBody, SlotRows, StepMode};
use bloomery_gpu::{Fault, FaultSite, GpuError, GpuModel, LAYER_HEAD, Slots};
use threads::helper::{Placement, spawn_helper};

use crate::{GateError, bits_equal, verdict};

/// Streams the harness runs, one a slot: two, the fewest that interleave.
pub const STREAMS: usize = 2;

const H1: &str = "slots H1 interleave";
const H2: &str = "slots H2 one pass";
const H3: &str = "slots H3 bytes";
const H4: &str = "slots H4 reset";
const H5: &str = "slots H5 poison";
const H5R: &str = "slots H5 one-slot";
const H6: &str = "slots H6 refusals";
/// How long one of H5's probe steps may take: a single step, which returns
/// in well under a second; one still out past this waits on the card for a
/// host service that will never come.
const PROBE_BOUND: Duration = Duration::from_secs(60);
/// How long the watchdog waits after naming a blocked probe before it ends
/// the process: a stack watch (`BLOOMERY_GATE_STACKS` above
/// [`PROBE_BOUND`] and under this) dumps the blocked threads in between.
const PROBE_GRACE: Duration = Duration::from_secs(120);
/// The slot sets a round of both slots poisons ([`SlotsAdapter::step_all`]):
/// both, by one pass of a [`SlotRows`] body (the qwen3moe, Qwen3.8 and GLM
/// adapters), or slot 0 alone, by the seat fallback (the V4.1 and qwen35moe
/// adapters), whose first step's error ends the round.
const ROUND_SETS: [&str; 2] = ["slots 0 and 1", "slot 0"];
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
    /// Continuation checks the body gate's own clauses read of one slot
    /// ([`Interleaved::continues`]), [`SlotsAdapter::TAIL`] ids each.
    const CONTINUES: usize = 1;

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

    /// H5's planter: the next host service's refusal planted on the load's
    /// host tier, one reach a family through its body's own mut path to
    /// the tier. The tier's input checks fire on state the card should
    /// already have refused, which no sound card produces, so the harness
    /// reaches them through this seam
    /// ([`bloomery_gpu::host::HostTier::plant_refusal`]). `false` when the
    /// load serves no host work: no host service to refuse, so H5's host
    /// cases do not reach it, and its line says so. A test seam; never on
    /// a serving path.
    fn plant_refusal(&self, m: &mut GpuModel<Self::Body>) -> Result<bool, GateError>;

    /// Whether the load's host tier is still poisoned: H5's window reads it
    /// after a reset of the slot that did not fail — the tier a refused
    /// round poisoned stays poisoned until the model's own poison lifts
    /// ([`GpuModel::reset`]), which a reset of one slot of the set is not.
    fn tier_poisoned(&self, m: &mut GpuModel<Self::Body>) -> Result<bool, GateError>;

    /// H5's round of several slots: every slot's last id `last` stepped on
    /// its own slot, each slot's greedy next token back, by the way the
    /// body serves several slots together — one pass
    /// ([`GpuModel::step_slots`], a body that implements [`SlotRows`]) or
    /// a select and a step a slot (the seat fallback, [`step_rows_in_turn`]
    /// of `gpu-gates`' bind). The default is the fallback: a body without
    /// [`SlotRows`] serves no pass.
    fn step_all(&self, m: &mut GpuModel<Self::Body>, last: &[u32]) -> Result<Vec<u32>, GpuError>
    where
        <Self::Body as Slots>::Seq: 'static,
    {
        let mut out = Vec::with_capacity(last.len());
        for (s, &t) in last.iter().enumerate() {
            m.select_slot(s)?;
            out.push(m.step(&[t])?);
        }
        Ok(out)
    }
}

/// What H5 plants: the next host service's refusal
/// ([`SlotsAdapter::plant_refusal`]), or the card's fault word at the head's
/// quant column — the fault the next readback carries ([`GpuError::Fault`],
/// [`GpuModel::plant_fault`]).
#[derive(Clone, Copy)]
pub enum Planted {
    /// The next step service refuses its input, the tier poisoned as
    /// refused ([`bloomery_gpu::host::HostTier::plant_refusal`]).
    Refusal,
    /// The card's fault word raised at `fault` ([`GpuModel::plant_fault`]).
    Fault(Fault),
}

impl Planted {
    /// The head's quant-column fault every family plants
    /// ([`Planted::Fault`]): no model layer claimed, the readback's own.
    #[must_use]
    pub fn fault() -> Planted {
        Planted::Fault(Fault::at(LAYER_HEAD, FaultSite::QuantColumn))
    }
}

/// What a [`SlotRows`] body's gate hands H2 besides its [`SlotsAdapter`]:
/// the launch counts of a pass of several slots.
pub trait PassAdapter: SlotsAdapter<Body: SlotRows> {
    /// The launches one more busy slot adds to a pass of the same rows,
    /// derived from the body's layer kinds and the launches each binds to
    /// one sequence.
    fn added_slot_launches(&self, m: &GpuModel<Self::Body>) -> Result<Launches, GateError>;

    /// The body's own count of the launches of the captured pass of `key`
    /// (each `(slot, rows)`, in pass order).
    fn slots_launches(
        &self,
        m: &GpuModel<Self::Body>,
        key: &[(usize, usize)],
    ) -> Result<usize, GateError>;
}

/// A sequence's device bytes as a derivation gives them, and the
/// derivation's terms in words for H3's line.
pub struct Derived {
    pub bytes: usize,
    pub terms: String,
}

/// A slot's launches as a derivation gives them, and the derivation's
/// terms in words for H2's line.
pub struct Launches {
    pub n: usize,
    pub terms: String,
}

/// One stream's solo run on the load's one sequence: its prompt's argmax
/// and then `STEPS + (2 + CONTINUES)·TAIL` greedy ids (the interleave's,
/// H2's, the body gate's continuations, H4's), and after the interleave's
/// steps and after H2's the last logits, the state hash and the position.
struct Solo {
    ids: Vec<u32>,
    at_steps: Read,
    at_pass: Read,
}

/// The selected slot's last logits, state hash and position.
struct Read {
    logits: Vec<f32>,
    hash: u64,
    pos: u32,
}

impl Read {
    fn of<A: SlotsAdapter>(a: &A, m: &mut GpuModel<A::Body>) -> Result<Read, GateError> {
        Ok(Read {
            logits: m.logits()?,
            hash: a.state_hash(m)?,
            pos: m.pos(),
        })
    }
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

    /// H4, H5, and the harness's verdict over every contract it ran. The
    /// model drops here, its added sequences with it.
    pub fn finish(mut self) -> Result<bool, GateError> {
        let reset = check(H4, self.reset_isolation());
        let poison = check(H5, self.poisons());
        let Interleaved { a, model, ok, .. } = self;
        // H5's one-slot clause opens a load of its own: the harness's goes
        // first, so the card holds one load at a time.
        drop(model);
        let one_slot = check(H5R, one_slot(a));
        Ok(reset && poison && one_slot && ok)
    }

    /// H5 (module doc), on the harness's model after H4: a round of both
    /// slots refused by the host (a), then one the card faulted (b), then a
    /// prompt call on slot 0 refused by the host (c) — each poison checked
    /// through its whole life: the call's error, every call refused naming
    /// the slots, a reset of one slot of the set alone lifting nothing, and after
    /// each slot's own reset both slots running their solo paths from their
    /// prompts again.
    fn poisons(&mut self) -> Result<(bool, String), GateError> {
        let refusal = self.poison_round(Planted::Refusal)?;
        let fault = self.poison_round(Planted::fault())?;
        let prompt = self.poison_prompt()?;
        let say = |case: &Option<(bool, bool, bool)>| match case {
            Some((held, window, back)) => format!(
                "the call's error named it {held}, every call and the tier refusing through the \
                 other slot's reset {window}, both slots their solo paths from their prompts \
                 again {back}"
            ),
            None => "not reached: the load serves no host work".to_string(),
        };
        let pass = [refusal, fault, prompt]
            .iter()
            .all(|c| c.is_none_or(|(held, window, back)| held && window && back));
        Ok((
            pass,
            format!(
                "(a) a two-slot round refused by the host: {}; (b) the same with a card fault: \
                 {}; (c) a prompt call on slot 0 refused by the host: {}",
                say(&refusal),
                say(&fault),
                say(&prompt)
            ),
        ))
    }

    /// Plant `what` for H5: `false` when it is a host refusal and the load
    /// serves no host work ([`SlotsAdapter::plant_refusal`]).
    fn plant(&mut self, what: Planted) -> Result<bool, GateError> {
        match what {
            Planted::Refusal => self.a.plant_refusal(&mut self.model),
            Planted::Fault(fault) => {
                self.model.plant_fault(fault)?;
                Ok(true)
            }
        }
    }

    /// One H5 case: plant `what`, run a round of both slots
    /// ([`SlotsAdapter::step_all`]), and check the poison it leaves through
    /// [`Interleaved::poison_life`]; `None` when the load has nothing to
    /// plant it on.
    fn poison_round(&mut self, what: Planted) -> Result<Option<(bool, bool, bool)>, GateError> {
        let lasts: Vec<u32> = (0..STREAMS)
            .map(|s| self.last(s))
            .collect::<Result<_, GateError>>()?;
        if !self.plant(what)? {
            return Ok(None);
        }
        let round = self.a.step_all(&mut self.model, &lasts);
        let held = match &what {
            Planted::Refusal => round
                .as_ref()
                .is_err_and(|e| e.to_string().contains("host saw undefined input")),
            // The readback carries the planted word as its own fault
            // ([`Head::token`]); which fault it is belongs to the planter.
            Planted::Fault(_) => matches!(round, Err(GpuError::Fault { .. })),
        };
        let tier = matches!(what, Planted::Refusal);
        let (life, back) = self.poison_life(lasts[0], &ROUND_SETS, tier)?;
        Ok(Some((held, life, back)))
    }

    /// H5(c) (module doc): a prompt call on slot 0 refused by the host —
    /// the service a prompt call runs, whatever port it serves by, the same
    /// poison through the recording owner a step's refusal takes
    /// ([`GpuModel::run_rows`]): the refusing slot's own set, checked
    /// through [`Interleaved::poison_life`].
    fn poison_prompt(&mut self) -> Result<Option<(bool, bool, bool)>, GateError> {
        let last = self.last(0)?;
        self.model.select_slot(0)?;
        if !self.plant(Planted::Refusal)? {
            return Ok(None);
        }
        let prompt = self.a.prompt(&mut self.model, 0);
        let held = prompt.is_err_and(|e| e.to_string().contains("host saw undefined input"));
        let (life, back) = self.poison_life(last, &["slot 0"], true)?;
        Ok(Some((held, life, back)))
    }

    /// The poison a failed call on slot 0 leaves, through its life: every
    /// call refused naming the call's slots (one of `sets`), one slot's
    /// reset alone lifting nothing — the step of a slot it left in the set
    /// still the model's poison refusal of that slot alone, not a step on
    /// the part-written state, and a host refusal's tier standing with it,
    /// lifted only with the model's ([`SlotsAdapter::tier_poisoned`]) — and
    /// both slots' solo paths from their prompts after each own reset. A
    /// set of slot 0 alone has slot 1 reset and slot 0 stepped; a pass's
    /// set of both has slot 0 reset and slot 1 stepped. Every reset follows
    /// the recovery order a failed pass forces: a pass that fails keeps
    /// each slot's rows waiting for a commit, and a body refuses to select
    /// away from a live slot whose rows wait (`Body38::swap_seq`, the GLM
    /// body's), so the live slot is reset first, then each other slot
    /// selected and reset — reset slot 0, select slot 1, reset slot 1 after
    /// a pass of both. Each probe step runs under a watchdog ([`watched`]):
    /// one that never returns is this clause's red line, by name.
    fn poison_life(
        &mut self,
        last: u32,
        sets: &[&str],
        tier: bool,
    ) -> Result<(bool, bool), GateError> {
        self.model.select_slot(0)?;
        let model = &mut self.model;
        let named = watched(H5, "slot 0's step after the failed call", || {
            model.step(&[last])
        })?;
        let set = poisoned_set(&named).map(str::to_string);
        let named = set.as_deref().is_some_and(|s| sets.contains(&s));
        let (reset, probe) = if set.as_deref() == Some(ROUND_SETS[0]) {
            (0, STREAMS - 1)
        } else {
            (STREAMS - 1, 0)
        };
        self.model.select_slot(reset)?;
        self.a.rewind(&mut self.model)?;
        self.model.select_slot(probe)?;
        let model = &mut self.model;
        let still = watched(H5, "a set slot's step after the other's reset", || {
            model.step(&[last])
        })?;
        let alone = format!("slot {probe}");
        let still = poisoned_set(&still) == Some(alone.as_str());
        let held = if tier {
            self.a.tier_poisoned(&mut self.model)?
        } else {
            // A card fault never poisons the tier; the model's poison is
            // the whole of it.
            true
        };
        // Every slot reset before any prompt — a slot of the set still
        // poisoned refuses every call, a prompt on a slot already reset
        // included — the live slot first, in the recovery order above.
        self.a.rewind(&mut self.model)?;
        for s in (0..STREAMS).filter(|&s| s != probe) {
            self.model.select_slot(s)?;
            self.a.rewind(&mut self.model)?;
        }
        let mut back = true;
        for s in 0..STREAMS {
            self.model.select_slot(s)?;
            let first = self.a.prompt(&mut self.model, s)?;
            let stream = &mut self.streams[s];
            stream.given = vec![first];
            greedy(&mut self.model, &mut stream.given, A::TAIL)?;
            back &= stream.given[..] == stream.solo.ids[..=A::TAIL];
        }
        Ok((named && still && held, back))
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

impl<A, B> Interleaved<'_, A>
where
    A: PassAdapter<Body = B>,
    B: SlotRows<Seq: 'static>,
{
    /// H2 (module doc), its verdict folded into [`Interleaved::finish`]'s.
    /// Every slot stays on its solo path: its rows are its next steps.
    pub fn one_pass(&mut self) {
        let arm = one_pass(self.a, &mut self.model, &mut self.streams);
        self.ok &= check(H2, arm);
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
    let at_steps = Read::of(a, m)?;
    greedy(m, &mut ids, A::TAIL)?;
    let at_pass = Read::of(a, m)?;
    greedy(m, &mut ids, (1 + A::CONTINUES) * A::TAIL)?;
    Ok(Solo {
        ids,
        at_steps,
        at_pass,
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

/// H5's one-slot clause (module doc): a host refusal in a single-slot step
/// on a load of one sequence — no parked slot for the poison to name — is
/// the load's own: the next step before the reset refused by name — the
/// tier's poison, or the body's own refusal ahead of it — with its stream
/// drained (a probe under [`watched`]), and the load's reset lifting it
/// exactly as before slots: the next step served, the fresh-context prompt
/// the first's bit for bit. Then a prompt call's host refusal on the same
/// load: the step after it, which only the tier's poison refuses, drained
/// and refused by name, and the reset lifting it.
fn one_slot<A, B>(a: &A) -> Result<(bool, String), GateError>
where
    A: SlotsAdapter<Body = B>,
    B: Slots<Seq: 'static>,
{
    let mut m = a.open(1)?;
    m.set_mode(StepMode::Graph);
    a.rewind(&mut m)?;
    let first = a.prompt(&mut m, 0)?;
    if !a.plant_refusal(&mut m)? {
        return Ok((
            true,
            "not reached: the load of one sequence serves no host work".to_string(),
        ));
    }
    let refused = m
        .step(&[first])
        .is_err_and(|e| e.to_string().contains("host saw undefined input"));
    let before = watched(H5R, "the step before the load's reset", || m.step(&[first]))?;
    let held = before.is_err();
    a.rewind(&mut m)?;
    let again = a.prompt(&mut m, 0)?;
    let served = m.step(&[again]).is_ok();
    let fresh = again == first;
    // A prompt call's host refusal on the same load: no parked slot for the
    // model to record it by and no failed step for the body to refuse by,
    // so the tier's poison alone stands until the reset — the next step,
    // launched, must be refused by it with its stream drained, not left
    // waiting on a host service that will not come.
    a.rewind(&mut m)?;
    a.plant_refusal(&mut m)?;
    let prompt_refused = a
        .prompt(&mut m, 0)
        .is_err_and(|e| e.to_string().contains("host saw undefined input"));
    let after = watched(H5R, "the step after the refused prompt call", || {
        m.step(&[first])
    })?;
    let drained = after.is_err();
    a.rewind(&mut m)?;
    let lifted = a.prompt(&mut m, 0)? == first;
    let said = |r: &Result<u32, GpuError>| match r {
        Ok(id) => format!("served id {id}"),
        Err(e) => format!("\"{e}\""),
    };
    Ok((
        refused && held && served && fresh && prompt_refused && drained && lifted,
        format!(
            "a one-slot step's host refusal {refused}; the next step before the reset refused by \
             name {held} ({}); the load's reset lifted it, the next step served \
             {served} and the prompt after it the first's bit for bit {fresh}; a prompt call's \
             host refusal {prompt_refused}, the step after it refused by name {drained} ({}), \
             the reset lifting it, the prompt the first's bit for bit {lifted}",
            said(&before),
            said(&after)
        ),
    ))
}

/// `run` on this thread under a watchdog, as `gate_swap`'s queue arm runs
/// its clauses: past [`PROBE_BOUND`] the watchdog prints `name`'s red line
/// naming `what` and, after [`PROBE_GRACE`], ends the process — a host
/// blocked inside the driver has nothing that returns it. A panic in `run`
/// is not a hang: it unwinds as it would unwatched.
fn watched<T>(name: &str, what: &str, run: impl FnOnce() -> T) -> Result<T, GateError> {
    let fail = format!(
        "check {name}: {what} did not return in {PROBE_BOUND:?}: the card waits on a host \
         service that will never come {}",
        verdict(false)
    );
    let (done, watch) = mpsc::channel::<()>();
    let (watchdog, _) = spawn_helper("slots-h5-watch", Placement::Float, move || {
        if matches!(
            watch.recv_timeout(PROBE_BOUND),
            Err(RecvTimeoutError::Timeout)
        ) {
            println!("{fail}");
            std::thread::sleep(PROBE_GRACE);
            std::process::abort();
        }
    })?;
    let ran = run();
    let _ = done.send(());
    let _ = watchdog.join();
    Ok(ran)
}

/// The slot set the model's poison refusal `r` names, as the model prints
/// it ("slot 0", "slots 0 and 1"); `None` for any other result — a step
/// served, or another refusal (the tier's own, a body's), which no poison
/// set of the model's names.
fn poisoned_set(r: &Result<u32, GpuError>) -> Option<&str> {
    match r {
        Err(GpuError::Shape { detail, .. }) => detail
            .split_once(" is poisoned")
            .or_else(|| detail.split_once(" are poisoned"))
            .map(|(set, _)| set),
        _ => None,
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
        let at = &solo.at_steps;
        let parts = [
            ("ids", stream.given[..] == solo.ids[..=A::STEPS]),
            ("logits", bits_equal(logits, &at.logits)),
            ("state", a.state_hash(m)? == at.hash),
            ("position", m.pos() == at.pos),
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

/// H2: the passes of one row a slot, then the node counts.
fn one_pass<A, B>(
    a: &A,
    m: &mut GpuModel<B>,
    streams: &mut [Stream],
) -> Result<(bool, String), GateError>
where
    A: PassAdapter<Body = B>,
    B: SlotRows<Seq: 'static>,
{
    let captured = A::TAIL.div_ceil(2);
    let mut logits = Vec::new();
    for r in 0..A::TAIL {
        m.set_mode(if r < captured {
            StepMode::Graph
        } else {
            StepMode::Eager
        });
        let lasts = streams
            .iter()
            .enumerate()
            .map(|(s, stream)| {
                stream
                    .given
                    .last()
                    .map(|&t| [t])
                    .ok_or_else(|| format!("slot {s} has given no id"))
            })
            .collect::<Result<Vec<[u32; 1]>, String>>()?;
        let rows: Vec<(usize, &[u32])> = lasts.iter().map(|t| &t[..]).enumerate().collect();
        let out = m.step_slots(&rows)?;
        if out.ids.len() != streams.len() {
            return Err(format!(
                "a pass of one row a slot over {} slots read back {} ids",
                streams.len(),
                out.ids.len()
            )
            .into());
        }
        for (stream, &id) in streams.iter_mut().zip(&out.ids) {
            stream.given.push(id);
        }
        if r + 1 == A::TAIL {
            logits = m.slots_logits()?;
        }
    }
    m.set_mode(StepMode::Graph);
    let mut off = Vec::new();
    for (s, stream) in streams.iter().enumerate() {
        m.select_slot(s)?;
        let solo = &stream.solo;
        let at = &solo.at_pass;
        let row = logits
            .get(s)
            .ok_or_else(|| format!("no logits row for slot {s}"))?;
        let parts = [
            ("ids", stream.given[..] == solo.ids[..=A::STEPS + A::TAIL]),
            ("logits", bits_equal(row, &at.logits)),
            ("state", a.state_hash(m)? == at.hash),
            ("position", m.pos() == at.pos),
        ];
        off.extend(
            parts
                .into_iter()
                .filter(|&(_, same)| !same)
                .map(|(part, _)| format!("slot {s} {part}")),
        );
    }
    let one = [(0, streams.len())];
    let each: Vec<(usize, usize)> = (0..streams.len()).map(|s| (s, 1)).collect();
    let (n_one, n_each) = (m.capture_slots(&one)?, m.capture_slots(&each)?);
    let (own_one, own_each) = (a.slots_launches(m, &one)?, a.slots_launches(m, &each)?);
    let derived = a.added_slot_launches(m)?;
    let added = i128::try_from(n_each)? - i128::try_from(n_one)?;
    let want = i128::try_from((streams.len() - 1) * derived.n)?;
    let nodes_ok = added == want && n_one == own_one && n_each == own_each;
    let same = off.is_empty();
    Ok((
        same && nodes_ok,
        format!(
            "{} passes of one row a slot ({captured} captured, the rest eager) against each \
             stream's solo run: {}; the pass of one row a slot holds {n_each} nodes and the pass \
             of {} rows on slot 0 {n_one}, {added} added for {} added slot(s), derived {} a slot \
             ({}), the body's own counts {own_each} and {own_one}",
            A::TAIL,
            if same {
                "every slot's ids, last logits, state hash and position bit for bit".to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            streams.len(),
            streams.len() - 1,
            derived.n,
            derived.terms,
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
