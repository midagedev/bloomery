//! The engine thread: the one place the engine is called. It takes the
//! requests the [`Board`] admits to slots, runs each one's prompt as its own
//! calls, and then steps every running request together: the plain steps of
//! several in one [`Engine::step_slots`] call, the drafted passes of several
//! in one [`Engine::advance_slots`] call (the steps first), a plain
//! [`Engine::next`] (or one drafted pass) when one runs. It runs the slot
//! actions the board reserved between those calls.
//!
//! On an engine that names a prompt quantum ([`Engine::prompt_quantum`]), a
//! prompt that takes a slot while another slot is busy generating, and has
//! more than one quantum left after what the cache kept, runs as calls cut on
//! the quantum's grid, one a round, each round's decode of the busy slots
//! between them: a long prompt does not stall the streams already answering. Its request
//! takes no decode round until its prompt ends. Without another busy slot, or
//! on an engine that names none, a prompt runs in one go.
//!
//! A request's events reach its HTTP thread over a channel of its own (the
//! choices of one request share theirs, each event tagged with its choice). Its
//! counters, its `/slots` view and its slot's release are booked before its
//! last event is sent, so the response a client reads finds the slot free
//! and the counters final. A client gone ends its request at the next event,
//! a stream's or a whole answer's alike: its HTTP thread's sink fails (a
//! stream's write or the shared transport-error check), the thread drops
//! its channel, the engine thread's send fails, and the slot is released.
//!
//! An engine whose slots take it in turns ([`Engine::turns`]) is run by the
//! rules of [`crate::swap`] instead: one request on the engine at a time, the
//! others' states parked, and the slot changing hands at step and turn
//! boundaries.
//!
//! An engine error is fatal: every request in the failed call gets the error,
//! every waiting one is refused, and the server ends ([`crate::api::End`]). A
//! panic on this thread ends the server the same way, by name.
//!
//! An orderly stop ([`Shared::begin_stop`]) is read before each action,
//! start, planned prompt call and decode round of a round (when the slots
//! take turns, at each round's top and before the running request's step):
//! the thread ends without calling the
//! engine again, and every request it holds or the board queues is answered
//! with the stop's cause before the loop's end, which the stop waits for
//! ([`Shared::wait_stop`]).

use std::io;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;

use crate::admit::Lease;
use crate::api::{Answers, End, EngineFailure, Stop, Waited, client_gone, panic_text, relock};
use crate::engine::{
    Drafted, EngineError, Sampler, SamplerFactory, Saved, SlotPass, SlotRow, StateError, Waits,
};
use crate::genloop::{
    Depth, Event, Gen, GenError, GenParams, Need, Outcome, Prompt, Slot, StopKind, Timings,
    cause_word, ms_since,
};
use crate::live::{Ending, Finished, Ring};
use crate::media::{Held, MediaFeed};
use crate::sched::{Board, Reserve, SlotView};
use crate::swap::{Entry, NoRoom, Park, ParkTable, QUANTUM, Turn};

/// The server's counters, `/metrics`' totals.
#[derive(Default)]
pub(crate) struct Stats {
    /// Prompt tokens evaluated: the prompt less what the cache kept.
    pub n_prompt_total: u64,
    /// Prompt tokens the cache kept instead (every request's `cache_n`).
    pub n_prompt_cached_total: u64,
    pub t_prompt_ms_total: f64,
    /// The prompt cache's work before each prompt (every request's
    /// `cache_ms`).
    pub t_cache_ms_total: f64,
    pub n_predicted_total: u64,
    pub t_predicted_ms_total: f64,
    /// Engine calls: a prompt's, and each step of one slot or several.
    pub n_decode_total: u64,
    /// Requests whose client or event receiver left before their prompt started.
    pub n_cancelled_before_prompt_total: u64,
    /// Over every engine call, the slots it carried.
    pub n_busy_slots_total: u64,
    /// The longest a request's sequence grew (prompt and generation, `n_past`).
    pub n_tokens_max: u64,
    /// The ids the draft proposed, the ones kept, and the passes that verified
    /// a proposal.
    pub n_draft_total: u64,
    pub n_draft_accepted_total: u64,
    pub n_draft_passes_total: u64,
    /// The draft's counts by depth, depth `d` at index `d - 1`: their sums
    /// are `n_draft_total` and `n_draft_accepted_total`.
    pub draft_depths: Vec<Depth>,
    /// The decode rounds by width, width `w` (the slots a round stepped or
    /// passed) at index `w - 1`.
    pub rounds: Vec<Round>,
    /// The engine's waits since the server began serving, every call's
    /// (prompts, rounds, the cache's), as last read: at each round's end,
    /// each request's, each slot action's and each pass of the engine
    /// thread's loop ([`Worker::publish_waits`]).
    pub waits: Waits,
    /// Slots that take the engine in turns: the switches that moved a state
    /// (a snapshot parked, a parked state put back or handed back to its
    /// slot's request) and their wall time; a handed-back state's put back
    /// is its request's (`cache_ms`).
    pub n_swaps_total: u64,
    pub t_swap_ms_total: f64,
    /// Requests refused because the running request's state could not be
    /// parked.
    pub n_swap_refused_total: u64,
    /// Under [`Park::Ids`]: the positions fed again at a parked request's
    /// turn, and the wall time of those calls.
    pub n_reprefill_total: u64,
    pub t_reprefill_ms_total: f64,
    /// The parked states' bytes now, and the most they may hold (the
    /// [`Park::States`] budget; 0 under [`Park::Ids`], which parks no state).
    pub parked_bytes: u64,
    pub park_budget: u64,
}

impl Stats {
    /// The prompt throughput since start, tokens a second over the prompts'
    /// time: `/metrics`' `prompt_tokens_seconds`.
    pub(crate) fn prompt_rate(&self) -> f64 {
        rate(self.n_prompt_total, self.t_prompt_ms_total)
    }

    /// The generation throughput since start: `/metrics`'
    /// `predicted_tokens_seconds`.
    pub(crate) fn decode_rate(&self) -> f64 {
        rate(self.n_predicted_total, self.t_predicted_ms_total)
    }
}

/// The decode rounds of one width: how many, their wall time on the engine
/// thread (the engine's calls and the server's work between them), and the
/// engine's waits inside them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Round {
    pub n: u64,
    pub ms: f64,
    pub waits: Waits,
}

/// `n` tokens over `ms` milliseconds as tokens a second; 0 before any. The
/// ratio of two counters, so reading it never changes it.
pub(crate) fn rate(n: u64, ms: f64) -> f64 {
    if n > 0 && ms > 0.0 {
        1e3 / ms * n as f64
    } else {
        0.0
    }
}

/// The seconds of generation between a request's stderr progress lines.
const NOTE_S: f64 = 30.0;

/// What the server's per-request stderr lines name a request by: its own
/// task id (what `/slots` shows as `id_task`; each choice has its own), the
/// path it was posted to, and its choice's index when the request asked for
/// more than one choice.
pub(crate) struct ReqName {
    pub id: u64,
    pub route: String,
    pub choice: usize,
    pub choices: usize,
}

/// A request handed to the engine thread.
pub(crate) struct Submit {
    pub p: GenParams,
    /// What `/slots` shows of it.
    pub prompt: Value,
    pub settings: Value,
    /// The prompt's images, in the order of their spans: each rides the
    /// prompt call that holds its span.
    pub media: Vec<MediaFeed>,
    /// The sampler the request draws with, `None` for a greedy one
    /// ([`crate::genloop::sampler_of`]).
    pub sampler: Option<Sampler>,
    pub events: Events,
    pub client: Arc<crate::api::Client>,
    /// What the server's stderr lines name the request by.
    pub name: ReqName,
}

/// A request's side of its HTTP thread's channel. The choices of one request
/// share the channel, each message tagged with its choice.
pub(crate) struct Events {
    tx: mpsc::Sender<(usize, Msg)>,
    choice: usize,
    /// The request's key place, held while the engine has the job.
    lease: Option<Lease>,
}

impl Events {
    pub(crate) fn new(
        tx: mpsc::Sender<(usize, Msg)>,
        choice: usize,
        lease: Option<Lease>,
    ) -> Events {
        Events { tx, choice, lease }
    }

    /// Sends the job's last message, its place given back first.
    fn last(mut self, m: Msg) {
        drop(self.lease.take());
        let _ = self.send(m);
    }

    /// Fails, as a gone client, once the HTTP thread dropped its end.
    pub(crate) fn send(&self, m: Msg) -> io::Result<()> {
        self.tx.send((self.choice, m)).map_err(|_| client_gone())
    }
}

/// What the engine thread tells a request's HTTP thread.
pub(crate) enum Msg {
    /// The request took this slot; sent before any other message.
    Started(usize),
    Prompt(Timings),
    /// Text and where its special tokens start ([`Event::Text`]).
    Text(String, Vec<(usize, usize)>, Timings),
    /// The last message.
    Done(Result<Outcome, GenError>),
    /// The request took its slot but did not start: the state of the request
    /// on the engine could not be parked. Its one message after
    /// [`Msg::Started`]; the request may succeed later as it is.
    Refused(String),
}

/// A slot action run on the engine thread: `run` on the reserved slot
/// (selected first for one slot), whose `reply` is sent once the slots are
/// released, and the engine error it met, if any, which is fatal.
pub(crate) struct Action {
    pub run: Box<dyn FnOnce(&mut Slot) -> Acted + Send>,
    /// It drops its slot's state and reads none ([`Slot::erase`]): when the
    /// slots take turns and the engine holds another slot's state, it runs
    /// where its slot's state lies and the engine stays as it is.
    pub drops: bool,
}

pub(crate) struct Acted {
    pub failure: Option<EngineError>,
    pub reply: Box<dyn FnOnce() + Send>,
}

/// What the HTTP threads and the engine thread share.
pub(crate) struct Shared {
    pub board: Mutex<Board<Submit, Action>>,
    /// Signalled when the board has a request or an action for the engine
    /// thread.
    pub work: Condvar,
    pub stats: Mutex<Stats>,
    /// The requests that ended, newest last: the live view's list
    /// ([`crate::live`]), kept where their counters are booked.
    pub done: Mutex<Ring>,
    /// The first failure that ended the server ([`Shared::record_fatal`]);
    /// the reason `/health` gives.
    pub fatal: Mutex<Option<String>>,
    /// The orderly stop's cause, once it began ([`Shared::begin_stop`]): the
    /// reason every request that needs the engine is refused with.
    pub stopping: Mutex<Option<String>>,
    /// The orderly stop's order to the engine thread: read between engine
    /// calls ([`Worker::run`], [`Worker::run_turns`]), it ends the thread.
    stop: AtomicBool,
    /// The engine thread left its loop ([`worker::serve`], on every path that
    /// ends its run: the stop's [`Worker::halt`], an engine failure, a caught
    /// panic): what the orderly stop waits for first ([`Shared::wait_stop`]).
    /// Past it the thread calls the engine no more and never drops it.
    loop_ended: Mutex<bool>,
    /// The server's per-request lines, collected in the tests' place of
    /// stderr.
    #[cfg(test)]
    stderr_lines: Mutex<Vec<String>>,
    /// Wakes [`Shared::wait_loop_end`].
    loop_ended_c: Condvar,
    /// The answers the connections are writing ([`crate::api::keep_alive`]
    /// counts them): what the orderly stop waits for once the loop left, so
    /// the answers the halt ended reach their clients before the process
    /// exits.
    pub answers: Answers,
    /// Each slot's turn, for an engine whose slots take turns; `None` for any
    /// other.
    pub turns: Mutex<Option<Vec<Turn>>>,
    pub end: mpsc::Sender<End>,
    pub sampler: SamplerFactory,
    ids: AtomicU64,
}

impl Shared {
    pub(crate) fn new(
        board: Board<Submit, Action>,
        end: mpsc::Sender<End>,
        sampler: SamplerFactory,
    ) -> Shared {
        Shared {
            board: Mutex::new(board),
            work: Condvar::new(),
            stats: Mutex::new(Stats::default()),
            done: Mutex::new(Ring::default()),
            fatal: Mutex::new(None),
            stopping: Mutex::new(None),
            stop: AtomicBool::new(false),
            loop_ended: Mutex::new(false),
            loop_ended_c: Condvar::new(),
            answers: Answers::default(),
            turns: Mutex::new(None),
            end,
            sampler,
            ids: AtomicU64::new(0),
            #[cfg(test)]
            stderr_lines: Mutex::new(Vec::new()),
        }
    }

    /// One of the server's per-request stderr lines. The tests collect it
    /// instead: the engine thread's stderr is beyond a test's capture.
    fn note(&self, line: String) {
        #[cfg(test)]
        relock(&self.stderr_lines).push(line);
        #[cfg(not(test))]
        eprintln!("{line}");
    }

    /// The per-request lines, in the order they were printed.
    #[cfg(test)]
    pub(crate) fn notes(&self) -> Vec<String> {
        relock(&self.stderr_lines).clone()
    }

    /// Adopts `f` as the failure that ends the server unless one already
    /// was: the first is kept. `/health` and every later request see its
    /// error, and the waiting requests' and actions' channels close on it,
    /// so their HTTP threads answer with it. `Some(f)` when `f` was adopted,
    /// which [`Shared::end_on`] then sends once the requests that met it have
    /// their answers; `None` when an earlier failure was, whose own sender
    /// ends the server.
    #[must_use]
    pub(crate) fn record_fatal(&self, f: EngineFailure) -> Option<EngineFailure> {
        let mut fatal = relock(&self.fatal);
        if fatal.is_some() {
            return None;
        }
        *fatal = Some(f.error.clone());
        drop(relock(&self.board).kill(&f.error));
        Some(f)
    }

    /// Ends the server on the failure [`Shared::record_fatal`] adopted;
    /// nothing for one it did not.
    pub(crate) fn end_on(&self, adopted: Option<EngineFailure>) {
        if let Some(f) = adopted {
            let _ = self.end.send(End::Engine(f));
        }
    }

    pub(crate) fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::SeqCst)
    }

    /// Whether the engine thread was told to stop.
    pub(crate) fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// The engine thread left its loop ([`Worker::run`] and
    /// [`Worker::run_turns`] returned): what the orderly stop waits for ends
    /// here. Called on every path that ends the thread's run; the engine it
    /// holds is the process exit's ([`serve`]).
    pub(crate) fn end_loop(&self) {
        *relock(&self.loop_ended) = true;
        self.loop_ended_c.notify_all();
    }

    /// Whether the engine thread left its loop within `bound`.
    pub(crate) fn wait_loop_end(&self, bound: Duration) -> bool {
        let left = relock(&self.loop_ended);
        let (left, _) = self
            .loop_ended_c
            .wait_timeout_while(left, bound, |left| !*left)
            .unwrap_or_else(|e| e.into_inner());
        *left
    }

    /// The orderly stop's wait, inside one `bound`: the engine thread leaves
    /// its loop ([`Shared::wait_loop_end`]) — the halt closed every request's
    /// channel by then ([`Worker::halt`]) — and then every answer in flight
    /// is written, the halted requests' last events among them.
    pub(crate) fn wait_stop(&self, bound: Duration) -> Waited {
        let until = Instant::now() + bound;
        if !self.wait_loop_end(bound) {
            return Waited::Loop;
        }
        self.answers.wait(until)
    }

    /// Begins the server's orderly stop, the one owner `POST /shutdown` and
    /// the first signal go through: from here no request that needs the
    /// engine is admitted (it gets a 503 naming `cause`), and the engine
    /// thread ends between its engine calls. The first cause wins; `false`
    /// when a stop was already under way.
    pub(crate) fn begin_stop(&self, cause: &Stop) -> bool {
        {
            let mut gate = relock(&self.stopping);
            if gate.is_some() {
                return false;
            }
            *gate = Some(cause.to_string());
        }
        // The engine thread reads `stop` and then waits on `work` with the
        // board locked ([`Worker::run`], [`Worker::run_turns`]): stored and
        // notified under that lock, the stop lands before its read or finds
        // it waiting, never between the two.
        let _board = relock(&self.board);
        self.stop.store(true, Ordering::SeqCst);
        self.work.notify_one();
        true
    }
}

/// The engine thread failed; nothing more runs.
struct Dead;

struct Active {
    slot: usize,
    job: Gen,
    events: Events,
    /// What the server's stderr lines name the request by.
    name: ReqName,
    /// The whole [`NOTE_S`] intervals of generation its last progress line
    /// counted; the next line waits for one more.
    noted: usize,
    /// The engine call the request needs next, taken from its generation
    /// by a pump that ran ahead of the call ([`Worker::hold_need`]).
    need: Option<Need>,
    /// Under turns: when it last left the engine (the turns' clock).
    left: u64,
    /// Under turns: the admission it took its slot in.
    batch: u64,
}

/// A request that took a slot and waits for its prompt's turn.
struct Pending {
    slot: usize,
    held: Held,
    sub: Submit,
    /// The admission it took its slot in: requests of one admission start
    /// one after another at turn boundaries, a later one's at a step.
    batch: u64,
}

/// The engine thread's part of slots that take the engine in turns.
struct Turns {
    table: ParkTable,
    /// The slot whose state the engine holds: admission's choice among slots
    /// that share as much of a prompt.
    on: usize,
    /// The slot of the request on the engine, if one is.
    running: Option<usize>,
    /// The running request's generated tokens when its turn began.
    turn_from: usize,
    /// Requests that took slots and wait for their prompts, shortest first.
    pending: Vec<Pending>,
    /// Slot actions reserved while a request was live.
    deferred: Vec<(Reserve, Action)>,
    /// A turn's end could not park the running request: the next turns keep
    /// it on the engine until a request ends or starts, which changes what the
    /// table holds.
    stuck: bool,
    clock: u64,
    /// Admissions so far.
    batches: u64,
}

/// What a move of the engine to another slot did.
enum Switched {
    Done,
    /// The live request on the engine could not be parked: nothing moved.
    Refused(String),
    /// The target's live request lost its parked state: the engine holds
    /// nothing for it.
    Lost(String),
}

/// A request whose prompt runs a call a round ([`Prompt`]).
struct Prompting {
    a: Active,
    plan: Prompt,
}

struct Worker {
    slot: Slot,
    sh: Arc<Shared>,
    /// The requests running, in the order they took their slots.
    active: Vec<Active>,
    /// The requests whose prompts run a call a round, in the order they took
    /// their slots; none is in a decode round until its prompt ends.
    prompting: Vec<Prompting>,
    /// The engine's prompt quantum; `None` when its slots take it in turns,
    /// where it holds one request's sequence at a time.
    quantum: Option<NonZeroUsize>,
    turns: Option<Turns>,
    /// The slots whose requests ended since the round's last seal
    /// ([`Worker::seal_ended`]).
    ended: Vec<usize>,
    /// The engine's waits when the thread began serving: `/metrics`' wait
    /// totals count from it.
    waits_from: Waits,
}

/// Runs the engine thread until the engine fails, the thread panics or the
/// server stops. The loop's end is the server's, and the process exit
/// reclaims the engine: the thread never drops it, nor the slots' states
/// the engine saved. Their Drop frees through the device driver, and a
/// driver call made while the exit tears the driver's libraries down can
/// corrupt the process's heap. What the Drop would tell other processes
/// the thread tells them itself, before the loop's end is published
/// ([`crate::Engine::close`]).
pub(crate) fn serve(slot: Slot, sh: Arc<Shared>) {
    let n = relock(&sh.board).slots().len();
    let turns = slot.engine.turns().map(|park| {
        *relock(&sh.turns) = Some(vec![Turn::Idle; n]);
        relock(&sh.stats).park_budget = match park {
            Park::States { budget } => budget,
            Park::Ids => 0,
        };
        Turns {
            table: ParkTable::new(park, n),
            on: 0,
            running: None,
            turn_from: 0,
            pending: Vec::new(),
            deferred: Vec::new(),
            stuck: false,
            clock: 0,
            batches: 0,
        }
    });
    let quantum = if turns.is_some() {
        None
    } else {
        slot.engine.prompt_quantum()
    };
    let waits_from = slot.engine.waits();
    let mut w = Worker {
        slot,
        sh: Arc::clone(&sh),
        active: Vec::new(),
        prompting: Vec::new(),
        quantum,
        turns,
        ended: Vec::new(),
        waits_from,
    };
    let ran = panic::catch_unwind(AssertUnwindSafe(|| {
        if w.turns.is_some() {
            w.run_turns();
        } else {
            w.run();
        }
    }));
    let adopted = match ran {
        Ok(()) => None,
        Err(p) => sh.record_fatal(EngineFailure {
            engine: "unknown: its thread panicked".to_owned(),
            error: format!("the engine thread panicked: {}", panic_text(&*p)),
        }),
    };
    // What the thread still holds is answered with what ended the loop, then
    // the engine closed: the processes serving part of the load told it
    // ended, the records of its last requests printed.
    w.close();
    if let Err(e) = w.slot.engine.close() {
        eprintln!("bloomery-serve: closing the engine: {e}");
    }
    // The stop's wait on the loop ends here: past this point the thread
    // calls the engine no more, and the engine and the slots' saved states
    // stay for the process exit.
    sh.end_loop();
    std::mem::forget(w);
    sh.end_on(adopted);
}

/// `a`'s call of `need` on its slot alone, the round's only call of it:
/// a select, then the step, the drafted pass or the sampled pass.
fn call_alone(slot: &mut Slot, a: &mut Active, need: Need) -> Result<(), EngineError> {
    slot.select(a.slot)?;
    match need {
        Need::Step(t) => step_alone(slot, a, t),
        Need::Advance(t) => slot.advance(t, a.job.kept_mut()).map(|d| a.job.advanced(d)),
        Need::Sampled(t) => a.job.advance_sampled(slot, t),
        Need::Fill => unreachable!("a round of fills alone makes no decode call"),
        Need::Done => unreachable!("a finished request left the running ones"),
    }
}

/// `a`'s step of `t` on its slot alone: on an engine of one slot, a request
/// that feeds the answer back unless it stops ([`Gen::step_ahead`]) lets the
/// engine launch the answer's step before the answer is read back
/// ([`Slot::next_ahead`]); any other a plain step. Slots that take the engine
/// in turns step plainly: a turn may end between two steps.
fn step_alone(slot: &mut Slot, a: &mut Active, t: u32) -> Result<(), EngineError> {
    let g = match a.job.step_ahead() {
        Some(ahead) => slot.next_ahead(t, ahead && slot.slots() == 1)?,
        None => slot.next(t, a.job.logits_out())?,
    };
    a.job.stepped(g);
    Ok(())
}

/// The sink a request's generation writes to: its channel.
fn sink_of(events: &Events) -> impl FnMut(Event<'_>) -> io::Result<()> + '_ {
    |ev| {
        let m = match ev {
            Event::Prompt(t) => Msg::Prompt(t.clone()),
            Event::Text(text, specials, t) => {
                Msg::Text(text.to_owned(), specials.to_vec(), t.clone())
            }
        };
        events.send(m)
    }
}

/// The tick a request's generation calls after every token: its slot's view,
/// and a progress line on the server's stderr once its generation passed
/// another [`NOTE_S`] seconds. The due check is the generation's own clock
/// against the lines printed: no lock, no allocation and no formatting but
/// the line's own.
fn tick_of<'a>(
    sh: &'a Shared,
    slot: usize,
    name: &'a ReqName,
    noted: &'a mut usize,
) -> impl FnMut(&Timings) + 'a {
    move |t| {
        if note_due(*noted, t) {
            *noted = intervals(t);
            sh.note(progress_line(name, t));
        }
        let mut b = relock(&sh.board);
        let v = b.view_mut(slot);
        v.n_decoded = t.predicted_n;
        v.n_past = t.n_past;
    }
}

/// Whether the generation crossed its next progress line: one every
/// [`NOTE_S`] seconds of generation, counted from the generation's start, so
/// a generation that stalls skips to the next multiple rather than printing
/// a run of lines when it resumes.
fn note_due(noted: usize, t: &Timings) -> bool {
    intervals(t) > noted
}

/// The whole [`NOTE_S`] intervals the generation has run.
fn intervals(t: &Timings) -> usize {
    (t.predicted_ms / (NOTE_S * 1e3)).floor() as usize
}

/// `n` ids over `ms` as a rate, one decimal; `-` over no time.
fn ids_rate(n: usize, ms: f64) -> String {
    if ms > 0.0 {
        format!("{:.1}", rate(n as u64, ms))
    } else {
        "-".to_owned()
    }
}

/// One phase of a request as its stderr lines print it: `<n> ids[ (<reused>
/// reused)] in <s> s, <rate> ids/s`, the seconds to two decimals.
fn phase(n: usize, ms: f64, reused: Option<usize>) -> String {
    let reuse = reused.map_or(String::new(), |c| format!(" ({c} reused)"));
    format!(
        "{n} ids{reuse} in {:.2} s, {} ids/s",
        ms / 1e3,
        ids_rate(n, ms)
    )
}

/// The request's own part of its stderr lines: `<id> <path>`, the choice's
/// index after the id for one choice of several.
fn req_tag(name: &ReqName) -> String {
    if name.choices > 1 {
        format!("{}.{} {}", name.id, name.choice, name.route)
    } else {
        format!("{} {}", name.id, name.route)
    }
}

/// The finish line, the one stderr line a request's end prints: its prompt
/// and generated phases, its stop cause, and where its context stands.
fn finish_line(name: &ReqName, t: &Timings, cause: &str) -> String {
    format!(
        "bloomery-serve: request {}: prompt {}; generated {}; stop {cause}; ctx {}/{}",
        req_tag(name),
        phase(t.prompt_n, t.prompt_ms, Some(t.cache_n)),
        phase(t.predicted_n, t.predicted_ms, None),
        t.n_past,
        t.n_ctx
    )
}

/// The progress line, while one request generates past an interval.
fn progress_line(name: &ReqName, t: &Timings) -> String {
    format!(
        "bloomery-serve: request {}: generating, {}; ctx {}/{}",
        req_tag(name),
        phase(t.predicted_n, t.predicted_ms, None),
        t.n_past,
        t.n_ctx
    )
}

impl Worker {
    /// The rounds: the actions, the starts, one call of each planned prompt
    /// and one decode round. The stop is read before each of the round's
    /// actions, starts and planned calls and before its decode round, so the
    /// thread leaves within the one the stop arrives in (a start runs its
    /// prompt in one go unless it is planned); what the round had not run
    /// yet ends with the halt.
    fn run(&mut self) {
        loop {
            let (actions, admitted) = {
                let mut b = relock(&self.sh.board);
                loop {
                    let actions = b.take_actions();
                    let admitted = b.admit(|i| self.slot.held_of(i), None);
                    if !actions.is_empty()
                        || !admitted.is_empty()
                        || !self.active.is_empty()
                        || !self.prompting.is_empty()
                        || self.sh.stopped()
                    {
                        break (actions, admitted);
                    }
                    b = self.sh.work.wait(b).unwrap_or_else(|e| e.into_inner());
                }
            };
            for (what, a) in actions {
                if self.sh.stopped() {
                    return self.halt();
                }
                if self.act(what, a, false).is_err() {
                    return;
                }
            }
            for (slot, held, sub) in admitted {
                if self.sh.stopped() {
                    return self.halt();
                }
                if self.start(slot, &held, sub).is_err() {
                    return;
                }
                self.pump_ready();
            }
            if self.prompts().is_err() {
                return;
            }
            if self.sh.stopped() {
                return self.halt();
            }
            if self.step().is_err() {
                return;
            }
            self.pump_ready();
            if self.seal_ended().is_err() {
                return;
            }
            self.publish_waits();
        }
    }

    /// The slots whose requests ended this round hand their states to the
    /// prompt cache ([`Slot::seal`]): an engine that copies on a stream of
    /// its own begins each copy here, after the round's step and before any
    /// request is admitted to the slot, so the next request that cuts the
    /// slot finds its state saved. An engine failure is fatal.
    fn seal_ended(&mut self) -> Result<(), Dead> {
        for slot in std::mem::take(&mut self.ended) {
            if self.sh.stopped() {
                break;
            }
            if let Err(e) = self.slot.seal(slot) {
                let f = self.fail(&e);
                return Err(self.end(f));
            }
        }
        Ok(())
    }

    /// Records an engine failure ([`Shared::record_fatal`]): unless an
    /// earlier failure ended the server, `/health` and every later request
    /// see it, and what waits is refused with it.
    fn fail(&mut self, e: &EngineError) -> Option<EngineFailure> {
        self.sh.record_fatal(EngineFailure {
            engine: self.slot.engine.describe(),
            error: e.to_string(),
        })
    }

    /// Ends the server once the requests that met the failure have their
    /// answers.
    fn end(&self, adopted: Option<EngineFailure>) -> Dead {
        self.sh.end_on(adopted);
        Dead
    }

    /// The engine thread's orderly end, what [`Shared::begin_stop`] ordered,
    /// without touching the engine again. Every request and action the
    /// thread holds or the board queues has its channel closed here, before
    /// the loop's end ([`Shared::end_loop`]) the stop waits for: the queued
    /// ones, the running ones, those between their prompt's calls, and under
    /// turns the parked, the pending and the deferred actions. Each HTTP
    /// thread then answers with the stop's cause ([`crate::api::engine_gone`]),
    /// the queued requests' answer: a stream already open ends with its error
    /// event, any other request is the 503. The requests that ran have their
    /// counters booked, as an ended request's are; their slots stay as they
    /// stood, so `/health` and `/slots` show the server as the stop found it.
    /// Nothing failed: no engine error is recorded, and no end is sent — the
    /// stop's sender already did.
    fn halt(&mut self) {
        let cause = relock(&self.sh.stopping).clone().expect(
            "the stop's cause: Shared::begin_stop sets it before it stores the stop flag this \
             halt was read from",
        );
        drop(relock(&self.sh.board).kill(&cause));
        let held = self
            .active
            .iter()
            .chain(self.prompting.iter().map(|p| &p.a));
        for a in held {
            self.book(a, Ending::Stopped, None);
        }
        self.close();
    }

    /// Drops every request and action the thread holds, which closes their
    /// channels: the running ones, those between their prompt's calls, and
    /// under turns the pending ones and the deferred actions. Each HTTP thread
    /// then answers with what ended the loop ([`crate::api::engine_gone`]).
    fn close(&mut self) {
        self.active.clear();
        self.prompting.clear();
        if let Some(t) = &mut self.turns {
            t.pending.clear();
            t.deferred.clear();
        }
    }

    /// `off`: the action drops its slot's state while the engine holds
    /// another's ([`Action::drops`]); the slot is selected off the engine.
    fn act(&mut self, what: Reserve, a: Action, off: bool) -> Result<(), Dead> {
        if let Reserve::One(i) = what
            && let Err(e) = if off {
                self.slot.select_off(i)
            } else {
                self.slot.select(i)
            }
        {
            let f = self.fail(&e);
            return Err(self.end(f));
        }
        let acted = (a.run)(&mut self.slot);
        self.publish_waits();
        if let Some(e) = acted.failure {
            let f = self.fail(&e);
            (acted.reply)();
            return Err(self.end(f));
        }
        {
            let mut b = relock(&self.sh.board);
            match what {
                Reserve::One(i) => {
                    b.view_mut(i).n_past = self.slot.held_of(i).len();
                    b.release(i);
                }
                Reserve::All => {
                    for i in 0..b.slots().len() {
                        b.release(i);
                    }
                }
            }
        }
        (acted.reply)();
        Ok(())
    }

    /// A request that took `slot`: its prompt (`held`, with its images'
    /// feeds), run now, or planned as calls on the engine's quantum that
    /// [`Worker::prompts`] runs one a round ([`Worker::interleaves`]). A
    /// request whose client left is cancelled before anything runs, here or,
    /// when the slots take turns, in [`Worker::begin`] before the engine
    /// moves. Returns whether it runs on (an error ended it otherwise).
    fn start(&mut self, slot: usize, held: &Held, sub: Submit) -> Result<bool, Dead> {
        if self.turns.is_none()
            && (sub.client.check().is_err() || sub.events.send(Msg::Started(slot)).is_err())
        {
            relock(&self.sh.stats).n_cancelled_before_prompt_total += 1;
            relock(&self.sh.board).release(slot);
            return Ok(false);
        }
        let Submit {
            p,
            prompt,
            settings,
            media,
            sampler,
            events,
            name,
            client: _,
        } = sub;
        *relock(&self.sh.board).view_mut(slot) = SlotView {
            id_task: name.id,
            prompt,
            settings,
            n_predict: p.n_predict,
            ..SlotView::default()
        };
        // The turns change with the view: `/slots` never shows the request
        // running with the counts of the one before it on the slot.
        if self.turns.is_some() {
            self.show_turns();
        }
        let n = held.ids.len();
        let mut job = Gen::new(&self.slot, sampler, n, &p);
        let kept = match self.slot.select(slot) {
            Ok(()) => job.open(&mut self.slot, held, &p),
            Err(e) => Err(GenError::Engine(e)),
        };
        let r = match kept {
            Ok(kept) => match self.interleaves(n + job.lead_len(), kept.cache_n) {
                Some(q) => job
                    .plan(
                        &self.slot,
                        held.ids.clone(),
                        media,
                        kept,
                        q,
                        p.return_progress,
                    )
                    .map(Some),
                None => job
                    .prompt_whole(
                        &mut self.slot,
                        &held.ids,
                        &media,
                        kept,
                        &p,
                        &mut sink_of(&events),
                    )
                    .map(|()| None),
            },
            Err(e) => Err(e),
        };
        let a = Active {
            slot,
            job,
            events,
            name,
            noted: 0,
            need: None,
            left: 0,
            batch: 0,
        };
        match r {
            Ok(None) => {
                self.active.push(a);
                Ok(true)
            }
            Ok(Some(plan)) => {
                self.prompting.push(Prompting { a, plan });
                Ok(true)
            }
            Err(GenError::Engine(e)) => {
                let f = self.fail(&e);
                self.end_request(a, Err(GenError::Engine(e)));
                Err(self.end(f))
            }
            Err(e) => {
                self.end_request(a, Err(e));
                Ok(false)
            }
        }
    }

    /// The quantum a prompt call of `n` ids (the prompt and the answer's lead
    /// the call carries) whose cache kept `cache_n` runs in, one call a round:
    /// the engine names one, another slot is busy generating, and more than
    /// one quantum of the call (`cache_n..n-1`) is left.
    fn interleaves(&self, n: usize, cache_n: usize) -> Option<NonZeroUsize> {
        self.quantum
            .filter(|q| !self.active.is_empty() && n - 1 - cache_n > q.get())
    }

    /// One call of each prompt run a call a round, in the order they took
    /// their slots. It runs after the starts and before the decode round
    /// ([`Worker::step`]), so a prompt's consecutive calls have a round of
    /// the busy slots between them. A prompt whose last call ran joins the
    /// running requests; a request whose client left ends, its slot freed;
    /// an engine error is fatal, as at a start. A stop read before a call
    /// leaves the calls not yet run to the halt ([`Worker::run`]).
    fn prompts(&mut self) -> Result<(), Dead> {
        let mut k = 0;
        while k < self.prompting.len() {
            if self.sh.stopped() {
                return Ok(());
            }
            let Prompting { a, plan } = &mut self.prompting[k];
            let r = match self.slot.select(a.slot) {
                Ok(()) => a
                    .job
                    .prompt_call(&mut self.slot, plan, &mut sink_of(&a.events)),
                Err(e) => Err(GenError::Engine(e)),
            };
            let slot = a.slot;
            match r {
                Ok(more) => {
                    // `/slots` shows the prompt's positions as its calls run.
                    relock(&self.sh.board).view_mut(slot).n_past = self.slot.held_of(slot).len();
                    if more {
                        k += 1;
                    } else {
                        let p = self.prompting.remove(k);
                        self.active.push(p.a);
                        self.pump_ready();
                    }
                }
                Err(GenError::Engine(e)) => {
                    let f = self.fail(&e);
                    let p = self.prompting.remove(k);
                    self.end_request(p.a, Err(GenError::Engine(e)));
                    return Err(self.end(f));
                }
                Err(e) => {
                    let p = self.prompting.remove(k);
                    self.end_request(p.a, Err(e));
                }
            }
        }
        Ok(())
    }

    /// Takes every running request's tokens to its next engine call, and makes
    /// that call: one slot's alone, or every slot's in one.
    fn step(&mut self) -> Result<(), Dead> {
        let mut needs = Vec::with_capacity(self.active.len());
        let mut k = 0;
        while k < self.active.len() {
            let Some(need) = self.take_need(k) else {
                continue;
            };
            match need {
                Need::Fill => {
                    // A call of its own, like a prompt's: cut on the
                    // quantum's grid beside other busy slots. The request
                    // steps next round.
                    let quantum = self.quantum.filter(|_| self.active.len() > 1);
                    let a = &mut self.active[k];
                    let called = self
                        .slot
                        .select(a.slot)
                        .and_then(|()| a.job.fill(&mut self.slot, quantum));
                    if let Err(e) = called {
                        return Err(self.die(&e));
                    }
                    // `/slots` shows the run's positions as its calls run.
                    tick_of(&self.sh, a.slot, &a.name, &mut a.noted)(a.job.timings());
                    if !self.hold_need(k) {
                        continue;
                    }
                    needs.push(Need::Fill);
                    k += 1;
                }
                need => {
                    needs.push(need);
                    k += 1;
                }
            }
        }
        let width = needs.iter().filter(|n| !matches!(n, Need::Fill)).count();
        if width == 0 {
            return Ok(());
        }
        let (started, before) = (Instant::now(), self.slot.engine.waits());
        let called = self.call_round(needs);
        self.book_round(width, started, before);
        let Err(e) = called else {
            return Ok(());
        };
        Err(self.die(&e))
    }

    /// The engine calls of one decode round, `needs` the running requests'
    /// in `active`'s order (a fill's already run): one slot's alone, or
    /// every slot's in one call of each kind. An orderly stop read between
    /// the two calls ends the round with no error.
    fn call_round(&mut self, mut needs: Vec<Need>) -> Result<(), EngineError> {
        if let [a] = self.active.as_mut_slice() {
            let need = needs.pop().expect("one running request, one need");
            call_alone(&mut self.slot, a, need)
        } else {
            // One engine round, its rows split by what they need: the steps
            // of two slots or more in one call (one step is a select and a
            // `next`, as `step_slots`' doc requires), then the drafted passes
            // of two slots or more in one (one pass is a select and an
            // `advance`, as `advance_slots`' doc requires: the engine's pass
            // of one slot is the pass a request alone makes). Within each call
            // the rows keep `active`'s order. A sampled request's pass rides
            // the drafted call with its draw (`Gen::pass_row`), its ids the
            // ones its pass alone draws.
            let mut steps: Vec<SlotRow<'_>> = Vec::new();
            let mut step_at: Vec<usize> = Vec::new();
            let mut pass_at: Vec<usize> = Vec::new();
            for (i, (a, need)) in self.active.iter_mut().zip(&needs).enumerate() {
                match *need {
                    Need::Step(last) => {
                        step_at.push(i);
                        steps.push(SlotRow {
                            slot: a.slot,
                            last,
                            logits: a.job.logits_out(),
                            next: 0,
                        });
                    }
                    Need::Advance(last) => {
                        pass_at.push(a.slot);
                        a.need = Some(Need::Advance(last));
                    }
                    Need::Sampled(last) => {
                        pass_at.push(a.slot);
                        a.need = Some(Need::Sampled(last));
                    }
                    Need::Fill => {}
                    Need::Done => unreachable!("a finished request left the running ones"),
                }
            }
            let mut called = if steps.len() > 1 {
                self.slot.step_slots(&mut steps)
            } else if let [row] = steps.as_mut_slice() {
                self.slot
                    .select(row.slot)
                    .and_then(|()| self.slot.next(row.last, row.logits.as_deref_mut()))
                    .map(|g| row.next = g)
            } else {
                Ok(())
            };
            if called.is_ok() {
                let answers: Vec<u32> = steps.iter().map(|r| r.next).collect();
                drop(steps);
                let mixed = !step_at.is_empty() && !pass_at.is_empty();
                for (i, g) in step_at.into_iter().zip(answers) {
                    self.active[i].job.stepped(g);
                }
                if mixed {
                    self.pump_ready();
                    if self.sh.stopped() {
                        return Ok(());
                    }
                }
                // Pumping may end a request and move the remaining entries.
                let mut uncalled: Vec<(usize, Need)> = Vec::with_capacity(pass_at.len());
                let mut active = self.active.iter_mut().enumerate();
                for at in &pass_at {
                    let (i, a) = active
                        .find(|(_, a)| a.slot == *at)
                        .expect("uncalled passes survive pumping in their original order");
                    let need = a
                        .need
                        .take()
                        .expect("an original pass keeps its uncalled need");
                    uncalled.push((i, need));
                }
                called = match uncalled.len() {
                    0 => Ok(()),
                    1 => {
                        let (i, need) = uncalled.pop().expect("one uncalled pass");
                        call_alone(&mut self.slot, &mut self.active[i], need)
                    }
                    _ => {
                        let mut passes: Vec<SlotPass<'_>> = Vec::with_capacity(uncalled.len());
                        for (i, a) in self.active.iter_mut().enumerate() {
                            let Some((_, need)) = uncalled.iter().find(|(at, _)| *at == i) else {
                                continue;
                            };
                            let (Need::Advance(last) | Need::Sampled(last)) = *need else {
                                unreachable!("an uncalled pass is an advance or a sampled pass");
                            };
                            passes.push(a.job.pass_row(a.slot, last));
                        }
                        let called = self.slot.advance_slots(&mut passes);
                        if called.is_ok() {
                            let drafted: Vec<Drafted> = passes.iter().map(|r| r.drafted).collect();
                            drop(passes);
                            for ((i, _), d) in uncalled.into_iter().zip(drafted) {
                                self.active[i].job.advanced(d);
                            }
                        }
                        called
                    }
                };
            }
            called
        }
    }

    /// Books a decode round of `width` slots begun at `started`, when the
    /// engine's waits read `before`: an engine call and its slots, and the
    /// round's time and waits into its width's row. A round counts whatever
    /// its calls returned, as a failed one still ran.
    fn book_round(&self, width: usize, started: Instant, before: Waits) {
        let ms = ms_since(started);
        let waits = self.slot.engine.waits();
        let mut s = relock(&self.sh.stats);
        s.n_decode_total += 1;
        s.n_busy_slots_total += width as u64;
        if s.rounds.len() < width {
            s.rounds.resize(width, Round::default());
        }
        let row = &mut s.rounds[width - 1];
        row.n += 1;
        row.ms += ms;
        row.waits += waits.since(before);
        s.waits = waits.since(self.waits_from);
    }

    /// `/metrics`' wait totals brought up to the engine's after engine work
    /// that no round or request end books (a slot action, a prompt call, a
    /// prompt cache save, a change of hands), so a scrape of an idle server
    /// reads them.
    fn publish_waits(&self) {
        let waits = self.slot.engine.waits().since(self.waits_from);
        relock(&self.sh.stats).waits = waits;
    }

    /// An engine failure in a call every running request waits on: each gets
    /// the error, those between their prompt's calls too, and the server ends.
    fn die(&mut self, e: &EngineError) -> Dead {
        let f = self.fail(e);
        let mut ended = std::mem::take(&mut self.active);
        ended.extend(std::mem::take(&mut self.prompting).into_iter().map(|p| p.a));
        for a in ended {
            self.end_request(a, Err(GenError::Engine(EngineError(e.0.clone()))));
        }
        self.end(f)
    }

    /// Books an ending request's counters, `/metrics`' totals, from its
    /// timings, prints its stderr finish line from the same numbers, and
    /// keeps it with how it ended for the live view; calls no engine (it
    /// reads the engine's waits, the totals' latest) and touches no slot.
    /// The line and the totals cannot disagree: both come from the timings
    /// here, the one place a request's counts are booked.
    fn book(&self, a: &Active, ending: Ending, stop: Option<StopKind>) {
        let (name, t) = (&a.name, a.job.timings());
        let failed = matches!(ending, Ending::EngineError | Ending::Error);
        self.sh.note(finish_line(name, t, cause_word(stop, failed)));
        relock(&self.sh.done).push(Finished {
            at: SystemTime::now(),
            timings: t.clone(),
            ending,
        });
        let waits = self.slot.engine.waits();
        let mut s = relock(&self.sh.stats);
        let (pn, dn) = (t.prompt_n as u64, t.predicted_n as u64);
        s.n_prompt_total += pn;
        s.n_prompt_cached_total += t.cache_n as u64;
        s.n_tokens_max = s.n_tokens_max.max(t.n_past as u64);
        s.t_prompt_ms_total += t.prompt_ms;
        s.t_cache_ms_total += t.cache_ms;
        s.n_predicted_total += dn;
        s.t_predicted_ms_total += t.predicted_ms;
        // The prompt's engine call, whose logits give the first generated
        // token; the steps were booked as they ran.
        if pn > 0 {
            s.n_decode_total += 1;
            s.n_busy_slots_total += 1;
        }
        s.n_draft_total += t.draft_n as u64;
        s.n_draft_accepted_total += t.draft_n_accepted as u64;
        s.n_draft_passes_total += t.draft_passes as u64;
        let depths = a.job.draft_depths();
        if s.draft_depths.len() < depths.len() {
            s.draft_depths.resize(depths.len(), Depth::default());
        }
        for (total, &d) in s.draft_depths.iter_mut().zip(depths) {
            *total += d;
        }
        s.waits = waits.since(self.waits_from);
    }

    /// Books a request's counters and view, releases its slot, and sends its
    /// last message.
    fn end_request(&mut self, a: Active, r: Result<Outcome, GenError>) {
        let t = a.job.timings();
        let ending = match &r {
            Ok(o) => Ending::Done(o.stop.finish_reason()),
            Err(GenError::Client(_)) => Ending::ClientClosed,
            Err(GenError::Engine(_)) => Ending::EngineError,
            Err(_) => Ending::Error,
        };
        let stop = r.as_ref().ok().map(|o| o.stop);
        self.book(&a, ending, stop);
        // An engine failure leaves no state to save; any other end leaves the
        // slot holding what it fed.
        if self.turns.is_none() && !matches!(r, Err(GenError::Engine(_))) {
            self.ended.push(a.slot);
        }
        {
            let mut b = relock(&self.sh.board);
            let v = b.view_mut(a.slot);
            v.n_past = t.n_past;
            v.n_decoded = t.predicted_n;
            if let Ok(o) = &r {
                v.stopped_eos = o.stop == StopKind::Eos;
                v.stopped_word = o.stop == StopKind::Word;
                v.stopped_limit = o.stop == StopKind::Limit;
                v.stopping_word.clone_from(&o.stopping_word);
            }
            b.release(a.slot);
        }
        if let Some(t) = &mut self.turns {
            t.stuck = false;
            self.show_turns();
        }
        a.events.last(Msg::Done(r));
    }
}

/// Slots that take the engine in turns ([`crate::swap`]).
impl Worker {
    fn t(&self) -> &Turns {
        self.turns
            .as_ref()
            .expect("the turns of an engine whose slots take turns")
    }

    fn t_mut(&mut self) -> &mut Turns {
        self.turns
            .as_mut()
            .expect("the turns of an engine whose slots take turns")
    }

    fn index_of(&self, slot: usize) -> Option<usize> {
        self.active.iter().position(|a| a.slot == slot)
    }

    fn run_turns(&mut self) {
        loop {
            let (actions, admitted) = {
                let mut b = relock(&self.sh.board);
                loop {
                    let actions = b.take_actions();
                    let admitted = b.admit(|i| self.slot.held_of(i), Some(self.t().on));
                    let t = self.t();
                    if !actions.is_empty()
                        || !admitted.is_empty()
                        || !self.active.is_empty()
                        || !t.pending.is_empty()
                        || !t.deferred.is_empty()
                        || self.sh.stopped()
                    {
                        break (actions, admitted);
                    }
                    b = self.sh.work.wait(b).unwrap_or_else(|e| e.into_inner());
                }
            };
            if self.sh.stopped() {
                return self.halt();
            }
            let t = self.t_mut();
            t.batches += u64::from(!admitted.is_empty());
            let batch = t.batches;
            let mut new: Vec<Pending> = admitted
                .into_iter()
                .map(|(slot, held, sub)| Pending {
                    slot,
                    held,
                    sub,
                    batch,
                })
                .collect();
            // Requests that took slots together: the shortest prompt first.
            new.sort_by_key(|p| p.held.ids.len());
            let t = self.t_mut();
            t.deferred.extend(actions);
            t.pending.extend(new);
            if self.active.is_empty() && self.t().pending.is_empty() {
                for (what, a) in std::mem::take(&mut self.t_mut().deferred) {
                    let off = match what {
                        Reserve::One(i) if a.drops && i != self.t().on => {
                            self.t_mut().table.take(i);
                            relock(&self.sh.stats).parked_bytes = self.t().table.bytes();
                            Ok(true)
                        }
                        Reserve::One(i) => self.switch(i, false).map(|_| false),
                        Reserve::All => Ok(false),
                    };
                    let Ok(off) = off else { return };
                    if self.act(what, a, off).is_err() {
                        return;
                    }
                }
            }
            self.show_turns();
            if self.turn().is_err() {
                return;
            }
            self.publish_waits();
        }
    }

    /// One step boundary: the running request's tokens in hand go out, the
    /// engine changes hands where a rule says so, and the request on it takes
    /// one engine call — unless the stop was read after the change of hands,
    /// which leaves the call to the halt at the next round's top.
    fn turn(&mut self) -> Result<(), Dead> {
        if let Some(i) = self.t().running.and_then(|r| self.index_of(r)) {
            self.hold_need(i);
        }
        let running = self.t().running;
        let on = running.and_then(|r| self.index_of(r));
        let turn_done = on.is_some_and(|i| {
            self.active[i].job.timings().predicted_n >= self.t().turn_from + QUANTUM
        });
        // A request that came after the running one began preempts it at this
        // step while it runs alone; one of its own admission waits its turn.
        let preempts = self.t().pending.first().is_some_and(|p| {
            self.active.len() <= 1 && on.is_some_and(|i| p.batch > self.active[i].batch)
        });
        if !self.t().pending.is_empty() && (running.is_none() || turn_done || preempts) {
            self.begin()?;
        } else if running.is_none() || turn_done {
            match self.longest_waiting() {
                Some(to) if running.is_none() || !self.t().stuck => self.rotate(to)?,
                // Alone on the engine, or stuck on it: the next turn starts now.
                _ => {
                    let i = running.and_then(|r| self.index_of(r));
                    let n = i.map_or(0, |i| self.active[i].job.timings().predicted_n);
                    self.t_mut().turn_from = n;
                }
            }
        }
        self.show_turns();
        if self.sh.stopped() {
            return Ok(());
        }
        self.step_running()
    }

    /// The live request that left the engine longest ago, the running one
    /// aside.
    fn longest_waiting(&self) -> Option<usize> {
        let running = self.t().running;
        self.active
            .iter()
            .filter(|a| Some(a.slot) != running)
            .min_by_key(|a| a.left)
            .map(|a| a.slot)
    }

    /// The first pending request's prompt: unless its client left, the engine
    /// moves to its slot (a live request's state that cannot be parked refuses
    /// it by name), its prompt runs, and its turn begins. Nothing after the
    /// move cancels it: the move may hand the slot's parked state back, which
    /// only its prompt takes ([`Slot::reuse`]).
    fn begin(&mut self) -> Result<(), Dead> {
        let p = self.t_mut().pending.remove(0);
        if p.sub.client.check().is_err() || p.sub.events.send(Msg::Started(p.slot)).is_err() {
            relock(&self.sh.stats).n_cancelled_before_prompt_total += 1;
            relock(&self.sh.board).release(p.slot);
            self.show_turns();
            return Ok(());
        }
        match self.switch(p.slot, true)? {
            Switched::Refused(why) => {
                relock(&self.sh.stats).n_swap_refused_total += 1;
                relock(&self.sh.board).release(p.slot);
                self.show_turns();
                p.sub.events.last(Msg::Refused(why));
                return Ok(());
            }
            // A pending request has no parked state to lose.
            Switched::Lost(_) | Switched::Done => {}
        }
        let t = self.t_mut();
        t.running = Some(p.slot);
        t.turn_from = 0;
        t.stuck = false;
        if !self.start(p.slot, &p.held, p.sub)? {
            self.t_mut().running = None;
        } else if let Some(i) = self.index_of(p.slot) {
            self.active[i].batch = p.batch;
        }
        Ok(())
    }

    /// The engine to `to`'s live request for its turn. A running request
    /// whose state cannot be parked keeps the engine for another turn; a
    /// parked state that does not resume ends its request by name.
    fn rotate(&mut self, to: usize) -> Result<(), Dead> {
        match self.switch(to, false)? {
            Switched::Refused(_) => {
                let i = self.t().running.and_then(|r| self.index_of(r));
                let n = i.map_or(0, |i| self.active[i].job.timings().predicted_n);
                let t = self.t_mut();
                t.turn_from = n;
                t.stuck = true;
            }
            Switched::Lost(why) => {
                self.t_mut().running = None;
                if let Some(i) = self.index_of(to) {
                    let a = self.active.remove(i);
                    self.end_request(a, Err(GenError::Engine(EngineError(why))));
                }
            }
            Switched::Done => {
                let n = self
                    .index_of(to)
                    .map_or(0, |i| self.active[i].job.timings().predicted_n);
                let t = self.t_mut();
                t.running = Some(to);
                t.turn_from = n;
            }
        }
        Ok(())
    }

    /// Every running request's tokens in hand to its client, each keeping its
    /// next engine call ([`Worker::hold_need`]); a stop read between them
    /// leaves the rest.
    fn pump_ready(&mut self) {
        let mut i = 0;
        while i < self.active.len() && !self.sh.stopped() {
            if self.hold_need(i) {
                i += 1;
            }
        }
    }

    /// Request `i`'s next engine call ([`Worker::take_need`]) kept on it until
    /// that call runs, so the tokens it holds reach its client before the
    /// engine's next call; `false` when it ended (and is booked).
    fn hold_need(&mut self, i: usize) -> bool {
        let Some(need) = self.take_need(i) else {
            return false;
        };
        self.active[i].need = Some(need);
        true
    }

    /// Request `i`'s next engine call: the one it holds from an earlier pump,
    /// else the one its generation gives once the tokens it holds have gone
    /// to its client; `None` when it ended (and is booked).
    fn take_need(&mut self, i: usize) -> Option<Need> {
        if let Some(need) = self.active[i].need.take() {
            return Some(need);
        }
        let a = &mut self.active[i];
        let pumped = a.job.pump(
            &mut sink_of(&a.events),
            &mut tick_of(&self.sh, a.slot, &a.name, &mut a.noted),
        );
        let ended = match pumped {
            Ok(Need::Done) => {
                let mut a = self.active.remove(i);
                let r = a.job.finish(&mut sink_of(&a.events));
                Some((a, r))
            }
            Ok(need) => return Some(need),
            Err(e) => Some((self.active.remove(i), Err(e))),
        };
        if let Some((a, r)) = ended {
            if self
                .turns
                .as_ref()
                .is_some_and(|t| t.running == Some(a.slot))
            {
                self.t_mut().running = None;
            }
            self.end_request(a, r);
        }
        None
    }

    /// The running request's next engine call.
    fn step_running(&mut self) -> Result<(), Dead> {
        let Some(i) = self.t().running.and_then(|r| self.index_of(r)) else {
            return Ok(());
        };
        let Some(need) = self.take_need(i) else {
            return Ok(());
        };
        if matches!(need, Need::Fill) {
            return match self.active[i].job.fill(&mut self.slot, None) {
                Ok(()) => Ok(()),
                Err(e) => Err(self.die(&e)),
            };
        }
        let (started, before) = (Instant::now(), self.slot.engine.waits());
        let a = &mut self.active[i];
        let called = match need {
            Need::Step(t) => self
                .slot
                .next(t, a.job.logits_out())
                .map(|g| a.job.stepped(g)),
            Need::Advance(t) => self
                .slot
                .advance(t, a.job.kept_mut())
                .map(|d| a.job.advanced(d)),
            Need::Sampled(t) => a.job.advance_sampled(&mut self.slot, t),
            Need::Fill | Need::Done => unreachable!("a fill is run above, a finished request left"),
        };
        self.book_round(1, started, before);
        match called {
            Ok(()) => Ok(()),
            Err(e) => Err(self.die(&e)),
        }
    }

    /// The slots whose parked idle states stay through a move to `to`: `to`,
    /// the pending requests' and the deferred actions'.
    fn kept_slots(&self, to: usize) -> Vec<usize> {
        let t = self.t();
        let mut keep = vec![to];
        keep.extend(t.pending.iter().map(|p| p.slot));
        keep.extend(t.deferred.iter().filter_map(|(what, _)| match what {
            Reserve::One(i) => Some(*i),
            Reserve::All => None,
        }));
        keep
    }

    /// Moves the engine from the slot it holds to `to`: the state it leaves
    /// parked as [`crate::swap`] says, then `to`'s parked state put back, its
    /// ids fed again, or the engine emptied. With `prompt` (the request
    /// about to start on `to`), `to`'s idle parked state is handed back to
    /// the slot instead of put back: that request's prompt puts it back only
    /// if it keeps some of it, and hands it to the prompt cache without a
    /// copy where the cache's rule saves it ([`Slot::hand_back`]). A slot
    /// whose idle state leaves the table, or that is left with no room, holds
    /// nothing after, and its `/slots` view says so; the state goes into the
    /// prompt cache first ([`Slot::offer`]), the one that left as it is, the
    /// one left with no room as the snapshot its park took.
    fn switch(&mut self, to: usize, prompt: bool) -> Result<Switched, Dead> {
        let from = self.t().on;
        if from == to {
            return Ok(Switched::Done);
        }
        let t0 = Instant::now();
        let live_from = self.index_of(from).is_some();
        let live_to = self.index_of(to).is_some();
        let keep = self.kept_slots(to);
        let held = self.slot.held_of(from).len();
        let mut voids = Vec::new();
        // The states of the voids the prompt cache takes, and whether each is
        // a snapshot taken here.
        let mut offers: Vec<(usize, Arc<dyn Saved>, bool)> = Vec::new();
        let mut moved = false;
        if held > 0 {
            let refused = match self.t().table.park() {
                Park::Ids if live_from => {
                    self.t_mut().table.put(from, Entry::Ids);
                    moved = true;
                    None
                }
                Park::Ids => Some(String::new()),
                Park::States { .. } => match self.slot.engine.snapshot() {
                    Err(StateError::Engine(e)) => return Err(self.die(&e)),
                    Err(e) => Some(format!("the engine took no snapshot: {e}")),
                    Ok(state) if state.n_tokens() != held => Some(format!(
                        "its snapshot holds {} positions, the slot {held}",
                        state.n_tokens()
                    )),
                    Ok(state) => match self.t_mut().table.room(state.n_bytes(), &keep) {
                        Ok(gone) => {
                            for (j, s) in gone {
                                voids.push(j);
                                offers.push((j, s, false));
                            }
                            let entry = Entry::State {
                                state,
                                live: live_from,
                                at: 0,
                            };
                            self.t_mut().table.put(from, entry);
                            moved = true;
                            None
                        }
                        Err(NoRoom {
                            need,
                            pinned,
                            budget,
                        }) => {
                            if !live_from {
                                offers.push((from, state, true));
                            }
                            Some(format!(
                                "its {need} bytes beside the {pinned} parked for live requests \
                                 pass the park budget of {budget} bytes"
                            ))
                        }
                    },
                },
            };
            match refused {
                Some(why) if live_from => {
                    return Ok(Switched::Refused(format!(
                        "the request on slot {from} cannot be parked: {why}"
                    )));
                }
                Some(_) => voids.push(from),
                None => {}
            }
        }
        if let Some(i) = self.index_of(from) {
            let t = self.t_mut();
            t.clock += 1;
            self.active[i].left = self.t().clock;
        }
        for (j, state, copied) in offers {
            self.slot.offer(j, state, copied);
        }
        for &j in &voids {
            if let Err(e) = self
                .slot
                .select(j)
                .and_then(|()| self.slot.erase().map(|_| ()))
            {
                return Err(self.die(&e));
            }
        }
        if !voids.is_empty() {
            let mut b = relock(&self.sh.board);
            for &j in &voids {
                b.view_mut(j).n_past = 0;
            }
        }
        if let Err(e) = self.slot.select(to) {
            return Err(self.die(&e));
        }
        self.t_mut().on = to;
        let mut out = Switched::Done;
        let mut fed_ms = 0.0;
        match self.t_mut().table.take(to) {
            Some(Entry::State {
                state, live: false, ..
            }) if prompt => {
                moved = true;
                self.slot.hand_back(state);
            }
            Some(Entry::State { state, .. }) => {
                moved = true;
                match self.slot.engine.resume(&state) {
                    Ok(()) => {}
                    Err(StateError::Engine(e)) => return Err(self.die(&e)),
                    Err(e) => {
                        if let Err(e) = self.slot.erase() {
                            return Err(self.die(&e));
                        }
                        if live_to {
                            out = Switched::Lost(format!(
                                "the parked state of slot {to} did not resume: {e}"
                            ));
                        }
                    }
                }
            }
            Some(Entry::Ids) => {
                moved = true;
                let held = self.slot.held_of(to).clone();
                let t1 = Instant::now();
                // Ids carry no image to feed a span again; the server refuses
                // an engine of images whose slots park ids when it binds.
                let fed = if held.media.is_empty() {
                    self.slot
                        .engine
                        .reset()
                        .and_then(|()| self.slot.engine.prefill(&held.ids))
                } else {
                    Err(EngineError(format!(
                        "slot {to} holds {} image span(s); ids parking cannot feed them again",
                        held.media.len()
                    )))
                };
                if let Err(e) = fed {
                    return Err(self.die(&e));
                }
                fed_ms = ms_since(t1);
                let mut s = relock(&self.sh.stats);
                s.n_reprefill_total += held.ids.len() as u64;
                s.t_reprefill_ms_total += fed_ms;
            }
            None => {
                assert!(
                    self.slot.held_of(to).is_empty(),
                    "slot {to} holds {} ids and no parked state",
                    self.slot.held_of(to).len()
                );
                if let Err(e) = self.slot.engine.reset() {
                    return Err(self.die(&e));
                }
            }
        }
        let mut s = relock(&self.sh.stats);
        if moved {
            s.n_swaps_total += 1;
            s.t_swap_ms_total += ms_since(t0) - fed_ms;
        }
        s.parked_bytes = self.t().table.bytes();
        Ok(out)
    }

    /// Each slot's turn, for `/slots`.
    fn show_turns(&self) {
        let t = self.t();
        let n = relock(&self.sh.board).slots().len();
        let turns = (0..n)
            .map(|slot| {
                if t.running == Some(slot) {
                    Turn::Running
                } else if self.index_of(slot).is_some() {
                    Turn::Parked
                } else if t.pending.iter().any(|p| p.slot == slot) {
                    Turn::Queued
                } else {
                    Turn::Idle
                }
            })
            .collect();
        *relock(&self.sh.turns) = Some(turns);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use super::{Acted, Action, ReqName, Shared, finish_line, progress_line, serve};
    use crate::api::{End, EngineFailure, relock};
    use crate::genloop::{Slot, StopKind, Timings, cause_word};
    use crate::sched::{Board, FifoPicker, Refusal, Reserve};
    use crate::{MockEngine, sampling};

    /// The first failure ends the server, once: a later one, from the
    /// engine thread or from a request's own thread, changes neither the
    /// reason `/health` and the board give nor the failure the server ends
    /// with.
    #[test]
    fn the_first_failure_is_the_one_that_ends_the_server() {
        let (end, ended) = mpsc::channel();
        let sh = Shared::new(
            Board::new(1, 4, Box::new(FifoPicker)),
            end,
            sampling::reference_factory(),
        );
        let failure = |error: &str| EngineFailure {
            engine: "mock".to_owned(),
            error: error.to_owned(),
        };
        let first = sh.record_fatal(failure("first"));
        assert!(sh.record_fatal(failure("second")).is_none());
        sh.end_on(first);
        assert_eq!(relock(&sh.fatal).as_deref(), Some("first"));
        let action = Action {
            run: Box::new(|_: &mut Slot| Acted {
                failure: None,
                reply: Box::new(|| {}),
            }),
            drops: false,
        };
        assert_eq!(
            relock(&sh.board).reserve(Reserve::All, action).err(),
            Some(Refusal::Dead("first".to_owned()))
        );
        match ended.try_recv() {
            Ok(End::Engine(f)) => assert_eq!(f.error, "first"),
            _ => panic!("no engine end"),
        }
        assert!(ended.try_recv().is_err(), "the server ends once");
    }

    /// The halt reads the stop's cause, which [`Shared::begin_stop`] sets
    /// before it stores the stop flag: a stop flag with no cause is that
    /// order broken, and the engine thread ends on a panic naming it — a
    /// defect by name, never a halt under a cause made up for it.
    #[test]
    fn a_stop_with_no_cause_is_a_named_panic() {
        let sh = Arc::new(Shared::new(
            Board::new(1, 4, Box::new(FifoPicker)),
            mpsc::channel().0,
            sampling::reference_factory(),
        ));
        sh.stop.store(true, Ordering::SeqCst);
        // The engine loop on this thread: it reads the stop at once and halts.
        serve(
            Slot::with_slots(Box::new(MockEngine::new(64)), 1),
            Arc::clone(&sh),
        );
        assert!(sh.wait_loop_end(Duration::ZERO), "the engine loop ended");
        let fatal = relock(&sh.fatal).clone().unwrap_or_default();
        assert!(
            fatal.starts_with("the engine thread panicked: the stop's cause:"),
            "the panic names the broken order: {fatal:?}"
        );
    }

    /// One progress line every interval of generation, on the generation's
    /// own clock: the due check a token costs is one comparison, and a
    /// generation that stalls past several intervals prints one line at its
    /// next token, not a run of them.
    #[test]
    fn a_progress_stderr_line_every_interval_of_generation() {
        let sh = Shared::new(
            Board::new(1, 4, Box::new(FifoPicker)),
            mpsc::channel().0,
            sampling::reference_factory(),
        );
        let name = ReqName {
            id: 0,
            route: "/completion".to_owned(),
            choice: 0,
            choices: 1,
        };
        // One token through the generation's own tick: whether it printed a line.
        let tick = |noted: &mut usize, ms: f64| {
            let before = sh.notes().len();
            super::tick_of(&sh, 0, &name, noted)(&Timings {
                predicted_ms: ms,
                ..Timings::default()
            });
            sh.notes().len() - before == 1
        };
        let mut noted = 0;
        assert!(!tick(&mut noted, 0.0));
        assert!(!tick(&mut noted, 29_999.0));
        assert!(tick(&mut noted, 30_000.0));
        assert!(!tick(&mut noted, 30_001.0));
        assert!(!tick(&mut noted, 59_999.0));
        assert!(tick(&mut noted, 60_000.0));
        // A stall past two intervals: one line, and the next waits for the
        // next multiple.
        assert!(tick(&mut noted, 95_000.0));
        assert!(!tick(&mut noted, 95_001.0));
        assert!(!tick(&mut noted, 119_999.0));
        assert!(tick(&mut noted, 120_000.0));
        // A stall from the start: one line at 95 s, none at the tokens after.
        let mut noted = 0;
        assert!(tick(&mut noted, 95_000.0));
        assert!(!tick(&mut noted, 95_001.0));
        assert!(!tick(&mut noted, 95_002.0));
    }

    /// The two lines' shapes: a fixed field order, the seconds to two
    /// decimals, the rates to one, `-` over no time, and the choice's index
    /// only for one choice of several.
    #[test]
    fn the_two_stderr_lines() {
        let of_two = ReqName {
            id: 7,
            route: "/completion".to_owned(),
            choice: 1,
            choices: 2,
        };
        let t = Timings {
            prompt_n: 12,
            prompt_ms: 850.0,
            predicted_n: 40,
            predicted_ms: 3_900.0,
            n_ctx: 64,
            n_past: 52,
            cache_n: 3,
            ..Timings::default()
        };
        assert_eq!(
            progress_line(&of_two, &t),
            "bloomery-serve: request 7.1 /completion: generating, 40 ids in 3.90 s, 10.3 ids/s; \
             ctx 52/64"
        );
        assert_eq!(
            finish_line(&of_two, &t, "length"),
            "bloomery-serve: request 7.1 /completion: prompt 12 ids (3 reused) in 0.85 s, \
             14.1 ids/s; generated 40 ids in 3.90 s, 10.3 ids/s; stop length; ctx 52/64"
        );
        let of_one = ReqName {
            id: 8,
            route: "/v1/messages".to_owned(),
            choice: 0,
            choices: 1,
        };
        let none = Timings {
            n_ctx: 64,
            ..Timings::default()
        };
        // A request that reached no token: the rates read `-`, the counts
        // stay what it reached.
        assert_eq!(
            finish_line(&of_one, &none, "error"),
            "bloomery-serve: request 8 /v1/messages: prompt 0 ids (0 reused) in 0.00 s, - ids/s; \
             generated 0 ids in 0.00 s, - ids/s; stop error; ctx 0/64"
        );
    }

    /// The stop cause words: the generation's own stop, and the two ways a
    /// request ends without one.
    #[test]
    fn the_stderr_line_cause_words() {
        assert_eq!(cause_word(Some(StopKind::Eos), false), "eos");
        assert_eq!(cause_word(Some(StopKind::Word), false), "stop");
        assert_eq!(cause_word(Some(StopKind::Limit), false), "length");
        assert_eq!(cause_word(Some(StopKind::Loop), false), "reasoning_loop");
        assert_eq!(cause_word(None, false), "cancel");
        assert_eq!(cause_word(None, true), "error");
    }
}
