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
//! more than one quantum left after what the cache kept, runs as calls of a
//! quantum, one a round, each round's decode of the busy slots between them:
//! a long prompt does not stall the streams already answering. Its request
//! takes no decode round until its prompt ends. Without another busy slot, or
//! on an engine that names none, a prompt runs in one go.
//!
//! A request's events reach its HTTP thread over a channel of its own. Its
//! counters, its `/slots` view and its slot's release are booked before its
//! last event is sent, so the response a client reads finds the slot free
//! and the counters final. A client gone ends its request at the next event,
//! a stream's or a whole answer's alike: its HTTP thread's sink fails (a
//! stream's write, a whole answer's socket reading closed), the thread drops
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

use std::io;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::api::{End, EngineFailure, Stop, client_gone, relock};
use crate::engine::{Drafted, EngineError, SamplerFactory, Saved, SlotPass, SlotRow, StateError};
use crate::genloop::{
    Event, Gen, GenError, GenParams, Need, Outcome, Prompt, Slot, StopKind, Timings, ms_since,
};
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
    /// Over every engine call, the slots it carried.
    pub n_busy_slots_total: u64,
    /// The longest a request's sequence grew (prompt and generation, `n_past`).
    pub n_tokens_max: u64,
    /// The ids the draft proposed, the ones kept, and the passes that verified
    /// a proposal.
    pub n_draft_total: u64,
    pub n_draft_accepted_total: u64,
    pub n_draft_passes_total: u64,
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

/// A request handed to the engine thread.
pub(crate) struct Submit {
    pub p: GenParams,
    /// What `/slots` shows of it.
    pub prompt: Value,
    pub settings: Value,
    /// The prompt's images, in the order of their spans: each rides the
    /// prompt call that holds its span.
    pub media: Vec<MediaFeed>,
    pub events: mpsc::Sender<Msg>,
}

/// What the engine thread tells a request's HTTP thread.
pub(crate) enum Msg {
    /// The request took this slot; sent before any other message.
    Started(usize),
    Prompt(Timings),
    Text(String, Timings),
    /// The last message.
    Done(Result<Outcome, GenError>),
    /// The request was not started: the state of the request on the engine
    /// could not be parked. Sent before any other message; the request may
    /// succeed later as it is.
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
    /// Set by the engine error that ended the server; the reason `/health` gives.
    pub fatal: Mutex<Option<String>>,
    /// The orderly stop's cause, once it began ([`Shared::begin_stop`]): the
    /// reason every request that needs the engine is refused with.
    pub stopping: Mutex<Option<String>>,
    /// The orderly stop's order to the engine thread: read between engine
    /// calls ([`Worker::run`], [`Worker::run_turns`]), it ends the thread.
    stop: AtomicBool,
    /// The engine thread left its loop ([`worker::serve`], on every path that
    /// ends its run: the stop's [`Worker::halt`], an engine failure, a caught
    /// panic), set before the engine it holds drops: what the orderly stop
    /// waits for ([`crate::api::exit_shutdown`]), not the thread's end, which
    /// the engine's Drop can hold past the stop's bound while the process
    /// exit reclaims the engine's memory anyway.
    loop_ended: Mutex<bool>,
    /// Wakes [`Shared::wait_loop_end`].
    loop_ended_c: Condvar,
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
            fatal: Mutex::new(None),
            stopping: Mutex::new(None),
            stop: AtomicBool::new(false),
            loop_ended: Mutex::new(false),
            loop_ended_c: Condvar::new(),
            turns: Mutex::new(None),
            end,
            sampler,
            ids: AtomicU64::new(0),
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
    /// here. Called on every path that ends the thread's run, before the
    /// engine it holds drops — its Drop can outlast the stop's bound (a host
    /// tier's pinned pages freed, its threads joined), and the process exit
    /// reclaims the engine's memory anyway.
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
    events: mpsc::Sender<Msg>,
    /// Under turns: the engine call the request needs, taken from its
    /// generation before its slot was parked.
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
}

/// Runs the engine thread until the engine fails or the thread panics.
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
    let mut w = Worker {
        slot,
        sh: Arc::clone(&sh),
        active: Vec::new(),
        prompting: Vec::new(),
        quantum,
        turns,
    };
    let ran = panic::catch_unwind(AssertUnwindSafe(|| {
        if w.turns.is_some() {
            w.run_turns();
        } else {
            w.run();
        }
    }));
    // The stop's wait ends here: past this point the thread only books a
    // caught panic and drops the engine it holds, which no stop waits for.
    sh.end_loop();
    if let Err(p) = ran {
        let what = p
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic without a message".to_owned());
        let error = format!("the engine thread panicked: {what}");
        *relock(&sh.fatal) = Some(error.clone());
        drop(relock(&sh.board).kill(&error));
        // The running requests' channels close: their HTTP threads answer
        // with the reason.
        drop(w);
        let _ = sh.end.send(End::Engine(EngineFailure {
            engine: "unknown: its thread panicked".to_owned(),
            error,
        }));
    }
}

/// The sink a request's generation writes to: its channel.
fn sink_of(events: &mpsc::Sender<Msg>) -> impl FnMut(Event<'_>) -> io::Result<()> + '_ {
    |ev| {
        let m = match ev {
            Event::Prompt(t) => Msg::Prompt(t.clone()),
            Event::Text(text, t) => Msg::Text(text.to_owned(), t.clone()),
        };
        events.send(m).map_err(|_| client_gone())
    }
}

/// The tick a request's generation calls after every token: its slot's view.
fn tick_of(sh: &Shared, slot: usize) -> impl FnMut(&Timings) + '_ {
    move |t| {
        let mut b = relock(&sh.board);
        let v = b.view_mut(slot);
        v.n_decoded = t.predicted_n;
        v.n_past = t.n_past;
    }
}

impl Worker {
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
            if self.sh.stopped() {
                return self.halt();
            }
            for (what, a) in actions {
                if self.act(what, a, false).is_err() {
                    return;
                }
            }
            for (slot, held, sub) in admitted {
                if self.start(slot, &held, sub).is_err() {
                    return;
                }
            }
            if self.prompts().is_err() {
                return;
            }
            if self.step().is_err() {
                return;
            }
        }
    }

    /// Records an engine failure: `/health` and every later request see it,
    /// and what waits is refused.
    fn fail(&mut self, e: &EngineError) -> EngineFailure {
        let f = EngineFailure {
            engine: self.slot.engine.describe(),
            error: e.to_string(),
        };
        *relock(&self.sh.fatal) = Some(f.error.clone());
        // The waiting requests' and actions' channels close: their HTTP
        // threads answer with the reason.
        drop(relock(&self.sh.board).kill(&f.error));
        f
    }

    /// Ends the server once the requests that met the failure have their
    /// answers.
    fn end(&self, f: EngineFailure) -> Dead {
        let _ = self.sh.end.send(End::Engine(f));
        Dead
    }

    /// The engine thread's orderly end, what [`Shared::begin_stop`] ordered:
    /// every queued request's and action's channel closes, so its HTTP thread
    /// answers with the stop's cause ([`crate::api::engine_gone`]), and the
    /// thread ends without touching the engine again. A request the worker
    /// already took keeps its channel in the worker (`active`, `prompting`,
    /// and `pending` under turns), which closes it when the worker drops at
    /// the end of [`serve`]: after [`Shared::end_loop`] and after the engine's
    /// Drop, so the process exit the stop's wait ends in can come first.
    /// Nothing failed: no engine error is recorded, and no end is sent — the
    /// stop's sender already did.
    fn halt(&self) {
        let cause = relock(&self.sh.stopping)
            .clone()
            .unwrap_or_else(|| "the server is stopping".to_owned());
        drop(relock(&self.sh.board).kill(&cause));
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
    /// feeds), run now, or planned as calls of the engine's quantum that
    /// [`Worker::prompts`] runs one a round ([`Worker::interleaves`]).
    /// Returns whether it runs on (an error ended it otherwise).
    fn start(&mut self, slot: usize, held: &Held, sub: Submit) -> Result<bool, Dead> {
        let Submit {
            p,
            prompt,
            settings,
            media,
            events,
        } = sub;
        let _ = events.send(Msg::Started(slot));
        *relock(&self.sh.board).view_mut(slot) = SlotView {
            id_task: self.sh.next_id(),
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
        let mut job = Gen::new(&self.slot, &self.sh.sampler, n, &p);
        let kept = match self.slot.select(slot) {
            Ok(()) => job.open(&mut self.slot, held, &p),
            Err(e) => Err(GenError::Engine(e)),
        };
        let r = match kept {
            Ok(kept) => match self.interleaves(n, kept.cache_n) {
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

    /// The quantum a prompt of `n` ids whose cache kept `cache_n` runs in,
    /// one call a round: the engine names one, another slot is busy
    /// generating, and more than one quantum of the prompt call
    /// (`cache_n..n-1`) is left.
    fn interleaves(&self, n: usize, cache_n: usize) -> Option<NonZeroUsize> {
        self.quantum
            .filter(|q| !self.active.is_empty() && n - 1 - cache_n > q.get())
    }

    /// One call of each prompt run a call a round, in the order they took
    /// their slots. It runs after the starts and before the decode round
    /// ([`Worker::step`]), so a prompt's consecutive calls have a round of
    /// the busy slots between them. A prompt whose last call ran joins the
    /// running requests; a request whose client left ends, its slot freed;
    /// an engine error is fatal, as at a start.
    fn prompts(&mut self) -> Result<(), Dead> {
        let mut k = 0;
        while k < self.prompting.len() {
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
            let a = &mut self.active[k];
            let pumped = a
                .job
                .pump(&mut sink_of(&a.events), &mut tick_of(&self.sh, a.slot));
            match pumped {
                Ok(Need::Done) => {
                    let mut a = self.active.remove(k);
                    let r = a.job.finish(&mut sink_of(&a.events));
                    self.end_request(a, r);
                }
                Ok(need) => {
                    needs.push(need);
                    k += 1;
                }
                Err(e) => {
                    let a = self.active.remove(k);
                    self.end_request(a, Err(e));
                }
            }
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let width = self.active.len() as u64;
        {
            let mut s = relock(&self.sh.stats);
            s.n_decode_total += 1;
            s.n_busy_slots_total += width;
        }
        let called = if let [a] = self.active.as_mut_slice() {
            match needs[0] {
                Need::Step(t) => self
                    .slot
                    .select(a.slot)
                    .and_then(|()| self.slot.next(t, a.job.logits_out()))
                    .map(|g| a.job.stepped(g)),
                Need::Advance(t) => self
                    .slot
                    .select(a.slot)
                    .and_then(|()| self.slot.advance(t, a.job.kept_mut()))
                    .map(|d| a.job.advanced(d)),
                Need::Sampled(t) => self
                    .slot
                    .select(a.slot)
                    .and_then(|()| a.job.advance_sampled(&mut self.slot, t)),
                Need::Done => unreachable!("a finished request left the running ones"),
            }
        } else {
            // One engine round, its rows split by what they need: the steps
            // of two slots or more in one call (one step is a select and a
            // `next`, as `step_slots`' doc requires), then the drafted passes
            // in one. Within each call the rows keep `active`'s order. A
            // sampled request passes only alone in its round: here it steps.
            let mut steps: Vec<SlotRow<'_>> = Vec::new();
            let mut step_at: Vec<usize> = Vec::new();
            let mut passes: Vec<SlotPass<'_>> = Vec::new();
            let mut pass_at: Vec<usize> = Vec::new();
            for (i, (a, need)) in self.active.iter_mut().zip(&needs).enumerate() {
                match *need {
                    Need::Step(last) | Need::Sampled(last) => {
                        step_at.push(i);
                        steps.push(SlotRow {
                            slot: a.slot,
                            last,
                            logits: a.job.logits_out(),
                            next: 0,
                        });
                    }
                    Need::Advance(last) => {
                        pass_at.push(i);
                        passes.push(SlotPass {
                            slot: a.slot,
                            last,
                            out: a.job.kept_mut(),
                            drafted: Drafted::default(),
                        });
                    }
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
            if called.is_ok() && !passes.is_empty() {
                called = self.slot.advance_slots(&mut passes);
            }
            if called.is_ok() {
                let answers: Vec<u32> = steps.iter().map(|r| r.next).collect();
                let drafted: Vec<Drafted> = passes.iter().map(|r| r.drafted).collect();
                drop(steps);
                drop(passes);
                for (i, g) in step_at.into_iter().zip(answers) {
                    self.active[i].job.stepped(g);
                }
                for (i, d) in pass_at.into_iter().zip(drafted) {
                    self.active[i].job.advanced(d);
                }
            }
            called
        };
        let Err(e) = called else {
            return Ok(());
        };
        Err(self.die(&e))
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

    /// Books a request's counters and view, releases its slot, and sends its
    /// last message.
    fn end_request(&mut self, a: Active, r: Result<Outcome, GenError>) {
        let t = a.job.timings();
        {
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
        let _ = a.events.send(Msg::Done(r));
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
        }
    }

    /// One step boundary: the running request's tokens in hand go out, the
    /// engine changes hands where a rule says so, and the request on it takes
    /// one engine call.
    fn turn(&mut self) -> Result<(), Dead> {
        if let Some(i) = self.t().running.and_then(|r| self.index_of(r))
            && self.active[i].need.is_none()
            && let Some(need) = self.pump_at(i)
        {
            self.active[i].need = Some(need);
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

    /// The first pending request's prompt: the engine moves to its slot (a
    /// live request's state that cannot be parked refuses it by name), its
    /// prompt runs, and its turn begins.
    fn begin(&mut self) -> Result<(), Dead> {
        let p = self.t_mut().pending.remove(0);
        match self.switch(p.slot, true)? {
            Switched::Refused(why) => {
                relock(&self.sh.stats).n_swap_refused_total += 1;
                relock(&self.sh.board).release(p.slot);
                self.show_turns();
                let _ = p.sub.events.send(Msg::Refused(why));
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

    /// Takes the tokens request `i` holds; its next engine call, or `None`
    /// when it ended (and is booked).
    fn pump_at(&mut self, i: usize) -> Option<Need> {
        let a = &mut self.active[i];
        let pumped = a
            .job
            .pump(&mut sink_of(&a.events), &mut tick_of(&self.sh, a.slot));
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
            if self.t().running == Some(a.slot) {
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
        let need = match self.active[i].need.take() {
            Some(n) => n,
            None => match self.pump_at(i) {
                Some(n) => n,
                None => return Ok(()),
            },
        };
        {
            let mut s = relock(&self.sh.stats);
            s.n_decode_total += 1;
            s.n_busy_slots_total += 1;
        }
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
            Need::Done => unreachable!("a finished request left the running ones"),
        };
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
