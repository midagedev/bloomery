//! The engine thread: the one place the engine is called. It takes the
//! requests the [`Board`] admits to slots, runs each one's prompt as its own
//! calls, and then steps every running request together: one row a slot, in
//! one [`Engine::step_slots`] call when two or more run, a plain
//! [`Engine::next`] (or a drafted pass) when one does. It runs the slot
//! actions the board reserved between those calls.
//!
//! A request's events reach its HTTP thread over a channel of its own. Its
//! counters, its `/slots` view and its slot's release are booked before its
//! last event is sent, so the response a client reads finds the slot free
//! and the counters final. A client gone (its channel closed) ends its request
//! at the next event; the slot is released.
//!
//! An engine error is fatal: every request in the failed call gets the error,
//! every waiting one is refused, and the server ends ([`crate::api::End`]). A
//! panic on this thread ends the server the same way, by name.

use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};

use serde_json::Value;

use crate::api::{End, EngineFailure, relock};
use crate::engine::{EngineError, SamplerFactory, SlotRow};
use crate::genloop::{Event, Gen, GenError, GenParams, Need, Outcome, Slot, StopKind, Timings};
use crate::sched::{Board, Reserve, SlotView};

/// The server's counters, `/metrics`' totals.
#[derive(Default)]
pub(crate) struct Stats {
    /// Prompt tokens evaluated: the prompt less what the cache kept.
    pub n_prompt_total: u64,
    /// Prompt tokens the cache kept instead (every request's `cache_n`).
    pub n_prompt_cached_total: u64,
    pub t_prompt_ms_total: f64,
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
}

/// A request handed to the engine thread.
pub(crate) struct Submit {
    pub p: GenParams,
    /// What `/slots` shows of it.
    pub prompt: Value,
    pub settings: Value,
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
}

/// A slot action run on the engine thread: `run` on the reserved slot
/// (selected first for one slot), whose `reply` is sent once the slots are
/// released, and the engine error it met, if any, which is fatal.
pub(crate) struct Action {
    pub run: Box<dyn FnOnce(&mut Slot) -> Acted + Send>,
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
            end,
            sampler,
            ids: AtomicU64::new(0),
        }
    }

    pub(crate) fn next_id(&self) -> u64 {
        self.ids.fetch_add(1, Ordering::SeqCst)
    }
}

/// The engine thread failed; nothing more runs.
struct Dead;

struct Active {
    slot: usize,
    job: Gen,
    events: mpsc::Sender<Msg>,
}

struct Worker {
    slot: Slot,
    sh: Arc<Shared>,
    /// The requests running, in the order they took their slots.
    active: Vec<Active>,
}

/// Runs the engine thread until the engine fails or the thread panics.
pub(crate) fn serve(slot: Slot, sh: Arc<Shared>) {
    let mut w = Worker {
        slot,
        sh: Arc::clone(&sh),
        active: Vec::new(),
    };
    let ran = panic::catch_unwind(AssertUnwindSafe(|| w.run()));
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
        events
            .send(m)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "the client went away"))
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
                    let admitted = b.admit();
                    if !actions.is_empty() || !admitted.is_empty() || !self.active.is_empty() {
                        break (actions, admitted);
                    }
                    b = self.sh.work.wait(b).unwrap_or_else(|e| e.into_inner());
                }
            };
            for (what, a) in actions {
                if self.act(what, a).is_err() {
                    return;
                }
            }
            for (slot, ids, sub) in admitted {
                if self.start(slot, &ids, sub).is_err() {
                    return;
                }
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

    fn act(&mut self, what: Reserve, a: Action) -> Result<(), Dead> {
        if let Reserve::One(i) = what
            && let Err(e) = self.slot.select(i)
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
                    let held = self.slot.held_of(i).to_vec();
                    b.view_mut(i).n_past = held.len();
                    b.release(i, held);
                }
                Reserve::All => {
                    for i in 0..b.slots().len() {
                        b.release(i, self.slot.held_of(i).to_vec());
                    }
                }
            }
        }
        (acted.reply)();
        Ok(())
    }

    /// A request that took `slot`: its prompt, run now.
    fn start(&mut self, slot: usize, ids: &[u32], sub: Submit) -> Result<(), Dead> {
        let Submit {
            p,
            prompt,
            settings,
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
        let mut job = Gen::new(&self.slot, &self.sh.sampler, ids.len(), &p);
        let r = match self.slot.select(slot) {
            Ok(()) => job.prompt(&mut self.slot, ids, &p, &mut sink_of(&events)),
            Err(e) => Err(GenError::Engine(e)),
        };
        let a = Active { slot, job, events };
        match r {
            Ok(()) => {
                self.active.push(a);
                Ok(())
            }
            Err(GenError::Engine(e)) => {
                let f = self.fail(&e);
                self.end_request(a, Err(GenError::Engine(e)));
                Err(self.end(f))
            }
            Err(e) => {
                self.end_request(a, Err(e));
                Ok(())
            }
        }
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
                Need::Done => unreachable!("a finished request left the running ones"),
            }
        } else {
            let mut rows: Vec<SlotRow<'_>> = self
                .active
                .iter_mut()
                .zip(&needs)
                .map(|(a, need)| {
                    let Need::Step(last) = *need else {
                        panic!(
                            "a drafted pass among {} running slots: a drafting engine serves one",
                            needs.len()
                        );
                    };
                    SlotRow {
                        slot: a.slot,
                        last,
                        logits: a.job.logits_out(),
                        next: 0,
                    }
                })
                .collect();
            let called = self.slot.step_slots(&mut rows);
            let answers: Vec<u32> = rows.iter().map(|r| r.next).collect();
            drop(rows);
            called.map(|()| {
                for (a, g) in self.active.iter_mut().zip(answers) {
                    a.job.stepped(g);
                }
            })
        };
        let Err(e) = called else {
            return Ok(());
        };
        let f = self.fail(&e);
        for a in std::mem::take(&mut self.active) {
            self.end_request(a, Err(GenError::Engine(EngineError(e.0.clone()))));
        }
        Err(self.end(f))
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
            b.release(a.slot, self.slot.held_of(a.slot).to_vec());
        }
        let _ = a.events.send(Msg::Done(r));
    }
}
