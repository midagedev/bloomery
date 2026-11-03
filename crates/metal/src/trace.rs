//! The encode trace: what the host asks of Metal, as text.
//!
//! The kernels' pipelines are what `bin/metal_scan.rs` hashes; the other half
//! of a run's program is the host's dispatch: which pipeline, how many groups
//! of how many threads, which buffer window and which `setBytes` value at each
//! slot, the barriers, and the command buffers' commit, event and wait order.
//! This module writes that, one line per event, while a trace is started
//! ([`start`], [`finish`]).
//!
//! A buffer is named by its allocation order (`#n`, the nth buffer
//! [`crate::MetalBuffer`] made in the process, counted as one thread would
//! make them), an offset and its length, never by an address; a command buffer
//! by the order it was made. A dispatch line
//! states the whole binding table the dispatch runs under (Metal keeps a slot's
//! binding until it is set again), so the order of the `set` calls between two
//! dispatches is not part of the trace. The host calls no
//! `setThreadgroupMemoryLength`; a dispatch's threadgroup memory is its
//! pipeline's static length, printed on the line.
//!
//! Every line is one of:
//!
//! ```text
//! cb N begin | cb N wait-event NAME=V | cb N signal NAME=V | cb N commit
//! enc N.M open serial|concurrent | enc N.M end
//! dispatch N.M.K LABEL max=… width=… tg_static=… grid=GXxGY tpg=T b0=… b1=…
//! barrier N.M
//! host settle V | host sync
//! refused REASON: COUNT
//! ```
//!
//! The order threads make buffers in is not the order a name may rest on, so
//! two things number them. A load whose jobs run on several threads ([`Load`])
//! names a buffer by its job and its place in that job, and numbers them when
//! it ends: jobs in index order, as if one thread had run them in that order.
//! Every other buffer is numbered in the order the one thread that numbers
//! them (the first to make a buffer outside a load) makes it. A buffer that
//! has no such place, made by no job inside a load or on another thread
//! outside one, gets no number: it is bound as `#?n`, and the trace carries a
//! `refused` line that `metal_scan --compare-trace` does not read as an event
//! and so refuses the trace by name.

use core::fmt::Write as _;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState, MTLDispatchType};

use crate::encode::MAX_BINDINGS;
use crate::library::Pipeline;

/// The format's version, the first line of a trace. A version moves when a
/// buffer's name means something else (v2: a load's buffers are numbered in
/// job order, not in the order threads made them).
pub const VERSION: &str = "# bloomery encode trace v2";

/// Why a buffer got no number.
const BY_NO_JOB: &str = "a buffer made inside a load by no job";
const OFF_THREAD: &str =
    "a buffer made outside a load on a thread other than the one that numbers them";
const NESTED: &str = "a load began while another was open";
const LOST_JOB: &str = "a buffer made by a job of a load that is not the open one";

/// How a buffer is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Name {
    /// The nth buffer, in order.
    Number(u64),
    /// Buffer `k` of job `job` of the open load, numbered when the load ends.
    Placed { job: usize, k: u64 },
}

/// A load whose jobs are making buffers.
struct OpenLoad {
    id: u64,
    /// The number the load's first buffer takes: the counter when it began.
    base: u64,
    /// The buffers each job has made so far.
    made: BTreeMap<usize, u64>,
}

struct State {
    /// A trace is being written.
    active: bool,
    lines: Vec<String>,
    /// Buffer address to (name, length): every buffer made since the process
    /// began, the last one at an address winning.
    buffers: BTreeMap<usize, (Name, usize)>,
    /// Buffers numbered so far; a load's are counted when it ends.
    made: u64,
    /// The thread that numbers the buffers made outside a load.
    sequence: Option<ThreadId>,
    /// The load now open, if one is.
    load: Option<OpenLoad>,
    /// Loads begun, the next one's id.
    loads: u64,
    /// Buffers that got no number, by reason.
    refused: BTreeMap<&'static str, u64>,
    /// Buffers bound but never made by [`crate::MetalBuffer`], by first sight.
    strangers: BTreeMap<usize, u64>,
    /// Command buffer address to its order.
    command_buffers: BTreeMap<usize, u64>,
    command_buffers_made: u64,
    /// Encoders opened on each command buffer so far.
    encoders: BTreeMap<u64, u64>,
}

static STATE: Mutex<State> = Mutex::new(State::new());

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

impl State {
    const fn new() -> Self {
        Self {
            active: false,
            lines: Vec::new(),
            buffers: BTreeMap::new(),
            made: 0,
            sequence: None,
            load: None,
            loads: 0,
            refused: BTreeMap::new(),
            strangers: BTreeMap::new(),
            command_buffers: BTreeMap::new(),
            command_buffers_made: 0,
            encoders: BTreeMap::new(),
        }
    }

    fn refuse(&mut self, reason: &'static str) {
        *self.refused.entry(reason).or_insert(0) += 1;
    }

    /// The name of a buffer just made on `thread` inside job scope `scope`
    /// (a load's id and a job's index), or why it has none.
    fn name(
        &mut self,
        scope: Option<(u64, usize)>,
        thread: ThreadId,
    ) -> Result<Name, &'static str> {
        match (scope, self.load.as_mut()) {
            (Some((id, job)), Some(load)) if load.id == id => {
                let made = load.made.entry(job).or_insert(0);
                let k = *made;
                *made += 1;
                Ok(Name::Placed { job, k })
            }
            (Some(_), _) => Err(LOST_JOB),
            (None, Some(_)) => Err(BY_NO_JOB),
            (None, None) => {
                if *self.sequence.get_or_insert(thread) != thread {
                    return Err(OFF_THREAD);
                }
                let n = self.made;
                self.made += 1;
                Ok(Name::Number(n))
            }
        }
    }

    /// Load `id` ends: the buffers of its jobs below `counted` are numbered
    /// from the load's base, jobs in index order and each job's in the order
    /// it made them; the buffers of the jobs from `counted` on are dropped
    /// (their jobs would not have run in order). Nothing when `id` is not the
    /// open load.
    fn close(&mut self, id: u64, counted: usize) {
        let Some(load) = self.load.take_if(|l| l.id == id) else {
            return;
        };
        let mut first = BTreeMap::new();
        let mut next = load.base;
        for (&job, &n) in load.made.range(..counted) {
            first.insert(job, next);
            next += n;
        }
        self.buffers.retain(|_, (name, _)| match *name {
            Name::Number(_) => true,
            Name::Placed { job, k } => first.get(&job).is_some_and(|&at| {
                *name = Name::Number(at + k);
                true
            }),
        });
        self.made = next;
    }

    fn emit(&mut self, line: impl FnOnce() -> String) {
        if self.active {
            let line = line();
            self.lines.push(line);
        }
    }

    /// The order of the command buffer at `address`.
    fn command_buffer(&mut self, address: usize) -> u64 {
        match self.command_buffers.get(&address) {
            Some(&n) => n,
            None => {
                let n = self.command_buffers_made;
                self.command_buffers_made += 1;
                self.command_buffers.insert(address, n);
                n
            }
        }
    }
}

/// Start a trace, dropping the lines an earlier one wrote.
pub fn start() {
    let mut state = state();
    state.active = true;
    state.lines.clear();
}

/// The trace written since [`start`], as text; the trace stops.
#[must_use]
pub fn finish() -> String {
    let mut state = state();
    state.active = false;
    let mut text = String::from(VERSION);
    text.push('\n');
    for (reason, n) in &state.refused {
        let _ = writeln!(text, "refused {reason}: {n}");
    }
    for line in state.lines.drain(..) {
        text.push_str(&line);
        text.push('\n');
    }
    text
}

thread_local! {
    /// The load and the job of it that the buffers this thread makes belong to.
    static SCOPE: Cell<Option<(u64, usize)>> = const { Cell::new(None) };
}

/// A buffer of `length` bytes at `address` was made.
pub(crate) fn buffer_made(address: usize, length: usize) {
    let scope = SCOPE.get();
    let thread = std::thread::current().id();
    let mut state = state();
    match state.name(scope, thread) {
        Ok(name) => {
            state.buffers.insert(address, (name, length));
        }
        Err(reason) => {
            state.refuse(reason);
            // An earlier buffer at this address is gone: bound, this one is a
            // stranger.
            state.buffers.remove(&address);
        }
    }
}

/// A load whose jobs make buffers on several threads. The threads' order is
/// not the buffers' order: each job runs under [`Load::job`], and when the
/// load ends its buffers are numbered as one thread running the jobs in index
/// order would have made them. Loads do not overlap: one begun inside another
/// is refused by name and numbers nothing.
pub(crate) struct Load {
    id: u64,
}

impl Load {
    /// A load begins; its buffers are numbered from the counter as it stands.
    pub(crate) fn begin() -> Self {
        let mut state = state();
        let id = state.loads;
        state.loads += 1;
        if state.load.is_some() {
            state.refuse(NESTED);
        } else {
            let base = state.made;
            state.load = Some(OpenLoad {
                id,
                base,
                made: BTreeMap::new(),
            });
        }
        Self { id }
    }

    /// Job `index` runs `run`, on this thread, start to end.
    pub(crate) fn job<R>(&self, index: usize, run: impl FnOnce() -> R) -> R {
        let _scope = Scope::enter(self.id, index);
        run()
    }

    /// The load ends with its jobs below `counted` having run: the buffers of
    /// those are numbered, the others dropped.
    pub(crate) fn end(self, counted: usize) {
        state().close(self.id, counted);
    }
}

/// A load dropped without [`Load::end`] (it unwound) counts every job it ran.
impl Drop for Load {
    fn drop(&mut self) {
        state().close(self.id, usize::MAX);
    }
}

/// The job scope of the running thread, restored when it ends.
struct Scope {
    previous: Option<(u64, usize)>,
}

impl Scope {
    fn enter(load: u64, job: usize) -> Self {
        Self {
            previous: SCOPE.replace(Some((load, job))),
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPE.set(self.previous);
    }
}

/// A command buffer was made.
pub(crate) fn command_buffer_made(buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
    let mut state = state();
    // A new command buffer at an address a finished one had is a new order.
    let address = core::ptr::from_ref(buffer).addr();
    state.command_buffers.remove(&address);
    let n = state.command_buffer(address);
    state.emit(|| format!("cb {n} begin"));
}

/// A command buffer encodes a wait for the event `name` to reach `value`.
pub(crate) fn wait_event(buffer: &ProtocolObject<dyn MTLCommandBuffer>, name: &str, value: u64) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} wait-event {name}={value}"));
}

/// A command buffer encodes a signal of the event `name` at `value`.
pub(crate) fn signal_event(buffer: &ProtocolObject<dyn MTLCommandBuffer>, name: &str, value: u64) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} signal {name}={value}"));
}

/// A command buffer was committed.
pub(crate) fn commit(buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} commit"));
}

/// The host waits: `what` is `settle V` or `sync`.
pub(crate) fn host(what: impl FnOnce() -> String) {
    state().emit(|| format!("host {what}", what = what()));
}

/// What one compute encoder has been asked: the trace's side of
/// [`crate::encode::ComputeEncoder`].
pub(crate) struct EncoderTrace {
    /// The encoder's name, `N.M`: command buffer N, its Mth encoder.
    name: String,
    pipeline: Option<String>,
    bound: [Option<String>; MAX_BINDINGS],
    dispatches: u64,
}

impl EncoderTrace {
    /// An encoder just made on `buffer` of dispatch type `dispatch`.
    pub(crate) fn open(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: MTLDispatchType,
    ) -> Self {
        let mut state = state();
        let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
        let m = *state.encoders.entry(n).and_modify(|m| *m += 1).or_insert(0);
        let name = format!("{n}.{m}");
        let kind = if dispatch == MTLDispatchType::Concurrent {
            "concurrent"
        } else {
            "serial"
        };
        state.emit(|| format!("enc {name} open {kind}"));
        Self {
            name,
            pipeline: None,
            bound: [const { None }; MAX_BINDINGS],
            dispatches: 0,
        }
    }

    /// The pipeline the next dispatches run.
    pub(crate) fn set_pipeline(&mut self, pipeline: &Pipeline) {
        let state = pipeline.metal();
        self.pipeline = Some(format!(
            "{} max={} width={} tg_static={}",
            pipeline.label(),
            pipeline.max_threads_per_group(),
            state.threadExecutionWidth(),
            state.staticThreadgroupMemoryLength()
        ));
    }

    /// Slot `index` holds a window of `buffer` from `offset` bytes.
    pub(crate) fn set_buffer(
        &mut self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) {
        let address = core::ptr::from_ref(buffer).addr();
        let mut state = state();
        let name = match state.buffers.get(&address) {
            Some(&(Name::Number(n), length)) => format!("#{n}+{offset}/{length}"),
            // Bound while its load runs: no number yet, and no id a compare reads.
            Some(&(Name::Placed { job, k }, length)) => format!("#!j{job}.{k}+{offset}/{length}"),
            None => {
                let next = state.strangers.len() as u64;
                let n = *state.strangers.entry(address).or_insert(next);
                format!("#?{n}+{offset}/{}", buffer.length())
            }
        };
        self.bound[index] = Some(name);
    }

    /// Slot `index` holds the value `data`.
    pub(crate) fn set_bytes(&mut self, index: usize, data: &[u8]) {
        self.bound[index] = Some(format!("bytes:{}", gguf::digest::hex(data)));
    }

    /// A dispatch of `groups_x` by `groups_y` groups of `threads` threads.
    pub(crate) fn dispatch(&mut self, groups_x: usize, groups_y: usize, threads: usize) {
        let k = self.dispatches;
        self.dispatches += 1;
        let mut line = format!(
            "dispatch {}.{k} {} grid={groups_x}x{groups_y} tpg={threads}",
            self.name,
            self.pipeline.as_deref().unwrap_or("-"),
        );
        for (index, slot) in self.bound.iter().enumerate() {
            if let Some(slot) = slot {
                let _ = write!(line, " b{index}={slot}");
            }
        }
        state().emit(|| line);
    }

    /// A barrier.
    pub(crate) fn barrier(&self) {
        state().emit(|| format!("barrier {}", self.name));
    }
}

impl Drop for EncoderTrace {
    fn drop(&mut self) {
        state().emit(|| format!("enc {} end", self.name));
    }
}

/// What the numbering's tests share: the process-wide state is reset and held
/// by one test at a time.
#[cfg(test)]
pub(crate) mod testing {
    use std::collections::BTreeMap;
    use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
    use std::time::Duration;

    use super::{Name, State, state};

    /// The state, reset, held until the guard drops.
    pub(crate) fn session() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        *state() = State::new();
        guard
    }

    /// Address to (number, length) of every buffer named; a buffer a load has
    /// not numbered yet is a failure.
    pub(crate) fn names() -> BTreeMap<usize, (u64, usize)> {
        state()
            .buffers
            .iter()
            .map(|(&address, &(name, length))| match name {
                Name::Number(n) => (address, (n, length)),
                Name::Placed { job, k } => {
                    panic!("buffer {k} of job {job} at {address:#x} was never numbered")
                }
            })
            .collect()
    }

    /// The buffers that got no number, by reason.
    pub(crate) fn refused() -> BTreeMap<&'static str, u64> {
        state().refused.clone()
    }

    /// An order the buffers of a load are made in, across threads: each
    /// `(job, k)` event waits for its turn, so an interleaving is the script's
    /// and no timing's.
    pub(crate) struct Script {
        order: Vec<(usize, u64)>,
        at: Mutex<usize>,
        turn: Condvar,
    }

    impl Script {
        pub(crate) fn new(order: Vec<(usize, u64)>) -> Self {
            Self {
                order,
                at: Mutex::new(0),
                turn: Condvar::new(),
            }
        }

        /// Runs `event` as the script's next step, which must be `(job, k)`.
        pub(crate) fn step(&self, job: usize, k: u64, event: impl FnOnce()) {
            let at = self.at.lock().unwrap_or_else(PoisonError::into_inner);
            let (mut at, waited) = self
                .turn
                .wait_timeout_while(at, Duration::from_secs(60), |at| {
                    self.order.get(*at) != Some(&(job, k))
                })
                .unwrap_or_else(PoisonError::into_inner);
            assert!(
                !waited.timed_out(),
                "the script never reached ({job}, {k}): step {at} is {:?}",
                self.order.get(*at)
            );
            event();
            *at += 1;
            self.turn.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::testing::{Script, names, refused, session};
    use super::{BY_NO_JOB, LOST_JOB, Load, NESTED, OFF_THREAD, buffer_made, finish};

    /// The buffers each job of the tests makes; job 2 makes none.
    const MADE: [u64; 5] = [2, 1, 0, 3, 2];

    /// Where buffer `k` of job `j` lives, and its length.
    fn address(j: usize, k: u64) -> usize {
        0x10_0000 + j * 0x1000 + usize::try_from(k).unwrap() * 0x10
    }
    fn length(j: usize, k: u64) -> usize {
        64 * (j + 1) + usize::try_from(k).unwrap()
    }

    /// What one thread running the jobs in index order, after `before` buffers,
    /// names them: job-major.
    fn in_index_order(before: u64) -> BTreeMap<usize, (u64, usize)> {
        let mut names = BTreeMap::new();
        let mut n = before;
        for (j, &made) in MADE.iter().enumerate() {
            for k in 0..made {
                names.insert(address(j, k), (n, length(j, k)));
                n += 1;
            }
        }
        names
    }

    /// The names after a load of [`MADE`]'s jobs, one thread each, made in the
    /// order of `order`, with two buffers made before it and one after.
    fn load_in(order: &[(usize, u64)]) -> BTreeMap<usize, (u64, usize)> {
        let _session = session();
        buffer_made(0x100, 8);
        buffer_made(0x200, 16);
        let load = Load::begin();
        let script = Script::new(order.to_vec());
        std::thread::scope(|s| {
            for (j, &made) in MADE.iter().enumerate() {
                let (load, script) = (&load, &script);
                s.spawn(move || {
                    load.job(j, || {
                        for k in 0..made {
                            script.step(j, k, || buffer_made(address(j, k), length(j, k)));
                        }
                    });
                });
            }
        });
        load.end(MADE.len());
        buffer_made(0x300, 32);
        assert!(refused().is_empty(), "{:?}", refused());
        names()
    }

    /// A buffer's name does not depend on which thread got to it first: the
    /// same names for jobs made in order, in reverse, and interleaved, and
    /// they are the names one thread running the jobs in order gives. Mutant:
    /// one counter for every buffer (the interleavings differ).
    #[test]
    fn a_loads_buffers_are_numbered_in_job_order_whatever_the_interleaving() {
        let mut expected = in_index_order(2);
        expected.extend([(0x100, (0, 8)), (0x200, (1, 16)), (0x300, (10, 32))]);
        let in_order = [
            (0, 0),
            (0, 1),
            (1, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
        ];
        let reversed = [
            (4, 0),
            (4, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (1, 0),
            (0, 0),
            (0, 1),
        ];
        let mixed = [
            (4, 0),
            (3, 0),
            (0, 0),
            (4, 1),
            (1, 0),
            (3, 1),
            (0, 1),
            (3, 2),
        ];
        for (name, order) in [
            ("in order", in_order),
            ("reversed", reversed),
            ("mixed", mixed),
        ] {
            assert_eq!(load_in(&order), expected, "{name}");
        }
    }

    /// Two buffers at one address: the one made later, in time, holds the
    /// name, whichever job it belongs to. Mutant: the first buffer made at an
    /// address keeps it.
    #[test]
    fn the_buffer_made_last_at_an_address_holds_the_name() {
        let _session = session();
        let load = Load::begin();
        // Job 5 made a buffer there and freed it; job 3's then took the address.
        load.job(5, || buffer_made(0x1000, 64));
        load.job(3, || buffer_made(0x1000, 128));
        load.end(6);
        assert_eq!(names(), BTreeMap::from([(0x1000, (0, 128))]));
    }

    /// The jobs a failed load did not need to run are not counted, and nor
    /// are the buffers a load that unwound made after the point it stopped.
    #[test]
    fn a_load_counts_the_jobs_it_is_told_to() {
        let _session = session();
        let load = Load::begin();
        for j in 0..4 {
            load.job(j, || buffer_made(0x1000 + j * 0x10, 8));
        }
        load.end(2);
        assert_eq!(
            names(),
            BTreeMap::from([(0x1000, (0, 8)), (0x1010, (1, 8))])
        );
        // The next buffer follows the two counted.
        buffer_made(0x2000, 8);
        assert_eq!(names()[&0x2000], (2, 8));
        // A load dropped before it ends counts every job it ran.
        let load = Load::begin();
        load.job(0, || buffer_made(0x3000, 8));
        drop(load);
        buffer_made(0x4000, 8);
        assert_eq!((names()[&0x3000].0, names()[&0x4000].0), (3, 4));
    }

    /// A buffer with no place in the order is not numbered and the trace says
    /// so: made on another thread outside a load, inside a load by no job, by
    /// a job of a load that is not the open one, or by a load begun inside
    /// another. The names around it do not move. Mutant: it takes a number.
    #[test]
    fn a_buffer_with_no_place_is_refused_by_name() {
        let _session = session();
        buffer_made(0x100, 8);
        std::thread::scope(|s| {
            s.spawn(|| buffer_made(0x200, 8));
        });
        buffer_made(0x300, 8);
        assert_eq!(names(), BTreeMap::from([(0x100, (0, 8)), (0x300, (1, 8))]));
        assert_eq!(refused(), BTreeMap::from([(OFF_THREAD, 1)]));

        let load = Load::begin();
        buffer_made(0x400, 8);
        let nested = Load::begin();
        nested.job(0, || buffer_made(0x500, 8));
        load.job(0, || buffer_made(0x600, 8));
        nested.end(1);
        load.end(1);
        assert_eq!(
            names(),
            BTreeMap::from([(0x100, (0, 8)), (0x300, (1, 8)), (0x600, (2, 8))])
        );
        assert_eq!(
            refused(),
            BTreeMap::from([(OFF_THREAD, 1), (BY_NO_JOB, 1), (NESTED, 1), (LOST_JOB, 1)])
        );
        let text = finish();
        assert!(text.starts_with(super::VERSION), "{text}");
        for reason in [OFF_THREAD, BY_NO_JOB, NESTED, LOST_JOB] {
            assert!(text.contains(&format!("\nrefused {reason}: 1\n")), "{text}");
        }
    }
}
