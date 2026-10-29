//! Read-ahead of the file bytes a load is about to copy: the counterpart of
//! [`PageDrop`](super::host_lock::PageDrop), which releases them once copied.
//!
//! A card upload first touches a segment's file pages inside its staging copy,
//! so the drive serves the faults of one copy at a time, a read-around each,
//! and sits idle between segments. [`run`] reads the pages of the segments
//! after the one being copied in ahead of the copy, on helper threads that each
//! take a chunk of a segment's bytes at a time (`MADV_POPULATE_READ`, the call
//! the host set's walk uses, which also maps the pages, so the copy takes no
//! fault). The segments are the load's steps, in the order it copies them.
//!
//! What waits for what: the copy never waits for a read it could do itself (a
//! page a helper has not reached is faulted by the copy as before); a helper
//! never reads a step the load has [retired](ReadAhead::retire); and a retire
//! waits for the chunks being read of the step, so that the release after it
//! does not race a read that would put the pages back.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

use gguf::{Gguf, Split};
use os::Populated;

use super::host_lock::{NO_POPULATE_READ, WALK_CHUNKS, advise_populate, page_bytes, populate_span};

/// The file bytes of one step, as `(shard, file offsets)` runs.
pub type Ranges = Vec<(usize, Range<u64>)>;

/// The most bytes of steps read ahead of the step being copied, past the
/// step's own: a fraction of a second of an NVMe drive, enough that the drive
/// never waits for the copy to finish a step, and little enough of the page
/// cache that the pages read and not yet copied cost the host set nothing it
/// would miss. One step past the window may
/// be queued whole (a step larger than the window still reads ahead), so the
/// bound is the window plus the largest step.
pub const WINDOW_BYTES: u64 = 1 << 30;

/// The bytes one helper reads in one call: 8 MiB amortises the call over
/// several of the drive's own request sizes, and 16 helpers keep 128 MiB in
/// flight, far past what the drive needs to run at its rate.
pub const CHUNK_BYTES: u64 = 8 << 20;

/// What a [`run`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The bytes of mapping the helpers finished reading, whole pages.
    pub advised_bytes: u64,
    /// The most steps queued past the step being copied at any [`ReadAhead::begin`].
    pub peak_ahead_steps: usize,
    /// The bytes of those steps.
    pub peak_ahead_bytes: u64,
}

/// One chunk to read: bytes `bytes` of shard `shard`'s mapping, whole pages.
#[derive(Clone, Debug)]
struct Job {
    step: usize,
    shard: usize,
    bytes: Range<usize>,
}

struct State {
    queue: VecDeque<Job>,
    /// Steps queued so far: `0..issued`.
    issued: usize,
    /// Steps before this one are retired: their queued chunks are dropped.
    consumed: usize,
    /// The step each helper is reading a chunk of.
    running: Vec<Option<usize>>,
    closed: bool,
    failed: Option<String>,
    stats: Stats,
}

/// The reads of one load, as the body of [`run`] drives them.
pub struct ReadAhead<'a> {
    maps: &'a [Option<&'a [u8]>],
    advise: &'a (dyn Fn(&[u8]) -> Result<(), String> + Sync),
    jobs: Vec<Vec<Job>>,
    /// `prefix[i]` is the bytes of steps `0..i`.
    prefix: Vec<u64>,
    window: u64,
    state: Mutex<State>,
    wake: Condvar,
}

impl ReadAhead<'_> {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'g>(&self, guard: MutexGuard<'g, State>) -> MutexGuard<'g, State> {
        self.wake
            .wait(guard)
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The load is about to copy `step`: queue every step the window holds
    /// after it, and `step` itself if nothing has yet. Returns at once; the
    /// reads happen on the helpers. A read that failed is refused here, by name.
    pub fn begin(&self, step: usize) -> Result<(), String> {
        let n = self.jobs.len();
        if step >= n {
            return Err(format!("read-ahead step {step} of {n}"));
        }
        let mut st = self.lock();
        if let Some(e) = &st.failed {
            return Err(e.clone());
        }
        while st.issued < n
            && (st.issued <= step || self.prefix[st.issued] - self.prefix[step + 1] < self.window)
        {
            let i = st.issued;
            st.queue.extend(self.jobs[i].iter().cloned());
            st.issued += 1;
        }
        let ahead = st.issued.saturating_sub(step + 1);
        let bytes = if ahead > 0 {
            self.prefix[st.issued] - self.prefix[step + 1]
        } else {
            0
        };
        st.stats.peak_ahead_steps = st.stats.peak_ahead_steps.max(ahead);
        st.stats.peak_ahead_bytes = st.stats.peak_ahead_bytes.max(bytes);
        drop(st);
        self.wake.notify_all();
        Ok(())
    }

    /// The load has copied `step`, and releases its pages next: drop its chunks
    /// still queued and wait for the ones being read, so that no read puts
    /// back a page the release is about to drop.
    pub fn retire(&self, step: usize) {
        let mut st = self.lock();
        st.consumed = st.consumed.max(step + 1);
        while st.running.iter().any(|r| r.is_some_and(|s| s <= step)) {
            st = self.wait(st);
        }
    }

    fn fail(&self, why: String) {
        let mut st = self.lock();
        st.failed.get_or_insert(why);
        st.queue.clear();
        drop(st);
        self.wake.notify_all();
    }

    fn close(&self) {
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// One helper's loop: take the next chunk of a live step, read it, repeat
    /// until the queue is closed and empty.
    fn work(&self, slot: usize) {
        if let Err(e) = threads::helper::float_caller() {
            self.fail(format!("read-ahead helper {slot}: {e}"));
            return;
        }
        loop {
            let job = {
                let mut st = self.lock();
                loop {
                    match st.queue.pop_front() {
                        Some(j) if j.step < st.consumed => {}
                        Some(j) => {
                            st.running[slot] = Some(j.step);
                            break Some(j);
                        }
                        None if st.closed => break None,
                        None => st = self.wait(st),
                    }
                }
            };
            let Some(job) = job else { return };
            let span = self.maps[job.shard].map(|m| &m[job.bytes.clone()]);
            let read = span.map_or(Ok(()), self.advise);
            let mut st = self.lock();
            st.running[slot] = None;
            match read {
                Ok(()) => st.stats.advised_bytes += job.bytes.len() as u64,
                Err(e) => {
                    let why = format!(
                        "read-ahead of shard {} bytes {}..{}: {e}",
                        job.shard, job.bytes.start, job.bytes.end
                    );
                    st.failed.get_or_insert(why);
                    st.queue.clear();
                }
            }
            drop(st);
            self.wake.notify_all();
        }
    }
}

/// Closes the queue when dropped, so that a body that unwinds still lets the
/// helpers end and the scope that waits for them return.
struct Closing<'r, 'a>(&'r ReadAhead<'a>);

impl Drop for Closing<'_, '_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// `steps`' file bytes cut into whole-page chunks of [`CHUNK_BYTES`], a list a
/// step; a shard with no mapping (`None`: its bytes are not a file's pages) has
/// nothing to read ahead, and bytes past a mapping's end are refused by name.
fn jobs_of(maps: &[Option<&[u8]>], page: u64, steps: &[Ranges]) -> Result<Vec<Vec<Job>>, String> {
    let chunk = CHUNK_BYTES.max(page);
    let mut out = Vec::with_capacity(steps.len());
    for (step, ranges) in steps.iter().enumerate() {
        let mut jobs = Vec::new();
        for (shard, at) in ranges {
            let map = match maps.get(*shard) {
                None => return Err(format!("shard {shard} is not in the split")),
                Some(None) => continue,
                Some(Some(map)) => map,
            };
            let len = map.len() as u64;
            if at.start > at.end || at.end > len {
                return Err(format!(
                    "shard {shard} bytes {}..{} run past its {len} bytes",
                    at.start, at.end
                ));
            }
            if at.start == at.end {
                continue;
            }
            let end = (at.end.div_ceil(page) * page).min(len);
            let mut from = at.start / page * page;
            while from < end {
                let to = (from + chunk).min(end);
                // Both fit usize: they are at most the mapping's length.
                jobs.push(Job {
                    step,
                    shard: *shard,
                    bytes: from as usize..to as usize,
                });
                from = to;
            }
        }
        out.push(jobs);
    }
    Ok(out)
}

/// [`run`] over given mappings and a given read, for a test to count with.
fn run_over<T, E>(
    maps: &[Option<&[u8]>],
    page: u64,
    steps: &[Ranges],
    window: u64,
    advise: &(dyn Fn(&[u8]) -> Result<(), String> + Sync),
    refuse: impl Fn(String) -> E,
    body: impl FnOnce(&ReadAhead<'_>) -> Result<T, E>,
) -> Result<(T, Stats), E> {
    let jobs = jobs_of(maps, page, steps).map_err(&refuse)?;
    let mut prefix = Vec::with_capacity(jobs.len() + 1);
    prefix.push(0u64);
    for step in &jobs {
        let bytes: u64 = step.iter().map(|j| j.bytes.len() as u64).sum();
        prefix.push(prefix[prefix.len() - 1] + bytes);
    }
    let chunks: usize = jobs.iter().map(Vec::len).sum();
    let helpers = chunks.min(usize::try_from(WALK_CHUNKS).unwrap_or(1));
    let ahead = ReadAhead {
        maps,
        advise,
        jobs,
        prefix,
        window,
        state: Mutex::new(State {
            queue: VecDeque::new(),
            issued: 0,
            consumed: 0,
            running: vec![None; helpers],
            closed: false,
            failed: None,
            stats: Stats::default(),
        }),
        wake: Condvar::new(),
    };
    let done = thread::scope(|scope| {
        for slot in 0..helpers {
            let ahead = &ahead;
            let spawned = thread::Builder::new()
                .name(format!("read-ahead-{slot}"))
                .spawn_scoped(scope, move || ahead.work(slot));
            if let Err(e) = spawned {
                ahead.fail(format!("read-ahead helper {slot} did not spawn: {e}"));
                break;
            }
        }
        let _closing = Closing(&ahead);
        body(&ahead)
    });
    let value = done?;
    let st = ahead.lock();
    match &st.failed {
        Some(e) => Err(refuse(e.clone())),
        None => Ok((value, st.stats)),
    }
}

/// Run `body` over a read-ahead of `steps` in `split`'s mappings, then join
/// the helpers. `body` calls [`ReadAhead::begin`] before it copies each step
/// and [`ReadAhead::retire`] after, in step order. A failed read, after
/// `body`'s own result, is `refuse`'s error. The helpers float off the
/// caller's affinity (`threads::helper::float_caller`), as a binary pins its
/// load thread to one cpu.
pub fn run<T, E>(
    split: &Split,
    steps: &[Ranges],
    refuse: impl Fn(String) -> E,
    body: impl FnOnce(&ReadAhead<'_>) -> Result<T, E>,
) -> Result<(T, Stats), E> {
    let page = page_bytes().map_err(|e| refuse(e.to_string()))?;
    let maps: Vec<Option<&[u8]>> = (0..split.shard_count())
        .map(|s| {
            split
                .shard(s)
                .filter(|g| g.file_backed())
                .map(Gguf::mapping)
        })
        .collect();
    let said = AtomicBool::new(false);
    let advise = |span: &[u8]| -> Result<(), String> {
        if populate_span(span, page, advise_populate)? == Populated::Touched
            && !said.swap(true, Ordering::Relaxed)
        {
            eprintln!("{NO_POPULATE_READ}");
        }
        Ok(())
    };
    run_over(&maps, page, steps, WINDOW_BYTES, &advise, refuse, body)
}

#[cfg(test)]
mod tests {
    use super::{Ranges, Stats, run_over};
    use std::ops::Range;
    use std::sync::{Condvar, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    const PAGE: u64 = 4096;
    const MIB: usize = 1 << 20;

    /// The reads a fake mapping saw, as byte offsets from its start, and a bell
    /// the test waits on.
    struct Seen {
        base: usize,
        spans: Mutex<Vec<Range<usize>>>,
        bell: Condvar,
    }

    impl Seen {
        fn of(map: &[u8]) -> Seen {
            Seen {
                base: map.as_ptr() as usize,
                spans: Mutex::new(Vec::new()),
                bell: Condvar::new(),
            }
        }

        fn note(&self, span: &[u8]) {
            let at = span.as_ptr() as usize - self.base;
            self.spans
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(at..at + span.len());
            self.bell.notify_all();
        }

        /// Whether every page of `want` has been read, waiting up to 10 s.
        fn read(&self, want: Range<usize>) -> bool {
            let until = Instant::now() + Duration::from_secs(10);
            let mut spans = self.spans.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                let all = (want.start..want.end)
                    .step_by(PAGE as usize)
                    .all(|p| spans.iter().any(|s| s.contains(&p)));
                if all {
                    return true;
                }
                let left = until.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                spans = self
                    .bell
                    .wait_timeout(spans, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
    }

    /// Four steps of 3 MiB in a 16 MiB shard, a window of 4 MiB: while a step is
    /// being copied the next is read already (the overlap), the steps queued
    /// past the one copied stay within the window and one step over it, and every
    /// step is read once before it is retired.
    #[test]
    fn reads_the_steps_after_the_one_copied_within_the_window() {
        let map = vec![0u8; 16 * MIB];
        let maps = [Some(&map[..])];
        let range = |i: usize| 3 * MIB * i..3 * MIB * (i + 1);
        let steps: Vec<Ranges> = (0..4)
            .map(|i| vec![(0, range(i).start as u64..range(i).end as u64)])
            .collect();
        let seen = Seen::of(&map);
        let advise = |span: &[u8]| -> Result<(), String> {
            seen.note(span);
            Ok(())
        };
        let (_, stats) = run_over::<(), String>(
            &maps,
            PAGE,
            &steps,
            4 << 20,
            &advise,
            |e| e,
            |ahead| {
                for i in 0..4 {
                    ahead.begin(i)?;
                    assert!(seen.read(range(i)), "step {i} was not read");
                    if i + 1 < 4 {
                        assert!(
                            seen.read(range(i + 1)),
                            "step {} was not read while step {i} is being copied",
                            i + 1
                        );
                    }
                    ahead.retire(i);
                }
                Ok(())
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            stats,
            Stats {
                advised_bytes: 12 << 20,
                peak_ahead_steps: 2,
                peak_ahead_bytes: 6 << 20,
            }
        );
    }

    /// A read that fails ends the load at its next `begin`, naming the shard
    /// and bytes, and a bound past the mapping is refused before any thread
    /// starts.
    #[test]
    fn refuses_a_failed_read_and_a_range_past_the_mapping_by_name() {
        let map = vec![0u8; 8 * MIB];
        let maps = [Some(&map[..])];
        let steps: Vec<Ranges> = vec![vec![(0, 0..4 << 20)], vec![(0, 4 << 20..8 << 20)]];
        let broken = |_: &[u8]| -> Result<(), String> { Err("EIO".to_string()) };
        let err = run_over::<(), String>(
            &maps,
            PAGE,
            &steps,
            u64::MAX,
            &broken,
            |e| e,
            |ahead| {
                ahead.begin(0)?;
                let until = Instant::now() + Duration::from_secs(10);
                loop {
                    ahead.begin(1)?;
                    assert!(Instant::now() < until, "the failed read was never refused");
                    std::thread::sleep(Duration::from_millis(1));
                }
            },
        )
        .unwrap_err();
        assert!(err.starts_with("read-ahead of shard 0 bytes "), "{err}");
        assert!(err.ends_with("EIO"), "{err}");
        let past: Vec<Ranges> = vec![vec![(0, 0..(8 << 20) + 1)]];
        let err = run_over::<(), String>(&maps, PAGE, &past, 1 << 30, &broken, |e| e, |_| Ok(()))
            .unwrap_err();
        assert_eq!(err, "shard 0 bytes 0..8388609 run past its 8388608 bytes");
    }
}
