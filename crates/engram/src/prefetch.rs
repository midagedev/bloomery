//! A helper thread that turns one token's engram read into a memcpy.
//!
//! The engine's step is its calling thread's serial time, so the question this
//! module answers is not "how long does the read take" but "how many
//! microseconds does the read steal from the step thread". Everything the read
//! costs — the `WILLNEED` syscall per row, the optional `MADV_POPULATE_READ`
//! pass, and the minor fault per row that consuming through the mapping takes —
//! is moved onto a helper. What crosses back is a buffer of plain bytes.
//!
//! The handoff is by ownership, not by sharing: two buffers circulate over a
//! pair of channels, so there is no lock on the step thread's path and no
//! `unsafe`. [`Prefetcher::submit`] gives the helper the next token's ids and
//! one free buffer; [`Prefetcher::wait`] takes the filled one back.
//!
//! A token may ask for fewer rows of a site than the prefetcher was sized for —
//! a cache in front of it hands over only its misses — and a token that asks
//! for none never reaches the helper: its `wait` returns at once.
//!
//! Steady state allocates nothing: the id vectors travel with the buffers and
//! are refilled in place, and the buffers are written once at construction so
//! that the first token's faults are the table's and not the allocator's.
//!
//! Where the helper runs is the owner's call ([`HelperOptions::cpu`]): a
//! helper that shares a core with a pool worker slows the worker's chunk by
//! more than the read it saves. [`caller_sibling`] names the one slot that is
//! free by construction for an owner that blocks on [`Prefetcher::wait`]: the
//! SMT sibling of the owner's own pinned core, idle while the owner waits.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::{Engram, EngramError, faults_thread};

/// How [`Prefetcher::with_options`] starts its helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HelperOptions {
    pub mode: FillMode,
    /// The logical cpu the helper pins itself to; `None` leaves it floating.
    pub cpu: Option<usize>,
    /// Count per token the rows whose pages were all in the page cache before
    /// the advise ([`crate::Site::resident_rows`], one `mincore` per row).
    /// Off, the helper makes no such call.
    pub classify: bool,
}

impl HelperOptions {
    /// A floating helper that classifies nothing.
    #[must_use]
    pub fn of(mode: FillMode) -> HelperOptions {
        HelperOptions {
            mode,
            cpu: None,
            classify: false,
        }
    }
}

/// What the helper does between advising the rows and copying them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMode {
    /// Nothing: the copy itself faults the pages in. One minor fault per row,
    /// on the helper.
    Touch,
    /// [`crate::Site::populate`] over the same rows first, so the copy finds
    /// every page already mapped. Only the user-mode trap per fault goes away —
    /// the kernel's fault work is the same work, done in a batch.
    Populate,
}

/// One token's work and the storage it is done into. Both travel with the job
/// so that neither side allocates per token.
struct Job {
    /// Row ids per site, in site order. Refilled in place by `submit`.
    ids: Vec<Vec<u32>>,
    /// Sites' rows concatenated in site order.
    buf: Box<[u8]>,
    /// The bytes of `buf` this token's rows fill; zero once a fill failed.
    len: usize,
}

struct Filled {
    job: Job,
    err: Option<EngramError>,
    /// Travels with the buffer rather than in a shared cell. A shared cell
    /// would be read after `recv` returns, by which time the helper has already
    /// taken the next queued token and begun overwriting it — the pipelined
    /// caller keeps one submitted at all times, so that window is always open.
    times: HelperTimes,
}

/// What the helper spent on the token [`Prefetcher::wait`] just returned, and
/// the faults it has taken since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HelperTimes {
    pub submit_ns: u64,
    pub fill_ns: u64,
    pub copy_ns: u64,
    pub major_faults: u64,
    pub minor_faults: u64,
    /// Rows the token asked for.
    pub rows: u64,
    /// Of those, the rows already in the page cache before the advise; zero
    /// unless the helper classifies ([`HelperOptions::classify`]).
    pub resident_rows: u64,
    /// The classification's own time, before `submit_ns`; zero unless the
    /// helper classifies.
    pub classify_ns: u64,
}

/// What the next [`Prefetcher::wait`] takes back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InFlight {
    /// A job on the helper.
    Helper,
    /// A token with no rows: nothing was sent.
    Empty,
}

/// A helper thread reading one token ahead for its owner.
pub struct Prefetcher {
    /// `None` only inside `drop`, which closes the channel to end the helper.
    jobs: Option<Sender<Job>>,
    done: Receiver<Filled>,
    /// Buffers not in flight. Two of them: one being filled, one being read.
    spare: Vec<Job>,
    /// The token `wait` last took back; what `filled` lends out.
    current: Option<Job>,
    /// The token submitted and not yet waited for, if any.
    in_flight: Option<InFlight>,
    /// Per site: the most rows one token may ask, and one row's bytes.
    max_rows: Vec<usize>,
    row_bytes: Vec<usize>,
    last: HelperTimes,
    /// The cpu the helper pinned itself to, if it was asked to and could.
    cpu: Option<usize>,
    handle: Option<JoinHandle<()>>,
}

impl Prefetcher {
    /// Start a helper for `engram`, sized for at most `max_rows_per_site[s]`
    /// rows of site `s` per token.
    ///
    /// `max_rows_per_site` must name every site. The helper holds an [`Arc`] of
    /// the table, so the mappings outlive it.
    pub fn new(
        engram: Arc<Engram>,
        max_rows_per_site: &[usize],
        mode: FillMode,
    ) -> Result<Prefetcher, EngramError> {
        Prefetcher::with_options(engram, max_rows_per_site, HelperOptions::of(mode))
    }

    /// [`Prefetcher::new`] with the helper's cpu and classification named.
    /// Returns once the helper has pinned itself (or failed to: a container
    /// may refuse, and the helper then floats — [`Prefetcher::pinned_cpu`]
    /// says which).
    pub fn with_options(
        engram: Arc<Engram>,
        max_rows_per_site: &[usize],
        options: HelperOptions,
    ) -> Result<Prefetcher, EngramError> {
        let sites = engram.sites();
        if max_rows_per_site.len() != sites.len() {
            return Err(EngramError::Prefetch(
                "max_rows_per_site must have one entry per site",
            ));
        }
        let row_bytes: Vec<usize> = sites.iter().map(|s| s.row_bytes() as usize).collect();
        let buf_len: usize = max_rows_per_site
            .iter()
            .zip(&row_bytes)
            .map(|(n, rb)| n * rb)
            .sum();

        // Written, not just allocated: `vec![0u8; n]` is lazily zero-mapped, and
        // the first write to it would fault on whichever thread got there first
        // and land in the token counts this module exists to separate.
        let spare: Vec<Job> = (0..2)
            .map(|_| Job {
                ids: max_rows_per_site
                    .iter()
                    .map(|&n| Vec::with_capacity(n))
                    .collect(),
                buf: vec![0xA5u8; buf_len].into_boxed_slice(),
                len: 0,
            })
            .collect();

        let (job_tx, job_rx) = channel::<Job>();
        let (done_tx, done_rx) = channel::<Filled>();
        let (pin_tx, pin_rx) = channel::<Option<usize>>();
        let handle = std::thread::Builder::new()
            .name("engram-prefetch".into())
            .spawn(move || {
                let pinned = options.cpu.filter(|&c| pin_to(c));
                // The owner waits on this before its first submit; a closed
                // receiver means it already gave up, and the jobs channel
                // tells the loop the same.
                let _ = pin_tx.send(pinned);
                helper(&engram, options, &job_rx, &done_tx);
            })?;
        let cpu = pin_rx
            .recv()
            .map_err(|_| EngramError::Prefetch("helper thread exited at start"))?;

        Ok(Prefetcher {
            jobs: Some(job_tx),
            done: done_rx,
            spare,
            current: None,
            in_flight: None,
            max_rows: max_rows_per_site.to_vec(),
            row_bytes,
            last: HelperTimes::default(),
            cpu,
            handle: Some(handle),
        })
    }

    /// The cpu the helper runs pinned to; `None` when it floats.
    #[must_use]
    pub fn pinned_cpu(&self) -> Option<usize> {
        self.cpu
    }

    /// Hand the helper one token's row ids and let it run.
    ///
    /// Returns as soon as the ids are copied into a free buffer — this is the
    /// step thread's whole share of the read. Each site may ask for any number
    /// of rows up to the prefetcher's size for it, and a token that asks for
    /// none is not sent at all. Two submits in a row without a
    /// [`Prefetcher::wait`] between them is a caller bug, not a runtime
    /// condition. `token_ids` yields one id list per site, in site order.
    pub fn submit<S: AsRef<[u32]>>(
        &mut self,
        token_ids: impl IntoIterator<Item = S>,
    ) -> Result<(), EngramError> {
        if self.in_flight.is_some() {
            return Err(EngramError::Prefetch(
                "a token is already in flight; wait() first",
            ));
        }
        let Some(mut job) = self.spare.pop() else {
            return Err(EngramError::Prefetch("no free buffer; wait() first"));
        };
        job.len = 0;
        let mut sites = 0usize;
        let mut refused = None;
        for (i, src) in token_ids.into_iter().enumerate() {
            let src = src.as_ref();
            sites = i + 1;
            // Checked before the copy: past its capacity an id vector would
            // reallocate, and the buffer would be too short for the rows.
            match (job.ids.get_mut(i), self.max_rows.get(i)) {
                (Some(dst), Some(&max)) if src.len() <= max => {
                    dst.clear();
                    dst.extend_from_slice(src);
                    job.len += src.len() * self.row_bytes[i];
                }
                (Some(_), Some(_)) => {
                    refused =
                        Some("token asks more rows of a site than the prefetcher was sized for");
                    break;
                }
                _ => {
                    refused = Some("token has a different site count");
                    break;
                }
            }
        }
        if refused.is_none() && sites != self.max_rows.len() {
            refused = Some("token has a different site count");
        }
        if let Some(why) = refused {
            self.spare.push(job);
            return Err(EngramError::Prefetch(why));
        }
        if job.len == 0 {
            self.spare.push(job);
            self.in_flight = Some(InFlight::Empty);
            return Ok(());
        }
        let Some(tx) = self.jobs.as_ref() else {
            self.spare.push(job);
            return Err(EngramError::Prefetch("helper is shutting down"));
        };
        tx.send(job)
            .map_err(|_| EngramError::Prefetch("helper thread exited"))?;
        self.in_flight = Some(InFlight::Helper);
        Ok(())
    }

    /// Block until the helper has finished the token submitted before this one,
    /// and take its buffer back.
    ///
    /// The bytes are then at [`Prefetcher::filled`] and the helper's own split
    /// at [`Prefetcher::last`]. Split from `filled` on purpose: the caller
    /// submits the next token while still holding these bytes, which a `wait`
    /// that returned the slice would forbid.
    ///
    /// A token that asked for no rows returns at once with nothing filled; the
    /// helper's times for it are zero and its fault counts carry over. A
    /// `wait` with nothing submitted is an error rather than a wait forever.
    pub fn wait(&mut self) -> Result<(), EngramError> {
        let Some(kind) = self.in_flight.take() else {
            return Err(EngramError::Prefetch("nothing in flight; submit() first"));
        };
        if kind == InFlight::Empty {
            if let Some(done) = self.current.take() {
                self.spare.push(done);
            }
            self.last = HelperTimes {
                submit_ns: 0,
                fill_ns: 0,
                copy_ns: 0,
                rows: 0,
                resident_rows: 0,
                classify_ns: 0,
                ..self.last
            };
            return Ok(());
        }
        let filled = self
            .done
            .recv()
            .map_err(|_| EngramError::Prefetch("helper thread exited"))?;
        self.last = filled.times;
        if let Some(done) = self.current.take() {
            self.spare.push(done);
        }
        let mut job = filled.job;
        if filled.err.is_some() {
            job.len = 0;
        }
        self.current = Some(job);
        match filled.err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The rows of the token [`Prefetcher::wait`] last took back: every site's
    /// rows concatenated in site order, `ids.len() * row_bytes()` per site.
    /// Empty before the first `wait`, after a token that asked for no rows, and
    /// after a fill that failed.
    pub fn filled(&self) -> &[u8] {
        self.current.as_ref().map_or(&[], |j| &j.buf[..j.len])
    }

    /// What the helper spent on that token.
    pub fn last(&self) -> HelperTimes {
        self.last
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Closing the channel is what ends the helper's receive loop; joining
        // after it is what keeps a dead round's helper from racing the next
        // one's on the same drive.
        self.jobs = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Advise, fill and copy one token at a time until the job channel closes.
///
/// Errors travel back with the buffer rather than ending the helper: the owner
/// is blocked in `wait` and has to be told, and the next token may well be
/// fine.
fn helper(engram: &Engram, options: HelperOptions, jobs: &Receiver<Job>, done: &Sender<Filled>) {
    let HelperOptions { mode, classify, .. } = options;
    let base = faults_thread();
    let sites = engram.sites();
    while let Ok(mut job) = jobs.recv() {
        let mut err = None;
        let rows: u64 = job.ids.iter().map(|ids| ids.len() as u64).sum();

        // Before the advise: afterwards every row's folio is in the cache,
        // its read in flight, and would count as resident.
        let (mut resident_rows, mut classify_ns) = (0u64, 0u64);
        if classify {
            let tc = Instant::now();
            for (site, ids) in sites.iter().zip(&job.ids) {
                match site.resident_rows(ids) {
                    Ok(n) => resident_rows += n,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
            classify_ns = tc.elapsed().as_nanos() as u64;
        }

        // WILLNEED over every row: one batch at the drive's queue depth.
        let t0 = Instant::now();
        for (site, ids) in sites.iter().zip(&job.ids) {
            if err.is_some() {
                break;
            }
            if let Err(e) = site.prefetch(ids) {
                err = Some(e);
                break;
            }
        }
        let submit_ns = t0.elapsed().as_nanos() as u64;

        let t1 = Instant::now();
        if err.is_none() && mode == FillMode::Populate {
            for (site, ids) in sites.iter().zip(&job.ids) {
                if let Err(e) = site.populate(ids) {
                    err = Some(e);
                    break;
                }
            }
        }
        let fill_ns = t1.elapsed().as_nanos() as u64;

        let t2 = Instant::now();
        if err.is_none() {
            let mut at = 0usize;
            for (site, ids) in sites.iter().zip(&job.ids) {
                let n = ids.len() * site.row_bytes() as usize;
                let Some(slot) = job.buf.get_mut(at..at + n) else {
                    err = Some(EngramError::Prefetch(
                        "token wants more bytes than the buffer",
                    ));
                    break;
                };
                if let Err(e) = site.copy_rows(ids, slot) {
                    err = Some(e);
                    break;
                }
                at += n;
            }
        }
        let copy_ns = t2.elapsed().as_nanos() as u64;

        let now = faults_thread();
        let times = HelperTimes {
            submit_ns,
            fill_ns,
            copy_ns,
            major_faults: now.major.saturating_sub(base.major),
            minor_faults: now.minor.saturating_sub(base.minor),
            rows,
            resident_rows,
            classify_ns,
        };
        if done.send(Filled { job, err, times }).is_err() {
            break;
        }
    }
}

/// The SMT sibling of the core the calling thread is pinned to, if the
/// thread's affinity is exactly one cpu and that core has a sibling
/// (`/sys/devices/system/cpu/cpu<c>/topology/thread_siblings_list`). `None`
/// for a floating caller or a core without SMT: the helper then floats too.
#[must_use]
pub fn caller_sibling() -> Option<usize> {
    // SAFETY: `set` is a zeroed cpu_set_t that `sched_getaffinity` fills with
    // the matching size, and `CPU_ISSET` only reads it.
    let set = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        set
    };
    let max = 8 * std::mem::size_of::<libc::cpu_set_t>();
    // SAFETY: `c < max`, the set's own bit count.
    let mut on = (0..max).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) });
    let (Some(cpu), None) = (on.next(), on.next()) else {
        return None;
    };
    let list = std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ))
    .ok()?;
    cpu_list(&list).find(|&c| c != cpu)
}

/// The cpus of a sysfs cpu list (`0,32`, `0-1`, `0-3,8`); a malformed entry
/// ends the list.
fn cpu_list(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.trim()
        .split(',')
        .map_while(|part| match part.split_once('-') {
            Some((a, b)) => Some(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
            None => part.parse::<usize>().ok().map(|c| c..=c),
        })
        .flatten()
}

/// Pin the calling thread to one logical cpu; false when the kernel refuses
/// (a container, a cpu outside the process's set).
fn pin_to(cpu: usize) -> bool {
    if cpu >= 8 * std::mem::size_of::<libc::cpu_set_t>() {
        return false;
    }
    // SAFETY: `set` is a zeroed cpu_set_t only written through `CPU_SET` with
    // an index inside it, and `sched_setaffinity` reads it with the matching
    // size.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}
