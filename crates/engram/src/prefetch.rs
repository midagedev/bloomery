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

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::{Engram, EngramError, faults_thread};

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
        let handle = std::thread::Builder::new()
            .name("engram-prefetch".into())
            .spawn(move || helper(&engram, mode, &job_rx, &done_tx))?;

        Ok(Prefetcher {
            jobs: Some(job_tx),
            done: done_rx,
            spare,
            current: None,
            in_flight: None,
            max_rows: max_rows_per_site.to_vec(),
            row_bytes,
            last: HelperTimes::default(),
            handle: Some(handle),
        })
    }

    /// Hand the helper one token's row ids and let it run.
    ///
    /// Returns as soon as the ids are copied into a free buffer — this is the
    /// step thread's whole share of the read. Each site may ask for any number
    /// of rows up to the prefetcher's size for it, and a token that asks for
    /// none is not sent at all. Two submits in a row without a
    /// [`Prefetcher::wait`] between them is a caller bug, not a runtime
    /// condition.
    pub fn submit(&mut self, token_ids: &[Vec<u32>]) -> Result<(), EngramError> {
        if self.in_flight.is_some() {
            return Err(EngramError::Prefetch(
                "a token is already in flight; wait() first",
            ));
        }
        if token_ids.len() != self.max_rows.len() {
            return Err(EngramError::Prefetch("token has a different site count"));
        }
        // Checked before any copy: past its capacity an id vector would
        // reallocate, and the buffer would be too short for the rows.
        if token_ids
            .iter()
            .zip(&self.max_rows)
            .any(|(ids, &max)| ids.len() > max)
        {
            return Err(EngramError::Prefetch(
                "token asks more rows of a site than the prefetcher was sized for",
            ));
        }
        if token_ids.iter().all(Vec::is_empty) {
            self.in_flight = Some(InFlight::Empty);
            return Ok(());
        }
        let Some(mut job) = self.spare.pop() else {
            return Err(EngramError::Prefetch("no free buffer; wait() first"));
        };
        job.len = 0;
        for ((dst, src), rb) in job.ids.iter_mut().zip(token_ids).zip(&self.row_bytes) {
            dst.clear();
            dst.extend_from_slice(src);
            job.len += src.len() * rb;
        }
        let Some(tx) = self.jobs.as_ref() else {
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
fn helper(engram: &Engram, mode: FillMode, jobs: &Receiver<Job>, done: &Sender<Filled>) {
    let base = faults_thread();
    let sites = engram.sites();
    while let Ok(mut job) = jobs.recv() {
        let mut err = None;

        // WILLNEED over every row: one batch at the drive's queue depth.
        let t0 = Instant::now();
        for (site, ids) in sites.iter().zip(&job.ids) {
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
        };
        if done.send(Filled { job, err, times }).is_err() {
            break;
        }
    }
}
