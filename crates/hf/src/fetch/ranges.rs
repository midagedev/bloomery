//! A file's download over ranges. `<file>.part` is filled by up to [`Split::connections`] curl
//! transfers at once, each asked for the file from its range's next byte (`--continue-at`, which
//! curl refuses to read from a hub that answers the whole file instead) and stopped at its range's
//! end. A worker whose range is full takes the second half of the range with the most bytes left,
//! so every connection stays busy to the file's end.
//!
//! A transfer that ends before its range's end is asked again from the byte it reached when the
//! network cut it, the hub answered busy or down, or the body stopped short ([`transient`]): after
//! the seconds the hub's `Retry-After` names (curl's `--dump-header`, at most
//! [`RETRY_AFTER_MAX`]), else a doubling backoff ([`backoff`]), until [`RETRIES`] attempts in a row
//! bring no byte. Any other failure stops every transfer of the file.
//!
//! The sidecar `.<sha256(part basename)>.ranges.lock.lock` holds the listing's digest and size, then each
//! range's first byte, the byte it reached and its end. It is written before the first byte lands, again every
//! few seconds, and when the fetch fails, so a later fetch resumes every range where it stopped.
//! A sidecar that names another digest or size, that does not tile the file, or that names bytes
//! past the part's end is not this part's: both are removed and the file is fetched again
//! (`stale`). A `.part` with no sidecar holds the file's first bytes, as a single transfer leaves
//! it.
//!
//! The digest is read behind the download: the calling thread hashes the bytes from the file's
//! start as far as every range before them is full, from the page cache the transfers just
//! wrote, so the file is not read again after its last byte lands. A resumed fetch hashes the
//! bytes it keeps as the transfers run.

use std::fmt::Write as _;
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread::{self, ScopedJoinHandle};
use std::time::{Duration, Instant};

use super::{
    Client, Event, Hasher, PROGRESS_EVERY, curl_why, http_status, hub_unavailable, network_failure,
    record_path, remove, remove_if_there, size_of, write_text,
};
use crate::{Entry, HfError};

/// Attempts in a row that bring no byte before a range's transfer is refused.
const RETRIES: u32 = 5;

/// The longest wait a hub's `Retry-After` is followed for.
const RETRY_AFTER_MAX: Duration = Duration::from_secs(300);

/// Seconds a transfer may bring no byte before curl ends it (exit 28), to be asked again.
const STALL_SECS: &str = "60";

/// The bytes one read of a transfer's output takes.
const READ_CHUNK: usize = 256 << 10;

/// The bytes one read of the digest takes.
const HASH_CHUNK: usize = 4 << 20;

/// How long a thread waits before it looks again: for bytes to hash, or for a failure that ends a
/// transfer's wait to be asked again.
const IDLE: Duration = Duration::from_millis(20);

/// How a file is cut into ranges.
#[derive(Clone, Copy, Debug)]
pub(super) struct Split {
    /// The fewest bytes a range is cut to: each range is a request of its own, which pays the
    /// hub's redirect and a new connection before its first byte.
    pub(super) min: u64,
    /// The most transfers one file runs at once.
    pub(super) connections: usize,
}

impl Split {
    /// The cut every fetch from the hub uses.
    pub(super) const HUB: Split = Split {
        min: 64 << 20,
        connections: 16,
    };
}

/// The sidecar of `part`, keyed by its basename and naming what each range holds. `.lock.lock` is
/// no listed file's, nor a listed file's lock: a listed name ending in `.lock` is refused.
fn sidecar(part: &Path) -> PathBuf {
    record_path(part, ".ranges.lock.lock")
}

/// Removes `part`'s sidecar, if it has one: the part is gone, or holds the file's first bytes.
pub(super) fn forget(part: &Path) -> Result<(), HfError> {
    remove_if_there(&sidecar(part))
}

/// Bytes `start..end` of the file, of which `start..pos` are in the part.
#[derive(Clone, Copy)]
struct Range {
    start: u64,
    pos: u64,
    end: u64,
}

/// What a part holds of a file: ranges that tile it.
pub(super) struct Ranges(Vec<Range>);

impl Ranges {
    /// What `part` holds of `entry`, and whether what was there was another file's: its
    /// sidecar's ranges when the sidecar is this part's, else the part's bytes as the file's first
    /// ones and the rest cut by `split`. A sidecar that is not this part's, or a part longer than
    /// the file, is removed with its part, and the whole file is cut.
    pub(super) fn open(
        entry: &Entry,
        part: &Path,
        split: Split,
    ) -> Result<(Ranges, bool), HfError> {
        let len = size_of(part)?;
        let sidecar = sidecar(part);
        let found = match fs::read_to_string(&sidecar) {
            Ok(text) => Ranges::parse(&text, entry, len.unwrap_or(0)),
            Err(e) if e.kind() == ErrorKind::NotFound => match len {
                Some(len) if len > entry.size => None,
                len => Some(Ranges::after(entry.size, len.unwrap_or(0), split)),
            },
            Err(e) => return Err(HfError::io(&sidecar, e)),
        };
        if let Some(ranges) = found {
            return Ok((ranges, false));
        }
        if len.is_some() {
            remove(part)?;
        }
        remove_if_there(&sidecar)?;
        Ok((Ranges::after(entry.size, 0, split), true))
    }

    /// The bytes the part holds.
    pub(super) fn held(&self) -> u64 {
        self.0.iter().map(|r| r.pos - r.start).sum()
    }

    /// A file of `size` bytes whose first `held` are in the part, the rest cut into ranges of at
    /// least `split.min` bytes, at most `split.connections` of them.
    fn after(size: u64, held: u64, split: Split) -> Ranges {
        let mut ranges = Vec::new();
        if held > 0 {
            ranges.push(Range {
                start: 0,
                pos: held,
                end: held,
            });
        }
        let left = size - held;
        let n = (left / split.min).clamp(1, split.connections as u64);
        let step = left / n;
        for i in 0..n {
            let start = held + i * step;
            let end = if i + 1 == n { size } else { start + step };
            if start < end {
                ranges.push(Range {
                    start,
                    pos: start,
                    end,
                });
            }
        }
        Ranges(ranges)
    }

    /// The sidecar's text for `entry`.
    fn text(&self, entry: &Entry) -> String {
        let mut text = format!("{} {}\n", entry.digest, entry.size);
        for r in &self.0 {
            writeln!(text, "{} {} {}", r.start, r.pos, r.end).expect("writing to a String");
        }
        text
    }

    /// The ranges `text` names when it is `entry`'s sidecar, its ranges tile the file and none
    /// holds bytes past the part's `len`.
    fn parse(text: &str, entry: &Entry, len: u64) -> Option<Ranges> {
        let mut lines = text.lines();
        let head: Vec<&str> = lines.next()?.split(' ').collect();
        let size = entry.size.to_string();
        if head != [entry.digest.as_str(), size.as_str()] || len > entry.size {
            return None;
        }
        let mut ranges = lines
            .map(|line| {
                let v: Vec<u64> = line
                    .split(' ')
                    .map(|w| w.parse().ok())
                    .collect::<Option<_>>()?;
                match v[..] {
                    [start, pos, end] => Some(Range { start, pos, end }),
                    _ => None,
                }
            })
            .collect::<Option<Vec<Range>>>()?;
        ranges.sort_by_key(|r| r.start);
        let mut at = 0;
        for r in &ranges {
            // A range that holds no byte yet claims nothing of the part, which ends at the last
            // byte any range wrote.
            let past_part = r.pos > r.start && r.pos > len;
            if r.start != at || r.pos < r.start || r.end < r.pos || past_part {
                return None;
            }
            at = r.end;
        }
        (at == entry.size).then_some(Ranges(ranges))
    }
}

/// A range as the transfers fill it.
struct Slot {
    range: Range,
    /// A worker fills it.
    taken: bool,
}

/// The ranges of one download, and the first failure, which stops every transfer.
struct Table {
    slots: Vec<Slot>,
    failed: Option<HfError>,
}

impl Table {
    /// The range for an idle worker: one no worker has with bytes left, else the second half of
    /// the taken range with the most bytes left when each half keeps `split.min`; `None` when
    /// there is neither or a transfer failed.
    fn take(&mut self, split: Split) -> Option<usize> {
        if self.failed.is_some() {
            return None;
        }
        if let Some(i) = self
            .slots
            .iter()
            .position(|s| !s.taken && s.range.pos < s.range.end)
        {
            self.slots[i].taken = true;
            return Some(i);
        }
        let widest = self
            .slots
            .iter_mut()
            .filter(|s| s.taken)
            .max_by_key(|s| s.range.end - s.range.pos)?;
        let left = widest.range.end - widest.range.pos;
        if left < 2 * split.min {
            return None;
        }
        let mid = widest.range.pos + left / 2;
        let end = std::mem::replace(&mut widest.range.end, mid);
        self.slots.push(Slot {
            range: Range {
                start: mid,
                pos: mid,
                end,
            },
            taken: true,
        });
        Some(self.slots.len() - 1)
    }

    /// The ranges in file order.
    fn ranges(&self) -> Ranges {
        let mut ranges: Vec<Range> = self.slots.iter().map(|s| s.range).collect();
        ranges.sort_by_key(|r| r.start);
        Ranges(ranges)
    }

    /// The bytes from the file's start that are in the part with every byte before them.
    fn ready(&self) -> u64 {
        let mut at = 0;
        for r in self.ranges().0 {
            at = r.pos;
            if r.pos < r.end {
                break;
            }
        }
        at
    }
}

/// What the transfers of one file share.
struct Job<'a> {
    url: String,
    what: String,
    part: &'a Path,
    /// The part as [`Job::follow`] reads it, from its first byte on.
    file: fs::File,
    size: u64,
    table: Mutex<Table>,
}

impl Job<'_> {
    fn table(&self) -> MutexGuard<'_, Table> {
        self.table.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records `e` as the download's failure unless one is recorded: every transfer stops.
    fn fail(&self, e: HfError) {
        self.table().failed.get_or_insert(e);
    }

    /// Waits `d`, or until a transfer fails; whether none did.
    fn pause(&self, d: Duration) -> bool {
        let until = Instant::now() + d;
        while self.table().failed.is_none() {
            let now = Instant::now();
            if now >= until {
                return true;
            }
            thread::sleep((until - now).min(IDLE));
        }
        false
    }

    /// The calling thread's part: hashes the bytes [`Table::ready`] gives, in order, and every
    /// [`PROGRESS_EVERY`] says the bytes held and saves the `sidecar`; until the transfers have
    /// ended and every byte is hashed, or a transfer failed. The bytes hashed.
    fn follow(
        &self,
        entry: &Entry,
        sidecar: Option<&Path>,
        transfers: &[ScopedJoinHandle<'_, ()>],
        hasher: &mut Hasher,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> u64 {
        let mut buf = vec![0u8; HASH_CHUNK];
        let (mut hashed, mut last) = (0u64, Instant::now());
        loop {
            let ended = transfers.iter().all(ScopedJoinHandle::is_finished);
            let ready = {
                let t = self.table();
                if t.failed.is_some() {
                    return hashed;
                }
                t.ready()
            };
            if ready > hashed {
                let n = usize::try_from(ready - hashed).map_or(HASH_CHUNK, |n| n.min(HASH_CHUNK));
                if let Err(e) = (&self.file).read_exact(&mut buf[..n]) {
                    self.fail(HfError::io(self.part, e));
                    return hashed;
                }
                hasher.update(&buf[..n]);
                hashed += n as u64;
            } else if ended {
                return hashed;
            } else {
                thread::sleep(IDLE);
            }
            if last.elapsed() >= PROGRESS_EVERY {
                last = Instant::now();
                let ranges = self.table().ranges();
                events(&Event::Progress {
                    file: &entry.path,
                    have: ranges.held(),
                    of: entry.size,
                });
                if let Some(sidecar) = sidecar
                    && let Err(e) = write_text(sidecar, &ranges.text(entry))
                {
                    self.fail(e);
                    return hashed;
                }
            }
        }
    }
}

/// How one transfer ended.
enum Attempt {
    /// Its range is full, or another transfer failed.
    Over,
    /// curl ended first: whether it wrote a byte, its status and its stderr.
    Ended {
        wrote: bool,
        status: ExitStatus,
        stderr: String,
    },
}

/// A curl child, killed and reaped when dropped before it ends on its own.
struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Client {
    /// The bytes of `entry` of `repo` that `ranges` leaves, fetched into `part` as the module
    /// says, its sidecar naming what each range holds; the digest of the whole file by its check.
    /// On a failure the part and its sidecar stay for the next fetch to resume.
    pub(super) fn download(
        &self,
        repo: &str,
        entry: &Entry,
        part: &Path,
        ranges: Ranges,
        split: Split,
        events: &mut dyn FnMut(&Event<'_>),
    ) -> Result<String, HfError> {
        if let Some(dir) = part.parent() {
            fs::create_dir_all(dir).map_err(|e| HfError::io(dir, e))?;
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(part)
            .map_err(|e| HfError::io(part, e))?;
        let left = entry.size - ranges.held();
        let open = ranges.0.iter().filter(|r| r.pos < r.end).count();
        let workers = if left == 0 {
            0
        } else {
            usize::try_from(left / split.min)
                .unwrap_or(usize::MAX)
                .max(open)
                .min(split.connections)
                .max(1)
        };
        // Before a byte lands past the part's first ones.
        let sidecar = (workers > 0).then(|| sidecar(part));
        let sidecar = sidecar.as_deref();
        if let Some(sidecar) = sidecar {
            write_text(sidecar, &ranges.text(entry))?;
        }
        let (repo, rev) = self.hub_at(repo);
        let url = format!("{}/{repo}/resolve/{rev}/{}", self.base, entry.path);
        let job = Job {
            what: format!("download {url}"),
            url,
            part,
            file,
            size: entry.size,
            table: Mutex::new(Table {
                slots: ranges
                    .0
                    .into_iter()
                    .map(|range| Slot {
                        range,
                        taken: false,
                    })
                    .collect(),
                failed: None,
            }),
        };
        let mut hasher = Hasher::new(entry.check, entry.size);
        let hashed = thread::scope(|s| {
            let transfers: Vec<_> = (0..workers)
                .map(|_| s.spawn(|| self.transfers(&job, split)))
                .collect();
            let hashed = job.follow(entry, sidecar, &transfers, &mut hasher, events);
            for t in transfers {
                t.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
            }
            hashed
        });
        let mut table = job
            .table
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = table.failed.take() {
            // The failure is the one to give: a save that fails leaves the last saved sidecar,
            // which a later fetch resumes from.
            if let Some(sidecar) = sidecar {
                let _ = write_text(sidecar, &table.ranges().text(entry));
            }
            return Err(e);
        }
        if hashed != entry.size {
            return Err(HfError::Size {
                file: part.display().to_string(),
                bytes: hashed,
                want: entry.size,
            });
        }
        Ok(hasher.finish())
    }

    /// One worker: fills the ranges [`Table::take`] gives it until there is none; its failure is
    /// the download's.
    fn transfers(&self, job: &Job<'_>, split: Split) {
        loop {
            let Some(i) = job.table().take(split) else {
                return;
            };
            if let Err(e) = self.fill(job, i) {
                job.fail(e);
                return;
            }
        }
    }

    /// Range `i` of `job` filled from the byte it reached, asked again as the module says.
    fn fill(&self, job: &Job<'_>, i: usize) -> Result<(), HfError> {
        let mut idle = 0;
        loop {
            let headers = self.headers_file("range");
            let attempt = self.transfer(job, i, &headers);
            let text = said(&headers);
            let _ = fs::remove_file(&headers);
            let (wrote, status, stderr) = match attempt? {
                Attempt::Over => return Ok(()),
                Attempt::Ended {
                    wrote,
                    status,
                    stderr,
                } => (wrote, status, stderr),
            };
            if status.success() {
                // The hub's file ended before the range did.
                return Err(HfError::Size {
                    file: job.part.display().to_string(),
                    bytes: job.table().slots[i].range.pos,
                    want: job.size,
                });
            }
            let why = curl_why(status.code(), &stderr);
            if !transient(status.code(), http_status(&stderr)) {
                return Err(HfError::Curl {
                    what: job.what.clone(),
                    why,
                });
            }
            let Some(wait) = retry(&mut idle, wrote, &text?) else {
                return Err(HfError::Curl {
                    what: job.what.clone(),
                    why: format!("{why}; {RETRIES} attempts in a row brought no byte"),
                });
            };
            if !job.pause(wait) {
                return Ok(());
            }
        }
    }

    /// One curl transfer into range `i` of `job` from the byte it reached to its end, the
    /// response headers dumped to `headers`.
    fn transfer(&self, job: &Job<'_>, i: usize, headers: &Path) -> Result<Attempt, HfError> {
        let mut pos = job.table().slots[i].range.pos;
        let mut cmd = self.command();
        cmd.args(["--speed-limit", "1", "--speed-time", STALL_SECS])
            .arg("--dump-header")
            .arg(headers);
        if pos > 0 {
            cmd.arg("--continue-at").arg(pos.to_string());
        }
        cmd.arg(&job.url).stdout(Stdio::piped());
        let io = |e| HfError::io(job.part, e);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(job.part)
            .map_err(io)?;
        file.seek(SeekFrom::Start(pos)).map_err(io)?;
        let mut child = Reaped(self.spawn(cmd, &job.what)?);
        let mut out = child.0.stdout.take().expect("curl's stdout is piped");
        let mut buf = vec![0u8; READ_CHUNK];
        let mut wrote = false;
        loop {
            let n = match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(HfError::Curl {
                        what: job.what.clone(),
                        why: format!("reading curl's output: {e}"),
                    });
                }
            };
            let end = {
                let t = job.table();
                if t.failed.is_some() {
                    return Ok(Attempt::Over);
                }
                t.slots[i].range.end
            };
            // A split may have moved the end below the bytes read: they are the file's bytes at
            // their offsets, so the new range's own transfer writes the same ones.
            let take = usize::try_from(end - pos).map_or(n, |left| left.min(n));
            file.write_all(&buf[..take]).map_err(io)?;
            pos += take as u64;
            wrote = true;
            let mut t = job.table();
            let range = &mut t.slots[i].range;
            range.pos = pos.min(range.end);
            if range.pos == range.end {
                return Ok(Attempt::Over);
            }
        }
        drop(out);
        let mut stderr = String::new();
        if let Some(mut s) = child.0.stderr.take() {
            let _ = s.read_to_string(&mut stderr);
        }
        let status = child.0.wait().map_err(|e| HfError::Curl {
            what: job.what.clone(),
            why: e.to_string(),
        })?;
        Ok(Attempt::Ended {
            wrote,
            status,
            stderr,
        })
    }
}

/// Whether a transfer curl ended with `code` (and the HTTP `status` it named) is asked again: the
/// network's failures ([`network_failure`], a stall's 28 among them), a busy or down hub
/// ([`hub_unavailable`]), a body cut short (18) or a failed HTTP/2 stream (16, 92).
fn transient(code: Option<i32>, status: Option<u16>) -> bool {
    network_failure(code) || hub_unavailable(code, status) || matches!(code, Some(16 | 18 | 92))
}

/// After a transfer that ended early: counts `idle`, the attempts in a row that brought no byte,
/// on by `wrote`, and gives the wait before the next attempt, the `Retry-After` of the hub's
/// `headers` else [`backoff`]; `None` once [`RETRIES`] attempts in a row brought no byte.
fn retry(idle: &mut u32, wrote: bool, headers: &str) -> Option<Duration> {
    *idle = if wrote { 0 } else { *idle + 1 };
    (*idle < RETRIES).then(|| retry_after(headers).unwrap_or_else(|| backoff(*idle)))
}

/// The text of curl's `--dump-header` file at `path`, read whatever bytes a header holds; empty
/// when curl wrote none.
fn said(path: &Path) -> Result<String, HfError> {
    match fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(HfError::io(path, e)),
    }
}

/// The seconds the last response of curl's `--dump-header` text asks for in its `Retry-After`,
/// at most [`RETRY_AFTER_MAX`]; `None` when it names none or names a date.
fn retry_after(headers: &str) -> Option<Duration> {
    headers
        .lines()
        .rev()
        .take_while(|l| !l.starts_with("HTTP/"))
        .filter_map(|l| l.split_once(':'))
        .find(|(name, _)| name.trim().eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, v)| v.trim().parse().ok())
        .map(|s| Duration::from_secs(s).min(RETRY_AFTER_MAX))
}

/// The wait when the hub named none: one second, doubled for each of the `idle` attempts in a row
/// that brought no byte.
fn backoff(idle: u32) -> Duration {
    Duration::from_secs(1 << idle.min(RETRIES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Check;
    use crate::fetch::tests::{content, fake_curl, hex, hub_answering, run_split, scratch, served};
    use crate::fetch::with_suffix;
    use sha2::Digest as _;

    /// Ranges of 16 KiB, four at once: [`content`] is cut into four.
    const SMALL: Split = Split {
        min: 16 << 10,
        connections: 4,
    };

    /// `curl` behind a script in `dir` that appends each call's `--continue-at` byte (0 without
    /// one) to `dir/calls`, then starts curl once `at_once` calls have begun; a call that waits
    /// five seconds for them exits 2, a failure no fetch asks again.
    fn logging_curl(dir: &Path, at_once: usize) -> std::ffi::OsString {
        fake_curl(
            dir,
            &format!(
                "#!/bin/sh\nd=$(dirname -- \"$0\")\nfrom=0\nprev=\nfor a in \"$@\"; do \
                 [ \"$prev\" = --continue-at ] && from=$a; prev=$a; done\necho $from >> \"$d/calls\"\n\
                 i=0\nwhile [ $(wc -l < \"$d/calls\") -lt {at_once} ]; do i=$((i+1)); \
                 [ $i -gt 50 ] && exit 2; sleep 0.1; done\nexec curl \"$@\"\n"
            ),
        )
        .into_os_string()
    }

    fn calls(dir: &Path) -> Vec<u64> {
        let mut v: Vec<u64> = fs::read_to_string(dir.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(|l| l.parse().unwrap())
            .collect();
        v.sort_unstable();
        v
    }

    /// An idle worker takes a range no worker has, then the back half of the taken range with the
    /// most bytes left while each half keeps `min`: the ranges still tile the file, and no byte
    /// left is dropped or asked for twice.
    #[test]
    fn an_idle_worker_takes_an_open_range_then_the_back_half_of_the_widest() {
        let split = Split {
            min: 10,
            connections: 16,
        };
        let slot = |start, pos, end, taken| Slot {
            range: Range { start, pos, end },
            taken,
        };
        let mut t = Table {
            slots: vec![
                slot(0, 5, 50, true),
                slot(50, 70, 100, true),
                slot(100, 100, 130, false),
            ],
            failed: None,
        };
        assert_eq!(t.take(split), Some(2));
        let half = t.take(split).expect("a half");
        let r = t.slots[half].range;
        assert_eq!((r.start, r.pos, r.end), (27, 27, 50));
        // Both halves of a split keep `min`, so the 105 bytes left make at most 10 ranges: a take
        // past that is a split that did not shrink the range it took from.
        let mut takes = 0;
        while t.take(split).is_some() {
            takes += 1;
            assert!(
                takes <= 105 / split.min,
                "more splits than the bytes left allow"
            );
        }
        let ranges = t.ranges().0;
        let tiles = ranges.windows(2).all(|w| w[0].end == w[1].start);
        assert!(tiles && ranges[0].start == 0 && ranges[ranges.len() - 1].end == 130);
        assert!(ranges.iter().all(|r| r.end - r.pos >= split.min));
        assert_eq!(
            ranges.iter().map(|r| r.end - r.pos).sum::<u64>(),
            45 + 30 + 30
        );
    }

    #[test]
    fn a_large_file_is_fetched_over_ranges_at_once() {
        let d = scratch("ranges-parallel");
        let body = content();
        let (mut c, e) = served(&d, "a/b", "m.gguf", &body);
        c.curl = logging_curl(&d, 4);
        let (r, seen) = run_split(&c, &e, SMALL);
        let p = r.expect("fetch");
        assert_eq!(fs::read(&p).unwrap(), body);
        assert_eq!(seen, ["file fetch 0", &format!("done {}", body.len())]);
        let starts = calls(&d);
        for at in [0, 75_000, 150_000, 225_000] {
            assert!(starts.contains(&at), "{starts:?}");
        }
        let part = with_suffix(&p, ".part");
        assert!(!part.exists() && !sidecar(&part).exists());
    }

    /// A killed fetch keeps the initial record even when later ranges reached the part's end
    /// before the first range wrote anything: holes must be fetched, not treated as held bytes.
    #[test]
    fn a_kill_before_the_first_save_resumes_the_initial_ranges() {
        let d = std::env::current_dir().unwrap();
        if d.join("curl").is_file() {
            let (mut c, e) = served(&d, "a/b", "m.gguf", &content());
            c.curl = d.join("curl").into_os_string();
            run_split(&c, &e, SMALL).0.expect("child fetch");
            return;
        }
        let d = scratch("ranges-initial-record");
        let body = content();
        let (mut c, e) = served(&d, "a/b", "m.gguf", &body);
        c.curl = logging_curl(&d, 4);
        let text = fs::read_to_string(d.join("curl")).unwrap().replace(
            r#"exec curl "$@""#,
            r#"if [ "$from" = 0 ]; then
i=0
while [ ! -f "$d/release" ]; do
i=$((i+1)); [ $i -gt 300 ] && exit 2; sleep 0.01
done
fi
exec curl "$@""#,
        );
        fake_curl(&d, &text);
        let dest = crate::cache_path(&c.root, "a/b", &e.path).unwrap();
        let part = with_suffix(&dest, ".part");
        let record = sidecar(&part);
        let mut child = Reaped(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "fetch::ranges::tests::a_kill_before_the_first_save_resumes_the_initial_ranges",
                ])
                .current_dir(&d)
                .spawn()
                .unwrap(),
        );
        let until = Instant::now() + Duration::from_secs(3);
        let mut out_of_order = false;
        while Instant::now() < until {
            if fs::read(&part).is_ok_and(|b| b.get(225_000..) == Some(&body[225_000..])) {
                out_of_order = true;
                break;
            }
            thread::sleep(IDLE);
        }
        child.0.kill().unwrap();
        child.0.wait().unwrap();
        fs::write(d.join("release"), "").unwrap();
        assert!(
            out_of_order,
            "later ranges did not write before the first save"
        );
        let initial = fs::read_to_string(&record).expect("initial range record survives kill");
        let rows: Vec<Vec<u64>> = initial
            .lines()
            .skip(1)
            .map(|line| {
                line.split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(rows.len(), 4, "{initial}");
        assert!(
            rows.iter().all(|r| r[0] == r[1]),
            "progress saved before kill: {initial}"
        );
        fs::write(d.join("calls"), "").unwrap();
        let (got, seen) = run_split(&c, &e, SMALL);
        assert_eq!(fs::read(got.expect("restart fetch")).unwrap(), body);
        assert_eq!(seen, ["file fetch 0", "done 300000"]);
        let starts = calls(&d);
        for at in [0, 75_000, 150_000, 225_000] {
            assert!(
                starts.contains(&at),
                "initial range not requested: {starts:?}"
            );
        }
        assert!(!part.exists() && !record.exists());
    }

    /// Both worker boundaries observe a hard failure: no new range is claimed, and a transfer
    /// already reading a peer's bytes stops before it writes them.
    #[test]
    fn a_hard_failure_stops_claiming_and_writing_other_ranges() {
        static BODY: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(content);
        let d = scratch("ranges-hard-failure");
        let c = hub_answering(&d, |_, request| {
            if !request.to_ascii_lowercase().contains("range: bytes=75000-") {
                return b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec();
            }
            let mut answer = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 75000-299999/300000\r\nContent-Length: 225000\r\n\r\n".to_vec();
            answer.extend_from_slice(&BODY[75_000..]);
            answer
        });
        let part = d.join("m.part");
        let file = fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&part)
            .unwrap();
        let job = Job {
            url: format!("{}/m", c.base),
            what: "hard-failure test".into(),
            part: &part,
            file,
            size: 300_000,
            table: Mutex::new(Table {
                slots: (0..4)
                    .map(|i| Slot {
                        range: Range {
                            start: i * 75_000,
                            pos: i * 75_000,
                            end: (i + 1) * 75_000,
                        },
                        taken: false,
                    })
                    .collect(),
                failed: None,
            }),
        };
        c.transfers(&job, SMALL);
        assert!(
            job.table()
                .failed
                .as_ref()
                .unwrap()
                .to_string()
                .contains("404")
        );
        assert!(!job.table().slots[1].taken, "failure claimed a peer range");
        assert_eq!(
            job.table().take(SMALL),
            None,
            "a failed file claimed another range"
        );
        assert!(matches!(
            c.transfer(&job, 1, &d.join("headers")).unwrap(),
            Attempt::Over
        ));
        assert_eq!(
            job.table().slots[1].range.pos,
            75_000,
            "peer wrote after another range failed"
        );
        assert_eq!(fs::metadata(&part).unwrap().len(), 0);
    }

    #[test]
    fn each_range_resumes_where_it_stopped_and_another_files_ranges_are_stale() {
        let d = scratch("ranges-resume");
        let body = content();
        let (mut c, e) = served(&d, "a/b", "m.gguf", &body);
        c.curl = logging_curl(&d, 2);
        let dest = crate::cache_path(&c.root, "a/b", &e.path).unwrap();
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let part = with_suffix(&dest, ".part");
        let sidecar = sidecar(&part);
        // Bytes no range holds are not the file's: the fetch writes over them.
        let mut held = vec![0xff; body.len()];
        let ranges = [
            (0, 40_000, 100_000),
            (100_000, 200_000, 200_000),
            (200_000, 250_000, 300_000),
        ];
        let mut text = format!("{} {}\n", e.digest, e.size);
        for (start, pos, end) in ranges {
            held[start..pos].copy_from_slice(&body[start..pos]);
            writeln!(text, "{start} {pos} {end}").unwrap();
        }
        fs::write(&part, &held[..250_000]).unwrap();
        fs::write(&sidecar, &text).unwrap();
        let (r, seen) = run_split(&c, &e, SMALL);
        assert_eq!(fs::read(r.expect("resume")).unwrap(), body);
        assert_eq!(seen, ["file resume 190000", "done 110000"]);
        let starts = calls(&d);
        assert!(
            starts.contains(&40_000) && starts.contains(&250_000),
            "{starts:?}"
        );
        assert!(
            !starts.contains(&0) && !starts.contains(&100_000),
            "{starts:?}"
        );
        assert!(!part.exists() && !sidecar.exists());

        // Ranges that hold no byte yet claim nothing past the part's end: the fetch resumes the
        // one that holds bytes.
        let untouched = format!(
            "{} {}\n0 40000 100000\n100000 100000 200000\n200000 200000 300000\n",
            e.digest, e.size
        );
        fs::remove_file(&dest).unwrap();
        fs::remove_file(with_suffix(&dest, ".verified")).unwrap();
        fs::write(&part, &body[..40_000]).unwrap();
        fs::write(&sidecar, &untouched).unwrap();
        fs::write(d.join("calls"), "").unwrap();
        let (r, seen) = run_split(&c, &e, SMALL);
        assert_eq!(fs::read(r.expect("resume")).unwrap(), body);
        assert_eq!(seen, ["file resume 40000", "done 260000"]);
        assert!(!calls(&d).contains(&0), "{:?}", calls(&d));

        for stale in [
            text.replace(&e.digest, &"0".repeat(64)),
            text.replace("250000 300000", "250000 299999"),
            text.replace("40000 100000", "260000 100000"),
        ] {
            fs::remove_file(&dest).unwrap();
            fs::remove_file(with_suffix(&dest, ".verified")).unwrap();
            fs::write(&part, &held[..250_000]).unwrap();
            fs::write(&sidecar, &stale).unwrap();
            let (r, seen) = run_split(&c, &e, SMALL);
            assert_eq!(fs::read(r.expect("refetch")).unwrap(), body, "{stale}");
            assert_eq!(seen[0], "file stale 0", "{stale}");
        }
    }

    /// A hub on a local port serving `body` by range, which answers its first request busy with
    /// `Retry-After: 2` and cuts its second body halfway; each request's first byte.
    fn flaky(body: &'static [u8], root: &Path) -> (Client, std::sync::Arc<Mutex<Vec<u64>>>) {
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let c = hub_answering(root, move |n, request| {
            let from: usize = request
                .to_ascii_lowercase()
                .split_once("range: bytes=")
                .and_then(|(_, r)| r.split('-').next()?.parse().ok())
                .unwrap_or(0);
            log.lock().unwrap().push(from as u64);
            let rest = &body[from..];
            let head = match n {
                0 => "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 0"
                    .to_owned(),
                _ if from == 0 => format!("HTTP/1.1 200 OK\r\nContent-Length: {}", rest.len()),
                _ => format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\nContent-Length: {}",
                    body.len() - 1,
                    body.len(),
                    rest.len()
                ),
            };
            let mut answer = format!("{head}\r\nConnection: close\r\n\r\n").into_bytes();
            answer.extend_from_slice(&rest[..if n == 1 { rest.len() / 2 } else { rest.len() }]);
            answer
        });
        (c, seen)
    }

    #[test]
    fn a_busy_or_cut_range_is_asked_again_from_its_byte_after_retry_after() {
        assert_eq!(
            retry_after(
                "HTTP/1.1 302 Found\r\nRetry-After: 9\r\n\r\nHTTP/2 429\r\nretry-after: 7\r\n\r\n"
            ),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            retry_after("HTTP/2 302\r\nRetry-After: 9\r\n\r\nHTTP/2 503\r\n\r\n"),
            None
        );
        assert_eq!(
            retry_after("HTTP/2 429\r\nRetry-After: Wed, 21 Oct 2026 07:28:00 GMT\r\n"),
            None
        );
        assert_eq!(
            retry_after("HTTP/2 429\r\nRetry-After: 86400\r\n"),
            Some(RETRY_AFTER_MAX)
        );
        // A cut that brought bytes waits a second; each attempt in a row with none doubles it,
        // and the fifth is the last.
        let mut idle = 0;
        let waits: Vec<_> = (0..RETRIES).map(|_| retry(&mut idle, false, "")).collect();
        let doubling = [2, 4, 8, 16].map(|s| Some(Duration::from_secs(s)));
        assert_eq!(waits[..4], doubling);
        assert_eq!(waits[4], None);
        assert_eq!(retry(&mut idle, true, ""), Some(Duration::from_secs(1)));
        assert_eq!(idle, 0);
        let mut idle = 0;
        let busy = "HTTP/2 503\r\nRetry-After: 7\r\n\r\n";
        assert_eq!(retry(&mut idle, false, busy), Some(Duration::from_secs(7)));
        // A header that is not UTF-8 keeps the hub's wait; a file curl never wrote is no header.
        let d = scratch("ranges-headers");
        let headers = d.join("h");
        fs::write(
            &headers,
            b"HTTP/2 503\r\nX-Bytes: \xff\r\nRetry-After: 7\r\n\r\n",
        )
        .unwrap();
        assert_eq!(
            retry_after(&said(&headers).unwrap()),
            Some(Duration::from_secs(7))
        );
        assert_eq!(said(&d.join("none")).unwrap(), "");

        static BODY: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(content);
        let d = scratch("ranges-retry");
        let (c, starts) = flaky(&BODY, &d.join("cache"));
        let e = Entry {
            path: "m.gguf".to_owned(),
            size: BODY.len() as u64,
            check: Check::Sha256,
            digest: hex(&sha2::Sha256::digest(&*BODY)),
        };
        let t = Instant::now();
        let (r, seen) = run_split(&c, &e, Split::HUB);
        assert_eq!(fs::read(r.expect("fetch")).unwrap(), *BODY);
        // Two seconds the hub asked for, then one of backoff after the cut body.
        assert!(t.elapsed() >= Duration::from_secs(3), "{:?}", t.elapsed());
        let half = (BODY.len() / 2) as u64;
        assert_eq!(*starts.lock().unwrap(), [0, 0, half]);
        assert_eq!(seen, ["file fetch 0", &format!("done {}", BODY.len())]);
    }
}
