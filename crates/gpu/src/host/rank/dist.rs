//! The head's distribution of a load's read ranges: rank 0 holds the source
//! files and every other rank receives only the byte ranges it reads, each at
//! its original offset in a sparse file of the original's length, with no
//! full copy on any node.
//!
//! A rank keeps what it received in a store folder ([`Store`]), keyed by the
//! file's digest, the SHA-256 of the whole original, not by the rank: a
//! file's sparse copy is `<digest>.sparse` and its coverage `<digest>.coverage`,
//! one line a range, `<offset> <len> <sha256>`. A range enters the coverage
//! only once its bytes hash to what rank 0 hashed as it sent them and they
//! are written back to the disk, so a later run's rank 0 sends no range a
//! rank's coverage holds. A sparse copy never takes a source file's name: a
//! partial file is not a checkpoint file, and only a reader that asks the
//! coverage reads it ([`Fetched`]).
//!
//! Rank 0 merges a rank's ranges of one file whose gaps are at most
//! [`GAP`] bytes into one: a cut along a tensor's input axis reads a short
//! piece of every row, and sending, writing and hashing each piece alone is
//! slower than sending the whole tensor. A rank so holds more than it reads.
//! Rank 0 then cuts every rank's ranges at each range's ends into pieces of
//! at most [`PIECE`] bytes, so each piece is read whole by every rank that
//! reads any of it; it reads and hashes a piece once, on [`READERS`]
//! threads, and sends it to every rank that reads it and does not hold it.
//! A rank's coverage lists the pieces it received.
//!
//! Rank 0 sends a file only once its bytes hash to the digest that keys it
//! ([`verify`], once per file state), and every name is a path inside the
//! folder ([`inside`]).
//!
//! The exchange runs on the control addresses before the group opens
//! ([`super::RankSession::open`]): every other rank dials rank 0, names the
//! port its relay listens on, opens its data streams, then sends the ranges
//! its store holds, verified and merged where they touch, in frames of at
//! most [`HAVE_FRAME`] ranges, and its stream count; rank 0 sends each rank
//! the file table and the bytes it will send and keep, sends the pieces on
//! the streams, and each rank answers once every piece is written. Rank 0
//! ends the exchange once every rank answered, so a rank relays until then.
//!
//! A data stream is a TCP connection to rank 0's data listener: [`STREAMS`]
//! at each of rank 0's fabric addresses a connection reaches. A rank that
//! reaches none (on a ring, a rank two links away, whose relayed fabric
//! address carries RDMA alone) opens them through every other rank it
//! reaches, which connects on to rank 0 and passes rank 0's bytes on; a rank
//! with neither opens them at rank 0's control address.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use engram::direct::DIRECT_ALIGN;
use gguf::digest::{Sha256, hex, unhex};

use super::{Control, RankList, array_at, read_frame, u16_at, write_frame};
use crate::GpuError;
use crate::host::node::wire;

const WHAT: &str = "the head's distribution";

/// The bytes a stream moves in one read and write.
const CHUNK: usize = 8 << 20;

/// The largest gap between two ranges of one file that rank 0 sends as one.
const GAP: u64 = 1 << 20;

/// The most bytes of one piece: rank 0 reads, hashes and sends a piece as one, and a rank's
/// coverage records it as one.
const PIECE: u64 = 32 << 20;

/// Rank 0's threads that read and hash pieces.
const READERS: usize = 8;

/// The pieces read for a rank and not yet sent, per rank.
const QUEUE: usize = 8;

/// A rank's data streams at each address it reaches rank 0 at.
const STREAMS: usize = 2;

/// How long a connection to one address may take before the address counts as not reached.
const REACH: Duration = Duration::from_secs(2);

/// The threads that check a store's ranges against their record.
const CHECKERS: usize = 8;

/// The received bytes a rank holds before it syncs them and records their
/// ranges in the coverage, apart from the streams ([`record`]).
const RECORD: u64 = 8 << 30;

/// A data stream's first byte, before the rank it names, and a relay request's.
const HELLO: u8 = b'S';
const RELAY: u8 = b'R';
/// A relay's answer once its connection to rank 0 stands.
const ACK: u8 = b'K';
/// The file index of the header that ends a data stream.
const END: u32 = u32::MAX;

/// One source file: its name in the checkpoint, the SHA-256 of its bytes and
/// its length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceFile {
    pub name: String,
    pub digest: [u8; 32],
    pub size: u64,
}

/// A byte range a rank reads of source file `file` (its index in the file
/// table), at its original offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadRange {
    pub file: usize,
    pub offset: u64,
    pub len: u64,
}

/// A range a store holds, written and hashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Covered {
    offset: u64,
    len: u64,
    hash: [u8; 32],
}

/// The ranges a store holds of one file, merged and in order.
#[derive(Clone, Debug, Default)]
pub struct Covering(Vec<(u64, u64)>);

impl Covering {
    fn of(ranges: impl IntoIterator<Item = (u64, u64)>) -> Covering {
        let ranges = ranges
            .into_iter()
            .map(|(offset, len)| ReadRange {
                file: 0,
                offset,
                len,
            })
            .collect();
        Covering(
            merged(ranges, 0)
                .into_iter()
                .map(|r| (r.offset, r.offset.saturating_add(r.len)))
                .collect(),
        )
    }

    /// Whether the `len` bytes at `offset` lie in one held range.
    #[must_use]
    pub fn holds(&self, offset: u64, len: u64) -> bool {
        let Some(end) = offset.checked_add(len) else {
            return false;
        };
        let i = self.0.partition_point(|&(start, _)| start <= offset);
        i > 0 && self.0[i - 1].1 >= end
    }
}

/// Whether `name` is a path inside a folder: relative, with no `..`, root or
/// empty part. A source file's name is one ([`serve`], [`fetch`]), so a rank
/// writes and links nothing outside its store and its view.
#[must_use]
pub fn inside(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// Check every file of `files` in folder `dir` against its digest, once: a
/// file whose digest, length, modification and change times and inode the record at
/// `record` lists is taken as checked; every other file is hashed, on
/// [`CHECKERS`] threads, and one whose bytes do not hash to its digest is
/// refused by name. The record then lists every file of `files`. A name not
/// [`inside`] is refused. Load-time only.
pub fn verify(dir: &Path, files: &[SourceFile], record: &Path) -> Result<(), GpuError> {
    use std::os::unix::fs::MetadataExt;
    let shape = |e: String| GpuError::shape(WHAT, e);
    let stamps = files
        .iter()
        .map(|f| {
            if !inside(&f.name) {
                return Err(shape(format!(
                    "{:?} is not a name inside the folder",
                    f.name
                )));
            }
            let path = dir.join(&f.name);
            let m =
                std::fs::metadata(&path).map_err(|e| shape(format!("{}: {e}", path.display())))?;
            Ok(format!(
                "{}\t{}\t{}.{:09}\t{}:{}\t{}.{:09}\t{}",
                hex(&f.digest),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.dev(),
                m.ino(),
                m.ctime(),
                m.ctime_nsec(),
                f.name
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let known = std::fs::read_to_string(record).unwrap_or_default();
    let known: std::collections::HashSet<&str> = known.lines().collect();
    let todo: Vec<&SourceFile> = files
        .iter()
        .zip(&stamps)
        .filter(|(_, st)| !known.contains(st.as_str()))
        .map(|(f, _)| f)
        .collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let checks: Vec<_> = (0..CHECKERS.min(todo.len()))
            .map(|_| {
                scope.spawn(|| -> Result<(), GpuError> {
                    while let Some(f) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let path = dir.join(&f.name);
                        let got = gguf::digest::sha256_file(&path)
                            .map_err(|e| shape(format!("{}: {e}", path.display())))?;
                        if got != hex(&f.digest) {
                            return Err(shape(format!(
                                "{}: its bytes hash to {got}, not to the digest the folder's \
                                 manifest names",
                                f.name
                            )));
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        checks.into_iter().try_for_each(|j| {
            j.join()
                .map_err(|_| shape("a checking thread panicked".into()))?
        })
    })?;
    if !todo.is_empty() {
        let at = |e: std::io::Error| shape(format!("{}: {e}", record.display()));
        if let Some(parent) = record.parent() {
            std::fs::create_dir_all(parent).map_err(at)?;
        }
        let new = record.with_extension("new");
        std::fs::write(&new, format!("{}\n", stamps.join("\n"))).map_err(at)?;
        std::fs::rename(&new, record).map_err(at)?;
    }
    Ok(())
}

/// The reads `reads` (each a file under `root` and a byte range of it) as
/// ranges of `files`, each file named by its path relative to `root`, the
/// runs of one file whose gap is at most [`GAP`] merged as rank 0 sends
/// them. A read of a file `files` does not name is refused by name.
pub fn read_ranges(
    files: &[SourceFile],
    root: &Path,
    reads: &[(PathBuf, std::ops::Range<u64>)],
) -> Result<Vec<ReadRange>, GpuError> {
    let by_name: std::collections::HashMap<&str, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.as_str(), i))
        .collect();
    let mut out: Vec<ReadRange> = Vec::new();
    let mut at: Option<(&Path, usize)> = None;
    for (path, run) in reads {
        let file = match at {
            Some((p, f)) if p == path.as_path() => f,
            _ => path
                .strip_prefix(root)
                .ok()
                .and_then(|p| p.to_str())
                .and_then(|n| by_name.get(n).copied())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!(
                            "{} is a read file the folder's manifest does not name",
                            path.display()
                        ),
                    )
                })?,
        };
        at = Some((path.as_path(), file));
        out.push(ReadRange {
            file,
            offset: run.start,
            len: run.end - run.start,
        });
    }
    Ok(merged(out, GAP))
}

/// `ranges` sorted by file and offset, those of one file whose gap is at
/// most `gap` bytes merged into one.
fn merged(mut v: Vec<ReadRange>, gap: u64) -> Vec<ReadRange> {
    v.sort_unstable_by_key(|r| (r.file, r.offset));
    let mut out: Vec<ReadRange> = Vec::with_capacity(v.len().min(1 << 16));
    for r in v {
        match out.last_mut() {
            Some(last)
                if last.file == r.file
                    && r.offset <= last.offset.saturating_add(last.len).saturating_add(gap) =>
            {
                let end = last
                    .offset
                    .saturating_add(last.len)
                    .max(r.offset.saturating_add(r.len));
                last.len = end - last.offset;
            }
            _ => out.push(r),
        }
    }
    out
}

/// The ranges of `per_rank` (each list sorted by file and offset, with no
/// two ranges touching, as [`merged`] leaves them) cut at every range's ends
/// and into at most `most` bytes, in file and offset order, each with the
/// indices in `per_rank` of the lists that hold it.
fn pieces(per_rank: &[Vec<ReadRange>], most: u64) -> Vec<(ReadRange, Vec<usize>)> {
    let mut ends: Vec<(usize, u64)> = per_rank
        .iter()
        .flatten()
        .flat_map(|r| [(r.file, r.offset), (r.file, r.offset + r.len)])
        .collect();
    ends.sort_unstable();
    ends.dedup();
    let mut next = vec![0usize; per_rank.len()];
    let mut out = Vec::new();
    for w in ends.windows(2) {
        let ((file, start), (to_file, end)) = (w[0], w[1]);
        if to_file != file {
            continue;
        }
        let ranks: Vec<usize> = per_rank
            .iter()
            .zip(&mut next)
            .enumerate()
            .filter_map(|(i, (list, at))| {
                while list
                    .get(*at)
                    .is_some_and(|r| (r.file, r.offset + r.len) <= (file, start))
                {
                    *at += 1;
                }
                let r = list.get(*at)?;
                (r.file == file && r.offset <= start).then_some(i)
            })
            .collect();
        if ranks.is_empty() {
            continue;
        }
        let mut offset = start;
        while offset < end {
            let len = (end - offset).min(most);
            out.push((ReadRange { file, offset, len }, ranks.clone()));
            offset += len;
        }
    }
    out
}

/// A rank's store folder of received ranges (module doc).
struct Store {
    dir: PathBuf,
}

impl Store {
    /// The store at `dir`, made when absent, by its absolute path, so a link
    /// to a sparse copy holds from any folder.
    fn open(dir: &Path) -> Result<Store, GpuError> {
        let dir = std::fs::create_dir_all(dir)
            .and_then(|()| std::fs::canonicalize(dir))
            .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", dir.display())))?;
        Ok(Store { dir })
    }

    /// The sparse copy of the file of `digest`.
    fn path(&self, digest: &[u8; 32]) -> PathBuf {
        self.dir.join(format!("{}.sparse", hex(digest)))
    }

    fn coverage_path(&self, digest: &[u8; 32]) -> PathBuf {
        self.dir.join(format!("{}.coverage", hex(digest)))
    }

    /// The ranges the store holds of the file of `digest`; none without a
    /// coverage. A line of another shape is refused by name.
    fn coverage(&self, digest: &[u8; 32]) -> Result<Vec<Covered>, GpuError> {
        let path = self.coverage_path(digest);
        let Some(f) = open_if(&path)? else {
            return Ok(Vec::new());
        };
        BufReader::new(f)
            .lines()
            .map(|line| {
                let bad = || {
                    GpuError::shape(WHAT, format!("{}: a line of another shape", path.display()))
                };
                let line = line.map_err(|_| bad())?;
                let w: Vec<&str> = line.split_whitespace().collect();
                let [offset, len, hash] = w[..] else {
                    return Err(bad());
                };
                let h = unhex(hash).ok_or_else(bad)?;
                Ok(Covered {
                    offset: offset.parse().map_err(|_| bad())?,
                    len: len.parse().map_err(|_| bad())?,
                    hash: h,
                })
            })
            .collect()
    }

    /// Whether the bytes of `c` in the sparse copy `data` still hash to what
    /// the coverage records; a copy too short for them does not.
    fn intact(data: &File, c: &Covered, buf: &mut [u8]) -> std::io::Result<bool> {
        let mut h = Sha256::new();
        let mut at = 0u64;
        while at < c.len {
            let n = (c.len - at).min(buf.len() as u64) as usize;
            match data.read_exact_at(&mut buf[..n], c.offset + at) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
                Err(e) => return Err(e),
            }
            h.update(&buf[..n]);
            at += n as u64;
        }
        Ok(h.finish() == c.hash)
    }

    /// Every range the store holds, by file digest, merged where they touch:
    /// those of every coverage whose bytes in the sparse copy still hash to
    /// their record, checked on [`CHECKERS`] threads. A range whose bytes do
    /// not is left out, so rank 0 sends it again.
    fn held(&self) -> Result<Vec<Held>, GpuError> {
        let mut ranges: Vec<([u8; 32], Covered)> = Vec::new();
        let listed = std::fs::read_dir(&self.dir)
            .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", self.dir.display())))?;
        for entry in listed {
            let entry =
                entry.map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", self.dir.display())))?;
            let name = entry.file_name();
            let Some(d) = name
                .to_str()
                .and_then(|n| n.strip_suffix(".coverage"))
                .and_then(unhex)
            else {
                continue;
            };
            ranges.extend(self.coverage(&d)?.into_iter().map(|c| (d, c)));
        }
        ranges.sort_unstable_by_key(|(d, c)| (*d, c.offset));
        let next = AtomicUsize::new(0);
        let kept: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let checks: Vec<_> = (0..CHECKERS)
                .map(|_| {
                    scope.spawn(|| -> Result<Vec<usize>, GpuError> {
                        let mut buf = vec![0u8; CHUNK];
                        let mut kept = Vec::new();
                        let mut open: Option<([u8; 32], Option<File>)> = None;
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((d, c)) = ranges.get(i) else {
                                return Ok(kept);
                            };
                            if open.as_ref().is_none_or(|(at, _)| at != d) {
                                open = Some((*d, open_if(&self.path(d))?));
                            }
                            let Some((_, Some(data))) = &open else {
                                continue;
                            };
                            let fine = Store::intact(data, c, &mut buf).map_err(|e| {
                                GpuError::shape(WHAT, format!("{}: {e}", self.path(d).display()))
                            })?;
                            if fine {
                                kept.push(i);
                            }
                        }
                    })
                })
                .collect();
            checks
                .into_iter()
                .map(|j| {
                    j.join()
                        .map_err(|_| GpuError::shape(WHAT, "a checking thread panicked"))?
                })
                .collect::<Result<_, _>>()
        })?;
        let mut kept: Vec<usize> = kept.into_iter().flatten().collect();
        kept.sort_unstable();
        let mut out = Vec::new();
        for group in kept.chunk_by(|&a, &b| ranges[a].0 == ranges[b].0) {
            let d = ranges[group[0]].0;
            let merged = Covering::of(group.iter().map(|&i| (ranges[i].1.offset, ranges[i].1.len)));
            out.extend(merged.0.into_iter().map(|(start, end)| (d, start, end)));
        }
        Ok(out)
    }

    /// The sparse copy of `file`, opened for writing at the original's length.
    fn sparse(&self, file: &SourceFile) -> Result<File, String> {
        let path = self.path(&file.digest);
        let out = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if out.metadata().map_err(|e| e.to_string())?.len() < file.size {
            out.set_len(file.size).map_err(|e| e.to_string())?;
        }
        Ok(out)
    }

    /// Sync the sparse copies in `open` and only then append the ranges in
    /// `pending` to their files' coverage, which is synced too.
    fn record(
        &self,
        files: &[SourceFile],
        open: &[&File],
        mut pending: Vec<(usize, Covered)>,
    ) -> Result<(), String> {
        for f in open {
            f.sync_data().map_err(|e| e.to_string())?;
        }
        pending.sort_unstable_by_key(|&(file, c)| (file, c.offset));
        for group in pending.chunk_by(|a, b| a.0 == b.0) {
            let cov = self.coverage_path(&files[group[0].0].digest);
            let mut text = String::new();
            for (_, c) in group {
                text.push_str(&format!("{} {} {}\n", c.offset, c.len, hex(&c.hash)));
            }
            let mut out = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&cov)
                .map_err(|e| format!("{}: {e}", cov.display()))?;
            out.write_all(text.as_bytes())
                .and_then(|()| out.sync_data())
                .map_err(|e| format!("{}: {e}", cov.display()))?;
        }
        Ok(())
    }
}

/// The file at `path` opened for reading, or `None` when there is none.
fn open_if(path: &Path) -> Result<Option<File>, GpuError> {
    match File::open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(GpuError::shape(WHAT, format!("{}: {e}", path.display()))),
    }
}

/// Read `bytes.len()` bytes of `file` at `offset` from `from` into `bytes`,
/// and return their hash once it is rank 0's, which follows them.
fn receive(
    file: &SourceFile,
    offset: u64,
    from: &mut impl Read,
    bytes: &mut [u8],
) -> Result<[u8; 32], String> {
    let mut h = Sha256::new();
    for part in bytes.chunks_mut(CHUNK) {
        from.read_exact(part)
            .map_err(|e| format!("{}'s bytes: {e}", file.name))?;
        h.update(part);
    }
    let mut want = [0u8; 32];
    from.read_exact(&mut want)
        .map_err(|e| format!("{}'s hash: {e}", file.name))?;
    if h.finish() != want {
        return Err(format!(
            "{} bytes {offset}..{}: the bytes do not hash to rank 0's",
            file.name,
            offset + bytes.len() as u64
        ));
    }
    Ok(want)
}

/// What one rank received in a distribution: each source file, its sparse
/// copy and what the store holds of it, and the bytes sent and kept.
#[derive(Debug)]
pub struct Fetched {
    pub files: Vec<(SourceFile, PathBuf, Covering)>,
    pub sent: u64,
    pub kept: u64,
}

/// What rank 0 sent one rank: its bytes sent and kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Served {
    pub rank: usize,
    pub sent: u64,
    pub kept: u64,
}

/// A data stream's first bytes: its kind and the rank it serves.
fn hello(kind: u8, rank: usize) -> [u8; 5] {
    let mut b = [kind; 5];
    b[1..].copy_from_slice(&(rank as u32).to_le_bytes());
    b
}

/// The addresses of `at` a TCP connection reaches within [`REACH`], each
/// with its connection, all tried at once.
fn reach(at: &[SocketAddr]) -> Vec<(SocketAddr, TcpStream)> {
    std::thread::scope(|scope| {
        let tries: Vec<_> = at
            .iter()
            .map(|a| scope.spawn(move || TcpStream::connect_timeout(a, REACH).map(|s| (*a, s))))
            .collect();
        tries
            .into_iter()
            .filter_map(|t| t.join().ok()?.ok())
            .collect()
    })
}

/// Rank 0's side: send every other rank of `list` the ranges `ranges(rank)`
/// of `files` in folder `dir` that its store does not hold, every rank at
/// once, each rank connecting within `wait`. A range past its file, a file
/// that cannot be read and a rank that fails are refused by name.
/// Load-time only.
pub fn serve(
    list: &RankList,
    wait: Duration,
    dir: &Path,
    files: &[SourceFile],
    ranges: &(dyn Fn(usize) -> Vec<ReadRange> + Sync),
) -> Result<Vec<Served>, GpuError> {
    let link = |e: String| GpuError::link(WHAT, e);
    if let Some(f) = files.iter().find(|f| !inside(&f.name)) {
        return Err(GpuError::shape(
            WHAT,
            format!("{:?} is not a name inside the folder", f.name),
        ));
    }
    let data = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))
        .and_then(|l| Ok((l.local_addr()?.port(), l)))
        .map_err(|e| link(format!("a data listener: {e}")))?;
    let mut control = Control::open(list, 0, wait)?;
    // A rank hashes its whole store before it lists it, which can take longer than the
    // connect wait; a rank that dies still ends the read.
    for s in &control.socks {
        s.set_read_timeout(None)
            .map_err(|e| link(format!("a control socket: {e}")))?;
    }
    let mut welcome = wire::Out::default();
    welcome.u32(data.0.into()).u32(0);
    for (i, s) in control.socks.iter_mut().enumerate() {
        let port = read_frame(s).and_then(|b| {
            let mut r = wire::In::new(&b);
            let port = u16_at(&mut r, "relay port")?;
            r.end().map(|()| port)
        });
        welcome.u32(
            port.map_err(|e| link(format!("rank {}: {e}", i + 1)))?
                .into(),
        );
    }
    for s in &mut control.socks {
        write_frame(s, welcome.done()).map_err(link)?;
    }
    let mut held = Vec::new();
    let mut want = Vec::new();
    for (i, s) in control.socks.iter_mut().enumerate() {
        let fail = |e: String| link(format!("rank {}: {e}", i + 1));
        let mut have = Vec::new();
        loop {
            let (part, last) = decode_have(&read_frame(s).map_err(fail)?).map_err(fail)?;
            have.extend(part);
            if last {
                break;
            }
        }
        held.push(
            files
                .iter()
                .map(|f| {
                    Covering::of(
                        have.iter()
                            .filter(|(d, _, _)| *d == f.digest)
                            .map(|&(_, start, end)| (start, end - start)),
                    )
                })
                .collect::<Vec<_>>(),
        );
        let count = read_frame(s).and_then(|b| {
            let mut r = wire::In::new(&b);
            let n = r.len("stream count")?;
            r.end().map(|()| n)
        });
        want.push(count.map_err(fail)?);
    }
    let streams = accept(&data.1, &want, wait)?;
    let per_rank: Vec<Vec<ReadRange>> = (1..list.len()).map(|r| merged(ranges(r), GAP)).collect();
    for (i, list) in per_rank.iter().enumerate() {
        if let Some(r) = list.iter().find(|r| {
            files
                .get(r.file)
                .is_none_or(|f| r.offset.checked_add(r.len).is_none_or(|e| e > f.size))
        }) {
            return Err(link(format!("rank {}: a range {r:?} past its file", i + 1)));
        }
    }
    let mut served: Vec<Served> = (1..list.len())
        .map(|rank| Served {
            rank,
            sent: 0,
            kept: 0,
        })
        .collect();
    let work: Vec<(ReadRange, Vec<usize>)> = pieces(&per_rank, PIECE)
        .into_iter()
        .filter_map(|(at, ranks)| {
            let to: Vec<usize> = ranks
                .into_iter()
                .filter(|&i| {
                    let kept = held[i][at.file].holds(at.offset, at.len);
                    let s = &mut served[i];
                    *if kept { &mut s.kept } else { &mut s.sent } += at.len;
                    !kept
                })
                .collect();
            (!to.is_empty()).then_some((at, to))
        })
        .collect();
    for (s, sv) in control.socks.iter_mut().zip(&served) {
        let mut table = wire::Out::default();
        table.len(files.len());
        for f in files {
            table.str(&f.name).bytes(&f.digest).u64(f.size);
        }
        table.u64(sv.sent).u64(sv.kept);
        write_frame(s, table.done()).map_err(|e| link(format!("rank {}: {e}", sv.rank)))?;
    }
    let opened = files
        .iter()
        .map(|f| {
            File::open(dir.join(&f.name))
                .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", f.name)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    send(&work, files, &opened, streams)?;
    for (i, s) in control.socks.iter_mut().enumerate() {
        let done = read_frame(s).map_err(|e| link(format!("rank {}: {e}", i + 1)))?;
        if done != b"done" {
            return Err(link(format!(
                "rank {}: {}",
                i + 1,
                String::from_utf8_lossy(&done)
            )));
        }
    }
    for s in &mut control.socks {
        write_frame(s, b"over").map_err(link)?;
    }
    Ok(served)
}

/// The data streams of every rank (`want[i]` of rank `i + 1`), each naming
/// its rank, accepted on `data` within `wait`.
fn accept(
    data: &TcpListener,
    want: &[usize],
    wait: Duration,
) -> Result<Vec<Vec<TcpStream>>, GpuError> {
    let link = |e: String| GpuError::link(WHAT, format!("a data stream: {e}"));
    super::accept_ranks(data, want, wait, |s| {
        // A piece's tail and hash leave at once, not after the rank's delayed ACK.
        s.set_nodelay(true).map_err(|e| e.to_string())?;
        let mut h = [0u8; 5];
        s.read_exact(&mut h).map_err(|e| e.to_string())?;
        if h[0] != HELLO {
            return Err(format!("a stream named itself {h:?}"));
        }
        Ok(u32::from_le_bytes(h[1..].try_into().expect("4 bytes")) as usize)
    })
    .map_err(link)
}

/// A piece read and hashed, on its way to the ranks that receive it.
struct Piece {
    at: ReadRange,
    bytes: Vec<u8>,
    hash: [u8; 32],
}

/// Send each piece of `work` to its ranks on their `streams` (rank `i + 1`'s
/// at `i`): [`READERS`] threads read and hash the pieces in order, each
/// rank's streams take its pieces as they come. A piece that cannot be read
/// and a stream that fails are refused by name.
fn send(
    work: &[(ReadRange, Vec<usize>)],
    files: &[SourceFile],
    opened: &[File],
    streams: Vec<Vec<TcpStream>>,
) -> Result<(), GpuError> {
    let next = AtomicUsize::new(0);
    let fault = |e: String| GpuError::link(WHAT, e);
    std::thread::scope(|scope| {
        let mut queues = Vec::new();
        let mut jobs = Vec::new();
        for (i, socks) in streams.into_iter().enumerate() {
            let (tx, rx) = sync_channel::<Arc<Piece>>(QUEUE);
            queues.push(tx);
            let rx = Arc::new(Mutex::new(rx));
            for s in socks {
                let rx = Arc::clone(&rx);
                jobs.push(scope.spawn(move || {
                    send_stream(s, &rx).map_err(|e| format!("rank {}: {e}", i + 1))
                }));
            }
        }
        for _ in 0..READERS {
            let (queues, next) = (queues.clone(), &next);
            jobs.push(scope.spawn(move || -> Result<(), String> {
                while let Some((at, to)) = work.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let mut bytes = vec![0u8; at.len as usize];
                    opened[at.file]
                        .read_exact_at(&mut bytes, at.offset)
                        .map_err(|e| format!("{}: {e}", files[at.file].name))?;
                    let mut h = Sha256::new();
                    h.update(&bytes);
                    let piece = Arc::new(Piece {
                        at: *at,
                        bytes,
                        hash: h.finish(),
                    });
                    for &i in to {
                        queues[i]
                            .send(Arc::clone(&piece))
                            .map_err(|_| format!("rank {}: its streams ended early", i + 1))?;
                    }
                }
                Ok(())
            }));
        }
        drop(queues);
        let mut first = None;
        for j in jobs {
            let r = j
                .join()
                .unwrap_or_else(|_| Err("a sending thread panicked".to_string()));
            if let Err(e) = r {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), |e| Err(fault(e)))
    })
}

/// Send every piece `rx` yields on `s`, each its header, bytes and hash,
/// then the header that ends the stream.
fn send_stream(mut s: TcpStream, rx: &Mutex<Receiver<Arc<Piece>>>) -> Result<(), String> {
    let head = |file: u32, offset: u64, len: u64| {
        let mut b = [0u8; 20];
        b[..4].copy_from_slice(&file.to_le_bytes());
        b[4..12].copy_from_slice(&offset.to_le_bytes());
        b[12..].copy_from_slice(&len.to_le_bytes());
        b
    };
    loop {
        let next = rx.lock().unwrap_or_else(PoisonError::into_inner).recv();
        let Ok(p) = next else { break };
        s.write_all(&head(p.at.file as u32, p.at.offset, p.at.len))
            .and_then(|()| s.write_all(&p.bytes))
            .and_then(|()| s.write_all(&p.hash))
            .map_err(|e| e.to_string())?;
    }
    s.write_all(&head(END, 0, 0)).map_err(|e| e.to_string())
}

/// The data addresses this rank reaches rank 0 at, once its streams know
/// them, for the relays it serves.
#[derive(Default)]
struct Reach(Mutex<Option<Vec<SocketAddr>>>, Condvar);

impl Reach {
    fn set(&self, at: Vec<SocketAddr>) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(at);
        self.1.notify_all();
    }

    /// The addresses, waiting up to `wait` for them; none without.
    fn get(&self, wait: Duration) -> Vec<SocketAddr> {
        let g = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let (g, _) = self
            .1
            .wait_timeout_while(g, wait, |at| at.is_none())
            .unwrap_or_else(PoisonError::into_inner);
        g.clone().unwrap_or_default()
    }
}

/// This rank's data streams to rank 0 (module doc), each named: at `port`
/// of every fabric address of rank 0 it reaches, which `direct` then
/// holds; through the relay at `relays[p]` of every rank `p` it reaches;
/// or at rank 0's control address.
fn open_streams(
    list: &RankList,
    me: usize,
    port: u16,
    relays: &[u16],
    direct: &Reach,
    wait: Duration,
) -> Result<Vec<TcpStream>, String> {
    let zero = list.rank(0);
    let at: Vec<SocketAddr> = zero.data.iter().map(|&a| (a, port).into()).collect();
    let reached = reach(&at);
    direct.set(reached.iter().map(|(a, _)| *a).collect());
    let more = |a: SocketAddr| {
        (1..STREAMS)
            .map(move |_| TcpStream::connect_timeout(&a, REACH).map_err(|e| format!("{a}: {e}")))
    };
    let mut out = Vec::new();
    for (a, s) in reached {
        out.push(s);
        for s in more(a) {
            out.push(s?);
        }
    }
    if out.is_empty() {
        let via: Vec<SocketAddr> = (1..list.len())
            .filter(|&p| p != me)
            .flat_map(|p| {
                list.rank(p)
                    .data
                    .iter()
                    .map(move |&a| (a, relays[p]).into())
            })
            .collect();
        let through = |mut s: TcpStream| -> Option<TcpStream> {
            s.write_all(&[RELAY]).ok()?;
            s.write_all(&hello(HELLO, me)).ok()?;
            s.set_read_timeout(Some(wait)).ok()?;
            let mut ack = [0u8];
            s.read_exact(&mut ack).ok()?;
            s.set_read_timeout(None).ok()?;
            (ack[0] == ACK).then_some(s)
        };
        for (a, s) in reach(&via) {
            out.extend(through(s));
            out.extend(more(a).filter_map(|s| through(s.ok()?)));
        }
        if !out.is_empty() {
            return Ok(out);
        }
        let a: SocketAddr = (zero.control.ip(), port).into();
        out.push(TcpStream::connect_timeout(&a, REACH).map_err(|e| format!("{a}: {e}"))?);
        out.extend(more(a).collect::<Result<Vec<_>, _>>()?);
    }
    for s in &mut out {
        s.write_all(&hello(HELLO, me)).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// The relays a rank serves while it fetches: the addresses it reaches
/// rank 0 at ([`Reach`]), whether to take no more requests, and the
/// connections of the streams it passes on, which a failure cuts.
#[derive(Default)]
struct Relays {
    direct: Reach,
    stop: AtomicBool,
    /// `None` once cut.
    live: Mutex<Option<Vec<TcpStream>>>,
}

impl Relays {
    fn new() -> Relays {
        Relays {
            live: Mutex::new(Some(Vec::new())),
            ..Relays::default()
        }
    }

    /// Keep `s` to cut on a failure; `None` once cut, and `s` is shut down.
    fn hold(&self, s: &TcpStream) -> Option<()> {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        match (live.as_mut(), s.try_clone()) {
            (Some(v), Ok(c)) => {
                v.push(c);
                Some(())
            }
            _ => {
                let _ = s.shutdown(Shutdown::Both);
                None
            }
        }
    }

    /// Shut down every passed-on stream, so no relay waits on rank 0 past this
    /// rank's failure, and every later one at once.
    fn cut(&self) {
        let live = self
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        for s in live.into_iter().flatten() {
            let _ = s.shutdown(Shutdown::Both);
        }
    }

    /// Serve relay requests on `listener` until `stop`: each connects on to
    /// rank 0 at an address of `direct`, in turn, answers [`ACK`] and passes
    /// rank 0's bytes on until rank 0 ends the stream.
    fn serve(&self, listener: &TcpListener, wait: Duration) {
        let turn = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            while !self.stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((s, _)) => {
                        let turn = &turn;
                        scope.spawn(move || self.one(s, turn, wait));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
    }

    /// One relayed stream (see [`Relays::serve`]); a request that fails ends
    /// with no answer, and the rank that asked counts no stream.
    /// `hold` keeps a handle of each stream, so every exit closes the ones it opened: the rank
    /// that asked reads the end at once, and rank 0 does not write on to a rank that is gone.
    fn one(&self, mut from: TcpStream, turn: &AtomicUsize, wait: Duration) -> Option<()> {
        self.hold(&from)?;
        let passed = (|| {
            from.set_nonblocking(false).ok()?;
            from.set_nodelay(true).ok()?;
            from.set_read_timeout(Some(wait)).ok()?;
            let mut ask = [0u8; 6];
            from.read_exact(&mut ask).ok()?;
            let at = self.direct.get(wait);
            if ask[0] != RELAY || at.is_empty() {
                return None;
            }
            let a = at[turn.fetch_add(1, Ordering::Relaxed) % at.len()];
            let mut to = TcpStream::connect_timeout(&a, REACH).ok()?;
            self.hold(&to)?;
            let passed = (|| {
                to.write_all(&ask[1..]).ok()?;
                from.write_all(&[ACK]).ok()?;
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = to.read(&mut buf).ok()?;
                    if n == 0 {
                        return Some(());
                    }
                    from.write_all(&buf[..n]).ok()?;
                }
            })();
            let _ = to.shutdown(Shutdown::Both);
            passed
        })();
        let _ = from.shutdown(Shutdown::Both);
        passed
    }
}

/// A sparse copy open for writing: through the page cache, and without it
/// (`O_DIRECT`) where the store's file system takes that. A drive writes
/// direct pages several times faster than the page cache's write-back.
struct Out {
    cached: File,
    direct: Option<File>,
}

impl Out {
    fn open(store: &Store, file: &SourceFile) -> Result<Out, String> {
        use std::os::unix::fs::OpenOptionsExt;
        let cached = store.sparse(file)?;
        let direct = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(store.path(&file.digest))
            .ok();
        Ok(Out { cached, direct })
    }

    /// Write `bytes` at `at`: the whole pages within them without the page
    /// cache when `bytes` lie in memory as the pages do in the file, the rest
    /// through it.
    fn write(&self, bytes: &[u8], at: u64) -> std::io::Result<()> {
        let page = DIRECT_ALIGN as u64;
        let end = at + bytes.len() as u64;
        let (first, last) = (at.next_multiple_of(page), end / page * page);
        let i = (first - at) as usize;
        match &self.direct {
            Some(d)
                if last > first && (bytes.as_ptr() as usize + i).is_multiple_of(DIRECT_ALIGN) =>
            {
                let j = (last - at) as usize;
                self.cached.write_all_at(&bytes[..i], at)?;
                d.write_all_at(&bytes[i..j], first)?;
                self.cached.write_all_at(&bytes[j..], last)
            }
            _ => self.cached.write_all_at(bytes, at),
        }
    }
}

/// A rank's received pieces on their way to its store: the sparse copies
/// open for writing, the bytes received, and every piece received, which
/// [`record`] syncs and records apart from the streams.
struct Sink<'a> {
    store: &'a Store,
    files: &'a [SourceFile],
    open: Mutex<Vec<Option<Arc<Out>>>>,
    got: Mutex<Vec<(usize, Covered)>>,
    bytes: AtomicU64,
}

impl Sink<'_> {
    fn file(&self, i: usize) -> Result<Arc<Out>, String> {
        let mut open = self.open.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(f) = &open[i] {
            return Ok(Arc::clone(f));
        }
        let f = Arc::new(Out::open(self.store, &self.files[i])?);
        open[i] = Some(Arc::clone(&f));
        Ok(f)
    }

    /// Receive and write every piece stream `s` carries, until its end, each
    /// written piece sent on to `done`.
    fn take(
        &self,
        s: TcpStream,
        done: &std::sync::mpsc::Sender<(usize, Covered)>,
    ) -> Result<(), String> {
        let mut input = BufReader::with_capacity(CHUNK, s);
        let mut buf = vec![0u8; PIECE as usize + 2 * DIRECT_ALIGN];
        loop {
            let mut head = [0u8; 20];
            input.read_exact(&mut head).map_err(|e| e.to_string())?;
            let file = u32::from_le_bytes(head[..4].try_into().expect("4 bytes"));
            if file == END {
                return Ok(());
            }
            let file = file as usize;
            let offset = u64::from_le_bytes(head[4..12].try_into().expect("8 bytes"));
            let len = u64::from_le_bytes(head[12..].try_into().expect("8 bytes"));
            let f = self
                .files
                .get(file)
                .filter(|f| len <= PIECE && offset.checked_add(len).is_some_and(|e| e <= f.size))
                .ok_or_else(|| format!("a range of file {file} at {offset} past its file"))?;
            // The bytes lie in `buf` as their pages do in the file, so whole pages go direct.
            let at =
                buf.as_ptr().align_offset(DIRECT_ALIGN) + (offset % DIRECT_ALIGN as u64) as usize;
            let bytes = &mut buf[at..at + len as usize];
            let hash = receive(f, offset, &mut input, bytes)?;
            self.file(file)?
                .write(bytes, offset)
                .map_err(|e| format!("{}'s sparse copy: {e}", f.name))?;
            self.bytes.fetch_add(len, Ordering::Relaxed);
            let c = Covered { offset, len, hash };
            self.got
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((file, c));
            let _ = done.send((file, c));
        }
    }
}

/// Record the pieces `done` yields in `sink`'s store: once [`RECORD`] bytes
/// of them are written, and once `done` ends, sync the sparse copies and only
/// then append those pieces to their coverage. The streams never wait on it.
fn record(sink: &Sink, done: Receiver<(usize, Covered)>) -> Result<(), String> {
    let flush = |batch: Vec<(usize, Covered)>| {
        let open: Vec<Arc<Out>> = sink
            .open
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .flatten()
            .cloned()
            .collect();
        let cached: Vec<&File> = open.iter().map(|o| &o.cached).collect();
        sink.store.record(sink.files, &cached, batch)
    };
    let (mut batch, mut bytes) = (Vec::new(), 0u64);
    for (file, c) in done {
        batch.push((file, c));
        bytes += c.len;
        if bytes >= RECORD {
            flush(std::mem::take(&mut batch))?;
            bytes = 0;
        }
    }
    flush(batch)
}

/// A rank's side: tell rank 0 of `list` what the store at `store` holds,
/// then receive and record every piece rank 0 sends, rank 0 reached within
/// `wait`, and relay other ranks' streams until rank 0 ends the exchange. A
/// piece that does not hash to rank 0's, a store that cannot be written and
/// a control channel that fails are refused by name. Load-time only.
pub fn fetch(
    list: &RankList,
    me: usize,
    wait: Duration,
    store: &Path,
) -> Result<Fetched, GpuError> {
    let fail = |e: String| GpuError::link(WHAT, format!("rank {me}: {e}"));
    if me == 0 {
        return Err(GpuError::shape(
            WHAT,
            "rank 0 holds the source and serves it",
        ));
    }
    let store = Store::open(store)?;
    let relay = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))
        .and_then(|l| {
            l.set_nonblocking(true)?;
            Ok((l.local_addr()?.port(), l))
        })
        .map_err(|e| fail(format!("a relay listener: {e}")))?;
    let mut control = Control::open(list, me, wait)?;
    let s = &mut control.socks[0];
    write_frame(s, wire::Out::default().u32(relay.0.into()).done()).map_err(fail)?;
    let welcome = read_frame(s).map_err(fail)?;
    let mut r = wire::In::new(&welcome);
    let port = u16_at(&mut r, "data port").map_err(fail)?;
    let relays = (0..list.len())
        .map(|_| u16_at(&mut r, "relay port"))
        .collect::<Result<Vec<_>, _>>()
        .map_err(fail)?;
    r.end().map_err(fail)?;
    // Rank 0 sends the table only after every rank listed its store, and ends the exchange only
    // after every rank took its pieces: either can take longer than the connect wait; a rank 0
    // that dies still ends the read.
    s.set_read_timeout(None)
        .map_err(|e| fail(format!("the control socket: {e}")))?;
    let relaying = Relays::new();
    std::thread::scope(|scope| {
        let serving = scope.spawn(|| relaying.serve(&relay.1, wait));
        let result = receive_all(s, &store, list, me, (port, &relays), &relaying.direct, wait);
        // An early failure leaves no relay waiting on this rank's addresses, tells rank 0 by the
        // control channel's end and cuts every stream this rank passes on, so no relay waits on
        // rank 0 and rank 0 waits on no list of this rank's.
        relaying.direct.set(Vec::new());
        relaying.stop.store(true, Ordering::Release);
        if result.is_err() {
            let _ = s.shutdown(Shutdown::Both);
            relaying.cut();
        }
        let _ = serving.join();
        result.map_err(fail)
    })
}

/// [`fetch`] once the relays listen, with rank 0's data port and every
/// rank's relay port: the streams, the store's list, the pieces, the answer
/// and rank 0's end of the exchange.
fn receive_all(
    s: &mut TcpStream,
    store: &Store,
    list: &RankList,
    me: usize,
    (port, relays): (u16, &[u16]),
    direct: &Reach,
    wait: Duration,
) -> Result<Fetched, String> {
    let streams = open_streams(list, me, port, relays, direct, wait)?;
    let have = store.held().map_err(|e| e.to_string())?;
    let parts: Vec<_> = have.chunks(HAVE_FRAME).collect();
    for (i, part) in parts.iter().enumerate() {
        write_frame(s, &encode_have(part, i + 1 == parts.len()))?;
    }
    if parts.is_empty() {
        write_frame(s, &encode_have(&[], true))?;
    }
    write_frame(s, wire::Out::default().len(streams.len()).done())?;
    let table = read_frame(s)?;
    let mut r = wire::In::new(&table);
    let files = (0..r.len("files")?)
        .map(|_| {
            let name = r.str("file name")?;
            if !inside(&name) {
                return Err(format!("{name:?} is not a name inside the folder"));
            }
            Ok(SourceFile {
                name,
                digest: array_at(&mut r, "digest")?,
                size: r.u64("size")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (sent, kept) = (r.u64("sent")?, r.u64("kept")?);
    r.end()?;
    let sink = Sink {
        store,
        files: &files,
        open: Mutex::new(files.iter().map(|_| None).collect()),
        got: Mutex::new(Vec::new()),
        bytes: AtomicU64::new(0),
    };
    let result = std::thread::scope(|scope| {
        let (done, pieces) = std::sync::mpsc::channel();
        let sink_at = &sink;
        let recording = scope.spawn(move || record(sink_at, pieces));
        let takes: Vec<_> = streams
            .into_iter()
            .map(|t| {
                let (sink, done) = (&sink, done.clone());
                scope.spawn(move || sink.take(t, &done))
            })
            .collect();
        drop(done);
        let mut first = None;
        for t in takes.into_iter().chain([recording]) {
            if let Err(e) = t
                .join()
                .unwrap_or_else(|_| Err("a receiving thread panicked".into()))
            {
                first.get_or_insert(e);
            }
        }
        first.map_or(Ok(()), Err)
    })
    .and_then(|()| {
        let got = sink.bytes.load(Ordering::Relaxed);
        (got == sent)
            .then_some(())
            .ok_or_else(|| format!("received {got} B of the {sent} B rank 0 sends"))
    });
    let answer = match &result {
        Ok(()) => b"done".to_vec(),
        Err(e) => e.clone().into_bytes(),
    };
    write_frame(s, &answer)?;
    result?;
    let over = read_frame(s)?;
    if over != b"over" {
        return Err(format!(
            "rank 0 ended with {:?}",
            String::from_utf8_lossy(&over)
        ));
    }
    let got = sink
        .got
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    let files = files
        .into_iter()
        .enumerate()
        .map(|(i, f)| {
            let held = Covering::of(
                have.iter()
                    .filter(|(d, _, _)| *d == f.digest)
                    .map(|&(_, start, end)| (start, end - start))
                    .chain(
                        got.iter()
                            .filter(|g| g.0 == i)
                            .map(|g| (g.1.offset, g.1.len)),
                    ),
            );
            let p = store.path(&f.digest);
            (f, p, held)
        })
        .collect();
    Ok(Fetched { files, sent, kept })
}

/// The held ranges one frame of a rank's store list carries: 48 bytes each,
/// well under the control channel's frame limit.
const HAVE_FRAME: usize = 1 << 16;

/// A held range as a rank lists it: the file's digest, the merged range's
/// start and end.
type Held = ([u8; 32], u64, u64);

/// One frame of a rank's held ranges, and whether it is the last frame.
fn encode_have(have: &[Held], last: bool) -> Vec<u8> {
    let mut o = wire::Out::default();
    o.u8(u8::from(last)).len(have.len());
    for (d, start, end) in have {
        o.bytes(d).u64(*start).u64(*end);
    }
    o.done().to_vec()
}

fn decode_have(b: &[u8]) -> Result<(Vec<Held>, bool), String> {
    let mut r = wire::In::new(b);
    let last = r.flag("last")?;
    let v = (0..r.len("ranges")?)
        .map(|_| {
            let d = array_at(&mut r, "digest")?;
            let (start, end) = (r.u64("start")?, r.u64("end")?);
            if end < start {
                return Err(format!("a held range {start}..{end}"));
            }
            Ok((d, start, end))
        })
        .collect::<Result<Vec<_>, String>>()?;
    r.end()?;
    Ok((v, last))
}

#[cfg(test)]
mod tests {
    use super::{
        ACK, DIRECT_ALIGN, HAVE_FRAME, HELLO, Out, RELAY, ReadRange, Relays, SourceFile, Store,
        fetch, hello, inside, read_ranges, serve, verify,
    };
    use crate::host::rank::RankList;
    use gguf::digest::Sha256;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bloomery-dist-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // Rank 0 sends each rank its ranges at their offsets on its data streams, a run of short
    // pieces (one cut along a tensor's input axis) as one range with its gaps, read once for both
    // ranks; a second run sends nothing the store holds and sends again only a piece whose kept
    // bytes no longer hash to their record; a store list longer than one frame goes in several;
    // the sparse copy never takes the source's name, and what a rank holds is what it received.
    #[test]
    fn a_rank_receives_its_ranges_once_and_keeps_them_by_digest() {
        let root = scratch("serve");
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let bytes: Vec<u8> = (0..3u32 << 20).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(src.join("model-00001.safetensors"), &bytes).unwrap();
        let mut h = Sha256::new();
        h.update(&bytes);
        let file = SourceFile {
            name: "model-00001.safetensors".into(),
            digest: h.finish(),
            size: bytes.len() as u64,
        };
        let port = 50_000 + (std::process::id() % 1000) as u16;
        let list = RankList::parse(&format!(
            "127.0.0.1:{port} 127.0.0.1\n127.0.0.2:1 127.0.0.2\n127.0.0.3:1 127.0.0.3\n"
        ))
        .unwrap();
        // The far range first: rank 0 sorts what it sends.
        let far = |rank: usize| (2 << 20) + 10_000 * rank;
        let ranges = |rank: usize| {
            let mut v = vec![ReadRange {
                file: 0,
                offset: far(rank) as u64,
                len: 5_000,
            }];
            v.extend((0..8).map(|row| ReadRange {
                file: 0,
                offset: row * 400,
                len: 18,
            }));
            v
        };
        // Rows 0 to 7 merge into 0..2818 with their gaps; the far range stays apart.
        let whole = 2_818 + 5_000;
        let stores: Vec<PathBuf> = (1..3).map(|r| root.join(format!("store{r}"))).collect();
        let runs = [
            (0, [whole as u64; 2]),
            (1, [0; 2]),
            (2, [2_818, 0]),
            (3, [whole as u64, 0]),
        ];
        for (run, want_sent) in runs {
            if run == 3 {
                // Rank 1's store holds the whole file in more short apart ranges than one frame
                // of its list carries; none holds rows 0 to 7 or the far range whole.
                let store = Store::open(&stores[0]).unwrap();
                std::fs::write(store.path(&file.digest), &bytes).unwrap();
                let mut lines = String::new();
                for i in 0..HAVE_FRAME as u64 + 10 {
                    let (offset, len) = (i * 40, 18u64);
                    let mut h = Sha256::new();
                    h.update(&bytes[offset as usize..(offset + len) as usize]);
                    lines.push_str(&format!(
                        "{offset} {len} {}\n",
                        gguf::digest::hex(&h.finish())
                    ));
                }
                let cov = stores[0].join(format!("{}.coverage", gguf::digest::hex(&file.digest)));
                std::fs::write(cov, lines).unwrap();
            }
            if run == 2 {
                // A flipped byte in rank 1's kept copy of rows 0 to 7.
                let path = Store::open(&stores[0]).unwrap().path(&file.digest);
                let mut copy = std::fs::read(&path).unwrap();
                copy[100] ^= 0xff;
                std::fs::write(&path, copy).unwrap();
            }
            let (served, fetched) = std::thread::scope(|s| {
                let f: Vec<_> = (1..3)
                    .map(|r| {
                        let (list, store) = (&list, &stores[r - 1]);
                        s.spawn(move || fetch(list, r, Duration::from_secs(10), store).unwrap())
                    })
                    .collect();
                let served = serve(
                    &list,
                    Duration::from_secs(10),
                    &src,
                    std::slice::from_ref(&file),
                    &ranges,
                )
                .unwrap();
                (
                    served,
                    f.into_iter().map(|j| j.join().unwrap()).collect::<Vec<_>>(),
                )
            });
            for (r, (sv, f)) in served.iter().zip(&fetched).enumerate() {
                let rank = r + 1;
                assert_eq!(
                    (sv.rank, sv.sent, f.sent),
                    (rank, want_sent[r], want_sent[r]),
                    "run {run}"
                );
                assert_eq!(sv.kept + sv.sent, whole as u64);
                let store = Store::open(&stores[r]).unwrap();
                let copy = std::fs::read(store.path(&file.digest)).unwrap();
                assert_eq!(copy.len(), bytes.len());
                let at = far(rank);
                assert_eq!(&copy[at..at + 5_000], &bytes[at..at + 5_000]);
                assert_eq!(&copy[..2_818], &bytes[..2_818]);
                let held = &f.files[0].2;
                assert!(held.holds(at as u64 + 10, 100) && held.holds(300, 10));
                assert!(!held.holds(2_818, 1) && !held.holds(1 << 20, 10));
                assert!(!stores[r].join(&file.name).exists());
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A rank that reaches rank 0 through a relay: the relay connects on to rank 0, passes the
    // stream's name on, answers, and passes rank 0's bytes back until rank 0 ends the stream; a
    // failure of the relaying rank cuts a stream rank 0 still holds open. A relay that reaches
    // rank 0 on no address ends the request at once.
    #[test]
    fn a_relay_passes_a_stream_on_to_rank_zero_until_a_failure_cuts_it() {
        let zero = TcpListener::bind("127.0.0.1:0").unwrap();
        let relay = TcpListener::bind("127.0.0.1:0").unwrap();
        relay.set_nonblocking(true).unwrap();
        let ask_at = |relay: &TcpListener| {
            let mut asker = TcpStream::connect(relay.local_addr().unwrap()).unwrap();
            asker.write_all(&[RELAY]).unwrap();
            asker.write_all(&hello(HELLO, 2)).unwrap();
            asker
        };
        let refused = Relays::new();
        refused.direct.set(Vec::new());
        let asked = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut asker = ask_at(&asked);
        let (from, _) = asked.accept().unwrap();
        assert!(
            refused
                .one(from, &AtomicUsize::new(0), Duration::from_secs(5))
                .is_none()
        );
        asker
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let ends = asker.read_to_end(&mut Vec::new()).is_ok_and(|n| n == 0);
        assert!(ends, "a refused request reads its end at once");
        let relays = Relays::new();
        let ask = || ask_at(&relay);
        let acked = |asker: &mut TcpStream| {
            let mut ack = [0u8];
            asker.read_exact(&mut ack).unwrap();
            assert_eq!(ack[0], ACK);
        };
        std::thread::scope(|s| {
            let serving = s.spawn(|| relays.serve(&relay, Duration::from_secs(5)));
            let mut asker = ask();
            relays.direct.set(vec![zero.local_addr().unwrap()]);
            let (mut at_zero, _) = zero.accept().unwrap();
            let mut name = [0u8; 5];
            at_zero.read_exact(&mut name).unwrap();
            assert_eq!(name, hello(HELLO, 2));
            acked(&mut asker);
            at_zero.write_all(b"piece").unwrap();
            drop(at_zero);
            let mut got = Vec::new();
            asker.read_to_end(&mut got).unwrap();
            assert_eq!(got, b"piece");
            let mut second = ask();
            let (_open, _) = zero.accept().unwrap();
            acked(&mut second);
            relays.stop.store(true, Ordering::Release);
            relays.cut();
            second.read_to_end(&mut got).unwrap();
            serving.join().unwrap();
        });
    }

    // The head hashes a source file once: a write that keeps the length and restores the
    // modification time still changes the change time, so the file is hashed again and refused. Only a name
    // inside the folder is one, and a read of a shard in a subfolder is that file's.
    #[test]
    fn the_head_checks_a_file_once_by_its_stamp_and_names_only_inside_its_folder() {
        let root = scratch("verify");
        let dir = root.join("src");
        std::fs::create_dir_all(dir.join("weights")).unwrap();
        let path = dir.join("weights/a.safetensors");
        std::fs::write(&path, b"weights").unwrap();
        let mut h = Sha256::new();
        h.update(b"weights");
        let file = SourceFile {
            name: "weights/a.safetensors".into(),
            digest: h.finish(),
            size: 7,
        };
        let one = std::slice::from_ref(&file);
        let record = root.join("record/verified.tsv");
        verify(&dir, one, &record).unwrap();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(b"WEIGHTS").unwrap();
        f.set_modified(mtime).unwrap();
        let e = verify(&dir, one, &record).unwrap_err().to_string();
        assert!(
            e.contains("weights/a.safetensors: its bytes hash to"),
            "{e}"
        );
        for (name, ok) in [
            ("weights/a.safetensors", true),
            ("../x", false),
            ("/etc/passwd", false),
            ("a/../../b", false),
            ("", false),
        ] {
            assert_eq!(inside(name), ok, "{name:?}");
        }
        let outside = SourceFile {
            name: "../x".into(),
            ..file.clone()
        };
        assert!(verify(&dir, std::slice::from_ref(&outside), &record).is_err());
        let reads = [(path.clone(), 0..3), (path, 4..6)];
        assert_eq!(
            read_ranges(one, &dir, &reads).unwrap(),
            [ReadRange {
                file: 0,
                offset: 0,
                len: 6
            }]
        );
        assert!(read_ranges(one, &dir, &[(dir.join("a.safetensors"), 0..3)]).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A piece whose bytes do not hash to rank 0's hash is refused; one that does is written at
    // its offset, its whole pages without the page cache where the store takes that.
    #[test]
    fn a_piece_is_written_once_it_hashes_to_rank_zeros() {
        let root = scratch("hash");
        let file = SourceFile {
            name: "f".into(),
            digest: [7; 32],
            size: 5 * DIRECT_ALIGN as u64,
        };
        let store = Store::open(&root).unwrap();
        let out = Out::open(&store, &file).unwrap();
        let (offset, len) = (100u64, 3 * DIRECT_ALIGN);
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let mut h = Sha256::new();
        h.update(&bytes);
        let good = h.finish();
        let mut bad = good;
        bad[0] ^= 1;
        let mut buf = vec![0u8; len + 2 * DIRECT_ALIGN];
        let at = buf.as_ptr().align_offset(DIRECT_ALIGN) + offset as usize;
        for (hash, ok) in [(bad, false), (good, true)] {
            let mut wire = bytes.clone();
            wire.extend(hash);
            let got = super::receive(&file, offset, &mut &wire[..], &mut buf[at..at + len]);
            assert_eq!(got.is_ok(), ok, "{hash:?}");
        }
        out.write(&buf[at..at + len], offset).unwrap();
        let copy = std::fs::read(store.path(&file.digest)).unwrap();
        assert_eq!(&copy[offset as usize..offset as usize + len], &bytes[..]);
        assert!(copy[..offset as usize].iter().all(|&b| b == 0));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
