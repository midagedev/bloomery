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
//! coverage reads it ([`Store::covers`]).
//!
//! Rank 0 merges a rank's ranges of one file whose gaps are at most
//! [`GAP`] bytes into one before it sends them: a cut along a tensor's input
//! axis reads a short piece of every row, and sending, writing and hashing
//! each piece alone is slower than sending the whole tensor. A rank so holds
//! more than it reads, and its coverage lists the merged ranges. A rank
//! writes and hashes a range in [`CHUNK`]s and records its verified ranges
//! once at least [`RECORD`] bytes of them are on the disk, and at the end.
//!
//! The exchange runs on the control addresses before the group opens
//! ([`super::RankSession::open`]): every other rank dials rank 0 and sends
//! the ranges its store holds, verified and merged where they touch, in
//! frames of at most [`HAVE_FRAME`] ranges; rank 0 sends each rank the file table, then
//! each of its ranges, kept or with its bytes and their hash; the rank
//! answers once every range is written.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use gguf::digest::{Sha256, hex};

use super::{Control, RankList, array_at, read_frame, write_frame};
use crate::GpuError;
use crate::host::node::wire;

const WHAT: &str = "the head's distribution";

/// The bytes a range moves in one read and write.
const CHUNK: usize = 8 << 20;

/// The largest gap between two ranges of one file that rank 0 sends as one.
pub const GAP: u64 = 1 << 20;

/// The received bytes a rank holds before it syncs them and records their
/// ranges in the coverage.
const RECORD: u64 = 1 << 30;

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

/// The ranges a store holds of one file, merged and in order
/// ([`Store::covering`]).
#[derive(Clone, Debug, Default)]
pub struct Covering(Vec<(u64, u64)>);

impl Covering {
    fn of(ranges: impl IntoIterator<Item = (u64, u64)>) -> Covering {
        let mut v: Vec<(u64, u64)> = ranges
            .into_iter()
            .map(|(offset, len)| (offset, offset.saturating_add(len)))
            .collect();
        v.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(v.len());
        for (start, end) in v {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        Covering(merged)
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

/// A rank's store folder of received ranges (module doc).
pub struct Store {
    dir: PathBuf,
}

impl Store {
    /// The store at `dir`, made when absent.
    pub fn open(dir: &Path) -> Result<Store, GpuError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", dir.display())))?;
        Ok(Store {
            dir: dir.to_path_buf(),
        })
    }

    /// The sparse copy of the file of `digest`.
    #[must_use]
    pub fn path(&self, digest: &[u8; 32]) -> PathBuf {
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

    /// The coverage of the file of `digest` whose bytes in the sparse copy
    /// still hash to what the coverage records; a range whose bytes do not is
    /// left out, so rank 0 sends it again.
    fn verified(&self, digest: &[u8; 32]) -> Result<Vec<Covered>, GpuError> {
        let path = self.path(digest);
        let Some(data) = open_if(&path)? else {
            return Ok(Vec::new());
        };
        let mut buf = vec![0u8; CHUNK];
        let mut kept = Vec::new();
        for c in self.coverage(digest)? {
            let mut h = Sha256::new();
            let mut at = 0u64;
            let read = loop {
                if at == c.len {
                    break Ok(());
                }
                let n = (c.len - at).min(CHUNK as u64) as usize;
                if let Err(e) = data.read_exact_at(&mut buf[..n], c.offset + at) {
                    break Err(e);
                }
                h.update(&buf[..n]);
                at += n as u64;
            };
            match read {
                Ok(()) if h.finish() == c.hash => kept.push(c),
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {}
                Err(e) => return Err(GpuError::shape(WHAT, format!("{}: {e}", path.display()))),
            }
        }
        Ok(kept)
    }

    /// What the store holds of the file of `digest`, its ranges whose bytes
    /// still hash to their record, read once for many checks: a reader of the
    /// sparse copy reads no byte it does not hold.
    pub fn covering(&self, digest: &[u8; 32]) -> Result<Covering, GpuError> {
        Ok(Covering::of(
            self.verified(digest)?.iter().map(|c| (c.offset, c.len)),
        ))
    }

    /// Whether the store holds the `len` bytes at `offset` of the file of
    /// `digest` ([`Store::covering`] for many checks).
    pub fn covers(&self, digest: &[u8; 32], offset: u64, len: u64) -> Result<bool, GpuError> {
        Ok(self.covering(digest)?.holds(offset, len))
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
        open: &[Option<File>],
        pending: &mut Vec<(usize, Covered)>,
    ) -> Result<(), String> {
        for f in open.iter().flatten() {
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
        pending.clear();
        Ok(())
    }
}

/// The digest whose 64 hex digits are `s`, or `None`.
fn unhex(s: &str) -> Option<[u8; 32]> {
    let mut d = [0u8; 32];
    if s.len() != 64 {
        return None;
    }
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(d)
}

/// The file at `path` opened for reading, or `None` when there is none.
fn open_if(path: &Path) -> Result<Option<File>, GpuError> {
    match File::open(path) {
        Ok(f) => Ok(Some(f)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(GpuError::shape(WHAT, format!("{}: {e}", path.display()))),
    }
}

/// Write `len` bytes from `from` at `offset` of `out`, the sparse copy of
/// `file`, through `buf`, and return their hash once it is rank 0's.
fn receive(
    file: &SourceFile,
    out: &File,
    offset: u64,
    len: u64,
    from: &mut impl Read,
    buf: &mut [u8],
) -> Result<[u8; 32], String> {
    let mut h = Sha256::new();
    let mut at = 0u64;
    while at < len {
        let n = (len - at).min(buf.len() as u64) as usize;
        from.read_exact(&mut buf[..n])
            .map_err(|e| format!("{}'s bytes: {e}", file.name))?;
        h.update(&buf[..n]);
        out.write_all_at(&buf[..n], offset + at)
            .map_err(|e| format!("{}'s sparse copy: {e}", file.name))?;
        at += n as u64;
    }
    let mut want = [0u8; 32];
    from.read_exact(&mut want)
        .map_err(|e| format!("{}'s hash: {e}", file.name))?;
    if h.finish() != want {
        return Err(format!(
            "{} bytes {offset}..{}: the bytes do not hash to rank 0's",
            file.name,
            offset + len
        ));
    }
    Ok(want)
}

/// What one rank received in a distribution: each source file and its
/// sparse copy, and the bytes sent and kept.
#[derive(Debug)]
pub struct Fetched {
    pub files: Vec<(SourceFile, PathBuf)>,
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

/// Rank 0's side: send every other rank of `list` the ranges `ranges(rank)`
/// of `files` in folder `dir` that its store does not hold, every rank at
/// once, each rank connecting within `wait`. A range past its file, a file that cannot be read and a rank that
/// fails are refused by name. Load-time only.
pub fn serve(
    list: &RankList,
    wait: Duration,
    dir: &Path,
    files: &[SourceFile],
    ranges: &(dyn Fn(usize) -> Vec<ReadRange> + Sync),
) -> Result<Vec<Served>, GpuError> {
    let mut control = Control::open(list, 0, wait)?;
    // A rank hashes its whole store before it lists it, which can take longer than the
    // connect wait; a rank that dies still ends the read.
    for s in &control.socks {
        s.set_read_timeout(None)
            .map_err(|e| GpuError::link(WHAT, format!("a control socket: {e}")))?;
    }
    let opened = files
        .iter()
        .map(|f| {
            File::open(dir.join(&f.name))
                .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", f.name)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    std::thread::scope(|scope| {
        let jobs: Vec<_> = control
            .socks
            .iter_mut()
            .enumerate()
            .map(|(i, s)| {
                let opened = &opened;
                scope.spawn(move || serve_one(s, i + 1, files, opened, &merged(ranges(i + 1), GAP)))
            })
            .collect();
        jobs.into_iter()
            .map(|j| {
                j.join()
                    .map_err(|_| GpuError::shape(WHAT, "a sending thread panicked"))?
            })
            .collect()
    })
}

fn serve_one(
    s: &mut std::net::TcpStream,
    rank: usize,
    files: &[SourceFile],
    opened: &[File],
    ranges: &[ReadRange],
) -> Result<Served, GpuError> {
    let fail = |e: String| GpuError::link(WHAT, format!("rank {rank}: {e}"));
    let mut have = Vec::new();
    loop {
        let (part, last) = decode_have(&read_frame(s).map_err(fail)?).map_err(fail)?;
        have.extend(part);
        if last {
            break;
        }
    }
    let held: Vec<Covering> = files
        .iter()
        .map(|f| {
            Covering::of(
                have.iter()
                    .filter(|(d, _, _)| *d == f.digest)
                    .map(|&(_, start, end)| (start, end - start)),
            )
        })
        .collect();
    let mut table = wire::Out::default();
    table.len(files.len());
    for f in files {
        table.str(&f.name).bytes(&f.digest).u64(f.size);
    }
    table.len(ranges.len());
    write_frame(s, table.done()).map_err(fail)?;
    let mut served = Served {
        rank,
        sent: 0,
        kept: 0,
    };
    let mut buf = vec![0u8; CHUNK];
    let mut out = std::io::BufWriter::with_capacity(CHUNK, &mut *s);
    for r in ranges {
        let f = files
            .get(r.file)
            .filter(|f| r.offset.checked_add(r.len).is_some_and(|e| e <= f.size))
            .ok_or_else(|| fail(format!("a range {r:?} past its file")))?;
        let kept = held[r.file].holds(r.offset, r.len);
        let mut head = Vec::with_capacity(21);
        head.extend((r.file as u32).to_le_bytes());
        head.extend(r.offset.to_le_bytes());
        head.extend(r.len.to_le_bytes());
        head.push(u8::from(!kept));
        out.write_all(&head).map_err(|e| fail(e.to_string()))?;
        if kept {
            served.kept += r.len;
            continue;
        }
        let mut h = Sha256::new();
        let mut at = 0u64;
        while at < r.len {
            let n = (r.len - at).min(CHUNK as u64) as usize;
            opened[r.file]
                .read_exact_at(&mut buf[..n], r.offset + at)
                .map_err(|e| fail(format!("{}: {e}", f.name)))?;
            h.update(&buf[..n]);
            out.write_all(&buf[..n]).map_err(|e| fail(e.to_string()))?;
            at += n as u64;
        }
        out.write_all(&h.finish())
            .map_err(|e| fail(e.to_string()))?;
        served.sent += r.len;
    }
    out.flush().map_err(|e| fail(e.to_string()))?;
    drop(out);
    let done = read_frame(s).map_err(fail)?;
    if done != b"done" {
        return Err(fail(String::from_utf8_lossy(&done).into_owned()));
    }
    Ok(served)
}

/// A rank's side: tell rank 0 of `list` what the store at `store` holds,
/// then receive and record every range rank 0 sends, rank 0 reached within
/// `wait`. A range that does not
/// hash to rank 0's, a store that cannot be written and a control channel
/// that fails are refused by name. Load-time only.
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
    let mut control = Control::open(list, me, wait)?;
    let s = &mut control.socks[0];
    // The file table comes after the hello, so the store says what it holds of every file it has.
    let mut have = Vec::new();
    for entry in std::fs::read_dir(&store.dir).map_err(|e| fail(e.to_string()))? {
        let name = entry.map_err(|e| fail(e.to_string()))?.file_name();
        let Some(h) = name.to_str().and_then(|n| n.strip_suffix(".coverage")) else {
            continue;
        };
        let Some(d) = unhex(h) else {
            continue;
        };
        have.extend(
            store
                .covering(&d)?
                .0
                .into_iter()
                .map(|(start, end)| (d, start, end)),
        );
    }
    let parts: Vec<_> = have.chunks(HAVE_FRAME).collect();
    for (i, part) in parts.iter().enumerate() {
        write_frame(s, &encode_have(part, i + 1 == parts.len())).map_err(fail)?;
    }
    if parts.is_empty() {
        write_frame(s, &encode_have(&[], true)).map_err(fail)?;
    }
    let table = read_frame(s).map_err(fail)?;
    let mut r = wire::In::new(&table);
    let files = (0..r.len("files").map_err(fail)?)
        .map(|_| {
            Ok(SourceFile {
                name: r.str("file name")?,
                digest: array_at(&mut r, "digest")?,
                size: r.u64("size")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(fail)?;
    let count = r.len("ranges").map_err(fail)?;
    r.end().map_err(fail)?;
    let (mut sent, mut kept) = (0, 0);
    let mut input = std::io::BufReader::with_capacity(CHUNK, &mut *s);
    let mut open: Vec<Option<File>> = files.iter().map(|_| None).collect();
    let mut pending: Vec<(usize, Covered)> = Vec::new();
    let mut unrecorded = 0u64;
    let mut buf = vec![0u8; CHUNK];
    let result = (|| {
        for _ in 0..count {
            let mut head = [0u8; 21];
            input.read_exact(&mut head).map_err(|e| e.to_string())?;
            let file = u32::from_le_bytes(head[0..4].try_into().expect("4 bytes")) as usize;
            let offset = u64::from_le_bytes(head[4..12].try_into().expect("8 bytes"));
            let len = u64::from_le_bytes(head[12..20].try_into().expect("8 bytes"));
            let f = files
                .get(file)
                .filter(|f| offset.checked_add(len).is_some_and(|e| e <= f.size))
                .ok_or_else(|| format!("a range of file {file} at {offset} past its file"))?;
            if head[20] == 0 {
                kept += len;
                continue;
            }
            if open[file].is_none() {
                open[file] = Some(store.sparse(f)?);
            }
            let out = open[file].as_ref().expect("opened above");
            let hash = receive(f, out, offset, len, &mut input, &mut buf)?;
            pending.push((file, Covered { offset, len, hash }));
            sent += len;
            unrecorded += len;
            if unrecorded >= RECORD {
                store.record(&files, &open, &mut pending)?;
                unrecorded = 0;
            }
        }
        store.record(&files, &open, &mut pending)
    })();
    drop(input);
    let answer = match &result {
        Ok(()) => b"done".to_vec(),
        Err(e) => e.clone().into_bytes(),
    };
    write_frame(s, &answer).map_err(fail)?;
    result.map_err(fail)?;
    let files = files
        .into_iter()
        .map(|f| {
            let p = store.path(&f.digest);
            (f, p)
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
    use super::{HAVE_FRAME, ReadRange, SourceFile, Store, fetch, serve};
    use crate::host::rank::RankList;
    use gguf::digest::Sha256;
    use std::path::PathBuf;
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bloomery-dist-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // Rank 0 sends each rank its ranges at their offsets, a run of short pieces (one cut along a
    // tensor's input axis) as one range with its gaps; a second run sends nothing the store
    // holds and sends again only a range whose kept bytes no longer hash to their record; a store
    // list longer than one frame goes in several; the sparse copy never takes the source's name,
    // and its coverage holds only what was received.
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
            "127.0.0.1:{port} 198.18.0.1\n127.0.0.2:1 198.18.0.2\n127.0.0.3:1 198.18.0.3\n"
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
                let held = Store::open(&stores[0])
                    .unwrap()
                    .covering(&file.digest)
                    .unwrap();
                assert!(!held.holds(0, 18) && held.holds(far(1) as u64, 5_000));
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
                let held = store.covering(&file.digest).unwrap();
                assert!(held.holds(at as u64 + 10, 100) && held.holds(300, 10));
                assert!(!held.holds(2_818, 1) && !store.covers(&file.digest, 1 << 20, 10).unwrap());
                assert!(!stores[r].join(&file.name).exists());
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    // A range whose bytes do not hash to rank 0's hash is refused, and nothing is recorded.
    #[test]
    fn a_range_that_does_not_hash_to_rank_zeros_is_refused() {
        let root = scratch("hash");
        let file = SourceFile {
            name: "f".into(),
            digest: [7; 32],
            size: 64,
        };
        let store = Store::open(&root).unwrap();
        let out = store.sparse(&file).unwrap();
        let bytes = [5u8; 16];
        let mut h = Sha256::new();
        h.update(&bytes);
        let good = h.finish();
        let mut bad = good;
        bad[0] ^= 1;
        let mut buf = [0u8; 8];
        for (hash, ok) in [(good, true), (bad, false)] {
            let mut wire = bytes.to_vec();
            wire.extend(hash);
            let got = super::receive(&file, &out, 8, 16, &mut &wire[..], &mut buf);
            assert_eq!(got.is_ok(), ok, "{hash:?}");
        }
        assert!(store.coverage(&file.digest).unwrap().is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
