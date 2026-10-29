//! A fixture's source read from a Hugging Face repo by HTTP Range: the bytes of a fixture are
//! a function of the seed and the source's headers alone (the module header), so a source
//! file is a sparse file of its listed size whose first bytes are the header, and the data
//! region costs no disk.
//!
//! [`sparse_shard`] reads one shard's header through a range reader: a first range, grown
//! until the header parses (`gguf` reports where its bytes ran out) and the data base is in
//! hand, then written at the front of a file extended to the listing's size. The file is
//! held to the listing: a header whose tensors end past that size, or more than an alignment
//! before it, is [`RemoteError::Size`]. [`sparse_set`] does that for each file of a listed
//! set through the hf crate's client, the one HTTP path. The files of a set are read up to
//! [`PARALLEL`] at a time, and a revision that is a commit leaves its sparse files in place for the
//! next read: a header already there at the listed size, which opens as the listing says, is not
//! read again.

use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use gguf::{Gguf, LoadError};
use hf::fetch::Client;
use hf::{Entry, HfError};

/// The bytes the first range of a header read asks for; a tokenizer's arrays make a header
/// several MiB, so a second request is the common case for a first shard.
pub const FIRST_RANGE: u64 = 4 << 20;

/// The most a header may be: a length past it is a corrupt file, not a header to read.
pub const MAX_HEADER: u64 = 256 << 20;

/// Reads `len` bytes of a remote file from byte `from`, exactly.
pub type ReadRange<'a> = dyn FnMut(u64, u64) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> + 'a;

#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    #[error(transparent)]
    Hf(#[from] HfError),
    #[error("fixture fetch reads the hub's listing; HF_HUB_OFFLINE is set")]
    Offline,
    #[error("{file}: reading {len} bytes at {from}: {source}")]
    Read {
        file: String,
        from: u64,
        len: u64,
        source: Box<dyn Error + Send + Sync>,
    },
    #[error("{file}: a range of {len} bytes at {from} came back as {got} bytes")]
    ShortRead {
        file: String,
        from: u64,
        len: u64,
        got: u64,
    },
    #[error("{path}: {op}: {source}")]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{file}: the header does not parse: {source}")]
    Header { file: String, source: LoadError },
    #[error("{file}: the header runs past the file's {size} bytes")]
    PastEnd { file: String, size: u64 },
    #[error("{file}: the header is longer than {MAX_HEADER} bytes")]
    HeaderTooLong { file: String },
    #[error(
        "{file}: the listing says {listed} bytes, and tensor {tensor} of its header ends at {end}"
    )]
    ListedShort {
        file: String,
        listed: u64,
        tensor: String,
        end: u64,
    },
    #[error(
        "{file}: the listing says {listed} bytes, and the header's tensors end at {end} \
         (alignment {align})"
    )]
    ListedLong {
        file: String,
        listed: u64,
        end: u64,
        align: u64,
    },
}

/// What one shard's read took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderStat {
    /// The shard's header bytes: everything before its data base.
    pub header: u64,
    /// Range requests made.
    pub reads: u32,
}

fn io_err(path: &Path, op: &'static str, source: io::Error) -> RemoteError {
    RemoteError::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
}

/// `file`'s header, `size` bytes in all, written to `dest` as a sparse file of `size` bytes
/// whose first bytes are the header; `read` is the range reader of the remote file. The
/// header is read in growing prefixes: each is written to `dest` and opened by `gguf`, whose
/// answer says what is missing — a read that ran out (`Truncated`: at least that many more
/// bytes, and at least double), or a header that parsed but whose tensors lie past the
/// prefix (`OutOfBounds` carries the data base). The data base's alignment padding is part
/// of the header; it is read too. A file `size` long that is not a GGUF is
/// [`RemoteError::Header`], a header that does not end inside the file
/// [`RemoteError::PastEnd`].
pub fn sparse_shard(
    dest: &Path,
    file: &str,
    size: u64,
    first: u64,
    read: &mut ReadRange<'_>,
) -> Result<HeaderStat, RemoteError> {
    let mut have: Vec<u8> = Vec::new();
    let mut reads = 0u32;
    let mut fetch = |have: &mut Vec<u8>, to: u64| -> Result<(), RemoteError> {
        let (from, len) = (have.len() as u64, to - have.len() as u64);
        let bytes = read(from, len).map_err(|source| RemoteError::Read {
            file: file.to_string(),
            from,
            len,
            source,
        })?;
        if bytes.len() as u64 != len {
            return Err(RemoteError::ShortRead {
                file: file.to_string(),
                from,
                len,
                got: bytes.len() as u64,
            });
        }
        have.extend_from_slice(&bytes);
        reads += 1;
        Ok(())
    };
    let mut want = first.min(size);
    let base = loop {
        if (have.len() as u64) < want {
            fetch(&mut have, want)?;
        }
        fs::write(dest, &have).map_err(|e| io_err(dest, "write the prefix", e))?;
        match Gguf::open(dest) {
            Ok(g) => break g.data_base(),
            Err(LoadError::OutOfBounds { base, .. }) => break base,
            Err(LoadError::Truncated { at, need, .. }) => {
                if want >= size {
                    return Err(RemoteError::PastEnd {
                        file: file.to_string(),
                        size,
                    });
                }
                let next = want.saturating_mul(2).max(at.saturating_add(need));
                if next > MAX_HEADER {
                    return Err(RemoteError::HeaderTooLong {
                        file: file.to_string(),
                    });
                }
                want = next.min(size);
            }
            Err(source) => {
                return Err(RemoteError::Header {
                    file: file.to_string(),
                    source,
                });
            }
        }
    };
    // A file with no tensor may end inside the padding; the header is what the file has.
    let header = base.min(size);
    if header > MAX_HEADER {
        return Err(RemoteError::HeaderTooLong {
            file: file.to_string(),
        });
    }
    if (have.len() as u64) < header {
        fetch(&mut have, header)?;
    }
    have.truncate(header as usize);
    fs::write(dest, &have).map_err(|e| io_err(dest, "write the header", e))?;
    OpenOptions::new()
        .write(true)
        .open(dest)
        .and_then(|f| f.set_len(size))
        .map_err(|e| io_err(dest, "extend to the listed size", e))?;
    check_size(dest, file, size)?;
    Ok(HeaderStat { header, reads })
}

/// The header of the GGUF file `file` (`size` bytes), read through `read` into a sparse file at
/// `dest` ([`sparse_shard`]) and removed after. `dest` is created here, so a file or link already
/// at that path is refused before any range is read, never written through.
pub fn shard_header(
    dest: &Path,
    file: &str,
    size: u64,
    read: &mut ReadRange<'_>,
) -> Result<gguf::Inventory, RemoteError> {
    fs::File::create_new(dest).map_err(|e| io_err(dest, "create the header file", e))?;
    let header = sparse_shard(dest, file, size, FIRST_RANGE, read).and_then(|_| {
        gguf::inventory_of(dest).map_err(|source| RemoteError::Header {
            file: file.to_string(),
            source,
        })
    });
    let removed = fs::remove_file(dest).map_err(|e| io_err(dest, "remove the header file", e));
    let header = header?;
    removed?;
    Ok(header)
}

/// The sparse file at `dest` opens, and the listing's `size` is what its header says: the
/// last tensor ends at most an alignment's padding before it.
fn check_size(dest: &Path, file: &str, size: u64) -> Result<(), RemoteError> {
    let g = match Gguf::open(dest) {
        Ok(g) => g,
        Err(source) => {
            // A tensor whose end passes u64 has no end to report: it keeps the reader's refusal.
            let short = match &source {
                LoadError::OutOfBounds {
                    name,
                    base,
                    off,
                    nbytes,
                    ..
                } => base
                    .checked_add(*off)
                    .and_then(|e| e.checked_add(*nbytes))
                    .map(|end| (name.clone(), end)),
                _ => None,
            };
            return Err(match short {
                Some((tensor, end)) => RemoteError::ListedShort {
                    file: file.to_string(),
                    listed: size,
                    tensor,
                    end,
                },
                None => RemoteError::Header {
                    file: file.to_string(),
                    source,
                },
            });
        }
    };
    let base = g.data_base();
    // A file with no tensor may end inside its header's padding: the header is all it has.
    let end = g
        .iter_tensors()
        .map(|t| base + t.offset + t.nbytes)
        .max()
        .unwrap_or(size.min(base));
    if size
        .checked_sub(end)
        .is_none_or(|extra| extra >= g.alignment())
    {
        return Err(RemoteError::ListedLong {
            file: file.to_string(),
            listed: size,
            end,
            align: g.alignment(),
        });
    }
    Ok(())
}

/// The most files of a set whose headers are read at once.
pub const PARALLEL: usize = 8;

/// `work(i)` for each `i` below `n`, up to [`PARALLEL`] at a time, and `done(i, &answer)` as each
/// answer is complete, in the order of `i`. The answers in that order, or the first error in it:
/// once one is in hand no new `work` starts, and the ones running end first.
fn each<T: Send>(
    n: usize,
    work: &(dyn Fn(usize) -> Result<T, RemoteError> + Sync),
    done: &mut dyn FnMut(usize, &T),
) -> Result<Vec<T>, RemoteError> {
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<(usize, Result<T, RemoteError>)>();
    let mut waiting: Vec<Option<Result<T, RemoteError>>> = (0..n).map(|_| None).collect();
    let mut answers: Vec<T> = Vec::with_capacity(n);
    let mut failed: Option<RemoteError> = None;
    std::thread::scope(|scope| {
        for _ in 0..PARALLEL.min(n) {
            let tx = tx.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let answer = work(i);
                    if answer.is_err() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    if tx.send((i, answer)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (i, answer) in rx {
            waiting[i] = Some(answer);
            while failed.is_none()
                && let Some(answer) = waiting.get_mut(answers.len()).and_then(Option::take)
            {
                match answer {
                    Ok(v) => {
                        done(answers.len(), &v);
                        answers.push(v);
                    }
                    Err(e) => failed = Some(e),
                }
            }
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(answers),
    }
}

/// Whether `dest` is the sparse file an earlier read of a `size`-byte file left: a read writes the
/// header, then extends the file to the listed size, so a file of that size is one whole read.
fn left_whole(dest: &Path, size: u64) -> bool {
    fs::metadata(dest).is_ok_and(|m| m.len() == size)
}

/// [`sparse_shard`], or the header an earlier read of the file left at `dest` when `reuse` is set
/// and that header is still the listing's: its file is `size` bytes and its tensors end where
/// the listing says. The answer then reads no range.
fn sparse_shard_or_kept(
    dest: &Path,
    file: &str,
    size: u64,
    reuse: bool,
    read: &mut ReadRange<'_>,
) -> Result<HeaderStat, RemoteError> {
    if reuse
        && left_whole(dest, size)
        && check_size(dest, file, size).is_ok()
        && let Ok(g) = Gguf::open(dest)
    {
        return Ok(HeaderStat {
            header: g.data_base().min(size),
            reads: 0,
        });
    }
    sparse_shard(dest, file, size, FIRST_RANGE, read)
}

/// Each of `files` (entries of `repo`'s listing at revision `rev`) as a sparse file named by
/// its own file name under `dir`: the header read by range through `client`, the data region
/// a hole. The files are read up to [`PARALLEL`] at a time. Where `rev` is a commit, a file an
/// earlier read left whole in `dir` is kept ([`sparse_shard_or_kept`]); a branch or a tag may move,
/// so it is read again. Calls `progress` after each file, in the order of `files`. The paths, in
/// `files`' order.
pub fn sparse_set(
    client: &Client,
    repo: &str,
    rev: &str,
    files: &[Entry],
    dir: &Path,
    progress: &mut dyn FnMut(&Entry, &HeaderStat),
) -> Result<Vec<PathBuf>, RemoteError> {
    fs::create_dir_all(dir).map_err(|e| io_err(dir, "create the header directory", e))?;
    let reuse = hf::is_commit(rev);
    each(
        files.len(),
        &|i| {
            let e = &files[i];
            sparse_shard_or_kept(
                &dir.join(e.name()),
                &e.path,
                e.size,
                reuse,
                &mut |from, len| {
                    client
                        .fetch_range(repo, rev, &e.path, from, len)
                        .map_err(Into::into)
                },
            )
        },
        &mut |i, stat| progress(&files[i], stat),
    )?;
    Ok(files.iter().map(|e| dir.join(e.name())).collect())
}

/// The directory under the cache `root` that holds the header-only copies of `repo` at `rev`: a
/// start by name plans a candidate's file from its header, and a copy here costs the header's
/// bytes instead of the download. A `/` in `rev` (a branch such as `refs/pr/1`) is one path part.
pub fn headers_dir(root: &Path, repo: &str, rev: &str) -> PathBuf {
    root.join("headers").join(repo).join(rev.replace('/', "_"))
}

/// `len` bytes of `file` from byte `from`, exactly: the reader's own error, or a body of another
/// length, named with the file.
fn read_exactly(
    read: &mut ReadRange<'_>,
    file: &str,
    from: u64,
    len: u64,
) -> Result<Vec<u8>, RemoteError> {
    let got = read(from, len).map_err(|source| RemoteError::Read {
        file: file.to_owned(),
        from,
        len,
        source,
    })?;
    let have = got.len() as u64;
    if have != len {
        return Err(RemoteError::ShortRead {
            file: file.to_owned(),
            from,
            len,
            got: have,
        });
    }
    Ok(got)
}

/// `file`, a safetensors shard of `size` bytes, as a sparse file at `dest`: its header (the
/// eight-byte little-endian length and the JSON it counts) at the front of a file extended to
/// `size`, the tensor data a hole. The header is bound by [`MAX_HEADER`] and by the file.
pub fn sparse_safetensors(
    dest: &Path,
    file: &str,
    size: u64,
    read: &mut ReadRange<'_>,
) -> Result<HeaderStat, RemoteError> {
    if size < 8 {
        return Err(RemoteError::PastEnd {
            file: file.to_owned(),
            size,
        });
    }
    let mut word = [0u8; 8];
    word.copy_from_slice(&read_exactly(read, file, 0, 8)?);
    let json = u64::from_le_bytes(word);
    if json > MAX_HEADER {
        return Err(RemoteError::HeaderTooLong {
            file: file.to_owned(),
        });
    }
    let end = 8 + json;
    if end > size {
        return Err(RemoteError::PastEnd {
            file: file.to_owned(),
            size,
        });
    }
    let header = read_exactly(read, file, 0, end)?;
    fs::write(dest, &header).map_err(|e| io_err(dest, "write the header", e))?;
    OpenOptions::new()
        .write(true)
        .open(dest)
        .and_then(|f| f.set_len(size))
        .map_err(|e| io_err(dest, "extend to the listed size", e))?;
    Ok(HeaderStat {
        header: end,
        reads: 2,
    })
}

/// `file` of `size` bytes, a small file of a checkpoint (its config, tokenizer or template),
/// whole at `dest`.
pub fn whole_file(
    dest: &Path,
    file: &str,
    size: u64,
    read: &mut ReadRange<'_>,
) -> Result<(), RemoteError> {
    if size > MAX_HEADER {
        return Err(RemoteError::HeaderTooLong {
            file: file.to_owned(),
        });
    }
    let bytes = read_exactly(read, file, 0, size)?;
    fs::write(dest, bytes).map_err(|e| io_err(dest, "write", e))
}

/// A checkpoint folder's `files` (entries of `repo`'s listing at revision `rev`, its top-level
/// files: [`hf::checkpoint_files`]) as a header-only copy in `dir`: each shard's header read by
/// range into a sparse file of the listed size ([`sparse_safetensors`]), every other file whole.
/// The files are read up to [`PARALLEL`] at a time, and where `rev` is a commit a file an earlier
/// read left whole in `dir` is kept. Calls `progress` after each file, in the order of `files`.
/// `dir`, which a folder reader opens as the folder.
pub fn sparse_checkpoint(
    client: &Client,
    repo: &str,
    rev: &str,
    files: &[Entry],
    dir: &Path,
    progress: &mut dyn FnMut(&Entry),
) -> Result<PathBuf, RemoteError> {
    fs::create_dir_all(dir).map_err(|e| io_err(dir, "create the header folder", e))?;
    let reuse = hf::is_commit(rev);
    each(
        files.len(),
        &|i| {
            let e = &files[i];
            let dest = dir.join(e.name());
            if reuse && left_whole(&dest, e.size) {
                return Ok(());
            }
            let mut read = |from: u64, len: u64| -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
                client
                    .fetch_range(repo, rev, &e.path, from, len)
                    .map_err(Into::into)
            };
            if e.path.ends_with(".safetensors") {
                sparse_safetensors(&dest, &e.path, e.size, &mut read).map(drop)
            } else {
                whole_file(&dest, &e.path, e.size, &mut read)
            }
        },
        &mut |i, _| progress(&files[i]),
    )?;
    Ok(dir.to_path_buf())
}

/// A fixture written at `out` taken down: its directory, and its sidecar file at `sidecar` and
/// the sidecar's directory when that holds nothing else (it is never the directory of a file
/// the writer did not make).
pub fn remove_fixture(out: &Path, sidecar: Option<&Path>) -> Result<(), RemoteError> {
    fs::remove_dir_all(out).map_err(|e| io_err(out, "remove the fixture", e))?;
    if let Some(side) = sidecar {
        match fs::remove_file(side) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err(side, "remove the sidecar", e)),
        }
        if let Some(dir) = side.parent() {
            match fs::remove_dir(dir) {
                Ok(()) => {}
                // Something else lives there: not ours to remove.
                Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_err(dir, "remove the sidecar's directory", e)),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::write::{Layout, TensorDecl, Writer};
    use gguf::{GENERAL_ARCHITECTURE, Value};
    use std::cell::RefCell;

    use crate::test_support::temp_dir as scratch;

    /// A GGUF of `n_tokens` tokenizer strings (a header of about `n_tokens * 30` bytes) and
    /// `tensors` F32 tensors of 8 values each, whole in memory.
    fn gguf_bytes(n_tokens: usize, tensors: usize) -> Vec<u8> {
        let tokens: Vec<Value> = (0..n_tokens)
            .map(|i| Value::String(format!("token-{i:020}")))
            .collect();
        let kvs = vec![
            (
                GENERAL_ARCHITECTURE.to_string(),
                Value::String("llama".into()),
            ),
            ("tokenizer.ggml.tokens".to_string(), Value::Array(tokens)),
        ];
        let decls: Vec<TensorDecl> = (0..tensors)
            .map(|i| TensorDecl {
                name: format!("t{i}"),
                dims: vec![8],
                type_id: 0,
                nbytes: 32,
            })
            .collect();
        let layout = Layout::new(&kvs, decls).expect("layout");
        let mut w = Writer::new(Vec::new(), layout).expect("writer");
        for i in 0..tensors {
            w.tensor(&format!("t{i}"), &[i as u8 + 1; 32])
                .expect("tensor");
        }
        w.finish().expect("finish")
    }

    /// The header length of `file`, as `gguf` reads it from the whole file.
    fn data_base(file: &[u8], dir: &Path) -> u64 {
        let p = dir.join("whole.gguf");
        fs::write(&p, file).expect("whole file");
        Gguf::open(&p).expect("whole file opens").data_base()
    }

    /// A range reader over `file` that records each request.
    fn reader<'a>(
        file: &'a [u8],
        log: &'a RefCell<Vec<(u64, u64)>>,
    ) -> impl FnMut(u64, u64) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> + 'a {
        move |from, len| {
            log.borrow_mut().push((from, len));
            let end = from.checked_add(len).filter(|&e| e <= file.len() as u64);
            match end {
                Some(e) => Ok(file[from as usize..e as usize].to_vec()),
                None => Err(format!("range {from}+{len} is past the end").into()),
            }
        }
    }

    /// The header read creates its file: a link already at the path is refused before any range
    /// is read and its target keeps its bytes, and a free path gives the header and is removed.
    #[cfg(unix)]
    #[test]
    fn a_shard_header_is_never_written_through_an_existing_path() {
        let d = scratch("header");
        let file = gguf_bytes(4, 1);
        let log = RefCell::new(Vec::new());
        let mut read = reader(&file, &log);
        let target = d.join("target");
        fs::write(&target, b"kept").unwrap();
        let dest = d.join("header.gguf");
        std::os::unix::fs::symlink(&target, &dest).unwrap();
        let size = file.len() as u64;
        let e = shard_header(&dest, "m.gguf", size, &mut read)
            .map(drop)
            .unwrap_err();
        assert!(e.to_string().contains("create the header file"), "{e}");
        assert!(log.borrow().is_empty());
        assert_eq!(fs::read(&target).unwrap(), b"kept");
        fs::remove_file(&dest).unwrap();
        let header = shard_header(&dest, "m.gguf", size, &mut read).unwrap();
        assert_eq!(
            header.value(GENERAL_ARCHITECTURE).and_then(Value::as_str),
            Some("llama")
        );
        assert!(!dest.exists());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_header_longer_than_the_first_range_is_read_whole() {
        let d = scratch("long");
        let file = gguf_bytes(4000, 3);
        let base = data_base(&file, &d);
        assert!(
            base > 3 * 4096,
            "the header ({base}) must outrun the first range"
        );
        let log = RefCell::new(Vec::new());
        let dest = d.join("shard.gguf");
        let stat = sparse_shard(
            &dest,
            "m.gguf",
            file.len() as u64,
            4096,
            &mut reader(&file, &log),
        )
        .expect("sparse shard");
        assert_eq!(stat.header, base);
        assert!(stat.reads >= 2, "{:?}", log.borrow());
        assert_eq!(fs::metadata(&dest).expect("stat").len(), file.len() as u64);
        let got = fs::read(&dest).expect("read");
        assert_eq!(got[..base as usize], file[..base as usize]);
        assert!(
            got[base as usize..].iter().all(|&b| b == 0),
            "the data is a hole"
        );
        // Each request starts where the last ended, and the range at least doubles: the
        // requests are the doublings from the first range to the header, plus the padding.
        let log = log.borrow();
        assert!(log.windows(2).all(|w| w[1].0 == w[0].0 + w[0].1), "{log:?}");
        let doublings = base.div_ceil(4096).next_power_of_two().ilog2() as usize;
        assert!(
            log.len() <= doublings + 2,
            "{} requests: {log:?}",
            log.len()
        );
    }

    #[test]
    fn a_header_inside_the_first_range_is_one_request() {
        let d = scratch("short");
        let file = gguf_bytes(4, 2);
        let base = data_base(&file, &d);
        let log = RefCell::new(Vec::new());
        let stat = sparse_shard(
            &d.join("s.gguf"),
            "m.gguf",
            file.len() as u64,
            FIRST_RANGE,
            &mut reader(&file, &log),
        )
        .expect("sparse shard");
        // The file is smaller than the first range: one request for all of it.
        assert_eq!((stat.header, stat.reads), (base, 1));
        assert_eq!(*log.borrow(), [(0, file.len() as u64)]);
    }

    #[test]
    fn the_padding_of_the_data_base_is_part_of_the_header() {
        let d = scratch("padding");
        let file = gguf_bytes(40, 2);
        let base = data_base(&file, &d);
        // The first range ends one byte before the data base, past the last tensor-info byte.
        let log = RefCell::new(Vec::new());
        let stat = sparse_shard(
            &d.join("s.gguf"),
            "m.gguf",
            file.len() as u64,
            base - 1,
            &mut reader(&file, &log),
        )
        .expect("sparse shard");
        assert_eq!(stat.header, base);
        assert_eq!(
            fs::read(d.join("s.gguf")).expect("read")[..base as usize],
            file[..base as usize]
        );
    }

    #[test]
    fn the_sparse_set_opens_through_split_open() {
        let d = scratch("split");
        let mut files = Vec::new();
        for i in 0..2u16 {
            let kvs = vec![
                (
                    GENERAL_ARCHITECTURE.to_string(),
                    Value::String("llama".into()),
                ),
                ("split.no".to_string(), Value::U16(i)),
                ("split.count".to_string(), Value::U16(2)),
                ("split.tensors.count".to_string(), Value::I32(2)),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(
                        (0..3000)
                            .map(|n| Value::String(format!("tok-{n:016}")))
                            .collect(),
                    ),
                ),
            ];
            let decl = TensorDecl {
                name: format!("t{i}"),
                dims: vec![8],
                type_id: 0,
                nbytes: 32,
            };
            let mut w = Writer::new(Vec::new(), Layout::new(&kvs, vec![decl]).expect("layout"))
                .expect("writer");
            w.tensor(&format!("t{i}"), &[i as u8 + 1; 32])
                .expect("tensor");
            files.push(w.finish().expect("finish"));
        }
        let log = RefCell::new(Vec::new());
        for (i, file) in files.iter().enumerate() {
            let name = format!("m-{:05}-of-00002.gguf", i + 1);
            sparse_shard(
                &d.join(&name),
                &name,
                file.len() as u64,
                1024,
                &mut reader(file, &log),
            )
            .expect("sparse shard");
        }
        let split = gguf::Split::open(d.join("m-00001-of-00002.gguf")).expect("split opens");
        assert_eq!(split.shard_count(), 2);
        assert_eq!(split.tensor_count(), 2);
        assert_eq!(split.architecture(), Some("llama"));
        let (s, info) = split.find("t1").expect("t1");
        assert_eq!((s, info.nbytes), (1, 32));
        // Its header is the real one; its data is a hole of the listed size.
        let real = gguf::Gguf::open(d.join("m-00002-of-00002.gguf")).expect("shard");
        assert_eq!(real.mapping().len() as u64, files[1].len() as u64);
        assert!(real.data(info).expect("data").iter().all(|&b| b == 0));
    }

    /// A header an earlier read left whole at a commit is kept: the second read makes no request.
    /// A branch may move, so it is read again; a file left short of the listed size is read again
    /// too.
    #[test]
    fn a_header_left_whole_is_not_read_again() {
        let d = scratch("kept");
        let file = gguf_bytes(40, 3);
        let size = file.len() as u64;
        let dest = d.join("m.gguf");
        let log = RefCell::new(Vec::new());
        let read = |reuse: bool, size: u64, log: &RefCell<Vec<(u64, u64)>>| {
            sparse_shard_or_kept(&dest, "m.gguf", size, reuse, &mut reader(&file, log))
        };
        let first = read(true, size, &log).expect("first read");
        assert_eq!(first.reads, 1);
        assert_eq!(log.borrow().len(), 1);
        let again = read(true, size, &log).expect("kept");
        assert_eq!((again.header, again.reads), (first.header, 0));
        assert_eq!(log.borrow().len(), 1, "a kept header asks the hub nothing");
        read(false, size, &log).expect("a branch is read again");
        assert_eq!(log.borrow().len(), 2);
        // A file that stopped short of the listed size is not a whole read.
        fs::OpenOptions::new()
            .write(true)
            .open(&dest)
            .and_then(|f| f.set_len(first.header))
            .expect("cut");
        read(true, size, &log).expect("a short file is read again");
        assert_eq!(log.borrow().len(), 3);
    }

    /// Files are read in parallel, handed out in order, and the first failure in file order ends
    /// the read: none starts after it, and the answers before it are still handed over.
    #[test]
    fn files_are_read_in_parallel_and_answered_in_order() {
        use std::time::Duration;

        let (running, widest) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut handed = Vec::new();
        let got = each(
            12,
            &|i| {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                widest.fetch_max(now, Ordering::SeqCst);
                // The early files take longest: answers complete out of order.
                std::thread::sleep(Duration::from_millis(60 - 4 * i as u64));
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(i * 10)
            },
            &mut |i, v: &usize| handed.push((i, *v)),
        )
        .expect("no failure");
        assert_eq!(got, (0..12).map(|i| i * 10).collect::<Vec<_>>());
        assert_eq!(handed, (0..12).map(|i| (i, i * 10)).collect::<Vec<_>>());
        let widest = widest.load(Ordering::SeqCst);
        assert!((2..=PARALLEL).contains(&widest), "widest {widest}");

        let mut handed = Vec::new();
        let failing = each(
            6,
            &|i| match i {
                3 | 4 => Err(RemoteError::PastEnd {
                    file: format!("f{i}"),
                    size: 0,
                }),
                _ => Ok(i),
            },
            &mut |i, _: &usize| handed.push(i),
        )
        .expect_err("file 3 fails");
        assert!(
            matches!(&failing, RemoteError::PastEnd { file, .. } if file == "f3"),
            "{failing:?}"
        );
        assert_eq!(handed, [0, 1, 2]);
        assert_eq!(
            each(0, &|_| Ok(0usize), &mut |_, _| {}).expect("empty"),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn a_listing_that_disagrees_with_the_header_is_named() {
        let d = scratch("size");
        let file = gguf_bytes(40, 3);
        let log = RefCell::new(Vec::new());
        for (listed, what) in [
            (file.len() as u64 - 8, "shorter"),
            (file.len() as u64 + 64, "longer"),
        ] {
            let got = sparse_shard(
                &d.join("s.gguf"),
                "m.gguf",
                listed,
                FIRST_RANGE,
                &mut |from, len| {
                    // The reader serves the real bytes within the real file; a listing
                    // longer than the file reads zeros past its end.
                    let mut v = vec![0u8; len as usize];
                    let end = file.len().min((from + len) as usize);
                    if (from as usize) < end {
                        v[..end - from as usize].copy_from_slice(&file[from as usize..end]);
                    }
                    log.borrow_mut().push((from, len));
                    Ok(v)
                },
            );
            match (got, what) {
                (Err(RemoteError::ListedShort { listed: l, end, .. }), "shorter") => {
                    assert_eq!((l, end), (listed, file.len() as u64));
                }
                (
                    Err(RemoteError::ListedLong {
                        listed: l,
                        end,
                        align,
                        ..
                    }),
                    "longer",
                ) => {
                    assert_eq!((l, end, align), (listed, file.len() as u64, 32));
                }
                (other, _) => panic!("{what}: {other:?}"),
            }
        }
    }

    /// A tensor whose end passes u64 is the reader's own refusal, not a listing that is too short:
    /// its offset, the last field of the header, set to `u64::MAX`.
    #[test]
    fn a_tensor_end_past_u64_keeps_the_readers_refusal() {
        let d = scratch("wrap");
        let mut file = gguf_bytes(4, 1);
        let p = d.join("whole.gguf");
        fs::write(&p, &file).expect("whole file");
        let at = gguf::inventory_of(&p).expect("inventory").header_end as usize - 8;
        file[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let log = RefCell::new(Vec::new());
        match sparse_shard(
            &d.join("s.gguf"),
            "m.gguf",
            file.len() as u64,
            FIRST_RANGE,
            &mut reader(&file, &log),
        ) {
            Err(RemoteError::Header {
                source: LoadError::OutOfBounds { name, off, .. },
                ..
            }) => assert_eq!((name.as_str(), off), ("t0", u64::MAX)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_file_with_no_tensor_may_end_inside_its_padding() {
        let d = scratch("nopad");
        for n in 40..80 {
            let file = gguf_bytes(n, 0);
            let end = file.iter().rposition(|&b| b != 0).expect("a nonzero byte") + 1;
            if end == file.len() {
                continue;
            }
            // The listing's size is the header without its padding, as a shard-1 header of
            // no tensors is published.
            let cut = &file[..end];
            let log = RefCell::new(Vec::new());
            let stat = sparse_shard(
                &d.join("s.gguf"),
                "m.gguf",
                cut.len() as u64,
                FIRST_RANGE,
                &mut reader(cut, &log),
            )
            .expect("sparse shard");
            assert_eq!(stat.header, cut.len() as u64);
            assert_eq!(fs::read(d.join("s.gguf")).expect("read"), cut);
            return;
        }
        panic!("no header size left padding to cut");
    }

    #[test]
    fn a_removed_fixture_takes_its_sidecar_and_only_what_the_writer_made() {
        let d = scratch("remove");
        let (out, side_dir) = (d.join("fx"), d.join("fx-r8"));
        fs::create_dir_all(out.join("draft")).expect("out");
        fs::write(out.join("a.gguf"), b"x").expect("shard");
        fs::write(out.join("draft/d.gguf"), b"x").expect("draft");
        fs::create_dir_all(&side_dir).expect("sidecar dir");
        let side = side_dir.join("fx-r8.gguf");
        fs::write(&side, b"x").expect("sidecar");
        remove_fixture(&out, Some(&side)).expect("remove");
        assert!(!out.exists() && !side_dir.exists());
        // A sidecar directory that holds a file the writer did not make stays.
        fs::create_dir_all(&out).expect("out");
        fs::create_dir_all(&side_dir).expect("sidecar dir");
        fs::write(&side, b"x").expect("sidecar");
        fs::write(side_dir.join("other.gguf"), b"y").expect("other");
        remove_fixture(&out, Some(&side)).expect("remove");
        assert!(!out.exists() && !side.exists());
        assert!(side_dir.join("other.gguf").exists());
        // No sidecar: the directory alone; a missing sidecar file is not an error.
        fs::create_dir_all(&out).expect("out");
        remove_fixture(&out, None).expect("remove");
        assert!(!out.exists());
    }

    #[test]
    fn a_file_that_is_not_gguf_and_a_header_past_the_end_are_named() {
        let d = scratch("bad");
        let log = RefCell::new(Vec::new());
        let junk = vec![0x41u8; 5000];
        match sparse_shard(
            &d.join("j.gguf"),
            "j.gguf",
            junk.len() as u64,
            FIRST_RANGE,
            &mut reader(&junk, &log),
        ) {
            Err(RemoteError::Header { source, .. }) => {
                assert!(matches!(source, LoadError::BadMagic(_)), "{source}");
            }
            other => panic!("{other:?}"),
        }
        // A file cut inside its header: the listing says it ends there.
        let file = gguf_bytes(400, 1);
        let cut = &file[..2000];
        match sparse_shard(
            &d.join("c.gguf"),
            "c.gguf",
            cut.len() as u64,
            512,
            &mut reader(cut, &log),
        ) {
            Err(RemoteError::PastEnd { size, .. }) => assert_eq!(size, 2000),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_reader_error_and_a_short_body_are_named() {
        let d = scratch("read");
        let file = gguf_bytes(40, 1);
        match sparse_shard(
            &d.join("e.gguf"),
            "e.gguf",
            file.len() as u64,
            64,
            &mut |_, _| Err("the hub said no".into()),
        ) {
            Err(RemoteError::Read {
                from, len, source, ..
            }) => {
                assert_eq!((from, len), (0, 64));
                assert_eq!(source.to_string(), "the hub said no");
            }
            other => panic!("{other:?}"),
        }
        match sparse_shard(
            &d.join("s.gguf"),
            "s.gguf",
            file.len() as u64,
            64,
            &mut |_, len| Ok(vec![0; len as usize - 1]),
        ) {
            Err(RemoteError::ShortRead { len, got, .. }) => assert_eq!((len, got), (64, 63)),
            other => panic!("{other:?}"),
        }
    }

    /// A safetensors file: the eight-byte length, a JSON header and `data` bytes.
    fn safetensors_bytes(json: &str, data: usize) -> Vec<u8> {
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(json.as_bytes());
        out.extend(std::iter::repeat_n(0xAB, data));
        out
    }

    /// A reader over `bytes` that serves any range inside them.
    fn slice_reader(
        bytes: &[u8],
    ) -> impl FnMut(u64, u64) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> + '_ {
        move |from, len| {
            let (from, len) = (from as usize, len as usize);
            bytes
                .get(from..from + len)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| "a range past the file".into())
        }
    }

    /// A shard's copy has its header at the front and the listed size, so a reader that checks
    /// tensor offsets against the file length is satisfied while the data costs no disk; a header
    /// longer than the file, a length past the bound and a file under eight bytes are named.
    #[test]
    fn a_safetensors_shard_is_its_header_and_a_hole() {
        let d = scratch("st-shard");
        let json = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let file = safetensors_bytes(json, 8);
        let dest = d.join("m.safetensors");
        let stat = sparse_safetensors(
            &dest,
            "m.safetensors",
            file.len() as u64,
            &mut slice_reader(&file),
        )
        .expect("a shard");
        assert_eq!(stat.header, 8 + json.len() as u64);
        let copy = fs::read(&dest).expect("copy");
        assert_eq!(copy.len(), file.len(), "the listed size");
        assert_eq!(&copy[..8 + json.len()], &file[..8 + json.len()]);
        assert!(
            copy[8 + json.len()..].iter().all(|b| *b == 0),
            "the data region is a hole"
        );
        // A length that runs past the file: the listing's size does not hold the header.
        let cut = &file[..20];
        match sparse_safetensors(&d.join("c"), "c", 20, &mut slice_reader(cut)) {
            Err(RemoteError::PastEnd { size, .. }) => assert_eq!(size, 20),
            other => panic!("{other:?}"),
        }
        let huge = (MAX_HEADER + 1).to_le_bytes();
        match sparse_safetensors(&d.join("h"), "h", 64, &mut slice_reader(&huge)) {
            Err(RemoteError::HeaderTooLong { .. }) => {}
            other => panic!("{other:?}"),
        }
        match sparse_safetensors(&d.join("s"), "s", 4, &mut slice_reader(&[0; 4])) {
            Err(RemoteError::PastEnd { size, .. }) => assert_eq!(size, 4),
            other => panic!("{other:?}"),
        }
    }

    /// A small file is written whole, a body of another length is named, and the header copies of
    /// two revisions of a repo are two directories.
    #[test]
    fn a_config_is_whole_and_a_revision_has_its_own_header_directory() {
        let d = scratch("st-whole");
        let body = br#"{"model_type":"x"}"#;
        whole_file(
            &d.join("config.json"),
            "config.json",
            body.len() as u64,
            &mut slice_reader(body),
        )
        .expect("whole");
        assert_eq!(fs::read(d.join("config.json")).expect("read"), body);
        match whole_file(
            &d.join("short.json"),
            "short.json",
            body.len() as u64 + 3,
            &mut slice_reader(body),
        ) {
            Err(RemoteError::Read { .. } | RemoteError::ShortRead { .. }) => {}
            other => panic!("{other:?}"),
        }
        let root = Path::new("/cache");
        assert_eq!(
            headers_dir(root, "o/n", "abc123"),
            Path::new("/cache/headers/o/n/abc123")
        );
        assert_eq!(
            headers_dir(root, "o/n", "refs/pr/1"),
            Path::new("/cache/headers/o/n/refs_pr_1")
        );
        assert_ne!(
            headers_dir(root, "o/n", "main"),
            headers_dir(root, "o/n", "abc123")
        );
    }
}
