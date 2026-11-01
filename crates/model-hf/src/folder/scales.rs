//! The block scales a checkpoint folder stores compressed (`csf`, `csf_32x2_v1`: [`hf::csf`]),
//! read back into one file of the scale store, so a load keeps an NVFP4 stack stored so in place
//! as it keeps a plain one ([`super::Origin::Stack`]): each expert's codes where the folder stores
//! them, its plain block scale in that file.
//!
//! The store is `$BLOOMERY_SCALES`, else `~/.cache/bloomery/scales`. Each generation has a file
//! named by the folder's canonical path and its key: a first page that holds its key once its
//! scales are written (the key's pending mark before), then the scales in the order the load asked for them.
//! The key is a digest of every scale's weight and bytes and of the size and modification time of
//! each of the folder's files those weights lie in, by canonical path. A folder's open makes the
//! file at its length and decodes nothing ([`Wanted::open`]); its scales are decoded once a load's
//! host will read them ([`Decoded::fill`]), so a load that holds every expert on its cards writes
//! no scale. A later open of the folder reads a file of its key or its pending mark as it is,
//! under the generation's lock, which the open holds until the header maps the file. A changed
//! key opens another path, so older headers can still reopen their generation. Older files stay
//! in the store while readers may hold their paths.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::UNIX_EPOCH;

use hf::modelopt::{self, CompressedScale};
use hf::quant_config::Placed;
use safetensors::Folder;

use super::Error;
use crate::part;

/// The file's first page: the key, once the scales after it are written.
const HEAD: usize = 4096;

/// The store's directory: `$BLOOMERY_SCALES` when set, else `~/.cache/bloomery/scales`; `None`
/// with neither that nor `HOME`.
pub(crate) fn store() -> Option<PathBuf> {
    std::env::var_os("BLOOMERY_SCALES")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| PathBuf::from(h).join(".cache/bloomery/scales"))
        })
}

/// FNV-1a over `bytes`.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The block scales one folder's open asks for, each at its place in the store's file.
pub(crate) struct Wanted {
    /// The store's file of the folder; `None` without a store, refused at the first scale asked.
    pub(super) path: Option<PathBuf>,
    /// Each scale's weight, how to decode it and where in the file it lies.
    scales: Vec<(String, CompressedScale, u64)>,
    /// The folder's files those weights and their scales lie in.
    files: BTreeSet<PathBuf>,
    bytes: u64,
}

impl Wanted {
    /// No scale yet of the folder `dir`, whose file lies in `store`.
    pub(crate) fn new(dir: &Path, store: Option<&Path>) -> Result<Wanted, Error> {
        let path = match store {
            Some(store) => {
                let dir = dir
                    .canonicalize()
                    .map_err(|e| format!("{}: {e}", dir.display()))?;
                let name = digest(dir.as_os_str().as_encoded_bytes());
                Some(store.join(format!("{name:016x}.scales")))
            }
            None => None,
        };
        Ok(Wanted {
            path,
            scales: Vec::new(),
            files: BTreeSet::new(),
            bytes: 0,
        })
    }

    /// The tensor scale of `weight`, NVFP4 rows placed as `placed` with its block scale stored
    /// compressed, and where its plain block scale lies: the store's file and the bytes in it.
    pub(crate) fn push(
        &mut self,
        folder: &Folder,
        weight: &str,
        placed: &Placed,
    ) -> Result<(f32, PathBuf, std::ops::Range<u64>), Error> {
        let path = self.path.clone().ok_or_else(|| {
            format!(
                "{weight}: its block scale is stored compressed and a load reads it back into the \
                 scale store, which has no directory (BLOOMERY_SCALES or HOME)"
            )
        })?;
        let (scale, compressed) =
            modelopt::nvfp4_compressed_rows_scale(&|n: &str| folder.raw(n), weight, placed)?;
        let at = HEAD as u64 + self.bytes;
        let end = at
            .checked_add(compressed.bytes())
            .ok_or_else(|| format!("{weight}: the scale store's file passes u64"))?;
        for name in std::iter::once(weight).chain(placed.companions.iter().map(String::as_str)) {
            let file = folder.file_range(name)?.0;
            let file = file
                .canonicalize()
                .map_err(|e| format!("{}: {e}", file.display()))?;
            self.files.insert(file);
        }
        self.bytes = end - HEAD as u64;
        self.scales.push((weight.to_string(), compressed, at));
        Ok((scale, path, at..end))
    }

    /// The store's file of the scales asked for, at its length, the scales not yet decoded unless
    /// the file holds them under the key already ([`Decoded::fill`]); `None` when none was asked.
    /// A file of another length, or whose head is neither the key nor the key's pending mark, is
    /// replaced by one under the pending mark and no scale. The store's lock comes back with the
    /// file: its holder maps the generation's path before another process may replace it.
    pub(crate) fn open(self) -> Result<Option<(Decoded, File)>, Error> {
        if self.scales.is_empty() {
            return Ok(None);
        }
        let key = self.key()?;
        let mut pending = key.clone();
        pending[0] = b'!';
        let base = self.path.ok_or_else(|| {
            "the scale store has no directory (BLOOMERY_SCALES or HOME)".to_string()
        })?;
        let path = base.with_extension(format!("{:016x}.scales", digest(&key)));
        let len = HEAD as u64 + self.bytes;
        let io = |what: &str, e: std::io::Error| {
            format!(
                "scale store {} (BLOOMERY_SCALES): {what}: {e}",
                path.display()
            )
        };
        let dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| io("create", e))?;
        let lock = lock(&path).map_err(|e| io("lock", e))?;
        let open = || OpenOptions::new().read(true).write(true).open(&path);
        let kept = match open() {
            Ok(f) => {
                let head = read_head(&f).map_err(|e| io("read", e))?;
                let ours = head == key || head == pending;
                (f.metadata().map_err(|e| io("stat", e))?.len() == len && ours).then_some(f)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(io("open", e).into()),
        };
        let file = match kept {
            Some(f) => f,
            None => {
                let tmp = path.with_extension(format!("tmp{}", std::process::id()));
                File::create(&tmp)
                    .and_then(|f| {
                        f.set_len(len)?;
                        f.write_all_at(&pending, 0)
                    })
                    .map_err(|e| io("make", e))?;
                std::fs::rename(&tmp, &path).map_err(|e| io("rename", e))?;
                open().map_err(|e| io("open", e))?
            }
        };
        let decoded = Decoded {
            path,
            key,
            scales: self.scales,
            file: Mutex::new(file),
        };
        Ok(Some((decoded, lock)))
    }

    /// The key of the scales asked for: the digest of each one's weight, place in the file and
    /// bytes, and of the size and modification time of each folder file its weight lies in, by
    /// its canonical path.
    fn key(&self) -> Result<Vec<u8>, Error> {
        let mut text = String::new();
        for (weight, compressed, at) in &self.scales {
            text.push_str(&format!("{weight} {at} {}\n", compressed.bytes()));
        }
        for file in &self.files {
            let named = |e: &dyn std::fmt::Display| format!("{}: {e}", file.display());
            let meta = std::fs::metadata(file).map_err(|e| named(&e))?;
            let modified = meta
                .modified()
                .map_err(|e| named(&e))?
                .duration_since(UNIX_EPOCH)
                .map_err(|e| named(&e))?
                .as_nanos();
            text.push_str(&format!("{} {} {modified}\n", file.display(), meta.len()));
        }
        Ok(format!(
            "bloomery block scales\nkey {:016x}\nbytes {}\n",
            digest(text.as_bytes()),
            self.bytes
        )
        .into_bytes())
    }
}

/// The store's lock beside `path`, held exclusive until dropped: one process at a time checks,
/// replaces or fills the file.
fn lock(path: &Path) -> std::io::Result<File> {
    let lock = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path.with_extension("lock"))?;
    lock.lock()?;
    Ok(lock)
}

/// The head of the store's file `f` less its trailing zeros: empty while its scales are unwritten.
fn read_head(f: &File) -> std::io::Result<Vec<u8>> {
    let mut head = vec![0; HEAD];
    let mut got = 0;
    while got < head.len() {
        match f.read_at(&mut head[got..], got as u64)? {
            0 => break,
            n => got += n,
        }
    }
    let end = head.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    head.truncate(end);
    Ok(head)
}

/// A folder's file of decoded block scales in the store ([`Wanted::open`]).
pub(crate) struct Decoded {
    path: PathBuf,
    key: Vec<u8>,
    scales: Vec<(String, CompressedScale, u64)>,
    file: Mutex<File>,
}

impl Decoded {
    /// The store's file, which a located header reads the scales from.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Decode every scale into the file, read from `folder`, and write the key last, unless the
    /// file holds them under it already. Each lane of the host's parallelism decodes every lane-th
    /// scale.
    pub(crate) fn fill(&self, folder: &Folder) -> Result<(), part::Error> {
        let file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        let io = |what: &str, e: std::io::Error| {
            part::Error::Scales(format!(
                "scale store {} (BLOOMERY_SCALES): {what}: {e}",
                self.path.display()
            ))
        };
        let _lock = lock(&self.path).map_err(|e| io("lock", e))?;
        if read_head(&file).map_err(|e| io("read", e))? == self.key {
            return Ok(());
        }
        let lanes = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(self.scales.len());
        let decode = |lane: usize| -> Result<(), part::Error> {
            for (weight, compressed, at) in self.scales.iter().skip(lane).step_by(lanes) {
                let bytes = compressed.decode(|n| folder.raw(n))?;
                file.write_all_at(&bytes, *at)
                    .map_err(|e| io(&format!("write {weight}"), e))?;
            }
            Ok(())
        };
        std::thread::scope(|scope| {
            let others = (1..lanes)
                .map(|lane| {
                    std::thread::Builder::new()
                        .name("scale-decode".to_owned())
                        .spawn_scoped(scope, move || {
                            threads::helper::float_caller()?;
                            decode(lane)
                        })
                        .map_err(|e| io("spawn", e))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let own = decode(0);
            for lane in others {
                lane.join()
                    .unwrap_or_else(|e| std::panic::resume_unwind(e))?;
            }
            own
        })?;
        file.sync_data().map_err(|e| io("sync", e))?;
        file.write_all_at(&self.key, 0)
            .and_then(|()| file.sync_data())
            .map_err(|e| io("write the key", e))
    }
}
