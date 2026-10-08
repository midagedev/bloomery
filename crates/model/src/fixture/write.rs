//! Writing a plan's files: into a temporary directory, each synced, renamed
//! into place once every file is complete.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::write::{Layout, Writer};
use gguf::{GgmlType, Split};

use super::plan::{FilePlan, plan};
use super::sidecar::{self, SidecarStat};
use super::spec::{FixtureSpec, Options};
use super::{FixtureError, io_err};
use crate::fileio;

/// One written tensor.
#[derive(Clone, Debug)]
pub struct TensorStat {
    pub name: String,
    pub ty: GgmlType,
    pub bytes: u64,
    pub file: String,
    /// Seconds generating its bytes, and writing them.
    pub gen_secs: f64,
    pub write_secs: f64,
}

/// What `generate` did.
#[derive(Clone, Debug)]
pub struct GenerateStats {
    pub out: PathBuf,
    pub tensors: usize,
    pub bytes: u64,
    pub file_bytes: u64,
    /// The card budget the files record: the caller's, or the family
    /// planner's choice.
    pub card_budget: u64,
    pub gen_secs: f64,
    pub write_secs: f64,
    pub sync_secs: f64,
    /// The r8 sidecar written beside a whole fixture of a family that has
    /// one ([`FixtureSpec::sidecar`]).
    pub sidecar: Option<SidecarStat>,
    pub secs: f64,
}

/// Rename directory `from` to `to`, refused when `to` exists.
fn rename_new(from: &Path, to: &Path) -> Result<(), FixtureError> {
    fileio::rename_noreplace(from, to).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => FixtureError::Exists {
            path: to.to_path_buf(),
        },
        _ => io_err(to, "rename", e),
    })
}

/// Write the fixture `spec` names of `source` (and of `draft`) into
/// directory `out`, which must not exist: the files go into
/// `<out>.tmp.<pid>`, each synced, and the directory is renamed to `out` once
/// every file is complete. A whole fixture of a family with an r8 sidecar then
/// gets it, from the files in place; a sidecar that cannot be written takes
/// the fixture down with it, so `out` is a complete fixture or nothing. On a
/// failure the temporary directory is removed.
pub fn generate(
    spec: &FixtureSpec,
    source: &Split,
    draft: Option<&Split>,
    out: &Path,
    opts: &Options,
    progress: &mut dyn FnMut(&TensorStat),
) -> Result<GenerateStats, FixtureError> {
    let t0 = Instant::now();
    if out.try_exists().map_err(|e| io_err(out, "stat", e))? {
        return Err(FixtureError::Exists {
            path: out.to_path_buf(),
        });
    }
    let plan = plan(spec, source, draft, opts)?;
    let card_budget = plan.card_budget;
    let mut files = plan.target.layouts()?;
    let mut sets = vec![(&plan.target, files.len())];
    if let Some(d) = &plan.draft {
        let l = d.layouts()?;
        sets.push((d, l.len()));
        files.extend(l);
    }
    let need: u64 = files.iter().map(|(_, l)| l.file_len()).sum();
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let free = fileio::free_bytes(parent).map_err(|e| io_err(parent, "statvfs", e))?;
    if free < need {
        return Err(FixtureError::Space {
            path: parent.to_path_buf(),
            need,
            free,
        });
    }
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    std::fs::create_dir(&tmp).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => FixtureError::Exists { path: tmp.clone() },
        _ => io_err(&tmp, "create the directory", e),
    })?;
    let written = write_files(&sets, files, &tmp, opts.seed, progress)
        .and_then(|stats| rename_new(&tmp, out).map(|()| stats))
        .and_then(|stats| {
            File::open(parent)
                .and_then(|d| d.sync_all())
                .map_err(|e| io_err(parent, "fsync", e))?;
            Ok(stats)
        });
    let w = match written {
        Ok(s) => s,
        Err(error) => {
            if let Err(cleanup) = std::fs::remove_dir_all(&tmp)
                && cleanup.kind() != io::ErrorKind::NotFound
            {
                return Err(FixtureError::Abandoned {
                    error: Box::new(error),
                    tmp,
                    cleanup,
                });
            }
            return Err(error);
        }
    };
    let sidecar = match (&spec.sidecar, &opts.tensors) {
        (Some(sc), None) => match sidecar::write(sc, &out.join(&plan.target.files[0])) {
            Ok(stat) => Some(stat),
            Err(error) => {
                if let Err(cleanup) = std::fs::remove_dir_all(out)
                    && cleanup.kind() != io::ErrorKind::NotFound
                {
                    return Err(FixtureError::Abandoned {
                        error: Box::new(error),
                        tmp: out.to_path_buf(),
                        cleanup,
                    });
                }
                return Err(error);
            }
        },
        _ => None,
    };
    Ok(GenerateStats {
        out: out.to_path_buf(),
        tensors: w.tensors,
        bytes: w.bytes,
        file_bytes: need,
        card_budget,
        gen_secs: w.gen_secs,
        write_secs: w.write_secs,
        sync_secs: w.sync_secs,
        sidecar,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// What [`write_files`] wrote: tensors, their bytes, and the seconds spent
/// generating, writing and syncing them.
#[derive(Clone, Copy, Debug, Default)]
struct Written {
    tensors: usize,
    bytes: u64,
    gen_secs: f64,
    write_secs: f64,
    sync_secs: f64,
}

/// Every file of `sets` (each with its count of `files`) into `dir`, one
/// buffer of the largest tensor's size reused for every tensor.
fn write_files(
    sets: &[(&FilePlan, usize)],
    files: Vec<(String, Layout)>,
    dir: &Path,
    seed: u64,
    progress: &mut dyn FnMut(&TensorStat),
) -> Result<Written, FixtureError> {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let largest = sets
        .iter()
        .flat_map(|(p, _)| p.tensors.iter())
        .map(|t| t.nbytes as usize)
        .max()
        .unwrap_or(0);
    let mut buf = vec![0u8; largest];
    let mut done = Written::default();
    let mut files = files.into_iter();
    for (plan, n) in sets {
        for range in plan.shards.iter().take(*n) {
            let (name, layout) = files.next().expect("one layout per shard");
            let path = dir.join(&name);
            if let Some(sub) = path.parent() {
                std::fs::create_dir_all(sub).map_err(|e| io_err(sub, "create the directory", e))?;
            }
            let file = File::create_new(&path).map_err(|e| io_err(&path, "create", e))?;
            let werr = |source| FixtureError::Write {
                path: path.clone(),
                source,
            };
            let mut w =
                Writer::new(BufWriter::with_capacity(1 << 20, file), layout).map_err(werr)?;
            for t in &plan.tensors[range.clone()] {
                let out = &mut buf[..t.nbytes as usize];
                let g = Instant::now();
                t.fill(seed, threads, out);
                let gen_secs = g.elapsed().as_secs_f64();
                let wt = Instant::now();
                w.tensor(&t.name, out).map_err(werr)?;
                let write_secs = wt.elapsed().as_secs_f64();
                done.tensors += 1;
                done.bytes += t.nbytes;
                done.gen_secs += gen_secs;
                done.write_secs += write_secs;
                progress(&TensorStat {
                    name: t.name.clone(),
                    ty: t.ty,
                    bytes: t.nbytes,
                    file: name.clone(),
                    gen_secs,
                    write_secs,
                });
            }
            let file = w
                .finish()
                .map_err(werr)?
                .into_inner()
                .map_err(|e| io_err(&path, "flush", e.into_error()))?;
            let s = Instant::now();
            file.sync_all().map_err(|e| io_err(&path, "fsync", e))?;
            done.sync_secs += s.elapsed().as_secs_f64();
        }
    }
    Ok(done)
}
