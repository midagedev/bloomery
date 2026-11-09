//! A split written behind the download that is filling its folder (`prepare --follow`): the plan
//! is read from the config alone ([`FolderMap::stored`]) before any shard is there, each shard is
//! opened as its last byte lands and its header held to the config's table, and an output shard
//! is written the moment the input shards it reads are all there ([`drive`]), whatever order the
//! download finished them in.
//!
//! What is complete comes from a listing of the download, `path<TAB>bytes` per line, the paths
//! relative to the folder or under it: a file is complete when its length is its listed bytes or
//! a sidecar `<file>.done` exists. Waiting is bounded: when none of the listed files has grown for
//! the stall bound, the run is refused naming what it waits on.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use safetensors::{Dtype, Folder, Raw, Safetensors};

use crate::folder::{self, Error, FolderMap, Options, Planned, Staged, Stored, refuse};

/// Each shard's tensors as the config derives them: name, dtype and shape.
type Table = BTreeMap<String, Vec<(String, Dtype, Vec<usize>)>>;

/// How often a wait says what it waits on.
const SAY_EVERY: Duration = Duration::from_secs(60);

/// A download's listing: each file's path under the folder and the bytes it ends at.
pub struct List {
    files: Vec<(String, u64)>,
}

/// Where a listed file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Absent,
    /// This many of its bytes are there.
    Part(u64),
    Done,
}

impl List {
    /// The listing at `path`, its entries made relative to the folder `dir`. A line with no tab
    /// or bytes that are no count, an empty listing, and one file listed twice are refused.
    pub fn read(path: &Path, dir: &Path) -> Result<List, Error> {
        let at = |n: usize, what: &str| {
            Error::Refused(format!("{}: line {}: {what}", path.display(), n + 1))
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Refused(format!("{}: {e}", path.display())))?;
        let mut files: Vec<(String, u64)> = vec![];
        for (n, line) in text
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim().is_empty())
        {
            let (file, bytes) = line
                .split_once('\t')
                .ok_or_else(|| at(n, "no tab between the path and its bytes"))?;
            let bytes: u64 = bytes
                .trim()
                .parse()
                .map_err(|_| at(n, &format!("{:?} is no byte count", bytes.trim())))?;
            let rel = Path::new(file).strip_prefix(dir).unwrap_or(Path::new(file));
            let rel = rel
                .to_str()
                .filter(|r| !r.is_empty())
                .ok_or_else(|| at(n, "the path is empty or not UTF-8"))?
                .to_string();
            if files.iter().any(|(f, _)| *f == rel) {
                return Err(at(n, &format!("{rel} is listed twice")));
            }
            files.push((rel, bytes));
        }
        if files.is_empty() {
            return Err(Error::Refused(format!("{}: lists no file", path.display())));
        }
        Ok(List { files })
    }

    /// The bytes listed for `rel`, or its refusal by name.
    fn bytes_of(&self, rel: &str) -> Result<u64, Error> {
        self.files
            .iter()
            .find(|(f, _)| f == rel)
            .map(|&(_, b)| b)
            .ok_or_else(|| refuse(rel, "the listing names no size for it"))
    }

    /// `rel` as the folder `dir` holds it now. A file longer than its listed bytes is refused.
    fn state(&self, dir: &Path, rel: &str) -> Result<State, Error> {
        let want = self.bytes_of(rel)?;
        if dir.join(format!("{rel}.done")).exists() {
            return Ok(State::Done);
        }
        match std::fs::metadata(dir.join(rel)) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::Absent),
            Err(e) => Err(refuse(rel, format!("stat: {e}"))),
            Ok(m) if m.len() > want => Err(refuse(
                rel,
                format!("{} bytes, more than the {want} the listing says", m.len()),
            )),
            Ok(m) if m.len() == want => Ok(State::Done),
            Ok(m) => Ok(State::Part(m.len())),
        }
    }

    /// The bytes of the listed files there now, and the bytes listed in all.
    fn present(&self, dir: &Path) -> Result<(u64, u64), Error> {
        let (mut have, mut all) = (0u64, 0u64);
        for (rel, want) in &self.files {
            all += want;
            have += match self.state(dir, rel)? {
                State::Absent => 0,
                State::Part(n) => n,
                State::Done => *want,
            };
        }
        Ok((have, all))
    }
}

/// The follow of one folder: its listing, the config's table of what its shards hold, and the
/// input shards open so far.
pub struct Follow {
    dir: PathBuf,
    list: List,
    stall: Duration,
    poll: Duration,
    /// The listed bytes present when they last changed, and when.
    seen: (u64, Instant),
    said: Option<Instant>,
    note: Box<dyn FnMut(&str)>,
    /// The shard of each tensor the config derives.
    shard_of: BTreeMap<String, String>,
    table: Table,
    skips: fn(&str) -> Option<&'static str>,
    open: BTreeMap<String, Safetensors>,
}

impl Follow {
    /// The follow of folder `dir` by the listing at `list`: waits poll every `poll` and are
    /// refused when no listed byte has arrived for `stall`.
    pub fn new(dir: &Path, list: &Path, stall: Duration, poll: Duration) -> Result<Follow, Error> {
        let list = List::read(list, dir)?;
        let present = list.present(dir)?.0;
        Ok(Follow {
            dir: dir.to_path_buf(),
            list,
            stall,
            poll,
            seen: (present, Instant::now()),
            said: None,
            note: Box::new(|_| {}),
            shard_of: BTreeMap::new(),
            table: BTreeMap::new(),
            skips: |_| None,
            open: BTreeMap::new(),
        })
    }

    /// Say what a wait waits on through `note`, once a minute.
    pub fn say(&mut self, note: impl FnMut(&str) + 'static) {
        self.note = Box::new(note);
    }

    /// Sleep one poll; refused when the listed files have not grown for the stall bound.
    fn wait(&mut self, waiting: &BTreeSet<String>, what: &str) -> Result<(), Error> {
        std::thread::sleep(self.poll);
        let (have, all) = self.list.present(&self.dir)?;
        if have != self.seen.0 {
            self.seen = (have, Instant::now());
        }
        let names: Vec<&str> = waiting.iter().take(3).map(String::as_str).collect();
        let more = waiting.len().saturating_sub(names.len());
        let on = format!(
            "{}{}",
            names.join(", "),
            if more > 0 {
                format!(" and {more} more")
            } else {
                String::new()
            }
        );
        if self.seen.1.elapsed() >= self.stall {
            return Err(Error::Refused(format!(
                "no listed byte arrived for {:.0} s ({have} of {all} bytes are there); {what} \
                 waits on {on}",
                self.stall.as_secs_f64()
            )));
        }
        if self.said.is_none_or(|t| t.elapsed() >= SAY_EVERY) {
            self.said = Some(Instant::now());
            (self.note)(&format!(
                "follow: {what} waits on {on}; {have} of {all} listed bytes are there"
            ));
        }
        Ok(())
    }

    /// Wait until every file of `files` is complete.
    fn wait_for(&mut self, files: &[String], what: &str) -> Result<(), Error> {
        loop {
            let mut missing = BTreeSet::new();
            for f in files {
                if self.list.state(&self.dir, f)? != State::Done {
                    missing.insert(f.clone());
                }
            }
            if missing.is_empty() {
                return Ok(());
            }
            self.wait(&missing, what)?;
        }
    }

    /// The plan of the folder as the map `M` reads it, once its config, its index and the files
    /// beside its shards are there: the shard headers the config derives are written as sparse
    /// files under `shell` and read as a folder ([`folder::read_staged`]). Refused by name: a
    /// shard the listing gives no size, a tensor the index names that the config does not derive
    /// and the map does not skip, and the reverse.
    pub fn plan<M: FolderMap>(
        &mut self,
        shell: &Path,
        q8: bool,
        options: &Options,
    ) -> Result<Staged<M>, Error> {
        let (config, index) = ("config.json".to_string(), Folder::INDEX.to_string());
        self.wait_for(&[config.clone(), index.clone()], "the plan")?;
        let config_json = folder::read_json(&self.dir.join(&config))?;
        let Stored { tensors, skips } = M::parse(&config_json)?.stored().ok_or_else(|| {
            Error::Refused(
                "the map derives no table of the tensors a folder stores from its config; \
                 --follow plans from one"
                    .into(),
            )
        })?;
        self.skips = skips;
        let weight_map = Folder::weight_map(&self.dir.join(&index))?;
        (self.shard_of, self.table) = assign(tensors, skips, &weight_map)?;
        for shard in self.table.keys() {
            self.list.bytes_of(shard)?;
        }
        let shards: BTreeSet<&str> = weight_map.values().map(String::as_str).collect();
        let beside: Vec<String> = self
            .list
            .files
            .iter()
            .map(|(f, _)| f)
            .filter(|f| {
                !f.contains('/') && !shards.contains(f.as_str()) && **f != config && **f != index
            })
            .cloned()
            .collect();
        self.wait_for(&beside, "the plan")?;
        stage_shell(&self.dir, shell, &self.table, &self.shard_of)?;
        folder::read_staged::<M>(&self.dir, shell, q8, options)
    }

    /// The input shards the tensors of `shard` read.
    pub fn needs<S>(
        &self,
        plan: &hf::quant_config::Plan,
        shard: &[Planned<S>],
    ) -> Result<BTreeSet<String>, Error> {
        let mut out = BTreeSet::new();
        for p in shard {
            if let folder::Make::Tensor(m) = &p.make {
                for name in m.reads(plan)? {
                    let of = self.shard_of.get(name).ok_or_else(|| {
                        refuse(name, "the plan reads it; the config does not derive it")
                    })?;
                    out.insert(of.clone());
                }
            }
        }
        Ok(out)
    }

    /// Whether input shard `shard` is complete, opened and held to the config's table on the
    /// way: a tensor of the table the header lacks, one of another dtype or shape, and one the
    /// table and the map's skips do not name are refused by name.
    fn ready(&mut self, shard: &str) -> Result<bool, Error> {
        if self.open.contains_key(shard) {
            return Ok(true);
        }
        if self.list.state(&self.dir, shard)? != State::Done {
            return Ok(false);
        }
        let st =
            Safetensors::open(&self.dir.join(shard)).map_err(|e| refuse(shard, e.to_string()))?;
        let derived = self.table.get(shard).map_or(&[][..], Vec::as_slice);
        for (name, dtype, shape) in derived {
            let e = st.entry(name).ok_or_else(|| {
                refuse(
                    name,
                    format!("the config derives it in {shard}; its header does not hold it"),
                )
            })?;
            if e.dtype != *dtype || e.shape != *shape {
                return Err(refuse(
                    name,
                    format!(
                        "{shard} stores {:?} {:?}; the config derives {dtype:?} {shape:?}",
                        e.dtype, e.shape
                    ),
                ));
            }
        }
        if let Some(name) = st
            .names()
            .find(|n| !derived.iter().any(|d| d.0 == *n) && (self.skips)(n).is_none())
        {
            return Err(refuse(
                name,
                format!(
                    "{shard} holds it; the config derives no such tensor and the map does not skip it"
                ),
            ));
        }
        self.open.insert(shard.to_string(), st);
        Ok(true)
    }

    /// The tensor `name` from its open shard.
    pub fn raw(&self, name: &str) -> Result<Raw<'_>, safetensors::Error> {
        let missing = || safetensors::Error::Missing(name.to_string());
        let shard = self.shard_of.get(name).ok_or_else(missing)?;
        self.open.get(shard).ok_or_else(missing)?.raw(name)
    }

    /// Close every open input shard `keep` does not name.
    pub fn retain(&mut self, keep: &BTreeSet<String>) {
        self.open.retain(|s, _| keep.contains(s));
    }
}

/// The shard of each tensor the config derives, and each shard's tensors, from the index's
/// `weight_map`. Refused by name: an index name the config does not derive and the map does not
/// skip, and a derived tensor the index lacks.
fn assign(
    tensors: Vec<(String, Dtype, Vec<usize>)>,
    skips: fn(&str) -> Option<&'static str>,
    weight_map: &BTreeMap<String, String>,
) -> Result<(BTreeMap<String, String>, Table), Error> {
    let derived: BTreeSet<&str> = tensors.iter().map(|t| t.0.as_str()).collect();
    let mut shard_of = BTreeMap::new();
    for (name, shard) in weight_map {
        if derived.contains(name.as_str()) {
            shard_of.insert(name.clone(), shard.clone());
        } else if skips(name).is_none() {
            return Err(refuse(
                name,
                format!(
                    "the index puts it in {shard}; the config derives no such tensor and the \
                     map does not skip it"
                ),
            ));
        }
    }
    let mut table = Table::new();
    for (name, dtype, shape) in tensors {
        let shard = shard_of.get(&name).ok_or_else(|| {
            refuse(
                &name,
                "the config derives it and the index does not name it",
            )
        })?;
        table
            .entry(shard.clone())
            .or_default()
            .push((name, dtype, shape));
    }
    Ok((shard_of, table))
}

/// The shell folder at `shell`: the folder's config and the index and header of each shard the
/// table holds, each shard a sparse file of its header and its tensors' length, so the folder
/// opens as a whole and no byte of it is data.
fn stage_shell(
    dir: &Path,
    shell: &Path,
    table: &Table,
    shard_of: &BTreeMap<String, String>,
) -> Result<(), Error> {
    let io = |p: &Path, e: std::io::Error| Error::Refused(format!("{}: {e}", p.display()));
    if shell.exists() {
        std::fs::remove_dir_all(shell).map_err(|e| io(shell, e))?;
    }
    std::fs::create_dir_all(shell).map_err(|e| io(shell, e))?;
    for f in ["config.json", "hf_quant_config.json"] {
        if dir.join(f).exists() {
            std::fs::copy(dir.join(f), shell.join(f)).map_err(|e| io(&dir.join(f), e))?;
        }
    }
    let index = serde_json::json!({"metadata": {}, "weight_map": shard_of});
    let index_path = shell.join(Folder::INDEX);
    std::fs::write(&index_path, index.to_string()).map_err(|e| io(&index_path, e))?;
    for (shard, tensors) in table {
        let mut entries = vec![];
        for (name, dtype, shape) in tensors {
            let bytes = shape
                .iter()
                .try_fold(dtype.size(), |n, &d| n.checked_mul(d))
                .ok_or_else(|| refuse(name, "its stored bytes pass usize"))?;
            entries.push((name.as_str(), dtype.name(), shape.as_slice(), bytes));
        }
        let head = safetensors::header(&entries);
        let data: usize = entries.iter().map(|e| e.3).sum();
        let path = shell.join(shard);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        }
        std::fs::write(&path, &head).map_err(|e| io(&path, e))?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|e| io(&path, e))?;
        file.set_len((head.len() + data) as u64)
            .map_err(|e| io(&path, e))?;
    }
    Ok(())
}

/// Run `write` for each output shard once the input shards it needs are all complete, in the
/// order they become so, and `wait` when none is. `needs[no]` is the input shards output shard
/// `no` reads and `done[no]` whether it is already written; `write` also gets the input shards
/// the shards still to write read, to close the others.
pub fn drive<S>(
    state: &mut S,
    needs: &[BTreeSet<String>],
    mut done: Vec<bool>,
    ready: impl Fn(&mut S, &str) -> Result<bool, Error>,
    mut write: impl FnMut(&mut S, usize, &BTreeSet<String>) -> Result<(), Error>,
    mut wait: impl FnMut(&mut S, &BTreeSet<String>) -> Result<(), Error>,
) -> Result<(), Error> {
    assert_eq!(needs.len(), done.len(), "one need and one mark a shard");
    loop {
        let mut missing = BTreeSet::new();
        let mut next = None;
        for no in (0..needs.len()).filter(|&n| !done[n]) {
            let mut all = true;
            for shard in &needs[no] {
                if !ready(state, shard)? {
                    all = false;
                    missing.insert(shard.clone());
                }
            }
            if all {
                next = Some(no);
                break;
            }
        }
        match next {
            Some(no) => {
                done[no] = true;
                let keep: BTreeSet<String> = (0..needs.len())
                    .filter(|&n| !done[n])
                    .flat_map(|n| needs[n].iter().cloned())
                    .collect();
                write(state, no, &keep)?;
            }
            None if missing.is_empty() => return Ok(()),
            None => wait(state, &missing)?,
        }
    }
}

impl Follow {
    /// [`drive`] over this follow's input shards: an output shard is written when the shards
    /// it needs are complete, and a wait is bounded.
    pub fn run(
        &mut self,
        needs: &[BTreeSet<String>],
        done: Vec<bool>,
        mut write: impl FnMut(&Follow, usize) -> Result<(), Error>,
    ) -> Result<(), Error> {
        drive(
            self,
            needs,
            done,
            Follow::ready,
            |f, no, keep| {
                write(f, no)?;
                f.retain(keep);
                Ok(())
            },
            |f, missing| f.wait(missing, "the split"),
        )
    }
}

#[cfg(test)]
mod tests;
