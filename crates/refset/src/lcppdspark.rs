//! llama.cpp's DSpark draft calls: what a draft file proposes, call by call, on a fixed list of prompts, beside the target rows it
//! was fed (`tools/ref/dump-ml4-dspark.sh`, from a llama.cpp tree patched with `tools/ref/ml4-dspark-dump.patch`). The target
//! decodes each prompt greedily with the draft on; the patch writes the rows the draft consumed and each draft call.
//!
//! `MANIFEST.tsv` states the set's identity and names its prompts, read by the column names its header line gives:
//!
//! ```text
//! # model         <path>   the target's first shard
//! # draft_model   <path>
//! # build         <llama.cpp commit>+patch-<md5 of the patch>
//! # arch          mistral-large4
//! # n_max         3
//! # prompt columns  name  n_taps  n_calls
//! prompt  p00  18  4
//! # complete  32  0
//! ```
//!
//! A prompt row names a directory of the set holding three files: `taps.f32`, the `n_taps` rows the draft was fed (`feat` f32
//! each, little-endian: the target layers tapped times the target's width), `taps.pos`, each row's position (`n_taps` u32,
//! little-endian), and `calls.tsv`, one line per draft call (`n_calls` of them): `pos0`, `id_last` and the proposed ids,
//! comma-separated.
//!
//! [`DsparkSet::open`] refuses a set that did not finish, one dumped from another target or draft file, by another llama.cpp
//! tree or patch, or for another architecture, and one whose prompt files do not hold the sizes the manifest names;
//! [`DsparkSet::prompt`] reads a prompt's files whole.

use crate::RefError;
use crate::columns::{Columns, bytes32, file_name, parse_field};
use crate::family::{Family, Identity};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(test)]
pub(crate) mod testkit;
#[cfg(test)]
mod tests;

/// One `prompt` row: a directory of the set and the sizes its files must hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRow {
    /// The directory's name under the set's directory.
    pub name: String,
    /// How many rows the draft was fed: the length of `taps.pos`, and `taps.f32` holds that many rows of one width.
    pub n_taps: usize,
    /// How many draft calls `calls.tsv` holds.
    pub n_calls: usize,
}

/// One draft call of the dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    /// The position the call proposes from.
    pub pos0: u32,
    /// The token the call proposes after (`id_last`).
    pub last: u32,
    /// The ids the draft proposed, in order.
    pub ids: Vec<u32>,
}

/// One prompt's files, read whole.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub name: String,
    /// Each fed row's position.
    pub pos: Vec<u32>,
    /// The fed rows, `pos.len()` of the width the caller named, finite.
    pub taps: Vec<f32>,
    /// The draft calls, in order; at least one.
    pub calls: Vec<Call>,
}

/// A draft-calls set's manifest, and once opened, its prompt files checked for size.
#[derive(Debug, Clone)]
pub struct DsparkSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# model`: the target file the set was dumped from.
    pub model: Option<String>,
    /// `# draft_model`: the draft file.
    pub draft_model: Option<String>,
    /// `# build`: the llama.cpp commit and the patch.
    pub build: Option<String>,
    /// `# arch`: the target's architecture.
    pub arch: Option<String>,
    /// `# n_max`: the ids a call may propose.
    pub n_max: Option<usize>,
    /// One row a prompt, in the manifest's order.
    pub prompts: Vec<PromptRow>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A manifest or prompt-file line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lcppdspark: {}:{}", self.0.display(), self.1)
    }
}

impl DsparkSet {
    /// Parse `dir/MANIFEST.tsv`. A set without one is `Missing`, naming the recipe that writes it. A set without its trailer
    /// comes back unread, `complete` false. A header line twice, a row before its column line or of another width, a count
    /// that does not parse, a prompt named twice or by more than a directory's name, an unknown row kind, or a `# complete`
    /// count other than the prompt rows, is `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<DsparkSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = DsparkSet {
            dir: dir.to_path_buf(),
            model: None,
            draft_model: None,
            build: None,
            arch: None,
            n_max: None,
            prompts: Vec::new(),
            complete: false,
        };
        // A manifest without its trailer is a dump that did not finish: its rows are not read, so the refusal is the one that
        // says so.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let mut cols: Option<Columns> = None;
        let mut seen = HashSet::new();
        let mut complete_rows = None;
        for (i, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# model" => put(&mut set.model, &f, &at, "# model\t<path>")?,
                "# draft_model" => put(&mut set.draft_model, &f, &at, "# draft_model\t<path>")?,
                "# build" => put(&mut set.build, &f, &at, "# build\t<commit>+patch-<md5>")?,
                "# arch" => put(&mut set.arch, &f, &at, "# arch\t<name>")?,
                "# n_max" => {
                    let [_, v] = f[..] else {
                        return Err(RefError::malformed(&at, "want # n_max\t<count>"));
                    };
                    if set.n_max.replace(parse_field(v, "n_max", &at)?).is_some() {
                        return Err(RefError::malformed(&at, "a second # n_max line"));
                    }
                }
                "# prompt columns" => {
                    cols = Some(Columns::new("prompt", f[1..].iter().copied()));
                }
                "# complete" => {
                    if set.complete {
                        return Err(RefError::malformed(&at, "a second # complete line"));
                    }
                    if f.len() != 3 || f[2] != "0" {
                        return Err(RefError::malformed(&at, "want # complete\t<count>\t0"));
                    }
                    complete_rows = Some(parse_field(f[1], "complete count", &at)?);
                    set.complete = true;
                }
                h if h.starts_with('#') => {}
                "prompt" => {
                    let c = cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a prompt row before its `# prompt columns` line")
                    })?;
                    let row = c.row(line, &at)?;
                    let name = row.text("name")?;
                    file_name(name, &at)?;
                    if !seen.insert(name.to_string()) {
                        return Err(RefError::malformed(&at, format!("{name} is named twice")));
                    }
                    set.prompts.push(PromptRow {
                        name: name.to_string(),
                        n_taps: row.parse("n_taps")?,
                        n_calls: row.parse("n_calls")?,
                    });
                }
                other => {
                    return Err(RefError::malformed(
                        &at,
                        format!("unknown row kind {other:?}"),
                    ));
                }
            }
        }
        if complete_rows != Some(set.prompts.len()) {
            return Err(RefError::malformed(
                path.display(),
                "completion count differs from prompt rows",
            ));
        }
        Ok(set)
    }

    /// The set at `dir` read and checked against `family`'s row.
    pub fn open(dir: &Path, family: &Family) -> Result<DsparkSet, RefError> {
        if family.identity != Identity::LcppDsparkCalls {
            return Err(RefError::missing(
                dir,
                format!(
                    "lcppdspark: {} is not a llama.cpp draft-calls family",
                    family.name
                ),
            ));
        }
        let set = DsparkSet::read(dir, family)?;
        set.check_family(family)?;
        Ok(set)
    }

    /// This set against its family's row: the trailer ([`RefError::Unfinished`]), the target and the draft file it states
    /// ([`RefError::Stale`]), its `# arch` and `# build` ([`RefError::Foreign`]), then `# n_max`, the prompt rows, and each
    /// prompt's files by size ([`RefError::Malformed`]). The files are not read: [`DsparkSet::prompt`] does.
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_draft(&self.dir, self.draft_model.as_deref())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())?;
        let manifest = self.dir.join("MANIFEST.tsv");
        if self.n_max.is_none_or(|n| n == 0) {
            return Err(RefError::malformed(
                manifest.display(),
                "no `# n_max` line of a positive count: the set does not say how many ids a call proposes",
            ));
        }
        if self.prompts.is_empty() {
            return Err(RefError::malformed(
                manifest.display(),
                "no prompt rows: the set names nothing to check",
            ));
        }
        for p in &self.prompts {
            self.check_prompt(p)?;
        }
        Ok(())
    }

    /// Prompt `p`'s files against its row, by size: `taps.pos` holds `n_taps` positions, `taps.f32` whole rows of one width
    /// for `n_taps` rows, and `calls.tsv` the `n_calls` lines.
    fn check_prompt(&self, p: &PromptRow) -> Result<(), RefError> {
        let dir = self.dir.join(&p.name);
        if p.n_taps == 0 || p.n_calls == 0 {
            return Err(RefError::malformed(
                dir.display(),
                format!(
                    "{} names {} fed rows and {} calls: a prompt holds both",
                    p.name, p.n_taps, p.n_calls
                ),
            ));
        }
        let rows = p.n_taps as u64;
        let pos = file_len(&dir.join("taps.pos"))?;
        if pos != rows * 4 {
            return Err(RefError::malformed(
                dir.join("taps.pos").display(),
                format!("taps.pos is {pos} bytes, {} names {rows} positions", p.name),
            ));
        }
        let taps = file_len(&dir.join("taps.f32"))?;
        if taps == 0 || taps % (rows * 4) != 0 {
            return Err(RefError::malformed(
                dir.join("taps.f32").display(),
                format!(
                    "taps.f32 is {taps} bytes, not whole rows of f32 for the {rows} rows {} names",
                    p.name
                ),
            ));
        }
        let calls = self.calls(p)?.len();
        if calls != p.n_calls {
            return Err(RefError::malformed(
                dir.join("calls.tsv").display(),
                format!(
                    "calls.tsv holds {calls} calls, {} names {}",
                    p.name, p.n_calls
                ),
            ));
        }
        Ok(())
    }

    /// Prompt `p`'s draft calls: `calls.tsv`, one `pos0`, `id_last`, ids line a call. A line of another shape, an id that
    /// does not parse, or no call at all is `Malformed`.
    pub fn calls(&self, p: &PromptRow) -> Result<Vec<Call>, RefError> {
        let path = self.dir.join(&p.name).join("calls.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("lcppdspark: cannot read {}: {e}", path.display()),
            )
        })?;
        let mut calls = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            let [pos0, last, ids] = f[..] else {
                return Err(RefError::malformed(
                    &at,
                    format!("the line {line:?} is not pos0, id_last, ids"),
                ));
            };
            calls.push(Call {
                pos0: parse_field(pos0, "pos0", &at)?,
                last: parse_field(last, "id_last", &at)?,
                ids: ids
                    .split(',')
                    .map(|id| parse_field(id, "proposed id", &at))
                    .collect::<Result<_, _>>()?,
            });
        }
        if calls.is_empty() {
            return Err(RefError::malformed(path.display(), "no draft call"));
        }
        Ok(calls)
    }

    /// Prompt `p`'s files read whole: the positions, the fed rows as `feat` values each, and the calls. A file that is not whole
    /// 32-bit values, positions other than the row's `n_taps`, a row count other than the positions' at `feat`, or a value
    /// that is not finite is `Malformed`.
    pub fn prompt(&self, p: &PromptRow, feat: usize) -> Result<Prompt, RefError> {
        let dir = self.dir.join(&p.name);
        let pos: Vec<u32> = words(&dir.join("taps.pos"))?
            .into_iter()
            .map(u32::from_le_bytes)
            .collect();
        if pos.len() != p.n_taps {
            return Err(RefError::malformed(
                dir.join("taps.pos").display(),
                format!("{} positions, {} names {}", pos.len(), p.name, p.n_taps),
            ));
        }
        let taps_path = dir.join("taps.f32");
        let taps: Vec<f32> = words(&taps_path)?
            .into_iter()
            .map(f32::from_le_bytes)
            .collect();
        if taps.len() != pos.len() * feat {
            return Err(RefError::malformed(
                taps_path.display(),
                format!(
                    "{}: {} tapped values for {} positions of {feat}",
                    p.name,
                    taps.len(),
                    pos.len()
                ),
            ));
        }
        if let Some(i) = taps.iter().position(|v| !v.is_finite()) {
            return Err(RefError::malformed(
                taps_path.display(),
                format!("{}: non-finite tapped value at index {i}", p.name),
            ));
        }
        Ok(Prompt {
            name: p.name.clone(),
            pos,
            taps,
            calls: self.calls(p)?,
        })
    }
}

/// Header line `f` (`# key`, value) into `slot`; a line of another shape, an empty value or a second line of the key is
/// `Malformed`.
fn put(
    slot: &mut Option<String>,
    f: &[&str],
    at: &dyn fmt::Display,
    want: &str,
) -> Result<(), RefError> {
    let [key, v] = f else {
        return Err(RefError::malformed(at, format!("want {want}")));
    };
    if v.is_empty() {
        return Err(RefError::malformed(at, format!("want {want}")));
    }
    if slot.replace((*v).to_string()).is_some() {
        return Err(RefError::malformed(at, format!("a second {key} line")));
    }
    Ok(())
}

/// The length of the file at `path`; an unreadable one is `Missing`, naming it.
fn file_len(path: &Path) -> Result<u64, RefError> {
    std::fs::metadata(path).map(|m| m.len()).map_err(|e| {
        RefError::missing(
            path,
            format!("lcppdspark: cannot read {}: {e}", path.display()),
        )
    })
}

/// The 32-bit words of the file at `path`, as bytes; a partial last word is `Malformed`.
fn words(path: &Path) -> Result<Vec<[u8; 4]>, RefError> {
    Ok(bytes32(path)?.as_chunks::<4>().0.to_vec())
}
