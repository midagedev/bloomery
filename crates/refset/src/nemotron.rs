//! Last-position logits of Nemotron 3 Nano (`tools/ref/nemotron_ref.cpp`,
//! `just dump-ref-nemotron`). One prompt is one row, read by the names on
//! `# prompt columns`. `logits` is the last position only, `n_vocab` f32,
//! little-endian; `ids` is that prompt, one id a line.
//!
//! ```text
//! # model    <path>
//! # build    llama-tokenize <md5> libllama.so <md5>
//! # arch     nemotron_h
//! # prompt columns  name  n_ids  n_vocab  ids  logits
//! prompt  code5  5  131072  code5.ids  code5.logits.f32
//! # complete  4  0
//! ```
//!
//! [`LogitsSet::open`] refuses a set that did not finish, one dumped from
//! another file, and one written by another mainline build.

use crate::RefError;
use crate::columns::{Columns, Row};
use crate::family::{Family, Identity};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// One prompt of the set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The row's `name`.
    pub name: String,
    /// How many ids the prompt holds, BOS included when the file adds one.
    pub n_ids: u32,
    /// The logits row's width.
    pub n_vocab: u32,
    /// The ids file under the set's directory.
    pub ids_file: String,
    /// The logits file under the set's directory (`n_vocab` f32).
    pub logits_file: String,
}

/// A logits set's manifest, and once opened, its files checked.
#[derive(Debug, Clone)]
pub struct LogitsSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# model`.
    pub model: Option<String>,
    /// `# build`, spelled as [`crate::tokenizer::build_id`].
    pub build: Option<String>,
    /// `# arch`.
    pub arch: Option<String>,
    /// One row a prompt, in the manifest's order.
    pub prompts: Vec<Prompt>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nemotron: {}:{}", self.0.display(), self.1)
    }
}

impl LogitsSet {
    /// Parse `dir/MANIFEST.tsv`. A set without one is `Missing`, naming the
    /// recipe that writes it. A set without its trailer comes back unread,
    /// `complete` false. A row before its column line, of another width, with
    /// a count that does not parse, or naming a prompt twice, is `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<LogitsSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = LogitsSet {
            dir: dir.to_path_buf(),
            model: None,
            build: None,
            arch: None,
            prompts: Vec::new(),
            complete: false,
        };
        // A manifest without its trailer is a dump that did not finish: its rows are not read, so
        // the refusal is the one that says so.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let mut cols: Option<Columns> = None;
        let mut seen = HashSet::new();
        for (i, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# model" => set.model = Some(one(&f, &at, "# model\t<path>")?),
                "# build" => set.build = Some(one(&f, &at, "# build\t<id>")?),
                "# arch" => set.arch = Some(one(&f, &at, "# arch\t<name>")?),
                "# prompt columns" => {
                    cols = Some(Columns::new("prompt", f[1..].iter().copied()));
                }
                "# complete" => set.complete = true,
                h if h.starts_with('#') => {}
                "prompt" => {
                    let c = cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a prompt row before its `# prompt columns` line")
                    })?;
                    let prompt = prompt_row(&c.row(line, &at)?, &at)?;
                    if !seen.insert(prompt.name.clone()) {
                        return Err(RefError::malformed(
                            &at,
                            format!("{} is named twice", prompt.name),
                        ));
                    }
                    set.prompts.push(prompt);
                }
                other => {
                    return Err(RefError::malformed(
                        &at,
                        format!("unknown row kind {other:?}"),
                    ));
                }
            }
        }
        Ok(set)
    }

    /// The set at `dir` read and checked against `family`'s row.
    pub fn open(dir: &Path, family: &Family) -> Result<LogitsSet, RefError> {
        if !matches!(family.identity, Identity::NemotronLogits) {
            return Err(RefError::missing(
                dir,
                format!(
                    "nemotron: {} is not a last-position logits family",
                    family.name
                ),
            ));
        }
        let set = LogitsSet::read(dir, family)?;
        set.check_family(family)?;
        Ok(set)
    }

    /// This set against its family's row: the trailer ([`RefError::Unfinished`]),
    /// the model file ([`RefError::Stale`]), the build and the architecture
    /// ([`RefError::Foreign`]), then each prompt's ids and logits files.
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        if self.prompts.is_empty() {
            return Err(RefError::malformed(
                self.dir.join("MANIFEST.tsv").display(),
                "no prompt rows: the set names nothing to check",
            ));
        }
        for p in &self.prompts {
            self.check_prompt(p)?;
        }
        Ok(())
    }

    /// Prompt `p`'s last-position logits, `n_vocab` f32, little-endian.
    pub fn logits(&self, p: &Prompt) -> Result<Vec<f32>, RefError> {
        let path = self.dir.join(&p.logits_file);
        let bytes = std::fs::read(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("nemotron: cannot read {}: {e}", path.display()),
            )
        })?;
        let want = p.n_vocab as usize * 4;
        if bytes.len() != want {
            return Err(RefError::malformed(
                path.display(),
                format!(
                    "{} is {} bytes, {} logits of {} want {want}",
                    p.logits_file,
                    bytes.len(),
                    p.name,
                    p.n_vocab
                ),
            ));
        }
        let mut out = Vec::with_capacity(p.n_vocab as usize);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(
                chunk.try_into().expect("chunks_exact(4)"),
            ));
        }
        Ok(out)
    }

    fn check_prompt(&self, p: &Prompt) -> Result<(), RefError> {
        let ids = self.dir.join(&p.ids_file);
        let text = std::fs::read_to_string(&ids).map_err(|e| {
            RefError::missing(
                &ids,
                format!("nemotron: cannot read {}: {e}", ids.display()),
            )
        })?;
        let n = text.lines().filter(|l| !l.is_empty()).count();
        if n != p.n_ids as usize {
            return Err(RefError::malformed(
                ids.display(),
                format!("{} holds {n} ids, {} wants {}", p.ids_file, p.name, p.n_ids),
            ));
        }
        self.logits(p)?;
        Ok(())
    }
}

fn one(f: &[&str], at: &dyn fmt::Display, want: &str) -> Result<String, RefError> {
    match f {
        [_, v] if !v.is_empty() => Ok((*v).to_string()),
        _ => Err(RefError::malformed(at, format!("want {want}"))),
    }
}

fn file_name(name: &str, at: &dyn fmt::Display) -> Result<(), RefError> {
    if name.is_empty() || name.contains('/') || matches!(name, "." | "..") {
        return Err(RefError::malformed(
            at,
            format!("{name:?} does not name a file of the set's directory"),
        ));
    }
    Ok(())
}

fn prompt_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<Prompt, RefError> {
    let name = r.text("name")?;
    let ids_file = r.text("ids")?;
    let logits_file = r.text("logits")?;
    file_name(name, at)?;
    file_name(ids_file, at)?;
    file_name(logits_file, at)?;
    Ok(Prompt {
        name: name.to_string(),
        n_ids: r.parse("n_ids")?,
        n_vocab: r.parse("n_vocab")?,
        ids_file: ids_file.to_string(),
        logits_file: logits_file.to_string(),
    })
}
