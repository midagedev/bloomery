//! The V4.1 vision oracle's fork side (`tools/ref/vision/visref.sh`): smalinin's
//! llama.cpp fork fed each case's ids and image rows (`visref_fork.cpp`), its
//! greedy answer, and the logits row each of the answer's first `keep` tokens
//! was picked from. The engine is fed the same ids and rows and scored
//! against those rows position by position.
//!
//! `MANIFEST.tsv` holds `#` lines — the model file the fork ran (`# model`),
//! the fork's commit (`# build`), `# arch`, the vision set the image rows came
//! from and that set's checkpoint (`# rows`), the vocabulary and row widths,
//! `keep` — then `case` and `file` rows, each read by the names its
//! `# <kind> columns` line gives, and `# complete` last. A case's files are
//! `<name>.ids.i32` (its ids, the span's positions holding the image token),
//! for a case with a span `<name>.rows.bf16` (one `n_embd` row per span
//! position) and `<name>.types.i32` (the reference's token types: start 0,
//! image 1, newline 2, end 3), `<name>.answer.i32` and `<name>.logits.f32`
//! (`[logits, n_vocab]` f32).

use crate::RefError;
use crate::columns::{Columns, Row, parse_field};
use crate::family::{Family, Identity};
use std::fmt;
use std::path::{Path, PathBuf};

/// One `case` row: the ids fed, where the image span sits, what the fork
/// answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    pub name: String,
    /// `image`, or a control: `text` (the span removed), `prose` (the span's
    /// positions holding prose ids).
    pub kind: String,
    /// The vision set's image stem, `-` for a control.
    pub image: String,
    pub n_ids: usize,
    pub span_at: usize,
    /// 0 for a case without a span.
    pub span_len: usize,
    /// The greedy tokens the fork was allowed.
    pub gen_max: usize,
    /// The greedy tokens it produced, the last an end of generation when
    /// `eog`.
    pub answer: usize,
    pub eog: bool,
    /// The logits rows kept: `min(keep, answer)`.
    pub logits: usize,
    pub question: String,
}

/// One `file` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetFile {
    pub name: String,
    pub bytes: u64,
    pub md5: String,
}

/// A fork comparison set's manifest.
#[derive(Debug, Clone)]
pub struct VisrefSet {
    pub dir: PathBuf,
    /// `# model`: the first shard of the model file the fork ran.
    pub model: Option<String>,
    /// `# build`: the fork's commit.
    pub build: Option<String>,
    pub arch: Option<String>,
    /// `# rows`: the vision set the image rows came from, and its checkpoint
    /// (`<repo>@<revision>`).
    pub rows_set: Option<String>,
    pub rows_checkpoint: Option<String>,
    pub image_token_id: u32,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub keep: usize,
    pub cases: Vec<Case>,
    pub files: Vec<SetFile>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "visref: {}:{}", self.0.display(), self.1)
    }
}

impl VisrefSet {
    /// Parse `dir/MANIFEST.tsv`. `# image_token_id`, `# n_vocab`, `# n_embd`,
    /// `# keep` and at least one `case` row are required; a row before its
    /// column line, of another width, or with a field that does not parse is
    /// `Malformed`, and so is a case whose files the manifest does not list.
    pub fn read(dir: &Path) -> Result<VisrefSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: just dump-ref-visref", dir.display()),
            )
        })?;
        let mut set = VisrefSet {
            dir: dir.to_path_buf(),
            model: None,
            build: None,
            arch: None,
            rows_set: None,
            rows_checkpoint: None,
            image_token_id: 0,
            n_vocab: 0,
            n_embd: 0,
            keep: 0,
            cases: Vec::new(),
            files: Vec::new(),
            complete: false,
        };
        let (mut case_cols, mut file_cols) = (None, None);
        let mut need = ["# image_token_id", "# n_vocab", "# n_embd", "# keep"].map(|k| (k, false));
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            let field = |n: usize| f.get(n).map(|s| s.to_string());
            match f[0] {
                "# model" => set.model = field(1),
                "# build" => set.build = field(1),
                "# arch" => set.arch = field(1),
                "# rows" => {
                    set.rows_set = field(1);
                    set.rows_checkpoint = field(2);
                }
                "# image_token_id" => {
                    set.image_token_id =
                        parse_field(f.get(1).copied().unwrap_or(""), "image_token_id", &at)?
                }
                "# n_vocab" => {
                    set.n_vocab = parse_field(f.get(1).copied().unwrap_or(""), "n_vocab", &at)?
                }
                "# n_embd" => {
                    set.n_embd = parse_field(f.get(1).copied().unwrap_or(""), "n_embd", &at)?
                }
                "# keep" => set.keep = parse_field(f.get(1).copied().unwrap_or(""), "keep", &at)?,
                "# case columns" => case_cols = Some(columns("case", &f, &at)?),
                "# file columns" => file_cols = Some(columns("file", &f, &at)?),
                "# complete" => set.complete = true,
                "case" => set.cases.push(case_row(
                    &need_cols(&case_cols, "case", &at)?.row(line, &at)?,
                )?),
                "file" => {
                    let r = need_cols(&file_cols, "file", &at)?.row(line, &at)?;
                    set.files.push(SetFile {
                        name: r.text("name")?.to_string(),
                        bytes: r.parse("bytes")?,
                        md5: r.text("md5")?.to_string(),
                    });
                }
                h if h.starts_with('#') => {}
                "" => {}
                k => return Err(RefError::malformed(&at, format!("unknown row kind {k:?}"))),
            }
            if let Some(n) = need.iter_mut().find(|(k, _)| *k == f[0]) {
                n.1 = true;
            }
        }
        if let Some((k, _)) = need.iter().find(|(_, seen)| !seen) {
            return Err(RefError::missing(
                &path,
                format!("{}: no {k} line", path.display()),
            ));
        }
        if set.cases.is_empty() {
            return Err(RefError::malformed(path.display(), "no case rows"));
        }
        for c in &set.cases {
            if c.span_at + c.span_len > c.n_ids || c.logits != c.answer.min(set.keep) {
                return Err(RefError::malformed(
                    path.display(),
                    format!(
                        "case {}: span {}+{} of {} ids, {} logits rows of an answer of {} at keep {}",
                        c.name, c.span_at, c.span_len, c.n_ids, c.logits, c.answer, set.keep
                    ),
                ));
            }
            for (suffix, bytes) in set.expected(c) {
                let name = format!("{}.{suffix}", c.name);
                match set.files.iter().find(|f| f.name == name) {
                    Some(f) if f.bytes == bytes => {}
                    Some(f) => {
                        return Err(RefError::malformed(
                            path.display(),
                            format!(
                                "{name}: its file row says {} bytes, the case wants {bytes}",
                                f.bytes
                            ),
                        ));
                    }
                    None => {
                        return Err(RefError::malformed(
                            path.display(),
                            format!("case {} has no file row {name}", c.name),
                        ));
                    }
                }
            }
        }
        Ok(set)
    }

    /// Read the set at `dir` and check it against `family`'s row: the
    /// completion trailer ([`RefError::Unfinished`]), the model file
    /// ([`RefError::Stale`]), the fork build and the architecture
    /// ([`RefError::Foreign`]), and the checkpoint the image rows came from,
    /// which must be the revision the family pins ([`RefError::Stale`]).
    pub fn open(dir: &Path, family: &Family) -> Result<VisrefSet, RefError> {
        let Identity::ForkManifest { rows_revision } = family.identity else {
            return Err(RefError::missing(
                dir,
                format!("the {} family is not a fork comparison family", family.name),
            ));
        };
        let set = VisrefSet::read(dir)?;
        if !set.complete {
            return Err(RefError::Unfinished {
                set: dir.display().to_string(),
            });
        }
        family.check_file(dir, "# model", set.model.as_deref())?;
        family.check_build(dir, set.build.as_deref())?;
        family.check_arch(dir, set.arch.as_deref())?;
        let named = set
            .rows_checkpoint
            .as_deref()
            .and_then(|s| s.rsplit_once('@'))
            .map(|(_, r)| r);
        if named != Some(rows_revision) {
            return Err(RefError::Stale {
                set: dir.display().to_string(),
                dumped_from: set.rows_checkpoint.clone().map_or_else(
                    || {
                        "image rows of an unstated checkpoint (the set has no # rows line)"
                            .to_string()
                    },
                    |c| format!("image rows of {c}"),
                ),
                runs: rows_revision.to_string(),
            });
        }
        Ok(set)
    }

    /// The case named `name`.
    pub fn case(&self, name: &str) -> Result<&Case, RefError> {
        self.cases.iter().find(|c| c.name == name).ok_or_else(|| {
            RefError::missing(
                &self.dir,
                format!("{}: no case {name:?}", self.dir.display()),
            )
        })
    }

    /// The files case `c` has and the bytes each must hold.
    fn expected(&self, c: &Case) -> Vec<(&'static str, u64)> {
        let mut v = vec![
            ("ids.i32", 4 * c.n_ids as u64),
            ("answer.i32", 4 * c.answer as u64),
            ("logits.f32", 4 * (c.logits * self.n_vocab) as u64),
        ];
        if c.span_len > 0 {
            v.push(("rows.bf16", 2 * (c.span_len * self.n_embd) as u64));
            v.push(("types.i32", 4 * c.span_len as u64));
        }
        v
    }

    fn raw(&self, c: &Case, suffix: &str) -> Result<Vec<u8>, RefError> {
        let (_, want) = self
            .expected(c)
            .into_iter()
            .find(|(s, _)| *s == suffix)
            .ok_or_else(|| {
                RefError::missing(&self.dir, format!("case {} has no {suffix}", c.name))
            })?;
        let path = self.dir.join(format!("{}.{suffix}", c.name));
        let b = std::fs::read(&path)
            .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
        if b.len() as u64 != want {
            return Err(RefError::malformed(
                path.display(),
                format!("{} bytes, the manifest says {want}", b.len()),
            ));
        }
        Ok(b)
    }

    /// Case `c`'s ids.
    pub fn ids(&self, c: &Case) -> Result<Vec<u32>, RefError> {
        let b = self.raw(c, "ids.i32")?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect())
    }

    /// Case `c`'s fork answer.
    pub fn answer(&self, c: &Case) -> Result<Vec<u32>, RefError> {
        let b = self.raw(c, "answer.i32")?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect())
    }

    /// Case `c`'s span rows, `span_len × n_embd` bf16 bit patterns.
    pub fn rows(&self, c: &Case) -> Result<Vec<u16>, RefError> {
        let b = self.raw(c, "rows.bf16")?;
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|w| u16::from_le_bytes(*w))
            .collect())
    }

    /// Case `c`'s span token types (start 0, image 1, newline 2, end 3); any
    /// other value is `Malformed`.
    pub fn types(&self, c: &Case) -> Result<Vec<u8>, RefError> {
        let b = self.raw(c, "types.i32")?;
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| match i32::from_le_bytes(*w) {
                t @ 0..=3 => Ok(t as u8),
                t => Err(RefError::malformed(
                    format!("{}.types.i32", c.name),
                    format!("token type {t}, the span's are 0 to 3"),
                )),
            })
            .collect()
    }

    /// Case `c`'s fork logits, `logits × n_vocab` f32.
    pub fn logits(&self, c: &Case) -> Result<Vec<f32>, RefError> {
        let b = self.raw(c, "logits.f32")?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect())
    }
}

fn columns(kind: &str, f: &[&str], at: &dyn fmt::Display) -> Result<Columns, RefError> {
    let names = f
        .get(1)
        .ok_or_else(|| RefError::malformed(at, format!("# {kind} columns names none")))?;
    Ok(Columns::new(kind, names.split(' ')))
}

fn need_cols<'a>(
    cols: &'a Option<Columns>,
    what: &str,
    at: &dyn fmt::Display,
) -> Result<&'a Columns, RefError> {
    cols.as_ref()
        .ok_or_else(|| RefError::malformed(at, format!("a {what} row before its column line")))
}

fn case_row(r: &Row<'_>) -> Result<Case, RefError> {
    let eog: u8 = r.parse("eog")?;
    if eog > 1 {
        return Err(RefError::malformed(
            r.text("name")?,
            format!("eog {eog}, want 0 or 1"),
        ));
    }
    Ok(Case {
        name: r.text("name")?.to_string(),
        kind: r.text("kind")?.to_string(),
        image: r.text("image")?.to_string(),
        n_ids: r.parse("n_ids")?,
        span_at: r.parse("span_at")?,
        span_len: r.parse("span_len")?,
        gen_max: r.parse("gen")?,
        answer: r.parse("answer")?,
        eog: eog == 1,
        logits: r.parse("logits")?,
        question: r.text("question")?.to_string(),
    })
}

#[cfg(test)]
mod tests;
