//! The tokenizer oracle (`crates/tokenizer/tools/oracle.sh`): the ids
//! `llama-tokenize` gives the texts the tokenizer gate compares ours with, for
//! one vocabulary, written next to the exact text each came from. A set is a
//! directory, `MANIFEST.tsv` first:
//!
//! ```text
//! # tokenizer    <path>  <md5>   the llama-tokenize executable
//! # libllama     <path>  <md5>   the libllama.so it loads (the tokenizer lives there)
//! # vocabulary   <path>          the GGUF file whose vocabulary it read, first shard
//! # text tree    <path>          where the corpus texts were gathered from
//! # cases        <path>  <md5>   the hand-picked cases file the case texts came from
//! set  bytes  text_md5  ids  nps_ids
//! code            …              one row a text: a corpus, or cases/<nn>
//! # complete
//! ```
//!
//! Row `<name>` has its text in `<name>.txt`, its ids in `<name>.ids` (special
//! tokens parsed, `llama-tokenize`'s default) and `<name>.nps.ids`
//! (`--no-parse-special`), one id a line. [`TokenizerSet::open`] refuses by
//! name a set dumped with another executable or library, from another
//! vocabulary file, with a text that is not the one dumped, or that never
//! finished.

use crate::RefError;
use crate::columns::{Columns, Row, parse_field};
use crate::family::Family;
use crate::md5::{check_file, check_hex, hex_of, path_and_md5};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// One text of the set and the ids the reference gave it.
#[derive(Debug, Clone)]
pub struct Text {
    dir: PathBuf,
    /// The row's name: a corpus (`code`) or a case (`cases/07`), the stem of
    /// the text's files under the set's directory.
    pub name: String,
    /// The text's length in bytes, as dumped.
    pub bytes: u64,
    /// The text's md5, as dumped.
    pub md5: String,
    /// How many ids the reference gave it with special tokens parsed.
    pub ids: usize,
    /// How many with `--no-parse-special`.
    pub nps_ids: usize,
}

impl Text {
    /// Whether the text is one of the hand-picked cases, not a corpus.
    #[must_use]
    pub fn is_case(&self) -> bool {
        self.name.starts_with("cases/")
    }

    /// The text's file, `<dir>/<name>.txt`.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.dir.join(format!("{}.txt", self.name))
    }

    /// The text, read from its file.
    pub fn read(&self) -> Result<Vec<u8>, RefError> {
        let path = self.path();
        std::fs::read(&path)
            .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))
    }

    /// The reference's ids: `parse_special` is its default mode (special
    /// tokens parsed), else `--no-parse-special`. A file whose id count is not
    /// the row's, or a line that is not an id, is `Malformed`.
    pub fn ids(&self, parse_special: bool) -> Result<Vec<u32>, RefError> {
        let (ext, want) = if parse_special {
            ("ids", self.ids)
        } else {
            ("nps.ids", self.nps_ids)
        };
        let path = self.dir.join(format!("{}.{ext}", self.name));
        let bytes = std::fs::read(&path)
            .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
        let text = String::from_utf8(bytes)
            .map_err(|e| RefError::malformed(path.display(), format!("not UTF-8: {e}")))?;
        let mut ids = Vec::with_capacity(want);
        for (i, line) in text.lines().enumerate() {
            ids.push(parse_field(line, "id", &At(&path, i + 1))?);
        }
        if ids.len() != want {
            return Err(RefError::malformed(
                path.display(),
                format!("{} ids, the manifest's row names {want}", ids.len()),
            ));
        }
        Ok(ids)
    }
}

/// A tokenizer set's manifest and, once opened, its texts' digests checked.
#[derive(Debug, Clone)]
pub struct TokenizerSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# tokenizer`: the llama-tokenize executable and its md5.
    pub binary: Option<(String, String)>,
    /// `# libllama`: the library it loads and its md5.
    pub library: Option<(String, String)>,
    /// `# vocabulary`: the GGUF file the set was dumped from.
    pub vocabulary: Option<String>,
    /// `# text tree`: where the corpus texts were gathered from.
    pub text_tree: Option<String>,
    /// `# cases`: the cases file the case texts came from and its md5.
    pub cases: Option<(String, String)>,
    /// One row a text, in the manifest's order.
    pub texts: Vec<Text>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer: {}:{}", self.0.display(), self.1)
    }
}

/// What a family's `build` pin names for the executable and the library:
/// `llama-tokenize <md5> libllama.so <md5>`. The one spelling the dumper's
/// two `# tokenizer` and `# libllama` lines are read into.
#[must_use]
pub fn build_id(binary_md5: &str, library_md5: &str) -> String {
    format!("llama-tokenize {binary_md5} libllama.so {library_md5}")
}

impl TokenizerSet {
    /// Parse `dir/MANIFEST.tsv`. A set without one is `Missing`, naming the
    /// recipe that writes it; one without its trailer comes back unread,
    /// `complete` false; a line of the wrong width, an md5 that is not
    /// one, a row before its column line, of another width or with a count
    /// that does not parse, a text named twice or outside the directory, is
    /// `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<TokenizerSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = TokenizerSet {
            dir: dir.to_path_buf(),
            binary: None,
            library: None,
            vocabulary: None,
            text_tree: None,
            cases: None,
            texts: Vec::new(),
            complete: false,
        };
        // A manifest without its trailer is an unfinished dump or another format: its rows are
        // not read, so the refusal is the one that says so.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let mut cols: Option<Columns> = None;
        let mut seen = HashSet::new();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# tokenizer" => set.binary = Some(path_and_md5(&f, &at)?),
                "# libllama" => set.library = Some(path_and_md5(&f, &at)?),
                "# cases" => set.cases = Some(path_and_md5(&f, &at)?),
                "# vocabulary" => set.vocabulary = Some(one_value(&f, &at)?),
                "# text tree" => set.text_tree = Some(one_value(&f, &at)?),
                "# complete" => set.complete = true,
                "set" => cols = Some(Columns::new("set", f[1..].iter().copied())),
                "" => {}
                h if h.starts_with('#') => {}
                _ => {
                    let c = cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a text row before its `set` column line")
                    })?;
                    let text = text_row(dir, &c.row(line, &at)?, &at)?;
                    if !seen.insert(text.name.clone()) {
                        return Err(RefError::malformed(
                            &at,
                            format!("{} is named twice", text.name),
                        ));
                    }
                    set.texts.push(text);
                }
            }
        }
        Ok(set)
    }

    /// The set at `dir` read ([`read`](Self::read)), checked against
    /// `family`'s row ([`check_family`](Self::check_family)) and its texts
    /// against their digests ([`check_texts`](Self::check_texts)).
    pub fn open(dir: &Path, family: &Family) -> Result<TokenizerSet, RefError> {
        let set = TokenizerSet::read(dir, family)?;
        set.check_family(family)?;
        set.check_texts()?;
        Ok(set)
    }

    /// The executable and the library the set names, as a family's `build`
    /// pin spells them ([`build_id`]); `None` unless both lines are there.
    #[must_use]
    pub fn build(&self) -> Option<String> {
        let (_, binary) = self.binary.as_ref()?;
        let (_, library) = self.library.as_ref()?;
        Some(build_id(binary, library))
    }

    /// This set against its family's row: the completion trailer
    /// ([`RefError::Unfinished`]), the vocabulary file ([`RefError::Stale`]),
    /// then the executable and library ([`RefError::Foreign`], field
    /// `build`).
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# vocabulary", self.vocabulary.as_deref())?;
        family.check_build(&self.dir, self.build().as_deref())
    }

    /// Every text's file against its row's length and md5: a text that is
    /// not the one dumped is `Malformed`, naming the file and both digests.
    pub fn check_texts(&self) -> Result<(), RefError> {
        for t in &self.texts {
            check_file(&t.path(), t.bytes, &t.md5)?;
        }
        Ok(())
    }

    /// The case texts' source, the hand-picked cases file the tree holds
    /// (`cases`, its bytes), against the md5 the set names: a set dumped from
    /// other cases is [`RefError::Stale`].
    pub fn check_cases(&self, cases: &[u8]) -> Result<(), RefError> {
        let md5 = hex_of(cases);
        match &self.cases {
            Some((_, stated)) if *stated == md5 => Ok(()),
            stated => Err(RefError::Stale {
                set: self.dir.display().to_string(),
                dumped_from: stated.as_ref().map_or_else(
                    || "an unstated cases file (the set has no # cases line)".to_string(),
                    |(path, md5)| format!("{path} md5 {md5}"),
                ),
                runs: format!("the cases file the tree holds, md5 {md5}"),
            }),
        }
    }

    /// The corpus texts, not the cases.
    pub fn corpora(&self) -> impl Iterator<Item = &Text> {
        self.texts.iter().filter(|t| !t.is_case())
    }

    /// The hand-picked case texts.
    pub fn cases(&self) -> impl Iterator<Item = &Text> {
        self.texts.iter().filter(|t| t.is_case())
    }
}

/// The one value of a `# <key>\t<value>` line.
fn one_value(f: &[&str], at: &dyn fmt::Display) -> Result<String, RefError> {
    match f {
        [_, v] => Ok((*v).to_string()),
        _ => Err(RefError::malformed(
            at,
            format!("{:?}: want <key>\t<value>, got {} fields", f[0], f.len()),
        )),
    }
}

fn text_row(dir: &Path, r: &Row<'_>, at: &dyn fmt::Display) -> Result<Text, RefError> {
    let name = r.text("set")?;
    if name.is_empty()
        || name.starts_with('/')
        || name.split('/').any(|c| matches!(c, "" | "." | ".."))
    {
        return Err(RefError::malformed(
            at,
            format!("{name:?} does not name a file under the set"),
        ));
    }
    let md5 = r.text("text_md5")?;
    check_hex("text_md5", md5, at)?;
    Ok(Text {
        dir: dir.to_path_buf(),
        name: name.to_string(),
        bytes: r.parse("bytes")?,
        md5: md5.to_string(),
        ids: r.parse("ids")?,
        nps_ids: r.parse("nps_ids")?,
    })
}

#[cfg(test)]
pub(crate) mod tests;
