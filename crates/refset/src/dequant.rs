//! The dequantization oracle (`tools/ref/dequant_ref.cpp`, run by
//! `tools/ref/dump-dequant.sh`): ggml's `to_float` of the first rows of one
//! tensor per type of a model file, or of synthetic rows, which the `gate-1-1`
//! gate compares our `dequant_row` with bit for bit. The set's files are
//! `dequant_ref`'s own (`manifest.txt`, `<type>.meta`, `<type>.raw`, and
//! `<type>.blocks` for the synthetic set); the identity is `DEQUANT.tsv`
//! beside them, written after them:
//!
//! ```text
//! # dequant_ref  <path>  <md5>   the harness executable
//! # libggml      <path>  <md5>   the libggml.so it loads (to_float lives there)
//! # model        <path>          the GGUF file the rows came from, first shard
//! file  bytes  md5
//! manifest.txt   …               one row a file of the set
//! # complete
//! ```
//!
//! [`DequantSet::open`] refuses by name a set dumped with another harness or
//! another ggml build, from another model file, with a file that is not the
//! one dumped, or that never finished.

use crate::RefError;
use crate::columns::{Columns, Row};
use crate::family::Family;
use crate::md5::{check_file, check_hex, path_and_md5};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// The file a synthetic set states in its `# model` line: its rows are made
/// by ggml's quantizer from a fixed seed, from no model file.
pub const SYNTHETIC: &str = "(synthetic rows)";

/// The name of the identity file in a set's directory.
pub const MANIFEST: &str = "DEQUANT.tsv";

/// One file of the set, as dumped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetFile {
    /// The file's name under the set's directory.
    pub name: String,
    pub bytes: u64,
    pub md5: String,
}

/// A dequant set's identity file and, once opened, its files' digests
/// checked.
#[derive(Debug, Clone)]
pub struct DequantSet {
    /// The set's directory: where `manifest.txt` and the type files lie.
    pub dir: PathBuf,
    /// `# dequant_ref`: the harness executable and its md5.
    pub binary: Option<(String, String)>,
    /// `# libggml`: the library it loads and its md5.
    pub library: Option<(String, String)>,
    /// `# model`: the GGUF file the rows came from, or [`SYNTHETIC`].
    pub model: Option<String>,
    /// One row a file, in the manifest's order.
    pub files: Vec<SetFile>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dequant: {}:{}", self.0.display(), self.1)
    }
}

/// What a family's `build` pin names for the harness and the library:
/// `dequant_ref <md5> libggml.so <md5>`. The one spelling the dumper's two
/// `# dequant_ref` and `# libggml` lines are read into.
#[must_use]
pub fn build_id(binary_md5: &str, library_md5: &str) -> String {
    format!("dequant_ref {binary_md5} libggml.so {library_md5}")
}

impl DequantSet {
    /// Parse `dir/DEQUANT.tsv`. A set without one is `Missing`, naming the
    /// recipe that writes it; one without its trailer comes back unread,
    /// `complete` false; a line of the wrong width, an md5 that is not
    /// one, a row before its column line, of another width or with a length
    /// that does not parse, a file named twice or outside the directory, is
    /// `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<DequantSet, RefError> {
        let path = dir.join(MANIFEST);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = DequantSet {
            dir: dir.to_path_buf(),
            binary: None,
            library: None,
            model: None,
            files: Vec::new(),
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
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# dequant_ref" => set.binary = Some(path_and_md5(&f, &at)?),
                "# libggml" => set.library = Some(path_and_md5(&f, &at)?),
                "# model" => match f[..] {
                    [_, m] => set.model = Some(m.to_string()),
                    _ => {
                        return Err(RefError::malformed(
                            &at,
                            format!("{line:?}: want # model\t<path>"),
                        ));
                    }
                },
                "# complete" => set.complete = true,
                "file" => cols = Some(Columns::new("file", f[1..].iter().copied())),
                "" => {}
                h if h.starts_with('#') => {}
                _ => {
                    let c = cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a file row before its `file` column line")
                    })?;
                    let file = file_row(&c.row(line, &at)?, &at)?;
                    if !seen.insert(file.name.clone()) {
                        return Err(RefError::malformed(
                            &at,
                            format!("{} is named twice", file.name),
                        ));
                    }
                    set.files.push(file);
                }
            }
        }
        Ok(set)
    }

    /// The set at `dir` read ([`read`](Self::read)), checked against
    /// `family`'s row ([`check_family`](Self::check_family)) and its files
    /// against their digests ([`check_files`](Self::check_files)).
    pub fn open(dir: &Path, family: &Family) -> Result<DequantSet, RefError> {
        let set = DequantSet::read(dir, family)?;
        set.check_family(family)?;
        set.check_files()?;
        Ok(set)
    }

    /// The harness and the library the set names, as a family's `build` pin
    /// spells them ([`build_id`]); `None` unless both lines are there.
    #[must_use]
    pub fn build(&self) -> Option<String> {
        let (_, binary) = self.binary.as_ref()?;
        let (_, library) = self.library.as_ref()?;
        Some(build_id(binary, library))
    }

    /// This set against its family's row: the completion trailer
    /// ([`RefError::Unfinished`]), the model file ([`RefError::Stale`]),
    /// then the harness and library ([`RefError::Foreign`], field `build`).
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_build(&self.dir, self.build().as_deref())
    }

    /// Every file against its row's length and md5: a file that is not the
    /// one dumped is `Malformed`, naming it and both digests.
    pub fn check_files(&self) -> Result<(), RefError> {
        if self.files.is_empty() {
            return Err(RefError::malformed(
                self.dir.join(MANIFEST).display(),
                "no file rows: the set names nothing to check",
            ));
        }
        for f in &self.files {
            check_file(&self.dir.join(&f.name), f.bytes, &f.md5)?;
        }
        Ok(())
    }
}

fn file_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<SetFile, RefError> {
    let name = r.text("file")?;
    if name.is_empty() || name.contains('/') || matches!(name, "." | "..") {
        return Err(RefError::malformed(
            at,
            format!("{name:?} does not name a file of the set's directory"),
        ));
    }
    let md5 = r.text("md5")?;
    check_hex("md5", md5, at)?;
    Ok(SetFile {
        name: name.to_string(),
        bytes: r.parse("bytes")?,
        md5: md5.to_string(),
    })
}

#[cfg(test)]
pub(crate) mod tests;
