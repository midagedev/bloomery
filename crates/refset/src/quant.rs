//! The quant oracle (`tools/ref/quant_oracle.cpp`, run by `tools/ref/quant-oracle.sh`): llama.cpp mainline's ggml on seeded rows,
//! per weight type and row length. One case holds the weight rows in the type's blocks (rows ggml quantized, then rows of random
//! codes with finite scales), their `to_float`, f32 activation rows with their `vec_dot_type` conversion, and every `vec_dot`
//! of a weight row with an activation row. The identity is `QUANT.tsv` beside the files, written after them:
//!
//! ```text
//! # harness  <md5>       tools/ref/quant_oracle.cpp, the source the files came from
//! # build    <commit>    mainline's commit, the first nine digits
//! # cpu      <features>  the CPU features ggml's dot kernels were selected under
//! # model    (synthetic rows)
//! # case     name type k blck type_size row_bytes vdt vdt_name act_row_bytes quant_rows random_rows act_rows imatrix
//! case       …           one row a type and row length
//! # file     name bytes md5
//! file       …           one row a file of the set
//! # complete <n>         how many file rows
//! ```
//!
//! [`QuantSet::open`] refuses by name a set dumped from another harness or another mainline commit, from a model file (it is
//! synthetic), with a file that is not the one dumped, a case whose files disagree with its row, or that never finished.

use crate::RefError;
use crate::columns::{Columns, Row, file_name, parse_field};
use crate::dequant::{SetFile, check_files, file_row};
use crate::family::{Family, Identity};
use crate::md5::check_hex;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub use crate::dequant::SYNTHETIC;

/// The name of the identity file in a set's directory.
pub const MANIFEST: &str = "QUANT.tsv";

/// What a case's file holds, which names the file after the case: `<case><suffix>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The weight rows in the type's blocks, `quant_rows` then `random_rows` of them.
    Blocks,
    /// f32 `to_float` of every weight row.
    Deq,
    /// f32 activation rows.
    X,
    /// The activation rows in the type's `vec_dot_type`.
    Act,
    /// f32 `vec_dot` of weight row `w` with activation row `a`, at `w * act_rows + a`.
    Dot,
}

impl Part {
    pub const ALL: [Part; 5] = [Part::Blocks, Part::Deq, Part::X, Part::Act, Part::Dot];

    /// The file name's suffix after the case's name.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Part::Blocks => ".blocks",
            Part::Deq => ".deq.f32",
            Part::X => ".x.f32",
            Part::Act => ".act",
            Part::Dot => ".dot.f32",
        }
    }

    /// Whether the file holds little-endian f32 values.
    #[must_use]
    pub const fn is_f32(self) -> bool {
        matches!(self, Part::Deq | Part::X | Part::Dot)
    }
}

/// One type at one row length, as the harness's `cases.tsv` row states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    /// `<ggml type name>.<k>`: the stem of the case's files.
    pub name: String,
    /// The ggml type id.
    pub ty: u32,
    /// The row length in values.
    pub k: usize,
    /// ggml's block size in values, and the block's size in bytes.
    pub blck: usize,
    pub type_size: usize,
    /// One weight row in bytes: `k / blck * type_size`.
    pub row_bytes: usize,
    /// The CPU backend's `vec_dot_type`, by id and ggml's name, and one activation row in it in bytes.
    pub vdt: u32,
    pub vdt_name: String,
    pub act_row_bytes: usize,
    /// Rows ggml quantized (with a seeded imatrix when `imatrix`), then rows of random codes.
    pub quant_rows: usize,
    pub random_rows: usize,
    pub act_rows: usize,
    pub imatrix: bool,
}

impl Case {
    /// The weight rows: the quantized ones, then the random-code ones.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.quant_rows + self.random_rows
    }

    /// ggml's name of the type: the case's name before its row length.
    #[must_use]
    pub fn type_name(&self) -> &str {
        self.name.rsplit_once('.').map_or(&self.name, |(t, _)| t)
    }

    /// The file of `part`.
    #[must_use]
    pub fn file(&self, part: Part) -> String {
        format!("{}{}", self.name, part.suffix())
    }

    /// The length of the file of `part` the row implies.
    #[must_use]
    pub fn bytes(&self, part: Part) -> usize {
        match part {
            Part::Blocks => self.rows() * self.row_bytes,
            Part::Deq => self.rows() * self.k * 4,
            Part::X => self.act_rows * self.k * 4,
            Part::Act => self.act_rows * self.act_row_bytes,
            Part::Dot => self.rows() * self.act_rows * 4,
        }
    }

    /// The row against itself: the name ends in the row length, the sizes agree, nothing is empty.
    fn check(&self, at: &dyn fmt::Display) -> Result<(), RefError> {
        let named = self
            .name
            .rsplit_once('.')
            .and_then(|(_, k)| k.parse::<usize>().ok());
        let bad = |what: String| {
            Err(RefError::malformed(
                at,
                format!("case {}: {what}", self.name),
            ))
        };
        if named != Some(self.k) {
            return bad(format!(
                "the name does not end in the row length {}",
                self.k
            ));
        }
        if self.blck == 0 || !self.k.is_multiple_of(self.blck) {
            return bad(format!(
                "{} values is not whole blocks of {}",
                self.k, self.blck
            ));
        }
        if self.row_bytes != self.k / self.blck * self.type_size {
            return bad(format!(
                "a row of {} bytes, not {} blocks of {}",
                self.row_bytes,
                self.k / self.blck,
                self.type_size
            ));
        }
        if self.quant_rows == 0 || self.act_rows == 0 || self.act_row_bytes == 0 {
            return bad(
                "no quantized rows, no activation rows or an empty activation row".to_string(),
            );
        }
        Ok(())
    }
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "quant: {}:{}", self.0.display(), self.1)
    }
}

/// A quant set's identity file and, once opened, its files and cases checked.
#[derive(Debug, Clone)]
pub struct QuantSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# harness`: the md5 of the harness source.
    pub harness: Option<String>,
    /// `# build`: mainline's commit.
    pub build: Option<String>,
    /// `# cpu`: the CPU features the dot kernels ran under.
    pub cpu: Option<String>,
    /// `# model`: [`SYNTHETIC`] for every set this crate takes.
    pub model: Option<String>,
    /// One row a type and row length, in the manifest's order.
    pub cases: Vec<Case>,
    /// One row a file, in the manifest's order.
    pub files: Vec<SetFile>,
    /// The file count the `# complete` trailer states; `None` without the trailer.
    pub complete: Option<usize>,
}

impl QuantSet {
    /// Parse `dir/QUANT.tsv`. A set without one is `Missing`, naming the recipe that writes it; one without its trailer
    /// comes back unread, `complete` `None`; a line of the wrong width, a value that does not parse, a row before its
    /// column line, a case or file named twice, a file outside the directory, or a trailer that counts other than the
    /// file rows, is `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<QuantSet, RefError> {
        let path = dir.join(MANIFEST);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = QuantSet {
            dir: dir.to_path_buf(),
            harness: None,
            build: None,
            cpu: None,
            model: None,
            cases: Vec::new(),
            files: Vec::new(),
            complete: None,
        };
        // A manifest without its trailer is a dump that did not finish: its rows are not read, so the refusal is the
        // one that says so.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let (mut case_cols, mut file_cols): (Option<Columns>, Option<Columns>) = (None, None);
        let (mut case_names, mut file_names) = (HashSet::new(), HashSet::new());
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# harness" => {
                    let md5 = value(&f, &at)?;
                    check_hex("# harness", md5, &at)?;
                    set.harness = Some(md5.to_string());
                }
                "# build" => set.build = Some(value(&f, &at)?.to_string()),
                "# cpu" => set.cpu = Some(value(&f, &at)?.to_string()),
                "# model" => set.model = Some(value(&f, &at)?.to_string()),
                "# complete" => {
                    set.complete = Some(parse_field(value(&f, &at)?, "# complete", &at)?)
                }
                "# case" => case_cols = Some(Columns::new("case", f[1..].iter().copied())),
                "# file" => file_cols = Some(Columns::new("file", f[1..].iter().copied())),
                "case" => {
                    let c = case_cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a case row before its `# case` column line")
                    })?;
                    let case = case_row(&c.row(line, &at)?, &at)?;
                    if !case_names.insert(case.name.clone()) {
                        return Err(RefError::malformed(
                            &at,
                            format!("case {} is named twice", case.name),
                        ));
                    }
                    set.cases.push(case);
                }
                "file" => {
                    let c = file_cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a file row before its `# file` column line")
                    })?;
                    let file = file_row(&c.row(line, &at)?, "name", &at)?;
                    if !file_names.insert(file.name.clone()) {
                        return Err(RefError::malformed(
                            &at,
                            format!("{} is named twice", file.name),
                        ));
                    }
                    set.files.push(file);
                }
                "" => {}
                h if h.starts_with('#') => {}
                h => {
                    return Err(RefError::malformed(
                        &at,
                        format!("{h:?} is no row kind of a quant set"),
                    ));
                }
            }
        }
        match set.complete {
            Some(n) if n != set.files.len() => Err(RefError::malformed(
                path.display(),
                format!(
                    "the trailer counts {n} files, the manifest names {}",
                    set.files.len()
                ),
            )),
            _ => Ok(set),
        }
    }

    /// The set at `dir` read ([`read`](Self::read)), checked against `family`'s row ([`check_family`](Self::check_family)),
    /// its files against their digests ([`check_files`](Self::check_files)) and its cases against its files
    /// ([`check_cases`](Self::check_cases)).
    pub fn open(dir: &Path, family: &Family) -> Result<QuantSet, RefError> {
        let set = QuantSet::read(dir, family)?;
        set.check_family(family)?;
        set.check_files()?;
        set.check_cases()?;
        Ok(set)
    }

    /// This set against its family's row: the completion trailer ([`RefError::Unfinished`]), the model line
    /// ([`RefError::Stale`]), then the harness and mainline's commit ([`RefError::Foreign`], fields `harness` and `build`).
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if self.complete.is_none() {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        let Identity::QuantManifest { harness } = family.identity else {
            return Err(RefError::missing(
                &self.dir,
                format!("the {} family is not a quant family", family.name),
            ));
        };
        if self.harness.as_deref() != Some(harness) {
            return Err(RefError::Foreign {
                set: self.dir.display().to_string(),
                family: family.name,
                field: "harness",
                got: self
                    .harness
                    .clone()
                    .unwrap_or_else(|| "(no # harness line)".to_string()),
                want: harness.to_string(),
            });
        }
        family.check_build(&self.dir, self.build.as_deref())
    }

    /// Every file against its row's length and md5: a file that is not the one dumped is `Malformed`, naming it and both
    /// digests.
    pub fn check_files(&self) -> Result<(), RefError> {
        check_files(&self.dir, MANIFEST, &self.files)
    }

    /// Every case against itself and its five files: each is a file row of the length the case implies, and no file row
    /// belongs to no case.
    pub fn check_cases(&self) -> Result<(), RefError> {
        let manifest = self.dir.join(MANIFEST);
        if self.cases.is_empty() {
            return Err(RefError::malformed(
                manifest.display(),
                "no case rows: the set holds no type",
            ));
        }
        let files: HashMap<&str, u64> = self
            .files
            .iter()
            .map(|f| (f.name.as_str(), f.bytes))
            .collect();
        let mut claimed = HashSet::new();
        for c in &self.cases {
            c.check(&manifest.display())?;
            for part in Part::ALL {
                let file = c.file(part);
                let Some(&bytes) = files.get(file.as_str()) else {
                    return Err(RefError::malformed(
                        manifest.display(),
                        format!("case {} names no file row {file}", c.name),
                    ));
                };
                if bytes != c.bytes(part) as u64 {
                    return Err(RefError::malformed(
                        manifest.display(),
                        format!(
                            "{file} is {bytes} bytes, case {} implies {}",
                            c.name,
                            c.bytes(part)
                        ),
                    ));
                }
                claimed.insert(file);
            }
        }
        match self.files.iter().find(|f| !claimed.contains(&f.name)) {
            Some(f) => Err(RefError::malformed(
                manifest.display(),
                format!("file {} belongs to no case", f.name),
            )),
            None => Ok(()),
        }
    }

    /// The case `name` (`<ggml type name>.<k>`).
    pub fn case(&self, name: &str) -> Result<&Case, RefError> {
        self.cases.iter().find(|c| c.name == name).ok_or_else(|| {
            RefError::missing(
                &self.dir,
                format!("{}: no case {name}", self.dir.join(MANIFEST).display()),
            )
        })
    }

    /// The cases of the type ggml names `type_name`, one a row length.
    pub fn of_type<'a>(&'a self, type_name: &'a str) -> impl Iterator<Item = &'a Case> + 'a {
        self.cases
            .iter()
            .filter(move |c| c.type_name() == type_name)
    }

    /// The bytes of `case`'s file of `part`, of the length the case implies.
    pub fn bytes(&self, case: &Case, part: Part) -> Result<Vec<u8>, RefError> {
        let path = self.dir.join(case.file(part));
        let bytes = std::fs::read(&path)
            .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
        if bytes.len() != case.bytes(part) {
            return Err(RefError::malformed(
                path.display(),
                format!(
                    "{} bytes, case {} implies {}",
                    bytes.len(),
                    case.name,
                    case.bytes(part)
                ),
            ));
        }
        Ok(bytes)
    }

    /// The values of `case`'s f32 file of `part` ([`Part::is_f32`]).
    pub fn f32s(&self, case: &Case, part: Part) -> Result<Vec<f32>, RefError> {
        if !part.is_f32() {
            return Err(RefError::malformed(
                self.dir.join(case.file(part)).display(),
                format!("{part:?} holds bytes, not f32 values"),
            ));
        }
        let bytes = self.bytes(case, part)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}

/// The value of a header line `# key\tvalue`; another width is `Malformed` at `at`.
fn value<'a>(f: &[&'a str], at: &dyn fmt::Display) -> Result<&'a str, RefError> {
    match f {
        [_, v] => Ok(v),
        _ => Err(RefError::malformed(
            at,
            format!("{:?}: want <key>\t<value>, got {} fields", f[0], f.len()),
        )),
    }
}

fn case_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<Case, RefError> {
    let name = r.text("name")?;
    file_name(name, at)?;
    let imatrix = match r.text("imatrix")? {
        "0" => false,
        "1" => true,
        v => {
            return Err(RefError::malformed(
                at,
                format!("imatrix {v:?}: want 0 or 1"),
            ));
        }
    };
    Ok(Case {
        name: name.to_string(),
        ty: r.parse("type")?,
        k: r.parse("k")?,
        blck: r.parse("blck")?,
        type_size: r.parse("type_size")?,
        row_bytes: r.parse("row_bytes")?,
        vdt: r.parse("vdt")?,
        vdt_name: r.text("vdt_name")?.to_string(),
        act_row_bytes: r.parse("act_row_bytes")?,
        quant_rows: r.parse("quant_rows")?,
        random_rows: r.parse("random_rows")?,
        act_rows: r.parse("act_rows")?,
        imatrix,
    })
}

#[cfg(test)]
mod tests;
