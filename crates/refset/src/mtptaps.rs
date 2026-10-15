//! The MTP step oracle's sets (`tools/ref/q35mtp_ref.cpp`, `just dump-ref-q35mtp`): llama.cpp mainline's
//! next-token (MTP) graph run on a teacher-forced prompt, with the graph's own nodes dumped at some of its
//! steps. Step `i` takes token `ids[i + 1]`, position `i` and the trunk's `h_nextn` row `i`; every step runs in
//! order and the taps are read at the steps `# rows` names. A tap is read by its graph name and its row.
//!
//! ```text
//! # model    <path>
//! # build    <mainline commit>
//! # arch     qwen35moe
//! # mtp      <n_layer_nextn>  <steps run>
//! # tokens_count  64
//! # rows     0  5  31  62
//! # trunk columns  name  type  rows  width  file
//! trunk  h_nextn  f32  64  2048  trunk_h.f32
//! # step columns   row  token  pos  mtp_argmax  target_next
//! step   0  <ids[1]>  0  <id>  <id>
//! # tap columns    name  row  type  n  file
//! tap    mtp_eh_proj  0  f32  2048  mtp_eh_proj.r0.f32
//! # complete  <tap rows>  0
//! ```
//!
//! `prompt.ids` beside the manifest is the prompt, one id a line. `# mtp` is written only when the model has a
//! next-token layer and the MTP context ran every step: a set without it was dumped without the MTP context.
//!
//! [`TapSet::open`] refuses a set that did not finish, one dumped from another file or by another mainline
//! build, for another architecture, without the MTP context, and one whose files disagree with its rows.

use crate::RefError;
use crate::columns::{Columns, Row, file_name, need, one, read_bytes, read_f32};
use crate::family::{Family, Identity};
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

/// The prompt's ids, one a line, beside the manifest.
pub const IDS_FILE: &str = "prompt.ids";

/// A tap's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elem {
    F32,
    I32,
}

impl Elem {
    fn parse(v: &str, at: &dyn fmt::Display) -> Result<Elem, RefError> {
        match v {
            "f32" => Ok(Elem::F32),
            "i32" => Ok(Elem::I32),
            v => Err(RefError::malformed(
                at,
                format!("type {v:?}, want f32 or i32"),
            )),
        }
    }
}

/// The trunk's `h_nextn` rows: the states the steps are fed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trunk {
    /// The rows' name (`h_nextn`).
    pub name: String,
    /// One row a prompt position.
    pub rows: u32,
    /// The row's width.
    pub width: u32,
    /// The f32 file under the set's directory, row `t` at `t * width`.
    pub file: String,
}

/// One MTP step's token and the two argmaxes the oracle kept beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// The step's row, `i`.
    pub row: u32,
    /// The token fed, `ids[i + 1]`.
    pub token: u32,
    /// The position fed, `i`.
    pub pos: u32,
    /// The MTP head's argmax at this step (the first id on a tie).
    pub mtp_argmax: u32,
    /// The trunk's own argmax at position `i + 1`: the token the target takes after `token`.
    pub target_next: u32,
}

/// One tapped node at one row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tap {
    /// The graph's name of the node, without its layer suffix.
    pub name: String,
    pub row: u32,
    pub elem: Elem,
    /// Elements.
    pub n: u32,
    /// The file under the set's directory, `n` little-endian values.
    pub file: String,
}

/// `# mtp`: the model's next-token layers and the steps the MTP context ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpRan {
    pub layers: u32,
    pub steps: u32,
}

/// A tap set's manifest, and once opened, its files checked.
#[derive(Debug, Clone)]
pub struct TapSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# model`.
    pub model: Option<String>,
    /// `# build`.
    pub build: Option<String>,
    /// `# arch`.
    pub arch: Option<String>,
    /// `# mtp`; `None` for a set dumped without the MTP context.
    pub mtp: Option<MtpRan>,
    /// `# tokens_count`: the prompt's length.
    pub tokens_count: Option<u32>,
    /// `# rows`: the steps the taps were read at.
    pub rows: Vec<u32>,
    pub trunk: Option<Trunk>,
    /// One row a step, in the manifest's order.
    pub steps: Vec<Step>,
    /// One row a (tap, row), in the manifest's order.
    pub taps: Vec<Tap>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mtptaps: {}:{}", self.0.display(), self.1)
    }
}

impl TapSet {
    /// Parse `dir/MANIFEST.tsv`. A set without one is `Missing`, naming the recipe that writes it. A set without
    /// its trailer comes back unread, `complete` false. A line of the wrong width, a row before its column line,
    /// a count that does not parse, a file name that leaves the directory or a (tap, row) named twice is
    /// `Malformed`.
    pub fn read(dir: &Path, family: &Family) -> Result<TapSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = TapSet {
            dir: dir.to_path_buf(),
            model: None,
            build: None,
            arch: None,
            mtp: None,
            tokens_count: None,
            rows: Vec::new(),
            trunk: None,
            steps: Vec::new(),
            taps: Vec::new(),
            complete: false,
        };
        // A manifest without its trailer is a dump that did not finish: its rows are not read.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let mut trunk_cols: Option<Columns> = None;
        let mut step_cols: Option<Columns> = None;
        let mut tap_cols: Option<Columns> = None;
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
                "# tokens_count" => {
                    set.tokens_count = Some(number(&one(&f, &at, "# tokens_count\t<n>")?, &at)?);
                }
                "# mtp" => {
                    let &[_, layers, steps] = f.as_slice() else {
                        return Err(RefError::malformed(&at, "want # mtp\t<layers>\t<steps>"));
                    };
                    set.mtp = Some(MtpRan {
                        layers: number(layers, &at)?,
                        steps: number(steps, &at)?,
                    });
                }
                "# rows" => {
                    set.rows = f[1..]
                        .iter()
                        .map(|v| number(v, &at))
                        .collect::<Result<_, _>>()?;
                }
                "# trunk columns" => {
                    trunk_cols = Some(Columns::new("trunk", f[1..].iter().copied()))
                }
                "# step columns" => step_cols = Some(Columns::new("step", f[1..].iter().copied())),
                "# tap columns" => tap_cols = Some(Columns::new("tap", f[1..].iter().copied())),
                "# complete" => {
                    check_trailer(&f, set.taps.len(), text.lines().skip(i + 1), &at)?;
                    set.complete = true;
                }
                h if h.starts_with('#') => {}
                "trunk" => {
                    let r = need(&trunk_cols, "trunk", &at)?.row(line, &at)?;
                    if set.trunk.is_some() {
                        return Err(RefError::malformed(&at, "a second trunk row"));
                    }
                    set.trunk = Some(trunk_row(&r, &at)?);
                }
                "step" => {
                    let r = need(&step_cols, "step", &at)?.row(line, &at)?;
                    let step = step_row(&r)?;
                    if !seen.insert(("step", step.row, String::new())) {
                        return Err(RefError::malformed(
                            &at,
                            format!("step {} is named twice", step.row),
                        ));
                    }
                    set.steps.push(step);
                }
                "tap" => {
                    let r = need(&tap_cols, "tap", &at)?.row(line, &at)?;
                    let tap = tap_row(&r, &at)?;
                    if !seen.insert(("tap", tap.row, tap.name.clone())) {
                        return Err(RefError::malformed(
                            &at,
                            format!("tap {} at row {} is named twice", tap.name, tap.row),
                        ));
                    }
                    set.taps.push(tap);
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
    pub fn open(dir: &Path, family: &Family) -> Result<TapSet, RefError> {
        if !matches!(family.identity, Identity::MtpTaps) {
            return Err(RefError::missing(
                dir,
                format!("mtptaps: {} is not an MTP tap family", family.name),
            ));
        }
        let set = TapSet::read(dir, family)?;
        set.check_family(family)?;
        Ok(set)
    }

    /// This set against its family's row: the trailer ([`RefError::Unfinished`]), the model file
    /// ([`RefError::Stale`]), the build, the architecture and the MTP context ([`RefError::Foreign`]), then
    /// the rows against each other and each file against its row.
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        let mtp = self.mtp.ok_or_else(|| RefError::Foreign {
            set: self.dir.display().to_string(),
            family: family.name,
            field: "mtp",
            got: "(no # mtp line)".to_string(),
            want: "a set dumped with the model's MTP context (llama_model_n_layer_nextn > 0, every step run)"
                .to_string(),
        })?;
        let at = self.dir.join("MANIFEST.tsv");
        let n = self
            .tokens_count
            .ok_or_else(|| RefError::malformed(at.display(), "no # tokens_count line"))?;
        if mtp.layers == 0 || u64::from(mtp.steps) + 1 != u64::from(n) {
            return Err(RefError::malformed(
                at.display(),
                format!(
                    "# mtp {} {}: the MTP context needs a next-token layer and one step a prompt id but the last ({n} ids)",
                    mtp.layers, mtp.steps
                ),
            ));
        }
        let ids = self.ids()?;
        if ids.len() != n as usize {
            return Err(RefError::malformed(
                self.dir.join(IDS_FILE).display(),
                format!(
                    "{IDS_FILE} holds {} ids, # tokens_count says {n}",
                    ids.len()
                ),
            ));
        }
        self.check_trunk(n)?;
        self.check_steps(&ids, mtp.steps)?;
        self.check_taps()
    }

    fn check_trunk(&self, n: u32) -> Result<(), RefError> {
        let at = self.dir.join("MANIFEST.tsv");
        let t = self
            .trunk
            .as_ref()
            .ok_or_else(|| RefError::malformed(at.display(), "no trunk row"))?;
        if t.rows != n {
            return Err(RefError::malformed(
                at.display(),
                format!("the trunk holds {} rows, the prompt {n} ids", t.rows),
            ));
        }
        self.trunk_rows().map(|_| ())
    }

    fn check_steps(&self, ids: &[u32], steps: u32) -> Result<(), RefError> {
        let at = self.dir.join("MANIFEST.tsv");
        if self.steps.len() != steps as usize {
            return Err(RefError::malformed(
                at.display(),
                format!(
                    "{} step rows, # mtp says {steps} steps ran",
                    self.steps.len()
                ),
            ));
        }
        for (i, s) in self.steps.iter().enumerate() {
            if s.row as usize != i || s.pos != s.row || ids.get(i + 1) != Some(&s.token) {
                return Err(RefError::malformed(
                    at.display(),
                    format!(
                        "step row {} is not step {i}: token {} at position {}, the prompt's id {} is {:?}",
                        s.row,
                        s.token,
                        s.pos,
                        i + 1,
                        ids.get(i + 1)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Every tapped row has every tap, every row is a step, and every file is its row's length.
    fn check_taps(&self) -> Result<(), RefError> {
        let at = self.dir.join("MANIFEST.tsv");
        if self.rows.is_empty() || self.taps.is_empty() {
            return Err(RefError::malformed(
                at.display(),
                "no # rows or no tap rows: the set names nothing to check",
            ));
        }
        let names: Vec<&str> = {
            let mut v: Vec<&str> = Vec::new();
            for t in &self.taps {
                if !v.contains(&t.name.as_str()) {
                    v.push(&t.name);
                }
            }
            v
        };
        for &row in &self.rows {
            if self.step(row).is_err() {
                return Err(RefError::malformed(
                    at.display(),
                    format!("# rows names {row}, which is no step"),
                ));
            }
            for name in &names {
                if !self.taps.iter().any(|t| t.row == row && t.name == *name) {
                    return Err(RefError::malformed(
                        at.display(),
                        format!("no tap {name} at row {row}, which # rows names"),
                    ));
                }
            }
        }
        if self.taps.len() != self.rows.len() * names.len() {
            return Err(RefError::malformed(
                at.display(),
                format!(
                    "{} tap rows for {} taps at {} rows: a tap at a row # rows does not name",
                    self.taps.len(),
                    names.len(),
                    self.rows.len()
                ),
            ));
        }
        for t in &self.taps {
            let path = self.dir.join(&t.file);
            let bytes = std::fs::metadata(&path)
                .map_err(|e| {
                    RefError::missing(
                        &path,
                        format!("mtptaps: cannot read {}: {e}", path.display()),
                    )
                })?
                .len();
            if bytes != u64::from(t.n) * 4 {
                return Err(RefError::malformed(
                    path.display(),
                    format!(
                        "{} is {bytes} bytes, tap {} of {} values wants {}",
                        t.file,
                        t.name,
                        t.n,
                        u64::from(t.n) * 4
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The prompt's ids.
    pub fn ids(&self) -> Result<Vec<u32>, RefError> {
        let path = self.dir.join(IDS_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("mtptaps: cannot read {}: {e}", path.display()),
            )
        })?;
        text.lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.parse().map_err(|e| {
                    RefError::malformed(path.display(), format!("{IDS_FILE}: id {l:?}: {e}"))
                })
            })
            .collect()
    }

    /// The trunk's `h_nextn` rows, `rows * width` f32, row `t` at `t * width`.
    pub fn trunk_rows(&self) -> Result<Vec<f32>, RefError> {
        let t = self.trunk.as_ref().ok_or_else(|| {
            RefError::missing(
                &self.dir,
                format!("mtptaps: {} has no trunk row", self.dir.display()),
            )
        })?;
        read_f32(
            &self.dir.join(&t.file),
            u64::from(t.rows) * u64::from(t.width),
            &t.file,
        )
    }

    /// Step `row`.
    pub fn step(&self, row: u32) -> Result<&Step, RefError> {
        self.steps.iter().find(|s| s.row == row).ok_or_else(|| {
            RefError::missing(
                &self.dir,
                format!("mtptaps: {} has no step {row}", self.dir.display()),
            )
        })
    }

    /// The tap `name` at `row`; a name or a row the set does not hold is `Missing`, naming both.
    pub fn tap(&self, name: &str, row: u32) -> Result<&Tap, RefError> {
        self.taps
            .iter()
            .find(|t| t.row == row && t.name == name)
            .ok_or_else(|| {
                RefError::missing(
                    &self.dir,
                    format!(
                        "mtptaps: {} has no tap {name} at row {row} (taps {:?}, rows {:?})",
                        self.dir.display(),
                        self.taps
                            .iter()
                            .map(|t| t.name.as_str())
                            .collect::<BTreeSet<_>>(),
                        self.rows
                    ),
                )
            })
    }

    /// The f32 tap `name` at `row`; an i32 tap is `Malformed` here, as a count of another length is.
    pub fn tap_f32(&self, name: &str, row: u32) -> Result<Vec<f32>, RefError> {
        let t = self.tap(name, row)?;
        if t.elem != Elem::F32 {
            return Err(RefError::malformed(
                self.dir.join(&t.file).display(),
                format!("tap {name} is i32, read as f32"),
            ));
        }
        read_f32(&self.dir.join(&t.file), u64::from(t.n), &t.file)
    }

    /// The i32 tap `name` at `row` (the routed expert ids).
    pub fn tap_i32(&self, name: &str, row: u32) -> Result<Vec<i32>, RefError> {
        let t = self.tap(name, row)?;
        if t.elem != Elem::I32 {
            return Err(RefError::malformed(
                self.dir.join(&t.file).display(),
                format!("tap {name} is f32, read as i32"),
            ));
        }
        let bytes = read_bytes(&self.dir.join(&t.file), u64::from(t.n), &t.file)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    }
}

/// `# complete <tap rows> 0`, the manifest's last line: a set whose trailer counts other tap rows
/// than it holds lost rows after the writer counted them. `after` is the lines after the trailer.
fn check_trailer<'a>(
    f: &[&str],
    taps: usize,
    mut after: impl Iterator<Item = &'a str>,
    at: &dyn fmt::Display,
) -> Result<(), RefError> {
    if f.len() != 3
        || f[2] != "0"
        || number(f[1], at)? as usize != taps
        || after.any(|l| !l.is_empty())
    {
        return Err(RefError::malformed(
            at,
            format!(
                "the trailer `{}` is not `# complete\t{taps}\t0` on the last line",
                f.join("\t")
            ),
        ));
    }
    Ok(())
}

fn number(v: &str, at: &dyn fmt::Display) -> Result<u32, RefError> {
    crate::columns::parse_field(v, "count", at)
}

/// The trunk row: its rows are f32 and its file is one name of the set's directory.
fn trunk_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<Trunk, RefError> {
    let file = r.text("file")?;
    file_name(file, at)?;
    if r.text("type")? != "f32" {
        return Err(RefError::malformed(at, "the trunk rows are f32"));
    }
    Ok(Trunk {
        name: r.text("name")?.to_string(),
        rows: r.parse("rows")?,
        width: r.parse("width")?,
        file: file.to_string(),
    })
}

fn step_row(r: &Row<'_>) -> Result<Step, RefError> {
    Ok(Step {
        row: r.parse("row")?,
        token: r.parse("token")?,
        pos: r.parse("pos")?,
        mtp_argmax: r.parse("mtp_argmax")?,
        target_next: r.parse("target_next")?,
    })
}

fn tap_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<Tap, RefError> {
    let name = r.text("name")?;
    let file = r.text("file")?;
    file_name(name, at)?;
    file_name(file, at)?;
    Ok(Tap {
        name: name.to_string(),
        row: r.parse("row")?,
        elem: Elem::parse(r.text("type")?, at)?,
        n: r.parse("n")?,
        file: file.to_string(),
    })
}
