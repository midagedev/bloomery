//! The vision encoder's oracle (`tools/ref/vision/dump_vision.py`): the
//! official checkpoint's taps per test image. Its `MANIFEST.tsv` holds `#`
//! lines (where the set came from, the reference's own sensitivity per tap)
//! and `image` and `file` rows, `# complete` last; `plans.tsv` holds the
//! reference's resize plan per input size.
//!
//! A `# <kind> columns` line names the fields of its rows, space-separated
//! in one field (`# sensitivity columns` adds a description after a tab);
//! `plans.tsv` names its columns in a `# kind\t…` line. Every row is read by
//! those names.

use crate::RefError;
use crate::columns::{Columns, Row, parse_field};
use crate::family::Family;
use std::fmt;
use std::path::{Path, PathBuf};

/// One `image` row: a test image, the reference's plan for it and its grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub name: String,
    pub sha256: String,
    pub w: usize,
    pub h: usize,
    pub best_w: usize,
    pub best_h: usize,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub n_tokens: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub off_x: usize,
    pub off_y: usize,
}

impl Image {
    /// The image's name without `.png`: the stem of its files.
    #[must_use]
    pub fn stem(&self) -> &str {
        self.name.strip_suffix(".png").unwrap_or(&self.name)
    }
}

/// One `file` row: a file of the set, its dtype and shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetFile {
    pub name: String,
    pub kind: String,
    pub dtype: String,
    /// The shape, outermost first (`5476x1024` is `[5476, 1024]`).
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub md5: String,
}

/// One `# sensitivity` row: the reference against itself with one bit of
/// its input flipped, at a tap.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivity {
    pub tap: String,
    pub max_rel: f64,
    pub rms_rel: f64,
    pub differ: f64,
}

/// One row of `plans.tsv`: an input size and the reference's plan for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub kind: String,
    pub w: usize,
    pub h: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub best_h: usize,
    pub best_w: usize,
    pub n_tokens: usize,
    pub resized_w: usize,
    pub resized_h: usize,
    pub off_x: usize,
    pub off_y: usize,
}

/// A vision set's manifest.
#[derive(Debug, Clone)]
pub struct VisionSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# checkpoint`: `<repo>@<revision>` of the checkpoint the taps came from.
    pub checkpoint: Option<String>,
    /// `# mmproj`: the projector file the set names, and its sha256.
    pub mmproj: PathBuf,
    pub mmproj_sha256: String,
    /// `# image_token_id`.
    pub image_token_id: u32,
    pub images: Vec<Image>,
    pub files: Vec<SetFile>,
    pub sensitivity: Vec<Sensitivity>,
    /// Whether the `# complete` trailer is there.
    pub complete: bool,
}

/// A vision manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vision: {}:{}", self.0.display(), self.1)
    }
}

#[derive(Default)]
struct SetColumns {
    image: Option<Columns>,
    file: Option<Columns>,
    sensitivity: Option<Columns>,
}

impl VisionSet {
    /// Parse `dir/MANIFEST.tsv`. The `# mmproj` and `# image_token_id`
    /// lines and at least one `image` row are required; a row before its
    /// column line, of another width, or with a field that does not parse is
    /// `Malformed`.
    pub fn read(dir: &Path) -> Result<VisionSet, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: just dump-ref-vision", dir.display()),
            )
        })?;
        let (mut checkpoint, mut mmproj, mut image_token_id) = (None, None, None);
        let (mut images, mut files, mut sensitivity) = (Vec::new(), Vec::new(), Vec::new());
        let mut complete = false;
        let mut cols = SetColumns::default();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# checkpoint" => checkpoint = f.get(1).map(|s| s.to_string()),
                "# mmproj" => match f[..] {
                    [_, p, "sha256", sha, ..] => mmproj = Some((PathBuf::from(p), sha.to_string())),
                    _ => {
                        return Err(RefError::malformed(
                            &at,
                            format!("{line:?}: want <path>\tsha256\t<digest>"),
                        ));
                    }
                },
                "# image_token_id" => {
                    image_token_id = Some(parse_field(
                        f.get(1).copied().unwrap_or(""),
                        "image_token_id",
                        &at,
                    )?);
                }
                "# image columns" => cols.image = Some(columns("image", &f, &at)?),
                "# file columns" => cols.file = Some(columns("file", &f, &at)?),
                "# sensitivity columns" => {
                    cols.sensitivity = Some(columns("sensitivity", &f, &at)?)
                }
                "# sensitivity" => {
                    let c = need(&cols.sensitivity, "# sensitivity", &at)?;
                    let row = line.trim_start_matches("# ");
                    sensitivity.push(sensitivity_row(&c.row(row, &at)?)?);
                }
                "# complete" => complete = true,
                "image" => images.push(image_row(
                    &need(&cols.image, "image", &at)?.row(line, &at)?,
                )?),
                "file" => files.push(file_row(
                    &need(&cols.file, "file", &at)?.row(line, &at)?,
                    &at,
                )?),
                h if h.starts_with('#') => {}
                "" => {}
                k => return Err(RefError::malformed(&at, format!("unknown row kind {k:?}"))),
            }
        }
        let head =
            |what: &str| RefError::missing(&path, format!("{}: no {what} line", path.display()));
        let (mmproj, mmproj_sha256) = mmproj.ok_or_else(|| head("# mmproj"))?;
        let image_token_id = image_token_id.ok_or_else(|| head("# image_token_id"))?;
        if images.is_empty() {
            return Err(RefError::malformed(path.display(), "no image rows"));
        }
        Ok(VisionSet {
            dir: dir.to_path_buf(),
            checkpoint,
            mmproj,
            mmproj_sha256,
            image_token_id,
            images,
            files,
            sensitivity,
            complete,
        })
    }

    /// Read the set at `dir` and check it against `family`'s row: the
    /// completion trailer ([`RefError::Unfinished`]) and the checkpoint's
    /// revision ([`RefError::Stale`]).
    pub fn open(dir: &Path, family: &Family) -> Result<VisionSet, RefError> {
        let set = VisionSet::read(dir)?;
        if !set.complete {
            return Err(RefError::Unfinished {
                set: dir.display().to_string(),
            });
        }
        family.check_revision(dir, set.checkpoint.as_deref())?;
        Ok(set)
    }

    /// `plans.tsv`, by the names of its `# kind` line.
    pub fn plans(&self) -> Result<Vec<Plan>, RefError> {
        let path = self.dir.join("plans.tsv");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
        let mut cols = None;
        let mut plans = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            if let Some(names) = line.strip_prefix("# kind\t") {
                cols = Some(Columns::new("kind", names.split('\t')));
            } else if !line.starts_with('#') && !line.is_empty() {
                let r = need(&cols, "# kind", &at)?.row(line, &at)?;
                plans.push(Plan {
                    kind: r.text("kind")?.to_string(),
                    w: r.parse("w")?,
                    h: r.parse("h")?,
                    n_llm_h: r.parse("n_llm_h")?,
                    n_llm_w: r.parse("n_llm_w")?,
                    best_h: r.parse("best_h")?,
                    best_w: r.parse("best_w")?,
                    n_tokens: r.parse("n_tokens")?,
                    resized_w: r.parse("resized_w")?,
                    resized_h: r.parse("resized_h")?,
                    off_x: r.parse("off_x")?,
                    off_y: r.parse("off_y")?,
                });
            }
        }
        Ok(plans)
    }
}

/// The columns a `# <kind> columns\t<names>[\t<description>]` line names.
fn columns(kind: &str, f: &[&str], at: &dyn fmt::Display) -> Result<Columns, RefError> {
    let names = f
        .get(1)
        .ok_or_else(|| RefError::malformed(at, format!("# {kind} columns names none")))?;
    Ok(Columns::new(kind, names.split(' ')))
}

fn need<'a>(
    cols: &'a Option<Columns>,
    what: &str,
    at: &dyn fmt::Display,
) -> Result<&'a Columns, RefError> {
    cols.as_ref()
        .ok_or_else(|| RefError::malformed(at, format!("a {what} row before its column line")))
}

fn image_row(r: &Row<'_>) -> Result<Image, RefError> {
    Ok(Image {
        name: r.text("name")?.to_string(),
        sha256: r.text("sha256")?.to_string(),
        w: r.parse("w")?,
        h: r.parse("h")?,
        best_w: r.parse("best_w")?,
        best_h: r.parse("best_h")?,
        n_vit_h: r.parse("n_vit_h")?,
        n_vit_w: r.parse("n_vit_w")?,
        n_llm_h: r.parse("n_llm_h")?,
        n_llm_w: r.parse("n_llm_w")?,
        n_tokens: r.parse("n_tokens")?,
        resized_w: r.parse("resized_w")?,
        resized_h: r.parse("resized_h")?,
        off_x: r.parse("off_x")?,
        off_y: r.parse("off_y")?,
    })
}

fn file_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<SetFile, RefError> {
    Ok(SetFile {
        name: r.text("name")?.to_string(),
        kind: r.text("kind")?.to_string(),
        dtype: r.text("dtype")?.to_string(),
        shape: r
            .text("shape")?
            .split('x')
            .map(|d| parse_field(d, "shape", at))
            .collect::<Result<_, _>>()?,
        bytes: r.parse("bytes")?,
        md5: r.text("md5")?.to_string(),
    })
}

fn sensitivity_row(r: &Row<'_>) -> Result<Sensitivity, RefError> {
    Ok(Sensitivity {
        tap: r.text("tap")?.to_string(),
        max_rel: r.parse("max_rel")?,
        rms_rel: r.parse("rms_rel")?,
        differ: r.parse("differ")?,
    })
}

#[cfg(test)]
mod tests;
