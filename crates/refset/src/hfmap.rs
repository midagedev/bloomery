//! A Hugging Face checkpoint's name map oracle (`tools/ref/glm5next-hf-map.py`): the checkpoint's
//! shard headers, small tensors range-read from it, and the reference GGUF's tensor table and the
//! same tensors as its converter wrote them. The map gate compares the names, dims and converted
//! values our map gives with the GGUF's. The identity is `MANIFEST.tsv` beside the files, written
//! after them:
//!
//! ```text
//! # harness     <md5>                          the writer's source
//! # checkpoint  <repo>@<revision>              the checkpoint the headers and hf.safetensors came from
//! # model       <repo>@<revision>/<quant>      the GGUF the table and gguf.safetensors came from
//! # arch        <arch>                         the architecture the GGUF names
//! # file        name bytes md5
//! file          …                              one row a file of the set
//! # complete    <n>                            how many file rows
//! ```
//!
//! [`HfMapSet::open`] refuses by name a set of another checkpoint, GGUF, architecture or writer,
//! a set without `hf.safetensors`, `tensors.tsv`, `gguf.safetensors` or a shard header, with a
//! file that is not the one written, or that never finished.

use crate::RefError;
use crate::columns::Columns;
use crate::dequant::{SetFile, check_files, file_row};
use crate::family::{Family, Identity, check_same};
use crate::md5::check_hex;
use crate::quant::value;
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

/// The name of the identity file in a set's directory.
const MANIFEST: &str = "MANIFEST.tsv";

/// The checkpoint's small tensors, one safetensors file.
pub const HF: &str = "hf.safetensors";
/// The GGUF's tensor table: `tensor` and `dimensions` (ggml order, `x`-joined) columns.
pub const TENSORS: &str = "tensors.tsv";
/// The GGUF's tensors of [`HF`]'s, as F32 safetensors.
pub const GGUF: &str = "gguf.safetensors";
/// The files every set holds besides its shard headers.
const REQUIRED: [&str; 3] = [HF, TENSORS, GGUF];
/// The suffix of a shard header file: the shard's JSON header, gzipped.
const HEADER: &str = ".header.json.gz";

/// A map oracle's identity file and, once opened, its files checked.
#[derive(Debug, Clone)]
pub struct HfMapSet {
    /// The set's directory.
    pub dir: PathBuf,
    /// `# harness`: the md5 of the writer's source.
    pub(crate) harness: Option<String>,
    /// `# checkpoint`: `<repo>@<revision>` of the checkpoint.
    checkpoint: Option<String>,
    /// `# model`: the GGUF, `<repo>@<revision>/<quant>`.
    pub(crate) model: Option<String>,
    /// `# arch`.
    arch: Option<String>,
    /// One row a file, in the manifest's order.
    files: Vec<SetFile>,
    /// The file count the `# complete` trailer states; `None` without the trailer.
    complete: Option<usize>,
}

struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hfmap: {}:{}", self.0.display(), self.1)
    }
}

impl HfMapSet {
    /// Parse `dir/MANIFEST.tsv`. A set without one is `Missing`, naming the recipe that writes it;
    /// one without its trailer comes back unread, `complete` `None`; a line of the wrong width, a
    /// row before its column line or of an unknown kind, a file named twice or outside the
    /// directory, or a trailer that counts other than the file rows, is `Malformed`.
    fn read(dir: &Path, family: &Family) -> Result<HfMapSet, RefError> {
        let path = dir.join(MANIFEST);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("{}: {e} — run: {}", path.display(), family.recipe),
            )
        })?;
        let mut set = HfMapSet {
            dir: dir.to_path_buf(),
            harness: None,
            checkpoint: None,
            model: None,
            arch: None,
            files: Vec::new(),
            complete: None,
        };
        // A manifest without its trailer is a write that did not finish: its rows are not read, so
        // the refusal is the one that says so.
        if !text
            .lines()
            .any(|l| l.split('\t').next() == Some("# complete"))
        {
            return Ok(set);
        }
        let mut cols: Option<Columns> = None;
        let mut names = HashSet::new();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# harness" => {
                    let md5 = value(&f, &at)?;
                    check_hex("# harness", md5, &at)?;
                    set.harness = Some(md5.to_string());
                }
                "# checkpoint" => set.checkpoint = Some(value(&f, &at)?.to_string()),
                "# model" => set.model = Some(value(&f, &at)?.to_string()),
                "# arch" => set.arch = Some(value(&f, &at)?.to_string()),
                "# complete" => {
                    set.complete = Some(crate::columns::parse_field(
                        value(&f, &at)?,
                        "# complete",
                        &at,
                    )?);
                }
                "# file" => cols = Some(Columns::new("file", f[1..].iter().copied())),
                "file" => {
                    let c = cols.as_ref().ok_or_else(|| {
                        RefError::malformed(&at, "a file row before its `# file` column line")
                    })?;
                    let file = file_row(&c.row(line, &at)?, "name", &at)?;
                    if !names.insert(file.name.clone()) {
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
                        format!("{h:?} is no row kind of a map oracle"),
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

    /// The set at `dir` read, checked against `family`'s row (the trailer, the GGUF, the
    /// checkpoint, the architecture and the writer), for its required files and against their
    /// digests.
    pub fn open(dir: &Path, family: &Family) -> Result<HfMapSet, RefError> {
        let set = HfMapSet::read(dir, family)?;
        set.check_family(family)?;
        set.check_files()?;
        Ok(set)
    }

    /// This set against its family's row: the completion trailer ([`RefError::Unfinished`]), the
    /// GGUF and the checkpoint ([`RefError::Stale`]), then the architecture and the writer
    /// ([`RefError::Foreign`], fields `arch` and `build`).
    fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if self.complete.is_none() {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        let Identity::HfMap { checkpoint } = family.identity else {
            return Err(RefError::missing(
                &self.dir,
                format!("the {} family is not a map oracle family", family.name),
            ));
        };
        check_same(
            &self.dir,
            "# checkpoint",
            self.checkpoint.as_deref(),
            checkpoint,
        )?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        family.check_build(&self.dir, self.harness.as_deref())
    }

    /// Every file of [`REQUIRED`] and at least one shard header named, and every file against its
    /// row's length and md5: a missing one is `Missing`, a file that is not the one written is
    /// `Malformed`, naming it and both digests.
    fn check_files(&self) -> Result<(), RefError> {
        let manifest = self.dir.join(MANIFEST);
        for name in REQUIRED {
            if !self.files.iter().any(|f| f.name == name) {
                return Err(RefError::missing(
                    &manifest,
                    format!("{}: no file row names {name}", manifest.display()),
                ));
            }
        }
        if self.headers().next().is_none() {
            return Err(RefError::missing(
                &manifest,
                format!("{}: no file row names a *{HEADER}", manifest.display()),
            ));
        }
        check_files(&self.dir, MANIFEST, &self.files)
    }

    /// The paths of the shard header files the manifest names.
    pub fn headers(&self) -> impl Iterator<Item = PathBuf> + '_ {
        self.files
            .iter()
            .filter(|f| f.name.ends_with(HEADER))
            .map(|f| self.dir.join(&f.name))
    }
}

#[cfg(test)]
mod tests {
    use super::{GGUF, HF, HfMapSet, MANIFEST, REQUIRED, TENSORS};
    use crate::RefError;
    use crate::family::{Build, Family, Identity};
    use crate::md5::hex_of;
    use std::path::Path;

    const HARNESS: &str = "26743c698bb627717351fab4dde24272";
    const CHECKPOINT: &str = "org/ckpt@1111";
    const MODEL: &str = "org/gguf@2222/Q4";

    fn model() -> Result<String, RefError> {
        Ok(MODEL.to_string())
    }

    static FAMILY: Family = Family {
        name: "test-hfmap",
        sets: &[],
        resolve: None,
        recipe: "just dump-ref-glm5next-hf-map",
        identity: Identity::HfMap {
            checkpoint: CHECKPOINT,
        },
        arch: Some("glm5next"),
        build: Some(Build::Is(HARNESS)),
        runs: Some(model),
        draft_runs: None,
        consumers: &[],
    };

    /// A set under `dir` of `files`, whose header lines are `head`; with the trailer when `complete`.
    fn write(dir: &Path, head: &[(&str, &str)], files: &[&str], complete: bool) {
        let mut lines: Vec<String> = head.iter().map(|(k, v)| format!("{k}\t{v}")).collect();
        lines.push("# file\tname\tbytes\tmd5".to_string());
        for name in files {
            let bytes = name.as_bytes();
            std::fs::write(dir.join(name), bytes).unwrap();
            lines.push(format!("file\t{name}\t{}\t{}", bytes.len(), hex_of(bytes)));
        }
        if complete {
            lines.push(format!("# complete\t{}", files.len()));
        }
        std::fs::write(dir.join(MANIFEST), format!("{}\n", lines.join("\n"))).unwrap();
    }

    const HEAD: [(&str, &str); 4] = [
        ("# harness", HARNESS),
        ("# checkpoint", CHECKPOINT),
        ("# model", MODEL),
        ("# arch", "glm5next"),
    ];
    const FILES: [&str; 4] = [
        HF,
        TENSORS,
        GGUF,
        "model-00001-of-00002.safetensors.header.json.gz",
    ];

    /// A complete set of the family's checkpoint, GGUF, architecture and writer opens, its
    /// headers listed; another of any of the four, an unfinished one, one without a required file
    /// or a header, and one with a changed file are refused by name.
    #[test]
    fn a_set_opens_only_as_its_family_states_it() {
        let dir = crate::clefvis::testkit::scratch("hfmap");
        write(&dir, &HEAD, &FILES, true);
        let set = HfMapSet::open(&dir, &FAMILY).unwrap();
        assert_eq!(set.headers().count(), 1);

        let refused = |head: &[(&str, &str)], files: &[&str], complete: bool| {
            write(&dir, head, files, complete);
            HfMapSet::open(&dir, &FAMILY).unwrap_err()
        };
        let with =
            |key: &str, v: &'static str| HEAD.map(|(k, old)| (k, if k == key { v } else { old }));
        let stale = |e: RefError, from: &str| {
            assert!(
                matches!(&e, RefError::Stale { dumped_from, .. } if dumped_from == from),
                "{e}"
            );
        };
        stale(
            refused(&with("# model", "org/gguf@3333/Q4"), &FILES, true),
            "org/gguf@3333/Q4",
        );
        stale(
            refused(&with("# checkpoint", "org/ckpt@0000"), &FILES, true),
            "org/ckpt@0000",
        );
        stale(
            refused(
                &HEAD[..1]
                    .iter()
                    .chain(&HEAD[2..])
                    .copied()
                    .collect::<Vec<_>>(),
                &FILES,
                true,
            ),
            "an unstated file (the set has no # checkpoint line)",
        );
        let other: &'static str = hex_of(b"another writer").leak();
        for (key, field) in [("# arch", "arch"), ("# harness", "build")] {
            let e = refused(&with(key, other), &FILES, true);
            assert!(
                matches!(&e, RefError::Foreign { field: f, .. } if *f == field),
                "{e}"
            );
        }
        let e = refused(&HEAD, &FILES, false);
        assert!(matches!(e, RefError::Unfinished { .. }), "{e}");
        for name in REQUIRED.iter().chain(&[FILES[3]]) {
            let files: Vec<&str> = FILES.iter().copied().filter(|f| f != name).collect();
            let e = refused(&HEAD, &files, true);
            assert!(matches!(e, RefError::Missing { .. }), "{name}: {e}");
        }
        write(&dir, &HEAD, &FILES, true);
        std::fs::write(dir.join(GGUF), b"changed").unwrap();
        let e = HfMapSet::open(&dir, &FAMILY).unwrap_err();
        assert!(matches!(e, RefError::Malformed { .. }), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
