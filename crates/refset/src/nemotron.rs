//! Last-position logits of Nemotron 3 Nano and 3.5 Bolt (`tools/ref/nemotron_ref.cpp`,
//! `just dump-ref-nemotron`, `just dump-ref-nemotron-moe`). One prompt is one row, read by the names on
//! `# prompt columns`. `logits` is the last position only, `n_vocab` f32,
//! little-endian; `ids` is that prompt, one id a line.
//!
//! ```text
//! # model    <path>
//! # build    llama-tokenize <md5> libllama.so <md5>
//! # arch     nemotron_h (the Bolt: nemotron_h_moe)
//! # prompt columns  name  n_ids  n_vocab  ids  logits
//! prompt  code5  5  131072  code5.ids  code5.logits.f32
//! # complete  4  0
//! ```
//!
//! [`LogitsSet::open`] refuses a set that did not finish, one dumped from
//! another file, and one written by another mainline build.

use crate::RefError;
use crate::columns::{Columns, Row, file_name};
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
    /// a count that does not parse, a `# complete` count other than the prompt
    /// rows, or naming a prompt twice, is `Malformed`.
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
        let mut complete_rows = None;
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
                "# complete" => {
                    if f.len() != 3 || f[2] != "0" {
                        return Err(RefError::malformed(&at, "want # complete\t<count>\t0"));
                    }
                    complete_rows = Some(crate::columns::parse_field(f[1], "complete count", &at)?);
                    set.complete = true;
                }
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
        if complete_rows != Some(set.prompts.len()) {
            return Err(RefError::malformed(
                path.display(),
                "completion count differs from prompt rows",
            ));
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
        for chunk in bytes.as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(*chunk));
        }
        if let Some(i) = out.iter().position(|v| !v.is_finite()) {
            return Err(RefError::malformed(
                path.display(),
                format!("non-finite logit at index {i}"),
            ));
        }
        Ok(out)
    }

    /// The prompt's token ids, in order: `ids_file`, one id a line, as many as `n_ids` names, each
    /// inside the vocabulary.
    pub fn ids(&self, p: &Prompt) -> Result<Vec<u32>, RefError> {
        let path = self.dir.join(&p.ids_file);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("nemotron: cannot read {}: {e}", path.display()),
            )
        })?;
        if p.n_ids == 0 || p.n_vocab == 0 {
            return Err(RefError::malformed(
                path.display(),
                "empty prompt or vocabulary",
            ));
        }
        let mut ids = Vec::with_capacity(p.n_ids as usize);
        for (i, line) in text.lines().enumerate() {
            if line.is_empty() {
                return Err(RefError::malformed(At(&path, i + 1), "empty token id"));
            }
            let id: u32 = crate::columns::parse_field(line, "token id", &At(&path, i + 1))?;
            if id >= p.n_vocab {
                return Err(RefError::malformed(
                    At(&path, i + 1),
                    format!("token id {id} outside vocabulary {}", p.n_vocab),
                ));
            }
            ids.push(id);
        }
        if ids.len() != p.n_ids as usize {
            return Err(RefError::malformed(
                path.display(),
                format!(
                    "{} holds {} ids, {} wants {}",
                    p.ids_file,
                    ids.len(),
                    p.name,
                    p.n_ids
                ),
            ));
        }
        Ok(ids)
    }

    fn check_prompt(&self, p: &Prompt) -> Result<(), RefError> {
        self.ids(p)?;
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

#[cfg(test)]
mod tests {
    use super::LogitsSet;
    use crate::RefError;
    use crate::family::{Build, Family, Identity};
    use std::path::Path;

    /// Every last-position logits family of the arch table runs each test below: a family added to the table is
    /// covered with no line here, and a family's own file holds only what names that family.
    fn each_family(check: impl Fn(&Family)) {
        let mut n = 0;
        for fam in crate::arch::all().filter(|f| f.identity == Identity::NemotronLogits) {
            check(fam);
            n += 1;
        }
        assert!(
            n >= 2,
            "the arch table holds {n} logits families, want the Nano's and the Bolt's"
        );
    }

    fn pin(fam: &Family) -> &'static str {
        match fam.build {
            Some(Build::Is(p)) => p,
            _ => panic!("{}: the logits family pins no build", fam.name),
        }
    }

    fn runs(fam: &Family) -> String {
        fam.runs().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Write one prompt's ids and logits and the manifest. `complete` adds the trailer.
    fn write(fam: &Family, dir: &Path, model: &str, build: &str, complete: bool) {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join("p.ids"), "0\n1\n2\n3\n0\n").unwrap_or_else(|e| panic!("{e}"));
        let logits: Vec<u8> = [0.0f32, 1.5, -0.5, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        std::fs::write(dir.join("p.logits.f32"), logits).unwrap_or_else(|e| panic!("{e}"));
        let arch = fam
            .arch
            .unwrap_or_else(|| panic!("the family names no arch"));
        let mut text = format!(
            "# model\t{model}\n# build\t{build}\n# arch\t{arch}\n\
             # prompt columns\tname\tn_vocab\tn_ids\tlogits\tids\n\
             prompt\tp\t4\t5\tp.logits.f32\tp.ids\n"
        );
        if complete {
            text.push_str("# complete\t1\t0\n");
        }
        std::fs::write(dir.join("MANIFEST.tsv"), text).unwrap_or_else(|e| panic!("{e}"));
    }

    fn dir(fam: &Family, tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "bloomery-{}-{tag}-{}",
            fam.name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A finished set opens, and `logits` is that column's file.
    #[test]
    fn a_finished_set_reads_logits_by_column_name() {
        each_family(|fam| {
            let d = dir(fam, "ok");
            write(fam, &d, &runs(fam), pin(fam), true);
            let set = LogitsSet::open(&d, fam).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(set.prompts.len(), 1);
            let row = set
                .logits(&set.prompts[0])
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(row[1], 1.5);
            assert_eq!(
                set.ids(&set.prompts[0]).unwrap_or_else(|e| panic!("{e}")),
                [0, 1, 2, 3, 0]
            );
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// No `# complete`: unfinished, even when the other lines are well formed.
    #[test]
    fn a_set_without_its_trailer_is_unfinished() {
        each_family(|fam| {
            let d = dir(fam, "trailer");
            write(fam, &d, &runs(fam), pin(fam), false);
            match LogitsSet::open(&d, fam) {
                Err(RefError::Unfinished { set }) => assert_eq!(set, d.display().to_string()),
                other => panic!("a set without its trailer: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// Another model path is stale.
    #[test]
    fn a_set_of_another_model_is_stale() {
        each_family(|fam| {
            let d = dir(fam, "model");
            write(fam, &d, "/models/other/file.gguf", pin(fam), true);
            match LogitsSet::open(&d, fam) {
                Err(RefError::Stale { dumped_from, .. }) => {
                    assert_eq!(dumped_from, "/models/other/file.gguf");
                }
                other => panic!("another file: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// Another mainline build is foreign.
    #[test]
    fn a_set_of_another_build_is_foreign() {
        each_family(|fam| {
            let d = dir(fam, "build");
            write(
                fam,
                &d,
                &runs(fam),
                "llama-tokenize 00000000000000000000000000000000 libllama.so 00000000000000000000000000000000",
                true,
            );
            match LogitsSet::open(&d, fam) {
                Err(RefError::Foreign { field, .. }) => assert_eq!(field, "build"),
                other => panic!("another build: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// A line that is not a token id is malformed, naming the field and the text.
    #[test]
    fn a_token_line_that_is_not_an_id_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "bad-id");
            write(fam, &d, &runs(fam), pin(fam), true);
            std::fs::write(d.join("p.ids"), "0\n1\n2\ngarbage\n0\n")
                .unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(
                        what.contains("token id") && what.contains("garbage"),
                        "{what}"
                    );
                }
                other => panic!("a token line that is not an id: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// A NaN or infinite logit is malformed, naming its index.
    #[test]
    fn a_non_finite_logit_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "nan-logit");
            write(fam, &d, &runs(fam), pin(fam), true);
            let logits: Vec<u8> = [0.0f32, f32::NAN, -0.5, 0.25]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            std::fs::write(d.join("p.logits.f32"), logits).unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(what.contains("non-finite logit at index 1"), "{what}");
                }
                other => panic!("a non-finite logit: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// An empty line among the ids is malformed, as the C++ reader that writes the set refuses it.
    #[test]
    fn an_empty_token_line_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "empty-id");
            write(fam, &d, &runs(fam), pin(fam), true);
            std::fs::write(d.join("p.ids"), "0\n\n1\n2\n3\n0\n").unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(what.contains("empty token id"), "{what}")
                }
                other => panic!("an empty token line: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// An id at or past `n_vocab` is malformed, naming the id and the vocabulary.
    #[test]
    fn a_token_id_outside_the_vocabulary_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "oob-id");
            write(fam, &d, &runs(fam), pin(fam), true);
            std::fs::write(d.join("p.ids"), "0\n1\n2\n3\n4\n").unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(what.contains("outside vocabulary"), "{what}");
                }
                other => panic!("a token id outside the vocabulary: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// The trailer's count is the prompt rows. One row and `# complete 2` is malformed.
    #[test]
    fn a_completion_count_that_is_not_the_prompt_rows_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "complete-n");
            write(fam, &d, &runs(fam), pin(fam), true);
            let text =
                std::fs::read_to_string(d.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
            std::fs::write(
                d.join("MANIFEST.tsv"),
                text.replace("# complete\t1\t0\n", "# complete\t2\t0\n"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(what.contains("completion count"), "{what}");
                }
                other => panic!("a completion count that is not the prompt rows: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// A completion count that does not parse is malformed, naming the field.
    #[test]
    fn a_completion_count_that_does_not_parse_is_malformed() {
        each_family(|fam| {
            let d = dir(fam, "complete-bad");
            write(fam, &d, &runs(fam), pin(fam), true);
            let text =
                std::fs::read_to_string(d.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
            std::fs::write(
                d.join("MANIFEST.tsv"),
                text.replace("# complete\t1\t0\n", "# complete\tgarbage\t0\n"),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(
                        what.contains("complete count") && what.contains("garbage"),
                        "{what}"
                    );
                }
                other => panic!("a completion count that does not parse: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        });
    }
}
