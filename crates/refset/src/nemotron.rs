//! Last-position logits of Nemotron 3 Nano, 3.5 Bolt and LFM3-1.7B (`tools/ref/nemotron_ref.cpp`,
//! `just dump-ref-nemotron`, `just dump-ref-nemotron-moe`, `just dump-ref-lfm3`). One prompt is one row, read by the names on
//! `# prompt columns`. `logits` is the last position only, `n_vocab` f32,
//! little-endian; `ids` is that prompt, one id a line.
//!
//! ```text
//! # model    <path>
//! # build    llama-tokenize <md5> libllama.so <md5>
//! # arch     nemotron_h (the Bolt: nemotron_h_moe, LFM3: lfm3)
//! # prompt columns  name  n_ids  n_vocab  ids  logits
//! prompt  code5  5  131072  code5.ids  code5.logits.f32
//! # complete  4  0
//! ```
//!
//! A set dumped on a fixture file ([`Identity::FixtureNemotronLogits`]) also carries the file's `# fixture` line. A
//! set of a file with routed experts carries one `# route layer` line per MoE layer, in layer order, and a `routes`
//! column: the experts each of those layers selected at the prompt's last position, as a set (`ggml_top_k` names no
//! order), i32 little-endian, `k` ids a layer:
//!
//! ```text
//! # route layer  2  240  22
//! # route layer  3  496  22
//! ```
//!
//! [`LogitsSet::open`] refuses a set that did not finish, one dumped from
//! another file or another generation of the fixture, and one written by another mainline build.

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
    /// The routes file under the set's directory, in a set that has [`LogitsSet::route_layers`].
    pub routes_file: Option<String>,
}

/// One `# route layer` line: an MoE layer whose last-position selection a set records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteLayer {
    /// The layer's index in the file.
    pub layer: u32,
    /// The layer's routed-expert count: every id is below it.
    pub n_expert: u32,
    /// How many experts the layer selects.
    pub k: u32,
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
    /// The text after the tab of `# fixture`, in a set dumped on a fixture file.
    pub fixture: Option<String>,
    /// The `# route layer` lines, in order; empty in a set of a file without routed experts.
    pub route_layers: Vec<RouteLayer>,
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
            fixture: None,
            route_layers: Vec::new(),
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
                "# fixture" => {
                    if set.fixture.is_some() {
                        return Err(RefError::malformed(&at, "a second # fixture line"));
                    }
                    set.fixture = Some(one(&f, &at, "# fixture\t<key=value ...>")?);
                }
                "# route layer" => {
                    let layer = route_layer(&f, &at)?;
                    if set
                        .route_layers
                        .last()
                        .is_some_and(|prev| prev.layer >= layer.layer)
                    {
                        return Err(RefError::malformed(
                            &at,
                            "route layers must be strictly increasing, with no repeated layer",
                        ));
                    }
                    set.route_layers.push(layer);
                }
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
        if !matches!(
            family.identity,
            Identity::NemotronLogits | Identity::FixtureNemotronLogits
        ) {
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
    /// the model file and, for a fixture family, the `# fixture` line ([`RefError::Stale`]), the build and the
    /// architecture ([`RefError::Foreign`]), then each prompt's ids, logits and routes files.
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if !self.complete {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_fixture(&self.dir, self.fixture.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        if self.prompts.is_empty() {
            return Err(RefError::malformed(
                self.dir.join("MANIFEST.tsv").display(),
                "no prompt rows: the set names nothing to check",
            ));
        }
        for p in &self.prompts {
            if p.routes_file.is_some() == self.route_layers.is_empty() {
                return Err(RefError::malformed(
                    self.dir.join("MANIFEST.tsv").display(),
                    format!(
                        "{}: a routes file and `# route layer` lines come together ({} lines, routes file {:?})",
                        p.name,
                        self.route_layers.len(),
                        p.routes_file
                    ),
                ));
            }
            self.check_prompt(p)?;
        }
        Ok(())
    }

    /// Prompt `p`'s last-position selection, one set of `k` ids per [`LogitsSet::route_layers`] entry, each id below
    /// its layer's expert count and none twice; `None` in a set without routes.
    pub fn routes(&self, p: &Prompt) -> Result<Option<Vec<Vec<u32>>>, RefError> {
        let Some(file) = &p.routes_file else {
            return Ok(None);
        };
        let path = self.dir.join(file);
        let bytes = std::fs::read(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("nemotron: cannot read {}: {e}", path.display()),
            )
        })?;
        let want: usize = self.route_layers.iter().map(|l| l.k as usize * 4).sum();
        if bytes.len() != want {
            return Err(RefError::malformed(
                path.display(),
                format!(
                    "{file} is {} bytes, the route layers want {want}",
                    bytes.len()
                ),
            ));
        }
        let mut ids = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c));
        let mut out = Vec::with_capacity(self.route_layers.len());
        for l in &self.route_layers {
            let mut layer = Vec::with_capacity(l.k as usize);
            for id in ids.by_ref().take(l.k as usize) {
                let id = u32::try_from(id)
                    .ok()
                    .filter(|&id| id < l.n_expert)
                    .ok_or_else(|| {
                        RefError::malformed(
                            path.display(),
                            format!(
                                "layer {}: expert id {id} outside its {} experts",
                                l.layer, l.n_expert
                            ),
                        )
                    })?;
                if layer.contains(&id) {
                    return Err(RefError::malformed(
                        path.display(),
                        format!("layer {}: expert {id} selected twice", l.layer),
                    ));
                }
                layer.push(id);
            }
            out.push(layer);
        }
        Ok(Some(out))
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
        self.routes(p)?;
        Ok(())
    }
}

fn one(f: &[&str], at: &dyn fmt::Display, want: &str) -> Result<String, RefError> {
    match f {
        [_, v] if !v.is_empty() => Ok((*v).to_string()),
        _ => Err(RefError::malformed(at, format!("want {want}"))),
    }
}

/// `# route layer\t<layer>\t<n_expert>\t<k>`: a layer that selects 1..=n_expert of its experts.
fn route_layer(f: &[&str], at: &At<'_>) -> Result<RouteLayer, RefError> {
    let [_, layer, n_expert, k] = f else {
        return Err(RefError::malformed(
            at,
            "want # route layer\t<layer>\t<n_expert>\t<k>",
        ));
    };
    let l = RouteLayer {
        layer: crate::columns::parse_field(layer, "route layer", at)?,
        n_expert: crate::columns::parse_field(n_expert, "route n_expert", at)?,
        k: crate::columns::parse_field(k, "route k", at)?,
    };
    if l.k == 0 || l.k > l.n_expert {
        return Err(RefError::malformed(
            at,
            format!(
                "layer {} selects {} of {} experts",
                l.layer, l.k, l.n_expert
            ),
        ));
    }
    Ok(l)
}

fn prompt_row(r: &Row<'_>, at: &dyn fmt::Display) -> Result<Prompt, RefError> {
    let name = r.text("name")?;
    let ids_file = r.text("ids")?;
    let logits_file = r.text("logits")?;
    let routes_file = r.opt("routes");
    file_name(name, at)?;
    file_name(ids_file, at)?;
    file_name(logits_file, at)?;
    if let Some(f) = routes_file {
        file_name(f, at)?;
    }
    Ok(Prompt {
        name: name.to_string(),
        n_ids: r.parse("n_ids")?,
        n_vocab: r.parse("n_vocab")?,
        ids_file: ids_file.to_string(),
        logits_file: logits_file.to_string(),
        routes_file: routes_file.map(str::to_string),
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
            n >= 5,
            "the arch table holds {n} logits families, want the Nano's, the Bolt's and LFM3's three"
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

    /// The logits family of a file with routed experts (the Bolt's), from the arch table by name.
    fn routed() -> &'static Family {
        crate::arch::named("ref-nemotron-moe-logits")
            .unwrap_or_else(|| panic!("the arch table holds no ref-nemotron-moe-logits"))
    }

    /// [`routed`]'s set of `write` with `lines` added before the column line, a `routes` column when `column`, and
    /// `ids` as `p.routes.i32`.
    fn with_routes(d: &Path, lines: &str, column: bool, ids: &[i32]) -> &'static Family {
        let fam = routed();
        write(fam, d, &runs(fam), pin(fam), true);
        let path = d.join("MANIFEST.tsv");
        let mut text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        text = text.replace("# prompt columns", &format!("{lines}# prompt columns"));
        if column {
            text = text
                .replace("\tids\n", "\tids\troutes\n")
                .replace("\tp.ids\n", "\tp.ids\tp.routes.i32\n");
        }
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("{e}"));
        let bytes: Vec<u8> = ids.iter().copied().flat_map(i32::to_le_bytes).collect();
        std::fs::write(d.join("p.routes.i32"), bytes).unwrap_or_else(|e| panic!("{e}"));
        fam
    }

    const TWO_LAYERS: &str = "# route layer\t2\t240\t2\n# route layer\t3\t496\t2\n";

    /// A set with route layers reads each layer's selection, in the lines' order, `k` ids a layer.
    #[test]
    fn a_set_with_routes_reads_each_layers_selection() {
        let d = dir(routed(), "routes");
        let fam = with_routes(&d, TWO_LAYERS, true, &[7, 239, 495, 0]);
        let set = LogitsSet::open(&d, fam).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            set.route_layers
                .iter()
                .map(|l| (l.layer, l.n_expert, l.k))
                .collect::<Vec<_>>(),
            [(2, 240, 2), (3, 496, 2)]
        );
        let routes = set
            .routes(&set.prompts[0])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(routes, Some(vec![vec![7, 239], vec![495, 0]]));
        std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
    }

    /// An id at or past its layer's expert count, a negative one, one selected twice in a layer, a file of
    /// another length, a routes column without route layers, route layers without the column, a layer that
    /// selects more experts than it has, and a route layer at or below the one before it are each
    /// malformed, by name.
    #[test]
    fn routes_outside_their_layers_are_malformed() {
        let cases: [(&str, &str, bool, &[i32], &str); 9] = [
            (
                "id-past",
                TWO_LAYERS,
                true,
                &[7, 240, 495, 0],
                "expert id 240 outside its 240 experts",
            ),
            (
                "id-neg",
                TWO_LAYERS,
                true,
                &[7, 1, -1, 0],
                "expert id -1 outside its 496 experts",
            ),
            (
                "twice",
                TWO_LAYERS,
                true,
                &[7, 7, 495, 0],
                "layer 2: expert 7 selected twice",
            ),
            (
                "short",
                TWO_LAYERS,
                true,
                &[7, 1, 495],
                "12 bytes, the route layers want 16",
            ),
            (
                "no-layers",
                "",
                true,
                &[7, 1],
                "a routes file and `# route layer` lines come together",
            ),
            (
                "no-column",
                TWO_LAYERS,
                false,
                &[7, 1, 495, 0],
                "a routes file and `# route layer` lines come together",
            ),
            (
                "k-past",
                "# route layer\t2\t4\t5\n",
                true,
                &[0, 1, 2, 3, 0],
                "layer 2 selects 5 of 4 experts",
            ),
            (
                "layer-twice",
                "# route layer\t2\t240\t2\n# route layer\t2\t240\t2\n",
                true,
                &[7, 1, 7, 1],
                "route layers must be strictly increasing, with no repeated layer",
            ),
            (
                "layers-reversed",
                "# route layer\t3\t496\t2\n# route layer\t2\t240\t2\n",
                true,
                &[7, 1, 239, 0],
                "route layers must be strictly increasing, with no repeated layer",
            ),
        ];
        for (tag, lines, column, ids, want) in cases {
            let d = dir(routed(), tag);
            let fam = with_routes(&d, lines, column, ids);
            match LogitsSet::open(&d, fam) {
                Err(RefError::Malformed { what, .. }) => {
                    assert!(what.contains(want), "{tag}: {what}")
                }
                other => panic!("{tag}: {other:?}"),
            }
            std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}
