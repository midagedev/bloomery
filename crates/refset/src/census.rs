//! Census stubs: header-only copies of real HF model files, pinned by the
//! manifest [`MANIFEST`] so a gate plans every stub through each family's
//! placement without reading whatever happens to sit in the stub directory.
//! A stub is the first `header_bytes` bytes of its file — the GGUF header
//! padded to the tensor data offset — fetched verbatim by HTTP range at the
//! revision its row pins, then the file truncated (sparse) to the real
//! length, so every tensor byte is zero and a GGUF reader opens it as the
//! real file. The stubs live under [`ROOT`], at `<root>/<org>/<repo>/<file>`;
//! the writer is `tools/type-census.py --stubs`.
//!
//! The census is not a [`crate::family::Family`] row: a family is one kind of
//! set dumped from the one model file the tree runs, while the census holds
//! the headers of many users' files; its identity is the manifest's pinned
//! revisions and digests, so [`StubSet::check`] refuses a stub that differs
//! from its row ([`RefError::StubDiffers`]), not a set dumped from another
//! file ([`RefError::Stale`]).
//!
//! [`Manifest::read`] refuses a manifest that is not the fixed nine-column
//! format — a column missing or named twice, a row of another width, a
//! number that does not parse or a header longer than its file, a digest or
//! revision that is not lowercase hex, a file named twice, a set of two
//! revisions, a shard set that misses or doubles a shard, a set that is not
//! the stem of its file, a path that leaves the stub root. Every refusal is
//! named, never a default or a skip.

use crate::RefError;
use crate::columns::Columns;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::io::Read;
use std::path::{Path, PathBuf};

/// The manifest's path in the repo, relative to the workspace root.
pub const MANIFEST: &str = "tools/census/stubs.tsv";

/// Where the stubs live on the box: each row's header bytes padded to
/// `general.alignment`, fetched verbatim by range from the row's revision,
/// then the file truncated (sparse) to its real size, every tensor byte
/// zero. The writer is `tools/type-census.py --stubs`. A regenerated census
/// is a new directory and a new [`MANIFEST`], changed in one commit.
pub const ROOT: &str = "/models/census-stubs-a134239f";

/// The nine columns the header line names, each once. The format is fixed
/// with the writer, so a header that names anything else is `Malformed`.
const COLUMNS: [&str; 9] = [
    "org",
    "repo",
    "revision",
    "set",
    "file",
    "file_bytes",
    "file_sha256",
    "header_bytes",
    "header_sha256",
];

/// One shard's manifest row: the file inside its repo, its real length and
/// LFS sha256, and the verbatim prefix the stub holds with its digest.
#[derive(Debug, Clone)]
pub struct Stub {
    /// The path inside the repo; it may carry the shard's directory.
    pub file: String,
    /// The real file's length in bytes; the stub's length equals it.
    pub file_bytes: u64,
    /// The real file's LFS sha256. Kept, never checked: the stub is not the
    /// real file.
    pub file_sha256: String,
    /// The tensor data offset: the length of the verbatim prefix.
    pub header_bytes: u64,
    /// The sha256 of the real file's bytes `[0, header_bytes)`, which the
    /// stub's prefix must equal.
    pub header_sha256: String,
}

/// One model's rows: the sets of one `(org, repo, dir, set)`, every shard
/// once, at one pinned revision.
#[derive(Debug, Clone)]
pub struct StubSet {
    pub org: String,
    pub repo: String,
    /// The part of the shards' `file` before its last `/`, "" at the repo
    /// root. Rows of one stem in two directories are two sets.
    pub dir: String,
    /// The pinned revision every shard was fetched at.
    pub revision: String,
    /// The stem the shards share.
    pub set: String,
    /// The shards in shard order; one entry for a single file.
    pub shards: Vec<Stub>,
}

/// The manifest read and grouped: every set, in the manifest's
/// first-appearance order.
#[derive(Debug, Clone)]
pub struct Manifest {
    sets: Vec<StubSet>,
}

/// A manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.0.display(), self.1)
    }
}

/// One data row parsed, before the sets are grouped.
struct ManifestRow {
    line: usize,
    org: String,
    repo: String,
    set: String,
    revision: String,
    shard: Option<(u32, u32)>,
    stub: Stub,
}

impl Manifest {
    /// Parse and group the manifest at `path`; every manifest-level refusal
    /// happens here. A manifest that is not there, holds no lines or no data
    /// rows is [`RefError::Missing`].
    pub fn read(path: &Path) -> Result<Manifest, RefError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RefError::missing(path, format!("{}: {e}", path.display())))?;
        let mut lines = text.lines().enumerate();
        let (_, header) = lines.next().ok_or_else(|| {
            RefError::missing(
                path,
                format!("{}: no lines — an empty manifest", path.display()),
            )
        })?;
        let cols = header_columns(header, &At(path, 1))?;
        let mut rows = Vec::new();
        for (i, line) in lines {
            let at = At(path, i + 1);
            rows.push(row_of(&cols.row(line, &at)?, &at)?);
        }
        if rows.is_empty() {
            return Err(RefError::missing(
                path,
                format!("{}: no stub rows — a header only", path.display()),
            ));
        }
        // One file in two rows: both lines named.
        let mut first_line: HashMap<(String, String, String), usize> = HashMap::new();
        for r in &rows {
            let key = (r.org.clone(), r.repo.clone(), r.stub.file.clone());
            if let Some(first) = first_line.insert(key, r.line) {
                return Err(RefError::malformed(
                    At(path, r.line),
                    format!(
                        "rows {first} and {} name the same file {:?}",
                        r.line, r.stub.file
                    ),
                ));
            }
        }
        // Group by (org, repo, dir, set), first appearance first.
        let mut sets: Vec<StubSet> = Vec::new();
        let mut groups: Vec<Vec<ManifestRow>> = Vec::new();
        let mut index: HashMap<(String, String, String, String), usize> = HashMap::new();
        for r in rows {
            let dir = dir_of(&r.stub.file);
            let key = (r.org.clone(), r.repo.clone(), dir.clone(), r.set.clone());
            let i = *index.entry(key).or_insert_with(|| {
                sets.push(StubSet {
                    org: r.org.clone(),
                    repo: r.repo.clone(),
                    dir,
                    revision: r.revision.clone(),
                    set: r.set.clone(),
                    shards: Vec::new(),
                });
                groups.push(Vec::new());
                sets.len() - 1
            });
            groups[i].push(r);
        }
        for (i, g) in groups.into_iter().enumerate() {
            finish_set(&mut sets[i], g)?;
        }
        Ok(Manifest { sets })
    }

    /// Every set, in the manifest's first-appearance order.
    pub fn sets(&self) -> &[StubSet] {
        &self.sets
    }
}

impl StubSet {
    /// `<org>/<repo>/<dir>/<set>`, or `<org>/<repo>/<set>` at the repo root —
    /// the name every refusal uses.
    pub fn name(&self) -> String {
        if self.dir.is_empty() {
            format!("{}/{}/{}", self.org, self.repo, self.set)
        } else {
            format!("{}/{}/{}/{}", self.org, self.repo, self.dir, self.set)
        }
    }

    /// Each shard's stub path under `root`, in shard order.
    pub fn paths(&self, root: &Path) -> Vec<PathBuf> {
        self.shards
            .iter()
            .map(|s| self.stub_path(root, s))
            .collect()
    }

    /// `<root>/<org>/<repo>/<file>`.
    fn stub_path(&self, root: &Path, s: &Stub) -> PathBuf {
        root.join(&self.org).join(&self.repo).join(&s.file)
    }

    /// Check every shard's stub under `root` against its row: there
    /// ([`RefError::Missing`]), of the row's length, and hashing to the
    /// row's digest over its first `header_bytes` bytes
    /// ([`RefError::StubDiffers`]). The first shard's path when all hold.
    pub fn check(&self, root: &Path) -> Result<PathBuf, RefError> {
        let mut first = None;
        for (i, shard) in self.shards.iter().enumerate() {
            let path = self.stub_path(root, shard);
            let mut f = std::fs::File::open(&path)
                .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?;
            let len = f
                .metadata()
                .map_err(|e| RefError::missing(&path, format!("{}: {e}", path.display())))?
                .len();
            if len != shard.file_bytes {
                return Err(RefError::StubDiffers {
                    stub: path,
                    what: "file_bytes",
                    want: shard.file_bytes.to_string(),
                    got: len.to_string(),
                });
            }
            let got = header_digest(&mut f, shard.header_bytes, &path)?;
            if got != shard.header_sha256 {
                return Err(RefError::StubDiffers {
                    stub: path,
                    what: "header_sha256",
                    want: shard.header_sha256.clone(),
                    got,
                });
            }
            if i == 0 {
                first = Some(path);
            }
        }
        Ok(first.expect("every set holds at least one shard: a manifest without rows is refused"))
    }
}

/// The header line as [`Columns`]: the nine names, each once, nothing else —
/// a column missing, named twice or unknown is `Malformed` at line 1.
fn header_columns(header: &str, at: &At) -> Result<Columns, RefError> {
    let fields: Vec<&str> = header.split('\t').collect();
    for name in COLUMNS {
        match (
            fields.iter().position(|f| *f == name),
            fields.iter().rposition(|f| *f == name),
        ) {
            (None, _) => {
                return Err(RefError::malformed(
                    at,
                    format!("the header names no {name} column"),
                ));
            }
            (Some(a), Some(b)) if a != b => {
                return Err(RefError::malformed(
                    at,
                    format!("the header names {name} twice"),
                ));
            }
            _ => {}
        }
    }
    for f in &fields {
        if !COLUMNS.contains(f) {
            return Err(RefError::malformed(
                at,
                format!("the header names {f:?}, not one of the nine columns"),
            ));
        }
    }
    Ok(Columns::new(fields[0], fields[1..].iter().copied()))
}

/// One data row by its column names: the per-row refusals (the widths are
/// [`Columns`]'s) happen here.
fn row_of(row: &crate::columns::Row, at: &At) -> Result<ManifestRow, RefError> {
    let org = row.text("org")?.to_string();
    let repo = row.text("repo")?.to_string();
    let revision = row.text("revision")?.to_string();
    let set = row.text("set")?.to_string();
    let file = row.text("file")?.to_string();
    let file_bytes: u64 = row.parse("file_bytes")?;
    let file_sha256 = row.text("file_sha256")?.to_string();
    let header_bytes: u64 = row.parse("header_bytes")?;
    let header_sha256 = row.text("header_sha256")?.to_string();
    check_component(&org, "org", at)?;
    check_component(&repo, "repo", at)?;
    check_hex(&revision, 40, "revision", at)?;
    if set.is_empty() {
        return Err(RefError::malformed(at, "an empty set"));
    }
    check_file(&file, at)?;
    check_hex(&file_sha256, 64, "file_sha256", at)?;
    check_hex(&header_sha256, 64, "header_sha256", at)?;
    if header_bytes == 0 {
        return Err(RefError::malformed(
            at,
            "header_bytes 0: the header is never empty",
        ));
    }
    if header_bytes > file_bytes {
        return Err(RefError::malformed(
            at,
            format!("header_bytes {header_bytes} > file_bytes {file_bytes}"),
        ));
    }
    let without_ext = &file[..file.len() - ".gguf".len()];
    let basename = without_ext.rsplit_once('/').map_or(without_ext, |(_, b)| b);
    let (stem, shard) = stem_and_shard(basename);
    if set != stem {
        return Err(RefError::malformed(
            at,
            format!("set {set:?} is not the stem of file {file:?} ({stem:?})"),
        ));
    }
    Ok(ManifestRow {
        line: at.1,
        org,
        repo,
        set,
        revision,
        shard,
        stub: Stub {
            file,
            file_bytes,
            file_sha256,
            header_bytes,
            header_sha256,
        },
    })
}

/// One set's rows into its shards: one revision, and either exactly one row
/// without the shard suffix or every shard 1..=N of one N, each once, shard 1
/// first.
fn finish_set(s: &mut StubSet, rows: Vec<ManifestRow>) -> Result<(), RefError> {
    let name = s.name();
    for r in rows.iter().skip(1) {
        if r.revision != s.revision {
            return Err(RefError::StubSet {
                set: name,
                what: format!(
                    "rows {} and {} name revisions {} and {}",
                    rows[0].line, r.line, s.revision, r.revision
                ),
            });
        }
    }
    let shard_rows: Vec<(u32, u32, usize)> = rows
        .iter()
        .filter_map(|r| r.shard.map(|(n, of)| (n, of, r.line)))
        .collect();
    let plain = rows.len() - shard_rows.len();
    match (plain, shard_rows.len()) {
        (1, 0) => {
            s.shards = vec![rows.into_iter().next().expect("the one plain row").stub];
        }
        (0, _) => {
            let (of0, line0) = (shard_rows[0].1, shard_rows[0].2);
            // Each shard number once — a repeated number is a repeated shard
            // even when its rows disagree about the count.
            let mut seen: HashMap<u32, usize> = HashMap::new();
            for &(n, _, line) in &shard_rows {
                if let Some(first) = seen.insert(n, line) {
                    return Err(RefError::StubSet {
                        set: name,
                        what: format!("shard {n} named by rows {first} and {line}"),
                    });
                }
            }
            for &(n, of, line) in &shard_rows {
                if of != of0 {
                    return Err(RefError::StubSet {
                        set: name,
                        what: format!("rows {line0} and {line} name -of-{of0} and -of-{of}"),
                    });
                }
                if n == 0 || n > of0 {
                    return Err(RefError::StubSet {
                        set: name,
                        what: format!("row {line} names shard {n} of {of0}"),
                    });
                }
            }
            let missing: Vec<u32> = (1..=of0).filter(|n| !seen.contains_key(n)).collect();
            if !missing.is_empty() {
                let names: Vec<String> = missing.iter().map(u32::to_string).collect();
                return Err(RefError::StubSet {
                    set: name,
                    what: format!("no shard {} of {of0}", names.join(" or ")),
                });
            }
            let mut ordered = rows;
            ordered.sort_by_key(|r| r.shard.map_or(u32::MAX, |(n, _)| n));
            s.shards = ordered.into_iter().map(|r| r.stub).collect();
        }
        (p, q) => {
            let what = if q == 0 {
                format!(
                    "{p} rows of one set without the shard suffix: a single-file set has one row"
                )
            } else {
                let plain_line = rows
                    .iter()
                    .find(|r| r.shard.is_none())
                    .expect("the mixed arm holds a plain row")
                    .line;
                let shard_line = rows
                    .iter()
                    .find(|r| r.shard.is_some())
                    .expect("the mixed arm holds a shard row")
                    .line;
                format!(
                    "rows {plain_line} and {shard_line} name a plain file and a shard of one set"
                )
            };
            return Err(RefError::StubSet { set: name, what });
        }
    }
    Ok(())
}

/// The part of `file` before its last `/`, "" at the repo root.
fn dir_of(file: &str) -> String {
    file.rsplit_once('/')
        .map_or_else(String::new, |(d, _)| d.to_string())
}

/// A single path component (`org`, `repo`): not empty, `.`, `..` or a `/` —
/// the stub's path must stay under its root.
fn check_component(v: &str, what: &str, at: &At) -> Result<(), RefError> {
    if v.is_empty() || v == "." || v == ".." || v.contains('/') {
        return Err(RefError::malformed(
            at,
            format!("{what} {v:?}: the stub must stay under its root (one path component)"),
        ));
    }
    Ok(())
}

/// The `file` column: a path inside the repo whose basename ends `.gguf` —
/// no leading `/`, no empty, `.` or `..` component, or the stub's path
/// leaves its root.
fn check_file(file: &str, at: &At) -> Result<(), RefError> {
    if file.starts_with('/') {
        return Err(RefError::malformed(
            at,
            format!("file {file:?} has a leading `/`: the stub must stay under its root"),
        ));
    }
    if !file.ends_with(".gguf") {
        return Err(RefError::malformed(
            at,
            format!("file {file:?} does not end .gguf"),
        ));
    }
    for c in file.split('/') {
        if c.is_empty() {
            return Err(RefError::malformed(
                at,
                format!(
                    "file {file:?} holds an empty component: the stub must stay under its root"
                ),
            ));
        }
        if c == "." || c == ".." {
            return Err(RefError::malformed(
                at,
                format!("file {file:?} holds a `{c}` component: the stub must stay under its root"),
            ));
        }
    }
    Ok(())
}

/// `Malformed` at `at` unless `v`, the `what` column, is `len` lowercase hex
/// digits — a digest the stub cannot match, or a revision that pins
/// nothing.
fn check_hex(v: &str, len: usize, what: &str, at: &At) -> Result<(), RefError> {
    if v.len() == len && v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Ok(());
    }
    Err(RefError::malformed(
        at,
        format!("{what} {v:?} is not {len} lowercase hex digits"),
    ))
}

/// `base` (a basename without `.gguf`) minus its shard suffix
/// `-NNNNN-of-NNNNN` when its five-digit parts are there, with the shard's
/// number and count; else `base` itself and no numbers.
fn stem_and_shard(base: &str) -> (&str, Option<(u32, u32)>) {
    let b = base.as_bytes();
    let l = b.len();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(|c| c.is_ascii_digit());
    if l >= 15
        && b[l - 15] == b'-'
        && digits(l - 14..l - 9)
        && &b[l - 9..l - 5] == b"-of-"
        && digits(l - 4..l)
    {
        let n =
            |r: std::ops::Range<usize>| b[r].iter().fold(0u32, |a, c| a * 10 + u32::from(c - b'0'));
        // The suffix is ASCII, so `l - 15` is a char boundary.
        return (&base[..l - 15], Some((n(l - 14..l - 9), n(l - 4..l))));
    }
    (base, None)
}

/// The sha256 of the file's first `header_bytes` bytes, read into a fixed
/// buffer: the check never reads past them.
fn header_digest(
    f: &mut std::fs::File,
    header_bytes: u64,
    path: &Path,
) -> Result<String, RefError> {
    let mut sha = Sha256::new();
    let mut r = f.take(header_bytes);
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = r
            .read(&mut buf)
            .map_err(|e| RefError::missing(path, format!("{}: {e}", path.display())))?;
        if n == 0 {
            return Ok(lower_hex(&sha.finalize()));
        }
        sha.update(&buf[..n]);
    }
}

/// `bytes` as lowercase hex.
fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    // The clause table: every refusal the manifest (M) and stub (C) checks
    // make, and the test that pins it. A clause with no test here is a check
    // nobody runs.
    //
    // | clause | refusal | test |
    // |---|---|---|
    // | M1 | a header column missing, named twice or unknown | `header_lacking_or_doubling_a_column_is_malformed` |
    // | M2 | a row of another width, its line named | `a_row_of_another_width_is_malformed_naming_the_line` |
    // | M3 | a number that does not parse, `header_bytes` 0 or over `file_bytes` | `a_number_or_header_size_out_of_range_is_malformed` |
    // | M4 | a digest or revision that is not lowercase hex | `a_digest_or_revision_not_lowercase_hex_is_malformed` |
    // | M5 | one file in two rows, both lines named | `a_file_named_twice_is_malformed_naming_both_lines` |
    // | M6 | a set of two revisions | `a_set_of_two_revisions_is_refused` |
    // | M7 | a shard set missing or doubling a shard, of two counts, or a plain file mixed with shards | `an_incomplete_shard_set_is_refused_naming_the_shard` |
    // | M8 | a set that is not the stem of its file; a file not ending `.gguf`; the suffix is five digits each way | `a_set_not_the_stem_of_its_file_is_malformed` |
    // | M9 | a manifest with no rows, no lines or no file | `a_manifest_without_rows_is_missing` |
    // | C1 | a stub that is not there | `a_stub_that_cannot_be_opened_is_missing` |
    // | C2 | a stub of another length, want and got named | `a_stub_of_another_length_differs` |
    // | C3 | a stub whose header digest differs; a byte past the header does not | `a_stub_whose_header_digest_differs_is_refused` |
    // | — | a path that leaves the stub root (`..`, a leading `/`, an `org` or `repo` that is not one component) | `a_component_leaving_the_root_is_malformed` |
    // | — | both kinds of set check green, the first shard returned, sets grouped across interleaved rows | `two_shards_and_a_single_file_check_green` |
    use super::Manifest;
    use crate::RefError;
    use std::path::{Path, PathBuf};

    /// The manifest's header line, its nine columns in the agreed order.
    const H: &str =
        "org\trepo\trevision\tset\tfile\tfile_bytes\tfile_sha256\theader_bytes\theader_sha256";

    /// A revision-shaped string, `c` 40 times.
    fn rev(c: char) -> String {
        c.to_string().repeat(40)
    }

    /// A digest-shaped string, `c` 64 times.
    fn dig(c: char) -> String {
        c.to_string().repeat(64)
    }

    /// One manifest row from its nine fields.
    #[allow(
        clippy::too_many_arguments,
        reason = "the manifest's nine columns, one argument each"
    )]
    fn row(
        org: &str,
        repo: &str,
        revision: &str,
        set: &str,
        file: &str,
        file_bytes: &str,
        file_sha256: &str,
        header_bytes: &str,
        header_sha256: &str,
    ) -> String {
        format!(
            "{org}\t{repo}\t{revision}\t{set}\t{file}\t{file_bytes}\t{file_sha256}\t{header_bytes}\t{header_sha256}"
        )
    }

    /// A well-formed shard row of the set `s` (no directory).
    fn shard(file: &str, revision: &str) -> String {
        row(
            "o",
            "r",
            revision,
            "s",
            file,
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        )
    }

    /// A fresh scratch dir under the temp dir, named by process id and test.
    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bloomery-census-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
        d
    }

    /// The manifest `lines` written at `<dir>/stubs.tsv`.
    fn manifest(dir: &Path, lines: &[String]) -> PathBuf {
        let p = dir.join("stubs.tsv");
        std::fs::write(&p, lines.join("\n") + "\n")
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        p
    }

    /// The refusal of reading the manifest `lines` (its header first). Every
    /// call gets its own scratch dir: the tests run in parallel and would
    /// overwrite each other's manifest.
    fn read_err(lines: Vec<String>) -> RefError {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = scratch(&format!("err{n}"));
        let m = manifest(&dir, &lines);
        let e = Manifest::read(&m).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        e
    }

    /// `lines` with the census header first.
    fn with_header(rows: &[String]) -> Vec<String> {
        std::iter::once(H.to_string())
            .chain(rows.iter().cloned())
            .collect()
    }

    /// sha256 of `bytes` as lowercase hex.
    fn hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let d: [u8; 32] = Sha256::digest(bytes).into();
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A stub at `<root>/<org>/<repo>/<file>`: `file_bytes` long, its first
    /// `header.len()` bytes `header`, the rest zero.
    fn stub(
        root: &Path,
        org: &str,
        repo: &str,
        file: &str,
        header: &[u8],
        file_bytes: u64,
    ) -> PathBuf {
        let p = root.join(org).join(repo).join(file);
        std::fs::create_dir_all(p.parent().unwrap_or_else(|| Path::new("/")))
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut bytes = header.to_vec();
        bytes.resize(file_bytes as usize, 0);
        std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        p
    }

    /// M1: the header names the nine columns, each once, nothing else.
    #[test]
    fn header_lacking_or_doubling_a_column_is_malformed() {
        let short = H.replace("\tfile_sha256", "");
        let e = read_err(vec![short, shard("s-00001-of-00002.gguf", &rev('1'))]).to_string();
        assert!(e.contains("names no file_sha256 column"), "{e}");
        let twice = format!("{H}\tset");
        let e = read_err(vec![twice, shard("s-00001-of-00002.gguf", &rev('1'))]).to_string();
        assert!(e.contains("names set twice"), "{e}");
        let extra = format!("{H}\tsize");
        let e = read_err(vec![extra, shard("s-00001-of-00002.gguf", &rev('1'))]).to_string();
        assert!(e.contains("not one of the nine columns"), "{e}");
    }

    /// M2: a row of another width is Malformed, naming its line.
    #[test]
    fn a_row_of_another_width_is_malformed_naming_the_line() {
        let narrow = [
            "o",
            "r",
            rev('1').as_str(),
            "s",
            "s-00001-of-00002.gguf",
            "100",
            "10",
        ]
        .join("\t");
        let e = read_err(with_header(&[narrow])).to_string();
        assert!(e.contains("stubs.tsv:2:"), "{e}");
        assert!(e.contains("its column line names 9"), "{e}");
        let wide = format!("{}\textra", shard("s-00001-of-00002.gguf", &rev('1')));
        let e = read_err(with_header(&[wide])).to_string();
        assert!(e.contains("stubs.tsv:2:"), "{e}");
    }

    /// M3: the numbers parse, the header is never empty and never longer
    /// than its file.
    #[test]
    fn a_number_or_header_size_out_of_range_is_malformed() {
        let rows = [row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s-00001-of-00002.gguf",
            "12x",
            &dig('f'),
            "10",
            &dig('f'),
        )];
        let e = read_err(with_header(&rows)).to_string();
        assert!(e.contains("file_bytes"), "{e}");
        assert!(e.contains("12x"), "{e}");
        let zero = shard("s-00001-of-00002.gguf", &rev('1')).replacen("\t10\t", "\t0\t", 1);
        let e = read_err(with_header(&[zero])).to_string();
        assert!(e.contains("header_bytes 0"), "{e}");
        let over = shard("s-00001-of-00002.gguf", &rev('1')).replacen("\t10\t", "\t200\t", 1);
        let e = read_err(with_header(&[over])).to_string();
        assert!(e.contains("header_bytes 200 > file_bytes 100"), "{e}");
    }

    /// M4: a digest is 64 lowercase hex, a revision 40.
    #[test]
    fn a_digest_or_revision_not_lowercase_hex_is_malformed() {
        let upper = row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s-00001-of-00002.gguf",
            "100",
            &"F".repeat(64),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[upper])).to_string();
        assert!(e.contains("file_sha256"), "{e}");
        assert!(e.contains("not 64 lowercase hex digits"), "{e}");
        let short = row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s-00001-of-00002.gguf",
            "100",
            &dig('f'),
            "10",
            &dig('f')[..63],
        );
        let e = read_err(with_header(&[short])).to_string();
        assert!(e.contains("header_sha256"), "{e}");
        let bad_rev = row(
            "o",
            "r",
            &"Z".repeat(40),
            "s",
            "s-00001-of-00002.gguf",
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[bad_rev])).to_string();
        assert!(e.contains("revision"), "{e}");
        assert!(e.contains("not 40 lowercase hex digits"), "{e}");
    }

    /// M5: one file in two rows is Malformed, naming both lines.
    #[test]
    fn a_file_named_twice_is_malformed_naming_both_lines() {
        let plain = row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s.gguf",
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[plain.clone(), plain])).to_string();
        assert!(
            e.contains("rows 2 and 3 name the same file \"s.gguf\""),
            "{e}"
        );
    }

    /// M6: a set whose rows name two revisions is refused, the set and both
    /// revisions named.
    #[test]
    fn a_set_of_two_revisions_is_refused() {
        let rows = [
            shard("s-00001-of-00002.gguf", &rev('1')),
            shard("s-00002-of-00002.gguf", &rev('2')),
        ];
        match read_err(with_header(&rows)) {
            RefError::StubSet { set, what } => {
                assert_eq!(set, "o/r/s", "{set}");
                assert!(
                    what.contains(&rev('1')) && what.contains(&rev('2')),
                    "{what}"
                );
            }
            e => panic!("want StubSet, got {e:?}"),
        }
    }

    /// M7: a shard set holds every shard 1..=N of one N, each once; a plain
    /// set holds one row; the two never mix. The missing shards are named.
    #[test]
    fn an_incomplete_shard_set_is_refused_naming_the_shard() {
        let missing = [
            shard("s-00001-of-00003.gguf", &rev('1')),
            shard("s-00003-of-00003.gguf", &rev('1')),
        ];
        match read_err(with_header(&missing)) {
            RefError::StubSet { what, .. } => assert!(what.contains("no shard 2"), "{what}"),
            e => panic!("want StubSet, got {e:?}"),
        }
        // The same shard number twice, whatever count the rows name.
        let twice = [
            shard("s-00001-of-00002.gguf", &rev('1')),
            shard("s-00001-of-00003.gguf", &rev('1')),
        ];
        match read_err(with_header(&twice)) {
            RefError::StubSet { what, .. } => {
                assert!(what.contains("shard 1 named by rows 2 and 3"), "{what}")
            }
            e => panic!("want StubSet, got {e:?}"),
        }
        // Two different shards, of two different counts.
        let counts = [
            shard("s-00001-of-00002.gguf", &rev('1')),
            shard("s-00002-of-00003.gguf", &rev('1')),
        ];
        match read_err(with_header(&counts)) {
            RefError::StubSet { what, .. } => {
                assert!(what.contains("-of-2 and -of-3"), "{what}")
            }
            e => panic!("want StubSet, got {e:?}"),
        }
        // A shard number past its own count.
        let past = [
            shard("s-00004-of-00003.gguf", &rev('1')),
            shard("s-00002-of-00003.gguf", &rev('1')),
        ];
        match read_err(with_header(&past)) {
            RefError::StubSet { what, .. } => {
                assert!(what.contains("names shard 4 of 3"), "{what}")
            }
            e => panic!("want StubSet, got {e:?}"),
        }
        let mixed = [
            row(
                "o",
                "r",
                &rev('1'),
                "s",
                "s.gguf",
                "100",
                &dig('f'),
                "10",
                &dig('f'),
            ),
            shard("s-00001-of-00002.gguf", &rev('1')),
        ];
        match read_err(with_header(&mixed)) {
            RefError::StubSet { what, .. } => {
                assert!(what.contains("a plain file and a shard"), "{what}")
            }
            e => panic!("want StubSet, got {e:?}"),
        }
    }

    /// M8: the set is the stem of its file's basename with
    /// `-NNNNN-of-NNNNN.gguf` (five digits each) or else `.gguf` removed,
    /// and the file ends `.gguf`.
    #[test]
    fn a_set_not_the_stem_of_its_file_is_malformed() {
        let wrong = row(
            "o",
            "r",
            &rev('1'),
            "other",
            "s.gguf",
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[wrong])).to_string();
        assert!(e.contains("is not the stem of file \"s.gguf\""), "{e}");
        let bin = row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s.bin",
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[bin])).to_string();
        assert!(e.contains("does not end .gguf"), "{e}");
        // Four digits each way is not the suffix: the stem keeps it.
        let four = row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s-0001-of-0002.gguf",
            "100",
            &dig('f'),
            "10",
            &dig('f'),
        );
        let e = read_err(with_header(&[four])).to_string();
        assert!(e.contains("is not the stem"), "{e}");
    }

    /// M9: a manifest without rows, without lines or without a file is
    /// Missing, never an empty group list.
    #[test]
    fn a_manifest_without_rows_is_missing() {
        let dir = scratch("m9");
        let header_only = manifest(&dir, &[H.to_string()]);
        assert!(matches!(
            Manifest::read(&header_only),
            Err(RefError::Missing { .. })
        ));
        let empty = dir.join("empty.tsv");
        std::fs::write(&empty, b"").unwrap();
        assert!(matches!(
            Manifest::read(&empty),
            Err(RefError::Missing { .. })
        ));
        assert!(matches!(
            Manifest::read(&dir.join("nope.tsv")),
            Err(RefError::Missing { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// C1: a stub that is not there is Missing, naming its path.
    #[test]
    fn a_stub_that_cannot_be_opened_is_missing() {
        let dir = scratch("c1");
        let header = b"the header".to_vec();
        let rows = [row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s.gguf",
            "100",
            &dig('f'),
            "10",
            &hex(&header),
        )];
        let m = manifest(&dir, &with_header(&rows));
        let sets = Manifest::read(&m).unwrap();
        match sets.sets()[0].check(&dir.join("root")) {
            Err(RefError::Missing { path, .. }) => {
                assert!(path.ends_with("o/r/s.gguf"), "{}", path.display())
            }
            e => panic!("want Missing, got {e:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// C2: a stub of another length differs from its row, want and got
    /// named.
    #[test]
    fn a_stub_of_another_length_differs() {
        let dir = scratch("c2");
        let root = dir.join("root");
        let header = b"the header".to_vec();
        let path = stub(&root, "o", "r", "s.gguf", &header, 100);
        let rows = [row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s.gguf",
            "100",
            &dig('f'),
            "10",
            &hex(&header),
        )];
        let m = manifest(&dir, &with_header(&rows));
        let sets = Manifest::read(&m).unwrap();
        let set = &sets.sets()[0];
        set.check(&root).unwrap();
        std::fs::write(&path, vec![0u8; 99]).unwrap();
        match set.check(&root) {
            Err(RefError::StubDiffers {
                stub,
                what,
                want,
                got,
            }) => {
                assert_eq!(stub, path);
                assert_eq!(what, "file_bytes");
                assert_eq!((want.as_str(), got.as_str()), ("100", "99"));
            }
            e => panic!("want StubDiffers, got {e:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// C3: only the header bytes are hashed — a byte inside them differs, a
    /// byte past them does not.
    #[test]
    fn a_stub_whose_header_digest_differs_is_refused() {
        let dir = scratch("c3");
        let root = dir.join("root");
        let header: Vec<u8> = (0..10u8).collect();
        let path = stub(&root, "o", "r", "s.gguf", &header, 100);
        let rows = [row(
            "o",
            "r",
            &rev('1'),
            "s",
            "s.gguf",
            "100",
            &dig('f'),
            "10",
            &hex(&header),
        )];
        let m = manifest(&dir, &with_header(&rows));
        let sets = Manifest::read(&m).unwrap();
        let set = &sets.sets()[0];
        set.check(&root).unwrap();
        // A byte past the header: the digest of [0, 10) cannot see it.
        let mut body = vec![0u8; 100];
        body[..10].copy_from_slice(&header);
        body[50] = 9;
        std::fs::write(&path, &body).unwrap();
        set.check(&root).unwrap();
        // A byte inside the header.
        body[3] = 99;
        std::fs::write(&path, &body).unwrap();
        match set.check(&root) {
            Err(RefError::StubDiffers {
                stub,
                what,
                want,
                got,
            }) => {
                assert_eq!(stub, path);
                assert_eq!(what, "header_sha256");
                assert_eq!(
                    (want.as_str(), got.as_str()),
                    (hex(&header).as_str(), hex(&body[..10]).as_str())
                );
            }
            e => panic!("want StubDiffers, got {e:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A path that leaves the stub root — a `..` component, a leading `/`,
    /// an empty component, or an `org` or `repo` that is not one component —
    /// is Malformed.
    #[test]
    fn a_component_leaving_the_root_is_malformed() {
        let cases = [
            (
                "dotdot",
                row(
                    "o",
                    "r",
                    &rev('1'),
                    "s",
                    "../s.gguf",
                    "100",
                    &dig('f'),
                    "10",
                    &dig('f'),
                ),
            ),
            (
                "leading",
                row(
                    "o",
                    "r",
                    &rev('1'),
                    "s",
                    "/abs/s.gguf",
                    "100",
                    &dig('f'),
                    "10",
                    &dig('f'),
                ),
            ),
            (
                "empty",
                row(
                    "o",
                    "r",
                    &rev('1'),
                    "s",
                    "d//s.gguf",
                    "100",
                    &dig('f'),
                    "10",
                    &dig('f'),
                ),
            ),
            (
                "org",
                row(
                    "a/b",
                    "r",
                    &rev('1'),
                    "s",
                    "s.gguf",
                    "100",
                    &dig('f'),
                    "10",
                    &dig('f'),
                ),
            ),
            (
                "repo",
                row(
                    "o",
                    "..",
                    &rev('1'),
                    "s",
                    "s.gguf",
                    "100",
                    &dig('f'),
                    "10",
                    &dig('f'),
                ),
            ),
        ];
        for (what, r) in cases {
            let e = read_err(with_header(&[r])).to_string();
            assert!(
                e.contains("the stub must stay under its root"),
                "{what}: {e}"
            );
        }
    }

    /// The happy path: a two-shard set in a directory and a single-file set
    /// at a repo root, their rows interleaved, group by first appearance,
    /// order the shards by number, and check green — the first shard's path
    /// back. One header spans the reader's buffer, so its digest takes more
    /// than one read.
    #[test]
    fn two_shards_and_a_single_file_check_green() {
        let dir = scratch("green");
        let root = dir.join("root");
        let h1: Vec<u8> = (0..70_008u32).map(|i| (i % 251) as u8).collect();
        let h2 = vec![7u8; 512];
        let h3 = b"a third header".to_vec();
        let p1 = stub(&root, "o1", "r1", "d/s1-00001-of-00002.gguf", &h1, 80_000);
        let p2 = stub(&root, "o1", "r1", "d/s1-00002-of-00002.gguf", &h2, 4096);
        let p3 = stub(&root, "o2", "r2", "s2.gguf", &h3, 4096);
        let rows = [
            row(
                "o1",
                "r1",
                &rev('1'),
                "s1",
                "d/s1-00001-of-00002.gguf",
                "80000",
                &dig('f'),
                "70008",
                &hex(&h1),
            ),
            row(
                "o2",
                "r2",
                &rev('2'),
                "s2",
                "s2.gguf",
                "4096",
                &dig('f'),
                "14",
                &hex(&h3),
            ),
            row(
                "o1",
                "r1",
                &rev('1'),
                "s1",
                "d/s1-00002-of-00002.gguf",
                "4096",
                &dig('f'),
                "512",
                &hex(&h2),
            ),
        ];
        let m = manifest(&dir, &with_header(&rows));
        let sets = Manifest::read(&m).unwrap();
        assert_eq!(sets.sets().len(), 2);
        let s1 = &sets.sets()[0];
        assert_eq!(s1.name(), "o1/r1/d/s1");
        assert_eq!(
            (s1.dir.as_str(), s1.revision.as_str()),
            ("d", rev('1').as_str())
        );
        assert_eq!(s1.shards.len(), 2);
        // Shard 1 first, though the single-file set's row sits between the
        // shard rows and shard 2 came last.
        assert_eq!(s1.shards[0].file, "d/s1-00001-of-00002.gguf");
        assert_eq!(s1.shards[0].header_bytes, 70008);
        assert_eq!(s1.paths(&root), vec![p1.clone(), p2]);
        assert_eq!(s1.check(&root).unwrap(), p1);
        let s2 = &sets.sets()[1];
        assert_eq!(s2.name(), "o2/r2/s2");
        assert_eq!(s2.dir, "");
        assert_eq!(s2.check(&root).unwrap(), p3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
