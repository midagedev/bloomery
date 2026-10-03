//! A model file from a Hugging Face repo: `--hf <repo>[:<quant>]` as
//! llama.cpp spells it, resolved to one GGUF set of the repo's listing and
//! fetched into a local cache.
//!
//! The pure part is here: the repo string ([`RepoRef`]), the listing the API
//! answers (`api/models/<repo>/tree/main?recursive=true`, [`listing`]), the
//! GGUF sets in it ([`gguf_sets`]) and the one a quant tag picks ([`pick`]),
//! the files named exactly ([`exact`]), the cache paths ([`cache_root`],
//! [`cache_path`]), where a model file comes from ([`source`]) and what a
//! model card says ([`card`]). The network part, [`fetch`], is `curl`, the
//! checks of what it wrote, and the cache standing in when the network
//! refuses the listing.
//!
//! The picking follows llama.cpp's `common/download.cpp`: a split set
//! (`-00001-of-0000N.gguf`) is one set by its prefix and count, and the
//! files a model is not (`mmproj`, `imatrix`, `mtp-`, `eagle3-`, `dflash-`,
//! `dspark-` in the name) are left out. It is stricter where llama.cpp takes
//! the first match: a tag that matches two sets, a set missing a shard, and
//! no tag on a repo of several sets are each refused by name, with the
//! candidates.

pub mod card;
pub mod fetch;
pub mod source;

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Why a repo, a listing, a pick, a fetch or a model source was refused.
#[derive(Debug, thiserror::Error)]
pub enum HfError {
    #[error(
        "--hf {0:?}: a repo is <owner>/<name>[:<quant>], each part of letters, digits, '-', '_' and '.'"
    )]
    Repo(String),
    #[error("{repo}: the API's listing is not one this reads: {why}")]
    Listing { repo: String, why: String },
    #[error("{repo}: the listing names {path:?}, which is not a relative path inside the repo")]
    Path { repo: String, path: String },
    #[error("{repo} holds no GGUF model file")]
    NoGguf { repo: String },
    #[error("{repo} holds {} GGUF sets and no quant was given; pick one with --hf {repo}:<quant>: {}", .sets.len(), .sets.join(", "))]
    NoQuant { repo: String, sets: Vec<String> },
    #[error("{repo}: no GGUF set carries the quant {quant:?}; its quants: {}", .tags.join(", "))]
    NoMatch {
        repo: String,
        quant: String,
        tags: Vec<String>,
    },
    #[error("{repo}: the quant {quant:?} matches {} sets: {}", .sets.len(), .sets.join(", "))]
    TwoMatches {
        repo: String,
        quant: String,
        sets: Vec<String>,
    },
    #[error("{repo}: the split set {set} of {count} shards lacks shard(s) {missing:?}")]
    IncompleteSet {
        repo: String,
        set: String,
        count: u32,
        missing: Vec<u32>,
    },
    #[error("{repo} holds no file {name:?}")]
    NoFile { repo: String, name: String },
    #[error("no cache directory: set BLOOMERY_CACHE, or HOME for ~/.cache/bloomery/hf")]
    NoCache,
    #[error("{0}")]
    Source(String),
    #[error("{file}: {bytes} bytes on disk, and the repo's file has {want}")]
    Size { file: String, bytes: u64, want: u64 },
    #[error("{file}: its {check} is {got}, and the repo's is {want}; the partial file was removed")]
    Hash {
        file: String,
        check: Check,
        got: String,
        want: String,
    },
    #[error("curl {what}: {why}")]
    Curl { what: String, why: String },
    #[error("curl {what}: {why} (the network, not the hub, refused)")]
    Network { what: String, why: String },
    #[error(
        "{repo}: the listing failed at the network ({why}), and the cache cannot stand in: {cache}"
    )]
    Offline {
        repo: String,
        why: String,
        cache: Box<HfError>,
    },
    #[error("{path}: {err}")]
    Io {
        path: String,
        #[source]
        err: std::io::Error,
    },
}

impl HfError {
    fn io(path: &Path, err: std::io::Error) -> HfError {
        HfError::Io {
            path: path.display().to_string(),
            err,
        }
    }
}

/// `<owner>/<name>` and the quant tag after a `:`, if one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoRef {
    pub repo: String,
    pub quant: Option<String>,
}

impl RepoRef {
    /// `owner/name[:quant]`: exactly one `/`, both sides and the quant
    /// non-empty, and every part of letters, digits, `-`, `_` and `.` (no
    /// part `.` or `..`). Anything else is [`HfError::Repo`].
    pub fn parse(s: &str) -> Result<RepoRef, HfError> {
        let bad = || HfError::Repo(s.to_owned());
        let (repo, quant) = match s.split_once(':') {
            Some((r, q)) => (r, Some(q)),
            None => (s, None),
        };
        let word = |p: &str| {
            !p.is_empty()
                && p != "."
                && p != ".."
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        };
        let (owner, name) = repo.split_once('/').ok_or_else(bad)?;
        if !word(owner) || !word(name) || quant.is_some_and(|q| !word(q)) {
            return Err(bad());
        }
        Ok(RepoRef {
            repo: repo.to_owned(),
            quant: quant.map(str::to_owned),
        })
    }
}

impl fmt::Display for RepoRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.quant {
            Some(q) => write!(f, "{}:{q}", self.repo),
            None => f.write_str(&self.repo),
        }
    }
}

/// How a file's content is checked: the LFS sha256 the API gives a large
/// file, or the git blob sha1 (`sha1("blob <size>\0" + content)`, the
/// entry's `oid`) of a file stored in git itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Sha256,
    GitSha1,
}

impl Check {
    pub fn name(self) -> &'static str {
        match self {
            Check::Sha256 => "sha256",
            Check::GitSha1 => "git-sha1",
        }
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A file of the listing: its path in the repo, its size, and the digest
/// its content is checked against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub path: String,
    pub size: u64,
    pub check: Check,
    /// Lowercase hex: the LFS sha256, or the git blob sha1.
    pub digest: String,
}

impl Entry {
    /// The file name, after the last `/`.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

fn hex_of(v: Option<&Value>, len: usize) -> Option<String> {
    let s = v?.as_str()?;
    (s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())).then(|| s.to_ascii_lowercase())
}

/// The files of one page of the tree listing `json` of `repo` (directories
/// left out). A file entry with no path, size or digest, an LFS size other
/// than the file's, or JSON that is not a list is [`HfError::Listing`].
pub fn listing(repo: &str, json: &str) -> Result<Vec<Entry>, HfError> {
    let bad = |why: String| HfError::Listing {
        repo: repo.to_owned(),
        why,
    };
    let v: Value = serde_json::from_str(json).map_err(|e| bad(e.to_string()))?;
    let items = v
        .as_array()
        .ok_or_else(|| bad("the answer is not a list".to_owned()))?;
    let mut out = Vec::new();
    for item in items {
        let ty = item.get("type").and_then(Value::as_str);
        if ty == Some("directory") {
            continue;
        }
        if ty != Some("file") {
            return Err(bad(format!("an entry of type {ty:?}: {item}")));
        }
        let path = item
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(format!("a file with no path: {item}")))?;
        let size = item
            .get("size")
            .and_then(Value::as_u64)
            .ok_or_else(|| bad(format!("{path} has no size")))?;
        let (check, digest) = match item.get("lfs") {
            Some(lfs) => {
                if lfs.get("size").and_then(Value::as_u64) != Some(size) {
                    return Err(bad(format!("{path}: its LFS size is not its size {size}")));
                }
                let d = hex_of(lfs.get("oid"), 64)
                    .ok_or_else(|| bad(format!("{path}: its LFS oid is not a sha256")))?;
                (Check::Sha256, d)
            }
            None => {
                let d = hex_of(item.get("oid"), 40)
                    .ok_or_else(|| bad(format!("{path}: its oid is not a git sha1")))?;
                (Check::GitSha1, d)
            }
        };
        out.push(Entry {
            path: path.to_owned(),
            size,
            check,
            digest,
        });
    }
    Ok(out)
}

/// The next page of a listing: the `rel="next"` URL of the `Link` header
/// among `headers` (an HTTP response's header lines), if one.
pub fn next_link(headers: &str) -> Option<String> {
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("link") {
            continue;
        }
        for link in value.split(',') {
            let mut parts = link.split(';');
            let url = parts.next()?.trim();
            if parts.any(|p| p.trim().replace(' ', "") == "rel=\"next\"") {
                return Some(url.trim_start_matches('<').trim_end_matches('>').to_owned());
            }
        }
    }
    None
}

/// A GGUF model set: one file, or the shards of a split set in shard order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufSet {
    /// The set's name: the path less `.gguf` and the split suffix.
    pub name: String,
    /// llama.cpp's tag of the set: the run of letters, digits and `_` after
    /// the name's last `-` or `.`, upper-cased; empty when there is none.
    pub tag: String,
    pub files: Vec<Entry>,
    /// A split set's shard count (1 for one file).
    pub count: u32,
}

impl GgufSet {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// The set's name and tag, as a refusal lists it.
    fn shown(&self) -> String {
        if self.count > 1 {
            format!("{} ({} shards)", self.name, self.count)
        } else {
            self.name.clone()
        }
    }
}

/// Whether `name` (a file name) is a model GGUF: llama.cpp's
/// `gguf_filename_is_model`.
fn is_model(name: &str) -> bool {
    name.ends_with(".gguf")
        && ["mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"]
            .iter()
            .all(|w| !name.contains(w))
}

/// `stem` (a path less `.gguf`) split as llama.cpp's `-NNNNN-of-NNNNN`
/// suffix reads it: the prefix, the shard index and the count; `None` for a
/// file that is not a shard.
fn split_of(stem: &str) -> Option<(&str, u32, u32)> {
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    let five = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !five(index) || !five(count) || prefix.is_empty() {
        return None;
    }
    Some((prefix, index.parse().ok()?, count.parse().ok()?))
}

/// llama.cpp's `re_tag`: `[-.]([A-Za-z0-9_]+)$` of `name`, upper-cased.
fn tag_of(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name);
    let tail = base
        .bytes()
        .rev()
        .take_while(|&b| b.is_ascii_alphanumeric() || b == b'_')
        .count();
    let at = base.len() - tail;
    if tail == 0 || at == 0 || !matches!(base.as_bytes()[at - 1], b'-' | b'.') {
        return String::new();
    }
    base[at..].to_ascii_uppercase()
}

/// The GGUF model sets of `entries`, in the listing's order of their first
/// file. A split set lacking a shard (or holding one twice) is
/// [`HfError::IncompleteSet`].
pub fn gguf_sets(repo: &str, entries: &[Entry]) -> Result<Vec<GgufSet>, HfError> {
    let mut sets: Vec<GgufSet> = Vec::new();
    for e in entries {
        if !is_model(e.name()) {
            continue;
        }
        let stem = &e.path[..e.path.len() - ".gguf".len()];
        let (name, _, count) = split_of(stem).unwrap_or((stem, 1, 1));
        match sets
            .iter_mut()
            .find(|s| s.name == name && s.count == count && count > 1)
        {
            Some(s) => s.files.push(e.clone()),
            None => sets.push(GgufSet {
                name: name.to_owned(),
                tag: tag_of(name),
                files: vec![e.clone()],
                count,
            }),
        }
    }
    for s in &mut sets {
        if s.count == 1 {
            continue;
        }
        let index = |e: &Entry| split_of(&e.path[..e.path.len() - 5]).map_or(0, |(_, i, _)| i);
        s.files.sort_by_key(index);
        let have: Vec<u32> = s.files.iter().map(index).collect();
        let missing: Vec<u32> = (1..=s.count).filter(|i| !have.contains(i)).collect();
        if !missing.is_empty() || have.len() != s.count as usize {
            return Err(HfError::IncompleteSet {
                repo: repo.to_owned(),
                set: s.name.clone(),
                count: s.count,
                missing,
            });
        }
    }
    Ok(sets)
}

/// Whether `quant` names the set `name`: case-insensitive, `quant` a run
/// of the file name (after the last `/`) that starts the name or follows
/// one of `-`, `.`, `_`, and ends it or is followed by `-` or `.` — as
/// llama.cpp's `<tag>[.-]` ends it, so `Q6_K` does not name `Q6_K_L` and
/// `Q4_K_M` does not name `IQ4_K_M`.
fn names(quant: &str, name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name).to_ascii_lowercase();
    let q = quant.to_ascii_lowercase();
    if q.is_empty() {
        return false;
    }
    let b = base.as_bytes();
    let mut from = 0;
    while let Some(i) = base[from..].find(&q) {
        let at = from + i;
        let end = at + q.len();
        let left = at == 0 || matches!(b[at - 1], b'-' | b'.' | b'_');
        let right = end == b.len() || matches!(b[end], b'-' | b'.');
        if left && right {
            return true;
        }
        from = at + 1;
    }
    false
}

/// The set `quant` picks among `sets` of `repo`: the one set it names
/// ([`names`]); with no quant, the repo's only set. No set, no match, two
/// matches and no quant on several sets are each refused by name with the
/// candidates.
pub fn pick<'a>(
    repo: &str,
    sets: &'a [GgufSet],
    quant: Option<&str>,
) -> Result<&'a GgufSet, HfError> {
    if sets.is_empty() {
        return Err(HfError::NoGguf {
            repo: repo.to_owned(),
        });
    }
    let Some(q) = quant else {
        return match sets {
            [one] => Ok(one),
            _ => Err(HfError::NoQuant {
                repo: repo.to_owned(),
                sets: sets.iter().map(GgufSet::shown).collect(),
            }),
        };
    };
    let hits: Vec<&GgufSet> = sets.iter().filter(|s| names(q, &s.name)).collect();
    match hits.as_slice() {
        [one] => Ok(one),
        [] => Err(HfError::NoMatch {
            repo: repo.to_owned(),
            quant: q.to_owned(),
            tags: sets
                .iter()
                .map(|s| {
                    if s.tag.is_empty() {
                        s.name.clone()
                    } else {
                        s.tag.clone()
                    }
                })
                .collect(),
        }),
        _ => Err(HfError::TwoMatches {
            repo: repo.to_owned(),
            quant: q.to_owned(),
            sets: hits.iter().map(|s| s.shown()).collect(),
        }),
    }
}

/// The entries of `repo` at exactly the paths `paths`, in that order; a
/// path the listing lacks is [`HfError::NoFile`].
pub fn exact<'a>(
    repo: &str,
    entries: &'a [Entry],
    paths: &[&str],
) -> Result<Vec<&'a Entry>, HfError> {
    paths
        .iter()
        .map(|p| {
            entries
                .iter()
                .find(|e| e.path == *p)
                .ok_or_else(|| HfError::NoFile {
                    repo: repo.to_owned(),
                    name: (*p).to_owned(),
                })
        })
        .collect()
}

/// The cache's root: `$BLOOMERY_CACHE` (`cache`) when set and non-empty,
/// else `$HOME/.cache/bloomery/hf`; neither is [`HfError::NoCache`].
pub fn cache_root(cache: Option<OsString>, home: Option<OsString>) -> Result<PathBuf, HfError> {
    match (cache, home) {
        (Some(c), _) if !c.is_empty() => Ok(PathBuf::from(c)),
        (_, Some(h)) if !h.is_empty() => Ok(PathBuf::from(h).join(".cache/bloomery/hf")),
        _ => Err(HfError::NoCache),
    }
}

/// Where `path` of `repo` lives under `root`: `<root>/<owner>/<name>/<path>`,
/// a split set's subfolder kept so its shards stay siblings. A path that is
/// empty, absolute, or has an empty, `.` or `..` part or a backslash is
/// [`HfError::Path`].
pub fn cache_path(root: &Path, repo: &str, path: &str) -> Result<PathBuf, HfError> {
    let bad = || HfError::Path {
        repo: repo.to_owned(),
        path: path.to_owned(),
    };
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return Err(bad());
    }
    let mut out = root.join(repo);
    for part in path.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return Err(bad());
        }
        out.push(part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    const CLEF: &str = "bartowski/Cloudflare_clef-flash-GGUF";
    const SPLIT: &str = "bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF";

    fn clef_sets() -> Vec<GgufSet> {
        let e = listing(CLEF, &fixture("bartowski-clef-flash-gguf.json")).expect("listing");
        gguf_sets(CLEF, &e).expect("sets")
    }

    #[test]
    fn repo_strings_parse_as_llama_cpp_splits_them() {
        let r = RepoRef::parse("bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M").expect("parse");
        assert_eq!(r.repo, CLEF);
        assert_eq!(r.quant.as_deref(), Some("Q5_K_M"));
        assert_eq!(r.to_string(), "bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M");
        let r = RepoRef::parse("Cloudflare/clef-flash").expect("parse");
        assert_eq!(r.quant, None);
        for bad in [
            "", "noslash", "a/b/c", "/b", "a/", "a/b:", "a/b:c:d", "a/..", "a b/c", "a/b:Q4 K",
        ] {
            assert!(
                matches!(RepoRef::parse(bad), Err(HfError::Repo(_))),
                "{bad:?} parsed"
            );
        }
    }

    #[test]
    fn the_clef_listing_reads_its_files_and_digests() {
        let e = listing(CLEF, &fixture("bartowski-clef-flash-gguf.json")).expect("listing");
        let q3 = e
            .iter()
            .find(|e| e.path == "Cloudflare_clef-flash-Q3_K_S.gguf")
            .expect("Q3_K_S");
        assert_eq!(q3.size, 4_260_308_288);
        assert_eq!(q3.check, Check::Sha256);
        assert_eq!(
            q3.digest,
            "a3da346b670ba23230efacb0ba4ad7f6791de97d3fa3b52f2b95df76d5393e9b"
        );
        let readme = e.iter().find(|e| e.path == "README.md").expect("README");
        assert_eq!(readme.check, Check::GitSha1);
        assert!(
            e.iter().all(|e| e.path != "layouts"),
            "a directory was read"
        );
    }

    #[test]
    fn mmproj_and_imatrix_are_not_sets() {
        let sets = clef_sets();
        assert_eq!(sets.len(), 22, "{sets:#?}");
        assert!(sets.iter().all(|s| !s.name.contains("mmproj")));
        assert!(sets.iter().all(|s| !s.name.contains("imatrix")));
        // `bf16` names the model and, but for the filter, mmproj-…-bf16 too.
        let s = pick(CLEF, &sets, Some("bf16")).expect("bf16");
        assert_eq!(s.name, "Cloudflare_clef-flash-bf16");
    }

    #[test]
    fn a_tag_picks_one_set_case_insensitively() {
        let sets = clef_sets();
        for (q, name) in [
            ("Q3_K_S", "Cloudflare_clef-flash-Q3_K_S"),
            ("q5_k_m", "Cloudflare_clef-flash-Q5_K_M"),
            ("Q6_K", "Cloudflare_clef-flash-Q6_K"),
            ("Q6_K_L", "Cloudflare_clef-flash-Q6_K_L"),
            ("IQ4_XS", "Cloudflare_clef-flash-IQ4_XS"),
        ] {
            let s = pick(CLEF, &sets, Some(q)).unwrap_or_else(|e| panic!("{q}: {e}"));
            assert_eq!(s.name, name, "{q}");
            assert_eq!(s.files.len(), 1);
        }
    }

    #[test]
    fn zero_and_two_matches_are_refused_with_the_candidates() {
        let sets = clef_sets();
        match pick(CLEF, &sets, Some("Q7_K")) {
            Err(HfError::NoMatch { tags, .. }) => {
                assert!(tags.contains(&"Q3_K_S".to_owned()), "{tags:?}");
                assert_eq!(tags.len(), 22);
            }
            other => panic!("Q7_K: {other:?}"),
        }
        // `K_M` sits between separators in Q3_K_M, Q4_K_M and Q5_K_M.
        match pick(CLEF, &sets, Some("K_M")) {
            Err(HfError::TwoMatches { sets, .. }) => assert_eq!(sets.len(), 3, "{sets:?}"),
            other => panic!("K_M: {other:?}"),
        }
        // A bare `Q4_K` names no set: `_M` after it is not a separator.
        assert!(matches!(
            pick(CLEF, &sets, Some("Q4_K")),
            Err(HfError::NoMatch { .. })
        ));
        // Several sets and no quant.
        match pick(CLEF, &sets, None) {
            Err(e @ HfError::NoQuant { .. }) => {
                assert!(e.to_string().contains("Cloudflare_clef-flash-Q8_0"), "{e}")
            }
            other => panic!("no quant: {other:?}"),
        }
    }

    #[test]
    fn a_split_set_is_one_set_of_every_shard() {
        let e = listing(
            SPLIT,
            &fixture("bartowski-qwen3-235b-a22b-instruct-2507-gguf.json"),
        )
        .expect("listing");
        let sets = gguf_sets(SPLIT, &e).expect("sets");
        let s = pick(SPLIT, &sets, Some("Q4_K_M")).expect("Q4_K_M");
        assert_eq!(s.count, 4);
        assert_eq!(s.tag, "Q4_K_M");
        let names: Vec<&str> = s.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            names,
            (1..=4)
                .map(|i| format!(
                    "Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M/Qwen_Qwen3-235B-A22B-Instruct-2507-Q4_K_M-{i:05}-of-00004.gguf"
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            s.bytes(),
            39_856_336_448 + 39_847_230_560 * 2 + 23_096_107_744
        );
        // Q2_K does not name Q2_K_L's shards.
        let q2 = pick(SPLIT, &sets, Some("Q2_K")).expect("Q2_K");
        assert_eq!(q2.files.len(), 3);
        assert!(q2.files.iter().all(|f| !f.path.contains("Q2_K_L")));
    }

    #[test]
    fn a_split_set_missing_a_shard_is_refused() {
        let json = fixture("bartowski-qwen3-235b-a22b-instruct-2507-gguf.json").replace(
            "Q4_K_M-00003-of-00004.gguf",
            "Q4_K_M-00003-of-00004.gguf.bak",
        );
        let e = listing(SPLIT, &json).expect("listing");
        match gguf_sets(SPLIT, &e) {
            Err(HfError::IncompleteSet { missing, count, .. }) => {
                assert_eq!((missing, count), (vec![3], 4));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn one_set_needs_no_quant() {
        let json = r#"[
            {"type":"file","path":"README.md","size":3,"oid":"0123456789012345678901234567890123456789"},
            {"type":"file","path":"m-Q4_K_M.gguf","size":10,"lfs":{"oid":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":10}},
            {"type":"file","path":"mmproj-m-f16.gguf","size":10,"lfs":{"oid":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size":10}}
        ]"#;
        let e = listing("a/b", json).expect("listing");
        let sets = gguf_sets("a/b", &e).expect("sets");
        assert_eq!(pick("a/b", &sets, None).expect("only").name, "m-Q4_K_M");
        let none = listing("a/b", "[]").expect("listing");
        assert!(matches!(
            pick("a/b", &gguf_sets("a/b", &none).expect("sets"), None),
            Err(HfError::NoGguf { .. })
        ));
    }

    #[test]
    fn a_listing_entry_without_a_digest_is_refused() {
        for json in [
            r#"{"type":"file"}"#,
            r#"[{"type":"file","path":"x.gguf","size":10}]"#,
            r#"[{"type":"file","path":"x.gguf","size":10,"lfs":{"oid":"ab","size":10}}]"#,
            r#"[{"type":"file","path":"x.gguf","size":10,"lfs":{"oid":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":11}}]"#,
            r#"[{"type":"submodule","path":"x"}]"#,
        ] {
            assert!(
                matches!(listing("a/b", json), Err(HfError::Listing { .. })),
                "{json}"
            );
        }
    }

    #[test]
    fn exact_names_the_head_files() {
        let e = listing(
            "Cloudflare/clef-flash",
            &fixture("cloudflare-clef-flash.json"),
        )
        .expect("listing");
        let got = exact(
            "Cloudflare/clef-flash",
            &e,
            &["joint_head.safetensors", "joint_head_config.json"],
        )
        .expect("both");
        assert_eq!(got[0].check, Check::Sha256);
        assert_eq!(got[1].check, Check::GitSha1);
        assert_eq!(got[1].size, 119);
        assert!(matches!(
            exact("Cloudflare/clef-flash", &e, &["nope.json"]),
            Err(HfError::NoFile { .. })
        ));
    }

    #[test]
    fn cache_paths_keep_the_repo_and_the_subfolder() {
        let root = cache_root(Some("/c".into()), Some("/h".into())).expect("root");
        assert_eq!(root, PathBuf::from("/c"));
        let root = cache_root(Some("".into()), Some("/h".into())).expect("root");
        assert_eq!(root, PathBuf::from("/h/.cache/bloomery/hf"));
        assert!(matches!(cache_root(None, None), Err(HfError::NoCache)));
        let p = cache_path(Path::new("/c"), SPLIT, "Q4_K_M/x-00001-of-00004.gguf").expect("p");
        assert_eq!(
            p,
            PathBuf::from(
                "/c/bartowski/Qwen_Qwen3-235B-A22B-Instruct-2507-GGUF/Q4_K_M/x-00001-of-00004.gguf"
            )
        );
        for bad in ["", "/etc/passwd", "../x", "a/../b", "a//b", "./a", "a\\b"] {
            assert!(
                matches!(
                    cache_path(Path::new("/c"), "a/b", bad),
                    Err(HfError::Path { .. })
                ),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_next_page_comes_from_the_link_header() {
        let h = "HTTP/2 200\r\ncontent-type: application/json\r\nLink: <https://huggingface.co/api/models/a/b/tree/main?recursive=true&cursor=xyz>; rel=\"next\"\r\n\r\n";
        assert_eq!(
            next_link(h).as_deref(),
            Some("https://huggingface.co/api/models/a/b/tree/main?recursive=true&cursor=xyz")
        );
        assert_eq!(next_link("HTTP/2 200\r\ncontent-type: x\r\n"), None);
    }
}
