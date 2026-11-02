//! The set the reader's tests and a family's tests write: one prompt of two fed rows of three values and two draft calls, under
//! the identity a test states.

use crate::RefError;
use crate::family::{Build, Family, Identity};
use std::path::{Path, PathBuf};

/// The row width of the test prompt's `taps.f32`.
pub(crate) const FEAT: usize = 3;

/// The ids a call proposes in the test set.
pub(crate) const N_MAX: usize = 3;

/// What the test family runs and states.
pub(crate) const MODEL: &str = "/models/t/target-00001-of-00002.gguf";
pub(crate) const DRAFT: &str = "/models/t/draft.gguf";
pub(crate) const BUILD: &str = "abcdef012+patch-0123456789abcdef0123456789abcdef";
pub(crate) const ARCH: &str = "test-arch";

fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

fn draft() -> Result<String, RefError> {
    Ok(DRAFT.to_string())
}

/// A family of the reader's own: the files, build and architecture above.
pub(crate) static FAMILY: Family = Family {
    name: "test-lcppdspark",
    sets: &["test-set"],
    resolve: None,
    recipe: "just dump-test-lcppdspark",
    identity: Identity::LcppDsparkCalls,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: Some(draft),
    consumers: &[],
};

/// What a set states about itself.
pub(crate) struct Own<'a> {
    pub model: &'a str,
    pub draft: &'a str,
    pub build: &'a str,
    pub arch: &'a str,
}

/// [`FAMILY`]'s own files, build and architecture.
pub(crate) const OWN: Own<'static> = Own {
    model: MODEL,
    draft: DRAFT,
    build: BUILD,
    arch: ARCH,
};

/// A fresh, empty directory for one test's set; tests of one process run in parallel, so each passes its own `what`.
pub(crate) fn fresh(family: &str, what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-{family}-{what}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

/// Write `bytes` as `dir/name`.
pub(crate) fn put(dir: &Path, name: &str, bytes: impl AsRef<[u8]>) {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The prompt directory `name` of the test set: positions 0 and 1, six tapped values (0 to 5) and two calls.
pub(crate) fn write_prompt(dir: &Path, name: &str) {
    let p = dir.join(name);
    std::fs::create_dir_all(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    put(&p, "taps.pos", [0u32, 1].map(u32::to_le_bytes).concat());
    put(
        &p,
        "taps.f32",
        [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .map(f32::to_le_bytes)
            .concat(),
    );
    put(&p, "calls.tsv", "2\t7\t10,11,12\n4\t11\t20,21,22\n");
}

/// The manifest's lines of the test set, the trailer last when `complete`.
pub(crate) fn manifest(own: &Own<'_>, complete: bool) -> String {
    let mut lines = vec![
        format!("# model\t{}", own.model),
        format!("# draft_model\t{}", own.draft),
        format!("# build\t{}", own.build),
        format!("# arch\t{}", own.arch),
        format!("# n_max\t{N_MAX}"),
        "# prompt columns\tname\tn_taps\tn_calls".to_string(),
        "prompt\tp00\t2\t2".to_string(),
    ];
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    format!("{}\n", lines.join("\n"))
}

/// The test set under `dir` as `own` states it: the prompt `p00` and the manifest.
pub(crate) fn write_set(dir: &Path, own: &Own<'_>, complete: bool) {
    write_prompt(dir, "p00");
    put(dir, "MANIFEST.tsv", manifest(own, complete));
}

/// The manifest of the set under `dir` with `from` replaced by `to`, once, and nothing else changed.
pub(crate) fn edit_manifest(dir: &Path, from: &str, to: &str) {
    let path = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(text.contains(from), "the manifest holds no {from:?}");
    put(dir, "MANIFEST.tsv", text.replacen(from, to, 1));
}

/// Remove the set's directory.
pub(crate) fn remove(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}
