//! The check every node-dump family keeps, run against a family row's own constants: a complete set of its file, build and
//! architecture passes and states that file and build, and a set of another file, another build or another architecture, or one
//! without its completion trailer, is refused by name. The families of the mainline-API dumper (`tools/ref/cb_dump.cpp`) differ in
//! the constants and in the sets they must refuse; the contract is one.

use crate::RefError;
use crate::family::Family;
use std::path::{Path, PathBuf};

/// A family row as its tests state it.
pub(crate) struct Row {
    pub family: &'static Family,
    /// The file the tree runs, the build and the architecture the family's sets name.
    pub model: &'static str,
    pub build: &'static str,
    pub arch: &'static str,
    /// Files whose sets are stale: another split set of the architecture (so only the `# model` line tells it apart), a shard that
    /// is not the first, and the first shard's name in another directory.
    pub other_files: &'static [&'static str],
    /// Builds whose sets are foreign.
    pub other_builds: &'static [&'static str],
    /// Architectures whose sets are foreign.
    pub other_arches: &'static [&'static str],
}

/// A fresh directory for one test's set; tests of one process run in parallel, so each passes its own `what`.
fn set_dir(row: &Row, what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-{}-{what}-{}",
        row.family.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

/// The family's check of a set of the dumper's shape at `dir`: the header lines given, one tensor row and, when `complete`, the
/// trailer.
fn check(
    row: &Row,
    dir: &Path,
    model: &str,
    build: &str,
    arch: &str,
    complete: bool,
) -> Result<(), RefError> {
    let mut lines = vec![
        "# cb_dump — a llama.cpp tree's intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
    ];
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    })
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    row.family.check_set(dir).map(|_| ())
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// A complete set of the family's file, build and architecture passes, and states that file and that build.
pub(crate) fn takes_a_set_of_its_file_build_and_arch(row: &Row) -> Result<(), RefError> {
    let dir = set_dir(row, "own")?;
    check(row, &dir, row.model, row.build, row.arch, true)?;
    let p = row.family.check_set(&dir)?;
    assert_eq!(p.dumped_from, row.model);
    assert_eq!(p.build.as_deref(), Some(row.build));
    remove(&dir)
}

/// A set of another file is stale, and the refusal names the set, the file it states and the file the tree runs.
pub(crate) fn refuses_a_set_of_another_file(row: &Row) -> Result<(), RefError> {
    let dir = set_dir(row, "file")?;
    let set = dir.display().to_string();
    for other in row.other_files {
        match check(row, &dir, other, row.build, row.arch, true) {
            Err(e @ RefError::Stale { .. }) => {
                let e = e.to_string();
                assert!(
                    e.contains(&set)
                        && e.contains(&format!("dumped from {other}"))
                        && e.contains(&format!("the tree runs {}", row.model)),
                    "{e}"
                );
            }
            r => panic!("a set of {other}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose build line names another tree is foreign by its build field, named for this family.
pub(crate) fn refuses_a_set_of_another_build(row: &Row) -> Result<(), RefError> {
    let dir = set_dir(row, "build")?;
    for build in row.other_builds {
        match check(row, &dir, row.model, build, row.arch, true) {
            Err(RefError::Foreign {
                field: "build",
                family,
                got,
                want,
                ..
            }) if family == row.family.name && got == *build && want == row.build => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose `# arch` names another architecture is foreign by its arch field.
pub(crate) fn refuses_a_set_of_another_arch(row: &Row) -> Result<(), RefError> {
    let dir = set_dir(row, "arch")?;
    for arch in row.other_arches {
        match check(row, &dir, row.model, row.build, arch, true) {
            Err(RefError::Foreign {
                field: "arch",
                family,
                got,
                want,
                ..
            }) if family == row.family.name && got == *arch && want == row.arch => {}
            r => panic!("a set of architecture {arch}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set without its completion trailer is unfinished, by the set's name.
pub(crate) fn refuses_a_set_without_its_trailer(row: &Row) -> Result<(), RefError> {
    let dir = set_dir(row, "trailer")?;
    let set = dir.display().to_string();
    match check(row, &dir, row.model, row.build, row.arch, false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    remove(&dir)
}
