use super::{ARCH, CAND, CAND_BUILD, D1C, IK, IK_BUILD};
use crate::RefError;
use std::path::{Path, PathBuf};

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-deepseek41-{what}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

/// The candidate family's check of a set at `dir` with the header lines
/// given, one tensor row, and the completion trailer when `complete`.
fn check(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) -> Result<(), RefError> {
    let mut lines = vec![
        "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tcand_block_score-20\t0\tf32\t64\t1\t1\t1\t256\t0\tPOOL_2D".to_string(),
    ];
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    CAND.check_set(dir).map(|_| ())
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// The candidate family is its own row: a complete set of the file the tree
/// runs, the candidate tree's build and this architecture passes and states
/// both, its set resolves to the name the file's sets carry, and the
/// architecture's node-dump family stays [`IK`].
#[test]
fn the_cand_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    let dir = set_dir("own")?;
    let runs = gguf::v41::model();
    check(&dir, &runs, CAND_BUILD, ARCH, true)?;
    let p = CAND.check_set(&dir)?;
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (runs.as_str(), Some(CAND_BUILD))
    );
    assert_eq!(CAND.path(D1C), crate::data_dir().join(gguf::v41::set(D1C)));
    assert!(std::ptr::eq(
        crate::arch::node_dumps(ARCH).unwrap_or(&CAND),
        &IK
    ));
    remove(&dir)
}

/// A set of another file — the same shard name in another directory — is
/// stale, and the refusal names the set, the file it states and the file the
/// tree runs.
#[test]
fn the_cand_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let (set, runs) = (dir.display().to_string(), gguf::v41::model());
    let other = "/models/elsewhere/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf";
    match check(&dir, other, CAND_BUILD, ARCH, true) {
        Err(e @ RefError::Stale { .. }) => {
            let e = e.to_string();
            assert!(
                e.contains(&set)
                    && e.contains(&format!("dumped from {other}"))
                    && e.contains(&format!("the tree runs {runs}")),
                "{e}"
            );
        }
        r => panic!("a set of {other}: {r:?}"),
    }
    remove(&dir)
}

/// A set of the node dumps' build, whose tree builds no candidate mask, or
/// of the candidate tree with uncommitted changes, is foreign by its build
/// field, named for this family.
#[test]
fn the_cand_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    let dir = set_dir("build")?;
    let runs = gguf::v41::model();
    for build in [IK_BUILD, "9d213966-dirty"] {
        match check(&dir, &runs, build, ARCH, true) {
            Err(RefError::Foreign {
                field: "build",
                family: "cand-deepseek41",
                got,
                want,
                ..
            }) if got == build && want == CAND_BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose `# arch` names another architecture is foreign by its arch
/// field, and one without its completion trailer is unfinished, by the
/// set's name.
#[test]
fn the_cand_family_refuses_another_arch_and_an_unfinished_set() -> Result<(), RefError> {
    let dir = set_dir("arch")?;
    let (set, runs) = (dir.display().to_string(), gguf::v41::model());
    match check(&dir, &runs, CAND_BUILD, "deepseek4", true) {
        Err(RefError::Foreign {
            field: "arch",
            family: "cand-deepseek41",
            got,
            want,
            ..
        }) if got == "deepseek4" && want == ARCH => {}
        r => panic!("a set of architecture deepseek4: {r:?}"),
    }
    match check(&dir, &runs, CAND_BUILD, ARCH, false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    remove(&dir)
}
