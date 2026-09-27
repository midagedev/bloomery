use super::{ARCH, IK, IK_BUILD, MODEL};
use crate::RefError;
use std::path::{Path, PathBuf};

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-glm5next-{what}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

/// The family's check of a set of the node dumps' shape at `dir`: the
/// header lines given, one tensor row, and the completion trailer when
/// `complete`.
fn check(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) -> Result<(), RefError> {
    let mut lines = vec![
        "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian".to_string(),
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
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    IK.check_set(dir).map(|_| ())
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// A complete set of the UD-Q4_K_XL file, the pinned build and this
/// architecture passes, and states that file and that build.
#[test]
fn the_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    let dir = set_dir("own")?;
    check(&dir, MODEL, IK_BUILD, ARCH, true)?;
    let p = IK.check_set(&dir)?;
    assert_eq!(p.dumped_from, MODEL);
    assert_eq!(p.build.as_deref(), Some(IK_BUILD));
    remove(&dir)
}

/// A set of another file — another quantization's first shard beside it,
/// or the same shard name in another directory — is stale, and the refusal
/// names the set, the file it states and the file the tree runs.
#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let set = dir.display().to_string();
    for other in [
        "/models/GLM-5.3-Flash-Q4_K_M/GLM-5.3-Flash-Q4_K_M-00001-of-00006.gguf",
        "/models/elsewhere/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf",
    ] {
        match check(&dir, other, IK_BUILD, ARCH, true) {
            Err(e @ RefError::Stale { .. }) => {
                let e = e.to_string();
                assert!(
                    e.contains(&set)
                        && e.contains(&format!("dumped from {other}"))
                        && e.contains(&format!("the tree runs {MODEL}")),
                    "{e}"
                );
            }
            r => panic!("a set of {other}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose build line names another commit, or the pin's `-dirty`
/// tree, is foreign by its build field, named for this family.
#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    let dir = set_dir("build")?;
    for build in ["49ef19d0", "db517b69-dirty"] {
        match check(&dir, MODEL, build, ARCH, true) {
            Err(RefError::Foreign {
                field: "build",
                family: "ik-glm5next",
                got,
                want,
                ..
            }) if got == build && want == IK_BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose `# arch` names another architecture — the V4.1 port's, which
/// shares the dumper and the ik build — is foreign by its arch field, named
/// for this family.
#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    let dir = set_dir("arch")?;
    match check(&dir, MODEL, IK_BUILD, "deepseek41", true) {
        Err(RefError::Foreign {
            field: "arch",
            family: "ik-glm5next",
            got,
            want,
            ..
        }) if got == "deepseek41" && want == ARCH => {}
        r => panic!("a set of architecture deepseek41: {r:?}"),
    }
    remove(&dir)
}

/// A set without its completion trailer is unfinished, by the set's name.
#[test]
fn the_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    let dir = set_dir("trailer")?;
    let set = dir.display().to_string();
    match check(&dir, MODEL, IK_BUILD, ARCH, false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    remove(&dir)
}
