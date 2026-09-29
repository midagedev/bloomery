use super::{ARCH, IK, IK_BUILD, MODEL};
use crate::RefError;
use std::path::{Path, PathBuf};

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-qwen4exp-{what}-{}",
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

/// A set of another file is stale, and the refusal names the set, the file
/// it states and the file the tree runs: the Qwen3.6 file (same dumper,
/// build and ids), the MTP file beside the split set, or the same shard
/// name in another directory.
#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let set = dir.display().to_string();
    for other in [
        "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
        "/models/elsewhere/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
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
    for build in ["c10fbbcc", "db517b69-dirty"] {
        match check(&dir, MODEL, build, ARCH, true) {
            Err(RefError::Foreign {
                field: "build",
                family: "ik-qwen4exp",
                got,
                want,
                ..
            }) if got == build && want == IK_BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose `# arch` names another architecture is foreign by its arch
/// field, named for this family: qwen35moe, whose sets share the dumper, the
/// ik build and the batch ids, so only this line and the model tell them
/// apart; and deepseek41.
#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    let dir = set_dir("arch")?;
    for arch in ["qwen35moe", "deepseek41"] {
        match check(&dir, MODEL, IK_BUILD, arch, true) {
            Err(RefError::Foreign {
                field: "arch",
                family: "ik-qwen4exp",
                got,
                want,
                ..
            }) if got == arch && want == ARCH => {}
            r => panic!("a set of architecture {arch}: {r:?}"),
        }
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

/// An MTP set's manifest at `dir` stating `model`, `draft` (no line when
/// `None`) and `build`, one block, and its trailer.
fn write_mtp(dir: &Path, model: &str, draft: Option<&str>, build: &str) {
    let mut lines = vec![
        "# dump_mtp — ik_llama.cpp MTP (NextN) draft tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
    ];
    lines.extend(draft.map(|d| format!("# draft_model\t{d}")));
    lines.extend([
        format!("# build\t{build}"),
        format!("# arch\t{ARCH}"),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tblock\trow\taccepted\tgraph"
            .to_string(),
        "# draft\tblock\trow\ttoken".to_string(),
        "# verify\tblock\tpos\tid_last\tcarry\tdrafted\taccepted\ttarget".to_string(),
        "tensor\tmtp_eh_proj-48\t0\tf32\t2560\t4\t1\t1\t40960\t0\tMUL_MAT\t0\t-\t0\tupdate"
            .to_string(),
        "draft\t0\t0\t15".to_string(),
        "verify\t0\t64\t4\t0\t1\t1\t4,15,99".to_string(),
        "# complete\t1\t0".to_string(),
    ]);
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The MTP family is its own row beside the node dumps', which stay the
/// architecture's node-dump family. A set of the target, the shared draft
/// and the MTP build passes and states both files; one naming the node
/// dumps' build is foreign.
#[test]
fn the_mtp_family_takes_its_files_and_the_mtp_build() -> Result<(), RefError> {
    use super::mtp::{DRAFT, MTP, MTP_BUILD, MTP_SET};
    assert!(std::ptr::eq(
        crate::arch::node_dumps(ARCH).unwrap_or(&MTP),
        &IK
    ));
    assert!(std::ptr::eq(
        crate::arch::named("mtp-qwen4exp").unwrap_or(&IK),
        &MTP
    ));
    assert_eq!(MTP.path(MTP_SET), crate::data_dir().join(MTP_SET));
    let dir = set_dir("mtp")?;
    write_mtp(&dir, MODEL, Some(DRAFT), MTP_BUILD);
    let p = MTP.check_set(&dir)?;
    assert_eq!(
        (
            p.dumped_from.as_str(),
            p.draft.as_deref(),
            p.build.as_deref()
        ),
        (MODEL, Some(DRAFT), Some(MTP_BUILD))
    );
    write_mtp(&dir, MODEL, Some(DRAFT), IK_BUILD);
    match MTP.check_set(&dir) {
        Err(RefError::Foreign {
            field: "build",
            family: "mtp-qwen4exp",
            got,
            want,
            ..
        }) if got == IK_BUILD && want == MTP_BUILD => {}
        r => panic!("an MTP set of the node dumps' build: {r:?}"),
    }
    remove(&dir)
}

/// The draft file is part of the set's identity: a set that states none, or
/// another draft file (the draft that carries its own matrices), is stale,
/// naming what it states and the file the tree runs.
#[test]
fn the_mtp_family_refuses_a_set_of_another_draft() -> Result<(), RefError> {
    use super::mtp::{DRAFT, MTP, MTP_BUILD};
    let dir = set_dir("mtp-draft")?;
    for draft in [
        None,
        Some("/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"),
    ] {
        write_mtp(&dir, MODEL, draft, MTP_BUILD);
        match MTP.check_set(&dir) {
            Err(RefError::Stale {
                dumped_from, runs, ..
            }) if runs == DRAFT
                && draft.is_none_or(|d| d == dumped_from)
                && (draft.is_some() || dumped_from.contains("# draft_model")) => {}
            r => panic!("a set of draft {draft:?}: {r:?}"),
        }
    }
    remove(&dir)
}
