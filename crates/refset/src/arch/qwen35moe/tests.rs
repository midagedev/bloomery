use super::{ARCH, IK, IK_BUILD, MODEL};
use crate::RefError;
use std::path::{Path, PathBuf};

/// A set of the node dumps' shape at `dir`: the header lines given, one
/// tensor row, and the completion trailer when `complete`.
fn write_set(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) {
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
}

/// The family's check of a set with these lines.
fn check(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) -> Result<(), RefError> {
    write_set(dir, model, build, arch, complete);
    IK.check_set(dir).map(|_| ())
}

/// The family takes a complete set of the Q4_K_M file, the pinned build and
/// this architecture, and refuses by name one of another file (a UD file of
/// the same model beside it, or the Q4_K_M name in another directory),
/// another build (another commit, the pin's `-dirty` tree), another
/// architecture, or one without its trailer.
#[test]
fn the_family_refuses_a_set_of_another_file_build_or_arch() -> Result<(), RefError> {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("bloomery-refset-qwen35moe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let set = dir.display().to_string();

    write_set(&dir, MODEL, IK_BUILD, ARCH, true);
    let p = IK.check_set(&dir)?;
    assert_eq!(p.dumped_from, MODEL);
    assert_eq!(p.build.as_deref(), Some(IK_BUILD));

    for other in [
        "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "/models/elsewhere/Qwen3.6-35B-A3B-Q4_K_M.gguf",
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
    for build in ["49ef19d0", "db517b69-dirty"] {
        match check(&dir, MODEL, build, ARCH, true) {
            Err(RefError::Foreign {
                field: "build",
                family: "ik-qwen35moe",
                ..
            }) => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    for arch in ["qwen3moe", "qwen3next"] {
        match check(&dir, MODEL, IK_BUILD, arch, true) {
            Err(RefError::Foreign {
                field: "arch",
                family: "ik-qwen35moe",
                ..
            }) => {}
            r => panic!("a set of architecture {arch}: {r:?}"),
        }
    }
    match check(&dir, MODEL, IK_BUILD, ARCH, false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}
