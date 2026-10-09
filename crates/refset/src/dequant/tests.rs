use super::{DequantSet, MANIFEST, SYNTHETIC, build_id};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::md5::hex_of;
use std::path::{Path, PathBuf};

const BINARY_MD5: &str = "e265fd2f62e16c4f8c5ebc5d28705f41";
const LIBRARY_MD5: &str = "75ce969e36e0fd4d520b5086a16c8b57";
const OTHER_MD5: &str = "0128e83c5e71e5f2b72f3d5d2c2de6b2";
const MODEL: &str = "/models/small/M.gguf";
const PIN: &str = "dequant_ref e265fd2f62e16c4f8c5ebc5d28705f41 \
                   libggml.so 75ce969e36e0fd4d520b5086a16c8b57";

fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

fn synthetic() -> Result<String, RefError> {
    Ok(SYNTHETIC.to_string())
}

static FAMILY: Family = Family {
    name: "test-dequant",
    sets: &[],
    resolve: None,
    recipe: "just dump-ref-dequant",
    identity: Identity::DequantManifest,
    arch: None,
    build: Some(Build::Is(PIN)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

static SYNTH_FAMILY: Family = Family {
    name: "test-dequant-synth",
    sets: &[],
    resolve: None,
    recipe: "just dump-ref-dequant",
    identity: Identity::DequantManifest,
    arch: None,
    build: Some(Build::Is(PIN)),
    runs: Some(synthetic),
    draft_runs: None,
    consumers: &[],
};

struct Stated<'a> {
    binary: &'a str,
    library: &'a str,
    model: &'a str,
    complete: bool,
}

const RIGHT: Stated<'static> = Stated {
    binary: BINARY_MD5,
    library: LIBRARY_MD5,
    model: MODEL,
    complete: true,
};

const FILES: [(&str, &[u8]); 2] = [
    ("manifest.txt", b"12 q4_K 53\n"),
    ("q4_K.raw", &[0, 1, 2, 3, 4, 5, 6, 7]),
];

fn set_dir(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-dequant-{what}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    dir
}

fn write_set(dir: &Path, s: &Stated<'_>) {
    let mut lines = vec![
        format!("# dequant_ref\t/data/bin/dequant_ref\t{}", s.binary),
        format!("# libggml\t/ik/build/ggml/src/libggml.so\t{}", s.library),
        format!("# model\t{}", s.model),
        "file\tbytes\tmd5".to_string(),
    ];
    for (name, bytes) in FILES {
        std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        lines.push(format!("{name}\t{}\t{}", bytes.len(), hex_of(bytes)));
    }
    if s.complete {
        lines.push("# complete".to_string());
    }
    std::fs::write(dir.join(MANIFEST), lines.join("\n") + "\n").unwrap_or_else(|e| panic!("{e}"));
}

fn remove(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

/// A complete set of the model, harness and library the family names opens,
/// reports them, and a synthetic set opens under the family that names no
/// file.
#[test]
fn a_set_of_the_familys_model_and_build_opens() {
    let dir = set_dir("open");
    write_set(&dir, &RIGHT);
    let set = DequantSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.build().as_deref(), Some(PIN));
    assert_eq!(build_id(BINARY_MD5, LIBRARY_MD5), PIN);
    assert_eq!(set.files.len(), 2);
    let p = FAMILY.check_set(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (MODEL, Some(PIN))
    );
    write_set(
        &dir,
        &Stated {
            model: SYNTHETIC,
            ..RIGHT
        },
    );
    SYNTH_FAMILY
        .check_set(&dir)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        DequantSet::open(&dir, &FAMILY),
        Err(RefError::Stale { .. })
    ));
    remove(&dir);
}

/// A set dumped with another harness, or by the same harness over another
/// ggml library, is foreign by its `build`.
#[test]
fn a_set_of_another_harness_or_library_is_foreign() {
    let dir = set_dir("build");
    for (stated, what) in [
        (
            Stated {
                binary: OTHER_MD5,
                ..RIGHT
            },
            "another harness",
        ),
        (
            Stated {
                library: OTHER_MD5,
                ..RIGHT
            },
            "another library",
        ),
    ] {
        write_set(&dir, &stated);
        match DequantSet::open(&dir, &FAMILY) {
            Err(RefError::Foreign {
                field: "build",
                family: "test-dequant",
                got,
                want,
                ..
            }) => assert!(want == PIN && got.contains(OTHER_MD5), "{what}: {got}"),
            r => panic!("a set of {what}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A set of another model file is stale, naming the set and both files; one
/// of an unfinished dump is unfinished; one with no identity file is missing
/// and names the recipe.
#[test]
fn a_set_of_another_model_or_unfinished_is_refused_by_name() {
    let dir = set_dir("model");
    let other = "/models/elsewhere/M.gguf";
    write_set(
        &dir,
        &Stated {
            model: other,
            ..RIGHT
        },
    );
    match DequantSet::open(&dir, &FAMILY) {
        Err(e @ RefError::Stale { .. }) => {
            let e = e.to_string();
            assert!(
                e.contains(&format!("dumped from {other}"))
                    && e.contains(&format!("the tree runs {MODEL}")),
                "{e}"
            );
        }
        r => panic!("a set of {other}: {r:?}"),
    }
    write_set(
        &dir,
        &Stated {
            complete: false,
            ..RIGHT
        },
    );
    assert!(matches!(
        DequantSet::open(&dir, &FAMILY),
        Err(RefError::Unfinished { .. })
    ));
    std::fs::remove_file(dir.join(MANIFEST)).unwrap_or_else(|e| panic!("{e}"));
    match DequantSet::open(&dir, &FAMILY) {
        Err(e @ RefError::Missing { .. }) => {
            assert!(e.to_string().contains("just dump-ref-dequant"), "{e}");
        }
        r => panic!("a set without its identity file: {r:?}"),
    }
    remove(&dir);
}

/// A file with one byte changed, cut or gone is not the one dumped: the
/// refusal names the file and both digests (a missing file, itself).
#[test]
fn a_file_that_is_not_the_one_dumped_is_refused_by_its_md5() {
    let dir = set_dir("file");
    for changed in [
        Some(vec![0, 1, 2, 3, 4, 5, 6, 8]),
        Some(vec![0, 1, 2]),
        None,
    ] {
        write_set(&dir, &RIGHT);
        let path = dir.join("q4_K.raw");
        match &changed {
            Some(b) => std::fs::write(&path, b).unwrap_or_else(|e| panic!("{e}")),
            None => std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{e}")),
        }
        match (DequantSet::open(&dir, &FAMILY), &changed) {
            (Err(RefError::Malformed { at, what }), Some(b)) => {
                assert_eq!(at, path.display().to_string());
                assert!(
                    what.contains(&hex_of(b)) && what.contains(&hex_of(FILES[1].1)),
                    "{what}"
                );
            }
            (Err(RefError::Missing { path: p, .. }), None) => assert_eq!(p, path),
            (r, _) => panic!("q4_K.raw changed to {changed:?}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A manifest line the reader cannot take is malformed at its line: an md5
/// that is not one, a file row before its column line, a file named twice or
/// outside the directory, a set that names no file.
#[test]
fn a_manifest_line_that_does_not_parse_is_malformed() {
    let dir = set_dir("lines");
    write_set(&dir, &RIGHT);
    let good = std::fs::read_to_string(dir.join(MANIFEST)).unwrap_or_else(|e| panic!("{e}"));
    let row = good
        .lines()
        .find(|l| l.starts_with("q4_K.raw\t"))
        .unwrap_or_default()
        .to_string();
    for (what, manifest) in [
        ("is not an md5", good.replace(LIBRARY_MD5, "xyz")),
        (
            "before its `file` column line",
            good.replace("file\tbytes\tmd5\n", ""),
        ),
        ("named twice", good.replace(&row, &format!("{row}\n{row}"))),
        (
            "does not name a file of the set",
            good.replace(&row, &row.replacen("q4_K", "../q4_K", 1)),
        ),
        (
            "want # model",
            good.replace("# model\t/models/small/M.gguf", "# model"),
        ),
    ] {
        std::fs::write(dir.join(MANIFEST), &manifest).unwrap_or_else(|e| panic!("{e}"));
        match DequantSet::read(&dir, &FAMILY) {
            Err(RefError::Malformed { at, what: w }) => {
                assert!(
                    at.starts_with("dequant: ") && (w.contains(what) || at.contains(what)),
                    "{what}: {at}: {w}"
                );
            }
            r => panic!("{what}: {r:?}"),
        }
    }
    let names_none: String = good
        .lines()
        .filter(|l| !l.contains(".raw") && !l.starts_with("manifest.txt"))
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(dir.join(MANIFEST), names_none).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        DequantSet::open(&dir, &FAMILY),
        Err(RefError::Malformed { .. })
    ));
    remove(&dir);
}
