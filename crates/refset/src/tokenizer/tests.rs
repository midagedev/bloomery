use super::{TokenizerSet, build_id};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::md5::hex_of;
use std::path::{Path, PathBuf};

pub(crate) const BINARY_MD5: &str = "d03083b01a24f6a9e5c2190abf7d38c6";
pub(crate) const LIBRARY_MD5: &str = "f855ad0b949f0b7d871973be1fcc419f";
pub(crate) const CASES_MD5: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OTHER_MD5: &str = "0128e83c5e71e5f2b72f3d5d2c2de6b2";
pub(crate) const VOCABULARY: &str = "/models/V/V-00001-of-00009.gguf";
const PIN: &str = "llama-tokenize d03083b01a24f6a9e5c2190abf7d38c6 \
                   libllama.so f855ad0b949f0b7d871973be1fcc419f";

fn vocabulary() -> Result<String, RefError> {
    Ok(VOCABULARY.to_string())
}

pub(crate) static FAMILY: Family = Family {
    name: "test-tokenizer",
    sets: &[],
    resolve: None,
    recipe: "just dump-ref-tokenizer test",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(PIN)),
    runs: Some(vocabulary),
    draft_runs: None,
    consumers: &[],
};

/// What the writer states in a set, each default the family's own.
pub(crate) struct Stated<'a> {
    pub binary: &'a str,
    pub library: &'a str,
    pub vocabulary: &'a str,
    pub cases: &'a str,
    pub complete: bool,
}

pub(crate) const RIGHT: Stated<'static> = Stated {
    binary: BINARY_MD5,
    library: LIBRARY_MD5,
    vocabulary: VOCABULARY,
    cases: "",
    complete: true,
};

/// The corpus and the case of a test set: name, text, ids, `--no-parse-special` ids.
const TEXTS: [(&str, &str, &str, &str); 2] = [
    ("code", "int main() {}\n", "11\n12\n13\n", "11\n12\n"),
    ("cases/01", "héllo", "21\n22\n", "21\n22\n"),
];

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
pub(crate) fn set_dir(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-tokenizer-{what}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cases")).unwrap_or_else(|e| panic!("{e}"));
    dir
}

fn count(ids: &str) -> usize {
    ids.lines().count()
}

/// Write the texts, their ids and a manifest stating `s` into `dir`; the
/// `cases` md5 is the cases file's, `s.cases` unless it is empty.
pub(crate) fn write_set(dir: &Path, s: &Stated<'_>, cases_file_md5: &str) {
    let w = |name: &str, bytes: &str| {
        std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    };
    let mut lines = vec![
        format!("# tokenizer\t/ik/build/bin/llama-tokenize\t{}", s.binary),
        format!("# libllama\t/ik/build/src/libllama.so\t{}", s.library),
        format!("# vocabulary\t{}", s.vocabulary),
        "# text tree\t/ik".to_string(),
        format!(
            "# cases\t/repo/crates/tokenizer/tests/cases.txt\t{}",
            if s.cases.is_empty() {
                cases_file_md5
            } else {
                s.cases
            }
        ),
        "set\tbytes\ttext_md5\tids\tnps_ids".to_string(),
    ];
    for (name, text, ids, nps) in TEXTS {
        w(&format!("{name}.txt"), text);
        w(&format!("{name}.ids"), ids);
        w(&format!("{name}.nps.ids"), nps);
        lines.push(format!(
            "{name}\t{}\t{}\t{}\t{}",
            text.len(),
            hex_of(text.as_bytes()),
            count(ids),
            count(nps)
        ));
    }
    if s.complete {
        lines.push("# complete".to_string());
    }
    w("MANIFEST.tsv", &(lines.join("\n") + "\n"));
}

fn remove(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

/// A complete set of the file, executable and library the family names
/// opens; its texts are read back as dumped, split into corpora and cases,
/// and its ids by both modes with the counts the rows state.
#[test]
fn a_set_of_the_familys_file_and_build_opens_and_reads() {
    let dir = set_dir("open");
    write_set(&dir, &RIGHT, CASES_MD5);
    let set = TokenizerSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.build().as_deref(), Some(PIN));
    assert_eq!(build_id(BINARY_MD5, LIBRARY_MD5), PIN);
    assert_eq!(set.vocabulary.as_deref(), Some(VOCABULARY));
    let corpora: Vec<_> = set.corpora().map(|t| t.name.as_str()).collect();
    let cases: Vec<_> = set.cases().map(|t| t.name.as_str()).collect();
    assert_eq!((corpora, cases), (vec!["code"], vec!["cases/01"]));
    let code = &set.texts[0];
    assert_eq!(
        code.read().unwrap_or_else(|e| panic!("{e}")),
        b"int main() {}\n"
    );
    assert_eq!(
        code.ids(true).unwrap_or_else(|e| panic!("{e}")),
        [11, 12, 13]
    );
    assert_eq!(code.ids(false).unwrap_or_else(|e| panic!("{e}")), [11, 12]);
    let p = FAMILY.check_set(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (VOCABULARY, Some(PIN))
    );
    remove(&dir);
}

/// A set dumped with another executable, or with another library under the
/// same executable, is foreign by its `build`, and the refusal names both
/// digests.
#[test]
fn a_set_of_another_binary_or_library_is_foreign() {
    let dir = set_dir("build");
    for (stated, what) in [
        (
            Stated {
                binary: OTHER_MD5,
                ..RIGHT
            },
            "another executable",
        ),
        (
            Stated {
                library: OTHER_MD5,
                ..RIGHT
            },
            "another library",
        ),
    ] {
        write_set(&dir, &stated, CASES_MD5);
        match TokenizerSet::open(&dir, &FAMILY) {
            Err(RefError::Foreign {
                field: "build",
                family: "test-tokenizer",
                got,
                want,
                ..
            }) => {
                assert_eq!(want, PIN, "{what}");
                assert!(got.contains(OTHER_MD5) && got != PIN, "{what}: {got}");
            }
            r => panic!("a set of {what}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A set of another vocabulary file is stale, naming the set, the file it
/// states and the one the tree runs; so is one that states none.
#[test]
fn a_set_of_another_vocabulary_is_stale() {
    let dir = set_dir("vocab");
    let other = "/models/W/W-00001-of-00009.gguf";
    write_set(
        &dir,
        &Stated {
            vocabulary: other,
            ..RIGHT
        },
        CASES_MD5,
    );
    match TokenizerSet::open(&dir, &FAMILY) {
        Err(e @ RefError::Stale { .. }) => {
            let e = e.to_string();
            assert!(
                e.contains(&dir.display().to_string())
                    && e.contains(&format!("dumped from {other}"))
                    && e.contains(&format!("the tree runs {VOCABULARY}")),
                "{e}"
            );
        }
        r => panic!("a set of {other}: {r:?}"),
    }
    write_set(&dir, &RIGHT, CASES_MD5);
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.tsv")).unwrap_or_default();
    let stripped: Vec<&str> = manifest
        .lines()
        .filter(|l| !l.starts_with("# vocabulary"))
        .collect();
    std::fs::write(dir.join("MANIFEST.tsv"), stripped.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        TokenizerSet::open(&dir, &FAMILY),
        Err(RefError::Stale { .. })
    ));
    remove(&dir);
}

/// A text with one byte changed, or cut, is not the one dumped: the refusal
/// names the file and both digests, whether the text is a corpus or a case.
#[test]
fn a_text_that_is_not_the_one_dumped_is_refused_by_its_md5() {
    let dir = set_dir("text");
    for (name, changed) in [
        ("code.txt", "int main() {}!"),
        ("code.txt", "int main() {}"),
        ("cases/01.txt", "hållo"),
    ] {
        write_set(&dir, &RIGHT, CASES_MD5);
        let path = dir.join(name);
        let original = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(&path, changed).unwrap_or_else(|e| panic!("{e}"));
        match TokenizerSet::open(&dir, &FAMILY) {
            Err(RefError::Malformed { at, what }) => {
                assert_eq!(at, path.display().to_string());
                assert!(
                    what.contains(&hex_of(changed.as_bytes()))
                        && what.contains(&hex_of(original.as_bytes())),
                    "{what}"
                );
            }
            r => panic!("{name} changed to {changed:?}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A set without its trailer is unfinished, by the set's name; one with no
/// manifest is missing and names the recipe that writes it.
#[test]
fn an_unfinished_or_absent_set_is_refused_by_name() {
    let dir = set_dir("done");
    write_set(
        &dir,
        &Stated {
            complete: false,
            ..RIGHT
        },
        CASES_MD5,
    );
    match TokenizerSet::open(&dir, &FAMILY) {
        Err(RefError::Unfinished { set }) => assert_eq!(set, dir.display().to_string()),
        r => panic!("a set without its trailer: {r:?}"),
    }
    // The manifest of the format before the trailer: its `cases` row does not fit the column
    // line, and the refusal is still the missing trailer, not a line that does not parse.
    std::fs::write(
        dir.join("MANIFEST.tsv"),
        "# tokenizer\t/t\tx\nset\tbytes\ttext_md5\tids\tnps_ids\ncode\t1\tx\t1\t1\ncases\t42\n",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        TokenizerSet::open(&dir, &FAMILY),
        Err(RefError::Unfinished { .. })
    ));
    std::fs::remove_file(dir.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
    match TokenizerSet::open(&dir, &FAMILY) {
        Err(e @ RefError::Missing { .. }) => {
            assert!(
                e.to_string().contains("just dump-ref-tokenizer test"),
                "{e}"
            );
        }
        r => panic!("a set without its manifest: {r:?}"),
    }
    remove(&dir);
}

/// The cases file the tree holds is the one the case texts came from: a set
/// dumped from other cases is stale, naming both digests.
#[test]
fn a_set_of_other_cases_is_stale() {
    let dir = set_dir("cases");
    let cases = b"one\ntwo\n";
    let md5 = hex_of(cases);
    write_set(&dir, &RIGHT, &md5);
    let set = TokenizerSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    set.check_cases(cases).unwrap_or_else(|e| panic!("{e}"));
    let other = b"one\ntwo\nthree\n";
    match set.check_cases(other) {
        Err(RefError::Stale {
            dumped_from, runs, ..
        }) => {
            assert!(dumped_from.contains(&md5), "{dumped_from}");
            assert!(runs.contains(&hex_of(other)), "{runs}");
        }
        r => panic!("a set of other cases: {r:?}"),
    }
    remove(&dir);
}

/// An ids file whose count is not its row's, or that holds a line that is
/// not an id, is malformed by its file.
#[test]
fn an_ids_file_that_disagrees_with_its_row_is_malformed() {
    let dir = set_dir("ids");
    write_set(&dir, &RIGHT, CASES_MD5);
    let set = TokenizerSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    let ids = dir.join("code.ids");
    for (bytes, want) in [("11\n12\n", "2 ids"), ("11\n12\nx\n", "\"x\"")] {
        std::fs::write(&ids, bytes).unwrap_or_else(|e| panic!("{e}"));
        match set.texts[0].ids(true) {
            Err(RefError::Malformed { what, .. }) => assert!(what.contains(want), "{what}"),
            r => panic!("ids {bytes:?}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A manifest line the reader cannot take is malformed at its line: an md5
/// that is not one, a text row before its column line, a text named twice, a
/// name that leaves the set's directory.
#[test]
fn a_manifest_line_that_does_not_parse_is_malformed() {
    let dir = set_dir("lines");
    write_set(&dir, &RIGHT, CASES_MD5);
    let good = std::fs::read_to_string(dir.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
    let row = good
        .lines()
        .find(|l| l.starts_with("code\t"))
        .unwrap_or_default()
        .to_string();
    for (what, manifest) in [
        (
            "is not an md5",
            good.replace(BINARY_MD5, "D03083B01A24F6A9E5C2190ABF7D38C6"),
        ),
        (
            "before its `set` column line",
            good.replace("set\tbytes\ttext_md5\tids\tnps_ids\n", ""),
        ),
        ("named twice", good.replace(&row, &format!("{row}\n{row}"))),
        (
            "does not name a file under the set",
            good.replace(&row, &row.replacen("code", "../code", 1)),
        ),
        (
            "want <key>",
            good.replace(
                &format!("# libllama\t/ik/build/src/libllama.so\t{LIBRARY_MD5}"),
                "# libllama\t/ik/build/src/libllama.so",
            ),
        ),
    ] {
        std::fs::write(dir.join("MANIFEST.tsv"), &manifest).unwrap_or_else(|e| panic!("{e}"));
        match TokenizerSet::read(&dir, &FAMILY) {
            Err(RefError::Malformed { at, what: w }) => {
                assert!(
                    at.starts_with("tokenizer: ") && (w.contains(what) || at.contains(what)),
                    "{what}: {at}: {w}"
                );
            }
            r => panic!("{what}: {r:?}"),
        }
    }
    remove(&dir);
}
