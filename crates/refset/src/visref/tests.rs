use super::VisrefSet;
use crate::RefError;
use crate::family::{Build, Family, Identity};
use std::path::{Path, PathBuf};

const RUNS: &str = "/models/v41/m-00001-of-00009.gguf";

fn runs() -> Result<String, RefError> {
    Ok(RUNS.to_string())
}

static FAMILY: Family = Family {
    name: "test-visref",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::ForkManifest {
        rows_revision: "r1",
    },
    arch: Some("deepseek41"),
    build: Some(Build::Is("f00d")),
    runs: Some(runs),
    draft_runs: None,
    consumers: &[],
};

/// A two-case set: an image case of a 3-position span (n_embd 2, n_vocab 3)
/// and its text control, `keep` 2.
struct Set {
    model: String,
    build: &'static str,
    rows: &'static str,
    complete: bool,
    /// The answer.i32 file row's bytes for the image case.
    answer_bytes: u64,
}

impl Default for Set {
    fn default() -> Set {
        Set {
            model: RUNS.to_string(),
            build: "f00d",
            rows: "org/model@r1",
            complete: true,
            answer_bytes: 12,
        }
    }
}

fn dir() -> PathBuf {
    std::env::temp_dir().join(format!("bloomery-visref-{}", std::process::id()))
}

fn write(dir: &Path, s: &Set) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{e}"));
    let le32 = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let lef = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let files: Vec<(String, Vec<u8>)> = vec![
        ("img.ids.i32".into(), le32(&[0, 5, 9, 9, 9, 7])),
        (
            "img.rows.bf16".into(),
            [0x3f80u16, 0, 1, 2, 0xbf80, 3]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        ),
        ("img.types.i32".into(), le32(&[0, 1, 3])),
        ("img.answer.i32".into(), le32(&[2, 1, 1])),
        (
            "img.logits.f32".into(),
            lef(&[0.0, 1.0, 2.0, 0.5, 3.0, -1.0]),
        ),
        ("img-text.ids.i32".into(), le32(&[0, 5, 7])),
        ("img-text.answer.i32".into(), le32(&[1])),
        ("img-text.logits.f32".into(), lef(&[0.0, 4.0, 1.0])),
    ];
    let mut lines = vec![
        "# oracle\tvisref_fork.cpp".to_string(),
        format!("# model\t{}", s.model),
        format!("# build\t{}\thttps://example.invalid/fork", s.build),
        "# arch\tdeepseek41".to_string(),
        format!("# rows\t/data/ref-vision/scenes\t{}", s.rows),
        "# image_token_id\t9".to_string(),
        "# n_vocab\t3".to_string(),
        "# n_embd\t2".to_string(),
        "# keep\t2".to_string(),
        "# case columns\tname kind image n_ids span_at span_len gen answer eog logits question"
            .to_string(),
        "case\timg\timage\tchart\t6\t2\t3\t8\t3\t1\t2\tWhat is it?".to_string(),
        "case\timg-text\ttext\t-\t3\t0\t0\t8\t1\t0\t1\tWhat is it?".to_string(),
        "# file columns\tname kind dtype shape bytes md5".to_string(),
    ];
    for (name, data) in &files {
        std::fs::write(dir.join(name), data).unwrap_or_else(|e| panic!("{e}"));
        let bytes = if name == "img.answer.i32" {
            s.answer_bytes
        } else {
            data.len() as u64
        };
        lines.push(format!("file\t{name}\tk\tdt\t1\t{bytes}\tabcd"));
    }
    if s.complete {
        lines.push("# complete\t2\t8".to_string());
    }
    std::fs::write(dir.join("MANIFEST.tsv"), lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{e}"));
}

/// The manifest reads by its column lines and each case's files by their
/// rows: ids, the span's bf16 rows and types, the answer, the logits.
#[test]
fn a_visref_set_reads_by_its_column_lines() {
    let d = dir().join("reads");
    write(&d, &Set::default());
    let set = VisrefSet::open(&d, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (set.n_vocab, set.n_embd, set.keep, set.image_token_id),
        (3, 2, 2, 9)
    );
    let img = set.case("img").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (
            img.kind.as_str(),
            img.span_at,
            img.span_len,
            img.answer,
            img.eog,
            img.logits
        ),
        ("image", 2, 3, 3, true, 2)
    );
    assert_eq!(
        set.ids(img).unwrap_or_else(|e| panic!("{e}")),
        [0, 5, 9, 9, 9, 7]
    );
    assert_eq!(set.rows(img).unwrap_or_else(|e| panic!("{e}"))[4], 0xbf80);
    assert_eq!(set.types(img).unwrap_or_else(|e| panic!("{e}")), [0, 1, 3]);
    assert_eq!(set.answer(img).unwrap_or_else(|e| panic!("{e}")), [2, 1, 1]);
    assert_eq!(set.logits(img).unwrap_or_else(|e| panic!("{e}"))[4], 3.0);
    let text = set.case("img-text").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((text.span_len, text.eog), (0, false));
    assert!(matches!(set.rows(text), Err(RefError::Missing { .. })));
    std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
}

/// A set of another model file or another rows checkpoint is `Stale`, of
/// another fork build `Foreign`, one without its trailer `Unfinished`, and
/// a file row whose bytes disagree with its case `Malformed`.
#[test]
fn a_visref_set_of_another_source_is_refused_by_name() {
    let d = dir().join("refused");
    write(
        &d,
        &Set {
            model: "/models/other-00001-of-00009.gguf".into(),
            ..Set::default()
        },
    );
    match VisrefSet::open(&d, &FAMILY) {
        Err(RefError::Stale { dumped_from, .. }) => {
            assert_eq!(dumped_from, "/models/other-00001-of-00009.gguf")
        }
        other => panic!("another model file: {other:?}"),
    }
    write(
        &d,
        &Set {
            build: "beef",
            ..Set::default()
        },
    );
    match VisrefSet::open(&d, &FAMILY) {
        Err(RefError::Foreign { field, got, .. }) => {
            assert_eq!((field, got.as_str()), ("build", "beef"))
        }
        other => panic!("another fork build: {other:?}"),
    }
    write(
        &d,
        &Set {
            rows: "org/model@r2",
            ..Set::default()
        },
    );
    match VisrefSet::open(&d, &FAMILY) {
        Err(RefError::Stale {
            dumped_from, runs, ..
        }) => {
            assert_eq!(
                (dumped_from.as_str(), runs.as_str()),
                ("image rows of org/model@r2", "r1")
            )
        }
        other => panic!("rows of another checkpoint: {other:?}"),
    }
    write(
        &d,
        &Set {
            complete: false,
            ..Set::default()
        },
    );
    assert!(matches!(
        VisrefSet::open(&d, &FAMILY),
        Err(RefError::Unfinished { .. })
    ));
    write(
        &d,
        &Set {
            answer_bytes: 8,
            ..Set::default()
        },
    );
    match VisrefSet::read(&d) {
        Err(RefError::Malformed { what, .. }) => {
            assert!(what.starts_with("img.answer.i32"), "{what}")
        }
        other => panic!("a file row of other bytes: {other:?}"),
    }
    std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
}
