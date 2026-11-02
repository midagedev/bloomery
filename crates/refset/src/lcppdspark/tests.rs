use super::testkit::{
    ARCH, BUILD, DRAFT, FAMILY, FEAT, MODEL, N_MAX, OWN, edit_manifest, fresh, put, remove,
    write_set,
};
use super::{Call, DsparkSet, PromptRow};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use std::path::Path;

/// A family of another identity, which the reader refuses by name.
static NOT_DSPARK: Family = Family {
    name: "test-not-dspark",
    sets: &[],
    resolve: None,
    recipe: "just dump-test-not-dspark",
    identity: Identity::Manifest,
    arch: None,
    build: Some(Build::Is(BUILD)),
    runs: None,
    draft_runs: None,
    consumers: &[],
};

/// The family's open of a finished set that `edit` changed, the directory removed after. Tests run in parallel, so each
/// passes a `test` tag of its own.
fn after(test: &str, edit: impl FnOnce(&Path)) -> Result<DsparkSet, RefError> {
    let dir = fresh("open", test);
    write_set(&dir, &OWN, true);
    edit(&dir);
    let r = DsparkSet::open(&dir, &FAMILY);
    remove(&dir);
    r
}

/// `r` is `Malformed` and says `want`, in the place it names or in what it says.
fn malformed<T: std::fmt::Debug>(what: &str, r: Result<T, RefError>, want: &str) {
    match r {
        Err(RefError::Malformed { at, what: said }) => {
            assert!(
                at.contains(want) || said.contains(want),
                "{what}: malformed at {at}: {said}; wanted {want:?}"
            );
        }
        other => panic!("{what}: {other:?}"),
    }
}

/// A finished set opens through its family, states its files, build and architecture, and gives the prompt's positions, rows
/// and calls as the files hold them.
#[test]
fn a_finished_set_opens_and_gives_its_prompt() -> Result<(), RefError> {
    let dir = fresh("open", "own");
    write_set(&dir, &OWN, true);
    let set = DsparkSet::open(&dir, &FAMILY)?;
    assert_eq!(set.model.as_deref(), Some(MODEL));
    assert_eq!(set.draft_model.as_deref(), Some(DRAFT));
    assert_eq!(set.build.as_deref(), Some(BUILD));
    assert_eq!(set.arch.as_deref(), Some(ARCH));
    assert_eq!(set.n_max, Some(N_MAX));
    let row = PromptRow {
        name: "p00".to_string(),
        n_taps: 2,
        n_calls: 2,
    };
    assert_eq!(set.prompts, std::slice::from_ref(&row));
    let p = set.prompt(&row, FEAT)?;
    assert_eq!(p.name, "p00");
    assert_eq!(p.pos, [0, 1]);
    assert_eq!(p.taps, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        p.calls,
        [
            Call {
                pos0: 2,
                last: 7,
                ids: vec![10, 11, 12]
            },
            Call {
                pos0: 4,
                last: 11,
                ids: vec![20, 21, 22]
            },
        ]
    );
    let provenance = FAMILY.check_set(&dir)?;
    assert_eq!(provenance.dumped_from, MODEL);
    assert_eq!(provenance.draft.as_deref(), Some(DRAFT));
    assert_eq!(provenance.build.as_deref(), Some(BUILD));
    remove(&dir);
    Ok(())
}

/// A manifest without its trailer is a dump that did not finish: the refusal says so by the set's name, whatever follows it.
#[test]
fn a_set_without_its_trailer_is_unfinished_and_unread() {
    let dir = fresh("open", "trailer");
    write_set(&dir, &OWN, false);
    put(
        &dir,
        "MANIFEST.tsv",
        format!(
            "{}tensor\tnot-a-row-of-this-set\n",
            super::testkit::manifest(&OWN, false)
        ),
    );
    match DsparkSet::open(&dir, &FAMILY) {
        Err(RefError::Unfinished { set }) => assert_eq!(set, dir.display().to_string()),
        other => panic!("{other:?}"),
    }
    remove(&dir);
}

/// A directory with no manifest is missing, and the refusal names the recipe that writes the set.
#[test]
fn a_set_without_a_manifest_names_its_recipe() {
    let dir = fresh("open", "missing");
    match DsparkSet::open(&dir, &FAMILY) {
        Err(RefError::Missing { what, .. }) => {
            assert!(
                what.contains("MANIFEST.tsv") && what.contains(FAMILY.recipe),
                "{what}"
            );
        }
        other => panic!("{other:?}"),
    }
    remove(&dir);
}

/// The reader serves its own identity only.
#[test]
fn a_family_of_another_identity_is_refused() {
    let dir = fresh("open", "identity");
    write_set(&dir, &OWN, true);
    match DsparkSet::open(&dir, &NOT_DSPARK) {
        Err(RefError::Missing { what, .. }) => {
            assert!(what.contains(NOT_DSPARK.name), "{what}");
        }
        other => panic!("{other:?}"),
    }
    remove(&dir);
}

/// A header line twice is read as neither: the manifest is malformed, naming the line.
#[test]
fn a_header_line_twice_is_malformed() {
    for line in [
        "# model\t/x/target.gguf",
        "# draft_model\t/x/draft.gguf",
        "# build\tx",
        "# arch\tx",
        "# n_max\t3",
        "# complete\t1\t0",
    ] {
        let r = after("twice", |dir| {
            edit_manifest(
                dir,
                "# prompt columns",
                &format!("{line}\n# prompt columns"),
            );
        });
        malformed(line, r, "second");
    }
}

/// A header line of another shape (no value, an empty one, a count that is not a number) is malformed.
#[test]
fn a_header_line_of_another_shape_is_malformed() {
    for (from, to) in [
        (format!("# model\t{MODEL}"), "# model".to_string()),
        (format!("# model\t{MODEL}"), "# model\t".to_string()),
        (format!("# build\t{BUILD}"), format!("# build\t{BUILD}\tx")),
        ("# n_max\t3".to_string(), "# n_max\tmany".to_string()),
        ("# n_max\t3".to_string(), "# n_max".to_string()),
        ("# complete\t1\t0".to_string(), "# complete\t1".to_string()),
        (
            "# complete\t1\t0".to_string(),
            "# complete\t1\t1".to_string(),
        ),
    ] {
        let r = after("shape", |dir| edit_manifest(dir, &from, &to));
        malformed(&to, r, "");
    }
}

/// The completion count is the prompt rows: a trailer that counts others is a manifest edited or merged.
#[test]
fn a_completion_count_other_than_the_prompt_rows_is_malformed() {
    let r = after("count", |dir| {
        edit_manifest(dir, "# complete\t1\t0", "# complete\t2\t0");
    });
    malformed("count", r, "completion count");
}

/// A prompt row is read by the column line before it, once, as one directory of the set.
#[test]
fn a_prompt_row_is_read_by_its_column_line_once_as_a_directory() {
    for (what, from, to, want) in [
        (
            "no columns",
            "# prompt columns\tname\tn_taps\tn_calls\n",
            "",
            "before its",
        ),
        ("width", "prompt\tp00\t2\t2", "prompt\tp00\t2", "fields"),
        (
            "kind",
            "prompt\tp00\t2\t2",
            "prompt\tp00\t2\t2\ntensor\tp01\t2\t2",
            "unknown row kind",
        ),
        (
            "twice",
            "prompt\tp00\t2\t2",
            "prompt\tp00\t2\t2\nprompt\tp00\t2\t2",
            "named twice",
        ),
        (
            "parent",
            "prompt\tp00\t2\t2",
            "prompt\t..\t2\t2",
            "directory",
        ),
        (
            "nested",
            "prompt\tp00\t2\t2",
            "prompt\tp/00\t2\t2",
            "directory",
        ),
        (
            "count",
            "prompt\tp00\t2\t2",
            "prompt\tp00\ttwo\t2",
            "n_taps",
        ),
        (
            "column",
            "name\tn_taps\tn_calls",
            "name\tn_fed\tn_calls",
            "n_taps",
        ),
    ] {
        let r = after("rows", |dir| edit_manifest(dir, from, to));
        malformed(what, r, want);
    }
}

/// A set that does not say how many ids a call proposes, or names no prompt, checks nothing and is refused.
#[test]
fn a_set_without_n_max_or_prompts_is_malformed() {
    for (what, from, to, want) in [
        ("none", "# n_max\t3\n", "", "n_max"),
        ("zero", "# n_max\t3", "# n_max\t0", "n_max"),
        (
            "no prompts",
            "prompt\tp00\t2\t2\n# complete\t1\t0",
            "# complete\t0\t0",
            "no prompt rows",
        ),
    ] {
        let r = after("nomax", |dir| edit_manifest(dir, from, to));
        malformed(what, r, want);
    }
}

/// A prompt's files hold the sizes its row names: a position file, a rows file of whole rows, and the calls. Each break is
/// refused at open, naming the file.
#[test]
fn a_prompt_whose_files_differ_from_its_row_is_malformed() {
    // (what, the file of `p00`, its new bytes, what the refusal says)
    let files: [(&str, &str, Vec<u8>, &str); 10] = [
        ("positions", "taps.pos", vec![0; 12], "taps.pos"),
        ("partial position", "taps.pos", vec![0; 7], "taps.pos"),
        ("ragged rows", "taps.f32", vec![0; 28], "whole rows"),
        ("partial value", "taps.f32", vec![0; 22], "whole rows"),
        ("no rows", "taps.f32", Vec::new(), "whole rows"),
        (
            "one call",
            "calls.tsv",
            b"2\t7\t10,11,12\n".to_vec(),
            "holds 1 calls",
        ),
        ("no call", "calls.tsv", Vec::new(), "no draft call"),
        (
            "short line",
            "calls.tsv",
            b"2\t7\n4\t11\t1,2,3\n".to_vec(),
            "is not pos0",
        ),
        (
            "empty ids",
            "calls.tsv",
            b"2\t7\t\n4\t11\t1,2,3\n".to_vec(),
            "proposed id",
        ),
        (
            "id not a number",
            "calls.tsv",
            b"2\t7\t1,x\n4\t11\t1,2,3\n".to_vec(),
            "proposed id",
        ),
    ];
    for (what, file, bytes, want) in files {
        let r = after("files", |dir| put(&dir.join("p00"), file, bytes));
        malformed(what, r, want);
    }
    // A prompt row that names no fed rows or no calls names nothing to check.
    for (what, to) in [
        ("no fed rows", "prompt\tp00\t0\t2"),
        ("no calls named", "prompt\tp00\t2\t0"),
    ] {
        let r = after("rownames", |dir| {
            edit_manifest(dir, "prompt\tp00\t2\t2", to)
        });
        malformed(what, r, "a prompt holds both");
    }
}

/// A prompt's file that is not there is missing, naming the file; the set is the manifest's and the files are the dump's.
#[test]
fn a_prompt_with_a_file_missing_is_missing_by_name() {
    for file in ["taps.pos", "taps.f32", "calls.tsv"] {
        let r = after("gone", |dir| {
            std::fs::remove_file(dir.join("p00").join(file)).unwrap_or_else(|e| panic!("{e}"));
        });
        match r {
            Err(RefError::Missing { what, .. }) => assert!(what.contains(file), "{what}"),
            other => panic!("{file}: {other:?}"),
        }
    }
}

/// The rows are read at the width the caller names, and a value that is not finite is not a tap.
#[test]
fn a_prompt_is_read_at_the_callers_width_and_finite() -> Result<(), RefError> {
    let dir = fresh("open", "read");
    write_set(&dir, &OWN, true);
    let set = DsparkSet::open(&dir, &FAMILY)?;
    let row = set.prompts[0].clone();
    malformed(
        "wrong width",
        set.prompt(&row, FEAT + 1),
        "tapped values for 2 positions of 4",
    );
    // The same size, so the open above still holds: the values are read only here.
    put(
        &dir.join("p00"),
        "taps.f32",
        [0.0f32, 1.0, f32::NAN, 3.0, 4.0, 5.0]
            .map(f32::to_le_bytes)
            .concat(),
    );
    malformed(
        "not finite",
        set.prompt(&row, FEAT),
        "non-finite tapped value at index 2",
    );
    // Files that changed after the open are read against the row again.
    put(&dir.join("p00"), "taps.pos", vec![0u8; 12]);
    malformed("grown", set.prompt(&row, FEAT), "3 positions");
    put(&dir.join("p00"), "taps.pos", vec![0u8; 7]);
    malformed("partial", set.prompt(&row, FEAT), "32-bit");
    remove(&dir);
    Ok(())
}
