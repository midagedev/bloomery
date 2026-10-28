use super::{Case, MANIFEST, Part, QuantSet, SYNTHETIC};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::md5::hex_of;
use std::path::{Path, PathBuf};

const HARNESS: &str = "86375a1d6ea83e3ea12c291f07d7befa";
const OTHER_MD5: &str = "0128e83c5e71e5f2b72f3d5d2c2de6b2";
const BUILD: &str = "36a73916e";

fn synthetic() -> Result<String, RefError> {
    Ok(SYNTHETIC.to_string())
}

static FAMILY: Family = Family {
    name: "test-quant",
    sets: &[],
    resolve: None,
    recipe: "just dump-ref-quant",
    identity: Identity::QuantManifest { harness: HARNESS },
    arch: None,
    build: Some(Build::Is(BUILD)),
    runs: Some(synthetic),
    draft_runs: None,
    consumers: &[],
};

/// What a set states in its header lines.
struct Stated<'a> {
    harness: &'a str,
    build: &'a str,
    model: &'a str,
    complete: bool,
}

const RIGHT: Stated<'static> = Stated {
    harness: HARNESS,
    build: BUILD,
    model: SYNTHETIC,
    complete: true,
};

/// Two blocks of q4_0 (18 bytes, 32 values) a row: 64 values, 2 quantized rows and 1 random one, 2 activation rows in q8_0
/// (34 bytes a block).
const CASE: &str = "case\tq4_0.64\t2\t64\t32\t18\t36\t8\tq8_0\t68\t2\t1\t2\t0";
const COLUMNS: &str = "# case\tname\ttype\tk\tblck\ttype_size\trow_bytes\tvdt\tvdt_name\tact_row_bytes\tquant_rows\trandom_rows\tact_rows\timatrix";

/// The table of ggml's types a set states: q4_0, which CASE is of, and i8, which has no to_float and no vec_dot to dump.
const TYPE_COLUMNS: &str = "# type\tname\tid\tblck\tbytes\tstatus\treason";
const TYPE_Q4_0: &str = "type\tq4_0\t2\t32\t18\tdumped\t-";
const TYPE_I8: &str = "type\ti8\t24\t1\t1\tskip\tto_float NULL; vec_dot NULL";

/// The five files of CASE: name and bytes. The dots are 1.0 to 6.0.
fn files() -> Vec<(String, Vec<u8>)> {
    let dots: Vec<u8> = (1..=6u8).flat_map(|v| f32::from(v).to_le_bytes()).collect();
    vec![
        ("q4_0.64.blocks".to_string(), vec![7; 3 * 36]),
        ("q4_0.64.deq.f32".to_string(), vec![0; 3 * 64 * 4]),
        ("q4_0.64.x.f32".to_string(), vec![0; 2 * 64 * 4]),
        ("q4_0.64.act".to_string(), vec![9; 2 * 68]),
        ("q4_0.64.dot.f32".to_string(), dots),
    ]
}

fn set_dir(what: &str) -> PathBuf {
    crate::clefvis::testkit::scratch(&format!("quant-{what}"))
}

fn write_set(dir: &Path, s: &Stated<'_>) {
    let mut lines = vec![
        format!("# harness\t{}", s.harness),
        format!("# build\t{}", s.build),
        "# cpu\tavx2 fma".to_string(),
        format!("# model\t{}", s.model),
        TYPE_COLUMNS.to_string(),
        TYPE_Q4_0.to_string(),
        TYPE_I8.to_string(),
        COLUMNS.to_string(),
        CASE.to_string(),
        "# file\tname\tbytes\tmd5".to_string(),
    ];
    let files = files();
    for (name, bytes) in &files {
        std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        lines.push(format!("file\t{name}\t{}\t{}", bytes.len(), hex_of(bytes)));
    }
    if s.complete {
        lines.push(format!("# complete\t{}", files.len()));
    }
    std::fs::write(dir.join(MANIFEST), {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

/// The manifest text with `from` replaced by `to`; `from` must be there.
fn edit(dir: &Path, from: &str, to: &str) {
    let path = dir.join(MANIFEST);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
    assert!(text.contains(from), "{from:?} is not in the manifest");
    std::fs::write(&path, text.replace(from, to)).unwrap_or_else(|e| panic!("{e}"));
}

fn remove(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

/// The error of opening the set at `dir`, or a panic when it opens.
fn refusal(dir: &Path) -> RefError {
    match QuantSet::open(dir, &FAMILY) {
        Ok(_) => panic!("a set that should be refused opened"),
        Err(e) => e,
    }
}

/// Whether `e` is a `Malformed` whose message names every word of `words`.
fn malformed_naming(e: &RefError, words: &[&str]) -> bool {
    matches!(e, RefError::Malformed { .. }) && words.iter().all(|w| e.to_string().contains(w))
}

/// A complete set of the harness and commit the family names opens, and says what it holds.
#[test]
fn a_set_of_the_familys_harness_and_build_opens() {
    let dir = set_dir("open");
    write_set(&dir, &RIGHT);
    let set = QuantSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.build.as_deref(), Some(BUILD));
    assert_eq!(set.cpu.as_deref(), Some("avx2 fma"));
    assert_eq!(set.files.len(), 5);
    let c = set.case("q4_0.64").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (c.type_name(), c.rows(), c.k, c.vdt_name.as_str()),
        ("q4_0", 3, 64, "q8_0")
    );
    assert!(!c.imatrix);
    assert_eq!(set.types.len(), 2);
    let q4 = set.type_row("q4_0").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (q4.id, q4.blck, q4.bytes, q4.skip.as_deref()),
        (2, 32, 18, None)
    );
    let i8 = set.type_row("i8").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (i8.id, i8.skip.as_deref()),
        (24, Some("to_float NULL; vec_dot NULL"))
    );
    assert!(set.type_row("q8_0").is_err());
    assert_eq!(set.of_type("q4_0").count(), 1);
    assert_eq!(set.of_type("q8_0").count(), 0);
    assert!(set.case("q4_0.4096").is_err());
    let dots = set.f32s(c, Part::Dot).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(dots, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        set.bytes(c, Part::Act)
            .unwrap_or_else(|e| panic!("{e}"))
            .len(),
        136
    );
    assert!(set.f32s(c, Part::Blocks).is_err());
    let p = FAMILY.check_set(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (SYNTHETIC, Some(BUILD))
    );
    remove(&dir);
}

/// A set dumped from another mainline commit is `Foreign`, field `build`, naming both.
#[test]
fn a_set_of_another_ggml_build_is_refused() {
    let dir = set_dir("build");
    write_set(
        &dir,
        &Stated {
            build: "deadbeef0",
            ..RIGHT
        },
    );
    match refusal(&dir) {
        RefError::Foreign {
            field: "build",
            got,
            want,
            ..
        } => assert_eq!((got.as_str(), want.as_str()), ("deadbeef0", BUILD)),
        e => panic!("a set of another build: {e:?}"),
    }
    remove(&dir);
}

/// A set that names a model file, as the dequant sets of a model do, is `Stale`: this oracle's rows come from no file.
#[test]
fn a_set_dumped_from_another_model_file_is_refused() {
    let dir = set_dir("model");
    write_set(
        &dir,
        &Stated {
            model: "/models/small/M.gguf",
            ..RIGHT
        },
    );
    match refusal(&dir) {
        RefError::Stale {
            dumped_from, runs, ..
        } => assert_eq!(
            (dumped_from.as_str(), runs.as_str()),
            ("/models/small/M.gguf", SYNTHETIC)
        ),
        e => panic!("a set of a model file: {e:?}"),
    }
    remove(&dir);
}

/// A set without its `# complete` trailer is `Unfinished`, whatever its rows say.
#[test]
fn a_set_without_its_trailer_is_refused() {
    let dir = set_dir("trailer");
    write_set(
        &dir,
        &Stated {
            complete: false,
            ..RIGHT
        },
    );
    assert!(matches!(refusal(&dir), RefError::Unfinished { .. }));
    remove(&dir);
}

/// A set of another harness source is `Foreign`, field `harness`; one with no harness line is too.
#[test]
fn a_set_of_another_harness_is_refused() {
    let dir = set_dir("harness");
    write_set(
        &dir,
        &Stated {
            harness: OTHER_MD5,
            ..RIGHT
        },
    );
    match refusal(&dir) {
        RefError::Foreign {
            field: "harness",
            got,
            want,
            ..
        } => assert_eq!((got.as_str(), want.as_str()), (OTHER_MD5, HARNESS)),
        e => panic!("a set of another harness: {e:?}"),
    }
    edit(&dir, &format!("# harness\t{OTHER_MD5}\n"), "");
    assert!(matches!(
        refusal(&dir),
        RefError::Foreign {
            field: "harness",
            ..
        }
    ));
    remove(&dir);
}

/// A set with no manifest is `Missing`, naming the recipe that writes it.
#[test]
fn a_directory_without_a_manifest_names_its_recipe() {
    let dir = set_dir("missing");
    let e = refusal(&dir);
    assert!(
        matches!(e, RefError::Missing { .. }) && e.to_string().contains("just dump-ref-quant"),
        "{e}"
    );
    remove(&dir);
}

/// A file that is not the one dumped (a byte changed, one cut) is refused by name, not read.
#[test]
fn a_file_that_is_not_the_one_dumped_is_refused() {
    let dir = set_dir("digest");
    write_set(&dir, &RIGHT);
    let path = dir.join("q4_0.64.blocks");
    let mut bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{e}"));
    bytes[0] ^= 1;
    std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("{e}"));
    assert!(malformed_naming(
        &refusal(&dir),
        &["q4_0.64.blocks", "not the one the set was dumped with"]
    ));
    bytes.pop();
    std::fs::write(&path, &bytes).unwrap_or_else(|e| panic!("{e}"));
    assert!(malformed_naming(&refusal(&dir), &["q4_0.64.blocks"]));
    std::fs::remove_file(&path).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(refusal(&dir), RefError::Missing { .. }));
    remove(&dir);
}

/// The cases against the files: a row inconsistent with itself, a file of another length than its case implies, a case missing
/// one, a file of no case, and a name twice are each `Malformed`, naming what disagrees.
#[test]
fn cases_that_disagree_with_their_files_are_refused() {
    // row_bytes is not k / blck * type_size
    let dir = set_dir("row");
    write_set(&dir, &RIGHT);
    edit(&dir, "\t36\t8\tq8_0", "\t40\t8\tq8_0");
    assert!(malformed_naming(
        &refusal(&dir),
        &["q4_0.64", "a row of 40 bytes"]
    ));
    remove(&dir);

    // the name does not end in the row length
    let dir = set_dir("name");
    write_set(&dir, &RIGHT);
    edit(&dir, "case\tq4_0.64\t", "case\tq4_0.32\t");
    assert!(malformed_naming(
        &refusal(&dir),
        &["q4_0.32", "row length 64"]
    ));
    remove(&dir);

    // a file of the wrong length for its case (its own digest is right)
    let dir = set_dir("length");
    write_set(&dir, &RIGHT);
    let short = vec![0u8; 12];
    std::fs::write(dir.join("q4_0.64.dot.f32"), &short).unwrap_or_else(|e| panic!("{e}"));
    let old = files().pop().unwrap_or_else(|| panic!("no files"));
    edit(
        &dir,
        &format!("file\t{}\t{}\t{}", old.0, old.1.len(), hex_of(&old.1)),
        &format!("file\t{}\t12\t{}", old.0, hex_of(&short)),
    );
    assert!(malformed_naming(
        &refusal(&dir),
        &["q4_0.64.dot.f32", "is 12 bytes", "implies 24"]
    ));
    remove(&dir);

    // a case with a file row missing, and the trailer that counts it
    let dir = set_dir("missing-file");
    write_set(&dir, &RIGHT);
    let act = files().remove(3);
    edit(
        &dir,
        &format!("file\t{}\t{}\t{}\n", act.0, act.1.len(), hex_of(&act.1)),
        "",
    );
    assert!(malformed_naming(
        &refusal(&dir),
        &["trailer counts 5 files", "names 4"]
    ));
    edit(&dir, "# complete\t5", "# complete\t4");
    assert!(malformed_naming(
        &refusal(&dir),
        &["case q4_0.64", "q4_0.64.act"]
    ));
    remove(&dir);

    // a file row that belongs to no case
    let dir = set_dir("stray");
    write_set(&dir, &RIGHT);
    std::fs::write(dir.join("stray.bin"), [1u8]).unwrap_or_else(|e| panic!("{e}"));
    edit(
        &dir,
        "# complete\t5",
        &format!("file\tstray.bin\t1\t{}\n# complete\t6", hex_of(&[1u8])),
    );
    assert!(malformed_naming(
        &refusal(&dir),
        &["stray.bin", "belongs to no case"]
    ));
    remove(&dir);

    // a case named twice
    let dir = set_dir("twice");
    write_set(&dir, &RIGHT);
    edit(&dir, CASE, &format!("{CASE}\n{CASE}"));
    assert!(malformed_naming(
        &refusal(&dir),
        &["case q4_0.64 is named twice"]
    ));
    remove(&dir);
}

/// A case row of q4_0 (32 values, 18 bytes a block) and q8_0 activations, with the sizes the arguments state:
/// `k`, `type_size`, `row_bytes`, `act_row_bytes` and the three row counts.
fn case_line(k: usize, sizes: [usize; 3], rows: [usize; 3]) -> String {
    let [type_size, row_bytes, act_row_bytes] = sizes;
    let [quant, random, act] = rows;
    format!(
        "case\tq4_0.{k}\t2\t{k}\t32\t{type_size}\t{row_bytes}\t8\tq8_0\t{act_row_bytes}\t{quant}\t{random}\t{act}\t0"
    )
}

/// A case whose lengths do not fit `usize`, or whose row is empty, is `Malformed` by name in release and debug alike: the
/// row counts and byte sizes would wrap (release) or panic (debug) in the file lengths the case implies. Each overflow row
/// carries one product past `usize` and the others inside it, one for each length `Case::bytes` and the row check compute.
#[test]
fn a_case_with_overflowing_or_empty_sizes_is_refused() {
    let max = usize::MAX;
    for (what, line, words) in [
        // MAX - 1 and 5 rows wrap to 3: the fixture's five file lengths still match
        (
            "rows",
            case_line(64, [18, 36, 68], [max - 1, 5, 2]),
            &["q4_0.64", "overflow"][..],
        ),
        (
            "row",
            case_line(64, [max, 36, 68], [2, 1, 2]),
            &["q4_0.64", "overflow"],
        ),
        (
            "blocks",
            case_line(64, [18, max, 68], [2, 1, 2]),
            &["q4_0.64", "overflow"],
        ),
        (
            "deq",
            case_line(64, [18, 36, 68], [1 << 57, 0, 2]),
            &["q4_0.64", "overflow"],
        ),
        (
            "x",
            case_line(64, [18, 36, 68], [2, 1, 1 << 57]),
            &["q4_0.64", "overflow"],
        ),
        (
            "act",
            case_line(64, [18, 36, max], [2, 1, 2]),
            &["q4_0.64", "overflow"],
        ),
        (
            "dot",
            case_line(64, [18, 36, 68], [1 << 40, 0, 1 << 40]),
            &["q4_0.64", "overflow"],
        ),
        (
            "k0",
            case_line(0, [18, 0, 68], [2, 1, 2]),
            &["q4_0.0", "empty weight row"],
        ),
        (
            "type_size0",
            case_line(64, [0, 0, 68], [2, 1, 2]),
            &["q4_0.64", "empty weight row"],
        ),
    ] {
        let dir = set_dir(&format!("size-{what}"));
        write_set(&dir, &RIGHT);
        edit(&dir, CASE, &line);
        assert!(
            malformed_naming(&refusal(&dir), words),
            "{what}: {}",
            refusal(&dir)
        );
        remove(&dir);
    }
}

/// The reader's parse refusals: a row before its column line, a row of the wrong width, a value that does not parse, an
/// md5 that is not one, an imatrix flag that is no flag, a row kind it does not know, and a set with no case.
#[test]
fn a_manifest_that_does_not_parse_is_refused() {
    for (what, from, to, words) in [
        (
            "early",
            COLUMNS,
            "",
            &["case row before its `# case` column line"][..],
        ),
        ("width", "\t68\t2\t1\t2\t0", "\t68\t2\t1\t2", &["fields"]),
        (
            "number",
            "\t36\t8\tq8_0",
            "\tx36\t8\tq8_0",
            &["row_bytes", "x36"],
        ),
        ("md5", HARNESS, "86375a1d", &["# harness", "not an md5"]),
        (
            "flag",
            "\t68\t2\t1\t2\t0",
            "\t68\t2\t1\t2\t2",
            &["imatrix", "want 0 or 1"],
        ),
        (
            "kind",
            "# cpu\tavx2 fma",
            "widget\t1",
            &["widget", "no row kind"],
        ),
    ] {
        let dir = set_dir(what);
        write_set(&dir, &RIGHT);
        edit(&dir, from, to);
        assert!(
            malformed_naming(&refusal(&dir), words),
            "{what}: {}",
            refusal(&dir)
        );
        remove(&dir);
    }
    let dir = set_dir("empty");
    write_set(&dir, &RIGHT);
    edit(&dir, CASE, "");
    assert!(malformed_naming(&refusal(&dir), &["no case rows"]));
    remove(&dir);
}

/// The table of ggml's types against the cases: a set with none (a harness older than the table), a row before its column
/// line, a case whose type has no row, is stated skipped or disagrees with its row about sizes, a dumped type with no case, a
/// type named twice (by name or by id), a status or reason that is not the table's and a type of an empty block are each
/// `Malformed`, naming what disagrees.
#[test]
fn types_that_disagree_with_the_cases_are_refused() {
    let both = format!("{TYPE_COLUMNS}\n{TYPE_Q4_0}\n{TYPE_I8}\n");
    for (what, from, to, words) in [
        (
            "none",
            both.as_str(),
            "",
            &["no type rows", "older than"][..],
        ),
        (
            "early",
            TYPE_COLUMNS,
            "",
            &["type row before its `# type` column line"],
        ),
        (
            "unnamed",
            TYPE_Q4_0,
            "type\tq4_1\t3\t32\t20\tdumped\t-",
            &["case q4_0.64", "no type row for q4_0"],
        ),
        (
            "skipped",
            "q4_0\t2\t32\t18\tdumped\t-",
            "q4_0\t2\t32\t18\tskip\tvec_dot NULL",
            &["case q4_0.64", "stated skipped", "vec_dot NULL"],
        ),
        (
            "size",
            "q4_0\t2\t32\t18\tdumped",
            "q4_0\t2\t32\t19\tdumped",
            &["case q4_0.64", "size 18", "size 19"],
        ),
        (
            "id",
            "q4_0\t2\t32\t18\tdumped",
            "q4_0\t3\t32\t18\tdumped",
            &["case q4_0.64", "id 2", "id 3"],
        ),
        (
            "caseless",
            TYPE_I8,
            "type\tq8_0\t8\t32\t34\tdumped\t-",
            &["type q8_0", "stated dumped", "no case"],
        ),
        (
            "twice-name",
            TYPE_I8,
            "type\tq4_0\t99\t1\t1\tskip\tvec_dot NULL",
            &["type q4_0 (id 99) repeats type q4_0 (id 2)"],
        ),
        (
            "twice-id",
            TYPE_I8,
            "type\tq8_0\t2\t32\t34\tskip\tvec_dot NULL",
            &["type q8_0 (id 2) repeats type q4_0 (id 2)"],
        ),
        (
            "status",
            "dumped\t-",
            "maybe\t-",
            &["type q4_0", "\"maybe\"", "want `dumped`"],
        ),
        (
            "reason",
            "dumped\t-",
            "dumped\tto_float NULL",
            &["type q4_0", "\"to_float NULL\"", "want `dumped`"],
        ),
        (
            "skip-no-reason",
            "skip\tto_float NULL; vec_dot NULL",
            "skip\t-",
            &["type i8", "\"skip\"", "ggml's reason"],
        ),
        (
            "empty-block",
            "type\ti8\t24\t1\t1\t",
            "type\ti8\t24\t0\t1\t",
            &["type i8", "a block of 0 values"],
        ),
    ] {
        let dir = set_dir(&format!("type-{what}"));
        write_set(&dir, &RIGHT);
        edit(&dir, from, to);
        assert!(
            malformed_naming(&refusal(&dir), words),
            "{what}: {}",
            refusal(&dir)
        );
        remove(&dir);
    }
}

/// The sizes a case implies for each part, from the row alone.
#[test]
fn a_case_implies_the_length_of_each_file() {
    let c = Case {
        name: "q4_K.4096".to_string(),
        ty: 12,
        k: 4096,
        blck: 256,
        type_size: 144,
        row_bytes: 2304,
        vdt: 15,
        vdt_name: "q8_K".to_string(),
        act_row_bytes: 4608,
        quant_rows: 32,
        random_rows: 16,
        act_rows: 4,
        imatrix: false,
    };
    let want = [
        (Part::Blocks, 48 * 2304),
        (Part::Deq, 48 * 4096 * 4),
        (Part::X, 4 * 4096 * 4),
        (Part::Act, 4 * 4608),
        (Part::Dot, 48 * 4 * 4),
    ];
    for (part, bytes) in want {
        assert_eq!(c.bytes(part), bytes, "{part:?}");
    }
    assert_eq!(c.file(Part::Deq), "q4_K.4096.deq.f32");
    assert_eq!(c.type_name(), "q4_K");
}
