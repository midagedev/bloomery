use super::{
    DEFAULT_ROOT, DIRS_TABLE, KEY_SUBSET, KEY_VARIANT, KEY_VERSION, LINE_PREFIX, ROOT_ENV, SELF,
    VARIANT_ENV, check_stated, dir_of, dirs, first_shard, first_shard_in, keys, line, parse_dirs,
    render, root_of, selected_of, variant_of, variant_root, variant_set_name, with_root,
    with_variant,
};
use crate::RefError;
use crate::arch::named;
use crate::family::{Build, Family, Provenance};
use crate::ik::RefManifest;
use crate::mtpref::MtpSet;
use gguf::Value;
use gguf::write::{Layout, Writer};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// A fresh, empty directory for one test; tests of one process run in
/// parallel, so each passes its own `what`.
fn temp(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-fixture-{what}-{}",
        std::process::id()
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    }
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

fn kv(key: &str, v: Value) -> (String, Value) {
    (key.to_string(), v)
}

/// A GGUF file of `kvs` and no tensor at `path`.
fn write_gguf(path: &Path, kvs: &[(String, Value)]) {
    let layout = Layout::new(kvs, Vec::new()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let file = std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Writer::new(file, layout)
        .and_then(Writer::finish)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The header keys of a fixture of seed `seed`: the five the generator writes.
fn fixture_kvs(seed: u64) -> Vec<(String, Value)> {
    vec![
        kv("general.name", Value::String("a fixture".to_string())),
        kv("bloomery.fixture.version", Value::U32(1)),
        kv("bloomery.fixture.seed", Value::U64(seed)),
        kv(
            "bloomery.fixture.source_layers",
            Value::Array(vec![Value::U32(0), Value::U32(3), Value::U32(5)]),
        ),
        kv(
            "bloomery.fixture.source_header_sha256",
            Value::String("0123abcd".to_string()),
        ),
        kv("bloomery.fixture.card_budget", Value::U64(8_589_934_592)),
    ]
}

/// The line of [`fixture_kvs`] of seed 1, written out: every fixture key but the card budget.
const LINE_SEED_1: &str = "# fixture\tbloomery.fixture.seed=1 \
                           bloomery.fixture.source_header_sha256=0123abcd \
                           bloomery.fixture.source_layers=0,3,5 bloomery.fixture.version=1";

/// `kvs` written to `<dir>/f.gguf` and the line read back.
fn line_of(dir: &Path, kvs: &[(String, Value)]) -> Result<String, RefError> {
    let path = dir.join("f.gguf");
    write_gguf(&path, kvs);
    line(&path)
}

/// The line holds every `bloomery.fixture.` key but the card budget, sorted by
/// byte order, each `key=value` and one space between, whatever order the file
/// lists them in; a key that only starts like the prefix is not in it.
#[test]
fn the_line_is_every_fixture_key_sorted() -> Result<(), RefError> {
    let dir = temp("line")?;
    let mut kvs = fixture_kvs(1);
    kvs.push(kv("bloomery.fixturex", Value::U32(9)));
    kvs.push(kv(
        "tokenizer.ggml.model",
        Value::String("not-a-fixture-key".to_string()),
    ));
    assert_eq!(line_of(&dir, &kvs)?, LINE_SEED_1);
    kvs.reverse();
    assert_eq!(line_of(&dir, &kvs)?, LINE_SEED_1);
    remove(&dir)
}

/// A value prints as its type says: integers in decimal (signed too), a bool
/// as `true` or `false`, a string verbatim, an array as its elements joined by
/// `,` with no space.
#[test]
fn the_values_print_by_their_type() -> Result<(), RefError> {
    let dir = temp("values")?;
    let kvs = vec![
        kv("bloomery.fixture.version", Value::U32(1)),
        kv("bloomery.fixture.a_u8", Value::U8(255)),
        kv("bloomery.fixture.b_i8", Value::I8(-8)),
        kv("bloomery.fixture.c_u16", Value::U16(65_535)),
        kv("bloomery.fixture.d_i16", Value::I16(-300)),
        kv("bloomery.fixture.e_i32", Value::I32(-70_000)),
        kv("bloomery.fixture.f_u64", Value::U64(u64::MAX)),
        kv("bloomery.fixture.g_i64", Value::I64(i64::MIN)),
        kv("bloomery.fixture.h_bool", Value::Bool(true)),
        kv("bloomery.fixture.i_bool", Value::Bool(false)),
        kv("bloomery.fixture.j_str", Value::String("x-y.z".to_string())),
        kv(
            "bloomery.fixture.k_strs",
            Value::Array(vec![
                Value::String("p".to_string()),
                Value::String("q".to_string()),
            ]),
        ),
        kv(
            "bloomery.fixture.l_ints",
            Value::Array(vec![Value::I32(-1), Value::I32(0), Value::I32(2)]),
        ),
    ];
    assert_eq!(
        line_of(&dir, &kvs)?,
        "# fixture\tbloomery.fixture.a_u8=255 bloomery.fixture.b_i8=-8 bloomery.fixture.c_u16=65535 \
         bloomery.fixture.d_i16=-300 bloomery.fixture.e_i32=-70000 \
         bloomery.fixture.f_u64=18446744073709551615 bloomery.fixture.g_i64=-9223372036854775808 \
         bloomery.fixture.h_bool=true bloomery.fixture.i_bool=false bloomery.fixture.j_str=x-y.z \
         bloomery.fixture.k_strs=p,q bloomery.fixture.l_ints=-1,0,2 bloomery.fixture.version=1"
    );
    remove(&dir)
}

/// The refusals, each by name: no version key (not a fixture), a subset key,
/// a float, a string holding whitespace, `=`, `,` or a control character; the
/// message names the file and the key.
#[test]
fn the_line_refuses_what_it_cannot_state() -> Result<(), RefError> {
    let dir = temp("refuse")?;
    let path = dir.join("f.gguf");
    let shown = path.display().to_string();
    let refused = |kvs: Vec<(String, Value)>, key: &str| {
        write_gguf(&path, &kvs);
        let e = line(&path).expect_err("refused");
        let e = e.to_string();
        assert!(e.contains(&shown) && e.contains(key), "{key}: {e}");
        e
    };

    let no_version: Vec<_> = fixture_kvs(1)
        .into_iter()
        .filter(|(k, _)| k != KEY_VERSION)
        .collect();
    let e = refused(no_version, KEY_VERSION);
    assert!(e.contains("not a fixture"), "{e}");

    let mut subset = fixture_kvs(1);
    subset.push(kv(
        KEY_SUBSET,
        Value::Array(vec![Value::String("tensor.a".to_string())]),
    ));
    let e = refused(subset, KEY_SUBSET);
    assert!(e.contains("only some tensors"), "{e}");

    for float in [Value::F32(0.5), Value::F64(0.5)] {
        let mut kvs = fixture_kvs(1);
        kvs.push(kv("bloomery.fixture.scale", float));
        let e = refused(kvs, "bloomery.fixture.scale");
        assert!(e.contains("float"), "{e}");
    }
    let mut in_array = fixture_kvs(1);
    in_array.push(kv(
        "bloomery.fixture.scales",
        Value::Array(vec![Value::F32(0.5), Value::F32(1.5)]),
    ));
    let e = refused(in_array, "bloomery.fixture.scales");
    assert!(e.contains("float"), "{e}");

    for bad in ["a b", "a=b", "a,b", "a\tb", "a\nb", "a\u{7f}b", "a\u{a0}b"] {
        let mut kvs = fixture_kvs(1);
        kvs.push(kv("bloomery.fixture.note", Value::String(bad.to_string())));
        let e = refused(kvs, "bloomery.fixture.note");
        assert!(e.contains("cannot be told apart"), "{bad:?}: {e}");
    }
    let mut in_strs = fixture_kvs(1);
    in_strs.push(kv(
        "bloomery.fixture.notes",
        Value::Array(vec![
            Value::String("ok".to_string()),
            Value::String("no ok".to_string()),
        ]),
    ));
    refused(in_strs, "bloomery.fixture.notes");
    remove(&dir)
}

/// What the writer and the reader both refuse before a line is formed — a
/// nested array, an empty one — is refused by the renderer too, by name,
/// since a file that holds one cannot be written to test it with.
#[test]
fn a_nested_or_empty_array_is_refused_by_the_renderer() {
    let nested = Value::Array(vec![Value::Array(vec![Value::U32(1)])]);
    let e = render("f.gguf", "bloomery.fixture.k", &nested).expect_err("nested");
    assert!(
        e.to_string().contains("a nested array") && e.to_string().contains("bloomery.fixture.k"),
        "{e}"
    );
    let e = render("f.gguf", "bloomery.fixture.k", &Value::Array(Vec::new())).expect_err("empty");
    assert!(e.to_string().contains("an empty array"), "{e}");
}

/// A GGUF v3 header of `kvs` as `(key, value type id, value bytes)` and no
/// tensor, written byte by byte: the cases the writer refuses to make.
fn write_raw(path: &Path, kvs: &[(&str, u32, Vec<u8>)]) {
    let put_str = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (key, ty, value) in kvs {
        put_str(&mut out, key);
        out.extend(ty.to_le_bytes());
        out.extend(value);
    }
    out.resize(out.len().next_multiple_of(32), 0);
    std::fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// A key the file gives twice is refused (the file would state two values),
/// and so is a nested array, which the reader refuses before a line is
/// formed; both by the key.
#[test]
fn a_key_given_twice_and_a_nested_array_are_refused() -> Result<(), RefError> {
    const U32: u32 = 4;
    const U64: u32 = 10;
    const ARRAY: u32 = 9;
    let dir = temp("raw")?;
    let path = dir.join("f.gguf");
    let version = ("bloomery.fixture.version", U32, 1u32.to_le_bytes().to_vec());

    let seed = |v: u64| ("bloomery.fixture.seed", U64, v.to_le_bytes().to_vec());
    write_raw(&path, &[version.clone(), seed(7), seed(8)]);
    let e = line(&path).expect_err("seed twice");
    assert!(matches!(e, RefError::Malformed { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&path.display().to_string())
            && e.contains("bloomery.fixture.seed is given twice"),
        "{e}"
    );

    let mut nested = Vec::new();
    nested.extend(ARRAY.to_le_bytes());
    nested.extend(1u64.to_le_bytes());
    nested.extend(U32.to_le_bytes());
    nested.extend(1u64.to_le_bytes());
    nested.extend(5u32.to_le_bytes());
    write_raw(&path, &[version, ("bloomery.fixture.nest", ARRAY, nested)]);
    let e = line(&path).expect_err("a nested array").to_string();
    assert!(
        e.contains(&path.display().to_string()) && e.contains("bloomery.fixture.nest"),
        "{e}"
    );
    remove(&dir)
}

/// A file that is no GGUF, and one that is not there, are refused by the
/// file's name.
#[test]
fn the_line_refuses_a_file_that_is_not_a_header() -> Result<(), RefError> {
    let dir = temp("nogguf")?;
    let path = dir.join("f.gguf");
    std::fs::write(&path, b"not a gguf header at all")
        .map_err(|e| RefError::missing(&path, e.to_string()))?;
    let e = keys(&path).expect_err("no GGUF").to_string();
    assert!(e.contains(&path.display().to_string()), "{e}");
    let gone = dir.join("gone.gguf");
    assert!(matches!(keys(&gone), Err(RefError::Missing { .. })));
    remove(&dir)
}

/// `first_shard_in`: none, one and two first shards of the directory are
/// `Missing`, the exact string and `Missing`; only the shell glob's matches
/// count (`*-00001-of-*.gguf`, no dotfile); the root joins as the shell joins
/// it; an architecture with no directory and a directory that is not there
/// are `Missing`, naming the architecture and the directory.
#[test]
fn the_first_shard_is_the_one_file_the_glob_matches() -> Result<(), RefError> {
    let root = temp("shard")?;
    let rs = root.display().to_string();
    let dir = root.join("glm5next");
    let named = |e: RefError| e.to_string();

    let e = named(first_shard_in(&rs, "glm5next").expect_err("no directory"));
    assert!(e.contains(&format!("{rs}/glm5next")), "{e}");

    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    for decoy in [
        "x-00002-of-00006.gguf",
        "x-00001-of-00006.txt",
        ".x-00001-of-00006.gguf",
        "x-0001-of-00006.gguf",
        "x-00001-00006.gguf",
    ] {
        std::fs::write(dir.join(decoy), b"").map_err(|e| RefError::missing(&dir, e.to_string()))?;
    }
    let e = named(first_shard_in(&rs, "glm5next").expect_err("none"));
    assert!(
        e.contains(&format!("{rs}/glm5next")) && e.contains("holds 0 first shards"),
        "{e}"
    );

    std::fs::write(dir.join("x-00001-of-00006.gguf"), b"")
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    assert_eq!(
        first_shard_in(&rs, "glm5next")?,
        format!("{rs}/glm5next/x-00001-of-00006.gguf")
    );
    assert_eq!(
        first_shard_in(&format!("{rs}/"), "glm5next")?,
        format!("{rs}//glm5next/x-00001-of-00006.gguf")
    );

    std::fs::write(dir.join("y-00001-of-00002.gguf"), b"")
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let e = named(first_shard_in(&rs, "glm5next").expect_err("two"));
    assert!(
        e.contains(&format!("{rs}/glm5next"))
            && e.contains("holds 2 first shards")
            && e.contains("x-00001-of-00006.gguf")
            && e.contains("y-00001-of-00002.gguf"),
        "{e}"
    );

    let e = named(first_shard_in(&rs, "deepseek2").expect_err("no fixture for the arch"));
    assert!(
        e.contains("deepseek2") && e.contains("no fixture directory"),
        "{e}"
    );
    let e = named(first_shard_in(&rs, "qwen4exp").expect_err("directory absent"));
    assert!(e.contains(&format!("{rs}/qwen38")), "{e}");
    remove(&root)
}

/// The root is `$BLOOMERY_FIXTURE_ROOT`, an empty value counting as unset.
#[test]
fn the_root_is_the_variable_or_the_default() {
    assert_eq!(root_of(None).as_deref().ok(), Some(DEFAULT_ROOT));
    assert_eq!(
        root_of(Some(OsString::new())).as_deref().ok(),
        Some(DEFAULT_ROOT)
    );
    assert_eq!(
        root_of(Some(OsString::from("/x/y"))).as_deref().ok(),
        Some("/x/y")
    );
}

/// The directory table is one file, `tools/fixture-dirs.tsv`: the embedded text parses, names a directory for every architecture
/// it lists but the `self` ones, and `dir_of` answers from it.
#[test]
fn the_embedded_table_parses_and_answers() {
    let rows = parse_dirs(DIRS_TABLE).unwrap_or_else(|e| panic!("tools/fixture-dirs.tsv: {e}"));
    let own = dirs();
    assert!(!own.is_empty(), "the table names no fixture directory");
    assert_eq!(
        own.len(),
        rows.iter().filter(|(_, d)| *d != SELF).count(),
        "dirs() holds the rows with a directory and no other"
    );
    for (arch, dir) in own {
        assert_eq!(dir_of(arch), Some(dir), "{arch}");
    }
    for (arch, _) in rows.iter().filter(|(_, d)| *d == SELF) {
        assert_eq!(dir_of(arch), None, "{arch} is a self family: no directory");
    }
    assert_eq!(dir_of("no-such-architecture"), None);
}

/// Every architecture with a family of sets dumped on a fixture file has a directory in the table: the sets' `# model` names
/// [`first_shard`] of it, so a row lost leaves a family no file to be stated against.
#[test]
fn every_architecture_with_fixture_sets_has_a_directory() {
    let mut seen = 0;
    for family in crate::arch::all().filter(|f| f.identity.on_fixture()) {
        let arch = family
            .arch
            .unwrap_or_else(|| panic!("the fixture family {} states no arch", family.name));
        assert!(
            dir_of(arch).is_some(),
            "the fixture family {} is of {arch}, which tools/fixture-dirs.tsv gives no directory",
            family.name
        );
        seen += 1;
    }
    assert!(seen > 0, "no family dumped on a fixture file in the table");
}

/// A table text that breaks a rule is refused by name, each rule by its own text.
#[test]
fn a_table_that_breaks_a_rule_is_refused_by_name() {
    let ok = "# c\n\na\tx\nb\ty-1\nc\tself\nd\tself\n";
    assert_eq!(
        parse_dirs(ok),
        Ok(vec![("a", "x"), ("b", "y-1"), ("c", "self"), ("d", "self")]),
        "comments and blank lines aside, the rows in file order, `self` shared"
    );
    for (text, want) in [
        ("a\tx\tz\n", "line 1 has 3 columns"),
        ("a\n", "line 1 has 1 columns"),
        ("a\tx\na\ty\n", "line 2: the architecture \"a\""),
        ("\tx\n", "line 1: the architecture \"\""),
        ("a\tX\n", "line 1: the directory \"X\""),
        ("a\t\n", "line 1: the directory \"\""),
        ("a\tnone\n", "line 1: the directory \"none\" is what"),
        ("a\tx\nb\tx\n", "line 2: the directory \"x\" is another"),
    ] {
        let e = parse_dirs(text).expect_err(text);
        assert!(e.contains(want), "{text:?}: {e}");
    }
}

/// What the shell keeps of its own is the root default and the file glob, and the directory table it reads from the file: its
/// `__fixture_root` default is [`DEFAULT_ROOT`] under [`ROOT_ENV`], the glob and the join are the ones [`first_shard_in`]
/// matches and forms, and `tools/ref/ref-paths.sh` holds no `case` row naming a directory.
#[test]
fn the_shell_reads_the_table_and_shares_the_root_and_glob() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/ref-paths.sh");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    assert!(
        text.contains("$REF_PATHS_DIR/../fixture-dirs.tsv"),
        "{} does not read tools/fixture-dirs.tsv",
        path.display()
    );
    let copies: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("__fixture_dir=") && l.contains(";;"))
        .collect();
    assert!(
        copies.is_empty(),
        "{} keeps a copy of the directory table: {copies:?}",
        path.display()
    );

    let root_line = text
        .lines()
        .find(|l| l.contains("__fixture_root="))
        .unwrap_or_else(|| panic!("{} has no `__fixture_root=` line", path.display()));
    assert_eq!(
        root_line.trim(),
        format!("__fixture_root=${{{ROOT_ENV}:-{DEFAULT_ROOT}}}"),
        "the shell's fixture root is not refset's"
    );
    assert!(
        text.contains("\"$__fixture_root/$__fixture_dir\"/*-00001-of-*.gguf"),
        "the shell's first-shard glob is not `<root>/<dir>/*-00001-of-*.gguf`"
    );
}

/// A fixture root holding `<dir>/f-00001-of-00001.gguf` of `kvs`, `<dir>`
/// being `arch`'s fixture directory; its first shard.
fn root_with(root: &Path, arch: &str, kvs: &[(String, Value)]) -> Result<String, RefError> {
    let dir = dir_of(arch).unwrap_or_else(|| panic!("no fixture directory for {arch}"));
    let d = root.join(dir);
    std::fs::create_dir_all(&d).map_err(|e| RefError::missing(&d, e.to_string()))?;
    write_gguf(&d.join("f-00001-of-00001.gguf"), kvs);
    first_shard_in(&root.display().to_string(), arch)
}

/// The text after the tab of a `# fixture` line.
fn body(line: &str) -> &str {
    line.strip_prefix("# fixture\t").expect("a fixture line")
}

/// The family of the table called `name`.
fn family(name: &str) -> &'static Family {
    named(name).unwrap_or_else(|| panic!("no family {name} in the table"))
}

/// A fixture family and what its sets must state, from the table: the
/// architecture and ik build of the family, and the file the real twin runs.
struct Under {
    family: &'static Family,
    arch: &'static str,
    build: &'static str,
    real_file: String,
}

fn under(fixture: &str, real: &str) -> Result<Under, RefError> {
    let family = family(fixture);
    let Some(Build::Is(build)) = family.build else {
        panic!("{fixture} pins no build");
    };
    Ok(Under {
        family,
        arch: family
            .arch
            .expect("a manifest family names its architecture"),
        build,
        real_file: self::family(real).runs()?,
    })
}

/// `family` checking the set at `set` with the fixture root at `root`.
fn open(family: &Family, root: &Path, set: &Path) -> Result<Provenance, RefError> {
    with_root(root, || family.check_set(set))
}

/// A node-dump set at `dir`: the header lines given (`None` leaves a line
/// out), one tensor row, and the trailer when `complete`.
fn write_node_set(
    dir: &Path,
    arch: &str,
    model: &str,
    fixture: Option<&str>,
    build: &str,
    complete: bool,
) {
    let mut lines = vec![
        "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
    ];
    lines.extend(fixture.map(|f| format!("# fixture\t{f}")));
    lines.extend([
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
    ]);
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
}

/// A node-dump set dumped from a copy of the fixture file at another root
/// (another machine's) is current when its `# model` names a file of the run
/// file's name and its `# fixture` line is the run file's; one whose line
/// differs is stale on that line, naming both lines and the run file's path.
#[test]
fn a_fixture_set_from_another_root_is_current_by_its_name_and_line() -> Result<(), RefError> {
    let root = temp("moved-root")?;
    let other = temp("moved-other")?;
    let set = temp("moved-set")?;
    let u = under("fx-ik-qwen35moe", "ik-qwen35moe")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let theirs = line_of(&other, &fixture_kvs(2))?;
    let name = Path::new(&first)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("a first shard has a file name");
    let elsewhere = format!("/models/fixtures/{arch}/{name}");
    assert_ne!(elsewhere, first);

    write_node_set(&set, arch, &elsewhere, Some(body(&ours)), build, true);
    let p = open(u.family, &root, &set)?;
    assert_eq!(p.dumped_from, elsewhere);

    write_node_set(&set, arch, &elsewhere, Some(body(&theirs)), build, true);
    let e = open(u.family, &root, &set).expect_err("another seed at another root");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&set.display().to_string())
            && e.contains(&format!("dumped from {theirs}"))
            && e.contains(&format!("the tree runs {ours}"))
            && e.contains(&first),
        "{e}"
    );

    remove(&root)?;
    remove(&other)?;
    remove(&set)
}

/// A node-dump set whose `# model` names a file of another name than the run
/// file's is stale on the file name, naming both paths, even when its
/// `# fixture` line is the run file's.
#[test]
fn a_fixture_set_of_another_file_name_is_stale() -> Result<(), RefError> {
    let root = temp("renamed-root")?;
    let set = temp("renamed-set")?;
    let u = under("fx-ik-qwen35moe", "ik-qwen35moe")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let renamed = Path::new(&first)
        .with_file_name("g-00001-of-00001.gguf")
        .display()
        .to_string();

    write_node_set(&set, arch, &renamed, Some(body(&ours)), build, true);
    let e = open(u.family, &root, &set).expect_err("another file name");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {renamed}"))
            && e.contains(&format!("the tree runs {first}"))
            && e.contains("file name"),
        "{e}"
    );

    remove(&root)?;
    remove(&set)
}

/// A draft set of the fixture MTP family whose `# draft_model` names a copy
/// of the draft file at another root is current: the draft of a fixture
/// family is named like the real draft's file, and its identity is that file
/// name (the generation is the `# fixture` line's, checked before it).
#[test]
fn a_fixture_draft_from_another_root_is_current_by_its_name() -> Result<(), RefError> {
    let root = temp("draft-root")?;
    let set = temp("draft-set")?;
    let u = under("fx-mtp-qwen4exp", "mtp-qwen4exp")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let real_draft = family("mtp-qwen4exp").draft_runs.expect("a draft file")()?;
    let name = real_draft
        .rsplit_once('/')
        .map_or(real_draft.as_str(), |(_, n)| n);
    let dir = first.rsplit_once('/').map(|(d, _)| d).expect("a directory");
    let draft = format!("{dir}/{name}");
    std::fs::write(&draft, b"").map_err(|e| RefError::missing(&draft, e.to_string()))?;
    let elsewhere = format!("/models/fixtures/{arch}/{name}");
    assert_ne!(elsewhere, draft);

    write_mtp_set(
        &set,
        arch,
        &first,
        Some(&elsewhere),
        Some(body(&ours)),
        build,
    );
    let p = open(u.family, &root, &set)?;
    assert_eq!(p.draft.as_deref(), Some(elsewhere.as_str()));

    remove(&root)?;
    remove(&set)
}

/// A draft set of the fixture MTP family whose `# draft_model` names a file
/// of another name than the draft the tree runs is stale on the file name,
/// naming both paths, even when its `# fixture` line is the run file's.
#[test]
fn a_fixture_draft_of_another_file_name_is_stale() -> Result<(), RefError> {
    let root = temp("draft-name-root")?;
    let set = temp("draft-name-set")?;
    let u = under("fx-mtp-qwen4exp", "mtp-qwen4exp")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let real_draft = family("mtp-qwen4exp").draft_runs.expect("a draft file")()?;
    let name = real_draft
        .rsplit_once('/')
        .map_or(real_draft.as_str(), |(_, n)| n);
    let dir = first.rsplit_once('/').map(|(d, _)| d).expect("a directory");
    let draft = format!("{dir}/{name}");
    std::fs::write(&draft, b"").map_err(|e| RefError::missing(&draft, e.to_string()))?;
    let other = format!("{dir}/other-draft.gguf");

    write_mtp_set(&set, arch, &first, Some(&other), Some(body(&ours)), build);
    let e = open(u.family, &root, &set).expect_err("another draft name");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {other}"))
            && e.contains(&format!("the tree runs {draft}"))
            && e.contains("file name"),
        "{e}"
    );

    remove(&root)?;
    remove(&set)
}

/// A node-dump set of the fixture the tree runs passes, and its `# fixture`
/// line is a header field, given at most once; one whose line differs in the
/// seed, or has none, is stale and names both lines; a set of the real file
/// opened as the fixture family is stale on `# model`, before the fixture
/// line is looked at; an unfinished set is refused before either, and the
/// fixture line is checked before the build.
#[test]
fn a_node_set_of_another_generation_is_stale() -> Result<(), RefError> {
    let root = temp("node-root")?;
    let other = temp("node-other")?;
    let set = temp("node-set")?;
    let u = under("fx-ik-glm5next", "ik-glm5next")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    assert_eq!(ours, LINE_SEED_1);
    let theirs = line_of(&other, &fixture_kvs(2))?;

    write_node_set(&set, arch, &first, Some(body(&ours)), build, true);
    let p = open(u.family, &root, &set)?;
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (first.as_str(), Some(build))
    );
    let read = RefManifest::read(&set)?;
    assert_eq!(read.header.fixture.as_deref(), Some(body(&ours)));
    assert!(
        read.header
            .other
            .iter()
            .all(|l| !l.starts_with("# fixture"))
    );

    write_node_set(&set, arch, &first, Some(body(&theirs)), build, true);
    let e = open(u.family, &root, &set).expect_err("another seed");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&set.display().to_string())
            && e.contains(&format!("dumped from {theirs}"))
            && e.contains(&format!("the tree runs {ours}"))
            && theirs.contains("seed=2")
            && ours.contains("seed=1"),
        "{e}"
    );

    write_node_set(&set, arch, &first, None, build, true);
    let e = open(u.family, &root, &set).expect_err("no # fixture line");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(e.contains("no # fixture line") && e.contains(&ours), "{e}");

    write_node_set(&set, arch, &u.real_file, None, build, true);
    let e = open(u.family, &root, &set).expect_err("a set of the real file");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {}", u.real_file))
            && e.contains(&format!("the tree runs {first}"))
            && !e.contains("fixture line"),
        "{e}"
    );

    write_node_set(&set, arch, &first, Some(body(&theirs)), build, false);
    let e = open(u.family, &root, &set);
    assert!(matches!(e, Err(RefError::Unfinished { .. })), "{e:?}");

    write_node_set(&set, arch, &first, None, "49ef19d0", true);
    let e = open(u.family, &root, &set);
    assert!(matches!(e, Err(RefError::Stale { .. })), "{e:?}");
    write_node_set(&set, arch, &first, Some(body(&ours)), "49ef19d0", true);
    let e = open(u.family, &root, &set);
    assert!(
        matches!(e, Err(RefError::Foreign { field: "build", .. })),
        "{e:?}"
    );

    write_node_set(&set, arch, &first, Some(body(&ours)), build, true);
    let manifest = set.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let twice = text.replacen(
        "# build",
        &format!("# fixture\t{}\n# build", body(&ours)),
        1,
    );
    std::fs::write(&manifest, twice).map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let e = RefManifest::read(&set)
        .expect_err("two # fixture lines")
        .to_string();
    assert!(e.contains("a second # fixture line"), "{e}");

    remove(&root)?;
    remove(&other)?;
    remove(&set)
}

/// `kvs` with the value of `key` replaced.
fn with_key(kvs: &[(String, Value)], key: &str, v: Value) -> Vec<(String, Value)> {
    let mut out = kvs.to_vec();
    let slot = out
        .iter_mut()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("{key} is not in the header"));
    slot.1 = v;
    out
}

/// The card budget is not a generation of the fixture: a set dumped before the budget moved, in the line's old form
/// (the budget token in it) or the new one (without), opens against the re-budgeted file; the same set against a file
/// that moved one key of the file's identity (the seed, the source's header digest, the layer map, the version) is still
/// stale and names both lines.
#[test]
fn a_budget_move_leaves_a_set_current_and_another_key_does_not() -> Result<(), RefError> {
    let root = temp("budget-root")?;
    let set = temp("budget-set")?;
    let u = under("fx-ik-glm5next", "ik-glm5next")?;
    let (arch, build) = (u.arch, u.build);
    let before = fixture_kvs(1);
    let first = root_with(&root, arch, &before)?;
    let ours = line(Path::new(&first))?;
    assert!(!ours.contains("card_budget"), "{ours}");
    // The old line: the same keys with the budget first, where the byte order put it.
    let old = format!("bloomery.fixture.card_budget=8589934592 {}", body(&ours));

    let moved = with_key(
        &before,
        "bloomery.fixture.card_budget",
        Value::U64(8_590_983_168),
    );
    root_with(&root, arch, &moved)?;
    assert_eq!(
        line(Path::new(&first))?,
        ours,
        "the budget is not in the line"
    );
    for stated in [old.as_str(), body(&ours)] {
        write_node_set(&set, arch, &first, Some(stated), build, true);
        let p = open(u.family, &root, &set)?;
        assert_eq!(p.dumped_from, first);
    }

    let moves = [
        (
            "seed",
            with_key(&moved, "bloomery.fixture.seed", Value::U64(2)),
        ),
        (
            "source_header_sha256",
            with_key(
                &moved,
                "bloomery.fixture.source_header_sha256",
                Value::String("4567ef01".to_string()),
            ),
        ),
        (
            "source_layers",
            with_key(
                &moved,
                "bloomery.fixture.source_layers",
                Value::Array(vec![Value::U32(0), Value::U32(3), Value::U32(6)]),
            ),
        ),
        (
            "version",
            with_key(&moved, "bloomery.fixture.version", Value::U32(2)),
        ),
    ];
    for (key, kvs) in moves {
        root_with(&root, arch, &kvs)?;
        let theirs = line(Path::new(&first))?;
        assert_ne!(theirs, ours, "{key}");
        for stated in [old.as_str(), body(&ours)] {
            write_node_set(&set, arch, &first, Some(stated), build, true);
            let e = open(u.family, &root, &set).expect_err(key);
            assert!(matches!(e, RefError::Stale { .. }), "{key}: {e:?}");
            let e = e.to_string();
            assert!(
                e.contains(&format!("dumped from # fixture\t{stated}"))
                    && e.contains(&format!("the tree runs {theirs}")),
                "{key}: {e}"
            );
        }
    }

    remove(&root)?;
    remove(&set)
}

/// A last-position logits set at `dir` (`nemotron_ref`'s): the header lines given (`None` leaves the fixture line
/// out), one prompt of four logits, and the trailer.
fn write_logits_set(dir: &Path, arch: &str, model: &str, fixture: Option<&str>, build: &str) {
    let mut lines = vec![format!("# model\t{model}")];
    lines.extend(fixture.map(|f| format!("# fixture\t{f}")));
    lines.extend([
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# prompt columns\tname\tn_ids\tn_vocab\tids\tlogits".to_string(),
        "prompt\tp\t2\t4\tp.ids\tp.logits.f32".to_string(),
        "# complete\t1\t0".to_string(),
    ]);
    let write = |name: &str, bytes: &[u8]| {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    };
    write(
        "MANIFEST.tsv",
        {
            let mut out = lines.join("\n");
            out.push('\n');
            out
        }
        .as_bytes(),
    );
    write("p.ids", b"0\n3\n");
    let logits: Vec<u8> = [0.5f32, -1.0, 2.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    write("p.logits.f32", &logits);
}

/// A logits set of the fixture the tree runs passes, with its `# fixture` line read; one of another seed, or
/// with no line, is stale and names both lines; a set of the real file is stale on `# model` first; two
/// `# fixture` lines are malformed. The real family takes the same set without a fixture line.
#[test]
fn a_logits_set_of_another_generation_is_stale() -> Result<(), RefError> {
    let root = temp("logits-root")?;
    let other = temp("logits-other")?;
    let set = temp("logits-set")?;
    let u = under("fx-ref-nemotron-moe-logits", "ref-nemotron-moe-logits")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let theirs = line_of(&other, &fixture_kvs(2))?;

    write_logits_set(&set, arch, &first, Some(body(&ours)), build);
    let p = open(u.family, &root, &set)?;
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (first.as_str(), Some(build))
    );
    let read = with_root(&root, || crate::nemotron::LogitsSet::open(&set, u.family))?;
    assert_eq!(read.fixture.as_deref(), Some(body(&ours)));

    write_logits_set(&set, arch, &first, Some(body(&theirs)), build);
    let e = open(u.family, &root, &set)
        .expect_err("another seed")
        .to_string();
    assert!(
        e.contains(&format!("dumped from {theirs}"))
            && e.contains(&format!("the tree runs {ours}")),
        "{e}"
    );

    write_logits_set(&set, arch, &first, None, build);
    let e = open(u.family, &root, &set).expect_err("no # fixture line");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    assert!(e.to_string().contains("no # fixture line"), "{e}");

    write_logits_set(&set, arch, &u.real_file, None, build);
    let e = open(u.family, &root, &set)
        .expect_err("a set of the real file")
        .to_string();
    assert!(
        e.contains(&format!("dumped from {}", u.real_file)) && !e.contains("fixture line"),
        "{e}"
    );
    let real = family("ref-nemotron-moe-logits");
    let p = real.check_set(&set)?;
    assert_eq!(p.dumped_from, u.real_file);

    write_logits_set(&set, arch, &first, Some(body(&ours)), build);
    let manifest = set.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let twice = text.replacen(
        "# build",
        &format!("# fixture\t{}\n# build", body(&ours)),
        1,
    );
    std::fs::write(&manifest, twice).map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let e = open(u.family, &root, &set)
        .expect_err("two # fixture lines")
        .to_string();
    assert!(e.contains("a second # fixture line"), "{e}");

    remove(&root)?;
    remove(&other)?;
    remove(&set)
}

/// An MTP set at `dir`: the header lines given (`None` leaves a line out),
/// one block and its trailer.
fn write_mtp_set(
    dir: &Path,
    arch: &str,
    model: &str,
    draft: Option<&str>,
    fixture: Option<&str>,
    build: &str,
) {
    let mut lines = vec![
        "# dump_mtp — ik_llama.cpp MTP (NextN) draft tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
    ];
    lines.extend(draft.map(|d| format!("# draft_model\t{d}")));
    lines.extend(fixture.map(|f| format!("# fixture\t{f}")));
    lines.extend([
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
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
    std::fs::write(&path, {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    })
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The same cases for an MTP set (GLM's, whose target carries the draft
/// layer) and for the family with a draft file of its own (Qwen3.8's): a set
/// of the fixture passes and states the draft beside the target; a line that
/// differs in the seed, or none, is stale and names both lines; the real
/// file's set is stale on `# model`; a set that states the real draft (the
/// same file name at another root) is current; a draft missing beside the
/// target is `Missing`.
#[test]
fn an_mtp_set_of_another_generation_is_stale() -> Result<(), RefError> {
    let root = temp("mtp-root")?;
    let other = temp("mtp-other")?;
    let set = temp("mtp-set")?;
    let u = under("fx-mtp-glm5next", "mtp-glm5next")?;
    let (arch, build) = (u.arch, u.build);
    let first = root_with(&root, arch, &fixture_kvs(1))?;
    let ours = line(Path::new(&first))?;
    let theirs = line_of(&other, &fixture_kvs(2))?;

    write_mtp_set(&set, arch, &first, None, Some(body(&ours)), build);
    let p = open(u.family, &root, &set)?;
    assert_eq!(
        (
            p.dumped_from.as_str(),
            p.draft.as_deref(),
            p.build.as_deref()
        ),
        (first.as_str(), None, Some(build))
    );
    assert_eq!(MtpSet::read(&set)?.fixture.as_deref(), Some(body(&ours)));

    write_mtp_set(&set, arch, &first, None, Some(body(&theirs)), build);
    let e = open(u.family, &root, &set).expect_err("another seed");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {theirs}"))
            && e.contains(&format!("the tree runs {ours}")),
        "{e}"
    );

    write_mtp_set(&set, arch, &first, None, None, build);
    let e = open(u.family, &root, &set).expect_err("no # fixture line");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(e.contains("no # fixture line") && e.contains(&ours), "{e}");

    write_mtp_set(&set, arch, &u.real_file, None, None, build);
    let e = open(u.family, &root, &set).expect_err("a set of the real file");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {}", u.real_file))
            && e.contains(&format!("the tree runs {first}"))
            && !e.contains("fixture line"),
        "{e}"
    );

    write_mtp_set(&set, arch, &first, None, Some(body(&ours)), build);
    let manifest = set.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let twice = text.replacen(
        "# build",
        &format!("# fixture\t{}\n# build", body(&ours)),
        1,
    );
    std::fs::write(&manifest, twice).map_err(|e| RefError::missing(&manifest, e.to_string()))?;
    let e = MtpSet::read(&set)
        .expect_err("two # fixture lines")
        .to_string();
    assert!(e.contains("a second # fixture line"), "{e}");

    let q = under("fx-mtp-qwen4exp", "mtp-qwen4exp")?;
    let (arch, build) = (q.arch, q.build);
    let qroot = temp("mtp-qroot")?;
    let qfirst = root_with(&qroot, arch, &fixture_kvs(1))?;
    let qours = line(Path::new(&qfirst))?;
    let real_draft = family("mtp-qwen4exp").draft_runs.expect("a draft file")()?;
    let name = real_draft
        .rsplit_once('/')
        .map_or(real_draft.as_str(), |(_, n)| n);
    let qdir = qfirst
        .rsplit_once('/')
        .map(|(d, _)| d)
        .expect("a directory");
    let draft = format!("{qdir}/{name}");

    write_mtp_set(&set, arch, &qfirst, Some(&draft), Some(body(&qours)), build);
    let e = open(q.family, &qroot, &set).expect_err("no draft beside the target");
    assert!(matches!(e, RefError::Missing { .. }), "{e:?}");
    let e = e.to_string();
    assert!(e.contains(&draft) && e.contains(&qfirst), "{e}");

    std::fs::write(&draft, b"").map_err(|e| RefError::missing(&draft, e.to_string()))?;
    let p = open(q.family, &qroot, &set)?;
    assert_eq!(p.draft.as_deref(), Some(draft.as_str()));

    write_mtp_set(
        &set,
        arch,
        &qfirst,
        Some(&real_draft),
        Some(body(&qours)),
        build,
    );
    let p = open(q.family, &qroot, &set)?;
    assert_eq!(p.draft.as_deref(), Some(real_draft.as_str()));

    write_mtp_set(
        &set,
        arch,
        &qfirst,
        Some(&draft),
        Some(body(&theirs)),
        build,
    );
    let e = open(q.family, &qroot, &set).expect_err("another seed");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    assert!(
        e.to_string().contains(&format!("the tree runs {qours}")),
        "{e}"
    );

    remove(&qroot)?;
    remove(&root)?;
    remove(&other)?;
    remove(&set)
}

/// The variant's tag is `$BLOOMERY_FIXTURE_VARIANT` (an empty value counting as unset), which moves the root to
/// `<parent of root>/fixture-variants/<tag>` and the sets' names to `fx_<tag>_<set>`; each rule is `tools/ref/ref-paths.sh`'s, whose
/// two lines are held here.
#[test]
fn a_variant_moves_the_root_and_names_its_sets() {
    assert_eq!(selected_of(None).ok(), Some(None));
    assert_eq!(selected_of(Some(OsString::new())).ok(), Some(None));
    assert_eq!(
        selected_of(Some(OsString::from("iq4xs"))).ok(),
        Some(Some("iq4xs".to_string()))
    );
    for bad in ["IQ4", "a/b", "..", "a_b", "a b"] {
        let e = selected_of(Some(OsString::from(bad)))
            .expect_err(bad)
            .to_string();
        assert!(
            e.contains(VARIANT_ENV) && e.contains("lower-case letters and digits"),
            "{e}"
        );
    }
    assert_eq!(variant_root("/m/fixtures", None), "/m/fixtures");
    assert_eq!(
        variant_root("/m/fixtures", Some("iq4xs")),
        "/m/fixture-variants/iq4xs"
    );
    assert_eq!(
        variant_set_name("iq4xs", "fx_ref_glm5next_d1k"),
        "fx_iq4xs_ref_glm5next_d1k"
    );
    assert_eq!(
        variant_set_name("iq4xs", "ref-mtp/fx_prose64_n64_k1"),
        "ref-mtp/fx_iq4xs_prose64_n64_k1"
    );
    assert_eq!(variant_set_name("iq4xs", "ref_glm5next"), "ref_glm5next");

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/ref-paths.sh");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let has = |want: &str| text.lines().any(|l| l.trim() == want);
    assert!(
        has("__fixture_root=${__fixture_root%/*}/fixture-variants/$FIXTURE_VARIANT"),
        "the shell's variant root is not `<parent of root>/fixture-variants/<tag>`"
    );
    assert!(
        has("SET_PREFIX=fx_${FIXTURE_VARIANT:+${FIXTURE_VARIANT}_}"),
        "the shell's set prefix is not `fx_<tag>_`"
    );
    assert!(
        has("FIXTURE_VARIANT=${BLOOMERY_FIXTURE_VARIANT:-}")
            && VARIANT_ENV == "BLOOMERY_FIXTURE_VARIANT",
        "the shell reads another variable than refset"
    );
}

/// The file a variant's tag selects is under `<parent of root>/fixture-variants/<tag>` and says so in its header; a file that
/// does not say it, or says it with no tag selected, is refused naming both.
#[test]
fn a_variant_file_is_found_under_its_root_and_says_so() -> Result<(), RefError> {
    let top = temp("variant-file")?;
    let root = top.join("fixtures");
    let base = root_with(&root, "glm5next", &fixture_kvs(1))?;
    let mut with_key = fixture_kvs(1);
    with_key.push(kv(KEY_VARIANT, Value::String("iq4xs".to_string())));
    let variants = top.join("fixture-variants/iq4xs");
    let var = root_with(&variants, "glm5next", &with_key)?;
    assert_eq!(variant_of(Path::new(&base))?, None);
    assert_eq!(variant_of(Path::new(&var))?.as_deref(), Some("iq4xs"));

    let first = |arch: &str| with_root(&root, || first_shard(arch));
    assert_eq!(first("glm5next")?, base);
    let var_first = with_root(&root, || with_variant("iq4xs", || first_shard("glm5next")))?;
    assert_eq!(var_first, var);
    assert!(
        var.starts_with(&format!("{}/fixture-variants/iq4xs/", top.display())),
        "{var}"
    );

    // The tag selects a file that does not carry it.
    let bare_top = temp("variant-bare")?;
    let bare = bare_top.join("fixtures");
    root_with(
        &bare_top.join("fixture-variants/iq4xs"),
        "glm5next",
        &fixture_kvs(1),
    )?;
    let e = with_root(&bare, || with_variant("iq4xs", || first_shard("glm5next")))
        .expect_err("a variant file with no key")
        .to_string();
    assert!(
        e.contains(VARIANT_ENV) && e.contains("\"iq4xs\"") && e.contains("none"),
        "{e}"
    );
    // A file that carries a tag, with none selected.
    let keyed = temp("variant-keyed")?;
    root_with(&keyed, "glm5next", &with_key)?;
    let e = with_root(&keyed, || first_shard("glm5next"))
        .expect_err("a keyed file, no tag")
        .to_string();
    assert!(e.contains("selects none") && e.contains("\"iq4xs\""), "{e}");
    // A tag whose directory is not there is the missing directory.
    let e = with_root(&root, || with_variant("other", || first_shard("glm5next")))
        .expect_err("another tag")
        .to_string();
    assert!(e.contains("fixture-variants/other/glm5next"), "{e}");

    remove(&top)?;
    remove(&bare_top)?;
    remove(&keyed)
}

/// A fixture family resolves a set's name at one place: the base name on a base fixture, `fx_<tag>_` on a
/// variant's. A set dumped from one file is stale for the other, at either name: the two files share a file name,
/// so the `# fixture` line refuses it by its variant key, naming both lines and the file the tree runs.
#[test]
fn a_variants_sets_are_named_and_checked_apart_from_the_bases() -> Result<(), RefError> {
    let top = temp("variant-sets")?;
    let root = top.join("fixtures");
    let data = temp("variant-data")?;
    let u = under("fx-ik-glm5next", "ik-glm5next")?;
    let mut with_key = fixture_kvs(1);
    with_key.push(kv(KEY_VARIANT, Value::String("iq4xs".to_string())));
    let base = root_with(&root, u.arch, &fixture_kvs(1))?;
    let var = root_with(&top.join("fixture-variants/iq4xs"), u.arch, &with_key)?;
    let (base_line, var_line) = (line(Path::new(&base))?, line(Path::new(&var))?);
    assert!(
        var_line.contains("bloomery.fixture.variant=iq4xs"),
        "{var_line}"
    );
    assert!(!base_line.contains("variant"), "{base_line}");

    let in_base = |set: &str| with_root(&root, || u.family.path_in(&data, set));
    let in_var = |set: &str| {
        with_root(&root, || {
            with_variant("iq4xs", || u.family.path_in(&data, set))
        })
    };
    assert_eq!(in_base("fx_ref_glm5next"), data.join("fx_ref_glm5next"));
    assert_eq!(
        in_var("fx_ref_glm5next"),
        data.join("fx_iq4xs_ref_glm5next")
    );
    assert_eq!(
        in_var("fx_ref_glm5next_step4"),
        data.join("fx_iq4xs_ref_glm5next_step4")
    );
    // No fixture file to ask: the family's own name, and runs() refuses by name.
    let none = temp("variant-none")?;
    assert_eq!(
        with_root(&none, || with_variant("iq4xs", || u
            .family
            .path_in(&data, "fx_ref_glm5next"))),
        data.join("fx_ref_glm5next")
    );

    let set = temp("variant-set")?;
    // The variant's own set passes under the variant, and is stale under the base fixture.
    write_node_set(&set, u.arch, &var, Some(body(&var_line)), u.build, true);
    let under_var =
        |set: &Path| with_root(&root, || with_variant("iq4xs", || u.family.check_set(set)));
    assert_eq!(under_var(&set)?.dumped_from, var);
    let e = open(u.family, &root, &set).expect_err("a variant set under the base fixture");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {var_line}"))
            && e.contains(&format!("the tree runs {base_line}"))
            && e.contains(&base),
        "{e}"
    );
    // The base's set under the variant: stale on the fixture line, which lacks the variant key.
    write_node_set(&set, u.arch, &base, Some(body(&base_line)), u.build, true);
    let e = under_var(&set).expect_err("a base set under the variant");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {base_line}"))
            && e.contains(&format!("the tree runs {var_line}"))
            && e.contains(&var),
        "{e}"
    );
    // The variant's file with the base's fixture line: the line is what differs.
    write_node_set(&set, u.arch, &var, Some(body(&base_line)), u.build, true);
    let e = under_var(&set).expect_err("the base's fixture line");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    assert!(
        e.to_string().contains("bloomery.fixture.variant=iq4xs"),
        "{e}"
    );

    remove(&top)?;
    remove(&data)?;
    remove(&none)?;
    remove(&set)
}

/// A safetensors shard at `path`: one 1-byte tensor and `meta` as its string `__metadata__`.
fn write_shard(path: &Path, meta: &[(&str, &str)]) {
    let pairs: Vec<String> = meta
        .iter()
        .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
        .collect();
    write_shard_text(path, &pairs.join(","));
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// A shard whose `__metadata__` object body is `meta_body`, written as given.
fn write_shard_text(path: &Path, meta_body: &str) {
    let mut header = format!(
        r#"{{"__metadata__":{{{meta_body}}},"t":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.push(0);
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The keys of a cut folder's first shard, as `tools/fixture-cuts/*.json` and the cutter write them.
const CUT_META: [(&str, &str); 8] = [
    ("format", "pt"),
    ("bloomery.fixture.version", "1"),
    ("bloomery.fixture.source", "MiniMaxAI/MiniMax-M3.1-Flash"),
    ("bloomery.fixture.rev", "76faee1b"),
    ("bloomery.fixture.layers", "0,3,59"),
    ("bloomery.fixture.ff", "512"),
    ("bloomery.fixture.manifest_sha256", "ab12"),
    ("bloomery.fixture.card_budget", "8589934592"),
];

/// A checkpoint folder's fixture is the folder's first shard beside its index: `first_shard_in`
/// takes the one `*-00001-of-*.safetensors` of a directory that holds `model.safetensors.index.json`,
/// refuses by name none, two, and a folder beside a GGUF first shard, and a shard with no index
/// beside it is no fixture; the GGUF rule of a directory with no index is as it was.
#[test]
fn a_cut_folders_first_shard_is_the_one_beside_its_index() -> Result<(), RefError> {
    let root = temp("folder-shard")?;
    let rs = root.display().to_string();
    let dir = root.join("minimaxm31");
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let put = |name: &str| {
        write_shard(&dir.join(name), &CUT_META);
    };
    put("model-00001-of-00002.safetensors");
    put("model-00002-of-00002.safetensors");
    let named = |e: RefError| e.to_string();

    let e = named(first_shard_in(&rs, "minimax-m3").expect_err("no index"));
    assert!(
        e.contains("holds 0 first shards (*-00001-of-*.gguf)"),
        "a shard with no index is no fixture: {e}"
    );

    std::fs::write(dir.join("model.safetensors.index.json"), b"{}")
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    assert_eq!(
        first_shard_in(&rs, "minimax-m3")?,
        format!("{rs}/minimaxm31/model-00001-of-00002.safetensors")
    );

    put("other-00001-of-00003.safetensors");
    let e = named(first_shard_in(&rs, "minimax-m3").expect_err("two first shards"));
    assert!(
        e.contains("holds 2 first shards (*-00001-of-*.safetensors)")
            && e.contains("model-00001-of-00002.safetensors")
            && e.contains("other-00001-of-00003.safetensors")
            && e.contains("fixture cut"),
        "{e}"
    );
    std::fs::remove_file(dir.join("other-00001-of-00003.safetensors"))
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;

    std::fs::write(dir.join("x-00001-of-00001.gguf"), b"")
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let e = named(first_shard_in(&rs, "minimax-m3").expect_err("both"));
    assert!(
        e.contains("holds both") && e.contains("x-00001-of-00001.gguf"),
        "{e}"
    );
    remove(&root)
}

/// The line of a cut folder is its first shard's `__metadata__` keys under `bloomery.fixture.`
/// sorted, as a GGUF's header keys are, without the card budget and the keys of other prefixes,
/// and a set dumped from it is current for the same keys and stale for any key that moved.
#[test]
fn a_cut_folders_line_is_its_first_shards_metadata_keys() -> Result<(), RefError> {
    let dir = temp("folder-line")?;
    let shard = dir.join("model-00001-of-00002.safetensors");
    write_shard(&shard, &CUT_META);
    let want = "# fixture\tbloomery.fixture.ff=512 bloomery.fixture.layers=0,3,59 \
                bloomery.fixture.manifest_sha256=ab12 bloomery.fixture.rev=76faee1b \
                bloomery.fixture.source=MiniMaxAI/MiniMax-M3.1-Flash bloomery.fixture.version=1";
    assert_eq!(line(&shard)?, want);
    assert_eq!(variant_of(&shard)?, None, "a folder fixture has no variant");

    let stated = want.strip_prefix(LINE_PREFIX).expect("a fixture line");
    check_stated(Path::new("set"), Some(stated), &shard)?;
    let mut moved = CUT_META;
    moved[5] = ("bloomery.fixture.ff", "256");
    write_shard(&shard, &moved);
    let e = check_stated(Path::new("set"), Some(stated), &shard).expect_err("ff moved");
    assert!(
        matches!(e, RefError::Stale { .. }) && e.to_string().contains("bloomery.fixture.ff=256"),
        "{e}"
    );
    remove(&dir)
}

/// The refusals of a cut folder's line, each by name, the message naming the file and the key: no
/// metadata or no version key (not a fixture), a subset key, a value that is empty or holds
/// whitespace, `=` or a control character, a key given twice, and a file that is no safetensors
/// header.
#[test]
fn a_cut_folders_line_refuses_what_it_cannot_state() -> Result<(), RefError> {
    let dir = temp("folder-refuse")?;
    let shard = dir.join("model-00001-of-00001.safetensors");
    fn with<'a>(extra: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
        let mut v = CUT_META.to_vec();
        v.extend_from_slice(extra);
        v
    }
    let refused = |what: &str| -> String {
        let e = line(&shard).expect_err(what).to_string();
        assert!(e.contains(&shard.display().to_string()), "{what}: {e}");
        e
    };

    write_shard(&shard, &[("format", "pt")]);
    assert!(matches!(line(&shard), Err(RefError::Missing { .. })));
    assert!(refused("no version").contains("bloomery.fixture.version"));

    write_shard(&shard, &with(&[("bloomery.fixture.subset", "x")]));
    assert!(matches!(line(&shard), Err(RefError::Malformed { .. })));
    assert!(refused("subset").contains("bloomery.fixture.subset"));

    for (value, why) in [
        ("", "empty"),
        ("a b", "whitespace"),
        ("a=b", "="),
        ("a\u{1}b", "control"),
    ] {
        write_shard(&shard, &with(&[("bloomery.fixture.bad", value)]));
        assert!(
            matches!(line(&shard), Err(RefError::Malformed { .. })),
            "{why}"
        );
        assert!(refused(why).contains("bloomery.fixture.bad"), "{why}");
    }

    let twice = CUT_META
        .iter()
        .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
        .chain([format!("{}:\"2\"", json_str("bloomery.fixture.version"))])
        .collect::<Vec<_>>()
        .join(",");
    write_shard_text(&shard, &twice);
    assert!(refused("twice").contains("given twice"));

    write_shard_text(&shard, r#""bloomery.fixture.version":1"#);
    assert!(refused("a number").contains("bloomery.fixture.version"));

    std::fs::write(&shard, b"not a safetensors header at all")
        .map_err(|e| RefError::missing(&shard, e.to_string()))?;
    assert!(matches!(line(&shard), Err(RefError::Malformed { .. })));
    let gone = dir.join("gone.safetensors");
    assert!(matches!(line(&gone), Err(RefError::Missing { .. })));
    remove(&dir)
}

/// The arch of a cut folder's row resolves through the table to the folder's first shard and the
/// shell's variant check holds: with no variant selected it is the shard, with one selected the
/// shard (which carries no variant key) is refused by name as a GGUF with no key is.
#[test]
fn a_cut_folder_resolves_through_the_table_and_takes_no_variant() -> Result<(), RefError> {
    let top = temp("folder-table")?;
    let root = top.join("fixtures");
    let dir = root.join(dir_of("minimax-m3").expect("the table's minimax-m3 row"));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    write_shard(&dir.join("model-00001-of-00001.safetensors"), &CUT_META);
    std::fs::write(dir.join("model.safetensors.index.json"), b"{}")
        .map_err(|e| RefError::missing(&dir, e.to_string()))?;
    let first = with_root(&root, || first_shard("minimax-m3"))?;
    assert!(
        first.ends_with("/model-00001-of-00001.safetensors"),
        "{first}"
    );
    let variants = top
        .join("fixture-variants/iq4xs")
        .join(dir.file_name().expect("a dir name"));
    std::fs::create_dir_all(&variants).map_err(|e| RefError::missing(&variants, e.to_string()))?;
    write_shard(
        &variants.join("model-00001-of-00001.safetensors"),
        &CUT_META,
    );
    std::fs::write(variants.join("model.safetensors.index.json"), b"{}")
        .map_err(|e| RefError::missing(&variants, e.to_string()))?;
    let e = with_root(&root, || {
        with_variant("iq4xs", || first_shard("minimax-m3"))
    })
    .expect_err("a variant of a folder fixture")
    .to_string();
    assert!(
        e.contains(VARIANT_ENV) && e.contains("\"iq4xs\"") && e.contains("none"),
        "{e}"
    );
    remove(&top)
}
