use super::{
    DEFAULT_ROOT, DIRS, KEY_SUBSET, KEY_VERSION, ROOT_ENV, first_shard_in, keys, line, render,
    root_of, with_root,
};
use crate::RefError;
use crate::arch::named;
use crate::family::{Build, Family, Provenance};
use crate::ik::RefManifest;
use crate::mtpref::MtpSet;
use gguf::Value;
use gguf::write::{Layout, Writer};
use std::collections::BTreeSet;
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

/// The line of [`fixture_kvs`] of seed 1, written out.
const LINE_SEED_1: &str = "# fixture\tbloomery.fixture.card_budget=8589934592 bloomery.fixture.seed=1 \
                           bloomery.fixture.source_header_sha256=0123abcd \
                           bloomery.fixture.source_layers=0,3,5 bloomery.fixture.version=1";

/// `kvs` written to `<dir>/f.gguf` and the line read back.
fn line_of(dir: &Path, kvs: &[(String, Value)]) -> Result<String, RefError> {
    let path = dir.join("f.gguf");
    write_gguf(&path, kvs);
    line(&path)
}

/// The line holds every `bloomery.fixture.` key, sorted by byte order, each
/// `key=value` and one space between, whatever order the file lists them in;
/// a key that only starts like the prefix is not in it.
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

/// The directory table and the root default are `tools/ref/ref-paths.sh`'s:
/// every `__fixture_dir=<dir>` row of its `case "$BLOOMERY_MODEL"` that names
/// a directory (neither `self` nor `none`) is a row here and the reverse, the
/// default of its `__fixture_root` is [`DEFAULT_ROOT`] under [`ROOT_ENV`], and
/// the file glob and the join are the ones [`first_shard_in`] matches and forms.
#[test]
fn the_table_is_the_shells() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/ref-paths.sh");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    let mut shell: BTreeSet<(String, String)> = BTreeSet::new();
    for l in text
        .lines()
        .filter(|l| l.contains("__fixture_dir=") && l.contains(";;"))
    {
        let (arches, dir) = l
            .split_once(')')
            .and_then(|(a, rest)| {
                let (_, after) = rest.split_once("__fixture_dir=")?;
                let (dir, _) = after.split_once(";;")?;
                Some((a.trim(), dir.trim()))
            })
            .unwrap_or_else(|| panic!("a `__fixture_dir=` row this test cannot read: {l:?}"));
        if arches == "*" || matches!(dir, "self" | "none") {
            continue;
        }
        for arch in arches.split('|').map(str::trim) {
            assert!(!arch.is_empty(), "{l:?}");
            shell.insert((arch.to_string(), dir.to_string()));
        }
    }
    let ours: BTreeSet<(String, String)> = DIRS
        .iter()
        .map(|(a, d)| (a.to_string(), d.to_string()))
        .collect();
    assert_eq!(
        shell,
        ours,
        "refset's fixture directories and {}'s case rows differ",
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
    let (_, dir) = DIRS
        .iter()
        .find(|(a, _)| *a == arch)
        .unwrap_or_else(|| panic!("no fixture directory for {arch}"));
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
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
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
    std::fs::write(&path, lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The same cases for an MTP set (GLM's, whose target carries the draft
/// layer) and for the family with a draft file of its own (Qwen3.8's): a set
/// of the fixture passes and states the draft beside the target; a line that
/// differs in the seed, or none, is stale and names both lines; the real
/// file's set is stale on `# model`; a set that states the real draft is stale
/// on the draft; a draft missing beside the target is `Missing`.
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
    let e = open(q.family, &qroot, &set).expect_err("the real draft");
    assert!(matches!(e, RefError::Stale { .. }), "{e:?}");
    let e = e.to_string();
    assert!(
        e.contains(&format!("dumped from {real_draft}"))
            && e.contains(&format!("the tree runs {draft}")),
        "{e}"
    );

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
