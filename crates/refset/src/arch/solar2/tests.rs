use super::read::{greedy, tokens, topk};
use super::{ARCH, MODEL, PAI, PAI_BUILD, SETS, fixture};
use crate::RefError;
use crate::family::Identity;
use crate::ik::RefManifest;
use std::path::{Path, PathBuf};

/// A fresh directory for one test's set; tests of one process run in parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-solar2-{what}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

fn put(dir: &Path, name: &str, text: &str) {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

const TOPK: &str = "0\t4498\t21.5\t-1.0481146574020386\n1\t6172\t20.25\t-1.2878550291061401\n2\t8081\t20.25\t-3.4789884090423584\n";
const GREEDY: &str =
    "# step\tid\tmargin\tlogprob\n0\t4498\t0.25\t-1.0481146574020386\n1\t20915\t1.5\t-0.5\n";

/// A set of the dumper's shape at `dir`: the header lines given, one tensor row, the two tables and, when `complete`, the trailer.
fn write(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) {
    let mut lines = vec![
        "# solar2-dump — the PAI tree's intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# tokens\t4677,10678,129326".to_string(),
        "# topk\ttopk.tsv\t3".to_string(),
        "# greedy\tgreedy.tsv\t2".to_string(),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
    ];
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    put(dir, "MANIFEST.tsv", &(lines.join("\n") + "\n"));
    put(dir, "topk.tsv", TOPK);
    put(dir, "greedy.tsv", GREEDY);
}

fn check(dir: &Path, model: &str, build: &str, arch: &str, complete: bool) -> Result<(), RefError> {
    write(dir, model, build, arch, complete);
    PAI.check_set(dir).map(|_| ())
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// A complete set of the Q4_K_M file, the pinned build and this architecture passes, and states that file and that build.
#[test]
fn the_family_takes_a_set_of_its_file_build_and_arch() -> Result<(), RefError> {
    let dir = set_dir("own")?;
    check(&dir, MODEL, PAI_BUILD, ARCH, true)?;
    let p = PAI.check_set(&dir)?;
    assert_eq!(p.dumped_from, MODEL);
    assert_eq!(p.build.as_deref(), Some(PAI_BUILD));
    remove(&dir)
}

/// A set of another file is stale, naming the set, the file it states and the file the tree runs: the IQ4_XS split set (the
/// same vocabulary and architecture, so only this line tells it from the Q4_K_M), a shard that is not the first, and the same
/// shard name in another directory.
#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let set = dir.display().to_string();
    for other in [
        "/models/Solar-Open2-250B-IQ4_XS/Solar-Open2-250B-IQ4_XS-00001-of-00004.gguf",
        "/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00002-of-00004.gguf",
        "/models/elsewhere/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf",
    ] {
        match check(&dir, other, PAI_BUILD, ARCH, true) {
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

/// A set whose build line names another tree, or the same commit without the patch (the unpatched tree has no such
/// architecture), is foreign by its build field, named for this family.
#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    let dir = set_dir("build")?;
    for build in ["6ea215d17", "6ea215d17+pai-000000000000", "f711c8f3"] {
        match check(&dir, MODEL, build, ARCH, true) {
            Err(RefError::Foreign {
                field: "build",
                family: "pai-solar2",
                got,
                want,
                ..
            }) if got == build && want == PAI_BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set whose `# arch` names another architecture is foreign by its arch field.
#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    let dir = set_dir("arch")?;
    for arch in ["glm5next", "solar_open2"] {
        match check(&dir, MODEL, PAI_BUILD, arch, true) {
            Err(RefError::Foreign {
                field: "arch",
                family: "pai-solar2",
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
    match check(&dir, MODEL, PAI_BUILD, ARCH, false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    remove(&dir)
}

/// The prompt's ids, the top-k table and the greedy continuation read as the dumper wrote them.
#[test]
fn the_tables_read_as_written() -> Result<(), RefError> {
    let dir = set_dir("read")?;
    write(&dir, MODEL, PAI_BUILD, ARCH, true);
    let man = RefManifest::open(&dir, &PAI)?;
    assert_eq!(tokens(&man)?, [4677, 10678, 129_326]);
    let top = topk(&man)?;
    assert_eq!(top.len(), 3);
    assert_eq!((top[0].id, top[0].logit), (4498, 21.5));
    assert!((top[1].logprob + 1.287_855_029_106_140_1).abs() < 1e-15);
    let g = greedy(&man)?;
    assert_eq!(g.ids, [4498, 20_915]);
    assert_eq!(g.margins, [0.25, 1.5]);
    remove(&dir)
}

/// A table that breaks its own order, repeats an id, states another count, or whose greedy continuation starts at another
/// token than the top-k table's argmax is refused by name, never read as a shorter or reordered table.
#[test]
fn a_table_that_disagrees_with_itself_is_refused() -> Result<(), RefError> {
    let dir = set_dir("bad")?;
    write(&dir, MODEL, PAI_BUILD, ARCH, true);
    let man = RefManifest::open(&dir, &PAI)?;
    let cases = [
        (
            "topk.tsv",
            "0\t4498\t21.5\t-1.0\n2\t6172\t20.25\t-1.2\n",
            "rank 2, want 1",
        ),
        (
            "topk.tsv",
            "0\t4498\t21.5\t-1.0\n1\t4498\t20.25\t-1.2\n",
            "descending order of distinct ids",
        ),
        (
            "topk.tsv",
            "0\t4498\t21.5\t-1.0\n1\t6172\t22.0\t-1.2\n",
            "descending order of distinct ids",
        ),
        ("topk.tsv", "0\t4498\t21.5\t0.5\n", "logprobs at most 0"),
        (
            "topk.tsv",
            "0\t4498\t21.5\t-1.0\n",
            "1 rows, the manifest states 3",
        ),
    ];
    for (file, text, says) in cases {
        put(&dir, file, text);
        match topk(&man) {
            Err(e) if e.to_string().contains(says) => {}
            r => panic!("{file} {text:?}: {r:?}"),
        }
        put(&dir, file, TOPK);
    }
    put(&dir, "greedy.tsv", "0\t4498\t-0.5\t-1.0\n");
    assert!(matches!(greedy(&man), Err(e) if e.to_string().contains("margin of at least 0")));
    put(&dir, "greedy.tsv", "0\t9\t0.5\t-1.0\n1\t9\t0.5\t-1.0\n");
    assert!(
        matches!(greedy(&man), Err(e) if e.to_string().contains("not the top-k table's first id"))
    );
    put(&dir, "greedy.tsv", "0\t4498\t0.5\t-1.0\n");
    assert!(
        matches!(greedy(&man), Err(e) if e.to_string().contains("1 steps, the manifest states 2"))
    );
    remove(&dir)
}

/// The fixture family mirrors the real one as every fixture family does (`arch::tests`): the same architecture, build and consumers,
/// the fixture twin of its identity, no per-file set name, and each set the real family's behind `fx_`.
#[test]
fn the_fixture_family_mirrors_the_real_one() {
    let fx = &fixture::PAI;
    assert_eq!(
        (fx.arch, fx.build, fx.consumers),
        (PAI.arch, PAI.build, PAI.consumers)
    );
    assert_eq!(
        (PAI.identity, fx.identity),
        (Identity::Manifest, Identity::FixtureManifest)
    );
    assert!(fx.resolve.is_none(), "one fixture file, no per-file suffix");
    assert!(
        fx.recipe
            .starts_with("BLOOMERY_TIER=fixture just dump-ref-"),
        "{}",
        fx.recipe
    );
    let twins: Vec<String> = fx
        .sets
        .iter()
        .map(|&s| s.strip_prefix("fx_").unwrap_or(s).to_string())
        .collect();
    assert_eq!(twins, SETS, "each fixture set is the real set behind fx_");
}
