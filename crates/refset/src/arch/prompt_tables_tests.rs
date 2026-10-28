use super::prompt_tables::{greedy, tokens, topk};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::ik::RefManifest;
use std::path::{Path, PathBuf};

const MODEL: &str = "/models/P/M-00001-of-00002.gguf";
const BUILD: &str = "test-build";
const ARCH: &str = "test-arch";

fn runs() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// A manifest family of the shape the node dumps have; the tables do not depend on which one.
static FAMILY: Family = Family {
    name: "test-prompt-tables",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::Manifest,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: &[],
};

/// A fresh directory for one test's set; tests of one process run in parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-prompt-tables-{what}-{}",
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

/// A set of the dumper's shape at `dir`: the header lines, one tensor row, the two tables and the trailer.
fn write(dir: &Path) {
    let lines = [
        "# cb_dump — a llama.cpp tree's intermediate tensors, raw f32, little-endian".to_string(),
        format!("# model\t{MODEL}"),
        format!("# build\t{BUILD}"),
        format!("# arch\t{ARCH}"),
        "# tokens\t4677,10678,129326".to_string(),
        "# topk\ttopk.tsv\t3".to_string(),
        "# greedy\tgreedy.tsv\t2".to_string(),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
        "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
        "# complete\t1\t0".to_string(),
    ];
    put(dir, "MANIFEST.tsv", &{
        let mut out = lines.join("\n");
        out.push('\n');
        out
    });
    put(dir, "topk.tsv", TOPK);
    put(dir, "greedy.tsv", GREEDY);
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// The prompt's ids, the top-k table and the greedy continuation read as the dumper wrote them.
#[test]
fn the_tables_read_as_written() -> Result<(), RefError> {
    let dir = set_dir("read")?;
    write(&dir);
    let man = RefManifest::open(&dir, &FAMILY)?;
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
    write(&dir);
    let man = RefManifest::open(&dir, &FAMILY)?;
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
