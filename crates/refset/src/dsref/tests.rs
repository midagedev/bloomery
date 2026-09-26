use super::{Draft, Dsref, Graph, Plain};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::ik::{FileElem, Layout, RowKind};
use std::path::{Path, PathBuf};

/// The column lines `dump_draft` writes.
const COLUMNS: [&str; 6] = [
    "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph",
    "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\tblock\trow\taccepted\tgraph",
    "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph",
    "# draft\tblock\trow\ttoken",
    "# verify\tblock\tpos\tid_last\tcarry\tdrafted\taccepted\ttarget",
    "# plain\tpos\ttoken",
];

const TENSOR: &str = "tensor\thc_pre_mixes-0\t0\tf32\t24\t3\t1\t1\t288\t0.5\tMUL_MAT\t1\t0\thc_attn_fn\tnode_7\t0\t-\t2\tblock";
const INPUT: &str =
    "input\tCUDA0#inp_pos#0\t0\ti32\t3\t1\t1\t1\t12\t3\tNONE\t1\t0\t-\t-\t0\t-\t2\tblock";
const KV: &str =
    "tensor\tdflash_kv_fused_target\t0\tf32\t4\t2\t1\t1\t32\t0\tADD\t1\t0\ta\tb\t0\t-\t2\tkv";
const INT: &str = "int\tCUDA0#inp_pos#0\t0\tinput\ti32\ti32\tflat\t3\t12\t3\t2\tb0.CUDA0#inp_pos#0.0.input.i32\t0\t-\t2\tblock";

fn set_dir(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bloomery-dsref-{what}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

/// A draft set of `rows` under the header `dump_draft` writes, from files
/// `model` and `draft`.
fn write_set(dir: &Path, model: &str, draft: &str, rows: &[&str]) {
    let mut lines = vec![
        "# dump_draft — ik_llama.cpp DSpark draft tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        "# build\tb0+draft 1".to_string(),
        "# arch\ta".to_string(),
        "# model_file\tM-00001-of-00009.gguf".to_string(),
        format!("# draft_model\t{draft}"),
    ];
    lines.extend(COLUMNS.iter().map(|c| c.to_string()));
    lines.extend(rows.iter().map(|r| r.to_string()));
    lines.push("# complete\t3\t0".to_string());
    let text = lines.join("\n") + "\n";
    std::fs::write(dir.join("MANIFEST.tsv"), text)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

fn runs() -> String {
    "/models/P/M-00001-of-00009.gguf".to_string()
}

fn draft_runs() -> Result<String, RefError> {
    Ok("/models/D/draft.gguf".to_string())
}

static FAMILY: Family = Family {
    name: "test-dsref",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::ManifestAndDraft,
    arch: Some("a"),
    build: Some(Build::Patched("b0")),
    runs: Some(runs),
    draft_runs: Some(draft_runs),
    consumers: &[],
};

/// Every row kind reads by its column line: a node's dims, op and sources
/// with its block and graph, an input and its integer twin, the proposals,
/// the verify row with its targets past the carry, a plain step; a file's
/// name carries its block, its graph and its layout.
#[test]
fn a_draft_set_reads_by_its_column_lines() {
    let dir = set_dir("read");
    let rows = [
        TENSOR,
        INPUT,
        KV,
        INT,
        "draft\t0\t0\t15",
        "draft\t0\t1\t16",
        "verify\t0\t64\t4\t0\t2\t1\t4,15,99",
        "plain\t95\t7",
        "skip\tx\t0\tq8_0\tquantized",
    ];
    write_set(&dir, &runs(), "/models/D/draft.gguf", &rows);
    let set = Dsref::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    let mix = set
        .find(0, Graph::Block, "hc_pre_mixes-0", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (mix.ne, mix.op.as_str(), mix.src0.as_deref()),
        ([24, 3, 1, 1], "MUL_MAT", Some("hc_attn_fn"))
    );
    assert_eq!(mix.block, 0);
    assert!(set.find(0, Graph::Kv, "hc_pre_mixes-0", 0).is_err());
    let op = set
        .find_op(0, "MUL_MAT", "hc_attn_fn", "")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(op.name, "hc_pre_mixes-0");
    assert_eq!(set.in_block(0, Graph::Block).len(), 2);
    assert_eq!(set.in_block(0, Graph::Kv).len(), 1);
    let pos = set
        .find(0, Graph::Block, "CUDA0#inp_pos#0", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(pos.kind, RowKind::Input);
    assert_eq!(
        Dsref::file_name(pos, Layout::Flat, FileElem::I32),
        "b0.CUDA0#inp_pos#0.0.input.i32"
    );
    let kv = set
        .find(0, Graph::Kv, "dflash_kv_fused_target", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        Dsref::file_name(kv, Layout::Logical, FileElem::F32),
        "b0.kv.dflash_kv_fused_target.0.logical.f32"
    );
    let int = set
        .int(0, Graph::Block, "CUDA0#inp_pos#0", 0, Layout::Flat)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (int.count, int.file.as_str()),
        (3, "b0.CUDA0#inp_pos#0.0.input.i32")
    );
    assert_eq!(
        set.drafts,
        [
            Draft {
                block: 0,
                row: 0,
                token: 15
            },
            Draft {
                block: 0,
                row: 1,
                token: 16
            }
        ]
    );
    assert_eq!(set.drafts_of(0), [15, 16]);
    let v = set.verify_of(0).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((v.id_last, v.accepted, v.targets()), (4, 1, &[15, 99][..]));
    assert_eq!(set.blocks(), [0]);
    assert_eq!(set.plain, [Plain { pos: 95, token: 7 }]);
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
}

/// A draft set of another target file, or of another draft file, is
/// `Stale` naming both paths; one of another build, or of the pin with no
/// patch, is `Foreign`.
#[test]
fn a_draft_set_of_another_file_is_stale() {
    let dir = set_dir("stale");
    write_set(
        &dir,
        "/models/Q/M-00001-of-00009.gguf",
        "/models/D/draft.gguf",
        &[TENSOR],
    );
    match Dsref::open(&dir, &FAMILY) {
        Err(RefError::Stale {
            dumped_from,
            runs: r,
            ..
        }) => {
            assert_eq!(
                (dumped_from.as_str(), r),
                ("/models/Q/M-00001-of-00009.gguf", runs())
            );
        }
        other => panic!("another target: {other:?}"),
    }
    write_set(&dir, &runs(), "/models/E/draft.gguf", &[TENSOR]);
    assert!(matches!(
        Dsref::open(&dir, &FAMILY),
        Err(RefError::Stale { .. })
    ));
    write_set(&dir, &runs(), "/models/D/draft.gguf", &[TENSOR]);
    let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).unwrap_or_default();
    for build in ["# build\tb1", "# build\tb0", "# build\tb0+"] {
        std::fs::write(
            dir.join("MANIFEST.tsv"),
            text.replace("# build\tb0+draft 1", build),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            matches!(Dsref::open(&dir, &FAMILY), Err(RefError::Foreign { .. })),
            "{build}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
}

/// `row` with field `field` (its column line's name) replaced by garbage
/// must be `Malformed` naming the field — the readers this crate replaced
/// read these fields as 0.
fn garbage(row: &str, columns: &str, field: &str) {
    let names: Vec<&str> = columns.trim_start_matches("# ").split('\t').collect();
    let at = names
        .iter()
        .position(|n| *n == field)
        .unwrap_or_else(|| panic!("no {field} column"));
    let mut f: Vec<&str> = row.split('\t').collect();
    f[at] = "x7";
    let bad = f.join("\t");
    let dir = set_dir(&format!("garbage-{field}-{}", &row[..3]));
    write_set(&dir, &runs(), "/models/D/draft.gguf", &[&bad]);
    match Dsref::read(&dir) {
        Err(RefError::Malformed { what, .. }) => {
            assert!(what.starts_with(&format!("{field} \"x7\"")), "{what}");
        }
        other => panic!("{field}: {other:?}"),
    }
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn a_garbage_ne0_is_malformed() {
    garbage(TENSOR, COLUMNS[0], "ne0");
    garbage(INPUT, COLUMNS[2], "ne0");
}

#[test]
fn a_garbage_ne1_is_malformed() {
    garbage(TENSOR, COLUMNS[0], "ne1");
    garbage(INPUT, COLUMNS[2], "ne1");
}

#[test]
fn a_garbage_ne2_is_malformed() {
    garbage(TENSOR, COLUMNS[0], "ne2");
    garbage(INPUT, COLUMNS[2], "ne2");
}

#[test]
fn a_garbage_ne3_is_malformed() {
    garbage(TENSOR, COLUMNS[0], "ne3");
    garbage(INPUT, COLUMNS[2], "ne3");
}

#[test]
fn a_garbage_int_count_is_malformed() {
    garbage(INT, COLUMNS[1], "count");
}

/// An integer file holds as many values as its row's `ne`: one of another
/// length is `Malformed` naming it, never a shorter list.
#[test]
fn an_integer_file_of_another_length_is_malformed() {
    let dir = set_dir("i32-len");
    write_set(&dir, &runs(), "/models/D/draft.gguf", &[INPUT]);
    let set = Dsref::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let pos = set
        .find(0, Graph::Block, "CUDA0#inp_pos#0", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    let file = dir.join(Dsref::file_name(pos, Layout::Flat, FileElem::I32));
    let ids = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(&file, ids(&[1, 2, 3])).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        set.i32s(pos, Layout::Flat)
            .unwrap_or_else(|e| panic!("{e}")),
        [1, 2, 3]
    );
    std::fs::write(&file, ids(&[1, 2])).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(
        set.i32s(pos, Layout::Flat),
        Err(RefError::Malformed { .. })
    ));
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
}
