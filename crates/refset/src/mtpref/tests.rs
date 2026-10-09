use super::{Graph, MtpSet};
use crate::RefError;
use crate::dsref::{Draft, Plain};
use crate::family::{Build, Family, Identity};
use crate::ik::{FileElem, Layout, RowKind};
use std::path::{Path, PathBuf};

const MODEL: &str = "/models/P/M-00001-of-00006.gguf";
const ARCH: &str = "a";
const MTP_BUILD: &str = "b0";

fn runs() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

static MTP: Family = Family {
    name: "test-mtp",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::MtpManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(MTP_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: &[],
};

/// The column lines `dump_mtp` writes.
const COLUMNS: [&str; 6] = [
    "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph",
    "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\tblock\trow\taccepted\tgraph",
    "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph",
    "# draft\tblock\trow\ttoken",
    "# verify\tblock\tpos\tid_last\tcarry\tdrafted\taccepted\ttarget",
    "# plain\tpos\ttoken",
];

const WARMUP: &str = "tensor\tmtp_fused-45\t0\tf32\t4096\t64\t1\t1\t1048576\t0.5\tMUL_MAT\t1\t0\tblk.45.nextn.eh_proj.weight\tmtp_concat-45\t-1\t-\t-\twarmup";
const GEN: &str = "tensor\tresult_output\t0\tf32\t8\t1\t1\t1\t32\t1.5\tMUL_MAT\t1\t0\toutput.weight\tresult_norm\t0\t-\t1\tgen";
const UPDATE: &str = "tensor\tresult_output\t0\tf32\t8\t1\t1\t1\t32\t2.5\tMUL_MAT\t1\t0\toutput.weight\tresult_norm\t0\t-\t1\tupdate";
const INPUT: &str =
    "input\tinp_tokens\t0\ti32\t2\t1\t1\t1\t8\t30\tNONE\t1\t0\t-\t-\t0\t-\t1\tupdate";
const INT: &str = "int\tinp_tokens\t0\tinput\ti32\ti32\tflat\t2\t8\t30\t16\tb0.update.inp_tokens.0.input.i32\t0\t-\t1\tupdate";

/// Block 0 from the carry-less start token 4: proposal 15 accepted, the
/// target's bonus 99; block 1 from the carry 99: proposal 20 rejected for 21.
const BLOCKS: [&str; 4] = [
    "draft\t0\t0\t15",
    "verify\t0\t64\t4\t0\t1\t1\t4,15,99",
    "draft\t1\t0\t20",
    "verify\t1\t66\t99\t1\t1\t0\t21",
];

fn set_dir(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bloomery-mtpref-{what}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

/// A set of `rows` under the header `dump_mtp` writes, stating `model`,
/// `build` and `arch`, with its trailer when `complete`.
fn write_set(dir: &Path, model: &str, build: &str, arch: &str, rows: &[&str], complete: bool) {
    let mut lines = vec![
        "# dump_mtp — ik_llama.cpp MTP (NextN) draft tensors, raw f32, little-endian".to_string(),
        format!("# model\t{model}"),
        format!("# build\t{build}"),
        format!("# arch\t{arch}"),
        "# model_file\tM-00001-of-00006.gguf".to_string(),
        "# tokens\t4,8,15".to_string(),
        "# spec\tmtp:n_max=1".to_string(),
    ];
    lines.extend(COLUMNS.iter().map(|c| c.to_string()));
    lines.extend(rows.iter().map(|r| r.to_string()));
    lines.push("# graphs\twarmup\t1\tgen\t1\tupdate\t2".to_string());
    if complete {
        lines.push("# complete\t3\t0".to_string());
    }
    std::fs::write(dir.join("MANIFEST.tsv"), lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

/// The family's own set: the file, build and arch it pins, every row kind.
fn own_rows() -> Vec<&'static str> {
    let mut rows = vec![WARMUP, GEN, UPDATE, INPUT, INT, "plain\t130\t7"];
    rows.extend(BLOCKS);
    rows
}

fn open_own(dir: &Path, rows: &[&str]) -> Result<MtpSet, RefError> {
    write_set(dir, MODEL, MTP_BUILD, ARCH, rows, true);
    MtpSet::open(dir, &MTP)
}

fn remove(dir: &Path) {
    std::fs::remove_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
}

/// Every row kind reads by its column line: a node's dims, op and sources in
/// its block and graph, an input and its integer twin, the proposals, the
/// verify rows with their targets past the carry, a plain step; a file's
/// name carries its block and its graph; the family's check passes and
/// states the file and the build.
#[test]
fn an_mtp_set_reads_by_its_column_lines() {
    let dir = set_dir("read");
    let set = open_own(&dir, &own_rows()).unwrap_or_else(|e| panic!("{e}"));
    let fused = set
        .find(-1, Graph::Warmup, "mtp_fused-45", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (fused.ne, fused.op.as_str(), fused.src1.as_deref()),
        ([4096, 64, 1, 1], "MUL_MAT", Some("mtp_concat-45"))
    );
    assert_eq!(
        MtpSet::file_name(fused, Layout::Flat, FileElem::F32),
        "w.warmup.mtp_fused-45.0.f32"
    );
    let gen_out = set
        .find(0, Graph::Gen, "result_output", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    let upd_out = set
        .find(0, Graph::Update, "result_output", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((gen_out.sum, upd_out.sum), (1.5, 2.5));
    assert_eq!(
        MtpSet::file_name(upd_out, Layout::Logical, FileElem::F32),
        "b0.update.result_output.0.logical.f32"
    );
    assert!(set.find(1, Graph::Update, "result_output", 0).is_err());
    assert_eq!(set.in_graph(0, Graph::Update).len(), 2);
    let tokens = set
        .find(0, Graph::Update, "inp_tokens", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(tokens.kind, RowKind::Input);
    assert_eq!(
        MtpSet::file_name(tokens, Layout::Flat, FileElem::I32),
        "b0.update.inp_tokens.0.input.i32"
    );
    let int = set
        .int(0, Graph::Update, "inp_tokens", 0, Layout::Flat)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (int.count, int.file.as_str()),
        (
            2,
            MtpSet::file_name(tokens, Layout::Flat, FileElem::I32).as_str()
        )
    );
    assert_eq!(
        set.drafts[0],
        Draft {
            block: 0,
            row: 0,
            token: 15
        }
    );
    assert_eq!((set.drafts_of(0), set.drafts_of(1)), (vec![15], vec![20]));
    let v0 = set.verify_of(0).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (v0.id_last, v0.accepted, v0.targets()),
        (4, 1, &[15, 99][..])
    );
    let v1 = set.verify_of(1).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((v1.carry, v1.accepted, v1.targets()), (1, 0, &[21][..]));
    assert_eq!(set.blocks(), [0, 1]);
    assert_eq!(set.plain, [Plain { pos: 130, token: 7 }]);
    assert_eq!(set.tokens.as_deref(), Some(&[4, 8, 15][..]));
    let p = MTP.check_set(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref(), p.draft),
        (MODEL, Some(MTP_BUILD), None)
    );
    remove(&dir);
}

/// A set of another file — another quantization's first shard, or the same
/// shard name in another directory — is stale, naming both paths.
#[test]
fn an_mtp_set_of_another_file_is_stale() {
    let dir = set_dir("stale");
    for other in [
        "/models/Q/M-00001-of-00006.gguf",
        "/models/elsewhere/M-00001-of-00006.gguf",
    ] {
        write_set(&dir, other, MTP_BUILD, ARCH, &own_rows(), true);
        match MtpSet::open(&dir, &MTP) {
            Err(RefError::Stale {
                dumped_from, runs, ..
            }) => assert_eq!((dumped_from.as_str(), runs.as_str()), (other, MODEL)),
            r => panic!("a set of {other}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A set of another ik build, or of the pinned tree with local changes, is
/// foreign by its build field, named for the family.
#[test]
fn an_mtp_set_of_another_build_is_foreign() {
    let dir = set_dir("build");
    for build in ["b1", "b0-dirty"] {
        write_set(&dir, MODEL, build, ARCH, &own_rows(), true);
        match MtpSet::open(&dir, &MTP) {
            Err(RefError::Foreign {
                field: "build",
                family: "test-mtp",
                got,
                want,
                ..
            }) if got == build && want == MTP_BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    remove(&dir);
}

/// A set naming another architecture is foreign by its arch field.
#[test]
fn an_mtp_set_of_another_arch_is_foreign() {
    let dir = set_dir("arch");
    write_set(&dir, MODEL, MTP_BUILD, "x", &own_rows(), true);
    match MtpSet::open(&dir, &MTP) {
        Err(RefError::Foreign {
            field: "arch", got, ..
        }) if got == "x" => {}
        r => panic!("a set of architecture x: {r:?}"),
    }
    remove(&dir);
}

/// A family whose draft layer is the target file's takes no draft file: a
/// set stating one is foreign by its `draft_model`, never read past it.
#[test]
fn a_draft_file_for_a_family_without_one_is_foreign() {
    let dir = set_dir("draft");
    write_set(&dir, MODEL, MTP_BUILD, ARCH, &own_rows(), true);
    let path = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}"));
    let text = text.replacen("# build", "# draft_model\t/models/P/draft.gguf\n# build", 1);
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("{e}"));
    match MtpSet::open(&dir, &MTP) {
        Err(RefError::Foreign {
            field: "draft_model",
            got,
            ..
        }) if got == "/models/P/draft.gguf" => {}
        r => panic!("a set stating a draft file: {r:?}"),
    }
    remove(&dir);
}

/// A set without its completion trailer is unfinished, by the set's name.
#[test]
fn an_mtp_set_without_its_trailer_is_unfinished() {
    let dir = set_dir("trailer");
    write_set(&dir, MODEL, MTP_BUILD, ARCH, &own_rows(), false);
    match MtpSet::open(&dir, &MTP) {
        Err(RefError::Unfinished { set }) if set == dir.display().to_string() => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    remove(&dir);
}

/// A graph label that is no MTP op — a draft set's `block`, say — is
/// `Malformed` naming it, never read as one of the three.
#[test]
fn an_unknown_graph_label_is_malformed() {
    let dir = set_dir("graph");
    let rows = [UPDATE.replace("\tupdate", "\tblock")];
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
    write_set(&dir, MODEL, MTP_BUILD, ARCH, &rows, true);
    match MtpSet::read(&dir) {
        Err(RefError::Malformed { what, .. }) => {
            assert!(what.starts_with("graph \"block\""), "{what}");
        }
        r => panic!("a graph labelled block: {r:?}"),
    }
    remove(&dir);
}

/// `blocks` in place of the set's own draft and verify rows must be
/// `Malformed` with `want` in the reason, by the block.
fn bad_blocks(what: &str, blocks: &[&str], want: &str) {
    let dir = set_dir(what);
    let mut rows = vec![WARMUP, GEN, UPDATE];
    rows.extend(blocks);
    match open_own(&dir, &rows) {
        Err(RefError::Malformed { at, what }) => {
            assert!(
                at.ends_with("block 0") && what.contains(want),
                "{at}: {what}"
            );
        }
        r => panic!("{blocks:?}: {r:?}"),
    }
    remove(&dir);
}

/// The proposal is ik's only when it agrees with ik's verdict: accepted with
/// another token than the target's, or rejected though it is the target's
/// token, is `Malformed`.
#[test]
fn a_proposal_that_disagrees_with_the_verdict_is_malformed() {
    bad_blocks(
        "accepted",
        &["draft\t0\t0\t16", "verify\t0\t64\t4\t0\t1\t1\t4,15,99"],
        "proposal 16, target 15, accepted 1",
    );
    bad_blocks(
        "rejected",
        &["draft\t0\t0\t15", "verify\t0\t64\t4\t0\t1\t0\t4,15"],
        "proposal 15, target 15, accepted 0",
    );
}

/// One proposal a block, one drafted token: a second draft row, none, a
/// round that drafted two, or a draft row with no verify row is
/// `Malformed`.
#[test]
fn a_block_that_is_not_one_token_is_malformed() {
    bad_blocks(
        "two",
        &[
            "draft\t0\t0\t15",
            "draft\t0\t1\t16",
            "verify\t0\t64\t4\t0\t1\t1\t4,15,99",
        ],
        "2 draft rows",
    );
    bad_blocks(
        "none",
        &["verify\t0\t64\t4\t0\t1\t1\t4,15,99"],
        "0 draft rows",
    );
    bad_blocks(
        "drafted",
        &["draft\t0\t0\t15", "verify\t0\t64\t4\t0\t2\t1\t4,15,99"],
        "drafted 2",
    );
    bad_blocks("orphan", &["draft\t0\t0\t15"], "no verify row");
}

/// A node's f32 file holds as many values as its row's `ne`: one of another
/// length is `Malformed` naming it.
#[test]
fn a_node_file_of_another_length_is_malformed() {
    let dir = set_dir("f32-len");
    let set = open_own(&dir, &own_rows()).unwrap_or_else(|e| panic!("{e}"));
    let out = set
        .find(0, Graph::Gen, "result_output", 0)
        .unwrap_or_else(|e| panic!("{e}"));
    let file = dir.join(MtpSet::file_name(out, Layout::Flat, FileElem::F32));
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let logits = [0.0, 1.0, -1.0, 0.5, 0.25, 0.0, 0.0, -0.25];
    std::fs::write(&file, f32s(&logits)).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.f32s(out).unwrap_or_else(|e| panic!("{e}")), logits);
    std::fs::write(&file, f32s(&logits[..7])).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(set.f32s(out), Err(RefError::Malformed { .. })));
    remove(&dir);
}
