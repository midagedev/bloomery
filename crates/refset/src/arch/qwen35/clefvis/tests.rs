//! The clefvis reader against the families' own identity checks: each kind of set built in a
//! scratch directory and read, and each refusal by name.

use super::{PREPROC, TAPS, TOWER_ARCH};
use crate::RefError;
use crate::arch::qwen35::{ARCH, HIDDEN_FLASH_Q8, LCPP_BUILD, MODEL_FLASH_Q8};
use crate::clefvis::{
    ClefvisSet, EMBD, IMAGE_PAD_ID, INP_RAW, Kind, MMPROJ_BF16, MMPROJ_BF16_SHA256, N_EMBD,
    tap_names,
};
use crate::family::{Family, Identity};
use crate::ik::{FileElem, Layout, RowKind, dump_file_name};
use std::path::{Path, PathBuf};

/// The table's family of `kind`.
fn family(kind: Kind) -> &'static Family {
    crate::arch::all()
        .find(|f| f.identity == Identity::Clefvis(kind))
        .unwrap_or_else(|| panic!("{kind:?}: no family in the table"))
}

/// A fresh directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bloomery-clefvis-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The header fields the identity checks read.
struct Head<'a> {
    kind: &'a str,
    model: &'a str,
    build: &'a str,
    arch: &'a str,
    mmproj_sha: &'a str,
    complete: bool,
}

impl<'a> Head<'a> {
    /// A tower set's (A or B): the projector is the model file.
    fn tower(kind: &'a str) -> Head<'a> {
        Head {
            kind,
            model: MMPROJ_BF16,
            build: LCPP_BUILD,
            arch: TOWER_ARCH,
            mmproj_sha: MMPROJ_BF16_SHA256,
            complete: true,
        }
    }

    /// A prompt set's (C, C′ or C″): the text model is the model file.
    fn prompt(kind: &'a str) -> Head<'a> {
        Head {
            kind,
            model: MODEL_FLASH_Q8,
            build: LCPP_BUILD,
            arch: ARCH,
            mmproj_sha: MMPROJ_BF16_SHA256,
            complete: true,
        }
    }

    fn lines(&self) -> Vec<String> {
        vec![
            "# dump_mtmd test".to_string(),
            format!("# model\t{}", self.model),
            format!("# build\t{}", self.build),
            format!("# arch\t{}", self.arch),
            format!("# clefvis\t{}", self.kind),
            format!("# mmproj\t{MMPROJ_BF16}\tsha256\t{}", self.mmproj_sha),
            "# device\tcuda\tthreads\t4\tcard\tNVIDIA RTX A6000\tggml\tCUDA0 NVIDIA RTX A6000"
                .to_string(),
        ]
    }
}

const IMAGE_COLUMNS: &str =
    "# image columns\tname png_sha256 rgb8_sha256 w h best_w best_h nx ny n_tokens n_pos";
/// 96x64 planned 128x96: a 4x3 grid of 12 tokens.
const IMAGE: &str = "# image\tg\taa\tbb\t96\t64\t128\t96\t4\t3\t12\t4";
/// 64x32: a 2x1 grid of 2 tokens, the image of the prompt sets.
const IMAGE_2X1: &str = "# image\tg\taa\tbb\t64\t32\t64\t32\t2\t1\t2\t2";

/// The tensor row and f32 file of shape `ne` filled with `v`.
fn put(dir: &Path, rows: &mut Vec<String>, name: &str, ne: [u64; 4], v: f32) {
    let n: u64 = ne.iter().product();
    let bytes: Vec<u8> = (0..n).flat_map(|_| v.to_le_bytes()).collect();
    let file = dump_file_name(name, 0, RowKind::Tensor, Layout::Flat, FileElem::F32);
    write(&dir.join(file), &bytes);
    rows.push(format!(
        "tensor\t{name}\t0\tf32\t{}\t{}\t{}\t{}\t{}\t{:.6}\tADD\t1\t0\t-\t-",
        ne[0],
        ne[1],
        ne[2],
        ne[3],
        4 * n,
        f64::from(v) * n as f64
    ));
}

fn finish(dir: &Path, head: &Head, extra: &[String], rows: &[String]) {
    let mut l = head.lines();
    l.extend(extra.iter().cloned());
    l.push("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1".to_string());
    l.push(
        "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile"
            .to_string(),
    );
    l.extend(rows.iter().cloned());
    if head.complete {
        let written = rows.iter().filter(|r| r.starts_with("tensor\t")).count();
        l.push(format!("# complete\t{written}\t0"));
    }
    write(&dir.join("MANIFEST.tsv"), (l.join("\n") + "\n").as_bytes());
}

/// A preproc set of one image (128x96, 4x3 tokens).
fn preproc(dir: &Path, head: &Head, image: &str) {
    let mut rows = Vec::new();
    put(
        dir,
        &mut rows,
        &format!("g/{INP_RAW}"),
        [128, 96, 3, 1],
        0.5,
    );
    finish(
        dir,
        head,
        &[IMAGE_COLUMNS.to_string(), image.to_string()],
        &rows,
    );
}

/// A taps set of the one image, tapped in full or `embd` only.
fn taps(dir: &Path, head: &Head, full: bool) {
    let mut rows = Vec::new();
    if full {
        for n in tap_names() {
            let ne = if n.starts_with("Qcur_rope") {
                [72, 16, 48, 1]
            } else {
                [1152, 48, 1, 1]
            };
            put(dir, &mut rows, &format!("g/{n}"), ne, 1.0);
        }
    }
    put(
        dir,
        &mut rows,
        &format!("g/{EMBD}"),
        [N_EMBD as u64, 12, 1, 1],
        2.0,
    );
    let how = if full { "full" } else { "inp_raw only" };
    let effect = format!(
        "# tap_effect\tg\tembd values {}\tdiffering 0\tmax_abs_diff 0\ttaps {how}",
        N_EMBD * 12
    );
    let extra = [IMAGE_COLUMNS.to_string(), IMAGE.to_string(), effect];
    finish(dir, head, &extra, &rows);
}

/// The positions of the prompt below as `[3, 6]`: two text rows, a 2x1 image at position 2,
/// two text rows after the image's `max(nx, ny)` = 2 positions.
const POS: [i32; 18] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5];

/// A prompt set of 6 ids: text, text, a span of 2 image rows, text, text.
fn prompt(dir: &Path, head: &Head, with_image: bool, pad_id: u32) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "result_norm", [N_EMBD as u64, 6, 1, 1], 1.5);
    let f32s: Vec<u8> = POS.iter().flat_map(|&p| (p as f32).to_le_bytes()).collect();
    let i32s: Vec<u8> = POS.iter().flat_map(|p| p.to_le_bytes()).collect();
    let stem = |e| dump_file_name("mrope_pos", 0, RowKind::Tensor, Layout::Flat, e);
    write(&dir.join(stem(FileElem::F32)), &f32s);
    write(&dir.join(stem(FileElem::I32)), &i32s);
    let sum: i32 = POS.iter().sum();
    rows.push(format!(
        "tensor\tmrope_pos\t0\ti32\t3\t6\t1\t1\t72\t{sum}.000000\tINPUT\t1\t0\t-\t-"
    ));
    rows.push(format!(
        "int\tmrope_pos\t0\ttensor\ti32\ti32\tflat\t18\t72\t{sum}\t5\t{}",
        stem(FileElem::I32)
    ));
    let mut extra = vec![
        "# tokens_count\t6".to_string(),
        format!("# image_pad_id\t{pad_id}"),
        "# span columns\tindex image at len nx ny n_pos start_pos end_pos".to_string(),
        "# span\t0\tg\t2\t2\t2\t1\t2\t2\t4".to_string(),
    ];
    if with_image {
        extra.push(IMAGE_COLUMNS.to_string());
        extra.push(IMAGE_2X1.to_string());
    }
    finish(dir, head, &extra, &rows);
}

fn refusal(r: Result<ClefvisSet, RefError>) -> String {
    match r {
        Ok(_) => panic!("the set was taken"),
        Err(e) => e.to_string(),
    }
}

/// A set of each kind is taken through its kind's family and read by name.
#[test]
fn each_kind_of_set_reads_by_name() {
    let dir = scratch("kinds");
    preproc(&dir, &Head::tower("preproc"), IMAGE);
    let set = ClefvisSet::open(&dir, family(Kind::Preproc)).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((set.cpu, set.card.as_str()), (false, "NVIDIA RTX A6000"));
    assert_eq!(set.image("g").unwrap_or_else(|e| panic!("{e}")).best_w, 128);
    assert_eq!(
        set.inp_raw("g").unwrap_or_else(|e| panic!("{e}")).len(),
        128 * 96 * 3
    );

    taps(&dir, &Head::tower("taps"), true);
    let set = ClefvisSet::open(&dir, family(Kind::Taps)).unwrap_or_else(|e| panic!("{e}"));
    let (row, v) = set
        .tap("g", "layer_out-26")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((row.ne[0], v.len()), (1152, 1152 * 48));
    assert!(
        set.tap_effect_of("g")
            .unwrap_or_else(|e| panic!("{e}"))
            .full
    );

    taps(&dir, &Head::tower("taps"), false);
    let set = ClefvisSet::open(&dir, family(Kind::Taps)).unwrap_or_else(|e| panic!("{e}"));
    assert!(set.tap("g", EMBD).is_ok());
    let e = set.tap("g", "ln1-0").expect_err("embd only").to_string();
    assert!(e.contains("carries embd only"), "{e}");

    prompt(&dir, &Head::prompt("hidden"), true, IMAGE_PAD_ID);
    let set = ClefvisSet::open(&dir, family(Kind::Hidden)).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.n_ids().unwrap_or_else(|e| panic!("{e}")), 6);
    assert_eq!(
        set.result_norm().unwrap_or_else(|e| panic!("{e}")).len(),
        N_EMBD * 6
    );
    let pos = set.positions().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(pos[2..5], [[2, 2, 2], [2, 2, 3], [4, 4, 4]]);
    assert!(set.inp_raw("g").is_err(), "a prompt set holds no inp_raw");

    prompt(&dir, &Head::prompt("prose"), false, IMAGE_PAD_ID);
    let set = ClefvisSet::open(&dir, family(Kind::Prose)).unwrap_or_else(|e| panic!("{e}"));
    assert!(set.images.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Each refusal by name: another model file (`Stale`), another mainline build
/// and another architecture (`Foreign`), no trailer (`Unfinished`), another
/// mmproj (`Stale`).
#[test]
fn a_set_of_another_file_build_arch_or_mmproj_is_refused_by_name() {
    let dir = scratch("identity");
    let base = || Head::tower("taps");
    let cases: Vec<(&str, Head, &str)> = vec![
        (
            "model",
            Head {
                model: "/models/other.gguf",
                ..base()
            },
            "dumped from",
        ),
        (
            "build",
            Head {
                build: "db517b69",
                ..base()
            },
            "build",
        ),
        (
            "arch",
            Head {
                arch: ARCH,
                ..base()
            },
            "arch",
        ),
        (
            "trailer",
            Head {
                complete: false,
                ..base()
            },
            "no `# complete` trailer",
        ),
        (
            "mmproj",
            Head {
                mmproj_sha: "00ff",
                ..base()
            },
            "mmproj",
        ),
    ];
    // every case is tried before the assertion, so a reader that lets one through names it with the rest
    let mut wrong = Vec::new();
    for (what, head, want) in cases {
        taps(&dir, &head, true);
        match ClefvisSet::open(&dir, family(Kind::Taps)) {
            Ok(_) => wrong.push(format!("{what}: the set was taken")),
            Err(e) if !e.to_string().contains(want) => {
                wrong.push(format!("{what}: refused as {e}, want {want:?}"));
            }
            Err(_) => {}
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    taps(&dir, &base(), true);
    assert!(ClefvisSet::open(&dir, family(Kind::Taps)).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The family's own `check_set` (what `refset-check` and `hw_every_set_in_place_is_its_familys`
/// call) reads the set as its kind: a set of another mmproj or another kind is refused by name there,
/// and a good set states its file and build.
#[test]
fn the_familys_check_set_refuses_a_set_of_another_mmproj_or_kind() {
    let dir = scratch("checkset");
    taps(&dir, &Head::tower("taps"), true);
    let p = TAPS.check_set(&dir).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (p.dumped_from.as_str(), p.build.as_deref()),
        (MMPROJ_BF16, Some(LCPP_BUILD))
    );
    taps(
        &dir,
        &Head {
            mmproj_sha: "00ff",
            ..Head::tower("taps")
        },
        true,
    );
    let e = TAPS
        .check_set(&dir)
        .expect_err("another mmproj")
        .to_string();
    assert!(e.contains("mmproj") && e.contains("00ff"), "{e}");
    taps(&dir, &Head::tower("taps"), true);
    let e = PREPROC
        .check_set(&dir)
        .expect_err("another kind")
        .to_string();
    assert!(e.contains("clefvis") && e.contains("taps"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A set is read as its own kind's only: a taps set read as preproc is
/// `Foreign` on `clefvis`, a prompt set read as a tower kind names the
/// other model file, and a hidden_ref set of the same file (no `# clefvis`
/// line) is no clefvis set.
#[test]
fn a_set_of_another_kind_or_family_is_refused_by_name() {
    let dir = scratch("kind");
    taps(&dir, &Head::tower("taps"), true);
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Preproc)));
    assert!(
        e.contains("clefvis") && e.contains("\"taps\"") && e.contains("preproc"),
        "{e}"
    );
    prompt(&dir, &Head::prompt("hidden"), true, IMAGE_PAD_ID);
    for kind in [Kind::Preproc, Kind::Taps] {
        let e = refusal(ClefvisSet::open(&dir, family(kind)));
        assert!(
            e.contains("dumped from") && e.contains(MODEL_FLASH_Q8),
            "{kind:?}: {e}"
        );
    }
    for kind in [Kind::Prose, Kind::Bf16Rows] {
        let e = refusal(ClefvisSet::open(&dir, family(kind)));
        assert!(
            e.contains("clefvis") && e.contains("hidden"),
            "{kind:?}: {e}"
        );
    }
    // hidden_ref's set of the same file, build and architecture: the family's check takes
    // it, the kind line is what refuses it
    let mut rows = Vec::new();
    put(&dir, &mut rows, "result_norm", [4, 1, 1, 1], 1.0);
    let lines: Vec<String> = Head::prompt("hidden")
        .lines()
        .into_iter()
        .filter(|l| !l.starts_with("# clefvis") && !l.starts_with("# mmproj"))
        .collect();
    write(
        &dir.join("MANIFEST.tsv"),
        format!(
            "{}\n# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n{}\n# complete\t1\t0\n",
            lines.join("\n"),
            rows.join("\n")
        )
        .as_bytes(),
    );
    assert!(HIDDEN_FLASH_Q8.check_set(&dir).is_ok());
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Hidden)));
    assert!(e.contains("no `# clefvis` line"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The new families sit after the old ones (the architecture's node-dump family stays the
/// first), the table holds exactly one family of each kind, and each holds only the sets of its kind.
#[test]
fn the_table_holds_one_clefvis_family_of_each_kind_after_the_node_dump_family() {
    let first = crate::arch::node_dumps(ARCH).map(|f| f.name);
    assert_eq!(first, Some("hidden-qwen35"));
    for kind in Kind::ALL {
        let of_kind = crate::arch::all()
            .filter(|f| f.identity == Identity::Clefvis(kind))
            .count();
        assert_eq!(of_kind, 1, "{kind:?}: {of_kind} families in the table");
        let stem = format!("ref_clefvis_{}", kind.as_str());
        let f = family(kind);
        assert!(
            f.sets.iter().all(|s| s.starts_with(&stem)),
            "{kind:?}: the family {} holds {:?}, want sets of {stem}",
            f.name,
            f.sets
        );
    }
}

/// A family of another identity is no clefvis family: the reader refuses it by name.
#[test]
fn a_family_of_another_identity_is_refused_by_the_reader() {
    let dir = scratch("notclefvis");
    taps(&dir, &Head::tower("taps"), true);
    let e = refusal(ClefvisSet::open(&dir, &HIDDEN_FLASH_Q8));
    assert!(e.contains("is not a clefvis family"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Rows that disagree with the header are refused naming what differs: a
/// row of another shape than the plan, a grid that is not its token count, an
/// image line of another width, a span that does not hold the image's tokens,
/// and an image-pad id of another vocabulary.
#[test]
fn a_set_whose_rows_disagree_with_its_header_is_malformed() {
    let dir = scratch("shape");
    // the input row of another size than the plan
    let mut rows = Vec::new();
    put(
        &dir,
        &mut rows,
        &format!("g/{INP_RAW}"),
        [128, 64, 3, 1],
        0.5,
    );
    finish(
        &dir,
        &Head::tower("preproc"),
        &[IMAGE_COLUMNS.to_string(), IMAGE.to_string()],
        &rows,
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Preproc)));
    assert!(
        e.contains("g/inp_raw is [128, 64, 3, 1], want [128, 96, 3, 1]"),
        "{e}"
    );
    // a 4x3 grid of 13 tokens
    preproc(
        &dir,
        &Head::tower("preproc"),
        &IMAGE.replace("\t12\t4", "\t13\t4"),
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Preproc)));
    assert!(e.contains("grid 4x3 of 13 tokens"), "{e}");
    // an image line a field short
    preproc(
        &dir,
        &Head::tower("preproc"),
        &IMAGE.replace("\t4\t3\t12\t4", "\t4\t3\t12"),
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Preproc)));
    assert!(e.contains("11 fields"), "{e}");
    // a span that holds 3 ids for a 2-token image
    prompt(&dir, &Head::prompt("hidden"), true, IMAGE_PAD_ID);
    let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
    write(
        &dir.join("MANIFEST.tsv"),
        text.replace("# span\t0\tg\t2\t2\t", "# span\t0\tg\t2\t3\t")
            .as_bytes(),
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Hidden)));
    assert!(e.contains("span 0 holds 3 ids"), "{e}");
    // an image-pad id of another vocabulary
    prompt(&dir, &Head::prompt("hidden"), true, 151_655);
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Hidden)));
    assert!(e.contains("image_pad_id") && e.contains("248056"), "{e}");
    // a prose set that carries image lines
    prompt(&dir, &Head::prompt("prose"), true, IMAGE_PAD_ID);
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Prose)));
    assert!(e.contains("prose set with 1 image lines"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
