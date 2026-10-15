use super::{MAINLINE, MAINLINE_BUILD, MODEL};
use crate::RefError;
use crate::mtptaps::{Elem, IDS_FILE, TapSet};
use std::path::{Path, PathBuf};

const IDS: [u32; 4] = [10, 11, 12, 13];

/// What the set's header and trailer state; `Default` is a finished set of the family's file, build and
/// architecture, with the MTP context's line.
struct Header<'a> {
    model: &'a str,
    build: &'a str,
    arch: &'a str,
    mtp: Option<&'a str>,
    complete: bool,
}

impl Default for Header<'_> {
    fn default() -> Self {
        Header {
            model: MODEL,
            build: MAINLINE_BUILD,
            arch: "qwen35moe",
            mtp: Some("1\t3"),
            complete: true,
        }
    }
}

fn dir(what: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bloomery-mtptaps-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    d
}

fn put(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A set of four prompt ids, three steps, a trunk of four rows of two, and two taps (`mtp_eh_proj`, f32 of
/// two; `ffn_moe_topk`, i32 of two) at rows 0 and 2. `extra` is appended before the trailer.
fn write_set(dir: &Path, h: &Header<'_>, extra: &[&str]) {
    put(dir, IDS_FILE, b"10\n11\n12\n13\n");
    put(dir, "trunk_h.f32", &f32s(&[0., 1., 2., 3., 4., 5., 6., 7.]));
    for row in [0, 2] {
        put(
            dir,
            &format!("mtp_eh_proj.r{row}.f32"),
            &f32s(&[row as f32, 0.5]),
        );
        let topk: Vec<u8> = [7i32, -1].into_iter().flat_map(i32::to_le_bytes).collect();
        put(dir, &format!("ffn_moe_topk.r{row}.i32"), &topk);
    }
    let mut lines = vec![
        "# q35mtp_ref — test".to_string(),
        format!("# model\t{}", h.model),
        format!("# build\t{}", h.build),
        format!("# arch\t{}", h.arch),
        "# tokens_count\t4".to_string(),
    ];
    if let Some(m) = h.mtp {
        lines.push(format!("# mtp\t{m}"));
    }
    lines.extend(
        [
            "# rows\t0\t2",
            "# trunk columns\tname\ttype\trows\twidth\tfile",
            "trunk\th_nextn\tf32\t4\t2\ttrunk_h.f32",
            "# step columns\trow\ttoken\tpos\tmtp_argmax\ttarget_next",
            "step\t0\t11\t0\t5\t5",
            "step\t1\t12\t1\t6\t7",
            "step\t2\t13\t2\t8\t8",
            "# tap columns\tname\trow\ttype\tn\tfile",
            "tap\tmtp_eh_proj\t0\tf32\t2\tmtp_eh_proj.r0.f32",
            "tap\tffn_moe_topk\t0\ti32\t2\tffn_moe_topk.r0.i32",
            "tap\tmtp_eh_proj\t2\tf32\t2\tmtp_eh_proj.r2.f32",
            "tap\tffn_moe_topk\t2\ti32\t2\tffn_moe_topk.r2.i32",
        ]
        .map(str::to_string),
    );
    lines.extend(extra.iter().map(|l| (*l).to_string()));
    if h.complete {
        lines.push("# complete\t4\t0".to_string());
    }
    put(dir, "MANIFEST.tsv", (lines.join("\n") + "\n").as_bytes());
}

fn open(d: &Path) -> Result<TapSet, RefError> {
    TapSet::open(d, &MAINLINE)
}

/// The refusal of the set at `d`, or a panic naming the case that opened.
fn refused(d: &Path, what: &str) -> RefError {
    match open(d) {
        Err(e) => e,
        Ok(_) => panic!("{what}: the set opened"),
    }
}

/// A finished set opens through the family's check, and a tap is read by its graph name and its row, as f32 or
/// i32 as the node is; a name or a row the set does not hold is refused naming both.
#[test]
fn a_finished_set_reads_a_tap_by_name_and_row() {
    let d = dir("ok");
    write_set(&d, &Header::default(), &[]);
    let p = MAINLINE.check_set(&d).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(p.dumped_from, MODEL);
    assert_eq!(p.build.as_deref(), Some(MAINLINE_BUILD));
    let set = open(&d).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(set.ids().unwrap_or_else(|e| panic!("{e}")), IDS);
    assert_eq!(
        set.tap_f32("mtp_eh_proj", 2)
            .unwrap_or_else(|e| panic!("{e}")),
        [2.0, 0.5]
    );
    assert_eq!(
        set.tap_i32("ffn_moe_topk", 0)
            .unwrap_or_else(|e| panic!("{e}")),
        [7, -1]
    );
    assert_eq!(
        set.tap("ffn_moe_topk", 0).map(|t| t.elem).ok(),
        Some(Elem::I32)
    );
    assert_eq!(
        set.trunk_rows().unwrap_or_else(|e| panic!("{e}"))[6..],
        [6.0, 7.0]
    );
    let s = set.step(2).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((s.token, s.pos, s.mtp_argmax, s.target_next), (13, 2, 8, 8));
    for (name, row) in [("mtp_nothing", 0), ("mtp_eh_proj", 1)] {
        let e = set.tap(name, row).err().map(|e| e.to_string());
        let e = e.unwrap_or_else(|| panic!("{name} at {row} was found"));
        assert!(e.contains(&format!("no tap {name} at row {row}")), "{e}");
    }
    let e = set.tap_f32("ffn_moe_topk", 0).err().map(|e| e.to_string());
    assert!(
        e.is_some_and(|e| e.contains("tap ffn_moe_topk is i32, read as f32")),
        "an i32 tap read as f32"
    );
    std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
}

/// The family refuses by name a set of another file, another mainline build, another architecture, one dumped
/// without the MTP context (no `# mtp` line, or a model with no next-token layer) and one without its trailer.
#[test]
fn the_family_refuses_a_set_of_another_file_build_or_arch_or_without_the_mtp_context() {
    let d = dir("refuse");
    let other_build = "0000000aa";
    let cases: [(&str, Header<'_>); 6] = [
        (
            "file",
            Header {
                model: "/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf",
                ..Header::default()
            },
        ),
        (
            "build",
            Header {
                build: other_build,
                ..Header::default()
            },
        ),
        (
            "arch",
            Header {
                arch: "qwen35",
                ..Header::default()
            },
        ),
        (
            "mtp",
            Header {
                mtp: None,
                ..Header::default()
            },
        ),
        (
            "layers",
            Header {
                mtp: Some("0\t3"),
                ..Header::default()
            },
        ),
        (
            "trailer",
            Header {
                complete: false,
                ..Header::default()
            },
        ),
    ];
    for (what, h) in &cases {
        write_set(&d, h, &[]);
        let e = refused(&d, what);
        match (*what, &e) {
            (
                "file",
                RefError::Stale {
                    dumped_from, runs, ..
                },
            ) => {
                assert_eq!((dumped_from.as_str(), runs.as_str()), (h.model, MODEL));
            }
            (
                "build",
                RefError::Foreign {
                    field: "build",
                    got,
                    want,
                    ..
                },
            ) => {
                assert_eq!((got.as_str(), want.as_str()), (other_build, MAINLINE_BUILD));
            }
            (
                "arch",
                RefError::Foreign {
                    field: "arch", got, ..
                },
            ) => assert_eq!(got, "qwen35"),
            (
                "mtp",
                RefError::Foreign {
                    field: "mtp", got, ..
                },
            ) => {
                assert_eq!(got, "(no # mtp line)");
                assert!(e.to_string().contains("MTP context"), "{e}");
            }
            ("layers", RefError::Malformed { what, .. }) => {
                assert!(what.contains("# mtp 0 3"), "{what}");
            }
            ("trailer", RefError::Unfinished { set }) => assert_eq!(set, &d.display().to_string()),
            _ => panic!("{what}: {e:?}"),
        }
    }
    std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
}

/// A set whose files or rows disagree is `Malformed`, naming the row or file: a tap file of another length, a
/// step that is not the prompt's next id, a tap named twice, a row naming a file outside the directory, a
/// tapped row that is no step, and a tap a row lacks.
#[test]
fn a_set_whose_files_or_rows_disagree_is_malformed() {
    let d = dir("malformed");
    let at = |what: &str, extra: &[&str], edit: &dyn Fn(&Path), want: &str| {
        write_set(&d, &Header::default(), extra);
        edit(&d);
        let e = refused(&d, what);
        assert!(
            matches!(e, RefError::Malformed { .. }) && e.to_string().contains(want),
            "{what}: {e}"
        );
    };
    at(
        "a short tap file",
        &[],
        &|d| put(d, "mtp_eh_proj.r2.f32", &f32s(&[1.0])),
        "mtp_eh_proj.r2.f32 is 4 bytes",
    );
    at(
        "the trunk file",
        &[],
        &|d| put(d, "trunk_h.f32", &f32s(&[1.0; 7])),
        "trunk_h.f32 is 28 bytes",
    );
    at(
        "a prompt of another length",
        &[],
        &|d| put(d, IDS_FILE, b"10\n11\n12\n"),
        "holds 3 ids",
    );
    at(
        "a tap named twice",
        &["tap\tmtp_eh_proj\t2\tf32\t2\tmtp_eh_proj.r2.f32"],
        &|_| {},
        "tap mtp_eh_proj at row 2 is named twice",
    );
    at(
        "a file outside the directory",
        &["tap\tx\t0\tf32\t2\t../trunk_h.f32"],
        &|_| {},
        "is not a name under the set's directory",
    );
    at(
        "a prompt whose id is not the step's token",
        &[],
        &|d| put(d, IDS_FILE, b"10\n11\n99\n13\n"),
        "step row 1 is not step 1",
    );
    at(
        "a tap missing at a tapped row",
        &["tap\tmtp_hnorm\t0\tf32\t2\tmtp_eh_proj.r0.f32"],
        &|_| {},
        "no tap mtp_hnorm at row 2",
    );
    std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
}
