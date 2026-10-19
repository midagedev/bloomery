//! The Bolt image-input families: a set of each of the two kinds the CPU dump writes, read by name, and each way a set is
//! refused — its identity (the projector's digest, the build), its layout against the family's, and rows whose shape
//! disagrees with the grid the set's own lines state. The reader's layout switches are the set's own lines, so the Qwen and
//! GLM seats' sets (`arch::qwen_vis_tests`, `arch::glm5next`) stay what they were.

use super::vis::{self, IMAGE_TOKEN_ID, LCPP_BUILD, MMPROJ, MMPROJ_SHA256};
use crate::RefError;
use crate::clefvis::testkit::{Head, IMAGE_COLUMNS, finish, put, refusal, scratch, write};
use crate::clefvis::{
    ClefvisSet, EMBD, Kind, PositionKind, ProjectorKind, RADIO_POS, RADIO_PREFIX_ROWS,
    RADIO_TAP_BLOCKS, RADIO_TAP_NODES, RADIO_WIDTH, TOWER_ARCH, radio_tap_names,
};
use crate::family::{Family, Identity};
use std::path::Path;

const PROJECTOR: &str = "# projector\tnemotron_omni\tpatch_size 16\tspatial_merge_size 2\talign 32\timage_min_pixels 262144\timage_max_pixels 3407872";

/// A 50x50 image planned 64x64: a 2x2 grid of 4 tokens of 32 pixels a side, 16 patches, advancing the decoder by 4.
const IMAGE: &str = "# image\tg\taa\tbb\t50\t50\t64\t64\t2\t2\t4\t4";

/// Patches of the image and the rows of a block node: the class and register rows, then the patches.
const PATCHES: u64 = 16;
const ROWS: u64 = PATCHES + RADIO_PREFIX_ROWS;

fn family(kind: Kind) -> &'static Family {
    crate::arch::all()
        .find(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
        .unwrap_or_else(|| panic!("{kind:?}: no family of the nemotron_omni seat in the table"))
}

fn tower(kind: Kind) -> Head<'static> {
    Head {
        kind: kind.as_str(),
        model: MMPROJ,
        build: LCPP_BUILD,
        arch: TOWER_ARCH,
        mmproj: MMPROJ,
        mmproj_sha: MMPROJ_SHA256,
        complete: true,
    }
}

fn lines(extra: &[&str]) -> Vec<String> {
    let mut v = vec![PROJECTOR.to_string()];
    v.extend(extra.iter().map(|l| (*l).to_string()));
    v
}

/// Set A: the 64x64 canvas of the one image.
fn preproc(dir: &Path) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "g/inp_raw", [64, 64, 3, 1], 0.5);
    finish(
        dir,
        &tower(Kind::Preproc),
        &lines(&[IMAGE_COLUMNS, IMAGE]),
        &rows,
    );
}

/// How a set B's rows are shaped: the rows of `inp_pos` and of a block node, and the scope its `# tap_effect` states.
struct Taps {
    scope: &'static str,
    pos_rows: u64,
    block_rows: u64,
}

const FULL: Taps = Taps {
    scope: "full",
    pos_rows: PATCHES,
    block_rows: ROWS,
};

fn taps(dir: &Path, t: &Taps) {
    let mut rows = Vec::new();
    if t.scope == "full" {
        put(
            dir,
            &mut rows,
            &format!("g/{RADIO_POS}"),
            [RADIO_WIDTH, t.pos_rows, 1, 1],
            1.0,
        );
        for b in RADIO_TAP_BLOCKS {
            for node in RADIO_TAP_NODES {
                // `Qcur` is three-dimensional: 16 heads of 80
                let ne = if node == "Qcur" {
                    [80, 16, t.block_rows, 1]
                } else {
                    [RADIO_WIDTH, t.block_rows, 1, 1]
                };
                put(dir, &mut rows, &format!("g/{node}-{b}"), ne, 1.0);
            }
        }
    }
    put(dir, &mut rows, &format!("g/{EMBD}"), [4096, 4, 1, 1], 2.0);
    let effect = format!(
        "# tap_effect\tg\tembd values 16384\tdiffering 0\tmax_abs_diff 0\ttaps {}",
        t.scope
    );
    finish(
        dir,
        &tower(Kind::Taps),
        &lines(&[IMAGE_COLUMNS, IMAGE, &effect]),
        &rows,
    );
}

fn ok(r: Result<ClefvisSet, RefError>) -> ClefvisSet {
    r.unwrap_or_else(|e| panic!("{e}"))
}

fn replace(dir: &Path, from: &str, to: &str) {
    let path = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let new = text.replace(from, to);
    assert_ne!(new, text, "the edit changed nothing");
    write(&path, new.as_bytes());
}

/// A set of each kind reads by name: the grid of 32-pixel tokens, the nodes of the layout (`inp_pos` of the patches, a block
/// node of the prefix rows and the patches, `Qcur` as heads), the embeddings, and the `<image>` token's id.
#[test]
fn a_nemotron_omni_set_of_each_kind_reads_by_name() {
    let dir = scratch("bolt-reads");
    preproc(&dir);
    let set = ok(ClefvisSet::open(&dir, family(Kind::Preproc)));
    assert_eq!(set.projector.kind, ProjectorKind::NemotronOmni);
    assert_eq!(set.projector.kind.positions(), PositionKind::Normal);
    assert_eq!((set.projector.align, set.projector.merge), (32, 2));
    assert_eq!(set.projector.image_max_pixels, Some(3_407_872));
    let image = set.image("g").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (image.best_w, image.nx, image.n_tokens, image.n_pos),
        (64, 2, 4, 4)
    );
    assert_eq!(
        set.inp_raw("g").unwrap_or_else(|e| panic!("{e}")).len(),
        64 * 64 * 3
    );

    taps(&dir, &FULL);
    let set = ok(ClefvisSet::open(&dir, family(Kind::Taps)));
    let want = radio_tap_names();
    assert_eq!(
        want.len(),
        1 + RADIO_TAP_BLOCKS.len() * RADIO_TAP_NODES.len()
    );
    for node in want.iter().map(String::as_str).chain([EMBD]) {
        let (row, v) = set.tap("g", node).unwrap_or_else(|e| panic!("{node}: {e}"));
        assert_eq!(v.len() as u64, row.ne.iter().product::<u64>(), "{node}");
    }
    let rows = |node: &str| set.tap("g", node).expect("tap").0.ne;
    assert_eq!(rows("inp_pos")[..2], [1280, 16]);
    assert_eq!(rows("layer_out-31")[..2], [1280, 26]);
    assert_eq!(rows("Qcur-15")[..3], [80, 16, 26]);
    assert_eq!(rows("embd")[..2], [4096, 4]);
    let e = set.tap("g", "ln1-0").expect_err("a Qwen node").to_string();
    assert!(e.contains("g/ln1-0"), "{e}");
    assert_eq!(IMAGE_TOKEN_ID, 18);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The sets of every kind the seat writes are in the table once, under the seat's own names, and a CPU-only seat's taps
/// family holds no `.cpu` twin.
#[test]
fn the_table_holds_the_two_kinds_the_nemotron_omni_seat_writes() {
    for (kind, sets) in [
        (Kind::Preproc, &["ref_nemotron_h_moe_vis_preproc"][..]),
        (Kind::Taps, &["ref_nemotron_h_moe_vis_taps"][..]),
    ] {
        let n = crate::arch::all()
            .filter(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
            .count();
        assert_eq!(n, 1, "{kind:?}: {n} families");
        let f = family(kind);
        assert_eq!(f.sets, sets, "{}", f.name);
        assert_eq!(f.name, format!("nemotron_h_moe-vis-{}", kind.as_str()));
        assert_eq!(f.arch, Some(TOWER_ARCH));
        assert_eq!(f.runs().unwrap_or_else(|e| panic!("{e}")), MMPROJ);
        assert!(super::FAMILIES.iter().any(|g| std::ptr::eq(*g, f)));
    }
}

/// An `inp_pos` of the block nodes' rows, or a block node of the patches' only, is refused by its name and what it should be.
#[test]
fn a_node_of_the_wrong_row_count_is_refused_by_name() {
    let dir = scratch("bolt-rows");
    taps(
        &dir,
        &Taps {
            block_rows: PATCHES,
            ..FULL
        },
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("g/layer_inp_normed-0 is [1280, 16, 1, 1], want 1280 values for each of the 26 rows (10 prefix rows and 16 patches)"),
        "{e}"
    );
    taps(
        &dir,
        &Taps {
            pos_rows: ROWS,
            ..FULL
        },
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("g/inp_pos is [1280, 26, 1, 1], want [1280, 16, 1, 1]"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tower of this layout has no output-end nodes: a set that states `taps final` is refused by name, and one with the
/// embeddings only reads.
#[test]
fn the_scopes_of_the_layout() {
    let dir = scratch("bolt-scope");
    taps(
        &dir,
        &Taps {
            scope: "inp_raw only",
            ..FULL
        },
    );
    let set = ok(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(set.tap("g", "inp_pos").is_err());
    assert!(set.tap("g", EMBD).is_ok());
    taps(&dir, &FULL);
    replace(&dir, "taps full", "taps final");
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("g: taps final, a nemotron_omni tower has no output-end nodes: it is tapped in full or its embeddings only"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A set of another layout, or of another file, build or digest, is not this seat's.
#[test]
fn a_set_of_another_identity_is_refused() {
    let dir = scratch("bolt-identity");
    taps(&dir, &FULL);
    replace(&dir, "# projector\tnemotron_omni", "# projector\tglm5v");
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("projector \"glm5v\", the nemotron_h_moe-vis-taps family wants nemotron_omni"),
        "{e}"
    );

    taps(&dir, &FULL);
    replace(&dir, MMPROJ_SHA256, &"0".repeat(64));
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains(&format!(
            "sha256 {}, the tree runs mmproj {MMPROJ} sha256 {MMPROJ_SHA256}",
            "0".repeat(64)
        )),
        "{e}"
    );

    taps(&dir, &FULL);
    replace(
        &dir,
        &format!("# build\t{LCPP_BUILD}"),
        "# build\t36a73916e",
    );
    let e = refusal(ClefvisSet::open(&dir, family(Kind::Taps)));
    assert!(
        e.contains(&format!(
            "build \"36a73916e\", the nemotron_h_moe-vis-taps family wants {LCPP_BUILD}"
        )),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
