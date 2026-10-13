//! The MiniCPM-V 4.7 tower reader: a scratch set of each kind with the canvas layout of two images, the identity every
//! other seat's set fails by name (the projector file, its digest, the oracle build, the `# clefvis` kind), the canvas
//! rules a tile line must keep, and the rows each kind carries.

use super::vis::{
    BLOCK_NODES, EMBD, GRAPH_NODES, INP_RAW, Kind, MC47_LCPP_BUILD, MMPROJ_F16, MMPROJ_F16_SHA256,
    McvisSet, N_EMBD, PREPROC, TAP_BLOCKS, TAPS, Tile, row_name, tap_names, tap_shape,
};
use crate::RefError;
use crate::clefvis::testkit::{Head, finish, put, scratch};
use crate::clefvis::{ClefvisSet, Profile, QVIS_LCPP_BUILD};
use crate::family::{Family, Identity};
use std::path::Path;

/// The columns of the two image lines and the tile lines every set carries.
const IMAGE_COLUMNS: &str =
    "# image columns\tname png_sha256 rgb8_sha256 w h grid_x grid_y n_tiles n_tokens n_pos";
const TILE_COLUMNS: &str =
    "# tile columns\timage k kind row col w h nx ny merger_rows suffix row0 rows";
const PROJECTOR: &str = "# projector\tminicpmv4_7\tpatch_size 14\timage_size 448\tn_merge 4\tprojection_dim 2048\tblocks 27\tinsert_layer_id 6";
const POSITIONS: &str = "# positions\tcanvas\tqwen35moe.rope.section_order zyxt";

/// Image `one`: an overview alone, a 56 x 56 tile of one merger row and the `</image>` row.
/// Image `grid`: an overview and a 2 x 2 grid of slices, each 56 x 56: the separator rows are written out here by hand
/// from mtmd's `clip_suffix_rows` (`tools/mtmd/clip-impl.h`) — overview {0,1}, slices {2,1}, {2,3,1}, {2,1}, {2} — so
/// the reader's rule is not checked against itself.
const IMAGES: [&str; 2] = [
    "# image\tone\taa\tbb\t56\t56\t0\t0\t1\t2\t2",
    "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t5\t15\t15",
];
const TILES: [&str; 6] = [
    "# tile\tone\t0\toverview\t-\t-\t56\t56\t1\t1\t1\t0\t0\t2",
    "# tile\tgrid\t0\toverview\t-\t-\t56\t56\t1\t1\t1\t0,1\t0\t3",
    "# tile\tgrid\t1\tslice\t0\t0\t56\t56\t1\t1\t1\t2,1\t3\t3",
    "# tile\tgrid\t2\tslice\t0\t1\t56\t56\t1\t1\t1\t2,3,1\t6\t4",
    "# tile\tgrid\t3\tslice\t1\t0\t56\t56\t1\t1\t1\t2,1\t10\t3",
    "# tile\tgrid\t4\tslice\t1\t1\t56\t56\t1\t1\t1\t2\t13\t2",
];
/// The rows of each tile (merger rows and separator rows), image by image: `one` 2; `grid` 3, 3, 4, 3, 2.
const ROWS: [(&str, [usize; 5], usize); 2] =
    [("one", [2, 0, 0, 0, 0], 1), ("grid", [3, 3, 4, 3, 2], 5)];

fn head(kind: Kind) -> Head<'static> {
    Head {
        kind: kind.as_str(),
        model: MMPROJ_F16,
        build: MC47_LCPP_BUILD,
        arch: "clip",
        mmproj: MMPROJ_F16,
        mmproj_sha: MMPROJ_F16_SHA256,
        complete: true,
    }
}

/// The header lines beyond the common ones, and the rows (with their files in `dir`) of a set of `kind`.
fn parts(dir: &Path, kind: Kind) -> (Vec<String>, Vec<String>) {
    let mut extra = vec![
        PROJECTOR.to_string(),
        POSITIONS.to_string(),
        IMAGE_COLUMNS.to_string(),
    ];
    extra.extend(IMAGES.iter().map(|l| (*l).to_string()));
    extra.push(TILE_COLUMNS.to_string());
    extra.extend(TILES.iter().map(|l| (*l).to_string()));
    let mut rows = Vec::new();
    for (name, rows_of, tiles) in ROWS {
        for (k, &rows_k) in rows_of.iter().enumerate().take(tiles) {
            match kind {
                Kind::Preproc => put(
                    dir,
                    &mut rows,
                    &row_name(name, k, INP_RAW),
                    [56, 56, 3, 1],
                    0.5,
                ),
                Kind::Taps => {
                    // a 56 x 56 tile: 4 x 4 = 16 patches, 4 after the ViT merger, 1 merger row
                    let tile = Tile {
                        index: k,
                        cell: None,
                        w: 56,
                        h: 56,
                        nx: 1,
                        ny: 1,
                        merger_rows: 1,
                        suffix: Vec::new(),
                        row0: 0,
                        rows: rows_k,
                    };
                    for n in tap_names() {
                        let ne = tap_shape(&n, &tile).unwrap_or_else(|| panic!("no shape for {n}"));
                        put(dir, &mut rows, &row_name(name, k, &n), ne, 1.0);
                    }
                    put(
                        dir,
                        &mut rows,
                        &row_name(name, k, EMBD),
                        [N_EMBD as u64, rows_k as u64, 1, 1],
                        2.0,
                    );
                }
            }
        }
    }
    if kind == Kind::Taps {
        extra.push(format!(
            "# tap_effect\tone\tembd values {}\tdiffering 0\tmax_abs_diff 0\ttaps full",
            2 * N_EMBD
        ));
        extra.push(format!(
            "# tap_effect\tgrid\tembd values {}\tdiffering 0\tmax_abs_diff 0\ttaps full",
            15 * N_EMBD
        ));
    }
    (extra, rows)
}

fn write_set(
    tag: &str,
    kind: Kind,
    h: &Head,
    edit: impl Fn(&mut Vec<String>),
) -> std::path::PathBuf {
    let dir = scratch(tag);
    let (mut extra, rows) = parts(&dir, kind);
    edit(&mut extra);
    finish(&dir, h, &extra, &rows);
    dir
}

/// Qwen3.6's tower-taps family and its seat, from the table: the projector file and digest another seat's sets name.
fn qwen36() -> (&'static Family, &'static Profile) {
    let family = crate::arch::named("qwen35moe-vis-taps").expect("the Qwen3.6 taps family");
    let Identity::Clefvis(_, profile) = family.identity else {
        panic!("{} is not a clefvis family", family.name);
    };
    (family, profile)
}

fn refusal(r: Result<McvisSet, RefError>) -> String {
    match r {
        Ok(_) => panic!("the set was taken"),
        Err(e) => e.to_string(),
    }
}

fn replace(extra: &mut [String], from: &str, to: &str) {
    let at = extra
        .iter()
        .position(|l| l.starts_with(from))
        .unwrap_or_else(|| panic!("no line {from}"));
    extra[at] = to.to_string();
}

#[test]
fn the_families_are_the_tower_kinds_and_the_prompt_kinds_of_the_seat() {
    let names: Vec<&str> = crate::arch::families(super::ARCH)
        .iter()
        .map(|f| f.name)
        .collect();
    assert_eq!(
        names,
        [
            "minicpmv47-vis-preproc",
            "minicpmv47-vis-taps",
            "minicpmv47-vis-hidden",
            "minicpmv47-vis-chatids"
        ]
    );
    for (kind, family) in [(Kind::Preproc, &PREPROC), (Kind::Taps, &TAPS)] {
        assert!(std::ptr::eq(kind.family(), family));
        assert_eq!(family.arch, Some("clip"));
        assert_eq!(family.runs().as_deref().ok(), Some(MMPROJ_F16));
        assert_eq!(
            family.build.map(|b| b.want()).as_deref(),
            Some(MC47_LCPP_BUILD)
        );
    }
    assert_eq!(
        TAPS.sets,
        ["ref_minicpmv47_vis_taps", "ref_minicpmv47_vis_taps.cpu"]
    );
}

#[test]
fn tap_names_are_the_graphs_nodes_of_four_blocks() {
    let names = tap_names();
    assert_eq!(
        names.len(),
        GRAPH_NODES.len() + TAP_BLOCKS.len() * BLOCK_NODES.len()
    );
    assert_eq!(names.len(), 39);
    assert_eq!(names.first().map(String::as_str), Some("pos_embed"));
    assert_eq!(names.last().map(String::as_str), Some("layer_out-26"));
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len());
}

#[test]
fn a_tile_of_560_by_392_has_the_shapes_the_dump_measured() {
    // the 1000 x 700 image's tiles on the box: 40 x 28 = 1,120 patches, 280 after the ViT merger, 70 merger rows
    let t = Tile {
        index: 1,
        cell: Some((0, 0)),
        w: 560,
        h: 392,
        nx: 10,
        ny: 7,
        merger_rows: 70,
        suffix: vec![2, 1],
        row0: 72,
        rows: 72,
    };
    let shape = |n: &str| tap_shape(n, &t);
    assert_eq!(shape("pos_embed"), Some([1152, 1120, 1, 1]));
    assert_eq!(shape("vit_merger_Qcur"), Some([72, 16, 1120, 1]));
    assert_eq!(shape("vit_merger_ds_normed"), Some([4608, 280, 1, 1]));
    assert_eq!(shape("post_ln"), Some([1152, 280, 1, 1]));
    assert_eq!(shape("merger_normed"), Some([4608, 70, 1, 1]));
    assert_eq!(shape("merger_out"), Some([2048, 70, 1, 1]));
    assert_eq!(shape("Qcur-6"), Some([72, 16, 1120, 1]));
    assert_eq!(shape("layer_out-6"), Some([1152, 1120, 1, 1]));
    assert_eq!(shape("Qcur-7"), Some([72, 16, 280, 1]));
    assert_eq!(shape("ffn_out-26"), Some([1152, 280, 1, 1]));
    for not_a_tap in [
        "layer_out-5",
        "layer_out",
        "Kcur-0",
        "embd",
        "inp_raw",
        "x-7",
    ] {
        assert_eq!(shape(not_a_tap), None, "{not_a_tap}");
    }
}

#[test]
fn a_set_of_each_kind_is_taken_and_read() {
    let dir = write_set("mcv-pre", Kind::Preproc, &head(Kind::Preproc), |_| {});
    let pre = McvisSet::open(&dir, Kind::Preproc).expect("a preproc set");
    assert_eq!(pre.kind, Kind::Preproc);
    assert!(!pre.cpu);
    assert_eq!(pre.card, "NVIDIA RTX A6000");
    assert_eq!(pre.images.len(), 2);
    assert_eq!(pre.inp_raw("grid", 4).expect("inp_raw").len(), 56 * 56 * 3);
    assert!(
        pre.tap("grid", 0, EMBD)
            .is_err_and(|e| e.to_string().contains("reads a canvas-taps set"))
    );

    let dir = write_set("mcv-taps", Kind::Taps, &head(Kind::Taps), |_| {});
    let set = McvisSet::open(&dir, Kind::Taps).expect("a taps set");
    let grid = set.image("grid").expect("grid");
    assert_eq!(
        (grid.grid_x, grid.grid_y, grid.n_tokens, grid.n_pos),
        (2, 2, 15, 15)
    );
    assert_eq!(grid.tiles.len(), 5);
    assert_eq!(grid.tiles[0].cell, None);
    assert_eq!(grid.tiles[3].cell, Some((1, 0)));
    assert_eq!(grid.tiles[2].suffix, [2, 3, 1]);
    assert_eq!((grid.tiles[4].row0, grid.tiles[4].rows), (13, 2));
    assert_eq!(set.tile("one", 0).expect("tile").rows, 2);
    assert!(set.tile("one", 1).is_err());
    assert!(set.image("nope").is_err());
    let (row, v) = set.tap("grid", 3, "layer_out-26").expect("tap");
    assert_eq!(row.ne, [1152, 4, 1, 1]);
    assert_eq!(v.len(), 1152 * 4);
    let (row, _) = set.tap("grid", 2, EMBD).expect("embd");
    assert_eq!(row.ne, [2048, 4, 1, 1]);
    assert!(
        set.tap("grid", 2, "layer_out-5")
            .is_err_and(|e| e.to_string().contains("no tapped node"))
    );
    assert!(
        set.inp_raw("grid", 0)
            .is_err_and(|e| e.to_string().contains("reads a canvas-preproc set"))
    );
    assert_eq!(set.tap_effect.len(), 2);
    assert_eq!(set.tap_effect[1].values, 15 * N_EMBD);
}

/// A set whose model, projector and digest are Qwen3.6's is refused twice over: by the family's model-file check
/// (`RefManifest::open`) and by the reader's `# mmproj` compare. FAIL-first: with the family's `runs`
/// returning Qwen3.6's file and the reader's path and digest compare off, this test reads "the set was taken"; either
/// layer alone leaves it green and reds the tests of the sets that name MiniCPM's file (`runs`) or its digest (the next).
#[test]
fn a_set_dumped_with_qwen36s_projector_is_refused() {
    let (_, q) = qwen36();
    let mut h = head(Kind::Taps);
    h.model = q.mmproj;
    h.mmproj = q.mmproj;
    h.mmproj_sha = q.mmproj_sha256;
    let dir = write_set("mcv-q36", Kind::Taps, &h, |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(e.contains("stale reference"), "{e}");
    assert!(e.contains(q.mmproj) && e.contains(MMPROJ_F16), "{e}");
}

#[test]
fn qwen36s_digest_under_the_minicpm_path_is_refused() {
    let (_, q) = qwen36();
    let mut h = head(Kind::Taps);
    h.mmproj_sha = q.mmproj_sha256;
    let dir = write_set("mcv-q36sha", Kind::Taps, &h, |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("stale reference")
            && e.contains(q.mmproj_sha256)
            && e.contains(MMPROJ_F16_SHA256),
        "{e}"
    );

    // the same file at another path
    let mut h = head(Kind::Taps);
    h.mmproj = q.mmproj;
    let dir = write_set("mcv-q36path", Kind::Taps, &h, |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(e.contains("stale reference") && e.contains(q.mmproj), "{e}");
}

#[test]
fn another_oracle_build_is_refused() {
    let mut h = head(Kind::Taps);
    h.build = QVIS_LCPP_BUILD;
    let dir = write_set("mcv-build", Kind::Taps, &h, |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("build") && e.contains(QVIS_LCPP_BUILD) && e.contains(MC47_LCPP_BUILD),
        "{e}"
    );
}

#[test]
fn a_qwen_seat_set_and_a_set_of_the_other_kind_are_refused_by_their_clefvis_line() {
    // Qwen's kinds are `preproc` and `taps`
    let mut h = head(Kind::Taps);
    h.kind = "taps";
    let dir = write_set("mcv-qkind", Kind::Taps, &h, |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("clefvis") && e.contains("\"taps\"") && e.contains("canvas-taps"),
        "{e}"
    );

    let dir = write_set("mcv-kind", Kind::Preproc, &head(Kind::Preproc), |_| {});
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("clefvis") && e.contains("canvas-preproc") && e.contains("canvas-taps"),
        "{e}"
    );
}

#[test]
fn the_qwen_readers_refuse_a_minicpm_set() {
    let dir = write_set("mcv-rev", Kind::Taps, &head(Kind::Taps), |_| {});
    let e = ClefvisSet::open(&dir, qwen36().0)
        .expect_err("a Qwen reader took the set")
        .to_string();
    assert!(
        e.contains("stale reference") && e.contains(MMPROJ_F16),
        "{e}"
    );
}

#[test]
fn a_set_without_its_trailer_is_unfinished() {
    let mut h = head(Kind::Taps);
    h.complete = false;
    let dir = write_set("mcv-unfinished", Kind::Taps, &h, |_| {});
    assert!(matches!(
        McvisSet::open(&dir, Kind::Taps),
        Err(RefError::Unfinished { .. })
    ));
}

#[test]
fn another_projector_layout_or_rope_order_is_refused() {
    let dir = write_set("mcv-proj", Kind::Taps, &head(Kind::Taps), |x| {
        replace(
            x,
            "# projector",
            &PROJECTOR.replace("insert_layer_id 6", "insert_layer_id 5"),
        );
    });
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("projector")
            && e.contains("insert_layer_id 5")
            && e.contains("insert_layer_id 6"),
        "{e}"
    );

    let dir = write_set("mcv-pos", Kind::Taps, &head(Kind::Taps), |x| {
        replace(
            x,
            "# positions",
            "# positions\tcanvas\tqwen35moe.rope.section_order tyxz",
        );
    });
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains("positions") && e.contains("tyxz") && e.contains("zyxt"),
        "{e}"
    );
}

#[test]
fn a_tile_line_that_breaks_the_canvas_rules_is_refused_naming_the_tile() {
    let edit_tile = |tag: &str, tile: &str, to: &str| {
        let dir = write_set(tag, Kind::Taps, &head(Kind::Taps), |x| {
            let from = format!("# tile\tgrid\t{tile}\t");
            replace(x, &from, to);
        });
        refusal(McvisSet::open(&dir, Kind::Taps))
    };
    // separator rows: tile 2 ends a row, so it carries the newline row
    let e = edit_tile(
        "mcv-t-suffix",
        "2",
        "# tile\tgrid\t2\tslice\t0\t1\t56\t56\t1\t1\t1\t2,1\t6\t3",
    );
    assert!(
        e.contains("grid") && e.contains("tile 2") && e.contains("separator rows"),
        "{e}"
    );
    // rows that do not follow the tile before
    let e = edit_tile(
        "mcv-t-row0",
        "3",
        "# tile\tgrid\t3\tslice\t1\t0\t56\t56\t1\t1\t1\t2,1\t11\t3",
    );
    assert!(e.contains("tile 3") && e.contains("rows"), "{e}");
    // a slice at the wrong cell
    let e = edit_tile(
        "mcv-t-cell",
        "1",
        "# tile\tgrid\t1\tslice\t1\t1\t56\t56\t1\t1\t1\t2,1\t3\t3",
    );
    assert!(e.contains("tile line 1"), "{e}");
    // a side that is no multiple of 56
    let e = edit_tile(
        "mcv-t-size",
        "4",
        "# tile\tgrid\t4\tslice\t1\t1\t60\t56\t1\t1\t1\t2\t13\t2",
    );
    assert!(e.contains("tile 4") && e.contains("multiple of 56"), "{e}");
    // merger rows that are not the token grid
    let e = edit_tile(
        "mcv-t-grid",
        "0",
        "# tile\tgrid\t0\toverview\t-\t-\t112\t56\t2\t1\t1\t0,1\t0\t3",
    );
    assert!(e.contains("tile 0") && e.contains("merger rows"), "{e}");
}

#[test]
fn an_image_line_that_disagrees_with_its_tiles_is_refused() {
    let dir = write_set("mcv-n-tokens", Kind::Taps, &head(Kind::Taps), |x| {
        replace(
            x,
            "# image\tgrid",
            "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t5\t16\t15",
        );
    });
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(e.contains("grid") && e.contains("hold 15 rows"), "{e}");
    let dir = write_set("mcv-n-pos", Kind::Taps, &head(Kind::Taps), |x| {
        replace(
            x,
            "# image\tgrid",
            "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t5\t15\t14",
        );
    });
    assert!(refusal(McvisSet::open(&dir, Kind::Taps)).contains("advance 15 positions"));
    // a grid with a tile line missing
    let dir = write_set("mcv-n-tiles", Kind::Taps, &head(Kind::Taps), |x| {
        x.retain(|l| !l.starts_with("# tile\tgrid\t4\t"));
    });
    assert!(refusal(McvisSet::open(&dir, Kind::Taps)).contains("4 tile lines"));
}

#[test]
fn a_missing_wrong_or_extra_row_is_refused_by_name() {
    // a shape: the overview of `one` has a merger_out of two rows, its tile one
    let dir = scratch("mcv-row-shape");
    let (extra, mut rows) = parts(&dir, Kind::Taps);
    let name = row_name("one", 0, "merger_out");
    rows.retain(|r| !r.starts_with(&format!("tensor\t{name}\t")));
    put(&dir, &mut rows, &name, [2048, 2, 1, 1], 1.0);
    finish(&dir, &head(Kind::Taps), &extra, &rows);
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(
        e.contains(&name) && e.contains("want f32 [2048, 1, 1, 1]"),
        "{e}"
    );

    // a row the tiles do not account for
    let dir = scratch("mcv-row-extra");
    let (extra, mut rows) = parts(&dir, Kind::Taps);
    put(&dir, &mut rows, "one/t0/stray", [4, 1, 1, 1], 1.0);
    finish(&dir, &head(Kind::Taps), &extra, &rows);
    assert!(refusal(McvisSet::open(&dir, Kind::Taps)).contains("tensor rows"));

    // a row gone
    let dir = scratch("mcv-row-gone");
    let (extra, mut rows) = parts(&dir, Kind::Taps);
    let name = row_name("grid", 2, "post_ln");
    rows.retain(|r| !r.starts_with(&format!("tensor\t{name}\t")));
    finish(&dir, &head(Kind::Taps), &extra, &rows);
    let e = refusal(McvisSet::open(&dir, Kind::Taps));
    assert!(e.contains(&name), "{e}");

    // a tap_effect line gone, and one that counts the wrong rows
    let dir = write_set("mcv-effect", Kind::Taps, &head(Kind::Taps), |x| {
        x.retain(|l| !l.starts_with("# tap_effect\tone\t"));
    });
    assert!(
        refusal(McvisSet::open(&dir, Kind::Taps)).contains("no `# tap_effect` line of \"one\"")
    );
    let dir = write_set("mcv-effect-n", Kind::Taps, &head(Kind::Taps), |x| {
        replace(
            x,
            "# tap_effect\tgrid",
            "# tap_effect\tgrid\tembd values 7\tdiffering 0\tmax_abs_diff 0\ttaps full",
        );
    });
    assert!(refusal(McvisSet::open(&dir, Kind::Taps)).contains("counts 7 values"));
}

/// The sets the recipe wrote, read through the reader: every row of every tile loads (f32, finite, the row's length).
#[test]
#[ignore = "hw: needs the MiniCPM-V 4.7 tower sets in $BLOOMERY_DATA on the box (just dump-ref-mcvis)"]
fn hw_the_sets_in_place_are_read() {
    let (mut sets, mut rows) = (0usize, 0usize);
    for kind in [Kind::Preproc, Kind::Taps] {
        for path in kind.family().in_place() {
            let set =
                McvisSet::open(&path, kind).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            for im in &set.images {
                for t in &im.tiles {
                    if kind == Kind::Preproc {
                        let v = set
                            .inp_raw(&im.name, t.index)
                            .unwrap_or_else(|e| panic!("{e}"));
                        assert_eq!(v.len(), t.w * t.h * 3, "{}/t{}", im.name, t.index);
                        rows += 1;
                        continue;
                    }
                    for n in tap_names().iter().map(String::as_str).chain([EMBD]) {
                        let (row, v) = set
                            .tap(&im.name, t.index, n)
                            .unwrap_or_else(|e| panic!("{e}"));
                        assert_eq!(v.len() as u64, row.count(), "{}/t{}/{n}", im.name, t.index);
                        rows += 1;
                    }
                }
            }
            println!(
                "hw_mcvis: {} {}: {} images, {} tiles — PASS",
                kind.as_str(),
                path.display(),
                set.images.len(),
                set.images.iter().map(|i| i.tiles.len()).sum::<usize>()
            );
            sets += 1;
        }
    }
    assert_eq!(sets, 3, "preproc, taps and the CPU twin");
    println!("hw_mcvis: {sets} sets, {rows} rows read");
}
