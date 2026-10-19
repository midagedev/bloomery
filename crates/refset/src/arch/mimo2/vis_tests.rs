//! The MiMo-V2.6-Flash image-input families: a set of each of the four kinds the dump writes, read by name, and each way a
//! set is refused — its identity (the projector's digest, the text model, the build), its layout against the family's, and
//! rows that disagree with the lines stating them. The reader's layout switches are the set's own lines, so the other seats'
//! sets (`arch::qwen_vis_tests`, `arch::glm5next::vis_tests`) stay what they were.

use super::vis::{self, IMAGE_PAD_ID, LCPP_BUILD, MMPROJ_BF16, MMPROJ_BF16_SHA256};
use super::{ARCH, MODEL};
use crate::RefError;
use crate::clefvis::testkit::{
    CHAT, Head, IMAGE_COLUMNS, finish, put, put_ints, put_ints_shaped, refusal, scratch, write,
};
use crate::clefvis::{
    ClefvisSet, EMBD, Kind, MIMO_HEAD_DIM, MIMO_HEADS, MIMO_PATCH_EMBED, MIMO_POST_LN, MIMO_WIDTH,
    PositionKind, ProjectorKind, TOWER_ARCH, TapScope, VIT_OUT, mimo_tap_names,
};
use crate::family::{Family, Identity};
use std::path::Path;

/// The seat's family of `kind`.
fn family(kind: Kind) -> &'static Family {
    crate::arch::all()
        .find(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
        .unwrap_or_else(|| panic!("{kind:?}: no family of the mimovl seat in the table"))
}

fn tower(kind: Kind) -> Head<'static> {
    Head {
        kind: kind.as_str(),
        model: MMPROJ_BF16,
        build: LCPP_BUILD,
        arch: TOWER_ARCH,
        mmproj: MMPROJ_BF16,
        mmproj_sha: MMPROJ_BF16_SHA256,
        complete: true,
    }
}

fn text(kind: Kind) -> Head<'static> {
    Head {
        model: MODEL,
        arch: ARCH,
        ..tower(kind)
    }
}

const PROJECTOR: &str = "# projector\tmimovl\tpatch_size 16\tspatial_merge_size 2\talign 32\timage_min_pixels 8192\timage_max_pixels 8388608";

/// A 96x64 image planned 128x96: a 4x3 grid of 12 tokens of 32 pixels a side, advancing the decoder by its 12 tokens.
const IMAGE: &str = "# image\tg\taa\tbb\t96\t64\t128\t96\t4\t3\t12\t12";

/// The 48 patches of that image.
const PATCHES: u64 = 48;

fn with(extra: &[&str]) -> Vec<String> {
    extra.iter().map(|l| (*l).to_string()).collect()
}

/// Set A: the 128x96 canvas of the one image.
fn preproc(dir: &Path) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "g/inp_raw", [128, 96, 3, 1], 0.5);
    finish(
        dir,
        &tower(Kind::Preproc),
        &with(&[IMAGE_COLUMNS, PROJECTOR, IMAGE]),
        &rows,
    );
}

/// Set B as the dump writes it for `scope`: a full tower holds the patch embedding, five blocks of six nodes, the final norm and
/// the merger's output; a final one the last two.
fn taps(dir: &Path, scope: TapScope) {
    let mut rows = Vec::new();
    let nodes = match scope {
        TapScope::Full => mimo_tap_names(),
        TapScope::Final => vec![MIMO_POST_LN.to_string(), VIT_OUT.to_string()],
        TapScope::EmbdOnly => Vec::new(),
    };
    for n in nodes {
        let ne = if n == VIT_OUT {
            [4096, 12, 1, 1]
        } else if n.starts_with("Qcur_rope") {
            [MIMO_HEAD_DIM, MIMO_HEADS, PATCHES, 1]
        } else {
            [MIMO_WIDTH, PATCHES, 1, 1]
        };
        put(dir, &mut rows, &format!("g/{n}"), ne, 1.0);
    }
    put(dir, &mut rows, &format!("g/{EMBD}"), [4096, 12, 1, 1], 2.0);
    let how = match scope {
        TapScope::Full => "full",
        TapScope::Final => "final",
        TapScope::EmbdOnly => "inp_raw only",
    };
    let effect =
        format!("# tap_effect\tg\tembd values 49152\tdiffering 0\tmax_abs_diff 0\ttaps {how}");
    let extra = with(&[IMAGE_COLUMNS, PROJECTOR, IMAGE, &effect]);
    finish(dir, &tower(Kind::Taps), &extra, &rows);
}

/// The header lines of the prompt of 16 ids: two text ids, the image's 12 tokens, two text ids.
fn prompt_lines() -> Vec<String> {
    vec![
        "# tokens_count\t16".to_string(),
        format!("# image_pad_id\t{IMAGE_PAD_ID}"),
        "# span columns\tindex image at len nx ny n_pos start_pos end_pos".to_string(),
        "# span\t0\tg\t2\t12\t4\t3\t12\t2\t14".to_string(),
        IMAGE_COLUMNS.to_string(),
        IMAGE.to_string(),
        PROJECTOR.to_string(),
    ]
}

/// Set C of the 16-id prompt: `pos` `[1, 16]`, one position a row, the image's rows included.
fn hidden(dir: &Path, positions: Option<&str>) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "result_norm", [4096, 16, 1, 1], 1.5);
    let pos: Vec<i32> = (0..16).collect();
    put_ints_shaped(dir, &mut rows, "pos", [1, 16], &pos);
    let mut extra = prompt_lines();
    extra.extend(positions.map(|p| format!("# positions\t{p}")));
    finish(dir, &text(Kind::Hidden), &extra, &rows);
}

/// The ids of the 16-id prompt: the image's 12 tokens are `<|image_pad|>`.
fn ids() -> Vec<i32> {
    let pad = IMAGE_PAD_ID as i32;
    let mut v = vec![11, 12];
    v.extend(std::iter::repeat_n(pad, 12));
    v.extend([13, 14]);
    v
}

/// Set E of the 16-id prompt.
fn chat_ids(dir: &Path, ids: &[i32]) {
    let mut rows = Vec::new();
    put_ints(dir, &mut rows, "ids", ids);
    let mut extra = prompt_lines();
    extra.extend([
        CHAT.to_string(),
        "# prompt\tq".to_string(),
        "# prompt_sha256\t00".to_string(),
    ]);
    finish(dir, &text(Kind::ChatIds), &extra, &rows);
}

fn edit(dir: &Path, f: impl Fn(&str) -> String) {
    let path = dir.join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let new = f(&text);
    assert_ne!(new, text, "the edit changed nothing");
    write(&path, new.as_bytes());
}

fn replace(dir: &Path, from: &str, to: &str) {
    edit(dir, |t| t.replace(from, to));
}

fn ok(r: Result<ClefvisSet, RefError>) -> ClefvisSet {
    r.unwrap_or_else(|e| panic!("{e}"))
}

fn open(dir: &Path, kind: Kind) -> Result<ClefvisSet, RefError> {
    ClefvisSet::open(dir, family(kind))
}

/// A set of each of the four kinds reads by name: the grid of 32-pixel tokens, the `pos` row of normal positions (and the
/// M-RoPE accessor refusing it), every node of the tower tapped in full at the shape the `mimovl` graph gives it, the
/// output-end pair of a final set, and the `<|image_pad|>` token of MiMo's vocabulary.
#[test]
fn a_mimovl_set_of_each_kind_reads_by_name() {
    let dir = scratch("mimo-reads");

    preproc(&dir);
    let set = ok(open(&dir, Kind::Preproc));
    assert_eq!(set.projector.kind, ProjectorKind::Mimovl);
    assert_eq!(set.projector.align, 32);
    assert_eq!(
        (
            set.projector.image_min_pixels,
            set.projector.image_max_pixels
        ),
        (Some(8192), Some(8_388_608))
    );
    let image = set.image("g").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (image.best_w, image.nx, image.n_tokens, image.n_pos),
        (128, 4, 12, 12)
    );
    assert_eq!(
        set.inp_raw("g").unwrap_or_else(|e| panic!("{e}")).len(),
        128 * 96 * 3
    );

    taps(&dir, TapScope::Full);
    let set = ok(open(&dir, Kind::Taps));
    let names = mimo_tap_names();
    assert_eq!(names.len(), 1 + 5 * 6 + 2);
    assert_eq!(names[0], MIMO_PATCH_EMBED);
    assert_eq!(names[names.len() - 2..], [MIMO_POST_LN, VIT_OUT]);
    for node in names.iter().map(String::as_str).chain([EMBD]) {
        let (row, v) = set.tap("g", node).unwrap_or_else(|e| panic!("{node}: {e}"));
        let want: [u64; 3] = if node == VIT_OUT || node == EMBD {
            [4096, 12, 1]
        } else if node.starts_with("Qcur_rope") {
            [64, 32, 48]
        } else {
            [1280, 48, 1]
        };
        assert_eq!(row.ne[..3], want, "{node}");
        assert_eq!(v.len() as u64, want.iter().product::<u64>(), "{node}");
    }
    assert!(set.tap("g", "norm_b-27").is_err(), "a Qwen node");

    taps(&dir, TapScope::Final);
    let set = ok(open(&dir, Kind::Taps));
    for node in [MIMO_POST_LN, VIT_OUT, EMBD] {
        set.tap("g", node).unwrap_or_else(|e| panic!("{node}: {e}"));
    }
    let e = set
        .tap("g", "ln1-0")
        .expect_err("a block node of a final set")
        .to_string();
    assert!(
        e.contains("carries post_ln, vit_out and embd only, not ln1-0"),
        "{e}"
    );

    hidden(&dir, Some("normal"));
    let set = ok(open(&dir, Kind::Hidden));
    assert_eq!(set.projector.kind.positions(), PositionKind::Normal);
    assert_eq!(
        set.normal_positions().unwrap_or_else(|e| panic!("{e}")),
        (0..16).collect::<Vec<i32>>()
    );
    let e = set.positions().expect_err("no M-RoPE triples").to_string();
    assert!(e.contains("normal positions"), "{e}");

    chat_ids(&dir, &ids());
    let set = ok(open(&dir, Kind::ChatIds));
    assert_eq!(set.projector.kind, ProjectorKind::Mimovl);
}

/// A node of the wrong shape is refused by name and shape: `Qcur_rope` as wide as the model (the head width of 64 times 32
/// heads is 2,048, not 1,280), a block node of the wrong patch count, and `vit_out` as the unmerged rows.
#[test]
fn a_node_of_the_wrong_shape_is_refused() {
    let dir = scratch("mimo-shapes");
    taps(&dir, TapScope::Full);
    replace(
        &dir,
        "tensor\tg/Qcur_rope-5\t0\tf32\t64\t32\t48\t1\t",
        "tensor\tg/Qcur_rope-5\t0\tf32\t1280\t48\t1\t1\t",
    );
    let e = refusal(open(&dir, Kind::Taps));
    assert!(
        e.contains("g/Qcur_rope-5 is [1280, 48, 1, 1], want [64, 32, 48, 1]"),
        "{e}"
    );

    taps(&dir, TapScope::Full);
    replace(
        &dir,
        "tensor\tg/layer_out-27\t0\tf32\t1280\t48\t1\t1\t",
        "tensor\tg/layer_out-27\t0\tf32\t1280\t12\t1\t1\t",
    );
    let e = refusal(open(&dir, Kind::Taps));
    assert!(
        e.contains("g/layer_out-27 is [1280, 12, 1, 1], want [1280, 48, 1, 1]"),
        "{e}"
    );

    taps(&dir, TapScope::Full);
    replace(
        &dir,
        "tensor\tg/vit_out\t0\tf32\t4096\t12\t1\t1\t",
        "tensor\tg/vit_out\t0\tf32\t1280\t48\t1\t1\t",
    );
    let e = refusal(open(&dir, Kind::Taps));
    assert!(
        e.contains("g/vit_out is [1280, 48, 1, 1], want [4096, 12, 1, 1]"),
        "{e}"
    );
}

/// The seat's identity: a set of another projector file, another text model or another build is refused by the family
/// whose line it breaks, and a Qwen-layout `# projector` line in a MiMo set is refused by the layout.
#[test]
fn a_set_of_another_identity_or_layout_is_refused() {
    let dir = scratch("mimo-identity");
    preproc(&dir);
    replace(&dir, MMPROJ_BF16_SHA256, &"0".repeat(64));
    assert!(refusal(open(&dir, Kind::Preproc)).contains("sha256"));

    preproc(&dir);
    replace(&dir, LCPP_BUILD, "36a73916e");
    assert!(refusal(open(&dir, Kind::Preproc)).contains("36a73916e"));

    chat_ids(&dir, &ids());
    replace(
        &dir,
        MODEL,
        "/models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
    );
    assert!(refusal(open(&dir, Kind::ChatIds)).contains("MiMo-V2.6-Flash-RL"));

    preproc(&dir);
    replace(&dir, "# projector\tmimovl", "# projector\tglm5v");
    let e = refusal(open(&dir, Kind::Preproc));
    assert!(e.contains("projector \"glm5v\""), "{e}");
}

/// The positions row is the layout's: a MiMo prompt set holding `mrope_pos` is refused by name, and the profile names the
/// layout, the image token and the text width the families hold.
#[test]
fn the_profile_names_the_layout_and_positions_are_normal() {
    assert_eq!(vis::PROFILE.projector, ProjectorKind::Mimovl);
    assert_eq!(vis::PROFILE.image_pad_id, 151_655);
    assert_eq!(vis::PROFILE.n_embd, 4096);
    assert!(vis::PROFILE.respell.is_none());
    let dir = scratch("mimo-positions");
    hidden(&dir, Some("normal"));
    replace(&dir, "tensor\tpos\t0", "tensor\tmrope_pos\t0");
    let e = refusal(open(&dir, Kind::Hidden));
    assert!(
        e.contains("the mimovl layout feeds normal positions, the pos row"),
        "{e}"
    );
}
