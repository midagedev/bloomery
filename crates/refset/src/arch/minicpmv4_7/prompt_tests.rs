//! The MiniCPM-V 4.7 prompt reader: a scratch set of each kind with the canvas layout of two images in one prompt, the
//! identity every other seat's set fails by name (the text file, the oracle build, the projector digest, the `# clefvis`
//! kind), the rules a span and a position lane must keep, and the rows each kind carries.

use super::prompt::{
    CHATIDS, HIDDEN, IMAGE_PAD_ID, LANES, PromptKind, PromptSet, TEXT_ARCH, TEXT_MODEL,
};
use super::vis::{MC47_LCPP_BUILD, MMPROJ_F16, MMPROJ_F16_SHA256, N_EMBD};
use crate::clefvis::QVIS_LCPP_BUILD;
use crate::clefvis::testkit::{Head, finish, put, put_ints, put_ints_shaped, refusal, scratch};
use std::path::PathBuf;

const PROJECTOR: &str = "# projector\tminicpmv4_7\tpatch_size 14\timage_size 448\tn_merge 4\tprojection_dim 2048\tblocks 27\tinsert_layer_id 6";
const POSITIONS: &str = "# positions\tcanvas\tqwen35moe.rope.section_order zyxt";
const IMAGE_COLUMNS: &str =
    "# image columns\tname png_sha256 rgb8_sha256 w h grid_x grid_y n_tiles n_tokens n_pos";
const SPAN_COLUMNS: &str =
    "# span columns\tindex image at len n_tiles grid_x grid_y n_pos start_pos end_pos";

/// Two images in one prompt of 23 ids: text, text, image `one`, text, image `grid`, text, text.
/// `one` is an overview alone of 2 merger rows and the `</image>` row: 3 rows, 2 positions (the overview and the separator).
/// `grid` is an overview and a 2 x 2 grid of slices, a merger row each, with their separators written out here by hand from
/// mtmd's `clip_suffix_rows` (`tools/mtmd/clip-impl.h`) — overview {0,1}, slices {2,1}, {2,3,1}, {2,1}, {2} — so the
/// reader's rule is not checked against itself: 5 + 10 = 15 rows, 15 positions.
const N_IDS: usize = 23;
const SPANS: [&str; 2] = [
    "# span\t0\tone\t2\t3\t1\t0\t0\t2\t2\t4",
    "# span\t1\tgrid\t6\t15\t5\t2\t2\t15\t5\t20",
];
const IMAGES: [&str; 2] = [
    "# image\tone\taa\tbb\t112\t56\t0\t0\t1\t3\t2",
    "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t5\t15\t15",
];
const CHAT: &str = "# chat\ttemplate_sha256 aa\tuse_jinja 1\tenable_thinking 1\treasoning_format deepseek\tadd_generation_prompt 1\tcontinue_final_message none";
const PROMPT: &str = "# prompt\t<|im_start|>user\\n<__media__><__media__>Describe.<|im_end|>\\n<|im_start|>assistant\\n";

/// The positions fed, `(t, y, x, z)` per id, written out by hand from `mtmd_canvas_decoder_pos`:
/// `one` starts at position 2 (origin 1), `grid` at 5 (origin 4) because `one` saved a position (3 rows, 2 positions).
fn lanes() -> Vec<[i32; 4]> {
    let mut v = vec![[0, 0, 0, 0], [1, 1, 1, 0]];
    v.extend([[2, 1, 1, 1], [2, 1, 2, 1], [3, 3, 3, 3]]); // one: the tile's two rows, `</image>`
    v.push([4, 4, 4, 0]);
    v.extend([
        [5, 4, 4, 4], // overview
        [6; 4],
        [7; 4],
        [8, 4, 4, 4], // slice (0, 0)
        [9; 4],
        [10; 4],
        [11, 4, 5, 4], // slice (0, 1)
        [12; 4],
        [13; 4],
        [14; 4],
        [15, 5, 4, 4], // slice (1, 0)
        [16; 4],
        [17; 4],
        [18, 5, 5, 4], // slice (1, 1)
        [19; 4],
    ]);
    v.extend([[20, 20, 20, 0], [21, 21, 21, 0]]);
    assert_eq!(v.len(), N_IDS);
    v
}

/// The ids of the chat request: the pad id fills the two spans.
fn ids() -> Vec<i32> {
    let pad = IMAGE_PAD_ID as i32;
    let mut v = vec![11, 12];
    v.extend([pad; 3]);
    v.push(13);
    v.extend([pad; 15]);
    v.extend([14, 15]);
    assert_eq!(v.len(), N_IDS);
    v
}

fn head(kind: PromptKind) -> Head<'static> {
    Head {
        kind: kind.as_str(),
        model: TEXT_MODEL,
        build: MC47_LCPP_BUILD,
        arch: TEXT_ARCH,
        mmproj: MMPROJ_F16,
        mmproj_sha: MMPROJ_F16_SHA256,
        complete: true,
    }
}

/// A set of `kind`, its header lines and position lanes edited by the caller before they are written.
fn write_set(
    tag: &str,
    kind: PromptKind,
    h: &Head,
    edit: impl Fn(&mut Vec<String>),
    tweak: impl Fn(&mut Vec<[i32; 4]>),
) -> PathBuf {
    let dir = scratch(tag);
    let mut extra = vec![
        PROJECTOR.to_string(),
        POSITIONS.to_string(),
        format!("# tokens_count\t{N_IDS}"),
        format!("# image_pad_id\t{IMAGE_PAD_ID}"),
    ];
    let mut rows = Vec::new();
    match kind {
        PromptKind::Hidden => {
            extra.push("# rows\tmtmd".to_string());
            extra.push("# pos_lanes\tt y x z\ta text row is fed p p p 0".to_string());
            put(
                &dir,
                &mut rows,
                "result_norm",
                [N_EMBD as u64, N_IDS as u64, 1, 1],
                1.5,
            );
            let mut pos = lanes();
            tweak(&mut pos);
            let flat: Vec<i32> = pos.iter().flatten().copied().collect();
            put_ints_shaped(
                &dir,
                &mut rows,
                "mrope_pos",
                [LANES as u64, N_IDS as u64],
                &flat,
            );
        }
        PromptKind::ChatIds => {
            put_ints(&dir, &mut rows, "ids", &ids());
            extra.push(CHAT.to_string());
            extra.push(PROMPT.to_string());
            extra.push("# prompt_sha256\t00".to_string());
        }
    }
    extra.push(SPAN_COLUMNS.to_string());
    extra.extend(SPANS.iter().map(|l| (*l).to_string()));
    extra.push(IMAGE_COLUMNS.to_string());
    extra.extend(IMAGES.iter().map(|l| (*l).to_string()));
    edit(&mut extra);
    finish(&dir, h, &extra, &rows);
    dir
}

fn hidden(tag: &str) -> PathBuf {
    write_set(
        tag,
        PromptKind::Hidden,
        &head(PromptKind::Hidden),
        |_| {},
        |_| {},
    )
}

fn replace(extra: &mut [String], from: &str, to: &str) {
    let at = extra
        .iter()
        .position(|l| l.starts_with(from))
        .unwrap_or_else(|| panic!("no line {from}"));
    extra[at] = to.to_string();
}

#[test]
fn the_prompt_families_are_the_seats_text_model_on_the_oracle_build() {
    for (kind, family) in [
        (PromptKind::Hidden, &HIDDEN),
        (PromptKind::ChatIds, &CHATIDS),
    ] {
        assert!(std::ptr::eq(kind.family(), family));
        assert_eq!(family.arch, Some(TEXT_ARCH));
        assert_eq!(family.runs().as_deref().ok(), Some(TEXT_MODEL));
        assert_eq!(
            family.build.map(|b| b.want()).as_deref(),
            Some(MC47_LCPP_BUILD)
        );
    }
    assert_eq!(
        HIDDEN.sets,
        [
            "ref_minicpmv47_vis_hidden_mc1",
            "ref_minicpmv47_vis_hidden_mc2",
            "ref_minicpmv47_vis_hidden_mc3",
            "ref_minicpmv47_vis_hidden_mc1.cpu",
        ]
    );
    assert_eq!(CHATIDS.sets[0], "ref_minicpmv47_vis_chatids_mc1");
    // the kinds are the canvas ones, never Qwen's `hidden` and `chatids`
    assert_eq!(PromptKind::Hidden.as_str(), "canvas-hidden");
    assert_eq!(PromptKind::ChatIds.as_str(), "canvas-chatids");
}

#[test]
fn a_hidden_set_is_taken_and_its_rows_and_four_lanes_are_read() {
    let set = PromptSet::open(&hidden("mcp-hid"), PromptKind::Hidden).expect("a hidden set");
    assert!(!set.cpu);
    assert_eq!(set.card, "NVIDIA RTX A6000");
    assert_eq!(set.n_ids, N_IDS);
    assert_eq!(set.spans.len(), 2);
    assert_eq!(
        (set.spans[1].at, set.spans[1].len, set.spans[1].start_pos),
        (6, 15, 5)
    );
    let grid = set.image("grid").expect("grid");
    assert_eq!(
        (grid.grid_x, grid.grid_y, grid.n_tiles, grid.separators()),
        (2, 2, 5, 10)
    );
    assert!(set.image("nope").is_err());
    assert_eq!(
        set.result_norm().expect("result_norm").len(),
        N_EMBD * N_IDS
    );
    let pos = set.positions().expect("positions");
    assert_eq!(pos, lanes());
    // z is the canvas origin in a tile's rows and the position itself in a separator's; a text row's is 0
    assert_eq!(pos[0][3], 0);
    assert!(pos[6..21].iter().all(|p| p[3] != 0));
    assert!(pos[9..12].iter().all(|p| p[3] >= 4));
    set.check_lanes().expect("the lanes keep the canvas rules");
    assert!(
        set.ids()
            .is_err_and(|e| e.to_string().contains("reads a canvas-chatids set"))
    );
}

#[test]
fn a_chat_ids_set_is_taken_and_its_ids_prompt_and_request_are_read() {
    let dir = write_set(
        "mcp-chat",
        PromptKind::ChatIds,
        &head(PromptKind::ChatIds),
        |_| {},
        |_| {},
    );
    let set = PromptSet::open(&dir, PromptKind::ChatIds).expect("a chat-ids set");
    let got = set.ids().expect("ids");
    assert_eq!(got.len(), N_IDS);
    assert_eq!(got.iter().filter(|&&i| i == IMAGE_PAD_ID).count(), 18);
    assert_eq!(
        set.prompt().expect("prompt"),
        "<|im_start|>user\n<__media__><__media__>Describe.<|im_end|>\n<|im_start|>assistant\n"
    );
    let chat = set.chat().expect("chat");
    assert!(chat.enable_thinking && chat.add_generation_prompt && !chat.continue_final_message);
    assert!(
        set.result_norm()
            .is_err_and(|e| e.to_string().contains("reads a canvas-hidden set"))
    );
    assert!(
        set.positions()
            .is_err_and(|e| e.to_string().contains("reads a canvas-hidden set"))
    );
}

/// FAIL-first: with the family's `runs` returning Qwen3.6's file the set below is taken; with the reader's kind check
/// off, a tower set or Qwen's own `hidden` would be.
#[test]
fn another_text_file_is_refused_by_the_family() {
    let mut h = head(PromptKind::Hidden);
    h.model = crate::arch::qwen35moe::MODEL;
    let dir = write_set("mcp-q36", PromptKind::Hidden, &h, |_| {}, |_| {});
    let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
    assert!(
        e.contains(crate::arch::qwen35moe::MODEL) && e.contains(TEXT_MODEL),
        "{e}"
    );
}

#[test]
fn another_oracle_build_projector_digest_or_architecture_is_refused() {
    let mut h = head(PromptKind::Hidden);
    h.build = QVIS_LCPP_BUILD;
    let e = refusal(PromptSet::open(
        &write_set("mcp-build", PromptKind::Hidden, &h, |_| {}, |_| {}),
        PromptKind::Hidden,
    ));
    assert!(
        e.contains(QVIS_LCPP_BUILD) && e.contains(MC47_LCPP_BUILD),
        "{e}"
    );

    let mut h = head(PromptKind::Hidden);
    h.mmproj_sha = "00";
    let e = refusal(PromptSet::open(
        &write_set("mcp-sha", PromptKind::Hidden, &h, |_| {}, |_| {}),
        PromptKind::Hidden,
    ));
    assert!(
        e.contains("stale reference") && e.contains(MMPROJ_F16_SHA256),
        "{e}"
    );

    let mut h = head(PromptKind::Hidden);
    h.arch = "clip";
    let e = refusal(PromptSet::open(
        &write_set("mcp-arch", PromptKind::Hidden, &h, |_| {}, |_| {}),
        PromptKind::Hidden,
    ));
    assert!(e.contains("clip") && e.contains(TEXT_ARCH), "{e}");

    // the digest under another file at the pinned path, or an unfinished dump
    let mut h = head(PromptKind::Hidden);
    h.complete = false;
    assert!(
        PromptSet::open(
            &write_set("mcp-partial", PromptKind::Hidden, &h, |_| {}, |_| {}),
            PromptKind::Hidden
        )
        .is_err()
    );
}

#[test]
fn a_set_of_another_kind_is_refused_by_its_clefvis_line() {
    // a chat-ids set read as hidden, a hidden set as chat ids, Qwen's kind spellings and the tower's
    let dir = write_set(
        "mcp-k1",
        PromptKind::ChatIds,
        &head(PromptKind::ChatIds),
        |_| {},
        |_| {},
    );
    let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
    assert!(
        e.contains("canvas-chatids") && e.contains("canvas-hidden"),
        "{e}"
    );
    let e = refusal(PromptSet::open(&hidden("mcp-k2"), PromptKind::ChatIds));
    assert!(
        e.contains("canvas-hidden") && e.contains("canvas-chatids"),
        "{e}"
    );
    for spelt in ["hidden", "chatids", "canvas-taps", "canvas-preproc"] {
        let mut h = head(PromptKind::Hidden);
        h.kind = spelt;
        let dir = write_set("mcp-k3", PromptKind::Hidden, &h, |_| {}, |_| {});
        let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
        assert!(
            e.contains(spelt) && e.contains("canvas-hidden"),
            "{spelt}: {e}"
        );
    }
}

#[test]
fn a_set_laid_out_for_another_projector_or_rope_order_is_refused() {
    let edit = |from: &'static str, to: &'static str| {
        move |extra: &mut Vec<String>| replace(extra, from, to)
    };
    let dir = write_set(
        "mcp-pos",
        PromptKind::Hidden,
        &head(PromptKind::Hidden),
        edit(
            "# positions",
            "# positions\tcanvas\tqwen35moe.rope.section_order tyxz",
        ),
        |_| {},
    );
    let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
    assert!(e.contains("zyxt") && e.contains("tyxz"), "{e}");
    let dir = write_set(
        "mcp-proj",
        PromptKind::Hidden,
        &head(PromptKind::Hidden),
        edit("# projector", "# projector\tqwen3vl_merger"),
        |_| {},
    );
    assert!(refusal(PromptSet::open(&dir, PromptKind::Hidden)).contains("qwen3vl_merger"));
}

#[test]
fn the_image_pad_id_is_the_vocabularys_unk() {
    for bad in ["248056", "248078", "x"] {
        let dir = write_set(
            "mcp-pad",
            PromptKind::Hidden,
            &head(PromptKind::Hidden),
            |e| replace(e, "# image_pad_id", &format!("# image_pad_id\t{bad}")),
            |_| {},
        );
        let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
        assert!(
            e.contains("image_pad_id") && e.contains("248077"),
            "{bad}: {e}"
        );
    }
}

#[test]
fn a_span_must_hold_its_images_rows_and_start_where_the_decoder_is() {
    let cases = [
        // a pad run shorter than the chunk
        (
            "# span\t1\t",
            "# span\t1\tgrid\t6\t14\t5\t2\t2\t15\t5\t20",
            "holds 14 ids",
        ),
        // a grid the image line does not state
        (
            "# span\t1\t",
            "# span\t1\tgrid\t6\t15\t5\t1\t4\t15\t5\t20",
            "1 x 4 grid",
        ),
        // the second image starts at its index, not less the position `one` saved
        (
            "# span\t1\t",
            "# span\t1\tgrid\t6\t15\t5\t2\t2\t15\t6\t21",
            "runs from position 6 to 21",
        ),
        // a span outside the ids
        (
            "# span\t1\t",
            "# span\t1\tgrid\t10\t15\t5\t2\t2\t15\t9\t24",
            "of 23 ids",
        ),
    ];
    for (from, to, expect) in cases {
        let dir = write_set(
            "mcp-span",
            PromptKind::Hidden,
            &head(PromptKind::Hidden),
            |e| replace(e, from, to),
            |_| {},
        );
        let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
        assert!(e.contains(expect), "{to}: {e}");
    }
    // an image line that breaks the canvas arithmetic: 4 tiles in a 2 x 2 grid, n_pos off by one
    for (to, expect) in [
        (
            "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t4\t15\t15",
            "4 tiles in a 2 x 2 grid",
        ),
        (
            "# image\tgrid\tcc\tdd\t112\t112\t2\t2\t5\t15\t14",
            "15 rows and 14 positions",
        ),
        (
            "# image\tgrid\tcc\tdd\t112\t112\t0\t2\t3\t15\t15",
            "tiles in a 0 x 2 grid",
        ),
    ] {
        let dir = write_set(
            "mcp-img",
            PromptKind::Hidden,
            &head(PromptKind::Hidden),
            |e| replace(e, "# image\tgrid", to),
            |_| {},
        );
        let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
        assert!(e.contains(expect), "{to}: {e}");
    }
}

#[test]
fn a_positions_row_of_three_lanes_is_refused_by_name() {
    let dir = scratch("mcp-lanes3");
    let mut rows = Vec::new();
    put(
        &dir,
        &mut rows,
        "result_norm",
        [N_EMBD as u64, N_IDS as u64, 1, 1],
        1.5,
    );
    let flat: Vec<i32> = lanes().iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    put_ints_shaped(&dir, &mut rows, "mrope_pos", [3, N_IDS as u64], &flat);
    let extra = vec![
        PROJECTOR.to_string(),
        POSITIONS.to_string(),
        format!("# tokens_count\t{N_IDS}"),
        format!("# image_pad_id\t{IMAGE_PAD_ID}"),
        "# rows\tmtmd".to_string(),
        "# pos_lanes\tt y x z\ta text row is fed p p p 0".to_string(),
        SPAN_COLUMNS.to_string(),
        SPANS[0].to_string(),
        SPANS[1].to_string(),
        IMAGE_COLUMNS.to_string(),
        IMAGES[0].to_string(),
        IMAGES[1].to_string(),
    ];
    finish(&dir, &head(PromptKind::Hidden), &extra, &rows);
    let e = refusal(PromptSet::open(&dir, PromptKind::Hidden));
    assert!(
        e.contains("mrope_pos") && e.contains("[4, 23, 1, 1]"),
        "{e}"
    );
}

/// Each tweak breaks one rule of the canvas positions; the set is still a well-formed hidden set, and only
/// [`PromptSet::check_lanes`] sees it.
#[test]
fn a_lane_that_breaks_the_canvas_rules_is_refused_by_check_lanes() {
    type Tweak = fn(&mut Vec<[i32; 4]>);
    let tweaks: [(&str, Tweak, &str); 7] = [
        ("a text row's z", |p| p[1][3] = 1, "a text row is fed"),
        (
            "the text row after an image",
            |p| p[5][0] = 5,
            "a text row is fed",
        ),
        ("a tile row's t", |p| p[9][0] = 9, "its tile's t is 8"),
        (
            "a separator in one lane",
            |p| p[7][3] = 0,
            "a separator row",
        ),
        ("a tile row's origin", |p| p[12][3] = 3, "a separator row"),
        (
            "a slice cell left of the origin",
            |p| p[16][2] = 3,
            "its tile's t",
        ),
        // the totals stay (5 tiles, 10 separator rows, 15 positions); slice (0, 1) loses one to slice (0, 0)
        (
            "a separator moved to the tile before",
            |p| {
                p[12] = [11; 4];
                p[13] = [12, 4, 5, 4];
                p[14] = [13; 4];
                p[15] = [14; 4];
            },
            "tile 1 of span 1 has 3 separator rows, a 2 x 2 grid gives it 2",
        ),
    ];
    for (what, tweak, expect) in tweaks {
        let dir = write_set(
            "mcp-lane",
            PromptKind::Hidden,
            &head(PromptKind::Hidden),
            |_| {},
            tweak,
        );
        let set = PromptSet::open(&dir, PromptKind::Hidden).expect("a well-formed set");
        let e = set.check_lanes().expect_err(what).to_string();
        assert!(e.contains(expect), "{what}: {e}");
    }
}

#[test]
fn the_ids_of_a_chat_set_keep_the_pad_in_the_spans_only() {
    let leak = |at: usize| {
        let dir = scratch("mcp-leak");
        let mut rows = Vec::new();
        let mut v = ids();
        v[at] = IMAGE_PAD_ID as i32;
        put_ints(&dir, &mut rows, "ids", &v);
        let extra = vec![
            PROJECTOR.to_string(),
            POSITIONS.to_string(),
            format!("# tokens_count\t{N_IDS}"),
            format!("# image_pad_id\t{IMAGE_PAD_ID}"),
            CHAT.to_string(),
            PROMPT.to_string(),
            "# prompt_sha256\t00".to_string(),
            SPAN_COLUMNS.to_string(),
            SPANS[0].to_string(),
            SPANS[1].to_string(),
            IMAGE_COLUMNS.to_string(),
            IMAGES[0].to_string(),
            IMAGES[1].to_string(),
        ];
        finish(&dir, &head(PromptKind::ChatIds), &extra, &rows);
        refusal(PromptSet::open(&dir, PromptKind::ChatIds))
    };
    assert!(leak(5).contains("ids[5]"));
    // a hole in a span
    let dir = scratch("mcp-hole");
    let mut rows = Vec::new();
    let mut v = ids();
    v[10] = 99;
    put_ints(&dir, &mut rows, "ids", &v);
    let mut extra = vec![
        PROJECTOR.to_string(),
        POSITIONS.to_string(),
        format!("# tokens_count\t{N_IDS}"),
        format!("# image_pad_id\t{IMAGE_PAD_ID}"),
        CHAT.to_string(),
        PROMPT.to_string(),
        SPAN_COLUMNS.to_string(),
        SPANS[0].to_string(),
        SPANS[1].to_string(),
        IMAGE_COLUMNS.to_string(),
        IMAGES[0].to_string(),
        IMAGES[1].to_string(),
    ];
    extra.push("# prompt_sha256\t00".to_string());
    finish(&dir, &head(PromptKind::ChatIds), &extra, &rows);
    assert!(refusal(PromptSet::open(&dir, PromptKind::ChatIds)).contains("ids[10]"));
}

#[test]
fn the_prompt_holds_one_media_marker_an_image_and_one_turn_rule() {
    let dir = write_set(
        "mcp-marker",
        PromptKind::ChatIds,
        &head(PromptKind::ChatIds),
        |e| replace(e, "# prompt\t", "# prompt\tno markers here"),
        |_| {},
    );
    assert!(refusal(PromptSet::open(&dir, PromptKind::ChatIds)).contains("0 <__media__> markers"));
    let dir = write_set(
        "mcp-turn",
        PromptKind::ChatIds,
        &head(PromptKind::ChatIds),
        |e| {
            replace(
                e,
                "# chat",
                &CHAT.replace("continue_final_message none", "continue_final_message auto"),
            )
        },
        |_| {},
    );
    assert!(refusal(PromptSet::open(&dir, PromptKind::ChatIds)).contains("never both or neither"));
}

/// The sets the recipe wrote, read through the reader: the hidden states of the three cases and the CPU twin of the first,
/// and the chat ids of all three. The numbers are the self-check of the dump (`mc1`: 64 tower rows and the `</image>` row,
/// `mc2`: the 360 rows `Marks::rows` gives 1000 x 700, `mc3`: two overviews alone), the canvas rules on the real positions,
/// and the distance of the CPU twin from the card.
#[test]
#[ignore = "hw: needs the MiniCPM-V 4.7 prompt sets in $BLOOMERY_DATA on the box (just dump-ref-mcvis --ids, then the hidden sets)"]
fn hw_the_prompt_sets_in_place_are_read() {
    let name = |p: &std::path::Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    };
    let (mut hidden, mut chat) = (std::collections::BTreeMap::new(), 0usize);
    for kind in [PromptKind::Hidden, PromptKind::ChatIds] {
        for path in kind.family().in_place() {
            let set =
                PromptSet::open(&path, kind).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let runs: Vec<usize> = set.spans.iter().map(|s| s.len).collect();
            let case = name(&path);
            let want: &[usize] = if case.contains("mc1") {
                &[65]
            } else if case.contains("mc2") {
                &[360]
            } else {
                &[65, 65]
            };
            assert_eq!(runs, want, "{case}");
            match kind {
                PromptKind::Hidden => {
                    set.check_lanes().unwrap_or_else(|e| panic!("{case}: {e}"));
                    let pos = set.positions().expect("positions");
                    for s in &set.spans {
                        assert!(
                            pos[s.at..s.at + s.len].iter().all(|p| p[3] > 0),
                            "{case}: z is the canvas origin in a chunk"
                        );
                    }
                    let v = set.result_norm().expect("result_norm");
                    assert_eq!(v.len(), N_EMBD * set.n_ids);
                    println!(
                        "hw_mcprompt: {case}: {} ids, spans {runs:?} — PASS",
                        set.n_ids
                    );
                    let first = (set.spans[0].at, set.spans[0].len);
                    hidden.insert(case, (pos, v, first));
                }
                PromptKind::ChatIds => {
                    let ids = set.ids().expect("ids");
                    assert_eq!(ids.len(), set.n_ids);
                    println!("hw_mcprompt: {case}: first 8 ids {:?} — PASS", &ids[..8]);
                    chat += 1;
                }
            }
        }
    }
    assert_eq!(hidden.len(), 4, "three card sets and the CPU twin of mc1");
    assert_eq!(chat, 3);
    let (card_pos, card, (at, len)) = &hidden["ref_minicpmv47_vis_hidden_mc1"];
    let (twin_pos, twin, _) = &hidden["ref_minicpmv47_vis_hidden_mc1.cpu"];
    assert_eq!(
        card_pos, twin_pos,
        "the positions fed are the same on the card and the CPU"
    );
    // the distance of a position: ‖twin − card‖ / ‖card‖, by region as the Qwen seats report theirs
    let dist = |p: usize| {
        let (a, b) = (
            &twin[p * N_EMBD..(p + 1) * N_EMBD],
            &card[p * N_EMBD..(p + 1) * N_EMBD],
        );
        let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
        (num / den).sqrt()
    };
    let quantile = |mut v: Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v[(q * (v.len() - 1) as f64).round() as usize]
    };
    for (what, rows) in [
        ("before the image", 0..*at),
        ("the image", *at..at + len),
        ("after the image", at + len..twin.len() / N_EMBD),
    ] {
        let d: Vec<f64> = rows.map(dist).collect();
        let (med, p90) = (quantile(d.clone(), 0.5), quantile(d, 0.9));
        println!(
            "hw_mcprompt: mc1 CPU twin against the card, {what}: median {med:.4} p90 {p90:.4}"
        );
        assert!(
            med < TWIN_MEDIAN,
            "{what}: the twin is {med:.4} from the card"
        );
    }
}

/// A bug catcher, not a precision claim: a wrong row or position moves a position's `result_norm` by order 1, the CPU
/// twin sits a few hundredths to a fifth from the card (the Qwen seats' twin distances).
const TWIN_MEDIAN: f64 = 0.5;
