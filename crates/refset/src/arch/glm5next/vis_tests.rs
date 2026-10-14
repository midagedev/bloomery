//! The GLM-5.3-Flash image-input families: a set of each of the five kinds the fixture tier's dump writes, read by
//! name, and each way a set is refused — its identity (the projector's digest, the fixture generation, the respelled
//! copy and its record), its layout against the family's, and rows that disagree with the lines stating them. The
//! reader's layout switches are the set's own lines, so the Qwen seats' sets (`arch::qwen_vis_tests`) stay what they were.

use super::vis::{
    self, FIXTURE_SHA256, MAINLINE_ARCH, MMPROJ_BF16, MMPROJ_BF16_SHA256, RESPELL_SET, TEXT_MODEL,
};
use crate::RefError;
use crate::clefvis::testkit::{
    CHAT, Head, IMAGE_COLUMNS, finish, put, put_ints, put_ints_shaped, refusal, scratch, write,
};
use crate::clefvis::{
    AFTER_FC_PROJ, AFTER_FFN_PROJ, CLEF, ClefvisSet, EMBD, Kind, POST_LN, PositionKind, Profile,
    ProjectorKind, QVIS_LCPP_BUILD, TOWER_ARCH, TapScope, VIT_OUT,
};
use crate::family::{Family, Identity};
use gguf::Value;
use gguf::write::{Layout, Writer};
use std::path::{Path, PathBuf};

/// A fixture root holding the original fixture's first shard: a header of one seed, the keys `fixture generate` writes.
struct Original {
    root: PathBuf,
    first: String,
}

impl Original {
    fn new(tag: &str, seed: u64) -> Original {
        let root = scratch(tag);
        let dir = root.join("glm5next");
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let path = dir.join("f-00001-of-00001.gguf");
        let kvs = vec![
            ("bloomery.fixture.version".to_string(), Value::U32(1)),
            ("bloomery.fixture.seed".to_string(), Value::U64(seed)),
        ];
        let layout =
            Layout::new(&kvs, Vec::new()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let file =
            std::fs::File::create(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Writer::new(file, layout)
            .and_then(Writer::finish)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Original {
            root,
            first: path.display().to_string(),
        }
    }

    /// The `# fixture` line the dumpers write for this file.
    fn line(&self) -> String {
        crate::fixture::line(Path::new(&self.first)).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `family` reading the set at `dir`, the tree's fixture root this one.
    fn open(&self, dir: &Path, family: &Family) -> Result<ClefvisSet, RefError> {
        crate::fixture::with_root(&self.root, || ClefvisSet::open(dir, family))
    }
}

/// The seat's family of `kind`.
fn family(kind: Kind) -> &'static Family {
    crate::arch::all()
        .find(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
        .unwrap_or_else(|| panic!("{kind:?}: no family of the glm5v seat in the table"))
}

/// Qwen3.6's family of `kind` and its profile, from the table: a Qwen-layout seat to read this seat's sets against.
fn qwen(kind: Kind) -> (&'static Family, &'static Profile) {
    let name = format!("qwen35moe-vis-{}", kind.as_str());
    let f = crate::arch::named(&name).unwrap_or_else(|| panic!("no {name} family"));
    match f.identity {
        Identity::Clefvis(_, p) => (f, p),
        other => panic!("{name} is {other:?}"),
    }
}

/// A Qwen3.6 set's head: the family's own file (`model`, what `Family::runs` says) and architecture, the profile's projector.
fn qwen_head(kind: Kind, model: &str) -> Head<'_> {
    let (f, p) = qwen(kind);
    Head {
        kind: kind.as_str(),
        model,
        build: QVIS_LCPP_BUILD,
        arch: f.arch.unwrap_or_else(|| panic!("{} names no arch", f.name)),
        mmproj: p.mmproj,
        mmproj_sha: p.mmproj_sha256,
        complete: true,
    }
}

fn tower(kind: Kind) -> Head<'static> {
    Head {
        kind: kind.as_str(),
        model: MMPROJ_BF16,
        build: QVIS_LCPP_BUILD,
        arch: TOWER_ARCH,
        mmproj: MMPROJ_BF16,
        mmproj_sha: MMPROJ_BF16_SHA256,
        complete: true,
    }
}

fn text(kind: Kind) -> Head<'static> {
    Head {
        model: TEXT_MODEL,
        arch: MAINLINE_ARCH,
        ..tower(kind)
    }
}

const PROJECTOR: &str = "# projector\tglm5v\tpatch_size 14\tspatial_merge_size 2\talign 28\timage_min_pixels 12544\timage_max_pixels 6272000";

/// A 50x50 image planned 56x56: a 2x2 grid of 4 tokens of 28 pixels a side, advancing the decoder by its 4 tokens.
const IMAGE: &str = "# image\tg\taa\tbb\t50\t50\t56\t56\t2\t2\t4\t4";

/// The identity lines every set of the seat states beyond a Qwen seat's.
fn identity(o: &Original) -> Vec<String> {
    vec![
        format!("# text_model\t{TEXT_MODEL}"),
        o.line(),
        format!(
            "# respell\tsource {}\tsource_sha256 {FIXTURE_SHA256}\tfrom glm5next\tset {RESPELL_SET}",
            o.first
        ),
        PROJECTOR.to_string(),
    ]
}

fn with(mut extra: Vec<String>, lines: &[&str]) -> Vec<String> {
    extra.extend(lines.iter().map(|l| (*l).to_string()));
    extra
}

/// Set A: the 56x56 canvas of the one image.
fn preproc(dir: &Path, o: &Original) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "g/inp_raw", [56, 56, 3, 1], 0.5);
    let extra = with(identity(o), &[IMAGE_COLUMNS, IMAGE]);
    finish(dir, &tower(Kind::Preproc), &extra, &rows);
}

/// What a set B of the one image holds: the scope its `# tap_effect` states, the rows of `vit_out` and the columns of
/// the merged nodes.
struct Taps {
    scope: &'static str,
    vit_rows: u64,
    merged: u64,
}

/// A set B as the dump writes it: 4 tokens, so 16 patches of `vit_out`.
const FINAL: Taps = Taps {
    scope: "final",
    vit_rows: 16,
    merged: 4,
};

fn taps(dir: &Path, o: &Original, t: &Taps) {
    let mut rows = Vec::new();
    put(
        dir,
        &mut rows,
        &format!("g/{VIT_OUT}"),
        [1024, t.vit_rows, 1, 1],
        1.0,
    );
    for node in [AFTER_FC_PROJ, AFTER_FFN_PROJ] {
        put(
            dir,
            &mut rows,
            &format!("g/{node}"),
            [4096, t.merged, 1, 1],
            1.0,
        );
    }
    put(dir, &mut rows, &format!("g/{EMBD}"), [4096, 4, 1, 1], 2.0);
    let effect = format!(
        "# tap_effect\tg\tembd values 16384\tdiffering 0\tmax_abs_diff 0\ttaps {}",
        t.scope
    );
    let extra = with(identity(o), &[IMAGE_COLUMNS, IMAGE, &effect]);
    finish(dir, &tower(Kind::Taps), &extra, &rows);
}

/// The header lines of the prompt of 8 ids: two text ids, the image's 4 tokens, two text ids.
fn prompt_lines(pad: u32) -> Vec<String> {
    vec![
        "# tokens_count\t8".to_string(),
        format!("# image_pad_id\t{pad}"),
        "# span columns\tindex image at len nx ny n_pos start_pos end_pos".to_string(),
        "# span\t0\tg\t2\t4\t2\t2\t4\t2\t6".to_string(),
        IMAGE_COLUMNS.to_string(),
        IMAGE.to_string(),
    ]
}

/// The rows that hold a prompt's positions.
enum Pos {
    /// `pos` `[1, 8]`.
    Normal,
    /// `mrope_pos` `[3, 8]`, a Qwen seat's row.
    Mrope,
    /// `pos` of the wrong shape, `[3, 8]`.
    Wide,
}

/// Set C of the 8-id prompt, its `# positions` line as given.
fn hidden(dir: &Path, o: &Original, pos: &Pos, positions: Option<&str>) {
    let mut rows = Vec::new();
    put(dir, &mut rows, "result_norm", [4096, 8, 1, 1], 1.5);
    let ids: Vec<i32> = (0..8).collect();
    let triples: Vec<i32> = ids.iter().flat_map(|&p| [p, p, p]).collect();
    match pos {
        Pos::Normal => put_ints_shaped(dir, &mut rows, "pos", [1, 8], &ids),
        Pos::Mrope => put_ints_shaped(dir, &mut rows, "mrope_pos", [3, 8], &triples),
        Pos::Wide => put_ints_shaped(dir, &mut rows, "pos", [3, 8], &triples),
    }
    let mut extra = identity(o);
    extra.extend(prompt_lines(vis::PROFILE.image_pad_id));
    extra.extend(positions.map(|p| format!("# positions\t{p}")));
    finish(dir, &text(Kind::Hidden), &extra, &rows);
}

/// The ids of the 8-id prompt.
fn ids() -> Vec<i32> {
    let pad = vis::PROFILE.image_pad_id as i32;
    vec![11, 12, pad, pad, pad, pad, 13, 14]
}

/// Set E of the 8-id prompt.
fn chat_ids(dir: &Path, o: &Original) {
    let mut rows = Vec::new();
    put_ints(dir, &mut rows, "ids", &ids());
    let mut extra = identity(o);
    extra.extend(prompt_lines(vis::PROFILE.image_pad_id));
    extra.extend([
        CHAT.to_string(),
        "# prompt\tq".to_string(),
        "# prompt_sha256\t00".to_string(),
    ]);
    finish(dir, &text(Kind::ChatIds), &extra, &rows);
}

/// Set F: three decode steps after the 8-id prompt, the first token at position 8.
fn decode(dir: &Path, o: &Original) {
    let mut rows = Vec::new();
    put_ints(dir, &mut rows, "ids", &[7, 7, 7]);
    put(dir, &mut rows, "result_norm", [4096, 3, 1, 1], 1.5);
    put_ints(dir, &mut rows, "n_past", &[9, 10, 11]);
    let mut extra = identity(o);
    extra.extend(prompt_lines(vis::PROFILE.image_pad_id));
    extra.push("# decode\tsteps\t3\tn_past_start\t8\tsampler\tgreedy".to_string());
    finish(dir, &text(Kind::Decode), &extra, &rows);
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

/// The set without its lines that open with `prefix`.
fn drop_lines(dir: &Path, prefix: &str) {
    edit(dir, |t| {
        t.lines()
            .filter(|l| !l.starts_with(prefix))
            .map(|l| format!("{l}\n"))
            .collect()
    });
}

fn ok(r: Result<ClefvisSet, RefError>) -> ClefvisSet {
    r.unwrap_or_else(|e| panic!("{e}"))
}

/// A set of each of the five kinds reads by name: the grid of 28-pixel tokens, the `pos` row of normal positions (and
/// the M-RoPE accessor refusing it), the tower's three output-end nodes, the `<|image|>` token, the decode's first position.
#[test]
fn a_glm5v_set_of_each_kind_reads_by_name() {
    let o = Original::new("glm-reads", 1);
    let dir = scratch("glm-reads-set");

    preproc(&dir, &o);
    let set = ok(o.open(&dir, family(Kind::Preproc)));
    assert_eq!(set.projector.kind, ProjectorKind::Glm5v);
    assert_eq!(set.projector.align, 28);
    assert_eq!(set.projector.image_max_pixels, Some(6_272_000));
    let image = set.image("g").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (image.best_w, image.nx, image.n_tokens, image.n_pos),
        (56, 2, 4, 4)
    );
    assert_eq!(
        set.inp_raw("g").unwrap_or_else(|e| panic!("{e}")).len(),
        56 * 56 * 3
    );

    taps(&dir, &o, &FINAL);
    let set = ok(o.open(&dir, family(Kind::Taps)));
    for (node, want) in [
        (VIT_OUT, [1024, 16]),
        (AFTER_FC_PROJ, [4096, 4]),
        (AFTER_FFN_PROJ, [4096, 4]),
        (EMBD, [4096, 4]),
    ] {
        let (row, v) = set.tap("g", node).unwrap_or_else(|e| panic!("{node}: {e}"));
        assert_eq!(row.ne[..2], want, "{node}");
        assert_eq!(v.len() as u64, want[0] * want[1], "{node}");
    }
    let e = set.tap("g", POST_LN).expect_err("a Qwen node").to_string();
    assert!(
        e.contains("carries vit_out, after_fc_proj, after_ffn_proj and embd only, not norm_b-27"),
        "{e}"
    );

    hidden(&dir, &o, &Pos::Normal, Some("normal"));
    let set = ok(o.open(&dir, family(Kind::Hidden)));
    assert_eq!(set.projector.kind.positions(), PositionKind::Normal);
    assert_eq!(
        set.normal_positions().unwrap_or_else(|e| panic!("{e}")),
        (0..8).collect::<Vec<i32>>()
    );
    let e = set.positions().expect_err("no M-RoPE triples").to_string();
    assert!(
        e.contains(
            "mrope_pos reads a set of mrope positions, this glm5v set feeds normal positions"
        ),
        "{e}"
    );
    assert_eq!(set.image_pad_id, Some(154_854));
    assert_eq!(
        set.result_norm().unwrap_or_else(|e| panic!("{e}")).len(),
        4096 * 8
    );
    let span = &set.spans[0];
    assert_eq!((span.at, span.len, span.n_pos), (2, 4, 4));
    assert_eq!(
        (span.start_pos, span.end_pos),
        (2, 6),
        "an image saves no positions"
    );

    chat_ids(&dir, &o);
    let set = ok(o.open(&dir, family(Kind::ChatIds)));
    let want: Vec<u32> = ids().iter().map(|&i| i as u32).collect();
    assert_eq!(set.ids().unwrap_or_else(|e| panic!("{e}")), want);

    decode(&dir, &o);
    let set = ok(o.open(&dir, family(Kind::Decode)));
    let line = set.decode().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((line.steps, line.n_past_start), (3, 8));
    assert_eq!(
        set.decode_n_past().unwrap_or_else(|e| panic!("{e}")),
        [9, 10, 11]
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The table holds one family of each kind the seat writes, none of the Qwen seats' controls (prose, bf16 rows), each
/// stating the file its kind reads: the projector for the tower's sets, the respelled copy for the prompts'. The
/// names are the seat's own (never `fx-…`, the mirrors of the node dumps'), the sets are `fx_ref_glm5next_vis_<kind>…`
/// and share no name with another family, and the architecture's node dumps stay ik's.
#[test]
fn the_table_holds_one_family_of_each_kind_the_glm5v_seat_writes() {
    let mut names = std::collections::HashSet::new();
    for f in crate::arch::all() {
        for s in f.sets {
            if !matches!(f.identity, Identity::Clefvis(_, p) if std::ptr::eq(p, &vis::PROFILE)) {
                assert!(names.insert(*s), "set name {s} twice in the table");
            }
        }
    }
    for kind in [
        Kind::Preproc,
        Kind::Taps,
        Kind::Hidden,
        Kind::ChatIds,
        Kind::Decode,
    ] {
        let of_kind = crate::arch::all()
            .filter(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
            .count();
        assert_eq!(of_kind, 1, "{kind:?}: {of_kind} families");
        let f = family(kind);
        assert!(
            f.name == format!("glm5next-vis-{}", kind.as_str()),
            "{}",
            f.name
        );
        let stem = format!("fx_ref_glm5next_vis_{}", kind.as_str());
        assert!(
            f.sets.iter().all(|s| s.starts_with(&stem)),
            "{}: {:?}",
            f.name,
            f.sets
        );
        for s in f.sets {
            assert!(names.insert(*s), "set name {s} twice in the table");
        }
        let tower = matches!(kind, Kind::Preproc | Kind::Taps);
        assert_eq!(
            f.arch,
            Some(if tower { TOWER_ARCH } else { MAINLINE_ARCH }),
            "{}",
            f.name
        );
        assert_eq!(
            f.runs().unwrap_or_else(|e| panic!("{e}")),
            if tower { MMPROJ_BF16 } else { TEXT_MODEL }
        );
        assert!(
            super::FAMILIES.iter().any(|g| std::ptr::eq(*g, f)),
            "{} is not in the architecture's list",
            f.name
        );
    }
    for kind in [Kind::Prose, Kind::Bf16Rows] {
        let n = crate::arch::all()
            .filter(|f| f.identity == Identity::Clefvis(kind, &vis::PROFILE))
            .count();
        assert_eq!(n, 0, "the seat writes no {kind:?} sets");
    }
    assert_eq!(
        crate::arch::node_dumps(super::ARCH).map(|f| f.name),
        Some("ik-glm5next")
    );
}

/// A projector that is not the seat's file is stale by its digest, naming both.
#[test]
fn a_set_made_with_another_projector_is_stale() {
    let o = Original::new("glm-mmproj", 1);
    let dir = scratch("glm-mmproj-set");
    taps(&dir, &o, &FINAL);
    let bad = "0".repeat(64);
    replace(&dir, MMPROJ_BF16_SHA256, &bad);
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains(&format!("sha256 {bad}")) && e.contains(&format!("sha256 {MMPROJ_BF16_SHA256}")),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// A set dumped on another generation of the fixture, or one that states none, is stale against the original's header.
#[test]
fn a_set_of_another_fixture_generation_is_stale() {
    let (one, two) = (Original::new("glm-gen1", 1), Original::new("glm-gen2", 2));
    let dir = scratch("glm-gen-set");
    taps(&dir, &one, &FINAL);
    ok(one.open(&dir, family(Kind::Taps)));
    let e = refusal(two.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("bloomery.fixture.seed=1") && e.contains("bloomery.fixture.seed=2"),
        "{e}"
    );
    drop_lines(&dir, "# fixture\t");
    let e = refusal(one.open(&dir, family(Kind::Taps)));
    assert!(e.contains("an unstated fixture"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
    for o in [&one, &two] {
        let _ = std::fs::remove_dir_all(&o.root);
    }
}

/// The text model of every set is the respelled copy: the original fixture's path, another file, or none is stale. A
/// tower set states it beside its projector, which is its model.
#[test]
fn a_set_whose_text_model_is_not_the_copy_is_stale() {
    let o = Original::new("glm-copy", 1);
    let dir = scratch("glm-copy-set");
    taps(&dir, &o, &FINAL);
    replace(
        &dir,
        &format!("# text_model\t{TEXT_MODEL}"),
        &format!("# text_model\t{}", o.first),
    );
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains(&format!("dumped from {}", o.first))
            && e.contains(&format!("the tree runs {TEXT_MODEL}")),
        "{e}"
    );
    drop_lines(&dir, "# text_model\t");
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(e.contains("no # text_model line"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// A set without the copy's `# respell` record is stale: nothing says which copy it was dumped from.
#[test]
fn a_set_without_its_respell_line_is_stale() {
    let o = Original::new("glm-norespell", 1);
    let dir = scratch("glm-norespell-set");
    for kind in [Kind::Taps, Kind::Hidden] {
        match kind {
            Kind::Taps => taps(&dir, &o, &FINAL),
            _ => hidden(&dir, &o, &Pos::Normal, Some("normal")),
        }
        ok(o.open(&dir, family(kind)));
        drop_lines(&dir, "# respell\t");
        let e = refusal(o.open(&dir, family(kind)));
        assert!(
            e.contains("an unstated respell (the set has no # respell line)")
                && e.contains(RESPELL_SET),
            "{kind:?}: {e}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// Each field of the record is the profile's: the set the copy changed (another, or none), the original's digest, the
/// architecture it was respelled from, the original's path; a record of another width is malformed.
#[test]
fn every_field_of_the_respell_record_is_held() {
    let o = Original::new("glm-fields", 1);
    let dir = scratch("glm-fields-set");
    let other = "f".repeat(64);
    let cases: [(String, String, &str); 6] = [
        (
            format!("\tset {RESPELL_SET}"),
            "\tset glm5-next.expert_shared_count=1->2".to_string(),
            "respell set glm5-next.expert_shared_count=1->2",
        ),
        (
            format!("\tset {RESPELL_SET}"),
            "\tset -".to_string(),
            "respell set -",
        ),
        (
            format!("source_sha256 {FIXTURE_SHA256}"),
            format!("source_sha256 {other}"),
            "respell source_sha256",
        ),
        (
            "\tfrom glm5next\t".to_string(),
            "\tfrom glm5-next\t".to_string(),
            "respell from glm5-next",
        ),
        (
            format!("source {}", o.first),
            "source /models/fixtures-old/glm5next/f-00001-of-00001.gguf".to_string(),
            "respell source /models/fixtures-old",
        ),
        (
            format!("\tset {RESPELL_SET}"),
            String::new(),
            "want source <path>\tsource_sha256 <hex>\tfrom <arch>\tset <list or ->",
        ),
    ];
    for (from, to, want) in cases {
        taps(&dir, &o, &FINAL);
        replace(&dir, &from, &to);
        let e = refusal(o.open(&dir, family(Kind::Taps)));
        assert!(e.contains(want), "{to:?}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The image rows' positions are the row the layout names: a `pos` row that is not `[1, n]`, the other layout's
/// `mrope_pos` in its place, a prompt set without its `# positions` line, a line that is not the layout's and one the
/// reader does not know are each refused by name; and a Qwen-layout set holding `pos` is refused the same way.
#[test]
fn positions_that_disagree_with_their_line_are_refused() {
    let o = Original::new("glm-pos", 1);
    let dir = scratch("glm-pos-set");
    let f = family(Kind::Hidden);
    hidden(&dir, &o, &Pos::Mrope, Some("normal"));
    let e = refusal(o.open(&dir, f));
    assert!(
        e.contains(
            "the glm5v layout feeds normal positions, the pos row; the set holds mrope_pos instead"
        ),
        "{e}"
    );
    hidden(&dir, &o, &Pos::Wide, Some("normal"));
    let e = refusal(o.open(&dir, f));
    assert!(e.contains("pos is [3, 8, 1, 1], want [1, 8, 1, 1]"), "{e}");
    hidden(&dir, &o, &Pos::Normal, None);
    let e = refusal(o.open(&dir, f));
    assert!(
        e.contains("a glm5v prompt set states `# positions normal`, this one states none"),
        "{e}"
    );
    hidden(&dir, &o, &Pos::Normal, Some("mrope"));
    let e = refusal(o.open(&dir, f));
    assert!(
        e.contains("`# positions mrope`, the glm5v layout feeds normal positions"),
        "{e}"
    );
    hidden(&dir, &o, &Pos::Normal, Some("causal"));
    let e = refusal(o.open(&dir, f));
    assert!(e.contains("want normal or mrope"), "{e}");

    // a Qwen seat's prompt set whose positions row is the other layout's
    let (qf, q) = qwen(Kind::Hidden);
    let model = qf.runs().unwrap_or_else(|e| panic!("{e}"));
    crate::clefvis::testkit::prompt(
        &dir,
        &qwen_head(Kind::Hidden, &model),
        true,
        q.image_pad_id,
        q,
    );
    ok(ClefvisSet::open(&dir, qf));
    replace(&dir, "tensor\tmrope_pos\t", "tensor\tpos\t");
    let e = refusal(ClefvisSet::open(&dir, qf));
    assert!(
        e.contains("the qwen3vl_merger layout feeds mrope positions, the mrope_pos row; the set holds pos instead"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// A set of the other layout is refused by name where the model and the projector match: the seat's identity with no
/// `# projector` line (a Qwen set's shape), and a Qwen seat's identity with the `glm5v` line. A Qwen seat's own set read as
/// the GLM seat's, and the reverse, is refused by its model file first.
#[test]
fn a_set_of_the_other_layout_is_refused_by_name() {
    let o = Original::new("glm-layout", 1);
    let dir = scratch("glm-layout-set");
    taps(&dir, &o, &FINAL);
    drop_lines(&dir, "# projector\t");
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("projector \"(no # projector line: the qwen3vl_merger layout)\"")
            && e.contains("the glm5next-vis-taps family wants glm5v"),
        "{e}"
    );

    let (qf, q) = qwen(Kind::Taps);
    let model = qf.runs().unwrap_or_else(|e| panic!("{e}"));
    crate::clefvis::testkit::taps(&dir, &qwen_head(Kind::Taps, &model), TapScope::Final, q);
    ok(ClefvisSet::open(&dir, qf));
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(e.contains("dumped from") && e.contains(q.mmproj), "{e}");
    replace(
        &dir,
        "# clefvis\ttaps\n",
        &format!("# clefvis\ttaps\n{PROJECTOR}\n"),
    );
    let e = refusal(ClefvisSet::open(&dir, qf));
    assert!(
        e.contains("projector \"glm5v\"")
            && e.contains("the qwen35moe-vis-taps family wants qwen3vl_merger"),
        "{e}"
    );
    taps(&dir, &o, &FINAL);
    let e = refusal(ClefvisSet::open(&dir, qf));
    assert!(e.contains("dumped from") && e.contains(MMPROJ_BF16), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The `# projector` line is read as stated: a layout the reader does not know, a grid unit that is not the patch size
/// times the merge size, and a line of another width are refused by name.
#[test]
fn a_projector_line_the_reader_does_not_know_is_refused() {
    let o = Original::new("glm-proj", 1);
    let dir = scratch("glm-proj-set");
    for (from, to, want) in [
        (
            "# projector\tglm5v",
            "# projector\tgemma3",
            "\"gemma3\" is neither qwen3vl_merger nor glm5v",
        ),
        (
            "\talign 28\t",
            "\talign 32\t",
            "align 32 is not patch_size 14 x spatial_merge_size 2",
        ),
        ("\timage_max_pixels 6272000", "", "want <layout>"),
        (
            "\timage_min_pixels 12544",
            "\timage_min_pixels many",
            "image_min_pixels is not a number",
        ),
    ] {
        taps(&dir, &o, &FINAL);
        replace(&dir, from, to);
        let e = refusal(o.open(&dir, family(Kind::Taps)));
        assert!(e.contains(want), "{to:?}: {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The grid is the layout's: a token is 28 pixels a side (a canvas of 32-pixel tokens, the Qwen unit, is refused on either
/// side alone), and an image advances the decoder by its tokens (the Qwen `max(nx, ny)` is refused).
#[test]
fn the_grid_follows_the_projector() {
    let o = Original::new("glm-grid", 1);
    let dir = scratch("glm-grid-set");
    for (to, want) in [
        (
            "\t56\t56\t2\t2\t4\t2",
            "image g: grid 2x2 of 4 tokens, planned 56x56, n_pos 2",
        ),
        (
            "\t64\t64\t2\t2\t4\t4",
            "image g: grid 2x2 of 4 tokens, planned 64x64, n_pos 4",
        ),
        (
            "\t64\t56\t2\t2\t4\t4",
            "image g: grid 2x2 of 4 tokens, planned 64x56, n_pos 4",
        ),
        (
            "\t56\t64\t2\t2\t4\t4",
            "image g: grid 2x2 of 4 tokens, planned 56x64, n_pos 4",
        ),
    ] {
        preproc(&dir, &o);
        replace(&dir, "\t56\t56\t2\t2\t4\t4", to);
        let e = refusal(o.open(&dir, family(Kind::Preproc)));
        assert!(e.contains(want), "{e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The image token is the seat's: another vocabulary's id is refused, naming `<|image|>`.
#[test]
fn the_image_token_is_the_seats() {
    let o = Original::new("glm-pad", 1);
    let dir = scratch("glm-pad-set");
    hidden(&dir, &o, &Pos::Normal, Some("normal"));
    replace(&dir, "# image_pad_id\t154854", "# image_pad_id\t248056");
    let e = refusal(o.open(&dir, family(Kind::Hidden)));
    assert!(
        e.contains("image_pad_id [\"248056\"], the vocabulary's <|image|> is 154854"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The tower is tapped at its output end: any other scope is refused by name, and the nodes keep the shapes of the
/// layout — `vit_out` four patches a token, the merged nodes a row a token as wide as the text model.
#[test]
fn a_glm5v_tower_is_tapped_at_its_output_end_with_its_shapes() {
    let o = Original::new("glm-taps", 1);
    let dir = scratch("glm-taps-set");
    for scope in ["full", "inp_raw only"] {
        taps(&dir, &o, &Taps { scope, ..FINAL });
        let e = refusal(o.open(&dir, family(Kind::Taps)));
        assert!(
            e.contains(&format!(
                "g: taps {scope}, a glm5v tower is tapped at its output end only (vit_out, after_fc_proj, after_ffn_proj)"
            )),
            "{e}"
        );
    }
    taps(
        &dir,
        &o,
        &Taps {
            vit_rows: 15,
            ..FINAL
        },
    );
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("g/vit_out is [1024, 15, 1, 1], want [width, 16, 1, 1]"),
        "{e}"
    );
    taps(&dir, &o, &Taps { merged: 3, ..FINAL });
    let e = refusal(o.open(&dir, family(Kind::Taps)));
    assert!(
        e.contains("g/after_fc_proj is [4096, 3, 1, 1], want [4096, 4, 1, 1]"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&o.root);
}

/// The seat's profile is the one the families hold: its projector layout and its record.
#[test]
fn the_profile_names_the_layout_and_the_record() {
    let p: &Profile = &vis::PROFILE;
    assert_eq!(p.projector, ProjectorKind::Glm5v);
    let r = p
        .respell
        .as_ref()
        .unwrap_or_else(|| panic!("no respell record"));
    assert_eq!(
        (r.from, r.copy, r.set),
        ("glm5next", TEXT_MODEL, RESPELL_SET)
    );
    assert_ne!(p, &CLEF);
    assert_ne!(p, qwen(Kind::Taps).1);
}
