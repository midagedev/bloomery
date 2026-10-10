//! The clefvis reader on the Qwen3.6 and Qwen3.8 seats: the kinds the Clef-Flash tests do not hold (chat ids, decode
//! steps, a final-tapped tower), the profile each family checks a set against, and each refusal by name.

use super::{qwen4exp, qwen35, qwen35moe};
use crate::RefError;
use crate::clefvis::testkit::{
    CHAT, DECODE_START, Head, chat_ids, chat_ids_of, decode, prompt, scratch, taps, write,
};
use crate::clefvis::{
    CLEF, ClefvisSet, EMBD, Kind, MMPROJ_BF16, POST_LN, Profile, QVIS_LCPP_BUILD, TOWER_ARCH,
    TapScope,
};
use crate::family::{Family, Identity};

/// One seat: its profile, its text model file and architecture.
struct Seat {
    profile: &'static Profile,
    model: &'static str,
    arch: &'static str,
}

fn q36() -> Seat {
    Seat {
        profile: &qwen35moe::vis::PROFILE,
        model: qwen35moe::MODEL,
        arch: qwen35moe::ARCH,
    }
}

fn q38() -> Seat {
    Seat {
        profile: &qwen4exp::vis::PROFILE,
        model: qwen4exp::MODEL,
        arch: qwen4exp::ARCH,
    }
}

impl Seat {
    /// The seat's family of `kind` in the table.
    fn family(&self, kind: Kind) -> &'static Family {
        crate::arch::all()
            .find(|f| f.identity == Identity::Clefvis(kind, self.profile))
            .unwrap_or_else(|| panic!("{kind:?}: no family of the seat in the table"))
    }

    /// A tower set's header: the projector is the model file.
    fn tower(&self, kind: &'static str) -> Head<'static> {
        Head {
            kind,
            model: self.profile.mmproj,
            build: QVIS_LCPP_BUILD,
            arch: TOWER_ARCH,
            mmproj: self.profile.mmproj,
            mmproj_sha: self.profile.mmproj_sha256,
            complete: true,
        }
    }

    /// A text-model set's header.
    fn text(&self, kind: &'static str) -> Head<'static> {
        Head {
            kind,
            model: self.model,
            build: QVIS_LCPP_BUILD,
            arch: self.arch,
            mmproj: self.profile.mmproj,
            mmproj_sha: self.profile.mmproj_sha256,
            complete: true,
        }
    }

    fn pad(&self) -> u32 {
        self.profile.image_pad_id
    }
}

fn refusal(r: Result<ClefvisSet, RefError>) -> String {
    match r {
        Ok(_) => panic!("the set was taken"),
        Err(e) => e.to_string(),
    }
}

const PROMPT: &str = "# prompt\t<|im_start|>user\\n<__media__>\\tq<|im_end|>\\n\\\\";

/// A chat-ids set reads by name on each seat: the ids, the prompt with its escapes undone, the chat line.
#[test]
fn a_chat_ids_set_reads_by_name_on_each_seat() {
    for seat in [q36(), q38()] {
        let dir = scratch("chatids");
        let ids: Vec<u32> = chat_ids_of(seat.pad()).iter().map(|&i| i as u32).collect();
        chat_ids(
            &dir,
            &seat.text("chatids"),
            &chat_ids_of(seat.pad()),
            CHAT,
            PROMPT,
            seat.pad(),
        );
        let set =
            ClefvisSet::open(&dir, seat.family(Kind::ChatIds)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(set.ids().unwrap_or_else(|e| panic!("{e}")), ids);
        assert_eq!(
            set.prompt().unwrap_or_else(|e| panic!("{e}")),
            "<|im_start|>user\n<__media__>\tq<|im_end|>\n\\"
        );
        let chat = set.chat().unwrap_or_else(|e| panic!("{e}"));
        assert!(chat.enable_thinking && chat.add_generation_prompt && !chat.continue_final_message);
        assert_eq!(set.n_ids().unwrap_or_else(|e| panic!("{e}")), 16);
        assert_eq!(set.spans.len(), 1);
        assert!(
            set.result_norm().is_err(),
            "a chat-ids set holds no hidden rows"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Each way a chat-ids set can disagree with itself is refused naming what differs.
#[test]
fn a_chat_ids_set_whose_ids_disagree_with_its_header_is_malformed() {
    let seat = q36();
    let dir = scratch("chatids-bad");
    let family = seat.family(Kind::ChatIds);
    let head = seat.text("chatids");
    let good = chat_ids_of(seat.pad());
    // an image-pad id outside every span
    let mut ids = good.clone();
    ids[0] = seat.pad() as i32;
    chat_ids(&dir, &head, &ids, CHAT, PROMPT, seat.pad());
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(
        e.contains("ids[0]") && e.contains("fills the spans and nothing else"),
        "{e}"
    );
    // a span position that is not an image-pad id
    let mut ids = good.clone();
    ids[5] = 5;
    chat_ids(&dir, &head, &ids, CHAT, PROMPT, seat.pad());
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(e.contains("ids[5]"), "{e}");
    // a header that counts one id more than the row holds
    chat_ids(&dir, &head, &good, CHAT, PROMPT, seat.pad());
    let text = std::fs::read_to_string(dir.join("MANIFEST.tsv")).unwrap_or_else(|e| panic!("{e}"));
    write(
        &dir.join("MANIFEST.tsv"),
        text.replace("# tokens_count\t16", "# tokens_count\t17")
            .as_bytes(),
    );
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(
        e.contains("ids is [16, 1, 1, 1], want [17, 1, 1, 1]"),
        "{e}"
    );
    // a request that both opens the next turn and continues the last
    chat_ids(
        &dir,
        &head,
        &good,
        &CHAT.replace("continue_final_message none", "continue_final_message auto"),
        PROMPT,
        seat.pad(),
    );
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(e.contains("never both or neither"), "{e}");
    // an escape the dump does not write
    chat_ids(&dir, &head, &good, CHAT, "# prompt\ta\\qb", seat.pad());
    let set = ClefvisSet::open(&dir, family).unwrap_or_else(|e| panic!("{e}"));
    let e = set.prompt().expect_err("a bad escape").to_string();
    assert!(e.contains("the escape \\q"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A decode set reads by name on each seat, and its first position is the prompt's ids less what the image saved.
#[test]
fn a_decode_set_reads_by_name_on_each_seat() {
    for seat in [q36(), q38()] {
        let dir = scratch("decode");
        let n_past: Vec<i32> = (1..=3).map(|k| DECODE_START as i32 + k).collect();
        decode(
            &dir,
            &seat.text("decode"),
            DECODE_START,
            &n_past,
            seat.profile,
            seat.pad(),
        );
        let set =
            ClefvisSet::open(&dir, seat.family(Kind::Decode)).unwrap_or_else(|e| panic!("{e}"));
        let line = set.decode().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((line.steps, line.n_past_start), (3, DECODE_START));
        assert_eq!(
            set.decode_n_past().unwrap_or_else(|e| panic!("{e}")),
            n_past
        );
        assert_eq!(
            set.decode_ids().unwrap_or_else(|e| panic!("{e}")),
            [7, 7, 7]
        );
        assert_eq!(
            set.decode_result_norm()
                .unwrap_or_else(|e| panic!("{e}"))
                .len(),
            seat.profile.n_embd * 3
        );
        assert!(set.ids().is_err(), "a decode set holds no chat ids");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A decode that does not advance the decoder by the images' positions, or whose steps are not consecutive, is refused.
#[test]
fn a_decode_set_whose_positions_disagree_is_malformed() {
    let seat = q36();
    let dir = scratch("decode-bad");
    let family = seat.family(Kind::Decode);
    let head = seat.text("decode");
    // the first token at the prompt's id count: the image's 12 tokens counted as 12 positions, not 4
    decode(&dir, &head, 16, &[17, 18], seat.profile, seat.pad());
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(
        e.contains("the first token sits at 16") && e.contains("give 8"),
        "{e}"
    );
    // a step that skips a position
    decode(
        &dir,
        &head,
        DECODE_START,
        &[9, 11],
        seat.profile,
        seat.pad(),
    );
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(
        e.contains("n_past[1] is 11, step 1 ends at position 10"),
        "{e}"
    );
    // a result_norm of the other seat's width
    decode(
        &dir,
        &head,
        DECODE_START,
        &[9, 10],
        q38().profile,
        seat.pad(),
    );
    let e = refusal(ClefvisSet::open(&dir, family));
    assert!(
        e.contains("result_norm is [2560, 2, 1, 1], want [2048, 2, 1, 1]"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tower tapped to its output end holds the post-LN node and the embeddings, and refuses every other node by name.
#[test]
fn a_final_tapped_tower_holds_the_post_ln_output_only() {
    for seat in [q36(), q38()] {
        let dir = scratch("final");
        taps(&dir, &seat.tower("taps"), TapScope::Final, seat.profile);
        let set = ClefvisSet::open(&dir, seat.family(Kind::Taps)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            set.tap_effect_of("g")
                .unwrap_or_else(|e| panic!("{e}"))
                .scope,
            TapScope::Final
        );
        let (row, v) = set.tap("g", POST_LN).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((row.ne[0], v.len()), (1152, 1152 * 48));
        let (row, _) = set.tap("g", EMBD).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(row.ne[0], seat.profile.n_embd as u64);
        let e = set.tap("g", "ln1-0").expect_err("final only").to_string();
        assert!(e.contains("carries norm_b-27 and embd only"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Each seat's families take its own sets and refuse the other seats' by name: the model file (`Stale`), the
/// projector (`Stale`), the mainline build (`Foreign`) and the text width of the rows.
#[test]
fn a_set_of_another_seat_is_refused_by_name() {
    let (a, b) = (q36(), q38());
    let dir = scratch("seats");
    // Qwen3.6's prompt set: its own family takes it, Qwen3.8's names the model file
    prompt(&dir, &a.text("hidden"), true, a.pad(), a.profile);
    assert!(ClefvisSet::open(&dir, a.family(Kind::Hidden)).is_ok());
    let e = refusal(ClefvisSet::open(&dir, b.family(Kind::Hidden)));
    assert!(e.contains("dumped from") && e.contains(a.model), "{e}");
    // Qwen3.8's own file with Qwen3.6's projector
    let mixed = Head {
        mmproj: a.profile.mmproj,
        mmproj_sha: a.profile.mmproj_sha256,
        ..b.text("hidden")
    };
    prompt(&dir, &mixed, true, b.pad(), b.profile);
    let e = refusal(ClefvisSet::open(&dir, b.family(Kind::Hidden)));
    assert!(e.contains("mmproj") && e.contains(a.profile.mmproj), "{e}");
    // Clef's mainline build
    let old = Head {
        build: qwen35::LCPP_BUILD,
        ..a.text("hidden")
    };
    prompt(&dir, &old, true, a.pad(), a.profile);
    let e = refusal(ClefvisSet::open(&dir, a.family(Kind::Hidden)));
    assert!(e.contains("build") && e.contains(QVIS_LCPP_BUILD), "{e}");
    // rows as wide as the other seat's text model
    prompt(&dir, &a.text("hidden"), true, a.pad(), b.profile);
    let e = refusal(ClefvisSet::open(&dir, a.family(Kind::Hidden)));
    assert!(
        e.contains("result_norm is [2560, 6, 1, 1], want [2048, 6, 1, 1]"),
        "{e}"
    );
    // a tower set of Qwen3.8, read as Qwen3.6's, and Clef's profile read as Qwen3.6's
    taps(&dir, &b.tower("taps"), TapScope::Final, b.profile);
    let e = refusal(ClefvisSet::open(&dir, a.family(Kind::Taps)));
    assert!(
        e.contains("dumped from") && e.contains(b.profile.mmproj),
        "{e}"
    );
    let clef = Head {
        model: MMPROJ_BF16,
        ..a.tower("taps")
    };
    taps(&dir, &clef, TapScope::Final, &CLEF);
    let e = refusal(ClefvisSet::open(&dir, a.family(Kind::Taps)));
    assert!(e.contains("dumped from"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The table holds one family of each kind for each Qwen seat, each of its kind's sets only, after the seat's node-dump
/// families, and a seat's sets never share a name with another seat's.
#[test]
fn the_table_holds_one_family_of_each_kind_a_qwen_seat() {
    let mut names = std::collections::HashSet::new();
    for seat in [q36(), q38()] {
        for kind in [
            Kind::Taps,
            Kind::Hidden,
            Kind::Prose,
            Kind::Bf16Rows,
            Kind::ChatIds,
            Kind::Decode,
        ] {
            let of_kind = crate::arch::all()
                .filter(|f| f.identity == Identity::Clefvis(kind, seat.profile))
                .count();
            assert_eq!(of_kind, 1, "{} {kind:?}: {of_kind} families", seat.arch);
            let f = seat.family(kind);
            let stem = format!("ref_{}_vis_{}", seat.arch, kind.as_str());
            assert!(
                f.sets.iter().all(|s| s.starts_with(&stem)),
                "{}: sets {:?}, want {stem}…",
                f.name,
                f.sets
            );
            for s in f.sets {
                assert!(names.insert(*s), "set name {s} twice in the table");
            }
        }
        assert_eq!(
            crate::arch::node_dumps(seat.arch).map(|f| f.name),
            Some(if seat.arch == qwen35moe::ARCH {
                "ik-qwen35moe"
            } else {
                "ik-qwen4exp"
            })
        );
    }
    // the seats' tower sets are the projector's: the profile is the only thing the Qwen seats share
    assert_ne!(q36().profile, q38().profile);
    assert_ne!(q36().profile, &CLEF);
}
