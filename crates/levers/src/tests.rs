use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use super::*;

/// The list of in-place reads `tools/check-levers.sh` holds the crates to.
const ALLOW_LIST: &str = include_str!("../../../tools/levers-direct.txt");

/// The rows as source text: `tools/check-levers.sh` reads the registry's
/// names from it.
const REGISTRY_SOURCE: &str = include_str!("registry.rs");

/// A regular file on every host the tests run on: this crate's manifest.
const A_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

/// The runners' one parser of `BLOOMERY_GATE_BOUND`.
const GATE_BOUND_PARSER: &str = include_str!("../../../tools/gate-bound.sh");

/// `pairs` as an environment.
fn env<V: AsRef<OsStr>>(pairs: &[(&str, V)]) -> Vec<(OsString, OsString)> {
    pairs
        .iter()
        .map(|(k, v)| (OsString::from(k), v.as_ref().to_os_string()))
        .collect()
}

/// The rows a reading here parses.
fn parsed() -> impl Iterator<Item = &'static LeverSpec> {
    REGISTRY
        .iter()
        .filter(|r| matches!(r.site, Site::Parsed { .. }))
}

/// The retired rows.
fn retired() -> impl Iterator<Item = &'static LeverSpec> {
    REGISTRY
        .iter()
        .filter(|r| matches!(r.site, Site::Retired { .. }))
}

/// Whether the pool reads it: it is on every binary's list.
fn pool(r: &LeverSpec) -> bool {
    r.name == THREADS || r.name == SPIN
}

/// Every file that reads a lever in place, with its round.
fn in_place(site: Site) -> &'static [InPlace] {
    match site {
        Site::Parsed { left } | Site::Retired { left, .. } => left,
        Site::Direct { at } => at,
        Site::Env { .. } => &[],
    }
}

/// A value `kind` takes.
fn sample(kind: Kind) -> String {
    match kind {
        Kind::Flag => "1".into(),
        Kind::OnOff => "on".into(),
        Kind::Words(words) => words[0].into(),
        Kind::Count { min, .. } => min.to_string(),
        Kind::Multiple { of } => of.to_string(),
        Kind::Bytes => "38G".into(),
        Kind::Path => "/data/set".into(),
        Kind::PathOr(words) => words[0].into(),
        Kind::File => A_FILE.into(),
        Kind::Text => "text".into(),
        Kind::Residency => "mid-p33-s1".into(),
    }
}

/// Values `kind` refuses: the empty value, other spellings, padding where
/// the kind does not trim, signs, leading zeros, words for numbers, the
/// range's neighbours and a number past `u64`.
fn garbage(kind: Kind) -> Vec<String> {
    const PAST_U64: &str = "18446744073709551616";
    let mut g: Vec<String> = vec![String::new()];
    match kind {
        Kind::Flag => g.extend(["1 ", " 1", "true", "2", "off", "no", "yes"].map(String::from)),
        Kind::OnOff => g.extend(["ON", "Off", " on", "1", "0", "maybe"].map(String::from)),
        Kind::Words(words) => {
            g.push("not-a-word".into());
            for w in words {
                let upper = w.to_uppercase();
                if upper != *w {
                    g.push(upper);
                }
                g.extend([format!(" {w}"), format!("{w} ")]);
            }
        }
        Kind::Count { min, max, trim } => {
            let v = min.max(1);
            g.extend([
                "-1".to_string(),
                format!("+{v}"),
                format!("0{v}"),
                "two".into(),
                PAST_U64.into(),
            ]);
            if min > 0 {
                g.push((min - 1).to_string());
            }
            if max < u64::MAX {
                g.push((max + 1).to_string());
            }
            if !trim {
                g.extend([format!(" {v}"), format!("{v} ")]);
            }
        }
        Kind::Multiple { of } => g.extend([
            "0".to_string(),
            (of + 1).to_string(),
            format!("+{of}"),
            format!("0{of}"),
            format!("-{of}"),
        ]),
        Kind::Bytes => {
            g.extend(["38g", "38 G", "-1", "38GB", "1.5G", "G", "17179869184G"].map(String::from))
        }
        Kind::File => g.extend([
            "/no/such/file".to_string(),
            env!("CARGO_MANIFEST_DIR").to_string(),
        ]),
        Kind::Path | Kind::PathOr(_) | Kind::Text => {}
        Kind::Residency => g.extend(
            [
                "OFF",
                " off",
                "off ",
                "mid",
                "mid-p",
                "mid-p1",
                "mid-p-s1",
                "mid-p1-s",
                "mid-p1-s0",
                "mid-p01-s1",
                "mid-p1-s01",
                "mid-p+1-s1",
                "mid-p1-s1 ",
                "MID-p1-s1",
                "mid-p18446744073709551616-s1",
            ]
            .map(String::from),
        ),
    }
    g
}

/// `BLOOMERY_PREFILL_GROUP`'s row defaults to [`PREFILL_GROUP_DEFAULT`],
/// the group GLM-5.3's plan reserves its prompt units for.
#[test]
fn prefill_group_default_is_the_rows() {
    let row = REGISTRY.iter().find(|r| r.name == PREFILL_GROUP);
    let Some(LeverSpec {
        default: Unset::Is(d),
        ..
    }) = row
    else {
        panic!("{PREFILL_GROUP} has no default value: {row:?}");
    };
    assert_eq!(*d, PREFILL_GROUP_DEFAULT.to_string());
}

/// `BLOOMERY_LANE_PREFETCH`'s row is a parsed same-binary arm taking `on`
/// and `off`, its default [`LANE_PREFETCH_DEFAULT`] (what the host union
/// holds in a binary that does not act on it), and its accessor reads unset
/// as that default, `on` as on and `off` as off.
#[test]
fn lane_prefetch_row_is_an_on_off_arm() {
    let row = REGISTRY.iter().find(|r| r.name == LANE_PREFETCH);
    let Some(LeverSpec {
        class: Class::A,
        kind: Kind::OnOff,
        default: Unset::Is(d),
        site: Site::Parsed { left },
        ..
    }) = row
    else {
        panic!("{LANE_PREFETCH} is not a parsed on/off arm with a default: {row:?}");
    };
    assert!(
        left.is_empty(),
        "{LANE_PREFETCH} is read in place: {left:?}"
    );
    assert_eq!(*d, if LANE_PREFETCH_DEFAULT { "on" } else { "off" });
    let at = |v: Option<&str>| {
        let e = v.map_or_else(Vec::new, |v| env(&[(LANE_PREFETCH, v)]));
        read(&e, Scope::Every)
            .expect("on, off and unset are taken")
            .lane_prefetch()
    };
    assert_eq!(at(None), LANE_PREFETCH_DEFAULT);
    assert!(at(Some("on")));
    assert!(!at(Some("off")));
}

/// Every name is a `BLOOMERY_*` variable, and no two rows share one.
#[test]
fn names_are_unique() {
    let mut names: Vec<&str> = REGISTRY.iter().map(|r| r.name).collect();
    assert!(
        names.iter().all(|n| n.starts_with("BLOOMERY_")),
        "{names:?}"
    );
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "a name is on two rows");
}

/// Every row can be read: a default given as a value is one of its kind, a
/// word list is not empty and names each word once, a count's range is not
/// empty, no text breaks a Markdown table cell; a name is no lever exactly
/// when its class says so, a lever a reading parses has a kind it can parse,
/// and a script is named from `tools/`.
#[test]
fn rows_are_well_formed() {
    for r in REGISTRY {
        if let Unset::Is(d) = r.default {
            assert!(r.kind.parse(d).is_ok(), "{}: default {d:?}", r.name);
        }
        match r.kind {
            Kind::Words(words) | Kind::PathOr(words) => {
                let mut w = words.to_vec();
                w.sort_unstable();
                w.dedup();
                assert!(!words.is_empty() && w.len() == words.len(), "{}", r.name);
            }
            Kind::Count { min, max, .. } => assert!(min <= max, "{}", r.name),
            Kind::Multiple { of } => assert!(of > 0, "{}", r.name),
            _ => {}
        }
        let (Unset::Is(unset) | Unset::Means(unset)) = r.default;
        let texts = [
            r.doc.to_string(),
            unset.to_string(),
            r.kind.takes(),
            r.site.describe(),
        ];
        assert!(!r.doc.is_empty(), "{}", r.name);
        assert!(texts.iter().all(|t| !t.contains('|')), "{}: a '|'", r.name);
        let no_lever = matches!(r.class, Class::P | Class::R);
        assert_eq!(
            no_lever,
            !r.is_lever(),
            "{}: class and site disagree",
            r.name
        );
        if let Site::Parsed { .. } = r.site {
            assert_ne!(r.kind, Kind::Text, "{}: parsed as text", r.name);
        }
        if let Site::Env {
            script: Some(script),
        } = r.site
        {
            assert!(
                !script.starts_with("tools/") && !script.starts_with('/'),
                "{}: {script} is not a path from tools/",
                r.name
            );
        }
    }
}

/// The whole `"BLOOMERY_…"` literals of the rows' source are the rows'
/// names: the set `tools/check-levers.sh` reads as the registry's is the one
/// a reading knows.
#[test]
fn registry_names_are_its_literals() {
    let mut literals: Vec<&str> = REGISTRY_SOURCE
        .match_indices("\"BLOOMERY_")
        .filter_map(|(at, _)| {
            let rest = &REGISTRY_SOURCE[at + 1..];
            let end = rest
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(rest.len());
            rest[end..].starts_with('"').then(|| &rest[..end])
        })
        .collect();
    literals.sort_unstable();
    literals.dedup();
    let mut names: Vec<&str> = REGISTRY.iter().map(|r| r.name).collect();
    names.sort_unstable();
    assert_eq!(literals, names);
}

/// Each accessor reads its own row: the defaults unset, the values as set —
/// the last of a name given twice — and the parsed rows are exactly the
/// accessors' rows.
#[test]
fn accessors_read_their_rows() {
    let unset = read(&[], Scope::Every).expect("nothing is set");
    assert_eq!(unset.threads(), None);
    assert_eq!(unset.spin(), 20_000);
    assert!(unset.ced());
    assert_eq!(unset.prefill(), "batch");
    assert_eq!(unset.prefill_group(), 2);
    assert!(unset.engram_helper());
    assert!(!unset.step_stats());
    assert_eq!(unset.card_budget_bytes(), None);
    assert!(unset.pin_main());
    assert_eq!(unset.draft(), None);
    assert_eq!(unset.mtp_head_rows(), None);
    assert_eq!(unset.mtp_draft(), None);
    assert_eq!(unset.route_trace(), None);
    assert!(!unset.check_finite());
    assert!(!unset.mtp_windows());
    assert_eq!(unset.qwen38_experts(), "card");
    assert_eq!(unset.qwen38_experts_set(), None);
    assert_eq!(unset.residency(), None);
    assert_eq!(unset.hoststream(), None);
    assert_eq!(unset.lane_prefetch(), LANE_PREFETCH_DEFAULT);
    assert_eq!(
        unset.host(),
        HostCfg {
            populate: true,
            lock: false,
            card_dontneed: true,
            r8: true
        }
    );

    let set = read(
        &env(&[
            (THREADS, " 3 "),
            (SPIN, "0"),
            (CED, "off"),
            (PREFILL, "steps"),
            (PREFILL_GROUP, "1"),
            (ENGRAM_HELPER, "0"),
            (STEP_STATS, "1"),
            (CARD_BUDGET, "38G"),
            (PIN_MAIN, "0"),
            (DRAFT, "dspark"),
            (MTP_HEAD_ROWS, "/data/rows.txt"),
            (MTP_DRAFT, A_FILE),
            (MTP_WINDOWS, "1"),
            (ROUTE_TRACE, "/data/trace"),
            (CHECK_FINITE, "1"),
            (HOST_POPULATE, "0"),
            (HOST_LOCK, "1"),
            (CARD_DONTNEED, "0"),
            (R8, "off"),
            (RESIDENCY, "mid-p40-s1"),
            (QWEN38_EXPERTS, "card"),
            (HOSTSTREAM, "on"),
            (QWEN3_KV, "q8_0"),
            (LANE_PREFETCH, "on"),
            (PREFILL_GROUP, "8"),
        ]),
        Scope::Every,
    )
    .expect("every value is one its kind takes");
    assert_eq!(set.threads(), Some(3));
    assert_eq!(set.spin(), 0);
    assert!(!set.ced());
    assert_eq!(set.prefill(), "steps");
    assert_eq!(set.prefill_group(), 8);
    assert!(!set.engram_helper());
    assert!(set.step_stats());
    assert_eq!(set.card_budget_bytes(), Some(38 << 30));
    assert!(!set.pin_main());
    assert_eq!(set.draft(), Some("dspark"));
    assert_eq!(
        set.mtp_head_rows(),
        Some(MtpHead::List(Path::new("/data/rows.txt")))
    );
    assert_eq!(set.mtp_draft(), Some(Path::new(A_FILE)));
    assert!(set.mtp_windows());
    assert_eq!(set.route_trace(), Some(Path::new("/data/trace")));
    assert!(set.check_finite());
    assert_eq!(set.residency(), Some("mid-p40-s1"));
    assert_eq!(set.qwen38_experts(), "card");
    assert_eq!(set.qwen38_experts_set(), Some("card"));
    assert_eq!(set.qwen3_kv_set(), Some("q8_0"));
    assert_eq!(set.hoststream(), Some(true));
    assert!(set.lane_prefetch());
    assert_eq!(
        set.host(),
        HostCfg {
            populate: false,
            lock: true,
            card_dontneed: false,
            r8: false
        }
    );

    let names: Vec<&str> = parsed().map(|r| r.name).collect();
    assert_eq!(
        names,
        [
            THREADS,
            SPIN,
            CED,
            PREFILL,
            PREFILL_GROUP,
            ENGRAM_HELPER,
            STEP_STATS,
            CARD_BUDGET,
            PIN_MAIN,
            DRAFT,
            MTP_HEAD_ROWS,
            MTP_DRAFT,
            MTP_WINDOWS,
            ROUTE_TRACE,
            CHECK_FINITE,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
            RESIDENCY,
            QWEN38_EXPERTS,
            HOSTSTREAM,
            QWEN3_KV,
            LANE_PREFETCH
        ]
    );
}

/// `BLOOMERY_QWEN3_KV` takes the two cache-format words, and unset is the
/// f16 default the load reads.
#[test]
fn qwen3_kv_takes_f16_and_q8_0() {
    for w in ["f16", "q8_0"] {
        let set = read(&env(&[(QWEN3_KV, w)]), Scope::Every).expect("a word the row takes");
        assert_eq!(set.qwen3_kv_set(), Some(w));
    }
    let Err(other) = read(&env(&[(QWEN3_KV, "q6_0")]), Scope::Every) else {
        panic!("a word the row does not take is refused");
    };
    assert!(other.to_string().contains("BLOOMERY_QWEN3_KV"), "{other}");
}

/// `BLOOMERY_DRAFT` takes `mtp` and `off` beside the V4.1 words: the row's
/// own kind, read by the qwen4exp binaries; a word it does not take is
/// refused by name.
#[test]
fn draft_row_takes_mtp_and_off() {
    for w in ["mtp", "off"] {
        let set = read(&env(&[(DRAFT, w)]), Scope::Every).expect("a word the row takes");
        assert_eq!(set.draft(), Some(w));
    }
    let Err(other) = read(&env(&[(DRAFT, "draft")]), Scope::Every) else {
        panic!("a word the row does not take is refused");
    };
    assert!(other.to_string().contains("BLOOMERY_DRAFT"), "{other}");
}

/// `BLOOMERY_MTP_DRAFT` set to a path with nothing there, or to a
/// directory, is refused saying why, beside what the kind takes
/// (`parsed_levers_refuse_what_their_kind_does_not_take` holds the rest).
#[test]
fn mtp_draft_refusal_says_no_file_is_there() {
    for value in ["/no/such/draft.gguf", env!("CARGO_MANIFEST_DIR")] {
        let Err(e) = read(&env(&[(MTP_DRAFT, value)]), Scope::Every) else {
            panic!("{value:?} is refused");
        };
        assert_eq!(
            e.to_string(),
            format!(
                "BLOOMERY_MTP_DRAFT={value:?} is refused: no regular file is at that path; it \
                 takes the path of an existing regular file"
            )
        );
    }
}

/// `BLOOMERY_RESIDENCY` unset follows the placement: the serving word under a
/// serving placement, `off` where the machine does not run, the first
/// condition that holds named; set, it is the load's word wherever it is.
#[test]
fn residency_unset_follows_the_placement() {
    let at = |serving_place, check_finite, route_trace, prefill_steps| ResidencyAt {
        serving_place,
        check_finite,
        route_trace,
        prefill_steps,
    };
    let unset = read(&[], Scope::Every).expect("nothing is set");
    for (at, word, why) in [
        (
            at(true, false, false, false),
            "mid-p40-s1",
            ResidencyWhy::Place,
        ),
        (
            at(false, false, false, false),
            "off",
            ResidencyWhy::FixedPlace,
        ),
        (ResidencyAt::FIXED, "off", ResidencyWhy::FixedPlace),
        (at(false, true, true, true), "off", ResidencyWhy::FixedPlace),
        (
            at(true, true, false, false),
            "off",
            ResidencyWhy::CheckFinite,
        ),
        (at(true, true, true, true), "off", ResidencyWhy::CheckFinite),
        (
            at(true, false, true, false),
            "off",
            ResidencyWhy::RouteTrace,
        ),
        (at(true, false, true, true), "off", ResidencyWhy::RouteTrace),
        (
            at(true, false, false, true),
            "off",
            ResidencyWhy::PrefillSteps,
        ),
    ] {
        assert_eq!(
            unset.residency_at(at),
            ResidencyPick { word, why },
            "unset at {at:?}"
        );
    }
    for w in ["off", "mid-p0-s1", "mid-p40-s1"] {
        let set = read(&env(&[(RESIDENCY, w)]), Scope::Every).expect("a word the row takes");
        for at in [at(true, false, false, false), ResidencyAt::FIXED] {
            assert_eq!(
                set.residency_at(at),
                ResidencyPick {
                    word: w,
                    why: ResidencyWhy::Set
                },
                "{w} at {at:?}"
            );
        }
    }
    let row = spec(RESIDENCY).expect("the row");
    row.kind
        .parse(RESIDENCY_SERVING)
        .unwrap_or_else(|e| panic!("the serving word is not one the row takes: {e}"));
}

/// `BLOOMERY_RESIDENCY` unset in `generate_qwen3moe`: `off` before the plan
/// where the machine does not run, the first condition that holds named;
/// on plan (a) `mid-p<P>-s1`, P half the plan's fewest card experts a layer,
/// and `off` with why when no layer holds one, when the fewest leave no room,
/// and when the churn pool at P does not fit the plan's host headroom or
/// what `MemAvailable` leaves.
#[test]
fn residency38_unset_follows_the_plan() {
    let at = |qwen38_file, dump_taps, place_a, route_trace, prefill_step| Residency38At {
        qwen38_file,
        dump_taps,
        place_a,
        route_trace,
        prefill_step,
    };
    let off = |why| Some(Residency38Pick { pinned: None, why });
    for (at, want) in [
        (at(true, false, true, false, false), None),
        (
            at(true, false, false, false, false),
            off(Residency38Why::Gate),
        ),
        (
            at(true, false, false, true, true),
            off(Residency38Why::Gate),
        ),
        (
            at(true, false, true, true, false),
            off(Residency38Why::RouteTrace),
        ),
        (
            at(true, false, true, true, true),
            off(Residency38Why::RouteTrace),
        ),
        (
            at(true, false, true, false, true),
            off(Residency38Why::PrefillStep),
        ),
        (
            at(false, false, true, false, false),
            off(Residency38Why::Family),
        ),
        (
            at(false, false, false, true, true),
            off(Residency38Why::Family),
        ),
        (
            at(false, true, true, false, false),
            off(Residency38Why::DumpTaps),
        ),
        (
            at(true, true, true, true, true),
            off(Residency38Why::DumpTaps),
        ),
    ] {
        assert_eq!(residency38_unset(at), want, "{at:?}");
    }

    // The pool is 10 B an expert past the pinned ones, on two layers of the
    // fewest; the headroom takes it or does not.
    let pool = |fewest: usize| move |p: usize| Ok::<u64, ()>(2 * 10 * (fewest - p) as u64);
    let pick_mem = |n_l: &[u64], headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        residency38_at_plan(n_l.iter().copied(), pool(fewest), headroom, mem_left)
            .expect("no pool error")
    };
    let pick = |n_l: &[u64], headroom: i128| pick_mem(n_l, headroom, 1 << 40);
    let derived = pick(&[0, 297, 298, 0, 297], 1 << 40);
    assert_eq!(
        derived,
        Residency38Pick {
            pinned: Some(148),
            why: Residency38Why::PlanA { fewest: 297 }
        }
    );
    assert_eq!(derived.word(), "mid-p148-s1");
    assert_eq!(
        derived.why.to_string(),
        "unset: plan (a), P half the plan's fewest card experts a layer (297)"
    );
    let row = spec(RESIDENCY).expect("the row");
    row.kind
        .parse(&derived.word())
        .unwrap_or_else(|e| panic!("the derived word at the plain plan's count: {e}"));
    assert_eq!(pick(&[3, 4], 1 << 40).word(), "mid-p1-s1");
    for n_l in [&[0u64, 0][..], &[]] {
        let p = pick(n_l, 1 << 40);
        assert_eq!(
            p,
            Residency38Pick {
                pinned: None,
                why: Residency38Why::NoCardExperts
            }
        );
        assert_eq!(p.word(), "off");
    }
    for fewest in [1u64, 2] {
        assert_eq!(
            pick(&[fewest, 9], 1 << 40),
            Residency38Pick {
                pinned: None,
                why: Residency38Why::NoRoom {
                    fewest: fewest as usize
                }
            }
        );
    }
    // 297 − 148 = 149 experts a layer, 2 layers, 10 B each: 2,980 B.
    assert_eq!(pick(&[297, 297], 2_980).pinned, Some(148));
    let short = pick(&[297, 297], 2_979);
    assert_eq!(
        short,
        Residency38Pick {
            pinned: None,
            why: Residency38Why::HostShort {
                needs: 2_980,
                leaves: 2_979
            }
        }
    );
    assert_eq!(
        short.why.to_string(),
        "unset: the churn pool needs 2980 B, the plan leaves 2979 B"
    );
    assert_eq!(
        pick(&[297], -5).why,
        Residency38Why::HostShort {
            needs: 2_980,
            leaves: -5
        }
    );
    assert_eq!(pick_mem(&[297, 297], 2_980, 2_980).pinned, Some(148));
    let mem = pick_mem(&[297, 297], 1 << 40, 2_979);
    assert_eq!(
        mem.why,
        Residency38Why::MemShort {
            needs: 2_980,
            leaves: 2_979
        }
    );
    assert_eq!(mem.word(), "off");
    assert_eq!(
        mem.why.to_string(),
        "unset: the churn pool needs 2980 B, MemAvailable leaves 2979 B past the plan's host need"
    );
    assert_eq!(
        residency38_at_plan([297u64], |_| Err::<u64, &str>("the pool"), 0, 0),
        Err("the pool"),
        "the pool's error is the call's"
    );
}

/// `BLOOMERY_DRAFT` unset on a qwen4exp file in `generate_qwen3moe`: the MTP
/// draft under `--place a` with its file there and room for the last
/// window; else the plain path, the first condition that holds named.
#[test]
fn draft38_unset_follows_the_place_and_the_file() {
    let file = Path::new("/models/q/mtp.gguf");
    let at = |place_a, logits, route_trace, file_is_there, need| Draft38At {
        place_a,
        logits,
        route_trace,
        file,
        file_is_there,
        need,
        ctx: 4352,
    };
    for (at, want) in [
        (at(true, false, false, true, 4194), None),
        (at(true, false, false, true, 4352), None),
        (
            at(true, false, false, true, 4353),
            Some(Draft38Off::Ctx {
                need: 4353,
                ctx: 4352,
            }),
        ),
        (
            at(true, false, false, false, 4353),
            Some(Draft38Off::NoFile(file.to_path_buf())),
        ),
        (
            at(true, false, true, false, 1),
            Some(Draft38Off::RouteTrace),
        ),
        (at(true, true, true, false, 1), Some(Draft38Off::Logits)),
        (at(false, true, true, false, 9999), Some(Draft38Off::Gate)),
        (at(false, false, false, true, 1), Some(Draft38Off::Gate)),
    ] {
        assert_eq!(draft38_unset(&at), want, "{at:?}");
    }
    assert_eq!(
        Draft38Off::NoFile(file.to_path_buf()).to_string(),
        "no file at /models/q/mtp.gguf"
    );
}

/// The GLM seat's levers unset: under a serving placement (`--place a` or
/// `bp`, both `serving_place`) the NextN draft and the
/// residency's default word; `off`, the first condition that holds named,
/// under `--place gate`, with stores short of a window (both), on a file of
/// other than one next-token layer (the draft), beside the step feed (the
/// residency).
#[test]
fn glm_unset_follows_the_place() {
    let at = |serving_place, nextn_layers, ctx, prefill_steps| GlmAt {
        serving_place,
        nextn_layers,
        need: 3,
        ctx,
        prefill_steps,
    };
    let pick = |word, why| GlmPick { word, why };
    let on_a = GlmUnset {
        draft: pick("mtp", GlmWhy::Serving),
        residency: pick("mid-p0-s1", GlmWhy::Serving),
    };
    for (at, want) in [
        (at(true, 1, 2048, false), on_a),
        (at(true, 1, 3, false), on_a),
        (
            at(false, 1, 2048, false),
            GlmUnset {
                draft: pick("off", GlmWhy::Gate),
                residency: pick("off", GlmWhy::Gate),
            },
        ),
        (
            at(false, 0, 2, true),
            GlmUnset {
                draft: pick("off", GlmWhy::Gate),
                residency: pick("off", GlmWhy::Gate),
            },
        ),
        (
            at(true, 1, 2, false),
            GlmUnset {
                draft: pick("off", GlmWhy::Ctx { need: 3, ctx: 2 }),
                residency: pick("off", GlmWhy::Ctx { need: 3, ctx: 2 }),
            },
        ),
        (
            at(true, 0, 2048, true),
            GlmUnset {
                draft: pick("off", GlmWhy::Nextn { layers: 0 }),
                residency: pick("off", GlmWhy::PrefillSteps),
            },
        ),
        (
            at(true, 2, 2048, false),
            GlmUnset {
                draft: pick("off", GlmWhy::Nextn { layers: 2 }),
                residency: pick("mid-p0-s1", GlmWhy::Serving),
            },
        ),
    ] {
        assert_eq!(glm_unset(at), want, "{at:?}");
    }
    assert_eq!(
        GlmWhy::Ctx { need: 3, ctx: 2 }.to_string(),
        "unset: one window needs 3 positions, past --ctx 2"
    );
    assert_eq!(
        GlmWhy::Gate.to_string(),
        "unset: --place gate keeps its fixed placement"
    );
    assert_eq!(GlmWhy::Serving.to_string(), "unset: --place a or bp");
}

/// The GLM seat's unset residency on the plan: the default word where it
/// fits, else `off` naming the first shortfall; `off` and a set word pass
/// through untouched.
#[test]
fn glm_residency_at_plan_takes_only_what_fits() {
    let mid = GlmPick {
        word: "mid-p0-s1",
        why: GlmWhy::Serving,
    };
    let off = |why| GlmPick { word: "off", why };
    let fit = |n: &[u64], pool: u64, headroom, mem| {
        glm_residency_at_plan(
            mid,
            n.iter().copied(),
            |p| {
                assert_eq!(p, 0, "the word's pinned count");
                Ok::<u64, &str>(pool)
            },
            headroom,
            mem,
        )
    };
    assert_eq!(fit(&[0, 66, 67], 10, 10, 10), Ok(mid));
    assert_eq!(fit(&[0, 0], 10, 10, 10), Ok(off(GlmWhy::NoCardExperts)));
    assert_eq!(
        fit(&[0, 1, 67], 10, 10, 10),
        Ok(off(GlmWhy::NoRoom { fewest: 1 }))
    );
    assert_eq!(
        fit(&[66], 10, 9, 10),
        Ok(off(GlmWhy::HostShort {
            needs: 10,
            leaves: 9
        }))
    );
    assert_eq!(
        fit(&[66], 10, 10, 9),
        Ok(off(GlmWhy::MemShort {
            needs: 10,
            leaves: 9
        }))
    );
    let gate = off(GlmWhy::Gate);
    assert_eq!(
        glm_residency_at_plan(gate, [66u64], |_| Err("not asked"), 0, 0),
        Ok(gate)
    );
    assert_eq!(fit(&[66], 10, 10, 10).map(|p| p.why), Ok(GlmWhy::Serving));
    assert_eq!(
        glm_residency_at_plan(mid, [66u64], |_| Err::<u64, &str>("the pool"), 0, 0),
        Err("the pool"),
        "the pool's error is the call's"
    );
}

/// `BLOOMERY_QWEN38_EXPERTS` unset is the card plan; set, as set.
#[test]
fn qwen38_experts_unset_is_card() {
    let unset = read(&[], Scope::Every).expect("nothing is set");
    assert_eq!(
        (unset.qwen38_experts(), unset.qwen38_experts_set()),
        ("card", None)
    );
    for w in ["host", "card"] {
        let set = read(&env(&[(QWEN38_EXPERTS, w)]), Scope::Every).expect("a word the row takes");
        assert_eq!(
            (set.qwen38_experts(), set.qwen38_experts_set()),
            (w, Some(w))
        );
    }
}

/// Every Parsed lever refuses, naming itself, each value its kind does not
/// take and a value that is not UTF-8 — a number past `u64` saying so — in a
/// binary's reading as in a harness's; a row that trims takes its value with
/// spaces around it.
#[test]
fn parsed_levers_refuse_what_their_kind_does_not_take() {
    let raw = OsStr::from_bytes(b"o\xffn");
    for r in parsed() {
        let acts_on = [r.name];
        let scopes = [
            Scope::Every,
            Scope::Main {
                bin: "b",
                acts_on: &acts_on,
            },
        ];
        let mut values: Vec<OsString> = garbage(r.kind).into_iter().map(OsString::from).collect();
        values.push(raw.to_os_string());
        for v in &values {
            for scope in scopes {
                let Err(e) = read(&env(&[(r.name, v)]), scope) else {
                    panic!("{}={v:?} was taken", r.name);
                };
                let [refusal] = e.refused.as_slice() else {
                    panic!("{}={v:?}: {e}", r.name);
                };
                let Why::Value(expected) = &refusal.why else {
                    panic!("{}={v:?}: {e}", r.name);
                };
                assert_eq!(refusal.name, r.name, "{e}");
                assert!(expected.contains(&r.kind.takes()), "{e}");
                if v.as_encoded_bytes().starts_with(b"1844674407370955161") || v == "17179869184G" {
                    assert!(expected.starts_with(OVERFLOW), "{e}");
                }
                if v == raw {
                    assert!(expected.starts_with("not UTF-8"), "{e}");
                }
            }
        }
        if let Kind::Count { trim: true, .. } = r.kind {
            let padded = format!(" {} ", sample(r.kind));
            let levers = read(&env(&[(r.name, padded.as_str())]), Scope::Every)
                .unwrap_or_else(|e| panic!("{}={padded:?}: {e}", r.name));
            assert_eq!(levers.entry(r.name).set.as_deref(), Some(padded.as_str()));
        }
    }
}

/// A retired name is refused, naming itself, whatever it is set to — empty,
/// a value, not UTF-8 — by a binary's reading, a harness's and the pool's.
#[test]
fn retired_names_are_refused_whatever_their_value() {
    for r in retired() {
        let values = [OsStr::new(""), OsStr::new("1"), OsStr::from_bytes(b"\xff")];
        let scopes = [
            Scope::Main {
                bin: "b",
                acts_on: &[],
            },
            Scope::Every,
            Scope::Pool,
        ];
        for v in values {
            for scope in scopes {
                let Err(e) = read(&env(&[(r.name, v)]), scope) else {
                    panic!("{}={v:?} was taken", r.name);
                };
                assert!(
                    matches!(e.refused.as_slice(), [f] if f.name == r.name
                        && matches!(f.why, Why::Retired(_))),
                    "{e}"
                );
            }
        }
    }
}

/// A binary refuses, naming itself and the lever, a Parsed lever set that it
/// does not act on, and takes one it does; the pool's two it always acts on.
/// Its table prints `-` for the rest, and a name that is not a Parsed lever's
/// among those it acts on is a caller's mistake, which its reading names.
#[test]
fn a_binary_refuses_the_levers_it_does_not_act_on() {
    for r in parsed() {
        let set = env(&[(r.name, sample(r.kind))]);
        let none = Scope::Main {
            bin: "b",
            acts_on: &[],
        };
        match read(&set, none) {
            Ok(_) => assert!(pool(r), "{} was taken by a binary that left it out", r.name),
            Err(e) => assert!(
                !pool(r)
                    && matches!(e.refused.as_slice(), [f] if f.name == r.name
                        && f.why == Why::NotActedOn("b".into())),
                "{e}"
            ),
        }
        let acts_on = [r.name];
        let levers = read(
            &set,
            Scope::Main {
                bin: "b",
                acts_on: &acts_on,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let table = levers.table();
        for p in parsed() {
            let line = table
                .lines()
                .find(|l| l.split_whitespace().next() == Some(p.name))
                .unwrap_or_else(|| panic!("no line for {}", p.name));
            let dash = line.split_whitespace().nth(3) == Some("-");
            assert_eq!(dash, p.name != r.name && !pool(p), "{line}");
        }
    }
    let parsed_names: Vec<&str> = parsed().map(|r| r.name).collect();
    assert_eq!(not_parsed(&parsed_names), None);
    for r in REGISTRY
        .iter()
        .filter(|r| !matches!(r.site, Site::Parsed { .. }))
    {
        assert_eq!(not_parsed(&[parsed_names[0], r.name]), Some(r.name));
    }
    assert_eq!(
        not_parsed(&["BLOOMERY_NO_SUCH_NAME"]),
        Some("BLOOMERY_NO_SUCH_NAME")
    );
}

/// A binary refuses a `BLOOMERY_*` name no row names, naming it and the row
/// name nearest to it; every row's name it knows, so a lever read in place
/// and a name that is no lever set together pass; a harness's reading and
/// the pool's leave other names alone.
#[test]
fn a_binary_refuses_a_name_no_row_names() {
    let main = Scope::Main {
        bin: "b",
        acts_on: &[],
    };
    for r in REGISTRY {
        let typo = &r.name[..r.name.len() - 1];
        if spec(typo).is_some() {
            continue;
        }
        let e = read(&env(&[(typo, "x")]), main).expect_err(typo);
        let [f] = e.refused.as_slice() else {
            panic!("{typo}: {e}")
        };
        let Why::Unknown(nearest) = f.why else {
            panic!("{typo}: {e}")
        };
        assert_eq!(f.name, typo, "{e}");
        if let Site::Parsed { .. } = r.site {
            assert_eq!(nearest, r.name, "{e}");
        }
        for scope in [Scope::Every, Scope::Pool] {
            assert!(read(&env(&[(typo, "x")]), scope).is_ok(), "{typo}");
        }
    }
    let known: Vec<(&str, &str)> = REGISTRY
        .iter()
        .filter(|r| matches!(r.site, Site::Direct { .. } | Site::Env { .. }))
        .map(|r| (r.name, "x"))
        .collect();
    let mut all = env(&known);
    all.push(("PATH".into(), "/bin".into()));
    read(&all, main).unwrap_or_else(|e| panic!("{e}"));
}

/// A reading reports every refusal at once: the values its levers' kinds do
/// not take, the levers its binary does not act on and the retired names, in
/// the registry's order, then the names no row names.
#[test]
fn a_reading_refuses_everything_at_once() {
    let levers: Vec<&LeverSpec> = parsed().filter(|r| !pool(r)).collect();
    let (bad, left_out) = (levers[0], levers[1]);
    let gone = retired().next().expect("a retired row");
    let acts_on = [bad.name];
    let e = read(
        &env(&[
            ("BLOOMERY_NO_SUCH_NAME", "1".to_string()),
            (gone.name, "1".to_string()),
            (left_out.name, sample(left_out.kind)),
            (bad.name, String::new()),
        ]),
        Scope::Main {
            bin: "b",
            acts_on: &acts_on,
        },
    )
    .expect_err("four refusals");
    let got: Vec<&str> = e.refused.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        got,
        [bad.name, left_out.name, gone.name, "BLOOMERY_NO_SUCH_NAME"],
        "{e}"
    );
    assert_eq!(e.to_string().lines().count(), 4, "{e}");
}

/// The pool reads its two levers alone — any other lever, whatever its value,
/// is its binary's — and refuses a retired name.
#[test]
fn the_pool_reads_its_two_levers() {
    let mut pairs: Vec<(&str, &str)> = parsed()
        .filter(|r| !pool(r))
        .map(|r| (r.name, "\u{1}garbage"))
        .collect();
    pairs.extend([(THREADS, " 3"), (SPIN, "7")]);
    let levers = read(&env(&pairs), Scope::Pool).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((levers.threads(), levers.spin()), (Some(3), 7));
    assert_eq!(levers.entries.len(), 2);
    let gone = retired().next().expect("a retired row");
    assert!(read(&env(&[(gone.name, "1")]), Scope::Pool).is_err());
}

#[test]
fn parse_bytes_takes_bytes_mib_and_gib() {
    assert_eq!(parse_bytes("40802189312"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("40_802_189_312"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("38G"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("24176M"), Ok(25_350_373_376));
    for bad in ["", "G", "38 G", "38g", "38GB", "-1", "1.5G"] {
        assert_eq!(parse_bytes(bad), Err(BytesError::NotBytes(bad.into())));
    }
    for big in ["18446744073709551616", "17179869184G"] {
        assert_eq!(parse_bytes(big), Err(BytesError::Overflow(big.into())));
    }
}

/// `BLOOMERY_GATE_BOUND`'s row is the runners' parser's: its default is the
/// parser's, and its kind takes a value exactly when the parser's pattern,
/// `^[1-9][0-9]*$`, does, up to `u64`.
#[test]
fn gate_bound_is_the_runners_parser() {
    const PATTERN: &str = "=~ ^[1-9][0-9]*$ ]]";
    let row = spec("BLOOMERY_GATE_BOUND").expect("a row");
    let default = GATE_BOUND_PARSER
        .split_once("${BLOOMERY_GATE_BOUND-")
        .and_then(|(_, rest)| rest.split_once('}'))
        .map(|(d, _)| d)
        .expect("the parser's default");
    assert_eq!(row.default, Unset::Is(default));
    assert!(
        GATE_BOUND_PARSER.contains(PATTERN),
        "the parser's pattern moved"
    );
    let pattern = |v: &str| {
        v.bytes().next().is_some_and(|b| (b'1'..=b'9').contains(&b))
            && v.bytes().all(|b| b.is_ascii_digit())
    };
    for v in [
        "",
        "0",
        "00",
        "007",
        "+5",
        "-1",
        "1",
        "900",
        "1680",
        " 5",
        "5 ",
        "abc",
        "1e3",
        "18446744073709551615",
    ] {
        assert_eq!(row.kind.parse(v).is_ok(), pattern(v), "{v:?}");
    }
}

/// One line of `tools/levers-direct.txt`: a file, the variables it reads in
/// place, and the round that converts or removes the read.
struct AllowLine<'a> {
    file: &'a str,
    vars: Vec<&'a str>,
    round: &'a str,
}

fn allow_lines() -> Vec<AllowLine<'static>> {
    ALLOW_LIST
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 3, "not `file<TAB>vars<TAB>round`: {l:?}");
            AllowLine {
                file: f[0],
                vars: f[1].split(',').collect(),
                round: f[2],
            }
        })
        .collect()
}

/// The registry and the allow-list name the same in-place reads of the
/// crates' sources: every file a lever's row reads it in place in is a line
/// of the list with that lever, in the same round, and every lever a line
/// under `crates/*/src/` names is a row that names that file and round. A
/// harness's read (`tests/`, a build script) is the list's alone, and so is
/// every read of a name that is no lever; every `BLOOMERY_*` name the list
/// gives is a row.
#[test]
fn registry_and_allow_list_agree() {
    let lines = allow_lines();
    let mut bad = Vec::new();
    for r in REGISTRY {
        for p in in_place(r.site) {
            if !lines
                .iter()
                .any(|l| l.file == p.file && l.round == p.round && l.vars.contains(&r.name))
            {
                bad.push(format!(
                    "{}: {} ({}) is not a line of the list",
                    r.name, p.file, p.round
                ));
            }
        }
    }
    for l in &lines {
        let engine = l.file.starts_with("crates/") && l.file.contains("/src/");
        for v in &l.vars {
            match spec(v) {
                None if v.starts_with("BLOOMERY_") => {
                    bad.push(format!("{v}: the list names it in {}, no row does", l.file));
                }
                Some(r)
                    if engine
                        && r.is_lever()
                        && !in_place(r.site)
                            .iter()
                            .any(|p| p.file == l.file && p.round == l.round) =>
                {
                    bad.push(format!(
                        "{v}: the list reads it in place in {} ({}), its row does not",
                        l.file, l.round
                    ));
                }
                _ => {}
            }
        }
    }
    assert!(bad.is_empty(), "\n{}", bad.join("\n"));
}

/// The Markdown table has a line per row, six cells each; printed, it is
/// the text the repository's documentation of every name takes.
#[test]
fn markdown_is_a_line_per_row() {
    let md = markdown();
    let body: Vec<&str> = md.lines().skip(2).collect();
    assert_eq!(body.len(), REGISTRY.len());
    for (line, r) in body.iter().zip(REGISTRY) {
        assert!(line.starts_with(&format!("| `{}` |", r.name)), "{line}");
        assert_eq!(line.matches('|').count(), 7, "{line}");
    }
    println!("{md}");
}

/// `--levers` prints a line per lever with the reading's value: as set, the
/// default, or `-` for a lever the reading does not parse; a name that is no
/// lever has no line.
#[test]
fn table_is_a_line_per_lever() {
    let r = parsed().find(|r| !pool(r)).expect("a Parsed lever");
    let v = sample(r.kind);
    let acts_on = [r.name];
    let t = read(
        &env(&[(r.name, v.as_str())]),
        Scope::Main {
            bin: "b",
            acts_on: &acts_on,
        },
    )
    .unwrap_or_else(|e| panic!("{e}"))
    .table();
    let levers: Vec<&LeverSpec> = REGISTRY.iter().filter(|r| r.is_lever()).collect();
    assert_eq!(t.lines().count(), levers.len(), "{t}");
    for (line, row) in t.lines().zip(levers) {
        let mut cells = line.split_whitespace();
        assert_eq!(cells.next(), Some(row.name), "{line}");
        let value = line
            .split_whitespace()
            .skip(3)
            .collect::<Vec<_>>()
            .join(" ");
        let want = if row.name == r.name {
            format!("set {v}")
        } else if pool(row) {
            match row.default {
                Unset::Is(d) => format!("default {d}"),
                Unset::Means(m) => format!("unset: {m}"),
            }
        } else {
            "-".to_string()
        };
        assert!(value.starts_with(&want), "{line}");
    }
}

/// The residency grammar's one owner reads each model's derived word (a P no
/// list could name ahead) and the spares, and refuses S = 0.
/// Mutant: a closed word list — `mid-p33-s1` refused.
#[test]
fn residency_words_take_any_p_and_s_at_least_1() {
    assert_eq!(residency_word("off"), Some(ResidencyWord::Off));
    for (v, pinned, spares) in [
        ("mid-p0-s1", 0, 1),
        ("mid-p33-s1", 33, 1),
        ("mid-p138-s2", 138, 2),
    ] {
        assert_eq!(
            residency_word(v),
            Some(ResidencyWord::Mid { pinned, spares }),
            "{v}"
        );
        assert!(Kind::Residency.parse(v).is_ok(), "{v}");
    }
    assert_eq!(residency_word("mid-p33-s0"), None);
}

/// `BLOOMERY_MTP_HEAD_ROWS`'s word is never a path, and every other value
/// is one: `full` is the full head, `./full` a list named `full`.
#[test]
fn mtp_head_rows_takes_full_or_a_path() {
    let read_one = |v: &str| read(&env(&[(MTP_HEAD_ROWS, v)]), Scope::Every);
    let full = read_one("full").expect("full is the word");
    assert_eq!(full.mtp_head_rows(), Some(MtpHead::Full));
    let named = read_one("./full").expect("a path");
    assert_eq!(
        named.mtp_head_rows(),
        Some(MtpHead::List(Path::new("./full")))
    );
    assert!(read_one("").is_err(), "the empty value is no path");
}
