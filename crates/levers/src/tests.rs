use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

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
/// The rows read below every binary's `main` (the pool's two, the host's
/// room, the NVMe tier's four, [`nvtier_levers`], the tier link,
/// [`tier_link`], the tiers' nodes, [`tier_nodes`], the need hook,
/// [`need_levers`], and the cards' margin, [`card_margin`]), which no
/// `at_main` list refuses.
fn pool(r: &LeverSpec) -> bool {
    r.name == CARD_MARGIN
        || r.name == THREADS
        || r.name == SPIN
        || r.name == HOST_ROOM
        || r.name == NEED_HOOK
        || r.name == NVTIER_BYTES
        || r.name == NVTIER_READ
        || r.name == NVTIER_MEM
        || r.name == NVTIER_EVICT
        || r.name == TIER_LINK
        || r.name == TIER_NODES
}

/// The Direct rows `by` reads.
fn direct(by: Readers) -> impl Iterator<Item = &'static LeverSpec> {
    REGISTRY
        .iter()
        .filter(move |r| matches!(r.site, Site::Direct { by: b, .. } if b == by))
}

/// A binary's reading that names no lever.
const NO_LEVER: Scope<'static> = Scope::Main {
    bin: "b",
    acts_on: &[],
};

/// Every file that reads a lever in place, with its round.
fn in_place(site: Site) -> &'static [InPlace] {
    match site {
        Site::Parsed { left, .. } | Site::Retired { left, .. } => left,
        Site::Direct { at, .. } => at,
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
        Kind::CardBytes => "CUDA0=38G,cuda1=20G".into(),
        Kind::Path => "/data/set".into(),
        Kind::PathOr(words) => words[0].into(),
        Kind::File | Kind::Program => A_FILE.into(),
        Kind::FileAt => format!("{A_FILE}:0"),
        Kind::Text => "text".into(),
        Kind::Residency => "mid-p33-s1".into(),
        Kind::Addrs => "127.0.0.1:7340".into(),
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
        Kind::CardBytes => g.extend(
            [
                "38g",
                "17179869184G",
                "CUDA0=38G,",
                "CUDA0=38G,CUDA0=20G",
                "CUDA0=38G,cuda0=20G",
                "CUDA01=38G",
                "CUDA=38G",
                "CUDA-1=38G",
                "GPU0=38G",
                "0=38G",
                "CUDA0=",
                "CUDA0=38g",
                "CUDA0 =38G",
                "CUDA0=38G CUDA1=20G",
                "CUDA0=17179869184G",
                "CUDA4294967296=38G",
                "38G,CUDA1=20G",
                "CUDA0=3x,CUDA1=20G",
                "CUDA0=0x,CUDA1=0x",
                "CUDA0=x",
                "CUDA0=03x",
                "CUDA0=3x,CUDA0=1x",
                "3x",
            ]
            .map(String::from),
        ),
        Kind::File => g.extend([
            "/no/such/file".to_string(),
            env!("CARGO_MANIFEST_DIR").to_string(),
        ]),
        Kind::Program => g.extend([
            "/no/such/file".to_string(),
            env!("CARGO_MANIFEST_DIR").to_string(),
            "Cargo.toml".to_string(),
            "./Cargo.toml".to_string(),
            "sh".to_string(),
        ]),
        Kind::FileAt => g.extend([
            A_FILE.to_string(),
            format!("{A_FILE}:"),
            format!("{A_FILE}:-1"),
            format!("{A_FILE}:01"),
            format!("{A_FILE}: 1"),
            format!("{A_FILE}:one"),
            format!("{A_FILE}:{PAST_U64}"),
            format!("{A_FILE}:0:0"),
            "/no/such/file:0".to_string(),
            format!("{}:0", env!("CARGO_MANIFEST_DIR")),
            ":0".to_string(),
        ]),
        Kind::Path | Kind::PathOr(_) | Kind::Text => {}
        Kind::Addrs => g.extend(
            [
                "127.0.0.1",
                "localhost:7340",
                " 127.0.0.1:7340",
                "127.0.0.1:7340,",
                "127.0.0.1:7340,127.0.0.1:7340",
                "127.0.0.1:65536",
                // PIN(2026-10-17): private IPv4 now uses the node/RDMA requirement check.
                "8.8.8.8:7340",
                "0.0.0.0:7340",
            ]
            .map(String::from),
        ),
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
        site: Site::Parsed { left, .. },
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

/// `BLOOMERY_HOST_LANES`'s row is a parsed same-binary arm taking `flat` and
/// `ccd`, its default [`HOST_LANES_DEFAULT`] (what the host tier's dispatch
/// holds in a binary that does not act on it), and its accessor reads unset
/// as that default, `flat` as flat and `ccd` as ccd; any other word is refused
/// by name.
#[test]
fn host_lanes_row_is_a_flat_ccd_arm() {
    let row = REGISTRY.iter().find(|r| r.name == HOST_LANES);
    let Some(LeverSpec {
        class: Class::A,
        kind: Kind::Words(words),
        default: Unset::Is(d),
        site: Site::Parsed { left, .. },
        ..
    }) = row
    else {
        panic!("{HOST_LANES} is not a parsed word arm with a default: {row:?}");
    };
    assert_eq!(*words, ["flat", "ccd"]);
    assert!(left.is_empty(), "{HOST_LANES} is read in place: {left:?}");
    assert_eq!(
        *d,
        match HOST_LANES_DEFAULT {
            HostLanes::Flat => "flat",
            HostLanes::Ccd => "ccd",
        }
    );
    let at = |v: Option<&str>| {
        let e = v.map_or_else(Vec::new, |v| env(&[(HOST_LANES, v)]));
        read(&e, Scope::Every)
            .expect("flat, ccd and unset are taken")
            .host_lanes()
    };
    assert_eq!(at(None), HOST_LANES_DEFAULT);
    assert_eq!(at(Some("flat")), HostLanes::Flat);
    assert_eq!(at(Some("ccd")), HostLanes::Ccd);
    let Err(other) = read(&env(&[(HOST_LANES, "numa")]), Scope::Every) else {
        panic!("a word the row does not take is refused");
    };
    assert!(other.to_string().contains(HOST_LANES), "{other}");
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
    assert_eq!(unset.card_budget(), CardBudget::Unset);
    assert_eq!(unset.card_margin(), CardBudget::Unset);
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
    assert_eq!(unset.host_lanes(), HOST_LANES_DEFAULT);
    assert_eq!(unset.nvtier_mem(), NvTierMem::Auto);
    assert_eq!(unset.gen_slots(), 1);
    assert!(!unset.prepare_q8());
    assert_eq!(
        unset.host(),
        HostCfg {
            populate: true,
            lock: false,
            card_dontneed: true,
            r8: true,
            route_trace: false
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
            (CARD_MARGIN, "1G"),
            (PIN_MAIN, "0"),
            (DRAFT, "dspark"),
            (MTP_HEAD_ROWS, "/data/rows.txt"),
            (MTP_DRAFT, A_FILE),
            (MTP_WINDOWS, "1"),
            (WARM_PROFILE, "/data/warm.bin"),
            (ROUTE_TRACE, "/data/trace"),
            (CHECK_FINITE, "1"),
            (HOST_POPULATE, "0"),
            (HOST_LOCK, "1"),
            (CARD_DONTNEED, "0"),
            (R8, "off"),
            (RESIDENCY, "mid-p40-s1"),
            (QWEN38_EXPERTS, "card"),
            (HOSTSTREAM, "on"),
            (XSTREAM, "split"),
            (SPLIT_WALK, "off"),
            (QWEN3_KV, "q8_0"),
            (LANE_PREFETCH, "on"),
            (HOST_LANES, "flat"),
            (NVTIER_MEM, "pinned"),
            (NVTIER_EVICT, "heat"),
            (PREFILL_GROUP, "8"),
            (GEN_SLOTS, "2"),
            (PREPARE_Q8, "1"),
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
    assert_eq!(set.card_budget(), CardBudget::Every(38 << 30));
    assert_eq!(set.card_margin(), CardBudget::Every(1 << 30));
    assert!(!set.pin_main());
    assert_eq!(set.draft(), Some("dspark"));
    assert_eq!(
        set.mtp_head_rows(),
        Some(MtpHead::List(Path::new("/data/rows.txt")))
    );
    assert_eq!(set.mtp_draft(), Some(Path::new(A_FILE)));
    assert!(set.mtp_windows());
    assert_eq!(set.warm_profile(), Some(Path::new("/data/warm.bin")));
    assert_eq!(set.route_trace(), Some(Path::new("/data/trace")));
    assert!(set.check_finite());
    assert_eq!(set.residency(), Some("mid-p40-s1"));
    assert_eq!(set.qwen38_experts(), "card");
    assert_eq!(set.qwen38_experts_set(), Some("card"));
    assert_eq!(set.qwen3_kv_set(), Some("q8_0"));
    assert_eq!(set.hoststream(), Some(true));
    assert_eq!(set.xstream(), Some("split"));
    assert_eq!(set.split_walk(), Some(false));
    assert!(set.lane_prefetch());
    assert_eq!(set.host_lanes(), HostLanes::Flat);
    assert_eq!(set.nvtier_mem(), NvTierMem::Pinned);
    assert_eq!(set.nvtier_evict(), NvTierEvict::Heat);
    // Each word as the tier's build reads it, and whether it pins the arena
    // on an integrated device and on a card with its own memory.
    for (word, integrated, discrete) in [
        ("auto", true, false),
        ("anon", false, false),
        ("pinned", true, true),
    ] {
        let mem = read(&env(&[(NVTIER_MEM, word)]), Scope::NvTier)
            .expect("a word the row takes")
            .nvtier_mem();
        assert_eq!(
            (mem.pinned(true), mem.pinned(false)),
            (integrated, discrete),
            "{word}"
        );
    }
    assert_eq!(set.gen_slots(), 2);
    assert!(set.prepare_q8());
    assert_eq!(
        set.host(),
        HostCfg {
            populate: false,
            lock: true,
            card_dontneed: false,
            r8: false,
            route_trace: true
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
            CARD_MARGIN,
            HOST_ROOM,
            NVTIER_BYTES,
            NVTIER_READ,
            NVTIER_MEM,
            NVTIER_EVICT,
            TIER_LINK,
            TIER_NODES,
            TP_NODES,
            PIN_MAIN,
            DRAFT,
            MTP_HEAD_ROWS,
            MTP_DRAFT,
            MTP_WINDOWS,
            MTP_WIDTH,
            WARM_PROFILE,
            ROUTE_TRACE,
            CHECK_FINITE,
            HOST_POPULATE,
            HOST_LOCK,
            NEED_HOOK,
            CARD_DONTNEED,
            R8,
            RESIDENCY,
            QWEN38_EXPERTS,
            HOSTSTREAM,
            XSTREAM,
            SPLIT_WALK,
            QWEN3_KV,
            LANE_PREFETCH,
            HOST_LANES,
            GEN_SLOTS,
            FORCE_IDS,
            SCREEN,
            SCHEMAJUMP,
            PREPARE_Q8
        ]
    );
}

/// `BLOOMERY_SCHEMAJUMP` is on unset, `off` is the step arm, and a word that is
/// neither is refused by name.
#[test]
fn schemajump_is_on_unless_off() {
    let set = read(&env::<&str>(&[]), Scope::Every).unwrap();
    assert!(set.schema_jump());
    let set = read(&env(&[(SCHEMAJUMP, "on")]), Scope::Every).unwrap();
    assert!(set.schema_jump());
    let set = read(&env(&[(SCHEMAJUMP, "off")]), Scope::Every).unwrap();
    assert!(!set.schema_jump());
    let Err(e) = read(&env(&[(SCHEMAJUMP, "0")]), Scope::Every) else {
        panic!("a word the row does not take is refused");
    };
    assert!(e.to_string().contains("BLOOMERY_SCHEMAJUMP"), "{e}");
}

/// `BLOOMERY_MTP_WIDTH` takes its two words, unset is no word (a run that
/// drafts reads it as `cost`), and another word is refused by name.
#[test]
fn mtp_width_takes_cost_and_fixed() {
    for w in ["cost", "fixed"] {
        let set = read(&env(&[(MTP_WIDTH, w)]), Scope::Every).expect("a word the row takes");
        assert_eq!(set.mtp_width(), Some(w));
    }
    let set = read(&env::<&str>(&[]), Scope::Every).unwrap();
    assert_eq!(set.mtp_width(), None);
    let Err(other) = read(&env(&[(MTP_WIDTH, "wide")]), Scope::Every) else {
        panic!("a word the row does not take is refused");
    };
    assert!(other.to_string().contains("BLOOMERY_MTP_WIDTH"), "{other}");
}

/// The need hook and the lock the need record states, read below every
/// binary's `main` ([`Scope::Need`]): unset is no hook and no lock, a regular
/// file is the hook, a path with no regular file is refused by name, and the
/// lock is the host-lock flag as its kind takes it. A binary's `main` takes
/// the hook without naming it and refuses the lock it does not act on.
#[test]
fn the_need_hook_is_read_below_main() {
    let unset = read(&env::<&str>(&[]), Scope::Need).expect("nothing is set");
    assert_eq!(unset.entry(NEED_HOOK).value, None);
    assert!(!unset.flag(HOST_LOCK));
    let set = read(&env(&[(NEED_HOOK, A_FILE), (HOST_LOCK, "1")]), Scope::Need)
        .expect("a regular file and a flag");
    assert_eq!(
        set.entry(NEED_HOOK).value,
        Some(Value::Path(PathBuf::from(A_FILE)))
    );
    assert!(set.flag(HOST_LOCK));
    for bad in ["/no/such/file", env!("CARGO_MANIFEST_DIR"), ""] {
        let Err(e) = read(&env(&[(NEED_HOOK, bad)]), Scope::Need) else {
            panic!("{bad:?} is no regular file");
        };
        assert!(e.to_string().contains(NEED_HOOK), "{e}");
    }
    // A program is run by its absolute path: a name with no directory is
    // searched in `PATH` and a relative one in the working directory of the
    // moment, so a file the lever finds by either may not be the one that runs.
    for relative in ["Cargo.toml", "./Cargo.toml", "src/../Cargo.toml"] {
        let Err(e) = read(&env(&[(NEED_HOOK, relative)]), Scope::Need) else {
            panic!("{relative:?} is a relative name");
        };
        assert!(e.to_string().contains("absolute path"), "{e}");
    }
    let Err(e) = read(&env(&[(HOST_LOCK, "yes")]), Scope::Need) else {
        panic!("a lock the flag does not take is refused");
    };
    assert!(e.to_string().contains(HOST_LOCK), "{e}");
    // Only the two names are in this scope: a bad value of another is not read.
    assert!(read(&env(&[(HOST_ROOM, "not bytes")]), Scope::Need).is_ok());
    // A binary that acts on nothing still takes the hook, and refuses the lock.
    let main = Scope::Main {
        bin: "b",
        acts_on: &[],
    };
    assert!(read(&env(&[(NEED_HOOK, A_FILE)]), main).is_ok());
    let Err(e) = read(&env(&[(HOST_LOCK, "1")]), main) else {
        panic!("a binary that leaves the lock out refuses it");
    };
    assert!(e.to_string().contains(HOST_LOCK), "{e}");
}

/// `BLOOMERY_NVTIER_EVICT` is read where the tier is built
/// ([`Scope::NvTier`]): unset is `tick`, main's rule, each word is its rule,
/// and any other word is refused by name with the words the row takes.
#[test]
fn nvtier_evict_takes_tick_or_heat_and_refuses_the_rest() {
    let evict = |pairs: &[(&str, &str)]| read(&env(pairs), Scope::NvTier).map(|l| l.nvtier_evict());
    assert_eq!(evict(&[]).unwrap(), NvTierEvict::Tick);
    assert_eq!(evict(&[(NVTIER_EVICT, "tick")]).unwrap(), NvTierEvict::Tick);
    assert_eq!(evict(&[(NVTIER_EVICT, "heat")]).unwrap(), NvTierEvict::Heat);
    for bad in ["lru", "HEAT", ""] {
        let Err(e) = evict(&[(NVTIER_EVICT, bad)]) else {
            panic!("{bad:?} is not a word of the row and must be refused");
        };
        let text = e.to_string();
        assert!(
            text.contains(NVTIER_EVICT) && text.contains("one of tick, heat"),
            "{text}"
        );
    }
}

/// `BLOOMERY_CARD_BUDGET` takes one byte count for every card or one per
/// card by CUDA ordinal, prints back as it is spelled, refuses a card named
/// twice, a key that names no device and bytes that are none by the entry,
/// and is read alone where a placement meets the census
/// ([`Scope::CardBudget`]).
#[test]
fn card_budget_takes_one_value_or_one_per_card() {
    assert_eq!(card_budget_value("38G"), Ok(CardBudget::Every(38 << 30)));
    let per = card_budget_value("CUDA0=38G,cuda1=20480M").expect("a list");
    assert_eq!(per, CardBudget::Per(vec![(0, 38 << 30), (1, 20 << 30)]));
    assert_eq!(per.to_string(), "CUDA0=40802189312,CUDA1=21474836480");
    assert_eq!(card_budget_value(&per.to_string()), Ok(per.clone()));
    assert_eq!(per.every(), None);
    assert!(per.is_set() && !CardBudget::Unset.is_set());
    assert_eq!(CardBudget::from(Some(7)), CardBudget::Every(7));
    assert_eq!(CardBudget::from(None), CardBudget::Unset);
    for (v, why) in [
        ("CUDA0=38G,CUDA0=20G", "CUDA0 is given twice"),
        ("CUDA2=1G,cuda2=2G", "CUDA2 is given twice"),
        ("GPU0=38G", "\"GPU0\" names no device as CUDA<N>"),
        ("CUDA0=38g", "\"38g\" for CUDA0 is no bytes"),
        ("CUDA0=38G,", "\"\" is no CUDA<N>=<bytes> entry"),
    ] {
        let e = read(&env(&[(CARD_BUDGET, v)]), Scope::Every)
            .expect_err(v)
            .to_string();
        assert!(e.contains(CARD_BUDGET) && e.contains(why), "{v}: {e}");
    }
    let alone = read(
        &env(&[(CARD_BUDGET, "CUDA1=20G"), (THREADS, "nope")]),
        Scope::CardBudget,
    )
    .expect("only the card budget is read");
    assert_eq!(alone.card_budget(), CardBudget::Per(vec![(1, 20 << 30)]));
}

/// `BLOOMERY_CARD_BUDGET` also takes shares, `CUDA<N>=<n>x` per card: whole
/// parts, a list all bytes or all shares, one share above 0 at least, printed
/// back as spelled.
#[test]
fn card_budget_takes_shares_per_card() {
    let shares = card_budget_value("CUDA0=3x,cuda1=1X").expect("shares");
    assert_eq!(shares, CardBudget::Shares(vec![(0, 3), (1, 1)]));
    assert_eq!(shares.to_string(), "CUDA0=3x,CUDA1=1x");
    assert_eq!(card_budget_value(&shares.to_string()), Ok(shares.clone()));
    assert_eq!(shares.every(), None);
    assert_eq!(shares.listed(), Some(&[(0, 3), (1, 1)][..]));
    assert!(shares.is_set());
    let held = card_budget_value("CUDA1=1x,CUDA0=0x").expect("a zero share beside a share");
    assert_eq!(held, CardBudget::Shares(vec![(1, 1), (0, 0)]));
    assert_eq!(
        card_budget_value("CUDA0=1G").expect("bytes").listed(),
        Some(&[(0, 1 << 30)][..])
    );
    assert_eq!(CardBudget::Every(1).listed(), None);
    for (v, why) in [
        (
            "CUDA0=3x,CUDA1=20G",
            "CUDA1: a list is all bytes or all shares",
        ),
        (
            "CUDA0=20G,CUDA1=3x",
            "CUDA1: a list is all bytes or all shares",
        ),
        ("CUDA0=0x,CUDA1=0x", "every share is 0"),
        ("CUDA0=x", "\"x\" for CUDA0 is no share"),
        ("CUDA0=03x", "\"03x\" for CUDA0 is no share"),
        ("CUDA0=1.5x", "\"1.5x\" for CUDA0 is no share"),
        ("CUDA0=3x,CUDA0=1x", "CUDA0 is given twice"),
        (
            "CUDA0=18446744073709551615x,CUDA1=1x",
            "the shares sum to more than a u64 holds",
        ),
    ] {
        let e = read(&env(&[(CARD_BUDGET, v)]), Scope::Every)
            .expect_err(v)
            .to_string();
        assert!(e.contains(CARD_BUDGET) && e.contains(why), "{v}: {e}");
    }
}

/// `BLOOMERY_CARD_MARGIN` reads the card budget's spelling for the bytes a
/// card keeps free, is acted on by every binary, is read with the card
/// budget where a placement meets the census, and unset is none.
#[test]
fn card_margin_is_read_with_the_card_budget_and_acted_on_by_every_binary() {
    let main = |pairs: &[(&str, &str)]| {
        read(
            &env(pairs),
            Scope::Main {
                bin: "x",
                acts_on: &[],
            },
        )
        .expect("read")
    };
    assert_eq!(main(&[]).card_margin(), CardBudget::Unset);
    assert_eq!(
        main(&[(CARD_MARGIN, "1G")]).card_margin(),
        CardBudget::Every(1 << 30)
    );
    let list = main(&[(CARD_MARGIN, "CUDA0=512M,CUDA1=2G")]).card_margin();
    assert_eq!(list, CardBudget::Per(vec![(0, 512 << 20), (1, 2 << 30)]));
    let alone = read(
        &env(&[(CARD_MARGIN, "CUDA1=1G"), (THREADS, "nope")]),
        Scope::CardBudget,
    )
    .expect("only the card levers are read");
    assert_eq!(alone.card_margin(), CardBudget::Per(vec![(1, 1 << 30)]));
    assert_eq!(alone.card_budget(), CardBudget::Unset);
    let e = read(&env(&[(CARD_MARGIN, "lots")]), Scope::Every)
        .expect_err("not bytes")
        .to_string();
    assert!(e.contains(CARD_MARGIN), "{e}");
}

/// `BLOOMERY_HOST_ROOM` takes bytes (`M` and `G` binary units) where the
/// host's reading is made ([`Scope::NvTier`]), unset is no room given, and a
/// value that is not bytes is refused by name.
#[test]
fn host_room_takes_bytes_and_refuses_the_rest() {
    let room = |v: &str| {
        read(&env(&[(HOST_ROOM, v)]), Scope::NvTier).map(|l| l.entry(HOST_ROOM).value.clone())
    };
    assert_eq!(room("27G").unwrap(), Some(Value::Bytes(27 << 30)));
    assert_eq!(room("24176M").unwrap(), Some(Value::Bytes(25_350_373_376)));
    let unset = read(&env::<&str>(&[]), Scope::NvTier).unwrap();
    assert_eq!(unset.entry(HOST_ROOM).value, None);
    for bad in ["27GB", "lots", "", "-1"] {
        let Err(e) = room(bad) else {
            panic!("{bad:?} is not bytes and must be refused");
        };
        assert!(e.to_string().contains(HOST_ROOM), "{e}");
    }
}

/// `BLOOMERY_QWEN3_KV` takes the two cache-format words, and unset reads as
/// unset: the seat, not the lever, picks the format then.
#[test]
fn qwen3_kv_takes_f16_and_q8_0() {
    let unset = read(&[], Scope::Every).expect("nothing is set");
    assert_eq!(unset.qwen3_kv_set(), None);
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
/// (`levers_refuse_what_their_kind_does_not_take` holds the rest).
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
            "mid-p0-s1",
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
/// on plan (a) `mid-p<P>-s1`, no seed expert pinned where the host takes
/// the churn pool at 0 and the first P whose pool fits where it does not,
/// and `off` with why when no layer holds one, when the fewest leave no room,
/// and when no pool the card side leaves fits the plan's host headroom or
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
        // Before the plan the load's residency is `off` where the unset rule turns it so, and
        // a set word is as given. FAIL-first: the stream staging asked only a set `off`, and
        // the route trace and the step feed reserved staging no stream reads.
        assert_eq!(residency38_off(None, at), want.is_some(), "{at:?}");
        assert!(residency38_off(Some("off"), at), "{at:?}");
        assert!(!residency38_off(Some("mid-p4-s1"), at), "{at:?}");
    }

    // The pool is 10 B an expert past the pinned ones, on two layers of the
    // fewest; the headroom takes it or does not.
    let pool = |fewest: usize| move |p: usize| Ok::<u64, ()>(2 * 10 * (fewest - p) as u64);
    // The host holds one expert unless a case says otherwise.
    let pick_mem = |n_l: &[u64], headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        residency38_at_plan(
            UnhostedPool::Carried,
            n_l.iter().copied(),
            1,
            pool(fewest),
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let pick = |n_l: &[u64], headroom: i128| pick_mem(n_l, headroom, 1 << 40);
    let derived = pick(&[0, 297, 298, 0, 297], 1 << 40);
    assert_eq!(
        derived,
        Residency38Pick {
            pinned: Some(0),
            why: Residency38Why::PlanA { fewest: 297 }
        }
    );
    assert_eq!(derived.word(), "mid-p0-s1");
    assert_eq!(
        derived.why.to_string(),
        "unset: plan (a), no seed expert pinned (the plan's fewest card experts a layer is 297)"
    );
    let row = spec(RESIDENCY).expect("the row");
    row.kind
        .parse(&derived.word())
        .unwrap_or_else(|e| panic!("the derived word at the plain plan's count: {e}"));
    assert_eq!(pick(&[3, 4], 1 << 40).word(), "mid-p0-s1");
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
    // A plan that leaves the host no routed expert: the pool would serve
    // none, whatever room the card and the host have.
    let all_on_cards = residency38_at_plan(
        UnhostedPool::Carried,
        [512u64, 512],
        0,
        pool(512),
        1 << 40,
        1 << 40,
    )
    .expect("no pool error");
    assert_eq!(
        all_on_cards,
        Residency38Pick {
            pinned: None,
            why: Residency38Why::NoHostExperts
        }
    );
    assert_eq!(all_on_cards.word(), "off");
    assert_eq!(
        all_on_cards.why.to_string(),
        "unset: the plan holds every routed expert on a card; a churn pool would serve none"
    );
    assert_eq!(
        residency38_at_plan(
            UnhostedPool::Carried,
            [0u64, 0],
            0,
            pool(0),
            1 << 40,
            1 << 40
        )
        .expect("no pool error")
        .why,
        Residency38Why::NoCardExperts,
        "no card expert is the first why"
    );
    // A plan whose one card layer holds a single expert: the word's P 0
    // needs it less the spares and one that moves — no room. At two experts
    // the unpinned default fits.
    assert_eq!(
        pick(&[1, 9], 1 << 40),
        Residency38Pick {
            pinned: None,
            why: Residency38Why::NoRoom { fewest: 1 }
        }
    );
    assert_eq!(
        pick(&[2, 9], 1 << 40).pinned,
        Some(0),
        "the unpinned default fits a plan of two card experts a layer"
    );
    // PIN(2026-10-13): was P 0 with the pool unhosted below the budget; an unset word now climbs on every body.
    // The churn pool at P 0 is 297 experts a layer, 2 layers, 10 B each:
    // 5,940 B, and 20 × (297 − P) B at P. A headroom of 2,980 B takes it from
    // P 148 (20 × 149), MemAvailable of 2,979 B from P 149: an unset word
    // climbs to the first P whose pool fits, whatever the body carries.
    let raised = |to: usize| Residency38Pick {
        pinned: Some(to),
        why: Residency38Why::PoolShrunk { from: 0, to },
    };
    let kept = pick(&[297, 297], 2_980);
    assert_eq!(kept, raised(148));
    assert_eq!(kept.word(), "mid-p148-s1");
    assert_eq!(kept.pool(), ChurnHosting::Hosted);
    assert_eq!(
        kept.why.to_string(),
        "unset: p0→p148 (host: the churn pool did not fit, raised)"
    );
    assert_eq!(pick_mem(&[297, 297], 1 << 40, 2_979), raised(149));
    // A negative budget takes no count: `off`, at the last pool probed (the
    // card ceiling 295's, 40 B).
    assert_eq!(
        pick(&[297, 297], -5),
        Residency38Pick {
            pinned: None,
            why: Residency38Why::PoolUnhostable(PoolUnhostable {
                needs: 40,
                headroom: -5,
                mem_left: 1 << 40,
            }),
        }
    );
    // The pool that fits both budgets is hosted.
    let fits = pick_mem(&[297, 297], 5_940, 5_940);
    assert_eq!(fits.pinned, Some(0));
    assert_eq!(fits.pool(), ChurnHosting::Hosted);
    assert_eq!(
        residency38_at_plan(
            UnhostedPool::Carried,
            [297u64],
            1,
            |_| Err::<u64, &str>("the pool"),
            0,
            0
        ),
        Err("the pool"),
        "the pool's error is the call's"
    );
}

/// A paged plan's unset residency resolves by each rule's own terms: its
/// stacks leave the churn pool empty at every P (a paged stack's victims are
/// the arena's, `ChurnPool::of`), so the Qwen3.8 word is `mid-p0-s1` while the
/// plan's host headroom and `MemAvailable` are not negative, and `off` with
/// the host's why where one is. Nothing about the tier's arena is an input.
// PIN(2026-10-11): the paged plan's pick was `off`, why `paged`, under a
// plan-time clause that read the tier's arena; a promotion reads the arena's
// slots, not the file mapping, so the clause is gone. Derivation: pool 0 at
// every P, the headroom 0 or more: `room_for`'s host side fits at P 0.
// FAIL-first: a `room_for` that turns a pool-0 plan `off` reds it.
#[test]
fn residency38_at_plan_resolves_a_paged_plan_by_its_own_rule() {
    let big = 1 << 40;
    let empty = |_: usize| Ok::<u64, ()>(0);
    let at = |headroom: i128, mem_left: i128| {
        residency38_at_plan(
            UnhostedPool::Carried,
            [297u64, 298],
            1,
            empty,
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let planned = Residency38Pick {
        pinned: Some(0),
        why: Residency38Why::PlanA { fewest: 297 },
    };
    assert_eq!(at(0, big), planned);
    assert_eq!(at(big, 0), planned);
    // PIN(2026-10-13): was P 0 unhosted; an unset word now climbs on every body.
    // A negative budget does not take even an empty pool, at any count:
    // `off`, at the last pool probed.
    assert_eq!(
        at(-1, big).why,
        Residency38Why::PoolUnhostable(PoolUnhostable {
            needs: 0,
            headroom: -1,
            mem_left: big,
        })
    );
    assert_eq!(at(big, -1).pinned, None);
    assert_eq!(
        room_for(false, 0, 1, [297u64, 298], empty, 0, big),
        Ok(Room::AsIs),
        "the room is the same rule under every family"
    );
}

/// The least fewest card experts a layer ([`residency_least_fewest`]) is
/// where [`room_for`] stops saying `NoRoom`, for every target and spares: the
/// count a default context's fit leaves the card, read from the rule's own
/// card side. The Qwen3.8 unset word (P 0, one spare) needs two
/// ([`residency38_least_fewest`]), and a plan of that many resolves the word
/// where one of one fewer resolves `off`.
/// FAIL-first: a least fewest of `target + spares`, one under what the card
/// side holds, turns the grid red at target 0 and one spare; a
/// `RESIDENCY38_PINNED` moved from 0 turns the Qwen3.8 pins at two red.
#[test]
fn the_least_fewest_is_where_the_card_side_holds_the_word() {
    let room = |target: usize, spares: usize, fewest: u64| {
        room_for(
            false,
            target,
            spares,
            [fewest],
            |_| Ok::<u64, ()>(0),
            1 << 40,
            1 << 40,
        )
        .expect("no pool error")
    };
    for target in 0..6 {
        for spares in 0..4 {
            let least = residency_least_fewest(target, spares);
            for fewest in 1..=least as u64 + 3 {
                let no_room = matches!(
                    room(target, spares, fewest),
                    Room::Short(RoomShort::NoRoom { .. })
                );
                assert_eq!(
                    no_room,
                    (fewest as usize) < least,
                    "target {target}, spares {spares}, fewest {fewest}, least {least}"
                );
            }
        }
    }
    // A target past what the fewest holds runs at half of it: 3 is the least
    // that leaves one pinned (the room test's `[3]` row).
    assert_eq!(
        residency_least_fewest(40, 1),
        3,
        "three card experts leave one pinned beside a spare and one that moves"
    );
    assert_eq!(
        residency_least_fewest(0, RESIDENCY38_SPARES),
        2,
        "the unset word's P 0 needs its spare and one that moves"
    );
    let least = residency38_least_fewest();
    assert_eq!(least, 2, "the Qwen3.8 word's own least fewest");
    let word_at = |fewest: u64| {
        residency38_at_plan(
            UnhostedPool::Carried,
            [fewest, fewest],
            1,
            |_| Ok::<u64, ()>(0),
            1 << 40,
            1 << 40,
        )
        .expect("no pool error")
        .pinned
    };
    assert_eq!(word_at(least - 1), None, "under the least the word is off");
    assert_eq!(word_at(least), Some(0), "at the least the word runs");
}

/// The plan's room for a family's target word ([`room_for`]): the word at
/// its target where its card slots and its host take it, at half the fewest
/// where its card slots do not, at that P with the pool unhosted where the
/// word asks for it, `off` where the card slots leave no count.
#[test]
fn room_for_keeps_the_word_where_the_plan_has_room() {
    // The pool shrinks as the pinned count grows: 10 B an expert past it, on
    // layers of the fewest's count.
    let pool = |fewest: usize| move |p: usize| Ok::<u64, ()>(10 * (fewest - p) as u64);
    let room = |target: usize, spares: usize, n_l: &[u64], headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        room_for(
            false,
            target,
            spares,
            n_l.iter().copied(),
            pool(fewest),
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let big = 1 << 40;
    // The word's target where its card slots hold it, its spares and one
    // that moves — equality included.
    assert_eq!(room(40, 1, &[42, 0, 43], big, big), Room::AsIs);
    assert_eq!(room(0, 1, &[2], big, big), Room::AsIs, "a target of 0 runs");
    // Half the fewest where the target does not fit: 29/2 = 14.
    for (n_l, to) in [
        (&[29u64, 30][..], 14usize),
        (&[30u64][..], 15),
        (&[3u64][..], 1),
    ] {
        assert_eq!(
            room(40, 1, n_l, big, big),
            Room::Moved {
                fewest: n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize,
                from: 40,
                to,
            },
            "{n_l:?}"
        );
    }
    // No half the card slots hold: `off`, naming the fewest.
    assert_eq!(
        room(40, 1, &[2], big, big),
        Room::Short(RoomShort::NoRoom { fewest: 2 })
    );
    for n_l in [&[0u64, 0][..], &[]] {
        assert_eq!(
            room(40, 1, n_l, big, big),
            Room::Short(RoomShort::NoCardExperts)
        );
    }
    // PIN(2026-10-13): was `Room::Unhosted` for an unset word on a carrying body; an unset word now climbs on every
    // body, and the asked word's room is pinned by
    // `a_word_that_asks_keeps_its_pool_unhosted_on_a_body_that_carries_it`.
    // An unset word climbs on a body that carries an unhosted pool too: the
    // pool at 40 (10 experts, 100 B) fits neither budget, and the first count
    // above whose pool fits raises P, never past the card ceiling 48.
    assert_eq!(
        room(40, 1, &[50], 50, big),
        Room::MovedForPool { from: 40, to: 45 }
    );
    assert_eq!(
        room(40, 1, &[50], big, 99),
        Room::MovedForPool { from: 40, to: 41 }
    );
    assert_eq!(room(40, 1, &[50], 100, 100), Room::AsIs);
    assert_eq!(
        room_for(false, 40, 1, [42u64], |_| Err(()), 0, 0),
        Err(()),
        "the pool's error is the call's"
    );
    // PIN(2026-10-13): a load that hosts every pool it runs climbs as the
    // rule did before the unhosted pool (`Room::Moved` by the host side,
    // `HostShort`, `MemShort`); the host side's moves and refusals now have
    // their own variants. The pool at the word's P fits neither budget, and
    // the first count above it whose pool fits raises it — 50 − 45 = 5
    // experts, 50 B — never past the card side's ceiling 48.
    let climb = |n_l: &[u64], headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        room_for(
            false,
            40,
            1,
            n_l.iter().copied(),
            pool(fewest),
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let raised = Room::MovedForPool { from: 40, to: 45 };
    assert_eq!(climb(&[50], 50, big), raised);
    assert_eq!(climb(&[50], big, 50), raised);
    assert_eq!(climb(&[50], 100, 100), Room::AsIs);
    // No count the card side leaves fits the host (the ceiling 48's pool is
    // 20 B): `off` at the last pool probed, naming both budgets.
    let unhostable = |headroom, mem_left| {
        Room::Short(RoomShort::PoolUnhostable(PoolUnhostable {
            needs: 20,
            headroom,
            mem_left,
        }))
    };
    assert_eq!(climb(&[50], 19, big), unhostable(19, big));
    assert_eq!(climb(&[50], big, 19), unhostable(big, 19));
    // The card side moves P the same under both: half of 29 is 14.
    assert_eq!(
        climb(&[29], big, big),
        Room::Moved {
            fewest: 29,
            from: 40,
            to: 14
        }
    );
}

/// [`RESIDENCY`] unset on a V4.1 plan ([`residency_at_plan`]): the serving
/// word `mid-p0-s1` wherever the plan's card slots hold the rule at all —
/// its P 0 fits every plan but a one-expert layer — raised to the first P
/// whose churn pool fits where the host does not take the pool at 0, `off`
/// where neither side leaves a count or the plan holds no host expert; a set
/// word and every other why pass through untouched.
#[test]
fn residency_at_plan_moves_the_unset_word_to_what_the_plan_leaves() {
    let pool = |fewest: usize| move |p: usize| Ok::<u64, ()>(10 * (fewest - p) as u64);
    let unset = ResidencyPick {
        word: RESIDENCY_SERVING,
        why: ResidencyWhy::Place,
    };
    let at_host = |n_l: &[u64], host: u64, headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        residency_at_plan(
            UnhostedPool::Carried,
            unset,
            n_l.iter().copied(),
            host,
            pool(fewest),
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let at = |n_l: &[u64], headroom: i128, mem_left: i128| at_host(n_l, 30, headroom, mem_left);
    let big = 1 << 40;
    // The serving word wherever the plan's layers hold the rule.
    assert_eq!(at(&[42, 43], big, big), unset);
    assert_eq!(at(&[42, 43], big, big).why.name(), "place");
    assert_eq!(at(&[29, 30], big, big), unset);
    assert_eq!(at(&[3], big, big), unset);
    // A plan whose one card layer holds a single expert: the word's P 0
    // needs it less the spares and one that moves — no room.
    assert_eq!(
        at(&[1, 9], big, big),
        ResidencyPick {
            word: "off",
            why: ResidencyWhy::NoRoom { fewest: 1 }
        }
    );
    assert_eq!(at(&[0, 0], big, big).why, ResidencyWhy::NoCardExperts);
    // A plan whose card holds every expert has none on the host for the
    // machine to move in: its spares would only displace card experts, so the
    // word is `off` with why, the card check first (a plan with no expert at
    // all is the card's).
    let whole = at_host(&[16, 16, 16, 16], 0, big, big);
    assert_eq!(
        (whole.word, whole.why),
        ("off", ResidencyWhy::NoHostExperts)
    );
    assert_eq!(whole.why.name(), "no_host_experts");
    assert_eq!(
        whole.why.detail().expect("the no-host why details"),
        "unset: the plan holds no routed expert on the host"
    );
    assert_eq!(
        at_host(&[0, 0], 0, big, big).why,
        ResidencyWhy::NoCardExperts
    );
    assert_eq!(at_host(&[16, 16], 1, big, big), unset);
    assert_eq!(
        at(&[1, 9], big, big)
            .why
            .detail()
            .expect("the no-room why details"),
        "unset: the plan's fewest card experts a layer (1) leave no room for the word's P or \
         half of them"
    );
    assert_eq!(
        ResidencyWhy::Place.detail(),
        None,
        "a why the plan did not move details nothing"
    );
    // PIN(2026-10-13): the keep-P block (P 0 with the pool unhosted below the budget) is gone; an unset word now climbs on
    // every body, pinned by `an_unset_word_climbs_on_a_body_that_carries_an_unhosted_pool`.
    // A set word and every other why pass through untouched, the pool unasked.
    let refused = |pick: ResidencyPick| {
        residency_at_plan(
            UnhostedPool::Carried,
            pick,
            [30u64],
            0,
            |_| Err::<u64, ()>(()),
            0,
            0,
        )
        .expect("no pool error")
    };
    assert_eq!(
        refused(ResidencyPick {
            word: RESIDENCY_SERVING,
            why: ResidencyWhy::Set,
        })
        .word,
        RESIDENCY_SERVING
    );
    assert_eq!(
        refused(ResidencyPick {
            word: "off",
            why: ResidencyWhy::FixedPlace,
        })
        .why,
        ResidencyWhy::FixedPlace
    );
}

// PIN(2026-11-01): residency_beside_paged_planes_turns_the_unset_word_off left with its rule: the
// NVMe arena fills a stack's NVFP4 planes, so a paged plan keeps the serving residency word.

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
        looked: &[],
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
            Some(Draft38Off::NoFile(vec![file.to_path_buf()])),
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
    // A caller that already walked the pick supplies that list. An empty
    // `looked` is the one-path fallback above; this is the other branch.
    let paths = vec![PathBuf::from("/links/mtp.gguf"), file.to_path_buf()];
    assert_eq!(
        draft38_unset(&Draft38At {
            looked: &paths,
            ..at(true, false, false, false, 1)
        }),
        Some(Draft38Off::NoFile(paths.clone()))
    );
    assert_eq!(
        Draft38Off::NoFile(vec![file.to_path_buf()]).to_string(),
        "no file at /models/q/mtp.gguf"
    );
    assert_eq!(
        Draft38Off::NoFile(vec![
            PathBuf::from("/links/mtp.gguf"),
            PathBuf::from("/real/mtp.gguf"),
            file.to_path_buf(),
        ])
        .to_string(),
        "no file at /links/mtp.gguf, /real/mtp.gguf, /models/q/mtp.gguf"
    );
    assert_eq!(
        Draft38Off::Borrowed {
            name: "output.weight".to_string(),
            ty: "f16".to_string()
        }
        .to_string(),
        "unset: the target's output.weight is f16, not a form the MTP draft reads"
    );
}

/// The unset draft's yield to the context ([`RungCost::of`]): it fires
/// when the drafted choice leaves a slot half the plain one's positions or
/// fewer, and only then; a draft that costs the fitted context a few steps,
/// or no position, stays on.
///
/// The card figures [measured, the 0.2.8 candidate on a RTX 3060 12 GB at
/// the serving defaults, logs sih-smoke028b (draft on: fit 702, the plan's
/// draft reserve 2,842,413,228 B) and sih-smoke028c (draft off: fit 6165)].
#[test]
fn draft_yields_to_the_context_only_when_it_costs_positions() {
    let r = 2_842_413_228_u64;
    let y = RungCost::of(Reserve::Draft, 702, 6165, r).expect("the draft yields");
    assert_eq!(
        y,
        RungCost {
            reserve: Reserve::Draft,
            with: 702,
            without: 6165,
            card_bytes: r,
        }
    );
    assert_eq!(
        y.to_string(),
        "unset: the draft's 2842413228 B on the card leave a slot 702 positions; without the \
         draft a slot takes 6165"
    );
    assert_eq!(Draft38Off::Yield(y).to_string(), y.to_string());
    // A draft that denies no position (both choices at one rung) stays on.
    assert!(
        RungCost::of(Reserve::Draft, 131_072, 131_072, r).is_none(),
        "same rung"
    );
    // A drafted choice that holds not even the floor yields to any plain one.
    assert!(
        RungCost::of(Reserve::Draft, 0, 4096, r).is_some(),
        "under the floor"
    );
    // Half the positions yields; a few steps of a fitted context do not.
    assert!(
        RungCost::of(Reserve::Draft, 65_536, 131_072, r).is_some(),
        "half"
    );
    assert!(
        RungCost::of(Reserve::Draft, 65_537, 131_072, r).is_none(),
        "past half"
    );
    assert!(
        RungCost::of(Reserve::Draft, 254_976, 262_144, r).is_none(),
        "a few steps"
    );
}

/// The image encoder's reserve names itself on the `ctx` line whenever it
/// lowers the context ([`RungCost::of`]), by a step or more: a reserve
/// that denies no position says nothing, and the encoder stays on either way.
#[test]
fn a_vision_reserve_names_itself_only_when_it_lowers_the_rung() {
    let bytes = 1_234_567_890_u64;
    let y = RungCost::of(Reserve::ImageEncoder, 32_768, 65_536, bytes)
        .expect("the reserve lowers the rung");
    assert_eq!(
        y,
        RungCost {
            reserve: Reserve::ImageEncoder,
            with: 32_768,
            without: 65_536,
            card_bytes: bytes,
        }
    );
    assert_eq!(
        y.to_string(),
        "the image encoder's 1234567890 B on the card leave a slot 32768 positions; without the \
         encoder a slot takes 65536"
    );
    assert!(
        RungCost::of(Reserve::ImageEncoder, 31_744, 32_768, bytes).is_some(),
        "one step"
    );
    assert!(
        RungCost::of(Reserve::ImageEncoder, 32_768, 32_768, bytes).is_none(),
        "same rung"
    );
    assert!(
        RungCost::of(Reserve::ImageEncoder, 65_536, 32_768, bytes).is_none(),
        "a higher rung"
    );
}

/// The paged rule ([`paged_columns`]): with no arena, or a draft that is off
/// or set, the plan serves as asked; on a paged plan an unset draft goes off.
///
/// The arena is the 3090 gate plan's at a 27 GiB room [derived on the Mac:
/// `crates/placement` `a_27_gib_room_puts_the_overflow_on_the_nvme_tier`].
#[test]
fn a_paged_plan_turns_only_an_unset_draft_off() {
    let arena = 23_434_203_136_u64;
    let at = |arena, draft, draft_set| PagedAt {
        arena,
        draft,
        draft_set,
    };
    for (at, want) in [
        // No arena: as asked, draft or not.
        (at(0, true, false), Paged::AsAsked),
        (at(0, true, true), Paged::AsAsked),
        // No draft: as asked.
        (at(arena, false, false), Paged::AsAsked),
        (at(arena, false, true), Paged::AsAsked),
        // The unset draft goes off.
        (at(arena, true, false), Paged::Reduced),
        // A set draft runs as asked.
        (at(arena, true, true), Paged::AsAsked),
    ] {
        assert_eq!(paged_columns(&at), want, "{at:?}");
    }
    // The paged plan's draft ([`paged_draft`]): a set draft runs; unset, a
    // draft that would run, or that only its yield to the context turned
    // off (judged on the plan of several slots the load abandons), goes off
    // by the paged rule; every other reason holds whatever the plan, and
    // stays. A plan with no arena keeps its draft and its reason as asked.
    let paged = Some(Draft38Off::Paged { arena });
    let y = RungCost::of(Reserve::Draft, 702, 6165, 2_842_413_228).expect("the 3060's yield");
    assert_eq!(
        paged_draft(&at(0, true, false), None),
        (true, None),
        "an unset draft runs on a plan with no arena"
    );
    assert_eq!(
        paged_draft(&at(0, false, false), Some(Draft38Off::Yield(y))),
        (false, Some(Draft38Off::Yield(y))),
        "a yield on a plan with no arena stays"
    );
    assert_eq!(
        paged_draft(&at(arena, true, true), None),
        (true, None),
        "a set draft runs on a paged plan"
    );
    assert_eq!(
        paged_draft(&at(arena, true, false), None),
        (false, paged.clone())
    );
    assert_eq!(
        paged_draft(&at(arena, false, false), Some(Draft38Off::Yield(y))),
        (false, paged),
        "a yield judged on the abandoned plan"
    );
    let file = Path::new("/models/q/mtp.gguf");
    for stays in [
        Draft38Off::Set,
        Draft38Off::Gate,
        Draft38Off::Logits,
        Draft38Off::RouteTrace,
        Draft38Off::NoFile(vec![file.to_path_buf()]),
        Draft38Off::Ctx { need: 5, ctx: 4 },
        Draft38Off::Borrowed {
            name: "output.weight".to_string(),
            ty: "f16".to_string(),
        },
    ] {
        assert_eq!(
            paged_draft(&at(arena, false, false), Some(stays.clone())),
            (false, Some(stays.clone())),
            "{stays:?}"
        );
    }
    assert_eq!(
        Draft38Off::Paged { arena }.to_string(),
        "unset: the plan pages host experts through the NVMe tier's 23434203136 B RAM arena, \
         and the draft is off by default there; set BLOOMERY_DRAFT=mtp to draft on a paged plan"
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
/// fits, `off` where no pinned count the plan leaves fits the host, and
/// moved up where a larger count's pool does; `off` and a set word pass
/// through untouched.
#[test]
fn glm_residency_at_plan_takes_only_what_fits() {
    let mid = GlmPick {
        word: "mid-p0-s1",
        why: GlmWhy::Serving,
    };
    let off = |why| GlmPick { word: "off", why };
    // The host has room: the pool is asked at the word's pinned count alone.
    let fit = |n: &[u64], pool: u64, headroom, mem| {
        glm_residency_at_plan(
            UnhostedPool::Carried,
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
    let gate = off(GlmWhy::Gate);
    assert_eq!(
        glm_residency_at_plan(
            UnhostedPool::Carried,
            gate,
            [66u64],
            |_| Err("not asked"),
            0,
            0
        ),
        Ok(gate)
    );
    assert_eq!(fit(&[66], 10, 10, 10).map(|p| p.why), Ok(GlmWhy::Serving));
    assert_eq!(
        glm_residency_at_plan(
            UnhostedPool::Carried,
            mid,
            [66u64],
            |_| Err::<u64, &str>("the pool"),
            0,
            0
        ),
        Err("the pool"),
        "the pool's error is the call's"
    );
    // PIN(2026-10-13): the keep-P block (P 0 with the pool unhosted below the budget) is gone; an unset word now climbs on
    // every body, pinned by `an_unset_word_climbs_on_a_body_that_carries_an_unhosted_pool`.
}

/// An unset word climbs on a body that carries an unhosted pool, as on any
/// other: where its pool at P fits neither host budget, the load runs at the
/// first P whose pool fits, in every family's resolution.
// PIN(2026-10-13): an unset word on a body that carried an unhosted pool kept
// P and left the pool unhosted; that word decoded far slower than the climbed
// word, each victim read from the file mapping on the token path. The
// unhosted pool is the word's suffix (`-unhosted`), not a default.
// FAIL-first: the rule before this pin returns P 0 with `pool_unhosted` here.
#[test]
fn an_unset_word_climbs_on_a_body_that_carries_an_unhosted_pool() {
    let big = 1 << 40;
    let pool = |p: usize| Ok::<u64, ()>(10 * (50 - p) as u64);
    let serving = residency_unset(ResidencyAt {
        serving_place: true,
        ..ResidencyAt::FIXED
    });
    // 10 B an expert past P on 50: the first count whose pool fits 50 B is
    // 45.
    let v41 = residency_at_plan(UnhostedPool::Carried, serving, [50u64], 1, pool, 50, big)
        .expect("no pool error");
    assert_eq!(
        (v41.word, v41.why, v41.pool()),
        (
            "mid-p45-s1",
            ResidencyWhy::PoolShrunk { from: 0, to: 45 },
            ChurnHosting::Hosted
        )
    );
    let q38 = residency38_at_plan(UnhostedPool::Carried, [50u64], 1, pool, 50, big)
        .expect("no pool error");
    assert_eq!(
        (q38.pinned, q38.why, q38.pool()),
        (
            Some(45),
            Residency38Why::PoolShrunk { from: 0, to: 45 },
            ChurnHosting::Hosted
        )
    );
    let glm = GlmPick {
        word: GLM_RESIDENCY_UNSET,
        why: GlmWhy::Serving,
    };
    let g = glm_residency_at_plan(UnhostedPool::Carried, glm, [50u64], pool, 50, big)
        .expect("no pool error");
    assert_eq!(
        (g.word, g.why, g.pool()),
        (
            "mid-p45-s1",
            GlmWhy::PoolShrunk { from: 0, to: 45 },
            ChurnHosting::Hosted
        )
    );
}

/// A word that asks for its churn pool unhosted (`-unhosted`) on a body that
/// carries one keeps its P with the pool unhosted, whether or not the pool
/// fits; a body that hosts every pool refuses the ask by name.
#[test]
fn a_word_that_asks_keeps_its_pool_unhosted_on_a_body_that_carries_it() {
    let pool = |fewest: usize| move |p: usize| Ok::<u64, ()>(10 * (fewest - p) as u64);
    let asked = |target: usize, n_l: &[u64], headroom: i128, mem_left: i128| {
        let fewest = n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0) as usize;
        room_for(
            UnhostedPool::Carried.takes(true),
            target,
            1,
            n_l.iter().copied(),
            pool(fewest),
            headroom,
            mem_left,
        )
        .expect("no pool error")
    };
    let big = 1 << 40;
    let unhosted = |pinned, needs, headroom, mem_left| Room::Unhosted {
        pinned,
        pool: PoolUnhosted {
            needs,
            headroom,
            mem_left,
        },
    };
    // The pool at P 40 (10 experts, 100 B) fits neither budget, or both: the
    // word keeps P 40 and the pool is unhosted.
    assert_eq!(asked(40, &[50], 50, big), unhosted(40, 100, 50, big));
    assert_eq!(asked(40, &[50], big, 99), unhosted(40, 100, big, 99));
    assert_eq!(asked(40, &[50], 100, 100), unhosted(40, 100, 100, 100));
    // Under the card side's half (29 / 2 = 14, 15 experts, 150 B) it keeps the
    // half; the card side's refusals stand.
    assert_eq!(asked(40, &[29], big, big), unhosted(14, 150, big, big));
    assert_eq!(
        asked(40, &[2], big, big),
        Room::Short(RoomShort::NoRoom { fewest: 2 })
    );
    assert_eq!(
        asked(40, &[0, 0], big, big),
        Room::Short(RoomShort::NoCardExperts)
    );
    // The Qwen3.8 pick of such a word hosts nothing.
    let pick = Residency38Pick {
        pinned: Some(0),
        why: Residency38Why::Unhosted(PoolUnhosted {
            needs: 1,
            headroom: 0,
            mem_left: 0,
        }),
    };
    assert_eq!(pick.pool(), ChurnHosting::Unhosted);
    // The pool is taken unhosted only where the word asks and the body carries
    // one.
    assert!(UnhostedPool::Carried.takes(true));
    assert!(!UnhostedPool::Carried.takes(false));
    assert!(!UnhostedPool::NotCarried.takes(true));
    // A body that hosts every pool it runs refuses the ask, naming the word.
    let word = "mid-p0-s1-unhosted";
    assert_eq!(UnhostedPool::Carried.refuse(word), None);
    assert_eq!(
        UnhostedPool::NotCarried.refuse(word).as_deref(),
        Some(
            "BLOOMERY_RESIDENCY=mid-p0-s1-unhosted: -unhosted asks to keep the churn pool \
             unhosted, and this load hosts every churn pool it runs"
        )
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

/// Every Parsed and Direct lever refuses, naming itself, each value its kind
/// does not take and a value that is not UTF-8 — a number past `u64` saying
/// so — in a binary's reading whether it reads the lever or not, and a Parsed
/// one in a harness's; a row that trims takes its value with spaces around
/// it, and a [`Kind::Text`] value is its owner's to check, so it is taken as
/// written. Mutant: a reading that leaves a Direct value to the reader in
/// place, so `BLOOMERY_QWEN3_UBATCH=0` is taken and refused only once a load
/// reads it.
#[test]
fn levers_refuse_what_their_kind_does_not_take() {
    let raw = OsStr::from_bytes(b"o\xffn");
    for r in parsed()
        .chain(direct(Readers::Every))
        .chain(direct(Readers::Named))
    {
        let names = [r.name];
        // An Every lever is read with no name in the list, which takes none.
        let main = Scope::Main {
            bin: "b",
            acts_on: if not_named(&names).is_none() {
                &names
            } else {
                &[]
            },
        };
        let other = if matches!(r.site, Site::Parsed { .. }) {
            Scope::Every
        } else {
            NO_LEVER
        };
        for scope in [main, other] {
            if r.kind == Kind::Text {
                for v in ["", "anything at all"] {
                    let got = read(&env(&[(r.name, v)]), scope).unwrap_or_else(|e| panic!("{e}"));
                    assert_eq!(got.entry(r.name).set.as_deref(), Some(v));
                    assert_eq!(got.entry(r.name).value, None);
                }
                continue;
            }
            let mut values: Vec<OsString> =
                garbage(r.kind).into_iter().map(OsString::from).collect();
            values.push(raw.to_os_string());
            for v in &values {
                let e = read(&env(&[(r.name, v)]), scope).expect_err("invalid value");
                let [f] = e.refused.as_slice() else {
                    panic!("{e}")
                };
                let Why::Value(expected) = &f.why else {
                    panic!("{e}")
                };
                assert_eq!(f.name, r.name, "{e}");
                assert!(expected.contains(&r.kind.takes()), "{e}");
                if v.as_encoded_bytes().starts_with(b"1844674407370955161") || v == "17179869184G" {
                    assert!(expected.starts_with(OVERFLOW), "{e}");
                }
                if v == raw {
                    assert!(expected.starts_with("not UTF-8"), "{e}");
                }
            }
            read(&env(&[(r.name, sample(r.kind))]), scope).unwrap_or_else(|e| panic!("{e}"));
            if let Kind::Count { trim: true, .. } = r.kind {
                let padded = format!(" {} ", sample(r.kind));
                let got = read(&env(&[(r.name, padded.as_str())]), scope)
                    .unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(got.entry(r.name).set.as_deref(), Some(padded.as_str()));
            }
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
        let main = Scope::Main {
            bin: "b",
            acts_on: &acts_on,
        };
        read(&set, main).unwrap_or_else(|e| panic!("{e}"));
        let table = reading(&set, main).table();
        for p in parsed() {
            let line = table
                .lines()
                .find(|l| l.split_whitespace().next() == Some(p.name))
                .unwrap_or_else(|| panic!("no line for {}", p.name));
            let cells: Vec<&str> = line.split_whitespace().collect();
            let unread = p.name != r.name && !pool(p);
            assert_eq!(cells[4] == "no" && cells[5] == "-", unread, "{line}");
        }
    }
    let named: Vec<&str> = parsed()
        .chain(direct(Readers::Named))
        .map(|r| r.name)
        .collect();
    assert_eq!(not_named(&named), None);
    for r in REGISTRY.iter().filter(|r| !named.contains(&r.name)) {
        assert_eq!(not_named(&[named[0], r.name]), Some(r.name));
    }
    assert_eq!(
        not_named(&["BLOOMERY_NO_SUCH_NAME"]),
        Some("BLOOMERY_NO_SUCH_NAME")
    );
}

/// A binary applies a Direct lever its path reads (one it names, or one
/// every binary reads), and takes a well-formed Named one it leaves out
/// without a refusal, since one environment may carry the levers of several
/// models: that one is not applied, and its line names the binary, the
/// lever and where it is read. A Parsed lever left out is still refused.
/// Mutants: every Direct lever counted as read, so `BLOOMERY_WEIGHTS=anon`
/// beside a server says nothing; a Named one refused, so a start with
/// another model's lever set stops.
#[test]
fn a_binary_names_the_direct_levers_it_does_not_apply() {
    for r in direct(Readers::Every) {
        let set = env(&[(r.name, sample(r.kind))]);
        let got = reading(&set, NO_LEVER);
        assert!(got.unapplied("b").is_empty(), "{}", r.name);
        got.levers().unwrap_or_else(|e| panic!("{e}"));
    }
    for r in direct(Readers::Named) {
        let v = sample(r.kind);
        let set = env(&[(r.name, v.as_str())]);
        let got = reading(&set, NO_LEVER);
        let [line] = got
            .unapplied("b")
            .try_into()
            .unwrap_or_else(|l| panic!("{l:?}"));
        let at = in_place(r.site)[0].file;
        assert!(
            line.starts_with(&format!(
                "{}={v:?} is not applied: this binary (b) does not read it",
                r.name
            )) && line.contains(at),
            "{line}"
        );
        let levers = got.levers().unwrap_or_else(|e| panic!("{e}"));
        assert!(!levers.entry(r.name).acts, "{}", r.name);
        let acts_on = [r.name];
        let main = Scope::Main {
            bin: "b",
            acts_on: &acts_on,
        };
        let got = reading(&set, main);
        assert!(got.unapplied("b").is_empty(), "{}", r.name);
        let levers = got.levers().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(levers.entry(r.name).set.as_deref(), Some(v.as_str()));
        assert!(levers.entry(r.name).acts, "{}", r.name);
    }
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
    // PIN(2026-10-27): a Direct lever's value is checked by its kind now
    // (levers_refuse_what_their_kind_does_not_take), so each takes a value of its kind.
    let known: Vec<(&str, String)> = REGISTRY
        .iter()
        .filter(|r| matches!(r.site, Site::Direct { .. } | Site::Env { .. }))
        .map(|r| (r.name, sample(r.kind)))
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
fn remote_nodes_require_private_addresses_and_rdma() {
    for address in ["10.0.0.2:7340", "172.16.0.2:7340", "192.168.0.2:7340"] {
        for link in ["shared", "loopback", "rdma"] {
            let values = env(&[(TIER_NODES, address), (TIER_LINK, link)]);
            let got = read(&values, Scope::TierNodes);
            if link == "rdma" {
                let parsed = got.expect("a private RDMA node");
                assert!(
                    matches!(&parsed.entry(TIER_NODES).value, Some(Value::Addrs(a)) if a.len() == 1 && a[0].to_string() == address)
                );
            } else {
                assert!(
                    got.expect_err("an off-host node needs RDMA")
                        .to_string()
                        .contains("requires BLOOMERY_TIER_LINK=rdma")
                );
            }
        }
    }
    for address in ["127.0.0.1:7340", "[::1]:7340"] {
        assert!(read(&env(&[(TIER_NODES, address)]), Scope::TierNodes).is_ok());
    }
    for ip in [
        "0.0.0.0",
        "8.8.8.8",
        "224.0.0.1",
        "::",
        "fe80::1",
        "198.17.255.255",
        "198.20.0.1",
    ] {
        assert!(!tier_ip(ip.parse().expect("an IP address")), "{ip}");
    }
    for ip in [
        "10.200.23.1",
        "172.16.0.1",
        "192.168.1.2",
        "198.18.4.1",
        "198.19.255.254",
    ] {
        assert!(tier_ip(ip.parse().expect("an IP address")), "{ip}");
    }
}

/// A tensor-parallel run's nodes read in rank order; an off-host node needs RDMA, as a tier
/// node does, and the run's nodes are not set beside expert tier nodes.
#[test]
fn tp_nodes_take_the_tier_rules_and_never_sit_beside_tier_nodes() {
    let main = Scope::Main {
        bin: "generate_glm5next",
        acts_on: &[TP_NODES],
    };
    let set = read(
        &env(&[
            (TP_NODES, "198.18.4.1:7340,198.18.5.1:7340,198.18.6.1:7340"),
            (TIER_LINK, "rdma"),
        ]),
        main,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let want: Vec<SocketAddr> = vec![
        "198.18.4.1:7340".parse().unwrap(),
        "198.18.5.1:7340".parse().unwrap(),
        "198.18.6.1:7340".parse().unwrap(),
    ];
    assert_eq!(set.tp_nodes(), want.as_slice());
    assert!(read(&env::<&str>(&[]), main).unwrap().tp_nodes().is_empty());
    for (pairs, want) in [
        (
            vec![(TP_NODES, "198.18.4.1:7340")],
            "a node off this host requires BLOOMERY_TIER_LINK=rdma",
        ),
        (
            vec![(TP_NODES, "127.0.0.1:7340"), (TIER_NODES, "127.0.0.1:7341")],
            "BLOOMERY_TIER_NODES names expert tier nodes, and the two are not set together",
        ),
    ] {
        let e = read(&env(&pairs), main).unwrap_err().to_string();
        assert!(e.contains(TP_NODES) && e.contains(want), "{e}");
    }
    let other = Scope::Main {
        bin: "generate_qwen3moe",
        acts_on: &[],
    };
    assert!(read(&env(&[(TP_NODES, "127.0.0.1:7340")]), other).is_err());
}

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

/// `BLOOMERY_GATE_BUILD_BOUND` is the same parser's second variable: its row's
/// default is the parser's, and its kind is the test bound's.
#[test]
fn gate_build_bound_is_the_runners_parser() {
    let row = spec("BLOOMERY_GATE_BUILD_BOUND").expect("a row");
    let default = GATE_BOUND_PARSER
        .split_once("${BLOOMERY_GATE_BUILD_BOUND-")
        .and_then(|(_, rest)| rest.split_once('}'))
        .map(|(d, _)| d)
        .expect("the parser's default");
    assert_eq!(row.default, Unset::Is(default));
    assert_eq!(row.kind, spec("BLOOMERY_GATE_BOUND").expect("a row").kind);
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

/// `--levers` prints a line per lever with the phase, whether the run applies
/// it and the reading's value: the phase as the row states it (`load`,
/// `runtime`) or `-` for a retired row, which states none; `yes` and the
/// value as set or the default for a lever the binary reads, Parsed or
/// Direct; `no` and the value as set for a Direct lever set that it does not
/// read; `refused` and the value as set for a name it refuses (a retired
/// one); `no` and `-` for the rest. A reading that refuses still prints
/// every line. A name that is no lever has no line. Mutant: a table of the
/// Parsed levers alone, so the Direct levers set beside them read `no -` and
/// their values never show.
#[test]
fn table_is_a_line_per_lever() {
    let r = parsed().find(|r| !pool(r)).expect("a Parsed lever");
    let mut named = direct(Readers::Named).filter(|r| r.kind != Kind::Text);
    let (unread, read_here) = (
        named.next().expect("a Direct lever"),
        named.next().expect("two"),
    );
    let gone = retired().next().expect("a retired row");
    let set = [
        (r.name, sample(r.kind)),
        (unread.name, sample(unread.kind)),
        (gone.name, "1".to_string()),
    ];
    let acts_on = [r.name, read_here.name];
    let t = reading(
        &env(&set),
        Scope::Main {
            bin: "b",
            acts_on: &acts_on,
        },
    )
    .table();
    let levers: Vec<&LeverSpec> = REGISTRY.iter().filter(|r| r.is_lever()).collect();
    assert_eq!(t.lines().count(), levers.len(), "{t}");
    for (line, row) in t.lines().zip(levers) {
        let cells: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(cells[0], row.name, "{line}");
        let phase = match row.site {
            Site::Parsed { phase, .. } | Site::Direct { phase, .. } => phase.word(),
            Site::Retired { .. } | Site::Env { .. } => "-",
        };
        assert_eq!(cells[3], phase, "{line}");
        let default = || match row.default {
            Unset::Is(d) => format!("default {d}"),
            Unset::Means(m) => format!("unset: {m}"),
        };
        let every = matches!(
            row.site,
            Site::Direct {
                by: Readers::Every,
                ..
            }
        );
        let want = match set.iter().find(|(name, _)| *name == row.name) {
            Some((_, v)) if row.name == r.name => format!("yes set {v}"),
            Some((_, v)) if row.name == unread.name => format!("no set {v}"),
            Some((_, v)) => format!("refused set {v}"),
            None if row.name == read_here.name || pool(row) || every => {
                format!("yes {}", default())
            }
            None => "no -".to_string(),
        };
        assert!(cells[4..].join(" ").starts_with(&want), "{line}");
    }
}

#[test]
fn cli_prints_the_table_before_refusing_a_direct_value() {
    if std::env::var_os("LEVERS_CLI_CHILD").is_some() {
        if let Err(error) = at_main(&[]) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        panic!("--levers returned without a refusal");
    }
    for (value, applies, success) in [("0", "refused", false), ("128", "no", true)] {
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "tests::cli_prints_the_table_before_refusing_a_direct_value",
                "--nocapture",
                "--",
                "--levers",
            ])
            .env_clear()
            .env("LEVERS_CLI_CHILD", "1")
            .env("BLOOMERY_QWEN3_UBATCH", value)
            .output()
            .expect("run --levers");
        let stdout = String::from_utf8(output.stdout).expect("UTF-8 table");
        let stderr = String::from_utf8(output.stderr).expect("UTF-8 refusal");
        assert_eq!(output.status.success(), success, "{stdout}\n{stderr}");
        let rows: Vec<_> = stdout
            .lines()
            .filter(|line| line.starts_with("BLOOMERY_"))
            .collect();
        let levers: Vec<_> = REGISTRY.iter().filter(|row| row.is_lever()).collect();
        assert_eq!(rows.len(), levers.len(), "{stdout}");
        for (line, row) in rows.iter().zip(levers) {
            assert_eq!(line.split_whitespace().next(), Some(row.name), "{line}");
        }
        let line = rows
            .iter()
            .find(|line| line.starts_with("BLOOMERY_QWEN3_UBATCH "))
            .expect("ubatch row");
        let cells: Vec<_> = line.split_whitespace().collect();
        assert_eq!(&cells[4..7], &[applies, "set", value], "{line}");
        let reason = if success {
            "is not applied"
        } else {
            "is refused"
        };
        assert!(
            stderr.contains(&format!("BLOOMERY_QWEN3_UBATCH={value:?} {reason}")),
            "{stderr}"
        );
        assert_eq!(
            stderr.matches("is not applied").count(),
            usize::from(success),
            "{stderr}"
        );
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
            Some(ResidencyWord::Mid {
                pinned,
                spares,
                unhosted: false
            }),
            "{v}"
        );
        assert!(Kind::Residency.parse(v).is_ok(), "{v}");
        // The suffix asks for the churn pool unhosted and changes nothing
        // else of the word.
        let asked = format!("{v}{UNHOSTED_SUFFIX}");
        assert_eq!(
            residency_word(&asked),
            Some(ResidencyWord::Mid {
                pinned,
                spares,
                unhosted: true
            }),
            "{asked}"
        );
        assert!(Kind::Residency.parse(&asked).is_ok(), "{asked}");
    }
    assert_eq!(residency_word("mid-p33-s0"), None);
    // The suffix ends a `mid` word and nothing else.
    for v in [
        "off-unhosted",
        "mid-p0-s1-unhosted-unhosted",
        "mid-p0-s0-unhosted",
        "mid-p0-unhosted",
        "mid-p0-s1unhosted",
        "-unhosted",
    ] {
        assert_eq!(residency_word(v), None, "{v}");
    }
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

#[test]
fn a_load_that_hosts_its_pool_climbs_through_every_family_rule() {
    // 10 B an expert past P on 50-expert layers: the pool at P 0 is 500 B,
    // and the first count whose pool fits 50 B is 45; none fits 19 B (the
    // card ceiling 48's pool is 20 B).
    let pool = |p: usize| Ok::<u64, ()>(10 * (50 - p) as u64);
    let big = 1 << 40;
    let unhostable = PoolUnhostable {
        needs: 20,
        headroom: 19,
        mem_left: big,
    };
    let serving = residency_unset(ResidencyAt {
        serving_place: true,
        ..ResidencyAt::FIXED
    });
    let at = |headroom| {
        residency_at_plan(
            UnhostedPool::NotCarried,
            serving,
            [50u64],
            1,
            pool,
            headroom,
            big,
        )
        .expect("no pool error")
    };
    let raised = at(50);
    assert_eq!(
        (raised.word, raised.why, raised.why.name(), raised.pool()),
        (
            "mid-p45-s1",
            ResidencyWhy::PoolShrunk { from: 0, to: 45 },
            "pool_shrunk",
            ChurnHosting::Hosted
        )
    );
    let off = at(19);
    assert_eq!(
        (off.word, off.why, off.why.name()),
        (
            "off",
            ResidencyWhy::PoolUnhostable(unhostable),
            "pool_unhostable"
        )
    );
    let at38 = |headroom| {
        residency38_at_plan(UnhostedPool::NotCarried, [50u64], 1, pool, headroom, big)
            .expect("no pool error")
    };
    assert_eq!(
        at38(50),
        Residency38Pick {
            pinned: Some(45),
            why: Residency38Why::PoolShrunk { from: 0, to: 45 }
        }
    );
    assert_eq!(
        at38(19),
        Residency38Pick {
            pinned: None,
            why: Residency38Why::PoolUnhostable(unhostable)
        }
    );
    let glm = GlmPick {
        word: GLM_RESIDENCY_UNSET,
        why: GlmWhy::Serving,
    };
    let at_glm = |headroom| {
        glm_residency_at_plan(UnhostedPool::NotCarried, glm, [50u64], pool, headroom, big)
            .expect("no pool error")
    };
    assert_eq!(
        at_glm(50),
        GlmPick {
            word: "mid-p45-s1",
            why: GlmWhy::PoolShrunk { from: 0, to: 45 }
        }
    );
    assert_eq!(
        at_glm(19),
        GlmPick {
            word: "off",
            why: GlmWhy::PoolUnhostable(unhostable)
        }
    );
    // PIN(2026-10-13): the closing check that a carrying body keeps P 0 on the same plan is gone; the carrying body now
    // climbs like these, pinned by `an_unset_word_climbs_on_a_body_that_carries_an_unhosted_pool`.
}

/// `names` as override pairs.
fn pairs<'a>(names: &[(&'a str, &str)]) -> Vec<(&'a str, String)> {
    names
        .iter()
        .map(|&(name, value)| (name, value.to_owned()))
        .collect()
}

/// An override replaces the environment's value and says so (nothing when the two are the
/// same), and a reading of the environment with the overrides after it takes the override.
#[test]
fn an_override_replaces_the_environment_value_and_says_so() {
    let held = env(&[(CARD_BUDGET, "8G"), (QWEN38_EXPERTS, "host")]);
    let (set, notes) = checked_overrides(
        &pairs(&[(CARD_BUDGET, "1073741824"), (QWEN38_EXPERTS, "host")]),
        &held,
    )
    .expect("two Parsed levers with values their kinds take");
    assert_eq!(
        set,
        [
            (CARD_BUDGET, "1073741824".to_owned()),
            (QWEN38_EXPERTS, "host".to_owned())
        ]
    );
    assert_eq!(
        notes,
        ["BLOOMERY_CARD_BUDGET: the environment's value \"8G\" is replaced by \"1073741824\""]
    );
    let levers = read(&merged(held, &set), Scope::Every).expect("a reading under the overrides");
    assert_eq!(levers.card_budget(), CardBudget::Every(1 << 30));
    let (_, none) = checked_overrides(&pairs(&[(CARD_BUDGET, "1G")]), &[]).expect("no environment");
    assert!(none.is_empty());
}

/// Every override a reading cannot honour is refused by name, all at once: a name no row names, a
/// retired name, a value its kind refuses, a Parsed lever a file still reads in place, a Direct
/// lever whose reader does not take overrides, and a name that is no lever.
#[test]
fn an_override_that_cannot_be_honoured_is_refused_by_name() {
    let retired = retired().next().expect("a retired row").name;
    let in_place = REGISTRY
        .iter()
        .find(|r| matches!(r.site, Site::Parsed { left, .. } if !left.is_empty()))
        .expect("a Parsed lever a file still reads in place")
        .name;
    let direct = REGISTRY
        .iter()
        .find(|r| matches!(r.site, Site::Direct { .. }) && !READ_WITH_OVERRIDE.contains(&r.name))
        .expect("a Direct lever whose reader takes no override")
        .name;
    let no_lever = REGISTRY
        .iter()
        .find(|r| matches!(r.site, Site::Env { .. }))
        .expect("a name that is no lever")
        .name;
    let asked = pairs(&[
        ("BLOOMERY_CARD_BUDGT", "1G"),
        (retired, "1"),
        (CARD_BUDGET, "lots"),
        (in_place, "1"),
        (direct, "1"),
        (no_lever, "1"),
    ]);
    let text = checked_overrides(&asked, &[])
        .expect_err("six overrides it cannot honour")
        .to_string();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 6, "{text}");
    assert!(
        lines[0].contains("the nearest is BLOOMERY_CARD_BUDGET"),
        "{text}"
    );
    assert!(lines[1].contains("it is no lever:"), "{text}");
    assert!(
        lines[2].starts_with("BLOOMERY_CARD_BUDGET=\"lots\" is refused: it takes"),
        "{text}"
    );
    for line in &lines[3..] {
        assert!(line.contains("a flag cannot set it"), "{text}");
    }
}

/// A Direct lever whose reader asks `override_of` is overridden by a value its kind takes (a Text
/// lever's is its owner's to check), and each such reader does ask: its file names the call.
#[test]
fn a_direct_lever_is_overridden_only_where_its_reader_asks() {
    assert!(!READ_WITH_OVERRIDE.is_empty());
    for &name in READ_WITH_OVERRIDE {
        let row = spec(name).expect(name);
        let Site::Direct { at, .. } = row.site else {
            panic!("{name} is not a Direct lever");
        };
        for place in at {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(place.file);
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(
                source.contains(&format!("override_of(\"{name}\")")),
                "{} reads {name} without asking override_of",
                place.file
            );
        }
        let (set, _) = checked_overrides(&pairs(&[(name, "cuda1")]), &[])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(set, [(row.name, "cuda1".to_owned())]);
    }
}

/// The one test that hands overrides to the process: they reach every reading of the process
/// environment and the in-place readers of a Direct lever, a second hand-over is refused by name,
/// and an empty one is not a hand-over. Nothing else here reads the process environment, so the
/// store this fills reaches no other test.
#[test]
fn the_process_hands_its_overrides_once() {
    assert_eq!(override_of(CARD_BUDGET), None);
    let notes = override_with(&pairs(&[
        (CARD_BUDGET, "1G"),
        ("BLOOMERY_DSPARK_CARD", "cuda1"),
    ]))
    .expect("a Parsed lever and a Direct one that asks");
    assert!(
        notes.iter().all(|n| !n.is_empty()),
        "a note a line: {notes:?}"
    );
    assert_eq!(override_of("BLOOMERY_DSPARK_CARD"), Some("cuda1"));
    let levers = Levers::from_env().expect("the process environment under the overrides");
    assert_eq!(levers.card_budget(), CardBudget::Every(1 << 30));
    assert_eq!(override_with(&[]), Ok(Vec::new()));
    let again = override_with(&pairs(&[(CARD_BUDGET, "2G")])).expect_err("a second hand-over");
    assert!(again.to_string().contains("handed over once already"));
    assert_eq!(override_of(CARD_BUDGET), Some("1G"));
}

/// A set `mid` word's least fewest is the fewest whose card side holds the
/// word's own P with its spares and one that moves: at it [`card_p`] gives P,
/// one under it does not. `off` and a value outside the grammar ask none.
/// FAIL-first: a count without the one that moves (`pinned + spares`) leaves
/// the card one slot short of P.
#[test]
fn a_set_words_least_fewest_holds_its_own_p() {
    for (word, pinned, spares) in [
        ("mid-p0-s1", 0usize, 1usize),
        ("mid-p0-s1-unhosted", 0, 1),
        ("mid-p3-s2", 3, 2),
        ("mid-p12-s1-unhosted", 12, 1),
    ] {
        let least = residency_word_least_fewest(word).expect(word);
        let least = usize::try_from(least).expect("a count");
        assert_eq!(card_p(pinned, spares, least), Some(pinned), "{word}");
        assert_ne!(card_p(pinned, spares, least - 1), Some(pinned), "{word}");
    }
    assert_eq!(
        residency_word_least_fewest("mid-p0-s1"),
        Some(residency38_least_fewest()),
        "the unset word's own parameters ask the same floor"
    );
    for word in ["off", "mid-p0-s0", "high"] {
        assert_eq!(residency_word_least_fewest(word), None, "{word}");
    }
}
