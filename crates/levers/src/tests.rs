use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use super::*;

/// The list of in-place reads `tools/check-levers.sh` holds the crates to.
const ALLOW_LIST: &str = include_str!("../../../tools/levers-direct.txt");

/// A lever's reading as a binary ends up with it: the value it uses, or the
/// refusal's text.
type Reading = Result<String, String>;

/// The contract a lever that took garbage as its default breaks: `garbage`
/// is refused, by a message that names the lever.
fn refuses(name: &str, garbage: &str, read: impl Fn(&str) -> Reading) -> Result<(), String> {
    match read(garbage) {
        Err(e) if e.contains(name) => Ok(()),
        Err(e) => Err(format!("refused {garbage:?} without naming {name}: {e}")),
        Ok(v) => Err(format!("took {garbage:?} as {v}")),
    }
}

/// `name` set to a value, read through the registry, then `get`.
fn registry(name: &'static str, get: fn(&Levers) -> String) -> impl Fn(&str) -> Reading {
    move |v| {
        Levers::from_pairs(&[(name, v)])
            .map(|l| get(&l))
            .map_err(|e| e.to_string())
    }
}

/// FAIL-first in one test: the refusal contract fails against `before`, the
/// expression the lever was read with before the registry, copied verbatim
/// with the environment's value in place of the `std::env::var` call, and
/// holds through the registry.
fn fail_first(
    name: &'static str,
    garbage: &str,
    before: impl Fn(&str) -> Reading,
    get: fn(&Levers) -> String,
) {
    let was = refuses(name, garbage, before);
    let is = refuses(name, garbage, registry(name, get));
    println!("{name}={garbage:?}: before {was:?}; registry {is:?}");
    assert!(
        was.is_err(),
        "{name}: the old expression refused {garbage:?} already"
    );
    assert_eq!(is, Ok(()), "{name}");
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
/// empty, and no text breaks a Markdown table cell.
#[test]
fn rows_are_well_formed() {
    for r in REGISTRY {
        if let Unset::Is(d) = r.default {
            assert!(r.kind.parse(d).is_some(), "{}: default {d:?}", r.name);
        }
        match r.kind {
            Kind::Words(words) => {
                let mut w = words.to_vec();
                w.sort_unstable();
                w.dedup();
                assert!(!words.is_empty() && w.len() == words.len(), "{}", r.name);
            }
            Kind::Count { min, max, .. } => assert!(min <= max, "{}", r.name),
            Kind::Multiple { of } => assert!(of > 0, "{}", r.name),
            _ => {}
        }
        let texts = [r.doc.to_string(), r.kind.takes(), r.site.describe()];
        assert!(!r.doc.is_empty(), "{}", r.name);
        assert!(texts.iter().all(|t| !t.contains('|')), "{}: a '|'", r.name);
    }
}

/// The migration check: every name the repository's lever prose (AGENTS.md
/// "Known state") and the tools audit's lever table name is a row. The lists
/// are fixed here as they stood when the registry replaced the prose; the
/// test goes when the prose does.
#[test]
fn prose_names_are_rows() {
    const AGENTS_PROSE: [&str; 34] = [
        "BLOOMERY_ATTN_BUNDLE",
        "BLOOMERY_ATTN_HALVES",
        "BLOOMERY_CARD_BUDGET",
        "BLOOMERY_CARD_DONTNEED",
        "BLOOMERY_CED",
        "BLOOMERY_CHECK_FINITE",
        "BLOOMERY_DRAFT",
        "BLOOMERY_ENGRAM_HELPER",
        "BLOOMERY_FLASH_MMA",
        "BLOOMERY_FLASH_SEG",
        "BLOOMERY_FLASH_SEGMENTS",
        "BLOOMERY_FLASH_SIMD",
        "BLOOMERY_GATE_BOUND",
        "BLOOMERY_GQA_MMA",
        "BLOOMERY_HOST_LOCK",
        "BLOOMERY_HOST_POPULATE",
        "BLOOMERY_HOT_LIST",
        "BLOOMERY_HYBRID_NL",
        "BLOOMERY_HYBRID_OVERLAP",
        "BLOOMERY_KV_PREFETCH",
        "BLOOMERY_KV_PREFETCH_ROWS",
        "BLOOMERY_LAUNCH_THREAD",
        "BLOOMERY_PIN_MAIN",
        "BLOOMERY_PREFILL_GROUP",
        "BLOOMERY_PROFILE",
        "BLOOMERY_PROFILE_DEPTH",
        "BLOOMERY_Q3K_SPLIT",
        "BLOOMERY_Q3K_SPLIT_ITERS",
        "BLOOMERY_QWEN3_UBATCH",
        "BLOOMERY_SPIN",
        "BLOOMERY_STEAL",
        "BLOOMERY_STEAL_BLOCKS",
        "BLOOMERY_STEP_STATS",
        "BLOOMERY_THREADS",
    ];
    // The audit's table (docs/research/audit/tools.md, TL-1) adds these.
    const AUDIT_TABLE: [&str; 8] = [
        "BLOOMERY_CARD_EXPERTS",
        "BLOOMERY_DEFER_QUANT",
        "BLOOMERY_EXPERT_LOG",
        "BLOOMERY_POISON",
        "BLOOMERY_POPULATE",
        "BLOOMERY_PREFILL",
        "BLOOMERY_STEP_PAIR",
        "BLOOMERY_WEIGHTS",
    ];
    let missing: Vec<&str> = AGENTS_PROSE
        .iter()
        .chain(&AUDIT_TABLE)
        .copied()
        .filter(|n| spec(n).is_none())
        .collect();
    assert!(missing.is_empty(), "not in the registry: {missing:?}");
}

/// Each accessor reads its own row: the defaults unset, the values as set —
/// the last of a name given twice — and the parsed rows are exactly the
/// accessors' rows.
#[test]
fn accessors_read_their_rows() {
    let unset = Levers::from_pairs::<&str>(&[]).expect("nothing is set");
    assert_eq!(unset.threads(), None);
    assert_eq!(unset.spin(), 20_000);
    assert!(unset.ced());
    assert_eq!(unset.prefill(), "batch");
    assert_eq!(unset.prefill_group(), 2);
    assert!(unset.engram_helper());
    assert!(!unset.step_stats());
    assert_eq!(unset.hot_list(), None);
    assert_eq!(unset.card_budget(), None);
    assert!(unset.pin_main());
    assert_eq!(unset.draft(), None);
    assert!(!unset.check_finite());

    let set = Levers::from_pairs(&[
        (THREADS, " 3 "),
        (SPIN, "0"),
        (CED, "off"),
        (PREFILL, "steps"),
        (PREFILL_GROUP, "1"),
        (ENGRAM_HELPER, "0"),
        (STEP_STATS, "1"),
        (HOT_LIST, "/data/hot.txt"),
        (CARD_BUDGET, "38G"),
        (PIN_MAIN, "0"),
        (DRAFT, "dspark"),
        (CHECK_FINITE, "1"),
        ("BLOOMERY_NOT_A_LEVER", "anything"),
        (PREFILL_GROUP, "8"),
    ])
    .expect("every value is one its kind takes");
    assert_eq!(set.threads(), Some(3));
    assert_eq!(set.spin(), 0);
    assert!(!set.ced());
    assert_eq!(set.prefill(), "steps");
    assert_eq!(set.prefill_group(), 8);
    assert!(!set.engram_helper());
    assert!(set.step_stats());
    assert_eq!(set.hot_list(), Some(Path::new("/data/hot.txt")));
    assert_eq!(set.card_budget(), Some(38 << 30));
    assert!(!set.pin_main());
    assert_eq!(set.draft(), Some("dspark"));
    assert!(set.check_finite());

    let parsed: Vec<&str> = REGISTRY
        .iter()
        .filter(|r| matches!(r.site, Site::Parsed { .. }))
        .map(|r| r.name)
        .collect();
    assert_eq!(
        parsed,
        [
            THREADS,
            SPIN,
            CED,
            PREFILL,
            PREFILL_GROUP,
            ENGRAM_HELPER,
            STEP_STATS,
            HOT_LIST,
            CARD_BUDGET,
            PIN_MAIN,
            DRAFT,
            CHECK_FINITE
        ]
    );
}

#[test]
fn step_stats_refuses_garbage() {
    // `std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1")`
    let before = |v: &str| -> Reading { Ok((v == "1").to_string()) };
    fail_first(STEP_STATS, "true", before, |l| l.step_stats().to_string());
}

#[test]
fn engram_helper_refuses_garbage() {
    // `!std::env::var("BLOOMERY_ENGRAM_HELPER").is_ok_and(|v| v == "0")`
    let before = |v: &str| -> Reading { Ok((v != "0").to_string()) };
    fail_first(ENGRAM_HELPER, "off", before, |l| {
        l.engram_helper().to_string()
    });
}

#[test]
fn pin_main_refuses_garbage() {
    // `!std::env::var("BLOOMERY_PIN_MAIN").is_ok_and(|v| v == "0")`
    let before = |v: &str| -> Reading { Ok((v != "0").to_string()) };
    fail_first(PIN_MAIN, "no", before, |l| l.pin_main().to_string());
}

/// Zero and a word were the physical core count; a count with spaces around
/// it is taken, as it was.
#[test]
fn threads_refuses_zero_and_words() {
    // `std::env::var("BLOOMERY_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok())
    //  .filter(|&t| t > 0).unwrap_or(physical)`
    let before = |v: &str| -> Reading {
        Ok(Some(v)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&t| t > 0)
            .map_or_else(|| "physical".to_string(), |t| t.to_string()))
    };
    let get: fn(&Levers) -> String = |l| format!("{:?}", l.threads());
    fail_first(THREADS, "0", before, get);
    fail_first(THREADS, "eight", before, get);
    assert_eq!(before(" 8 "), Ok("8".to_string()));
    assert_eq!(registry(THREADS, get)(" 8 "), Ok("Some(8)".to_string()));
}

#[test]
fn spin_refuses_words() {
    // `std::env::var("BLOOMERY_SPIN").ok().and_then(|s| s.trim().parse::<u64>().ok())
    //  .unwrap_or(DEFAULT_SPIN)`
    let before = |v: &str| -> Reading {
        Ok(Some(v)
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(20_000)
            .to_string())
    };
    let get: fn(&Levers) -> String = |l| l.spin().to_string();
    fail_first(SPIN, "fast", before, get);
    assert_eq!(registry(SPIN, get)(" 0"), Ok("0".to_string()));
}

/// A retired name that is set is refused whatever its value: nothing read
/// it after its arm was deleted, so a stale command line timed the default
/// and labelled it the arm.
#[test]
fn card_experts_is_refused() {
    let before = |_: &str| -> Reading { Ok("nothing reads it".to_string()) };
    fail_first(CARD_EXPERTS, "expert", before, |_| String::new());
}

#[test]
fn step_pair_is_refused() {
    let before = |_: &str| -> Reading { Ok("nothing reads it".to_string()) };
    fail_first(STEP_PAIR, "1", before, |_| String::new());
}

/// Refused at `main` in every binary that parses there, which before was
/// only the V2-Lite flash's first use (`flash::seg_keys`), a V4.1 binary
/// never.
#[test]
fn flash_mma_is_refused() {
    let before = |_: &str| -> Reading { Ok("nothing reads it in a V4.1 binary".to_string()) };
    fail_first(FLASH_MMA, "0", before, |_| String::new());
}

/// The levers that refused a value by name before the registry still do,
/// naming the lever: one test each.
macro_rules! refuses_by_name {
    ($test:ident, $name:expr, $($garbage:expr),+) => {
        #[test]
        fn $test() {
            for garbage in [$($garbage),+] {
                let e = Levers::from_pairs(&[($name, garbage)]).expect_err(garbage);
                assert!(e.to_string().starts_with(&format!("{}=", $name)), "{e}");
            }
        }
    };
}

refuses_by_name!(ced_refuses_by_name, CED, "maybe", "ON", "");
refuses_by_name!(prefill_refuses_by_name, PREFILL, "chunks", "");
refuses_by_name!(
    prefill_group_refuses_by_name,
    PREFILL_GROUP,
    "0",
    "9",
    " 2",
    "two"
);
refuses_by_name!(draft_refuses_by_name, DRAFT, "off", "");
refuses_by_name!(check_finite_refuses_by_name, CHECK_FINITE, "yes", "2");
refuses_by_name!(hot_list_refuses_by_name, HOT_LIST, "");
refuses_by_name!(
    card_budget_refuses_by_name,
    CARD_BUDGET,
    "38g",
    "38 G",
    "-1"
);

/// A value that is not UTF-8 is refused by name. `BLOOMERY_CED`,
/// `BLOOMERY_PREFILL` and the 0/1 flags read one as unset before.
#[test]
fn not_utf8_is_refused_by_name() {
    let raw = OsStr::from_bytes(b"o\xffn");
    for name in [CED, PREFILL, STEP_STATS, HOT_LIST] {
        let e = Levers::from_pairs(&[(name, raw)]).expect_err(name);
        assert_eq!(
            (e.name, e.expected.starts_with("not UTF-8")),
            (name, true),
            "{e}"
        );
    }
}

#[test]
fn parse_bytes_takes_bytes_mib_and_gib() {
    assert_eq!(parse_bytes("40802189312"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("40_802_189_312"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("38G"), Ok(40_802_189_312));
    assert_eq!(parse_bytes("24176M"), Ok(25_350_373_376));
    for bad in [
        "",
        "G",
        "38 G",
        "38g",
        "38GB",
        "-1",
        "1.5G",
        "18446744073709551616",
        "17179869184G",
    ] {
        assert!(parse_bytes(bad).is_err(), "{bad:?} parsed");
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

/// The registry and the allow-list name the same in-place reads: every file
/// a row reads its lever in place in is a line of the list with that lever,
/// in the same round, and every lever the list names is a row that names
/// that file and round.
#[test]
fn registry_and_allow_list_agree() {
    let lines = allow_lines();
    let mut bad = Vec::new();
    for r in REGISTRY {
        for p in r.site.in_place() {
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
        for v in &l.vars {
            if let Some(r) = spec(v)
                && !r
                    .site
                    .in_place()
                    .iter()
                    .any(|p| p.file == l.file && p.round == l.round)
            {
                bad.push(format!(
                    "{v}: the list reads it in place in {} ({}), its row does not",
                    l.file, l.round
                ));
            }
        }
    }
    assert!(bad.is_empty(), "\n{}", bad.join("\n"));
}

/// The Markdown table has a line per row, six cells each; printed, it is
/// the text the repository's lever documentation takes.
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

/// `--levers` prints a line per row with this reading's value: as set, the
/// default, or `-` for a lever the reading does not parse.
#[test]
fn table_is_a_line_per_row() {
    let t = Levers::from_pairs(&[(CED, "off")])
        .expect("off is a CED value")
        .table();
    assert_eq!(t.lines().count(), REGISTRY.len());
    let line = |name: &str| {
        t.lines()
            .find(|l| l.split_whitespace().next() == Some(name))
            .unwrap_or_else(|| panic!("no line for {name}"))
            .to_string()
    };
    assert!(line(CED).contains("set off"), "{}", line(CED));
    assert!(line(PREFILL_GROUP).contains("default 2"));
    assert!(line(HOT_LIST).contains("unset: "));
    assert!(line("BLOOMERY_GQA_MMA").contains(" - "));
}
