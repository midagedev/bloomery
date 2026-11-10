//! kcover: what each fixture cannot reach of a real file (`model::arch::coverage::reach`), from
//! headers alone. Two contracts, one test each:
//!
//! - the diff names a class of every kind it reads (a need, a type pin, a width, a router row) on
//!   the side that holds it, and only that kind for an edit of one kind: four descriptions edited
//!   from the Kimi K3 header, one kind each;
//! - the table `tools/kcover/unreached.tsv` holds exactly the gaps of every fixture against the
//!   real headers checked in under this directory (`<stem>_hdr.kv.json`, `<stem>_tensors.tsv`),
//!   each fixture planned from its source's header, and no row names a checked-in header no pair
//!   computes. The census stubs' rows are held by `fixture reach --check` where the stubs are.

use std::collections::BTreeSet;
use std::path::Path;

use gguf::{GgmlType, Split};
use model::arch::Read;
use model::arch::coverage::reach::{
    self, Class, ClassKind, Reach, Side, check_rows, classes, parse_table,
};
use model::fixture::{FixtureSpec, VARIANT_TABLE, VariantRow, variant_parse_table};
use models::Ffn;

#[path = "common/header_tables.rs"]
mod header_tables;

const KIMI_KV: &str = include_str!("kimi_k3_hdr.kv.json");
const KIMI_TENSORS: &str = include_str!("kimi_k3_tensors.tsv");
const MISTRAL_KV: &str = include_str!("mistrallarge4_hdr.kv.json");
const MISTRAL_TENSORS: &str = include_str!("mistrallarge4_tensors.tsv");
const MIX_KV: &str = include_str!("mistrallarge4_mix256_hdr.kv.json");
const MIX_TENSORS: &str = include_str!("mistrallarge4_mix256_tensors.tsv");
const MUSE_KV: &str = include_str!("musespark_hdr.kv.json");
const MUSE_TENSORS: &str = include_str!("musespark_tensors.tsv");
const MIMO_KV: &str = include_str!("mimo2_rl_hdr.kv.json");
const MIMO_TENSORS: &str = include_str!("mimo2_rl_tensors.tsv");

/// The repository root.
fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model sits two below the root")
}

fn read_of(split: &Split) -> Read {
    model::arch::spec(split).unwrap_or_else(|e| panic!("spec: {e}"))
}

/// The header of tables whose file has no `vocab_size` key, so its readers count the token list:
/// the summarized list (`{array_len, head}`) stands as `array_len` empty strings, the file's count
/// and none of its texts, which no class reads.
fn counted_header(tag: &str, kv: &str, tensors: &str) -> Split {
    let mut table: serde_json::Value = serde_json::from_str(kv).expect("the header table parses");
    let tokens = &mut table["tokenizer.ggml.tokens"];
    let n = tokens["array_len"]
        .as_u64()
        .expect("the table summarizes the token list by its length");
    *tokens = serde_json::Value::Array(vec![serde_json::Value::from(""); n as usize]);
    header_tables::header_of(tag, &table.to_string(), tensors)
}

fn kimi() -> Split {
    header_tables::header_of("kcover-k3", KIMI_KV, KIMI_TENSORS)
}

/// The kinds of the classes on each side of `real` against `edited`.
fn kinds(real: &Read, edited: &Read) -> (BTreeSet<ClassKind>, BTreeSet<ClassKind>) {
    let d = reach::diff(real, edited).unwrap_or_else(|e| panic!("diff: {e}"));
    let k = |s: &BTreeSet<Class>| s.iter().map(|c| c.kind).collect();
    (k(&d.real), k(&d.fixture))
}

/// The diff's contract, by kind: an edit of one kind's facts shows on both sides as that kind
/// alone (a router edit also moves the router need, which names the same shape); an unedited copy
/// shows nothing; two architectures are refused by name.
#[test]
fn kcover_diff_names_each_kind() {
    let real = read_of(&kimi());
    let mut bad = Vec::new();
    let same = reach::diff(&real, &real).expect("one architecture");
    if !same.is_empty() {
        bad.push(format!("an unedited copy differs: {same:?}"));
    }

    // need: the dense lead's width, which no tensor's dims are read from here.
    let mut need = read_of(&kimi());
    let lead = need
        .spec
        .layers
        .iter_mut()
        .find_map(|l| match &mut l.ffn {
            Ffn::Dense { ff, .. } => Some(ff),
            Ffn::Moe(_) => None,
        })
        .expect("the K3 header has a dense lead");
    *lead += 256;
    // pin: the output head stored in another type.
    let mut pin = read_of(&kimi());
    let head = pin
        .tensors
        .tensors
        .iter_mut()
        .find(|t| t.name == "output.weight")
        .expect("the K3 header has an output head");
    head.ty = if head.ty == GgmlType::Q8_0 {
        GgmlType::Q6_K
    } else {
        GgmlType::Q8_0
    };
    // width: the output head one block of rows wider.
    let mut width = read_of(&kimi());
    let head = width
        .tensors
        .tensors
        .iter_mut()
        .find(|t| t.name == "output.weight")
        .expect("the K3 header has an output head");
    head.dims[1] += 256;
    // router: every mixture over the fixture's 8 experts, 2 kept.
    let mut router = read_of(&kimi());
    for l in &mut router.spec.layers {
        if let Ffn::Moe(m) = &mut l.ffn {
            m.experts = 8;
            m.top_k = 2;
        }
    }

    let one = |k: ClassKind| BTreeSet::from([k]);
    for (what, edited, want) in [
        ("need", &need, one(ClassKind::Need)),
        ("pin", &pin, one(ClassKind::Pin)),
        ("width", &width, one(ClassKind::Width)),
        (
            "router",
            &router,
            BTreeSet::from([ClassKind::Need, ClassKind::Router]),
        ),
    ] {
        let (r, f) = kinds(&real, edited);
        if r != want || f != want {
            bad.push(format!(
                "a {what} edit: real side {r:?}, fixture side {f:?}, want {want:?} on each"
            ));
        }
    }

    let mistral = read_of(&header_tables::header_of(
        "kcover-ml4",
        MISTRAL_KV,
        MISTRAL_TENSORS,
    ));
    if let Ok(d) = reach::diff(&mistral, &real) {
        bad.push(format!("two architectures diffed: {d:?}"));
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// One fixture this test plans, and the checked-in real headers it is held against.
struct Pair {
    fixture: &'static str,
    spec: FixtureSpec,
    source: Split,
    reals: Vec<(&'static str, Split)>,
}

/// The fixture directory of the registry row whose canonical architecture is `arch`.
fn dir_of(arch: &str) -> &'static str {
    model::arch::registry::fixture_of(arch)
        .unwrap_or_else(|| panic!("{arch} has a GGUF fixture row"))
        .family
}

fn spec_of(arch: &str) -> FixtureSpec {
    model::arch::registry::fixture_spec(arch).unwrap_or_else(|| panic!("{arch} has a spec"))
}

/// Every pair the checked-in headers make: each fixture whose source header is checked in here
/// (the profile fixtures with their rows' ones), against each checked-in header of its
/// architecture. The full-shape cuts (K3's, Muse Spark's) and the MiMo fixture hold no row: each
/// reaches every class of its real header.
fn pairs() -> Vec<Pair> {
    let ml4 = || header_tables::header_of("kcover-ml4", MISTRAL_KV, MISTRAL_TENSORS);
    let muse = || counted_header("kcover-muse", MUSE_KV, MUSE_TENSORS);
    let mimo = || counted_header("kcover-mimo", MIMO_KV, MIMO_TENSORS);
    let profile = |p: &str| {
        model::arch::registry::profile_fixture(p)
            .unwrap_or_else(|| panic!("{p} is a profile fixture"))
    };
    let full = |p: &str, source: Split, real: (&'static str, Split)| {
        let fx = profile(p);
        Pair {
            fixture: fx.family,
            spec: (fx.spec)(),
            source,
            reals: vec![real],
        }
    };
    vec![
        Pair {
            fixture: dir_of("kimi-k3"),
            spec: spec_of("kimi-k3"),
            source: kimi(),
            reals: vec![("tests/kimi_k3", kimi())],
        },
        full("kimi-k3-full", kimi(), ("tests/kimi_k3", kimi())),
        Pair {
            fixture: dir_of("muse-spark"),
            spec: spec_of("muse-spark"),
            source: muse(),
            reals: vec![("tests/musespark", muse())],
        },
        full("muse-spark-full", muse(), ("tests/musespark", muse())),
        Pair {
            fixture: dir_of("mimo2"),
            spec: spec_of("mimo2"),
            source: mimo(),
            reals: vec![("tests/mimo2_rl", mimo())],
        },
        Pair {
            fixture: dir_of("mistral-large4"),
            spec: spec_of("mistral-large4"),
            source: ml4(),
            reals: vec![
                ("tests/mistrallarge4", ml4()),
                (
                    "tests/mistrallarge4_mix256",
                    header_tables::header_of("kcover-ml4-mix", MIX_KV, MIX_TENSORS),
                ),
            ],
        },
    ]
}

fn text(rel: &str) -> String {
    let p = root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The table's contract: its rows of every computed pair are the pair's gaps, row for row, and a
/// row whose real header is a checked-in one names a pair computed here.
#[test]
fn kcover_table_holds_the_checked_in_headers() {
    let table = parse_table(&text(reach::TABLE)).unwrap_or_else(|e| panic!("{e}"));
    let variants: Vec<VariantRow> =
        variant_parse_table(&text(VARIANT_TABLE)).unwrap_or_else(|e| panic!("{e}"));
    let mut bad = Vec::new();
    let mut computed: Vec<(String, String)> = Vec::new();
    for p in pairs() {
        let reach = Reach::of(p.fixture, &p.spec, &p.source)
            .unwrap_or_else(|e| panic!("{}: {e}", p.fixture));
        for (real, split) in &p.reals {
            let gaps = reach
                .gaps(&read_of(split), &variants)
                .unwrap_or_else(|e| panic!("{} against {real}: {e}", p.fixture));
            let count = |s: Side| gaps.iter().filter(|g| g.side == s).count();
            println!(
                "{} against {real}: {} classes, {} the fixture does not reach, {} only the fixture runs",
                p.fixture,
                classes(&read_of(split)).len(),
                count(Side::Real),
                count(Side::Fixture)
            );
            bad.extend(check_rows(&table, p.fixture, real, &gaps));
            computed.push((p.fixture.to_string(), (*real).to_string()));
        }
    }
    for r in &table {
        if r.real.starts_with("tests/")
            && !computed
                .iter()
                .any(|(f, real)| *f == r.fixture && *real == r.real)
        {
            bad.push(format!("a row of a pair no clause computes: {}", r.line()));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// A variant reaches a pin: a fixture-variants row of the fixture's architecture that retypes a
/// stem to the type a real file stores it in takes that pin out of the gaps, and only it; a row of
/// another architecture takes nothing.
#[test]
fn kcover_a_variant_reaches_its_pin() {
    let ml4 = header_tables::header_of("kcover-ml4-v", MISTRAL_KV, MISTRAL_TENSORS);
    let mix = read_of(&header_tables::header_of(
        "kcover-ml4-mix-v",
        MIX_KV,
        MIX_TENSORS,
    ));
    let spec = spec_of("mistral-large4");
    let reach = Reach::of(dir_of("mistral-large4"), &spec, &ml4).unwrap_or_else(|e| panic!("{e}"));
    let pins = |variants: &[VariantRow]| -> BTreeSet<String> {
        reach
            .gaps(&mix, variants)
            .unwrap_or_else(|e| panic!("{e}"))
            .into_iter()
            .filter(|g| g.side == Side::Real && g.class.kind == ClassKind::Pin)
            .map(|g| g.class.name)
            .collect()
    };
    let all = pins(&[]);
    let row = |family: &str| VariantRow {
        tag: "kcover".into(),
        family: family.into(),
        scope: model::fixture::VariantScope::Trunk,
        map: vec![("attn_q".into(), GgmlType::IQ4_XS)],
        recipes: vec!["gate-none".into()],
    };
    let mut bad = Vec::new();
    if !all.contains("attn_q iq4_xs") {
        bad.push(format!("the mix file's IQ4_XS attn_q is no gap: {all:?}"));
    }
    let netted = pins(&[row(spec.arch)]);
    let gone: Vec<&String> = all.difference(&netted).collect();
    if gone != ["attn_q iq4_xs"] {
        bad.push(format!("a variant of attn_q=iq4_xs takes {gone:?}"));
    }
    if pins(&[row("glm5next")]) != all {
        bad.push("a variant of another architecture takes a pin".into());
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Every fixture a pair plans reads, through its family's own kind check (`Family::check_kinds`,
/// what `fixture verify` runs on a written file), as the cut of its source: the minis at their
/// widths, the full-shape cuts and the MiMo fixture at the source's.
#[test]
fn kcover_each_planned_fixture_is_its_cut() {
    let mut bad = Vec::new();
    for p in pairs() {
        let fx = reach::fixture_header(&p.spec, &p.source)
            .unwrap_or_else(|e| panic!("{}: {e}", p.fixture));
        if let Err(e) = p.spec.family.check_kinds(&p.spec, &fx, &p.source) {
            bad.push(format!("{}: {e}", p.fixture));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The MiMo fixture records a planned card budget: the planner's least cap for the cut of its
/// source's header, under which the plan at the end-to-end gate's context holds half of the routed
/// experts on every card-eligible layer (`check_budget` proves the least-cap property), and the
/// budget is the plan's own, not a constant a plan can refuse: a plan under it succeeds, and one
/// at the old fixed 2 GiB is refused by name below its floor (the layer's dense, stores, context,
/// scratch and prompt reserve past the 2 GiB the old constant gave the experts).
#[test]
fn kcover_the_mimo_fixture_budget_is_its_plan() {
    let p = pairs()
        .into_iter()
        .find(|p| p.fixture == dir_of("mimo2"))
        .expect("the mimo2 pair");
    let fx = reach::fixture_header(&p.spec, &p.source).expect("the fixture's header");
    let model::fixture::CardBudget::Planned(planner) = p.spec.card_budget else {
        panic!("the mimo2 fixture's card budget is a constant, not a plan's value");
    };
    let (cap, _) = model::fixture::plan_budget(&p.spec.card_budget, &fx).expect("a budget plans");
    let got = model::fixture::check_budget(&planner, &fx, cap).expect("the cap holds its half");
    let full = (planner.card_experts)(&fx, None).expect("the budgetless plan");
    assert_eq!(got.experts, full.experts);
    for (l, (&n, &all)) in got.per_layer.iter().zip(&full.per_layer).enumerate() {
        assert_eq!(n, if all > 0 { all / 2 } else { 0 }, "layer {l}");
    }
    assert!(got.per_layer.iter().any(|&n| n > 0), "a card expert");
    // The old constant: 2 GiB is below the cut's floor (the plan's own terms, read from the
    // refusal's line: dense 1,082,130,432 + KV 16,711,680 + context 536,870,912 + scratch
    // 67,108,864 + margin 1,073,741,824 + beside 316,919,808 = 3,093,483,520 B).
    let e = (planner.card_experts)(&fx, Some(2 << 30)).expect_err("2 GiB is below the floor");
    assert!(e.to_string().contains("is below its floor 3093483520 B"), "{e}");
    assert!(cap > 3_093_483_520, "the planned cap {cap} holds the floor and the experts");
}
