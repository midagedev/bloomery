//! `--fit`'s contracts, host only and with no card: the units and their bytes from a plan, the table's refusals, the
//! selection (the least KLD per byte saved first, monotone in the budget, a budget no pick meets refused by name), the
//! rules the pick writes (read back by the rule reader to the types picked, the plan under them the pick's bytes to
//! the byte), and the whole `fit::run` over a synthetic exaone-moe set.
//!
//! FAIL-first (mutants, each applied alone on the Mac and red where noted): `select` taking the steps by falling slope
//! (the cheapest-first, budget, rules and run tests); `hull` without its chord pass or without its dominated drop (the
//! dominated and non-convex test); `stored` without `pad` (the ladder, rules and run tests); `units_for` without its
//! overlap check or its coverage check (the units test, by case); `select` ending at `bytes < budget` (the
//! cheapest-first, tie and dominated tests).

mod common;

use common::scratch;
use gguf::Value;
use gguf::quant::GgmlType;
use quantize::cli::{Args, Fit, Parsed, parse};
use quantize::fit::{
    CLASSES, Cand, Table, Unit, UnitKey, class_named, class_of, ladder, parse_bytes, render_ladder,
    rules_text, select, stem_of, type_name, units_for,
};
use quantize::plan::Plan;
use quantize::rules::Rules;
use quantize::run::Prepared;
use quantize::source::Source;
use quantize::{Error, Result};
use std::borrow::Cow;
use std::ops::Range;

/// The synthetic set: exaone-moe's shape at toy widths. Block 0 is dense, blocks 1 to 4 routed, block 5 the next-token
/// block; the routed stacks hold two experts of four rows of 256 values.
struct Synth {
    keys: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, GgmlType)>,
}

impl Synth {
    fn new() -> Synth {
        let keys = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("exaone-moe".to_string()),
            ),
            ("exaone-moe.block_count".to_string(), Value::U32(6)),
            ("exaone-moe.expert_count".to_string(), Value::U32(2)),
        ];
        let mut t: Vec<(String, Vec<u64>, GgmlType)> = Vec::new();
        let mut add = |n: String, d: &[u64], ty: GgmlType| t.push((n, d.to_vec(), ty));
        add("token_embd.weight".into(), &[256, 8], GgmlType::BF16);
        add("output.weight".into(), &[256, 8], GgmlType::BF16);
        add("output_norm.weight".into(), &[256], GgmlType::F32);
        for l in 0..5 {
            add(format!("blk.{l}.attn_q.weight"), &[256, 8], GgmlType::BF16);
            add(format!("blk.{l}.attn_norm.weight"), &[256], GgmlType::F32);
        }
        for l in 0..2 {
            for s in ["ffn_gate", "ffn_up", "ffn_down"] {
                add(format!("blk.{l}.{s}.weight"), &[256, 4], GgmlType::BF16);
            }
        }
        for l in 1..5 {
            for s in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                add(format!("blk.{l}.{s}.weight"), &[256, 4, 2], GgmlType::BF16);
            }
            add(
                format!("blk.{l}.ffn_gate_inp.weight"),
                &[256, 2],
                GgmlType::F32,
            );
            add(
                format!("blk.{l}.ffn_down_shexp.weight"),
                &[256, 4],
                GgmlType::BF16,
            );
        }
        add(
            "blk.5.nextn.eh_proj.weight".into(),
            &[256, 4],
            GgmlType::BF16,
        );
        Synth { keys, tensors: t }
    }
}

impl Source for Synth {
    fn label(&self) -> String {
        "synthetic exaone-moe".to_string()
    }
    fn keys(&self) -> &[(String, Value)] {
        &self.keys
    }
    fn shard_count(&self) -> usize {
        1
    }
    fn tensors(&self) -> Vec<(usize, gguf::TensorInfo)> {
        let mut off = 0;
        self.tensors
            .iter()
            .map(|(name, dims, ty)| {
                let n: u64 = dims.iter().product();
                let nbytes = n / ty.blck_size().unwrap() * ty.type_size().unwrap();
                let t = gguf::TensorInfo {
                    name: name.clone(),
                    dims: dims.clone(),
                    ty: *ty,
                    offset: off,
                    nbytes,
                };
                off += nbytes;
                (0, t)
            })
            .collect()
    }
    fn read(&self, name: &str, _bytes: Range<u64>) -> Result<Cow<'_, [u8]>> {
        Err(Error::Refused(format!(
            "{name}: the synthetic set holds no bytes"
        )))
    }
}

/// The bootstrap rules of exaone-moe's file at toy scale: the routed stacks Q4_K, the trunk and the lead Q8_0.
const BASE: &str = "ffn_(down|gate|up)_exps\\.weight$=q4_k\n\
    \\.(attn_q|ffn_down|ffn_gate|ffn_up|ffn_down_shexp)\\.weight$=q8_0\n\
    nextn\\.eh_proj\\.weight$=q8_0\n";

fn args() -> Args {
    let Parsed::Run(a) = parse(
        [
            "in",
            "out",
            "q4_k_m",
            "--plan",
            "--output-tensor-type",
            "q8_0",
            "--token-embedding-type",
            "q8_0",
        ]
        .map(String::from),
    )
    .unwrap() else {
        panic!("a run")
    };
    a
}

fn plan(set: &Synth, rules: &str) -> Plan {
    Plan::build(set, &Rules::parse(rules).unwrap(), None, None, &args()).unwrap()
}

fn unit(class: &str, layers: Option<(i64, i64)>) -> UnitKey {
    UnitKey::new(class_named(class).unwrap(), layers)
}

fn cand(ty: GgmlType, bytes: u64, dkld: f64) -> Cand {
    Cand {
        ty,
        bytes,
        dkld,
        se: 0.0,
    }
}

const GU: &str = "routed_gate_up";
const DN: &str = "routed_down";

#[test]
fn the_ladder_names_each_units_types_and_padded_bytes() {
    let set = Synth::new();
    let p = plan(&set, BASE);
    let l = ladder(&p, "exaone-moe", 0).unwrap();
    let keys: Vec<String> = l.units.iter().map(|u| u.key.to_string()).collect();
    assert_eq!(keys, [format!("{GU}@1-4"), format!("{DN}@1-4")]);
    // classes the reader gives one type: no choice
    assert_eq!(
        l.single,
        ["attn", "shared", "dense_ffn", "nextn", "head", "embed"]
    );
    let bytes = |u: usize| -> Vec<(GgmlType, u64)> {
        l.units[u].cands.iter().map(|c| (c.ty, c.bytes)).collect()
    };
    // gate and up of four layers: 8 tensors of 8 rows; Q4_K 144 B a row, IQ2_XXS 66, IQ1_S 50, each tensor padded to 32
    let rows = 8 * 8;
    assert_eq!(
        bytes(0),
        [
            (GgmlType::Q4_K, 144 * rows),
            (GgmlType::IQ2_XXS, 4 * 2 * (66 * 8u64).next_multiple_of(32)),
            (GgmlType::IQ1_S, 4 * 2 * (50 * 8u64).next_multiple_of(32)),
        ]
    );
    assert_eq!(
        bytes(1),
        [
            (GgmlType::Q4_K, 4 * 144 * 8),
            (GgmlType::IQ1_S, 4 * (50 * 8u64).next_multiple_of(32)),
        ]
    );
    let g2 = ladder(&p, "exaone-moe", 2).unwrap();
    let keys: Vec<String> = g2.units.iter().map(|u| u.key.to_string()).collect();
    assert_eq!(
        keys,
        [
            format!("{GU}@1-1"),
            format!("{GU}@2-3"),
            format!("{GU}@4-4"),
            format!("{DN}@1-1"),
            format!("{DN}@2-3"),
            format!("{DN}@4-4"),
        ]
    );
    let text = render_ladder(&l, 0, p.out_file_bytes);
    assert!(text.contains(&format!(
        "{GU}\t1-4\tIQ1_S\t{}\t^blk\\.(1|2|3|4)\\.(ffn_gate_exps\\.weight|ffn_up_exps\\.weight)$=iq1_s\n",
        4 * 2 * (50 * 8u64).next_multiple_of(32)
    )));
    // the reader's list decides: a model the table does not know is refused by name
    let e = ladder(&p, "no-such-arch", 0).unwrap_err();
    assert!(e.to_string().contains("no-such-arch"), "{e}");
}

#[test]
fn a_stem_and_its_class() {
    assert_eq!(stem_of("blk.12.ffn_up_exps.weight"), "ffn_up_exps.weight");
    assert_eq!(
        stem_of("blk.6.nextn.eh_proj.weight"),
        "nextn.eh_proj.weight"
    );
    assert_eq!(stem_of("output.weight"), "output.weight");
    assert_eq!(class_of("blk.3.ffn_gate_exps.weight").unwrap().name, GU);
    assert_eq!(class_of("blk.3.ffn_up_exps.weight").unwrap().name, GU);
    assert!(class_of("blk.3.ffn_norm.weight").is_none());
    let names: Vec<&str> = CLASSES.iter().map(|c| c.name).collect();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len()
    );
}

const HEAD: &str = "class\tlayers\ttype\tdkld\tse\tpositions\n";

fn table(rows: &str) -> Result<Table> {
    Table::parse(&format!("# a measured table\n{HEAD}{rows}"), "t.tsv")
}

#[test]
fn a_table_refuses_what_it_cannot_read_by_line() {
    assert!(
        table("routed_down\t1-4\tQ4_K\t0\t0\t100\nrouted_down\t1-4\tIQ1_S\t0.2\t0.01\t100\n")
            .is_ok()
    );
    for (rows, why) in [
        ("routed_down\t1-4\tQ4_K\t0\t0\n", "5 columns"),
        ("rooted\t1-4\tQ4_K\t0\t0\t9\n", "class \"rooted\""),
        ("routed_down\t4-1\tQ4_K\t0\t0\t9\n", "layers \"4-1\""),
        ("routed_down\tx\tQ4_K\t0\t0\t9\n", "layers \"x\""),
        ("routed_down\t1\tF32\t0\t0\t9\n", "not one a fit can choose"),
        ("routed_down\t1\tQ3_K\t0\t0\t9\n", "Q3_K"),
        ("routed_down\t1\tQ4_K\t-0.1\t0\t9\n", "dkld \"-0.1\""),
        ("routed_down\t1\tQ4_K\tNaN\t0\t9\n", "dkld \"NaN\""),
        ("routed_down\t1\tQ4_K\t0\t0\t0\n", "positions \"0\""),
        (
            "routed_down\t1\tQ4_K\t0\t0\t9\nrouted_down\t1\tQ4_K\t0\t0\t9\n",
            "given twice",
        ),
    ] {
        let e = table(rows).unwrap_err().to_string();
        assert!(e.starts_with("t.tsv line"), "{e}");
        assert!(e.contains(why), "{rows:?}: {e}");
    }
    assert!(Table::parse("routed_down\t1\tQ4_K\t0\t0\t9\n", "t.tsv").is_err());
    assert!(Table::parse(HEAD, "t.tsv").is_err());
    assert!(Table::parse("", "t.tsv").is_err());
}

/// A table over all the ladder's candidates of the synthetic set, by unit, each type's KLD from `dk`.
fn full_table(group: usize, dk: impl Fn(&str, GgmlType) -> f64) -> String {
    let set = Synth::new();
    let l = ladder(&plan(&set, BASE), "exaone-moe", group).unwrap();
    let mut s = String::from(HEAD);
    for u in &l.units {
        for (i, c) in u.cands.iter().enumerate() {
            let d = if i == 0 {
                0.0
            } else {
                dk(u.key.class().name, c.ty)
            };
            s += &format!(
                "{}\t{}\t{}\t{d}\t0.001\t2000\n",
                u.key.class().name,
                u.key.layers_text(),
                c.ty
            );
        }
    }
    s
}

fn sensitivity(ty: &str, c: &str) -> f64 {
    match (c, ty) {
        (GU, "IQ2_XXS") => 0.05,
        (GU, "IQ1_S") => 0.30,
        (DN, "IQ1_S") => 0.12,
        _ => panic!("{c} {ty}"),
    }
}

#[test]
fn the_table_over_a_plan_holds_every_tensor_once_and_the_bytes_add_up() {
    let set = Synth::new();
    let p = plan(&set, BASE);
    let a = args();
    let t = Table::parse(&full_table(0, |c, t| sensitivity(&type_name(t), c)), "t").unwrap();
    let f = units_for(&p, "exaone-moe", &t, &a).unwrap();
    // every unit at its largest type is the plan under the base rules: the bytes and the headers of the rest are fixed
    let top: u64 = f
        .units
        .iter()
        .map(|u| u.cands.iter().map(|c| c.bytes).max().unwrap())
        .sum();
    assert_eq!(f.fixed + top, p.out_file_bytes);

    let cases: [(&str, &str); 4] = [
        (
            "routed_down\t1-2\tQ4_K\t0\t0\t9\nrouted_down\t2-4\tQ4_K\t0\t0\t9\n",
            "is in two units",
        ),
        (
            "routed_down\t1-3\tQ4_K\t0\t0\t9\n",
            "covers for other layers only",
        ),
        ("routed_down\t7-8\tQ4_K\t0\t0\t9\n", "holds no tensor"),
        ("routed_down\t1-4\tIQ2_XXS\t0\t0\t9\n", "allow Q4_K, IQ1_S"),
    ];
    for (rows, why) in cases {
        let e = units_for(&p, "exaone-moe", &table(rows).unwrap(), &a)
            .unwrap_err()
            .to_string();
        assert!(e.contains(why), "{rows:?}: {e}");
    }
    // a flag-typed tensor cannot be a unit
    let e = units_for(
        &p,
        "exaone-moe",
        &table("head\t-\tQ8_0\t0\t0\t9\n").unwrap(),
        &a,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("--output-tensor-type"), "{e}");
}

fn units(spec: &[&[(GgmlType, u64, f64)]]) -> Vec<Unit> {
    spec.iter()
        .enumerate()
        .map(|(i, cs)| Unit {
            key: unit(GU, Some((i as i64, i as i64))),
            tensors: Vec::new(),
            cands: cs.iter().map(|&(t, b, d)| cand(t, b, d)).collect(),
        })
        .collect()
}

use GgmlType::{IQ1_S, IQ2_XXS, Q4_K, Q8_0};

/// Three units: `a` costs 0.02 per 100 B saved to its middle type, `b` 0.2, `c` has one type.
fn three() -> Vec<Unit> {
    units(&[
        &[(Q4_K, 1000, 0.0), (IQ2_XXS, 600, 0.08), (IQ1_S, 400, 0.30)],
        &[(Q4_K, 1000, 0.0), (IQ1_S, 500, 1.0)],
        &[(Q8_0, 700, 0.0)],
    ])
}

/// The least KLD per byte saved goes first, and the pick is monotone in the budget: a larger budget never gives any
/// unit fewer bytes, never more loss.
#[test]
fn the_cheapest_step_per_byte_goes_first_and_the_pick_is_monotone() {
    let u = three();
    let top = 1000 + 1000 + 700;
    let pick = select(&u, 0, top).unwrap();
    assert_eq!(
        (pick.steps, pick.choice.clone(), pick.bytes),
        (0, vec![0, 0, 0], top)
    );
    // one step: a's 1000 -> 600 costs 0.08 / 400 B = 2e-4 a byte, b's 1.0 / 500 = 2e-3, a's second 0.22 / 200 = 1.1e-3
    let pick = select(&u, 0, top - 1).unwrap();
    assert_eq!(pick.choice, vec![1, 0, 0]);
    assert!((pick.dkld - 0.08).abs() < 1e-12);
    // a's second step (1.1e-3) comes before b's (2e-3)
    let pick = select(&u, 0, top - 400 - 1).unwrap();
    assert_eq!(pick.choice, vec![2, 0, 0]);
    let pick = select(&u, 0, top - 600 - 1).unwrap();
    assert_eq!(pick.choice, vec![2, 1, 0]);
    assert_eq!((pick.steps, pick.steps_total), (3, 3));

    let mut last: Option<(Vec<u64>, f64)> = None;
    for budget in (1600..=top).rev().step_by(7) {
        let p = select(&u, 0, budget).unwrap();
        assert!(p.bytes <= budget, "{budget}: {}", p.bytes);
        let sizes: Vec<u64> = p
            .choice
            .iter()
            .zip(&u)
            .map(|(&c, un)| un.cands[c].bytes)
            .collect();
        if let Some((prev, d)) = &last {
            assert!(
                sizes.iter().zip(prev).all(|(a, b)| a <= b),
                "{budget}: a smaller budget gave a unit more bytes"
            );
            assert!(p.dkld >= *d - 1e-12, "{budget}: a smaller budget lost less");
        }
        last = Some((sizes, p.dkld));
    }
}

/// The smallest pick is allowed to meet the budget exactly and a byte less is refused with both numbers.
#[test]
fn a_budget_no_pick_can_meet_is_refused_by_name() {
    let u = three();
    let smallest = 400 + 500 + 700;
    let p = select(&u, 0, smallest).unwrap();
    assert_eq!((p.bytes, p.choice), (smallest, vec![2, 1, 0]));
    let e = select(&u, 0, smallest - 1).unwrap_err().to_string();
    assert!(
        e.contains(&format!("{} B cannot be met", smallest - 1))
            && e.contains(&format!(
                "smallest file this table can pick is {smallest} B"
            )),
        "{e}"
    );
    let e = select(&u, 5, smallest + 4).unwrap_err().to_string();
    assert!(
        e.contains(&format!("{} B", smallest + 5)),
        "fixed bytes count: {e}"
    );
}

/// A type another one beats at no more bytes is never picked, and a point above its neighbours' chord is stepped over.
#[test]
fn a_dominated_type_is_dropped_and_a_non_convex_one_is_stepped_over() {
    // Q8_0 at 900 bytes is worse than Q4_K at 800: dominated. IQ2_XXS (500, 0.5) lies above the chord of
    // (800, 0.01) and (200, 0.6): the steps are 800 -> 200 at 1e-3 a byte; the hull skips the middle.
    let u = units(&[&[
        (Q8_0, 900, 0.02),
        (Q4_K, 800, 0.01),
        (IQ2_XXS, 500, 0.5),
        (IQ1_S, 200, 0.6),
    ]]);
    for budget in [800, 799, 500, 200] {
        let p = select(&u, 0, budget).unwrap();
        let ty = u[0].cands[p.choice[0]].ty;
        assert!(matches!(ty, Q4_K | IQ1_S), "{budget}: {ty}");
        assert_eq!(ty == IQ1_S, budget < 800, "{budget}");
    }
    let p = select(&u, 0, 900).unwrap();
    assert_eq!(
        u[0].cands[p.choice[0]].ty, Q4_K,
        "the dominated Q8_0 is never the top"
    );
}

#[test]
fn ties_break_by_unit_so_the_pick_is_the_same_every_time() {
    let u = units(&[
        &[(Q4_K, 1000, 0.0), (IQ1_S, 500, 0.1)],
        &[(Q4_K, 1000, 0.0), (IQ1_S, 500, 0.1)],
    ]);
    let p = select(&u, 0, 1500).unwrap();
    assert_eq!(
        p.choice,
        vec![1, 0],
        "the earlier unit takes the first of two equal steps"
    );
    assert_eq!(select(&u, 0, 1500).unwrap(), p);
}

#[test]
fn byte_counts_read_with_their_units() {
    for (s, b) in [
        ("6500000000", 6_500_000_000),
        ("6.5GB", 6_500_000_000),
        ("6.5gb", 6_500_000_000),
        ("1GiB", 1 << 30),
        ("1.5MiB", 1_572_864),
        ("12B", 12),
    ] {
        assert_eq!(parse_bytes(s).unwrap(), b, "{s}");
    }
    for bad in [
        "",
        "GB",
        "0",
        "0.0GB",
        "-1",
        "3 ",
        "1XB",
        "1.2.3",
        "99999999999999999999TB",
    ] {
        let e = parse_bytes(bad).unwrap_err();
        assert_eq!(e.status(), 64, "{bad:?}: {e}");
    }
}

/// The pick's rules, read back by the rule reader: each unit's tensors take the type picked, every other tensor the
/// base rules', and the plan under them is the pick's bytes to the byte.
#[test]
fn the_rules_a_pick_writes_type_exactly_its_units_and_the_plan_is_its_bytes() {
    let set = Synth::new();
    let p = plan(&set, BASE);
    let a = args();
    let text = full_table(1, |c, t| sensitivity(&type_name(t), c));
    let t = Table::parse(&text, "t").unwrap();
    let f = units_for(&p, "exaone-moe", &t, &a).unwrap();
    let top: u64 = f.fixed
        + f.units
            .iter()
            .map(|u| u.cands.iter().map(|c| c.bytes).max().unwrap())
            .sum::<u64>();
    assert_eq!(top, p.out_file_bytes);
    // the budgets from the top to the smallest: at each the plan under the rules is the pick
    let low: u64 = f.fixed
        + f.units
            .iter()
            .map(|u| u.cands.iter().map(|c| c.bytes).min().unwrap())
            .sum::<u64>();
    let mut seen = std::collections::BTreeSet::new();
    for budget in (low..=top).rev().step_by(53).chain([low, top]) {
        let pick = select(&f.units, f.fixed, budget).unwrap();
        let rules = rules_text(&f.units, &pick, Some(BASE));
        let r = Rules::parse(&rules).unwrap();
        assert_eq!(
            r.len(),
            f.units.len() + 3,
            "a token a unit, then the base's"
        );
        for (u, &c) in f.units.iter().zip(&pick.choice) {
            let want = type_name(u.cands[c].ty);
            for &i in &u.tensors {
                assert_eq!(
                    r.type_of(&p.items[i].name),
                    Some(want.as_str()),
                    "{}",
                    p.items[i].name
                );
            }
        }
        // a tensor of no unit keeps the base rules' type
        assert_eq!(r.type_of("blk.0.attn_q.weight"), Some("Q8_0"));
        assert_eq!(r.type_of("blk.5.nextn.eh_proj.weight"), Some("Q8_0"));
        let again = Plan::build(&set, &r, None, None, &a).unwrap();
        assert_eq!(again.out_file_bytes, pick.bytes, "budget {budget}");
        assert!(pick.bytes <= budget);
        seen.insert(pick.bytes);
    }
    assert!(
        seen.len() > 5,
        "the budgets walked {} distinct files",
        seen.len()
    );
}

/// The whole run over files: the rules land beside the plan's check, an existing file is refused, and a budget no
/// pick meets leaves no file.
#[test]
fn fit_run_writes_its_rules_once_and_refuses_what_it_cannot_meet() {
    let dir = scratch("fit-run");
    let table_path = dir.join("t.tsv");
    let rules_path = dir.join("base.rules");
    std::fs::write(
        &table_path,
        full_table(0, |c, t| sensitivity(&type_name(t), c)),
    )
    .unwrap();
    std::fs::write(&rules_path, BASE).unwrap();
    let set = Synth::new();
    let base_plan = plan(&set, BASE);
    let mk = |budget: u64, out: &str| -> Args {
        let Parsed::Run(a) = parse(
            [
                "in",
                "out",
                "q4_k_m",
                "--output-tensor-type",
                "q8_0",
                "--token-embedding-type",
                "q8_0",
                "--tensor-type-file",
                rules_path.to_str().unwrap(),
                "--fit",
                &budget.to_string(),
                "--sensitivity",
                table_path.to_str().unwrap(),
                "--fit-out",
                dir.join(out).to_str().unwrap(),
            ]
            .map(String::from),
        )
        .unwrap() else {
            panic!("a run")
        };
        a
    };
    let prep = |a: &Args| Prepared {
        set: Box::new(Synth::new()),
        im: None,
        plan: Plan::build(&set, &Rules::parse(BASE).unwrap(), None, None, a).unwrap(),
    };
    let budget = base_plan.out_file_bytes - 5000;
    let a = mk(budget, "fit.rules");
    let Some(fit @ Fit::Select { .. }) = a.fit.as_deref().cloned() else {
        panic!("a select")
    };
    let out = quantize::fit::run(&a, &prep(&a), &fit).unwrap();
    assert!(out.contains("bloomery-quantize --fit: budget"), "{out}");
    assert!(out.contains("the plan under them holds the same"), "{out}");
    let written = std::fs::read_to_string(dir.join("fit.rules")).unwrap();
    assert!(written.ends_with(&format!("{}\n", BASE.split_whitespace().last().unwrap())));
    let r = Rules::parse(&written).unwrap();
    let again = Plan::build(&set, &r, None, None, &a).unwrap();
    assert!(again.out_file_bytes <= budget && again.out_file_bytes > budget - 3000);

    // an existing output is refused and left as it is
    let e = quantize::fit::run(&a, &prep(&a), &fit).unwrap_err();
    assert!(matches!(e, Error::Io { .. }), "{e}");
    assert_eq!(
        std::fs::read_to_string(dir.join("fit.rules")).unwrap(),
        written
    );

    // a budget below the smallest pick: named, and no file
    let a = mk(1000, "none.rules");
    let Some(fit) = a.fit.as_deref().cloned() else {
        panic!("a fit")
    };
    let e = quantize::fit::run(&a, &prep(&a), &fit).unwrap_err();
    assert!(e.to_string().contains("cannot be met"), "{e}");
    assert!(!dir.join("none.rules").exists());

    let a = mk(budget, "ladder.rules");
    let text = quantize::fit::run(&a, &prep(&a), &Fit::Ladder { group: 0 }).unwrap();
    assert!(
        text.starts_with("# bloomery-quantize --fit-ladder: 2 units"),
        "{text}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
