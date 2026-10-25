use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GENERAL_ARCHITECTURE, GgmlType, Split, Value};

use super::{Row, Scope, check, custom_q, expand, override_kv, parse_table, record_budget, row};
use crate::fixture::{Budget, CardBudget, CardExperts, FixtureError, KEY_CARD_BUDGET, KEY_VARIANT};

const TABLE_ROW: &str = "v1\tfam\ttrunk\tffn_gate_exps=iq3_s ffn_up_exps=iq3_s ffn_down_exps=iq4_xs\tgate-a gate-b\ta reason";

static DIRS: AtomicUsize = AtomicUsize::new(0);

/// A fresh temporary directory, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Dir {
        let d = std::env::temp_dir().join(format!(
            "bloomery-fixture-variant-{}-{}",
            std::process::id(),
            DIRS.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).expect("a temporary directory");
        Dir(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The tensor of `layer`'s `role`: `<block prefix>.<layer>.<role>.weight`.
fn tn(layer: usize, role: &str) -> String {
    ["blk", &layer.to_string(), role, "weight"].join(".")
}

/// One tensor of a written set: name, dims, type, shard, and its bytes.
#[derive(Clone)]
struct T {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    shard: usize,
    bytes: Vec<u8>,
}

fn size_of(ty: GgmlType, dims: &[u64]) -> usize {
    let (blck, size) = (ty.blck_size().unwrap(), ty.type_size().unwrap());
    (size * (dims[0] / blck) * dims[1..].iter().product::<u64>()) as usize
}

/// `n` bytes from `seed`, so two tensors of one shape differ.
fn bytes(seed: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn t(name: String, dims: &[u64], ty: GgmlType, shard: usize, seed: u8) -> T {
    T {
        name,
        dims: dims.to_vec(),
        ty,
        shard,
        bytes: bytes(seed, size_of(ty, dims)),
    }
}

/// The source's tensors: a dense layer 0, two trunk layers with routed stacks (1, 2) and the NextN layer 3,
/// which carries stacks of its own.
fn source_tensors() -> Vec<T> {
    use GgmlType::*;
    let stack = [256, 2, 3];
    vec![
        t(tn(0, "attn_q"), &[32, 4], Q8_0, 0, 1),
        t(tn(0, "attn_norm"), &[8], F32, 0, 2),
        t(tn(1, "ffn_gate_exps"), &stack, Q4_K, 0, 3),
        t(tn(1, "ffn_up_exps"), &stack, Q4_K, 0, 4),
        t(tn(1, "ffn_down_exps"), &stack, Q5_K, 1, 5),
        t(tn(2, "ffn_gate_exps"), &stack, Q4_K, 1, 6),
        t(tn(2, "ffn_up_exps"), &stack, Q4_K, 1, 7),
        t(tn(2, "ffn_down_exps"), &stack, Q5_K, 1, 8),
        t(tn(3, "nextn.eh_proj"), &[32, 4], Q8_0, 1, 9),
        t(tn(3, "ffn_gate_exps"), &stack, Q4_K, 1, 10),
        t(tn(3, "ffn_up_exps"), &stack, Q4_K, 1, 11),
        t(tn(3, "ffn_down_exps"), &stack, Q5_K, 1, 12),
    ]
}

fn source_kvs() -> Vec<(String, Value)> {
    vec![
        (GENERAL_ARCHITECTURE.into(), Value::String("fam".into())),
        ("general.file_type".into(), Value::U32(15)),
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("bloomery.fixture.version".into(), Value::U32(1)),
        // The planner's value for the trunk: 4 bytes an expert, half of 8, cap 16.
        (
            "bloomery.fixture.card_budget".into(),
            Value::U64(TRUNK_BUDGET),
        ),
        (
            "quantize.imatrix.file".into(),
            Value::String("source.imatrix".into()),
        ),
        ("quantize.imatrix.entries_count".into(), Value::U32(7)),
    ]
}

/// The trunk's recorded card budget, what the planner of [`card_budget`] finds for the source.
const TRUNK_BUDGET: u64 = 16;
/// The variant's own, the same planner over the variant's smaller stacks.
const VARIANT_BUDGET: u64 = 8;

/// The family's budget the checks plan with: an expert costs 4 bytes on the trunk and 2 on the
/// variant (its stacks take the map's smaller types), so the smallest cap holding half of the 8
/// experts on each of the two card-eligible layers is 16 for the trunk and 8 for the variant (the
/// whole-MiB rounding keeps neither).
fn card_budget() -> CardBudget {
    CardBudget::Planned(Budget {
        ctx: 1,
        card_experts: |split, cap| {
            let per = if split.value(KEY_VARIANT).is_some() {
                2
            } else {
                4
            };
            let n = match cap {
                None => 8,
                Some(0) => {
                    return Err(FixtureError::Budget("no room for dense weights".into()));
                }
                Some(c) => (c / per).min(8),
            };
            Ok(CardExperts {
                per_layer: vec![n, 0, n],
                experts: 8,
            })
        },
        card_set_aside: None,
    })
}

/// The variant `source` gives under the table row: the trunk stacks at the map's types.
fn variant_tensors(source: &[T]) -> Vec<T> {
    let mut out = source.to_vec();
    for x in &mut out {
        let trunk = (1..=2).any(|l| {
            [tn(l, "ffn_gate_exps"), tn(l, "ffn_up_exps")].contains(&x.name)
                || x.name == tn(l, "ffn_down_exps")
        });
        if !trunk {
            continue;
        }
        x.ty = if (1..=2).any(|l| x.name == tn(l, "ffn_down_exps")) {
            GgmlType::IQ4_XS
        } else {
            GgmlType::IQ3_S
        };
        x.bytes = bytes(100, size_of(x.ty, &x.dims));
    }
    out
}

/// Set `key`'s value in `kvs`, which must hold it.
fn set(kvs: &mut [(String, Value)], key: &str, v: Value) {
    kvs.iter_mut()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("{key} is not in the kvs"))
        .1 = v;
}

fn variant_kvs() -> Vec<(String, Value)> {
    let mut kvs = source_kvs();
    set(&mut kvs, "general.file_type", Value::U32(14));
    // The quantizer's own words about the matrix it was given, not the file's content.
    set(
        &mut kvs,
        "quantize.imatrix.file",
        Value::String("flat.imatrix".into()),
    );
    set(&mut kvs, "quantize.imatrix.entries_count", Value::U32(6));
    // The budget the quantizer copies is the trunk's; the variant's own replaces it
    // (`record_budget`).
    set(
        &mut kvs,
        "bloomery.fixture.card_budget",
        Value::U64(VARIANT_BUDGET),
    );
    kvs.push((KEY_VARIANT.into(), Value::String("v1".into())));
    kvs
}

/// The variant's header with `budget` recorded, the quantizer's own state at `TRUNK_BUDGET`.
fn variant_kvs_at(budget: u64) -> Vec<(String, Value)> {
    let mut kvs = variant_kvs();
    set(&mut kvs, "bloomery.fixture.card_budget", Value::U64(budget));
    kvs
}

/// Write `tensors` and `kvs` as a set of `shards` files named `stem`-0000N-of-0000M.gguf in `dir`; the
/// first shard's path.
fn write(dir: &Path, stem: &str, kvs: &[(String, Value)], tensors: &[T], shards: usize) -> PathBuf {
    let mut first = PathBuf::new();
    for s in 0..shards {
        let mut kv = kvs.to_vec();
        if shards > 1 {
            kv.push(("split.no".into(), Value::U16(s as u16)));
            kv.push(("split.count".into(), Value::U16(shards as u16)));
            kv.push((
                "split.tensors.count".into(),
                Value::I32(tensors.len() as i32),
            ));
        }
        let mine: Vec<&T> = tensors.iter().filter(|x| x.shard == s).collect();
        let decls = mine
            .iter()
            .map(|x| TensorDecl {
                name: x.name.clone(),
                dims: x.dims.clone(),
                type_id: x.ty.as_u32(),
                nbytes: x.bytes.len() as u64,
            })
            .collect();
        let path = dir.join(format!("{stem}-{:05}-of-{:05}.gguf", s + 1, shards));
        let layout = Layout::new(&kv, decls).expect("a layout");
        let mut w = Writer::new(BufWriter::new(File::create(&path).expect("a file")), layout)
            .expect("a header");
        for x in mine {
            w.tensor(&x.name, &x.bytes).expect("a tensor");
        }
        w.finish().expect("a whole file");
        if s == 0 {
            first = path;
        }
    }
    first
}

fn the_row() -> Row {
    row(&parse_table(TABLE_ROW).expect("a table"), "v1")
        .expect("v1")
        .clone()
}

fn open(p: &Path) -> Split {
    Split::open(p).expect("a set")
}

fn red(e: Result<impl std::fmt::Debug, FixtureError>, parts: &[&str]) {
    let e = e.expect_err("a refusal").to_string();
    for p in parts {
        assert!(e.contains(p), "{p:?} is not in the refusal {e:?}");
    }
}

/// The table's lines are six tab columns, and each refusal names what it refused.
#[test]
fn the_table_parses_and_refuses_by_name() {
    let rows = parse_table(&format!("# a comment\n\n{TABLE_ROW}\n")).expect("one row");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].family, "fam");
    assert_eq!(
        rows[0].map,
        [
            ("ffn_gate_exps".to_string(), GgmlType::IQ3_S),
            ("ffn_up_exps".to_string(), GgmlType::IQ3_S),
            ("ffn_down_exps".to_string(), GgmlType::IQ4_XS),
        ]
    );
    assert_eq!(rows[0].recipes, ["gate-a", "gate-b"]);
    red(row(&rows, "nope"), &["nope", "names no such variant", "v1"]);
    let bad = |line: &str, parts: &[&str]| red(parse_table(line), parts);
    bad("v1\tfam\ttrunk\tx=q4_K", &["4 columns"]);
    bad(&format!("{TABLE_ROW}\n{TABLE_ROW}"), &["v1", "twice"]);
    bad("V1\tfam\ttrunk\ta=q4_K\tg\tr", &["lower-case"]);
    bad(
        "v1\tfam\tpool\ta=q4_K\tg\tr",
        &["scope", "`trunk` or `all`"],
    );
    let all = parse_table("v1\tfam\tall\ta=q4_K\tg\tr").expect("an all-scope row");
    assert_eq!(all[0].scope, Scope::All);
    bad("v1\tfam\ttrunk\ta=q9_9\tg\tr", &["q9_9"]);
    bad("v1\tfam\ttrunk\ta\tg\tr", &["not role=type"]);
    bad("v1\tfam\ttrunk\ta=q4_K a=q5_K\tg\tr", &["twice"]);
    bad("v1\tfam\ttrunk\t\tg\tr", &["not empty"]);
    bad("v1\tfam\ttrunk\ta=q4_K\t\tr", &["not empty"]);
    bad("v1\tfam\ttrunk\ta=q4_K\tg\t ", &["not empty"]);
}

/// The shipped table is the one `fixture variant-rules` reads: every row parses, with each map word a type ggml names.
#[test]
fn the_shipped_table_parses() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/fixture-variants.tsv"
    ))
    .expect("tools/fixture-variants.tsv");
    let rows = parse_table(&text).expect("the shipped table parses");
    assert!(!rows.is_empty(), "the shipped table has no row");
}

/// A map reaches the trunk layers' tensors of its roles in the source's order, and the NextN layer keeps its types.
#[test]
fn a_map_expands_to_the_trunk_tensors() {
    let d = Dir::new();
    let src = open(&write(&d.0, "s", &source_kvs(), &source_tensors(), 2));
    let got = expand(&the_row(), &src).expect("an expansion");
    let names: Vec<&str> = got.iter().map(|r| r.name.as_str()).collect();
    let want: Vec<String> = [
        tn(1, "ffn_gate_exps"),
        tn(1, "ffn_up_exps"),
        tn(1, "ffn_down_exps"),
        tn(2, "ffn_gate_exps"),
        tn(2, "ffn_up_exps"),
        tn(2, "ffn_down_exps"),
    ]
    .into();
    assert_eq!(names, want);
    assert!(
        got.iter().all(|r| !r.name.contains(&tn(3, ""))),
        "the NextN layer keeps its types"
    );
    assert_eq!((got[0].from, got[0].to), (GgmlType::Q4_K, GgmlType::IQ3_S));
    assert_eq!((got[2].from, got[2].to), (GgmlType::Q5_K, GgmlType::IQ4_XS));
}

/// The `all` scope reaches the NextN layer's tensors of the map's roles too, in the source's order; `trunk` leaves them.
#[test]
fn the_all_scope_reaches_the_nextn_layer() {
    let d = Dir::new();
    let src = open(&write(&d.0, "s", &source_kvs(), &source_tensors(), 2));
    let all = Row {
        scope: Scope::All,
        ..the_row()
    };
    let names: Vec<String> = expand(&all, &src)
        .expect("an expansion")
        .into_iter()
        .map(|r| r.name)
        .collect();
    let want: Vec<String> = [1, 2, 3]
        .into_iter()
        .flat_map(|l| ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(|r| tn(l, r)))
        .collect();
    assert_eq!(names, want);
}

/// A row that changes nothing, names nothing, names another family or cannot be held is refused by name.
#[test]
fn a_map_that_bites_nothing_is_refused() {
    let d = Dir::new();
    let src = open(&write(&d.0, "s", &source_kvs(), &source_tensors(), 2));
    let with = |map: &[(&str, GgmlType)]| Row {
        map: map.iter().map(|(r, t)| ((*r).to_string(), *t)).collect(),
        ..the_row()
    };
    red(
        expand(&with(&[("ffn_missing_exps", GgmlType::IQ3_S)]), &src),
        &["ffn_missing_exps", "NextN layer"],
    );
    red(
        expand(&with(&[("ffn_gate_exps", GgmlType::Q4_K)]), &src),
        &[
            &format!("{}, {}", tn(1, "ffn_gate_exps"), tn(2, "ffn_gate_exps")),
            "already at q4_K",
        ],
    );
    red(
        expand(&with(&[("attn_q", GgmlType::IQ3_S)]), &src),
        &[&tn(0, "attn_q"), "whole blocks"],
    );
    // The NextN layer's own tensor is outside the scope, so a role only it has is a role nothing has.
    red(
        expand(&with(&[("nextn.eh_proj", GgmlType::Q4_K)]), &src),
        &["nextn.eh_proj", "no blk.<L>."],
    );
    red(
        expand(
            &Row {
                family: "other".into(),
                ..the_row()
            },
            &src,
        ),
        &["variant of other", "\"fam\""],
    );
    let mut kvs = source_kvs();
    kvs.push((KEY_VARIANT.into(), Value::String("v0".into())));
    let carried = open(&write(&d.0, "c", &kvs, &source_tensors(), 1));
    red(
        expand(&the_row(), &carried),
        &[KEY_VARIANT, "the family's own fixture"],
    );
}

/// A role that names a tensor outside the layers (`token_embd`) reaches that tensor in either scope,
/// beside the layers' tensors of the map's other roles; one already at the type is refused as a layer's is.
#[test]
fn a_role_outside_the_layers_reaches_its_tensor() {
    let d = Dir::new();
    let mut tensors = source_tensors();
    tensors.insert(
        0,
        t("token_embd.weight".into(), &[256, 4], GgmlType::Q3_K, 0, 13),
    );
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    let with = |scope: Scope, ty: GgmlType| Row {
        scope,
        map: vec![
            ("token_embd".to_string(), ty),
            ("ffn_gate_exps".to_string(), GgmlType::IQ3_S),
        ],
        ..the_row()
    };
    for scope in [Scope::Trunk, Scope::All] {
        let got = expand(&with(scope, GgmlType::Q5_K), &src).expect("an expansion");
        let first = got.first().expect("a retyped tensor");
        assert_eq!(
            (first.name.as_str(), first.from, first.to),
            ("token_embd.weight", GgmlType::Q3_K, GgmlType::Q5_K),
            "the tensor outside the layers leads the source's order"
        );
        assert!(
            got.iter()
                .skip(1)
                .all(|r| r.name.ends_with("ffn_gate_exps.weight")),
            "the layers' tensors of the other role follow"
        );
    }
    red(
        expand(&with(Scope::Trunk, GgmlType::Q3_K), &src),
        &["token_embd.weight", "already at q3_K"],
    );
    let arg = custom_q(
        "v1",
        &src,
        &expand(&with(Scope::Trunk, GgmlType::Q5_K), &src).expect("an expansion"),
    )
    .expect("an argument");
    assert!(
        arg.starts_with("^token_embd\\.weight$=q5_K,"),
        "the quantizer's argument retypes it by its exact name: {arg}"
    );
}

/// A tensor already at the map's type, beside another of its role that is not, is left out of the
/// expansion; the quantizer's argument keeps it at its own type.
#[test]
fn a_tensor_already_at_the_type_is_left_out() {
    let d = Dir::new();
    let mut tensors = source_tensors();
    let two = tn(2, "ffn_down_exps");
    let at = tensors
        .iter()
        .position(|t| t.name == two)
        .expect("layer 2's down");
    tensors[at] = t(two.clone(), &[256, 2, 3], GgmlType::IQ4_XS, 1, 8);
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    let got = expand(&the_row(), &src).expect("an expansion");
    let names: Vec<&str> = got.iter().map(|r| r.name.as_str()).collect();
    assert!(
        names.contains(&tn(1, "ffn_down_exps").as_str()),
        "layer 1's down is retyped"
    );
    assert!(
        !names.contains(&two.as_str()),
        "layer 2's down is already iq4_xs: {names:?}"
    );
    let arg = custom_q("v1", &src, &got).expect("an argument");
    assert!(
        arg.contains(&format!("^{}$=iq4_xs", two.replace('.', "\\."))),
        "layer 2's down at its own type in {arg}"
    );
}

/// The quantizer's argument names every tensor of the source once, in order, by its exact name: the map's at
/// their new type, the rest at their own.
#[test]
fn the_quantizer_argument_names_every_tensor() {
    let d = Dir::new();
    let src = open(&write(&d.0, "s", &source_kvs(), &source_tensors(), 2));
    let got = custom_q("v1", &src, &expand(&the_row(), &src).expect("an expansion"))
        .expect("an argument");
    let rules: Vec<&str> = got.split(',').collect();
    assert_eq!(rules.len(), 12);
    let rule = |layer: usize, role: &str, ty: &str| {
        format!("^{}$={ty}", tn(layer, role).replace('.', "\\."))
    };
    assert_eq!(rules[0], rule(0, "attn_q", "q8_0"));
    assert_eq!(rules[1], rule(0, "attn_norm", "f32"));
    assert_eq!(rules[2], rule(1, "ffn_gate_exps", "iq3_s"));
    assert_eq!(rules[4], rule(1, "ffn_down_exps", "iq4_xs"));
    assert_eq!(rules[8], rule(3, "nextn.eh_proj", "q8_0"));
    assert_eq!(rules[9], rule(3, "ffn_gate_exps", "q4_K"));
    assert_eq!(rules[11], rule(3, "ffn_down_exps", "q5_K"));
    assert_eq!(override_kv("v1"), "bloomery.fixture.variant=str:v1");
}

/// A variant that is its source under the map alone passes, and the check counts what it compared.
#[test]
fn a_variant_that_differs_by_the_map_alone_passes() {
    let d = Dir::new();
    let tensors = source_tensors();
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    let var = open(&write(
        &d.0,
        "v",
        &variant_kvs(),
        &variant_tensors(&tensors),
        2,
    ));
    let mut lines = Vec::new();
    let stats = check(&the_row(), &src, &var, card_budget(), &mut |l| {
        lines.push(l.to_string())
    })
    .expect("a variant");
    assert_eq!((stats.tensors, stats.retyped, stats.shards), (12, 6, 2));
    let stacks = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    let copied: usize = tensors
        .iter()
        .filter(|x| !(1..=2).any(|l| stacks.iter().any(|r| x.name == tn(l, r))))
        .map(|x| x.bytes.len())
        .sum();
    assert_eq!(stats.copied_bytes, copied as u64);
    assert_eq!(lines.len(), 12);
}

/// Every difference the map does not name is refused, naming the tensor or the key.
#[test]
fn a_variant_that_differs_otherwise_is_refused_by_name() {
    let tensors = source_tensors();
    let good = variant_tensors(&tensors);
    let d = Dir::new();
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    let mut n = 0;
    let mut run = |kvs: Vec<(String, Value)>, ts: Vec<T>, shards: usize, parts: &[&str]| {
        n += 1;
        let var = open(&write(&d.0, &format!("v{n}"), &kvs, &ts, shards));
        red(
            check(&the_row(), &src, &var, card_budget(), &mut |_| {}),
            parts,
        );
    };
    let mutate = |name: &str, f: &dyn Fn(&mut T)| {
        let mut ts = good.clone();
        f(ts.iter_mut().find(|x| x.name == name).expect("a tensor"));
        ts
    };
    let (attn_q, nextn_up) = (tn(0, "attn_q"), tn(3, "ffn_up_exps"));

    // One byte of an unlisted tensor.
    let ts = mutate(&attn_q, &|x| x.bytes[5] ^= 1);
    run(variant_kvs(), ts, 2, &[&attn_q, "not in the map", "byte 5"]);
    // The NextN layer's stack retyped, which the map does not reach.
    let ts = mutate(&nextn_up, &|x| {
        x.ty = GgmlType::IQ3_S;
        x.bytes = bytes(1, size_of(GgmlType::IQ3_S, &x.dims));
    });
    run(
        variant_kvs(),
        ts,
        2,
        &[&nextn_up, "type iq3_s, want q4_K", "does not name it"],
    );
    // A listed tensor left at the source's type, and one given another type than the map's.
    let down2 = tn(2, "ffn_down_exps");
    let ts = mutate(&down2, &|x| {
        x.ty = GgmlType::Q5_K;
        x.bytes = bytes(1, size_of(GgmlType::Q5_K, &x.dims));
    });
    run(variant_kvs(), ts, 2, &[&down2, "type q5_K, want iq4_xs"]);
    let gate1 = tn(1, "ffn_gate_exps");
    let ts = mutate(&gate1, &|x| {
        x.ty = GgmlType::IQ4_XS;
        x.bytes = bytes(1, size_of(GgmlType::IQ4_XS, &x.dims));
    });
    run(variant_kvs(), ts, 2, &[&gate1, "type iq4_xs, want iq3_s"]);
    // A shape.
    let ts = mutate(&attn_q, &|x| {
        x.dims = vec![64, 2];
    });
    run(variant_kvs(), ts, 2, &[&attn_q, "shape [64, 2]"]);
    // A tensor missing, a tensor added.
    let mut ts = good.clone();
    let norm = tn(0, "attn_norm");
    ts.retain(|x| x.name != norm);
    run(variant_kvs(), ts, 2, &["11 tensors, the source's 12"]);
    let mut ts = good.clone();
    let extra = tn(9, "extra");
    ts.push(t(extra.clone(), &[8], GgmlType::F32, 1, 1));
    run(variant_kvs(), ts, 2, &["13 tensors", &extra]);
    // A tensor in the other shard, and another shard count.
    let ts = mutate(&attn_q, &|x| x.shard = 1);
    run(
        variant_kvs(),
        ts,
        2,
        &[&attn_q, "shard 1, the source's is 0"],
    );
    run(
        variant_kvs(),
        good.iter()
            .cloned()
            .map(|mut x| {
                x.shard = 0;
                x
            })
            .collect(),
        1,
        &["1 shards, the source's 2"],
    );
    // The header: the tag missing or another, a kept key dropped or changed, a key added.
    let mut kvs = variant_kvs();
    kvs.pop();
    run(kvs, good.clone(), 2, &[KEY_VARIANT, "missing"]);
    let mut kvs = variant_kvs();
    let tag = kvs.len() - 1;
    kvs[tag].1 = Value::String("v2".into());
    run(kvs, good.clone(), 2, &[KEY_VARIANT, "\"v2\""]);
    let mut kvs = variant_kvs();
    kvs.retain(|(k, _)| k != "bloomery.fixture.card_budget");
    run(
        kvs,
        good.clone(),
        2,
        &["bloomery.fixture.card_budget", "missing"],
    );
    let mut kvs = variant_kvs();
    set(
        &mut kvs,
        "tokenizer.ggml.model",
        Value::String("llama".into()),
    );
    run(kvs, good.clone(), 2, &["tokenizer.ggml.model", "\"llama\""]);
    let mut kvs = variant_kvs();
    kvs.push(("bloomery.fixture.seed".into(), Value::U64(2)));
    run(
        kvs,
        good.clone(),
        2,
        &["bloomery.fixture.seed", "not in the source"],
    );
}

/// A variant's card budget is its own file's plan, not its trunk's copied: the check holds a
/// planned family's variant to the planner's value for the written file — the trunk's is another
/// number — and a fixed family's to the trunk's. Mutants: the budget clause dropped (the trunk's
/// passes); the key held equal to the source's again (the variant's own refuses).
#[test]
fn a_variants_card_budget_is_its_own_not_its_trunks() {
    let d = Dir::new();
    let tensors = source_tensors();
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    // The planner's value for this file.
    let var = open(&write(
        &d.0,
        "v",
        &variant_kvs(),
        &variant_tensors(&tensors),
        2,
    ));
    check(&the_row(), &src, &var, card_budget(), &mut |_| {}).expect("the variant's own budget");
    // The trunk's, which the quantizer copies: refused by name, both numbers in the message.
    let var = open(&write(
        &d.0,
        "w",
        &variant_kvs_at(TRUNK_BUDGET),
        &variant_tensors(&tensors),
        2,
    ));
    red(
        check(&the_row(), &src, &var, card_budget(), &mut |_| {}),
        &[
            KEY_CARD_BUDGET,
            "U64(16)",
            "the planner's 8",
            "its own file",
        ],
    );
    // A missing key.
    let mut kvs = variant_kvs();
    kvs.retain(|(k, _)| k != KEY_CARD_BUDGET);
    let var = open(&write(&d.0, "x", &kvs, &variant_tensors(&tensors), 2));
    red(
        check(&the_row(), &src, &var, card_budget(), &mut |_| {}),
        &[KEY_CARD_BUDGET, "missing", "the planner's 8"],
    );
    // A fixed family keeps the trunk's value: it passes, and the variant's own is a changed key.
    let var = open(&write(
        &d.0,
        "y",
        &variant_kvs_at(TRUNK_BUDGET),
        &variant_tensors(&tensors),
        2,
    ));
    check(
        &the_row(),
        &src,
        &var,
        CardBudget::Fixed(1 << 20),
        &mut |_| {},
    )
    .expect("the trunk's value");
    let var = open(&write(
        &d.0,
        "z",
        &variant_kvs(),
        &variant_tensors(&tensors),
        2,
    ));
    red(
        check(
            &the_row(),
            &src,
            &var,
            CardBudget::Fixed(1 << 20),
            &mut |_| {},
        ),
        &["the header:", KEY_CARD_BUDGET, "the source's is U64(16)"],
    );
}

/// The written set's shard `i` of `n` named `stem`.
fn shard(dir: &Path, stem: &str, i: usize, n: usize) -> PathBuf {
    dir.join(format!("{stem}-{:05}-of-{:05}.gguf", i + 1, n))
}

/// `record_budget` replaces the trunk's budget the quantizer copied with the variant's own: the
/// bytes that move stay inside the value's eight, on every shard that holds the key, and the
/// reopened set reads the new value. Mutants: the offset off by one (a byte outside the window
/// moves, and the reopened value is wrong); the value written big-endian (the reopened value is
/// wrong); the type check dropped (a shard holding the key as a string is patched instead of
/// refused).
#[test]
fn record_budget_rewrites_only_the_value() {
    let d = Dir::new();
    let tensors = source_tensors();
    let src = open(&write(&d.0, "s", &source_kvs(), &tensors, 2));
    drop(src);
    let first = write(
        &d.0,
        "v",
        &variant_kvs_at(TRUNK_BUDGET),
        &variant_tensors(&tensors),
        2,
    );
    let before: Vec<Vec<u8>> = (0..2)
        .map(|i| std::fs::read(shard(&d.0, "v", i, 2)).expect("the shard before"))
        .collect();
    // A value that differs from the trunk's 16 at both ends of the u64, so a write at a wrong
    // offset moves a byte the window does not hold.
    const TO: u64 = 0x0100_0000_0000_0008;
    let var = open(&first);
    let was = record_budget("v1", &var, TO).expect("the trunk's budget");
    assert_eq!(was, TRUNK_BUDGET, "the quantizer copied the trunk's");
    drop(var);
    for (i, before) in before.iter().enumerate() {
        let after = std::fs::read(shard(&d.0, "v", i, 2)).expect("the shard after");
        let at: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(at, _)| at)
            .collect();
        assert!(!at.is_empty(), "the value's bytes move");
        assert!(
            at.len() <= 8,
            "no more than the value's eight bytes move: {at:?}"
        );
        let lo = at[0];
        assert!(at.iter().all(|&x| x < lo + 8), "inside the value: {at:?}");
    }
    assert_eq!(open(&first).value(KEY_CARD_BUDGET), Some(&Value::U64(TO)));
    // A shard holding the key as another type: refused by name, nothing written.
    let mut kvs = variant_kvs_at(TRUNK_BUDGET);
    set(&mut kvs, KEY_CARD_BUDGET, Value::String("sixteen".into()));
    let strung = write(&d.0, "u", &kvs, &variant_tensors(&tensors), 1);
    red(
        record_budget("v1", &open(&strung), TO),
        &[KEY_CARD_BUDGET, "not a u64"],
    );
    assert_eq!(
        open(&strung).value(KEY_CARD_BUDGET),
        Some(&Value::String("sixteen".into())),
        "the refusal writes nothing"
    );
}
