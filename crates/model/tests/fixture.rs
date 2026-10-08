//! The gate fixture generator (`model::fixture`, bin `fixture`) on the V4.1 spec
//! (`model::arch::deepseek41::fixture`), against the real file's header (`gguf::v41::model`) and
//! the real DSpark draft's (`$BLOOMERY_DSPARK_MODEL`). Five contracts:
//!
//! 1. the plan: the nine layers are the map's, each tensor its source's name,
//!    dims and type, the metadata overrides exactly the list, no nested
//!    array, the layout's total the sum of its parts, and the engine reads
//!    the header-only files as the source's layer kinds and draft;
//! 2. determinism: the same seed gives the same bytes on one thread or all,
//!    another seed other bytes;
//! 3. scales: every sampled block of every tensor holds its rule (`d`/`dmin`
//!    a normal f16 in [2^-13, 2^-10], scales in band, dequant = the rule's
//!    formula) and its dequantized RMS is within ±10 % of 1/√K;
//! 4. end to end: the binary writes small subset files holding every type at
//!    its real shape, `Split::open` and `verify` pass, a flipped byte and an
//!    existing directory are refused by name; each written file's sha256 is
//!    printed, the record that today's generator writes today's bytes;
//! 5. refusals: a draft naming more target layers than the fixture has, a
//!    `general.alignment` the writer would not take, an ff override the V4.1
//!    spec has no rule for, a flag its verb does not take, and a file whose
//!    architecture has no spec are refused by name;
//! 6. a second family: a synthetic family plugs in through `FixtureSpec`
//!    alone — its key rules, a table that resizes a global, an ff override, a
//!    layer-id list — and its fixture is planned, written and verified; the
//!    spec's ratios one short, a layer list the map does not hold, a draft for
//!    a family with none and another architecture are refused by name;
//! 7. the Q5_1 and IQ4_NL rules under the rule window, [2^-13, 2^-10]: a filled chunk
//!    dequantizes (`gguf::dequant_row`) to an RMS within ±10 % of 1/√K and
//!    every block holds its rule; a K whose `d` no code choice puts in the
//!    window is refused with `NoScale`;
//! 8. the window is the spec's: a wider one admits Q5_1 at K = 640, which
//!    V4.1's refuses, and a window that leaves the normal f16 values is
//!    refused by name.
//!
//! The Qwen3.8 spec (`model::arch::qwen35moe::fixture`) against the real file's header and the
//! real shared MTP draft's, header-only files only (each file a hole beyond its header):
//!
//! 9. the plan: four layers from source layers 2, 3, 1 and 47, the interval written 2, and the
//!    engine's own hparams reader reading the header-only files back as four layers of kinds
//!    [GDN, QSA, GDN, QSA], pools [0, 4, 0, 4], the PLE site at layer 2, and the draft as a draft
//!    of the target;
//! 10. the card budget: the plan of the written file under its recorded budget holds half of the
//!     experts (256) on every layer `card_routed` admits and none on the others, its budgetless
//!     plan all of them; a budget one step either side of the recorded one is refused by name, and
//!     `verify` refuses a header whose recorded budget is not the plan's;
//! 11. the silent traps of ik's reader: a per-layer array one short, a missing `ple.layers`, and
//!     a missing `full_attention_interval` are refused by name, in the source and in the written
//!     file, as is a draft's ratios array one short of its layers.
//!
//! The r8 sidecar and the V4.1 spec's ff 512 (the plan's draft and routed stacks at ff 512 are
//! contract 1's; the subset files of contract 4 are given their budget, a part of a plan having
//! none to choose from):
//!
//! 12. the sidecar of a toy family is written beside the fixture, checked by `verify`, and refused
//!     when absent, of another stack set or one flipped byte; a family naming no stack leaves no
//!     fixture behind;
//! 13. V4.1's budget: the plan of the written file at the gate machine and `CTX_MAX` holds half
//!     the experts (192 of 384) on every card-eligible layer, a budget 64 MiB either side is
//!     refused, `verify` refuses an edited record, and the plan's conditions are printed;
//! 14. V4.1's two router biases differ between experts (a constant 0 and a uniform draw, a stream
//!     a layer), and the family's header check refuses zeros, equal biases and a constant shift;
//! 15. V4.1's r8 stacks are the gate and up stacks of the nine layers, Q3_K at the source's type
//!     and width, at ff 512 on r8file's grid.
//!
//! The GLM-5.3-Flash spec (`model::arch::glm5next::fixture`) against the real file's header:
//!
//! 16. the plan: seven layers from source layers 0, 1, 4, 7, 8, 11 and 45, read back by the
//!     engine's hparams as kinds [KDA, KDA, KDA, latent, KDA, latent, latent], a dense prefix of
//!     two, the NextN layer believed, every (role, type) pair of the routed stacks, the routed ff
//!     512 and the shared expert's own width untouched;
//! 17. the budget: the plan of the written file with its NextN layer holds 144 of 288 experts on
//!     each card-eligible layer and none on the dense and the Q6_K-down layers; a budget 64 MiB
//!     either side and an edited record are refused, and the plan's conditions are printed;
//! 18. what the reader and the kernels' constants take silently, refused by name: a map that
//!     drops the NextN layer or puts a dense layer after a routed one, a changed constant in the
//!     source or in the written file, a fixture ff and dense count off the header's, an ff off the
//!     block grid.
//!
//! Files go under this crate's `CARGO_TARGET_TMPDIR` and are removed.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::fixture::{self as v41, LAYER_MAP};
use model::arch::deepseek41::place::PlanInputs as V41Inputs;
use model::arch::glm5next::fixture as glm;
use model::arch::glm5next::hparams::Kind as GlmKind;
use model::arch::glm5next::place::{NextnInputs, PlanInputs as GlmInputs};
use model::arch::qwen35moe::fixture as q38;
use model::arch::qwen35moe::hparams::Kind;
use model::arch::qwen35moe::place::{Experts, PlanInputs, UBATCH_PLANNED, machine_for_experts};
use model::fileio;
use model::fixture::{
    self, CHUNK_TARGET, CardBudget, DEFAULT_SHARD_BYTES, Family, FilePlan, FixtureError,
    FixtureSpec, KEY_CARD_BUDGET, KEY_SEED, KEY_SOURCE_LAYERS, KEY_SOURCE_SHA256, KEY_VERSION,
    KeyRule, Options, Plan, PlannedTensor, Rule, Sample, SidecarSpec, Tables, Window, rule_for,
};
use model::placement::PlanLevers;
use model::placement::workstation::{self, RTX_3090};
use sha2::{Digest, Sha256};

const ARCH: &str = "deepseek41";

fn source() -> Split {
    let path = gguf::v41::model();
    Split::open(&path).unwrap_or_else(|e| panic!("open the V4.1 file {path}: {e}"))
}

fn draft_path() -> String {
    std::env::var("BLOOMERY_DSPARK_MODEL")
        .expect("BLOOMERY_DSPARK_MODEL unset: run through `just gate-fixture`")
}

fn draft() -> Split {
    let path = draft_path();
    Split::open(&path).unwrap_or_else(|e| panic!("open the DSpark draft {path}: {e}"))
}

fn full_plan(src: &Split, d: &Split) -> Plan {
    let spec = v41::spec();
    fixture::plan(&spec, src, Some(d), &spec.options()).expect("the plan of the real file")
}

/// A fresh directory under the target's tmp dir, removed when dropped —
/// also when the clause panics, so a red run leaves no files behind.
struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("could not remove {}: {e}", self.0.display());
        }
    }
}

fn dir(tag: &str) -> Dir {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("fixture-{}-{tag}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    Dir(d)
}

/// Every file of `p` as its header and a hole up to its length: a file the
/// strict reader opens, holding no tensor bytes.
fn header_only(p: &FilePlan, d: &Path) -> PathBuf {
    for (name, layout) in p.layouts().unwrap() {
        let path = d.join(&name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = File::create(&path).unwrap();
        let len = layout.file_len();
        drop(Writer::new(&file, layout).unwrap());
        file.set_len(len).unwrap();
    }
    d.join(&p.files[0])
}

/// `bytes` over tensor `name` of the fixture whose first shard is `first`
/// (a header-only file's hole, or a written one's tensor).
fn put(first: &Path, name: &str, bytes: &[u8]) {
    use std::os::unix::fs::FileExt;
    let fx = Split::open(first).unwrap();
    let (s, info) = fx.find(name).unwrap_or_else(|| panic!("no {name}"));
    assert_eq!(bytes.len() as u64, info.nbytes, "{name}");
    let at = fx.shard(s).unwrap().data_base() + info.offset;
    let path = fx.shard_path(s).unwrap().to_path_buf();
    drop(fx);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .write_all_at(bytes, at)
        .unwrap();
}

/// The f32 values of little-endian `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The little-endian bytes of `v`.
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The bytes the generator writes for `t` under `seed`.
fn filled(t: &PlannedTensor, seed: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; t.nbytes as usize];
    t.fill(seed, 1, &mut bytes);
    bytes
}

/// A V4.1 fixture's router biases as the generator writes them, over a
/// header-only file's holes: the file's data is zeros, which the family's
/// bias check refuses, so a clause that reads a header-only fixture as a
/// V4.1 file writes them first.
fn write_biases(p: &FilePlan, first: &Path) {
    for t in p
        .tensors
        .iter()
        .filter(|t| leaf_of(&t.name).starts_with("exp_probs_b"))
    {
        put(first, &t.name, &filled(t, fixture::DEFAULT_SEED));
    }
}

/// A tensor's name past `blk.N.`; empty for a global.
fn leaf_of(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or("", |(_, rest)| rest)
}

/// `dims` of the V4.1 or DSpark tensor `leaf` at the fixture's ff, written
/// here by hand (not through the family's `ff_axis`): the routed stacks and the
/// shared expert carry the ff in the second dim of gate and up and the first of
/// down.
fn ff_dims(leaf: &str, dims: &[u64]) -> Vec<u64> {
    let mut d = dims.to_vec();
    match leaf {
        "ffn_gate_exps.weight"
        | "ffn_up_exps.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight" => d[1] = v41::FIXTURE_FF,
        "ffn_down_exps.weight" | "ffn_down_shexp.weight" => d[0] = v41::FIXTURE_FF,
        _ => {}
    }
    d
}

fn nested(v: &Value) -> bool {
    matches!(v, Value::Array(a) if a.iter().any(|x| matches!(x, Value::Array(_))))
}

fn unsigned_items(v: &Value) -> Vec<u64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_unsigned().unwrap()).collect(),
        other => panic!("{other:?} is not an array"),
    }
}

/// Contract 1: the plan.
#[test]
#[ignore = "needs the box, the V4.1 file and $BLOOMERY_DSPARK_MODEL (just gate-fixture)"]
fn hw_fixture_plan() {
    let src = source();
    let d = draft();
    let p = full_plan(&src, &d);
    let t = &p.target;

    // Every fixture layer is its source layer, tensor for tensor.
    let mut src_layers: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut src_globals = Vec::new();
    for (_, info) in src.iter_tensors() {
        match info
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
        {
            Some((l, rest)) => src_layers
                .entry(l.parse().unwrap())
                .or_default()
                .push(rest.to_string()),
            None => src_globals.push(info.name.clone()),
        }
    }
    let site_rows: Vec<u64> = v41::engram_rows(&src)
        .unwrap()
        .iter()
        .map(|&(fx, _)| fx)
        .collect();
    for (f, &l) in LAYER_MAP.iter().enumerate() {
        let ours: Vec<&PlannedTensor> = t.tensors.iter().filter(|x| x.layer == Some(f)).collect();
        let theirs = &src_layers[&l];
        assert_eq!(
            ours.len(),
            theirs.len(),
            "layer {f} holds {} tensors, source layer {l} {}",
            ours.len(),
            theirs.len()
        );
        for (x, rest) in ours.iter().zip(theirs) {
            assert_eq!(x.name, format!("blk.{f}.{rest}"));
            assert_eq!(x.source, format!("blk.{l}.{rest}"));
            let (_, s) = src.find(&x.source).unwrap();
            assert_eq!(x.ty, s.ty, "{}", x.name);
            if rest == "engram_embd.weight" {
                let site = [1usize, 4]
                    .iter()
                    .position(|&e| e == f)
                    .expect("an engram layer");
                assert_eq!(
                    x.dims,
                    vec![s.dims[0], site_rows[site]],
                    "{}: rows follow the fixture primes",
                    x.name
                );
            } else {
                assert_eq!(x.dims, ff_dims(rest, &s.dims), "{}", x.name);
            }
        }
    }
    let globals: Vec<&str> = t
        .tensors
        .iter()
        .filter(|x| x.layer.is_none())
        .map(|x| x.name.as_str())
        .collect();
    assert_eq!(
        globals,
        src_globals.iter().map(String::as_str).collect::<Vec<_>>()
    );
    println!("plan: 9 layers {LAYER_MAP:?} tensor for tensor, globals {globals:?}");

    // The metadata overrides are exactly the list.
    let theirs: HashMap<&str, &Value> = src.iter_kv().collect();
    let ours: Vec<(&str, &Value)> = t.kvs.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let ours_map: HashMap<&str, &Value> = ours.iter().copied().collect();
    let mut changed: Vec<&str> = ours
        .iter()
        .filter(|(k, v)| theirs.get(k).is_some_and(|s| s != v))
        .map(|(k, _)| *k)
        .collect();
    changed.sort_unstable();
    let key = |s: &str| format!("{ARCH}.{s}");
    let mut want_changed = vec![
        key("block_count"),
        key("expert_feed_forward_length"),
        key("attention.compress_ratios"),
        key("swiglu_clamp_exp"),
        key("swiglu_clamp_shexp"),
        key("engram.layer_ids"),
        key("engram.primes"),
        key("engram.offsets"),
        "split.count".to_string(),
        "split.tensors.count".to_string(),
    ];
    want_changed.sort_unstable();
    assert_eq!(changed, want_changed, "the overridden keys");
    let removed: Vec<&&str> = theirs
        .keys()
        .filter(|k| !ours_map.contains_key(*k))
        .collect();
    assert!(removed.is_empty(), "keys dropped: {removed:?}");
    let added: Vec<&str> = ours
        .iter()
        .filter(|(k, _)| !theirs.contains_key(k))
        .map(|(k, _)| *k)
        .collect();
    assert_eq!(
        added,
        [
            KEY_VERSION,
            KEY_SEED,
            KEY_SOURCE_LAYERS,
            KEY_SOURCE_SHA256,
            KEY_CARD_BUDGET
        ]
    );
    let get = |k: &str| ours_map[k];
    assert_eq!(get(&key("block_count")).as_u64(), Some(9));
    let ratios = unsigned_items(theirs[key("attention.compress_ratios").as_str()]);
    let n_layer = 40;
    let want_ratios: Vec<u64> = [0, 0, 2, 2, 2, 1, 1, 1, 1]
        .into_iter()
        .chain(ratios[n_layer..].iter().copied())
        .collect();
    assert_eq!(
        unsigned_items(get(&key("attention.compress_ratios"))),
        want_ratios
    );
    for k in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
        let Value::Array(s) = theirs[key(k).as_str()] else {
            panic!("{k}")
        };
        let want: Vec<Value> = LAYER_MAP.iter().map(|&l| s[l].clone()).collect();
        assert_eq!(
            get(&key(k)),
            &Value::Array(want),
            "{k}: the source layers' values, nine"
        );
    }
    assert_eq!(unsigned_items(get(&key("engram.layer_ids"))), [1, 4]);
    let primes = unsigned_items(get(&key("engram.primes")));
    let offsets = unsigned_items(get(&key("engram.offsets")));
    assert_eq!(primes.len(), 48);
    let is_prime = |n: u64| {
        (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
    };
    let all: Vec<u64> = (v41::ENGRAM_PRIME_FLOOR + 1..)
        .filter(|&n| is_prime(n))
        .take(48)
        .collect();
    assert_eq!(primes, all, "the 48 smallest primes above 2^14");
    for (site, (ps, os)) in primes.chunks(24).zip(offsets.chunks(24)).enumerate() {
        let sums: Vec<u64> = ps
            .iter()
            .scan(0, |a, &p| {
                let o = *a;
                *a += p;
                Some(o)
            })
            .collect();
        assert_eq!(
            os,
            sums.as_slice(),
            "site {site}'s offsets are its exclusive prefix sums"
        );
        assert_eq!(ps.iter().sum::<u64>(), site_rows[site]);
    }
    assert_eq!(get(KEY_VERSION).as_u64(), Some(1));
    assert_eq!(
        unsigned_items(get(KEY_SOURCE_LAYERS)),
        LAYER_MAP.map(|l| l as u64)
    );
    assert_eq!(
        get(KEY_SOURCE_SHA256).as_str(),
        Some(fixture::header_sha256(&src).as_str())
    );
    assert_eq!(get(KEY_CARD_BUDGET).as_u64(), Some(p.card_budget));
    assert_eq!(get(&key("expert_feed_forward_length")).as_u64(), Some(512));
    let dp = p.draft.as_ref().unwrap();
    let dtheirs: HashMap<&str, &Value> = d.iter_kv().collect();
    let dchanged: Vec<&str> = dp
        .kvs
        .iter()
        .filter(|(k, v)| dtheirs.get(k.as_str()).is_some_and(|s| *s != v))
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(
        dchanged,
        ["dflash.expert_feed_forward_length", "dflash.target_layers"],
        "the draft's overridden keys, in file order"
    );
    let dff = dp
        .kvs
        .iter()
        .find(|(k, _)| k == "dflash.expert_feed_forward_length")
        .unwrap();
    assert_eq!(
        dff.1.as_u64(),
        Some(512),
        "the draft's ff moves with the target's"
    );
    for t in &dp.tensors {
        let (_, s) = d.find(&t.source).unwrap();
        assert_eq!(
            t.dims,
            ff_dims(leaf_of(&t.name), &s.dims),
            "{}: the draft's ff axes at 512",
            t.name
        );
    }
    assert_eq!(
        unsigned_items(
            &dp.kvs
                .iter()
                .find(|(k, _)| k == "dflash.target_layers")
                .unwrap()
                .1
        ),
        [6, 7, 8]
    );
    for (k, v) in t.kvs.iter().chain(&dp.kvs) {
        assert!(!nested(v), "{k} is a nested array");
    }
    println!("plan: overrides {changed:?}, added {added:?}, draft {dchanged:?}; no nested array");

    // The layout's total is the sum of its parts.
    let layouts = t.layouts().unwrap();
    let total: u64 = layouts.iter().map(|(_, l)| l.file_len()).sum();
    let headers: u64 = layouts.iter().map(|(_, l)| l.data_base()).sum();
    let layers: u64 = t
        .tensors
        .iter()
        .filter(|x| x.layer.is_some())
        .map(|x| t.padded(x))
        .sum();
    let glob: u64 = t
        .tensors
        .iter()
        .filter(|x| x.layer.is_none())
        .map(|x| t.padded(x))
        .sum();
    assert_eq!(total, headers + layers + glob);
    let spans = t.spanning_layers();
    assert!(!spans.is_empty(), "a layer spans a shard boundary");
    let draft_len = dp.layouts().unwrap()[0].1.file_len();
    println!(
        "plan: total {total} B = headers {headers} + layers {layers} + globals {glob} ({:.2} GB; 14.76 GB [derived from the header dump]); draft {draft_len} B ({:.2} GB; 2.27 GB [derived]); {} shards, layers {spans:?} span",
        total as f64 / 1e9,
        draft_len as f64 / 1e9,
        layouts.len()
    );

    // The engine reads the header-only files as the source's kinds.
    let guard = dir("plan");
    let dd = &guard.0;
    let first = header_only(t, dd);
    write_biases(t, &first);
    let draft_file = header_only(dp, dd);
    let fx = Split::open(&first).unwrap();
    let spec = v41::spec();
    let hp = v41::check_kinds(&spec, &fx, &src).unwrap();
    let dx = Split::open(&draft_file).unwrap();
    let dhp = v41::check_draft(&spec, &dx, &fx, true).unwrap();
    let real_bytes: u64 = std::fs::read_dir(dd)
        .unwrap()
        .map(|e| {
            use std::os::unix::fs::MetadataExt;
            e.unwrap().metadata().unwrap().blocks() * 512
        })
        .sum();
    println!(
        "plan: header-only files ({real_bytes} real bytes) read as {} layers of the source kinds, draft target_layers {:?}",
        hp.n_layer, dhp.target_layers
    );
}

/// A sample of each tensor: its sampled chunks, one after another.
fn sample(t: &PlannedTensor, seed: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for c in fixture::sample_chunks(t) {
        let r = t.chunk_range(c);
        let at = out.len();
        out.resize(at + r.len(), 0);
        t.fill_chunk(seed, c, &mut out[at..]);
    }
    out
}

fn sha(bytes: &[u8]) -> String {
    fileio::hex(&Sha256::digest(bytes)[..8])
}

/// One `e2e: sha256` line per `.gguf` file under `dir`, by path below it.
fn print_file_shas(dir: &Path) {
    let mut files = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                todo.push(p);
            } else if p.extension().is_some_and(|x| x == "gguf") {
                files.push(p);
            }
        }
    }
    files.sort();
    for f in files {
        let hex = fileio::sha256_hex(&std::fs::read(&f).unwrap());
        println!(
            "e2e: sha256 {hex}  {}",
            f.strip_prefix(dir).unwrap().display()
        );
    }
}

/// Contract 2: determinism.
#[test]
#[ignore = "needs the box, the V4.1 file and $BLOOMERY_DSPARK_MODEL (just gate-fixture)"]
fn hw_fixture_determinism() {
    let src = source();
    let d = draft();
    let p = full_plan(&src, &d);
    let all: Vec<&PlannedTensor> = p
        .target
        .tensors
        .iter()
        .chain(&p.draft.as_ref().unwrap().tensors)
        .collect();
    let mut picked: Vec<&PlannedTensor> = Vec::new();
    for t in &all {
        let kind = |x: &PlannedTensor| {
            (
                x.ty.as_u32(),
                matches!(x.rule, Rule::Const { .. }),
                x.dims.len() >= 3,
            )
        };
        if !picked.iter().any(|x| kind(x) == kind(t)) {
            picked.push(t);
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    for t in &picked {
        let a = sample(t, 7);
        let b = sample(t, 7);
        let c = sample(t, 8);
        assert!(a == b, "{}: the same seed, other bytes", t.name);
        let is_const = matches!(t.rule, Rule::Const { .. });
        assert_eq!(
            a == c,
            is_const,
            "{}: another seed must move every random tensor",
            t.name
        );
        if t.nbytes <= 64 << 20 {
            let mut serial = vec![0u8; t.nbytes as usize];
            t.fill(7, 1, &mut serial);
            let mut parallel = vec![0u8; t.nbytes as usize];
            t.fill(7, threads, &mut parallel);
            let at = serial.iter().zip(&parallel).position(|(x, y)| x != y);
            assert!(
                at.is_none(),
                "{}: 1 thread and {threads} write other bytes, first at byte {at:?}",
                t.name
            );
        }
        println!(
            "determinism: {} {} {:?} seed7 {} seed7 {} seed8 {}",
            t.name,
            t.ty,
            t.dims,
            sha(&a),
            sha(&b),
            sha(&c)
        );
    }
    println!(
        "determinism: {} tensors, one per (type, constant, stack)",
        picked.len()
    );
}

/// Contract 3: scales.
#[test]
#[ignore = "needs the box, the V4.1 file and $BLOOMERY_DSPARK_MODEL (just gate-fixture)"]
fn hw_fixture_scales() {
    let src = source();
    let d = draft();
    let p = full_plan(&src, &d);
    let window = v41::spec().window;
    let all: Vec<&PlannedTensor> = p
        .target
        .tensors
        .iter()
        .chain(&p.draft.as_ref().unwrap().tensors)
        .collect();
    let mut by_type: BTreeMap<String, (usize, usize, f64, f64)> = BTreeMap::new();
    let t0 = Instant::now();
    let results: Vec<(String, Sample, Option<f64>)> = std::thread::scope(|s| {
        let handles: Vec<_> = all
            .iter()
            .map(|t| {
                s.spawn(move || {
                    let mut total = Sample::default();
                    for c in fixture::sample_chunks(t) {
                        let r = t.chunk_range(c);
                        let mut buf = vec![0u8; r.len()];
                        t.fill_chunk(7, c, &mut buf);
                        let unit = t.ty.type_size().unwrap() as usize;
                        let got = fixture::check_units(t, &buf, r.start / unit, window)
                            .unwrap_or_else(|e| panic!("{e}"));
                        total.blocks += got.blocks;
                        total.values += got.values;
                        total.sum_sq += got.sum_sq;
                    }
                    fixture::rms_within(t, &total).unwrap_or_else(|e| panic!("{e}"));
                    (t.ty.to_string(), total, t.sigma())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (ty, s, sigma) in results {
        let e = by_type.entry(ty).or_insert((0, 0, f64::MAX, f64::MIN));
        e.0 += 1;
        e.1 += s.blocks;
        if let Some(sg) = sigma {
            e.2 = e.2.min(s.rms() / sg);
            e.3 = e.3.max(s.rms() / sg);
        }
    }
    for t in &all {
        let ds: Vec<u16> = match &t.rule {
            Rule::Q3K { d, .. } | Rule::Q6K { d, .. } | Rule::Q8_0 { d, .. } => vec![*d],
            Rule::Q4K { d, dmin, .. } | Rule::Q5K { d, dmin, .. } => vec![*d, *dmin],
            _ => Vec::new(),
        };
        assert!(
            ds.iter().all(|&b| window.holds(b)),
            "{}: {}",
            t.name,
            t.rule.describe()
        );
    }
    for (ty, (n, blocks, lo, hi)) in &by_type {
        if *lo <= *hi {
            println!(
                "scales: {ty}: {n} tensors, {blocks} blocks checked, RMS / (1/sqrt K) in [{lo:.4}, {hi:.4}]"
            );
        } else {
            println!("scales: {ty}: {n} tensors, {blocks} blocks checked (constants)");
        }
    }
    let types: Vec<&str> = by_type.keys().map(String::as_str).collect();
    for want in [
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
        GgmlType::MXFP4,
        GgmlType::BF16,
        GgmlType::F32,
    ] {
        assert!(
            types.contains(&want.to_string().as_str()),
            "no {want} tensor was checked"
        );
    }
    println!(
        "scales: {} tensors in {:.1} s",
        all.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// The binary on `args`: status, stdout, stderr, echoed.
fn run(args: &[&str]) -> (Option<i32>, String, String) {
    let t = Instant::now();
    let o = Command::new(env!("CARGO_BIN_EXE_fixture"))
        .args(args)
        .output()
        .unwrap();
    let (out, err) = (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    );
    let tail: Vec<&str> = out.lines().filter(|l| !l.contains(" type=")).collect();
    println!(
        "$ fixture {}\n{}\n{err}rc={:?} wall={:.2}s",
        args.join(" "),
        tail.join("\n"),
        o.status.code(),
        t.elapsed().as_secs_f64()
    );
    (o.status.code(), out, err)
}

/// Peak RSS of the largest waited-for child, KiB.
fn children_peak_kib() -> i64 {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `ru` is storage for one `rusage`, which the call fills on success.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, ru.as_mut_ptr()) };
    assert_eq!(rc, 0, "getrusage(RUSAGE_CHILDREN)");
    // SAFETY: the call returned 0, so it wrote every field of `ru`.
    unsafe { ru.assume_init() }.ru_maxrss
}

fn flip(path: &Path, at: u64) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut b = [0u8; 1];
    f.seek(SeekFrom::Start(at)).unwrap();
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    f.write_all(&[b[0] ^ 0x10]).unwrap();
}

/// The card budget a subset file records: its family plans none from a part of
/// a plan.
const SUBSET_BUDGET: u64 = 1 << 30;

/// Contract 4: end to end.
#[test]
#[ignore = "needs the box, the V4.1 file and $BLOOMERY_DSPARK_MODEL (just gate-fixture)"]
fn hw_fixture_end_to_end() {
    let src_path = gguf::v41::model();
    let src = source();
    let draft_src = draft_path();
    let guard = dir("e2e");
    let d = &guard.0;

    // A target subset holding every target type at its real shape; the cap
    // puts layer 0 across the shard boundary.
    let subset = [
        "output.weight",
        "blk.0.attn_kv.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_sinks.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.hc_attn_scale.weight",
        "blk.1.engram_embd.weight",
        "blk.2.ffn_down_shexp.weight",
    ];
    let spec = v41::spec();
    // A subset file holds no whole plan to choose the budget from: it is given.
    let opts = Options {
        tensors: Some(subset.map(String::from).to_vec()),
        shard_bytes: u64::MAX,
        card_budget: Some(SUBSET_BUDGET),
        ..spec.options()
    };
    let p = fixture::plan(&spec, &src, None, &opts).unwrap();
    let cap = p.target.padded(&p.target.tensors[0]) + p.target.padded(&p.target.tensors[1]);
    let types: Vec<String> = p
        .target
        .tensors
        .iter()
        .map(|t| format!("{} {} {:?}", t.name, t.ty, t.dims))
        .collect();
    let bytes: u64 = p.target.tensors.iter().map(|t| t.nbytes).sum();
    println!(
        "e2e: target subset, {bytes} B at real shapes:\n  {}",
        types.join("\n  ")
    );
    assert!(bytes < 1 << 30);
    let out = d.join("target");
    let out_s = out.to_str().unwrap();
    let list = subset.join(",");
    let cap_s = cap.to_string();
    let budget_s = SUBSET_BUDGET.to_string();
    let t0 = Instant::now();
    let (rc, stdout, _) = run(&[
        "generate",
        &src_path,
        out_s,
        "--seed",
        "7",
        "--tensors",
        &list,
        "--shard-bytes",
        &cap_s,
        "--card-budget",
        &budget_s,
    ]);
    let gen_wall = t0.elapsed().as_secs_f64();
    assert_eq!(rc, Some(0));
    assert!(
        stdout.contains("fixture: generate done tensors=9 "),
        "no summary"
    );
    for n in subset {
        assert!(
            stdout.contains(&format!("fixture: tensor {n} ")),
            "no line for {n}"
        );
    }
    let peak = children_peak_kib();
    let first = out.join(format!("{}-00001-of-00002.gguf", v41::STEM));
    let fx = Split::open(&first).unwrap();
    assert_eq!(fx.shard_count(), 2);
    let first_s = first.to_str().unwrap();
    let (rc, stdout, _) = run(&["verify", first_s, "--source", &src_path]);
    assert_eq!(rc, Some(0));
    assert!(stdout.contains("fixture: verify done tensors=9 ") && stdout.contains("subset=true"));
    print_file_shas(&out);
    let (rc, _, stderr) = run(&[
        "verify",
        first_s,
        "--source",
        &src_path,
        "--draft-source",
        &draft_src,
    ]);
    assert_eq!(rc, Some(1));
    assert!(
        stderr.contains("names a draft source, but"),
        "verify must refuse a draft source with no draft fixture"
    );
    let (rc, _, stderr) = run(&[
        "generate",
        &src_path,
        out_s,
        "--seed",
        "7",
        "--tensors",
        &list,
        "--card-budget",
        &budget_s,
    ]);
    assert_eq!(rc, Some(1));
    assert!(
        stderr.contains("a fixture is never written over"),
        "generate must name the existing directory"
    );
    let (s, info) = fx.find("blk.2.ffn_down_shexp.weight").unwrap();
    let at = fx.shard(s).unwrap().data_base() + info.offset + 4096;
    let path = fx.shard_path(s).unwrap().to_path_buf();
    drop(fx);
    flip(&path, at);
    let (rc, _, stderr) = run(&["verify", first_s, "--source", &src_path]);
    assert_eq!(rc, Some(1));
    assert!(
        stderr.contains("differs from the generator's"),
        "verify must name the flipped byte"
    );
    std::fs::remove_dir_all(&out).unwrap();

    // The draft subset: a one-tensor target beside every draft type at its
    // real shape (`markov_w1` also gives the draft reader its rank). Written
    // after the target subset is gone.
    let dsub = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.attn_kv.weight",
        "blk.0.hc_attn_fn.weight",
        "blk.0.attn_norm.weight",
        "markov_w1.weight",
    ];
    let dlist = dsub.join(",");
    let out = d.join("draft");
    let out_s = out.to_str().unwrap();
    let t1 = Instant::now();
    let (rc, stdout, _) = run(&[
        "generate",
        &src_path,
        out_s,
        "--seed",
        "7",
        "--draft",
        &draft_src,
        "--tensors",
        "blk.0.attn_norm.weight",
        "--draft-tensors",
        &dlist,
        "--card-budget",
        &budget_s,
    ]);
    let draft_wall = t1.elapsed().as_secs_f64();
    assert_eq!(rc, Some(0));
    assert!(
        stdout.contains("fixture: generate done tensors=6 "),
        "no summary"
    );
    let peak2 = children_peak_kib();
    let beside: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".gguf"))
        .collect();
    assert_eq!(
        beside,
        [format!("{}-00001-of-00001.gguf", v41::STEM)],
        "a directory reader of the target's shards must not meet the draft"
    );
    let dfile = out.join(v41::DRAFT_FILE);
    let dx = Split::open(&dfile).unwrap();
    let dtypes: Vec<String> = dx
        .iter_tensors()
        .map(|(_, t)| format!("{} {} {:?}", t.name, t.ty, t.dims))
        .collect();
    let dbytes: u64 = dx.iter_tensors().map(|(_, t)| t.nbytes).sum();
    println!(
        "e2e: draft subset, {dbytes} B at real shapes:\n  {}",
        dtypes.join("\n  ")
    );
    assert!(dbytes < 1 << 30);
    drop(dx);
    print_file_shas(&out);
    let first = out.join(format!("{}-00001-of-00001.gguf", v41::STEM));
    let (rc, stdout, _) = run(&[
        "verify",
        first.to_str().unwrap(),
        "--source",
        &src_path,
        "--draft-source",
        &draft_src,
    ]);
    assert_eq!(rc, Some(0));
    assert!(
        stdout.contains("draft_tensors=5 "),
        "the draft was verified"
    );
    std::fs::remove_dir_all(&out).unwrap();
    println!(
        "e2e: target subset {bytes} B generated in {gen_wall:.2} s, draft subset {dbytes} B in {draft_wall:.2} s; children's peak RSS {peak} KiB, then {peak2} KiB; never both on disk"
    );
}

/// A header-only draft of `kvs`, no tensors, at `path`.
fn draft_header(path: &Path, kvs: &[(String, Value)]) -> Split {
    let layout = Layout::new(kvs, Vec::new()).unwrap();
    Writer::new(File::create(path).unwrap(), layout)
        .unwrap()
        .finish()
        .unwrap();
    Split::open(path).unwrap()
}

/// Set the u32 `general.alignment` of the file at `path` to `to`, and pad
/// the file so its data base, rounded to `to`, lies inside it.
fn patch_alignment(path: &Path, to: u32) {
    let mut bytes = std::fs::read(path).unwrap();
    let key = gguf::GENERAL_ALIGNMENT.as_bytes();
    let at = bytes
        .windows(key.len())
        .position(|w| w == key)
        .expect("the file holds the alignment key")
        + key.len();
    let tag = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert_eq!(tag, 4, "the alignment is a u32");
    bytes[at + 4..at + 8].copy_from_slice(&to.to_le_bytes());
    bytes.resize(bytes.len() + to as usize, 0);
    std::fs::write(path, bytes).unwrap();
}

/// A draft's metadata: the dflash architecture, the real draft's ff (the
/// V4.1 spec moves it with the target's), `target_layers`, and `extra`, which
/// replaces a key the draft already has.
fn draft_kvs(target_layers: std::ops::Range<u32>, extra: &[(&str, Value)]) -> Vec<(String, Value)> {
    let mut kvs = vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(model::arch::DFLASH.into()),
        ),
        (
            format!("{}.expert_feed_forward_length", model::arch::DFLASH),
            Value::U32(2304),
        ),
        (
            format!("{}.target_layers", model::arch::DFLASH),
            Value::Array(target_layers.map(Value::U32).collect()),
        ),
    ];
    for (k, v) in extra {
        match kvs.iter_mut().find(|(have, _)| have == k) {
            Some(slot) => slot.1 = v.clone(),
            None => kvs.push((k.to_string(), v.clone())),
        }
    }
    kvs
}

/// Contract 5: refusals. Every case runs; the failures are listed together.
#[test]
#[ignore = "needs the box and the V4.1 file (just gate-fixture)"]
fn hw_fixture_refusals() {
    let src_path = gguf::v41::model();
    let src = source();
    let n_layer = u32::try_from(src.arch_get_u64("block_count").unwrap()).unwrap();
    let guard = dir("refusals");
    let d = &guard.0;
    let spec = v41::spec();
    let mut bad: Vec<String> = Vec::new();

    // A draft that reads the source's last ten layers: more than the fixture's nine.
    let n = LAYER_MAP.len() as u32 + 1;
    let wide = draft_header(&d.join("wide.gguf"), &draft_kvs(n_layer - n..n_layer, &[]));
    match fixture::plan(&spec, &src, Some(&wide), &spec.options()) {
        Err(e @ FixtureError::Metadata { .. })
            if e.to_string()
                .contains("more target layers than the fixture's 9") =>
        {
            println!("refusals: {n} target layers: {e}");
        }
        Err(e) => bad.push(format!("{n} target layers: refused as {e}")),
        Ok(p) => bad.push(format!(
            "{n} target layers: planned, draft kvs {:?}",
            p.draft.map(|d| d.kvs)
        )),
    }

    // A draft whose alignment is 48: the reader takes it, the writer takes only
    // a power of two. Our writer refuses to write one, so its value is patched.
    let align_path = d.join("align.gguf");
    drop(draft_header(
        &align_path,
        &draft_kvs(
            n_layer - 3..n_layer,
            &[(gguf::GENERAL_ALIGNMENT, Value::U32(32))],
        ),
    ));
    patch_alignment(&align_path, 48);
    let align = Split::open(&align_path).unwrap();
    match fixture::plan(&spec, &src, Some(&align), &spec.options()) {
        Err(e @ FixtureError::Alignment { .. }) => println!("refusals: alignment 48: {e}"),
        Err(e) => bad.push(format!("alignment 48: refused as {e}")),
        Ok(_) => bad.push("alignment 48: planned".to_string()),
    }

    // An ff override where the family has no ff rule: the Qwen3.8 spec names no
    // ff key (PIN(2026-10-08): this clause refused the same override on V4.1's
    // spec until V4.1 gained its ff rule that day; the override is now V4.1's own
    // contract, below, and the refusal for a family with no rule moved to the
    // spec that has none).
    let qsrc = q38_source();
    let rowless = FixtureSpec {
        ff: Some(512),
        ..q38::spec()
    };
    match fixture::plan(&rowless, &qsrc, None, &rowless.options()) {
        Err(e @ FixtureError::Metadata { .. }) if e.to_string().contains("has no ff key") => {
            println!("refusals: ff 512 on a spec with no ff rule: {e}");
        }
        Err(e) => bad.push(format!("ff 512 on Qwen3.8: refused as {e}")),
        Ok(_) => bad.push("ff 512 on Qwen3.8: planned".to_string()),
    }

    // An ff the K-quant blocks do not tile: the down stacks' K is the ff.
    let ragged = FixtureSpec {
        ff: Some(500),
        ..v41::spec()
    };
    match fixture::plan(&ragged, &src, None, &ragged.options()) {
        Err(e @ FixtureError::Tensor { .. })
            if e.to_string().contains("not whole blocks of 256") =>
        {
            println!("refusals: ff 500: {e}");
        }
        Err(e) => bad.push(format!("ff 500: refused as {e}")),
        Ok(_) => bad.push("ff 500: planned".to_string()),
    }

    // A draft whose ff is not the target's: one override moves both.
    let odd = draft_header(
        &d.join("odd-ff.gguf"),
        &draft_kvs(
            n_layer - 3..n_layer,
            &[("dflash.expert_feed_forward_length", Value::U32(1024))],
        ),
    );
    match fixture::plan(&spec, &src, Some(&odd), &spec.options()) {
        Err(e @ FixtureError::Metadata { .. })
            if e.to_string().contains("not the target's ff 2304") =>
        {
            println!("refusals: a draft at ff 1024: {e}");
        }
        Err(e) => bad.push(format!("a draft at ff 1024: refused as {e}")),
        Ok(_) => bad.push("a draft at ff 1024: planned".to_string()),
    }

    // A file of an architecture no spec covers, and the V4-Flash string the
    // engine reads with the V4.1 module: the binary refuses both by name.
    for arch in ["nofamily", model::arch::DEEPSEEK4] {
        let path = d.join(format!("{arch}.gguf"));
        let kvs = [(
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(arch.into()),
        )];
        drop(draft_header(&path, &kvs));
        let want = format!("architecture Some({arch:?}) has no fixture spec");
        let (rc, _, stderr) = run(&["plan", path.to_str().unwrap()]);
        if rc != Some(1) || !stderr.contains(&want) {
            bad.push(format!("{arch}: rc {rc:?}, not refused with {want:?}"));
        }
    }

    // Flags a verb does not take, a flag twice, and a draft subset with no draft.
    let fx = d.join("none.gguf");
    let fx_s = fx.to_str().unwrap();
    for (args, want) in [
        (
            vec!["verify", fx_s, "--seed", "7"],
            "verify does not take --seed",
        ),
        (
            vec!["plan", &src_path, "--source", &src_path],
            "plan does not take --source",
        ),
        (
            vec!["plan", &src_path, "--seed", "1", "--seed", "2"],
            "--seed is given twice",
        ),
        (
            vec!["plan", &src_path, "--draft-tensors", "a"],
            "--draft-tensors needs --draft",
        ),
    ] {
        let (rc, _, stderr) = run(&args);
        if rc != Some(1) || !stderr.contains(want) {
            bad.push(format!("{args:?}: rc {rc:?}, not refused with {want:?}"));
        }
    }
    assert!(
        bad.is_empty(),
        "not refused by name:\n  {}",
        bad.join("\n  ")
    );
    println!("refusals: all 11 refused by name");
}

/// Contract 13: V4.1's recorded card budget makes the written file's plan, at
/// the gates' machine and context, hold half of every layer's experts on the
/// card (192 of 384), as `verify` re-plans it. The condition table is printed:
/// the budget is the plan of one resident sequence at [`v41::BUDGET_CTX`], and
/// a gate that plans another sequence count or context reads another table row.
#[test]
#[ignore = "needs the box and the V4.1 file (just gate-fixture)"]
fn hw_fixture_v41_budget() {
    let src = source();
    let spec = v41::spec();
    let CardBudget::Planned(planner) = spec.card_budget else {
        panic!("the V4.1 spec plans its card budget");
    };
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let recorded = p.card_budget;
    let mut kvs = p.target.kvs.clone();
    assert_eq!(*kv(&mut kvs, KEY_CARD_BUDGET), Value::U64(recorded));

    let guard = dir("v41budget");
    let first = header_only(&p.target, &guard.0);
    // The family's bias check reads the biases: a hole file has none to read.
    write_biases(&p.target, &first);
    let fx = Split::open(&first).unwrap();
    let inputs = V41Inputs::describe(&fx).unwrap();
    let (layers, experts) = (inputs.model.layers, inputs.model.experts);
    assert_eq!((layers, experts), (9, 384));
    let machine = workstation::plan_gate(layers);
    let at = |budget: Option<u64>, ctx: u64, slots: usize| -> Result<Vec<u64>, String> {
        let levers = PlanLevers {
            card_budget_bytes: budget,
        };
        let slots = NonZeroUsize::new(slots).unwrap();
        inputs
            .plan_with_slots(&machine, ctx, &levers, slots)
            .map(|plan| plan.n_l.clone())
            .map_err(|e| e.to_string())
    };
    let ctx = v41::BUDGET_CTX;
    // f0 and f1 are the source's first two layers, whose Q5_K down no card
    // kernel runs: their experts stay on the host.
    let want = |n: u64| -> Vec<u64> { (0..layers).map(|l| if l < 2 { 0 } else { n }).collect() };
    assert_eq!(
        at(None, ctx, 1).unwrap(),
        want(experts),
        "the budgetless plan holds every expert it can"
    );
    assert_eq!(
        at(Some(recorded), ctx, 1).unwrap(),
        want(experts / 2),
        "the plan under {recorded}"
    );
    assert!(
        recorded < RTX_3090.usable_bytes(),
        "a budget {recorded} that does not bind a card of {}",
        RTX_3090.usable_bytes()
    );
    fixture::check_budget(&planner, &fx, recorded).unwrap();

    let mut bad: Vec<String> = Vec::new();
    for (what, budget) in [
        ("64 MiB below", recorded - (64 << 20)),
        ("64 MiB above", recorded + (64 << 20)),
    ] {
        match fixture::check_budget(&planner, &fx, budget) {
            Err(e @ FixtureError::Budget(_)) => println!("v41 budget: {what}: {e}"),
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(c) => bad.push(format!("{what}: holds {:?}", c.per_layer)),
        }
    }

    // `verify` re-plans the written file under the budget its header records.
    let mut edited = p.target.clone();
    *kv(&mut edited.kvs, KEY_CARD_BUDGET) = Value::U64(recorded - (64 << 20));
    let g2 = dir("v41budget-edited");
    let first2 = header_only(&edited, &g2.0);
    write_biases(&edited, &first2);
    let fx2 = Split::open(&first2).unwrap();
    match fixture::verify(&spec, &fx2, &src, None, &mut |_, _| {}) {
        Err(e @ FixtureError::Budget(_)) if e.to_string().contains("plan under the budget") => {
            println!(
                "v41 budget: verify of a header recording {}: {e}",
                recorded - (64 << 20)
            );
        }
        other => bad.push(format!("verify of an edited budget: {other:?}")),
    }
    // The control: the recorded header passes that check and meets its holes.
    match fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}) {
        Err(FixtureError::Budget(m)) => bad.push(format!("verify of the recorded budget: {m}")),
        Err(e) => println!("v41 budget: control, the recorded header passes the budget check: {e}"),
        Ok(_) => bad.push("a file of holes verified".into()),
    }

    println!(
        "v41 budget: recorded {recorded} B ({} MiB) at ctx {ctx}, 1 slot, plan_gate({layers}); \
         budgetless {:?}; recorded {:?}",
        recorded / (1 << 20),
        at(None, ctx, 1).unwrap(),
        at(Some(recorded), ctx, 1).unwrap(),
    );
    for slots in [1, 2] {
        for c in [4096, 8192, 16384, ctx] {
            println!(
                "v41 budget: condition ctx {c} slots {slots}: under the recorded budget {:?}, \
                 budgetless {:?}",
                at(Some(recorded), c, slots),
                at(None, c, slots)
            );
        }
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// `bias`'s refusal is the family's, by the layer's name and the coverage
/// clause it breaks.
fn bias_refused(
    bad: &mut Vec<String>,
    what: &str,
    got: Result<impl std::fmt::Debug, FixtureError>,
    layer: usize,
) {
    match got {
        Err(e @ FixtureError::Mismatch { .. })
            if e.to_string()
                .contains(&format!("layer {layer}'s router biases"))
                && e.to_string().contains("coverage clause") =>
        {
            println!("v41 biases: {what}: {e}");
        }
        Err(e) => bad.push(format!("{what}: refused as {e}")),
        Ok(v) => bad.push(format!("{what}: accepted, {v:?}")),
    }
}

/// Contract 14: V4.1's routed layers carry two router biases that differ
/// between experts, which the media gate's coverage clause needs (a constant
/// difference moves no pick): the text bias is a constant 0, the media bias a
/// uniform draw, each layer's its own; and the family's header check refuses a
/// file whose biases are zeros, equal, or one a constant shift of the other.
#[test]
#[ignore = "needs the box and the V4.1 file (just gate-fixture)"]
fn hw_fixture_v41_biases() {
    let src = source();
    let spec = v41::spec();
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let t = &p.target;
    let find = |name: &str| {
        t.tensors
            .iter()
            .find(|x| x.name == name)
            .unwrap_or_else(|| panic!("no {name}"))
    };
    let routed: Vec<usize> = (0..LAYER_MAP.len())
        .filter(|f| {
            t.tensors
                .iter()
                .any(|x| x.name == format!("blk.{f}.exp_probs_b_vl.bias"))
        })
        .collect();
    assert_eq!(routed.len(), 9, "every layer of the map is routed");
    let (text_of, media_of) = (
        |f: usize| format!("blk.{f}.exp_probs_b.bias"),
        |f: usize| format!("blk.{f}.exp_probs_b_vl.bias"),
    );

    let mut spreads: Vec<Vec<f32>> = Vec::new();
    for &f in &routed {
        assert_eq!(find(&text_of(f)).rule, Rule::Const { value: 0.0 });
        let m = find(&media_of(f));
        assert_eq!(
            m.rule,
            Rule::Uniform {
                ty: fixture::FloatTy::F32,
                half_width: v41::VL_BIAS_HALF_WIDTH
            }
        );
        let v = floats(&filled(m, fixture::DEFAULT_SEED));
        let (lo, hi) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(
            v.iter().all(|x| x.abs() <= v41::VL_BIAS_HALF_WIDTH) && hi - lo >= v41::BIAS_SPREAD_MIN,
            "layer {f}: media bias spans [{lo}, {hi}]"
        );
        spreads.push(v);
    }
    for (i, a) in spreads.iter().enumerate() {
        for b in &spreads[i + 1..] {
            assert!(a != b, "two layers' media biases are one stream");
        }
    }
    println!(
        "v41 biases: {} routed layers, text const 0, media uniform ±{}; spans [{:.4}, {:.4}] over {} experts",
        routed.len(),
        v41::VL_BIAS_HALF_WIDTH,
        spreads
            .iter()
            .map(|v| v.iter().fold(f32::MAX, |a, &x| a.min(x)))
            .fold(f32::MAX, f32::min),
        spreads
            .iter()
            .map(|v| v.iter().fold(f32::MIN, |a, &x| a.max(x)))
            .fold(f32::MIN, f32::max),
        spreads[0].len(),
    );

    // The header check on a header-only file whose biases take each form.
    let guard = dir("v41biases");
    let first = header_only(t, &guard.0);
    let check = || v41::check_kinds(&spec, &Split::open(&first).unwrap(), &src);
    let mut bad: Vec<String> = Vec::new();
    bias_refused(&mut bad, "biases of zeros (holes)", check(), routed[0]);
    write_biases(t, &first);
    if let Err(e) = check() {
        bad.push(format!("the generated biases: {e}"));
    }
    let l = routed[4];
    let (text, media) = (text_of(l), media_of(l));
    let spread = f32_bytes(&spreads[4]);
    let zeros = vec![0u8; spread.len()];
    // Text and media the same non-constant draw.
    put(&first, &text, &spread);
    bias_refused(&mut bad, "equal biases", check(), l);
    // Media a constant shift of text.
    let shifted: Vec<f32> = spreads[4].iter().map(|x| x + 0.3).collect();
    put(&first, &media, &f32_bytes(&shifted));
    bias_refused(&mut bad, "a constant shift", check(), l);
    // Restored.
    put(&first, &text, &zeros);
    put(&first, &media, &spread);
    if let Err(e) = check() {
        bad.push(format!("the biases restored: {e}"));
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// Contract 15: V4.1's r8 sidecar takes the gate and up stack of every layer
/// of the fixture, in layer order, each a Q3_K stack of the source's type and
/// width on `r8file`'s grid at the fixture's ff.
#[test]
#[ignore = "needs the box and the V4.1 file (just gate-fixture)"]
fn hw_fixture_v41_r8() {
    let src = source();
    let spec = v41::spec();
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let guard = dir("v41r8");
    let fx = Split::open(header_only(&p.target, &guard.0)).unwrap();
    let sidecar = spec.sidecar.expect("V4.1 has an r8 sidecar");
    let names = (sidecar.stacks)(&fx).unwrap();
    let want: Vec<String> = (0..LAYER_MAP.len())
        .flat_map(|f| {
            [
                format!("blk.{f}.ffn_gate_exps.weight"),
                format!("blk.{f}.ffn_up_exps.weight"),
            ]
        })
        .collect();
    assert_eq!(names, want);
    let mut bytes = 0;
    for (i, n) in names.iter().enumerate() {
        let (_, info) = fx.find(n).unwrap();
        let source_name = format!(
            "blk.{}.{}",
            LAYER_MAP[i / 2],
            n.split_once('.').unwrap().1.split_once('.').unwrap().1
        );
        let (_, s) = src.find(&source_name).unwrap();
        assert_eq!(info.ty, GgmlType::Q3_K, "{n}");
        assert_eq!(info.ty, s.ty, "{n}");
        assert_eq!(info.dims, ff_dims(leaf_of(n), &s.dims), "{n}");
        assert_eq!(info.dims[1], v41::FIXTURE_FF, "{n}");
        assert!(
            info.dims[0] % 256 == 0 && info.dims[1] % 8 == 0,
            "{n}: r8's grid"
        );
        bytes += info.nbytes;
    }
    println!(
        "v41 r8: {} stacks {:?}, {bytes} B of Q3_K ({:.2} GB); the sidecar's data is that many bytes",
        names.len(),
        fx.find(&names[0]).unwrap().1.dims,
        bytes as f64 / 1e9
    );
}

/// The second family's rules: every generic key rule, a constant for its
/// norms, a spread for its bias, an ff axis on each expert stack, and a table.
/// `both` also gives the bias a constant: a family that names both is refused.
struct Toy {
    both: bool,
}

/// The half-width of the toy bias's spread.
const TOY_SPREAD: f32 = 0.25;

impl Family for Toy {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        Some(match suffix {
            "block_count" => KeyRule::BlockCount,
            "embedding_length" => KeyRule::Copy,
            "ratios" => KeyRule::Ratios,
            "layer_list" => KeyRule::LayerIds,
            "per_layer" => KeyRule::PerLayer,
            "expert_feed_forward_length" => KeyRule::Ff,
            "table_rows" => KeyRule::Table,
            _ => return None,
        })
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        (matches!(leaf, "norm.weight" | "output_norm.weight")
            || (self.both && leaf == "bias.weight"))
            .then_some(1.0)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        (leaf == "bias.weight").then_some(TOY_SPREAD)
    }

    fn ff_axis(&self, leaf: &str) -> Option<usize> {
        match leaf {
            "ffn_up_exps.weight" => Some(1),
            "ffn_down_exps.weight" => Some(0),
            _ => None,
        }
    }

    fn tables(
        &self,
        _source: &Split,
        _spec: &FixtureSpec,
    ) -> Result<Box<dyn Tables>, FixtureError> {
        Ok(Box::new(ToyTable))
    }

    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        _source: &Split,
    ) -> Result<(), FixtureError> {
        let n = fixture.arch_get_u64("block_count");
        if n == Some(spec.layers.len() as u64) {
            Ok(())
        } else {
            Err(FixtureError::Mismatch {
                what: "layer count".into(),
                detail: format!("{n:?}"),
            })
        }
    }
}

/// The second family's table: `tab.weight` and its row count cut to
/// [`TOY_ROWS`].
struct ToyTable;

const TOY_ROWS: u64 = 100;

impl Tables for ToyTable {
    fn key(&self, key: &str, suffix: &str, _v: &Value) -> Result<Value, FixtureError> {
        match suffix {
            "table_rows" => Ok(Value::U32(TOY_ROWS as u32)),
            _ => Err(FixtureError::Metadata {
                key: key.to_string(),
                detail: "not a toy table key".into(),
            }),
        }
    }

    fn dims(
        &self,
        _name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        Ok((layer.is_none() && leaf == "tab.weight").then(|| vec![dims[0], TOY_ROWS]))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!("toy table rows={TOY_ROWS} (source 1000)")]
    }
}

static TOY: Toy = Toy { both: false };
static TOY_BOTH: Toy = Toy { both: true };

/// A four-layer source of the second family: one file, its tensors zero.
fn toy_source(path: &Path) -> Split {
    let arr = |v: &[u32]| Value::Array(v.iter().map(|&x| Value::U32(x)).collect());
    let mut tensors: Vec<(&str, Vec<u64>, GgmlType)> = vec![
        ("tab.weight", vec![64, 1000], GgmlType::BF16),
        ("output_norm.weight", vec![64], GgmlType::F32),
    ];
    let names: Vec<[String; 6]> = (0..4)
        .map(|l| {
            ["norm", "attn", "ffn_up_exps", "ffn_down_exps", "bias", "r8"]
                .map(|n| format!("blk.{l}.{n}.weight"))
        })
        .collect();
    for n in &names {
        tensors.push((&n[0], vec![64], GgmlType::F32));
        tensors.push((&n[1], vec![4096, 8], GgmlType::Q8_0));
        tensors.push((&n[2], vec![64, 512, 4], GgmlType::BF16));
        tensors.push((&n[3], vec![512, 64, 4], GgmlType::BF16));
        tensors.push((&n[4], vec![4096], GgmlType::F32));
        // A Q3_K stack the r8 sidecar takes: 4 experts of 16 rows of 256.
        tensors.push((&n[5], vec![256, 16, 4], GgmlType::Q3_K));
    }
    let kvs = vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String("toy".into()),
        ),
        ("split.no".to_string(), Value::U16(0)),
        ("split.count".to_string(), Value::U16(1)),
        (
            "split.tensors.count".to_string(),
            Value::I32(tensors.len() as i32),
        ),
        ("toy.block_count".to_string(), Value::U32(4)),
        ("toy.embedding_length".to_string(), Value::U32(64)),
        ("toy.ratios".to_string(), arr(&[5, 6, 6, 7, 9])),
        ("toy.layer_list".to_string(), arr(&[3])),
        ("toy.per_layer".to_string(), arr(&[10, 11, 12, 13])),
        (
            "toy.expert_feed_forward_length".to_string(),
            Value::U32(512),
        ),
        ("toy.table_rows".to_string(), Value::U32(1000)),
    ];
    let decls: Vec<TensorDecl> = tensors
        .iter()
        .map(|(name, dims, ty)| {
            let (_, blck, tsz) = gguf::ggml_type_info(ty.as_u32()).unwrap();
            TensorDecl {
                name: name.to_string(),
                dims: dims.clone(),
                type_id: ty.as_u32(),
                nbytes: tsz * (dims[0] / blck) * dims[1..].iter().product::<u64>(),
            }
        })
        .collect();
    let bytes: Vec<u64> = decls.iter().map(|t| t.nbytes).collect();
    let layout = Layout::new(&kvs, decls).unwrap();
    let mut w = Writer::new(File::create(path).unwrap(), layout).unwrap();
    for ((name, _, _), n) in tensors.iter().zip(bytes) {
        w.tensor(name, &vec![0u8; n as usize]).unwrap();
    }
    w.finish().unwrap();
    Split::open(path).unwrap()
}

fn toy_spec() -> FixtureSpec {
    FixtureSpec {
        arch: "toy",
        stem: "toy-fixture",
        layers: vec![0, 3],
        ratios: vec![5, 7],
        card_budget: CardBudget::Fixed(1 << 30),
        window: Window::new(1.0 / 16384.0, 1.0 / 64.0).unwrap(),
        ff: Some(256),
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source: String::new,
        draft: None,
        sidecar: None,
        family: &TOY,
    }
}

/// Contract 6: a second family.
#[test]
fn fixture_second_family() {
    let guard = dir("toy");
    let d = &guard.0;
    let src = toy_source(&d.join("toy-00001-of-00001.gguf"));
    let spec = toy_spec();
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let kv = |k: &str| {
        &p.target
            .kvs
            .iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("no {k}"))
            .1
    };
    assert_eq!(kv("toy.block_count"), &Value::U32(2));
    assert_eq!(unsigned_items(kv("toy.ratios")), [5, 7, 9]);
    assert_eq!(unsigned_items(kv("toy.layer_list")), [1]);
    assert_eq!(unsigned_items(kv("toy.per_layer")), [10, 13]);
    assert_eq!(kv("toy.expert_feed_forward_length"), &Value::U32(256));
    assert_eq!(kv("toy.table_rows"), &Value::U32(TOY_ROWS as u32));
    assert_eq!(unsigned_items(kv(KEY_SOURCE_LAYERS)), [0, 3]);
    let got: Vec<(String, String, Vec<u64>)> = p
        .target
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.source.clone(), t.dims.clone()))
        .collect();
    let mut want = vec![
        (
            "tab.weight".to_string(),
            "tab.weight".to_string(),
            vec![64, TOY_ROWS],
        ),
        (
            "output_norm.weight".into(),
            "output_norm.weight".into(),
            vec![64],
        ),
    ];
    for (f, l) in [(0, 0), (1, 3)] {
        for (n, dims) in [
            ("norm", vec![64]),
            ("attn", vec![4096, 8]),
            ("ffn_up_exps", vec![64, 256, 4]),
            ("ffn_down_exps", vec![256, 64, 4]),
            ("bias", vec![4096]),
            ("r8", vec![256, 16, 4]),
        ] {
            want.push((
                format!("blk.{f}.{n}.weight"),
                format!("blk.{l}.{n}.weight"),
                dims,
            ));
        }
    }
    assert_eq!(got, want);
    assert_eq!(
        p.notes,
        [format!("toy table rows={TOY_ROWS} (source 1000)")]
    );
    println!(
        "second family: plan {} tensors, notes {:?}",
        got.len(),
        p.notes
    );

    let out = d.join("out");
    let opts = Options {
        seed: 3,
        ..spec.options()
    };
    let s = fixture::generate(&spec, &src, None, &out, &opts, &mut |_| {}).unwrap();
    let fx = Split::open(out.join("toy-fixture-00001-of-00001.gguf")).unwrap();
    let (v, none) = fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}).unwrap();
    assert_eq!((v.tensors, v.subset, none.is_none()), (14, false, true));
    assert!(
        s.sidecar.is_none() && v.sidecar.is_none() && !d.join("out-r8").exists(),
        "a family with no sidecar writes and checks none"
    );
    println!(
        "second family: wrote {} tensors, {} bytes; verify {} tensors, {} blocks",
        s.tensors, s.file_bytes, v.tensors, v.blocks
    );

    // A spread tensor: a uniform draw in ±the family's half-width, never a
    // constant, and each tensor's own (its stream is keyed by its name).
    let bias = |l: usize| {
        p.target
            .tensors
            .iter()
            .find(|t| t.name == format!("blk.{l}.bias.weight"))
            .unwrap()
    };
    let (b0, b1) = (bias(0), bias(1));
    assert_eq!(
        b0.rule,
        Rule::Uniform {
            ty: fixture::FloatTy::F32,
            half_width: TOY_SPREAD
        }
    );
    assert!((b0.sigma().unwrap() - f64::from(TOY_SPREAD) / 3.0f64.sqrt()).abs() < 1e-9);
    let (v0, v1) = (sample(b0, 3), sample(b1, 3));
    let (f0, f1) = (floats(&v0), floats(&v1));
    assert!(
        f0.iter().all(|x| x.abs() <= TOY_SPREAD),
        "inside ±{TOY_SPREAD}"
    );
    assert!(
        f0.iter().any(|&x| x != f0[0]),
        "a spread tensor holds more than one value"
    );
    assert!(f0 != f1, "two layers' biases are one stream");
    println!(
        "second family: bias spread ±{TOY_SPREAD}, blk.0 {} distinct of {}, blk.0 ≠ blk.1",
        {
            let mut d = f0.clone();
            d.sort_by(f32::total_cmp);
            d.dedup();
            d.len()
        },
        f0.len()
    );

    let mut bad: Vec<String> = Vec::new();
    let refusals: [(&str, FixtureSpec, bool, &str); 5] = [
        (
            "a constant and a spread for one leaf",
            FixtureSpec {
                family: &TOY_BOTH,
                ..toy_spec()
            },
            false,
            "names both a constant and a spread",
        ),
        (
            "the spec's ratios one short",
            FixtureSpec {
                ratios: vec![5],
                ..toy_spec()
            },
            false,
            "the fixture is built for [5]",
        ),
        (
            "a layer list the map does not hold",
            FixtureSpec {
                layers: vec![0, 1],
                ratios: vec![5, 6],
                ..toy_spec()
            },
            false,
            "lists layer 3, which the map does not hold",
        ),
        (
            "a draft for a family with none",
            toy_spec(),
            true,
            "the toy fixture has no draft file",
        ),
        (
            "another architecture",
            FixtureSpec {
                arch: "other",
                ..toy_spec()
            },
            false,
            "not a other file",
        ),
    ];
    for (what, spec, with_draft, want) in refusals {
        let draft = with_draft.then_some(&src);
        match fixture::plan(&spec, &src, draft, &spec.options()) {
            Err(e) if e.to_string().contains(want) => println!("second family: {what}: {e}"),
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(_) => bad.push(format!("{what}: planned")),
        }
    }
    assert!(
        bad.is_empty(),
        "not refused by name:\n  {}",
        bad.join("\n  ")
    );
}

/// The stacks the toy's sidecar holds: each layer's `r8` tensor, in file order.
fn toy_stacks(split: &Split) -> Result<Vec<String>, FixtureError> {
    Ok(split
        .iter_tensors()
        .filter(|(_, t)| t.name.ends_with(".r8.weight"))
        .map(|(_, t)| t.name.clone())
        .collect())
}

/// Only the first layer's stack: a family that names fewer stacks than the
/// sidecar beside the file holds.
fn toy_first_stack(split: &Split) -> Result<Vec<String>, FixtureError> {
    Ok(toy_stacks(split)?.into_iter().take(1).collect())
}

/// A family that names no stack.
fn toy_no_stacks(_: &Split) -> Result<Vec<String>, FixtureError> {
    Ok(Vec::new())
}

fn toy_sidecar_spec(stacks: fn(&Split) -> Result<Vec<String>, FixtureError>) -> FixtureSpec {
    FixtureSpec {
        sidecar: Some(SidecarSpec { stacks }),
        ..toy_spec()
    }
}

/// `got` is the sidecar refusal whose text holds `want`, else `bad` gains why not.
fn sidecar_refused<T: std::fmt::Debug>(
    bad: &mut Vec<String>,
    what: &str,
    got: Result<T, FixtureError>,
    want: &str,
) {
    match got {
        Err(e @ FixtureError::Sidecar { .. }) if e.to_string().contains(want) => {
            println!("sidecar: {what}: {e}");
        }
        Err(e) => bad.push(format!("{what}: refused as {e}")),
        Ok(v) => bad.push(format!("{what}: accepted, {v:?}")),
    }
}

/// Contract 12: the r8 sidecar. A family with one gets it written beside the
/// fixture by `generate` and checked by `verify` against the fixture's own
/// bytes: absent, another stack set and a flipped byte are refused by name,
/// and a family that names no stack leaves no fixture behind.
#[test]
fn fixture_sidecar() {
    let guard = dir("sidecar");
    let d = &guard.0;
    let src = toy_source(&d.join("toy-00001-of-00001.gguf"));
    let spec = toy_sidecar_spec(toy_stacks);
    let opts = Options {
        seed: 3,
        ..spec.options()
    };
    let out = d.join("out");
    let s = fixture::generate(&spec, &src, None, &out, &opts, &mut |_| {}).unwrap();
    let first = out.join("toy-fixture-00001-of-00001.gguf");
    let side = s
        .sidecar
        .as_ref()
        .expect("a family with a sidecar writes one");
    let want = d.join("out-r8").join("toy-fixture-r8.gguf");
    assert_eq!(side.path, want, "the sidecar's name is r8file's");
    assert_eq!(side.path, fixture::sidecar_path(&first).unwrap());
    assert_eq!(side.tensors, 2, "one stack a layer");
    assert!(side.path.is_file());
    let len = std::fs::metadata(&side.path).unwrap().len();
    assert_eq!(side.bytes, len);

    let fx = Split::open(&first).unwrap();
    let (v, _) = fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}).unwrap();
    let checked = v.sidecar.as_ref().expect("verify checks the sidecar");
    assert_eq!((checked.tensors, &checked.path), (2, &side.path));
    println!(
        "sidecar: wrote {} stacks, {len} B in {:.3} s; verify read {} stack bytes in {:.3} s",
        side.tensors, side.secs, checked.bytes, checked.secs
    );

    let mut bad: Vec<String> = Vec::new();
    let check = |spec: &FixtureSpec| fixture::verify(spec, &fx, &src, None, &mut |_, _| {});

    // The sidecar moved away: the fixture's verify names it.
    let aside = side.path.with_extension("moved");
    std::fs::rename(&side.path, &aside).unwrap();
    sidecar_refused(&mut bad, "absent", check(&spec), "is absent");
    std::fs::rename(&aside, &side.path).unwrap();

    // The sidecar of another stack set than the family names.
    let fewer = toy_sidecar_spec(toy_first_stack);
    sidecar_refused(
        &mut bad,
        "one stack named, two held",
        check(&fewer),
        "holds 2 stacks",
    );

    // One byte of the last stack flipped: its repack no longer unpacks to the
    // fixture's bytes.
    flip(&side.path, len - 3520);
    sidecar_refused(
        &mut bad,
        "a flipped byte",
        check(&spec),
        "unpacks to other bytes",
    );
    flip(&side.path, len - 3520);
    if let Err(e) = check(&spec) {
        bad.push(format!("the byte flipped back: {e}"));
    }

    // A family that names no stack: the generate fails and the fixture is
    // not left without its sidecar.
    let none = toy_sidecar_spec(toy_no_stacks);
    let out2 = d.join("out2");
    sidecar_refused(
        &mut bad,
        "no stack named",
        fixture::generate(&none, &src, None, &out2, &opts, &mut |_| {}),
        "no tensors to convert",
    );
    if out2.exists() || d.join("out2-r8").exists() {
        bad.push("a failed sidecar left files behind".into());
    }
    assert!(
        bad.is_empty(),
        "not refused by name:\n  {}",
        bad.join("\n  ")
    );
}

/// A tensor of `ty` with rows of `k` values under `rule`, as many rows as one
/// chunk holds.
fn one_chunk(ty: GgmlType, k: u64, rule: Rule) -> PlannedTensor {
    let unit = ty.type_size().unwrap();
    let rows = CHUNK_TARGET as u64 / unit * 32 / k;
    PlannedTensor {
        name: format!("t.{ty}.{k}"),
        source: String::new(),
        layer: None,
        dims: vec![k, rows],
        ty,
        nbytes: rows * (k / 32) * unit,
        rule,
    }
}

/// `t`'s one chunk filled under seed 7 and dequantized by `gguf::dequant_row`:
/// its RMS over `1/√K`, after every block passes `check_units` and the sample
/// `rms_within` under `window`.
fn filled_ratio(t: &PlannedTensor, window: Window) -> Result<f64, FixtureError> {
    let mut bytes = vec![0u8; t.nbytes as usize];
    t.fill_chunk(7, 0, &mut bytes);
    let mut y = vec![0f32; bytes.len() / t.ty.type_size().unwrap() as usize * 32];
    gguf::dequant_row(t.ty, &bytes, &mut y).unwrap();
    let rms = (y.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / y.len() as f64).sqrt();
    let sample = fixture::check_units(t, &bytes, 0, window)?;
    fixture::rms_within(t, &sample)?;
    Ok(rms * (t.dims[0] as f64).sqrt())
}

/// The window the rule contracts below are written at: [2^-13, 2^-10], which
/// is no longer the V4.1 spec's. The contracts test the rules' fit and refusal
/// at this window, so they keep it as their own value. At ff 512 a Q4_K or Q5_K
/// down stack's `dmin` passes 2^-10 at the widest scale band, so the spec's
/// upper bound is 2^-9.
// PIN(2026-10-08): the contracts' window, formerly the V4.1 spec's.
fn rule_window() -> Window {
    Window::new(1.0 / 8192.0, 1.0 / 1024.0).unwrap()
}

/// Contract 7: the Q5_1 and IQ4_NL rules fill at σ under the rule window. Every
/// case runs; the failures are listed together.
#[test]
fn fixture_new_rules_fill_at_sigma() {
    let window = rule_window();
    let mut bad = Vec::new();
    for (ty, k) in [
        (GgmlType::Q5_1, 16384),
        (GgmlType::Q5_1, 65536),
        (GgmlType::IQ4_NL, 160),
        (GgmlType::IQ4_NL, 4096),
    ] {
        let t = match rule_for("t", ty, k, window) {
            Ok(rule) => one_chunk(ty, k, rule),
            Err(e) => {
                bad.push(format!("{ty} K = {k}: {e}"));
                continue;
            }
        };
        match filled_ratio(&t, window) {
            Ok(r) if (0.9..=1.1).contains(&r) => println!(
                "new rules: {ty} K = {k} in {window}: {}; RMS / (1/sqrt K) = {r:.4}",
                t.rule.describe()
            ),
            Ok(r) => bad.push(format!("{ty} K = {k}: RMS / (1/sqrt K) = {r:.4}")),
            Err(e) => bad.push(format!("{ty} K = {k}: {e}")),
        }
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// Contract 7: a σ whose `d` no code choice puts in the rule window is refused
/// with `NoScale`, at both ends of each rule's range. Every case runs.
#[test]
fn fixture_new_rules_refuse_out_of_window() {
    let window = rule_window();
    let mut bad = Vec::new();
    for (ty, k) in [
        (GgmlType::Q5_1, 640),
        (GgmlType::Q5_1, 1 << 21),
        (GgmlType::IQ4_NL, 32),
        (GgmlType::IQ4_NL, 1 << 27),
    ] {
        match rule_for("t", ty, k, window) {
            Err(e @ FixtureError::NoScale { .. }) => println!("new rules: {ty} K = {k}: {e}"),
            Err(e) => bad.push(format!("{ty} K = {k}: refused as {e}, not NoScale")),
            Ok(r) => bad.push(format!("{ty} K = {k}: planned {}", r.describe())),
        }
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// Contract 8: the second family's window, [2^-14, 2^-6], is wider than the
/// rule window and admits Q5_1 at K = 640 (`d = σ/√85.25 = 2^-7.87`), which
/// the rule window refuses; its blocks fill at σ and hold under the wide
/// window, and the rule window refuses them by name.
#[test]
fn fixture_wider_window_admits_q5_1_at_k640() {
    let (ty, k) = (GgmlType::Q5_1, 640);
    let narrow = rule_window();
    let wide = toy_spec().window;
    match rule_for("t", ty, k, narrow) {
        Err(e @ FixtureError::NoScale { .. }) => println!("window: {ty} K = {k}: {e}"),
        other => panic!("{ty} K = {k} under {narrow}: {other:?}, not NoScale"),
    }
    let t = one_chunk(ty, k, rule_for("t", ty, k, wide).unwrap());
    let r = filled_ratio(&t, wide).unwrap();
    println!(
        "window: {ty} K = {k} in {wide}: {}; RMS / (1/sqrt K) = {r:.4}",
        t.rule.describe()
    );
    assert!((0.9..=1.1).contains(&r), "RMS / (1/sqrt K) = {r:.4}");
    match filled_ratio(&t, narrow) {
        Err(e @ FixtureError::Block { .. }) if e.to_string().contains(&narrow.to_string()) => {
            println!("window: the same blocks under {narrow}: {e}");
        }
        other => panic!("the K = 640 blocks under {narrow}: {other:?}, not a Block error"),
    }
}

/// Contract 8: a window that leaves the normal f16 values `[2^-14, 65504]` is
/// refused by name; their whole range is a window.
#[test]
fn fixture_window_leaves_normal_f16_refused() {
    let mut bad = Vec::new();
    for (lo, hi) in [
        (1.0 / 32768.0, 1.0 / 1024.0),
        (1.0 / 8192.0, 65536.0),
        (1.0 / 1024.0, 1.0 / 8192.0),
        (f32::NAN, 1.0),
    ] {
        match Window::new(lo, hi) {
            Err(e @ FixtureError::BadWindow { .. }) => println!("window: [{lo:e}, {hi:e}]: {e}"),
            Err(e) => bad.push(format!("[{lo:e}, {hi:e}]: refused as {e}")),
            Ok(w) => bad.push(format!("[{lo:e}, {hi:e}]: taken as {w}")),
        }
    }
    if let Err(e) = Window::new(1.0 / 16384.0, 65504.0) {
        bad.push(format!("the normal f16 range itself: {e}"));
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

// ------------------------------------------------------------------ Qwen3.8

const RATIOS: &str = "qwen4exp.attention.compress_ratios";
const PLE_LAYERS: &str = "qwen4exp.ple.layers";
const INTERVAL: &str = "qwen4exp.full_attention_interval";

fn q38_source() -> Split {
    Split::open(q38::DEFAULT_MODEL)
        .unwrap_or_else(|e| panic!("open the Qwen3.8 file {}: {e}", q38::DEFAULT_MODEL))
}

fn q38_draft() -> Split {
    Split::open(q38::DEFAULT_MTP_MODEL)
        .unwrap_or_else(|e| panic!("open the Qwen3.8 MTP draft {}: {e}", q38::DEFAULT_MTP_MODEL))
}

/// The value of `key` in `kvs`.
fn kv<'a>(kvs: &'a mut [(String, Value)], key: &str) -> &'a mut Value {
    &mut kvs
        .iter_mut()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("no {key}"))
        .1
}

/// `kvs` without `key`, which it must hold.
fn drop_key(kvs: &mut Vec<(String, Value)>, key: &str) {
    let n = kvs.len();
    kvs.retain(|(k, _)| k != key);
    assert_eq!(kvs.len(), n - 1, "no {key}");
}

/// The array `key` without its last item.
fn pop_item(kvs: &mut [(String, Value)], key: &str) {
    match kv(kvs, key) {
        Value::Array(a) => {
            a.pop().expect("a non-empty array");
        }
        other => panic!("{key} is {other:?}, not an array"),
    }
}

/// The integer `v` as `n` in `v`'s type.
fn retype(v: &Value, n: u64) -> Value {
    match v {
        Value::U8(_) => Value::U8(n as u8),
        Value::U16(_) => Value::U16(n as u16),
        Value::U32(_) => Value::U32(n as u32),
        Value::U64(_) => Value::U64(n),
        Value::I8(_) => Value::I8(n as i8),
        Value::I16(_) => Value::I16(n as i16),
        Value::I32(_) => Value::I32(n as i32),
        Value::I64(_) => Value::I64(n as i64),
        other => panic!("{other:?} is not an integer"),
    }
}

/// `src`'s header as one sparse file `name` under `d`: its metadata through
/// `edit` (a split set's keys as a one-file set's) and its tensors the ones
/// `keep` takes — what the planner reads of a source, whatever the source's
/// size.
fn edited_header(
    src: &Split,
    d: &Path,
    name: &str,
    keep: impl Fn(&str) -> bool,
    edit: impl FnOnce(&mut Vec<(String, Value)>),
) -> Split {
    let tensors: Vec<TensorDecl> = src
        .iter_tensors()
        .filter(|(_, t)| keep(&t.name))
        .map(|(_, t)| TensorDecl {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.ty.as_u32(),
            nbytes: t.nbytes,
        })
        .collect();
    let n = tensors.len() as u64;
    let mut kvs: Vec<(String, Value)> = src
        .iter_kv()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    for (k, v) in &mut kvs {
        let one = match k.as_str() {
            "split.no" => 0,
            "split.count" => 1,
            "split.tensors.count" => n,
            _ => continue,
        };
        *v = retype(v, one);
    }
    edit(&mut kvs);
    let layout = Layout::new(&kvs, tensors).unwrap();
    let path = d.join(name);
    let len = layout.file_len();
    let file = File::create(&path).unwrap();
    drop(Writer::new(&file, layout).unwrap());
    file.set_len(len).unwrap();
    Split::open(&path).unwrap()
}

/// A tensor of `map`'s layers, or no layer's.
fn keeps(map: &[usize], name: &str) -> bool {
    match name.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
        Some((l, _)) => l.parse::<usize>().is_ok_and(|l| map.contains(&l)),
        None => true,
    }
}

fn q38_keep(name: &str) -> bool {
    keeps(&q38::LAYER_MAP, name)
}

/// The integer key `key` set to `n`, in its own type.
fn set_int(kvs: &mut [(String, Value)], key: &str, n: u64) {
    let v = kv(kvs, key);
    *v = retype(&v.clone(), n);
}

/// An edit of a header's metadata.
type Edit = fn(&mut Vec<(String, Value)>);

/// Contract 9: the plan, and the engine's reader reading it back.
#[test]
#[ignore = "needs the box and the Qwen3.8 file and MTP draft (just gate-fixture)"]
fn hw_fixture_q38_plan() {
    let src = q38_source();
    let draft = q38_draft();
    let spec = q38::spec();
    let p = fixture::plan(&spec, &src, Some(&draft), &spec.options())
        .expect("the plan of the real files");
    let get = |plan: &FilePlan, k: &str| -> Value {
        plan.kvs
            .iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("no {k}"))
            .1
            .clone()
    };
    let t = &p.target;
    assert_eq!(unsigned_items(&get(t, KEY_SOURCE_LAYERS)), [2, 3, 1, 47]);
    assert_eq!(get(t, "qwen4exp.block_count").as_unsigned(), Some(4));
    assert_eq!(src.arch_get_u64("full_attention_interval"), Some(4));
    assert_eq!(get(t, INTERVAL).as_unsigned(), Some(2));
    assert_eq!(unsigned_items(&get(t, PLE_LAYERS)), [2]);
    assert_eq!(unsigned_items(&get(t, RATIOS)), [0, 4, 0, 4]);
    for pt in &t.tensors {
        if let Some(f) = pt.layer {
            assert!(pt.name.starts_with(&format!("blk.{f}.")), "{}", pt.name);
            let l = q38::LAYER_MAP[f];
            assert!(pt.source.starts_with(&format!("blk.{l}.")), "{}", pt.source);
        }
    }

    // `ssm_a` is `−e^A_log`, folded by the converter: a positive constant
    // would grow the recurrent state every step (decay `exp(softplus · ssm_a)`).
    let decays: Vec<f32> = t
        .tensors
        .iter()
        .filter(|pt| pt.name.ends_with(".ssm_a"))
        .map(|pt| match pt.rule {
            Rule::Const { value } => value,
            ref other => panic!("{}: {}, not a constant", pt.name, other.describe()),
        })
        .collect();
    assert_eq!(decays.len(), 2, "the two GDN layers' ssm_a");
    assert!(
        decays.iter().all(|&v| v < 0.0),
        "ssm_a {decays:?}: a decay below 1 needs it negative"
    );

    let guard = dir("q38plan");
    let first = header_only(t, &guard.0);
    let fx = Split::open(&first).unwrap();
    let hp = q38::check_kinds(&spec, &fx, &src).unwrap();
    assert_eq!((hp.n_layer, hp.interval), (4, 2));
    assert_eq!(
        hp.kinds,
        [
            Kind::DeltaRule,
            Kind::Attention,
            Kind::DeltaRule,
            Kind::Attention
        ]
    );
    let exp = hp.exp.as_ref().expect("a qwen4exp file");
    assert_eq!(exp.ratios, [0, 4, 0, 4]);
    assert_eq!(exp.ple.map(|p| p.layer), Some(2));

    let dplan = p.draft.as_ref().expect("the MTP companion's plan");
    let dfirst = header_only(dplan, &guard.0);
    // A run with no draft lever opens the real draft's file name beside its
    // target (`refset`'s `qwen4exp::mtp::draft_file`), else the real draft:
    // the companion is written under that name, beside the fixture.
    assert_eq!(
        Path::new(q38::DRAFT_FILE).file_name(),
        Path::new(q38::DEFAULT_MTP_MODEL).file_name(),
        "the companion's name is the real draft's"
    );
    assert_eq!(
        dfirst.parent(),
        first.parent(),
        "the companion lies beside the target"
    );
    let dfx = Split::open(&dfirst).unwrap();
    spec.draft
        .expect("the spec names the MTP companion")
        .rules
        .check(&spec, &dfx, &fx, true)
        .expect("the companion reads as a draft of the fixture");
    println!(
        "q38 plan: {} layers at interval {} kinds {:?} pools {:?} ple at {:?}; target {} B, draft {} B",
        hp.n_layer,
        hp.interval,
        hp.kinds,
        exp.ratios,
        exp.ple.map(|p| p.layer),
        t.layouts()
            .unwrap()
            .iter()
            .map(|(_, l)| l.file_len())
            .sum::<u64>(),
        dplan
            .layouts()
            .unwrap()
            .iter()
            .map(|(_, l)| l.file_len())
            .sum::<u64>(),
    );
}

/// Contract 10: the recorded card budget makes the written file's plan hold
/// half of the experts on every layer the card loads, as the gates plan it.
#[test]
#[ignore = "needs the box and the Qwen3.8 file (just gate-fixture)"]
fn hw_fixture_q38_budget() {
    let src = q38_source();
    let spec = q38::spec();
    let CardBudget::Planned(planner) = spec.card_budget else {
        panic!("the Qwen3.8 spec plans its card budget");
    };
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let recorded = p.card_budget;
    let mut kvs = p.target.kvs.clone();
    assert_eq!(*kv(&mut kvs, KEY_CARD_BUDGET), Value::U64(recorded));

    let guard = dir("q38budget");
    let fx = Split::open(header_only(&p.target, &guard.0)).unwrap();
    // The plan the gates make of a file (`gate_qwen4exp_e2e`'s `open_at`).
    let inputs = PlanInputs::describe(&fx).unwrap();
    let layers = inputs.spec.layers.len();
    let experts = inputs.model.experts;
    assert_eq!((layers, experts), (4, 512));
    // A layer whose stacks no card kernel loads keeps all its experts on the
    // host; the engine lists them, and the plan is held to that list.
    let host_only: Vec<usize> = inputs.host_only().iter().map(|h| h.layer).collect();
    let machine = machine_for_experts(
        RTX_3090,
        layers,
        UBATCH_PLANNED.min(q38::BUDGET_CTX),
        Experts::Card,
    );
    let at = |budget: Option<u64>| -> Vec<u64> {
        let levers = PlanLevers {
            card_budget_bytes: budget,
        };
        inputs
            .plan_with_slots(&machine, q38::BUDGET_CTX, &levers, Experts::Card, 1)
            .unwrap()
            .n_l
            .clone()
    };
    let want = |n: u64| -> Vec<u64> {
        (0..layers)
            .map(|l| if host_only.contains(&l) { 0 } else { n })
            .collect()
    };
    assert_eq!(
        at(None),
        want(experts),
        "the budgetless plan holds every expert it can"
    );
    assert_eq!(
        at(Some(recorded)),
        want(experts / 2),
        "the plan under {recorded}"
    );
    assert!(
        recorded < RTX_3090.usable_bytes(),
        "a budget {recorded} that does not bind a card of {}",
        RTX_3090.usable_bytes()
    );
    fixture::check_budget(&planner, &fx, recorded).unwrap();

    let mut bad: Vec<String> = Vec::new();
    for (what, budget) in [
        ("64 MiB below", recorded - (64 << 20)),
        ("1 GiB above", recorded + (1 << 30)),
    ] {
        match fixture::check_budget(&planner, &fx, budget) {
            Err(e @ FixtureError::Budget(_)) => println!("q38 budget: {what}: {e}"),
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(c) => bad.push(format!("{what}: holds {:?}", c.per_layer)),
        }
    }

    // `verify` re-plans the written file under the budget its header records.
    let mut edited = p.target.clone();
    *kv(&mut edited.kvs, KEY_CARD_BUDGET) = Value::U64(recorded - (64 << 20));
    let g2 = dir("q38budget-edited");
    let fx2 = Split::open(header_only(&edited, &g2.0)).unwrap();
    match fixture::verify(&spec, &fx2, &src, None, &mut |_, _| {}) {
        Err(e @ FixtureError::Budget(_)) if e.to_string().contains("plan under the budget") => {
            println!(
                "q38 budget: verify of a header recording {}: {e}",
                recorded - (64 << 20)
            );
        }
        other => bad.push(format!("verify of an edited budget: {other:?}")),
    }
    // The control: the recorded header passes that check and meets its holes.
    match fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}) {
        Err(FixtureError::Budget(m)) => bad.push(format!("verify of the recorded budget: {m}")),
        Err(e) => println!("q38 budget: control, the recorded header passes the budget check: {e}"),
        Ok(_) => bad.push("a file of holes verified".into()),
    }
    println!(
        "q38 budget: recorded {recorded} B ({} MiB); layers {layers}; host-only {host_only:?}; \
         unbounded {:?}; recorded {:?}",
        recorded / (1 << 20),
        at(None),
        at(Some(recorded)),
    );
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// Contract 11: the traps ik's reader falls into silently are refused by
/// name, in the source, in the written file and in the draft.
#[test]
#[ignore = "needs the box and the Qwen3.8 file and MTP draft (just gate-fixture)"]
fn hw_fixture_q38_refusals() {
    let src = q38_source();
    let draft = q38_draft();
    let spec = q38::spec();
    let guard = dir("q38refusals");
    let d = &guard.0;
    let mut bad: Vec<String> = Vec::new();

    // The control: the real header rewritten as a one-file set plans, to the
    // real plan's tensors, so each refusal below is its edit's.
    let real = fixture::plan(&spec, &src, Some(&draft), &spec.options()).unwrap();
    let names =
        |p: &FilePlan| -> Vec<String> { p.tensors.iter().map(|t| t.name.clone()).collect() };
    let same = edited_header(&src, d, "same.gguf", q38_keep, |_| {});
    match fixture::plan(&spec, &same, None, &spec.options()) {
        Ok(p) if names(&p.target) == names(&real.target) => {
            println!(
                "q38 refusals: the control plans {} tensors",
                p.target.tensors.len()
            );
        }
        Ok(_) => bad.push("the control plans other tensors".into()),
        Err(e) => bad.push(format!("the control: {e}")),
    }
    std::fs::remove_file(d.join("same.gguf")).unwrap();

    let source_rows: [(&str, Edit, &[&str]); 3] = [
        (
            "source ratios one short",
            |kvs| pop_item(kvs, RATIOS),
            &[RATIOS, "has 47 values for 48 layers"],
        ),
        (
            "source ple.layers missing",
            |kvs| drop_key(kvs, PLE_LAYERS),
            &[PLE_LAYERS, "is absent"],
        ),
        (
            "source interval missing",
            |kvs| drop_key(kvs, INTERVAL),
            &[INTERVAL, "is absent from the source"],
        ),
    ];
    for (what, edit, want) in source_rows {
        let file = format!("{}.gguf", what.replace(' ', "-"));
        let edited = edited_header(&src, d, &file, q38_keep, edit);
        match fixture::plan(&spec, &edited, None, &spec.options()) {
            Err(e) if want.iter().all(|w| e.to_string().contains(w)) => {
                println!("q38 refusals: {what}: {e}");
            }
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(_) => bad.push(format!("{what}: planned")),
        }
        std::fs::remove_file(d.join(&file)).unwrap();
    }

    // The draft: the MTP layer's pool read from an array one short of the
    // draft's layers is the main layers' last, which ik takes silently.
    let short = edited_header(
        &draft,
        d,
        "mtp-short.gguf",
        |_| true,
        |kvs| pop_item(kvs, RATIOS),
    );
    match fixture::plan(&spec, &src, Some(&short), &spec.options()) {
        Err(e) if e.to_string().contains("has 48 values, the main layers'") => {
            println!("q38 refusals: draft ratios one short: {e}");
        }
        Err(e) => bad.push(format!("draft ratios one short: refused as {e}")),
        Ok(_) => bad.push("draft ratios one short: planned".into()),
    }

    // The written file, from the real plan's header with one edit.
    let fixture_rows: [(&str, Edit, &[&str]); 2] = [
        (
            "fixture ratios one short",
            |kvs| pop_item(kvs, RATIOS),
            &[RATIOS, "has 3 values for 4 layers"],
        ),
        (
            "fixture ple.layers missing",
            |kvs| drop_key(kvs, PLE_LAYERS),
            &["per_layer_token_embd.weight", "carries none there"],
        ),
    ];
    for (what, edit, want) in fixture_rows {
        let mut plan = real.target.clone();
        edit(&mut plan.kvs);
        let g = dir(&format!("q38refusals-{}", what.replace(' ', "-")));
        let fx = Split::open(header_only(&plan, &g.0)).unwrap();
        match fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}) {
            Err(e) if want.iter().all(|w| e.to_string().contains(w)) => {
                println!("q38 refusals: {what}: {e}");
            }
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(_) => bad.push(format!("{what}: verified")),
        }
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

// ------------------------------------------------------------------ GLM-5.3-Flash

fn glm_source() -> Split {
    Split::open(glm::DEFAULT_MODEL)
        .unwrap_or_else(|e| panic!("open the GLM-5.3-Flash file {}: {e}", glm::DEFAULT_MODEL))
}

const GLM_HC: &str = "glm5next.hyper_connection.count";
const GLM_EXPERTS: &str = "glm5next.expert_count";
const GLM_KPOOL: &str = "glm5next.attention.indexer.kpool";
const GLM_NEXTN: &str = "glm5next.nextn_predict_layers";
const GLM_DENSE: &str = "glm5next.leading_dense_block_count";
const GLM_FF: &str = "glm5next.expert_feed_forward_length";

/// Contract 16: the plan of the GLM-5.3-Flash fixture, and the engine's reader
/// reading it back: seven layers from source layers 0, 1, 4, 7, 8, 11 and 45,
/// kinds [KDA, KDA, KDA, latent, KDA, latent, latent], a dense prefix of two,
/// the NextN layer believed, the routed ff 512 on the three stacks and the
/// shared expert's own width untouched, every (role, type) pair of the real
/// file's routed stacks kept.
#[test]
#[ignore = "needs the box and the GLM-5.3-Flash file (just gate-fixture)"]
fn hw_fixture_glm_plan() {
    let src = glm_source();
    let spec = glm::spec();
    let p = fixture::plan(&spec, &src, None, &spec.options()).expect("the plan of the real file");
    let t = &p.target;
    let get = |k: &str| -> Value {
        t.kvs
            .iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("no {k}"))
            .1
            .clone()
    };
    assert_eq!(
        unsigned_items(&get(KEY_SOURCE_LAYERS)),
        [0, 1, 4, 7, 8, 11, 45]
    );
    assert_eq!(get("glm5next.block_count").as_unsigned(), Some(7));
    assert_eq!(get(GLM_DENSE).as_unsigned(), Some(2));
    assert_eq!(get(GLM_NEXTN).as_unsigned(), Some(1));
    assert_eq!(get(GLM_FF).as_unsigned(), Some(glm::FIXTURE_FF));
    let shared = "glm5next.expert_shared_feed_forward_length";
    assert_eq!(
        get(shared).as_unsigned(),
        src.arch_get_u64("expert_shared_feed_forward_length"),
        "the shared expert keeps its own width"
    );

    // Each tensor is its source's, renamed through the map, in the source's
    // type, at the source's dims but the routed stacks' ff.
    let mut pairs: Vec<(String, GgmlType)> = Vec::new();
    for pt in &t.tensors {
        let (_, info) = src
            .find(&pt.source)
            .unwrap_or_else(|| panic!("no source tensor {}", pt.source));
        assert_eq!(pt.ty, info.ty, "{}", pt.name);
        let mut dims = info.dims.clone();
        match leaf_of(&pt.name) {
            "ffn_gate_exps.weight" | "ffn_up_exps.weight" => dims[1] = glm::FIXTURE_FF,
            "ffn_down_exps.weight" => dims[0] = glm::FIXTURE_FF,
            _ => {}
        }
        assert_eq!(pt.dims, dims, "{}", pt.name);
        if let Some(f) = pt.layer {
            assert!(pt.name.starts_with(&format!("blk.{f}.")), "{}", pt.name);
            let l = glm::LAYER_MAP[f];
            assert!(pt.source.starts_with(&format!("blk.{l}.")), "{}", pt.source);
        }
        if leaf_of(&pt.name).starts_with("ffn_") && leaf_of(&pt.name).contains("_exps.") {
            pairs.push((leaf_of(&pt.name).to_string(), pt.ty));
        }
    }
    pairs.sort_by_key(|(n, ty)| (n.clone(), ty.to_string()));
    pairs.dedup();
    let pairs: Vec<(&str, String)> = pairs
        .iter()
        .map(|(n, ty)| (n.as_str(), ty.to_string()))
        .collect();
    assert_eq!(
        pairs,
        [
            ("ffn_down_exps.weight", GgmlType::Q5_K.to_string()),
            ("ffn_down_exps.weight", GgmlType::Q6_K.to_string()),
            ("ffn_gate_exps.weight", GgmlType::Q4_K.to_string()),
            ("ffn_gate_exps.weight", GgmlType::Q5_K.to_string()),
            ("ffn_up_exps.weight", GgmlType::Q4_K.to_string()),
            ("ffn_up_exps.weight", GgmlType::Q5_K.to_string()),
        ],
        "every (role, type) pair of the real file's routed stacks"
    );

    // `ssm_a` is the folded `−e^A_log`, `ssm_dt.bias` its companion: a
    // positive `ssm_a` would grow the state every step.
    let decays: Vec<f32> = t
        .tensors
        .iter()
        .filter(|pt| leaf_of(&pt.name) == "ssm_a")
        .map(|pt| match pt.rule {
            Rule::Const { value } => value,
            ref other => panic!("{}: {}, not a constant", pt.name, other.describe()),
        })
        .collect();
    assert_eq!(decays.len(), 4, "the four KDA layers' ssm_a");
    assert!(decays.iter().all(|&v| v < 0.0), "ssm_a {decays:?}");

    let guard = dir("glmplan");
    let first = header_only(t, &guard.0);
    let fx = Split::open(&first).unwrap();
    let hp = glm::check_kinds(&spec, &fx, &src).unwrap();
    assert_eq!((hp.n_layer, hp.n_trunk, hp.dense_lead), (7, 6, 2));
    assert_eq!(
        hp.kinds,
        [
            GlmKind::Kda,
            GlmKind::Kda,
            GlmKind::Kda,
            GlmKind::Latent,
            GlmKind::Kda,
            GlmKind::Latent,
            GlmKind::Latent
        ]
    );
    assert_eq!(
        (hp.expert_ff, hp.shared_ff, hp.n_expert, hp.n_used),
        (512, 2048, 288, 8)
    );
    let inputs = GlmInputs::describe(&fx).unwrap();
    assert!(
        inputs.unimplemented().is_empty(),
        "the engine runs the fixture: {:?}",
        inputs.unimplemented()
    );
    NextnInputs::read(&inputs).expect("the NextN layer plans");

    let layouts = t.layouts().unwrap();
    let total: u64 = layouts.iter().map(|(_, l)| l.file_len()).sum();
    let spans = t.spanning_layers();
    assert!(
        layouts.len() >= 2 && !spans.is_empty(),
        "a layer spans a shard boundary"
    );
    println!(
        "glm plan: {} layers kinds {:?} dense {}; total {total} B ({:.2} GB; 8.57 GB [derived from the header dump]), {} shards, layers {spans:?} span",
        hp.n_layer,
        hp.kinds,
        hp.dense_lead,
        total as f64 / 1e9,
        layouts.len()
    );
}

/// Contract 17: the recorded card budget makes the plan of the written file
/// with its NextN layer loaded, at the e2e gate's context, hold half of the
/// experts (144 of 288) on every card-eligible layer (f2 to f4) and none on the
/// dense layers and the Q6_K-down layer, as `verify` re-plans it. The condition
/// table is printed: the budget is the NextN plan at [`glm::BUDGET_CTX`].
#[test]
#[ignore = "needs the box and the GLM-5.3-Flash file (just gate-fixture)"]
fn hw_fixture_glm_budget() {
    let src = glm_source();
    let spec = glm::spec();
    let CardBudget::Planned(planner) = spec.card_budget else {
        panic!("the GLM spec plans its card budget");
    };
    let p = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let recorded = p.card_budget;
    let mut kvs = p.target.kvs.clone();
    assert_eq!(*kv(&mut kvs, KEY_CARD_BUDGET), Value::U64(recorded));

    let guard = dir("glmbudget");
    let fx = Split::open(header_only(&p.target, &guard.0)).unwrap();
    let inputs = GlmInputs::describe(&fx).unwrap();
    let nextn = NextnInputs::read(&inputs).unwrap();
    let (layers, experts) = (inputs.model.layers, inputs.model.experts);
    assert_eq!((layers, experts), (6, 288));
    let machine = workstation::plan_gate(layers);
    let with_nextn = |budget: Option<u64>, ctx: u64| -> Result<Vec<u64>, String> {
        let levers = PlanLevers {
            card_budget_bytes: budget,
        };
        inputs
            .plan_nextn(&machine, ctx, &levers, &nextn)
            .map(|plan| plan.plan.n_l.clone())
            .map_err(|e| e.to_string())
    };
    let plain = |budget: Option<u64>, ctx: u64| -> Result<Vec<u64>, String> {
        let levers = PlanLevers {
            card_budget_bytes: budget,
        };
        inputs
            .plan(&machine, ctx, &levers)
            .map(|plan| plan.n_l.clone())
            .map_err(|e| e.to_string())
    };
    let ctx = glm::BUDGET_CTX;
    // Dense f0 and f1, and f5 (a Q6_K down no card kernel runs), hold none on
    // the card; f2 to f4 are the card-eligible layers.
    let want = |n: u64| vec![0, 0, n, n, n, 0];
    assert_eq!(
        with_nextn(None, ctx).unwrap(),
        want(experts),
        "the budgetless plan holds every expert it can"
    );
    assert_eq!(
        with_nextn(Some(recorded), ctx).unwrap(),
        want(experts / 2),
        "the plan under {recorded}"
    );
    assert!(
        recorded < RTX_3090.usable_bytes(),
        "a budget {recorded} that does not bind a card of {}",
        RTX_3090.usable_bytes()
    );
    fixture::check_budget(&planner, &fx, recorded).unwrap();

    let mut bad: Vec<String> = Vec::new();
    for (what, budget) in [
        ("64 MiB below", recorded - (64 << 20)),
        ("64 MiB above", recorded + (64 << 20)),
    ] {
        match fixture::check_budget(&planner, &fx, budget) {
            Err(e @ FixtureError::Budget(_)) => println!("glm budget: {what}: {e}"),
            Err(e) => bad.push(format!("{what}: refused as {e}")),
            Ok(c) => bad.push(format!("{what}: holds {:?}", c.per_layer)),
        }
    }

    // `verify` re-plans the written file under the budget its header records.
    let mut edited = p.target.clone();
    *kv(&mut edited.kvs, KEY_CARD_BUDGET) = Value::U64(recorded - (64 << 20));
    let g2 = dir("glmbudget-edited");
    let fx2 = Split::open(header_only(&edited, &g2.0)).unwrap();
    match fixture::verify(&spec, &fx2, &src, None, &mut |_, _| {}) {
        Err(e @ FixtureError::Budget(_)) if e.to_string().contains("plan under the budget") => {
            println!(
                "glm budget: verify of a header recording {}: {e}",
                recorded - (64 << 20)
            );
        }
        other => bad.push(format!("verify of an edited budget: {other:?}")),
    }
    // The control: the recorded header passes that check and meets its holes.
    match fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}) {
        Err(FixtureError::Budget(m)) => bad.push(format!("verify of the recorded budget: {m}")),
        Err(e) => println!("glm budget: control, the recorded header passes the budget check: {e}"),
        Ok(_) => bad.push("a file of holes verified".into()),
    }

    println!(
        "glm budget: recorded {recorded} B ({} MiB) at ctx {ctx} with the NextN layer, plan_gate({layers}); \
         budgetless {:?}; recorded {:?}",
        recorded / (1 << 20),
        with_nextn(None, ctx).unwrap(),
        with_nextn(Some(recorded), ctx).unwrap(),
    );
    for c in [1024, 2048, 2051, ctx] {
        println!(
            "glm budget: condition ctx {c}: NextN loaded, under the recorded budget {:?}, budgetless {:?}; \
             no NextN, under it {:?}, budgetless {:?}",
            with_nextn(Some(recorded), c),
            with_nextn(None, c),
            plain(Some(recorded), c),
            plain(None, c)
        );
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// `got` is a refusal whose text holds every one of `want`, else `bad` gains
/// why not.
fn refused_with<T: std::fmt::Debug>(
    bad: &mut Vec<String>,
    what: &str,
    got: Result<T, FixtureError>,
    want: &[&str],
) {
    match got {
        Err(e) if want.iter().all(|w| e.to_string().contains(w)) => {
            println!("glm refusals: {what}: {e}");
        }
        Err(e) => bad.push(format!("{what}: refused as {e}")),
        Ok(v) => bad.push(format!("{what}: accepted, {v:?}")),
    }
}

/// Contract 18: what the GLM reader and the kernels take silently, refused by
/// name: a map that drops the NextN layer (the reader then reads the file as a
/// trunk of every layer), a dense layer after a routed one, a constant of the
/// kernels changed in the source or in the written file, a NextN count, a dense
/// count and an ff the written file has wrong, and an ff off the block grid.
#[test]
#[ignore = "needs the box and the GLM-5.3-Flash file (just gate-fixture)"]
fn hw_fixture_glm_refusals() {
    let src = glm_source();
    let spec = glm::spec();
    let guard = dir("glmrefusals");
    let d = &guard.0;
    let mut bad: Vec<String> = Vec::new();

    // The control: the real header rewritten as a one-file set plans, to the
    // real plan's tensors, so each refusal below is its edit's.
    let real = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    let names =
        |p: &FilePlan| -> Vec<String> { p.tensors.iter().map(|t| t.name.clone()).collect() };
    // Every tensor of the source is declared: the reader checks every layer's.
    let same = edited_header(&src, d, "same.gguf", |_| true, |_| {});
    match fixture::plan(&spec, &same, None, &spec.options()) {
        Ok(p) if names(&p.target) == names(&real.target) => {
            println!(
                "glm refusals: the control plans {} tensors",
                p.target.tensors.len()
            );
        }
        Ok(_) => bad.push("the control plans other tensors".into()),
        Err(e) => bad.push(format!("the control: {e}")),
    }
    std::fs::remove_file(d.join("same.gguf")).unwrap();

    // Maps the spec's layers can be replaced by.
    for (what, layers, want) in [
        (
            "a map that drops the NextN layer",
            vec![0, 1, 4, 7, 8, 11],
            vec!["does not end in the source's NextN layer 45"],
        ),
        (
            "a map that ends in a trunk layer",
            vec![0, 1, 4, 7, 11, 8, 45, 12],
            vec!["does not end in the source's NextN layer 45"],
        ),
        (
            "a dense layer after a routed one",
            vec![0, 4, 1, 7, 8, 11, 45],
            vec!["after a routed one"],
        ),
    ] {
        let s = FixtureSpec {
            layers,
            ..glm::spec()
        };
        refused_with(
            &mut bad,
            what,
            fixture::plan(&s, &src, None, &s.options()),
            &want,
        );
    }

    // An ff the K-quant blocks do not tile.
    let ragged = FixtureSpec {
        ff: Some(500),
        ..glm::spec()
    };
    refused_with(
        &mut bad,
        "ff 500",
        fixture::plan(&ragged, &src, None, &ragged.options()),
        &["not whole blocks of 256"],
    );

    // The source, with a constant of the kernels or a key the file's reader
    // needs changed.
    let source_rows: [(&str, Edit, &[&str]); 5] = [
        (
            "source hyper_connection.count 8",
            |kvs| set_int(kvs, GLM_HC, 8),
            &[GLM_HC, "is 8 in the source", "built for 4"],
        ),
        (
            "source expert_count 256",
            |kvs| set_int(kvs, GLM_EXPERTS, 256),
            &[GLM_EXPERTS, "is 256 in the source", "built for 288"],
        ),
        (
            "source indexer.kpool 8",
            |kvs| set_int(kvs, GLM_KPOOL, 8),
            &[GLM_KPOOL, "is 8 in the source", "built for 4"],
        ),
        (
            "source nextn_predict_layers 0",
            |kvs| set_int(kvs, GLM_NEXTN, 0),
            &["blk.45.hc_attn_fn.weight", "is not in the file"],
        ),
        (
            "source leading_dense_block_count missing",
            |kvs| drop_key(kvs, GLM_DENSE),
            &["blk.0.ffn_gate_inp.weight", "is not in the file"],
        ),
    ];
    for (what, edit, want) in source_rows {
        let file = format!("{}.gguf", what.replace(' ', "-"));
        let edited = edited_header(&src, d, &file, |_| true, edit);
        refused_with(
            &mut bad,
            what,
            fixture::plan(&spec, &edited, None, &spec.options()),
            want,
        );
        std::fs::remove_file(d.join(&file)).unwrap();
    }

    // The written file, from the real plan's header with one edit.
    let fixture_rows: [(&str, Edit, &[&str]); 4] = [
        (
            "fixture indexer.kpool 8",
            |kvs| set_int(kvs, GLM_KPOOL, 8),
            &[GLM_KPOOL, "is 8 in the fixture", "built for 4"],
        ),
        (
            "fixture nextn_predict_layers 0",
            |kvs| set_int(kvs, GLM_NEXTN, 0),
            &["blk.6.hc_attn_fn.weight", "is not in the file"],
        ),
        (
            "fixture expert_feed_forward_length 2048",
            |kvs| set_int(kvs, GLM_FF, 2048),
            &["expert_ff", "2048, want 512"],
        ),
        (
            "fixture leading_dense_block_count 3",
            |kvs| set_int(kvs, GLM_DENSE, 3),
            &["blk.2.ffn_gate.weight", "is not in the file"],
        ),
    ];
    for (what, edit, want) in fixture_rows {
        let mut plan = real.target.clone();
        edit(&mut plan.kvs);
        let g = dir(&format!("glmrefusals-{}", what.replace(' ', "-")));
        let fx = Split::open(header_only(&plan, &g.0)).unwrap();
        refused_with(
            &mut bad,
            what,
            fixture::verify(&spec, &fx, &src, None, &mut |_, _| {}),
            want,
        );
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}
