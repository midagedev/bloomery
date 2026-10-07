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
//! 7. the Q5_1 and IQ4_NL rules under the V4.1 spec's window: a filled chunk
//!    dequantizes (`gguf::dequant_row`) to an RMS within ±10 % of 1/√K and
//!    every block holds its rule; a K whose `d` no code choice puts in the
//!    window is refused with `NoScale`;
//! 8. the window is the spec's: a wider one admits Q5_1 at K = 640, which
//!    V4.1's refuses, and a window that leaves the normal f16 values is
//!    refused by name.
//!
//! Files go under this crate's `CARGO_TARGET_TMPDIR` and are removed.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::fixture::{self as v41, LAYER_MAP};
use model::fileio;
use model::fixture::{
    self, CHUNK_TARGET, Family, FilePlan, FixtureError, FixtureSpec, KEY_CARD_BUDGET, KEY_SEED,
    KEY_SOURCE_LAYERS, KEY_SOURCE_SHA256, KEY_VERSION, KeyRule, Options, Plan, PlannedTensor, Rule,
    Sample, Tables, Window, rule_for,
};
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
                assert_eq!(x.dims, s.dims, "{}", x.name);
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
    assert_eq!(
        get(KEY_CARD_BUDGET).as_u64(),
        Some(v41::DEFAULT_CARD_BUDGET)
    );
    let dp = p.draft.as_ref().unwrap();
    let dtheirs: HashMap<&str, &Value> = d.iter_kv().collect();
    let dchanged: Vec<&str> = dp
        .kvs
        .iter()
        .filter(|(k, v)| dtheirs.get(k.as_str()).is_some_and(|s| *s != v))
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(dchanged, ["dflash.target_layers"]);
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
        "plan: total {total} B = headers {headers} + layers {layers} + globals {glob} ({:.2} GB; the design's 60.9 GB [derived]); draft {draft_len} B ({:.2} GB; the design's ≈ 8.5 GB); {} shards, layers {spans:?} span",
        total as f64 / 1e9,
        draft_len as f64 / 1e9,
        layouts.len()
    );

    // The engine reads the header-only files as the source's kinds.
    let guard = dir("plan");
    let dd = &guard.0;
    let first = header_only(t, dd);
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
    let opts = Options {
        tensors: Some(subset.map(String::from).to_vec()),
        shard_bytes: u64::MAX,
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

/// A draft's metadata: the dflash architecture, `target_layers`, and `extra`.
fn draft_kvs(target_layers: std::ops::Range<u32>, extra: &[(&str, Value)]) -> Vec<(String, Value)> {
    let mut kvs = vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(model::arch::DFLASH.into()),
        ),
        (
            format!("{}.target_layers", model::arch::DFLASH),
            Value::Array(target_layers.map(Value::U32).collect()),
        ),
    ];
    kvs.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
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

    // An ff override: the V4.1 spec names no ff key, so it has no rule for one.
    let narrow = FixtureSpec {
        ff: Some(512),
        ..v41::spec()
    };
    match fixture::plan(&narrow, &src, None, &narrow.options()) {
        Err(e @ FixtureError::Metadata { .. }) if e.to_string().contains("has no ff key") => {
            println!("refusals: ff 512: {e}");
        }
        Err(e) => bad.push(format!("ff 512: refused as {e}")),
        Ok(_) => bad.push("ff 512: planned".to_string()),
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
    println!("refusals: all 9 refused by name");
}

/// The second family's rules: every generic key rule, a constant for its
/// norms, an ff axis on each expert stack, and a table.
struct Toy;

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
        matches!(leaf, "norm.weight" | "output_norm.weight").then_some(1.0)
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

static TOY: Toy = Toy;

/// A four-layer source of the second family: one file, its tensors zero.
fn toy_source(path: &Path) -> Split {
    let arr = |v: &[u32]| Value::Array(v.iter().map(|&x| Value::U32(x)).collect());
    let mut tensors: Vec<(&str, Vec<u64>, GgmlType)> = vec![
        ("tab.weight", vec![64, 1000], GgmlType::BF16),
        ("output_norm.weight", vec![64], GgmlType::F32),
    ];
    let names: Vec<[String; 4]> = (0..4)
        .map(|l| {
            ["norm", "attn", "ffn_up_exps", "ffn_down_exps"].map(|n| format!("blk.{l}.{n}.weight"))
        })
        .collect();
    for n in &names {
        tensors.push((&n[0], vec![64], GgmlType::F32));
        tensors.push((&n[1], vec![4096, 8], GgmlType::Q8_0));
        tensors.push((&n[2], vec![64, 512, 4], GgmlType::BF16));
        tensors.push((&n[3], vec![512, 64, 4], GgmlType::BF16));
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
        card_budget: 1 << 30,
        window: Window::new(1.0 / 16384.0, 1.0 / 64.0).unwrap(),
        ff: Some(256),
        default_source: String::new,
        draft: None,
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
    assert_eq!((v.tensors, v.subset, none.is_none()), (10, false, true));
    println!(
        "second family: wrote {} tensors, {} bytes; verify {} tensors, {} blocks",
        s.tensors, s.file_bytes, v.tensors, v.blocks
    );

    let mut bad: Vec<String> = Vec::new();
    let refusals: [(&str, FixtureSpec, bool, &str); 4] = [
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

/// Contract 7: the Q5_1 and IQ4_NL rules fill at σ under V4.1's window. Every
/// case runs; the failures are listed together.
#[test]
fn fixture_new_rules_fill_at_sigma() {
    let window = v41::spec().window;
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

/// Contract 7: a σ whose `d` no code choice puts in V4.1's window is refused
/// with `NoScale`, at both ends of each rule's range. Every case runs.
#[test]
fn fixture_new_rules_refuse_out_of_window() {
    let window = v41::spec().window;
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

/// Contract 8: the second family's window, [2^-14, 2^-6], is wider than
/// V4.1's and admits Q5_1 at K = 640 (`d = σ/√85.25 = 2^-7.87`), which V4.1's
/// refuses; its blocks fill at σ and hold under the wide window, and V4.1's
/// window refuses them by name.
#[test]
fn fixture_wider_window_admits_q5_1_at_k640() {
    let (ty, k) = (GgmlType::Q5_1, 640);
    let narrow = v41::spec().window;
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
