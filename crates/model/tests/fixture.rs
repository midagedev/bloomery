//! The V4.1 gate fixture's generator (`model::arch::deepseek41::fixture`, bin `v41fixture`),
//! against the real file's header (`gguf::v41::model`) and the real DSpark
//! draft's (`$BLOOMERY_DSPARK_MODEL`). Four contracts:
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
//!    `general.alignment` the writer would not take, and a flag its verb does
//!    not take are refused by name.
//!
//! Files go under this crate's `CARGO_TARGET_TMPDIR` and are removed.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use gguf::write::{Layout, Writer};
use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::fixture::{
    self, FilePlan, FixtureError, KEY_CARD_BUDGET, KEY_SEED, KEY_SOURCE_LAYERS, KEY_SOURCE_SHA256,
    KEY_VERSION, LAYER_MAP, Options, Plan, PlannedTensor, Rule, Sample,
};
use model::fileio;
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
    fixture::plan(src, Some(d), &Options::default()).expect("the plan of the real file")
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
    let site_rows: Vec<u64> = p.engram_rows.iter().map(|&(fx, _)| fx).collect();
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
    let all: Vec<u64> = (fixture::ENGRAM_PRIME_FLOOR + 1..)
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
        Some(fixture::DEFAULT_CARD_BUDGET)
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
    let hp = fixture::check_kinds(&fx, &src).unwrap();
    let dx = Split::open(&draft_file).unwrap();
    let dhp = fixture::check_draft(&dx, &fx, true).unwrap();
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
                        let got = fixture::check_units(t, &buf, r.start / unit)
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
            ds.iter().all(|&b| fixture::d_in_window(b)),
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
    let o = Command::new(env!("CARGO_BIN_EXE_v41fixture"))
        .args(args)
        .output()
        .unwrap();
    let (out, err) = (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    );
    let tail: Vec<&str> = out.lines().filter(|l| !l.contains(" type=")).collect();
    println!(
        "$ v41fixture {}\n{}\n{err}rc={:?} wall={:.2}s",
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
    let opts = Options {
        tensors: Some(subset.map(String::from).to_vec()),
        shard_bytes: u64::MAX,
        ..Options::default()
    };
    let p = fixture::plan(&src, None, &opts).unwrap();
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
        stdout.contains("v41fixture: generate done tensors=9 "),
        "no summary"
    );
    for n in subset {
        assert!(
            stdout.contains(&format!("v41fixture: tensor {n} ")),
            "no line for {n}"
        );
    }
    let peak = children_peak_kib();
    let first = out.join(format!("{}-00001-of-00002.gguf", fixture::STEM));
    let fx = Split::open(&first).unwrap();
    assert_eq!(fx.shard_count(), 2);
    let first_s = first.to_str().unwrap();
    let (rc, stdout, _) = run(&["verify", first_s, "--source", &src_path]);
    assert_eq!(rc, Some(0));
    assert!(
        stdout.contains("v41fixture: verify done tensors=9 ") && stdout.contains("subset=true")
    );
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
        stdout.contains("v41fixture: generate done tensors=6 "),
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
        [format!("{}-00001-of-00001.gguf", fixture::STEM)],
        "a directory reader of the target's shards must not meet the draft"
    );
    let dfile = out.join(fixture::DRAFT_FILE);
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
    let first = out.join(format!("{}-00001-of-00001.gguf", fixture::STEM));
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
    let mut bad: Vec<String> = Vec::new();

    // A draft that reads the source's last ten layers: more than the fixture's nine.
    let n = LAYER_MAP.len() as u32 + 1;
    let wide = draft_header(&d.join("wide.gguf"), &draft_kvs(n_layer - n..n_layer, &[]));
    match fixture::plan(&src, Some(&wide), &Options::default()) {
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
    match fixture::plan(&src, Some(&align), &Options::default()) {
        Err(e @ FixtureError::Alignment { .. }) => println!("refusals: alignment 48: {e}"),
        Err(e) => bad.push(format!("alignment 48: refused as {e}")),
        Ok(_) => bad.push("alignment 48: planned".to_string()),
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
    println!("refusals: all 6 refused by name");
}
