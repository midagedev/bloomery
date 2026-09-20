//! Oracle gate for stage 1, round 1-1. Both tests are `hw_` (model file +
//! box) and `#[ignore]`d: `cargo nextest` is not installed on the box
//! (checked 2026-09-19), so run them with
//! `cargo test -p gguf -- --ignored --nocapture` via `tools/box.sh`, after
//! `bash tools/ref/build-dequant.sh && $BLOOMERY_DATA/bin/dequant_ref`.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

use gguf::{GgmlType, Gguf, dequant_row};

const MODEL: &str = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

fn data_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("BLOOMERY_DATA").expect("BLOOMERY_DATA must be set (run via tools/box.sh)"),
    )
}

/// Every tensor in the file resolves to a name, a dims vector, a type and an
/// in-bounds slice; the type set and per-type counts equal the oracle's
/// manifest; names are unique; the stage-1 hyperparameter keys are present.
#[test]
#[ignore = "hw: needs the model on the box plus the oracle dump in $BLOOMERY_DATA"]
fn hw_coverage() {
    let g = Gguf::open(MODEL).expect("open model");

    let arch = g.architecture().expect("general.architecture");
    println!(
        "arch={arch} kv={} tensors={} data_base={} alignment={}",
        g.kv_count(),
        g.tensor_count(),
        g.data_base(),
        g.alignment()
    );
    for key in [
        "block_count",
        "expert_count",
        "expert_used_count",
        "embedding_length",
        "attention.head_count",
    ] {
        let v = g
            .arch_get_u64(key)
            .unwrap_or_else(|| panic!("{arch}.{key} missing"));
        println!("  {arch}.{key} = {v}");
    }

    let mut counts: BTreeMap<GgmlType, usize> = BTreeMap::new();
    let mut names: HashSet<&str> = HashSet::new();
    for i in 0..g.tensor_count() {
        let t = g
            .tensor(i)
            .unwrap_or_else(|| panic!("tensor slot {i} missing"));
        assert!(!t.name.is_empty(), "tensor {i}: empty name");
        assert!(!t.dims.is_empty(), "tensor {}: no dims", t.name);
        assert!(
            t.ty.blck_size().is_some(),
            "tensor {}: type {} outside the size table",
            t.name,
            t.ty
        );
        // In-bounds check: data() is checked slicing over the mmap.
        let bytes = g
            .data(t)
            .unwrap_or_else(|e| panic!("tensor {}: {e}", t.name));
        assert_eq!(
            bytes.len() as u64,
            t.nbytes,
            "tensor {}: short slice",
            t.name
        );
        assert!(
            names.insert(t.name.as_str()),
            "duplicate tensor name {}",
            t.name
        );
        *counts.entry(t.ty).or_default() += 1;
    }

    println!("type counts from the loader:");
    let mut total = 0usize;
    for (ty, n) in &counts {
        println!("  {ty} ({}) x{n}", ty.as_u32());
        total += n;
    }
    assert_eq!(total, g.tensor_count());

    // The oracle's manifest: "<type_num> <type_name> <count>" per type.
    let manifest =
        fs::read_to_string(data_dir().join("ref/manifest.txt")).expect("read ref/manifest.txt");
    let oracle: BTreeMap<u32, (String, usize)> = manifest
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split_whitespace();
            let num: u32 = it.next().unwrap().parse().unwrap();
            let name = it.next().unwrap().to_string();
            let count: usize = it.next().unwrap().parse().unwrap();
            (num, (name, count))
        })
        .collect();

    let mine: BTreeMap<u32, usize> = counts.iter().map(|(ty, n)| (ty.as_u32(), *n)).collect();
    assert_eq!(
        mine,
        oracle
            .iter()
            .map(|(&k, &(_, c))| (k, c))
            .collect::<BTreeMap<_, _>>(),
        "type set / per-type counts disagree with the ggml oracle"
    );
    // And the names the oracle used match our enum's names (dump filename
    // compatibility for the dequant test).
    for (ty, n) in &counts {
        let (oname, ocount) = oracle.get(&ty.as_u32()).unwrap();
        assert_eq!(
            ty.name().unwrap(),
            oname.as_str(),
            "type-name mismatch for {ty}"
        );
        assert_eq!(n, ocount, "per-type count mismatch for {ty}");
    }
}

/// Per type: re-dequantize the oracle's rows from our own mmap slice and
/// compare against ggml's to_float. Gate: max |diff| <= 1e-6 for every type.
#[test]
#[ignore = "hw: needs the model on the box plus the oracle dump in $BLOOMERY_DATA"]
fn hw_dequant_matches_ggml() {
    let g = Gguf::open(MODEL).expect("open model");
    let dir = data_dir().join("ref");
    let manifest = fs::read_to_string(dir.join("manifest.txt")).expect("read ref/manifest.txt");

    let mut worst: Vec<(String, f64)> = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let type_num: u32 = it.next().unwrap().parse().unwrap();
        let tname = it.next().unwrap();
        let _count: usize = it.next().unwrap().parse().unwrap();
        let ty = GgmlType::from_u32(type_num);

        // meta: tensor=..., type=..., dims=..., rows=..., rowlen=...
        let meta = fs::read_to_string(dir.join(format!("{tname}.meta")))
            .unwrap_or_else(|e| panic!("read {tname}.meta: {e}"));
        let mut tensor_name = String::new();
        let mut rows = 0usize;
        let mut rowlen = 0usize;
        for l in meta.lines() {
            if let Some(v) = l.strip_prefix("tensor=") {
                tensor_name = v.to_string();
            } else if let Some(v) = l.strip_prefix("rows=") {
                rows = v.parse().unwrap();
            } else if let Some(v) = l.strip_prefix("rowlen=") {
                rowlen = v.parse().unwrap();
            }
        }
        assert!(!tensor_name.is_empty(), "{tname}.meta: no tensor name");

        let t = g
            .find(&tensor_name)
            .unwrap_or_else(|| panic!("tensor {tensor_name} not found by our loader"));
        assert_eq!(t.ty, ty, "tensor {tensor_name}: type disagreement");
        assert_eq!(t.dims[0] as usize, rowlen, "tensor {tensor_name}: ne[0]");
        let bytes = g
            .data(t)
            .unwrap_or_else(|e| panic!("tensor {tensor_name}: {e}"));

        let blck = ty.blck_size().unwrap() as usize;
        let tsz = ty.type_size().unwrap() as usize;
        let nvals = rows * rowlen;
        let need = (nvals / blck) * tsz;
        let mut mine = vec![0.0f32; nvals];
        dequant_row(ty, &bytes[..need], &mut mine).unwrap_or_else(|e| panic!("{tname}: {e}"));

        let raw = fs::read(dir.join(format!("{tname}.raw")))
            .unwrap_or_else(|e| panic!("read {tname}.raw: {e}"));
        assert_eq!(raw.len(), nvals * 4, "{tname}.raw size");
        let mut maxd = 0.0f64;
        for (m, r) in mine.iter().zip(raw.as_chunks::<4>().0) {
            let r = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
            maxd = maxd.max((*m - r).abs() as f64);
        }
        println!(
            "{tname:6} tensor={tensor_name} rows={rows} rowlen={rowlen} max_abs_diff={maxd:.3e}"
        );
        assert!(
            maxd <= 1e-6,
            "{tname}: max abs diff {maxd:.3e} exceeds the 1e-6 gate"
        );
        worst.push((tname.to_string(), maxd));
    }
    println!("per-type max |diff| (gate 1e-6):");
    for (n, d) in &worst {
        println!("  {n:6} {d:.3e}");
    }
}

/// The resident backings hold the file's bytes at the file's offsets: every
/// tensor slice equals the mapped one, so no gate downstream can tell them apart.
#[test]
#[ignore = "hw: needs the model on the box"]
fn hw_resident_copy_is_the_file() {
    let mapped = Gguf::open(MODEL).expect("open model");
    for huge in [false, true] {
        let res = Gguf::open_backed(MODEL, gguf::Weights::Resident { huge }).expect("resident");
        assert_eq!(res.tensor_count(), mapped.tensor_count());
        for i in 0..mapped.tensor_count() {
            let t = mapped.tensor(i).unwrap();
            let a = mapped.data(t).unwrap();
            let b = res.data(res.tensor(i).unwrap()).unwrap();
            assert!(
                a == b,
                "tensor {} differs in the resident copy (huge={huge})",
                t.name
            );
        }
    }
}
