//! Oracle gate for stage 1, round 1-1. The tests are `hw_` (model files +
//! box) and `#[ignore]`d: `cargo nextest` is not installed on the box
//! (checked 2026-09-19), so `just gate-1-1` runs them through `tools/box.sh`
//! after `dequant_ref` has dumped V2-Lite into `$BLOOMERY_DATA/ref` and the
//! f32, bf16 and q8_0 tensors of V4.1's first shard into
//! `$BLOOMERY_DATA/ref-v41`, and its synthetic q2_K and i-quant rows into
//! `$BLOOMERY_DATA/ref-synth`.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

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
/// compare against ggml's to_float bit for bit. Every one of these decodes is
/// exact in f32 (quant.rs's module doc), so a difference of any size is a bug.
/// Every type is compared before the verdict, so a red names each type that
/// differs and how many values.
#[test]
#[ignore = "hw: needs the model on the box plus the oracle dump in $BLOOMERY_DATA"]
fn hw_dequant_matches_ggml() {
    let g = Gguf::open(MODEL).expect("open model");
    let dir = data_dir().join("ref");
    let manifest = fs::read_to_string(dir.join("manifest.txt")).expect("read ref/manifest.txt");

    let mut worst: Vec<(String, f64, usize)> = Vec::new();
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
        let (mut maxd, mut differ) = (0.0f64, 0usize);
        for (m, r) in mine.iter().zip(raw.as_chunks::<4>().0) {
            let r = f32::from_le_bytes(*r);
            maxd = maxd.max((*m - r).abs() as f64);
            differ += usize::from(m.to_bits() != r.to_bits());
        }
        println!(
            "{tname:6} tensor={tensor_name} rows={rows} rowlen={rowlen} values={nvals} bit_mismatches={differ} max_abs_diff={maxd:.3e}"
        );
        worst.push((tname.to_string(), maxd, differ));
    }
    // PIN(2026-09-24): bit identity; was max |diff| <= 1e-6, which a one-ulp error passed.
    println!("per-type bit mismatches (gate 0):");
    for (n, d, k) in &worst {
        println!("  {n:6} {k} (max |diff| {d:.3e})");
    }
    let red: Vec<_> = worst.iter().filter(|w| w.2 != 0).collect();
    assert!(red.is_empty(), "values differ from ggml: {red:?}");
}

/// `<dir>/<tname>.meta`'s tensor name, row count and row length.
fn read_meta(dir: &Path, tname: &str) -> (String, usize, usize) {
    let meta = fs::read_to_string(dir.join(format!("{tname}.meta")))
        .unwrap_or_else(|e| panic!("read {tname}.meta: {e}"));
    let field = |key: &str| {
        meta.lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or_else(|| panic!("{tname}.meta: no {key}"))
            .to_string()
    };
    let rows = field("rows=").parse().unwrap();
    (field("tensor="), rows, field("rowlen=").parse().unwrap())
}

/// V4.1's first shard opens strictly, and each type it holds that this engine
/// dequantizes on the host side reproduces ggml's `to_float` bit for bit:
/// every one of these conversions is exact, so any difference is a bug, not
/// rounding. The mixed file's are f32, bf16 and q8_0; the public file's are
/// every type of its first shard (its engram and embedding rows, its gains
/// and scales are all decoded on the host), from the dump
/// `ref-v41` + [`gguf::v41::set_suffix_of`]. Each requested type must be in
/// the dump (a type the dump lacks fails here instead of passing with nothing
/// compared), with ggml's tensor count for it equal to the loader's.
#[test]
#[ignore = "hw: needs V4.1's first shard on the box plus the oracle dump in $BLOOMERY_DATA/ref-v41[_plain]"]
fn hw_dequant_matches_ggml_v41() {
    let path = gguf::v41::model();
    let g = Gguf::open(&path).unwrap_or_else(|e| panic!("strict open of {path}: {e}"));
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for t in g.iter_tensors() {
        *counts.entry(t.ty.name().unwrap()).or_default() += 1;
    }
    let set = format!("ref-v41{}", gguf::v41::set_suffix_of(&path));
    let dir = data_dir().join(&set);
    let manifest = fs::read_to_string(dir.join("manifest.txt"))
        .unwrap_or_else(|e| panic!("read {set}/manifest.txt: {e}"));
    let oracle: BTreeMap<&str, (u32, usize)> = manifest
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let num: u32 = it.next().unwrap().parse().unwrap();
            let name = it.next().unwrap();
            let count: usize = it.next().unwrap().parse().unwrap();
            (name, (num, count))
        })
        .collect();

    let want: Vec<&str> = if gguf::v41::set_suffix_of(&path).is_empty() {
        vec!["f32", "bf16", "q8_0"]
    } else {
        counts.keys().copied().collect()
    };
    for tname in want {
        let &(type_num, ggml_count) = oracle
            .get(tname)
            .unwrap_or_else(|| panic!("{tname} is not in {set}/manifest.txt"));
        let ty = GgmlType::from_u32(type_num);
        assert_eq!(ty.name(), Some(tname), "type {type_num}: name mismatch");
        assert_eq!(
            counts.get(tname).copied(),
            Some(ggml_count),
            "{tname}: tensor count, loader vs ggml"
        );
        let (tensor_name, rows, rowlen) = read_meta(&dir, tname);
        let t = g
            .find(&tensor_name)
            .unwrap_or_else(|| panic!("tensor {tensor_name} not found by our loader"));
        assert_eq!(t.ty, ty, "tensor {tensor_name}: type disagreement");
        assert_eq!(t.dims[0] as usize, rowlen, "tensor {tensor_name}: ne[0]");
        let nvals = rows * rowlen;
        let need = nvals / ty.blck_size().unwrap() as usize * ty.type_size().unwrap() as usize;
        let bytes = g
            .data(t)
            .unwrap_or_else(|e| panic!("tensor {tensor_name}: {e}"));
        let mut mine = vec![0.0f32; nvals];
        dequant_row(ty, &bytes[..need], &mut mine).unwrap_or_else(|e| panic!("{tname}: {e}"));

        let raw = fs::read(dir.join(format!("{tname}.raw")))
            .unwrap_or_else(|e| panic!("read {tname}.raw: {e}"));
        assert_eq!(raw.len(), nvals * 4, "{tname}.raw size");
        let differ: Vec<usize> = mine
            .iter()
            .zip(raw.as_chunks::<4>().0)
            .enumerate()
            .filter(|(_, (m, r))| m.to_bits() != u32::from_le_bytes(**r))
            .map(|(i, _)| i)
            .collect();
        println!(
            "{tname:6} tensors={ggml_count} tensor={tensor_name} rows={rows} rowlen={rowlen} values={nvals} bit_mismatches={}",
            differ.len()
        );
        assert!(
            differ.is_empty(),
            "{tname}: {} of {nvals} values differ from ggml, first at {}",
            differ.len(),
            differ[0]
        );
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

/// The types no model file on the box holds — q2_K, iq2_xs, iq3_xxs, iq4_xs —
/// from `dequant_ref --synthetic`: ggml-quantized rows plus rows of random
/// codes, decoded by our `dequant_row` and compared with ggml's `to_float` bit
/// for bit. Every product in these decodes is exact (quant.rs's
/// `iq_products_are_exact`), so any difference is a bug, not rounding. The
/// dump must hold exactly these four types; the codebook coverage the rows
/// reached is printed, and a set that misses a grid entry is refused.
#[test]
#[ignore = "hw: needs the synthetic oracle dump in $BLOOMERY_DATA/ref-synth"]
fn hw_dequant_matches_ggml_synthetic() {
    let dir = data_dir().join("ref-synth");
    let manifest =
        fs::read_to_string(dir.join("manifest.txt")).expect("read ref-synth/manifest.txt");
    let mut seen: Vec<&str> = Vec::new();
    let mut red: Vec<String> = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let type_num: u32 = it.next().unwrap().parse().unwrap();
        let tname = it.next().unwrap();
        let ty = GgmlType::from_u32(type_num);
        assert_eq!(ty.name(), Some(tname), "type {type_num}: name mismatch");
        seen.push(tname);

        let meta = fs::read_to_string(dir.join(format!("{tname}.meta")))
            .unwrap_or_else(|e| panic!("read {tname}.meta: {e}"));
        let field = |key: &str| -> usize {
            meta.lines()
                .find_map(|l| l.strip_prefix(key))
                .unwrap_or_else(|| panic!("{tname}.meta: no {key}"))
                .parse()
                .unwrap()
        };
        let (rows, rowlen, quantized) =
            (field("rows="), field("rowlen="), field("quantized_rows="));
        let blck = ty.blck_size().unwrap() as usize;
        let row_bytes = rowlen / blck * ty.type_size().unwrap() as usize;
        let blocks = fs::read(dir.join(format!("{tname}.blocks")))
            .unwrap_or_else(|e| panic!("read {tname}.blocks: {e}"));
        assert_eq!(blocks.len(), rows * row_bytes, "{tname}.blocks size");
        let raw = fs::read(dir.join(format!("{tname}.raw")))
            .unwrap_or_else(|e| panic!("read {tname}.raw: {e}"));
        assert_eq!(raw.len(), rows * rowlen * 4, "{tname}.raw size");

        let mut mine = vec![0.0f32; rows * rowlen];
        dequant_row(ty, &blocks, &mut mine).unwrap_or_else(|e| panic!("{tname}: {e}"));
        let differ: Vec<usize> = mine
            .iter()
            .zip(raw.as_chunks::<4>().0)
            .enumerate()
            .filter(|(_, (m, r))| m.to_bits() != u32::from_le_bytes(**r))
            .map(|(i, _)| i)
            .collect();
        let in_quantized = differ.iter().filter(|&&i| i < quantized * rowlen).count();
        let coverage = codebook_coverage(ty, &blocks);
        println!(
            "{tname:8} rows={rows} (quantized {quantized}) rowlen={rowlen} values={} bit_mismatches={} (quantized rows {in_quantized}) {coverage}",
            mine.len(),
            differ.len()
        );
        if let Some(&i) = differ.first() {
            red.push(format!(
                "{tname}: {} of {} values differ from ggml, first at {i} (ours {:e}, ggml {:e})",
                differ.len(),
                mine.len(),
                mine[i],
                f32::from_le_bytes(raw.as_chunks::<4>().0[i])
            ));
        }
    }
    assert!(red.is_empty(), "{}", red.join("\n"));
    seen.sort_unstable();
    assert_eq!(
        seen,
        ["iq2_xs", "iq3_xxs", "iq4_xs", "q2_K"],
        "ref-synth type set"
    );
}

/// How many of a codebook's entries the blocks index, as `grid a/b signs c/d`;
/// panics when a grid or sign entry is never reached (the random rows are
/// there so that every one is). Types without a codebook report nothing.
fn codebook_coverage(ty: GgmlType, blocks: &[u8]) -> String {
    let (grid, signs): (Vec<usize>, Vec<usize>) = match ty {
        GgmlType::IQ2_XS => blocks
            .as_chunks::<74>()
            .0
            .iter()
            .flat_map(|b| {
                b[2..66]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|q| u16::from_le_bytes(*q))
            })
            .map(|q| (usize::from(q & 511), usize::from(q >> 9)))
            .unzip(),
        GgmlType::IQ3_XXS => {
            let blks = blocks.as_chunks::<98>().0;
            let grid = blks
                .iter()
                .flat_map(|b| b[2..66].iter().map(|&i| usize::from(i)))
                .collect();
            let signs = blks
                .iter()
                .flat_map(|b| {
                    b[66..98]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| u32::from_le_bytes(*w))
                })
                .flat_map(|w| (0..4).map(move |l| ((w >> (7 * l)) & 127) as usize))
                .collect();
            (grid, signs)
        }
        _ => return String::new(),
    };
    let n_grid = if ty == GgmlType::IQ2_XS { 512 } else { 256 };
    let distinct = |v: &[usize]| v.iter().collect::<HashSet<_>>().len();
    let (g, s) = (distinct(&grid), distinct(&signs));
    assert_eq!((g, s), (n_grid, 128), "{ty}: codebook entries reached");
    format!("grid {g}/{n_grid} signs {s}/128")
}
