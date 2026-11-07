//! `gate_kimi_k3_imatrix` — the importance matrix of Kimi K3's own prompt walk against `llama-imatrix`'s, on
//! the kimi-k3 file of the profile (`BLOOMERY_REF_MODEL`; the fixture under the fixture tier).
//!
//!     gate_kimi_k3_imatrix --dir <dir> [--arms a,bp]
//!
//! One text ([`TEXT_FILE`], this repository's AGENTS.md) is cut into [`CHUNKS`] chunks of [`CTX`] ids. `llama-imatrix`
//! of the K3 quantization tree (`K3Q_TREE`, the tree `llama-quantize` of `gate_iq_quant` comes from) reads it
//! on the CPU with `-c 512 -b 512 -ub 512 --chunks 2`, and `generate_kimi_k3 --imatrix` (beside this binary)
//! walks the same ids on each arm of `--arms` (`a`: the whole-card load, the prompt stepped; `bp`: the placed
//! load, `--walk pass`; the recipe's `BLOOMERY_IMAT_ARMS`). Each arm's file is held to llama's:
//!
//! - `names`: both open through the quantizer's reader (`quantize::imatrix::Imatrix`), and ours names every
//!   weight llama's does but those of [`NOT_COLLECTED`], each with its reason, and no weight llama's lacks;
//!   a name of that list ours does collect is red too (the list cannot go stale).
//! - `counts`: a weight every row reaches (every dense linear, the router) has llama's counts exactly. A routed
//!   stack's counts total exactly the rows times the picks a row makes, and each expert's count is within
//!   [`COUNT_FLIP`] of llama's.
//! - `in_sum2`: every column of a dense weight within [`COL_BAND`] of its layer's, the sums of every weight
//!   within [`TOTAL_BAND`] in total, and the middle column of a routed stack within [`ROUTED_MEDIAN`].
//! - `quantize`: `bloomery-quantize` over the fixture (requantized to MXFP4 stacks by the tree's
//!   `llama-quantize`, as `gate_iq_quant`'s end-to-end clause does) with the first arm's imatrix writes a
//!   set that opens, holding both IQ1_S and IQ2_XXS stacks.
//!
//! The bands. Both files hold `Σ x²` over the same `N = CHUNKS · CTX` rows (the same ids: the first layer's
//! inputs agree to 5e-7), in row order, f32 each. For one column the sum order costs at most `N · 2⁻²⁴`
//! relative, every term being a square: [`SUM_ORDER`], 6e-5, and no band here is that small, because the inputs
//! `x` are not the same bits. Ours are the card's activations, llama's the CPU's: each linear quantizes its
//! input rows to its own q8 form before the dot, ours as the card's kernels do and llama's as ggml's CPU
//! kernels do, so every layer's output differs by that rounding (a relative ~1e-3 a linear, the kernels' own
//! band) and the residual carries the difference on: the fixture's dense columns differ by at most 5.7e-3 in
//! layer 0 and 1.8e-1 in layer 3 (the median 1e-3 and 4e-2), a squared value doubling it. A routed stack's
//! counts follow the router's top-k of scores computed on those activations, so the picks nearest the cut
//! flip: 3 to 7 % of an expert's rows here, and a flipped row moves its column sums with it.

use bloomery_gpu_gates::{GateError, beside, die_with_parent, ref_model_path, verdict};
use gguf::{GgmlType, Gguf};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The chunk length both engines run.
const CTX: usize = 512;

/// The chunks both engines run.
const CHUNKS: usize = 2;

/// The text: AGENTS.md of the tree this gate was built from, longer than two chunks.
const TEXT_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../AGENTS.md");

/// The weights the engine's walk does not collect though `llama-imatrix` does, by stem, each with why: the
/// card's attention kernel reads the per-head `q_nope` (for `attn_k_b`) and the attention-weighted latent (for
/// `attn_v_b`) inside one fused launch, so neither is a buffer the layer's code holds.
const NOT_COLLECTED: [(&str, &str); 2] = [
    (
        "attn_k_b.weight",
        "its input is the per-head q_nope the fused latent attention reads inside its launch",
    ),
    (
        "attn_v_b.weight",
        "its input is the attention-weighted latent the fused latent attention reads inside its launch",
    ),
];

/// A sum of `N` squares in f32, one after another: `N · 2⁻²⁴` relative at most.
const SUM_ORDER: f64 = (CHUNKS * CTX) as f64 / 16_777_216.0;

/// The widest relative difference of one column of a dense weight's sums by the layer of the weight, 1.3 times
/// the worst of that layer's weights in the first measurement on the fixture (5.7e-3 / 2.2e-2 for layer 0's
/// attention output and dense ffn, 3.3e-2 layer 1, 1.3e-1 layer 2, 1.8e-1 layer 3): the activations' difference
/// grows with the layers it passes through (module doc), so the band does.
/// PIN(2026-11-07): measured on the fixture over this text at ctx 512, 2 chunks; a layer past the table takes
/// its last entry.
const COL_BAND: [f64; 4] = [0.03, 0.045, 0.17, 0.24];

/// The widest relative difference of a weight's total sum: the columns' differences average over the
/// columns (the worst weight measured 3.7e-3, a routed stack, the picks that flip moving rows between its
/// experts), so a collection that took the wrong rows, the wrong weight or a scale is far outside.
/// PIN(2026-11-07): measured on the fixture, 2.7 times the worst.
const TOTAL_BAND: f64 = 0.01;

/// The widest relative difference of a routed stack's middle column (its columns hold the rows of the experts
/// the router picked, which differ with the picks that flip): the measured worst median was 8.8e-2.
/// PIN(2026-11-07): measured on the fixture, 1.4 times the worst.
const ROUTED_MEDIAN: f64 = 0.12;

/// The widest relative difference of one routed expert's count (the picks that flip, module doc): 3.4 to 7.1 %
/// measured on the fixture's three routed layers.
/// PIN(2026-11-07): measured on the fixture, 1.4 times the worst.
const COUNT_FLIP: f64 = 0.10;

/// The layer of `blk.N.…`, 0 for a name outside the blocks.
fn layer(name: &str) -> usize {
    name.strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// Whether `name` is a routed stack: its columns are one a routed expert.
fn routed(name: &str) -> bool {
    kind(name).ends_with("_exps.weight")
}

/// One weight's collected values: `ne0`, its matrices' count, the sums and the counts.
struct Entry {
    ne0: usize,
    sums: Vec<f32>,
    counts: Vec<f32>,
}

/// Every `<stem>.in_sum2` of `path` with its `.counts`, by stem.
fn read(path: &Path) -> Result<BTreeMap<String, Entry>, GateError> {
    let g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let floats = |name: &str| -> Result<Vec<f32>, GateError> {
        let t = g
            .find(name)
            .ok_or_else(|| format!("{}: no {name}", path.display()))?;
        if t.ty != GgmlType::F32 {
            return Err(format!("{}: {name} is {}, not F32", path.display(), t.ty).into());
        }
        let bytes = g
            .data(t)
            .map_err(|e| format!("{}: {name}: {e}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let mut out = BTreeMap::new();
    let stems: Vec<String> = g
        .iter_tensors()
        .filter_map(|t| t.name.strip_suffix(".in_sum2").map(str::to_owned))
        .collect();
    for stem in stems {
        let sums = floats(&format!("{stem}.in_sum2"))?;
        let counts = floats(&format!("{stem}.counts"))?;
        let ne0 = sums.len() / counts.len().max(1);
        out.insert(stem, Entry { ne0, sums, counts });
    }
    Ok(out)
}

/// The stem without its `blk.N.` prefix.
fn kind(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, k)| k)
}

fn exempt(name: &str) -> Option<&'static str> {
    NOT_COLLECTED
        .iter()
        .find(|(k, _)| kind(name) == *k)
        .map(|(_, why)| *why)
}

/// `cmd` run to its end: success, and what it printed.
fn run(cmd: &mut Command, what: &str) -> Result<(bool, String, String), GateError> {
    die_with_parent(cmd);
    let o = cmd.output().map_err(|e| format!("{what}: {e}"))?;
    Ok((
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    ))
}

fn tail(s: &str) -> String {
    s.lines().rev().take(8).collect::<Vec<_>>().join(" | ")
}

/// Clauses `names`, `counts` and `in_sum2` of `ours` against `llama`: whether all three held.
fn compare(arm: &str, ours: &Path, llama: &Path) -> Result<bool, GateError> {
    let open = |p: &Path| quantize::imatrix::Imatrix::open(&p.to_string_lossy());
    let (ri, oi) = (open(llama), open(ours));
    let opened = ri.is_ok() && oi.is_ok();
    let (a, b) = (read(llama)?, read(ours)?);
    let missing: Vec<&String> = a
        .keys()
        .filter(|k| !b.contains_key(*k) && exempt(k).is_none())
        .collect();
    let extra: Vec<&String> = b.keys().filter(|k| !a.contains_key(*k)).collect();
    let stale: Vec<&String> = b.keys().filter(|k| exempt(k).is_some()).collect();
    let skipped = a
        .keys()
        .filter(|k| !b.contains_key(*k) && exempt(k).is_some())
        .count();
    let names = opened && missing.is_empty() && extra.is_empty() && stale.is_empty();
    println!(
        "{} names [{arm}]: both open through the quantizer's reader ({}); ours {} weights, llama's {}; missing {:?}, extra {:?}, listed but collected {:?}; {skipped} of the not-collected list left out",
        verdict(names),
        opened,
        b.len(),
        a.len(),
        missing,
        extra,
        stale
    );
    let mut bad_counts = Vec::new();
    let mut flips = 0.0f64;
    let mut bad_sums = Vec::new();
    let mut worst = (0.0f64, String::new());
    let mut worst_total = (0.0f64, String::new());
    let mut worst_median = (0.0f64, String::new());
    for (name, x) in &a {
        let Some(y) = b.get(name) else { continue };
        let total = |e: &Entry| e.counts.iter().map(|&c| f64::from(c)).sum::<f64>();
        if x.ne0 != y.ne0 || x.counts.len() != y.counts.len() {
            bad_counts.push(name.clone());
            continue;
        }
        if routed(name) {
            let worst_flip = x
                .counts
                .iter()
                .zip(&y.counts)
                .map(|(&u, &v)| f64::from((u - v).abs()) / f64::from(u.max(v)).max(1.0))
                .fold(0.0, f64::max);
            flips = flips.max(worst_flip);
            if total(x) != total(y) || worst_flip > COUNT_FLIP {
                bad_counts.push(name.clone());
            }
        } else if x.counts != y.counts {
            bad_counts.push(name.clone());
        }
        let sum = |v: &[f32]| v.iter().map(|&c| f64::from(c)).sum::<f64>();
        let drift = (sum(&x.sums) / sum(&y.sums) - 1.0).abs();
        if drift > worst_total.0 {
            worst_total = (drift, name.clone());
        }
        let mut rels: Vec<f64> = Vec::with_capacity(x.sums.len());
        for (i, (&u, &v)) in x.sums.iter().zip(&y.sums).enumerate() {
            let (u, v) = (f64::from(u), f64::from(v));
            let scale = u.max(v);
            let rel = if scale > 0.0 {
                (u - v).abs() / scale
            } else {
                0.0
            };
            rels.push(rel);
            if !routed(name) && rel > worst.0 {
                worst = (rel, format!("{name}[{i}] llama {u:e} ours {v:e}"));
            }
        }
        let band = COL_BAND[layer(name).min(COL_BAND.len() - 1)];
        let outside = if routed(name) {
            rels.sort_by(f64::total_cmp);
            let median = rels[rels.len() / 2];
            if median > worst_median.0 {
                worst_median = (median, name.clone());
            }
            median > ROUTED_MEDIAN
        } else {
            rels.iter().any(|&r| r > band)
        };
        if outside || drift > TOTAL_BAND {
            bad_sums.push(name.clone());
        }
    }
    let counts = bad_counts.is_empty();
    println!(
        "{} counts [{arm}]: dense counts equal llama's exactly, a routed stack's total exactly and each expert's within {COUNT_FLIP} (worst {flips:.3}) ({} weights compared); outside: {:?}",
        verdict(counts),
        a.keys().filter(|k| b.contains_key(*k)).count(),
        bad_counts
    );
    let sums = bad_sums.is_empty();
    println!(
        "{} in_sum2 [{arm}]: dense columns within {COL_BAND:?} by layer (sum order alone {SUM_ORDER:.1e}; worst {:.3e} at {}), totals within {TOTAL_BAND} (worst {:.2e} at {}), routed middle column within {ROUTED_MEDIAN} (worst {:.2e} at {}); outside: {} weights {:?}",
        verdict(sums),
        worst.0,
        worst.1,
        worst_total.0,
        worst_total.1,
        worst_median.0,
        worst_median.1,
        bad_sums.len(),
        &bad_sums[..bad_sums.len().min(6)]
    );
    Ok(names && counts && sums)
}

/// Clause `quantize`: `bloomery-quantize` over the fixture with `imatrix`.
fn quantize_clause(
    dir: &Path,
    tree: &Path,
    fixture: &Path,
    imatrix: &Path,
) -> Result<bool, GateError> {
    let lq = tree.join("build/bin/llama-quantize");
    let tool = beside("bloomery-quantize")?;
    let work = dir.join("quantize");
    std::fs::create_dir_all(&work)?;
    let rules_src = work.join("rules-src.txt");
    std::fs::write(&rules_src, "ffn_(down|gate|up)_exps\\.weight=mxfp4\n")?;
    let src = work.join("src.gguf");
    let flags = [
        "--output-tensor-type",
        "q6_k",
        "--token-embedding-type",
        "q4_k",
    ];
    let (ok, _, err) = run(
        Command::new(&lq)
            .env("CUDA_VISIBLE_DEVICES", "")
            .args(["--allow-requantize", "--tensor-type-file"])
            .arg(&rules_src)
            .args(flags)
            .arg(fixture)
            .arg(&src)
            .args(["Q4_K_M", "16"]),
        "llama-quantize",
    )?;
    if !ok {
        println!(
            "{} quantize: llama-quantize could not make the MXFP4 stand-in: {}",
            verdict(false),
            tail(&err)
        );
        return Ok(false);
    }
    let rules = work.join("rules.txt");
    std::fs::write(
        &rules,
        concat!(
            "ffn_(gate|up)_exps\\.weight=IQ2_XXS\n",
            "ffn_down_exps\\.weight=IQ1_S\n",
            "\\.attn_k_b\\.weight$=q8_0\n",
            "\\.ssm_f_b\\.weight$=q8_0\n",
            "\\.ssm_beta\\.weight$=f32\n",
            "\\.attn_v_b\\.weight$=q6_k\n",
            "\\.(attn_gate|attn_k|attn_kv_a_mqa|attn_output|attn_q|attn_q_a|attn_q_b|attn_v|ffn_down|ffn_gate|ffn_up|",
            "ffn_down_shexp|ffn_gate_shexp|ffn_up_shexp|ffn_routed_down|ffn_routed_up|ssm_f_a|ssm_g)\\.weight$=q4_k\n",
        ),
    )?;
    let out = work.join("ours.gguf");
    let (ok, _, err) = run(
        Command::new(&tool)
            .env_remove("BLOOMERY_CARD_BUDGET")
            .env_remove("BLOOMERY_CARD_MARGIN")
            .args(["--allow-requantize", "--imatrix"])
            .arg(imatrix)
            .arg("--tensor-type-file")
            .arg(&rules)
            .args(flags)
            .arg(&src)
            .arg(&out)
            .arg("Q4_K_M"),
        "bloomery-quantize",
    )?;
    if !ok {
        println!(
            "{} quantize: bloomery-quantize over our imatrix ended nonzero: {}",
            verdict(false),
            tail(&err)
        );
        return Ok(false);
    }
    let g = Gguf::open(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let kinds = |ty: GgmlType| g.iter_tensors().filter(|t| t.ty == ty).count();
    let (n1, n2) = (kinds(GgmlType::IQ1_S), kinds(GgmlType::IQ2_XXS));
    let pass = n1 > 0 && n2 > 0;
    println!(
        "{} quantize: bloomery-quantize over the fixture with our imatrix writes a set that opens ({} tensors: {n1} IQ1_S, {n2} IQ2_XXS)",
        verdict(pass),
        g.tensor_count()
    );
    Ok(pass)
}

fn gate() -> Result<(), GateError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let dir = PathBuf::from(flag("--dir").ok_or("gate_kimi_k3_imatrix: --dir DIR is required")?);
    let arms = flag("--arms").unwrap_or_else(|| "a,bp".to_owned());
    std::fs::create_dir_all(&dir)?;
    let fixture = ref_model_path()?;
    let tree = PathBuf::from(std::env::var("K3Q_TREE").map_err(|_| {
        "K3Q_TREE names the K3.1 quantization tree (the recipe sets it from tools/ref/ref-paths.sh)"
    })?);
    let imat = tree.join("build/bin/llama-imatrix");
    if !imat.is_file() {
        return Err(format!(
            "no {} (just build-k3-quant-tree builds the tree)",
            imat.display()
        )
        .into());
    }
    let text = dir.join("text.txt");
    std::fs::copy(TEXT_FILE, &text).map_err(|e| format!("{TEXT_FILE}: {e}"))?;
    let llama = dir.join("llama.imatrix.gguf");
    let (ok, _, err) = run(
        Command::new(&imat)
            .env("CUDA_VISIBLE_DEVICES", "")
            .arg("-m")
            .arg(&fixture)
            .arg("-f")
            .arg(&text)
            .arg("-o")
            .arg(&llama)
            .args(["--output-format", "gguf", "-c", &CTX.to_string()])
            .args(["-b", &CTX.to_string(), "-ub", &CTX.to_string()])
            .args([
                "--chunks",
                &CHUNKS.to_string(),
                "--no-escape",
                "--no-ppl",
                "-ngl",
                "0",
            ]),
        "llama-imatrix",
    )?;
    if !ok {
        return Err(format!("llama-imatrix ended nonzero: {}", tail(&err)).into());
    }
    let mut all = true;
    let mut first = None;
    for arm in arms.split(',') {
        let (place, walk) = match arm {
            "a" => ("a", "steps"),
            "bp" => ("bp", "pass"),
            o => return Err(format!("--arms: {o} is a or bp").into()),
        };
        let ours = dir.join(format!("ours-{arm}.imatrix.gguf"));
        let (ok, out, err) = run(
            Command::new(beside("generate_kimi_k3")?)
                .arg("--imatrix")
                .arg(&ours)
                .arg("--text")
                .arg(&text)
                .args(["--ctx", &CTX.to_string(), "--chunks", &CHUNKS.to_string()])
                .args(["--place", place, "--walk", walk]),
            "generate_kimi_k3",
        )?;
        if !ok {
            println!(
                "{} run [{arm}]: generate_kimi_k3 --imatrix ended nonzero: {}",
                verdict(false),
                tail(&err)
            );
            all = false;
            continue;
        }
        for l in out.lines().filter(|l| l.starts_with("imatrix ")) {
            println!("  [{arm}] {l}");
        }
        all &= compare(arm, &ours, &llama)?;
        first.get_or_insert(ours);
    }
    let first = first.ok_or("no arm produced a file")?;
    all &= quantize_clause(&dir, &tree, &fixture, &first)?;
    if all {
        println!("gate_kimi_k3_imatrix: PASSED");
        Ok(())
    } else {
        Err("gate_kimi_k3_imatrix: FAILED".into())
    }
}

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_imatrix", gate())
}
