//! The V4.1 importance collection against `llama-imatrix`'s (`gate_deepseek41_step --imatrix DIR`), on the V4.1
//! fixture (`BLOOMERY_REF_MODEL`; the fixture tier only: ik reads no real V4.1 file whole).
//!
//! One text ([`TEXT_FILE`], this repository's AGENTS.md) is cut into [`CHUNKS`] chunks of [`CTX`] ids. ik's
//! `llama-imatrix` (the tree `IK` names, the V4.1 oracle) reads it on the CPU with `-c 512 -b 512 -ub 512
//! --chunks 2` into `DIR/ik.dat`, while the gate's own load walks the same ids (`shared/ds41_imatrix.rs`: the
//! prompt batch with the prompt triangle off, as `ppl_ds41 --imatrix` walks) into `DIR/ours-a.gguf` and, from a
//! reset collector, again into `DIR/ours-b.gguf`. ik writes its legacy form: per weight the mean of `x²` over the
//! rows that reached it, times its number of calls, so ours is held to it as `in_sum2 / counts * chunks`.
//!
//! - `names` (fixture oracle): ours opens through the quantizer's reader, and holds every weight ik's file does but
//!   those of [`NOT_COLLECTED`], each with its reason, and none ik's lacks but those of [`OURS_ONLY`]; a name of
//!   either list that is on the other side's side of the rule is red too (the lists cannot go stale).
//! - `counts` (self-consistency): a dense weight's count is the rows walked, `CHUNKS * CTX`; a routed gate or up
//!   stack's counts total the rows times the picks a row makes; a down's are the picks the card computed, none
//!   for a pick the host tier served (its combine is never an f32 row on the card), so at most its up's per
//!   expert; a gate weight's counts and sums are its up weight's, which sees the same rows and picks.
//! - `means` (fixture oracle): per dense weight, the median and the 99th percentile over its columns of the
//!   relative difference of a column's mean from ik's within [`COL_MEDIAN`] and [`COL_P99`] by layer; the total
//!   mean of every weight within [`TOTAL_BAND`] (a routed down's within [`DOWN_TOTAL`]); the middle column of a
//!   routed gate or up stack within [`ROUTED_MEDIAN`] and of a down within [`DOWN_MEDIAN`]; and ik's number of
//!   calls the chunks walked. A statistic over the columns, not their maximum: the widest of thousands of columns
//!   is a few low-energy columns whose rounding moves with the card, and a wrong buffer moves the whole of them.
//! - `repeat` (self-consistency): the second walk's file is the first's, byte for byte.
//!
//! The bands. Both sides hold `Σ x²` over the same rows. For one column the sum order costs at most
//! `N · 2⁻²⁴` relative ([`SUM_ORDER`]); no band here is that small because the inputs are not the same bits:
//! ours are the card's activations and ik's the CPU's, each linear quantizing its input rows to its own q8 form
//! before the dot, so every layer's output differs by that rounding and the residual carries it on, a squared
//! value doubling it. A routed stack's counts follow the router's top-k of scores computed on those activations,
//! so the picks nearest the cut flip, and a flipped row moves its column sums with it.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};

use bloomery_gpu::imatrix;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_deepseek41::body;
use bloomery_gpu_deepseek41::router::N_USED;
use bloomery_gpu_gates::tier::{self, Tag, Tier};
use bloomery_gpu_gates::{GateError, die_with_parent, imatrix_walk, verdict};
use gguf::{GgmlType, Gguf, Split};
use tokenizer::Tokenizer;

use crate::imx;

/// The ik tree whose `llama-imatrix` reads the fixture: the recipe's `. tools/ref/ref-paths.sh && export IK`.
const IK_ENV: &str = "IK";

/// The fixture-oracle clauses this module decides.
pub const FIXTURE_ORACLE: usize = 2;

/// The chunk length both engines run.
const CTX: usize = 512;

/// The chunks both engines run.
const CHUNKS: usize = 2;

/// The text: AGENTS.md of the tree this gate was built from, longer than two chunks.
const TEXT_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../AGENTS.md");

/// The weights ik's file holds that the walk does not collect, by stem, each with why.
const NOT_COLLECTED: [(&str, &str); 1] = [(
    "indexer.attn_k.weight",
    "its input is the compressor's pooled row, formed inside the chunk's compressor launch and never an f32 \
     buffer the layer holds a view of per token",
)];

/// The weights the walk collects that ik's file lacks, each with why: by stem, or by full name where the stem is
/// in ik's file at the other layers.
const OURS_ONLY: [(&str, &str); 6] = [
    (
        "blk.0.ffn_up_exps.weight",
        "ik's file holds no entry of the first block's routed up, though its graph runs the stack",
    ),
    (
        "blk.0.ffn_down_exps.weight",
        "ik's file holds no entry of the first block's routed down, though its graph runs the stack",
    ),
    (
        "ffn_gate_exps.weight",
        "ik fuses the routed gate and up into one op and names only the up weight; both see the same rows and picks",
    ),
    (
        "ffn_gate_shexp.weight",
        "ik fuses the shared gate and up into one op and names only the up weight; both see the same rows",
    ),
    (
        "indexer.proj.weight",
        "ik's collector holds no entry of it on this fixture, though its graph runs the weight",
    ),
    (
        "indexer.attn_q_b.weight",
        "ik's collector holds no entry of it on this fixture, though its graph runs the weight",
    ),
];

/// A sum of `N` squares in f32, one after another: `N · 2⁻²⁴` relative at most.
const SUM_ORDER: f64 = (CHUNKS * CTX) as f64 / 16_777_216.0;

/// The widest median, over a dense weight's columns, of the relative difference of a column's mean, by the layer
/// of the weight: the noise grows with the depth, as the rounding of each layer's output rides the residual into
/// the next.
/// PIN(2026-11-14): 1.3 times the largest of eight sittings' worst by layer, [0.005, 0.019, 0.035, 0.048, 0.052,
/// 0.059, 0.066, 0.073, 0.080] (2 chunks of 512; the V4.1 fixture and its three variant files, each on the 3090
/// and the A6000).
const COL_MEDIAN: [f64; 9] = [
    0.007, 0.025, 0.046, 0.063, 0.068, 0.077, 0.086, 0.095, 0.104,
];

/// The widest 99th percentile of the same, by layer.
/// PIN(2026-11-14): 1.3 times the largest of the same eight sittings' worst by layer, [0.022, 0.078, 0.144, 0.193,
/// 0.224, 0.228, 0.248, 0.265, 0.286].
const COL_P99: [f64; 9] = [
    0.029, 0.102, 0.188, 0.251, 0.292, 0.297, 0.323, 0.345, 0.372,
];

/// The widest relative difference of a weight's total mean.
/// PIN(2026-11-14): 2 times the largest of the same eight sittings' worst, 1.18e-2 at blk.7.attn_output_b.
const TOTAL_BAND: f64 = 0.024;

/// The widest relative difference of a routed down stack's total mean: ours holds the picks the card computed,
/// ik's all of them.
/// PIN(2026-11-14): 1.5 times the largest of the same eight sittings' worst, 4.91e-2 at blk.1.ffn_down_exps (the
/// IQ-glue variant on the 3090).
const DOWN_TOTAL: f64 = 0.075;

/// The widest relative difference of a routed gate or up stack's middle column.
/// PIN(2026-11-14): 1.3 times the largest of the same eight sittings' worst, 2.50e-1 at blk.8.ffn_up_exps.
const ROUTED_MEDIAN: f64 = 0.33;

/// The widest relative difference of a routed down stack's middle column.
/// PIN(2026-11-14): 1.3 times the largest of the same eight sittings' worst, 5.08e-1 at blk.8.ffn_down_exps.
const DOWN_MEDIAN: f64 = 0.66;

/// The layer of `blk.N.…`, 0 for a name outside the blocks.
fn layer(name: &str) -> usize {
    name.strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// The stem without its `blk.N.` prefix.
fn kind(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, k)| k)
}

/// Whether `name` is a routed stack: its columns are one a routed expert.
fn routed(name: &str) -> bool {
    kind(name).ends_with("_exps.weight")
}

/// One weight of our file: `ne0` columns a matrix, the sums and the counts (a count a matrix).
struct Ours {
    ne0: usize,
    sums: Vec<f32>,
    counts: Vec<f32>,
}

impl Ours {
    /// The mean of `x²` of column `i`, times `calls`: what ik's file holds, `None` where no row reached the column.
    fn mean(&self, i: usize, calls: f64) -> Option<f64> {
        let n = f64::from(self.counts[i / self.ne0]);
        (n > 0.0).then(|| f64::from(self.sums[i]) / n * calls)
    }
}

/// One weight of ik's file: its number of calls and the values (the mean of `x²` times the calls).
struct Ik {
    calls: i32,
    vals: Vec<f32>,
}

/// Every `<stem>.in_sum2` of `path` with its `.counts`, by stem.
fn read_ours(path: &Path) -> Result<BTreeMap<String, Ours>, GateError> {
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
    let stems: Vec<String> = g
        .iter_tensors()
        .filter_map(|t| t.name.strip_suffix(".in_sum2").map(str::to_owned))
        .collect();
    let mut out = BTreeMap::new();
    for stem in stems {
        let sums = floats(&format!("{stem}.in_sum2"))?;
        let counts = floats(&format!("{stem}.counts"))?;
        if counts.is_empty() || sums.len() % counts.len() != 0 {
            return Err(format!(
                "{}: {stem}: {} sums over {} counts do not split evenly",
                path.display(),
                sums.len(),
                counts.len()
            )
            .into());
        }
        out.insert(
            stem,
            Ours {
                ne0: sums.len() / counts.len(),
                sums,
                counts,
            },
        );
    }
    Ok(out)
}

/// ik's legacy imatrix file: `i32` entries, each `i32` name length, the name, `i32` calls, `i32` values, the
/// `f32` values; then `i32` last call and the input file's name. The stem loses the ` (reshaped)` ik adds to a
/// grouped weight.
fn read_ik(path: &Path) -> Result<BTreeMap<String, Ik>, GateError> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize, what: &str| -> Result<&[u8], GateError> {
        let b = bytes
            .get(*at..*at + n)
            .ok_or_else(|| format!("{}: ends inside {what}", path.display()))?;
        *at += n;
        Ok(b)
    };
    let int = |at: &mut usize, what: &str| -> Result<i32, GateError> {
        let b = take(at, 4, what)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let entries = int(&mut at, "the entry count")?;
    let mut out = BTreeMap::new();
    for _ in 0..entries {
        let len = usize::try_from(int(&mut at, "a name length")?)?;
        let name = String::from_utf8(take(&mut at, len, "a name")?.to_vec())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let calls = int(&mut at, "a call count")?;
        let n = usize::try_from(int(&mut at, "a value count")?)?;
        let vals = take(&mut at, 4 * n, &name)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let stem = name.strip_suffix(" (reshaped)").unwrap_or(&name).to_owned();
        out.insert(stem, Ik { calls, vals });
    }
    Ok(out)
}

fn tail(path: &Path) -> String {
    let text = std::fs::read(path).unwrap_or_default();
    String::from_utf8_lossy(&text)
        .lines()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .join(" | ")
}

fn exempt(table: &[(&'static str, &'static str)], name: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(k, _)| *k == name || kind(name) == *k)
        .map(|(_, w)| *w)
}

/// Clause `names` of `ours` against `ik`: whether it held.
fn names(ours: &BTreeMap<String, Ours>, ik: &BTreeMap<String, Ik>, opened: bool) -> bool {
    let missing: Vec<&String> = ik
        .keys()
        .filter(|k| !ours.contains_key(*k) && exempt(&NOT_COLLECTED, k).is_none())
        .collect();
    let extra: Vec<&String> = ours
        .keys()
        .filter(|k| !ik.contains_key(*k) && exempt(&OURS_ONLY, k).is_none())
        .collect();
    let stale: Vec<&String> = ours
        .keys()
        .filter(|k| exempt(&NOT_COLLECTED, k).is_some())
        .chain(ik.keys().filter(|k| exempt(&OURS_ONLY, k).is_some()))
        .collect();
    let ok = opened && missing.is_empty() && extra.is_empty() && stale.is_empty();
    println!(
        "{} names: the quantizer's reader opens ours ({opened}); ours {} weights, ik's {}; missing {missing:?}, \
         extra {extra:?}, listed but on the wrong side {stale:?}",
        verdict(ok),
        ours.len(),
        ik.len()
    );
    ok
}

/// Clause `counts` of `ours` alone: whether it held.
fn counts(ours: &BTreeMap<String, Ours>) -> bool {
    let rows = (CHUNKS * CTX) as f64;
    let total = |e: &Ours| e.counts.iter().map(|&c| f64::from(c)).sum::<f64>();
    let mut bad = Vec::new();
    let (mut down_total, mut up_total) = (0.0f64, 0.0f64);
    for (name, e) in ours {
        let want = if routed(name) {
            rows * N_USED as f64
        } else {
            rows
        };
        let ok = if kind(name) == "ffn_down_exps.weight" {
            // A pick the card did not compute has no f32 row: the down holds the card's picks, at most the
            // up's per expert.
            let up = ours.get(&name.replace("down", "up"));
            down_total += total(e);
            up_total += up.map_or(0.0, total);
            total(e) > 0.0
                && up.is_some_and(|u| {
                    u.counts.len() == e.counts.len()
                        && e.counts.iter().zip(&u.counts).all(|(d, u)| d <= u)
                })
        } else if routed(name) {
            total(e) == want
        } else {
            e.counts.len() == 1 && e.counts[0] as f64 == want
        };
        if !ok {
            bad.push(format!("{name} counts {} want {want}", total(e)));
        }
    }
    let mut unequal = Vec::new();
    for (name, gate) in ours {
        let Some(up) = [
            ("gate_exps.weight", "up_exps.weight"),
            ("gate_shexp.weight", "up_shexp.weight"),
        ]
        .iter()
        .find_map(|(g, u)| name.strip_suffix(g).map(|p| format!("{p}{u}"))) else {
            continue;
        };
        if ours
            .get(&up)
            .is_none_or(|u| u.counts != gate.counts || u.sums != gate.sums)
        {
            unequal.push(name.clone());
        }
    }
    let ok = bad.is_empty() && unequal.is_empty();
    println!(
        "{} counts: a dense weight's count the {} rows walked, a routed stack's total {} rows times {N_USED} picks, a \
         gate weight's counts and sums its up weight's, a down's the card's picks within its up's per expert \
         ({:.1} % of the up picks here) ({} weights); outside {:?}, gate unequal to up {unequal:?}",
        verdict(ok),
        CHUNKS * CTX,
        CHUNKS * CTX,
        100.0 * down_total / up_total.max(1.0),
        ours.len(),
        &bad[..bad.len().min(6)]
    );
    ok
}

/// Clause `means` of `ours` against `ik`: whether it held.
fn means(ours: &BTreeMap<String, Ours>, ik: &BTreeMap<String, Ik>) -> bool {
    let calls = CHUNKS as f64;
    let mut bad_calls = Vec::new();
    let mut bad = Vec::new();
    let mut seen = Worst::default();
    let mut compared = 0usize;
    for (name, x) in ik {
        let Some(y) = ours.get(name) else { continue };
        compared += 1;
        if x.calls != CHUNKS as i32 {
            bad_calls.push(format!("{name} calls {}", x.calls));
        }
        if x.vals.len() != y.sums.len() {
            bad.push(format!(
                "{name}: ik {} values, ours {}",
                x.vals.len(),
                y.sums.len()
            ));
            continue;
        }
        let mut rels = Vec::with_capacity(x.vals.len());
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for (i, &u) in x.vals.iter().enumerate() {
            let Some(v) = y.mean(i, calls) else { continue };
            let u = f64::from(u);
            sx += u;
            sy += v;
            let scale = u.max(v);
            rels.push(if scale > 0.0 {
                (u - v).abs() / scale
            } else {
                0.0
            });
        }
        rels.sort_by(f64::total_cmp);
        let at = |q: usize| rels.get(rels.len() * q / 100).copied().unwrap_or(0.0);
        let (median, p99, max) = (at(50), at(99), rels.last().copied().unwrap_or(0.0));
        let drift = if sx > 0.0 { (sy / sx - 1.0).abs() } else { 0.0 };
        let l = layer(name).min(COL_MEDIAN.len() - 1);
        let outside = match kind(name) {
            "ffn_down_exps.weight" => {
                seen.down_total.note(drift, name);
                seen.down_median.note(median, name);
                drift > DOWN_TOTAL || median > DOWN_MEDIAN
            }
            k if k.ends_with("_exps.weight") => {
                seen.total.note(drift, name);
                seen.routed_median.note(median, name);
                drift > TOTAL_BAND || median > ROUTED_MEDIAN
            }
            _ => {
                seen.total.note(drift, name);
                seen.median[l] = seen.median[l].max(median);
                seen.p99[l] = seen.p99[l].max(p99);
                seen.max.note(max, name);
                drift > TOTAL_BAND || median > COL_MEDIAN[l] || p99 > COL_P99[l]
            }
        };
        if outside {
            bad.push(name.clone());
        }
    }
    let ok = bad_calls.is_empty() && bad.is_empty() && compared > 0;
    println!(
        "{} means: ik's calls {CHUNKS} everywhere (off: {bad_calls:?}); sum order alone {SUM_ORDER:.1e}; dense \
         columns by layer: median within {COL_MEDIAN:?} (worst {:.3?}), 99th percentile within {COL_P99:?} (worst \
         {:.3?}), the widest column {}; totals within {TOTAL_BAND} (worst {}), a down's within {DOWN_TOTAL} (worst \
         {}); routed gate and up middle column within {ROUTED_MEDIAN} (worst {}), a down's within {DOWN_MEDIAN} \
         (worst {}); {compared} weights compared; outside: {} weights {:?}",
        verdict(ok),
        seen.median,
        seen.p99,
        seen.max,
        seen.total,
        seen.down_total,
        seen.routed_median,
        seen.down_median,
        bad.len(),
        &bad[..bad.len().min(6)]
    );
    ok
}

/// The largest value of a statistic and the weight it came from.
#[derive(Default)]
struct Peak(f64, String);

impl Peak {
    fn note(&mut self, v: f64, name: &str) {
        if v > self.0 {
            *self = Peak(v, name.to_owned());
        }
    }
}

impl std::fmt::Display for Peak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2e} at {}", self.0, self.1)
    }
}

/// What the means clause saw: the worst of each statistic, by layer where a band is by layer.
#[derive(Default)]
struct Worst {
    median: [f64; 9],
    p99: [f64; 9],
    max: Peak,
    total: Peak,
    down_total: Peak,
    routed_median: Peak,
    down_median: Peak,
}

/// The whole clause: ik's run beside two of our walks of the fixture, then the four clauses above. `Ok(true)` when
/// they all held; the real tier decides none (module doc) and loads nothing.
pub fn run(dir: &str, path: &str, cfg: &body::OpenCfg) -> Result<bool, GateError> {
    const NAME: &str = "gate_deepseek41_step imatrix";
    if Tier::from_env()? == Tier::Real {
        for c in ["imatrix names against ik's", "imatrix means against ik's"] {
            tier::run_clause(c, Tag::FixtureOracle)?;
        }
        println!(
            "{NAME}: the real tier decides nothing: ik reads no real V4.1 file whole; `ppl_ds41 --imatrix` is its \
             smoke"
        );
        return Ok(true);
    }
    let dir = Path::new(dir);
    std::fs::create_dir_all(dir).map_err(|e| format!("{NAME}: {}: {e}", dir.display()))?;
    let (ik_dat, ik_log) = (dir.join("ik.dat"), dir.join("ik.log"));
    let ik_tree = std::env::var(IK_ENV).map_err(|_| {
        format!(
            "{NAME}: {IK_ENV} names the ik tree whose llama-imatrix reads the fixture (the recipe sets it from \
             tools/ref/ref-paths.sh)"
        )
    })?;
    let log =
        std::fs::File::create(&ik_log).map_err(|e| format!("{NAME}: {}: {e}", ik_log.display()))?;
    let mut cmd = Command::new(format!("{ik_tree}/build/bin/llama-imatrix"));
    cmd.args(["-m", path, "-f", TEXT_FILE, "-o"])
        .arg(&ik_dat)
        .args([
            "-c",
            &CTX.to_string(),
            "-b",
            &CTX.to_string(),
            "-ub",
            &CTX.to_string(),
        ])
        .args(["--chunks", &CHUNKS.to_string(), "--no-ppl", "-ngl", "0"])
        .env(
            "LD_LIBRARY_PATH",
            format!("{ik_tree}/build/src:{ik_tree}/build/ggml/src:{ik_tree}/build/bin"),
        )
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    die_with_parent(&mut cmd);
    let mut ik = cmd
        .spawn()
        .map_err(|e| format!("{NAME}: {ik_tree}/build/bin/llama-imatrix: {e}"))?;

    let mut cfg = cfg.clone();
    cfg.body.ced = false;
    let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut m = body::open(
        file,
        crate::gate_card::plan_gate,
        usize::try_from(model::placement::workstation::CTX_MAX)?,
        &cfg,
    )?;
    m.set_mode(StepMode::Eager);
    body::prepare_prefill(&mut m)?;
    imx::ready(&mut m)?;
    let tok = Tokenizer::from_gguf(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    let (a, b) = (dir.join("ours-a.gguf"), dir.join("ours-b.gguf"));
    for (i, out) in [&a, &b].into_iter().enumerate() {
        if i > 0 {
            imatrix::attach().reset();
        }
        imatrix_walk::collect(
            &mut m,
            &tok,
            (TEXT_FILE, &out.to_string_lossy()),
            (CTX, Some(CHUNKS)),
            None,
            0,
            imx::chunk,
        )?;
    }
    let status = ik.wait()?;
    if !status.success() {
        return Err(format!("{NAME}: llama-imatrix ended {status}: {}", tail(&ik_log)).into());
    }

    let opened = quantize::imatrix::Imatrix::open(&a.to_string_lossy()).is_ok();
    let (ours, theirs) = (read_ours(&a)?, read_ik(&ik_dat)?);
    let mut pass = true;
    if tier::run_clause("imatrix names against ik's", Tag::FixtureOracle)? {
        pass &= names(&ours, &theirs, opened);
    }
    crate::ds41_tier::sc(
        "imatrix counts: the rows walked, the picks made, the card's picks of a down, a gate weight is its up weight",
    )?;
    pass &= counts(&ours);
    if tier::run_clause("imatrix means against ik's", Tag::FixtureOracle)? {
        pass &= means(&ours, &theirs);
    }
    crate::ds41_tier::sc(
        "imatrix repeat: a second walk from a reset collector writes the first's bytes",
    )?;
    let same = std::fs::read(&a)? == std::fs::read(&b)?;
    println!(
        "{} repeat: {} and {} are {}",
        verdict(same),
        a.display(),
        b.display(),
        if same {
            "the same bytes"
        } else {
            "different files"
        }
    );
    pass &= same;
    Ok(pass)
}
