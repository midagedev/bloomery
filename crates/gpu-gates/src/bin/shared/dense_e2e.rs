//! The dense Body35 generation gate over a family's row: a dense trunk of one model file whole on one card, the Body35
//! chain a request runs — the prompt call, the decode step, the pass, the checkpoints and the resident slots — against
//! llama.cpp mainline's forced-logits rows for that file (`refset::arch::forced_logits`, one set family a file). The
//! row ([`Row`]) is what differs between models: the file, the sets, the geometry its header must give, the flash pass
//! its load opens with, and the bands the oracle's own CPU-twin floor sets ([`Bands`]); every clause is the same code.
//! `gate_qwen35_dense` (the Qwen3.8-27B, gated-delta and gated GQA layers) and `gate_llama_dense` (Nanbeige4.3-33B,
//! attention layers alone) are bins over it.
//!
//! What is asserted:
//! - (a) load: the whole-fit verdict's weights term, the buffers the load uploaded and the header derivation
//!   (`weights::part_format` of every tensor but the row's unused block) are the same bytes; no unused-block tensor
//!   is resident; the verdict's cache term and the body's store bytes are their derivation from the header's geometry.
//! - (b) oracle: for each set (the 64- and 4,096-id prompts, 32 teacher-forced rows each) the prompt call and the
//!   forced decode steps' logits rows against mainline's CUDA rows: each row's relative L2 within the row's rel band,
//!   the median within its median band, the 90th percentile within its p90 band; and the argmax equal on every row
//!   whose reference top-2 margin exceeds its stability bound (the row's `stable_k` times the oracle's own spread on
//!   that row, mainline's CPU twin against its CUDA rows); the rows at or under it are counted and pinned
//!   (`other_rows`).
//! - (c) graph = eager = pass: five tokens as graph decode steps, one pass of five rows, eager steps and graph steps
//!   after a `reset`, each leaving the same logits, tokens and every layer's store bit for bit.
//! - (d) reset: a 600-id sequence with 8 greedy steps, a `reset`, a shorter sequence, a `reset`, the first again: the
//!   two runs equal bit for bit (logits, tokens, every store).
//! - (e) checkpoints. With a delta layer: a prompt of [`KP`] ids takes its points at the ubatch's multiples and its
//!   end; one checkpoint's bytes and slots are their derivation; a cut to a non-checkpoint is refused by name; a cut
//!   to the last inner mark and the re-fed tail equal the uncut call bit for bit; a `reset` drops every point.
//!   Without one (no store to copy): the call takes no point; a cut past the position is refused by name; a cut to
//!   any position and the re-fed tail equal the uncut call bit for bit; a `reset` leaves no point.
//! - (f) slots: the slots harness (`slots_gate`, H1-H7) over two streams on one load, a pass of one row a slot equal to
//!   each alone, the pass refusals and each slot's checkpoints its own.

use super::q35_clauses::{
    self as q35, Geometry35, PathRun, Slots35, fresh, lcg_ids, same_run, stores, uncut_call,
};
use super::qwen35_open;
use bloomery_gpu::arch::qwen3moe::{Body35, Open35, Qwen35moeModel};
use bloomery_gpu::kv_format::KvQ8;
use bloomery_gpu::model::StepMode;
use bloomery_gpu::weights::part_format;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::{
    GateError, checks_failed, data_dir, gpu_census, kld, quantile, ref_model_path, verdict,
};
use gguf::{GgmlType, Split};
use refset::arch::forced_logits::{ForcedLogits, Kind};
use std::path::Path;

/// The cache every clause loads: the oracle sets' context (a 4,096-id prompt and 31 forced steps fit), with the room
/// the checkpoint clause's marks need.
const CTX: usize = 4352;
/// The load's ubatch: four units of the 4,096-id prompt, marks at its multiples.
const U_GATE: usize = 1024;

/// Greedy steps after each prompt of (d).
const D_STEPS: usize = 8;
/// (d)'s two prompts: the longer first, the shorter after a `reset`, so no row of the first lies past the second's.
const D_LONG: usize = 600;
const D_SHORT: usize = 64;

/// (e)'s prompt: past two marks (1,024 and 2,048) with a run of 200 ids left after them, short of the third.
const KP: usize = 2 * U_GATE + 200;

/// (f)'s streams, A and B: each past the load's first mark by a run the mark keeps.
const SLOT_A: usize = U_GATE + 16;
const SLOT_B: usize = U_GATE + 32;

/// The bands one family's (b) holds its rows to, each derived from the oracle's own floor (`dense-logits.py floor`) in
/// the row's `PIN` comments.
#[derive(Clone, Copy, Debug)]
pub struct Bands {
    /// The bound on one row's relative L2 distance from mainline's CUDA row.
    pub rel: f64,
    /// The bound on a set's median row.
    pub median: f64,
    /// The bound on a set's 90th percentile.
    pub p90: f64,
    /// The factor on a row's stability bound: the move taken is the oracle's own spread on that row times this.
    pub stable_k: f64,
    /// The rows at or under their stability bound (the only rows that could flip), per set; a count that moves means
    /// the sets or the twin moved, not the engine.
    pub other_rows: [usize; 2],
}

/// One model file's row of the gate.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    /// The bin's name: the gate's first line and the `what` its refusals name.
    pub gate: &'static str,
    /// The file the oracle sets were dumped from (the refset family's `MODEL`).
    pub model: &'static str,
    /// The 64- and 4,096-id sets' directory names under `data_dir()`.
    pub sets: [&'static str; 2],
    /// The real file's header literals, beside the geometry its header gives.
    pub real: Geometry35,
    /// The `blk.<n>.` prefix of tensors the Body35 reader leaves unused (a next-token block), if the file has one.
    pub unused_block: Option<&'static str>,
    /// Whether the load runs the tensor-core flash pass: a group of 7 query heads a KV head has no other.
    pub mma: bool,
    pub bands: Bands,
}

/// The load every clause opens: the oracle sets' context at the gate's ubatch, f16 K/V, the row's flash pass.
fn options(mma: bool) -> Open35 {
    Open35 {
        ctx: CTX,
        mma,
        ubatch: U_GATE,
        kv: KvQ8::F16,
        media: false,
    }
}

fn load(path: &Path, mma: bool) -> Result<(Qwen35moeModel, Split), GateError> {
    qwen35_open::open_options(path, options(mma), &|_| {})
}

/// `(‖a − b‖₂, max |a − b|, ‖b‖₂)` in f64; a NaN reads infinite.
pub fn norms(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut num, mut den, mut max) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let e = f64::from(x) - f64::from(y);
        num += e * e;
        den += f64::from(y).powi(2);
        max = if e.is_nan() {
            f64::INFINITY
        } else {
            max.max(e.abs())
        };
    }
    let l2 = if num.is_nan() {
        f64::INFINITY
    } else {
        num.sqrt()
    };
    (l2, max, den.sqrt())
}

/// Run the gate on `row`'s file (`ref_model_path`), its sets read as family `K`'s.
pub fn run<K: Kind>(row: &Row) -> Result<(), GateError> {
    bloomery_levers::at_main(&[])?;
    let path = ref_model_path()?;
    println!("{}: {}", row.gate, path.display());
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let geom = Geometry35::of(&split)?;
    let witness = geom == row.real;
    let own = path == Path::new(row.model);
    println!(
        "geometry from the header {geom:?} (the real file's {:?}{}) {}",
        row.real,
        if own { "" } else { ", another file" },
        verdict(witness || !own)
    );
    let mut ok = witness || !own;
    drop(split);

    // `--only <a-f>` runs one clause (the FAIL-first mutants' loop); no flag runs them all.
    let only = std::env::args().skip_while(|a| a != "--only").nth(1);
    let want = |c: &str| only.as_deref().is_none_or(|o| o == c);
    if let Some(o) = &only
        && !["a", "b", "c", "d", "e", "f"].contains(&o.as_str())
    {
        return Err(format!("--only {o}: a clause is one of a, b, c, d, e, f").into());
    }
    let vocab = if ["a", "b", "c", "d", "e"].iter().any(|c| want(c)) {
        if want("a") {
            ok &= load_clause(row, &path, geom)?;
        }
        let (mut m, split) = load(&path, row.mma)?;
        if want("a") {
            ok &= resident_clause(row, &m, &split, geom)?;
        }
        drop(split);
        m.set_mode(StepMode::Graph);
        let mut sets = Vec::new();
        for name in row.sets {
            sets.push(ForcedLogits::<K>::open(&data_dir().join(name))?);
        }
        if want("b") {
            println!("(b) forced logits rows against mainline's CUDA rows");
            for (i, (name, prompt)) in [(row.sets[0], 64), (row.sets[1], 4096)].iter().enumerate() {
                let twin = twin_of(name, *prompt, &sets[i])?;
                let ours = rows_of(&mut m, &sets[i])?;
                ok &= oracle(&ours, row.bands, (*name, *prompt), (&sets[i], &twin), i)?;
            }
        }
        if want("c") {
            println!("(c) graph = eager = pass");
            let toks: Vec<u32> = sets[0].prompt.iter().take(5).copied().collect();
            ok &= q35::paths(&mut m, &toks)?.0;
        }
        if want("d") {
            ok &= reset_clause(row, &mut m)?;
        }
        if want("e") {
            ok &= checkpoint_clause(row, &mut m, geom)?;
        }
        m.body(row.gate)?.vocab()
    } else {
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        split
            .find("token_embd.weight")
            .map(|(_, t)| t.dims[1] as usize)
            .ok_or("the file holds no token_embd.weight")?
    };

    if want("f") {
        println!("(f) resident slots: the harness's contracts over two lcg streams");
        expect_q6k_head(&path)?;
        let mma = row.mma;
        let slots = Slots35::new(
            lcg_ids(SLOT_A + SLOT_B, vocab),
            SLOT_A,
            (U_GATE, CTX),
            (geom, 1 + 2 * geom.n_delta + 3 * geom.n_attn),
            Box::new(move || load(&path, mma).map(|(m, _)| m)),
        )?;
        ok &= q35::slots_two(&slots)?;
    }
    if ok { Ok(()) } else { Err(checks_failed()) }
}

/// The head is Q6_K, else refused by name: the slots adapter's pass-node count is the Q6_K head's.
fn expect_q6k_head(path: &Path) -> Result<(), GateError> {
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let ty = split.find("output.weight").map(|(_, t)| t.ty);
    match ty {
        Some(GgmlType::Q6_K) => Ok(()),
        other => Err(format!(
            "output.weight is {other:?}; the slots derivation is written for Q6_K"
        )
        .into()),
    }
}

/// Tensors of the header but the row's unused block: `(name, resident bytes)` by `weights::part_format`.
fn derived_weights(row: &Row, split: &Split) -> Result<Vec<(String, usize)>, GateError> {
    split
        .iter_tensors()
        .map(|(_, t)| t)
        .filter(|t| row.unused_block.is_none_or(|b| !t.name.starts_with(b)))
        .map(|t| {
            part_format(t)
                .map(|(_, bytes)| (t.name.clone(), bytes))
                .map_err(Into::into)
        })
        .collect()
}

/// (a), the verdict half, before the load: the whole-fit line, its weights and cache terms against the derivation.
fn load_clause(row: &Row, path: &Path, geom: Geometry35) -> Result<bool, GateError> {
    let without = row
        .unused_block
        .map_or(String::new(), |b| format!(" without {b}*"));
    println!("(a) load: the whole-fit verdict and the header derivation{without}");
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let census = gpu_census::census()?;
    let card = Place::A.on(&census)?.card_specs()?[0];
    let need = qwen35_open::whole_verdict(&split, options(row.mma), card, &|_| {})?;
    println!("  whole fit at ctx {CTX}: {need}");
    let derived: usize = derived_weights(row, &split)?.iter().map(|d| d.1).sum();
    let all: usize = split
        .iter_tensors()
        .map(|(_, t)| part_format(t).unwrap_or_else(|e| panic!("{e}")).1)
        .sum();
    let kv = geom.store_bytes(CTX) as u64;
    // With an unused block the file holds bytes the load leaves out, and the clause sees that it does.
    let leaves_out = row.unused_block.is_none() || all > derived;
    let ok = need.weights_unrounded == derived as u64 && need.kv == kv && leaves_out;
    let with_unused = row.unused_block.map_or(String::new(), |b| {
        format!(" (with {b}* {all} B, {} B more)", all - derived)
    });
    println!(
        "  verdict weights {} B of buffers, derived {derived} B{with_unused}; cache {} B, derived {kv} B ({} attention \
         layers' K/V planes of 2 x {} heads x {CTX} x {} f16 + {} layers' state and conv ring) {}",
        need.weights_unrounded,
        need.kv,
        geom.n_attn,
        geom.n_kv,
        geom.head,
        geom.n_delta,
        verdict(ok)
    );
    Ok(ok)
}

/// (a), the load half: the weights the load holds against the derivation, name for name; none is in the unused block;
/// the body's stores are the geometry's.
fn resident_clause(
    row: &Row,
    m: &Qwen35moeModel,
    split: &Split,
    geom: Geometry35,
) -> Result<bool, GateError> {
    let want = derived_weights(row, split)?;
    let w = m.weights();
    let names: Vec<&str> = w.names().filter(|n| !n.starts_with("derived.")).collect();
    let held: usize = names
        .iter()
        .filter_map(|n| w.get(n))
        .map(|d| d.resident_bytes())
        .sum();
    let want_names: Vec<&str> = want.iter().map(|d| d.0.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    let mut want_sorted = want_names.clone();
    want_sorted.sort_unstable();
    let unused = row
        .unused_block
        .map_or(0, |b| names.iter().filter(|n| n.starts_with(b)).count());
    let derived: usize = want.iter().map(|d| d.1).sum();
    let stores = m.body(row.gate)?.store_bytes();
    let ok =
        sorted == want_sorted && held == derived && unused == 0 && stores == geom.store_bytes(CTX);
    let resident = row
        .unused_block
        .map_or(String::new(), |b| format!(", {unused} of {b}* resident"));
    println!(
        "  resident: {} file tensors {held} B (derived {derived} B over {} tensors){resident}; stores {stores} B \
         (derived {} B); resident_bytes {} B {}",
        names.len(),
        want.len(),
        geom.store_bytes(CTX),
        m.resident_bytes(),
        verdict(ok)
    );
    Ok(ok)
}

/// The prompt call and the forced steps of `set` on `m` from a `reset`: the logits row after the prompt, then after
/// each forced id fed.
pub fn rows_of<K: Kind>(
    m: &mut Qwen35moeModel,
    set: &ForcedLogits<K>,
) -> Result<Vec<Vec<f32>>, GateError> {
    m.reset()?;
    <Body35 as app::Prompt>::prompt(m, &set.prompt)?;
    let mut rows = vec![m.logits()?];
    for &id in &set.forced[..set.rows() - 1] {
        m.step(&[id])?;
        rows.push(m.logits()?);
    }
    Ok(rows)
}

/// The CPU twin of `set` (`<name>.cpu`, mainline's CPU kernels on the same file): the oracle's own floor, refused by
/// name unless it ran the prompt and forced ids `set` holds and `set` is the `prompt`-id set.
pub fn twin_of<K: Kind>(
    name: &str,
    prompt: usize,
    set: &ForcedLogits<K>,
) -> Result<ForcedLogits<K>, GateError> {
    let twin = ForcedLogits::<K>::open_cpu(&data_dir().join(format!("{name}.cpu")))?;
    if twin.prompt != set.prompt || twin.forced != set.forced || set.prompt.len() != prompt {
        return Err(format!(
            "{name}: its twin is not the CPU run of the same prompt and forced ids"
        )
        .into());
    }
    Ok(twin)
}

/// (b) on one set (`set_at` its place in `b.other_rows`): `ours`, the logits rows `rows_of` gave after the set's prompt
/// and forced ids, against `set`'s CUDA rows within `b`, the stability rule drawn from `twin`'s spread, and the pinned
/// count of rows under it. The one owner of the per-row rule: `gate_qwen35_dense`'s fixture-oracle G3 calls it with
/// bands of its own.
pub fn oracle<K: Kind>(
    ours: &[Vec<f32>],
    b: Bands,
    (name, prompt): (&str, usize),
    (set, twin): (&ForcedLogits<K>, &ForcedLogits<K>),
    set_at: usize,
) -> Result<bool, GateError> {
    let other = b.other_rows[set_at];
    let (mut rel, mut flips_stable, mut stable, mut under, mut flips_under) =
        (Vec::new(), 0usize, 0usize, 0usize, 0usize);
    let mut worst = (0.0f64, 0usize);
    for (i, row_logits) in ours.iter().enumerate() {
        let reference = set.row(i);
        if row_logits.len() != reference.len() {
            return Err(format!(
                "{name} row {i}: {} logits, the set's {}",
                row_logits.len(),
                reference.len()
            )
            .into());
        }
        let (l2, _, norm) = norms(row_logits, reference);
        rel.push(l2 / norm);
        if l2 / norm > worst.0 || l2.is_infinite() {
            worst = (l2 / norm, i);
        }
        let (twin_l2, twin_max, _) = norms(twin.row(i), reference);
        let bound = b.stable_k * f64::min(2.0 * twin_max, std::f64::consts::SQRT_2 * twin_l2);
        let (ours_top, _, _, _) = kld::top2(row_logits);
        let (ref_top, first, _, second) = kld::top2(reference);
        let margin = f64::from(first) - f64::from(second);
        if margin > bound {
            stable += 1;
            flips_stable += usize::from(ours_top != ref_top);
        } else {
            under += 1;
            flips_under += usize::from(ours_top != ref_top);
        }
    }
    let (median, p90) = (quantile(&rel, 0.5), quantile(&rel, 0.9));
    let ok = rel.iter().all(|&r| r <= b.rel)
        && median <= b.median
        && p90 <= b.p90
        && flips_stable == 0
        && under == other;
    println!(
        "  {name}: {} rows after {prompt} ids; rel L2 median {median:.4} (band {:.4}), p90 {p90:.4} (band {:.4}), worst \
         {:.4} at row {} (band {:.4}); argmax equal on {}/{stable} stable rows; {under} rows at or under their stability \
         bound (pinned {other}), {flips_under} of them flipped {}",
        ours.len(),
        b.median,
        b.p90,
        worst.0,
        worst.1,
        b.rel,
        stable - flips_stable,
        verdict(ok)
    );
    Ok(ok)
}

/// `ids` as one prompt call and `steps` greedy steps from the model's state, every position's logits and token and
/// every store read back.
fn sequence(m: &mut Qwen35moeModel, ids: &[u32], steps: usize) -> Result<PathRun, GateError> {
    let mut tok = <Body35 as app::Prompt>::prompt(m, ids)?;
    let (mut logits, mut tokens) = (vec![m.logits()?], vec![tok]);
    for _ in 0..steps {
        tok = m.step(&[tok])?;
        logits.push(m.logits()?);
        tokens.push(tok);
    }
    Ok(PathRun {
        logits,
        tokens,
        stores: stores(m)?,
    })
}

/// (d): the first sequence again after a `reset`, a shorter sequence between, equal bit for bit.
fn reset_clause(row: &Row, m: &mut Qwen35moeModel) -> Result<bool, GateError> {
    println!("(d) reset: the same sequence after a reset and another sequence");
    let vocab = m.body(row.gate)?.vocab();
    let (long, short) = (lcg_ids(D_LONG, vocab), lcg_ids(D_SHORT, vocab));
    fresh(m)?;
    let first = sequence(m, &long, D_STEPS)?;
    m.reset()?;
    sequence(m, &short, D_STEPS)?;
    m.reset()?;
    let again = sequence(m, &long, D_STEPS)?;
    Ok(same_run(
        "the first sequence after a reset and a shorter one",
        &again,
        &first,
    ))
}

/// (e): the checkpoints of a marked prompt call and a cut with its re-fed tail (module doc). A load with a delta layer
/// takes points at the ubatch's marks and restores one on a cut; one without holds no store to copy, takes none and
/// keeps every position a cut asks for.
fn checkpoint_clause(
    row: &Row,
    m: &mut Qwen35moeModel,
    geom: Geometry35,
) -> Result<bool, GateError> {
    println!("(e) checkpoints: the points, the cut, the re-feed and the reset");
    let recurrent = geom.n_delta > 0;
    m.body_parts(row.gate)?.2.set_checkpoints(true);
    let ids = lcg_ids(KP, m.body(row.gate)?.vocab());
    let ub = u32::try_from(m.ubatch()?)?;
    let c = m.body(row.gate)?.checkpoints();
    let want_bytes = geom.checkpoint_bytes() as u64;
    let want_slots = if recurrent {
        let fit = runtime::seqstate::capacity(runtime::seqstate::HOST_BUDGET, want_bytes)
            .map_err(|e| e.to_string())?;
        (CTX.div_ceil(ub as usize) + 1).min(fit)
    } else {
        0
    };
    let bytes_ok = c.bytes() == want_bytes && c.capacity() == want_slots;
    println!(
        "  one copy {} B (derived {want_bytes} B), {} slots (want {want_slots}) {}",
        c.bytes(),
        c.capacity(),
        verdict(bytes_ok)
    );
    let uncut = uncut_call(m, &ids)?;
    let want_pts: Vec<u32> = if recurrent {
        (1..)
            .map(|k| ub * k)
            .take_while(|&p| p < KP as u32)
            .chain([KP as u32])
            .collect()
    } else {
        Vec::new()
    };
    let body = m.body(row.gate)?;
    let (pts, walks) = (body.checkpoints().positions(), body.prompt_walks());
    let walks_want = (KP as u32).div_ceil(ub) as usize;
    let points_ok = pts == want_pts && walks == walks_want;
    println!(
        "  a prompt of {KP} ids took its points at {pts:?} (want {want_pts:?}), {walks} walks (want {walks_want}) {}",
        verdict(points_ok)
    );
    let ask = (KP - 50) as u32;
    let (refused_ok, cut_at) = if recurrent {
        let refused = m
            .rollback(ask)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let ok = refused.contains("no checkpoint restores it")
            && m.body(row.gate)?.checkpoints().positions() == want_pts;
        println!(
            "  a cut to {ask}: {refused}; the points stand {}",
            verdict(ok)
        );
        (ok, ub * (ask / ub))
    } else {
        let past = KP as u32 + 1;
        let refused = m
            .rollback(past)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let ok = refused.contains("back to position") && m.pos() == KP as u32;
        println!(
            "  a cut to {past}: {refused}; the position stands at {} {}",
            m.pos(),
            verdict(ok)
        );
        (ok, ask)
    };
    let refed_ok = q35::refed_after_cut(m, &ids, cut_at, &uncut, &want_pts, ub)?;
    fresh(m)?;
    let dropped = m.body(row.gate)?.checkpoints().positions().is_empty();
    println!("  a reset drops every point {}", verdict(dropped));
    m.set_mode(StepMode::Graph);
    Ok(bytes_ok && points_ok && refused_ok && refed_ok && dropped)
}
