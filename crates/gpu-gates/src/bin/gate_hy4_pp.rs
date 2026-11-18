//! The Hy4-flash prompt-batch gate: a prompt walked as one batch (`bloomery_gpu_hy4::prompt`: one
//! pass per layer over every row, the projections and the routed experts as grouped int8 GEMMs, the
//! sparse layers' selector and the window layers' staged ring over the call's rows) against the
//! decode chain over the same ids (one id a step, the `Body` of the e2e gate) on the fixture file
//! (five layers, the cut [0, 1, 6, 11, 41]: sparse + dense, sparse + engram + routed, window,
//! window + engram, sparse; window 64, top-k 64).
//!
//! What the two walks differ by, and so how they are held. A Q8_0 projection reads its input in f32
//! in the decode chain's gemv and in int8 blocks of 32 in the batch's GEMM, a rounding of about
//! 0.7 % of each block's largest value: about 1e-2 relative L2 a sub-layer, the size of the gap
//! between the decode chain and the vendor's CPU path (`gate-gpu-hy4-e2e` pins its attention at
//! 1.41e-2 and its feed-forward at 2.12e-2), and the batch's is the vendor's rounding. The free
//! chains then part by the discrete choices that rounding turns (the router's 16 of 512, the
//! selector's kept pools: the fixture's random weights leave both near ties), so a free-chain
//! logits row differs by 0.19 to 0.33 relative L2 (the e2e gate's `σ̂` is 0.16 to 0.42) and stands as a
//! printed measure. The gate holds what a bug moves:
//!
//! - `forced_attention_is_decode`: on each of the five fixture sets (set `e`: 93 ids, past the
//!   64-row window and the selector's kept pools, so the ring's truncation and the indexer's
//!   selection run inside the batch), each layer's attention run alone by the decode chain over the
//!   batch's own normed rows of every position (`attn_norm`), reading the batch's selected rows on a
//!   sparse layer, gives the batch's output at the last row within [`ATTN_BAND`]; the decode chain's
//!   own selection of the same rows is the batch's as a set unless the cut is a tie at the two
//!   scorings' deviation, and the pool scores themselves are within [`SCORE_BAND`].
//! - `forced_ffn_is_decode`: each layer's feed-forward block run alone by the decode chain on the
//!   batch's own normed row (`ffn_norm`) of every position gives the batch's output within
//!   [`FFN_BAND`], and a routed layer's picks are the batch's for every row.
//! - `stores_are_read_by_decode`: after the batch walks a prompt, each layer's attention at the next
//!   position run by the decode chain over the stores the batch left (the window ring, the cache,
//!   the index ring, the pooled keys) against the same attention over stores the decode chain
//!   built from the batch's own normed rows, within [`STORE_BAND`], the selection the same or a tie
//!   and its scores within [`SCORE_BAND`]; on the five sets and on the long prompt, so the window
//!   ring and the index ring wrap.
//! - `batch_is_single`: the batch of a prompt against the same batch rule one id a call (`cap` 1:
//!   the same kernels over one row, so no rounding differs and no choice turns): every stage of
//!   every layer of every row, the head's fold and the logits row within [`SINGLE_BAND`], and the
//!   stores read through [`CONT_STEPS`] decode steps fed the same ids, each logits row within the
//!   band. This reads every store the batch writes (the cache rows, the index ring, the pooled
//!   keys, the window ring, the engram history, the position). It runs on the five sets and on a
//!   prompt of [`LONG`] ids (the index ring of 259 rows and the window ring wrap, a call of several
//!   walks).
//! - `chunking_is_invariant`: the same prompt through a batch whose walks, attention chunks and
//!   feed-forward units are a few rows each gives the large geometry's logits and continuation
//!   within [`CHUNK_BAND`]: on set `e`, on the long prompt and on a prompt of [`BIG`] ids at the
//!   released geometry (4,096 rows a walk, 2,560 a feed-forward unit).
//! - `refusals_are_named`: a geometry the kernels cannot hold, a prompt of no id and a prompt past
//!   the context end by name, and the model keeps decoding after the last one.
//!
//! The prompt sets only supply ids: the vendor's dumps are not read as an oracle here. The fixture
//! tier only: the released file (124 GB) does not fit one card and this gate opens the whole-card
//! body, so the real tier ends by name before anything loads.

#[cfg(feature = "hy4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_pp", gate::run())
}

#[cfg(feature = "hy4")]
mod gate {
    use bloomery_gpu::latent::POOL;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, open_split, verdict};
    use bloomery_gpu_hy4::batch::{Cfg, Prompt, Q_MAX, WalkTaps, stage_width};
    use bloomery_gpu_hy4::{Body, TAP_STAGES, prompt};
    use model::arch::Arch;
    use refset::arch::hy4::fixture::{self, A, B, C, D, E};
    use refset::arch::prompt_tables;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_hy4_pp";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-hy4-pp";

    /// The sets whose ids the gate walks, in the order they run.
    const SETS: &[&str] = &[A, B, C, D, E];

    /// The cut's layers and their kinds, (sparse, engram site), in layer order.
    const KINDS: [(bool, bool); 5] = [
        (true, false),
        (true, true),
        (false, false),
        (false, true),
        (true, false),
    ];

    /// The pools the fixture's selector keeps beside the open one: top-k 64 rows over pools of 4.
    const KEPT: usize = 15;

    /// The ids of the long prompt: set `e` repeated, past the index ring (259 rows) and many
    /// windows of 64.
    const LONG: usize = 300;
    /// The ids of the big prompt: two walks of the released geometry (4,096 rows, then 104), and
    /// two feed-forward units in the first (2,560 rows, then 1,536).
    const BIG: usize = 4200;
    /// The decode steps after a prompt that read its stores.
    const CONT_STEPS: usize = 8;
    /// Positions of every store: the big prompt and the continuation.
    const CTX: usize = BIG + 64;

    /// The rows the tapped batch holds: the long prompt in one walk.
    const TAP_ROWS: usize = LONG;

    /// One id a call: the same kernels over one row.
    const ONE: Cfg = Cfg {
        cap: 1,
        q_rows: 1,
        ffn_rows: 1,
    };

    /// The small geometry of `chunking_is_invariant`: walks of 40 rows, attention chunks of 16 and
    /// feed-forward units of 24, so set `e` crosses three walks, six chunks and four units a walk.
    const SMALL: Cfg = Cfg {
        cap: 40,
        q_rows: 16,
        ffn_rows: 24,
    };

    /// A layer's attention run alone by the decode chain over the batch's rows, against the batch's
    /// output at the last row, relative L2: the e2e gate's own distance to the vendor (1.41e-2) is
    /// the same rounding.
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.18e-2 over five sets and five layers.
    const ATTN_BAND: f64 = 1.8e-2;

    /// A layer's attention at the position after the prompt, over the batch's stores, against the
    /// decode chain's over stores it built from the same rows, relative L2: the cache rows carry
    /// the batch's GEMM rounding of the key projection.
    /// PIN(2026-11-19): 1.5 x the worst measured, 5.97e-3 over six prompts and five layers.
    const STORE_BAND: f64 = 9.0e-3;

    /// A sparse layer's pool scores of the last row, the batch's against the decode chain's on the
    /// batch's own rows, relative L2: rounding of the key and query projections, which the pooled
    /// keys and the dot carry.
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.04e-2 over sets `e` and the long prompt.
    const SCORE_BAND: f64 = 1.6e-2;

    /// A layer's feed-forward block run alone by the decode chain on the batch's rows, against the
    /// batch's output over the rows, relative L2 (the e2e gate pins the vendor's at 2.12e-2).
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.75e-2 over five sets and five layers.
    const FFN_BAND: f64 = 2.6e-2;

    /// A stage or a logits row of the batch against the same rule one id a call, relative L2: the
    /// same kernels over one row, bit-identical in every run (0 over six prompts, every stage of
    /// every row); the band leaves room for a reduction-order tie and nothing more.
    /// PIN(2026-11-19): measured 0.000e0.
    const SINGLE_BAND: f64 = 1.0e-6;

    /// A logits row of the small geometry against the large one, relative L2: the same kernels over
    /// other row sets, bit-identical in every run (0 on three prompts, 4,200 ids the largest).
    /// PIN(2026-11-19): measured 0.000e0.
    const CHUNK_BAND: f64 = 1.0e-6;

    /// The ids of set `name`.
    fn set_ids(name: &'static str) -> Result<Vec<u32>, GateError> {
        let path = tier::fixture_set("the prompt batch", &fixture::LCPP, name)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        Ok(prompt_tables::tokens(&man)?.to_vec())
    }

    /// `ids` repeated to `n` ids.
    fn repeated(ids: &[u32], n: usize) -> Vec<u32> {
        ids.iter().copied().cycle().take(n).collect()
    }

    /// The worst relative L2 of the logits rows `a` against `b`, and which row.
    fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, usize) {
        a.iter()
            .zip(b)
            .map(|(x, y)| rel(x, y))
            .enumerate()
            .fold(
                (0.0, 0),
                |(w, wi), (i, r)| {
                    if r > w { (r, i) } else { (w, wi) }
                },
            )
    }

    /// From the model's current position: [`CONT_STEPS`] decode steps, fed `feed` where given and
    /// its own argmax otherwise; the ids and logits rows after the prompt's own (`first`).
    fn continuation(
        m: &mut GpuModel<Body>,
        first: u32,
        feed: Option<&[u32]>,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        let mut ids = vec![first];
        let mut logits = vec![m.logits()?];
        for j in 0..CONT_STEPS {
            let input = feed.map_or(ids[ids.len() - 1], |f| f[j]);
            ids.push(m.step(&[input])?);
            logits.push(m.logits()?);
        }
        Ok((ids, logits))
    }

    /// `ids` through the batch `p` from a reset model: its argmax.
    fn batch_prompt(m: &mut GpuModel<Body>, p: &mut Prompt, ids: &[u32]) -> Result<u32, GateError> {
        m.reset()?;
        Ok(prompt(m, p, ids)?)
    }

    /// `ids` through the decode chain from a reset model, one id a step: its argmax.
    fn decode_prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GateError> {
        m.reset()?;
        let mut token = 0;
        for &id in ids {
            token = m.step(&[id])?;
        }
        Ok(token)
    }

    /// Row `t` of `stage` of layer `l` in `taps`.
    fn row(taps: &WalkTaps, (l, stage): (usize, usize), t: usize, hidden: usize) -> &[f32] {
        let w = stage_width(stage, hidden);
        &taps.stages[l * TAP_STAGES.len() + stage][t * w..(t + 1) * w]
    }

    /// The tap slot of the stage named `name`.
    fn slot(name: &str) -> usize {
        TAP_STAGES
            .iter()
            .position(|s| *s == name)
            .unwrap_or(usize::MAX)
    }

    /// Two runs' selections of a sparse layer at the position whose live key count is `count`: they
    /// agree when the rows are the same as sets, or differ only where the cut is a tie at the two
    /// scorings' widest deviation, and the pool scores are within [`SCORE_BAND`]. The verdict and
    /// the line that shows it.
    fn selection_agrees(
        count: usize,
        (rows_a, scores_a): (&[u32], &[f32]),
        (rows_b, scores_b): (&[u32], &[f32]),
    ) -> (bool, String) {
        let (mut a, mut b) = (rows_a.to_vec(), rows_b.to_vec());
        a.sort_unstable();
        b.sort_unstable();
        // The pools a row scores: `(c - 1) / 4` of its count `c`, scored when more than the kept.
        let scored = (count - 1) / POOL;
        let (mut dev, mut margin, mut score_rel) = (0.0f32, f32::INFINITY, 0.0f64);
        if scored > KEPT {
            let (x, y) = (&scores_a[..scored], &scores_b[..scored]);
            score_rel = rel(x, y);
            dev = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max);
            let mut sorted = y.to_vec();
            sorted.sort_unstable_by(|p, q| q.total_cmp(p));
            margin = sorted[KEPT - 1] - sorted[KEPT];
        }
        let same = a == b;
        let tie = !same && margin <= 2.0 * dev;
        let agree = (same || tie) && score_rel <= SCORE_BAND;
        (
            agree,
            format!(
                "rows {} vs {} same={same} tie={tie} scored_pools={scored} scores_rel={score_rel:.3e} \
                 widest={dev:.3e} cut_margin={margin:.3e}",
                a.len(),
                b.len()
            ),
        )
    }

    /// Clauses `forced_attention_is_decode` and `forced_ffn_is_decode` of set `name`: the batch
    /// walks `ids`, then the decode chain runs each block alone on the batch's inputs.
    fn forced_is_decode(
        m: &mut GpuModel<Body>,
        p: &mut Prompt,
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        let attn = tier::run_clause(
            &format!("forced_attention_is_decode set={name}"),
            Tag::SelfConsistency,
        )?;
        let ffn = tier::run_clause(
            &format!("forced_ffn_is_decode set={name}"),
            Tag::SelfConsistency,
        )?;
        if !attn && !ffn {
            return Ok(true);
        }
        batch_prompt(m, p, ids)?;
        let t = ids.len();
        let taps = p.taps(m.gpu().stream(), t)?;
        let hidden = m.body(NAME)?.hidden();
        let (gpu, w, body) = m.body_parts(NAME)?;
        let mut ok = true;
        if attn {
            let (norm, out) = (slot("attn_norm"), slot("attn_out"));
            let (mut worst_out, mut cells) = (0.0f64, Vec::new());
            for l in 0..KINDS.len() {
                let rows: Vec<Vec<f32>> = (0..t)
                    .map(|i| row(&taps, (l, norm), i, hidden).to_vec())
                    .collect();
                let theirs = taps.sel[l].as_ref();
                let read = theirs.map(|s| s.0.as_slice());
                let (got, ours) = body.attention_alone(gpu, w, l, &rows, read)?;
                let r = rel(&got, row(&taps, (l, out), t - 1, hidden));
                worst_out = worst_out.max(r);
                cells.push(format!("{l}={r:.2e}"));
                let (Some((rows, scores)), Some((picked, our_scores))) = (theirs, ours) else {
                    continue;
                };
                let (agree, line) = selection_agrees(t, (rows, scores), (&picked, &our_scores));
                ok &= agree;
                println!("  selection set={name} layer={l} {line}");
            }
            let pass = ok && worst_out <= ATTN_BAND;
            println!(
                "forced_attention_is_decode set={name} attn_out_rel=[{}] worst={worst_out:.3e} \
                 band={ATTN_BAND:.1e} {}",
                cells.join(" "),
                verdict(pass)
            );
            ok = pass;
        }
        if ffn {
            let (norm, out) = (slot("ffn_norm"), slot("ffn_out"));
            let (mut worst_out, mut cells, mut flips) = (0.0f64, Vec::new(), 0usize);
            for l in 0..KINDS.len() {
                let (mut got, mut want) = (Vec::new(), Vec::new());
                for i in 0..t {
                    let (y, picks) = body.ffn_alone(gpu, w, l, row(&taps, (l, norm), i, hidden))?;
                    got.extend_from_slice(&y);
                    want.extend_from_slice(row(&taps, (l, out), i, hidden));
                    if let Some((mut ours, _)) = picks {
                        ours.sort_unstable();
                        let n = ours.len();
                        let mut theirs = taps.picks[l][i * n..(i + 1) * n].to_vec();
                        theirs.sort_unstable();
                        flips += usize::from(ours != theirs);
                    }
                }
                let r = rel(&got, &want);
                worst_out = worst_out.max(r);
                cells.push(format!("{l}={r:.2e}"));
            }
            let pass = flips == 0 && worst_out <= FFN_BAND;
            println!(
                "forced_ffn_is_decode set={name} ffn_out_rel=[{}] worst={worst_out:.3e} \
                 band={FFN_BAND:.1e} pick_mismatches={flips} {}",
                cells.join(" "),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `stores_are_read_by_decode` of the prompt `name`: after the batch walks `ids`, each
    /// layer's attention at the next position, run by the decode chain over the stores the batch
    /// left, against the same attention over stores the decode chain built from the batch's own
    /// normed rows (the position's input the last row's, any row standing in), reading the cache
    /// rows the first run selected.
    fn stores_are_read_by_decode(
        m: &mut GpuModel<Body>,
        p: &mut Prompt,
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("stores_are_read_by_decode set={name}"),
            Tag::SelfConsistency,
        )? {
            return Ok(true);
        }
        batch_prompt(m, p, ids)?;
        let t = ids.len();
        let taps = p.taps(m.gpu().stream(), t)?;
        let hidden = m.body(NAME)?.hidden();
        let (gpu, w, body) = m.body_parts(NAME)?;
        let norm = slot("attn_norm");
        let at = u32::try_from(t).map_err(|_| format!("{NAME}: a prompt past u32"))?;
        let (mut ok, mut worst_out, mut cells) = (true, 0.0f64, Vec::new());
        for l in 0..KINDS.len() {
            let mut rows: Vec<Vec<f32>> = (0..t)
                .map(|i| row(&taps, (l, norm), i, hidden).to_vec())
                .collect();
            rows.push(rows[t - 1].clone());
            let (read, sel_b) = body.attention_at(gpu, w, l, (&rows[t], at))?;
            let list = sel_b.as_ref().map(|s| s.0.as_slice());
            let (want, sel_d) = body.attention_alone(gpu, w, l, &rows, list)?;
            let r = rel(&read, &want);
            worst_out = worst_out.max(r);
            cells.push(format!("{l}={r:.2e}"));
            if let (Some(b), Some(d)) = (&sel_b, &sel_d) {
                let (agree, line) = selection_agrees(t + 1, (&b.0, &b.1), (&d.0, &d.1));
                ok &= agree;
                println!("  selection set={name} layer={l} at the next position {line}");
            }
        }
        let pass = ok && worst_out <= STORE_BAND;
        println!(
            "stores_are_read_by_decode set={name} ids={t} attn_rel=[{}] worst={worst_out:.3e} \
             band={STORE_BAND:.1e} {}",
            cells.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `batch_is_single` of the prompt `name`: the batch `p` against the single-id batch
    /// `one` (both tapped), stages, logits and the stores through the
    /// continuation; the free chains' distance from the decode chain prints beside it.
    fn batch_is_single(
        m: &mut GpuModel<Body>,
        (p, one): (&mut Prompt, &mut Prompt),
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(&format!("batch_is_single set={name}"), Tag::SelfConsistency)? {
            return Ok(true);
        }
        let layers = KINDS.len();
        let stages = layers * TAP_STAGES.len();
        // 1. The single-id arm: one call an id, its stages gathered row by row.
        m.reset()?;
        let mut first = 0;
        let mut single: Vec<Vec<f32>> = vec![Vec::new(); stages];
        let mut single_head = Vec::new();
        let mut single_last = None;
        for &id in ids {
            first = prompt(m, one, &[id])?;
            let t = one.taps(m.gpu().stream(), 1)?;
            for (acc, rows) in single.iter_mut().zip(&t.stages) {
                acc.extend_from_slice(rows);
            }
            single_head = t.head.clone();
            single_last = Some(t);
        }
        let (feed, single_logits) = continuation(m, first, None)?;
        // 2. The batch arm.
        let got = batch_prompt(m, p, ids)?;
        let batch_taps = Some(p.taps(m.gpu().stream(), ids.len())?);
        let (_, batch_logits) = continuation(m, got, Some(&feed))?;
        let (mut stage_worst, mut at) = (0.0f64, String::new());
        let mut head_rel = 0.0;
        if let Some(bt) = &batch_taps {
            for (i, (have, want)) in bt.stages.iter().zip(&single).enumerate() {
                let r = rel(have, want);
                if r > stage_worst {
                    let stage = TAP_STAGES[i % TAP_STAGES.len()];
                    (stage_worst, at) = (r, format!("layer {} {stage}", i / TAP_STAGES.len()));
                }
            }
            head_rel = rel(&bt.head, &single_head);
            // The first stage, in the chain's order, and the first row where the two differ at all.
            let hidden = m.body(NAME)?.hidden();
            let first_diff =
                bt.stages
                    .iter()
                    .zip(&single)
                    .enumerate()
                    .find_map(|(i, (have, want))| {
                        let w = stage_width(i % TAP_STAGES.len(), hidden);
                        let r = (0..ids.len())
                            .find(|&r| have[r * w..(r + 1) * w] != want[r * w..(r + 1) * w])?;
                        Some((i / TAP_STAGES.len(), TAP_STAGES[i % TAP_STAGES.len()], r))
                    });
            if let Some((l, stage, r)) = first_diff {
                println!("  first difference set={name}: layer {l} {stage} at row {r}");
            }
            if let Some(last) = &single_last {
                for (l, (a, b)) in bt.sel.iter().zip(&last.sel).enumerate() {
                    let (Some((ra, sa)), Some((rb, sb))) = (a, b) else {
                        continue;
                    };
                    let scored = ids.len() / POOL;
                    let dev = sa[..scored]
                        .iter()
                        .zip(&sb[..scored])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "  selection at the last row set={name} layer={l}: lists_same={} rows={} widest_score={dev:.3e}",
                        ra == rb,
                        ra.len()
                    );
                }
            }
        }
        let (logits_worst, step) = worst(&batch_logits, &single_logits);
        let pass =
            stage_worst <= SINGLE_BAND && head_rel <= SINGLE_BAND && logits_worst <= SINGLE_BAND;
        // 3. The free chains' distance from the decode chain, a measure.
        let decode = decode_prompt(m, ids)?;
        let decode_logits = m.logits()?;
        println!(
            "batch_is_single set={name} ids={} stages_worst={stage_worst:.3e} ({at}) \
             hc_head={head_rel:.3e} logits_worst={logits_worst:.3e} (step {step}) \
             band={SINGLE_BAND:.1e} top1 batch={got} single={first} | free chain vs decode: \
             logits_rel={:.3e} top1 decode={decode} {}",
            ids.len(),
            rel(&batch_logits[0], &decode_logits),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `chunking_is_invariant` of the prompt `name`.
    fn chunking_is_invariant(
        m: &mut GpuModel<Body>,
        (large, small): (&mut Prompt, &mut Prompt),
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("chunking_is_invariant set={name}"),
            Tag::SelfConsistency,
        )? {
            return Ok(true);
        }
        let first = batch_prompt(m, large, ids)?;
        let (feed, a) = continuation(m, first, None)?;
        let got = batch_prompt(m, small, ids)?;
        let (_, b) = continuation(m, got, Some(&feed))?;
        let (w, at) = worst(&b, &a);
        let pass = w <= CHUNK_BAND;
        println!(
            "chunking_is_invariant set={name} ids={} worst_logits_rel={w:.3e} (step {at}) \
             band={CHUNK_BAND:.1e} top1 large={first} small={got} {}",
            ids.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `refusals_are_named`: the geometries and calls the batch refuses, each by its own
    /// name, and the model decoding after a refused call.
    fn refusals_are_named(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<bool, GateError> {
        if !tier::run_clause("refusals_are_named", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let mut ok = true;
        let mut check = |what: &str, got: Result<(), String>, want: &str| {
            let (hit, text) = match got {
                Ok(()) => (false, "accepted".to_string()),
                Err(e) => (e.contains(want), e),
            };
            println!("  refusal {what}: {text}");
            ok &= hit;
        };
        let (gpu, w, body) = m.body_parts(NAME)?;
        let try_cfg = |cfg: Cfg| {
            Prompt::new(gpu, w, body, cfg)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        check(
            "an attention chunk past the index ring",
            try_cfg(Cfg {
                q_rows: Q_MAX + 1,
                ..SMALL
            }),
            "index ring",
        );
        check(
            "a feed-forward unit past the GEMM's slots",
            try_cfg(Cfg {
                ffn_rows: 4096,
                ..SMALL
            }),
            "slots",
        );
        check(
            "a walk of no row",
            try_cfg(Cfg { cap: 0, ..SMALL }),
            "needs rows",
        );
        let mut p = Prompt::new(gpu, w, body, SMALL)?;
        m.reset()?;
        check(
            "a prompt of no id",
            prompt(m, &mut p, &[])
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "no id",
        );
        let past = vec![ids[0]; CTX + 1];
        check(
            "a prompt past the context end",
            prompt(m, &mut p, &past)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "exceeds the resident cache",
        );
        m.reset()?;
        let first = prompt(m, &mut p, &ids[..8])?;
        let next = m.step(&[first])?;
        println!("  the model decodes after the refusals: {first} then {next}");
        println!("refusals_are_named {}", verdict(ok));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the released file (124 GB) does not fit one card and this gate opens the \
                 whole-card body, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<(&str, Vec<u32>)> = SETS
            .iter()
            .map(|&n| Ok((n, set_ids(n)?)))
            .collect::<Result<_, GateError>>()?;
        let e = &sets[SETS.len() - 1].1;
        if e.len() != 93 || e.len() > TAP_ROWS {
            return Err(format!("{NAME}: set e is {} ids, the gate holds 93", e.len()).into());
        }
        let file = open_split(Arch::Hy4, RECIPE)?;
        let t = std::time::Instant::now();
        let mut m = Body::open(Gpu::new()?, file, CTX)?;
        let kinds = m.body(NAME)?.kinds();
        if kinds != KINDS {
            return Err(
                format!("{NAME}: the fixture's layers are {kinds:?}, not {KINDS:?}").into(),
            );
        }
        println!(
            "opened the fixture: resident_bytes={} store_bytes={} ctx={CTX} in {:.1} s",
            m.resident_bytes(),
            m.body(NAME)?.store_bytes(),
            t.elapsed().as_secs_f64()
        );
        let tapped = Cfg {
            cap: TAP_ROWS,
            q_rows: Q_MAX,
            ffn_rows: Cfg::PROMPT.ffn_rows,
        };
        let (mut p, mut one) = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            (
                Prompt::new(gpu, w, body, tapped)?,
                Prompt::new(gpu, w, body, ONE)?,
            )
        };
        p.arm_taps(m.gpu())?;
        one.arm_taps(m.gpu())?;
        let mut ok = true;
        for (name, ids) in &sets {
            ok &= forced_is_decode(&mut m, &mut p, name, ids)?;
        }
        let long = repeated(e, LONG);
        for (name, ids) in sets.iter().map(|(n, i)| (*n, i)).chain([("long", &long)]) {
            ok &= stores_are_read_by_decode(&mut m, &mut p, name, ids)?;
        }
        for (name, ids) in sets.iter().map(|(n, i)| (*n, i)).chain([("long", &long)]) {
            ok &= batch_is_single(&mut m, (&mut p, &mut one), name, ids)?;
        }
        drop(one);
        p.disarm_taps();
        let mut small = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            Prompt::new(gpu, w, body, SMALL)?
        };
        ok &= chunking_is_invariant(&mut m, (&mut p, &mut small), "e", e)?;
        ok &= chunking_is_invariant(&mut m, (&mut p, &mut small), "long", &long)?;
        drop(p);
        let mut big_p = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            Prompt::new(gpu, w, body, Cfg::PROMPT)?
        };
        println!("batch bytes at the released geometry: {}", big_p.bytes());
        let big = repeated(e, BIG);
        ok &= chunking_is_invariant(&mut m, (&mut big_p, &mut small), "big", &big)?;
        ok &= refusals_are_named(&mut m, e)?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: every attention within {ATTN_BAND:.1e} and every feed-forward block \
             within {FFN_BAND:.1e} of the decode chain's on the batch's own inputs with its picks \
             and selection, the stores the batch left read by the decode chain within \
             {STORE_BAND:.1e}, the batch the single-id batch within {SINGLE_BAND:.1e} in every stage \
             and store on the sets and {LONG} ids, the geometry invariant within {CHUNK_BAND:.1e} \
             up to {BIG} ids, the refusals named"
        );
        Ok(())
    }
}
