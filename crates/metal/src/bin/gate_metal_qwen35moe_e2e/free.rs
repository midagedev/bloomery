//! `a_free`: the session's tokens against ik's batch set, the CUDA gate's
//! free contract (`crates/gpu-gates/src/bin/gate_qwen35moe_e2e.rs` `free`)
//! on this backend.
//!
//! The batch set's tokens run host-fed one at a time through the traced
//! step: every layer's output, every router's picks and logits, and the
//! last logits are read as the step leaves them. Each layer's output over
//! the rows ik kept is held to [`FREE_BAND`]; the last position's argmax to
//! the fixture tier's tie band at the head's input error. The router picks
//! are the session's own, read back, so a near-tie flip is excused by the
//! CUDA gate's rule with no probe: a flip that is no near tie, picks that
//! are not the top set of their own logits, held rows under
//! [`FREE_COVER_MIN`] or more than [`FREE_FLIPS_MAX`] flips are red. The
//! same tokens then run through the plain step: its picks and its last
//! logits must equal the traced run's bit for bit, the instrument held to
//! the step it reads.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::session::{Node, Op, Session};
use bloomery_metal::kernels::{ROUTED_IDS, router};

use super::flip::{self, RouteFlip, held_rows, picks_its_top, rel_to, router_tie, worse};
use super::says;
use super::sets::{self, Set};

/// PIN(2026-09-27): the CUDA gate's free-running bound on a layer output's
/// relative distance from ik's, its derivation unchanged: per-layer errors
/// of about 4e-2 of a layer's update, added in quadrature over 40 layers,
/// √40 · 4e-2 ≈ 0.25.
pub const FREE_BAND: f64 = 0.26;

/// PIN(2026-10-13): the CUDA gate's floor on the held share of the rows ik
/// kept, its derivation unchanged: a layer-0 flip of token 0 leaves four of
/// sixteen rows (25 %), and 20 % refuses one row less.
pub const FREE_COVER_MIN: f64 = 0.20;

/// PIN(2026-10-13): the CUDA gate's cap on near-tie flips a set may carry,
/// its derivation unchanged: a broken router flips about every one of the
/// sixteen sites, a correct one about eight; the cap is three quarters.
pub const FREE_FLIPS_MAX: usize = 12;

/// What one token's traced step left at each layer.
struct Row {
    out: Vec<f32>,
    ids: Vec<u32>,
    logits: Vec<f32>,
}

pub fn a_free(s: &mut Session) -> Clause {
    let man = sets::open(Set::Batch)?;
    let (_, toks, _) = man
        .step()
        .map_err(|e| format!("the batch set's tokens: {e}"))?;
    let toks = toks.to_vec();
    let n = s.layer_count();
    let vocab = s.vocab();
    let experts = router::EXPERTS;
    let ik_out: Vec<Vec<f32>> = (0..n)
        .map(|l| sets::tap(&man, &format!("l_out-{l}")))
        .collect::<Result<_, _>>()?;
    let ik_logits = sets::tap(&man, "result_output")?;
    let ik_last = ik_logits
        .len()
        .checked_sub(vocab)
        .map(|at| &ik_logits[at..])
        .ok_or("result_output holds no row")?;

    // The traced run.
    s.reset().map_err(says)?;
    let t_n = toks.len();
    let mut rows: Vec<Vec<Row>> = (0..n).map(|_| Vec::with_capacity(t_n)).collect();
    let mut traced_ids = Vec::with_capacity(t_n);
    let mut traced_logits = Vec::new();
    for &tok in &toks {
        let mut layer_rows: Vec<Option<Row>> = (0..n).map(|_| None).collect();
        let stepped = s
            .step_traced(tok, |op, probe| {
                match op {
                    Op::Experts(l) => {
                        let logits = probe.read(Node::RouterLogits(l))?;
                        layer_rows[l] = Some(Row {
                            out: probe.read(Node::Residual)?,
                            ids: probe.router_ids(l)?,
                            logits,
                        });
                    }
                    Op::Head => traced_logits = probe.read(Node::Logits)?,
                    _ => {}
                }
                Ok(())
            })
            .map_err(says)?;
        traced_ids.push(stepped.id);
        for (l, row) in layer_rows.into_iter().enumerate() {
            rows[l].push(row.ok_or_else(|| format!("layer {l}: the traced step left no output"))?);
        }
    }

    // The plain steps on the same tokens.
    s.reset().map_err(says)?;
    let mut plain_ids = Vec::with_capacity(t_n);
    for &tok in &toks {
        plain_ids.push(s.step(tok).map_err(says)?.id);
    }
    let plain_logits = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    let same_bits = plain_logits.len() == traced_logits.len()
        && plain_logits
            .iter()
            .zip(&traced_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let instrument_ok = plain_ids == traced_ids && same_bits;
    println!(
        "a_free instrument: plain picks {plain_ids:?}, traced {traced_ids:?}; last logits bit for \
         bit {}",
        if same_bits { "equal" } else { "DIFFER" }
    );

    // Every layer's output over the rows ik kept.
    let hidden = rows
        .first()
        .and_then(|layer| layer.first())
        .ok_or("no token: the set holds no step row")?
        .out
        .len();
    let mut rel = vec![vec![None::<f64>; t_n]; n];
    for (l, ik) in ik_out.iter().enumerate() {
        let kept = ik.len() / hidden;
        if kept == 0 || kept > t_n || kept * hidden != ik.len() {
            return Err(format!(
                "l_out-{l}: {} values for {t_n} rows of {hidden}",
                ik.len()
            ));
        }
        for (t, row) in rows[l].iter().enumerate() {
            let Some(i) = (t + kept).checked_sub(t_n) else {
                continue;
            };
            rel[l][t] = Some(rel_to(&row.out, &ik[i * hidden..(i + 1) * hidden]));
        }
    }
    let top = *plain_ids.last().ok_or("no token")?;
    let per_layer: Vec<f64> = (0..n)
        .map(|l| rel[l].iter().flatten().copied().fold(0.0f64, worse))
        .collect();
    let head_in = *per_layer.last().ok_or("the model holds no layer")?;
    let (gap, cap) = (
        flip::head_gap(ik_last, top),
        flip::head_cap(head_in, ik_last),
    );
    let argmax_ok = head_in.is_finite() && flip::head_tie(ik_last, top, head_in);
    let ik_top = flip::argmax(ik_last);

    // The router sites over the rows ik kept.
    let mut sites = 0usize;
    let mut flips: Vec<RouteFlip> = Vec::new();
    let mut inconsistent: Vec<(usize, usize)> = Vec::new();
    for (l, by_token) in rows.iter().enumerate().take(n) {
        let lg = sets::tap(&man, &format!("ffn_moe_logits-{l}"))?;
        let kept = lg.len() / experts;
        let ids = sets::topk(&man, l, experts as u32)?;
        if kept == 0 || kept > t_n || kept * experts != lg.len() || ids.len() != kept * ROUTED_IDS {
            return Err(format!(
                "layer {l}: {} router logits and {} ids for {t_n} rows",
                lg.len(),
                ids.len()
            ));
        }
        for k in 0..kept {
            let t = t_n - kept + k;
            let row = &by_token[t];
            let ours = &row.ids[..ROUTED_IDS];
            let ours_lg = &row.logits[..experts];
            let theirs = &ids[k * ROUTED_IDS..(k + 1) * ROUTED_IDS];
            let ik_lg = &lg[k * experts..(k + 1) * experts];
            sites += 1;
            if !picks_its_top(ours, ours_lg, ROUTED_IDS) {
                inconsistent.push((l, t));
            }
            if ours.iter().all(|&e| theirs.contains(&(e as i32))) {
                continue;
            }
            let margin = flip::margin(ik_lg, theirs) as f32;
            let err = ours_lg
                .iter()
                .zip(ik_lg)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            flips.push(RouteFlip {
                layer: l,
                token: t,
                ours: ours.to_vec(),
                theirs: theirs.to_vec(),
                margin,
                err,
                tie: router_tie(margin, err),
            });
        }
    }
    let held = held_rows(n, t_n, &flips);
    let held_worst: Vec<f64> = (0..n)
        .map(|l| {
            (0..t_n)
                .filter(|&t| held[l][t])
                .filter_map(|t| rel[l][t])
                .fold(0.0f64, worse)
        })
        .collect();
    let count = |keep: &dyn Fn(usize, usize) -> bool| -> Vec<usize> {
        (0..n)
            .map(|l| {
                (0..t_n)
                    .filter(|&t| rel[l][t].is_some() && keep(l, t))
                    .count()
            })
            .collect()
    };
    let (all_n, held_n) = (count(&|_, _| true), count(&|l, t| held[l][t]));
    for (l, e) in held_worst.iter().enumerate() {
        println!(
            "a_free layer={l} l_out_rel={e:.3e} (every kept row {:.3e})",
            per_layer[l]
        );
    }
    let mut ok = instrument_ok && argmax_ok;
    let worst = held_worst.iter().copied().fold(0.0f64, worse);
    ok &= worst <= FREE_BAND;
    for f in &flips {
        let only_ours: Vec<u32> = f
            .ours
            .iter()
            .copied()
            .filter(|&e| !f.theirs.contains(&(e as i32)))
            .collect();
        let only_ik: Vec<i32> = f
            .theirs
            .iter()
            .copied()
            .filter(|&e| !f.ours.contains(&(e as u32)))
            .collect();
        println!(
            "a_free flip layer={} token={}: ours {only_ours:?} ik {only_ik:?}, ik margin {:.3e}, \
             our logits' error {:.3e}: {}",
            f.layer,
            f.token,
            f.margin,
            f.err,
            if f.tie {
                "a near tie (counted)"
            } else {
                "FAIL: not a near tie"
            }
        );
        ok &= f.tie;
    }
    for &(l, t) in &inconsistent {
        println!(
            "a_free route layer={l} token={t}: the ids are not the top set of their own logits FAIL"
        );
    }
    ok &= inconsistent.is_empty();
    let (held_all, rows_all) = (held_n.iter().sum::<usize>(), all_n.iter().sum::<usize>());
    let cover = held_all as f64 / rows_all.max(1) as f64;
    let ties = flips.iter().filter(|f| f.tie).count();
    let (cover_ok, cap_ok) = (cover >= FREE_COVER_MIN, ties <= FREE_FLIPS_MAX);
    ok &= cover_ok && cap_ok;
    let line = format!(
        "{t_n} tokens {toks:?}: worst held l_out_rel={worst:.3e} (band {FREE_BAND:.2}); held \
         {held_all} of {rows_all} rows ({:.0} %, floor {:.0} %), per layer {held_n:?} of \
         {all_n:?}; route sites {sites}, flips {} (near ties {ties}, cap {FREE_FLIPS_MAX}), \
         inconsistent {}; last position argmax ours={top} ik={ik_top} gap {gap:.3} cap {cap:.3} \
         at head_in {head_in:.4} ({}), logits_rel={:.3e}; the instrument {}",
        100.0 * cover,
        100.0 * FREE_COVER_MIN,
        flips.len(),
        inconsistent.len(),
        if argmax_ok {
            "tie band"
        } else {
            "past the cap"
        },
        rel_to(&plain_logits, ik_last),
        if instrument_ok {
            "equals the plain steps"
        } else {
            "DIFFERS from the plain steps"
        }
    );
    if ok {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}
