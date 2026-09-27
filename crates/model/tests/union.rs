//! Union host call gate: `moe::HostLayer::experts_union_into` over `k` token
//! columns equals, column for column and bit for bit, `experts_into` of that
//! column and its own list — on every routed layer of the V4.1 file, for
//! k = 1, 2, 3, 6, 8 and 9, under both deferral arms (`ops::set_defer_quant`:
//! the claim inside the dispatch, and the caller-side pre-pass). k = 8 and 9
//! are the width where the claims stop and the sums leave the caller: the
//! widest call that claims, with the most claim states in use, and the
//! narrowest that runs every pass over the pool. The bit
//! contract holds by construction — the same `dot_row` per (row, column), the
//! same per-column quantization, the same list-order sum from zero — so any
//! differing cell is a bug, not rounding.
//!
//! The lists are real routing: the ids of `ffn_moe_topk-L` (the logical twin)
//! and the weights of `ffn_moe_weights_scaled-L` of the 5-token oracle set
//! `ref_deepseek41`, whose consecutive positions share experts the way a
//! verify pass's rows do. k = 6, 8 and 9 need columns the set does not have:
//! the i-th past the set's tokens is token i's row doubled (any column is a
//! valid input — the contract is per column) with token i + 1's list
//! reversed and token i's first expert repeated at a negative weight, so the
//! cases also carry lists whose order is neither the router's nor the
//! union's, and an expert a list names twice.
//!
//! Every union call is also held to its pool dispatches, by the scratch's
//! count of passes (`UnionScratch::passes`) and by the pool's: five a call
//! past `ops::DEFER_MAX_COLS` columns, two at or under it with the claims,
//! four with the pre-pass, and a cut call its calls' — so a pass split back
//! into dispatches of its own turns the gate red.
//!
//! The wide shape — a prefill ubatch — is held to the same contract at
//! k = 16, 64 and 512 on two routed layers (one Q5_K down, one Q4_K down):
//! synthetic lists over 32, 128 and all of the layer's experts, a hot expert
//! per column so some experts carry many columns (runs of the tile kernel,
//! chunks whose downs are kept in the store), every list length from empty
//! to `LIST`, a repeat at a negative weight, and activation
//! columns that are the set's tokens rescaled. A routed case at k = 512 draws
//! each column's `n_used` experts uniformly from the layer's: the binomial
//! spread of columns per expert a prefill ubatch has, so one chunk mixes
//! experts of one tile run with experts of two or three, and most chunks'
//! downs carry a combine wider than a claim takes. Every wide call quantizes
//! `x` once, runs every expert's gate and up in one row dispatch, every
//! combine in one pass and every down in one row dispatch, lanes cut by the
//! tile's cost; the k <= 8 cases above claim their quantizations instead.
//!
//! The free entry `moe::experts_union_into` (one file, a `MoeBlockPlan`) is
//! held to its own `experts_into` on V2-Lite's block 1 at k = 6, over seeded
//! activation columns with lists overlapping by construction. Then every
//! named refusal: more columns than the scratch was made for, a scratch past
//! `UNION_MAX_COLS` (or of none), an expert id past the layer's, a list count
//! other than `x`'s columns, a list past the scratch's routed width, an `out` of
//! another length and a scratch made for other widths.
//!
//! The group tail — one call over up to `UNION_TAIL_MAX_GROUPS` batches'
//! columns through a scratch sized by a slot budget (`UnionScratch::new_tail`)
//! — is held to the calls one batch at a time (`UNION_MAX_COLS` columns each,
//! views at their offsets into the same columns), column for column and bit
//! for bit, on the same two layers under both deferral arms: eight full
//! batches, and three with the last one 300 columns short. Its lists are a
//! group tail's — each column's experts drawn from a skewed curve over the
//! layer's, cut to the cold half in list order — so most experts carry a few
//! columns spread over several batches and the one call plans its experts
//! unlike any batch call. Both run whole, and so does a routing of every
//! column to one of four experts, each past a batch's columns; a uniform
//! routing of every column over the layer's experts, past the budget, runs
//! cut into batch calls and is counted (`UnionScratch::cut_calls`). The tail
//! scratch's refusals (no batch or too many, a budget under one batch's
//! worst or past its columns') join the refusals below.
//!
//! The r8 lane — a layer whose gate and up come from the r8 sidecar, the
//! row-lane tile over 8-row groups — is held to the file's rows on a
//! synthetic layer (`common/r8layer.rs`: Q3_K gate and up, Q4_K down, its
//! sidecar made by `r8file::convert`, which is `qdot::repack_q3k_r8`): the
//! union call at k = 1, 8, 9, 64 and 512 and `experts_into` of every column
//! equal, bit for bit, `experts_into` of the same layer read from the file
//! (each row's `dot_row`), under both deferral arms, in the calls'
//! dispatches.
//!
//! The Q8_0 down lane — a routed layer whose down is Q8_0, as five of Qwen3.8's are — is held
//! on a synthetic layer (Q4_K gate and up, a Q8_0 down of 480-value rows: three x4 groups and
//! three q8_2 tail blocks) served for ten slots, one expert a column at weight 1.0, so each
//! output column is its slot's down: every cell within the derived band of the f64 dot of the
//! down's dequantized row with the combine (the combine from the same qdot calls the tier
//! makes), and the union call at k = 1, 8 and 10 equal to `experts_into` of every column, bit
//! for bit, under both deferral arms, in its dispatches.
//!
//! PIN(2026-09-26): the four-expert routing runs whole — the scratch's slabs
//! are indexed by slot, so no expert's column count bounds a call.
//!
//! PIN(2026-09-25): removed — the "union past UNION_MAX" refusal: the scratch
//! keeps down columns per listed slot, so no distinct-expert bound exists and
//! the per-list and column bounds are the only ones a call can reach.
//!
//! `hw_`: needs the V4.1 shards, the oracle set and V2-Lite on the box
//! (`just gate-union`).

#[path = "common/r8layer.rs"]
mod r8layer;
#[path = "common/v41set.rs"]
mod v41set;

use std::sync::Arc;

use gguf::{GgmlType, Split, Weights};
use model::arch::deepseek41::host;
use model::arch::deepseek41::hparams::Hparams;
use model::moe::{
    HostLayer, HostLayerSpec, HostScratch, UNION_MAX_COLS, UNION_TAIL_MAX_GROUPS, UnionScratch,
    union_batch_slots,
};
use model::ops::{self, RowLayout, Tensor2, Tensor2View};
use model::placement::workstation;
use model::r8file::{HostR8, R8Source, Sidecar};

/// The oracle set: the V4.1 node dumps' 5-token batch set.
const SET: &str = refset::arch::deepseek41::BATCH;

/// Column counts the gate runs.
const KS: [usize; 6] = [1, 2, 3, 6, 8, 9];

/// The columns the narrow cases' scratch is made for: the widest, one past
/// `ops::DEFER_MAX_COLS`.
const SMALL_COLS: usize = 9;

/// The wide column counts, each with the expert pool its lists draw from
/// (`None`: every expert of the layer).
const WIDE: [(usize, Option<usize>); 3] = [(16, Some(32)), (64, Some(128)), (512, None)];

/// The routed case's columns: one prefill ubatch.
const ROUTED_K: usize = UNION_MAX_COLS;

/// The deferral arms: the claim inside the row dispatch, then the pre-pass.
const ARMS: [(&str, bool); 2] = [("defer", true), ("prepass", false)];

/// The group tails the gate runs: batches of `UNION_MAX_COLS`, and the
/// columns the last one lacks.
const TAILS: [(usize, usize); 2] = [(UNION_TAIL_MAX_GROUPS, 0), (3, 300)];

/// The routed width every scratch here is made for: the longest list the
/// cases carry, past V4.1's six (the cases past the set's tokens add one to a
/// routed list).
const LIST: usize = 8;

/// The tail scratch's slot budget here: two batches at their worst.
const TAIL_SLOTS: usize = 2 * union_batch_slots(LIST);

/// One case: `k` columns and their lists.
struct Case {
    x: Tensor2,
    lists: Vec<Vec<(u32, f32)>>,
}

impl Case {
    fn slices(&self) -> Vec<&[(u32, f32)]> {
        self.lists.iter().map(Vec::as_slice).collect()
    }

    /// Distinct experts over every list.
    fn union(&self) -> usize {
        let mut ids: Vec<u32> = self.lists.iter().flatten().map(|&(e, _)| e).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    fn slots(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }
}

/// The pool dispatches one union call of `k` columns over `slots` listed
/// slots issues under deferral arm `defer`: five past `ops::DEFER_MAX_COLS`
/// columns, two at or under it with the quantizations claimed, four with the
/// pre-pass, none when no slot is listed.
fn call_dispatches(k: usize, slots: usize, defer: bool) -> u64 {
    match (slots, k > ops::DEFER_MAX_COLS, defer) {
        (0, _, _) => 0,
        (_, true, _) => 5,
        (_, false, true) => 2,
        (_, false, false) => 4,
    }
}

/// `call` through `us`, and the pool dispatches it issued by the scratch's
/// count and by the pool's (`None` with one pool thread, which dispatches
/// nothing).
fn dispatched<R>(
    us: &mut UnionScratch,
    call: impl FnOnce(&mut UnionScratch) -> R,
) -> (R, u64, Option<u64>) {
    let (s0, p0) = (us.passes(), threads::pool().stats().dispatches);
    let r = call(us);
    let pool = (threads::pool().threads() > 1).then(|| threads::pool().stats().dispatches - p0);
    (r, us.passes() - s0, pool)
}

/// Whether a call's dispatches, by the scratch's count and the pool's, are
/// `want`.
fn dispatches_ok(scratch: u64, pool: Option<u64>, want: u64) -> bool {
    scratch == want && pool.is_none_or(|p| p == want)
}

/// A call's dispatch counts for its line: the scratch's, the pool's, the
/// contract's.
fn dispatch_note(scratch: u64, pool: Option<u64>, want: u64) -> String {
    let pool = pool.map_or_else(|| "-".to_string(), |p| p.to_string());
    format!("dispatches={scratch} pool={pool} want={want}")
}

/// Cells of `got` whose bits differ from `want`, a length mismatch counting
/// every cell.
fn diff_cells(got: &[f32], want: &[f32]) -> usize {
    if got.len() != want.len() {
        return got.len().max(want.len());
    }
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// The V4.1 cases of one layer: the set's first k tokens for k <= 5, and
/// past them the columns of the module doc.
fn v41_cases(x_all: &[f32], lists: &[Vec<(u32, f32)>], embd: usize) -> Vec<Case> {
    let n_tokens = lists.len();
    KS.iter()
        .map(|&k| {
            let mut data = Vec::with_capacity(k * embd);
            let mut ls = Vec::with_capacity(k);
            for j in 0..k {
                if j < n_tokens {
                    data.extend_from_slice(&x_all[j * embd..(j + 1) * embd]);
                    ls.push(lists[j].clone());
                } else {
                    let i = (j - n_tokens) % n_tokens;
                    data.extend(x_all[i * embd..(i + 1) * embd].iter().map(|&v| 2.0 * v));
                    let mut l: Vec<(u32, f32)> =
                        lists[(i + 1) % n_tokens].iter().rev().copied().collect();
                    l.push((lists[i][0].0, -0.5));
                    ls.push(l);
                }
            }
            Case {
                x: Tensor2::from_vec(embd, k, data),
                lists: ls,
            }
        })
        .collect()
}

/// Each column of `case` through the one-column entry, concatenated.
fn per_column(
    layer: &HostLayer,
    src: R8Source<'_>,
    case: &Case,
    scratch: &mut HostScratch,
) -> Vec<f32> {
    let embd = case.x.ne0;
    let mut want = vec![f32::NAN; embd * case.x.ne1];
    for (j, list) in case.lists.iter().enumerate() {
        let xj = Tensor2::from_vec(embd, 1, case.x.col(j).to_vec());
        layer
            .experts_into(src, &xj, list, &mut want[j * embd..(j + 1) * embd], scratch)
            .unwrap_or_else(|e| panic!("experts_into column {j}: {e}"));
    }
    want
}

/// The refusal text of a union call that must fail.
fn refusal(r: Result<(), model::ModelError>, what: &str) -> String {
    match r {
        Ok(()) => panic!("{what}: the union call must be refused"),
        Err(e) => e.to_string(),
    }
}

/// Every named refusal of the union entry, on one V4.1 layer.
fn refusals(layer: &HostLayer, src: R8Source<'_>, embd: usize, ff: usize, us: &mut UnionScratch) {
    let one = [(0u32, 1.0f32)];
    let check = |got: String, want: &str| {
        assert!(
            got.contains(want),
            "refusal must name {want:?}, got {got:?}"
        );
        println!("refusal {want:?}: {got}");
    };

    let k = us.max_cols() + 1;
    let x = Tensor2::zeros(embd, k);
    let lists = vec![&one[..]; k];
    let mut out = vec![0.0f32; embd * k];
    check(
        refusal(
            layer.experts_union_into(src, &x, &lists, &mut out, us),
            "k past the scratch's columns",
        ),
        "at most the columns the scratch was made for",
    );

    for cols in [0, UNION_MAX_COLS + 1] {
        check(
            refusal(
                UnionScratch::new_routed(embd, ff, cols, LIST).map(|_| ()),
                "scratch columns outside 1..=UNION_MAX_COLS",
            ),
            "1..=UNION_MAX_COLS columns",
        );
    }
    for groups in [0, UNION_TAIL_MAX_GROUPS + 1] {
        check(
            refusal(
                UnionScratch::new_tail(embd, ff, groups, union_batch_slots(LIST), LIST).map(|_| ()),
                "tail scratch batches outside 1..=UNION_TAIL_MAX_GROUPS",
            ),
            "1..=UNION_TAIL_MAX_GROUPS batches",
        );
    }
    for slots in [union_batch_slots(LIST) - 1, 2 * union_batch_slots(LIST) + 1] {
        check(
            refusal(
                UnionScratch::new_tail(embd, ff, 2, slots, LIST).map(|_| ()),
                "tail scratch slots outside one batch's worst ..= every column's",
            ),
            "one batch's slots at worst ..= every column's slots",
        );
    }

    let past = [(layer.n_expert() as u32, 1.0f32)];
    let x = Tensor2::zeros(embd, 2);
    let mut out = vec![0.0f32; embd * 2];
    check(
        refusal(
            layer.experts_union_into(src, &x, &[&one[..], &past[..]], &mut out, us),
            "expert id past the layer's",
        ),
        &format!("expert {}", layer.n_expert()),
    );

    let x = Tensor2::zeros(embd, 3);
    let lists = vec![&one[..]; 2];
    let mut out = vec![0.0f32; embd * 3];
    check(
        refusal(
            layer.experts_union_into(src, &x, &lists, &mut out, us),
            "ne1 != k",
        ),
        "one column per list",
    );

    let long: Vec<(u32, f32)> = (0..=LIST as u32).map(|e| (e, 1.0)).collect();
    let x = Tensor2::zeros(embd, 1);
    let mut out = vec![0.0f32; embd];
    check(
        refusal(
            layer.experts_union_into(src, &x, &[&long[..]], &mut out, us),
            "list past the scratch's routed width",
        ),
        "at most the scratch's routed width of experts per column",
    );

    let mut short = vec![0.0f32; embd - 1];
    check(
        refusal(
            layer.experts_union_into(src, &x, &[&one[..]], &mut short, us),
            "short out",
        ),
        "every column's width",
    );

    let mut other = UnionScratch::new_routed(embd, ff / 2, SMALL_COLS, LIST)
        .expect("a scratch of SMALL_COLS columns");
    check(
        refusal(
            layer.experts_union_into(src, &x, &[&one[..]], &mut out, &mut other),
            "scratch of other widths",
        ),
        "scratch must be made for the block's widths",
    );
}

#[test]
#[ignore = "hw: needs the box, the V4.1 shards and $BLOOMERY_DATA/ref_deepseek41"]
fn hw_union_matches_per_column_v41() {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let src = R8Source::rows(&split);
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    assert!(
        n_tokens >= 2,
        "{SET}: {n_tokens} tokens, the k = 6 column needs two"
    );
    let mut host = HostScratch::new(embd, ff, LIST).expect("a scratch of LIST experts");
    let mut us = UnionScratch::new_routed(embd, ff, SMALL_COLS, LIST)
        .expect("a scratch of SMALL_COLS columns");
    // Per k: layers, columns, slots, distinct experts, differing cells.
    let mut tally = [[0usize; 5]; KS.len()];
    let mut failed: Vec<(usize, usize, &str)> = Vec::new();
    let mut miscounted: Vec<(usize, usize, &str)> = Vec::new();
    let mut first_layer = None;
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(src, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
        else {
            println!("layer={l} no routed experts");
            continue;
        };
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        let ids = set.i32s_logical(&format!("ffn_moe_topk-{l}"), n_used * n_tokens);
        let ws = set.f32s(
            &format!("ffn_moe_weights_scaled-{l}"),
            [1, n_used, n_tokens],
        );
        let lists: Vec<Vec<(u32, f32)>> = (0..n_tokens)
            .map(|t| {
                (0..n_used)
                    .map(|s| {
                        let id = ids[t * n_used + s];
                        let e = u32::try_from(id)
                            .unwrap_or_else(|_| panic!("layer {l}: routed id {id} is negative"));
                        (e, ws[t * n_used + s])
                    })
                    .collect()
            })
            .collect();
        for (ki, case) in v41_cases(&x_all, &lists, embd).iter().enumerate() {
            let k = case.x.ne1;
            let want = per_column(&layer, src, case, &mut host);
            let lists = case.slices();
            let mut line = format!(
                "layer={l} k={k} slots={} union={}",
                case.slots(),
                case.union()
            );
            let mut layer_diff = 0;
            let mut counted = true;
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let mut got = vec![f32::NAN; embd * k];
                let (r, ds, dp) = dispatched(&mut us, |us| {
                    layer.experts_union_into(src, &case.x, &lists, &mut got, us)
                });
                ops::set_defer_quant(None);
                r.unwrap_or_else(|e| panic!("layer {l} k {k} {arm}: {e}"));
                let d = diff_cells(&got, &want);
                let cols_equal = (0..k)
                    .filter(|&j| {
                        diff_cells(
                            &got[j * embd..(j + 1) * embd],
                            &want[j * embd..(j + 1) * embd],
                        ) == 0
                    })
                    .count();
                let want_d = call_dispatches(k, case.slots(), on);
                line += &format!(
                    " {arm}: cols_bits_equal={cols_equal}/{k} diff_cells={d} {}",
                    dispatch_note(ds, dp, want_d)
                );
                if d != 0 {
                    failed.push((l, k, arm));
                }
                if !dispatches_ok(ds, dp, want_d) {
                    miscounted.push((l, k, arm));
                    counted = false;
                }
                layer_diff += d;
            }
            let pass = layer_diff == 0 && counted;
            println!("{line} {}", if pass { "PASS" } else { "FAIL" });
            let t = &mut tally[ki];
            t[0] += 1;
            t[1] += k;
            t[2] += case.slots();
            t[3] += case.union();
            t[4] += layer_diff;
        }
        if first_layer.is_none() {
            first_layer = Some(layer);
        }
    }
    for (ki, &k) in KS.iter().enumerate() {
        let [layers, cols, slots, union, diff] = tally[ki];
        println!(
            "union k={k} layers={layers} columns={cols} slots={slots} distinct={union} \
             union/slots={:.3} arms=defer,prepass diff_cells={diff} {}",
            union as f64 / slots.max(1) as f64,
            if diff == 0 { "PASS" } else { "FAIL" }
        );
    }
    let layer = first_layer.expect("the file has a routed layer");
    refusals(&layer, src, embd, ff, &mut us);
    assert!(
        failed.is_empty(),
        "{} (layer, k, arm) cases differ from their one-column calls, the first {:?}",
        failed.len(),
        &failed[..failed.len().min(8)]
    );
    assert!(
        miscounted.is_empty(),
        "{} (layer, k, arm) calls issued other than their dispatches, the first {:?}",
        miscounted.len(),
        &miscounted[..miscounted.len().min(8)]
    );
    println!(
        "PASSED: union — experts_union_into equals experts_into column for column, bit for bit, \
         on every routed layer at k = 1, 2, 3, 6, 8, 9 under both deferral arms, in 2 (claims) \
         and 4 (pre-pass) pool dispatches a call up to k = 8 and 5 at k = 9; every refusal named"
    );
}

/// Deterministic xorshift64* stream for the wide cases.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16_777_216.0
    }
}

/// The wide case of `k` columns over `pool` experts (of `n_expert`) on one
/// layer: column `j` is token `j % n_tokens` of `x_all` scaled by
/// `1 + j/64`; its list opens with one of four hot experts, then walks the
/// rest of the pool, with lengths cycling through 0, 1, 2, 3, 6, 7 and
/// `LIST`, and every 13th list (from column 2) repeats its first
/// expert at a negative weight.
fn wide_case(x_all: &[f32], embd: usize, k: usize, pool: usize, n_expert: usize) -> Case {
    let n_tokens = x_all.len() / embd;
    let mut rng = Stream(0x5eed_0000 ^ ((k as u64) << 16) ^ (pool as u64));
    let mut ids: Vec<u32> = (0..n_expert as u32).collect();
    for i in (1..ids.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    let pool = &ids[..pool];
    let (hot, rest) = pool.split_at(4);
    let mut data = Vec::with_capacity(k * embd);
    let mut lists = Vec::with_capacity(k);
    let mut walk = 0usize;
    for j in 0..k {
        let scale = 1.0 + j as f32 / 64.0;
        let t = j % n_tokens;
        data.extend(x_all[t * embd..(t + 1) * embd].iter().map(|&v| v * scale));
        let len = match j % 10 {
            3 => LIST,
            4 => 0,
            5 => 1,
            6 => 3,
            8 => 7,
            9 => 2,
            _ => 6,
        };
        let mut l: Vec<(u32, f32)> = Vec::with_capacity(LIST);
        if len > 0 {
            l.push((hot[j % hot.len()], 0.05 + 0.25 * rng.unit()));
        }
        while l.len() < len {
            let e = rest[walk % rest.len()];
            walk += 1;
            if l.iter().all(|&(f, _)| f != e) {
                l.push((e, 0.05 + 0.25 * rng.unit()));
            }
        }
        if j % 13 == 2 && !l.is_empty() && l.len() < LIST {
            l.push((l[0].0, -0.5));
        }
        lists.push(l);
    }
    Case {
        x: Tensor2::from_vec(embd, k, data),
        lists,
    }
}

/// The routed case of `k` columns on one layer: column `j` is token
/// `j % n_tokens` of `x_all` scaled by `1 - j/(2k)`, routed to `n_used`
/// distinct experts drawn uniformly from the layer's `n_expert` at weights in
/// [0.05, 0.3).
fn routed_case(x_all: &[f32], embd: usize, k: usize, n_expert: usize, n_used: usize) -> Case {
    let n_tokens = x_all.len() / embd;
    let mut rng = Stream(0x0b17_ba5e_0000 ^ k as u64);
    let mut data = Vec::with_capacity(k * embd);
    let mut lists = Vec::with_capacity(k);
    for j in 0..k {
        let scale = 1.0 - j as f32 / (2 * k) as f32;
        let t = j % n_tokens;
        data.extend(x_all[t * embd..(t + 1) * embd].iter().map(|&v| v * scale));
        let mut l: Vec<(u32, f32)> = Vec::with_capacity(n_used);
        while l.len() < n_used {
            let e = (rng.next() % n_expert as u64) as u32;
            if l.iter().all(|&(f, _)| f != e) {
                l.push((e, 0.05 + 0.25 * rng.unit()));
            }
        }
        lists.push(l);
    }
    Case {
        x: Tensor2::from_vec(embd, k, data),
        lists,
    }
}

/// The columns each distinct expert of `case` carries.
fn cols_by_expert(case: &Case) -> std::collections::HashMap<u32, usize> {
    let mut n = std::collections::HashMap::new();
    for l in &case.lists {
        let mut seen: Vec<u32> = l.iter().map(|&(e, _)| e).collect();
        seen.sort_unstable();
        seen.dedup();
        for e in seen {
            *n.entry(e).or_insert(0usize) += 1;
        }
    }
    n
}

/// Columns per distinct expert of `case`: (fewest, most, experts past one
/// tile run of `qdot::TILE_COLS`).
fn cols_per_expert(case: &Case) -> (usize, usize, usize) {
    let n = cols_by_expert(case);
    let lo = n.values().copied().min().unwrap_or(0);
    let hi = n.values().copied().max().unwrap_or(0);
    let past = n.values().filter(|&&m| m > qdot::TILE_COLS).count();
    (lo, hi, past)
}

/// The columns the most-listed expert of `case` carries.
fn max_cols_per_expert(case: &Case) -> usize {
    cols_by_expert(case).into_values().max().unwrap_or(0)
}

#[test]
#[ignore = "hw: needs the box, the V4.1 shards and $BLOOMERY_DATA/ref_deepseek41"]
fn hw_union_wide_matches_per_column_v41() {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let src = R8Source::rows(&split);
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    let mut host = HostScratch::new(embd, ff, LIST).expect("a scratch of LIST experts");
    let mut us = UnionScratch::new_routed(embd, ff, UNION_MAX_COLS, LIST)
        .expect("a scratch of UNION_MAX_COLS");
    // One routed layer per down type the file carries for its routed experts.
    let mut picked: Vec<(usize, HostLayer)> = Vec::new();
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(src, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
        else {
            continue;
        };
        let ty = layer.stacks()[2].info().ty;
        if picked.iter().all(|(_, p)| p.stacks()[2].info().ty != ty) {
            picked.push((l, layer));
        }
    }
    assert!(
        picked.len() >= 2,
        "the gate wants a Q5_K-down and a Q4_K-down routed layer, the file gives {}",
        picked.len()
    );
    let t0 = std::time::Instant::now();
    let mut failed: Vec<(usize, usize, &str)> = Vec::new();
    let mut miscounted: Vec<(usize, usize, &str)> = Vec::new();
    let mut every_expert = true;
    let mut routed_spread = true;
    for (l, layer) in &picked {
        let n_expert = layer.n_expert();
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        for (k, pool) in WIDE {
            let pool = pool.unwrap_or(n_expert).min(n_expert);
            let case = wide_case(&x_all, embd, k, pool, n_expert);
            let want = per_column(layer, src, &case, &mut host);
            let lists = case.slices();
            let mut line = format!(
                "layer={l} down={:?} k={k} pool={pool} slots={} union={} max_cols_per_expert={}",
                layer.stacks()[2].info().ty,
                case.slots(),
                case.union(),
                max_cols_per_expert(&case)
            );
            if pool == n_expert {
                every_expert &= case.union() == n_expert;
            }
            let mut diff = 0;
            let mut counted = true;
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let mut got = vec![f32::NAN; embd * k];
                let (r, ds, dp) = dispatched(&mut us, |us| {
                    layer.experts_union_into(src, &case.x, &lists, &mut got, us)
                });
                ops::set_defer_quant(None);
                r.unwrap_or_else(|e| panic!("layer {l} k {k} {arm}: {e}"));
                let d = diff_cells(&got, &want);
                let want_d = call_dispatches(k, case.slots(), on);
                line += &format!(" {arm}: diff_cells={d} {}", dispatch_note(ds, dp, want_d));
                if d != 0 {
                    failed.push((*l, k, arm));
                }
                if !dispatches_ok(ds, dp, want_d) {
                    miscounted.push((*l, k, arm));
                    counted = false;
                }
                diff += d;
            }
            println!(
                "{line} {}",
                if diff == 0 && counted { "PASS" } else { "FAIL" }
            );
        }
        let case = routed_case(&x_all, embd, ROUTED_K, n_expert, n_used);
        let want = per_column(layer, src, &case, &mut host);
        let lists = case.slices();
        let (lo, hi, past) = cols_per_expert(&case);
        let mut line = format!(
            "layer={l} down={:?} k={ROUTED_K} routed slots={} union={} cols_per_expert={lo}..{hi} \
             past_one_run={past}",
            layer.stacks()[2].info().ty,
            case.slots(),
            case.union(),
        );
        routed_spread &= lo < hi && past > 0 && past < case.union();
        let mut diff = 0;
        let mut counted = true;
        for (arm, on) in ARMS {
            ops::set_defer_quant(Some(on));
            let mut got = vec![f32::NAN; embd * ROUTED_K];
            let (r, ds, dp) = dispatched(&mut us, |us| {
                layer.experts_union_into(src, &case.x, &lists, &mut got, us)
            });
            ops::set_defer_quant(None);
            r.unwrap_or_else(|e| panic!("layer {l} routed {arm}: {e}"));
            let d = diff_cells(&got, &want);
            let want_d = call_dispatches(ROUTED_K, case.slots(), on);
            line += &format!(" {arm}: diff_cells={d} {}", dispatch_note(ds, dp, want_d));
            if d != 0 {
                failed.push((*l, ROUTED_K, arm));
            }
            if !dispatches_ok(ds, dp, want_d) {
                miscounted.push((*l, ROUTED_K, arm));
                counted = false;
            }
            diff += d;
        }
        println!(
            "{line} {}",
            if diff == 0 && counted { "PASS" } else { "FAIL" }
        );
    }
    println!("union wide: wall {:.1} s", t0.elapsed().as_secs_f64());
    assert!(
        routed_spread,
        "the routed case must mix experts of one tile run with experts of more"
    );
    assert!(
        every_expert,
        "the k = 512 case must serve every expert of its layer in one call"
    );
    assert!(
        failed.is_empty(),
        "{} (layer, k, arm) wide cases differ from their one-column calls: {failed:?}",
        failed.len()
    );
    assert!(
        miscounted.is_empty(),
        "{} (layer, k, arm) wide calls issued other than five dispatches: {miscounted:?}",
        miscounted.len()
    );
    println!(
        "PASSED: union wide — experts_union_into equals experts_into column for column, bit for \
         bit, at k = 16, 64, 512 (the last over every expert of the layer) and a routed k = 512 \
         on a Q5_K-down and a Q4_K-down layer under both deferral arms, five pool dispatches a call"
    );
}

/// A group tail's case of `cols` columns on one layer: column `j` is token
/// `j % n_tokens` of `x_all` scaled by `1 + j/(4k)`; its `n_used` distinct
/// experts are drawn by rank from weights `1/(rank + 1)` over a shuffled order
/// of the layer's `n_expert`, at weights in [0.05, 0.3), and the list keeps
/// those of the cold half's ranks in draw order; every 13th list (from column
/// 2) that kept one and has room repeats its first at a negative weight.
fn tail_case(x_all: &[f32], embd: usize, cols: usize, n_expert: usize, n_used: usize) -> Case {
    let n_tokens = x_all.len() / embd;
    let mut rng = Stream(0x7a11_0000 ^ cols as u64);
    let mut order: Vec<u32> = (0..n_expert as u32).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut cdf = Vec::with_capacity(n_expert);
    let mut acc = 0.0f64;
    for r in 0..n_expert {
        acc += 1.0 / (r as f64 + 1.0);
        cdf.push(acc);
    }
    let cold = n_expert / 2;
    let mut data = Vec::with_capacity(cols * embd);
    let mut lists = Vec::with_capacity(cols);
    for j in 0..cols {
        let scale = 1.0 + j as f32 / (4 * cols) as f32;
        let t = j % n_tokens;
        data.extend(x_all[t * embd..(t + 1) * embd].iter().map(|&v| v * scale));
        let mut ranks: Vec<usize> = Vec::with_capacity(n_used);
        let mut l: Vec<(u32, f32)> = Vec::with_capacity(LIST);
        while ranks.len() < n_used {
            let u = f64::from(rng.unit()) * acc;
            let r = cdf.partition_point(|&c| c <= u).min(n_expert - 1);
            if ranks.contains(&r) {
                continue;
            }
            ranks.push(r);
            let w = 0.05 + 0.25 * rng.unit();
            if r >= cold {
                l.push((order[r], w));
            }
        }
        if j % 13 == 2 && !l.is_empty() && l.len() < LIST {
            l.push((l[0].0, -0.5));
        }
        lists.push(l);
    }
    Case {
        x: Tensor2::from_vec(embd, cols, data),
        lists,
    }
}

/// `case` through the calls one batch at a time: `UNION_MAX_COLS` columns a
/// call, each a view of `case.x` at its offset, through `us`, concatenated.
fn batch_calls(
    layer: &HostLayer,
    src: R8Source<'_>,
    case: &Case,
    us: &mut UnionScratch,
) -> Vec<f32> {
    let (embd, k) = (case.x.ne0, case.x.ne1);
    let lists = case.slices();
    let mut want = vec![f32::NAN; embd * k];
    for c0 in (0..k).step_by(UNION_MAX_COLS) {
        let c1 = (c0 + UNION_MAX_COLS).min(k);
        let x = Tensor2View::new(&case.x.data[c0 * embd..c1 * embd], embd, c1 - c0)
            .unwrap_or_else(|e| panic!("view of columns {c0}..{c1}: {e}"));
        layer
            .experts_union_into(src, x, &lists[c0..c1], &mut want[c0 * embd..c1 * embd], us)
            .unwrap_or_else(|e| panic!("batch call of columns {c0}..{c1}: {e}"));
    }
    want
}

/// Experts of `case` whose columns fall in more than one batch of
/// `UNION_MAX_COLS`.
fn experts_across_batches(case: &Case) -> usize {
    let mut batches: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (j, l) in case.lists.iter().enumerate() {
        for &(e, _) in l {
            let b = batches.entry(e).or_default();
            if !b.contains(&(j / UNION_MAX_COLS)) {
                b.push(j / UNION_MAX_COLS);
            }
        }
    }
    batches.values().filter(|b| b.len() > 1).count()
}

#[test]
#[ignore = "hw: needs the box, the V4.1 shards and $BLOOMERY_DATA/ref_deepseek41"]
fn hw_union_tail_matches_batches_v41() {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let src = R8Source::rows(&split);
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    let mut us = UnionScratch::new_routed(embd, ff, UNION_MAX_COLS, LIST)
        .expect("a scratch of UNION_MAX_COLS");
    let mut tail = UnionScratch::new_tail(embd, ff, UNION_TAIL_MAX_GROUPS, TAIL_SLOTS, LIST)
        .expect("a tail scratch of UNION_TAIL_MAX_GROUPS batches");
    let mut picked: Vec<(usize, HostLayer)> = Vec::new();
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(src, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
        else {
            continue;
        };
        let ty = layer.stacks()[2].info().ty;
        if picked.iter().all(|(_, p)| p.stacks()[2].info().ty != ty) {
            picked.push((l, layer));
        }
    }
    assert!(
        picked.len() >= 2,
        "the gate wants a Q5_K-down and a Q4_K-down routed layer, the file gives {}",
        picked.len()
    );
    let t0 = std::time::Instant::now();
    let mut failed: Vec<(usize, usize, &str)> = Vec::new();
    let mut miscounted: Vec<(usize, usize, &str)> = Vec::new();
    let mut spread = true;
    let mut whole_and_cut = true;
    for (l, layer) in &picked {
        let n_expert = layer.n_expert();
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        // Each case with whether it runs cut and whether it is a tail's lists.
        let mut cases: Vec<(Case, bool, bool)> = TAILS
            .iter()
            .map(|&(g, short)| {
                let cols = g * UNION_MAX_COLS - short;
                (tail_case(&x_all, embd, cols, n_expert, n_used), false, true)
            })
            .collect();
        let wide = UNION_TAIL_MAX_GROUPS * UNION_MAX_COLS;
        // Past the budget: every slot of every column.
        cases.push((
            routed_case(&x_all, embd, wide, n_expert, n_used),
            true,
            false,
        ));
        // One slot a column, each of four experts listed by about a quarter
        // of the columns: under the budget, each expert past a batch's
        // columns.
        cases.push((routed_case(&x_all, embd, wide, 4, 1), false, false));
        for (case, cut, tail_lists) in &cases {
            let k = case.x.ne1;
            let lists = case.slices();
            let across = experts_across_batches(case);
            let mut line = format!(
                "layer={l} down={:?} k={k} slots={} union={} max_cols_per_expert={} \
                 experts_across_batches={across}",
                layer.stacks()[2].info().ty,
                case.slots(),
                case.union(),
                max_cols_per_expert(case)
            );
            if *tail_lists {
                spread &= across > 0 && case.union() > LIST;
            }
            let mut diff = 0;
            let mut counted = true;
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let want = batch_calls(layer, src, case, &mut us);
                let cuts = tail.cut_calls();
                let mut got = vec![f32::NAN; embd * k];
                let (r, ds, dp) = dispatched(&mut tail, |tail| {
                    layer.experts_union_into(src, &case.x, &lists, &mut got, tail)
                });
                ops::set_defer_quant(None);
                r.unwrap_or_else(|e| panic!("layer {l} tail k {k} {arm}: {e}"));
                let ran_cut = tail.cut_calls() == cuts + 1;
                whole_and_cut &= ran_cut == *cut && tail.cut_calls() <= cuts + 1;
                let want_d = if *cut {
                    (0..k)
                        .step_by(UNION_MAX_COLS)
                        .map(|c0| {
                            let part = &case.lists[c0..(c0 + UNION_MAX_COLS).min(k)];
                            call_dispatches(part.len(), part.iter().map(Vec::len).sum(), on)
                        })
                        .sum()
                } else {
                    call_dispatches(k, case.slots(), on)
                };
                let d = diff_cells(&got, &want);
                line += &format!(
                    " {arm}: {} diff_cells={d} {}",
                    if ran_cut { "cut" } else { "whole" },
                    dispatch_note(ds, dp, want_d)
                );
                if d != 0 {
                    failed.push((*l, k, arm));
                }
                if !dispatches_ok(ds, dp, want_d) {
                    miscounted.push((*l, k, arm));
                    counted = false;
                }
                diff += d;
            }
            println!(
                "{line} {}",
                if diff == 0 && counted { "PASS" } else { "FAIL" }
            );
        }
    }
    println!("union tail: wall {:.1} s", t0.elapsed().as_secs_f64());
    assert!(
        spread,
        "a tail case must spread experts over several batches and plan more than one list's experts"
    );
    assert!(
        whole_and_cut,
        "the tail cases and the four-expert routing must run whole and the routing past the budget \
         cut, once a call"
    );
    assert!(
        failed.is_empty(),
        "{} (layer, k, arm) tail calls differ from their batch calls: {failed:?}",
        failed.len()
    );
    assert!(
        miscounted.is_empty(),
        "{} (layer, k, arm) tail calls issued other than their dispatches: {miscounted:?}",
        miscounted.len()
    );
    println!(
        "PASSED: union tail — one call over {} and {} columns through a tail scratch of {TAIL_SLOTS} \
         slots equals the calls one batch at a time, column for column, bit for bit, on a Q5_K-down \
         and a Q4_K-down layer under both deferral arms, and so does a routing with experts past a \
         batch's columns; a routing past the budget runs cut into batch calls, counted; five pool \
         dispatches a call",
        TAILS[0].0 * UNION_MAX_COLS - TAILS[0].1,
        TAILS[1].0 * UNION_MAX_COLS - TAILS[1].1
    );
}

/// V2-Lite, the file the stage-1 gates open: `$BLOOMERY_MODEL` or the box's
/// default path.
fn v2lite_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}

/// `n` seeded values in [-1, 1), every 61st scaled by 8 so a block's scale
/// is set by an outlier the way real activations set it.
fn seeded(n: usize, key: u64) -> Vec<f32> {
    let mut s = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 8_388_608.0 - 1.0;
            if i % 61 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

#[test]
#[ignore = "hw: needs the box and the V2-Lite model file"]
fn hw_union_file_matches_per_column_v2lite() {
    let g = gguf::Gguf::open(v2lite_path()).unwrap();
    let derived = model::arch::deepseek2::derived::Derived::new(&g).unwrap();
    let plan = derived.block_plan(1).unwrap().moe().unwrap();
    let embd = plan.gate_views[0].dims[0] as usize;
    let n_expert = plan.gate_views.len() as u32;
    // Six seeded columns, six experts each from a rotating window of eight
    // ids: consecutive columns share most of their experts, in orders that
    // differ; column 2 repeats an expert at a negative weight.
    let pool = [3u32, 17, 60, 5, 9, 40, 22, 1].map(|e| e % n_expert);
    let k = 6;
    let mut data = Vec::with_capacity(k * embd);
    let mut lists: Vec<Vec<(u32, f32)>> = Vec::with_capacity(k);
    for j in 0..k {
        data.extend(seeded(embd, j as u64 + 1));
        let mut l: Vec<(u32, f32)> = (0..6)
            .map(|s| (pool[(j + 5 * s) % pool.len()], 0.125 * (s as f32 + 1.0)))
            .collect();
        if j == 2 {
            l.push((l[0].0, -0.5));
        }
        lists.push(l);
    }
    let x = Tensor2::from_vec(embd, k, data);
    let mut host =
        model::moe::HostScratch::new(embd, plan.meta.ff, LIST).expect("a scratch of LIST experts");
    let mut want = vec![f32::NAN; embd * k];
    for (j, list) in lists.iter().enumerate() {
        let xj = Tensor2::from_vec(embd, 1, x.col(j).to_vec());
        model::moe::experts_into(
            &g,
            plan,
            &xj,
            list,
            &mut want[j * embd..(j + 1) * embd],
            &mut host,
        )
        .unwrap();
    }
    let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
    let mut us = UnionScratch::new_routed(embd, plan.meta.ff, SMALL_COLS, LIST)
        .expect("a scratch of SMALL_COLS columns");
    let mut total = 0;
    let mut counted = true;
    for (arm, on) in ARMS {
        ops::set_defer_quant(Some(on));
        let mut got = vec![f32::NAN; embd * k];
        let (r, ds, dp) = dispatched(&mut us, |us| {
            model::moe::experts_union_into(&g, plan, &x, &slices, &mut got, us)
        });
        ops::set_defer_quant(None);
        r.unwrap_or_else(|e| panic!("v2lite union {arm}: {e}"));
        let d = diff_cells(&got, &want);
        let want_d = call_dispatches(k, slices.iter().map(|l| l.len()).sum(), on);
        println!(
            "v2lite block=1 k={k} {arm}: diff_cells={d} {}",
            dispatch_note(ds, dp, want_d)
        );
        counted &= dispatches_ok(ds, dp, want_d);
        total += d;
    }
    assert_eq!(
        total, 0,
        "the one-file union entry must equal experts_into column for column, bit for bit"
    );
    assert!(
        counted,
        "the one-file union entry issues its calls' dispatches"
    );
    println!("PASSED: union v2lite — experts_union_into (one file) equals experts_into per column");
}

/// The r8 lane's synthetic layer: `embd`, `ff` and experts. A gate row is two
/// super-blocks, an expert 32 row-lane groups — past the 18 of a steal block,
/// so blocks cut experts — and the lists draw from all 32 experts.
const R8_LAYER: (usize, usize, usize) = (512, 256, 32);

/// The r8 lane's column counts: one column, the widest call that claims, the
/// narrowest past the claims, and two wide calls whose experts carry many
/// columns (tile runs cut evenly, of every length up to eight).
const R8_KS: [usize; 5] = [1, 8, 9, 64, 512];

#[test]
#[ignore = "hw: the box's CPU (the row-lane tile runs on AVX2); reads no model file"]
fn hw_union_r8_matches_file_rows() {
    let layer = r8layer::Layer::write("union", R8_LAYER.0, R8_LAYER.1, R8_LAYER.2, GgmlType::Q4_K);
    let (embd, ff, n_expert) = (layer.embd, layer.ff, layer.n_expert);
    let split = Split::open(&layer.source).unwrap();
    let side = HostR8::On(Arc::new(
        Sidecar::open(&layer.sidecar, &split, Weights::Mapped { populate: false })
            .unwrap_or_else(|e| panic!("open {}: {e}", layer.sidecar.display())),
    ));
    let src = R8Source::of(&split, &side).unwrap();
    let spec = layer.spec();
    let rows = HostLayer::build(R8Source::rows(&split), &spec).unwrap();
    let r8 = HostLayer::build(src, &spec).unwrap();
    assert_eq!(
        r8.layouts(),
        [RowLayout::R8, RowLayout::R8, RowLayout::Rows],
        "the gate and the up are the sidecar's"
    );
    let x_all: Vec<f32> = (0..4).flat_map(|t| seeded(embd, 0x7e8 + t)).collect();
    let mut host = HostScratch::new(embd, ff, LIST).expect("a scratch of LIST experts");
    let mut us = UnionScratch::new_routed(embd, ff, UNION_MAX_COLS, LIST)
        .expect("a scratch of UNION_MAX_COLS");
    let mut failed: Vec<(usize, &str)> = Vec::new();
    let mut miscounted: Vec<(usize, &str)> = Vec::new();
    let mut spread = (usize::MAX, 0usize);
    for k in R8_KS {
        let case = wide_case(&x_all, embd, k, n_expert, n_expert);
        let want = per_column(&rows, R8Source::rows(&split), &case, &mut host);
        assert!(
            want.iter().all(|v| v.is_finite()),
            "k {k}: the file's rows give finite values"
        );
        let decode = per_column(&r8, src, &case, &mut host);
        let dd = diff_cells(&decode, &want);
        if dd != 0 {
            failed.push((k, "experts_into"));
        }
        let (lo, hi, past) = cols_per_expert(&case);
        spread = (spread.0.min(lo), spread.1.max(hi));
        let lists = case.slices();
        let mut line = format!(
            "r8 k={k} slots={} union={} cols_per_expert={lo}..{hi} past_one_run={past} \
             experts_into: diff_cells={dd}",
            case.slots(),
            case.union()
        );
        let mut clean = dd == 0;
        for (arm, on) in ARMS {
            ops::set_defer_quant(Some(on));
            let mut got = vec![f32::NAN; embd * k];
            let (r, ds, dp) = dispatched(&mut us, |us| {
                r8.experts_union_into(src, &case.x, &lists, &mut got, us)
            });
            ops::set_defer_quant(None);
            r.unwrap_or_else(|e| panic!("r8 k {k} {arm}: {e}"));
            let d = diff_cells(&got, &want);
            let want_d = call_dispatches(k, case.slots(), on);
            line += &format!(" {arm}: diff_cells={d} {}", dispatch_note(ds, dp, want_d));
            if d != 0 {
                failed.push((k, arm));
                clean = false;
            }
            if !dispatches_ok(ds, dp, want_d) {
                miscounted.push((k, arm));
                clean = false;
            }
        }
        println!("{line} {}", if clean { "PASS" } else { "FAIL" });
    }
    assert!(
        spread.0 <= 1 && spread.1 > 2 * qdot::TILE_COLS,
        "the cases must give experts one column and experts several tile runs, got {spread:?}"
    );
    assert!(
        failed.is_empty(),
        "{} (k, call) r8 cases differ from the file's rows: {failed:?}",
        failed.len()
    );
    assert!(
        miscounted.is_empty(),
        "{} (k, arm) r8 calls issued other than their dispatches: {miscounted:?}",
        miscounted.len()
    );
    println!(
        "PASSED: union r8 — a layer whose gate and up are the sidecar's row-lane copies equals the \
         file's rows column for column, bit for bit, through experts_into and the union call at \
         k = 1, 8, 9, 64, 512 under both deferral arms, in 2 / 4 pool dispatches a call up to \
         k = 8 and 5 past it"
    );
}

/// The Q8_0 down lane's synthetic layer: `embd`, `ff` and experts. A down row of `ff` = 480
/// values is 15 Q8_0 blocks — three x4 groups of its q8_2 column and three tail blocks, the
/// most a column carries.
const Q8_LAYER: (usize, usize, usize) = (512, 480, 4);

/// The Q8_0 lane's ten columns, each listing one expert at weight 1.0: experts 0 to 3 carry 4,
/// 3, 2 and 1 columns.
const Q8_EXPERTS: [u32; 10] = [0, 1, 2, 3, 0, 1, 2, 0, 1, 0];

/// The Q8_0 lane's column counts: one column, the widest call that claims, and all ten slots.
const Q8_KS: [usize; 3] = [1, 8, 10];

const Q8_GATE: &str = "blk.0.ffn_gate_exps.weight";
const Q8_UP: &str = "blk.0.ffn_up_exps.weight";
const Q8_DOWN: &str = "blk.0.ffn_down_exps.weight";

/// f32's unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;

/// One routed layer — Q4_K gate and up, a Q8_0 down — written with `gguf::write` into its own
/// directory (removed on drop), and the stacks' bytes the file holds.
struct Q8Layer {
    dir: std::path::PathBuf,
    source: std::path::PathBuf,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl Q8Layer {
    fn write(embd: usize, ff: usize, n_expert: usize) -> Q8Layer {
        use gguf::write::{Layout, TensorDecl, Writer};
        let dir = std::env::temp_dir().join(format!("q8layer-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("layer.gguf");
        let gate = q8_stack(GgmlType::Q4_K, [embd, ff, n_expert], 0x6a7e);
        let up = q8_stack(GgmlType::Q4_K, [embd, ff, n_expert], 0x0b);
        let down = q8_stack(GgmlType::Q8_0, [ff, embd, n_expert], 0xd0);
        let stacks = [
            (Q8_GATE, GgmlType::Q4_K, [embd, ff, n_expert], &gate),
            (Q8_UP, GgmlType::Q4_K, [embd, ff, n_expert], &up),
            (Q8_DOWN, GgmlType::Q8_0, [ff, embd, n_expert], &down),
        ];
        let decls: Vec<TensorDecl> = stacks
            .iter()
            .map(|&(name, ty, dims, b)| TensorDecl {
                name: name.to_string(),
                dims: dims.map(|d| d as u64).to_vec(),
                type_id: ty.as_u32(),
                nbytes: b.len() as u64,
            })
            .collect();
        let layout = Layout::new(&[], decls).unwrap();
        let file = std::io::BufWriter::new(std::fs::File::create(&source).unwrap());
        let mut w = Writer::new(file, layout).unwrap();
        for &(name, _, _, b) in &stacks {
            w.tensor(name, b).unwrap();
        }
        w.finish().unwrap();
        Q8Layer {
            dir,
            source,
            gate,
            up,
            down,
        }
    }
}

impl Drop for Q8Layer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A stack `[k, n, n_expert]` of `ty` (Q4_K or Q8_0): xorshift bytes, each block's scales set
/// finite — Q4_K's `d` 2^-7 and `dmin` 2^-8, Q8_0's `d` of either sign at 2^-7 .. 2^-3.
fn q8_stack(ty: GgmlType, [k, n, n_expert]: [usize; 3], seed: u64) -> Vec<u8> {
    let (bs, ts) = (
        ty.blck_size().unwrap() as usize,
        ty.type_size().unwrap() as usize,
    );
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut b: Vec<u8> = (0..k / bs * n * n_expert * ts)
        .map(|_| next() as u8)
        .collect();
    for blk in b.chunks_mut(ts) {
        match ty {
            GgmlType::Q4_K => {
                blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x1C00u16.to_le_bytes());
            }
            GgmlType::Q8_0 => {
                let r = next();
                let (exp, mant, sign) = (
                    (8 + r % 5) as u16,
                    (r >> 8) as u16 & 0x3ff,
                    (r >> 20) as u16 & 1,
                );
                let h = (sign << 15) | (exp << 10) | mant;
                blk[0..2].copy_from_slice(&h.to_le_bytes());
            }
            _ => panic!("no synthetic stack of {ty}"),
        }
    }
    b
}

/// `n·u / (1 − n·u)`: the relative bound of `n` roundings.
fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
}

/// The Q8_0 down lane. Each column's output is its one slot's down (`0 + 1.0·d`, exact), held
/// to the f64 dot `R` of the down row's dequantized values `w` (exact in f32: an f16 scale
/// times an i8 code) with the combine `h` — `h` from the calls the tier makes, `quantize_col`,
/// `dot_row` of the Q4_K gate and up, `swiglu_clamp` at the layer's limit. The band, derived
/// before the run, per 32-value block `b` of the row, `W_b = Σ|w|` and `M_b = max|h|` over it:
///
/// 1. h's q8_2 quantization. `d_a = bf16(M_b/127)` is within 2^-8 of `M_b/127`; a code is
///    `nearest_int(fl(h·fl(1/d_a)))`, at most `127/(1 − 2^-8) < 127.5` in magnitude, so none
///    saturates, and `|d_a·q − h| ≤ d_a·(1/2 + 268u) ≤ (M_b/254)(1 + 2^-7)`. Summed:
///    `Σ_b W_b·(M_b/254)(1 + 2^-7)`.
/// 2. Float order. A leaf — one block's lane partial times `f16(d_w)·bf16(d_a)` — takes one
///    product rounding, one FMA per term its lane accumulates (`nb/4` groups and `nb%4` tail
///    blocks: 6 here), three hsum levels: `n = 1 + 6 + 3 = 10`, so `γ_10·Σ|leaves|`, with
///    `|w·d_a·q| ≤ |w|·M_b·(128/127)(1 + 2^-7)`.
/// 3. The f64 reference's own sum: `ff·2^-53·Σ|w·h|`.
///
/// Then the union call at k = 1, 8 and 10 against `experts_into` of every column, bit for bit.
#[test]
#[ignore = "hw: the box's CPU (qdot's fused kernels run on AVX2); reads no model file"]
fn hw_union_q8_0_down_matches_dequant() {
    let (embd, ff, n_expert) = Q8_LAYER;
    let layer = Q8Layer::write(embd, ff, n_expert);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let spec = HostLayerSpec {
        gate: Q8_GATE,
        up: Q8_UP,
        down: Q8_DOWN,
        n_expert,
        embd,
        ff,
        swiglu_limit: 0.0,
    };
    let host_layer =
        HostLayer::build(src, &spec).unwrap_or_else(|e| panic!("a Q8_0 down must build: {e}"));
    let k = Q8_EXPERTS.len();
    let x_all: Vec<f32> = (0..k)
        .flat_map(|j| seeded(embd, 0x0800 + j as u64))
        .collect();
    let case = Case {
        x: Tensor2::from_vec(embd, k, x_all),
        lists: Q8_EXPERTS.iter().map(|&e| vec![(e, 1.0f32)]).collect(),
    };
    let mut host = HostScratch::new(embd, ff, LIST).expect("a scratch of LIST experts");
    let want = per_column(&host_layer, src, &case, &mut host);

    let (row_gu, row_d) = (144 * embd / 256, 34 * ff / 32);
    let nb = ff / 32;
    let n = 1.0 + (nb / 4 + nb % 4) as f64 + 3.0;
    let mut xq = vec![0u8; qdot::col_bytes(GgmlType::Q4_K, embd)];
    let (mut g, mut u, mut h) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; ff]);
    let mut w = vec![0.0f32; ff];
    let (mut outside, mut worst) = (0usize, 0.0f64);
    for (j, &e) in Q8_EXPERTS.iter().enumerate() {
        let e = e as usize;
        qdot::quantize_col(GgmlType::Q4_K, case.x.col(j), &mut xq);
        for r in 0..ff {
            let at = (e * ff + r) * row_gu;
            g[r] = qdot::dot_row(GgmlType::Q4_K, &layer.gate[at..at + row_gu], &xq, embd).unwrap();
            u[r] = qdot::dot_row(GgmlType::Q4_K, &layer.up[at..at + row_gu], &xq, embd).unwrap();
        }
        qdot::swiglu_clamp(&g, &u, spec.swiglu_limit, &mut h);
        let m: Vec<f64> = h
            .chunks(32)
            .map(|b| b.iter().fold(0.0f64, |a, &v| a.max(f64::from(v).abs())))
            .collect();
        for o in 0..embd {
            let at = (e * embd + o) * row_d;
            gguf::dequant_row(GgmlType::Q8_0, &layer.down[at..at + row_d], &mut w).unwrap();
            let r: f64 = w
                .iter()
                .zip(&h)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let (mut quant, mut leaves) = (0.0f64, 0.0f64);
            for (bw, &mb) in w.chunks(32).zip(&m) {
                let wb: f64 = bw.iter().map(|&v| f64::from(v).abs()).sum();
                quant += wb * mb / 254.0 * (1.0 + 2f64.powi(-7));
                leaves += wb * mb * (128.0 / 127.0) * (1.0 + 2f64.powi(-7));
            }
            let wh: f64 = w
                .iter()
                .zip(&h)
                .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                .sum();
            let band = quant + gamma(n) * leaves + ff as f64 * 2f64.powi(-53) * wh;
            let err = (f64::from(want[j * embd + o]) - r).abs();
            if err.is_nan() || err > band {
                outside += 1;
            }
            worst = worst.max(err / band);
        }
    }
    println!(
        "q8_0 down: {k} slots x {embd} rows against the f64 dequant dot: {outside} cells outside \
         the band, max err/band {worst:.3}"
    );

    let mut us = UnionScratch::new_routed(embd, ff, k, LIST).expect("a scratch of ten columns");
    let mut failed: Vec<(usize, &str)> = Vec::new();
    let mut miscounted: Vec<(usize, &str)> = Vec::new();
    for kk in Q8_KS {
        let sub = Case {
            x: Tensor2::from_vec(embd, kk, case.x.data[..kk * embd].to_vec()),
            lists: case.lists[..kk].to_vec(),
        };
        let lists = sub.slices();
        let mut line = format!(
            "q8_0 down k={kk} slots={} union={}",
            sub.slots(),
            sub.union()
        );
        for (arm, on) in ARMS {
            ops::set_defer_quant(Some(on));
            let mut got = vec![f32::NAN; embd * kk];
            let (r, ds, dp) = dispatched(&mut us, |us| {
                host_layer.experts_union_into(src, &sub.x, &lists, &mut got, us)
            });
            ops::set_defer_quant(None);
            r.unwrap_or_else(|e| panic!("q8_0 down k {kk} {arm}: {e}"));
            let d = diff_cells(&got, &want[..kk * embd]);
            let want_d = call_dispatches(kk, sub.slots(), on);
            line += &format!(" {arm}: diff_cells={d} {}", dispatch_note(ds, dp, want_d));
            if d != 0 {
                failed.push((kk, arm));
            }
            if !dispatches_ok(ds, dp, want_d) {
                miscounted.push((kk, arm));
            }
        }
        println!("{line}");
    }
    assert_eq!(
        outside, 0,
        "every Q8_0 down cell lies within the derived band of the f64 dequant dot"
    );
    assert!(
        failed.is_empty(),
        "{} (k, arm) Q8_0 union calls differ from experts_into: {failed:?}",
        failed.len()
    );
    assert!(
        miscounted.is_empty(),
        "{} (k, arm) Q8_0 union calls issued other than their dispatches: {miscounted:?}",
        miscounted.len()
    );
    println!(
        "PASSED: union q8_0 down — a Q8_0 down served for {k} slots within the derived band of the \
         f64 dequant dot, and the union call at k = 1, 8, 10 equal to experts_into bit for bit \
         under both deferral arms"
    );
}
