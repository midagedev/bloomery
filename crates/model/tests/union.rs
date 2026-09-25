//! Union host call gate: `moe::HostLayer::experts_union_into` over `k` token
//! columns equals, column for column and bit for bit, `experts_into` of that
//! column and its own list — on every routed layer of the V4.1 file, for
//! k = 1, 2, 3 and 6, under both deferral arms (`ops::set_defer_quant`: the
//! claim inside the dispatch, and the caller-side pre-pass). The bit
//! contract holds by construction — the same `dot_row` per (row, column), the
//! same per-column quantization, the same list-order sum from zero — so any
//! differing cell is a bug, not rounding.
//!
//! The lists are real routing: the ids of `ffn_moe_topk-L` (the logical twin)
//! and the weights of `ffn_moe_weights_scaled-L` of the 5-token oracle set
//! `ref_deepseek41`, whose consecutive positions share experts the way a
//! verify pass's rows do. k = 6 needs a sixth column the set does not have:
//! token 0's row doubled (any column is a valid input — the contract is per
//! column) with token 1's list reversed and token 0's first expert repeated
//! at a negative weight, so the case also carries a list whose order is
//! neither the router's nor the union's, and an expert a list names twice.
//!
//! The wide shape — a prefill ubatch — is held to the same contract at
//! k = 16, 64 and 512 on two routed layers (one Q5_K down, one Q4_K down):
//! synthetic lists over 32, 128 and all of the layer's experts, a hot expert
//! per column so some experts carry many columns (runs of the tile kernel,
//! chunks whose downs are kept in the store), every list length from empty
//! to `EXPERTS_INTO_MAX`, a repeat at a negative weight, and activation
//! columns that are the set's tokens rescaled. A routed case at k = 512 draws
//! each column's `n_used` experts uniformly from the layer's: the binomial
//! spread of columns per expert a prefill ubatch has, so one chunk mixes
//! experts of one tile run with experts of two or three, and most chunks'
//! downs carry a combine wider than a claim takes. Every wide call quantizes
//! `x` once for all its chunks, produces its wide chunks' combines in the
//! pre-pass and cuts those chunks' lanes by the tile's cost; the k <= 8 cases
//! above never do.
//!
//! The free entry `moe::experts_union_into` (one file, a `MoeBlockPlan`) is
//! held to its own `experts_into` on V2-Lite's block 1 at k = 6, over seeded
//! activation columns with lists overlapping by construction. Then every
//! named refusal: more columns than the scratch was made for, a scratch past
//! `UNION_MAX_COLS` (or of none), an expert id past the layer's, a list count
//! other than `x`'s columns, a list past `EXPERTS_INTO_MAX`, an `out` of
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
//! columns spread over several batches and the one call chunks its experts
//! unlike any batch call. Both run whole; a uniform routing of every column
//! over the layer's experts, past the budget, and a routing of every column
//! to one of four experts, under the budget but each expert past a batch's
//! columns, run cut into batch calls and are counted
//! (`UnionScratch::cut_calls`). The tail scratch's refusals (no
//! batch or too many, a budget under one batch's worst or past its columns')
//! join the refusals below.
//!
//! PIN(2026-09-25): removed — the "union past UNION_MAX" refusal: the scratch
//! keeps down columns per listed slot, so no distinct-expert bound exists and
//! the per-list and column bounds are the only ones a call can reach.
//!
//! `hw_`: needs the V4.1 shards, the oracle set and V2-Lite on the box
//! (`just gate-union`).

#[path = "common/v41set.rs"]
mod v41set;

use gguf::Split;
use model::arch::deepseek41::host;
use model::arch::deepseek41::hparams::Hparams;
use model::moe::{
    EXPERTS_INTO_MAX, HostLayer, HostScratch, UNION_BATCH_SLOTS, UNION_CHUNK, UNION_MAX_COLS,
    UNION_TAIL_MAX_GROUPS, UnionScratch,
};
use model::ops::{self, Tensor2, Tensor2View};
use model::placement::workstation;

/// The oracle set and the ik tree it was dumped from.
const SET: &str = "ref_deepseek41";
// PIN(2026-09-23): the sink-fixed oracle tree the V4.1 sets carry.
const BUILD: &str = "db517b69";

/// Column counts the gate runs.
const KS: [usize; 4] = [1, 2, 3, 6];

/// The columns the decode-shape scratch is made for.
const SMALL_COLS: usize = 8;

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

/// The tail scratch's slot budget here: two batches at their worst.
const TAIL_SLOTS: usize = 2 * UNION_BATCH_SLOTS;

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
/// k = 6 with the sixth column of the module doc.
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
                    data.extend(x_all[..embd].iter().map(|&v| 2.0 * v));
                    let mut l: Vec<(u32, f32)> = lists[1].iter().rev().copied().collect();
                    l.push((lists[0][0].0, -0.5));
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
    split: &Split,
    case: &Case,
    scratch: &mut HostScratch,
) -> Vec<f32> {
    let embd = case.x.ne0;
    let mut want = vec![f32::NAN; embd * case.x.ne1];
    for (j, list) in case.lists.iter().enumerate() {
        let xj = Tensor2::from_vec(embd, 1, case.x.col(j).to_vec());
        layer
            .experts_into(
                split,
                &xj,
                list,
                &mut want[j * embd..(j + 1) * embd],
                scratch,
            )
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
fn refusals(layer: &HostLayer, split: &Split, embd: usize, ff: usize, us: &mut UnionScratch) {
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
            layer.experts_union_into(split, &x, &lists, &mut out, us),
            "k past the scratch's columns",
        ),
        "at most the columns the scratch was made for",
    );

    for cols in [0, UNION_MAX_COLS + 1] {
        check(
            refusal(
                UnionScratch::new(embd, ff, cols).map(|_| ()),
                "scratch columns outside 1..=UNION_MAX_COLS",
            ),
            "1..=UNION_MAX_COLS columns",
        );
    }
    for groups in [0, UNION_TAIL_MAX_GROUPS + 1] {
        check(
            refusal(
                UnionScratch::new_tail(embd, ff, groups, UNION_BATCH_SLOTS).map(|_| ()),
                "tail scratch batches outside 1..=UNION_TAIL_MAX_GROUPS",
            ),
            "1..=UNION_TAIL_MAX_GROUPS batches",
        );
    }
    for slots in [UNION_BATCH_SLOTS - 1, 2 * UNION_BATCH_SLOTS + 1] {
        check(
            refusal(
                UnionScratch::new_tail(embd, ff, 2, slots).map(|_| ()),
                "tail scratch slots outside UNION_BATCH_SLOTS..=EXPERTS_INTO_MAX · columns",
            ),
            "UNION_BATCH_SLOTS..=EXPERTS_INTO_MAX · columns slots",
        );
    }

    let past = [(layer.n_expert() as u32, 1.0f32)];
    let x = Tensor2::zeros(embd, 2);
    let mut out = vec![0.0f32; embd * 2];
    check(
        refusal(
            layer.experts_union_into(split, &x, &[&one[..], &past[..]], &mut out, us),
            "expert id past the layer's",
        ),
        &format!("expert {}", layer.n_expert()),
    );

    let x = Tensor2::zeros(embd, 3);
    let lists = vec![&one[..]; 2];
    let mut out = vec![0.0f32; embd * 3];
    check(
        refusal(
            layer.experts_union_into(split, &x, &lists, &mut out, us),
            "ne1 != k",
        ),
        "one column per list",
    );

    let long: Vec<(u32, f32)> = (0..=EXPERTS_INTO_MAX as u32).map(|e| (e, 1.0)).collect();
    let x = Tensor2::zeros(embd, 1);
    let mut out = vec![0.0f32; embd];
    check(
        refusal(
            layer.experts_union_into(split, &x, &[&long[..]], &mut out, us),
            "list past EXPERTS_INTO_MAX",
        ),
        "at most EXPERTS_INTO_MAX experts per column",
    );

    let mut short = vec![0.0f32; embd - 1];
    check(
        refusal(
            layer.experts_union_into(split, &x, &[&one[..]], &mut short, us),
            "short out",
        ),
        "every column's width",
    );

    let mut other = UnionScratch::new(embd, ff / 2, SMALL_COLS).expect("a scratch of 8 columns");
    check(
        refusal(
            layer.experts_union_into(split, &x, &[&one[..]], &mut out, &mut other),
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
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET, BUILD);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    assert!(
        n_tokens >= 2,
        "{SET}: {n_tokens} tokens, the k = 6 column needs two"
    );
    let mut host = HostScratch::new(embd, ff);
    let mut us = UnionScratch::new(embd, ff, SMALL_COLS).expect("a scratch of 8 columns");
    // Per k: layers, columns, slots, distinct experts, differing cells.
    let mut tally = [[0usize; 5]; KS.len()];
    let mut failed: Vec<(usize, usize, &str)> = Vec::new();
    let mut first_layer = None;
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(&split, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
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
            let want = per_column(&layer, &split, case, &mut host);
            let lists = case.slices();
            let mut line = format!(
                "layer={l} k={k} slots={} union={}",
                case.slots(),
                case.union()
            );
            let mut layer_diff = 0;
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let mut got = vec![f32::NAN; embd * k];
                let r = layer.experts_union_into(&split, &case.x, &lists, &mut got, &mut us);
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
                line += &format!(" {arm}: cols_bits_equal={cols_equal}/{k} diff_cells={d}");
                if d != 0 {
                    failed.push((l, k, arm));
                }
                layer_diff += d;
            }
            println!("{line} {}", if layer_diff == 0 { "PASS" } else { "FAIL" });
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
    refusals(&layer, &split, embd, ff, &mut us);
    assert!(
        failed.is_empty(),
        "{} (layer, k, arm) cases differ from their one-column calls, the first {:?}",
        failed.len(),
        &failed[..failed.len().min(8)]
    );
    println!(
        "PASSED: union — experts_union_into equals experts_into column for column, bit for bit, \
         on every routed layer at k = 1, 2, 3, 6 under both deferral arms; every refusal named"
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
/// `EXPERTS_INTO_MAX`, and every 13th list (from column 2) repeats its first
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
            3 => EXPERTS_INTO_MAX,
            4 => 0,
            5 => 1,
            6 => 3,
            8 => 7,
            9 => 2,
            _ => 6,
        };
        let mut l: Vec<(u32, f32)> = Vec::with_capacity(EXPERTS_INTO_MAX);
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
        if j % 13 == 2 && !l.is_empty() && l.len() < EXPERTS_INTO_MAX {
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
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET, BUILD);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    let mut host = HostScratch::new(embd, ff);
    let mut us = UnionScratch::new(embd, ff, UNION_MAX_COLS).expect("a scratch of UNION_MAX_COLS");
    // One routed layer per down type the file carries for its routed experts.
    let mut picked: Vec<(usize, HostLayer)> = Vec::new();
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(&split, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
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
    let mut every_expert = true;
    let mut routed_spread = true;
    for (l, layer) in &picked {
        let n_expert = layer.n_expert();
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        for (k, pool) in WIDE {
            let pool = pool.unwrap_or(n_expert).min(n_expert);
            let case = wide_case(&x_all, embd, k, pool, n_expert);
            let want = per_column(layer, &split, &case, &mut host);
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
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let mut got = vec![f32::NAN; embd * k];
                let r = layer.experts_union_into(&split, &case.x, &lists, &mut got, &mut us);
                ops::set_defer_quant(None);
                r.unwrap_or_else(|e| panic!("layer {l} k {k} {arm}: {e}"));
                let d = diff_cells(&got, &want);
                line += &format!(" {arm}: diff_cells={d}");
                if d != 0 {
                    failed.push((*l, k, arm));
                }
                diff += d;
            }
            println!("{line} {}", if diff == 0 { "PASS" } else { "FAIL" });
        }
        let case = routed_case(&x_all, embd, ROUTED_K, n_expert, n_used);
        let want = per_column(layer, &split, &case, &mut host);
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
        for (arm, on) in ARMS {
            ops::set_defer_quant(Some(on));
            let mut got = vec![f32::NAN; embd * ROUTED_K];
            let r = layer.experts_union_into(&split, &case.x, &lists, &mut got, &mut us);
            ops::set_defer_quant(None);
            r.unwrap_or_else(|e| panic!("layer {l} routed {arm}: {e}"));
            let d = diff_cells(&got, &want);
            line += &format!(" {arm}: diff_cells={d}");
            if d != 0 {
                failed.push((*l, ROUTED_K, arm));
            }
            diff += d;
        }
        println!("{line} {}", if diff == 0 { "PASS" } else { "FAIL" });
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
    println!(
        "PASSED: union wide — experts_union_into equals experts_into column for column, bit for \
         bit, at k = 16, 64, 512 (the last over every expert of the layer) and a routed k = 512 \
         on a Q5_K-down and a Q4_K-down layer under both deferral arms"
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
        let mut l: Vec<(u32, f32)> = Vec::with_capacity(EXPERTS_INTO_MAX);
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
        if j % 13 == 2 && !l.is_empty() && l.len() < EXPERTS_INTO_MAX {
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
fn batch_calls(layer: &HostLayer, split: &Split, case: &Case, us: &mut UnionScratch) -> Vec<f32> {
    let (embd, k) = (case.x.ne0, case.x.ne1);
    let lists = case.slices();
    let mut want = vec![f32::NAN; embd * k];
    for c0 in (0..k).step_by(UNION_MAX_COLS) {
        let c1 = (c0 + UNION_MAX_COLS).min(k);
        let x = Tensor2View::new(&case.x.data[c0 * embd..c1 * embd], embd, c1 - c0)
            .unwrap_or_else(|e| panic!("view of columns {c0}..{c1}: {e}"));
        layer
            .experts_union_into(
                split,
                x,
                &lists[c0..c1],
                &mut want[c0 * embd..c1 * embd],
                us,
            )
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
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let set = v41set::Set::open(&split, SET, BUILD);
    let (embd, ff, n_used) = (hp.n_embd, hp.experts.ff, hp.experts.n_used);
    let n_tokens = set
        .shapes
        .get("ffn_norm-0")
        .expect("the set has ffn_norm-0")[1];
    let mut us = UnionScratch::new(embd, ff, UNION_MAX_COLS).expect("a scratch of UNION_MAX_COLS");
    let mut tail = UnionScratch::new_tail(embd, ff, UNION_TAIL_MAX_GROUPS, TAIL_SLOTS)
        .expect("a tail scratch of UNION_TAIL_MAX_GROUPS batches");
    let mut picked: Vec<(usize, HostLayer)> = Vec::new();
    for l in 0..hp.n_layer {
        let Some(layer) = host::layer(&split, &hp, l).unwrap_or_else(|e| panic!("layer {l}: {e}"))
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
    let mut spread = true;
    let mut whole_and_cut = true;
    for (l, layer) in &picked {
        let n_expert = layer.n_expert();
        let x_all = set.f32s(&format!("ffn_norm-{l}"), [embd, n_tokens, 1]);
        let mut cases: Vec<(Case, bool)> = TAILS
            .iter()
            .map(|&(g, short)| {
                let cols = g * UNION_MAX_COLS - short;
                (tail_case(&x_all, embd, cols, n_expert, n_used), false)
            })
            .collect();
        let wide = UNION_TAIL_MAX_GROUPS * UNION_MAX_COLS;
        // Past the store: every slot of every column.
        cases.push((routed_case(&x_all, embd, wide, n_expert, n_used), true));
        // Past the blocks: one slot a column, each of four experts listed by
        // about a quarter of the columns.
        cases.push((routed_case(&x_all, embd, wide, 4, 1), true));
        for (case, cut) in &cases {
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
            if !cut {
                spread &= across > 0 && case.union() > UNION_CHUNK;
            }
            let mut diff = 0;
            for (arm, on) in ARMS {
                ops::set_defer_quant(Some(on));
                let want = batch_calls(layer, &split, case, &mut us);
                let cuts = tail.cut_calls();
                let mut got = vec![f32::NAN; embd * k];
                let r = layer.experts_union_into(&split, &case.x, &lists, &mut got, &mut tail);
                ops::set_defer_quant(None);
                r.unwrap_or_else(|e| panic!("layer {l} tail k {k} {arm}: {e}"));
                let ran_cut = tail.cut_calls() == cuts + 1;
                whole_and_cut &= ran_cut == *cut && tail.cut_calls() <= cuts + 1;
                let d = diff_cells(&got, &want);
                line += &format!(
                    " {arm}: {} diff_cells={d}",
                    if ran_cut { "cut" } else { "whole" }
                );
                if d != 0 {
                    failed.push((*l, k, arm));
                }
                diff += d;
            }
            println!("{line} {}", if diff == 0 { "PASS" } else { "FAIL" });
        }
    }
    println!("union tail: wall {:.1} s", t0.elapsed().as_secs_f64());
    assert!(
        spread,
        "a tail case must spread experts over several batches and chunk more than UNION_CHUNK"
    );
    assert!(
        whole_and_cut,
        "the tail cases must run whole and the routings past the store or the blocks cut, once a \
         call"
    );
    assert!(
        failed.is_empty(),
        "{} (layer, k, arm) tail calls differ from their batch calls: {failed:?}",
        failed.len()
    );
    println!(
        "PASSED: union tail — one call over {} and {} columns through a tail scratch of {TAIL_SLOTS} \
         slots equals the calls one batch at a time, column for column, bit for bit, on a Q5_K-down \
         and a Q4_K-down layer under both deferral arms; a routing past the budget, or with an \
         expert past a batch's columns, runs cut into batch calls, counted",
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
    let mut host = model::moe::HostScratch::new(embd, plan.meta.ff);
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
    let mut us = UnionScratch::new(embd, plan.meta.ff, SMALL_COLS).expect("a scratch of 8 columns");
    let mut total = 0;
    for (arm, on) in ARMS {
        ops::set_defer_quant(Some(on));
        let mut got = vec![f32::NAN; embd * k];
        let r = model::moe::experts_union_into(&g, plan, &x, &slices, &mut got, &mut us);
        ops::set_defer_quant(None);
        r.unwrap_or_else(|e| panic!("v2lite union {arm}: {e}"));
        let d = diff_cells(&got, &want);
        println!("v2lite block=1 k={k} {arm}: diff_cells={d}");
        total += d;
    }
    assert_eq!(
        total, 0,
        "the one-file union entry must equal experts_into column for column, bit for bit"
    );
    println!("PASSED: union v2lite — experts_union_into (one file) equals experts_into per column");
}
