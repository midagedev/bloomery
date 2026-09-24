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
//! The free entry `moe::experts_union_into` (one file, a `MoeBlockPlan`) is
//! held to its own `experts_into` on V2-Lite's block 1 at k = 6, over seeded
//! activation columns with lists overlapping by construction. Then every
//! named refusal: more columns than
//! `UNION_MAX_COLS`, a union past `UNION_MAX`, a list count other than `x`'s
//! columns, a list past `EXPERTS_INTO_MAX`, an `out` of another length and a
//! scratch made for other widths.
//!
//! `hw_`: needs the V4.1 shards, the oracle set and V2-Lite on the box
//! (`just gate-union`).

#[path = "common/v41set.rs"]
mod v41set;

use gguf::Split;
use model::arch::deepseek41::host;
use model::arch::deepseek41::hparams::Hparams;
use model::moe::{
    EXPERTS_INTO_MAX, HostLayer, HostScratch, UNION_MAX, UNION_MAX_COLS, UnionScratch,
};
use model::ops::{self, Tensor2};
use model::placement::workstation;

/// The oracle set and the ik tree it was dumped from.
const SET: &str = "ref_deepseek41";
// PIN(2026-09-23): the sink-fixed oracle tree the V4.1 sets carry.
const BUILD: &str = "db517b69";

/// Column counts the gate runs.
const KS: [usize; 4] = [1, 2, 3, 6];

/// The deferral arms: the claim inside the row dispatch, then the pre-pass.
const ARMS: [(&str, bool); 2] = [("defer", true), ("prepass", false)];

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

    let k = UNION_MAX_COLS + 1;
    let x = Tensor2::zeros(embd, k);
    let lists = vec![&one[..]; k];
    let mut out = vec![0.0f32; embd * k];
    check(
        refusal(
            layer.experts_union_into(split, &x, &lists, &mut out, us),
            "k past UNION_MAX_COLS",
        ),
        "at most UNION_MAX_COLS columns",
    );

    // Seven columns of six distinct experts each: 42 distinct.
    let wide: Vec<Vec<(u32, f32)>> = (0..7u32)
        .map(|j| (0..6u32).map(|s| (6 * j + s, 1.0)).collect())
        .collect();
    let lists: Vec<&[(u32, f32)]> = wide.iter().map(Vec::as_slice).collect();
    const { assert!(7 * 6 > UNION_MAX) };
    let x = Tensor2::zeros(embd, 7);
    let mut out = vec![0.0f32; embd * 7];
    check(
        refusal(
            layer.experts_union_into(split, &x, &lists, &mut out, us),
            "union past UNION_MAX",
        ),
        "at most UNION_MAX distinct experts",
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

    let mut other = UnionScratch::new(embd, ff / 2);
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
    let mut us = UnionScratch::new(embd, ff);
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
    let mut us = UnionScratch::new(embd, plan.meta.ff);
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
