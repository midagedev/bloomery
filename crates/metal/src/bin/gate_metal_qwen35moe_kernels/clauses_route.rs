//! The prompt pass's routing and grouped experts: the router's logits over
//! rows (the f32-weight GEMM), the selection of every row's slots, the route
//! table, the grouped GEMM of the experts over the table's tiles, the
//! combine, and the head-norm scale over rows.
//!
//! - `route_rows_select`: the selection of a row of logits is the decode
//!   router's, bit for bit — the same logits (read from the decode router's
//!   own run) give the same ids and weights, ties to the smaller id, a
//!   non-finite logit the router site and the fallback ids with NaN weights.
//! - `route_table`: every slot is placed once, inside its group's run of
//!   places, the places invert, and the tiles cut the groups into runs of at
//!   most [`ROUTE_TILE`] in group order, against the host rule below.
//! - `grouped_gemm`: the SwiGLU pair and the down over the table's tiles, for
//!   the fixture's real gate, up and down stacks, each (row, slot) held to the
//!   f64 oracle's band, the shared expert's group included.
//! - `combine`: the residual plus the slots' outputs in slot order, bit for
//!   bit against the same f32 sum.
//! - `headnorm_rows`: the numerator times each head's scale against f64.
//! - `gemm_f32w`: the router's GEMM over f32 weight rows against f64.
//! - `route_refusals`: the launch modules' refusals, by name.
//!
//! # Bands
//!
//! The grouped GEMM's pair and down are the dense GEMM's tile with a
//! different place, so their dot band is the dense GEMM's
//! ([`super::clauses_gemm::gemm_band`], with its two subnormal terms). The pair
//! epilogue is `silu(g) * u * w` of the tile's own accumulators: silu's slope
//! is under 1.1, exp is taken to be off by 4 ulp (`8 U` relative) and the
//! addition, division and rounding add `3 U (|g| + |silu|)`, as the pair's
//! band (`oracle_experts::pair_width`); the route weight multiplies once.

use std::collections::HashMap;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elem, Elems, Fault};
use bloomery_metal::MetalBuffer;
use bloomery_metal::kernels::router;
use bloomery_metal::kernels::{
    CombineParams, EXPERT_K, EXPERT_SLOTS, F32GemmParams, HEAD_COLS, HeadNormRowsParams,
    LOGIT_STRIDE, Operand, PARTIALS_PER_HEAD, Quant, ROUTE_TILE, RouteSelectParams,
    RouteTableParams, RoutedKind, RoutedParams, VALUE_HEADS, route_table_words, route_tiles,
};
use gguf::quant::GgmlType;

use super::clauses_gemm::{HALF_SUBNORMAL, gemm_band, weight_abs_sums};
use super::clauses_router;
use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, U};
use super::{Harness, SENTINEL_BITS};

/// The hidden width every row has.
const K: usize = router::K;

/// The routed experts.
const EXPERTS: usize = router::EXPERTS;

/// Routed picks a row keeps.
const TOP_K: usize = 8;

/// Slots a row holds: the picks and the shared expert.
const PER_ROW: usize = EXPERT_SLOTS;

/// The down's outputs.
const HIDDEN: usize = K;

fn upload<T: Elem>(h: &Harness<'_>, data: &[T]) -> Result<MetalBuffer<T>, String> {
    h.device.upload(h.stream, data).map_err(why)
}

fn read<T: Elem + Default + Clone>(
    h: &Harness<'_>,
    buffer: &MetalBuffer<T>,
    len: usize,
) -> Result<Vec<T>, String> {
    let mut out = vec![T::default(); len];
    h.device
        .read(h.stream, buffer.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    Ok(out)
}

fn sentinel() -> f32 {
    f32::from_bits(SENTINEL_BITS)
}

fn same_words<T: PartialEq + Copy + core::fmt::Debug>(
    what: &str,
    got: &[T],
    want: &[T],
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{what}: {} words, not {}", got.len(), want.len()));
    }
    match got.iter().zip(want).position(|(g, w)| g != w) {
        None => Ok(()),
        Some(i) => Err(format!("{what}[{i}] is {:?}, not {:?}", got[i], want[i])),
    }
}

fn expect_site(fault: Option<Fault>, what: &str, site: FaultSite) -> Result<(), String> {
    match fault {
        Some(f)
            if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            site.name()
        )),
        None => Err(format!("{what} raised nothing, not {}", site.name())),
    }
}

// ---------------------------------------------------------------- selection

/// One row of the decode router's run: its inputs and what it answered.
struct Decoded {
    logits: Vec<f32>,
    ids: Vec<u32>,
    weights: Vec<f32>,
}

fn decoded(
    h: &Harness<'_>,
    routed: &[f32],
    shared: &[f32],
    seed: u64,
    eps: f32,
) -> Result<Decoded, String> {
    let x = oracle::activations(K, seed, 1.0);
    let gain = vec![1.0f32; K];
    let out = clauses_router::run(h, routed, shared, &x, &gain, eps)?;
    Ok(Decoded {
        logits: out.logits,
        ids: out.ids,
        weights: out.weights,
    })
}

struct Selected {
    ids: Vec<u32>,
    weights: Vec<f32>,
    fault: Option<Fault>,
}

fn select_run(h: &Harness<'_>, rows: &[&Decoded]) -> Result<Selected, String> {
    let n = rows.len();
    let mut logits = vec![sentinel(); n * LOGIT_STRIDE];
    for (r, d) in rows.iter().enumerate() {
        logits[r * LOGIT_STRIDE..r * LOGIT_STRIDE + router::ROWS].copy_from_slice(&d.logits);
    }
    let logits_b = upload(h, &logits)?;
    let mut ids = upload(h, &vec![0xFFFF_FFFFu32; n * (TOP_K + 1)])?;
    let mut weights = upload(h, &vec![sentinel(); n * (TOP_K + 1)])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.route_rows_select(
            encoder,
            RouteSelectParams {
                rows: n,
                top_k: TOP_K,
                logits: logits_b.window(Elems(0), Elems(logits.len()))?,
                logit_stride: LOGIT_STRIDE,
                ids: ids.window_mut(Elems(0), Elems(n * (TOP_K + 1)))?,
                weights: weights.window_mut(Elems(0), Elems(n * (TOP_K + 1)))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    Ok(Selected {
        ids: read(h, &ids, n * (TOP_K + 1))?,
        weights: read(h, &weights, n * (TOP_K + 1))?,
        fault: h.fault()?,
    })
}

fn same_selection(got: &Selected, rows: &[&Decoded]) -> Result<(), String> {
    for (r, d) in rows.iter().enumerate() {
        let at = r * (TOP_K + 1);
        same_words(
            &format!("row {r} ids"),
            &got.ids[at..at + TOP_K + 1],
            &d.ids,
        )?;
        let gw: Vec<u32> = got.weights[at..at + TOP_K + 1]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let dw: Vec<u32> = d.weights.iter().map(|v| v.to_bits()).collect();
        same_words(&format!("row {r} weights"), &gw, &dw)?;
    }
    Ok(())
}

/// The selection of rows of logits is the decode router's, bit for bit.
pub fn route_rows_select(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let eps = fixture.rms_eps()?;
    let mut cases = Vec::new();
    let zero_shared = vec![0f32; K];
    for seed in 0..4u64 {
        let routed = oracle::activations(EXPERTS * K, 0x5E1 + seed, 0.05);
        let shared = oracle::activations(K, 0x5E9 + seed, 0.05);
        cases.push(decoded(h, &routed, &shared, 0x700 + seed, eps)?);
    }
    // Ties: three routed rows equal to one another, so three logits are the
    // same bits and the smaller ids win.
    let mut tied = oracle::activations(EXPERTS * K, 0x5EF, 0.05);
    for e in [77usize, 200] {
        tied.copy_within(3 * K..4 * K, e * K);
    }
    cases.push(decoded(h, &tied, &zero_shared, 0x710, eps)?);
    // Every logit equal: the first eight ids win.
    cases.push(decoded(
        h,
        &vec![0f32; EXPERTS * K],
        &zero_shared,
        0x711,
        eps,
    )?);
    h.clear_fault()?;
    let refs: Vec<&Decoded> = cases.iter().collect();
    let got = select_run(h, &refs)?;
    if let Some(f) = got.fault {
        return Err(format!("clean logits raised {f}"));
    }
    same_selection(&got, &refs)?;
    // A logit that is not finite: the decode router's own fallback.
    let routed = oracle::activations(EXPERTS * K, 0x5F0, 0.05);
    let mut nan_shared = vec![0.01f32; K];
    nan_shared[10] = f32::NAN;
    let bad = decoded(h, &routed, &nan_shared, 0x720, eps)?;
    expect_site(
        h.fault()?,
        "the decode router on a NaN shared row",
        FaultSite::Router,
    )?;
    let got = select_run(h, &[&bad])?;
    expect_site(got.fault, "the selection on a NaN logit", FaultSite::Router)?;
    same_selection(&got, &[&bad])?;
    Ok(Verdict::Noted(format!(
        "{} rows (random, tied, all-equal) are the decode router's ids and weights bit for bit; a NaN logit is its fallback",
        refs.len()
    )))
}

// -------------------------------------------------------------------- table

struct Table {
    sorted: Vec<u32>,
    inv: Vec<u32>,
    tiles: Vec<u32>,
    fault: Option<Fault>,
}

fn table_run(h: &Harness<'_>, ids: &[u32]) -> Result<Table, String> {
    let slots = ids.len();
    let ids_b = upload(h, ids)?;
    let mut sorted = upload(h, &vec![0xFFFF_FFFFu32; slots])?;
    let mut inv = upload(h, &vec![0xFFFF_FFFFu32; slots])?;
    let words = route_table_words(slots, EXPERTS);
    let mut tiles = upload(h, &vec![0xFFFF_FFFFu32; words])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.route_table(
            encoder,
            RouteTableParams {
                slots,
                per_row: PER_ROW,
                experts: EXPERTS,
                ids: ids_b.window(Elems(0), Elems(slots))?,
                sorted: sorted.window_mut(Elems(0), Elems(slots))?,
                inv: inv.window_mut(Elems(0), Elems(slots))?,
                tiles: tiles.window_mut(Elems(0), Elems(words))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    Ok(Table {
        sorted: read(h, &sorted, slots)?,
        inv: read(h, &inv, slots)?,
        tiles: read(h, &tiles, words)?,
        fault: h.fault()?,
    })
}

/// The slots of `rows` rows over the experts the picks name: row `r` picks
/// `picks(r)` and the shared expert last. A row's picks are distinct.
pub struct Picks {
    pub ids: Vec<u32>,
}

/// The ids of a pass: a hot expert every row picks (so its group takes more
/// than one tile), the rest pseudo-random and distinct within a row.
pub fn picks(rows: usize, hot: Option<u32>, seed: u64) -> Picks {
    let mut rng = oracle::Rng::new(seed);
    let mut ids = Vec::with_capacity(rows * PER_ROW);
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::new();
        if let Some(e) = hot {
            row.push(e);
        }
        while row.len() < TOP_K {
            let e = (rng.next() % EXPERTS as u64) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
        ids.push(EXPERTS as u32);
    }
    Picks { ids }
}

/// The host rule: a slot's group is its id; a group's places are the run
/// starting at the sum of the counts before it; its tiles cut the run in
/// ROUTE_TILE-sized pieces, in group order.
struct Rule {
    start: Vec<usize>,
    count: Vec<usize>,
    tiles: Vec<[u32; 3]>,
}

fn rule(ids: &[u32]) -> Rule {
    let groups = EXPERTS + 1;
    let mut count = vec![0usize; groups];
    for &id in ids {
        count[id as usize] += 1;
    }
    let mut start = vec![0usize; groups];
    let mut tiles = Vec::new();
    let mut at = 0usize;
    for g in 0..groups {
        start[g] = at;
        let mut j = 0;
        while j * ROUTE_TILE < count[g] {
            tiles.push([
                g as u32,
                (at + j * ROUTE_TILE) as u32,
                (count[g] - j * ROUTE_TILE).min(ROUTE_TILE) as u32,
            ]);
            j += 1;
        }
        at += count[g];
    }
    Rule {
        start,
        count,
        tiles,
    }
}

fn check_table(ids: &[u32], t: &Table, what: &str) -> Result<usize, String> {
    let slots = ids.len();
    let r = rule(ids);
    if let Some(f) = t.fault {
        return Err(format!("{what}: a clean table raised {f}"));
    }
    for (p, &s) in t.sorted.iter().enumerate() {
        if s as usize >= slots {
            return Err(format!("{what}: place {p} holds {s:#x}, not a slot"));
        }
        if t.inv[s as usize] as usize != p {
            return Err(format!(
                "{what}: place {p} holds slot {s}, whose place is {}",
                t.inv[s as usize]
            ));
        }
        let g = ids[s as usize] as usize;
        if p < r.start[g] || p >= r.start[g] + r.count[g] {
            return Err(format!(
                "{what}: slot {s} of group {g} sits at place {p}, outside {}..{}",
                r.start[g],
                r.start[g] + r.count[g]
            ));
        }
    }
    let n = t.tiles[0] as usize;
    if n != r.tiles.len() {
        return Err(format!(
            "{what}: {n} tiles, the host rule's {}",
            r.tiles.len()
        ));
    }
    for (i, want) in r.tiles.iter().enumerate() {
        let got = &t.tiles[4 * (1 + i)..4 * (1 + i) + 4];
        if got[..3] != want[..] {
            return Err(format!(
                "{what}: tile {i} is {:?}, the host rule's {want:?}",
                &got[..3]
            ));
        }
    }
    let cap = route_table_words(slots, EXPERTS);
    if t.tiles[4 * (1 + n)..cap].iter().any(|&w| w != 0xFFFF_FFFF) {
        return Err(format!("{what}: a word past the last tile was written"));
    }
    Ok(n)
}

/// The slots are sorted by group and cut into tiles by the host rule.
pub fn route_table(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let mut parts = Vec::new();
    for (rows, hot) in [
        (1usize, None),
        (9, None),
        (40, Some(5u32)),
        (130, Some(255)),
    ] {
        let ids = picks(rows, hot, 0x7AB + rows as u64).ids;
        let t = table_run(h, &ids)?;
        let tiles = check_table(&ids, &t, &format!("{rows} rows"))?;
        parts.push(format!("{rows}:{tiles}"));
    }
    // An id outside its slot's range raises the expert-id site, is placed in
    // group 0, and moves no other slot's group.
    let mut ids = picks(9, None, 0x7AC).ids;
    ids[4] = EXPERTS as u32 + 3;
    ids[PER_ROW + 8] = 17;
    let t = table_run(h, &ids)?;
    expect_site(
        t.fault,
        "a slot with an id outside its range",
        FaultSite::ExpertId,
    )?;
    let mut fixed = ids.clone();
    fixed[4] = 0;
    fixed[PER_ROW + 8] = 0;
    let mut clean = t;
    clean.fault = None;
    for (p, &s) in clean.sorted.iter().enumerate() {
        if clean.inv[s as usize] as usize != p {
            return Err(format!("a raised table: slot {s} does not invert"));
        }
    }
    let r = rule(&fixed);
    for (p, &s) in clean.sorted.iter().enumerate() {
        let g = fixed[s as usize] as usize;
        if p < r.start[g] || p >= r.start[g] + r.count[g] {
            return Err(format!("a raised table: slot {s} sits outside group {g}"));
        }
    }
    Ok(Verdict::Noted(format!(
        "rows:tiles {}: every slot inside its group, places invert, tiles are the host rule's; a bad id raises the expert-id site",
        parts.join(", ")
    )))
}

// ------------------------------------------------------------- grouped gemm

/// A family's stacks and shared tensors, for the pair and the down.
struct Experts {
    quant: Quant,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
}

fn stack(
    fix: &Fixture,
    name: &str,
    ty: GgmlType,
    cols: usize,
    rows: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for e in 0..EXPERTS {
        out.extend_from_slice(fix.expert(name, ty, cols, rows, e)?);
    }
    Ok(out)
}

fn experts_of(fix: &Fixture, quant: Quant) -> Result<Experts, String> {
    let ty = quant.ggml();
    Ok(Experts {
        quant,
        gate: stack(fix, "blk.0.ffn_gate_exps.weight", ty, K, EXPERT_K)?,
        up: stack(fix, "blk.0.ffn_up_exps.weight", ty, K, EXPERT_K)?,
        down: stack(fix, "blk.0.ffn_down_exps.weight", ty, EXPERT_K, HIDDEN)?,
        shared_gate: fix
            .rows("blk.0.ffn_gate_shexp.weight", ty, K, EXPERT_K)?
            .to_vec(),
        shared_up: fix
            .rows("blk.0.ffn_up_shexp.weight", ty, K, EXPERT_K)?
            .to_vec(),
        shared_down: fix
            .rows("blk.0.ffn_down_shexp.weight", ty, EXPERT_K, HIDDEN)?
            .to_vec(),
    })
}

impl Experts {
    /// The bytes of expert `e` of a stack (or the shared tensor for the
    /// table's last group), `one` bytes each.
    fn of<'a>(stack: &'a [u8], shared: &'a [u8], e: usize) -> &'a [u8] {
        if e == EXPERTS {
            shared
        } else {
            let one = stack.len() / EXPERTS;
            &stack[e * one..(e + 1) * one]
        }
    }
}

/// The scene of a grouped launch: the pass's rows, their ids and route
/// weights.
struct Scene {
    rows: usize,
    ids: Vec<u32>,
    route_w: Vec<f32>,
    x: Vec<f32>,
}

fn scene(rows: usize, hot: Option<u32>, seed: u64) -> Scene {
    let ids = picks(rows, hot, seed).ids;
    let mut rng = oracle::Rng::new(seed ^ 0x5CE);
    let route_w = ids
        .iter()
        .map(|_| 0.1 + 0.9 * (rng.next() >> 40) as f32 / (1u64 << 24) as f32)
        .collect();
    let mut x = Vec::with_capacity(rows * K);
    for r in 0..rows {
        x.extend(oracle::activations(K, seed + r as u64, 1.0));
    }
    Scene {
        rows,
        ids,
        route_w,
        x,
    }
}

/// The bytes of one uploaded tensor.
fn bytes(h: &Harness<'_>, data: &[u8]) -> Result<MetalBuffer<u8>, String> {
    upload(h, data)
}

struct Grouped {
    /// The pair's outputs, one row of EXPERT_K a place.
    h: Vec<f32>,
    /// The down's outputs, one row of HIDDEN a place.
    y: Vec<f32>,
    table: Table,
    /// What the pair launch and the down launch raised.
    pair_fault: Option<Fault>,
    down_fault: Option<Fault>,
}

fn grouped_run(
    h: &Harness<'_>,
    ex: &Experts,
    sc: &Scene,
    operand: Operand,
) -> Result<Grouped, String> {
    let slots = sc.rows * PER_ROW;
    let table = table_run(h, &sc.ids)?;
    if let Some(f) = table.fault {
        return Err(format!("the table raised {f}"));
    }
    let tiles_b = upload(h, &table.tiles)?;
    let sorted_b = upload(h, &table.sorted)?;
    let w_b = upload(h, &sc.route_w)?;
    let x_b = upload(h, &sc.x)?;
    let gate = bytes(h, &ex.gate)?;
    let up = bytes(h, &ex.up)?;
    let down = bytes(h, &ex.down)?;
    let sgate = bytes(h, &ex.shared_gate)?;
    let sup = bytes(h, &ex.shared_up)?;
    let sdown = bytes(h, &ex.shared_down)?;
    let mut hid = upload(h, &vec![sentinel(); slots * EXPERT_K])?;
    let mut out = upload(h, &vec![sentinel(); slots * HIDDEN])?;
    let max_tiles = route_tiles(slots, EXPERTS);
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.gemm_routed(
            encoder,
            RoutedParams {
                quant: ex.quant,
                kind: RoutedKind::Pair,
                operand,
                cols: K,
                rows: EXPERT_K,
                experts: EXPERTS,
                stack: gate.window(Elems(0), Elems(ex.gate.len()))?,
                stack_up: Some(up.window(Elems(0), Elems(ex.up.len()))?),
                shared: sgate.window(Elems(0), Elems(ex.shared_gate.len()))?,
                shared_up: Some(sup.window(Elems(0), Elems(ex.shared_up.len()))?),
                x: x_b.window(Elems(0), Elems(sc.x.len()))?,
                x_rows: sc.rows,
                x_stride: K,
                gather: true,
                y: hid.window_mut(Elems(0), Elems(slots * EXPERT_K))?,
                y_stride: EXPERT_K,
                slots,
                max_tiles,
                sorted: sorted_b.window(Elems(0), Elems(slots))?,
                route_w: w_b.window(Elems(0), Elems(slots))?,
                tiles: tiles_b.window(Elems(0), Elems(table.tiles.len()))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let pair_fault = h.fault()?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.gemm_routed(
            encoder,
            RoutedParams {
                quant: ex.quant,
                kind: RoutedKind::Plain,
                operand,
                cols: EXPERT_K,
                rows: HIDDEN,
                experts: EXPERTS,
                stack: down.window(Elems(0), Elems(ex.down.len()))?,
                stack_up: None,
                shared: sdown.window(Elems(0), Elems(ex.shared_down.len()))?,
                shared_up: None,
                x: hid.window(Elems(0), Elems(slots * EXPERT_K))?,
                x_rows: slots,
                x_stride: EXPERT_K,
                gather: false,
                y: out.window_mut(Elems(0), Elems(slots * HIDDEN))?,
                y_stride: HIDDEN,
                slots,
                max_tiles,
                sorted: sorted_b.window(Elems(0), Elems(slots))?,
                route_w: w_b.window(Elems(0), Elems(slots))?,
                tiles: tiles_b.window(Elems(0), Elems(table.tiles.len()))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let down_fault = h.fault()?;
    Ok(Grouped {
        h: read(h, &hid, slots * EXPERT_K)?,
        y: read(h, &out, slots * HIDDEN)?,
        table,
        pair_fault,
        down_fault,
    })
}

/// Σ|w| per output row of one expert's tensor, cached.
struct AbsSums {
    quant: Quant,
    cache: HashMap<(u8, usize), Vec<f64>>,
}

impl AbsSums {
    fn of(
        &mut self,
        tensor: u8,
        e: usize,
        bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<&[f64], String> {
        if !self.cache.contains_key(&(tensor, e)) {
            let sums = weight_abs_sums(self.quant, bytes, rows, cols)?;
            self.cache.insert((tensor, e), sums);
        }
        Ok(&self.cache[&(tensor, e)])
    }
}

/// The error of a dot of the dense GEMM's band, as the width of one output:
/// `factor * magnitude + subnormal terms`.
fn dot_width(factor: f64, ops: usize, sub: f64, magnitude: f64, x_abs: f64, w_abs: f64) -> f64 {
    factor * magnitude * (1.0 + U)
        + sub * (x_abs + w_abs)
        + ops as f64 * f64::from(f32::MIN_POSITIVE)
}

/// Check every `stride`th place (and every place of the shared group and of
/// the hot expert) of a grouped run against the oracle: the pair, then the
/// down on the pair's own outputs. Returns the worst error over its band of
/// each.
fn check_grouped(
    ex: &Experts,
    sc: &Scene,
    run: &Grouped,
    operand: Operand,
    stride: usize,
    every: Option<u32>,
) -> Result<(f64, f64), String> {
    let slots = sc.rows * PER_ROW;
    let (f_pair, ops_pair) = gemm_band(ex.quant, K, operand);
    let (f_down, ops_down) = gemm_band(ex.quant, EXPERT_K, operand);
    // The f32 arm has no cast, so no subnormal term; its band is the dense
    // GEMM's own (`gemm_band`), and the subnormal terms are the f16 arm's.
    let sub = match operand {
        Operand::F16 => HALF_SUBNORMAL,
        Operand::F32 => 0.0,
    };
    let mut abs = AbsSums {
        quant: ex.quant,
        cache: HashMap::new(),
    };
    let mut worst_pair = 0f64;
    let mut worst_down = 0f64;
    for p in 0..slots {
        let s = run.table.sorted[p] as usize;
        let e = sc.ids[s] as usize;
        if !(p % stride == 0 || e == EXPERTS || every == Some(e as u32)) {
            continue;
        }
        let row = s / PER_ROW;
        let x: Vec<f64> = sc.x[row * K..(row + 1) * K]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let x_abs: f64 = x.iter().map(|v| v.abs()).sum();
        let gate = Experts::of(&ex.gate, &ex.shared_gate, e);
        let up = Experts::of(&ex.up, &ex.shared_up, e);
        let rg = oracle::reference(ex.quant, gate, EXPERT_K, K, &x)?;
        let ru = oracle::reference(ex.quant, up, EXPERT_K, K, &x)?;
        let wg = abs.of(0, e, gate, EXPERT_K, K)?.to_vec();
        let wu = abs.of(1, e, up, EXPERT_K, K)?.to_vec();
        let w = f64::from(sc.route_w[s]);
        for o in 0..EXPERT_K {
            let (g, u) = (rg.y[o], ru.y[o]);
            let sg = oracle::silu(g);
            let g_err = dot_width(f_pair, ops_pair, sub, rg.magnitude[o], x_abs, wg[o]);
            let u_err = dot_width(f_pair, ops_pair, sub, ru.magnitude[o], x_abs, wu[o]);
            let silu_err =
                1.1 * g_err + 2.0 * U * (g.abs() + g_err) + 3.0 * U * (g.abs() + sg.abs());
            let prod =
                silu_err * u.abs() + sg.abs() * u_err + silu_err * u_err + U * (sg * u).abs();
            let want = sg * u * w;
            let width = w * prod + U * want.abs() + 1e-30;
            let got = f64::from(run.h[p * EXPERT_K + o]);
            let err = (got - want).abs();
            if !got.is_finite() || err > width {
                return Err(format!(
                    "pair place {p} (slot {s}, group {e}) output {o}: got {got:e}, want {want:e}, error {err:e} over the band {width:e}"
                ));
            }
            worst_pair = worst_pair.max(err / width);
        }
        // The down, on the kernel's own pair outputs at this place.
        let hx: Vec<f64> = run.h[p * EXPERT_K..(p + 1) * EXPERT_K]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let h_abs: f64 = hx.iter().map(|v| v.abs()).sum();
        let down = Experts::of(&ex.down, &ex.shared_down, e);
        let rd = oracle::reference(ex.quant, down, HIDDEN, EXPERT_K, &hx)?;
        let wd = abs.of(2, e, down, HIDDEN, EXPERT_K)?.to_vec();
        for n in 0..HIDDEN {
            let width = dot_width(f_down, ops_down, sub, rd.magnitude[n], h_abs, wd[n]);
            let got = f64::from(run.y[p * HIDDEN + n]);
            let err = (got - rd.y[n]).abs();
            if !got.is_finite() || err > width {
                return Err(format!(
                    "down place {p} (slot {s}, group {e}) output {n}: got {got:e}, want {:e}, error {err:e} over the band {width:e}",
                    rd.y[n]
                ));
            }
            worst_down = worst_down.max(err / width);
        }
    }
    Ok((worst_pair, worst_down))
}

/// An activation past the half range raises the quantizer's site and makes
/// NaN of every output of the slots that read its row; the other rows' slots
/// are finite.
fn range_fault(h: &Harness<'_>, ex: &Experts) -> Result<(), String> {
    let mut sc = scene(5, None, 0x6B0);
    let bad_row = 2usize;
    sc.x[bad_row * K + 100] = 7.0e4;
    let run = grouped_run(h, ex, &sc, Operand::F16)?;
    expect_site(
        run.pair_fault,
        "the pair on an activation past the half range",
        FaultSite::QuantColumn,
    )?;
    for (p, &s) in run.table.sorted.iter().enumerate() {
        let row = s as usize / PER_ROW;
        let out = &run.h[p * EXPERT_K..(p + 1) * EXPERT_K];
        if row == bad_row && !out.iter().all(|v| v.is_nan()) {
            return Err(format!(
                "place {p} reads the bad row and is not NaN throughout"
            ));
        }
        if row != bad_row && out.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "place {p} of row {row} is not finite beside the bad row"
            ));
        }
    }
    Ok(())
}

fn grouped_quant(
    h: &Harness<'_>,
    fix: &Fixture,
    quant: Quant,
    operands: &[Operand],
) -> Result<String, String> {
    let ex = experts_of(fix, quant)?;
    let mut parts = Vec::new();
    for &operand in operands {
        // 40 rows: every row picks the hot expert, so its group and the
        // shared group both take two tiles (32 + 8).
        let sc = scene(40, Some(5), 0x6A0);
        let run = grouped_run(h, &ex, &sc, operand)?;
        for (what, fault) in [("pair", run.pair_fault), ("down", run.down_fault)] {
            if let Some(f) = fault {
                return Err(format!("the clean {what} launch raised {f}"));
            }
        }
        let (pair, down) = check_grouped(&ex, &sc, &run, operand, 7, Some(5))?;
        parts.push(format!(
            "{} {operand:?} pair {pair:.3} down {down:.3}",
            quant.name()
        ));
    }
    if operands.contains(&Operand::F16) {
        range_fault(h, &ex)?;
        parts.push(format!(
            "{} an activation past the half range is the fault and NaN",
            quant.name()
        ));
    }
    Ok(parts.join(", "))
}

/// The pair and the down over the table's tiles are the oracle's, for every
/// (row, slot), the shared expert's group and a group of two tiles included.
pub fn grouped_gemm(h: &Harness<'_>, fixture: &Fixture, mlx: &Result<Fixture, String>) -> Clause {
    let mut notes = vec![grouped_quant(
        h,
        fixture,
        Quant::Q4K,
        &[Operand::F16, Operand::F32],
    )?];
    match mlx {
        Ok(m) => notes.push(grouped_quant(h, m, Quant::A4, &[Operand::F16])?),
        Err(e) => return Err(format!("the A4 fixture: {e}")),
    }
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}",
        notes.join(", ")
    )))
}

// ------------------------------------------------------------------ combine

/// `x += y[inv[row][0]] + ... + y[inv[row][8]]` in slot order, the sum started
/// at zero, bit for bit against the same f32 sum.
pub fn combine(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let rows = 33usize;
    let slots = rows * PER_ROW;
    let (y_stride, x_stride) = (HIDDEN + 4, HIDDEN + 8);
    let mut rng = oracle::Rng::new(0xC0B);
    let mut inv: Vec<u32> = (0..slots as u32).collect();
    for i in (1..slots).rev() {
        inv.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let mut y = vec![sentinel(); slots * y_stride];
    for p in 0..slots {
        for (c, v) in oracle::activations(HIDDEN, 0xC0C + p as u64, 3.0)
            .iter()
            .enumerate()
        {
            y[p * y_stride + c] = *v;
        }
    }
    let mut x = vec![sentinel(); rows * x_stride];
    for r in 0..rows {
        for (c, v) in oracle::activations(HIDDEN, 0xC0D + r as u64, 5.0)
            .iter()
            .enumerate()
        {
            x[r * x_stride + c] = *v;
        }
    }
    let mut want = x.clone();
    for r in 0..rows {
        for c in 0..HIDDEN {
            let mut acc = 0f32;
            for s in 0..PER_ROW {
                acc += y[inv[r * PER_ROW + s] as usize * y_stride + c];
            }
            want[r * x_stride + c] = x[r * x_stride + c] + acc;
        }
    }
    let y_b = upload(h, &y)?;
    let inv_b = upload(h, &inv)?;
    let mut x_b = upload(h, &x)?;
    h.encode(|encoder| {
        h.kernels.combine_rows(
            encoder,
            CombineParams {
                rows,
                per_row: PER_ROW,
                cols: HIDDEN,
                y: y_b.window(Elems(0), Elems(y.len()))?,
                y_rows: slots,
                y_stride,
                inv: inv_b.window(Elems(0), Elems(slots))?,
                x: x_b.window_mut(Elems(0), Elems(x.len()))?,
                x_stride,
            },
        )
    })?;
    let got = read(h, &x_b, x.len())?;
    let g: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let w: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    same_words("combine", &g, &w)?;
    Ok(Verdict::Noted(format!(
        "{rows} rows of {PER_ROW} slots through a shuffled place list are the f32 sum in slot order, bit for bit; the stride gaps are untouched"
    )))
}

// --------------------------------------------------------------- headnorm

/// The numerator times each head's scale against f64; a head the norm cannot
/// take is NaN alone and raises the linear-gate site.
pub fn headnorm_rows(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let rows = 9usize;
    let width = VALUE_HEADS * HEAD_COLS;
    let (g_stride, p_stride) = (width + 4, VALUE_HEADS * PARTIALS_PER_HEAD + 4);
    let eps = 1.0e-6f32;
    let mut g = vec![sentinel(); rows * g_stride];
    let mut partials = vec![sentinel(); rows * p_stride];
    for r in 0..rows {
        for (c, v) in oracle::activations(width, 0xE00 + r as u64, 2.0)
            .iter()
            .enumerate()
        {
            g[r * g_stride + c] = *v;
        }
        let mut rng = oracle::Rng::new(0xE10 + r as u64);
        for p in 0..VALUE_HEADS * PARTIALS_PER_HEAD {
            partials[r * p_stride + p] = 0.5 + 40.0 * (rng.signed().abs());
        }
    }
    let run = |partials: &[f32]| -> Result<(Vec<f32>, Option<Fault>), String> {
        let mut g_m = upload(h, &g)?;
        let p_b = upload(h, partials)?;
        h.clear_fault()?;
        h.encode(|encoder| {
            h.kernels.headnorm_rows(
                encoder,
                HeadNormRowsParams {
                    rows,
                    g: g_m.window_mut(Elems(0), Elems(g.len()))?,
                    g_stride,
                    partials: p_b.window(Elems(0), Elems(partials.len()))?,
                    partials_stride: p_stride,
                    eps,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        })?;
        Ok((read(h, &g_m, g.len())?, h.fault()?))
    };
    let (got, fault) = run(&partials)?;
    if let Some(f) = fault {
        return Err(format!("clean heads raised {f}"));
    }
    for r in 0..rows {
        for head in 0..VALUE_HEADS {
            let sum: f64 = (0..PARTIALS_PER_HEAD)
                .map(|j| f64::from(partials[r * p_stride + head * PARTIALS_PER_HEAD + j]))
                .sum();
            let scale = 1.0 / (sum / HEAD_COLS as f64 + f64::from(eps)).sqrt();
            for c in 0..HEAD_COLS {
                let at = r * g_stride + head * HEAD_COLS + c;
                let want = f64::from(g[at]) * scale;
                let err = (f64::from(got[at]) - want).abs();
                if err > 8.0 * U * want.abs() + f64::from(f32::MIN_POSITIVE) {
                    return Err(format!(
                        "row {r} head {head} column {c}: got {:e}, want {want:e}",
                        got[at]
                    ));
                }
            }
        }
        let pad = &got[r * g_stride + width..(r + 1) * g_stride];
        if pad.iter().any(|v| v.to_bits() != SENTINEL_BITS) {
            return Err(format!("row {r}: a word past the row was written"));
        }
    }
    let mut nan = partials.clone();
    nan[3 * p_stride + 5 * PARTIALS_PER_HEAD + 2] = f32::NAN;
    let (got, fault) = run(&nan)?;
    expect_site(fault, "a NaN partial", FaultSite::LinearGate)?;
    for r in 0..rows {
        for head in 0..VALUE_HEADS {
            let span = r * g_stride + head * HEAD_COLS..r * g_stride + (head + 1) * HEAD_COLS;
            let all_nan = got[span.clone()].iter().all(|v| v.is_nan());
            if (r, head) == (3, 5) {
                if !all_nan {
                    return Err("the NaN head is not NaN throughout".to_string());
                }
            } else if got[span].iter().any(|v| v.is_nan()) {
                return Err(format!(
                    "row {r} head {head} became NaN beside the bad head"
                ));
            }
        }
    }
    Ok(Verdict::Noted(format!(
        "{rows} rows of {VALUE_HEADS} heads are g / sqrt(mean + eps) within 8 U; a NaN head is NaN alone and raises the linear-gate site"
    )))
}

// ----------------------------------------------------------------- f32 gemm

/// The router's GEMM over f32 weight rows against f64, an edge tile on each
/// side included.
pub fn gemm_f32w(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let (rows, acts) = (300usize, 70usize);
    let (x_stride, y_stride) = (K + 4, rows + 4);
    let w = oracle::activations(rows * K, 0xF32, 0.05);
    let mut x = vec![sentinel(); acts * x_stride];
    for r in 0..acts {
        for (c, v) in oracle::activations(K, 0xF33 + r as u64, 1.0)
            .iter()
            .enumerate()
        {
            x[r * x_stride + c] = *v;
        }
    }
    let w_b = upload(h, &w)?;
    let x_b = upload(h, &x)?;
    let mut y_b = upload(h, &vec![sentinel(); acts * y_stride])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.gemm_f32(
            encoder,
            F32GemmParams {
                rows,
                cols: K,
                acts,
                weights: w_b.window(Elems(0), Elems(w.len()))?,
                x: x_b.window(Elems(0), Elems(x.len()))?,
                x_stride,
                y: y_b.window_mut(Elems(0), Elems(acts * y_stride))?,
                y_stride,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    common::expect_clean(h, "the f32 gemm")?;
    let got = read(h, &y_b, acts * y_stride)?;
    let factor = oracle::gamma_bound(1 + K / 8 + 8);
    let floor = (K / 8 + 9) as f64 * f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for r in 0..acts {
        for n in 0..rows {
            let (mut dot, mut mag) = (0f64, 0f64);
            for c in 0..K {
                let p = f64::from(w[n * K + c]) * f64::from(x[r * x_stride + c]);
                dot += p;
                mag += p.abs();
            }
            let width = factor * mag * (1.0 + U) + floor;
            let v = f64::from(got[r * y_stride + n]);
            let err = (v - dot).abs();
            if !v.is_finite() || err > width {
                return Err(format!(
                    "row {r} column {n}: got {v:e}, want {dot:e}, error {err:e} over the band {width:e}"
                ));
            }
            worst = worst.max(err / width);
        }
        let pad = &got[r * y_stride + rows..(r + 1) * y_stride];
        if pad.iter().any(|v| v.to_bits() != SENTINEL_BITS) {
            return Err(format!(
                "row {r}: a word past the row's {rows} columns was written"
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "{acts} rows by {rows} weight rows, worst error over its band {worst:.3}"
    )))
}

// ----------------------------------------------------------------- refusals

/// The launch modules refuse a shape or a window that is not the kernel's,
/// by name, and encode nothing.
pub fn route_refusals(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let slots = 2 * PER_ROW;
    let ids_b = upload(h, &[0u32; 2 * PER_ROW])?;
    let mut sorted = upload(h, &[0u32; 2 * PER_ROW])?;
    let mut inv = upload(h, &[0u32; 2 * PER_ROW])?;
    let words = route_table_words(slots, EXPERTS);
    let mut tiles = upload(h, &vec![0u32; words])?;
    let logits_b = upload(h, &[0f32; 2 * LOGIT_STRIDE])?;
    let mut sel_ids = upload(h, &[0u32; 2 * (TOP_K + 1)])?;
    let mut sel_w = upload(h, &[0f32; 2 * (TOP_K + 1)])?;
    let mut checks = 0usize;
    let mut expect =
        |what: &str, needle: &str, got: Result<(), bloomery_device::DeviceError>| match got {
            Err(e) if e.to_string().contains(needle) => {
                checks += 1;
                Ok(())
            }
            Err(e) => Err(format!("{what}: refused as {e}, not as {needle:?}")),
            Ok(()) => Err(format!("{what}: accepted")),
        };
    let mut recording = h.stream.begin().map_err(why)?;
    {
        let mut enc = recording.compute_encoder().map_err(why)?;
        let table = |enc: &mut bloomery_metal::encode::ComputeEncoder<'_>,
                     sorted: &mut MetalBuffer<u32>,
                     inv: &mut MetalBuffer<u32>,
                     tiles: &mut MetalBuffer<u32>,
                     slots: usize,
                     per_row: usize,
                     experts: usize,
                     tile_words: usize|
         -> Result<(), bloomery_device::DeviceError> {
            h.kernels.route_table(
                enc,
                RouteTableParams {
                    slots,
                    per_row,
                    experts,
                    ids: ids_b.window(Elems(0), Elems(slots))?,
                    sorted: sorted.window_mut(Elems(0), Elems(slots))?,
                    inv: inv.window_mut(Elems(0), Elems(slots))?,
                    tiles: tiles.window_mut(Elems(0), Elems(tile_words))?,
                    fault: h.device.fault_word(),
                    layer: LAYER,
                },
            )
        };
        expect(
            "slots that are not whole rows",
            "not whole rows",
            table(
                &mut enc,
                &mut sorted,
                &mut inv,
                &mut tiles,
                slots - 1,
                PER_ROW,
                EXPERTS,
                words,
            ),
        )?;
        expect(
            "an expert count that is not the router's",
            "routed experts",
            table(
                &mut enc,
                &mut sorted,
                &mut inv,
                &mut tiles,
                slots,
                PER_ROW,
                EXPERTS - 1,
                words,
            ),
        )?;
        expect(
            "a table of the wrong length",
            "tiles window",
            table(
                &mut enc,
                &mut sorted,
                &mut inv,
                &mut tiles,
                slots,
                PER_ROW,
                EXPERTS,
                words - 4,
            ),
        )?;
        for (what, needle, rows, top_k, stride) in [
            ("no rows", "rows of 0", 0usize, TOP_K, LOGIT_STRIDE),
            (
                "a top_k past the router's",
                "top_k",
                2,
                TOP_K + 1,
                LOGIT_STRIDE,
            ),
            (
                "a stride shorter than the row",
                "stride",
                2,
                TOP_K,
                router::ROWS - 1,
            ),
        ] {
            expect(
                what,
                needle,
                h.kernels.route_rows_select(
                    &mut enc,
                    RouteSelectParams {
                        rows,
                        top_k,
                        logits: logits_b
                            .window(Elems(0), Elems(2 * LOGIT_STRIDE))
                            .map_err(why)?,
                        logit_stride: stride,
                        ids: sel_ids
                            .window_mut(Elems(0), Elems(2 * (TOP_K + 1)))
                            .map_err(why)?,
                        weights: sel_w
                            .window_mut(Elems(0), Elems(2 * (TOP_K + 1)))
                            .map_err(why)?,
                        fault: h.device.fault_word(),
                        layer: LAYER,
                    },
                ),
            )?;
        }
        enc.end();
    }
    h.stream.submit(recording).map_err(why)?;
    Ok(Verdict::Noted(format!(
        "{checks} shapes and windows are refused by name before anything is encoded"
    )))
}

pub fn table<'a>(
    ctx: &'a super::ClauseCtx<'a>,
    mlx: &'a Result<Fixture, String>,
) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("gemm_f32w", Box::new(|| ctx.with(gemm_f32w))),
        (
            "route_rows_select",
            Box::new(|| ctx.with(route_rows_select)),
        ),
        ("route_table", Box::new(|| ctx.with(route_table))),
        (
            "grouped_gemm",
            Box::new(|| match ctx.fixture {
                Ok(fixture) => grouped_gemm(ctx.harness, fixture, mlx),
                Err(e) => Err(e.clone()),
            }),
        ),
        ("combine", Box::new(|| ctx.with(combine))),
        ("headnorm_rows", Box::new(|| ctx.with(headnorm_rows))),
        ("route_refusals", Box::new(|| ctx.with(route_refusals))),
    ]
}
