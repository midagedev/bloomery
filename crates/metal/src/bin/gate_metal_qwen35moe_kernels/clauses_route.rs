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
//!   f64 oracle's band, the shared expert's group included when the router has
//!   one. `grouped_q6_down` runs the Q6_K down of a file whose down type
//!   varies.
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
use std::ops::Range;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elem, Elems, Fault};
use bloomery_metal::MetalBuffer;
use bloomery_metal::geom::Geometry;
use bloomery_metal::instance::REGISTERED;
use bloomery_metal::kernels::{
    APART_DOWN, APART_GATE_UP, CombineParams, F32GemmParams, HeadNormRowsParams, Kernels, Operand,
    PARTIALS_PER_HEAD, Quant, ROUTE_TILE, RouteSelectParams, RouteTableParams, RoutedKind,
    RoutedParams, route_table_words, route_tiles,
};
use bloomery_metal::library::{GemmPath, Library, sources_of};
use gguf::quant::GgmlType;

use super::clauses_gemm::{HALF_SUBNORMAL, gemm_band, weight_abs_sums};
use super::clauses_router;
use super::common::{self, LAYER};
use super::fixture::Fixture;
use super::oracle::{self, U};
use super::synthetic;
use super::{Harness, SENTINEL_BITS};

/// The numbers the routing clauses size a launch, a scene and an oracle by:
/// the geometry of the row's kernels.
#[derive(Clone, Copy)]
pub struct Dims {
    /// The hidden width every row has: the router's columns and the down's
    /// outputs.
    k: usize,
    /// The routed experts.
    experts: usize,
    /// One when the router has a shared expert's row, else zero.
    shared: usize,
    /// Routed picks a row keeps.
    top_k: usize,
    /// Slots a row holds: the picks and the shared expert.
    per_row: usize,
    /// The expert intermediate width: the pair's rows and the down's columns.
    expert_k: usize,
    /// The rows the router reads.
    rows: usize,
    /// The logits row stride a pass keeps.
    logit_stride: usize,
}

impl Dims {
    /// The ids a row of the selection writes: the picks, and the shared
    /// expert's id when the router has a shared row.
    fn row_ids(&self) -> usize {
        self.top_k + self.shared
    }

    fn of(h: &Harness<'_>) -> Self {
        let g = h.geom();
        Self {
            k: g.hidden,
            experts: g.router.experts,
            shared: g.router.shared,
            top_k: g.experts.routed_ids,
            per_row: g.experts.slots,
            expert_k: g.experts.k,
            rows: g.router.rows,
            logit_stride: g.router.logit_stride,
        }
    }
}

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
    let k = h.geom().hidden;
    let x = oracle::activations(k, seed, 1.0);
    let gain = vec![1.0f32; k];
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

/// Words past the selection's ids and weights, left as they were: a row
/// written at a stride wider than the router's lands in them, not in the next
/// buffer.
const SELECT_GUARD: usize = 64;

fn select_run(h: &Harness<'_>, rows: &[&Decoded]) -> Result<Selected, String> {
    let d = Dims::of(h);
    let n = rows.len();
    let written = n * d.row_ids();
    let mut logits = vec![sentinel(); n * d.logit_stride];
    for (r, row) in rows.iter().enumerate() {
        logits[r * d.logit_stride..r * d.logit_stride + d.rows].copy_from_slice(&row.logits);
    }
    let logits_b = upload(h, &logits)?;
    let mut ids = upload(h, &vec![0xFFFF_FFFFu32; written + SELECT_GUARD])?;
    let mut weights = upload(h, &vec![sentinel(); written + SELECT_GUARD])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.route_rows_select(
            encoder,
            RouteSelectParams {
                rows: n,
                top_k: d.top_k,
                logits: logits_b.window(Elems(0), Elems(logits.len()))?,
                logit_stride: d.logit_stride,
                ids: ids.window_mut(Elems(0), Elems(written))?,
                weights: weights.window_mut(Elems(0), Elems(written))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut ids = read(h, &ids, written + SELECT_GUARD)?;
    let mut weights = read(h, &weights, written + SELECT_GUARD)?;
    if ids[written..].iter().any(|&id| id != 0xFFFF_FFFF)
        || weights[written..]
            .iter()
            .any(|w| w.to_bits() != SENTINEL_BITS)
    {
        return Err(format!(
            "the selection wrote past its {written} ids and weights ({n} rows of {})",
            d.row_ids()
        ));
    }
    ids.truncate(written);
    weights.truncate(written);
    Ok(Selected {
        ids,
        weights,
        fault: h.fault()?,
    })
}

fn same_selection(d: Dims, got: &Selected, rows: &[&Decoded]) -> Result<(), String> {
    for (r, row) in rows.iter().enumerate() {
        let at = r * d.row_ids();
        same_words(
            &format!("row {r} ids"),
            &got.ids[at..at + d.row_ids()],
            &row.ids,
        )?;
        let gw: Vec<u32> = got.weights[at..at + d.row_ids()]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let dw: Vec<u32> = row.weights.iter().map(|v| v.to_bits()).collect();
        same_words(&format!("row {r} weights"), &gw, &dw)?;
    }
    Ok(())
}

/// The selection of rows of logits is the decode router's, bit for bit.
pub fn route_rows_select(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let d = Dims::of(h);
    let eps = fixture.rms_eps()?;
    let mut cases = Vec::new();
    let zero_shared = vec![0f32; d.k];
    for seed in 0..4u64 {
        let routed = oracle::activations(d.experts * d.k, 0x5E1 + seed, 0.05);
        let shared = oracle::activations(d.k, 0x5E9 + seed, 0.05);
        cases.push(decoded(h, &routed, &shared, 0x700 + seed, eps)?);
    }
    // Ties: three routed rows equal to one another, so three logits are the
    // same bits and the smaller ids win. The two copies' rows wrap into a
    // router of fewer experts.
    let mut tied = oracle::activations(d.experts * d.k, 0x5EF, 0.05);
    for e in [77usize % d.experts, 200 % d.experts] {
        tied.copy_within(3 * d.k..4 * d.k, e * d.k);
    }
    cases.push(decoded(h, &tied, &zero_shared, 0x710, eps)?);
    // Every logit equal: the first eight ids win.
    cases.push(decoded(
        h,
        &vec![0f32; d.experts * d.k],
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
    same_selection(d, &got, &refs)?;
    // A logit that is not finite: the decode router's own fallback. The
    // router's last row carries it: the shared expert's gate, or the last
    // routed expert's when the router has no shared row.
    let mut routed = oracle::activations(d.experts * d.k, 0x5F0, 0.05);
    let mut nan_shared = vec![0.01f32; d.k];
    if d.shared == 1 {
        nan_shared[10] = f32::NAN;
    } else {
        routed[(d.experts - 1) * d.k + 10] = f32::NAN;
    }
    let bad = decoded(h, &routed, &nan_shared, 0x720, eps)?;
    expect_site(
        h.fault()?,
        "the decode router on a NaN logit in its last row",
        FaultSite::Router,
    )?;
    let got = select_run(h, &[&bad])?;
    expect_site(got.fault, "the selection on a NaN logit", FaultSite::Router)?;
    same_selection(d, &got, &[&bad])?;
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
    let d = Dims::of(h);
    let slots = ids.len();
    let ids_b = upload(h, ids)?;
    let mut sorted = upload(h, &vec![0xFFFF_FFFFu32; slots])?;
    let mut inv = upload(h, &vec![0xFFFF_FFFFu32; slots])?;
    let words = route_table_words(slots, d.rows);
    let mut tiles = upload(h, &vec![0xFFFF_FFFFu32; words])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.route_table(
            encoder,
            RouteTableParams {
                slots,
                per_row: d.per_row,
                experts: d.experts,
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
/// `picks(r)` and the shared expert last when the router has one. A row's
/// picks are distinct.
pub struct Picks {
    pub ids: Vec<u32>,
}

/// The ids of a pass: a hot expert every row picks (so its group takes more
/// than one tile), the rest pseudo-random and distinct within a row.
pub fn picks(d: Dims, rows: usize, hot: Option<u32>, seed: u64) -> Picks {
    let mut rng = oracle::Rng::new(seed);
    let mut ids = Vec::with_capacity(rows * d.per_row);
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::new();
        if let Some(e) = hot {
            row.push(e);
        }
        while row.len() < d.top_k {
            let e = (rng.next() % d.experts as u64) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
        if d.shared == 1 {
            ids.push(d.experts as u32);
        }
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

fn rule(d: Dims, ids: &[u32]) -> Rule {
    let groups = d.experts + d.shared;
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

fn check_table(d: Dims, ids: &[u32], t: &Table, what: &str) -> Result<usize, String> {
    let slots = ids.len();
    let r = rule(d, ids);
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
    let cap = route_table_words(slots, d.rows);
    if t.tiles[4 * (1 + n)..cap].iter().any(|&w| w != 0xFFFF_FFFF) {
        return Err(format!("{what}: a word past the last tile was written"));
    }
    Ok(n)
}

/// The slots are sorted by group and cut into tiles by the host rule.
pub fn route_table(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let d = Dims::of(h);
    let mut parts = Vec::new();
    for (rows, hot) in [
        (1usize, None),
        (9, None),
        (40, Some(5u32)),
        (130, Some(d.experts as u32 - 1)),
    ] {
        let ids = picks(d, rows, hot, 0x7AB + rows as u64).ids;
        let t = table_run(h, &ids)?;
        let tiles = check_table(d, &ids, &t, &format!("{rows} rows"))?;
        parts.push(format!("{rows}:{tiles}"));
    }
    // An id outside its slot's range raises the expert-id site, is placed in
    // group 0, and moves no other slot's group. The second bad id is a
    // shared slot's that is not the shared id; a router with no shared row
    // has none, and the shared id (the group past the routed ones) in a routed
    // slot takes its place.
    let mut ids = picks(d, 9, None, 0x7AC).ids;
    let (second, second_id) = if d.shared == 1 {
        (d.per_row + d.top_k, 17)
    } else {
        (d.per_row + 3, d.experts as u32)
    };
    ids[4] = d.experts as u32 + 3;
    ids[second] = second_id;
    let t = table_run(h, &ids)?;
    expect_site(
        t.fault,
        "a slot with an id outside its range",
        FaultSite::ExpertId,
    )?;
    let mut fixed = ids.clone();
    fixed[4] = 0;
    fixed[second] = 0;
    let mut clean = t;
    clean.fault = None;
    for (p, &s) in clean.sorted.iter().enumerate() {
        if clean.inv[s as usize] as usize != p {
            return Err(format!("a raised table: slot {s} does not invert"));
        }
    }
    let r = rule(d, &fixed);
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

/// A family's stacks and shared tensors, for the pair and the down. The
/// shared tensors are empty when the router has no shared row.
struct Experts {
    /// The gate and up stacks' type.
    quant: Quant,
    /// The down stack's type: the pair's, or the type of a layer whose down
    /// is wider.
    down_quant: Quant,
    /// The type of the shared gate and up: the stacks', or the second of a
    /// pair of the gate-up table.
    shared_quant: Quant,
    /// The type of the shared down: the down stack's, or the second of a
    /// pair of the down table.
    shared_down_quant: Quant,
    dims: Dims,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
}

fn stack(
    experts: usize,
    fix: &Fixture,
    name: &str,
    ty: GgmlType,
    cols: usize,
    rows: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for e in 0..experts {
        out.extend_from_slice(fix.expert(name, ty, cols, rows, e)?);
    }
    Ok(out)
}

/// The shared tensor `name`'s bytes when the router has a shared row, none
/// otherwise.
fn shared_rows(
    d: Dims,
    fix: &Fixture,
    name: &str,
    ty: GgmlType,
    (cols, rows): (usize, usize),
) -> Result<Vec<u8>, String> {
    if d.shared == 0 {
        return Ok(Vec::new());
    }
    Ok(fix.rows(name, ty, cols, rows)?.to_vec())
}

fn experts_of(d: Dims, fix: &Fixture, quant: Quant) -> Result<Experts, String> {
    let ty = quant.ggml();
    Ok(Experts {
        quant,
        down_quant: quant,
        shared_quant: quant,
        shared_down_quant: quant,
        dims: d,
        gate: stack(
            d.experts,
            fix,
            "blk.0.ffn_gate_exps.weight",
            ty,
            d.k,
            d.expert_k,
        )?,
        up: stack(
            d.experts,
            fix,
            "blk.0.ffn_up_exps.weight",
            ty,
            d.k,
            d.expert_k,
        )?,
        down: stack(
            d.experts,
            fix,
            "blk.0.ffn_down_exps.weight",
            ty,
            d.expert_k,
            d.k,
        )?,
        shared_gate: shared_rows(d, fix, "blk.0.ffn_gate_shexp.weight", ty, (d.k, d.expert_k))?,
        shared_up: shared_rows(d, fix, "blk.0.ffn_up_shexp.weight", ty, (d.k, d.expert_k))?,
        shared_down: shared_rows(d, fix, "blk.0.ffn_down_shexp.weight", ty, (d.expert_k, d.k))?,
    })
}

/// The family of the Q4_K pair over a Q6_K down: layer 0's gate and up, and
/// the down stack of the first layer the fixture keeps at Q6_K, as a model
/// whose layers differ in the down's type runs them. The router has no shared
/// row: the shared expert of such a file has no second type to run.
fn q6_down_experts(d: Dims, fix: &Fixture) -> Result<Experts, String> {
    if d.shared == 1 {
        return Err("the Q6_K-down family is written for a router with no shared row".to_string());
    }
    let layer = fix.first_layer_of("ffn_down_exps.weight", GgmlType::Q6_K)?;
    Ok(Experts {
        down_quant: Quant::Q6K,
        shared_down_quant: Quant::Q6K,
        down: stack(
            d.experts,
            fix,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            GgmlType::Q6_K,
            d.expert_k,
            d.k,
        )?,
        ..experts_of(d, fix, Quant::Q4K)?
    })
}

/// A weight type's stacks of synthetic rows ([`synthetic::matrix`]), for a
/// quant no fixture holds: every expert its own rows, each stack and shared
/// tensor from its own seed.
fn synthetic_experts(d: Dims, quant: Quant, seed: u64) -> Experts {
    let shared = |rows: usize, cols: usize, seed: u64| {
        if d.shared == 1 {
            synthetic::matrix(quant, rows, cols, seed)
        } else {
            Vec::new()
        }
    };
    Experts {
        quant,
        down_quant: quant,
        shared_quant: quant,
        shared_down_quant: quant,
        dims: d,
        gate: synthetic::matrix(quant, d.experts * d.expert_k, d.k, seed),
        up: synthetic::matrix(quant, d.experts * d.expert_k, d.k, seed + 1),
        down: synthetic::matrix(quant, d.experts * d.k, d.expert_k, seed + 2),
        shared_gate: shared(d.expert_k, d.k, seed + 3),
        shared_up: shared(d.expert_k, d.k, seed + 4),
        shared_down: shared(d.k, d.expert_k, seed + 5),
    }
}

impl Experts {
    /// The bytes of expert `e` of a stack (or the shared tensor for the
    /// table's last group), `one` bytes each.
    fn of<'a>(&self, stack: &'a [u8], shared: &'a [u8], e: usize) -> &'a [u8] {
        let experts = self.dims.experts;
        if e == experts {
            shared
        } else {
            let one = stack.len() / experts;
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

fn scene(d: Dims, rows: usize, hot: Option<u32>, seed: u64) -> Scene {
    let ids = picks(d, rows, hot, seed).ids;
    let mut rng = oracle::Rng::new(seed ^ 0x5CE);
    let route_w = ids
        .iter()
        .map(|_| 0.1 + 0.9 * (rng.next() >> 40) as f32 / (1u64 << 24) as f32)
        .collect();
    let mut x = Vec::with_capacity(rows * d.k);
    for r in 0..rows {
        x.extend(oracle::activations(d.k, seed + r as u64, 1.0));
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
    /// The pair's outputs, one row of the expert width a place.
    h: Vec<f32>,
    /// The down's outputs, one row of the hidden width a place.
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
    let d = ex.dims;
    let table = table_run(h, &sc.ids)?;
    if let Some(f) = table.fault {
        return Err(format!("the table raised {f}"));
    }
    let over = OverTable {
        kernels: h.kernels,
        sorted: &table.sorted,
        tiles: &table.tiles,
        max_tiles: route_tiles(sc.rows * d.per_row, d.rows),
    };
    let out = grouped_launch(h, ex, sc, operand, &over)?;
    Ok(Grouped {
        h: out.h,
        y: out.y,
        table,
        pair_fault: out.pair_fault,
        down_fault: out.down_fault,
    })
}

/// The kernels a grouped launch runs and the table it runs over.
struct OverTable<'a> {
    kernels: &'a Kernels,
    sorted: &'a [u32],
    tiles: &'a [u32],
    max_tiles: usize,
}

/// What a grouped launch wrote and raised.
struct Outs {
    h: Vec<f32>,
    y: Vec<f32>,
    pair_fault: Option<Fault>,
    down_fault: Option<Fault>,
}

/// The pair, then the down on the pair's outputs, by `over`'s kernels over
/// its table.
fn grouped_launch(
    h: &Harness<'_>,
    ex: &Experts,
    sc: &Scene,
    operand: Operand,
    over: &OverTable<'_>,
) -> Result<Outs, String> {
    let d = ex.dims;
    let slots = sc.rows * d.per_row;
    let tiles_b = upload(h, over.tiles)?;
    let sorted_b = upload(h, over.sorted)?;
    let w_b = upload(h, &sc.route_w)?;
    let x_b = upload(h, &sc.x)?;
    let gate = bytes(h, &ex.gate)?;
    let up = bytes(h, &ex.up)?;
    let down = bytes(h, &ex.down)?;
    // The shared tensors exist when the router has a shared row.
    let shared = if d.shared == 1 {
        Some([
            bytes(h, &ex.shared_gate)?,
            bytes(h, &ex.shared_up)?,
            bytes(h, &ex.shared_down)?,
        ])
    } else {
        None
    };
    let mut hid = upload(h, &vec![sentinel(); slots * d.expert_k])?;
    let mut out = upload(h, &vec![sentinel(); slots * d.k])?;
    let max_tiles = over.max_tiles;
    h.clear_fault()?;
    h.encode(|encoder| {
        over.kernels.gemm_routed(
            encoder,
            RoutedParams {
                quant: ex.quant,
                shared_quant: ex.shared_quant,
                kind: RoutedKind::Pair,
                operand,
                cols: d.k,
                rows: d.expert_k,
                experts: d.experts,
                stack: gate.window(Elems(0), Elems(ex.gate.len()))?,
                stack_up: Some(up.window(Elems(0), Elems(ex.up.len()))?),
                shared: shared
                    .as_ref()
                    .map(|s| s[0].window(Elems(0), Elems(ex.shared_gate.len())))
                    .transpose()?,
                shared_up: shared
                    .as_ref()
                    .map(|s| s[1].window(Elems(0), Elems(ex.shared_up.len())))
                    .transpose()?,
                x: x_b.window(Elems(0), Elems(sc.x.len()))?,
                x_rows: sc.rows,
                x_stride: d.k,
                gather: true,
                y: hid.window_mut(Elems(0), Elems(slots * d.expert_k))?,
                y_stride: d.expert_k,
                slots,
                max_tiles,
                sorted: sorted_b.window(Elems(0), Elems(slots))?,
                route_w: w_b.window(Elems(0), Elems(slots))?,
                tiles: tiles_b.window(Elems(0), Elems(over.tiles.len()))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let pair_fault = h.fault()?;
    h.clear_fault()?;
    h.encode(|encoder| {
        over.kernels.gemm_routed(
            encoder,
            RoutedParams {
                quant: ex.down_quant,
                shared_quant: ex.shared_down_quant,
                kind: RoutedKind::Plain,
                operand,
                cols: d.expert_k,
                rows: d.k,
                experts: d.experts,
                stack: down.window(Elems(0), Elems(ex.down.len()))?,
                stack_up: None,
                shared: shared
                    .as_ref()
                    .map(|s| s[2].window(Elems(0), Elems(ex.shared_down.len())))
                    .transpose()?,
                shared_up: None,
                x: hid.window(Elems(0), Elems(slots * d.expert_k))?,
                x_rows: slots,
                x_stride: d.expert_k,
                gather: false,
                y: out.window_mut(Elems(0), Elems(slots * d.k))?,
                y_stride: d.k,
                slots,
                max_tiles,
                sorted: sorted_b.window(Elems(0), Elems(slots))?,
                route_w: w_b.window(Elems(0), Elems(slots))?,
                tiles: tiles_b.window(Elems(0), Elems(over.tiles.len()))?,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let down_fault = h.fault()?;
    Ok(Outs {
        h: read(h, &hid, slots * d.expert_k)?,
        y: read(h, &out, slots * d.k)?,
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
    path: GemmPath,
) -> Result<(f64, f64), String> {
    let d = ex.dims;
    let slots = sc.rows * d.per_row;
    let (f_pair, ops_pair) = gemm_band(ex.quant, d.k, operand, path);
    let (f_down, ops_down) = gemm_band(ex.down_quant, d.expert_k, operand, path);
    // The shared group's tiles run in the shared types.
    let (f_pair_sh, ops_pair_sh) = gemm_band(ex.shared_quant, d.k, operand, path);
    let (f_down_sh, ops_down_sh) = gemm_band(ex.shared_down_quant, d.expert_k, operand, path);
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
    let mut abs_down = AbsSums {
        quant: ex.down_quant,
        cache: HashMap::new(),
    };
    let mut abs_sh = AbsSums {
        quant: ex.shared_quant,
        cache: HashMap::new(),
    };
    let mut abs_down_sh = AbsSums {
        quant: ex.shared_down_quant,
        cache: HashMap::new(),
    };
    let mut worst_pair = 0f64;
    let mut worst_down = 0f64;
    for p in 0..slots {
        let s = run.table.sorted[p] as usize;
        let e = sc.ids[s] as usize;
        if !(p % stride == 0 || e == d.experts || every == Some(e as u32)) {
            continue;
        }
        let row = s / d.per_row;
        let x: Vec<f64> = sc.x[row * d.k..(row + 1) * d.k]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let x_abs: f64 = x.iter().map(|v| v.abs()).sum();
        let shared = e == d.experts;
        let (pair_quant, f_pair, ops_pair) = if shared {
            (ex.shared_quant, f_pair_sh, ops_pair_sh)
        } else {
            (ex.quant, f_pair, ops_pair)
        };
        let abs = if shared { &mut abs_sh } else { &mut abs };
        let gate = ex.of(&ex.gate, &ex.shared_gate, e);
        let up = ex.of(&ex.up, &ex.shared_up, e);
        let rg = oracle::reference(pair_quant, gate, d.expert_k, d.k, &x)?;
        let ru = oracle::reference(pair_quant, up, d.expert_k, d.k, &x)?;
        let wg = abs.of(0, e, gate, d.expert_k, d.k)?.to_vec();
        let wu = abs.of(1, e, up, d.expert_k, d.k)?.to_vec();
        let w = f64::from(sc.route_w[s]);
        for o in 0..d.expert_k {
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
            let got = f64::from(run.h[p * d.expert_k + o]);
            let err = (got - want).abs();
            if !got.is_finite() || err > width {
                return Err(format!(
                    "pair place {p} (slot {s}, group {e}) output {o}: got {got:e}, want {want:e}, error {err:e} over the band {width:e}"
                ));
            }
            worst_pair = worst_pair.max(err / width);
        }
        // The down, on the kernel's own pair outputs at this place.
        let hx: Vec<f64> = run.h[p * d.expert_k..(p + 1) * d.expert_k]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let h_abs: f64 = hx.iter().map(|v| v.abs()).sum();
        let (down_quant, f_down, ops_down) = if shared {
            (ex.shared_down_quant, f_down_sh, ops_down_sh)
        } else {
            (ex.down_quant, f_down, ops_down)
        };
        let abs_down = if shared {
            &mut abs_down_sh
        } else {
            &mut abs_down
        };
        let down = ex.of(&ex.down, &ex.shared_down, e);
        let rd = oracle::reference(down_quant, down, d.k, d.expert_k, &hx)?;
        let wd = abs_down.of(2, e, down, d.k, d.expert_k)?.to_vec();
        for (n, &wd_n) in wd.iter().enumerate().take(d.k) {
            let width = dot_width(f_down, ops_down, sub, rd.magnitude[n], h_abs, wd_n);
            let got = f64::from(run.y[p * d.k + n]);
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
    let d = ex.dims;
    let mut sc = scene(d, 5, None, 0x6B0);
    let bad_row = 2usize;
    sc.x[bad_row * d.k + 100] = 7.0e4;
    let run = grouped_run(h, ex, &sc, Operand::F16)?;
    expect_site(
        run.pair_fault,
        "the pair on an activation past the half range",
        FaultSite::QuantColumn,
    )?;
    for (p, &s) in run.table.sorted.iter().enumerate() {
        let row = s as usize / d.per_row;
        let out = &run.h[p * d.expert_k..(p + 1) * d.expert_k];
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

fn grouped_quant(h: &Harness<'_>, ex: &Experts, operands: &[Operand]) -> Result<String, String> {
    let quant = ex.quant;
    let label = if ex.down_quant == quant {
        quant.name().to_string()
    } else {
        format!("{} with a {} down", quant.name(), ex.down_quant.name())
    };
    let mut parts = Vec::new();
    for &operand in operands {
        // A tile and eight rows: every row picks the hot expert, so its
        // group, and the shared group when there is one, take two tiles.
        let sc = scene(ex.dims, ROUTE_TILE + 8, Some(5), 0x6A0);
        let run = grouped_run(h, ex, &sc, operand)?;
        for (what, fault) in [("pair", run.pair_fault), ("down", run.down_fault)] {
            if let Some(f) = fault {
                return Err(format!("the clean {what} launch raised {f}"));
            }
        }
        let (pair, down) =
            check_grouped(ex, &sc, &run, operand, 7, Some(5), h.kernels.gemm_path())?;
        parts.push(format!("{label} {operand:?} pair {pair:.3} down {down:.3}"));
    }
    if operands.contains(&Operand::F16) {
        range_fault(h, ex)?;
        parts.push(format!(
            "{label} an activation past the half range is the fault and NaN"
        ));
    }
    Ok(parts.join(", "))
}

/// A stack of `quant` gate and up and `down_quant` down beside a shared expert
/// of `shared_quant` and `shared_down_quant`, from synthetic rows: each
/// tensor its own seed.
fn synthetic_apart_experts(
    d: Dims,
    (quant, shared_quant): (Quant, Quant),
    (down_quant, shared_down_quant): (Quant, Quant),
    seed: u64,
) -> Experts {
    Experts {
        quant,
        down_quant,
        shared_quant,
        shared_down_quant,
        dims: d,
        gate: synthetic::matrix(quant, d.experts * d.expert_k, d.k, seed),
        up: synthetic::matrix(quant, d.experts * d.expert_k, d.k, seed + 1),
        down: synthetic::matrix(down_quant, d.experts * d.k, d.expert_k, seed + 2),
        shared_gate: synthetic::matrix(shared_quant, d.expert_k, d.k, seed + 3),
        shared_up: synthetic::matrix(shared_quant, d.expert_k, d.k, seed + 4),
        shared_down: synthetic::matrix(shared_down_quant, d.k, d.expert_k, seed + 5),
    }
}

/// The grouped GEMM over a shared expert of another type than its stacks:
/// every pair of the gate-up table with a pair of the down table, so each
/// tile of each table runs, every arm of [`grouped_arms`] (the shared group's
/// places held to the shared type's band).
pub fn grouped_apart(h: &Harness<'_>) -> Clause {
    let d = Dims::of(h);
    if d.shared != 1 {
        return Err("the Apart tiles need a router with a shared row".to_string());
    }
    let mut names = Vec::new();
    for (i, (&pair, &down)) in APART_GATE_UP.iter().zip(&APART_DOWN).enumerate() {
        let name = format!(
            "{}|{} pair with {}|{} down",
            pair.0.name(),
            pair.1.name(),
            down.0.name(),
            down.1.name()
        );
        let ex = synthetic_apart_experts(d, pair, down, 0x6C00 + 16 * i as u64);
        grouped_arms(h, &ex).map_err(|e| format!("{name}: {e}"))?;
        names.push(name);
    }
    Ok(Verdict::Noted(format!(
        "every grouped arm over {}",
        names.join(", ")
    )))
}

/// The pair and the down over the table's tiles are the oracle's, for every
/// (row, slot), the shared expert's group (when the router has one) and a
/// group of two tiles included. `a4` is the MLX fixture of the A4 family, for
/// the instance whose gate runs it.
pub fn grouped_gemm(h: &Harness<'_>, fixture: &Fixture, a4: Option<&Fixture>) -> Clause {
    let d = Dims::of(h);
    let mut notes = vec![grouped_quant(
        h,
        &experts_of(d, fixture, Quant::Q4K)?,
        &[Operand::F16, Operand::F32],
    )?];
    if let Some(m) = a4 {
        notes.push(grouped_quant(
            h,
            &experts_of(d, m, Quant::A4)?,
            &[Operand::F16],
        )?);
    }
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}",
        notes.join(", ")
    )))
}

/// The Q6_K family's stacks. The fixture's Q6_K experts are blk.2's down
/// (and its shared down); its gate and up are Q4_K in every layer, so the
/// pair's stacks are Q6_K rows of blk.2's attn_qkv (8192 rows of K columns)
/// taken in turn through every expert's rows — real Q6_K blocks at the
/// pair's shape, the gate's from row 0 on and the up's from row 4096 on.
fn q6k_experts(d: Dims, fix: &Fixture) -> Result<Experts, String> {
    let ty = GgmlType::Q6_K;
    let qkv_rows = 8192;
    let qkv = fix.rows("blk.2.attn_qkv.weight", ty, d.k, qkv_rows)?;
    let row = qkv.len() / qkv_rows;
    let cycled = |first: usize, rows: usize| {
        let mut out = Vec::with_capacity(rows * row);
        for r in 0..rows {
            let at = (first + r) % qkv_rows;
            out.extend_from_slice(&qkv[at * row..(at + 1) * row]);
        }
        out
    };
    Ok(Experts {
        quant: Quant::Q6K,
        down_quant: Quant::Q6K,
        shared_quant: Quant::Q6K,
        shared_down_quant: Quant::Q6K,
        dims: d,
        gate: cycled(0, d.experts * d.expert_k),
        up: cycled(qkv_rows / 2, d.experts * d.expert_k),
        down: stack(
            d.experts,
            fix,
            "blk.2.ffn_down_exps.weight",
            ty,
            d.expert_k,
            d.k,
        )?,
        shared_gate: cycled(1000, d.expert_k),
        shared_up: cycled(5000, d.expert_k),
        shared_down: fix
            .rows("blk.2.ffn_down_shexp.weight", ty, d.expert_k, d.k)?
            .to_vec(),
    })
}

/// What a launch over a table with one tile the grouped GEMM refuses leaves:
/// the pair and the down raise the expert-id site, the tile's `places` hold
/// no store of either, and the place after them (the next tile's first) is
/// stored.
fn expect_refused(out: &Outs, d: Dims, places: Range<usize>, what: &str) -> Result<(), String> {
    expect_site(
        out.pair_fault,
        &format!("the pair on {what}"),
        FaultSite::ExpertId,
    )?;
    expect_site(
        out.down_fault,
        &format!("the down on {what}"),
        FaultSite::ExpertId,
    )?;
    for p in places.clone() {
        for (side, row) in [
            ("pair", &out.h[p * d.expert_k..(p + 1) * d.expert_k]),
            ("down", &out.y[p * d.k..(p + 1) * d.k]),
        ] {
            if let Some(v) = row.iter().find(|v| v.to_bits() != SENTINEL_BITS) {
                return Err(format!(
                    "the {side} stored {v:?} at place {p} of the refused tile"
                ));
            }
        }
    }
    let after = places.end;
    if out.h[after * d.expert_k..(after + 1) * d.expert_k]
        .iter()
        .any(|v| v.to_bits() == SENTINEL_BITS)
    {
        return Err(format!(
            "place {after}, the next tile's first, was not stored"
        ));
    }
    Ok(())
}

/// A tile the table cannot write — its run one slot longer than a tile —
/// raises the expert-id site in the pair and in the down, and neither
/// stores a word of its places; the table's other tiles are stored.
fn bad_tile(h: &Harness<'_>, ex: &Experts) -> Result<(), String> {
    let d = ex.dims;
    let sc = scene(d, ROUTE_TILE + 8, Some(5), 0x6A1);
    let table = table_run(h, &sc.ids)?;
    if let Some(f) = table.fault {
        return Err(format!("the table raised {f}"));
    }
    // Tile 0's words are (group, first place, run, unused).
    let mut tiles = table.tiles.clone();
    let first = tiles[4 + 1] as usize;
    let count = tiles[4 + 2] as usize;
    tiles[4 + 2] = (ROUTE_TILE + 1) as u32;
    let over = OverTable {
        kernels: h.kernels,
        sorted: &table.sorted,
        tiles: &tiles,
        max_tiles: route_tiles(sc.rows * d.per_row, d.rows),
    };
    let out = grouped_launch(h, ex, &sc, Operand::F16, &over)?;
    expect_refused(
        &out,
        d,
        first..first + count,
        "a tile past the tile's slots",
    )
}

/// The table's group one past the routed experts: the shared expert's when
/// the instance has one, a group the table cannot write when it has none. A
/// hand-built table of two tiles, that group over the first two rows' places
/// and the last routed expert over the rest. With a shared group both tiles
/// are the oracle's within the band, the first against the shared tensors;
/// with none the first raises the expert-id site in the pair and in the down
/// and stores nothing, and the second is stored. Returns what it showed.
fn past_routed(h: &Harness<'_>, ex: &Experts) -> Result<&'static str, String> {
    let d = ex.dims;
    let mut sc = scene(d, 4, None, 0x6A3);
    let slots = sc.rows * d.per_row;
    let cut = 2 * d.per_row;
    let (past, last) = (d.experts as u32, d.experts as u32 - 1);
    // The oracle reads a slot's group from the scene's ids.
    sc.ids = (0..slots)
        .map(|s| if s < cut { past } else { last })
        .collect();
    let max_tiles = route_tiles(slots, d.rows);
    let mut tiles = vec![0xFFFF_FFFFu32; 4 * (1 + max_tiles)];
    tiles[..12].copy_from_slice(&[
        2,
        0,
        0,
        0,
        past,
        0,
        cut as u32,
        0,
        last,
        cut as u32,
        (slots - cut) as u32,
        0,
    ]);
    let sorted: Vec<u32> = (0..slots as u32).collect();
    let over = OverTable {
        kernels: h.kernels,
        sorted: &sorted,
        tiles: &tiles,
        max_tiles,
    };
    let out = grouped_launch(h, ex, &sc, Operand::F16, &over)?;
    if d.shared == 0 {
        expect_refused(
            &out,
            d,
            0..cut,
            "a tile of the group past the routed experts",
        )?;
        return Ok(
            "the group past the routed experts raises the expert-id site and stores nothing",
        );
    }
    for (side, fault) in [("pair", out.pair_fault), ("down", out.down_fault)] {
        if let Some(f) = fault {
            return Err(format!("the {side} on the shared group's tile raised {f}"));
        }
    }
    let run = Grouped {
        h: out.h,
        y: out.y,
        table: Table {
            inv: sorted.clone(),
            sorted,
            tiles,
            fault: None,
        },
        pair_fault: None,
        down_fault: None,
    };
    check_grouped(ex, &sc, &run, Operand::F16, 1, None, h.kernels.gemm_path())
        .map_err(|e| format!("the shared group's tile: {e}"))?;
    Ok("the group past the routed experts is the shared expert's, within the band")
}

/// The grouped GEMM's every arm of one weight type on `h`'s kernels (the
/// tensor-op kernels, for the `tensor_grouped_*` clauses): the pair and the
/// down in both operands, every checked place within the band of `h`'s
/// GEMM path; runs of 20 and 40 places (the hot group's and the shared
/// group's), which end inside a SIMD group's second block of places and
/// inside the second group's first; an activation past the half range the
/// fault and NaN; a tile past the tile's slots the expert-id site with
/// nothing stored; the group past the routed experts the shared expert's or
/// that site, by whether the instance has one.
fn grouped_arms(h: &Harness<'_>, ex: &Experts) -> Clause {
    let note = grouped_quant(h, ex, &[Operand::F16, Operand::F32])?;
    let mut runs = Vec::new();
    for rows in [20usize, 40] {
        let sc = scene(ex.dims, rows, Some(9), 0x6A2 + rows as u64);
        let run = grouped_run(h, ex, &sc, Operand::F16)?;
        for (what, fault) in [("pair", run.pair_fault), ("down", run.down_fault)] {
            if let Some(f) = fault {
                return Err(format!(
                    "the clean {what} launch on runs of {rows} raised {f}"
                ));
            }
        }
        let (pair, down) = check_grouped(
            ex,
            &sc,
            &run,
            Operand::F16,
            1,
            Some(9),
            h.kernels.gemm_path(),
        )
        .map_err(|e| format!("runs of {rows}: {e}"))?;
        runs.push(format!("runs of {rows} pair {pair:.3} down {down:.3}"));
    }
    bad_tile(h, ex)?;
    let past = past_routed(h, ex)?;
    Ok(Verdict::Noted(format!(
        "{} kernels, worst error over its band: {note}, {}; a tile past the tile's slots raises the expert-id site and stores nothing; {past}",
        h.kernels.gemm_path().name(),
        runs.join(", ")
    )))
}

pub fn tensor_grouped_q4k(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    grouped_arms(h, &experts_of(Dims::of(h), fixture, Quant::Q4K)?)
}

pub fn tensor_grouped_q6k(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    grouped_arms(h, &q6k_experts(Dims::of(h), fixture)?)
}

pub fn tensor_grouped_a4(h: &Harness<'_>, mlx: &Fixture) -> Clause {
    grouped_arms(h, &experts_of(Dims::of(h), mlx, Quant::A4)?)
}

/// The grouped GEMM's every arm over the Q4_K pair and a Q6_K down, as a
/// layer of a file whose down type varies runs them.
pub fn grouped_q6_down(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    grouped_arms(h, &q6_down_experts(Dims::of(h), fixture)?)
}

/// Every grouped arm of `quant` on `h`'s kernels over synthetic stacks:
/// [`grouped_arms`]'s cases.
pub fn grouped_synthetic(h: &Harness<'_>, quant: Quant, seed: u64) -> Clause {
    grouped_arms(h, &synthetic_experts(Dims::of(h), quant, seed))
}

// ------------------------------------------------------------ grouped oracle

/// The routed tile the gate keeps as the oracle, from `#define ROUTED_FRAGS_A`
/// on.
const ORACLE: &str = include_str!("routed_oracle.metal");

/// The shipped routed GEMM's file, whose routed tile the oracle replaces.
const SHIPPED: &str = include_str!("../../../kernels/gemm_routed.metal");

/// The table tile the oracle's kernel takes.
const ORACLE_TILE: usize = 32;

/// Rows of the oracle scene, and the run lengths each of its eight lanes
/// cuts them into: every A fragment count of a tile, the fragment boundaries
/// 8 k and 8 k + 1, a full tile, and groups of two and three tiles (the
/// last lane's three among them).
const LANE_ROWS: usize = 136;
const LANES: [&[usize]; 8] = [
    &[1, 7, 8, 9, 15, 16, 17, 63],
    &[64, 65, 7],
    &[23, 24, 25, 31, 33],
    &[39, 40, 41, 16],
    &[47, 48, 41],
    &[55, 56, 25],
    &[129, 7],
    &[136],
];

/// What `LANES` is written for, held for every registered row: one lane a
/// routed pick.
const _: () = {
    let mut i = 0;
    while i < REGISTERED.len() {
        assert!(Geometry::of(&REGISTERED[i]).experts.routed_ids == LANES.len());
        i += 1;
    }
};

/// The kernels compiled from the shipped source with the routed tile swapped
/// for the oracle's, and the table's tile and the routed tile's activation
/// rows at the oracle's.
fn oracle_kernels(h: &Harness<'_>) -> Result<Kernels, String> {
    let (source, _) = sources_of(h.library.instance()).map_err(why)?;
    let at = SHIPPED
        .find("#define ROUTED_FRAGS_A")
        .ok_or("gemm_routed.metal has no `#define ROUTED_FRAGS_A`; the tile's start moved")?;
    let from = format!("#define ROUTE_TILE {ROUTE_TILE}\n");
    let to = format!("#define ROUTE_TILE {ORACLE_TILE}\n");
    let acts_from = format!("#define ROUTED_ACTS {ROUTE_TILE}   //");
    let acts_to = format!("#define ROUTED_ACTS {ORACLE_TILE}   //");
    let mut text = source.to_string();
    for (find, with) in [
        (&SHIPPED[at..], ORACLE),
        (from.as_str(), to.as_str()),
        (acts_from.as_str(), acts_to.as_str()),
    ] {
        let found = text.matches(find).count();
        if found != 1 {
            return Err(format!(
                "the oracle's edit finds {:?} {found} times, not 1",
                find.lines().next().unwrap_or("")
            ));
        }
        text = text.replacen(find, with, 1);
    }
    let library = Library::from_source_text(h.device, &text).map_err(why)?;
    Kernels::from_library(&library).map_err(why)
}

/// The oracle scene: row r takes, in lane j, the expert `experts / 8 * j + i`
/// of the run i its lane puts r in, then the shared expert when the router
/// has one.
fn lane_scene(d: Dims) -> Result<Scene, String> {
    if ROUTE_TILE != 64 {
        return Err(format!(
            "the oracle scene's runs are cut for a 64-slot tile, and the table's is {ROUTE_TILE}"
        ));
    }
    let mut ids = Vec::with_capacity(LANE_ROWS * d.per_row);
    for r in 0..LANE_ROWS {
        for (j, lane) in LANES.iter().enumerate() {
            let mut end = 0;
            let run = lane.iter().position(|&n| {
                end += n;
                r < end
            });
            let i = run.ok_or_else(|| format!("lane {j}'s runs end before row {r}"))?;
            ids.push((d.experts / LANES.len() * j + i) as u32);
        }
        if d.shared == 1 {
            ids.push(d.experts as u32);
        }
    }
    if let Some(j) = LANES
        .iter()
        .position(|l| l.iter().sum::<usize>() != LANE_ROWS)
    {
        return Err(format!("lane {j}'s runs are not {LANE_ROWS} rows"));
    }
    let mut sc = scene(d, LANE_ROWS, None, 0x0AC);
    sc.ids = ids;
    Ok(sc)
}

/// The table cut again at the oracle's tile: each entry's run in pieces of
/// `ORACLE_TILE`, in order.
fn oracle_tiles(tiles: &[u32]) -> (Vec<u32>, usize) {
    let n = tiles[0] as usize;
    let mut out = vec![0u32; 4];
    for i in 0..n {
        let e = &tiles[4 * (1 + i)..4 * (1 + i) + 4];
        let mut at = 0u32;
        while at < e[2] {
            let take = (e[2] - at).min(ORACLE_TILE as u32);
            out.extend([e[0], e[1] + at, take, 0]);
            at += take;
        }
    }
    let pieces = out.len() / 4 - 1;
    out[0] = pieces as u32;
    (out, pieces)
}

/// Every output written, and each the oracle's bits.
fn same_bits(what: &str, got: &[f32], want: &[f32], cols: usize) -> Result<(), String> {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() == SENTINEL_BITS {
            return Err(format!(
                "{what}: place {} column {} is not written",
                i / cols,
                i % cols
            ));
        }
        if g.to_bits() != w.to_bits() {
            return Err(format!(
                "{what}: place {} column {}: {g:?} ({:#x}), the oracle's {w:?} ({:#x})",
                i / cols,
                i % cols,
                g.to_bits(),
                w.to_bits()
            ));
        }
    }
    Ok(())
}

/// The shipped tile's pair and down are the oracle tile's bit for bit, on
/// runs of every A fragment count, its boundaries and groups of several
/// tiles: a tile's height and its split over the SIMD groups do not move an
/// output's K order.
pub fn grouped_oracle(h: &Harness<'_>, fixture: &Fixture, a4: Option<&Fixture>) -> Clause {
    let d = Dims::of(h);
    let oracle = oracle_kernels(h)?;
    let sc = lane_scene(d)?;
    let mut families = vec![(fixture, Quant::Q4K, &[Operand::F16, Operand::F32][..])];
    if let Some(m) = a4 {
        families.push((m, Quant::A4, &[Operand::F16][..]));
    }
    let mut notes = Vec::new();
    for (fix, quant, operands) in families {
        let ex = experts_of(d, fix, quant)?;
        for &operand in operands {
            let ours = grouped_run(h, &ex, &sc, operand)?;
            let (tiles, pieces) = oracle_tiles(&ours.table.tiles);
            let theirs = grouped_launch(
                h,
                &ex,
                &sc,
                operand,
                &OverTable {
                    kernels: &oracle,
                    sorted: &ours.table.sorted,
                    tiles: &tiles,
                    max_tiles: pieces,
                },
            )?;
            for (what, fault) in [
                ("pair", ours.pair_fault),
                ("down", ours.down_fault),
                ("oracle pair", theirs.pair_fault),
                ("oracle down", theirs.down_fault),
            ] {
                if let Some(f) = fault {
                    return Err(format!("the clean {what} launch raised {f}"));
                }
            }
            let name = format!("{} {operand:?}", quant.name());
            same_bits(&format!("{name} pair"), &ours.h, &theirs.h, d.expert_k)?;
            same_bits(&format!("{name} down"), &ours.y, &theirs.y, d.k)?;
            notes.push(format!("{name} {} tiles for {pieces}", ours.table.tiles[0]));
        }
    }
    Ok(Verdict::Noted(format!(
        "pair and down bit for bit the oracle tile's at {LANE_ROWS} rows: {}",
        notes.join(", ")
    )))
}

// ------------------------------------------------------------------ combine

/// `x += y[inv[row][0]] + ... + y[inv[row][slots - 1]]` in slot order, the sum started
/// at zero, bit for bit against the same f32 sum.
pub fn combine(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let d = Dims::of(h);
    let rows = 33usize;
    let slots = rows * d.per_row;
    let (y_stride, x_stride) = (d.k + 4, d.k + 8);
    let mut rng = oracle::Rng::new(0xC0B);
    let mut inv: Vec<u32> = (0..slots as u32).collect();
    for i in (1..slots).rev() {
        inv.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let mut y = vec![sentinel(); slots * y_stride];
    for p in 0..slots {
        for (c, v) in oracle::activations(d.k, 0xC0C + p as u64, 3.0)
            .iter()
            .enumerate()
        {
            y[p * y_stride + c] = *v;
        }
    }
    let mut x = vec![sentinel(); rows * x_stride];
    for r in 0..rows {
        for (c, v) in oracle::activations(d.k, 0xC0D + r as u64, 5.0)
            .iter()
            .enumerate()
        {
            x[r * x_stride + c] = *v;
        }
    }
    let mut want = x.clone();
    for r in 0..rows {
        for c in 0..d.k {
            let mut acc = 0f32;
            for s in 0..d.per_row {
                acc += y[inv[r * d.per_row + s] as usize * y_stride + c];
            }
            want[r * x_stride + c] = x[r * x_stride + c] + acc;
        }
    }
    let y_b = upload(h, &y)?;
    // Zero words follow the places, so a sum that reads past them reads a
    // place and not another buffer.
    let mut inv_padded = inv.clone();
    inv_padded.extend([0u32; 16]);
    let inv_b = upload(h, &inv_padded)?;
    let mut x_b = upload(h, &x)?;
    h.encode(|encoder| {
        h.kernels.combine_rows(
            encoder,
            CombineParams {
                rows,
                per_row: d.per_row,
                cols: d.k,
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
        "{rows} rows of {} slots through a shuffled place list are the f32 sum in slot order, bit for bit; the stride gaps are untouched",
        d.per_row
    )))
}

// --------------------------------------------------------------- headnorm

/// The numerator times each head's scale against f64; a head the norm cannot
/// take is NaN alone and raises the linear-gate site.
pub fn headnorm_rows(h: &Harness<'_>) -> Clause {
    let delta = &h.geom().delta;
    let rows = 9usize;
    let width = delta.value_heads * delta.head_cols;
    let (g_stride, p_stride) = (width + 4, delta.value_heads * PARTIALS_PER_HEAD + 4);
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
        for p in 0..delta.value_heads * PARTIALS_PER_HEAD {
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
        for head in 0..delta.value_heads {
            let sum: f64 = (0..PARTIALS_PER_HEAD)
                .map(|j| f64::from(partials[r * p_stride + head * PARTIALS_PER_HEAD + j]))
                .sum();
            let scale = 1.0 / (sum / delta.head_cols as f64 + f64::from(eps)).sqrt();
            for c in 0..delta.head_cols {
                let at = r * g_stride + head * delta.head_cols + c;
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
        for head in 0..delta.value_heads {
            let span =
                r * g_stride + head * delta.head_cols..r * g_stride + (head + 1) * delta.head_cols;
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
        "{rows} rows of {} heads are g / sqrt(mean + eps) within 8 U; a NaN head is NaN alone and raises the linear-gate site",
        delta.value_heads
    )))
}

// ----------------------------------------------------------------- f32 gemm

/// The router's GEMM over f32 weight rows against f64, an edge tile on each
/// side included.
pub fn gemm_f32w(h: &Harness<'_>, _fixture: &Fixture) -> Clause {
    let d = Dims::of(h);
    let (rows, acts) = (300usize, 70usize);
    let (x_stride, y_stride) = (d.k + 4, rows + 4);
    let w = oracle::activations(rows * d.k, 0xF32, 0.05);
    let mut x = vec![sentinel(); acts * x_stride];
    for r in 0..acts {
        for (c, v) in oracle::activations(d.k, 0xF33 + r as u64, 1.0)
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
                cols: d.k,
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
    let factor = oracle::gamma_bound(1 + d.k / 8 + 8);
    let floor = (d.k / 8 + 9) as f64 * f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for r in 0..acts {
        for n in 0..rows {
            let (mut dot, mut mag) = (0f64, 0f64);
            for c in 0..d.k {
                let p = f64::from(w[n * d.k + c]) * f64::from(x[r * x_stride + c]);
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
    let d = Dims::of(h);
    let slots = 2 * d.per_row;
    let ids_b = upload(h, &vec![0u32; slots])?;
    let mut sorted = upload(h, &vec![0u32; slots])?;
    let mut inv = upload(h, &vec![0u32; slots])?;
    let words = route_table_words(slots, d.rows);
    let mut tiles = upload(h, &vec![0u32; words])?;
    let logits_b = upload(h, &vec![0f32; 2 * d.logit_stride])?;
    let mut sel_ids = upload(h, &vec![0u32; 2 * d.row_ids()])?;
    let mut sel_w = upload(h, &vec![0f32; 2 * d.row_ids()])?;
    // The grouped GEMM's windows: the shared-tensor refusals come before any
    // window's length is read, so each is a few words.
    let weights_b = upload(h, &[0u8; 64])?;
    let x_b = upload(h, &[0f32; 64])?;
    let mut y_b = upload(h, &[0f32; 64])?;
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
                d.per_row,
                d.experts,
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
                d.per_row,
                d.experts - 1,
                words,
            ),
        )?;
        expect(
            "a row of slots that is not the instance's",
            "slots a row",
            table(
                &mut enc,
                &mut sorted,
                &mut inv,
                &mut tiles,
                slots,
                2 * d.per_row,
                d.experts,
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
                d.per_row,
                d.experts,
                words - 4,
            ),
        )?;
        for (what, needle, rows, top_k, stride) in [
            ("no rows", "rows of 0", 0usize, d.top_k, d.logit_stride),
            (
                "a top_k past the router's",
                "top_k",
                2,
                d.top_k + 1,
                d.logit_stride,
            ),
            (
                "a stride shorter than the row",
                "stride",
                2,
                d.top_k,
                d.rows - 1,
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
                            .window(Elems(0), Elems(2 * d.logit_stride))
                            .map_err(why)?,
                        logit_stride: stride,
                        ids: sel_ids
                            .window_mut(Elems(0), Elems(2 * d.row_ids()))
                            .map_err(why)?,
                        weights: sel_w
                            .window_mut(Elems(0), Elems(2 * d.row_ids()))
                            .map_err(why)?,
                        fault: h.device.fault_word(),
                        layer: LAYER,
                    },
                ),
            )?;
        }
        // The grouped GEMM's up and shared tensors: the pair takes the up's
        // stack (and the shared up with a shared expert), the plain tile
        // neither, and a shared tensor is given exactly when the router has
        // a shared row.
        let shared_needle = if d.shared == 1 {
            "has a shared group"
        } else {
            "has no shared group"
        };
        for (what, needle, kind, up, shared, shared_up) in [
            (
                "a pair with no up stack",
                "up's stack",
                RoutedKind::Pair,
                false,
                d.shared == 1,
                d.shared == 1,
            ),
            (
                "a plain tile with an up stack",
                "takes neither",
                RoutedKind::Plain,
                true,
                d.shared == 1,
                false,
            ),
            (
                "a shared tensor that disagrees with the router's shared row",
                shared_needle,
                RoutedKind::Pair,
                true,
                d.shared == 0,
                d.shared == 0,
            ),
        ] {
            let w = |on: bool| {
                on.then(|| weights_b.window(Elems(0), Elems(1)))
                    .transpose()
                    .map_err(why)
            };
            expect(
                what,
                needle,
                h.kernels.gemm_routed(
                    &mut enc,
                    RoutedParams {
                        quant: Quant::Q4K,
                        shared_quant: Quant::Q4K,
                        kind,
                        operand: Operand::F16,
                        cols: d.k,
                        rows: 64,
                        experts: d.experts,
                        stack: weights_b.window(Elems(0), Elems(1)).map_err(why)?,
                        stack_up: w(up)?,
                        shared: w(shared)?,
                        shared_up: w(shared_up)?,
                        x: x_b.window(Elems(0), Elems(1)).map_err(why)?,
                        x_rows: 1,
                        x_stride: d.k,
                        gather: true,
                        y: y_b.window_mut(Elems(0), Elems(1)).map_err(why)?,
                        y_stride: 64,
                        slots: 1,
                        max_tiles: 1,
                        sorted: sorted.window(Elems(0), Elems(1)).map_err(why)?,
                        route_w: x_b.window(Elems(0), Elems(1)).map_err(why)?,
                        tiles: tiles.window(Elems(0), Elems(4)).map_err(why)?,
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

/// The clauses of the whole table: the routing and the grouped experts at the
/// instance whose gate runs every clause, with the A4 family of its MLX
/// fixture.
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
            Box::new(move || match (ctx.fixture, mlx) {
                (Ok(fixture), Ok(a4)) => grouped_gemm(ctx.harness, fixture, Some(a4)),
                (Err(e), _) => Err(e.clone()),
                (_, Err(e)) => Err(format!("the A4 fixture: {e}")),
            }),
        ),
        (
            "grouped_oracle",
            Box::new(move || match (ctx.fixture, mlx) {
                (Ok(fixture), Ok(a4)) => grouped_oracle(ctx.harness, fixture, Some(a4)),
                (Err(e), _) => Err(e.clone()),
                (_, Err(e)) => Err(format!("the A4 fixture: {e}")),
            }),
        ),
        ("combine", Box::new(|| ctx.with(combine))),
        ("headnorm_rows", Box::new(|| headnorm_rows(ctx.harness))),
        ("route_refusals", Box::new(|| ctx.with(route_refusals))),
    ]
}

/// The clauses of the instance whose gate runs the kernels its geometry
/// shapes: the routing and the grouped experts of a router with no shared row
/// and layers of one weight family (Q4_K, with a Q6_K down in some) beside the
/// A4 family of its MLX fixture, without the head-norm scale of the delta net.
pub fn shaped<'a>(
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
            Box::new(move || match (ctx.fixture, mlx) {
                (Ok(fixture), Ok(a4)) => grouped_gemm(ctx.harness, fixture, Some(a4)),
                (Err(e), _) => Err(e.clone()),
                (_, Err(e)) => Err(format!("the A4 fixture: {e}")),
            }),
        ),
        ("grouped_q6_down", Box::new(|| ctx.with(grouped_q6_down))),
        (
            "grouped_oracle",
            Box::new(move || match (ctx.fixture, mlx) {
                (Ok(fixture), Ok(a4)) => grouped_oracle(ctx.harness, fixture, Some(a4)),
                (Err(e), _) => Err(e.clone()),
                (_, Err(e)) => Err(format!("the A4 fixture: {e}")),
            }),
        ),
        ("combine", Box::new(|| ctx.with(combine))),
        ("route_refusals", Box::new(|| ctx.with(route_refusals))),
    ]
}
