//! The output head's clauses.
//!
//! PIN(2026-10-22): the logit band is `oracle::band(Quant::Q6K, nblocks, Kind::Norm)`.
//! Its operation count is the one that function already derives for the norm on
//! the load (the Q6_K store path, the gain product, the scale product, and the
//! scale's own path). The normed input is `oracle::rms_norm`. No second oracle.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::kernels::{
    HEAD_K, HEAD_SIMD_GROUPS, HEAD_THREADS, HeadParams, Quant, ROWS_PER_SIMD_GROUP, head_groups,
};
use gguf::Gguf;
use gguf::quant::{GgmlType, dequant_row};

use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::{Fixture, OUTPUT_ROWS};
use super::oracle::{self, Kind};
use super::{Harness, SENTINEL_BITS};

/// The rows one threadgroup owns.
const ROWS_PER_GROUP: usize = HEAD_SIMD_GROUPS * ROWS_PER_SIMD_GROUP;

const _: () = assert!(HEAD_K == 2_048);
const _: () = assert!(HEAD_SIMD_GROUPS == 8);
const _: () = assert!(HEAD_THREADS == 256);
const _: () = assert!(ROWS_PER_GROUP == HEAD_SIMD_GROUPS * ROWS_PER_SIMD_GROUP);
// The fixture's head is the full grid the shipped dispatch runs.
const _: () = assert!(head_groups(OUTPUT_ROWS) == 7_760);
const _: () = assert!(head_groups(OUTPUT_ROWS) * ROWS_PER_GROUP == OUTPUT_ROWS);

const OUTPUT: &str = "output.weight";
const GAIN: &str = "output_norm.weight";

/// The block's scale, a little-endian f16 at the end of a Q6_K block.
const Q6K_D: usize = 208;

pub struct Out {
    pub logits: Vec<f32>,
    pub id: u32,
    pub ticket: u32,
}

fn block_bytes() -> usize {
    Quant::Q6K.block_bytes()
}

fn row_bytes(k: usize) -> usize {
    k / 256 * block_bytes()
}

/// The fixture's head tensors are the shipped shape.
fn expect_head(fixture: &Fixture) -> Result<(), String> {
    let file = Gguf::open(&fixture.path).map_err(|e| format!("{OUTPUT}: {e}"))?;
    let weights = file
        .find(OUTPUT)
        .ok_or_else(|| format!("the fixture has no tensor {OUTPUT}"))?;
    if weights.ty != GgmlType::Q6_K || weights.dims != [HEAD_K as u64, OUTPUT_ROWS as u64] {
        return Err(format!(
            "{OUTPUT} is {:?} {:?}, the gate reads Q6_K [{HEAD_K}, {OUTPUT_ROWS}]",
            weights.ty, weights.dims
        ));
    }
    let gain = file
        .find(GAIN)
        .ok_or_else(|| format!("the fixture has no tensor {GAIN}"))?;
    if gain.ty != GgmlType::F32 || gain.dims != [HEAD_K as u64] {
        return Err(format!(
            "{GAIN} is {:?} {:?}, the head reads {HEAD_K} f32s",
            gain.ty, gain.dims
        ));
    }
    Ok(())
}

fn dequant(row: &[u8], cols: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; cols];
    dequant_row(GgmlType::Q6_K, row, &mut out).map_err(|e| format!("dequantizing a row: {e}"))?;
    Ok(out)
}

/// The packed key. `!i` is the 32-bit complement, zero-extended.
fn argmax_key(v: f32, i: u32) -> u64 {
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    let o = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    };
    ((o as u64) << 32) | ((!i) as u64)
}

pub fn host_argmax(logits: &[f32]) -> Result<(f32, u32), String> {
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (row, &v) in logits.iter().enumerate() {
        let i = u32::try_from(row).map_err(|_| format!("row {row} does not fit the step id"))?;
        let takes = v > best_v || (v == best_v && i < best_i);
        if takes {
            best_v = v;
            best_i = i;
        }
    }
    Ok((best_v, best_i))
}

fn group_key(logits: &[f32], base: usize, rows: usize) -> Result<u64, String> {
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    let end = (base + ROWS_PER_GROUP).min(rows);
    let Some(span) = end.checked_sub(base) else {
        return Ok(argmax_key(best_v, best_i));
    };
    for (row, &v) in logits.iter().enumerate().skip(base).take(span) {
        let i = u32::try_from(row).map_err(|_| format!("row {row} does not fit the step id"))?;
        let takes = v > best_v || (v == best_v && i < best_i);
        if takes {
            best_v = v;
            best_i = i;
        }
    }
    Ok(argmax_key(best_v, best_i))
}

/// One head launch's own inputs: what every caller of [`run`] hands it.
pub struct HeadRun<'a> {
    pub quant: Quant,
    pub weights: &'a [u8],
    pub x: &'a [f32],
    pub gain: &'a [f32],
    pub rows: usize,
    pub k: usize,
    pub eps: f32,
}

/// One head launch. Logits start as the sentinel, the ticket as 0, the step
/// id as `u32::MAX`.
pub fn run(h: &Harness<'_>, case: HeadRun<'_>) -> Result<Out, String> {
    let HeadRun {
        quant,
        weights,
        x,
        gain,
        rows,
        k,
        eps,
    } = case;
    let (device, stream) = (h.device, h.stream);
    let groups = head_groups(rows);
    let weights_b = device.upload(stream, weights).map_err(why)?;
    let x_b = device.upload(stream, x).map_err(why)?;
    let gain_b = device.upload(stream, gain).map_err(why)?;
    let mut logits_b = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); rows])
        .map_err(why)?;
    let mut candidates_b = device.upload(stream, &vec![0u64; groups]).map_err(why)?;
    let mut ticket_b = device.upload(stream, &[0]).map_err(why)?;
    let mut step_b = device.upload(stream, &[u32::MAX]).map_err(why)?;
    let fault = device.fault_word();
    h.encode(|encoder| {
        h.kernels.head(
            encoder,
            HeadParams {
                quant,
                rows,
                k,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: weights_b.window(Elems(0), Elems(weights.len()))?,
                x: x_b.window(Elems(0), Elems(x.len()))?,
                gain: gain_b.window(Elems(0), Elems(gain.len()))?,
                logits: logits_b.window_mut(Elems(0), Elems(rows))?,
                candidates: candidates_b.window_mut(Elems(0), Elems(groups))?,
                ticket: ticket_b.window_mut(Elems(0), Elems(1))?,
                step_id: step_b.window_mut(Elems(0), Elems(1))?,
                eps,
                fault,
                layer: LAYER,
            },
        )
    })?;
    let mut logits = vec![0f32; rows];
    let mut id = [0u32];
    let mut ticket = [1u32];
    device
        .read(stream, logits_b.whole().map_err(why)?, &mut logits)
        .map_err(why)?;
    device
        .read(stream, step_b.whole().map_err(why)?, &mut id)
        .map_err(why)?;
    device
        .read(stream, ticket_b.whole().map_err(why)?, &mut ticket)
        .map_err(why)?;
    Ok(Out {
        logits,
        id: id[0],
        ticket: ticket[0],
    })
}

fn norm_band(k: usize) -> oracle::Band {
    oracle::band(Quant::Q6K, k / 256, Kind::Norm)
}

/// The f64 logit of one row with gain 1, and the width of that row plus a zero
/// row. A non-positive logit is not a margin.
fn margin(row: &[u8], k: usize, eps: f32) -> Result<(f64, f64), String> {
    let x = dequant(row, k)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Ok((0.0, f64::MAX));
    }
    let gain = vec![1f32; k];
    let xhat = oracle::rms_norm(&x, &gain, eps);
    let reference = oracle::reference(Quant::Q6K, row, 1, k, &xhat)?;
    let band = norm_band(k);
    let width = band.row_width(reference.magnitude[0]) + band.row_width(0.0);
    Ok((reference.y[0], width))
}

fn check_logits(
    name: &str,
    got: &[f32],
    weights: &[u8],
    x: &[f32],
    gain: &[f32],
    k: usize,
    eps: f32,
) -> Result<f64, String> {
    let xhat = oracle::rms_norm(x, gain, eps);
    let reference = oracle::reference(Quant::Q6K, weights, got.len(), k, &xhat)?;
    let band = norm_band(k);
    oracle::check(got, &reference, band, Kind::Norm, None).map_err(|e| format!("{name}: {e}"))
}

pub fn head_logit(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    h.clear_fault()?;
    expect_head(fixture)?;
    let eps = fixture.rms_eps()?;
    let gain = fixture.f32s(GAIN, HEAD_K)?;
    let samples = [
        ("output.weight head", 0usize, 32usize, 0x6EAD_u64),
        ("output.weight tail", OUTPUT_ROWS - 17, 17, 0x6EAE),
    ];
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, at, rows, seed) in samples {
        let weights = fixture.rows_at(OUTPUT, GgmlType::Q6_K, HEAD_K, at, rows)?;
        let x = oracle::activations(HEAD_K, seed, 0.5);
        let out = run(
            h,
            HeadRun {
                quant: Quant::Q6K,
                weights,
                x: &x,
                gain: &gain,
                rows,
                k: HEAD_K,
                eps,
            },
        )?;
        if let Some(f) = h.fault()? {
            return Err(format!("{name} raised {f}"));
        }
        if out.ticket != 0 {
            return Err(format!(
                "{name}: the ticket counter is {}, not 0",
                out.ticket
            ));
        }
        let ratio = check_logits(name, &out.logits, weights, &x, &gain, HEAD_K, eps)?;
        worst = worst.max(ratio);
        notes.push(format!("{name} {rows}x{HEAD_K} {ratio:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; {}",
        notes.join(", ")
    )))
}

/// `−0` and `+0` are one key, an equal value prefers the smaller index, a
/// negative loses to zero, and a larger value wins.
fn key_order() -> Result<(), String> {
    if argmax_key(0.0, 0) != argmax_key(-0.0, 0) {
        return Err("−0 and +0 are different keys".to_string());
    }
    if argmax_key(1.0, 1) <= argmax_key(1.0, 3) {
        return Err("an equal value did not prefer the smaller index".to_string());
    }
    if argmax_key(-1.0, 0) >= argmax_key(0.0, 4) {
        return Err("a negative value did not lose to zero".to_string());
    }
    if argmax_key(2.0, 9) <= argmax_key(1.0, 0) {
        return Err("a larger value did not win".to_string());
    }
    Ok(())
}

pub fn head_id(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    h.clear_fault()?;
    expect_head(fixture)?;
    key_order()?;
    let eps = fixture.rms_eps()?;
    let wide = wide_margin(h, fixture, eps)?;
    let tie = bit_tie(h, fixture, eps)?;
    if let Some(f) = h.fault()? {
        return Err(format!("head-id raised {f}"));
    }
    Ok(Verdict::Noted(format!("{wide}; {tie}")))
}

/// A real output row whose logit, normed by itself at gain 1, is a wide positive
/// margin over a zero row: the row, that logit and the margin's width.
fn wide_row(fixture: &Fixture, eps: f32) -> Result<(&[u8], f64, f64), String> {
    let bytes = row_bytes(HEAD_K);
    let pool = fixture.rows(OUTPUT, GgmlType::Q6_K, HEAD_K, 32)?;
    for row in pool.chunks_exact(bytes) {
        let (y, width) = margin(row, HEAD_K, eps)?;
        if y.is_finite() && y > 0.0 && y > width {
            return Ok((row, y, width));
        }
    }
    Err("construction: no output row is a wide positive margin over a zero row".to_string())
}

fn wide_margin(h: &Harness<'_>, fixture: &Fixture, eps: f32) -> Result<String, String> {
    const WINNER: usize = 100;
    let bytes = row_bytes(HEAD_K);
    let (row, y, width) = wide_row(fixture, eps)?;
    let mut weights = vec![0u8; OUTPUT_ROWS * bytes];
    weights[WINNER * bytes..(WINNER + 1) * bytes].copy_from_slice(row);
    let x = dequant(row, HEAD_K)?;
    let gain = vec![1f32; HEAD_K];
    let out = run(
        h,
        HeadRun {
            quant: Quant::Q6K,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: OUTPUT_ROWS,
            k: HEAD_K,
            eps,
        },
    )?;
    if out.ticket != 0 {
        return Err(format!(
            "head-id wide: the ticket counter is {}, not 0",
            out.ticket
        ));
    }
    let gap = y;
    if gap <= width {
        println!("head-id: wide-margin gap {gap:.3e} is inside the band {width:.3e}");
        return Ok("wide printed".to_string());
    }
    let want = u32::try_from(WINNER).map_err(|_| "the winner index does not fit the step id")?;
    if out.id != want {
        return Err(format!(
            "head-id wide: step id is {}, the f64 argmax is {want}",
            out.id
        ));
    }
    Ok(format!("wide id {want}"))
}

fn bit_tie(h: &Harness<'_>, fixture: &Fixture, eps: f32) -> Result<String, String> {
    const ROWS: usize = 32;
    const K: usize = 256;
    const LOW: usize = 5;
    const HIGH: usize = 9;
    let bytes = row_bytes(HEAD_K);
    let block = block_bytes();
    let pool = fixture.rows(OUTPUT, GgmlType::Q6_K, HEAD_K, 64)?;
    let mut chosen: Option<Vec<u8>> = None;
    for row in pool.chunks_exact(bytes) {
        let one = &row[..block];
        let (y, width) = margin(one, K, eps)?;
        if y.is_finite() && y > 0.0 && y > width {
            chosen = Some(one.to_vec());
            break;
        }
    }
    let Some(good) = chosen else {
        return Err(
            "construction: no Q6_K block is a wide positive margin over a zero row".to_string(),
        );
    };
    let mut weights = vec![0u8; ROWS * block];
    weights[LOW * block..(LOW + 1) * block].copy_from_slice(&good);
    weights[HIGH * block..(HIGH + 1) * block].copy_from_slice(&good);
    let x = dequant(&good, K)?;
    let gain = vec![1f32; K];
    let out = run(
        h,
        HeadRun {
            quant: Quant::Q6K,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: ROWS,
            k: K,
            eps,
        },
    )?;
    if out.ticket != 0 {
        return Err(format!(
            "head-id tie: the ticket counter is {}, not 0",
            out.ticket
        ));
    }
    if out.logits[LOW].to_bits() != out.logits[HIGH].to_bits() {
        return Err(format!(
            "head-id tie: rows {LOW} and {HIGH} are {:e} and {:e}, not the same bits",
            out.logits[LOW], out.logits[HIGH]
        ));
    }
    let tied = out.logits[LOW];
    if !tied.is_finite() {
        return Err(format!(
            "head-id tie: the tied logit {tied:e} is not finite"
        ));
    }
    for (i, &v) in out.logits.iter().enumerate() {
        if i == LOW || i == HIGH {
            continue;
        }
        let above = tied > v;
        if !above {
            return Err(format!(
                "head-id tie: row {i} is {v:e}, not strictly under the tied {tied:e}"
            ));
        }
    }
    let low = u32::try_from(LOW).map_err(|_| "the tie index does not fit the step id")?;
    if out.id != low {
        return Err(format!(
            "head-id tie: step id is {}, the smaller index is {low}",
            out.id
        ));
    }
    Ok(format!("tie id {low}"))
}

/// The head over a row count no threadgroup divides, as a vocabulary of 248,144
/// is: the grid has full groups and a partial last one, and the true argmax is
/// planted in the partial group's last row. A grid that leaves that group out,
/// or a reduction that does not read its candidate, loses the winner; the
/// logits of the last two groups are held to the f64 oracle.
pub fn head_partial(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    /// The rows of the launch: 7,754 full groups and a last group of 16.
    const ROWS: usize = 248_144;
    /// The first row of the last group.
    const LAST_BASE: usize = ROWS / ROWS_PER_GROUP * ROWS_PER_GROUP;
    /// The rows the last group owns.
    const LAST_ROWS: usize = ROWS - LAST_BASE;
    /// The rows the oracle reads back: the last full group and the last one.
    const CHECKED: usize = ROWS_PER_GROUP + LAST_ROWS;
    /// The planted winner: the last row the head owns.
    const WINNER: usize = ROWS - 1;
    const _: () = assert!(LAST_ROWS > 0 && LAST_ROWS < ROWS_PER_GROUP);
    h.clear_fault()?;
    expect_head(fixture)?;
    let eps = fixture.rms_eps()?;
    let bytes = row_bytes(HEAD_K);
    let (row, _, _) = wide_row(fixture, eps)?;
    let mut weights = vec![0u8; ROWS * bytes];
    weights[WINNER * bytes..(WINNER + 1) * bytes].copy_from_slice(row);
    let x = dequant(row, HEAD_K)?;
    let gain = vec![1f32; HEAD_K];
    let out = run(
        h,
        HeadRun {
            quant: Quant::Q6K,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: ROWS,
            k: HEAD_K,
            eps,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("head-partial raised {f}"));
    }
    if out.ticket != 0 {
        return Err(format!(
            "head-partial: the ticket counter is {}, not 0",
            out.ticket
        ));
    }
    let from = ROWS - CHECKED;
    let ratio = check_logits(
        "head-partial last groups",
        &out.logits[from..],
        &weights[from * bytes..],
        &x,
        &gain,
        HEAD_K,
        eps,
    )?;
    let want = u32::try_from(WINNER).map_err(|_| "the winner index does not fit the step id")?;
    if out.id != want {
        return Err(format!(
            "head-partial: step id is {}, the f64 argmax is {want}",
            out.id
        ));
    }
    Ok(Verdict::Noted(format!(
        "{ROWS} rows in {} groups, the last of {LAST_ROWS}: id {want}, last {CHECKED} rows {ratio:.3}",
        head_groups(ROWS)
    )))
}

pub fn head_ticket(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    const LAUNCHES: usize = 1_000;
    const BATCH: usize = 100;
    /// The threadgroups the ticket handshake runs, whatever the geometry.
    const GROUPS: usize = 2_304;
    const ROWS: usize = ROWS_PER_GROUP * GROUPS;
    const K: usize = 256;
    h.clear_fault()?;
    expect_head(fixture)?;
    let groups = ROWS.div_ceil(ROWS_PER_GROUP);
    if groups != GROUPS {
        return Err(format!(
            "{ROWS} rows are {groups} threadgroups, the clause runs {GROUPS}"
        ));
    }
    let eps = fixture.rms_eps()?;
    let full = fixture.rows(OUTPUT, GgmlType::Q6_K, HEAD_K, ROWS)?;
    let block = block_bytes();
    let wide = row_bytes(HEAD_K);
    let mut weights = vec![0u8; ROWS * block];
    for (row, dst) in full.chunks_exact(wide).zip(weights.chunks_exact_mut(block)) {
        dst.copy_from_slice(&row[..block]);
    }
    let mut xs = vec![0f32; LAUNCHES * K];
    let mut rng = oracle::Rng::new(0x6EAD_7100);
    for v in &mut xs {
        *v = 0.5 * rng.signed();
    }
    let (device, stream) = (h.device, h.stream);
    let weights_b = device.upload(stream, &weights).map_err(why)?;
    let x_b = device.upload(stream, &xs).map_err(why)?;
    let gain = vec![1f32; K];
    let gain_b = device.upload(stream, &gain).map_err(why)?;
    let mut logits_b = device
        .upload(
            stream,
            &vec![f32::from_bits(SENTINEL_BITS); LAUNCHES * ROWS],
        )
        .map_err(why)?;
    let mut candidates_b = device
        .upload(stream, &vec![0u64; LAUNCHES * groups])
        .map_err(why)?;
    let mut ticket_b = device.upload(stream, &[0]).map_err(why)?;
    let mut step_b = device
        .upload(stream, &vec![u32::MAX; LAUNCHES])
        .map_err(why)?;
    let fault = device.fault_word();
    for batch in (0..LAUNCHES).step_by(BATCH) {
        let end = (batch + BATCH).min(LAUNCHES);
        h.encode(|encoder| {
            for launch in batch..end {
                h.kernels.head(
                    encoder,
                    HeadParams {
                        quant: Quant::Q6K,
                        rows: ROWS,
                        k: K,
                        simdgroups: HEAD_SIMD_GROUPS,
                        weights: weights_b.window(Elems(0), Elems(weights.len()))?,
                        x: x_b.window(Elems(launch * K), Elems(K))?,
                        gain: gain_b.window(Elems(0), Elems(K))?,
                        logits: logits_b.window_mut(Elems(launch * ROWS), Elems(ROWS))?,
                        candidates: candidates_b
                            .window_mut(Elems(launch * groups), Elems(groups))?,
                        ticket: ticket_b.window_mut(Elems(0), Elems(1))?,
                        step_id: step_b.window_mut(Elems(launch), Elems(1))?,
                        eps,
                        fault,
                        layer: LAYER,
                    },
                )?;
            }
            Ok(())
        })?;
        let mut word = [1u32];
        device
            .read(
                stream,
                ticket_b.window(Elems(0), Elems(1)).map_err(why)?,
                &mut word,
            )
            .map_err(why)?;
        if word[0] != 0 {
            return Err(format!(
                "after {end} launches the ticket counter is {}, not 0",
                word[0]
            ));
        }
    }
    let mut logits = vec![0f32; LAUNCHES * ROWS];
    let mut ids = vec![0u32; LAUNCHES];
    let mut candidates = vec![0u64; LAUNCHES * groups];
    device
        .read(stream, logits_b.whole().map_err(why)?, &mut logits)
        .map_err(why)?;
    device
        .read(stream, step_b.whole().map_err(why)?, &mut ids)
        .map_err(why)?;
    device
        .read(stream, candidates_b.whole().map_err(why)?, &mut candidates)
        .map_err(why)?;
    if let Some(f) = h.fault()? {
        return Err(format!("head-ticket raised {f}"));
    }
    for launch in 0..LAUNCHES {
        let rows = &logits[launch * ROWS..(launch + 1) * ROWS];
        if rows.iter().any(|v| !v.is_finite()) {
            return Err(format!("launch {launch} wrote a non-finite logit"));
        }
        let (_, want) = host_argmax(rows)?;
        if ids[launch] != want {
            return Err(format!(
                "launch {launch} step id is {}, the logits select {want}",
                ids[launch]
            ));
        }
        let packed = &candidates[launch * groups..(launch + 1) * groups];
        for (g, &got) in packed.iter().enumerate() {
            let key = group_key(rows, g * ROWS_PER_GROUP, ROWS)?;
            if got != key {
                return Err(format!(
                    "launch {launch} group {g} packed {got:#x}, the group's argmax is {key:#x}"
                ));
            }
        }
    }
    Ok(Verdict::Noted(format!(
        "{LAUNCHES} launches, ids and candidates match the logits, ticket is 0"
    )))
}

pub fn head_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    const ROWS: usize = 32;
    const K: usize = 256;
    const NAN_ROW: usize = 0;
    const WINNER: usize = 7;
    h.clear_fault()?;
    expect_head(fixture)?;
    let eps = fixture.rms_eps()?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(format!("construction: eps {eps} is not usable"));
    }
    let bytes = row_bytes(HEAD_K);
    let block = block_bytes();
    let pool = fixture.rows(OUTPUT, GgmlType::Q6_K, HEAD_K, 64)?;
    let mut chosen: Option<Vec<u8>> = None;
    for row in pool.chunks_exact(bytes) {
        let one = &row[..block];
        let (y, width) = margin(one, K, eps)?;
        if y.is_finite() && y > width {
            chosen = Some(one.to_vec());
            break;
        }
    }
    let Some(good) = chosen else {
        return Err(
            "construction: no Q6_K block is a wide positive margin over a zero row".to_string(),
        );
    };
    let x = dequant(&good, K)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err("construction: the row's dequant is not finite".to_string());
    }
    let mut weights = vec![0u8; ROWS * block];
    {
        let nan_row = &mut weights[NAN_ROW * block..(NAN_ROW + 1) * block];
        nan_row.copy_from_slice(&good);
        nan_row[Q6K_D] = 0x00;
        nan_row[Q6K_D + 1] = 0x7E;
    }
    weights[WINNER * block..(WINNER + 1) * block].copy_from_slice(&good);
    let gain = vec![1f32; K];
    let out = run(
        h,
        HeadRun {
            quant: Quant::Q6K,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: ROWS,
            k: K,
            eps,
        },
    )?;
    if !out.logits[NAN_ROW].is_nan() {
        return Err(format!(
            "the NaN scale's logit is {:e}, not NaN",
            out.logits[NAN_ROW]
        ));
    }
    common::expect_fault(h, "a NaN scale", FaultSite::Logit)?;
    let (_, want) = host_argmax(&out.logits)?;
    let winner = u32::try_from(WINNER).map_err(|_| "the winner index does not fit the step id")?;
    if want != winner || out.id != winner {
        return Err(format!(
            "head-nan: step id is {}, the finite rows select {want}, the winner is {winner}",
            out.id
        ));
    }
    if out.ticket != 0 {
        return Err(format!(
            "head-nan: the ticket counter is {}, not 0",
            out.ticket
        ));
    }
    Ok(Verdict::Noted(
        "logit site, the NaN stays stored, the id is the finite argmax".to_string(),
    ))
}

/// Every logit is NaN. The step id is not a token.
pub fn head_all_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    const ROWS: usize = 32;
    const K: usize = 256;
    h.clear_fault()?;
    expect_head(fixture)?;
    let eps = fixture.rms_eps()?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(format!("construction: eps {eps} is not usable"));
    }
    let bytes = row_bytes(HEAD_K);
    let block = block_bytes();
    let pool = fixture.rows(OUTPUT, GgmlType::Q6_K, HEAD_K, 64)?;
    let mut chosen: Option<Vec<u8>> = None;
    for row in pool.chunks_exact(bytes) {
        let one = &row[..block];
        let (y, width) = margin(one, K, eps)?;
        if y.is_finite() && y > width {
            chosen = Some(one.to_vec());
            break;
        }
    }
    let Some(good) = chosen else {
        return Err(
            "construction: no Q6_K block is a wide positive margin over a zero row".to_string(),
        );
    };
    let x = dequant(&good, K)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err("construction: the row's dequant is not finite".to_string());
    }
    let mut weights = vec![0u8; ROWS * block];
    for row in 0..ROWS {
        let dst = &mut weights[row * block..(row + 1) * block];
        dst.copy_from_slice(&good);
        dst[Q6K_D] = 0x00;
        dst[Q6K_D + 1] = 0x7E;
    }
    let gain = vec![1f32; K];
    let out = run(
        h,
        HeadRun {
            quant: Quant::Q6K,
            weights: &weights,
            x: &x,
            gain: &gain,
            rows: ROWS,
            k: K,
            eps,
        },
    )?;
    if out.logits.iter().any(|v| !v.is_nan()) {
        return Err("an all-NaN head stored a finite logit".to_string());
    }
    if out.id != u32::MAX {
        return Err(format!("an all-NaN head stored id {}, a token id", out.id));
    }
    common::expect_fault(h, "an all-NaN head", FaultSite::Logit)?;
    if out.ticket != 0 {
        return Err(format!(
            "head-all-nan: the ticket counter is {}, not 0",
            out.ticket
        ));
    }
    Ok(Verdict::Noted(
        "logit site, every logit stays NaN, the id is not a token".to_string(),
    ))
}

/// The launch of one case: a width, a row count, a SIMD-group count, an
/// epsilon or a window, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    k: usize,
    rows: usize,
    simdgroups: usize,
    eps: f32,
    weights: usize,
    x: usize,
    gain: usize,
    logits: usize,
    candidates: usize,
    step_id: usize,
    ticket: usize,
}

impl Args {
    const fn whole(k: usize, rows: usize, weights_len: usize) -> Self {
        Self {
            k,
            rows,
            simdgroups: HEAD_SIMD_GROUPS,
            eps: 1e-5,
            weights: weights_len,
            x: k,
            gain: k,
            logits: rows,
            candidates: 1,
            step_id: 1,
            ticket: 1,
        }
    }
}

/// The buffers a bounds launch slices. One struct, so the launch stays under
/// the argument limit.
struct BoundBufs<'a> {
    weights: &'a MetalBuffer<u8>,
    x: &'a MetalBuffer<f32>,
    gain: &'a MetalBuffer<f32>,
    logits: &'a mut MetalBuffer<f32>,
    candidates: &'a mut MetalBuffer<u64>,
    ticket: &'a mut MetalBuffer<u32>,
    step_id: &'a mut MetalBuffer<u32>,
}

fn refuse(
    h: &Harness<'_>,
    bufs: &mut BoundBufs<'_>,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    h.kernels.head(
        encoder,
        HeadParams {
            quant: Quant::Q6K,
            rows: a.rows,
            k: a.k,
            simdgroups: a.simdgroups,
            weights: bufs.weights.window(Elems(0), Elems(a.weights))?,
            x: bufs.x.window(Elems(0), Elems(a.x))?,
            gain: bufs.gain.window(Elems(0), Elems(a.gain))?,
            logits: bufs.logits.window_mut(Elems(0), Elems(a.logits))?,
            candidates: bufs.candidates.window_mut(Elems(0), Elems(a.candidates))?,
            ticket: bufs.ticket.window_mut(Elems(0), Elems(a.ticket))?,
            step_id: bufs.step_id.window_mut(Elems(0), Elems(a.step_id))?,
            eps: a.eps,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
    whole: Args,
) -> BadArg<'a, BoundBufs<'a>> {
    BadArg::new(
        what,
        needle,
        Box::new(move |bufs, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, bufs, &a, encoder)
        }),
    )
}

pub fn head_bounds(h: &Harness<'_>) -> Clause {
    const K: usize = 256;
    const ROWS: usize = 16;
    let block = block_bytes();
    let weights_len = ROWS * block;
    let (device, stream) = (h.device, h.stream);
    let weights = device
        .upload(stream, &vec![0u8; weights_len + 4])
        .map_err(why)?;
    let x = device.upload(stream, &vec![0f32; K + 4]).map_err(why)?;
    let gain = device.upload(stream, &vec![1f32; K + 4]).map_err(why)?;
    let mut logits = device
        .upload(stream, &[f32::from_bits(SENTINEL_BITS); ROWS + 4])
        .map_err(why)?;
    let mut candidates = device.upload(stream, &[0u64; 4]).map_err(why)?;
    let mut ticket = device.upload(stream, &[u32::MAX, u32::MAX]).map_err(why)?;
    let mut step_id = device.upload(stream, &[u32::MAX, u32::MAX]).map_err(why)?;
    let mut bufs = BoundBufs {
        weights: &weights,
        x: &x,
        gain: &gain,
        logits: &mut logits,
        candidates: &mut candidates,
        ticket: &mut ticket,
        step_id: &mut step_id,
    };
    let whole = Args::whole(K, ROWS, weights_len);
    let case = |what: &'static str, needle: &'static str, edit: Box<dyn FnOnce(&mut Args)>| {
        bad(what, needle, edit, h, whole)
    };
    let cases = vec![
        case("K of 0", "multiple of 256", Box::new(|a| a.k = 0)),
        case("K of 128", "multiple of 256", Box::new(|a| a.k = 128)),
        case("K of 255", "multiple of 256", Box::new(|a| a.k = 255)),
        case("no rows", "row count", Box::new(|a| a.rows = 0)),
        case(
            "a non-finite eps",
            "not finite",
            Box::new(|a| a.eps = f32::NAN),
        ),
        case(
            "an infinite eps",
            "not finite",
            Box::new(|a| a.eps = f32::INFINITY),
        ),
        case(
            "a short x window",
            "x window",
            Box::new(|a| a.x = whole.x - 1),
        ),
        case(
            "a long x window",
            "x window",
            Box::new(|a| a.x = whole.x + 1),
        ),
        case(
            "a short gain window",
            "gain",
            Box::new(|a| a.gain = whole.gain - 1),
        ),
        case(
            "a long gain window",
            "gain",
            Box::new(|a| a.gain = whole.gain + 1),
        ),
        case(
            "a short logits window",
            "logits",
            Box::new(|a| a.logits = whole.logits - 1),
        ),
        case(
            "a long logits window",
            "logits",
            Box::new(|a| a.logits = whole.logits + 1),
        ),
        case(
            "an empty candidates window",
            "candidates",
            Box::new(|a| a.candidates = 0),
        ),
        case(
            "a long candidates window",
            "candidates",
            Box::new(|a| a.candidates = 2),
        ),
        case(
            "an empty step id window",
            "step id",
            Box::new(|a| a.step_id = 0),
        ),
        case(
            "a long step id window",
            "step id",
            Box::new(|a| a.step_id = 2),
        ),
        case(
            "an empty ticket window",
            "ticket",
            Box::new(|a| a.ticket = 0),
        ),
        case("a long ticket window", "ticket", Box::new(|a| a.ticket = 2)),
        case(
            "a short weights window",
            "weights",
            Box::new(|a| a.weights = whole.weights - 1),
        ),
        case(
            "a long weights window",
            "weights",
            Box::new(|a| a.weights = whole.weights + 1),
        ),
        case(
            "no SIMD groups",
            "SIMD-group",
            Box::new(|a| a.simdgroups = 0),
        ),
        case(
            "4 SIMD groups",
            "SIMD-group",
            Box::new(|a| a.simdgroups = 4),
        ),
        case(
            "16 SIMD groups",
            "SIMD-group",
            Box::new(|a| a.simdgroups = 16),
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &BoundBufs<'_>| {
        common::still(h, b.logits, |v| v.to_bits() == SENTINEL_BITS)?;
        common::still(h, b.ticket, |v| *v == u32::MAX)?;
        common::still(h, b.step_id, |v| *v == u32::MAX)
    };
    let count = common::refuse_launches(h, "head", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, BoundBufs<'_>> = Box::new(|bufs, encoder| {
        let a = whole;
        refuse(h, bufs, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, b.logits, |v| v.to_bits() != SENTINEL_BITS)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals left the outputs unchanged, and an accepted launch wrote"
    )))
}

/// The output-head clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("head_logit", Box::new(|| ctx.with(head_logit))),
        ("head_id", Box::new(|| ctx.with(head_id))),
        ("head_partial", Box::new(|| ctx.with(head_partial))),
        ("head_ticket", Box::new(|| ctx.with(head_ticket))),
        ("head_nan", Box::new(|| ctx.with(head_nan))),
        ("head_all_nan", Box::new(|| ctx.with(head_all_nan))),
        ("head_bounds", Box::new(|| head_bounds(ctx.harness))),
    ]
}
