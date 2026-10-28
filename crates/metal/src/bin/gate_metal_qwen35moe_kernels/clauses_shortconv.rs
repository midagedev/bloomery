//! The short conv's clauses and the sandwich residual's.
//!
//! The kernels take every size as an argument, so these run at the LFM3
//! mixer's width (2048 channels) with conv widths 7 and 3, on seeded synthetic
//! weights and activations, with no fixture. The oracle is the f64 history of
//! `oracle_shortconv.rs`: a sequence's `u` position by position, with the ring
//! as the contract's cache of it. Every ring starts as large finite garbage, so
//! a read of a slot the contract does not name moves a bit.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::delta_geom::ring_rows as stores_ring_rows;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::{
    NormAddRowsParams, NormRowsParams, SHORTCONV_MAX_SLOTS, ShortConvRowsParams,
    ShortConvStepParams,
};
use bloomery_metal::router_geom;

use super::common::{self, BadArg, Edit, LAYER};
use super::oracle;
use super::oracle_shortconv::{self, SandwichIn, Sequence};
use super::{ClauseRun, Harness};

/// The channels of the mixer the clauses launch at.
const HIDDEN: usize = 2048;
/// One row of the in_proj output `B | C | x`.
const BCX_ROW: usize = 3 * HIDDEN;

/// The conv widths the clauses cover: LFM3's and LFM2.5's.
const TAPS: [usize; 2] = [7, 3];

/// A conv's width and its ring's rows.
#[derive(Clone, Copy)]
struct Conv {
    taps: usize,
    ring_rows: usize,
}

/// The conv weights of one layer: `hidden · taps`, a channel's taps together.
fn weights(taps: usize, seed: u64) -> Vec<f32> {
    let mut rng = oracle::Rng::new(seed);
    (0..HIDDEN * taps).map(|_| 0.5 * rng.signed()).collect()
}

/// `rows` rows of in_proj output, `B | C | x` of `HIDDEN` values each, dense;
/// every 61st value ten times larger.
fn rows_of(rows: usize, seed: u64) -> Vec<f32> {
    let mut rng = oracle::Rng::new(seed);
    (0..rows * 3 * HIDDEN)
        .map(|i| {
            let v = rng.signed();
            if i % 61 == 7 { 10.0 * v } else { v }
        })
        .collect()
}

/// A ring's contents before any position landed: finite, far from the values
/// the conv produces, so a stale read cannot hide.
fn garbage(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = oracle::Rng::new(seed);
    (0..len).map(|_| 100.0 + 50.0 * rng.signed()).collect()
}

fn read_buffer(h: &Harness<'_>, buffer: &MetalBuffer<f32>) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; buffer.len().get()];
    h.device
        .read(h.stream, buffer.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    Ok(out)
}

/// No output element is still the word the output began as.
fn written(y: &[f32]) -> Result<(), String> {
    match y.iter().position(|v| v.to_bits() == super::SENTINEL_BITS) {
        Some(i) => Err(format!("output element {i} was never written")),
        None => Ok(()),
    }
}

/// One chunk launch over the sequence's ring, `conv.ring_rows` rows at row
/// `ring_at` floats into `ring`. The output rows, dense.
fn launch_rows(
    h: &Harness<'_>,
    conv: Conv,
    w: &MetalBuffer<f32>,
    ring: &mut MetalBuffer<f32>,
    ring_at: usize,
    bcx: &[f32],
    pos: u32,
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let rows = bcx.len() / (3 * HIDDEN);
    let bcx_b = device.upload(stream, bcx).map_err(why)?;
    let mut y_b = device
        .upload(
            stream,
            &vec![f32::from_bits(super::SENTINEL_BITS); rows * HIDDEN],
        )
        .map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.shortconv_rows(
            encoder,
            ShortConvRowsParams {
                rows,
                bcx: bcx_b.window(Elems(0), Elems(bcx.len()))?,
                bcx_stride: 3 * HIDDEN,
                conv_w: w.window(Elems(0), Elems(HIDDEN * conv.taps))?,
                ring: ring.window_mut(Elems(ring_at), Elems(conv.ring_rows * HIDDEN))?,
                y: y_b.window_mut(Elems(0), Elems(rows * HIDDEN))?,
                y_stride: HIDDEN,
                faults: device.fault_word(),
                hidden: HIDDEN,
                taps: conv.taps,
                ring_rows: conv.ring_rows,
                layer: LAYER,
                pos,
            },
        )
    })?;
    let mut y = vec![0f32; rows * HIDDEN];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    written(&y)?;
    Ok(y)
}

/// One decode step over `pos.len()` slots, whose rings are the whole of
/// `ring`. The output rows, dense.
fn launch_step(
    h: &Harness<'_>,
    conv: Conv,
    w: &MetalBuffer<f32>,
    ring: &mut MetalBuffer<f32>,
    bcx: &[f32],
    pos: &[u32],
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let slots = pos.len();
    let bcx_b = device.upload(stream, bcx).map_err(why)?;
    let mut y_b = device
        .upload(
            stream,
            &vec![f32::from_bits(super::SENTINEL_BITS); slots * HIDDEN],
        )
        .map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.shortconv_step(
            encoder,
            ShortConvStepParams {
                bcx: bcx_b.window(Elems(0), Elems(bcx.len()))?,
                bcx_stride: 3 * HIDDEN,
                conv_w: w.window(Elems(0), Elems(HIDDEN * conv.taps))?,
                ring: ring.window_mut(Elems(0), Elems(slots * conv.ring_rows * HIDDEN))?,
                y: y_b.window_mut(Elems(0), Elems(slots * HIDDEN))?,
                y_stride: HIDDEN,
                faults: device.fault_word(),
                hidden: HIDDEN,
                taps: conv.taps,
                ring_rows: conv.ring_rows,
                layer: LAYER,
                pos,
            },
        )
    })?;
    let mut y = vec![0f32; slots * HIDDEN];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    written(&y)?;
    Ok(y)
}

/// A step of one slot from the ring `ring`: the output and the ring after.
fn step_from(
    h: &Harness<'_>,
    conv: Conv,
    w: &MetalBuffer<f32>,
    ring: &[f32],
    bcx: &[f32],
    pos: &[u32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut ring_b = h.device.upload(h.stream, ring).map_err(why)?;
    let y = launch_step(h, conv, w, &mut ring_b, bcx, pos)?;
    Ok((y, read_buffer(h, &ring_b)?))
}

/// A chunk from the ring `ring`: the output rows.
fn rows_from(
    h: &Harness<'_>,
    conv: Conv,
    w: &MetalBuffer<f32>,
    ring: &[f32],
    bcx: &[f32],
    pos: u32,
) -> Result<Vec<f32>, String> {
    let mut ring_b = h.device.upload(h.stream, ring).map_err(why)?;
    launch_rows(h, conv, w, &mut ring_b, 0, bcx, pos)
}

/// The rings of every slot in `ring` are the oracle's, bit for bit.
fn fit_rings(
    h: &Harness<'_>,
    ring: &MetalBuffer<f32>,
    seqs: &[Sequence],
    init: &[f32],
    conv: Conv,
    what: &str,
) -> Result<(), String> {
    let per = conv.ring_rows * HIDDEN;
    let mut want = Vec::with_capacity(init.len());
    for (s, seq) in seqs.iter().enumerate() {
        want.extend(seq.ring(conv.ring_rows, &init[s * per..(s + 1) * per])?);
    }
    oracle_shortconv::same_ring(&read_buffer(h, ring)?, &want, what)
}

/// A chunk's rows are the oracle's, in band: an in-chunk predecessor is the
/// chunk's own row, an earlier one the ring's, and the commit leaves the last
/// positions in their slots for the next chunk. Three chunks of one sequence
/// over a ring the kernel carries itself, the second opening inside the taps
/// the first left and the third longer than the ring, at both conv widths and
/// at the canonical ring and the smallest one.
pub fn shortconv_rows(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let mut parts = Vec::new();
    for taps in TAPS {
        let w = weights(taps, 0x5C01 + taps as u64);
        let w_b = device.upload(stream, &w).map_err(why)?;
        for ring_rows in [stores_ring_rows(taps), taps] {
            let conv = Conv { taps, ring_rows };
            let seed = 0x5C10 + (100 * taps + ring_rows) as u64;
            let init = garbage(ring_rows * HIDDEN, seed);
            let mut ring = device.upload(stream, &init).map_err(why)?;
            let mut seq = Sequence::new(HIDDEN, taps, &w)?;
            let mut pos = 0usize;
            let mut worst = 0.0f64;
            for (i, rows) in [9usize, 5, ring_rows + 6].into_iter().enumerate() {
                let bcx = rows_of(rows, seed + 1 + i as u64);
                let y = launch_rows(h, conv, &w_b, &mut ring, 0, &bcx, pos as u32)?;
                common::expect_clean(h, "a chunk")?;
                for row in bcx.as_chunks::<BCX_ROW>().0 {
                    seq.push(row)?;
                }
                let ratio = seq.fit_y(pos, &y).map_err(|e| {
                    format!("taps {taps}, ring rows {ring_rows}, chunk {i} at position {pos}: {e}")
                })?;
                worst = worst.max(ratio);
                oracle_shortconv::same_ring(
                    &read_buffer(h, &ring)?,
                    &seq.ring(ring_rows, &init)?,
                    &format!("taps {taps}, ring rows {ring_rows}, the ring after chunk {i}"),
                )?;
                pos += rows;
            }
            parts.push(format!("taps {taps} ring {ring_rows} {worst:.3}"));
        }
    }
    Ok(Verdict::Noted(format!(
        "three chunks a case, each in band and each ring bit for bit, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A predecessor before the sequence's first position is 0 whatever the ring
/// holds: a chunk and a step of four slots at position 0 are the bits of the
/// same launch over a zeroed ring. A chunk past the start does read the ring
/// (the control that makes the garbage visible).
pub fn shortconv_start(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let mut parts = Vec::new();
    for taps in TAPS {
        let conv = Conv {
            taps,
            ring_rows: stores_ring_rows(taps),
        };
        let per = conv.ring_rows * HIDDEN;
        let seed = 0x5C80 + 0x10 * taps as u64;
        let w = weights(taps, seed);
        let w_b = device.upload(stream, &w).map_err(why)?;
        let loud = garbage(4 * per, seed + 1);
        let quiet = vec![0f32; 4 * per];

        let rows = taps + 2;
        let bcx = rows_of(rows, seed + 2);
        let y_loud = rows_from(h, conv, &w_b, &loud[..per], &bcx, 0)?;
        let y_quiet = rows_from(h, conv, &w_b, &quiet[..per], &bcx, 0)?;
        if !common::bits_eq(&y_loud, &y_quiet) {
            return Err(format!("taps {taps}: a chunk at position 0 read the ring"));
        }
        let mut seq = Sequence::new(HIDDEN, taps, &w)?;
        for row in bcx.as_chunks::<BCX_ROW>().0 {
            seq.push(row)?;
        }
        let chunk = seq.fit_y(0, &y_loud)?;
        let after_loud = rows_from(h, conv, &w_b, &loud[..per], &bcx, taps as u32)?;
        let after_quiet = rows_from(h, conv, &w_b, &quiet[..per], &bcx, taps as u32)?;
        if common::bits_eq(&after_loud, &after_quiet) {
            return Err(format!(
                "taps {taps}: a chunk at position {taps} did not read the ring"
            ));
        }

        let slots = 4;
        let bcx4 = rows_of(slots, seed + 3);
        let pos = vec![0u32; slots];
        let (step_loud, _) = step_from(h, conv, &w_b, &loud, &bcx4, &pos)?;
        let (step_quiet, _) = step_from(h, conv, &w_b, &quiet, &bcx4, &pos)?;
        if !common::bits_eq(&step_loud, &step_quiet) {
            return Err(format!("taps {taps}: a step at position 0 read the ring"));
        }
        let mut step = 0.0f64;
        for (s, row) in bcx4.as_chunks::<BCX_ROW>().0.iter().enumerate() {
            let mut one = Sequence::new(HIDDEN, taps, &w)?;
            one.push(row)?;
            step = step.max(one.fit_y(0, &step_loud[s * HIDDEN..(s + 1) * HIDDEN])?);
        }
        parts.push(format!("taps {taps} chunk {chunk:.3} step {step:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "position 0 reads no ring slot in a chunk or a step, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// The prompt length of slot `slot`: shorter than the taps, inside the ring,
/// longer than the ring (the pass wrapped it), and one token.
fn prompt_len(slot: usize, taps: usize, ring_rows: usize) -> usize {
    match slot % 4 {
        0 => 2,
        1 => taps + 2,
        2 => ring_rows + 3,
        _ => 1,
    }
}

/// State carries: each slot's prompt pass leaves its ring, and decode steps of
/// every slot at once, each at its own position, read it and keep it. Slots
/// are independent (each its own ring and position), the ring wraps (the steps
/// outlast it), and a short prompt still reads zeros before its start. The
/// whole ring buffer is the oracle's after the pass and after every step.
pub fn shortconv_state(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let mut cases = Vec::new();
    for taps in TAPS {
        for ring_rows in [stores_ring_rows(taps), taps] {
            for slots in [1usize, 4] {
                cases.push((taps, ring_rows, slots));
            }
        }
    }
    cases.push((7, stores_ring_rows(7), 8));
    let mut parts = Vec::new();
    for (n, (taps, ring_rows, slots)) in cases.into_iter().enumerate() {
        let conv = Conv { taps, ring_rows };
        let case = format!("taps {taps}, ring rows {ring_rows}, {slots} slots");
        let seed = 0x5D00 + 0x100 * n as u64;
        let w = weights(taps, seed);
        let w_b = device.upload(stream, &w).map_err(why)?;
        let per = ring_rows * HIDDEN;
        let init = garbage(slots * per, seed + 1);
        let mut ring = device.upload(stream, &init).map_err(why)?;
        let mut seqs = Vec::new();
        for _ in 0..slots {
            seqs.push(Sequence::new(HIDDEN, taps, &w)?);
        }
        let mut worst = 0.0f64;
        let mut lens = Vec::new();
        for (s, seq) in seqs.iter_mut().enumerate() {
            let len = prompt_len(s, taps, ring_rows);
            lens.push(len);
            let first = len.div_ceil(2);
            let mut pos = 0usize;
            for (i, rows) in [first, len - first].into_iter().enumerate() {
                if rows == 0 {
                    continue;
                }
                let bcx = rows_of(rows, seed + 2 + 16 * s as u64 + i as u64);
                let y = launch_rows(h, conv, &w_b, &mut ring, s * per, &bcx, pos as u32)?;
                common::expect_clean(h, "a prompt chunk")?;
                for row in bcx.as_chunks::<BCX_ROW>().0 {
                    seq.push(row)?;
                }
                let ratio = seq
                    .fit_y(pos, &y)
                    .map_err(|e| format!("{case}, slot {s}, prompt chunk {i}: {e}"))?;
                worst = worst.max(ratio);
                pos += rows;
            }
        }
        fit_rings(
            h,
            &ring,
            &seqs,
            &init,
            conv,
            &format!("{case}, the rings after the prompt pass"),
        )?;
        for i in 0..ring_rows + taps + 2 {
            let bcx = rows_of(slots, seed + 0x80 + i as u64);
            let pos: Vec<u32> = lens.iter().map(|len| (len + i) as u32).collect();
            let y = launch_step(h, conv, &w_b, &mut ring, &bcx, &pos)?;
            common::expect_clean(h, "a decode step")?;
            for (s, row) in bcx.as_chunks::<BCX_ROW>().0.iter().enumerate() {
                seqs[s].push(row)?;
                let ratio = seqs[s]
                    .fit_y(lens[s] + i, &y[s * HIDDEN..(s + 1) * HIDDEN])
                    .map_err(|e| format!("{case}, slot {s}, step {i}: {e}"))?;
                worst = worst.max(ratio);
            }
            fit_rings(
                h,
                &ring,
                &seqs,
                &init,
                conv,
                &format!("{case}, the rings after step {i}"),
            )?;
        }
        parts.push(format!("{taps}/{ring_rows}/{slots} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "prompt passes in two chunks, then decode steps past the ring's wrap, each in band and every ring bit for bit; taps/ring rows/slots, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A value that is not finite raises the linear-conv site and is NaN at its
/// element alone: a B, a C or a weight that is not finite, a predecessor the
/// ring holds, and an in-chunk predecessor. The ring keeps the u it was given.
pub fn shortconv_nan(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let taps = 7;
    let conv = Conv {
        taps,
        ring_rows: stores_ring_rows(taps),
    };
    let per = conv.ring_rows * HIDDEN;
    let w = weights(taps, 0x5CC0);
    let w_b = device.upload(stream, &w).map_err(why)?;
    let mut ring0 = device.upload(stream, &vec![0f32; per]).map_err(why)?;
    launch_rows(h, conv, &w_b, &mut ring0, 0, &rows_of(10, 0x5CC1), 0)?;
    common::expect_clean(h, "the history chunk")?;
    let state = read_buffer(h, &ring0)?;
    let row = rows_of(1, 0x5CC2);
    let pos = [10u32];
    let (clean_y, clean_ring) = step_from(h, conv, &w_b, &state, &row, &pos)?;
    common::expect_clean(h, "the clean step")?;
    let slot = 10 % conv.ring_rows;

    let only = |y: &[f32], bad: usize, what: &str| -> Result<(), String> {
        for (c, (&got, &want)) in y.iter().zip(&clean_y).enumerate() {
            if c == bad {
                if !got.is_nan() {
                    return Err(format!("{what}: channel {c} is {got:?}, not NaN"));
                }
            } else if got.to_bits() != want.to_bits() {
                return Err(format!("{what}: channel {c} moved"));
            }
        }
        Ok(())
    };

    // B, the gate on x: a NaN there is a NaN u, which the ring keeps.
    let mut bad_b = row.clone();
    bad_b[100] = f32::NAN;
    let (y, ring_after) = step_from(h, conv, &w_b, &state, &bad_b, &pos)?;
    common::expect_fault(h, "a NaN B", FaultSite::LinearConv)?;
    only(&y, 100, "a NaN B")?;
    for (i, (&got, &want)) in ring_after.iter().zip(&clean_ring).enumerate() {
        if i == slot * HIDDEN + 100 {
            if !got.is_nan() {
                return Err(format!("the ring kept {got:?} for a NaN u, not NaN"));
            }
        } else if got.to_bits() != want.to_bits() {
            return Err(format!("a NaN B moved ring word {i}"));
        }
    }

    // C, the epilogue gate.
    let mut bad_c = row.clone();
    bad_c[HIDDEN + 9] = f32::NAN;
    let (y, _) = step_from(h, conv, &w_b, &state, &bad_c, &pos)?;
    common::expect_fault(h, "a NaN C", FaultSite::LinearConv)?;
    only(&y, 9, "a NaN C")?;

    // A weight that is not finite.
    let mut bad_w = w;
    bad_w[7 * taps + 3] = f32::INFINITY;
    let bad_w_b = device.upload(stream, &bad_w).map_err(why)?;
    let (y, _) = step_from(h, conv, &bad_w_b, &state, &row, &pos)?;
    common::expect_fault(h, "an infinite weight", FaultSite::LinearConv)?;
    only(&y, 7, "an infinite weight")?;

    // The previous position's u, read from the ring.
    let mut bad_ring = state.clone();
    bad_ring[(9 % conv.ring_rows) * HIDDEN + 50] = f32::INFINITY;
    let (y, _) = step_from(h, conv, &w_b, &bad_ring, &row, &pos)?;
    common::expect_fault(h, "an infinite ring word", FaultSite::LinearConv)?;
    only(&y, 50, "an infinite ring word")?;

    // An in-chunk predecessor: row 3's x at channel 50 reaches rows 3..8.
    let chunk = rows_of(8, 0x5CC3);
    let clean_rows = rows_from(h, conv, &w_b, &state, &chunk, 10)?;
    common::expect_clean(h, "the clean chunk")?;
    let mut bad_chunk = chunk;
    bad_chunk[3 * 3 * HIDDEN + 2 * HIDDEN + 50] = f32::NAN;
    let rows = rows_from(h, conv, &w_b, &state, &bad_chunk, 10)?;
    common::expect_fault(h, "a NaN x in a chunk", FaultSite::LinearConv)?;
    for r in 0..8 {
        for c in 0..HIDDEN {
            let at = r * HIDDEN + c;
            let (got, want) = (rows[at], clean_rows[at]);
            if r >= 3 && c == 50 {
                if !got.is_nan() {
                    return Err(format!("row {r} channel {c} is {got:?}, not NaN"));
                }
            } else if got.to_bits() != want.to_bits() {
                return Err(format!("a NaN x in row 3 moved row {r} channel {c}"));
            }
        }
    }
    Ok(Verdict::Noted(
        "a NaN B or C, an infinite weight, an infinite ring word and a NaN in-chunk predecessor each raise the linear-conv site and are NaN at their element alone, and the ring keeps the NaN u"
            .to_string(),
    ))
}

/// What a bounds case launches.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Step,
    Rows,
}

const BOUNDS_HIDDEN: usize = 128;
const BOUNDS_TAPS: usize = 7;
const BOUNDS_RING: usize = 8;
const BOUNDS_SLOTS: usize = 4;
const BOUNDS_ROWS: usize = 5;
const BOUNDS_POS: [u32; SHORTCONV_MAX_SLOTS + 1] = [10; SHORTCONV_MAX_SLOTS + 1];

/// The launch of one bounds case, every size a knob.
#[derive(Clone, Copy)]
struct Knobs {
    mode: Mode,
    slots: usize,
    rows: usize,
    pos0: u32,
    hidden: usize,
    taps: usize,
    ring_rows: usize,
    bcx_len: usize,
    bcx_stride: usize,
    w_len: usize,
    ring_len: usize,
    y_len: usize,
    y_stride: usize,
}

impl Knobs {
    const fn step() -> Self {
        Self {
            mode: Mode::Step,
            slots: BOUNDS_SLOTS,
            rows: 0,
            pos0: 0,
            hidden: BOUNDS_HIDDEN,
            taps: BOUNDS_TAPS,
            ring_rows: BOUNDS_RING,
            bcx_len: BOUNDS_SLOTS * 3 * BOUNDS_HIDDEN,
            bcx_stride: 3 * BOUNDS_HIDDEN,
            w_len: BOUNDS_HIDDEN * BOUNDS_TAPS,
            ring_len: BOUNDS_SLOTS * BOUNDS_RING * BOUNDS_HIDDEN,
            y_len: BOUNDS_SLOTS * BOUNDS_HIDDEN,
            y_stride: BOUNDS_HIDDEN,
        }
    }

    const fn rows() -> Self {
        Self {
            mode: Mode::Rows,
            slots: 0,
            rows: BOUNDS_ROWS,
            pos0: 10,
            hidden: BOUNDS_HIDDEN,
            taps: BOUNDS_TAPS,
            ring_rows: BOUNDS_RING,
            bcx_len: BOUNDS_ROWS * 3 * BOUNDS_HIDDEN,
            bcx_stride: 3 * BOUNDS_HIDDEN,
            w_len: BOUNDS_HIDDEN * BOUNDS_TAPS,
            ring_len: BOUNDS_RING * BOUNDS_HIDDEN,
            y_len: BOUNDS_ROWS * BOUNDS_HIDDEN,
            y_stride: BOUNDS_HIDDEN,
        }
    }
}

/// The buffers of the bounds table: every window a case takes is a prefix of
/// one of them, so a launch refused by a length is refused by the launch.
struct Bufs {
    bcx: MetalBuffer<f32>,
    w: MetalBuffer<f32>,
    ring: MetalBuffer<f32>,
    y: MetalBuffer<f32>,
}

fn bounds_launch(
    h: &Harness<'_>,
    b: &mut Bufs,
    k: &Knobs,
    encoder: &mut ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    match k.mode {
        Mode::Step => h.kernels.shortconv_step(
            encoder,
            ShortConvStepParams {
                bcx: b.bcx.window(Elems(0), Elems(k.bcx_len))?,
                bcx_stride: k.bcx_stride,
                conv_w: b.w.window(Elems(0), Elems(k.w_len))?,
                ring: b.ring.window_mut(Elems(0), Elems(k.ring_len))?,
                y: b.y.window_mut(Elems(0), Elems(k.y_len))?,
                y_stride: k.y_stride,
                faults: h.device.fault_word(),
                hidden: k.hidden,
                taps: k.taps,
                ring_rows: k.ring_rows,
                layer: LAYER,
                pos: &BOUNDS_POS[..k.slots],
            },
        ),
        Mode::Rows => h.kernels.shortconv_rows(
            encoder,
            ShortConvRowsParams {
                rows: k.rows,
                bcx: b.bcx.window(Elems(0), Elems(k.bcx_len))?,
                bcx_stride: k.bcx_stride,
                conv_w: b.w.window(Elems(0), Elems(k.w_len))?,
                ring: b.ring.window_mut(Elems(0), Elems(k.ring_len))?,
                y: b.y.window_mut(Elems(0), Elems(k.y_len))?,
                y_stride: k.y_stride,
                faults: h.device.fault_word(),
                hidden: k.hidden,
                taps: k.taps,
                ring_rows: k.ring_rows,
                layer: LAYER,
                pos: k.pos0,
            },
        ),
    }
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Knobs) + 'a,
    h: &'a Harness<'_>,
    base: Knobs,
) -> BadArg<'a, Bufs> {
    BadArg::new(
        what,
        needle,
        Box::new(move |b, encoder| {
            let mut k = base;
            edit(&mut k);
            bounds_launch(h, b, &k, encoder)
        }),
    )
}

/// A shape, a position window and a window length the launches refuse by
/// name, nothing written; the well-formed step and chunk are accepted and
/// write.
pub fn shortconv_bounds(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    // One set of buffers serves both launches, so each holds the longer of
    // their windows.
    let slack = 8;
    let step = Knobs::step();
    let rows = Knobs::rows();
    let mut bufs = Bufs {
        bcx: device
            .upload(
                stream,
                &vec![0.5f32; step.bcx_len.max(rows.bcx_len) + slack],
            )
            .map_err(why)?,
        w: device
            .upload(stream, &vec![1f32; step.w_len.max(rows.w_len) + slack])
            .map_err(why)?,
        ring: device
            .upload(
                stream,
                &vec![1f32; step.ring_len.max(rows.ring_len) + slack],
            )
            .map_err(why)?,
        y: device
            .upload(stream, &vec![1f32; step.y_len.max(rows.y_len) + slack])
            .map_err(why)?,
    };
    let wide = 3 * BOUNDS_HIDDEN;
    let cases = vec![
        bad("no slots", "slots of 0", |k| k.slots = 0, h, step),
        bad(
            "more slots than a bytes binding holds",
            "slots, the step takes at most",
            |k| k.slots = SHORTCONV_MAX_SLOTS + 1,
            h,
            step,
        ),
        bad("no channels", "hidden of 0", |k| k.hidden = 0, h, step),
        bad("no taps", "taps of 0", |k| k.taps = 0, h, step),
        bad(
            "a ring of fewer rows than taps",
            "ring rows",
            |k| k.ring_rows = BOUNDS_TAPS - 1,
            h,
            step,
        )
        .naming("taps"),
        bad(
            "a stride shorter than a row of B | C | x",
            "bcx stride",
            move |k| k.bcx_stride = wide - 1,
            h,
            step,
        ),
        bad(
            "an input window one short of its last row",
            "bcx window",
            move |k| k.bcx_len = (BOUNDS_SLOTS - 1) * wide + wide - 1,
            h,
            step,
        ),
        bad(
            "an output window one short of its last row",
            "y window",
            |k| k.y_len = BOUNDS_SLOTS * BOUNDS_HIDDEN - 1,
            h,
            step,
        ),
        bad(
            "a weight window one short",
            "conv weight window",
            |k| k.w_len -= 1,
            h,
            step,
        ),
        bad(
            "a weight window one long",
            "conv weight window",
            |k| k.w_len += 1,
            h,
            step,
        ),
        bad(
            "a ring window one short",
            "ring window",
            |k| k.ring_len -= 1,
            h,
            step,
        ),
        bad(
            "a ring window one long",
            "ring window",
            |k| k.ring_len += 1,
            h,
            step,
        ),
        bad("no rows", "rows of 0", |k| k.rows = 0, h, rows),
        bad(
            "positions past the 32-bit word",
            "32-bit position word",
            |k| k.pos0 = u32::MAX,
            h,
            rows,
        ),
        bad(
            "a chunk over the rings of four slots",
            "ring window",
            |k| k.ring_len = BOUNDS_SLOTS * BOUNDS_RING * BOUNDS_HIDDEN,
            h,
            rows,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Bufs| {
        common::still(h, &b.y, |v| *v == 1.0)?;
        common::still(h, &b.ring, |v| *v == 1.0)
    };
    let count = common::refuse_launches(h, "short conv", &mut bufs, cases, unchanged)?;
    let good_step: Edit<'_, Bufs> = Box::new(|b, encoder| bounds_launch(h, b, &step, encoder));
    common::accepted_writes(h, &mut bufs, good_step, |h, b| {
        common::moved(h, &b.y, |v| *v != 1.0)
    })?;
    bufs.y = device
        .upload(stream, &vec![1f32; rows.y_len + slack])
        .map_err(why)?;
    let good_rows: Edit<'_, Bufs> = Box::new(|b, encoder| bounds_launch(h, b, &rows, encoder));
    common::accepted_writes(h, &mut bufs, good_rows, |h, b| {
        common::moved(h, &b.y, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals, each by name, nothing written"
    )))
}

/// The rows of `rows` laid `stride` apart, the words between them `pad`.
fn strided(rows: &[Vec<f32>], stride: usize, pad: f32) -> Vec<f32> {
    let mut out = vec![pad; rows.len() * stride];
    for (r, row) in rows.iter().enumerate() {
        out[r * stride..r * stride + row.len()].copy_from_slice(row);
    }
    out
}

/// The `rows` rows of `strided`, `stride` apart, `HIDDEN` values each.
fn dense(strided: &[f32], rows: usize, stride: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| strided[r * stride..r * stride + HIDDEN].iter().copied())
        .collect()
}

/// One launch of the norm or the sandwich over `rows` rows.
#[derive(Clone, Copy)]
struct Norm<'a> {
    x: &'a [f32],
    gain: &'a [f32],
    /// The output's contents before the launch: the stream, for the sandwich.
    y0: &'a [f32],
    rows: usize,
    x_stride: usize,
    y_stride: usize,
    eps: f32,
}

/// The norm (`add` none) or the sandwich residual (`add` some) of `n`; the
/// output, `y_stride` apart. The words between the rows are checked as they
/// were.
fn norm_launch(h: &Harness<'_>, n: &Norm<'_>, add: Option<f32>) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let x_b = device.upload(stream, n.x).map_err(why)?;
    let gain_b = device.upload(stream, n.gain).map_err(why)?;
    let mut y_b = device.upload(stream, n.y0).map_err(why)?;
    h.clear_fault()?;
    h.encode(|encoder| {
        let norm = NormRowsParams {
            rows: n.rows,
            cols: HIDDEN,
            x: x_b.window(Elems(0), Elems(n.x.len()))?,
            x_stride: n.x_stride,
            gain: gain_b.window(Elems(0), Elems(n.gain.len()))?,
            y: y_b.window_mut(Elems(0), Elems(n.y0.len()))?,
            y_stride: n.y_stride,
            eps: n.eps,
            fault: device.fault_word(),
            layer: LAYER,
        };
        match add {
            Some(add) => h
                .kernels
                .norm_add_rows(encoder, NormAddRowsParams { norm, add }),
            None => h.kernels.norm_rows(encoder, norm),
        }
    })?;
    let mut y = vec![0f32; n.y0.len()];
    device
        .read(stream, y_b.whole().map_err(why)?, &mut y)
        .map_err(why)?;
    for r in 0..n.rows {
        let span = r * n.y_stride + HIDDEN..(r + 1) * n.y_stride;
        if !common::bits_eq(&y[span.clone()], &n.y0[span]) {
            return Err(format!(
                "row {r}: a word past the row's {HIDDEN} columns was written"
            ));
        }
    }
    Ok(y)
}

/// The sandwich residual `y += add · norm(x)`: the bits of the norm kernel's
/// own output, added into the stream with one rounding for the product and one
/// for the sum (the sum order of the norm is the existing norm's), and within
/// the f64 oracle's band (the independent path). Decode's one row, the add 0.5
/// of a half-scaled sandwich, strided rows, and adds that are not powers of
/// two.
pub fn sandwich(h: &Harness<'_>) -> Clause {
    let k = HIDDEN;
    let threads = h.kernels.router_threads();
    let mut rng = oracle::Rng::new(0x5D01);
    let gain: Vec<f32> = (0..k).map(|_| 1.0 + 0.5 * rng.signed()).collect();
    let pad = f32::from_bits(super::SENTINEL_BITS);
    let cases = [
        (1usize, 1.0f32, 1.0e-5f32, (k, k)),
        (4, 0.5, 1.0e-6, (k, k)),
        (3, 1.0, 1.0e-5, (k + 4, k + 8)),
        (2, 0.3, 1.0e-5, (k, k)),
        (2, -0.75, 1.0e-6, (k + 4, k + 4)),
    ];
    let mut parts = Vec::new();
    for (i, (rows, add, eps, (xs, ys))) in cases.into_iter().enumerate() {
        let case = format!("{rows} rows, add {add}, eps {eps:e}");
        let seed = 0x5D10 + 0x40 * i as u64;
        let outs: Vec<Vec<f32>> = (0..rows)
            .map(|r| oracle::activations(k, seed + r as u64, 3.7))
            .collect();
        let streams: Vec<Vec<f32>> = (0..rows)
            .map(|r| oracle::activations(k, seed + 0x20 + r as u64, 1.0))
            .collect();
        let x = strided(&outs, xs, f32::NAN);
        let stream = strided(&streams, ys, pad);
        let blank = vec![pad; rows * ys];
        let norm = Norm {
            x: &x,
            gain: &gain,
            y0: &blank,
            rows,
            x_stride: xs,
            y_stride: ys,
            eps,
        };
        let plain = norm_launch(h, &norm, None)?;
        common::expect_clean(h, &format!("{case}: the plain norm"))?;
        let after = norm_launch(
            h,
            &Norm {
                y0: &stream,
                ..norm
            },
            Some(add),
        )?;
        common::expect_clean(h, &format!("{case}: the sandwich"))?;

        let normed = dense(&plain, rows, ys);
        let before = dense(&stream, rows, ys);
        let got = dense(&after, rows, ys);
        let want: Vec<f32> = before
            .iter()
            .zip(&normed)
            .map(|(s, v)| s + add * v)
            .collect();
        if let Some(j) = got
            .iter()
            .zip(&want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
        {
            return Err(format!(
                "{case}: element {j} is {:e}, the norm kernel's value added into the stream is {:e}",
                got[j], want[j]
            ));
        }
        let bands = oracle_shortconv::sandwich(&SandwichIn {
            cols: k,
            eps,
            add,
            threads,
            simd_groups: router_geom::SIMD_GROUPS,
            out: &dense(&x, rows, xs),
            gain: &gain,
            stream: &before,
        })?;
        let worst =
            oracle_shortconv::fit_sandwich(&got, &bands).map_err(|e| format!("{case}: {e}"))?;
        parts.push(format!("{rows}x{add} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "the norm kernel's bits added into the stream, and in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A row the norm cannot take is NaN in the stream alone and raises the norm
/// site; an infinite epsilon is every row; an infinite gain moves one column.
pub fn sandwich_nan(h: &Harness<'_>) -> Clause {
    let k = HIDDEN;
    let rows = 5usize;
    let (add, eps) = (1.0f32, 1.0e-5f32);
    let mut rng = oracle::Rng::new(0x5DA0);
    let gain: Vec<f32> = (0..k).map(|_| 1.0 + 0.5 * rng.signed()).collect();
    let pad = f32::from_bits(super::SENTINEL_BITS);
    let clean: Vec<Vec<f32>> = (0..rows)
        .map(|r| oracle::activations(k, 0x5DA1 + r as u64, 3.7))
        .collect();
    let streams: Vec<Vec<f32>> = (0..rows)
        .map(|r| oracle::activations(k, 0x5DB0 + r as u64, 1.0))
        .collect();
    let stream = strided(&streams, k, pad);
    let run = |outs: &[Vec<f32>], gain: &[f32], eps: f32| -> Result<Vec<f32>, String> {
        let x = strided(outs, k, f32::NAN);
        norm_launch(
            h,
            &Norm {
                x: &x,
                gain,
                y0: &stream,
                rows,
                x_stride: k,
                y_stride: k,
                eps,
            },
            Some(add),
        )
    };
    let want = run(&clean, &gain, eps)?;
    common::expect_clean(h, "the clean chunk")?;

    let mut nan = clean.clone();
    nan[2][700] = f32::NAN;
    let y = run(&nan, &gain, eps)?;
    common::expect_fault(h, "a NaN input", FaultSite::NormQuant)?;
    for r in 0..rows {
        let (got, base) = (&y[r * k..(r + 1) * k], &want[r * k..(r + 1) * k]);
        if r == 2 {
            if !got.iter().all(|v| v.is_nan()) {
                return Err("a NaN input left a stream element that is not NaN".to_string());
            }
        } else if !common::bits_eq(got, base) {
            return Err(format!("a NaN input in row 2 moved row {r}"));
        }
    }

    let y = run(&clean, &gain, f32::INFINITY)?;
    common::expect_fault(h, "an infinite epsilon", FaultSite::NormQuant)?;
    if !y.iter().all(|v| v.is_nan()) {
        return Err("an infinite epsilon left a stream element that is not NaN".to_string());
    }

    let mut bad_gain = gain;
    bad_gain[11] = f32::INFINITY;
    let y = run(&clean, &bad_gain, eps)?;
    common::expect_fault(h, "an infinite gain", FaultSite::NormQuant)?;
    for r in 0..rows {
        for c in 0..k {
            let (got, base) = (y[r * k + c], want[r * k + c]);
            if c == 11 {
                if got.is_finite() {
                    return Err(format!("an infinite gain left row {r} column 11 finite"));
                }
            } else if got.to_bits() != base.to_bits() {
                return Err(format!("an infinite gain moved row {r} column {c}"));
            }
        }
    }
    Ok(Verdict::Noted(
        "a NaN input is NaN in its stream row alone, an infinite epsilon is every row, an infinite gain moves one column, each raising the norm site"
            .to_string(),
    ))
}

/// The short conv's clauses and the sandwich residual's, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        (
            "shortconv_rows",
            Box::new(|| shortconv_rows(h)) as ClauseRun,
        ),
        ("shortconv_start", Box::new(|| shortconv_start(h))),
        ("shortconv_state", Box::new(|| shortconv_state(h))),
        ("shortconv_nan", Box::new(|| shortconv_nan(h))),
        ("shortconv_bounds", Box::new(|| shortconv_bounds(h))),
        ("sandwich", Box::new(|| sandwich(h))),
        ("sandwich_nan", Box::new(|| sandwich_nan(h))),
    ]
}
