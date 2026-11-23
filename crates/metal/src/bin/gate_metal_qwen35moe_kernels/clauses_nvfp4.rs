//! The NVFP4 mat-vec's clauses (`kernels/nvfp4.metal`): every layout the
//! routed experts' bytes come in — mainline's 36-byte block of 64 values (the
//! GGUF file's), bloomery's two planes an expert (the MLX folder's), the
//! codas' Q8_0 and MXFP8 — against an f64
//! dot of an independent decode, the decode itself held to the `gguf` crate's
//! decoders, a scale byte that is no UE4M3, the launch's refusals and the
//! planes at the Mini's two launch shapes.
//!
//! The decode: a code `c` is `KVALUES[c] · ue4m3(scale) / 2`, the doubled E2M1
//! table and the halved scale both exact in f32; the dot's products and sums
//! are the only rounding, then the expert's scale one multiply.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
use bloomery_metal::inkling2_geom::TOP_K;
use bloomery_metal::kernels::{Nvfp4Params, NvfpLayout};
use gguf::quant::{GgmlType, Nvfp4Rows, dequant_row};

use super::common::{self, LAYER};
use super::ink2::{self, U};
use super::oracle::Rng;
use super::{ClauseRun, Harness};

/// The doubled E2M1 values a code byte's nibble names.
const KVALUES: [f64; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];
/// The expert's global scale the launches carry.
const W_SCALE: f32 = 0.0173;

/// `ue4m3(byte) / 2` of a valid scale byte.
fn half_scale(b: u8) -> f64 {
    let (e, m) = (i32::from(b >> 3), f64::from(b & 7));
    let raw = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    raw * 0.5
}

/// The value of finite f16 bits `b`.
fn f16_value(b: u16) -> f64 {
    let (e, m) = (i32::from((b >> 10) & 31), f64::from(b & 0x3ff));
    let v = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    };
    if b & 0x8000 != 0 { -v } else { v }
}

/// A scale byte in a range that keeps the dot's terms near 1.
fn scale_byte(rng: &mut Rng) -> u8 {
    0x28 + (rng.next() % 0x28) as u8
}

/// `rows` rows of `k` values in `layout`, seeded.
fn stack(layout: NvfpLayout, rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut bytes = Vec::new();
    match layout {
        NvfpLayout::Blocks => {
            for _ in 0..rows * (k / 64) {
                for _ in 0..4 {
                    bytes.push(scale_byte(&mut rng));
                }
                for _ in 0..32 {
                    bytes.push(rng.next() as u8);
                }
            }
        }
        // One expert: every row's codes, then every row's scale bytes.
        NvfpLayout::Planes => {
            for _ in 0..rows * k / 2 {
                bytes.push(rng.next() as u8);
            }
            for _ in 0..rows * (k / 16) {
                bytes.push(scale_byte(&mut rng));
            }
        }
        NvfpLayout::Q8_0 => {
            for _ in 0..rows * (k / 32) {
                // An f16 scale near 1/64, its low mantissa bits seeded.
                let d = 0x2400u16 | (rng.next() % 0x400) as u16;
                bytes.extend(d.to_le_bytes());
                for _ in 0..32 {
                    bytes.push(rng.next() as u8);
                }
            }
        }
        NvfpLayout::Mxfp8Rows => {
            for _ in 0..rows * (k / 32) {
                bytes.push(120 + (rng.next() % 6) as u8);
                for _ in 0..32 {
                    // Every E4M3 code but the two NaNs.
                    let c = rng.next() as u8;
                    bytes.push(if c & 0x7f == 0x7f { c ^ 1 } else { c });
                }
            }
        }
    }
    bytes
}

/// Row `r` of one expert's stack of `rows` rows as [`decode`] reads it: the
/// row's own span, or, of the planes, its groups gathered (each group's scale
/// byte, then its 8 code bytes: the MLX folder's NVFP4 rows type).
fn row_of(layout: NvfpLayout, bytes: &[u8], rows: usize, r: usize, k: usize) -> Vec<u8> {
    if layout != NvfpLayout::Planes {
        let per = row_bytes(layout, k);
        return bytes[r * per..(r + 1) * per].to_vec();
    }
    let (codes, scales) = (&bytes[r * k / 2..], &bytes[rows * k / 2 + r * (k / 16)..]);
    (0..k / 16)
        .flat_map(|g| std::iter::once(scales[g]).chain(codes[8 * g..8 * g + 8].iter().copied()))
        .collect()
}

/// The values of row `r` of one expert's stack of `rows` rows.
fn values(layout: NvfpLayout, bytes: &[u8], rows: usize, r: usize, k: usize) -> Vec<f64> {
    decode(layout, &row_of(layout, bytes, rows, r, k), k)
}

/// The values of one row as [`row_of`] gives it, decoded from first
/// principles.
fn decode(layout: NvfpLayout, row: &[u8], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k];
    match layout {
        NvfpLayout::Blocks => {
            for (b, blk) in row.chunks(36).enumerate() {
                for s in 0..4 {
                    let d = half_scale(blk[s]);
                    for j in 0..8 {
                        let byte = blk[4 + 8 * s + j];
                        let base = b * 64 + s * 16;
                        out[base + j] = KVALUES[usize::from(byte & 15)] * d;
                        out[base + j + 8] = KVALUES[usize::from(byte >> 4)] * d;
                    }
                }
            }
        }
        NvfpLayout::Planes => {
            for (g, grp) in row.chunks(9).enumerate() {
                let d = half_scale(grp[0]);
                for i in 0..8 {
                    out[g * 16 + 2 * i] = KVALUES[usize::from(grp[1 + i] & 15)] * d;
                    out[g * 16 + 2 * i + 1] = KVALUES[usize::from(grp[1 + i] >> 4)] * d;
                }
            }
        }
        NvfpLayout::Q8_0 => {
            for (b, blk) in row.chunks(34).enumerate() {
                let d = f16_value(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    out[b * 32 + i] = f64::from(blk[2 + i] as i8) * d;
                }
            }
        }
        NvfpLayout::Mxfp8Rows => {
            for (g, grp) in row.chunks(33).enumerate() {
                let s = 2f64.powi(i32::from(grp[0]) - 127);
                for i in 0..32 {
                    let c = grp[1 + i];
                    let (e, m) = (i32::from((c >> 3) & 15), f64::from(c & 7));
                    let v = if e == 0 {
                        m * 2f64.powi(-9)
                    } else {
                        (1.0 + m / 8.0) * 2f64.powi(e - 7)
                    };
                    out[g * 32 + i] = if c & 0x80 != 0 { -v } else { v } * s;
                }
            }
        }
    }
    out
}

/// One launch over a route of `ids.len()` slots of the stack `bytes` (one
/// expert a scale of `scales`), each slot's rows at its plane's place; a
/// slot reads its own input when `x` holds one a slot. The output read back.
#[allow(clippy::too_many_arguments)]
fn launch_route(
    h: &Harness<'_>,
    layout: NvfpLayout,
    rows: usize,
    k: usize,
    bytes: &[u8],
    ids: &[u32],
    scales: &[f32],
    route_weights: Option<&[f32]>,
    x: &[f32],
    plane: usize,
) -> Result<Vec<f32>, String> {
    let slots = ids.len();
    let w = ink2::up(h, bytes)?;
    let idb = ink2::up(h, ids)?;
    let sb = ink2::up(h, scales)?;
    let rw = route_weights.map(|r| ink2::up(h, r)).transpose()?;
    let xb = ink2::up(h, x)?;
    let mut y = ink2::up(h, &ink2::blank(slots * rows))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.nvfp4_matvec(
            encoder,
            Nvfp4Params {
                layout,
                rows,
                k,
                weights: w.window(Elems(0), Elems(bytes.len()))?,
                ids: idb.window(Elems(0), Elems(slots))?,
                scales: sb.window(Elems(0), Elems(scales.len()))?,
                route_weights: rw
                    .as_ref()
                    .map(|b| b.window(Elems(0), Elems(slots)))
                    .transpose()?,
                x: xb.window(Elems(0), Elems(x.len()))?,
                x_per_slot: x.len() != k,
                y: y.window_mut(Elems(0), Elems(slots * rows))?,
                plane,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the mat-vec", &out)?;
    Ok(out)
}

/// One expert's rows at the scale `W_SCALE`: a one-slot route.
fn launch(
    h: &Harness<'_>,
    layout: NvfpLayout,
    rows: usize,
    k: usize,
    bytes: &[u8],
    x: &[f32],
) -> Result<Vec<f32>, String> {
    launch_route(h, layout, rows, k, bytes, &[0], &[W_SCALE], None, x, rows)
}

fn row_bytes(layout: NvfpLayout, k: usize) -> usize {
    layout.bytes(1, k).expect("a whole row")
}

/// Every layout's mat-vec against the oracle's dot, at the Mini's width over
/// a row count that is no multiple of a threadgroup's eight (nor of a planes
/// SIMD group's sixteen), and at the smallest row of each layout; the decode
/// first held to the `gguf` crate's.
pub fn nvfp4_rows(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (layout, name) in [
        (NvfpLayout::Blocks, "blocks"),
        (NvfpLayout::Planes, "planes"),
        (NvfpLayout::Q8_0, "q8_0"),
        (NvfpLayout::Mxfp8Rows, "mxfp8 rows"),
    ] {
        for (case, (rows, k)) in [(37usize, 2304usize), (9, layout.step())]
            .into_iter()
            .enumerate()
        {
            let seed = 0x4A00 + 32 * case as u64 + 16 * layout as u64;
            let bytes = stack(layout, rows, k, seed);
            let x = ink2::draw(k, seed + 1, 1.0);
            let row0 = row_of(layout, &bytes, rows, 0, k);
            // The decode against the crate that reads the files.
            let mut gg = vec![0f32; k];
            match layout {
                NvfpLayout::Blocks => dequant_row(GgmlType::NVFP4, &row0, &mut gg)
                    .map_err(|e| format!("gguf's NVFP4 decode: {e}"))?,
                NvfpLayout::Planes => Nvfp4Rows::new(1.0)
                    .decode(&row0, &mut gg)
                    .map_err(|e| format!("gguf's NVFP4 rows decode: {e}"))?,
                NvfpLayout::Q8_0 => dequant_row(GgmlType::Q8_0, &row0, &mut gg)
                    .map_err(|e| format!("gguf's Q8_0 decode: {e}"))?,
                NvfpLayout::Mxfp8Rows => dequant_row(GgmlType::MXFP8, &row0, &mut gg)
                    .map_err(|e| format!("gguf's MXFP8 decode: {e}"))?,
            }
            let mine = values(layout, &bytes, rows, 0, k);
            if let Some(i) = gg.iter().zip(&mine).position(|(a, b)| f64::from(*a) != *b) {
                return Err(format!(
                    "{name}: value {i} decodes to {} in gguf and {} in the oracle",
                    gg[i], mine[i]
                ));
            }
            let got = launch(h, layout, rows, k, &bytes, &x)?;
            common::expect_clean(h, "the mat-vec")?;
            let (mut want, mut band) = (Vec::new(), Vec::new());
            for r in 0..rows {
                let w = values(layout, &bytes, rows, r, k);
                let dot: f64 = w.iter().zip(&x).map(|(a, b)| a * f64::from(*b)).sum();
                let mag: f64 = w
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (a * f64::from(*b)).abs())
                    .sum();
                want.push(dot * f64::from(W_SCALE));
                band.push(
                    mag * f64::from(W_SCALE) * (ink2::gamma_walk(k / 32 + 8) + 2.0 * U) + 1e-30,
                );
            }
            let worst = ink2::fit(&format!("{name} {rows}x{k}"), &got, &want, &band)?;
            parts.push(format!("{name} {rows}x{k} {worst:.3}"));
        }
    }
    Ok(Verdict::Noted(format!(
        "the four layouts decode as the gguf crate does and dot in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A byte with no defined value — an NVFP4 scale that is no UE4M3 (bit 7 set,
/// or E4M3's NaN 0x7f), a Q8_0 scale that is no finite f16, an MXFP8 scale of
/// 0xff or E4M3 NaN code — makes its row NaN and raises the quant-column site,
/// in every layout; every other row is the clean launch's bits.
pub fn nvfp4_nan(h: &Harness<'_>) -> Clause {
    let (rows, k) = (11usize, 2304usize);
    for (layout, name) in [
        (NvfpLayout::Blocks, "blocks"),
        (NvfpLayout::Planes, "planes"),
        (NvfpLayout::Q8_0, "q8_0"),
        (NvfpLayout::Mxfp8Rows, "mxfp8 rows"),
    ] {
        let bytes = stack(layout, rows, k, 0x4A80);
        let x = ink2::draw(k, 0x4A81, 1.0);
        let clean = launch(h, layout, rows, k, &bytes, &x)?;
        common::expect_clean(h, "the clean mat-vec")?;
        let per = row_bytes(layout, k);
        let mut bad = bytes.clone();
        // A byte of row 3's last block or group, one of row 8's middle one.
        let [(last, last_byte), (mid, mid_byte)] = match layout {
            NvfpLayout::Blocks => [
                (3 * per + per - 36 + 3, 0x80),
                (8 * per + 36 * 17 + 2, 0x7f),
            ],
            NvfpLayout::Planes => {
                let (scales, g) = (rows * k / 2, k / 16);
                [(scales + 3 * g + g - 1, 0x80), (scales + 8 * g + 70, 0x7f)]
            }
            // The scale's high byte: a NaN, then an infinity.
            NvfpLayout::Q8_0 => [
                (3 * per + per - 34 + 1, 0x7e),
                (8 * per + 34 * 17 + 1, 0x7c),
            ],
            // The scale byte 0xff, then a NaN code.
            NvfpLayout::Mxfp8Rows => [(3 * per + per - 33, 0xff), (8 * per + 33 * 17 + 5, 0x7f)],
        };
        bad[last] = last_byte;
        bad[mid] = mid_byte;
        let got = launch(h, layout, rows, k, &bad, &x)?;
        common::expect_fault(h, name, FaultSite::QuantColumn)?;
        for r in 0..rows {
            if r == 3 || r == 8 {
                if !got[r].is_nan() {
                    return Err(format!(
                        "{name}: row {r} is {:e}, not NaN beside a bad scale",
                        got[r]
                    ));
                }
            } else if got[r].to_bits() != clean[r].to_bits() {
                return Err(format!("{name}: a bad scale in rows 3 and 8 moved row {r}"));
            }
        }
    }
    Ok(Verdict::Noted(
        "a byte with no defined value is NaN in its row alone and raises the quant-column site, in the four layouts"
            .to_string(),
    ))
}

/// The shapes and windows the launch refuses by name.
pub fn nvfp4_bounds(h: &Harness<'_>) -> Clause {
    let (rows, k) = (4usize, 128usize);
    let blocks = stack(NvfpLayout::Blocks, rows, k, 0x4AC0);
    let planes = stack(NvfpLayout::Planes, rows, k, 0x4AC3);
    let x = ink2::draw(256, 0x4AC2, 1.0);
    let mut padded = vec![0u8; 2];
    padded.extend_from_slice(&blocks);
    padded.extend_from_slice(&[0u8; 36]);
    // The planes four bytes into their buffer: on a 32-bit word, off an
    // 8-byte one.
    let mut planes_off = vec![0u8; 4];
    planes_off.extend_from_slice(&planes);
    let run = |label: &str,
               layout: NvfpLayout,
               k: usize,
               (at, len): (usize, usize),
               which: usize,
               (slots, x_at): (usize, usize),
               needle: &str|
     -> Result<(), String> {
        let src = [&blocks, &padded, &planes, &planes_off][which];
        let w = ink2::up(h, src)?;
        let xb = ink2::up(h, &x)?;
        let idb = ink2::up(h, &vec![0u32; slots])?;
        let sb = ink2::up(h, &[W_SCALE])?;
        let mut y = ink2::up(h, &ink2::blank(slots * rows))?;
        ink2::refused(
            label,
            h.encode(|encoder| {
                h.kernels.nvfp4_matvec(
                    encoder,
                    Nvfp4Params {
                        layout,
                        rows,
                        k,
                        weights: w.window(Elems(at), Elems(len))?,
                        ids: idb.window(Elems(0), Elems(slots))?,
                        scales: sb.window(Elems(0), Elems(1))?,
                        route_weights: None,
                        x: xb.window(Elems(x_at), Elems(k))?,
                        x_per_slot: false,
                        y: y.window_mut(Elems(0), Elems(slots * rows))?,
                        plane: rows,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let (bl, pl) = (blocks.len(), planes.len());
    let one = (1, 0);
    run(
        "a partial block",
        NvfpLayout::Blocks,
        100,
        (0, bl),
        0,
        one,
        "whole groups",
    )?;
    run(
        "a partial planes group",
        NvfpLayout::Planes,
        100,
        (0, pl),
        2,
        one,
        "whole groups",
    )?;
    run(
        "a short stack",
        NvfpLayout::Blocks,
        k,
        (0, bl - 36),
        0,
        one,
        "weights window",
    )?;
    run(
        "a long stack",
        NvfpLayout::Blocks,
        k,
        (0, bl + 36),
        1,
        one,
        "weights window",
    )?;
    run(
        "blocks off a word",
        NvfpLayout::Blocks,
        k,
        (2, bl),
        1,
        one,
        "32-bit words",
    )?;
    run(
        "planes off an 8-byte word",
        NvfpLayout::Planes,
        k,
        (4, pl),
        3,
        one,
        "8-byte words",
    )?;
    run(
        "planes' input off a float4",
        NvfpLayout::Planes,
        k,
        (0, pl),
        2,
        (1, 2),
        "16-byte aligned",
    )?;
    run(
        "a route of more slots than picks",
        NvfpLayout::Blocks,
        k,
        (0, bl),
        0,
        (TOP_K + 1, 0),
        "slots",
    )?;
    Ok(Verdict::Noted(
        "eight refusals by name, each before an encode".to_string(),
    ))
}

/// A route's slots in one launch: each slot reads its expert's id on the
/// card, its expert's rows at the id's offset in the stack, its expert's
/// scale (times its route weight on a down) and, on a down, its own input,
/// and lands at its plane's slot block — every slot's rows the bits of a
/// one-slot launch of its expert alone. A slot whose id is past the stack is
/// NaN and raises the expert-id site; the other slots keep those bits. In the
/// five layouts.
pub fn nvfp4_route(h: &Harness<'_>) -> Clause {
    const EXPERTS: usize = 5;
    const SCALES: [f32; EXPERTS] = [0.5, 1.25, 2.0, 0.75, 3.0];
    const ROUTE: [f32; 3] = [0.3, 0.9, 0.11];
    let (rows, k) = (24usize, 256usize);
    let plane = rows / 2;
    for (layout, name) in [
        (NvfpLayout::Blocks, "blocks"),
        (NvfpLayout::Planes, "planes"),
        (NvfpLayout::Q8_0, "q8_0"),
        (NvfpLayout::Mxfp8Rows, "mxfp8 rows"),
    ] {
        let per = layout.bytes(rows, k).expect("whole groups");
        let seed = 0x4B00 + 16 * layout as u64;
        // The planes are an expert's own: each expert drawn apart.
        let bytes: Vec<u8> = if layout == NvfpLayout::Planes {
            (0..EXPERTS as u64)
                .flat_map(|e| stack(layout, rows, k, seed + 64 * e))
                .collect()
        } else {
            stack(layout, EXPERTS * rows, k, seed)
        };
        let x = ink2::draw(3 * k, 0x4B01 + 16 * layout as u64, 1.0);
        let alone = |e: u32, scale: f32, x: &[f32]| {
            let e = e as usize;
            launch_route(
                h,
                layout,
                rows,
                k,
                &bytes[e * per..(e + 1) * per],
                &[0],
                &[scale],
                None,
                x,
                rows,
            )
        };
        let same = |what: &str, s: usize, e: u32, got: f32, want: f32, r: usize| {
            if got.to_bits() == want.to_bits() {
                Ok(())
            } else {
                Err(format!(
                    "{name} {what}: slot {s} (expert {e}) row {r} is {got:e}, {want:e} alone"
                ))
            }
        };
        // The gate·up form: one shared input, two planes, no route weight.
        let ids = [3u32, 0, 4];
        let got = launch_route(
            h,
            layout,
            rows,
            k,
            &bytes,
            &ids,
            &SCALES,
            None,
            &x[..k],
            plane,
        )?;
        common::expect_clean(h, "the gate·up route")?;
        for (s, &e) in ids.iter().enumerate() {
            let want = alone(e, SCALES[e as usize], &x[..k])?;
            for (r, w) in want.iter().enumerate() {
                let at = ((r / plane) * ids.len() + s) * plane + r % plane;
                same("gate·up", s, e, got[at], *w, r)?;
            }
        }
        // The down form: a slot's own input, its route weight times its
        // expert's scale, one plane; then a slot whose id is past the stack.
        for (ids, bad) in [([3u32, 0, 4], None), ([1, EXPERTS as u32 + 2, 2], Some(1))] {
            let got = launch_route(
                h,
                layout,
                rows,
                k,
                &bytes,
                &ids,
                &SCALES,
                Some(&ROUTE),
                &x,
                rows,
            )?;
            match bad {
                Some(_) => common::expect_fault(h, name, FaultSite::ExpertId)?,
                None => common::expect_clean(h, "the down route")?,
            }
            for (s, &e) in ids.iter().enumerate() {
                if Some(s) == bad {
                    if let Some(r) = (0..rows).find(|&r| !got[s * rows + r].is_nan()) {
                        return Err(format!(
                            "{name}: slot {s}'s id {e} is past the stack's {EXPERTS} experts, and its row {r} is {:e}",
                            got[s * rows + r]
                        ));
                    }
                    continue;
                }
                let scale = SCALES[e as usize] * ROUTE[s];
                let want = alone(e, scale, &x[s * k..(s + 1) * k])?;
                for (r, w) in want.iter().enumerate() {
                    same("down", s, e, got[s * rows + r], *w, r)?;
                }
            }
        }
    }
    Ok(Verdict::Noted(
        "every slot of a route is its expert's one-slot launch to the bit, at its plane's place, with its own input and route weight on a down; an id past the stack is NaN in its slot alone and raises the expert-id site, in the four layouts"
            .to_string(),
    ))
}

/// The planes at the Mini's two launches, a route of three slots over a
/// stack of three experts: the gate·up (2304 rows over the 2304-wide row, one
/// input, output planes of 1152) and the down (2304 rows over 1152, each
/// slot's own input and route weight). Every slot's rows against the f64
/// oracle's dot of its expert's rows: each expert's two planes, each row's
/// codes and scale bytes, each slot's input and each lane's groups at the
/// widths the decode launches are the ones its rows read.
pub fn nvfp4_planes(h: &Harness<'_>) -> Clause {
    const EXPERTS: usize = 3;
    const SCALES: [f32; EXPERTS] = [0.5, 1.25, 2.0];
    const ROUTE: [f32; 3] = [0.3, 0.9, 0.11];
    let ids = [2u32, 0, 1];
    let layout = NvfpLayout::Planes;
    let mut parts = Vec::new();
    for (name, rows, k, plane, down) in [
        ("gate·up", 2304usize, 2304usize, 1152usize, false),
        ("down", 2304, 1152, 2304, true),
    ] {
        let per = layout.bytes(rows, k).expect("whole groups");
        let seed = 0x4C00 + k as u64;
        let bytes: Vec<u8> = (0..EXPERTS as u64)
            .flat_map(|e| stack(layout, rows, k, seed + 64 * e))
            .collect();
        let x = ink2::draw(if down { ids.len() * k } else { k }, seed + 1, 1.0);
        let got = launch_route(
            h,
            layout,
            rows,
            k,
            &bytes,
            &ids,
            &SCALES,
            down.then_some(&ROUTE[..]),
            &x,
            plane,
        )?;
        common::expect_clean(h, "the planes route")?;
        let (mut mine, mut want, mut band) = (Vec::new(), Vec::new(), Vec::new());
        for (s, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let expert = &bytes[e * per..(e + 1) * per];
            let xs = if down { &x[s * k..(s + 1) * k] } else { &x[..] };
            let scale = if down {
                f64::from(SCALES[e] * ROUTE[s])
            } else {
                f64::from(SCALES[e])
            };
            for r in 0..rows {
                let w = values(layout, expert, rows, r, k);
                let dot: f64 = w.iter().zip(xs).map(|(a, b)| a * f64::from(*b)).sum();
                let mag: f64 = w
                    .iter()
                    .zip(xs)
                    .map(|(a, b)| (a * f64::from(*b)).abs())
                    .sum();
                mine.push(got[((r / plane) * ids.len() + s) * plane + r % plane]);
                want.push(dot * scale);
                band.push(mag * scale * (ink2::gamma_walk(k / 32 + 8) + 2.0 * U) + 1e-30);
            }
        }
        let worst = ink2::fit(&format!("planes {name} {rows}x{k}"), &mine, &want, &band)?;
        parts.push(format!("{name} {rows}x{k} {worst:.3}"));
    }
    Ok(Verdict::Noted(format!(
        "the planes at the Mini's launches, three slots over three experts, dot in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// The mat-vec's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("nvfp4_rows", Box::new(|| nvfp4_rows(h)) as ClauseRun),
        ("nvfp4_nan", Box::new(|| nvfp4_nan(h))),
        ("nvfp4_bounds", Box::new(|| nvfp4_bounds(h))),
        ("nvfp4_route", Box::new(|| nvfp4_route(h))),
        ("nvfp4_planes", Box::new(|| nvfp4_planes(h))),
    ]
}
