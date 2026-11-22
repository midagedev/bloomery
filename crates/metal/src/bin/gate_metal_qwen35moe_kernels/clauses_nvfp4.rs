//! The NVFP4 mat-vec's clauses (`kernels/nvfp4.metal`): both layouts the
//! routed experts' bytes come in — mainline's 36-byte block of 64 values (the
//! GGUF file's) and bloomery's 9-byte row group of 16 (the MLX folder's, as
//! stored) — against an f64 dot of an independent decode, the decode itself
//! held to the `gguf` crate's two decoders, a scale byte that is no UE4M3, and
//! the launch's refusals.
//!
//! The decode: a code `c` is `KVALUES[c] · ue4m3(scale) / 2`, the doubled E2M1
//! table and the halved scale both exact in f32; the dot's products and sums
//! are the only rounding, then the expert's scale one multiply.

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Elems};
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
        NvfpLayout::Rows => {
            for _ in 0..rows * (k / 16) {
                bytes.push(scale_byte(&mut rng));
                for _ in 0..8 {
                    bytes.push(rng.next() as u8);
                }
            }
        }
    }
    bytes
}

/// The values of one row, decoded from first principles.
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
        NvfpLayout::Rows => {
            for (g, grp) in row.chunks(9).enumerate() {
                let d = half_scale(grp[0]);
                for i in 0..8 {
                    out[g * 16 + 2 * i] = KVALUES[usize::from(grp[1 + i] & 15)] * d;
                    out[g * 16 + 2 * i + 1] = KVALUES[usize::from(grp[1 + i] >> 4)] * d;
                }
            }
        }
    }
    out
}

fn launch(
    h: &Harness<'_>,
    layout: NvfpLayout,
    rows: usize,
    k: usize,
    bytes: &[u8],
    x: &[f32],
) -> Result<Vec<f32>, String> {
    let w = ink2::up(h, bytes)?;
    let xb = ink2::up(h, x)?;
    let mut y = ink2::up(h, &ink2::blank(rows))?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.nvfp4_matvec(
            encoder,
            Nvfp4Params {
                layout,
                rows,
                k,
                weights: w.window(Elems(0), Elems(bytes.len()))?,
                x: xb.window(Elems(0), Elems(k))?,
                y: y.window_mut(Elems(0), Elems(rows))?,
                w_scale: W_SCALE,
                faults: h.device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let out = ink2::read(h, &y)?;
    ink2::written("the mat-vec", &out)?;
    Ok(out)
}

fn row_bytes(layout: NvfpLayout, k: usize) -> usize {
    layout.bytes(1, k).expect("a whole row")
}

/// Both layouts' mat-vecs against the oracle's dot, at the Mini's width over
/// a row count that is no multiple of a threadgroup's eight, and at the
/// smallest row of each layout; the decode first held to the `gguf` crate's.
pub fn nvfp4_rows(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (layout, name) in [(NvfpLayout::Blocks, "blocks"), (NvfpLayout::Rows, "rows")] {
        for (case, (rows, k)) in [(37usize, 2304usize), (9, layout.step())]
            .into_iter()
            .enumerate()
        {
            let seed = 0x4A00 + 32 * case as u64 + if name == "rows" { 16 } else { 0 };
            let bytes = stack(layout, rows, k, seed);
            let x = ink2::draw(k, seed + 1, 1.0);
            let per = row_bytes(layout, k);
            // The decode against the crate that reads the files.
            let mut gg = vec![0f32; k];
            match layout {
                NvfpLayout::Blocks => dequant_row(GgmlType::NVFP4, &bytes[..per], &mut gg)
                    .map_err(|e| format!("gguf's NVFP4 decode: {e}"))?,
                NvfpLayout::Rows => Nvfp4Rows::new(1.0)
                    .decode(&bytes[..per], &mut gg)
                    .map_err(|e| format!("gguf's NVFP4 rows decode: {e}"))?,
            }
            let mine = decode(layout, &bytes[..per], k);
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
                let w = decode(layout, &bytes[r * per..(r + 1) * per], k);
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
        "both layouts decode as the gguf crate does and dot in the f64 oracle's band, worst error over its band: {}",
        parts.join(", ")
    )))
}

/// A scale byte that is no UE4M3 (bit 7 set, or E4M3's NaN 0x7f) makes its row
/// NaN and raises the quant-column site, in both layouts; every other row is
/// the clean launch's bits.
pub fn nvfp4_nan(h: &Harness<'_>) -> Clause {
    let (rows, k) = (11usize, 2304usize);
    for (layout, name) in [(NvfpLayout::Blocks, "blocks"), (NvfpLayout::Rows, "rows")] {
        let bytes = stack(layout, rows, k, 0x4A80);
        let x = ink2::draw(k, 0x4A81, 1.0);
        let clean = launch(h, layout, rows, k, &bytes, &x)?;
        common::expect_clean(h, "the clean mat-vec")?;
        let per = row_bytes(layout, k);
        let mut bad = bytes.clone();
        // Row 3's last scale byte, row 8's middle one.
        let (scale_of_last, scale_mid) = match layout {
            NvfpLayout::Blocks => (3 * per + per - 36 + 3, 8 * per + 36 * 17 + 2),
            NvfpLayout::Rows => (3 * per + per - 9, 8 * per + 9 * 70),
        };
        bad[scale_of_last] = 0x80;
        bad[scale_mid] = 0x7f;
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
        "a scale byte that is no UE4M3 is NaN in its row alone and raises the quant-column site, in both layouts"
            .to_string(),
    ))
}

/// The shapes and windows the launch refuses by name.
pub fn nvfp4_bounds(h: &Harness<'_>) -> Clause {
    let (rows, k) = (4usize, 128usize);
    let blocks = stack(NvfpLayout::Blocks, rows, k, 0x4AC0);
    let groups = stack(NvfpLayout::Rows, rows, k, 0x4AC1);
    let x = ink2::draw(256, 0x4AC2, 1.0);
    let mut padded = vec![0u8; 2];
    padded.extend_from_slice(&blocks);
    padded.extend_from_slice(&[0u8; 36]);
    let run = |label: &str,
               layout: NvfpLayout,
               k: usize,
               at: usize,
               len: usize,
               which: usize,
               w_scale: f32,
               needle: &str|
     -> Result<(), String> {
        let src = [&blocks, &groups, &padded][which];
        let w = ink2::up(h, src)?;
        let xb = ink2::up(h, &x)?;
        let mut y = ink2::up(h, &ink2::blank(rows))?;
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
                        x: xb.window(Elems(0), Elems(k))?,
                        y: y.window_mut(Elems(0), Elems(rows))?,
                        w_scale,
                        faults: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
            needle,
        )
    };
    let bl = blocks.len();
    let gl = groups.len();
    run(
        "a partial block",
        NvfpLayout::Blocks,
        100,
        0,
        bl,
        0,
        W_SCALE,
        "whole groups",
    )?;
    run(
        "a partial group",
        NvfpLayout::Rows,
        100,
        0,
        gl,
        1,
        W_SCALE,
        "whole groups",
    )?;
    run(
        "a short stack",
        NvfpLayout::Blocks,
        k,
        0,
        bl - 36,
        0,
        W_SCALE,
        "weights window",
    )?;
    run(
        "a long stack",
        NvfpLayout::Blocks,
        k,
        0,
        bl + 36,
        2,
        W_SCALE,
        "weights window",
    )?;
    run(
        "blocks off a word",
        NvfpLayout::Blocks,
        k,
        2,
        bl,
        2,
        W_SCALE,
        "32-bit words",
    )?;
    run(
        "a scale that is not a number",
        NvfpLayout::Blocks,
        k,
        0,
        bl,
        0,
        f32::NAN,
        "scale",
    )?;
    Ok(Verdict::Noted(
        "six refusals by name, each before an encode".to_string(),
    ))
}

/// The mat-vec's clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    vec![
        ("nvfp4_rows", Box::new(|| nvfp4_rows(h)) as ClauseRun),
        ("nvfp4_nan", Box::new(|| nvfp4_nan(h))),
        ("nvfp4_bounds", Box::new(|| nvfp4_bounds(h))),
    ]
}
