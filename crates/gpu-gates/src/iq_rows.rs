//! What the i-quant row gates (`gate_iq`, `gate_iq_sel`) share: the rows
//! they read and the host side they hold the card to.
//!
//! Rows. [`synth_rows`] is `dequant_ref --synthetic`'s dump for `just
//! gate-1-1` (`$BLOOMERY_DATA/ref-synth/<type>.blocks`: 64 rows ggml
//! quantized, then 16 rows of random codes, [`SYNTH_K`] values each) for the
//! formats the harness writes. IQ2_XXS, IQ2_S, IQ1_S and IQ1_M have no dump: a
//! new synthetic type moves the harness's md5, which the dequant families pin
//! (`crates/refset/src/arch/dequant.rs`), so their 80 rows are
//! [`seeded_blocks`] — random code bytes under a finite `d` — which reach
//! every grid index, sign index and scale; IQ2_XXS's decode is held to ggml's
//! by `gguf`'s `iq2_xxs_matches_a_ggml_made_block`, the other three's by
//! [`ref_dequant`].
//!
//! Reference. [`dot_host`] is the module rule of `bloomery_gpu::iq` on the
//! host (`lane_partial_host` per lane, then the butterfly), the card's bits.
//! [`dot_ref`] is the exact dot of [`ref_dequant`]'s row with the f32
//! activations, in f64, and the bound on the card's error (`gate_iq.rs`'s
//! derivation): the activation term `Σ_B (d_B/2)(1 + 256u)·Σ_{i∈B}|w_i|` over
//! 128-value blocks, `γ(2·n_it + 6)·Σ_k |t_k|` over the lane terms
//! ([`term_mag`]), and for Q2_K, whose dequant rounds once, `u·Σ|w_i x_i|`.

use crate::rounding::{U, butterfly, gamma};
use crate::{GateError, data_dir};
use bloomery_gpu::iq::{IqFormat, SUB_VALUES, lane_partial_host, sub_scales_host, sub_sums_host};
use gguf::iq_tables::{IQ1S_GRID, IQ2S_GRID, KMASK_IQ2XS};
use gguf::quant::{QuantError, dequant_row, half_to_f32};

/// Values per synthetic row.
pub const SYNTH_K: usize = 4096;
/// Rows of the synthetic set: 64 ggml-quantized and 16 random.
pub const SYNTH_ROWS: usize = 80;
/// The seed of IQ2_XXS's synthetic rows.
const SEED_IQ2_XXS: u64 = 0x1a2b_3c4d_5e6f_7788 ^ 16;
/// The seeds of IQ2_S's, IQ1_S's and IQ1_M's.
const SEED_IQ2_S: u64 = 0x1a2b_3c4d_5e6f_7788 ^ 0x25;
const SEED_IQ1_S: u64 = 0x1a2b_3c4d_5e6f_7788 ^ 0x15;
const SEED_IQ1_M: u64 = 0x1a2b_3c4d_5e6f_7788 ^ 0x1d;

/// One activation column: value-order int8 codes and 128-value block scales.
pub type Col = (Vec<i8>, Vec<f32>);

/// The f16 scale byte offsets of a block of `fmt`; IQ1_M has none (its f16 is
/// spread over the top nibbles of its scale halfwords, [`seeded_blocks`]).
fn scale_offsets(fmt: IqFormat) -> &'static [usize] {
    match fmt {
        IqFormat::Q2K => &[80, 82],
        IqFormat::Iq1M => &[],
        IqFormat::Iq2Xxs
        | IqFormat::Iq2Xs
        | IqFormat::Iq2S
        | IqFormat::Iq1S
        | IqFormat::Iq3Xxs
        | IqFormat::Iq3S
        | IqFormat::Iq4Xs => &[0],
    }
}

/// Write the f16 `bits` as the scale `d` of `blk`, a block of `fmt`: its first
/// two bytes, and for IQ1_M, which has no such field, the top nibbles of its
/// four scale halfwords (`bits` low nibble first).
pub fn set_d(fmt: IqFormat, blk: &mut [u8], bits: u16) {
    if fmt == IqFormat::Iq1M {
        for j in 0..4 {
            let top = ((bits >> (4 * j)) & 0xf) as u8;
            blk[49 + 2 * j] = (blk[49 + 2 * j] & 0x0f) | (top << 4);
        }
    } else {
        blk[..2].copy_from_slice(&bits.to_le_bytes());
    }
}

/// Bolt's shared expert input and intermediate widths.
pub const BOLT_SHARED: (usize, usize) = (4096, 5376);

/// `n` seeded blocks of `fmt` ([`crate::GateRng`] from `seed | 1`): every byte random,
/// then each f16 scale a finite half of either sign in ±[2^-10, 2^-2)
/// (exponent field 5..=12). An IQ1_M block's f16 is written into the top
/// nibbles of its four scale halfwords, the rest of the halfwords staying random.
#[must_use]
pub fn seeded_blocks(fmt: IqFormat, n: usize, seed: u64) -> Vec<u8> {
    let mut rng = crate::GateRng(seed | 1);
    let mut next = || rng.next();
    let bb = fmt.block_bytes();
    let mut v = vec![0u8; n * bb];
    for c in v.chunks_mut(8) {
        let r = next().to_le_bytes();
        c.copy_from_slice(&r[..c.len()]);
    }
    for blk in v.chunks_exact_mut(bb) {
        for &at in scale_offsets(fmt) {
            let r = next();
            let e = 5 + (r % 8) as u16;
            let h = ((r >> 8) as u16 & 0x8000) | (e << 10) | ((r >> 24) as u16 & 0x3ff);
            blk[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
        if fmt == IqFormat::Iq1M {
            let r = next();
            let e = 5 + (r % 8) as u16;
            let h = ((r >> 8) as u16 & 0x8000) | (e << 10) | ((r >> 24) as u16 & 0x3ff);
            set_d(fmt, blk, h);
        }
    }
    v
}

/// The format's [`SYNTH_ROWS`] synthetic rows of [`SYNTH_K`] values, row
/// after row (module doc), or the error naming the file.
pub fn synth_rows(fmt: IqFormat) -> Result<Vec<u8>, GateError> {
    let bb = fmt.block_bytes();
    let row_bytes = SYNTH_K / 256 * bb;
    let seed = match fmt {
        IqFormat::Iq2Xxs => Some(SEED_IQ2_XXS),
        IqFormat::Iq2S => Some(SEED_IQ2_S),
        IqFormat::Iq1S => Some(SEED_IQ1_S),
        IqFormat::Iq1M => Some(SEED_IQ1_M),
        _ => None,
    };
    if let Some(seed) = seed {
        return Ok(seeded_blocks(fmt, SYNTH_ROWS * SYNTH_K / 256, seed));
    }
    let name = fmt.ggml().name().ok_or("an IqFormat names its type")?;
    let path = data_dir().join("ref-synth").join(format!("{name}.blocks"));
    let all = std::fs::read(&path)
        .map_err(|e| format!("{}: {e} (run `just gate-1-1` first)", path.display()))?;
    if all.is_empty() || !all.len().is_multiple_of(row_bytes) {
        return Err(format!(
            "{}: {} bytes is not whole rows of {row_bytes}",
            path.display(),
            all.len()
        )
        .into());
    }
    Ok(all)
}

/// The row of `bytes` dequantized by ggml's rule: `gguf::dequant_row` for the
/// formats it decodes, and for IQ2_S, IQ1_S and IQ1_M a port of ik's
/// `dequantize_row_iq2_s`, `dequantize_row_iq1_s` and `dequantize_row_iq1_m`
/// (`ggml-quants.c:3747`, `:3854`, `:3879`) that is the gate's own, written from
/// the C and not from the card rule: the f32 products are the C's, left to
/// right (`dl·(grid + delta)` and `dl·grid·(±1)`). `gguf::dequant_row` has no
/// decode for those three until the host decode lands, and a port of it then
/// would be a second reading of the same C.
pub fn ref_dequant(fmt: IqFormat, bytes: &[u8], out: &mut [f32]) -> Result<(), QuantError> {
    let bb = fmt.block_bytes();
    let half = |b: &[u8], at: usize| half_to_f32(u16::from_le_bytes([b[at], b[at + 1]]));
    match fmt {
        IqFormat::Iq2S => {
            for (blk, y) in bytes.chunks_exact(bb).zip(out.as_chunks_mut::<256>().0) {
                let d = half(blk, 0);
                for ib32 in 0..8 {
                    let sc = blk[74 + ib32];
                    let db = [
                        d * (0.5 + f32::from(sc & 0xf)) * 0.25,
                        d * (0.5 + f32::from(sc >> 4)) * 0.25,
                    ];
                    for l in 0..4 {
                        let index = usize::from(blk[2 + 4 * ib32 + l])
                            | ((usize::from(blk[66 + ib32]) << (8 - 2 * l)) & 0x300);
                        let grid = IQ2S_GRID[index].to_le_bytes();
                        let signs = blk[34 + 4 * ib32 + l];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 {
                                -1.0f32
                            } else {
                                1.0
                            };
                            y[32 * ib32 + 8 * l + j] = db[l / 2] * f32::from(grid[j]) * sign;
                        }
                    }
                }
            }
            Ok(())
        }
        IqFormat::Iq1S => {
            for (blk, y) in bytes.chunks_exact(bb).zip(out.as_chunks_mut::<256>().0) {
                let d = half(blk, 0);
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([blk[34 + 2 * ib], blk[35 + 2 * ib]]);
                    let dl = d * (2 * i32::from((qh >> 12) & 7) + 1) as f32;
                    let delta = if qh & 0x8000 != 0 { -0.125f32 } else { 0.125 };
                    for l in 0..4 {
                        let index = usize::from(blk[2 + 4 * ib + l])
                            | (usize::from((qh >> (3 * l)) & 7) << 8);
                        let grid = IQ1S_GRID[index].to_le_bytes();
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta);
                        }
                    }
                }
            }
            Ok(())
        }
        IqFormat::Iq1M => {
            for (blk, y) in bytes.chunks_exact(bb).zip(out.as_chunks_mut::<256>().0) {
                let sc: [u16; 4] =
                    std::array::from_fn(|i| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]));
                // The C's expression, not the card rule's `iq1m_scale_bits`.
                let d = half_to_f32(
                    (sc[0] >> 12)
                        | ((sc[1] >> 8) & 0x00f0)
                        | ((sc[2] >> 4) & 0x0f00)
                        | (sc[3] & 0xf000),
                );
                for ib in 0..8 {
                    let dl = [
                        d * (2 * i32::from((sc[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as f32,
                        d * (2 * i32::from((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as f32,
                    ];
                    let (qh0, qh1) = (blk[32 + 2 * ib], blk[33 + 2 * ib]);
                    let idx = [
                        usize::from(blk[4 * ib]) | ((usize::from(qh0) << 8) & 0x700),
                        usize::from(blk[4 * ib + 1]) | ((usize::from(qh0) << 4) & 0x700),
                        usize::from(blk[4 * ib + 2]) | ((usize::from(qh1) << 8) & 0x700),
                        usize::from(blk[4 * ib + 3]) | ((usize::from(qh1) << 4) & 0x700),
                    ];
                    let delta = [
                        if qh0 & 0x08 != 0 { -0.125f32 } else { 0.125 },
                        if qh0 & 0x80 != 0 { -0.125 } else { 0.125 },
                        if qh1 & 0x08 != 0 { -0.125 } else { 0.125 },
                        if qh1 & 0x80 != 0 { -0.125 } else { 0.125 },
                    ];
                    for l in 0..4 {
                        let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                        for j in 0..8 {
                            y[32 * ib + 8 * l + j] =
                                dl[l / 2] * (f32::from(grid[j] as i8) + delta[l]);
                        }
                    }
                }
            }
            Ok(())
        }
        _ => dequant_row(fmt.ggml(), bytes, out),
    }
}

/// One row's dot by the host rule: the 32 lane partials, then the
/// butterfly.
#[must_use]
pub fn dot_host(fmt: IqFormat, row: &[u8], col: &Col) -> f32 {
    butterfly(std::array::from_fn(|lane| {
        lane_partial_host(fmt, row, &col.0, &col.1, lane)
    }))
}

/// `Σ_k |t_k|` of the rule's lane terms for one row and column, and the
/// largest lane iteration count `n_it`.
#[must_use]
pub fn term_mag(fmt: IqFormat, row: &[u8], col: &Col) -> (f64, usize) {
    let bb = fmt.block_bytes();
    let codes = col.0.as_chunks::<SUB_VALUES>().0;
    let n_sub = 8 * row.len() / bb;
    let mut mag = 0.0f64;
    for (b, xq) in codes.iter().enumerate().take(n_sub) {
        let blk = &row[bb * (b / 8)..bb * (b / 8 + 1)];
        let (i, m) = sub_sums_host(fmt, blk, b % 8, xq);
        let (si, sm) = sub_scales_host(fmt, blk);
        let dx = f64::from(col.1[b / 4]);
        mag +=
            (f64::from(si) * dx * f64::from(i)).abs() + (f64::from(sm) * dx * f64::from(m)).abs();
    }
    (mag, n_sub.div_ceil(32))
}

/// The exact dot of a dequantized row `w` with the f32 activations `x`, and
/// the module doc's bound on the card's error; `col` is `x` quantized.
#[must_use]
pub fn dot_ref(fmt: IqFormat, row: &[u8], w: &[f32], x: &[f32], col: &Col) -> (f64, f64) {
    let (mut exact, mut quant, mut wx) = (0.0f64, 0.0f64, 0.0f64);
    for (b, (wb, xb)) in w.chunks(128).zip(x.chunks(128)).enumerate() {
        let d = f64::from(col.1[b]);
        let mut sw = 0.0f64;
        for (&wi, &xi) in wb.iter().zip(xb) {
            exact += f64::from(wi) * f64::from(xi);
            sw += f64::from(wi).abs();
            wx += (f64::from(wi) * f64::from(xi)).abs();
        }
        quant += d / 2.0 * (1.0 + 256.0 * U) * sw;
    }
    let (mag, n_it) = term_mag(fmt, row, col);
    let deq = if fmt == IqFormat::Q2K { U * wx } else { 0.0 };
    (exact, quant + gamma(2 * n_it + 6) * mag + deq)
}
