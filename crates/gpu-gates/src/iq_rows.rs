//! What the i-quant row gates (`gate_iq`, `gate_iq_sel`) share: the rows
//! they read and the host side they hold the card to.
//!
//! Rows. [`synth_rows`] is `dequant_ref --synthetic`'s dump for `just
//! gate-1-1` (`$BLOOMERY_DATA/ref-synth/<type>.blocks`: 64 rows ggml
//! quantized, then 16 rows of random codes, [`SYNTH_K`] values each) for the
//! formats the harness writes. IQ2_XXS has no dump: a new synthetic type
//! moves the harness's md5, which the dequant families pin
//! (`crates/refset/src/arch/dequant.rs`), so its 80 rows are
//! [`seeded_blocks`] — random code bytes under a finite `d` — which reach
//! every grid index, sign index and scale; its decode is held to ggml's by
//! `gguf`'s `iq2_xxs_matches_a_ggml_made_block`.
//!
//! Reference. [`dot_host`] is the module rule of `bloomery_gpu::iq` on the
//! host (`lane_partial_host` per lane, then the butterfly), the card's bits.
//! [`dot_ref`] is the exact dot of `gguf::dequant_row`'s row with the f32
//! activations, in f64, and the bound on the card's error (`gate_iq.rs`'s
//! derivation): the activation term `Σ_B (d_B/2)(1 + 256u)·Σ_{i∈B}|w_i|` over
//! 128-value blocks, `γ(2·n_it + 6)·Σ_k |t_k|` over the lane terms
//! ([`term_mag`]), and for Q2_K, whose dequant rounds once, `u·Σ|w_i x_i|`.

use crate::rounding::{U, butterfly, gamma};
use crate::{GateError, data_dir};
use bloomery_gpu::iq::{IqFormat, SUB_VALUES, lane_partial_host, sub_scales_host, sub_sums_host};

/// Values per synthetic row.
pub const SYNTH_K: usize = 4096;
/// Rows of the synthetic set: 64 ggml-quantized and 16 random.
pub const SYNTH_ROWS: usize = 80;
/// The seed of IQ2_XXS's synthetic rows.
const SEED_IQ2_XXS: u64 = 0x1a2b_3c4d_5e6f_7788 ^ 16;

/// One activation column: value-order int8 codes and 128-value block scales.
pub type Col = (Vec<i8>, Vec<f32>);

/// The f16 scale byte offsets of a block of `fmt`.
fn scale_offsets(fmt: IqFormat) -> &'static [usize] {
    match fmt {
        IqFormat::Q2K => &[80, 82],
        IqFormat::Iq2Xxs
        | IqFormat::Iq2Xs
        | IqFormat::Iq3Xxs
        | IqFormat::Iq3S
        | IqFormat::Iq4Xs => &[0],
    }
}

/// Bolt's shared expert input and intermediate widths.
pub const BOLT_SHARED: (usize, usize) = (4096, 5376);

/// `n` seeded blocks of `fmt` ([`crate::GateRng`] from `seed | 1`): every byte random,
/// then each f16 scale a finite half of either sign in ±[2^-10, 2^-2)
/// (exponent field 5..=12).
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
    }
    v
}

/// The format's [`SYNTH_ROWS`] synthetic rows of [`SYNTH_K`] values, row
/// after row (module doc), or the error naming the file.
pub fn synth_rows(fmt: IqFormat) -> Result<Vec<u8>, GateError> {
    let bb = fmt.block_bytes();
    let row_bytes = SYNTH_K / 256 * bb;
    if fmt == IqFormat::Iq2Xxs {
        return Ok(seeded_blocks(fmt, SYNTH_ROWS * SYNTH_K / 256, SEED_IQ2_XXS));
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
