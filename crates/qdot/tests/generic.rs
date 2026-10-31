//! The generic host dot (`qdot::generic`): one block decoder per type, one kernel per activation
//! form.
//!
//! The proof types are three that already have hand kernels, one per family: Q2_K (a K-quant,
//! scale and min per 16 values, Q8_K), IQ4_XS (an i-quant through a 16-entry codebook, a 6-bit
//! scale per 32 values, Q8_K) and IQ4_NL (the same codebook in 32-value blocks, Q8_2_X4). What
//! the clauses pin:
//!
//! - each decoder is the type's own dequant: `(d·scale)·code − dmin·min` reproduces
//!   `gguf::quant::dequant_row` bit for bit, and its codes stay in `-127..=127`;
//! - a decoder's vector forms are its scalar decode, lane for lane, including the defaults a
//!   decoder with nothing else gets (`Plain`, which overrides none of them);
//! - the generic dot is the hand kernel's dot bit for bit, scalar and AVX2, on every shape of
//!   `k` the hand kernel takes, because its float steps are the hand kernel's (the Q8_K steps
//!   for Q2_K and IQ4_XS, the Q8_2_X4 group and tail steps for IQ4_NL);
//! - the dot is within the f64 bound of the exact product of the decoded weights and the
//!   quantized column, independent of any hand kernel;
//! - the entry the dispatch reads agrees with the type's row and does what the hand dispatch
//!   does for the same type (column bytes, the quantized column, the dot);
//! - the entry's encoder refuses by name a column that is not whole blocks and an output buffer
//!   that is not the column's bytes, on both encoder paths;
//! - every row of `gguf::QROWS` that names a quantized activation form has a kernel of that
//!   form, and a row that names none has no kernel.
//!
//! IQ2_S and IQ1_M are the listed types (`qdot::generic::ENTRIES`), with no hand kernel to
//! be equal to: the decode, vector, bound, refusal and entry clauses above hold them, a dispatch
//! clause holds every entry point to the generic kernels, and `gate-quant` holds their dots to
//! mainline's `vec_dot`. IQ1_S's decoder is held by the same decode, vector and bound clauses
//! (its hand kernel owns the dispatch; `tests/qdot.rs` holds that kernel to its mirror).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;
use std::marker::PhantomData;

use gguf::quant::dequant_row;
use gguf::{GgmlType, HostAct, QROWS};
use qdot::generic::{Entry, HostDecode, Iq1m, Iq1s, Iq2s, Iq4Nl, Iq4Xs, Q2k, dot_avx2, dot_scalar};
use qdot::{
    col_bytes, dot_row, dot_row_avx2, dot_row_cols, dot_row_scalar, has_tile, has_wide,
    k_granularity, quantize_col, quantize_col_scalar, supports,
};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32
    }
}

/// A decoder that overrides none of the vector forms: the path a type with only its scalar
/// decode takes.
struct Plain<D>(PhantomData<D>);

impl<D: HostDecode> HostDecode for Plain<D> {
    const TY: GgmlType = D::TY;
    const ACT: HostAct = D::ACT;
    const BYTES: usize = D::BYTES;
    const VALUES: usize = D::VALUES;
    const UNSIGNED: bool = D::UNSIGNED;
    const MINS: bool = D::MINS;

    fn head(blk: &[u8]) -> (f32, f32) {
        D::head(blk)
    }
    fn scale(blk: &[u8], group: usize) -> i32 {
        D::scale(blk, group)
    }
    fn min(blk: &[u8], group: usize) -> i32 {
        D::min(blk, group)
    }
    fn codes(blk: &[u8], sub: usize) -> [i8; 32] {
        D::codes(blk, sub)
    }
}

/// Writes the f16 factors of a block of `ty`, each from its own `next()`. IQ1_M has no field for
/// its `d`: the four nibbles of the value are the top nibbles of its four `sc`.
fn set_f16_fields(ty: GgmlType, blk: &mut [u8], mut next: impl FnMut() -> u16) {
    let mut at = |off: usize| blk[off..off + 2].copy_from_slice(&next().to_le_bytes());
    match ty {
        GgmlType::Q2_K => {
            at(80);
            at(82);
        }
        GgmlType::IQ4_XS | GgmlType::IQ4_NL | GgmlType::IQ2_S | GgmlType::IQ1_S => at(0),
        GgmlType::IQ1_M => {
            let bits = next();
            for i in 0..4 {
                let high = 49 + 2 * i;
                blk[high] = (blk[high] & 0x0f) | (((bits >> (4 * i)) & 0xf) as u8) << 4;
            }
        }
        other => panic!("no f16 fields listed for {other:?}"),
    }
}

/// A finite f16 with an exponent in 2^-7 ..= 2^6, either sign.
fn finite_f16(rng: &mut Lcg) -> u16 {
    let exp = 8 + rng.next() % 14;
    ((rng.next() & 1) << 15 | exp << 10 | rng.next() & 0x3ff) as u16
}

fn block_bytes(ty: GgmlType) -> usize {
    ty.row().expect("a proof type has a row").bytes as usize
}

/// `nb` blocks of random bytes with finite f16 factors.
fn random_row(rng: &mut Lcg, ty: GgmlType, nb: usize) -> Vec<u8> {
    let mut row: Vec<u8> = (0..nb * block_bytes(ty))
        .map(|_| rng.next() as u8)
        .collect();
    for blk in row.chunks_mut(block_bytes(ty)) {
        set_f16_fields(ty, blk, || finite_f16(rng));
    }
    row
}

/// `nb` blocks of `fill` bytes with the f16 factors set to 1.0: every code, scale and min at one
/// extreme.
fn end_row(ty: GgmlType, nb: usize, fill: u8) -> Vec<u8> {
    let mut row = vec![fill; nb * block_bytes(ty)];
    for blk in row.chunks_mut(block_bytes(ty)) {
        set_f16_fields(ty, blk, || 0x3c00);
    }
    row
}

fn weight_rows(rng: &mut Lcg, ty: GgmlType, nb: usize) -> Vec<Vec<u8>> {
    let mut rows = vec![end_row(ty, nb, 0x00), end_row(ty, nb, 0xff)];
    rows.extend((0..8).map(|_| random_row(rng, ty, nb)));
    rows
}

/// Activation columns of `k` values: random, spiked (one large value per 32), constant and zero.
fn activations(rng: &mut Lcg, k: usize) -> Vec<Vec<f32>> {
    let random = |rng: &mut Lcg| {
        (0..k)
            .map(|_| (rng.unit() - 0.5) * 8.0)
            .collect::<Vec<f32>>()
    };
    let spiked = |rng: &mut Lcg| {
        (0..k)
            .map(|i| {
                let v = (rng.unit() - 0.5) * 0.5;
                if i % 32 == 7 { v * 200.0 } else { v }
            })
            .collect::<Vec<f32>>()
    };
    vec![
        random(rng),
        random(rng),
        spiked(rng),
        vec![3.5; k],
        vec![-3.5; k],
        vec![0.0; k],
    ]
}

fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

// ------------------------------------------------------------------ the decode

/// `(d·scale)·code − dmin·min` for every value of block `blk`, in ggml's order of multiplies.
fn decoded_values<D: HostDecode>(blk: &[u8]) -> Vec<f32> {
    let (d, dmin) = D::head(blk);
    let mut out = Vec::new();
    for sub in 0..D::VALUES / 32 {
        let codes = D::codes(blk, sub);
        for (n, &code) in codes.iter().enumerate() {
            let g = 2 * sub + n / 16;
            let dl = d * D::scale(blk, g) as f32;
            let ml = dmin * D::min(blk, g) as f32;
            out.push(dl * f32::from(code) - ml);
        }
    }
    out
}

fn check_decode<D: HostDecode>() {
    let mut rng = Lcg(0x6_e0de_c0de ^ D::BYTES as u64);
    let nb = 24;
    let mut rows = weight_rows(&mut rng, D::TY, nb);
    rows.extend((0..16).map(|_| random_row(&mut rng, D::TY, nb)));
    for (r, row) in rows.iter().enumerate() {
        let mut want = vec![0.0f32; nb * D::VALUES];
        dequant_row(D::TY, row, &mut want).expect("a proof type dequantizes");
        for (b, blk) in row.chunks(D::BYTES).enumerate() {
            let got = decoded_values::<D>(blk);
            for (v, (&g, &w)) in got.iter().zip(&want[b * D::VALUES..]).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{:?} row {r} block {b} value {v}: decoded {g:e}, dequant {w:e}",
                    D::TY
                );
            }
            for sub in 0..D::VALUES / 32 {
                assert!(
                    D::codes(blk, sub).iter().all(|&c| c != i8::MIN),
                    "{:?} row {r} block {b} sub {sub}: a code of -128 breaks the sign fold",
                    D::TY
                );
            }
            if !D::MINS {
                assert!(
                    (0..D::VALUES / 16).all(|g| D::min(blk, g) == 0),
                    "{:?}: a type without MINS has a min",
                    D::TY
                );
            }
        }
    }
}

#[test]
fn decoders_reproduce_ggmls_dequant() {
    check_decode::<Q2k>();
    check_decode::<Iq4Xs>();
    check_decode::<Iq4Nl>();
    check_decode::<Iq2s>();
    check_decode::<Iq1s>();
    check_decode::<Iq1m>();
}

#[target_feature(enable = "avx2")]
unsafe fn vector_forms<D: HostDecode>(blk: &[u8], sub: usize) -> ([i8; 32], [i16; 16]) {
    let (mut codes, mut scales) = ([0i8; 32], [0i16; 16]);
    // SAFETY: AVX2 per this function's contract; `blk` is a whole block and `sub` is in range.
    let (c, s) = unsafe { (D::codes_avx2(blk, sub), D::scales_avx2(blk, sub)) };
    // SAFETY: 32 writable bytes each.
    unsafe {
        _mm256_storeu_si256(codes.as_mut_ptr().cast(), c);
        _mm256_storeu_si256(scales.as_mut_ptr().cast(), s);
    }
    (codes, scales)
}

fn check_vector_decode<D: HostDecode>() {
    assert!(is_x86_feature_detected!("avx2"));
    let mut rng = Lcg(0x7e_c7_0b ^ D::BYTES as u64);
    for row in weight_rows(&mut rng, D::TY, 16) {
        for blk in row.chunks(D::BYTES) {
            for sub in 0..D::VALUES / 32 {
                // SAFETY: AVX2 asserted above; whole block, `sub` in range.
                let (codes, scales) = unsafe { vector_forms::<D>(blk, sub) };
                assert_eq!(codes, D::codes(blk, sub), "{:?} sub {sub}: codes", D::TY);
                for (lane, &s) in scales.iter().enumerate() {
                    let want = D::scale(blk, 2 * sub + lane / 8);
                    assert_eq!(
                        i32::from(s),
                        want,
                        "{:?} sub {sub} lane {lane}: scale",
                        D::TY
                    );
                }
            }
        }
    }
}

#[test]
fn vector_decodes_are_the_scalar_decode() {
    check_vector_decode::<Q2k>();
    check_vector_decode::<Iq4Xs>();
    check_vector_decode::<Iq4Nl>();
    check_vector_decode::<Plain<Q2k>>();
    check_vector_decode::<Plain<Iq4Xs>>();
    check_vector_decode::<Plain<Iq4Nl>>();
    check_vector_decode::<Iq2s>();
    check_vector_decode::<Iq1s>();
    check_vector_decode::<Iq1m>();
    check_vector_decode::<Plain<Iq2s>>();
    check_vector_decode::<Plain<Iq1s>>();
    check_vector_decode::<Plain<Iq1m>>();
}

// --------------------------------------------------------------------- the dot

/// Every shape of `k` the proof types' hand kernels take: whole blocks, and for the 32-value
/// type each remainder of four blocks.
const Q8K_KS: [usize; 4] = [256, 512, 768, 4096];
const Q82X4_KS: [usize; 8] = [32, 64, 96, 128, 160, 224, 512, 2688];

/// The generic dot against the hand dot on the same rows and columns, all four paths bit for bit,
/// and the entry's plumbing against the hand dispatch's.
fn check_dot<D: HostDecode>(ks: &[usize]) {
    assert!(has_avx2_fma(), "the box has AVX2 and FMA");
    let ty = D::TY;
    let entry = Entry::of::<D>();
    let mut rng = Lcg(0x5eed_0d07 ^ D::BYTES as u64);
    for &k in ks {
        let nb = k / D::VALUES;
        assert_eq!(k_granularity(ty), D::VALUES);
        assert_eq!(
            entry.col_bytes(k),
            col_bytes(ty, k),
            "{ty:?} k = {k}: column bytes"
        );
        let rows = weight_rows(&mut rng, ty, nb);
        for (c, x) in activations(&mut rng, k).iter().enumerate() {
            let mut acol = vec![0u8; col_bytes(ty, k)];
            quantize_col(ty, x, &mut acol);
            let (mut hand_s, mut entry_v, mut entry_s) = (acol.clone(), acol.clone(), acol.clone());
            quantize_col_scalar(ty, x, &mut hand_s);
            entry.quantize(x, &mut entry_v, true);
            entry.quantize(x, &mut entry_s, false);
            assert_eq!(
                entry_v, acol,
                "{ty:?} k = {k} col {c}: entry quantize, vector"
            );
            assert_eq!(
                entry_s, hand_s,
                "{ty:?} k = {k} col {c}: entry quantize, scalar"
            );
            for (r, row) in rows.iter().enumerate() {
                let hand_avx2 = dot_row_avx2(ty, row, &acol, k).expect("hand avx2 dot");
                let hand_scalar = dot_row_scalar(ty, row, &acol, k).expect("hand scalar dot");
                let scalar = dot_scalar::<D>(row, &acol, nb);
                // SAFETY: AVX2 and FMA asserted above; the row and the column hold `nb` blocks.
                let avx2 = unsafe { dot_avx2::<D>(row, &acol, nb) };
                let e_scalar = entry.dot_scalar(row, &acol, nb);
                // SAFETY: as above.
                let e_avx2 = unsafe { entry.dot_avx2(row, &acol, nb) };
                for (path, got) in [
                    ("generic scalar", scalar),
                    ("generic avx2", avx2),
                    ("entry scalar", e_scalar),
                    ("entry avx2", e_avx2),
                    ("hand scalar", hand_scalar),
                ] {
                    assert_eq!(
                        got.to_bits(),
                        hand_avx2.to_bits(),
                        "{ty:?} k = {k} col {c} row {r}: {path} {got:e}, hand avx2 {hand_avx2:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn generic_dot_is_the_hand_dot_bit_for_bit() {
    check_dot::<Q2k>(&Q8K_KS);
    check_dot::<Iq4Xs>(&Q8K_KS);
    check_dot::<Iq4Nl>(&Q82X4_KS);
    check_dot::<Plain<Q2k>>(&Q8K_KS);
    check_dot::<Plain<Iq4Xs>>(&Q8K_KS);
    check_dot::<Plain<Iq4Nl>>(&Q82X4_KS);
}

/// A type with no hand kernel reaches its generic kernels through every entry point and nothing
/// else: its entry is the one the dispatch finds, the quantized column is the entry's, and
/// `dot_row`, `dot_row_avx2`, `dot_row_scalar` and the column tile (which has no kernel of its
/// own for it) write the kernels' bits, the AVX2 and the scalar mirror alike.
fn check_dispatch<D: HostDecode>(ks: &[usize]) {
    assert!(has_avx2_fma(), "the box has AVX2 and FMA");
    let ty = D::TY;
    let entry = qdot::generic::entry(ty).expect("a listed type has an entry");
    assert!(supports(ty), "{ty:?}: no kernel");
    assert!(
        !has_tile(ty) && !has_wide(ty),
        "{ty:?}: a tile or wide kernel the generic form does not have"
    );
    let mut rng = Lcg(0xd15_9a7c4 ^ D::BYTES as u64);
    for &k in ks {
        let nb = k / D::VALUES;
        assert_eq!(k_granularity(ty), D::VALUES);
        let rows = weight_rows(&mut rng, ty, nb);
        let cols: Vec<Vec<u8>> = activations(&mut rng, k)
            .iter()
            .map(|x| {
                let mut acol = vec![0u8; col_bytes(ty, k)];
                quantize_col(ty, x, &mut acol);
                let mut want = vec![0u8; entry.col_bytes(k)];
                entry.quantize(x, &mut want, true);
                assert_eq!(acol, want, "{ty:?} k = {k}: the dispatch's column");
                acol
            })
            .collect();
        for (r, row) in rows.iter().enumerate() {
            let mut tiled = vec![0f32; cols.len()];
            let refs: Vec<&[u8]> = cols.iter().map(Vec::as_slice).collect();
            dot_row_cols(ty, row, &refs, k, &mut tiled).expect("a tile of columns");
            for (c, acol) in cols.iter().enumerate() {
                // SAFETY: AVX2 and FMA asserted above; the row and the column hold `nb` blocks.
                let avx2 = unsafe { dot_avx2::<D>(row, acol, nb) };
                let scalar = dot_scalar::<D>(row, acol, nb);
                assert_eq!(
                    avx2.to_bits(),
                    scalar.to_bits(),
                    "{ty:?} k = {k} col {c} row {r}: avx2 {avx2:e}, scalar mirror {scalar:e}"
                );
                for (path, got) in [
                    ("dot_row", dot_row(ty, row, acol, k).expect("dot_row")),
                    (
                        "dot_row_avx2",
                        dot_row_avx2(ty, row, acol, k).expect("dot_row_avx2"),
                    ),
                    (
                        "dot_row_scalar",
                        dot_row_scalar(ty, row, acol, k).expect("dot_row_scalar"),
                    ),
                    ("tile", tiled[c]),
                ] {
                    // `dot_row_scalar` is the scalar mirror, the rest the AVX2 kernel; they are equal.
                    assert_eq!(
                        got.to_bits(),
                        avx2.to_bits(),
                        "{ty:?} k = {k} col {c} row {r}: {path} {got:e}, kernel {avx2:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_type_with_no_hand_kernel_reaches_its_generic_kernels() {
    // PIN(2026-10-31): IQ1_S left this clause when its hand kernel took its dispatch; tests/qdot.rs holds that
    // kernel to its mirror and its column tile to the one-column dot.
    check_dispatch::<Iq2s>(&Q8K_KS);
    check_dispatch::<Iq1m>(&Q8K_KS);
}

/// The quantized column's blocks as `(dy, codes)`: one Q8_K block of 256 values, or one Q8_2
/// block of 32 (a group's four, then the tail blocks).
fn column_blocks(act: HostAct, acol: &[u8], nb: usize) -> Vec<(f64, Vec<i8>)> {
    match act {
        HostAct::Q8K => (0..nb)
            .map(|i| {
                let b = &acol[296 * i..296 * (i + 1)];
                let dy = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                (dy, b[8..264].iter().map(|&q| q as i8).collect())
            })
            .collect(),
        HostAct::Q8_2X4 => {
            let nb4 = nb / 4 * 4;
            (0..nb)
                .map(|i| {
                    let (head, qs) = if i < nb4 {
                        let g = &acol[144 * (i / 4)..144 * (i / 4 + 1)];
                        let l = i % 4;
                        (&g[2 * l..2 * l + 2], &g[16 + 32 * l..16 + 32 * l + 32])
                    } else {
                        let t = nb4 / 4 * 144 + (i - nb4) * 36;
                        (&acol[t..t + 2], &acol[t + 4..t + 36])
                    };
                    let dy =
                        f32::from_bits(u32::from(u16::from_le_bytes([head[0], head[1]])) << 16);
                    (f64::from(dy), qs.iter().map(|&q| q as i8).collect())
                })
                .collect()
        }
        other => panic!("no column form for {other:?}"),
    }
}

/// The exact value in f64 of the decoded weights against the quantized column, and the sum of the
/// magnitudes of its terms (the scale part and the min part of a block apart).
fn exact_dot<D: HostDecode>(row: &[u8], acol: &[u8], nb: usize) -> (f64, f64) {
    let (mut exact, mut mass) = (0.0f64, 0.0f64);
    for (i, (dy, q)) in column_blocks(D::ACT, acol, nb).iter().enumerate() {
        let blk = &row[D::BYTES * i..D::BYTES * (i + 1)];
        let (d, dmin) = D::head(blk);
        let (mut coded, mut coded_abs, mut folded, mut folded_abs) = (0i64, 0i64, 0i64, 0i64);
        for v in 0..D::VALUES {
            let g = v / 16;
            let code = i64::from(D::codes(blk, v / 32)[v % 32]);
            let y = i64::from(q[v % q.len()]);
            let scale = i64::from(D::scale(blk, g));
            let min = i64::from(D::min(blk, g));
            coded += scale * code * y;
            coded_abs += (scale * code * y).abs();
            folded += min * y;
            folded_abs += (min * y).abs();
        }
        let (d, dmin) = (f64::from(d) * dy, f64::from(dmin) * dy);
        exact += d * coded as f64 - dmin * folded as f64;
        mass += d.abs() * coded_abs as f64 + dmin.abs() * folded_abs as f64;
    }
    (exact, mass)
}

fn check_bound<D: HostDecode>(ks: &[usize]) {
    assert!(has_avx2_fma(), "the box has AVX2 and FMA");
    let mut rng = Lcg(0xb0_0d ^ D::BYTES as u64);
    for &k in ks {
        let nb = k / D::VALUES;
        // Two roundings per block on a lane (the min fold and the codes, or one group step),
        // the integer to float and the scale product on each term, and the lane reduction.
        let n_u = (2 * nb + 10) as f64 * 2f64.powi(-24);
        for x in activations(&mut rng, k) {
            let mut acol = vec![0u8; col_bytes(D::TY, k)];
            quantize_col(D::TY, &x, &mut acol);
            for (r, row) in weight_rows(&mut rng, D::TY, nb).iter().enumerate() {
                let (exact, mass) = exact_dot::<D>(row, &acol, nb);
                let bound = n_u / (1.0 - n_u) * mass;
                // SAFETY: AVX2 and FMA asserted above; the row and the column hold `nb` blocks.
                let avx2 = unsafe { dot_avx2::<D>(row, &acol, nb) };
                for (path, got) in [("scalar", dot_scalar::<D>(row, &acol, nb)), ("avx2", avx2)] {
                    let got = f64::from(got);
                    assert!(
                        (got - exact).abs() <= bound,
                        "{:?} k = {k} row {r} {path}: got {got:e}, exact {exact:e}, bound {bound:e}",
                        D::TY
                    );
                }
            }
        }
    }
}

#[test]
fn generic_dot_is_within_its_f64_bound() {
    check_bound::<Q2k>(&Q8K_KS);
    check_bound::<Iq4Xs>(&Q8K_KS);
    check_bound::<Iq4Nl>(&Q82X4_KS);
    check_bound::<Iq2s>(&Q8K_KS);
    check_bound::<Iq1s>(&Q8K_KS);
    check_bound::<Iq1m>(&Q8K_KS);
}

// ------------------------------------------------------------ the encoder's refusals

/// `entry.quantize(x, out, avx2)` panics, and its message (an `assert_eq!` wraps the text it was
/// given) names `refusal`.
fn assert_refused(entry: &Entry, x: &[f32], out: &mut [u8], avx2: bool, refusal: &str) {
    let (ty, values, bytes) = (entry.ty, x.len(), out.len());
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        entry.quantize(x, out, avx2);
    })) else {
        panic!("{ty:?} avx2 = {avx2}: {values} values into {bytes} bytes were not refused");
    };
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("a refusal carries a text message");
    assert!(
        message.contains(refusal),
        "{ty:?} avx2 = {avx2}: {values} values into {bytes} bytes refused with {message:?}, not {refusal:?}"
    );
}

fn proof_entries() -> [Entry; 6] {
    [
        Entry::of::<Q2k>(),
        Entry::of::<Iq4Xs>(),
        Entry::of::<Iq4Nl>(),
        Entry::of::<Iq2s>(),
        Entry::of::<Iq1s>(),
        Entry::of::<Iq1m>(),
    ]
}

#[test]
fn quantize_refuses_a_partial_block() {
    for entry in proof_entries() {
        // A non-finite value past the last whole block is what a silent remainder would drop.
        let mut x = vec![1.0f32; 2 * entry.values];
        x.push(f32::NAN);
        for avx2 in [false, true] {
            // Sized for the partial column, so only the block check can refuse it.
            let mut out = vec![0u8; entry.col_bytes(x.len())];
            assert_refused(
                &entry,
                &x,
                &mut out,
                avx2,
                "generic qdot: incomplete activation block",
            );
        }
    }
}

#[test]
fn quantize_refuses_a_wrong_buffer_length() {
    for entry in proof_entries() {
        // Five blocks: a Q8_2_X4 column of that size is one group and one tail block.
        let x = vec![1.0f32; 5 * entry.values];
        let want = entry.col_bytes(x.len());
        for avx2 in [false, true] {
            for len in [0, want - 1, want + 1, 2 * want] {
                let mut out = vec![0u8; len];
                assert_refused(
                    &entry,
                    &x,
                    &mut out,
                    avx2,
                    "generic qdot: wrong activation buffer length",
                );
            }
        }
    }
}

// ---------------------------------------------------------------- the dispatch

#[test]
fn entries_agree_with_their_rows() {
    for entry in proof_entries() {
        assert!(
            entry.agrees_with_row(),
            "{:?}: bytes {}, values {}, form {:?} against its row {:?}",
            entry.ty,
            entry.bytes,
            entry.values,
            entry.act,
            entry.ty.row()
        );
    }
}

/// The bytes one column of `k` values takes in the activation form a row names.
fn form_bytes(act: HostAct, k: usize) -> usize {
    match act {
        HostAct::Q8K => k / 256 * 296,
        HostAct::Q8_2X4 => k / 128 * 144 + (k % 128) / 32 * 36,
        HostAct::Q8_0 => k / 32 * 34,
        other => panic!("{other:?} is no quantized form"),
    }
}

#[test]
fn a_row_names_a_quantized_form_exactly_when_a_kernel_reads_it() {
    assert!(has_avx2_fma(), "the box has AVX2 and FMA");
    for row in QROWS {
        let quantized = matches!(row.host_act, HostAct::Q8K | HostAct::Q8_2X4 | HostAct::Q8_0);
        assert_eq!(
            supports(row.ty),
            quantized,
            "{}: host_act {:?}, but the kernel list says {}",
            row.name,
            row.host_act,
            supports(row.ty)
        );
        if quantized {
            assert_eq!(
                k_granularity(row.ty) as u64,
                row.blck,
                "{}: block values",
                row.name
            );
            assert_eq!(
                col_bytes(row.ty, 512),
                form_bytes(row.host_act, 512),
                "{}: column bytes in the form {:?}",
                row.name,
                row.host_act
            );
        }
    }
}
