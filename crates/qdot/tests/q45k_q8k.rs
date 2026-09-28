//! The Q4_K and Q5_K x q8_K column tile gate (`qdot::dot_q45k_q8k_cols`, the
//! host tier's routed down): the tile against its scalar mirror, bit for bit,
//! at every column count 1..=TILE_COLS and every tile slot a column can sit
//! in; the mirror against the f64 dot of ggml's dequantized rows
//! (`gguf::quant::dequant_row`) on the same codes, within its float band; the
//! q8_K encoder's two paths byte for byte and its refusal of a non-finite
//! block; and the entry's shape refusals.
//!
//! Rows: synthetic blocks at the ends of the rule's integer terms (codes 0
//! and the type's top, scales and mins 0 and 63, a ramp), then the V4.1 set's
//! routed down stacks at k = 2304 — Q5_K from the first shard (block 0), Q4_K
//! from the first shard past it that carries one. Columns: seeded ones over
//! six magnitudes, zeros and ±3 through the encoder, and synthetic blocks
//! whose codes are −127, 127, −128 (outside the encoder's range, still
//! exact) and ±127 by 32-value sub-block. The bit contract holds by
//! construction — exact integer sums per super-block, then the same two fmas
//! per column — so any differing value is a bug, not rounding.
//!
//! `hw_` tests need the box (AVX2+FMA+F16C; the V4.1 shards) and are
//! `#[ignore]`d; `just gate-qdot` runs every test.

use gguf::GgmlType;
use gguf::quant::dequant_row;
use qdot::{
    QdotError, TILE_COLS, dot_q45k_q8k_cols, dot_q45k_q8k_cols_scalar, q8k_col_bytes,
    quantize_col_q8k, quantize_col_q8k_scalar,
};

/// f32's unit roundoff.
const U: f64 = 1.0 / 16_777_216.0;

/// Rows each real-row clause dots.
const ROWS: usize = 128;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32) / 2147483648.0 - 1.0
    }
}

/// Bytes of one super-block of `ty`.
fn block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_K => 144,
        GgmlType::Q5_K => 176,
        _ => panic!("no q8_K tile for {ty:?}"),
    }
}

/// The largest code of `ty`.
fn top(ty: GgmlType) -> u8 {
    if ty == GgmlType::Q5_K { 31 } else { 15 }
}

/// A synthetic Q4_K or Q5_K super-block: code `i` is `q(i)`, sub-block `s`
/// has scale `sc(s)` and min `m(s)` (six bits), and the f16 bits `d`, `dmin`
/// — packed as `get_scale_min_k4` reads them.
fn kblock(
    ty: GgmlType,
    q: impl Fn(usize) -> u8,
    sc: impl Fn(usize) -> u8,
    m: impl Fn(usize) -> u8,
    d: u16,
    dmin: u16,
) -> Vec<u8> {
    let mut b = vec![0u8; block_bytes(ty)];
    b[0..2].copy_from_slice(&d.to_le_bytes());
    b[2..4].copy_from_slice(&dmin.to_le_bytes());
    for j in 0..4 {
        let (s0, s1, m0, m1) = (sc(j), sc(j + 4), m(j), m(j + 4));
        assert!(s0 < 64 && s1 < 64 && m0 < 64 && m1 < 64);
        b[4 + j] = s0 | ((s1 >> 4) << 6);
        b[8 + j] = m0 | ((m1 >> 4) << 6);
        b[12 + j] = (s1 & 0xF) | ((m1 & 0xF) << 4);
    }
    let q5 = ty == GgmlType::Q5_K;
    let qs = if q5 { 48 } else { 16 };
    for i in 0..256 {
        let v = q(i);
        assert!(v <= top(ty), "code {i}: {v}");
        let (chunk, r) = (i / 64, i % 64);
        let at = qs + 32 * chunk + r % 32;
        b[at] |= if r < 32 { v & 0xF } else { (v & 0xF) << 4 };
        if q5 && v & 0x10 != 0 {
            b[16 + r % 32] |= 1 << (i / 32);
        }
    }
    b
}

/// Synthetic rows of `ty`, two super-blocks each (k = 512): the codes' ends
/// (0, the top, alternating, a ramp) against scales and mins at 0, 63 and
/// ramps, `d` and `dmin` at 1.0 / 0.5 and 2^-7 / 2^-9.
fn synthetic_rows(ty: GgmlType) -> Vec<u8> {
    let t = top(ty);
    type Kind = (
        Box<dyn Fn(usize) -> u8>,
        Box<dyn Fn(usize) -> u8>,
        Box<dyn Fn(usize) -> u8>,
    );
    let kinds: Vec<Kind> = vec![
        (Box::new(move |_| t), Box::new(|_| 63), Box::new(|_| 63)),
        (Box::new(|_| 0), Box::new(|_| 63), Box::new(|_| 63)),
        (Box::new(move |_| t), Box::new(|_| 63), Box::new(|_| 0)),
        (
            Box::new(move |i| if i % 2 == 0 { t } else { 0 }),
            Box::new(|s| 8 * s as u8 + 7),
            Box::new(|s| 63 - 8 * s as u8),
        ),
        (
            Box::new(move |i| (i % (t as usize + 1)) as u8),
            Box::new(|s| if s % 2 == 0 { 63 } else { 1 }),
            Box::new(|s| s as u8),
        ),
        (
            Box::new(move |i| if (i / 32) % 2 == 0 { t } else { 1 }),
            Box::new(|s| 63 - s as u8),
            Box::new(|s| if s % 2 == 0 { 0 } else { 63 }),
        ),
    ];
    let mut rows = Vec::new();
    for (q, sc, m) in &kinds {
        for (d, dmin) in [(0x3C00u16, 0x3800u16), (0x2000, 0x1800)] {
            rows.extend(kblock(ty, q, sc, m, d, dmin));
        }
    }
    rows
}

/// A synthetic q8_K column of `k` values in the encoder's layout: code `i`
/// is `q(i)`, every block's scale `d`, and each block's 16 sums of 16 codes.
fn q8k_col(k: usize, q: impl Fn(usize) -> i8, d: f32) -> Vec<u8> {
    let mut a = vec![0u8; q8k_col_bytes(k)];
    for (sb, blk) in a.as_chunks_mut::<296>().0.iter_mut().enumerate() {
        blk[0..4].copy_from_slice(&d.to_le_bytes());
        for j in 0..16 {
            let mut sum = 0i32;
            for l in 0..16 {
                let v = q(256 * sb + 16 * j + l);
                blk[8 + 16 * j + l] = v as u8;
                sum += i32::from(v);
            }
            let sum = i16::try_from(sum).expect("16 i8 codes sum within i16");
            blk[264 + 2 * j..266 + 2 * j].copy_from_slice(&sum.to_le_bytes());
        }
    }
    a
}

/// The clauses' columns of `k` values: seeded ones over six magnitudes with
/// outliers every 61st value, zeros and ±3 alternating, each through the
/// encoder, then the synthetic ends.
fn columns(k: usize) -> Vec<Vec<u8>> {
    let mut rng = Lcg(0x08c5_d0e7);
    let mut xs: Vec<Vec<f32>> = [1e-3f32, 0.02, 0.3, 1.0, 7.5, 1e2]
        .iter()
        .map(|&mag| {
            (0..k)
                .map(|i| {
                    let u = rng.unit() * mag;
                    if i % 61 == 0 { u * 8.0 } else { u }
                })
                .collect()
        })
        .collect();
    xs.push(vec![0.0; k]);
    xs.push(
        (0..k)
            .map(|i| if i % 2 == 0 { -3.0 } else { 3.0 })
            .collect(),
    );
    let mut cols: Vec<Vec<u8>> = xs
        .iter()
        .map(|x| {
            let mut a = vec![0u8; q8k_col_bytes(k)];
            quantize_col_q8k(x, &mut a);
            a
        })
        .collect();
    cols.push(q8k_col(k, |_| -127, 0.01));
    cols.push(q8k_col(k, |_| 127, 0.01));
    cols.push(q8k_col(k, |_| -128, 0.01));
    cols.push(q8k_col(
        k,
        |i| if (i / 32) % 2 == 0 { -127 } else { 127 },
        0.02,
    ));
    cols
}

/// For c = 1..=TILE_COLS and every starting column of the cyclic list
/// `acols`, every row's tile over the c columns from there equals the mirror
/// of each column alone, bit for bit; outputs start as NaN, so an unwritten
/// one fails too. Returns the calls made.
fn assert_tile_matches(
    ty: GgmlType,
    label: &str,
    k: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) -> usize {
    let row_bytes = block_bytes(ty) * k / 256;
    let rows = bytes.len() / row_bytes;
    let n = acols.len();
    assert!(n >= TILE_COLS, "{label}: {n} columns cannot fill a tile");
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        for (j, a) in acols.iter().enumerate() {
            let mut v = [f32::NAN];
            dot_q45k_q8k_cols_scalar(ty, src, &[a], k, &mut v).unwrap();
            want[r * n + j] = v[0];
        }
    }
    let mut calls = 0;
    for c in 1..=TILE_COLS {
        for s in 0..n {
            let idx: Vec<usize> = (0..c).map(|i| (s + i) % n).collect();
            let cols: Vec<&[u8]> = idx.iter().map(|&j| acols[j].as_slice()).collect();
            for r in 0..rows {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let mut out = vec![f32::NAN; c];
                dot_q45k_q8k_cols(ty, src, &cols, k, &mut out).unwrap();
                for (slot, (&j, &got)) in idx.iter().zip(&out).enumerate() {
                    let w = want[r * n + j];
                    assert_eq!(
                        got.to_bits(),
                        w.to_bits(),
                        "{label}: row {r}, c = {c}, slot {slot} (column {j}): tile {got:e} \
                         (bits {:#x}) vs mirror {w:e} (bits {:#x})",
                        got.to_bits(),
                        w.to_bits()
                    );
                }
                calls += 1;
            }
        }
    }
    calls
}

/// `dmin · m_s` of each value of a super-block, `m_s` its sub-block's min
/// (`get_scale_min_k4`, read here on its own).
fn mins_of(blk: &[u8]) -> [f64; 256] {
    let q = &blk[4..16];
    let dmin = f64::from(gguf::quant::half_to_f32(u16::from_le_bytes([
        blk[2], blk[3],
    ])));
    let m = |j: usize| -> f64 {
        f64::from(if j < 4 {
            q[j + 4] & 63
        } else {
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
        })
    };
    std::array::from_fn(|i| dmin * m(i / 32))
}

/// Every row's mirror value against `Σ_i w_i · dy · a_i` in f64 over ggml's
/// dequantized row (`dequant_row`, f32 within u of `d·sc·q − dmin·m`): within
/// `64u · Σ(|w_i| + 2·dmin·m_i)·|dy · a_i|`, which bounds the mirror's leaves
/// (`|d·sc·q| ≤ |w| + dmin·m`) — its float path is two products and two fmas
/// a super-block, and the f32 conversion of an integer sum past 2^24.
/// Returns the worst ratio to the band.
fn assert_mirror_is_ggml(
    ty: GgmlType,
    label: &str,
    k: usize,
    bytes: &[u8],
    acols: &[Vec<u8>],
) -> f64 {
    let row_bytes = block_bytes(ty) * k / 256;
    let rows = bytes.len() / row_bytes;
    let mut w = vec![0.0f32; k];
    let mut worst = 0.0f64;
    for r in 0..rows {
        let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
        dequant_row(ty, src, &mut w).unwrap();
        let bb = block_bytes(ty);
        let mins: Vec<f64> = src.chunks_exact(bb).flat_map(mins_of).collect();
        for (j, a) in acols.iter().enumerate() {
            let (mut exact, mut mag) = (0.0f64, 0.0f64);
            for (i, &wv) in w.iter().enumerate() {
                let blk = &a[(i / 256) * 296..(i / 256 + 1) * 296];
                let dy = f64::from(f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]));
                let ya = dy * f64::from(blk[8 + i % 256] as i8);
                exact += f64::from(wv) * ya;
                mag += (f64::from(wv).abs() + 2.0 * mins[i]) * ya.abs();
            }
            let mut v = [f32::NAN];
            dot_q45k_q8k_cols_scalar(ty, src, &[a], k, &mut v).unwrap();
            let band = 64.0 * U * mag + f64::from(f32::from_bits(1));
            let ratio = (f64::from(v[0]) - exact).abs() / band;
            assert!(
                ratio <= 1.0,
                "{label}: row {r}, column {j}: mirror {:e} vs ggml's rows in f64 {exact:e}, \
                 band {band:e}",
                v[0]
            );
            worst = worst.max(ratio);
        }
    }
    worst
}

/// The V4.1 routed down stack of `ty`: (name, k, its first `rows` rows).
fn v41_down(ty: GgmlType, rows: usize) -> (String, usize, Vec<u8>) {
    use std::os::unix::fs::FileExt;
    let first = gguf::v41::model();
    let shards: Vec<String> = match first.find("-00001-of-") {
        Some(at) => (1..=99)
            .map(|i| format!("{}-{i:05}-of-{}", &first[..at], &first[at + 10..]))
            .take_while(|p| std::path::Path::new(p).exists())
            .collect(),
        None => vec![first.clone()],
    };
    for path in &shards {
        let inv = gguf::inventory_of(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let Some(t) = inv
            .tensors
            .iter()
            .find(|t| t.type_id == ty.as_u32() && t.name.contains("ffn_down_exps"))
        else {
            continue;
        };
        let k = t.dims[0] as usize;
        let row_bytes = block_bytes(ty) * k / 256;
        let mut bytes = vec![0u8; rows * row_bytes];
        std::fs::File::open(path)
            .and_then(|f| f.read_exact_at(&mut bytes, inv.data_base + t.offset))
            .unwrap_or_else(|e| panic!("{path}: read {}: {e}", t.name));
        return (t.name.clone(), k, bytes);
    }
    panic!("no shard of {first} carries a {ty:?} ffn_down_exps stack");
}

/// The tile clause, then the mirror clause, on synthetic rows at k = 512 and
/// on the V4.1 down stack of each type.
#[test]
#[ignore = "hw: needs the box (AVX2+FMA+F16C) and the V4.1 shards"]
fn hw_q45k_q8k_tile_matches_mirror_and_ggml() {
    assert!(
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"),
        "the tile clause would compare the mirror with itself"
    );
    for ty in [GgmlType::Q4_K, GgmlType::Q5_K] {
        let k = 512;
        let rows = synthetic_rows(ty);
        let cols = columns(k);
        let calls = assert_tile_matches(ty, "synthetic ends", k, &rows, &cols);
        let worst = assert_mirror_is_ggml(ty, "synthetic ends", k, &rows, &cols);
        eprintln!(
            "{ty:?} q8_K tile: synthetic ends (k = {k}): {calls} calls, c = 1..=8, bit-identical; \
             mirror within {worst:.3} of ggml's band"
        );
        let (name, k, bytes) = v41_down(ty, ROWS);
        assert_eq!(k, 2304, "{name}: the routed down contracts ff = 2304");
        let cols = columns(k);
        let calls = assert_tile_matches(ty, &name, k, &bytes, &cols);
        let worst = assert_mirror_is_ggml(ty, &name, k, &bytes, &cols);
        eprintln!(
            "{ty:?} q8_K tile: {name} (k = {k}): {calls} calls, c = 1..=8, bit-identical; \
             mirror within {worst:.3} of ggml's band"
        );
    }
}

/// The encoder's AVX2 and scalar paths write the same bytes, and those are
/// the Q3_K activation column's (the one q8_K encoder).
#[test]
#[ignore = "hw: needs the box (the AVX2 encoder)"]
fn hw_q8k_encoder_paths_agree() {
    let k = 2304;
    let mut rng = Lcg(0x5eed);
    for mag in [1e-6f32, 1e-2, 1.0, 3e3] {
        let x: Vec<f32> = (0..k).map(|_| rng.unit() * mag).collect();
        let (mut a, mut b, mut c) = (
            vec![0xA5u8; q8k_col_bytes(k)],
            vec![0x5Au8; q8k_col_bytes(k)],
            vec![0x33u8; q8k_col_bytes(k)],
        );
        quantize_col_q8k(&x, &mut a);
        quantize_col_q8k_scalar(&x, &mut b);
        qdot::quantize_col(GgmlType::Q3_K, &x, &mut c);
        assert_eq!(a, b, "magnitude {mag}: the AVX2 and the scalar encoder");
        assert_eq!(
            a, c,
            "magnitude {mag}: the q8_K encoder and the Q3_K column's"
        );
    }
}

/// A block holding a NaN or an infinity is refused by name by both encoder
/// paths, in any block of the column, never encoded.
#[test]
fn q8k_encoder_refuses_a_non_finite_block() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in [0usize, 255, 256 + 77, 2303] {
            let mut x = vec![0.25f32; 2304];
            x[at] = bad;
            for enc in [
                quantize_col_q8k as fn(&[f32], &mut [u8]),
                quantize_col_q8k_scalar,
            ] {
                let mut out = vec![0u8; q8k_col_bytes(2304)];
                let r =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enc(&x, &mut out)));
                let msg = r
                    .err()
                    .and_then(|e| e.downcast::<String>().ok().map(|s| *s))
                    .unwrap_or_else(|| panic!("{bad} at {at}: the column was encoded"));
                assert!(
                    msg.contains("non-finite activation"),
                    "{bad} at {at}: refused, but not by name: {msg}"
                );
            }
        }
    }
}

/// One of the two entries: the tile or the mirror.
type Entry = fn(GgmlType, &[u8], &[&[u8]], usize, &mut [f32]) -> Result<(), QdotError>;

/// A type without a q8_K kernel, a column count outside `1..=TILE_COLS`, an
/// `out` of another length, a `k` off the grid, a short row and a short
/// column are named refusals, on both entries.
#[test]
fn q45k_q8k_refuses_bad_shapes() {
    let k = 512;
    let wrow = vec![0u8; 2 * 176];
    let a = vec![0u8; q8k_col_bytes(k)];
    let many: Vec<&[u8]> = vec![&a; TILE_COLS + 1];
    let mut out = vec![0.0f32; TILE_COLS + 1];
    let entries: [(&str, Entry); 2] = [
        ("tile", dot_q45k_q8k_cols),
        ("mirror", dot_q45k_q8k_cols_scalar),
    ];
    for (name, f) in entries {
        assert_eq!(
            f(GgmlType::Q6_K, &wrow, &many[..1], k, &mut out[..1]),
            Err(QdotError::NoQ8kKernel(GgmlType::Q6_K)),
            "{name}"
        );
        assert_eq!(
            f(GgmlType::Q4_K, &wrow, &many, k, &mut out),
            Err(QdotError::TileShape {
                cols: TILE_COLS + 1,
                outs: TILE_COLS + 1
            }),
            "{name}"
        );
        assert_eq!(
            f(GgmlType::Q4_K, &wrow, &[], k, &mut []),
            Err(QdotError::TileShape { cols: 0, outs: 0 }),
            "{name}"
        );
        assert_eq!(
            f(GgmlType::Q4_K, &wrow, &many[..2], k, &mut out[..3]),
            Err(QdotError::TileShape { cols: 2, outs: 3 }),
            "{name}"
        );
        assert_eq!(
            f(GgmlType::Q5_K, &wrow, &many[..1], 384, &mut out[..1]),
            Err(QdotError::UnalignedK { k: 384, gran: 256 }),
            "{name}"
        );
        assert_eq!(
            f(GgmlType::Q5_K, &wrow[..300], &many[..1], k, &mut out[..1]),
            Err(QdotError::ShortWeightRow {
                have: 300,
                need: 352,
                k
            }),
            "{name}"
        );
        let short = &a[..500];
        assert_eq!(
            f(GgmlType::Q4_K, &wrow, &[&a, short], k, &mut out[..2]),
            Err(QdotError::ShortActivationCol {
                have: 500,
                need: 592,
                k
            }),
            "{name}"
        );
    }
}
