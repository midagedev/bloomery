use super::{KQuantError, KType, quantize_row};
use crate::quant::dequant_row;

/// A deterministic row: a slow wave plus a hash-like ripple, no two blocks alike.
fn row(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            let ripple = ((h >> 8) & 0xffff) as f32 / 65_536.0 - 0.5;
            ((i as f32) * 0.013 + seed as f32).sin() * 0.7 + ripple * 0.3
        })
        .collect()
}

const ALL: [KType; 4] = [KType::Q4K, KType::Q5K, KType::Q6K, KType::Q8_0];

#[test]
fn a_row_is_whole_blocks_of_the_types_size_and_reads_back_close() {
    let x = row(512, 3);
    let qw: Vec<f32> = x.iter().map(|v| v * v + 0.01).collect();
    for kt in ALL {
        for w in [None, Some(&qw[..])] {
            let q = quantize_row(kt, &x, w).unwrap();
            assert_eq!(
                q.len(),
                512 / kt.block_values() * kt.block_bytes(),
                "{kt:?}"
            );
            let mut back = vec![0.0f32; 512];
            dequant_row(kt.ggml(), &q, &mut back).unwrap();
            let err = x
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let bound = match kt {
                KType::Q4K => 0.12,
                KType::Q5K => 0.06,
                KType::Q6K | KType::Q8_0 => 0.02,
            };
            assert!(err < bound, "{kt:?} qw={}: max error {err}", w.is_some());
        }
    }
}

#[test]
fn a_zero_row_is_zero_bytes_and_the_same_with_or_without_importance() {
    let x = vec![0.0f32; 256];
    let qw = vec![1.0f32; 256];
    for kt in [KType::Q4K, KType::Q5K, KType::Q6K, KType::Q8_0] {
        let a = quantize_row(kt, &x, None).unwrap();
        let b = quantize_row(kt, &x, Some(&qw)).unwrap();
        assert!(a.iter().all(|&v| v == 0), "{kt:?}");
        assert_eq!(a, b, "{kt:?}");
    }
}

#[test]
fn q8_0_ignores_the_importance_the_k_types_do_not() {
    let x = row(256, 9);
    let qw: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32).collect();
    assert_eq!(
        quantize_row(KType::Q8_0, &x, None).unwrap(),
        quantize_row(KType::Q8_0, &x, Some(&qw)).unwrap()
    );
    for kt in [KType::Q4K, KType::Q5K, KType::Q6K] {
        assert_ne!(
            quantize_row(kt, &x, None).unwrap(),
            quantize_row(kt, &x, Some(&qw)).unwrap(),
            "{kt:?}"
        );
    }
}

#[test]
fn refusals_are_named() {
    let x = row(256, 1);
    assert!(matches!(
        quantize_row(KType::Q4K, &x[..255], None),
        Err(KQuantError::RowLength { n: 255, block: 256 })
    ));
    assert!(matches!(
        quantize_row(KType::Q8_0, &x[..48], None),
        Err(KQuantError::RowLength { n: 48, block: 32 })
    ));
    assert!(matches!(
        quantize_row(KType::Q6K, &x, Some(&x[..10])),
        Err(KQuantError::Weights { got: 10, n: 256 })
    ));
    let mut bad = x.clone();
    bad[17] = f32::NAN;
    let e = quantize_row(KType::Q5K, &bad, None).unwrap_err();
    assert!(e.to_string().contains("value 17"), "{e}");
    let mut w = vec![1.0f32; 256];
    w[5] = f32::INFINITY;
    assert!(matches!(
        quantize_row(KType::Q4K, &x, Some(&w)),
        Err(KQuantError::WeightNotFinite { index: 5, .. })
    ));
}
