//! The host K-quant rules (`gguf::kquant`) against ggml's own quantizer: `iq_quant_ref` (`tools/ref/iq_quant_ref.cpp`)
//! linked to the K3.1 quantization tree's libggml-base, the library llama-quantize calls. For each of Q4_K, Q5_K,
//! Q6_K and Q8_0 and for each of a set of seeded row families, every row quantized with an importance row per expert
//! (ggml's `_impl` rules) and without (its `_ref` rules) must equal the oracle's bytes. Run by
//! `just gate-quantize-oracle` (the box, no card).

mod common;

use common::{Rng, f32_bytes, oracle, scratch};
use gguf::kquant::{KType, quantize_row};
use gguf::quant::GgmlType;

struct Family {
    name: &'static str,
    n_per_row: usize,
    rows: usize,
    x: Vec<f32>,
}

fn families(rng: &mut Rng) -> Vec<Family> {
    let (n, rows) = (2048usize, 24usize);
    let fill = |rng: &mut Rng, f: &mut dyn FnMut(&mut Rng, usize) -> f32| -> Vec<f32> {
        (0..n * rows).map(|i| f(rng, i)).collect()
    };
    let mut out = Vec::new();
    let mut add = |name, n_per_row, rows, x| {
        out.push(Family {
            name,
            n_per_row,
            rows,
            x,
        })
    };
    add("uniform", n, rows, fill(rng, &mut |r, _| r.sym()));
    add(
        "sum-of-four",
        n,
        rows,
        fill(rng, &mut |r, _| {
            (r.sym() + r.sym() + r.sym() + r.sym()) * 0.5
        }),
    );
    add(
        "zero-blocks-and-small-integers",
        n,
        rows,
        fill(rng, &mut |r, i| {
            if (i / 32) % 3 == 0 {
                0.0
            } else {
                (r.below(7) as f32) - 3.0
            }
        }),
    );
    add(
        "tiny",
        n,
        rows,
        fill(rng, &mut |r, i| {
            if (i / 32) % 2 == 0 {
                r.sym() * 1e-14
            } else {
                r.sym()
            }
        }),
    );
    add(
        "one-large-in-eight",
        n,
        rows,
        fill(rng, &mut |r, i| {
            if i % 8 == 0 {
                r.sym() * 40.0
            } else {
                r.sym() * 0.1
            }
        }),
    );
    add(
        "three-decimals-ties",
        n,
        rows,
        fill(rng, &mut |r, _| (r.sym() * 1000.0).round() / 1000.0),
    );
    add(
        "all-positive",
        n,
        rows,
        fill(rng, &mut |r, _| 0.05 + r.unit()),
    );
    add(
        "all-negative",
        n,
        rows,
        fill(rng, &mut |r, _| -0.05 - r.unit()),
    );
    add(
        "constant-blocks",
        n,
        rows,
        fill(rng, &mut |r, i| {
            if (i / 32) % 2 == 0 { 0.25 } else { r.sym() }
        }),
    );
    add(
        "mixed-magnitude",
        n,
        rows,
        fill(rng, &mut |r, i| {
            r.sym() * 10f32.powi((i / 256 % 7) as i32 - 3)
        }),
    );
    add(
        "sixteenths",
        n,
        rows,
        fill(rng, &mut |r, _| (r.below(33) as f32 - 16.0) / 16.0),
    );
    let wide = 33792;
    add(
        "ffn-width",
        wide,
        4,
        (0..wide * 4).map(|_| rng.sym() * 0.05).collect(),
    );
    out
}

/// An importance row per expert: positive, with a zero in every 13 (a column the calibration never reached) and a
/// dead block in the second expert.
fn importance(rng: &mut Rng, n_per_row: usize, n_exp: usize) -> Vec<f32> {
    (0..n_per_row * n_exp)
        .map(|i| {
            if i % 13 == 7 || (i / n_per_row == 1 && (i % n_per_row) / 32 == 3) {
                0.0
            } else {
                rng.unit() * rng.unit() * 3.0 + 1e-3
            }
        })
        .collect()
}

#[test]
#[ignore = "hardware gate: the box, with the oracle built (just build-iq-quant-ref)"]
fn hw_kquant_rows_equal_ggml_with_and_without_importance() {
    let dir = scratch("kquant");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let fams = families(&mut rng);
    let mut failures = 0usize;
    for kt in [KType::Q4K, KType::Q5K, KType::Q6K, KType::Q8_0] {
        let (mut rows_total, mut differing) = ([0usize; 2], [0usize; 2]);
        for f in &fams {
            let per = f.rows.div_ceil(2);
            let n_exp = f.rows.div_ceil(per);
            let qw = importance(&mut rng, f.n_per_row, n_exp);
            for (k, w) in [(0usize, Some(&qw[..])), (1, None)] {
                let want = oracle(
                    &dir,
                    kt.ggml(),
                    GgmlType::F32,
                    &f32_bytes(&f.x),
                    f.n_per_row,
                    per,
                    w,
                );
                let rb = f.n_per_row / kt.block_values() * kt.block_bytes();
                assert_eq!(want.len(), f.rows * rb, "{} {kt:?}", f.name);
                for r in 0..f.rows {
                    let x = &f.x[r * f.n_per_row..(r + 1) * f.n_per_row];
                    let e = r / per;
                    let wr = w.map(|q| &q[e * f.n_per_row..(e + 1) * f.n_per_row]);
                    let got = quantize_row(kt, x, wr).unwrap();
                    rows_total[k] += 1;
                    if got != want[r * rb..(r + 1) * rb] {
                        differing[k] += 1;
                        if differing[k] <= 3 {
                            let p = got
                                .iter()
                                .zip(&want[r * rb..(r + 1) * rb])
                                .position(|(a, b)| a != b);
                            println!(
                                "  {kt:?} {} row {r} imatrix={}: first byte differing {p:?}",
                                f.name,
                                w.is_some()
                            );
                        }
                    }
                }
            }
        }
        println!(
            "{kt:?}: with imatrix {} rows compared, {} differing; without {} rows compared, {} differing",
            rows_total[0], differing[0], rows_total[1], differing[1]
        );
        failures += differing[0] + differing[1];
    }
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(failures, 0, "rows differ from ggml's bytes");
}

/// The host K-quant rate on one thread, for `--plan`'s derived wall: weights per second per type, with importance.
#[test]
#[ignore = "hardware gate: prints the rate the plan's derived term is carried from"]
fn hw_kquant_rate_one_thread() {
    let mut rng = Rng(7);
    let n = 7168usize;
    let x: Vec<f32> = (0..n * 64).map(|_| rng.sym() * 0.05).collect();
    let qw: Vec<f32> = (0..n).map(|_| rng.unit() + 0.01).collect();
    for kt in [KType::Q4K, KType::Q5K, KType::Q6K, KType::Q8_0] {
        let t0 = std::time::Instant::now();
        for r in 0..64 {
            std::hint::black_box(
                quantize_row(
                    kt,
                    &x[r * n..(r + 1) * n - n % kt.block_values()],
                    Some(&qw[..n - n % kt.block_values()]),
                )
                .unwrap(),
            );
        }
        let s = t0.elapsed().as_secs_f64();
        println!(
            "rate {kt:?}: {:.3e} weights/s on one thread ({} weights in {s:.2} s)",
            (64 * n) as f64 / s,
            64 * n
        );
    }
}
