//! Synthetic weight files and activation columns for dense-format gates.

use bloomery_gpu::Gpu;
use bloomery_gpu::gemm::{Gemm32Kernels, GemmAct32, GemmInput, GemmKernels, GemmRoute};
use bloomery_gpu::site::{self, SiteTy, SiteWeight, WideIn, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu_gates::{GateError, GateRng, bits_equal, par_map, verdict};
use cuda_core::DeviceBuffer;
use gguf::quant::{GgmlType, dequant_row};

pub(crate) type Decl = (String, Vec<u64>, GgmlType, Vec<u8>);
type Columns = (Vec<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>);

/// One launch's outputs against the host: output `at` is row `r` of
/// column `c` by `index(at)`, `rule(r, c)` the bits the kernel must give,
/// `w` the host rows for the band. Prints the clauses `written`, `band`
/// and `bits` and returns whether all hold.
pub(crate) fn check(
    (ty, clause): (&str, &str),
    tag: &str,
    (y, x, k): (&[f32], &[f32], usize),
    w: &[Vec<f32>],
    index: &(dyn Fn(usize) -> (usize, usize) + Sync),
    rule: &(dyn Fn(usize, usize) -> f32 + Sync),
) -> bool {
    use bloomery_gpu_gates::rounding::{U64, gamma};
    use bloomery_gpu_gates::verdict;

    let per: Vec<(bool, f64, f64)> = par_map(y.len(), |at| {
        let (r, c) = index(at);
        let (sum, mag) =
            w[r].iter()
                .zip(&x[c * k..][..k])
                .fold((0.0f64, 0.0f64), |(s, m), (&a, &b)| {
                    let t = f64::from(a) * f64::from(b);
                    (s + t, m + t.abs())
                });
        (
            y[at].to_bits() == rule(r, c).to_bits(),
            (f64::from(y[at]) - sum).abs(),
            (gamma(k / 32 + 5) + k as f64 * U64) * mag,
        )
    });
    let written = y.iter().all(|v| v.is_finite());
    let bits_n = per.iter().filter(|p| p.0).count();
    let band_ok = per.iter().all(|p| p.1 <= p.2);
    let worst = per
        .iter()
        .map(|p| if p.2 > 0.0 { p.1 / p.2 } else { p.1 })
        .fold(0.0f64, f64::max);
    println!("{ty} clause={clause}.written {tag} {}", verdict(written));
    println!(
        "{ty} clause={clause}.band {tag} worst_err_over_band={worst:.3e} {}",
        verdict(band_ok)
    );
    println!(
        "{ty} clause={clause}.bits {tag} equal={bits_n}/{} {}",
        per.len(),
        verdict(bits_n == per.len())
    );
    written && band_ok && bits_n == per.len()
}

/// `rows` Q8_0 rows of `k` values drawn from `seed`: f16 scales of
/// `2^-8` to `2^1`, codes uniform over all 256 bytes.
pub(crate) fn q8_0_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = GateRng(seed);
    let mut out = Vec::with_capacity(rows * k / 32 * 34);
    for _ in 0..rows * k / 32 {
        let bits = ((7 + rng.below(9)) << 10 | rng.below(1024)) as u16;
        out.extend(bits.to_le_bytes());
        out.extend((0..32).map(|_| rng.below(256) as u8));
    }
    out
}

/// A weight of type `ty` of `rows` rows of `k` as a tensor of a file.
pub(crate) fn decl(name: &str, ty: GgmlType, (rows, k): (usize, usize), bytes: Vec<u8>) -> Decl {
    (name.to_string(), vec![k as u64, rows as u64], ty, bytes)
}

/// The host rows of a file tensor's `bytes`: `rows` rows of `k` values
/// of type `ty`.
pub(crate) fn host_rows(ty: GgmlType, bytes: &[u8], rows: usize, k: usize) -> Vec<Vec<f32>> {
    let row_bytes = bytes.len() / rows;
    par_map(rows, |r| {
        let mut v = vec![0.0f32; k];
        dequant_row(ty, &bytes[r * row_bytes..][..row_bytes], &mut v)
            .unwrap_or_else(|e| panic!("row {r}: {e}"));
        v
    })
}

/// `m` columns of `k` activations uniform on ±2 from `seed`, multiplied
/// by `scale`, on the card, and an output of `m * rows` NaN sentinels.
pub(crate) fn columns(
    gpu: &Gpu,
    (m, k, rows): (usize, usize, usize),
    seed: u64,
    scale: f32,
) -> Result<Columns, GateError> {
    let mut rng = GateRng(seed);
    let x: Vec<f32> = (0..m * k).map(|_| rng.uni(-2.0, 2.0) * scale).collect();
    let x_dev = DeviceBuffer::from_host(gpu.stream(), &x)?;
    let y_dev = DeviceBuffer::from_host(gpu.stream(), &vec![f32::NAN; m * rows])?;
    Ok((x, x_dev, y_dev))
}

/// A compute capability 12 device's dense prompt product through `site::gemm` (`src`, `rows` rows,
/// over `route` filled for the columns of `x`) keeps each column's bits
/// whatever another column of its tile holds: with a subnormal first value in
/// column 1, which the tensor cores refuse, every other column's outputs equal
/// the run without it. Prints the clause `mma.neighbours`.
pub(crate) fn mma_neighbours(
    gpu: &Gpu,
    kernels: &WideKernels<'_>,
    (src, route): ((SiteTy, &Weights, SiteWeight<'_>), &GemmRoute),
    (x, k, rows): (&[f32], usize, usize),
    (ty, tag): (&str, &str),
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let m = x.len() / k;
    let run = |x: &[f32]| -> Result<Vec<f32>, GateError> {
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; m * rows])?;
        let x_in = WideIn {
            q128: None,
            q32: None,
            f32: Some(&x_dev),
        };
        let table = (route, GemmInput::PerSlot);
        site::gemm(gpu, kernels, src, rows, &x_in, table, m, &mut y)?;
        Ok(y.to_host_vec(stream)?)
    };
    let plain = run(x)?;
    let mut odd = x.to_vec();
    odd[k] = f32::from_bits(0x200);
    let beside = run(&odd)?;
    let kept = (0..m)
        .filter(|&c| c != 1)
        .all(|c| bits_equal(&plain[c * rows..][..rows], &beside[c * rows..][..rows]));
    println!(
        "{ty} clause=mma.neighbours {tag} subnormal_column=1 other_columns_bits={kept} {}",
        verdict(kept)
    );
    Ok(kept)
}

/// The dense calls by name over the resident weight `name`, `rows` rows of
/// `k` values, of type `ty`: each clause holds one launch's outputs bit for
/// bit to another's, so it holds at any type the site launches by name.
/// - `by_name.cols`: `site::resident_gemv_cols` at three columns, each
///   column the one-column `site::resident_gemv` of it;
/// - `by_name.wide`: `site::resident_gemm_rows` over rows `at .. at + n`
///   alone on the one-expert table `route` filled for `wide` slots,
///   `site::resident_gemm`'s at those rows;
/// - `by_name.past`: rows past the weight refused by name on the same call.
///
/// PIN(2026-10-31): `by_name.rows` (`site::resident_gemv_rows` over a part's rows) is removed with
/// its function: no body projects rows of a joined weight on the gemv arm (Qwen3.8's joined β·α
/// projection runs whole); `by_name.wide` keeps the part's rows and `by_name.past` its refusal.
pub(crate) fn by_name(
    gpu: &Gpu,
    (g32, gemm, route, wide): (&Gemm32Kernels, &GemmKernels, &GemmRoute, usize),
    (ty, w, name): (&str, &Weights, &str),
    (rows, k): (usize, usize),
    (at, n): (usize, usize),
    seed: u64,
) -> Result<bool, GateError> {
    const M: usize = 3;
    let stream = gpu.stream();
    let bits = |y: &[f32], want: &[f32]| y.iter().all(|v| v.is_finite()) && bits_equal(y, want);
    let part = |y: &[f32], cols: usize| -> Vec<f32> {
        (0..cols)
            .flat_map(|c| y[c * rows + at..][..n].to_vec())
            .collect()
    };
    let mut ok = true;
    let mut line = |clause: &str, shown: &str, pass: bool| {
        println!(
            "{ty} clause=by_name.{clause} tensor={name} {shown} {}",
            verdict(pass)
        );
        ok &= pass;
    };

    let (x, x_dev, mut whole) = columns(gpu, (M, k, rows), seed, 1.0)?;
    site::resident_gemv_cols(gpu, Some(g32), (w, name), &x_dev, M, &mut whole)?;
    let mut one = Vec::with_capacity(M * rows);
    for c in 0..M {
        let xc = DeviceBuffer::from_host(stream, &x[c * k..][..k])?;
        let mut yc = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows])?;
        site::resident_gemv(gpu, w, name, &xc, &mut yc)?;
        stream.synchronize()?;
        one.extend(yc.to_host_vec(stream)?);
    }
    let whole = whole.to_host_vec(stream)?;
    line("cols", &format!("m={M}"), bits(&whole, &one));

    let kernels = WideKernels {
        gemm,
        g32: Some(g32),
    };
    let (_, xw, mut gw) = columns(gpu, (wide, k, rows), seed ^ 0x77, 1.0)?;
    let mut act = GemmAct32::new(stream, wide, k)?;
    let sink = gpu.unlabelled_sink();
    site::resident_quantize(
        gpu,
        &kernels,
        (w, &[name]),
        (&xw, wide),
        (None, &mut act),
        sink,
    )?;
    let x_in = WideIn {
        q128: None,
        q32: Some(&act),
        f32: Some(&xw),
    };
    site::resident_gemm(gpu, &kernels, (w, name), &x_in, route, wide, &mut gw)?;
    let mut pw = DeviceBuffer::from_host(stream, &vec![f32::NAN; wide * n])?;
    let src = (w, name, (at, n));
    site::resident_gemm_rows(gpu, &kernels, src, &x_in, route, wide, &mut pw)?;
    stream.synchronize()?;
    let (gw, pw) = (gw.to_host_vec(stream)?, pw.to_host_vec(stream)?);
    line(
        "wide",
        &format!("rows={at}+{n} m={wide}"),
        bits(&pw, &part(&gw, wide)),
    );

    let past = (w, name, (at, rows - at + 1));
    let mut py = DeviceBuffer::from_host(stream, &vec![f32::NAN; wide * (rows - at + 1)])?;
    let got = site::resident_gemm_rows(gpu, &kernels, past, &x_in, route, wide, &mut py);
    let refused = got
        .as_ref()
        .is_err_and(|e| e.to_string().contains(name) && e.to_string().contains("rows"));
    let shown = match got {
        Ok(()) => "accepted".to_owned(),
        Err(e) => format!("{:?}", e.to_string()),
    };
    line(
        "past",
        &format!("rows={at}+{}: {shown}", rows - at + 1),
        refused,
    );
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn numeric_contract() {
        let w = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x = [1.0, 2.0, 4.0, 8.0];
        let rule = |r: usize, c: usize| w[r][0] * x[c * 2] + w[r][1] * x[c * 2 + 1];
        let layouts: [fn(usize) -> (usize, usize); 3] = [
            |at| (at % 2, at / 2),
            |at| (at / 2, at % 2),
            |at| (1 - at / 2, at % 2),
        ];
        for index in layouts {
            let y: Vec<f32> = (0..4)
                .map(|at| {
                    let (r, c) = index(at);
                    rule(r, c)
                })
                .collect();
            assert!(check(
                ("dense", "test"),
                "layout",
                (&y, &x, 2),
                &w,
                &index,
                &rule
            ));
            assert!(!check(
                ("dense", "test"),
                "bits",
                (&y, &x, 2),
                &w,
                &index,
                &|r, c| -rule(r, c)
            ));
            let mut bad = y.clone();
            bad[0] += 1.0;
            assert!(!check(
                ("dense", "test"),
                "band",
                (&bad, &x, 2),
                &w,
                &index,
                &|r, c| {
                    if (r, c) == index(0) {
                        rule(r, c) + 1.0
                    } else {
                        rule(r, c)
                    }
                }
            ));
            for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let bad = vec![v; 4];
                assert!(!check(
                    ("dense", "test"),
                    "finite",
                    (&bad, &x, 2),
                    &w,
                    &index,
                    &|_, _| v
                ));
            }
        }
    }
}
