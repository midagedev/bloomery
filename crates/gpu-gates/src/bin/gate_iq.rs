//! GPU gate for the IQ2_XS, IQ3_XXS, IQ4_XS and Q2_K row cores
//! (`bloomery_gpu::iq`) on the 3090.
//!
//! Weights are the synthetic rows `dequant_ref --synthetic` writes for
//! `just gate-1-1` (`$BLOOMERY_DATA/ref-synth/<type>.blocks`: 64 rows ggml
//! quantized, then 16 rows of random codes, 4096 values each): all 80 rows at
//! K = 4096, and the first 79 rows cut to their first nine super-blocks (K =
//! 2304: a partial last lane iteration, three q8 groups, an odd count of
//! f16 scales). Activations are eight seeded columns uniform on ±2, quantized
//! by the engine's q8_1 rule on the host and uploaded in the Q4_K
//! permutation, so the gate sees the cores alone.
//!
//! Per format and K:
//! 1. **Host rule, bits.** Every output equals the module's host
//!    transcription (`lane_partial_host` per lane, then the butterfly) bit
//!    for bit; a second launch equals the first; an m = 3 launch equals the
//!    first three columns of the m = 8 one.
//! 2. **f32 reference, band.** Against the rows ggml's rule dequantizes
//!    (`gguf::dequant_row`) times the f32 activations, summed in f64: every
//!    output's error within the bound below, and `max|ours − ref| / max|ref|`
//!    over each geometry's outputs under the format's pin.
//!
//! The bound of item 2 (the MXFP4 gate's derivation, `gate_dspark_experts`).
//! A value `x` of a 128-value block with scale `d = amax/127` is stored as
//! `q·d` with `|q·d − x| ≤ (d/2)(1 + 256u)`; with exact weights `w` a dot
//! errs by at most `Σ_B (d_B/2)(1 + 256u) Σ_{i∈B} |w_i|` from the
//! activations. The rule's integer sums are exact, so the arithmetic adds
//! one rounding per scale product and one per fused multiply-add on a lane
//! (two per sub-block for Q2_K) and five butterfly adds: at most `γ(2·n_it +
//! 6)·Σ_k |t_k|`, `t_k` the lane terms and `n_it` a lane's sub-blocks. Q2_K's
//! dequant rounds `dl·q − ml` once, so its reference weights carry `u·|w|`
//! more: `+ u·Σ|w_i x_i|`. That is the worst case; the expected error is the
//! random-rounding one, `≈ (d/√12)·‖w‖` against a dot of size `≈
//! rms(x)·‖w‖` — for x uniform on ±2 about 0.4 % of a typical dot whatever
//! the format, since the weights enter both sides alike.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_iq: built without the `gpu` feature; see `just gate-gpu-iq`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_iq", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::iq::{
        IqFormat, IqKernels, IqRows, SUB_VALUES, lane_partial_host, sub_sums_host,
    };
    use bloomery_gpu_gates::rounding::{U, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, data_dir, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{dequant_row, half_to_f32};

    /// Activation columns per launch.
    const M: usize = 8;
    /// Values per full synthetic row.
    const K_FULL: usize = 4096;
    /// The cut geometry: nine super-blocks, 79 rows.
    const K_CUT: usize = 2304;
    const ROWS_CUT: usize = 79;
    /// What `y` holds before a launch, so an output the kernel skips reads
    /// back as these bits.
    const SENT: f32 = 1.0e30;

    /// PIN(2026-09-24): per format, the larger `max|ours − ref| / max|ref|`
    /// of the two geometries, rounded up: 3.630e-3 (IQ2_XS, K = 4096), 5.788e-3
    /// (IQ3_XXS, 2304), 3.813e-3 (IQ4_XS, 2304), 5.650e-3 (Q2_K, 4096) — the
    /// derivation's 0.4 % for every format.
    const REL_PIN: [(IqFormat, f64); 4] = [
        (IqFormat::Iq2Xs, 3.7e-3),
        (IqFormat::Iq3Xxs, 5.8e-3),
        (IqFormat::Iq4Xs, 3.9e-3),
        (IqFormat::Q2K, 5.7e-3),
    ];

    const FORMATS: [IqFormat; 4] = [
        IqFormat::Iq2Xs,
        IqFormat::Iq3Xxs,
        IqFormat::Iq4Xs,
        IqFormat::Q2K,
    ];

    /// Values seeded by an LCG (Numerical Recipes' constants), uniform on
    /// `lo .. lo + span`.
    fn seeded(n: usize, seed: u32, lo: f32, span: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                lo + ((s >> 8) as f32 / (1u32 << 24) as f32) * span
            })
            .collect()
    }

    /// The engine's q8_1 rule of one column on the host: per 128-value block
    /// `d = amax/127` (1 for an all-zero block), `q = round(x/d)` half away
    /// from zero, clamped to ±127.
    fn q8_1_host(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let (mut q, mut d8) = (Vec::with_capacity(x.len()), Vec::new());
        for b in x.chunks(128) {
            let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            d8.push(d);
            q.extend(
                b.iter()
                    .map(|&v| (v / d).round().clamp(-127.0, 127.0) as i8),
            );
        }
        (q, d8)
    }

    /// The q8_1 quantizer's Q4_K slot of value-order word `v`.
    fn q4_slot(v: usize) -> usize {
        256 * (v >> 8) + 32 * (v & 7) + 8 * ((v >> 6) & 3) + ((v >> 3) & 7)
    }

    /// The card's q words and scales of `cols` host-quantized columns.
    fn act_words(cols: &[(Vec<i8>, Vec<f32>)], k: usize) -> (Vec<u32>, Vec<f32>) {
        let col_words = 256 * (k / 256).div_ceil(4);
        let mut q = vec![0u32; cols.len() * col_words];
        let mut d8 = Vec::new();
        for (c, (xq, d)) in cols.iter().enumerate() {
            for v in 0..k / 4 {
                q[c * col_words + q4_slot(v)] =
                    u32::from_le_bytes(std::array::from_fn(|i| xq[4 * v + i] as u8));
            }
            d8.extend_from_slice(d);
        }
        (q, d8)
    }

    /// One row's dot by the host rule: the 32 lane partials, then the
    /// butterfly.
    fn dot_host(fmt: IqFormat, row: &[u8], col: &(Vec<i8>, Vec<f32>)) -> f32 {
        butterfly(std::array::from_fn(|lane| {
            lane_partial_host(fmt, row, &col.0, &col.1, lane)
        }))
    }

    /// `Σ_k |t_k|` of the rule's lane terms for one row and column, and the
    /// largest lane iteration count `n_it`.
    fn term_mag(fmt: IqFormat, row: &[u8], col: &(Vec<i8>, Vec<f32>)) -> (f64, usize) {
        let bb = fmt.block_bytes();
        let codes = col.0.as_chunks::<SUB_VALUES>().0;
        let n_sub = 8 * row.len() / bb;
        let mut mag = 0.0f64;
        for (b, xq) in codes.iter().enumerate().take(n_sub) {
            let blk = &row[bb * (b / 8)..bb * (b / 8 + 1)];
            let (i, m) = sub_sums_host(fmt, blk, b % 8, xq);
            let dx = f64::from(col.1[b / 4]);
            let h = |at: usize| f64::from(half_to_f32(u16::from_le_bytes([blk[at], blk[at + 1]])));
            mag += match fmt {
                IqFormat::Iq2Xs => (h(0) / 8.0 * dx * f64::from(i)).abs(),
                IqFormat::Iq3Xxs => (h(0) / 4.0 * dx * f64::from(i)).abs(),
                IqFormat::Iq4Xs => (h(0) * dx * f64::from(i)).abs(),
                IqFormat::Q2K => {
                    (h(80) * dx * f64::from(i)).abs() + (h(82) * dx * f64::from(m)).abs()
                }
            };
        }
        (mag, n_sub.div_ceil(32))
    }

    /// The exact dot of a dequantized row with f32 activations, and the
    /// module doc's bound on our error.
    fn dot_ref(
        fmt: IqFormat,
        row: &[u8],
        w: &[f32],
        x: &[f32],
        col: &(Vec<i8>, Vec<f32>),
    ) -> (f64, f64) {
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

    /// One geometry's rows: file bytes per row and the stack's row count.
    struct Rows {
        bytes: Vec<u8>,
        rows: usize,
        k: usize,
    }

    /// The format's synthetic rows, all of them at K = 4096, or the cut.
    fn load_rows(fmt: IqFormat, cut: bool) -> Result<Rows, GateError> {
        let name = fmt.ggml().name().ok_or("an IqFormat names its type")?;
        let path = data_dir().join("ref-synth").join(format!("{name}.blocks"));
        let all = std::fs::read(&path)
            .map_err(|e| format!("{}: {e} (run `just gate-1-1` first)", path.display()))?;
        let bb = fmt.block_bytes();
        let row_bytes = K_FULL / 256 * bb;
        if all.is_empty() || !all.len().is_multiple_of(row_bytes) {
            return Err(format!(
                "{}: {} bytes is not whole rows of {row_bytes}",
                path.display(),
                all.len()
            )
            .into());
        }
        if !cut {
            let rows = all.len() / row_bytes;
            return Ok(Rows {
                bytes: all,
                rows,
                k: K_FULL,
            });
        }
        let keep = K_CUT / 256 * bb;
        let bytes = all
            .chunks(row_bytes)
            .take(ROWS_CUT)
            .flat_map(|r| r[..keep].iter().copied())
            .collect();
        Ok(Rows {
            bytes,
            rows: ROWS_CUT,
            k: K_CUT,
        })
    }

    /// One launch of `m` columns into a sentinel-filled `y`, read back.
    fn launch(
        gpu: &Gpu,
        kern: &IqKernels,
        w: &IqRows,
        q: &DeviceBuffer<u32>,
        d8: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<Vec<f32>, GateError> {
        let s = gpu.stream();
        let mut y = DeviceBuffer::from_host(s, &vec![SENT; m * w.rows()])?;
        kern.enqueue_rows(s, w, q, d8, m, &mut y)?;
        Ok(y.to_host_vec(s)?)
    }

    /// The format's checks at one geometry, and its `max|ours − ref| /
    /// max|ref|`.
    fn check(
        gpu: &Gpu,
        kern: &IqKernels,
        fmt: IqFormat,
        cut: bool,
    ) -> Result<(bool, f64), GateError> {
        let r = load_rows(fmt, cut)?;
        let s = gpu.stream();
        let x = seeded(M * r.k, 7001 + r.k as u32, -2.0, 4.0);
        let cols: Vec<_> = x.chunks(r.k).map(q8_1_host).collect();
        let (qw, d8w) = act_words(&cols, r.k);
        let (q, d8) = (
            DeviceBuffer::from_host(s, &qw)?,
            DeviceBuffer::from_host(s, &d8w)?,
        );
        let w = IqRows::upload(s, fmt, &r.bytes, r.rows, r.k)?;

        let y = launch(gpu, kern, &w, &q, &d8, M)?;
        let again = launch(gpu, kern, &w, &q, &d8, M)?;
        let y3 = launch(gpu, kern, &w, &q, &d8, 3)?;

        let row_bytes = r.bytes.len() / r.rows;
        let mut host = vec![0.0f32; M * r.rows];
        let mut first_miss = None;
        let (mut err_max, mut ref_max, mut ratio_max) = (0.0f64, 0.0f64, 0.0f64);
        let mut w_f32 = vec![0.0f32; r.k];
        for row in 0..r.rows {
            let rb = &r.bytes[row * row_bytes..(row + 1) * row_bytes];
            dequant_row(fmt.ggml(), rb, &mut w_f32)?;
            for (c, col) in cols.iter().enumerate() {
                let at = c * r.rows + row;
                host[at] = dot_host(fmt, rb, col);
                if host[at].to_bits() != y[at].to_bits() && first_miss.is_none() {
                    first_miss = Some((row, c, host[at], y[at]));
                }
                let (exact, bound) = dot_ref(fmt, rb, &w_f32, &x[c * r.k..(c + 1) * r.k], col);
                let err = (f64::from(y[at]) - exact).abs();
                err_max = err_max.max(err);
                ref_max = ref_max.max(exact.abs());
                ratio_max = ratio_max.max(err / bound);
            }
        }
        let same = host
            .iter()
            .zip(&y)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        let bits = first_miss.is_none();
        let rerun = bits_equal(&y, &again);
        let masked = bits_equal(&y[..3 * r.rows], &y3);
        let in_bound = ratio_max <= 1.0;
        let name = fmt.ggml().name().unwrap_or("?");
        println!(
            "{name:8} k={} rows={} m={M}: host rule {same}/{} bit-identical {}; rerun {}; m=3 {}; \
             f32 ref max|err| {err_max:.3e} max|ref| {ref_max:.3e} rel {:.3e}, worst err/bound {ratio_max:.3e} {}",
            r.k,
            r.rows,
            host.len(),
            verdict(bits),
            verdict(rerun),
            verdict(masked),
            err_max / ref_max,
            verdict(in_bound),
        );
        if let Some((row, c, h, o)) = first_miss {
            println!(
                "  first mismatch: row {row} column {c}: host {h:e} ({:#010x}) card {o:e} ({:#010x})",
                h.to_bits(),
                o.to_bits()
            );
        }
        Ok((bits && rerun && masked && in_bound, err_max / ref_max))
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let kern = IqKernels::load(gpu.context())?;
        let mut ok = true;
        for fmt in FORMATS {
            let mut rel = 0.0f64;
            for cut in [false, true] {
                let (pass, r) = check(&gpu, &kern, fmt, cut)?;
                ok &= pass;
                rel = rel.max(r);
            }
            let pin = REL_PIN
                .iter()
                .find(|(f, _)| *f == fmt)
                .map(|&(_, p)| p)
                .ok_or("every format has a pin")?;
            let under = rel <= pin;
            ok &= under;
            println!(
                "{:8} max|rel|, the larger of the two geometries: {rel:.4e} (pin {pin:.1e}) {}",
                fmt.ggml().name().unwrap_or("?"),
                verdict(under)
            );
        }
        println!("gate_iq: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
