//! GPU gate for the IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS and Q2_K row cores
//! (`bloomery_gpu::iq`) on one card.
//!
//! Weights are the synthetic rows of `bloomery_gpu_gates::iq_rows`
//! (`dequant_ref --synthetic`'s dump for `just gate-1-1`, 64 rows ggml
//! quantized and 16 of random codes, 4096 values each; IQ2_XXS's 80 rows
//! seeded blocks, which the harness does not write): all 80 rows at K = 4096,
//! and the first 79 rows cut to their first nine super-blocks (K = 2304: a
//! partial last lane iteration, three q8 groups, an odd count of f16
//! scales). IQ2_XXS also runs Nemotron 3.5 Bolt's shared expert, seeded
//! blocks at its two shapes: the up, 5,376 rows of K = 4,096, and the down,
//! 4,096 rows of K = 5,376 (21 super-blocks: a partial last lane iteration
//! and an odd count of f16 scales). Activations are eight seeded columns
//! uniform on ±2, quantized by the engine's q8_1 rule on the host and
//! uploaded in the Q4_K permutation, so the gate sees the cores alone.
//!
//! Per format and geometry:
//! 1. **Host rule, bits.** Every output equals the module's host
//!    transcription (`lane_partial_host` per lane, then the butterfly) bit
//!    for bit; a second launch equals the first; an m = 3 launch equals the
//!    first three columns of the m = 8 one.
//! 2. **f32 reference, band.** Against the rows ggml's rule dequantizes
//!    (`gguf::dequant_row`) times the f32 activations, summed in f64: every
//!    output finite and written (none left at the sentinel), its error within
//!    the bound below, and `max|ours − ref| / max|ref|` over each
//!    geometry's outputs under the format's pin.
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

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_iq", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::iq::{IqFormat, IqKernels, IqRows};
    use bloomery_gpu_gates::act_rule::q8_1_codes;
    use bloomery_gpu_gates::iq_rows::{Col, SYNTH_K, dot_host, dot_ref, seeded_blocks, synth_rows};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::dequant_row;

    /// Activation columns per launch.
    const M: usize = 8;
    /// The cut geometry: nine super-blocks, 79 rows.
    const K_CUT: usize = 2304;
    const ROWS_CUT: usize = 79;
    /// What `y` holds before a launch, so an output the kernel skips reads
    /// back as these bits.
    const SENT: f32 = 1.0e30;

    /// Bolt's shared expert: hidden 4,096, width 5,376.
    const BOLT_HIDDEN: usize = 4096;
    const BOLT_SHARED: usize = 5376;
    /// The seeds of the shared expert's two stacks.
    const SEED_SHARED: [u64; 2] = [0xb017_5a2e_0001, 0xb017_5a2e_0002];

    /// PIN(2026-09-24): per format, the larger `max|ours − ref| / max|ref|`
    /// of the two geometries, rounded up: 3.630e-3 (IQ2_XS, K = 4096), 5.788e-3
    /// (IQ3_XXS, 2304), 3.813e-3 (IQ4_XS, 2304), 5.650e-3 (Q2_K, 4096) — the
    /// derivation's 0.4 % for every format.
    /// PIN(2026-10-16): IQ2_XXS, the largest of its four geometries, rounded
    /// up: 4.559e-3 (K = 2304; 4.537e-3 at Bolt's shared up) — the same 0.4 %.
    const REL_PIN: [(IqFormat, f64); 5] = [
        (IqFormat::Iq2Xxs, 4.6e-3),
        (IqFormat::Iq2Xs, 3.7e-3),
        (IqFormat::Iq3Xxs, 5.8e-3),
        (IqFormat::Iq4Xs, 3.9e-3),
        (IqFormat::Q2K, 5.7e-3),
    ];

    const FORMATS: [IqFormat; 5] = [
        IqFormat::Iq2Xxs,
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

    /// One geometry's rows: file bytes per row and the stack's row count.
    struct Rows {
        bytes: Vec<u8>,
        rows: usize,
        k: usize,
    }

    /// The format's geometries: its synthetic rows at K = 4096 and cut to
    /// K = 2304 (module doc), and for IQ2_XXS Bolt's shared expert, up and
    /// down.
    fn geometries(fmt: IqFormat) -> Result<Vec<Rows>, GateError> {
        let all = synth_rows(fmt)?;
        let bb = fmt.block_bytes();
        let row_bytes = SYNTH_K / 256 * bb;
        let keep = K_CUT / 256 * bb;
        let cut = all
            .chunks(row_bytes)
            .take(ROWS_CUT)
            .flat_map(|r| r[..keep].iter().copied())
            .collect();
        let mut g = vec![
            Rows {
                rows: all.len() / row_bytes,
                bytes: all,
                k: SYNTH_K,
            },
            Rows {
                bytes: cut,
                rows: ROWS_CUT,
                k: K_CUT,
            },
        ];
        if fmt == IqFormat::Iq2Xxs {
            for (i, (rows, k)) in [(BOLT_SHARED, BOLT_HIDDEN), (BOLT_HIDDEN, BOLT_SHARED)]
                .into_iter()
                .enumerate()
            {
                g.push(Rows {
                    bytes: seeded_blocks(fmt, rows * k / 256, SEED_SHARED[i]),
                    rows,
                    k,
                });
            }
        }
        Ok(g)
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
        r: &Rows,
    ) -> Result<(bool, f64), GateError> {
        let s = gpu.stream();
        let x = seeded(M * r.k, 7001 + r.k as u32, -2.0, 4.0);
        let cols: Vec<Col> = x.chunks(r.k).map(q8_1_codes).collect();
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
        let written = y
            .iter()
            .chain(&y3)
            .all(|v| v.is_finite() && v.to_bits() != SENT.to_bits());
        let name = fmt.ggml().name().unwrap_or("?");
        println!(
            "{name:8} k={} rows={} m={M}: host rule {same}/{} bit-identical {}; rerun {}; m=3 {}; \
             finite and written {}; f32 ref max|err| {err_max:.3e} max|ref| {ref_max:.3e} rel {:.3e}, \
             worst err/bound {ratio_max:.3e} {}",
            r.k,
            r.rows,
            host.len(),
            verdict(bits),
            verdict(rerun),
            verdict(masked),
            verdict(written),
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
        Ok((
            bits && rerun && masked && written && in_bound,
            err_max / ref_max,
        ))
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let kern = IqKernels::load(gpu.context())?;
        let mut ok = true;
        for fmt in FORMATS {
            let mut rel = 0.0f64;
            for r in geometries(fmt)? {
                let (pass, e) = check(&gpu, &kern, fmt, &r)?;
                ok &= pass;
                rel = rel.max(e);
            }
            let pin = REL_PIN
                .iter()
                .find(|(f, _)| *f == fmt)
                .map(|&(_, p)| p)
                .ok_or("every format has a pin")?;
            let under = rel <= pin;
            ok &= under;
            println!(
                "{:8} max|rel|, the largest of its geometries: {rel:.4e} (pin {pin:.1e}) {}",
                fmt.ggml().name().unwrap_or("?"),
                verdict(under)
            );
        }
        println!("gate_iq: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
