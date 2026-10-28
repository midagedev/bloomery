//! GPU gate for Qwen3.8's gated-residual hyper-connections on the card
//! (`bloomery_gpu::hc_gated`): the mix (norm, down, up and mean), the combine
//! and the head's mix, against the host rules of `runtime::hc_gated`.
//! Synthetic inputs at the file's shape (4 streams of 2560, rank 320), no
//! model file: Q8_0 blocks of random codes (−128 included) with a positive
//! normal f16 scale, F32 `γ` near 1 and F32 inject rows.
//!
//! 1. band: an inject site over two columns and the head over one, `xn`,
//!    `lo`, the combine weights and `mixed` each within its pinned band
//!    (`XN_BAND`, `LO_BAND`, `MIXED_BAND`) of `mix_ref`, the f64 rule.
//! 2. stages: on the inject site, every kernel's output is its card rule
//!    (`runtime::hc_gated::card`) applied to the card's readback of its
//!    inputs, bit for bit: `xn` the norm of the streams; the down partials
//!    the q8f32 gemv (`q8_0_gemv_mcol`) of the `[rank·4 × 2560]` view, and
//!    the rule; `lo` the silu of the partials in stream order; the inject
//!    partials; the weights; `mixed` the up rows' sigmoid gate and mean over
//!    the card's `lo` and `xn`, and again with the up rows through
//!    `q8_0_gemv_mcol`; then the head after the site: its `xn` and `mixed`
//!    the rules', the site's weights left in the scratch.
//! 3. cols: an eight-column mix is each column's one-column mix bit for bit
//!    in every output (streams, `xn`, partials, `lo`, weights, `mixed`), and
//!    a rerun is bit-identical.
//! 4. combine: a mix after `Before::Combine` leaves the streams at the rule's
//!    `fma(wgt, y, res)` over the previous site's weights, bit for bit;
//!    `Before::Init` leaves every stream the embedding; the combine alone is
//!    the same rule.
//! 5. refuse: a shape no instance serves (3 streams; rank 256), a hidden
//!    width off the 256 grid, 0 and 9 columns, the up's planes in the down's
//!    place, an inject of the wrong width and a file type other than F32 /
//!    Q8_0 or BF16, each refused by name.
//! 6. fault: a NaN in one stream of one column raises the site
//!    (`hc_gated::SITE`, unlabelled) and leaves the other column bit for bit
//!    the clean launch's, the NaN column's `mixed` all NaN; an infinity in
//!    the combine's `y` raises it too; clean launches raise nothing.
//! 7. shape: the entries compile with no local depot.
//! 8. wide: the wide arm (`HcWideKernels`, a ubatch's mix over any column
//!    count): its norm and combine the capped entries' bits at three
//!    columns, the combine alone from BF16 streams into F32 ones the rule
//!    over the widened values; over BF16 streams (a ubatch walk's) at 3 and
//!    37 columns, plain, combined and initialized, the streams the F32
//!    step's rounded to nearest even and every output the F32 streams' mix
//!    over them, bit for bit; over 37 columns its glue — the
//!    bottleneck, the weights, the mixed values — the card rule on the
//!    card's readback of its inputs, bit for bit; each column's `xn`, `lo`,
//!    weights and `mixed` within the bands the error model derives for the q8
//!    step per 32 values (`wide_lo_band`, `wide_g_rel`) of `mix_ref`; four of
//!    the columns each its one-column mix bit for bit and a rerun
//!    bit-identical; over 8 columns the same bits and fault eager and in a
//!    graph; a NaN in one column raises the quantizer's site and the mix's
//!    and leaves the others clean;
//!    38 columns on a 37-column scratch refused by name, and — before any
//!    launch, the streams as they went in — planes of another shape and a
//!    one-expert table filled for fewer slots than the mix's columns.
//! 9. form: on a dense MMA card the BF16 site's wide mix over 37 columns,
//!    asked for the BF16 form of `mixed` and, on a native block-scaled
//!    build, its MXFP8 form, into a scratch whose every column an earlier
//!    form refused, leaves `mixed` and the fault the mix's without a form,
//!    bit for bit, and a next product over the form the one over the
//!    quantizer's form of the same rows, also with a NaN in one column,
//!    with one column of subnormal values (the form refuses that column,
//!    so the next product replays it exact) and after a graph replay over
//!    another mix's form; on a native build the other form of the rows
//!    from before the mix is left unread; over 3 columns, where the up and the mix run apart, a
//!    form of the rows from before the mix is left unread; an output
//!    scratch of another K is refused by name before any launch, the
//!    streams as they went in.
//! 10. hidden: at hidden 4352, past one norm chunk a thread (4096 values),
//!     a Q8_0 site's `xn` the rule's bits in the capped arm, plain and after
//!     a combine (the streams the rule's too), and in the wide arm.
//!
//! A BF16 site (down and up in their words, `HcMatrix::Bf16`, values on
//! the Q8_0 site's ranges) runs clauses 1, 2, 3 and 8 again: its down and
//! up rows dotted in `card::f32_dot`'s order over the widened values, the
//! dense gemv beside them `bf16_gemv`, and its wide arm the BF16 prompt
//! product with no quantizer, so a NaN raises the mix's site alone and a
//! subnormal column leaves the other columns' bits unchanged. On a compute capability 12 device that
//! product runs on BF16 tensor cores over `xn` kept as BF16 values: the
//! wide norm's `xn` is the capped entry's rounded to nearest even, within
//! `XN_BAND` plus BF16's unit roundoff of the rule, the glue's `mixed` the
//! rule on those values; the down and up within the BF16 operand bands
//! (`check_mma`), the inject rows in the down's product
//! (`hc_gated_down_mma_4`, dots `[m][324]`) within the same band and its
//! weights within the F32 band plus that bound through `2·σ(·/4)`, a
//! subnormal column's down dots the F32 tile's bits over its widened `xn`,
//! and in a range fixture each cell the MMA product when its row and column
//! are in range and the exact replay when either is not. Maximum-finite BF16 up weights that overflow the up
//! projection raise the site and leave NaN mixed outputs in both capped and
//! wide arms, with a finite bottleneck.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hc_gated", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmAct32, GemmKernels, GemmRoute, MmaForm};
    use bloomery_gpu::hc_gated::{
        Before, HcGatedKernels, HcMatrix, HcScratch, HcWideKernels, HcWideScratch, MixArgs, SITE,
        SiteWeights, Streams, WideMixArgs, check_types,
    };
    use bloomery_gpu::linear::{sigmoid, silu};
    use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvMcolArgs, Q8F32Kernels};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE};
    use bloomery_gpu_gates::rounding::{U_BF16, bf16_dot_rel, bf16_round, check_mma, q8_32_rel};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, max_rel_err, max_ulps, no_local_depot, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{GgmlType, Q8Block, bf16_bits_to_f32, fp32_to_bf16_bits, half_to_f32};
    use runtime::hc_gated::card::{self, Act};
    use runtime::hc_gated::{
        Geometry, HcRefused, LO_BAND, MAX_COLS, MIXED_BAND, MixWeights, XN_BAND, combine, mix_ref,
    };

    const S: usize = 4;
    const R: usize = 320;
    const D: usize = 2560;
    const WIDE: usize = S * D;
    /// The file's `rms_eps` (`crates/model/tests/qwen4exp_meta.rs` pins its bits,
    /// `0x358637bd`).
    const EPS: f32 = 1e-6;
    const ACT: Act = Act { silu, sigmoid };

    /// A fixed xorshift64 stream.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// `rows` Q8_0 rows of `k` values: every code random over all 256 bytes,
    /// each block's scale a positive normal f16 with exponent field `exp`
    /// (`[2^(exp−15), 2^(exp−14))`). The q8f32 planes and the rows
    /// dequantized (`q·d`, exact in f32).
    fn q8_rows(rows: usize, k: usize, exp: u16, seed: u64) -> (Vec<u32>, Vec<u16>, Vec<f32>) {
        let mut rng = Rng(seed);
        let blocks: Vec<Q8Block> = (0..rows * k / 32)
            .map(|_| {
                let d = (exp << 10) | (rng.next() as u16 & 0x3ff);
                let mut q = [0i8; 32];
                for v in &mut q {
                    *v = rng.next() as u8 as i8;
                }
                Q8Block { d, q }
            })
            .collect();
        let deq = blocks
            .iter()
            .flat_map(|b| {
                let d = half_to_f32(b.d);
                b.q.map(|q| f32::from(q) * d)
            })
            .collect();
        let (qs, d) = q8_0_planes(&blocks);
        (qs, d, deq)
    }

    /// `n` BF16 values uniform on `±scale`, in their words (value `2j` the
    /// low half of word `j`) and widened on the host (exact).
    fn bf16_rows(n: usize, scale: f64, seed: u64) -> (Vec<u32>, Vec<f32>) {
        let mut rng = Rng(seed);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
                let v = ((2.0 * u - 1.0) * scale) as f32;
                f32::from_bits(v.to_bits() & 0xFFFF_0000)
            })
            .collect();
        let words = vals
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[lo, hi]| (lo.to_bits() >> 16) | (hi.to_bits() & 0xFFFF_0000))
            .collect();
        (words, vals)
    }

    /// A site's down and up on the card, each with the down's view as `R·S`
    /// rows of `D` (the dense gemv's): Q8_0 planes, or BF16 words.
    enum Planes {
        Q8_0 {
            down_qs: DeviceTensor<u32>,
            down_d: DeviceTensor<u16>,
            up_qs: DeviceTensor<u32>,
            up_d: DeviceTensor<u16>,
            view_qs: DeviceTensor<u32>,
            view_d: DeviceTensor<u16>,
        },
        Bf16 {
            down: DeviceTensor<u32>,
            up: DeviceTensor<u32>,
            view: DeviceTensor<u32>,
        },
    }

    /// One site's weights, on the card and dequantized on the host.
    struct Site {
        /// A down or up row's dot in the card's lane order: `card::q8_dot`
        /// for Q8_0, `card::f32_dot` for BF16.
        dot: fn(&[f32], &[f32]) -> f32,
        gamma: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Vec<f32>,
        gamma_d: DeviceBuffer<f32>,
        planes: Planes,
        inject_d: DeviceTensor<f32>,
    }

    impl Site {
        /// The norm's gain and the inject rows from `seed` for streams of
        /// `d` values, with `planes` and their host rows.
        fn with(
            gpu: &Gpu,
            (seed, d): (u64, usize),
            dot: fn(&[f32], &[f32]) -> f32,
            planes: Planes,
            (down, up): (Vec<f32>, Vec<f32>),
        ) -> Result<Site, GateError> {
            let st = gpu.stream();
            let gamma: Vec<f32> = activations(S * d, 1, seed as u32)
                .iter()
                .map(|v| 1.0 + v / 64.0)
                .collect();
            let inject: Vec<f32> = activations(S * S * d, 1, seed as u32 + 1)
                .iter()
                .map(|v| v / 64.0)
                .collect();
            Ok(Site {
                dot,
                gamma_d: DeviceBuffer::from_host(st, &gamma)?,
                planes,
                inject_d: DeviceTensor::upload(st, &inject, S, S * d)?,
                gamma,
                down,
                up,
                inject,
            })
        }

        /// The Q8_0 site for streams of `d` values.
        fn new(gpu: &Gpu, seed: u64, d: usize) -> Result<Site, GateError> {
            let st = gpu.stream();
            let wide = S * d;
            // Down scales in [2^-12, 2^-11), up scales in [2^-8, 2^-7): the
            // bottleneck and the gate both move off their flat ends.
            let (dq, dd, down) = q8_rows(R, wide, 3, seed ^ 0xd0d0);
            let (uq, ud, up) = q8_rows(wide, R, 7, seed ^ 0x0909);
            let planes = Planes::Q8_0 {
                down_qs: DeviceTensor::upload(st, &dq, R, wide / 4)?,
                down_d: DeviceTensor::upload(st, &dd, R, wide / 32)?,
                up_qs: DeviceTensor::upload(st, &uq, wide, R / 4)?,
                up_d: DeviceTensor::upload(st, &ud, wide, R / 32)?,
                view_qs: DeviceTensor::upload(st, &dq, R * S, d / 4)?,
                view_d: DeviceTensor::upload(st, &dd, R * S, d / 32)?,
            };
            Site::with(gpu, (seed, d), card::q8_dot, planes, (down, up))
        }

        /// The BF16 site: down values on `±2^-4` and up values on `±1`,
        /// the Q8_0 site's ranges.
        fn new_bf16(gpu: &Gpu, seed: u64) -> Result<Site, GateError> {
            let st = gpu.stream();
            let (dw, down) = bf16_rows(R * WIDE, 1.0 / 16.0, seed ^ 0xd0d0);
            let (uw, up) = bf16_rows(WIDE * R, 1.0, seed ^ 0x0909);
            let planes = Planes::Bf16 {
                down: DeviceTensor::upload(st, &dw, R, WIDE / 2)?,
                up: DeviceTensor::upload(st, &uw, WIDE, R / 2)?,
                view: DeviceTensor::upload(st, &dw, R * S, D / 2)?,
            };
            Site::with(gpu, (seed, D), card::f32_dot, planes, (down, up))
        }

        fn weights(&self, inject: bool) -> SiteWeights<'_> {
            let (down, up) = match &self.planes {
                Planes::Q8_0 {
                    down_qs,
                    down_d,
                    up_qs,
                    up_d,
                    ..
                } => (
                    HcMatrix::Q8_0 {
                        qs: down_qs,
                        d: down_d,
                    },
                    HcMatrix::Q8_0 { qs: up_qs, d: up_d },
                ),
                Planes::Bf16 { down, up, .. } => (HcMatrix::Bf16(down), HcMatrix::Bf16(up)),
            };
            SiteWeights {
                gamma: &self.gamma_d,
                down,
                up,
                inject: inject.then_some(&self.inject_d),
            }
        }

        /// The dense gemv of the down's view (`view`) or the up rows against
        /// `m` columns of `x`, row-major: `q8_0_gemv_mcol` or `bf16_gemv`.
        fn gemv(
            &self,
            c: &Ctx,
            view: bool,
            x: &DeviceBuffer<f32>,
            m: usize,
            y: &mut DeviceBuffer<f32>,
        ) -> Result<(), GpuError> {
            let st = c.gpu.stream();
            match &self.planes {
                Planes::Q8_0 {
                    up_qs,
                    up_d,
                    view_qs,
                    view_d,
                    ..
                } => {
                    let (qs, d) = if view {
                        (view_qs, view_d)
                    } else {
                        (up_qs, up_d)
                    };
                    c.q8.enqueue_q8_0_gemv_mcol(
                        st,
                        Q8_0GemvMcolArgs {
                            qs,
                            d,
                            x,
                            m,
                            out: GemvOut::RowMajor,
                            y,
                        },
                    )
                }
                Planes::Bf16 { up, view: v_w, .. } => {
                    c.q8.enqueue_bf16_gemv(st, if view { v_w } else { up }, x, m, y)
                }
            }
        }

        /// The gemv [`Site::gemv`] runs, as the clauses name it.
        fn gemv_name(&self) -> &'static str {
            match self.planes {
                Planes::Q8_0 { .. } => "q8_0_gemv_mcol",
                Planes::Bf16 { .. } => "bf16_gemv",
            }
        }

        fn host(&self, inject: bool) -> MixWeights<'_> {
            MixWeights {
                gamma: &self.gamma,
                down: &self.down,
                up: &self.up,
                inject: inject.then_some(self.inject.as_slice()),
            }
        }
    }

    struct Ctx {
        gpu: Gpu,
        hc: HcGatedKernels,
        q8: Q8F32Kernels,
        geo: Geometry,
        scratch: HcScratch,
    }

    /// Everything one mix leaves, read back, for its `m` columns.
    struct Out {
        res: Vec<f32>,
        xn: Vec<f32>,
        dpart: Vec<f32>,
        ipart: Vec<f32>,
        lo: Vec<f32>,
        wgt: Vec<f32>,
        mixed: Vec<f32>,
        fault: Option<Fault>,
    }

    /// What a mix starts from: nothing, or `y` for a combine or an init.
    #[derive(Clone, Copy)]
    enum Pre<'a> {
        Plain,
        Combine(&'a [f32]),
        Init(&'a [f32]),
    }

    impl Pre<'_> {
        /// `y` on the card when the mix reads it: a combine's or an init's.
        fn y_on(self, st: &CudaStream) -> Result<Option<DeviceBuffer<f32>>, GateError> {
            Ok(match self {
                Pre::Plain => None,
                Pre::Combine(y) | Pre::Init(y) => Some(DeviceBuffer::from_host(st, y)?),
            })
        }

        /// The launch's start over `y`, the card's copy [`Pre::y_on`] made.
        fn before<'b>(self, y: Option<&'b DeviceBuffer<f32>>) -> Before<'b> {
            match (self, y) {
                (Pre::Combine(_), Some(y)) => Before::Combine { y },
                (Pre::Init(_), Some(y)) => Before::Init { y },
                _ => Before::Plain,
            }
        }
    }

    fn mix(
        c: &mut Ctx,
        site: &Site,
        inject: bool,
        res: &[f32],
        pre: Pre<'_>,
        m: usize,
    ) -> Result<Out, GateError> {
        let st = c.gpu.stream();
        let z = c.geo.sizes(m);
        let mut res_d = DeviceBuffer::from_host(st, res)?;
        let mut mixed_d = DeviceBuffer::from_host(st, &vec![f32::NAN; z.mixed])?;
        let y_d = pre.y_on(st)?;
        let before = pre.before(y_d.as_ref());
        c.hc.enqueue_mix(
            st,
            MixArgs {
                res: &mut res_d,
                before,
                w: site.weights(inject),
                eps: EPS,
                m,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut c.scratch,
                mixed: &mut mixed_d,
            },
        )?;
        let s = &c.scratch;
        let out = Out {
            res: res_d.to_host_vec(st)?,
            xn: s.xn.to_host_vec(st)?[..z.xn].to_vec(),
            dpart: s.dpart.to_host_vec(st)?[..z.down_part].to_vec(),
            ipart: s.ipart.to_host_vec(st)?[..z.inject_part].to_vec(),
            lo: s.lo.to_host_vec(st)?[..z.lo].to_vec(),
            wgt: s.wgt.to_host_vec(st)?[..z.wgt].to_vec(),
            mixed: mixed_d.to_host_vec(st)?,
            fault: c.gpu.take_fault()?,
        };
        Ok(out)
    }

    fn col(v: &[f32], per: usize, c: usize) -> &[f32] {
        &v[c * per..(c + 1) * per]
    }

    fn shown(f: Option<Fault>) -> String {
        f.map_or_else(|| "none".to_owned(), |f| f.to_string())
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let hc = HcGatedKernels::load(gpu.context())?;
        let q8 = Q8F32Kernels::load(gpu.context())?;
        let geo = Geometry::new(S as u32, R as u32, D as u32)?;
        let scratch = HcScratch::new(gpu.stream(), geo)?;
        let mut c = Ctx {
            gpu,
            hc,
            q8,
            geo,
            scratch,
        };
        if let Some(f) = c.gpu.take_fault()? {
            return Err(format!("gate_hc_gated: the fault word held {f} before any launch").into());
        }
        let site = Site::new(&c.gpu, 0x38c0_0001, D)?;
        let mut ok = true;
        ok &= check_band(&mut c, &site)?;
        ok &= check_stages(&mut c, &site)?;
        ok &= check_cols(&mut c, &site)?;
        ok &= check_combine(&mut c, &site)?;
        ok &= check_refuse(&mut c, &site)?;
        ok &= check_fault(&mut c, &site)?;
        ok &= check_wide(&mut c, &site)?;
        ok &= check_hidden(&mut c)?;
        let bf16 = Site::new_bf16(&c.gpu, 0x38c0_0002)?;
        println!("site=bf16");
        ok &= check_band(&mut c, &bf16)?;
        ok &= check_stages(&mut c, &bf16)?;
        ok &= check_cols(&mut c, &bf16)?;
        ok &= check_wide(&mut c, &bf16)?;
        ok &= check_wide_forms(&mut c, &bf16)?;
        ok &= check_overflow(&mut c)?;
        ok &= no_local_depot(&[
            "hc_gated_norm_4",
            "hc_gated_down_4x320",
            "hc_gated_up_mix_4x320",
            "hc_gated_down_bf16_4x320",
            "hc_gated_up_mix_bf16_4x320",
            "hc_gated_combine_4",
            "hc_gated_norm_4w",
            "hc_gated_norm_4w_bf16",
            "hc_gated_norm_4w16",
            "hc_gated_norm_4w16_bf16",
            "hc_gated_combine_4w",
            "hc_gated_lo_4x320",
            "hc_gated_mix_4",
            "hc_gated_mix_4_bf16",
            "hc_gated_up_mma_mix_4",
            "hc_gated_up_mma_mix_4_bf16",
            "hc_gated_down_mma_4",
            // PIN(2026-10-23): bf16_mma_gemm_columns is gone: bf16_mma_gemm keeps every
            // cell to its own row and column, so the HC prompt products run it.
            "bf16_mma_gemm",
        ])?;
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        println!(
            "PASSED: gate_hc_gated the mix within its bands of the f64 rule at 4x2560 rank 320 \
             (inject site and head); every stage its card rule bit for bit on the card's \
             inputs, the down partials the q8f32 gemv of the view; eight columns each column's \
             one-column mix and a rerun bit for bit; combine and init as the rule; shapes, \
             columns, planes and types refused by name; a NaN stream raises the site and \
             leaves the other column clean; the wide arm's norm and combine the capped \
             entries' bits, its glue the card rule, its mix within the error model's bands, \
             each column its own, a NaN raised; the norm at hidden 4352 the rule's bits in \
             both arms; a BF16 site the same over its widened values; no local depot"
        );
        Ok(())
    }

    /// Clause 1 (module doc).
    fn check_band(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let mut ok = true;
        for (tag, inject, m) in [("site", true, 2usize), ("head", false, 1)] {
            let res = activations(WIDE, m, 0x0b0 + m as u32);
            let o = mix(c, site, inject, &res, Pre::Plain, m)?;
            for k in 0..m {
                let r = mix_ref(c.geo, site.host(inject), col(&res, WIDE, k), EPS);
                let as32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
                let mut lines = vec![
                    (
                        "xn",
                        max_rel_err(col(&o.xn, WIDE, k), &as32(&r.xn))?,
                        XN_BAND,
                    ),
                    ("lo", max_rel_err(col(&o.lo, R, k), &as32(&r.lo))?, LO_BAND),
                    (
                        "mixed",
                        max_rel_err(col(&o.mixed, D, k), &as32(&r.mixed))?,
                        MIXED_BAND,
                    ),
                ];
                if let Some(w) = &r.wgt {
                    lines.push(("wgt", max_rel_err(col(&o.wgt, S, k), &as32(w))?, LO_BAND));
                }
                for (name, err, band) in lines {
                    let pass = err <= band;
                    ok &= pass;
                    println!(
                        "band[{tag}:col{k}:{name}] max_rel_err={err:.3e} band={band:e} {}",
                        verdict(pass)
                    );
                }
            }
            let clean = o.fault.is_none();
            ok &= clean;
            println!(
                "band[{tag}:fault] fault=\"{}\" {}",
                shown(o.fault),
                verdict(clean)
            );
        }
        Ok(ok)
    }

    /// Clause 2 (module doc).
    fn check_stages(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let m = 2;
        let res = activations(WIDE, m, 0x57a6);
        let o = mix(c, site, true, &res, Pre::Plain, m)?;
        let mut ok = o.fault.is_none();
        println!("stages[fault] fault=\"{}\" {}", shown(o.fault), verdict(ok));
        let mut line = |name: &str, got: &[f32], want: &[f32]| {
            let pass = bits_equal(got, want);
            ok &= pass;
            println!(
                "stages[{name}] values={} bit_equal={pass} max_ulps={} {}",
                got.len(),
                if got.len() == want.len() {
                    max_ulps(got, want)
                } else {
                    u32::MAX
                },
                verdict(pass)
            );
        };
        let per_col = |f: &dyn Fn(usize) -> Vec<f32>| (0..m).flat_map(f).collect::<Vec<f32>>();
        let geo = c.geo;
        let xn = per_col(&|k| card::norm(geo, col(&res, WIDE, k), &site.gamma, EPS));
        line("xn", &o.xn, &xn);
        let dot = site.dot;
        let dp = per_col(&|k| card::down_parts_by(geo, dot, &site.down, col(&o.xn, WIDE, k)));
        line("down_parts", &o.dpart, &dp);
        let lo = per_col(&|k| card::lo(geo, col(&o.dpart, R * S, k), ACT));
        line("lo", &o.lo, &lo);
        let ip = per_col(&|k| card::inject_parts(geo, &site.inject, col(&o.xn, WIDE, k)));
        line("inject_parts", &o.ipart, &ip);
        let w = per_col(&|k| card::wgt(geo, col(&o.ipart, S * S, k), ACT));
        line("wgt", &o.wgt, &w);
        let mx = per_col(&|k| {
            card::up_mix_by(
                geo,
                dot,
                (&site.up, col(&o.lo, R, k)),
                col(&o.xn, WIDE, k),
                ACT,
            )
        });
        line("mixed", &o.mixed, &mx);
        // The down partials through the dense gemv on the view: stream s's
        // columns against every view row, rows j·S + s kept.
        let st = c.gpu.stream();
        let mut gemv = vec![0.0f32; m * R * S];
        for s in 0..S {
            let x: Vec<f32> = (0..m)
                .flat_map(|k| col(&o.xn, WIDE, k)[s * D..(s + 1) * D].to_vec())
                .collect();
            let x_d = DeviceBuffer::from_host(st, &x)?;
            let mut y_d = DeviceBuffer::from_host(st, &vec![0.0f32; R * S * m])?;
            site.gemv(c, true, &x_d, m, &mut y_d)?;
            let y = y_d.to_host_vec(st)?;
            for k in 0..m {
                for j in 0..R {
                    gemv[(k * R + j) * S + s] = y[(j * S + s) * m + k];
                }
            }
        }
        line(
            &format!("down_parts_vs_{}", site.gemv_name()),
            &o.dpart,
            &gemv,
        );
        // The up rows through the dense gemv on the card's `lo`, then the gate
        // and the mean on the host over the card's `xn`.
        let lo_d = DeviceBuffer::from_host(st, &o.lo)?;
        let mut g_d = DeviceBuffer::from_host(st, &vec![0.0f32; WIDE * m])?;
        site.gemv(c, false, &lo_d, m, &mut g_d)?;
        let g = g_d.to_host_vec(st)?;
        let mut mx = vec![0.0f32; m * D];
        for k in 0..m {
            let xn = col(&o.xn, WIDE, k);
            for i in 0..D {
                let mut acc = 0.0f32;
                for s in 0..S {
                    let sg = sigmoid(g[(s * D + i) * m + k]);
                    acc = if s == 0 {
                        xn[s * D + i] * sg
                    } else {
                        xn[s * D + i].mul_add(sg, acc)
                    };
                }
                mx[k * D + i] = acc * (1.0 / S as f32);
            }
        }
        line(&format!("mixed_vs_{}", site.gemv_name()), &o.mixed, &mx);
        // The head after the site: no inject, its own stages, the site's
        // weights left in the scratch for the next combine.
        let head_res = activations(WIDE, 1, 0x57a7);
        let h = mix(c, site, false, &head_res, Pre::Plain, 1)?;
        let hx = card::norm(geo, &head_res, &site.gamma, EPS);
        let hmx = card::up_mix_by(geo, dot, (&site.up, &h.lo), &h.xn, ACT);
        let (p_x, p_m) = (bits_equal(&h.xn, &hx), bits_equal(&h.mixed, &hmx));
        let kept = bits_equal(&h.wgt, &o.wgt[..S]);
        let pass = p_x && p_m && kept && h.fault.is_none();
        ok &= pass;
        println!(
            "stages[head] xn_bit_equal={p_x} mixed_bit_equal={p_m} site_weights_kept={kept} \
             fault=\"{}\" {}",
            shown(h.fault),
            verdict(pass)
        );
        Ok(ok)
    }

    /// Clause 3 (module doc).
    fn check_cols(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let m = MAX_COLS;
        let res = activations(WIDE, m, 0xc015);
        let y = activations(D, m, 0xc016);
        // A first mix leaves the weights the combine below reads.
        let _ = mix(c, site, true, &res, Pre::Plain, m)?;
        let wide8 = mix(c, site, true, &res, Pre::Combine(&y), m)?;
        let mut ok = wide8.fault.is_none();
        let mut same = true;
        for k in 0..m {
            // Each column alone, after the same first mix of that column.
            let _ = mix(c, site, true, col(&res, WIDE, k), Pre::Plain, 1)?;
            let one = mix(
                c,
                site,
                true,
                col(&res, WIDE, k),
                Pre::Combine(col(&y, D, k)),
                1,
            )?;
            let parts = [
                ("res", col(&wide8.res, WIDE, k), &one.res[..]),
                ("xn", col(&wide8.xn, WIDE, k), &one.xn[..]),
                ("down_parts", col(&wide8.dpart, R * S, k), &one.dpart[..]),
                ("inject_parts", col(&wide8.ipart, S * S, k), &one.ipart[..]),
                ("lo", col(&wide8.lo, R, k), &one.lo[..]),
                ("wgt", col(&wide8.wgt, S, k), &one.wgt[..]),
                ("mixed", col(&wide8.mixed, D, k), &one.mixed[..]),
            ];
            for (name, a, b) in parts {
                if !bits_equal(a, b) {
                    same = false;
                    println!("cols[col{k}:{name}] differs max_ulps={}", max_ulps(a, b));
                }
            }
            ok &= one.fault.is_none();
        }
        ok &= same;
        println!(
            "cols[m8_vs_m1] every output of 8 columns bit_equal={same} {}",
            verdict(same)
        );
        let _ = mix(c, site, true, &res, Pre::Plain, m)?;
        let again = mix(c, site, true, &res, Pre::Combine(&y), m)?;
        let rerun = bits_equal(&again.mixed, &wide8.mixed)
            && bits_equal(&again.res, &wide8.res)
            && bits_equal(&again.wgt, &wide8.wgt);
        ok &= rerun;
        println!("cols[rerun] bit_equal={rerun} {}", verdict(rerun));
        Ok(ok)
    }

    /// Clause 10 (module doc).
    /// Streams past one norm chunk a thread (`hidden` above 16 values of
    /// 256 threads, 4096): a Q8_0 site at hidden 4352 runs the norm's second
    /// walk over the streams, its `xn` the rule's bits, plain and after a
    /// combine (whose streams the rule's too) in the capped arm and plain in
    /// the wide arm. The context's geometry and scratch are swapped in for
    /// it and restored.
    fn check_hidden(c: &mut Ctx) -> Result<bool, GateError> {
        const HID: usize = 4352;
        let geo = Geometry::new(S as u32, R as u32, HID as u32)?;
        let scratch = HcScratch::new(c.gpu.stream(), geo)?;
        let geo0 = std::mem::replace(&mut c.geo, geo);
        let scratch0 = std::mem::replace(&mut c.scratch, scratch);
        let ok = hidden_lines(c, HID);
        c.geo = geo0;
        c.scratch = scratch0;
        ok
    }

    /// [`check_hidden`]'s lines over streams of `hid` values, on the
    /// context's geometry.
    fn hidden_lines(c: &mut Ctx, hid: usize) -> Result<bool, GateError> {
        let (m, wide, geo) = (2, S * hid, c.geo);
        let site = Site::new(&c.gpu, 0x38c0_0004, hid)?;
        let res = activations(wide, m, 0x4352);
        let y = activations(hid, m, 0x4353);
        let norm = |r: &[f32]| -> Vec<f32> {
            (0..m)
                .flat_map(|k| card::norm(geo, col(r, wide, k), &site.gamma, EPS))
                .collect()
        };
        let first = mix(c, &site, true, &res, Pre::Plain, m)?;
        let after = mix(c, &site, true, &res, Pre::Combine(&y), m)?;
        let mut streams = res.clone();
        combine(geo, m, &mut streams, &y, &first.wgt);
        let mut w = Wide::new(c)?;
        let wide_out = wide_mix(c, &mut w, &site, true, &res, Pre::Plain, m)?;
        let mut ok = true;
        for (name, pass) in [
            (
                "plain",
                bits_equal(&first.xn, &norm(&res)) && first.fault.is_none(),
            ),
            (
                "combine",
                bits_equal(&after.res, &streams)
                    && bits_equal(&after.xn, &norm(&streams))
                    && after.fault.is_none(),
            ),
            (
                "wide_plain",
                bits_equal(&wide_out.xn, &norm(&res)) && wide_out.fault.is_none(),
            ),
        ] {
            ok &= pass;
            println!(
                "hidden[{name}] hidden={hid} m={m} xn the rule's bit_equal={pass} {}",
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// Clause 4 (module doc).
    fn check_combine(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let m = 3;
        let geo = c.geo;
        let res = activations(WIDE, m, 0xc0b1);
        let y = activations(D, m, 0xc0b2);
        let first = mix(c, site, true, &res, Pre::Plain, m)?;
        let after = mix(c, site, true, &res, Pre::Combine(&y), m)?;
        let mut want = res.clone();
        combine(geo, m, &mut want, &y, &first.wgt);
        let p1 = bits_equal(&after.res, &want) && after.fault.is_none();
        println!(
            "combine[before_mix] m={m} res_bit_equal={p1} max_ulps={} {}",
            max_ulps(&after.res, &want),
            verdict(p1)
        );
        let init = mix(c, site, false, &res, Pre::Init(&y), m)?;
        let mut want_init = Vec::with_capacity(m * WIDE);
        for k in 0..m {
            for _ in 0..S {
                want_init.extend_from_slice(col(&y, D, k));
            }
        }
        let p2 = bits_equal(&init.res, &want_init) && init.fault.is_none();
        println!(
            "combine[init] every stream the embedding bit_equal={p2} {}",
            verdict(p2)
        );
        // The combine alone, over the weights the site before left.
        let _ = mix(c, site, true, &res, Pre::Plain, m)?;
        let st = c.gpu.stream();
        let mut res_d = DeviceBuffer::from_host(st, &res)?;
        let y_d = DeviceBuffer::from_host(st, &y)?;
        c.hc.enqueue_combine(st, &mut res_d, &y_d, m, c.gpu.unlabelled_sink(), &c.scratch)?;
        let alone = res_d.to_host_vec(st)?;
        let f = c.gpu.take_fault()?;
        let p3 = bits_equal(&alone, &want) && f.is_none();
        println!(
            "combine[alone] res_bit_equal={p3} fault=\"{}\" {}",
            shown(f),
            verdict(p3)
        );
        Ok(p1 && p2 && p3)
    }

    /// Clause 5 (module doc).
    fn check_refuse(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let mut ok = true;
        let mut line = |case: &str, err: Option<String>, needle: &str| {
            let pass = err.as_deref().is_some_and(|e| e.contains(needle));
            ok &= pass;
            println!(
                "refuse[{case}] error=\"{}\" names=\"{needle}\" {}",
                err.as_deref().unwrap_or("none"),
                verdict(pass)
            );
        };
        for (case, s, r, d, needle) in [
            ("streams_3", 3u32, 320u32, 2560u32, "3 streams at rank 320"),
            ("rank_256", 4, 256, 2560, "4 streams at rank 256"),
            ("hidden_2688", 4, 320, 2688, "hidden 2688"),
        ] {
            line(
                case,
                Geometry::new(s, r, d).err().map(|e| e.to_string()),
                needle,
            );
        }
        let st = c.gpu.stream();
        for (case, m) in [("cols_0", 0usize), ("cols_9", 9)] {
            let mut res_d = DeviceBuffer::from_host(st, &vec![0.0f32; WIDE * 9])?;
            let mut mixed_d = DeviceBuffer::from_host(st, &vec![0.0f32; D * 9])?;
            let e = c.hc.enqueue_mix(
                st,
                MixArgs {
                    res: &mut res_d,
                    before: Before::Plain,
                    w: site.weights(true),
                    eps: EPS,
                    m,
                    fault: c.gpu.unlabelled_sink(),
                    scratch: &mut c.scratch,
                    mixed: &mut mixed_d,
                },
            );
            let want = HcRefused::Cols { m }.to_string();
            line(case, e.err().map(|e| e.to_string()), &want);
        }
        let mut res_d = DeviceBuffer::from_host(st, &vec![0.0f32; WIDE])?;
        let mut mixed_d = DeviceBuffer::from_host(st, &vec![0.0f32; D])?;
        let mut w = site.weights(true);
        w.down = w.up;
        let e = c.hc.enqueue_mix(
            st,
            MixArgs {
                res: &mut res_d,
                before: Before::Plain,
                w,
                eps: EPS,
                m: 1,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut c.scratch,
                mixed: &mut mixed_d,
            },
        );
        line("up_as_down", e.err().map(|e| e.to_string()), "down planes");
        let short = DeviceTensor::upload(st, &site.inject[..S * D], S, D)?;
        let mut w = site.weights(true);
        w.inject = Some(&short);
        let e = c.hc.enqueue_mix(
            st,
            MixArgs {
                res: &mut res_d,
                before: Before::Plain,
                w,
                eps: EPS,
                m: 1,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut c.scratch,
                mixed: &mut mixed_d,
            },
        );
        line(
            "inject_4x2560",
            e.err().map(|e| e.to_string()),
            "inject is 4x2560",
        );
        let e = check_types(
            GgmlType::F32,
            GgmlType::Q4_K,
            GgmlType::Q8_0,
            Some(GgmlType::F32),
        );
        line(
            "down_q4_k",
            e.err().map(|e| e.to_string()),
            "hc down is Q4_K",
        );
        let e = check_types(
            GgmlType::F32,
            GgmlType::Q8_0,
            GgmlType::Q8_0,
            Some(GgmlType::F16),
        );
        line(
            "inject_f16",
            e.err().map(|e| e.to_string()),
            "hc inject is F16",
        );
        let clean = check_types(GgmlType::F32, GgmlType::Q8_0, GgmlType::Q8_0, None).is_ok()
            && check_types(GgmlType::F32, GgmlType::BF16, GgmlType::BF16, None).is_ok();
        ok &= clean;
        println!(
            "refuse[file_types] f32/q8_0/q8_0 and f32/bf16/bf16 accepted={clean} {}",
            verdict(clean)
        );
        let f = c.gpu.take_fault()?;
        ok &= f.is_none();
        println!(
            "refuse[no_launch_raised] fault=\"{}\" {}",
            shown(f),
            verdict(f.is_none())
        );
        Ok(ok)
    }

    /// Clause 6 (module doc).
    fn check_fault(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let m = 2;
        let res = activations(WIDE, m, 0xfa01);
        let clean = mix(c, site, true, &res, Pre::Plain, m)?;
        let want = Some(Fault::at(LAYER_NONE, SITE));
        let mut bad = res.clone();
        bad[WIDE + 2 * D + 77] = f32::NAN;
        let o = mix(c, site, true, &bad, Pre::Plain, m)?;
        let col0 = bits_equal(col(&o.mixed, D, 0), col(&clean.mixed, D, 0))
            && bits_equal(col(&o.xn, WIDE, 0), col(&clean.xn, WIDE, 0))
            && bits_equal(col(&o.lo, R, 0), col(&clean.lo, R, 0))
            && bits_equal(col(&o.wgt, S, 0), col(&clean.wgt, S, 0));
        let nan1 = col(&o.mixed, D, 1).iter().all(|v| v.is_nan());
        let p1 = clean.fault.is_none() && o.fault == want && col0 && nan1;
        println!(
            "fault[nan_stream] fault=\"{}\" want=\"{}\" column0_clean={col0} column1_mixed_all_nan={nan1} {}",
            shown(o.fault),
            shown(want),
            verdict(p1)
        );
        let _ = mix(c, site, true, &res, Pre::Plain, m)?;
        let mut y = activations(D, m, 0xfa02);
        y[D + 5] = f32::INFINITY;
        let o = mix(c, site, true, &res, Pre::Combine(&y), m)?;
        let p2 = o.fault == want;
        println!(
            "fault[inf_combine] fault=\"{}\" want=\"{}\" {}",
            shown(o.fault),
            shown(want),
            verdict(p2)
        );
        let mut norm_ok = true;
        for (case, x, eps) in [
            ("sum", f32::from_bits(0x5f80_0000), EPS),
            ("eps", f32::from_bits(0x5c00_0000), f32::MAX),
        ] {
            let st = c.gpu.stream();
            let mut res_d = DeviceBuffer::from_host(st, &vec![x; WIDE])?;
            let mut mixed_d = DeviceBuffer::from_host(st, &vec![f32::NAN; D])?;
            c.hc.enqueue_mix(
                st,
                MixArgs {
                    res: &mut res_d,
                    before: Before::Plain,
                    w: site.weights(false),
                    eps,
                    m: 1,
                    fault: c.gpu.unlabelled_sink(),
                    scratch: &mut c.scratch,
                    mixed: &mut mixed_d,
                },
            )?;
            let nan_xn = c.scratch.xn.to_host_vec(st)?[..WIDE]
                .iter()
                .all(|v| v.is_nan());
            let nan_mix = mixed_d.to_host_vec(st)?.iter().all(|v| v.is_nan());
            let fault = c.gpu.take_fault()?;
            let pass = nan_xn && nan_mix && fault == want;
            println!(
                "fault[norm_{case}_overflow] xn_all_nan={nan_xn} mixed_all_nan={nan_mix} fault=\"{}\" want=\"{}\" {}",
                shown(fault),
                shown(want),
                verdict(pass)
            );
            norm_ok &= pass;
        }
        Ok(p1 && p2 && norm_ok)
    }

    // ------------------------------------------------------ 8. the wide arm

    /// Columns the wide scratch holds: past the eight-column cap, and an odd
    /// count for the independence clause.
    const WIDE_COLS: usize = 37;

    fn check_overflow(c: &mut Ctx) -> Result<bool, GateError> {
        let mut w = Wide::new(c)?;
        let want = Some(Fault::at(LAYER_NONE, SITE));
        let mut ok = true;
        for (stage, db, ub, inj) in [
            ("up", 0x3980u32, 0x7f7fu32, 0.0f32),
            ("down", 0x7980, 0, 0.0),
            ("inject_pos", 0, 0, f32::from_bits(0x7980_0000)),
            ("inject_neg", 0, 0, -f32::from_bits(0x7980_0000)),
        ] {
            let st = c.gpu.stream();
            let down = vec![f32::from_bits(db << 16); R * WIDE];
            let up = vec![f32::from_bits(ub << 16); WIDE * R];
            let planes = Planes::Bf16 {
                down: DeviceTensor::upload(st, &vec![db | db << 16; R * WIDE / 2], R, WIDE / 2)?,
                up: DeviceTensor::upload(st, &vec![ub | ub << 16; WIDE * R / 2], WIDE, R / 2)?,
                view: DeviceTensor::upload(st, &vec![db | db << 16; R * WIDE / 2], R * S, D / 2)?,
            };
            let mut site = Site::with(&c.gpu, (0x38c0_0003, D), card::f32_dot, planes, (down, up))?;
            site.gamma.fill(1.0);
            site.gamma_d.copy_from_host(st, &site.gamma)?;
            site.inject.fill(inj);
            site.inject_d = DeviceTensor::upload(st, &site.inject, S, WIDE)?;
            let inject = inj != 0.0;
            for m in [1, WIDE_COLS] {
                let res = vec![1.0; m * WIDE];
                let (lo, wgt, mixed, partials_finite, fault) = if m == 1 {
                    let o = mix(c, &site, inject, &res, Pre::Plain, m)?;
                    let part = if inject { &o.ipart } else { &o.dpart };
                    let finite = part.iter().all(|v| v.is_finite());
                    (o.lo, o.wgt, o.mixed, finite, o.fault)
                } else {
                    let o = wide_mix(c, &mut w, &site, inject, &res, Pre::Plain, m)?;
                    (o.lo, o.wgt, o.mixed, true, o.fault)
                };
                let lo_ok = lo.iter().all(|v| {
                    if stage == "down" {
                        v.is_nan()
                    } else {
                        v.is_finite()
                    }
                });
                let mixed_ok = mixed
                    .iter()
                    .all(|v| if inject { v.is_finite() } else { v.is_nan() });
                let wgt_ok = !inject || wgt.iter().all(|v| v.is_nan());
                let pass = partials_finite && lo_ok && mixed_ok && wgt_ok && fault == want;
                println!(
                    "fault[bf16_{stage}_overflow] m={m} partials_finite={partials_finite} lo_ok={lo_ok} mixed_ok={mixed_ok} wgt_ok={wgt_ok} fault=\"{}\" want=\"{}\" {}",
                    shown(fault),
                    shown(want),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        Ok(ok)
    }

    /// `v` followed by zeros to `len` values: a device buffer's copy takes its
    /// whole length, and the wide scratch holds more columns than a site's run.
    fn padded(v: &[f32], len: usize) -> Vec<f32> {
        let mut out = vec![0.0; len];
        out[..v.len()].copy_from_slice(v);
        out
    }

    /// PIN(2026-09-28): the wide mix's distance from the f64 rule, derived,
    /// never measured, from the q8-32 error model
    /// (`bloomery_gpu_gates::rounding::q8_32_rel`, 1.2858e-2 of a column's
    /// RMS): the down reads one such quantization (`xn`), the bottleneck
    /// `silu(·/4)` passes it with a slope of at most 1.1, so `lo` sits within
    /// 1.1 · 1.2858e-2 = 1.4144e-2 in RMS; the up reads a second (`lo`), so
    /// `g` sits within √2 · 1.4144e-2 = 2.0003e-2 of its rule in RMS
    /// relative; `mixed` is Σ xn·σ(g)/4, whose relative error is at most the
    /// absolute error of `g` (σ'/σ = 1 − σ ≤ 1), so it sits within
    /// 2.0003e-2 · rms(g). The inject is the F32 tile on `xn`: no
    /// quantization, the narrow arm's band.
    fn wide_lo_band() -> f64 {
        1.1 * q8_32_rel()
    }

    /// [`wide_lo_band`] through the up's second quantization.
    fn wide_g_rel() -> f64 {
        std::f64::consts::SQRT_2 * wide_lo_band()
    }

    /// The wide arm's kernels and scratch.
    struct Wide {
        hcw: HcWideKernels,
        g32: Gemm32Kernels,
        gemm: GemmKernels,
        route: GemmRoute,
        scratch: HcWideScratch,
    }

    impl Wide {
        fn new(c: &Ctx) -> Result<Self, GateError> {
            Ok(Self {
                hcw: HcWideKernels::load(c.gpu.context())?,
                g32: Gemm32Kernels::load(c.gpu.context())?,
                gemm: GemmKernels::load(c.gpu.context())?,
                route: GemmRoute::new(c.gpu.stream(), WIDE_COLS, 1)?,
                scratch: HcWideScratch::new(c.gpu.stream(), c.geo, WIDE_COLS)?,
            })
        }
    }

    /// Everything one wide mix leaves, read back, for its `m` columns.
    struct WideOut {
        res: Vec<f32>,
        xn: Vec<f32>,
        down: Vec<f32>,
        inj: Vec<f32>,
        lo: Vec<f32>,
        lo_refused: Vec<u32>,
        g: Vec<f32>,
        wgt: Vec<f32>,
        mixed: Vec<f32>,
        fault: Option<Fault>,
    }

    fn wide_mix(
        c: &mut Ctx,
        w: &mut Wide,
        site: &Site,
        inject: bool,
        res: &[f32],
        pre: Pre<'_>,
        m: usize,
    ) -> Result<WideOut, GateError> {
        let st = c.gpu.stream();
        let z = c.geo.sizes(m);
        let mut res_d = DeviceBuffer::from_host(st, res)?;
        let mut mixed_d = DeviceBuffer::from_host(st, &vec![f32::NAN; z.mixed])?;
        let y_d = pre.y_on(st)?;
        let before = pre.before(y_d.as_ref());
        w.gemm
            .enqueue_route_dense(st, m, &mut w.route, c.gpu.unlabelled_sink())?;
        let (hcw, g32, route) = (&w.hcw, &w.g32, &w.route);
        let enqueue = |st: &CudaStream,
                       res: &mut DeviceBuffer<f32>,
                       mixed: &mut DeviceBuffer<f32>,
                       scratch: &mut HcWideScratch| {
            hcw.enqueue_mix(
                st,
                WideMixArgs {
                    res: Streams::F32(res),
                    before,
                    w: site.weights(inject),
                    eps: EPS,
                    m,
                    fault: c.gpu.unlabelled_sink(),
                    scratch,
                    gemm: g32,
                    q8f32: &c.q8,
                    dense: route,
                    mixed,
                    out: None,
                },
            )
        };
        enqueue(st, &mut res_d, &mut mixed_d, &mut w.scratch)?;
        let found_fault = if m > MAX_COLS && matches!(pre, Pre::Plain) {
            let eager = mixed_d.to_host_vec(st)?;
            let eager_wgt = w.scratch.wgt.to_host_vec(st)?;
            let eager_lo = w.scratch.lo.to_host_vec(st)?;
            let eager_words = w.scratch.lo_words.to_host_vec(st)?;
            let eager_refused = w.scratch.lo_refused.to_host_vec(st)?;
            let eager_fault = c.gpu.take_fault()?;
            res_d.copy_from_host(st, res)?;
            mixed_d.copy_from_host(st, &vec![f32::NAN; z.mixed])?;
            let graph = c
                .gpu
                .capture(|st| enqueue(st, &mut res_d, &mut mixed_d, &mut w.scratch))?;
            w.scratch
                .lo
                .copy_from_host(st, &vec![f32::NAN; eager_lo.len()])?;
            w.scratch
                .lo_words
                .copy_from_host(st, &vec![0xdead_beef; eager_words.len()])?;
            w.scratch
                .lo_refused
                .copy_from_host(st, &vec![u32::MAX; eager_refused.len()])?;
            graph.launch(st)?;
            let same = bits_equal(&eager, &mixed_d.to_host_vec(st)?)
                && bits_equal(&eager_wgt, &w.scratch.wgt.to_host_vec(st)?)
                && bits_equal(&eager_lo[..z.lo], &w.scratch.lo.to_host_vec(st)?[..z.lo])
                && eager_words[..z.lo / 2] == w.scratch.lo_words.to_host_vec(st)?[..z.lo / 2]
                && eager_refused[..m] == w.scratch.lo_refused.to_host_vec(st)?[..m];
            let graph_fault = c.gpu.take_fault()?;
            let same = same && eager_fault == graph_fault;
            println!(
                "wide[graph] m={m} eager_graph_bits={same} {}",
                verdict(same)
            );
            if !same {
                return Err("HC wide eager/graph outputs or faults differ".into());
            }
            graph_fault
        } else {
            c.gpu.take_fault()?
        };
        let res = res_d.to_host_vec(st)?;
        let mixed = mixed_d.to_host_vec(st)?;
        wide_out(c, w, (site, inject), m, (res, mixed, found_fault))
    }

    /// [`wide_mix`] over streams held as BF16 values (`res16`, a ubatch
    /// walk's), eager: the streams after it widened in `res`.
    fn wide_mix16(
        c: &mut Ctx,
        w: &mut Wide,
        (site, inject): (&Site, bool),
        res16: &[u16],
        pre: Pre<'_>,
        m: usize,
    ) -> Result<WideOut, GateError> {
        let st = c.gpu.stream();
        let mut res_d = DeviceBuffer::from_host(st, res16)?;
        let mut mixed_d = DeviceBuffer::from_host(st, &vec![f32::NAN; c.geo.sizes(m).mixed])?;
        let y_d = pre.y_on(st)?;
        let before = pre.before(y_d.as_ref());
        w.gemm
            .enqueue_route_dense(st, m, &mut w.route, c.gpu.unlabelled_sink())?;
        w.hcw.enqueue_mix(
            st,
            WideMixArgs {
                res: Streams::Bf16(&mut res_d),
                before,
                w: site.weights(inject),
                eps: EPS,
                m,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut w.scratch,
                gemm: &w.g32,
                q8f32: &c.q8,
                dense: &w.route,
                mixed: &mut mixed_d,
                out: None,
            },
        )?;
        let fault = c.gpu.take_fault()?;
        let res = widened(&res_d.to_host_vec(st)?);
        let mixed = mixed_d.to_host_vec(st)?;
        wide_out(c, w, (site, inject), m, (res, mixed, fault))
    }

    /// A wide mix's outputs read back from the scratch beside its streams,
    /// mixed output and fault; the BF16 bottleneck words checked against the
    /// F32 bottleneck.
    fn wide_out(
        c: &mut Ctx,
        w: &mut Wide,
        (site, inject): (&Site, bool),
        m: usize,
        (res, mixed, fault): (Vec<f32>, Vec<f32>, Option<Fault>),
    ) -> Result<WideOut, GateError> {
        let st = c.gpu.stream();
        let z = c.geo.sizes(m);
        if let Planes::Bf16 { up, .. } = &site.planes
            && m > MAX_COLS
        {
            // Materialize the unfused product for the existing independent glue oracle.
            c.q8.enqueue_bf16_prompt(st, up, &w.scratch.lo, m, &mut w.scratch.g)?;
        }
        let s = &w.scratch;
        let lo = s.lo.to_host_vec(st)?[..z.lo].to_vec();
        let words = s.lo_words.to_host_vec(st)?;
        let pairs = lo.iter().enumerate().all(|(i, &value)| {
            let bits = (words[i / 2] >> (16 * (i % 2))) as u16;
            if value.is_nan() {
                bf16_bits_to_f32(bits).is_nan()
            } else {
                bits == fp32_to_bf16_bits(value)
            }
        });
        println!(
            "wide[lo_bf16] m={m} rounded_pairs={pairs} {}",
            verdict(pairs)
        );
        if !pairs {
            return Err("HC bottleneck BF16 words differ from the F32 bottleneck".into());
        }
        let dots = s.down.to_host_vec(st)?;
        let (down, inj) = if packs(c, site, inject)? {
            let cols = dots[..m * (R + S)].as_chunks::<{ R + S }>().0;
            (
                cols.iter().flat_map(|dot| &dot[..R]).copied().collect(),
                cols.iter().flat_map(|dot| &dot[R..]).copied().collect(),
            )
        } else {
            (dots[..z.lo].to_vec(), dots[z.lo..z.lo + z.wgt].to_vec())
        };
        Ok(WideOut {
            res,
            xn: wide_xn(c, site, s, m)?,
            down,
            inj,
            lo,
            lo_refused: s.lo_refused.to_host_vec(st)?[..m].to_vec(),
            g: s.g.to_host_vec(st)?[..z.xn].to_vec(),
            wgt: s.wgt.to_host_vec(st)?[..z.wgt].to_vec(),
            mixed,
            fault,
        })
    }

    /// Whether the wide arm keeps `site`'s normed streams as BF16 values: a
    /// BF16 site on a compute capability 12 device.
    fn xn_bf16(c: &Ctx, site: &Site) -> Result<bool, GateError> {
        Ok(matches!(site.planes, Planes::Bf16 { .. }) && bloomery_gpu::dense_mma(c.gpu.context())?)
    }

    /// Whether the wide arm runs `site`'s down and inject as one product, its
    /// dots `[m][R + S]` in the scratch's `down` (else `[m][R]`, then
    /// `[m][S]`): a BF16 inject site on a compute capability 12 device.
    fn packs(c: &Ctx, site: &Site, inject: bool) -> Result<bool, GateError> {
        Ok(inject && xn_bf16(c, site)?)
    }

    /// `v` rounded to BF16 (nearest even) and widened back.
    fn bf16_rounded(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&x| bf16_round(x)).collect()
    }

    /// `v` rounded to BF16 (nearest even), as BF16 bits.
    fn bf16_bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| fp32_to_bf16_bits(x)).collect()
    }

    /// BF16 values widened.
    fn widened(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&b| bf16_bits_to_f32(b)).collect()
    }

    /// The wide scratch's normed streams of `m` columns as values: its F32
    /// values, or for a BF16 site on a compute capability 12 device its BF16 halfwords widened.
    fn wide_xn(c: &Ctx, site: &Site, s: &HcWideScratch, m: usize) -> Result<Vec<f32>, GateError> {
        let n = c.geo.sizes(m).xn;
        let raw = s.xn.to_host_vec(c.gpu.stream())?;
        if !xn_bf16(c, site)? {
            return Ok(raw[..n].to_vec());
        }
        Ok((0..n)
            .map(|i| bf16_bits_to_f32((raw[i / 2].to_bits() >> (16 * (i % 2))) as u16))
            .collect())
    }

    /// One wide mix of `m` columns after a combine, the one-expert table
    /// filled for `fill` slots, `weights` in place of the site's and `out`
    /// for the form of `mixed`: the call's result, and whether the streams
    /// came back as they went in.
    fn wide_try(
        c: &mut Ctx,
        w: &mut Wide,
        weights: SiteWeights<'_>,
        (fill, m): (usize, usize),
        out: Option<(MmaForm, &mut GemmAct32)>,
    ) -> Result<(Result<(), GpuError>, bool), GateError> {
        let st = c.gpu.stream();
        let res = activations(WIDE, m, 0x3a05);
        let mut res_d = DeviceBuffer::from_host(st, &res)?;
        let y_d = DeviceBuffer::from_host(st, &activations(D, m, 0x3a06))?;
        let mut mixed_d = DeviceBuffer::from_host(st, &vec![f32::NAN; c.geo.sizes(m).mixed])?;
        // Combine weights of one, so a norm launched before a refusal would
        // change the streams.
        let ones = vec![1.0f32; w.scratch.wgt.len()];
        w.scratch.wgt.copy_from_host(st, &ones)?;
        w.gemm
            .enqueue_route_dense(st, fill, &mut w.route, c.gpu.unlabelled_sink())?;
        let got = w.hcw.enqueue_mix(
            st,
            WideMixArgs {
                res: Streams::F32(&mut res_d),
                before: Before::Combine { y: &y_d },
                w: weights,
                eps: EPS,
                m,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut w.scratch,
                gemm: &w.g32,
                q8f32: &c.q8,
                dense: &w.route,
                mixed: &mut mixed_d,
                out,
            },
        );
        let kept = bits_equal(&res_d.to_host_vec(st)?, &res);
        c.gpu.take_fault()?;
        Ok((got, kept))
    }

    /// `‖a − b‖ / ‖b‖` in f64, infinite on a NaN or a length mismatch.
    fn rms_rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// Clause 9 (module doc).
    fn check_wide_forms(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        use bloomery_gpu::q8f32::{E4m3Scales, E4m3Weight};
        if !bloomery_gpu::dense_mma(c.gpu.context())? {
            println!("wide[form] skipped: no dense MMA tile on this card");
            return Ok(true);
        }
        let bundles = bloomery_gpu_gates::ptx::current_exe_bundles()?;
        let native = bloomery_gpu_gates::ptx::modules(&bundles)?.iter().any(|m| {
            bloomery_gpu_gates::ptx::body(std::slice::from_ref(m), "hc_gated_up_mma_mix_4_mxfp8")
                .is_some()
        });
        let forms: &[MmaForm] = if native {
            &[MmaForm::Bf16, MmaForm::Mxfp8]
        } else {
            &[MmaForm::Bf16]
        };
        let mut w = Wide::new(c)?;
        let st = c.gpu.stream();
        // The next sites' weights, 128 rows over the mix's D values: BF16
        // words of 1/8 .. 1/4, and E4M3 codes (no NaN code) on scales 2^-7 ..
        // 2^0.
        let next_bf16: Vec<u32> = (0..128 * D / 2)
            .map(|i| 0x3e00_3e00u32 | ((i as u32).wrapping_mul(2_654_435_761) & 0x007f_007f))
            .collect();
        let next_bf16 = DeviceTensor::upload(st, &next_bf16, 128, D / 2)?;
        let codes: Vec<u32> = (0..128 * D / 4)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) & 0xfefe_fefe)
            .collect();
        let scales: Vec<u32> = (0..128 * D / 128)
            .map(|i| 0x7879_7a7b + (i as u32 % 4))
            .collect();
        let (qs, e) = (
            DeviceTensor::upload(st, &codes, 128, D / 4)?,
            DeviceTensor::upload(st, &scales, 128, D / 128)?,
        );
        let next_mx = E4m3Weight {
            qs: &qs,
            e: &e,
            scales: E4m3Scales::Mxfp8,
        };
        let mut ok = true;
        for &form in forms {
            // An `out` of another K, refused before any launch.
            let m = WIDE_COLS;
            let mut other_k =
                GemmAct32::new(c.gpu.stream(), m, D + 128)?.with_form(c.gpu.stream(), form)?;
            let weights = site.weights(true);
            let (got, kept) = wide_try(c, &mut w, weights, (m, m), Some((form, &mut other_k)))?;
            let named = matches!(&got, Err(e) if e.to_string().contains(&format!("not {D}")));
            ok &= named && kept;
            println!(
                "wide[form] form={form:?} m={m} edge=out_k refused={named} streams_kept={kept} {}",
                verdict(named && kept)
            );
            let edges = [
                (WIDE_COLS, "normal"),
                (WIDE_COLS, "nan"),
                (WIDE_COLS, "tiny"),
                (3, "apart"),
                (WIDE_COLS, "other"),
            ];
            for (m, edge) in edges.into_iter().filter(|&(_, e)| native || e != "other") {
                let st = c.gpu.stream();
                let mut res = activations(WIDE, m, 0x3a21);
                match edge {
                    "nan" => res[2 * WIDE + D + 11] = f32::NAN,
                    // Column 2 subnormal in every stream: its norm scale is
                    // the eps's, so its mixed values stay subnormal, a range
                    // the form refuses.
                    "tiny" => res[2 * WIDE..3 * WIDE].fill(f32::from_bits(0x200)),
                    _ => {}
                }
                w.gemm
                    .enqueue_route_dense(st, m, &mut w.route, c.gpu.unlabelled_sink())?;
                let mut res_d = DeviceBuffer::from_host(st, &res)?;
                let blank = vec![f32::NAN; m * D];
                let mut mixed_d = DeviceBuffer::from_host(st, &blank)?;
                let other = match form {
                    MmaForm::Bf16 => MmaForm::Mxfp8,
                    MmaForm::Mxfp8 => MmaForm::Bf16,
                };
                let mut act = GemmAct32::new(st, m, D)?.with_form(st, form)?;
                if edge == "other" {
                    act = act.with_form(st, other)?;
                }
                let (hcw, g32, route) = (&w.hcw, &w.g32, &w.route);
                let enqueue =
                    |st: &CudaStream,
                     res: &mut DeviceBuffer<f32>,
                     mixed: &mut DeviceBuffer<f32>,
                     scratch: &mut HcWideScratch,
                     out: Option<(MmaForm, &mut GemmAct32)>| {
                        hcw.enqueue_mix(
                            st,
                            WideMixArgs {
                                res: Streams::F32(res),
                                before: Before::Plain,
                                w: site.weights(true),
                                eps: EPS,
                                m,
                                fault: c.gpu.unlabelled_sink(),
                                scratch,
                                gemm: g32,
                                q8f32: &c.q8,
                                dense: route,
                                mixed,
                                out,
                            },
                        )
                    };
                // The next product over `form` of `x` in `act`; `None` when it
                // refuses a form of other rows, which `act` no longer holds.
                let next = |st: &CudaStream,
                            form: MmaForm,
                            x: &DeviceBuffer<f32>,
                            act: &GemmAct32|
                 -> Result<Option<Vec<f32>>, GateError> {
                    let mut y = DeviceBuffer::from_host(st, &vec![f32::NAN; 128 * m])?;
                    let r = match form {
                        MmaForm::Bf16 => {
                            g32.enqueue_bf16_mma_cols(st, &next_bf16, x, act, m, &mut y)
                        }
                        MmaForm::Mxfp8 => g32.enqueue_mxfp8_w8a8(st, next_mx, x, act, m, &mut y),
                    };
                    match r {
                        Ok(()) => Ok(Some(y.to_host_vec(st)?)),
                        Err(GpuError::State { missing, .. })
                            if missing.contains("form of x's first n columns") =>
                        {
                            Ok(None)
                        }
                        Err(e) => Err(e.into()),
                    }
                };
                enqueue(st, &mut res_d, &mut mixed_d, &mut w.scratch, None)?;
                let plain = mixed_d.to_host_vec(st)?;
                let plain_fault = c.gpu.take_fault()?;
                // The quantizer's form of the rows `x`.
                let quantize = |form: MmaForm, x: &DeviceBuffer<f32>, act: &mut GemmAct32| {
                    g32.enqueue_form_cols(st, form, x, m, act)
                };
                let mut quantized = GemmAct32::new(st, m, D)?.with_form(st, form)?;
                quantize(form, &mixed_d, &mut quantized)?;
                let want = next(st, form, &mixed_d, &quantized)?;
                // Every column of the scratch refused by subnormal rows
                // before, which the mix's form clears.
                let tiny_d = DeviceBuffer::from_host(st, &vec![f32::from_bits(0x200); m * D])?;
                quantize(form, &tiny_d, &mut act)?;
                mixed_d.copy_from_host(st, &blank)?;
                match edge {
                    // A form of these rows from before the mix, which the
                    // mix that does not write it must leave unread.
                    "apart" => quantize(form, &mixed_d, &mut act)?,
                    // The other form of these rows from before the mix,
                    // which the mix that writes `form` must leave unread.
                    "other" => quantize(other, &mixed_d, &mut act)?,
                    _ => {}
                }
                enqueue(
                    st,
                    &mut res_d,
                    &mut mixed_d,
                    &mut w.scratch,
                    Some((form, &mut act)),
                )?;
                let mixed = mixed_d.to_host_vec(st)?;
                let fault = c.gpu.take_fault()?;
                let got = next(st, form, &mixed_d, &act)?;
                let same = bits_equal(&mixed, &plain) && fault == plain_fault;
                let wanted = |y: &Option<Vec<f32>>| match (y, &want) {
                    (Some(a), Some(b)) => bits_equal(a, b),
                    _ => false,
                };
                let (form_bits, graph_bits) = if edge == "apart" {
                    (got.is_none() && want.is_some(), true)
                } else if edge == "other" {
                    let stale = next(st, other, &mixed_d, &act)?.is_none();
                    (wanted(&got) && stale, true)
                } else {
                    let form_bits = wanted(&got);
                    // A graph replay writes the form again over another
                    // mix's form of the same rows.
                    let graph = c.gpu.capture(|st| {
                        enqueue(
                            st,
                            &mut res_d,
                            &mut mixed_d,
                            &mut w.scratch,
                            Some((form, &mut act)),
                        )
                    })?;
                    res_d.copy_from_host(st, &activations(WIDE, m, 0x3a22))?;
                    enqueue(
                        st,
                        &mut res_d,
                        &mut mixed_d,
                        &mut w.scratch,
                        Some((form, &mut act)),
                    )?;
                    res_d.copy_from_host(st, &res)?;
                    graph.launch(st)?;
                    let replayed = next(st, form, &mixed_d, &act)?;
                    c.gpu.take_fault()?;
                    (form_bits, wanted(&replayed))
                };
                let pass = same && form_bits && graph_bits;
                ok &= pass;
                println!(
                    "wide[form] form={form:?} m={m} edge={edge} mixed_bits={same} \
                     next_product_bits={form_bits} eager_graph={graph_bits} {}",
                    verdict(pass)
                );
            }
        }
        Ok(ok)
    }

    /// Clause 8's BF16 streams (a ubatch walk's): a mix over streams held as
    /// BF16 values leaves the combine's or the init's F32 value rounded to
    /// nearest even and normalizes those values, so every output is the F32
    /// streams' plain mix over them, bit for bit; a plain mix leaves the
    /// streams as they were.
    fn check_wide_bf16(c: &mut Ctx, w: &mut Wide, site: &Site) -> Result<bool, GateError> {
        let mut ok = true;
        for m in [3, WIDE_COLS] {
            let res = activations(WIDE, m, 0x3a0c);
            let res16 = bf16_bits(&res);
            let held = widened(&res16);
            let y = activations(D, m, 0x3a0d);
            let wgt = padded(&activations(S, m, 0x3a0e), w.scratch.wgt.len());
            for (name, pre) in [
                ("plain", Pre::Plain),
                ("combine", Pre::Combine(&y)),
                ("init", Pre::Init(&y)),
            ] {
                w.scratch.wgt.copy_from_host(c.gpu.stream(), &wgt)?;
                let got = wide_mix16(c, w, (site, true), &res16, pre, m)?;
                // The F32 streams' value of the same step, rounded.
                let streams = if matches!(pre, Pre::Plain) {
                    held.clone()
                } else {
                    w.scratch.wgt.copy_from_host(c.gpu.stream(), &wgt)?;
                    bf16_rounded(&wide_mix(c, w, site, true, &held, pre, m)?.res)
                };
                let want = wide_mix(c, w, site, true, &streams, Pre::Plain, m)?;
                let p = bits_equal(&got.res, &streams)
                    && bits_equal(&got.xn, &want.xn)
                    && bits_equal(&got.lo, &want.lo)
                    && bits_equal(&got.wgt, &want.wgt)
                    && bits_equal(&got.mixed, &want.mixed)
                    && got.fault.is_none()
                    && want.fault.is_none();
                ok &= p;
                println!(
                    "wide[bf16_streams] m={m} {name}: the streams the F32 step's rounded to \
                     nearest even, every output the F32 streams' mix over them, bit_equal={p} {}",
                    verdict(p)
                );
            }
        }
        Ok(ok)
    }

    /// Clause 8 (module doc).
    fn check_wide(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let mut w = Wide::new(c)?;
        let mut ok = true;
        // (a) the norm and the combine: the capped entries' bits at m <= 8.
        let m = 3;
        let res = activations(WIDE, m, 0x3a01);
        let y = activations(D, m, 0x3a02);
        let first = mix(c, site, true, &res, Pre::Plain, m)?;
        let narrow = mix(c, site, true, &res, Pre::Combine(&y), m)?;
        let _ = wide_mix(c, &mut w, site, true, &res, Pre::Plain, m)?;
        // The wide scratch's weights are the wide site's, not the narrow
        // one's: seed them with the narrow first site's so the combines read
        // the same weights.
        w.scratch
            .wgt
            .copy_from_host(c.gpu.stream(), &padded(&first.wgt, w.scratch.wgt.len()))?;
        let wide = wide_mix(c, &mut w, site, true, &res, Pre::Combine(&y), m)?;
        // A BF16 site on a compute capability 12 device keeps xn as the capped entry's values rounded
        // to BF16.
        let bf16 = xn_bf16(c, site)?;
        let xn = if bf16 {
            bf16_rounded(&narrow.xn)
        } else {
            narrow.xn.clone()
        };
        let p = bits_equal(&wide.res, &narrow.res) && bits_equal(&wide.xn, &xn);
        ok &= p;
        println!(
            "wide[norm_combine] m={m} streams and xn the capped entries' (xn_bf16={bf16}) \
             bit_equal={p} {}",
            verdict(p)
        );
        // The combine alone reads BF16 streams (a ubatch walk's) and writes
        // F32 ones: the rule over the widened values.
        let res16 = bf16_bits(&res);
        let mut want = widened(&res16);
        combine(c.geo, m, &mut want, &y, &first.wgt);
        w.scratch
            .wgt
            .copy_from_host(c.gpu.stream(), &padded(&first.wgt, w.scratch.wgt.len()))?;
        let st = c.gpu.stream();
        let streams_d = DeviceBuffer::from_host(st, &res16)?;
        let mut res_d = DeviceBuffer::from_host(st, &vec![f32::NAN; res.len()])?;
        let y_d = DeviceBuffer::from_host(st, &y)?;
        let io = (&streams_d, &mut res_d);
        w.hcw
            .enqueue_combine(st, io, &y_d, m, c.gpu.unlabelled_sink(), &w.scratch)?;
        let alone = res_d.to_host_vec(st)?;
        let f = c.gpu.take_fault()?;
        let p = bits_equal(&alone, &want) && f.is_none();
        ok &= p;
        println!(
            "wide[combine_alone] F32 streams the rule over the BF16 streams bit_equal={p} \
             fault=\"{}\" {}",
            shown(f),
            verdict(p)
        );
        ok &= check_wide_bf16(c, &mut w, site)?;
        // (b) the glue: each output the card rule on the card's readback of
        // its inputs, bit for bit.
        let m = WIDE_COLS;
        let res = activations(WIDE, m, 0x3a03);
        let o = wide_mix(c, &mut w, site, true, &res, Pre::Plain, m)?;
        let inv = 1.0 / S as f32;
        let lo: Vec<f32> = o.down.iter().map(|&v| silu(v * inv)).collect();
        let wgt: Vec<f32> = o.inj.iter().map(|&v| 2.0 * sigmoid(v * inv)).collect();
        let mixed: Vec<f32> = (0..m * D)
            .map(|i| {
                let (k, h) = (i / D, i % D);
                let mut acc = 0.0f32;
                for s in 0..S {
                    let at = (k * S + s) * D + h;
                    let sg = sigmoid(o.g[at]);
                    acc = if s == 0 {
                        o.xn[at] * sg
                    } else {
                        o.xn[at].mul_add(sg, acc)
                    };
                }
                acc * inv
            })
            .collect();
        let glue = [
            ("lo", bits_equal(&o.lo, &lo)),
            ("lo_refused", o.lo_refused.iter().all(|&word| word == 0)),
            ("wgt", bits_equal(&o.wgt, &wgt)),
            ("mixed", bits_equal(&o.mixed, &mixed)),
        ];
        for (name, p) in glue {
            ok &= p;
            println!(
                "wide[glue:{name}] m={m} the card rule on the readback bit_equal={p} {}",
                verdict(p)
            );
        }
        let packed = packs(c, site, true)?;
        if packed {
            let down: Vec<Vec<f32>> = site
                .down
                .as_chunks::<WIDE>()
                .0
                .iter()
                .map(|r| r.to_vec())
                .collect();
            let up: Vec<Vec<f32>> = site
                .up
                .as_chunks::<R>()
                .0
                .iter()
                .map(|r| r.to_vec())
                .collect();
            ok &= check_mma("hc_down", (&o.down, &o.xn, WIDE), &down);
            ok &= check_mma("hc_up", (&o.g, &o.lo, R), &up);
            let inject: Vec<Vec<f32>> = site
                .inject
                .as_chunks::<WIDE>()
                .0
                .iter()
                .map(|r| r.to_vec())
                .collect();
            ok &= check_mma("hc_inject", (&o.inj, &o.xn, WIDE), &inject);

            // A subnormal inject value is out of the MMA's range: its row's dots
            // replay on the exact owner, the F32 tile's bits over the card's `xn`.
            let st = c.gpu.stream();
            let mut odd = site.inject.clone();
            odd[0] = f32::from_bits(1);
            let odd_d = DeviceTensor::upload(st, &odd, S, WIDE)?;
            let weights = SiteWeights {
                inject: Some(&odd_d),
                ..site.weights(true)
            };
            let (got, _) = wide_try(c, &mut w, weights, (m, m), None)?;
            got?;
            let st = c.gpu.stream();
            let xn_d = DeviceBuffer::from_host(st, &wide_xn(c, site, &w.scratch, m)?)?;
            let mut want = DeviceBuffer::from_host(st, &vec![f32::NAN; m * S])?;
            w.g32.enqueue_f32_tile(st, &odd_d, &xn_d, m, &mut want)?;
            let (dots, want) = (w.scratch.down.to_host_vec(st)?, want.to_host_vec(st)?);
            let replayed = (0..m).all(|k| dots[k * (R + S) + R].to_bits() == want[k * S].to_bits());
            ok &= replayed;
            println!(
                "wide[inject-range] m={m} the out-of-range row the F32 tile's bit_equal={replayed} {}",
                verdict(replayed)
            );

            // Power-of-two weights, with one bit below BF16 in every activation,
            // make each cell's path exact and visible: a cell in an in-range row
            // and column keeps the MMA product of the BF16-rounded activation; a
            // cell whose row (subnormal weights) or column (subnormal or above
            // 2^47) holds an out-of-range operand is replayed on F32 activations.
            let (rows, k, n) = (65, 64, 9);
            let odd_rows = [5, 13, 21, 29, 37, 45, 53, 61, 64];
            let odd_cols = [1, 2, 3, 8];
            let mut words = vec![0x1f80_1f80u32; rows * k / 2];
            for row in odd_rows {
                words[row * k / 2..(row + 1) * k / 2].fill(0x0001_0001);
            }
            words[6 * k / 2..7 * k / 2].fill(0x3f80_3f80);
            let mut input = vec![f32::from_bits(0x4f80_2000); n * k];
            input[k..2 * k].fill(f32::from_bits(0x7b80_2000));
            for col in [2, 3, 8] {
                input[col * k..(col + 1) * k].fill(f32::from_bits(0x200));
            }
            let st = c.gpu.stream();
            let weight = DeviceTensor::upload(st, &words, rows, k / 2)?;
            let x = DeviceBuffer::from_host(st, &input)?;
            let mut y = DeviceBuffer::from_host(st, &vec![f32::NAN; rows * n])?;
            c.q8.enqueue_bf16_prompt(st, &weight, &x, n, &mut y)?;
            let actual = y.to_host_vec(st)?;
            let cells = actual.iter().enumerate().all(|(i, value)| {
                let (row, col) = (i % rows, i / rows);
                let w = f32::from_bits((words[row * k / 2] & 0xffff) << 16);
                let x = if odd_rows.contains(&row) || odd_cols.contains(&col) {
                    input[col * k]
                } else {
                    bf16_round(input[col * k])
                };
                let expected = (f64::from(w) * f64::from(x) * k as f64) as f32;
                value.to_bits() == expected.to_bits()
            });
            ok &= cells;
            println!(
                "wide[bf16-cell-range] mma_kept_and_masked_replayed={cells} {}",
                verdict(cells)
            );
        }
        // (c) the error model's bands against the f64 rule, per column.
        // PIN(2026-10-28): BF16 xn (a BF16 site on a compute capability 12 device) is the F32 norm
        // rounded to nearest even once, each value within BF16's unit
        // roundoff 2^-8 of itself: within 2^-8 · (1 + `XN_BAND`) of max|xn|
        // beyond `XN_BAND`.
        let xn_band = f64::from(XN_BAND)
            + if bf16 {
                U_BF16 * (1.0 + f64::from(XN_BAND))
            } else {
                0.0
            };
        let mut worst = [0.0f64; 4];
        let mut past = Vec::new();
        for k in 0..m {
            let r = mix_ref(c.geo, site.host(true), col(&res, WIDE, k), EPS);
            let as32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
            let wgt_ref = as32(r.wgt.as_deref().unwrap_or(&[]));
            // PIN(2026-10-24): where the inject rows share the down's BF16
            // product, each inject dot sits within `bf16_dot_rel` of its terms'
            // absolute sum over the card's `xn` (`check_mma`'s bound), and
            // `2·σ(·/4)` passes it at a slope of at most 1/8, on top of the F32
            // band.
            let wgt_band = f64::from(LO_BAND)
                + if packed {
                    let x = col(&o.xn, WIDE, k);
                    let dot = site
                        .inject
                        .as_chunks::<WIDE>()
                        .0
                        .iter()
                        .map(|row| {
                            let mag: f64 = row
                                .iter()
                                .zip(x)
                                .map(|(&w, &v)| (f64::from(w) * f64::from(v)).abs())
                                .sum();
                            bf16_dot_rel(WIDE) * mag
                        })
                        .fold(0.0, f64::max);
                    let top = wgt_ref
                        .iter()
                        .fold(0.0f64, |a, &v| a.max(f64::from(v).abs()));
                    dot / 8.0 / top
                } else {
                    0.0
                };
            let g = col(&o.g, WIDE, k);
            let g_rms =
                (g.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / g.len() as f64).sqrt();
            let lines = [
                (
                    "xn",
                    f64::from(max_rel_err(col(&o.xn, WIDE, k), &as32(&r.xn))?),
                    xn_band,
                ),
                (
                    "lo",
                    rms_rel(col(&o.lo, R, k), &as32(&r.lo)),
                    wide_lo_band(),
                ),
                (
                    "mixed",
                    rms_rel(col(&o.mixed, D, k), &as32(&r.mixed)),
                    wide_g_rel() * g_rms,
                ),
                (
                    "wgt",
                    f64::from(max_rel_err(col(&o.wgt, S, k), &wgt_ref)?),
                    wgt_band,
                ),
            ];
            for (i, (name, err, band)) in lines.into_iter().enumerate() {
                worst[i] = worst[i].max(err / band);
                if err > band {
                    past.push((k, name, err, band));
                }
            }
        }
        let p = past.is_empty() && o.fault.is_none();
        ok &= p;
        println!(
            "wide[band] m={m} worst err/band xn {:.3} lo {:.3} mixed {:.3} wgt {:.3}; past the \
             band {:?}; fault=\"{}\" {}",
            worst[0],
            worst[1],
            worst[2],
            worst[3],
            &past[..past.len().min(4)],
            shown(o.fault),
            verdict(p)
        );
        // (d) columns: each of m columns its one-column wide mix bit for bit,
        // and a rerun bit-identical.
        let mut same = true;
        for k in [0, 1, m / 2, m - 1] {
            let one = wide_mix(c, &mut w, site, true, col(&res, WIDE, k), Pre::Plain, 1)?;
            let parts = [
                ("xn", col(&o.xn, WIDE, k), &one.xn[..]),
                ("lo", col(&o.lo, R, k), &one.lo[..]),
                ("wgt", col(&o.wgt, S, k), &one.wgt[..]),
                ("mixed", col(&o.mixed, D, k), &one.mixed[..]),
            ];
            for (name, a, b) in parts {
                if !bits_equal(a, b) {
                    same = false;
                    println!(
                        "wide[cols:col{k}:{name}] differs max_ulps={}",
                        max_ulps(a, b)
                    );
                }
            }
        }
        let again = wide_mix(c, &mut w, site, true, &res, Pre::Plain, m)?;
        let rerun = bits_equal(&again.mixed, &o.mixed) && bits_equal(&again.wgt, &o.wgt);
        ok &= same && rerun;
        println!(
            "wide[cols] m={m}: columns 0, 1, {}, {} each its one-column mix bit_equal={same}; \
             rerun bit_equal={rerun} {}",
            m / 2,
            m - 1,
            verdict(same && rerun)
        );
        // (e) no silent failure: a NaN in one stream of one column raises the
        // quantizer's QuantColumn and, as that column's NaN reaches the mix,
        // the mix's site, and leaves the other columns their clean bits; a
        // column count past the scratch is refused by name.
        let mut bad = res.clone();
        bad[2 * WIDE + D + 11] = f32::NAN;
        let o2 = wide_mix(c, &mut w, site, true, &bad, Pre::Plain, m)?;
        let want = Some(match site.planes {
            Planes::Q8_0 { .. } => Fault::of_sites(LAYER_NONE, &[FaultSite::QuantColumn, SITE]),
            // A BF16 down and up read the f32 rows: no quantizer runs.
            Planes::Bf16 { .. } => Fault::of_sites(LAYER_NONE, &[SITE]),
        });
        let others = (0..m).filter(|&k| k != 2).all(|k| {
            bits_equal(col(&o2.mixed, D, k), col(&o.mixed, D, k))
                && bits_equal(col(&o2.wgt, S, k), col(&o.wgt, S, k))
        });
        let refused = o2
            .lo_refused
            .iter()
            .enumerate()
            .all(|(k, &word)| word == u32::from(k == 2));
        let p = o2.fault == want && others && refused;
        ok &= p;
        println!(
            "wide[fault] fault=\"{}\" want=\"{}\" the other columns clean={others} {}",
            shown(o2.fault),
            shown(want),
            verdict(p)
        );
        if matches!(site.planes, Planes::Bf16 { .. }) {
            let mut tiny = res.clone();
            tiny[2 * WIDE..3 * WIDE].fill(f32::from_bits(0x100));
            let many = wide_mix(c, &mut w, site, true, &tiny, Pre::Plain, m)?;
            let one = wide_mix(c, &mut w, site, true, col(&tiny, WIDE, 2), Pre::Plain, 1)?;
            let others = (0..m).filter(|&k| k != 2).all(|k| {
                bits_equal(col(&many.mixed, D, k), col(&o.mixed, D, k))
                    && bits_equal(col(&many.wgt, S, k), col(&o.wgt, S, k))
            });
            let same = bits_equal(col(&many.mixed, D, 2), &one.mixed)
                && bits_equal(col(&many.wgt, S, 2), &one.wgt)
                && bits_equal(col(&many.lo, R, 2), &one.lo)
                && bits_equal(col(&many.down, R, 2), &one.down);
            let refused = many
                .lo_refused
                .iter()
                .enumerate()
                .all(|(k, &word)| word == u32::from(k == 2));
            // On a compute capability 12 device the column's normed values are BF16 subnormals, which the
            // down's tile refuses as their copies land: its dots are the F32
            // tile's bits over the widened values.
            let replayed = if xn_bf16(c, site)? {
                let st = c.gpu.stream();
                let rows = DeviceTensor::upload(st, &site.down, R, WIDE)?;
                let xn_d = DeviceBuffer::from_host(st, &one.xn)?;
                let mut want = DeviceBuffer::from_host(st, &vec![f32::NAN; R])?;
                w.g32.enqueue_f32_tile(st, &rows, &xn_d, 1, &mut want)?;
                bits_equal(col(&many.down, R, 2), &want.to_host_vec(st)?)
            } else {
                true
            };
            let p = others
                && same
                && refused
                && replayed
                && one.lo_refused == [1]
                && many.fault.is_none()
                && one.fault.is_none();
            ok &= p;
            println!(
                "wide[subnormal_columns] other_columns_clean={others} one_column_bits={same} \
                 down_replayed_f32_tile={replayed} {}",
                verdict(p)
            );
        }
        let past_cols = WIDE_COLS + 1;
        let res_long = activations(WIDE, past_cols, 0x3a04);
        // A route as wide as the call, so the refusal is the scratch's own.
        let mut route = GemmRoute::new(c.gpu.stream(), past_cols, 1)?;
        std::mem::swap(&mut w.route, &mut route);
        let got = wide_mix(c, &mut w, site, true, &res_long, Pre::Plain, past_cols);
        std::mem::swap(&mut w.route, &mut route);
        let named = matches!(&got, Err(e) if e.to_string().contains("the wide scratch holds"));
        ok &= named;
        println!(
            "wide[refuse] {past_cols} columns -> {} {}",
            match &got {
                Ok(_) => "accepted".to_owned(),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        // Refused before any launch, the streams as they went in: planes of
        // another shape (the down's and the up's exchanged), and a one-expert
        // table filled for fewer slots than the mix's columns.
        let m = 5;
        let sw = site.weights(true);
        let swapped = SiteWeights {
            down: sw.up,
            up: sw.down,
            ..sw
        };
        let cases = [
            ("plane_shape", swapped, m, "down planes are"),
            (
                "table_short",
                site.weights(true),
                m - 1,
                "the one-expert table is filled for Some(4) slots",
            ),
        ];
        for (name, weights, fill, want) in cases {
            let (got, kept) = wide_try(c, &mut w, weights, (fill, m), None)?;
            let named = matches!(&got, Err(e) if e.to_string().contains(want));
            let p = named && kept;
            ok &= p;
            println!(
                "wide[refuse:{name}] m={m}, the table filled for {fill} -> {}; the streams \
                 as they went in: {kept} (want an error naming {want:?}) {}",
                match &got {
                    Ok(()) => "accepted".to_owned(),
                    Err(e) => e.to_string(),
                },
                verdict(p)
            );
        }
        Ok(ok)
    }
}
