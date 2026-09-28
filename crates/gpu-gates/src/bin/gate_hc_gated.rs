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
//!    Q8_0, each refused by name.
//! 6. fault: a NaN in one stream of one column raises the site
//!    (`hc_gated::SITE`, unlabelled) and leaves the other column bit for bit
//!    the clean launch's, the NaN column's `mixed` all NaN; an infinity in
//!    the combine's `y` raises it too; clean launches raise nothing.
//! 7. shape: the eight entries compile with no local depot.
//! 8. wide: the wide arm (`HcWideKernels`, a ubatch's mix over any column
//!    count): its norm and combine the capped entries' bits at three
//!    columns, the combine alone the rule; over 37 columns its glue — the
//!    bottleneck, the weights, the mixed values — the card rule on the
//!    card's readback of its inputs, bit for bit; each column's `xn`, `lo`,
//!    weights and `mixed` within the bands the error model derives for the q8
//!    step per 32 values (`WIDE_LO_BAND`, `WIDE_G_REL`) of `mix_ref`; four of
//!    the columns each its one-column mix bit for bit and a rerun
//!    bit-identical; a NaN in one column raises and leaves the others clean;
//!    38 columns on a 37-column scratch refused by name.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_hc_gated: built without the `gpu` feature; see `just gate-gpu-hc-gated`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hc_gated", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmKernels, GemmRoute};
    use bloomery_gpu::hc_gated::{
        Before, HcGatedKernels, HcScratch, HcWideKernels, HcWideScratch, MixArgs, SITE,
        SiteWeights, WideMixArgs, check_types,
    };
    use bloomery_gpu::linear::{sigmoid, silu};
    use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvMcolArgs, Q8F32Kernels};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, max_rel_err, max_ulps, no_local_depot, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, Q8Block, half_to_f32};
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

    /// One site's weights, on the card and dequantized on the host.
    struct Site {
        gamma: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Vec<f32>,
        gamma_d: DeviceBuffer<f32>,
        down_qs: DeviceTensor<u32>,
        down_d: DeviceTensor<u16>,
        up_qs: DeviceTensor<u32>,
        up_d: DeviceTensor<u16>,
        inject_d: DeviceTensor<f32>,
        /// The down's planes as `R·S` rows of `D`: the q8f32 gemv's view.
        view_qs: DeviceTensor<u32>,
        view_d: DeviceTensor<u16>,
    }

    impl Site {
        fn new(gpu: &Gpu, seed: u64) -> Result<Site, GateError> {
            let st = gpu.stream();
            let gamma: Vec<f32> = activations(WIDE, 1, seed as u32)
                .iter()
                .map(|v| 1.0 + v / 64.0)
                .collect();
            let inject: Vec<f32> = activations(S * WIDE, 1, seed as u32 + 1)
                .iter()
                .map(|v| v / 64.0)
                .collect();
            // Down scales in [2^-12, 2^-11), up scales in [2^-8, 2^-7): the
            // bottleneck and the gate both move off their flat ends.
            let (dq, dd, down) = q8_rows(R, WIDE, 3, seed ^ 0xd0d0);
            let (uq, ud, up) = q8_rows(WIDE, R, 7, seed ^ 0x0909);
            Ok(Site {
                gamma_d: DeviceBuffer::from_host(st, &gamma)?,
                down_qs: DeviceTensor::upload(st, &dq, R, WIDE / 4)?,
                down_d: DeviceTensor::upload(st, &dd, R, WIDE / 32)?,
                up_qs: DeviceTensor::upload(st, &uq, WIDE, R / 4)?,
                up_d: DeviceTensor::upload(st, &ud, WIDE, R / 32)?,
                inject_d: DeviceTensor::upload(st, &inject, S, WIDE)?,
                view_qs: DeviceTensor::upload(st, &dq, R * S, D / 4)?,
                view_d: DeviceTensor::upload(st, &dd, R * S, D / 32)?,
                gamma,
                down,
                up,
                inject,
            })
        }

        fn weights(&self, inject: bool) -> SiteWeights<'_> {
            SiteWeights {
                gamma: &self.gamma_d,
                down_qs: &self.down_qs,
                down_d: &self.down_d,
                up_qs: &self.up_qs,
                up_d: &self.up_d,
                inject: inject.then_some(&self.inject_d),
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
        let y_d = match pre {
            Pre::Plain => None,
            Pre::Combine(y) | Pre::Init(y) => Some(DeviceBuffer::from_host(st, y)?),
        };
        let before = match (pre, y_d.as_ref()) {
            (Pre::Combine(_), Some(y)) => Before::Combine { y },
            (Pre::Init(_), Some(y)) => Before::Init { y },
            _ => Before::Plain,
        };
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
        let site = Site::new(&c.gpu, 0x38c0_0001)?;
        let mut ok = true;
        ok &= check_band(&mut c, &site)?;
        ok &= check_stages(&mut c, &site)?;
        ok &= check_cols(&mut c, &site)?;
        ok &= check_combine(&mut c, &site)?;
        ok &= check_refuse(&mut c, &site)?;
        ok &= check_fault(&mut c, &site)?;
        ok &= check_wide(&mut c, &site)?;
        ok &= no_local_depot(&[
            "hc_gated_norm_4",
            "hc_gated_down_4x320",
            "hc_gated_up_mix_4x320",
            "hc_gated_combine_4",
            "hc_gated_norm_4w",
            "hc_gated_combine_4w",
            "hc_gated_lo_4x320",
            "hc_gated_mix_4",
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
             each column its own, a NaN raised; no local depot"
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
        let dp = per_col(&|k| card::down_parts(geo, &site.down, col(&o.xn, WIDE, k)));
        line("down_parts", &o.dpart, &dp);
        let lo = per_col(&|k| card::lo(geo, col(&o.dpart, R * S, k), ACT));
        line("lo", &o.lo, &lo);
        let ip = per_col(&|k| card::inject_parts(geo, &site.inject, col(&o.xn, WIDE, k)));
        line("inject_parts", &o.ipart, &ip);
        let w = per_col(&|k| card::wgt(geo, col(&o.ipart, S * S, k), ACT));
        line("wgt", &o.wgt, &w);
        let mx =
            per_col(&|k| card::up_mix(geo, &site.up, col(&o.lo, R, k), col(&o.xn, WIDE, k), ACT));
        line("mixed", &o.mixed, &mx);
        // The down partials through the q8f32 gemv on the view: stream s's
        // columns against every view row, rows j·S + s kept.
        let st = c.gpu.stream();
        let mut gemv = vec![0.0f32; m * R * S];
        for s in 0..S {
            let x: Vec<f32> = (0..m)
                .flat_map(|k| col(&o.xn, WIDE, k)[s * D..(s + 1) * D].to_vec())
                .collect();
            let x_d = DeviceBuffer::from_host(st, &x)?;
            let mut y_d = DeviceBuffer::from_host(st, &vec![0.0f32; R * S * m])?;
            c.q8.enqueue_q8_0_gemv_mcol(
                st,
                Q8_0GemvMcolArgs {
                    qs: &site.view_qs,
                    d: &site.view_d,
                    x: &x_d,
                    m,
                    out: GemvOut::RowMajor,
                    y: &mut y_d,
                },
            )?;
            let y = y_d.to_host_vec(st)?;
            for k in 0..m {
                for j in 0..R {
                    gemv[(k * R + j) * S + s] = y[(j * S + s) * m + k];
                }
            }
        }
        line("down_parts_vs_q8_0_gemv_mcol", &o.dpart, &gemv);
        // The up rows through the q8f32 gemv on the card's `lo`, then the gate
        // and the mean on the host over the card's `xn`.
        let lo_d = DeviceBuffer::from_host(st, &o.lo)?;
        let mut g_d = DeviceBuffer::from_host(st, &vec![0.0f32; WIDE * m])?;
        c.q8.enqueue_q8_0_gemv_mcol(
            st,
            Q8_0GemvMcolArgs {
                qs: &site.up_qs,
                d: &site.up_d,
                x: &lo_d,
                m,
                out: GemvOut::RowMajor,
                y: &mut g_d,
            },
        )?;
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
        line("mixed_vs_q8_0_gemv_mcol", &o.mixed, &mx);
        // The head after the site: no inject, its own stages, the site's
        // weights left in the scratch for the next combine.
        let head_res = activations(WIDE, 1, 0x57a7);
        let h = mix(c, site, false, &head_res, Pre::Plain, 1)?;
        let hx = card::norm(geo, &head_res, &site.gamma, EPS);
        let hmx = card::up_mix(geo, &site.up, &h.lo, &h.xn, ACT);
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
        w.down_qs = &site.up_qs;
        w.down_d = &site.up_d;
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
        let clean = check_types(GgmlType::F32, GgmlType::Q8_0, GgmlType::Q8_0, None).is_ok();
        ok &= clean;
        println!(
            "refuse[file_types] f32/q8_0/q8_0 accepted={clean} {}",
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
        Ok(p1 && p2)
    }

    // ------------------------------------------------------ 8. the wide arm

    /// Columns the wide scratch holds: past the eight-column cap, and an odd
    /// count for the independence clause.
    const WIDE_COLS: usize = 37;

    /// PIN(2026-09-28): the wide mix's distance from the f64 rule, derived,
    /// never measured: a q8 activation of 32 values (scale amax/127) moves a
    /// value by at most half a step, uniformly, d/√12 in RMS, so a column's
    /// RMS error over its RMS is at most √32/(127·√12) = 1.286e-2 (a block of
    /// one nonzero value, the largest crest); the down reads one such
    /// quantization (`xn`), the bottleneck `silu(·/4)` passes it with a slope
    /// of at most 1.1, so `lo` sits within 1.1 · 1.286e-2 = 1.415e-2 in RMS;
    /// the up reads a second (`lo`), so `g` sits within √2 · 1.415e-2 = 2.0e-2
    /// of its rule in RMS relative; `mixed` is Σ xn·σ(g)/4, whose relative
    /// error is at most the absolute error of `g` (σ'/σ = 1 − σ ≤ 1), so it
    /// sits within 2.0e-2 · rms(g). The inject is the F32 tile on `xn`: no
    /// quantization, the narrow arm's band.
    const WIDE_Q: f64 = 1.286e-2;
    const WIDE_LO_BAND: f64 = 1.1 * WIDE_Q;
    const WIDE_G_REL: f64 = std::f64::consts::SQRT_2 * WIDE_LO_BAND;

    /// The wide arm's kernels and scratch.
    struct Wide {
        hcw: HcWideKernels,
        g32: Gemm32Kernels,
        gemm: GemmKernels,
        route: GemmRoute,
        scratch: HcWideScratch,
    }

    /// Everything one wide mix leaves, read back, for its `m` columns.
    struct WideOut {
        res: Vec<f32>,
        xn: Vec<f32>,
        down: Vec<f32>,
        inj: Vec<f32>,
        lo: Vec<f32>,
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
        let y_d = match pre {
            Pre::Plain => None,
            Pre::Combine(y) | Pre::Init(y) => Some(DeviceBuffer::from_host(st, y)?),
        };
        let before = match (pre, y_d.as_ref()) {
            (Pre::Combine(_), Some(y)) => Before::Combine { y },
            (Pre::Init(_), Some(y)) => Before::Init { y },
            _ => Before::Plain,
        };
        w.gemm
            .enqueue_route_dense(st, m, &mut w.route, c.gpu.unlabelled_sink())?;
        w.hcw.enqueue_mix(
            st,
            WideMixArgs {
                res: &mut res_d,
                before,
                w: site.weights(inject),
                eps: EPS,
                m,
                fault: c.gpu.unlabelled_sink(),
                scratch: &mut w.scratch,
                gemm: &w.g32,
                dense: &w.route,
                mixed: &mut mixed_d,
            },
        )?;
        let s = &w.scratch;
        Ok(WideOut {
            res: res_d.to_host_vec(st)?,
            xn: s.xn.to_host_vec(st)?[..z.xn].to_vec(),
            down: s.down.to_host_vec(st)?[..z.lo].to_vec(),
            inj: s.inj.to_host_vec(st)?[..z.wgt].to_vec(),
            lo: s.lo.to_host_vec(st)?[..z.lo].to_vec(),
            g: s.g.to_host_vec(st)?[..z.xn].to_vec(),
            wgt: s.wgt.to_host_vec(st)?[..z.wgt].to_vec(),
            mixed: mixed_d.to_host_vec(st)?,
            fault: c.gpu.take_fault()?,
        })
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

    /// Clause 8 (module doc).
    fn check_wide(c: &mut Ctx, site: &Site) -> Result<bool, GateError> {
        let mut w = Wide {
            hcw: HcWideKernels::load(c.gpu.context())?,
            g32: Gemm32Kernels::load(c.gpu.context())?,
            gemm: GemmKernels::load(c.gpu.context())?,
            route: GemmRoute::new(c.gpu.stream(), WIDE_COLS, 1)?,
            scratch: HcWideScratch::new(c.gpu.stream(), c.geo, WIDE_COLS)?,
        };
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
        w.scratch.wgt.copy_from_host(c.gpu.stream(), &first.wgt)?;
        let wide = wide_mix(c, &mut w, site, true, &res, Pre::Combine(&y), m)?;
        let p = bits_equal(&wide.res, &narrow.res) && bits_equal(&wide.xn, &narrow.xn);
        ok &= p;
        println!(
            "wide[norm_combine] m={m} streams and xn the capped entries' bit_equal={p} {}",
            verdict(p)
        );
        let mut want = res.clone();
        combine(c.geo, m, &mut want, &y, &first.wgt);
        w.scratch.wgt.copy_from_host(c.gpu.stream(), &first.wgt)?;
        let st = c.gpu.stream();
        let mut res_d = DeviceBuffer::from_host(st, &res)?;
        let y_d = DeviceBuffer::from_host(st, &y)?;
        w.hcw
            .enqueue_combine(st, &mut res_d, &y_d, m, c.gpu.unlabelled_sink(), &w.scratch)?;
        let alone = res_d.to_host_vec(st)?;
        let f = c.gpu.take_fault()?;
        let p = bits_equal(&alone, &want) && f.is_none();
        ok &= p;
        println!(
            "wide[combine_alone] res the rule bit_equal={p} fault=\"{}\" {}",
            shown(f),
            verdict(p)
        );
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
        // (c) the error model's bands against the f64 rule, per column.
        let mut worst = [0.0f64; 4];
        let mut past = Vec::new();
        for k in 0..m {
            let r = mix_ref(c.geo, site.host(true), col(&res, WIDE, k), EPS);
            let as32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
            let g = col(&o.g, WIDE, k);
            let g_rms =
                (g.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / g.len() as f64).sqrt();
            let lines = [
                (
                    "xn",
                    f64::from(max_rel_err(col(&o.xn, WIDE, k), &as32(&r.xn))?),
                    f64::from(XN_BAND),
                ),
                ("lo", rms_rel(col(&o.lo, R, k), &as32(&r.lo)), WIDE_LO_BAND),
                (
                    "mixed",
                    rms_rel(col(&o.mixed, D, k), &as32(&r.mixed)),
                    WIDE_G_REL * g_rms,
                ),
                (
                    "wgt",
                    f64::from(max_rel_err(
                        col(&o.wgt, S, k),
                        &as32(r.wgt.as_deref().unwrap_or(&[])),
                    )?),
                    f64::from(LO_BAND),
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
        // (e) no silent failure: a NaN in one stream of one column raises (the
        // quantizer's QuantColumn, the lowest code the launches raise) and
        // leaves the other columns their clean bits; a column count past the
        // scratch is refused by name.
        let mut bad = res.clone();
        bad[2 * WIDE + D + 11] = f32::NAN;
        let o2 = wide_mix(c, &mut w, site, true, &bad, Pre::Plain, m)?;
        let want = Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
        let others = (0..m).filter(|&k| k != 2).all(|k| {
            bits_equal(col(&o2.mixed, D, k), col(&o.mixed, D, k))
                && bits_equal(col(&o2.wgt, S, k), col(&o.wgt, S, k))
        });
        let p = o2.fault == want && others;
        ok &= p;
        println!(
            "wide[fault] fault=\"{}\" want=\"{}\" the other columns clean={others} {}",
            shown(o2.fault),
            shown(want),
            verdict(p)
        );
        let past_cols = WIDE_COLS + 1;
        let res_long = activations(WIDE, past_cols, 0x3a04);
        let got = wide_mix(c, &mut w, site, true, &res_long, Pre::Plain, past_cols);
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
        Ok(ok)
    }
}
