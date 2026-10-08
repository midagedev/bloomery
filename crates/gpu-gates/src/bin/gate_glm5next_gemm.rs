//! GPU gate for GLM-5.3-Flash's prompt projections on the tensor-core GEMM
//! (`bloomery_gpu_glm5next::gemm::GemmFront`), on the model file's own
//! weights: only the projections tested are made resident (a KDA layer's
//! five, joined as the load joins them, a latent layer's stack, query heads
//! and output projection, and the dense lead's first block), never the
//! model. Layers: the dense lead's first ([`LEAD`], a KDA mixer and a dense
//! block), a routed KDA layer ([`KDA`]) and a latent layer ([`LATENT`]). The
//! inputs are fixed-seed columns, as many as a prompt batch holds (`T_MAX`),
//! and on the KDA layer first and on the dense block [`SHORT`], a batch's
//! tail.
//!
//! Clauses:
//! - (g1) `bits`: per projection, every output of every column bit for bit
//!   the host transcription of the GEMM's contract
//!   (`bloomery_gpu_gates::gemm32`, `gate_gemm`'s): the input quantized by
//!   the host quantizer, then per output the exact i32 block dots in the
//!   contract's order, over the file's Q8_0 blocks of the weight's rows (the
//!   joined `qkv` as `attn_q`, `attn_k`, `attn_v` in that order, the latent
//!   stack's two row ranges as rows `0 .. 1536` and `1536 .. 2304` of
//!   `attn_q_a`, `attn_kv_a_mqa`, `indexer.attn_k`,
//!   `indexer_compressor_gate`; the latent query heads `attn_q_b` into the
//!   front's query rows and the output `attn_output` from its heads' rows;
//!   the dense block's `ffn_gate`, `ffn_up` and `ffn_down` with the host's
//!   SwiGLU rows between, `qdot::swiglu_clamp` at the file's
//!   `swiglu_clamp_shexp` for the layer, quantized by the host quantizer),
//!   written token-major. `g_b` reads the host's own `g_a` rows and the down
//!   the host's own SwiGLU rows, so the chain is held, not only each link.
//!   Each output starts NaN, so a row or column left unwritten fails.
//! - (g2) `quantize`: one quantizer launch per distinct input — two for a
//!   KDA layer's input projections (`xn`, `g_a`'s rows), one for its output
//!   projection, one for the latent stack's two ranges, one each for the
//!   query heads and the output projection, two for a dense block (`xn` and
//!   the SwiGLU's) — and one GEMM per projection; the dense table filled
//!   once per new column count (three times over the calls).
//! - (g3) `refuse`: an F32 tensor where a Q8_0 projection belongs, one more
//!   column than the scratch holds, a projection of another K than its
//!   input's, query heads of other rows than the front's query rows, and a
//!   dense block on a front sized for none each give their named error,
//!   with no launch made.
//!
//! Tiers (`BLOOMERY_TIER`): every clause is a self-consistency clause (`tier::sc`) on the layers of the opened file
//! (layer 0 dense, layer 4 KDA, layer 3 latent), so the fixture tier runs them all on the fixture.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_gemm: built without the `glm5next` feature; see `just gate-gpu-glm5next-gemm`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_gemm", gate::run())
}

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
mod gate {
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::gemm32::{HostAct, dot32, host_act};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_glm5next::T_MAX;
    use bloomery_gpu_glm5next::gemm::{
        DenseNames, FrontShape, FrontStats, GemmFront, KdaInNames, KdaInRows, LatentInRows,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, half_to_f32};
    use model::arch::glm5next::hparams::Hparams;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::FrontWidths;

    use crate::glm5next_tier;

    /// The dense lead's first layer, a routed KDA layer and a latent layer.
    const LEAD: usize = 0;
    const KDA: usize = 4;
    const LATENT: usize = 3;

    /// A batch's short tail: no whole number of 64-column tiles.
    const SHORT: usize = 300;

    /// A 64-bit LCG (Knuth's MMIX constants).
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// `m` columns of `k` values from `seed`: a near-normal value (four
    /// uniforms summed) per entry, every 37th channel eight times larger, so
    /// a block's scale is set by an outlier in some blocks and not others.
    fn columns(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut g = Lcg(seed);
        (0..m * k)
            .map(|i| {
                let v = g.unit() + g.unit() + g.unit() + g.unit() - 2.0;
                if (i % k).is_multiple_of(37) {
                    8.0 * v
                } else {
                    v
                }
            })
            .collect()
    }

    /// A projection's rows as the contract reads them: per row `k` codes and
    /// `k/32` block scales, from the file's Q8_0 blocks.
    struct HostW {
        k: usize,
        rows: usize,
        q: Vec<i8>,
        d: Vec<f32>,
    }

    impl HostW {
        /// The file tensors `parts`, rows in that order (the load's join).
        fn of(file: &Split, parts: &[String]) -> Result<HostW, GateError> {
            let mut w = HostW {
                k: 0,
                rows: 0,
                q: Vec::new(),
                d: Vec::new(),
            };
            for name in parts {
                let (shard, t) = file
                    .find(name)
                    .ok_or_else(|| format!("{name} is not in the file"))?;
                if t.ty != GgmlType::Q8_0 {
                    return Err(format!("{name} is {}, not Q8_0", t.ty).into());
                }
                let k = t.dims[0] as usize;
                let rows = t.dims[1..].iter().product::<u64>() as usize;
                if w.k != 0 && w.k != k {
                    return Err(format!("{name} has K {k}, the join's first part {}", w.k).into());
                }
                w.k = k;
                w.rows += rows;
                let data = file
                    .shard(shard)
                    .ok_or_else(|| format!("{name}: shard {shard} missing"))?
                    .data(t)?;
                for b in data.as_chunks::<34>().0 {
                    w.d.push(half_to_f32(u16::from_le_bytes([b[0], b[1]])));
                    w.q.extend(b[2..].iter().map(|&c| c as i8));
                }
            }
            if w.q.len() != w.rows * w.k {
                return Err(format!("{parts:?}: {} codes for {} rows", w.q.len(), w.rows).into());
            }
            Ok(w)
        }

        /// The contract's outputs of rows `r0 .. r0 + n` over every column
        /// of `a` (`m` columns), token-major (`y[c · n + r]`).
        fn project(&self, a: &HostAct, m: usize, (r0, n): (usize, usize)) -> Vec<f32> {
            let (k, kb) = (self.k, self.k / 32);
            let mut y = vec![0.0f32; m * n];
            let threads = std::thread::available_parallelism().map_or(8, |t| t.get().min(32));
            let per = m.div_ceil(threads);
            std::thread::scope(|sc| {
                for (i, part) in y.chunks_mut(per * n).enumerate() {
                    sc.spawn(move || {
                        for (j, out) in part.chunks_mut(n).enumerate() {
                            let c = i * per + j;
                            for (r, o) in out.iter_mut().enumerate() {
                                let row = r0 + r;
                                let wq = &self.q[row * k..(row + 1) * k];
                                let wd = &self.d[row * kb..(row + 1) * kb];
                                *o = dot32(wq, wd, &[], a, c, false).0;
                            }
                        }
                    });
                }
            });
            y
        }
    }

    /// `n` f32 on the card, every one NaN.
    fn nan_buf(stream: &CudaStream, n: usize) -> Result<DeviceBuffer<f32>, GateError> {
        Ok(DeviceBuffer::from_host(stream, &vec![f32::NAN; n])?)
    }

    /// One clause line, and whether it passed.
    fn line(what: &str, pass: bool, detail: &str) -> bool {
        println!("{what}: {detail} {}", verdict(pass));
        pass
    }

    /// The card's outputs against the host's, bit for bit: a line per
    /// projection naming the first output that differs.
    fn same(what: &str, got: &[f32], want: &[f32], rows: usize) -> bool {
        let off = got
            .iter()
            .zip(want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        let first = got
            .iter()
            .zip(want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
            .map_or(String::new(), |i| {
                format!(
                    " first at column {} row {}: card {:e} host {:e}",
                    i / rows,
                    i % rows,
                    got[i],
                    want[i]
                )
            });
        line(
            &format!("g1 bits {what}"),
            off == 0 && got.len() == want.len(),
            &format!("{} outputs, {off} off the contract{first}", want.len()),
        )
    }

    /// The launch counts `after` less `before`.
    fn delta(before: FrontStats, after: FrontStats) -> FrontStats {
        FrontStats {
            quantize: after.quantize - before.quantize,
            gemm: after.gemm - before.gemm,
            route: after.route - before.route,
        }
    }

    /// The names a KDA layer's input projections read.
    struct KdaNames {
        qkv: String,
        g_a: String,
        beta: String,
        g_b: String,
        out: String,
    }

    impl KdaNames {
        fn of(l: usize) -> KdaNames {
            KdaNames {
                qkv: names::attn_qkv(l),
                g_a: names::ssm_g_a(l),
                beta: names::ssm_beta(l),
                g_b: names::ssm_g_b(l),
                out: names::attn_output(l),
            }
        }

        fn input(&self) -> KdaInNames<'_> {
            KdaInNames {
                qkv: &self.qkv,
                g_a: &self.g_a,
                beta: &self.beta,
                g_b: &self.g_b,
            }
        }
    }

    /// A latent layer's four projections of its normed input, in the joined
    /// stack's order (`Body::derive`'s).
    fn stack_parts(l: usize) -> [String; 4] {
        [
            names::attn_q_a(l),
            names::attn_kv_a_mqa(l),
            names::indexer_attn_k(l),
            names::indexer_compressor_gate(l),
        ]
    }

    /// The file tensors this gate makes resident, joined as the load joins
    /// them (`Body::derive`): a KDA layer's q, k and v into `attn_qkv`, a
    /// latent layer's four into `attn_a_stack`, its query heads and output
    /// projection, the dense lead's first block; and for the refusals the
    /// KDA layer's F32 norm and the latent layer's indexer query (K of the
    /// query low rank, other rows).
    fn load(stream: &CudaStream, file: &Split) -> Result<Weights, GateError> {
        let mut keep: Vec<String> = Vec::new();
        for l in [LEAD, KDA] {
            keep.extend([names::attn_q(l), names::attn_k(l), names::attn_v(l)]);
            keep.extend([
                names::ssm_g_a(l),
                names::ssm_beta(l),
                names::ssm_g_b(l),
                names::attn_output(l),
            ]);
        }
        keep.extend(stack_parts(LATENT));
        keep.extend([
            names::attn_q_b(LATENT),
            names::attn_output(LATENT),
            names::indexer_attn_q_b(LATENT),
        ]);
        keep.extend([
            names::ffn_gate(LEAD),
            names::ffn_up(LEAD),
            names::ffn_down(LEAD),
        ]);
        keep.push(names::attn_norm(KDA));
        let mut w = Weights::load_where(stream, file, |n| keep.iter().any(|k| k == n))?;
        for l in [LEAD, KDA] {
            let (q, k, v) = (names::attn_q(l), names::attn_k(l), names::attn_v(l));
            w.join_rows(stream, &[&q, &k, &v], names::attn_qkv(l))?;
        }
        let parts = stack_parts(LATENT);
        let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
        w.join_rows(stream, &parts, names::attn_a_stack(LATENT))?;
        Ok(w)
    }

    /// Clauses g1 and g2 on KDA layer `l` over `m` columns.
    fn kda_layer(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        front: &mut GemmFront,
        (l, m): (usize, usize),
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let nm = KdaNames::of(l);
        let qkv = HostW::of(
            file,
            &[names::attn_q(l), names::attn_k(l), names::attn_v(l)],
        )?;
        let [g_a, beta, g_b, out] =
            [&nm.g_a, &nm.beta, &nm.g_b, &nm.out].map(|n| HostW::of(file, std::slice::from_ref(n)));
        let (g_a, beta, g_b, out) = (g_a?, beta?, g_b?, out?);
        let s = front.shape();
        let seed = (l * 1000 + m) as u64;
        let xn = columns(m, s.widths.embd, 0x5eed_0000 + seed);
        let gated = columns(m, s.widths.gated, 0x6a7e_0000 + seed);
        let xn_d = DeviceBuffer::from_host(stream, &xn)?;
        let gated_d = DeviceBuffer::from_host(stream, &gated)?;
        let mut ys: Vec<DeviceBuffer<f32>> = [&qkv, &g_a, &beta, &g_b, &out]
            .iter()
            .map(|h| nan_buf(stream, m * h.rows))
            .collect::<Result<_, _>>()?;
        let before = front.stats();
        {
            let [y_qkv, y_ga, y_beta, y_z, _] = &mut ys[..] else {
                return Err("five outputs".into());
            };
            front.kda_in(
                gpu,
                w,
                l,
                m,
                &xn_d,
                nm.input(),
                KdaInRows {
                    qkv: y_qkv,
                    ga: y_ga,
                    beta_raw: y_beta,
                    z: y_z,
                },
            )?;
        }
        let mid = front.stats();
        front.kda_out(gpu, w, l, m, &gated_d, &nm.out, &mut ys[4])?;
        let after = front.stats();
        stream.synchronize()?;
        if let Some(f) = gpu.fault()? {
            return Err(format!("layer {l}: the front raised {f:?}").into());
        }
        let got: Vec<Vec<f32>> = ys
            .iter()
            .map(|y| y.to_host_vec(stream))
            .collect::<Result<_, _>>()?;

        let (a_xn, _) = host_act(&xn, s.widths.embd, m);
        let want_ga = g_a.project(&a_xn, m, (0, g_a.rows));
        let (a_ga, _) = host_act(&want_ga, s.widths.low, m);
        let (a_gated, _) = host_act(&gated, s.widths.gated, m);
        let want = [
            ("qkv", qkv.project(&a_xn, m, (0, qkv.rows)), qkv.rows),
            ("g_a", want_ga, g_a.rows),
            ("beta", beta.project(&a_xn, m, (0, beta.rows)), beta.rows),
            ("g_b", g_b.project(&a_ga, m, (0, g_b.rows)), g_b.rows),
            ("out", out.project(&a_gated, m, (0, out.rows)), out.rows),
        ];
        let mut pass = true;
        for ((what, want, rows), got) in want.iter().zip(&got) {
            pass &= same(&format!("layer {l} m={m} {what}"), got, want, *rows);
        }
        let (d_in, d_out) = (delta(before, mid), delta(mid, after));
        let ok = d_in.quantize == 2 && d_in.gemm == 4 && d_out.quantize == 1 && d_out.gemm == 1;
        pass &= line(
            &format!("g2 quantize layer {l} m={m}"),
            ok,
            &format!(
                "input projections {} quantize / {} gemm (want 2 / 4), output {} / {} (want 1 / 1); \
                 {} table fills",
                d_in.quantize, d_in.gemm, d_out.quantize, d_out.gemm, d_in.route
            ),
        );
        Ok(pass)
    }

    /// Clauses g1 and g2 on latent layer `l`'s stack.
    fn latent_layer(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        front: &mut GemmFront,
        l: usize,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let parts = stack_parts(l);
        let stack = HostW::of(file, &parts)?;
        let q_lora = HostW::of(file, &parts[..1])?.rows;
        let kv = stack.rows - q_lora;
        let s = front.shape();
        let m = s.cols;
        let xn = columns(m, s.widths.embd, 0x1a7e_0000 + l as u64);
        let xn_d = DeviceBuffer::from_host(stream, &xn)?;
        let mut qa = nan_buf(stream, m * q_lora)?;
        let mut kv_d = nan_buf(stream, m * kv)?;
        let before = front.stats();
        front.latent_in(
            gpu,
            w,
            l,
            m,
            &xn_d,
            (&names::attn_a_stack(l), q_lora, kv),
            LatentInRows {
                qa: &mut qa,
                kv: &mut kv_d,
            },
        )?;
        let d = delta(before, front.stats());
        stream.synchronize()?;
        if let Some(f) = gpu.fault()? {
            return Err(format!("layer {l}: the front raised {f:?}").into());
        }
        let (a, _) = host_act(&xn, s.widths.embd, m);
        let mut pass = same(
            &format!("layer {l} q_a"),
            &qa.to_host_vec(stream)?,
            &stack.project(&a, m, (0, q_lora)),
            q_lora,
        );
        pass &= same(
            &format!("layer {l} kv_a·index_k·gate"),
            &kv_d.to_host_vec(stream)?,
            &stack.project(&a, m, (q_lora, kv)),
            kv,
        );
        pass &= line(
            &format!("g2 quantize layer {l}"),
            d.quantize == 1 && d.gemm == 2 && d.route == 0,
            &format!(
                "stack {} quantize / {} gemm / {} table fills (want 1 / 2 / 0: the batch's table stands)",
                d.quantize, d.gemm, d.route
            ),
        );
        Ok(pass)
    }

    /// Clauses g1 and g2 on latent layer `l`'s query heads and output
    /// projection over the front's rows: the query heads from fixed-seed
    /// query low-rank columns into the query rows, then the output from
    /// fixed-seed heads' outputs written into the front's rows.
    fn latent_heads(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        front: &mut GemmFront,
        l: usize,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let q_b = HostW::of(file, &[names::attn_q_b(l)])?;
        let out = HostW::of(file, &[names::attn_output(l)])?;
        let s = front.shape();
        let (m, fw) = (s.cols, s.widths);
        let qr = columns(m, fw.q_low, 0x9b00_0000 + l as u64);
        let av = columns(m, fw.heads_v, 0xa700_0000 + l as u64);
        let qr_d = DeviceBuffer::from_host(stream, &qr)?;
        {
            let (q, av_d) = front.heads_rows();
            q.copy_from_host(stream, &vec![f32::NAN; q.len()])?;
            av_d.copy_from_host(stream, &av)?;
        }
        let mut y = nan_buf(stream, m * out.rows)?;
        let before = front.stats();
        front.latent_q(gpu, w, (l, m), &qr_d, &names::attn_q_b(l))?;
        let mid = front.stats();
        front.latent_out(gpu, w, (l, m), &names::attn_output(l), &mut y)?;
        let after = front.stats();
        stream.synchronize()?;
        if let Some(f) = gpu.fault()? {
            return Err(format!("layer {l}: the front raised {f:?}").into());
        }
        let got_q = front.heads_rows().0.to_host_vec(stream)?;
        let (a_qr, _) = host_act(&qr, fw.q_low, m);
        let (a_av, _) = host_act(&av, fw.heads_v, m);
        let mut pass = same(
            &format!("layer {l} q_b"),
            &got_q,
            &q_b.project(&a_qr, m, (0, q_b.rows)),
            q_b.rows,
        );
        pass &= same(
            &format!("layer {l} out"),
            &y.to_host_vec(stream)?,
            &out.project(&a_av, m, (0, out.rows)),
            out.rows,
        );
        let (d_q, d_out) = (delta(before, mid), delta(mid, after));
        pass &= line(
            &format!("g2 quantize layer {l} heads"),
            d_q.quantize == 1
                && d_q.gemm == 1
                && d_out.quantize == 1
                && d_out.gemm == 1
                && d_q.route + d_out.route == 0,
            &format!(
                "query heads {} quantize / {} gemm, output {} / {} (want 1 / 1 each); {} table \
                 fills (want 0: the batch's table stands)",
                d_q.quantize,
                d_q.gemm,
                d_out.quantize,
                d_out.gemm,
                d_q.route + d_out.route
            ),
        );
        Ok(pass)
    }

    /// Clauses g1 and g2 on layer `l`'s dense block over `m` columns at the
    /// SwiGLU limit `limit`.
    fn dense_block(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        front: &mut GemmFront,
        (l, m): (usize, usize),
        limit: f32,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (gate_n, up_n, down_n) = (names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l));
        let [gate, up, down] =
            [&gate_n, &up_n, &down_n].map(|n| HostW::of(file, std::slice::from_ref(n)));
        let (gate, up, down) = (gate?, up?, down?);
        let fw = front.shape().widths;
        let xn = columns(m, fw.embd, 0xdf00_0000 + (l * 1000 + m) as u64);
        let xn_d = DeviceBuffer::from_host(stream, &xn)?;
        let mut y = nan_buf(stream, m * down.rows)?;
        let before = front.stats();
        front.dense_ffn(
            gpu,
            w,
            l,
            m,
            &xn_d,
            DenseNames {
                gate: &gate_n,
                up: &up_n,
                down: &down_n,
            },
            limit,
            &mut y,
        )?;
        let d = delta(before, front.stats());
        stream.synchronize()?;
        if let Some(f) = gpu.fault()? {
            return Err(format!("layer {l}: the front raised {f:?}").into());
        }
        let (a_xn, _) = host_act(&xn, fw.embd, m);
        let g = gate.project(&a_xn, m, (0, gate.rows));
        let u = up.project(&a_xn, m, (0, up.rows));
        let mut h = vec![0.0f32; g.len()];
        qdot::swiglu_clamp(&g, &u, limit, &mut h);
        let (a_h, _) = host_act(&h, fw.ff, m);
        let mut pass = same(
            &format!("layer {l} m={m} dense gate·up·SwiGLU·down at limit {limit}"),
            &y.to_host_vec(stream)?,
            &down.project(&a_h, m, (0, down.rows)),
            down.rows,
        );
        pass &= line(
            &format!("g2 quantize layer {l} m={m} dense"),
            d.quantize == 2 && d.gemm == 3,
            &format!(
                "{} quantize / {} gemm (want 2 / 3: xn and the SwiGLU rows; gate, up, down); {} \
                 table fills",
                d.quantize, d.gemm, d.route
            ),
        );
        Ok(pass)
    }

    /// Clause g3: each refusal names its cause and makes no launch.
    fn refusals(gpu: &Gpu, w: &Weights, front: &mut GemmFront) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let s = front.shape();
        let nm = KdaNames::of(KDA);
        let norm = names::attn_norm(KDA);
        let gated = nan_buf(stream, (s.cols + 1) * s.widths.gated)?;
        let mut y = nan_buf(stream, (s.cols + 1) * 4 * s.widths.gated)?;
        let mut pass = true;
        // A case: the error's text must hold `needle`; the launch counts are
        // held unmoved over all three after them.
        let case = |what: &str, needle: &str, r: Result<(), GpuError>| -> bool {
            let msg = match r {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            };
            line(
                &format!("g3 refuse {what}"),
                msg.contains(needle),
                &format!("{msg:?} (want {needle:?})"),
            )
        };
        let before = front.stats();
        let r = front.kda_out(gpu, w, KDA, s.cols, &gated, &norm, &mut y);
        pass &= case("f32 weight", "is not a q8_0 weight", r);
        let r = front.kda_out(gpu, w, KDA, s.cols + 1, &gated, &nm.out, &mut y);
        pass &= case("oversize m", "the scratch holds", r);
        let r = front.kda_out(gpu, w, KDA, s.cols, &gated, &nm.qkv, &mut y);
        pass &= case("other K", "activations were sized for K", r);
        let qr = nan_buf(stream, s.cols * s.widths.q_low)?;
        let r = front.latent_q(
            gpu,
            w,
            (LATENT, s.cols),
            &qr,
            &names::indexer_attn_q_b(LATENT),
        );
        pass &= case("other rows", "query rows were sized for", r);
        let after = front.stats();
        pass &= line(
            "g3 refuse launches",
            after == before,
            &format!("{before:?} -> {after:?} (want unchanged)"),
        );
        // A front sized for no dense block: the block refused by name.
        let mut bare = GemmFront::open(
            gpu,
            FrontShape {
                cols: 1,
                widths: FrontWidths { ff: 0, ..s.widths },
            },
        )?;
        let xn = nan_buf(stream, s.widths.embd)?;
        let mut y1 = nan_buf(stream, s.widths.embd)?;
        let r = bare.dense_ffn(
            gpu,
            w,
            LEAD,
            1,
            &xn,
            DenseNames {
                gate: &names::ffn_gate(LEAD),
                up: &names::ffn_up(LEAD),
                down: &names::ffn_down(LEAD),
            },
            0.0,
            &mut y1,
        );
        pass &= case("no dense width", "on a front sized for none", r);
        pass &= line(
            "g3 refuse launches no dense width",
            bare.stats() == FrontStats::default(),
            &format!("{:?} (want none)", bare.stats()),
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let file = glm5next_tier::open()?;
        let w = load(stream, &file)?;
        let hp = Hparams::read(&file)?;
        let low = HostW::of(&file, &[names::ssm_g_a(KDA)])?.rows;
        let qkv = names::attn_qkv(KDA);
        let out = names::attn_output(KDA);
        let k_of = |name: &str| -> Result<usize, GateError> {
            match w.get(name) {
                Some(DevWeight::Q8_0 { k, .. }) => Ok(*k),
                _ => Err(format!("{name} is not resident as Q8_0").into()),
            }
        };
        let rows_of = |name: &str| -> Result<usize, GateError> {
            match w.get(name) {
                Some(q @ DevWeight::Q8_0 { .. }) => Ok(q.rows()),
                _ => Err(format!("{name} is not resident as Q8_0").into()),
            }
        };
        let shape = FrontShape {
            cols: T_MAX,
            widths: FrontWidths {
                embd: k_of(&qkv)?,
                low,
                gated: k_of(&out)?,
                q_low: k_of(&names::attn_q_b(LATENT))?,
                q: rows_of(&names::attn_q_b(LATENT))?,
                heads_v: k_of(&names::attn_output(LATENT))?,
                ff: k_of(&names::ffn_down(LEAD))?,
            },
        };
        let mut front = GemmFront::open(&gpu, shape)?;
        println!(
            "front: {shape:?}, {} device bytes; layers lead {LEAD}, kda {KDA}, latent {LATENT}",
            front.bytes()
        );
        let mut pass = true;
        // A short count first, then the batch's: the table refilled for each
        // new count, every column of both written.
        tier::sc("(g1) bits: every projection bit for bit the host transcription of the contract")?;
        tier::sc("(g2) quantize: one launch per distinct input, one GEMM per projection")?;
        tier::sc("(g3) refuse: the named errors, with no launch made")?;
        for at in [(KDA, SHORT), (LEAD, T_MAX), (KDA, T_MAX)] {
            pass &= kda_layer(&gpu, &file, &w, &mut front, at)?;
        }
        pass &= latent_layer(&gpu, &file, &w, &mut front, LATENT)?;
        pass &= latent_heads(&gpu, &file, &w, &mut front, LATENT)?;
        let limit = *hp
            .limit_shexp
            .get(LEAD)
            .ok_or_else(|| format!("no swiglu_clamp_shexp for layer {LEAD}"))?;
        pass &= dense_block(&gpu, &file, &w, &mut front, (LEAD, SHORT), limit)?;
        let fills = front.stats().route;
        pass &= line(
            "g2 table fills",
            fills == 3,
            &format!(
                "{fills} over the calls at column counts {SHORT}, then {T_MAX}, then {SHORT} \
                 (want 3)"
            ),
        );
        pass &= refusals(&gpu, &w, &mut front)?;
        println!("gate_glm5next_gemm: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_glm5next_gemm — every projection bit for bit the contract, one quantize per input, every refusal named"
        );
        Ok(())
    }
}
