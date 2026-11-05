//! GPU gate for Kimi K3's gated MLA layer on the card, model-less at the layer's shapes (2 and 96
//! heads, nope 128, 64 unrotated rope lanes beside the 512 latent, value 128) against an f64
//! reference written from llama.cpp's `build_mla_layer` in `src/models/kimi-k3.cpp`. Inputs are
//! seeded; nothing is read from a model file.
//!
//! The graph and the card's launches are one sequence. `Q` is the absorbed nope query
//! (`attn_k_b · q_nope`, 512 wide) then `q_pe` as it left `attn_q_b`; `K` is the normed latent
//! then `k_pe` as it left `attn_kv_a_mqa`; `V` is the normed latent alone; the scale is
//! `1/sqrt(nope + rope)`; there is no RoPE, so the 64 rope lanes stay in the dot unrotated; the
//! heads of `attn_v_b · softmax(QKᵀ) V` are multiplied by `sigmoid(attn_gate · x)` before
//! `attn_output`, with `x` the normed layer input. The cache is one 576-wide row a position, the
//! latent doing both the key's and the value's part. `latent_gated::attend` and `attend_rows` launch
//! exactly that — the row append, the per-head absorb, the rope gather, the latent flash, the
//! per-head value projection and the sigmoid gate — so the gate drives them at K3's geometry; the
//! file adds no kernel. The projections that feed `attend` (`attn_q_a`, its norm, `attn_q_b`,
//! `attn_kv_a_mqa`, `attn_gate`) and `attn_output` are dense products other gates own; here
//! their outputs are the seeded inputs, as the graph names them.
//!
//! 1. `decode`: `attend`, one row a step from position 0, at 2 and 96 heads and with `attn_v_b`
//!    as Q6_K and as Q8_0, over a cache of 64 rows (the single-launch pass; keys 1 and 7) and one
//!    of 1024 (the segment pass; keys 1, 7, 64, 65 and 300): the relative L2 of the gated heads
//!    against the graph in f64 within [`BAND`].
//! 2. `separation`: at 300 keys, references each missing one clause — the rope lanes, the lanes
//!    rotated by position (K3 has no RoPE), the scale at `1/sqrt(128)`, the gate — and the card
//!    run with the gate forced to 1 each lie past [`SEP`] bands of the card's gated heads.
//! 3. `prefill`: `attend_rows` over 8 rows (the pass's width) from positions 0 (on a 64-row and
//!    a 1024-row cache), 60 (the pass crosses a segment's edge) and 292: every row against the
//!    graph over its own causal prefix and against the decode step at its position on a cache of
//!    the same height (the single-launch and the segment pass are different arithmetic), each
//!    within [`BAND`]; the reference in which every row sees all the pass's keys lies past
//!    [`SEP`] bands of the pass's first seven rows.
//! 4. `refusals`: a shape the launches do not take is refused by name — the rope part not 64
//!    wide, a value head not 128, a cache not 576 wide, a latent gain not 512 long, an absorb
//!    whose nope is not the dims' 128, a value projection of another head count, a pass of more
//!    rows than the pass takes.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_mla", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::latent::LATENT;
    use bloomery_gpu::latent_gated::{
        AttendIn, AttnBufs, MlaDims, MlaKernels, ValueW, attend, attend_rows, partials_lens_rows,
    };
    use bloomery_gpu::linear::PASS_ROWS;
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
    use bloomery_gpu_gates::{GateError, GateRng, bits_equal, checks_failed, par_map, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};

    /// A query head's values before the absorption.
    const NOPE: usize = 128;
    /// The rope lanes beside the latent: unrotated, shared by the heads.
    const EXTRA: usize = 64;
    /// A head's value width.
    const V_HEAD: usize = 128;
    /// The cache row: the rope lanes, then the latent.
    const ROW: usize = EXTRA + LATENT;
    /// A query head's width as `attn_q_b` leaves it: `[nope | rope]`.
    const HEAD_K: usize = NOPE + EXTRA;
    /// `attention.layer_norm_rms_epsilon`, the one epsilon every K3 norm takes.
    const EPS: f32 = 1e-5;
    /// The heads of the fixture and of the real file.
    const HEADS: [usize; 2] = [2, 96];
    /// Positions a run covers at most.
    const N: usize = 300;
    /// The f16 cache (2^-11 a value) and the f16 query rows of the tensor-core pass each move a
    /// score or a value by under 5e-4 relative; 4e-3 is eight times that.
    const BAND: f64 = 4e-3;
    /// The bands a reference missing a clause must lie past.
    const SEP: f64 = 10.0;
    /// A gate row that sigmoid rounds to exactly 1 in f32.
    const GATE_ONE: f32 = 30.0;
    /// Each decode run: the cache's rows, the steps, the positions checked.
    const RUNS: [(usize, usize, &[usize]); 2] = [(64, 7, &[0, 6]), (1024, N, &[0, 6, 63, 64, 299])];
    /// Each pass: the first position and the cache's rows.
    const PASSES: [(usize, usize); 4] = [(0, 64), (0, 1024), (60, 1024), (292, 1024)];
    /// Q6_K bytes a 256-value super-block.
    const SB_BYTES: usize = 210;

    /// The layer's widths at `heads`; `attend` reads the heads, the nope, the rope lanes and the
    /// value width. The rest are the file's.
    fn dims(heads: usize) -> MlaDims {
        MlaDims {
            embd: 7168,
            heads,
            q_lora: 1536,
            nope: NOPE,
            extra: EXTRA,
            v_head: V_HEAD,
            rms_eps: EPS,
            latent_eps: EPS,
        }
    }

    fn fill(r: &mut GateRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| r.uni(lo, hi)).collect()
    }

    /// `‖a − b‖₂ / ‖b‖₂` in f64.
    fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - y).powi(2);
            den += y * y;
        }
        if a.len() != b.len() || !num.is_finite() {
            return f64::INFINITY;
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // ------------------------------------------------------------ the layer's weights and rows

    /// A per-head value projection as the card holds it.
    enum Planes {
        Q6K(Vec<u32>),
        Q8(Vec<u32>, Vec<u16>),
    }

    /// Which `attn_v_b` type a run uses.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Vb {
        Q6K,
        Q8,
    }

    impl Vb {
        const ALL: [Vb; 2] = [Vb::Q6K, Vb::Q8];

        fn name(self) -> &'static str {
            match self {
                Vb::Q6K => "Q6_K",
                Vb::Q8 => "Q8_0",
            }
        }
    }

    /// One layer's weights (the card's planes and their f64 dequantization) and `N` positions of
    /// its projections' outputs.
    struct Layer {
        heads: usize,
        /// `attn_k_b`, Q8_0: `[heads][LATENT][NOPE]`.
        kb_planes: Q8Planes,
        kb: Vec<f64>,
        /// `attn_v_b`: `[heads][V_HEAD][LATENT]`, as Q6_K and as Q8_0.
        v6: Planes,
        v6_f: Vec<f64>,
        v8: Planes,
        v8_f: Vec<f64>,
        /// `attn_kv_a_norm`.
        gain: Vec<f32>,
        /// `[N][ROW]`: `attn_kv_a_mqa`'s output as the card reads it, the latent then the rope
        /// lanes.
        kv: Vec<f32>,
        /// `[N][heads·HEAD_K]`: `attn_q_b`'s output, `[nope | rope]` a head.
        q: Vec<f32>,
        /// `[N][heads·V_HEAD]`: `attn_gate`'s output.
        gate: Vec<f32>,
    }

    /// Q8_0 planes as the card holds them: the quants as words, the f16 scales.
    type Q8Planes = (Vec<u32>, Vec<u16>);
    /// One decode run's saved outputs: (position, the gated heads' output row).
    type StepRows = Vec<(usize, Vec<f32>)>;

    /// `f` as Q8_0 planes and their f64 values.
    fn q8(f: &[f32]) -> Result<(Q8Planes, Vec<f64>), GateError> {
        let blocks: Vec<Q8Block> = f
            .as_chunks::<32>()
            .0
            .iter()
            .map(|c| Q8Block::quantize_ref(c).ok_or("a non-finite Q8_0 block"))
            .collect::<Result<_, _>>()?;
        let vals = blocks
            .iter()
            .flat_map(|b| {
                let s = f64::from(half_to_f32(b.d));
                b.q.map(|q| s * f64::from(q))
            })
            .collect();
        Ok((q8_0_planes(&blocks), vals))
    }

    /// `rows` Q6_K rows of `LATENT` values: random codes under a small finite scale, as words
    /// and as f64 values.
    fn q6k(r: &mut GateRng, rows: usize) -> Result<(Vec<u32>, Vec<f64>), GateError> {
        let row_bytes = LATENT / 256 * SB_BYTES;
        let mut bytes = vec![0u8; rows * row_bytes];
        for (i, by) in bytes.iter_mut().enumerate() {
            *by = (r.unit() * 256.0) as u8;
            // Each super-block's f16 `d` small and finite.
            if i % SB_BYTES == SB_BYTES - 1 {
                *by = 0x1c;
            }
        }
        let mut vals = Vec::with_capacity(rows * LATENT);
        let mut buf = vec![0.0f32; LATENT];
        for rr in 0..rows {
            dequant_row(
                GgmlType::Q6_K,
                &bytes[rr * row_bytes..(rr + 1) * row_bytes],
                &mut buf,
            )?;
            vals.extend(buf.iter().map(|&v| f64::from(v)));
        }
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Ok((words, vals))
    }

    impl Layer {
        fn new(seed: u64, heads: usize) -> Result<Layer, GateError> {
            let mut r = GateRng(seed);
            let kb_f = fill(&mut r, heads * LATENT * NOPE, -1.0, 1.0);
            let (kb_planes, kb) = q8(&kb_f)?;
            let v8_src = fill(&mut r, heads * V_HEAD * LATENT, -1.0, 1.0);
            let ((v8_qs, v8_d), v8_f) = q8(&v8_src)?;
            let (v6_words, v6_f) = q6k(&mut r, heads * V_HEAD)?;
            let mut kv = Vec::with_capacity(N * ROW);
            for _ in 0..N {
                kv.extend(fill(&mut r, LATENT, -1.0, 1.0));
                kv.extend(fill(&mut r, EXTRA, -2.0, 2.0));
            }
            let mut q = Vec::with_capacity(N * heads * HEAD_K);
            for _ in 0..N * heads {
                q.extend(fill(&mut r, NOPE, -0.25, 0.25));
                q.extend(fill(&mut r, EXTRA, -2.5, 2.5));
            }
            Ok(Layer {
                heads,
                kb_planes,
                kb,
                v6: Planes::Q6K(v6_words),
                v6_f,
                v8: Planes::Q8(v8_qs, v8_d),
                v8_f,
                gain: fill(&mut r, LATENT, 0.5, 1.5),
                kv,
                q,
                gate: fill(&mut r, N * heads * V_HEAD, -3.0, 3.0),
            })
        }

        fn vb_f(&self, vb: Vb) -> &[f64] {
            match vb {
                Vb::Q6K => &self.v6_f,
                Vb::Q8 => &self.v8_f,
            }
        }
    }

    // ------------------------------------------------------------------ the f64 reference

    /// Which clause of the graph a reference leaves out.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Drop {
        None,
        /// The rope lanes out of the dot.
        Extra,
        /// The rope lanes rotated by their positions.
        Rope,
        /// The score scale at `1/sqrt(nope)`.
        Scale,
        /// The output gate out.
        Gate,
    }

    /// The graph's per-position keys: `kv_cmpr` through `attn_kv_a_norm` (`x / sqrt(mean(x²) +
    /// eps) · gain`) and `k_pe` as it came.
    struct Keys {
        ln: Vec<Vec<f64>>,
        kpe: Vec<Vec<f64>>,
    }

    fn keys(l: &Layer, n: usize) -> Keys {
        let mut ln = Vec::with_capacity(n);
        let mut kpe = Vec::with_capacity(n);
        for t in 0..n {
            let row = &l.kv[t * ROW..(t + 1) * ROW];
            let lat = &row[..LATENT];
            let ms = lat.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / LATENT as f64;
            let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
            ln.push(
                lat.iter()
                    .zip(&l.gain)
                    .map(|(&x, &g)| f64::from(x) * inv * f64::from(g))
                    .collect(),
            );
            kpe.push(row[LATENT..].iter().map(|&x| f64::from(x)).collect());
        }
        Keys { ln, kpe }
    }

    /// `x`'s adjacent pairs rotated by `pos` at the rope base's per-pair angles.
    fn rotate(x: &[f64], pos: usize) -> Vec<f64> {
        let mut out = x.to_vec();
        for i in 0..x.len() / 2 {
            let theta = pos as f64 * 10_000f64.powf(-2.0 * i as f64 / x.len() as f64);
            let (s, c) = theta.sin_cos();
            out[2 * i] = x[2 * i] * c - x[2 * i + 1] * s;
            out[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c;
        }
        out
    }

    /// The graph for the query at position `p` over keys `0 .. nkeys`, gated:
    /// `[heads·V_HEAD]`.
    fn graph(l: &Layer, vb: Vb, ks: &Keys, (p, nkeys): (usize, usize), drop: Drop) -> Vec<f64> {
        let kpe: Vec<Vec<f64>> = (0..nkeys)
            .map(|t| {
                if drop == Drop::Rope {
                    rotate(&ks.kpe[t], t)
                } else {
                    ks.kpe[t].clone()
                }
            })
            .collect();
        let width = if drop == Drop::Scale { NOPE } else { HEAD_K };
        let scale = 1.0 / (width as f64).sqrt();
        let vb_f = l.vb_f(vb);
        let per_head = par_map(l.heads, |h| {
            let q = &l.q[(p * l.heads + h) * HEAD_K..(p * l.heads + h + 1) * HEAD_K];
            // q_nope_absorbed = wk_b · q_nope: one weight row a latent value.
            let qa: Vec<f64> = (0..LATENT)
                .map(|c| {
                    let row = &l.kb[(h * LATENT + c) * NOPE..(h * LATENT + c + 1) * NOPE];
                    row.iter()
                        .zip(&q[..NOPE])
                        .map(|(&a, &b)| a * f64::from(b))
                        .sum()
                })
                .collect();
            let q_pe: Vec<f64> = q[NOPE..].iter().map(|&x| f64::from(x)).collect();
            let q_pe = if drop == Drop::Rope {
                rotate(&q_pe, p)
            } else {
                q_pe
            };
            // Q·Kᵀ over K = concat(kv_cmpr, k_pe).
            let logits: Vec<f64> = (0..nkeys)
                .map(|t| {
                    let mut s: f64 = qa.iter().zip(&ks.ln[t]).map(|(a, b)| a * b).sum();
                    if drop != Drop::Extra {
                        s += q_pe.iter().zip(&kpe[t]).map(|(a, b)| a * b).sum::<f64>();
                    }
                    s * scale
                })
                .collect();
            let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logits.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            // The softmax-weighted latent: V = kv_cmpr.
            let mut lat = vec![0.0f64; LATENT];
            for (w, ln) in e.iter().zip(&ks.ln) {
                for (a, &b) in lat.iter_mut().zip(ln) {
                    *a += w * b;
                }
            }
            // wv_b per head, then the gate.
            (0..V_HEAD)
                .map(|r| {
                    let row = &vb_f[(h * V_HEAD + r) * LATENT..(h * V_HEAD + r + 1) * LATENT];
                    let v: f64 = row.iter().zip(&lat).map(|(a, b)| a * b).sum::<f64>() / z;
                    let g = if drop == Drop::Gate {
                        1.0
                    } else {
                        sigmoid(f64::from(l.gate[(p * l.heads + h) * V_HEAD + r]))
                    };
                    v * g
                })
                .collect::<Vec<f64>>()
        });
        per_head.into_iter().flatten().collect()
    }

    // ------------------------------------------------------------------ the card

    /// One layer's resident weights and kernels.
    struct Dev<'a> {
        gpu: &'a Gpu,
        l: &'a Layer,
        d: MlaDims,
        mk: MlaKernels,
        kb: (DeviceTensor<u32>, DeviceTensor<u16>),
        v6: DeviceTensor<u32>,
        v8: (DeviceTensor<u32>, DeviceTensor<u16>),
        gain: DeviceBuffer<f32>,
    }

    /// One run's cache, inputs and scratch, sized for a pass of [`PASS_ROWS`] rows.
    struct Session {
        cache: DeviceTensor<u16>,
        kv: DeviceBuffer<f32>,
        q: DeviceBuffer<f32>,
        gate: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        n_keys: DeviceBuffer<u32>,
        qabs: DeviceBuffer<f32>,
        att: DeviceBuffer<f32>,
        av: DeviceBuffer<f32>,
        part_v: DeviceBuffer<f32>,
        part_ms: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
    }

    impl<'a> Dev<'a> {
        fn new(gpu: &'a Gpu, l: &'a Layer) -> Result<Dev<'a>, GateError> {
            let s = gpu.stream();
            let d = dims(l.heads);
            let h = l.heads;
            let planes = |qs: &[u32], d: &[u16], rows: usize, k: usize| {
                Ok::<_, GateError>((
                    DeviceTensor::upload(s, qs, rows, k / 4)?,
                    DeviceTensor::upload(s, d, rows, k / 32)?,
                ))
            };
            let (Planes::Q6K(w6), Planes::Q8(q8s, q8d)) = (&l.v6, &l.v8) else {
                return Err("the layer's value planes are Q6_K then Q8_0".into());
            };
            Ok(Dev {
                gpu,
                l,
                d,
                mk: MlaKernels::load(gpu, &d, PASS_ROWS)?,
                kb: planes(&l.kb_planes.0, &l.kb_planes.1, h * LATENT, NOPE)?,
                v6: DeviceTensor::upload(s, w6, h * V_HEAD, LATENT / 256 * SB_BYTES / 4)?,
                v8: planes(q8s, q8d, h * V_HEAD, LATENT)?,
                gain: DeviceBuffer::from_host(s, &l.gain)?,
            })
        }

        fn session(&self, ctx: usize) -> Result<Session, GateError> {
            let s = self.gpu.stream();
            let (h, hv) = (self.l.heads, self.l.heads * V_HEAD);
            let z = |n: usize| DeviceBuffer::from_host(s, &vec![0.0f32; n]);
            let zu = |n: usize| DeviceBuffer::from_host(s, &vec![0u32; n]);
            let (pv, pm) = partials_lens_rows(&self.d, ctx, PASS_ROWS);
            Ok(Session {
                cache: DeviceTensor::<u16>::zeroed(s, ctx, ROW)?,
                kv: z(PASS_ROWS * ROW)?,
                q: z(PASS_ROWS * h * HEAD_K)?,
                gate: z(PASS_ROWS * hv)?,
                pos: zu(PASS_ROWS)?,
                n_keys: zu(PASS_ROWS)?,
                qabs: z(PASS_ROWS * h * ROW)?,
                att: z(PASS_ROWS * h * LATENT)?,
                av: z(PASS_ROWS * hv)?,
                part_v: z(pv)?,
                part_ms: z(pm)?,
                y: z(PASS_ROWS * hv)?,
            })
        }

        /// Stage rows `base .. base + m` of the inputs, the gate row forced to [`GATE_ONE`] when
        /// `gate_one`; the buffers hold a full pass, so the rows past `m` are zero.
        fn stage(
            &self,
            ss: &mut Session,
            (base, m): (usize, usize),
            gate_one: bool,
        ) -> Result<(), GateError> {
            let s = self.gpu.stream();
            let (h, hv) = (self.l.heads, self.l.heads * V_HEAD);
            let rows = |src: &[f32], width: usize| {
                let mut v = vec![0.0f32; PASS_ROWS * width];
                v[..m * width].copy_from_slice(&src[base * width..(base + m) * width]);
                v
            };
            ss.kv.copy_from_host(s, &rows(&self.l.kv, ROW))?;
            ss.q.copy_from_host(s, &rows(&self.l.q, h * HEAD_K))?;
            let mut gate = rows(&self.l.gate, hv);
            if gate_one {
                gate[..m * hv].fill(GATE_ONE);
            }
            ss.gate.copy_from_host(s, &gate)?;
            let mut pos = vec![0u32; PASS_ROWS];
            let mut keys = vec![0u32; PASS_ROWS];
            for t in 0..m {
                pos[t] = (base + t) as u32;
                keys[t] = (base + t) as u32 + 1;
            }
            ss.pos.copy_from_host(s, &pos)?;
            ss.n_keys.copy_from_host(s, &keys)?;
            Ok(())
        }

        fn value<'b>(&'b self, vb: Vb) -> ValueW<'b> {
            match vb {
                Vb::Q6K => ValueW::Q6K(&self.v6),
                Vb::Q8 => ValueW::Q8_0(&self.v8.0, &self.v8.1),
            }
        }

        /// `attend` over the session's staged row with the layer's own weights.
        fn step(&self, ss: &mut Session, vb: Vb) -> Result<(), GateError> {
            let k_b = (&self.kb.0, &self.kb.1);
            self.attend_with(ss, None, &self.gain, k_b, self.value(vb))?;
            Ok(())
        }

        /// `attend` over the session's staged row, any of its weights or its cache replaced: how
        /// the refusals hand it a shape it does not take.
        fn attend_with(
            &self,
            ss: &mut Session,
            cache: Option<&mut DeviceTensor<u16>>,
            kv_norm: &DeviceBuffer<f32>,
            k_b: (&DeviceTensor<u32>, &DeviceTensor<u16>),
            v_b: ValueW<'_>,
        ) -> Result<(), GpuError> {
            let Session {
                cache: own,
                kv,
                q,
                gate,
                pos,
                n_keys,
                qabs,
                att,
                av,
                part_v,
                part_ms,
                y,
            } = ss;
            attend(
                self.gpu,
                &self.mk,
                &self.d,
                AttendIn {
                    kv,
                    q,
                    gate,
                    kv_norm,
                    k_b,
                    v_b,
                    pos,
                    n_keys,
                    fault: self.gpu.unlabelled_sink(),
                },
                AttnBufs {
                    qabs,
                    att,
                    av,
                    part_v,
                    part_ms,
                },
                cache.unwrap_or(own),
                y,
            )
        }

        /// `attend_rows` over `m` staged rows.
        fn rows(&self, ss: &mut Session, vb: Vb, m: usize) -> Result<(), GateError> {
            attend_rows(
                self.gpu,
                &self.mk,
                &self.d,
                m,
                AttendIn {
                    kv: &ss.kv,
                    q: &ss.q,
                    gate: &ss.gate,
                    kv_norm: &self.gain,
                    k_b: (&self.kb.0, &self.kb.1),
                    v_b: self.value(vb),
                    pos: &ss.pos,
                    n_keys: &ss.n_keys,
                    fault: self.gpu.unlabelled_sink(),
                },
                AttnBufs {
                    qabs: &mut ss.qabs,
                    att: &mut ss.att,
                    av: &mut ss.av,
                    part_v: &mut ss.part_v,
                    part_ms: &mut ss.part_ms,
                },
                &mut ss.cache,
                &mut ss.y,
            )?;
            Ok(())
        }

        /// The first `m` rows of `y`.
        fn out(&self, ss: &Session, m: usize) -> Result<Vec<f32>, GateError> {
            let s = self.gpu.stream();
            s.synchronize()?;
            let mut y = ss.y.to_host_vec(s)?;
            y.truncate(m * self.l.heads * V_HEAD);
            Ok(y)
        }

        /// `steps` decode steps from position 0 on a fresh cache of `ctx` rows; the gated heads
        /// at each of `saves`.
        fn decode(
            &self,
            vb: Vb,
            (ctx, steps): (usize, usize),
            saves: &[usize],
            gate_one: bool,
        ) -> Result<Vec<(usize, Vec<f32>)>, GateError> {
            let mut ss = self.session(ctx)?;
            let mut out = Vec::new();
            for p in 0..steps {
                self.stage(&mut ss, (p, 1), gate_one)?;
                self.step(&mut ss, vb)?;
                if saves.contains(&p) {
                    out.push((p, self.out(&ss, 1)?));
                }
            }
            self.gpu.stream().synchronize()?;
            Ok(out)
        }

        /// The first `base` positions as decode steps, then one pass of `m` rows from `base`:
        /// the pass's gated heads, `[m][heads·V_HEAD]`.
        fn pass(
            &self,
            vb: Vb,
            ctx: usize,
            (base, m): (usize, usize),
        ) -> Result<Vec<f32>, GateError> {
            let mut ss = self.session(ctx)?;
            for p in 0..base {
                self.stage(&mut ss, (p, 1), false)?;
                self.step(&mut ss, vb)?;
            }
            self.stage(&mut ss, (base, m), false)?;
            self.rows(&mut ss, vb, m)?;
            self.out(&ss, m)
        }
    }

    // ------------------------------------------------------------------------- the clauses

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!(
            "gate_kimi_k3_mla: device {} — MLA at {HEADS:?} heads, row {ROW} (rope {EXTRA} + latent \
             {LATENT}), nope {NOPE}, value {V_HEAD}, synthetic",
            gpu.device_name()?
        );
        let mut results = Vec::new();
        for (i, heads) in HEADS.into_iter().enumerate() {
            let l = Layer::new(0x6b33_6d6c_6100_0000 + i as u64 + 1, heads)?;
            let dev = Dev::new(&gpu, &l)?;
            let ks = keys(&l, N);
            for vb in Vb::ALL {
                results.extend(decode_clauses(&dev, &ks, vb)?);
            }
            results.extend(separation_clauses(&dev, &ks)?);
            for vb in Vb::ALL {
                results.extend(prefill_clauses(&dev, &ks, vb)?);
            }
        }
        results.extend(refusal_clauses(&gpu)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_kimi_k3_mla: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    fn decode_clauses(dev: &Dev<'_>, ks: &Keys, vb: Vb) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        for (ctx, steps, at) in RUNS {
            let got = dev.decode(vb, (ctx, steps), at, false)?;
            for (p, y) in got {
                let err = rel_l2(&y, &graph(dev.l, vb, ks, (p, p + 1), Drop::None));
                let pass = err <= BAND && dev.gpu.fault()?.is_none();
                println!(
                    "  decode {} heads, attn_v_b {}, cache {ctx} rows, {} keys vs the graph in f64: \
                     rel-L2 {err:.3e} (band {BAND:.0e}): {}",
                    dev.l.heads,
                    vb.name(),
                    p + 1,
                    verdict(pass)
                );
                out.push(pass);
            }
        }
        Ok(out)
    }

    fn separation_clauses(dev: &Dev<'_>, ks: &Keys) -> Result<Vec<bool>, GateError> {
        let p = N - 1;
        let drops = [
            (Drop::Extra, "without the rope lanes"),
            (Drop::Rope, "with the rope lanes rotated"),
            (Drop::Scale, "at scale 1/sqrt(128)"),
            (Drop::Gate, "without the gate"),
        ];
        let mut seps = [f64::INFINITY; 5];
        for vb in Vb::ALL {
            let at = [p];
            let gated = dev.decode(vb, (1024, N), &at, false)?;
            for (sep, (drop, _)) in seps.iter_mut().zip(drops) {
                *sep = sep.min(rel_l2(&gated[0].1, &graph(dev.l, vb, ks, (p, p + 1), drop)));
            }
            let forced = dev.decode(vb, (1024, N), &at, true)?;
            let want = graph(dev.l, vb, ks, (p, p + 1), Drop::None);
            seps[4] = seps[4].min(rel_l2(&forced[0].1, &want));
        }
        let mut out = Vec::new();
        let names = drops.map(|(_, what)| format!("the card vs the reference {what}"));
        let names = names.into_iter().chain([String::from(
            "the card with the gate row forced to 1 vs the gated reference",
        )]);
        for (sep, what) in seps.iter().zip(names) {
            let pass = *sep > SEP * BAND;
            println!(
                "  separation {} heads, 300 keys: {what}: least rel-L2 {sep:.3e}, past {SEP}x band: {}",
                dev.l.heads,
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    fn prefill_clauses(dev: &Dev<'_>, ks: &Keys, vb: Vb) -> Result<Vec<bool>, GateError> {
        let m = PASS_ROWS;
        let hv = dev.l.heads * V_HEAD;
        let mut out = Vec::new();
        // The decode steps' own outputs at the positions the passes cover, on a cache of each
        // pass's height: the single-launch and the segment pass are different arithmetic.
        let mut steps: Vec<(usize, StepRows)> = Vec::new();
        for (_, ctx) in PASSES {
            if steps.iter().all(|(c, _)| *c != ctx) {
                let saves: Vec<usize> = PASSES
                    .iter()
                    .filter(|&&(_, c)| c == ctx)
                    .flat_map(|&(b, _)| b..b + m)
                    .collect();
                let last = saves.iter().max().map_or(0, |p| p + 1);
                steps.push((ctx, dev.decode(vb, (ctx, last), &saves, false)?));
            }
        }
        for (base, ctx) in PASSES {
            let y = dev.pass(vb, ctx, (base, m))?;
            let (mut vs_ref, mut vs_step) = (0.0f64, 0.0f64);
            let mut bits = true;
            for t in 0..m {
                let row = &y[t * hv..(t + 1) * hv];
                let want = graph(dev.l, vb, ks, (base + t, base + t + 1), Drop::None);
                vs_ref = vs_ref.max(rel_l2(row, &want));
                let step = &steps
                    .iter()
                    .find(|(c, _)| *c == ctx)
                    .and_then(|(_, at)| at.iter().find(|(p, _)| *p == base + t))
                    .ok_or("a decode step the pass compares to was not saved")?
                    .1;
                let step64: Vec<f64> = step.iter().map(|&v| f64::from(v)).collect();
                vs_step = vs_step.max(rel_l2(row, &step64));
                bits &= bits_equal(row, step);
            }
            let pass = vs_ref <= BAND && vs_step <= BAND && dev.gpu.fault()?.is_none();
            println!(
                "  prefill {} heads, attn_v_b {}, {m} rows from position {base}, cache {ctx} rows: \
                 worst row vs the graph {vs_ref:.3e}, vs the decode step {vs_step:.3e} (band \
                 {BAND:.0e}), bit-identical to the steps: {}: {}",
                dev.l.heads,
                vb.name(),
                if bits { "yes" } else { "no" },
                verdict(pass)
            );
            out.push(pass);
            if base == 0 && ctx == 1024 {
                // Every row seeing all the pass's keys: the first m-1 rows differ from the causal
                // ones.
                let (mut got, mut all) = (Vec::new(), Vec::new());
                for t in 0..m - 1 {
                    got.extend_from_slice(&y[t * hv..(t + 1) * hv]);
                    all.extend(graph(dev.l, vb, ks, (base + t, base + m), Drop::None));
                }
                let sep = rel_l2(&got, &all);
                let pass = sep > SEP * BAND;
                println!(
                    "  separation {} heads, attn_v_b {}: the pass's first {} rows vs the reference \
                     with no causal mask: rel-L2 {sep:.3e}, past {SEP}x band: {}",
                    dev.l.heads,
                    vb.name(),
                    m - 1,
                    verdict(pass)
                );
                out.push(pass);
            }
        }
        Ok(out)
    }

    /// The text of the refusal `r` is, or `None` when it ran.
    fn refused<T>(r: Result<T, GpuError>) -> Option<String> {
        r.err().map(|e| e.to_string())
    }

    fn refusal(what: &str, got: Option<String>, names: &str) -> bool {
        let pass = got.as_deref().is_some_and(|m| m.contains(names));
        println!(
            "  refusal {what}: {}: {}",
            got.as_deref().unwrap_or("ran without a refusal"),
            verdict(pass)
        );
        pass
    }

    fn refusal_clauses(gpu: &Gpu) -> Result<Vec<bool>, GateError> {
        let l = Layer::new(0x6b33_6d6c_61ff_0001, 2)?;
        let dev = Dev::new(gpu, &l)?;
        let s = gpu.stream();
        let mut out = Vec::new();
        let bad = MlaDims {
            extra: 32,
            ..dims(2)
        };
        out.push(refusal(
            "the rope part 32 wide (a 544 row)",
            refused(bad.check()),
            "576",
        ));
        out.push(refusal(
            "the load at a rope part of 32",
            refused(MlaKernels::load(gpu, &bad, PASS_ROWS)),
            "576",
        ));
        let bad = MlaDims {
            v_head: 64,
            ..dims(2)
        };
        out.push(refusal("a value head of 64", refused(bad.check()), "128"));
        // `attend` with one input of a wrong shape and the rest the layer's.
        let mut ss = dev.session(64)?;
        dev.stage(&mut ss, (0, 1), false)?;
        let mut narrow = DeviceTensor::<u16>::zeroed(s, 64, ROW - 32)?;
        let kv_norm = DeviceBuffer::from_host(s, &vec![1.0f32; LATENT / 2])?;
        let nope96 = (
            DeviceTensor::upload(s, &vec![0u32; 2 * LATENT * 24], 2 * LATENT, 24)?,
            DeviceTensor::upload(s, &vec![0u16; 2 * LATENT * 3], 2 * LATENT, 3)?,
        );
        let one_head = (
            DeviceTensor::upload(s, &vec![0u32; V_HEAD * LATENT / 4], V_HEAD, LATENT / 4)?,
            DeviceTensor::upload(s, &vec![0u16; V_HEAD * LATENT / 32], V_HEAD, LATENT / 32)?,
        );
        let good_kb = (&dev.kb.0, &dev.kb.1);
        out.push(refusal(
            "a cache 544 wide",
            refused(dev.attend_with(
                &mut ss,
                Some(&mut narrow),
                &dev.gain,
                good_kb,
                dev.value(Vb::Q8),
            )),
            "576",
        ));
        out.push(refusal(
            "a latent gain of 256 (kv_lora 256)",
            refused(dev.attend_with(&mut ss, None, &kv_norm, good_kb, dev.value(Vb::Q8))),
            "gain",
        ));
        out.push(refusal(
            "an absorb whose nope is 96",
            refused(dev.attend_with(
                &mut ss,
                None,
                &dev.gain,
                (&nope96.0, &nope96.1),
                dev.value(Vb::Q8),
            )),
            "nope",
        ));
        out.push(refusal(
            "a value projection of one head under two",
            refused(dev.attend_with(
                &mut ss,
                None,
                &dev.gain,
                good_kb,
                ValueW::Q8_0(&one_head.0, &one_head.1),
            )),
            "attn_v_b",
        ));
        let rows = PASS_ROWS + 1;
        let mut wide = dev.session(64)?;
        dev.stage(&mut wide, (0, PASS_ROWS), false)?;
        out.push(refusal(
            "a pass of one row more than the pass takes",
            refused(attend_rows(
                gpu,
                &dev.mk,
                &dev.d,
                rows,
                AttendIn {
                    kv: &wide.kv,
                    q: &wide.q,
                    gate: &wide.gate,
                    kv_norm: &dev.gain,
                    k_b: good_kb,
                    v_b: dev.value(Vb::Q8),
                    pos: &wide.pos,
                    n_keys: &wide.n_keys,
                    fault: gpu.unlabelled_sink(),
                },
                AttnBufs {
                    qabs: &mut wide.qabs,
                    att: &mut wide.att,
                    av: &mut wide.av,
                    part_v: &mut wide.part_v,
                    part_ms: &mut wide.part_ms,
                },
                &mut wide.cache,
                &mut wide.y,
            )),
            &format!("{rows} rows"),
        ));
        Ok(out)
    }
}
