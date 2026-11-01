//! GPU gate for Qwen3.6's DSpark draft (RedHatAI's speculator, an HF directory of `config.json` and
//! `model.safetensors`).
//!
//! 0. The dense layers, model-less: the real draft's layers on the card against a host f64
//!    computation of the same layers from the file's bf16 weights (no numpy, no card). A block of 8
//!    rows over a ring of `R` context rows, `R` = 3 and `R` = 2100 (past the window of 2048, the
//!    ring wrapped; the reference reads the last 2048 context rows only), under the file's mask (the
//!    block causal in its sliding layers). The context rows are appended in chunks of changing size,
//!    so one append runs past the ring's end. The captured block pass replays to the eager bits.
//!    The loader opens the whole draft (fc, head, Markov tables, id map, embedding table) and the
//!    card bytes it reports are the header-only count. Staging: on fresh layers an eager block pass
//!    and an eager append, run before their stage call, are refused by name; a replay before any
//!    stage call is refused by name, and a capture before any stage call records (the owner
//!    captures at load). After one stage call the replay is allowed.
//!
//! 1. Lossless. Greedy from one fixed prompt, the first 64 ids of
//!    `$BLOOMERY_DATA/qwen35moe/corpus-prose.ids` (Qwen3.6's own vocabulary). The draft run's first
//!    256 decoded ids equal a plain `Qwen35moeModel::open` of the same file with no draft. The line
//!    is `ids identical 256`, or the first index where the two runs differ.
//!
//! 2. Acceptance floor, over the first 512 ids of that file as the prompt walk, decoding 128 tokens
//!    at proposal width 7. E is the committed tokens per verify pass, and the clause holds at
//!    E ≥ 2.0. RedHat's card reports the acceptance its training reached; a port that misreads the
//!    taps, the map or the mask falls near E ≈ 1.
//!
//!     gate_qwen35_dspark [--draft <dir>] [--only 0|1|2]...
//!
//! `--draft` defaults to the box's copy. With no `--only`, every clause runs; each `--only` selects
//! one clause.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35_dspark", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/dspark_hf.rs"]
mod dspark_hf;

#[cfg(feature = "gpu")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "gpu")]
#[path = "shared/greedy.rs"]
mod greedy;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use app::Session;
    use app::dspark::CardDspark;
    use bloomery_gpu::arch::qwen3moe::{Body35, Open35, Qwen35moeModel};
    use bloomery_gpu::dspark::dense::{DenseDims, DenseLayers};
    use bloomery_gpu::dspark::head::HeadTail;
    use bloomery_gpu::dspark::{DraftLayers, DsparkBody, HeadSource, Q8Rows};
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::rounding::U;
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, prose_ids, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::quant::quantize_q8_0;
    use refset::arch::qwen35moe::MODEL;
    use runtime::dspark::DraftVocab;
    use runtime::{Advance, Draft, Speculative, Target, Verify};

    use crate::dspark_hf::{self, HfFile, Parts};
    use crate::greedy::greedy;
    use crate::quiet::Quiet;

    /// The context rows of the two cases: a ring that is nearly empty and one past the window.
    const CASES: [usize; 2] = [3, 2100];
    /// The contexts the draft is opened for.
    const CTX: usize = 4096;
    /// Sizes the context rows are appended in, in turn: the append that starts at row 2,045 is 7 rows
    /// and runs across the ring's end (the clause counts it).
    const CHUNKS: [usize; 6] = [8, 3, 5, 1, 7, 2];
    /// The target's position of the first context row of each case.
    const FIRST: u32 = 0;
    /// The token ids of the block's row 0 (the others are the mask token).
    const ANCHOR_ID: usize = 1000;
    /// The rows of a block pass and of an append: the layers' widest.
    const BLOCK: usize = <DenseLayers as DraftLayers>::MAX_ROWS;
    const APPEND: usize = <DenseLayers as DraftLayers>::APPEND_GROUP;
    /// The Q8_0 rule's error per weight, in the block's scale `d = amax/127` (the module doc of
    /// `decision`'s Q8_0 head bound has the same two terms): the rounding `d/2`, and the scale's own
    /// rounding to f16, `|q| · |d − f16(d)| <= 127 · d · 2^-11`.
    const Q8_ERR: f64 = 0.5 + 127.0 / 2048.0;

    /// Clause 1's prompt: the first ids of the Qwen3.6 prose file, one ubatch.
    const PROMPT_FIXED: usize = 64;
    /// Clause 1 compares this many decoded ids.
    const DECODE_LOSSLESS: usize = 256;
    /// Clause 2's prompt, the first ids of the same file.
    const PROSE: usize = 512;
    /// Clause 2 decodes this many tokens with the draft.
    const DECODE_FLOOR: usize = 128;
    /// Clause 2's proposal width. The target verifies 8 rows, so a draft proposes 7.
    const PROPOSAL: usize = 7;
    /// A verify of the proposal: the token already at the target's position, then the 7 drafted ids.
    const VERIFY_ROWS: usize = PROPOSAL + 1;
    /// Clause 2 holds when the committed tokens per verify pass reach this.
    const FLOOR: f64 = 2.0;
    /// The target's context: a 512-id prompt plus the decoded tokens and one verify of 8 rows.
    const TARGET_CTX: usize = 768;
    /// The prompt call's ubatch, the same one the verifying gate opens with.
    const UBATCH: usize = 64;

    type Seat = Session<Body35>;
    type Spark = CardDspark<DenseLayers>;
    type Spec = Speculative<Spark, VERIFY_ROWS>;

    /// The draft's compile-time width is the clause's proposal width.
    const _: () = assert!(<Spark as Draft<Seat>>::WIDTH == PROPOSAL);

    /// A seeded stream of standard normals (xorshift64*, Box–Muller).
    struct Normals(u64);

    impl Normals {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
            ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    // ------------------------------------------------------------ the reference

    /// One layer's weights in f64, the file's layout (NEOX pairs, no permutation).
    struct W {
        ln_in: Vec<f64>,
        ln_post: Vec<f64>,
        q_norm: Vec<f64>,
        k_norm: Vec<f64>,
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        o: Vec<f64>,
        gate: Vec<f64>,
        up: Vec<f64>,
        down: Vec<f64>,
    }

    /// Run `f` over `0..n` on all cores, in order.
    fn par_map<T: Send>(n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .clamp(1, n.max(1));
        let per = n.div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    scope
                        .spawn(move || (t * per..((t + 1) * per).min(n)).map(f).collect::<Vec<T>>())
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a reference thread does not panic"))
                .collect()
        })
    }

    /// The weight `name` as f64: the file's bf16 widened, or (`q8`) as the card holds it, the file's
    /// values through `quantize_q8_0` and back (a block's scale is its f16 bits).
    fn weight(file: &HfFile, name: &str, q8: bool) -> Result<Vec<f64>, GateError> {
        let (bits, _) = file.st().bf16(name).map_err(|e| format!("{name}: {e}"))?;
        if !q8 {
            return Ok(bits
                .iter()
                .map(|&b| f64::from(f32::from_bits(u32::from(b) << 16)))
                .collect());
        }
        let blocks = bits.len() / 32;
        let per_block = |b: usize| -> [f64; 32] {
            let mut vals = [0.0f32; 32];
            for (v, &x) in vals.iter_mut().zip(&bits[b * 32..][..32]) {
                *v = f32::from_bits(u32::from(x) << 16);
            }
            let blk = quantize_q8_0(&vals);
            let d = f64::from(half_to_f32(blk.d));
            let mut out = [0.0f64; 32];
            for (o, &q) in out.iter_mut().zip(&blk.q) {
                *o = d * f64::from(q);
            }
            out
        };
        Ok(par_map(blocks, &per_block).into_iter().flatten().collect())
    }

    fn layer_weights(file: &HfFile, l: usize, q8: bool) -> Result<W, GateError> {
        let n = |tail: &str| format!("layers.{l}.{tail}");
        let gain = |tail: &str| weight(file, &n(tail), false);
        Ok(W {
            ln_in: gain("input_layernorm.weight")?,
            ln_post: gain("post_attention_layernorm.weight")?,
            q_norm: gain("self_attn.q_norm.weight")?,
            k_norm: gain("self_attn.k_norm.weight")?,
            q: weight(file, &n("self_attn.q_proj.weight"), q8)?,
            k: weight(file, &n("self_attn.k_proj.weight"), q8)?,
            v: weight(file, &n("self_attn.v_proj.weight"), q8)?,
            o: weight(file, &n("self_attn.o_proj.weight"), q8)?,
            gate: weight(file, &n("mlp.gate_proj.weight"), q8)?,
            up: weight(file, &n("mlp.up_proj.weight"), q8)?,
            down: weight(file, &n("mlp.down_proj.weight"), q8)?,
        })
    }

    /// `Σ a·b` with four partial sums.
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        let mut s = [0.0f64; 4];
        for (ca, cb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
            for i in 0..4 {
                s[i] += ca[i] * cb[i];
            }
        }
        (s[0] + s[1]) + (s[2] + s[3])
    }

    fn matvec(w: &[f64], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..rows).map(|r| dot(&w[r * k..][..k], x)).collect()
    }

    /// `x / sqrt(mean(x²) + eps) · gain`, over each `gain.len()` run of `x`.
    fn rms_norm(x: &[f64], gain: &[f64], eps: f64) -> Vec<f64> {
        let n = gain.len();
        x.chunks(n)
            .flat_map(|v| {
                let scale = 1.0 / (v.iter().map(|a| a * a).sum::<f64>() / n as f64 + eps).sqrt();
                v.iter().zip(gain).map(move |(a, g)| a * scale * g)
            })
            .collect()
    }

    /// How a head is turned: by angles computed here in f64, or by the card's f32 table row.
    enum Rot<'a> {
        Exact { pos: f64, theta: f64 },
        Table(&'a [f32]),
    }

    /// NEOX rotation of every head of `x` (runs of `hd`): pair `(i, i + hd/2)` by the angle
    /// `pos · theta^(-2i/hd)`, or by the table row's `(cos, sin)` of pair `i`.
    fn neox(x: &mut [f64], hd: usize, rot: &Rot<'_>) {
        for head in x.chunks_mut(hd) {
            for i in 0..hd / 2 {
                let (c, s) = match rot {
                    Rot::Exact { pos, theta } => {
                        let (s, c) = (pos * theta.powf(-2.0 * i as f64 / hd as f64)).sin_cos();
                        (c, s)
                    }
                    Rot::Table(row) => (f64::from(row[2 * i]), f64::from(row[2 * i + 1])),
                };
                let (x0, x1) = (head[i], head[i + hd / 2]);
                head[i] = x0 * c - x1 * s;
                head[i + hd / 2] = x0 * s + x1 * c;
            }
        }
    }

    /// `v` rounded to f16 and back: the rows the card's cache holds.
    fn round_f16(v: &mut [f64]) {
        for x in v {
            *x = f64::from(half_to_f32(f32_to_f16_bits(*x as f32)));
        }
    }

    /// What the reference computes besides the block's output.
    struct RefStats {
        /// The largest `|q · k| / sqrt(hd)` over every layer, head, row and key.
        s_max: f64,
    }

    struct Run<'a> {
        dims: DenseDims,
        eps: f64,
        theta: f64,
        window: Option<usize>,
        causal: bool,
        /// Context rows `r × hidden` and the position of the first.
        ctx_x: &'a [f64],
        r: usize,
        ctx_pos0: u32,
        /// The block's rows `m × hidden`.
        block_x: &'a [f64],
        m: usize,
        /// The card's rotation rows for positions `ctx_pos0 ..` (`r + m` of them), else the angles
        /// are computed here in f64.
        table: Option<&'a [Vec<f32>]>,
        /// Whether K and V are rounded to f16 as the card's cache stores them.
        kv_f16: bool,
    }

    /// The layers on the host, in f64: `layers` of them from `load`, the file's rule.
    fn forward(
        run: &Run<'_>,
        layers: usize,
        load: &dyn Fn(usize) -> Result<W, GateError>,
    ) -> Result<(Vec<f64>, RefStats), GateError> {
        let DenseDims {
            hidden: h,
            n_q,
            n_kv,
            head_dim: hd,
            inter,
        } = run.dims;
        let (m, r) = (run.m, run.r);
        let group = n_q / n_kv;
        let lo = r - run.window.map_or(r, |w| w.min(r));
        let n_vis = r - lo;
        let pos0 = f64::from(run.ctx_pos0);
        let rot = |i: usize| match run.table {
            Some(rows) => Rot::Table(&rows[i]),
            None => Rot::Exact {
                pos: pos0 + i as f64,
                theta: run.theta,
            },
        };
        let scale = 1.0 / (hd as f64).sqrt();
        let mut x = run.block_x.to_vec();
        let mut stats = RefStats { s_max: 0.0 };
        for l in 0..layers {
            let w = load(l)?;
            // Each visible context row's K (normed, turned) and V in this layer.
            let ctx_kv: Vec<(Vec<f64>, Vec<f64>)> = par_map(n_vis, &|j| {
                let row = &run.ctx_x[(lo + j) * h..][..h];
                let mut k = matvec(&w.k, n_kv * hd, h, row);
                k = rms_norm(&k, &w.k_norm, run.eps);
                neox(&mut k, hd, &rot(lo + j));
                let mut v = matvec(&w.v, n_kv * hd, h, row);
                if run.kv_f16 {
                    round_f16(&mut k);
                    round_f16(&mut v);
                }
                (k, v)
            });
            // The block's rows through the layer.
            let rows: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = par_map(m, &|t| {
                let xn = rms_norm(&x[t * h..][..h], &w.ln_in, run.eps);
                let mut q = rms_norm(&matvec(&w.q, n_q * hd, h, &xn), &w.q_norm, run.eps);
                neox(&mut q, hd, &rot(r + t));
                let mut k = rms_norm(&matvec(&w.k, n_kv * hd, h, &xn), &w.k_norm, run.eps);
                neox(&mut k, hd, &rot(r + t));
                let mut v = matvec(&w.v, n_kv * hd, h, &xn);
                if run.kv_f16 {
                    round_f16(&mut k);
                    round_f16(&mut v);
                }
                (q, k, v)
            });
            let outs: Vec<(Vec<f64>, f64)> = par_map(m, &|t| {
                let mut attn = vec![0.0f64; n_q * hd];
                let mut s_max = 0.0f64;
                let seen = if run.causal { t + 1 } else { m };
                for head in 0..n_q {
                    let kvh = head / group;
                    let q = &rows[t].0[head * hd..][..hd];
                    let mut keys: Vec<(&[f64], &[f64])> = ctx_kv
                        .iter()
                        .map(|(k, v)| (&k[kvh * hd..][..hd], &v[kvh * hd..][..hd]))
                        .collect();
                    keys.extend(
                        rows[..seen]
                            .iter()
                            .map(|(_, k, v)| (&k[kvh * hd..][..hd], &v[kvh * hd..][..hd])),
                    );
                    let scores: Vec<f64> = keys.iter().map(|(k, _)| dot(q, k) * scale).collect();
                    s_max = scores.iter().fold(s_max, |a, s| a.max(s.abs()));
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                    let z: f64 = e.iter().sum();
                    let out = &mut attn[head * hd..][..hd];
                    for ((_, v), p) in keys.iter().zip(&e) {
                        for (o, v) in out.iter_mut().zip(*v) {
                            *o += p / z * v;
                        }
                    }
                }
                let o = matvec(&w.o, h, n_q * hd, &attn);
                let x1: Vec<f64> = x[t * h..][..h].iter().zip(&o).map(|(a, b)| a + b).collect();
                let xn2 = rms_norm(&x1, &w.ln_post, run.eps);
                let g = matvec(&w.gate, inter, h, &xn2);
                let u = matvec(&w.up, inter, h, &xn2);
                let act: Vec<f64> = g
                    .iter()
                    .zip(&u)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let d = matvec(&w.down, h, inter, &act);
                (x1.iter().zip(&d).map(|(a, b)| a + b).collect(), s_max)
            });
            for (t, (row, s)) in outs.into_iter().enumerate() {
                x[t * h..][..h].copy_from_slice(&row);
                stats.s_max = stats.s_max.max(s);
            }
        }
        Ok((x, stats))
    }

    // ------------------------------------------------------------ the card

    /// `max |a − b| / max |b|`.
    fn rel_max(a: &[f64], b: &[f64]) -> f64 {
        let top = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
            / top
    }

    fn f32s(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    /// The band of the card against the f64 network on the weights it holds (the Q8_0 values), K and V
    /// rounded to f16 and the rotation rows its table gives: only the card's f32 arithmetic is left.
    /// Random-rounding model: a sum of `n` f32 terms is off by about `u · sqrt(n)` of its size, and a
    /// layer runs five projections over 2,048 inputs (q, k, v, gate, up), the output projection over
    /// 4,096 and the down projection over 6,144, 20,480 terms in all; the layers add in quadrature. The
    /// largest of the block's 16,384 outputs sits near 4σ and the softmax, the norms and the two
    /// `exp`s round once each more, so the band takes 8 times that.
    fn band_arith(layers: usize) -> f64 {
        8.0 * U * (20_480.0 * layers as f64).sqrt()
    }

    /// The Q8_0 model of the weights, at layer 0's query projection on `xn`: every output of the Q8_0
    /// weights is within `Σ|x| · 0.5625 d` of the bf16's, `d` each input block's scale. Returns the
    /// largest ratio of the actual difference to that bound.
    fn q8_bound_ratio(file: &HfFile, xn: &[f64]) -> Result<f64, GateError> {
        let name = "layers.0.self_attn.q_proj.weight";
        let (bits, shape) = file.st().bf16(name).map_err(|e| format!("{name}: {e}"))?;
        let (rows, k) = (shape[0], shape[1]);
        let ratios = par_map(rows, &|r| {
            let (mut y_bf16, mut y_q8, mut bound) = (0.0f64, 0.0f64, 0.0f64);
            for (b, blk) in bits[r * k..][..k].as_chunks::<32>().0.iter().enumerate() {
                let mut vals = [0.0f32; 32];
                for (v, &x) in vals.iter_mut().zip(blk) {
                    *v = f32::from_bits(u32::from(x) << 16);
                }
                let q = quantize_q8_0(&vals);
                let held = f64::from(half_to_f32(q.d));
                let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let xs = &xn[b * 32..][..32];
                for ((&w, &c), &x) in vals.iter().zip(&q.q).zip(xs) {
                    y_bf16 += f64::from(w) * x;
                    y_q8 += held * f64::from(c) * x;
                }
                bound +=
                    Q8_ERR * (f64::from(amax) / 127.0) * xs.iter().map(|x| x.abs()).sum::<f64>();
            }
            if bound == 0.0 {
                0.0
            } else {
                (y_bf16 - y_q8).abs() / bound
            }
        });
        Ok(ratios.into_iter().fold(0.0f64, f64::max))
    }

    /// Row `row` of `bits` (rows of `k` values) as the card's Q8_0 planes: the rule `Q8Rows::from_bf16` runs.
    fn row_planes(bits: &[u16], row: usize, k: usize) -> (Vec<u32>, Vec<u16>) {
        let blocks: Vec<_> = bits[row * k..][..k]
            .as_chunks::<32>()
            .0
            .iter()
            .map(|blk| {
                let mut vals = [0.0f32; 32];
                for (v, &x) in vals.iter_mut().zip(blk) {
                    *v = f32::from_bits(u32::from(x) << 16);
                }
                quantize_q8_0(&vals)
            })
            .collect();
        q8_0_planes(&blocks)
    }

    /// The card's planes of `q` hold the file tensor's first, middle and last rows as the rule makes them.
    fn planes_hold(
        q: &Q8Rows,
        file: &HfFile,
        name: &str,
        rows: usize,
        k: usize,
        stream: &CudaStream,
    ) -> Result<bool, GateError> {
        let (bits, _) = file.st().bf16(name).map_err(|e| format!("{name}: {e}"))?;
        let (qs, d) = (
            q.qs.buf().to_host_vec(stream)?,
            q.d.buf().to_host_vec(stream)?,
        );
        let mut ok = qs.len() == rows * k / 4 && d.len() == rows * k / 32;
        for r in [0, rows / 2, rows - 1] {
            let (want_q, want_d) = row_planes(bits, r, k);
            ok &= qs[r * (k / 4)..][..k / 4] == want_q[..]
                && d[r * (k / 32)..][..k / 32] == want_d[..];
        }
        Ok(ok)
    }

    /// The loader's uploads against the file: fc's and the head's Q8_0 rows, the Markov tables' words,
    /// the id map and the confidence head.
    fn loader_check(parts: &Parts, file: &HfFile, stream: &CudaStream) -> Result<bool, GateError> {
        let h = parts.layers.dims().hidden;
        let taps_width = parts.shape.taps.len() * parts.shape.hidden;
        let fc = planes_hold(&parts.fc.fc, file, "fc.weight", h, taps_width, stream)?;
        let HeadSource::Own(own) = &parts.head.source else {
            return Err("the loader's head is not the draft's own".into());
        };
        let rows = parts.shape.vocab.rows();
        let head = planes_hold(own, file, "lm_head.weight", rows, h, stream)?;
        let words = |name: &str| -> Result<Vec<u32>, GateError> {
            let raw = file.st().raw(name).map_err(|e| format!("{name}: {e}"))?;
            Ok(raw
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect())
        };
        let HeadTail::Markov(tables) = &parts.head.tail else {
            return Err("the loader's head is not a Markov head".into());
        };
        let w1 =
            tables.markov_w1.buf().to_host_vec(stream)? == words("markov_head.markov_w1.weight")?;
        let w2 =
            tables.markov_w2.buf().to_host_vec(stream)? == words("markov_head.markov_w2.weight")?;
        let map = match (&parts.shape.vocab, &tables.to_target) {
            (DraftVocab::Reduced { to_target, .. }, Some(card)) => {
                card.to_host_vec(stream)? == *to_target
            }
            _ => false,
        };
        let conf = match &tables.conf {
            Some((w, b)) => {
                let (fw, _) = file.f32s("confidence_head.proj.weight")?;
                let (fb, _) = file.f32s("confidence_head.proj.bias")?;
                bits_equal(&w.buf().to_host_vec(stream)?, &fw) && b.to_bits() == fb[0].to_bits()
            }
            None => false,
        };
        let norms = {
            let (hn, _) = file.f32s("hidden_norm.weight")?;
            let (n, _) = file.f32s("norm.weight")?;
            bits_equal(&parts.fc.norm.buf().to_host_vec(stream)?, &hn)
                && bits_equal(&parts.head.norm.buf().to_host_vec(stream)?, &n)
        };
        println!(
            "loader: fc rows {} head rows {} markov_w1 {} markov_w2 {} id map {} confidence {} gains {}",
            verdict(fc),
            verdict(head),
            verdict(w1),
            verdict(w2),
            verdict(map),
            verdict(conf),
            verdict(norms),
        );
        Ok(fc && head && w1 && w2 && map && conf && norms)
    }

    fn run_case(
        gpu: &Gpu,
        layers: &mut DenseLayers,
        file: &HfFile,
        mask_id: usize,
        r: usize,
    ) -> Result<bool, GateError> {
        let dims = layers.dims();
        let h = dims.hidden;
        let m = BLOCK;
        let stream = gpu.stream();
        let spec = layers.spec().clone();
        let (eps, theta) = (f64::from(spec.eps), f64::from(spec.rope_theta));
        // Seeded inputs at the scales the draft sees: the context rows are `hidden_norm`'s output
        // (unit-RMS rows times its gain), the block rows are rows of the draft's own embedding table.
        let (hn, _) = file.f32s("hidden_norm.weight")?;
        let mut rng = Normals(0x9e37_79b9_7f4a_7c15 ^ r as u64);
        let ctx_x: Vec<f64> = (0..r * h)
            .map(|i| rng.normal() * f64::from(hn[i % h]))
            .collect();
        let (emb, _) = file
            .st()
            .bf16("embed_tokens.weight")
            .map_err(|e| format!("{e}"))?;
        let block_x: Vec<f64> = (0..m)
            .flat_map(|t| {
                let id = if t == 0 { ANCHOR_ID } else { mask_id };
                emb[id * h..][..h]
                    .iter()
                    .map(|&b| f64::from(f32::from_bits(u32::from(b) << 16)))
            })
            .collect();

        // The card: append the context in chunks, then one block pass.
        layers.reset(gpu)?;
        let t0 = Instant::now();
        let (mut done, mut turn) = (0usize, 0usize);
        while done < r {
            let n = CHUNKS[turn % CHUNKS.len()].min(r - done).min(APPEND);
            turn += 1;
            let rows = f32s(&ctx_x[done * h..][..n * h]);
            let x = DeviceBuffer::from_host(stream, &rows)?;
            layers.stage_kv(gpu, FIRST + u32::try_from(done)?, n)?;
            layers.append_kv(gpu, &x, n)?;
            done += n;
        }
        stream.synchronize()?;
        let appended = t0.elapsed();
        let pos = FIRST + u32::try_from(r)?;
        let input = f32s(&block_x);
        let mut x = DeviceBuffer::from_host(stream, &input)?;
        layers.stage_block(gpu, 0, pos, m, layers.visible())?;
        layers.enqueue_block(gpu, &mut x, m)?;
        stream.synchronize()?;
        let card: Vec<f64> = x
            .to_host_vec(stream)?
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        if let Some(f) = gpu.fault()? {
            return Err(format!("the card raised {f:?}").into());
        }
        // The captured pass replays to the eager bits.
        let mut x2 = DeviceBuffer::from_host(stream, &input)?;
        let graph = gpu.capture(|_| layers.enqueue_block(gpu, &mut x2, m))?;
        x2.copy_from_host(stream, &input)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let replayed = bits_equal(&x2.to_host_vec(stream)?, &f32s(&card));
        let stats = layers.stats();

        // The references: the file's bf16 with exact angles (A), and the network on the weights the card
        // holds, its rotation rows and its f16 cache (B).
        let rope = RopeTable::new(&RopeSpec::window(spec.rope_theta, dims.head_dim))?;
        let rows: Vec<Vec<f32>> = (0..r + m)
            .map(|i| {
                let mut row = Vec::new();
                rope.push(
                    FIRST + u32::try_from(i).unwrap_or(0),
                    Direction::Forward,
                    &mut row,
                );
                row
            })
            .collect();
        let exact = Run {
            dims,
            eps,
            theta,
            window: spec.window,
            causal: spec.block_causal,
            ctx_x: &ctx_x,
            r,
            ctx_pos0: FIRST,
            block_x: &block_x,
            m,
            table: None,
            kv_f16: false,
        };
        let held = Run {
            table: Some(&rows),
            kv_f16: true,
            ..exact
        };
        let t1 = Instant::now();
        let (a, sa) = forward(&exact, spec.layers, &|l| layer_weights(file, l, false))?;
        let (b, _) = forward(&held, spec.layers, &|l| layer_weights(file, l, true))?;
        let host = t1.elapsed();
        let xn0 = rms_norm(
            &block_x[..h],
            &weight(file, "layers.0.input_layernorm.weight", false)?,
            eps,
        );
        let ratio = q8_bound_ratio(file, &xn0)?;
        let (card_a, card_b, b_a) = (rel_max(&card, &a), rel_max(&card, &b), rel_max(&b, &a));
        let band = band_arith(spec.layers);
        let arith = card_b <= band;
        let model = card_a <= b_a + band;
        let q8 = ratio <= 1.0;
        let wrapped = r <= layers.ring_rows() || stats.wrapped > 0;
        println!(
            "R={r}: card vs the Q8_0 model {card_b:.3e} (band {band:.3e}) {}; card vs bf16 {card_a:.3e}              (the model's {b_a:.3e} plus the band) {}; max score {:.2}",
            verdict(arith),
            verdict(model),
            sa.s_max
        );
        println!(
            "R={r}: Q8_0 weights within the per-block bound at layer 0's query projection              (largest ratio {ratio:.3}) {}; {} appends, {} past the ring's end {}; the captured block              pass {} the eager bits; append {appended:?}, host reference {host:?}",
            verdict(q8),
            stats.appends,
            stats.wrapped,
            verdict(wrapped),
            if replayed {
                "replays"
            } else {
                "DOES NOT replay"
            }
        );
        Ok(arith && model && q8 && wrapped && replayed)
    }

    /// Staging, on layers that no stage call has run on (a reset): the eager block pass and the
    /// eager append are refused, each naming the stage call it needs; a replay is refused by name
    /// the same way; the same passes captured, as the owner captures them at load, record. After
    /// one `stage_kv` the replay is allowed. The layers are reset again before return.
    fn staging_clause(gpu: &Gpu, layers: &mut DenseLayers) -> Result<bool, GateError> {
        layers.reset(gpu)?;
        let rows = vec![0.0f32; BLOCK.max(APPEND) * layers.dims().hidden];
        let mut x = DeviceBuffer::from_host(gpu.stream(), &rows)?;
        let names = |r: Result<(), GpuError>, call: &str| {
            r.err().is_some_and(|e| e.to_string().contains(call))
        };
        let block = names(layers.enqueue_block(gpu, &mut x, BLOCK), "stage_block");
        let append = names(layers.append_kv(gpu, &x, APPEND), "stage_kv");
        let replay_text = layers.replayable().err().map(|e| e.to_string());
        let replay = replay_text
            .as_deref()
            .is_some_and(|s| s.contains("stage_kv") && s.contains("stage_block"));
        let block_captured = gpu
            .capture(|_| layers.enqueue_block(gpu, &mut x, BLOCK))
            .is_ok();
        let append_captured = gpu.capture(|_| layers.append_kv(gpu, &x, APPEND)).is_ok();
        layers.stage_kv(gpu, 0, APPEND)?;
        let replay_after = layers.replayable().is_ok();
        layers.reset(gpu)?;
        println!(
            "staging: the eager block pass before stage_block is refused by name {}; the eager \
             append before stage_kv {}; a replay before either stage call {}; a replay after \
             stage_kv {}; the captures before any stage call record {}",
            verdict(block),
            verdict(append),
            verdict(replay),
            verdict(replay_after),
            verdict(block_captured && append_captured)
        );
        Ok(block && append && replay && replay_after && block_captured && append_captured)
    }

    /// Clause 0. The draft `Gpu` drops with this function, before a target load.
    fn dense(dir: &Path) -> Result<bool, GateError> {
        let gpu = Gpu::new()?;
        println!(
            "gate_qwen35_dspark: device {} — draft {}",
            gpu.device_name()?,
            dir.display()
        );
        let t0 = Instant::now();
        let mut parts = dspark_hf::open(&gpu, dir, CTX)?;
        let bytes = dspark_hf::card_bytes(dir, CTX)?;
        println!(
            "loader: {:?}; {} taps {:?}, block {}, window {:?}, {} draft ids of {}; card bytes {} \
             ({} MiB) as the reserve {:?}; layers hold {} B at {CTX} positions",
            t0.elapsed(),
            parts.shape.taps.len(),
            parts.shape.taps,
            parts.shape.block,
            parts.shape.window,
            parts.shape.vocab.rows(),
            parts.shape.vocab.n_target(),
            bytes,
            bytes >> 20,
            dspark_hf::RESERVE,
            parts.layers.card_bytes(CTX),
        );
        let mut ok = true;
        let sized = bytes >= parts.layers.card_bytes(CTX);
        println!(
            "loader: card bytes cover the layers {}; fc {}x{}, head norm {}, embed {} x {}",
            verdict(sized),
            parts.fc.fc.d.rows(),
            parts.fc.fc.d.cols() * 32,
            parts.head.norm.cols(),
            match &parts.embed {
                bloomery_gpu::dspark::EmbedSource::OwnHost(e) => e.rows,
                bloomery_gpu::dspark::EmbedSource::Target => 0,
            },
            parts.layers.dims().hidden,
        );
        ok &= sized;
        let file = HfFile::open(dir)?;
        ok &= loader_check(&parts, &file, gpu.stream())?;
        ok &= staging_clause(&gpu, &mut parts.layers)?;
        let mask_id = usize::try_from(parts.shape.mask_id)?;
        for r in CASES {
            let case = run_case(&gpu, &mut parts.layers, &file, mask_id, r)?;
            println!("dense layers, R={r} {}", verdict(case));
            ok &= case;
        }
        Ok(ok)
    }

    /// A plain or verifying load of the Qwen3.6 file, whole card, context [`TARGET_CTX`].
    fn open_seat(verifying: bool) -> Result<Seat, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!("{MODEL} is {:?}, not qwen35moe", file.architecture()).into());
        }
        let o = Open35 {
            ctx: TARGET_CTX,
            mma: true,
            ubatch: UBATCH,
            kv: KvQ8::F16,
            media: false,
        };
        let gpu = Gpu::new()?;
        let m = if verifying {
            Qwen35moeModel::open_verifying(gpu, file, o)?
        } else {
            Qwen35moeModel::open(gpu, file, o)?
        };
        println!(
            "load verifying={verifying} resident_bytes={} ctx={TARGET_CTX}",
            m.resident_bytes()
        );
        Ok(Seat::from_model(m, u32::try_from(TARGET_CTX)?))
    }

    /// The draft beside `seat`, proposing [`PROPOSAL`] ids. Refused by name when this file's block
    /// cannot.
    fn attach(seat: &mut Seat, dir: &Path) -> Result<Spec, GateError> {
        let gpu = Gpu::new()?;
        let (free0, _) = gpu.mem_info()?;
        let Parts {
            shape,
            layers,
            fc,
            head,
            embed,
        } = dspark_hf::open(&gpu, dir, CTX)?;
        let verify_rows = <Seat as Verify>::MAX_ROWS;
        let width = shape.width(verify_rows);
        if width != PROPOSAL {
            return Err(format!(
                "clause 2 asks for a proposal of {PROPOSAL}; this file's block {} (row 0 drafts \
                 {}) proposes {width} against a verify of {verify_rows} rows",
                shape.block, shape.row0_drafts
            )
            .into());
        }
        let taps = shape.taps.clone();
        let mask_id = shape.mask_id;
        let body = DsparkBody::new(&gpu, &shape, layers, fc, head, embed)?;
        let (free1, _) = gpu.mem_info()?;
        let taken = free0.saturating_sub(free1);
        let own = body.own_held();
        let held = body.layers().held(&own);
        let target = Arc::clone(seat.model().gpu().context());
        let card = CardDspark::new::<Body35>(body, gpu, target, "gate", shape, held, taken)?;
        println!(
            "draft: taps {taps:?}, mask {mask_id}, proposal width {width}, resident {} B, taken \
             {taken} B",
            card.resident_bytes()
        );
        // Capture reads the stream's current context and does not bind it. The draft's load left
        // its own context current.
        seat.model().gpu().context().bind_to_thread()?;
        Ok(seat.with_draft::<Spark, VERIFY_ROWS>(card, &mut Quiet)?)
    }

    /// Committed tokens of clause 2, and how often draft slot `k` (1-based) was kept.
    struct Accept {
        passes: usize,
        kept: usize,
        /// `slot[k - 1]` counts the passes that kept draft slot `k`.
        slot: [usize; PROPOSAL],
    }

    /// `n` drafted ids from `prompt`. A pass may run past `n`; the caller reads the first `n`.
    /// With `acc`, every pass is a proposal of [`PROPOSAL`] ids or the run stops by name.
    fn draft_n(
        spec: &mut Spec,
        seat: &mut Seat,
        prompt: &[u32],
        n: usize,
        mut acc: Option<&mut Accept>,
    ) -> Result<Vec<u32>, GateError> {
        seat.reset()?;
        let first = Advance::prompt(spec, seat, prompt)?;
        Advance::begin(spec, seat, prompt, first)?;
        let mut tokens = vec![first];
        while tokens.len() < n {
            let before = tokens.len();
            let last = tokens[tokens.len() - 1];
            let c = Advance::pass(spec, seat, last, &mut tokens)?;
            let appended = tokens.len() - before;
            if appended != c.kept {
                return Err(format!("a pass kept {} and appended {appended}", c.kept).into());
            }
            if let Some(acc) = acc.as_mut() {
                if !c.proposed || c.rows != VERIFY_ROWS {
                    return Err(format!(
                        "clause 2 is a proposal of {PROPOSAL}: a pass proposed {} over {} rows",
                        c.proposed, c.rows
                    )
                    .into());
                }
                acc.passes += 1;
                acc.kept += c.kept;
                for k in 1..=PROPOSAL {
                    if c.kept > k {
                        acc.slot[k - 1] += 1;
                    }
                }
            }
        }
        spec.draft_mut().check_fault()?;
        if tokens.len() < n {
            return Err(
                format!("the draft run decoded {} ids, {n} asked for", tokens.len()).into(),
            );
        }
        Ok(tokens)
    }

    /// Clause 1: the first `n` ids, or the first index that differs.
    fn same_prefix(plain: &[u32], draft: &[u32], n: usize) -> Result<bool, GateError> {
        if plain.len() < n || draft.len() < n {
            return Err(format!(
                "clause 1 compares {n} ids: the plain run holds {}, the draft {}",
                plain.len(),
                draft.len()
            )
            .into());
        }
        match (0..n).find(|&i| plain[i] != draft[i]) {
            None => {
                println!("clause 1: ids identical {n} {}", verdict(true));
                Ok(true)
            }
            Some(i) => {
                println!(
                    "clause 1: first divergence at {i} (plain {} draft {}) {}",
                    plain[i],
                    draft[i],
                    verdict(false)
                );
                Ok(false)
            }
        }
    }

    /// Clause 2's line. No verify pass is a named error, not an E of 0. `decoded` is the run's
    /// ids, prompt argmax included; E is over the verify passes only.
    fn floor_line(acc: &Accept, decoded: usize) -> Result<bool, GateError> {
        if acc.passes == 0 {
            return Err("clause 2: no verify pass".into());
        }
        let e = acc.kept as f64 / acc.passes as f64;
        let ok = e >= FLOOR;
        let slots = (0..PROPOSAL)
            .map(|k| format!("{}:{}", k + 1, acc.slot[k]))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "clause 2: E={e:.3} over {} verify passes, {} committed tokens, {decoded} decoded; \
             accept {slots} (floor {FLOOR}) {}",
            acc.passes,
            acc.kept,
            verdict(ok)
        );
        Ok(ok)
    }

    /// Clauses 1 and 2. The plain load is dropped before the verifying load.
    fn target(dir: &Path, c1: bool, c2: bool) -> Result<bool, GateError> {
        let mut ok = true;
        let fixed = prose_ids("qwen35moe", PROMPT_FIXED)?;
        let plain = if c1 {
            let mut seat = open_seat(false)?;
            let (ids, _) = greedy(&mut seat, &fixed, DECODE_LOSSLESS, false)?;
            drop(seat);
            Some(ids)
        } else {
            None
        };
        let prose = if c2 {
            prose_ids("qwen35moe", PROSE)?
        } else {
            Vec::new()
        };
        let mut seat = open_seat(true)?;
        let mut spec = attach(&mut seat, dir)?;
        if let Some(plain) = plain {
            let drafted = draft_n(&mut spec, &mut seat, &fixed, DECODE_LOSSLESS, None)?;
            ok &= same_prefix(&plain, &drafted, DECODE_LOSSLESS)?;
        }
        if c2 {
            let mut acc = Accept {
                passes: 0,
                kept: 0,
                slot: [0; PROPOSAL],
            };
            let drafted = draft_n(&mut spec, &mut seat, &prose, DECODE_FLOOR, Some(&mut acc))?;
            if drafted.len() < DECODE_FLOOR {
                return Err(format!(
                    "clause 2 decoded {} ids, {DECODE_FLOOR} asked for",
                    drafted.len()
                )
                .into());
            }
            ok &= floor_line(&acc, drafted.len())?;
        }
        Ok(ok)
    }

    /// `--draft <dir>` (else the box's copy) and `--only 0|1|2`, repeatable. No `--only` runs all.
    fn parse_args() -> Result<(PathBuf, [bool; 3]), GateError> {
        let mut dir = dspark_hf::default_dir();
        let mut only = [false; 3];
        let mut named = false;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--draft" => {
                    dir = PathBuf::from(args.next().ok_or("--draft needs a directory")?);
                }
                "--only" => {
                    let n: usize = args
                        .next()
                        .ok_or("--only needs 0, 1 or 2")?
                        .parse()
                        .map_err(|_| -> GateError { "--only needs 0, 1 or 2".into() })?;
                    if n > 2 {
                        return Err(format!("--only {n}: clauses are 0, 1 and 2").into());
                    }
                    only[n] = true;
                    named = true;
                }
                other => return Err(format!("unknown argument {other:?}").into()),
            }
        }
        if !named {
            only = [true, true, true];
        }
        Ok((dir, only))
    }

    pub fn run() -> Result<(), GateError> {
        let (dir, only) = parse_args()?;
        let mut ok = true;
        if only[0] {
            ok &= dense(&dir)?;
        }
        if only[1] || only[2] {
            ok &= target(&dir, only[1], only[2])?;
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
