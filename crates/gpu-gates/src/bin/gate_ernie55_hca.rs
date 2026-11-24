//! GPU gate for ERNIE 5.5's HCA attention sub-layer on the card
//! (`crates/gpu-ernie55/src/attn_hca.rs`) and its head-mix kernel (`crates/gpu/src/vha.rs`), seeded inputs at the released
//! file's HCA shapes — hidden 4096, 64 heads of 512 values, one latent row a key and its value, window 128, pooling ratio
//! 128, eight output groups of 1024, a query latent of 1024, a rope tail of 64 under YaRN — nothing read from a model file.
//! The weights are seeded in the file's formats (FP8 128-blocks, BF16, F32) and made resident through `Weights::add_parts`.
//!
//! The reference is the formula of `tools/ref/ernie55_ref.py`'s `_hca_attention` (its sources: llama.cpp's `deepseek4.cpp`,
//! `build_attention_impl` and `build_hca_compressed_kv_from_state`), in f64 over the same dequantized weights and the same rope
//! rows the kernels read: the query path (`q_a`, its weighted norm, `q_b`, the weightless per-head norm, the tail rope), the
//! compressor (the scores plus the position table's row, the per-channel softmax pool, the norm, the rope at the block's first
//! position), the sink attention over the f16 window rows and compressed rows with the query rounded to f16 as the tensor
//! cores stage it, the head mix folded by its own rule, the inverse rope, the grouped projection, the sigmoid gate and
//! `attn_output`. [`BAND`] holds the card's f32 sums and its f16 roundings taken here in f64 first.
//!
//! 1. `step`: the sub-layer at positions 0, 1, 127, 128, 129, 255, 256 and 1000, each after every position before it ran
//!    through the same layer on the same state, within [`BAND`] of the f64 rule; no fault raised.
//! 2. `prompt`: the prompt form from 0 over 1001 tokens, and on that state from 1001 (a start no piece aligns with) over
//!    300 more: every row the step form's at its position bit for bit (the first at the eight positions above, the second at
//!    each of its 300).
//! 3. `vha`: the head mix with `U = V = 0` leaves its input bit for bit; with the seeded mix it is within [`BAND_VHA`] of the
//!    f64 rule over the same folded matrix.
//! 4. `gate`: the output gate at 8192 values a token (64 heads of the gate's 128) over 3 tokens: the host rule's bits.
//! 5. `refusals`: a step past the context, a launch over words that hold no run or another run, a short row, a prompt of no
//!    token or past the context, a run over a piece, the head mix at 48 heads, at a width off its columns, with a short matrix
//!    and no token, shapes the kernels do not take, a head mix of another length, and a layer whose weights are not resident:
//!    each refused by name before anything is enqueued.
//! 6. `normed input`: the step over the trunk's normed row is the step over the fold bit for bit at each of the first
//!    [`NORMED_STEPS`] positions.
//! 7. `replay`: the step captured on a state no run's words were set on (as a session's open captures it), then replayed at
//!    every position from 0 through the step clause's last with that position's words set before each replay: the eight
//!    rows the step form's bits; no fault raised.

#[cfg(feature = "ernie55")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_hca", gate::run())
}

#[cfg(feature = "ernie55")]
mod gate {
    use bloomery_gpu::linear::out_gate::{OutGateArgs, out_gate_host};
    use bloomery_gpu::rope_table::{Direction, RopeTable};
    use bloomery_gpu::vha::{fold, mix_host};
    use bloomery_gpu::weights::{TensorPart, Weights};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_ernie55::attn_hca::{
        HcaDims, HcaIn, HcaKernels, HcaKeys, HcaLayer, HcaState, Vha,
    };
    use bloomery_gpu_gates::{
        GateError, GateRng, bits_equal, checks_failed, max_rel_err, par_map, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::TensorInfo;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits, half_to_f32};
    use model::arch::ernie55::hparams::{Hca, Rope};
    use model::arch::ernie55::names;
    use model::arch::models::OutGate;

    /// The file's shapes: `config.json` of `baidu/ERNIE-5.5-Flash-Preview`.
    const HIDDEN: usize = 4096;
    const HEADS: usize = 64;
    const HEAD: usize = 512;
    const ROPE: usize = 64;
    const RATIO: usize = 128;
    const WINDOW: usize = 128;
    const O_GROUPS: usize = 8;
    const O_LORA: usize = 1024;
    const Q_LORA: usize = 1024;
    const GROUPED: usize = O_GROUPS * O_LORA;
    const VHA_RANK: usize = 16;
    const EPS: f32 = 1e-6;
    /// The context the state is sized for: past the last position the clauses reach.
    const CTX: usize = 2048;
    /// The positions the step clause reads, and the last position any clause reaches.
    const STEPS: [usize; 8] = [0, 1, 127, 128, 129, 255, 256, 1000];
    const LAST: usize = 1300;
    /// The steps the normed-input clause runs both forms of: past a completed pooling block, so
    /// the compressor's rows a normed input feeds are read back too.
    const NORMED_STEPS: usize = 130;
    /// [derived] Against an f64 rule over the same weights and rope rows, the card's errors are the f32 sums of the
    /// projections (at most 8192 products), of the attention (at most 138 keys), of the 64-term head mix, the norms' f32
    /// trees, the sigmoid and the two rotations — each a few `1e-6` of its own magnitude and carried through at most nine
    /// stages: `2e-5` their sum at the output's largest value; `1e-3` is fifty times that.
    const BAND: f32 = 1e-3;
    /// [derived] The head mix's f32 fma chain of 64 terms against the f64 rule: `64 · 2⁻²⁴ ≈ 4e-6` of a row's largest
    /// value at most; `2e-5` is five times that.
    const BAND_VHA: f32 = 2e-5;
    /// The output gate's values a head (`bloomery_gpu::linear::out_gate`).
    const GATE_HEAD: usize = 128;

    /// Print one clause's line and return whether it passed.
    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{} {name}: {detail}", verdict(pass));
        pass
    }

    /// The band check of a row against its reference.
    fn banded(name: &str, band: f32, y: &[f32], want: &[f64]) -> bool {
        let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        match max_rel_err(y, &want) {
            Ok(rel) => clause(
                name,
                rel <= band,
                &format!("max_rel {rel:.3e} (band {band:.0e})"),
            ),
            Err(e) => clause(name, false, &e.to_string()),
        }
    }

    /// The card, its stream and the entries.
    struct Dev {
        gpu: Gpu,
        k: HcaKernels,
    }

    impl Dev {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn up<T: DeviceCopy>(&self, v: &[T]) -> Result<DeviceBuffer<T>, GateError> {
            Ok(DeviceBuffer::from_host(self.stream(), v)?)
        }

        fn read(&self, b: &DeviceBuffer<f32>) -> Result<Vec<f32>, GateError> {
            self.stream().synchronize()?;
            Ok(b.to_host_vec(self.stream())?)
        }

        /// No launch raised its fault word.
        fn no_fault(&self, name: &str) -> Result<bool, GateError> {
            let f = self.gpu.fault()?;
            Ok(clause(
                &format!("{name} no_fault"),
                f.is_none(),
                &format!("{f:?}"),
            ))
        }
    }

    /// The description's keys at the file's shapes.
    fn keys() -> HcaKeys {
        HcaKeys {
            hidden: HIDDEN,
            heads: HEADS,
            hca: Hca {
                head: HEAD,
                ratio: RATIO,
                window: WINDOW,
                o_groups: O_GROUPS,
                o_lora: O_LORA,
            },
            rope: ROPE,
            q_lora: Q_LORA,
            eps: EPS,
            yarn: Rope {
                base: 8_000_000.0,
                compress_base: 160_000.0,
                factor: 16.0,
                original_ctx: 65_536,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        }
    }

    // ------------------------------------------------------------ seeded weights

    /// One tensor's seeded bytes in its file format, with its header entry.
    struct Seeded {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        bytes: Vec<u8>,
    }

    /// F8_E4M3's positive codes for `0, ½, 1, 1½, 2, 3, 4, 6`: sign 1, exponent 4 (bias 7), mantissa 3.
    const E4M3: [u8; 8] = [0x00, 0x10, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34];

    impl Seeded {
        /// A projection of `rows` rows by `k` values in FP8 128-blocks (`[k, rows]`): each block's scale a power of two
        /// in `2⁻⁵ .. 2⁻⁷`, its values the E4M3 grid's.
        fn fp8(name: String, rows: usize, k: usize, rng: &mut GateRng) -> Seeded {
            let mut bytes = Vec::with_capacity(rows * (k / 128) * 132);
            for _ in 0..rows * (k / 128) {
                bytes.extend_from_slice(&2.0f32.powi(-((5 + rng.below(3)) as i32)).to_le_bytes());
                for _ in 0..128 {
                    bytes.push(E4M3[rng.below(8)] | ((rng.below(2) as u8) << 7));
                }
            }
            Seeded {
                name,
                dims: vec![k as u64, rows as u64],
                ty: GgmlType::FP8_B128,
                bytes,
            }
        }

        /// A projection of `rows` rows by `k` values in BF16 (`[k, rows]`): values in `[-0.08, 0.08)`, rounded to
        /// nearest even.
        fn bf16(name: String, rows: usize, k: usize, rng: &mut GateRng) -> Seeded {
            let mut bytes = Vec::with_capacity(rows * k * 2);
            for _ in 0..rows * k {
                let bits = rng.uni(-0.08, 0.08).to_bits();
                let round = bits + 0x8000 + ((bits >> 16) & 1);
                bytes.extend_from_slice(&((round >> 16) as u16).to_le_bytes());
            }
            Seeded {
                name,
                dims: vec![k as u64, rows as u64],
                ty: GgmlType::BF16,
                bytes,
            }
        }

        /// `n` F32 values in `[lo, hi)` (`[n]`).
        fn f32(name: String, n: usize, lo: f32, hi: f32, rng: &mut GateRng) -> Seeded {
            let bytes: Vec<u8> = (0..n).flat_map(|_| rng.uni(lo, hi).to_le_bytes()).collect();
            Seeded {
                name,
                dims: vec![n as u64],
                ty: GgmlType::F32,
                bytes,
            }
        }

        /// Its values as the card's loaders dequantize them: `dims[1]` (else 1) rows of `dims[0]` values.
        fn values(&self) -> Vec<f32> {
            let (k, rows) = (
                self.dims[0] as usize,
                self.dims.get(1).copied().unwrap_or(1) as usize,
            );
            let row = self.bytes.len() / rows;
            let mut out = vec![0.0f32; k * rows];
            for r in 0..rows {
                dequant_row(
                    self.ty,
                    &self.bytes[r * row..],
                    &mut out[r * k..(r + 1) * k],
                )
                .expect("a seeded tensor dequantizes");
            }
            out
        }

        fn part(&self) -> TensorPart<'_> {
            TensorPart {
                info: TensorInfo {
                    name: self.name.clone(),
                    dims: self.dims.clone(),
                    ty: self.ty,
                    offset: 0,
                    nbytes: self.bytes.len() as u64,
                },
                spans: vec![&self.bytes],
                scale: None,
            }
        }
    }

    /// The dequantized weights of the rule, row-major: a projection's rows of its `k` values.
    #[derive(Clone)]
    struct Tensors {
        norm: Vec<f32>,
        q_a: Vec<f32>,
        q_a_norm: Vec<f32>,
        q_b: Vec<f32>,
        kv: Vec<f32>,
        kv_norm: Vec<f32>,
        comp_kv: Vec<f32>,
        comp_gate: Vec<f32>,
        comp_norm: Vec<f32>,
        /// `[ratio, head]`: row `pos % ratio` is the pooling scores' bias of that slot.
        ape: Vec<f32>,
        gate: Vec<f32>,
        output_a: Vec<f32>,
        output_b: Vec<f32>,
        sinks: Vec<f32>,
    }

    /// Layer 0's tensors, seeded and made resident in `w`; their dequantized values.
    fn seed_weights(
        stream: &CudaStream,
        w: &mut Weights,
        rng: &mut GateRng,
    ) -> Result<Tensors, GateError> {
        let (group_k, gain) = (HEADS * HEAD / O_GROUPS, (0.5f32, 1.5f32));
        let seeds = [
            Seeded::f32(names::attn_norm(0), HIDDEN, gain.0, gain.1, rng),
            Seeded::fp8(names::attn_q_a(0), Q_LORA, HIDDEN, rng),
            Seeded::f32(names::attn_q_a_norm(0), Q_LORA, gain.0, gain.1, rng),
            Seeded::fp8(names::attn_q_b(0), HEADS * HEAD, Q_LORA, rng),
            Seeded::fp8(names::attn_kv(0), HEAD, HIDDEN, rng),
            Seeded::f32(names::attn_kv_a_norm(0), HEAD, gain.0, gain.1, rng),
            Seeded::bf16(names::attn_compressor_kv(0), HEAD, HIDDEN, rng),
            Seeded::bf16(names::attn_compressor_gate(0), HEAD, HIDDEN, rng),
            Seeded::f32(names::attn_compressor_norm(0), HEAD, gain.0, gain.1, rng),
            Seeded::f32(names::attn_compressor_ape(0), RATIO * HEAD, -1.0, 1.0, rng),
            Seeded::fp8(names::attn_gate(0), GROUPED, Q_LORA, rng),
            Seeded::fp8(names::attn_output_a(0), GROUPED, group_k, rng),
            Seeded::fp8(names::attn_output_b(0), HIDDEN, GROUPED, rng),
            Seeded::f32(names::attn_sinks(0), HEADS, -2.0, 4.0, rng),
        ];
        let parts: Vec<TensorPart<'_>> = seeds.iter().map(Seeded::part).collect();
        w.add_parts(stream, &parts)?;
        let v: Vec<Vec<f32>> = seeds.iter().map(Seeded::values).collect();
        let [
            norm,
            q_a,
            q_a_norm,
            q_b,
            kv,
            kv_norm,
            comp_kv,
            comp_gate,
            comp_norm,
            ape,
            gate,
            output_a,
            output_b,
            sinks,
        ] = v.try_into().expect("fourteen tensors");
        Ok(Tensors {
            norm,
            q_a,
            q_a_norm,
            q_b,
            kv,
            kv_norm,
            comp_kv,
            comp_gate,
            comp_norm,
            ape,
            gate,
            output_a,
            output_b,
            sinks,
        })
    }

    /// The head mix's factors, seeded: `U` and `V` each `heads` rows of `rank` values in `[-0.12, 0.12)`.
    fn seed_vha(rng: &mut GateRng) -> Vha {
        let n = HEADS * VHA_RANK;
        Vha {
            rank: VHA_RANK,
            u: (0..n).map(|_| rng.uni(-0.12, 0.12)).collect(),
            v: (0..n).map(|_| rng.uni(-0.12, 0.12)).collect(),
        }
    }

    // ------------------------------------------------------------ the f64 rule

    /// `y = W · x` for `rows` rows of `k` values, a row a unit.
    fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        par_map(rows, |r| {
            (0..k).map(|c| f64::from(w[r * k + c]) * x[c]).sum()
        })
    }

    /// `y = W · x` for `rows` rows of `k` values on the calling thread: what a unit of [`matvec`]'s
    /// walk runs.
    fn matvec_on(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        w.chunks_exact(k)
            .take(rows)
            .map(|row| row.iter().zip(x).map(|(&v, &u)| f64::from(v) * u).sum())
            .collect()
    }

    /// `x · (1 / √(mean(x²) + eps)) · gain`: the weightless norm where `gain` is all ones.
    fn rms(x: &[f64], gain: &[f32], eps: f64) -> Vec<f64> {
        let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let scale = 1.0 / (mean + eps).sqrt();
        x.iter()
            .zip(gain)
            .map(|(&v, &g)| scale * f64::from(g) * v)
            .collect()
    }

    /// The NORM-mode rotation of the last `dims` values of `x` by the table row `cs` (`[cos, sin]` pairs).
    fn rope_into(x: &mut [f64], cs: &[f32], dims: usize) {
        let off = x.len() - dims;
        for i in 0..dims / 2 {
            let (c, s) = (f64::from(cs[2 * i]), f64::from(cs[2 * i + 1]));
            let (a, b) = (x[off + 2 * i], x[off + 2 * i + 1]);
            x[off + 2 * i] = a * c - b * s;
            x[off + 2 * i + 1] = a * s + b * c;
        }
    }

    /// A value as the card's f16 caches hold it: rounded once to f16, in f64.
    fn f16(v: f64) -> f64 {
        f64::from(half_to_f32(f32_to_f16_bits(v as f32)))
    }

    /// The latent path's products of one position: what the ring, the compressor and the query path take from it.
    struct Latent {
        normed: Vec<f64>,
        /// The normed, roped latent row.
        kv: Vec<f64>,
        comp_kv: Vec<f64>,
        score: Vec<f64>,
    }

    /// The f64 rule of the sub-layer: the reference's formula over the dequantized weights, with the caches a sequence
    /// carries.
    struct Rule {
        d: HcaDims,
        table: RopeTable,
        t: Tensors,
        /// The head mix `m[h][g] = δ(h, g) + Σ_k V[h][k] · U[g][k]`, folded in f64 and rounded to f32 as the layer's.
        mix: Vec<f64>,
        /// The window ring's rows, f16: slot `p % window`.
        ring: Vec<Vec<f64>>,
        /// The compressed rows, f16.
        comp: Vec<Vec<f64>>,
        /// The unfinished block's pooling projections and scores.
        pend: Vec<(Vec<f64>, Vec<f64>)>,
    }

    impl Rule {
        fn new(d: &HcaDims, t: Tensors, vha: &Vha) -> Result<Rule, GateError> {
            let heads = d.heads;
            let mut mix = vec![0.0f64; heads * heads];
            for g in 0..heads {
                for h in 0..heads {
                    let mut acc = f64::from(u32::from(h == g));
                    for k in 0..vha.rank {
                        acc +=
                            f64::from(vha.v[h * vha.rank + k]) * f64::from(vha.u[g * vha.rank + k]);
                    }
                    mix[h * heads + g] = f64::from(acc as f32);
                }
            }
            Ok(Rule {
                d: d.clone(),
                table: RopeTable::new(&d.yarn)?,
                ring: vec![vec![0.0; d.head]; d.window],
                comp: Vec::new(),
                pend: Vec::new(),
                mix,
                t,
            })
        }

        /// Rope row `pos` in `dir`, as the kernels read it.
        fn rows(&self, pos: usize, dir: Direction) -> Vec<f32> {
            let mut row = Vec::with_capacity(self.d.rope);
            self.table.push(pos as u32, dir, &mut row);
            row
        }

        /// The latent path's products of the input `x` at `pos`, on the calling thread (the positions
        /// are the units of the pass that calls it).
        fn latent(&self, pos: usize, x: &[f64]) -> Latent {
            let (hd, eps) = (self.d.head, f64::from(self.d.eps));
            let normed = rms(x, &self.t.norm, eps);
            let kvraw = matvec_on(&self.t.kv, hd, self.d.hidden, &normed);
            let mut kv = rms(&kvraw, &self.t.kv_norm, eps);
            rope_into(&mut kv, &self.rows(pos, Direction::Forward), self.d.rope);
            let comp_kv = matvec_on(&self.t.comp_kv, hd, self.d.hidden, &normed);
            let at = (pos % self.d.ratio) * hd;
            let score: Vec<f64> = matvec_on(&self.t.comp_gate, hd, self.d.hidden, &normed)
                .into_iter()
                .enumerate()
                .map(|(i, v)| v + f64::from(self.t.ape[at + i]))
                .collect();
            Latent {
                normed,
                kv,
                comp_kv,
                score,
            }
        }

        /// Take the latent path of `pos` into the caches: the ring's slot, the pending block, and the block `pos`
        /// completes.
        fn push(&mut self, pos: usize, l: &Latent) {
            let (hd, r) = (self.d.head, self.d.ratio);
            self.ring[pos % self.d.window] = l.kv.iter().map(|&v| f16(v)).collect();
            self.pend.push((l.comp_kv.clone(), l.score.clone()));
            if !(pos + 1).is_multiple_of(r) {
                return;
            }
            // Per channel, the softmax over the block's scores, the weighted sum of the projections.
            let pooled: Vec<f64> = par_map(hd, |c| {
                let mx = self
                    .pend
                    .iter()
                    .map(|(_, s)| s[c])
                    .fold(f64::NEG_INFINITY, f64::max);
                let (mut sum, mut res) = (0.0f64, 0.0f64);
                for (kv, sc) in &self.pend {
                    let w = (sc[c] - mx).exp();
                    sum += w;
                    res += w * kv[c];
                }
                res / sum
            });
            let mut row = rms(&pooled, &self.t.comp_norm, f64::from(self.d.eps));
            rope_into(
                &mut row,
                &self.rows(pos + 1 - r, Direction::Forward),
                self.d.rope,
            );
            self.comp.push(row.iter().map(|&v| f16(v)).collect());
            self.pend.clear();
        }

        /// The sub-layer's output row at `pos`, [`Rule::push`] of its latent path taken just before.
        fn step(&self, pos: usize, l: &Latent) -> Vec<f64> {
            let d = &self.d;
            let (hd, heads, eps) = (d.head, d.heads, f64::from(d.eps));
            let q_a = matvec(&self.t.q_a, d.q_lora, d.hidden, &l.normed);
            let qr = rms(&q_a, &self.t.q_a_norm, eps);
            let qraw = matvec(&self.t.q_b, d.q_width(), d.q_lora, &qr);
            let gate = matvec(&self.t.gate, d.grouped(), d.q_lora, &qr);
            let (fwd, back) = (
                self.rows(pos, Direction::Forward),
                self.rows(pos, Direction::Back),
            );
            let ones = vec![1.0f32; hd];
            // Every head's attention row: the weightless norm, the tail rope, the f16 query against the
            // f16 window rows and compressed rows, the head's sink in the denominator.
            let attn: Vec<Vec<f64>> = par_map(heads, |h| {
                let mut head = rms(&qraw[h * hd..(h + 1) * hd], &ones, eps);
                rope_into(&mut head, &fwd, d.rope);
                let q: Vec<f64> = head.iter().map(|&v| f16(v)).collect();
                let lo = pos.saturating_sub(d.window - 1);
                let keys: Vec<&[f64]> = (lo..=pos)
                    .map(|p| self.ring[p % d.window].as_slice())
                    .chain(self.comp.iter().map(Vec::as_slice))
                    .collect();
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|k| f64::from(d.scale) * (0..hd).map(|c| q[c] * k[c]).sum::<f64>())
                    .collect();
                let sink = f64::from(self.t.sinks[h]);
                let mx = logits.iter().copied().fold(sink, f64::max);
                let p: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
                let den = p.iter().sum::<f64>() + (sink - mx).exp();
                (0..hd)
                    .map(|c| p.iter().zip(&keys).map(|(&pi, k)| pi * k[c]).sum::<f64>() / den)
                    .collect()
            });
            // The head mix of every head's row, then the inverse rope of its tail.
            let mixed: Vec<Vec<f64>> = par_map(heads, |h| {
                let mut o: Vec<f64> = (0..hd)
                    .map(|c| {
                        (0..heads)
                            .map(|g| self.mix[h * heads + g] * attn[g][c])
                            .sum::<f64>()
                    })
                    .collect();
                rope_into(&mut o, &back, d.rope);
                o
            });
            let mut o = vec![0.0f64; d.q_width()];
            for (h, row) in mixed.into_iter().enumerate() {
                o[h * hd..(h + 1) * hd].copy_from_slice(&row);
            }
            // The grouped output projection: group `g`'s rows read `g`'s heads.
            let (group_k, o_lora) = (d.group_k(), d.o_lora);
            let mut oa = Vec::with_capacity(d.grouped());
            for g in 0..d.o_groups {
                oa.extend(matvec(
                    &self.t.output_a[g * o_lora * group_k..],
                    o_lora,
                    group_k,
                    &o[g * group_k..(g + 1) * group_k],
                ));
            }
            for (v, g) in oa.iter_mut().zip(&gate) {
                *v /= 1.0 + (-g).exp();
            }
            matvec(&self.t.output_b, d.hidden, d.grouped(), &oa)
        }
    }

    /// The input rows of positions `0..n`, seeded in `[-1, 1)`.
    fn inputs(n: usize, rng: &mut GateRng) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..HIDDEN).map(|_| rng.uni(-1.0, 1.0)).collect())
            .collect()
    }

    /// The step form of `layer` over positions `from ..= to` on `st`, each row read back.
    fn run_steps(
        dev: &Dev,
        w: &Weights,
        layer: &HcaLayer,
        st: &mut HcaState,
        xs: &[Vec<f32>],
        from: usize,
        to: usize,
    ) -> Result<Vec<Vec<f32>>, GateError> {
        let (stream, hidden) = (dev.stream(), st.dims().hidden);
        let mut out = DeviceBuffer::zeroed(stream, hidden)?;
        let mut rows = Vec::with_capacity(to + 1 - from);
        for (pos, x) in xs.iter().enumerate().take(to + 1).skip(from) {
            let x = dev.up(x)?;
            st.set_run(stream, pos as u32, 1)?;
            layer.enqueue_step(
                &dev.gpu,
                &dev.k,
                w,
                st,
                stream,
                pos as u32,
                HcaIn::Fold(&x),
                &mut out,
            )?;
            rows.push(dev.read(&out)?);
        }
        Ok(rows)
    }

    /// What the step clause leaves its caller: its verdicts, the rows it read and the state it ran.
    struct Stepped {
        results: Vec<bool>,
        rows: Vec<Vec<f32>>,
        state: HcaState,
    }

    /// 1. The step at the eight positions, each after every position before it; the rows it read and the state it left.
    fn step_clauses(
        dev: &Dev,
        w: &Weights,
        d: &HcaDims,
        layer: &HcaLayer,
        vha: &Vha,
        t: &Tensors,
        xs: &[Vec<f32>],
    ) -> Result<Stepped, GateError> {
        let last = STEPS.iter().copied().max().expect("a step position");
        let mut rule = Rule::new(d, t.clone(), vha)?;
        let x64: Vec<Vec<f64>> = xs[..=last]
            .iter()
            .map(|r| r.iter().map(|&v| f64::from(v)).collect())
            .collect();
        let latents: Vec<Latent> = par_map(x64.len(), |pos| rule.latent(pos, &x64[pos]));
        let mut st = HcaState::new(dev.stream(), d, 1, |_| true)?;
        let rows = run_steps(dev, w, layer, &mut st, xs, 0, last)?;
        let mut results = Vec::new();
        let mut kept = Vec::new();
        for pos in 0..=last {
            rule.push(pos, &latents[pos]);
            if STEPS.contains(&pos) {
                results.push(banded(
                    &format!("step pos {pos}"),
                    BAND,
                    &rows[pos],
                    &rule.step(pos, &latents[pos]),
                ));
                kept.push(rows[pos].clone());
            }
        }
        results.push(dev.no_fault("step")?);
        Ok(Stepped {
            results,
            rows: kept,
            state: st,
        })
    }

    /// The prompt form's rows are the step form's at each position: from 0 over 1001 tokens, which holds the
    /// eight positions, and on that state from 1001 — a start no piece aligns with — over 300 more, each row.
    fn prompt_clauses(
        dev: &Dev,
        w: &Weights,
        d: &HcaDims,
        layer: &HcaLayer,
        xs: &[Vec<f32>],
        steps: &[Vec<f32>],
        step_state: &mut HcaState,
    ) -> Result<Vec<bool>, GateError> {
        let (stream, hidden, piece) = (dev.stream(), d.hidden, d.piece);
        let mut results = Vec::new();
        let mut st = HcaState::new(stream, d, 1, |_| true)?;
        let first = STEPS.iter().copied().max().expect("a step position") + 1;
        let second = LAST + 1 - first;
        let flat = |rows: &[Vec<f32>]| rows.iter().flatten().copied().collect::<Vec<f32>>();
        let x = dev.up(&flat(&xs[..first]))?;
        let mut out = DeviceBuffer::zeroed(stream, first * hidden)?;
        layer.enqueue_prompt(
            &dev.gpu,
            &dev.k,
            w,
            &mut st,
            stream,
            0,
            first,
            HcaIn::Fold(&x),
            &mut out,
        )?;
        let got = dev.read(&out)?;
        for (row, &pos) in steps.iter().zip(&STEPS) {
            let mine = &got[pos * hidden..(pos + 1) * hidden];
            results.push(clause(
                &format!("prompt pos {pos} = step"),
                bits_equal(mine, row),
                &diff_text(mine, row),
            ));
        }
        // The same state takes a prompt from 1001; the step form runs the same positions on the step clause's state.
        let x2 = dev.up(&flat(&xs[first..]))?;
        let mut out2 = DeviceBuffer::zeroed(stream, second * hidden)?;
        layer.enqueue_prompt(
            &dev.gpu,
            &dev.k,
            w,
            &mut st,
            stream,
            first as u32,
            second,
            HcaIn::Fold(&x2),
            &mut out2,
        )?;
        let got2 = dev.read(&out2)?;
        let want = run_steps(dev, w, layer, step_state, xs, first, LAST)?;
        let differing = got2
            .chunks(hidden)
            .zip(&want)
            .filter(|(g, w)| !bits_equal(g, w))
            .count();
        results.push(clause(
            &format!("prompt from {first} over {second} tokens = step, each row"),
            differing == 0,
            &format!("{differing} of {} rows differ in bits (an unaligned head of {} steps, then pieces)", want.len(), piece - first % piece),
        ));
        results.push(dev.no_fault("prompt")?);
        Ok(results)
    }

    /// Clause 7: the step captured on a fresh state whose words hold no run, then replayed at every position
    /// `0 ..= 1000` with the position's row in the captured input and its words set first: each of the eight rows the
    /// step clause kept, bit for bit. Mutant: a launch whose source offset is the capture's position (the position
    /// table's row), or a capture refused over the words it finds.
    fn replay_clauses(
        dev: &Dev,
        w: &Weights,
        d: &HcaDims,
        layer: &HcaLayer,
        xs: &[Vec<f32>],
        steps: &[Vec<f32>],
    ) -> Result<Vec<bool>, GateError> {
        let (stream, hidden) = (dev.stream(), d.hidden);
        let last = STEPS.iter().copied().max().expect("a step position");
        let mut st = HcaState::new(stream, d, 1, |_| true)?;
        let mut x_cap = DeviceBuffer::<f32>::zeroed(stream, hidden)?;
        let mut o_cap = DeviceBuffer::<f32>::zeroed(stream, hidden)?;
        let graph = dev.gpu.capture(|s| {
            layer.enqueue_step(
                &dev.gpu,
                &dev.k,
                w,
                &mut st,
                s,
                0,
                HcaIn::Fold(&x_cap),
                &mut o_cap,
            )
        })?;
        let mut results = Vec::new();
        let mut kept = steps.iter();
        for (pos, x) in xs.iter().enumerate().take(last + 1) {
            x_cap.copy_from_host(stream, x)?;
            st.set_run(stream, pos as u32, 1)?;
            graph.launch(stream)?;
            if STEPS.contains(&pos) {
                let got = dev.read(&o_cap)?;
                let want = kept.next().expect("a kept row each step position");
                results.push(clause(
                    &format!("replay pos {pos} = step"),
                    bits_equal(&got, want),
                    &diff_text(&got, want),
                ));
            }
        }
        results.push(dev.no_fault("replay")?);
        Ok(results)
    }

    /// How many values of two rows differ in bits.
    fn diff_text(got: &[f32], want: &[f32]) -> String {
        format!(
            "{} of {} values differ in bits",
            got.iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            want.len()
        )
    }

    /// 3. The head mix: the identity bit for bit, the seeded mix within band.
    fn vha_clauses(dev: &Dev, vha: &Vha, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let stream = dev.stream();
        let tokens = 3;
        let o: Vec<f32> = (0..tokens * HEADS * HEAD)
            .map(|_| rng.uni(-1.0, 1.0))
            .collect();
        let zero = vec![0.0f32; HEADS * VHA_RANK];
        let (eye, seeded) = (
            fold(&zero, &zero, HEADS, VHA_RANK, "gate")?,
            fold(&vha.u, &vha.v, HEADS, VHA_RANK, "gate")?,
        );
        let want = mix_host(&o, &seeded, HEADS, HEAD, tokens);
        let mut results = Vec::new();
        for (name, mt) in [("vha identity", eye), ("vha seeded", seeded)] {
            let (mt, mut y) = (dev.up(&mt)?, dev.up(&o)?);
            dev.k.vha.enqueue_mix(
                stream,
                &mt,
                &mut y,
                HEADS,
                HEAD,
                tokens,
                dev.gpu.unlabelled_sink(),
            )?;
            let got = dev.read(&y)?;
            results.push(if name.ends_with("identity") {
                clause(name, bits_equal(&got, &o), &diff_text(&got, &o))
            } else {
                banded(
                    name,
                    BAND_VHA,
                    &got,
                    &want.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                )
            });
        }
        results.push(dev.no_fault("vha")?);
        Ok(results)
    }

    /// 4. The output gate at 64 heads of 128 over 3 tokens: the host rule's bits.
    fn gate_clause(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let stream = dev.stream();
        let (tokens, n_head) = (3usize, GROUPED / GATE_HEAD);
        let attn: Vec<f32> = (0..tokens * GROUPED).map(|_| rng.uni(-2.0, 2.0)).collect();
        let gate: Vec<f32> = (0..tokens * GROUPED).map(|_| rng.uni(-6.0, 6.0)).collect();
        let (a, g) = (dev.up(&attn)?, dev.up(&gate)?);
        let mut y = DeviceBuffer::zeroed(stream, attn.len())?;
        dev.k.out_gate.enqueue_out_gate(
            stream,
            OutGateArgs {
                kind: OutGate::Proj,
                attn: &a,
                gate: &g,
                n_head,
                m: tokens,
                fault: dev.gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        let (got, want) = (dev.read(&y)?, out_gate_host(&attn, &gate));
        Ok(vec![
            clause("gate", bits_equal(&got, &want), &diff_text(&got, &want)),
            dev.no_fault("gate")?,
        ])
    }

    /// A shape [`HcaDims::new`] refuses: what it breaks in the keys, and the text it is refused by.
    type BadShape = (&'static str, fn(&mut HcaKeys), &'static str);

    /// The step over the trunk's normed fold ([`HcaIn::Normed`]) is the step over the fold
    /// ([`HcaIn::Fold`]) bit for bit at every position it runs: the norm the layer launches on a
    /// fold is the one the trunk's pre launches (`rms_norm`, the layer's `attn_norm` gain, its
    /// epsilon). Mutant: the normed form normed again, or a row of it skipped.
    fn normed_input_clauses(
        dev: &Dev,
        w: &Weights,
        d: &HcaDims,
        layer: &HcaLayer,
        xs: &[Vec<f32>],
    ) -> Result<Vec<bool>, GateError> {
        let (stream, hidden) = (dev.stream(), d.hidden);
        let mut fold_st = HcaState::new(stream, d, 1, |_| true)?;
        let mut norm_st = HcaState::new(stream, d, 1, |_| true)?;
        let mut out = DeviceBuffer::zeroed(stream, hidden)?;
        let mut normed = DeviceBuffer::zeroed(stream, hidden)?;
        let gain = w.f32_buf("gate_ernie55_hca", &layer.norm)?;
        let (mut folded, mut pre_normed) = (Vec::new(), Vec::new());
        for (pos, x) in xs.iter().enumerate().take(NORMED_STEPS) {
            let (x, pos) = (dev.up(x)?, pos as u32);
            fold_st.set_run(stream, pos, 1)?;
            layer.enqueue_step(
                &dev.gpu,
                &dev.k,
                w,
                &mut fold_st,
                stream,
                pos,
                HcaIn::Fold(&x),
                &mut out,
            )?;
            folded.extend(dev.read(&out)?);
            dev.gpu
                .elem()
                .enqueue_rms_norm(stream, &x, gain, d.eps, hidden, 1, &mut normed)?;
            norm_st.set_run(stream, pos, 1)?;
            layer.enqueue_step(
                &dev.gpu,
                &dev.k,
                w,
                &mut norm_st,
                stream,
                pos,
                HcaIn::Normed(&normed),
                &mut out,
            )?;
            pre_normed.extend(dev.read(&out)?);
        }
        Ok(vec![
            clause(
                &format!("normed input = fold over {NORMED_STEPS} steps"),
                bits_equal(&pre_normed, &folded),
                &diff_text(&pre_normed, &folded),
            ),
            dev.no_fault("normed input")?,
        ])
    }

    /// One refusal: the call refused with `text` in its error.
    fn refusal(name: &str, r: Result<(), GpuError>, text: &str) -> bool {
        let got = r
            .as_ref()
            .err()
            .map_or_else(|| "accepted".to_string(), ToString::to_string);
        clause(name, r.is_err() && got.contains(text), &got)
    }

    /// 5. The refusals: each refused by name before anything is enqueued.
    fn refusal_clauses(
        dev: &Dev,
        w: &Weights,
        d: &HcaDims,
        layer: &HcaLayer,
        vha: &Vha,
        xs: &[Vec<f32>],
    ) -> Result<Vec<bool>, GateError> {
        let (stream, hidden, fault) = (dev.stream(), d.hidden, dev.gpu.unlabelled_sink());
        let (mut r, piece) = (Vec::new(), d.piece);
        let (x, mut out) = (dev.up(&xs[0])?, DeviceBuffer::zeroed(stream, hidden)?);
        let mut fresh = HcaState::new(stream, d, 1, |_| true)?;
        let step =
            |st: &mut HcaState, pos: u32, x: &DeviceBuffer<f32>, out: &mut DeviceBuffer<f32>| {
                layer.enqueue_step(&dev.gpu, &dev.k, w, st, stream, pos, HcaIn::Fold(x), out)
            };
        let past = format!("ending at or before the context's {}", d.ctx);
        r.push(refusal(
            "refuse step past the context",
            fresh
                .set_run(stream, d.ctx as u32, 1)
                .and_then(|()| step(&mut fresh, d.ctx as u32, &x, &mut out)),
            &past,
        ));
        r.push(refusal(
            "refuse step without words",
            step(&mut fresh, 5, &x, &mut out),
            "over words that hold None",
        ));
        fresh.set_run(stream, 3, 1)?;
        r.push(refusal(
            "refuse step over another run's words",
            step(&mut fresh, 4, &x, &mut out),
            "over words that hold Some((3, 1))",
        ));
        fresh.set_run(stream, 0, 1)?;
        let short = DeviceBuffer::zeroed(stream, 10)?;
        r.push(refusal(
            "refuse short row",
            step(&mut fresh, 0, &short, &mut out),
            "a run of 1 tokens over x of 10",
        ));
        let prompt = |st: &mut HcaState,
                      pos0: u32,
                      tokens: usize,
                      x: &DeviceBuffer<f32>,
                      out: &mut DeviceBuffer<f32>| {
            layer.enqueue_prompt(
                &dev.gpu,
                &dev.k,
                w,
                st,
                stream,
                pos0,
                tokens,
                HcaIn::Fold(x),
                out,
            )
        };
        r.push(refusal(
            "refuse prompt of no token",
            prompt(&mut fresh, 0, 0, &x, &mut out),
            "a prompt of 0 tokens",
        ));
        let (wide, mut wide_out) = (
            DeviceBuffer::zeroed(stream, 300 * hidden)?,
            DeviceBuffer::zeroed(stream, 300 * hidden)?,
        );
        r.push(refusal(
            "refuse prompt past the context",
            prompt(&mut fresh, (d.ctx - 100) as u32, 300, &wide, &mut wide_out),
            &past,
        ));
        r.push(refusal(
            "refuse run over a piece",
            fresh.set_run(stream, 0, piece + 1),
            &format!("1 ..= {piece} tokens"),
        ));
        // The head mix's own.
        let (mt, mt48, ten) = (
            dev.up(&vec![0.0f32; HEADS * HEADS])?,
            dev.up(&vec![0.0f32; 48 * 48])?,
            dev.up(&[0.0f32; 10])?,
        );
        let (mut y, mut y48) = (
            dev.up(&vec![0.5f32; HEADS * HEAD])?,
            dev.up(&vec![0.5f32; 48 * HEAD])?,
        );
        let mix = |mt: &DeviceBuffer<f32>,
                   y: &mut DeviceBuffer<f32>,
                   heads: usize,
                   width: usize,
                   tokens: usize| {
            dev.k
                .vha
                .enqueue_mix(stream, mt, y, heads, width, tokens, fault)
        };
        r.push(refusal(
            "refuse head mix at 48 heads",
            mix(&mt48, &mut y48, 48, HEAD, 1),
            "heads must be a positive multiple of 32",
        ));
        r.push(refusal(
            "refuse head mix at width 510",
            mix(&mt, &mut y, HEADS, 510, 1),
            "width must be a positive multiple of 4",
        ));
        r.push(refusal(
            "refuse head mix with a short matrix",
            mix(&ten, &mut y, HEADS, HEAD, 1),
            "mt.len()",
        ));
        r.push(refusal(
            "refuse head mix of no token",
            mix(&mt, &mut y, HEADS, HEAD, 0),
            "need tokens >= 1",
        ));
        // The shapes the kernels do not take.
        let bad: [BadShape; 7] = [
            (
                "head 256",
                |k| k.hca.head = 256,
                "the latent append are 512 wide",
            ),
            (
                "40 heads",
                |k| k.heads = 40,
                "an attention block is 16 heads",
            ),
            (
                "48 heads",
                |k| k.heads = 48,
                "the head mix takes a multiple of 32",
            ),
            ("rope tail 6", |k| k.rope = 6, "a multiple of 4"),
            (
                "ratio 1",
                |k| k.hca.ratio = 1,
                "the compressor pools at 2 or above",
            ),
            (
                "o_lora 100",
                |k| k.hca.o_lora = 100,
                "the grouped output projection's output is 800",
            ),
            (
                "hidden 4100",
                |k| k.hidden = 4100,
                "the 128-value block divides",
            ),
        ];
        for (what, set, text) in bad {
            let mut k = keys();
            set(&mut k);
            r.push(refusal(
                &format!("refuse dims {what}"),
                HcaDims::new(k, CTX, piece).map(|_| ()),
                text,
            ));
        }
        r.push(refusal(
            "refuse dims piece 7",
            HcaDims::new(keys(), CTX, 7).map(|_| ()),
            "a positive multiple of 8 tokens",
        ));
        r.push(refusal(
            "refuse head mix fold of another length",
            fold(&[0.0f32; 10], &vha.v, HEADS, VHA_RANK, "gate").map(|_| ()),
            &format!("takes {} of each", HEADS * VHA_RANK),
        ));
        // A layer whose weights are not resident: the first tensor, by name.
        let missing = HcaLayer::load(stream, &Weights::default(), d, 0, vha).map(|_| ());
        let got = missing
            .as_ref()
            .err()
            .map_or_else(|| "accepted".to_string(), ToString::to_string);
        r.push(clause(
            "refuse layer without its weights",
            missing.is_err() && got.contains(&names::attn_norm(0)),
            &got,
        ));
        r.push(dev.no_fault("refusals")?);
        Ok(r)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = HcaKernels::load(gpu.context())?;
        let d = HcaDims::new(keys(), CTX, HcaDims::PIECE)?;
        println!(
            "gate_ernie55_hca: device {} — hidden {HIDDEN}, {HEADS} heads of {HEAD}, window {WINDOW}, ratio {RATIO}, \
             groups {O_GROUPS}x{O_LORA}, query latent {Q_LORA}, rope {ROPE} (YaRN), context {CTX}, piece {}, synthetic",
            gpu.device_name()?,
            HcaDims::PIECE
        );
        let dev = Dev { gpu, k };
        let (stream, mut rng) = (dev.stream(), GateRng(0x4552_4e49_4535_6863));
        let mut w = Weights::default();
        let t = seed_weights(stream, &mut w, &mut rng)?;
        let vha = seed_vha(&mut rng);
        let layer = HcaLayer::load(stream, &w, &d, 0, &vha)?;
        let xs = inputs(LAST + 1, &mut rng);
        let mut results = Vec::new();
        let mut stepped = step_clauses(&dev, &w, &d, &layer, &vha, &t, &xs)?;
        results.extend(stepped.results);
        results.extend(prompt_clauses(
            &dev,
            &w,
            &d,
            &layer,
            &xs,
            &stepped.rows,
            &mut stepped.state,
        )?);
        results.extend(replay_clauses(&dev, &w, &d, &layer, &xs, &stepped.rows)?);
        results.extend(vha_clauses(&dev, &vha, &mut rng)?);
        results.extend(normed_input_clauses(&dev, &w, &d, &layer, &xs)?);
        results.extend(gate_clause(&dev, &mut rng)?);
        results.extend(refusal_clauses(&dev, &w, &d, &layer, &vha, &xs)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_ernie55_hca: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
